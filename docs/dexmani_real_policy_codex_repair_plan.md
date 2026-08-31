# DexMani Real / DexMani Policy 策略部署简化与可靠性修复方案

> **用途**：交给 Codex 按阶段实施。  
> **目标**：在不削弱真实机器人安全边界的前提下，使 `dexmani_policy → dexmani_real → robot` 的策略导出、加载、推理、调度、验证和操作入口更简洁、准确、可复现。  
> **生成日期**：2026-08-31  
> **审查基线**：
>
> - `dexmani_real/main`: `f76384789fce5c2f35dd55dcfdc6f7dc276098a6`
> - `dexmani_policy/main`: `a9de8b9b8c082edc7192b5a5bf7ffaf91a7f252a`
> - 对照参考：
>   - `Universal-Control/ManiUniCon`
>   - `real-stanford/diffusion_policy/diffusion_policy/real_world`
>
> **事实优先级**：实际源码与测试 > schema/config 定义 > README/专题文档 > AGENTS/CLAUDE 中的快照描述。

---

## 0. 本轮 `dexmani_policy` 更新后的结论

`dexmani_policy` 已更新到 `a9de8b9...`。最新提交主要调整 ActionFlow、DQ-RISE 配置和文档，但没有改变本方案的核心方向：

- `BaseAgent` 仍明确输出 `pred_action`、`control_action` 和可选 `tail`；
- DQ-RISE 仍输出标准 `control_action`，但不返回 `tail`；
- 当前仓库仍没有正式 `dexmani.deployment.v2` exporter；
- `build_train_params()` 仍缺少 `use_aux_ee`；
- ActionFlow 仍配置 `state_dim: ${action_dim}`，`action_ee` 时会与固定 19-D `joint_state` 冲突；
- DQ-RISE 当前默认 `action_key=action_ee`，部署时必须按 EE action 进入 Real 的 IK 路径；
- `training/eval_utils.py` 的 `raw_state` 仍只在 `train_params is not None` 分支内赋值；
- `pyproject.toml` 与 `requirements.txt` 的 Hydra 版本仍不一致；
- 当前 trainer 只保存 milestone/interrupt checkpoint，`monitor={}`；DQ-RISE config 中关于在线 `val_loss` checkpoint selection 的描述不符合当前训练调用链，best checkpoint 仍应由离线评测结果确定。

因此不要因 Policy 更新而跳过下面的 correctness PR。

---

## 1. 最终设计决策

### 1.1 必须采用

1. **继续使用现有 `dexmani.deployment.v2`，暂不设计 v3。**
2. **在 `dexmani_policy` 中新增正式 deployment exporter。**
3. **`dexmani_real` 只执行 `agent.predict_action(...)["control_action"]`。**
4. 完整 `pred_action` 只做 full-output shape、finite 和 consistency 检查，不进入 scheduler。
5. **保留 Real 的核心因果与安全边界：**
   - source monotonic timestamp；
   - camera clock generation；
   - causal state-to-camera alignment；
   - run generation；
   - immutable action target/deadline；
   - reject-only `SafetyGate`；
   - atomic coupled-command ticket；
   - worker SDK 前最终检查。
6. 简化 Real 的主要方式：
   - 删除重复 causal reader；
   - 删除错误的 full-tail execution；
   - 修正真实 deadline readiness；
   - 收敛重复 timing/identity 字段；
   - 降低默认终端日志；
   - 简化 `run_policy` CLI；
   - 在 differential evidence 后再决定是否替换 `ActionBuffer`。
7. 第一阶段 deployment 只支持已经能形成 self-contained restore 的 **point-cloud policies**。
8. RGB 与 dynamic task-text 在 preprocessing / conditioning contract 完整前明确拒绝。
9. 研究便捷性通过 `research` / `audited` 两级 provenance strictness 提升，而不是删除 checkpoint、schema 和安全校验。

### 1.2 明确不采用

- 不立即引入 `deployment.v3`。
- 不建立通用 `BuilderRegistry / PluginManager / Service / Controller` 层级。
- 不让 Real 直接读取 `simple.v1` training checkpoint。
- 不让用户 YAML 提供任意 `_target_`。
- 不复制 Stanford Diffusion Policy / ManiUniCon 的“全部动作过期后重定时最后一个动作”。
- 不为了保证执行而整体平移 action chunk。
- 不对 learned action 做 workspace/joint clipping 后继续执行。
- 不立即删除 `ActionBuffer`。
- 不在同一 PR 中同时改 action semantics、scheduler、SafetyGate 和 hand startup。
- 不在 deployment runtime 中自动下载 pretrained model。
- 不把 `shadow` 描述为完全没有硬件副作用；当前 hand worker startup 仍可能 `reset_home()`。
- 不把 `torch.compile` 设为通用 deployment 默认。
- Codex 不得运行任何连接、homing 或控制真实硬件的命令。

---

## 2. 三套 real-world runtime 的取舍

| 机制 | Stanford Diffusion Policy | ManiUniCon | DexMani Real 决策 |
|---|---|---|---|
| Observation time | camera receive wall time | latest/history timestamp | 保留 device→host source monotonic mapping |
| 多模态对齐 | nearest past sample | wrapper 自行拼装 | 保留 state source ≤ camera source |
| 全部 action 过期 | 重定时最后 action | 重定时最后 action | 丢弃整个 prediction |
| action timestamp | 可被重新安排 | synchronized mode 可重排 | immutable logical target |
| action horizon | 显式 steps per inference | 显式 action_horizon | 使用 Policy `n_action_steps` |
| policy→robot | 直接 schedule | policy 写 action storage | proposal→coordinator→SafetyGate |
| Safety | clip / speed limit | clip / workspace | reject-only + collision |
| STOP race | 较弱 | 较弱 | generation + atomic ticket + worker fence |

应借鉴：

- ManiUniCon 的 `obs wrapper → model → action wrapper` 可读性；
- Stanford 的 timestamped chunk 与明确 action horizon；
- 两者较简洁的入口组织。

不得借鉴：

- stale-all action fallback；
- synchronized retiming to execute all actions；
- clipping learned action 后继续执行；
- policy process直接越过 Real command boundary。

---

## 3. 目标架构

```text
dexmani_policy
────────────────────────────────────────────────────────────
training config + Real Policy Zarr v5
        ↓
simple.v1 training checkpoint
        ↓
deployment exporter
        ↓
dexmani.deployment.v2 checkpoint + schema-v2 sidecar
        │
        │  model config / model or EMA weights / normalizer
        │  data contract / producer provenance
        ▼
dexmani_real
────────────────────────────────────────────────────────────
resolve selector + sidecar
        ↓
no-follow / identity / SHA-256 / provenance verification
        ↓
safe deployment-v2 decode
        ↓
deployment-safe Hydra instantiate cfg.agent
        ↓
strict state restore + normalizer validation
        ↓
causal ObservationBatch
        ↓
agent.predict_action()
        ├── pred_action      full output，validate only
        └── control_action   only executable output
                ↓
expired-prefix removal on immutable logical grid
                ↓
policy_plan_ring
                ↓
ActionBuffer / future EndpointBuffer
                ↓
joint direct 或 EE → collision-aware IK
                ↓
SafetyGate
                ↓
shadow validation 或 atomic coupled command publication
                ↓
arm/hand worker final permit + bounds check
                ↓
SDK
```

### 3.1 仓库责任边界

| 责任 | `dexmani_policy` | `dexmani_real` |
|---|---:|---:|
| Agent architecture / Hydra agent config | ✓ | 只消费 |
| Diffusion / Flow solver、NFE、EMA | ✓ | 不重新解释 |
| model-specific preprocessing | ✓ | 只提供 canonical raw input |
| training checkpoint | ✓ | 不读取 |
| deployment-v2 exporter | ✓ | 不生产 |
| artifact no-follow / hash / provenance |  | ✓ |
| causal observation assembly |  | ✓ |
| model output timestamping |  | ✓ |
| scheduling |  | ✓ |
| IK / collision / SafetyGate |  | ✓ |
| robot command / ACK |  | ✓ |
| hardware lifecycle |  | ✓ |

---

## 4. 不可削弱的因果时序与安全不变量

### 4.1 Observation causality

每个 inference frame：

```text
0 < source_monotonic_ns <= publish_monotonic_ns <= anchor_monotonic_ns
```

camera 额外满足：

```text
source <= receive <= payload_ready/publish <= anchor
```

视觉帧：

```text
desired_grid_time = run_started_ns + k * control_dt_ns
frame.source_ns <= desired_grid_time
desired_grid_time - frame.source_ns <= max_grid_lag_ns
```

state 对齐：

```text
state.source_ns <= visual.source_ns
visual.source_ns - state.source_ns <= max_observation_skew_ns
```

禁止 future state、B前 history、上一 generation history、重复旧视觉帧伪造完整 history、跨 camera generation 混用，以及用 receive time 冒充 source time。

### 4.2 Action timing

```text
target_i = observation_logical_step_ns + i * control_dt_ns
```

允许：

```text
过期 prefix → 删除
全部过期    → 丢弃整个 prediction
```

禁止 stale action retiming。

### 4.3 Run isolation

每次 `ARMED → RUNNING`：

- 递增 `run_generation`；
- 设置新的 `run_started_monotonic_ns`；
- 清空旧 command ticket；
- inference reset；
- scheduler reset；
- 旧 observation、plan、command、ACK 均失效。

### 4.4 三层 command fence

1. **Coordinator SafetyGate**：representation、finite、limits、delta、workspace、collision。
2. **Atomic publication permit**：同一 `motion_lock` 下检查 state/generation、写 ring、登记 active sequence。
3. **Worker final permit**：SDK 前检查 ticket、generation、running、fault、estop、expiry、local bounds。

三层分别覆盖不同 race window，不是可删除的重复代码。

---

## 5. 当前问题清单

### 5.1 P0

| ID | 仓库 | 问题 | 后果 |
|---|---|---|---|
| P0-01 | Real | 执行完整 `pred_action` future，而非 `control_action` | 8-step policy被改成最多15-step open-loop |
| P0-02 | Real | warmup按 raw horizon window，而非真实 source deadline | false-ready |
| P0-03 | Policy | 无正式 deployment-v2 exporter | 普通experiment不能直接部署 |
| P0-04 | Policy | ActionFlow `state_dim=${action_dim}` | EE action时21 vs joint_state 19 |
| P0-05 | Policy | eval loader `raw_state` scope错误 | metadata缺失时崩溃 |
| P0-06 | Real/Policy | RGB eval preprocessing未进入contract | train/deploy observation不一致 |
| P0-07 | Policy | pretrained constructor可读外部文件/网络 | restore非self-contained |
| P0-08 | Real | physical seed可静默为0 | stochastic rollout不可复现 |

### 5.2 P1

| ID | 仓库 | 问题 |
|---|---|---|
| P1-01 | Policy | `build_train_params()` 缺 `use_aux_ee` |
| P1-02 | Policy | Hydra依赖版本冲突，editable install依赖描述不完整 |
| P1-03 | Policy | DQ-RISE `val_loss` checkpoint描述与milestone-only trainer不一致 |
| P1-04 | Real | `ipc/causal.py` 与 deployment worker重复因果逻辑 |
| P1-05 | Real | `run_policy.py` parser混合所有模式 |
| P1-06 | Real | 1秒metrics、per-endpoint、full JSON污染终端 |
| P1-07 | Real | 业务错误使用 `parser.error()` |
| P1-08 | Real | shadow文档忽略startup hand motion |
| P1-09 | Real | 文档把15步称作actionable |
| P1-10 | Policy | AGENTS错误声称joint_state维度等于action_dim |
| P1-11 | Real | AGENTS错误声称无general tests |
| P1-12 | Real | preflight被误解为每次online run前的必经重复加载 |

### 5.3 P2

| ID | 仓库 | 候选简化 |
|---|---|---|
| P2-01 | Real | arbitrary `valid_mask` 改为prefix slicing |
| P2-02 | Real | `max_plan_age_s` 在默认下被source deadline支配 |
| P2-03 | Real | `InferenceContext.step_dt_ns` 未跨IPC使用 |
| P2-04 | Real | created与delivery target当前恒等 |
| P2-05 | Real | plan_id / observation_id / ring sequence身份重复 |
| P2-06 | Real | ActionBuffer可评估target-indexed buffer |
| P2-07 | Real | coordinator多次permit read可局部合并 |
| P2-08 | Real | hand startup可评估observe-only |
| P2-09 | Real | 固定shape encode可在profile支持后复用buffer |

---

## 6. `dexmani_policy` 修改方案

# Phase P0：当前代码正确性

## P0-A. ActionFlow state contract

文件：

- `dexmani_policy/configs/action_flow.yaml`
- `dexmani_policy/agents/core/action_flow.py`
- `AGENTS.md`
- focused config/shape test

修改：

```yaml
# before
state_dim: ${action_dim}

# after
state_dim: 19
```

原因：

```text
joint_state = arm7 + hand12 = 19
action      = 19
action_ee   = 21
```

验收：

- joint config：`action_dim=19, state_dim=19`；
- EE config：`action_dim=21, state_dim=19`；
- synthetic forward接受 `joint_state[...,19]`；
- 不修改ActionFlow solver/NFE/topology。

## P0-B. checkpoint metadata

`dexmani_policy/common/checkpoint_io.py`：

```python
params = {
    ...
    "control_action_dim": int(model.control_action_dim),
    "use_aux_ee": bool(getattr(model, "use_aux_ee", False)),
}
```

要求：

- 新checkpoint native保存；
- 旧checkpoint exporter可标记retrofit；
- metadata与config冲突时fail closed；
- 不改变simple.v1 tensor/schema。

## P0-C. eval loader

`training/eval_utils.py`：

```python
raw_state = checkpoint.model_state
if use_ema:
    if checkpoint.ema_model_state is None:
        warn(...)
    else:
        raw_state = checkpoint.ema_model_state
```

simulation eval保留现有warning fallback；deployment export中 `use_ema=true` 且EMA缺失必须hard error。

## P0-D. best checkpoint描述

当前 trainer保存milestone/interrupt checkpoint，`monitor={}`，不应把 `workspace.checkpoint_cfg.monitor_key=val_loss` 描述成在线训练自动select best。

修改 DQ-RISE config注释：

```text
online trainer保存milestone；best checkpoint由offline evaluation生成的
best_ckpt.json / best.pt确定。
```

Exporter的 `--checkpoint best` 必须使用 `resolve_best_checkpoint()` 的链：

```text
best_ckpt.json → best.pt → latest.pt → error
```

不得直接调用会对无score milestone排序的通用fallback。

## P0-E. config invariants

明确验证：

```text
joint_state_dim == 19
n_obs_steps - 1 + n_action_steps <= horizon
control_action_dim <= action_dim
use_aux_ee → action_key == action
use_aux_ee → action_dim == joint_dim + ee_dim
```

---

# Phase P1：Policy-native deployment-v2 exporter

建议最小目录：

```text
dexmani_policy/deployment/
├── __init__.py
└── export.py
```

不要先建registry/factory tree。

公开API：

```python
export_deployment_artifact(
    experiment_dir: Path,
    checkpoint_selector: str,
    output_path: Path | None = None,
    verify: bool = True,
) -> ExportReceipt
```

CLI：

```bash
python -m dexmani_policy.deployment.export EXP \
  --checkpoint best \
  --verify
```

### Export authority

| 字段 | 来源 |
|---|---|
| Agent topology | resolved `cfg.agent` |
| task/action/horizon | config + checkpoint cross-check |
| weights | selected simple.v1 checkpoint |
| normalizer | selected model/EMA state |
| EMA/NFE | resolved eval config |
| modalities | dataset config + Zarr attrs |
| dt/alignment/frame | Zarr root attrs |
| producer | current Policy commit/tree hash |
| sidecar allocation | validated deployment contract |

### 数据流

```text
resolve experiment
→ load resolved config
→ resolve selected simple.v1
→ validate train_params
→ read Zarr root attrs directly
→ validate Real Policy Zarr v5
→ construct deployment-safe cfg.agent
→ select model或EMA state
→ build deployment-v2
→ atomic checkpoint write
→ compute SHA-256
→ atomic canonical schema-v2 sidecar
→ atomic relative deployment_latest.pt symlink
→ roundtrip verify
```

### 关键要求

- training checkpoint是Policy侧trusted local input；
- output必须被Real当前 `weights_only=True` decoder直接读取；
- 不写optimizer/scheduler/dataset/workspace/env_runner；
- 不改变原training checkpoint；
- 不从文件名猜模型或task；
- sidecar逐项对照Real当前parser/golden fixture；
- 不覆盖既有artifact，除非显式 `--force`，且仍atomic。

### Zarr contract

`ReplayBuffer.copy_from_path()` 不保留root attrs，exporter必须直接读：

```python
root = zarr.open_group(str(cfg.zarr_path), mode="r")
attrs = dict(root.attrs)
```

验证：

```text
schema_name == dexmani-real-policy-zarr
schema_version == 5
domain == real
deployment_equivalent == true
task_name
dt
obs_alignment
observation_reference
state_alignment
action_semantics
sensor_modalities
point-cloud semantics
```

Sim Zarr或缺contract：hard error。

### 第一阶段策略支持

#### DP3 / ManiFlow / SAT / ActionFlow

保留fully resolved agent config。ActionFlow当前：

```text
denoise_steps=2 means NFE
solver=midpoint
midpoint requires even NFE
use_step_conditioning=false
```

Exporter不能把它改写成DDIM语义。

#### DQ-RISE

restore config：

```yaml
agent:
  codebook_path: null
```

保留并检查：

```text
tcp_dim
hand_dim
codebook_num_groups
codebook_size
action_key
normalizer/codebook consistency
```

当前默认 `action_key=action_ee`，Real按EE→IK部署。

#### R3D

restore config：

```yaml
agent:
  pc_encoder_config:
    use_pretrained_weights: false
```

只构造topology，再strict restore。

#### RGB / MoE-DP / MultiTask

第一阶段明确拒绝，不生成partial artifact。

### Export tests

- exact deployment-v2 keys；
- finite plain metadata；
- canonical state keys；
- strict restore；
- normalizer完整；
- `pred_action`/`control_action` shape；
- direct/export parity；
- no-network；
- 删除训练期codebook/pretrained文件后仍restore；
- unsupported RGB明确失败且不留下staging文件。

---

# Phase P2：依赖与文档

当前至少存在：

```text
pyproject: hydra-core >= 1.3
requirements: hydra-core == 1.2.0
```

选定一个经过train/eval/export验证的版本并统一；不要顺带升级Torch/Diffusers/Transformers。

如果 `pip install -e .` 仍不能覆盖所有策略依赖，AGENTS/README必须明确它只安装core dependencies。

可增加：

```bash
python -m dexmani_policy.deployment.export --check-environment
```

只报告缺失，不自动安装。

---

## 7. `dexmani_real` 修改方案

# Phase R0：CLI、日志与研究strictness（行为不变）

此阶段禁止修改：

```text
PolicyPrediction
ActionBuffer
target timestamp
SafetyGate
coupled publication
worker validation
hand startup
```

## R0-A. subcommands

```text
examples/run_policy.py
        ↓
dexmani_real/deployment/cli.py
```

`examples/run_policy.py` 只做thin wrapper。

```bash
python examples/run_policy.py inspect EXP
python examples/run_policy.py check EXP --device cuda:0 --seed 1066
python examples/run_policy.py shadow EXP --device cuda:0 --seed 1066
python examples/run_policy.py h4 PROFILE.yaml
python examples/run_policy.py run PROFILE.yaml
```

| 命令 | Torch/GPU | 硬件 | learned command |
|---|---:|---:|---:|
| inspect | 否 | 否 | 否 |
| check | 是 | 否 | 否 |
| shadow | 是 | 是 | 0 coupled writes |
| h4 | 是 | 是 | 最多1 endpoint |
| run | 是 | 是 | profile bounds |

无subcommand只显示help。

H4内部固定1 endpoint，不要求用户重复输入。

Physical seed必须由profile或CLI显式提供。

## R0-B. research / audited

### research

用于：

```text
inspect/check/replay/shadow
```

始终严格：

- checkpoint SHA；
- schema；
- `weights_only`；
- strict state restore；
- normalizer；
- observation/action contract；
- experiment-local package injection拒绝；
- runtime network download拒绝。

允许但记录warning/receipt：

- dirty Policy source；
- forked repository；
- commit与artifact producer不完全一致。

### audited

用于：

```text
h4/run
```

保留当前全部clean commit/tree/provenance检查。Physical命令不得通过CLI降级到research。

## R0-C. headless

支持：

```text
inspect/check
shadow --autostart --duration <finite>
```

禁止：

```text
headless h4/run
```

Physical继续要求H/B/S/ESC和现场e-stop。

## R0-D. run profile

```yaml
schema_version: 1
experiment: /path/to/experiment
runtime_config: configs/runtime.yaml
deployment_config: configs/policy_runtime.yaml
device: cuda:0
seed: 1066
strictness: audited
execution:
  mode: task
  max_running_s: 30.0
  max_published_endpoints: 331
  acknowledgement_timeout_s: 0.75
approval:
  checkpoint_sha256: "..."
  policy_commit: "..."
  real_commit: "..."
```

禁止覆盖artifact-owned fields。

## R0-E. typed errors

只对syntax使用 `parser.error()`。业务异常使用：

```text
ArtifactError
PolicyCompatibilityError
PolicyLoadError
HardwareStartupError
RuntimeFault
```

终端只打印精确错误与一个hint。

## R0-F. logging

```text
logger level = DEBUG
file handler = DEBUG
console handler = INFO
```

DEBUG：1秒metrics、per-endpoint、full config、full receipt、重复worker lifecycle。

INFO：policy summary、aggregated ready、ARMED/RUNNING/STOPPED、compact status、receipt path、final result。

日志名加入PID，避免spawn process同秒碰撞。

## R0-G. single-load语义

`check` 是offline diagnostic，不是每次online run前的必经步骤。

正常 `shadow/h4/run` 由inference child完成authoritative verified load；硬件worker仍只在model ready后启动。不要自动先跑一个独立preflight再重新加载。

---

# Phase R1：执行 `control_action`

重点文件：

- `integrations/dexmani_policy.py`
- `deployment/manifest.py`
- `deployment/contracts.py`
- preflight/adapter tests
- `docs/policy_deployment.md`

```python
with torch.inference_mode():
    result = agent.predict_action(obs_dict, denoise_timesteps=...)

pred = require_tensor(result, "pred_action")
control = require_tensor(result, "control_action")
```

验证：

```text
pred.shape == [1,horizon,model_action_dim]
control.shape == [1,n_action_steps,control_action_dim]
both finite
```

preflight/warmup额外验证：

```python
expected = pred[
    :,
    n_obs_steps - 1:n_obs_steps - 1 + n_action_steps,
    :control_action_dim,
]
assert_close(control, expected, rtol=0, atol=0)
```

hot path不必每帧exact compare。

只decode `control_action`。

注意：

- 不要求 `tail` key；
- DQ-RISE不返回tail是合法的；
- R3D aux：full 28-D验证，19-D执行；
- DQ-RISE EE：21-D拆成EE pose + hand，coordinator IK。

文档改为：

```text
prediction future steps = 15
executable control steps = 8
```

`required_action_steps` 仅是legacy serialized prediction length，不是execution length。

---

# Phase R2：真实deadline readiness

新增 `deployment/timing.py` pure helper：

```python
def usable_target_mask(
    *,
    logical_step_ns,
    steps,
    step_dt_ns,
    inference_finished_ns,
    observation_source_ns,
    command_lead_ns,
    max_plan_age_ns,
    max_source_to_command_age_ns,
):
    earliest = inference_finished_ns + command_lead_ns
    deadline = min(
        inference_finished_ns + max_plan_age_ns,
        observation_source_ns + max_source_to_command_age_ns,
    )
    targets = logical_step_ns + np.arange(steps) * step_dt_ns
    return (targets > earliest) & (targets < deadline)
```

同一helper用于warmup/stamping/deadline tests。

保留5次warmup、最后3次stable；每个stable sample至少2个usable endpoint，H4可要求至少1个。

详细100次benchmark放在 `check --benchmark-samples`，不增加online启动时间。

worker层统一 `torch.inference_mode()`。

---

# Phase R3：causal reader去重

扩展 `ipc/causal.py`：

```python
read_causal_structured_history(...)
align_history_to_reference_sources(...)
```

统一：

- source/publish/anchor；
- run-start not-before；
- max age；
- health flags；
- state source≤visual source；
- max skew。

`deployment/worker.py` 只保留：

- pointcloud/RGB payload validity；
- camera generation；
- visual grid；
- modality assembly；
- metrics。

删除/委托 `_read_state_history()` 和重复alignment。

合并 `_select_camera_control_grid` / `_select_pointcloud_control_grid` 为 `_select_visual_control_grid`。

必须做旧/新实现differential test，逐字段相同。

---

# Phase R4：wire/scheduler第二轮简化

只在R1/R2/R3新baseline后实施。

## R4-A. valid_mask→prefix slice

严格递增target的过期mask只能是prefix：

```text
00001111
```

```python
first_valid = np.searchsorted(
    targets,
    inference_finished_ns + command_lead_ns,
    side="right",
)
if first_valid == len(targets):
    return None
actions = actions[first_valid:]
targets = targets[first_valid:]
```

删除 `JointActionChunk.valid_mask`、IPC field和ActionBuffer mask branches。禁止重定时。

## R4-B. deadline收敛

默认下 `source+0.75s` 始终早于 `finish+1.0s`。先做differential evidence，再评估删除 `max_plan_age_s`，收敛为：

```text
max_observation_to_command_age_s
deadline = observation_source_ns + max_age
```

## R4-C. 字段删除候选

按独立PR依次评估：

1. `InferenceContext.step_dt_ns`；
2. `ActionCandidate.target_monotonic_ns`；
3. explicit `plan_id`。

每次同步dataclass、IPC、writer、reader、receipt、tests。

## R4-D. EndpointBuffer实验

不要直接替换ActionBuffer。先实现target-indexed buffer并对recorded plan stream双跑。

比较：selected target、observation identity、action tensor、commit/discard、stale behavior、command silence。

只有完全等价后才能offline→shadow→H4切换；否则保留ActionBuffer。

---

# Phase R5：安全局部去重

必须保留：raw gate、shaped hand第二次gate、atomic permit、worker final permit、worker bounds、ACK、generation、H4/task bounds。

可将coordinator内部多次permit read收敛为：

```text
early cheap gate
→ feedback + SafetyGate
→ final atomic state/generation/expiry/ring write
→ worker final gate
```

不把hardware IO放入lock。

hand mechanical preflight即使部分数学重复，成本低且是清晰defense-in-depth；当前保留并注明intentional。

---

# Phase R6：Hand observe-only（独立硬件PR）

当前shadow只保证learned coupled writes=0，不保证zero hardware side effect。

顺序：fake driver→XHand read-only诊断→反馈稳定性→live shadow→再引入 `OBSERVE_ONLY`。

不得与exporter/control_action/scheduler合并。

---

# Phase R7：性能优化（测量后）

先使用 `torch.inference_mode()` 与真实latency report。

只有profiler证明allocation/copy是显著瓶颈时，才在Policy runtime内部复用固定shape input buffer。不要预先引入复杂pinned-memory staging。

`torch.compile` 保持artifact/profile级opt-in；ActionFlow、R3D、point-cloud CUDA op分别benchmark，不能全局默认开启。

---

## 8. RGB deployment后续方案

RGB第一阶段不支持。

当前training/eval：

```text
raw → /255 → dataset resize → validation center crop → model ImageProcessor
```

当前Real：

```text
raw → direct model ImageProcessor resize
```

支持前artifact必须表达：

```yaml
rgb_preprocess:
  input_shape: [H,W,3]
  value_range: uint8_0_255
  color_order: rgb
  resize: [240,240]
  eval_crop: [224,224]
  crop_mode: center
  interpolation: bilinear
```

并实现无网络的DINO/R3M topology construction、replay parity和dedicated shadow。

MultiTask还需显式static task-text contract；dynamic text暂不支持。

---

## 9. `run_policy` 最终体验

### Inspect

```bash
python examples/run_policy.py inspect EXP
```

显示：policy/task/checkpoint/SHA/producer、Observation shape、action representation、15 prediction steps、8 execution steps、control/inference rate和compatibility。

`--json` 时stdout只输出JSON。

### Check

```bash
python examples/run_policy.py check EXP \
  --device cuda:0 --seed 1066 --benchmark-samples 20
```

显示strict restore、normalizer、pred/control shape、warmup latency、usable steps、GPU peak memory。

### Shadow

```bash
python examples/run_policy.py shadow EXP \
  --device cuda:0 --seed 1066 --duration 30
```

明确打印：learned coupled writes disabled、hardware workers connected、hand startup behavior。

### H4/Run

```bash
python examples/run_policy.py h4 profile.yaml
python examples/run_policy.py run profile.yaml
```

启动前显示checkpoint/Policy/Real identity、device、seed和bounds。

---

## 10. 测试矩阵

### Policy

| 测试 | 断言 |
|---|---|
| ActionFlow joint/EE config | state_dim均为19 |
| train_params | 含control_action_dim/use_aux_ee |
| eval loader no metadata | 无UnboundLocalError |
| best selector | best_ckpt→best.pt→latest |
| strict EMA export | EMA缺失时拒绝 |
| exporter schema | exact v2 |
| sidecar | exact schema-v2 |
| roundtrip | strict restore |
| parity | direct/export control_action一致 |
| DQ-RISE | 无external codebook file仍restore |
| R3D | 无pretrained file仍restore |
| no-network | HF/gdown/socket被禁仍restore |
| ActionFlow | midpoint/NFE=2保持 |
| RGB | 明确拒绝且无partial artifact |

### Real

| 测试 | 断言 |
|---|---|
| inspect isolation | 不导入torch/hardware |
| no subcommand | 不启动worker |
| physical seed | 缺失拒绝 |
| research/audited | physical不能降级 |
| control_action sentinel | tail永不入plan |
| R3D aux | 28-D validate/19-D execute |
| DQ-RISE EE | 21-D正确拆分 |
| warmup deadline | false-ready case拒绝 |
| causal parity | 新旧reader输出一致 |
| generation reset | 旧数据全部失效 |
| all expired | 不重定时、不发布 |
| shadow | coupled sequence不变 |
| atomic ticket | STOP race不能越过SDK |
| worker expiry | expired command不执行 |
| logging snapshot | 无默认spam |
| ActionBuffer differential | legacy/new逐endpoint一致 |

### Cross-repo

```text
Policy export fixture
→ install exact Policy commit
→ Real inspect
→ Real isolated check
→ recorded replay
```

支持策略至少覆盖DP3、ActionFlow、ManiFlow、SAT、DQ-RISE、R3D。

---

## 11. PR顺序

### PR 0：基线与CI

- 两仓记录HEAD；
- 添加最小CI；
- branch protection为manual repository setting；
- 保存DP3 v2 golden fixture与shadow baseline；
- 不改runtime。

### PR 1：Policy correctness

- ActionFlow state_dim=19；
- AGENTS修正；
- use_aux_ee metadata；
- eval raw_state bug；
- DQ-RISE checkpoint描述；
- focused tests。

### PR 2：Policy exporter

- minimal v2 exporter；
- point-cloud支持；
- Zarr contract；
- atomic output；
- strict/no-network tests。

### PR 3：Real `control_action`

- full-output validation；
- control decode；
- tail不执行；
- docs/tests。

旧H4/task evidence不再适用于新semantic。

### PR 4：Real timing/readiness

- shared timing helper；
- usable endpoint；
- inference_mode；
- benchmark report。

### PR 5：Real CLI/logging/research mode

- subcommands；
- profile；
- strictness；
- headless shadow；
- typed errors；
- quiet console。

此PR不改控制数据流。

### PR 6：causal reader去重

- ipc causal history/alignment；
- worker委托；
- visual wrapper删除；
- differential tests。

### PR 7：wire/scheduler小PR序列

1. valid_mask→prefix；
2. deadline收敛；
3.字段删除；
4.EndpointBuffer experiment。

### PR 8：Safety局部去重

只合并重复permit read。

### PR 9：Hand observe-only

独立硬件验证。

### PR 10：RGB deployment

在完整preprocessing/self-contained restore后实施。

---

## 12. Codex执行规则

每个PR开始：

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

必须：

- 保留无关用户修改；
- 创建独立branch；
- 先读AGENTS及更深规则；
- 一个PR一个主要semantic variable；
- 不做无关rename/reformat；
- 更新对应文档；
- 明确未验证项；
- 不运行硬件命令。

### Real验证

```bash
python -m compileall -q dexmani_real examples tests
pytest -q tests/test_deployment_timing.py
pytest -q tests/test_action_buffer.py
pytest -q tests/test_coupled_command_publication.py
pytest -q tests/test_worker_command_validation.py
pytest -q tests/test_safety_gate_command_delta.py
git diff --check
```

### Policy验证

```bash
python -m compileall -q dexmani_policy
python dexmani_policy/smoke_test.py dp3
python dexmani_policy/smoke_test.py action_flow
python dexmani_policy/smoke_test.py maniflow
python dexmani_policy/smoke_test.py sat
git diff --check
```

新增tests后：

```bash
pytest -q tests/deployment
```

### 停止条件

立即停止，不做fallback猜测：

- HEAD发生不可解释漂移；
- schema与Real parser不一致；
- metadata/config冲突；
- strict restore key不匹配；
- restore访问网络；
- control_action与pred slice不一致；
- causal differential不同；
- shadow coupled sequence增加；
- worker fence失败；
- SafetyGate被绕过；
- 结论必须依赖真机才能确认。

---

## 13. 文档修复清单

### `dexmani_real/docs/policy_deployment.md`

- current main与last hardware-evidenced revision分开；
- 15 predicted / 8 executable；
- Real执行control_action；
- shadow只保证zero learned coupled writes；
- startup hand行为；
- usable-endpoint warmup；
- RGB降级为pending；
- DQ-RISE action_ee；
- 新CLI；
- semantic改变后必须重新H4。

### `dexmani_real/README.md`

- 新命令；
- inspect/check无硬件；
- shadow连接硬件；
- receipt/log路径。

### `dexmani_real/AGENTS.md`

- 删除“无general unit tests”的过时描述；
- 写明focused pytest suite。

### `dexmani_policy/AGENTS.md`

- joint_state固定19；
- editable install依赖边界；
- exporter入口；
- point-cloud-first支持矩阵；
- joint/EE smoke要求。

### `dexmani_policy/README.md`

明确：

```text
Training checkpoint != deployment artifact
```

给出train→evaluate/select best→export→Real inspect/check/shadow流程。

### Config comments

无稳定实验记录链接的claim（如“verified +2.9pp”）从runtime config移出或附上证据。

---

## 14. Definition of Done

### Policy

- [ ] ActionFlow joint/EE均使用19-D joint_state。
- [ ] 新checkpoint包含use_aux_ee。
- [ ] eval loader无scope bug。
- [ ] best checkpoint描述与真实流程一致。
- [ ] 一个命令导出deployment-v2和schema-v2 sidecar。
- [ ] point-cloud restore不依赖网络/外部初始化文件。
- [ ] RGB/MultiTask未支持时明确拒绝。
- [ ] direct/export control_action parity通过。

### Real

- [ ] run_policy具备inspect/check/shadow/h4/run。
- [ ] 无subcommand不连接硬件。
- [ ] physical seed显式且强制audited。
- [ ] 默认终端无metrics/per-endpoint/full JSON spam。
- [ ] 只transport n_action_steps control_action。
- [ ] tail永不默认执行。
- [ ] warmup按真实deadline。
- [ ] causal reader只有一个公共实现。
- [ ] stale action永不重定时。
- [ ] SafetyGate/atomic ticket/worker fence均保留。
- [ ] live shadow仍为zero learned coupled writes。
- [ ] 新semantic重新完成H4后才task rollout。

### 证据链

- [ ] offline export/restore；
- [ ] recorded replay；
- [ ] multiprocess shadow E2E；
- [ ] live shadow；
- [ ] H4 one endpoint；
- [ ] bounded task transport；
- [ ] task success独立评估，不把transport success当作policy success。

---

## 15. 第一项交给 Codex 的任务

先只执行 **PR 1：Policy correctness**：

```text
1. action_flow.yaml: state_dim=19
2. AGENTS.md: 修正joint_state描述
3. build_train_params: 增加use_aux_ee
4. eval_utils: 修复raw_state scope
5. dqrise.yaml: 修正checkpoint selection描述
6. 添加joint/action_ee config tests
7. compile + smoke
```

此阶段不碰：

```text
deployment exporter
Real runtime
checkpoint schema
SafetyGate
hardware
```

PR 1 合并并建立clean baseline后，再实施Policy exporter。
