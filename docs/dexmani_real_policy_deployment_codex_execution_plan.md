# DexMani Real Learned-Policy Deployment 简化与 Bug 修复
## Codex 执行文档

**目标仓库**：`dexmani_real`
**目标分支**：`temp-0902`
**适用场景**：个人机器人研究实验、可信本地机器、固定 xArm7 + XHand + RealSense、Policy 来源固定为 `dexmani_policy`
**参考项目**：ManiUniCon (`Universal-Control/ManiUniCon`)
**核心原则**：

> **Policy knows the model.**
> **Real knows the robot.**
> **CLI only expresses experiment + run intent.**

本文用于直接交给 Codex 执行。要求优先完成 offline / fake-hardware / synthetic validation，尽量避免请求真机权限；所有确实需要真机的步骤统一放在文档末尾的“真机验证阶段”。

---

# 1. 背景与问题定义

当前 `dexmani_real` 的 learned-policy deployment 已经超出个人研究实验所需要的复杂度。现有链路不仅负责模型推理和真实机器人执行，还承担了大量 release / qualification / evidence 职责，包括：

- artifact resolution；
- checkpoint sidecar；
- SHA-256；
- source-tree identity；
- producer provenance；
- canonical deployment receipt；
- H4 one-publication execute profile；
- bounded task execute profile；
- standalone preflight；
- deployment projection；
- physical evidence metrics；
- one-process-one-rollout 限制。

这些机制使一个本来应该回答：

```text
运行哪个 experiment？
是否允许机器人执行动作？
```

的问题，变成了一个 release-oriented deployment protocol。

当前项目的实际定位是：

```text
single user
trusted local workstation
fixed robot embodiment
fixed sensor stack
fixed policy repository
fast research iteration
```

因此，本轮目标不是“少几个 argparse 参数”，而是重新划分职责并删除错误 abstraction。

---

# 2. Review 后的核心结论

## 2.1 应删除的复杂度

以下复杂度主要服务于 release / qualification / provenance，不再适合当前个人研究定位：

```text
artifact resolver
checkpoint sidecar
checkpoint SHA verification
embedded contract SHA
runtime projection SHA
source-tree SHA
git clean admission gate
producer commit gate
canonical JSON receipt
H4 execute profile
task execute profile
bounded publication evidence
standalone preflight workflow
```

## 2.2 必须保留的复杂度

以下复杂度属于真实机器人系统本身，不应因为“简化”而删除：

```text
SafetyGate
RuntimeChannels
SDK single owner
separate inference process
worker-side command validation
generation
coupled command ticket
atomic arm/hand publication
ACK
heartbeat
readiness
supervisor
e-stop
error latch
causal observation timing
freshness
observation skew
IK validation
collision checks
verified shutdown
```

这些机制解决的是：

```text
hardware uncertainty
asynchronous sensing
multi-process concurrency
stale actions
partial publication
actuator delivery uncertainty
```

而不是 release engineering。

---

# 3. ManiUniCon 对照结论

ManiUniCon 适合作为 **interface simplicity / ownership clarity** 参考，不适合作为 DexMani 的 safety architecture 模板。

## 3.1 值得借鉴

ManiUniCon 的 `main.py` 非常薄，基本只做：

```text
resolve config
→ construct SharedStorage
→ instantiate robot
→ instantiate policy
→ instantiate sensors
→ start
```

Policy-specific 参数，例如：

```text
checkpoint
observation horizon
action horizon
point-cloud point count
task semantics
```

主要跟随 policy config，而不是由 `main.py` 理解。

这一点应直接吸收到 DexMani：

> Real 不应理解 checkpoint payload、EMA selection、normalizer structure 和模型内部 config。

## 3.2 不应照搬

ManiUniCon 支持多机器人 × 多 policy × 多 sensor，因此需要：

```text
Hydra config groups
robot=...
policy=...
sensors=...
obs_wrapper
act_wrapper
RobotInterface abstraction
```

DexMani 当前并没有这种自由组合需求。

固定组合是：

```text
xArm7
+
XHand
+
RealSense
+
DexMani Policy
```

因此不应重新引入：

```text
GenericPolicyFactory
PolicyRegistry
GenericObservationWrapper
GenericActionWrapper
robot/policy/sensor Hydra matrix
```

## 3.3 Safety 上不能复制 ManiUniCon

ManiUniCon 的 robot process startup 会自动 reset/init，这种启动即产生 physical motion 的语义不适合 DexMani。

DexMani 应追求：

> **ManiUniCon-level experiment simplicity + DexMani-specific actuator safety.**

---

# 4. 本轮第一优先级 Bug

# BUG-01：Shadow startup 可能导致 XHand 真实运动

当前 `hand_worker` startup 包含：

```text
connect
→ calibrate tactile
→ reset_home()
→ initial state
→ ready
```

由于 shadow 模式同样可能启动 hand worker，因此：

```text
shadow startup != zero physical motion
```

这是本轮最高优先级 bug。

## 修复目标

`hand_worker` startup 必须变为只读：

```text
construct driver
→ connect
→ calibrate tactile
→ read initial state
→ publish initial feedback
→ ready
```

禁止 startup 触发：

```text
reset_home
send_action
servo target
physical motion
```

Physical home 只能由：

```text
operator H
→ control/hand_home.py
→ control/arm_home.py
```

进入。

## Codex 修改范围

优先检查并修改：

```text
dexmani_real/robot/hand_worker.py
dexmani_real/deployment/operator.py
dexmani_real/control/hand_home.py
```

不要新增新的 home mechanism。

## Offline regression

必须增加 fake/mock 测试：

```text
hand worker startup
→ connect called
→ calibration called
→ initial state read
→ ready

assert reset_home not called
assert send_action not called
```

同时增加 shadow invariant test：

```text
shadow policy loop
→ candidate validated
→ actuator publication function not called
```

真机验证不要在此阶段执行。

---

# 5. 第二类问题：Policy / Real Ownership 错位

当前 `dexmani_real` 自己理解：

```text
deployment-v2 checkpoint payload
model vs EMA state
Hydra model construction
normalizer dimensions
model metadata
data contract
manifest
producer provenance
```

同时 `dexmani_policy` 已经拥有大量相同语义。

这会导致未来修改：

```text
action representation
auxiliary action layout
RGB policy
point-cloud shape
normalizer
EMA behavior
Diffusion/Flow inference parameters
```

时必须同步修改两个 repo，容易产生语义漂移。

因此，本轮必须把 model semantics 重新收回 `dexmani_policy`。

---

# 6. `dexmani_policy` 新 public API

建议 Codex 在 `dexmani_policy` 中建立稳定 deployment API。

## 6.1 `PolicySpec`

建议形态：

```python
@dataclass(frozen=True)
class PolicySpec:
    action_key: str
    action_dim: int
    control_action_dim: int

    horizon: int
    n_obs_steps: int
    n_action_steps: int

    sensor_modalities: tuple[str, ...]

    point_cloud_num_points: int | None = None
    point_cloud_feature_dim: int | None = None

    rgb_shape: tuple[int, int, int] | None = None
    rgb_color_order: str | None = None

    control_dt_s: float = 0.0
    requires_hand: bool = True
```

注意：

```text
PolicySpec 不是用户配置。
PolicySpec 是 experiment 本身的只读 contract。
```

## 6.2 `resolve_experiment()`

支持：

```python
path = resolve_experiment(selector)
```

规则：

```text
如果 selector 是有效 experiment 路径：直接使用
否则：由 dexmani_policy 解释为短 experiment selector
```

Real 不应理解 Policy repo 的 experiment directory layout。

## 6.3 `list_experiments()`

为了研究便利性，增加简单 filesystem-based listing：

```python
list_experiments(filter: str | None = None)
```

禁止演化为：

```text
ExperimentRegistry
ExperimentDatabase
ExperimentManager
```

## 6.4 `inspect_experiment()`

```python
spec = inspect_experiment(experiment)
```

职责：

```text
resolve experiment
resolve checkpoint selection
read model config / metadata
derive PolicySpec
```

要求：

```text
NO CUDA
NO model instantiate
NO Real import
NO hardware
```

## 6.5 `load_experiment()`

```python
runtime = load_experiment(
    experiment,
    device="cuda:0",
    seed=0,
)
```

由 Policy 完整负责：

```text
checkpoint load
agent construction
EMA/model selection
strict load_state_dict
normalizer validation
model preprocessing config
denoise/flow inference config
eval()
```

Real 不再知道：

```text
weights.model
weights.ema_model
deployment-v2 payload structure
Hydra agent target
checkpoint internal state layout
```

---

# 7. `run_policy.py` 实验便利性专项设计

本章属于本轮核心目标，不是后续优化。

研究工作流是：

```text
train
→ select experiment
→ check
→ shadow
→ run
→ diagnose
→ modify
→ repeat
```

CLI、模型 load、process restart 和日志定位都直接影响每天实验效率。

---

# 8. 最终 CLI

标准命令只保留：

```bash
python examples/run_policy.py list
python examples/run_policy.py check EXPERIMENT
python examples/run_policy.py shadow EXPERIMENT
python examples/run_policy.py run EXPERIMENT
```

Advanced 参数最多保留：

```text
--device
--config
```

且 README 的标准 workflow 不展示它们。

删除：

```text
--execution-mode
--execute-*
--task-*
--hand
--inference-seed
--deployment-config
--preflight-only
--print-config
checkpoint SHA 参数
```

---

# 9. `list` 设计

```bash
python examples/run_policy.py list
```

输出示例：

```text
Recent experiments:

NAME                        POLICY   TASK       CHECKPOINT
dp3/pick/0902_seed0         DP3      pick       best
dp3/pick/0902_seed1         DP3      pick       best
r3d/place/0901_aux          R3D      place      latest
```

可选：

```bash
python examples/run_policy.py list dp3
```

约束：

```text
NO model load
NO CUDA
NO sensor
NO robot
```

不建议支持魔法：

```bash
run_policy.py run latest
```

Physical experiment 应明确选择 experiment。

---

# 10. 启动时打印 Experiment Summary

每次 `check / shadow / run` 都先打印紧凑 summary：

```text
── Policy Experiment ──

Mode          : RUN
Experiment    : dp3/pick/0902_seed0
Policy        : DP3
Checkpoint    : best.pt
Device        : cuda:0

Observation   : joint_state + point_cloud
History       : 2
Point Cloud   : 2048 × 6

Action        : joint19
Chunk         : 8
Control       : 16 Hz
Inference     : 10 Hz

───────────────────────
```

不要打印：

```text
SHA
source-tree hash
producer metadata
projection hash
canonical receipt
```

---

# 11. `check`：真正实用的 model smoke test

命令：

```bash
python examples/run_policy.py check EXP
```

流程：

```text
resolve experiment
→ inspect
→ load model
→ strict restore
→ normalizer validation
→ synthetic observation
→ warmup
→ predict
→ output contract validation
→ exit
```

输出：

```text
restore .......... OK
normalizer ....... OK
warmup ........... OK
prediction ....... OK

inference warmup p50: 43 ms
inference warmup max: 47 ms

READY
```

错误必须按 owner 定位：

```text
[POLICY] checkpoint restore failed: ...
[POLICY] normalizer.point_cloud dim=3, expected=6
[COMPAT] Policy requires 2048 points, Real provides 1024
```

禁止返回模糊错误：

```text
invalid deployment receipt/config
```

---

# 12. `shadow`：最低摩擦硬件 connected dry run

命令：

```bash
python examples/run_policy.py shadow EXP
```

允许：

```text
real camera
real point cloud
real arm feedback
real hand feedback
policy inference
IK
SafetyGate
candidate validation
```

禁止：

```text
physical home
arm command publication
hand command publication
SDK action
```

启动后：

```text
Policy ready
Camera ready
Pointcloud ready
Arm feedback ready
Hand feedback ready

SHADOW ARMED
Physical motion disabled

[B] Begin
[S] Stop
[Q] Quit
[ESC] E-stop
```

Shadow 必须支持多 episode：

```text
B → S
B → S
B → S
```

---

# 13. `run`：显式两阶段物理授权

命令：

```bash
python examples/run_policy.py run EXP
```

启动后：

```text
RUN ARMED

[H] Home
[B] Begin
[S] Stop
[Q] Quit
[ESC] E-stop
```

Physical run 必须：

```text
H success
→ physical_home_completed=True

fresh B
→ new generation
→ RUNNING
```

第一版继续要求每个 episode 前重新 H：

```text
H → B → S
H → B → S
H → B → S
```

不要在第一轮重构中增加 “skip home” 模式。

---

# 14. 同一 Process 多 Episode

这是本轮最重要的实验效率提升之一。

当前 qualification-driven 逻辑倾向于：

```text
one process ≈ one rollout
```

应修改为：

```text
load model once
warmup once
connect camera once
connect robot once

H → B → S
H → B → S
H → B → S

Q
```

收益：

```text
不重复 CUDA initialization
不重复 checkpoint restore
不重复 warmup
不重复 camera startup
不重复 robot connect
```

删除：

```text
operator.home_attempted
coordinator.policy_session_started
```

保留：

```text
physical_home_completed
run_generation
```

建议语义：

```text
H success:
    physical_home_completed = True

B success:
    require physical_home_completed
    consume it
    physical_home_completed = False
    begin new generation
```

---

# 15. 每 Episode 输出 Compact Summary

删除 canonical evidence receipt 后，保留研究者真正需要的 diagnostics。

示例：

```text
── Episode 3 ──

duration              8.41 s
stop reason           operator
commands published    116
commands dropped      3
safety rejects        0

inference p50         42.1 ms
inference p95         46.7 ms
observation age p95   33.2 ms
usable horizon p50    281 ms

status                STOPPED
─────────────────
```

失败：

```text
status: ABORTED
reason: point-cloud observation stale for 0.31 s
```

可选 lightweight JSONL：

```text
logs/policy_runs.jsonl
```

但它只是 research log：

```text
不参与 admission
不要求 canonical JSON
不要求 SHA
不影响运行成功/失败
```

第一阶段甚至可以只打印，不落盘。

---

# 16. Error Message 规范

统一 startup/runtime owner 前缀：

```text
[POLICY]
[COMPAT]
[CAMERA]
[POINTCLOUD]
[ARM]
[HAND]
[SAFETY]
```

例：

```text
[POLICY] missing normalizer key 'point_cloud'

[COMPAT] Policy requires 2048 points, Real pointcloud config provides 1024

[CAMERA] no valid RealSense frame received within timeout

[SAFETY] episode aborted: arm feedback stale
```

研究便利性优先于生成复杂 machine-readable receipt。

---

# 17. 删除 Deployment Framework

在 Policy public API 可用后，Real 侧逐步删除：

```text
dexmani_real/deployment/artifact.py
dexmani_real/deployment/manifest.py
dexmani_real/deployment/policy_checkpoint.py
dexmani_real/deployment/run_identity.py
```

---

# 18. 删除 `artifact.py`

删除：

```text
deployment sidecar
O_NOFOLLOW artifact resolver
lstat identity
inode/device identity
directory identity
one-hop symlink validation
TOCTOU recheck
producer metadata
checkpoint digest metadata
```

替代：

```text
Real
→ dexmani_policy.resolve_experiment
→ dexmani_policy.inspect_experiment
→ PolicySpec
```

---

# 19. 删除 `policy_checkpoint.py`

Real 不再解析：

```text
dexmani.deployment.v2
state
weights
model
ema_model
train_params
inference_config
data_contract
producer
deployment_contract
```

全部移动到 `dexmani_policy.load_experiment()`。

---

# 20. 删除 `manifest.py`

当前 model contract 被多层重复描述：

```text
PolicyArtifactContract
→ DeploymentConfig
→ DeploymentManifest
→ restored agent validation
```

最终压缩为：

```text
PolicySpec
    ├── Policy validates restored model
    └── Real validates runtime compatibility
```

---

# 21. 删除 `run_identity.py`

删除：

```text
Real git clean gate
Real Python source-tree SHA
Policy source-tree SHA
producer commit admission
canonical run receipt
```

如果实验需要版本记录，仅作为 debug/research metadata：

```text
Real git HEAD
Policy git HEAD
experiment path
checkpoint filename
```

禁止这些信息成为 physical execution admission 条件。

---

# 22. 删除独立 Preflight

事实说明：

当前 `--preflight-only` 是独立 invocation，并不是同一 lifecycle 中自动 load 两次模型。

实际重复是：

```text
manual preflight invocation
→ restore + warmup + predict
→ exit

actual run invocation
→ restore + warmup
→ run
```

删除 standalone preflight 的理由是：正常 inference worker 本身已经可以成为 operational preflight。

目标启动：

```text
resolve PolicySpec
→ allocate RuntimeChannels
→ spawn inference ONLY
→ load_experiment
→ strict restore
→ warmup
→ latency qualification
→ set_ready("inference")
→ only then start hardware workers
```

`check` 则直接在单独进程中：

```text
load
→ warmup
→ predict
→ exit
```

---

# 23. 删除 SHA 体系

用户明确不需要 SHA verification。

从 learned-policy runtime 删除：

```text
checkpoint SHA
index SHA
embedded contract SHA
projection SHA
runtime SHA admission
Policy source-tree SHA
Real source-tree SHA
```

保留普通：

```text
experiment selector
checkpoint filename
```

如果论文复现后续确实需要 hash，可作为 offline metadata 工具单独添加，不进入 realtime control path。

---

# 24. Execution Mode 简化

当前：

```text
shadow
execute
 task
```

其中 `execute/task` 是 qualification vocabulary。

外部改成：

```text
check
shadow
run
```

内部不要传播 CLI vocabulary。

`check`：单独 code path。

runtime 只需要：

```python
execute: bool
```

即：

```text
execute=False → shadow validation
execute=True  → physical publication
```

删除：

```text
H4ExecuteBounds
TaskExecuteBounds
physical_execute_bounds
```

---

# 25. Coordinator 简化

当前 coordinator 同时维护 scheduler、安全、H4/task qualification、receipt 和 bounded publication state。

最终只保留 realtime controller 职责：

```text
receive latest plan
→ select due endpoint
→ joint or EE→IK
→ build ActionCandidate
→ SafetyGate
→ execute=False: validate only
→ execute=True: coupled publish
→ ACK
```

删除 coordinator state：

```text
shadow_run_generation
shadow_start_coupled_sequence
execute_run_generation
execute_start_coupled_sequence
execute_first_publication_ns
execute_last_publication_ns
execute_published_endpoints
execute receipt
H4/task counters
policy_session_started
```

保留：

```text
run_generation
scheduler
previous_arm_command
last_valid_command_time
run_started_time
pending_ack
```

---

# 26. Shadow no-write 机制简化

不再需要通过 receipt + coupled ring sequence 证明 shadow 没有写入。

优先让 no-write 成为结构属性：

```python
validate_and_send_candidate(
    ...,
    execute=False,
)
```

内部：

```text
SafetyGate
→ if not execute: return VALIDATED
→ publish
```

测试：

```text
execute=False
→ publication function never invoked
```

这比运行后再做 evidence arithmetic 更直接。

---

# 27. ACK 保留

删除的是：

```text
ACK as qualification evidence
```

保留的是：

```text
ACK as actuator delivery validation
```

Physical command 仍然需要：

```text
publish
→ generation + ticket
→ arm worker final validation
→ hand worker final validation
→ both applied
→ ACK
```

ACK 属于真实控制语义，不属于过度工程化。

---

# 28. Config Ownership 收敛

当前存在：

```text
ResolvedRuntimeConfig
DeploymentConfig
PolicyRuntimeConfig
ResolvedPolicyRuntimeConfig
CoordinatorConfig
```

目标 ownership：

```text
Policy-owned:
    PolicySpec

Real-owned:
    ResolvedRuntimeConfig

Session-owned:
    experiment
    execute bool
    device
```

Real timing：

```text
inference_hz
max_input_age_s
max_observation_skew_s
max_grid_lag_s
max_plan_age_s
max_source_to_command_age_s
command_lead_s
max_command_silence_s
action_validity_s
first_command_timeout_s
```

统一归入：

```text
ResolvedRuntimeConfig.policy
```

删除：

```text
DeploymentConfig
ResolvedPolicyRuntimeConfig
H4ExecuteBounds
TaskExecuteBounds
```

如果 inference worker 仍需要一个窄配置，可用：

```python
@dataclass(frozen=True)
class PolicyWorkerConfig:
    experiment: str
    device: str
    seed: int = 0
```

不要再保存 model shape/horizon 的重复配置。

---

# 29. `--hand` 删除

固定 embodiment 本身包含 XHand。

是否需要 hand 应由：

```text
PolicySpec.requires_hand
```

决定。

Real 只做 compatibility：

```text
PolicySpec.requires_hand
→ Real supports XHand?
```

不要再要求操作者输入：

```text
--hand
```

---

# 30. `--inference-seed` 删除

普通研究运行：

```text
seed = 0
```

作为确定性默认值。

需要 stochastic ablation 时，seed 应进入 experiment/config，而不是日常 real-run CLI。

---

# 31. Metrics 简化

原则：

> **删 evidence metrics，不删 experiment diagnostics。**

删除：

```text
H4 evidence counters
bounded execute receipt
shadow receipt
provenance metrics
physical evidence counters
```

保留/加强：

```text
observation_age_ms
observation_skew_ms
inference_ms
plan_age_ms
usable_horizon_ms
plans_created
plans_dropped
endpoints_published
endpoints_stale
SafetyGate rejects
IK failures
ACK latency
ACK failure
command_silence_abort
```

---

# 32. Observation Timing 不应删除

禁止为了减少代码量删除：

```text
source <= publish <= anchor
camera generation
max age
state-camera causal alignment
max observation skew
control-grid selection
run-start lower bound
logical-grid advance
```

这些属于真实异步 sensing semantics。

可以清理：

```text
alias
重复 diagnostics
重复 contract wrapper
```

但不要修改 causal semantics，除非有独立实验假设和回归测试。

---

# 33. ActionBuffer 最后处理

`ActionBuffer` 当前较复杂：

```text
multiple plans
EndpointToken
supersession map
eviction
watermark
exact-target resurrection prevention
```

但它是实际 scheduling semantics，不应和 artifact/receipt cleanup 混在一起。

第一轮重构保持现状。

最终可单独尝试：

```python
@dataclass
class ActivePlan:
    plan_id: int
    generation: int
    observation_id: int
    deadline_ns: int
    chunk: JointActionChunk
    next_index: int
```

必须使用 differential tests 比较：

```text
overlapping plans
late plan
expired plan
transient defer
generation switch
```

如果不能证明行为合理，则保留 ActionBuffer。

---

# 34. 目标目录结构

最终 learned-policy deployment 建议收敛为：

```text
dexmani_real/
├── deployment/
│   ├── contracts.py
│   ├── observation.py
│   ├── timing.py
│   ├── scheduler.py
│   ├── worker.py
│   ├── coordinator.py
│   ├── operator.py
│   ├── lifecycle.py
│   └── metrics.py
│
├── integrations/
│   └── dexmani_policy.py
│
├── control/
│   ├── safety_gate.py
│   ├── publication.py
│   ├── arm_home.py
│   └── hand_home.py
│
├── runtime/
│   ├── safety.py
│   ├── workers.py
│   └── supervisor.py
│
└── robot/
    ├── arm_worker.py
    └── hand_worker.py
```

目标删除：

```text
deployment/artifact.py
deployment/manifest.py
deployment/policy_checkpoint.py
deployment/run_identity.py
```

`deployment/config.py` 最终应大幅缩减或完全消失。

---

# 35. Codex 实施 Phase

必须按以下顺序执行，避免 giant refactor。

---

## Phase 0 — 修复 XHand startup side effect

### 修改

```text
robot/hand_worker.py
```

删除 implicit `reset_home()`。

### 增加 offline test

```text
startup no-motion
shadow no-publication
```

### 验收

```text
compile passes
fake-hand tests pass
no hardware required
```

### Commit

```text
fix: remove implicit XHand motion from worker startup
```

---

## Phase 1 — 建立 Policy public API

### 在 dexmani_policy 中增加

```text
PolicySpec
resolve_experiment()
list_experiments()
inspect_experiment()
load_experiment()
```

优先复用已有 restore / normalizer / prediction validation，不重写一套新的 model loader。

### 验收

对本地真实 experiment 文件进行 offline：

```text
resolve
inspect
load
synthetic predict
```

不连接 robot/sensor。

### Commit

```text
feat(policy): expose experiment inspection and runtime loading API
```

---

## Phase 2 — 先重做 `run_policy` UX

先让 researcher 能使用：

```bash
run_policy.py list
run_policy.py check EXP
run_policy.py shadow EXP
run_policy.py run EXP
```

此时内部仍可暂时调用旧 lifecycle wrapper，重点是先验证新的 research workflow。

### 验收

`list/check` 完全 offline。

`shadow/run` 只做 CLI / codepath validation，不实际启动真机。

### Commit

```text
refactor: simplify run_policy to list-check-shadow-run
```

---

## Phase 3 — 删除 model artifact framework

在 Policy API 已可用后删除：

```text
artifact.py
policy_checkpoint.py
manifest.py
run_identity.py
```

同时移除：

```text
sidecar dependency
checkpoint SHA
source-tree SHA
producer provenance
git clean gate
canonical receipt
```

### 验收

```text
check EXP
```

可以完整 restore + predict。

### Commits

```text
refactor: move model restore ownership into dexmani_policy
refactor: remove artifact and manifest deployment layers
refactor: remove deployment provenance and SHA qualification
```

---

## Phase 4 — 删除 standalone preflight

用 inference readiness 作为 operational preflight。

### 验收

fake lifecycle：

```text
model load failure
→ inference not ready
→ hardware workers never started
```

### Commit

```text
refactor: remove standalone deployment preflight
```

---

## Phase 5 — 删除 H4 / task semantics

删除：

```text
execute profile
 task profile
H4ExecuteBounds
TaskExecuteBounds
publication bounds
bounded receipts
```

内部改成：

```text
execute: bool
```

### 验收

offline coordinator tests：

```text
execute=False → validate only
execute=True → publication path invoked
```

### Commit

```text
refactor: collapse execute-task profiles into shadow-run semantics
```

---

## Phase 6 — 支持同进程多 Episode

删除：

```text
operator.home_attempted
coordinator.policy_session_started
```

实现：

```text
shadow:
B → S → B → S

run:
H → B → S → H → B → S
```

使用 generation 隔离 episode。

### Offline 验收

用 fake shared state / fake operator signal 验证：

```text
new B advances generation
old plan invalid
old ticket invalid
second episode allowed
```

### Commit

```text
refactor: support repeated policy episodes in one process
```

---

## Phase 7 — Coordinator cleanup

删除 qualification/evidence state，只保留 realtime controller state。

### Offline regression

必须覆盖：

```text
wrong generation
stale plan
SafetyGate rejection
ACK timeout
worker rejection
operator stop
first-command timeout
command-silence timeout
```

### Commit

```text
refactor: simplify policy coordinator execution state
```

---

## Phase 8 — Config ownership cleanup

把 Real-owned timing 合并到：

```text
ResolvedRuntimeConfig.policy
```

删除：

```text
DeploymentConfig
ResolvedPolicyRuntimeConfig
```

### Commit

```text
refactor: merge policy runtime timing into Real runtime config
```

---

## Phase 9 — Metrics / docs cleanup

删除 evidence receipt 系统，增加 compact episode summary。

更新：

```text
README.md
repo_map.md
relevant deployment documentation
```

注意当前文档 inventory 与实际 `docs/` 内容可能不一致，重构后需要同步修正。

### Commit

```text
cleanup: replace deployment evidence with experiment diagnostics

docs: rewrite learned-policy experiment workflow
```

---

## Phase 10 — Optional scheduler simplification

独立 commit / branch。

尝试：

```text
ActionBuffer → ActivePlan
```

只有 differential tests 通过后才保留。

### Commit

```text
refactor: simplify latest-plan scheduler
```

这一阶段不是本轮完成标准的硬要求。

---

# 36. 最小 Offline Regression Suite

当前仓库缺少 general unit test suite。本轮必须至少建立 focused tests。

## Model

```text
strict restore
missing normalizer
wrong action dim
wrong observation contract
NaN/Inf prediction
```

## Startup

```text
hand worker startup performs no motion
inference failure prevents hardware worker startup
```

## Shadow

```text
zero actuator publication
SafetyGate still runs
```

## Timing

```text
late prediction prefix masking
expired plan rejection
source-to-command deadline
logical target grid unchanged
```

## Safety

```text
joint limits
workspace
arm jump
hand mechanical bounds
collision rejection
```

## Generation

```text
old plan rejected after new episode
old ticket rejected
```

## Publication

```text
arm + hand share same ticket
partial coupled command impossible
```

## Worker

```text
stale command rejected
wrong generation rejected
expired command rejected
invalid bounds rejected
```

## ACK

```text
arm + hand ACK success
ACK timeout
worker reject
```

## Operator

```text
B before H rejected in run mode
H → B accepted
S returns ARMED
second H → B accepted
ESC latches e-stop
```

---

# 37. Codex 默认验证命令

除非某阶段明确需要更多检查，优先使用：

```bash
git status --short
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
```

如果存在 focused tests，再运行对应测试。

禁止把 examples 直接当测试执行，除非已确认不会触发硬件。

---

# 38. Codex 执行约束

## 38.1 不要 giant refactor

禁止同一 commit 同时修改：

```text
Policy API
CLI
artifact deletion
coordinator semantics
scheduler semantics
observation timing
```

## 38.2 每个阶段报告

Codex 每完成一个阶段，应报告：

```text
Changed:
Removed:
Validated:
Not validated:
Remaining risks:
```

## 38.3 不要为删除而删除

如果某机制满足以下任一条件，默认保留：

```text
prevents stale command
prevents partial arm/hand command
provides actuator delivery acknowledgement
protects worker SDK boundary
validates causal sensor timing
supports e-stop / fault containment
```

## 38.4 不新增通用框架

禁止为了“未来扩展”新增：

```text
registry
factory
plugin manager
event bus
service/controller hierarchy
generic robot/policy framework
```

除非有当前已存在的第二个实现证明 abstraction 有必要。

---

# 39. 真机权限策略

**此前所有 Phase 默认不请求真机权限。**

优先使用：

```text
source inspection
compile
unit tests
synthetic observations
fake XHand
fake arm worker
fake RuntimeChannels
mock publication
mock ACK
```

只有完成 offline 重构并通过 regression 后，再进入以下真机验证阶段。

---

# 40. 真机验证阶段（集中执行）

本阶段需要明确用户授权后再执行。

建议顺序从低风险到高风险。

---

## Hardware Test 1 — `check`

```bash
python examples/run_policy.py check EXP
```

实际上不需要 robot motion，也不应该连接硬件。

验证：

```text
Policy restore
normalizer
warmup
prediction
```

---

## Hardware Test 2 — Shadow startup no-motion

```bash
python examples/run_policy.py shadow EXP
```

验证：

```text
camera ready
pointcloud ready
arm feedback ready
hand feedback ready
policy ready
```

同时人工确认：

```text
XHand startup does not move
arm does not move
```

开始 shadow episode：

```text
B
...
S
```

确认：

```text
zero arm SDK command
zero hand SDK command
```

这是最重要的真机 regression。

---

## Hardware Test 3 — Single low-risk physical episode

```bash
python examples/run_policy.py run EXP
```

流程：

```text
H
→ verify safe home

B
→ short run

S
```

观察：

```text
SafetyGate
coupled ticket
arm/hand publication
worker validation
ACK
stop → ARMED
```

---

## Hardware Test 4 — Multi-episode lifecycle

同一 process：

```text
H → B → S
H → B → S
H → B → S
```

验证：

```text
no process restart required
new generation each episode
old commands invalid
home required each episode
```

---

## Hardware Test 5 — Operator safety

在安全条件下验证：

```text
S immediately revokes motion
Q shuts down cleanly
ESC latches e-stop
```

不要为覆盖 test case 故意制造危险 hardware fault。

---

# 41. 最终验收标准

## UX

普通 researcher 只需要：

```bash
run_policy.py list
run_policy.py check EXP
run_policy.py shadow EXP
run_policy.py run EXP
```

## Ownership

修改 model/checkpoint semantics 时：

```text
主要修改 dexmani_policy
```

Real 不再需要同步 checkpoint parser。

## Shadow

```text
startup zero actuator motion
runtime zero actuator publication
```

## Physical Run

每个动作仍经过：

```text
PolicyPrediction
→ timing
→ SafetyGate
→ coupled publication
→ generation + ticket
→ worker final validation
→ SDK
→ ACK
```

## Multi-episode

同一 process 支持：

```text
H → B → S
H → B → S
```

## 删除完成

以下模块消失：

```text
artifact.py
manifest.py
policy_checkpoint.py
run_identity.py
```

以下概念从 normal workflow 消失：

```text
H4
 task execute
checkpoint SHA
producer provenance
canonical receipt
deployment sidecar
```

## 保留高价值复杂度

仍然存在：

```text
SafetyGate
generation
ticket
ACK
worker validation
causal timing
supervisor
e-stop
```

---

# 42. 推荐 Commit 序列

```text
1. fix: remove implicit XHand motion from worker startup

2. test: enforce no-motion worker and shadow startup

3. feat(policy): expose experiment resolver and PolicySpec

4. feat(policy): expose inspect/load/list experiment APIs

5. refactor: simplify run_policy to list-check-shadow-run

6. refactor: move checkpoint restore ownership to dexmani_policy

7. refactor: remove artifact and manifest deployment layers

8. refactor: remove deployment provenance and SHA qualification

9. refactor: remove standalone deployment preflight

10. refactor: collapse execute-task profiles into shadow-run semantics

11. refactor: support repeated policy episodes in one process

12. refactor: simplify policy coordinator execution state

13. refactor: merge policy runtime timing into Real runtime config

14. cleanup: replace deployment evidence with experiment diagnostics

15. docs: rewrite learned-policy experiment workflow

16. optional: simplify latest-plan scheduler
```

---

# 43. 最终设计原则

本轮重构不是让系统变成“小脚本”。

真实机器人 deployment 必然需要：

```text
sensor synchronization
real-time scheduling
SafetyGate
IK
command publication
generation
ticket
ACK
e-stop
worker lifecycle
```

合理的目标是把复杂度集中在这些地方，而不是：

```text
SHA256
canonical JSON
git dirty check
source-tree hashes
sidecars
H4 receipts
release qualification
artifact provenance
```

因此，本轮最终原则是：

> **删除为了证明系统“合格”而存在的复杂度。**
>
> **保留为了让机器人正确、安全运行而存在的复杂度。**

以及：

> **ManiUniCon-like usability, DexMani-specific safety.**
