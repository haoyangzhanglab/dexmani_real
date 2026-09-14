# CODEX_TASK — P0-1：从根源解耦 Real Replanning Cadence 与 Policy `n_action_steps`

> 临时 Codex 实现任务书。开始工作前必须先阅读并遵守仓库根目录 `AGENTS.md`。
> 完成实现、独立检查和离线验证后，**在最终变更中删除本文件**，不要把一次性任务计划长期留在仓库。
>
> 本任务按 `haoyangzhanglab/dexmani_real@923238039466fe162e3aafa93777f52db85f99cc` 编写。
> 如果实际 worktree 已前进，以当前源码、schema、resolved config 为唯一事实来源；先重新核对调用链，再按本任务的终态语义实施，禁止机械套 patch。

---

## 1. 任务目标

将真实机器人 learned-policy rollout 的 replanning cadence 从 `PolicySpec.n_action_steps` 中**彻底解耦**。

当前 `dexmani_real/deployment/inference/worker.py` 使用：

```python
step_dt_ns = int(round(float(config.spec.control_dt_s) * 1e9))
inference_period_ns = int(config.spec.n_action_steps) * step_dt_ns
```

这使 `n_action_steps` 同时承担了两个本应独立的职责：

```text
dexmani_policy:
    n_action_steps
    = Policy / simulation 中 canonical control_action slice 的长度

dexmani_real:
    n_action_steps
    = 真实机多久重新 observation + inference 一次
```

本任务完成后的终态必须是：

```text
dexmani_policy owns:
    horizon
    n_obs_steps
    n_action_steps
    chunk_size / full future prediction
    action representation

dexmani_real owns:
    control_hz
    replan_steps
    causal observation timing
    inference scheduling
    prediction execution / safety
```

核心运行语义：

```text
inference_period = replan_steps * control_dt_s
```

而不是：

```text
inference_period = n_action_steps * control_dt_s
```

### 最重要的架构验收条件

本任务完成后，`dexmani_real` 的 functional code **不得再读取、比较、fallback 或推导 `PolicySpec.n_action_steps`**。

`n_action_steps` 继续保留在相邻 `dexmani_policy` 中；本任务不修改它，也不修改相邻仓库。

---

## 2. 当前源码事实：必须保留

实施前重新打开并核对这些路径。

### 2.1 Real 已经消费 full future chunk

Real 的模型边界 `dexmani_real/deployment/inference/runtime.py::PolicyRuntime` 提供：

```python
def predict_action_chunk(self, observation: PolicyObservation) -> np.ndarray: ...
```

`dexmani_real/deployment/inference/dexmani_policy.py` 直接转发到 Policy public runtime 的 `predict_action_chunk()`。

`dexmani_real/deployment/inference/worker.py::_predict_action_chunk()` 当前要求：

```python
actions.shape == (spec.chunk_size, spec.control_action_dim)
```

因此 Real 真正可执行的 proposal 长度由：

```text
PolicySpec.chunk_size
```

定义，而不是由 `n_action_steps` 定义。

### 2.2 Executor 已经按 absolute logical time 对齐 prediction

`PolicyExecutor` 已使用：

```text
prediction.logical_step_monotonic_ns
step_dt_ns
first_future_step_index(...)
```

跳过 inference latency 导致已经过期的 chunk prefix。

不要重写或复制这套时间轴。

### 2.3 Inference worker 已经是独立进程

Inference 和 Executor 已经异步解耦：模型推理时 Executor 可以继续消费已有 prediction。

不要引入：

```text
ThreadPoolExecutor
background inference queue
第二套 step counter
第二套 async broker
```

### 2.4 当前 deadline catch-up 语义是正确的

Worker 在 inference 完成后调用：

```python
deadline_ns = next_periodic_deadline_ns(
    deadline_ns,
    inference_period_ns,
    finished_ns,
)
```

这意味着若一次 inference 超过一个或多个目标周期，不会积压补跑历史 inference，而是直接跳到第一个未来 deadline。

因此 `replan_steps` 的准确语义是：

> target inference / replanning period measured in Policy control-grid steps.

它**不是**：

> guarantee that a fresh prediction will arrive after exactly every `replan_steps` executed actions.

保留当前 deadline mechanism，不新增补偿状态机。

---

## 3. 设计终态

新增一个且只有一个 Real-owned cadence 参数：

```python
@dataclass(frozen=True)
class PolicyParams:
    control_hz: float = 16.0
    replan_steps: int = 8
    ...
```

### 为什么是普通 `int`

必须是：

```python
replan_steps: int = 8
```

不要设计：

```python
replan_steps: int | None = None
```

也不要设计任何 fallback：

```python
if replan_steps is None:
    replan_steps = spec.n_action_steps
```

`8` 是当前 DexMani Real 默认 rollout protocol 的明确 Real 配置，而不是从 Policy 推断出来的兼容值。

### 唯一跨边界约束

Real 可以在下一次 inference 到来前继续执行当前 full future chunk，因此合法范围是：

```text
1 <= replan_steps <= policy_spec.chunk_size
```

不要使用：

```text
replan_steps <= policy_spec.n_action_steps
```

否则会重新建立本任务正在删除的耦合。

---

## 4. 明确禁止的兼容方案

本任务要求根治旧机制，不接受以下任何实现：

```text
replan_steps=None
fallback to n_action_steps
legacy_replan_mode
use_policy_action_steps_for_replan
replan_steps aliasing n_action_steps
if new config missing, read spec.n_action_steps
R <= n_action_steps compatibility check
```

也不要同时保留两套：

```text
old inference_period path
new inference_period path
```

最终 Worker 必须只有一个 source of truth：

```text
policy.replan_steps
```

---

## 5. Expected edit surface

预计需要修改：

```text
MODIFY  dexmani_real/config/defaults.py
MODIFY  dexmani_real/deployment/config.py
MODIFY  dexmani_real/deployment/inference/worker.py
MODIFY  examples/run_policy.py
MODIFY  dexmani_real/deployment/lifecycle.py
OPTIONAL/MINIMAL  README.md
```

`README.md` 仅在需要使 public `run_policy.py` 参数描述不陈旧时做一行级更新；不要复制具体默认值或 timing 细节到 README。

不应修改：

```text
dexmani_real/deployment/inference/runtime.py
dexmani_real/deployment/inference/dexmani_policy.py
dexmani_real/deployment/prediction.py
dexmani_real/deployment/timing.py
dexmani_real/deployment/executor.py
dexmani_real/ipc/schema.py
dexmani_real/ipc/channels.py
dexmani_real/control/safety_gate.py
```

如果当前 HEAD 已变化，允许为同一个 contract 做必要的最小额外修改，但必须在最终 handoff 解释原因。

绝对不要修改相邻 `dexmani_policy` 仓库来完成本任务。

---

# 6. Task A — 在 `PolicyParams` 中加入明确的 `replan_steps`

文件：

```text
dexmani_real/config/defaults.py
```

在 `PolicyParams` 的 `control_hz` 附近加入：

```python
replan_steps: int = 8
```

推荐局部注释表达 ownership，而不是历史兼容：

```text
PolicySpec owns model prediction shape/history and action-grid spacing.
Real owns how often it re-observes and replans on that grid.
```

不要写诸如：

```text
same as current n_action_steps
kept for compatibility
matches training config
```

### 最小 validation

在 `PolicyParams.validate()` 中只验证本地属性：

```python
if isinstance(self.replan_steps, bool) or not isinstance(self.replan_steps, int):
    raise TypeError(...)
if self.replan_steps < 1:
    raise ValueError(...)
```

保持与本文件现有 dataclass validation 风格一致即可；不要建立新的 validation framework。

这里不要访问 `PolicySpec`，因为 `defaults.py` 不拥有跨仓库 contract。

---

# 7. Task B — 在现有 Policy/Real compatibility boundary 检查 chunk coverage

文件：

```text
dexmani_real/deployment/config.py
```

现有 `validate_policy_runtime_compatibility(policy_spec, runtime)` 已经同时拥有：

```text
PolicySpec
resolved ExperimentConfig
```

并已经检查 `policy_spec.chunk_size <= MAX_PREDICTION_STEPS`。

在这里增加唯一需要的跨边界约束：

```python
if runtime.policy.replan_steps > policy_spec.chunk_size:
    raise ValueError(
        "policy.replan_steps exceeds the available Policy future chunk"
    )
```

措辞可以按当前代码风格微调。

### 不要增加

```text
replan_steps <= n_action_steps
replan_steps divides chunk_size
replan_steps divides horizon
replan_steps must be power of two
inference latency based validation
```

这些都不是 correctness constraint。

---

# 8. Task C — Worker 只使用 `policy.replan_steps`

文件：

```text
dexmani_real/deployment/inference/worker.py
```

将：

```python
step_dt_ns = int(round(float(config.spec.control_dt_s) * 1e9))
inference_period_ns = int(config.spec.n_action_steps) * step_dt_ns
```

改为：

```python
step_dt_ns = int(round(float(config.spec.control_dt_s) * 1e9))
inference_period_ns = int(policy.replan_steps) * step_dt_ns
```

除此之外，不改主循环 scheduling state machine。

特别不要改：

```text
deadline initialization
next_periodic_deadline_ns(...)
observation logical-grid construction
last_logical_step_ns gating
prediction publication
run_generation reset
warmup
heartbeat
```

可以增加一条低频 startup `logger.info`，明确打印：

```text
control_dt
replan_steps
nominal replan period
```

但不是必要条件；不要增加 per-step logging。

---

# 9. Task D — `run_policy.py` 暴露显式 research knob

文件：

```text
examples/run_policy.py
```

新增 CLI：

```text
--replan-steps N
```

使用现有 `_positive_int` parser helper：

```python
parser.add_argument(
    "--replan-steps",
    type=_positive_int,
    default=None,
    ...
)
```

注意：CLI `default=None` 只是表示“没有 CLI override”，**不是 runtime fallback**。

解析 runtime 时使用现有配置优先级：

```python
runtime = resolve_experiment_config(
    cli_overrides={
        "policy.replan_steps": args.replan_steps,
    }
)
```

`resolve_experiment_config()` 的 dotted override 已经会忽略值为 `None` 的项，所以：

```text
no --replan-steps
    -> defaults.py / YAML 中明确的 Real replan_steps

--replan-steps 2
    -> resolved Real replan_steps = 2
```

**不要**先 inspect Policy 后这样写：

```python
args.replan_steps or info.spec.n_action_steps
```

也不要在 CLI 层读取 `info.spec.n_action_steps`。

### Session metadata

`run_config.yaml` 必须记录**resolved** value：

```yaml
replan_steps: <runtime.policy.replan_steps>
```

不要只记录 raw CLI value，因为 CLI 未传时 raw value 是 `None`，无法复现实验。

Console summary 建议增加一行：

```text
Replan         : 2 steps (125 ms nominal)
```

period 从：

```text
runtime.policy.replan_steps * info.spec.control_dt_s
```

计算即可。

不要额外持久化一堆重复 derived fields。

### 更新入口 docstring

把顶部 usage 中加入 `--replan-steps N`，保持 CLI 文档与实际一致。

---

# 10. Task E — Raw rollout provenance 记录 resolved cadence

文件：

```text
dexmani_real/deployment/lifecycle.py
```

现有 `_rollout_recorder_config(...)` 已记录：

```text
policy selector
checkpoint
inference_steps
seed
max_running_s
```

在同一 provenance 中增加：

```python
"replan_steps": str(runtime.policy.replan_steps)
```

只记录原始实验 knob，不重复记录 `replan_period_ms` 等 derived value。

这项修改不能影响 Recorder transaction、save/discard 或 runtime authority。

---

# 11. 不要修改 Executor / temporal fusion

本任务只改变“什么时候发起新的 observation + inference”。

Executor 当前应继续：

```text
read latest Prediction
    ↓
first_future_step_index(...)
    ↓
skip stale prefix
    ↓
latest valid chunk becomes active
    ↓
decode / IK / safety / publish
```

不要在本任务加入：

```text
chunk blending
cosine transition
temporal ensemble
action smoothing
prediction queue
multiple in-flight models
```

这些若未来研究，应作为独立任务和单变量 ablation。

---

# 12. 不要修改 IPC 或 prediction schema

无需把 `replan_steps` 写入：

```text
Prediction
PREDICTION_DTYPE
RuntimeChannels
shared-memory records
```

`replan_steps` 是 session/runtime configuration，不是每条 prediction 的 payload。

保持：

```text
Prediction.actions.shape == [chunk_size, control_action_dim]
```

---

## 13. Source-level cleanup requirement

本任务的一个核心目的就是删除旧耦合。

完成代码修改后，执行：

```bash
rg "n_action_steps" dexmani_real examples
```

期望：

```text
没有 dexmani_real functional code 依赖 n_action_steps
```

如果当前源码中出现与本任务无关的文字/历史文档匹配，逐项判断；不要为了满足 grep 机械删除有价值内容。

但以下任何 functional pattern 都视为任务未完成：

```text
spec.n_action_steps
policy_spec.n_action_steps
getattr(..., "n_action_steps", ...)
fallback to n_action_steps
compare replan_steps with n_action_steps
```

不要修改相邻 Policy 来使搜索结果消失。

---

## 14. Offline validation

### 14.1 严禁硬件运行

未经用户单独明确授权，不要运行：

```text
examples/run_policy.py
teleoperation
homing
physical replay
robot/camera SDK acquisition
calibration writes
```

不要声称完成了 hardware validation。

### 14.2 必做低成本检查

按仓库 `AGENTS.md`：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

### 14.3 Focused pure-config check

可以使用纯 Python、不会打开硬件的检查确认 config override：

```python
from dexmani_real.config.experiment import resolve_experiment_config

base = resolve_experiment_config()
assert base.policy.replan_steps == 8

override = resolve_experiment_config(
    cli_overrides={"policy.replan_steps": 2}
)
assert override.policy.replan_steps == 2
```

如果仓库默认 YAML/环境使默认值与 `8` 不同，则以当前 resolved source of truth 调整检查；重点是证明：

```text
resolved config owns replan_steps
CLI dotted override works
```

### 14.4 Invalid local value check

用 pure config construction/override 验证 `replan_steps <= 0` 会在 config validation 边界被拒绝。

不要为这一项创建新的 test framework。

### 14.5 Cross-boundary check

通过 source inspection 确认 `validate_policy_runtime_compatibility()` 使用：

```text
replan_steps <= chunk_size
```

且完全不读取 `n_action_steps`。

如果本地已可安全 import `dexmani_policy` metadata 而无需模型/CUDA，可做额外离线检查；不是必须项。缺少相邻 repo/env 时报告 NOT VERIFIED，不要改生产代码绕过。

---

## 15. Behavioral acceptance criteria

代码 review 时逐项确认：

### Ownership

- [ ] `PolicyParams.replan_steps` 是唯一 Real replanning cadence source of truth。
- [ ] `replan_steps` 是普通 positive `int`，不是 optional runtime fallback。
- [ ] Real 不从 `n_action_steps` 推导 cadence。

### Worker

- [ ] `inference_period_ns = replan_steps * step_dt_ns`。
- [ ] deadline catch-up mechanism 原样保留。
- [ ] observation/action logical grid 仍使用 `PolicySpec.control_dt_s`。
- [ ] full prediction chunk 仍完整发布。

### Compatibility

- [ ] `replan_steps >= 1` 在 Real config owner 校验。
- [ ] `replan_steps <= PolicySpec.chunk_size` 在 Policy/Real boundary 校验。
- [ ] 不存在 `replan_steps <= n_action_steps`。

### User-facing experiment control

- [ ] `run_policy.py` 支持 `--replan-steps`。
- [ ] 未提供 CLI 时使用 Real resolved config，而不是 Policy fallback。
- [ ] `run_config.yaml` 记录 resolved `replan_steps`。
- [ ] recorder provenance 记录 resolved `replan_steps`。

### No scope creep

- [ ] 没有修改 IPC/prediction schema。
- [ ] 没有修改 Executor scheduling/safety。
- [ ] 没有 temporal fusion/smoothing。
- [ ] 没有 remote/async broker。
- [ ] 没有修改 `dexmani_policy`。

---

## 16. Suggested implementation order for Codex

按以下顺序工作，减少返工：

```text
1. Read AGENTS.md + git status.
2. Re-open current PolicyParams, compatibility validator, worker, run_policy, lifecycle.
3. Search all dexmani_real n_action_steps references.
4. Add PolicyParams.replan_steps + local validation.
5. Add replan_steps <= chunk_size compatibility check.
6. Replace worker cadence source; do not touch scheduler state machine.
7. Add CLI override + run_config/summary metadata.
8. Add recorder provenance.
9. Re-search n_action_steps to prove old Real coupling is gone.
10. Run offline validation.
11. Inspect focused diff for accidental cleanup/scope expansion.
12. Delete CODEX_TASK.md in the same final change set.
```

---

## 17. Final handoff requirements

Codex 最终回复必须简洁报告：

1. 修改了哪些文件；
2. `n_action_steps` 从 Real runtime 中如何被彻底移除；
3. `replan_steps` 的最终 ownership 和合法范围；
4. 执行了哪些离线检查及结果；
5. 明确说明 **hardware rollout NOT VERIFIED**，除非用户另行授权并实际完成；
6. 若当前 HEAD 与任务书假设不同，说明做了哪些必要适配。

不要在最终 handoff 中把未执行的检查写成通过。
