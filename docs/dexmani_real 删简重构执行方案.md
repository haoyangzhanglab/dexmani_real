# dexmani_real 删简重构执行方案

## 0. 文档目的

本方案用于将 `dexmani_real` 从当前偏 production-style 的复杂 runtime，收敛为适合个人机器人论文研究的实现：

- 控制链更短；
- invariant owner 唯一；
- hot path validation 最少；
- 避免重复 memory copy；
- 异常尽量 fail-fast；
- 真机 physical safety 不削弱；
- causal observation / train-deploy alignment 不削弱；
- 不引入新的 framework；
- 不为了删代码而重写底层 IPC。

本次重构的核心原则：

> **先删除重复 ownership，再删除 mirror state，最后才修改行为语义。**

---

# 1. 固定基线

开始任何修改前执行：

```bash
git status --short
git rev-parse HEAD
```

预期基线：

```text
fb53db8975057a09376decebbb21ebcec2239151
```

如果 HEAD 不同：

1. 不要机械套用本方案；
2. 先检查从 `fb53db8` 到当前 HEAD 的 diff；
3. 重新阅读所有被后续提交修改过的目标文件；
4. 再决定对应步骤是否仍成立。

如果 working tree 存在用户未提交修改：

- 不得 reset；
- 不得 checkout 覆盖；
- 不得删除用户修改；
- 重构必须绕开或兼容已有修改。

参考架构：

```text
dexmani_real baseline:
fb53db8975057a09376decebbb21ebcec2239151

ManiUniCon reference:
85c6f2e32ecf9f2bed62d202b058c39623444686
```

ManiUniCon 只用于参考以下设计原则：

```text
single shared-storage owner
simple lifecycle
local action validation
direct data flow
few global state mirrors
```

不要照搬：

```text
workspace clipping for learned policy
broad exception swallowing
only is_running/error_state as safety state
policy directly driving hardware without dexmani_real fences
```

---

# 2. 最终目标架构

目标 Policy deployment 数据流：

```text
Sensor Workers
      │
      ▼
RuntimeChannels
      │
      ▼
Causal Observation Assembly
      │
      ▼
DexMani Policy Runtime
      │
      ▼
Prediction Ring
      │
      ▼
PolicyExecutor
      │
      ├─ decode action
      ├─ validate representation where needed
      ├─ canonicalize tiny XHand roundoff
      ├─ reject raw policy-domain violation
      ├─ hand slew shaping
      └─ one final SafetyGate
      │
      ▼
Atomic Coupled Command Publication
      │
      ▼
Coupled Command Ring
      │
      ├──────────────────────┐
      ▼                      ▼
Arm Worker              Hand Worker
      │                      │
finite                   finite
hard limits              hard mechanical limits
final jump               final slew
      │                      │
final command authority check
      │                      │
      ▼                      ▼
xArm SDK                XHand SDK
```

保留 inference process 和 executor process 分离。

原因：

```text
Inference Worker:
    potentially slow/blocking GPU model inference

PolicyExecutor:
    real-time chunk scheduling and command publication
```

除非后续 profiling 明确证明：

```text
sync-only
AND
worst-case inference latency << control budget
AND
不需要 inference/control overlap
```

否则不要合并这两个 process。

---

# 3. Invariant Ownership

重构后必须遵守下面的 ownership。

| Invariant | 唯一 Owner |
|---|---|
| PolicySpec 内部结构合法 | `dexmani_policy` |
| Policy 是否能被 Real 当前 runtime 支持 | `deployment/config.py` |
| Observation causal/fresh/aligned | `deployment/worker.py` |
| Policy tensor shape/dtype | `dexmani_policy.LoadedPolicy` |
| Policy action representation decode | `deployment/executor.py` |
| Candidate physical safety | `control/SafetyGate` |
| Publication 时是否仍允许 motion | `runtime/safety.py` |
| SDK 前 command 是否仍 current | arm/hand worker final fence |
| Arm hardware hard limits / final jump | arm worker |
| Hand mechanical hard limits / final slew | hand worker |
| Process liveness | supervisor |
| Sensor data freshness | sensor record / consumer |
| Actuator command acceptance progress | lightweight progress watchdog |
| Dataset 可训练性 | dataset/export consumer |
| Cryptographic artifact integrity | explicit audit tool |

禁止出现：

```text
同一个 invariant 在两个以上 runtime layer 中重新完整验证。
```

---

# 4. 必须永久保留的机制

以下机制不属于“过度设计”，不得因为删简而删除。

## Physical safety

- E-stop；
- sticky hardware/runtime fault；
- `SafetyState`;
- `run_generation`;
- command deadline；
- SDK 前 final authority check；
- target finite check；
- arm hardware joint limits；
- hand mechanical/rated hard limits；
- arm final jump guard；
- hand final slew/rate guard；
- learned-policy workspace safety；
- replay/home collision validation。

## Research correctness

- causal observation；
- source timestamp；
- publish timestamp；
- control-grid alignment；
- cross-modal skew；
- camera generation；
- point-cloud/RGB causal pairing；
- tactile calibration/provenance；
- policy-aligned state；
- submitted/executed action recording。

## Runtime reliability

- worker process death detection；
- slow/blocking inference process isolation；
- actuator acceptance-stall detection；
- clean worker shutdown before SHM unlink。

---

# 5. 明确禁止的重构

本轮不要：

1. 重写 `SharedMemoryRingBuffer`；
2. 用 `multiprocessing.Queue` 全面替换 SHM；
3. 把 RuntimeChannels 拆成一堆新的 channel class；
4. 引入 Hydra/OmegaConf/Pydantic 作为新依赖；
5. 删除 `run_generation`；
6. 删除 deadline；
7. 删除 worker final safety guard；
8. 删除 causal observation logic；
9. 将 learned-policy unsafe action 改为 workspace clipping；
10. 修改 raw dataset schema；
11. 将 `action_id` 与 ring sequence 合并；
12. 合并 inference worker 和 PolicyExecutor；
13. 在同一个 commit 同时重构 runtime 与数据 schema。

---

# 6. Phase Dependency

严格按照：

```text
Phase 0  Baseline tests
   ↓
Phase 1  Policy boundary ownership
   ↓
Phase 2  Observation copy elimination
   ↓
Phase 3  Publication / worker hot-path cleanup
   ↓
Phase 4  Lifecycle / supervisor simplification
   ↓
Phase 5  Command ownership simplification
   ↓
Phase 6  Learned-hand safety path simplification
   ↓
Phase 7  Runtime/config/cleanup simplification
   ↓
Phase 8  Data/replay integrity separation
   ↓
Phase 9  Optional research-scope reductions
```

不要跨 Phase 批量修改。

---

# 7. Phase 0 — 固定 Safety / Behavior Baseline

## 风险

```text
Risk: none
Behavior change: none
```

## 目标

在删除 implementation mechanism 前，先把真正需要保持的 semantic properties 固定成测试。

优先检查和完善：

```text
tests/test_prediction_ipc.py
tests/test_hand_worker_startup.py
tests/test_policy_executor.py
tests/test_runtime_supervision.py
tests/test_policy_lifecycle.py
tests/test_policy_multi_episode.py
tests/test_policy_multimodal_observation.py
tests/test_dexmani_policy_adapter.py
```

## 必须存在的 semantic tests

### Command ownership

```text
old generation command
→ SDK must not be called
```

```text
command read
→ STOP / generation invalidation
→ SDK must not be called
```

```text
old command
→ newer coupled command published
→ old command must not reach SDK
```

### Deadline

```text
command valid during initial validation
→ deadline passes before final SDK fence
→ SDK must not be called
```

### Physical safety

```text
NaN / Inf arm target
→ no SDK
```

```text
NaN / Inf hand target
→ no SDK
```

```text
arm hard joint-limit violation
→ no SDK
```

```text
hand mechanical hard-limit violation
→ no SDK
```

```text
arm final jump violation
→ no SDK
```

```text
hand final slew violation
→ no SDK
```

### Valid path

```text
fresh feedback
+ valid generation
+ valid deadline
+ safe target
→ SDK is called exactly once
```

### Observation

```text
selected frame.source <= frame.publish <= anchor
```

```text
no observation includes frame from future causal cut
```

```text
camera restart generation cannot mix inside one model history
```

```text
stale observation
→ no new physical Policy publication
```

### XHand policy semantic bound

必须增加：

```text
raw policy hand target outside operational limits
+
slew shaping would move it back inside
→ raw target must still be rejected
```

以及：

```text
tiny allowed float32 endpoint roundoff
→ canonicalized to exact operational boundary
```

### Acceptance watchdog

```text
command continuously published
+
worker process alive
+
feedback process alive
+
accepted watermark does not progress
→ watchdog detects failure
```

## 删除 implementation-coupled tests

逐步删除以后只保护如下 implementation 的测试：

```text
must call validator N times
must use active_coupled_command_sequence
must call intermediate ticket check
must expose a particular internal enum
```

测试 target 应是 behavior，而不是 implementation structure。

## Acceptance Criteria

```bash
pytest -q
```

必须在 baseline 上通过。

如果完整测试 suite 本身已有失败：

- 记录 baseline failures；
- 不得把 baseline failure 当成重构引入；
- 后续每个 Phase 不允许增加新 failure。

---

# 8. Phase 1 — Policy / Real Boundary Consolidation

## 风险

```text
Risk: low
Expected behavior change: none
Expected code reduction: high
```

## 修改文件

```text
dexmani_real/deployment/config.py
dexmani_real/integrations/dexmani_policy.py
dexmani_real/deployment/lifecycle.py
dexmani_real/deployment/worker.py

tests/test_dexmani_policy_adapter.py
tests/test_policy_multimodal_observation.py
tests/test_policy_lifecycle.py
```

---

## 8.1 收窄 `validate_policy_runtime_compatibility`

### 删除 Policy-owned invariant

不要在 Real 重复验证：

```text
positive action_dim
positive control_action_dim
positive horizon
positive n_obs_steps
positive n_action_steps
action_dim >= control_action_dim
obs/action window <= horizon
observation fields non-empty
observation field uniqueness
control_dt finite/positive
requires_hand bool type
```

这些属于 `dexmani_policy.PolicySpec`。

### Real 必须继续验证

注意：不要把 `validate_policy_spec()` 整个简单删掉后什么都不保留。

Real 仍然必须检查自己的 capability：

```text
action_key 能否被当前 Real decoder 支持
control_action_dim 是否为当前 decoder 支持的 19 / 21
requested modality 是否属于 Real 当前可产生集合
joint_state raw shape/dtype
contact_force raw shape/dtype
fingertip_points raw shape/dtype
point_cloud [N,6] float32
rgb [H,W,3] uint8
n_action_steps <= Prediction IPC capacity
Policy control_dt == Real control period
point_cloud N == runtime pointcloud N
RGB H/W == runtime camera H/W
当前 deployment 是否满足 hand requirement
```

建议最终 API：

```python
def validate_policy_runtime_compatibility(policy_spec, runtime) -> None:
    ...
```

只保留这一套。

如果 `_validate_observation_fields()` 仍有价值，可以改名：

```python
_validate_real_observation_capability(...)
```

使 ownership 清晰。

---

## 8.2 每次 deployment 只验证一次 compatibility

当前生命周期及 helper 会重复验证。

要求：

```text
run_policy_deployment()
    └── validate_policy_runtime_compatibility()  # exactly once
```

之后：

```text
build_policy_worker_specs()
inference_loop()
PolicyWorkerConfig
```

信任已验证的输入。

删除：

```text
build_policy_worker_specs 内重复 compatibility validation
PolicyWorkerConfig.__post_init__ 内 validate_policy_spec
```

可以保留非常窄的 pickle/config sanity：

```text
experiment non-empty
device non-empty
seed policy
```

但不要再校验 PolicySpec。

---

## 8.3 薄化 `DexManiPolicyRuntime`

目标：

```python
class DexManiPolicyRuntime:
    def __init__(self, loaded_policy, expected_spec):
        if loaded_policy.spec != expected_spec:
            raise RuntimeError("PolicySpec changed between inspect and load")
        self._policy = loaded_policy
        self.spec = expected_spec

    def warmup(self, *, samples):
        return self._policy.warmup(samples=samples)

    def reset_episode(self):
        self._policy.reset_episode()

    def predict(self, observation):
        return self._policy.predict(observation.arrays)

    def close(self):
        ...
```

删除：

```text
hasattr public API reflection checks
_validate_loaded_spec field-by-field reconstruction
_encode_observation shape/dtype validation
_validate_control_action shape/dtype/finite validation
adapter-level rot6d geometry validation
unnecessary output copy
```

### Rot6D owner

对于 `action_ee`：

```text
Policy runtime
→ validates output finite/shape

Real decode
→ rot6d_to_quat_wxyz()
→ validates non-degenerate rotation geometry
```

因此 adapter 不再预验证一次。

---

## Phase 1 Tests

运行：

```bash
pytest -q \
  tests/test_dexmani_policy_adapter.py \
  tests/test_policy_multimodal_observation.py \
  tests/test_policy_lifecycle.py \
  tests/test_policy_executor.py
```

然后：

```bash
pytest -q
```

## Acceptance Criteria

- Real compatibility mismatch 仍 fail-fast；
- Policy internal malformed observation/output 由 Policy runtime 捕获；
- valid Policy output 完全一致；
- `action_ee` degenerate rot6d 仍在 decode 时被拒绝；
- compatibility startup validation 只发生一次；
- 无真机控制语义变化。

## 推荐 commit

```text
refactor(policy): remove duplicate Real-side policy contract validation
```

---

# 9. Phase 2 — Observation Copy Elimination

这是优化后的方案中被前移的阶段。

原因：

> 当前不只是 validation ceremony，而是存在实际 RGB / point-cloud memory copy overhead。

## 风险

```text
Risk: low-medium
Behavior change: none
Expected runtime gain: potentially high for RGB/PCD policies
```

## 原则

这一 Phase：

> **只删除 copy，不修改 causal selection algorithm。**

严禁同时重写：

```text
control-grid selection
camera matching
max skew
freshness
run epoch
pointcloud history selection
RGB/point-cloud alignment
tactile provenance
```

---

## 9.1 明确 ownership-copy 边界

SHM ring reader 返回的数据已经是 process-owned copy。

因此：

```text
SHM read
→ ownership copy
→ process-local objects may reference it directly
```

不要：

```text
SHM read copy
→ Frame copy
→ immutable frame copy
→ PolicyObservation copy
→ Policy copy
```

---

## 9.2 修改 `FrameWindow`

文件：

```text
deployment/observation.py
```

当前：

```python
freeze_array(values)
freeze_array(sequence)
freeze_array(source_ns)
freeze_array(publish_ns)
freeze_array(mask)
```

改为：

```python
values = np.asarray(...)
source_sequence = np.asarray(...)
...
```

第一阶段可以继续保留：

```text
shape
dtype
timestamp relation
valid mask
```

但不要：

```text
np.array(..., copy=True)
flags.writeable = False
```

---

## 9.3 修改 `PointCloudFrame`

Point-cloud payload 已从 ring ownership-copy。

不要再次：

```text
validate
→ np.array(copy=True)
→ readonly
```

保留 payload semantic validation：

```text
[N,6]
float32
finite
RGB range
provenance
```

但返回原 owned array / contiguous view。

---

## 9.4 修改 `RgbFrame`

Camera `read_sequence()` 已经 ownership-copy。

`RgbFrame` 不要再复制。

保留：

```text
uint8
[H,W,3]
provenance
```

---

## 9.5 修改 `PolicyObservation`

`_to_policy_observation()` 已经生成：

```text
np.concatenate
np.stack
np.ascontiguousarray
```

因此 final model arrays 本身已经属于 inference process。

删除：

```text
每个 modality 再 freeze/copy
MappingProxyType
readonly flags
```

可以保留一个普通：

```python
@dataclass
class PolicyObservation:
    observation_id: int
    run_generation: int
    anchor_monotonic_ns: int
    latest_source_monotonic_ns: int
    logical_step_monotonic_ns: int
    arrays: dict[str, np.ndarray]
```

第一阶段仍可保留轻量 metadata consistency validation。

---

## 9.6 不要让最终 arrays readonly

原因：

Policy 最新 runtime 会：

```python
contiguous = np.ascontiguousarray(value)
if not contiguous.flags.writeable:
    contiguous = contiguous.copy()
```

因此如果 Real 主动把 final model tensor 标记 readonly，会逼 Policy 再 copy 一次。

要求 final observation arrays：

```text
C-contiguous
correct dtype
writeable
owned by inference process
```

---

## 9.7 Phase 2 暂时保留 `ObservationBatch` causal assertions

不要在同一个 commit 同时删除 `ObservationBatch.__post_init__` 大量 causal checking。

先获得：

```text
copy reduction
```

再在后续 Phase 考虑 validation ceremony。

---

## Phase 2 Benchmark

增加一个非 CI 强制 benchmark，使用真实 deployment shape：

```text
joint_state
point cloud N=实际配置
RGB = 当前 camera H/W
n_obs_steps = 当前 PolicySpec
```

记录 `_to_policy_observation()`：

```text
median
p95
allocated bytes / obvious copies
```

至少人工比较 before/after。

不要添加硬编码 CI timing threshold。

---

## Phase 2 Tests

重点：

```bash
pytest -q tests/test_policy_multimodal_observation.py
pytest -q tests/test_dexmani_policy_adapter.py
pytest -q
```

必须验证：

```text
final tensor numerical equality
dtype equality
shape equality
causal provenance equality
RGB byte equality
pointcloud equality
```

## Acceptance Criteria

对于大数组：

```text
SHM payload ownership-copy
→ no container-level full payload recopy
→ one final stack/concat for model batch
→ no Real readonly-induced Policy recopy
```

## 推荐 commit

```text
perf(observation): remove redundant process-local array copies
```

---

# 10. Phase 3 — Publication / Worker Hot Path Cleanup

## 风险

```text
Risk: medium
Behavior change: intended none
```

---

# 10.1 删除 physical publication 前的 non-atomic duplicate precheck

文件：

```text
control/publication.py
runtime/safety.py
```

真实 publication 的唯一 authority 必须是：

```text
motion_lock
→ is_running
→ e-stop
→ error/fault
→ SafetyState
→ generation
→ deadline
→ ring.write
```

因此 physical publish path：

```python
publish_command(...)
```

不要先调用：

```python
_command_publishability_reason(...)
```

再调用 atomic publisher。

直接调用：

```python
publish_coupled_command_if_motion_permitted(...)
```

### Shadow mode 例外

对于：

```text
execute=False
```

允许继续使用 non-mutating publishability query。

---

# 10.2 删除 `read_hand_feedback` 重复 shape/finite

如果：

```python
diagnose_hand_feedback(...)
```

已经检查 qpos shape/finite：

删除后面的：

```python
if qpos.shape != ... or not np.all(np.isfinite(qpos)):
```

同类重复检查一起搜索清理。

---

# 10.3 Worker validator 改为 target-level final guard

当前 worker validator 不应该再次完整验证 producer-built record。

最终 Arm hot path只关心：

```text
target finite
target inside arm mechanical limits
target jump <= configured max
```

命令是否仍 current：

```text
交给 final authority fence
```

deadline：

```text
也由 final authority fence重新检查
```

static config：

```text
ArmParams.__post_init__
```

负责。

### Arm worker target helper

推荐变成类似：

```python
def validate_arm_target(
    target,
    previous_target,
    *,
    lower,
    upper,
    max_jump,
) -> str | None:
    ...
```

不要继续接收整个 structured command 再重新证明 IPC schema。

---

# 10.4 Hand worker只保留 mechanical hardware guard

Hand producer 的 operational bounds 属于 control semantics。

Hand worker hardware boundary 只需：

```text
finite
mechanical/rated hard limit
final slew/rate
```

不要每 tick 重新运行：

```text
rated ⊇ mechanical ⊇ operational
```

这个 nesting 在 `HandParams` startup 已经验证。

---

# 10.5 Worker authority check 收敛成一次 final fence

目标：

```text
read command
   ↓
extract target
   ↓
target finite / hard limit
   ↓
jump or slew shaping/check
   ↓
FINAL AUTHORITY CHECK
   ↓
SDK
```

不要：

```text
authority
validation
authority
shape
authority
bounds
authority
SDK
```

### Final fence 必须包含

```text
motion still allowed
run_generation matches
command still latest
deadline not expired
runtime not stopped
no sticky fault
no e-stop
```

### 注意

final fence 必须发生在所有 potentially expensive validation/shaping 之后。

这样：

```text
deadline crosses during validation
```

也会在 SDK 前被拒绝。

---

## Phase 3 Tests

必须特别覆盖 race semantics：

```text
command valid initially
→ validation
→ deadline crossed
→ final fence rejects
```

```text
command read
→ new command published
→ old final fence rejects
```

```text
command read
→ STOP
→ old final fence rejects
```

运行：

```bash
pytest -q \
  tests/test_prediction_ipc.py \
  tests/test_hand_worker_startup.py \
  tests/test_policy_multi_episode.py

pytest -q
```

## 推荐 commits

```text
refactor(publication): remove duplicate pre-publication authority checks
```

```text
refactor(workers): reduce command validation to hardware-boundary guards
```

---

# 11. Phase 4 — Lifecycle / Supervisor Simplification

## 风险

```text
Risk: medium
```

## 目标

学习 ManiUniCon 的一个核心优点：

> lifecycle 应该能从上到下快速读懂。

但保留 dexmani_real 更强的 safety/freshness semantics。

---

# 11.1 删除 `proc_names`

不要同时维护：

```text
procs
proc_names
```

process 本身已经：

```python
process.name
```

因此：

```python
run_supervisor(shared, procs, ...)
```

直接 derive names。

---

# 11.2 删除 parallel `heartbeat_names`

不要让 caller 同时维护：

```text
WorkerSpec list
process list
process_names
heartbeat_names
heartbeat_timeout mapping
```

建议：

```python
run_supervisor(
    shared,
    processes,
    heartbeat_timeouts_s,
)
```

监控：

```text
process.name in heartbeat_timeouts_s
```

即可。

如果某 worker 不需要 heartbeat：

```text
不要给它 timeout entry
```

---

# 11.3 `WorkerSpec` 保持简单

不要为了修 parallel list 再把 WorkerSpec 变成复杂 framework。

保留：

```python
@dataclass(frozen=True)
class WorkerSpec:
    name: str
    target: Callable
    args: tuple
    ready_name: str | None = None
    daemon: bool = False
```

不要急着加入：

```text
十几个 health policy fields
```

readiness timeout 可以继续由 runtime config mapping 提供。

---

# 11.4 合并 readiness API

删除：

```text
RuntimeChannels.wait_ready()
```

保留：

```text
runtime.supervisor.wait_subsystem_ready()
```

因为后者还能检查：

```text
error_state
process liveness
timeout
```

进一步可以让其直接接收：

```text
WorkerSpec + Process pairs
```

避免 caller 自己重建 `(name, timeout)`。

---

# 11.5 删除 inference ready 后立即 `is_alive()` 二次检查

保留：

```text
inference first
→ wait inference ready
→ then start hardware
```

这是好的 staging。

但：

```text
wait_ready returned
→ immediately is_alive again
```

没有真正消除 race。

只要后续 unified readiness 在 DISARMED 状态下继续检查所有 started process，并且 ARMED 之前 process 均健康，就足够。

---

# 11.6 Process liveness 与 source freshness 分离

不要把 heartbeat 同时解释为：

```text
process is alive
AND
sensor data is fresh
```

分别定义：

## Liveness

```text
process alive / heartbeat
```

适用于：

```text
inference
policy executor
recorder
```

## Data freshness

```text
latest source timestamp
```

适用于：

```text
camera
arm
hand
VR
pointcloud
```

消费数据的 control/inference path 使用 source timestamp 决定数据是否可用。

---

# 11.7 修复 Camera false-healthy degraded mode

当前 camera read failure 不应继续让系统无限表现为健康。

改为 bounded failure：

```python
consecutive_read_failures += 1

if successful_read:
    consecutive_read_failures = 0
```

如果：

```text
failure duration / count exceeds configured threshold
```

则：

```text
raise
→ worker exits
→ supervisor faults
```

不要在失败 read 上更新代表“数据健康”的 timestamp。

### 阈值

不要凭空发明新的 magic number。

优先：

1. 查看已有 camera freshness/read timeout config；
2. 用现有 timeout 推导；
3. 如果必须新增参数，只新增一个明确参数；
4. 用实际 hardware log 调整。

---

# 11.8 XHand stale state bounded failure

允许：

```text
single transient get_state failure
```

但不能无限：

```text
reuse last qpos
+ qpos_stale
+ heartbeat forever
```

实现 bounded failure。

策略同 camera：

```text
short transient failure
→ degraded/no valid observation

sustained failure
→ worker fault / episode stop
```

---

## Phase 4 Tests

运行：

```bash
pytest -q \
  tests/test_runtime_supervision.py \
  tests/test_policy_lifecycle.py \
  tests/test_policy_multi_episode.py

pytest -q
```

新增：

```text
ready worker dies before ARMED
→ deployment fails

heartbeat worker stalls
→ supervisor fails

camera sustained failure
→ does not remain indefinitely healthy

hand sustained feedback failure
→ does not remain indefinitely healthy
```

## 推荐 commit

```text
refactor(runtime): simplify worker readiness and supervision
```

---

# 12. Phase 5 — Command Ownership Simplification

## 风险

```text
Risk: medium-high
Requires hardware regression: yes
```

这是第一次真正删除 runtime mirror ownership state。

---

# 12.1 当前需要收敛的身份

现状同时存在：

```text
run_generation
ring sequence
active_coupled_command_sequence
deadline
action_id
```

最终用途应拆开：

```text
run_generation
    = run / episode ownership

ring sequence
    = latest command ownership

deadline
    = command lifetime

action_id
    = audit / acceptance watermark
```

因此：

```text
active_coupled_command_sequence
```

是 ring latest sequence 的 mirror state。

---

# 12.2 删除 `active_coupled_command_sequence`

删除：

```text
RuntimeChannels.active_coupled_command_sequence

_clear_coupled_command_locked()

publication 中 set active sequence

invalidate 时 clear active sequence

相关 tests / docs
```

---

# 12.3 新 current-ticket predicate

目标：

```python
def _ticket_is_current_locked(shared, ticket):
    permit = _read_motion_permit_locked(shared)
    return (
        permit.allows_motion
        and permit.run_generation == ticket.run_generation
        and shared.coupled_cmd_ring.latest_sequence == ticket.ring_sequence
    )
```

最终 execution fence再加：

```text
is_running
!error_state
!estop
now < valid_until
```

---

# 12.4 Invalidation 只需要 bump generation

```python
def _invalidate_coupled_commands_locked(shared):
    return _advance_run_generation_locked(shared)
```

旧 ring record 无需擦除。

因为：

```text
record.generation != current generation
```

已经失效。

---

# 12.5 Cancel current command

逻辑：

```text
ticket sequence == ring latest
AND
generation current
→ bump generation
```

如果已经有 newer command：

```text
cancel old ticket
→ must not revoke newer command
```

保留该 test。

---

# 12.6 暂时保留 `action_id`

不要在这个 Phase 同时：

```text
action_id → ring sequence
```

因为 action_id 当前仍用于：

```text
recording
arm acceptance
hand acceptance
executor progress
tests
```

把 identity simplification 与 schema migration 分开。

---

# 12.7 保留 ticket `published_monotonic_ns`

不要因为删除 active sequence 而误删：

```text
CoupledCommandTicket.published_monotonic_ns
```

executor 仍用它初始化 acceptance progress timing。

---

## Phase 5 Tests

核心：

```text
newer record supersedes old ticket
```

```text
generation bump invalidates current record
```

```text
cancel old ticket does not revoke newer command
```

```text
deadline still enforced
```

```text
STOP race still blocks SDK
```

```bash
pytest -q \
  tests/test_prediction_ipc.py \
  tests/test_policy_multi_episode.py \
  tests/test_hand_worker_startup.py \
  tests/test_policy_executor.py
pytest -q
```

## 真机回归

必须测试：

```text
rapid B/S
rapid repeated S
e-stop during command stream
high-rate policy publication
new prediction superseding previous endpoint
expired command
```

## 推荐 commit

```text
refactor(commands): derive latest command ownership from ring sequence
```

---

# 13. Phase 5B — Acceptance Watchdog Simplification

不要删除 command progress capability。

XHand command may be：

```text
ACCEPTED
CRC_UNCONFIRMED
REJECTED
```

因此：

```text
worker process alive
```

并不能证明：

```text
command path works
```

---

## 13.1 目标状态

保留最少：

```python
latest_published_action_id
latest_publish_ns

arm_accepted_action_id
arm_progress_ns

hand_accepted_action_id
hand_progress_ns
```

每个 run reset。

---

## 13.2 语义

```text
if no command outstanding:
    healthy

if accepted watermark advances:
    update progress timestamp

if accepted < latest_published
and no progress for timeout:
    fault
```

latest-wins 允许：

```text
1 → 4
```

直接跳过中间 action ID。

不需要：

```text
1,2,3,4 每个都有 ACK
```

---

## 13.3 保留 final truncation acceptance

如果 `max_action_steps` 的语义要求：

```text
final published action
must have reached both workers
before TRUNCATED
```

则继续检查：

```text
arm_accepted >= final_action_id
hand_accepted >= final_action_id
```

---

## 推荐 commit

```text
refactor(executor): simplify actuator acceptance progress watchdog
```

---

# 14. Phase 6 — Learned Hand Safety Path Simplification

## 风险

```text
Risk: medium-high
Hardware regression required: yes
```

目标是把：

```text
gate
→ shape
→ partial gate
→ bounds
```

改成逻辑清晰的：

```text
raw semantic check
→ shaping
→ one final physical gate
```

---

# 14.1 正确顺序

对于 learned Policy：

```text
raw hand output
    ↓
tiny float32 roundoff canonicalization
    ↓
RAW operational envelope validation
    ↓
hand slew shaping
    ↓
ONE final SafetyGate
    ↓
atomic publication
```

注意：

> raw operational validation 必须在 shaping 前。

否则：

```text
Policy outputs illegal target
→ slew shaping happens to pull it inside bound
```

会静默掩盖 model-domain violation。

---

# 14.2 简化 `canonicalize_policy_hand_endpoint_roundoff`

保留：

```text
shape
finite
strict mechanical violation rejection
operational tolerance
clip only <= allowed float32 tolerance
```

删除每个 tick 重复：

```text
validate_hand_limit_nesting(...)
```

因为 `HandParams.__post_init__` 已经保证：

```text
rated ⊇ mechanical ⊇ operational
```

---

# 14.3 拆出 target-only operational check

例如：

```python
def hand_target_within_operational_bounds(
    target,
    lower,
    upper,
) -> bool:
    ...
```

不要为了验证一个 target 又重新验证所有 static limit arrays。

---

# 14.4 Shaping

```python
shaped = limit_hand_target_delta(
    raw_target,
    measured_hand_qpos,
    max_delta_per_tick,
)
```

如果 `limit_hand_target_delta` 继续在每 tick 验证 static `max_delta`：

可以在后续把：

```text
max delta positive/finite
```

移动 startup。

但本 Phase 不需要同时过度重构。

---

# 14.5 Final SafetyGate exactly once

在 final shaped candidate 上：

```python
gate.validate(...)
```

然后删除：

```text
SafetyGate.validate_shaped_hand()
```

及对应测试。

---

# 14.6 Worker 保持独立 hardware guard

即使 producer SafetyGate 已验证 hand operational limits：

hand worker 仍检查：

```text
finite
mechanical/rated hard limit
final actuator slew
final authority
```

这是故障隔离边界，不能删除。

---

## Phase 6 Tests

新增/保留：

```text
raw target slightly beyond tolerance
→ rejected
```

```text
raw target tiny float32 roundoff
→ canonicalized
```

```text
raw target illegal but shaped target legal
→ still rejected
```

```text
valid raw target
→ shaped
→ final gate called once
```

```text
shaped target hard-mechanically invalid
→ rejected
```

## 推荐 commit

```text
refactor(hand): separate policy-domain bounds from final physical safety
```

---

# 15. Phase 7 — Runtime / Config / Cleanup Simplification

这一阶段主要删除 production-style ceremony。

---

# 15.1 固定值不要做 configurable-but-forbidden

例如：

```text
prediction_ring_maxlen = 1
然后 validator 又要求必须 == 1
```

改成 module constant：

```python
PREDICTION_RING_MAXLEN = 1
```

类似：

```text
initial_safety_state = 0
```

如果永远必须 DISARMED：

不要允许 caller 配置然后再验证。

直接创建：

```python
safety_state = ctx.Value("i", int(SafetyState.DISARMED))
```

注意避免 IPC 层循环 import；必要时定义稳定 wire constant。

---

# 15.2 不优先条件化创建所有 rings

虽然 RuntimeChannels 当前会创建大量 ring，但不要为了“看起来简洁”把：

```text
camera_ring optional
pointcloud_ring optional
record rings optional
...
```

全部变成 Optional attributes。

这很可能：

```text
少一些 SHM
换来几十处 Optional branching
```

除非 profiling 明确显示 memory/startup 是实际问题，否则保持一个 central data plane 更简单。

---

# 15.3 Config validation owner

`ArmParams/HandParams/...` 已经拥有自己的 invariant。

因此 runtime config resolver 不要重复：

```text
joint limit order
workspace order
positive loop_hz
hand nesting
```

resolver 只负责：

```text
YAML parsing
override
dataclass construction
genuine cross-section compatibility
```

---

# 15.4 不引入 Hydra

目标：

```text
PyYAML
→ narrow override merge
→ dataclass constructors
→ cross-section validation
```

不是：

```text
delete custom config
→ add Hydra/OmegaConf dependency
```

---

# 15.5 简化 RuntimeChannels.close

保留最重要 invariant：

> **只有确认所有 child process 都停止后，parent 才 unlink SHM。**

可以保留：

```text
join
→ terminate
→ kill
```

这是合理 shutdown robustness。

删除/简化：

```text
_close_completed_operations ledger
expected operation sets
multi-round operation bookkeeping
allocation rollback retry transaction semantics
```

可以使用普通：

```python
errors = []

for ring in rings:
    try:
        ring.close()
    except Exception:
        errors.append(...)

    try:
        ring.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        errors.append(...)
```

---

# 15.6 Cleanup failure 不等于 Physical FAULT

区分：

## Motion runtime failure

```text
worker died during operation
hardware failure
e-stop
cannot stop live child process safely
```

可以进入：

```text
SafetyState.FAULT
```

## Robot 已停止之后

```text
SHM unlink error
resource_tracker issue
log file cleanup failure
```

是：

```text
process/resource cleanup error
```

可以返回 non-zero / log error。

但不要重新赋予 physical FAULT 语义。

---

## 推荐 commits

```text
refactor(config): remove redundant runtime invariant validation
```

```text
refactor(ipc): simplify RuntimeChannels fixed configuration and cleanup
```

---

# 16. Phase 8 — Data / Replay Integrity Separation

不要在前面 runtime refactor 尚未稳定时做。

---

# 16.1 EpisodeReader hash 改 explicit audit

普通：

```python
EpisodeReader(path)
```

不应该默认重新 SHA256 大文件。

改为：

```python
EpisodeReader(path, verify_hash=False)
```

普通 read 验证：

```text
required files exist
schema version
dataset lengths
required shape
basic semantic consistency
```

完整：

```text
SHA256
sidecar integrity
full video frame count
artifact attestation
```

进入：

```text
explicit audit
```

---

# 16.2 提供 audit API / CLI

例如：

```bash
python -m dexmani_real.tools.audit_episode EPISODE
```

或者现有工具扩展：

```text
verify_hash=True
```

目标是：

```text
integrity capability 保留
但不污染每次普通读取
```

---

# 16.3 Processed writer full validation optional

当前：

```text
write
→ reopen
→ full validate
→ publish
```

然后 exporter：

```text
→ full validate again
```

改为：

```text
writer:
    write
    structural sanity
    atomic publish
```

完整自检：

```text
--verify-output
```

consumer/export boundary 继续严格验证：

```text
shape
dtype
finite
alignment
semantic attributes
```

---

# 16.4 Replay physical safety 不削弱

Replay 必须继续：

```text
live start state
joint hard limits
start → first target transition
workspace
collision
adjacent trajectory transitions
```

这些不能因为 hash audit 分离而删除。

---

# 16.5 Artifact hash mismatch 改 reproducibility warning

例如：

```text
config SHA mismatch
URDF SHA mismatch
SRDF SHA mismatch
raw data SHA mismatch
```

应优先表达：

```text
reproducibility / provenance mismatch
```

如果当前 geometry 和 physical preflight 已重新验证完整 trajectory：

默认可以 warning/audit，而不是把 cryptographic equality 当 physical safety predicate。

如果某 hash 变化意味着当前 physical checker根本无法解释 trajectory，则例外保留 hard reject。

---

## 推荐 commit

```text
refactor(data): separate runtime data validation from artifact integrity audit
```

---

# 17. Phase 9 — 暂缓的高风险删简

以下项只有研究 scope 明确后再做。

---

## 17.1 Raw schema vNext

当前 schema 同时承载：

```text
training data
runtime tracing
IPC debugging
hardware telemetry
```

长期建议拆：

### Canonical dataset

```text
timestamps
arm/hand state
actual/submitted actions
RGB/depth
tactile
task/success
policy-aligned state
important source timestamps
valid masks
```

### Debug trace

```text
ring sequence
internal action lifecycle timestamps
SDK ACK watermarks
camera backlog
worker counters
per-stage latency
board diagnostics
```

但这是 schema migration。

不要和 runtime simplification 同时做。

---

## 17.2 Sync / Async

如果论文最终确认：

```text
只研究 sync policy execution
```

则删除 async 可以产生明显收益。

但在研究 scope 明确前：

```text
KEEP
```

---

## 17.3 `action_id → ring_sequence`

理论上可以继续减少 identity。

但目前 action_id 被：

```text
recording
worker acceptance
executor progress
tests
```

广泛使用。

不是当前优先级。

---

## 17.4 IPC rewrite

不做。

除非 profiling 明确显示：

```text
current ring complexity造成 measurable bug/performance problem
```

否则收益不足以覆盖风险。

---

## 17.5 Merge inference + executor

默认不做。

只有在：

```text
sync-only
+
profiling证明 inference worst-case latency满足 control
+
无需 GPU/control overlap
```

才重新评估。

---

# 18. 最终 Supervisor 模型

目标 supervisor 只回答：

```text
e-stop?
sticky fault?
worker died?
required heartbeat stalled?
operator quit?
```

Data consumers 自己回答：

```text
sensor data stale?
```

PolicyExecutor 自己回答：

```text
actuator acceptance stalled?
```

不要让 supervisor 发展成整个系统业务状态中心。

---

# 19. 最终 Command Authority 模型

重构完成后，command authority 应只有：

```text
Motion Permission
    SafetyState
    is_running
    error_state
    estop

Run Identity
    run_generation

Latest Ownership
    coupled_cmd_ring.latest_sequence

Lifetime
    valid_until_monotonic_ns
```

`action_id`：

```text
只用于 audit / acceptance
```

不要参与 motion ownership。

---

# 20. 最终 Policy Safety 模型

Policy raw output：

```text
Policy output
    │
    ├─ representation decode validity
    ├─ policy-domain operational validity
    │
    ▼
shaping
    │
    ▼
SafetyGate
    │
    ├─ operational joint bounds
    ├─ arm delta
    ├─ workspace
    └─ optional collision
    │
    ▼
atomic publication
```

Worker：

```text
command
    │
    ├─ finite
    ├─ mechanical hard limits
    ├─ final discontinuity/slew
    ├─ latest/current/deadline
    ▼
SDK
```

---

# 21. Efficiency Targets

重构后应满足以下可观测目标。

## Policy startup

```text
Real Policy compatibility validation:
1 time per deployment
```

---

## Observation

Large payload：

```text
SHM ownership copy
→ at most one final model-batch stack/concat
→ Policy tensor conversion
```

不要再出现多个完整 RGB/PCD copy。

---

## Action preparation

一条 learned action：

```text
raw semantic check
→ optional shaping
→ exactly one full SafetyGate
```

---

## Publication

真实 publish：

```text
exactly one atomic motion-permission check
```

---

## Worker

每次 SDK command：

```text
target physical guard
→ one final authority fence
→ SDK
```

不要多次反复获取同一个 ticket permission。

---

## Lifecycle

不再有：

```text
processes
process_names
heartbeat_names
```

三份平行 source of truth。

---

## Command ownership

不再有：

```text
active_coupled_command_sequence
```

mirror。

---

# 22. Codex 修改规则

Codex 每执行一个 Phase，必须：

### Before editing

```bash
git status --short
git rev-parse HEAD
```

然后 repo-wide 搜索所有要删除 symbol：

```bash
rg "SYMBOL_NAME" .
```

检查：

```text
production callsites
tests
docs
examples
```

---

### During editing

必须：

```text
prefer deletion over adding abstraction
prefer local function over new class
prefer direct data flow over wrapper
do not introduce new dependency
do not change unrelated formatting
do not modify dataset schema in runtime phases
```

如果为了删除 100 行旧逻辑新增了：

```text
3 new class
2 factory
1 protocol layer
```

停止并重新设计。

---

### After editing

运行 targeted tests。

然后：

```bash
pytest -q
```

并：

```bash
git diff --stat
git diff
```

人工检查：

```text
是否真的减少状态/分支
是否引入新 duplicate owner
是否无意改变 safety semantics
```

---

# 23. 每个 Commit 的要求

每个 commit 只回答一个问题。

推荐顺序：

```text
1. refactor(policy):
   remove duplicate Real-side policy validation

2. perf(observation):
   remove redundant process-local copies

3. refactor(publication):
   collapse duplicate publication permission checks

4. refactor(workers):
   reduce worker command validation to final hardware guards

5. refactor(runtime):
   simplify readiness and supervision

6. refactor(commands):
   remove active command sequence mirror

7. refactor(executor):
   simplify actuator acceptance watchdog

8. refactor(hand):
   simplify learned-hand safety pipeline

9. refactor(config):
   remove duplicate static invariant validation

10. refactor(ipc):
    simplify channel cleanup bookkeeping

11. refactor(data):
    move cryptographic integrity checks to audit path
```

不要 squash 成一个巨型 refactor 才测试。

---

# 24. Hardware Regression Gate

从 Phase 3 开始，涉及 physical control 的 Phase 合并前必须执行真机 regression。

至少：

```text
1. arm-only safe target
2. coupled arm+hand safe target
3. STOP while streaming
4. repeated B/S
5. e-stop while streaming
6. command supersede
7. intentionally expired command
8. invalid joint-limit target
9. arm jump reject
10. hand slew limiting
11. XHand ACCEPTED
12. XHand CRC_UNCONFIRMED
13. XHand REJECTED
14. camera transient failure
15. camera sustained failure
16. hand transient read failure
17. hand sustained read failure
```

每项记录：

```text
expected
observed
pass/fail
```

---

# 25. Stop Conditions

Codex 在以下情况下必须停止当前 Phase，不继续大范围修改：

```text
baseline tests unexpectedly fail

发现 current HEAD 已修改核心语义

发现要删除的状态还有未知 production writer

发现删除 validation 会移除唯一 physical safety guard

发现 causal observation parity 无法证明

发现需要 dataset schema migration 才能完成

发现需要新增第三方 dependency

发现无法确定 XHand vendor behavior

发现 hardware behavior change 无法通过 offline tests 判断
```

这时应输出：

```text
what was discovered
why the planned change is unsafe
smallest next investigation needed
```

而不是猜测实现。

---

# 26. 优先级总结

## 第一批：立即执行

```text
Phase 0
Phase 1
Phase 2
```

原因：

```text
高收益
低风险
不改变 hardware semantics
直接降低代码和 inference overhead
```

---

## 第二批：Runtime 主干

```text
Phase 3
Phase 4
```

目标：

```text
publication authority single-owner
worker hot path minimal
lifecycle readable
```

---

## 第三批：状态模型

```text
Phase 5
Phase 5B
Phase 6
```

这是控制系统核心 simplification。

必须逐 commit + 真机 regression。

---

## 第四批：外围复杂度

```text
Phase 7
Phase 8
```

让 config、cleanup、data provenance 回到 research-code 合理复杂度。

---

## 暂缓

```text
raw schema migration
sync/async removal
action_id removal
IPC rewrite
inference/executor merge
```

---

# 27. 最终完成标准

本轮删简完成后，仓库应满足：

### Architecture

```text
一个 invariant 一个 owner
```

### Policy

```text
PolicySpec correctness only in dexmani_policy
Real only checks Real compatibility
```

### Observation

```text
causal semantics unchanged
large-array copies significantly reduced
```

### Control

```text
one semantic/raw validation
one final SafetyGate
one atomic publication
one final SDK authority fence
```

### Runtime

```text
SafetyState + generation + ring sequence + deadline
```

足以描述 command authority。

### Health

```text
process liveness
sensor freshness
actuator progress
```

三者明确分离。

### Dataset

```text
research correctness strict
cryptographic audit optional
runtime telemetry不再无限扩张
```

### Readability

新研究者应能够沿以下路径理解 Policy deployment：

```text
deployment/lifecycle.py
→ deployment/worker.py
→ dexmani_policy public runtime
→ deployment/executor.py
→ control/publication.py
→ runtime/safety.py
→ robot/{arm,hand}_worker.py
→ SDK
```

而不需要先理解大量重复 contract 和 mirror state。

---

# 28. 最终设计判断

本次重构不是把 `dexmani_real` 变成 ManiUniCon。

目标是保留 `dexmani_real` 真正比 ManiUniCon 更需要的能力：

```text
dexterous arm+hand coupled control
causal multimodal observations
real-time policy scheduling
generation fencing
deadline
replay/home physical safety
research-grade data alignment
```

同时吸收 ManiUniCon 最有价值的工程原则：

```text
short data flow
local ownership
simple lifecycle
few shared state variables
validation concentrated at boundaries
```

最终原则：

> **复杂度只为真实的 physical safety、timing correctness 和 research correctness 服务；不为“防止内部代码写错”构建第二套 runtime framework。**