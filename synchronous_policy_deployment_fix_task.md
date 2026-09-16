# Task: 修正并锁定 Synchronous Policy Deployment 时序语义

> **性质：** one-off implementation task。用于完成本次同步 Policy deployment 修正，不是永久架构文档。
>
> **位置：** 按用户要求临时放在 repository root。实现合并、行为稳定且 README/source 已反映最终语义后，应删除或归档本文件。
>
> **安全：** 本任务仅允许 source inspection 与 offline validation。未经用户单独明确授权，不得运行 policy rollout、teleoperation、replay、homing、calibration、device discovery、camera acquisition，或任何会打开真实 robot/camera SDK 的命令。
>
> **Reviewed baseline:** `dexmani_real` main at `ad5511c4d4792dcd32b08d8f6c9ac73f9a18ac93`。若实现时 HEAD 已变化，以当前 source/config/schema 为事实来源，先重新检查 producer → transformation → consumer → side effect 链路，不要机械套用本文 patch。

---

## 0. 目标

将 learned-policy 实机执行语义收敛为一个简单、可证明、与 `dexmani_policy` 仿真 contract 一致的 **synchronous re-anchor baseline**：

```text
actual command publication
        ↓
wait one PolicySpec.control_dt_s
        ↓
fresh causal observation
        ↓
blocking policy inference
        ↓
chunk[0] immediately dispatchable
        ↓
actual publication becomes next cadence anchor
```

最终系统必须满足：

1. `observation_t → control_action[0]_t` 的训练/仿真/实机时间语义一致。
2. action cadence 只由 **actual publication time** 决定，不由 executor poll grid、episode grid 或 recording grid 决定。
3. blocking inference 可以造成 chunk-boundary pause；这是 synchronous baseline 的真实代价，不得通过隐式 action phase shift 隐藏。
4. late inference 不 catch-up、不 burst、不补发。
5. stale / unavailable feedback 或 stale first prediction 只触发 whole-chunk replan；连续不可恢复的 policy-control failure 才升级为 session failure。
6. policy/model/recording failure 与 physical/control fault 严格分离。
7. 保留当前已经正确的 single-feedback transaction、SafetyGate、generation fence、worker final SDK fence 和 truthful recording timestamps。

---

## 1. 设计依据

### 1.1 `dexmani_policy` contract

当前 public runtime 返回：

```text
[n_action_steps, control_action_dim]
```

其 action window 从 `n_obs_steps - 1` 开始，因此最新 observation 对应当前要执行的第一条 control action，而不是下一个额外延迟 step。

仿真评测也遵循：

```text
obs_t
→ predict action_chunk
→ env.step(action_chunk[0])
→ obs_{t+1}
```

因此 Real runtime 必须保持：

```text
obs_t → action_t
```

不能变成：

```text
obs_t → wait one dt → execute action_t
```

### 1.2 LeRobot synchronous lesson

参考当前 LeRobot：

```text
src/lerobot/rollout/inference/sync.py
src/lerobot/rollout/strategies/core.py
src/lerobot/utils/cycle_timer.py
```

可借鉴原则：

- fresh observation 进入 synchronous `select_action()`；
- queue refill 后第一条 action 在该 control iteration 中执行；
- chunk cache 可以存在，但不会把新 chunk 的 action[0] 人为平移一个 control step；
- loop overrun 可以被记录，但不做 catch-up burst。

不要复制 LeRobot 与本项目无关的 strategy / interpolation / text-query complexity。

### 1.3 ManiUniCon lesson

参考：

```text
maniunicon/policies/torch_model.py
maniunicon/core/robot.py
maniunicon/customize/act_wrapper/chunk_wrapper.py
```

ManiUniCon synchronized mode：

```text
chunk finished
→ fresh state/camera
→ blocking inference
→ new chunk re-anchored to current time
→ first action offset = 0
```

ManiUniCon asynchronous mode 才使用 future timestamps + stale-prefix pruning。

DexMani 本任务选择 **synchronized semantics**，不实现 stale-prefix skip / RTC / async scheduler。

---

## 2. 当前必须修正的问题

### Bug A — chunk-boundary observation 提前一个 control phase

当前 `ad5511c4` 在上一 chunk 最后一条 command publish 后，会通过特殊 `continue` 立即进入下一轮 observation + inference；但新 chunk action[0] 仍受：

```text
last_publication_ns + step_dt_ns
```

约束。

结果是：

```text
publish old[-1]
→ immediately observe
→ infer new chunk
→ wait until old_publish + dt
→ execute new[0]
```

这把 `obs_t → action_t` 改成了近似 `obs_t → action_t executed at t+1`，属于 temporal phase bug。

### Bug B — stale model-input invalidation 与 feedback invalidation continuity 不一致

当前 feedback continuity break 走 `_invalidate_chunk()`，会清：

```text
actions
previous_arm_command_qpos
```

但 stale model input 路径只 `actions.clear()`。

长 inference / stall 后继续保留旧 `previous_arm_command_qpos` 会让新 chunk 的 projection continuity reference 与 fresh measured state 脱节。

### Bug C — repeated stale prediction 可能形成无限 GPU replan loop

如果 prediction 每次 inference 后都超过 `max_input_age_s`，runtime 可能反复：

```text
fresh obs
→ expensive inference
→ stale before first publish
→ discard
→ immediate replan
→ ...
```

必须限制 **连续“完成 inference 后、chunk[0] 尚未 publish 就因 stale 被丢弃”** 的次数。

注意：普通 observation unavailable、camera 尚未出新帧、还没到 deadline 的 poll iteration 不得计入该 counter。

### Bug D — policy/model exception 不应退化为 ambiguous critical worker death

以下属于 experiment/session failure，不等于 physical robot fault：

```text
CUDA OOM
model forward exception
prediction shape mismatch
NaN/Inf prediction
policy runtime contract violation
```

应在 PolicyRunner 边界显式捕获 `Exception`，安全 fence 当前 rollout，设置 `session_failed + quit_requested`，再由 verified shutdown 收尾。

不得 catch `BaseException`。

### Risk E — observation 的整体 timing invariant 需要一个明确出口检查

各 modality selector 已做 causal / lag / skew 局部约束，但最终 assembled observation 应在 owner boundary 明确确认：

```text
latest source age <= max_input_age_s
cross-modal latest-source skew <= max_observation_skew_s
```

不要依赖若干局部 threshold 的数值关系“间接保证”。

---

## 3. 明确非目标

本任务 **禁止** 扩展为：

```text
async/background inference
RTC / real-time chunking
future-action timestamp scheduling
stale-prefix skipping
action replacement / overlapping prediction
robot_ready / policy_ready handshake
per-action ACK barrier
physical convergence wait
new policy thread/process
new action queue process
new persisted rollout schema
new freshness config knob
new telemetry framework
修改 dexmani_policy 模型/训练语义
```

也不要回滚或弱化：

```text
CommandFeedbackSnapshot transaction
generation/liveness fences
SafetyGate
command validity / ticket checks
worker final SDK fences
joint/workspace limits
session_failed vs error_state separation
causal multimodal observation assembly
```

---

## 4. Final Timing Contract

### 4.1 唯一 action cadence source of truth

定义：

```python
next_control_ns = (
    None
    if last_publication_ns is None
    else last_publication_ns + step_dt_ns
)
```

其中：

```text
step_dt_ns = PolicySpec.control_dt_s
last_publication_ns = 实际成功 publish / dry-run publishability 完成时刻
```

不得再用以下任意一个作为 action scheduling grid：

```text
executor poll grid
episode_start + k*dt
recording next_record_ns
camera frame timestamps
LoopRate internal deadline
```

### 4.2 Episode first chunk

```text
last_publication_ns = None
queue empty
→ immediately build fresh observation
→ blocking inference
→ dispatch chunk[0] immediately
```

### 4.3 Normal chunk execution

```text
queue non-empty
→ before last_publication + dt: no dispatch
→ at/after deadline: dispatch exactly one queue head
→ successful publication updates last_publication_ns
```

### 4.4 Chunk boundary

```text
publish old_chunk[-1] @ t0
→ queue empty
→ do NOT query before t0 + dt
→ at/after t0 + dt build fresh observation
→ blocking inference
→ generation/liveness/freshness checks
→ publish new_chunk[0] immediately
```

因此 synchronous boundary gap 是：

```text
dt + inference/host work
```

这是预期行为，不是 bug。

### 4.5 Late inference

如果 inference 很慢：

```text
infer end > nominal boundary
→ do NOT skip action[0]
→ do NOT catch up
→ if first prediction still fresh, publish action[0] immediately
→ its actual publication becomes the next cadence anchor
```

如果 inference 导致 model input stale，则 whole prediction discard + replan。

---

## 5. `deployment/executor.py` 修改方案

### 5.1 保留高频 executor polling

保留：

```text
runtime.policy.executor_poll_hz
```

不要恢复 PolicyRunner 的 `LoopRate`。

executor polling 负责：

```text
heartbeat
lifecycle boundary
recorder poll
metrics
wake-up near action deadline
```

Policy action cadence 仍完全由 actual publication deadline 控制。

### 5.2 增加一个极小 helper

建议：

```python
def _next_control_boundary_ns(self) -> int | None:
    if self.last_publication_ns is None:
        return None
    return self.last_publication_ns + self.step_dt_ns
```

不要建立新的 scheduler abstraction / enum state machine。

### 5.3 重写 `_run_active_tick()` 为最小两分支

目标形态：

```python
def _run_active_tick(self, now_ns: int) -> None:
    if not self._running_generation_is_live():
        return
    if self._running_time_expired(now_ns):
        return

    boundary_ns = self._next_control_boundary_ns()

    # Previous physical publication has not yet occupied one full control period.
    if boundary_ns is not None and now_ns < boundary_ns:
        return

    # Existing chunk: one action per eligible control boundary.
    if self.actions:
        self._dispatch_action(self.actions[0])
        return

    # Queue empty and boundary reached: fresh synchronous replan.
    self.observation_id += 1
    anchor_ns = time.monotonic_ns()
    observation = _build_observation(..., anchor_ns=anchor_ns, ...)
    if observation is None:
        return

    policy_observation = _to_policy_observation(...)

    if not self._running_generation_is_live():
        return

    infer_started_ns = time.monotonic_ns()
    try:
        predicted = self.model_runtime.predict(policy_observation)
    except Exception as exc:
        self._fail_policy_session(...)
        return
    infer_finished_ns = time.monotonic_ns()

    self.stats.inference_latency_ms = (infer_finished_ns - infer_started_ns) / 1e6

    if not self._running_generation_is_live():
        self._invalidate_chunk("inference_run_boundary")
        return
    if self._running_time_expired(infer_finished_ns):
        return
    if not self._poll_recorder():
        return

    self.chunk_sources = observation_sources(observation)
    self.chunk_action_index = 0
    self.actions.extend(predicted)

    # No extra dt here. Boundary elapsed before observation/inference.
    self._dispatch_action(self.actions[0])
```

实现时保留当前已有的 logging、recording、generation 等必要逻辑；上面只定义 control flow，不要求机械复制。

### 5.4 删除当前 phase-shift 特殊路径

删除 `run()` 中等价于：

```python
if (
    run_started
    and not self.actions
    and self.last_publication_ns != previous_publication_ns
):
    continue
```

的“上一 chunk 尾 action 发布后立即重新 query”逻辑。

`run()` 不应知道“刚发的是不是 chunk tail”。

### 5.5 `run()` 的 sleep 只负责高频 poll + deadline wake-up

建议保持：

```python
wait_s = self.poll_period_s
boundary_ns = self._next_control_boundary_ns()
if boundary_ns is not None:
    remaining_s = (boundary_ns - time.monotonic_ns()) / 1e9
    if remaining_s > 0:
        wait_s = min(wait_s, remaining_s)
time.sleep(max(0.0, wait_s))
```

不 busy-wait；robot-facing precision 仍由已有 hardware workers 负责。

---

## 6. 统一 Chunk Invalidation

将 `_invalidate_chunk()` 改成显式 reason：

```python
def _invalidate_chunk(self, reason: str) -> None:
    ...
    self.actions.clear()
    self.chunk_sources.clear()
    self.chunk_action_index = 0
    self.previous_arm_command_qpos = None
```

必须保留：

```text
last_publication_ns
run_generation
observation_id
model episode RNG/state
recording session state
```

原因：最后一个真实 command 的 publication time 仍是物理 cadence anchor；replan 不是新 episode。

以下 recoverable discontinuity 统一走该 helper：

```text
temporary/unavailable command feedback
stale command feedback
same feedback snapshot ages out before publication
first predicted chunk model-input stale
generation-safe prediction discard（按现有 lifecycle 语义处理）
```

Fatal feedback issue 仍走 hardware/control `_fault()`。

---

## 7. Freshness 分层，不得混用

### 7.1 Observation freshness — inference 前

Owner：`_build_observation()`。

最终 assembled observation 必须满足：

```text
causal: every selected source <= anchor
latest source age <= policy.max_input_age_s
cross-modal latest source skew <= policy.max_observation_skew_s
```

可以复用现有 `observation_timing_ms()` 计算，但 validation 应位于 observation owner boundary，避免 caller 各自重复。

如果 observation 暂时不可用：

```text
return None / wait for next poll
```

这不是错误，不增加 stale-prediction/error counter。

### 7.2 Prediction freshness — 仅 chunk[0] publication 前

保留现有“model input source time → current publication time”的检查。

只有 `chunk_action_index == 0` 才检查整块 prediction 的 input age。

若 stale：

```text
stale_prediction_count += 1
consecutive_stale_predictions += 1
_invalidate_chunk("stale_model_input")
```

不要检查 chunk[1...N-1] 对初始 observation 的 age；它们本来就是 open-loop future actions。

### 7.3 Command-feedback freshness — 每个 dispatch

保持当前 Patch A：

```text
read_command_feedback once
→ immutable CommandFeedbackSnapshot
→ decode / IK / projection
→ prepare_command(feedback_snapshot=...)
→ SafetyGate
→ same-snapshot freshness recheck
→ publish
```

Policy action admission 的 feedback freshness source of truth 继续使用：

```text
runtime.policy.max_input_age_s
```

不要退回 heartbeat timeout。

---

## 8. 防止无限 stale-prediction replan

新增 process-local：

```python
self.consecutive_stale_predictions = 0
```

只在以下事件递增：

```text
一次完整 blocking inference 已完成
AND
预测 chunk 尚未发布任何 action
AND
chunk[0] 因 model-input stale 被整块丢弃
```

以下事件 **绝不递增**：

```text
还没到 next control boundary
_build_observation() 返回 None
camera / state 暂时没新数据
普通 executor poll
feedback transient stale（已有自己的 replan semantics）
```

成功 publish 一个新 chunk 的 `action[0]` 后重置为 0。

使用现有：

```text
runtime.policy.max_consecutive_errors
```

作为上限，不新增 config。

达到上限：

```text
abort current rollout
session_failed = True
quit_requested = True
verified shutdown
```

不得置 `error_state`，因为这是 policy timing/session failure，不是 physical fault。

---

## 9. Warmup Timing 只作为诊断，不作为硬证明

当前 policy startup 已有：

```text
model_runtime.warmup(samples=5)
```

保留 timings logging。

如果 5 个 warmup samples 全部明显超过 `max_input_age_s`，可以增加 warning，提示当前 artifact/runtime 很可能无法满足 publication freshness。

**不要仅凭有限 warmup sample 直接 hard-fail startup。**

原因：warmup timing 是经验样本，不是未来 latency 的数学下界；最终 runtime 仍由 consecutive stale-prediction guard 给出真实 failure decision。

---

## 10. Policy / Model Failure Taxonomy

在 PolicyRunner 的 model boundary 捕获 `Exception`：

```text
CUDA OOM
model forward exception
invalid/NaN/Inf prediction
adapter/runtime contract violation
```

处理顺序：

```text
1. fence / abort current rollout using existing lifecycle primitive
2. stop/discard recorder according to existing rollout failure semantics
3. session_failed = True
4. quit_requested = True
5. log full exception
6. return to supervisor for verified shutdown
```

不得：

```text
set error_state=True
revoke directly to FAULT
把 model failure 伪装成 hardware fault
```

不得 catch：

```text
BaseException
KeyboardInterrupt
SystemExit
GeneratorExit
```

Fatal robot feedback/controller/SDK integrity failure 仍由现有 `_fault()` / supervisor 路径进入 `SafetyState.FAULT`。

---

## 11. Failure State Model

最终只需要三层：

```text
TRANSIENT REPLAN
    observation temporarily unavailable
    feedback transient/stale
    one-off stale first prediction
        ↓
    remain RUNNING

SESSION FAILURE
    repeated stale first predictions
    policy/model exception
    invalid model output
    recorder/service failure
        ↓
    safe fence → session_failed → verified shutdown

CONTROL / PHYSICAL FAULT
    fatal robot feedback
    robot controller/SDK failure
    unsafe/unverified shutdown
        ↓
    error_state → SafetyState.FAULT
```

不要再新增平行 error-state hierarchy。

---

## 12. Observation Alignment

保留 `ad5511c4` 的 query-anchored history：

```text
references = [anchor-(T-1)dt, ..., anchor]
```

并保持各 modality 使用同一组 reference times。

这是正确改动；本任务只修正 **anchor 产生的时刻**：

```text
错误：chunk tail 刚 publish 就立即 anchor/query
正确：last publication + dt 到达后才 anchor/query
```

保留：

```text
causal newest-source selection
warm-up edge repeat
camera generation consistency
pointcloud/RGB source pairing
arm/hand/tactile same-sample provenance
```

不要恢复旧 episode-start fixed grid 作为 observation anchor。

---

## 13. Recording Semantics

保留当前 truthful timing 原则：

```text
actual action publication
→ immediately record action decision with actual timestamp

blocking inference
→ sampling hook may pause
→ do not synthesize missing action rows

fixed-FPS rgb.mp4
→ visualization only
HDF5 timestamp
→ actual control timing source of truth
```

允许在长期无新 command、超过一个 control period 后按现有逻辑记录 held/idle state；但 action publication 后一个 control period 内不得追加重复 held row。

不要为了“看起来固定 16 Hz”补造 control samples。

不要修改 persisted recording schema。

---

## 14. Metrics

保留现有：

```text
inference_latency_ms
observation_age_ms
observation_skew_ms
publication_interval_ms
publication_input_age_ms
stale_prediction_count
safety_rejection_count
ik_rejection_count
```

本任务不要求新 telemetry framework。

如能以很小改动复用现有 `PolicyStats`，可选增加：

```text
chunk_boundary_interval_ms
deadline_lateness_ms
```

但它们是 optional，不应阻塞核心修复。

---

## 15. 需要修改的文件

主修改面应尽量限制为：

```text
dexmani_real/deployment/executor.py
dexmani_real/deployment/inference/observation.py
```

可选：

```text
dexmani_real/deployment/metrics.py
README.md
examples/run_policy.py
```

原则上 **不要修改**：

```text
dexmani_policy
control/safety_gate.py
control/publication.py（除非当前 source 已变化并证明最小兼容修改必要）
deployment/lifecycle.py（Patch B semantics 已足够）
IPC schema / persisted schema
```

---

## 16. Offline Regression Validation

Repository 当前没有强制的 pytest dependency；不要为了本任务引入大型测试基础设施或新测试依赖。

如果新增持久回归测试，优先：

```text
Python standard-library unittest
pure helper / fake clock
no SDK/device import side effect
```

### 16.1 Scheduler invariants

固定 `dt = 62.5 ms`：

```text
Case A — fast inference
last publish = 1000.0 ms
query >= 1062.5 ms
inference = 20 ms
new chunk[0] ≈ 1082.5 ms
NOT 1125.0 ms

Case B — slow inference
last publish = 1000.0 ms
query >= 1062.5 ms
inference = 100 ms
new chunk[0] ≈ 1162.5 ms

Case C — no catch-up
new chunk[0] = 1162.5 ms
new chunk[1] >= 1225.0 ms
```

### 16.2 Replan invariants

```text
feedback stale during chunk
→ remaining queue cleared
→ previous_arm_command_qpos cleared
→ last_publication_ns preserved

prediction stale before chunk[0]
→ same whole-chunk continuity reset
→ consecutive_stale_predictions +1
```

### 16.3 Observation invariants

```text
all source timestamps <= anchor
latest source age <= max_input_age_s
cross-modal latest-source skew <= max_observation_skew_s
future timestamps rejected
```

### 16.4 Lifecycle invariant

```text
generation/state changes during blocking inference
→ inference result discarded
→ zero publication from revoked generation
```

### 16.5 Failure taxonomy

```text
model exception / repeated stale prediction
→ session_failed
→ no error_state

fatal robot feedback/control failure
→ error_state / FAULT
```

### 16.6 Recording invariant

```text
inference boundary gap
→ HDF5 contains real timestamp gap
→ no fabricated action sample
```

### 16.7 Highest-value Policy ↔ Real contract test

构造 fake Policy：

```text
latest observation logical index = k
→ predicted first action encodes k
```

验证：

```text
Real queries observation k
→ first published action must encode k
```

该测试用于永久防止 `obs_t → action_t executed at t+1` 的 off-by-one / phase-shift regression。

---

## 17. 最小安全验证命令

实现前先 inspect 当前 source；不要执行硬件入口。

完成后至少运行 repository contract 已列出的 offline checks：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

若实现了 standard-library offline tests，再运行对应：

```bash
python -m unittest <focused_test_module>
```

不得把以下命令当测试：

```text
examples/run_policy.py
teleop/replay/homing/calibration entrypoints
任何连接 camera/robot SDK 的脚本
```

最后 inspect focused diff，确认没有：

```text
new async path
new scheduler abstraction
new config knob
schema drift
SafetyGate weakening
hidden catch-up
stale-prefix skip
unrelated cleanup
```

---

## 18. Acceptance Criteria

全部满足才算完成：

1. Queue-empty replan **不会早于**上一条 actual publication + `control_dt_s`。
2. Fresh inference 完成后，新 chunk `action[0]` **不再额外等待一个 `control_dt_s`**。
3. 所有成功 action publication 间隔以 actual publication 重新 anchor；late inference 后无 catch-up。
4. Episode first chunk 仍是 fresh observation → inference → action[0] immediately。
5. Stale model input 与 stale feedback 都执行 whole-chunk invalidation，并清 continuity reference。
6. Replan 保留 `last_publication_ns`，不会人为再多插一个空 control period。
7. Observation owner boundary 显式维持 causal / age / cross-modal skew invariant。
8. 单次 stale prediction 是 transient replan；连续 stale prediction 达到既有上限后成为 session failure。
9. Observation unavailable / waiting poll 不计入连续 stale-prediction counter。
10. Policy/model exception 安全终止 session，不写 `error_state`。
11. Fatal robot/control integrity failure 仍 fail closed 到 FAULT。
12. Recording 不制造虚假固定频率 action；真实 timing 可从 HDF5 timestamps 恢复。
13. `dexmani_policy` 不需要修改。
14. 不新增 async/RTC/timestamp-prefix scheduler。
15. Offline checks 通过；未进行硬件验证时必须明确报告“hardware unverified”。

---

## 19. 推荐提交顺序

### Commit 1 — Fix synchronous policy timing semantics

只做：

```text
remove immediate post-tail query
query only after actual last publication + dt
publish new chunk[0] immediately after blocking inference
preserve actual-publication re-anchor
preserve no-catch-up
simplify run() wait logic
```

不要在这个 commit 顺便改 failure taxonomy。

### Commit 2 — Harden replan and policy failure semantics

做：

```text
unified _invalidate_chunk(reason)
reset continuity on stale first prediction
consecutive stale-prediction guard
observation final timing admission
policy/model Exception → session failure
warmup latency warning only
```

### Commit 3 — Lock timing contract and update docs

做：

```text
focused offline regression tests if appropriate
README / run_policy wording
remove stale comments describing old semantics
```

---

## 20. 最终预期时间线示例

配置：

```text
control_hz = 16 Hz
dt = 62.5 ms
n_action_steps = 8
inference = 40 ms
```

正确行为：

```text
chunk k

a0      a1      a2                  a7
0       62.5    125                437.5 ms
                                  │
                                  │ hold one dt
                                  ▼
                               500.0 ms
                             fresh query
                                  │
                                  │ blocking inference 40 ms
                                  ▼
                               540.0 ms
                         publish chunk k+1 a0
                                  │
                                  │ +62.5 ms from actual publish
                                  ▼
                               602.5 ms
                         publish chunk k+1 a1
```

因此：

```text
normal chunk interval = 62.5 ms
chunk boundary interval = 102.5 ms
```

`102.5 ms` 是 synchronous blocking inference 的真实代价，不是 scheduler bug。

若未来要把它优化回接近 `62.5 ms`，应另立任务研究：

```text
overlapped inference
RTC
future timestamp scheduling
stale-prefix skip
action replacement
async inference service
```

不得在本 baseline 中通过提前 observation + 延迟 stale action[0] 的方式隐式实现。

---

## 21. 最终设计原则

```text
simple synchronous semantics > hidden latency tricks
actual publication > nominal action grid
fresh replan > resume stale chunk
one feedback transaction > TOCTOU reads
transient replan > premature session failure
session failure > false physical FAULT
truthful timestamps > fabricated fixed-rate data
minimal coherent change > runtime redesign
```

实现完成后，从 source 再次验证：

```text
observation producer
→ Policy public contract
→ prediction
→ action decode/projection
→ SafetyGate
→ command publication
→ robot worker
→ recording side effect
```

确认 README 与最终行为一致，然后删除/归档本 task 文件。