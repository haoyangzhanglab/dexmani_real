# Task: 根治 Policy Runtime 时序/反馈事务与数据语义问题

> **性质**：one-off implementation task，用于 Claude Code 执行本轮根因修复；不是永久架构文档。
>
> **位置**：按用户要求临时放在 repository root。实现完成、验证通过、README/source 已反映最终行为后，应删除或归档本文件。
>
> **Reviewed baseline**：`dexmani_real` `main@835cb8c740fcab3c5668df4f3cdc44c20749bb78`。如果执行时 HEAD 已变化，以当前 source/config/schema 为最终事实来源，先重新核对调用链和 invariants，不要机械套本文伪代码。
>
> **安全**：本任务仅允许 source inspection 与 offline validation。未经用户单独明确授权，不得运行 policy rollout、teleoperation、physical replay、homing、calibration、camera/device discovery，或任何会打开真实 robot/camera SDK 的代码。

---

## 0. 总目标

在**不重新设计 synchronous scheduler、不放宽安全判据、不新增异步推理、不修改 dexmani_policy** 的前提下，根治以下已确认问题：

1. `read_command_feedback()` 的时间采样顺序允许正常并发 frame 被误判为 `future_timestamp`，导致错误的 physical FAULT。
2. `CommandFeedbackSnapshot` 缺少 ring commit provenance，无法区分 producer timestamp bug、ring/clock invariant bug 与普通 stale。
3. `tests/test_sync_policy_timing.py` 的 fake clock 没有真正替换 `executor.time`，最高价值 timing contract test 不能可靠执行。
4. 现有 timing regression 的 fake policy 绑定 `observation_id` 而不是 observation time，不能从 action value 本身捕获 phase shift。
5. `consecutive_stale_predictions` 未在 execution/episode boundary reset，可能跨 episode 继承。
6. raw `policy_eval` rollout 如果进入现有 fixed-dt processing/replay，会被错误解释：
   - processing 无条件写成 `teleop_published_joint_target`；
   - physical replay 按 nominal `1/control_hz` 压缩 synchronous chunk-boundary gap。
7. 当前 follow-up 文档中部分分析已经被源码复核证伪，不能继续作为 coding instructions。

最终目标是建立四个明确 contract：

```text
Feedback transaction:
select immutable frames
→ post-selection validation time
→ provenance/freshness admission
→ one immutable snapshot
→ no ring reread during dispatch

Policy timing tests:
query time and action semantics由同一 deterministic clock/observation time定义

Prediction failure accounting:
one episode / one consecutive-failure sequence
→ new execution boundary resets local consecutive count

Dataset/replay boundary:
policy_eval irregular synchronous timing
→ cannot silently enter fixed-dt teleop processing/replay
→ fail closed until explicit support exists
```

---

## 1. 开始前必须做的事

完整阅读：

```text
AGENTS.md
CLAUDE.md
synchronous_policy_timing_followup.md
policy_runtime_root_cause_fix_task.md
```

然后至少检查：

```bash
git status --short
git log -8 --oneline
```

必须保护所有与本任务无关的用户修改。

随后追踪真实调用链，而不是只看本文：

```text
robot arm/hand worker
→ state ring write
→ SharedMemoryRingBuffer.read_latest
→ _read_arm_feedback / read_hand_feedback
→ read_command_feedback
→ CommandFeedbackSnapshot
→ PolicyRunner._dispatch_action
→ decode / IK / projection
→ prepare_command(feedback_snapshot=...)
→ command_feedback_is_fresh
→ publish_command

raw episode provenance/timestamp
→ EpisodeReader
→ dataset processing
→ processed HDF5
→ Policy Zarr export

raw episode provenance/timestamp
→ replay.trajectory.load_trajectory
→ EpisodeReplayer fixed-rate scheduling
```

如果当前 HEAD 与 reviewed baseline 不一致，先判断本文描述的问题是否仍存在，再修改。

---

## 2. 已确认根因 A：Command feedback 的 pre-read `now` race

### 2.1 当前错误语义

当前 `read_command_feedback()` 先执行：

```python
now_ns = time.monotonic_ns()
```

然后才读取 arm/hand rings，并把这个旧 `now_ns` 传给两个 reader 做 freshness/future validation。

因此合法并发可以产生：

```text
T0          policy captures now_ns
T0+0.2 ms   arm worker timestamps + publishes new frame
T0+0.3 ms   policy read_latest() gets that frame

frame.source_ns > old now_ns
→ FUTURE_TIMESTAMP
→ PolicyRunner._fault()
→ physical FAULT
```

这不要求 TSC drift、CLOCK_MONOTONIC bug 或 torn read；普通 producer/consumer concurrency 即可构造。

### 2.2 必须保持的事实

- `diagnose_arm_feedback` / `diagnose_hand_feedback` 的 `age_s < 0` fail-closed predicate 本身**不要放宽**。
- standalone `_read_arm_feedback()` / `read_hand_feedback()` 在 `now_monotonic_ns=None` 时已经是“先 read frame，再取 now”语义；不要为了修 Policy 路径破坏 teleop/replay/calibration/hand-homing callers。
- Patch A 的核心 invariant 必须保留：一个 dispatch 只选一次 feedback snapshot，decode/IK/SafetyGate/pre-publication recheck 都复用该 snapshot，中途不再读 rings。

---

## 3. Feedback transaction 的最终修复方案

### 3.1 最小修改原则

优先修 `read_command_feedback()` 聚合路径，不要全局重构所有 feedback caller。

禁止继续：

```text
capture common now BEFORE ring reads
→ pass old now into both ring readers
```

推荐语义：

```text
read+validate arm against a post-arm-read local now
        ↓
read+validate hand against a post-hand-read local now
        ↓
assemble immutable snapshot
        ↓
capture one final common validation_now
        ↓
revalidate assembled snapshot provenance + freshness against final common now
        ↓
return CommandFeedbackSnapshot
```

这既消除 false future race，又保证 arm 在较慢 hand read 后不会悄悄变 stale。

### 3.2 Ring commit provenance

`SharedMemoryRingBuffer.read_latest()` 已返回：

```text
(data, ring_commit_timestamp_ns, logical_sequence)
```

必须停止丢弃该 commit timestamp。

在 `_ArmFeedbackSnapshot`、`_HandFeedbackSnapshot` 以及最终 `CommandFeedbackSnapshot` 中携带必要的 ring publish/commit timestamp（字段名遵循当前项目命名习惯，例如 `*_publish_monotonic_ns`）。

每个选中的 modality 必须满足：

```text
0 < source_monotonic_ns
source_monotonic_ns <= ring_publish_monotonic_ns
ring_publish_monotonic_ns <= validation_now_ns
```

然后 freshness：

```text
validation_now_ns - source_monotonic_ns <= max_age
```

### 3.3 Timestamp-order issue 必须与 stale 区分

不要把 provenance/order violation 伪装成普通 stale。

推荐新增清晰的 machine-readable code，例如：

```python
FeedbackIssueCode.TIMESTAMP_ORDER
```

或同等明确命名。

至少区分：

```text
source > ring_commit
→ producer/provenance ordering invalid
→ fatal

ring_commit > post-selection validation_now
→ ring/clock invariant invalid
→ fatal

validation_now - source > max_age
→ STALE
→ recoverable replan in PolicyRunner
```

不要添加 1ms / 5ms “future tolerance”。

### 3.4 `command_feedback_is_fresh()`

该函数仍负责 **同一个 immutable snapshot** 在 decode/IK/SafetyGate 之后、publish 前的 freshness recheck。

如果 `CommandFeedbackSnapshot` 新增 ring provenance，建议这里也 defense-in-depth 验证 timestamp ordering，但不得重新读 ring。

函数保持 pure boolean 是可以接受的；详细 typed diagnostic 应在 initial command-feedback admission 处产生。

---

## 4. Feedback diagnostics

当前 fatal path 只记录：

```text
fatal command feedback: future_timestamp
```

必须提升到能定位时序 provenance 的级别。

至少输出：

```text
modality
issue code
issue detail
source_monotonic_ns
ring_publish_monotonic_ns
validation_now_ns
source_age_ms
source_to_ring_publish_ms
```

不要只使用 `.3f s` 量化亚毫秒 future/order 问题；必要时使用 ms/us 或原始 ns。

PolicyRunner fatal log 至少应保留 `issue.detail`，不能继续只打印 `issue.code.value`。

---

## 5. 不得误伤其它 feedback workflows

以下 caller 使用 `validate_arm_feedback` / `validate_hand_feedback` 或 standalone `read_hand_feedback`：

```text
teleop
keyboard teleop
physical replay
camera calibration
hand homing
```

本任务不要改变它们的安全语义，除非当前源码证明必须共享同一最小 helper。

特别验证：

```text
standalone read_hand_feedback()
```

仍保持：

```text
read frame
→ post-read current time validation
```

不要为了 Policy common-now 修复把它改回 pre-read now。

---

## 6. 已确认问题 B：Timing regression test 的 fake clock 没生效

`tests/test_sync_policy_timing.py` 的 `_Clock` 声称替换 `executor.time`，但当前 patchers 未 patch `executor_module.time`。

而 `_run_active_tick()` 内部仍直接调用模块级：

```python
time.monotonic_ns()
```

所以最高价值的 query-anchor assertion 在完整依赖环境会使用真实 monotonic clock。

### 6.1 最小正确修复

在 test patchers 中加入类似：

```python
mock.patch.object(executor_module, "time", self.clock)
```

不要为了测试引入 production-wide Clock framework。

### 6.2 Clock ownership 边界

测试 fake clock 只负责模拟：

```text
executor scheduling/query/inference wall clock
```

真机 production 的 command publication time 仍必须来自：

```text
publish_command
→ PublishResult.ticket.published_monotonic_ns
```

并继续作为真实 action cadence anchor。

不要让测试抽象改变真机 publication timestamp ownership。

---

## 7. 强化 Policy ↔ Real temporal contract test

当前 fake model 用：

```text
observation_id → action value
```

这只编码“第几个 query”，并没有真正编码 observation time。

必须把最高价值 test 改成：

```text
observation logical t = deterministic function(anchor_ns, step_dt_ns)
policy action[0] = encode(logical t)
```

例如可使用：

```python
logical_t = anchor_ns // step_dt_ns
```

或等价的、不会产生歧义的 deterministic mapping。

验证至少同时成立：

```text
query anchor == expected boundary time
first predicted action encodes same logical t
publish time == query time + fake inference latency
next action waits one full dt from actual publication
```

这样 query 提前一个 control phase 时，action value 本身也会 off-by-one，而不只是 `self.queries` assertion 失败。

---

## 8. 必须增加 feedback race regression

新增 deterministic offline test，构造旧实现会误判的顺序：

```text
hypothetical old now = T0
worker frame source/commit = T0 + ε
reader subsequently selects该 frame
```

新 `read_command_feedback()` 不应因为一个**在读取前已经完整 commit、但晚于旧 hypothetical now** 的合法 frame 产生 false `FUTURE_TIMESTAMP`。

同时测试真正异常：

```text
source > ring_commit
→ fatal timestamp-order issue

ring_commit > final validation_now
→ fatal timestamp-order issue

source too old
→ STALE
```

测试不得连接任何硬件。

---

## 9. 已确认问题 C：`consecutive_stale_predictions` 跨 episode 继承

当前该 counter 在 `__init__` 初始化，但 `_clear_execution()` 未 reset。

语义上它表示当前连续 prediction failures，而不是 session lifetime cumulative statistic。

修改：

```python
def _clear_execution(...):
    ...
    self.consecutive_stale_predictions = 0
```

累计统计仍由：

```text
PolicyStats.stale_prediction_count
```

承担。

增加测试：

```text
Episode A accumulated stale count > 0
→ execution cleared / new episode starts
→ consecutive_stale_predictions == 0
```

不要重构 `_input_is_fresh()`，当前没有已证明的“双计数一次 stale” functional bug。

---

## 10. 已确认问题 D：`policy_eval` 被错误送入 fixed-dt teleop pipeline

### 10.1 现有事实

Policy rollout recorder 已写 raw provenance：

```text
provenance_workflow = "policy_eval"
```

但当前 processing 无条件写 processed attr：

```text
action_semantics = "teleop_published_joint_target"
```

所以：

```text
policy_eval raw
→ current process_episodes
→ processed artifact silently claims teleop semantics
```

这是 semantic corruption。

### 10.2 Timing 事实

Raw policy-eval synchronous rollout 的 control rows 可以有真实 chunk-boundary gap。

Raw schema 已保存：

```text
timestamp
observation_anchor_monotonic_ns
```

Processed/Zarr 已保留：

```text
observation_anchor_monotonic_ns
```

因此“actual time 完全丢失”不是事实，不要为此新增重复 timestamp/schema。

真正的问题是现有 downstream contract 仍按 fixed-dt teleop sequence 解释 rows。

---

## 11. Processing 边界必须 fail closed on unsupported workflow

当前 processing pipeline 的语义是 fixed-step teleop imitation dataset。

在进入昂贵 processing 之前，检查 raw episode `/meta` provenance workflow。

推荐 acceptance：

```text
provenance_workflow absent / empty
→ legacy existing recordings: preserve current behavior

provenance_workflow == "teleop"
→ accept if such explicit provenance is introduced/currently supported

provenance_workflow == "policy_eval"
→ reject loudly

any other explicit non-empty workflow
→ fail closed as unsupported unless current source already defines support
```

错误信息必须明确：

```text
policy_eval rollout has synchronous/irregular execution timing and cannot enter the current fixed-dt teleop processing pipeline
```

不要把 policy-eval artifact改标成 teleop，也不要偷偷 resample。

### 11.1 不需要 schema bump

本任务禁止仅为了此问题执行：

```text
raw schema bump
processed schema bump
Zarr schema bump
新增重复 timestamp dataset
```

如果未来需要 policy-eval → DAgger/self-training/offline RL，应另立任务设计 explicit resampling 或 time-aware policy contract。

---

## 12. Physical replay 边界必须 fail closed on `policy_eval`

当前 physical replay scheduler 使用：

```text
fps = reader.timing.rate_hz
period = 1 / fps
next_deadline += period
```

因此 policy-eval raw 中真实 synchronous inference gap 会被压缩回 nominal fixed rate。

本任务不要顺带重写 replay scheduler 为 actual-time replay。

在 raw trajectory admission/load/preflight boundary 检查 provenance：

```text
policy_eval
→ reject fixed-rate physical replay
```

错误信息必须明确：

```text
policy_eval rollout contains synchronous irregular timing; current physical replay is fixed-rate and must not silently time-compress it
```

Legacy/current supported teleop replay 行为保持不变。

`load_processed_trajectory()` 最终回到 raw `source_path` 并调用 raw load；确保同一 provenance gate 也覆盖 processed-entry replay，不要维护两个互相漂移的判断。

---

## 13. 推荐 workflow provenance helper 的原则

可以选择：

- 在 processing/replay 两个 boundary 内各做极小判断；或
- 如果当前结构证明共享 helper 更清晰，定义一个非常小的 artifact provenance classifier。

要求：

```text
single meaning
no new framework
no config knob
no hidden default that converts policy_eval to teleop
```

优先复用 raw `/meta.attrs["provenance_workflow"]` 这一已存在 source of truth。

---

## 14. 本任务明确不做

以下方案已被 review 后排除，不得实现：

```text
future_timestamp 1ms/5ms tolerance
全局弱化 diagnose_arm_feedback / diagnose_hand_feedback
把 FUTURE_TIMESTAMP 降级成 transient replan
production-wide Clock/Scheduler framework
再次修改 synchronous scheduler timing semantics
async/background inference
RTC / stale-prefix skip / action replacement
recorder fabricated fixed grid
processed/Zarr 增加重复 actual timestamp only for this issue
raw/processed/Zarr schema bump only for timing
actual-time physical replay redesign
multimodal temporal resampling
_input_is_fresh 大规模重构
observation_id stale-dedup extra state
warmup all() → any()
删除 observation owner-boundary age/skew gate
修改 dexmani_policy
```

---

## 15. `synchronous_policy_timing_followup.md` 的处理

该文档目前包含已经被源码复核证伪/修正的分析：

- `future_timestamp` 主要归因于 clock jitter，并优先建议 tolerance —— 不再作为当前主假设；
- `_input_is_fresh` “一次 stale 会双计数，10≈5 attempts” —— 当前 control flow 不支持该结论；
- warmup `all()` 必须改 `any()` —— 不成立；
- recording grid 问题的责任层需要改写为 workflow/fixed-dt semantic boundary。

实现完成后：

- 如果该 follow-up 仍需保留，必须更新为最终事实；
- 如果已失去长期价值，按 AGENTS.md documentation discipline 删除/归档。

不要让旧诊断继续误导未来 coding agent。

---

## 16. 推荐修改文件范围

预计主要涉及：

```text
dexmani_real/control/publication.py
dexmani_real/utils/feedback.py                 # only if adding typed timestamp-order code
dexmani_real/deployment/executor.py
dexmani_real/dataset/processing.py
dexmani_real/replay/trajectory.py
tests/test_sync_policy_timing.py
focused new/extended offline tests
README.md                                      # only if user-visible supported workflow semantics need clarification
synchronous_policy_timing_followup.md          # update/remove after adjudication
```

原则上不要修改：

```text
dexmani_policy
deployment synchronous scheduler semantics
SafetyGate architecture
IPC/ring implementation
robot arm/hand worker scheduling
recording schema
processed/Zarr schema solely for this task
```

如果当前 HEAD 已变化导致需要额外文件，必须能明确说明它位于哪个 contract boundary。

---

## 17. 建议实施顺序

### Commit A — Fix command feedback temporal transaction

只做：

```text
remove pre-read common now race from read_command_feedback
preserve standalone feedback API behavior
retain ring commit timestamp
validate source <= ring_commit <= final validation_now
add typed timestamp-order diagnostic if appropriate
improve fatal feedback detail
```

不要同时改 dataset/replay。

### Commit B — Fix and strengthen offline regression tests

做：

```text
patch executor_module.time with deterministic _Clock
action encodes observation anchor/logical time
feedback read-order race regression
timestamp-order/stale regression
```

### Commit C — Reset episode-local stale counter

做：

```text
_clear_execution resets consecutive_stale_predictions
focused regression
```

可与 Commit B 合并，如果 diff 很小且审查更清晰。

### Commit D — Fail closed on policy_eval fixed-dt misuse

做：

```text
processing rejects policy_eval raw
physical replay rejects policy_eval raw
processed-entry replay inherits same raw gate
legacy/supported teleop behavior unchanged
focused provenance tests
```

最后再更新/删除 stale follow-up 文档。

---

## 18. 必须的 offline tests

至少覆盖以下 invariants。

### 18.1 Feedback transaction

```text
legal frame committed after a hypothetical old timestamp but before actual read
→ no false FUTURE_TIMESTAMP

source <= ring_commit <= final now
→ healthy if within max_age

source > ring_commit
→ fatal timestamp-order issue

ring_commit > final now
→ fatal timestamp-order issue

source old beyond max_age
→ STALE
```

### 18.2 Single snapshot reuse

确认 Policy dispatch 仍然：

```text
one arm/hand selection
→ decode/IK
→ prepare_command(feedback_snapshot=...)
→ same snapshot freshness recheck
→ no ring reread
```

### 18.3 Scheduler contract

```text
last publish = 1000ms
dt = 62.5ms
query = 1062.5ms
fake inference = 20ms
chunk[0] publish = 1082.5ms
next action >= 1145.0ms
```

Action value必须编码 query/observation logical time，而非 query ordinal。

### 18.4 Episode stale counter

```text
episode A consecutive count > 0
_clear_execution / new episode
→ count == 0
```

### 18.5 Processing provenance

```text
legacy/no workflow raw
→ existing processing behavior preserved

explicit supported teleop workflow (if supported by current source)
→ accepted

policy_eval
→ rejected before semantic relabeling/expensive processing

unknown explicit workflow
→ fail closed
```

### 18.6 Replay provenance

```text
legacy/current supported teleop raw
→ existing replay load/preflight preserved

policy_eval raw
→ fixed-rate physical replay admission rejects

processed artifact whose source raw is policy_eval
→ same rejection via raw source path
```

---

## 19. Offline validation

至少运行 repository contract 中的低成本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

运行新增/修改的 focused unittest。

关键 timing suite 必须在具备完整部署依赖的环境中**实际执行**；如果因为依赖不足被 skip：

```text
NOT RUN / SKIPPED
```

不得报告为 PASS。

禁止运行任何硬件入口作为验证。

---

## 20. 最终自审 checklist

完成前逐项确认：

```text
[ ] read_command_feedback 不再 capture pre-read common now。
[ ] 合法并发新 frame 不会被旧 now 误判 future。
[ ] source <= ring_commit <= post-selection validation_now 有明确 invariant。
[ ] 真正 timestamp-order violation 仍 fail closed，没有 tolerance。
[ ] Patch A single immutable feedback snapshot 语义未回退。
[ ] standalone hand/arm feedback workflows 未被意外改变。
[ ] executor timing test 的 fake clock 真正控制 executor.time。
[ ] temporal contract test 的 action encode observation time，而不是 observation_id。
[ ] consecutive_stale_predictions 在新 execution/episode boundary reset。
[ ] _input_is_fresh 未被无必要大改。
[ ] policy_eval raw 无法进入 fixed-dt teleop processing。
[ ] policy_eval raw 无法进入 fixed-rate physical replay。
[ ] legacy/current teleop processing/replay 行为未破坏。
[ ] 没有新增 schema bump、重复 timestamp、resampling 或 actual-time replay。
[ ] synchronous scheduler 3a4ff85 之后的正确语义没有被再次修改。
[ ] fatal feedback logs 足以定位 modality/source/commit/now。
[ ] stale follow-up 文档已更新或删除。
```

---

## 21. Acceptance Criteria

全部满足才算完成：

1. 正常 producer/consumer concurrency 不再产生 false `FUTURE_TIMESTAMP`。
2. 真正 `source > ring_commit` 或 `ring_commit > final validation_now` 仍然 fail closed。
3. Feedback snapshot在一个 dispatch 中只选择一次，并继续贯穿 decode/IK/SafetyGate/pre-publication recheck。
4. Fatal feedback diagnostic 能定位 modality 和 timestamp provenance，不再只有 issue code。
5. Timing regression suite 使用 deterministic executor clock，最高价值 test 不再依赖真实 monotonic clock。
6. Policy↔Real temporal test 用 observation time 决定 action value，能真实捕获 one-step phase shift。
7. `consecutive_stale_predictions` 不跨 episode/execution boundary 继承。
8. `policy_eval` raw 不能被 processed pipeline静默标成 teleop fixed-dt dataset。
9. `policy_eval` raw/processed-entry不能被 fixed-rate physical replay静默 time-compress。
10. Legacy/current supported teleop processing/replay behavior保持兼容。
11. 不增加 future tolerance、不修改 global feedback health semantics。
12. 不新增 production Clock framework、schema bump、actual-time replay、resampling 或 async scheduler。
13. Offline focused tests真实运行并通过；被 skip 的测试不能算 PASS。
14. 未进行硬件验证时，最终报告必须明确 `hardware unverified`。

---

## 22. 最终汇报格式

Claude Code 完成后请按以下结构汇报：

```text
Implemented
- ...

Root-cause invariants
- feedback transaction: ...
- timing test: ...
- workflow provenance: ...

Files changed
- ...

Offline validation
- PASS: ...
- SKIPPED / NOT RUN: ... because ...

Hardware validation
- Not performed

Residual risks / deliberately deferred
- actual-time replay remains unsupported
- policy_eval→training requires a future explicit resampling/time-aware contract
- ...
```

不要声称执行过任何实际未执行的硬件验证。

实现和文档稳定后，删除/归档本 one-off task 文件。
