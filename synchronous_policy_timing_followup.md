# 同步策略时序修复 — 剩余问题记录

> 范围：`3a4ff85`（时序）、`a1b7489`（重规划/失败语义）、`fe3b963`（离线测试+文档）
> 三个提交已合入 `main`。本文记录这批改动**遗留且尚未处理**的问题，按
> 「已诊断待修 → 待用户决策 → 待调查」排序。所有 `file:line` 以当前 `main` 为准。

---

## 概览

| # | 问题 | 严重度 | 状态 | 性质 |
|---|---|---|---|---|
| A1 | `future_timestamp` FAULT 只打 `issue.code`，丢了模态+幅度 | 中 | 已诊断，一行可修 | 诊断日志缺口 |
| A2 | 未来时间戳判定过脆（1ns 未来即停机） | 高(安全语义) | 已诊断，需定阈值 | fail-closed 语义 |
| B | 离线测试锚点/时钟补丁缺口 | 中 | 已确认，可修 | 测试基础设施 bug |
| C | 录制 grid 非均匀（`next_record_ns` 非定相位） | 高 | 待调查 | 数据契约 |
| D | `_input_is_fresh` 副作用重复触发 | 中 | 待确认 | 逻辑耦合 |
| E | warmup 告警谓词 `all()` 掩盖慢样本 | 低 | 已确认，可修 | 谓词错误 |
| F | 上一轮残险分析的三处修正 | — | 已确认 | 认知更正 |

---

## A. 真机 `future_timestamp` FAULT（已诊断，两处缺口未修）

### A.0 真机日志（用户补充，完整时间线）

单 session 三 episode 计划（`--num-episodes 3`），`Episode 1/3` 刚 RUNNING 约 3 秒即停机：

```text
[22:26:11] camera_loop: ready after first verified frame @ device 30.0 Hz
[22:26:11] camera_loop: skipped_frames=5 threshold=1 depth_frame=16 color_frame=42
            generation=1 health=OK(0) backlog_ms=51.626 (publishing current frame with reported health)
[22:26:11] pointcloud_loop: ready (shape=(1024,6), frame=xarm_base)
[22:26:13] CollisionModel ready: 19 DOF (7 arm + 12 hand), 40 geometries, ...
[22:26:15] operator: physical home sequence completed; press B to start
[22:26:17] safety: consumed B; ARMED(1) → RUNNING(2), generation=5 epoch_ns=191970310820912
            Episode 1/3 RUNNING
[22:26:20] safety: revoked motion RUNNING(2) → FAULT(3), generation=6
[22:26:20] policy summary: safety_rejections=0 ik_rejections=0
[22:26:20] policy: episode ended: fatal command feedback: future_timestamp
[22:26:20] [CRITICAL] policy: runtime fault: fatal command feedback: future_timestamp
[22:26:20] safety: revoked motion FAULT(3) → FAULT(3), generation=7   # 8…24 重复
[22:26:20] Episode saved: .../session_20260916_222601/episode_001 frames=49
            Episode 1/3 saved
[22:26:20] pointcloud_loop / RecorderIO / arm_loop: exited   # serial close
[22:26:23] camera_loop: exited
  shutdown: policy=graceful:0  arm=graceful:0  camera=graceful:0  pointcloud=graceful:0
            hand=graceful:0  recorder=graceful:0
── Session End ──
  exit_reason=error_state set  safety=FAULT  supervisor_normal=False  clean=False
```

关键事实：

- **3 秒内从 RUNNING 到 FAULT**：`22:26:17` B 触发 RUNNING（`epoch_ns=191970310820912`），`22:26:20`
  即 `RUNNING(2) → FAULT(3)`，generation 5→6。`epoch_ns≈191970.31s` 是单调时钟值（≈53h，非 Unix 时间）。
- **`safety_rejections=0 ik_rejections=0`**：排除了安全门拒绝和 IK 拒绝，说明 episode 只死在
  `fatal command feedback` 这一条 fail-closed 路径上，不是动作被拒。
- **整 session 中止**：只存下 `Episode 1/3`（49 帧），Episode 2/3、3/3 从未运行；
  `supervisor_normal=False clean=False` 表明 supervisor 经 `error_state` 异常退出，非正常收尾。
- **附带信号**：相机就绪即 `30.0 Hz`（印证 F2「16.7Hz 已过时」），且启动时
  `skipped_frames=5 backlog_ms=51.626`——视觉有约 51ms 积压（与 arm/hand 的 future_timestamp 无直接因果，
  但与 C 节录制/观测时序相关，供后续调查参考）。

### 结论

**不是这次 3 个 commit 引入的。** 故障路径早于本次改动：

- `FUTURE_TIMESTAMP` 枚举在 `2037b78`（"0829 temp snapshot"）加入 `utils/feedback.py:36`；
- `_fault("fatal command feedback: …")` 在 `4e030f9`（"Patch A"）加入 `deployment/executor.py:1142`。

本次三个提交只动调度门、`_invalidate_chunk`、`predict` 包裹和观测准入，未触碰反馈校验路径。

### 机制

`executor.py:1142` 的 `_fault` 由 `read_command_feedback` 返回 `feedback=None` 且 issue 非 `STALE` 触发。
`future_timestamp` 来自 `diagnose_arm_feedback` / `diagnose_hand_feedback`：

```python
# utils/feedback.py:98-103（arm）/ 140-145（hand）
age_s = (now_monotonic_ns - source_monotonic_ns) * 1e-9
if age_s < 0.0:
    return FeedbackIssue(FeedbackIssueCode.FUTURE_TIMESTAMP, f"... {abs(age_s):.3f}s in the future")
```

即 arm/hand 的 `source_monotonic_ns` 比 executor 的 `now_monotonic_ns` 还"晚"。两者都是宿主
`time.monotonic_ns()`（arm 在 `arm_worker.py:394`、hand 在 `hand_worker.py:300` 打源时间戳，executor
在 `control/publication.py:225` 一次性取 `now`）。

### 证据：瞬态，非系统性漂移

对 `session_20260916_222601/episode_001/data.h5` 逐帧提取（49 帧）：

- `arm` 源恒比发布时刻早 **0.84–33.3 ms**，`hand` 早 **0.68–33.0 ms**；
- **0 帧**出现"未来"（`arm > pub`、`hand > pub` 均为 0）。

时间对齐佐证「瞬态、非漂移」：日志 RUNNING 的 `epoch_ns=191970310820912` ≈ 191970.31s，而 HDF5
`timestamp[0]=191970.386s`（首个录制帧在 RUNNING 后约 75ms 才出现），`timestamp[-1]−timestamp[0]=3.227s`。
即 episode 以 16Hz 跑了约 49 个动作（3.2s）都正常，故障落在**第 50 次**（frame 49 之后），且那次
反馈读取失败（`feedback=None`）未进入录制——与"成功下发的 49 帧里无任何未来帧"完全一致。
机器为裸机 `AMD Ryzen 9 9900X`、`tsc` 时钟源（不变 TSC），跨进程 `CLOCK_MONOTONIC` 系统性漂移基本可排除，
故判定为**瞬时毛刺**。seqlock 环形缓冲（`ipc/ring.py`）经核对无 torn-read 路径能产生未来源时间戳。

### A1（中，一行可修）— 诊断日志丢掉 `issue.detail`

`executor.py:1142` 只打 `issue.code.value`（`"future_timestamp"`），把 `issue.detail`
（如 `"arm state timestamp is 0.003s in the future"`）丢了。当前日志无法区分是 arm 还是 hand、
超前多少 ms —— 而这正是判断"良性抖动"还是"真时钟故障"的唯一依据。

**建议**：

```python
self._fault(f"fatal command feedback: {issue.code.value}: {issue.detail}")
```

### A2（高，需用户定阈值）— 判定过脆

`age_s < 0.0` 意味着**哪怕超前 1ns 也直接 FAULT 停机**。裸机不变 TSC 上的微小瞬时偏移属良性抖动，
不该触发 fail-closed 停机；真正的时钟故障应体现为"大幅超前"。

**方向**（改变当前刻意的 fail-closed 语义，阈值需你拍板）：

1. 在 `diagnose_arm_feedback` / `diagnose_hand_feedback` 里把"微小未来"（如 <1–5 ms）clamp 到 0 视为新鲜，
   只有超过阈值才当 fatal；或
2. 把"微小未来"降级为 `_invalidate_chunk` 的 transient replan（回到下一次轮询），而非 `_fault`。

---

## B. 离线测试锚点/时钟补丁缺口（已确认，可修）

`tests/test_sync_policy_timing.py` 的 `_Clock` 类 docstring 声称
"used to replace ``executor.time``"（第 38 行），但 `_install_observation_fakes`（第 114–121 行）
只 patch 了 5 个模块助手（`_build_observation`、`_to_policy_observation`、`observation_sources`、
`observation_timing_ms`、`_select_control_grid_reference_ns`），**从未 patch `executor.time`**。

而 `_run_active_tick` 内部取的是真实 `time.monotonic_ns()`：

- `executor.py:1372` `anchor_ns=time.monotonic_ns()`；
- `executor.py:1387` `started_ns = time.monotonic_ns()`。

因此 `test_obs_t_to_action_t_at_chunk_boundary` 里 `self.assertEqual(self.queries, [1_062_500_000])`
会拿到真实单调时钟值而失败 —— 这正是整套测试最高价值的"`obs_t → action[0]_t`"契约断言。
其余 7 个测试不断言锚点时间，仍会通过（当前沙箱缺 `scipy`，8 个全 `skip`；部署机有 `scipy` 时该用例会跑并失败）。

**建议**：在 `patchers` 列表追加 `mock.patch.object(executor_module, "time", self.clock)`，
使 `_Clock` 真正接管 `executor.time`。

> 更正：上一轮口头结论里的"7/8 失败"是过度概括；准确数字是该套件中 **1/8（锚点契约用例）** 会失败。

---

## C. 录制 grid 非均匀（高，待调查）

`_record_rollout_tick`（`executor.py:704`）用**实际轮询时刻**而非固定控制相位做录制锚点：

- `executor.py:1466` `record_started_ns = time.monotonic_ns()` —— 每轮循环的抖动墙钟；
- `executor.py:819` `self.next_record_ns = now_ns + self.step_dt_ns` —— 以"上一个 `now_ns` + dt"作门限，
  而非 `run_started_ns + k*dt` 的定相位网格；
- held 行按 recorder-now 节奏（`:798`）、命令行按发布时间记录，两条节奏独立交错。

结果：HDF5 的时间戳网格**非均匀**（行间隔抖动），与 `recording/recorder.py:262`
"nominal grid rate; dt = 1/control_hz" 的注释语义相悖。下游（`export_policy_zarr`、物理 replay）
若假定 `control_dt` 均匀网格，可能受影响。

此外，`_select_control_grid_reference_ns`（`observation.py:333-343`）的 `np.maximum(run_started_ns, …)`
在启动预热段会把前置 slot 压平到 `run_started_ns`，产生非均匀 reference（属预热期固有，非本项主因）。

**待验证**：确认下游消费方是否假定均匀网格；若假定，需把录制锚点改成定相位 `run_started_ns + k*dt`，
或将均匀性保证显式写进录制契约。此项标注高严重度**待调查**，机制已定位但下游影响未证实。

---

## D. `_input_is_fresh` 副作用重复触发（中，待确认）

`_input_is_fresh`（`executor.py:1089`）在单次 `_dispatch_action` 内被调用**两次**：

- `executor.py:1129` 入口处；
- `executor.py:1197` 发布前复查。

每次调用都携带完整的 stale 副作用：`stale_prediction_count += 1`（`:1104`）与
`consecutive_stale_predictions += 1`（`:1105`）+ `_invalidate_chunk`（`:1106`）+ 达到阈值则
`_invalidate_rollout`/`_request_failed_session_shutdown`（`:1107-1116`）。若一块观测在入口检查时新鲜，
但经过 decode/IK/prepare 的耗时后、在 `:1197` 复查时越过 `max_input_age_s`，会为"同一块已被接受的观测"
再触发一次完整升级副作用，使 `consecutive_stale_predictions` 向 `max_consecutive_errors` 阈值加倍逼近。

另：`publication_input_age_ms` 在 `:1094`（`_input_is_fresh` 内）与 `:1226`（发布成功后）两处被写，
语义相近但取不同时刻，最终值取决于哪次调用后写。

**后果（上一轮已定位，具体触发路径待复核）**：计数器只在 chunk[0] **成功发布**时清零（`:1259-1260`），
而 `_input_is_fresh` 在单次 dispatch 里被调两次，一次 stale 可能在 `:1129` 与 `:1197` 两处各计一次，
于是 `max_consecutive_errors=10` 实际约等于 **5 次**真实 stale 尝试即触发 session failure，
`stats.stale_prediction_count` 也随之双计。

**待确认/建议**：明确 `:1197` 发布前复查是否应计入升级计数（建议只 gate 发布、不升级计数），
并厘清 `stale_prediction_count`（终身计数）与 `consecutive_stale_predictions`（连续计数）的职责是否重叠。

---

## E. warmup 告警谓词（低，已确认可修）

`policy_runner_loop`（`executor.py:1548-1557`）：

```python
warm_durations = [value for value in timings_s if np.isfinite(value) and value >= 0]
if warm_durations and all(value >= max_input_age_s for value in warm_durations):
    logger.warning(...)
```

`warmup(samples=5)` 返回的是**逐样本 `predict()` 延迟**（秒，`dexmani_policy` runtime.py:273-310，
每个样本一次 `time.perf_counter` 计时）。`all(...)` 要求**所有**样本都 ≥ `max_input_age_s`（0.15s）才告警；
但首样本通常是冷 CUDA 最慢、后续变快，于是 `all` 在"只有首个慢样本超阈值"的常见情形下不告警。

**更根本的语义错配（上一轮已定位）**：真正会导致 publication freshness 违反的是
`obs_age + 推理 + decode/IK/prepare > max_input_age_s` 的端到端场景；而 warmup 只测了其中"推理"一项，
再用 `all` 与完整的 `max_input_age_s` 预算比较——那种真实失败场景下这条告警**永远不会触发**，
最终只剩 `:1098` 一条 debug 级 discard 日志，运维无法提前预警。

**建议**：改 `all(...)` 为 `any(...)` 或 `max(...) >= max_input_age_s`，让任一慢样本即触发告警；
并注意 warmup 只测 `predict` 延迟，是真实端到端新鲜度预算的下界。

---

## F. 上一轮残险分析的三处修正（认知更正，非代码）

1. **skew 门不是慢相机的绑定约束。** 默认 `max_observation_skew_s=0.10`，但更紧的是
   `max_grid_lag_s=0.08`（`config/defaults.py:607-609`）。观测准入里 cross-modal 最新源 skew
   被 grid-lag 窗口（0.08s）所限制，故 0.10s 的 skew 门在默认配置下几乎不可触发；绑定约束是
   `max_grid_lag_s`。标注为分析性结论，建议用一次慢相机实机或注入验证。
2. **"16.7Hz RGB" 已过时。** 根因是 AE-priority ON 的 60ms 曝光，已设 `auto_exposure_priority=0`
   为全局默认，实际回到 30Hz/33ms。
3. **离线测试失败数更正为 1/8**（见 B 节），非 7/8。

---

## 建议处理顺序

1. **A1** — 一行日志修复，安全、立刻做（下次复现即能定位模态+幅度）。
2. **B** — 补 `executor.time` 补丁，纯离线、无硬件风险，恢复契约用例的有效性。
3. **A2** — 需你定未来容差阈值（安全语义变更），拿到 A1 的真实超前量后再定更稳。
4. **E** — `all()`→`any()`，小改。
5. **D** — 明确 `:1197` 复查的计数语义后再动。
6. **C** — 先查下游是否假定均匀网格，再决定是否改录制锚点。
