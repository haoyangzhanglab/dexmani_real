# dexmani_real 最终简化与可靠性重构实施方案

> Status: canonical execution plan  
> Repository: `haoyangzhanglab/dexmani_real`  
> Reviewed branch: `main`  
> Reviewed SHA: `668e1e9065d51fd85b52c3e0a8ce6a59db469f0d`  
> Review date: 2026-09-09

本文档是后续 Codex/人工实施 `dexmani_real` 重构时的唯一主计划。若 `main` 已前进，实施者必须先重新搜索相关 callsites、tests 和 contracts，再执行对应 phase，不能机械套用旧 patch。

---

## 1. 项目定位与优化目标

`dexmani_real` 的定位是个人 PhD 研究使用的 real-robot data collection、teleoperation、replay 和 policy deployment/evaluation repository，不是通用机器人产品框架，也不是 enterprise fault-recovery runtime。

优先级固定为：

1. Physical Safety
2. Experimental Correctness
3. Temporal / Causal Correctness
4. Recording Integrity
5. Reproducibility
6. Code Simplicity
7. Operational Convenience
8. Extensibility

因此，本轮重构的目标不是“文件越少越好”，而是：

- 删除重复执行；
- 删除重复 lifecycle ownership；
- 删除无价值 config projection；
- 删除无 downstream consumer 的 passive diagnostics；
- 保留真正保护 physical safety、causality、raw data truth 和 train/deploy contract 的复杂度。

参考 ManiUniCon / LeRobot 时只学习它们的直接 composition、owner 清晰和短 data path，不复制其更弱的 action-clipping、timestamp provenance 或 recording semantics。

---

## 2. 最后一次源码核查后的关键调整

### 2.1 `TeleopConfig` 保留

`dexmani_real/teleop/session.py` 在 spawn `policy` process 时，`TeleopConfig` 真实承载：

- canonical `ExperimentConfig` snapshot；
- `task_label`；
- `operator`；
- teleop-only hand URDF path；
- VR transform path。

它是合理的 process-boundary DTO。不要为了删除它把 `teleop_loop()` 变成大量 positional/keyword 参数，也不要新建另一套 session args DTO。

真正需要收敛的是 `TeleopCommandLimits`：它是从 `TeleopConfig.runtime` 解析出的 NumPy cache，只服务 teleop loop/grid。Phase 1 将其变成 teleop control-loop 内部 private value object，而不是第二套 public config surface。

### 2.2 Safety / freshness review 的结论变成全局保护项，而不是重构 phase

以下机制虽然比 ManiUniCon / LeRobot 严格，但分别保护真实 failure mode，因此不在主线中削弱：

- `SafetyState` / sticky hardware fault；
- `run_generation`；
- latest command ticket ownership；
- command validity/expiry；
- worker-side final SDK fence；
- heartbeat / generic fail-stop supervisor；
- `_CommandProgress`；
- `max_input_age_s`；
- `max_grid_lag_s`；
- `max_observation_skew_s`；
- strictly advancing visual history；
- source/publish/anchor causality。

不新增 actuator/recoverable/session fault severity framework。

### 2.3 Dataset 主优化从“删 STRICT”升级为“single-pass canonical processing”

当前 `examples/process_episodes.py` 先逐 episode 调 `process_episode_root(..., dry_run=True)`，随后生成临时 exclusion annotations，再对整个 batch 调 `process_episode_root()`；library 第二次又完整分析 source。

Canonical CLI 应只做一次 batch analysis，并只写 accepted episodes。

### 2.4 最新 Zarr export gap tolerance 是 protected behavior

最新 `main` 在 `dexmani_real/dataset/export.py` 中明确：

- leading source-row trim 可接受；
- 内部 gap 只有在缺失不超过 2 行、source sample 与 row 同步推进、source time delta 与缺失 control-grid steps 在 `source_contiguity_tolerance_s` 内一致时才可接受；
- 更大或无法解释的 gap 整条 episode reject；
- 不在 gap 处拆分 episode。

这个逻辑来自真实 `pick_place_toy` export incident，修复了过严准入导致的大量 false rejection。Phase 2 处理 raw->processed admission 时 **不得重写或回滚** 这套 processed->Zarr export-owned rule。

换言之，必须严格区分：

- raw -> processed：hard validity + temporal audit；
- processed -> Zarr：whole-episode structural continuity contract。

### 2.5 Recorder 必须采用两步 ownership 迁移

不允许一次性重写 `EpisodeRecorder` 和 `RecorderIO`。先加入同步 finalization API，再让 RecorderIO-owned finalizer 使用它，最后删除 EpisodeRecorder 内部 async lifecycle。每一步都必须保持 targeted tests 可运行。

### 2.6 Deployment semantic contract 默认保留

`deployment/config.py` 中 EEF / tactile / fingertip / point-cloud semantics 不是“字符串太多就可以删”的 metadata。特别是 point-cloud sampling / transform / color source / policy identity 可能改变输入分布。

Phase 5 改为 contract audit，不设强制 LOC reduction 目标。

### 2.7 `fsync_tree()` 不进入主线删除

`atomic_publish()` 同时服务 raw recording、processed batch、Zarr 和 pointcloud artifact。没有 profiling evidence 前不修改 durability contract。若 Phase 4 后仍证实 recursive fsync 是实际 bottleneck，再作为独立优化处理。

---

## 3. 全局设计规则

### R1 — Workflow is the composition root

以下 owner 继续直接组装 concrete workers/resources：

- `teleop/session.py`
- `teleop/keyboard_session.py`
- `deployment/lifecycle.py`
- `replay/session.py`
- `calibration/*/session.py`

禁止新增没有真实资源 ownership 的 `RuntimeManager`、`ApplicationManager`、`WorkerRegistry`、`PipelineManager` 等框架对象。

### R2 — One resource, one owner

典型 ownership：

- xArm SDK -> arm worker
- XHand SDK -> hand worker
- RealSense SDK -> camera worker
- policy model -> inference worker
- rollout scheduling -> `PolicyExecutor`
- episode lifecycle -> `RecorderIO`
- keyboard capture -> operator-input owner

### R3 — One transformation boundary

Observation：

```text
RuntimeChannels
    -> PolicyObservationBuilder
    -> PolicyObservation
```

这里统一完成 causal selection、history、freshness、multimodal alignment、pointcloud、EEF、fingertip、tactile 和 tensor shape/dtype。

Action：

```text
Prediction
    -> decode_policy_action()
    -> ActionCandidate
```

这里统一完成 `action` / `action_ee`、Rot6D、IK、arm/hand slicing 和 scheduling interpretation。

### R4 — Config is wiring

Canonical truth 是 `ExperimentConfig`。允许 process-boundary immutable config；不允许 config projection cascade。

### R5 — Complexity must protect something

可以复杂：physical safety、causality、IPC correctness、recording integrity。  
优先删除：passive diagnostics、duplicate validation、duplicate lifecycle、config projection、diagnostic-only taxonomy、duplicate CLI orchestration。

### R6 — Async requires a blocking reason

合法 async：

- `CameraStreamWriter`：active encode/write；
- Recorder finalizer：HDF5/video close、validate、filesystem publication。

不允许仅为了维护第二份 lifecycle state 而增加 thread/process。

---

## 4. Protected invariants

任何 phase、任何顺手 bugfix 都不得破坏以下契约。

### 4.1 Physical safety

- e-stop；
- sticky hardware fault；
- `SafetyState`；
- `run_generation`；
- latest command ticket；
- command expiry；
- arm/hand mechanical limits；
- workspace；
- required collision checks；
- worker-side final SDK fence；
- SDK/controller error fail-closed。

### 4.2 Deployment watchdog

- `_CommandProgress`；
- first-command timeout；
- command silence watchdog；
- command progress watchdog；
- stale-prediction prefix skipping；
- no catch-up scheduling。

### 4.3 Freshness / causality

- `source <= receive <= publish <= anchor`；
- run epoch isolation；
- `DeviceClockMapper`；
- `camera_generation`；
- `max_input_age_s`；
- `max_grid_lag_s`；
- `max_observation_skew_s`；
- strictly advancing visual history；
- state-camera causal pairing；
- tactile source identity。

### 4.4 Recording integrity

- exact sample sequence；
- `STOP through_sequence`；
- ring overflow detection；
- missing-sequence detection；
- ownership copy before sample ACK；
- real timestamps / real timing gaps；
- camera/source row-count consistency；
- temp staging；
- validation before final publication；
- same-filesystem atomic publication；
- exactly one `RecordingFinished` per stop lifecycle。

### 4.5 Dataset pipeline

Canonical pipeline remains：

```text
raw v25 -> processed HDF5 v14 -> Policy Zarr v7
```

并保持最新 export gap-tolerance contract。

---

## 5. Explicit KEEP / PARDONED

除非用户重新打开，不进入 cleanup proposal：

- generic EEF planner；
- `.claude/settings.local.json`；
- robot assets；
- `examples/xhand_control_example.py`；
- replay `eef_quat_wxyz`。

功能性强制保留：

- replay；
- all keyboard functionality；
- AudioFeedback；
- `dexmani_policy` external repo boundary；
- `control/action.py`；
- `deployment/prediction.py`；
- `deployment/timing.py`；
- `DexManiPolicyAdapter`；
- fixed `RuntimeChannels` graph；
- seqlock IPC；
- verified shutdown。

禁止修改 `dexmani_policy`。

---

## 6. 顺手修复小 Bug 的规则

允许在当前 phase 中修复小 bug，但必须同时满足：

1. 与当前修改文件/控制路径直接相关；
2. 原因确定，不依赖研究假设；
3. 不改变公开设计目标；
4. 不新增 subsystem/framework/dependency；
5. 修改局部；
6. 能新增或修改 deterministic regression test；
7. 不触碰 protected invariants；
8. 不需要新的硬件/研究决策。

推荐分类：

- B0：physical/data correctness；
- B1：deterministic functional bug；
- B2：usability/diagnostic/documentation bug；
- B3：possible bug / hypothesis。

B0/B1 若满足上述规则可立即修；B2 极小可修；B3 只记录，不改代码。

顺手 bugfix 应独立 commit：

```text
fix: <precise bug>
```

不得把 behavior fix 隐藏在 refactor commit 中。

不允许顺手修：研究语义、安全阈值/策略、schema/version、IPC schema、跨多个核心 subsystem 的 redesign。

---

# Phase 0 — Guardrails

## 目标

先锁定 correctness boundary，不引入大型测试框架。

## 基础 gate

```bash
python -m compileall dexmani_real
python -m pytest -q
git diff --check
```

记录 Python LOC / Python file count / test result；性能只做开发 baseline，不建立 flaky wall-clock CI threshold。

## Characterization tests

### Keyboard

锁定 held key、keydown/up、one-shot event、ESC sticky、listener failure。

### Dataset

至少覆盖：

- valid episode；
- unannotated rejected episode；
- explicit `include: true` but rejected；
- explicit `include: false`。

并保存代表性 raw->processed 与 processed->Zarr golden。

### Recorder

锁定：

```text
sample 1..N
STOP through_sequence=N
ring advances to N+k
=> output exactly through N
```

以及 episode-local failure 后 next START 仍可工作。

### Pointcloud example

锁定 CLI/defaults、synthetic RGB-D numerical output、table fit、benchmark semantics。

## Exit gate

所有 full tests green，characterization committed。

---

# Phase 1 — Leaf Simplification

低风险、局部、明显减少 production complexity。

## 1A. Keyboard passive diagnostics

删除 `_KeyboardMotionDiagnostics` 中 publication/SDK gap/tracking/qvel/IK timing/pending-frame 等 passive telemetry。

保留真实控制 disposition，例如 published / IK rejected / safety rejected。若 `_KeyboardPublishResult` 字段不参与任何 caller branch，才删除字段。

## 1B. `LoopRateStats`

Keyboard diagnostics 删除后，移除 `LoopRateStats`、`LoopRate.stats` 和仅用于统计的 counters。

保留 `LoopRate.wait()` / `reset()` / absolute deadline / no-catch-up / long-block re-anchor / 必要 overrun warning。

## 1C. `ArmParams.device_profile`

当前无 runtime consumer，删除字段与对应 validation/tests。

## 1D. `teleop/health.py`

- `advance_arm_feedback_error_count` -> `grid.py` private helper；
- arm/hand feedback adapters -> caller 直接使用 `utils.feedback`；
- 删除 `teleop/health.py`。

## 1E. `smoothing.py`

`ema_smooth_pose()` 行为不变地迁入唯一 owner `teleop/control_loop/action_proposal.py`，删除独立 module。

## 1F. Teleop command limits

- `TeleopConfig` KEEP；
- `TeleopCommandLimits` 改为 `teleop/control_loop/grid.py` 内部/private value object，或等价的单一 control-loop internal owner；
- 在 `teleop_loop` 启动时 resolve 一次 NumPy arrays；
- 不在 hot loop 重复转换；
- 不增加新的 public DTO。

## Exit gate

Targeted teleop/keyboard/config tests + full pytest；production LOC 应明显下降；无 public behavior change。

---

# Phase 2 — Dataset Admission + Single-Pass Canonical Processing

这是优先级最高的实质运行效率改造。

## 2A. STRICT -> audit-only

删除：

- `QualityPolicy.STRICT`；
- `strict_guard_before_frames`；
- `strict_guard_after_frames`；
- strict-only exclusion mask；
- CLI `--strict`；
- STRICT-specific tests。

保留：

- `HARD_ONLY`；
- `AUDIT`；
- temporal detectors；
- quality summary。

原则：hard validity 决定 processed admission；temporal quality 只报告，不自动裁掉 otherwise-valid rows。

## 2B. Library 增加一个窄的 batch policy 参数

建议：

```python
process_episode_root(
    ...,
    skip_rejected_unannotated: bool = False,
)
```

默认 `False`，保持现有 direct library caller 的 blocking contract。

当 canonical CLI 传 `True` 时：

- explicit `include: false` -> skip；
- explicit `include: true` 且 rejected -> block whole batch；
- unannotated rejected -> auto skip；
- accepted -> write。

不要创建新的 DatasetProcessor/Manager class。

## 2C. Canonical CLI single-pass

删除 `examples/process_episodes.py` 的第二套 orchestration：

- `_audit_episode()`；
- `_audit_episodes()`；
- `_merge_exclusions()`；
- 由 audit 结果生成临时 include=False annotation 的路径。

正常 path：

```text
parse
-> resolve runtime/config
-> load/apply user annotations
-> process_episode_root(..., skip_rejected_unannotated=True)
-> print report summary
```

`--compare-profiles` 仍可显式对每个 profile 做 dry-run，因为它本身就是分析功能。

## 2D. Publication transaction clean-up

Library 已在 staging 中生成 canonical `process_log/invalid_frames_report.json`。不要在 atomic publication 之后再次修改 final tree。

删除：

- CLI-side `_write_invalid_frames_report()`；
- `--write-report`；
- `_write_process_logs()`。

保留 `--dry-run`、`--compare-profiles`、`--verify-output`。

## 2E. 明确禁止修改 Zarr export gap tolerance

Phase 2 不修改：

- `_MAX_TOLERATED_INTERIOR_GAP_ROWS`；
- `_whole_episode_rejection()`；
- `source_contiguity_tolerance_s`；
- whole-episode export rule。

这属于 processed->Zarr contract，不属于 raw->processed temporal-quality simplification。

## Verification

必须证明：

- 默认 AUDIT 下代表性 processed arrays 与 before 等价；
- accepted/rejected source episode 语义符合新规则；
- explicit include 仍 blocking；
- canonical batch source analysis 不重复；
- process reports publication 前完成；
- Zarr v7 gap-tolerance tests完全不退化。

---

# Phase 3 — Workflow Simplification

固定顺序：Homing -> Pointcloud Example -> Keyboard Input。

## 3A. Homing

保留 `do_configured_teleop_home()` 的 caller API。

删除 `_do_teleop_home()` 大量 config-derived kwargs surface，把主体并入 `do_configured_teleop_home()`，内部直接从 `config.runtime.*` 读取 immutable配置。

动态状态继续显式传入：shared、planner、prev hand qpos、audio、estop callback、mapper、retargeter、hand availability。

必须保持：hand-first、SDK acceptance、fresh arm feedback、ArmHomeConfig、collision/path checks、generation/e-stop cancellation、table fallback、mapper/retarget reset、audio cue ordering、`home_done`。

## 3B. `examples/pointcloud_process_example.py`

严格 behavior-preserving：

```text
parse_args
-> load config/calibration
-> open camera
-> capture_rgbd
-> canonical process_frame
-> visualize / benchmark / table calibration
```

所有 mode 共享同一个 canonical preprocessing result。

禁止 Manager/Strategy/Pipeline framework；禁止修改 CLI/default/output/numerics/interactive behavior。

## 3C. Keyboard listener merge

当前 `KeyboardInput` 与 `KeyboardState` 是两个独立 pynput listeners。

目标：扩展现有 `KeyboardInput` 支持：

- `poll()`；
- `is_pressed()`；
- `pop_event()`；
- `wait_for_release()`；
- `quiesce()`；
- `healthy`；
- `estop_latched`。

覆盖 B/C/S/D/H/Q、WASD、arrows、ESC、x、space、enter、backspace；迁移 keyboard teleop 和 camera calibration；最后删除 `KeyboardState`。

必须先 characterization 当前 per-key / auto-repeat / release semantics；ESC 保持 sticky。

---

# Phase 4 — Recorder Ownership

本轮风险最高，必须独立 PR。

最终结构：

```text
RecorderClient
    -> control/result queues
RecorderIO process                 # sole lifecycle owner
    |- EpisodeRecorder             # synchronous serializer
    |- CameraStreamWriter thread   # active encoding/write
    `- finalizer thread            # blocking finish/validate/publish
```

## 4A. 先增加 synchronous finalization API

为 `EpisodeRecorder` 增加同步 `finish_episode(save, reason)`（具体返回值可复用现有信息，不要发明新的 public result hierarchy）。

旧 `stop_episode/poll_stop/join_stop` 暂留，使现有 tests 可继续运行。

## 4B. RecorderIO-owned finalizer

当 STOP boundary drain 完成：

1. freeze recorder writes；
2. start process-local finalizer thread；
3. thread 调 `finish_episode()`；
4. RecorderIO main 继续 heartbeat / control drain / finalizer poll；
5. main thread负责向 `shared.record_result_q` 发布 exactly one terminal result。

process-local thread result 优先使用标准库 `queue.Queue` 或等价最小机制，不新增 shared IPC schema。

## 4C. 再删除 EpisodeRecorder async lifecycle

确认 RecorderIO 已完全切换后，删除：

- `_stop_thread`；
- old StopResult；
- `poll_stop()`；
- `join_stop()`；
- live recorder registry / atexit lifecycle；
- duplicate previous-stop ceremony。

迁移 direct EpisodeRecorder tests。

## 4D. Episode-local failure 不永久污染 RecorderIO

删除 `_RecorderIOSession.had_failure` 作为 session sticky history。

例如 sample unavailable、ring overflow、decode/write failure、camera writer failure：

```text
discard current episode
-> RecordingFinished(error)
-> recorder remains usable
-> next START allowed
-> later clean process shutdown remains clean
```

真正 fatal 的 RecorderIO uncaught exception、result transport failure、finalizer无法安全回收仍 non-zero；finalization timeout继续 fail-stop。

## 4E. Single final artifact validation gate

最终：

```text
CameraStreamWriter.close
-> camera/source row count
-> flush data
-> write final meta
-> close HDF5
-> _validate_temp_episode()
-> atomic_publish()
```

删除 `_validate_temp_episode()` 之前重复的 sidecar existence gate。

## Protected Recorder contract

START sequence、STOP through_sequence、exact drain、copy-before-ACK、overflow/missing sequence、no post-STOP rows、one terminal result、temp staging、atomic visibility全部不得改变。

---

# Phase 5 — Contract Audit

Audit first，production diff optional。

## 5A. Deployment semantic gates 默认 KEEP

EEF / tactile / fingertip / point-cloud field semantics 已有 fail-closed tests；point-cloud processing identity也可能改变真实 tensor distribution。

禁止 Codex 仅因为 validator“字符串很多”就批量删除 `deployment/config.py::_validate_field_semantics` 或 expected semantics。

## 5B. 修正文档与代码 contract 描述

README / repo_map 应明确：`PolicySpec.observation_fields[*].semantics` 是 Real 使用的 public compatibility contract 一部分，而不仅是 shape/dtype。

## 5C. Processed->Zarr validator 分类审计

将检查分为：

- STRUCTURAL；
- PAYLOAD_NUMERICAL；
- INTERPRETATION；
- VARIABLE_PREPROCESSING；
- FIXED_SCHEMA_IDENTITY。

前四类默认 KEEP。只有同时满足以下条件的 FIXED_SCHEMA_IDENTITY 才可删除：

1. schema v14 已唯一确定；
2. 不影响 tensor 数值；
3. 不影响 tensor interpretation；
4. 不影响 variable preprocessing identity；
5. 无 downstream consumer。

最新 Zarr gap-tolerance / provenance logic属于 STRUCTURAL/temporal export contract，不得在此阶段顺手简化。

---

# Phase 6 — Post-refactor Cleanup

结构稳定后才删除已经失去用途的 observability：

- `StageTimer`；
- arm/hand pure counters；
- CameraStreamWriter percentile telemetry；
- pointcloud rolling performance telemetry；
- 已完成使命的历史 refactor docs。

`PolicyStats` 不整模块删除；进入 `result.json.metrics` 的正式实验字段继续保留。

Feedback taxonomy 只在多个 code 最终产生相同 control disposition 时才简化。

`fsync_tree()` 只有 profiling证明它是实际 finalization/export bottleneck 时才进入独立优化；否则 KEEP。

新加入的 `docs/invalid_frames_export_incident.md` 至少保留到相关 export admission 完全稳定，并将其仍有效的不变量抽到 canonical docs 后再考虑归档/删除。

---

# Phase 7 — Conditional Cleanup

只有现实条件明确后执行：

- TAG 长期唯一 retarget backend -> 才删 DexPilot；
- L515 长期唯一 RealSense -> 才删 D400；
- XHand 永久只走 serial -> 才删 EtherCAT；
- 无 wrist camera 研究计划 -> 才删 incomplete eye-in-hand；
- raw v24 migration冻结 -> 才做 raw v26 compatibility cleanup；
- 所有 physical workflow 都有 calibrated table -> 才删 `table_z_surface_m` fallback。

默认全部 DEFER。

---

## 7. 推荐 PR 划分

### PR1 — Leaf Cleanup

- keyboard passive diagnostics；
- LoopRateStats；
- `device_profile`；
- teleop health wrapper；
- smoothing；
- private command limits。

### PR2 — Dataset

- STRICT removal；
- `skip_rejected_unannotated`；
- single-pass canonical CLI；
- publication transaction clean-up；
- **不得修改最新 Zarr export gap tolerance**。

### PR3 — Workflow

- homing passthrough；
- pointcloud example；
- keyboard listener merge。

三个独立 commits。

### PR4 — Recorder

- synchronous finish；
- RecorderIO-owned finalizer；
- delete EpisodeRecorder async lifecycle；
- remove had_failure poisoning；
- one final integrity gate。

完全独立，不同时改 dataset/SafetyGate/atomic I/O。

### PR5 — Contract / Cleanup

- docs contract reconciliation；
- post-refactor diagnostics；
- only proven redundant semantic checks；
- optional durability tuning only with profiling evidence。

---

## 8. 推荐 commit 粒度

```text
cleanup: remove keyboard passive diagnostics
cleanup: remove LoopRateStats
cleanup: remove unused arm device profile
teleop: inline health policy into grid owner
teleop: co-locate EMA pose smoothing
teleop: privatize resolved command limits

dataset: remove strict temporal exclusion
dataset: support skipping unannotated rejected episodes
dataset: make canonical processing single-pass
dataset: keep process artifacts inside publication transaction

teleop: simplify configured homing
pointcloud: refactor example without behavior change
input: unify operator keyboard listener

recording: add synchronous episode finalization
recording: move finalizer ownership to RecorderIO
recording: remove EpisodeRecorder async stop lifecycle
recording: keep episode failures local
recording: consolidate final artifact validation

docs: document PolicySpec semantic contract
cleanup: remove obsolete runtime telemetry
```

任何顺手 bug：

```text
fix: <specific bug>
```

独立 commit。

---

## 9. 每个 task 的实施协议

### Step 1 — Verify base

先获取当前 HEAD；若与本计划 reviewed SHA 不同，先 compare/re-search。

### Step 2 — Search before delete

删除任何 symbol 前必须全 repo 搜 production callsites、tests、docs、indirect imports。

### Step 3 — One hypothesis per commit

每个 commit只做一种：删除重复机制、明确 ownership、或修复 deterministic bug。

### Step 4 — No opportunistic redesign

除符合“小 bug”规则外，不允许顺手改 protected invariants、research semantics、hardware policy。

### Step 5 — Targeted tests first

Recorder：`test_recorder_queue_io.py`、`test_recorder_queue_client.py`、`test_recorder_io_boundary.py`、`test_raw_v25_recording.py`。  
Deployment：`test_policy_rollout.py`、`test_prediction_boundary.py`、`test_observation_builder.py`、`test_deployment_eef_tactile.py`。  
Dataset：`test_processed_v14.py`、`test_zarr_v7_projection.py` 以及 dataset processing/quality tests。

### Step 6 — Full gate

```bash
python -m compileall dexmani_real
python -m pytest -q
git diff --check
```

---

## 10. Review metrics

不只看 LOC。每个 PR review：

- dependency edges；
- lifecycle owners；
- duplicate passes；
- duplicate mutable states；
- public API surface；
- class/dataclass count；
- production LOC；
- tests。

好的 simplification：dependency/owners/duplicate passes/public surface下降，tests稳定。  
危险 simplification：LOC下降但 safety、causality、data contract 变弱。

---

## 11. Hardware smoke test

PR3 / PR4 / PR5 完成后至少执行一次：

1. startup/readiness；
2. H home + audio；
3. VR teleop；
4. keyboard teleop；
5. ESC/e-stop；
6. record/save/discard；
7. raw inspect；
8. raw->processed；
9. processed->Zarr；
10. replay；
11. policy shadow；
12. controlled policy run/eval。

重点检查 stale command、unexpected FAULT、heartbeat timeout、record tail loss、keyboard release、camera stall、audio ordering、policy timing。

---

## 12. 最终成功标准

### Architecture

- workflow直接；
- one real resource/lifecycle owner；
- config projection减少；
- model tensor interpretation boundary唯一。

### Runtime

- safety unchanged；
- freshness unchanged；
- IPC causality unchanged。

### Dataset

- canonical processing 不重复 analysis；
- temporal heuristic 不再决定 otherwise-valid row admission；
- processed publication transaction clean；
- latest Zarr gap tolerance完全保留。

### Recorder

- RecorderIO sole lifecycle owner；
- async只隔离真正 blocking finalization；
- episode-local failure不污染后续 session；
- exact FIFO/STOP transaction unchanged。

### Maintenance

- passive telemetry减少；
- wrappers减少；
- public DTO只保留真实 process/producer-consumer boundary。

最终原则：

> 删除重复执行、重复 ownership、重复配置和无消费 observability；保留所有真正保护 physical safety、temporal causality、raw data truth 和 train/deploy tensor contract 的机制。
