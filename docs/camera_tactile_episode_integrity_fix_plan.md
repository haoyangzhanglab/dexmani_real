# Camera / Tactile 对齐与 Whole-Episode 数据完整性修复计划

> 面向 Claude Code 的 canonical 执行文档。
>
> Baseline：`main` @ `b5128927d920cca09dde40d910bcde037ad70ba7`（2026-09-09）。
> 当前 baseline 为 raw v26 / processed v15 / Policy Zarr v8。
> 若执行时源码已前移，以源码为准；schema 版本从当时当前版本各自只递增一次，不要复用本文数字硬覆盖新版本。
>
> 本文目标不是“放宽阈值”，而是把错误的 validity ownership 修正到正确层级：
> **正常 camera timing variation 不删训练行；tactile 在录制时按 deployment 语义完成 camera-source causal alignment；任何真正 internal hard-invalid 都不允许被 exporter 压紧跨越；一个 processed HDF5 最多对应一个 Policy Zarr episode。**

---

## 0. Executive decision summary

本任务只做一个小而完整的 vertical change，最终必须同时满足以下五条：

1. **Policy Zarr 只包含完整单 episode。** 一个 processed HDF5 要么整体作为一个 Zarr episode 进入，要么整体拒绝；禁止拆分，禁止把内部缺口压紧后继续导出。
2. **删除 exporter 的 interior-gap tolerance。** `_MAX_TOLERATED_INTERIOR_GAP_ROWS` 及等价 heuristic 不应存在于 canonical exporter。
3. **camera “当前 control-grid tick 没拿到新帧”不是 data corruption。** 只要已有 frame payload 合法、因果、age 在预算内，保留该 row，并把 `not-new/reused/duplicate` 作为 audit telemetry；真正的 clock reset、超 freshness budget、payload corruption 才是 hard-invalid。
4. **frame-0 tactile 不是 XHand startup failure。** BEGIN 前代码已经要求 tactile `fresh + calibrated + recent`。历史 frame-0 缺口的根因是：raw 只持久化了 control-grid-anchor tactile，而 visual processing 后来要求 camera-source-aligned tactile；episode 开始前虽然 ring 中存在更早 tactile history，但没有写进 raw，因此 offline 无法恢复。
5. **新数据不再 offline 从 16 Hz persisted rows 猜 camera-aligned tactile。** recording 直接从高频 hand/tactile ring 按 camera source time 做与 deployment 一致的 causal selection，并持久化其 payload + provenance。

一句话目标架构：

```text
high-rate sensor rings
        │
        ├── camera source C
        │
        ├── arm/hand newest source <= C
        └── tactile newest source <= C
                    │
                    ▼
          raw row stores exact
        camera-aligned policy obs
                    │
                    ▼
          processed = validation
          (no cross-row repair)
                    │
                    ▼
     whole-episode export admission
       accept one HDF5 / reject one HDF5
```

---

## 1. Scope and non-goals

### 1.1 In scope

- `camera` transient timing / reused-frame 的 offline data-quality 语义。
- visual profile 的 camera-aligned `contact_force` / dense `tactile_force` 录制。
- frame-0 tactile alignment deficit 的根因修复。
- processed provenance 对 aligned tactile 的直接表达。
- Policy Zarr whole-episode admission 回归严格语义。
- 针对历史 61 episodes 的只读诊断与 proposed-rule simulation。
- synthetic / offline tests 与明确 hardware experiment protocol。

### 1.2 Explicitly out of scope

本任务不要顺手做以下工作：

- 不设计通用 SensorManager / modality registry / plugin framework。
- 不改变 XHand `source_monotonic_ns` 为“SDK 调用开始时间”。当前时间戳是 host read-completion 近似；在没有硬件 device timestamp 证据前，不能伪造 acquisition time。
- 不对 tactile 做 interpolation、future fill、linear interpolation 或 nearest-future selection。
- 不把无效 tactile zero-fill 当成 no-contact。
- 不重新设计 `contact_force` vs dense `tactile_force` 的最终 Policy profile；两者的独立 validity 已由 v26 路径建立，本任务只保证 alignment provenance 正确。是否让 contact-only processed profile 完全不要求 dense tactile，作为后续独立任务。
- 不修改 robot safety、command freshness、collision、generation 或 hardware fault policy 来换取更多数据。
- 不自动运行会连接 xArm7 / XHand / RealSense / Quest 的程序。Hardware experiment 只实现必要 telemetry / protocol，实际运行必须由操作者显式执行。

---

## 2. Current facts, hypotheses, and engineering decisions

必须严格区分事实与待验证假设。

### 2.1 Confirmed code facts

#### F1. BEGIN 前 tactile 已经健康

`teleop/loop.py::_begin_feedback_issue()` 在 recording + hand enabled 时要求：

```text
hand_tactile exists
fresh == true
calibrated == true
0 < source <= begin_now
begin_now - source <= 250 ms
```

失败则 `cannot begin`，不会调用 `recorder.start_episode()`。

因此历史 frame-0 `tactile_invalid` **不是 XHand 尚未启动、尚未校准或 tactile worker 尚未 ready**。

#### F2. XHand tactile timestamp 不是 hardware acquisition timestamp

`robot/hand_worker.py` 的 normal loop 是：

```text
state = hand.get_state()
last_source_ns = time.monotonic_ns()
publish hand state + tactile with last_source_ns
```

因此 `tactile_source_monotonic_ns` 更准确的语义是：

```text
host timestamp after SDK read returned
```

它对 strict causality 是保守时间戳，但不能声称等于 tactile physical acquisition time。

#### F3. Recording 与 visual processing 目前使用不同 reference

Recording grid：

```text
read_hand_tactile_causal(... anchor = control-grid anchor T)
=> proves tactile source H <= T
```

Visual cleaner：

```text
tactile_reference_ns = camera_source_monotonic_ns C
=> requires selected tactile H <= C
```

而通常 `C < T`，所以合法的 grid-anchor tactile 完全可能落在：

```text
C < H < T
```

这不是 tactile payload failure。

#### F4. Offline selector 只能看到 persisted 16 Hz rows

`clean.py::select_tactile_rows_to_references()` 只允许使用在当前/更早 raw row 中已经持久化的 tactile source。

frame 0 没有 row -1，因此一旦 `tactile[0].source > camera[0].source` 就只能返回 `-1`。frame 1 起则可重用 row 0 tactile。

#### F5. Deployment 已经从高频 ring 做真正的 camera-reference alignment

`deployment/inference/observation.py`：

- 先读 arm / hand / aggregate tactile / tactile provenance / dense tactile histories；
- camera policy 先选择 causal camera grid；
- 再对每个 camera source timestamp 调 `_align_state_history_to_reference_ns()`：选择 newest `source <= camera_reference` 且 skew 有界的 sample；
- contact path 还要求 aggregate tactile 与 tactile provenance 的 `source_monotonic_ns` 完全相等。

因此 recording 应复用同一语义，而不是离线重新近似。

#### F6. Camera worker 已经区分 health 与 frame gap

`CameraHealth` 当前包含 `OK / CLOCK_RESET / DUPLICATE / FRAME_GAP / DELIVERY_DELAY`。
`FRAME_GAP` 本身不使 recovered current frame invalid；worker 明确保留当前 frame。

#### F7. `CameraFreshnessTracker.camera_fresh` 混合了 “new” 和 “usable”

当前 `camera_fresh=True` 同时要求：

```text
ring sequence new
frame number new
health == OK
source after episode start
age <= max_age
```

因此 `camera_fresh=False` 不能直接等价为 training row invalid；它可能只是该 16 Hz grid tick 复用了仍然 recent 的上一 camera frame。

### 2.2 Historical measured facts from the 61-episode incident

当前 incident 文档已经确认：

- 43/61 被 invalid report 标记；
- 大多数只在 `[0,1)`，即 frame 0；
- 41 条受影响 episode 的 frame-0 tactile timestamp 比 camera source 晚 `0.38–41.1 ms`，均值约 `16.3 ms`；
- 2 条没有 frame-0 标记，其 tactile 刚好比 camera 早 `10.7 / 3.0 ms`；
- 9 个 camera 标记 row 相对前一有效 camera source 的间隔为 `33.8–67.7 ms`，相邻 row 正常；
- 唯一明显 persistent behavior-quality event 是 `episode_20260827_224527` 的 19-frame IK failure hold。

### 2.3 Hypotheses to test, not claims

#### H1. Offline tactile has extra latency versus deployment

因为 processing 只能从 16 Hz persisted rows 重选，而 deployment 可从高频 ring 选，所以 visual training 中大量 row 可能使用 previous persisted tactile，造成额外数十毫秒 lag。

必须先统计：

```text
selected_tactile_row - current_row
camera_source_ns - selected_tactile_source_ns
```

不能在测量前声称“所有 tactile 都错后一帧”。

#### H2. Low-light exposure contributes to camera not-new events

仓库已经验证 AE priority ON 在暗场会把 exposure 拉长并降低 RGB fps；但历史 raw 没有保存 exposure / gain / actual fps，不能把现有 9 个事件确定归因于曝光。

正确表述：

```text
low light / exposure is a plausible contributor;
current historical data cannot prove it is the unique root cause.
```

---

## 3. Target invariants

Claude Code 修改完成后，以下 invariant 必须能从代码和 tests 直接看出。

### I1. One source HDF5 -> at most one Policy Zarr episode

绝不 split：

```text
processed_episode_42.h5
    ├── segment A -> Zarr episode 42a   # FORBIDDEN
    └── segment B -> Zarr episode 42b   # FORBIDDEN
```

### I2. No interior-gap compaction

以下数据绝不能在 Zarr 中变成相邻步：

```text
source row 100
source row 101
[row 102 invalid]
source row 103
```

不得导出成：

```text
... 100, 101, 103 ...   # false dt adjacency
```

### I3. New canonical schema accepts only source-complete episodes

对新 schema（本任务之后采集/处理的数据），Policy Zarr admission 应要求：

```text
no automatically dropped source rows
one contiguous source sequence
no internal segment boundary
no suffix truncation caused by hard-invalid data
```

如果某 row 真正 hard-invalid，该 processed HDF5 可以保留用于 audit，但 canonical Zarr exporter 整条拒绝。

### I4. Legacy artifacts are not silently reinterpreted

raw v26 无法恢复 episode 开始前没有持久化的 tactile ring history，因此不能写一个“v26 -> new schema”转换器去伪造 camera-aligned frame-0 tactile。

历史 v26/v15/v8 可继续作为 frozen legacy experiment artifacts 使用；新 schema correctness 不能靠改写旧字段语义实现。

### I5. `camera not new` != `camera hard invalid`

Training hard-invalid 只由直接的 usable conditions 决定；`not-new/reused/duplicate within age` 进入 audit。

### I6. Recording-time visual tactile == deployment selection semantics

对于 visual observation reference `C`：

```text
selected tactile source H
must satisfy H <= C
and C - H <= max_observation_skew
and required validity/calibration/unit gates
```

contact 与 tactile provenance 必须 source-match；dense tactile 额外要求 dense `fresh`。

### I7. No future information

绝不使用：

```text
H > C
```

的 tactile 去“修” camera observation，即使它只晚 1 ms。

### I8. XHand timestamp semantics remain honest

不把 host read-completion timestamp 改名/改义成 acquisition timestamp；若未来 SDK 提供真实 device timestamp，再单独升级 contract。

---

## 4. Target data flow

### 4.1 Current problematic path

```text
XHand high-rate ring
       │
       ▼
control-grid anchor T selects tactile H <= T
       │
       ▼
raw row stores H
       │
       ▼
visual processing changes reference to camera C
       │
       ▼
search only persisted raw rows for H <= C
       │
       ├── frame0: no older row -> -1
       └── later: often previous-row tactile
```

### 4.2 Target path

```text
                           recording grid anchor T
                                  │
                  causal camera row C <= T
                                  │
           ┌──────────────────────┼─────────────────────┐
           ▼                      ▼                     ▼
       arm ring               hand ring            tactile ring
           │                      │                     │
           └──── newest valid source <= C, skew bounded ┘
                                  │
                                  ▼
                 explicit policy-observation tactile
                                  │
                                  ▼
                           raw current row
                                  │
                                  ▼
                    processed copies + validates
                     (NO cross-row re-selection)
                                  │
                                  ▼
                    strict whole-episode exporter
```

---

## 5. Implementation plan

按以下顺序执行。不要先删 exporter tolerance 再处理 upstream，否则会把历史 false-positive 放大成大量 whole-episode rejection。

## Phase A — Add read-only baseline diagnostics first

新增一个纯离线工具，建议：

```text
tools/analyze_camera_tactile_alignment.py
```

要求：

- 只读 raw/processed artifacts；
- 不 import / construct hardware SDK；
- 默认输出 concise terminal summary；
- `--write-json PATH` 可写 machine-readable 结果；
- 不修改任何 episode。

至少统计：

### Tactile

```text
frame0_causal_deficit_count
same_row_selected_count
previous_row_selected_count
older_than_previous_row_count
selected_row_offset histogram
camera_minus_tactile_ms: p50/p90/p95/p99/max
```

对 visual profile 使用当前 legacy selector 做“现状测量”，但不要把其结果当 target implementation。

### Camera

```text
flag_camera_fresh_false_count
same_depth_frame_number_as_previous_count
same_color_frame_number_as_previous_count
camera_source_delta_ms distribution
camera_age_ms distribution
```

### Coupling

分别计算：

```text
tactile lag | camera fresh
vs
tactile lag | camera not-new
```

用于检验 H2：camera timing transient 是否显著放大 tactile previous-row reuse。

### Episode integrity

输出每条：

```text
source frame count
hard-invalid count
leading-invalid count
internal-invalid count
internal segment count
exportable under current rule
exportable under proposed strict rule
```

**Phase A 完成前不要修改 admission behavior。** 先保存 baseline JSON，后面用于 before/after 对比。

---

## Phase B — Persist camera health and recording-time aligned tactile

### B1. Persist `camera_health`

Raw 当前只持久化 `flag_camera_fresh`，它无法区分：

```text
not-new but still usable
vs
CLOCK_RESET / DELIVERY_DELAY
```

新增 raw provenance field：

```text
camera_health: uint8
```

其值直接来自当前 camera frame header；不要在 recorder 重新分类。

`flag_camera_fresh` 保持当前既有语义，不静默改义。它以后主要用于 audit / runtime progression evidence。

### B2. Extend recording policy-observation signals

Primary owner：

```text
dexmani_real/teleop/control_loop/grid.py
_recording_policy_observation_signals(...)
```

在已有 camera-aligned arm/hand qpos 基础上增加 tactile。

推荐 raw fields：

```text
policy_observation_contact_force                 float64 [5,3]
policy_observation_contact_force_valid           bool
policy_observation_tactile_force                 float64 [5,120,3]
policy_observation_tactile_force_valid           bool
policy_observation_tactile_source_monotonic_ns   uint64
policy_observation_tactile_calibrated            bool
policy_observation_tactile_unit_code             uint8
```

如果 source code 显示更小字段集即可完整证明 processed semantics，可以缩减；但不能只存 payload 而丢掉 source/calibration/unit proof。

### B3. Reuse existing causal primitive

优先复用：

```text
ipc/causal.py::read_structured_frame_aligned_to_source
```

不要再造一套 selector class。

Reference：

```text
reference_ns = camera_frame.source_monotonic_ns
anchor_ns    = recording control-grid anchor
```

Aggregate contact：

1. 从 `hand_state_ring` 找 `source <= camera_reference` 的 newest state；要求 `state_valid`, `!qpos_stale`, `tactile_sum_valid`，payload finite。
2. 从 `hand_tactile_ring` 找同 reference 下 newest calibration/unit provenance；contact-only 不要求 dense `fresh`。
3. 两侧 `source_monotonic_ns` 必须完全相等，匹配 deployment 当前 contract。
4. `camera_reference - source <= max_observation_skew`。

Dense tactile：

- 使用同 source 的 tactile ring row；
- 额外要求 `fresh=True`；
- payload finite；
- calibration/unit valid。

如果 aggregate valid 但 dense invalid，要真实表达：

```text
contact_force_valid = true
tactile_force_valid = false
```

不要再次把两个 validity collapse 成一个 runtime boolean。

### B4. Invalid payload representation

- invalid float payload persist as NaN，不能 zero-fill；
- valid flag 明确为 false；
- source timestamp 保留实际 selected source（若根本无 candidate 则 0）；
- 不用 future sample 修复。

### B5. Schema version

Baseline raw v26。上述 persisted fields 是新语义，推荐 raw schema 递增到 v27。

**不要提供伪造 camera-aligned tactile 的 v26 -> v27 migration。** 缺失的 pre-episode ring history 无法由 v26 file 证明性恢复。

---

## Phase C — Correct camera row classification in cleaner

Primary owner：

```text
dexmani_real/dataset/clean.py
```

### C1. Stop using `flag_camera_fresh` as a direct hard gate

删除当前等价于：

```text
~flag_camera_fresh -> camera_invalid -> hard_invalid
```

的语义。

将 camera 分为 hard 与 audit 两层。

### C2. Hard-invalid camera conditions

一条 visual row 只有出现下面条件才因 camera 被 hard-invalid：

```text
camera source timestamp missing / non-positive
camera source is in the future of observation anchor
camera age > configured max camera/policy budget
camera payload structurally invalid
camera health == CLOCK_RESET
camera health == DELIVERY_DELAY
unknown / malformed camera health code
```

如果现有 camera payload validator 已经覆盖 RGB/depth structural invalidity，复用它，不重复扫描。

### C3. Audit-only camera conditions

以下默认 **KEEP + AUDIT**：

```text
flag_camera_fresh == false
same camera frame reused on adjacent control-grid rows
CameraHealth.DUPLICATE while payload/source remain recent and structurally valid
small frame-number gap followed by a valid current frame
source delta around one/two camera periods
```

建议 audit names：

```text
camera_reused_on_grid
camera_duplicate
camera_frame_gap
camera_timing_jitter
```

不必全部都做；优先最小集合：`camera_reused_on_grid` + `camera_duplicate`。

### C4. `observation_valid` becomes audit for training admission

当前 `observation_valid` 是 teleop recording-health composite，包含 VR/camera 等 runtime 条件。

不要继续：

```text
observation_valid == false -> unconditional hard-invalid
```

改为 audit evidence；真正 hard gate 使用直接 modality conditions：

```text
arm / hand source validity
policy_observation_valid
camera hard validity
required tactile validity
action validity
```

非 visual control-grid profile 的 existing direct source-age gate 可保持。

### C5. Do not modify `CameraFreshnessTracker` runtime stall logic in this task

它仍可用 “new + healthy + recent” 判断是否持续 stall 并在 2 s 级别丢弃 recording episode。

本任务只停止把其 `camera_fresh` 输出误当作 offline row corruption。

这样 runtime safety/health 与 training admission 解耦，风险最小。

---

## Phase D — Remove offline visual tactile row re-selection

对新 raw schema 的 visual profile：

```text
DO NOT call select_tactile_rows_to_references()
```

直接读取 Phase B 持久化的 camera-aligned fields。

### D1. `contact_force`

```text
processed contact_force[t]
= raw policy_observation_contact_force[t]
```

且 `policy_observation_contact_force_valid[t]` 必须为 true。

### D2. `tactile_force`

```text
processed tactile_force[t]
= raw policy_observation_tactile_force[t]
```

当前 processed profile 若仍要求 dense tactile，则 dense valid 必须为 true。

### D3. Processed provenance

直接持久化：

```text
observation_reference_monotonic_ns = camera_source_monotonic_ns
tactile_source_monotonic_ns        = policy_observation_tactile_source_monotonic_ns
```

validator 检查：

```text
0 < tactile_source <= observation_reference
delta <= max_observation_skew
```

不允许通过跨 raw row selection 改写 source identity。

### D4. Remove obsolete repair semantics

对于新 schema，`tactile_forward_fill` / `previous-row repair` 不应再是 canonical processing 机制。

如果 helper 只剩 legacy diagnostic 使用，把它留在 offline analysis tool 或 legacy path；否则删除 dead code。

### D5. Processed schema version

因为 visual `contact_force/tactile_force` 的 persisted alignment semantics 从“offline 16 Hz row re-selection”变为“recording-time high-rate ring camera alignment”，这是语义变化。

Baseline processed v15，推荐递增到 v16。

---

## Phase E — Restore strict whole-episode Policy Zarr admission

Primary owner：

```text
dexmani_real/dataset/export.py
examples/export_policy_zarr.py
```

### E1. Delete the heuristic

删除：

```python
_MAX_TOLERATED_INTERIOR_GAP_ROWS = 2
```

及所有 “1–2 missing rows are okay” 分支和 tests。

### E2. Canonical rule

新-schema artifact：

```text
one processed HDF5
    + zero dropped source rows
    + one source-contiguous sequence
    + valid payload/schema/semantics
=> one Zarr episode
```

否则：

```text
reject the whole HDF5
```

不 split，不 compact，不 bridge。

推荐显式验证：

```text
source_keep_mask is all true
source_row_index == arange(source_frames)
source_segment_ends == [source_frames]
```

具体使用哪一个作为 canonical proof 由现有 `validate_processed_provenance` 的 ownership 决定；不要复制三套等价判定。

### E3. Rejection report remains useful

Whole-episode reject report继续列：

```text
invalid frame count
ranges
root reasons
```

但 report 不改变 admission。

### E4. No legacy exception inside canonical exporter

不要为了 61 条旧数据在新 exporter 里再次加入：

```text
leading frame0 exception
<=2 frame exception
camera exception
```

历史 v26/v15/v8 artifacts 保持 frozen legacy；新 exporter contract 保持简单。

如果论文实验必须继续使用旧 v8 dataset，明确标记其 legacy provenance，不把旧数据伪装为新 schema。

### E5. Policy Zarr schema version

“每条 Zarr episode 必须是完整 source episode”是 persisted dataset semantic invariant。

Baseline v8，推荐递增到 v9，并在 attrs / validator 中明确 source-complete episode contract。

---

## Phase F — Documentation cleanup

实现完成后至少同步：

```text
docs/data_schema.md
docs/invalid_frames_export_incident.md
README.md
repo_map.md              # only where stable boundary/navigation changed
```

### Incident 文档必须修正的表述

旧表述：

```text
frame0 是 startup tactile artifact
camera marked rows 是真实 camera invalid frames
exporter 应 tolerance <=2 interior gaps
```

目标表述：

```text
frame0 historical deficit:
    grid-anchor recording / camera-reference processing mismatch;
    XHand was already fresh+calibrated before BEGIN.

camera transient:
    observed timing/reuse event;
    not automatically training invalid when payload is causal/recent/healthy.

export:
    no interior-gap tolerance;
    true hard-invalid => reject whole processed HDF5.
```

---

## 6. Exact code ownership checklist

Claude Code 开始改代码前按表追 producer -> representation -> consumer。

| Boundary | Primary files | Required action |
|---|---|---|
| XHand timestamp | `robot/hand_worker.py`, `robot/drivers/xhand.py` | **Read-only verification**；不要改 timestamp 语义 |
| Causal ring selection | `ipc/causal.py` | 复用已有 aligned-to-source primitive；只在必要时加一个很窄的 helper |
| Recording visual policy obs | `teleop/control_loop/grid.py` | 增加 camera-aligned contact/dense tactile signals |
| Raw signal construction | `teleop/episode_samples.py`, `recording/frame.py`, `recording/sample.py` | 透传 aligned tactile + `camera_health`；invalid float payload 用 NaN |
| Raw schema | `recording/storage/schema.py` + writer/reader/tests | 新 fields + schema bump |
| Camera acquisition | `sensor/camera/worker.py`, `sensor/camera/realsense.py` | 行为原则上不改；只确认 health semantics |
| Cleaner | `dataset/clean.py` | camera hard/audit 分层；visual tactile direct consume；`observation_valid` 降为 audit |
| Processed writer | `dataset/processing.py` | 写 recording-time aligned tactile，更新 provenance |
| Processed validator | `dataset/processed.py` | 验证新 tactile source/reference semantics |
| Exporter | `dataset/export.py` | 删除 gap tolerance，whole-HDF5 strict admission |
| CLI | `examples/process_episodes.py`, `examples/export_policy_zarr.py` | 保持 thin；只更新报告/文案/版本 |
| Diagnostics | `tools/analyze_camera_tactile_alignment.py` | 新增纯离线 read-only baseline tool |

不要因为表中列出了文件就机械修改；只有当前 call path 真正需要时才改。

---

## 7. Required tests

优先 pure / synthetic / offline。任何测试不得连接硬件。

## 7.1 Recording-time tactile alignment tests

至少覆盖：

### T1. frame-0 pre-existing tactile history is selected

Synthetic ring：

```text
tactile sources: 983 ms, 1016 ms
camera source:   1000 ms
grid anchor:     1025 ms
```

期待：

```text
selected = 983 ms
1016 ms MUST NOT be selected
frame0 contact/tactile policy observation valid
```

这是本 bug 的核心 regression test。

### T2. No causal candidate

```text
all tactile sources > camera source
```

期待 invalid；不得 future fill。

### T3. Skew boundary

candidate older than `max_observation_skew` => invalid。

### T4. Aggregate/provenance source mismatch

`hand_state.tactile_sum` 与 `hand_tactile provenance` source 不同 => contact invalid。

### T5. Aggregate valid, dense invalid

必须能表达：

```text
contact valid = true
dense valid   = false
```

即使当前 processed profile 后续仍选择因 dense requirement 拒绝，也不能在 recording 层丢失这个事实。

---

## 7.2 Camera classification tests

### C1. Same recent frame reused on adjacent grid row

```text
same frame number
same source timestamp
age < max_age
health = OK
payload valid
```

期待：

```text
KEEP
audit camera_reused_on_grid
NOT hard-invalid
```

### C2. Duplicate health, still recent

若 `CameraHealth.DUPLICATE` payload/time 仍合理：默认 KEEP + AUDIT。

### C3. Clock reset

`CLOCK_RESET` => hard-invalid。

### C4. Delivery delay / stale

`age > budget` 或 `DELIVERY_DELAY` => hard-invalid。

### C5. Frame-number gap followed by healthy current frame

current frame valid => KEEP；gap 只 audit。

---

## 7.3 Cleaner / processing tests

### P1. Visual row uses exact persisted policy tactile

构造 raw：same-row grid tactile 在 camera future，但 policy-observation tactile 在 camera past。

期待 processed 使用后者，不调用 previous raw row repair。

### P2. `observation_valid=False` alone does not delete otherwise valid training row

用于防止 composite runtime health 再次污染 offline admission。

### P3. True modality invalid still deletes/rejects

例如 camera stale、policy observation qpos invalid、tactile required but no causal source。

---

## 7.4 Whole-episode exporter tests

原 `TestWholeEpisodeGapTolerance` 应替换为 strict tests。

### E1. Fully complete episode

accept。

### E2. One interior source row removed

reject whole HDF5。

### E3. Two interior rows removed

reject whole HDF5。

### E4. Long IK gap

reject whole HDF5。

### E5. No splitting

验证 `episode_ends` 数量 == accepted HDF5 数量，绝不会因一个 HDF5 多 segment 而增加 Zarr episode 数。

### E6. Dropped leading/suffix row in new schema

canonical new-schema export reject；防止未来重新出现“特殊 prefix tolerance”侵入主路径。

---

## 7.5 End-to-end synthetic test

建立一个最小 synthetic raw current-schema episode：

```text
N >= horizon
frame0 has pre-start camera-causal tactile
one middle control tick reuses recent camera frame
all actions/states valid
```

期待：

```text
raw -> processed:
    zero hard-invalid rows
    no source gap

processed -> Zarr:
    exactly one episode
    episode length unchanged
```

再建立相同 fixture，但中间插入 persistent IK hard-invalid：

```text
processed may exist for audit
export rejects entire HDF5
Zarr gets zero episode from it
```

---

## 8. Historical 61-episode experiment

这是**诊断实验**，不是新 schema correctness test。

## 8.1 Baseline

先用 Phase A tool 保存：

```text
artifacts/camera_tactile_alignment_baseline.json
```

或用户指定的非 tracked output path。

至少复现 incident 的已知 totals，确保 tool 没有读错数据。

## 8.2 Tactile lag experiment

对每个 visual row，使用 legacy selector 计算：

```text
row_offset = selected_tactile_row - current_row
lag_ms     = camera_source - selected_tactile_source
```

汇总：

```text
same row %
previous row %
<= -2 row %
lag p50/p90/p95/p99/max
```

重点回答：

> 历史数据是否只有 frame0 bookkeeping deficit，还是 visual tactile 在大量后续 row 也比 deployment 可获得的 high-rate tactile 更旧？

注意：历史 raw 没有 high-rate ring history，无法直接计算“deployment would have selected X”；这里只能量化 persisted-row selector 的 extra staleness proxy。

## 8.3 Camera/tactile coupling

比较：

```text
lag distribution where flag_camera_fresh=True
lag distribution where flag_camera_fresh=False
```

若 not-new rows lag 明显更大，支持“camera timing transient 放大 offline tactile lag”的机制解释。

## 8.4 Proposed camera rule simulation

不改文件，只模拟：

```text
camera not-new but age valid -> audit keep
clock reset / stale / malformed -> hard
```

输出：

```text
how many historical camera rows stop being hard-invalid
how many episodes lose internal gaps solely caused by benign camera timing
```

## 8.5 Do not fabricate legacy frame0 tactile

对 v26 frame0，没有 pre-episode ring history就无法证明一个 camera-causal tactile payload。

禁止：

```text
copy row0 tactile backward in time
use row1 future tactile
set zero as no-contact
change source timestamp only
```

旧 v8 dataset 若继续用于旧实验，应被标为 legacy contract；不要重新发布成新 v9。

---

## 9. Hardware experiment protocol

**Claude Code 只实现必要 telemetry / test entry，不自动运行。**

## 9.1 Camera illumination experiment

目标：区分“dark exposure effect”和一般 host/USB timing jitter。

三组：

```text
A. normal light + AE priority OFF
B. dark light   + AE priority OFF
C. dark light   + AE priority ON
```

每组建议 2–3 min 静态采集，不控制机器人运动。

记录：

```text
depth/color frame number
device timestamp delta
host receive delta
source->receive backlog
camera health
actual exposure metadata (if SDK frame exposes it)
gain metadata (if available)
```

不为这个实验把 exposure/gain 加进 production raw schema；优先在专用 diagnostic example 中打印/CSV。

判断：

- C 若稳定出现约 60 ms exposure / ~16.7 Hz，复现已知 AE priority 行为；
- B 若仍有少量 host-delivery jitter但 source cadence约 30 Hz，说明 current transient 不应笼统叫“坏帧”；
- 任何结论必须区分 device cadence 与 host delivery latency。

## 9.2 New recording tactile alignment experiment

在新 schema 上录制至少 10 条短 episode（正常无接触 + 轻接触均可），检查：

```text
frame0 policy_observation_contact_force_valid
frame0 policy_observation_tactile_force_valid
camera_source - tactile_source lag distribution
source match between aggregate/provenance/dense where required
```

硬验收：

```text
healthy BEGIN 后，正常采集不应系统性出现 frame0 no-causal-tactile
all selected tactile source <= camera source
all lag <= configured max_observation_skew
```

若仍出现 frame0 invalid，先查 ring capacity / source history / selection implementation，不要重新增加 exporter tolerance。

---

## 10. Acceptance criteria

任务只有同时满足以下条件才算完成。

### Code / architecture

- [ ] `_MAX_TOLERATED_INTERIOR_GAP_ROWS` 及等价 interior-gap allowance 已从 canonical exporter 删除。
- [ ] 一个 processed HDF5 不会被拆成多个 Zarr episode。
- [ ] 新 canonical exporter 不 bridge 任意 internal source gap。
- [ ] `flag_camera_fresh=False` 不再单独导致 offline hard-invalid。
- [ ] `observation_valid=False` 不再单独导致 offline hard-invalid。
- [ ] visual tactile 在 recording 时按 camera source 从高频 ring 选择。
- [ ] visual processing 不再从 16 Hz raw rows跨行重选 tactile。
- [ ] no future tactile / no zero repair / no timestamp fabrication。
- [ ] XHand host-read-completion timestamp semantics 未被伪装成 sensor acquisition time。

### Tests

- [ ] frame0 pre-existing tactile-history synthetic regression test passes。
- [ ] camera reused recent frame KEEP+AUDIT test passes。
- [ ] clock reset / stale camera hard-invalid tests pass。
- [ ] one-row interior gap whole-HDF5 rejection test passes。
- [ ] end-to-end complete episode -> exactly one Zarr episode test passes。
- [ ] end-to-end hard-invalid interior row -> zero Zarr episode test passes。

### Offline validation

- [ ] Phase A historical baseline generated before behavior edit。
- [ ] before/after proposed camera classification numbers recorded。
- [ ] historical tactile row-offset / lag distribution recorded。
- [ ] documentation no longer claims historical camera transients are proven exposure failures or generic bad frames。

### Standard repository validation

至少：

```bash
python -m compileall -q dexmani_real examples tools
git diff --check
git diff --stat
git status --short
```

再运行与本改动直接相关的 focused tests。不要用硬件 example 代替 tests。

---

## 11. Claude Code execution order

严格按下列顺序，减少返工和 attribution ambiguity。

### Step 0 — Preserve worktree

```bash
git status --short
```

读取：

```text
AGENTS.md
CLAUDE.md
code_style.md
```

保留任何 unrelated user changes。

### Step 1 — Baseline only

实现/read-run offline alignment diagnostic，保存 baseline；此 commit 不改 runtime/data semantics。

### Step 2 — Recording boundary

完成：

```text
camera_health persistence
camera-aligned tactile policy observation
raw schema bump
producer/writer/reader tests
```

此阶段不要动 exporter。

### Step 3 — Processing boundary

完成：

```text
camera hard/audit split
observation_valid audit downgrade
visual tactile direct consumption
processed provenance/schema bump
```

跑 synthetic processing tests。

### Step 4 — Export boundary

最后删除 gap tolerance，恢复 strict whole-episode admission；更新 export tests / Zarr schema contract。

### Step 5 — Evidence and docs

重新运行 Phase A tool 的 proposed-rule simulation / focused tests，更新：

```text
invalid_frames_export_incident.md
data_schema.md
README.md
repo_map.md
```

在本文件末尾追加一个简短 `Implementation evidence` 小节，写 commit、tests、未执行 hardware checks；不要复制整份 test log。

---

## 12. Failure handling / rollback rules

遇到问题时不要退回 gap tolerance。

### Case A. New frame0 tactile still invalid

检查顺序：

```text
1. camera reference 是否正确
2. pre-camera tactile 是否仍在 ring capacity 中
3. aggregate/provenance source match 是否过严/实现错误
4. calibration/unit/dense freshness 哪一项失败
5. max_observation_skew 是否真被超过
```

只有实测证明正常硬件的真实 causal skew 超过当前预算，才单独讨论 skew config；不能直接用 future tactile。

### Case B. Camera reused rows数量较多

先看：

```text
camera age
camera health
source cadence
host delivery cadence
```

只要 recent + structurally valid，就保持 audit；不要用“重复率高”反推 corruption。

### Case C. Strict exporter 导致很多 episode 被拒

这是 upstream classifier 的 signal，不是 exporter bug。

逐 reason 检查 false positive；不要恢复：

```text
<=1 gap
<=2 gap
percentage missing threshold
```

### Case D. Historical v26 cannot pass new contract

这是预期的 provenance limitation。保持 legacy artifacts；不要伪造 v27 fields。

---

## 13. Recommended minimal research experiment matrix

若时间有限，只做下面四项即可判断修复是否有研究价值：

| Exp | Data | Change | Measure | Purpose |
|---|---|---|---|---|
| E0 | historical 61 | current | tactile row offset / lag + camera reuse | baseline |
| E1 | historical 61 | proposed camera classifier simulation | hard rows / affected episodes | quantify false-positive camera rejection |
| E2 | new 10 episodes | recording-time camera-aligned tactile | frame0 valid rate + lag | verify root-cause fix |
| E3 | synthetic exporter fixtures | strict whole-episode rule | accept/reject + episode count | prove no gap compaction/splitting |

不要在这轮先跑 policy training ablation。只有 E0–E3 通过后，若发现 historical/new tactile lag差异显著，再做：

```text
legacy alignment vs corrected alignment
same model / seed / train steps
real success rate
```

否则先把 engineering correctness 完成即可。

---

## 14. Chat-reading summary

后续在聊天中快速恢复上下文时，只需要读这一节：

1. **Zarr contract**：一个 processed HDF5 最多一个 Zarr episode；true hard-invalid internal row => whole HDF5 reject；不 split、不 bridge、不 `<=2 gap tolerance`。
2. **Camera bug**：`flag_camera_fresh` 混合 “new” 与 “usable”。短时同帧 reuse / duplicate / 33–67 ms timing variation 在 payload recent/causal 时应 KEEP + AUDIT，不应自动删 row。Clock reset、真正 stale、payload corruption 才 hard。
3. **Tactile frame0 root cause**：BEGIN 前 tactile 已 fresh+calibrated；不是 startup failure。问题是 recording 用 grid anchor `T` 取 tactile，而 visual processing 后改用 camera source `C`，通常 `C < H < T`。frame0 之前的 ring history未持久化，所以 offline selector无 row -1 可用。
4. **Tactile fix**：在 recording 时就从 high-rate hand/tactile ring 按 camera source选择 newest `source <= C`，与 deployment 完全一致；raw 显式保存 camera-aligned contact/dense tactile + provenance；processing 直接 consume，不再跨 16 Hz raw rows repair。
5. **Timestamp rule**：XHand timestamp 仍是 host read-completion 近似，不伪称 hardware acquisition timestamp；不 interpolation、不 future fill。
6. **Legacy data**：v26 无法证明性恢复 pre-episode tactile history；不伪造 migration。历史 61 条只做 read-only lag/camera-reuse analysis，旧 v8 artifact 需要时保持 legacy 标记。
7. **Execution order**：先 baseline diagnostic，再 recording/schema，再 cleaner/processed，最后 exporter strictness；这样不会把 upstream false positives 放大成数据大面积丢失。

---

## 15. Implementation evidence

> 由 Claude Code 在实际实现完成后填写。保持简短。

```text
Baseline commit:
Implementation commits:
Schema versions:
Focused tests:
Historical diagnostic result:
Hardware validation performed: yes/no
Unperformed checks / remaining hypotheses:
```
