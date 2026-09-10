# invalid_frames_report 大批量标记与 Zarr 导出准入过严复盘

日期：2026-09-09。范围：`pick_place_toy` 任务 61 条 episode 的离线清洗产物
（`episodes_processed/pick_place_toy/`）、`process_log/invalid_frames_report.json`、
Policy Zarr 导出准入（`dexmani_real/dataset/export.py`）。全程离线分析，未连接硬件。

> **历史说明（2026-09-09）**：下文保留当时的 camera-master / row-cleaning
> 调查，作为 incident forensic record；它不是当前架构规范。当前规范是
> [`control_step_dataset_simplification_plan.md`](control_step_dataset_simplification_plan.md)：
> control step 是唯一 timeline，accepted episode 保留全部 source rows，camera/tactile
> 不再通过 processed provenance 做跨行修复。下文提到的旧 schema、selector 和 gap
> machinery 均为历史行为。

## 修正后的根因

1. **frame0 触觉缺口**不是 XHand 启动伪影：BEGIN 前代码已要求 tactile `fresh + calibrated +
   recent`。真正根因是 recording 用 control-grid anchor `T` 取 tactile，而 visual processing
   后来改用 camera source `C` 作 reference，通常 `C < H < T`；episode 开始前高频 tactile ring
   history 未持久化进 raw，offline 无 row -1 可恢复。修复是在 recording 时直接按 camera
   source 从高频 ring 选择并持久化 camera-aligned tactile。
2. **中段相机标记**是 timing/reuse 事件，不是「真实坏帧」：`flag_camera_fresh=False` 可能只是
   该 16 Hz grid tick 复用了仍然 recent 的上一 camera frame。payload 合法、source 因果、age 在
   预算内、无 clock reset 时保留为 audit，只有 clock reset / delivery delay / 真正 stale /
   payload 损坏才是 hard-invalid。
3. **导出**不再做 `<=2` 内部缺口容忍：true hard-invalid 内部行 => 整份 processed HDF5 拒绝。

## 1. 范围与结论

调试数据集导出时观察到以下现象：

- `invalid_frames_report.json` 标记了 61 条 episode 中的 43 条，直观上像"大批数据被判非法"；
- 其中约 9 成 episode 的无效帧区间恒为 `[0, 1]`（半开区间，即仅第 0 帧），原因对恒为
  `tactile_invalid` + `nonfinite_real_modality`；
- 少量 episode 在随机中段位置出现 1–2 帧 `observation_invalid` + `camera_invalid`；
- 一条 episode（`episode_20260827_224527`）额外出现 19 帧 `long_ik_failure_hold`；
- 按旧导出规则 dry-run，`Exported 18/61 episode(s); rejected 43`——只有零丢帧的
  18 条能进入 Policy Zarr。

结论（全部经代码追踪 + 对 61 个 processed HDF5 的只读实测 + 对抗复核确认）：

1. `invalid_frames_report.json` 是**审计日志，不是淘汰名单**。清洗器接受了全部 61 条
   episode（0 条拒绝），整个数据集只删除了 70/14309 个 source 行（0.49%）。
2. 帧 0 标记是**结构性 grid-reference / camera-reference 对齐缺口，数据本身健康**：相机
   曝光时间戳系统性早于触觉采样时间戳，帧 0 没有更早的触觉行可供因果选择。不是传感器
   故障、不是 XHand 启动伪影（BEGIN 前 tactile 已 fresh+calibrated）。
3. 中段相机标记是**观测到的 timing/reuse 事件**：`flag_camera_fresh=False` 可能只是复用了
   仍然 recent 的 camera frame；payload causal/recent/healthy 时不应自动当作训练损坏。
4. `long_ik_failure_hold` 是**唯一真正的数据质量事件**（机械臂逼近 workspace 边缘导致
   IK 持续失败），检测正确，该 episode 应当整条拒绝。
5. 真正需要修复的缺陷在**导出端**：旧 `_whole_episode_rejection` 把"存在任何被删除的
   source 行"一律整条拒绝，使 35 条仅丢失帧 0 的 episode 陪葬（57% 数据损失）。当时
   改为缺口容忍规则（§4），dry-run 实测 `Exported 60/61`；该容忍规则现已**回退**为严格
   whole-episode 准入（见顶部修正），帧 0 缺口改为在录制端根治。

## 2. 现场证据

### 2.1 报告模式分解（43 条被标记 episode）

| 模式 | 数量 | 帧区间 | 原因对 | 定性 |
| --- | --- | --- | --- | --- |
| 仅帧 0 | 35 | 恒 `[0,1]` | `tactile_invalid` + `nonfinite_real_modality` | 良性结构伪影 |
| 帧 0 + 中段相机 | 5 | `[0,1]` + 随机中段 1–2 帧 | 上述 + `observation_invalid` + `camera_invalid` | 伪影 + 真实瞬态 |
| 仅中段相机 | 2（`220112`、`223607`） | 随机中段 1 帧 | `observation_invalid` + `camera_invalid` | 真实瞬态 |
| IK 长保持 | 1（`224527`） | `[140,159]` 19 帧 + 帧 0 + 1 相机帧 | 上述 + `long_ik_failure_hold` | 真实异常 |

桶合计 18（零丢帧）+ 35 + 5 + 2 + 1 = 61，与 dry-run 准入结果一一对应。

### 2.2 关键测量

| 现象 | 可以确认的事实 | 不能单独推出的结论 |
| --- | --- | --- |
| 帧 0 双原因恒成对 | raw 帧 0 触觉全零且有限、`tactile_fresh/calibrated=True`；41 条受影响 episode 的帧 0 触觉时间戳比相机曝光晚 0.38–41.1 ms（均值 16.3 ms，无一为负） | 不是传感器死值（XHand 无接触时输出精确零而非 NaN），不是某日硬件故障（2026-08-27 与 2026-09-08 的 episode 模式一致） |
| 2 条 episode 无帧 0 标记 | 其帧 0 触觉时间戳恰好**早于**相机 10.7 / 3.0 ms，因果选择器可选 | 帧 0 标记并非"必然"，而是时序偏置下的高概率事件 |
| 中段相机标记帧 | raw 中该帧 `flag_camera_fresh=False`、`observation_valid=False`；全部 9 个受影响帧相对前一有效帧的相机 source 时间戳间隔实测 33.8–67.7 ms（≈1–2 个标称帧间隔），远小于 250 ms 陈旧度上限，相邻帧全部正常 | 是"这一帧不是新帧"（重复/延迟），不是相机长时间失效 |
| `224527` 19 帧保持 | `flag_frame_status=2`（IK 失败）连续 19 帧 > 瞬态阈值 4；tracking_error 峰值 0.235 rad 后衰减；观测时间戳最大间隔 1266.3 ms（其余 60 条中位数 66.6 ms） | 保持本身说明安全机制在正确地"不动"，不是失控 |
| 旧规则 18/61 | `tactile_invalid`、`nonfinite_real_modality` 等全部丢弃原因都属于 hard-invalid（`clean.py` `hard_invalid_reason_names` = 除 `annotation_excluded_row` 外全部） | 18/61 不代表 43 条数据质量差——其中 35 条只缺第 0 帧 |

### 2.3 数据量核对

- 全部 61 条 processed HDF5 均在（cleaner 零拒绝）；raw 与 processed 目录 1:1 对应。
- 总丢帧 70/14309（0.49%）；单条最大损失为 `224527` 的 21/197（10.7%），
  其 `source_segment_ends=[132,138,176]`（三段 132+6+38=176 行）。
- 典型 episode 中位长度 225 帧，丢帧 0 约 0.4%。

## 3. 根因分析

### 3.1 帧 0 `tactile_invalid` + `nonfinite_real_modality`（结构性，良性）

清洗器的触觉对齐合同（`clean.py` `select_tactile_rows_to_references`）要求：对视觉
profile，以相机曝光时间戳（`camera_source_monotonic_ns`）为 reference，只能选择
**source 不晚于 reference**、skew 有界（≤ `max_observation_skew_s`=100 ms）、
fresh + calibrated + unit_code 0 + hand-source 匹配的触觉行（`searchsorted side="right"`）。

录制端两个 worker 的打戳事件本质不同：相机按**设备曝光时刻**（经
`sensor/camera/clock_sync.py` 映射到主机单调钟），手部按**主机读取时刻**
（`hand_worker`），中间隔着 USB 传输延迟与控制网格的读取顺序（先相机后触觉）。
因此触觉时间戳系统性晚于同帧相机曝光 0.4–41 ms。帧 1 起可以借用上一帧的触觉行
（skew 在 100 ms 内）；唯独帧 0 没有更早的行可借，选择器返回 -1：

```text
tactile_source_rows[0] = -1
  → tactile_valid[0] = False                      （tactile_invalid）
  → aligned_contact[0] 保持 NaN 初始化
  → real_modalities_finite[0] = False             （nonfinite_real_modality）
```

两个原因恒成对，因为它们是同一事件的因果链（丢弃位 1280 = 2⁸ + 2¹⁰）。
`docs/raw_v24_migration.md` 早在 v24→v25 重构前就把"source row [0,1) 有
tactile_invalid 和 nonfinite_real_modality"记录为已知预期行为。

### 3.2 中段 `camera_invalid` + `observation_invalid`（timing/reuse 事件）

RealSense 偶发重复帧/时钟回跳/投递延迟，`sensor/camera/worker.py` 的
`_camera_health()` 将其分类为非 OK，`teleop/control_loop/camera_freshness.py` 的
`CameraFreshnessTracker` 据此置 `flag_camera_fresh=False`；录制端 `observation_valid` 把相机新鲜度列为必需源
（`episode_samples.py`），因此两个原因成对出现。

**修正**：这些是 timing/reuse 事件，不是自动的训练损坏。`flag_camera_fresh=False` 可能只是
该 16 Hz grid tick 复用了仍然 recent 的上一 camera frame（9 个受影响帧相对前一有效 camera
source 的间隔仅 33.8–67.7 ms，远小于 250 ms 陈旧上限）。修复后 offline camera hard-invalid
只看 source>0、source<=anchor、age<=budget、health 合法且非 CLOCK_RESET/DELIVERY_DELAY；
`flag_camera_fresh` 与 `observation_valid` 降为 audit。

### 3.3 `long_ik_failure_hold`（真实异常，检测正确）

`224527` 在帧 140–158 连续 19 帧 IK 求解失败（`frame_status=2`），远超瞬态阈值
`_MAX_TRANSIENT_IK_HOLD_FRAMES=4`（`clean.py`），被分类为 persistent 并整段删除。
直接诱因是机械臂逼近 workspace 边缘（操作者确认）；部署侧 coordinator 此后已接入
planner workspace 检查，此类 episode 预期减少。这是 43 条中唯一应当整条拒绝的
轨迹：19 帧缺口使压紧数组断成 3 段，若导出，训练窗口跨缺口会把相隔 1.27 s 的
状态当作相邻时间步。

### 3.4 导出端过度拒绝（本次修复的真正缺陷）

旧 `_whole_episode_rejection`（`export.py`）在两个独立触发器上任一成立即整条拒绝：

1. 任何 source 行带 hard-invalid 丢弃位（无首行豁免、无数量阈值）；
2. `len(segment_ends) != 1`（任何内部断段）。

由于 §3.1 的帧 0 丢弃在所有 profile 下都是 hard-invalid，触发器 1 把 35 条只缺帧 0
的良性 episode 全部拒绝；触发器 2 再叠加拒绝 8 条中段缺口 episode。该规则由
commit `fd7d275`（"0828 0 fix dataset export"）引入，用整条拒绝替代了旧版"按段拆分
为多条训练 episode"的行为，但帧 0 副作用没有任何文档记载——与
`docs/raw_v24_migration.md` 已知的帧 0 预期行为相矛盾。

需要强调：帧 0 丢弃**不产生**内部断段（保留行从 source row 1 起连续），对训练
语义零影响——episode 只是晚一个 dt 开始；而中段缺口会在导出的连续数组中制造
相邻行实际相隔 2–3 个 dt 的伪转移（Policy Zarr v7 只有 `data/*` 与
`meta/episode_ends`，没有任何字段能表达集内不连续）。旧规则没有区分这两种情况。

## 4. 修复措施

### 4.1 导出准入规则（已回退为严格 whole-episode）

> 本节描述的 `<=2` 内部缺口容忍规则已**回退**。当前 `_whole_episode_rejection`
> （`export.py`，`POLICY_ZARR_SCHEMA_VERSION = 9`）只接受保留全部 source 行且 source 序列
> 连续的 processed HDF5；`_MAX_TOLERATED_INTERIOR_GAP_ROWS` 已删除。任何 source 行删除、
> 内部缺口或时间/样本跳变整条拒绝，不拆分、不压紧、不桥接。下面是当时的容忍规则，仅作历史记录。

`_whole_episode_rejection` 当时改为纯结构化判定（`export.py`，常量
`_MAX_TOLERATED_INTERIOR_GAP_ROWS = 2`）：

- **首部裁剪恒容忍**：第一个保留行之前的任何丢弃（含帧 0 伪影）不破坏保留行连续性；
- **内部段边界容忍**，当且仅当同时满足：
  1. 缺失连续 source 行 ≤ 2（`row_delta - 1 <= 2`）；
  2. 样本索引随行同步推进（`sample_delta == row_delta`）；
  3. source 时间差与缺失网格步吻合：`|ts_delta - row_delta·dt| <= source_contiguity_tolerance_s`；
- 任一边界不满足即**整条拒绝**（不在缺口处拆分的行为保持不变）；
- hard-invalid 位计算保留，仅用于生成拒绝报告（reasons/counts/ranges），不再作为
  独立拒绝触发器。

### 4.2 为什么去掉 hard-invalid 触发器仍然 fail-closed

`validate_processed_provenance`（`processed.py`）在拒绝判定**之前**运行并以异常
拦截非法输入，其不变量保证：`source_rows == flatnonzero(keep_mask)`、
`segment_ends` 与 (行/样本/时间戳) 三种不连续谓词严格一致。因此每个非首部的
丢弃行**必然**表现为某个内部边界的 `row_delta > 1`，不可能绕过缺口检查；无丢行的
纯时间戳断段因条件 3 不满足仍被拒绝；样本失步因条件 2 被拒绝。新规则拒绝的
情形是旧规则的严格子集加上结构化豁免，豁免范围由用户明确批准（≤2 行瞬态）。

### 4.3 配套修改

- `clean.py`：处理期写入 `source_decision_json` 的警告
  `"policy export will reject this episode: ..."` 在容忍缺口下是错误预测，改为
  非预测措辞（`"N source discontinuity boundary(s); policy export applies its own
  gap tolerance"`）。容忍规则的唯一 owner 是导出端，cleaner 不复制该判定；
  `_source_gap_findings` docstring 同步。
- 文档：`README.md` 3 处、`docs/data_schema.md` 2 处的准入描述更新为新规则。
- 测试：`tests/test_zarr_v7_projection.py` 新增 `TestWholeEpisodeGapTolerance`
  7 个合成 provenance 单测，覆盖连续/首裁/1–2 行容忍/3 行拒绝/纯时间戳断段拒绝/
  样本失步拒绝/拒绝报告内容。

## 5. 验证

- `python -m compileall -q dexmani_real examples tests` 通过；
- `tests/test_zarr_v7_projection.py` + `tests/test_processed_v14.py` 共 41 passed；
- 真实 CLI dry-run（只读）：`Exported 60/61 episode(s); rejected 1`，唯一拒绝为
  `episode_20260827_224527`（21 无效帧、5 原因 + `source_discontinuity`），
  拒绝输出与修改前逐字一致；
- 三视角对抗评审（正确性/合同/文档，7 agent）：0 blocker；确认的 2 major + 2 minor
  （clean.py 警告、测试缺口、README 措辞一致性）均已修复；
- 独立复算：61 条 episode 的边界谓词逐条重算，60/1 判决与实测一致。

未验证项：Zarr 实际写入未执行（仅 dry-run）；新规则对其他任务/其他日期数据的
准入分布未测量。

## 6. 当时的权衡与回退路径（已被当前合同取代）

- 7 条相机缺口 episode 共 7 个中段缺口（合计 9 个被删帧）进入训练数据：每个缺口使
  跨缺口窗口含一个 2–3 dt 的伪动作转移，全数据集帧污染率约 0.75%。这是知情接受的权衡；若训练指标
  异常，回退选项为 `_MAX_TOLERATED_INTERIOR_GAP_ROWS = 1`（额外拒 `172305`、
  `175240` 两条 2 帧缺口 episode，保 58/61）或完全恢复内部断段拒绝（保 53/61）。
- 存量 61 个 processed HDF5 的 `source_decision_json` 中仍是旧警告文本（数据而非
  代码），重跑 `examples/process_episodes.py` 后刷新；不影响导出准入。
- 帧 0 丢弃本身未改动（清洗端因果合同保持原样）：导出侧豁免比放松清洗合同更小、
  且不触碰 raw→processed 语义。

## 7. 当时相关文件（历史职责，非当前 API）

| 文件 | 角色 |
| --- | --- |
| [dataset/export.py](../dexmani_real/dataset/export.py) | `_whole_episode_rejection` 新准入规则与常量 |
| `dataset/clean.py`（已删除） | 当时的帧无效原因位、段边界审计、处理期警告 |
| [dataset/processed.py](../dexmani_real/dataset/processed.py) | `ProcessedProvenance` 不变量与连续性校验 |
| [examples/export_policy_zarr.py](../examples/export_policy_zarr.py) | 导出 CLI（`--dry-run` 只读预检） |
| `tests/test_zarr_v7_projection.py`（历史文件） | v7 投影回归 + 缺口容忍单测 |
| [docs/data_schema.md](data_schema.md) | 当时为 v14/v7；现已更新为当前合同 |
| [README.md](../README.md) | 导出工作流描述（已同步） |
| `episodes_processed/pick_place_toy/process_log/invalid_frames_report.json` | 本次现象来源（审计日志，数据文件） |

## 8. 保留的历史 forensic evidence（旧分析工具已移除）

旧基线 artifact 的完整统计已在删除前核对并压缩记录如下（原 artifact 含机器路径，
不再作为 tracked evidence 保留）：

- 61 条 episode；frame-0 causal tactile deficit 41 条；same-row / previous-row /
  older-than-previous 选择数为 `6078 / 8226 / 5`。
- camera-minus-tactile lag（14268 个有效样本）的 p50/p95/p99/max 为
  `38.2799635 / 64.25902655 / 66.41135582 / 98.793855 ms`。
- `flag_camera_fresh=False` 共 10 行；camera source delta p50/p95/max 为
  `66.612992 / 66.6796408 / 88.235688 ms`；camera age p50/p95/max 为
  `20.7844 / 35.9608084 / 58.951321 ms`。
- camera fresh 与 not-new 两类的 tactile lag p50 分别为 `38.2799635` 与
  `38.645827 ms`，不能据此把 not-new 事件当作系统性 tactile failure；旧规则的
  camera hard rows 为 10，按 causal/age/health 规则重分类后为 0。
- 旧导出模拟为 current/strict `60 / 18` 条可导出 episode，内部相机缺口 episode
  共 7 条。这些数字解释 incident，但不构成当前 admission logic。

补充可定位的旧统计：previous-row 占比约 57.5%；旧 hard-invalid 共 43 行。
depth/color consecutive frame-number reuse 均为 0；fresh/not-new lag 样本数分别为
14258/10。五个 older-than-previous 选择分别出现在 `episode_20260827_194525`（1 行）
与 `episode_20260827_203113`（4 行）。七条 gap episode 为同日的
`172305, 175240, 220112, 220747, 220919, 223607, 223729`。
这些 camera/tactile 差异不能证明旧数据暗场曝光是根因；该归因没有本次硬件证据。

旧计划的实现记录也已转移到本 incident：baseline commit 为
`b5128927d920cca09dde40d910bcde037ad70ba7`；当时 focused tests 覆盖
`tests/test_recording_tactile_alignment.py`、`tests/test_processed_v15.py`、
`tests/test_zarr_v8_projection.py::TestWholeEpisodeStrictAdmission` 与
`tests/test_strict_export_end_to_end.py`，记录为 full pytest `417 passed`，硬件未连接。
历史实现涉及 raw `v26→v27`、processed `v15→v16`、Policy Zarr `v8→v9`；这些版本号
只用于解释旧 artifact，不能伪装成当前 schema。
相关历史实现 commits：`b52b9de`、`e8dd0e7`、`18690af`、`a8b2e32`、`e76b8a3`、
`59e880a`。冻结 migration scripts 仍保留；旧分析工具、基线 JSON 与 camera-master
计划已删除，其独有证据记录在本节而不是保留第二套公开分析 API。

## 9. 当前 control-step salvage evidence（2026-09-10）

当前离线 salvage 保持 `episodes/pick_place_toy/` 原始文件 bitwise 不变：源数据是
60 条 converted-v24 与 1 条 direct-v25，明确 lineage 的 frozen out-of-place
`v25→v26` 转换结果位于 `episodes/salvage_v26/pick_place_toy`；未依据 payload magnitude
猜 tactile scale，也未对原始文件做 in-place migration。当前 schema28 reader 不声明
对 original v25 的 physical replay 支持。

实际结果为：61 source episodes，60 accepted，1 rejected；唯一拒绝
`episode_20260827_224527`，原因为 persistent IK failure；source frames `14309`，
accepted retained frames `14112`，Policy Zarr episodes `60`。每个 accepted processed
episode 均满足 `processed_steps == source_frames`，每个 Zarr episode 对应一个 processed
file，且 retained row 未做 compact/repair。
[manifest](../artifacts/pick_place_toy_salvage_manifest.json) 记录 episode identity、schema、
counts、lineage、admission 与验证摘要，不含机器绝对路径。

执行 baseline 为 `6d9655fdc2f774f0f5eb0613cea0cbcca8816eb6`。原始 183 个文件的 size/SHA-256
清单在转换前、salvage 后与最终检查时一致；清单 fingerprint 为
`a0a4679248a8c345b37d648f3348ded1526f9e8afebede645b7811f3112a1c22`。
实际源数据全部为 v25；60 条显式 converted_from_schema=24，1 条 direct-v25。
v24 引入 commit `c195991d88b6b63db85b6fc2b611515ecfce5d83` 的 driver 已使用 0.1 scale，
至 `c7dced05d55d9ae148a19f94c2aa81ee719aa903` 替换 v24 期间没有 scale 变化；冻结
v24→v25 按原值复制 tactile。SDK-native scale 恢复于之后的
`aac23d27e1fafcd264a24c122f9acf4d37962040`。因此本次只在新的 v26 working copies
执行一次 *10，其他数组保持一致，绝未对已恢复的 v26 再次缩放。

新 processed v17 位于 `episodes_processed/salvage_v17/pick_place_toy`，Zarr v10 位于
`datasets/salvage_v10/pick_place_toy.zarr`；均未覆盖旧产物。所有 accepted core arrays 与
normalized raw 同行比较，600 个 Zarr episode-array 与 processed 精确比较通过。
大型生成数据不进入 git。

可复现 processing/export 命令（normalized v26 必须先完成上述 lineage 审计）：

```bash
python examples/process_episodes.py episodes/salvage_v26/pick_place_toy \
  --output-root episodes_processed/salvage_v17/pick_place_toy \
  --profile rgb_pc --pointcloud-num-points 1024 --task-name pick_place_toy
python - <<'PY'
from dexmani_real.dataset.export import PolicyZarrExportConfig, export_processed_hdf5_to_zarr
export_processed_hdf5_to_zarr(
    "episodes_processed/salvage_v17/pick_place_toy",
    "datasets/salvage_v10/pick_place_toy.zarr",
    PolicyZarrExportConfig(expected_task_name="pick_place_toy"),
)
PY
```

上述库调用用于显式选择新的 salvage 输出目录；CLI 仍固定输出 `datasets/<task>.zarr`。
已存在的输出会拒绝覆盖，不要删除原始数据或旧产物来重复执行。

### 历史信息限制与新的录制修正

`episode_20260827_195951` 的 zero-based row 157：旧 collapsed tactile flag 为 false、
calibrated=false、hand_contact 是有限全零值；hand_qpos_stale=false、hand_connected=true、
frame_status=0，hand/tactile source 相等且在 anchor 前 18.70 ms。
v25 没有独立 aggregate validity，无法从现有数据区分合法无接触与旧驱动的无效零占位。
本次保留原行，不推断、不补值、不据此追加 rejection；该信息无法恢复。

最终源码审查发现，原录制代码在 aggregate-invalid 时也可能复制零占位，而 deployment
会保持 validity/provenance source-match 的 fail-closed 行为。raw v28 因此增加一个必要的
`hand_contact_source_monotonic_ns`：录制按当前 control anchor 从 live hand ring 独立选择
最新有效 aggregate，保持最新 hand qpos/current 与 dense 原义不变；没有有效读数则保存
NaN/source0。旧但有效 aggregate 只将 freshness telemetry 置 false，不触发离线删行。
这不是跨 raw row 修复；当前 processing 仍严格 `contact_force[t] = raw hand_contact[t]`。
deployment 的 calibration/unit/source-match、freshness 与 command safety 没有放宽。

审查修正后全量离线测试：`python -m pytest -q`，399 passed、99 subtests passed；
`python -m compileall -q dexmani_real examples tools` 与 `git diff --check` 通过。
最终 normalized-v26 RGB-PC dry-run 仍为 60 accepted / 1 persistent-IK rejection；
独立 adversarial source review 已通过，没有未解决的 material Real-code finding。
`astra-high` 用受支持的 `gpt-6-astra/high`；中等实现阶段复用该高档 agent
（线程限制，未声称使用不存在的 astra-medium alias）；机械工作使用 `luna-max`。

Hardware validation: NOT RUN

当前已安装的相邻 `../dexmani_policy` exporter/consumer 仍是 Policy Zarr v9、
camera-master `observation_reference` 语义；它会拒绝 current v10 artifacts 缺少的
旧 attr。该 integration limitation 已知且未通过修改 external repository 或削弱 current
contract 解决；可用的 v10 control-step consumer 只在内存验证过，未写回 external tree。
