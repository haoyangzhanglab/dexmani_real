# invalid_frames_report 大批量标记与 Zarr 导出准入过严复盘

日期：2026-09-09。范围：`pick_place_toy` 任务 61 条 episode 的离线清洗产物
（`episodes_processed/pick_place_toy/`）、`process_log/invalid_frames_report.json`、
Policy Zarr 导出准入（`dexmani_real/dataset/export.py`）。全程离线分析，未连接硬件。

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
2. 帧 0 标记是**结构性启动时序伪影，数据本身健康**：相机曝光时间戳系统性早于触觉
   采样时间戳，帧 0 没有更早的触觉行可供因果选择。不是传感器故障，不是录制 bug。
3. 中段相机标记是**真实的 RealSense 单帧瞬态**（重复帧/时钟回跳/投递延迟），检测正确。
4. `long_ik_failure_hold` 是**唯一真正的数据质量事件**（机械臂逼近 workspace 边缘导致
   IK 持续失败），检测正确，该 episode 应当整条拒绝。
5. 真正需要修复的缺陷在**导出端**：旧 `_whole_episode_rejection` 把"存在任何被删除的
   source 行"一律整条拒绝，使 35 条仅丢失良性帧 0 的 episode 陪葬（57% 数据损失）。
   已改为缺口容忍规则（§4），dry-run 实测 `Exported 60/61`，仅拒 `224527`。

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

### 3.2 中段 `camera_invalid` + `observation_invalid`（真实相机瞬态）

RealSense 偶发重复帧/时钟回跳/投递延迟，`sensor/camera/worker.py` 的
`_camera_health()` 将其分类为非 OK，`teleop/control_loop/camera_freshness.py` 的
`CameraFreshnessTracker` 据此置 `flag_camera_fresh=False`；录制端 `observation_valid` 把相机新鲜度列为必需源
（`episode_samples.py`），因此两个原因成对出现。这是采集链路如实上报的单帧丢失，
8 条 episode 各损失 1–2 帧，处理端按合同将其从压紧数组中删除并形成段边界。

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

### 4.1 新导出准入规则（缺口容忍，≤2 行）

`_whole_episode_rejection` 改为纯结构化判定（`export.py`，常量
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

## 6. 已知权衡与回退路径

- 7 条相机缺口 episode 共 7 个中段缺口（合计 9 个被删帧）进入训练数据：每个缺口使
  跨缺口窗口含一个 2–3 dt 的伪动作转移，全数据集帧污染率约 0.75%。这是知情接受的权衡；若训练指标
  异常，回退选项为 `_MAX_TOLERATED_INTERIOR_GAP_ROWS = 1`（额外拒 `172305`、
  `175240` 两条 2 帧缺口 episode，保 58/61）或完全恢复内部断段拒绝（保 53/61）。
- 存量 61 个 processed HDF5 的 `source_decision_json` 中仍是旧警告文本（数据而非
  代码），重跑 `examples/process_episodes.py` 后刷新；不影响导出准入。
- 帧 0 丢弃本身未改动（清洗端因果合同保持原样）：导出侧豁免比放松清洗合同更小、
  且不触碰 raw→processed 语义。

## 7. 相关文件

| 文件 | 角色 |
| --- | --- |
| [dataset/export.py](../dexmani_real/dataset/export.py) | `_whole_episode_rejection` 新准入规则与常量 |
| [dataset/clean.py](../dexmani_real/dataset/clean.py) | 帧无效原因位、段边界审计、处理期警告 |
| [dataset/processed.py](../dexmani_real/dataset/processed.py) | `ProcessedProvenance` 不变量与连续性校验 |
| [examples/export_policy_zarr.py](../examples/export_policy_zarr.py) | 导出 CLI（`--dry-run` 只读预检） |
| [tests/test_zarr_v7_projection.py](../tests/test_zarr_v7_projection.py) | v7 投影回归 + 缺口容忍单测 |
| [docs/data_schema.md](data_schema.md) | processed v14 / Policy Zarr v7 合同（已同步） |
| [README.md](../README.md) | 导出工作流描述（已同步） |
| `episodes_processed/pick_place_toy/process_log/invalid_frames_report.json` | 本次现象来源（审计日志，数据文件） |
