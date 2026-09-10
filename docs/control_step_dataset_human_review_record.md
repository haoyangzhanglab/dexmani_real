# Control-step dataset simplification 与 Policy v10 集成人工 Review 记录

记录日期：2026-09-10。范围：DexMani Real 数据架构简化、历史 pick_place_toy salvage、
DexMani Policy v10 消费端更新。本文件是已执行工作及证据的交接记录，不是第二份架构规范，
也不代表人工 review 已完成。

## 1. 如何使用本记录

建议按「提交范围 → 数据同行映射 → aggregate 来源 → deployment safety → 历史证据 →
Policy 消费端 → 未验证事项」顺序 review。第 11 节的 checkbox 留给人工填写。

权威来源：当前源码/runtime → schema/config →
[canonical plan](control_step_dataset_simplification_plan.md) → focused docs。
字段参考见 [data_schema.md](data_schema.md)，历史 forensic evidence 见
[invalid_frames_export_incident.md](invalid_frames_export_incident.md)，逐 episode 清单见
[salvage manifest](../artifacts/pick_place_toy_salvage_manifest.json)。旧 camera-master 计划已删除，
不得用旧版本注释或测试来要求恢复旧架构。

本文的测试次数是执行阶段的实际记录；编写本文只核对代码、提交、文档和 manifest，
不将历史测试结果表述为本次文档编辑又重跑了一次。生成数据没有进入 Git，reviewer 需在
拥有相应数据的机器上执行数据级复核。

## 2. 仓库快照与提交范围

| 仓库 | 实施 baseline | 本记录覆盖到的 HEAD |
|---|---|---|
| dexmani_real | `6d9655fdc2f774f0f5eb0613cea0cbcca8816eb6` | `cdc4f1edf748c01014019f7a50e79aeb9ff33069` |
| dexmani_policy | `2bfee2bf74a2c99f867166d9cd20e8cf742967d6` | `2bc5b85e3718fad7bfbb469f2cef7741c3f96032` |

Real 最初检查到 `837f818`，用户补拉 canonical plan 后以 `6d9655f` 作为执行 baseline。
两仓库实施前和本记录开始前均无 worktree 修改。本记录自身的后续文档提交不属于上表代码快照。

| 提交 | 内容 |
|---|---|
| Real `74a1dea` | row-preserving processing/export、raw28、logical-grid deployment、aggregate source 修正、replay、CLI/viewer 与测试清理 |
| Real `b126612` | 当前 schema/workflow 文档、历史证据转移、salvage manifest；删除过时计划与基线 JSON |
| Policy `2bc5b85` | export 严格消费 v10/control-step；测试 v9→v10；README |
| Real `cdc4f1e` | 更新 Policy v9 阻塞已解决的文档状态，记录实际消费验证 |

实现过程按 salvage hard gate 顺序执行；最后整理为 coherent commits，而非每个执行阶段
一个提交。不要依据 `74a1dea` 内文件排列推断 destructive cleanup 早于 salvage。
所有提交均为本地提交；本记录不声明已 push 或建立 PR。

从 Real 仓库根目录查看固定差异：

```bash
git diff --stat 6d9655f cdc4f1e
git diff 6d9655f cdc4f1e -- dexmani_real/dataset
git diff 6d9655f cdc4f1e -- dexmani_real/ipc dexmani_real/recording dexmani_real/teleop
git diff 6d9655f cdc4f1e -- dexmani_real/deployment dexmani_real/replay
git -C ../dexmani_policy diff 2bfee2b 2bc5b85 -- dexmani_policy/deployment/export.py
git -C ../dexmani_policy diff 2bfee2b 2bc5b85 -- tests README.md
```

Policy 相邻路径只是本地 review 布局约定，不是运行时依赖路径或数据绝对路径。

## 3. 架构变化与不变量

| 边界 | Before | After |
|---|---|---|
| Timeline | 视觉 profile 按 camera source 对齐额外 policy observation | 所有 profile 使用 control step，模态可异步 |
| Processing | 行级准入、跨行 tactile selection、compact/segments/provenance | accepted 整条逐行映射；否则整条失败/拒绝 |
| Behavioral rejection | cleaner 与时序框架交织 | 仅 persistent IK，沿用连续 >4 行边界 |
| Processed | EEF、dense、row provenance、decision/quality JSON | 五个 core datasets + profile 视觉字段 + 必要语义/identity |
| Export | projection、gap/keep-mask/source-segment rejection | owning validation → uniformity → 一文件一 episode → transactional publish |
| Deployment | 视觉帧时间用于机器人/tactile history reference | logical control-grid reference；原有在线门槛保留 |
| Annotation | include/exclude ranges | 仅 whole-episode include 与 task_name |
| Policy | v9、camera-master metadata | 严格 v10/control-step，无旧版 alias |

核心关系：

```text
T[t] = control observation anchor
0 < required modality source_ns <= T[t]

raw row t
  → processed row t
  → 同一 Zarr episode 的 row t
```

`camera_source < contact_source <= T[t]` 合法，不是 future leakage。
canonical period 默认 1/16 s；不要求实际 timestamp 差值在每行都精确等于 dt。
错过 tick 不补行；时序抖动不触发普通删行。sample identity/时间轴严重损坏仍属于技术错误。

### 3.1 直接映射

| Processed / Zarr field | shape、dtype | raw row t 来源 |
|---|---|---|
| joint_state | `[N,19]` float32 | arm_qpos + hand_qpos |
| action | `[N,19]` float32 | action_arm_joint_sent + action_hand_joint |
| action_ee | `[N,21]` float32 | action_arm_ee + action_hand_joint |
| contact_force | `[N,5,3]` float32 | hand_contact |
| fingertip_points | `[N,5,3]` float32 | 本行 joint_state 的共享 arm+hand FK |
| RGB/depth | uint8 / uint16 | 本行 raw camera payload，经 resize |
| camera_intrinsic/extrinsic | float32 | 本 episode calibration 与 resize 几何 |
| point_cloud | `[N,P,6]` float32 | 本行 RGB-D 经 canonical point-cloud preprocessing |

raw float64 → processed float32 是已定义转换，数值比较需使用相同 cast。fingertip 从实际
持久化的 float32 joint_state 计算，避免把未持久化高精度状态当作 consumer 输入。
`action_arm_joint_sent` 表示确切发布的 arm target，不等于 SDK ACK 或物理已到达目标；
手部仍是 logical hand target，不宣称记录了 worker 的每个 intermediate setpoint。

### 3.2 Episode admission

- 连续 1–4 行 `flag_frame_status == 2`：整条保留；连续至少 5 行：persistent IK 整条拒绝。
- 必需文件缺失、shape/dtype/帧数损坏、必需 payload NaN/Inf、媒体不可读、严重
  timestamp/sample identity 错误：技术失败，不能构造正常 processed artifact。
- camera fresh=False、timing jitter、observation_valid=False 单独出现、短 IK/tracking transient、
  aggregate 可用而 dense 不可用：不作为新增 episode rejection。
- 显式 annotation include:false 是用户排除，不属于自动行为质量判断。
- CLI 跳过未标注 rejected episode；显式 include 的失败阻断整批。库默认不跳过未标注 rejection。
- accepted 文件发布前完整验证；export 遇到无效 processed input 是整次 export 失败，
  不是跳过坏行或输出局部训练集。

## 4. Aggregate contact：最终审查发现并修复的问题

### 4.1 根因

旧 driver 在 aggregate-invalid 时可能提供有限全零占位；hand state 的其他字段仍可有效。
旧 recording 直接复制 `tactile_sum`，无法保证该值是最新有效 aggregate。
deployment 则保留 validity/calibration/provenance source-match 检查，并不会因此接纳无效零值。

只把 processed 改成同行 hand_contact，会暴露这处原有录制问题；不能通过离线 previous-row
repair、放松 deployment gate 或把无效零解释成 no-contact 来修复。

### 4.2 最小修正

[ipc/causal.py](../dexmani_real/ipc/causal.py) 的 `read_hand_contact_causal` 在 live hand ring
从新到旧选择 aggregate，要求：

```text
state_valid == 1
tactile_sum_valid == 1
qpos_stale == 0
finite float64[5,3] tactile_sum
0 < source <= payload_publish <= ring_publish <= control_anchor
```

随后显式传入 `build_episode_state`；来源 scalar 经 EpisodeState → frame mapping → sample IPC →
raw HDF5 传播。适用三个 live caller：teleop normal、teleop held、executor rollout recording。

| 值 | 来源/失败行为 |
|---|---|
| hand_qpos/current 与 hand_source | 原来的最新 hand feedback；不随 aggregate 回退 |
| hand_contact 与 hand_contact_source_monotonic_ns | 独立选中的有效 aggregate 与实际 source |
| hand_tactile_force 与 tactile_source | 原来的 dense ring 来源，不复用为 aggregate source |
| 没有有效 aggregate | NaN contact、source0、freshFalse，不退回当前 hand 的零占位 |
| 旧但有效 aggregate | 保留；250 ms 年龄阈值只决定 tactile_sum_fresh telemetry |

新 scalar 是一次 raw28 边界变更的一部分，没有额外再 bump schema。
不能使用 hand_source 或 dense source 冒充独立 aggregate 时间。录制选择不新增 run-start
过滤，否则会重新制造 frame0 缺失；在线 inference 原来的 not_before/run_started 门槛不变。

此处 live-ring selection 不是“从旧 raw row 修复新 raw row”。processed 始终读取当前 raw
hand_contact；新 raw28 校验独立 contact source，窄 legacy26 路径仍按其原 hand_source 解释。
calibrated/unit telemetry 继续属于 dense/provenance，不为另一个时间的 aggregate 伪造证明。

### 4.3 不能混淆的两个合同

训练可接受传感器不完美；deployment 必须 fail closed。两者共享 logical timeline 和来源语义，
不表示训练 episode 内每个 observation 都能在当前 live gates 下被模型立即使用。
特别是 aggregate/source 与最新 provenance 不匹配时，deployment 仍可能拒绝该步。

## 5. 源码 Review 导航

以下链接定位当前 owning module；精确回看本次实现请使用第 2 节固定 commit diff。

| Review 边界 | 入口 / 重点函数 | 要核实的事实 |
|---|---|---|
| Teleop 观察与发布 | [grid.py](../dexmani_real/teleop/control_loop/grid.py) | anchor 传播、发布 target 与录制 target 一致、旧 duplicate path 删除 |
| Contact selection | [causal.py](../dexmani_real/ipc/causal.py) / read_hand_contact_causal | 选择条件、publication 因果、不影响 command feedback |
| 录制传播 | [episode_samples.py](../dexmani_real/teleop/episode_samples.py)、[sample.py](../dexmani_real/recording/sample.py)、[frame.py](../dexmani_real/recording/frame.py) | normal/held、NaN/source0、freshness telemetry、owned copy |
| IPC / raw | [ipc/schema.py](../dexmani_real/ipc/schema.py)、[storage/schema.py](../dexmani_real/recording/storage/schema.py) | uint64 source、raw28 共 41 个 data.h5 datasets、37 个 source fields |
| Legacy / admission | [processing.py](../dexmani_real/dataset/processing.py) / _open_processing_episode、analyze_episode | current28 或 normalized26 窄入口；无 fake migration；IK boundary |
| 同行写入 | 同文件 / _write_processed_episode | full slices、exact sent action、contact 同行、visual enumerate 同行 |
| Processed boundary | [processed.py](../dexmani_real/dataset/processed.py) / validate_processed_hdf5 | source_frames==episode_steps、shape/dtype/payload/语义 |
| Export | [export.py](../dexmani_real/dataset/export.py) | uniformity、每文件一个 offset、episode_ends、atomic publish |
| Inference alignment | [observation.py](../dexmani_real/deployment/inference/observation.py) | _select_control_grid_reference_ns、_align_state_history_to_reference_ns、_build_observation |
| Rollout recording | [executor.py](../dexmani_real/deployment/executor.py) / _record_rollout_tick | contact selection 仅改变录制，不参与 action admission |
| Replay | [trajectory.py](../dexmani_real/replay/trajectory.py) / load_processed_trajectory | owning validation 后读取完整 raw float64，不选择旧 provenance rows |
| Policy 消费 | 相邻 Policy：dexmani_policy/deployment/export.py | _validate_core_zarr_attrs、_build_observation_contract、metadata 传递 |
| Policy 公共 parser | 相邻 Policy：dexmani_policy/deployment/contract.py | parse_deployment_contract 不依赖旧 observation_reference |

保留的在线约束包括 run_started/not_before、source/publication 因果、max_input_age、max_grid_lag、
state_valid、qpos_stale、camera live health、calibration、unit identity、必要 source-match、显式
请求 dense 时 freshness，以及 command safety、collision、generation、freshness、shutdown。
“未削弱 collision”指已有路径保持不变，不是本次新增实时碰撞检测或改变各运行模式的检查范围。

## 6. Schema / 删除清单 / 有意保留项

| Schema | Before → After |
|---|---|
| Current raw | 27 → 28 |
| Processed | 16 → 17 |
| Policy Zarr | 9 → 10 |
| Historical source / normalized working copy | 原件仍 v25；冻结 out-of-place v25→v26，不伪装 v28 |

processed 必要 identity：source_path、source_episode、source_schema_version、source_frames、
episode_steps；source_path 指向实际处理输入，salvage 因此指向 normalized26 working copy。
source_frames==episode_steps 在 owning validator 证明，不是外部篡改不可伪造的密码学证明。
point-cloud profile 保留 processing_config_json 用于重现 point-cloud preprocessing。

删除文件：

- dexmani_real/dataset/clean.py、quality.py。
- tools/analyze_camera_tactile_alignment.py。
- artifacts/camera_tactile_alignment_baseline.json。
- docs/camera_tactile_episode_integrity_fix_plan.md。
- tests/test_processed_v15.py、test_recording_tactile_alignment.py、test_strict_export_end_to_end.py、
  test_tactile_selector.py、test_zarr_v8_projection.py。
- Policy tests/test_deployment_zarr_v9.py；由当前 v10 contract tests 替代。

删除机制：

- _empty_policy_observation_signals、_recording_policy_observation_signals、
  TeleopGridObservation.policy_observation_signals。
- read_structured_frame_aligned_to_source、read_valid_structured_frame_aligned_to_source。
- select_tactile_rows_to_references、forward fill、keep/drop masks、source segments、row compaction。
- QualityPolicy、TemporalQualityConfig、TemporalQualityAssessment、assess_temporal_quality。
- processed /provenance、source_decision_json、quality_summary_json、invalid-frame range reports。
- processed eef_pose 及其 attrs/validator、processed dense tactile_force。
- _align_state_history_to_camera_frames，以及 exporter 的旧 gap/row rejection helpers。

十个删除的 raw/sample/IPC 字段：

```text
policy_observation_arm_qpos
policy_observation_hand_qpos
policy_observation_valid
policy_observation_contact_force
policy_observation_contact_force_valid
policy_observation_tactile_force
policy_observation_tactile_force_valid
policy_observation_tactile_source_monotonic_ns
policy_observation_tactile_calibrated
policy_observation_tactile_unit_code
```

移除处理 CLI flags：--horizon、--min-full-windows、--max-camera-age-s、--max-observation-skew-s、
--quality-policy、--abrupt-arm-step-rad、--abrupt-hand-step-rad、--compare-profiles、--verify-output。
目前仅 input_root、--output-root、--profile、--pointcloud-num-points、--task-name、--annotations、
--dry-run。annotation include_ranges/exclude_ranges 被拒绝，不保留 aliases。

有意保留：raw dense、action_ee、fingertip、FK implementation、冻结历史迁移工具及测试。
`_compute_policy_observation_ring_capacities` 名称会命中 policy_observation_ 搜索，但它是独立
在线容量计算，不是旧 recording duplicate path。点云空间过滤的 keep_mask 不是 dataset 行过滤。

## 7. 历史 Salvage 证据

### 7.1 Hard gate 与实际结果

| 项目 | 实际值 |
|---|---|
| Source episodes / frames | 61 / 14309 |
| Accepted episodes / frames | 60 / 14112 |
| Rejected episodes / frames | 1 / 197 |
| 唯一 rejected episode | episode_20260827_224527 |
| 原因 | persistent IK failure；观察到最长连续 19 行 |
| Zarr episodes | 60 |
| Accepted row loss | NO |
| Original raw modified | NO |
| Timing-only episode rejection | NO |

执行顺序：原始目录与 lineage 审计 → out-of-place frozen normalization → dry-run 逐一检查
rejection → 新 processed → 新 Zarr → exact counts/arrays/hash 验证 → manifest → raw/recording/
deployment destructive cleanup。最终 source 修正后又复跑 normalized26 dry-run，仍为 60/61。
这些计数是验证预期，不是 admission hardcode。

### 7.2 Tactile scale lineage

实际原件并非假设中的已经 normalized v26，而是全部 v25：60 条 converted_from_schema=24，
1 条 direct-v25（episode_20260908_212811）。因此需要先证明 v24-derived 的 scale。

- `c195991d88b6b63db85b6fc2b611515ecfce5d83` 引入 v24 时 driver 使用 0.1 scale。
- 到 `c7dced05d55d9ae148a19f94c2aa81ee719aa903` 替换 v24 期间，没有 tactile scale 变化。
- 冻结 v24→v25 converter 原值复制 tactile。
- `aac23d27e1fafcd264a24c122f9acf4d37962040` 在之后恢复 SDK-native scale。

据此为 derived-v24 明确选择 legacy-0.1，使用
[convert_raw_v25_to_v26_tactile.py](../tools/convert_raw_v25_to_v26_tactile.py) 在新副本上执行
一次 *10。没有依据 payload magnitude 猜 scale，也没有对已恢复的 v26 再次乘十。
其他数组保持原值，媒体未改变；新 working-copy 媒体可能与原件为 hardlink，后续必须按只读
处理，不能因为目录不同就假设可安全原地修改媒体。

### 7.3 输出与原始完整性

所有路径相对于 Real 仓库根目录：

```text
原始（不改）：episodes/pick_place_toy
normalized： episodes/salvage_v26/pick_place_toy
processed：  episodes_processed/salvage_v17/pick_place_toy
Zarr：       datasets/salvage_v10/pick_place_toy.zarr
manifest：   artifacts/pick_place_toy_salvage_manifest.json
```

原始 183 个文件的 size/SHA256 清单在转换前、salvage 后及最终检查一致。
fingerprint：`a0a4679248a8c345b37d648f3348ded1526f9e8afebede645b7811f3112a1c22`。
编码方式：相对路径 → [byte_size, SHA256] 的 mapping，按 key 排序 compact JSON 后再 SHA256。
manifest 保留 fingerprint，不包含全部 183 个文件的逐项哈希清单；可用原始文件重新计算比较。

已验证 accepted core arrays 对应 normalized raw 同行；全部 600 个 Zarr episode-array
（60 episodes × RGB-PC 的 10 keys）与 processed 精确比较。RGB/depth/pointcloud 全行读取，
不意味着 RGB MP4 编码前后无损或真实相机同步曝光；这些都不是此次宣称的证明。

### 7.4 无法恢复的一行历史信息

episode_20260827_195951，zero-based row 157：collapsed tactile flag=False、calibrated=False、
hand_contact 有限全零；hand_qpos_stale=False、hand_connected=True、frame_status=0，hand/dense
source 相等且在 anchor 前约 18.70 ms。

旧 telemetry 不能区分真实 no-contact 与 invalid aggregate 零占位。该行按要求保留，未推断
有效性、未重写 raw、未跨行修复、未追加 rejection。新 raw28 source 修复避免继续产生同类
来源歧义，但不能倒推旧数据。人工应明确接受这一证据限制，而不是把它写成“全部历史 contact
均已证明有效”。

## 8. Policy v10 消费端修正

最初 Real v10 输出被 Policy v9 gate 拒绝；用户随后明确授权 Policy 仓库修改。
Policy 仅改 export.py、README 与 focused test 模块，没有模型/encoder/decoder/config/训练
循环/sampler 改动，也没有复制数据到 robot_data 或修改 checkpoint。

严格要求并保存：

```text
schema_version = integer 10
observation_alignment = control_step_latest_causal
state_alignment = control_step
contact_force_source = raw_hand_contact_control_step
action_semantics = teleop_published_joint_target
```

删除旧 observation_reference 与按视觉/非视觉区分 alignment 的分支；任务、dt、action、
episode_start_policy、contact native units/SI flag、pointcloud/fingertip/RGB 语义校验保留。
旧 v8/v9、字符串或浮点版本号、camera-master state、旧 contact source 被拒绝。
不增加版本 registry、fallback 或兼容 wrapper。

真实 v10 artifact 已通过三组 selected fields：

1. joint_state + contact_force。
2. joint_state + point_cloud + contact_force + fingertip_points。
3. joint_state + rgb + point_cloud + contact_force + fingertip_points。

公共 parser 离线验证 joint/EE action 19/21 维、dt=0.0625、horizon=16、n_obs_steps=2、
n_action_steps=8、canonical future chunk=15 不变。ReplayBuffer 在实际数据上读取 core subset，
保持 60 episodes/14112 rows；SequenceSampler 用原有 sequence_length=16、pad_before=1、
pad_after=7，生成 13692 个窗口，逐窗口确认不跨 episode，首尾样本 shape/finite 检查通过。

这里没有证明每种模型都完成真实权重恢复或每个 RGB/PC tensor 都经过真实 encoder。
full_history attr 与既有 pad_before=1 的训练窗口行为应由人工核查其训练使用方式；本次不改
sampler，不将“无跨 episode 窗口”扩大为“完全没有边缘 padding”的证明。

## 9. 验证记录与证据强度

### 9.1 实际执行记录

| 命令 / 检查 | 环境 | 实际结果 | 不证明什么 |
|---|---|---|---|
| python -m pytest -q | Real / real_robot | 399 passed、99 subtests passed；最终一次 4.24 s | 非真机验证 |
| python -m compileall -q dexmani_real examples tools | Real / real_robot | PASS | 不实例化真实 SDK 或模型 |
| git diff --check / staged check | 两仓库 | PASS | 不证明语义正确 |
| RGB-PC processing dry-run | Real / real_robot | 最终 60 accepted、唯一 persistent IK rejection | 不自动代表 physical replay 可用 |
| 实际 processed/Zarr 构建与对比 | Real / real_robot | 60/14112；数组/hash/count 检查通过 | 不证明 demonstration success 或模型效果 |
| conda run -n policy python -m pytest -q -p no:cacheprovider tests | Policy / policy | 26 passed、36 subtests passed；1.09 s | 无真实 checkpoint export/restore |
| Python AST syntax 检查 | Policy / policy | export.py、v10 test PASS | 不是全仓库 compileall |
| 真实 Zarr 合同 + parser + ReplayBuffer/SequenceSampler | Policy / policy | 60/14112/13692，无跨 episode 窗口 | 无训练、无 GPU 模型验证 |

一次早期 Policy pytest 因仓库 cache 写权限出现 PytestCacheWarning；之后禁用 pytest cache，
最终通过无该警告。该权限问题没有导致核心代码修改。Real 环境普通 shell 的 python 不在 PATH，
执行时使用受管 real_robot interpreter；不要把环境路径问题误当成测试逻辑失败。

### 9.2 核心回归覆盖

| 要求 | 测试模块 / 断言 |
|---|---|
| D1 row preserving | test_control_step_dataset.py：N=40 全保留 |
| D2 camera timing | 同模块：freshFalse 不删行，必要媒体结构可用 |
| D3 tactile newer than camera | 同模块：contact 来自当前 raw row |
| D4 dense optional | 同模块：dense unavailable/source0 不 gate usable aggregate |
| D5 / D6 IK | 同模块：<=4 保留、>4 整条拒绝、不生成 compact artifact |
| D7 exact sent action | 同模块：读取 action_arm_joint_sent 而非 candidate |
| D8 export | test_control_step_export.py：两个 40-row 文件 → ends [40,80]；uniformity/atomic failure |
| D9 deployment | test_control_step_observation.py：state/contact 可新于 camera；age/run-start/health/future 等失败门槛 |
| Aggregate source 修正 | test_recording_control_contact.py：current/fallback/old/missing、future publish、三录制 caller、IPC/raw source、legacy26 |
| Replay | test_control_step_replay.py：完整 raw float64 与 source length |
| Task/annotation transaction | test_dataset_admission.py：whole-episode admission、identity、CLI、range rejection |
| Policy v10 | Policy test_deployment_zarr_v10.py：四 profile、旧版本/语义拒绝、native units、parser/action dimensions |

其他现有 raw/producer/recorder、deployment/safety 和 frozen migration tests 均包含在最终 suite。
测试名保留历史数字（如 test_raw_v26_recording.py）不等于 runtime 保留该版本兼容。

### 9.3 Review 过程

Real 使用独立 adversarial source review；首次发现 aggregate-invalid material finding，修复后
重新审查通过，无未解决的 material Real-code finding。高难度使用受支持 gpt-6-astra/high；
受线程限制，中等实施阶段复用高档 agent；机械工作为 luna-max，未声称使用不存在的 alias。
Policy 补充阶段独立 reviewer 因额度限制未能运行，主 agent 完成源码复核与测试；不能把这
一阶段标记为独立 review 通过。后续人工 review 正是补充独立判断，而不是照抄自动结论。

## 10. 安全的离线复核命令

以下从 Real 根目录执行。conda env 名基于本机环境；没有对应依赖时先记录环境差异，
不要为让检查通过而改安全或数据语义。命令不连接硬件、不启动训练、不覆盖已有产物。

```bash
git status --short
git -C ../dexmani_policy status --short
conda run -n real_robot python -m pytest -q
conda run -n real_robot python -m compileall -q dexmani_real examples tools
conda run -n policy python -m pytest -q -p no:cacheprovider ../dexmani_policy/tests
git diff --check
git -C ../dexmani_policy diff --check
```

Policy 测试应确认 import 的 editable 包指向待 review 的相邻 checkout；有疑问时切换到
Policy 根目录运行其测试。不要仅凭测试成功假定 import 到了目标代码。

```bash
conda run -n real_robot python examples/process_episodes.py \
  episodes/salvage_v26/pick_place_toy \
  --output-root episodes_processed/salvage_v17/pick_place_toy \
  --profile rgb_pc --pointcloud-num-points 1024 --task-name pick_place_toy --dry-run

rg -n 'policy_observation_' dexmani_real examples tests
rg -n 'select_tactile_rows_to_references|tactile_forward_fill' dexmani_real examples tests
rg -n 'source_keep_mask|source_drop_reason_bits|source_segment_ends' dexmani_real examples tests
rg -n 'TemporalQualityConfig|QualityPolicy|assess_temporal_quality' dexmani_real examples tests
rg -n '_align_state_history_to_camera_frames' dexmani_real tests
```

预期第一条 rg 仅命中 frozen migration fixtures 与独立容量 helper；其余零命中。
rg 无命中退出码 1 是正常结果，不要为追求字面零命中删除仍有独立用途的源码。
数据重新构建命令见 incident；不要移除 --dry-run 后盲目重跑到已有目录。
所有真实硬件、homing、physical replay、calibration、live rollout 需单独授权和现场协议。

## 11. 人工 Review checklist（尚未签署）

以下 unchecked 表示待人工独立确认，不表示已知代码失败。

- [ ] **R1 提交范围**：两仓库 baseline/HEAD、diff 与本记录一致，无无关模型/config/safety 改动。
- [ ] **R2 同行语义**：raw t → processed t → Zarr t；action 来自发布值；无 alternate row lookup。
- [ ] **R3 行保持准入**：source_frames==episode_steps；IK transient boundary 未重调；技术错误不 compact。
- [ ] **R4 Contact source**：三 caller 显式传 sum/source；current hand qpos 不回退；missing NaN0；raw/IPC 无假 alias。
- [ ] **R5 在线安全**：logical reference 改动未移除 age、run-start、health、state、calibration/unit/source-match 或 command/lifecycle gates。
- [ ] **R6 历史 scale**：Git lineage 与 frozen converter 一致；未根据 magnitude 猜 scale，未二次缩放。
- [ ] **R7 历史 integrity**：61/60/1、14309/14112/197、唯一 IK rejection、fingerprint 与实际数据匹配。
- [ ] **R8 历史歧义**：明确接受 row157 无法恢复 validity 的限制，不把所有 finite zero 当作已证明 no-contact。
- [ ] **R9 Export transaction**：一个文件一个 episode；无 partial export/覆盖旧产物；invalid input fail closed。
- [ ] **R10 Policy metadata**：v10 属性进入 data_contract；旧 gate 删除；units/shape/dt/action 校验保留。
- [ ] **R11 训练窗口**：既有 pad_before/pad_after/full_history 的实际行为与当前实验意图一致；本轮无修改。
- [ ] **R12 简洁性**：无新 registry、legacy hierarchy、manager、duplicate old/new path 或不可解释的 validator。
- [ ] **R13 证据边界**：离线数据验证未被误报为训练、真实 checkpoint 或真机验证。

建议把 findings 写入下表，附源码/提交位置及复现证据；material finding 在关闭前阻止签署。

| ID | 严重性 | 位置/证据 | 问题与影响 | 处理结论/验证 | 状态 |
|---|---|---|---|---|---|
| 待填写 | — | — | — | — | — |

人工签署：reviewer ______；日期 ______；Real commit ______；Policy commit ______；
结论（通过 / 有条件通过 / 阻止）______；未关闭 findings ______。

## 12. 剩余限制与下一步边界

1. 历史 row157 aggregate validity 无法恢复；不通过自动 repair 掩盖。
2. 当前 raw reader/physical replay 不支持 original v25 或 normalized26；后者仅有窄 offline processing 路径。
3. 旧 v8/v9 Policy Zarr 有意拒绝；旧 checkpoint 的训练 timeline 不会因重新贴 v10 标签而改变。
   如需新 control-step 模型，应使用明确的新 dataset/训练 provenance，不能把重新 export 当作重新训练。
4. 未执行训练、真实 checkpoint export/restore、GPU encoder smoke 或实际 rollout。
   消费端 contract/采样验证不覆盖真实模型效果、性能与真机输入可用性。
5. 新 raw28 录制路径通过 fake/IPC/schema 测试，但尚未由新硬件采集 episode 验证。
6. source_path 是绝对的实际输入路径；移动原始/normalized 数据后，不保证 processed physical replay
   自动寻找新位置。manifest 使用相对路径与 fingerprint 提供实验身份记录，不建立 migration framework。

此前“Policy v9 gate 阻塞新 v10”的问题已经解决，不再列为待修代码问题。
当前自动验证未发现本轮剩余 material code finding；这不是替代人工审查的保证。

Hardware validation: NOT RUN
