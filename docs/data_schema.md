# Real 数据集 schema 参考

本文是 DexMani Real 持久化数据的字段参考，覆盖当前 raw HDF5 v27、processed HDF5 v16 与
Policy Zarr v9。运行行为和精确校验仍以
[`recording/storage/schema.py`](../dexmani_real/recording/storage/schema.py)、
[`dataset/processing.py`](../dexmani_real/dataset/processing.py) 与
[`dataset/export.py`](../dexmani_real/dataset/export.py) 为准。

本文只描述 Real 域。Sim 数据独立生成、训练和部署，不能与 Real 数据按同一坐标系或
标签语义混用。

## 目录

- [约定与数据流](#约定与数据流)
- [raw episode HDF5 v27](#1-raw-episode-hdf5-v27)
- [processed HDF5 v16](#2-processed-hdf5-v16)
- [Policy Zarr v9](#3-policy-zarr-v9)
- [字段映射摘要](#字段映射摘要)
- [读取与训练注意事项](#读取与训练注意事项)

## 约定与数据流

```text
episodes/<task>/episode_*          raw v27 directory
    data.h5 + depth.h5 + rgb.mp4
                │ process_episodes.py
                ▼
episodes_processed/<task>/*.h5     processed v16, one file per raw episode
                │ export_policy_zarr.py
                ▼
datasets/<task>.zarr               Policy Zarr v9, one profile per store
```

- `N` 是单个 raw 或 processed episode 的帧数；`T` 是 Zarr 中全部 episode 的总帧数；
  `P` 是已持久化点云的点数。离线处理的 `PointCloudConfig.num_points` 只要求为正整数，
  因而 processed HDF5 与 Zarr 的 `P` 由其处理配置确定。官方处理 CLI 与实时 IPC 为避免
  模型/共享内存 shape 不匹配，当前仅接受 `1024`、`2048`、`4096`、`8192`。
- `H_raw,W_raw` 是记录期 RGB/depth 对齐图像尺寸；`H_p,W_p` 是 processed 图像尺寸，默认
  为 `240,320`，可由处理配置改变。
- `float64`/`float32`、`int64` 等表示 NumPy/HDF5 dtype；`bool` 表示 `np.bool_`。表中 shape
  都包含时间维，除非特别说明为 attribute。
- `xarm_base` 是 Real 的机器人世界坐标系。位置、点云 XYZ 与 fingertip 坐标单位为 m；
  关节与手部目标单位为 rad；rot6d 是无单位旋转表示；点云 RGB 是 `[0,1]` 的 float32。
- 逐行动作对齐语义为 `obs[t]_before_action[t]`。新采集 raw 不补时间网格；
  迁移行结合 `fill_reason`、`flag_sample_valid` 和 source 索引解释。

## 1. raw episode HDF5 v27

发布目录仍是 `data.h5 + depth.h5 + rgb.mp4`。所有 dataset 的第一维等于
`meta.num_frames`，depth 是与 RGB 对齐的 uint16 图像。Reader 只接受 schema 27，
检查必需文件、dataset shape/dtype 与帧数；构造和 finalize 不完整解码 RGB，也不重放 runtime proof。
历史 v24 必须先运行[冻结转换器](raw_v24_migration.md) 得到 v25，再运行
`tools/convert_raw_v25_to_v26_tactile.py` 迁移到 v26（直接录制的 v25 按 `* 10` 还原
SDK 原生刻度，v24 派生的 v25 需要显式 `--converted-v24-scale`）；v26 是 frozen legacy
artifact，不再被当前 Reader 直接读取，没有 compatibility Reader。

### 必需的 50 个 dataset

| 字段 | 每行 shape | dtype |
|---|---|---|
| timestamp | scalar | float64 |
| source_sample_index | scalar | int64 |
| fill_reason | scalar | uint8 |
| flag_sample_valid | scalar | bool |
| arm_qpos, arm_qvel, arm_tau | (7,) | float64 |
| hand_qpos, hand_current | (12,) | float64 |
| hand_contact | (5,3) | float64 |
| hand_tactile_force | (5,120,3) | float64 |
| arm_connected, hand_connected, hand_qpos_stale | scalar | bool |
| tracking_error | scalar | float64 |
| arm_last_cmd_seq | scalar | int64 |
| action_arm_joint_sent | (7,) | float64 |
| action_hand_joint | (12,) | float64 |
| action_arm_ee | (9,) | float64 |
| flag_action_queued | scalar | bool |
| flag_frame_status | scalar | uint8 |
| observation_anchor_monotonic_ns | scalar | uint64 |
| observation_valid | scalar | bool |
| arm_source_monotonic_ns, hand_source_monotonic_ns, tactile_source_monotonic_ns, vr_source_monotonic_ns, camera_source_monotonic_ns | scalar | uint64 |
| tactile_sum_fresh, tactile_fresh, tactile_calibrated | scalar | bool |
| tactile_unit_code | scalar | uint8 |
| flag_camera_fresh | scalar | bool |
| camera_health | scalar | uint8 |
| camera_depth_frame_number, camera_color_frame_number | scalar | uint64 |
| policy_observation_arm_qpos | (7,) | float64 |
| policy_observation_hand_qpos | (12,) | float64 |
| policy_observation_valid | scalar | bool |
| policy_observation_contact_force | (5,3) | float64 |
| policy_observation_contact_force_valid | scalar | bool |
| policy_observation_tactile_force | (5,120,3) | float64 |
| policy_observation_tactile_force_valid | scalar | bool |
| policy_observation_tactile_source_monotonic_ns | scalar | uint64 |
| policy_observation_tactile_calibrated | scalar | bool |
| policy_observation_tactile_unit_code | scalar | uint8 |
| vr_wrist_pos | (3,) | float64 |
| vr_wrist_rot6d | (6,) | float64 |
| vr_landmarks | (21,3) | float64 |
| head_quat_wxyz | (4,) | float64 |

每个 controller-emitted sample 恰好写一行；timestamp 是实际 observation/controller anchor（秒）。
`source_sample_index=0,1,...`、`fill_reason=0 (SOURCE)`、`flag_sample_valid=True`。
漏掉的 tick 保持时间缺口，不生成 HOLD/PLACEHOLDER。32 行磁盘 batch 只有 IO 职责。
迁移数据保留历史 row/media identity，`fill_reason=1 (CAUSAL_HOLD_LAST)` 与
`2 (LEADING_PLACEHOLDER)` 由 cleaner 排除。

动作训练使用必需的 `action_arm_joint_sent`，不得用未发送的候选替代。
`action_arm_ee` 是不能从 joint target 精确重建的控制意图：xarm_base position(m)+rot6d。
`flag_frame_status` 为 0 OK、1 HELD、2 IK_FAIL、3 SAFETY_REJECT、4 RETARGET_FAIL。
1–4 行的连续 IK_FAIL 保留为短暂暂停，长连续失败拒绝。独立 held/IK/retarget proof flags 不再持久化。

`hand_contact` 是 XHand SDK `calc_force`（每指 fx/fy/fz，软件 bias 校正），
`hand_tactile_force` 是 SDK `raw_force`（每指 120 点 fx/fy/fz，软件 bias 校正）；
单位为 xhand_sdk_native_unknown_si（SDK 原生数值刻度），不能当作已验证 SI 力。
`tactile_sum_fresh` 标记 `hand_contact`（aggregate）有效，`tactile_fresh` 标记
`hand_tactile_force`（dense）有效，二者共享 `tactile_calibrated`/`tactile_unit_code`。

v27 新增的 `policy_observation_contact_force`/`policy_observation_tactile_force` 是**录制时**
从高频 hand/tactile ring 按 camera source 因果选择的 camera-aligned tactile payload，附带
`*_valid`、`policy_observation_tactile_source_monotonic_ns`、`*_calibrated`、`*_unit_code`
provenance；invalid 浮点 payload 存 NaN 且 valid 标志为 false，绝不 zero-fill。视觉 profile
直接消费这些字段、不再跨 16 Hz raw rows 重选；JOINT profile 仍用 grid-anchor selector：
选择 persisted row <= 当前行、source <= reference、hand/tactile source 相等、
fresh/calibrated/unit_code=0 且 skew 有界的最新 source。选中 payload 非有限就拒绝，
不用别的 payload 修复。视觉 reference 是 camera source，JOINT reference 是 observation anchor。
JOINT profile 仍要求 arm/hand 相对 anchor 的年龄不超过处理配置的 max skew；
该数据集质量阈值比在线硬件 stale 阈值更严格。视觉配对状态的 freshness 在生产者中保证。

`camera_health` 直接持久化 camera header 的 health enum（OK/CLOCK_RESET/DUPLICATE/FRAME_GAP/
DELIVERY_DELAY）。`flag_camera_fresh` 保留原“new + healthy + recent”运行时语义，仅供 audit；
offline camera hard-invalid 只看 source>0、source<=anchor、age<=budget、health 合法且非
CLOCK_RESET/DELIVERY_DELAY。

机器人 joint state 保留物理来源；EEF/fingertip 只在 processing/viewer 中派生。
视觉 `policy_observation_*_qpos` 保留 camera-source 因果对齐后的状态，其他 runtime
sequence/publish/receive/history/skew proof、SDK ACK、device clock、profiling 均不属于 raw schema。

### Metadata 与 RGB-D 几何

保留 schema_version=27、num_frames、control_hz、task_label、operator；
`grid_dt_s=1/control_hz` 只表示名义采样周期。保留必要 camera_name/serial/type/payload_mode、
depth_scale、depth/color 原生 intrinsics、width/height、distortion_model/coeffs、
`camera_T_color_from_depth`，以及适用的 `camera_T_xarm_base_from_color`、
`camera_T_xarm_base_from_depth`、`camera_T_eef_from_depth`。可保留廉价 code provenance。
RGB/pointcloud processing 在几何边界验证 calibration、serial 与 rigid transforms。

## 2. processed HDF5 v16

processed 文件是 `episodes_processed/<task>/*.h5`。它从 raw v27 选择、清洗和压紧行；其 `N`
因此不一定等于 raw 的 `num_frames`。它用于离线训练、导出与可视化；物理回放可将其作为
保留 raw 行的 provenance 清单，但绝不发送其 `float32` 动作。回放在 `source_path` 找到 raw episode
后，从中读取 recorded published arm target 与 recorded logical hand target。根 attrs
必须满足：
`schema_name=dexmani-real-processed-hdf5`、`schema_version=16`、`domain=real`。

v16 在 v15 之上把视觉 profile 的 `contact_force`/`tactile_force` 对齐语义从“离线 16 Hz row
re-selection”改为“录制时高频 ring 按 camera source 对齐”：processed 直接消费 raw
`policy_observation_contact_force`/`policy_observation_tactile_force`，provenance 持久化
`observation_reference_monotonic_ns = camera_source_monotonic_ns`、`tactile_source_monotonic_ns =
policy_observation_tactile_source_monotonic_ns`。v15 文件不做伪迁移，必须由 raw v27 重新处理生成 v16。

处理入口在 discovery 边界将每个 raw episode 路径解析为 canonical absolute path；新生成的
processed artifact 将它持久化在 `source_decision_json.source_path`。processed replay 只消费该路径，
要求其下的 raw `data.h5` 存在；不推断或 fallback 到其他 source path。

Dataset 拥有 processed task identity 合同：`task_name` 必须是非空、已去除首尾空白、无 ASCII 控制字符（C0/DEL）且
不为 `unknown` 的字符串。处理入口按全局 override → annotation → raw `task_label` 解析一次；
全局 override 与 annotation 的显式冲突仍拒绝。所有 accepted episodes 的 identity 必须一致，
无效 task identity 或 mixed task batch 在创建发布 staging 前失败。普通 structural publication
gate、完整 processed validator 和 Zarr artifact inspection 均验证该合同。Canonical processing
CLI 还要求 output root basename 与 task identity 相同，以满足 Zarr CLI 的 expected task；
direct library 仍可使用任意临时输出目录。

删除无效 raw 行可能使压紧数组包含多个 source 连续段。v16 不把缺口两侧伪装成相邻时间步：
`source_segment_ends` 明确记录每段边界，质量窗口只在段内计数。Policy Zarr 不再把这些段
展开为多个训练 episode；一份 processed 文件只允许对应一个完整训练 episode。导出端不再容忍
任何 source 行删除：只有保留全部 source 行且 source 序列连续的 processed HDF5 才整份准入，
首部/尾部裁剪、内部缺口或时间/样本跳变都整份拒绝。

处理入口只接受通过 raw reader 结构校验的 v27 episode，sent action 必需。RGB 与
点云 profile 还要求 raw RGB-D 几何、depth scale 与 `T_xarm_base_from_color` 完整有效。视觉
profile 使用 `/policy_observation_*_qpos` 生成 `joint_state`；`joint` profile 使用 control-grid
state。processed action 是已发布的 teleop joint target；离线处理不模拟或审计 learned-policy
endpoint 约束。

### temporal quality 与 stall window

`processing_config_json.temporal_quality` 默认使用 `policy=audit`。其中
`stall_window_frames=8` 表示每个 stall window 包含 8 个样本，端点为
`start` 与 `start+7`（不是 8 个 index 的差值），且窗口不能跨越 source-contiguous segment；
detector 标记受影响的 `start+1` 到 `end` 行。`audit` 只记录 temporal findings，`hard_only` 则
关闭 temporal detectors。机械 action 硬边界仍独立执行 hard-invalid 判定。

### profile 与数据集集合

所有 profile 都包含 core 七项；RGB 与点云字段由 profile 决定。校验器只要求 profile 声明的
字段与 `/provenance` 必须存在，文件可以保留额外的研究字段、group 或 attrs；Policy Zarr
导出只复制显式 v8 legacy 投影字段（见第 3 节），processed-only 的 `eef_pose` 与
`tactile_force` 不进入 Zarr。

| profile | 固定数据集 | 附加数据集 |
|---|---|---|
| `joint` | core | 无 |
| `rgb` | core | `rgb`, `depth`, `camera_intrinsic`, `camera_extrinsic` |
| `pointcloud` | core | `point_cloud` |
| `rgb_pc` | core | RGB 全部字段与 `point_cloud` |

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/joint_state` | `(N,19)` | float32 | 视觉 profile 为 `policy_observation_arm_qpos(7)+policy_observation_hand_qpos(12)`；`joint` profile 为 control-grid state，单位 rad。 |
| `/eef_pose` | `(N,9)` | float32 | `position_m(3)+rot6d(6)`，`xarm_base`；由本 artifact 的 `/joint_state[:,:7]`（已因果对齐的 measured arm qpos）经 canonical Arm FK 推导，rot6d 必须 canonical；不使用 xArm firmware Cartesian pose。每 timestep 只执行一次 Arm FK，`/fingertip_points` 复用同一 EEF 结果。processed-only，不进入 Zarr v8。 |
| `/action` | `(N,19)` | float32 | `action_arm_joint_sent(7)+action_hand_joint(12)`；是 teleop 已发布 target，单位 rad；arm 部分不表示 SDK accepted state 或物理到位。 |
| `/action_ee` | `(N,21)` | float32 | `eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)`，EEF 在 `xarm_base`；rot6d 两列必须是 canonical 单位正交列。 |
| `/contact_force` | `(N,5,3)` | float32 | 视觉 profile 直接等于 raw `policy_observation_contact_force`（录制时按 camera source 对齐的 aggregate `calc_force`）；JOINT profile 为 `hand_contact` 选择不晚于 observation reference、skew 有界且 hand/tactile source 相等的最新 fresh+calibrated+unit-proven 行。单位/轴由 root attrs 指定。 |
| `/tactile_force` | `(N,5,120,3)` | float32 | 视觉 profile 直接等于 raw `policy_observation_tactile_force`（录制时 camera-aligned dense `raw_force`）；JOINT profile 与同行 `/contact_force` 来自**同一条** causally selected raw tactile source row（同一 `tactile_source_row_index`）。不假设 `contact_force == tactile_force.sum(...)`；sensor/point 顺序只是 SDK `sensor_data`/`raw_force` 顺序，SI 单位与 taxel 空间几何未验证。processed-only，不进入 Zarr v9。 |
| `/fingertip_points` | `(N,5,3)` | float32 | 五指指尖坐标（m），`xarm_base`；所有 profile 均从本 artifact 的 `/joint_state` 经共享 arm+hand FK 重新计算（与 `/eef_pose` 共用同一次 Arm FK）。 |
| `/rgb` | `(N,H_p,W_p,3)` | uint8 | 仅 RGB profile；resize、不裁剪。 |
| `/depth` | `(N,H_p,W_p)` | uint16 | 仅 RGB profile；对齐到 RGB，nearest resize；0 无效，米值由 `depth_scale_m_per_unit` 给出。 |
| `/camera_intrinsic` | `(N,9)` | float32 | resize 后 color K，row-major 展平的 3×3。 |
| `/camera_extrinsic` | `(N,4,4)` | float32 | `T_xarm_base_from_color`；native color optical → xarm base。 |
| `/point_cloud` | `(N,P,6)` | float32 | 仅 pointcloud profile；`xyz_m(3)+rgb_[0,1](3)`，`xarm_base`。 |

`rgb` 与 `rgb_pc` profile 的完整校验强制 `/rgb` 与 `/depth` 具有相同的帧数和空间尺寸：
`rgb` 必须为 `(N,H,W,3)`、`depth` 必须为 `(N,H,W)`，且两者的 `N/H/W` 必须逐项一致；
不一致的文件在完整 payload 读取或导出时拒绝。`visualize_episode_processed.py` 只在实际
启用 RGB/点云时读取对应的 shape、dtype 和必要元数据，不再为 `--info` 或打开窗口执行完整
payload/provenance 扫描。

`/provenance` 只存在于 processed HDF5：

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/provenance/source_row_index` | `(N,)` | int64 | processed row 对应的 raw grid row。 |
| `/provenance/source_sample_index` | `(N,)` | int64 | 对应 raw source sample。 |
| `/provenance/source_timestamp_s` | `(N,)` | float64 | 对应保留 raw row 的 `/timestamp`（实际 controller anchor，s）。 |
| `/provenance/source_segment_ends` | `(S,)` | int64 | 各 source 连续段在 processed 紧凑数组中的累积结束下标（exclusive）；严格递增，末值为 `N`。连续性同时要求 raw row/source sample 各加一且 timestamp 差为 `dt`（允许记录的容差）。 |
| `/provenance/source_keep_mask` | `(source_frames,)` | bool | 所有 raw row 的保留掩码。 |
| `/provenance/source_drop_reason_bits` | `(source_frames,)` | uint64 | 每个 raw row 的拒绝原因位图；位名在 provenance attrs。 |
| `/provenance/tactile_source_row_index` | `(N,)` | int64 | 每个 processed row 的 `contact_force`/`tactile_force` 实际采用的 raw tactile source row；视觉 profile 恒等于同行 `source_row_index`（录制时 camera-aligned），JOINT profile 合法 forward-fill 时可早于同行 `source_row_index`，validator 要求 `0 <= tactile_source_row_index <= source_row_index < source_frames`。 |
| `/provenance/observation_reference_monotonic_ns` | `(N,)` | int64 | 该 processed row 的 observation reference 时间（视觉 profile 为 camera source，`joint` profile 为 grid anchor），必须 `> 0`。 |
| `/provenance/tactile_source_monotonic_ns` | `(N,)` | int64 | 被选 tactile source row 的采样时间，必须 `> 0`、`<= observation_reference_monotonic_ns`，且差值不超过 `max_observation_skew_s`。 |

### processed root attrs

下表列出当前 writer 写入的 attrs。`string` 包含 UTF-8 文本与 JSON 文本；数值/vector 的
类型和 shape 与写入数组一致。它们是 dataset 之外的语义边界，不能仅凭字段名推断。

| 分组 | keys | 类型 / shape | 固定值或语义 |
|---|---|---|---|
| schema 与来源 | `schema_name`、`schema_version`、`domain`、`source_episode`、`source_frames` | string / int | `dexmani-real-processed-hdf5`、`16`、`real`，以及 raw 输入身份。 |
| 长度与训练标签 | `profile`、`episode_steps`、`dt`、`time_semantics`、`source_contiguity`、`source_contiguity_tolerance_s`、`obs_alignment`、`observation_reference`、`state_alignment`、`max_observation_skew_s`、`action_semantics`、`task_name`、`action_dim`、`action_ee_dim`、`action_space` | string / int / float | profile、压紧后长度、段边界 provenance、`obs[t]_before_action[t]`，以及 state/camera 对齐、观察 skew 和唯一的 `teleop_published_joint_target` 动作语义。 |
| Real core 语义 | `fingertip_points_frame`、`fingertip_points_unit`、`fingertip_points_derivation`、`fingertip_points_policy_id`、`action_ee_frame`、`action_ee_components`、`contact_force_representation`、`contact_force_source`、`contact_force_alignment`、`contact_force_unit`、`contact_force_si_verified`、`contact_force_frame`、`contact_force_fresh_required`、`contact_force_calibrated_required`、`contact_force_unit_code`、`contact_force_causal_to_reference`、`contact_force_hand_source_match_required` | string / bool / int | xarm-base 位置与 EEF frame；指尖单位 m；`fk_from_processed_joint_state` 与 FK algorithm identity。tactile attrs 保留来源、因果选择、单位、原生轴与逐行 proof 要求。 |
| EEF 语义 | `eef_pose_frame`、`eef_pose_components`、`eef_pose_derivation`、`eef_pose_algorithm_id` | string | 固定为 `xarm_base`、`position_m(3)+rot6d(6)`、`canonical_arm_fk_from_aligned_qpos` 与 `xarm7_custom_eef_pinocchio_fk_v1`；常量由 `planning/kinematics/arm_fk.py` 拥有，训练与部署共用同一 identity。 |
| full tactile 语义 | `tactile_force_representation`、`tactile_force_sensor_order`、`tactile_force_point_order`、`tactile_force_axis_labels`、`tactile_force_unit`、`tactile_force_si_verified`、`tactile_force_spatial_geometry_verified`、`tactile_force_fresh_required`、`tactile_force_calibrated_required`、`tactile_force_unit_code`、`tactile_force_causal_to_reference`、`tactile_force_hand_source_match_required` | string / bool / int | 只记录源码可证明事实：`xhand_sdk_raw_force_fx_fy_fz_bias_corrected`、SDK `sensor_data`/`raw_force` 顺序、`fx_fy_fz` 轴标签、`xhand_sdk_native_unknown_si` 单位；`si_verified=False`、`spatial_geometry_verified=False`；gating attrs 与 `contact_force` 相同（fresh/calibrated/unit_code=0/因果/hand-source 一致）。 |
| 处理与审计 | `processing_config_json`、`quality_summary_json`、`source_decision_json` | JSON string | 处理配置、选择/拒绝结论和 raw 行来源；`source_decision_json.hard_invalid_reason_names` 是必填的硬无效 reason 名称列表。 |
| RGB-D（仅 RGB/RGB-PC） | `rgb_transform`、`depth_transform`、`depth_unit`、`depth_scale_m_per_unit`、`depth_invalid_value`、`camera_intrinsic_semantics`、`camera_extrinsic_semantics` | string / float / int | 无裁剪 resize、aligned depth 的 nearest resize、depth 单位与无效值 `0`、K/T 语义。 |
| RGB-D provenance（仅 RGB/RGB-PC） | `source_camera_depth_intrinsics_native`、`source_camera_depth_distortion_model`、`source_camera_depth_distortion_coeffs`、`camera_color_distortion_model`、`camera_color_distortion_coeffs`、`camera_T_color_from_depth` | float64 `(9,)`；string；float64 `(K_d,)`；float64 `(4,4)` | native depth K、depth/color 畸变与 native depth optical → color optical 外参。 |
| 点云（仅 pointcloud/RGB-PC） | `point_cloud_frame`、`point_cloud_shape`、`point_cloud_color_source`、`point_cloud_policy_id`、`point_cloud_table_plane_abcd_json`、`point_cloud_sampling`、`point_cloud_transform` | string；int64 `(2,)`；JSON string | `xarm_base`、`(P,6)` 与点云策略、桌面和变换身份。 |

`/provenance.attrs["drop_reason_bit_names_json"]` 是 JSON object：键是 `uint64`
`source_drop_reason_bits` 的 bit 编号，值是对应拒绝原因名称。

浮点数据必须有限；processed validator 还检查 depth 非全零、canonical K、刚体外参、点云
RGB 范围、每帧非零 XYZ、持久化 workspace 边界、canonical action_ee 与 eef_pose rot6d，以及
profile/config/provenance 的完整性。`source_row_index`、`source_sample_index`、
`source_segment_ends`、`tactile_source_row_index`、`observation_reference_monotonic_ns`、
`tactile_source_monotonic_ns` 必须是实际 HDF5 `int64`，`source_timestamp_s` 是 `float64`，
`source_keep_mask` 是 `bool`，`source_drop_reason_bits` 是 `uint64`；row mapping 与段边界会
从这些数组重新计算，tactile provenance 会按第 2 节的因果不等式逐行验证。validator 不检查
`tactile_force != 0`，也不假设 `contact_force` 等于 `tactile_force` 的 taxel 求和。
导出 admission 不负责建立 processed 文件的来源信任；它只检查内部
schema、payload 与 provenance。

## 3. Policy Zarr v9

Zarr 是同一 `task_name`、同一 profile、同一 `dt`、同一 tail shape/dtype 与同一 Real
语义 attrs 的 processed episode 拼接结果：

```text
<task>.zarr/
├── data/                 # T 帧连续数组
│   └── <processed dataset keys>
└── meta/
    └── episode_ends       # int64, shape (E,)
```

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/data/<key>` | `(T, *processed_tail_shape)` | 与 processed 相同 | 逐 episode 按文件名字典序拼接；key 集合是显式的 v9 legacy 投影：core 五项 `joint_state/action/action_ee/contact_force/fingertip_points` 加 profile 的 RGB/点云字段。processed v16 的 `eef_pose`/`tactile_force` 及其 attrs 不进入 Zarr。 |
| `/meta/episode_ends` | `(E,)` | int64 | 所有合格 processed 文件完整长度的累积结束下标（exclusive）；第 `i` 个训练 episode 是 `[0 if i=0 else ends[i-1], ends[i])`。`E` 等于被准入的 processed 文件数。 |

数组使用 Zstd（默认 level 3）；时间 chunk 默认为 100 帧，最后 chunk 可更短。导出会逐数组
校验 shape、dtype、episode ends 和浮点值。

导出顺序是先完整验证 processed v16（包括 `eef_pose`/`tactile_force` payload 与 tactile
provenance），再显式投影 v9 keys 与 v9 attrs；不允许让 v16 新字段自动泄漏进 v9 输出。

Zarr root attrs 是最小运行语义，而不是 processed 全部 provenance：

| 范围 | attrs | 类型 / 语义 |
|---|---|---|
| schema 与任务 | `schema_name`、`schema_version`、`domain`、`profile`、`task_name`、`dt`、`episode_start_policy`、`obs_alignment`、`observation_reference`、`state_alignment`、`max_observation_skew_s`、`action_semantics` | string / int / float；固定为 `dexmani-real-policy-zarr`、`9`、`real`、`full_history`、`obs[t]_before_action[t]` 和 `teleop_published_joint_target`。训练不得用左侧 observation padding 构造 episode 起始样本。 |
| Real core | `contact_force_*` proof attrs、`fingertip_points_frame`、`fingertip_points_unit`、`fingertip_points_derivation`、`fingertip_points_policy_id`、`action_ee_frame` | string / bool / int；来自 processed 输入并要求全部 episode 一致。`eef_pose_*` 与 `tactile_force_*` attrs 是 processed-v16-only 语义，**不属于** v9 root attrs。 |
| RGB-PC profile | `depth_scale_m_per_unit`、`depth_invalid_value`、`camera_extrinsic_semantics` | float / int / string；depth 单位、无效像素值与 `T_xarm_base_from_color` 语义。 |
| pointcloud/RGB-PC profile | `point_cloud_frame`、`point_cloud_color_source`、`point_cloud_policy_id`、`point_cloud_table_plane_abcd_json`、`point_cloud_sampling`、`point_cloud_transform` | string（其中 table plane 为 JSON string）；点云 frame、构建策略与处理身份。 |

Policy Zarr v9 接受四种 profile 中语义 attrs 一致、压紧行保持网格连续的 processed v16
输入：只有保留全部 source 行且 source 序列连续的 processed HDF5 才整份准入。任何 source 行
删除（含首部/尾部裁剪）、内部缺口或无法解释的时间/样本跳变整条拒绝。其他合格
文件仍可进入同一批导出。Zarr
**不保留** processed 的 `/provenance`、质量摘要、raw 选择原因、
`action_ee_components` 或完整相机 calibration provenance；它保留 `obs_alignment` 和其他运行
语义 root attrs。需要审计、可视化
或重新处理时，应回到 processed HDF5。

## 字段映射摘要

| raw v27 | processed v16 | Policy Zarr v9 | 变换 |
|---|---|---|---|
| `policy_observation_arm_qpos + policy_observation_hand_qpos` | `joint_state` | `data/joint_state` | visual profile state，按 camera source 因果对齐后拼接 7+12，float64 → float32。 |
| `joint_state[:,:7]`（因果对齐的 measured arm qpos） | `eef_pose` | —（processed-only） | canonical Arm FK 一次计算 `position_m(3)+rot6d(6)`；不使用 firmware pose。 |
| `action_arm_joint_sent + action_hand_joint` | `action` | `data/action` | 拼接 7+12，使用实际 arm 提交流。 |
| `action_arm_ee + action_hand_joint` | `action_ee` | `data/action_ee` | 拼接 9+12。 |
| `policy_observation_contact_force`（视觉，录制时 camera-aligned）或 `hand_contact` + hand/tactile source proof（JOINT selector） | `contact_force` | `data/contact_force` | 视觉直接消费录制时 camera-aligned aggregate calc_force；JOINT 按 observation reference 选择最新因果且 skew 有界的同-source fresh+calibrated 样本，保留 `(5,3)` 轴语义。 |
| `policy_observation_tactile_force`（视觉）或 `hand_tactile_force` + 同一 tactile source proof（JOINT） | `tactile_force` | —（processed-only） | 视觉直接消费录制时 camera-aligned dense raw_force；JOINT 与 `contact_force` 共用同一 selected raw row（safe unique+inverse gather），保留 `(5,120,3)` SDK raw_force（软件 bias 校正）；无插值、无零填充。 |
| visual profile 的 camera-aligned arm/hand qpos 或 joint profile 的 control-grid arm/hand qpos | `joint_state` → `fingertip_points` | `data/joint_state`、`data/fingertip_points` | 每个 profile 先持久化自己的 7+12 `joint_state`，再通过共享 arm+hand FK 计算指尖，复用 `eef_pose` 的同一次 Arm FK；float64 → float32。 |
| `rgb.mp4` + `depth.h5:/depth` + camera meta | `rgb/depth/K/T` | 对应 `data/*` | RGB/depth resize 到 processed 尺寸；depth 已对齐 RGB。 |
| raw RGB-D 与 calibration | `point_cloud` | `data/point_cloud` | 使用 canonical builder，输出 xarm-base `xyzrgb`。 |
| raw grid/provenance | `/provenance`（含 `source_segment_ends`） | `meta/episode_ends` | 只有保留全部 source 行且单一 source 连续段的 processed 文件进入一个训练 episode；Zarr 不保留逐行来源。 |

## 读取与训练注意事项

- 使用 profile 所需的 key，不要根据同名字段猜测不同域的语义。
- `camera_intrinsic` 是 9 值展平矩阵；使用要求 `(...,3,3)` 的视觉组件前必须显式 reshape。
- 读取 depth 时必须应用 `depth_scale_m_per_unit`；不要假定所有设备的 Z16 单位相同。
- `contact_force` 的 SI 单位仅在 `contact_force_si_verified=True` 时成立。
- `fingertip_points`、`contact_force`、`tactile_force` 的每帧 first axis（history 的 axis 1）
  一致为 **thumb, index, middle, ring, pinky**。这是 `xhand_sdk_sensor_data_order` 的
  anatomical resolution：sensor indices `0..4` 对应 finger IDs `(2,5,7,9,11)`，canonical
  定义位于 `robot/model.py`。已有 serialized identifier 保留 `thumb_index_mid_ring_pinky`。
  processed v16 的既有 `tactile_force_sensor_order` 已表达该顺序，无需新增 required attr。
- Real 部署对请求的 `tactile_force` 同时校验 shape/dtype 与 PolicySpec semantics：
  `representation=xhand_sdk_raw_force_fx_fy_fz_bias_corrected`、`finger_order=thumb_index_mid_ring_pinky`、
  `sensor_order=xhand_sdk_sensor_data_order`、`point_order=xhand_sdk_sensor_data_raw_force_order`、
  `axis_labels=fx_fy_fz`、`unit=xhand_sdk_native_unknown_si`，并严格要求两个 verification flags 为
  boolean `False`。缺失或不匹配直接拒绝。
- `tactile_force` 当前 `si_verified=False`、`spatial_geometry_verified=False`：不要假设
  SI Newton、120-point spatial XYZ、taxel 邻接，也不要假设逐 taxel 求和等于 `contact_force`。
- `eef_pose` 与 `tactile_force` 只存在于 processed v16 与 Real 部署能力中；Policy Zarr v9
  不包含它们，训练 loader 不应在 v9 store 里寻找这两个 key。
- 训练 Zarr 前应保留其对应 processed HDF5；Zarr 是训练传输格式，不是完整审计归档。
- Real 训练 loader 必须校验 `schema_version=9`、`episode_start_policy=full_history`、camera-source
  state alignment 与 `teleop_published_joint_target` 动作语义，并使用
  `pad_before=n_obs_steps-1`、`pad_after=n_action_steps-1` 和 repeat-edge padding。当前 DP3 的
  `n_obs_steps=2`、`n_action_steps=8`，实例值为 `1/7`；不得把实例值写成通用常数。
- 训练 checkpoint 必须复制 Zarr root 语义及实际 point-cloud shape 作为数据合同。Real 部署
  在模型构造前核对 domain/schema、`dt`、point-cloud 的 preprocessing identity 与实时 worker，
  并在请求 fingertip 时核对其 derivation 与 algorithm ID；
  不能仅凭模型权重或配置文件名推断数据域。
