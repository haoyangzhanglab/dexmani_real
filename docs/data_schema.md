# Real 数据集 schema 参考

本文是 DexMani Real 持久化数据的字段参考，覆盖当前 raw HDF5 v23、processed HDF5 v10
与 Policy Zarr v5。运行行为和精确校验仍以
[`recording/schema.py`](../dexmani_real/recording/schema.py)、
[`data/process.py`](../dexmani_real/data/process.py) 与
[`data/export.py`](../dexmani_real/data/export.py) 为准。

本文只描述 Real 域。Sim 数据独立生成、训练和部署，不能与 Real 数据按同一坐标系或
标签语义混用。

## 目录

- [约定与数据流](#约定与数据流)
- [raw episode HDF5 v23](#1-raw-episode-hdf5-v23)
- [processed HDF5 v10](#2-processed-hdf5-v10)
- [Policy Zarr v5](#3-policy-zarr-v5)
- [字段映射摘要](#字段映射摘要)
- [读取与训练注意事项](#读取与训练注意事项)

## 约定与数据流

```text
episodes/<task>/episode_*          raw v23 directory
    data.h5 + depth.h5 + rgb.mp4
                │ process_episodes.py
                ▼
episodes_processed/<task>/*.h5     processed v10, one file per raw episode
                │ export_policy_zarr.py
                ▼
dataset/<task>.zarr                Policy Zarr v5, one profile per store
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
- 逐行动作对齐语义为 `obs[t]_before_action[t]`。raw 的时间网格允许 causal hold 与
  leading placeholder，必须结合 `fill_reason`、`flag_sample_valid` 和 source 索引解释。

## 1. raw episode HDF5 v23

一个 raw episode 是原子发布的目录，而不是一个单文件：

| 成员 | 内容 | 存储 |
|---|---|---|
| `data.h5` | 控制网格状态、动作、质量与相机时序；根下有 `meta` group | 每个 dataset gzip 压缩 |
| `depth.h5` | `/depth`，已由 depth-to-color 对齐的 Z16 depth | gzip level 1 |
| `rgb.mp4` | 与控制网格一一对应的 RGB 帧 | 编码视频，不是 HDF5 dataset |

`data.h5/meta.attrs["schema_version"] == 23`。运行时 reader 只接受该版本；更早的 episode
只能在运行时外归档或检查，不能补出缺失的因果 policy observation，因此必须重新采集才能进入
当前训练链。v23 的 camera-source policy-observation 语义是 deployment-equivalent 训练数据的前提。

### 处理前提

raw v23 可以是 `meta.arm_sent_stream=False` 的有效录制；这时
`/action_arm_joint_sent` 合法地不存在。但当前 processed v10 writer 无条件以该字段生成
`/action[:,:7]`，所以要生成 processed HDF5 或 Policy Zarr，原始 episode **必须**有
`meta.arm_sent_stream=True` 与 `/action_arm_joint_sent`。这是处理链的输入前提，不是 raw
schema 的通用必填条件。

### `data.h5`：对齐与因果来源

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/timestamp` | `(N,)` | float64 | 单调递增的逻辑控制网格时间（s）。 |
| `/flag_sample_valid` | `(N,)` | bool | 此 grid row 是否直接来自 source sample；等价于 `fill_reason=SOURCE`。 |
| `/source_sample_index` | `(N,)` | int64 | 原始 source row；leading placeholder 为负数。 |
| `/source_timestamp` | `(N,)` | float64 | 选中 source 的时间（s）；leading placeholder 为 NaN。 |
| `/fill_reason` | `(N,)` | uint8 | `SOURCE`、`CAUSAL_HOLD_LAST` 或 `LEADING_PLACEHOLDER` 的枚举值。 |

### `data.h5`：机器人、触觉与动作

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/arm_qpos`, `/arm_qvel`, `/arm_tau` | `(N,7)` | float64 | xArm 关节位置（rad）、速度（rad/s）与 effort；`arm_tau` 单位未验证。 |
| `/arm_ee` | `(N,9)` | float64 | 末端位姿：`position_m(3)+rot6d(6)`，`xarm_base`。 |
| `/arm_connected` | `(N,)` | bool | arm state 连通标记。 |
| `/arm_last_cmd_seq` | `(N,)` | int64 | arm worker 最近命令序号。 |
| `/arm_last_cmd_is_hold` | `(N,)` | bool | 最近 arm 命令是否为 hold。 |
| `/hand_qpos`, `/hand_current` | `(N,12)` | float64 | XHand 关节位置（rad）与电流。关节顺序由 `XHAND_SDK_JOINT_NAMES` 固定。 |
| `/hand_fingertip` | `(N,5,3)` | float64 | 五指指尖位置（m），`xarm_base`。 |
| `/hand_contact` | `(N,5,3)` | float64 | 每指传感器原生三轴 tactile sum；单位见 `meta.tactile_unit`，默认不是已验证 SI force。 |
| `/hand_tactile_force` | `(N,5,120,3)` | float64 | 五指、每指 120 个触点、三轴的原生 tactile force。 |
| `/hand_tactile_contact` | `(N,5)` | bool | 每指接触判定。 |
| `/hand_tipboard_err`, `/hand_commboard_err`, `/hand_jointboard_err` | `(N,12)` | int32 | 对应 XHand board 错误码。 |
| `/hand_connected`, `/hand_qpos_stale`, `/tactile_fresh`, `/tactile_calibrated` | `(N,)` | bool | hand 连通、位置是否沿用旧值、触觉新鲜度与 bias 校准状态。 |
| `/tactile_source_monotonic_ns`, `/tactile_unit_code` | `(N,)` | int64 | 触觉 source 时间与设备单位代码。 |
| `/action_arm_joint_raw` | `(N,7)` | float64 | arm IK 原始候选；仅在 `flag_sample_valid & ~flag_held & flag_ik_ok` 时有保守有效语义。 |
| `/action_arm_joint` | `(N,7)` | float64 | grid row 的 arm joint action。 |
| `/action_arm_joint_sent` | `(N,7)` | float64 | 条件字段；仅 `meta.arm_sent_stream=True` 时存在，是实际提交给 arm 的命令流。processed 的 `action[:,:7]` 使用它。 |
| `/action_hand_joint_raw` | `(N,12)` | float64 | 手部 retarget 原始候选；仅在 `flag_sample_valid & ~flag_held & flag_retarget_ok` 时有保守有效语义。 |
| `/action_hand_joint` | `(N,12)` | float64 | grid row 的 XHand target（rad）；首帧相对初始反馈、后续相对前一已发布 endpoint 限速。 |
| `/action_arm_ee` | `(N,9)` | float64 | arm EEF target：`position_m(3)+rot6d(6)`，`xarm_base`。 |
| `/target_eef_pos_raw`, `/target_pos_before_clamp` | `(N,3)` | float64 | 限制前 EEF 位置候选（m）。 |
| `/target_eef_rot6d_raw` | `(N,6)` | float64 | 限制前 EEF rot6d 候选。 |

### `data.h5`：观测来源、VR、相机与质量

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/observation_id`, `/action_id` | `(N,)` | int64 | 观测和动作身份号。 |
| `/observation_anchor_monotonic_ns` | `(N,)` | int64 | 因果观测锚点时间。 |
| `/arm_source_sequence`, `/hand_source_sequence`, `/vr_source_sequence`, `/camera_source_sequence` | `(N,)` | int64 | 各输入 ring 的被选 sequence。 |
| `/arm_source_monotonic_ns`, `/hand_source_monotonic_ns`, `/vr_source_monotonic_ns`, `/camera_source_monotonic_ns` | `(N,)` | int64 | 各输入 source 时间。 |
| `/arm_publish_monotonic_ns`, `/hand_publish_monotonic_ns`, `/vr_publish_monotonic_ns`, `/camera_publish_monotonic_ns` | `(N,)` | int64 | 各输入 publish 时间。 |
| `/observation_source_receive_monotonic_ns` | `(N,4)` | uint64 | arm、hand、VR、camera 的 receive 时间。 |
| `/observation_source_age_s`, `/observation_source_skew_s` | `(N,4)` | float64 | 四类输入相对锚点的 age/skew（s）。 |
| `/observation_history_valid_mask` | `(N,4,1)` | bool | 四类输入的 history 有效性。 |
| `/observation_valid` | `(N,)` | bool | 当前观测能否作为有效因果观测。 |
| `/observation_skew_s` | `(N,)` | float64 | 观测总体 skew（s）。 |
| `/policy_observation_arm_qpos`, `/policy_observation_hand_qpos` | `(N,7)`, `(N,12)` | float64 | 供视觉 policy 使用的 arm/hand state；各自选择 `source_time <= camera_source_time` 的最新有效反馈，不等同于 grid-cut 最新 state。 |
| `/policy_observation_reference_monotonic_ns` | `(N,)` | int64 | 等于该 row 的 `camera_source_monotonic_ns`。 |
| `/policy_observation_arm_source_sequence`, `/policy_observation_hand_source_sequence` | `(N,)` | int64 | 所选 arm/hand state ring sequence。 |
| `/policy_observation_arm_source_monotonic_ns`, `/policy_observation_hand_source_monotonic_ns` | `(N,)` | int64 | 所选 state source 时间；不得晚于 reference。 |
| `/policy_observation_arm_publish_monotonic_ns`, `/policy_observation_hand_publish_monotonic_ns` | `(N,)` | int64 | 所选 state publish 时间；不得晚于 grid anchor。 |
| `/policy_observation_valid`, `/policy_observation_skew_s` | `(N,)`, `(N,)` | bool, float64 | 完整 source/publish 因果链是否有效，以及 `reference - min(arm_source, hand_source)`（s）。 |
| `/hand_accepted_target_action_id` | `(N,)` | int64 | hand worker 已由 SDK 接受的精确 target action id；不代表物理到位或 arm/hand 同步到位。 |
| `/action_created_monotonic_ns`, `/action_target_monotonic_ns`, `/action_valid_until_monotonic_ns` | `(N,)` | int64 | 动作创建、目标和失效时间。 |
| `/flag_action_queued` | `(N,)` | bool | 动作是否进入发布队列。 |
| `/vr_wrist_pos`, `/vr_wrist_rot6d` | `(N,3)`, `(N,6)` | float64 | VR wrist 位置与 rot6d。 |
| `/vr_landmarks` | `(N,21,3)` | float64 | VR 手部 landmark。 |
| `/head_quat_wxyz` | `(N,4)` | float64 | 头部四元数，顺序 wxyz。 |
| `/camera_health` | `(N,)` | int64 | 相机健康枚举。 |
| `/flag_camera_fresh`, `/camera_clock_reset`, `/camera_duplicate` | `(N,)` | bool | 相机新鲜度、时钟 reset 与重复帧标记。 |
| `/camera_depth_frame_number`, `/camera_color_frame_number`, `/camera_ring_sequence`, `/camera_generation`, `/camera_frame_gap` | `(N,)` | int64 | depth/color frame number、ring sequence、时钟 generation 与 frame gap telemetry。 |
| `/camera_depth_device_timestamp_s`, `/camera_color_device_timestamp_s`, `/camera_age_s`, `/camera_backlog_s`, `/pointcloud_valid_depth_ratio` | `(N,)` | float64 | 相机设备时间、age/backlog 与有效 depth 比率。 |
| `/camera_wait_return_monotonic_ns`, `/camera_payload_ready_monotonic_ns`, `/camera_depth_timestamp_domain`, `/camera_color_timestamp_domain` | `(N,)` | int64 | 相机取帧、payload ready 与 native 时间戳 domain telemetry。 |
| `/camera_delivery_delay_above_floor_s` | `(N,)` | float64 | 相机 delivery delay telemetry。 |
| `/flag_ik_ok`, `/flag_ik_attempted`, `/flag_retarget_ok`, `/flag_held`, `/flag_safety_reject` | `(N,)` | bool | IK、retarget、hold 与 safety gate 质量标记。 |
| `/flag_frame_status` | `(N,)` | int64 | 录制 frame status 枚举。 |
| `/tracking_error`, `/ik_solve_time_ms`, `/policy_map_time_ms`, `/hand_retarget_time_ms`, `/transition_check_time_ms`, `/policy_compute_time_ms` | `(N,)` | float64 | 控制质量和耗时诊断；`*_ms` 单位为 ms。 |

### raw camera payload

| 位置 | shape / dtype | 语义 |
|---|---|---|
| `depth.h5:/depth` | `(N,H_raw,W_raw)` uint16 | `librealsense_align_depth_to_color_z16`；像素在 color optical frame，与同 index RGB 像素对齐；`0` 为无效值。米值为 `depth * meta.depth_scale`。 |
| `rgb.mp4` frame | `(H_raw,W_raw,3)` uint8 | 与控制网格逐帧对应的 RGB。codec、pixel format、宽高和 fps 在 `meta`。 |

### `data.h5:/meta` attrs

下表的类型是 writer 写入时的逻辑 HDF5 attribute 类型；字符串由 h5py 以其支持的字符串
编码保存。固定 attrs 与调用方可扩展 attrs 分开列出，后者不能当作稳定 schema。

| 分组 | keys | 类型 / shape | 含义与条件 |
|---|---|---|---|
| schema 与录制 | `schema_version`、`task_label`、`operator`、`control_hz`、`fps`、`grid_dt_s`、`num_frames`、`success`、`truncated`、`stop_reason` | int / string / float / bool | episode 身份、控制网格与终止结果；当前 writer 的 `schema_version=23`。 |
| 时长与基本质量 | `duration`、`wall_duration_s`、`grid_duration_s`、`non_sampled_duration_s`、`wall_fps`、`min_frames_met`、`has_camera`、`has_timestamps` | float / bool | 录制耗时、网格覆盖、实际帧率与基本可用性。 |
| 逐行汇总质量 | `ik_hold_frame_count`、`camera_invalid_frame_count`、`observation_invalid_frame_count`、`sample_invalid_frame_count`、`safety_reject_frame_count`、`command_quiescence_count` | int scalar | 由 raw flags 与 grid timestamp 汇总；诊断用途，不替代逐行 flags。 |
| 录制配置 | `resolved_config_sha256`、`skip_initial_frames`、`arm_sent_stream` | string / int / bool | 解析配置哈希与跳过帧数；`arm_sent_stream` 仅在 true 时写入，决定条件数据集是否存在。 |
| 坐标、触觉与相机时序语义 | `robot_world_frame`、`robot_world_equals_xarm_base`、`arm_ee_frame`、`action_arm_ee_frame`、`hand_fingertip_frame`、`action_*_raw_validity_expression`、`tactile_*`、`arm_tau_*`、`camera_payload_mode`、`camera_*_semantics`、`policy_observation_*_semantics` | string / bool / float | `SEMANTIC_META_ATTRS` 写入的固定语义；包括 `xarm_base` frame、raw action 有效性、tactile 单位/标定/接触、arm effort、camera 时间/时钟及 v23 policy observation 配对定义。 |
| 视频与 writer | `camera_writer_queue_size`、`camera_encoding_codec`、`camera_encoding_crf`、`camera_encoding_preset`、`camera_encoding_pixel_format`、`camera_encoding_width/height/fps`、`camera_health_taxonomy_json`、`camera_frame_gap_semantics`、`camera_depth_storage`、`camera_depth_payload_semantics`、`camera_stream_frames`、`camera_writer_error` | int / string / float | MP4 编码、payload 定义、camera writer 状态与健康 telemetry。 |
| writer 性能汇总 | `camera_writer_queue_high_watermark`、`camera_writer_queue_capacity`、`camera_writer_close_s`、`camera_encode_{p50,p95,p99,max}_s`、`camera_hdf5_{p50,p95,p99,max}_s` | int / float | 相机 writer 队列、关闭和编码/HDF5 写入耗时。 |
| 调用方扩展 | `provenance_<key>`、`camera_metadata` mapping 中的每个 key | string；调用方给定 dtype | 非固定 key；前者被转换为 string，后者原样写入且可能覆盖已有同名 attr。因此下游不得把它们当作稳定契约。 |

以下相机 attrs 不是任意 raw v23 都具备；处理 RGB 或点云 profile 时，reader 会在几何边界
要求所需字段存在且有效。

| 条件 attrs | 类型 / shape | 写入条件与语义 |
|---|---|---|
| `camera_serial`、`camera_name` | string | 对应 pending metadata 提供时写入。 |
| `camera_depth_intrinsics`、`camera_color_intrinsics` | float64-like `(9,)` | 提供 `camera_geometry` 时写入；row-major 3×3 native K。 |
| `camera_depth_width/height`、`camera_color_width/height` | int scalar | 提供 `camera_geometry` 时写入。 |
| `camera_*_distortion_model`、`camera_*_distortion_coeffs` | string；float64-like `(K_d,)` | 提供 `camera_geometry` 时写入。 |
| `camera_T_color_from_depth` | float64-like `(16,)` | 提供 `camera_geometry` 时写入；native depth optical → color optical。 |
| `depth_scale` | float scalar | pending `depth_scale` 非空时写入；Z16 unit 到 m 的乘数。 |
| `camera_T_xarm_base_from_color`、`camera_T_xarm_base_from_depth` | float64-like `(16,)` | 同时有 calibration、camera name 与 world-camera 外参时写入；color/depth optical → `xarm_base`。 |
| `camera_T_eef_from_depth` | float64-like `(16,)` | calibration 提供 EEF-camera 外参时写入。 |
| `camera_type`、`camera_calibration_source_optical_frame` | string | calibration 与 camera name 可解析时写入；后者固定为 `camera_color_optical`。 |

## 2. processed HDF5 v10

processed 文件是 `episodes_processed/<task>/*.h5`。它从 raw v23 选择、清洗和压紧行；其
`N` 因此不一定等于 raw 的 `num_frames`。它用于离线训练、导出与可视化；物理回放可将其作为
保留 raw 行的 provenance 清单，但绝不发送其 `float32` 动作。回放必须在 `source_path` 找到并
校验原始 `data.h5` hash，再从 raw episode 读取精确已发送命令和完整模型 provenance。根 attrs
必须满足：
`schema_name=dexmani-real-processed-hdf5`、`schema_version=10`、`domain=real`。

删除无效 raw 行可能使压紧数组包含多个 source 连续段。v10 不把缺口两侧伪装成相邻时间步：
`source_segment_ends` 明确记录每段边界，质量窗口只在段内计数，Policy Zarr 导出时每段成为
独立 episode。一个 processed 文件仍对应一个 raw episode，但不一定只对应一个训练 episode。

它只接受通过 raw reader 校验的 v23 episode，并适用前述 `arm_sent_stream` 前提；RGB 与
点云 profile 还要求 raw RGB-D 几何、depth scale 与 `T_xarm_base_from_color` 完整有效。视觉
profile 使用 `/policy_observation_*_qpos` 生成 `joint_state`；`joint` profile 使用 control-grid
state。每个 retained source segment 的首个 action 仍以 raw `/arm_qpos` 与 `/hand_qpos` 的
control-grid feedback 校验 endpoint delta，与 deployment safety gate 的首命令语义一致。

### profile 与数据集集合

所有 profile 都包含 core 五项；RGB 与点云字段由 profile 决定，不能在一个 processed 文件
中任意增删。

| profile | 固定数据集 | 附加数据集 |
|---|---|---|
| `joint` | core | 无 |
| `rgb` | core | `rgb`, `depth`, `camera_intrinsic`, `camera_extrinsic` |
| `pointcloud` | core | `point_cloud` |
| `rgb_pc` | core | RGB 全部字段与 `point_cloud` |

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/joint_state` | `(N,19)` | float32 | 视觉 profile 为 `policy_observation_arm_qpos(7)+policy_observation_hand_qpos(12)`；`joint` profile 为 control-grid state，单位 rad。 |
| `/action` | `(N,19)` | float32 | `action_arm_joint_sent(7)+action_hand_joint(12)`；arm 部分是实际提交命令，单位 rad。 |
| `/action_ee` | `(N,21)` | float32 | `eef_position_m(3)+eef_rot6d(6)+xhand_target_rad(12)`，EEF 在 `xarm_base`。 |
| `/contact_force` | `(N,5,3)` | float32 | raw `hand_contact` 的每指三轴 tactile sum；单位/轴由 root attrs 指定。 |
| `/fingertip_points` | `(N,5,3)` | float32 | 五指指尖坐标（m），`xarm_base`。 |
| `/rgb` | `(N,H_p,W_p,3)` | uint8 | 仅 RGB profile；resize、不裁剪。 |
| `/depth` | `(N,H_p,W_p)` | uint16 | 仅 RGB profile；对齐到 RGB，nearest resize；0 无效，米值由 `depth_scale_m_per_unit` 给出。 |
| `/camera_intrinsic` | `(N,9)` | float32 | resize 后 color K，row-major 展平的 3×3。 |
| `/camera_extrinsic` | `(N,4,4)` | float32 | `T_xarm_base_from_color`；native color optical → xarm base。 |
| `/point_cloud` | `(N,P,6)` | float32 | 仅 pointcloud profile；`xyz_m(3)+rgb_[0,1](3)`，`xarm_base`。 |

`/provenance` 只存在于 processed HDF5：

| 路径 | shape | dtype | 语义 |
|---|---:|---|---|
| `/provenance/source_row_index` | `(N,)` | int64 | processed row 对应的 raw grid row。 |
| `/provenance/source_sample_index` | `(N,)` | int64 | 对应 raw source sample。 |
| `/provenance/source_timestamp_s` | `(N,)` | float64 | 对应保留 raw grid row 的 `/timestamp`（逻辑控制网格时间，s）；**不是** raw `/source_timestamp` 的 producer sample 时间。 |
| `/provenance/source_segment_ends` | `(S,)` | int64 | 各 source 连续段在 processed 紧凑数组中的累积结束下标（exclusive）；严格递增，末值为 `N`。连续性同时要求 raw row/source sample 各加一且 timestamp 差为 `dt`（允许记录的容差）。 |
| `/provenance/source_keep_mask` | `(source_frames,)` | bool | 所有 raw row 的保留掩码。 |
| `/provenance/source_drop_reason_bits` | `(source_frames,)` | uint64 | 每个 raw row 的拒绝原因位图；位名在 provenance attrs。 |

### processed root attrs

下表列出当前 writer 写入的 attrs。`string` 包含 UTF-8 文本与 JSON 文本；数值/vector 的
类型和 shape 与写入数组一致。它们是 dataset 之外的语义边界，不能仅凭字段名推断。

| 分组 | keys | 类型 / shape | 固定值或语义 |
|---|---|---|---|
| schema 与来源 | `schema_name`、`schema_version`、`domain`、`source_episode`、`source_frames` | string / int | `dexmani-real-processed-hdf5`、`10`、`real`，以及 raw 输入身份。 |
| 长度与训练标签 | `profile`、`episode_steps`、`dt`、`time_semantics`、`source_contiguity`、`source_contiguity_tolerance_s`、`obs_alignment`、`observation_reference`、`state_alignment`、`max_observation_skew_s`、`action_semantics`、`arm_max_delta_rad_per_tick`、`hand_max_delta_rad_per_tick`、`deployment_equivalent`、`task_name`、`action_dim`、`action_ee_dim`、`action_space` | string / int / float / bool | profile、压紧后长度、段边界 provenance、`obs[t]_before_action[t]`，以及 state/camera 对齐、观察 skew、动作 endpoint 限制和是否可用于 deployment training 的显式合同。 |
| Real core 语义 | `fingertip_points_frame`、`fingertip_points_unit`、`action_ee_frame`、`action_ee_components`、`contact_force_source`、`contact_force_unit`、`contact_force_si_verified`、`contact_force_frame` | string / bool | xarm-base 位置与 EEF frame；指尖单位 m；tactile 的来源、单位和原生轴。 |
| 处理与审计 | `processing_config_json`、`quality_summary_json`、`source_decision_json`、`source_member_sha256_json`、`source_resolved_config_sha256` | JSON string / string | 处理配置、选择/拒绝结论、三个 raw 成员哈希与录制配置哈希。 |
| RGB-D（仅 RGB/RGB-PC） | `rgb_transform`、`depth_transform`、`depth_unit`、`depth_scale_m_per_unit`、`depth_invalid_value`、`camera_intrinsic_semantics`、`camera_extrinsic_semantics` | string / float / int | 无裁剪 resize、aligned depth 的 nearest resize、depth 单位与无效值 `0`、K/T 语义。 |
| RGB-D provenance（仅 RGB/RGB-PC） | `source_camera_depth_intrinsics_native`、`source_camera_depth_distortion_model`、`source_camera_depth_distortion_coeffs`、`camera_color_distortion_model`、`camera_color_distortion_coeffs`、`camera_T_color_from_depth` | float64 `(9,)`；string；float64 `(K_d,)`；float64 `(4,4)` | native depth K、depth/color 畸变与 native depth optical → color optical 外参。 |
| 点云（仅 pointcloud/RGB-PC） | `point_cloud_frame`、`point_cloud_shape`、`point_cloud_color_source`、`point_cloud_policy_id`、`point_cloud_config_sha256`、`point_cloud_table_plane_abcd_json`、`point_cloud_sampling`、`point_cloud_transform` | string；int64 `(2,)`；JSON string | `xarm_base`、`(P,6)` 与可复现的点云策略、桌面和变换身份。 |

`/provenance.attrs["drop_reason_bit_names_json"]` 是 JSON object：键是 `uint64`
`source_drop_reason_bits` 的 bit 编号，值是对应拒绝原因名称。

浮点数据必须有限；processed validator 还检查 depth 非全零、K、刚体外参、点云 RGB 范围和
非零几何，以及 profile/config/provenance 的完整性。

## 3. Policy Zarr v5

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
| `/data/<key>` | `(T, *processed_tail_shape)` | 与 processed 相同 | 逐 episode 按文件名字典序拼接；可用 key 集合仍由 profile 决定。 |
| `/meta/episode_ends` | `(E,)` | int64 | 所有 processed source 连续段的累积结束下标（exclusive）；第 `i` 个训练 episode 是 `[0 if i=0 else ends[i-1], ends[i])`。`E` 可大于 processed 文件数。 |

数组使用 Zstd（默认 level 3）；时间 chunk 默认为 100 帧，最后 chunk 可更短。导出会逐数组
校验 shape、dtype、episode ends 和内容 SHA-256。

Zarr root attrs 是最小运行语义，而不是 processed 全部 provenance：

| 范围 | attrs | 类型 / 语义 |
|---|---|---|
| schema 与任务 | `schema_name`、`schema_version`、`domain`、`profile`、`task_name`、`dt`、`episode_start_policy`、`obs_alignment`、`observation_reference`、`state_alignment`、`max_observation_skew_s`、`action_semantics`、`arm_max_delta_rad_per_tick`、`hand_max_delta_rad_per_tick`、`deployment_equivalent` | string / int / float / bool；固定为 `dexmani-real-policy-zarr`、`5`、`real`、`full_history`、`obs[t]_before_action[t]`，并持久化 deployment observation 与 action 的精确合同。训练不得用左侧 observation padding 构造 episode 起始样本。 |
| pointcloud/RGB-PC core | `contact_force_unit`、`contact_force_si_verified`、`contact_force_frame`、`fingertip_points_frame`、`action_ee_frame` | string / bool；来自 processed 输入并要求全部 episode 一致。 |
| RGB-PC profile | `depth_scale_m_per_unit`、`depth_invalid_value`、`camera_extrinsic_semantics` | float / int / string；depth 单位、无效像素值与 `T_xarm_base_from_color` 语义。 |
| pointcloud/RGB-PC profile | `point_cloud_frame`、`point_cloud_color_source`、`point_cloud_policy_id`、`point_cloud_config_sha256`、`point_cloud_table_plane_abcd_json`、`point_cloud_sampling`、`point_cloud_transform` | string（其中 table plane 为 JSON string）；点云 frame、构建策略与处理身份。 |

Policy Zarr v5 只接受 `deployment_equivalent=True` 的 `pointcloud` 或 `rgb_pc` processed v10 输入。Zarr
**不保留** processed 的 `/provenance`、source 文件 hash、质量摘要、raw 选择原因、
`action_ee_components` 或完整相机 calibration provenance；它保留 `obs_alignment` 和其他运行
语义 root attrs。需要审计、可视化
或重新处理时，应回到 processed HDF5。

## 字段映射摘要

| raw v23 | processed v10 | Policy Zarr v5 | 变换 |
|---|---|---|---|
| `policy_observation_arm_qpos + policy_observation_hand_qpos` | `joint_state` | `data/joint_state` | visual profile state，按 camera source 因果对齐后拼接 7+12，float64 → float32。 |
| `action_arm_joint_sent + action_hand_joint` | `action` | `data/action` | 拼接 7+12，使用实际 arm 提交流。 |
| `action_arm_ee + action_hand_joint` | `action_ee` | `data/action_ee` | 拼接 9+12。 |
| `hand_contact` | `contact_force` | `data/contact_force` | 重命名，保留 `(5,3)` 轴语义。 |
| `hand_fingertip` | `fingertip_points` | `data/fingertip_points` | 重命名，float64 → float32。 |
| `rgb.mp4` + `depth.h5:/depth` + camera meta | `rgb/depth/K/T` | 对应 `data/*` | RGB/depth resize 到 processed 尺寸；depth 已对齐 RGB。 |
| raw RGB-D 与 calibration | `point_cloud` | `data/point_cloud` | 使用 canonical builder，输出 xarm-base `xyzrgb`。 |
| raw grid/provenance | `/provenance`（含 `source_segment_ends`） | `meta/episode_ends` | 每个 source 连续段映射为独立训练 episode；Zarr 不保留逐行来源。 |

## 读取与训练注意事项

- 使用 profile 所需的 key，不要根据同名字段猜测不同域的语义。
- `camera_intrinsic` 是 9 值展平矩阵；使用要求 `(...,3,3)` 的视觉组件前必须显式 reshape。
- 读取 depth 时必须应用 `depth_scale_m_per_unit`；不要假定所有设备的 Z16 单位相同。
- `contact_force` 的 SI 单位仅在 `contact_force_si_verified=True` 时成立。
- 训练 Zarr 前应保留其对应 processed HDF5；Zarr 是训练传输格式，不是完整审计归档。
- Real 训练 loader 必须校验 `schema_version=5`、`episode_start_policy=full_history`、camera-source
  state alignment 与 deployment action 合同，并使用
  `pad_before=0`；每个 episode 的前 `n_obs_steps-1` 个位置不生成训练样本。
- 训练 checkpoint 必须复制 Zarr root 语义及实际 point-cloud shape 作为数据合同。Real 部署
  在模型构造前核对 domain/schema、`dt`、点云 policy/config/table identity 与实时 worker；
  不能仅凭模型权重或配置文件名推断数据域。
