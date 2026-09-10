# Real 数据集 schema 参考

本文覆盖 raw HDF5 v28、processed HDF5 v17 与 Policy Zarr v10。精确合同由
[raw schema](../dexmani_real/recording/storage/schema.py)、
[processing](../dexmani_real/dataset/processing.py)、
[processed validator](../dexmani_real/dataset/processed.py) 和
[exporter](../dexmani_real/dataset/export.py) 拥有。
架构规范见 [control-step plan](control_step_dataset_simplification_plan.md)；
旧 incident 仅作为历史证据，不定义当前行为。

## 约定与数据流

```text
raw episode: N control rows
    → accepted: processed HDF5 N rows
    → one Policy Zarr episode N rows

rejected: whole episode; no partial artifact
```

Canonical training timeline 是 16 Hz control step。对于 observation anchor `T[t]`，
各模态独立选择之前可用的最新有效 observation，满足 `0 < source_ns <= T[t]`。
相机不是 state/contact 的主时钟；`camera_source < contact_source <= T[t]` 合法。
`obs[t]_before_action[t]` 中 action 是该步实际发布的 target，而不是 IK candidate。

处理阶段始终保持 `processed row i == raw row i`；不删中间行、不压紧、不拆 source
segments、不从其他 raw row 修复 tactile。正常时序抖动、camera fresh=False、
observation_valid=False 单独出现或 dense tactile 不可用，不参与 episode 准入。

- `N`：单个 raw/processed episode 行数；`T`：Zarr 总行数；`P`：点云点数。
- `xarm_base`：Real 世界坐标系。位置/XYZ/fingertip 用 m；关节/手目标用 rad。
- rot6d 无单位；点云 RGB 为 float32 `[0,1]`。Real 与 Sim 不可混用坐标系/标签。
- processed RGB 默认 `H=240,W=320`；CLI 点数支持 1024/2048/4096/8192。
  离线库允许其他正整数点数，实时 IPC/Policy shape 必须另行匹配。

## 1. Raw episode HDF5 v28

```text
episodes/<task>/episode_*/
    data.h5
    depth.h5
    rgb.mp4
```

所有 dataset 第一维等于 `meta.num_frames`。下表给出每行 shape；
RGB 为 uint8 aligned color，depth 为 uint16 depth-to-color aligned 图像。
`EpisodeReader` 只读当前 v28，校验必需文件、shape/dtype 和 sidecar 帧数；
reader/finalize 不完整解码 RGB，也不重放在线安全证明。

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
| arm_source_monotonic_ns, hand_source_monotonic_ns | scalar | uint64 |
| hand_contact_source_monotonic_ns, tactile_source_monotonic_ns | scalar | uint64 |
| vr_source_monotonic_ns, camera_source_monotonic_ns | scalar | uint64 |
| tactile_sum_fresh, tactile_fresh, tactile_calibrated | scalar | bool |
| tactile_unit_code | scalar | uint8 |
| flag_camera_fresh | scalar | bool |
| camera_health | scalar | uint8 |
| camera_depth_frame_number, camera_color_frame_number | scalar | uint64 |
| vr_wrist_pos | (3,) | float64 |
| vr_wrist_rot6d | (6,) | float64 |
| vr_landmarks | (21,3) | float64 |
| head_quat_wxyz | (4,) | float64 |

每个 controller-emitted sample 恰好一行，timestamp 是 observation/controller anchor 秒值。
新录制 `source_sample_index=0,1,...`、`fill_reason=0`、`flag_sample_valid=True`；
错过 tick 保留实际时间间隔，不制造插入行。磁盘 batch 只有 IO 职责。

`flag_frame_status`：0 OK、1 HELD、2 IK_FAIL、3 SAFETY_REJECT、4 RETARGET_FAIL。
当前已验证的 behavioral boundary 是连续 1–4 行 IK_FAIL 保留，连续 5 行起拒绝整条
episode；不重新调整阈值，其他 flags 不构成新的行为质量 detector。

`hand_contact` 是软件 bias 校正的 SDK `calc_force`。
录制在同一 control anchor 独立读取最新有效 aggregate：state_valid、tactile_sum_valid、
非 qpos_stale、有限 (5,3) payload 且 source/publication 因果成立。
它不回退 command feedback，也不改变 hand_qpos/current：
`hand_contact_source_monotonic_ns` 单独说明选中的 aggregate 来源，不能借用较新的
`hand_source_monotonic_ns` 或 dense 来源。无有效 aggregate 时保存 NaN/source=0，
不把无效读数伪装成无接触零值。`tactile_sum_fresh` 是选中 aggregate 的年龄 telemetry，
旧但有效读数不因该 flag=False 被删行。

`hand_tactile_force` 是 SDK `raw_force`，昂贵的原始 dense 数据保留在 raw。
`tactile_source_monotonic_ns` 仍属于 dense ring；
`tactile_fresh`、`tactile_calibrated`、`tactile_unit_code` 描述其 dense/provenance
telemetry，不为不同时间的 aggregate 提供虚构证明。
单位为 `xhand_sdk_native_unknown_si`，SI Newton 与 taxel 空间几何均未验证；
不假设 dense taxel 求和等于 aggregate。

Metadata 保留 schema_version、num_frames、control_hz、task_label、operator、camera identity、
depth_scale、native depth/color intrinsics/尺寸/distortion、`camera_T_color_from_depth`
与适用的 base/EEF 外参。点云几何边界校验 calibration、serial、rigid transforms。
camera health/fresh/frame numbers 是事实 telemetry，不是 offline timing-quality gate。

### Historical raw

历史 raw 保持历史版本。冻结 v24→v25 与 v25→v26 工具保留；不提供 v26/v27→v28
伪迁移，也不原地改写原始文件。直接 v25 的旧 0.1 tactile scale 由冻结转换器还原；
v24-derived 数据必须先证明 lineage，再显式选择 scale，不能靠数值大小猜测。

仅离线 processing 有一个窄的 normalized-v26 读取路径；其 aggregate 来源仍是原来的
hand_source，同一 raw row 原样映射。当前 runtime reader、raw viewer 与 physical replay
没有历史版本兼容路径。实际 pick_place_toy lineage、不可恢复的信息限制及验证结果见
[incident evidence](invalid_frames_export_incident.md) 和
[salvage manifest](../artifacts/pick_place_toy_salvage_manifest.json)。

## 2. Processed HDF5 v17

`episodes_processed/<task>/*.h5`，每个 accepted raw episode 一个文件。
Required core 与全部 profile 都使用同一 control-step 映射：

| Dataset | shape | dtype | 同行 raw 来源 |
|---|---|---|---|
| joint_state | (N,19) | float32 | arm_qpos + hand_qpos |
| action | (N,19) | float32 | action_arm_joint_sent + action_hand_joint |
| action_ee | (N,21) | float32 | action_arm_ee + action_hand_joint |
| contact_force | (N,5,3) | float32 | hand_contact |
| fingertip_points | (N,5,3) | float32 | 本行 joint_state 经共享 arm+hand FK |
| rgb | (N,H,W,3) | uint8 | 本行 RGB，resize/no crop |
| depth | (N,H,W) | uint16 | 本行 aligned depth，nearest resize |
| camera_intrinsic | (N,9) | float32 | resized color K，展平 |
| camera_extrinsic | (N,4,4) | float32 | T_xarm_base_from_color |
| point_cloud | (N,P,6) | float32 | 本行 RGB-D 经 canonical preprocessing |

`joint` 只含 core；`rgb` 加四项 RGB-D；`pointcloud` 加点云；
`rgb_pc` 两者都有。processed 不保存 `eef_pose`、dense `tactile_force` 或
`/provenance`。FK implementation 保留，fingertip 复用每步 Arm FK；
`action_ee` 是无法从 IK 后 joint target 无损恢复的控制意图，必须保留。

### 必需身份与语义 attrs

| 属性 | 值/含义 |
|---|---|
| schema_name / schema_version | dexmani-real-processed-hdf5 / 17 |
| domain / profile / task_name | real / profile / 单一有效任务名 |
| source_path / source_episode / source_schema_version | 实际输入 raw 身份；source_path 为绝对路径 |
| source_frames / episode_steps | 正整数且二者相等 |
| dt | control period，默认 1/16 s |
| obs_alignment | obs[t]_before_action[t] |
| observation_alignment | control_step_latest_causal |
| state_alignment | control_step |
| contact_force_source | raw_hand_contact_control_step |
| action_semantics | teleop_published_joint_target |

另外保留 contact representation/unit/frame/SI-unverified、fingertip frame/unit/derivation/
policy identity 与 action_ee frame/components。RGB profile 保留 resize、depth scale/invalid value、
intrinsic/extrinsic semantics 和必要相机几何；点云 profile 保留 frame/color/sampling/transform/
policy identity、shape、桌面平面，以及仅用于重现点云的 `processing_config_json`。
没有 row-decision JSON、quality summary、repair mask 或 source-segment attrs。

processing 独占 whole-episode admission。必需 payload 非有限、shape/dtype/帧数损坏、
必需媒体不可读取、严重 sample/timestamp identity 错误会失败；视觉数据要求 source
正值且不晚于 control anchor，但不比较 tactile 与 camera 时间。
persistent IK 是唯一自动行为质量拒绝；不会用 timing、tracking 或 dense flags 删行。

每份输出在原子发布前经过完整 owning validator。它验证 `source_frames==episode_steps`、
schema/shape/dtype/finite payload、action_ee canonical rot6d、RGB-D 几何与点云语义。
export 不再另造 source-row 检查。外部篡改的 source 身份不是密码学证明；
保留原始 raw 才能重建实验。

Annotation 仅支持：

```yaml
episode_name:
  include: true
  task_name: optional_task_override
```

省略 include 默认为 true；不支持 include_ranges/exclude_ranges。
task 名必须非空、无首尾空白/ASCII 控制字符，且不能是 unknown。全局 task override 不得
冲突 annotation；整批 accepted episodes 必须只有一个任务。CLI 跳过未标注 rejected
episode；显式 include 的失败阻断整批。直接 library 默认不跳过未标注 rejection。

## 3. Policy Zarr v10

```text
datasets/<task>.zarr/
    data/<profile keys>
    meta/episode_ends
```

每个 processed file 按文件名字典序追加成一个完整 episode。
`data/<key>` 为 `(T,*tail_shape)`，key/shape/dtype 与 processed profile 相同；
`episode_ends` 是 int64、严格递增、exclusive 累积行数，末值等于 T。

Exporter：发现 HDF5 → owning processed validation → task/profile/dt/shape/dtype/语义一致性 →
逐文件完整追加 → 写 episode_ends → 校验输出结构 → transactional publish。
任何无效 processed input 使本次导出失败，不创建局部 episode，不跳过部分 source 行。
已有输出（含符号链接）拒绝覆盖。

root attrs 为 `schema_name=dexmani-real-policy-zarr`、`schema_version=10`、`domain=real`、
profile/task_name/dt、`episode_start_policy=full_history`、同一 control-step alignment/action
语义及各 profile 必需 modality semantics；不复制 raw source paths、row provenance、
完整 camera calibration 或 audit JSON。

## Deployment、replay 与训练边界

Deployment history 的 reference 是 logical policy control-grid times。每个 reference
分别选择最新 causal camera/state/contact；不再以实际 camera exposure time 对齐机器人。
run_started/not_before、source/publication 因果、max_input_age、max_grid_lag、state_valid、
qpos_stale、camera health、calibration/unit、必要的 contact provenance source match 与显式
请求 dense tactile 时的 freshness 检查全部保留；训练接纳 sensor imperfection 不代表
在线放宽准入。command safety、collision、generation、freshness、shutdown 不变。

physical replay 验证 processed 后，仅按 source_path 读取完整 raw float64 命令，
不发送 processed float32 action；仍执行完整 live-start/limits/workspace/collision 检查。
normalized-v26 salvage 不由当前 raw reader 进行 physical replay。

contact/fingertip 的 finger 顺序为 thumb,index,middle,ring,pinky；
SDK sensor indices 0..4 对应 finger IDs (2,5,7,9,11)，定义在 robot/model.py。
不要把 raw dense SDK point order 当作已知 taxel XYZ/邻接关系。
部署仍可支持显式请求的 EEF/dense modality，但当前 processed/Zarr 不生产这些 key。

训练 consumer 应接受 v10 control-step semantics，并保存 dataset contract 到 checkpoint；
full_history 禁止用 episode 左侧 padding 冒充已观测历史。不要把某个模型的 horizon/padding
实例值写成通用合同。相邻 dexmani_policy 自 commit `2bc5b85` 严格接受 v10/control-step
合同并持久化 observation_alignment/state_alignment/contact_force_source；旧 v8/v9 不兼容。
实际 salvage Zarr 的读取和采样已验证，未执行真实 checkpoint export 或硬件 rollout。

Hardware validation: NOT RUN
