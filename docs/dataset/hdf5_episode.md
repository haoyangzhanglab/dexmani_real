# DexMani Real schema v17

本文只描述 Real runtime 的当前 episode 合同。字段的最终来源是
[`recording/episode_schema.py`](../../dexmani_real/recording/episode_schema.py)、writer 和
[`recording/episode_reader.py`](../../dexmani_real/recording/episode_reader.py)；本文不承载历史审计报告。

Sim 数据请读独立的 [`sim_hdf5_zarr.md`](sim_hdf5_zarr.md)，Real→Sim 的标签边界请读
[`real_to_sim_mapping.md`](real_to_sim_mapping.md)。两者不能直接套用 Real 字段语义。

## 1. 发布单元

一个已发布 episode 是目录而不是单个 HDF5：

```text
episode_.../
├── data.h5     # 控制网格、状态、命令、质量和 metadata
├── depth.h5    # 与控制网格对齐的 depth 侧车
└── rgb.mp4     # 与控制网格对齐的 RGB 侧车
```

writer 先写临时目录，完成帧数、schema、相机侧车和 metadata 校验后再原子发布。失败、相机写盘错误、队列溢出、codec 错误或磁盘错误不能发布半成品。

runtime 只接受 `schema_version=17`。非 v17 文件必须在 runtime 外迁移；不能只修改
version attribute 伪装成 v17。

## 2. 固定合同

| 项目 | 当前语义 |
|---|---|
| 行语义 | 一个 `1 / control_hz` 控制网格槽；不是传感器到达事件 |
| 数量 | `N = /meta.num_frames`，所有逐帧 dataset 首维都应为 `N` |
| 基础 dataset | `BASE_DATASET_SPECS_V17` 定义的 93 个无条件字段 |
| sent arm stream | `/meta.arm_sent_stream=True` 且存在 `action_arm_joint_sent` 时增加；不是所有 v17 都无条件拥有 |
| dtype/shape | 由 `episode_schema.py` 统一定义；writer、reader、finalizer 共用校验器 |
| recorder backend | RecorderIO 消费固定共享内存控制/样本 payload，并输出 schema-v17 episode |
| point cloud | 不写 `pointcloud.h5`；从 depth、相机 metadata 和点云配置在消费边界确定性派生 |

不要根据 dataset 数量猜 schema；先读 `/meta.schema_version`、`arm_sent_stream` 和实际 keys，再调用 `EpisodeReader.require_valid()`。

## 3. Shape、单位和坐标

约定：`A=7`（xArm）、`H=12`（XHand）、`N` 为 episode 网格槽数。角度字段以 rad 为单位，位置以 m 为单位，除非字段名或 metadata 明确说明其他单位。

- `arm_qpos`、`arm_qvel`、`arm_tau`：`(N, A)`；`arm_tau` 是 SDK 的 current-estimated effort，SI 单位未验证，不得当作 N·m 真值。
- `hand_qpos`、`hand_current`：`(N, H)`；手部触觉字段的单位由 metadata 说明，未知时不能解释成 N。
- `arm_ee`、`action_arm_ee`：`(N, 9)` 的 position + rot6d，当前受支持 runtime 使用 `xarm_base`；当前 `world == xarm_base` 是运行不变量，不代表支持任意 base pose。
- `hand_fingertip`：`(N, 5, 3)`，继承同一 Real robot frame。
- `vr_landmarks`：`(N, 21, 3)`，为 Unity→FLU 后的 VR 坐标，尚未变换到机器人 frame。
- 四元数按 `[w,x,y,z]`；rot6d 是旋转矩阵前两列，不是欧拉角。
- `timestamp` 是控制网格的 monotonic 秒；暂停期间不合成 hold 样本，恢复后的新 generation 会留下时间跳变。

相机字段要区分 pixel viewport、depth source Z 和外参 frame。`camera_K` 不能单独证明 aligned depth 已成为 color-optical Z；消费点云时以保存的 profile、depth scale、外参和处理器配置为准。

## 4. 字段分组

完整字段、tail shape 和 dtype 只看 `BASE_DATASET_SPECS_V17`。以下分组用于定位，不是第二份 schema：

| 分组 | 典型字段 | 语义 |
|---|---|---|
| 网格/因果 | `timestamp`, `source_sample_index`, `source_timestamp`, `fill_reason`, `flag_sample_valid` | 槽位是否来自真实 source，以及是否由 causal fill 产生 |
| arm 反馈 | `arm_qpos`, `arm_qvel`, `arm_tau`, `arm_ee`, `arm_connected` | 设备反馈和 FK 状态 |
| hand 反馈 | `hand_qpos`, `hand_fingertip`, `hand_contact`, `hand_tactile_*`, `hand_*_valid` | 手部状态、触觉和健康标志 |
| action | `action_arm_joint`, `action_arm_joint_raw`, `action_hand_joint`, `action_hand_joint_raw`, `action_arm_ee` | 候选、最终发布选择和诊断 raw |
| VR | `vr_wrist_pos`, `vr_wrist_rot6d`, `vr_landmarks` | 网格对齐的输入观测 |
| camera/quality | `camera_*`, `flag_camera_fresh`, `flag_frame_status`, `flag_ik_ok`, `flag_retarget_ok`, `flag_held` | 相机质量、控制结果和性能诊断 |

## 5. Action 语义

`action` 不是一个无条件等价的“机器人实际动作”：

- `action_arm_joint` 是 SafetyGate 后的 arm candidate/final selection。
- `action_arm_joint_sent` 只在 sent stream 开启时表示已转发给 arm worker；它不是固件执行 ACK。
- `action_hand_joint` 是 latest-wins ring 的 queued target，没有独立的 sent/ACK stream。
- `action_arm_ee` 是 IK tracking intent，不是 `action_arm_joint_sent` 的无损重表达。
- `*_raw` 只在对应成功路径有 solver/retarget raw 语义。held、失败或兼容路径可能写入最终 hold 值；读取 raw 前使用 `EpisodeReader` 的 validity mask。

因此，训练或 replay 配置必须显式选择字段，不能把 `action_arm_joint`、sent target 和 measured qpos 混用。

## 6. 相机侧车

`data.h5` 不保存 RGB 数组；RGB 从 `rgb.mp4` 顺序解码。`depth.h5` 保存 `/depth` 及其 metadata。标准发布要求两个侧车与 `N` 对齐；`has_camera=True` 不代表每一槽都 fresh，逐槽使用 `flag_camera_fresh`、`camera_health` 和 `camera_age_s`。

世界点云是消费边界的派生值：

```text
depth + camera_K + camera extrinsic + depth_scale + pointcloud config
    → PointCloudProcessor
    → finite、固定点数、带 RGB 的 world/Real-native cloud
```

派生点云不能仅改名为 SAPIEN-world 点云；Real frame、点数、滤波和单位必须随输出一起声明。

## 7. 有效性与读取

推荐读取路径：

```python
from dexmani_real.recording.episode_reader import EpisodeReader

with EpisodeReader(episode_dir) as reader:
    reader.require_valid("training or replay")
    qpos = reader.h5f["arm_qpos"][:]
    actions = reader.h5f["action_arm_joint"][:]
    rgb = reader.read_camera_all("rgb")
```

`VALID` 只表示容器、schema、shape、dtype、长度、时间和必要侧车满足 reader 合同；它不表示任务成功、每帧 observation 完整、action 已被硬件执行，或相机/触觉每帧 fresh。

常见下游筛选：

```python
rows = (
    reader.h5f["flag_sample_valid"][:]
    & ~reader.h5f["flag_held"][:]
    & reader.h5f["flag_ik_ok"][:]
    & reader.h5f["flag_retarget_ok"][:]
)
```

具体任务应按使用的模态补充 arm/hand feedback、camera freshness、point-cloud quality 和 continuity 条件。不要删除坏行后把两侧时间序列拼成一条新轨迹；离线清洗由 [`data_processing/`](../../dexmani_real/data_processing) 负责切段并记录 source range。

## 8. Metadata 中容易误读的字段

- `/meta.success` 是 episode 保存/发布结果，不是任务成功。
- `/meta.min_frames_met` 是质量标签，不单独决定 episode 是否可读。
- `/meta.duration`/`wall_duration_s` 包含未采样时段；`grid_duration_s` 只描述网格槽。
- `/meta.ik_hold_frame_count`、`camera_invalid_frame_count`、
  `observation_invalid_frame_count`、`sample_invalid_frame_count`、
  `safety_reject_frame_count` 汇总逐帧质量标志；
  `command_quiescence_count` 统计 timestamp 中明确保留的静默分段。
- `/meta.hand_read_error_count` 和 `hand_overcurrent_count` 是本 episode 录制窗口内
  XHand worker 的累计事件差值；瞬态事件即使未被 16 Hz 帧命中也不会从摘要消失。
  是否升级为运行时 sticky fault 由 resolved config 中的 XHand 过流次数/时间窗阈值决定。
- `resolved_config_sha256` 不能还原完整配置；需要外部配置文件。
- 触觉和 effort 的未知单位必须保留为 unknown，不能为了训练方便补写 SI 单位。
- `flag_sample_valid=True` 不等于 `flag_observation_valid`、IK、retarget、camera 或 tactile 都有效。

## 9. 修改格式时

先修改 `recording/episode_schema.py`，再同步 writer、reader、processing、visualization、replay 和文档。任何字段名、shape、dtype、单位、frame 或时间语义变化都应按 schema 变更处理；不要在 v17 文件中静默混入新含义。
