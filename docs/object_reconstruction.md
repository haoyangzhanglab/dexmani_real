# 物体重建与 HOI

CordViP 的 `hand_pointcloud` 和 `object_pointcloud` 默认各包含 1024 个米制 float32 XYZ，坐标系均为 xArm Base。通用场景字段 `point_cloud` 单独由 RGB-D 反投影和场景滤波生成。

## 固定模板与坐标变换

- **手部。** 手模板按 URDF 的 visual 变换在 link 局部坐标系采样，以 `part_names` 和 `part_offsets` 保留分段和点序。xArm 的 7 个关节通过 FK 得到 `T_base_from_eef`，XHand 的 12 个 SDK 关节按现有映射进入 URDF FK。`hand_geometry()` 按 `T_base_from_eef @ T_eef_from_hand @ T_hand_from_link` 变换各段，并可同时生成指尖坐标。
- **物体。** CAD 网格为米制，简化后保留原点和朝向。任务模板的 `points_local` 按网格表面积分配，共 1024 点，每对象至少 64 点。FoundationPose 输出 `T_camera_from_object`，`transform_task_points()` 用 `T_base_from_camera @ T_camera_from_object` 将各段局部点变换到基座，沿用 `object_names` 和 `offsets` 的顺序。

## 实验流程

在 `real_robot` 环境中，从仓库根目录运行。标注和部署前先安装 HOI 跟踪依赖并准备模型权重。准备、采样、标注和部署共用同一个任务资产目录。采样按 `task.json` 的 `objects` 列表生成模板，后续沿用模板的 `object_names` 顺序。每个对象的参考图默认为 `<name>/reference.jpg`，已有米制 CAD 放在 `<name>/mesh/mesh_m.obj`。[安装与权重路径](../third_party/cordvip_geometry/README.md#安装与命令)

```bash
python scripts/collect_teleop.py pick --episodes-dir /data/episodes
python scripts/cordvip/prepare_task.py --task-dir /data/tasks/pick
python scripts/cordvip/sample_template.py --task-dir /data/tasks/pick
python scripts/cordvip/annotate_episodes.py /data/episodes/pick --task-dir /data/tasks/pick
python scripts/export_zarr.py /data/episodes/pick --output /data/pick.zarr \
    --fields joint_state action hand_pointcloud object_pointcloud
```

`prepare_task.py` 先检查待重建对象的参考图路径、尺度和重建参数，再加载模型。米制网格写完后才发布到 `<name>/mesh/mesh_m.obj`。`sample_template.py` 在生成或复用 `hoi/template.npz` 前检查所需 CAD 文件，生成时最多支持 16 个对象。更换 CAD 后，给模板采样与标注命令依次加 `--overwrite`，再导出到新的 Zarr 路径。项目已提供默认手模板。修改手部 URDF 或 visual 网格后，可用 `python scripts/generate_hand_template.py --overwrite` 重新生成。

标注支持单个 Episode 或直接包含 `episode_*` 子目录的数据目录。新标注先校验 Raw 相机时间、序列、段界、标定和深度数组，再加载模型跟踪。训练后通过 `python scripts/run_policy.py CHECKPOINT --task-dir /data/tasks/pick` 部署。完整训练命令见 [README 的 CordViP 流程](../README.md#从示教训练并部署-cordvip)。

### 跟踪配置

在线与离线统一使用 OSTrack 和 FoundationPose 跟踪。SAM2 默认每 10 个处理帧做一次单图分割，深度证据不足时可限频补充分割。用可信本地 Python 文件导出 `CONFIG` 可以调整跟踪参数。

```python
from dexmani_real.config.perception import HOITrackingConfig
from dexmani_real.config.real import RealConfig

CONFIG = RealConfig(hoi_tracking=HOITrackingConfig(segmentation_interval_frames=12))
```

`annotate_episodes.py` 和 `run_policy.py` 均从 `--config experiment.py` 读取 `RealConfig.hoi_tracking`。使用新参数重算已有标注时传入 `--overwrite`。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `ostrack_checkpoint` | `third_party/weights/ostrack/OSTrack_ep0300.pth.tar` | 接受绝对路径或相对仓库根目录的路径。 |
| `segmentation_interval_frames` | `10` | 按实际处理帧计数的分割周期。 |
| `lost_timeout_s` | `0.1` | 当前位姿未通过验证时，距该物体最近一次有效观测达到此年龄，即在当前帧尝试重新检测。 |
| `recovery_enabled` | `True` | 启用重新检测与注册。关闭后继续用局部跟踪尝试恢复。 |
| `recovery_retry_interval_frames` | `2` | 相邻两次恢复尝试至少相隔的处理帧数。 |
| `recovery_crop` | `True` | 用当前 OSTrack 框裁剪单物体首次恢复的 DINO 输入。 |
| `xy_prior_gain` | `0.2` | 二维框对位姿初值的修正增益，设为 `0` 可关闭框修正。 |
| `kalman_xy_prior` | `False` | 实验性开关，将已验证位姿的 XY 运动预测用于普通跟踪初值。 |

部署在连接设备前解析和校验 HOI 配置，有会话输出目录时将有效配置写入 `run_config.yaml` 的 `effective_parameters.hoi_tracking`。离线标注用同一套默认值和校验，并将有效配置写入 `hoi.h5` 根属性 `tracking_config`。安装和恢复流程见[感知后端说明](../third_party/cordvip_geometry/README.md#安装与命令)。

## 跟踪与逐行快照

在线 Worker 从 Camera Ring 读取，离线标注从 Raw 读取。两条路径共用 `track_hoi_frame()` 和 `transform_task_points()`，按模板顺序处理位姿和有效性。离线标注额外保存逐物体质量。每个在线请求和每个离线 Raw Episode 分别初始化跟踪，初始化失败时用下一来源帧重试。在线跳过已处理的相机序列，离线要求同序列具有相同来源时间，并复用跟踪结果。意外后端错误沿各入口的故障路径传播。质量与恢复规则见[后端说明](../third_party/cordvip_geometry/README.md#质量检查)和[跟踪与恢复](../third_party/cordvip_geometry/README.md#跟踪与恢复)。

在线将当前控制采样行的最新 Arm / Hand 实测关节状态与可用的 HOI 物体点云组合。离线将 `hoi.h5` 的物体点云与对应 Raw 行的 `arm_qpos`、`hand_qpos` 组合。两条路径都用本行实测关节生成 `joint_state`，并通过 FK 计算 `eef_pose`、指尖和手点云。

软件按观测行配对快照，相机、Arm 和 Hand 可以在不同物理时刻采样。短暂失效时，在线可沿用同一段上一次通过校验的完整物体点云和伴随 RGB，使用原始相机时间检查时效。Arm 和 Hand 保留各自的实测反馈时间。[部署恢复流程](policy_deployment.md#hoi-暂停与恢复) · [实验数据](data.md)

## 标注与查看

标注沿用 Raw 相机来源时间与序列。每行按观测时间与来源相机时间之差检查时效，达到 `execution.hoi_max_age_s`（默认 0.25 秒）即无效。同一相机序列对应的重复行也逐行检查。时效内缺少完整点云时，可沿用同一 Raw 段最近一次直接通过跟踪校验的完整观测，仍按该观测的原始相机时间计算年龄。`tracking_config.short_hold.held_rows` 记录保持行的零基 Raw 行号及实际采用的相机序列。保持和过期行的几何质量指标留空，无效对象的位姿存为 NaN，整帧不完整时物体点云为零。

首次短暂失跟时，可用当前 RGB-D 验证备用位姿。仍采用备用位姿的行和物体名记录在 `tracking_config.provisional_rows`。开启 `--repair-gaps` 时可用可靠位姿重新 refinement 这些行。

离线分块计算并原子发布 `hoi.h5`。复用、可视化和导出使用同一套标注结构及 Raw 相机对齐检查。`--overwrite` 在重算成功后替换旧文件。

离线标注可加 `--repair-gaps`。同一 Raw 段内，缺帧两侧须有通过几何验证且未过期的物体位姿，两者相距不超过 1 秒。先从后一个位姿反向 refinement，仍无效的帧再以前一个可靠位姿为初值分别 refinement。每帧须通过当帧的时效检查和 RGB-D 几何验证。证据不足时停止，接回相邻有效帧时拒绝超过 90° 的单帧翻转。回补结果使用该帧自身的几何指标，保持资格沿用原跟踪结果。最终保留的回补位姿按物体名记录相机序列，见 `tracking_config.offline_backtrack`。用 `--repair-gaps --overwrite` 重算已有标注。[GALATEA 的反向恢复](https://github.com/boyuan-an/GALATEA/blob/36b90481ba458d9beb5b6ac1396c639ea49a4747/recon/galatea_recon/pipeline/object_recovery_fp.py#L287-L300) · [CordViP 的逐帧跟踪](https://github.com/xuanxuanzzzii/cordvip/blob/ad6d441abe7b0480cf79b0311db266965c1313b6/generate_pc/FoundationPose/get_foundation.py#L107-L152)

| HDF5 字段 | 内容 |
| --- | --- |
| `object_pointcloud` | 基座系物体点云，float32 `[T,1024,3]` |
| `object_pose_camera`、`object_valid` | 物体到相机位姿与逐对象可用性，包含期限内的保持位姿 |
| `quality/`、`template/` | 逐对象质量指标和有序固定模板 |
| `camera_timing`、`base_from_camera` | 与 Raw 核对的相机来源时间、序列和外参 |

```bash
python scripts/visualize_episode.py /data/episodes/pick/episode_001 --max-frames 50
```

用实际 Episode 路径核对同一 Raw 行的手物点云。Time Series 的 HOI 页签显示整帧有效性、逐物体质量，以及 Arm 和 Hand 各自的反馈时间减去相机来源时间的毫秒差。可视化保留无效行的时间位置和本行 FK 手部几何，隐藏无效物体点云。`--no-hoi` 跳过 HOI 标注。严格 HOI 导出要求所选行的物体均有效。完整规则见[数据导出](data.md#按需导出)。
