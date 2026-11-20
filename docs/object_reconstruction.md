# 物体重建与 HOI

CordViP 的 `hand_pointcloud` 和 `object_pointcloud` 默认各包含 1024 个米制 float32 XYZ，坐标系均为 xArm Base。通用场景字段 `point_cloud` 单独由 RGB-D 反投影和场景滤波生成。

## 固定模板与坐标变换

- **手部。** 手模板按 URDF 的 visual 变换在 link 局部坐标系采样，以 `part_names` 和 `part_offsets` 保留分段和点序。xArm 的 7 个关节通过 FK 得到 `T_base_from_eef`，XHand 的 12 个 SDK 关节按现有映射进入 URDF FK。`hand_geometry()` 按 `T_base_from_eef @ T_eef_from_hand @ T_hand_from_link` 变换各段，并可同时生成指尖坐标。
- **物体。** CAD 网格为米制，简化后保留原点和朝向。任务模板的 `points_local` 按网格表面积分配，共 1024 点，每对象至少 64 点。FoundationPose 输出 `T_camera_from_object`，`transform_task_points()` 用 `T_base_from_camera @ T_camera_from_object` 将各段局部点变换到基座，沿用 `object_names` 和 `offsets` 的顺序。

## 实验流程

在 `real_robot` 环境中，从仓库根目录运行。为准备、采样、标注和部署指定同一个任务资产目录。采样按 `task.json` 的 `objects` 列表生成模板，后续标注与部署沿用模板的 `object_names` 顺序。每个对象的参考图默认为 `<name>/reference.jpg`，已有米制 CAD 放在 `<name>/mesh/mesh_m.obj`。资产准备的尺度、实例选择与后端依赖见[感知后端说明](../third_party/cordvip_geometry/README.md#安装与命令)。

```bash
python scripts/collect_teleop.py pick --episodes-dir /data/episodes
python scripts/cordvip/prepare_task.py --task-dir /data/tasks/pick
python scripts/cordvip/sample_template.py --task-dir /data/tasks/pick
python scripts/cordvip/annotate_episodes.py /data/episodes/pick --task-dir /data/tasks/pick
python scripts/export_zarr.py /data/episodes/pick --output /data/pick.zarr \
    --fields joint_state action hand_pointcloud object_pointcloud
```

`sample_template.py` 生成或复用任务目录下的 `hoi/template.npz`。更换 CAD 后，给模板采样与标注命令依次加 `--overwrite`，再导出到新的 Zarr 路径。项目已提供默认手模板。修改手部 URDF 或 visual 网格后，可用 `python scripts/generate_hand_template.py --overwrite` 重新生成。

标注支持单个 Episode 或直接包含 `episode_*` 子目录的数据目录，使用 Raw 中的 RGB-D、深度单位与相机外参。训练后通过 `python scripts/run_policy.py CHECKPOINT --task-dir /data/tasks/pick` 部署。完整训练命令见 [README 的 CordViP 流程](../README.md#从示教训练并部署-cordvip)。

### 跟踪配置

在线与离线默认使用 `sam2_video`，每 10 个处理帧周期刷新 SAM2，并在深度验证失败等事件发生时补充分割。安装 OSTrack 后，可用可信本地 Python 文件导出 `CONFIG`，为一次实验选择 `ostrack`。

```python
from dexmani_real.config.perception import HOITrackingConfig
from dexmani_real.config.real import RealConfig

CONFIG = RealConfig(
    hoi_tracking=HOITrackingConfig(
        tracker_2d="ostrack",
        segmentation_interval_frames=10,
        xy_prior_gain=0.2,
    ),
)
```

`annotate_episodes.py` 和 `run_policy.py` 均从 `--config experiment.py` 读取 `RealConfig.hoi_tracking`。使用新参数重算已有标注时传入 `--overwrite`。默认权重为 `third_party/weights/ostrack/OSTrack_ep0300.pth.tar`，可通过 `ostrack_checkpoint` 指定绝对路径或相对仓库根目录的路径。`xy_prior_gain=0` 关闭二维位姿辅助。`recovery_retry_interval_frames` 省略时，OSTrack 模式使用 8 帧，SAM2 视频模式使用 15 帧。

部署在连接设备前解析和校验 HOI 配置，有会话输出目录时将有效配置写入 `run_config.yaml` 的 `effective_parameters.hoi_tracking`。离线标注用同一套默认值和校验，并将有效配置写入 `hoi.h5` 根属性 `tracking_config`。安装和恢复流程见[感知后端说明](../third_party/cordvip_geometry/README.md#安装与命令)。

### 跟踪基准

在同一 CUDA 环境中，用同一个有标定、RGB-D 和任务模板的 Episode 分别回放两种模式。默认先逐帧预热 50 帧，再投递 300 个测量帧。`paired` 等待每帧发布，用于比较同一批输入的耗时。`paced` 按相机频率向 Camera Ring 送帧，分别报告投递数、实际处理数、有效 FPS 和来源到发布延迟。相机帧率取自录制配置，也可用 `--camera-fps` 显式指定。

```bash
python scripts/benchmark_ostrack.py paired EPISODE --task-dir TASK --tracker sam2_video
python scripts/benchmark_ostrack.py paired EPISODE --task-dir TASK --tracker ostrack
python scripts/benchmark_ostrack.py paced EPISODE --task-dir TASK --tracker sam2_video
python scripts/benchmark_ostrack.py paced EPISODE --task-dir TASK --tracker ostrack
```

独立 OSTrack 使用 `solo`，共驻 FoundationPose 和 SAM2 使用 `resident`。两者都用 `--box NAME X1 Y1 X2 Y2` 提供去畸变后首帧的 XYXY 框，可重复传入多个对象。Worker 回放使用常规初始化流程，可用 `--xy-prior-gain 0` 核对二维先验的影响。

## 跟踪与逐行快照

在线 Worker 从 Camera Ring 读取，离线标注从 Raw 读取。两条路径共用 `track_hoi_frame()` 和 `transform_task_points()`，按模板顺序处理位姿和有效性。离线标注额外保存逐物体质量。每个在线请求和每个离线 Raw Episode 分别初始化跟踪，初始化失败时用下一来源帧重试。在线跳过已处理的相机序列，离线要求同序列具有相同来源时间，并复用跟踪结果。意外后端错误沿各入口的故障路径传播。质量与恢复规则见[后端说明](../third_party/cordvip_geometry/README.md#质量检查)和[遮挡恢复](../third_party/cordvip_geometry/README.md#遮挡恢复)。

在线将当前控制采样行的最新 Arm / Hand 实测关节状态与可用的 HOI 物体点云组合。离线将 `hoi.h5` 的物体点云与对应 Raw 行的 `arm_qpos`、`hand_qpos` 组合。两条路径都用本行实测关节生成 `joint_state`，并通过 FK 计算 `eef_pose`、指尖和手点云。

软件按观测行配对快照，相机、Arm 和 Hand 可以在不同物理时刻采样。物体点云仍对应原始相机序列和来源时间，Arm 和 Hand 各自保留反馈时间。在线继续检查 HOI 有效性、新鲜度和设备反馈。参见[部署恢复流程](policy_deployment.md#hoi-暂停与恢复)和[实验数据](data.md)。

## 标注与查看

标注沿用 Raw 相机来源时间与序列。无效对象的位姿为 NaN，保留逐对象有效性和已有质量指标。只有全部对象有效时才计算该帧物体点云，否则点云为零且整帧无效。离线分块计算并原子发布 `hoi.h5`。复用标注前检查点云和有效性数组的结构，并核对 Raw 的相机时间、序列与外参。`--overwrite` 在重算成功后替换旧文件。

| HDF5 字段 | 内容 |
| --- | --- |
| `object_pointcloud` | 基座系物体点云，float32 `[T,1024,3]` |
| `object_pose_camera`、`object_valid` | 物体到相机位姿与逐对象有效性 |
| `quality/`、`template/` | 逐对象质量指标和有序固定模板 |
| `camera_timing`、`base_from_camera` | 与 Raw 核对的相机来源时间、序列和外参 |

```bash
python scripts/visualize_episode.py /data/episodes/pick/episode_001 --max-frames 50
python -B -m unittest discover -s tests -v
```

用实际 Episode 路径核对同一 Raw 行的手物点云。Time Series 的 HOI 页签显示整帧有效性、逐物体质量，以及 Arm 和 Hand 各自的反馈时间减去相机来源时间的毫秒差。可视化保留无效行的时间位置和本行 FK 手部几何，隐藏无效物体点云。`--no-hoi` 跳过 HOI 标注。严格 HOI 导出要求所选行的物体均有效。完整规则见[数据导出](data.md#按需导出)。
