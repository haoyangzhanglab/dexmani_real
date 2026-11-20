# 物体重建与 HOI

CordViP 的 `hand_pointcloud` 和 `object_pointcloud` 默认各包含 1024 个米制 float32 XYZ，坐标系均为 xArm Base。通用场景字段 `point_cloud` 单独由 RGB-D 反投影和场景滤波生成。

## 固定模板与坐标变换

- **手部。** 手模板按 URDF 的 visual 变换在 link 局部坐标系采样，以 `part_names` 和 `part_offsets` 保留分段和点序。xArm 的 7 个关节通过 FK 得到 `T_base_from_eef`，XHand 的 12 个 SDK 关节按现有映射进入 URDF FK。`hand_geometry()` 按 `T_base_from_eef @ T_eef_from_hand @ T_hand_from_link` 变换各段，并可同时生成指尖坐标。
- **物体。** CAD 网格为米制，简化后保留原点和朝向。任务模板的 `points_local` 按网格表面积分配，共 1024 点，每对象至少 64 点。FoundationPose 输出 `T_camera_from_object`，`transform_task_points()` 用 `T_base_from_camera @ T_camera_from_object` 将各段局部点变换到基座，沿用 `object_names` 和 `offsets` 的顺序。

## 实验流程

在 `real_robot` 环境中，从仓库根目录运行。为准备、采样、标注和部署指定同一个任务资产目录。其 `task.json` 的 `objects` 列表定义对象名称和顺序。每个对象的参考图默认为 `<name>/reference.jpg`，已有米制 CAD 放在 `<name>/mesh/mesh_m.obj`。资产准备的尺度、实例选择与后端依赖见[感知后端说明](../third_party/cordvip_geometry/README.md#安装与命令)。

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

在线与离线默认每 10 个处理帧做一次 SAM2 周期校正，并开启丢失恢复。用可信本地 Python 文件导出 `CONFIG` 即可调整参数。

```python
from dexmani_real.config.perception import HOITrackingConfig
from dexmani_real.config.real import RealConfig

CONFIG = RealConfig(
    hoi_tracking=HOITrackingConfig(
        segmentation_interval_frames=15,
        recovery_enabled=True,
    ),
)
```

`annotate_episodes.py` 和 `run_policy.py` 均从 `--config experiment.py` 读取 `RealConfig.hoi_tracking`。使用新参数重算已有标注时传入 `--overwrite`。部署有会话输出目录时，将两个实际参数写入 `run_config.yaml` 的 `effective_parameters.hoi_tracking`。标注将完整后端配置写入 `hoi.h5` 根属性 `tracking_config`。

## 跟踪与时间对齐

在线 Worker 从 Camera Ring 读取，离线标注从 Raw 读取。两条路径共用 `track_hoi_frame()` 和 `transform_task_points()`，按模板顺序处理位姿、有效性与质量。每个在线请求和每个离线 Raw Episode 分别初始化跟踪，失败时用下一来源帧重试。在线跳过已处理的相机序列，离线同序列的记录行复用跟踪结果。意外后端错误沿各入口的故障路径传播。质量与恢复规则见[后端说明](../third_party/cordvip_geometry/README.md#质量检查)和[遮挡恢复](../third_party/cordvip_geometry/README.md#遮挡恢复)。

在线从反馈历史查找相邻端点，离线在各 Raw 连续段内批量查找。两者使用同一插值规则，将所需臂手关节对齐到物体点云对应的原始相机时间。精确命中直接取样，其余目标须被前后反馈包围，且间隔不超过 `execution.hoi_max_joint_gap_s`（默认 0.10 秒）。相同时间戳的冲突反馈会报错。导出和部署可通过同一个 `--config experiment.py` 设置该间隔。

对齐关节用于策略的 `joint_state`、`eef_pose`、指尖和手点云。Raw 和设备反馈检查使用实测反馈。在线对齐缺失时走现有暂停与恢复流程，离线严格导出报告不可对齐的设备与原始行号。参见[部署恢复流程](policy_deployment.md#hoi-暂停与恢复)和[实验数据](data.md)。

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

用实际 Episode 路径核对手物点云在同一帧的基座几何、关节来源时差、逐物体质量和有效性。可视化保留无效行的时间位置，并隐藏无效物体点云和无法对齐的手部几何。使用 `--no-hoi` 可查看原始关节几何。严格 HOI 导出要求所选行的物体均有效，且所需关节均能对齐。完整规则见[数据导出](data.md#按需导出)。
