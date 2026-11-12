# CordViP 实验入口

这些脚本连接物体几何后端与 DexMani Real 的实验数据，完成任务资产准备、固定模板生成、Raw 标注与训练数据导出。整体表示与架构见[物体重建与 HOI](../../docs/object_reconstruction.md)。

## 任务级 HOI 流程

**米制物体资产 → 固定任务模板 → Raw 离线标注 → 检查结果 → Zarr 训练数据**。

| 阶段 | 入口 | 作用与产物 |
| --- | --- | --- |
| 准备物体资产 | [build_task.py](build_task.py) | 按任务配置重建物体并确定实测尺度，或复用已有米制 Mesh。 |
| 固定任务模板 | [allocate_points.py](allocate_points.py) | 从各物体 Mesh 采样，固定任务点云的对象分段与点序。 |
| 标注示教 | [annotate_episodes.py](annotate_episodes.py) | 从已采集的 Raw RGB-D 跟踪物体，将整段成功的物体点云写回 Raw。 |
| 检查结果 | [visualize_episode.py](../visualize_episode.py) | 查看录制数据与标注点云。 |
| 导出训练数据 | [export_zarr.py](../export_zarr.py) | 将物体点云、由关节状态生成的手点云及所选状态、动作导出为 Zarr。 |

任务资产位于 `assets/pointcloud_template/object/<task_name>/`，按物体组织 Mesh，并保存任务级模板。实验期间固定资产与模板，训练数据导出和在线部署使用同一套几何表示。

Zarr 交给 dexmani_policy 训练；部署时使用同一任务的资产，从实时 RGB-D 跟踪生成物体点云。

## 单物体重建

[reconstruct_object.py](reconstruct_object.py) 提供独立的参考图分割与 Mesh 重建入口。提供实测尺寸后可生成米制资产，供任务模板采样与位姿跟踪使用。

脚本从仓库根目录运行，具体参数见各入口的 `--help`。后端准备见 [cordvip_geometry 安装与命令](../../third_party/cordvip_geometry/README.md#安装与命令)，接口流程见[几何核心接口](../../third_party/cordvip_geometry/docs/usage.md)。
