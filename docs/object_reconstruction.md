# 物体重建与点云资产

物体重建属于独立的 CordVIP 离线几何流程，将单张物体图像和实测尺寸转换为米制网格及固定采样点云，用于物体几何分析与位姿跟踪。该流程尚未接入主包的策略观测。

## 数据流

**物体图像 → 文本检测与分割 → 前景图像 → 网格生成 → 尺度校准 → 表面点云与法向**。

Grounding DINO 根据物体描述定位候选，SAM2 生成分割掩膜；确认前景后，Hunyuan3D 生成物体网格。网格依据实测尺寸转换为米制，再按表面面积采样点及对应法向，保留物体局部坐标系。

单图不可见部分由模型推断，真实尺度来自外部测量。输出资产包括网格、点云、法向和来源元数据；点云可独立用于几何分析，FoundationPose 位姿跟踪还需要网格。

## 核心模块

算法实现位于 [cordvip_geometry](../third_party/cordvip_geometry/)，主仓库通过 [scripts/cordvip](../scripts/cordvip/) 提供实验入口。

| 模块 | 职责 |
| --- | --- |
| [reconstruct_object.py](../scripts/cordvip/reconstruct_object.py) | 串联图像分割、网格重建与资产准备。 |
| [allocate_points.py](../scripts/cordvip/allocate_points.py) | 为多物体资产分配固定的总采样点数。 |
| [track_object.py](../scripts/cordvip/track_object.py) | 结合物体资产与 Raw RGB-D 数据进行位姿跟踪和离线几何分析。 |

## 与主项目的关系

机器人与手部几何由主包的 `geometry`、`perception` 提供，物体几何由 CordVIP 离线生成。两者可在离线分析中结合，但物体重建不是示教录制或常规 Zarr 导出的前置步骤。

已有物体模板存放于 [assets/pointcloud_template/object](../assets/pointcloud_template/object/)。

返回[项目架构](../README.md)。
