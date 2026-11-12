# 几何核心接口与流程

`cordvip_geometry` 提供米制物体资产与 RGB-D 位姿跟踪接口。主项目使用这些结果生成任务点云，并组织 Raw 标注与在线策略观测。

## 核心接口

以下接口均可从 `cordvip_geometry` 导入。

| 接口 | 职责 |
| --- | --- |
| [prepare_asset()](../src/cordvip_geometry/assets.py) | 根据实测尺度将源 Mesh 转为米制资产，保留物体局部原点。 |
| [load_asset()](../src/cordvip_geometry/assets.py) | 读取米制 Mesh 及其元数据。 |
| [RGBDFrame](../src/cordvip_geometry/rgbd.py) | 携带同一彩色相机网格上的 RGB、米制深度、内参与畸变信息。 |
| [SceneTracker](../src/cordvip_geometry/tracking.py) | 管理多物体分割、位姿估计与跟踪有效性判断。 |
| [TrackingFailure](../src/cordvip_geometry/foundationpose.py) | 表示当前场景无法继续可靠跟踪。 |

## 跟踪流程

1. **组织输入**：用 `RGBDFrame` 表示当前相机帧；主项目可通过 `RGBDFrame.from_camera()` 结合相机标定转换原始 RGB-D。
2. **加载场景**：通过 `with SceneTracker.open(assets, workspace=...) as tracker` 加载物体资产与模型，`assets` 将对象名称对应到各自的米制资产目录。
3. **初始化首帧**：调用 `detect_masks(frame, prompts)` 定位、分割各对象，再将结果传给 `step(frame, initial_masks=masks)` 初始化位姿。
4. **持续跟踪**：后续帧调用 `step(frame)`，结合分割、深度与 Mesh 更新各对象位姿，并检查观测与模型的一致性。
5. **切换 Episode**：新段先调用 `reset()` 清除场景状态，再从首帧初始化；模型与资产继续复用，退出上下文时释放资源。

## 输出与主项目衔接

`step()` 成功返回 `{object_name: pose_camera}`，每个位姿是物体局部坐标到相机坐标的变换。任一物体跟踪失效会抛出 `TrackingFailure`，调用方结束当前段，下一段重置场景后重新初始化。

主项目将这些位姿与相机外参组合，把固定物体模板变换到 xArm base 坐标系。离线标注与在线 HOI Worker 共用这条几何路径；手点云由主项目根据机器人关节状态和 FK 单独生成。

任务操作入口见 [CordViP 实验流程](../../../scripts/cordvip/README.md)，整体数据链路见[物体重建与 HOI](../../../docs/object_reconstruction.md)，后端模块与安装见[模块概览](../README.md)。
