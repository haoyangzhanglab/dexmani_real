# CordVIP 离线几何入口

本目录连接 DexMani Real 的物体资产、Raw 数据与独立的 `cordvip_geometry` 几何核心，提供重建、点数分配和跟踪三个实验入口。当前用于离线分析，尚未接入主包策略观测。

## 数据流

**物体图像与实测尺寸 → 米制网格与表面点云 → 多物体点数分配 → 跟踪资产**。

**Raw RGB-D + 首帧物体掩膜 + 可选跟踪资产 → 逐行分割与位姿估计 → 离线几何结果**。

资产提供物体的完整模型几何，Raw 提供实际场景观测。有资产时可估计物体位姿；无资产时仍可提取分割区域内的实测点云。手部几何可由同一 Raw 行的实测关节与机器人模型计算。

## 脚本职责

| 入口 | 职责 |
| --- | --- |
| [reconstruct_object.py](reconstruct_object.py) | 串联物体分割、网格生成、尺度校准和资产准备。 |
| [allocate_points.py](allocate_points.py) | 按闭合物体的实体体积分配固定的总点数，生成多物体采样资产。 |
| [track_object.py](track_object.py) | 按 Raw 行顺序处理单物体或多物体场景，保存分割、实测点、位姿及可选手部几何。 |

重建与跟踪脚本调用子包的 CLI；点数分配脚本复用子包的资产读写与表面采样。模型算法集中在子包中，脚本负责衔接实验输入与输出。

重建流程见[物体重建](../../docs/object_reconstruction.md)，内部模块与结果含义见[几何核心架构](../../third_party/cordvip_geometry/docs/usage.md)。
