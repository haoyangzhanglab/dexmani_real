# CordVIP 几何核心

`cordvip_geometry` 提供物体资产准备、RGB-D 分割与位姿跟踪，以及物体和手部点云分析。CLI 与 Python API 共用同一套几何核心；DexMani Real 的实验脚本负责衔接 Raw 数据与结果保存。

## 架构与数据流

资产链：**物体图像 → Grounding DINO 检测 → SAM2 分割 → Hunyuan3D 网格生成 → 实测尺度校准 → 固定表面点与法向**。

跟踪链：**RGB-D 与首帧掩膜 → SAM2 掩膜传播 → 实测点云提取 → 可选 FoundationPose 位姿估计与深度支持检查 → 场景几何结果**。

两条链通过米制物体资产连接。首帧掩膜可由检测与人工选择获得，也可直接使用已有掩膜。提供资产的对象进行模型位姿跟踪；未提供资产的对象仅输出分割和实测点云，两类对象可存在于同一场景。

## 核心模块

以下模块位于 [src/cordvip_geometry](../src/cordvip_geometry/) 下。

| 模块 | 职责 |
| --- | --- |
| [sam2.py](../src/cordvip_geometry/sam2.py) | 对接 Grounding DINO 与 SAM2，提供目标检测、首帧分割及掩膜传播。 |
| [reconstruction.py](../src/cordvip_geometry/reconstruction.py) / [assets.py](../src/cordvip_geometry/assets.py) | 生成网格，校准尺度，采样与加载物体资产。 |
| [rgbd.py](../src/cordvip_geometry/rgbd.py) / [geometry.py](../src/cordvip_geometry/geometry.py) | 表达 RGB-D 与相机几何，提供网格采样、投影和坐标变换。 |
| [tracking.py](../src/cordvip_geometry/tracking.py) | 组织多物体跟踪，组合分割、实测点云、模型位姿和深度支持结果。 |
| [foundationpose.py](../src/cordvip_geometry/foundationpose.py) / [model_depth.py](../src/cordvip_geometry/model_depth.py) | 估计物体位姿，渲染模型深度并供跟踪核心检查与实测深度的一致性。 |
| [raw.py](../src/cordvip_geometry/raw.py) / [hand.py](../src/cordvip_geometry/hand.py) | 适配 Raw 逐行输入，利用实测关节和 URDF 正运动学生成手部几何。 |
| [cli](../src/cordvip_geometry/cli/) / [io.py](../src/cordvip_geometry/io.py) / [viewer.py](../src/cordvip_geometry/viewer.py) | 编排实验流程、保存结果并提供离线可视化。 |

## Python API

`RGBDFrame` 统一表达图像、米制深度和相机标定；`SceneTracker.open()` 加载资产、校验参数并管理模型生命周期，`step()` 按输入顺序生成场景结果。`reset()` 清除场景状态并保留模型；`detect_masks()` 批量检测且拒绝有歧义的类别。HOI 离线入口跨 Episode 复用模型，每段重新初始化，每次丢失最多三次重新注册；在线不自动重新注册。`model_points` 根据有效位姿变换物体模板，得到相机或指定目标坐标系下的模型点。

`track_raw` 将 DexMani Real 的 Raw 逐行转换为跟踪输入，可附加同一行实测关节对应的手部几何。它返回内存结果，保存由调用方负责；宿主也可直接提供相机帧。HOI 离线入口按 camera_sequence 去重并写入 Raw 内部 /hoi；在线 Worker 复用同一跟踪核心，不修改 RawWriter。

安装脚本幂等应用 `scripts/foundationpose_tracking_cache.patch`，使单姿态跟踪复用 CUDA allocator 缓存；多候选注册保留释放行为。已有 checkout 可在宿主仓库根目录先用 `git apply --directory=third_party/FoundationPose --reverse --check third_party/cordvip_geometry/scripts/foundationpose_tracking_cache.patch` 检查是否已应用；未应用时先 `--check` 再 `git apply`，不覆盖冲突。正式链路使用现有 SAM2/FP，TensorRT 仅作后续 Benchmark。实测结论和限制见[当前实现与性能结论](../../../docs/object_reconstruction.md#当前实现与性能结论)。

## 输出与查看

场景结果区分三种几何：**实测点**来自掩膜内的深度，**模型点**来自固定资产与有效位姿，**手部点**来自实测关节与机器人正运动学。物体模板位于物体局部坐标系，场景点默认位于彩色相机坐标系，单位均为米。

实测点有效性与模型位姿有效性独立：位姿丢失时仍可能保留实测点，无资产对象不生成模型位姿或模型点。

离线输出由场景描述、固定模型模板和逐帧结果组成；查看器从这些结果展示分割及实测／模型点云，无需重新执行模型推理。

实验入口见[脚本说明](../../../scripts/cordvip/README.md)，资产生成思路见[物体重建](../../../docs/object_reconstruction.md)。
