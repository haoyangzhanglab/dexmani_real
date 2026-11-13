# cordvip_geometry

物体分割、单图重建、尺度调整与 RGB-D 位姿跟踪后端。模型在同一 Python 进程直接调用；任务模板与 Raw 写入由主项目负责。

## 安装与命令

在兼容 Torch/CUDA 的 `real_robot` 环境中，从仓库根目录执行：

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
```

安装脚本拉取固定版本的源码、应用补丁并编译 CUDA 扩展。FoundationPose 权重需另行放入 `third_party/FoundationPose/weights/`：

| 模型 | 权重目录 |
| --- | --- |
| refine | `2023-10-28-18-33-37` |
| score | `2024-01-11-20-02-45` |

使用 Hunyuan3D 重建时，再执行：

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh reconstruction
python -m cordvip_geometry download --models shape
```

CLI 提供 `segment`、`select`、`generate`、`reconstruct`、`scale`、`preview`、`download`，参数见各命令的 `--help`。例如：

```bash
python -m cordvip_geometry reconstruct --image reference.jpg --prompt apple --headless --output /tmp/apple
python -m cordvip_geometry scale --mesh /tmp/apple/mesh_unscaled.obj --size-axis x --size-meters 0.08 --output /tmp/apple/metric
```

重建先输出 `mesh_unscaled.obj`，提供尺度后另存米制的 `mesh_m.obj`。尺度使用 `--scale-to-meters`，或配对的 `--size-axis` / `--size-meters`。

`segment` 保存 mask、RGBA 和叠加预览。使用 `--list-only` 时保存原图与候选，随后可通过 `select --segmentation-dir ...` 选择物体。

检测框裁剪到图像边界并移除退化框后，按标签执行 CPU NMS（IoU 0.8）。单图分割再过滤空 mask；`--instance` 对应剩余候选按置信度降序排列的零起始索引。

## Python 接口

```python
from cordvip_geometry import RGBDFrame, SceneTracker

tracker = SceneTracker(
    {"apple": "assets/pointcloud_template/object/pnp_apple_banana/apple/mesh/mesh_m.obj"},
    workspace=".",
)
frame = RGBDFrame.from_camera(camera, camera_row)
result = tracker.initialize(frame, {"apple": "apple"})
poses, quality = result.poses, result.quality
result = tracker.track(next_frame)
```

`initialize()` 和 `track()` 均返回 `SceneResult(poses, quality)`，两份字典均以对象名为键。位姿表示物体局部坐标到相机坐标的变换，平移单位为米。

- 每段调用 `initialize()`，复用模型并重置位姿、视频状态及场景缓存。
- 同一段内相机参数和图像尺寸须保持固定。
- 跟踪失败抛出 `TrackingFailure`，调用方结束当前段，下一段重新初始化。

`SAM2Segmenter` 提供单图和视频接口：

| 方法 | 输入与输出 |
| --- | --- |
| `detect(rgb, text)` | 返回检测框、分数和标签。 |
| `predict(rgb, text)` | 在检测候选中加入单图分割 mask，并过滤空 mask。 |
| `initialize(rgb, boxes)` | 接收 `{对象名: XYXY 检测框}`，初始化视频并返回 `{对象名: mask}`。 |
| `track(rgb)` | 返回当前帧的 `{对象名: mask}`。 |

`reconstruction.load_shape_model()` 加载可复用的 Hunyuan3D 模型；`reconstruct_mesh(model, rgba, ...)` 清理退化面并返回简化网格；`scale_mesh()` 返回保留原点和朝向的缩放副本，由调用方保存。

## 质量检查

`TrackingConfig` 集中定义质量阈值，通过 `SceneTracker(..., config=...)` 传入：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `depth_tolerance_m` | 0.015 m | 实测与模型深度的匹配容差 |
| `min_depth_pixels` | 30 | stride=2 网格上的最少深度支持像素 |
| `min_depth_fraction` | 0.5 | mask 内有效深度的最低匹配比例 |
| `min_precision`、`min_recall` | 0.5 | 可见轮廓的最低 precision、recall |

每帧渲染完整模型投影。只有 mask 外的有效实测深度比模型近、且差值超过深度容差时，才将对应投影视为前景遮挡；缺失深度不作为遮挡证据。`quality` 同时返回轮廓 IoU、深度误差中位数及遮挡比例等诊断值。

阈值是工程初值，需用实际 RGB-D 回放验证。分割深度不足或质量检查失败均抛出 `TrackingFailure`；相机参数变化属于输入错误，直接报错。

## 推理实现

`SceneTracker` 在构造时加载 Grounding DINO 并预热固定尺寸网络。每段首帧先对 RGB-D 去畸变，再检测目标并用框初始化 SAM2，所有物体共享一次图像编码。每个描述须匹配唯一候选；缺失或多个匹配均终止初始化。首帧与后续帧均由 SAM2 按最高 mask 分数分配重叠像素。

SAM2 流式前处理使用 1024 输入和 LANCZOS 缩放，上传 uint8 图像后查表获得 float32 输入，保持 float64 归一化后转 float32 的数值结果。[SAM2 补丁](scripts/sam2_inference.patch) 保留多尺度图像特征，仅计算 memory attention 所需的最终层位置编码。

FoundationPose 后续帧复用图像缓冲、相机投影矩阵及采样网格常量，共享深度预处理。位姿保留在 GPU，从上一帧位姿开始更新；各物体独立渲染，观测裁剪与归一化合批执行。两轮 refinement 后，逐物体渲染完整投影，检查遮挡感知的双向轮廓与实测深度一致性。

RGB 与几何输入共用采样网格：refine 直接裁剪 XYZ，score 从裁剪深度重建 XYZ。跟踪使用 PyTorch 原生 CUDA 卷积；注册和跟踪的 cuDNN 设置仅在调用内生效。批量大小会影响浮点结果，近对称物体的旋转可能不稳定。

FoundationPose [推理补丁](scripts/foundationpose_inference.patch) 按上述两份权重的固定格式加载模型，使用局部固定种子采样网格直径。refine 和 score 直接返回位姿与分数；可视化由调用入口负责。

Hunyuan3D 重建使用分层体积解码、完整 attention 和 Marching Cubes。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `camera.py`、`mesh.py` | 相机去畸变、网格处理与射线深度 |
| `segmentation.py`、`reconstruction.py`、`foundationpose.py` | 模型加载与推理 |
| `tracking.py` | 多物体跟踪与轮廓/深度质量检查 |
| `cli.py`、`visualization.py`、`weights.py` | 命令行、结果预览与模型下载 |

表面模板采样由主项目 `geometry/hoi.py` 负责，后端不依赖主项目或 fpsample。

任务流程见 [实验入口](../../scripts/cordvip/README.md)，主项目表示见 [物体重建与 HOI](../../docs/object_reconstruction.md)。
