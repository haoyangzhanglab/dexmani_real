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
poses = tracker.initialize(frame, {"apple": "apple"})
poses = tracker.track(next_frame)
```

位姿表示物体局部坐标到相机坐标的变换，平移单位为米。每段调用 `initialize()` 重置场景并复用模型；跟踪失败抛出 `TrackingFailure`，调用方结束当前段，下一段重新初始化。

首帧先对 RGB-D 去畸变，再检测并用框初始化 SAM2 视频状态，所有物体共享一次图像编码。每个描述须匹配唯一候选；缺失或多个匹配均终止初始化。首帧与后续帧统一由 SAM2 按最高 mask 分数分配重叠像素。

`SAM2Segmenter.detect()` 返回检测框、分数和标签；`predict()` 在候选中加入单图分割 mask 并过滤空 mask。视频接口 `initialize(rgb, boxes)` 接收 `{对象名: XYXY 检测框}`，与 `track(rgb)` 一样返回 `{对象名: mask}`。

FoundationPose 后续帧共享深度预处理，各物体独立裁剪、归一化，合批执行两轮 refinement，再分别检查分割与模型的深度一致性。RGB 与几何输入共用采样网格；refine 直接裁剪 XYZ，score 从裁剪深度重建 XYZ。

跟踪网络使用 PyTorch 原生 CUDA 卷积；注册和跟踪的 cuDNN 设置仅在调用内生效，调用后恢复。批量大小会影响浮点结果，近对称物体的旋转可能不稳定。

`reconstruction.load_shape_model()` 加载可复用的 Hunyuan3D 模型，使用分层体积解码、完整 attention 和 Marching Cubes。`reconstruct_mesh(model, rgba, ...)` 清理退化面并返回简化网格；`scale_mesh()` 返回保留原点和朝向的缩放副本，由调用方保存。

| 模块 | 职责 |
| --- | --- |
| `camera.py`、`mesh.py` | 相机去畸变、网格处理与射线深度 |
| `segmentation.py`、`reconstruction.py`、`foundationpose.py` | 模型加载与推理 |
| `tracking.py` | 多物体跟踪与深度一致性检查 |
| `cli.py`、`visualization.py`、`weights.py` | 命令行、结果预览与模型下载 |

任务流程见 [实验入口](../../scripts/cordvip/README.md)，主项目表示见 [物体重建与 HOI](../../docs/object_reconstruction.md)。
