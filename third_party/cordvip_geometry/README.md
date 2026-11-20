# cordvip_geometry

物体分割、单图重建、尺度调整与 RGB-D 位姿跟踪后端。模型在同一 Python 进程直接调用；任务模板与 Raw 写入由主项目负责。

## 目录与维护

`third_party/cordvip_geometry/` 保存随主仓库维护的后端源码、安装脚本和推理补丁。上游源码保存在 `third_party/` 下的本地 Git 工作树中，由主仓库忽略。

| 目录 | 用途 |
| --- | --- |
| `Grounded-SAM-2/` | SAM2 单图分割源码。 |
| `FoundationPose/` | RGB-D 位姿注册和跟踪。 |
| `OSTrack/` | OSTrack-256 二维框跟踪。 |
| `nvdiffrast/` | FoundationPose 使用的 CUDA 光栅化。 |
| `Hunyuan3D-2.1/` | 可选的单图网格重建。 |

复用已有源码时，指定目录必须是 Git 工作树根目录，且 `HEAD` 必须匹配安装脚本中的固定版本。安装会跳过已完整应用的补丁，保留其他本地修改。目录、版本或补丁冲突时，安装会报错退出。需要重新获取固定版本时，可先移走冲突的工作树以保留本地修改，再运行安装。[安装脚本](scripts/install_perception.sh)

`third_party/FoundationPose/mycpp/build/` 中的 `mycpp` 扩展供位姿跟踪在运行时加载，清理构建产物时须保留。

## 安装与命令

在兼容 Torch/CUDA 的 `real_robot` 环境中，从仓库根目录安装跟踪依赖，再下载 Grounding DINO 和 SAM2 权重。

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
```

默认安装包含 OSTrack、Grounded-SAM-2、FoundationPose 和 nvdiffrast 的固定版本源码、推理补丁和扩展，并安装后端的 `cloud` 和 `vision` 依赖。可运行 `bash third_party/cordvip_geometry/scripts/install_perception.sh ostrack` 单独安装 OSTrack。仅生成模板时，在主项目环境中运行 `python -m pip install -e 'third_party/cordvip_geometry[cloud]'`，安装网格读取、简化和射线渲染依赖。

`download` 命令默认将检测和分割权重写入 `third_party/weights/`。跟踪还需准备下表中的文件。FoundationPose 每个模型目录都放入对应的 `config.yml` 和 `model_best.pth`。

| 模型 | 路径 |
| --- | --- |
| FoundationPose refine | `third_party/FoundationPose/weights/2023-10-28-18-33-37/` |
| FoundationPose score | `third_party/FoundationPose/weights/2024-01-11-20-02-45/` |
| OSTrack-256 | `third_party/weights/ostrack/OSTrack_ep0300.pth.tar`。使用 `vitb_256_mae_ce_32x4_ep300` 配置。[官方权重](https://drive.google.com/file/d/1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV/view) |

需要单图重建时，在完成跟踪安装的同一环境中安装 Hunyuan3D 依赖，再下载对应权重：

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
frame = RGBDFrame.from_camera(camera, camera_row, source_sequence=sequence)
result = tracker.initialize(frame, {"apple": "apple"})
result = tracker.track(next_frame)
if result.valid["apple"]:
    pose_camera_from_object = result.poses["apple"]
```

`initialize()` 和 `track()` 返回 `SceneResult`。位姿表示物体局部到相机的变换 `T_camera_from_object`，平移单位为米。

| 字段 | 含义 |
| --- | --- |
| `poses` | 仅包含当前有效对象的位姿，键为原始对象名。 |
| `quality` | 每个对象的当前质量指标；未计算时为空字典。 |
| `valid` | 覆盖全部对象，表示当前帧位姿是否有效。 |
| `failures` | 无效对象的失败原因或等待重试状态。 |

- 每段调用 `initialize()`，复用模型并重置位姿、二维跟踪状态及场景缓存。
- 同一段内相机参数和图像尺寸须保持固定。RGB 和 depth 必须匹配同一彩色相机网格。
- `RGBDRectifier` 按 `distortion_model` 与 `distortion_coeffs` 初始化，可接受带 `distortion.` 前缀的模型名。`none` 直接返回输入的 RGB 和 depth，`brown_conrady` 在系数非零时建立映射。映射对 RGB 使用双线性采样，对深度使用最近邻采样，越界填零。
- 初始化时检测匹配、深度或候选位姿不满足要求，抛出 `TrackingFailure`，后续新帧重试初始化；全部对象注册成功后才进入正常跟踪。
- 跟踪期逐对象返回有效性。检测、分割、注册或跟踪的后端异常（包括 CUDA OOM）沿调用链传播。观测失效与恢复见[跟踪与恢复](#跟踪与恢复)。

`SAM2Segmenter` 提供检测和单图分割接口。

| 方法 | 输入与输出 |
| --- | --- |
| `detect(rgb, text)` | 返回检测框、分数和标签。 |
| `predict(rgb, text)` | 在检测候选中加入单图分割 mask，并过滤空 mask。 |
| `segment_boxes(rgb, boxes)` | 一次编码当前 RGB 图像，对非空 XYXY 框字典解码并返回 `{对象名: mask}`。同一帧补充分割可用 `reuse_image=True` 复用编码。 |

`reconstruction.load_shape_model()` 加载可复用的 Hunyuan3D 模型；`reconstruct_mesh(model, rgba, ...)` 清理退化面并返回简化网格；`scale_mesh()` 返回保留原点和朝向的缩放副本，由调用方保存。

## 质量检查

`SceneTracker(..., config=TrackingConfig(...))` 配置质量阈值：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `depth_tolerance_m` | 0.015 m | 实测与 CAD 深度的匹配容差 |
| `min_depth_pixels` | 8 | stride=4 网格上的最少匹配点数和支持点数 |
| `min_depth_fraction` | 0.5 | 最低深度匹配比例 |
| `min_depth_coverage` | 0.5 | 检查区域的最低有效深度覆盖率 |
| `min_precision`、`min_recall` | 0.5 | 可见轮廓的最低一致性 |

仍在尝试正常跟踪的对象合批执行 FoundationPose，再验证候选。注册用 Mask 深度生成平移初值。正常跟踪可用实测深度直接验证候选。

每个候选的 SE(3)、运动连续性和 CAD 渲染只计算一次。相机射线、实测深度、SAM2 Mask 和渲染深度共用 stride=4 的像素网格，640×480 图像对应 160×120 个采样位置。`MeshRaycaster` 将 CAD 局部包围盒的 8 个角点投影到图像，扩展 8 像素余量后裁剪到网格，只渲染区域内的射线。深度回填到完整网格，区域外为零。包围盒跨越 1 mm 近平面时使用全图，完全位于近平面后方时返回全零。

新 Mask 参与验证时复用本帧 CAD 深度。`pose_quality()` 用 Mask 外更近且超出容差的有效深度识别前景遮挡。无当前 Mask 时使用 `pose_depth_quality()`，按有效深度与 CAD 深度的绝对差是否超过容差判断匹配。

两个质量函数均返回 `(metrics, verdict)`。其中 `OCCLUDED` 和 `LOST` 表示本帧证据的分类。

| `verdict` | 判据 |
| --- | --- |
| VALID | 深度支持和匹配满足阈值，有当前 Mask 时可见轮廓也须通过 |
| OCCLUDED | 证据不足，且没有充分支持的几何矛盾 |
| LOST | 有充分支持的深度或轮廓矛盾 |

有 Mask 时，分别检查 Mask 和去除前景遮挡后的 CAD 可见区域。相应区域达到最少支持点数及覆盖率，才能将不匹配判为 LOST；VALID 要求两个区域的支持均充分。纯深度路径先检查支持点数和覆盖率，再判断深度匹配比例。

FoundationPose 输出须为有限 `[4,4]` 位姿，齐次末行容差为 `1e-6`，SO(3) 容差为 `1e-3`，且居中网格位于相机前方。非法候选在渲染前拒绝，帧末统一恢复未通过对象的内部位姿。此检查使用网格中心，允许 CAD 原点偏离物体。

主要指标：

| 字段 | 含义 |
| --- | --- |
| `depth_pixels`、`depth_fraction` | 匹配点数及比例；Mask 路径以 Mask 内有效深度数为分母 |
| `depth_support_pixels`、`depth_coverage` | 检查区域的支持点数，以及 CAD 总投影中的有效深度比例 |
| `mask_depth_coverage`、`visible_depth_coverage` | Mask 和去遮挡后 CAD 可见区域的有效深度覆盖率 |
| `precision`、`recall`、`iou` | 可见轮廓一致性 |
| `supported_precision`、`supported_recall` | 有效深度支持下的可见轮廓一致性 |
| `occluded_fraction` | 有深度证据支持的前景遮挡比例 |
| `mask_source`、`mask_age_frames` | 是否使用当前 Mask，以及距该物体最近一次分割的相机来源序列差 |
| `dt_s`、`source_frame_gap` | 相邻处理观测的真实时间与来源序列间隔 |
| `translation_delta_m`、`rotation_delta_rad`、`pose_dt_s` | 相对参考位姿的平移、旋转和时间间隔 |

有效对象的质量指标均为有限数值。纯深度路径未计算的 Mask 指标为 −1；注册没有运动参考，运动指标为 0。

### 物体对称性

当前按非对称物体跟踪，FoundationPose 使用默认单位对称变换。对称物体可能出现几何等价姿态之间的旋转歧义。

`pose_motion()` 直接比较居中网格位姿的平移距离和 SO(3) 测地角。启用运动门控时，候选须同时满足已配置的平移和旋转限制。

## 跟踪与恢复

首帧由 Grounding DINO 检测、SAM2 单图分割、FoundationPose 注册，通过几何检查的 Mask 用于建立 OSTrack 模板。全场景共享一份 ViT-B，各物体依次推理。项目内部用半开 `XYXY` 框，适配器在调用官方 OSTrack 时转换 `XYWH`。

- `TRACKING` 中每帧更新二维框，合批产生 6D 候选。每 `segmentation_interval_frames` 个处理帧（默认 10）批量分割。候选通过刚体和运动检查但深度证据不足时，可提前分割，最短间隔为周期的一半向下取整且至少为 2 帧。复用当前 CAD 渲染来验证新 Mask。失败时回滚到最后可靠内部位姿并发布当前无效。
- `REACQUIRE` 从连续无效满 `lost_timeout_s`（默认 0.3 秒）后的下一处理帧开始。用最后可靠框分割和注册，对仍失败的对象再做 DINO 检测。通过几何检查后用当前 Mask 重置模板并当帧发布有效位姿。失败时每 `recovery_retry_interval_frames`（默认 8）个处理帧重试。`recovery_enabled=False` 时保持无效并报告恢复已禁用。

分割和重试按实际处理帧计数，失效期限和运动连续性按相机来源时间计算。`RGBDFrame` 的 `source_sequence` 和 `timestamp_ns` 来自 Camera Ring 或 Raw timing。调用方跳过重复序列，SceneTracker 要求新序列递增、时间不倒退。在线发布、离线存储和训练过滤见[主项目说明](../../docs/object_reconstruction.md#标注与查看)。

FoundationPose 的 `pose_last` 使用居中网格。注册和后续经几何验证的无遮挡 Mask 将居中原点投影锚定到紧致 Mask 框，再换算到同帧 OSTrack 框的归一化坐标。后续像素残差按半个框宽、高限幅，以默认 gain 0.2 修正 refinement 前候选的相机系 XY 平移。先验保持 Z 和旋转，对外位姿仍对应原始网格。`xy_prior_gain=0` 可关闭先验。

## 推理实现

`SceneTracker` 在构造时加载 Grounding DINO，并预热 SAM2 图像编码器和 FoundationPose refiner。每个物体描述须匹配唯一候选。SAM2 使用 `build_sam2` 构建图像模型，同帧多个物体和 DINO 兜底共享一次图像编码。[SAM2 补丁](scripts/sam2_inference.patch) 保留多尺度图像特征，省去额外尺度的位置编码。

FoundationPose 后续帧复用图像缓冲、相机投影矩阵及采样网格常量，共享深度预处理。内部位姿保留在 GPU，各物体独立渲染，观测裁剪与归一化合批执行两轮 refinement。

RGB 与几何输入共用采样网格：refine 直接裁剪 XYZ，score 从裁剪深度重建 XYZ。跟踪使用 PyTorch 原生 CUDA 卷积；注册和跟踪的 cuDNN 设置仅在调用内生效。批量大小会影响浮点结果，近对称物体的旋转可能不稳定。

FoundationPose [推理补丁](scripts/foundationpose_inference.patch) 按上述两份权重的固定格式加载模型，使用局部固定种子采样网格直径。refine 和 score 直接返回位姿与分数；可视化由调用入口负责。

Hunyuan3D 重建使用分层体积解码、完整 attention 和 Marching Cubes。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `camera.py`、`mesh.py` | 相机去畸变、网格处理与射线深度 |
| `segmentation.py`、`reconstruction.py`、`foundationpose.py` | 模型加载与推理 |
| `ostrack.py` | 共享 OSTrack 网络与逐对象模板 |
| `tracking.py` | 多物体跟踪与轮廓/深度质量检查 |
| `cli.py`、`visualization.py`、`weights.py` | 命令行、结果预览与模型下载 |

FoundationPose 与 MeshRaycaster 使用 `simplify_mesh()` 处理后的 CAD 网格。主项目生成表面模板时复用同一网格简化函数，再由 `geometry/hoi.py` 采样。后端不依赖主项目或 fpsample。

任务命令与资产结构见[主项目实验流程](../../docs/object_reconstruction.md#实验流程)。
