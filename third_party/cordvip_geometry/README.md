# cordvip_geometry

物体分割、单图重建、尺度调整与 RGB-D 位姿跟踪后端。模型在同一 Python 进程直接调用；任务模板与 Raw 写入由主项目负责。

## 目录与维护

`third_party/cordvip_geometry/` 保存随主仓库维护的后端源码、安装脚本和推理补丁。上游源码保存在 `third_party/` 下的本地 Git 工作树中，由主仓库忽略。

| 目录 | 用途 |
| --- | --- |
| `Grounded-SAM-2/` | SAM2 单图分割和视频掩码跟踪。 |
| `FoundationPose/` | RGB-D 位姿注册和跟踪。 |
| `OSTrack/` | 可选的 OSTrack-256 二维框跟踪。 |
| `nvdiffrast/` | FoundationPose 使用的 CUDA 光栅化。 |
| `Hunyuan3D-2.1/` | 可选的单图网格重建。 |

复用已有源码时，指定目录必须是 Git 工作树根目录，且 `HEAD` 必须匹配安装脚本中的固定版本。版本匹配时保留本地修改，跳过已应用的补丁。目录或版本不符、补丁冲突时，安装会报错退出。固定版本见 [安装脚本](scripts/install_perception.sh)。

`third_party/FoundationPose/mycpp/build/` 中的 `mycpp` 扩展供位姿跟踪在运行时加载，清理构建产物时须保留。

## 安装与命令

在兼容 Torch/CUDA 的 `real_robot` 环境中，从仓库根目录安装跟踪依赖，再下载 Grounding DINO 和 SAM2 权重：

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
```

安装脚本准备跟踪源码、应用补丁并编译扩展，同时通过 `[cloud]` 安装读取 CAD 用的 trimesh，以及简化网格和射线渲染用的 Open3D。仅生成模板时，在主项目环境中运行 `python -m pip install -e 'third_party/cordvip_geometry[cloud]'`。

`download` 命令默认将权重写入 `third_party/weights/`。FoundationPose 的两个模型权重另行放入 `third_party/FoundationPose/weights/` 的对应子目录。

| 模型 | 权重目录 |
| --- | --- |
| refine | `2023-10-28-18-33-37` |
| score | `2024-01-11-20-02-45` |

使用 OSTrack 时，在同一环境中安装固定版本的官方源码和推理依赖。将官方 `vitb_256_mae_ce_32x4_ep300` 权重保存为 `third_party/weights/ostrack/OSTrack_ep0300.pth.tar`。[OSTrack-256 权重](https://drive.google.com/file/d/1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV/view)

```bash
bash third_party/cordvip_geometry/scripts/install_perception.sh ostrack
```

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
- 跟踪期逐对象返回有效性。检测、分割、注册或跟踪的后端异常（包括 CUDA OOM）沿调用链传播。两种模式的观测失效与恢复见[遮挡恢复](#遮挡恢复)和[OSTrack 跟踪与恢复](#ostrack-跟踪与恢复)。

`SAM2Segmenter` 提供单图和视频接口：

| 方法 | 输入与输出 |
| --- | --- |
| `detect(rgb, text)` | 返回检测框、分数和标签。 |
| `predict(rgb, text)` | 在检测候选中加入单图分割 mask，并过滤空 mask。 |
| `initialize(rgb, boxes)` | 接收 `{对象名: XYXY 检测框}`，初始化视频并返回 `{对象名: mask}`。 |
| `track(rgb)` | 返回当前帧的 `{对象名: mask}`。 |
| `segment_boxes(rgb, boxes)` | 对非空检测框字典做单图分割，不修改视频状态。 |
| `correct_masks(masks)` | 校正当前已跟踪帧的已有对象 ID，返回合并后的全部 mask。 |

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

Mask 更新或校正后复用 CAD 深度。`pose_quality()` 用 Mask 外更近且超出容差的有效深度识别前景遮挡。无当前 Mask 时使用 `pose_depth_quality()`，按有效深度与 CAD 深度的绝对差是否超过容差判断匹配。

两个质量函数均返回 `(metrics, verdict)`：

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
| `mask_source`、`mask_age_frames` | 是否使用当前 Mask，以及距最近 SAM2 输入的来源序列差 |
| `dt_s`、`source_frame_gap` | 相邻处理观测的真实时间与来源序列间隔 |
| `translation_delta_m`、`rotation_delta_rad`、`pose_dt_s` | 相对参考位姿的平移、旋转和时间间隔 |

有效对象的质量指标均为有限数值。纯深度路径未计算的 Mask 指标为 −1；注册没有运动参考，运动指标为 0。

### Object Symmetry

当前按非对称物体跟踪，FoundationPose 使用默认单位对称变换。对称物体可能出现几何等价姿态之间的旋转歧义。

`pose_motion()` 直接比较居中网格位姿的平移距离和 SO(3) 测地角。启用运动门控时，候选须同时满足已配置的平移和旋转限制。

## 事件触发分割

本节和下节描述默认的 `sam2_video` 模式。

首帧用 Grounding DINO 检测框初始化 SAM2 视频状态。后续帧在 LOST 重检到期、恢复确认或周期到期时刷新。若当前帧尚无 Mask，通过位姿和运动检查的候选在纯深度验证失败时也会触发刷新，并用当前 Mask 复查。各对象共享同帧结果，每个来源帧最多追加一次图像。

深度异常按帧处理。持续遮挡时，下一帧若仍需用当前轮廓区分遮挡和位姿偏差，可以再次刷新。

默认 `segmentation_interval_frames=10`，活动对象距上次 SAM2 输入满 10 个处理帧时进行周期校正。事件刷新后重新计数，设为 1 时每个活动处理帧都刷新。全部对象 LOST 且无待确认恢复时，分割随到期重检触发。

`RGBDFrame` 必须携带真实的 `source_sequence` 和 `timestamp_ns`。`from_camera(camera, row, source_sequence=...)` 从相机行读取时间；在线来源为 Camera Ring，离线来源为 Raw timing。三种计数分别是：

- `frame_index`：实际处理次数，用于分割和恢复调度。
- `source_sequence`：Camera Ring 序列，用于相机帧间隔和 Mask 年龄；不是 RealSense 硬件帧号。
- SAM2 内部视频索引：仅随实际输入图像递增，校正不增加索引。

调用方跳过重复来源序列，跳帧不补造图像。SceneTracker 要求来源序列和时间均为正数、新序列递增、时间不倒退；遮挡超时和运动变化始终使用来源时间计算。

## 遮挡恢复

| 状态 | 行为 |
| --- | --- |
| TRACKING | 当前候选通过验证后输出有效位姿 |
| OCCLUDED | 保留最后可靠内部位姿，继续 Track；当前 `valid=False` |
| LOST | 停止正常 refinement，按间隔检测并注册；当前 `valid=False` |

`occlusion_timeout_s=0.5` 从首次证据不足的观测计时。等待期间证据恢复可直接返回 TRACKING；明确矛盾或遮挡超时进入 LOST，从下一处理帧开始恢复。

| 恢复参数 | 默认值 |
| --- | --- |
| `recovery_enabled` | `True` |
| `recovery_retry_interval_frames` | `15` 个处理帧 |
| `recovery_box_threshold`、`recovery_text_threshold` | `0.3`、`0.25` |

到期对象共享 DINO 检测，按规范化 Prompt 唯一匹配，经 SAM2 单图分割、FoundationPose 注册和质量检查后，用 `correct_masks()` 校正当前视频帧的原对象 ID。合并 Mask 后复查全部当前候选，各对象按复查结果独立提交或回滚。

恢复需要两帧确认。首次通过后保存 Pending 位姿，保持 LOST；下一处理帧以 Pending 为临时初值，与健康对象合批 Track，再次通过后才提交 TRACKING。Pending 失败则清理并保留原重试间隔。所有不可信候选均回滚到最后可靠内部位姿，历史位姿和 Pending 不作为当前有效观测发布。

连续性指标与对称变换处理见 [Object Symmetry](#object-symmetry)。`max_translation_speed_m_s`、`max_rotation_speed_rad_s` 默认 `None`，只记录变化量；配置实测上限后才启用硬门控，余量由深度容差和物体直径换算。遮挡时参考最后可靠位姿，恢复确认时参考 Pending，均使用对应的真实时间间隔。

在线发布、离线存储和训练过滤见[主项目说明](../../docs/object_reconstruction.md#标注与查看)。

## OSTrack 跟踪与恢复

设置 `TrackingConfig(tracker_2d="ostrack")` 后，首帧由 Grounding DINO 检测，SAM2 单图分割，FoundationPose 注册。用通过几何检查的 Mask 生成各物体的 OSTrack 模板和初始框。全场景共享一份 ViT-B，各物体依次推理。场景以半开区间的 `XYXY` 表示框，适配器在调用官方 OSTrack 时做 `XYWH` 转换。

`TRACKING` 中每帧更新二维框和 6D 候选。每 10 个处理帧用当前框批量做 SAM2 单图分割，其余正常帧用 CAD 深度验证。失败帧保留最后可靠内部位姿作为下次 refinement 的初值，并发布当前无效。连续无效 0.3 秒进入 `REACQUIRE`。

`REACQUIRE` 先用最后可靠框分割和注册，再对仍失败的对象用 DINO 重检。恢复通过同样的几何检查后，从当前 Mask 重置模板并当帧发布有效位姿。失败后间隔 8 个处理帧重试。

FoundationPose 的 `pose_last` 使用居中网格。注册时用 Mask 框宽、高归一化框中心与居中原点投影的偏移。后续从当前框中心扣除按当前宽、高还原的偏移，得到居中原点的目标像素。合批 refinement 前将像素残差按半个框宽、高限幅，以默认 gain 0.2 修正候选的相机系 XY 平移。先验保持候选 Z 和旋转，对外位姿仍对应原始网格。初始投影不在 Mask 框内的对象跳过先验，`xy_prior_gain=0` 可关闭先验。

## 推理实现

`SceneTracker` 在构造时加载 Grounding DINO，并预热 SAM2 图像编码器和 FoundationPose refiner。每段首帧先对 RGB-D 去畸变，再检测目标并用框调用 SAM2，所有物体共享一次图像编码。每个描述须匹配唯一候选，缺失或多个匹配均终止初始化。SAM2 视频分支按最高 mask 分数分配重叠像素。

SAM2 流式前处理使用 1024 输入和 LANCZOS 缩放，上传 uint8 图像后查表获得 float32 输入，保持 float64 归一化后转 float32 的数值结果。[SAM2 补丁](scripts/sam2_inference.patch) 保留多尺度图像特征，仅计算 memory attention 所需的最终层位置编码。

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
