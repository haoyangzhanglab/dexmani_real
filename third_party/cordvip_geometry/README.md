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

- 每段调用 `initialize()`，复用模型并重置位姿、视频状态及场景缓存。
- 同一段内相机参数和图像尺寸须保持固定。
- 初始化失败抛出 `TrackingFailure`，后续新帧重试初始化；全部对象注册成功后才进入正常跟踪。
- 跟踪期逐对象返回有效性。观测不足进入 OCCLUDED，充分几何矛盾、非法位姿或单对象推理运行时失败进入 LOST；输入、接口及程序错误直接报错。共享 SAM2 视频推理或校正失败可能留下部分状态，不作为可恢复遮挡吞掉，调用方停止使用该实例。

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
| `min_depth_pixels` | 30 | stride=2 网格上的最少匹配点数和支持点数 |
| `min_depth_fraction` | 0.5 | 最低深度匹配比例 |
| `min_depth_coverage` | 0.5 | 检查区域的最低有效深度覆盖率 |
| `min_precision`、`min_recall` | 0.5 | 可见轮廓的最低一致性 |

TRACKING 和 OCCLUDED 对象先合批执行 FoundationPose，再验证候选。注册需要 Mask 深度生成平移初值；正常 Track 不以 Mask 深度预检。

每个候选的 SE(3)、运动连续性和 CAD 渲染只计算一次。Mask 更新或校正后，复用 CAD 深度重算当前可见证据。有 Mask 时使用 `pose_quality()`：仅 Mask 外更近、超出容差的有效深度支持前景遮挡，缺失深度不算遮挡。无 Mask 时使用 `pose_depth_quality()`，双向深度残差均计为不匹配。

两个质量函数均返回 `(metrics, verdict)`：

| `verdict` | 判据 |
| --- | --- |
| VALID | 匹配点数、比例、覆盖率和可见轮廓均通过 |
| OCCLUDED | 证据不足，且没有充分支持的几何矛盾 |
| LOST | 有充分支持的深度或轮廓矛盾 |

有 Mask 时，分别检查 Mask 和去除前景遮挡后的 CAD 可见区域。相应区域达到最少支持点数及覆盖率，才能将不匹配判为 LOST；VALID 要求两个区域的支持均充分。纯深度路径先检查支持点数和覆盖率，再判断深度匹配比例。

FoundationPose 输出须为有限 `[4,4]` 位姿，齐次末行容差为 `1e-6`，SO(3) 容差为 `1e-3`，且居中网格位于相机前方；渲染前逐对象拒绝非法候选并回滚。此检查使用网格中心，允许 CAD 原点偏离物体。

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

## 事件触发分割

SAM2 在初始化、周期到期、候选深度质量不通过、恢复注册或恢复确认时共享刷新，同帧图像只追加一次。`segmentation_interval_frames=2` 表示活动对象每两个处理帧刷新，设为 1 时逐帧分割。全部对象 LOST 时，等待恢复到期再刷新。DINO 仅用于初始化和到期 LOST 对象的重新检测。

`RGBDFrame.from_camera(camera, row, source_sequence=...)` 的时间取自行内 `timestamp_ns`，来源序列通过参数传入。在线、离线分别使用 Camera Ring 和 Raw timing 中的元数据。三种计数各自负责：

- `frame_index`：实际处理次数，用于分割和恢复调度。
- `source_sequence`：Camera Ring 序列，用于相机帧间隔和 Mask 年龄；不是 RealSense 硬件帧号。
- SAM2 内部视频索引：仅随实际输入图像递增，校正不增加索引。

调用方跳过重复来源序列，跳帧不补造图像。SceneTracker 要求新序列递增、时间不倒退。直接调用省略元数据时，使用处理计数和单调时钟；离线调用应传入记录时间，以正确计算遮挡超时和运动变化。

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

到期对象共享 DINO 检测，按规范化 Prompt 唯一匹配，经 SAM2 分割、FoundationPose 注册和质量检查后，用原 ID 的 `correct_masks()` 校正。合并 Mask 后复查全部当前候选，健康对象独立保留。

恢复需要两帧确认。首次通过后保存 Pending 位姿，保持 LOST；下一处理帧以 Pending 为临时初值，与健康对象合批 Track，再次通过后才提交 TRACKING。Pending 失败则清理并保留原重试间隔。所有不可信候选均回滚到最后可靠内部位姿，历史位姿和 Pending 不作为当前有效观测发布。

连续性检查在居中网格坐标中计算平移及 SO(3) 测地距离，复用 FoundationPose 声明的对称变换。`max_translation_speed_m_s`、`max_rotation_speed_rad_s` 默认 `None`，只记录变化量；配置实测上限后才启用硬门控，余量由深度容差和物体直径换算。同一个对称等价位姿须同时满足两种限制。遮挡时参考最后可靠位姿，恢复确认时参考 Pending，均使用对应的真实时间间隔。

多物体合批 refinement 运行时失败后才逐对象隔离；只有一个活动对象时直接拒绝并回滚，避免同帧重复推理。共享 SAM2 刷新或校正失败时，先回滚全部内部位姿再抛出，调用方停止使用该实例。

在线发布、离线存储和训练过滤见[主项目说明](../../docs/object_reconstruction.md#标注导出与回放)。

## 推理实现

`SceneTracker` 在构造时加载 Grounding DINO 并预热固定尺寸网络。每段首帧先对 RGB-D 去畸变，再检测目标并用框初始化 SAM2，所有物体共享一次图像编码。每个描述须匹配唯一候选；缺失或多个匹配均终止初始化。每次 SAM2 推理按最高 mask 分数分配重叠像素。

SAM2 流式前处理使用 1024 输入和 LANCZOS 缩放，上传 uint8 图像后查表获得 float32 输入，保持 float64 归一化后转 float32 的数值结果。[SAM2 补丁](scripts/sam2_inference.patch) 保留多尺度图像特征，仅计算 memory attention 所需的最终层位置编码。

FoundationPose 后续帧复用图像缓冲、相机投影矩阵及采样网格常量，共享深度预处理。位姿保留在 GPU，从上一帧位姿开始更新；各物体独立渲染，观测裁剪与归一化合批执行。两轮 refinement 后，逐物体渲染完整投影，按当前 mask 是否可用执行完整质量检查或深度门控。

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
