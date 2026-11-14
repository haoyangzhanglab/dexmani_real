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
- 跟踪期逐对象返回有效性。单对象推理的运行时失败和质量不通过进入 LOST；输入、接口及程序错误直接报错。共享 SAM2 视频推理或校正失败可能留下部分状态，不作为可恢复遮挡吞掉，调用方停止使用该实例。

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

`TrackingConfig` 集中定义质量阈值，通过 `SceneTracker(..., config=...)` 传入：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `segmentation_interval_frames` | 2 | SAM2 最大处理帧间隔；严格正整数，1 为逐帧基线 |
| `depth_tolerance_m` | 0.015 m | 实测与模型深度的匹配容差 |
| `min_depth_pixels` | 30 | stride=2 网格上的最少匹配深度像素 |
| `min_depth_fraction` | 0.5 | 对应检查区域内的最低深度匹配比例 |
| `min_depth_coverage` | 0.5 | 纯深度路径的最低有效深度覆盖率，有限实数且在 (0, 1] 内 |
| `min_precision`、`min_recall` | 0.5 | 可见轮廓的最低 precision、recall |

每个候选位姿渲染一次完整模型投影。有当前 mask 时，使用遮挡感知的轮廓与深度检查：只有 mask 外的有效实测深度比模型近且超过容差，才计为前景遮挡；缺失深度不作为遮挡证据。无当前 mask 时，仅使用深度几何门控。

纯深度检查以有效 CAD 投影与有效传感器深度的交集为 support，统计绝对残差不超过 `depth_tolerance_m` 的匹配点数：

| 指标 | 计算方式 | 接受条件 |
| --- | --- | --- |
| `depth_pixels` | `matched_count` | ≥ `min_depth_pixels` |
| `depth_coverage` | `support_count / max(1, projected_count)` | ≥ `min_depth_coverage` |
| `depth_fraction` | `matched_count / max(1, support_count)` | ≥ `min_depth_fraction` |

`min_depth_coverage` 拒绝布尔值、字符串及范围外数值（包括 NaN/Inf），统一抛出 `ValueError`。

三项条件必须同时满足。覆盖率衡量完整 CAD 投影获得了多少真实深度，匹配比例衡量这些有效深度与位姿的一致性，二者不能互相替代。没有有效投影或深度支持时，三项指标均为零并拒绝。无当前 mask 时，更近和更远的深度偏差都计为不匹配，不推断遮挡。

有 mask 的 `pose_quality()` 输出同定义的 `depth_coverage`，仅用于诊断；其 `depth_fraction` 仍以 mask 内有效深度为分母，原有接受条件与遮挡算法不变。

阈值需用实际 RGB-D 回放验证。当前 mask 深度不足或完整质量检查失败仅将对应对象置为 LOST；相机参数变化属于输入错误，直接报错。

## 事件触发分割

健康对象每个处理帧执行 FoundationPose 和 CAD/RGB-D 验证。SAM2 首帧初始化，随后在以下情况刷新：

- 距上次实际推理达到 `segmentation_interval_frames`。
- 启用恢复且 LOST 对象到达重试时间。
- 任一中间帧候选未通过深度门控，包括有效深度全部匹配但覆盖率不足。

调度使用相机处理帧号 `frame_index`，每次刷新更新 `last_sam_frame`；重复相机序列不推进计数。SAM2 内部帧号仅在追加视频帧时递增，校正和历史裁剪使用内部帧号。`memory_temporal_stride_for_eval` 仅控制 SAM2 记忆。

任一候选深度门控失败，本帧立即执行一次共享 SAM2，并用当前 mask 对全部候选执行原有深度与完整质量检查。位姿无需重复 refinement，同一候选 CAD 深度复用于这些检查以及恢复 mask 合并后的检查；最终失败对象才进入 LOST。FoundationPose 非法位姿和推理异常保留原有隔离及回滚。

共享 SAM2 刷新或校正抛出 RuntimeError 时，先回滚本帧位姿再向外报错；调用方停止使用该实例。

| 质量字段 | 含义 |
| --- | --- |
| `mask_source` | 1：当前 SAM2 mask 完整检查；0：纯深度检查 |
| `mask_age_frames` | 距最近 SAM2 推理的处理帧数；刷新帧为 0 |

纯深度路径的 `mask_pixels`、`visible_pixels`、`precision`、`recall`、`iou`、`occluded_fraction` 统一为未计算占位值 −1；深度指标为实际测量。有效对象的全部质量字段必须有限。

## 遮挡恢复

有当前 mask 的帧先检查 mask 和深度，再合批执行 FoundationPose refinement；中间帧按上述深度门控流程处理。失败对象恢复最后接受的内部 `pose_last`，进入 `LOST`；历史位姿只用于内部回滚，不作为当前观测发布。批量推理发生运行时异常时先回滚，再逐对象调用以隔离失败。非有限候选直接逐对象拒绝，不触发整批重算。后端只替换内部位姿张量，回滚保留原张量引用，无需深拷贝。

`LOST` 对象不再执行正常 refinement。到期帧先刷新 SAM2 视频状态，再由到期对象共享一次 Grounding DINO 检测，按规范化 prompt 唯一匹配候选，随后重新分割并调用现有 `register()`。缺失、歧义或几何校验失败时继续保持 LOST；恢复使用与正常跟踪相同的质量阈值。

| `TrackingConfig` 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `recovery_enabled` | `True` | 是否尝试恢复 LOST 对象。 |
| `recovery_retry_interval_frames` | `15` | 恢复失败后的重试间隔，按处理帧计数。 |
| `recovery_box_threshold` | `0.3` | 恢复检测框阈值。 |
| `recovery_text_threshold` | `0.25` | 恢复检测文本阈值。 |

首次恢复在丢失后的下一处理帧尝试。在线 worker 和离线标注负责跳过重复相机序列，可通过 `tracking_config` 参数传入相同配置。

通过注册校验的 mask 使用 SAM2 已有 ID 的当前帧 `add_new_mask()` 校正，不重置健康对象。合并后重新校验各对象，全部检查完成后才提交各对象的 TRACKING/LOST 状态；候选在合并后被拒绝时保留原定重试间隔。过期提示输入、输出和合并索引共同裁剪，保留首帧 conditioning。

在线发布、离线有效性和训练数据过滤契约见[主项目说明](../../docs/object_reconstruction.md#标注导出与回放)。

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
