# CordVIP 几何核心

`cordvip_geometry` 提供米制物体资产、RGB-D 分割与位姿跟踪，以及物体和手部点云分析。本文说明核心接口、坐标约定和结果格式。安装与权重准备见[包说明](../README.md#安装与命令)，任务构建、Raw 标注和导出命令见[实验入口](../../../scripts/cordvip/README.md)。

## 架构与数据流

资产链：**物体图像 → Grounding DINO 检测 → SAM2 分割 → Hunyuan3D 网格生成 → 实测尺度校准 → 固定表面点与法向**。

跟踪链：**RGB-D 与首帧掩膜 → SAM2 掩膜传播 → 实测点云提取 → 可选 FoundationPose 位姿估计与深度支持检查 → 场景几何结果**。

两条链通过米制物体资产连接。提供资产的对象进行模型位姿跟踪；未提供资产的对象仅输出分割和实测点云，两类对象可存在于同一场景。CLI 负责首帧选择和结果保存，Python API 返回内存结果，输入来源与保存方式由调用方决定。

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

### 资产与帧输入

`prepare_asset(mesh_path, output, ...)` 将网格按统一尺度转换为米制，并写入新的资产目录。尺度必须由 `scale_to_meters` 或完整的 `size_axis`、`size_meters` 指定，保留网格原点与轴向关系。`load_asset(path)` 返回 `(mesh, points, normals, metadata)`，固定点和法向为只读 `float32[N,3]`，位于物体局部坐标系。

`RGBDFrame` 接收已对齐到原始彩色相机网格的 RGB 与深度：

| 字段 | 约定 |
| --- | --- |
| `rgb` | `uint8[H,W,3]`，RGB 通道顺序。 |
| `depth_m` | 浮点 `[H,W]`，内部转为 `float32`；有限、非负，表示相机 Z 深度，单位米，0 表示缺测。 |
| `K` | 有限的零 skew 针孔内参 `[3,3]`，焦距为正。 |
| `distortion_model`、`distortion_coeffs` | 支持 `none` 和 `brown_conrady`，均需提供 5 个有限系数；`none` 对应全零系数。 |
| `source_id`、`timestamp_ns`、`clock`、`row_index` | 可选来源信息，随结果保留。 |

宿主持有相机快照时，可用 `RGBDFrame.from_camera(camera, row.camera)` 将原始 `uint16` 深度按相机记录的尺度转换为米制帧，再交给 `tracker.step(frame)`。

### 顺序处理 Raw

以下示例在仓库根目录、已配置感知依赖的环境中运行。将 Episode、资产和首帧 mask 路径替换为实际输入；mask 必须对应该 Episode 的首帧原始 RGB：

```python
from pathlib import Path

from cordvip_geometry import SceneTracker, track_raw
from dexmani_real.data.raw import RawReader

episode = Path("episodes/pnp_apple_banana/episode_20261101_165909")
assets = {"apple": Path("assets/pointcloud_template/object/pnp_apple_banana/apple")}
selections = {"apple": Path("path/to/first_frame_mask.png")}

with (
    RawReader(episode) as reader,
    SceneTracker.open(assets, initial_selections=selections) as tracker,
):
    for result in track_raw(reader, tracker):
        item = result["objects"]["apple"]
        observed = item["cloud"].observed_points_camera
        model = tracker.model_points(result, "apple")
        print(result["frame"].row_index, item["state"], len(observed), len(model))
```

`assets` 将对象名映射到资产目录或 `None`；`None` 表示仅提取分割与实测点云。对象名须为唯一、非空的单级目录名。相对资产路径和选择文件路径以 `workspace` 为基准，默认是当前工作目录。

`track_raw(reader, tracker)` 按 HDF5 行顺序推进，每行调用一次 `step()`，保留原始 `camera_ns` 与行号。它返回内存结果，Reader 与 Tracker 由调用方关闭。指定 `with_hand=True` 时，还需 Raw 中的实测 Arm／Hand 关节及相机外参；结果增加 `hand`，由同一行反馈与组合 URDF 正运动学得到手部几何。行对齐与时间戳差只描述记录关系，不证明硬件物理同步。

### 初始化与生命周期

`SceneTracker.open()` 是上下文管理器，一次加载资产与模型，退出时释放 SAM 子进程及跟踪资源。SAM 日志默认输出到宿主 stderr，指定 `log_dir` 可保存日志。

首帧掩膜有两种传入方式，二选一：

- `open(..., initial_selections=selections)`：按对象名提供 PNG／NPY 文件或含 `selection.json` 的目录；目录通过 `mask_file` 指向掩膜。文件支持 bool／uint8 单通道 NPY 和 `1/L/P` PNG，非零像素为前景，`P` 模式先按调色板转灰度。
- `step(frame, initial_masks=masks)`：直接传入对象名到 `bool[H,W]` 的映射。每个有资产对象都需有同名掩膜；可先调用 `detect_masks(frame, prompts)` 批量检测，取得无歧义的首帧掩膜。

输入掩膜始终使用原始 RGB 网格。Tracker 内部去畸变并排除首帧物体之间的重叠像素；`result["rgb"]`、点云掩膜和像素索引对应去畸变后的网格，`result["frame"]` 保留原输入。

后续帧调用 `step(frame)`，同一场景内图像尺寸、内参和畸变参数须固定。Tracker 按调用顺序推进，时间戳不参与去重、排序或 TTL 判断。

新场景先调用 `reset()`，它清空场景状态并保留模型、资产及 `initial_selections` 配置。跨场景使用不同 mask 时，可在 `open()` 时省略该配置，每次 reset 后显式传入新的 `initial_masks`。

### 位姿门控参数

以下参数传给 `SceneTracker.open()`，深度支持检查在有资产的对象上生效：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `pixel_stride` | `2` | 实测点云的图像采样步长。 |
| `depth_tolerance_m` | `0.015` | 实测 Z 与模型 Z 差值的绝对值上限，单位米。 |
| `min_matched_pixels` | `30` | 首帧采样实测点数及位姿深度支持点数的下限；FoundationPose 另有初始化深度要求。 |
| `min_matched_fraction` | `0.15` | 模型深度支持点数占当前采样实测点数的比例下限。 |
| `max_reregistrations` | `0` | 非负整数，限制每次丢失后的连续重新注册次数，成功恢复后清零。默认丢失后保持失效，直到场景重置。 |

## 返回结果与几何语义

`step()` 返回 `frame`、去畸变后的 `rgb`、按对象名组织的 `objects`，以及值为 `initialize` 或 `track` 的 `operation`。每个 `objects[name]` 的常用字段如下：

| 字段 | 含义 |
| --- | --- |
| `cloud` | `ObjectCloud`；`observed_points_camera` 为 `float32[N,3]` 实测点，`observed_colors` 为对应 RGB；`semantic_mask` 为分割掩膜，`observed_mask` 标记采样后有有效深度的像素。 |
| `pose`、`pose_valid` | 物体局部系到彩色相机系的 `T_camera_object[4,4]` 及其有效标记；无效时 `pose` 为 `None`。 |
| `candidate_pose` | 本次位姿估计候选，可能被深度门控拒绝；使用有效模型几何应以 `pose_valid` 为准。 |
| `observed_valid` | 当前实测点数是否大于 0，与位姿有效性独立。 |
| `state`、`error` | 对象状态及最近的位姿失败原因；正常时 `error` 为 `None`。 |

| 状态 | 含义 |
| --- | --- |
| `TRACKING` | 有资产且位姿有效，可由固定物体模板生成模型点。 |
| `POSE_LOST` | 有资产但位姿无效，实测点仍可能存在。 |
| `OBSERVED_ONLY` | 未提供资产，只输出分割和实测点；零深度时点集为空，`observed_valid=False`。 |

三种几何均以米为单位：**实测点**来自掩膜内的深度，**模型点**来自固定资产与有效位姿，**手部点**来自实测关节与正运动学。场景点默认位于彩色相机坐标系，资产模板位于物体局部坐标系。`tracker.model_points(result, name, T_target_camera=...)` 接收彩色相机系到目标系的变换，返回目标系中的模型点；位姿无效时返回空的 `float32[0,3]`。

`cloud.model_supported_fraction` 的分母是当前采样实测点数。`observed_model_depth_residual_m` 按实测点顺序保存“实测 Z − 模型 Z”，无模型表面处为 NaN；尚未执行模型深度查询时该字段为 `None`。

`detect_masks()` 的候选缺失或歧义、首帧掩膜经处理后为空，以及有资产对象的首帧支持深度不足，会抛出 `SceneInitializationError`，调用方可结束本轮并在新场景重置。位姿注册抛出 `PoseRegistrationError` 或深度门控未通过时，Tracker 通过 `POSE_LOST` 与 `error` 报告。配置、模型执行、CUDA 和 IPC 错误继续传播。

## 离线结果与查看

`track` CLI 将结果写入新的输出目录，Raw 保持只读。主要文件如下：

| 路径 | 内容 |
| --- | --- |
| `scene.json` | 对象 ID、资产与首帧选择来源、单位和坐标系，以及可选手部模型信息。 |
| `model_templates/<id>.npz` | 有资产对象的固定 `points_object`、`normals_object`，均位于物体局部坐标系。 |
| `frames/<row_index:06d>.npz` | 去畸变 RGB、内参、来源字段及逐对象结果；对象字段以 `<id>__` 为前缀。 |

逐帧 NPZ 用 `T_camera_object` 保存有效位姿，无效时为 NaN 矩阵；`candidate_T_camera_object` 单独保留估计候选。固定模型点通过模板与有效位姿重建；指定手部输出时，逐帧还包含 `hand_points_camera`、关节状态与来源时间。字段展开实现见 [scene_arrays](../src/cordvip_geometry/io.py)。

查看命令在仓库根目录执行，读取已保存结果，无需重新推理或打开 Raw：

```bash
python -m cordvip_geometry view logs/cordvip/apple_track
python -m cordvip_geometry view logs/cordvip/apple_track --no-3d
python -m cordvip_geometry view logs/cordvip/apple_track --headless
```

默认显示 RGB 叠加与 3D 点云；`--no-3d` 只显示 RGB 叠加，`--headless` 只遍历 RGB 与来源字段并打印行信息。窗口中按 `n`／空格／Enter 前进，`v` 切换实测与模型点，`q`／Esc 退出。模型模式下只显示位姿有效对象的模型点，手部点仍可显示。

## 与 HOI 流程的衔接

主仓库的 HOI 离线入口按 `camera_sequence` 去重、跨 Episode 复用模型，并在每段开始时重置场景；每次丢失最多允许三次重新注册，结果写入原 Raw 的 `/hoi`。在线 Worker 使用默认的零次重新注册，每轮重新初始化。固定 1024 点模板、xArm base 变换、训练数据和 Checkpoint 契约见 [HOI 五阶段流程](../../../docs/object_reconstruction.md#hoi-五阶段流程)。

正式后端使用 SAM2 与 FoundationPose。[安装脚本](../scripts/install_perception.sh) 幂等应用 [tracking-cache 补丁](../scripts/foundationpose_tracking_cache.patch)，使单姿态跟踪复用 CUDA allocator 缓存，多候选注册保留释放行为；冲突直接报错。TensorRT 仅用于对照实验，实测性能与验证限制见[当前实现与性能结论](../../../docs/object_reconstruction.md#当前实现与性能结论)。
