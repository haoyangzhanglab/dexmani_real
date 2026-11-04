# 物体资产与 RGB-D 几何

CLI 提供资产准备、Raw 逐行追踪和结果查看；Python API 返回内存结果，供宿主处理 Raw 或相机样本。命令默认在 workspace 根目录执行。来源数据和物体资产只读，追踪结果写入新目录。

## 安装与权重

先准备兼容的 Torch、Torchvision、NumPy、Transformers、CUDA 编译器和头文件，以及 Boost system/program_options 和 OpenMP，再运行安装脚本：

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam shape
python -m cordvip_geometry check --json
```

`real_robot` 是环境名示例。安装脚本约束已有基础版本、复用本地源码，补齐增量依赖并编译 nvdiffrast / FoundationPose mycpp。保留已有 `CPATH`、`MAX_JOBS` 和 `TORCH_CUDA_ARCH_LIST`；离线编译需按最终 GPU 设置架构。

源码默认 revision：

| 上游 | revision |
| --- | --- |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | `a1b694b83e633c2cb6115b9063d940a687759392` |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | `253ac4fcea7de5f396371124af597e6cc957bfae` |

源码放在 workspace 的 `third_party/`，权重默认位于 `third_party/weights/`，可用 `--models-dir` 更改根目录。下载采用 [models](../src/cordvip_geometry/models.py) 中固定的官方 revision，文件齐全时直接复用；运行仅加载本地模型。

FoundationPose 权重按其[官方说明](https://github.com/NVlabs/FoundationPose)另行取得，分别放入 `third_party/FoundationPose/weights/2023-10-28-18-33-37` 和 `2024-01-11-20-02-45`，各含 `config.yml` 与 `model_best.pth`。授权以各上游仓库和模型条款为准。

SAM 默认 small、BF16、eager。large 先下载权重，再为相关命令传 `--sam-variant large`；自训模型同时传 `--sam-config` 和 `--sam-checkpoint`。`--sam-python` 可指定独立解释器，父子进程通过标准 Python 序列化传递数组。分割、位姿与建模均使用进程可见的第 0 张 CUDA 卡，通过 `CUDA_VISIBLE_DEVICES` 选择。

`check` 只核对本地解释器、源码、权重与扩展路径；GPU 推理、扩展 ABI 和长序列显存需单独实测。普通 import 和 help 不加载模型或设备。

## 准备物体资产

单图建模和已有 mesh 校准流程见[物体重建与点云](../../../docs/object_reconstruction.md)。阶段命令为 `object segment/select/generate/build/prepare/preview`，输出参数统一为 `--output`，文本参数为 `--prompt`。`object generate --image PATH` 直接接收含有效前景和透明背景的 RGBA 文件，不依赖分割目录或确认记录；`build` 直接使用分割阶段生成的 `foreground_rgba.png`。

Hunyuan 在独立生成进程中运行；生成 mesh 导出前简化到最多 20,000 面，仍需显式指定米制尺度。`generation_result.json` 记录 `source_faces`、`mesh_simplification` 和实际导出的 `faces` / `vertices`。prepare 保留原点与朝向，在米制网格上按指定 seed / 点数采样固定点及三角面法向。

多个物体的总点数分配：

```bash
python scripts/cordvip/allocate_points.py \
  assets/pointcloud_template/object/pnp_apple_banana
```

默认生成 `task/sampled_1024/`；已有输出拒绝覆盖，可用 `--output` 指定新目录。每个物体至少 1 点，剩余按闭合实体体积比例分配并用最大余数法补齐，总计 1024 点。碗计算壁和底的实体体积。新目录包含完整资产、NPY、PLY 和 `allocation.json`，可直接作为跟踪资产。

## 首帧确认与逐行追踪

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 \
  --asset /tmp/cordvip/object_asset --prompt 'white basket.' \
  --with-hand --output /tmp/cordvip/scene_001
```

首帧冻结后人工确认候选，或省略 prompt 手画 mask。选择保存为 `initial_selections/<id>/`。`--initial-selection PATH` 和 scene 的 `initial_selection` 都接受 PNG/NPY mask 文件，或含 `selection.json` 的目录；目录只需 `mask_file` 指向目录内的真实 mask。输入 mask 明确应用于当前首帧原始 RGB 网格，由 tracker 统一去畸变一次，不要求确认标签或候选索引。目录若提供 `source_rgbd.npz`，其中 RGB、米制深度、内参、畸变模型与系数必须与实际首帧逐项精确相等；没有绑定文件时不验证来源。`source_id`、时间戳、时钟和行号仅作说明，复用不改写已有输入。

多物体通过场景文件声明，路径相对 workspace：

```json
{"objects": [
  {"id": "toy", "role": "object", "asset": "/tmp/cordvip/toy_asset",
   "initial_selection": "/tmp/cordvip/toy_first_frame"},
  {"id": "basket", "role": "reference", "asset": "/tmp/cordvip/basket_asset",
   "initial_selection": "/tmp/cordvip/basket_first_frame"}
]}
```

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 --scene scene.json \
  --with-hand --headless --output /tmp/cordvip/scene_002
```

无 asset 的对象输出 mask 和独立实测点。`role` 仅作 metadata。headless 要求每个对象提供当前首帧的有效初始 mask。默认处理全部 Raw 行；`--max-frames N` 显式限制前缀，结果标为 partial。同一 Raw 行提供 RGB-D、实测关节与时钟，行关联不证明物理同步。

## Python API

独立子包安装后直接 import。`SceneTracker.open` 负责一次加载与收尾；`workspace` 默认当前目录，资产和选择的相对路径基于 workspace。`initial_selections` 与首帧 `step` 的 `initial_masks` 二选一。`models_dir`、`sam_python` 及 SAM 配对可按需指定。`log_dir` 省略时，SAM 日志输出到 stderr。

```python
from dexmani_real.data.raw import RawReader
from cordvip_geometry import SceneTracker, track_raw

with (
    RawReader(episode) as reader,
    SceneTracker.open(
        {"object": asset_dir}, initial_selections={"object": selection_dir}
    ) as tracker,
):
    for result in track_raw(reader, tracker, with_hand=True):
        points_camera = tracker.model_points(result, "object")
```

`track_raw` 不接管 reader / tracker 的生命周期，也不保存结果；`result["hand"]` 是同行手几何。宿主持有相机时：

```python
from cordvip_geometry import RGBDFrame, SceneTracker

with SceneTracker.open({"object": asset_dir}) as tracker:
    first = RGBDFrame.from_camera(camera, first_row.camera)
    result = tracker.step(first, initial_masks={"object": confirmed_mask})
    for row in rows:
        result = tracker.step(RGBDFrame.from_camera(camera, row.camera))
        points_base = tracker.model_points(
            result, "object", T_target_camera=camera.T_xarm_base_from_color
        )
```

`camera` 是标定快照，样本包含 `uint8 rgb`、原始 `uint16 depth`、`timestamp_ns` 与 `ring_sequence`。Frame 统一检查动态数组、米制深度、标定和时钟；tracker 固定使用一套相机几何，按来源时序处理。实时重复帧复用结果，Raw 每行独立处理。`model_points` 返回米制 `float32 (N, 3)`；默认 camera frame，指定有效 `T_target_camera` 可变换到目标系。pose 无效或无 asset 时返回空点。

宿主在所需位置主动调用这些接口；当前主包的控制、Raw writer 和策略观测未接线。

## 输出与查看

| 输出 | 内容 |
| --- | --- |
| `scene.json` | 对象 ID、role、asset、选择与标签 |
| `run.json` | 实际参数、资产来源、颜色、单位及可选手标定 |
| `model_templates/<id>.npz` | 有资产对象的固定 `points_object` / `normals_object`，米制 float32，原顺序 |
| `frames/<row>.npz` | 来源行/时钟、RGB、K、mask、实测点、位姿状态和深度支持；可附手几何 |
| `models.json` | 工厂已返回的模型来源；初始化失败时只记录已取得的信息 |
| `result.json` | completed / partial / interrupted / failed、失败原因及处理行数 |
| `sam2_worker.log` | CLI 的分割进程日志与 traceback |

逐帧对象字段使用 `<id>__` 前缀。`pose_valid` 与 `observed_valid` 独立；pose 失效时 `T_camera_object` 为 NaN，实测点仍可有效。`candidate_T_camera_object` 仅供诊断；已计算的深度支持 mask、残差和统计在失败帧中也保存。模型点由模板与有效 pose 重建：

```python
import numpy as np

with np.load(directory / "model_templates/object.npz", allow_pickle=False) as saved:
    points_object = saved["points_object"]
with np.load(directory / "frames/000000.npz", allow_pickle=False) as row:
    if bool(row["object__pose_valid"]):
        pose = row["object__T_camera_object"]
        points_camera = (points_object @ pose[:3, :3].T + pose[:3, 3]).astype(
            np.float32
        )
```

输出可位于 `episodes` 或 `rollouts` 下的独立新目录，例如 `rollouts/derived/objects`；仍拒绝写入实际输入、Raw episode 内部或已有目标，悬空符号链接也不能覆盖。模板与帧文件独占创建；未完成的 NPZ 会清理。技术错误直接结束运行并保留成功前缀，深度支持不足按算法形成 `POSE_LOST`。模板使用 object frame，实测点和手点使用 color-camera frame，单位均为米；启用手几何时每帧附外参、关节状态与来源时间。法向仅表示网格表面几何。

```bash
python -m cordvip_geometry view /tmp/cordvip/scene_001
python -m cordvip_geometry view /tmp/cordvip/scene_001 --no-3d
python -m cordvip_geometry view /tmp/cordvip/scene_001 --headless
```

查看器使用 `n` / Space 前进、`v` 切换 observed / model、`q` / Esc 或关窗退出。每行只读取显示字段一次，模板在首次 model 显示时加载；静止时只轮询事件与渲染。`--no-3d` 只显示二维结果，`--headless` 只读取 RGB 与来源行/时钟。查看仅依赖输出目录。
