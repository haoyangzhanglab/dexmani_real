# 手—物点云与位姿追踪

当前路径为 Grounding DINO 首帧定位与 SAM2.1 small 连续分割 → FoundationPose++ 2D 引导与位姿细化 → 独立的实测点、固定模型点与 FK 手点。FoundationPose++ 只采用 bbox 引导核心，不含 Kalman、Cutie、Qwen 或 SAM-HQ。分割与单图 3D 重建是不同任务；Hunyuan3D-2.1 的资产准备流程保留。

命令从 workspace 根目录执行，运行产物输出到新目录，默认使用 `reports/`，也可指定其他位置；不覆盖输入文件或写回 Raw，模型下载可复用已有缓存目录。Raw 始终只读。`--help` 和 `--check` 不连接设备。

## 安装与权重

命令中的 `real_robot` 是示例环境名，不作准入条件。安装脚本根据当前已安装版本生成基础依赖约束，避免安装增量依赖时替换 Torch、Torchvision、NumPy 和 Transformers；新工作站先按 `scripts/install_perception.sh --help` 准备所需组件。缺少 FoundationPose 源码时下载 revision `a1b694b83e633c2cb6115b9063d940a687759392`，已有源码及本地修改直接复用。其官方权重需放入 `third_party/FoundationPose/weights/2023-10-28-18-33-37` 与 `2024-01-11-20-02-45`，每个目录包含 `config.yml` 和 `model_best.pth`。下载入口见 [FoundationPose 官方说明](https://github.com/NVlabs/FoundationPose)。

安装脚本补齐缺少的源码和增量依赖；已准备好的环境可直接下载权重并检查本地文件与依赖：

```bash
conda activate real_robot
# 新工作站配置完成基础环境后执行；已有环境不必重复安装。
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam
python -m cordvip_geometry check --offline-input --headless
```

[Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) 默认下载 revision `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`；运行允许使用本地修改。默认权重为 `reports/cordvip_models/sam2.1_hiera_small.pt` 和 `reports/cordvip_models/grounding-dino-base/`，下载使用明确的官方 revision 并保存来源记录。SAM2 从源码目录直接加载；禁用填洞后处理，不需要编译 SAM2 可选 CUDA 扩展。Grounding DINO 使用本地 safetensors，关闭自定义算子，不在推理时联网下载。模型加载要求所用源码与依赖提供实际调用的 API。

`--sam-python` 默认当前解释器，也可指向另一个具备推理依赖的环境；不按环境名或精确版本判断。[SAM 配方](../src/cordvip_geometry/sam2_config.py)检查本地文件与依赖是否存在，首帧文本检测额外检查 Grounding DINO 依赖。父进程按实际能力检查：mask-only 配合 `--no-3d` 不要求父环境的 Torch 或 Open3D；位姿估计仍需要 Open3D，即使不显示 3D；图像处理始终需要 cv2。手点另需 Real 包的离线 FK 依赖。

`--sam-root`、`--sam-checkpoint`、`--sam-config`、`--dino-weights` 可指定位置。源码目录不要求 Git checkout，权重与模型配置的实际结构须匹配。`--box-threshold` 默认 0.3，`--text-threshold` 默认 0.25，只控制首帧检测，不表示追踪或位姿置信度。分割和位姿估计使用各进程可见的第 0 张 CUDA 卡，通过 `CUDA_VISIBLE_DEVICES` 选卡；建模命令的 `--device` 仅控制 Hunyuan 生成阶段。静态检查不证明 GPU 推理或扩展 ABI 可用。

追踪默认 BF16 autocast、eager。首帧文本检测按需加载 Grounding DINO，确认后释放；SAM2 图像预测和视频传播共用一份模型。复用确认 mask 或手动画多边形时，不加载也不要求 Grounding DINO 权重。每个追踪帧共享一次图像编码，图像和特征缓存各 1 帧；保留首帧 conditioning，历史窗口由模型所需的时序记忆和指针决定，并记录在 `models.json`。SAM2 对初始化 mask 会重采样，不承诺首帧输出与输入逐像素相等。

如需 large 对照，先 `python -m cordvip_geometry download --models sam --sam-variant large`，然后给追踪或建模命令同时追加：

```bash
--sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
--sam-checkpoint reports/cordvip_models/sam2.1_hiera_large.pt
```

`models.json` 记录实际配置、源码和权重路径、依赖版本及加载耗时；`records.jsonl` 记录分割计算、IPC、各物体位姿耗时与场景处理总耗时。首条记录包含初始化；稳态处理耗时不等于含解码、FK、显示、存盘和策略推理的端到端帧率。

## 完整离线回放

单物体与多物体均由同一个 scene 核心执行：

```bash
python scripts/cordvip/track_object.py \
  --scene reports/cordvip_multiobject/toy_basket.json \
  --episode episodes/pick_place_toy/episode_20260827_165923 \
  --with-hand --headless --all-frames --save-frames --save-video \
  --output reports/cordvip_demo/toy_basket_sam2
```

已有本地 toy 资产也可直接运行单物体：

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place_toy/episode_20260827_165923 \
  --asset reports/cordvip_toy/asset \
  --initial-selection reports/cordvip_toy/first_frame_selected \
  --with-hand --headless --all-frames \
  --output reports/cordvip_demo/toy_sam2
```

本地资产与 Raw 不随 Git 分发。单物体输出使用 `object__` 前缀，与多物体使用相同格式。首帧选择核对原始 RGB、深度、内参、畸变、身份和时间；已有选择省去文本检测，但仍需 Grounded-SAM2 做后续传播。无窗口模式每个实例都须提供已确认选择；不会自动取最高分候选。

场景文件格式，路径相对 workspace：

```json
{"objects": [
  {"id": "toy", "role": "object", "asset": "reports/cordvip_toy/asset",
   "initial_selection": "reports/cordvip_toy/first_frame_selected"},
  {"id": "basket", "role": "reference", "asset": "reports/cordvip_basket/tracking_asset",
   "initial_selection": "reports/cordvip_basket/first_frame_selected"}
]}
```

省略某个 `asset` 就只输出该实例的语义实测点，不估计其 pose。场景实例在首帧确认后固定，实际容量受显存和 IPC 消息大小限制。新录像或 live 相机不能套用历史选择；省略 `initial_selection`，可给 `prompt` 并在窗口确认，或手动画多边形。初始化始终使用同一冻结帧，选择保存为 `initial_selections/` 和可复用的 `replay_scene.json`。

省略 `--headless` 打开窗口，左右面板分别显示已处理源 RGB 与该帧场景叠加图；初始化时显示冻结首帧。`v` 切换实测/模型图层、`s` 截图、`q` 退出。初始化及追踪在推理线程执行，GUI 可处理取消。省略 `--all-frames` 则按 `--replay-fps` 提供最新帧；推理较慢会跳过源帧，不补结果、不改源时间。`--max-frames N` 明确限制处理前缀。

相机尺寸和采集帧率仅用于实时源，手点数仅用于 `--with-hand`；`--replay-fps` 用于有节拍的离线输入或保存视频。检测阈值仅在首帧文本检测时检查，位姿匹配参数仅在场景包含资产时检查；未启用的分支不阻塞独立实测点云。

输入也可用 `--rgbd-dir DIR`。其中 NPZ 采用六位编号，包含 RGB uint8、米制 `depth_m`、`K`、畸变模型及五项系数、`source_id/timestamp_ns/clock` 和可选 `row_index`。不猜测缺失单位、内参或时间。复用 Raw 首帧选择时，保存帧必须包含匹配的 `row_index`，缺失时拒绝复用。

只看输入而不加载模型：

```bash
python -m cordvip_geometry replay --episode /path/to/raw_episode \
  --headless --max-frames 10 --output reports/cordvip_demo/observed
```

离线位姿追踪使用 `track`；需要处理每个源帧时加 `--all-frames`。

只读 RGB-D 预览的 `--headless` / `--no-3d` 不计算三维显示点，也不要求 `--display-stride` 有效；`--video-fps` 仅用于 `--save-video`。RGB-D 图像与来源记录照常保存。

预览退出时将最后一个完整处理帧保存一次到 `last_frame.png`，包括 Q、帧数上限或后续处理异常退出；没有完整帧时不保存。GUI 与 `--save-video` 仍逐帧生成画面，主动截图与 `--save-inputs` 行为不变。

## 实时相机入口

下面命令会连接 RealSense，须由操作者明确启动；不会读取机器人关节或发送运动命令：

```bash
python scripts/cordvip/track_object.py \
  --asset reports/cordvip_demo/object_asset --object-prompt 'white basket' \
  --serial CAMERA_SERIAL --output reports/cordvip_demo/live
```

模型加载后冻结首帧，确认候选或多边形，再开始传播和位姿估计。当前 live 入口只输出物体几何，`--with-hand` 仅支持 Raw。真实相机吞吐、传感器同步和物理位姿精度需要单独验收。

## 几何和失效语义

首帧候选生成与人工选择使用原始 RGB 网格，确认 mask 随该帧 RGB-D 去畸变；后续传播、位姿估计和点云构建使用同帧、同网格的去畸变数据。深度 scale、内参来自输入。初始布尔 mask 的重叠像素先排除，若某实例因此变空则拒绝初始化。随后 SAM2 根据 logits 施加互斥约束，输出与记忆编码使用互斥 mask。空 mask 保持空；不做填洞后处理、不用旧 mask 顶替丢失目标，也不做高层自动增删 ID。模型仍可能把篮孔后的背景预测为物体，须结合源图检查。

FoundationPose++ 2D 引导来自 [上游核心](https://github.com/teal024/FoundationPose-plus-plus) revision `58aa7154de23ae86d542366858fc6bcca3b747b0`：当前 exclusive mask 的 bbox 中心沿相机射线更新内部居中 mesh pose 的 x/y，保留上一帧 z 和旋转，再调用原版 refinement。它不从 bbox 中心取深度，不改资产原点。外部始终返回 `T_camera_object`，物体单位 m。缺少有效深度、细化错误或模型/观测不一致会锁存 pose loss；需要显式新会话注册，不自动恢复或平滑。

同帧各物体共享位姿细化所需的 RGB、深度 erosion / bilateral filtering 和 XYZ map，滤波参数沿用上游；缓存不跨源帧。实测点云继续使用原始实测深度。资产初始化固定 NumPy 采样种子 0，避免上游随机 mesh diameter 改变 voxelization 和 crop window；采样种子与计算出的直径写入模型元数据，初始化结束恢复调用者的 NumPy RNG。

实测点由当前语义 mask 与原始实测深度反投影，独立于模型；完整模型点由固定资产采样与 pose 得到。支持像素和残差只诊断二者一致性，不是真值或置信度。软玩具形变、单图不可见面和尺度误差不会因替换模型自动解决。

`--with-hand` 读取同 Raw 行的实测 arm/hand 关节，使用对应相机外参与当前实验安装配置（可设 `--experiment-config`）做 FK。默认 1024 手点和固定 link IDs；缺失关节不能用 action、零或历史值填充。无真实时钟证据时 `hand_timing_available=False`、skew 为 NaN；Raw 行关联不等于物理同步。

Raw 几何通过 `EpisodeReader.read_timing_rows()` 读取所需的源时间与身份列，要求 `timing/rows` 符合当前 Raw schema、dtype 和行数。结构缺失或不匹配直接报错；列中的 `-1` 缺失源时钟保持原值，仅支持按行关联的诊断，不代表通过训练或物理回放准入。

## 内存接口与产物

宿主已有的 RGB-D 和实测关节应直接调用内存接口，不另开设备。模型加载及顺序调用须位于同一个推理 owner，GUI 可用 `SceneWorker`：

```python
from cordvip_geometry.segmentation import GroundedSAM2Segmenter
from cordvip_geometry.scene_tracking import SceneTracker, scene_payload
from cordvip_geometry.rgbd import rectify_frame

segmenter = GroundedSAM2Segmenter(sam_python, sam_root, checkpoint,
                                 sam_config='configs/sam2.1/sam2.1_hiera_s.yaml',
                                 dino_weights=dino_weights, log_path=run_dir / 'sam2_worker.log')
tracker = None
try:
    segmenter.start()
    # assets: {id: (metric_mesh, fixed_points, fixed_normals)}
    tracker = SceneTracker(segmenter, assets, foundationpose_root=fp_root,
                           debug_dir=run_dir / 'pose')
    # frame 保留原始 RGB-D；confirmed_masks 使用同一去畸变网格。
    rgb, depth_m, K, _ = rectify_frame(frame)
    result = tracker.step(frame, rgb, depth_m, K, initial_masks=confirmed_masks)
    packet = scene_payload(result, roles, valid=True, state='REPLAY',
                           max_age_ms=500, hand=hand_for_this_frame)
    # 后续帧不再提供 initial_masks；消费时检查实例有效性和来源时间。
finally:
    tracker.close() if tracker is not None else segmenter.close()
```

`HandCloudModel.transform(measured_hand_qpos, T_camera_handbase)` 供已有宿主生成固定 FK 手点。每帧结果带源身份、row、时钟、时间、每个实例的 cloud、pose、固定模型点/法向与独立有效性；内存数组借用，异步保存者应复制。

`Frame` 检查原始 RGB-D 与内参，`SceneTracker.step()` 检查宿主传入的去畸变数组。内部场景提取、逐物体跟踪和渲染复用入口保证；新模型位姿及转换为 float32 后的几何结果仍需检查。静态资产的点和法向在 `SceneTracker` 构造时检查并复制，调用方后续修改原数组不会改变追踪资产。帧和结果中的借用数组在消费期间应保持不变；`scene_payload()` 单独检查宿主提供的手部几何和来源信息。

| 输出 | 内容 |
|---|---|
| `run.json` / `models.json` | 配置、资产/标定来源、源码与权重路径、依赖版本、加载耗时 |
| `records.jsonl` | 实际处理行、分割/IPC/位姿耗时、GPU 内存、状态与诊断 |
| `latest.npz` | 原子发布最近快照；退出时写失效快照，失败则尝试删除 |
| `result.json` | 最终状态、错误、各资源清理状态与快照失效确认；发布失败会报告错误并以失败状态退出 |
| `frames/` / `inputs/` | `--save-frames` / `--save-inputs` 可选逐帧归档 |
| `overlay.mp4` | `--save-video` 的检查视频；播放率不代替源时间 |
| `sam2_worker.log` | 隔离分割进程的日志 |

NPZ 的 `object_ids/object_roles` 给出实例顺序；数组按 `toy__`、`basket__` 或单物体 `object__` 前缀保存。顶层 `valid` 表示此次源帧发布有效，每实例另有 `pose_valid/observed_valid`。pose 失效时该实例 pose 为 NaN、模型点为空，当前实测点仍可有效。成功发布的会话失效快照会清空全部可消费几何。手点另含相机/base 系点和法向、link IDs、joint_state、源时间与 skew。

实时消费者须自行核对原始源时间、时钟与进程存活；`published_ns` 新不等于图像新。强制终止或文件系统故障可能留下旧的 valid 快照，不能只凭其标志继续消费。`REPLAY` 不做 live 年龄准入，也不授权机器人运动。库没有接入主包 runtime、Raw writer、13 字段导出或策略部署。

## 新物体资产

```bash
python -m cordvip_geometry download --models shape
python scripts/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'white basket' \
  --output-dir reports/cordvip_demo/build --preview
```

Grounded-SAM2 候选明确人工确认后生成 RGBA；Hunyuan3D-2.1 在另一阶段加载，输出未定尺度 mesh、默认 1024 表面点与法向。没有真实尺寸不能追踪。检查模型轴后，若 z 轴确实对应实测 12 cm 高度：

```bash
python -m cordvip_geometry prepare \
  --mesh reports/cordvip_demo/build/generation/mesh_unscaled.obj \
  --size-axis z --size-meters 0.12 \
  --output-dir reports/cordvip_demo/asset --preview
```

已有 CAD/扫描 mesh 可直接 `prepare`。完整单图流程使用 `object build`，与 `reconstruct_object.py` 共用参数和实现；分阶段使用 `object segment --list-only` 保存候选、`object select` 继续选择、`object generate` 生成。无窗口选择必须指定 `--instance`；候选 score 是 Grounding DINO 检测分数，mask_score 是 SAM2 预测 IoU，均不是人工真值。

首帧复用读取 `selection.json` 指向的 mask，并核对原始 RGB-D、相机参数、来源身份、时间和 Raw 行号。分割建模保存一份 `mask.png`，场景追踪保存一份 `mask.npy`；`foreground_rgba.png` 是 `object generate` 的实际输入。

## 数据语义与资源归属

相机/replay iterator 和模型分别由各自的线程持有；GUI 读取 latest mailbox，每次最多一个推理请求。输入数组发布后不可变。关闭先撤销待处理结果，再等待 owner 清理；已经进入 SDK/CUDA 的调用不能由 GUI 强行中断。SAM cancel 仅撤销自己的进程组，不中断父进程 FoundationPose CUDA 调用。

Frame 保留原 RGB-D、source identity、row 和 clock。没有相机来源证据时显式使用 Raw row/relative identity；主机接收时间不是曝光时间。离线首帧可以等待注册，live source 持续前进；MP4 解码 RGB 不声称与最初采集像素逐位相同，耗尽迭代器才验证完整帧数。

几何统一在 color-camera frame 中使用 m。深度是 camera Z，零代表缺测；Open3D ray direction 保持 Z=1，因此命中参数也是 Z，不能归一化成欧氏射线。去畸变仅接收已支持的 forward Brown-Conrady 模型，depth/mask 使用最近邻，避免跨边界伪造深度。

实测点云只来自同帧互斥 semantic mask 与深度，不修补孔洞、不分配重叠像素，也不依赖 mesh。错误 mask 仍可能包含背景。像素按全局 stride 网格上的 row-major (v,u) 排列；模型支持比较复用相同点序，残差为 measured Z − rendered Z，无模型表面处为 NaN。coverage/support fraction 是几何诊断，不是置信度，未定义时为 None。

实例缺资产仍可提供实测点；pose 丢失按对象锁存，需新会话和明确首帧 mask 重新注册。会话失效清空可消费几何，单独 pose 失效不删除独立实测点。FoundationPose++ 的 bbox 中心仅引导 centered mesh 的 2-D 位置，保留旧 Z/rotation，不用深度孔洞构造 3-D 质心；输出变换为 `T_camera_asset = T_camera_centered @ T_centered_asset`。

手部几何按明确关节名称顺序接收 rad，URDF origins/mesh scales 定义 m。每个 link 至少一个固定样本，其余按表面积分配，sample/link ID 在模型生命周期内固定；法向来自 mesh faces。Raw 手点使用同一 row 的实测关节、数据对应的相机外参与当前 mount，不插值，也不连接设备。

Hunyuan 仅在独立阶段使用本地 2.1 权重，输出任意尺度与对象轴；物理定尺度是单独的显式步骤。生成不隐式执行 rembg、几何清理、居中或缩放。来源记录保留源码、权重、输入与输出路径，以及实际参数和依赖版本。
