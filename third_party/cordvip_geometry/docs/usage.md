# 物体位姿与手—物交互点云

`examples/cordvip/reconstruct_object.py` 从一张 RGB 图像或已保存 RGB-D 生成物体 mesh 与点云：Grounding DINO + SAM 2.1 文本分割、人工确认目标、Hunyuan3D-2.1 纯形状生成、固定点采样；提供尺寸时再准备米制资产。`examples/cordvip/track_object.py` 从 Raw、RGB-D NPZ 或 RealSense 输入连续估计 `T_camera_object`，显示物体实测点与模型点；Raw 回放可附加同一行实测关节驱动的 XHand 点云。输入、推理与界面分别运行，注册或推理时窗口仍能响应。

第一次使用先看[两个示例的快速开始](../../../examples/cordvip/README.md)，已有 toy 数据可直接运行下文“离线手—物回放”。接入代码时再读“手—物模态与 Python 接口”；安装和新物体准备按本文其余步骤进行。

下文命令从 workspace 根目录执行（本项目即 `dexmani_real` 根目录）。两个示例直接接收各自工作流的参数，无需再写子命令；拍照、分阶段生成、尺度校准、回放等辅助操作通过已安装的 `python -m cordvip_geometry` 使用。建模、输入读取、追踪回路、子进程和可视化均在[本库](../README.md)实现。Raw/RealSense 适配集中在库的可选 `integrations/dexmani.py`；其余模块不依赖主包。不接入 `dexmani_real` 的 runtime、Raw writer、训练导出或部署，不连接机器人，不写标定。

生成 mesh 和采样点位于 **object frame，尺度未定**；校准后的资产仍在 object frame，单位为 m。物体追踪输出位于 **color camera frame，单位 m**，单独运行物体追踪不需要相机外参。`--with-hand` 额外需要录制相机外参、实测臂/手关节和当前手安装配置，将两类模型点同时变换到 robot base。实时输入须包含 RGB、对齐深度与内参，不能直接使用普通 RGB 视频；实测点云来自每帧深度反投影，当前没有 TSDF 等跨帧表面融合。

## 多物体连续分割与独立几何

推荐从 [toy + basket 快速开始](../../../examples/cordvip/README.md#toy--篮子连续分割双位姿与手点云)运行 `track_object.py --scene scene.json`。场景文件只列 `objects`，逐实例给出唯一 `id`、`role`（object/reference）、可选米制 `asset`、已确认 `initial_selection` 或交互 `prompt`。路径相对 workspace。省略 asset 时实例只有语义实测点，不能把其点云质心或 PCA 方向作为 6D pose。

首帧逐个确认后，SAM2 因果传播固定实例 ID，每处理帧只计算一次共享图像特征；最近必要 memory 和首帧 conditioning 有界保留。实例重叠由 SAM logits 决定互斥输出，不是标定过的置信度；通用几何接口另外拒绝将重叠像素计入两个 observed clouds。默认禁用 mask 小孔填充，但模型本身仍可能覆盖孔后的背景。

多物体默认 small 权重与配套配置，单物体提示初始化仍默认 large。可覆盖 `--sam-checkpoint` 和 `--sam-config`，两者必须匹配。SAM 当前 pinned 流式实现需要同一进程中的一致源码，不能混装另一个 sam2。使用模型和权重均记录在运行目录；不需要生成模型常驻。

`SceneTracker.step(frame, rgb, depth_m, K, initial_masks=...)` 返回内存结果，首帧提供 masks，此后省略；输入为同帧、同网格、去畸变 RGB-D。固定物体采样随各自 pose 变换，实测点直接从语义 mask 反投影；模型支持点另外计算。调用者提供单一连续源并保持采集时间证据，不混用不同相机会话。tracker 顺序调用共享 FoundationPose 权重，不能并发操作这些共享 predictor。源倒退/内参变化需显式新会话；位姿失效按实例锁存，分割与实测点可继续产生。

`scene_payload` 和原单物体 `build_observation` 的格式不同，不能把两者混作同一个策略输入。NPZ 的 `object_ids/object_roles` 给出实例顺序，实例数组以 `toy__` 等 ID 为前缀。顶层 `valid` 描述源帧是否可发布，各实例分别保存 `pose_valid/observed_valid`；可能有有效实测点而没有有效 pose。失效 pose 为 NaN、模型点为空，不发布其模型支持字段；`error` 保留原因。`records.jsonl` 保存几何诊断及其候选位姿供检查，候选不等于有效 pose。

`--with-hand` 保留同 Raw 行的实测关节和 FK 点，并输出 camera/base 坐标；未知来源时差仍为 NaN。`REPLAY` 不证明传感器物理同步。实时消费者必须自行检查源时间及进程存活，不能永久信任磁盘 `latest.npz` 的 valid 值。正常结束/错误退出会发布失效且无可用几何的 latest。

默认最新帧模式只有一个源帧邮箱，不积累待处理队列；`--all-frames` 仅用于离线完整逐帧验收，`--max-frames` 明确限制处理前缀。归档只保存实际处理的源帧，不为跳帧补结果。窗口 `v` 切换图层、`s` 保存截图、`q` 退出；每个实例独立着色，末帧自动结束。输出中的 `replay_scene.json` 可复用同一录像的已确认首帧；不能用于新的相机帧。

下面后续原有单物体章节的 mesh-depth 筛选、字段和按键说明仅对应不带 `--scene` 的模式。

**重建是可选的离线资产准备步骤。** 没有 mesh 时默认使用单图生成；已有匹配的 CAD/扫描 mesh 可以直接准备资产。重建不进入追踪循环，也不改变主包的数据流。当前 FoundationPose 路径仍需要米制物体 mesh。

已在 `real_robot` 完成安装、权重校验和实际模型推理：Grounded-SAM-2 分割玩具照片，Hunyuan3D-2.1 成功生成 mesh 与点云，再按用户实测尺寸准备米制资产，并用 FoundationPose 处理对应 Raw 的全部 213 帧 RGB-D。真实异步追踪与 OpenCV/Open3D 窗口也已完成这 213 帧回放。Hunyuan3D 使用预训练权重，**只做推理，不训练或微调**。没有连接真实相机或机器人，尚未验收物理位姿精度和持续相机帧率。当前结果见[本机实施记录](../../../reports/cordvip_install/implementation_status.md)。

先看[玩具单图生成与离线追踪结果](../../../reports/cordvip_toy/README.md)。此前的[收纳篮文本分割](../../../reports/cordvip_install/basket_candidates/candidates.jpg)、[二维追踪窗口](../../../reports/cordvip_install/actual_live_gui_memory/tracking_06.png)、[三维追踪窗口](../../../reports/cordvip_install/actual_live_gui_memory/tracking_06_3d.png)仍保留供对照；其中窗口截图使用合成 RGB-D 和真实 FoundationPose 网络。这些本机产物保存在 Git 忽略的 `reports/`，不随代码分发。

## 安装与复现

安装脚本针对已有 `real_robot` 做增量安装，基线为 **Python 3.10、Torch 2.4.1+cu124、torchvision 0.19.1+cu124、NumPy 1.26.4、transformers 4.48.0**。它先校验这些版本，安装过程中用约束文件保留基线；不自动升级 Torch 或替换 CUDA。编译需现有 conda 前缀中的 CUDA 12.4 `nvcc` 和 target headers，以及系统 C++、CMake、Boost system/program_options 和环境中的 pybind11。

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download
```

安装脚本 [`scripts/install_perception.sh`](../scripts/install_perception.sh) 默认以当前目录为 workspace；从其他目录调用时，在命令后加 `--workspace /path/to/dexmani_real`。它按固定 commit 准备官方 Grounded-SAM-2、Hunyuan3D-2.1、FoundationPose 和 nvdiffrast，安装 [增量依赖](../requirements-perception.txt)，再编译 SAM2、nvdiffrast 与 FoundationPose `mycpp`。Hunyuan 直接从源码调用纯形状模块，不安装纹理模块。已有 checkout 必须匹配固定版本且没有 tracked 修改；脚本不会 reset、覆盖或删除你的源码改动。构建目标是当前 RTX 4090 的 CUDA architecture 8.9，编译并行度为 4。环境前后快照、源码版本与日志写入 `reports/cordvip_install/install_*/`。

`download` 按固定 revision 下载 Grounding DINO、SAM 2.1 和 Hunyuan3D-2.1 到 `reports/cordvip_models/`。FoundationPose 权重单独按[官方说明](https://github.com/NVlabs/FoundationPose/blob/main/readme.md)获取：

```bash
gdown --folder --no-cookies \
  'https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i' \
  -O third_party/FoundationPose/weights

python examples/cordvip/track_object.py --check --object-prompt 'red mug.'
```

检查权重目录应直接包含下面列出的两个时间戳子目录；下载中断可重新执行下载命令。安装、下载与 `--check` 均不连接真实设备，下载成功也不等同于模型推理或真机验证通过。

## 最短启动流程

### 1. 准备一次物体资产

在仓库根目录激活安装好依赖的环境：

```bash
conda activate real_robot
```

没有 mesh 时，用第一个示例从已有照片生成并查看 mesh 与点云：

```bash
python examples/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'red mug.' \
  --output-dir reports/cordvip_demo/mug_build --preview
```

确认分割候选后，程序生成 `generation/mesh_unscaled.obj`、`generation/points_unscaled.npy`、`generation/normals.npy` 与 `generation/generation_result.json`；默认 1024 点，可用 `--num-points` 调整。mesh 和点云都在模型的 object frame 内，单位未定；法向为单位向量。`--preview` 显示 mesh、点云、包围盒和坐标轴。生成记录保留采样参数、坐标系、单位、文件和模型来源。未提供尺寸时不生成米制 `asset/`。

照片尽量清楚地展示关键几何，物体不要被手遮挡。以下是可单独检查或恢复的分阶段流程。需要从 RealSense 拍照时，由你主动执行下面的命令，**它会连接并配置相机**，不连接机器人：

```bash
python -m cordvip_geometry capture \
  --serial CAMERA_SERIAL \
  --output-dir reports/cordvip_demo/object_capture
```

窗口左侧显示 RGB，右侧显示 aligned depth；深度伪彩色仅用于显示，不改变保存值。按空格冻结/恢复，Enter 保存当前冻结帧并退出，`q` / Esc 取消。相机在后台读取，冻结后原图、深度、内参和来源时钟始终来自同一帧。退出等待读取线程释放资源，`result.json` 明确报告保存和资源回收结果。输出包括 `rgb.png`、`frame.npz`、`capture.json`；后续优先使用 `frame.npz`，保留深度和内参。

```bash
python -m cordvip_geometry object segment \
  --rgbd reports/cordvip_demo/object_capture/frame.npz --text 'red mug.' \
  --output-dir reports/cordvip_demo/mug_segmentation
```

拍照入口可通过 `--rgbd-dir /path/to/numbered_npz --replay-fps 10` 使用保存的 RGB-D 测试窗口、冻结和保存，不连接相机；回放结束后保留最后一帧供确认。文件采用后文定义的六位编号 NPZ 格式。`capture --help` 不加载相机 SDK 或连接设备。

使用已有照片时替换为：

```bash
python -m cordvip_geometry object segment \
  --image /path/to/object.jpg --text 'red mug.' \
  --output-dir reports/cordvip_demo/mug_segmentation
```

窗口中 `n` / `p` 切换候选、Enter 确认、Esc 取消；不会自动把最高分检测当作你要追踪的实例。保存完整分辨率的 `rgb.png`、`mask.png`、`mask.npy`、`foreground_rgba.png`、`overlay.png`，以及输入、模型权重和候选的追溯信息。`--image` 可替换为 `--rgbd /path/to/000000.npz`；该模式只读取已保存帧，保留深度和标定，不连接相机、不猜测真实尺寸。

确认分割后生成 mesh：

```bash
python -m cordvip_geometry object generate \
  --segmentation-dir reports/cordvip_demo/mug_segmentation \
  --output-dir reports/cordvip_demo/mug_generation --preview
```

这是 **Hunyuan3D-2.1 纯几何生成**，不运行纹理生成、不生成 PBR 材质。默认 50 步、octree resolution 384，使用已有本地权重，不在推理时下载。该分阶段入口同样输出未定尺度的 mesh、点云和法向，支持 `--num-points`。Open3D 显示 mesh、采样点、包围盒与坐标轴，终端打印三个轴的尺寸；红色为 x、绿色为 y、蓝色为 z。单图的隐藏面是预测结果，需检查把手、孔口、厚度和接触面。

生成 mesh 尚无米制尺度。识别与实测尺寸对应的 mesh 轴后准备资产，例如 **仅当 mesh z 轴确实对应实测 12 cm 高度**。以下使用分阶段生成的路径；若使用第一个示例入口，替换为 `reports/cordvip_demo/mug_build/generation/mesh_unscaled.obj`：

```bash
python -m cordvip_geometry prepare \
  --mesh reports/cordvip_demo/mug_generation/mesh_unscaled.obj \
  --size-axis z --size-meters 0.12 \
  --output-dir reports/cordvip_demo/object_asset --preview
```

0.12 是示例，必须替换为真实测量。单次统一缩放不修复生成比例错误；检查其他方向的尺寸。当前不自动使用可见深度包围盒推断完整物体尺寸，也不在生成阶段声称已完成 RGB-D 位姿对齐。模型与实测深度的叠加检查在 FoundationPose 注册后进行。

熟悉坐标轴和物体尺寸后，可以串联这三个阶段：

```bash
python examples/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'red mug.' \
  --size-axis z --size-meters 0.12 \
  --output-dir reports/cordvip_demo/mug_build --preview
```

示例分别启动分割和生成子进程，阶段结束后释放 GPU。提供尺寸时，额外在 `asset/` 保存米制 `mesh_m.obj`、`points.npy`、`normals.npy` 和 `asset.json`，可直接交给追踪入口。省略尺寸时保存未定尺度 mesh 与点云并打印后续校准命令，不自动启动相机。每个输出目录必须位于 `reports/` 下且尚不存在；失败时保留已完成阶段，可用 `object select`、`object generate` 或 `prepare` 继续，无需重跑已有结果。等价库命令为 `python -m cordvip_geometry object build ...`。

无桌面环境先检查候选图，再明确选择：

```bash
python -m cordvip_geometry object segment \
  --image /path/to/object.jpg --text 'red mug.' --list-only \
  --output-dir reports/cordvip_demo/mug_candidates

# 查看 mug_candidates/candidates.jpg 后，将 0 换成目标对应的候选编号。
python -m cordvip_geometry object select \
  --segmentation-dir reports/cordvip_demo/mug_candidates \
  --headless --instance 0 \
  --output-dir reports/cordvip_demo/mug_selected
```

`object select` 不加载 GPU 模型。若已经明确候选编号，`object segment` / `reconstruct_object.py` 可直接加 `--headless --instance N`，不要使用 `--preview`；`generate` 不带 `--preview` 时也不导入图形窗口。候选编号按检测分数排序，从 0 开始，仅对本次保存的候选有效。

默认模型目录：

```text
reports/cordvip_models/
  grounding-dino-base/             # Hugging Face Grounding DINO 本地模型和 processor
  sam2.1_hiera_large.pt
  Hunyuan3D-2.1/hunyuan3d-dit-v2-1/
```

源码默认位于 `third_party/Grounded-SAM-2` 与 `third_party/Hunyuan3D-2.1`，可用 `--sam-root`、`--hunyuan-root` 指定；模型路径可用 `--dino-weights`、`--sam-checkpoint`、`--weights` 指定。Hunyuan 参数只属于生成工作流（`reconstruct_object.py` / `object generate` / `object build`）。`--help` 不导入 Torch、模型或相机 SDK。

已有米制 mesh 时直接跳过分割与生成：

```bash
python -m cordvip_geometry prepare \
  --mesh /path/to/object_in_meters.obj \
  --scale-to-meters 1 \
  --output-dir reports/cordvip_demo/object_asset --preview
```

输入单位是 mm 时使用 `--scale-to-meters 0.001`。未知尺度时可用 `--size-axis x --size-meters 0.08`，其中 0.08 必须替换为与 mesh x 轴对应的实测尺寸。统一缩放不能修正形状比例，追踪前核对其他方向尺寸和关键表面。

输出 `mesh_m.obj`、固定的 `points.npy`、`normals.npy`、`asset.json`。默认点数 1024，在线直接读取点表，不逐帧重新采样，也不改变 mesh 原点和朝向。准备步骤需要 NumPy、trimesh；`--preview` 额外需要 Open3D。

### 2. 检查追踪依赖

在已激活的 `real_robot` 中执行；FoundationPose 源码和扩展要求见[官方说明](https://github.com/NVlabs/FoundationPose)：

```bash
python examples/cordvip/track_object.py --check \
  --foundationpose-root /path/to/FoundationPose
```

实时入口的 `--check` 不需要 `--asset` 或图形会话，不连接或枚举相机，不导入 CUDA 模型，不创建窗口。GUI 启动才检查图形会话。未传源码目录时只使用仓库的 `third_party/FoundationPose`，不自动搜索历史调研目录。

需要 JSON 或无桌面环境检查时，可单独运行：

```bash
python -m cordvip_geometry check \
  --foundationpose-root /path/to/FoundationPose --headless --json
```

`check --offline-input` 可跳过 RealSense SDK 检查。独立检查脚本默认检查 `DISPLAY`/`WAYLAND_DISPLAY`，加 `--headless` 才跳过；这一点与实时入口的 `--check` 不同。静态检查汇总源码、非空权重、模块及 `mycpp` 扩展缺项，通过不证明 CUDA 可用、二进制 ABI 兼容或模型推理成功。

上游目录需要：

```text
FoundationPose/
  estimater.py
  Utils.py
  datareader.py
  learning/
  mycpp/build/mycpp*.so
  weights/2023-10-28-18-33-37/{config.yml,model_best.pth}
  weights/2024-01-11-20-02-45/{config.yml,model_best.pth}
```

推理入口不自动安装依赖或下载权重；环境是否完整以本次 `--check` 输出为准。静态检查不能替代模型推理和设备验收。

### 3. 启动实时追踪

关闭其他占用同一相机的程序，在桌面会话执行：

```bash
python examples/cordvip/track_object.py \
  --asset reports/cordvip_demo/object_asset \
  --foundationpose-root /path/to/FoundationPose \
  --object-prompt 'red mug.' \
  --serial CAMERA_SERIAL \
  --save-inputs \
  --output reports/cordvip_demo/live_run
```

执行此命令会连接并配置 RealSense；这不是静态检查。多相机时明确指定序列号。输出目录必须是 `reports/` 下的新目录。`--no-3d` 可关闭三维窗口，但仍使用 RGB GUI，几何投影和 FoundationPose 仍需要 Open3D；`--headless` 仅支持带已确认首帧选择的离线源。Raw/RealSense 输入还需要当前 workspace 中的 `dexmani_real` 宿主可被 Python 导入；NPZ 回放不依赖宿主。当前 RealSense 示例不读取机器人关节，`--with-hand` 只支持 `--episode`。

若本机 Open3D 创建窗口时报 GLX 错误，可在该次启动命令前显式加 `__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=true`，选择 Mesa 软件渲染。本机已用这一设置完成离线窗口检查；示例不自动切换渲染器。

建模入口使用 `--text 'red mug.'`，实时入口使用 `--object-prompt 'red mug.'`，两处描述应对应同一物体。提供 `--object-prompt` 后，按 `i` 或 `r` 冻结当前帧并在后台进行文本分割；选中具体候选后 Enter 注册。`[` / `]`、数字键或鼠标选择候选，`p` 可手动画 mask。没有文本参数时直接进入多边形圈选：左键添加点、右键撤销、Enter 注册。圈选和首次注册期间保持物体静止，状态进入有效追踪后再移动。选择 mask 至少包含 16 个像素，且至少有 4 个有效深度像素；这只是最低输入条件，不保证注册准确。

## 界面与交互

| 区域 | 含义 |
|---|---|
| 左侧 RGB | 最新相机图像；圈选时冻结为初始化源帧 |
| 右侧 RGB | 最近完成推理的源图像及对应模型点投影、坐标轴 |
| 3D 绿色点 | 源 RGB-D 中靠近预测 mesh 前表面的实测目标点 |
| 3D 青色点 | 固定物体点经 `T_camera_object` 变换后的模型点云 |
| 3D 橙色点 | `--with-hand` 的固定 URDF 表面点，经同 Raw 行实测关节 FK 变换 |
| 状态栏 | 当前状态、源帧、单次推理耗时、来源年龄和几何检查结果 |

右侧结果可能比左侧图像旧，位姿不会悄悄叠加到其他帧。失效时界面保留的历史模型以灰色显示，仅供检查。绿色目标点来自几何筛选，不是独立语义分割真值；青色点包含不可见表面，是模型预测，不是新增实测观测。橙色手点来自机器人几何和实测关节，不是视觉重建或接触测量。

将键盘焦点放在 RGB 窗口：

| 按键 | 操作 |
|---|---|
| `i` / `r` | 冻结当前帧，选择物体并注册/重新注册 |
| `[` / `]`、数字键、鼠标 | 文本分割后选择具体候选 |
| `p` | 文本分割期间切换手动多边形选择 |
| 左键 / 右键 | 选择期间添加/撤销多边形顶点 |
| Enter | 用所选帧和 mask 注册 |
| 空格 | 暂停/恢复追踪；暂停期间输出失效 |
| `v` | 3D 叠加 / 实测目标点 / 模型点图层 |
| `s` | 保存界面/三维截图及当下位姿、点云 NPZ 快照 |
| Esc | 圈选时取消并回到预览；其他状态退出 |
| `q` | 退出 |

GPU 使用单个 tracker 串行推理，后台采集只保留最新 RGB-D，忙时不积压历史输入。相机 FPS、实际推理 FPS 和显示 FPS 不同。重新注册后，旧调用稍后返回的结果不会成为新一轮有效输出。

离线回放启动时停在首帧，首次注册通过后才按 `--replay-fps` 继续，源结束后自动退出。这样首次分割/注册不会消耗短片段；回放期间仍采用最新帧策略，不能保证处理每个源帧。

退出时先进入 `STOPPING` 并使快照失效，窗口继续响应，等待相机/推理线程正常结束，最多 60 秒。此时再次按 `q` 可结束等待；若资源尚未释放，程序以非零退出码和 `result.json` 中的失败状态报告，不能视作正常完成。

## 有效性与获取点云

检查包括 SE(3) 数值、可见表面深度一致性、匹配覆盖和结果来源年龄。当前参数为：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--max-age-ms` | 500 | 结果的最大来源年龄 |
| `--depth-tolerance-m` | 0.015 | 实测深度与模型前表面的匹配容差，m |
| `--min-matched-pixels` | 30 | 最少匹配像素数 |
| `--min-matched-fraction` | 0.15 | 深度一致采样点占模型投影采样点的最小比例 |
| `--pixel-stride` | 2 | 几何检查/目标点提取的像素步长 |

这些是尚需真实物体验证的示例阈值，不是 FoundationPose 概率置信度，也不能保证检测所有漂移。几何不一致导致丢失时按 `r` 重新注册；不自动外推、保持旧位姿或重初始化。来源过期不会被当作新观测发布。

`latest.npz` 是当前结果快照，每次结果和状态变化都原子更新，可用 `numpy.load(..., allow_pickle=False)` 读取：

| 字段 | 语义 |
|---|---|
| `valid`、`state` | 当前是否可用及状态，读取方必须先检查 |
| `source_id`、`timestamp_ns`、`clock` | 生成结果的 RGB-D 身份及来源时间语义 |
| `completed_ns`、`published_ns` | 本机 monotonic 的推理完成和快照发布时间 |
| `max_age_ms` | 当前运行配置的最大来源年龄 |
| `T_camera_object` | 物体到 color camera 的 4×4 变换 |
| `model_points_camera` | 完整模型点云，`N×3`，m |
| `model_normals_camera` | 与固定模型点一一对应的 mesh-face 单位法向 |
| `observed_points_camera`、`observed_colors` | 几何检查选中的实测目标点及颜色 |
| `interaction_valid` | 手与物体几何是否可组成同来源观测，不代表时序准入 |
| `T_base_camera`、`T_base_object` | 相机/物体到 robot base 的变换，需 `--with-hand` |
| `object_points_base`、`hand_points_base` | robot-base 下的完整物体模型点与手模型点，m |
| `object_normals_base`、`hand_normals_base` | 对应模型表面的单位法向 |
| `hand_points_camera`、`hand_normals_camera`、`hand_link_ids` | 相机系手点、法向及稳定的 link IDs |
| `joint_state` | 实测臂 7 关节 + 手 SDK 顺序 12 关节，弧度 |
| `arm_timestamp_ns`、`hand_timestamp_ns`、`camera_timestamp_ns` | Raw 源时间证据，缺失为 -1，不以行时间替代 |
| `hand_timing_available`、`hand_camera_skew_ms` | 同时钟域下三源最大时间差；未知为 False / NaN |
| `K` | 此结果使用的去畸变针孔内参 |
| `diagnostics_json` | 可用 JSON 解析的几何诊断量 |

失效快照的 pose 为 NaN，模型和实测点云为空，保留来源和诊断信息；不会用单位矩阵或旧点云冒充有效输出。界面可保留灰色历史模型供人工检查，与可消费快照区分。

`valid=True` 只描述发布时刻。实时读取方还必须按本机 `time.monotonic_ns()` 检查来源年龄是否超过 `max_age_ms`，并确认生产进程仍在运行。`published_ns` 较新不代表相机帧新鲜。离线回放状态标为 `REPLAY`，不做实时来源年龄准入；其历史 clock 不可与当前本机时间直接相减，也不可作为实时物体结果消费。

这是可检查的示例输出，**不是可直接用于机器人闭环的可靠实时 IPC**。进程被强制结束时，磁盘最后一个文件可能来不及失效；不能永久相信文件存在或其中的布尔值。

`interaction_valid=True` 仅表示手—物几何来自同一行和声明的同一来源。若 Raw 缺少真实源时钟，仍可用于几何回放，但 `hand_timing_available=False`、`hand_camera_skew_ms=NaN`。即使有时钟，消费方也必须按实验要求检查时间差和观测年龄，不能把几何有效性当成已同步或可执行的策略观测。

每次运行还保存 `run.json`、`records.jsonl`、`result.json`，关联配置、来源、处理结果和结束原因。启用手点时记录 URDF/STL 哈希、采样、关节顺序、相机外参与当前安装变换来源。每次注册的源 RGB-D 与去畸变 mask 必存；`--save-inputs` 额外保存所有真正提交推理的 RGB-D，不是整个相机视频，不写回 Raw。

`--save-frames` 把实际处理的结果保存到 `frames/`，不补齐跳过的输入。`--save-video` 保存处理结果叠加视频，播放率为 `--replay-fps`；视频是检查产物，时间依据仍是对应结果的源时钟。两者默认关闭，避免不必要的逐帧归档开销；内存接口无需经磁盘读取结果。

按 `s` 保存的 `snapshot_<monotonic_ns>.npz` 与 `latest.npz` 字段相同，可单独检查或提取目标点云。它保留保存时的有效性和来源时间，不会随之后的状态变化更新；读取时仍须检查时间语义。

## 离线辅助检查

### 离线手—物回放

复用本机已有 toy 资产与首帧选择：

```bash
python examples/cordvip/track_object.py \
  --episode episodes/pick_place_toy/episode_20260827_165923 \
  --asset reports/cordvip_toy/asset \
  --initial-selection reports/cordvip_toy/first_frame_selected \
  --with-hand --hand-points 1024 --replay-fps 16 \
  --save-frames --save-video \
  --output reports/cordvip_demo/toy_interaction
```

加 `--headless` 即可无窗口运行；再次运行需指定新的输出目录。已有 `--initial-selection` 时不必指定 `--object-prompt`，也不必重新加载分割模型。它必须包含人工确认的 `selection.json`、原始 `source_rgbd.npz` 与匹配的 mask；程序核对 RGB、深度、内参、畸变、源身份和时钟，拒绝复用其他帧或其他 episode 的 mask。

`--with-hand` 需要同 Raw 行的 `arm_qpos` / `hand_qpos` 及 Raw 相机外参。手安装默认取当前 `ExperimentConfig`，有实验 YAML 时加 `--experiment-config /path/to/experiment.yaml`。缺失或非有限关节不能用 action、零值或上一帧替代。此 toy 为历史迁移数据，其名义 16 Hz 行时间无法证明手/相机同步；输出明确保留该限制。

### 其他离线输入

实时入口可用保存的输入回放 GUI 和追踪流程：

```bash
python examples/cordvip/track_object.py \
  --rgbd-dir /path/to/numbered_rgbd_npz --replay-fps 30 \
  --asset reports/cordvip_demo/object_asset \
  --foundationpose-root /path/to/FoundationPose \
  --output reports/cordvip_demo/replay_run
```

也可将 `--rgbd-dir ...` 替换为 `--episode /path/to/raw_episode`。首帧等待圈选与注册完成，然后按指定速率供给最新帧，结束自动退出。推理较慢时会跳过输入，不是逐帧离线评测；`--replay-fps` 不修改源时间。要处理每一帧，使用下面的顺序离线入口。

没有模型环境时，可保留原来的 RGB-D 离线预览作为输入检查：

```bash
python -m cordvip_geometry replay \
  --episode /path/to/raw_episode --headless --max-frames 10 \
  --save-inputs --output reports/cordvip_demo/observed_run
```

它默认预览实测场景点云，不产生物体位姿。NPZ 用 `000000.npz` 等六位编号，包含 `rgb`、米制 `depth_m`、`K`、`distortion_model`、五项 `distortion_coeffs`、`source_id`、`timestamp_ns`、`clock`，可带 `row_index`；禁止 pickle，不猜测缺失单位和内参。

顺序离线追踪可显式启用 FoundationPose，不跳过后续帧：

```bash
python -m cordvip_geometry replay \
  --episode /path/to/raw_episode --mode track \
  --asset reports/cordvip_demo/object_asset \
  --foundationpose-root /path/to/FoundationPose \
  --save-inputs --output reports/cordvip_demo/offline_run
```

没有 `--mask` 时在首帧圈选；headless 追踪须显式提供对应首帧 mask。该辅助工具同步逐帧运行，推理期间 UI 等待，不是实时主入口。

RGB、aligned depth 和 mask 使用同一去畸变网格。深度 scale 和内参来自相机或对应 Raw；不支持的畸变模型明确报错。Raw RGB 来自保存的视频解码，另存无损 NPZ 不会恢复压缩前像素。

## 手—物模态与 Python 接口

默认手 1024 点 + 物体 1024 点、XArm7 + XHand 的 19 维关节是本地表示，不能未经训练适配直接送入作者的模型检查点。手点在初始化时按 link 表面积采样，保留固定样本与 link IDs；每帧只做 FK，区别于逐帧 FPS。物体点保持资产中的固定样本顺序，便于后续处理时序对应。可变数量的实测可见点另存，不混入完整模型点。

库不在线生成 contact/SDF 标签，也不将几何近邻当作触觉真值。这里的法向来自 mesh faces，经旋转变换；不等同于 CordViP contact 计算的 KNN 法向。CordViP/ProxiDex 的点数、采样、归一化及 proximity 构造应由 `dexmani_policy` 明确实现。

手点核心独立于宿主，初始化一次后重复调用。以下变量由调用方提供；关节顺序、弧度和安装变换必须明确：

```python
from cordvip_geometry.hand_cloud import HandCloudModel

hand_model = HandCloudModel(
    hand_urdf_path, sdk_joint_names,
    excluded_links=marker_link_names, num_points=1024, seed=0,
)
# T_camera_handbase 映射 hand base 到 aligned color-camera frame，单位 m。
hand_geometry = hand_model.transform(measured_hand_qpos, T_camera_handbase)
```

在当前 Raw 路径里，`RawHandCloudProvider` 补齐关节、外参和同一行的时钟证据。下面接续已有 `PoseResult result`、投影结果 `cloud` 以及资产 `points_local / normals_local`，直接生成内存观测；无需写入再读取 NPZ：

```python
from cordvip_geometry.geometry import transform_points
from cordvip_geometry.integrations.dexmani import RawHandCloudProvider
from cordvip_geometry.observation import build_observation

hand_provider = RawHandCloudProvider(episode_path, config_path=experiment_yaml)
# 以下在每次实际处理完 RGB-D 后调用；object_valid 来自当前结果的有效性检查。
packet = build_observation(
    state="REPLAY", valid=object_valid, result=result, cloud=cloud,
    model_points=transform_points(points_local, result.pose) if object_valid else None,
    model_normals=normals_local @ result.pose[:3, :3].T if object_valid else None,
    hand=hand_provider.for_frame(result.frame) if object_valid else None,
    max_age_ms=500,
)
```

`max_age_ms` 是观测元数据；`build_observation` 不自行判断所有传感器的新鲜度，调用方负责准入。返回数组供立即消费，异步保留者应复制。之后遥操作/策略应复用已有 RGB-D、实测关节与源时钟，将手几何和这些证据交给同一构造函数，不再启动独立机器人/相机采集。**当前只实现 Raw 手适配，未接入 live hand 或主包 runtime。** 机器人运动、策略执行与安全边界仍由宿主负责，离线 `REPLAY` 观测不能授权运动。

## CordViP 原版对照：TripoSR

默认单图路线见上面的 Grounded-SAM-2 + Hunyuan3D-2.1。保留 TripoSR 作为 CordViP 原版对照，使用已有 [TripoSR 环境](https://github.com/VAST-AI-Research/TripoSR)。输入须为已去背景、合成灰底并裁剪好的 RGB 图像；不能直接将透明 RGBA 传给这个灰底 RGB 入口。本地权重含 `config.yaml` 和 `model.ckpt`。

```bash
python -m cordvip_geometry.cli.prepare_object reconstruct \
  --triposr-root /path/to/TripoSR \
  --weights /path/to/local_triposr_weights \
  --image /path/to/foreground_rgb.png \
  --python /path/to/triposr_env/bin/python \
  --output-dir reports/cordvip_demo/reconstruction

python -m cordvip_geometry prepare \
  --mesh reports/cordvip_demo/reconstruction/0/mesh.obj \
  --size-axis x --size-meters 0.08 \
  --output-dir reports/cordvip_demo/object_asset --preview
```

0.08 仅是 8 cm 示例，必须替换为物体实测尺寸。重建默认 `cuda:0`，只有显式 `--device cpu` 才允许 CPU。上游自动去背景及在线下载被关闭。单图不可见面的几何属于预测，需要检查后再追踪。

## 尚需真实设备验收

离线检查覆盖候选选择、完整分辨率 alpha、文件校验、续建、米制采样、取消和退出清理。玩具照片已完成真实 Hunyuan3D 生成，对应 Raw 的 213 帧已完成 FoundationPose 位姿和点云输出，真实 GUI 异步回放也全部处理并正常退出；没有物理位姿真值，几何检查通过不能证明追踪精度。后续仍需在真实相机上检查首次注册、平移/旋转、部分遮挡、出画、重新注册和退出释放，并报告耗时、来源年龄、有效率和点云重合情况。没有执行实时相机连接；离线检查不能代替真实设备验收。
