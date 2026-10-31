# Raw 离线几何与物体资产

主流程固定为首帧人工确认、顺序处理 Raw 每行、保存结果、独立查看。命令从 workspace 根目录执行；脚本入口以 dexmani_real 根目录为 workspace，模块入口默认当前目录。输出必须使用新目录，不覆盖输入或写回 Raw。

## 安装与权重

`real_robot` 只是示例环境名。先准备兼容的 Torch、Torchvision、NumPy、Transformers、CUDA 编译器及头文件，再运行安装脚本；脚本约束已有基础版本，补齐增量依赖及 nvdiffrast / FoundationPose mycpp，不替换已有源码和本地修改。

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh --help
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam
python -m cordvip_geometry check --headless
```

环境清单和版本查询只在显式 `check` 或追踪的 `--check` 中执行。正常追踪直接加载所需源码、权重并执行边界检查；不默认同步 GPU 生成逐步耗时、显存或环境报告。算法使用的深度支持量和 pose-valid 门控仍逐行保存。

安装脚本为缺失源码选择以下 revision；已有目录直接复用：

| 上游 | revision |
| --- | --- |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28` |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | `a1b694b83e633c2cb6115b9063d940a687759392` |
| [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | `82920d643c0dc2f7bfd7255f45f62d386edfe60c` |
| [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | `253ac4fcea7de5f396371124af597e6cc957bfae` |

FoundationPose 的官方权重需按其[官方说明](https://github.com/NVlabs/FoundationPose)另行取得，放入 `third_party/FoundationPose/weights/2023-10-28-18-33-37` 和 `2024-01-11-20-02-45`；每个目录包含 `config.yml` 与 `model_best.pth`。源码、权重与再分发分别遵循上游仓库和模型来源的许可证；本仓不变更其授权。

下载命令从 [sam2_config.py](../src/cordvip_geometry/sam2_config.py) 和 [download_models.py](../src/cordvip_geometry/cli/download_models.py) 的明确官方 revision 获取所选模型，并保存下载来源。默认位置是 `reports/cordvip_models/`。运行只加载本地权重，不隐式联网下载。安装脚本保留 `TORCH_CUDA_ARCH_LIST`；无可见 GPU 的离线编译需按最终设备显式设置。

SAM 默认 small、BF16、eager；使用 `--sam-variant large` 时自动配对 large 的配置与本地权重：

```bash
python -m cordvip_geometry download --models sam --sam-variant large
# 在 track 或 object segment/build 命令中追加 --sam-variant large
```

自训模型必须同时给出 `--sam-config` 与 `--sam-checkpoint`，并省略 `--sam-variant`。`--sam-root`、`--dino-weights` 可指定本地路径，实际源码 API、配置和 checkpoint 结构必须匹配。`--sam-python` 默认当前解释器，可指向独立环境；SAM RPC 与独立取消仍保留。分割和位姿使用各进程可见的第 0 张 CUDA 卡；建模的 `--device` 单独选择 Hunyuan 设备。

Grounding DINO 仅用于首帧文本候选，确认后释放；复用 mask 或手动画多边形不要求检测权重。SAM2 按模型所需历史窗口清理长序列状态，仍保留首帧 conditioning；初始 mask 会重采样，不承诺首帧传播结果逐像素相同。mask-only 不要求 FoundationPose；带 asset 的位姿分支仍需要其 CUDA/Open3D 环境。FK 手点需 Real 的离线运动学依赖，3D 结果查看另需 Open3D。help、普通 import 和静态检查不加载模型或连接设备；静态检查不证明实际推理、ABI、显存或吞吐。

## 首帧确认与逐行追踪

单物体使用米制资产及文本选择：

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 \
  --asset outputs/object_asset --object-prompt 'white basket.' \
  --with-hand --output outputs/scene_001
```

Raw 首帧被冻结，用户确认候选，或省略 prompt 手画 mask。选择保存到 `initial_selections/`；已有选择通过 `--initial-selection DIR` 复用。复用仍核对原始 RGB、深度、K、畸变、来源时间、时钟及 Raw 行身份，不能用于另一段录像。Raw 原样搬迁目录时，内容与行身份核对仍须成立。

多物体由 `scene.json` 声明，路径相对 workspace：

```json
{"objects": [
  {"id": "toy", "role": "object", "asset": "outputs/toy_asset",
   "initial_selection": "outputs/toy_first_frame"},
  {"id": "basket", "role": "reference", "asset": "outputs/basket_asset",
   "initial_selection": "outputs/basket_first_frame"}
]}
```

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pick_place/episode_001 --scene scene.json \
  --with-hand --headless --output outputs/scene_002
```

`role` 仅是 metadata 标签。省略某个 `asset` 就保留该实例的 mask 和语义实测点，不估计或伪造 pose；全部省略时不需要 FoundationPose。新场景可用 `prompt` 逐个确认，也可手画。无窗口运行要求每个实例都有当前首帧的确认选择，不自动接受最高分候选。单物体同样支持 mask-only：省略 `--asset` 即可。

默认逐行处理全部 Raw，不按墙钟等待或跳帧。`--max-frames N` 明确限制前缀，最终结果标为 partial；没有全帧、存盘或实时源的第二条运行路径。一个 RawReader 同行提供 RGB-D、实测关节和来源时钟。首帧深度支持不足时明确失败；pose 丢失按对象保持失效，重新注册需要重新运行并确认首帧。

## 输出与查看

| 输出 | 内容 |
| --- | --- |
| `scene.json` | 一次保存对象 ID、role、asset、确认选择与标签 |
| `run.json` | 实际参数、资产与相机/FK 来源、单位及数据语义 |
| `frames/000000.npz` 等 | 每个已处理 Raw row 的 mask、observed/model 点、pose-valid、法向与可选手点 |
| `records.jsonl` | 逐行源 ID、row、时钟、状态与算法使用的几何诊断 |
| `models.json` | 实际模型来源、配置及加载事实 |
| `result.json` | completed / partial / interrupted / failed、已处理行数及清理错误 |
| `sam2_worker.log` | 隔离分割进程的原始日志 |

SAM 失败保留子进程原始 traceback，父进程错误指向该日志。单物体 FoundationPose 失败进入 `POSE_LOST`，原始 traceback 同时输出到终端，其他独立实测点仍保留。

逐行结果必存，文件名对应源 row，不是墙钟序号。对象字段使用 `toy__`、`basket__` 或单物体 `object__` 前缀，对象清单与 role 只存于 `scene.json`。每个实例分别保存 `pose_valid` 和 `observed_valid`；pose 无效时变换为 NaN、模型点为空，独立实测点仍可有效。

```bash
python -m cordvip_geometry view outputs/scene_001
python -m cordvip_geometry view outputs/scene_001 --no-3d
python -m cordvip_geometry view outputs/scene_001 --headless
```

查看器只按 `n`/Space 前进一行、`v` 切换 observed/model、`q` 退出；`--headless` 读取检查结果而不开窗口。查看不重新执行模型。追踪退出后保留已写行及明确最终状态，不再生成 latest、心跳、STALE 或退出失效快照。旧 notebook 应直接读取 `scene.json` 和逐行 NPZ；没有实时相机、通用外部 Frame 宿主或 Raw/NPZ 多源播放器。

## 准备物体资产

```bash
python -m cordvip_geometry download --models shape
python scripts/cordvip/reconstruct_object.py \
  --image /path/to/object.jpg --text 'white basket.' \
  --output-dir outputs/object_build --preview
```

分割需要人工确认，生成的 `foreground_rgba.png` 才是 Hunyuan 输入。分割和 prepare 是普通函数调用，Hunyuan 生成单独运行一次子进程，退出释放 GPU；SAM 仍有自己的隔离子进程。生成只输出未定尺度 mesh 与来源记录，不再落盘未定尺度点云。`--preview` 临时采样，不影响最终资产。

检查模型轴后，若 z 轴确实对应实测 12 cm 高度：

```bash
python -m cordvip_geometry prepare \
  --mesh outputs/object_build/generation/mesh_unscaled.obj \
  --size-axis z --size-meters 0.12 --num-points 1024 --seed 0 \
  --output-dir outputs/object_asset --preview
```

也可显式提供 `--scale-to-meters`；已有米制 mesh 使用 1。最终资产包含 `mesh_m.obj`、`points.npy`、`normals.npy` 与 `asset.json`。缩放作用于原点偏移，保持对象朝向，不自动居中；按原 seed 和点数在最终米制 mesh 上只采样一次。已有 CAD/扫描 mesh 可直接 prepare。

完整流程 `object build` 与 `reconstruct_object.py` 共用实现。分阶段可用 `object segment --list-only` 保存候选、`object select` 确认、`object generate` 生成；无窗口选择必须指定 `--instance`。score 是 Grounding DINO 检测分数，mask_score 是 SAM2 预测 IoU，均非人工真值。Hunyuan 使用本地 2.1 权重，不隐式执行 rembg、清理、居中或尺度估计。

## 数据语义

几何使用 color-camera frame 和米。深度是 camera Z，0 表示缺测；ray direction 保持 Z=1，因此命中参数也是 Z，不归一化成欧氏距离。去畸变只接受已支持的 forward Brown-Conrady 模型，depth/mask 使用最近邻。源 ID、row、clock 和来源时间保持原值，主机接收时间不等于曝光时间。

语义实测点只来自同帧互斥 mask 与实测深度，不修补孔洞、不分配重叠像素，也不依赖 mesh。像素按全局 stride 网格的 row-major `(v,u)` 排列。模型支持与残差复用同一点序，残差为 measured Z − rendered Z，无模型表面处为 NaN。默认 stride 为 2、深度容差 0.015 m、最少匹配 30 像素及支持比例 0.15；这些支持条件参与 pose-valid 和 `POSE_LOST`，不能写成仅诊断。coverage/support fraction 不是概率置信度，未定义时记录 None。

模型点来自固定资产及有效 pose，不是实测接触。FoundationPose++ 的 bbox 中心只引导 centered mesh 的 2D 位置，保留旧 Z/rotation，不用深度孔洞构造 3D 质心；输出为 `T_camera_asset = T_camera_centered @ T_centered_asset`。单个 pose 失败不删除别的物体或独立实测点。

`--with-hand` 用同一 Raw 行的 19 维实测 arm/hand 关节（rad）、数据对应的相机外参与 URDF `link_base` 生成手点，不用 action、不插值、不连接设备。固定样本、link ID 与法向沿用 URDF/mesh，保留 camera/base 两坐标系；时间证据未知时明确标为不可用。同行关联不证明手/相机物理同步，几何有效也不代表通过策略输入的时序准入。

控制、训练适配及 contact/proximity 标签仍属于主包和配套 Policy。本工具不接管设备或动作发布；真实 GPU 模型的结果、长序列内存与性能须用实际模型和 Raw 单独验证。
