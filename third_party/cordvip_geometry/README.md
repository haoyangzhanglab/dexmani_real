# cordvip_geometry

独立手—物几何库：准备米制物体资产、RGB-D 点云、连续多实例分割、FoundationPose 位姿追踪、实测关节 FK 手点和可视化。重建是可选的离线步骤，不进入追踪循环；多物体模式无资产也能获取语义实测点。

`examples/cordvip` 保留两个薄入口：`reconstruct_object.py` 从 RGB 图像生成 mesh 与固定点云，`track_object.py` 从 RGB-D 流获取物体位姿、实测点和模型点，并可在 Raw 回放中附加 XHand 点。建议先读[示例快速开始](../../examples/cordvip/README.md)，再看下面的库边界与[完整使用说明](docs/usage.md)。仓库入口显式传入当前仓库作为 workspace，库不通过自身安装路径猜测仓库位置，不导入 examples。

## 代码边界

```text
src/cordvip_geometry/
  __main__.py                       统一命令入口
  geometry.py, object_cloud.py       几何变换、固定采样、实测点筛选
  hand_cloud.py, observation.py      固定 URDF 手点、内存手—物观测与可选落盘
  segmentation.py, reconstruction.py, foundationpose.py
                                    独立模型适配
  rgbd.py, io.py                     RGB-D 帧、NPZ 回放、资产与输出文件
  live_workers.py, segmentation_worker.py
                                    最新帧读取、串行位姿推理、分割子进程
  viewer.py, live_viewer.py           二维/三维可视化
  integrations/dexmani.py            可选 Raw / RealSense 输入、Raw 实测关节适配
  cli/                              拍照、建模、追踪、下载与检查工作流
scripts/install_perception.sh       固定版本安装与扩展编译
requirements-perception.txt         real_robot 环境的增量依赖
docs/usage.md                       安装、操作、可视化与输出说明
```

几何、手点 FK、内存观测、NPZ 回放和模型适配不依赖 `dexmani_real`。只有显式使用宿主适配时，`integrations/dexmani.py` 才导入对应模块。手点核心使用 NumPy、trimesh 与 Pinocchio，初始化时固定采样，每帧只做 FK 与变换。Raw 适配读取实测关节，使用录制相机外参与当前实验配置的手安装变换，并保存来源；不将名义安装 URDF 直接当作真实安装标定。

后续遥操作/策略调用应传入其已有 RGB-D、实测关节与来源时钟，复用库的 `HandCloudModel.transform` 和 `build_observation`，不另开机器人或相机连接。目前 `--with-hand` 仅适配 Raw，未接入主包 runtime、teleop 或 deployment；实时 RealSense 路径仍只提供物体点。

## 调用

在仓库根目录使用薄入口：

```bash
conda activate real_robot
python examples/cordvip/reconstruct_object.py --help
python examples/cordvip/track_object.py --check --object-prompt 'white basket.'
```

已安装本库时也可以从 workspace 根目录直接调用模块：

```bash
python -m cordvip_geometry check --headless --offline-input
python -m cordvip_geometry object --help
```

库的辅助子命令为 `capture`、`object`、`prepare`、`track`、`replay`、`check`、`download`。`object` 包含 `segment/select/generate/build/preview`；两个示例分别直接接收 `object build` 和 `track` 的参数。`prepare` 直接接收 mesh 和尺度参数，无需重复子命令。TripoSR 对照保留为 `python -m cordvip_geometry.cli.prepare_object reconstruct`，不加入默认操作流程。

生成阶段即使没有尺寸，也输出未定尺度的 mesh、`points_unscaled.npy` 和法向；默认 1024 点，可用 `--num-points` 调整。指定实测尺寸或单位换算后再输出米制资产。FoundationPose 使用米制资产、RGB-D 和内参；多物体模式的语义实测点仅需 RGB-D 和当前 mask。完整模型点和当前实测点独立保存，不做跨帧表面融合。

Python 调用 CLI 时可显式传入 `main(argv, workspace=Path(...))`；模块入口默认以当前工作目录为 workspace。默认模型路径为 workspace 下的 `third_party/` 和 `reports/cordvip_models/`，产物仍只能写入其 `reports/` 新目录。用户给出的相对输入路径保持相对于调用时的当前目录；工作流启动子进程前将输入路径固定为绝对路径。安装使用 `bash third_party/cordvip_geometry/scripts/install_perception.sh`；从其他目录调用时加 `--workspace /path/to/workspace`。

离线分割和生成保持独立子进程边界，通过 `python -m cordvip_geometry...` 启动，避免相互叠加模型显存。多物体回路共享一个连续 SAM2 编码器，多个 FoundationPose 实例串行共享网络与渲染 context，每实例保留独立 mesh/pose。`--help`、静态检查和普通 import 不加载模型或连接设备。安装和验证不会自动启动相机。

离线追踪可用 `--initial-selection` 复用与首帧完全匹配的已确认分割，省去重复分割。首帧注册完成后才开始回放，结束自动退出；`--headless` 需要离线源与该选择。`--save-frames` 和 `--save-video` 是可选归档，内存观测可直接消费。用法与可复制的 toy 命令见[示例](../../examples/cordvip/README.md)。

`track_object.py --scene scene.json` 为多个实例配置角色、可选资产和首帧选择。默认使用 SAM2.1 small 逐处理帧传播 mask，历史缓存有界；几何提取不填篮孔，不用模型深度替代相机深度。Python 宿主可持有 `SceneTracker`，将同源、同帧校正后的 RGB-D 传入 `step`；其返回值按实例包含实测点、可选 pose/模型点和独立有效性。首帧显式传入已确认 masks，此后因果推进。源重启、回放倒退或内参变化应创建新会话；库不做静默重注册。详见[多物体接口](docs/usage.md#多物体连续分割与独立几何)。

手—物输出同时保留 camera frame 与 robot base 的几何，包含法向、link IDs 与实测关节。默认 1024 手点 + 1024 物体点、XArm7 + XHand 的 19 关节是本地变体；固定 link 采样也不同于原版逐帧 FPS。`interaction_valid` 只说明当前几何可组成观测，时间差未知时为 NaN，不能据此声称可用于闭环。库不生成 contact/SDF 标签，mesh-face 法向不等同于 CordViP contact 计算使用的 KNN 法向；模型输入变换与训练留在 `dexmani_policy`。
