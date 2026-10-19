# cordvip_geometry

独立的 CordViP / ProxiDex 手—物感知工具。Grounding DINO 提供首帧文本候选，SAM2.1 small 生成 mask 并逐帧传播；FoundationPose++ 核心用当前 mask 的 2D bbox 中心引导位姿细化。当前变体不使用 Kalman。Hunyuan3D-2.1 继续负责可选的离线单图建模，生成 mesh 须经实测尺度校准。

两个示例入口 [reconstruct_object.py](../../examples/cordvip/reconstruct_object.py) 和 [track_object.py](../../examples/cordvip/track_object.py) 分别调用 `python -m cordvip_geometry object build` 与 `track` 的同一实现。先看[示例](../../examples/cordvip/README.md)，安装、Python 调用和输出说明见[使用文档](docs/usage.md)。

```bash
conda activate real_robot
bash third_party/cordvip_geometry/scripts/install_perception.sh
python -m cordvip_geometry download --models grounding sam
python -m cordvip_geometry check --offline-input --headless
```

`real_robot` 是示例环境名。SAM 子解释器按 Python 和固定依赖版本检查，父环境按实际启用的位姿、手点或显示能力检查；具体要求见[使用文档](docs/usage.md)。源码与权重版本固定，支持合法 Git worktree，SAM 文档修改不影响源码准入。模型从本地加载；`--help`、静态检查和普通 import 不加载模型或连接设备。

默认 SAM2.1 small、BF16、eager。文本检测只用于首帧候选确认，进入跟踪后释放 Grounding DINO；多实例共享图像编码与位姿细化预处理。已有确认 mask 可直接开始传播。large 对照、参数及内存接口见[使用文档](docs/usage.md)。

单物体与多物体使用同一个追踪入口。默认处理最新帧，离线加 `--all-frames` 逐帧处理。`replay` 提供无模型 RGB-D 预览。

语义实测点来自当前 mask 和原始实测深度；完整物体点来自固定资产和追踪位姿。二者独立保存，模型与深度的一致性仅用于诊断。分割误差、遮挡、物体形变及资产尺度误差仍须结合源数据评估。

`--with-hand` 从同一 Raw 行的实测关节生成 FK 手点，使用数据对应的相机外参与当前实验安装配置。手—物表示与原始模型输入的适配、训练和 contact/proximity 标签属于 `dexmani_policy`。

实时相机命令会连接 RealSense；`--with-hand` 目前只支持只读 Raw。宿主接入时应复用已有 RGB-D、实测关节和时间证据。主包 runtime、采集、15 字段导出与部署尚未接入本工具。

运行产物与模型放在 workspace 的 `reports/` 下，不写回 Raw。脚本入口显式传入 workspace，模块入口默认使用当前目录。
