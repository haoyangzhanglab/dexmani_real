# XHand 手部重定向

将 VR/HTS 的人体右手关键点转换为 XHand 关节目标。腕部运动由 VR 控制器另行映射为机械臂末端目标，两者组成完整遥操作动作。

## 数据流

**21 个关键点 → 掌系与骨长标定 → XHand 目标几何 → 有界优化 → 平滑关节目标**。

请求几何表达人体动作意图，可以超出机器人可达范围。求解器在关节范围和变化率约束内逼近目标，通过时间阻尼与平滑保持连续；对指目标描述指尖间距，不表示真实接触。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| [landmarks.py](../dexmani_real/teleop/retargeting/landmarks.py) | 建立掌系、标定骨长，按 XHand 机构生成目标几何。 |
| [objective.py](../dexmani_real/teleop/retargeting/objective.py) | 组合位置、方向、对指和阻尼目标，计算目标值与解析梯度。 |
| [solver.py](../dexmani_real/teleop/retargeting/solver.py) | 用 NLopt SLSQP 求解，管理平滑与提交参考。 |
| [HandFK](../dexmani_real/geometry/kinematics.py) | 基于 Pinocchio 与 URDF 提供正运动学和雅可比，也供导出与策略观测使用。 |

## 与控制链的关系

[VR 接收端](../dexmani_real/sensor/vr.py) 提供关键点，[VRController](../dexmani_real/teleop/vr.py) 组合腕部映射与 `HandRetargeter` 输出，再由共享动作映射和遥操作循环提交、录制。重定向模块不直接发送设备命令；求解参考从实测姿态建立，仅在命令提交正常返回后更新。

## 离线预览

使用已有 Raw 独立运行手部算法：

```bash
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --config experiments/hand.py
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --max-frames 300 --no-view
```

入口只读取所选前缀的实测手关节、VR landmarks、时间行及段边界，不读取图像或 HOI，不连接设备、不写回数据。路径支持 `~`，相对路径以 cwd 为准。

预览与在线重定向共用 NLopt 求解器；依赖安装见 [CPU 离线环境](offline_cpu.md)。无窗口预览不需要 Ruckig、Policy 或 Rerun。关键点每次最多读取 256 行，计算块之间延续同一个求解参考，只有录制段界会重置。返回的关节曲线仍按所选帧数保留。

默认从 `effective_parameters` 恢复完整的 `hand_retargeting`、`hand_retargeting_calibration` 及手部限位。`--config` 加载导出 `CONFIG: RealConfig` 的 Python 文件，仅替换 `hand_retargeting`；标定和限位始终取自 Raw。缺少所需记录或配置错误直接报错。

计算使用记录的 `observation_ns`：段首按实测关节和该行时间 `reset`，只展示参考；后续有效候选以同一行时间 `commit`。拒绝行保留上一参考位置与时间，显示原因并断开候选曲线；段间重置平滑状态。这里的 `commit` 表示离线接受候选，不能证明真实硬件执行成功。

Rerun 在手部局部坐标系对照实测与有效候选的指尖；段首和拒绝行显示保留参考。关节曲线同时保留实测、候选与参考，标出段界及拒绝行，使用时间轴浏览。`--no-view` 不初始化 viewer，仅打印帧数、拒绝原因和求解耗时；耗时只包含 `retarget` 调用。`--max-frames` 限制读取和计算的前缀。

## 数值验证与提交语义

`tests/test_hand_retargeting.py` 使用真实 Pinocchio，按 SDK 关节顺序有限差分检查位置雅可比、位置/近段方向/末段方向/对指/阻尼及总目标。覆盖关节上下限邻域、内部姿态、对指关闭/部分/完全启用、8/62.5/200 ms 间隔及 Huber 转折；梯度验收容差为 `atol=1e-7, rtol=1e-5`。另检查掌系的刚体变换不变性、最终命令变化率、平滑、reset、退化几何和优化器坏输出。

`retarget` 不推进提交参考；拒绝或发送抛错也不推进。在线若成功重发旧命令，位置保持、提交时间更新；这是一次实际提交，与离线拒绝行不调用 `commit` 的行为不同。零时间间隔返回独立的参考副本，时间倒退报错。

## 固定样本与性能对照

固定样本 `tests/fixtures/hand_retargeting/data.h5` 来自真实录制 `pnp_apple_banana/episode_20261101_174738`，保留全部 397 帧、原生单段、时间、VR 关键点、实测关节及提交目标；未保留图像。数值数组与来源一致，参数按当前 `effective_parameters` 格式保存，来源名称保存在元数据中。合成测试另行覆盖段界、坏输入和拒绝。

准备基线 checkout 后交错比较：

```bash
git worktree add --detach /tmp/dexmani-baseline 3d1a69c4d67ce813a63e9fda8a43fe266ee20fb0
python scripts/benchmark_hand_retargeting.py tests/fixtures/hand_retargeting \
  --baseline /tmp/dexmani-baseline --warmup 2 --repeats 10 --output /tmp/hand-benchmark.json
```

基准启动独立进程，轮换前后版本的先后顺序，每个进程预热两次完整回放，再测一遍。读取、导入、初始化、任务空间误差计算不进入完整调用计时；分项独立测量，不能直接相加。输入、float64、线程数、NLopt SLSQP 的 50 次评估/25 ms 预算及全部参数相同。

JSON 仅保存一次样本路径、各 checkout 的 Git 版本、公共参数与环境，再逐轮记录耗时、拒绝、成功求解的评估次数和任务空间误差。每一对运行都比较命令及拒绝结果，报告所有轮次中的最大命令差；单个版本的调用预算由记录参数决定。checkout 有未提交修改时，Git 版本带 `-dirty`。样本至少需要一个段首之后的求解行。

2026-11-17，Ryzen 9 9900X，Linux x86_64，Python 3.10.20、NumPy 1.26.4、Pinocchio 2.7.0、NLopt 2.7.1。下表为各轮 P50/P95 的中位数；原始结果见 [10 轮测量 JSON](hand_retargeting_cpu.json)。基线为 `3d1a69c`，优化代码为 `701b0d1`。

| 项目 | 修改前 P50/P95（μs） | 修改后 P50/P95（μs） |
| --- | --- | --- |
| 目标构造（含掌系转换） | 162.03 / 167.85 | 162.37 / 168.83 |
| 目标对象构造 | 5.35 / 5.61 | 5.44 / 5.76 |
| FK | 8.84 / 9.19 | 8.83 / 9.21 |
| FK／雅可比 | 28.81 / 31.16 | 28.82 / 30.52 |
| 含梯度目标评估 | 65.51 / 82.35 | 60.04 / 75.23 |
| 仅目标值评估 | 31.60 / 41.49 | 24.56 / 32.06 |
| 完整 retarget | 865.95 / 1155.38 | 803.45 / 1057.53 |

完整调用 P50/P95 降低约 **7.2% / 8.5%**，10 轮配对均改善。每轮双方都执行 4038 次目标评估（逐调用 P50/P95 为 10/14），0 次拒绝，95 帧启用对指；最大命令差 `8.0e-13 rad`。指尖 RMSE P50/P95 均为 `3.633/8.108 mm`，平均方向误差为 `2.745/3.830°`，启用对指的距离误差为 `3.158/5.182 mm`。这些是对请求几何的跟踪残差，不是真实硬件误差。

收益来自复用平方范数、直接归约与相邻切片，并在仅求目标值时省去导数数组。目标函数与求解预算不变。另测了 FK／雅可比直接填充数组：5 轮 P50 改善落在 `-1.0%～+1.2%`，未形成收益，因此保留原实现。目标构造暂不新增缓存。

返回[项目架构](../README.md)。
