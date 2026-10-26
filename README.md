# DexMani Real

DexMani Real 是 DexMani 灵巧操作（Dexterous Manipulation）研究中的真实机器人实验仓库。系统基于 **xArm7 + XHand + RealSense RGB-D + VR/HTS**，连接机器人控制、多模态感知与学习策略，支持遥操作示教、数据采集、策略部署和真机评估。

## 项目定位

DexMani 的研究代码分为三个相互关联的仓库：

| 仓库 | 主要职责 |
| --- | --- |
| **dexmani_real** | 真实设备接入、实验运行、数据采集与真机评估 |
| [**dexmani_policy**](https://github.com/haoyangzhanglab/dexmani_policy) | 策略模型、训练与推理 |
| [**dexmani_sim**](https://github.com/haoyangzhanglab/dexmani_sim) | 仿真环境与仿真实验 |

Real 聚焦真实世界中的感知与动作执行，并对接 Policy 的训练结果；仿真相关实验由 Sim 承载。

## 系统架构

系统以遥操作、策略评估和轨迹回放为实验入口，共享几何、观测表示、控制计算和运行机制。

| 组成 | 模块 | 职责 |
| --- | --- | --- |
| 设备接入 | [`robot/`](dexmani_real/robot/) · [`sensor/`](dexmani_real/sensor/) | 机械臂、灵巧手、RGB-D 相机及 VR 的连接与采集；机器人协调入口为 `robot/system.py` |
| 几何与感知 | [`geometry/`](dexmani_real/geometry/) · [`perception/`](dexmani_real/perception/) | 位姿、坐标变换和 FK；在线/离线共享点云、指尖与手表面表示 |
| 动作控制 | [`control/`](dexmani_real/control/) | 动作映射、在线 IK、目标约束、关节等价角与 HOME 路径；在线接触和 HOME 间隙规则保持分离 |
| 共享运行 | [`runtime/`](dexmani_real/runtime/) · [`ipc/`](dexmani_real/ipc/) | 会话状态、键盘输入、观测、资源启停、HOME/jog 协调及进程间通信 |
| 实验执行 | [`teleop/`](dexmani_real/teleop/) · [`deployment/`](dexmani_real/deployment/) · [`replay/`](dexmani_real/replay/) | 遥操作、学习策略执行与轨迹回放 |
| 数据管理 | [`data/`](dexmani_real/data/) | Raw 记录、Canonical 处理与导出；`session_results.py` 管理会话结果 |
| 配置与标定 | [`config.py`](dexmani_real/config.py) · [`calibration/`](dexmani_real/calibration/) | 实验配置装配、标定算法、交互会话及默认参数；配置定义仍位于各领域 |
| 通用工具 | [`utils/`](dexmani_real/utils/) | 原子写入、计时、日志与可视化 |

遥操作和策略部署分别处理来自操作者与学习模型的动作，但共享机器人状态、运动规划和底层执行机制。数据记录和结果分析围绕实验过程展开，不与特定策略模型绑定。

## 实验流程

- **遥操作示教**：通过 VR/HTS 获取操作者运动，完成机械臂与灵巧手的动作映射及示教采集。
- **策略部署与评估**：将真实设备观测送入训练得到的策略，执行预测动作并记录真机实验结果。
- **离线分析与回放**：查看、处理实验数据，并利用已记录的动作轨迹开展回放和分析。

标定与设备诊断为这些流程提供共同的实验支持。

相机标定位于 [`calibration/camera/`](dexmani_real/calibration/camera/)，其中 `capture.py` 负责静止采样，`session.py` 负责运动、样本管理与标定发布。桌面拟合与质量评估位于 [`geometry/plane.py`](dexmani_real/geometry/plane.py)，路径解析和标定读写位于 [`calibration/table_plane.py`](dexmani_real/calibration/table_plane.py)；VR 对齐与启动质量检查位于 [`calibration/vr_alignment.py`](dexmani_real/calibration/vr_alignment.py)。默认参数集中在 [`calibration/parameters/`](dexmani_real/calibration/parameters/)，路径常量由 `dexmani_real.calibration` 提供，消费者需要时才读取。

实验配置使用 `from dexmani_real.config import load_experiment_config`，覆盖优先级为 CLI > YAML > dataclass 默认值。配置加载不读取标定或连接设备，运行约束在使用边界验证。`environment.table.plane_path` 的相对路径以仓库根目录解析；显式指定的必需标定缺失时直接报错。VR 标定脚本的 `--force` 仅允许将低质量结果导出到显式指定、非默认且未占用的 `--output` 文件；遥操作启动仍拒绝低质量标定。

机器人会话在 preflight 统一检查设备、HOME、工作空间及实际启用的碰撞配置；相机及 XHand 独立工具在自己的入口检查。内部 driver、controller 和计算函数接收已经通过入口检查的配置与标定，不重复预检。新的 SDK 反馈、模型输出、文件数据及最终发送目标仍在各自边界校验。`--print-config` 仅展示声明值；`run_policy()` 在加载策略前检查运行轮数和时长。

## 代码导航

核心实现位于 [`dexmani_real/`](dexmani_real/)，实验入口与设备工具集中在 [`scripts/`](scripts/)；[`assets/`](assets/) 存放机器人模型等资源，[`third_party/`](third_party/) 包含第三方相关代码。

部署保留 `deployment/run.py → session.py → rollout/episode.py`：入口装配配置，会话持有设备和模型，单轮协调观测、提交与录制。`rollout/action_schedule.py` 只根据传入的时钟与状态决定 sync/async/RTC 调度，`inference.py` 独占模型线程。回放的帧索引一致性指标位于 [`replay/consistency.py`](dexmani_real/replay/consistency.py)。

`control/constraints.py` 统一等价角、绝对跳变和 `MotionConstraints` 端点约束。`control/ik.py` 的 `IKSolver` 接收共享约束、位置和旋转容差，`solve(..., hand_qpos=...)` 显式指定碰撞手姿；动作映射、键盘 jog 和相机标定共用这一接口。

IK 保留上一命令、实测状态及正负零空间初值；评分只使用关节距离和限位惩罚，搜索权重与步长固定在算法模块。位姿误差用于容差准入，SVD 只用于生成零空间方向。候选 FK 同时用于误差和工作空间检查，碰撞模型统一管理自碰撞、静态障碍和可选桌面。`HomePlanner(environment, arm, table_plane_abcd=...)` 与执行器共用 `ArmConfig` 的运行限位、到位容差及桌面间隙配置；`plan(current, target)` 保留三种关节路径、1° 采样及绝对等价分支对齐。无需移动的 HOME 仍检查几何约束，并由实测反馈确认停稳。

`geometry` 中，`camera.py` 负责内参与反投影，`pose.py` 负责位姿与刚体变换，`kinematics.py` 提供 `ArmFK`、`HandFK`，`plane.py` 拟合桌面。`perception/pointcloud.py` 将 RGB-D 经深度边缘过滤、工作空间与桌面裁剪、体素均值、半径去噪和 QuickFPS 转为基座系 XYZRGB；`hand.py` 生成指尖与固定手表面点云，`worker.py` 发布在线点云。诊断快照与性能测量位于 `scripts/tools/view_pointcloud.py`。

点云参数集中在 `perception/config.py`。`table_margin_m` 是点到向上桌面平面的最小距离，默认 7 mm；该范围内的低矮物体表面也会被裁掉。点数不足时循环已有点，空点云不发布。Canonical 保存完整实验参数，部署按保存参数恢复；旧版连通域、双阈值桌面配置及采样描述字段不再接受，需要重新导出数据并重新训练受影响的 checkpoint。诊断快照直接写入独立目录，失败时可能保留部分文件。

单阈值裁剪对桌面标定残差更敏感。可根据实验物体高度和当前标定调整 YAML 中的 `pointcloud.table_margin_m`；增大余量会减少桌面残留，也会裁掉更多低矮表面。离线导出和查看在配置入口校验手部安装位姿，指尖名称与 SDK→URDF 关节顺序由 `HandFK` 核对。

XHand 的 SDK 载荷解析、反馈布局与缓存位于 `robot/xhand/feedback.py`；机械臂反馈布局位于 `robot/xarm7/protocol.py`。

机械臂由 `robot/xarm7/client.py` 的 `ArmClient` 提交目标，`worker.py` 的 `ArmController` 独占 SDK 与默认 100 Hz 伺服，Ruckig 负责连续参考。共享通道及操作消息集中在 `protocol.py`；HOME 总截止时间从运行层传到父、子进程，取消使用同一个停止信号。操作回复直接携带超时、取消或设备异常，保留失败原因；停止失败始终作为设备故障报告。

控制与设备配置类使用 `MotionConfig`、`ArmConfig`、`HomingConfig`、`HandConfig`。SDK 超时和伺服间隔预算均固定为 20 ms，进程等待为 3 s；HOME 的到位附加预算、稳定持续时间、收敛等待分别为 0.5、0.3、15 s，轮询周期使用机械臂反馈频率。XHand 串口稳定等待和初始反馈等待均为 1 s。这些预算不再接受 YAML 覆盖；机械限位和指尖名称由 `robot/model.py` 提供，实验仍可配置运行关节限位、控制频率、HOME 姿态、收敛容差及硬件增益。

`JogController.submit()` 负责输入代际、运动授权和提交后参考更新。录制使用 `AsyncEpisodeRecorder.finish()` 显式指定保存意图、终止原因和技术状态。

`runtime` 按职责平铺为 `state`、`keyboard`、`observation`、`lifecycle`、`preflight`、`home` 和 `jog`。`SessionState` 管理运行代际与停止请求，`SensorWorkers` 持有传感进程，`close_runtime()` 按顺序回收资源。`home_hand()`、`home_arm()` 直接使用机器人对象中的设备配置，并分别负责本次运动的停止；`home_robot()` 顺序执行两者。

`Keyboard.poll()` 返回小写按键字符串；命令、标定和预览共用一个事件队列。离散操作每次按下只触发一次，长按 jog 仍连续移动，松键会使旧提案失效。每个键盘实例只启动一次，`discard_events(*keys)` 用于丢弃阻塞操作期间的旧事件。

运行资源超时使用代码常量：相机、点云、VR 启动分别为 15、30、120 秒，正常退出等待预算为 65 秒，定义于 `runtime/lifecycle.py`；模型启动为 300 秒，定义于 `deployment/session.py`。部署清理共享同一截止时间；仍存活的传感进程依次执行 terminate、kill，各等待至多 1 秒，因此总清理时间可能超过正常退出预算。超时不属于 YAML 配置项，未知配置键会报错。

项目依赖与打包配置见 [`pyproject.toml`](pyproject.toml)，开发约定见 [`AGENTS.md`](AGENTS.md)。

## 运行接口与错误处理

遥操作入口为 `teleop.session.run_teleop(...)`，回放入口为 `replay.session.run_replay(...)`，对应脚本解析参数并调用入口。点云诊断的装配、运行、快照输出及收尾集中在 `scripts/tools/view_pointcloud.py`。

遥操作配置直接使用以下字段，覆盖顺序为 CLI > YAML > 默认值；未定义的配置键会报错。

```yaml
teleop:
  control_hz: 16.0
  ema_alpha_pos: 0.7
  ema_alpha_rot: 0.5
  vr_pos_scale: 0.4
  vr_stale_threshold_s: 0.5
  hand_retargeting_type: tag
```

VR 接收器在发布前拒绝无效传感数据。映射器按已接纳的观测建立锚点，映射或重置发生技术错误时直接抛出异常；重置失败会清除旧锚点。遥操作控制器复用动作检查器的 FK 初始化 VR 锚点与 EMA，EMA 只在命令正常提交后更新参考。

IK 使用入口明确提供的运行关节限位，正常无解返回 `IKResult`，求解器、FK 或碰撞后端故障直接抛出原始异常。HOME 仅对 `HomePathRejected` 表示的几何拒绝尝试其他候选路径，候选耗尽、取消或等待超时返回失败结果；非法目标与技术异常传播到会话错误处理和收尾边界。

XHand 保留 serial 与 EtherCAT，连接失败直接报告，不自动重试。普通发送遇到 CRC 只等待下一条目标，持续失败超过反馈预算即报错；停止请求可在 0.25 s 内重发。状态码明确不可用的触觉被屏蔽，有限向量缓存保留原始来源时间；SDK 结构错误和复位调用异常直接传播。连接清理失败不会覆盖初始错误，未释放的句柄仍由设备对象保留。

回放先保存实测数据，再计算并写入指标。没有有效样本的指标显式记为 `null`；计算产生的非有限数值由 JSON 写入边界拒绝并报告评估失败，保留已保存的实测数据。

回放结束后，H 请求回初始位，S 取消 HOME，Q 退出，Esc 急停。S 优先于同批次 H；完成停止确认后可重新按 H。HOME 期间积累的 H 不会在返回后自动执行，S 中断记录为取消。

## 观测与产物契约

传感器 IPC 布局和通道分配位于 [`ipc/sensors.py`](dexmani_real/ipc/sensors.py)，[`SharedRing`](dexmani_real/ipc/ring.py) 面向 Linux x86_64，通过 `spawn` 传给单个生产者和多个消费者。`write(**fields)` 返回发布序列；`read(sequence=None, fields=None)` 返回 `(data, sequence)`，完整帧为结构化标量，按字段读取为独立副本字典。空缓冲、目标序列被覆盖或复制期间发生覆盖均返回 `None`，不重试、不缓存旧帧。

来源时间由载荷携带，IPC 不生成替代来源时间的发布时钟。点云携带 `source_camera_sequence` 和原始相机 `timestamp_ns`；同时需要像素与点云时，只读取对应相机序列，匹配失败视为观测缺失。控制所需观测缺失时，遥操作或部署按现有准入规则结束本轮；标定采样继续在自身期限内等待。

`read_camera_frame()` 返回来源 `timestamp_ns`、共享内存发布序列 `ring_sequence`、RGB 设备帧号 `color_frame_number` 及请求的像素通道。发布序列不是设备帧身份；时间戳保留 RGB-D 中较早通道的首次主机接收时间，不代表曝光时刻。标定采样和预览只读取 RGB，采样仍要求屏障后的新鲜、不同 RGB 帧。

`termination_reason` 保留首次终止原因，`technical_status` 反映该次产物的技术完成状态，发布状态单独描述文件是否已发布。发布前已知的停止或录制失败不会因为首次原因是 operator、quit 或正常时长结束而变为技术正常。失败产物可保留有效数据前缀；已发布 Raw 不因后续独立会话清理错误而回写。Teleop 的显式 DISCARD 在停止失败时仍保持丢弃意图。

Teleop 中断会登记技术失败并传播，关闭时继续按原保存或丢弃意图收尾；recorder 的首次收尾请求不会被重入覆盖。收尾超时仍报告 pending，不在同一次关闭中重新开启等待预算。Policy 在会话关闭末尾核对对应 episode 的 Raw 发布状态、实际路径和帧数，再写入 `session_result.json`；该摘要反映最后一次确认，不保证仍 pending 的 writer 之后的结果，也不会清除首次终止原因或已有错误。

`EpisodeReader` 在首次读取时统一校验已定义数值字段和 `timing/rows` 的布局，并缓存数据集。`read_timing_rows(fields)` 按请求顺序返回结构化字段，保留 `int64` 纳秒及 `-1` 缺失标记。诊断读取不隐式执行训练或物理回放准入；Canonical 导出与物理回放均显式调用 `validate_completion()`、`validate_rows()` 执行各自所需的技术状态和时序检查。CordViP 可读取失败产物或缺失来源时钟的诊断数据，但同一 Raw 行不能证明手和相机物理同步。Reader 关闭后文件读取与校验接口报错，已返回的 NumPy 数组仍可使用。

## 离线验证

在已安装项目及相关依赖的 Python 环境中运行：

```bash
python -m compileall -q dexmani_real scripts
python -m unittest discover -s tests -v
python scripts/run_policy.py --print-config
python scripts/collect_teleop.py --print-config
git diff --check
```

已有 Ruff/mypy 的开发环境可补充静态检查，无需为运行实验安装它们：

```bash
ruff check --no-cache --isolated --select E9,F dexmani_real scripts tests third_party/cordvip_geometry/src
mypy --ignore-missing-imports --follow-imports silent --explicit-package-bases \
  dexmani_real/ipc dexmani_real/sensor/camera/worker.py \
  dexmani_real/geometry dexmani_real/perception dexmani_real/control dexmani_real/robot \
  dexmani_real/calibration dexmani_real/runtime dexmani_real/deployment \
  dexmani_real/data/canonical \
  dexmani_real/data/session_results.py dexmani_real/data/raw/reader.py \
  dexmani_real/teleop/runner.py dexmani_real/replay/replayer.py \
  scripts/tools/view_pointcloud.py scripts/tools/view_realsense.py \
  scripts/visualize_episode.py scripts/export_zarr.py \
  third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py
```

mypy 按现有类型标注检查上述模块，不检查未标注函数体或外部无类型声明的 SDK；不能替代数值与运行边界测试。

`tests/test_control.py` 使用真实模型和可控求解结果检查 IK、碰撞、HOME 及策略动作格式；`tests/test_robot.py` 使用假 SDK 和进程管道检查时效、取消、异常传递、停止、CRC 和资源收尾，并验证真实 Ruckig 的连续性与制动限位。测试不会连接机器人，需要已安装 Pinocchio、MPlib、HPP-FCL 和 Ruckig。

`tests/test_geometry.py` 与 `tests/test_perception.py` 检查坐标约定、畸变反投影、桌面拟合、采样以及在线和离线几何一致性；导出测试使用实际 HDF5、视频和 Zarr，覆盖完整参数恢复与无效点云 episode 的回滚。

键盘遥操作入口保留 `--help`，没有 `--print-config` 参数。

真机启停、取消/急停、HOME/jog、遥操作录制、三种策略模式和回放仍需操作者单独验收。软件授权撤销和 SDK 接受请求不代表物理停稳或到位。

默认标定 JSON 随 wheel 打包；机器人 `assets/` 仍沿用 checkout 定位方式，验证 wheel 中的 JSON 不等于验证了完整独立安装的机器人运行环境。
