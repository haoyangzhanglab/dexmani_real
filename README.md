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

相机标定位于 [`calibration/camera/`](dexmani_real/calibration/camera/)，其中 `aruco.py` 负责检测、预览与静止采样，`hand_eye.py` 使用五种算法择优求解，`run.py` 负责运动、样本管理与标定保存。标记和质量阈值在所属模块就近定义。只有位置 RMS ≤ 5 mm 且旋转 RMS ≤ 3° 的结果可以保存；未达标结果保留残差，X 按两种残差相对门限的最大值删除最差样本，之后需重新求解。

桌面拟合与质量评估位于 [`geometry/plane.py`](dexmani_real/geometry/plane.py)，路径解析和标定读写位于 [`calibration/table_plane.py`](dexmani_real/calibration/table_plane.py)；VR 对齐与启动质量检查位于 [`calibration/vr_heading.py`](dexmani_real/calibration/vr_heading.py)。默认参数集中在 [`calibration/parameters/`](dexmani_real/calibration/parameters/)，路径常量由 `dexmani_real.calibration` 提供，消费者需要时才读取。

相机外参通过 `load_camera_extrinsics()` 按 serial 返回矩阵字典；可选录制仅允许默认文件缺失，损坏文件、显式路径错误或文件中缺少当前相机 serial 均直接报错。

实验配置使用 `from dexmani_real.config import load_experiment_config`，覆盖优先级为 CLI > YAML > dataclass 默认值。配置加载不读取标定或连接设备，运行约束在使用边界验证。`environment.table.plane_path` 的相对路径以仓库根目录解析；显式指定的必需标定缺失时直接报错。VR 标定脚本的 `--force` 仅允许将低质量结果导出到显式指定、非默认且未占用的 `--output` 文件；遥操作启动仍拒绝低质量标定。

机器人会话在 preflight 统一检查设备、HOME、工作空间及实际启用的碰撞配置；相机及 XHand 独立工具在自己的入口检查。内部 driver、controller 和计算函数接收已经通过入口检查的配置与标定，不重复预检。新的 SDK 反馈、模型输出、文件数据及最终发送目标仍在各自边界校验。`--print-config` 仅展示声明值；`run_policy()` 在加载策略前检查运行轮数和时长。

## 代码导航

核心实现位于 [`dexmani_real/`](dexmani_real/)，实验入口与设备工具集中在 [`scripts/`](scripts/)；[`assets/`](assets/) 存放机器人模型等资源，[`third_party/`](third_party/) 包含第三方相关代码。

部署沿 `deployment/run.py → session.py → rollout.py` 执行：`config.py` 统一解析与校验执行配置，会话持有设备和模型，`PolicyRollout` 协调单轮观测、提交与录制。`schedule.py` 根据时钟与状态决定 sync/async/RTC 调度，`observation.py` 按需组装输入，`inference.py` 独占模型线程。来源时效在观测入口检查，策略输入的数值与形状在 `dexmani_policy` 边界检查；历史快照仅在提交预测时创建。回放沿 `replay/run.py → player.py` 执行，`data.py` 读取轨迹并缓存实测数据，`metrics.py` 在硬件收尾后计算匹配帧的关节与末端误差。

`control/constraints.py` 统一等价角、绝对跳变和 `MotionConstraints` 端点约束。`control/ik.py` 的 `IKSolver` 接收共享约束、位置和旋转容差，`solve(..., hand_qpos=...)` 显式指定碰撞手姿；动作映射、键盘 jog 和相机标定共用这一接口。

IK 保留上一命令、实测状态及正负零空间初值；评分只使用关节距离和限位惩罚，搜索权重与步长固定在算法模块。位姿误差用于容差准入，SVD 只用于生成零空间方向。候选 FK 同时用于误差和工作空间检查，碰撞模型统一管理自碰撞、静态障碍和可选桌面。`HomePlanner(environment, arm, table_plane_abcd=...)` 与执行器共用 `ArmConfig` 的运行限位和到位容差，桌面间隙参数由 `EnvironmentConfig` 提供；`plan(current, target)` 保留三种关节路径、1° 采样及绝对等价分支对齐。无需移动的 HOME 仍检查几何约束，并由实测反馈确认停稳。

`geometry` 中，`camera.py` 负责内参与反投影，`pose.py` 负责位姿与刚体变换，`kinematics.py` 提供 `ArmFK`、`HandFK` 和批量 `eef_poses`，`plane.py` 拟合桌面。`perception/pointcloud.py` 的 `PointCloudProcessor` 在初始化时计算基座系射线和桌面高度系数，逐帧通过 `process(depth_raw, color)` 完成深度过滤、裁剪、体素均值、半径去噪和 QuickFPS。每个相机或 episode 复用一个处理器，几何参数变化时重新构造。`hand_cloud.py` 提供 `HandPointCloud(hand_fk, num_points)` 和 `hand_geometry`，指尖与手点云每行共用一次 FK；`worker.py` 发布在线点云。

`perception/config.py` 只保留点云点数、工作空间、桌面开关与余量，以及手点云点数。`table_margin_m` 是点到向上桌面平面的最小距离，默认 7 mm；该范围内的低矮物体表面也会被裁掉。深度范围 0.30–1.50 m、断层阈值 30 mm、表面带宽 8 mm、体素 5 mm、去噪半径 12 mm 和最少 6 个邻居固定在算法模块。点数不足时循环已有点，空点云在线不发布、离线导出报错。

手表面按当前 XHand URDF 的单个本地网格、零 visual 位姿和内嵌颜色读取，这些资产前提由测试检查；采样按面积分配，按 link 顺序使用从 0 开始的固定种子。Canonical 从每个 episode 的相机快照统一派生点云与相机参数，并保存保留的实验参数，部署按保存参数恢复。包含已删除过滤参数或 `seed` 的旧配方不兼容，须重新导出数据并重新训练受影响的 checkpoint；未知 YAML 参数直接报错。

`scripts/tools/view_pointcloud.py` 提供诊断快照与性能测量；快照直接写入独立目录，失败时可能保留部分文件。`scripts/tools/view_realsense.py` 提供实时预览，点云计算、标定读取或设备故障会退出并报告失败；退出时停止键盘监听、断开 viewer、恢复原曝光优先级并断开相机，恢复失败也会报告失败。

单阈值裁剪对桌面标定残差更敏感。可根据实验物体高度和当前标定调整 YAML 中的 `pointcloud.table_margin_m`；增大余量会减少桌面残留，也会裁掉更多低矮表面。离线导出和查看在配置入口校验手部安装位姿，指尖名称与 SDK→URDF 关节顺序由 `HandFK` 核对。

XHand 的 SDK 载荷解析、错误码和板卡诊断位于 `robot/xhand/driver.py`，标准化状态、反馈布局与缓存位于 `feedback.py`；机械臂反馈布局位于 `robot/xarm7/protocol.py`。

机械臂由 `robot/xarm7/process.py` 的 `ArmProcess` 提交目标，`worker.py` 的 `ArmController` 独占 SDK 与默认 100 Hz 伺服，`trajectory.py` 的 `ArmTrajectory.step(now_ns, goal=None)` 用 Ruckig 在每周期采样一次连续参考。共享通道及操作消息集中在 `protocol.py`，`ArmChannels` 负责目标发布、消费和状态快照；HOME 总截止时间从运行层传到父、子进程，取消使用同一个停止信号。操作回复直接携带超时、取消或设备异常，保留失败原因；停止失败始终作为设备故障报告。

控制与设备配置类使用 `MotionConfig`、`ArmConfig`、`HomingConfig`、`HandConfig`。SDK 超时和伺服间隔预算均固定为 20 ms，进程等待为 3 s；HOME 的到位附加预算、稳定持续时间、收敛等待分别为 0.5、0.3、15 s，轮询周期使用机械臂反馈频率。XHand 串口稳定等待和初始反馈等待均为 1 s。这些预算不再接受 YAML 覆盖；机械限位和指尖名称由 `robot/model.py` 提供，实验仍可配置运行关节限位、控制频率、HOME 姿态、收敛容差及硬件增益。

桌面代理检查的 YAML 字段为 `environment.table_z_surface_m`（默认 0.022 m）和 `environment.hand_safety_margin_m`（默认 0.05 m）。标定桌面的 mesh 距离检查仍使用 `environment.table.soft_clearance_m`。

机器人同步操作及 XHand 诊断脚本的取消统一抛出 `OperationCancelled`，运行边界以 `RobotError` 报告设备故障；初始化和收尾保留原始失败原因。SDK 原生输出直接显示。

`JogController.submit()` 负责输入代际、运动授权和提交后参考更新。`EpisodeRecorder` 使用 `start()`、`append()`、`finish()`，收尾时显式指定保存意图、终止原因和技术状态，`result()` 返回最后可观察的录制结果。

`sensor` 平铺为 `config.py`、`realsense.py` 和 `vr.py`，进程入口为 `run_camera` 和 `run_vr`，SDK 在实际使用时导入。配置类型为 `CameraConfig`、`VRConfig`；相机预热固定为 10 帧，来源停滞期限为 2 秒，这两项不接受 YAML 覆盖。`RealSenseCamera.read()` 返回 `RGBDFrame`，单次读取等待固定为 300 ms，短暂等待超时或重复帧返回 `None`，持续停滞或 SDK 故障抛出异常。驱动统一维护来源时间与去重，`depth` 为对齐到 color 网格的原始深度，`color_intrinsics`、`depth_scale_m` 在连接成功后直接读取。

`runtime` 按职责平铺为 `state`、`keyboard`、`observation`、`lifecycle`、`preflight`、`home` 和 `jog`。`SessionState` 管理运行代际与停止请求，`SensorWorkers` 持有传感进程，`close_runtime()` 按顺序回收资源。`home_hand()`、`home_arm()` 直接使用机器人对象中的设备配置，并分别负责本次运动的停止；`home_robot()` 顺序执行两者。

`Keyboard.poll()` 返回小写按键字符串；命令、标定和预览共用一个事件队列。离散操作每次按下只触发一次，长按 jog 仍连续移动，松键会使旧提案失效。每个键盘实例只启动一次，`discard_events(*keys)` 用于丢弃阻塞操作期间的旧事件。

运行资源超时使用代码常量：相机、点云、VR 启动分别为 15、30、120 秒，正常退出等待预算为 65 秒，定义于 `runtime/lifecycle.py`；模型启动为 300 秒，定义于 `deployment/session.py`。部署清理共享同一截止时间；仍存活的传感进程依次执行 terminate、kill，各等待至多 1 秒，因此总清理时间可能超过正常退出预算。超时不属于 YAML 配置项，未知配置键会报错。

项目依赖与打包配置见 [`pyproject.toml`](pyproject.toml)，开发约定见 [`AGENTS.md`](AGENTS.md)。

## 运行接口与错误处理

遥操作入口为 `teleop.run.run_teleop(...)`，回放入口为 `replay.run.run_replay(...)`，对应脚本解析参数并调用入口。

回放输出 `replay_data.npz`（实测关节、候选及来源/提交时间戳）、`metrics.json`（按帧索引比较的误差）和 `session_result.json`（回放、回零与收尾结果）。候选和提交完成都不代表物理到位；手部准备不占回放行，准备时长保存在运行摘要中。零行指标使用 `null`，不表示零误差。

回放使用 Raw 的帧率安排提交，过期帧立即中止；目标的执行期限沿用 `execution.max_decision_age_s`，入口要求该值为有限正数。

点云诊断的装配、运行、快照输出及收尾集中在 `scripts/tools/view_pointcloud.py`。

`teleop` 中，`run.py` 装配资源并完成手部回零与 VR 启动，`loop.py` 的 `TeleopLoop` 管理分段控制和录制收尾，`vr.py` 的 `VRController` 负责锚点映射与 EMA；键盘点动和语音分别位于 `keyboard.py`、`audio.py`。手部算法保留 `retargeting/tag.py`、`dexpilot.py` 和共用的 `landmarks.py`。配置类型统一使用 `Config` 后缀。TAG 的机器人指长、坐标约定及优化器容差和迭代上限固定在算法内部；人体指长、缩放、平滑和捏合权重仍可配置。

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

VR 接收器在发布前拒绝无效传感数据。映射器按已接纳的观测建立锚点，映射或重置发生技术错误时直接抛出异常；重置失败会清除旧锚点。遥操作控制器复用动作检查器的 FK 初始化 VR 锚点与 EMA，手部重定向仅在建立新锚点时用实测关节角重置，EMA 只在命令正常提交后更新参考。

`TAGRetargeter` 集中实现两阶段目标函数和解析梯度，仅在优化器请求梯度时计算雅可比，复用 `HandFK` 的固定模型与关节顺序；`DexPilotRetargeter` 直接调用现有 DexPilot 目标函数、NLopt 和低通滤波器。退化手部几何不产生目标；求解异常（包括 `RoundoffLimited`）停止当前段并沿会话错误路径退出，不返回旧目标或退回 TAG 第一阶段。显式提供的非法初值直接报错；省略初值仍从关节限位中点开始。

`control/actions.py` 提供 `map_joint` 与 `map_eef`，统一返回 `ActionResult`；关节拒绝仍保留诊断候选，发送入口必须检查 `rejection_reason`。IK 直接使用 MPlib 的 `PinocchioModel` 和入口提供的运行关节限位，以 `IKResult.qpos is None` 表示正常无解，求解器、FK 或碰撞后端故障直接抛出原始异常。HOME 每个路径点共用一次运动学更新完成碰撞、EEF 工作空间和桌面间距检查，仅对 `HomePathRejected` 表示的几何拒绝尝试其他候选路径，候选耗尽、取消或等待超时返回失败结果；非法目标与技术异常传播到会话错误处理和收尾边界。

XHand 保留 serial 与 EtherCAT，连接失败直接报告，不自动重试。普通发送遇到 CRC 只等待下一条目标，持续失败超过反馈预算即报错；停止请求可在 0.25 s 内重发。状态码明确不可用的触觉被屏蔽，有限向量缓存保留原始来源时间；SDK 结构错误和复位调用异常直接传播。连接清理失败不会覆盖初始错误，未释放的句柄仍由设备对象保留。

回放先保存实测数据，再计算并写入指标。没有有效样本的指标显式记为 `null`；计算产生的非有限数值由 JSON 写入边界拒绝并报告评估失败，保留已保存的实测数据。

回放结束后，H 请求回初始位，S 取消 HOME，Q 退出，Esc 急停。S 优先于同批次 H；完成停止确认后可重新按 H。HOME 期间积累的 H 不会在返回后自动执行，S 中断记录为取消。

## 观测与产物契约

传感器 IPC 布局和通道分配位于 [`ipc/sensors.py`](dexmani_real/ipc/sensors.py)，[`SharedRing`](dexmani_real/ipc/ring.py) 面向 Linux x86_64，通过 `spawn` 传给单个生产者和多个消费者。`write(**fields)` 返回发布序列；`read(sequence=None, fields=None)` 返回 `(data, sequence)`，完整帧为结构化标量，按字段读取为独立副本字典。空缓冲、目标序列被覆盖或复制期间发生覆盖均返回 `None`，不重试、不缓存旧帧。

来源时间由载荷携带，IPC 不生成替代来源时间的发布时钟。点云携带 `source_camera_sequence` 和原始相机 `timestamp_ns`；同时需要像素与点云时，只读取对应相机序列，匹配失败视为观测缺失。控制所需观测缺失时，遥操作或部署按现有准入规则结束本轮；标定采样继续在自身期限内等待。

`read_camera_frame()` 返回来源 `timestamp_ns`、共享内存发布序列 `ring_sequence`、RGB 设备帧号 `color_frame_number` 及请求的像素通道。发布序列不是设备帧身份；时间戳保留 RGB-D 中较早通道的首次主机接收时间，不代表曝光时刻。标定采样和预览只读取 RGB，采样仍要求屏障后的新鲜、不同 RGB 帧。

`termination_reason` 保留首次终止原因，Teleop 的准备与运行阶段均使用 `RunEndReason` 的小写名称，具体中止上下文写入 `termination_details`。`technical_status` 反映该次产物的技术完成状态，发布状态单独描述文件是否已发布。控制中止后的完整记录可以保存为 `failed` episode；录制队列满、写入或关闭失败则报错并保留临时目录，不发布正式 episode。Teleop 的 D 键回调先锁存丢弃意图并撤销运动，再发布按键事件；收尾请求前到达的 DISCARD 在准备、求解或停止失败时仍保持丢弃意图，writer 退出后删除临时目录。已发布 Raw 不因后续独立会话清理错误而回写。

Teleop 中断会登记技术失败并传播，关闭时继续按原保存或丢弃意图收尾；recorder 的首次收尾请求不会被重入覆盖。收尾超时仍报告 pending，不在同一次关闭中重新开启等待预算。Policy 在会话关闭末尾核对对应 episode 的 Raw 发布状态、实际路径和帧数，再写入 `session_result.json`；该摘要反映最后一次确认，不保证仍 pending 的 writer 之后的结果，也不会清除首次终止原因或已有错误。

`data/raw.py` 集中提供 `RawReader`、`RawWriter` 和元数据编解码。Raw 以 `timing/rows.observation_ns` 为唯一观测时钟，以该表长度作为帧数。`timestamps_s` 在整数时钟相减后转换为相对秒数，空录制返回空数组；反复访问时间轴的调用方应在本地保存该数组。

`RawReader` 首次读取时校验数值字段和 `timing/rows` 的布局，并缓存数据集。`read_camera()` 返回 `CameraSnapshot`，`read_timing_rows(fields)` 保留原始 `int64` 纳秒和 `-1` 缺失标记，文件与视频由 Reader 统一关闭。诊断读取与训练、物理回放准入分开；导出和物理回放显式调用 `validate_completion()`、`validate_rows()`。CordViP 的 Raw 行关联不证明手和相机物理同步；缺失相机时钟时显式使用观测时钟，原始相机时钟仍保留 `-1`。

Raw 写入缓冲按 8 MiB 预算选择 1–32 帧，单帧超过预算时保留一帧。`RawWriter.finish()` 刷新行缓冲、结束视频编码并写入终止详情，再写入技术状态；`close()` 释放资源。录制器在成功收尾后原子发布，写入或收尾失败时保留临时产物并报告错误。

Canonical 由 `data/export.py` 的 `export_zarr(..., processing=ExportConfig(...))` 导出，提供 13 个训练字段、episode 边界及物理参数。导出遇到损坏或不满足训练要求的 episode 立即报错，保留已有临时结果；使用 `--exclude EPISODE` 显式排除后重新导出。最终输出必须尚未占用且不与输入目录重叠，全部完成后才原子发布。处理批次根据 64 MiB 数组预算和最多 100 帧确定，不再提供 `--chunk-frames`；存储块仍按字段大小独立选择。

## 离线验证

在已安装项目及相关依赖的 Python 环境中运行：

```bash
python -m compileall -q dexmani_real scripts
python -m unittest discover -s tests -v
python scripts/run_policy.py --print-config
python scripts/collect_teleop.py --print-config
git diff --check
```

使用 pytest 时运行 `python -m pytest -q`；默认只收集 `tests/`，不收集 `third_party/` 中的示例测试。

已有 Ruff/mypy 的开发环境可补充静态检查，无需为运行实验安装它们：

```bash
ruff check --no-cache --isolated --select E9,F dexmani_real scripts tests third_party/cordvip_geometry/src
mypy --ignore-missing-imports --follow-imports silent --explicit-package-bases \
  dexmani_real/ipc dexmani_real/sensor \
  dexmani_real/geometry dexmani_real/perception dexmani_real/control dexmani_real/robot \
  dexmani_real/calibration dexmani_real/runtime dexmani_real/deployment dexmani_real/utils \
  dexmani_real/data \
  dexmani_real/teleop dexmani_real/config.py dexmani_real/replay \
  scripts/tools/view_pointcloud.py scripts/tools/view_realsense.py scripts/tools/control_xhand.py \
  scripts/visualize_episode.py scripts/export_zarr.py \
  third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py
```

mypy 按现有类型标注检查上述模块，不检查未标注函数体或外部无类型声明的 SDK；不能替代数值与运行边界测试。

`tests/test_control.py` 使用真实模型和可控求解结果检查 IK、碰撞、HOME 及策略动作格式；`tests/test_robot.py` 使用假 SDK 和进程管道检查时效、取消、异常传递、停止、CRC 和资源收尾，并验证真实 Ruckig 的连续性与制动限位。测试不会连接机器人，需要已安装 Pinocchio、MPlib、HPP-FCL 和 Ruckig。

`tests/test_teleop.py` 检查 VR 映射、EMA 提交时机、TAG 梯度、DexPilot 数值等价及求解错误传播，并用假设备和播放器覆盖停止、录制收尾与音频抢占；还需遥操作原有的 NLopt 和 dex_retargeting 依赖。

`tests/test_geometry.py` 与 `tests/test_perception.py` 检查坐标约定、批量旋转、畸变反投影、桌面拟合、XHand 资产前提、固定采样以及在线和离线几何一致性；导出测试使用实际 HDF5、视频和 Zarr，覆盖参数恢复、字段完整性、坏数据立即报错和显式排除。`tests/test_runtime.py` 覆盖实时预览模式切换、异常退出与曝光恢复；`tests/test_sensor.py` 使用假 SDK 检查来源时间、去重、停滞、资源释放和 VR 接收；`tests/test_data.py` 检查批量写盘、失败保留、显式丢弃及发布超时。

`tests/test_deployment.py` 使用假设备和可控模型结果检查三种调度、LAMP 历史、来源期限、初始化失败、迟到推理和收尾，核对 checkpoint 选择、配置恢复与实验结果字段，并通过已安装的 `dexmani_policy` 验证策略输入边界。`tests/test_utils.py` 检查原子写入和计时超期；这些测试均不连接硬件。

键盘遥操作入口保留 `--help`，没有 `--print-config` 参数。

`tests/test_replay.py` 覆盖 Raw 轨迹读取、核心误差、部分采样、过期提交、退出、急停与手部准备；`tests/test_calibration.py` 使用合成姿态和图像验证五算法求解、双轴激励、质量门限、ArUco 采样及 VR 环绕，并检查标定读写和取消。测试均不连接硬件。

真机启停、取消/急停、HOME/jog、遥操作录制、三种策略模式和回放仍需操作者单独验收。软件授权撤销和 SDK 接受请求不代表物理停稳或到位。

默认标定 JSON 随 wheel 打包；机器人 `assets/` 仍沿用 checkout 定位方式，验证 wheel 中的 JSON 不等于验证了完整独立安装的机器人运行环境。
