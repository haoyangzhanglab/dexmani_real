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
| 数据管理 | [`data/`](dexmani_real/data/) | Raw 记录、Canonical 处理与导出；部署结果由 `Rollout` 和 `PolicySession` 持有 |
| 配置与标定 | [`config.py`](dexmani_real/config.py) · [`calibration/`](dexmani_real/calibration/) | 实验配置装配、标定算法、交互会话及默认参数；配置定义仍位于各领域 |
| 通用工具 | [`utils/`](dexmani_real/utils/) | 原子写入、计时、日志与可视化 |

遥操作和策略部署分别处理来自操作者与学习模型的动作，但共享机器人状态、运动规划和底层执行机制。数据记录和结果分析围绕实验过程展开，不与特定策略模型绑定。

## 实验流程

- **遥操作示教**：通过 VR/HTS 获取操作者运动，完成机械臂与灵巧手的动作映射及示教采集。
- **策略部署与评估**：将真实设备观测送入训练得到的策略，执行预测动作并记录真机实验结果。
- **离线分析与回放**：查看、处理实验数据，并利用已记录的动作轨迹开展回放和分析。

标定与设备诊断为这些流程提供共同的实验支持。

相机标定位于 [`calibration/camera/`](dexmani_real/calibration/camera/)，其中 `aruco.py` 的 `ArucoSampler` 负责检测与静止采样，`hand_eye.py` 使用五种算法择优求解，`calibrate.py` 的 `CameraCalibrator` 负责运动、预览、样本管理与标定保存。标记和质量阈值在所属模块就近定义。只有位置 RMS ≤ 5 mm 且旋转 RMS ≤ 3° 的结果可以保存；未达标结果保留残差，X 按两种残差相对门限的最大值删除最差样本，之后需重新求解。

桌面拟合与质量评估位于 [`geometry/plane.py`](dexmani_real/geometry/plane.py)，路径解析和标定读写位于 [`calibration/table_plane.py`](dexmani_real/calibration/table_plane.py)；VR 对齐与启动质量检查位于 [`calibration/vr_heading.py`](dexmani_real/calibration/vr_heading.py)。默认参数集中在 [`calibration/parameters/`](dexmani_real/calibration/parameters/)，路径常量由 `dexmani_real.calibration` 提供，消费者需要时才读取。

相机外参 JSON 以 serial 为键，条目包含 `T_base_from_camera` 四阶矩阵（相机到基座，平移单位米）和可选的 `calibration_capture` 元数据。`load_camera_extrinsics()` 返回 serial 到矩阵的字典；读写使用同一文件校验，更新一台相机时保留其他条目。可选录制仅允许默认文件缺失，损坏文件、显式路径错误或文件中缺少当前相机 serial 均直接报错。

实验配置使用 `from dexmani_real.config import RealConfig, load_config`，覆盖优先级为 CLI > YAML > dataclass 默认值。`load_config(path=None, overrides={"execution": {"mode": "async"}})` 使用嵌套字典；覆盖中的 `None` 表示未指定，YAML 的 `null` 保留原意。执行方式由 `execution.mode` 指定；CLI 使用 `--execution-mode`。配置加载不读取标定或连接设备，运行约束在使用边界验证。`environment.table.plane_path` 的相对路径以仓库根目录解析；显式指定的必需标定缺失时直接报错。桌面读取使用 `load_table_plane(table)`，由调用方决定是否需要加载。VR 标定模块提供 `collect_headings()`、`estimate_heading()` 和 `save_vr_transform()`；脚本只保存合格结果，保存成功后播放完成提示音。相机标定前必须拆除 XHand 或将其固定在 HOME 姿态，脚本禁用手部驱动，碰撞检查统一使用固定 HOME 包络；采样期间持续检查关节速度，并核对前后姿态漂移。

机器人会话在 preflight 统一检查设备、HOME、工作空间及实际启用的碰撞配置；相机及 XHand 独立工具在自己的入口检查。内部 driver、controller 和计算函数接收已经通过入口检查的配置与标定，不重复预检。新的 SDK 反馈、模型输出、文件数据及最终发送目标仍在各自边界校验。`run_policy.py --print-config` 展示与运行快照相同的八个配置分区，数值仍为声明值；`run_policy()` 在加载策略前检查运行轮数和时长。

## 代码导航

核心实现位于 [`dexmani_real/`](dexmani_real/)，实验入口与设备工具集中在 [`scripts/`](scripts/)；[`assets/`](assets/) 存放机器人模型等资源，[`third_party/`](third_party/) 包含第三方相关代码。

部署沿 `deployment/run.py → session.py → rollout.py` 执行：`config.py` 解析与校验执行配置，`PolicySession` 持有设备和模型，`Rollout` 持有单轮录制器及唯一的 `EpisodeResult`。`schedule.py` 通过 `Prediction` 和 `ActionChunk` 表达 sync/async/RTC 调度；`ObservationBuilder` 集中管理字段依赖、FK 与相机参数，派生特征随观测行释放。`PolicyWorker` 在单线程中串行加载、重置、预测和关闭模型，只允许一个未回收任务。来源时效在观测入口检查，策略输入的数值与形状在 `dexmani_policy` 边界检查。

部署先初始化配置与几何，再启动键盘、预热模型，最后连接设备；启动期间 S/Q 取消实验，Esc 按故障退出，H/B/T 不排队执行。启动姿态的机械臂容差固定为 0.5°，预测等待固定为 2 s。运行和等待使用绝对截止时间，不补发错过的控制步。终止按撤销权限、停止、追加最后一帧、录制收尾、登记结果的顺序执行；模型任务或 writer 尚未结束时，所属轮次继续持有资源。

sync 按计划周期采样并顺序使用动作，推理完成只提交本步结果，不再次采样或追加 history；超周期后移到下一未来网格点，不突发补发。普通 async 按实际 slot 跳过过时动作，迟到预测仍按原观测锚点索引；有效尾部可继续执行，耗尽或过期仍停止。RTC 保留严格交接时刻、冻结前缀和执行前复核，错过 slot 仍停止。动作历史记录已提交的命令目标，不表示硬件已经到位。

回放沿 `replay/run.py → joint_replay.py` 执行，`JointReplay` 负责逐帧提交，`trajectory.py` 读取轨迹并缓存、保存实测数据，`metrics.py` 只计算匹配帧的关节与末端误差。运行入口在硬件收尾后保存 NPZ、计算指标并写入 JSON；指标失败时原始采样仍可用于离线评估。

策略选择交给 Policy 的公开 `inspect_policy`：Real 默认 latest，也支持 best、20pct 等 milestone 和显式 checkpoint 路径。显式相对路径以当前工作目录解析；weights/NFE 的显式参数优先。执行长度仍取 execution 配置或模型 n_action_steps。

`control/constraints.py` 统一等价角、运行限位和 `MotionConstraints` 端点约束。目标不再受固定跳变角或目标距离阈值限制；Ruckig 负责速度和加速度约束，保留包含制动段的位置极值检查、输出有限性和参考—实测跟踪检查。端点碰撞检查不保证连续路径无碰撞。`control/ik.py` 的 `IKSolver` 接收共享约束、位置和旋转容差，`solve(..., hand_qpos=...)` 显式指定碰撞手姿；动作映射、键盘 jog 和相机标定共用这一接口。

IK 保留上一命令、实测状态及正负零空间初值；评分只使用关节距离和限位惩罚，搜索权重与步长固定在算法模块。位姿误差用于容差准入，SVD 只用于生成零空间方向。候选 FK 同时用于误差和工作空间检查，碰撞模型统一管理自碰撞、静态障碍和可选桌面。`HomePlanner(environment, arm, table_plane_abcd=...)` 与执行器共用 `ArmConfig` 的运行限位和到位容差，桌面间隙参数由 `EnvironmentConfig` 提供；`plan(current, target)` 保留三种关节路径、1° 采样及绝对等价分支对齐。无需移动的 HOME 仍检查几何约束，并由实测反馈确认停稳。

`geometry` 中，`camera.py` 负责内参与反投影，`pose.py` 负责位姿与刚体变换，`kinematics.py` 提供 `ArmFK`、`HandFK` 和批量 `eef_poses`，`plane.py` 拟合桌面。`perception/pointcloud.py` 的 `PointCloudProcessor` 在初始化时计算基座系射线和桌面高度系数，逐帧通过 `process(depth_raw, color)` 完成深度过滤、裁剪、体素均值、半径去噪和 QuickFPS。每个相机或 episode 复用一个处理器，几何参数变化时重新构造。`hand_cloud.py` 提供 `HandPointCloud(hand_fk, num_points)` 和 `hand_geometry`，指尖与手点云每行共用一次 FK；`worker.py` 发布在线点云。

`perception/config.py` 只保留点云点数、工作空间、桌面开关与余量，以及手点云点数。`table_margin_m` 是点到向上桌面平面的最小距离，默认 7 mm；该范围内的低矮物体表面也会被裁掉。深度范围 0.30–1.50 m、断层阈值 30 mm、表面带宽 8 mm、体素 5 mm、去噪半径 12 mm 和最少 6 个邻居固定在算法模块。点数不足时循环已有点，空点云在线不发布、离线导出报错。

手表面按当前 XHand URDF 的单个本地网格、零 visual 位姿和内嵌颜色读取，这些资产前提由测试检查；采样按面积分配，按 link 顺序使用从 0 开始的固定种子。Canonical 从每个 episode 的相机快照统一派生点云与相机参数，并保存保留的实验参数，部署按保存参数恢复。未知 YAML 参数直接报错。

`scripts/tools/view_pointcloud.py` 提供诊断快照与性能测量；快照直接写入独立目录，失败时可能保留部分文件。`scripts/tools/view_realsense.py` 提供实时预览，点云计算、标定读取或设备故障会退出并报告失败；退出时停止键盘监听、断开 viewer、恢复原曝光优先级并断开相机，恢复失败也会报告失败。

单阈值裁剪对桌面标定残差更敏感。可根据实验物体高度和当前标定调整 YAML 中的 `pointcloud.table_margin_m`；增大余量会减少桌面残留，也会裁掉更多低矮表面。离线导出和查看在配置入口校验手部安装位姿，指尖名称与 SDK→URDF 关节顺序由 `HandFK` 核对。

XHand 的 SDK 载荷解析、错误码和板卡诊断位于 `robot/xhand/driver.py`，标准化状态只含关节与触觉，不读取未使用的 torque。反馈布局与缓存位于 `feedback.py`；机械臂反馈布局位于 `robot/xarm7/protocol.py`。`Robot.send_action()` 在任一执行器发生副作用前预检组合目标，驱动内部发送路径接收已验证数组；HOME、独立手部工具和 stop/hold 各自覆盖直接入口。

机械臂由 `robot/xarm7/process.py` 的 `ArmProcess` 提交目标，`worker.py` 的 `ArmController` 独占 SDK 与默认 100 Hz 伺服，`trajectory.py` 的 `ArmTrajectory.step(now_ns, goal=None)` 用 Ruckig 在每周期采样一次连续参考。共享通道及操作消息集中在 `protocol.py`，`ArmChannels` 负责目标发布、消费和状态快照；HOME 总截止时间从运行层传到父、子进程，取消使用同一个停止信号。操作回复直接携带超时、取消或设备异常，保留失败原因；停止失败始终作为设备故障报告。

硬件反馈、策略采样/推理和机械臂伺服使用各自节拍。机械臂子进程按 `arm.feedback_hz` 发布反馈；父进程的 SDK 所有者线程以 30 Hz 服务 XHand，并在模型等待、HOME 和空闲时继续处理反馈与取消。策略只消费缓存，5 Hz 策略不再因反馈年龄预算被拒绝。缓存读取、CRC 和健康查询不刷新来源时间；硬件健康期限与策略、VR、jog 命令的执行期限分别检查。

控制与设备配置类使用 `MotionConfig`、`ArmConfig`、`HomingConfig`、`HandConfig`。SDK 超时和伺服间隔预算均固定为 20 ms，进程等待为 3 s；HOME 的到位附加预算、稳定持续时间、收敛等待分别为 0.5、0.3、15 s，轮询周期使用机械臂反馈频率。XHand 串口稳定等待和初始反馈等待均为 1 s。这些预算不再接受 YAML 覆盖；机械限位和指尖名称由 `robot/model.py` 提供，实验仍可配置运行关节限位、控制频率、HOME 姿态、收敛容差及硬件增益。

桌面代理检查的 YAML 字段为 `environment.table_z_surface_m`（默认 0.022 m）和 `environment.hand_safety_margin_m`（默认 0.05 m）。标定桌面的 mesh 距离检查仍使用 `environment.table.soft_clearance_m`。

机器人同步操作及 XHand 诊断脚本的取消统一抛出 `OperationCancelled`，运行边界以 `RobotError` 报告设备故障；初始化和收尾保留原始失败原因。SDK 原生输出直接显示。

`CartesianJog.submit()` 负责输入代际、运动授权和提交后参考更新。`EpisodeRecorder` 使用 `start()`、`append()`、`finish()`，收尾时显式指定保存意图、终止原因和技术状态，`result()` 返回最后可观察的录制结果。

`sensor` 平铺为 `config.py`、`realsense.py` 和 `vr.py`，进程入口为 `run_camera` 和 `run_vr`，SDK 在实际使用时导入。配置类型为 `CameraConfig`、`VRConfig`；相机预热固定为 10 帧，来源停滞期限为 2 秒，这两项不接受 YAML 覆盖。`RealSenseCamera.read()` 返回 `RGBDFrame`，单次读取等待固定为 300 ms，短暂等待超时或重复帧返回 `None`，持续停滞或 SDK 故障抛出异常。驱动统一维护来源时间与去重，`depth` 为对齐到 color 网格的原始深度，`color_intrinsics`、`depth_scale_m` 在连接成功后直接读取。

`runtime` 按职责平铺为 `state`、`keyboard`、`observation`、`lifecycle`、`preflight`、`home` 和 `jog`。`SessionState` 管理运行代际与停止请求，`SensorWorkers` 持有传感进程，`close_runtime()` 按顺序回收资源。`home_hand()`、`home_arm()` 直接使用机器人对象中的设备配置，并分别负责本次运动的停止；`home_robot()` 顺序执行两者。

`Keyboard(bindings={...})` 将离散按键绑定到会话操作，ESC 独立锁存急停。`Keyboard.poll()` 返回小写按键字符串；命令、标定和预览共用一个事件队列。离散操作每次按下只触发一次，长按 jog 仍连续移动，松键会使旧提案失效。每个键盘实例只启动一次，`discard_events(*keys)` 用于丢弃阻塞操作期间的旧事件。

键盘遥操作与相机标定共用 `CartesianJog.step()`：准备、采样、IK、代际核对和提交位于同一路径，每个普通控制周期只读取一次机器人反馈，阻塞准备后再取新反馈。两者分别以观测时间和机械臂来源时间计算目标执行期限。输入代际在发布与首次 SDK 发送前核对；IK 拒绝或提交前输入失效时，当周期立即停止并清除目标参考。`pause()` 用于标定采样及保存前停止运动，并要求释放全部 jog 键后才能恢复。

`utils/io.py` 提供 `write_json()`、`write_yaml()` 和 `publish()`，共用临时文件清理与原子发布；JSON 固定保留中文并拒绝非有限数值，`overwrite=False` 使用 Linux `renameat2` 禁止覆盖。`Rate(hz)` 每周期至多休眠一次，小幅迟到保留原时钟网格，错过整周期则重新计时；HOME 后调用 `reset()`。

运行资源超时使用代码常量：相机、点云、VR 启动分别为 15、30、120 秒，正常退出等待预算为 65 秒，定义于 `runtime/lifecycle.py`；模型启动为 300 秒，定义于 `deployment/session.py`。部署清理共享同一截止时间；仍存活的传感进程依次执行 terminate、kill，各等待至多 1 秒，因此总清理时间可能超过正常退出预算。超时不属于 YAML 配置项，未知配置键会报错。

项目依赖与打包配置见 [`pyproject.toml`](pyproject.toml)，开发约定见 [`AGENTS.md`](AGENTS.md)。

## 运行接口与错误处理

遥操作入口为 `teleop.run.run_teleop(...)`，回放入口为 `replay.run.run_replay(...)`，对应脚本解析参数并调用入口。

回放输出 `replay_data.npz`（实测关节、候选及来源/提交时间戳）、`metrics.json`（按帧索引比较的误差）和 `session_result.json`（回放、回零与收尾结果）。候选和提交完成都不代表物理到位；手部准备不占回放行，准备时长保存在运行摘要中。零行指标使用 `null`，不表示零误差。摘要的 `status`、`reason` 表示会话最终结果，`trajectory_status` 独立保留回放阶段的结束状态；回零失败不会抹去已完成的回放状态。启动及准备阶段按 Q 取消也记为 `user_quit`；文件写入失败不会报告成功。

回放启动保留当前手部姿态。操作者应先摆好录像首帧姿态；新鲜实测关节与首帧相差超过 10° 时拒绝回放。`--hand-start-duration` 默认 0.5 秒，按“平滑插值 → 三个不同的新鲜反馈确认到位”准备首个手部目标；设为 0 可跳过准备。回放结束后支持 H 手动回零、S 停止和 Q 退出。

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

`TAGRetargeter` 集中实现两阶段目标函数和解析梯度，仅在优化器请求梯度时计算雅可比，复用 `HandFK` 的固定模型与关节顺序；`DexPilotRetargeter` 直接调用现有 DexPilot 目标函数、NLopt 和低通滤波器。退化手部几何不产生目标；仅 TAG 第二阶段捏合细化的 `RoundoffLimited` 使用本帧已通过有限性与限位处理的第一阶段解，并更新后续解状态。第一阶段及其他程序异常继续传播，不返回旧目标。显式提供的非法初值直接报错；省略初值仍从关节限位中点开始。

`control/actions.py` 提供 `map_joint` 与 `map_eef`，统一返回 `ActionResult`；关节拒绝仍保留诊断候选，发送入口必须检查 `rejection_reason`。IK 直接使用 MPlib 的 `PinocchioModel` 和入口提供的运行关节限位，以 `IKResult.qpos is None` 表示正常无解，求解器、FK 或碰撞后端故障直接抛出原始异常。HOME 每个路径点共用一次运动学更新完成碰撞、EEF 工作空间和桌面间距检查，仅对 `HomePathRejected` 表示的几何拒绝尝试其他候选路径，候选耗尽、取消或等待超时返回失败结果；非法目标与技术异常传播到会话错误处理和收尾边界。

XHand 保留 serial 与 EtherCAT，连接失败直接报告，不自动重试。CRC 故障变化仅由驱动记录：读 CRC 丢弃整个 payload，旧反馈及来源时间保持不变，过期则失败；普通发送遇到 CRC 不补发，下一周期发送最新目标。连续发送 CRC 达到 `hand.send_crc_limit`（默认 5）时报错；成功或既有非致命发送响应清零，读取、等待和阶段切换均不清零。正常返回允许继续尝试，动作历史中的提交目标不代表硬件确认执行。默认上限是未经硬件定标的工程起点。停止请求沿用独立的 0.25 s 预算，不受普通发送次数上限截断。

aggregate、dense 在 SDK 边界独立解析，非空输出均为准确形状、有限值的 `float32` 完整帧。不可用状态或格式错误使对应通道返回 `None`，不影响有效关节；`latest` 快照保留该通道最后完整有效帧及来源时间，新快照发布后只读。任一 taxel 无效都会使整个 dense 通道保持旧帧。初始化沿用反馈重置时固定的截止时间，等待控制和录制实际必需的触觉；运行中由消费者检查来源时效。Raw 触觉仅支持 `sdk_native_bounded_previous_valid_channel_frame_hold` 标记，其他处理语义明确拒绝。未知 SDK 错误和明确关节／通信板故障继续传播；诊断脚本读取停止反馈失败时仍尝试停止，并传播原始错误。连接清理失败不会覆盖初始错误，未释放的句柄仍由设备对象保留。

回放先保存实测数据，再计算并写入指标。没有有效样本的指标显式记为 `null`；计算产生的非有限数值由 JSON 写入边界拒绝并报告评估失败，保留已保存的实测数据。

回放结束后，H 请求回初始位，S 取消 HOME，Q 退出，Esc 急停。S 优先于同批次 H；完成停止确认后可重新按 H。HOME 期间积累的 H 不会在返回后自动执行，S 中断记录为取消。

## 观测与产物契约

传感器 IPC 布局和通道分配位于 [`ipc/sensors.py`](dexmani_real/ipc/sensors.py)，[`SharedRing`](dexmani_real/ipc/ring.py) 面向 Linux x86_64，通过 `spawn` 传给单个生产者和多个消费者。`write(**fields)` 返回发布序列；`read(sequence=None, fields=None)` 返回 `(data, sequence)`，完整帧为结构化标量，按字段读取为独立副本字典。空缓冲、目标序列被覆盖或复制期间发生覆盖均返回 `None`，不重试、不缓存旧帧。

来源时间由载荷携带，IPC 不生成替代来源时间的发布时钟。点云携带 `source_camera_sequence` 和原始相机 `timestamp_ns`；同时需要像素与点云时，只读取对应相机序列，匹配失败视为观测缺失。控制所需观测缺失时，遥操作或部署按现有准入规则结束本轮；标定采样继续在自身期限内等待。

`ipc/sensors.py` 提供 `read_camera_frame(fields=("rgb", "depth"))`、`read_vr_frame()` 和 `snapshot_camera_parameters()`。相机读取返回来源 `timestamp_ns`、共享内存发布序列 `ring_sequence`、RGB 设备帧号 `color_frame_number` 及请求的像素通道。发布序列不是设备帧身份；时间戳保留 RGB-D 中较早通道的首次主机接收时间，不代表曝光时刻。标定采样和预览只读取 RGB，采样仍要求屏障后的新鲜、不同 RGB 帧。

`ObservationNeeds.camera_fields` 直接声明所需像素字段；静态参数需求独立声明，部署入口一次构造依赖和 needs 并传给 session、builder、rollout。`read_observation()` 读取反馈缓存与传感器快照，不触发同一轮硬件查询；手部初始化等待由准备阶段显式完成。`check_observation()` 统一返回来源问题与提交期限。相机尺寸和点云有限性在生产者发布前检查，IPC 保留字段完整性、复制前后序列核对。观测与策略历史使用已脱离共享缓冲的数组，提交给推理或录制后，生产者不再修改这些快照。开启录制仍要求 RGB-D、聚合和密集触觉；推理等待期间的来源或 writer 故障仍终止控制。

`termination_reason` 保留首次终止原因，Teleop 的准备与运行阶段均使用 `RunEndReason` 的小写名称，具体中止上下文写入 `termination_details`。`technical_status` 反映该次产物的技术完成状态，发布状态单独描述文件是否已发布。控制中止后的完整记录可以保存为 `failed` episode；录制队列满、写入或关闭失败则报错并保留临时目录，不发布正式 episode。Teleop 的 D 键回调先锁存丢弃意图并撤销运动，再发布按键事件；收尾请求前到达的 DISCARD 在准备、求解或停止失败时仍保持丢弃意图，writer 退出后删除临时目录。已发布 Raw 不因后续独立会话清理错误而回写。

Teleop 中断会登记技术失败并传播，关闭时继续按原保存或丢弃意图收尾；recorder 的首次收尾请求不会被重入覆盖。收尾超时仍报告 pending，不在同一次关闭中重新开启等待预算。Policy 即使最后一帧入队失败也会请求录制收尾；轮次结束后才返回的模型异常仍登记在该轮 errors 中，不覆盖首次终止原因。模型关闭异常保留摘要及日志中的原始 traceback。会话关闭末尾核对对应 episode 的 Raw 发布状态、实际路径和帧数，再写入 `session_result.json`；该摘要反映最后一次确认，不保证仍 pending 的 writer 之后的结果。

部署的 `run_config.yaml` 保留策略元数据、实际运行配置、seed/device、轮数与时长、录制开关及有效标定；runtime 仅包含 execution、arm、hand、motion、camera、pointcloud、hand_pointcloud、environment，不采集 Git 版本或文件哈希。`session_result.json` 保留会话起止 UTC、outcome、reason、shutdown、errors 和 episodes；每轮包含序号、开始 UTC、时长、终止原因与详情、errors、raw 和 metrics。metrics 仅包含成功预测的次数、平均/最大耗时及跳过步数；`inference_mean_s`/`inference_max_s` 从观测构造前计时，覆盖 FK、手点云、数组组装、模型推理直至返回 CPU 动作。错误统一为 stage、exception_type、message；shutdown 取 null/pending/failed/completed，单轮技术错误、清理错误、pending 或结果写入失败均返回非零退出码；S/Q 导致的正常取消不记作技术错误。设备、键盘与传感器的清理错误写入会话 errors；一项清理失败或被中断后仍继续回收其他资源。

`data/raw.py` 集中提供 `RawReader`、`RawWriter` 和元数据编解码。Raw 以 `timing/rows.observation_ns` 为唯一观测时钟，以该表长度作为帧数。`timestamps_s` 在整数时钟相减后转换为相对秒数，空录制返回空数组；反复访问时间轴的调用方应在本地保存该数组。

`RawReader` 首次读取时校验数值字段和 `timing/rows` 的布局，并缓存数据集。`read_camera()` 返回 `CameraSnapshot`，`read_timing_rows(fields)` 保留原始 `int64` 纳秒和 `-1` 缺失标记，文件与视频由 Reader 统一关闭。物理回放仍显式调用 `validate_completion()`、`validate_rows()`，严格要求正常录制与完整的观测节拍；训练导出使用下述独立准入。CordViP 的 Raw 行关联不证明手和相机物理同步；缺失相机时钟时显式使用观测时钟，原始相机时钟仍保留 `-1`，不满足训练导出的来源时间要求。

Raw 写入缓冲按 8 MiB 预算选择 1–32 帧，单帧超过预算时保留一帧。`RawWriter.finish()` 刷新行缓冲、结束视频编码并写入终止详情，再写入技术状态；`close()` 释放资源。录制器在成功收尾后原子发布，写入或收尾失败时保留临时产物并报告错误。

Canonical 由 `data/export.py` 的 `export_zarr(..., processing=ExportConfig(...))` 导出，提供 13 个训练字段、episode 边界及物理参数。导出遇到损坏或不满足训练要求的 episode 立即报错，保留已有临时结果；使用 `--exclude EPISODE` 显式排除后重新导出。最终输出必须尚未占用且不与输入目录重叠，全部完成后才原子发布。处理批次根据 64 MiB 数组预算和最多 100 帧确定，不再提供 `--chunk-frames`；存储块仍按字段大小独立选择。

训练时间准入以 `control_ns` 判断连续周期，控制间隔与累计相位仍限制在 `dt/4` 内；`observation_ns` 保留读取完成时间，允许一个周期内且不越过下一控制起点的局部读取抖动，例如 16 Hz 下单次 20 ms 延迟。持续控制频率偏移、真实缺步、失效来源时间或非有限行会在首个问题处截断。每条 Raw 最多导出从第零行开始的一个连续前缀，所有观测、动作、RGB-D 和派生字段使用相同行范围，不补帧、不重采样。技术失败的 Raw 未逐行保存命令提交结果，因此保守排除最后一行，原始 `failed` 状态保持不变。空前缀拒绝；字段、形状、单位及视频整段帧数等结构损坏仍拒绝整条。Zarr 的 `episode_quality` 报告 `[row_start, row_stop)`、裁切原因、控制网格偏差及读取耗时；`readout_over_tolerance_rows` 统计 `observation_ns - control_ns > dt/4` 的行数，仅用于诊断，不作为拒绝条件。`timing_admission` 描述实际准入规则。

## 离线验证

在已安装项目及相关依赖的 Python 环境中运行：

```bash
python -m compileall -q dexmani_real scripts
python scripts/run_policy.py --print-config
python scripts/collect_teleop.py --print-config
python -m pytest -q tests
git diff --check
```

已有 Ruff/mypy 的开发环境可补充静态检查，无需为运行实验安装它们：

```bash
ruff check --no-cache --isolated --select E9,F dexmani_real scripts tests third_party/cordvip_geometry/src
mypy --cache-dir=/dev/null --ignore-missing-imports --follow-imports silent --explicit-package-bases \
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

键盘遥操作入口保留 `--help`，没有 `--print-config` 参数。

真机启停、取消/急停、HOME/jog、遥操作录制、三种策略模式和回放仍需操作者单独验收。软件授权撤销和 SDK 接受请求不代表物理停稳或到位。

默认标定 JSON 随 wheel 打包；机器人 `assets/` 仍沿用 checkout 定位方式，验证 wheel 中的 JSON 不等于验证了完整独立安装的机器人运行环境。
