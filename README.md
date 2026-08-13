# DexMani Real

> 面向 **xArm7（7 自由度）+ XHand（12 自由度）+ Quest VR + Intel RealSense L515** 的 VR 遥操作、数据采集、策略部署与轨迹回放系统。

DexMani Real 将硬件能力封装在独立进程中，以共享内存传递结构化状态和指令；主进程只负责创建资源、检查就绪、监督健康状态与有序退出。这样既避免跨进程共享 SDK 对象，也将控制、记录和安全职责放在明确的领域模块中。

## 目录

- [系统概览](#系统概览)
- [运行模型与数据流](#运行模型与数据流)
- [从入口到核心模块](#从入口到核心模块)
- [环境与安全边界](#环境与安全边界)
- [项目地图：`dexmani_real`](#项目地图dexmani_real)
- [项目地图：`examples`](#项目地图examples)
- [配置、资源与延伸文档](#配置资源与延伸文档)

## 系统概览

项目覆盖四条相互衔接的工作流：

1. **VR 遥操作与采集**：读取 Quest、相机和机器人状态，生成安全的臂/手指令，并按固定控制网格记录 episode。
2. **实验性学习策略部署**：仅在提供外部 adapter、PolicySpec、模型资源和离线验证后启用隔离推理链；默认遥操作运行时不分配其 IPC。
3. **标定与诊断**：标定相机外参和 VR 朝向；以受限、可观测的方式诊断 RealSense、点云和 XHand。
4. **回放与离线分析**：检查 HDF5 episode、在启动前直接执行密集预检、运行受控 live replay，并评估或可视化数据质量。

```text
                    ┌───────────── sensor/ ──────────────┐
                    │  RealSense / VR receiver / pointcloud│
                    └──────────────┬──────────────────────┘
                                   │ state frames
                                   ▼
┌──────── robot/ ────────┐   ┌──────────────── shm/ ────────────────┐
│ xArm loop / XHand loop │◀──│ SharedStorage · typed rings · queues │
└──────────▲─────────────┘   └─────────────────┬────────────────────┘
           │ arm queue / hand ring               │ snapshots
           │                                     ▼
           │                         ┌──── teleop/ 或 policy/ ────┐
           └─────────────────────────│ 映射、IK、动作校验、决策时钟 │
                                     └──────────────┬─────────────┘
                                                    │ aligned samples
                                                    ▼
                                    ┌────────── recording/ ──────────┐
                                    │ RecorderIO → HDF5 episode v16  │
                                    └───────────────────┬───────────┘
                                                        ▼
                                               examples/visualize_episode.py
```

### 设计边界

- **IPC 边界**：所有跨进程数据经 `SharedStorage` 传递；有效负载由 `utils/schema.py` 的固定 NumPy dtype 定义。
- **硬件边界**：xArm/XHand SDK 仅由各自的执行 worker 使用，RealSense SDK 由 `sensor/` 持有；其他进程不共享活的 SDK 对象。
- **控制边界**：策略/遥操作 worker 决定动作与采样网格；`RecorderIO` 只负责序列化、校验和事务式发布。录制的 START/STOP、状态与每格元数据使用固定 dtype，不在共享内存中嵌入 JSON；v16 episode 保存已解析配置的 SHA-256，而非整份配置文本。
- **安全边界**：`SafetyState` 管理 `DISARMED → ARMED → RUNNING → FAULT`；固件仍是最后一道安全保护。

## 运行模型与数据流

### 进程职责

标准遥操作运行时包含五个控制/设备 worker；开始录制后增加一个 `RecorderIO` worker：

```text
camera ────────┐
VR ────────────┼──► teleop / policy ──► arm endpoint/HOME queue ──► arm worker
                           │              └──► priority STOP/RESUME ring ───┘
arm state ─────┤          │
hand state ────┤          └────────────► hand ring ──► hand worker
               │
               └──► shared-memory state

teleop / policy ──► fixed-grid sample ring ──► RecorderIO ──► HDF5
```

| 通道 | 语义 | 关键约束 |
|---|---|---|
| 臂动作队列 | 有序、短队列的未来关节目标 | `maxsize=2` 的反压是有意设计 |
| 臂控制环 | 松键 STOP 与显式 RESUME | 固定 dtype、latest-wins，worker 优先于端点队列读取 |
| 手动作环 | 最新目标覆盖旧目标 | latest-wins，避免手部控制滞后 |
| 状态环 | 相机、VR、臂、手的共享快照 | seqlock 验证读，跨进程不传可变对象图 |
| 录制采样环 | 对齐后的机器人、动作与传感器样本 | 固定为 `1 / control_hz` 网格，不以到达时间采样 |

### 关键路径

| 使用场景 | 入口 | 主要调用链 |
|---|---|---|
| VR 采集 | `examples/collect_teleop.py` | `teleop/loop.py` → `planning/`、`robot/`、`recording/`（实验生命周期自包含在 examples 中）|
| 键盘控制 | `examples/keyboard_teleop.py` | `teleop/keyboard.py` → State-6 松键刹停 / 实测位姿显式恢复协议 → 安全动作协议 |
| 实验性学习策略 | `examples/deploy_policy.py` | 自包含入口 → `inference_process.py` → `learned_coordinator.py`（部署生命周期自包含在 examples 中）|
| Episode 回放 | `examples/replay_episode.py` | — | Self-contained script; dry-run by default; `--live` reruns dense preflight |
| 相机标定 | `examples/calibrate_camera.py` | 自包含 ArUco 手眼标定；会采集设备数据并原子写入 cameras.json |
| 离线数据分析 | `examples/visualize_episode.py` | Rerun 3D episode 可视化；`python examples/visualize_episode.py <episode>` |

## 从入口到核心模块

建议按下面顺序阅读代码；每一层只建立在前面层的稳定接口上。

1. **配置与协议**：先读 `config/defaults.py`、`config/runtime.py`、`utils/schema.py`，了解默认参数与跨进程数据形状。
2. **数据平面与生命周期**：再读 `shm/shared_storage.py`、`shm/ring_buffer.py`、`runtime/supervisor.py`，了解进程如何共享数据、就绪和停止。
3. **设备和运动能力**：阅读 `sensor/`、`robot/` 与 `planning/`，它们分别产生观测、执行动作、计算 FK/IK/碰撞和路径。
4. **业务控制环**：`teleop/` 是 VR 控制和记录决策中心；`policy/` 是学习策略的隔离推理与动作调度中心。
5. **持久化和事后工作流**：`recording/` 写入/读取 episode；`examples/replay_episode.py` 和 `examples/visualize_episode.py` 消费这些数据。

## 环境与安全边界

项目目标环境是 Python 3.10 的 conda 环境 `real_robot`。`pyproject.toml` 管理可移植 Python 依赖；Pinocchio、MPlib、CUDA、RealSense、xArm/XHand SDK 等原生/设备依赖仍由该环境管理。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate real_robot
export PYTHONPATH=.

# 不连接任何硬件的语法检查
python -m compileall -q dexmani_real examples
```

> [!WARNING]
> `examples/` 下的所有入口都可能影响硬件。不要仅为“试一下”而运行遥操作、回放、标定或设备诊断；执行前应确认工作空间清空、设备状态正常，并获得相应操作授权。`replay_episode.py` 默认为离线检查，但 `--live` 会跨越硬件安全边界。

## 项目地图：`dexmani_real`

以下清单覆盖当前包内的 **92 个 Python 源文件**（包含各包的 `__init__.py`）。除根包外，表中路径均相对于 `dexmani_real/`；`__init__.py` 若只负责导出接口，也会单独列出，便于从导入路径反查实现位置。

### 根包

| 文件 | 作用 |
|---|---|
| `dexmani_real/__init__.py` | 定义包级路径 `PACKAGE_DIR`、`ASSET_DIR`，并概述机器人、遥操作、规划、录制和传感器子系统。 |

### `config/` — 默认值、运行时快照与标定数据

| 文件 | 作用 |
|---|---|
| `config/__init__.py` | 导出运行时配置解析器、不可变配置节点和环境碰撞配置。 |
| `config/camera_calib.py` | 加载相机外参，按物理序列号校验条目，并统一 eye-to-hand / eye-in-hand 坐标变换。 |
| `config/defaults.py` | 所有数值默认值的单一来源：臂、手、VR、相机、策略、键盘、安全、碰撞与录制参数。 |
| `config/runtime.py` | 将默认 dataclass 与 YAML/点路径覆盖合并、验证并冻结为可跨进程使用的运行时配置快照。 |

### `planning/` — 运动学、IK、碰撞与轨迹规划

| 文件 | 作用 |
|---|---|
| `planning/__init__.py` | 导出规划器、位姿和规划配置/运行 profile。 |
| `planning/collision_model.py` | 基于 Pinocchio 的 xArm7+XHand 自碰撞/环境碰撞模型；把标定桌面构造成倾斜碰撞几何，允许基座安装接触，并提供真实网格距离与路径段检测。 |
| `planning/constants.py` | 集中定义 XHand SDK 关节序与 URDF/Pinocchio 关节序之间的重排常量。 |
| `planning/hand_kinematics.py` | 用 Pinocchio 计算手部正向运动学和五指指尖位置，并处理关节顺序重映射。 |
| `planning/ik.py` | 遥操作位置 IK 求解器：确定性种子、快速接受、多候选回退、误差/碰撞检查与零空间优化。 |
| `planning/ik_candidates.py` | 生成、过滤、评分和规范化 IK 候选，输出结构化拒绝原因诊断。 |
| `planning/kinematics.py` | 提供不依赖 MPlib 的臂 FK，以及带 MPlib 集成的完整 xArm7 运动学/雅可比能力。 |
| `planning/path_utils.py` | 关节路径插值、稠密化、角度等价包裹，以及两阶段回零和 band 对齐路径规划。 |
| `planning/planner.py` | `XArm7MotionPlanner`：组合 MPlib、运动学、IK 候选和碰撞模型以完成 IK/路径规划。 |
| `planning/pose_utils.py` | 位姿组合/求逆、四元数与 rot6d 转换、关节形状检查和位置姿态误差计算。 |
| `planning/types.py` | 定义 `Pose`、IK/路径结果、碰撞信息以及规划/遥操作 profile 等核心数据类。 |

### `policy/` — 动作协议与实验性学习策略部署

| 文件 | 作用 |
|---|---|
| `policy/safety.py` | 单一安全门 (SafetyGate) — 良构→关节限位→工作空间；速度包络与碰撞/过渡几何检查已移除（2026-08-12，由 xArm Mode 6 固件兜底，回零路径经 `plan_joint_home_path`/`plan_band_alignment_path` 独立规划碰撞），不裁剪 action；机械臂 STOP/RESUME 使用不与两槽端点队列争用容量的高优先级固定 dtype 环；hand-home 会生成显式合法里程碑并逐条等待 SDK 接受回执。 |
| `policy/inference_process.py` | 隔离推理 worker，加载 adapter，编解码单个当前 tick 候选动作，并验证模型输出是否满足策略契约。 |
| `policy/learned_coordinator.py` | 以单一时钟协调 observation、当前 tick 推理结果、动作执行和退出前 hold 的学习策略控制环。 |
| `policy/loop_timing.py` | 以滑动窗口统计控制环各阶段耗时的轻量 `StageTimer`。 |
| `policy/observation.py` | 构建不可变、因果一致的 observation 快照，防止推理读取到混合时刻的数据。 |
| `policy/observation_sources.py` | 将共享状态环字段映射为策略观测来源，并校验容量、dtype、形状与帧有效性。 |
| `policy/runtime.py` | 定义观测模态、观测/动作规格、冻结数组映射、带 run generation 的观测快照和单 tick 动作候选的数据契约。 |
| `policy/spec.py` | 加载并校验策略 YAML、模型资源 SHA-256、观测与动作规格，形成不可变 `PolicySpec`。 |
| `policy/tensor_block.py` | 将 `ObservationSpec` 映射为固定 dtype 的共享 observation tensor block。 |

学习策略部署中，adapter 每次只向共享 mailbox 写入一个当前 tick 候选。协调器仅接受与
当前 `run_generation` 及其 observation 一致且未过期的最新候选，随后才分配 action ID、执行
SafetyGate 校验并发布。开始、暂停、回零、反馈故障和相机重新预热都会推进该 generation；
模型原生 action chunk 必须在 adapter 内部收敛。候选的 `valid_until_monotonic_ns` 负责新鲜度，
实际 worker target 由发布边界生成。

### `recording/` — Episode 持久化与离线分析

| 文件 | 作用 |
|---|---|
| `recording/__init__.py` | 导出 episode 读写器、时间信息和停止结果的公共接口。 |
| `recording/camera_stream_writer.py` | 在独立写线程中编码并写入相机流，隔离视频 I/O 以免阻塞控制环。 |
| `recording/episode_reader.py` | 读取已原子发布的 v16 episode、合并流和元数据，并提供时间/有效性视图。 |
| `recording/episode_recorder.py` | 管理单个 episode 的 HDF5 数据集、相机写入器、停止校验与最终发布。 |
| `recording/io_process.py` | `RecorderIO` worker 及其客户端协议；以固定 start/stop 与样本 dtype 消费对齐环，再驱动记录器。 |
| `recording/timestamp_buffer.py` | 按目标时间戳插值、前向填充和标记缺口原因，保证采样网格对齐。 |
| `recording/transaction.py` | 目录 fsync 和原子发布工具，避免半成品 episode 被当作完成数据。 |
| `recording/video_codec.py` | 基于 PyAV 的视频编码器/解码器及其配置，服务 HDF5 旁路视频流。 |

### `examples/replay_episode.py` — 检查、授权与受控回放

Episode 回放功能整体位于单一自包含脚本 `examples/replay_episode.py` 中（约 2100 行）。默认执行离线检查（dry-run）；`--live --source sent` 跨越硬件安全边界，在启动 arm/hand worker 前执行密集几何和来源预检，通过 `SharedStorage` 回放轨迹，捕获回放状态并计算关节/末端跟踪一致性指标与时间延迟。

### `robot/` — xArm7、XHand 与安全状态

| 文件 | 作用 |
|---|---|
| `robot/__init__.py` | 标识 xArm7、XHand 驱动和执行 worker 所在包。 |
| `robot/arm_loop.py` | xArm Mode 6 伺服 worker：读取有序臂命令、执行带 ACK 的 State 6 减速停止（兼容固件上报 State 5/6）、发布 FK 状态，并处理 C24 恢复与碰撞故障；成功初始化时收敛 SDK 冗余输出，失败时保留原生诊断。 |
| `robot/hand_process.py` | XHand worker：读取 latest-wins 手指令、复核命令/机械限位、发布关节/触觉反馈与最后成功 action ID；不以目标—反馈不收敛判定故障。 |
| `robot/homing.py` | 执行并验证机械臂回零，包含状态/心跳检查、路径候选拒绝信息和 e-stop 处理。 |
| `robot/safety.py` | 定义 `SafetyState` 与合法状态迁移/强制迁移检查。 |
| `robot/types.py` | 定义文档化的机器人状态、动作、臂/手/触觉 dataclass；实际 IPC 格式由 `utils/schema.py` 决定。 |
| `robot/xhand.py` | 封装 XHand SDK 的连接、配置、关节/触觉读写和安全的资源释放；超过运行或厂商机械限位的命令整条拒绝，绝不隐式 clip，运行配置只能收紧而不能放宽额定机械包络；成功初始化时汇总原生 SDK 噪声，连接失败时回放完整诊断。 |

### `runtime/` — 进程生命周期与状态码

| 文件 | 作用 |
|---|---|
| `runtime/__init__.py` | 导出组件阶段、故障码和退出原因等运行时状态枚举。 |
| `runtime/processes.py` | 提供 spawn 上下文、进程退出报告，以及可验证的停止/回收/共享内存关闭流程。 |
| `runtime/status.py` | 定义跨模块使用的组件阶段、故障和退出原因的整数枚举。 |
| `runtime/session.py` | 提供 `ManagedProcessGroup`，封装进程组的启动、关闭与共享资源清理。 |
| `runtime/supervisor.py` | 完成 worker 就绪等待、心跳/进程监督、健康摘要和协调关闭。 |

### `sensor/` — 相机、点云与 VR 接收

| 文件 | 作用 |
|---|---|
| `sensor/__init__.py` | 标识 RealSense、VR 接收与点云处理能力所在包。 |
| `sensor/camera_process.py` | 相机 worker：采集、打包并发布 RGB-D 帧，维护相机健康状态和心跳。 |
| `sensor/clock_sync.py` | 将设备时钟映射到主机单调时钟，检测重置/漂移，供帧新鲜度判断使用。 |
| `sensor/pointcloud_processor.py` | 将 RGB-D、内外参和工作空间裁剪转换为采样/下采样后的点云观测。 |
| `sensor/realsense.py` | RealSense D400/L515 驱动：设备发现、配置、对齐、时钟映射、帧采集和生命周期管理。 |
| `sensor/vr_receiver_process.py` | Quest/VR 接收 worker：校验姿态与关键点数据、转换四元数顺序并发布 VR 帧。 |

### `shm/` — 共享内存数据平面

| 文件 | 作用 |
|---|---|
| `shm/__init__.py` | 说明共享内存公共接口及其与回零/监督模块的职责边界。 |
| `shm/ring_buffer.py` | 通用共享内存 seqlock 环和相机专用环，提供零拷贝写入与已验证读取。 |
| `shm/shared_storage.py` | 创建并持有共享环、队列、标志和事件；默认仅分配遥操作/采集能力，推理 IPC 需显式启用。 |

### `teleop/` — VR 映射、控制环与采集决策

| 文件 | 作用 |
|---|---|
| `teleop/__init__.py` | 标识 VR 遥操作、手部重定向、键盘和音频反馈子系统。 |
| `teleop/arm_mapper.py` | 将 VR 手腕位姿映射为受工作空间、旋转增量和四元数校验约束的臂末端目标。 |
| `teleop/audio_feedback.py` | 管理按键/运动门控下的音频提示播放与节流。 |
| `teleop/config.py` | 遥操作配置薄视图：仅持有 `runtime` 快照引用与 4 个会话专属字段（task_label/operator/hand_urdf_path/vr_transform_path），运行时值统一经 `config.runtime.<section>.<field>` 直读。 |
| `teleop/control_state.py` | 表示控制 hold 与回零交接状态，统一记录控制环暂停原因。 |
| `teleop/episode_samples.py` | 将因果状态、动作、VR/相机数据对齐为记录帧，并处理 start/stop/held 样本。 |
| `teleop/hand_control.py` | 对畸形重定向输出做 shape/finite 快速失败（区分「畸形」与「良构但被拒」以触发 hold）；关节限位与命令间增量校验归属 SafetyGate（控制器）与 worker/SDK 边界。 |
| `teleop/hand_retarget.py` | 校验手部 landmarks，并提供启发式 XHand 和 TAG 优化两类手部重定向器。 |
| `teleop/keyboard.py` | 处理终端/全局键盘输入、运动活动锁存、臂手反馈检查和末端位姿增量；终端输入抑制持续到设备进程退出，恢复终端时丢弃积压的 canonical 输入；停止回调后不为 Linux/XRecord 守护线程的延迟退出阻塞停机。 |
| `teleop/loop.py` | 核心 VR policy worker：读取快照、映射/IK、动作安全门、记录决策、状态机与错误恢复。 |
| `teleop/recording_session.py` | 处理退出时的保存、丢弃和停机决策。 |
| `teleop/safety.py` | 遥操作安全辅助：候选动作生效性、arm-only hold、接触停滞与回零流程（臂-手过渡碰撞检查已移除，由 Mode 6 固件兜底）；return-home 逐条确认有界 hand-home 里程碑已被 SDK 接受，但不等待手指角度收敛。 |
| `teleop/snapshot.py` | 从共享环读取同一因果锚点附近的臂、手、VR、触觉、相机快照，并跟踪相机新鲜度。 |
| `teleop/tag_retargeting/__init__.py` | 导出 TAG 两阶段手部重定向的优化器与 Pinocchio 梯度计算器。 |
| `teleop/tag_retargeting/optimizer.py` | 使用 NLopt 执行 TAG 手部两阶段优化，平衡指尖目标、关节限制与平滑性。 |
| `teleop/tag_retargeting/pin_grad.py` | 用 Pinocchio 计算指尖位置及雅可比/梯度，并校验指尖 frame 名称。 |

### `utils/` — 无领域耦合的通用工具

| 文件 | 作用 |
|---|---|
| `utils/__init__.py` | 标识日志、序列化、限速、信号、schema、数组和限位校验工具的公共包。 |
| `utils/array_utils.py` | 提供 NaN 初始化数组与安全 resize 等数值数组小工具。 |
| `utils/limits.py` | 校验 XHand 三级关节限位层级（rated ⊇ mechanical ⊇ command），收敛 config/robot/hand_process 三处重复的嵌套校验。 |
| `utils/log.py` | 创建统一 logger、可选文件日志和按时间节流的告警器。 |
| `utils/pointcloud_utils.py` | 实现内参/变换/工作空间校验、RGB-D 到点云、裁剪、下采样、采样与深度可视化。 |
| `utils/rate_manager.py` | 以单调时钟稳定控制循环频率，并报告周期统计信息。 |
| `utils/retry.py` | 提供可重置的连续失败计数器，供设备读写 watchdog 使用。 |
| `utils/schema.py` | 跨进程 NumPy dtype 与关节/末端尺寸常量的唯一定义源（原 `ipc/` 已合并至此）。 |
| `utils/serialization.py` | 按 dataclass 类型注解将字典安全转换为嵌套对象和 NumPy 数组。 |
| `utils/signal_utils.py` | 提供四元数安全归一化和位姿 EMA 平滑。 |

## 项目地图：`examples`

`examples/` 目前有 **10 个 Python 文件**。入口点专有逻辑（如实验生命周期、控制循环）直接放在 examples 中；共享库代码留在 `dexmani_real` 包内。

| 文件 | 调用的领域入口 | 作用与风险 |
|---|---|---|
| `examples/collect_teleop.py` | — | 标准 VR 遥操作与数据采集入口；实验生命周期自包含；会启动真实设备 worker。 |
| `examples/deploy_policy.py` | — | 实验性学习策略入口；部署生命周期自包含；需要外部 adapter/spec/模型并会进入真实执行器控制链。 |
| `examples/keyboard_teleop.py` | — | 以有界前视目标执行键盘 Cartesian jog（默认目标速度 0.24 m/s、最大前视 40 mm）；松键会使旧 generation 失效，并请求固件 State 6 减速停止，待 worker ACK 和连续两帧低速反馈后用实测位置重建参考，不发布可能导致反向回弹的滞后 hold 终点；R 会先确认 hand-home SDK 接受、再执行 arm home；终端输入抑制保持到 worker 完全退出；硬件相关。 |
| `examples/replay_episode.py` | — | episode 检查/回放入口；默认 dry-run，`--live` 会在启动 worker 前执行密集预检。 |
| `examples/calibrate_camera.py` | — | ArUco 眼到手标定入口；自包含脚本，会采集设备数据并原子写入 cameras.json。 |
| `examples/calibrate_vr_heading.py` | — | VR 朝向标定入口；自包含脚本，会读取 VR 数据并在确认后写入 vr_transform.json。 |
| `examples/realsense_record_example.py` | — | 交互式 RealSense RGB-D 实时采集与点云生成测试；默认只读。 |
| `examples/pointcloud_process_example.py` | `sensor.pointcloud_processor` | 生产点云管道诊断与桌面平面标定；显式确认后才写入标定。 |
| `examples/xhand_control_example.py` | — | 独立 XHand SDK 诊断；动作命令需显式硬件授权。 |
| `examples/visualize_episode.py` | — | 离线 Rerun 3D 可视化；读取 HDF5 episode 并展示点云、图像、动作、触觉和元数据；无硬件控制。 |

## 配置、资源与延伸文档

| 位置 | 内容 |
|---|---|

| `dexmani_real/config/cameras.json` | 物理相机序列号、类型和外参，是运行时校验的一部分。 |
| `dexmani_real/config/desk_plane.json` | 点云过滤、在线动作安全与回零路径共同使用的桌面平面标定数据。 |
| `dexmani_real/config/vr_transform.json` | VR 朝向标定得到的坐标变换运行数据。 |
| `assets/` | URDF/SRDF、网格、手部重定向配置和音频资源。 |
| `CLAUDE.md` | 更详细的架构、运行流程、安全/碰撞、录制 schema 与运维背景。 |
| `AGENTS.md` | 面向代码修改者的仓库约定、硬件安全边界和跨模块变更检查清单。 |

对于涉及 dtype、共享内存、录制 schema、IK/碰撞、安全状态机或速率默认值的改动，请先阅读 `AGENTS.md` 的“Cross-module change checklist”，再沿本 README 的关键路径追踪所有生产者和消费者。
