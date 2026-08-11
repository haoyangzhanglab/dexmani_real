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
┌──────── robot/ ────────┐   ┌───────────── shm/ + ipc/ ──────────────┐
│ xArm loop / XHand loop │◀──│  SharedStorage · typed rings · queues  │
└──────────▲─────────────┘   └──────────────────┬─────────────────────┘
           │ arm queue / hand ring               │ snapshots
           │                                     ▼
           │                         ┌──── teleop/ 或 policy/ ────┐
           └─────────────────────────│ 映射、IK、动作校验、决策时钟 │
                                     └──────────────┬─────────────┘
                                                    │ aligned samples
                                                    ▼
                                    ┌────────── recording/ ──────────┐
                                    │ RecorderIO → HDF5 episode v15  │
                                    └───────┬─────────────┬───────────┘
                                            ▼             ▼
                                       replay/     recording/analysis/
```

### 设计边界

- **IPC 边界**：所有跨进程数据经 `SharedStorage` 传递；有效负载由 `ipc/schema.py` 的固定 NumPy dtype 定义。
- **硬件边界**：xArm/XHand SDK 仅由各自的执行 worker 使用，RealSense SDK 由 `sensor/` 持有；其他进程不共享活的 SDK 对象。
- **控制边界**：策略/遥操作 worker 决定动作与采样网格；`RecorderIO` 只负责序列化、校验和事务式发布。
- **安全边界**：`SafetyState` 管理 `DISARMED → ARMED → RUNNING → FAULT`；固件仍是最后一道安全保护。

## 运行模型与数据流

### 进程职责

标准遥操作运行时包含五个控制/设备 worker；开始录制后增加一个 `RecorderIO` worker：

```text
camera ────────┐
VR ────────────┼──► teleop / policy ──► arm queue ──► arm worker
arm state ─────┤          │
hand state ────┤          └────────────► hand ring ──► hand worker
               │
               └──► shared-memory state

teleop / policy ──► fixed-grid sample ring ──► RecorderIO ──► HDF5
```

| 通道 | 语义 | 关键约束 |
|---|---|---|
| 臂动作队列 | 有序、短队列的未来关节目标 | `maxsize=2` 的反压是有意设计 |
| 手动作环 | 最新目标覆盖旧目标 | latest-wins，避免手部控制滞后 |
| 状态环 | 相机、VR、臂、手的共享快照 | seqlock 验证读，跨进程不传可变对象图 |
| 录制采样环 | 对齐后的机器人、动作与传感器样本 | 固定为 `1 / control_hz` 网格，不以到达时间采样 |

### 关键路径

| 使用场景 | 入口 | 主要调用链 |
|---|---|---|
| VR 采集 | `examples/collect_teleop.py` | `teleop/experiment.py` → `teleop/loop.py` → `planning/`、`robot/`、`recording/` |
| 键盘控制 | `examples/keyboard_teleop_real.py` | `teleop/keyboard_experiment.py` → `keyboard.py` → 安全动作协议 |
| 实验性学习策略 | `examples/deploy_policy.py` | `policy/deployment.py` → `inference_process.py` → `learned_coordinator.py` |
| Episode 回放 | `examples/replay_episode.py` | `replay/episode.py` → `preflight.py` → `session.py` / `runner.py` |
| 相机标定 | `examples/calibrate_camera.py` | 自包含 ArUco 手眼标定；会采集设备数据并原子写入 cameras.json |
| 离线数据分析 | 无额外包装入口 | `python -m dexmani_real.recording.analysis.episode_quality` 或 `visualize_episode` |

## 从入口到核心模块

建议按下面顺序阅读代码；每一层只建立在前面层的稳定接口上。

1. **配置与协议**：先读 `config/defaults.py`、`config/runtime.py`、`ipc/schema.py`，了解默认参数与跨进程数据形状。
2. **数据平面与生命周期**：再读 `shm/shared_storage.py`、`shm/ring_buffer.py`、`runtime/supervisor.py`，了解进程如何共享数据、就绪和停止。
3. **设备和运动能力**：阅读 `sensor/`、`robot/` 与 `planning/`，它们分别产生观测、执行动作、计算 FK/IK/碰撞和路径。
4. **业务控制环**：`teleop/` 是 VR 控制和记录决策中心；`policy/` 是学习策略的隔离推理与动作调度中心。
5. **持久化和事后工作流**：`recording/` 写入/读取 episode；`replay/` 和 `recording/analysis/` 消费这些数据。

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

以下清单覆盖当前包内的 **109 个 Python 源文件**（包含各包的 `__init__.py`）。除根包外，表中路径均相对于 `dexmani_real/`；`__init__.py` 若只负责导出接口，也会单独列出，便于从导入路径反查实现位置。

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

### `ipc/` — 进程无关的数据协议

| 文件 | 作用 |
|---|---|
| `ipc/__init__.py` | 集中导出所有跨进程 dtype：状态、命令、确认、相机、VR、推理和录制帧。 |
| `ipc/schema.py` | 权威的固定形状 NumPy dtype 与关节/末端尺寸常量；构造录制样本复合 dtype。 |

### `planning/` — 运动学、IK、碰撞与轨迹规划

| 文件 | 作用 |
|---|---|
| `planning/__init__.py` | 导出规划器、位姿和规划配置/运行 profile。 |
| `planning/collision_model.py` | 基于 Pinocchio 的 xArm7+XHand 自碰撞/环境碰撞模型，加载统一 SRDF 过滤规则并提供路径段检测。 |
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
| `policy/__init__.py` | 提供延迟加载的旧式 VR policy 兼容导入，避免离线主进程提前载入遥操作代码。 |
| `policy/action_protocol.py` | 定义 prepare/commit/ack 协议、动作安全门、反馈校验、动作发布器与因果关节动作调度器。 |
| `policy/deployment.py` | 实验性学习策略部署 CLI/生命周期：显式启用 inference IPC，解析 spec 并监督推理与执行 worker。 |
| `policy/inference_process.py` | 隔离推理 worker，加载 adapter，编解码候选动作，并验证模型输出是否满足策略契约。 |
| `policy/learned_coordinator.py` | 以单一时钟协调 observation、推理结果、动作执行和退出前 hold 的学习策略控制环。 |
| `policy/loop_timing.py` | 以滑动窗口统计控制环各阶段耗时的轻量 `StageTimer`。 |
| `policy/observation.py` | 构建不可变、因果一致的 observation 快照，防止推理读取到混合时刻的数据。 |
| `policy/observation_sources.py` | 将共享状态环字段映射为策略观测来源，并校验容量、dtype、形状与帧有效性。 |
| `policy/runtime.py` | 定义观测模态、观测/动作规格、冻结数组映射、观测快照和动作候选/块的数据契约。 |
| `policy/spec.py` | 加载并校验策略 YAML、模型资源 SHA-256、观测与动作规格，形成不可变 `PolicySpec`。 |
| `policy/tensor_block.py` | 将 `ObservationSpec` 映射为固定 dtype 的共享 observation tensor block。 |
| `policy/vr_teleop_policy.py` | 保留旧导入路径，将旧 `PolicyConfig` / `policy_loop` 转发至 `teleop` 实现。 |

### `recording/` — Episode 持久化与离线分析

| 文件 | 作用 |
|---|---|
| `recording/__init__.py` | 导出 episode 读写器、时间信息和停止结果的公共接口。 |
| `recording/camera_stream_writer.py` | 在独立写线程中编码并写入相机流，隔离视频 I/O 以免阻塞控制环。 |
| `recording/episode_reader.py` | 读取新旧 HDF5 episode、合并流和元数据，并提供时间/有效性视图。 |
| `recording/episode_recorder.py` | 管理单个 episode 的 HDF5 数据集、相机写入器、停止校验与最终发布。 |
| `recording/io_process.py` | `RecorderIO` worker 及其客户端协议；从对齐样本环消费数据并驱动记录器。 |
| `recording/timestamp_buffer.py` | 按目标时间戳插值、前向填充和标记缺口原因，保证采样网格对齐。 |
| `recording/transaction.py` | 目录 fsync 和原子发布工具，避免半成品 episode 被当作完成数据。 |
| `recording/video_codec.py` | 基于 PyAV 的视频编码器/解码器及其配置，服务 HDF5 旁路视频流。 |
| `recording/analysis/__init__.py` | 标识仅离线使用的 episode 分析与可视化子包。 |
| `recording/analysis/episode_quality.py` | 质量、健康和一致性分析 CLI；支持批量评估、验证、筛选和冻结/缺帧诊断。 |
| `recording/analysis/visualize_episode.py` | 读取 episode 并借助 Rerun 展示 3D、图像、动作、触觉和元数据。 |

### `replay/` — 检查、授权与受控回放

| 文件 | 作用 |
|---|---|
| `replay/__init__.py` | 标识 episode 检查、预检与回放子系统。 |
| `replay/data.py` | 从 HDF5 加载并规范化为回放所需的状态/动作 `TrajectoryData`。 |
| `replay/episode.py` | 回放 CLI：默认离线检查；live 模式在启动 worker 前直接执行密集预检。 |
| `replay/metrics.py` | 捕获回放期间状态，计算关节/末端跟踪和时延指标，并保存报告与原始数据。 |
| `replay/preflight.py` | 在 live replay 启动 worker 前 fail-closed 地重验轨迹来源、模型、几何路径和手部模式。 |
| `replay/runner.py` | 通过 `SharedStorage` 执行已预检轨迹，监控臂/手反馈、动作确认和安全终态。 |
| `replay/session.py` | 启停 live 回放 worker，整合预检、运行结果、回零选项与事后指标报告。 |

### `robot/` — xArm7、XHand 与安全状态

| 文件 | 作用 |
|---|---|
| `robot/__init__.py` | 标识 xArm7、XHand 驱动和执行 worker 所在包。 |
| `robot/arm_loop.py` | xArm Mode 6 伺服 worker：读取有序臂命令、发布 FK 状态、处理 C24 恢复与碰撞故障。 |
| `robot/hand_process.py` | XHand worker：读取 latest-wins 手指令、发布关节/触觉反馈，并检测跟踪停滞。 |
| `robot/homing.py` | 执行并验证机械臂回零，包含状态/心跳检查、路径候选拒绝信息和 e-stop 处理。 |
| `robot/safety.py` | 定义 `SafetyState` 与合法状态迁移/强制迁移检查。 |
| `robot/types.py` | 定义文档化的机器人状态、动作、臂/手/触觉 dataclass；实际 IPC 格式由 `ipc/schema.py` 决定。 |
| `robot/xhand.py` | 封装 XHand SDK 的连接、配置、关节/触觉读写和安全的资源释放。 |

### `runtime/` — 进程生命周期与状态码

| 文件 | 作用 |
|---|---|
| `runtime/__init__.py` | 导出组件阶段、故障码和退出原因等运行时状态枚举。 |
| `runtime/processes.py` | 提供 spawn 上下文、进程退出报告，以及可验证的停止/回收/共享内存关闭流程。 |
| `runtime/status.py` | 定义跨模块使用的组件阶段、故障和退出原因的整数枚举。 |
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
| `shm/robot_ring.py` | 兼容旧导入路径；实际 seqlock 与 `get_last_k()` 实现在 `ring_buffer.py`。 |
| `shm/shared_storage.py` | 创建并持有共享环、队列、标志和事件；默认仅分配遥操作/采集能力，推理 IPC 需显式启用。 |

### `teleop/` — VR 映射、控制环与采集决策

| 文件 | 作用 |
|---|---|
| `teleop/__init__.py` | 标识 VR 遥操作、手部重定向、键盘和音频反馈子系统。 |
| `teleop/arm_mapper.py` | 将 VR 手腕位姿映射为受工作空间、旋转增量和四元数校验约束的臂末端目标。 |
| `teleop/audio_feedback.py` | 管理按键/运动门控下的音频提示播放与节流。 |
| `teleop/config.py` | 汇集遥操作控制环所需的强类型配置。 |
| `teleop/control_state.py` | 表示控制 hold 与回零交接状态，统一记录控制环暂停原因。 |
| `teleop/episode_samples.py` | 将因果状态、动作确认、VR/相机数据对齐为记录帧，并处理 start/stop/held 样本。 |
| `teleop/experiment.py` | VR 采集实验 CLI/生命周期：创建共享存储、启动 worker、预检、监督并有序退出。 |
| `teleop/hand_control.py` | 从手部重定向结果生成平滑、限幅、可回零的 XHand 指令。 |
| `teleop/hand_retarget.py` | 校验手部 landmarks，并提供启发式 XHand 和 TAG 优化两类手部重定向器。 |
| `teleop/keyboard.py` | 处理终端/全局键盘输入、运动活动锁存、臂手反馈检查和末端位姿增量。 |
| `teleop/keyboard_experiment.py` | 键盘遥操作实验的 worker 编排、反馈预检、控制循环、故障与退出处理。 |
| `teleop/loop.py` | 核心 VR policy worker：读取快照、映射/IK、动作安全门、记录决策、状态机与错误恢复。 |
| `teleop/recording_session.py` | 处理退出时的录制决策，并构建带运行时/资源溯源的 episode 启动元数据。 |
| `teleop/safety.py` | 遥操作安全辅助：候选动作生效性、hold 后反馈、无碰撞转移、接触停滞与回零流程。 |
| `teleop/snapshot.py` | 从共享环读取同一因果锚点附近的臂、手、VR、触觉、相机快照，并跟踪相机新鲜度。 |
| `teleop/tag_retargeting/__init__.py` | 导出 TAG 两阶段手部重定向的优化器与 Pinocchio 梯度计算器。 |
| `teleop/tag_retargeting/optimizer.py` | 使用 NLopt 执行 TAG 手部两阶段优化，平衡指尖目标、关节限制与平滑性。 |
| `teleop/tag_retargeting/pin_grad.py` | 用 Pinocchio 计算指尖位置及雅可比/梯度，并校验指尖 frame 名称。 |

### `utils/` — 无领域耦合的通用工具

| 文件 | 作用 |
|---|---|
| `utils/__init__.py` | 标识日志、序列化、限速、信号和数组工具的公共包。 |
| `utils/array_utils.py` | 提供 NaN 初始化数组与安全 resize 等数值数组小工具。 |
| `utils/log.py` | 创建统一 logger、可选文件日志和按时间节流的告警器。 |
| `utils/pointcloud_utils.py` | 实现内参/变换/工作空间校验、RGB-D 到点云、裁剪、下采样、采样与深度可视化。 |
| `utils/rate_manager.py` | 以单调时钟稳定控制循环频率，并报告周期统计信息。 |
| `utils/retry.py` | 提供可重置的连续失败计数器，供设备读写 watchdog 使用。 |
| `utils/serialization.py` | 按 dataclass 类型注解将字典安全转换为嵌套对象和 NumPy 数组。 |
| `utils/signal_utils.py` | 提供四元数安全归一化和位姿 EMA 平滑。 |

## 项目地图：`examples`

`examples/` 目前有 **9 个 Python 文件**；它们全都是薄 CLI，只负责把仓库根加入 `sys.path` 并调用相应领域模块的 `main()`。业务逻辑不应继续堆放在这些文件中。

| 文件 | 调用的领域入口 | 作用与风险 |
|---|---|---|
| `examples/collect_teleop.py` | `teleop.experiment.main` | 标准 VR 遥操作与数据采集入口；会启动真实设备 worker。 |
| `examples/deploy_policy.py` | `policy.deployment.main` | 实验性学习策略入口；需要外部 adapter/spec/模型并会进入真实执行器控制链。 |
| `examples/keyboard_teleop_real.py` | `teleop.keyboard_experiment.main` | 以键盘驱动机械臂、默认使用实测 XHand 反馈的入口；硬件相关。 |
| `examples/replay_episode.py` | `replay.episode.main` | episode 检查/回放入口；默认 dry-run，`--live` 会在启动 worker 前执行密集预检。 |
| `examples/calibrate_camera.py` | — | ArUco 眼到手标定入口；自包含脚本，会采集设备数据并原子写入 cameras.json。 |
| `examples/calibrate_vr_heading.py` | — | VR 朝向标定入口；自包含脚本，会读取 VR 数据并在确认后写入 vr_transform.json。 |
| `examples/realsense_record_example.py` | — | 交互式 RealSense RGB-D 实时采集与点云生成测试；默认只读。 |
| `examples/pointcloud_process_example.py` | `sensor.pointcloud_processor` | 生产点云管道诊断与桌面平面标定；显式确认后才写入标定。 |
| `examples/xhand_control_example.py` | — | 独立 XHand SDK 诊断；动作命令需显式硬件授权。 |

## 配置、资源与延伸文档

| 位置 | 内容 |
|---|---|
| `examples/configs/teleop_lab.yaml` | 遥操作实验覆盖配置示例；只记录刻意偏离默认值的少量字段。 |
| `dexmani_real/config/cameras.json` | 物理相机序列号、类型和外参，是运行时校验的一部分。 |
| `dexmani_real/config/desk_plane.json` | 点云工作空间使用的桌面平面运行数据。 |
| `dexmani_real/config/vr_transform.json` | VR 朝向标定得到的坐标变换运行数据。 |
| `assets/` | URDF/SRDF、网格、手部重定向配置和音频资源。 |
| `CLAUDE.md` | 更详细的架构、运行流程、安全/碰撞、录制 schema 与运维背景。 |
| `AGENTS.md` | 面向代码修改者的仓库约定、硬件安全边界和跨模块变更检查清单。 |

对于涉及 dtype、共享内存、录制 schema、IK/碰撞、安全状态机或速率默认值的改动，请先阅读 `AGENTS.md` 的“Cross-module change checklist”，再沿本 README 的关键路径追踪所有生产者和消费者。
