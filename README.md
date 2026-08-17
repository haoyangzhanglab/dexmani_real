# DexMani Real

> 面向 **xArm7（7 自由度）+ XHand（12 自由度）+ Quest VR + Intel RealSense L515** 的 VR 遥操作、数据采集、轨迹回放与策略部署系统。

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
2. **标定与诊断**：标定相机外参和 VR 朝向；以受限、可观测的方式诊断 RealSense、点云和 XHand。
3. **回放与离线分析**：检查 HDF5 episode、在启动前直接执行密集预检、运行受控 live replay，并评估或可视化数据质量。
4. **策略部署**：通过 `examples/run_policy.py` 运行 learned-policy；推理 worker 只写 `policy_plan_ring`，coordinator 经同一安全边界发布机器人动作。
5. **离线数据处理**：保持 Real v17 原始 episode 只读，按输出模态清洗和切分连续轨迹，生成 real-domain 的 Sim-label HDF5；点云在消费边界从 depth 确定性地派生。

```text
                    ┌───────────── sensor/ ──────────────┐
                    │   RealSense (RGB-D) / VR receiver   │
                    └──────────────┬──────────────────────┘
                                   │ state frames
                                   ▼
┌──────── robot/ ────────┐   ┌──────────────── shm/ ────────────────┐
│ xArm loop / XHand loop │◀──│ SharedStorage · typed rings · queues │
└──────────▲─────────────┘   └─────────────────┬────────────────────┘
           │ arm queue / hand ring               │ snapshots
           │                                     ▼
           │                         ┌──── teleop/          ────┐
           └─────────────────────────│ 映射、IK、动作校验、决策时钟 │
                                     └──────────────┬─────────────┘
                                                    │ aligned samples
                                                    ▼
                                    ┌────────── recording/ ──────────┐
                                    │ RecorderIO → HDF5 episode v17  │
                                    └───────────────────┬───────────┘
                                                        ▼
                                               examples/visualize_episode.py
```

### 设计边界

- **IPC 边界**：所有跨进程数据经 `SharedStorage` 传递；有效负载由 `utils/schema.py` 的固定 NumPy dtype 定义。
- **硬件边界**：xArm/XHand SDK 仅由各自的执行 worker 使用，RealSense SDK 由 `sensor/` 持有；其他进程不共享活的 SDK 对象。
- **控制边界**：遥操作 worker 决定动作与采样网格；`RecorderIO` 只负责序列化、校验和事务式发布。录制的 START/STOP、状态与每格元数据使用固定 dtype，不在共享内存中嵌入 JSON；v17 episode 保存已解析配置的 SHA-256，而非整份配置文本。
- **安全边界**：`SafetyState` 管理 `DISARMED → ARMED → RUNNING → FAULT`；固件仍是最后一道安全保护。
- **策略边界**：`integrations/` 只依赖 `deployment/`，绝不反向；推理 worker 只写 `policy_plan_ring`，`deployment/coordinator.py` 是唯一的 learned-policy 机器人动作生产者，经共享 `SafetyGate` 边界发布。

## 运行模型与数据流

### 进程职责

标准遥操作运行时包含五个控制/设备 worker；开始录制后增加一个 `RecorderIO` worker：

```text
camera ────────┐
VR ────────────┼──► teleop ──► arm endpoint/HOME queue ──► arm worker
arm state ─────┤          │
hand state ────┤          └────────────► hand ring ──► hand worker
               │
               └──► shared-memory state

teleop ──► fixed-grid sample ring ──► RecorderIO ──► HDF5
```

| 通道 | 语义 | 关键约束 |
|---|---|---|
| 臂动作队列 | 有序、短队列的未来关节目标 | `maxsize=2` 的反压是有意设计；worker 丢弃非当前 generation 的待消费 endpoint |
| 手动作环 | 最新目标覆盖旧目标 | latest-wins，避免手部控制滞后；worker 同样复核 generation 与有效期 |
| 状态环 | 相机、VR、臂、手的共享快照 | seqlock 验证读，跨进程不传可变对象图 |
| 录制采样环 | 对齐后的机器人、动作与传感器样本 | 固定为 `1 / control_hz` 网格，不以到达时间采样 |

### 关键路径

| 使用场景 | 入口 | 主要调用链 |
|---|---|---|
| VR 采集 | `examples/collect_teleop.py` | `teleop/loop.py` → `planning/`、`robot/`、`recording/`（实验生命周期自包含在 examples 中）|
| 键盘控制 | `examples/keyboard_teleop.py` | `teleop/keyboard.py` → 松键推进 generation / 命令静默 / 实测位姿重锚 → 安全动作协议 |
| 策略部署 | `examples/run_policy.py` | `deployment/lifecycle.py` → `deployment/worker.py`（推理）→ `deployment/coordinator.py`（动作发布，经共享安全边界）|
| Episode 回放 | `examples/replay_episode.py` | 自包含脚本；默认 live 完整回放并产出结果；`--dry-run` 仅离线校验 |
| Episode 清洗/映射 | `examples/process_episodes.py` | `data_processing/` → profile-aware hard mask/切段 → 事务式 Sim-label HDF5；纯离线，不访问硬件 |
| 相机标定 | `examples/calibrate_camera.py` | 自包含 ArUco 手眼标定；会采集设备数据并原子写入 cameras.json |
| 离线数据分析 | `examples/visualize_episode.py` | Rerun 3D episode 可视化；`python examples/visualize_episode.py <episode>` |
| Hand retarget 调参 | `examples/tune_hand_retarget.py` | 离线顺序重放 TAG/DexPilot，输出关节/指尖/平滑/耗时指标与前 4 帧 home 估计；不访问硬件 |

### 普通暂停与录制语义

普通暂停（C/S/D/Q、VR stale、手反馈异常、音频门控，以及键盘松键）统一采用“命令静默”：

```text
推进 run_generation → 停止发布 arm/hand action
→ worker 丢弃仍在 IPC 中的旧 generation 命令
→ xArm Mode 6 完成最后一个已经接受的 endpoint
→ 恢复前从新鲜实测反馈重锚
```

这不是急停，也不会调用 State 6 或发布“实测位置 hold”。C 只能恢复 C 建立的暂停；S/D/时长触顶后的下一轮必须按 B，每次 B 另开一个 `run_generation`。命令静默期间不产生 action sample；恢复后的首个样本携带新 `control_run_generation`，RecorderIO 把下一个存储槽重锚到该样本真实时间（保留 wall-time 跳变，不补造 hold action）。`min_record_duration_s` 是质量标签而非发布硬门槛（短 episode 保持 v17 有效，标记 `min_frames_met=False`）。

完整语义见 `CLAUDE.md` §4（Critical behavior paths）；跨模块契约见 `AGENTS.md`。

## 从入口到核心模块

建议按下面顺序阅读代码；每一层只建立在前面层的稳定接口上。

1. **配置与协议**：先读 `config/defaults.py`、`config/runtime.py`、`utils/schema.py`，了解默认参数与跨进程数据形状。
2. **数据平面与生命周期**：再读 `shm/shared_storage.py`、`shm/ring_buffer.py`、`runtime/supervisor.py`，了解进程如何共享数据、就绪和停止。
3. **设备和运动能力**：阅读 `sensor/`、`robot/` 与 `planning/`，它们分别产生观测、执行动作、计算 FK/IK/碰撞和路径。
4. **业务控制环**：`teleop/` 是 VR 控制和记录决策中心；`deployment/` 是 learned-policy 部署的控制环。
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
> `examples/` 下的所有入口都可能影响硬件。不要仅为“试一下”而运行遥操作、回放、标定或设备诊断；执行前应确认工作空间清空、设备状态正常，并获得相应操作授权。`replay_episode.py` 默认会跨越硬件安全边界，重新执行完整轨迹；仅 `--dry-run` 是离线检查。

## 项目地图：`dexmani_real`

以下清单覆盖当前包内的 **101 个 Python 源文件**（包含各包的 `__init__.py`）。除根包外，表中路径均相对于 `dexmani_real/`；`__init__.py` 若只负责导出接口，也会单独列出，便于从导入路径反查实现位置。

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

> **法兰转接件补偿（已知几何决策）**：实体法兰转接件厚 0.033 m，但 `xarm7_xhand_right.urdf` 与 `xarm7_xhand_collision.urdf` 仍以 `flange_joint2` 固定关节编码 0.043 m。FK/规划路径经 `HandParams.T_eef_handbase_pos_xyz` 的 −0.010 m 修正精确对齐（residual 0 m）；碰撞模型直接加载原始 URDF、无等价修正，故手部碰撞网格残留 ±10 mm 系统偏移，已折入 `TableCollisionConfig.soft_clearance_m`（0.01 → 0.02 m）。`hand_safety_margin_m`（0.05 m，仅固定-Z 回退路径）无需改动。保留代码层补偿、不改 URDF，以免破坏 episode 的 URDF SHA-256 provenance。

### `planning/` — 运动学、IK、碰撞与轨迹规划

| 文件 | 作用 |
|---|---|
| `planning/__init__.py` | 导出规划器、位姿和规划配置/运行 profile。 |
| `planning/collision_model.py` | 基于 Pinocchio 的 xArm7+XHand 自碰撞/环境碰撞模型；把标定桌面构造成倾斜碰撞几何，允许基座安装接触，并提供真实网格距离与路径段检测。 |
| `planning/constants.py` | 集中定义 XHand SDK 关节序与 URDF/Pinocchio 关节序之间的重排常量。 |
| `planning/hand_kinematics.py` | 用 Pinocchio 计算手部正向运动学和五指指尖位置，并处理关节顺序重映射。 |
| `planning/ik.py` | 遥操作位置 IK 求解器：确定性种子、快速接受、多候选回退（归一化可操作度与位姿误差加权评分）、误差/碰撞检查、零空间优化，并按拒绝门输出结构化失败分类（unreachable/delta/collision/…）。 |
| `planning/ik_candidates.py` | 生成、过滤、评分和规范化 IK 候选，输出结构化拒绝原因诊断。 |
| `planning/kinematics.py` | 提供不依赖 MPlib 的臂 FK，以及带 MPlib 集成的完整 xArm7 运动学/雅可比能力。 |
| `planning/path_utils.py` | 关节路径插值、稠密化、角度等价包裹，以及两阶段回零和 band 对齐路径规划。 |
| `planning/planner.py` | `XArm7MotionPlanner`：组合 MPlib、运动学、IK 候选和碰撞模型以完成 IK/路径规划。 |
| `planning/pose_utils.py` | 位姿组合/求逆、四元数与 rot6d 转换、关节形状检查和位置姿态误差计算。 |
| `planning/types.py` | 定义 `Pose`、IK/路径结果、碰撞信息以及规划/遥操作 profile 等核心数据类。 |

### `policy/` — 动作协议与安全校验

| 文件 | 作用 |
|---|---|
| `policy/__init__.py` | 标识动作协议、安全校验与控制环计时包。 |
| `policy/safety.py` | 单一安全门 (SafetyGate) — 良构→关节限位→工作空间；发布边界统一检查运行态、arm/hand feedback 健康度（`_hand_feedback_snapshot` fail-closed：`connected`/`state_valid`/`error_state`/`send_healthy`/`read_healthy` + `qpos`/`last_cmd_qpos` shape+finite）和 coupled-hand 机械/增量预检，并返回类型化拒绝/传输结果，worker 在 SDK 前仍独立复核，不裁剪 action；`run_generation` 使暂停前候选失效；hand-home 会生成显式合法里程碑并逐条等待 SDK 接受回执。 |
| `policy/loop_timing.py` | 以滑动窗口统计控制环各阶段耗时的轻量 `StageTimer`。 |
| `policy/runtime.py` | 定义单 tick 动作候选 `ActionCandidate` 的数据契约，含 run generation、时效与只读数组封装。 |

### `deployment/` — Learned-Policy 部署运行时

| 文件 | 作用 |
|---|---|
| `deployment/__init__.py` | 标识后端/观测/动作适配器边界与 learned-policy 部署运行时（模型输出只是 proposal，不是 robot command）。 |
| `deployment/config.py` | `DeploymentConfig` 与 `resolve_deployment_config`（CLI > file/data > defaults + SHA-256）；模型内部参数绝不进配置。 |
| `deployment/contracts.py` | `PolicyBackend` / `ObservationAdapter` / `ActionAdapter` 三个 Protocol + `JointActionChunk` / `InferenceContext`，均不 import torch 或 SharedStorage。 |
| `deployment/observation.py` | 进程本地不可变观测窗 `ObservationBatch` / `FrameWindow` / `CameraWindow`（不进 SharedStorage）。 |
| `deployment/loader.py` | `module:symbol` 惰性加载器，实例化后 Protocol 校验，失败 fail-closed；parent 不 import torch。 |
| `deployment/fake.py` | 确定性、无 torch、无硬件的 fake 后端（backend-swap fixture）。 |
| `deployment/worker.py` | `inference_loop`：观测 → encode → infer → decode → 只写 `policy_plan_ring`；不写 arm/hand transport、不碰 SafetyState。 |
| `deployment/coordinator.py` | `coordinator_loop`：唯一的 learned-policy robot-action producer——采纳计划、调度单个 due endpoint（latest-wins、不插值）、走共享 `build_action_candidate`/`validate_and_send_candidate` 发布边界、手部 delta 预检、命令静默 watchdog。 |
| `deployment/lifecycle.py` | `build_policy_worker_specs` + `run_policy_deployment`：组合 A/B 冻结的 `WorkerSpec`/`run_supervisor`/`shutdown_processes`，无第二套健康机制。 |
| `deployment/metrics.py` | 无 Prometheus/OTel 的 counter/gauge 注册表 + 结构化 flush。 |
| `deployment/provenance.py` | 一次性启动 provenance 日志（commit/target/checkpoint/runtime SHA-256），不进高频 IPC payload。 |

### `integrations/` — 模型仓库适配器

| 文件 | 作用 |
|---|---|
| `integrations/__init__.py` | 说明依赖方向：`deployment/*` 不得 import 本包；integration → deployment。 |
| `integrations/dexmani_policy.py` | DexMani Policy 模型仓库适配器：`DexManiObservationAdapter`/`DexManiPolicyBackend`/`DexManiActionAdapter`；`load()` 内惰性 import，native-joint-only，EE checkpoint 启动即拒绝。 |

### `data_processing/` — Real episode 清洗与 Sim-label HDF5

| 文件 | 作用 |
|---|---|
| `data_processing/__init__.py` | 导出处理 profile、不可变配置/决策合同和批处理入口。 |
| `data_processing/contracts.py` | 定义 profile-aware 配置、人工 annotation、连续 segment 与 episode 决策。 |
| `data_processing/cleaning.py` | 纯决策层：按 core/模态 hard gate 生成 mask，不拼接缺口，计算训练窗口和软质量指标。 |
| `data_processing/transforms.py` | RGB/K resize 与 point-cloud 确定性 FPS/补点；不改变 real 坐标 frame。 |
| `data_processing/pipeline.py` | 发现/审计源 episode，流式写多个 HDF5（点云从 depth 逐帧派生），写后 fail-closed 校验并目录级原子发布。 |
| `data_processing/cli.py` | argparse、profile 对比、dry-run 和批处理编排；不包含数据处理业务逻辑。 |

### `recording/` — Episode 持久化与离线分析

| 文件 | 作用 |
|---|---|
| `recording/__init__.py` | 导出 episode 读写器、时间信息和停止结果的公共接口。 |
| `recording/camera_stream_writer.py` | 在独立写线程中编码并写入相机流，隔离视频 I/O 以免阻塞控制环。 |
| `recording/episode_schema.py` | v17 的 93 个基础 dataset、条件 sent-command 字段、固定 diagnostics 和共享 layout 校验合同。 |
| `recording/episode_reader.py` | 读取已原子发布的 v17 episode、合并流和元数据，并提供内部有效性、最短时长质量视图及顺序 RGB iterator。 |
| `recording/episode_recorder.py` | 管理单个 episode 的 HDF5 数据集、相机写入器、停止校验与最终发布。 |
| `recording/io_process.py` | `RecorderIO` 非阻塞事务 worker 及其客户端协议；固定 dtype 携带 generation、FINALIZING/终态和会话失败结果。 |
| `recording/recorder_client.py` | policy 侧 `RecorderClient` 与共享控制面协议类型；持有录制决策与固定 sample 构造，与 RecorderIO 依赖单向。 |
| `recording/timestamp_buffer.py` | 对同一控制段按 deadline 因果补帧，并在命令静默后的新 generation 上重锚时间网格。 |
| `recording/transaction.py` | 目录 fsync 和原子发布工具，避免半成品 episode 被当作完成数据。 |
| `recording/video_codec.py` | 基于 PyAV 的视频编码器/解码器及其配置，服务 HDF5 旁路视频流。 |

### `examples/replay_episode.py` — 检查、授权与受控回放

Episode 回放功能整体位于单一自包含脚本 `examples/replay_episode.py` 中（约 2000 行）。默认执行 live 完整回放：跨越硬件安全边界，在启动 arm/hand worker 前执行密集几何和来源预检，通过 `SharedStorage` 回放完整 `sent` 轨迹，捕获回放状态并计算关节/末端跟踪一致性指标与时间延迟；`--dry-run` 退化为纯离线检查。

### `robot/` — xArm7、XHand 与安全状态

| 文件 | 作用 |
|---|---|
| `robot/__init__.py` | 标识 xArm7、XHand 驱动和执行 worker 所在包。 |
| `robot/arm_loop.py` | xArm Mode 6 伺服 worker：按 generation 读取有序臂命令、发布 FK 状态；任意运行期 controller error 或终止性 SDK/API 失败进入单一 sticky fault 路径（不隐式清错）；DISARMED、FAULT、紧停后备与退出确认 State 4，成功初始化时收敛 SDK 冗余输出，失败时保留原生诊断。 |
| `robot/arm_sdk.py` | 共享 xArm SDK 表面：`ArmLoopConfig`（Mode 6 在线轨迹规划配置，`from_runtime` 解析）与叶子级 live-read 原语 `_read_live_error_code`/`_require_sdk_ok`，供 `arm_loop.py`（伺服）与 `homing.py`（回零执行）共同使用，保持依赖无环。 |
| `robot/hand_process.py` | XHand worker：读取 latest-wins 手指令、复核命令/机械限位、发布关节/触觉反馈与最后成功 action ID；不以目标—反馈不收敛判定故障。伺服 PID 增益与逐关节电流上限从 `config.defaults.HandParams` 解析（`kp`/`tor_max_ma` 逐关节，`ki`/`kd` 均匀）。 |
| `robot/homing.py` | 回零编排与执行：`send_arm_home` 编排候选路径，`run_planned_homing` 为执行入口（Mode 0 里程碑执行，返回 provisional 结果；Mode 6 恢复由 arm loop 单点 finalize）；包含状态/心跳检查、路径候选拒绝信息和 e-stop 处理。 |
| `robot/safety.py` | 定义 `SafetyState` 与合法状态迁移/强制迁移检查。 |
| `robot/types.py` | 定义文档化的机器人状态、动作、臂/手/触觉 dataclass；实际 IPC 格式由 `utils/schema.py` 决定。 |
| `robot/xhand.py` | 封装 XHand SDK 的连接、配置、关节/触觉读写和安全的资源释放；生产默认通过 `/dev/ttyUSB0` 上的 3 Mbps RS485（配置值 `serial`）连接，EtherCAT 仅作为显式配置回退；RS485 打开后默认稳定等待 1 秒，反馈使用只刷新状态、不重放动作的 `read_state(..., True)`；1501070 CRC 对相同绝对位置目标默认只重试一次，而只读实时状态事务默认重试两次（均退避 80 ms），耗尽后仍按失败处理；1501018–1501020 保留有效关节反馈并按字段降级传感数据（合力失败保守失效、分布力失败保留合力/接触、温度失败保留全部力数据），下一次成功读取立即恢复，超时和板卡错误仍立即失败；超过运行或厂商机械限位的命令整条拒绝，绝不隐式 clip，运行配置只能收紧而不能放宽额定机械包络；成功初始化时汇总原生 SDK 噪声，连接失败时回放完整诊断。 |

### `runtime/` — 进程生命周期与状态码

| 文件 | 作用 |
|---|---|
| `runtime/__init__.py` | 导出退出原因等运行时状态枚举。 |
| `runtime/processes.py` | `WorkerSpec` 声明 worker target/args，`build_processes`/`start_processes` 收敛 spawn/start 拓扑；`ProcessExit`/`ShutdownReport` 报告退出，并提供可验证的停止/回收/共享内存关闭流程。 |
| `runtime/status.py` | 定义跨模块使用的退出原因整数枚举。 |
| `runtime/supervisor.py` | 完成 worker 就绪等待、心跳/进程监督、健康摘要和协调关闭。 |

### `sensor/` — 相机、点云与 VR 接收

| 文件 | 作用 |
|---|---|
| `sensor/__init__.py` | 标识 RealSense、VR 接收与点云处理能力所在包。 |
| `sensor/camera_process.py` | 相机 worker：采集、打包并发布 RGB-D 帧，维护相机健康状态和心跳。 |
| `sensor/clock_sync.py` | 将设备时钟映射到主机单调时钟，检测重置/漂移，供帧新鲜度判断使用。 |
| `sensor/pointcloud_processor.py` | 将 RGB-D、内外参和工作空间裁剪确定性地派生为采样/下采样后的点云观测（depth 的纯函数，无 RNG）；由消费边界（`data_processing` 离线、未来视觉适配器在线）调用，相机循环不再内联计算。 |
| `sensor/realsense.py` | RealSense D400/L515 驱动：设备发现、配置、对齐、时钟映射、帧采集和生命周期管理。 |
| `sensor/vr_receiver_process.py` | Quest/VR 接收 worker：校验姿态与关键点数据、转换四元数顺序并发布 VR 帧。 |

### `shm/` — 共享内存数据平面

| 文件 | 作用 |
|---|---|
| `shm/__init__.py` | 说明共享内存公共接口及其与回零/监督模块的职责边界。 |
| `shm/camera_ring.py` | 大相机帧（RGB+depth）的变长槽共享内存环 `CameraRingBuffer`，复用同一 seqlock 提交/发布合同。 |
| `shm/causal_reader.py` | 从各状态环读取因果帧（`0 < source <= publish <= anchor`）的公共读取器；不含 age threshold，供遥操作快照与 deployment 观测共用。 |
| `shm/ring_buffer.py` | 通用共享内存 seqlock 环（`SeqlockSlot` + `SharedMemoryRingBuffer`），提供零拷贝写入与已验证读取；相机专用变长槽环见 `shm/camera_ring.py`。 |
| `shm/shared_storage.py` | 创建并持有共享环、队列、标志和事件；默认仅分配遥操作/采集能力，`policy_plan_ring` 供 learned-policy 部署使用。 |

### `teleop/` — VR 映射、控制环与采集决策

| 文件 | 作用 |
|---|---|
| `teleop/__init__.py` | 标识 VR 遥操作、手部重定向、键盘和音频反馈子系统。 |
| `teleop/arm_mapper.py` | 将 VR 手腕位姿映射为受工作空间、旋转增量和四元数校验约束的臂末端目标。 |
| `teleop/audio_feedback.py` | 管理按键/运动门控下的音频提示播放与节流。 |
| `teleop/config.py` | 遥操作配置薄视图：仅持有 `runtime` 快照引用与 4 个会话专属字段（task_label/operator/hand_urdf_path/vr_transform_path），运行时值统一经 `config.runtime.<section>.<field>` 直读。 |
| `teleop/control_state.py` | 表示 command quiescence 与回零交接状态，记录首次暂停原因和反馈新鲜度边界。 |
| `teleop/episode_samples.py` | 将因果状态、动作、VR/相机数据对齐为记录帧，并处理 start/stop 与主动安全回退的 held 样本；命令静默期间不补造样本。 |
| `teleop/hand_control.py` | 手部命令生成与重定向器状态辅助：每个 verified VR ring sequence 最多调用一次有状态 solver（成功/失败均缓存，ramp 仍按控制网格推进）；对 shaped 目标做后备校验，违规时优雅 hold 而非升级为粘滞 fault。 |
| `teleop/hand_retarget.py` | 校验手部 landmarks，并提供 TAG（默认）与 DexPilot 两类重定向器；二者输出统一为 schema 定义的 XHand SDK 关节顺序。 |
| `teleop/hand_retarget_eval.py` | 纯离线 episode 重放、有界参数搜索、后端中立指标和静态收敛 home 估计；不启动设备或 shared-memory 生命周期。 |
| `teleop/keyboard.py` | 处理终端/全局键盘输入、运动活动锁存、臂手反馈检查和末端位姿增量；终端输入抑制持续到设备进程退出，恢复终端时丢弃积压的 canonical 输入；停止回调后不为 Linux/XRecord 守护线程的延迟退出阻塞停机。 |
| `teleop/loop.py` | 核心 VR policy worker：读取快照、映射/IK、动作安全门、记录决策、状态机与错误恢复。 |
| `teleop/recording_session.py` | 处理退出时的保存、丢弃和停机决策。 |
| `teleop/safety.py` | 遥操作安全辅助：候选动作生效性、arm-only hold、接触停滞与回零流程；return-home 逐条确认有界 hand-home 里程碑已被 SDK 接受，但不等待手指角度收敛。 |
| `teleop/snapshot.py` | 从共享环读取同一因果锚点附近的臂、手、VR、触觉、相机快照，并跟踪相机新鲜度。 |
| `teleop/vr_transform.py` | 定义 schema-v1 VR 朝向标定契约：加载/校验 SO(3) 旋转、坐标约定与机器可读质量，POOR 质量拒绝运行。 |
| `teleop/tag_retargeting/__init__.py` | 导出 TAG 两阶段手部重定向的优化器与 Pinocchio 梯度计算器。 |
| `teleop/tag_retargeting/optimizer.py` | 使用 NLopt 执行 TAG 手部两阶段优化，平衡指尖目标、关节限制与平滑性。 |
| `teleop/tag_retargeting/pin_grad.py` | 用 Pinocchio 计算指尖位置及雅可比/梯度，并校验指尖 frame 名称。 |

### `utils/` — 无领域耦合的通用工具

| 文件 | 作用 |
|---|---|
| `utils/__init__.py` | 标识日志、序列化、限速、信号、schema、数组和限位校验工具的公共包。 |
| `utils/array_utils.py` | 提供 NaN 初始化数组与安全 resize 等数值数组小工具。 |
| `utils/hand_health.py` | arm/hand 反馈健康谓词（`validate_arm_feedback` / `validate_hand_feedback`），供 teleop、policy 发布边界与 replay 共用。 |
| `utils/limits.py` | 校验 XHand 三级关节限位层级（rated ⊇ mechanical ⊇ command），收敛 config/robot/hand_process 三处重复的嵌套校验。 |
| `utils/log.py` | 创建统一 logger、可选文件日志和按时间节流的告警器。 |
| `utils/pointcloud_utils.py` | 实现内参/变换/工作空间校验、RGB-D 到点云、裁剪、下采样、采样与深度可视化。 |
| `utils/rate_manager.py` | 以单调时钟稳定控制循环频率，并报告周期统计信息。 |
| `utils/retry.py` | 提供可重置的连续失败计数器，供设备读写 watchdog 使用。 |
| `utils/schema.py` | 跨进程 NumPy dtype 与关节/末端尺寸常量的唯一定义源。 |
| `utils/serialization.py` | 按 dataclass 类型注解将字典安全转换为嵌套对象和 NumPy 数组。 |
| `utils/signal_utils.py` | 提供四元数安全归一化和位姿 EMA 平滑。 |

## 项目地图：`examples`

`examples/` 目前有 **12 个 Python 文件**。入口点专有逻辑（如实验生命周期、控制循环）直接放在 examples 中；共享库代码留在 `dexmani_real` 包内。

| 文件 | 调用的领域入口 | 作用与风险 |
|---|---|---|
| `examples/collect_teleop.py` | — | 标准 VR 遥操作与数据采集入口；实验生命周期自包含；会启动真实设备 worker。 |
| `examples/run_policy.py` | `deployment.lifecycle` | learned-policy 部署入口：argparse → 解析 runtime/deployment 配置 → 运行生命周期 → 退出码；薄 CLI，无模型/调度/安全/存储逻辑。 |
| `examples/keyboard_teleop.py` | — | 以有界前视目标执行键盘 Cartesian jog（默认目标速度 0.24 m/s、最大前视 40 mm）；松键推进 generation 后停止发布，控制器自然完成最后一个已接受 endpoint，空闲期间持续从实测关节/FK 重建命令基准；R 会先确认 hand-home SDK 接受、再执行 arm home；终端输入抑制保持到 worker 完全退出；硬件相关。 |
| `examples/replay_episode.py` | — | episode 回放入口；默认 live 完整回放，`--dry-run` 仅离线校验；退出时推进 generation 并停止发布。 |
| `examples/process_episodes.py` | `data_processing.cli` | 纯离线薄 CLI：比较 profile、dry-run 或把 Real v17 批量清洗为 real-domain Sim-label HDF5。 |
| `examples/calibrate_camera.py` | — | ArUco 眼到手标定入口；自包含脚本，会采集设备数据并原子写入 cameras.json。 |
| `examples/calibrate_vr_heading.py` | — | VR 朝向标定入口；自包含脚本，会读取 VR 数据并在确认后写入 vr_transform.json。 |
| `examples/realsense_record_example.py` | — | 交互式 RealSense RGB-D 实时采集与点云生成测试；默认只读。 |
| `examples/pointcloud_process_example.py` | `sensor.pointcloud_processor` | 生产点云管道诊断与桌面平面标定；显式确认后才写入标定。 |
| `examples/xhand_control_example.py` | — | 独立 XHand SDK 诊断；默认运行 home + 预设动作（读取/打印后直接动作），无 CLI 参数门控；RS485 固定使用 `/dev/ttyUSB0`，打开前只读检查设备节点与当前用户权限，完整 SDK 会话在隔离子进程中运行，厂商串口线程异常退出会转换为可操作诊断且不让启动进程 core dump；复用生产的 1 秒打开稳定等待、动作 CRC 单次同目标重试、实时读取 CRC 两次重试及触觉字段错误分类；命令响应报告触觉降级时，驻留后仅用实时只读状态请求有限验证恢复，绝不为传感错误重放动作；未解决的初始关节读取或动作事务失败会中止后续预设，传感恢复失败则明确标记而不改变已接受的动作。 |
| `examples/visualize_episode.py` | — | 离线 Rerun 3D 可视化；读取 HDF5 episode 并展示点云、图像、动作、触觉和元数据；无硬件控制。 |
| `examples/tune_hand_retarget.py` | `teleop.hand_retarget_eval` | 离线 TAG/DexPilot 基准、有界搜索和 home 估计；只读 episode，输出 JSON，无硬件控制。 |

## 配置、资源与延伸文档

| 位置 | 内容 |
|---|---|
| `dexmani_real/config/cameras.json` | 物理相机序列号、类型和外参，是运行时校验的一部分。 |
| `dexmani_real/config/desk_plane.json` | 点云过滤、在线动作安全与回零路径共同使用的桌面平面标定数据。 |
| `dexmani_real/config/vr_transform.json` | schema-v1 VR 朝向标定；启动前校验 SO(3)、坐标约定与机器可读质量，POOR 质量拒绝运行。 |
| `assets/` | URDF/SRDF、网格、手部重定向配置和音频资源。 |
| `CLAUDE.md` | 实现导航：任务路由、所有权、数据契约、关键行为路径、硬件工程事实与命令入口。 |
| `AGENTS.md` | 面向代码修改者的仓库契约：架构不变量、硬件安全边界和跨模块变更检查清单。 |
| `docs/dataset/hdf5_episode.md` | Real v17 与 Sim HDF5/Zarr 统一中文数据字典：metadata、dataset、shape、dtype、单位、坐标/时序、转换规则、实测普查与已知问题。 |
| `docs/dataset/sim_hdf5_zarr.md` | DexMani Sim HDF5/Zarr 独立审计版；相同内容已并入上述统一数据字典第 11 节。 |
| `docs/dataset/real_to_sim_mapping.md` | Real v17 episode → Sim/Policy 标签映射表：仅登记字段来源、结构关系和语义差异，不修改录制数值。 |
| `docs/dataset/processed_hdf5.md` | Real v17 清洗、模态相关切段、数值转换与 `dexmani-real-simlabel-hdf5/v1` 输出合同。 |
| `docs/hand_retargeting.md` | hand retarget 当前控制合同：逐 VR observation 求解缓存、TAG/DexPilot 两后端、命令整形/验证/发布边界、状态推进时机与碰撞/触觉边界。 |

对于涉及 dtype、共享内存、录制 schema、IK/碰撞、安全状态机或速率默认值的改动，请先阅读 `AGENTS.md` 的跨模块变更清单，再沿本 README 的关键路径追踪所有生产者和消费者。
