# DexMani Real Repository Map

本文件记录当前版本控制树中每个文件的主要职责，帮助开发者从入口快速找到 owner。
运行行为仍以源码、schema 和配置为准；本文件不定义 API、数据格式或参数默认值。

维护规则：新增、删除、移动文件或改变文件的主要职责时，同步更新本文件。运行时生成且
被 `.gitignore` 排除的 `episodes/`、`episodes_processed/`、`dataset/`、`replay_results/`
等目录不属于本清单。

## 1. 顶层与 agent 配置

| 文件 | 职责 |
|---|---|
| `.gitignore` | 排除虚拟环境产物、缓存、构建物、实验输出、episode、dataset 与本地 IDE 文件。 |
| `.codex/config.toml` | 仓库本地 Codex sandbox、审批和网络访问默认设置。 |
| `.claude/settings.local.json` | 当前机器的 Claude Code 权限白名单；包含本地路径和命令授权，不是运行时配置。 |
| `.claude/skills/karpathy-guidelines/SKILL.md` | Claude Code 的小改动、显式假设和可验证目标工作指南。 |
| `AGENTS.md` | 所有 coding agent 的仓库级工程、安全、范围与验收契约。 |
| `CLAUDE.md` | Claude Code 的精简入口与执行清单，具体规则委托给 `AGENTS.md`。 |
| `code_style.md` | 面向个人博士科研、数据采集和模型部署的具体编码风格与审查清单。 |
| `README.md` | 面向使用者的能力概览、架构、环境、命令、配置和数据流说明。 |
| `repo_map.md` | 当前逐文件职责索引。 |
| `l515_camera_timing_known_limitation.md` | L515 color 实际约 16.68 Hz、重复 RGB 与动态颜色错位的实测证据、当前接受范围和后续议题。 |
| `user_design.md` | 使用者确认的遥操作与物理回放桌面碰撞、XHand 抓取过流行为取舍及其安全边界。 |
| `pyproject.toml` | Python 包元数据、基础依赖、setuptools 包发现、内置 JSON 数据声明与 Black-compatible isort 配置。 |

## 2. 运行时主路径

| 路径 | Owner 与数据流 |
|---|---|
| 遥操作 | `examples/collect_teleop.py` → `teleop/session.py` → device workers + `teleop/loop.py` → operator controls / control grid → safety gate → command publication → command IPC。 |
| 键盘控制 | `examples/keyboard_teleop.py` → `teleop/keyboard_session.py` → safety gate → command publication → arm/hand workers。 |
| 策略部署 | `examples/run_policy.py` → `deployment/lifecycle.py` →（按 observation contract 可选 camera → latest-only pointcloud）→ inference worker → plan ring → coordinator → safety gate。 |
| 录制 | teleop fixed-grid sample → `recording/recorder_client.py` → `EpisodeFrame` ownership-copy → RecorderIO / `EpisodeRecorder` transaction → `EpisodeDataWriter` data.h5 append + camera sidecars → finalize。 |
| 回放 | `examples/replay_episode.py` → `robot/replay_trajectory.py` preflight → `robot/episode_replay.py` lifecycle → `robot/replay_controller.py` → safety gate / command publication → workers → `robot/replay_evaluation.py`。 |
| 离线数据 | native RGB-D raw v20 → processed HDF5 v4 → Policy Zarr v2。 |

## 3. Python package

### `dexmani_real/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/__init__.py` | 包说明，并解析 `PACKAGE_DIR` 与开发树/安装包两种布局下的 `ASSET_DIR`。 |

### `dexmani_real/config/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/config/__init__.py` | 配置子包标记。 |
| `dexmani_real/config/defaults.py` | arm、hand、policy、VR、camera、retarget、安全与环境参数的 canonical dataclass 默认值。 |
| `dexmani_real/config/runtime.py` | 按 CLI > YAML > defaults 合并并校验不可变运行时配置，生成 canonical 内容与 SHA-256。 |
| `dexmani_real/config/camera_calib.py` | 加载按相机序列号绑定的 eye-to-hand/eye-in-hand 外参，并生成录制 metadata。 |
| `dexmani_real/config/cameras.json` | 当前 RealSense 相机序列号、安装类型和外参标定。 |
| `dexmani_real/config/vr_transform.json` | VR/HTS 坐标系到机器人坐标系的朝向标定和质量信息。 |
| `dexmani_real/config/desk_plane.json` | 世界坐标系中的桌面平面方程与标定 provenance。 |

### `dexmani_real/shm/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/shm/__init__.py` | 共享内存子包标记。 |
| `dexmani_real/shm/ring_buffer.py` | 基于 seqlock slot 的共享内存 latest/history ring，以及面向有界 FIFO consumer 的精确 sequence ownership-copy。 |
| `dexmani_real/shm/camera_ring.py` | 针对大尺寸 RGB/depth 帧的共享内存 ring。 |
| `dexmani_real/shm/causal_reader.py` | 按 observation anchor 做因果读取的 arm、hand、tactile、VR、camera helper。 |
| `dexmani_real/shm/shared_storage.py` | 跨进程 data/control plane owner：typed rings、home queue、events、heartbeat、ready 与 safety state。 |

### `dexmani_real/runtime/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/runtime/__init__.py` | 运行时生命周期子包标记。 |
| `dexmani_real/runtime/status.py` | supervisor 与 worker 共用的结构化退出原因枚举。 |
| `dexmani_real/runtime/processes.py` | spawn-only worker spec、进程构建/启动、退出优先级和可验证 shutdown。 |
| `dexmani_real/runtime/supervisor.py` | readiness 等待、heartbeat/进程健康监督、健康摘要和 shutdown 入口。 |

### `dexmani_real/robot/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/robot/__init__.py` | 机器人硬件子包标记。 |
| `dexmani_real/robot/types.py` | `RobotState` 与 `RobotAction` 的 shape/finite 校验数据类型。 |
| `dexmani_real/robot/safety.py` | 共享 robot safety state、run generation 及合法状态转换。 |
| `dexmani_real/robot/command_validation.py` | arm/hand worker 在硬件调用前执行的 shape、finite、freshness、generation 与 expiry 复核。 |
| `dexmani_real/robot/xarm7.py` | 唯一的 xArm Python SDK 驱动边界；连接、状态读取、模式切换、servo 与 homing。 |
| `dexmani_real/robot/xhand.py` | XHand controller 驱动边界；12-DoF 命令、状态、software-only 触觉 bias 和有界 RS485 错误处理。 |
| `dexmani_real/robot/arm_loop.py` | xArm7 Mode 6 arm worker；消费命令/home 请求并发布状态与执行反馈。 |
| `dexmani_real/robot/hand_process.py` | XHand worker；消费 hand command ring，在不确定发送后以 live state 重同步；可恢复抓取过流不发布伪造反馈，恢复帧携带累计计数。 |
| `dexmani_real/robot/homing.py` | 各入口共享的 typed arm-homing 合同、碰撞检查路径选择、请求等待与 abort 处理。 |
| `dexmani_real/robot/episode_replay.py` | 真实机器人回放 session owner；负责 worker、safety transition、operator flow、评估调用与 cleanup。 |
| `dexmani_real/robot/replay_controller.py` | preflight 后轨迹的 fixed-rate safety-gated 命令调度、反馈检查与 terminal outcome。 |
| `dexmani_real/robot/replay_capture.py` | 回放期间 measured state、sent command 与 rejection provenance 的有界内存捕获。 |
| `dexmani_real/robot/replay_trajectory.py` | raw replay trajectory 加载与 processed HDF5 v4 加载、hand-action requirement、provenance 与模型/几何 preflight。 |
| `dexmani_real/robot/replay_evaluation.py` | 回放跟踪/一致性指标、报告和结果文件持久化。 |

### `dexmani_real/sensor/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/sensor/__init__.py` | 传感器子包标记。 |
| `dexmani_real/sensor/camera_geometry.py` | 纯数据 native depth/color intrinsics 与 `T_color_from_depth` 合同。 |
| `dexmani_real/sensor/realsense.py` | RealSense D400/L515 驱动；native RGB-D ownership copy、L515 preset 与分流时序。 |
| `dexmani_real/sensor/camera_process.py` | RealSense worker；发布 native RGB-D、静态 geometry、时间映射、健康状态和 camera ring。 |
| `dexmani_real/sensor/pointcloud.py` | SDK-free native depth/RGB 到 xArm-base 固定点云的确定性几何链。 |
| `dexmani_real/sensor/pointcloud_process.py` | latest-only realtime worker；从 camera ring 生成固定 `float32[N,6]` 并携来源时序发布到 pointcloud ring。 |
| `dexmani_real/sensor/clock_sync.py` | device clock 到 host monotonic clock 的保守映射、reset 检测与相对 delivery-delay 诊断。 |
| `dexmani_real/sensor/vr_receiver_process.py` | crash-isolated HTS/Quest 接收 worker 与 VR frame 发布。 |
| `dexmani_real/sensor/camera_calibration.py` | ArUco 检测、eye-to-hand 求解、残差筛选与标定文件持久化；不打开设备或 GUI。 |
| `dexmani_real/sensor/camera_calibration_control.py` | 标定用 arm feedback、运动状态、workspace clipping、gated publish、quit hold 与归位。 |
| `dexmani_real/sensor/camera_calibration_session.py` | xArm7 + RealSense 标定 lifecycle 的 side-effect owner：设备、采样、GUI、交互与失败 cleanup。 |

### `dexmani_real/planning/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/planning/__init__.py` | 运动规划子包标记。 |
| `dexmani_real/planning/constants.py` | canonical URDF/SRDF 路径及 XHand SDK ↔ URDF 关节顺序映射。 |
| `dexmani_real/planning/types.py` | Pose、IK/path result、collision 信息和 planner/profile 配置类型。 |
| `dexmani_real/planning/pose_utils.py` | pose 合成/求逆/误差、quaternion、rotation matrix 与 rot6d 转换。 |
| `dexmani_real/planning/kinematics.py` | 基于 Pinocchio 的 xArm7 FK、Jacobian 和 pose transform。 |
| `dexmani_real/planning/hand_kinematics.py` | XHand FK 与 fingertip position helper。 |
| `dexmani_real/planning/ik.py` | MPlib teleop position IK、确定性 seed 与 null-space 优化。 |
| `dexmani_real/planning/ik_candidates.py` | IK candidate 生成、规范化、过滤、排序和选择。 |
| `dexmani_real/planning/collision_model.py` | xArm7 + XHand Pinocchio collision model、自碰撞与环境碰撞检查。 |
| `dexmani_real/planning/path_utils.py` | waypoint 插值、关节 wrap、densification 与 typed `ALREADY_HOME`/`SAFE`/`UNSAFE` home/band-alignment path 结果。 |
| `dexmani_real/planning/planner.py` | MPlib motion-planner facade，组合 IK、路径与碰撞检查。 |

### `dexmani_real/policy/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/policy/__init__.py` | 动作协议与安全子包标记。 |
| `dexmani_real/policy/runtime.py` | backend-neutral `ActionCandidate` 合同及动作表示校验。 |
| `dexmani_real/policy/safety.py` | 兼容性显式出口与 hand-home helper；具体边界由下列模块拥有。 |
| `dexmani_real/policy/safety_gate.py` | `ActionCandidate` 的 representation、generation、joint-limit 与 workspace fail-closed 校验。 |
| `dexmani_real/policy/command_publication.py` | controller 侧 runtime/feedback gate、candidate 序列化、arm/hand 发布与 acknowledgement。 |
| `dexmani_real/policy/loop_timing.py` | teleop 固定控制网格的分阶段 timing 采集。 |

### `dexmani_real/teleop/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/teleop/__init__.py` | 遥操作子包标记。 |
| `dexmani_real/teleop/config.py` | 从 resolved runtime 投影出的窄 `TeleopConfig` 与 command-limit 视图。 |
| `dexmani_real/teleop/session.py` | VR teleop 主进程 owner：预检、SharedStorage、worker topology、readiness、监督与 cleanup。 |
| `dexmani_real/teleop/loop.py` | VR teleop coordinator；构造资源、等待 readiness、调度 operator/control grid 并执行 cleanup；桌面不参与在线动作拒绝。 |
| `dexmani_real/teleop/operator_controls.py` | BEGIN/pause/home/quit 信号、retargeter session 初始化及 bounded recording disposition。 |
| `dexmani_real/teleop/control_grid.py` | 单个 causal grid 的 observation read/check、proposal、validate、publish 与 recording。 |
| `dexmani_real/teleop/action_proposal.py` | 纯 EEF、hand 与 arm typed proposal 计算/限幅；不发布命令、不读 shared memory、不写录制。 |
| `dexmani_real/teleop/keyboard_session.py` | 键盘 teleop lifecycle、typed feedback/publish 结果与实时控制流；桌面不参与在线动作拒绝。 |
| `dexmani_real/teleop/keyboard.py` | pynput 全局快捷键、控制信号和键盘 EEF delta 计算。 |
| `dexmani_real/teleop/control_state.py` | teleop coordinator mutable state、control directive 与确定性 command-quiescence 状态转换。 |
| `dexmani_real/teleop/arm_mapper.py` | 将 VR wrist 相对运动映射为 robot-frame target EEF pose。 |
| `dexmani_real/teleop/vr_transform.py` | VR heading 标定文件的 schema、质量等级与 rotation 校验。 |
| `dexmani_real/teleop/camera_freshness.py` | 录制网格上的 camera frame freshness、duplicate、gap 与 generation 分类。 |
| `dexmani_real/teleop/hand_control.py` | hand observation cache、retarget 状态和 hand command 生成 helper。 |
| `dexmani_real/teleop/hand_retarget.py` | VR landmarks 到 XHand 的 TAG/DexPilot retarget facade、校验与状态管理。 |
| `dexmani_real/teleop/dexpilot_prior.py` | 带 human-flexion prior 的 in-repo DexPilot optimizer wrapper。 |
| `dexmani_real/teleop/episode_samples.py` | 从因果 snapshot 构造并发布 typed fixed-grid recording sample。 |
| `dexmani_real/teleop/recording_session.py` | 退出时保存或丢弃 active recording 的有界操作者决策。 |
| `dexmani_real/teleop/safety.py` | teleop pause/hold、re-anchor、contact guard 与 configured homing helper。 |
| `dexmani_real/teleop/audio_feedback.py` | teleop 状态变化的非阻塞中文语音提示。 |
| `dexmani_real/teleop/tag_retargeting/__init__.py` | in-repo TAG retargeting 子包标记。 |
| `dexmani_real/teleop/tag_retargeting/optimizer.py` | 两阶段 NLopt TAG hand-retarget optimizer。 |
| `dexmani_real/teleop/tag_retargeting/pin_grad.py` | Pinocchio hand-kinematics analytical gradient engine 与 fingertip frame 校验。 |

### `dexmani_real/recording/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/recording/__init__.py` | 录制子包标记。 |
| `dexmani_real/recording/episode_schema.py` | raw episode v17–v20 dataset shape/dtype、native RGB-D provenance 与 quality metric 合同。 |
| `dexmani_real/recording/episode_frame.py` | shared-memory record/legacy inputs 到 immutable typed episode row 的唯一解码、schema shaping 与 ownership-copy 边界。 |
| `dexmani_real/recording/timestamp_buffer.py` | 多速率输入到 fixed-grid row 的 timestamp 对齐、填充原因和容量管理。 |
| `dexmani_real/recording/episode_recorder.py` | raw v20 `EpisodeFrame` 的 fixed-grid 对齐、事务式录制、质量汇总、sidecar 验证与原子发布。 |
| `dexmani_real/recording/episode_data_writer.py` | 单个 `data.h5` handle、dataset append 与 flushed offset 的唯一 owner。 |
| `dexmani_real/recording/camera_stream_writer.py` | grid-aligned RGB/depth 的有界后台 writer 与失败传播。 |
| `dexmani_real/recording/video_codec.py` | PyAV RGB video encoder/decoder 与 codec 配置。 |
| `dexmani_real/recording/recorder_client.py` | policy/teleop 侧 recorder control protocol、sample 发布和 stop result。 |
| `dexmani_real/recording/io_process.py` | 独立 RecorderIO worker；`_RecorderIOSession` 按 sequence 连续取得未确认 sample、监控 backlog，拥有 active generation、episode transaction 与有界 finalize。 |
| `dexmani_real/recording/transaction.py` | 同文件系统 fsync、atomic publish 和 atomic JSON helper。 |
| `dexmani_real/recording/episode_reader.py` | published v17–v20 episode 校验、HDF5 sidecar merged view 与 RGB/depth 读取。 |

### `dexmani_real/data_processing/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/data_processing/__init__.py` | 离线数据处理子包标记。 |
| `dexmani_real/data_processing/contracts.py` | output profile、quality/bridge policy、processing config、annotation 与 decision 合同。 |
| `dexmani_real/data_processing/quality.py` | 停滞、抖动、突变等 temporal quality 的纯函数审计。 |
| `dexmani_real/data_processing/cleaning.py` | 将 raw flags、limits、annotations 与质量结果组合为保留/拒绝决策。 |
| `dexmani_real/data_processing/transforms.py` | RGB、native depth 与 color intrinsics resize 的确定性数值变换。 |
| `dexmani_real/data_processing/pipeline.py` | 逐 native raw v20 episode 生成 processed HDF5 v4，点云只派生一次，并事务式验证/发布整批。 |
| `dexmani_real/data_processing/zarr_export.py` | 校验同任务 processed HDF5，并事务式导出最小 Policy Zarr v2。 |

### `dexmani_real/deployment/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/deployment/__init__.py` | learned-policy 部署子包标记。 |
| `dexmani_real/deployment/contracts.py` | inference context、joint action chunk 与 backend/observation/action adapter protocols。 |
| `dexmani_real/deployment/config.py` | deployment YAML/CLI 合并、点云 N/类型校验、required target 校验与 SHA-256 identity。 |
| `dexmani_real/deployment/observation.py` | 因果、不可变 arm/hand/tactile history windows、最新点云帧与 observation batch。 |
| `dexmani_real/deployment/worker.py` | inference worker：`module:symbol` backend/adapter factory 惰性加载、读取并校验因果历史（含新鲜点云）、编码 observation、运行 backend、解码并发布 plan。 |
| `dexmani_real/deployment/coordinator.py` | learned-policy 唯一 command producer；采纳 plan、选择 due step 并通过安全门发布。 |
| `dexmani_real/deployment/lifecycle.py` | policy worker topology、SharedStorage、readiness、ARMED、supervision、verified shutdown 与启动 provenance 日志（含文件 SHA-256）。 |
| `dexmani_real/deployment/metrics.py` | inference/coordinator 计数、reject 分类和周期性日志。 |
| `dexmani_real/deployment/fake.py` | CPU-only、无 torch、确定性的 fake adapters/backend，用于离线验证协议链路。 |

### `dexmani_real/integrations/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/integrations/__init__.py` | 外部模型仓库集成子包标记。 |
| `dexmani_real/integrations/dexmani_policy.py` | `dexmani_policy` 的 observation/backend/action 三个适配器、单帧 `(N,6)` 点云输出与 joint-action fail-closed 边界。 |

### `dexmani_real/utils/`

| 文件 | 职责 |
|---|---|
| `dexmani_real/utils/__init__.py` | 通用工具子包标记。 |
| `dexmani_real/utils/schema.py` | cross-process NumPy dtype、固定点云 N/shape、field size、recording sample dtype 与 NaN array 构造的 canonical 定义。 |
| `dexmani_real/utils/limits.py` | XHand mechanical/command/home limit 层级一致性校验。 |
| `dexmani_real/utils/hand_health.py` | teleop、replay、deployment 共用的 arm/hand feedback freshness 与 finite 健康判断。 |
| `dexmani_real/utils/serialization.py` | dataclass 从 mapping 递归恢复的共享 helper。 |
| `dexmani_real/utils/signal_utils.py` | Cartesian pose EMA 与 quaternion-aware 平滑。 |
| `dexmani_real/utils/rate_manager.py` | 可选精确忙等的 absolute-deadline rate limiting 与 overrun 统计。 |
| `dexmani_real/utils/retry.py` | consecutive-event 计数与阈值升级。 |
| `dexmani_real/utils/log.py` | 中央 logging、native stdout capture、diagnostic 提取与 throttled warning。 |

## 4. 可执行入口

`examples/` 中的文件是薄入口或自包含的诊断/可视化/离线分析程序，不是自动化测试。
除纯 `--help`、`--print-config` 和明确的离线数据命令外，均应按可能影响硬件处理。

| 文件 | 职责 |
|---|---|
| `examples/collect_teleop.py` | 解析 runtime overrides、task/operator/no-hand/no-record，并启动 VR teleop 数据采集。 |
| `examples/keyboard_teleop.py` | 解析 runtime config 和 no-hand 确认，并启动 keyboard teleop。 |
| `examples/run_policy.py` | 解析 runtime/deployment config、backend targets、checkpoint 与点云 N，并启动 policy deployment。 |
| `examples/replay_episode.py` | 加载 raw episode 或（`--processed`）processed HDF5 v4、解析 replay runtime、显示 provenance，并启动真实机器人回放。 |
| `examples/process_episodes.py` | raw → processed HDF5 离线处理 CLI：选择标准点云 N、逐 episode 审计进度（tqdm）、损坏 episode 自动跳过与 JSON report；管线在 `data_processing/`。 |
| `examples/export_policy_zarr.py` | processed HDF5 → Policy Zarr 离线导出 CLI：argparse 与 JSON report；导出事务在 `data_processing/zarr_export.py`。 |
| `examples/visualize_episode.py` | 自包含 Rerun raw-episode 可视化（离线、不连硬件）：metadata、robot state 与实际存储的 camera 数据；不合成点云。 |
| `examples/visualize_episode_processed.py` | 自包含 Rerun processed-HDF5-v4 可视化（离线、不连硬件）：校验并显示文件内 native-derived rgb/depth、固定 `(N,6)` xArm-base 点云、joint/action/触觉时间序列。 |
| `examples/calibrate_camera.py` | xArm7 + RealSense eye-to-hand ArUco 标定入口。 |
| `examples/calibrate_vr_heading.py` | 收集 HTS head/wrist orientation、评估质量并原子更新 VR transform。 |
| `examples/inspect_l515.py` | L515 native RGB-D 几何、option readback、跨流时序与 Z16 场景基线采集；只连接相机，无 GUI，不写标定。 |
| `examples/check_l515_native_shadow.py` | 离线读取带 RGB 的 L515 inspection capture，以标准 N 运行 native xArm-base 点云与 RGB projection shadow gate。 |
| `examples/xhand_control_example.py` | XHand 独立连接、状态/触觉读取和 preset command 交互诊断。 |

## 5. Retargeting 与语音资源

| 文件 | 职责 |
|---|---|
| `assets/retargeting/xhand_right_dexpilot.yml` | DexPilot 的 XHand URDF、wrist/tip link、target joint、scale 与 filter 参数。 |
| `assets/audio/准备退出遥操作.wav` | 进入退出确认时的语音提示。 |
| `assets/audio/即将回到初始姿态.wav` | 开始回到 home 前的语音提示。 |
| `assets/audio/工作继续.wav` | 暂停后恢复工作的语音提示。 |
| `assets/audio/已经回到初始姿态.wav` | homing 完成后的语音提示。 |
| `assets/audio/已退出，是否需要保存轨迹.wav` | 退出后询问是否保存 episode 的语音提示。 |
| `assets/audio/意外的事情出现了.wav` | fault/异常路径的语音提示。 |
| `assets/audio/成功保存轨迹.wav` | episode 成功发布后的语音提示。 |
| `assets/audio/操作暂停.wav` | teleop 进入暂停状态的语音提示。 |
| `assets/audio/操作结束.wav` | 操作正常结束的语音提示。 |
| `assets/audio/放弃保存轨迹.wav` | 操作者丢弃本次 recording 的语音提示。 |
| `assets/audio/轴向已标定.wav` | VR heading/axis 标定成功的语音提示。 |
| `assets/audio/遥操作启动.wav` | teleop session 启动的语音提示。 |

## 6. 机器人模型资源

### xArm7 模型

| 文件 | 职责 |
|---|---|
| `assets/robots/xarm7/xarm7_glb.urdf` | 独立 7-DoF xArm7 kinematic/visual/collision 模型。 |
| `assets/robots/xarm7/xarm7_glb_mplib.srdf` | 独立 xArm7 的 MPlib self-collision disable pairs。 |

### xArm7 collision meshes

这些 OBJ 是 URDF 中 base/link1–link7 的简化碰撞几何。

| 文件 | 对应 link |
|---|---|
| `assets/robots/xarm7/meshes/collision/link_base.obj` | xArm7 base。 |
| `assets/robots/xarm7/meshes/collision/link1.obj` | xArm7 link1。 |
| `assets/robots/xarm7/meshes/collision/link2.obj` | xArm7 link2。 |
| `assets/robots/xarm7/meshes/collision/link3.obj` | xArm7 link3。 |
| `assets/robots/xarm7/meshes/collision/link4.obj` | xArm7 link4。 |
| `assets/robots/xarm7/meshes/collision/link5.obj` | xArm7 link5。 |
| `assets/robots/xarm7/meshes/collision/link6.obj` | xArm7 link6。 |
| `assets/robots/xarm7/meshes/collision/link7.obj` | xArm7 link7。 |

### xArm7 visual meshes

每个 link 的 GLB 是当前 URDF 使用的 visual mesh；OBJ 是同一 visual geometry 的
通用格式版本，MTL 是对应 OBJ 的材质 sidecar。

| 文件 | 职责 |
|---|---|
| `assets/robots/xarm7/meshes/visual/link_base.glb` | base GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link_base.obj` | base OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link_base.mtl` | base OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link1.glb` | link1 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link1.obj` | link1 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link1.mtl` | link1 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link2.glb` | link2 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link2.obj` | link2 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link2.mtl` | link2 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link3.glb` | link3 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link3.obj` | link3 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link3.mtl` | link3 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link4.glb` | link4 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link4.obj` | link4 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link4.mtl` | link4 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link5.glb` | link5 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link5.obj` | link5 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link5.mtl` | link5 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link6.glb` | link6 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link6.obj` | link6 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link6.mtl` | link6 OBJ material。 |
| `assets/robots/xarm7/meshes/visual/link7.glb` | link7 GLB visual。 |
| `assets/robots/xarm7/meshes/visual/link7.obj` | link7 OBJ visual。 |
| `assets/robots/xarm7/meshes/visual/link7.mtl` | link7 OBJ material。 |

### xArm7 + XHand / standalone XHand models

| 文件 | 职责 |
|---|---|
| `assets/robots/xhand/xarm7_xhand_right.urdf` | 完整 19-DoF arm + right-hand 模型，用于 FK、碰撞与 provenance。 |
| `assets/robots/xhand/xarm7_xhand_collision.urdf` | XHand 固定在 open/home 几何的 arm planning collision 模型。 |
| `assets/robots/xhand/xarm7_xhand.srdf` | 完整 arm-hand 模型的 self-collision disable pairs。 |
| `assets/robots/xhand/xhand_right.urdf` | standalone 12-DoF XHand 模型，用于 retarget 与 hand kinematics。 |

### XHand STL meshes

这些文件分别提供 `xhand_right.urdf` 和组合 URDF 中对应 link 的 visual/collision
geometry。

| 文件 | 对应 link/部件 |
|---|---|
| `assets/robots/xhand/meshes/flange.STL` | xArm-to-XHand flange。 |
| `assets/robots/xhand/meshes/right_hand_link.STL` | 手掌主 link。 |
| `assets/robots/xhand/meshes/right_hand_back_link.STL` | 手背 link。 |
| `assets/robots/xhand/meshes/right_hand_ee_link.STL` | hand end-effector reference link。 |
| `assets/robots/xhand/meshes/right_hand_thumb_bend_link.STL` | 拇指 bend link。 |
| `assets/robots/xhand/meshes/right_hand_thumb_rota_link1.STL` | 拇指 rotation link1。 |
| `assets/robots/xhand/meshes/right_hand_thumb_rota_link2.STL` | 拇指 rotation link2。 |
| `assets/robots/xhand/meshes/right_hand_thumb_rota_tip.STL` | 拇指 fingertip。 |
| `assets/robots/xhand/meshes/right_hand_thumb_rotaback_link1.STL` | 拇指 rear/support link1。 |
| `assets/robots/xhand/meshes/right_hand_thumb_rotaback_link2.STL` | 拇指 rear/support link2。 |
| `assets/robots/xhand/meshes/right_hand_index_bend_link.STL` | 食指 bend link。 |
| `assets/robots/xhand/meshes/right_hand_index_rota_link1.STL` | 食指 rotation link1。 |
| `assets/robots/xhand/meshes/right_hand_index_rota_link2.STL` | 食指 rotation link2。 |
| `assets/robots/xhand/meshes/right_hand_index_rota_tip.STL` | 食指 fingertip。 |
| `assets/robots/xhand/meshes/right_hand_index_rotaback_link1.STL` | 食指 rear/support link1。 |
| `assets/robots/xhand/meshes/right_hand_index_rotaback_link2.STL` | 食指 rear/support link2。 |
| `assets/robots/xhand/meshes/right_hand_mid_link1.STL` | 中指 link1。 |
| `assets/robots/xhand/meshes/right_hand_mid_link2.STL` | 中指 link2。 |
| `assets/robots/xhand/meshes/right_hand_mid_tip.STL` | 中指 fingertip。 |
| `assets/robots/xhand/meshes/right_hand_midback_link1.STL` | 中指 rear/support link1。 |
| `assets/robots/xhand/meshes/right_hand_midback_link2.STL` | 中指 rear/support link2。 |
| `assets/robots/xhand/meshes/right_hand_ring_link1.STL` | 无名指 link1。 |
| `assets/robots/xhand/meshes/right_hand_ring_link2.STL` | 无名指 link2。 |
| `assets/robots/xhand/meshes/right_hand_ring_tip.STL` | 无名指 fingertip。 |
| `assets/robots/xhand/meshes/right_hand_ringback_link1.STL` | 无名指 rear/support link1。 |
| `assets/robots/xhand/meshes/right_hand_ringback_link2.STL` | 无名指 rear/support link2。 |
| `assets/robots/xhand/meshes/right_hand_pinky_link1.STL` | 小指 link1。 |
| `assets/robots/xhand/meshes/right_hand_pinky_link2.STL` | 小指 link2。 |
| `assets/robots/xhand/meshes/right_hand_pinky_tip.STL` | 小指 fingertip。 |
| `assets/robots/xhand/meshes/right_hand_pinkyback_link1.STL` | 小指 rear/support link1。 |
| `assets/robots/xhand/meshes/right_hand_pinkyback_link2.STL` | 小指 rear/support link2。 |
