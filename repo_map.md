# DexMani Real Repository Map

本文件是当前版本控制树的职责索引。运行行为以源码、schema 与配置为准；新增、删除、
移动文件或改变主要职责时必须同步更新本文件。被 `.gitignore` 排除的 episode、dataset、
replay result 与缓存不在清单中。

## 1. 架构总览

```text
device input ──> sensor workers ──> ipc.RuntimeChannels ──> teleop/deployment/replay
hardware SDK <── robot workers <── control publication <── safety gate/action candidate

teleop sample ──> recording worker ──> raw episode
raw episode ──> data process ──> processed HDF5 ──> data export ──> Policy Zarr
```

所有重要 mutable resource 只有一个 owner：SDK 只在对应 worker/driver 内，IPC allocation
只在 `RuntimeChannels`，动作发布只在 `control/publication.py`，录制 transaction 只在
recording worker/recorder。当前架构门禁要求 package cycle、禁用依赖边和跨模块私有 import
均为零。

## 2. 顶层文件

| 文件 | 主要职责 |
|---|---|
| `AGENTS.md` | 仓库级工程、安全、范围与验收契约。 |
| `CLAUDE.md` | Claude Code 精简入口，具体规则委托给 `AGENTS.md`。 |
| `code_style.md` | 本研究代码库的具体编码与审查约定。 |
| `README.md` | 面向使用者的能力、架构、环境与工作流。 |
| `repo_map.md` | 当前文件与 owner 索引。 |
| `user_design.md` | 已确认的机器人行为与安全取舍。 |
| `.gitignore` | 生成物、本地环境、数据集与运行输出排除规则。 |
| `pyproject.toml` | 包元数据、依赖、package data 与工具配置。 |

## 3. `dexmani_real/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 包说明及 `PACKAGE_DIR`/`ASSET_DIR` 解析。 |
| `robot_spec.py` | 机器人 DoF/shape、XHand joint order、URDF/SRDF canonical 路径。 |

### `config/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 配置子包出口。 |
| `defaults.py` | arm、hand、policy、sensor 与 safety canonical defaults。 |
| `runtime.py` | CLI > YAML > defaults 合并、校验、冻结与 SHA-256 identity。 |
| `pointcloud.py` | 实时/离线共享的不可变点云策略与 identity。 |
| `camera_calib.py` | 校验并冻结相机序列号绑定的外参快照与来源 hash。 |
| `cameras.json` | 相机序列号、安装类型与外参。 |
| `desk_plane.json` | 桌面平面与 provenance。 |
| `vr_transform.json` | VR 到机器人坐标系的标定与质量信息。 |

### `ipc/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量 IPC 子包标记。 |
| `schema.py` | 稳定 NumPy wire dtype、record sample 与 point-cloud ABI。 |
| `ring.py` | 通用 seqlock shared-memory ring。 |
| `camera_ring.py` | 大尺寸 RGB-D shared-memory ring。 |
| `causal.py` | 按 observation anchor 读取 arm/hand/tactile/VR/camera。 |
| `channels.py` | `RuntimeChannels`：rings、queues、flags、heartbeat、readiness 的唯一 allocation owner。 |

### `runtime/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量运行时子包标记。 |
| `safety.py` | safety state、run generation 与合法状态转换。 |
| `status.py` | worker/supervisor 共用的结构化退出原因。 |
| `workers.py` | spawn-only worker spec、构建、启动与退出优先级。 |
| `supervisor.py` | readiness、heartbeat、进程健康、摘要与 verified shutdown。 |

### `robot/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量硬件子包标记。 |
| `types.py` | 校验后的 `RobotState` 与 `RobotAction`。 |
| `command_validation.py` | worker 在 SDK 调用前执行 generation/freshness/shape/expiry 复核。 |
| `xarm7.py` | xArm Python SDK 的唯一 driver 边界。 |
| `xhand.py` | XHand controller 的唯一 driver 边界。 |
| `arm_worker.py` | xArm Mode-6 command/home consumer 与状态 publisher。 |
| `hand_worker.py` | XHand latest-target servo、状态/触觉 publisher 与 fault latch。 |

### `sensor/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量传感器子包标记。 |
| `camera_geometry.py` | native RGB-D intrinsics 与 `T_color_from_depth` 数据合同。 |
| `clock_sync.py` | device clock 到 host monotonic 的保守映射。 |
| `realsense.py` | RealSense driver、native frame ownership copy 与设备配置。 |
| `camera_worker.py` | RealSense lifecycle、时序/健康信息与 camera ring 发布。 |
| `pointcloud.py` | SDK-free native RGB-D 到 xArm-base 点云算法。 |
| `pointcloud_worker.py` | latest-only 固定 `float32[N,6]` point-cloud publisher。 |
| `vr_worker.py` | crash-isolated Quest/HTS receiver 与 VR ring publisher。 |

### `calibration/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 标定子包标记。 |
| `table.py` | 桌面 RANSAC/最小二乘拟合与确认后的原子发布。 |
| `camera/__init__.py` | 相机标定子包标记。 |
| `camera/solver.py` | ArUco 检测、hand-eye 求解、残差筛选与标定持久化。 |
| `camera/control.py` | 标定 arm state、workspace clipping、gated publish 与 homing。 |
| `camera/session.py` | xArm/RealSense/GUI/采样 lifecycle 与失败 cleanup。 |

### `planning/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | planner 与核心类型出口。 |
| `types.py` | Pose、IK/path/collision result 与 planner profile。 |
| `poses.py` | pose、quaternion、matrix 与 rot6d 纯运算。 |
| `arm_fk.py` | xArm7 Pinocchio FK、Jacobian 与 pose transform。 |
| `hand_fk.py` | XHand Pinocchio fingertip FK。 |
| `ik.py` | MPlib position IK、seed 与 null-space 优化。 |
| `candidates.py` | IK candidate 生成、过滤、规范化、排序与选择。 |
| `collision.py` | xArm7+XHand 自碰撞、环境碰撞与 transition 检查。 |
| `paths.py` | 插值、wrap、densification 与 typed home path 结果。 |
| `planner.py` | 组合 IK、路径与 collision 的 motion-planner facade。 |

### `control/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量控制子包标记。 |
| `action.py` | backend-neutral `ActionCandidate` 与 representation 校验。 |
| `safety_gate.py` | generation、limits、delta、workspace 与 collision fail-closed gate。 |
| `publication.py` | controller feedback/runtime gate、序列化、发布与 acknowledgement。 |
| `arm_home.py` | collision-checked arm homing 合同、排队、等待与 abort。 |
| `hand_home.py` | exact hand-home 发布与 worker acknowledgement。 |

### `teleop/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 遥操作子包标记。 |
| `config.py` | 从 resolved runtime 投影的窄 teleop 配置。 |
| `session.py` | VR teleop 主进程 owner、预检、worker topology 与 cleanup。 |
| `loop.py` | coordinator 资源构造、readiness 与 operator/grid 调度。 |
| `control_grid.py` | 单个 causal grid 的读取、proposal、gate、publish 与 sample。 |
| `control_state.py` | coordinator mutable state 与 command-quiescence 转换。 |
| `operator_controls.py` | BEGIN/pause/home/quit 与录制 disposition。 |
| `keyboard.py` | 全局快捷键、control signal 与 EEF delta。 |
| `keyboard_session.py` | 键盘 teleop lifecycle 与实时控制流。 |
| `action_proposal.py` | 纯 EEF/arm/hand proposal 计算。 |
| `arm_mapper.py` | VR wrist 相对运动到 robot-frame EEF target。 |
| `hand_control.py` | hand observation cache、ramp 与 command helper。 |
| `safety.py` | pause/hold、re-anchor、feedback guard 与 configured home。 |
| `episode_samples.py` | 因果 snapshot 到 typed fixed-grid record sample。 |
| `recording_session.py` | 退出时有界保存/丢弃决策。 |
| `camera_freshness.py` | camera duplicate/gap/generation/freshness 分类。 |
| `vr_transform.py` | VR heading 标定 schema 与 rotation 校验。 |
| `timing.py` | 控制网格分阶段 timing 统计。 |
| `audio_feedback.py` | 非阻塞操作者语音反馈。 |
| `retarget/__init__.py` | retargeting 子包出口。 |
| `retarget/facade.py` | TAG/DexPilot 统一 facade、校验与状态。 |
| `retarget/dexpilot.py` | 带 human-flexion prior 的 DexPilot wrapper。 |
| `retarget/tag_optimizer.py` | 两阶段 NLopt TAG optimizer。 |
| `retarget/pin_grad.py` | Pinocchio analytical gradient engine。 |

### `recording/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | `EpisodeReader`/`EpisodeRecorder` 等稳定公开 facade。 |
| `schema.py` | raw episode v21 persisted schema。 |
| `frame.py` | IPC record 到 immutable owned `EpisodeFrame` 的唯一 decode 边界。 |
| `timeline.py` | 多速率输入到 fixed grid 的 timestamp 对齐与填充原因。 |
| `recorder.py` | raw episode transaction、质量汇总、验证与原子发布。 |
| `hdf5_writer.py` | 单个 `data.h5` handle、dataset append 与 offset owner。 |
| `camera_writer.py` | RGB/depth sidecar 的有界后台 writer。 |
| `video.py` | PyAV RGB 编解码与 codec 配置。 |
| `client.py` | controller 侧 recorder protocol、sample publication 与 stop result。 |
| `worker.py` | RecorderIO active generation、sequence continuity、transaction 与 finalize。 |
| `reader.py` | published v21 校验、HDF5 merged view 与 RGB/depth 读取。 |

### `data/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量离线数据子包标记。 |
| `contracts.py` | output/quality/bridge policy、processing config 与 decision。 |
| `quality.py` | 停滞、抖动、突变等 temporal quality 纯函数审计。 |
| `clean.py` | raw flags、limits、annotations 与质量结果到保留/拒绝决定。 |
| `transforms.py` | RGB/depth/intrinsics 的确定性数值变换。 |
| `process.py` | raw v21 到 processed HDF5 v5 的事务式管线。 |
| `export.py` | processed HDF5 到 Policy Zarr v3 的事务式导出。 |

### `deployment/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | learned-policy 部署子包标记。 |
| `contracts.py` | inference context、action chunk 与 `PolicyRuntime` protocol。 |
| `config.py` | deployment YAML/CLI、target 与 point-cloud contract 校验。 |
| `manifest.py` | checkpoint/config/runtime manifest 组装与 fail-closed 一致性检查。 |
| `observation.py` | 因果不可变 arm/hand/tactile/pointcloud history batch。 |
| `worker.py` | runtime factory 惰性加载、observation 校验、predict 与 plan 发布。 |
| `coordinator.py` | learned-policy 唯一 command producer、plan scheduling 与 watchdog。 |
| `lifecycle.py` | worker topology、readiness、supervision 与 verified shutdown。 |
| `operator.py` | B/S/H/Q/ESC 请求及 collision-checked homing 编排。 |
| `metrics.py` | inference/coordinator 指标与 reject 分类。 |
| `fake.py` | CPU-only deterministic fake runtime。 |

### `replay/`

| 文件 | 主要职责 |
|---|---|
| `__init__.py` | 轻量回放子包标记。 |
| `trajectory.py` | raw/processed trajectory 加载、provenance 与模型 preflight。 |
| `controller.py` | fixed-rate safety-gated physical replay 调度。 |
| `capture.py` | measured state、sent command 与 rejection 的有界捕获。 |
| `evaluation.py` | tracking/consistency 指标与结果持久化。 |
| `session.py` | worker、safety transition、operator flow、评估与 cleanup owner。 |

### `integrations/` 与 `utils/`

| 文件 | 主要职责 |
|---|---|
| `integrations/__init__.py` | 外部集成子包标记。 |
| `integrations/dexmani_policy.py` | 外部 `dexmani_policy` runtime 与 manifest adapter。 |
| `utils/__init__.py` | 轻量通用工具子包标记。 |
| `utils/atomic_io.py` | fsync、atomic publish 与 atomic JSON。 |
| `utils/feedback.py` | arm/hand feedback freshness、finite 与 health predicate。 |
| `utils/limits.py` | XHand rated/mechanical/command limit nesting。 |
| `utils/log.py` | logging、native stdout capture 与 throttled warning。 |
| `utils/rate.py` | absolute-deadline `LoopRate` 与 overrun 统计。 |
| `utils/serialization.py` | dataclass mapping 恢复 helper。 |
| `utils/smoothing.py` | quaternion-aware Cartesian pose EMA。 |

## 4. 可执行入口

`examples/` 是薄入口或诊断程序，不是测试；除明确离线工具外均按可能影响硬件处理。

| 文件 | 主要职责 |
|---|---|
| `collect_teleop.py` | VR teleop 数据采集入口。 |
| `keyboard_teleop.py` | keyboard teleop 入口。 |
| `run_policy.py` | learned-policy deployment 入口。 |
| `replay_episode.py` | 物理回放入口。 |
| `process_episodes.py` | raw → processed HDF5 离线处理。 |
| `export_policy_zarr.py` | processed HDF5 → Policy Zarr 离线导出。 |
| `visualize_episode.py` | raw episode 离线可视化。 |
| `visualize_episode_processed.py` | processed episode 离线可视化。 |
| `calibrate_camera.py` | xArm+RealSense hand-eye 标定入口。 |
| `calibrate_vr_heading.py` | VR heading 采集、质量门禁与 transform 发布。 |
| `realsense_record_example.py` | RealSense RGB-D/point-cloud 交互诊断。 |
| `pointcloud_process_example.py` | L515 桌面点云诊断与显式确认后的 plane 发布。 |
| `xhand_control_example.py` | 使用 canonical hand 限位的 XHand 独立硬件诊断。 |

## 5. 离线验证与架构门禁

| 路径 | 主要职责 |
|---|---|
| `tools/check_architecture.py` | AST-only package cycle、forbidden edge、private import、defaults 与 heavy-init 门禁。 |
| `tests/architecture/` | 架构指标 ratchet 与主要运行入口/公开 API import smoke test。 |
| `tests/config/` | immutable runtime/calibration snapshot、spawn pickle 只读性与配置传播。 |
| `tests/control/` | fail-closed feedback health/publication。 |
| `tests/examples/` | 可执行入口的离线生命周期与非关键反馈降级测试。 |
| `tests/ipc/` | ABI manifest、ring exact dtype/shape 与重复共享内存名称的 fail-closed 门禁。 |
| `tests/planning/` | collision/SRDF fail-closed 行为。 |
| `tests/recording/` | raw/processed/Zarr schema contract。 |
| `tests/fixtures/contracts/` | 冻结的 architecture、IPC ABI 与 storage schema manifest。 |

## 6. 静态资源

- `assets/robots/xarm7/`：xArm7 URDF、SRDF 与 visual/collision meshes。
- `assets/robots/xhand/`：xArm7+XHand/standalone XHand URDF、SRDF 与 meshes。
- `assets/retargeting/`：DexPilot XHand 配置。
- `assets/audio/`：teleop 中文状态提示音。

模型与标定资源路径必须通过 `robot_spec.py` 或 canonical config 解析，不在 worker、
entry script 或文档中复制第二份默认值。
