# dexmani_real vs ManiUniCon 全面对比分析

> **日期**: 2026-06-22 | **对比版本**: dexmani_real (main), ManiUniCon (Reference)

---

## 1. Executive Summary

### 1.1 总览对比表

| 维度 | dexmani_real | ManiUniCon | 优势方 |
|------|-------------|------------|--------|
| **架构模式** | 单线程 50Hz 顺序循环 | 多线程接收/控制/插值分离 | ManiUniCon |
| **编程语言** | Python 3 | Python 3 | 持平 |
| **VR 输入** | Meta Quest (HTS SDK) | Meta Quest (raw TCP) | 持平 |
| **VR 帧率** | 50 Hz (HTS 原生) | ~30 Hz → 插值 200Hz | ManiUniCon |
| **手臂 IK** | DLS + MPlib Position IK 回退 | Pinocchio IK (task-driven) | dexmani |
| **IK 后端** | Pinocchio + MPlib | Pinocchio | 持平 |
| **手部重定向** | DexPilot + XHandRefAdapter | DexPilot (Quest) | dexmani |
| **通信协议** | 直接 Python API 调用 | SharedMemoryQueue (跨进程) | ManiUniCon |
| **安全机制** | ★★★★★ 四层 + 10bit QualityFlags | ★★★★☆ 三层: 位置/方向/ workspace 限制 | dexmani |
| **数据录制** | HDF5 (episode + quality flags) | Zarr/LeRobot v3.0 (multi-ep) | ManiUniCon |
| **双手支持** | 右手 only | 右手 only | 持平 |
| **插值框架** | 无 (EMA 可选, 默认关闭) | PoseTrajectoryInterpolator 200Hz | ManiUniCon |
| **频率控制** | RateLimiter (50Hz 单循环) | 策略层 30Hz → 插值 200Hz → 硬件 200Hz | ManiUniCon |
| **进程模型** | 单进程 | 多进程 (相机独立) | ManiUniCon |
| **配置系统** | Python dataclass (PipelineConfig) | Hydra YAML (递归实例化) | ManiUniCon |
| **成熟度评分** | ★★★★☆ (4/5) | ★★★★☆ (4/5) | 持平 |

### 1.2 Top 5 可采纳改进

| # | 改进项 | 来源 | 优先级 | 预期影响 |
|---|--------|------|--------|----------|
| 1 | 增加 EEF 方向工作空间边界 (Orientation Bounds) | ManiUniCon | **P0** | 防止 wrist 极值姿态导致自碰撞/ wrist 奇点 |
| 2 | 录制帧数硬上限 max_record_frames | ManiUniCon | **P0** | 防止录制卡死导致磁盘耗尽 |
| 3 | VR 帧间 Cartesian Pose 插值 | ManiUniCon | **P1** | 消除 stale reuse，平滑运动，减少抖动 |
| 4 | 中间恢复路径 REARM (不重启脚本) | ManiUniCon | **P1** | 减少瞬态 IK 失败导致的全流程重启 |
| 5 | 集中化 validate_action() 安全检查门 | ManiUniCon | **P1** | 统一安全逻辑，防止检查遗漏 |

---

## 2. ManiUniCon 架构总览

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ManiUniCon Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────────┐     30 Hz      ┌──────────────────────────────┐   │
│  │ Meta Quest VR    │───────────────→│ SharedMemoryStorage          │   │
│  │ (quest_controller)│  write_action  │  write_action() / read_all() │   │
│  └──────────────────┘                └──────────┬───────────────────┘   │
│                                                  │                        │
│  ┌──────────────────┐     30 Hz                 │ 30 Hz                 │
│  │ QuestPolicy      │───────────────────────────┘                        │
│  │ policies/quest.py│  run() → _calculate_action()                       │
│  │                  │  → delta computation (pos + rot)                   │
│  │                  │  → _check_safety_limits()                          │
│  │                  │  → _apply_safety_limits()                          │
│  └────────┬─────────┘                                                    │
│           │ target pose (30 Hz)                                           │
│           ▼                                                               │
│  ┌──────────────────────┐  200 Hz   ┌──────────────────────────────┐    │
│  │ PoseTrajectory       │──────────→│ IKSolver                     │    │
│  │ Interpolator         │ schedule  │  solve() → _create_tasks()   │    │
│  │ drive_to_waypoint()  │ waypoint  │  Pinocchio IK (task-driven)  │    │
│  │ schedule_waypoint()  │           │                              │    │
│  └──────────────────────┘           └──────────────┬───────────────┘    │
│                                                     │ qpos target         │
│                                                     ▼                     │
│  ┌──────────────────────────────┐  200 Hz  ┌─────────────────────────┐  │
│  │ JointSpaceSmoother           │          │ Robot.run() Control Loop │  │
│  │ filter.py:77-138 smooth()    │          │ core/robot.py:279-466    │  │
│  └──────────────────────────────┘          │                          │  │
│                                             │  ┌────────────────────┐ │  │
│  ┌──────────────────────────────┐          │  │ state receiver     │ │  │
│  │ Camera Processes (独立)      │          │  │ (thread, :174-220) │ │  │
│  │ 多进程 SharedMemoryRingBuffer│          │  └────────────────────┘ │  │
│  └──────────────────────────────┘          │  ┌────────────────────┐ │  │
│                                             │  │ interpolator       │ │  │
│  ┌──────────────────────────────┐          │  │ scheduling         │ │  │
│  │ Zarr / LeRobot Exporter      │          │  │ (:405-437)         │ │  │
│  │ 多 episode + 归一化统计      │          │  └────────────────────┘ │  │
│  └──────────────────────────────┘          └─────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ Robot Control System (main.py:163-170)                              │ │
│  │  RobotControlSystem → 启动所有子系统, 管理生命周期                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流详解

```
Quest VR(30Hz)          QuestPolicy(30Hz)           IK Solver(200Hz)          Robot(200Hz)
    │                        │                          │                        │
    ├─wrist pos/quat────────→│                          │                        │
    │                        ├─R_ve 构造(:46-53)        │                        │
    │                        ├─delta 计算(:192-338)     │                        │
    │                        ├─safety check(:134-158)   │                        │
    │                        ├─safety apply(:160-190)   │                        │
    │                        ├─write_action(:382)──────→│                        │
    │                        │                          ├─interpolator(:405-437) │
    │                        │                          ├─smooth(:77-138)       │
    │                        │                          ├─IK solve(:171-217)───→│
    │                        │                          │  ├─_create_tasks       │
    │                        │                          │  └─Pinocchio IK        │
    │                        │                          │                        └→ hardware
    │                        │                          │
    │                        │                          │ state_receiver(:174-220)
    │                        │                          │ ← Arm + Hand state
    │                        │                          │
    │  Camera Process ──────────────────────────────────┘
    │  (独立 mp.Process, 不阻塞控制回路)
```

---

## 3. 七个维度详细对比

### 3.1 进程架构

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **主控制线程** | 单线程 while self.running → _tick() | 多线程: 主控制 + state_receiver + 键盘监听 | ManiUniCon 的 state_receiver 线程(:174-220) 将状态读取从控制循环中解耦 |
| **频率控制** | RateLimiter(50Hz) 等待式限速 | 策略层 30Hz、插值 200Hz、硬件 200Hz 三层解耦 | ManiUniCon 的频率层级更合理: 慢 VR 输入被插值填满 |
| **相机进程** | 同步 add_frame(camera_frame=...) 阻塞 | 独立 mp.Process + SharedMemoryRingBuffer | ManiUniCon 相机崩溃不会 kill 控制器 |
| **键盘处理** | 线程 pynput → mp.Queue → poll() | 线程 keyboard → mp.Queue → get() | 类似，但 ManiUniCon 的 reset_event('h'键) 可中途重置 |
| **危机恢复** | EMERGENCY_STOP → running=False 必须重启 | reset_event → reset_to_init() + sync_state() 进程内恢复 | ManiUniCon 容错性更好 |
| **生命周期管理** | TeleopController 自管理 | RobotControlSystem (main.py:163-170) 集中管理 | ManiUniCon 结构更清晰 |

**关键代码路径对比:**

- dexmani: `controller.py:167-193 run()` 单循环，`controller.py:199-285 _tick()` 每个 step 串行执行全部任务
- ManiUniCon: `robot.py:279-466 run()` 控制循环 + `robot.py:174-220` state_receiver 线程并行接收状态

### 3.2 IK 求解

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **主 IK 算法** | Damped Least-Squares (DLS) 差分 IK | Pinocchio task-driven IK | dexmani 使用 DLS 伪逆，ManiUniCon 使用 Pinocchio 内置 IK |
| **回退策略** | Position IK (MPlib) 两个 seed (prev_cmd / current_qpos) | 无（单策略） | dexmani 的两层回退更鲁棒 |
| **阻尼策略** | 固定 λ²=0.0004 | 近零阻尼 1e-12 (QP 求解器) | ManiUniCon 的 QP 求解器自然处理秩亏，dexmani 固定阻尼有 ~1-2mm 偏差 |
| **可操作性利用** | compute_manipulability() 已实现但未在 teleop 热点使用 | QP 求解器隐式处理 | 两者都未显式利用可操作性做自适应 |
| **自碰撞检测** | TeleopProfile.check_self_collision + ik_mgr.has_self_collision() | 未在 IK 路径中检查 | dexmani 更安全 |
| **Fingertip 桌面安全** | FingertipDeskSafety FK 检测 (:732-742) | 未实现 | dexmani 独有 |
| **速度限制** | XArm7._limit_joint_step() 驱动层限制 | 插值器速度限制 (0.25 m/s, 0.5 rad/s) | 分层不同: dexmani 驱动层, ManiUniCon 规划层 |
| **IK 失败处理** | hold 前一帧命令 + error_handler 记录 | hold (默认行为) | 类似 |

**关键代码路径对比:**

- dexmani: `ik.py:46 solve()` → `ik.py:231-288 solve_differential_ik()` (DLS) → `ik.py:115-176 solve_position_ik()` (MPlib fallback)
- ManiUniCon: `ik_solver.py:171-217 solve()` → `ik_solver.py:103-124 _create_tasks()` (构建 Pinocchio IK 任务)

### 3.3 VR 控制 (手臂映射)

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **坐标映射方式** | 重置相对式 (ArmWristMapper) | 增量式 (_calculate_action delta) | dexmani 使用 T=0 锚点偏移, ManiUniCon 使用帧间增量 |
| **比例缩放** | pos_scale + rot_scale (可配置) | 固定 1:1 | dexmani 更灵活 |
| **坐标系旋转变换** | vr_to_base_rot (VR→基座) | R_ve 构造 (VR→EEF) | 类似，变体形式不同 |
| **位置 delta 限制** | eef_delta_bounds (shape (3,2)) | max_delta_pos=0.5m 单值上限 | dexmani 更精细 (每轴独立), ManiUniCon 有 per-frame cap |
| **旋转 delta 限制** | 无 per-step 旋转上限 | max_delta_rot=1.0rad | ManiUniCon 更安全 (防止 VR 跟踪跳变) |
| **四元数连续性** | continuous_quat() 符号检测 | 默认 quat 连续性 | dexmani 显式处理 ≥0 符号翻转 |
| **VR 方向安全性** | 无方向工作空间检查 | _check_safety_limits(:134-158) + _apply_safety_limits(:160-190) | ManiUniCon 独有，防止 wrist 极值姿态 |
| **VR 重置** | reset() → 锚定当前位置 | sync_state() → 同步最新 state | 类似 |

**关键代码路径对比:**

- dexmani: `arm_mapper.py:49-75 map()` → 位置偏移 `delta_pos_vr = wrist_pos - wrist_pos0`, 旋转偏移 `delta_rot_vr = wrist_rot @ wrist_rot0.T`, 缩放后变换到基座坐标系
- ManiUniCon: `quest_controller.py:46-53` R_ve 构造 → `quest_controller.py:192-338 _calculate_action()` delta 计算 → `quest_controller.py:134-158 _check_safety_limits()` → `quest_controller.py:160-190 _apply_safety_limits()`

### 3.4 频率与插值

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **VR 帧率** | 50 Hz (HTS 原生) | 30 Hz (Quest raw TCP) | dexmani VR 原始帧率更高 |
| **控制帧率** | 50 Hz (1:1 跟随 VR) | 200 Hz (插值填满) | ManiUniCon 控制频率 4x |
| **VR→控制桥接** | 直接重读同一 VR 帧 (stale reuse) | PoseTrajectoryInterpolator 插值 | ManiUniCon 无 stale reuse |
| **位置插值** | 无 | 线性插值 (通过时间参数化) | ManiUniCon 独有 |
| **旋转插值** | 无 | SLERP (四元数球面线性插值) | ManiUniCon 独有 |
| **速度限制** | XArm7 驱动层 bottleneck scaling | 插值器速度限制 (0.25 m/s pos, 0.5 rad/s rot) | 分层不同 |
| **EMA 平滑** | ema_alpha_arm (默认 1.0 = 关闭) | JointSpaceSmoother (filter.py:77-138) | ManiUniCon 的 JointSpaceSmoother 比 EMA 更复杂 |
| **数据缓存** | 无插值缓存 | schedule_waypoint() deque 缓存 | ManiUniCon 支持异步调度 |

**关键代码路径对比:**

- dexmani: `controller.py:199 _tick()` → 每帧调用 `_read_vr_frame()` 获取最新 VR 帧 → 直接计算 IK, 无插值
- ManiUniCon: `quest_controller.py:190-377 run()` 30Hz 输出 target pose → `pose_trajectory_interpolator.py:103-185 schedule_waypoint()` 调度 → `pose_trajectory_interpolator.py:187-207 __call__()` 200Hz 插值 → IK solver

### 3.5 安全机制

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **安全层数** | 4 层: 跟踪质量 / workspace / joint limit / 质量标记 | 3 层: position limit / orientation limit / torque limit | dexmani 更全面 |
| **质量标记** | 10-bit QualityFlags (TRACKING_OK, IK_SUCCESS, RETARGET_OK, RETARGET_VALID, JOINT_JUMP_OK, IN_WORKSPACE, ARM_TORQUE_OK, HAND_CURRENT_OK, HAND_TEMP_OK, HAND_COMM_OK) | 无显式质量标记系统 | dexmani 独有，可过滤高质量数据 |
| **Workspace 检查** | EEF 位置 bounds (3,2) + IN_WORKSPACE flag | 位置 workspace + 方向 workspace (Euler) | ManiUniCon 的方向检查 dexmani 缺失 |
| **关节限制** | check_arm_joint_limits → E-Stop, check_hand_joint_limits → 警告 | 内置 (驱动层) | dexmani 更严格 |
| **Torque 检查** | check_arm_torque 每关节 20-50 Nm (ARM_TORQUE_OK flag) | validate_action 中统一检查 | dexmani 更细粒度 |
| **手部安全检查** | check_hand_current (500mA), check_hand_temperature (70°C), check_hand_comm (board error) | 未显式实现 | dexmani 独有 |
| **Joint Jump Clamp** | _ARM_JUMP_LIMIT_RAD = 5°/frame, _HAND_JUMP_LIMIT_RAD = 10°/frame | 未显式 clamp | dexmani 独有 |
| **安全阀门集中化** | 分散在 controller._tick() + controller._compute_action() + safety.py + interface.py | validate_action() 统一门 | ManiUniCon 集中化更清晰 |
| **Fingertip 桌面安全** | FingertipDeskSafety FK 检测 | 未实现 | dexmani 独有 |
| **E-Stop 行为** | running=False + robot.emergency_stop() | 进程级终止 | 类似 |

**关键代码路径对比:**

- dexmani: safety 检查分散在三个位置:
  - `controller.py:222-226` — state observation 质量检查 (ARM_TORQUE_OK, HAND_CURRENT_OK, HAND_TEMP_OK, HAND_COMM_OK)
  - `controller.py:229-252` — joint limit 硬安全检查 (arm→E-Stop, hand→warning)
  - `controller.py:319-322` — IN_WORKSPACE check (hold on failure)
- ManiUniCon: 安全检查集中在 quest_controller:
  - `quest_controller.py:134-158 _check_safety_limits()` — 位置 + 方向 workspace
  - `quest_controller.py:160-190 _apply_safety_limits()` — clamps

### 3.6 录制系统

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **数据格式** | HDF5 (.h5), 单 episode | Zarr + LeRobot v3.0, 多 episode | 不同生态: dexmani 独立格式, ManiUniCon 兼容 Diffusion Policy |
| **质量标记** | 10-bit per-frame QualityFlags | 无 per-frame 质量标记 | dexmani 可基于质量过滤训练数据 |
| **录制结构** | /obs/arm_qpos(N,7), /action/arm_qpos(N,7), /vr/..., /quality_flags, /camera/ | /data/obs, /data/action, /meta/norm_stats | dexmani 更丰富的传感器数据 |
| **传感器覆盖** | arm_qpos/qvel/tau, eef_pos/quat, hand_qpos/current/tactile/temperature, fingertip_pos, VR landmarks, camera RGBD | arm_qpos, hand_qpos (基础) | dexmani 数据维度远高于 ManiUniCon |
| **帧数限制** | 无上限 (可无限录制) | max_record_steps=5000 | ManiUniCon 有硬上限防磁盘耗尽 |
| **压缩方式** | HDF5 gzip chunked | blosc (Zarr) | ManiUniCon 压缩率更高 |
| **归一化统计** | 无 | 预计算 obs_mean/std, action_mean/std | ManiUniCon 可直接训练 |
| **相机录制** | 同步 add_frame(camera_frame=...) | 独立 Camera Process | ManiUniCon 异步不阻塞 |
| **配置快照** | PipelineConfig.to_dict() → HDF5 /meta | Hydra merged YAML → output dir | 两者都可追溯 |
| **Episodic 组织** | 自动编号 episode_000.h5 / episode_001.h5 | Zarr group 多 episode | 不同组织方式 |

**关键代码路径对比:**

- dexmani: `recording/episode_recorder.py:37-260` EpisodeRecorder → `episode_recorder.py:120 add_frame()` → 按 HDF5 group 组织
- ManiUniCon: `shared_memory/shared_storage.py:382 write_action()` → `shared_storage.py:404 read_all_action()` → Zarr 多 episode 导出

### 3.7 配置系统

| 对比项 | dexmani_real | ManiUniCon | 差异分析 |
|--------|-------------|------------|----------|
| **配置框架** | Python @dataclass (PipelineConfig + RobotInterfaceConfig + PlanningProfile + TeleopProfile) | Hydra (YAML + OmegaConf) | Hydra 更标准化, dataclass 更类型安全 |
| **配置源** | Python 源码 (dataclass instantiation) | YAML 文件 (configs/*.yaml) | ManiUniCon 无需修改 Python 代码 |
| **序列化/反序列化** | to_dict() 支持, from_dict() 未实现 | 自动 YAML→object→YAML 往返 | ManiUniCon 配置可完全还原 |
| **递归实例化** | 手动逐层构造 (~25 LOC) | Hydra instantiate() 递归自动 | ManiUniCon 更便捷 |
| **配置层级** | PipelineConfig → RobotInterfaceConfig → XArm7Config/XHandConfig → CollisionConfig | default.yaml → robot/xarm6.yaml → policy/quest.yaml | 类似三层结构 |
| **配置版本管理** | 无内置版本 | YAML 文件天然支持 git diff | ManiUniCon 可 diff 配置变更 |
| **频率配置** | control_rate_hz=50.0 (单一) | 三层: policy 30Hz / interpolator 200Hz / hardware 200Hz | ManiUniCon 更细粒度 |
| **Buffer 配置** | 隐式 (无 buffer 配置) | shared_memory buffer 大小配置 | ManiUniCon 显式管理 |

**关键代码路径对比:**

- dexmani: `config/pipeline_config.py:35-59 PipelineConfig` — dataclass 聚合 robot/planning_profile/teleop_profile
- ManiUniCon: `configs/default.yaml:1-8` — Hydra YAML 频率/buffer 配置, `configs/robot/xarm6.yaml:1-48`, `configs/policy/quest.yaml:1-13`

---

## 4. 可采纳改进 (优先级排序)

### P0 — 紧急: 安全/可靠性缺失

#### P0-1: 增加 EEF 方向工作空间边界 (Orientation Workspace Bounds)

**来源**: ManiUniCon `quest_controller.py:134-158 _check_safety_limits()` + `quest_controller.py:160-190 _apply_safety_limits()`

**问题描述**: dexmani 的 WorkspaceSafety (`planner.py:638-665`) 仅检查 EEF 位置 (x, y, z)。当 wrist 姿态在位置边界内但方向处于极值 (±180° roll 或极端 pitch) 时，可能导致:
1. 手臂连杆自碰撞 (arm links hit each other)
2. wrist 奇点接近 (J6-J7 对齐)
3. 位置 workspace 无法捕获

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/planner.py` 的 WorkspaceSafety 类中添加方向检查:

```python
class WorkspaceSafety:
    """EEF workspace bounds checking and clamping.
    
    workspace_bounds: (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters.
    orientation_bounds: (3, 2) array [[roll_min, roll_max], [pitch_min, pitch_max], [yaw_min, yaw_max]] in radians.
        None disables orientation checking (backward compatible).
    """

    def __init__(self, workspace_bounds: np.ndarray, orientation_bounds: np.ndarray | None = None) -> None:
        self.bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if self.bounds.shape != (3, 2):
            raise ValueError(f"workspace_bounds must have shape (3, 2), got {self.bounds.shape}.")
        self.ori_bounds = None if orientation_bounds is None else np.asarray(orientation_bounds, dtype=np.float64)

    def check_orientation(self, eef_quat_wxyz: np.ndarray) -> bool:
        """Check whether EEF orientation (as Euler XYZ) is within orientation bounds."""
        if self.ori_bounds is None:
            return True
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        return bool(
            (euler[0] >= self.ori_bounds[0, 0]) and (euler[0] <= self.ori_bounds[0, 1])
            and (euler[1] >= self.ori_bounds[1, 0]) and (euler[1] <= self.ori_bounds[1, 1])
            and (euler[2] >= self.ori_bounds[2, 0]) and (euler[2] <= self.ori_bounds[2, 1])
        )

    def clamp_orientation(self, eef_quat_wxyz: np.ndarray) -> np.ndarray:
        """Clip EEF orientation to orientation bounds."""
        if self.ori_bounds is None:
            return np.asarray(eef_quat_wxyz)
        from scipy.spatial.transform import Rotation
        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        euler = np.clip(euler, self.ori_bounds[:, 0], self.ori_bounds[:, 1])
        clamped_quat_xyzw = Rotation.from_euler('XYZ', euler).as_quat()
        return xyzw_to_wxyz(clamped_quat_xyzw)
    
    # ... existing check() and clamp() methods unchanged
```

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/types.py` 的 RobotInterfaceConfig 中添加字段:

```python
@dataclass
class RobotInterfaceConfig:
    # ... existing fields ...
    
    # Orientation workspace safety (Euler XYZ, radians).
    # None disables orientation checking (backward compatible).
    workspace_orientation_bounds: np.ndarray | None = None
```

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:_compute_action()` 中添加方向检查 (在 320 行 workspace 检查之后):

```python
# Workspace check on computed arm command
arm_eef_pos = self.planner.compute_eef_pose_world(arm_cmd).p
in_workspace = self.robot.check_workspace(arm_eef_pos)
# NEW: Orientation workspace check
ori_ok = self.robot.check_workspace_orientation(arm_eef_quat)
quality.set(IN_WORKSPACE, in_workspace and ori_ok)
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/planner.py:638-665` (WorkspaceSafety 类)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/types.py:112-151` (RobotInterfaceConfig)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/interface.py:249-252` (check_workspace → 增加方向)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:319-322` (_compute_action)

**预计工作量**: 2-3 小时

---

#### P0-2: 录制帧数硬上限 max_record_frames

**来源**: ManiUniCon max_record_steps=5000

**问题描述**: dexmani 的 EpisodeRecorder (`recording/episode_recorder.py:37-260`) 无帧数上限。如果操作员忘记按下停止键或键盘输入卡住，录制将无限增长直至磁盘耗尽。

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/recording/episode_recorder.py` 中修改:

```python
class EpisodeRecorder:
    def __init__(
        self,
        data_dir: str,
        max_frames: int = 10000,  # NEW: hard limit
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames  # NEW
        # ... rest unchanged
    
    def add_frame(
        self,
        state: RobotState,
        action: RobotAction,
        vr_frame: dict[str, Any],
        quality_flags: int,
        camera_frame: dict[str, Any] | None = None,
        T_base_eef: np.ndarray | None = None,
    ) -> bool:
        if not self._recording or self._file is None:
            return False
        
        # NEW: Early stop on max frame count
        if self._frame_count >= self.max_frames:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "Episode reached max_frames=%d, auto-stopping.", self.max_frames
            )
            self._file.attrs["stopped_reason"] = "max_frames"
            self.stop_episode(success=True)
            return True
        
        # ... rest unchanged
```

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py` 中添加:

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    max_record_frames: int = 10000  # NEW: hard cap on recording frames per episode
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/recording/episode_recorder.py:37-60` (EpisodeRecorder.__init__)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/recording/episode_recorder.py:120-130` (add_frame early-stop)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py:35-59` (PipelineConfig)

**预计工作量**: 30 分钟

---

### P1 — 高: 功能增强/用户体验

#### P1-1: VR 帧间 Cartesian Pose 插值

**来源**: ManiUniCon PoseTrajectoryInterpolator (`pose_trajectory_interpolator.py:78-207`)

**问题描述**: dexmani 的 50Hz 控制循环在 VR 更新帧率 ~25-30Hz 时会重读同一 VR 帧 2 次 (stale reuse)，导致:
1. 两帧相同的命令 → 不流畅的运动
2. 人手抖动 (~2-3mm, 8-12Hz) 和 VR 跟踪噪声直接传播到机器人
3. 关节空间 EMA (默认关闭) 只能后 IK 缓解，无法消除 Cartesian 源头的抖动

**实现指导**:

新建 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/vr/pose_interpolator.py`:

```python
"""Cartesian pose interpolator — smooths between discrete VR frames."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class CartPoseInterpolator:
    """Interpolates between discrete VR-frame poses for smooth robot motion.
    
    Receives target poses at VR rate (~30 Hz) and produces interpolated
    poses at the controller's sampling rate (50 Hz) via:
      - Linear interpolation for position
      - SLERP for rotation
      - Speed-limited temporal scheduling
    """

    def __init__(
        self,
        max_pos_speed: float = 0.25,  # m/s
        max_rot_speed: float = 0.5,   # rad/s
        max_history: int = 5,
    ) -> None:
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self._waypoints: deque[tuple[float, np.ndarray, np.ndarray]] = deque(maxlen=max_history)
        self._last_pos: np.ndarray | None = None
        self._last_rot: Rotation | None = None
        self._earliest_arrival_time: float = 0.0

    def push_target_pose(
        self, pos: np.ndarray, quat_wxyz: np.ndarray, timestamp: float | None = None
    ) -> None:
        """Enqueue a new target waypoint (called at VR frame rate)."""
        ts = timestamp if timestamp is not None else time.monotonic()
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
        
        # Compute earliest arrival respecting speed limits
        if self._last_pos is not None and self._last_rot is not None:
            pos_dist = float(np.linalg.norm(pos - self._last_pos))
            rot_dist = self._rotation_distance(quat_wxyz, self._last_rot)
            pos_time = pos_dist / self.max_pos_speed
            rot_time = rot_dist / self.max_rot_speed
            travel_time = max(pos_time, rot_time)
            self._earliest_arrival_time = max(ts, self._earliest_arrival_time + travel_time)
        else:
            self._earliest_arrival_time = ts
        
        self._last_pos = pos.copy()
        self._last_rot = Rotation.from_quat(
            np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        )
        self._waypoints.append((self._earliest_arrival_time, pos.copy(), quat_wxyz.copy()))

    def get_interpolated_pose(self, now: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """Get interpolated pose at current time (called at controller rate)."""
        if len(self._waypoints) < 2:
            if len(self._waypoints) == 1:
                _, pos, quat = self._waypoints[0]
                return pos.copy(), quat.copy()
            return None
        
        now = now if now is not None else time.monotonic()
        
        # Purge stale waypoints
        while len(self._waypoints) > 1 and self._waypoints[1][0] < now:
            self._waypoints.popleft()
        
        if len(self._waypoints) < 2:
            return None
        
        t_prev, pos_prev, quat_prev = self._waypoints[0]
        t_next, pos_next, quat_next = self._waypoints[1]
        
        if t_next <= t_prev:
            return pos_prev.copy(), quat_prev.copy()
        
        alpha = (now - t_prev) / (t_next - t_prev)
        alpha = max(0.0, min(1.0, alpha))
        
        # Linear pos interpolation
        interp_pos = pos_prev + alpha * (pos_next - pos_prev)
        
        # SLERP rotation interpolation
        rot_prev = Rotation.from_quat([quat_prev[1], quat_prev[2], quat_prev[3], quat_prev[0]])
        rot_next = Rotation.from_quat([quat_next[1], quat_next[2], quat_next[3], quat_next[0]])
        slerp = Slerp([t_prev, t_next], Rotation.concatenate([rot_prev, rot_next]))
        interp_rot = slerp(now)
        interp_quat_xyzw = interp_rot.as_quat()
        interp_quat = np.array([interp_quat_xyzw[3], interp_quat_xyzw[0], 
                                 interp_quat_xyzw[1], interp_quat_xyzw[2]])
        
        return interp_pos, interp_quat / np.linalg.norm(interp_quat)

    def _rotation_distance(self, quat_wxyz: np.ndarray, last_rot: Rotation) -> float:
        """Angular distance between two rotations in radians."""
        rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        delta = rot * last_rot.inv()
        angle = np.linalg.norm(delta.as_rotvec())
        return float(angle)

    def reset(self) -> None:
        self._waypoints.clear()
        self._last_pos = None
        self._last_rot = None
        self._earliest_arrival_time = 0.0
```

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py` 中集成插值器:

```python
class TeleopController:
    def __init__(
        self,
        # ... existing params ...
        use_cartesian_interpolation: bool = False,  # NEW
    ) -> None:
        # ... existing init ...
        self._pose_interpolator = CartPoseInterpolator() if use_cartesian_interpolation else None
        self._use_cartesian_interpolation = use_cartesian_interpolation

    def _compute_arm_command(
        self, vr_frame: dict, state: RobotState, prev_arm_cmd: np.ndarray, quality: QualityFlags,
    ) -> tuple[np.ndarray, bool, np.ndarray | None]:
        # ... existing code ...
        if self.arm_mapper.is_ready():
            mapped = self.arm_mapper.map(wrist_pos, wrist_quat_wxyz)
            if mapped is not None:
                target_eef_pos = mapped["pos"]
                target_eef_quat = mapped["quat_wxyz"]
                
                # NEW: Cartesian interpolation
                if self._pose_interpolator is not None:
                    self._pose_interpolator.push_target_pose(
                        target_eef_pos, target_eef_quat
                    )
                    result = self._pose_interpolator.get_interpolated_pose()
                    if result is not None:
                        target_eef_pos, target_eef_quat = result
                
                # ... rest of IK logic unchanged ...
```

**需修改文件**:
- NEW: `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/vr/pose_interpolator.py` (~150 LOC)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:98-115` (__init__ 添加插值器)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:349-394` (_compute_arm_command 集成)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/types.py:107-133` (TeleopProfile 添加插值配置)

**预计工作量**: 1-2 天

---

#### P1-2: 中间恢复路径 REARM (不重启脚本)

**来源**: ManiUniCon reset_event (键盘 'h' 或 Quest 'A' 按钮)

**问题描述**: dexmani 的 EMERGENCY_STOP (`controller.py:591-596`) 设置 `self.running = False`, 需要完全重启脚本。瞬态 IK 失败或手部通信恢复后仍需全流程重启。

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/control/keyboard.py` 中添加 REARM 信号:

```python
class ControlSignal(Enum):
    TELEOP = "T"
    RECORD = "R"
    STOP = "S"
    HOME = "H"
    EMERGENCY_STOP = "ESC"
    REARM = "REARM"   # NEW
    QUIT = "Q"

_KEY_MAP = {
    "t": ControlSignal.TELEOP,
    "r": ControlSignal.RECORD,
    "s": ControlSignal.STOP,
    "h": ControlSignal.HOME,
    "x": ControlSignal.REARM,   # NEW: 'x' key re-arm
    "q": ControlSignal.QUIT,
}
```

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py` 中添加:

```python
def _transition(self, signal: ControlSignal) -> None:
    # ... existing signals ...
    
    if signal == ControlSignal.REARM:
        self._rearm()
        return
    
    # ... rest unchanged ...

def _rearm(self) -> None:
    """Re-arm from EMERGENCY_STOP without script restart.
    
    Clears errors, resets tracking, transitions to IDLE.
    No-op if not in EMERGENCY_STOP state.
    """
    if self.state != ControllerState.EMERGENCY_STOP:
        logger.info("REARM ignored: not in EMERGENCY_STOP (current=%s)", self.state.value)
        return
    
    logger.info("REARM: clearing errors and resetting...")
    self.running = True
    self.error_handler.clear()
    self.tracking_quality.reset()
    if not self.dry_run:
        try:
            self.robot.arm.clear_error()  # xArm clear error
            self.robot.reset_soft_start()
        except Exception as e:
            logger.warning("REARM: error clearing robot state: %s", e)
    self.state = ControllerState.IDLE
    self._last_arm_cmd = None
    self._last_hand_cmd = None
    logger.info("REARM complete. State: IDLE")
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/control/keyboard.py:10-17` (ControlSignal + _KEY_MAP)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:482-523` (_transition 添加 REARM)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:591` (添加 _rearm 方法)

**预计工作量**: 2-3 小时

---

#### P1-3: 集中化 validate_action() 安全检查门

**来源**: ManiUniCon validate_action()

**问题描述**: dexmani 的安全检查分散在四个位置:
1. `controller.py:222-226` — 状态观察 (扭矩/电流/温度/通信)
2. `controller.py:229-252` — 关节限制 (arm→E-Stop, hand→warning)
3. `controller.py:319-322` — workspace IN_WORKSPACE flag
4. `safety.py` — 各类静态检查函数

这导致安全检查可能遗漏或重复。

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/interface.py` 中添加:

```python
def validate_action(self, action: RobotAction, state: RobotState) -> tuple[bool, dict[str, bool]]:
    """Centralized safety gate: run ALL checks before sending action to hardware.
    
    Returns:
        (all_ok: bool, details: dict[str, bool])
    
    Checks in order:
      1. arm_qpos within joint limits
      2. hand_qpos within joint limits  
      3. FK EEF position within workspace bounds
      4. FK EEF orientation within orientation bounds (if P0 implemented)
      5. Self-collision (via planner)
      6. Fingertip desk clearance
    """
    from dexmani_real.teleop.control import safety
    
    details = {}
    
    # 1. Arm joint limits
    arm_ok = safety.check_arm_joint_limits(state, self.arm.config.qpos_min, self.arm.config.qpos_max)
    details["arm_joint_limits"] = arm_ok
    if not arm_ok:
        return False, details
    
    # 2. Hand joint limits
    hand_ok = safety.check_hand_joint_limits(state, self.hand.config.qpos_min, self.hand.config.qpos_max)
    details["hand_joint_limits"] = hand_ok
    
    # 3. EEF workspace
    in_ws = self.check_workspace(state.eef_pos)
    details["in_workspace"] = in_ws
    if not in_ws:
        return False, details
    
    # 4-6: Additional checks (self-collision, desk-safety, orientation)
    # (implement as callbacks to planner/workspace)
    
    return all(details.values()), details
```

在 `controller.py:_tick()` 中调用:

```python
# 5. Safety checks on state (arm torque, hand current, hand temp, hand comm)
# ... existing per-flag checks ...

# NEW: Centralized validate_action gate
if not self.dry_run:
    all_ok, details = self.robot.validate_action(action, state)
    if not all_ok:
        logger.warning("validate_action failed: %s", details)
        hold = self.error_handler.hold_action()
        action = RobotAction(
            arm_qpos_cmd=hold.arm_qpos_cmd,
            hand_qpos_cmd=hold.hand_qpos_cmd,
        )
        # Skip send_action for this frame
        return
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/interface.py` (添加 validate_action 方法)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/core/controller.py:199-285` (_tick 替换分散检查)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/control/safety.py` (添加 check_self_collision 桩)

**预计工作量**: 4-6 小时

---

### P2 — 中: 架构增强

#### P2-1: 进程级 Camera Recording Daemon

**来源**: ManiUniCon 独立 Camera Process

**问题描述**: dexmani 的相机录制在 `add_frame(camera_frame=...)` 中同步执行 (`controller.py:259`)。RealSense USB 断开或固件 hang 会导致控制循环崩溃。

**实现指导**:

设计为可选: 当提供 CameraCalib 时, 启动独立 `mp.Process`:
- CameraProcess 循环: capture frame → SharedMemoryRingBuffer → set mp.Event
- Controller._tick() 非阻塞 poll Event → 读取最新 frame → recorder.add_frame(camera_frame=...)
- Crash 时 CameraProcess 设置错误 flag; 控制器记录 warning 但继续运行

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/sensor/realsense.py` (添加 CameraProcess 类)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/recording/episode_recorder.py:120` (异步帧消费)
- `/home/zhy/Desktop/dexmani_real/scripts/real/test_quest_hand_teleop.py` (连接进程生命周期)

**预计工作量**: 3-5 天 (基础设施构建, 建议在 P0/P1 完成后开展)

---

#### P2-2: Zarr/L3DC 导出 (Diffusion Policy 生态)

**来源**: ManiUniCon Zarr store + LeRobot v3.0

**问题描述**: dexmani 直接录制 HDF5, 但行为克隆的主导框架 (Diffusion Policy) 需要 Zarr 格式。需要导出工具打通训练流程。

**实现指导**:

新建 `/home/zhy/Desktop/dexmani_real/scripts/tools/export_hdf5_to_zarr.py` (~200 LOC):
- 接受 --data_dir (glob episode_*.h5), --output zarr 路径, --norm_stats 输出路径
- 读取 episode: arm_qpos、hand_qpos from /obs; arm_qpos、hand_qpos from /action; quality_flags
- 过滤到 ALL_GOOD_MASK
- Stack into (total_frames, obs_dim) / (total_frames, action_dim)
- 计算 obs_mean/std, action_mean/std
- 写入 Zarr: /data/obs, /data/action, /meta/norm_stats
- 使用 numcodecs.Blosc 压缩

**需修改文件**:
- NEW: `/home/zhy/Desktop/dexmani_real/scripts/tools/export_hdf5_to_zarr.py`
- NEW: `/home/zhy/Desktop/dexmani_real/dexmani_real/recording/lerobot_exporter.py` (可选后续)

**预计工作量**: 1-2 天

---

#### P2-3: VR Pose Per-Step 旋转 Delta Caps

**来源**: ManiUniCon max_delta_rot=1.0rad

**问题描述**: dexmani 的 ArmWristMapper 有 eef_delta_bounds 用于位置，但没有 per-frame 旋转 delta cap。VR 跟踪 glitch (丢失 1 帧后大角度跳变) 会产生不连续 wrist 旋转，直接送入 IK。关节 jump clamp (5°/frame) 在关节层捕获，但 Cartesian 层捕获可以更早过滤。

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/vr/arm_mapper.py` 中添加:

```python
class ArmWristMapper:
    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_base_rot: np.ndarray | None = None,
        eef_delta_bounds: np.ndarray | None = None,
        max_delta_rot_rad: float = 1.0,  # NEW: ~57°, generous, catches glitches
    ) -> None:
        # ... existing fields ...
        self.max_delta_rot_rad = max_delta_rot_rad
    
    def clip_delta_rot(self, rot_3x3: np.ndarray) -> np.ndarray:
        """Clamp rotation delta per frame to prevent VR tracking glitches."""
        axis, angle = mat2axangle(rot_3x3)
        if angle > self.max_delta_rot_rad:
            logger = __import__('dexmani_real.log', fromlist=['get_logger']).get_logger(__name__)
            logger.debug("clip_delta_rot: clamping %.3f rad -> %.3f rad", angle, self.max_delta_rot_rad)
            return axangle2mat(axis, self.max_delta_rot_rad, is_normalized=True)
        return rot_3x3
    
    def map(self, wrist_pos, wrist_quat_wxyz):
        # ... existing code ...
        delta_rot_vr = wrist_rot @ self.wrist_rot0.T
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self.clip_delta_rot(delta_rot_vr)  # NEW: add here
        delta_rot_base = self.vr_to_base_rot @ delta_rot_vr @ self.vr_to_base_rot.T
        # ... rest unchanged ...
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/vr/arm_mapper.py:16-28` (__init__ 添加 max_delta_rot_rad)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/teleop/vr/arm_mapper.py:49-75` (map 添加 clip call)

**预计工作量**: 30 分钟

---

### P3 — 低: 工程优化/便利性

#### P3-1: Config Builders (from_dict / from_yaml Factory)

**来源**: ManiUniCon Hydra instantiate()

**问题描述**: 构建 PipelineConfig 需要手动实例化 5+ dataclasses (~25 LOC per script)。

**实现指导**:

在 PipelineConfig 和所有子 dataclass 上添加 `from_dict()` 类方法，使用 `dataclasses.fields()` 递归构造。在 `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py` 中添加:

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PipelineConfig':
        """Reconstruct PipelineConfig from a dict (reverse of to_dict())."""
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name in d:
                val = d[f.name]
                if hasattr(f.type, 'from_dict'):
                    kw[f.name] = f.type.from_dict(val)
                elif isinstance(val, (list, tuple)):
                    kw[f.name] = np.array(val) if f.name.endswith('_bounds') else val
                else:
                    kw[f.name] = val
        return cls(**kw)
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py:35-59` (from_dict)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/types.py:112` (RobotInterfaceConfig.from_dict)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/types.py:73-133` (PlanningProfile/TeleopProfile.from_dict)
- NEW: `/home/zhy/Desktop/dexmani_real/configs/pipeline_default.json`

**预计工作量**: 1-2 小时

---

#### P3-2: 基于可操作性的自适应阻尼

**来源**: ManiUniCon QP 求解器近零阻尼 (1e-12)

**问题描述**: dexmani 的 DLS 使用固定阻尼 λ=0.02，在非奇异区域产生 ~1-2mm 持续跟踪偏差。`compute_manipulability()` (kinematics.py:69-74) 已实现但未在 teleop 热点使用。

**实现指导**:

在 `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/ik.py:solve_differential_ik()` 中:

```python
def solve_differential_ik(self, ...):
    # ... existing code ...
    jacobian = self.kin.compute_eef_jacobian(current_qpos)
    
    # NEW: Adaptive damping based on manipulability
    if profile.adaptive_damping:
        mu = self.kin.compute_manipulability(current_qpos)  # kin is self.kin
        threshold = profile.manipulability_threshold
        if mu > threshold:
            damping = profile.differential_ik_min_damping
        elif mu < 0.001:
            damping = profile.differential_ik_max_damping
        else:
            damping = profile.differential_ik_min_damping + \
                (profile.differential_ik_max_damping - profile.differential_ik_min_damping) * \
                (1.0 - mu / threshold)
    else:
        damping = float(profile.differential_ik_damping)
    
    lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
    # ... rest unchanged ...
```

在 TeleopProfile 中添加:

```python
@dataclass(kw_only=True)
class TeleopProfile:
    # ... existing fields ...
    adaptive_damping: bool = True
    differential_ik_min_damping: float = 0.001
    differential_ik_max_damping: float = 0.05
    manipulability_threshold: float = 0.005
```

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/ik.py:259-262` (solve_differential_ik 自适应阻尼)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/planning/types.py:107-133` (TeleopProfile 添加字段)

**预计工作量**: 1-2 小时

---

#### P3-3: PipelineConfig.to_dict() Round-Trip 完整性

**来源**: ManiUniCon Hydra 自动 YAML 往返

**问题描述**: dexmani 的 PipelineConfig.to_dict() 序列化到 HDF5 /meta 但 from_dict() 未实现, 无法还原完整配置。这破坏了录制 episode 的可复现性。

**实现指导**:

实现 from_dict() 工厂方法 (依赖 P3-1)，验证往返:
```python
d = config.to_dict()
config2 = PipelineConfig.from_dict(d)
assert d == config2.to_dict()
```

确保 `_ndarray_to_list` 有反向 `_list_to_ndarray` 函数保持往返忠实度。

**需修改文件**:
- `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py:19-31` (添加 _list_to_ndarray)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/config/pipeline_config.py:35` (from_dict → to_dict 往返)
- `/home/zhy/Desktop/dexmani_real/dexmani_real/robot/types.py:112` (all child classes from_dict)

**预计工作量**: 30 分钟 (依赖 P3-1)

---

## 5. 代码路径交叉引用

### 5.1 ManiUniCon 关键文件:行号

| 组件 | 文件路径 | 行号 | 功能描述 |
|------|----------|------|----------|
| **系统启动** | main.py | 163-170 | RobotControlSystem 实例化 |
| **控制循环** | core/robot.py | 279-466 | Robot.run() 主控制循环 |
| **状态接收线程** | core/robot.py | 174-220 | state_receiver thread 并行接收 |
| **插值器调度** | core/robot.py | 405-437 | interpolator scheduling in control loop |
| **VR 坐标系构造** | utils/quest_controller.py | 46-53 | R_ve rotation matrix construction |
| **增量计算** | utils/quest_controller.py | 192-338 | _calculate_action() delta computation |
| **安全检查** | utils/quest_controller.py | 134-158 | _check_safety_limits() |
| **安全应用** | utils/quest_controller.py | 160-190 | _apply_safety_limits() |
| **策略主循环** | policies/quest.py | 190-377 | QuestPolicy.run() |
| **IK 求解** | utils/ik_solver.py | 171-217 | IKSolver.solve() |
| **IK 任务构建** | utils/ik_solver.py | 103-124 | _create_tasks() |
| **pose 插值 - drive** | utils/pose_trajectory_interpolator.py | 78-101 | drive_to_waypoint() |
| **pose 插值 - schedule** | utils/pose_trajectory_interpolator.py | 103-185 | schedule_waypoint() |
| **pose 插值 - call** | utils/pose_trajectory_interpolator.py | 187-207 | __call__() (interpolation query) |
| **关节空间平滑** | utils/filter.py | 77-138 | JointSpaceSmoother.smooth() |
| **共享内存写** | utils/shared_memory/shared_storage.py | 382 | write_action() |
| **共享内存读** | utils/shared_memory/shared_storage.py | 404 | read_all_action() |
| **消息队列写入** | utils/shared_memory/shared_memory_queue.py | 88-107 | put() |
| **消息队列读取** | utils/shared_memory/shared_memory_queue.py | 140-149 | get_all() |
| **Cartesian IK (hot)** | robot_interface/xarm6_robotiq.py | 226-251 | Cartesian IK in hot path |
| **默认配置** | configs/default.yaml | 1-8 | frequencies and buffer configs |
| **机器人配置** | configs/robot/xarm6.yaml | 1-48 | robot parameters |
| **策略配置** | configs/policy/quest.yaml | 1-13 | policy parameters |

### 5.2 dexmani_real 关键文件:行号

| 组件 | 文件路径 | 行号 | 功能描述 |
|------|----------|------|----------|
| **控制循环 tick** | teleop/core/controller.py | 199-285 | _tick() 主步进逻辑 |
| **动作计算** | teleop/core/controller.py | 291-347 | _compute_action() |
| **手臂 IK 命令** | teleop/core/controller.py | 349-394 | _compute_arm_command() |
| **手部重定向** | teleop/core/controller.py | 396-432 | _compute_hand_command() |
| **关节跳变钳位** | teleop/core/controller.py | 434-460 | _apply_jump_clamp() |
| **状态转换** | teleop/core/controller.py | 482-521 | _transition() 状态机 |
| **紧急停止** | teleop/core/controller.py | 591-596 | _escalate_to_emergency() |
| **VR 帧读取** | teleop/core/controller.py | 615-618 | _read_vr_frame() |
| **Teleop IK solve** | planning/ik.py | 46-109 | solve() 两阶段 IK |
| **DLS 差分 IK** | planning/ik.py | 231-288 | solve_differential_ik() |
| **Position IK 回退** | planning/ik.py | 115-176 | solve_position_ik() |
| **可操作性计算** | planning/kinematics.py | 69-74 | compute_manipulability() |
| **手臂映射** | teleop/vr/arm_mapper.py | 49-75 | map() VR→EEF |
| **手部重定向** | teleop/vr/hand_retarget.py | 93-115 | retarget() |
| **Torque 检查** | teleop/control/safety.py | 21-32 | check_arm_torque() |
| **Joint limit 检查** | teleop/control/safety.py | 59-68 | check_arm_joint_limits() |
| **手部 safety 检查** | teleop/control/safety.py | 35-80 | check_hand_current/temperature/comm() |
| **10-bit QualityFlags** | recording/quality_flags.py | 1-78 | QualityFlags builder |
| **动作发送** | robot/interface.py | 330-365 | send_action() to arm+hand |
| **状态获取** | robot/interface.py | 265-328 | get_state() |
| **Workspace 检查** | planning/planner.py | 638-665 | WorkspaceSafety.check() |
| **Fingertip 桌面安全** | planning/planner.py | 668-742 | FingertipDeskSafety |
| **Episode 录制** | recording/episode_recorder.py | 37-260 | EpisodeRecorder |
| **PipelineConfig** | config/pipeline_config.py | 34-59 | PipelineConfig dataclass |
| **RobotInterfaceConfig** | robot/types.py | 111-151 | RobotInterfaceConfig |
| **TeleopProfile** | planning/types.py | 106-133 | TeleopProfile dataclass |

---

## 6. 配置参数对比

### 6.1 频率/loop 配置

| 参数 | dexmani_real | ManiUniCon | 说明 |
|------|-------------|------------|------|
| VR 帧率 | 50 Hz (HTS SDK 原生) | 30 Hz (Quest TCP) | dexmani VR 原始帧率更高 |
| 控制帧率 | 50 Hz (RateLimiter) | 200 Hz | ManiUniCon 4× 更高 |
| IK 帧率 | 50 Hz (同控制帧率) | 200 Hz (同控制帧率) | ManiUniCon 更平滑 |
| 插值帧率 | N/A (无插值) | 200 Hz | ManiUniCon 独有 |
| 策略帧率 | N/A (VR 直出) | 30 Hz | ManiUniCon 策略层 |
| Buffer size | 无显式配置 | SharedMemoryRingBuffer 可配 | |

### 6.2 安全工作空间

| 参数 | dexmani_real | ManiUniCon |
|------|-------------|------------|
| 位置 bounds | workspace_bounds: (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]] | workspace limits (position) |
| 方向 bounds | **缺失** | orientation workspace (Euler XYZ) |
| max_delta_pos | eef_delta_bounds: (3,2) or None | max_delta_pos=0.5m (per-frame) |
| max_delta_rot | **缺失** (仅 joint jump clamp) | max_delta_rot=1.0rad (per-frame) |
| 速度限制 | XArm7._limit_joint_step() (驱动层) | 0.25 m/s pos, 0.5 rad/s rot (插值层) |

### 6.3 IK 参数

| 参数 | dexmani_real | ManiUniCon |
|------|-------------|------------|
| 主算法 | DLS (差分 IK) | Pinocchio task-driven IK |
| 阻尼 | λ²=0.0004 (固定) | 1e-12 (QP 近零) |
| 回退 | Position IK (MPlib) | 无 (单策略) |
| max_pose_error_pos | 0.008m (TeleopProfile) | task 容差 |
| max_pose_error_rot | 0.08rad (TeleopProfile) | task 容差 |
| max_ik_jump | (30,...,60) deg | 未显式限制 |
| 增益 | 1.0 (全跟踪) | 1.0 |
| 自适应阻尼 | **缺失** | QP 自然处理秩亏 |

### 6.4 录制参数

| 参数 | dexmani_real | ManiUniCon |
|------|-------------|------------|
| 格式 | HDF5 (.h5) | Zarr + LeRobot v3.0 |
| 质量过滤 | 10-bit QualityFlags | 无 per-frame 标记 |
| 最大帧数 | **无限制** | max_record_steps=5000 |
| 压缩 | gzip chunked | blosc |
| 归一化统计 | 无 | obs_mean/std, action_mean/std 预计算 |
| 传感器数据 | arm_qpos/qvel/tau, eef, hand_qpos/current/tactile/temp, fingertip_pos, camera | arm_qpos, hand_qpos (基础) |
| 配置快照 | PipelineConfig.to_dict() | Hydra merged YAML |

---

## 7. 附录 — 关键文件地图

### 7.1 dexmani_real 文件地图

```
dexmani_real/
├── config/
│   └── pipeline_config.py              # 35-59  PipelineConfig dataclass
├── planning/
│   ├── ik.py                           # 46     solve() 两阶段 IK 入口
│   │                                   # 115    solve_position_ik() 回退
│   │                                   # 231    solve_differential_ik() DLS
│   ├── kinematics.py                   # 69-74  compute_manipulability()
│   ├── planner.py                      # 638-665 WorkspaceSafety
│   │                                   # 668-742 FingertipDeskSafety
│   │                                   # 153    solve_ik()
│   │                                   # 161    solve_teleop_ik()
│   └── types.py                        # 55-71  XArm7PlannerConfig
│                                       # 73-104 PlanningProfile
│                                       # 106-133 TeleopProfile
├── recording/
│   ├── quality_flags.py                # 1-78   10-bit QualityFlags
│   └── episode_recorder.py             # 37-260 EpisodeRecorder
│                                       # 120    add_frame()
│                                       # 207    stop_episode()
├── robot/
│   ├── interface.py                    # 246    is_connected()
│   │                                   # 249    check_workspace()
│   │                                   # 253    is_error()
│   │                                   # 261    emergency_stop()
│   │                                   # 265    get_state()
│   │                                   # 330    send_action()
│   │                                   # 367    reset_soft_start()
│   ├── types.py                        # 21-79  RobotState
│   │                                   # 82-108 RobotAction
│   │                                   # 111-151 RobotInterfaceConfig
│   ├── xarm7.py                        #        XArm7Config, XArm7
│   └── xhand.py                        #        XHandConfig, XHand
├── teleop/
│   ├── core/
│   │   └── controller.py               # 70-75  ControllerState Enum
│   │                                   # 88-153 TeleopController.__init__
│   │                                   # 167    run() 主循环
│   │                                   # 199-285 _tick() 步进
│   │                                   # 291-347 _compute_action()
│   │                                   # 349-394 _compute_arm_command()
│   │                                   # 396-432 _compute_hand_command()
│   │                                   # 434-460 _apply_jump_clamp()
│   │                                   # 482    _transition()
│   │                                   # 591    _escalate_to_emergency()
│   ├── vr/
│   │   ├── arm_mapper.py               # 13-104 ArmWristMapper
│   │   │                               # 49-75  map()
│   │   └── hand_retarget.py            # 18-115 XHandRetargeter
│   │                                   # 93-115 retarget()
│   └── control/
│       ├── safety.py                   # 21-32  check_arm_torque()
│       │                               # 35-42  check_hand_current()
│       │                               # 45-52  check_hand_temperature()
│       │                               # 55-56  check_hand_comm()
│       │                               # 59-68  check_arm_joint_limits()
│       │                               # 71-80  check_hand_joint_limits()
│       │                               # 83-92  check_retarget_valid()
│       └── keyboard.py                 # 10-17  ControlSignal Enum
└── sensor/
    └── realsense.py                    #        RealSense driver
```

### 7.2 ManiUniCon 文件地图 (参考)

```
maniunicon/
├── main.py                             # 163-170 RobotControlSystem
├── core/
│   └── robot.py                        # 174-220 state_receiver thread
│                                       # 279-466 run() control loop
│                                       # 405-437 interpolator scheduling
├── policies/
│   └── quest.py                        # 190-377 QuestPolicy.run()
├── utils/
│   ├── quest_controller.py             # 46-53  R_ve construction
│   │                                   # 134-158 _check_safety_limits()
│   │                                   # 160-190 _apply_safety_limits()
│   │                                   # 192-338 _calculate_action()
│   ├── ik_solver.py                    # 103-124 _create_tasks()
│   │                                   # 171-217 solve()
│   ├── pose_trajectory_interpolator.py # 78-101 drive_to_waypoint()
│   │                                   # 103-185 schedule_waypoint()
│   │                                   # 187-207 __call__()
│   ├── filter.py                       # 77-138 JointSpaceSmoother.smooth()
│   └── shared_memory/
│       ├── shared_storage.py           # 382    write_action()
│       │                               # 404    read_all_action()
│       └── shared_memory_queue.py      # 88-107 put()
│                                       # 140-149 get_all()
├── robot_interface/
│   └── xarm6_robotiq.py                # 226-251 Cartesian IK in hot path
└── configs/
    ├── default.yaml                    # 1-8   frequencies and buffer
    ├── robot/xarm6.yaml                # 1-48  robot config
    └── policy/quest.yaml               # 1-13  policy config
```

---

> **文档维护**: 本文件基于 2026-06-22 的代码状态生成。两个框架在持续演进中，建议在每个主要版本更新后重新验证对比数据。
