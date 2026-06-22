# XArm7 控制回路架构文档

> 生成日期: 2026-06-22  
> 覆盖代码: `dexmani_real/robot/xarm7/`, `dexmani_real/teleop/core/controller.py`, `dexmani_real/robot/interface.py`, `dexmani_real/planning/ik.py`, `dexmani_real/teleop/control/safety.py`, `dexmani_real/teleop/vr/arm_mapper.py`

---

## 1. 总体架构（5 层，自上而下）

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: 主控制循环 (50Hz)                                       │
│  TeleopController.run() → _tick() → limiter.wait()                │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: VR 输入层                                                │
│  QuestHandTracker (本地) / VRFrameSubscriber (ZMQ 远程)           │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: 运动规划层                                               │
│  ArmWristMapper → TeleopIKSolver → XHandRetargeter               │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: RobotInterface (统一门面)                                │
│  get_state() / send_action() / safety checks                      │
├──────────────────────────────────────────────────────────────────┤
│  Layer 5: 硬件驱动层                                               │
│  ┌────────────────┐   ┌─────────────────────────┐                │
│  │ Mode 1 Servo   │   │ Mode 4 Velocity + PID    │                │
│  │ 直接位置伺服     │   │ 250Hz 内环线程           │                │
│  └────────────────┘   └─────────────────────────┘                │
│  XArmAPI (xArm SDK, Ethernet)                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer 1: 主控制循环

**文件**: `dexmani_real/teleop/core/controller.py`

### 2.1 状态机

```
IDLE ──T(键盘)──→ TELEOP ──R(键盘)──→ RECORDING
  │                 │   S→IDLE           │   H→IDLE
  H(键盘)           H(ESC)              H(ESC)
  ↓                 ↓                    ↓
  HOME          EMERGENCY_STOP      EMERGENCY_STOP
```

**状态转换触发** (`_transition`, L594-637):

| 信号 | 触发条件 | 动作 |
|------|---------|------|
| `TELEOP` | IDLE 状态 | `_reset_mapper()` + `reset_soft_start()` → TELEOP |
| `RECORD` | TELEOP 状态 | `_reset_mapper()` + `recorder.start_episode()` → RECORDING |
| `STOP` | RECORDING → TELEOP; TELEOP → IDLE | 停止录制或回到待机 |
| `HOME` | 任意 | `return_to_home()` → IDLE |
| `EMERGENCY_STOP` | 任意 | `emergency_stop()` → 退出循环 |
| `QUIT` | 任意 | `running = False` |
| `REARM` | EMERGENCY_STOP 状态 | `clear_error()` + `reset_soft_start()` → IDLE |

### 2.2 主循环体 (`run()`, L237-263)

```python
while self.running:
    self._handle_keyboard()   # 非阻塞轮询键盘队列，触发状态转换
    self._tick()               # 核心逐帧逻辑
    self.limiter.wait()        # RateLimiter @ 50Hz (config.target_hz)
```

### 2.3 逐帧流水线 (`_tick()`, L295-392)

每帧严格执行 8 个步骤，顺序不可变：

```
1. _read_vr_frame()          ── 获取 VR 数据（wrist_pos/wrist_quat/landmarks）
2. tracking_quality.check()  ── VR 帧时效性检查
   ├── age < 0.2s      → 正常执行
   ├── 0.2s < age < 1.0s → 软减速（指数拉回硬件位置）
   └── age > 1.0s      → 急停（EMERGENCY_STOP）
3. robot.get_state()         ── 读取 arm(7) + hand(12) 状态 + EEF FK
4. _compute_action()         ── VR → 目标关节角（核心计算，见第 4 节）
5. 硬件状态质量标志           ── 力矩/电流/温度/通信 写入 QualityFlags
6. recorder.add_frame()      ── 录制（仅 RECORDING 状态）
7. Pre-send safety gate      ── 安全闸门校验
   ├── robot.is_error()     → 急停
   └── 力矩/电流/温度超标    → hold 上一帧命令
8. robot.send_action()       ── 发送关节命令到硬件
9. 周期性状态日志（每 2s）
```

### 2.4 VR 过期处理

**文件**: `dexmani_real/teleop/core/tracking.py`

| 状态 | 帧龄阈值 | 连续丢失阈值 | 行为 |
|------|---------|------------|------|
| 正常 | `< 0.2s` (`max_frame_age_s`) | — | 执行 IK + retarget |
| 过时 (stale) | `> 0.2s` | `< 1.0s` (`ema_loss_timeout_s`) | 指数软减速 |
| 丢失 (lost) | — | `> 1.0s` | EMERGENCY_STOP |

**软减速算法** (`_apply_soft_deceleration`, L267-291):

$$
\text{arm\_cmd} = e^{-3t} \cdot \text{last\_cmd} + (1 - e^{-3t}) \cdot \text{hw\_qpos}
$$

时间常数 τ = 1/3 ≈ 0.333s。每次调用都发送命令到硬件，确保平滑减速。

---

## 3. Layer 2: VR 输入层

**文件**: controller.py L766-771, `dexmani_real/teleop/vr/`

### 数据源

| 数据源 | 类 | 通信方式 |
|--------|---|---------|
| 本地 | `QuestHandTracker` | Quest 头显直连 |
| 远程 | `VRFrameSubscriber` | ZMQ SUB socket |

### VR 帧数据结构

```python
{
    "wrist_pos": np.ndarray,       # (3,)  float64  手腕位置（VR 坐标系）
    "wrist_quat_wxyz": np.ndarray, # (4,)  float64  手腕姿态四元数 (w,x,y,z)
    "landmarks": np.ndarray,       # (21,3) float64 手部 21 个关键点
    "sequence_id": int,            # 帧序号
    "timestamp": float,            # 时间戳
    "local_recv_ns": int,          # 本地接收时间（纳秒），用于帧龄计算
}
```

---

## 4. Layer 3: 运动规划层

**入口**: `_compute_action(vr_frame, state)` → `(RobotAction, QualityFlags)`

**文件**: controller.py L395-453

### 4.1 整体计算流程

```
_compute_action(vr_frame, state)
│
├── error_handler.init_fallback(current_qpos)   ── 初始化 hold 回退位置
│
├── _compute_arm_command()                       ── 手臂 IK
│   ├── ArmWristMapper.map()                    ── VR→世界 EEF 目标位姿
│   ├── [可选] CartPoseInterpolator             ── 位姿插值
│   ├── TeleopIKSolver.solve()                  ── IK 求解（见 4.4）
│   └── ema_smooth()                            ── EMA 平滑
│
├── workspace check                              ── EEF 位姿工作空间检查
│   ├── check_workspace(eef_pos)                ── 位置边界
│   └── check_workspace_orientation(eef_quat)   ── 姿态边界
│   └── 不通过 → hold 上一帧
│
├── _compute_hand_command()  (仅 workspace 通过时执行)
│   ├── estimate_frame_from_hand_points()       ── 估计手部参考系
│   ├── landmarks → mano 坐标变换
│   └── retargeter.retarget()                   ── 手指重定向 (21→12)
│
└── _apply_jump_clamp()                          ── IK 异常跳跃防御
    ├── arm: 5°/frame (≈250°/s) 裁剪
    └── hand: 10°/frame 裁剪
```

### 4.2 ArmWristMapper — VR 到 EEF 位姿映射

**文件**: `dexmani_real/teleop/vr/arm_mapper.py`

增量映射公式：

$$
T_{\text{target}} = \Delta T_{\text{VR}} \cdot T_{\text{eef0}}
$$

其中 $\Delta T_{\text{VR}}$ 是当前 VR 手腕位姿相对于锚点位姿的增量。

**处理步骤**:
1. 位置增量: `delta_pos = pos_scale * (vr_to_base_rot @ (wrist_pos - wrist_pos0))`
2. 位置裁剪: `clip_delta_pos()` — 可配置的 eef_delta_bounds
3. 旋转增量: `delta_rot = wrist_rot @ wrist_rot0^T`
4. 旋转缩放: `scale_rot()` — `rot_scale` 参数缩放旋转角
5. 旋转裁剪: `_clip_delta_rot()` — 单帧旋转上限 `max_delta_rot_rad` (默认 ~57°)
6. 四元数连续化: `continuous_quat()` — 检测四元数翻转（点积 < 0 → 取反），确保 IK 种子一致性

### 4.3 CartPoseInterpolator（可选）

**文件**: `dexmani_real/teleop/vr/pose_interpolator.py`

在 VR 帧之间进行线性位置插值 + SLERP 旋转插值。消除 VR 帧率抖动导致的步进运动。通过 `TeleopProfile.use_cartesian_interpolation` 控制开关。

### 4.4 TeleopIKSolver — IK 求解器

**文件**: `dexmani_real/planning/ik.py`

**三级求解策略** (`solve()`, L53-115):

```
Priority 1: 微分 IK (Damped Least Squares)
    │
    ├── 成功 + 位姿误差 < 阈值  → 返回结果
    ├── 收敛但位姿误差过大        → 降级到 Priority 2
    └── 失败（奇异/碰撞）         → 降级到 Priority 2
    │
Priority 2: 位置 IK (MPlib, 随机种子)
    │
    ├── prev_cmd 种子 (n_init=3) → 硬件最近候选
    ├── current_qpos 种子 (n_init=2) → 硬件最近候选
    └── 失败 → 降级到 Priority 3
    │
Priority 3: Hold
    └── 返回 previous_qpos_cmd（保持原位）
```

#### 微分 IK 细节 (`solve_differential_ik()`, L353-432)

- **线性化点**: `current_qpos`（硬件实际位置），不是 `previous_qpos_cmd`
- **算法**: Damped Least Squares (DLS): $dq = J^T (JJ^T + \lambda^2 I)^{-1} e$
- **误差**: 世界坐标系误差经 Jacobian 旋转到 EEF 局部坐标系
- **阻尼**: 支持自适应阻尼（基于 Yoshikawa manipulability）或固定阻尼
- **奇异回避**: manipulability 过低时自动增加阻尼重试（`singularity_damping_scale`）

#### 位置 IK 细节 (`solve_position_ik()`, L119-189)

- **候选选择**: 硬件最近原则（LeFranX current_distance penalty）
- **快速接受**: `prev_cmd` 种子结果距硬件 ≤ `position_ik_fast_accept_rad` → 立即接受
- **提前退出**: 高质量解（误差<30%阈值 且 硬件距离<50%阈值）→ 立即返回
- **过滤条件**: 位姿误差 + 跳跃幅度 + 连续关节等价规范化

#### IK 失败诊断 (`_build_ik_diagnostic()`, L194-280)

失败分类映射到可行动的根因:
- `singular` → 接近奇异点，阻尼耗尽
- `pose_error` → 收敛但 FK 残差过大（目标可能在工作空间外）
- `self_collision` → 解导致自碰撞
- `unreachable` → MPlib 无解（目标真正不可达）
- `filtered` → 所有候选被过滤（极限/delta/位姿误差/碰撞）

### 4.5 EMA 平滑

**文件**: `dexmani_real/utils/signal_utils.py`

公式: `output = α * new_val + (1-α) * prev_val`

- **Arm**: 默认 α=1.0（不平滑）。α < 1 时平滑 IK 输出
- **Hand**: 不使用 EMA。`dex-retargeting` 外部库内置 `low_pass_alpha` 处理平滑

### 4.6 Jump Clamp

**文件**: controller.py L552-583

| 对象 | 阈值 | 值 (°/帧) | 等效速度 (°/s @50Hz) | 性质 |
|------|------|----------|---------------------|------|
| Arm | `_ARM_JUMP_LIMIT_RAD` | 5° | 250°/s | IK 异常防御 |
| Hand | `_HAND_JUMP_LIMIT_RAD` | 10° | 500°/s | IK 异常防御 |

> **注意**: 这些阈值远高于正常限速（arm max_qvel ≈ 90-150°/s），仅在 IK 产生不连续解（如穿越奇异点导致的跳跃）时触发。

---

## 5. Layer 4: RobotInterface 统一接口

**文件**: `dexmani_real/robot/interface.py`

### 5.1 核心 API

| 方法 | 功能 |
|------|------|
| `connect()` | arm.connect() + hand.connect() |
| `get_state()` | 读取 arm(7) + hand(12) 状态 + EEF FK + 指尖 FK |
| `send_action(action)` | arm.send_action() + hand.send_action()，返回 `{"arm_ok", "hand_ok", "arm_cmd", "hand_cmd"}` |
| `is_error()` | `arm.is_error() or hand.is_error()` |
| `emergency_stop()` | `arm.stop()` + `hand.stop()` |
| `check_workspace(pos)` | EEF 位置边界检查 |
| `check_workspace_orientation(quat)` | EEF 姿态 Euler 边界检查 |
| `return_to_home()` | 两阶段回零（见 5.2） |
| `reset_soft_start()` | 重置 arm 软启动计数器 |

### 5.2 return_to_home() 两阶段回零

**文件**: interface.py L272-372

```
return_to_home()
│
├── _snap_to_nearest_equivalent()     ── 连续关节等价对齐
├── _at_home() 检查                   ── 已在零位 → 直接返回
│
├── Phase 0: _reset_hand_before_planning()
│   └── hand.reset() + 主动轮询收敛等待（3s 超时, 5°阈值）
│
├── Phase 1: _execute_phase1_eef_cartesian()
│   ├── planner.plan_path(home_eef)   ── screw → RRT 多策略规划
│   ├── _dense_interpolate()          ── 1°/step 稠密插值
│   ├── 指尖桌面安全检查               ── FK 验证
│   ├── 逐 waypoint send_action()     ── 支持 cancel_event 中断
│   └── _wait_for_arm_convergence()   ── 闭环收敛等待（3°阈值）
│
├── Phase 2: _execute_phase2_joint_space()
│   ├── _safe_joint_path()            ── 线性插值 + 碰撞验证
│   └── 逐 waypoint send_action()
│
└── Fallback: _return_to_home_direct()
    ├── _lift_eef_z_safe()            ── IK 解 Z+ 提升
    └── arm.reset()                   ── SDK 内置关节空间回零
```

---

## 6. Layer 5: XArm7 硬件驱动

**文件**: `dexmani_real/robot/xarm7/xarm7.py`

### 6.1 两种控制模式

| 特性 | Mode 1: Servo (默认) | Mode 4: Velocity + PID |
|------|---------------------|------------------------|
| 配置 | `use_servo_control=True` | `use_servo_control=False` |
| xArm SDK 模式 | Mode 1 (位置伺服) | Mode 4 (速度控制) |
| SDK API | `set_servo_angle_j()` | `vc_set_joint_velocity()` |
| 调用频率 | 50Hz（跟随 send_action） | 250Hz（独立 daemon 线程） |
| 平滑方式 | 步进式，可能有微跳 | 连续速度输出，更平滑 |
| 限速方式 | `_limit_joint_step()` 瓶颈缩放 | PID → `_clip_arm_velocity()` 瓶颈缩放 |

### 6.2 send_action 流程 (L355-431)

```
send_action(action)
│
├── SDK 预检查 (L376-384)
│   ├── arm.connected?   → 标记 error_state
│   └── arm.error_code?  → 记录 SDK 错误码 + 标记 error_state
│
├── _limit_joint_range(target_qpos)    ── 关节极限裁剪 (np.clip)
│
├── [Mode 1: Servo]
│   ├── _limit_joint_step(target_qpos) ── 瓶颈缩放限速（见 6.3）
│   ├── _set_mode(1)                   ── 模式切换 (0→1→验证)
│   └── arm.set_servo_angle_j()        ── 发送位置命令
│
└── [Mode 4: Velocity + PID]
    ├── _set_mode(4)                   ── 模式切换 (0→4→验证)
    └── 写入 _arm_pos_target           ── Lock 保护，由 250Hz 内环消费
```

### 6.3 瓶颈缩放限速 (`_limit_joint_step`, L778-861)

**策略**: 比例瓶颈缩放（bottleneck scaling），保持关节空间轨迹形状。

```
1. 读取硬件实际位置 hw_qpos（而非上次命令 — 防止跟踪滞后累积）
2. dt = clamp(now - last_cmd_time, config.dt, config.dt * 10)
   ├── dt floor: 防止指令过快导致除以零
   └── dt ceiling (10×): 防止长时间停顿后的巨大跳跃（200ms 封顶 @50Hz）
3. max_step = max_qvel * dt（带软启动渐变，见 6.5）
4. delta = target_qpos - ref（ref = hw_qpos 或 last_qpos_cmd）
5. 瓶颈因子 = max(|delta_i| / max_step_i)
6. 如果瓶颈因子 > 1: delta /= 瓶颈因子（所有关节等比缩放）
```

**与 XHand 的关键差异**:

| 特性 | XArm7 | XHand |
|------|-------|-------|
| 限速策略 | 瓶颈缩放（保持轨迹形状） | 独立裁剪（`np.clip(raw_step, -max_step, max_step)`） |
| 原因 | 手臂 7-DOF 运动学耦合，形状保持重要 | 手指关节独立运动，耦合无意义 |
| 位置参考 | 每次 `send_action` 调用 `_read_qpos()` | 优先用背景线程缓存，fallback 用 last_qpos_cmd |

### 6.4 Mode 4: PID 内环 (`_internal_control_arm_qpos`, L557-643)

**250Hz 独立 daemon 线程**，将位置误差转为连续速度信号。

```
while not _arm_should_stop:
    rate_limiter.wait()                    ── 250Hz 精确节拍

    # 读取目标（线程安全）
    with _arm_lock:
        target = _arm_pos_target          ── 由 send_action 50Hz 写入

    # 读取硬件位置
    arm.get_joint_states()                 ── 读 xArm SDK

    # PID 控制
    error = target - current_qpos
    qvel = _arm_pid.control(error, dt)    ── PID 输出

    # 速度裁剪
    safe_qvel = _clip_arm_velocity(qvel)   ── 瓶颈缩放 + 软启动

    # 发送硬件命令
    arm.vc_set_joint_velocity(safe_qvel)   ── 写入 xArm
```

#### PID 控制器 (`PIDController`, L40-102)

$$
\text{vel} = K_p \cdot \text{err} + K_d \cdot \frac{\text{err} - \text{err\_prev}}{dt} + K_i \cdot \text{cum\_err}
$$

- **Ki = 0 默认**：遥操作目标持续变化，积分项会导致饱和（windup）
- **Kd = Kp / 20 默认**

#### PID 参数表

| 关节 | Kp | Kd | pid_max_vel (rad/s) | max_qvel (rad/s) | max_qvel (°/s) |
|------|----|----|------|------|------|
| J1 (基座旋转) | 10.0 | 0.5 | 1.2 | ~1.57 | ~90 |
| J2 (基座俯仰) | 10.0 | 0.5 | 1.2 | ~2.09 | ~120 |
| J3 (肩) | 5.0 | 0.25 | 1.2 | ~1.57 | ~90 |
| J4 (肘) | 5.0 | 0.25 | 1.2 | ~1.57 | ~90 |
| J5 (腕俯仰) | 5.0 | 0.25 | 1.6 | ~2.09 | ~120 |
| J6 (腕旋转) | 5.0 | 0.25 | 1.6 | ~2.62 | ~150 |
| J7 (手旋转) | 5.0 | 0.25 | 2.0 | ~2.62 | ~150 |

> pid_max_vel ≈ 80% 的 max_qvel，为 PID 超调保留裕量。

### 6.5 软启动机制

两种模式使用不同的软启动策略：

| | Servo Mode (位置) | Velocity Mode (速度) |
|---|---|---|
| 实现方式 | 帧计数 | 计时器 |
| 变量 | `_cmd_count` / `soft_start_frames` (20帧) | `_vel_ramp_start` / `pid_soft_start_duration_s` (0.4s) |
| 效果 | 速度上限从 0.3 rad/s 线性渐变到 max_qvel | 速度上限从 30% 线性渐变到 100% pid_max_vel |
| 收敛检测 | 无（纯帧计数） | 有（`pid_convergence_threshold_rad` = 2°，B3 特性） |
| 收敛逻辑 | — | 所有关节误差 < 2° 后触发渐变；阈值 ≤ 0 时退化为纯时间渐变 |

**触发软启动的场景**:
- `connect()` — 连接时
- `reset_soft_start()` — TELEOP 进入时（controller 调用）
- `_set_mode()` — 模式切换时
- `clear_error()` — 错误恢复时

### 6.6 碰撞检测配置

**文件**: xarm7.py L170-181

| 参数 | 值 | 说明 |
|------|-----|------|
| `tcp_load_kg` | 1.2 | XHand 重量，用于动力学力矩估计 |
| `tcp_load_cog_mm` | [0, 0, 80] | 负载重心相对法兰坐标系 |
| `collision_sensitivity` | 0 (disabled) | 防止遥操作快速运动触发误报 C31 |

> C31 误报原理: xArm 通过动力学模型估计理论关节力矩，与实际力矩比较。不配置 TCP 负载时模型假定 0kg，驱动 XHand (~1.2kg) 所需的力矩被误判为碰撞。

### 6.7 初始化序列 (`robot_init()`, L691-727)

```
1. clean_error() + clean_warn()
2. motion_enable(True)
3. _set_mode()        ── 0→目标模式→双 set_state(0) 确认
4. set_collision_sensitivity(0)
5. set_tcp_load(1.2kg, CoG)
6. get_err_warn_code() 最终验证
7. reset_soft_start()
```

**模式切换** (`_set_mode()`, L656-689): 通过 Mode 0 (idle) 中间状态过渡，防止 xArm 状态机进入未定义状态。序列: `mode0 → target_mode → double set_state(0) → 错误验证`。

---

## 7. 安全层

**文件**: `dexmani_real/teleop/control/safety.py`

### 7.1 逐帧安全标志

| 标志 | 检查内容 | 阈值 | 来源 |
|------|---------|------|------|
| `ARM_TORQUE_OK` | 关节力矩 (7,) | J1-2: 50, J3-5: 30, J6-7: 20 Nm | URDF 硬件属性 (`robot/types.py`) |
| `HAND_CURRENT_OK` | 手指电流 (12,) | 500 mA | XHand 保守默认值 |
| `HAND_TEMP_OK` | 手指温度 (12,) | 70°C | XHand 保守默认值 |
| `HAND_COMM_OK` | 通信板错误码 | — | `xhand.get_state()` |
| `RETARGET_VALID` | 手指关节角 (12,) | [-0.75, 2.0] rad | XHand 硬件极限 + 0.05 margin |
| `JOINT_JUMP_OK` | arm:5°, hand:10°/帧 | — | controller 层 IK 异常防御 |

### 7.2 Pre-send Safety Gate

**文件**: controller.py L358-378

在 `send_action` 前执行最终安全检查：

1. `robot.is_error()` → EMERGENCY_STOP（急停）
2. `ARM_TORQUE_OK` 失败 → hold 上一帧命令
3. `HAND_CURRENT_OK` 失败 → hold 上一帧命令
4. `HAND_TEMP_OK` 失败 → hold 上一帧命令

> 注意: `HAND_COMM_OK` **不**触发 pre-send hold。通信板错误可能导致关节位置读数不准，但不应阻止正常遥操作指令发送。

### 7.3 Hold 机制 (`TeleopErrorHandler`)

**文件**: `dexmani_real/teleop/core/error_handler.py`

- `init_fallback()`: 每帧开始时用当前状态初始化回退位置（仅首次）
- `hold_action()`: 返回上次成功的 arm + hand 命令
- `update_good_positions()`: IK + retarget 都成功时更新

---

## 8. 完整数据流

```
Quest VR (≥72Hz)
     │ wrist_pos(3), wrist_quat_wxyz(4), landmarks(21x3)
     ▼
┌──────────────────────────────────────────────────────────────┐
│ TeleopController.run() [50Hz]                                 │
│                                                               │
│ 1. VR → _read_vr_frame()                                      │
│ 2. tracking_quality.check() ── age<0.2s? stale? lost?         │
│ 3. robot.get_state() ── arm_qpos(7), hand_qpos(12), EEF FK   │
│                                                               │
│ 4. _compute_action():                                         │
│    ┌─────────────────────────────────────────────────────┐   │
│    │ ArmWristMapper.map()                                 │   │
│    │   └─ VR增量→世界EEF目标位姿 (pos(3)+quat(4))        │   │
│    │ [CartPoseInterpolator.get_interpolated_pose()]       │   │
│    │   └─ 线性位置+SLERP旋转插值                          │   │
│    │ TeleopIKSolver.solve(target_pose, hw_qpos, prev_cmd) │   │
│    │   ├─ 1. Diff IK: DLS @ hw_qpos 线性化点            │   │
│    │   ├─ 2. Pos IK: prev_cmd→hw_qpos 种子, 硬件最近     │   │
│    │   └─ 3. Hold: 返回 prev_cmd                         │   │
│    │ ema_smooth(raw_arm, last_arm_cmd, α)                 │   │
│    │   └─ α*new + (1-α)*prev (α=1 默认不平滑)           │   │
│    │ workspace check + orientation check                   │   │
│    │ XHandRetargeter.retarget(landmarks)                   │   │
│    │   └─ 手部关键点→12-DOF手指关节角                     │   │
│    │ [dex-retargeting 内置 low_pass_alpha 平滑]           │   │
│    │ _apply_jump_clamp()                                   │   │
│    │   └─ arm:5°, hand:10°/帧 裁剪                        │   │
│    └─────────────────────────────────────────────────────┘   │
│                                                               │
│ 5. safety checks → QualityFlags                              │
│ 6. recorder.add_frame() [RECORDING 状态]                     │
│ 7. Pre-send gate: torque/current/temp → hold or send         │
│ 8. robot.send_action(action)                                 │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ RobotInterface.send_action()                                  │
│                                                               │
│ arm.send_action(arm_qpos_cmd(7))                              │
│   ├─ _limit_joint_range() → np.clip(qpos_min, qpos_max)      │
│   ├─ [Servo] _limit_joint_step() → bottleneck scale → SDK    │
│   │   └─ arm.set_servo_angle_j() [50Hz]                      │
│   └─ [Velocity] _arm_pos_target ← target → 250Hz PID thread  │
│       └─ PID.control(err,dt) → _clip_arm_velocity(bottleneck)│
│           → arm.vc_set_joint_velocity() [250Hz]              │
│                                                               │
│ hand.send_action(hand_qpos_cmd(12))                           │
│   ├─ _limit_joint_range() → np.clip(qpos_min, qpos_max)      │
│   ├─ _limit_joint_step() → per-joint independent clip         │
│   ├─ [XHand] EMA (config.ema_alpha)                           │
│   └─ write_command_positions() → send_command() [RS485]      │
└──────────────────────────────────────────────────────────────┘
     │
     ▼
┌──────────────┐    ┌──────────────────────┐
│ xArm SDK     │    │ xhand_controller SDK  │
│ (Ethernet)   │    │ (RS485)               │
└──────┬───────┘    └──────────┬───────────┘
       │                       │
       ▼                       ▼
   伺服驱动器               直流无刷电机
   (7-DOF Arm)           (12-DOF Hand)
```

---

## 9. 关键设计决策

### 9.1 职责分离

| 层 | 职责 | 不负责 |
|----|------|--------|
| Controller | VR→世界坐标映射、IK 求解、安全闸门、状态机 | 关节限速、硬件限位 |
| Driver | 关节限速、关节限位、硬件通信、PID 内环 | 工作空间检查、VR 处理 |
| Safety | 力矩/电流/温度软故障检测 | 硬件急停（driver 层） |

### 9.2 瓶颈缩放 vs 独立裁剪

- **Arm (瓶颈缩放)**: 任一关节超速 → **所有关节等比缩放** — 保持笛卡尔轨迹形状。因为手臂 7-DOF 耦合，缩放任一关节会扭曲末端运动路径
- **Hand (独立裁剪)**: 每根手指独立 `np.clip` — 手指运动不需要轨迹一致性

### 9.3 硬件位置作为 delta 参考

`_limit_joint_step` 以**硬件实际位置**（非上次命令）为 delta 基准。这防止了跟踪滞后导致的命令叠加发散：

```
错误方式:  delta = target - last_cmd         → 硬件滞后时 delta 被低估
正确方式:  delta = target - hw_qpos          → 以真实物理位置为基准
```

### 9.4 Mode 4 的优势

Mode 4 (Velocity + PID) 相比 Mode 1 (Servo) 产生更平滑的运动：
- Servo 以 50Hz 发送离散位置命令，逐步跳跃
- Velocity 以 250Hz 连续输出速度信号，PID 平滑过渡
- Ki=0 避免遥操作中目标持续变化导致的积分饱和

### 9.5 四元数连续化

`ArmWristMapper.continuous_quat()` 检测四元数符号翻转（点积 < 0 → 取反）。这是因为 $q$ 和 $-q$ 表示相同旋转，但 IK 求解器对不同符号敏感，符号跳变会导致 IK 发散。

### 9.6 DT 天花板

`_limit_joint_step` 中 `dt = min(max(dt_raw, dt), dt * 10)`:
- **floor** (dt): 防止指令过快导致除以零/无穷速度
- **ceiling** (10×dt=0.2s @50Hz): 防止 GC/系统暂停后的大跳跃（500ms 停顿 → 最多允许 10 帧运动量）

### 9.7 碰撞检测

`collision_sensitivity=0` (disabled) 是经过验证的选择：即使正确配置了 TCP 负载，遥操作的快速运动仍可能触发 C31 误报。碰撞安全由 geometric FK 桌面安全检查和 workspace bounds 保证。

---

## 10. 关键文件索引

| 文件 | 内容 |
|------|------|
| `dexmani_real/teleop/core/controller.py` | 主控制循环、状态机、安全闸门 |
| `dexmani_real/teleop/core/tracking.py` | VR 帧时效性检查 |
| `dexmani_real/teleop/core/error_handler.py` | Hold-on-failure 回退机制 |
| `dexmani_real/teleop/vr/arm_mapper.py` | VR→EEF 位姿增量映射 |
| `dexmani_real/teleop/vr/hand_retarget.py` | 手指重定向 (21→12 DOF) |
| `dexmani_real/teleop/vr/pose_interpolator.py` | 笛卡尔位姿插值（可选） |
| `dexmani_real/teleop/control/safety.py` | 力矩/电流/温度安全检查 |
| `dexmani_real/planning/ik.py` | IK 求解器（Diff IK + Position IK） |
| `dexmani_real/planning/planner.py` | 运动规划（plan_path + 验证） |
| `dexmani_real/robot/interface.py` | RobotInterface 统一门面 |
| `dexmani_real/robot/xarm7/xarm7.py` | XArm7 驱动（Servo + Velocity+PID） |
| `dexmani_real/robot/xhand/xhand.py` | XHand 驱动（独立裁剪） |
| `dexmani_real/robot/types.py` | RobotState, RobotAction, 力矩极限 |
| `dexmani_real/utils/signal_utils.py` | EMA 平滑工具 |
