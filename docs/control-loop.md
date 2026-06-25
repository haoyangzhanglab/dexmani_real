# DexMani 控制回路架构

> xArm7 (7-DOF) + XHand (12-DOF) 遥操作控制系统的完整控制链路。
> 基于进程隔离 + 位置伺服 (mode 1) + 自适应 EMA 平滑。
>
> 主要参考: BunnyVisionPro (双层频率), LeFranX (EMA 思想), ManiUniCon (进程隔离)

---

## 目录

1. [架构总览](#1-架构总览)
2. [参考矩阵](#2-参考矩阵)
3. [进程边界隔离](#3-进程边界隔离)
4. [PID 进程 (250Hz 内环)](#4-pid-进程-250hz-内环)
5. [主进程 (50Hz 外环)](#5-主进程-50hz-外环)
6. [安全架构](#6-安全架构)
7. [录制系统](#7-录制系统)
8. [Return-to-Home](#8-return-to-home)
9. [端到端延迟](#9-端到端延迟)
10. [关键参数速查](#10-关键参数速查)
11. [与参考项目对比](#11-与参考项目对比)
12. [文件索引](#12-文件索引)

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MAIN PROCESS (50Hz)                                                     │
│                                                                          │
│  _tick() 清洁管线:                                                        │
│  1. 读 VR frame (ZMQ / Tracker)                                         │
│  2. VR staleness check (单一阈值 0.5s → None-sentinel 自然减速)           │
│  3. 读 arm state ← PIDStateChannel (SharedMemory)                       │
│  4. IK: 迭代 DLS (λ²=1e-5, 100 iter)                                    │
│  5. RobustEMA (α=0.95→0.3 自适应, LeFranX+ManiUniCon 融合)              │
│  6. Hand retarget → RobotInterface.send_action() (直接发送)              │
│  7. validate_action (3 项) → 写 target → PIDTargetChannel               │
│                                                                          │
│  状态机: IDLE ⇄ TELEOP ⇄ PAUSED → EMERGENCY_STOP                        │
│  PAUSED / VR丢失 → PIDTargetChannel.write(None) → 保持当前位置            │
└──────────┬───────────────────────────────────────────┬───────────────────┘
           │ PIDTargetChannel (target →, 9 floats)    │ PIDStateChannel
           │ SharedMemory, lock-free, <1μs            │ (state ←, 9 floats)
           ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PID PROCESS (独立 mp.Process, 2 线程)                                   │
│                                                                          │
│  Target Loop Thread (250Hz):                State Reader Thread (50Hz):  │
│  1. read target from PIDTargetChannel      1. XArmAPI.get_joint_states() │
│  2. 200ms timeout → hold position          2. write → PIDStateChannel    │
│  3. NaN guard → hold last valid                                          │
│  4. arm.set_servo_angle_j() (mode 1)                                     │
│                                                                          │
│  拥有独立的 XArmAPI() 连接 — Main 不直接访问 xArm SDK                    │
│  对标 ManiUniCon Robot(mp.Process) 双线程 + SDK + SharedMemory           │
│                                                                          │
│  位置伺服模式 (mode 1): arm 内部伺服负责 PID、平滑、速度限制               │
│  PIDProcess 仅做 target 转发 + 超时/NaN 安全兜底                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键时序

| 特性 | 主进程 (外环) | PID 进程 (内环) |
|------|-------------|----------------|
| 频率 | 50 Hz | 250 Hz |
| 周期 | 20 ms | 4 ms |
| 过采样比 | — | 5:1 |
| 线程 | 主线程 (`run()`) | Target Loop Thread + State Reader Thread |
| 同步方式 | `RateLimiter.wait()` 补偿式 | 简单 `time.sleep` 闭包 |
| 指令类型 | 位置目标 (`arm_qpos_cmd[7]`) | 位置目标 (`set_servo_angle_j`) |
| SDK 连接 | **无** (仅通过 SharedMemory) | **拥有** XArmAPI 连接 |
| 控制模式 | — | mode 1 (position servo) |

---

## 2. 参考矩阵

DexMani 控制回路的设计有意识地参考了三个项目，同时保留了经过验证的独家机制。

### 2.1 采纳的外部模式

| 来源 | 模式 | DexMani 实现 | 对齐度 |
|------|------|-------------|--------|
| **BVP** | 双层频率 (50Hz + 250Hz) | controller.py + pid_process.py | ✅ |
| **BVP** | 迭代 DLS IK (λ²=1e-5, 100 iter) | `planning/ik.py` | ✅ |
| **BVP** | Pinocchio FK/Jacobian | `planning/kinematics.py` | ✅ |
| **LeFranX** | EMA 关节空间平滑概念 | `robust_ema()` — α 不同 (0.95 vs 0.3) | ⚠️ |
| **LeFranX** | 差分式 VR 映射 (delta from initial) | `teleop/vr/arm_mapper.py` | ✅ |
| **LeFranX** | 命令超时保护 | PIDTargetChannel 200ms 超时 | ✅ |
| **ManiUniCon** | 进程边界隔离 (SharedMemory) | PIDTargetChannel + PIDStateChannel | ✅ |
| **ManiUniCon** | 双线程 PID 进程 (control + state reader) | `pid_process.py` | ✅ |
| **ManiUniCon** | RateLimiter 补偿式频率控制 | `utils/rate_limiter.py` | ✅ |
| **ManiUniCon** | 自适应 α 概念 (MA+EWMA) | `robust_ema()` 连续 α 自适应 | ✅ |

### 2.2 不采纳的模式及原因

| 不采纳 | 来源 | 原因 |
|--------|------|------|
| D-on-error PID | BVP | 位置伺服模式，arm 内部伺服处理 PID |
| 用户态 PID 控制 | BVP | 位置伺服模式无需用户态 PID；简化为 target 转发 |
| 无 EMA 平滑 | BVP | robust_ema 自适应 α 优于零平滑 |
| TCP 进程边界 | LeFranX | SharedMemory 延迟 <1μs (vs TCP ~1-2ms) |
| Ruckig OTG (C++ 实时) | LeFranX | xArm SDK 不支持 libfranka 式用户态回调 |
| 解析 IK | LeFranX | Franka 专用，xArm7 无通用解析解 |
| 6 级 JointSpaceSmoother | ManiUniCon | ~50-100ms 相位滞后不可接受 |
| Pink QP 基 IK | ManiUniCon | QP 开销 > 50Hz 约束；DLS 已验证稳定 |
| PoseTrajectoryInterpolator | ManiUniCon | 关节空间位置伺服已满足需求 |
| 多策略/多机器人抽象 | ManiUniCon | DexMani 专注 xArm7+XHand 深度优化 |
| 速度控制模式 (mode 4) | BVP | mode 1 位置伺服避免双连接模式冲突 |

### 2.3 独家保留机制

经四项目交叉验证，以下 2 项为 DexMani 独有且保留:

| # | 机制 | 本质 | 四个参考项目均无 |
|---|------|------|-----------------|
| **C4** | validate_action 预发送安全门 | 3 项 fail-fast 检查 (error/connection/workspace) | BVP 仅 error code; LeFranX 仅 max_relative_target; ManiUniCon 仅 joint clip |
| **C8** | None-sentinel 协议 | target=None → 保持当前位置 | 三者均无此语义，各自独立处理减速 |

---

## 3. 进程边界隔离

### 3.1 设计动机

三个参考项目都有进程边界 (BVP: ZMQ, LeFranX: TCP, ManiUniCon: SharedMemory)。
DexMani 原架构使用同进程 `threading.Lock`，无故障隔离 — 外环崩溃时内环线程在过期 target 上持续运行。

新架构对齐 ManiUniCon: PID 进程拥有 XArmAPI 连接，Main 通过 SharedMemory 通道通信。

### 3.2 双通道设计

```
Main Process (外环 50Hz)              PID Process (内环 250Hz)
┌───────────────────────┐              ┌────────────────────────────┐
│ compute_action()      │              │ Target Loop Thread (250Hz) │
│        │              │              │   read target →            │
│        ▼              │              │   NaN guard →              │
│  PIDTargetChannel     │── target ──→│   timeout check →          │
│  (9 float64, 72B)     │              │   set_servo_angle_j()      │
│                       │              │                            │
│  PIDStateChannel      │←─ state ────│ State Reader Thread (50Hz) │
│  (9 float64, 72B)     │              │   XArmAPI.get_joint_states │
│    arm_qpos(7)        │              │   write state → channel   │
│    error_flag          │              └────────────────────────────┘
└───────────────────────┘
```

通道布局 (9 × float64 = 72 bytes):
- `[0:7)` — 数据 (target_qpos 或 arm_qpos)
- `[7]` — 标志 (valid_flag 或 error_flag)
- `[8]` — 时间戳 (perf_counter)

### 3.3 故障响应矩阵 — 双向独立检测

每方不信任对端，所有故障都有独立的、不依赖对端的停转路径:

| 故障 | 谁检测 | 检测方式 | 响应 | 最坏延迟 |
|------|--------|---------|------|---------|
| **Main 崩溃** | PID 进程 | `now - target_ts > 200ms` | 保持当前位姿 | 200ms |
| **PID 崩溃** | xArm SDK | `set_servo_angle_j` 停发 | SDK 内置超时停转 | ~100ms |
| **PID 崩溃** | Main 进程 | `now - state_ts > 100ms` 或 error_flag=1 | 急停 + 告警 | ~100ms |
| **xArm 断连** | PID 进程 | `get_joint_states()` 失败 | error_flag=1, PID 退出 | 即时 |
| **SharedMemory 损坏** | 读方 | NaN/Inf 校验 | 保持位姿 (PID) / 急停 (Main) | 即时 |

**双向检测优势**: 三个参考项目仅单向检测（计算端崩被检测），DexMani 提供双向 — PIDStateChannel 让 Main 能独立发现 PID 进程存活状态。这是 SharedMemory 双向通道相对 ZMQ/TCP 单工的结构性优势。

---

## 4. PID 进程 (250Hz 内环)

### 4.1 设计决策: 位置伺服 vs 速度控制

**最终选择: mode 1 位置伺服 (`set_servo_angle_j`)**

| 方案 | 优点 | 缺点 |
|------|------|------|
| mode 4 速度控制 + 用户态 PID | 完全控制 PID 行为、可加 jerk 限幅 | 需要两个 XArmAPI 连接、mode 冲突、PID 调参复杂 |
| **mode 1 位置伺服 (采用)** | 无 mode 冲突、arm 内部伺服处理平滑/PID、代码简洁 | 无法自定义 PID 参数、无 jerk 限幅 |

**关键原因**: xArm 同一物理臂只能处于一个 mode。Main 进程的 `RobotInterface` (用于 reset/home) 和 PID 进程如果同时持有 XArmAPI 连接且使用不同 mode，会导致 `ControllerError 22`。统一使用 mode 1 消除了 mode 冲突。

### 4.2 主循环

```python
# pid_process.py: run() — 子进程主循环 @ 250Hz
while not stopped:
    rate_limiter()  # 精确 4ms 周期

    # 1. 读目标 (SharedMemory, 无锁)
    target, target_ts = target_ch.read()

    # 2. 超时检测: 200ms 无新目标 → 保持当前位姿
    if target is None or timeout:
        # 读当前关节位置，重新发送作为 hold 命令
        arm.set_servo_angle_j(angles=current_qpos, is_radian=True)
        continue

    # 3. NaN 守卫 → 发送上次有效位姿
    if not all(isfinite(target)):
        arm.set_servo_angle_j(angles=last_valid_qpos, is_radian=True)
        continue

    # 4. 发送目标位姿 (arm 内部伺服处理 PID/平滑/速度限制)
    arm.set_servo_angle_j(angles=target[:7], is_radian=True)
```

### 4.3 核心行为

| 场景 | 行为 | 说明 |
|------|------|------|
| **正常遥操作** | 转发 target → `set_servo_angle_j()` | arm 内部伺服跟踪目标 |
| **超时 (200ms)** | 读当前位姿 → 重新发送 | 不依赖 last target，读真实硬件位姿 |
| **NaN target** | 发送 `last_valid_qpos` | 防止 NaN 传播到硬件 |
| **PAUSED / VR 丢失** | Main 写 `None` → PID 读当前位姿保持 | None-sentinel 协议 |

### 4.4 State Reader 线程 (50Hz)

```python
# 独立线程，与 Target Loop 并行
while not stopped:
    code, states = arm.get_joint_states()
    if ok:
        state_ch.write(qpos, error_state=False)
    else:
        state_ch.write(zeros(7), error_state=True)
    sleep(0.02)
```

State Reader 的存在使得 Main 进程无需访问 xArm SDK 即可获取实时关节位置（用于 FK 计算、录制等）。

### 4.5 与原始设计的差异

原始计划使用 mode 4 速度控制 + D-on-Measurement PID + Jerk 限幅 + Bottleneck 裁剪，但在实际调试中发现:

1. **Mode 冲突不可解**: 两个 XArmAPI 连接同时操作同一 arm 的不同 mode → `ControllerError 22`
2. **位置伺服已足够**: arm 内部伺服 (kHz 级) 的 PID/平滑/限速优于 Python 250Hz 用户态实现
3. **简化收益大**: 删除 ~80 行 PID/jerk/bottleneck 代码，消除一整类 bug

`utils/signal_utils.py` 中的 `limit_jerk()` 函数保留但不被 PID 进程使用 — 供未来可能的离线轨迹平滑使用。

---

## 5. 主进程 (50Hz 外环)

### 5.1 状态机 (4 状态)

```
IDLE ──B(begin)──→ TELEOP ──R(自动录制)──→ (recording=True)
  ↑       │   S(stop)→IDLE       │   H(home)→IDLE
  ├──H(home)─────┘                │
  └──ESC / error: EMERGENCY_STOP
```

| 状态 | 进入条件 | 行为 |
|------|---------|------|
| **IDLE** | 启动 / HOME / STOP / QUIT | 管道空闲，不发送命令 |
| **TELEOP** | B 键 | 运行完整管线 (IK→EMA→retarget→send) |
| **PAUSED** | C 键 (toggle) | PIDTargetChannel.write(None) → 保持当前位姿 |
| **EMERGENCY_STOP** | ESC / 硬件错误 / PID 超时 | 调用 robot.emergency_stop()，仅 Q 退出或 H 恢复 |

**变更**: 原 5 状态含 SAVE_PROMPT（录制停止后确认保存/丢弃）。现改为自动保存 — STOP/QUIT 时自动调用 `collection_loop.stop_episode()`。

### 5.2 `_tick()` 流程

```
_tick() @ 50Hz
│
├─[Guard] EMERGENCY_STOP → return
├─[Guard] PAUSED → write(None) → return
├─[Guard] IDLE → return
│
├─[1] 读 VR frame (ZMQ / Tracker)
│
├─[2] VR staleness: age > 0.5s 或 frame=None → write(None) → return
│     单一阈值，对标 LeFranX 500ms 命令超时
│
├─[3] 读 arm state ← PIDStateChannel
│     error_flag=1 或 state_ts > 100ms → 急停
│
├─[4] pipeline.compute_action()
│     arm: IK (DLS 迭代) → robust_ema()
│     hand: landmarks → retarget
│
├─[5] 录制帧 (如 recording=True)
│
├─[6] validate_action() — 3 项检查
│     error/not_connected → 急停
│     workspace violation → hold last good
│     → 写 arm target → PIDTargetChannel
│     → 发 hand command → RobotInterface.send_action()
│
├─[7] 周期性状态日志 (每 2s)
│
└─[8] 循环超限检测 (>30ms → warning)
```

### 5.3 RobustEMA 平滑

```python
# utils/signal_utils.py: robust_ema()
# 自适应 α 单极点滤波器:
#   正常帧: α = 0.95 (LeFranX EMA 模型, ~2ms 滞后)
#   异常帧: α = 0.3  (ManiUniCon EWMA α, ~10ms 单帧, 无 window buffer)
#   异常判定: max|raw - prev_raw| > 0.05 rad (~2.9°)
```

| 参数 | 值 | 来源 |
|------|-----|------|
| α (正常) | 0.95 | DexMani 自优化 |
| α (异常) | 0.3 | ManiUniCon EWMA |
| 异常阈值 | 0.05 rad | 单帧跳变超过此值触发 α 自适应 |
| 状态量 | `_prev_raw_arm` (7 floats) | 帧间差分检测 |

### 5.4 IK 策略

DLS 迭代 IK (对齐 BVP):
- **算法**: Pinocchio 迭代 Newton-Raphson, 阻尼伪逆
- **参数**: λ²=1e-5, 收敛 ‖error‖<1e-3, max 100 iter
- **步长**: 每 iter ×0.05, 最终 delta 限幅 pos≤0.02m/rot≤5°
- **Fallback**: 失败 → hold prev_cmd (BVP 同)
- **执行位置**: 主进程 `planning/ik.py`

### 5.5 Hand Retarget

```
VR landmarks(21,3) → estimate_frame_from_hand_points → wrist_rot
  → landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT → MANO 骨骼
  → dex-retargeting 优化器 → 12-DOF

发送路径: controller → RobotInterface.send_action() → XHand.send_action()
          (直接发送，无 PID 内环)
Bounds check: [-0.75, 2.0] rad (内联在 controller._compute_action())
```

---

## 6. 安全架构

### 6.1 三层安全

| 层 | 位置 | 内容 |
|----|------|------|
| **L1** | controller._tick() | VR staleness (0.5s) → None-sentinel 保持位姿 |
| **L2** | validate_action() | 3 项 fail-fast (error state / connection / workspace bounds) |
| **L3** | xArm SDK | collision_sensitivity=1 (kHz 级硬件碰撞检测) |

### 6.2 validate_action() — 3 项检查

```python
# robot/validate.py (~44 行)
def validate_action(robot, action) -> tuple[bool, str]:
    # 1. SDK error state (覆盖 arm error_code + connected_flag + hand comm errors)
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Workspace bounds (FK from action command)
    eef = robot.kinematics.compute_eef_pose_world(action.arm_qpos_cmd)
    if not robot.workspace.check(eef.p):
        return False, "workspace position violation"

    return True, "ok"
```

**精简说明**: 原 7 项检查删除了 4 项冗余:
- 力矩检查 → `collision_sensitivity=1` (SDK kHz 级碰撞检测替代 50Hz Python 轮询)
- 手部电流 → XHand 固件 `tor_max=400mA` 硬件限流
- 手部温度 → 固件内置温度保护
- 手部通信 → `is_error()` 已逐行检查 commboard/jointboard/tipboard

### 6.3 xArm 碰撞检测恢复

```python
# xarm7.py XArm7Config
collision_sensitivity: int = 1  # 0→1: 恢复 SDK 内置 kHz 级碰撞检测
tcp_load_kg: float = 1.2       # XHand 实际重量
tcp_load_cog_mm: [0, 0, 80]    # XHand 重心
```

对标 BVP: `sensitivity=3` (出厂默认，最敏感) + 更低增益 (Kp=5), 无 C31 误触发。
DexMani 用 `sensitivity=1` (最不敏感) + 正确 TCP 负载。

### 6.4 None-Sentinel 协议

| 场景 | 触发 | PID 内环行为 |
|------|------|-------------|
| PAUSED | `pid_target.write(None)` | 读当前位姿 → 保持 |
| VR 丢失 (>0.5s) | `pid_target.write(None)` | 读当前位姿 → 保持 |
| PID 超时 (200ms) | PID 进程自检 | 读当前位姿 → 保持 |
| 紧急停止 | `robot.emergency_stop()` | arm.stop() + hand.stop() |

---

## 7. 录制系统

### 简化设计

| 特性 | 精简前 | 精简后 |
|------|--------|--------|
| 预录制缓冲 | ✅ (按下 R 前 N 秒) | ❌ 已删除 |
| 批次写入 | ✅ (100 帧 flush) | ❌ 逐帧写入 |
| 文件路由 | ✅ (success_dir / failure_dir) | ❌ 单一路径 |
| SAVE_PROMPT | ✅ (确认保存/丢弃) | ❌ 自动保存 |
| CollectionLoop | ~501 行 | ~181 行 |

### 录制流程

```
B 键 → start_episode() + recording=True (IDLE→TELEOP)
每帧 → record_frame(state, action, vr_frame, camera)
S/Q 键 → stop_episode(success=True) → 自动保存
H/ESC → stop_episode(success=False) → 自动保存
```

### HDF5 格式 (不变)

```
episode_000.h5
  /meta: task_label, operator, tags, duration, fps, num_frames
  /obs: arm_qpos(7), arm_qvel(7), arm_tau(7), eef_pos(3), eef_quat(4),
        hand_qpos(12), hand_tactile_sum(5,3)
  /action: arm_qpos(7), hand_qpos(12)
  /vr: wrist_pos(3), wrist_quat(4), landmarks(21,3)
  /camera/<serial>/rgb, depth, timestamps
  /timestamps, /vr_timestamps
```

---

## 8. Return-to-Home

三阶段执行:

```
Phase 1: plan_path(home EEF) — 完整碰撞检测 Cartesian 路径
         plan_path → execute_dense_waypoints (1° 分辨率)
Phase 2: 碰撞检测关节空间插值 → 接近 init_qpos
         若 joint delta < 0.5° 或路径有碰撞 → 跳过
Finalize: arm.reset() 阻塞收敛 — set_servo_angle(wait=True)
          SDK 内部闭环等待到位 → 精确 init_qpos
```

关键设计:
- Phase 1/2 使用 `send_action()` → `set_servo_angle_j()` (非阻塞)
- Finalize 使用 `arm.reset()` → `set_servo_angle(wait=True)` (阻塞)
- 非阻塞阶段有碰撞检测，阻塞阶段确保精度
- 若 planner 不可用: fallback → 直接 `arm.reset()` (SDK 阻塞 move)
- PID 进程在 return-to-home 前被停止，return-to-home 完成后重启
  (两个进程都使用 mode 1，停止 PID 是为了避免两个连接同时向 arm 发送位置指令)

---

## 9. 端到端延迟

```
VR @ 120Hz → 采集+传输 (~15-25ms)
  → 外环排队 (平均 10ms)
  → ArmMapper (<0.1ms)
  → IK DLS (~1ms, 通常 3-5 iter 收敛)
  → RobustEMA (~2ms 正常帧)
  → PIDTargetChannel SharedMemory (<1μs)
  → PID 进程 target 转发 + arm 内部伺服跟踪 (10-30ms)
  → 机电执行 (~5ms)

═══════════════════════════════
端到端总延迟 (典型): 35-65ms
═══════════════════════════════
```

| 阶段 | 历时 | 占比 |
|------|------|------|
| VR 采集+传输+排队 | ~15-25ms | ~45% |
| RobustEMA (正常帧) | ~2ms | ~3% |
| IK + 计算 | ~1ms | ~2% |
| arm 内部伺服跟踪 | 10-30ms | ~40% |
| 机电执行 | ~5ms | ~10% |

---

## 10. 关键参数速查

### 10.1 时序

| 参数 | 值 | 位置 |
|------|-----|------|
| 外环频率 | 50 Hz (20ms) | `controller.py` `target_hz=50.0` |
| 内环频率 | 250 Hz (4ms) | `pid_process.py` `dt=1/250.0` |
| 过采样比 | 5:1 | — |
| 命令超时 (PID 进程) | 200ms | `pid_process.py` `target_timeout_s=0.2` |
| State 超时 (Main 进程) | 100ms | `controller.py` |
| VR staleness 阈值 | 0.5s | `controller.py` `_VR_STALE_THRESHOLD_S` |
| 循环超限报警 | >30ms (150% × 20ms) | `controller.py` |

### 10.2 PID 进程 (位置伺服)

| 参数 | 值 | 说明 |
|------|-----|------|
| 控制模式 | mode 1 (position servo) | 与 Main 进程统一，避免 mode 冲突 |
| SDK 指令 | `set_servo_angle_j()` | arm 内部伺服处理 PID/平滑/速度 |
| 超时行为 | 读当前位姿 → 重新发送 | 200ms 无新 target 触发 |
| NaN 行为 | 发送 `last_valid_qpos` | 防止 NaN 传播 |
| PID / 平滑 | arm 固件内置 (kHz 级) | 无用户态 PID 控制器 |

### 10.3 平滑

| 参数 | 值 | 来源 |
|------|-----|------|
| 算法 | RobustEMA (自适应 α) | LeFranX EMA + ManiUniCon 自适应 |
| α (正常帧) | 0.95 | ~2ms 滞后 |
| α (异常帧) | 0.3 | ~10ms 单帧 |
| 异常阈值 | 0.05 rad (~2.9°) | 帧间跳变 |
| 手部平滑 | dex-retargeting 内置 low_pass_alpha | 不变 |

### 10.4 IK

| 参数 | 值 | 来源 |
|------|-----|------|
| 算法 | 迭代 DLS (Pinocchio) | BVP |
| 阻尼 | λ²=1e-5 | BVP |
| 收敛 | ‖error‖ < 1e-3 | BVP |
| max iter | 100 | BVP |
| Fallback | 失败 → hold prev_cmd | BVP 同 |

### 10.5 安全限值

| 参数 | 值 | 位置 |
|------|-----|------|
| 工作空间 X | [0.28, 0.72] m | `types.py` |
| 工作空间 Y | [-0.45, 0.45] m | `types.py` |
| 工作空间 Z | [0.05, 0.50] m | `types.py` |
| 碰撞检测 | sensitivity=1 | `xarm7.py` |
| TCP 负载 | 1.2kg, [0,0,80]mm | `xarm7.py` |
| 手部 retarget 范围 | [-0.75, 2.0] rad | `controller.py` |
| 桌面安全 Z (指尖) | 0.03 m (仅在 planner 离线) | `desk_safety.py` |

---

## 11. 与参考项目对比

### 11.1 架构对比

| 维度 | BVP | LeFranX | ManiUniCon | DexMani |
|------|-----|---------|------------|---------|
| **控制层数** | 双层 (50+250Hz) | 单层 + C++ 1kHz | 单层 (100Hz) | 双层 (50+250Hz) |
| **进程边界** | ZMQ (网络) | TCP (网络) | SharedMemory (进程) | SharedMemory (进程) |
| **故障检测** | 单向 | 单向 | 单向 | **双向** |
| **内环控制** | 用户态 D-on-error PID | Ruckig + 阻抗控制 | 硬件内置 | **arm 内部伺服** (mode 1) |
| **平滑** | 无 | EMA (α=0.3) | 6 级管线 (~50-100ms) | RobustEMA (α=0.95→0.3) |
| **安全门** | 1 项 (error code) | ~3 项 | ~2-3 项 | 3 项 |
| **录制** | dict → HDF5 | LeRobot 框架 | 分进程 dump | HDF5 逐帧 |
| **代码量** | ~430 行 | ~2,300 行 | ~23,500 行 | ~13,000 行 |

### 11.2 设计哲学

```
BVP:         极简主义 ── "越少代码越少 bug"
LeFranX:     工业轨迹 ── "最优轨迹 = 最优安全"
ManiUniCon:  信号优先 ── "干净信号 = 安全运动"
DexMani:     实用主义 ── "arm 内部伺服 + 进程隔离 + 双向安全"
```

### 11.3 独家优势

| 优势 | 四个参考项目均无 |
|------|-----------------|
| 双向进程级故障检测 | 三者仅单向 |
| None-sentinel 统一保持协议 | 三者各自独立处理 |
| RobustEMA 自适应平滑 | 单滤波器替代 EMA+MA 级联 |
| mode 1 统一 (无 mode 冲突) | BVP 使用 mode 4 速度控制 |

---

## 12. 文件索引

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **TeleopController** | `teleop/core/controller.py` | 621 | 外环主循环, 状态机, 录制管理 |
| **TeleopPipeline** | `teleop/core/pipeline.py` | 161 | 管线: IK → robust EMA → retarget |
| **PIDProcess** | `robot/pid_process.py` | 240 | 独立进程, 250Hz 位置伺服 target 转发 |
| **PID Channels** | `shm/pid_channels.py` | 135 | SharedMemory 双通道 |
| **XArm7** | `robot/xarm7/xarm7.py` | 326 | 精简硬件 wrapper (servo 模式 blocking moves) |
| **RobotInterface** | `robot/interface.py` | 449 | 统一接口: hand send + arm return-to-home |
| **validate_action** | `robot/validate.py` | 44 | 3 项预发送安全门 |
| **signal_utils** | `utils/signal_utils.py` | 120 | robust_ema, limit_jerk, ema_smooth |
| **CollectionLoop** | `recording/collection_loop.py` | 181 | 录制生命周期 |
| **EpisodeRecorder** | `recording/episode_recorder.py` | 279 | HDF5 逐帧写入 |
| **IK Solver** | `planning/ik.py` | — | 迭代 DLS IK |
| **Planner** | `planning/planner.py` | — | MPlib 运动规划 (solve_teleop_ik, plan_path) |
| **ArmMapper** | `teleop/vr/arm_mapper.py` | — | VR wrist → EEF pose 差分映射 |
| **HandRetargeter** | `teleop/vr/hand_retarget.py` | — | landmarks → 12-DOF |
| **Types** | `robot/types.py` | — | RobotState, RobotAction, Config |
| **Keyboard** | `teleop/control/keyboard.py` | — | 键盘处理器 |
| **RateLimiter** | `utils/rate_limiter.py` | — | 补偿式频率限制 |

### 已删除文件

| 文件 | 原因 |
|------|------|
| `teleop/control/safety.py` (186 行) | 合并到 `robot/validate.py` + 内联 bounds check |
| `recording/frame_buffer.py` (481 行) | 批次写入删除，仅保留逐帧写入路径 |
| `planning/return_home.py` | 合并到 `robot/interface.py` |

---

## 附录: 演进历史

### 删除的机制

| 机制 | 原因 |
|------|------|
| D-on-Measurement PID (用户态) | mode 1 位置伺服，arm 内部伺服处理 PID |
| 速度控制模式 (mode 4) | 双连接 mode 冲突 (ControllerError 22) |
| Jerk 限幅 (用户态) | 位置伺服模式不需要速度级 jerk 限幅 |
| Bottleneck 速度裁剪 | 位置伺服模式不需要速度缩放 |
| Soft-start 斜坡 | 位置伺服模式不需要速度启动 |
| CartPoseInterpolator | VR 50Hz = 控制频率，无频率解耦需求 |
| 跳变钳位 (arm 5°/hand 10°) | 驱动层 + retargeter low_pass_alpha 覆盖 |
| 滑动窗口趋势监控 | 仅 warning，不参与控制决策 |
| 跟踪发散检测 | PID 内环 try/except + error_state 已覆盖 |
| Target Lead Governor | 逐轴 clip 扭曲运动方向 |
| VR 3 层时效分级 (C5) | 简化为单阈值 0.5s + None-sentinel |
| PID 线程存活监控 (C6) | 进程隔离后 PIDStateChannel 停更检测替代 |
| 指尖桌面 FK hot path (C7) | 保留在 planner 离线，hot path 删除 |
| SAVE_PROMPT 状态 (C11) | 改为自动保存 |
| 预录制缓冲区 (C12) | 简化录制流程 |
| 批次 HDF5 写入 (C13) | 逐帧写入已满足需求 |
| 录制文件路由 (C14) | 单一路径简化管理 |
| IK 三层 fallback (C2/C3) | 简化为 DLS → Hold |

### 新增的机制

| 机制 | 来源 |
|------|------|
| 进程边界隔离 (PIDProcess + SharedMemory) | ManiUniCon |
| RobustEMA (自适应 α) | LeFranX + ManiUniCon |
| 命令超时 (200ms) | LeFranX |
| 双向故障检测 | 独家 |
| collision_sensitivity=1 | BVP 验证 |
| mode 1 统一 (消除 mode 冲突) | 独家 |
