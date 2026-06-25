# DexMani 控制回路架构

> xArm7 (7-DOF) + XHand (12-DOF) 遥操作控制系统的完整控制链路。
> 基于线程内环 + 速度控制 (mode 4) + 用户态 PID + 自适应 EMA 平滑。
>
> 主要参考: BunnyVisionPro (双层频率 + 线程内环 + mode 4 PID), LeFranX (EMA 思想), T-Rex (action_buffer 模式)
>
> **2025-06-25 架构演进**: PID 进程隔离 (mp.Process + SharedMemory) → ArmInnerLoop (threading.Thread + Lock)
> → 默认 mode 4 速度控制 + 用户态 PID (对标 BVP)

---

## 目录

1. [架构总览](#1-架构总览)
2. [参考矩阵](#2-参考矩阵)
3. [进程边界隔离](#3-进程边界隔离)
4. [Arm 内环线程 (250Hz, 默认 Mode 4)](#4-arm-内环线程-250hz-默认-mode-4)
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
│  MAIN PROCESS (单进程, 3+ 线程)                                           │
│                                                                          │
│  Main Thread (50Hz):                                                     │
│  1. 读 VR frame (ZMQ / Tracker)                                         │
│  2. VR staleness check (单一阈值 0.5s → inner.set_target(None))          │
│  3. 读 arm state ← inner.get_state() (Lock)                             │
│  4. IK: 迭代 DLS (λ²=1e-5, 100 iter)                                    │
│  5. RobustEMA (α=0.95→0.3 自适应, LeFranX+ManiUniCon 融合)              │
│  6. Hand retarget → RobotInterface.send_action() (直接发送)              │
│  7. validate_action + actual_arm_qpos → inner.set_target(cmd)           │
│                                                                          │
│  状态机: IDLE ⇄ TELEOP ⇄ PAUSED → EMERGENCY_STOP                        │
│  PAUSED / VR丢失 → inner.set_target(None) → 发送零速度 (mode 4)          │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ Inner Loop Thread (250Hz, daemon):                               │     │
│  │  1. with lock: target = self._arm_target                         │     │
│  │  2. 200ms timeout → send zero velocity (mode 4) / hold (mode 1)  │     │
│  │  3. NaN guard → send zero velocity / last_valid                  │     │
│  │  4. arm.get_joint_states() → update shared _arm_qpos              │     │
│  │  5. PID: error → velocity → clip → accel limit → jerk limit      │     │
│  │  6. arm.vc_set_joint_velocity(qvel) (mode 4)                     │     │
│  │                                                                  │     │
│  │  拥有独立的 XArmAPI() 连接 — 对标 BunnyVisionPro                   │     │
│  │  _internal_control_arm_qpos() 线程模式                            │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  RecordingWriter Thread (daemon):                                        │
│    queue.Queue → collection_loop.record_frame() — offloads HDF5 I/O      │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键时序

| 特性 | 主线程 (外环) | 内环线程 (ArmInnerLoop) |
|------|-------------|--------------------------|
| 频率 | 50 Hz | 250 Hz |
| 周期 | 20 ms | 4 ms |
| 过采样比 | — | 5:1 |
| 运行位置 | 主进程主线程 | 主进程 daemon 线程 |
| 同步方式 | `RateLimiter.wait()` 补偿式 | `RateLimiter.wait()` 补偿式 |
| 指令类型 | 位置目标 (`arm_qpos_cmd[7]`) | 速度指令 (`vc_set_joint_velocity`) |
| SDK 连接 | **无** (通过 inner.set_target()) | **拥有** XArmAPI 连接 |
| 控制模式 | — | mode 4 (velocity control + 用户态 PID) |
| 通信 | `threading.Lock` + numpy array | ← 同进程共享变量 |

---

## 2. 参考矩阵

DexMani 控制回路的设计有意识地参考了三个项目，同时保留了经过验证的独家机制。

### 2.1 采纳的外部模式

| 来源 | 模式 | DexMani 实现 | 对齐度 |
|------|------|-------------|--------|
| **BVP** | 双层频率 (50Hz + 250Hz) | controller.py + inner_loop.py | ✅ |
| **BVP** | 迭代 DLS IK (λ²=1e-5, 100 iter) | `planning/ik.py` | ✅ |
| **BVP** | Pinocchio FK/Jacobian | `planning/kinematics.py` | ✅ |
| **BVP** | mode 4 速度控制 + 用户态 PID | `PIDController` (inner_loop.py) | ✅ |
| **BVP** | 线程内环 (threading.Lock + 共享变量) | `ArmInnerLoop` (inner_loop.py) | ✅ |
| **LeFranX** | EMA 关节空间平滑概念 | `robust_ema()` — α 不同 (0.95 vs 0.3) | ⚠️ |
| **LeFranX** | 差分式 VR 映射 (delta from initial) | `teleop/vr/arm_mapper.py` | ✅ |
| **LeFranX** | 命令超时保护 | ArmInnerLoop 200ms 超时 | ✅ |
| **ManiUniCon** | RateLimiter 补偿式频率控制 | `utils/rate_limiter.py` | ✅ |
| **ManiUniCon** | 自适应 α 概念 (MA+EWMA) | `robust_ema()` 连续 α 自适应 | ✅ |

### 2.2 不采纳的模式及原因

| 不采纳 | 来源 | 原因 |
|--------|------|------|
| 无 EMA 平滑 | BVP | robust_ema 自适应 α 优于零平滑 |
| TCP 进程边界 | LeFranX | threading.Lock 延迟 ~100ns (vs TCP ~1-2ms) |
| Ruckig OTG (C++ 实时) | LeFranX | xArm SDK 不支持 libfranka 式用户态回调 |
| mp.Process 进程隔离 | ManiUniCon | 2025-06-25 移除；线程模型 (BVP) 更简单且足够 |
| 6 级 JointSpaceSmoother | ManiUniCon | ~50-100ms 相位滞后不可接受 |
| Pink QP 基 IK | ManiUniCon | QP 开销 > 50Hz 约束；DLS 已验证稳定 |

### 2.3 独家保留机制

经四项目交叉验证，以下 2 项为 DexMani 独有且保留:

| # | 机制 | 本质 | 四个参考项目均无 |
|---|------|------|-----------------|
| **C4** | validate_action 预发送安全门 | 3 项 fail-fast 检查 (error/connection/workspace) | BVP 仅 error code; LeFranX 仅 max_relative_target; ManiUniCon 仅 joint clip |
| **C8** | None-sentinel 协议 | target=None → 零速度停止 (mode 4) / 保持位姿 (mode 1) | 三者均无此语义，各自独立处理减速 |

---

## 3. 线程内环架构

### 3.1 设计动机

2025-06-25 架构演进: 移除 mp.Process 进程隔离，采用 BunnyVisionPro 线程模型。

原 PID 进程隔离的问题:
- 进程生命周期管理复杂 (return_to_home 时 stop/restart)
- SharedMemory 崩溃残留 (`/dev/shm/pid_*`)
- 30s 启动等待
- 调试困难 (子进程无堆栈)
- 手部架构不对称 (arm 有隔离, hand 无)

线程模型优势:
- 同进程 `threading.Lock` + numpy array 通信，延迟 ~100ns
- 无 SHM 生命周期管理
- 启动同步 <1s (vs 30s)
- 同进程异常可见，完整堆栈
- 对标 BVP/T-Rex 成熟模式

### 3.2 通信机制

```
Main Thread (50Hz)                    Inner Loop Thread (250Hz, daemon)
──────────────────                    ─────────────────────────────────
inner.set_target(cmd)  ──Lock──→     self._arm_target (np.ndarray)
qpos, err, ts = get_state() ←─Lock── self._arm_qpos, self._error_state
```

通信由 `ArmInnerLoop` 类内部管理:
- `set_target(target)`: Lock 保护下写入 `_arm_target` (numpy array) 或 None
- `get_state()`: Lock 保护下读取 `_arm_qpos` + `_error_state`
- 对标 BVP 的 `_arm_pos_target` + `_arm_lock` 模式

### 3.3 故障响应

| 故障 | 检测方式 | 响应 | 最坏延迟 |
|------|---------|------|---------|
| **Main 线程崩溃** | 内环线程 `try/finally` | arm.disconnect() + SDK 超时停转 | ~100ms |
| **内环线程异常** | `_error_state=True` + 主线程 `error_state` 检测 | 急停 | 20ms (50Hz) |
| **xArm 断连** | 内环 `get_joint_states()` 失败 | error_state=True, 线程退出 | 即时 |
| **Target 超时 (200ms)** | 内环自检 `now - target_ts` | 发送零速度 (mode 4) / 保持位姿 (mode 1) | 200ms |
| **NaN target** | 内环 `np.all(np.isfinite)` | 发送零速度 (mode 4) / last_valid_qpos (mode 1) | 4ms (250Hz) |

---

## 4. Arm 内环线程 (250Hz, 默认 Mode 4)

### 4.1 设计决策: 速度控制 (Mode 4) vs 位置伺服 (Mode 1)

**默认选择: mode 4 速度控制 + 用户态 PID (`vc_set_joint_velocity`)**

| 方案 | 优点 | 缺点 |
|------|------|------|
| **mode 4 + 用户态 PID (默认)** | 完全控制 PID 行为、可加 jerk/accel 限幅、对标 BVP | PID 调参需验证 |
| mode 1 位置伺服 (fallback) | 无 mode 冲突、arm 固件处理一切、代码简洁 | 无法自定义 PID 参数 |

**PID 增益**: kp=[10,10,10,10,10,10,10], kd=[0.04,…,0.04], ki=0 (对称化，保证各关节协调跟踪)
**kd = kp × dt 黄金比例**: 在 250Hz (dt=0.004s) 下，kd = 10 × 0.004 = 0.04 恰好处于临界不震荡点——既不引发超调，又提供最小必要阻尼。
**速度限幅**: [1.2, 1.0, 1.2, 1.0, 1.5, 1.0, 1.5] rad/s (BVP 遥操作安全值, ~33-50% 硬件能力)

### 4.2 主循环 (Mode 4 默认)

```python
# inner_loop.py: _run() — 内环线程主循环 @ 250Hz, mode 4
while not stopped:
    limiter.wait()

    # 1. 读目标 (Lock 保护)
    target = self._arm_target

    # 2. 超时 → vc_set_joint_velocity(zeros)
    # 3. NaN → vc_set_joint_velocity(zeros)
    # 4. 可选 position EMA 平滑 (smooth_position_alpha > 0)
    # 5. 读关节状态 → current_qpos

    # 6. PID 控制: error → velocity
    error = target - current_qpos
    qvel = PID.control(error, dt)

    # 7. 多级限幅: velocity → accel → jerk
    qvel = clip(qvel, max_velocity)
    qvel = limit_accel(qvel, prev_qvel, max_accel)
    qvel = limit_jerk(qvel, prev_qvel, prev_qacc, max_jerk)

    # 8. 发送速度指令
    arm.vc_set_joint_velocity(qvel)
```

### 4.3 PID 参数分析

**D-term 安全边界**: kd < kp × dt，否则 D 项在 250Hz 采样率下主导 P 项，引发"猛冲-急刹"振荡。

| 参数 | 值 | 物理意义 |
|------|-----|---------|
| kp | 10 | 比例增益: 1 rad 误差 → 10 rad/s 速度 (10× 放大) |
| kd | 0.04 | 阻尼增益: kd/dt=0.04/0.004=10→D 项等效 kp=10 (与 P 项平分) |
| kd/kp 比 | 0.004 (dt) | 黄金比例 — D 项恰好阻尼高频而不超调 |
| 等效时间常数 | ~30ms | (1/kp)*2.5 个周期 → 稳定时间 ~304ms |
| D 项噪声放大 | ~0.01 rad/s | 1e-4 rad 量化噪声 × 10 = 极低 |

**调参历程**:
1. BVP 原始: kp=[10,10,5,5,5,5,5], kd=[2,2,1,1,1,1,1] — 非对称增益导致 J1-J2 跟踪快于 J3-J7，产生 EEF 寄生位移
2. 方案B: kp=[5,5,5,5,5,5,5], kd=[1,1,1,1,1,1,1] — 对称但 kd=1.0 过大 (kd/dt=250, 50× 安全边界)，D 项振荡导致净速度仅 2%
3. 当前: kp=[10,10,10,10,10,10,10], kd=[0.04,…,0.04] — 对称 + 黄金比例，89% 斜坡跟踪率，0% 超调，~25mm EEF 滞后

### 4.4 核心行为

| 场景 | Mode 4 (默认) | Mode 1 |
|------|------|------|
| **正常遥操作** | PID(error)→速度→限幅→`vc_set_joint_velocity()` | `set_servo_angle_j()` |
| **超时 (200ms)** | 发送零速度 (停止) | 读当前位姿→保持 |
| **NaN target** | 发送零速度 (停止) | 发送 last_valid_qpos |
| **PAUSED / VR 丢失** | Main 调 `set_target(None)` → 零速度停止 | Main 调 `set_target(None)` → 保持位姿 |

### 4.5 状态读取 (统一循环)

与旧 PIDProcess 不同，ArmInnerLoop 没有独立的 State Reader 线程。
状态读取和目标转发在同一个 250Hz 循环中完成：

```
每 4ms 周期:
  1. 读 target (Lock)
  2. 超时/NaN 检查
  3. 可选 position EMA 平滑
  4. arm.get_joint_states() → 更新 self._arm_qpos (Lock)
  5. PID: error → velocity → 多级限幅
  6. arm.vc_set_joint_velocity(qvel)
```

Main 线程通过 `get_state()` 获取最新 arm 状态（250Hz 更新频率，比旧 50Hz 快 5×）。

### 4.6 与原始 PID 进程的差异

| 维度 | 旧 (PIDProcess) | 新 (ArmInnerLoop) |
|------|:---:|:---:|
| 运行方式 | mp.Process (独立进程) | threading.Thread (daemon) |
| 通信 | SharedMemory 双通道 (72B) | Lock + numpy array |
| 控制模式 | mode 1 (位置伺服) | **mode 4 (速度控制 + 用户态 PID)** |
| 状态线程 | 独立 State Reader (50Hz) | 统一循环 (250Hz) |
| PID 位置 | 无 (arm 固件 PID) | 用户态 PIDController |
| 限幅管线 | 无 | velocity → accel → jerk |
| 速率控制 | 简单 time.sleep | 补偿式 RateLimiter |
| 启动 | 30s 轮询等待 | wait_ready() <1s |
| 清理 | SHM unlink/close | 进程退出 OS 回收 |
| 调试 | 无堆栈 | 完整堆栈 |

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
| **PAUSED** | C 键 (toggle) | inner.set_target(None) → 零速度停止 (mode 4) |
| **EMERGENCY_STOP** | ESC / 硬件错误 / 内环超时 | 调用 robot.emergency_stop()，仅 Q 退出或 H 恢复 |

**变更**: 原 5 状态含 SAVE_PROMPT（录制停止后确认保存/丢弃）。现改为自动保存 — STOP/QUIT 时自动调用 `collection_loop.stop_episode()`。

### 5.2 `_tick()` 流程

```
_tick() @ 50Hz
│
├─[Guard] EMERGENCY_STOP → return
├─[Guard] PAUSED → set_target(None) → return
├─[Guard] IDLE → return
│
├─[1] 读 VR frame (ZMQ / Tracker)
│
├─[2] VR staleness: age > 0.5s 或 frame=None → set_target(None) → return
│     单一阈值，对标 LeFranX 500ms 命令超时
│
├─[3] 读 arm state ← inner.get_state()
│     error_flag=True 或 state_ts > 100ms → 急停
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
│     → inner.set_target(cmd)
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
Bounds check: 从 XHandConfig.qpos_min/qpos_max 读取逐关节限位
```

---

## 6. 安全架构

### 6.1 三层安全

| 层 | 位置 | 内容 |
|----|------|------|
| **L1** | controller._tick() | VR staleness (0.5s) → None-sentinel 零速度停止 |
| **L2** | validate_action() | 3 项 fail-fast (error state / connection / workspace bounds) |
| **L3** | xArm SDK | collision_sensitivity=1 (kHz 级硬件碰撞检测) |

### 6.2 validate_action() — 3 项检查

```python
# robot/validate.py (~55 行)
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

| 场景 | 触发 | 内环行为 (mode 4) |
|------|------|-------------|
| PAUSED | `inner.set_target(None)` | 发送零速度 → 停止 |
| VR 丢失 (>0.5s) | `inner.set_target(None)` | 发送零速度 → 停止 |
| 内环超时 (200ms) | 内环自检 | 发送零速度 → 停止 |
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

### 8.1 三级 Fallback 架构

```
Tier 1: plan_path(home EEF)
  └─ screw/RRT Cartesian 路径 + 完整碰撞检测 (self + env + desk + workspace)
     └─ 成功 → execute_waypoints (1° 分辨率插补)

Tier 2: _safe_joint_home_fallback()          ← plan_path 失败时
  └─ 稠密关节空间线性插补 (1° 分辨率), 碰撞检查
     └─ 通过 → execute_waypoints

Tier 3: arm.reset()                           ← Tier 1 + Tier 2 均失败
  └─ SDK 原生阻塞 move, 无碰撞检测 (最后手段)
```

### 8.2 执行流程

```
1. 读当前位置 qpos
2. 无 planner → Tier 2 safe joint fallback → 失败则 Tier 3
3. Snap 连续关节 (J0/J2/J4/J6) 到最近 2π 等效位
4. 已在 home (delta < 1°) → 直接返回
5. Hand reset (对齐 FK 模型)
6. Tier 1: plan_path(home_eef, qpos)
   ├─ 成功 → _execute_waypoints (dense 1° interpolation)
   └─ 失败 → Tier 2: _safe_joint_home_fallback
              ├─ 成功 → 继续
              └─ 失败 → Tier 3: arm.reset()
7. Phase 2: 关节空间插补到精确 home (_execute_joint_homing)
8. Finalize: arm.reset() — set_servo_angle(wait=True) 阻塞收敛到 init_qpos
```

### 8.3 关键设计

- **home_dt 参数**: 控制 waypoint 间隔时间。默认 0.02s (~50°/s)，keyboard_teleop 使用 0.04s (~25°/s) 更安全
- **非阻塞阶段** (Tier 1/2, Phase 2): 使用 `send_action()` → `set_servo_angle_j()` 逐点执行，有碰撞检测
- **阻塞阶段** (Finalize): 使用 `arm.reset()` → `set_servo_angle(wait=True)`，确保子度级精度
- **线程模型**: 无需停止/重启内环线程 — return_to_home 期间内环线程自动超时停止 (200ms 无 target)
- **max_waypoint_delta_deg=360.0**: keyboard_teleop 中禁用该冗余检查，因为执行层已做 1° 密集插补

---

## 9. 端到端延迟

```
VR @ 120Hz → 采集+传输 (~15-25ms)
  → 外环排队 (平均 10ms)
  → ArmMapper (<0.1ms)
  → IK DLS (~1ms, 通常 3-5 iter 收敛)
  → RobustEMA (~2ms 正常帧)
  → inner.set_target() (Lock ~100ns)
  → 内环 PID → 速度指令 (4ms 内)
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
| 内环 PID + arm 跟踪 | 10-30ms | ~40% |
| 机电执行 | ~5ms | ~10% |

---

## 10. 关键参数速查

### 10.1 时序

| 参数 | 值 | 位置 |
|------|-----|------|
| 外环频率 | 50 Hz (20ms) | `controller.py` `target_hz=50.0` |
| 内环频率 | 250 Hz (4ms) | `inner_loop.py` `dt=1/250.0` |
| 过采样比 | 5:1 | — |
| 命令超时 (内环) | 200ms | `inner_loop.py` `target_timeout_s=0.2` |
| State 超时 (Main 进程) | 100ms | `controller.py` |
| VR staleness 阈值 | 0.5s | `controller.py` `_VR_STALE_THRESHOLD_S` |
| 循环超限报警 | >30ms (150% × 20ms) | `controller.py` |

### 10.2 内环线程 (Mode 4 速度控制, 默认)

| 参数 | 值 | 说明 |
|------|-----|------|
| 控制模式 | mode 4 (velocity control) | `vc_set_joint_velocity()` + 用户态 PID |
| PID 增益 (kp) | `[10,10,10,10,10,10,10]` | 对称化 (所有关节统一，消除 EEF 寄生位移) |
| PID 增益 (kd) | `[0.04,…,0.04]` | kp × dt 黄金比例 (临界阻尼，零超调) |
| PID 增益 (ki) | `0` (默认关闭) | 可开启；抗积分饱和 (windup_limit=0.3) |
| 速度限幅 | `[1.2,1.0,1.2,1.0,1.5,1.0,1.5]` rad/s | ~33-50% 硬件能力 |
| 加速度限幅 | `None` (可选) | 可设 ~3 rad/s² |
| 加加速度限幅 | `None` (可选) | 可设 ~10 rad/s³, Ruckig 等效 |
| 位置平滑 α | `0.0` (可选) | 可设 0.3-0.8 减 IK 抖动 |
| 超时行为 | 发送零速度 | 200ms 无新 target |
| Fallback 模式 | mode 1 (position servo) | 设置 `control_mode=1` |

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
| 手部 retarget 范围 | 从 XHandConfig 逐关节读取 | `controller.py` |
| 桌面安全 Z (指尖) | 0.03 m (仅在 planner 离线) | `desk_safety.py` |
| Return-to-Home 速度 | ~25°/s (home_dt=0.04s) | `keyboard_teleop_real.py` |

---

## 11. 与参考项目对比

### 11.1 架构对比

| 维度 | BVP | LeFranX | ManiUniCon | DexMani |
|------|-----|---------|------------|---------|
| **控制层数** | 双层 (50+250Hz) | 单层 + C++ 1kHz | 单层 (100Hz) | 双层 (50+250Hz) |
| **进程边界** | ZMQ (网络) | TCP (网络) | SharedMemory (进程) | threading.Lock (同进程) |
| **故障检测** | 单向 | 单向 | 单向 | 双向 (Lock + error_state) |
| **内环运行** | 用户态 D-on-error PID (mode 4) | Ruckig + 阻抗控制 | 硬件内置 | **用户态 PID (mode 4)** 对标 BVP |
| **平滑** | 无 | EMA (α=0.3) | 6 级管线 (~50-100ms) | RobustEMA (α=0.95→0.3) |
| **安全门** | 1 项 (error code) | ~3 项 | ~2-3 项 | 3 项 |
| **录制** | dict → HDF5 | LeRobot 框架 | 分进程 dump | HDF5 逐帧 |
| **代码量** | ~430 行 | ~2,300 行 | ~23,500 行 | ~13,000 行 |

### 11.2 设计哲学

```
BVP:         极简主义 ── "越少代码越少 bug"
LeFranX:     工业轨迹 ── "最优轨迹 = 最优安全"
ManiUniCon:  信号优先 ── "干净信号 = 安全运动"
DexMani:     实用主义 ── "对标 BVP mode 4 PID + 线程内环 + 队列录制"
```

### 11.3 独家优势

| 优势 | 说明 |
|------|------|
| 线程内环 + Lock 通信 (对标 BVP) | 2025-06-25 移除进程隔离，延迟 ~100ns |
| mode 4 速度控制 + 用户态 PID (对标 BVP) | kp=10, kd=0.04 黄金比例，零超调 |
| RobustEMA 自适应平滑 | 单滤波器替代 EMA+MA 级联 |
| validate_action 预发送安全门 | 3 项 fail-fast + actual_qpos workspace check |
| 录制队列异步写入 | queue.Queue 解耦录制 I/O，不影响 50Hz 热路径 |
| Return-to-Home 三级 Fallback | plan_path → safe joint → arm.reset，逐级降级保安全 |

---

## 12. 文件索引

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **TeleopController** | `teleop/core/controller.py` | 667 | 外环主循环, 状态机, 录制管理 |
| **TeleopPipeline** | `teleop/core/pipeline.py` | 161 | 管线: IK → robust EMA → retarget |
| **ArmInnerLoop** | `robot/inner_loop.py` | 557 | 内环线程, 250Hz mode 4 PID + 多级限幅 |
| **PIDController** | `robot/inner_loop.py` | — | 用户态逐关节 PID + 抗积分饱和 |
| **RecordWriter** | `controller.py:_recording_writer()` | ~15 | 异步 HDF5 写入 daemon 线程 |
| **XArm7** | `robot/xarm7/xarm7.py` | 326 | 精简硬件 wrapper |
| **RobotInterface** | `robot/interface.py` | 513 | 统一接口: hand send + arm return-to-home (3-tier) |
| **validate_action** | `robot/validate.py` | 55 | 3 项预发送安全门 |
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
| `robot/pid_process.py` (240 行) | 被 `robot/inner_loop.py` 替代 — 线程模型无需进程隔离 |
| `shm/pid_channels.py` (135 行) | 不再需要跨进程 SharedMemory 通信 |
| `teleop/control/safety.py` (186 行) | 合并到 `robot/validate.py` + 内联 bounds check |
| `recording/frame_buffer.py` (481 行) | 批次写入删除，仅保留逐帧写入路径 |
| `planning/return_home.py` | 合并到 `robot/interface.py` |

---

## 附录: 演进历史

### 删除的机制

| 机制 | 原因 |
|------|------|
| PID 进程隔离 (mp.Process + SharedMemory) | 线程模型 (BVP) 消除进程管理、SHM 残留、启动延迟 |
| mode 1 位置伺服 (默认) | 切换为 mode 4 速度控制 + 用户态 PID (对标 BVP) |
| Bottleneck 速度裁剪 | mode 4 PID 已内置多级 velocity/accel/jerk 限幅 |
| Soft-start 斜坡 | PID 自然过渡，无需显式斜坡 |
| CartPoseInterpolator | VR 50Hz = 控制频率，无频率解耦需求 |
| 跳变钳位 (arm 5°/hand 10°) | 驱动层 + retargeter low_pass_alpha 覆盖 |
| 滑动窗口趋势监控 | 仅 warning，不参与控制决策 |
| 跟踪发散检测 | PID 内环 try/except + error_state 已覆盖 |
| Target Lead Governor | 逐轴 clip 扭曲运动方向 |
| VR 3 层时效分级 (C5) | 简化为单阈值 0.5s + None-sentinel |
| PID 线程存活监控 (C6) | 内环 error_state + is_alive() 替代 |
| 指尖桌面 FK hot path (C7) | 保留在 planner 离线，hot path 删除 |
| SAVE_PROMPT 状态 (C11) | 改为自动保存 |
| 预录制缓冲区 (C12) | 简化录制流程 |
| 批次 HDF5 写入 (C13) | 逐帧写入已满足需求 |
| 录制文件路由 (C14) | 单一路径简化管理 |
| IK 三层 fallback (C2/C3) | 简化为 DLS → Hold |

### 新增的机制

| 机制 | 来源 |
|------|------|
| 线程内环 (ArmInnerLoop + threading.Lock) | BVP |
| mode 4 速度控制 + 用户态 PID | BVP |
| PID 多级限幅管线 (velocity → accel → jerk) | BVP + LeFranX |
| kd = kp × dt 黄金比例 (零超调) | 独家 |
| PID 抗积分饱和 (windup_limit=0.3) | 独家 (BVP 无) |
| RobustEMA (自适应 α) | LeFranX + ManiUniCon |
| 命令超时 (200ms) → 零速度停止 | LeFranX |
| 双向故障检测 (error_state + is_alive) | 独家 |
| collision_sensitivity=1 | BVP 验证 |
| Return-to-Home 三级 Fallback (plan_path → safe joint → arm.reset) | 独家 |
| Hand bounds 从 XHandConfig 逐关节读取 | 独家 |
| 录制异步写入 (queue.Queue + daemon 线程) | 独家 |
| VR SharedMemory 零拷贝路径 | 独家 |
