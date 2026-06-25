# DexMani XHand 控制回路

> XHand (12-DOF) 灵巧手遥操作控制系统的完整链路。
> 基于 RS485 直驱 + 线程内发送 + dex-retargeting + LeFranX 自适应小指缩放。
>
> 主要参考: LeFranX (配置参数、EMA 概念、小指自适应), DexUMI (MotorTrajectoryInterpolator、触觉检测), skill-teleop (关节限位参考)
>
> **2025-06-25 驱动简化**: 移除 E1 后台状态读取线程、移除远端关节 5° 最小限位、移除 trajectory 自动速度延长，
> 替换为 DexUMI MotorTrajectoryInterpolator (scipy 线性插值)、对齐 LeFranX 的 kp/kd/dt 配置。

---

## 目录

1. [架构总览](#1-架构总览)
2. [参考矩阵](#2-参考矩阵)
3. [XHand 驱动层](#3-xhand-驱动层)
4. [轨迹插值器](#4-轨迹插值器)
5. [VR 手部重定向](#5-vr-手部重定向)
6. [安全架构](#6-安全架构)
7. [错误处理与恢复](#7-错误处理与恢复)
8. [触觉传感](#8-触觉传感)
9. [与参考项目对比](#9-与参考项目对比)
10. [关键参数速查](#10-关键参数速查)
11. [文件索引](#11-文件索引)

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  MAIN PROCESS — Hand Control Path (50Hz 外环)                            │
│                                                                          │
│  VR Tracker (Quest TCP/UDP)                                              │
│    │ landmarks (21,3) + wrist_rot (3,3)                                  │
│    ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ XHandRetargeter (hand_retarget.py)                               │     │
│  │  1. landmarks → wrist_rot @ OPERATOR2MANO_RIGHT → MANO space    │     │
│  │  2. adaptive_retargeting_xhand() — LeFranX pinky chain scaling  │     │
│  │  3. dex_retargeting DexPilot optimizer (NLopt SLSQP)            │     │
│  │  4. LPFilter (α=0.6, dex_retargeting 内置)                      │     │
│  │  5. Output: 12-DOF qpos (sapien joint order)                    │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│    │ target_qpos (12,)                                                    │
│    ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ XHand.send_action(qpos) — 直接发送，无 PID 内环                   │     │
│  │  1. np.clip(qpos, qpos_min, qpos_max) — 硬件关节限位              │     │
│  │  2. EMA smoothing (可选, α 可配, 默认 0.0 关闭)                  │     │
│  │  3. write_command_positions → control.send_command()             │     │
│  │  4. 错误追踪: _consecutive_send_errors (成功归零, 失败+1)        │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│    │ RS485 @ 3Mbps → XHand 固件 (30Hz 控制频率)                          │
│    ▼                                                                     │
│  XHand Hardware (12 电机, 5 指尖触觉传感器)                               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 关键时序

| 特性 | 外环 (Main Thread) | XHand 驱动层 |
|------|-------------------|-------------|
| 频率 | 50 Hz | 30 Hz (dt=1/30s) |
| 周期 | 20 ms | ~33 ms |
| 指令类型 | 位置目标 `qpos_cmd[12]` | `send_command(position)` |
| 控制模式 | — | mode 3 (位置伺服, 固件 PID) |
| 同步方式 | `RateLimiter.wait()` 补偿式 | 按需发送 (无独立线程) |
| 连接类型 | — | RS485 `/dev/ttyUSB0` @ 3Mbps |
| 架构对标 | — | LeFranX / DexUMI (线程内发送) |

### 与 Arm 控制回路的关键差异

| 维度 | Arm (xArm7) | Hand (XHand) |
|------|------------|-------------|
| 控制层数 | 双层 (50Hz + 250Hz 内环) | 单层 (50Hz 外环直接发送) |
| 控制模式 | mode 4 (速度控制 + 用户态 PID) | mode 3 (位置伺服, 固件 PID) |
| 内环线程 | ArmInnerLoop (daemon, 250Hz) | 无 (无需高速 PID) |
| 平滑 | RobustEMA (α=0.95→0.3 自适应) | 可选 EMA (α 固定, 默认关闭) |
| IK | DLS 迭代 IK | dex_retargeting 优化器 (NLopt) |
| 通信 | threading.Lock + numpy array | 函数调用 (同线程) |
| 安全门 | validate_action (3 项) | 关节限位 clip + 错误恢复 |

---

## 2. 参考矩阵

DexMani XHand 驱动有意识地参考了三个项目，同时保留了经过验证的独家机制。

### 2.1 采纳的外部模式

| 来源 | 模式 | DexMani 实现 | 对齐度 |
|------|------|-------------|--------|
| **LeFranX** | kp=80, ki=0, kd=0 | `XHandConfig` 默认值 | ✅ |
| **LeFranX** | dt=1/30s (30Hz 控制频率) | `XHandConfig.dt` | ✅ |
| **LeFranX** | open_serial_retries=3 | `XHandConfig.open_serial_retries` | ✅ |
| **LeFranX** | 已知传感器错误过滤 | `filter_known_sensor_errors` + `_KNOWN_SENSOR_ERROR_PATTERNS` | ✅ |
| **LeFranX** | EMA 关节空间平滑概念 | `ema_alpha` (默认 0.0 关闭) | ⚠️ |
| **LeFranX** | adaptive_retargeting_xhand (小指自适应) | `adaptive_retargeting_xhand()` landmark 空间缩放 | ✅ |
| **DexUMI** | MotorTrajectoryInterpolator (scipy interp1d) | `motor_trajectory_interpolator.py` | ✅ |
| **DexUMI** | 触觉接触检测 (L2 norm > threshold) | `detect_contact()` binary_cutoff=10 | ✅ |
| **DexUMI** | 逐关节增益覆盖 | `kp_per_joint` / `ki_per_joint` / `kd_per_joint` | ✅ |
| **skill-teleop** | URDF 逐关节限位 | `qpos_min` / `qpos_max` (精确 rad 值) | ✅ |

### 2.2 不采纳的模式及原因

| 不采纳 | 来源 | 原因 |
|--------|------|------|
| E1 后台状态读取线程 | DexMani 旧版 | 移除 — 线程内同步读取已满足 50Hz 需求 |
| 远端关节 5° 最小限位 | LeFranX | LeFranX 独有机械特性，XHand 无需 |
| trajectory 自动速度延长 | DexMani 旧版 | 替换为 DexUMI MotorTrajectoryInterpolator |
| tactile_scale (0.1) | DexMani 旧版 | 无参考项目使用，移除 — 直接记录原始传感器值 |
| stitched_retargeting | skill-teleop | 需要策略网络，不适合遥操作管线 |
| EtherCAT 通信 | 未来选项 | RS485 已验证稳定 (~0.5% CRC 率)，无需升级 |

### 2.3 运动范围差异：URDF vs skill-teleop

DexMani 使用 URDF 机械限位，skill-teleop 使用自定义数值。两者在拇指关节上存在显著差异：

| 关节 | DexMani (URDF) | skill-teleop | 差异 |
|------|---------------|-------------|------|
| J0 thumb_abd | [0, 1.832] | [0, 1.830] | 上限差 0.002 rad (可忽略) |
| **J1 thumb_j1** | **[-0.698, 1.57]** | **[-1.05, 1.57]** | **下限差 -0.352 rad (-20°) ⚠️** |
| **J2 thumb_j2** | **[0, 1.57]** | **[-0.17, 1.83]** | **下限 +0.17 rad / 上限 +0.26 rad (+15°) ⚠️** |
| J3 index_abd | [-0.175, 0.174] | [-0.175, 0.175] | 上限差 0.001 rad (可忽略) |
| J4-J11 | [0, 1.919] | [0, 1.920] | 上限差 0.001 rad (可忽略) |

**为什么会有这些差异？**

skill-teleop 的限位是配合其 BC 策略网络手工调整的，策略网络在训练时不会输出极端值，因此放宽限位无害。DexMani 直接遥操作，操作者的手势可能触及物理极限，使用 URDF 精确值更安全：

- **J1 下限 (-0.698 vs -1.05)**：URDF 中 thumb_j1 负方向被机械结构限制在 -40°。skill-teleop 允许 -60° 是因为策略网络实际输出范围在安全区间内，放宽下限不会触发。
- **J2 范围 ([0, 1.57] vs [-0.17, 1.83])**：URDF 中 thumb_j2 不能反向弯曲（下限=0），且上限 1.57 rad (90°)。skill-teleop 允许 -0.17 rad 反向和 1.83 rad (105°) 上限，同样是策略网络调优产物。

**DexMani 的选择**：使用 URDF 精确值而非 `deg2rad()` 转换，避免浮点舍入误差（`deg2rad(105°)=1.8326` vs URDF `1.832`，差 0.0006 rad）。

### 2.4 独家保留机制

| # | 机制 | 本质 | 三个参考项目均无 |
|---|------|------|-----------------|
| **H1** | clip_report_tolerance (0.01 rad) | 过滤 retargeter 边界噪声，避免误报 CLIP — 始终执行 np.clip 但只在实际越界时报告 | LeFranX/DexUMI 无条件 clip；skill-teleop 无此概念 |
| **H2** | 错误码分级恢复 (CRC 50ms / Boot 500ms) | 根据错误码类型选择退避延迟，避免 CRC→Boot CMD 雪崩 | 三者均为简单 clear + retry |
| **H3** | 熔断重连 (_consecutive_send_errors ≥ 10) | 连续错误计数触发 disconnect→connect 完整硬件复位 | 三者均无此保护 |

---

## 3. XHand 驱动层

### 3.1 连接生命周期

```
XHand(config) → connect()
  ├─ 枚举设备 (enumerate_devices)
  ├─ open_serial() — 最多 3 次重试 (间隔 2s)
  │   失败 → error_state + 诊断日志
  ├─ 初始化状态读取 (3 帧强制刷新, 避免零缓存)
  │   有效 → last_qpos_cmd = 当前 qpos
  │   无效 → last_qpos_cmd = home_qpos
  └─ connected_flag = True

disconnect()
  └─ control.close_device()
```

### 3.2 send_action() 流程

```
send_action(qpos[12])
│
├─[1] _array12(action) — 安全 reshape 到 (12,)
├─[2] _limit_joint_range(qpos) — np.clip(qpos_min, qpos_max)
│      max|raw - clipped| > clip_report_tolerance (0.01 rad) → CLIP flag
├─[3] EMA smoothing (可选, ema_alpha > 0)
│      qpos_cmd = (1-α) * prev + α * current
├─[4] write_command_positions(qpos_cmd) — 更新 HandCommand_t 结构体
├─[5] control.send_command(device_id, command) — SDK C++ 调用
│
├─ 成功 → last_qpos_cmd 更新, _consecutive_send_errors = 0
└─ 失败 → _record_error(err), _consecutive_send_errors += 1
```

### 3.3 状态读取

```
get_state(full=False, force_update=True)
│
├─ control.read_state(device_id, force_update)
├─ parse_state(hand_state)
│   ├─ 12 关节: position, torque(current), raw_position, temperature
│   ├─ 错误标志: commboard_err, jointboard_err, tipboard_err
│   └─ 5 指尖触觉: (5,120,3) raw force + (5,3) calc_force
└─ detect_contact(tactile_force_sum) — L2 norm > 10.0
```

**force_update 说明**: 默认 True，强制 SDK 从硬件刷新状态（避免缓存）。设为 False 可减少总线流量，但 connect() 初始化阶段始终强制刷新。

---

## 4. 轨迹插值器

### 4.1 设计动机

移除旧的 trajectory 自动速度延长（在 `send_trajectory` 中硬编码 waypoint 时间扩展），替换为 DexUMI 的 `MotorTrajectoryInterpolator`。

旧方案问题:
- 自动速度延长逻辑与 send_trajectory 耦合
- 无 speed-limited waypoint driving 语义
- 不支持轨迹裁剪和错过路点恢复

DexUMI 方案优势:
- 基于 scipy.interp1d，线性插值，简单可靠
- 支持 speed-limited waypoint: 自动延长到达时间以遵守速度限制
- 支持轨迹裁剪 (trim) 和路点调度 (schedule_waypoint)

### 4.2 核心 API

```python
from dexmani_real.robot.xhand import MotorTrajectoryInterpolator

# 多点插值
times = np.array([0.0, 1.0, 2.0])
values = np.array([[0]*12, [1]*12, [2]*12])  # (3, 12)
interp = MotorTrajectoryInterpolator(times, values)
pos = interp(1.5)  # t=1.5 时的插值位置

# 单点保持
interp = MotorTrajectoryInterpolator(times=np.array([0.0]), values=home[None, :])
pos = interp(10.0)  # 始终返回 home

# Speed-limited waypoint driving
interp2 = interp.drive_to_waypoint(
    value=target, time=2.0, curr_time=0.5, max_speed=3.0
)
# 若 target 距离 > max_speed × (2.0-0.5)，自动延长到达时间

# 轨迹裁剪
trimmed = interp.trim(t_start=0.5, t_end=1.5)
```

### 4.3 send_trajectory() 集成

```python
xhand.send_trajectory(waypoints, duration_s, max_speed=None)
# 默认 max_speed = min(config.max_qvel) = 180°/s
# 单 waypoint → send_action()
# 多 waypoint → MotorTrajectoryInterpolator + dt 步进执行
```

---

## 5. VR 手部重定向

### 5.1 完整管线

```
VR landmarks (21,3) [Quest/MediaPipe 坐标系]
  │
  ├─ estimate_frame_from_hand_points(landmarks)
  │    SVD 估计手部坐标系 → wrist_rot (3,3)
  │
  ├─ landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
  │    转换到 MANO 骨骼空间 → mano_landmarks (21,3)
  │
  ├─ adaptive_retargeting_xhand(mano_landmarks)          ← LeFranX 小指自适应
  │    拉伸 pinky MCP→PIP→DIP→TIP 链段
  │    自适应缩放: 伸展时 scale↑ (max 2.2), 弯曲时 scale↓ (min 1.2)
  │
  ├─ DexPilotOptimizer.retarget(ref_value)
  │    origin→task 向量差 → NLopt SLSQP → 12-DOF qpos
  │    LPFilter (α=0.6) 内置平滑
  │
  └─ retargeted_joint_order 重排 → sapien 关节顺序
```

### 5.2 LeFranX 小指自适应

```python
# hand_retarget.py: adaptive_retargeting_xhand()
# 对标 LeFranX vr_hand_detector_adapter.py:27-84

# 手指伸展程度: MCP→TIP 距离
extension = ||landmarks[PINKY_TIP] - landmarks[PINKY_MCP]||

# 归一化 (0=完全弯曲, 1=完全伸展)
ratio = clip((extension - 0.03) / (0.10 - 0.03), 0, 1)

# 自适应缩放因子
scale = 1.2 + (2.2 - 1.2) × ratio  # 范围 [1.2, 2.2]

# 沿运动链逐段缩放 (MCP 不动)
PIP = MCP + scale × (PIP_old - MCP)
DIP = PIP + scale × (DIP_old - PIP)
TIP = DIP + scale × (TIP_old - DIP)
```

**设计原理**: LeFranX 方案直接在 MANO landmark 空间操作，比旧 `XHandRefAdapter` (ref-value 空间 blend) 更简单直接。缩放后的 landmarks 直接进入 dex_retargeting 优化器，无需额外的 blend 参数。

### 5.3 关节限位与 CLIP 标志

dex_retargeting 的 NLopt SLSQP 优化器在边界手势下可能超出 URDF 限位 0.5-3°（优化器 epsilon=1e-3 rad 的 slack）。`clip_report_tolerance=0.01 rad` 过滤这些亚度级噪声：

| 偏差 | CLIP 标志 | 实际行为 |
|------|----------|---------|
| < 0.01 rad (0.57°) | 不报告 | np.clip 始终执行 |
| ≥ 0.01 rad | 报告 CLIP | np.clip 始终执行 |

### 5.4 关节顺序映射

```
dex_retargeting 输出顺序 (URDF dof_joint_names):
  thumb_bend → thumb_rota_j1 → thumb_rota_j2 → index_bend →
  index_j1 → index_j2 → mid_j1 → mid_j2 → ring_j1 → ring_j2 →
  pinky_j1 → pinky_j2

XHand 硬件顺序 (JOINT_NAMES):
  thumb_abd → thumb_j1 → thumb_j2 → index_abd →
  index_j1 → index_j2 → mid_j1 → mid_j2 → ring_j1 → ring_j2 →
  little_j1 → little_j2

通过 retargeted_joint_order 重排对齐
```

---

## 6. 安全架构

### 6.1 三层防护

| 层 | 位置 | 内容 |
|----|------|------|
| **L1** | XHandConfig.qpos_min/max | 逐关节硬件限位 (URDF 精确值) — `_limit_joint_range()` np.clip |
| **L2** | XHand 固件 | tor_max=300mA 硬件限流, 温度保护, 通信错误检测 |
| **L3** | 错误恢复 | 错误码分级退避 + 熔断重连 (≥10 连续错误) |

### 6.2 关节限位 (URDF 精确值)

| 关节 | 名称 | 下限 (rad) | 上限 (rad) | 上限 (deg) |
|------|------|-----------|-----------|-----------|
| J0 | thumb_abd | 0.0 | 1.832 | 105° |
| J1 | thumb_j1 | -0.698 (-40°) | 1.57 | 90° |
| J2 | thumb_j2 | 0.0 | 1.57 | 90° |
| J3 | index_abd | -0.175 (-10°) | 0.174 | 10° |
| J4-J11 | 各指关节 | 0.0 | 1.919 | 110° |

**注意**: 使用 URDF 精确 rad 值而非 `deg2rad()` 转换，避免浮点舍入误差 (如 `deg2rad(105°)=1.8326` vs URDF `1.832`，差值 0.0006 rad)。

### 6.3 已知传感器错误过滤

参考 LeFranX，以下硬件级告警不影响关节位置读取和运动控制，仅记录 debug 日志：

- 传感器组合力读取失败
- 传感器分布力读取失败
- 传感器温度读取失败
- 通信 CRC 错误
- 硬件版本不支持力控模式

由 `filter_known_sensor_errors=True` + `_KNOWN_SENSOR_ERROR_PATTERNS` 控制。过滤后不会触发 `error_state`。

---

## 7. 错误处理与恢复

### 7.1 错误码体系

| 错误码 | 常量 | 含义 | 恢复延迟 | 处理策略 |
|--------|------|------|---------|---------|
| 0 | `ERR_OK` | 成功 | — | — |
| 1501070 | `ERR_CRC` | RS485 通信 CRC 错误 | 50ms | 短暂延迟后重试（瞬态总线错误） |
| 1501036 | `ERR_BOOT_CMD` | 手控制器重初始化中 | 500ms | 长延迟等待固件完成 boot |
| 其他 | — | 未知错误 | 100ms | 保守延迟后重试 |

### 7.2 三级恢复策略

```
send_action() 失败
│
├─[1] _record_error(err) → error_state + _consecutive_send_errors++
│
├─[2] 调用方获取 get_recovery_delay(error_code)
│      CRC (1501070) → 50ms
│      Boot (1501036) → 500ms
│      未知 → 100ms
│
├─[3] clear_error() → 清零 Python 侧 error_state
│
├─[4] time.sleep(delay) → 等待硬件恢复
│
├─[5] 连续错误 < 10 → retry send_action()
│
└─[6] 连续错误 ≥ 10 → 熔断!
       reset_connection()
         ├─ disconnect() → 等待 1s → connect()
         ├─ 成功 → 重置计数器, 继续遥操作
         └─ 失败 → 退出遥操作循环
```

### 7.3 CRC→Boot 雪崩防护

**问题**: 旧代码在 CRC 错误后立即 retry (50Hz 无延迟)。手固件因通信故障进入 boot 恢复模式，拒绝所有位置命令 (1501036)，导致 ~100+ 连续错误，约 4 秒宕机。

**修复**: 根据错误码分级退避。CRC 错误延迟 50ms 让 RS485 总线清空后再重试 → 手固件不进入 boot 模式，避免雪崩。

**效果对比**:

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| CRC 后 1501036 雪崩 | ~100+ 次 (4.2s 宕机) | 0 次 (完全消除) |
| 单次 CRC 恢复时间 | 4.2s (依赖固件自行恢复) | 50ms (退避后立即成功) |
| 熔断重连触发 | 无此机制 | ≥10 连续错误自动触发 |

### 7.4 reset_connection() 熔断

```python
def reset_connection(self) -> bool:
    # 硬件级完整复位
    disconnect()        # 关闭串口
    time.sleep(1.0)     # 等待硬件稳定
    ok = connect()      # 重新枚举 + open_serial + 状态初始化
    # 成功 → _consecutive_send_errors = 0
    return ok
```

---

## 8. 触觉传感

### 8.1 传感器布局

5 个指尖传感器 (thumb/index/middle/ring/little)，每个传感器:

| 数据 | 形状 | 说明 |
|------|------|------|
| raw_force | (120, 3) | 120 个感测点 × 3 轴 (fx, fy, fz) |
| calc_force | (3,) | 组合力 (L2 合成) |
| temperature | (20,) | 传感器温度分布 |

### 8.2 接触检测

```python
# detect_contact(threshold=10.0)
# 对标 DexUMI eval_xhand.py:72 binary_cutoff=[10,10,10]
force_sum = parse_tactile_sum(state)  # (5, 3)
norm = ||force_sum||₂                 # (5,) 每指 L2 范数
contact = norm > 10.0                 # bool[5]
```

**无 tactile_scale**: 直接记录原始传感器值，不做缩放。参考 DexUMI 和 skill-teleop 均无 scale 参数。

---

## 9. 与参考项目对比

### 9.1 架构对比

| 维度 | LeFranX | DexUMI | skill-teleop | DexMani |
|------|---------|--------|-------------|---------|
| **通信方式** | EtherCAT | RS485 | RS485 (推测) | RS485 @ 3Mbps |
| **控制频率** | 30Hz | 30Hz | — | 30Hz (dt=1/30) |
| **控制模式** | mode 3 (位置伺服) | mode 3 | mode 3 | mode 3 (位置伺服) |
| **PID 增益** | kp=80, ki=0, kd=0 | kp=80, ki=0, kd=0 | kp=100, ki=0, kd=1 | kp=80, ki=0, kd=0 |
| **扭矩限制** | 400mA | 400mA | 100mA | 300mA |
| **连接重试** | 无 | — | — | 3 次 |
| **重定向** | dex_retargeting + adaptive_retargeting_xhand | 策略网络 (BC) | 策略网络 (BC) | dex_retargeting + adaptive_retargeting_xhand |
| **小指自适应** | landmark 空间 | 无 | — | landmark 空间 (LeFranX 同) |
| **触觉检测** | — | binary_cutoff=10 | — | threshold=10.0 |
| **关节限位来源** | URDF | URDF | 自定义数组 | URDF (精确 rad 值) |
| **错误过滤** | ✅ filter_known | — | — | ✅ filter_known |
| **CRC 恢复** | — | — | — | 分级退避 + 熔断重连 |
| **Tactile scale** | — | 无 | 无 | 无 |
| **轨迹插值** | — | MotorTrajectoryInterpolator | — | MotorTrajectoryInterpolator |
| **EMA 平滑** | α=0.3 (遥操作层) | — | — | α=0.0 (默认关闭) |

### 9.2 设计哲学

```
LeFranX:     工业鲁棒 ── "固件处理一切，上层只发目标"
DexUMI:      策略优先 ── "插值器 + 接触检测 = 安全操作"
skill-teleop: 策略中心 ── "BC 策略驱动，配置最小化"
DexMani:     实用主义 ── "LeFranX 配置 + DexUMI 工具 + 独家错误恢复"
```

### 9.3 独家优势

| 优势 | 说明 |
|------|------|
| 错误码分级恢复 | CRC 50ms / Boot 500ms 差异化退避，消除 1501036 雪崩 |
| 熔断重连 | ≥10 连续错误自动 hardware reset，避免无限 retry |
| clip_report_tolerance | 过滤 retargeter 边界噪声，CLIP 标志只在真越界时报告 |
| URDF 精确限位 | 使用 URDF 原始 rad 值，避免 deg2rad 浮点舍入误差 |
| 逐关节增益覆盖 | 支持 kp/ki/kd_per_joint，远端关节可独立调参 |
| MotorTrajectoryInterpolator 集成 | DexUMI scipy 插值 + speed-limited waypoint + trim/schedule |

---

## 10. 关键参数速查

### 10.1 通信与连接

| 参数 | 值 | 位置 |
|------|-----|------|
| 通信类型 | RS485 | `XHandConfig.comm_type` |
| 设备路径 | `/dev/ttyUSB0` | `XHandConfig.device_name` |
| 波特率 | 3,000,000 | `XHandConfig.baudrate` |
| 连接重试次数 | 3 | `XHandConfig.open_serial_retries` |
| 重试间隔 | 2.0s | `XHandConfig.open_serial_retry_delay_s` |
| 初始化状态读取 | 3 帧 | `XHandConfig.init_state_read_attempts` |

### 10.2 控制参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 控制频率 | 30 Hz (dt=1/30s) | `XHandConfig.dt` |
| 控制模式 | mode 3 (位置伺服) | `XHandConfig.mode` |
| kp | 80 | P 增益 (对标 LeFranX) |
| ki | 0 | I 增益 |
| kd | 0 | D 增益 |
| tor_max | 300 mA | 扭矩限制 (LeFranX=400, 降低 25% 安全余量) |
| force_update_state | True | 强制从硬件刷新 (避免 SDK 缓存) |

### 10.3 关节限位

| 参数 | 值 | 位置 |
|------|-----|------|
| qpos_min | `[0,-0.698,0,-0.175,0,0,0,0,0,0,0,0]` rad | `XHandConfig` |
| qpos_max | `[1.832,1.57,1.57,0.174,1.919×8]` rad | `XHandConfig` (URDF 精确值) |
| max_qvel | 180°/s (π rad/s) × 12 | `XHandConfig.max_qvel` |
| clip_report_tolerance | 0.01 rad (~0.57°) | `XHandConfig.clip_report_tolerance` |

### 10.4 平滑与滤波

| 参数 | 值 | 位置 |
|------|-----|------|
| EMA 平滑 | α=0.0 (默认关闭) | `XHandConfig.ema_alpha` |
| LeFranX 推荐 α | 0.3 | 参考 |
| dex_retargeting LPFilter | α=0.6 | dex_retargeting 内置 |
| 已知错误过滤 | True | `XHandConfig.filter_known_sensor_errors` |

### 10.5 小指自适应

| 参数 | 值 | 来源 |
|------|-----|------|
| 最小伸展 | 0.03 (完全弯曲) | LeFranX |
| 最大伸展 | 0.10 (完全伸展) | LeFranX |
| 基础缩放 | 1.2 (弯曲时) | LeFranX |
| 最大缩放 | 2.2 (伸展时) | LeFranX |

### 10.6 触觉

| 参数 | 值 | 位置 |
|------|-----|------|
| 接触检测阈值 | 10.0 (L2 norm) | `XHandConfig.tactile_contact_threshold` |
| raw_force 形状 | (5, 120, 3) | `parse_tactile()` |
| calc_force 形状 | (5, 3) | `parse_tactile_sum()` |
| 缩放 | 无 (原始值) | 对标 DexUMI/skill-teleop |

### 10.7 错误恢复

| 参数 | 值 | 说明 |
|------|-----|------|
| CRC 恢复延迟 | 50ms | RS485 瞬态总线错误 |
| Boot CMD 恢复延迟 | 500ms | 手固件重初始化 |
| 未知错误恢复延迟 | 100ms | 保守默认值 |
| 熔断阈值 | 10 次连续错误 | 触发 `reset_connection()` |
| 熔断重连等待 | 1.0s | disconnect → connect 间隔 |

### 10.8 Home 位置

| 关节 | 角度 |
|------|------|
| thumb_abd | 0° |
| thumb_j1 | 45° |
| thumb_j2 | 0° |
| index_abd | 0° |
| index_j1-j2 | 0° |
| mid_j1-j2 | 0° |
| ring_j1-j2 | 0° |
| little_j1-j2 | 0° |

---

## 11. 文件索引

| 模块 | 文件 | 行数 | 说明 |
|------|------|------|------|
| **XHand 驱动** | `robot/xhand/xhand.py` | 805 | 硬件驱动: 连接/状态/发送/错误恢复 |
| **XHandConfig** | `robot/xhand/xhand.py` | 58-190 | 配置 dataclass: 限位/增益/触觉/EMA |
| **MotorTrajectoryInterpolator** | `robot/xhand/motor_trajectory_interpolator.py` | 252 | DexUMI scipy 轨迹插值器 |
| **HandRetargeter** | `teleop/vr/hand_retarget.py` | 170 | VR→12-DOF 重定向 + 小指自适应 |
| **VRTracker** | `teleop/vr/vr_tracker.py` | — | Quest 手部追踪 (TCP/UDP) |
| **ArmMapper** | `teleop/vr/arm_mapper.py` | — | VR wrist → EEF pose |
| **ConnectionStateMixin** | `robot/_connection_state.py` | — | 连接状态管理基类 |
| **测试入口** | `examples/real/test_quest_hand_teleop.py` | 273 | 真机 VR 手部遥操作测试 |
| **仿真入口** | `examples/sim/vr_teleop_sim.py` | — | SAPIEN 仿真遥操作 |

### 演进历史

| 日期 | 变更 | 来源 |
|------|------|------|
| 2025-06-25 | 移除 E1 后台状态读取线程 | DexUMI/skill-teleop 对齐 |
| 2025-06-25 | 移除远端关节 5° 最小限位 | LeFranX 独有，XHand 无需 |
| 2025-06-25 | MotorTrajectoryInterpolator 替换旧速度延长 | DexUMI |
| 2025-06-25 | kp/kd/dt 对齐 LeFranX (80/0/30Hz) | LeFranX |
| 2025-06-25 | 移除 tactile_scale，阈值改为 10 | DexUMI/skill-teleop 对齐 |
| 2025-06-25 | qpos_max 改为 URDF 精确 rad 值 | URDF 直接提取 |
| 2025-06-25 | 新增 clip_report_tolerance (0.01 rad) | 独家 — 过滤 retargeter 边界噪声 |
| 2025-06-25 | 错误码分级恢复 + 熔断重连 | 独家 — 消除 CRC→Boot 雪崩 |
| 2025-06-25 | LeFranX adaptive_retargeting_xhand 替换 XHandRefAdapter | LeFranX |

---

## 附录: 与 Arm 文档的交叉引用

| 主题 | Arm 文档章节 | Hand 文档章节 |
|------|------------|-------------|
| 架构总览 | §1 | §1 |
| 参考矩阵 | §2 | §2 |
| 控制模式决策 | §4.1 (mode 4 vs mode 1) | §3.2 (mode 3 位置伺服) |
| PID 参数 | §4.3 | §10.2 (固件 PID, 无用户态) |
| EMA 平滑 | §5.3 (RobustEMA) | §3.2 (固定 α EMA, 默认关闭) |
| IK/Retarget | §5.4 (DLS IK) | §5 (dex_retargeting) |
| 安全架构 | §6 (3 层) | §6 (3 层) |
| 错误处理 | §3.3 (故障响应) | §7 (分级恢复 + 熔断) |
| Return-to-Home | §8 (三级 Fallback) | §3.1 (home_qpos, 6 步 ~120ms) |
| 端到端延迟 | §9 | §1 (50Hz 外环直接发送, 无内环) |
| 参数速查 | §10 | §10 |
| 项目对比 | §11 | §9 |
| 文件索引 | §12 | §11 |
