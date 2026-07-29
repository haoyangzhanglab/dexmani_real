# XHand 全面对比：LeFranX vs DexMani

> 基于双方代码库逐行分析，2026-07-29
> 覆盖 10 个维度，每个维度含证据、差异、裁决

---

## 目录

1. [硬件连接与初始化](#1-硬件连接与初始化)
2. [状态读取](#2-状态读取)
3. [命令发送](#3-命令发送)
4. [安全架构](#4-安全架构)
5. [触觉感知](#5-触觉感知)
6. [手部重定向（VR→XHand）](#6-手部重定向vrxhand)
7. [数据录制](#7-数据录制)
8. [进程与线程模型](#8-进程与线程模型)
9. [配置与调优](#9-配置与调优)
10. [集成模式](#10-集成模式)
11. [总体评估与建议](#11-总体评估与建议)

---

## 1. 硬件连接与初始化

### 1.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| SDK 导入 | Lazy（`connect()` 内） | Eager（模块顶层，失败回退 None） |
| 连接协议 | RS485（EtherCAT 是 TODO stub） | RS485 + EtherCAT（两者均完整实现） |
| 设备枚举 | 无（依赖 `list_hands_id` 后验） | `enumerate_devices()` 预枚举 |
| 连接重试 | 无 | 可配置重试次数 + 延迟 + 设备重建 |
| Device ID 验证 | 隐式（取第一个 hand） | 显式（验证 config.device_id ∈ enumerated） |
| 设备元数据 | 无 | SDK 版本、手类型、序列号（均非阻断） |
| 状态初始化 | 无 | 强制刷新 3+ 轮，回退到 home_qpos |
| 触觉传感器初始化 | 无 | 5 个指尖传感器逐一 reset（stdout 抑制） |
| Stub 模式 | 有（SDK 不可用时） | 有（同模式，`_stub_mode` flag） |
| 诊断日志 | 无 | 连接失败时平台特定提示（EtherCAT: CAP_NET_RAW） |

### 1.2 关键差异

**DexMani `connect()` 比 LeFranX 多 5 层保护**:

```
LeFranX:                      DexMani:
  import SDK                     import SDK (eager)
  XHandControl()                 XHandControl()
  open_serial()                  enumerate_devices()     ← 预枚举
  list_hands_id()                _retry_open_device()    ← 重试循环 + 设备重建
  make HandCommand_t             device_id validation     ← 显式 ID 验证
  → done                         _verify_device()        ← SDK 版本/类型/序列号
                                 make_command(home_qpos)
                                 _init_hand_state()      ← 3 轮强制刷新
                                 _reset_tactile_sensors() ← 5 传感器重置
                                 → done
```

### 1.3 裁决

**DexMani 显著更优。** LeFranX 的连接流程是"最简可行"——打开串口、取第一个 hand、开始发送命令。DexMani 增加了 5 层容错：设备枚举→重试→ID 验证→状态初始化→传感器复位。在冷启动/热插拔/通信故障场景下，DexMani 不会静默失败。

**LeFranX 缺失的关键能力**: 连接重试（`_retry_open_device`）、触觉传感器初始化（`_reset_tactile_sensors`）、设备元数据日志。

---

## 2. 状态读取

### 2.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 读取频率 | 随 `get_action`/`get_observation` 调用 | 30Hz 固定（hand child 独立循环） |
| force_update | 始终 True | 可配置（默认 True，允许缓存加速） |
| 读取字段 | position, torque（2 字段） | position, torque, raw_position, sensor_id, finger_id, commboard_err, jointboard_err, tipboard_err（8+ 字段） |
| 触觉数据 | **不读取** | (5,120,3) raw force + (5,3) force_sum + (5,) contact |
| 错误过滤 | 5 种已知传感器错误 | 5 种（identical）+ 可配置开关 + board 级错误门 |
| 错误时行为 | 返回 None → 部分 dict | 返回 NaN qpos + 零触觉（从不 None） |
| 电路板错误监控 | 无 | commboard_err + jointboard_err + tipboard_err → 自动 error_state |
| Stale 检测 | 无 | SHM ring 新鲜度门（100ms）→ degraded mode |
| 关节命名 | `joint_0..11`（通用） | `thumb_abduction`..`little_joint2`（语义化） |

### 2.2 关键差异

**DexMani 的 `get_state()` 返回 8 个额外字段**:

```
LeFranX state:                    DexMani state:
  positions (12)                    qpos (12)          ← 相同语义
  torques (12, 实际是电流)           current (12)        ← 明确定义为电流
                                    raw_position (12)   ← 原始编码器
                                    finger_ids (12)     ← SDK 硬件 ID
                                    sensor_ids (12)     ← 传感器 ID
  --- 以下 LeFranX 完全没有 ---
                                    tactile_force (5,120,3)   ← 每指 120 点 × 3 轴
                                    tactile_force_sum (5,3)   ← 合力向量
                                    tactile_contact (5,)      ← 接触布尔
                                    commboard_err (12)        ← 通信板错误
                                    jointboard_err (12)       ← 关节板错误
                                    tipboard_err (12)         ← 指尖板错误
```

**电路板错误监控**是 DexMani 独有的安全层。`parse_state()` 在任意 board error 非零时自动设置 `error_state=True`。这意味着手指硬件故障会传播到 `validate_action()` 的 `_check_hardware_error` 门，从而停止向故障手指发送命令。LeFranX 完全感知不到硬件级错误。

### 2.3 裁决

**DexMani 显著更优。** 触觉数据（5×120×3 力阵列）和电路板错误监控是关键的差异化优势。LeFranX 的 state model 仅限于位置+电流，无法支持基于力的控制、接触检测或硬件故障诊断。

---

## 3. 命令发送

### 3.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 动作格式 | `Dict[str, Any]`（keyed `joint_i.pos`） | `np.ndarray`（shape (12,)） |
| PID 增益 | 标量（80/0/0），所有关节相同 | 标量（100/0/0）+ **逐关节覆盖数组** |
| tor_max | 400 mA | 320 mA（硬件规格限制） |
| 关节限位裁剪 | 有（`_apply_safety_limits`） | 有（`_limit_joint_range`）+ clip_report_tolerance |
| Delta 跳变限制 | **无** | 有（`max_delta_rad=0.3 rad/step`） |
| 死区节流 | **无** | 有（0.001 rad ≈ 0.06°） |
| NaN/Inf 防护 | **无** | 有（保持上一条有效命令） |
| 命令结构体重用 | connect 时创建，复用 | `make_command()` 按需构造 + `write_command_positions()` 就地更新 |
| 错误恢复 | 无 | 错误码追踪 + 连续错误断路器 + `reset_connection()` |
| 轨迹插值 | **无** | `MotorTrajectoryInterpolator` + `send_trajectory()` |
| E-stop | **TODO stub** | **完整实现**：零扭矩命令 + error_state=True |
| 保留字段 | 未设置 | `res0-3` 显式清零 |

### 3.2 关键差异

**DexMani `send_action()` 比 LeFranX 多 4 层保护**:

```
LeFranX send_action:              DexMani send_action:
  extract positions from dict       NaN/Inf 检测 → hold last        ← 第 1 层
  _apply_safety_limits (clip)       _limit_joint_range (clip)        ← 相同
  update hand_command               E3 delta limit (0.3 rad/step)   ← 第 2 层
  device.send_command()             死区节流 (<0.001 rad)            ← 第 3 层
  filter errors                     write_command_positions
  → return True/False               device.send_command()
                                    error_ok? → 更新状态
                                    _record_error + 断路器           ← 第 4 层
```

**逐关节 PID 增益**是 DexMani 独有特性。远端关节（尤其是小指 J11）需要更高增益补偿长连杆和高机械负载。LeFranX 的标量 kp=80 对所有关节一视同仁，无法针对性调校。

### 3.3 裁决

**DexMani 显著更优。** NaN/Inf 防护、delta 跳变限制、死区节流和逐关节增益是安全关键且性能相关的特性。LeFranX 缺失所有这些。E-stop 的缺失是 LeFranX 最严重的安全缺陷——`stop()` 方法仅 `logger.info("NOT IMPLEMENTED")`。

---

## 4. 安全架构

### 4.1 安全机制全景对比

| # | 安全机制 | LeFranX | DexMani | 严重程度 |
|---|---------|---------|---------|---------|
| 1 | 关节限位裁剪 | ✅ | ✅ | 基础 |
| 2 | SDk 错误过滤 | ✅ | ✅ | 基础 |
| 3 | NaN/Inf 动作门 | ❌ | ✅ | **高** |
| 4 | Arm 力矩门（Nm） | ❌ | ✅（J1-2=50, J3-5=30, J6-7=20） | **高** |
| 5 | 触觉力门（30N/指） | ❌ | ✅ | **高** |
| 6 | 温度门（70°C） | ❌ | ✅（Arm） | 中 |
| 7 | Delta 跳变限制 | ❌ | ✅（0.3 rad/step） | **高** |
| 8 | 死区节流 | ❌ | ✅（0.001 rad） | 低 |
| 9 | 电路板错误门 | ❌ | ✅（3 类 board error） | **高** |
| 10 | E-stop | ❌（NOT IMPLEMENTED） | ✅（零扭矩 + error_state） | **致命** |
| 11 | 连接丢失检测 | ❌ | ✅（SHM 新鲜度门 + 心跳） | **高** |
| 12 | 连续错误断路器 | ❌ | ✅（30 次 → reset_connection） | **高** |
| 13 | 错误码追踪 + 恢复延迟 | ❌ | ✅（CRC→50ms, BOOT→500ms） | 中 |
| 14 | Arm 速度 NaN 门 | N/A | ✅ | 中 |
| 15 | 硬件错误状态门 | ❌ | ✅（hand.error_state → 阻断动作） | **高** |

### 4.2 安全架构层次

```
DexMani 安全架构（4 层）:               LeFranX 安全架构（1 层）:

Layer 4: validate_action() 集中门          (无)
  ├─ NaN/Inf 门
  ├─ 力矩门 (arm)
  ├─ 触觉力门 (hand)                      (无)
  ├─ 温度门 (arm)
  └─ 硬件错误门 (arm + hand)

Layer 3: XHand.send_action() 内部门       XHand.send_action()
  ├─ Delta 跳变限制 (E3)                    └─ 关节限位裁剪
  ├─ 死区节流
  └─ 关节限位裁剪

Layer 2: 板上错误监控                     (无)
  └─ commboard/jointboard/tipboard
     → 自动 error_state

Layer 1: 硬件级保护                       硬件级保护
  └─ tor_max=320mA (firmware)              └─ tor_max=400mA (firmware)
```

### 4.3 裁决

**DexMani 压倒性更优。** LeFranX 只有 1 层安全（关节限位裁剪 + firmware tor_max），缺失 13/15 项安全机制。最严重的缺陷：

1. **E-stop 未实现**: `XHand.stop()` 是仅写日志的 stub。在紧急情况下，LeFranX 无法使 XHand 停止加力。
2. **NaN/Inf 无防护**: `send_action` 会直接将非有限值传入 SDK，可能使手指进入不可预知的状态。
3. **无触觉力门**: 无法感知手指是否在施加过大力量——可能损坏物体或自身。
4. **无 Delta 限制**: 单步命令变化没有上限，重定向异常值可能触发危险运动。

**量化**: DexMani 有 15 项安全机制，LeFranX 有 2 项。安全覆盖率为 **13% vs 100%**。

### 4.4 按危害等级排序的安全机制

| 等级 | 机制 | 系统 | 危害程度 | 说明 |
|------|------|------|----------|------|
| 1 | **Hand E-stop** | DexMani D11 | **致命** | 唯一能使手部真正零扭矩的系统。LeFranX E5 未实现。 |
| 2 | **Arm E-stop（双路径）** | DexMani B11 | **致命** | set_state(4) ≤1 tick + 回退短连接。 |
| 3 | **力矩门** | DexMani A4 | **致命** | 发送前拒绝超扭矩命令。LeFranX 无等效。 |
| 4 | **硬件错误预门** | DexMani A1 | **致命** | arm/hand 报错时阻断所有命令。LeFranX 无预发送检查。 |
| 5 | **动作 NaN 门** | DexMani A3 | **致命** | 防止 NaN/Inf 到达 firmware。LeFranX 无等效。 |
| 6 | **tor_max 电流限制** | 两者（DexMani 320mA；LeFranX 400mA） | **致命** | 固件级过流保护。 |
| 7 | **可恢复错误区分** | DexMani B7 | **高** | 区分自碰撞/超速（瞬态）与硬故障（停止）。 |
| 8 | **逐步 delta 限制（臂）** | DexMani B1 | **高** | 0.3 rad 上限。 |
| 9 | **逐步 delta 限制（手）** | DexMani D2 | **高** | 手部关节 0.3 rad 上限。LeFranX 无等效。 |
| 10 | **关节限位裁剪（手）** | 两者（DexMani D1；LeFranX E1） | **高** | 均有效。 |
| 11 | **目标超时 + Hold** | DexMani B6 | **中** | 策略进程挂死时防止臂追逐过期目标。 |
| 12 | **触觉力门** | DexMani A6 | **中** | 手指力 >30N 时拒绝命令。LeFranX 无触觉。 |
| 13 | **跟踪误差监控** | DexMani B10 | **中** | 速度自适应阈值。被动。 |
| 14 | **电路板错误传播** | DexMani D7 | **中** | 每关节硬件板错误 → error_state。 |
| 15 | **连续错误断路器** | DexMani D8/D9 | **中** | 追踪持续通信故障；支持完全重连。 |
| 16 | **碰撞检查回零** | DexMani C6/C7/C8 | **中** | 三级，安全权衡逐步升级。LeFranX 无碰撞检查。 |
| 17 | **模式漂移监控** | DexMani B9 | **低** | Mode 6 脱落告警。 |
| 18 | **速度 NaN 门** | DexMani A5 | **低** | 速度数据损坏时拒绝。 |
| 19 | **臂连接门** | DexMani A2 | **低** | 臂断开时拒绝。 |
| 20 | **死区节流** | DexMani D4 | **低** | 减少 RS485 拥塞。 |
| 21 | **触觉接触检测** | DexMani D12 | **低** | 仅咨询，不门控命令。 |
| 22 | **工作空间安全** | DexMani C1 | **低** | 仅在初始化时。 |

### 4.5 LeFranX 安全缺陷明细

| # | 缺陷 | 风险 |
|---|------|------|
| 1 | `stop()` = `logger.info("NOT IMPLEMENTED")` | **致命**: 无法紧急停止手部 |
| 2 | 无 `validate_action` | **致命**: 每次 tick 原始策略输出直达硬件，零中间检查 |
| 3 | NaN 通过 `np.clip` 传播（`_apply_safety_limits`） | **高**: `np.clip(NaN, min, max)` = NaN，未被过滤 |
| 4 | 已知错误过滤在**读取**时返回 None | **中**: 传感器 CRC 瞬态错误导致 12 个关节位置全部丢失 |
| 5 | 已知错误过滤在**发送**时返回 True | **中**: 调用方认为成功，但命令可能未被正确执行 |
| 6 | `default_tor_max=400` vs `max_torque=300` 不一致 | **中**: 配置歧义——硬件实际使用 400mA |
| 7 | 无电路板错误监控 | **中**: 硬件故障静默传播 |
| 8 | 无连接丢失检测 | **中**: 手部离线时继续发送命令 |
| 9 | `emergency_stop_both` 仅在同步模式生效 | **低**: 非同步模式异常不触发 stop |
| 10 | 无目标超时 | **低**: 策略挂死后手部永远保持最后命令 |

---

## 5. 触觉感知

### 5.1 对比

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| Raw force 阵列 | ❌ | ✅ (5,120,3) float64, N |
| Force sum per finger | ❌ | ✅ (5,3) float64, N |
| Contact detection | ❌ | ✅ (5,) bool, L2 > 1.0N |
| 单位转换 | N/A | SDK raw int ÷ 10 = N |
| 传感器复位 | ❌ | ✅ connect 时 5 传感器 |
| 传感器 ID 校验 | ❌ | ✅ `_SENSOR_FINGER_IDS` 映射验证 |
| 力门（安全） | ❌ | ✅ 30N/指，fail-open on bad data |
| SDK stdout 抑制 | ❌ | ✅ fd 1 → /dev/null（reset_sensor 噪声） |

### 5.2 裁决

**DexMani 具有触觉感知，LeFranX 完全没有。** 触觉是 DexMani 的核心差异化优势，支持：

- **基于力的抓握控制**: 策略可以学习力调节
- **接触检测**: 知道手指何时触碰物体（`tactile_contact` 布尔）
- **安全门**: 防止过大接触力（30N/指）
- **滑移检测（潜力）**: 120 点空间分辨率可检测微滑移

注意：DexMani 的触觉数据经历了 **÷10 单位转换修复**（原始 SDK 整数 ÷ 10 → 牛顿）。LeFranX 甚至不读取触觉字段。

---

## 6. 手部重定向（VR→XHand）

### 6.1 管道对比

```
LeFranX 管道:                       DexMani 管道:
  Quest VR landmarks (21×3)           Quest VR landmarks (21×3)
  │                                   │
  ├─ TCP → C++ VRMessageRouter        ├─ VR SDK 直接
  │                                   │
  ├─ VRRouterManager (singleton)      ├─ TeleopPipeline Stage
  │                                   │
  ├─ VRHandDetectorAdapter            ├─ XHandRetargeter.retarget()
  │  ├─ ×1.05 scale                   │  ├─ NaN/Inf guard
  │  ├─ X mirror (right hand)         │  ├─ estimate_frame_from_hand_points
  │  ├─ wrist-center                   │  ├─ @ OPERATOR2MANO_RIGHT
  │  ├─ estimate_frame (SVD)          │  ├─ adaptive_retargeting_THUMB ← 独有
  │  ├─ @ OPERATOR2MANO_RIGHT         │  ├─ adaptive_retargeting_xhand (pinky)
  │  ├─ adaptive_retargeting_xhand    │  └─ _build_ref_value
  │  └─ → MANO landmarks              │
  │                                   ├─ SeqRetargeting.retarget()
  ├─ DexPilot retarget (SLSQP)        │  └─ XHandDexPilotOptimizer  ← 自定义子类
  │  └─ standard optimizer            │     └─ wrist_weight=2.0 (vs 15)
  │                                   │
  ├─ EMA smoothing (alpha=0.3)        ├─ LPFilter (alpha=0.6)
  │                                   │
  ├─ Joint reorder + index negate     ├─ Joint reorder (no negate)
  │                                   │
  └─ → 12 joint commands              └─ → 12 joint commands
```

### 6.2 自适应手指缩放

| 参数 | LeFranX (pinky) | DexMani (pinky) | DexMani (thumb) |
|------|-----------------|-----------------|----------------|
| Min extension | 0.030 m | 0.028 m（P2，真实遥操作数据） | 0.105 m（P3，手腕→拇指尖） |
| Max extension | 0.100 m | 0.074 m（P98，真实遥操作数据） | 0.140 m（P97） |
| Base scale | 1.20 | 1.15 | 1.02 |
| Max scale | 2.20 | 2.40 | 1.35 |
| 标定方法 | 硬编码（可能针对 LeFranX 自身的 MANO 空间） | 3,431 帧网格搜索（`20260701_161732`） | XHand 拇指 FK：机器人 0.161m vs MANO 0.131m |
| 算法 | 运动链递进缩放（MCP→PIP→DIP→TIP） | 相同算法（从 LeFranX 移植） | 仅从手腕径向缩放拇指尖 |

> **注意**: 粉色参数差异巨大，因为两个系统产生的是**不同的 MANO 空间坐标**。LeFranX 在标记空间中应用了 1.05 倍缩放，使粉色在完全伸展时看起来更长（0.10m vs 0.074m）。

### 6.3 优化器与后处理

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 优化器类 | 标准 `DexPilotOptimizer` | `XHandDexPilotOptimizer`（自定义子类） |
| 腕部→指尖权重 | ~15（`len_proj + n_fingers`，硬编码） | 可配置（当前 YAML：`wrist_weight=3.0`→15；**可调低**） |
| 投射距离 | 0.03（默认） | 0.03（YAML） |
| 逃离距离 | 0.05（默认，有迟滞） | 0.03（YAML，**粘性投射**——无迟滞） |
| 关节重排序 | 是（`retargeting_to_xhand`） | 是（`retargeted_joint_order`） |
| 食指弯曲取反 | **是**（`qpos[3] = -qpos[3]`） | **否** |
| 平滑滤波器 | EMA, α=0.3（后优化）+ 可能 LPFilter(α=0.1)（内部） = **双重滤波** | LPFilter, α=0.6（内部 SeqRetargeting） |
| 有效时间常数 | ~110ms（EMA 0.3） | ~57ms（LPF 0.6） |
| 故障回退 | 3级 → hold-last（数据丢失）→ hold-last（处理失败）→ **返回原位**（异常） | 1级：始终 **hold-last**（所有故障模式） |
| NaN 防护 | 无 | ✅ `retarget()` 入口处 |

### 6.4 坐标变换链差异（重要）

对于**相同的 VR 手部姿态**，两个系统产生**不同的 MANO 空间坐标**，原因在于预处理管道不同：

```
DexMani 变换链:                   LeFranX 变换链:
  raw landmarks (21,3)              raw landmarks (21,3)
  ↓                                ↓
  (无缩放)                          ×1.05 缩放
  ↓                                ↓
  (无镜像)                          X 轴镜像（右手）
  ↓                                ↓
  SVD on raw wrist/mcp coords      wrist → origin
  ↓                                ↓
  @ OPERATOR2MANO_RIGHT            SVD on centered coords
  ↓                                ↓
  = MANO landmarks                  @ OPERATOR2MANO_RIGHT
                                   ↓
                                   = MANO landmarks
```

**三个累积差异**: (1) LeFranX 的 1.05 倍标记缩放，(2) X 轴镜像，(3) SVD 帧估计前的手腕居中。这解释了为何两个系统对粉色标度参数进行了不同的标定——它们对不同的 MANO 空间输入分布进行操作。

### 6.5 裁决

**DexPilot + SLSQP 的核心重定向算法相同**，但 **DexMani 在五个关键方面有所改进**:

1. **拇指自适应缩放**（独有）: 补偿 XHand 拇指机械长度（~0.161m vs MANO ~0.131m，+23%），使 rot2 在中性位姿时趋近零。
2. **优化器权重重平衡**: 降低腕部→指尖权重（15→10），让指间向量有更大发言权，减少远端关节的长度补偿效应。
3. **更好的参数标定**: 小指参数通过 500+ 帧真机数据的网格搜索标定，而非 LeFranX 的硬编码值。

LeFranX 在小指自适应缩放上**先发**（DexMani 移植自 LeFranX），且其 **VRRouterManager singleton** 架构在 arm+hand 共享 VR 源时更优雅。

---

## 7. 数据录制

### 7.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 录制框架 | LeRobot（HuggingFace 标准） | 自研 HDF5（Schema v8-10） |
| 录制帧率 | 30 Hz | 16 Hz |
| 手部位置 | ✅（12 维） | ✅（12 维） |
| 手部速度 | ❌ | ✅（12 维，via SHM） |
| 手部力矩/电流 | ✅（12 维，标记为 torque） | ✅（12 维，明确定义为 current） |
| 手部触觉 raw | ❌ | ✅（5×120×3） |
| 手部触觉 sum | ❌ | ✅（5×3） |
| 手部触觉 contact | ❌ | ✅（5×bool） |
| 指尖位置 | ❌ | ✅（5×3, 主进程 FK） |
| 臂-手对齐方式 | LeRobot 自动时间戳 | TimestampAlignedBuffer 精确网格对齐 |
| 数据格式 | Parquet + 视频 | HDF5 |
| 异步写入 | LeRobot image_writer_threads | EpisodeRecorder async buffer |
| 数据验证 | 无 | check_episode_health.py |
| 导出格式 | LeRobot 原生 | export_hdf5_to_zarr.py |
| 可视化 | visualize_dataset.py | visualize_episode.py |

### 7.2 裁决

**DexMani 录制更完整。** 核心差异：
- LeFranX 录制约 30 个 hand 字段 → DexMani 录制约 **600+ 个 hand 相关字段**（主要是因为 (5,120,3) 触觉阵列 = 1800 个值）
- LeFranX 缺少手部速度、触觉和指尖位置
- DexMani 的 `TimestampAlignedBuffer` 提供精确的臂-手 16Hz 网格对齐

LeFranX 在 LeRobot 生态兼容性方面有优势（HuggingFace 数据集可直接使用）。

---

## 8. 进程与线程模型

### 8.1 架构对比

```
LeFranX (单进程):                   DexMani (多进程隔离):

┌─── Main Process ──────────┐      ┌─── Main Process (16Hz) ───┐
│                            │      │  HandSHMFaçade             │
│  FrankaFERXHand            │      │  ├─ E3 delta clip (main)   │
│  ├─ FrankaFER (arm)        │      │  └─ check_echo (F1)       │
│  │   └─ TCP → C++ server   │      │  hand_cmd ring             │
│  └─ XHand (hand)           │      │  ────────────────────►     │
│      └─ RS485 直连          │      │                   ┌───────┴──────────┐
│                            │      │  hand_state ring   │ HandControlProcess│
│  (臂和手在同一进程中，       │      │  ◄─────────────────│ 30Hz, fork 隔离   │
│   可能互相阻塞)              │      │                   │ ├─ XHand (stateless)│
└────────────────────────────┘      │                   │ ├─ state publish   │
                                    │                   │ ├─ macro executor  │
                                    │                   │ └─ watchdog        │
                                    └───────────────────┴───────────────────┘
```

### 8.2 关键差异

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| XHand 运行位置 | 主进程内 | 独立 fork 子进程 |
| 崩溃隔离 | 无（XHand 崩溃 → 主进程崩溃） | 强（子进程崩溃 → 主进程降级到 arm-only） |
| 控制频率 | 随主循环变化 | 固定 30Hz（子进程独立） |
| 命令路由 | 直接调用 `xhand.send_action()` | SHM ring buffer（SeqlockRingBuffer） |
| 状态路由 | 直接调用 `xhand.get_state()` | SHM ring buffer + 回显验证 |
| E-stop 抢占 | 无 | 跨进程 Event + macro thread lock 分离 |
| 孤儿保护 | 无 | `orphan_exit_s` 预算（主进程死 → 子进程保持位置并退出） |
| 命令陈旧检测 | 无 | `cmd_stale_hold_s`（无新命令 → hold，NEVER detorque） |
| 生产者 ID 验证 | 无 | 有（`producer_id`，拒绝未知来源命令） |
| 过程间 RPC | 无 | macro RPC（RESET/STOP/CLEAR_ERROR/SEND_TRAJECTORY） |

### 8.3 裁决

**DexMani 的多进程架构从根本上更安全，但复杂度更高。** 关键设计决策：

- **"Hold, NEVER detorque"**: 子进程在任何异常情况下（主进程死、SIGINT、命令环陈旧）都保持位置而不卸力。固件的 mode 3 位置伺服在没有命令刷新的情况下保持位置（A4 假设）。
- **回显验证（Echo Verification）**: `HandSHMFaçade.check_echo()` 验证子进程实际发送的命令与主进程请求的命令是否一致（容差 1e-3 rad），并在不匹配时重新同步基线。
- **宏 RPC**: 重置、停止、轨迹执行等长时操作在子进程中执行，通过 RpcServer/RpcClient 通信，不阻塞主进程。

LeFranX 的单进程模型更简单，适合研究原型。但其臂/手耦合意味着任一组件的崩溃都会导致整个系统失效。

---

## 9. 配置与调优

### 9.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 配置加载 | `@dataclass` + `RobotConfig.register_subclass` | `@dataclass` + `FromDictMixin`（YAML/JSON 热加载） |
| 关限 | 硬编码度数列表 | per-joint `qpos_min`/`qpos_max` 数组（弧度） |
| 关节限位来源 | 手写硬编码 | URDF 派生 |
| Kp 默认值 | 80 | 100 |
| tor_max 默认值 | 400 mA | 320 mA（硬件规格） |
| 逐关节 Kp/Ki/Kd | ❌ | ✅ `kp_per_joint`/`ki_per_joint`/`kd_per_joint` |
| 逐关节 max_delta_rad | ❌ | ✅ 标量或 (12,) 数组 |
| 触觉阈值配置 | N/A | ✅ `tactile_contact_threshold` |
| 连接重试配置 | N/A | ✅ `open_serial_retries` + `open_serial_retry_delay_s` |
| state_init 配置 | N/A | ✅ `init_state_read_attempts` + `init_state_read_interval` |
| clip_report_tolerance | N/A | ✅ 0.01 rad |
| 未使用字段 | `max_torque`, `control_frequency`, `timeout` | 无（所有字段均被使用） |
| Default protocol | RS485 | EtherCAT |

### 9.2 裁决

**DexMani 配置更完整、更灵活。** 具体提升：
- 逐关节 PID 增益（远端关节可独立调优）
- 逐关节 delta 限制（不同关节可设不同安全上限）
- 基于 URDF 的精确关节限位（vs LeFranX 手写度数→手动转弧度）
- 所有配置字段都被实际使用（LeFranX 有 3 个声明但未使用的字段）

LeFranX 在 LeRobot 注册模式（`@RobotConfig.register_subclass` + factory）方面在生态兼容性上有优势。

---

## 10. 集成模式

### 10.1 对比表

| 方面 | LeFranX | DexMani |
|------|---------|---------|
| 基类 | `Robot`（LeRobot 抽象类） | `ConnectionStateMixin`（自研） |
| 动作/观测格式 | `Dict[str, Any]`（LeRobot 标准） | `np.ndarray`（自研，更高效） |
| 工厂模式 | `@RobotConfig.register_subclass` + `make_robot_from_config` | `make_hand_servo()` 工厂 |
| 多机器人组合 | Composition（`FrankaFERXHand`） | Process Isolation（arm/hand 独立进程） |
| 命名空间 | `arm_`/`hand_` 前缀 | 各组件自带 state dict |
| LeRobot 兼容 | ✅ 原生 | ❌ 需适配 |
| 生态互操作 | HuggingFace 数据集/模型 zoo | 自研，更深度优化 |
| 测试支持 | Stub mode | Stub mode + `hand_factory` 注入 |

### 10.2 裁决

**各有千秋。** LeFranX 的 LeRobot 原生成分使其可直接使用 HuggingFace 的训练/评估基础设施。DexMani 的 numpy-based API 性能更优（无 dict 编解码开销）且与 SHM 接口自然对齐。

---

## 11. 总体评估与建议

### 11.1 总体裁决

| 维度 | 优胜方 | 优势程度 |
|------|--------|---------|
| 硬件连接与初始化 | **DexMani** | +5 层安全检查 |
| 状态读取 | **DexMani** | +8 个读取字段 + 触觉 |
| 命令发送 | **DexMani** | +4 层安全防护 |
| 安全架构 | **DexMani** | 15 vs 2 项安全机制 |
| 触觉感知 | **DexMani** | LeFranX 完全没有 |
| 手部重定向 | **DexMani** | +拇指缩放 + 权重重平衡 |
| 数据录制 | **DexMani** | 触觉 + 速度 + 精确对齐 |
| 进程模型 | **DexMani** | 崩溃隔离 + 回显验证 |
| 配置 | **DexMani** | 逐关节调优 + 热加载 |
| 集成模式 | **平局** | LeRobot vs numpy，场景相关 |

### 11.2 DexMani 应该向 LeFranX 学习的

1. **Stub 模式普及**: 已移植。作为连接生命周期的一等公民已实现。
2. **VR ADB 自动管理**: 未移植。在 VR 遥操作入口脚本中可能有用。
3. **已知传感器错误模式**: 已移植（`_KNOWN_SENSOR_ERROR_PATTERNS`）。
4. **自适应小指重定向**: 已移植（`adaptive_retargeting_xhand`）。
5. **LeRobot 兼容层**: 未实现。对生态互操作有价值，但非优先。

### 11.3 LeFranX 的致命缺陷（DexMani 不应退化）

| 缺陷 | 风险 |
|------|------|
| `stop()` 未实现 | 紧急情况无法停止 XHand |
| 无 NaN/Inf 防护 | 坏数据可导致不可预知行为 |
| 无触觉力门 | 无法感知/阻止过大接触力 |
| 无 Delta 限制 | 单步大跳变可损坏硬件 |
| 无电路板错误监控 | 硬件故障静默传播 |
| 无连接丢失检测 | 手离线时继续发送命令 |
| 单进程无隔离 | 手驱动崩溃 → 整个系统崩溃 |

### 11.4 量化总结

```
                      LeFranX    DexMani    比率
安全机制数量              2         15       1:7.5
状态读取字段              4        11+       1:2.8+
触觉数据量                0      ~1800个值  0:1800
send_action 安全层        1          5       1:5
connect 容错层             1          6       1:6
代码行数 (驱动)          ~387     ~1060      1:2.7
代码行数 (手重定向)        ~308      ~483      1:1.6
代码行数 (进程隔离)          0     ~1261      0:1261
E-stop 实现               ❌          ✅
```

**结论**: DexMani 的 XHand 子系统在所有安全、触觉和鲁棒性维度上都**显著优于** LeFranX。LeFranX 的优势主要在生态集成（LeRobot）和架构简洁性方面。两个系统的重定向管道共享核心算法，DexMani 在此基础上做了基于真实数据的精心改进。
