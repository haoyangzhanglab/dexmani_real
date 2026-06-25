# 机械臂内外环控制架构分析

> 基于代码事实校验，涵盖 `TeleopController`（外环 @ 50Hz）和 `XArm7 PID`（内环 @ 250Hz）的完整控制链路。
> 以 BunnyVisionPro (`xarm7_ability.py` ~300 行 teleop ~260 行) 为参照基准。

---

## 目录

1. [架构总览](#1-架构总览)
2. [与 BunnyVisionPro 对比](#2-与-bunnyvisionpro-对比)
3. [外环 — 50Hz 感知-规划-指令流水线](#3-外环--50hz-感知-规划-指令流水线)
4. [内环 — 250Hz PID 位置→速度控制](#4-内环--250hz-pid-位置速度控制)
5. [内外环接口解耦](#5-内外环接口解耦)
6. [安全纵深防御](#6-安全纵深防御)
7. [端到端延迟分析](#7-端到端延迟分析)
8. [关键参数速查表](#8-关键参数速查表)
9. [核心设计决策](#9-核心设计决策)
10. [简化演进历史](#10-简化演进历史)

---

## 1. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            OUTER LOOP (外环)                                 │
│                         TeleopController @ 50 Hz                            │
│                        周期: 20ms | 线程: 主线程                              │
│                                                                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────────────┐  │
│  │ VR 追踪   │───→│ 位姿映射  │───→│  IK 求解   │───→│   EMA 平滑           │  │
│  │ (120Hz)  │    │ ArmMapper│    │ DLS→MPlib │    │ 固定 α (默认 0.95)   │  │
│  └──────────┘    └──────────┘    └───────────┘    └──────────┬───────────┘  │
│                                                              │              │
│  ┌──────────────────────────────────────────────────────────┘              │
│  │  动作安全层 (管道内)                                                      │
│  │  ├─ 工作空间夹持 + 重解 IK（ref: ManiUniCon _clip_action_to_bounds）      │
│  │  └─ 速度限制步长 (max_qvel × dt 等比缩放，在驱动层执行)                    │
│  │                                                                          │
│  │  发送前安全门                                                             │
│  │  ├─ validate_action() 7 项检查                                           │
│  │  └─ 指尖桌面 FK 碰撞检测                                                  │
│  └────────────────────────────────────────────────────────────────────────── │
│                                    │                                         │
│                         RobotAction                                         │
│                    arm_qpos_cmd(7) / hand_qpos_cmd(12)                      │
│                                    │                                         │
│  ══════════════════════════════════╪══════════════════════════════════════  │
│                                    │  threading.Lock (临界区 < 1μs)          │
│                                    ▼                                         │
│                            INNER LOOP (内环)                                 │
│                         XArm7 PID @ 250 Hz                                  │
│                      周期: 4ms | 线程: 守护线程                               │
│                                                                              │
│  ┌──────────────────────┐    ┌───────────┐    ┌──────────────────┐          │
│  │ PID 控制 (D-on-M)     │───→│ 速度裁剪   │───→│ vc_set_joint_    │          │
│  │ P + D + I (I=0 遥操作)│    │ bottleneck│    │ velocity()       │          │
│  │                       │    │ +soft-start│   │ → xArm7 硬件     │          │
│  └──────────────────────┘    └───────────┘    └──────────────────┘          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 关键时序关系

| 特性 | 外环 | 内环 |
|------|------|------|
| 频率 | 50 Hz | 250 Hz |
| 周期 | 20 ms | 4 ms |
| 线程 | 主线程 (`run()`) | 守护线程 (`_pid_loop_impl()`) |
| 同步方式 | `RateLimiter.wait()` 补偿计算耗时 | `RateLimiter.wait()` |
| 指令类型 | 位置目标 (`arm_qpos_cmd[7]`) | 速度指令 (`qvel[7]`) |
| 过采样比 | — | 5:1 (每个外环目标被内环消费 5 次) |

---

## 2. 与 BunnyVisionPro 对比

### 2.1 核心差异

| 维度 | BunnyVisionPro | DexMani |
|------|---------------|---------|
| 核心代码量 | `xarm7_ability.py` ~300 行 + teleop ~260 行 | `xarm7.py` ~1200 行 + `controller.py` ~1300 行 + `pipeline.py` ~400 行 |
| 内环 PID | D-on-error, Kp=10, Kd=0.5, Ki=0 | **D-on-measurement (F1)**, Kp=7, Kd=0.35, anti-windup |
| EMA | **无** | 固定 α=0.95，作用于 IK 输出（关节位置） |
| IK 策略 | 100 iter DLS, 固定 λ²=1e-5 | **100 iter DLS**（对齐 BVP）, 固定 λ²=1e-5 → MPlib 采样 → Hold |
| 速率控制 | `time.sleep(dt)` | `RateLimiter.wait()` 补偿式 |
| 安全层 | ~2 (速度裁剪 + 错误码检查) | **6 层纵深防御**（含已删除的跳变钳位和发散检测） |
| 数据录制 | dict append → HDF5 | 预录制缓冲 + HDF5 元数据 + 多相机 |

### 2.2 信号流对比

```
BunnyVisionPro (极简, ~4 阶段):
  VR ZMQ → TeleopClient → [lock] _arm_pos_target
    → 250Hz: PID(D-on-error) → clip_vel → vc_set_joint_velocity

DexMani (多层, ~8 阶段, 已精简):
  VR Tracker → ArmMapper → IK(100 iter DLS ↗ MPlib ↘ Hold) → workspace clamp+reIK
    → ✱EMA(joint space, α=0.95)✱ → validate_action(7)
      → desk_FK → send_action → [lock] → 250Hz: PID(F1 D-on-M) → clip_vel+soft_start
```

### 2.3 差异来源分析

DexMani 代码量更大，主要原因：

1. **安全纵深防御**：VR 追踪质量门、工作空间夹持+重解、指尖桌面 FK、跟踪发散检测（已删除）等，BVP 无等效机制
2. **完整遥操作生命周期**：预录制缓冲、SAVE_PROMPT 状态机、多相机录制、HDF5 元数据 — BVP 无数据录制
3. **MPlib 运动规划**：自主路径规划（`plan_path`, screw/RRT），遥操作热路径不调用但存在
4. **模块解耦**：`pipeline.py`（动作计算）、`controller.py`（状态机）、`xarm7.py`（驱动）三层分离；BVP 全部在 `xarm7_ability.py` 中

---

## 3. 外环 — 50Hz 感知-规划-指令流水线

### 3.1 主循环结构

```python
# controller.py: run()
while self.running:
    self._handle_keyboard()    # 非阻塞轮询键盘队列 → ControlSignal
    self._tick()               # 完整一帧
    self.limiter.wait()        # sleep(20ms - 计算耗时), 非 sleep(20ms)
```

**`RateLimiter.wait()` 补偿机制**：计算「目标周期 - 实际耗时」后精确睡眠，而非盲目 sleep 20ms。计算时间波动不会累积成频率漂移。

### 3.2 `_tick()` 完整流程

```
_tick() @ 50Hz
│
├─[Step 0] 状态守卫 (controller.py)
│   ├─ EMERGENCY_STOP     → return (冻结一切运动)
│   ├─ PAUSED             → clear_target() 自然减速 / hold 当前位置
│   └─ IDLE / SAVE_PROMPT → return
│
├─[Step 1] VR 帧读取
│   来源: ZMQ subscriber (远程 Quest @ 120Hz) / 本地 DummyTracker
│   内容: wrist_pos(3), wrist_quat_wxyz(4), landmarks(21,3), local_recv_ns
│
├─[Step 2] VR 追踪质量门 — 三层时效分级
│   ┌──────────────┬─────────────────────────────────────────────────┐
│   │ 帧年龄       │ 行为                                            │
│   ├──────────────┼─────────────────────────────────────────────────┤
│   │ < 0.1s       │ FRESH — 清除丢失计时，正常处理                   │
│   │ 0.1s ~ 0.5s  │ SOFT LOST — 开始计时，丢弃本帧(无 IK/无运动)    │
│   │ 0.5s ~ 1.0s  │ HARD LOST — 自然减速 (PID target=None→零速度)   │
│   │ > 1.0s       │ EMERGENCY — 紧急停止                            │
│   └──────────────┴─────────────────────────────────────────────────┘
│   恢复条件: 任意帧年龄 < 0.1s → _vr_lost_since = None
│
├─[Step 3] 机器人状态读取
│   state = robot.get_state()
│   → arm_qpos(7), arm_qvel(7), arm_tau(7), eef_pos(3), eef_quat_wxyz(4),
│     hand_qpos(12), hand_tactile_sum(5,3), fingertip_pos(5,3)
│
├─[Step H3] PID 线程存活检查 (每 50 帧 ~1s)
│   条件: PID 守护线程已死亡 → 紧急停止
│   目的: 防止线程静默死亡后手臂在过期速度指令上漂移
│
├─[Step 4] 动作计算 — pipeline.compute_action()
│   ┌─────────────────────────────────────────────────────────────┐
│   │                                                             │
│   │  [4a] ArmMapper.map(wrist → EEF pose)                       │
│   │     delta = wrist_pose ⊖ wrist_pose₀  (VR 帧空间)            │
│   │     delta_base = R_vr_to_base @ delta  (转到机器人基座标系)   │
│   │     target_eef = eef_pose₀ ⊕ delta_base                     │
│   │     ├─ pos_scale (默认 1.0): 位置缩放                        │
│   │     ├─ rot_scale (默认 1.0): 旋转角度缩放                    │
│   │     ├─ max_delta_rot_rad (默认 1.0 rad): 单帧旋转上限       │
│   │     └─ eef_delta_bounds: 增量位置硬边界                      │
│   │                                                             │
│   │  [4b] 固定 EMA 平滑 (pipeline.py)                            │
│   │     ┌────────────────┬────────────────────────────────────┐  │
│   │     │ alpha          │ 效果                               │  │
│   │     ├────────────────┼────────────────────────────────────┤  │
│   │     │ 1.0 (Pipeline) │ 无平滑, 直接使用原始 IK 输出        │  │
│   │     │ 0.95 (Controller)│ ~1帧时间常数, ~2ms 滞后, 轻微滤波 │  │
│   │     │ 0.75           │ ~3帧时间常数, ~7ms 滞后, 手颤滤波   │  │
│   │     │ 0.5            │ 重滤波, 适合高精度静态操作          │  │
│   │     └────────────────┴────────────────────────────────────┘  │
│   │     EMA 参考值使用 _last_raw_arm (速度限制前原始 IK 输出),   │
│   │     而非 _last_arm_cmd → 防止速度限制滞后在 EMA 中累积 (F2) │
│   │                                                             │
│   │  [4c] IK 求解 — 迭代 DLS（对齐 BunnyVisionPro）             │
│   │     ┌──────────────────────────────────────────────────────┐│
│   │     │ Tier 1: 迭代 DLS 微分 IK (确定性, 主要路径)          ││
│   │     │   • 100 次迭代, 固定阻尼 λ²=1e-5                     ││
│   │     │   • 每迭代: FK → Jacobian → DLS → integrate(×0.05) ││
│   │     │   • 收敛条件: ‖error‖ < 1e-3                         ││
│   │     │   • 自适应阻尼可选 (默认关闭, 对齐 BVP 固定阻尼)      ││
│   │     │   • 最终 delta 步长限制: pos ≤ 0.02m, rot ≤ 5°      ││
│   │     │   • 迭代内无步长限制 — 仅最终结果限幅                  ││
│   │     │   → 成功时验证 FK 残留: pos ≤ 0.008m, rot ≤ 0.08rad││
│   │     │                                                    ││
│   │     │ Tier 2: MPlib 位置 IK (随机, 后备)                   ││
│   │     │   • 仅 DLS 失败时调用                                ││
│   │     │   • 种子优先级: prev_cmd(n=3) → current_qpos(n=2)  ││
│   │     │   • 过滤: 位姿误差 + 关节跳变 (≤90°) + 自碰撞       ││
│   │     │   • fast_accept: prev_cmd 距离硬件 < 15° → 立即返回 ││
│   │     │   • early_exit: 优秀质量 → 提前终止                 ││
│   │     │   • 追踪硬件最接近候选 (避免肘关节翻转)              ││
│   │     │                                                    ││
│   │     │ Tier 3: Hold — 返回 prev_cmd (held=True)            ││
│   │     └──────────────────────────────────────────────────────┘│
│   │                                                             │
│   │  [4d] 工作空间检查 + 夹持 + 重解 IK (ref: ManiUniCon)       │
│   │     条件: EEF 位置超出 [0.28,0.72]×[-0.45,0.45]×[0.05,0.5]│
│   │     处理: np.clip 到边界 → 重解 IK                         │
│   │     重解失败: 保持上一帧命令                                  │
│   │                                                             │
│   │  [4e] Hand 重定向 — landmarks(21,3) → hand_qpos(12)          │
│   │     landmarks @ wrist_rot @ OP2MANO_RIGHT → MANO 骨骼       │
│   │     → dex-retargeting 优化器 → 12-DOF                       │
│   │     手部平滑由 retargeter 内置 low_pass_alpha 处理           │
│   └─────────────────────────────────────────────────────────────┘
│
├─[Step 5] 录制帧
│   ├─ 预录制缓冲: 始终缓冲, 按下 R 时捕获前 N 秒上下文
│   ├─ 实际录制: self.recording == True 时写 HDF5
│   └─ 多相机/单相机兼容路径
│
├─[Step 7] 发送前安全门
│   ├─ validate_action() — 7 项顺序检查 (fail-fast):
│   │   (1) 硬件错误状态    (5) 手部温度 < 70°C
│   │   (2) 手臂连接状态    (6) 手部通信错误
│   │   (3) 手臂力矩 < 限值  (7) 工作空间边界
│   │   (4) 手部电流 < 500mA
│   │   error/not_connected → 急停; 其他 → hold 上一帧
│   │
│   ├─ 指尖桌面安全 (FK Z 轴检查):
│   │   计算指令 qpos 下指尖世界 Z 坐标
│   │   阈值为 table_z + hand_safe_margin = 0.0 + 0.03 = 0.03m
│   │   违规 → hold 上一帧
│   │
│   └─ robot.send_action(action) → arm.send_action + hand.send_action
│       └─ XArm7._limit_joint_step() — 驱动层速度限制 (bottleneck scaling + soft-start)
│
├─[Step 8] 周期性状态日志 (每 2s)
│
└─[Step 9] 循环超限检测
    阈值: tick 耗时 > 目标周期 × 150% (= 30ms)
```

### 3.3 外环状态机

```
                    B(开始)
    ┌──────────────────────────────────────────────┐
    │                                              │
    ▼                                              │
  IDLE ──────B(开始)──────→ TELEOP ──R(录制)──→ RECORDING*
    ▲                        │  ▲                    │
    │                        │  │                    │
    │                  H(归位)│  │C(暂停/恢复)         │
    │                        │  │                    │
    │                        ▼  │                    │
    ├────────H(归位)───── PAUSED                     │
    │                                                 │
    │     S(停止录制) ←───────────────────────────────┘
    │         │
    │         ▼
    │    SAVE_PROMPT ──S(保存)→ IDLE
    │         │
    │         └──Q(丢弃)→ IDLE
    │
    └── 任意状态 ──ESC/超时/急停──→ EMERGENCY_STOP
                                        │
                                   Q → 退出程序
                                   H → 恢复+归位
```

\* RECORDING 不是独立状态，是 TELEOP 状态下的 `self.recording == True` 子模式。

### 3.4 外环关键状态追踪

| 变量 | 含义 | 写入时机 | 用途 |
|------|------|----------|------|
| `_last_raw_arm` | 速度限制**前**的原始 IK 输出 | Step 4: IK 成功后 | EMA 平滑参考值 |
| `_last_arm_cmd` | 速度限制**后**的实际发送命令 | 速度限制后 | 速度限制步长参考 |
| `_last_hand_cmd` | 上次发送的手部命令 | retarget 成功后 | fallback |
| `_last_good_arm` | 最后一次通过安全门的手臂位置 | send_action 成功后 | validate_action 失败时的 fallback |
| `_last_good_hand` | 最后一次通过安全门的手部位置 | send_action 成功后 | validate_action 失败时的 fallback |
| `_ik_miss_count` | 连续 IK 失败计数 | Step 4 | 连续 ≥3 帧报警, 成功清零 |

**F2 设计精髓**: `_last_raw_arm` 与 `_last_arm_cmd` 分离。EMA 参考 `_last_raw_arm`（原始意图轨迹），速度限制器参考 `_last_arm_cmd`（实际发送轨迹）。若 EMA 参考 `_last_arm_cmd`，速度限制滞后会通过 EMA 递归累积，产生额外延迟。

---

## 4. 内环 — 250Hz PID 位置→速度控制

### 4.1 内环结构

```python
# xarm7.py: _pid_loop_impl() — 守护线程
while not self._arm_should_stop.is_set():
    rate_limiter.wait()  # 精确 4ms 周期

    # [模式门] mode != 4 → 跳过 (防止在 SDK 模式切换时发送速度指令)
    if not self._velocity_control_active or self.arm.mode != 4:
        continue

    # [读取目标] 带锁读取外环写入的 _arm_pos_target
    with self._arm_lock:
        target = self._arm_pos_target

    # [None-sentinel] 外环请求自然减速 (PAUSED/软减速)
    if target is None:
        vc_set_joint_velocity(zeros(7))
        continue

    # [NaN 守卫] 防御 NaN 通过 IEEE 754 语义绕过速度裁剪
    if not all(isfinite(target)):
        vc_set_joint_velocity(zeros(7))
        continue

    # [读取硬件] arm.get_joint_states()
    code, xarm_state = arm.get_joint_states(is_radian=True)

    # [PID] 关节空间位置误差 → 速度
    error = target[:7] - arm_current_qpos[:7]
    qvel = pid.control(error, dt, max_vel=pid_max_vel,
                       current=arm_current_qpos)  # F1: D-on-measurement

    # [速度裁剪] bottleneck 等比缩放 + soft-start 斜坡
    safe_qvel = _clip_arm_velocity(qvel)

    # [发送硬件]
    vc_set_joint_velocity(safe_qvel.tolist())
```

### 4.2 F1: D-on-Measurement — 消除微分冲击

| 方案 | 微分项 | 目标阶跃时的行为 |
|------|--------|-----------------|
| 传统 PID | Kd × d(error)/dt | error 瞬时跳变 → 微分项 → ∞ (冲击) |
| **F1 D-on-M** | Kd × -d(measurement)/dt | measurement 连续 → 微分项不变 (无冲击) |

```python
# F1 实现: 将当前实际位置传入 PID 控制器
qvel = pid.control(error, dt, max_vel, current=arm_current_qpos)
#                                               ↑ 微分对象是实际位置, 不是误差
```

### 4.3 速度裁剪 — Soft-Start 斜坡

`_clip_arm_velocity()` 在每次 `connect()/reset()/clear_error()/reset_soft_start()` 后重置:

```
pid_convergence_threshold_rad > 0 (两阶段模式):
  Phase 1: 30% 硬上限 → 直到所有关节 |error| < threshold
  Phase 2: 0% → 100% 线性斜坡 (duration: 0.3s)

pid_convergence_threshold_rad = 0 (简化模式, 默认):
  单一 0% → 100% 线性斜坡 (duration: 0.3s)
  若 soft_start_ramp_duration = 0 → 跳过斜坡, 直接全速

斜坡细节:
  elapsed = now - _vel_ramp_start
  ramp_progress = min(elapsed / 0.3, 1.0)
  effective_limit = pid_max_vel × ramp_progress
```

### 4.4 内环 NaN 三层防御

| 层 | 位置 | 机制 |
|----|------|------|
| H2-L1 | `clear_error()` | NaN 目标拒绝入锁 |
| H2-L2 | PID 内环 | NaN 目标 → 发送零速度 |
| H2-L3 | `_clip_arm_velocity()` | NaN 速度 → 返回零 (IEEE 754: `NaN > 1.0 → False` 会绕过裁剪) |

### 4.5 PID 控制器参数

控制律（`PIDController.control()`, `xarm7.py`）：

```
qvel = Kp × err  -  Kd × (q_current - q_prev) / dt  +  Ki × cum_err
         ↑ P 项         ↑ D-on-measurement (F1)           ↑ I 项 (遥操作恒为 0)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pid_kp` | `[7.0] × 7` | 比例增益, 统一值保证笛卡尔轨迹形状 |
| `pid_kd` | `[0.35] × 7` (= Kp/20) | 微分增益, D-on-measurement 避免目标阶跃冲击 |
| `pid_ki` | `[0.0] × 7` | 积分增益, 遥操作永久关闭防止 windup |
| `pid_max_vel` | `[1.885] × 7` rad/s (108°/s) | PID 速度输出上限, 60% × max_qvel |
| `max_qvel` | `[π] × 7` rad/s (180°/s) | 硬件最大速度, 全局统一 |
| `inner_control_dt` | 0.004 s (250 Hz) | 内环离散化步长 |
| `pid_convergence_threshold_rad` | 0.0 | 两阶段软启动 Phase 1 收敛判定 (默认关闭) |
| `soft_start_ramp_duration` | 0.3 s | 冷启动速度斜坡时间 |
| Anti-windup | `‖Ki × cum_err‖ ≤ pid_max_vel` | 积分限幅 (PIDController 内建, 仅 Ki≠0 时生效) |

---

## 5. 内外环接口解耦

### 5.1 通信协议

```
外环 (50Hz, 主线程)                   内环 (250Hz, 守护线程)
══════════════════                    ══════════════════════

send_action(q_cmd):                   while not stopped:
  with _arm_lock:                         with _arm_lock:
    _arm_pos_target = q_cmd  ────────→      target = _arm_pos_target
    _arm_pos_target_ts = now                target_ts = _arm_pos_target_ts
    _arm_pos_target_prev = old              target_prev = _arm_pos_target_prev
    _arm_pos_target_prev_ts = old_ts        target_prev_ts = _arm_pos_target_prev_ts
                                      │
                                      error = target - qpos
                                      qvel = PID(error)
                                      vc_set_joint_velocity(qvel)
```

### 5.2 接口合约

| 合约项 | 约定 |
|--------|------|
| 生产者 | 外环 `send_action()` (唯一写入 `_arm_pos_target`) |
| 消费者 | 内环 `_pid_loop_impl()` (唯一读取 `_arm_pos_target`) |
| 同步原语 | `threading.Lock` (`_arm_lock`) — 临界区仅一次赋值/拷贝 |
| 空值协议 | `target = None` → 内环发送零速度实现自然减速 |
| 速率解耦 | IK 失败时外环仍每 20ms 发送上一帧目标, 内环完全不受影响 |
| 生命周期 | `_arm_should_stop` Event 信号终止, `join(timeout=2s)` |

### 5.3 None-Sentinel 的使用场景

| 场景 | 触发方式 | 内环行为 |
|------|----------|----------|
| PAUSED (PID 模式) | `arm.clear_target()` 设 None | 零速度 → 自然摩擦减速 |
| VR 追踪丢失 (Hard Stale) | 外环跳过 send_action, 旧 target 过期 | 零速度 → 自然减速 |
| 紧急停止 | `arm.stop()` → 清空 target | 零速度 → 自然减速 |
| 模式切换 | `_set_mode()` 设 `_velocity_control_active=False` | 跳过内环迭代 |

---

## 6. 安全纵深防御

```
Layer 1  VR 追踪质量门      时效分级 0.1/0.5/1.0s → 丢弃/减速/急停
         (controller.py)
─────────────────────────────────────────────────────────────────
Layer 2  Pipeline 检查       NaN 守卫 + 工作空间夹持+重解
         (pipeline.py)
─────────────────────────────────────────────────────────────────
Layer 3  速度限制层          驱动层 _limit_joint_step() (bottleneck scaling + soft-start)
         (xarm7.py)         外环不再做独立的速度限制步长
─────────────────────────────────────────────────────────────────
Layer 4  发送前安全门         validate_action() 7 项 + 指尖桌面 FK
         (interface.py)
─────────────────────────────────────────────────────────────────
Layer 5  PID 线程存活检查     守护线程死亡 → 急停 (5 行, 低开销冗余防御)
         (controller.py, 每 50 帧 ~1s)
─────────────────────────────────────────────────────────────────
Layer 6  硬件级安全           力矩/电流/温度/关节限位/碰撞检测/急停
         (xarm7.py, xhand.py, xArm SDK)
```

### Layer 4 详表: `validate_action()` 7 项检查

| # | 检查项 | 函数 | 失败条件 | 失败行为 |
|---|--------|------|----------|----------|
| 1 | 硬件错误状态 | `is_error()` | arm/hand SDK 错误标志 | **→ 急停** |
| 2 | 手臂连接状态 | `arm.is_connected()` | 断开 | **→ 急停** |
| 3 | 手臂力矩 | `check_arm_torque(tau)` | `|tau_i| ≥ limit_i` 或 NaN | hold |
| 4 | 手部电流 | `check_hand_current(cur)` | `max(cur) ≥ 500mA` 或 NaN | hold |
| 5 | 手部温度 | `check_hand_temperature(temp)` | `max(temp) ≥ 70°C` 或 NaN | hold |
| 6 | 手部通信 | `check_hand_comm(errs)` | 任一板级错误标志为 True | hold |
| 7 | 工作空间 | `workspace.check(fk_pos)` | EEF 位置超出 bounds | hold |

力矩限值 (Nm): J1=50, J2=50, J3=30, J4=30, J5=30, J6=20, J7=20

---

## 7. 端到端延迟分析

```
VR 头显 @ 120Hz
  │ 采集延迟 ~4ms (曝光+读出)
  ▼
ZMQ 传输 (~1-2ms, 取决于网络)
  │
  ▼
等待下一外环 tick (平均 10ms, 最大 20ms 排队延迟)
  │
  ▼
ArmMapper 映射 (< 0.1ms)
  │
  ▼
EMA 滤波 (α=0.95): 有效滞后 ~2ms (1帧时间常数)
  │
  ▼
IK 求解
  ├─ 迭代 DLS: ~1ms (100 iter → 早期收敛通常 3-5 iter)
  └─ MPlib 位置 IK: ~5-10ms (后备, 极少触发)
  │
  ▼
send_action: 锁传输 < 0.01ms
  │
  ▼
PID 跟踪带宽: ~10-30ms (取决于 PID 增益)
  │
  ▼
硬件执行 (机电延迟): ~5ms
  │
  ▼
机械臂到达目标位置

═══════════════════════════════════════════
端到端总延迟 (典型): 35-65ms
═══════════════════════════════════════════
```

### 延迟分解

| 阶段 | 历时 | 占比 |
|------|------|------|
| VR 采集+传输+排队 | ~15-25ms | ~45% |
| EMA 平滑 (α=0.95) | ~2ms | ~3% |
| IK + 计算 | ~1ms | ~2% |
| PID 跟踪 | 10-30ms | ~40% |
| 机电执行 | ~5ms | ~10% |

> **注意**: α=0.75 时 EMA 有效滞后约 7ms (群延迟+相位滞后)，非早期估计的 60ms。60ms 是整个 EMA+vel_limit+PID 跟踪的串联延迟感知总量，不是 EMA 滤波器本身的延迟。

---

## 8. 关键参数速查表

### 8.1 时序参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 外环频率 | 50 Hz (20ms) | `controller.py` `target_hz=50.0` |
| 内环频率 | 250 Hz (4ms) | `xarm7.py` `inner_control_dt=1/250.0` |
| 过采样比 | 5:1 | — |
| 循环超限报警 | > 30ms (150% × 20ms) | `controller.py` |
| 状态日志间隔 | 2s | `controller.py` |
| PID 线程检查间隔 | 50 帧 (~1s) | `controller.py` |

### 8.2 VR 追踪参数

| 参数 | 值 | 位置 |
|------|-----|------|
| SOFT_STALE | 0.1s | `controller.py` |
| HARD_STALE | 0.5s | `controller.py` |
| EMERGENCY | 1.0s | `controller.py` |

### 8.3 平滑参数

| 参数 | 值 | 位置 |
|------|-----|------|
| EMA alpha (Controller 默认) | 0.95 | `controller.py` |
| EMA alpha (Pipeline 默认) | 1.0 (即不滤波) | `pipeline.py` |
| 手部平滑 | 由 dex-retargeting 内置 `low_pass_alpha` 处理 | — |

### 8.4 安全限值

| 参数 | 值 | 位置 |
|------|-----|------|
| 手臂力矩 J1-J2 | 50 Nm | `types.py` |
| 手臂力矩 J3-J5 | 30 Nm | `types.py` |
| 手臂力矩 J6-J7 | 20 Nm | `types.py` |
| 手部电流硬限 | 500 mA | `safety.py` |
| 手部温度硬限 | 70°C | `safety.py` |
| 手部retarget范围 | [-0.75, 2.0] rad | `safety.py` |
| 工作空间 X | [0.28, 0.72] m | `types.py` |
| 工作空间 Y | [-0.45, 0.45] m | `types.py` |
| 工作空间 Z | [0.05, 0.50] m | `types.py` |
| 桌面安全 Z (指尖) | 0.03 m (table_z + margin) | `desk_safety.py` |

### 8.5 运动限值

| 参数 | 值 | 位置 |
|------|-----|------|
| max_qvel (7 关节统一) | 180°/s (π rad/s) | `xarm7.py` |
| pid_max_vel (7 关节统一) | 108°/s (1.885 rad/s, 60% max_qvel) | `xarm7.py` |
| Soft-start 斜坡 | 0.3 s | `xarm7.py` |
| IK 最大 IK 跳变 | 90° (全关节) | `types.py` `max_ik_jump_deg` |
| IK DLS 迭代次数 | 100 (早期收敛通常 3-5) | `types.py` |
| IK DLS 阻尼 | λ²=1e-5 (λ=0.003162) | `types.py` |
| IK DLS 最终位姿步长限制 | pos 0.02m, rot 5° | `types.py` |
| IK 最终精度 | pos ≤ 0.008m, rot ≤ 0.08rad | `types.py` |

### 8.6 ArmMapper 参数

| 参数 | 默认值 | 位置 |
|------|--------|------|
| pos_scale | 1.0 | `arm_mapper.py` |
| rot_scale | 1.0 | `arm_mapper.py` |
| max_delta_rot_rad | 1.0 rad (~57°) | `arm_mapper.py` |
| vr_to_base_rot | I₃ (单位阵) | `arm_mapper.py` |

---

## 9. 核心设计决策

### 9.1 为何双层控制 (50Hz 外环 + 250Hz 内环)?

| 考量 | 外环 @ 50Hz | 内环 @ 250Hz |
|------|-------------|--------------|
| **计算需求** | 感知 (VR), 规划 (IK), 决策 (安全门) — 无法实时 | PID 纯数学, O(7) — 可高频 |
| **任务性质** | 非线性, 有全局约束 (工作空间, 碰撞, 奇异点) | 线性跟踪, 纯局部 — PID 足够 |
| **传感器带宽** | VR 帧最多 120Hz, 人手运动 ~5-10Hz | 关节编码器 > 1kHz |
| **失效模式** | IK 偶尔失败 → hold, 不影响内环连续性 | 硬件故障 → 急停 |
| **解耦收益** | IK 抖动不传递给速度指令 | 速度平滑不受外环计算波动影响 |

### 9.2 为何 IK 确定性优先 (DLS → MPlib → Hold)?

- **DLS (迭代微分 IK)**: 确定性 — 相同输入总是产生相同输出, 无随机种子导致的肘关节翻转。迭代 100 次（通常 3-5 次收敛），与 BunnyVisionPro 对齐。
- **MPlib (采样 IK)**: 随机 — 每次求解可能返回不同分支 (如 elbow up/down), 导致视觉上不连贯的跳跃
- **策略**: 确定性方法覆盖 95%+ 情况; 随机方法仅作为近奇异/大误差时的后备

### 9.3 为何 bottleneck 等比缩放而非逐关节独立限幅?

```
独立限幅 (❌):            等比缩放 (✅):
joint_1: +3°  ok          joint_1: +3° → 3/5 = 0.6
joint_2: +5°  → clip→5°   joint_2: +5° → 5/5 = 1.0 (超限)
joint_3: +2°  ok          joint_3: +2° → 2/5 = 0.4
                          scale = 1/1.0 = 1.0? No...
                          
                          实际: 所有关节 delta × (limit/max_delta)
                          若 joint_2=+7°: scale=5/7=0.714
                          joint_1=+2.14°, joint_2=+5.0°, joint_3=+1.43°
                          
                          → 轨迹形状在关节空间被完整保留
```

等比缩放保持关节空间轨迹的**方向**, 只是缩小了幅度。独立限幅会改变方向, 在笛卡尔空间产生意外的末端轨迹扭曲。

### 9.4 为何 EMA 参考 `_last_raw_arm` 而非 `_last_arm_cmd`? (F2)

```
❌ 错误方案: EMA 参考 _last_arm_cmd (速度限制后的发送值)
  帧 N:   raw_IK=100 → vel_limit → sent=95    EMA: 0.95×95 + 0.05×prev = ...
  帧 N+1: raw_IK=105 → vel_limit → sent=98    EMA: 0.95×98 + 0.05×...  = ...
  速度限制器每帧都"吃掉"一点位移 → EMA 中的滞后逐渐累积

✅ 正确方案: EMA 参考 _last_raw_arm (原始 IK 输出)
  帧 N:   raw_IK=100 → vel_limit → sent=95    EMA: 0.95×100 + 0.05×prev = ...
  帧 N+1: raw_IK=105 → vel_limit → sent=98    EMA: 0.95×105 + 0.05×... = ...
  EMA 反映的是意图轨迹, 速度限制器独立作用于 EMA 输出
  → 两个滤波器解耦, 不累积滞后
```

### 9.5 EMA 参数选择指南

alpha 值通过 `TeleopControllerConfig.ema_alpha_arm` 配置（当前默认 0.95）：

| alpha | 时间常数 (50Hz) | 有效滞后 | 适用场景 |
|-------|----------------|----------|----------|
| 1.0 | 无平滑 | 0ms | 仿真 / 调试 / 最低延迟需求 (Pipeline 默认) |
| 0.95 | ~1 帧 | ~2ms | **Controller 默认** — 接近原始跟踪, 微滤波 |
| 0.75 | ~3 帧 | ~7ms | 平衡手颤滤波与延迟 |
| 0.50 | ~5 帧 | ~20ms | 精密静态操作, 重手颤滤波 |

公式: `cmd_smoothed = α × cmd_raw + (1-α) × cmd_prev`

仿真脚本 (`examples/sim/vr_teleop_sim.py`) 默认使用 α=1.0（关闭 EMA），因为仿真环境无手颤噪声。

### 9.6 为何选择迭代 DLS (对齐 BVP) 而非单步 DLS?

| 维度 | 单步 DLS (旧设计) | 迭代 DLS (当前, 对齐 BVP) |
|------|------------------|--------------------------|
| 计算 | 1 次 FK+Jacobian, 1 次 DLS | 最多 100 次 FK+Jacobian+DLS |
| 延迟 | ~0.5ms | ~1ms (通常 3-5 次收敛) |
| 精度 | ~1-2mm 额外误差 (λ² 偏置) | 收敛到 ‖error‖ < 1e-3 |
| 阻尼 | 需要较大阻尼 (0.02) 保证稳定性 | 可用极小阻尼 (λ²=1e-5), 精度更高 |
| 步长限制 | 每步限 0.02m/5° (输入裁剪) | 仅最终 delta 限幅 (内部迭代无限制) |

迭代 DLS 以 ~0.5ms 额外延迟换取显著更好的 IK 精度，且与 BVP 架构对齐，便于参考调试。

---

## 10. 简化演进历史

以下是已执行的控制环路简化，按时间排序：

### 已删除的机制

| 日期 | 删除项 | 原因 | 影响 |
|------|--------|------|------|
| 2026-06 | **CartPoseInterpolator** (线性+SLERP 插值) | VR 原生 50Hz = 控制频率, 无频率解耦需求；EMA 已处理帧间滤波；BVP/LeFranX/T-Rex 均无此机制 | -20ms 延迟, -~50 行 |
| 2026-06 | **跳变钳位 (apply_jump_clamp)** — 手臂 5°/帧 + 手部 10°/帧 | 手臂：速度限制步长已在驱动层覆盖；手部：retargeter 内置 low_pass_alpha 平滑 | -~30 行 |
| 2026-06 | **滑动窗口趋势监控** (温度/电流) | 仅产生 warning 日志, 不参与控制决策；validate_action() 硬限是真正的安全门 | -~30 行, -1 I/O/tick |
| 2026-06 | **跟踪发散检测** (5.0 rad × 3 帧) | PID 内环双层 try/except + error_state 机制已覆盖, validate_action() 在 ~20ms 内捕获 | -~25 行 |
| 2026-06 | **Target Lead Governor** (MAX_LEAD=3cm, chase_pos) | BVP 无此机制；逐轴 clip 扭曲运动方向；删除后 target 直接入 IK | -~10 行 |
| 2026-06 | **EMA α 调整** 0.75 → 0.95 | BVP 无 EMA；降低 α 减少延迟 (~7ms → ~2ms) 同时保留微滤波 | 延迟 -5ms |
| 2026-06 | **单步 DLS → 迭代 DLS** | 对齐 BVP：100 iter, 固定 λ²=1e-5, 收敛判定 ‖error‖<1e-3 | 精度提升, +0.5ms |

### 保留的机制及其理由

| 机制 | 保留原因 |
|------|---------|
| **EMA 平滑** (α=0.95) | 微滤波防止 IK 噪声抖动, ~2ms 延迟可接受；D-on-M PID 无法完全替代 |
| **MPlib IK 后备** | DLS 失败时的降级路径；`plan_path`（自主规划）仍依赖 mplib |
| **速度限制步长** (驱动层 bottleneck scaling) | 运动质量关键机制 — 限制 PID 跟踪误差 ≤ 3.6°, PID 输出 ≤ 25°/s |
| **PID 线程存活检查** | 仅 5 行代码, 作为 error_state 的冗余防御, 删除收益微 |
| **SAVE_PROMPT 状态** | UX 决策 — 防止误操作丢失录制数据 |

### 累计效果

| 指标 | 简化前 | 简化后 | 变化 |
|------|--------|--------|------|
| 外环每 tick 处理步骤 | ~13 | ~9 | -30% |
| 端到端延迟 | ~50-80ms | ~35-65ms | -~15ms |
| 跳变钳位 | 手臂 5° + 手部 10° | 无 (驱动层+retargeter 覆盖) | — |
| EMA 延迟贡献 | ~7ms (α=0.75) | ~2ms (α=0.95) | -5ms |
| DLS 精度 | ~1-2mm 额外误差 | 收敛到 <1e-3 | 提升 |

---

## 附录: 文件索引

| 模块 | 文件 | 说明 |
|------|------|------|
| TeleopController (外环主循环) | `teleop/core/controller.py` | 状态机, _tick(), 键盘处理 |
| TeleopPipeline (动作计算) | `teleop/core/pipeline.py` | compute_action(), EMA, workspace clamp |
| XArm7 (内环 PID) | `robot/xarm7/xarm7.py` | _pid_loop_impl(), _clip_arm_velocity(), soft-start |
| RobotInterface (统一接口+安全检查) | `robot/interface.py` | validate_action(), send_action() |
| TeleopIKSolver (IK 两层策略) | `planning/ik.py` | solve(), solve_differential_ik(), solve_position_ik() |
| XArm7MotionPlanner (运动规划) | `planning/planner.py` | plan_path(), solve_teleop_ik() |
| Safety checks (安全函数) | `teleop/control/safety.py` | check_arm_torque, check_hand_current, check_hand_temperature |
| ArmWristMapper (VR→EEF 映射) | `teleop/vr/arm_mapper.py` | map(), reset() |
| Robot types (状态/动作/配置) | `robot/types.py` | RobotState, RobotAction, RobotInterfaceConfig |
| Planning types (IK 相关类型) | `planning/types.py` | TeleopProfile, PlanningProfile, Pose, IKResult |
| WorkspaceSafety | `planning/workspace_safety.py` | 工作空间边界检查 |
| FingertipDeskSafety | `planning/desk_safety.py` | FK 指尖 Z 轴桌面碰撞检测 |
| EMA smoothing | `utils/signal_utils.py` | ema_smooth() |
| RateLimiter | `utils/rate_limiter.py` | 补偿式频率限制 |
