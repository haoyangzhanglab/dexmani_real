# XArm7 控制 · IK · 安全 机制梳理

> 最后更新：2026-06-18
> 基于 BunnyVisionPro 与 dexmani_real 对比后的系统性梳理。

---

## 目录

- [一、XArm7 控制机制](#一xarm7-控制机制)
  - [1.1 双模架构](#11-双模架构)
  - [1.2 PIDController 设计](#12-pidcontroller-设计)
  - [1.3 250Hz 内环线程](#13-250hz-内环线程)
  - [1.4 速度限制](#14-速度限制)
  - [1.5 速度模式 PID 软启动](#15-速度模式-pid-软启动)
  - [1.6 生命周期管理](#16-生命周期管理)
  - [1.7 关键配置参数速查](#17-关键配置参数速查)
  - [1.8 线程安全设计](#18-线程安全设计)
- [二、IK 系统机制](#二ik-系统机制)
  - [2.1 两级回退策略](#21-两级回退策略)
  - [2.2 DLS 微分 IK](#22-dls-微分-ik)
  - [2.3 Position IK](#23-position-ik)
  - [2.4 等效关节规范化](#24-等效关节规范化)
  - [2.5 IK 性能特征](#25-ik-性能特征)
  - [2.6 速度限制分层](#26-速度限制分层)
- [三、机械臂安全机制](#三机械臂安全机制)
  - [3.1 四层防护体系概览](#31-四层防护体系概览)
  - [3.2 驱动层安全](#32-驱动层安全)
  - [3.3 接口层安全](#33-接口层安全)
  - [3.4 控制层安全](#34-控制层安全)
  - [3.5 路径层安全](#35-路径层安全)

---

## 一、XArm7 控制机制

### 1.1 双模架构

`XArm7`（`robot/xarm7/xarm7.py:157`）支持两种控制模式，由 `XArm7Config.use_servo_control` 切换（`xarm7.py:134`）：

| 模式 | SDK 模式 | 控制方式 | 适用场景 |
|------|---------|---------|---------|
| **Mode 1（位置伺服，默认）** | `set_mode(1)` | 直接发送关节角度命令 `set_servo_angle_j` | 简单遥操作、兼容模式 |
| **Mode 4（速度控制）** | `set_mode(4)` | PID 内环将位置误差转为速度 `vc_set_joint_velocity` | 平滑运动、消除步进跳变 |

**核心差异**：Mode 1 直接下发位置指令，SDK 内部做插值。Mode 4 由 250Hz 内环线程持续计算位置误差 → PID → 速度，产生连续的速度指令，运动更平滑，参照 `BunnyVisionPro xarm7_ability.py`。

**模式切换时机**（`xarm7.py:199, 261, 387, 408`）：
- `connect()` 时根据 `use_servo_control` 设置初始模式
- `clear_error()` 恢复原模式
- `send_action()` 中检测当前模式不匹配时自动切换
- `reset()` 先切 Mode 0（阻塞位置模式），完成后恢复

> **架构差异（vs BunnyVisionPro）**：BunnyVisionPro 的 250Hz 内环线程对**两种模式均生效**——伺服模式下在线程内调用 `set_servo_angle_j`，速度模式运行 PID+`vc_set_joint_velocity`。dexmani 采用双路径架构：伺服模式下 `send_action()` 在主线程直接调用硬件（无内环线程），速度模式才启动 250Hz 内环线程。这一设计使伺服模式热路径更简洁，但调度行为与 BVPro 不同。

### 1.2 PIDController 设计

`PIDController`（`xarm7.py:37-99`）是 7-DOF 独立关节 PID 控制器，参照 `BunnyVisionPro xarm7_ability.py:11-36`。

**控制律**（`xarm7.py:90-93`）：

```
vel = Kp × err  +  Kd × (err − prev_err) / dt  +  Ki × cum_err
```

**关键设计决策**：

| 参数 | 值 | 原因 |
|------|-----|------|
| `Kp` | `[10, 10, 5, 5, 5, 5, 5]` | 基座关节(J1-J2)增益更高以承担更多负载（`xarm7.py:136-138`） |
| `Kd` | `Kp / 20`，即 `[0.5, 0.5, 0.25, ...]` | 默认 Kd = Kp/20，提供足够阻尼（`xarm7.py:63-66, 139-141`） |
| `Ki` | **全零（禁用）** | 遥操作中目标持续变化，积分项会导致 windup（`xarm7.py:43-45`） |

**状态管理**（`xarm7.py:70-73`）：
- `reset()` — 清零误差历史，在模式切换 / 错误恢复 / arm reset 时调用
- `_prev_err` 初始化 — 首次 `control()` 调用时自动复制当前误差，无历史突变

### 1.3 250Hz 内环线程

当 `use_servo_control=False`（Mode 4 速度控制）时，`connect()` 启动后台线程 `_internal_control_arm_qpos`（`xarm7.py:217-231`）。

**线程循环**（`xarm7.py:487-562`）：

```
┌─────────────────────────────────────────────────┐
│  外层 (50Hz):  send_action(qpos)               │
│                  └→ store _arm_pos_target       │
│                                                 │
│  内环 (250Hz):  while not stop:                 │
│                   read _arm_pos_target (lock)    │
│                   read hardware qpos             │
│                   error = target - current       │
│                   velocity = PID(error, dt)      │
│                   safe_vel = clip_velocity(vel)  │
│                   vc_set_joint_velocity(safe_vel)│
└─────────────────────────────────────────────────┘
```

**数据流细节**：

1. **目标来源**：`send_action()` 在 Mode 4 下不直接控制硬件，仅将目标写入 `_arm_pos_target`（`xarm7.py:389-390`）
2. **硬件读取**：每周期调用 `get_joint_states(is_radian=True)`（`xarm7.py:515`）
3. **误差计算**：7D 位置误差 `target[:7] - current[:7]`（`xarm7.py:535`）
4. **PID 控制**：`_arm_pid.control(error, dt)` 输出速度（`xarm7.py:536`）
5. **速度钳制**：`_clip_arm_velocity()` bottleneck 缩放 + 软启动（`xarm7.py:537`）
6. **硬件发送**：`vc_set_joint_velocity(safe_qvel.tolist())`（`xarm7.py:541`）

**错误处理**（`xarm7.py:516-562`）：
- `get_joint_states` 失败 → `error_state = True`
- SDK 返回非零 code → `error_state = True` + 刷新 SDK 错误码
- `vc_set_joint_velocity` 失败 → `error_state = True`
- 错误不终止线程，`continue` 等待下一个周期恢复

**线程生命周期**：
- 启动：`connect()` → `_arm_thread.start()`（`xarm7.py:226`）
- 停止：`disconnect()` / `stop()` → `_stop_inner_thread()`（`xarm7.py:564-573`）
- `_arm_should_stop` Event 控制循环退出
- `join(timeout=2.0)` 等待线程退出，超时报警

### 1.4 速度限制

#### 1.4.1 伺服模式速度限制

`send_action()` 在 Mode 1 下调用 `_limit_joint_step()`（`xarm7.py:355`）进行三步处理：

1. **关节范围钳制**（`xarm7.py:351`）：`_limit_joint_range()` 将目标 qpos 钳制到 `[qpos_min, qpos_max]`
2. **步长限制**（`xarm7.py:355`）：`_limit_joint_step()` 做 bottleneck 缩放

**Bottleneck 缩放**（`xarm7.py:629-703`，参照 `BunnyVisionPro xarm7_ability.py clip_arm_next_qpos`）：

- **原理**：当任一关节超速时，**所有关节**按相同比例缩放 — 保留关节空间轨迹形状
- **参考基准**：使用**硬件实际位置**而非上次命令，防止跟踪滞后导致命令叠加（`xarm7.py:683-686`）
- **计算公式**：

```
delta = target_qpos − hw_qpos
normalized = |delta| / max_step          (per-joint)
max_ratio = max(normalized)
if max_ratio > 1.0:
    delta = delta / max_ratio            (scalar scaling)
```

**软启动 ramp**（`xarm7.py:669-677`，帧计数器策略参照 `ufactory_teleop uf_robot.py L206`，线性插值为 dexmani 增强）：

- 前 `soft_start_frames`（默认 20 帧）内，有效速度从 `soft_start_speed_rad_s`（默认 0.3 rad/s）线性 ramp 到 `max_qvel`
- 对比：ufactory_teleop 使用硬开关（0.2 rad/s → 全速，第 20 帧跳变），dexmani 改为连续线性插值消除速度跳变
- 线性插值公式：`(1 − progress) × soft_start + progress × max_qvel`
- 计数器 `_cmd_cnt` 追踪帧数，`reset_soft_start()` / `_set_mode()` 可归零

#### 1.4.2 速度模式速度限制

`_clip_arm_velocity()`（`xarm7.py:446-485`）对 PID 输出的速度做 bottleneck 缩放。

**与伺服模式一致的 bottleneck 逻辑**：

```
velocity_overshoot = |qvel| / pid_max_vel
max_overshoot = max(velocity_overshoot)
if max_overshoot > 1.0: safe_vel = qvel / max_overshoot
```

**独立的软启动机制**：见 [1.5 速度模式 PID 软启动](#15-速度模式-pid-软启动)。

### 1.5 速度模式 PID 软启动

速度控制模式下，PID 内环有独立的**时间驱动**软启动（`xarm7.py:462-472`。BunnyVisionPro 使用**二进制减速**—在初始化收敛时将 `max_arm_velocity` 降至 1/3 然后一次性恢复；dexmani 扩展为时间驱动的线性 ramp 0.4s 30%→100%，更为平滑）：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `pid_soft_start_duration_s` | `0.4` s | 软启动持续时间（`xarm7.py:130`） |
| 起始速度比例 | `30%` | PID 最大速度的 30%（`xarm7.py:467`） |
| 终止速度比例 | `100%` | PID 最大速度的 100% |

**启动时机**：
- `connect()` 启动内环线程时（`xarm7.py:220`）
- `clear_error()` 恢复后（通过 `_set_mode()` 间接触发，`xarm7.py:580`）
- `reset()` 后（`xarm7.py:421-423`）
- `reset_soft_start()` 调用时（`xarm7.py:439-440`）

**Ramp 曲线**：

```
effective_limit = pid_max_vel × (0.3 + 0.7 × elapsed / duration)
```

Ramp 完成后 `_vel_ramp_start = None`（热路径零开销，`xarm7.py:469`）。

### 1.6 生命周期管理

```
connect() ─→ send_action() ─→ [error] ─→ clear_error() ─→ send_action() ...
    │                                       │
    │                                       └→ stop() (紧急停止)
    │
    └→ reset() ─→ ... ─→ disconnect()
```

**各阶段详解**：

| 阶段 | 方法 | 行号 | 关键操作 |
|------|------|------|---------|
| **连接** | `connect()` | `xarm7.py:183-232` | `XArmAPI(ip)`, `clean_error/warn`, `motion_enable(True)`, `_set_mode()`, `_configure_collision_params()`, 启动内环线程 |
| **发送动作** | `send_action()` | `xarm7.py:332-395` | Mode 1: `_limit_joint_range` → `_limit_joint_step` → `set_servo_angle_j` / Mode 4: store target |
| **状态读取** | `get_state()` | `xarm7.py:286-326` | `get_joint_states` (qpos, qvel, tau) + 可选全量诊断 |
| **错误检测** | `is_error()` | `xarm7.py:244-253` | 检查 arm None / connected_flag / error_state / arm.error_code |
| **错误恢复** | `clear_error()` | `xarm7.py:255-270` | `clean_error/warn`, `motion_enable`, 恢复模式, 复位 PID + 同步目标 |
| **紧急停止** | `stop()` | `xarm7.py:272-284` | 发送零速 → 停内环 → `set_state(4)` |
| **回零** | `reset()` | `xarm7.py:397-425` | 切 Mode 0 → `set_servo_angle(wait=True)` → 恢复模式 → 复位 PID |
| **断开** | `disconnect()` | `xarm7.py:234-239` | 停内环 → `arm.disconnect()` |

### 1.7 关键配置参数速查

完整定义见 `robot/xarm7/xarm7.py:103-155`（`XArm7Config`）。

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `ip` | `192.168.1.111` | xArm 控制器 IP |
| `dt` | `1/50` s | 外层控制周期 |
| `init_qpos` | `[-30°, -45°, 0°, 20°, -180°, 25°, 0°]` | 初始/回零位置 |
| `qpos_min` / `qpos_max` | 全关节范围 | 硬件关节限制 |
| `max_qvel` | `[1.57, 1.57, 1.57, 1.57, 2.09, 2.09, 2.62]` rad/s（≈ `[90, 90, 90, 90, 120, 120, 150]°/s`） | 每关节最大速度（`np.deg2rad` 在构造时转换） |
| `reset_speed` | `20°/s` | 回零线速度 |
| `reset_acc` | `180°/s²` | 回零加速度 |
| `use_delta_limit` | `True` | 启用步长限制 |
| `clip_joint_limit` | `True` | 启用关节范围钳制 |
| `soft_start_frames` | `20` | 伺服模式软启动帧数 |
| `soft_start_speed_rad_s` | `0.3` rad/s | 伺服模式软启动起始速度 |
| `use_servo_control` | `True` | True=Mode 1（位置伺服）, False=Mode 4（速度控制） |
| `inner_control_dt` | `1/250` s | 内环控制周期 |
| `pid_kp` | `[10, 10, 5, 5, 5, 5, 5]` | PID 比例增益 |
| `pid_kd` | `[0.5, 0.5, 0.25, ...]` | PID 微分增益（默认 Kp/20） |
| `pid_max_vel` | `[1.2, 1.2, 1.2, 1.2, 1.6, 1.6, 2.0]` rad/s（~76% of max_qvel） | PID 输出的最大速度限制（2026-06-18 从 BVPro 原值 `[0.8,0.8,0.8,0.8,1.0,1.0,1.5]` ~51% 提升至 76%，缩小双模速度差距） |
| `pid_soft_start_duration_s` | `0.4` s | 速度模式软启动持续时间 |
| `tcp_load_kg` | `1.2` kg | TCP 负载质量（XHand 重量） |
| `tcp_load_cog_mm` | `[0, 0, 80]` mm | 负载质心（法兰坐标系） |
| `collision_sensitivity` | `3` | 碰撞灵敏度（0=关闭, 1=最不敏感, 5=最敏感, 3=出厂默认） |

> **参数来源说明**：`pid_kp`、`pid_kd`、`pid_max_vel`、`inner_control_dt`、`use_servo_control` 继承自 BunnyVisionPro。`tcp_load_kg`、`tcp_load_cog_mm`、`collision_sensitivity`、`soft_start_frames`、`soft_start_speed_rad_s`、`pid_soft_start_duration_s`、`clip_joint_limit`、`use_delta_limit` 为 dexmani 扩展。`dt`、`init_qpos`、`qpos_min/max`、`max_qvel`、`reset_speed`、`reset_acc` 为通用 xArm SDK 配置。

### 1.8 线程安全设计

**同步原语**（`xarm7.py:174-176`）：

| 原语 | 类型 | 用途 |
|------|------|------|
| `_arm_lock` | `threading.Lock` | 保护 `_arm_pos_target` 读写 |
| `_arm_should_stop` | `threading.Event` | 信号内环线程停止 |
| `_arm_pos_target` | `np.ndarray` (atomic ref) | 外层写入目标，内环读取 |

**并发模型**：

- **外层**（50Hz，主线程）：`send_action()` 获取 `_arm_lock` 写入目标（`xarm7.py:389`）
- **内环**（250Hz，daemon 线程）：`_internal_control_arm_qpos` 获取 `_arm_lock` 读取目标（`xarm7.py:508-509`）
- Lock 粒度极小（仅保护目标读写），不阻塞硬件 I/O
- 内环线程为 daemon，通过 `_arm_should_stop` Event + `join(timeout=2.0)` 实现优雅停止（非依赖 daemon 自动终止）

---

## 二、IK 系统机制

### 2.1 两级回退策略

`TeleopIKSolver.solve()`（`planning/ik.py:46-109`）实现两级回退：

```
Level 1: 微分 IK (DLS)
  ├─ 成功 + 位姿误差通过阈值 → 返回
  └─ 失败 / 位姿误差过大
       │
       └→ Level 2: Position IK (MPlib)
            ├─ 找到硬件最近的有效候选 → 返回
            └─ 全部失败
                 │
                 └→ Hold: 返回上一帧命令 (held=True)
```

**关键设计原则**：
- Level 1（DLS）是**确定性的** — 相同输入必得相同输出，无随机性，无分支切换
- Level 2（Position IK）是**随机的** — 仅作回退，采硬件最近候选避免分支跳变
- 两级都失败 → Hold 上一帧命令，不发送危险值
- 两级均可通过 `TeleopProfile.use_differential_ik_fallback` / `use_position_ik` 独立开关；同时关闭则直接 Hold

> **回退串配置化**：`ik.py:63-83` 中的两级回退受 `TeleopProfile` 布尔开关控制。默认串行（DLS→Position IK→Hold），但可通过关闭 `use_differential_ik_fallback` 跳过 DLS 直接走 Position IK，或关闭 `use_position_ik` 仅使用 DLS。以上描述基于默认配置。

### 2.2 DLS 微分 IK

`solve_differential_ik()`（`planning/ik.py:215-272`）实现阻尼最小二乘（Damped Least Squares）微分 IK。

**算法流程**：

```
1. 计算当前 EEF 位姿 → 位姿误差向量（世界坐标系）
2. gain=1.0  × 误差（全跟踪，速度安全由驱动层处理）
3. 旋转误差到 EEF 局部坐标系（Jacobian 定义在局部系）
4. 计算 Jacobian (6×7)
5. DLS 求解：dq = J^T @ (J @ J^T + λ²I)⁻¹ @ error
6. new_qpos = current_qpos + dq
7. canonicalize → 组装 IKResult
```

**关键参数**（`planning/types.py:114-117`）：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `differential_ik_gain` | `1.0` | 全跟踪增益，速度安全由驱动层 bottleneck 处理 |
| `differential_ik_damping` | `0.02` | DLS damping λ，防奇异 |
| `differential_ik_max_pos_step_m` | `0.02` m | 位姿误差钳制（世界坐标系） |
| `differential_ik_max_rot_step_rad` | `5°` | 旋转误差钳制 |

> **vs BunnyVisionPro**：BVPro 使用**迭代 DLS**（最多 100 次迭代，`step_size=0.05`，`damping=1e-5`），每次都重新计算 FK 和 Jacobian，用于离线路径生成。dexmani 使用**单步 DLS**（`gain=1.0`，`damping=0.02`），更大的 damping 使单步求解足够精确，速度安全由驱动层 bottleneck 处理。这一选择使遥操作热路径更轻量（~1ms vs ~10ms+）。

**Jacobian 来源**（`planning/kinematics.py:58-67`）：
- Pinocchio `compute_single_link_jacobian` 在 EEF link 局部坐标系计算
- Jacobian 为 `6 × dof`，截取前 7 列
- 支持 `pad_move_group_qpos` 处理含灵巧手的完整 URDF

**DLS 求解**（`planning/ik.py:246-256`）：
- `J @ J^T + λ²I` 建 6×6 对称正定矩阵
- `np.linalg.solve` 求解，失败返回 `LinAlgError` → fallback

**验证**（`planning/ik.py:69-73`）：DLS 结果需通过 `max_pose_error_pos_m`（默认 8mm）和 `max_pose_error_rot_rad`（默认 0.08 rad）阈值检查，不通过则回退到 Position IK。

### 2.3 Position IK

`solve_position_ik()`（`planning/ik.py:115-176`）是随机回退 IK，仅在 DLS 失败时调用。

**种子策略**（`planning/ik.py:131-134`）：

| 优先级 | 种子 | n_init_qpos | 理由 |
|--------|------|-------------|------|
| 1 | `prev_cmd`（上一帧命令） | 3 | 最可能接近当前解 |
| 2 | `current_qpos`（当前实际位置） | 2 | 备选 |

**候选评估流程**（`planning/ik.py:139-175`）：

```
for seed in [prev_cmd, current_qpos]:
    status, raw_qpos = MPlib.IK(seed, n_init_qpos)
    if failed → 跳过
    qpos = canonicalize_qpos(raw_qpos, prev_cmd)
    if 位姿误差 > 阈值 → 跳过
    if 关节跳变 > max_ik_jump_deg → 跳过
    记录硬件最近候选 (best)
    if (prev_cmd seed) and (hw_dist ≤ 15°):
        立即接受 (快速路径)
返回 best 候选（硬件最近）
```

**Position IK 候选过滤与评分**（`planning/ik_candidates.py:112-194`）：

路径规划场景使用更完整的候选管理：

1. **生成**（`generate_ik_seeds`, `ik_candidates.py:42-53`）：当前 qpos + `num_random_ik_seeds` 个随机偏移种子
2. **收集**（`collect_ik_candidates`, `ik_candidates.py:57-108`）：对每个种子调用 MPlib IK → 规范化 → 过滤
3. **过滤**（`filter_ik_candidate`, `ik_candidates.py:112-169`）：
   - 规划限制检查（`limit_violation`）
   - 关节跳变检查（`max_ik_delta_deg`）
   - 位姿误差检查
   - 自碰撞检查（可选）
4. **评分**（`score_ik_candidate`, `ik_candidates.py:173-194`）：

```
score = joint_delta_weight × joint_cost              (关节变化代价)
      + pose_error_weight × pose_cost                (位姿误差代价)
      + joint_limit_weight × limit_cost              (关节限制代价)
      − manipulability_weight × manipulability       (可操度奖励)
      + neutral_weight × neutral_distance            (中立位姿距离)
```

5. **排序**：按 score 升序排列，取前 `num_ik_candidates` 个

> **vs LeFranX**：LeFranX `weighted_ik.cpp` 使用 3 项评分（manipulability − neutral − current_distance），无位姿误差项和关节限制项。dexmani 扩展为 5 项：新增关节 delta 代价和位姿误差代价作为主对齐项，并将 `current_distance` 替换为规范化感知的 qpos delta。

### 2.4 等效关节规范化

`canonicalize_qpos()`（`planning/ik_candidates.py:230-253`）处理冗余关节的等效角度。

**背景**：7-DOF 机械臂的某些关节（连续旋转关节）具有周期性 — `q` 和 `q + 2π` 代表同一物理构型，但数学上差异巨大。

**算法**：

1. 识别等效关节：`equivalent_joint_mask` 标记关节范围 ≥ π 的关节（`planner.py:84`）
2. 周期性：`periods[j] = min(2π, joint_range[j])`
3. 对于每个等效关节：
   ```
   k_min = ceil((low[j] − qpos[j] − tol) / period[j])
   k_max = floor((high[j] − qpos[j] + tol) / period[j])
   k = clip(round((ref[j] − qpos[j]) / period[j]), k_min, k_max)
   qpos[j] += k × period[j]
   ```
   （当 `k_min > k_max` 时设 k=0，防止无效偏移）
4. 钳制到 `[low, high]`

> 注：简化描述 `k = round(...)` 省略了 `k_min/k_max` 的边界感知逻辑（`ik_candidates.py:246-252`）。当规划极限约束可能的 wrap 区间时，该逻辑确保 k 不会超出可行范围，防止产生违反极限的结果。

**应用场景**：
- **遥操作**：将 IK 结果规范化到最接近上一帧命令的等效角度，避免分支切换（`ik.py:149`）
- **路径规划**：路径逐点规范化到前一点，保证连续性（`canonicalize_path_to_planning_limits`, `ik_candidates.py:255-265`）
- **距离计算**：`compute_qpos_delta()`（`ik_candidates.py:278-285`）将差值 wrap 到 `[−period/2, period/2]`

### 2.5 IK 性能特征

| 方法 | 典型耗时 | 确定性 | 适用场景 |
|------|---------|--------|---------|
| **微分 IK (DLS)** | ~1 ms | ✅ 确定 | 遥操作主路径（小幅跟踪） |
| **Position IK (MPlib)** | ~10 ms | ❌ 随机 | 回退（大幅误差 / 近奇异） |
| **plan_screw** | ~50-200 ms | ✅ 确定 | 路径规划首选 |
| **plan_qpos (RRT)** | ~0.5-2.0 s | ❌ 随机 | 路径规划回退（plan_screw 失败） |

**遥操作中的应用**（`planning/ik.py:46-55`）：
- 主路径：微分 IK（快速、确定性）
- 回退：Position IK（稳健）
- 最后防线：Hold 上一帧

**路径规划中的应用**（`planning/planner.py:131-194`）：
- 首选：`plan_screw`（快速、确定性）
- 回退：多候选 `plan_qpos` RRT（多种 `rrt_range` × 多次尝试）
- 验证：通过 `validate_path` 的 10 步检查

### 2.6 速度限制分层

速度限制不在 IK 层执行 — 全部由驱动层处理：

| 层级 | 组件 | 速度限制 | 文件:行号 |
|------|------|---------|-----------|
| **IK 层** | `TeleopIKSolver` | ⚠️ 仅位姿误差钳制（`max_pos_step`/`max_rot_step`），不做关节级速度限制 | `ik.py:195-197, 228-233` |
| **控制层** | `TeleopController._apply_jump_clamp` | ⚠️ 大跳钳制（5° arm / 10° hand） | `controller.py:421-456` |
| **驱动层** | `XArm7._limit_joint_step` | ✅ bottleneck 缩放 + 软启动 | `xarm7.py:629-703` |
| **驱动层** | `XArm7._clip_arm_velocity` | ✅ bottleneck 缩放 + 软启动 | `xarm7.py:446-485` |

**设计理由**（`ik.py:195-197` 注释）：
> Speed limiting is NOT done here — it is handled exclusively by XArm7._limit_joint_step() in the hardware driver layer.

这一分层确保 IK 层只关注运动学求解，速度安全统一由驱动层保证，避免重复限制导致运动迟钝。

---

## 三、机械臂安全机制

### 3.1 四层防护体系概览

```
┌─────────────────────────────────────────────────────┐
│  路径层 (Path Layer)                                 │
│  10 步路径验证、自碰撞/环境碰撞、elbow flip、关节限制 │
│  planner.py:430-525                                  │
├─────────────────────────────────────────────────────┤
│  控制层 (Control Layer)                              │
│  11-flag 质量标记（1 位保留未实现）、紧急停止、力矩/电流/温度监控        │
│  跳跃钳制、跟踪质量门控                              │
│  controller.py:198-240, safety.py                    │
├─────────────────────────────────────────────────────┤
│  接口层 (Interface Layer)                            │
│  工作空间边界、桌面点云碰撞体、回 home 多阶段安全     │
│  interface.py:175-180, 678-735, 354-418              │
├─────────────────────────────────────────────────────┤
│  驱动层 (Driver Layer)                               │
│  C31 防误触发、关节限制、速度限制、PID 状态复位       │
│  xarm7.py:582-613, 621-703, 446-562                  │
└─────────────────────────────────────────────────────┘
```

> **关于四层分类**：这是一个概念模型，而非严格隔离——某些检查在多层中互补存在。例如，工作空间检查同时出现在接口层（`send_action` 前的 `logger.debug` 诊断）和路径层（`validate_path` 的 waypoint 逐点校验）：接口层做命令级诊断，路径层做路径级拒绝，实时 hold 在控制层（`IN_WORKSPACE` 质量标记）。七个参考项目中仅 BunnyVisionPro 做了部分类似的层级分离（单层速度裁剪），其余项目均无此多层架构，因此四层分类是 dexmani 独有的安全设计。

### 3.2 驱动层安全

#### 3.2.1 C31 防误触发

`XArm7._configure_collision_params()`（`xarm7.py:582-613`）配置两项参数防止 C31（Collision Caused Abnormal Current）误触发：

**C31 检测机制**：
1. xArm 控制器通过动力学模型估算理论关节力矩
2. 比较实际扭矩（电机电流）与理论扭矩
3. 偏差超过阈值 → C31 紧急停止

**误触发根因**：未配置负载时，动力学模型假设 0kg → 理论力矩严重低估 → 驱动 XHand（~1.2kg）所需的实际扭矩被误判为碰撞。

**防护措施**：

| 配置 | 默认值 | API 调用 |
|------|-------|---------|
| TCP 负载质量 | `1.2` kg | `set_tcp_load(1.2, [0, 0, 80])`（`xarm7.py:599-601`） |
| 负载质心 | `[0, 0, 80]` mm | 同上（法兰坐标系 Z+ 80mm） |
| 碰撞灵敏度 | `3`（出厂默认） | `set_collision_sensitivity(3)`（`xarm7.py:609`，0=关闭/1=最不敏感/5=最敏感） |

#### 3.2.2 关节限制与速度限制

| 机制 | 方法 | 行号 | 说明 |
|------|------|------|------|
| 关节范围钳制 | `_limit_joint_range` | `xarm7.py:621-627` | `np.clip(qpos, qpos_min, qpos_max)` |
| 步长限制 | `_limit_joint_step` | `xarm7.py:629-703` | Bottleneck 缩放 + 软启动（详见 1.4） |
| PID 速度限制 | `_clip_arm_velocity` | `xarm7.py:446-485` | Bottleneck 缩放 + 软启动（详见 1.4） |
| 关节限制跟踪 | `last_joint_limit_clipped` | `xarm7.py:626` | 记录是否发生钳制，通过 `get_state(full=True)` 暴露 |

#### 3.2.3 PID 状态复位

在以下时机复位 PID（防误差累积）：
- `clear_error()` — `_arm_pid.reset()` + 同步 `_arm_pos_target` 到当前硬件位置（`xarm7.py:265-267`）
- `reset()` — `_arm_pid.reset()` + 同步目标到复位后位置（`xarm7.py:420-423`）
- `_set_mode()` — 复位速度软启动（`xarm7.py:580`）

### 3.3 接口层安全

#### 3.3.1 工作空间边界

`RobotInterface.__init__()` 中验证 home EEF 在工作空间内（`interface.py:174-184`）：
- 检查 `init_qpos` FK 得出的 EEF 位置是否在 `workspace_bounds` 内
- 配置错误 → 抛出 `ValueError`（启动即失败原则）
- NaN FK → 降级为 `warnings.warn`

`send_action()` 中做命令 EEF 的工作空间检查（`interface.py:318-325`）：
- 命令超出工作空间 → `logger.debug` 记录（实际执行由控制器层 `IN_WORKSPACE` 安全标记 hold 决定）

`WorkspaceSafety` 类（`planning/planner.py:582-609`）：
- `check(eef_pos)` — 布尔检查
- `clamp(target_pos)` — 钳制到边界（用于回 home 时的 EEF 目标修正）

#### 3.3.2 桌面点云碰撞体

`RobotInterface._setup_table_collision()`（`interface.py:678-735`）生成稠密桌面点云：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `table_z_world` | 配置值 | 桌面高度（世界坐标系） |
| `margin_xy` | `0.15` m | XY 方向外扩边距 |
| `n_layers` | `5` | 点云层数（向下堆叠） |
| `layer_spacing` | `0.01` m | 层间距 |
| `xy_resolution` | `0.02` m | XY 方向分辨率 |
| `x_min_clearance` | `0.15` m | X 方向最小安全距离（防底座碰撞） |

点云通过 `mplib.Planner.update_point_cloud()` 转换为 octree，`plan_screw` / `plan_qpos` / IK 自动避障。

#### 3.3.3 回 Home 多阶段安全

`return_to_home()`（`interface.py:354-418`）分两阶段执行：

**Phase 0**：手部复位 → 等待收敛（最多 3s）→ 确保手部 URDF 配置与真实配置一致（`interface.py:472-503`）

**Phase 1**：EEF 笛卡尔路径 → 稠密插值（1° 步长）→ 逐点发送 → 闭环收敛等待（`interface.py:505-586`）
- 路径前验证指尖在桌面上方（`interface.py:527-541`）
- `cancel_event` 支持 SIGINT 中断（`interface.py:547`）
- 收敛阈值：3°/joint，最多等 5× 理论时间（`interface.py:558-584`）

**Phase 2**：关节空间线性插值 → 每点碰撞检查 → 逐点发送（`interface.py:588-621`）
- 自碰撞/环境碰撞检查（`_safe_joint_path`, `interface.py:627-649`）
- 碰撞风险 → 跳过 Phase 2（EEF 已由 Phase 1 到位）

**Fallback**：直接复位（`_return_to_home_direct`, `interface.py:651-676`）
- Phase A：Z+ 抬升 50mm 清桌面
- Phase B：SDK 线性关节空间回零

### 3.4 控制层安全

#### 3.4.1 10-Flag 质量标记

`QualityFlags`（`recording/quality_flags.py`）使用 10-bit 位掩码标记每帧数据质量：

| Bit | Flag | 检查内容 | 检查位置 |
|-----|------|---------|---------|
| 0 | `TRACKING_OK` | VR 跟踪有效（帧年龄 < 200ms） | `controller.py:199-204` |
| 1 | `IK_SUCCESS` | IK 求解成功 | `controller.py:380` |
| 2 | `RETARGET_OK` | 手部重定向成功 | `controller.py:417` |
| 3 | `RETARGET_VALID` | 重定向结果在生理范围内 [-0.5, 2.5] | `controller.py:418` |
| 4 | `JOINT_JUMP_OK` | 关节跳变在限制内 (arm 5°, hand 10°) | `controller.py:449` |
| 5 | `IN_WORKSPACE` | EEF 在工作空间边界内 | `controller.py:308-309` |
| 7 | `ARM_TORQUE_OK` | 关节力矩 < per-joint 限制 [50,50,30,30,30,20,20] Nm（来自 URDF） | `safety.py:17-24` |
| 8 | `HAND_CURRENT_OK` | 手部电流 < 500 mA（阈值来源待 XHand datasheet 确认） | `safety.py:27-34` |
| 9 | `HAND_TEMP_OK` | 手部温度 < 70°C（阈值来源待 XHand datasheet 确认） | `safety.py:37-44` |
| 10 | `HAND_COMM_OK` | 手部通讯正常（无 board error） | `safety.py:47-48` |

> **Bit 6 (已移除)**：原 `CAMERA_OK` 定义了但从未在控制循环中 set，2026-06-18 移除。待相机帧新鲜度检查实现后再恢复。

**检查顺序**（`controller.py:197-219`）：
1. VR 帧 → 跟踪质量门控（`controller.py:199-204`）
2. 机器人状态读取（`controller.py:207-210`）
3. 动作计算 → IK/RETARGET/JUMP/WORKSPACE 标记（`controller.py:213`）
4. 硬件安全检查 → TORQUE/CURRENT/TEMP/COMM 标记（`controller.py:216-219`）
5. 关节限制硬安全检查（`controller.py:222-239`）

#### 3.4.2 紧急停止 (E-Stop)

触发条件（`controller.py:578-583`）：

| 触发源 | 条件 | 引用 |
|--------|------|------|
| 键盘 | ESC 键 → `EMERGENCY_STOP` 信号 | `controller.py:475-477` |
| VR 跟踪丢失 | 连续丢失超过阈值 | `controller.py:202-204` |
| 关节限制 | Arm 关节超出 `[qpos_min, qpos_max]` | `controller.py:222-231` |
| 机器人错误 | `robot.is_error()` 在 send_action 前为 True | `controller.py:255-257` |

**E-Stop 流程**：
- `_escalate_to_emergency(reason)` → 设状态为 `EMERGENCY_STOP`
- 调用 `robot.emergency_stop()` → `XArm7.stop()` + `XHand.stop()`（`controller.py:582, interface.py:235-236`）
- `self.running = False` 退出控制循环

#### 3.4.3 Per-Frame 安全检查

`safety.py` 提供无状态、共享的安全检查函数：

| 函数 | 阈值 | 失败处理 |
|------|------|---------|
| `check_arm_torque` | per-joint URDF 限制 [50, 50, 30, 30, 30, 20, 20] Nm | QUALITY flag → 数据标记为不良 |
| `check_hand_current` | 500 mA（参照 LEAP Hand Dynamixel 配置，待 XHand datasheet 确认） | QUALITY flag → 数据标记为不良 |
| `check_hand_temperature` | 70°C（阈值来源待 XHand datasheet 确认） | QUALITY flag → 数据标记为不良 |
| `check_hand_comm` | — | QUALITY flag → 数据标记为不良 |
| `check_arm_joint_limits` | `qpos_min/qpos_max` | **HARD** — 直接触发 E-Stop |
| `check_hand_joint_limits` | `qpos_min/qpos_max` | 警告（不触发 E-Stop） |
| `check_retarget_valid` | `[-0.5, 2.5]` | QUALITY flag → IK fallback |

#### 3.4.4 跳跃钳制

`_apply_jump_clamp()`（`controller.py:421-456`）限制相邻帧间命令跳变：

| 设备 | 限制 | 处理方式 |
|------|------|---------|
| Arm | 5°/帧 | `prev_cmd + clip(delta, -5°, 5°)` |
| Hand | 10°/帧 | `prev_cmd + clip(delta, -10°, 10°)` |

钳制后 `JOINT_JUMP_OK` 标记为 `False`，同时触发 `error_handler.hold_action()` 回退。

#### 3.4.5 频率控制与超时告警

`RateLimiter`（`utils/rate_limiter.py`）实现补偿计算耗时的频率控制：

- `wait()` 方法计算 `sleep_time = dt - elapsed`，只睡眠剩余时间
- 当周期超预算时发出节流警告（~1 次/秒），记录 overdue 比例
- 参照 `BunnyVisionPro wait_until_next_control_signal`

### 3.5 路径层安全

#### 3.5.1 10 步路径验证

`validate_path()`（`planning/planner.py:430-525`）对所有 plan_screw / plan_qpos 输出执行 10 项检查：

| # | 检查项 | 失败条件 | 行号 |
|---|--------|---------|------|
| 1 | 等效关节快照 | — | `planner.py:439` |
| 2 | 路径规范化到规划限制 | — | `planner.py:440` |
| 3 | Shortcut 平滑 | — | `planner.py:441` |
| 4 | 关节限制违反 | 任一路点超出 `planning_limits` | `planner.py:446-449` |
| 5 | **Elbow branch flip** | 路径中 elbow 跨越分支 | `planner.py:450-455` |
| 6 | 自碰撞 | 路径含自碰撞点 | `planner.py:456-462` |
| 7 | 环境碰撞 | 路径含环境碰撞点 | `planner.py:463-470` |
| 8 | 起点偏差 | `start_qpos_error > max_waypoint_delta_deg` | `planner.py:471-478` |
| 9 | 路点步长 | `max_waypoint_delta > max_waypoint_delta_deg` | `planner.py:479-482` |
| 10 | 终点误差 | 终点 EEF 位姿误差 > 阈值 | `planner.py:483-489` |
| — | 工作空间（可选） | 路点 EEF 超出 `workspace_bounds` | `planner.py:490-519` |

**碰撞检查细节**（`planning/ik_candidates.py:317-389`）：
- 路径段以 `collision_step_size=0.02`（L∞ 关节距离）进行稠密插值
- 每个插值点调用 `MPlib.check_for_self_collision` / `check_for_env_collision`
- 参照 dimos `collision_step_size=0.02`

#### 3.5.2 Elbow Flip 检测

`check_elbow_consistency()`（`planning/planner.py:566-577`）检测路径中 elbow（joint4）的分支翻转：

**判定条件**（同时满足）：
- `elbow_min < -5°`
- `elbow_max > 15°`
- `elbow_span > 45°`

**物理意义**：7-DOF 臂存在 elbow up/down 两个运动学分支。路径跨越分支时关节位移可能极大且不可预测，应拒绝。

#### 3.5.3 Shortcut 平滑

`shortcut_smooth_path()`（`planning/planner.py:381-402`）移除冗余路点：

- 对相邻路点 `prev → next`，检查是否可以直接跳跃中间点
- 验证条件：中点关节限制 + 稠密碰撞检查
- 迭代 3 轮，直到无更多冗余点

---

## 附录 A：核心数据流总览

```
VR Tracker ─→ TeleopController._tick() (50Hz)
                 │
                 ├→ TrackingQuality.check() ──→ TRACKING_OK
                 ├→ RobotInterface.get_state()
                 ├→ ArmMapper.map() ──→ 目标 EEF 位姿
                 ├→ Planner.solve_teleop_ik()
                 │     ├→ DLS diff IK (确定, ~1ms)
                 │     └→ Position IK (随机, ~10ms, fallback)
                 ├→ Retargeter.retarget() ──→ hand qpos
                 ├→ EMA smooth (arm only)
                 ├→ Safety checks (arm torque, hand current/temp/comm)
                 ├→ Jump clamp (arm 5°, hand 10°)
                 ├→ RobotInterface.send_action()
                 │     ├→ XArm7.send_action()
                 │     │     ├→ Mode 1: _limit_joint_range → _limit_joint_step → set_servo_angle_j
                 │     │     └→ Mode 4: store _arm_pos_target
                 │     │           250Hz 内环: PID → clip_velocity → vc_set_joint_velocity
                 │     └→ XHand.send_action()
                 ├→ QualityFlags (11-bit)
                 └→ Recorder.add_frame() (if RECORDING)
```

## 附录 B：关键文件索引

| 文件 | 核心类/函数 | 用途 |
|------|------------|------|
| `robot/xarm7/xarm7.py` | `XArm7`, `PIDController`, `XArm7Config` | 硬件驱动、PID 控制、双模控制 |
| `robot/xarm7/__init__.py` | — | 包导出（`PIDController`, `XArm7`, `XArm7Config`） |
| `robot/_connection_state.py` | `ConnectionStateMixin` | 连接状态基础属性（flag/error/message） |
| `robot/interface.py` | `RobotInterface`, `HandKinematics` | 统一接口、工作空间、回 home、桌面碰撞 |
| `planning/ik.py` | `TeleopIKSolver` | 遥操作 IK（DLS + Position IK 回退） |
| `planning/ik_candidates.py` | `IKCandidateManager` | 候选生成/过滤/评分/规范化 |
| `planning/kinematics.py` | `XArm7Kinematics` | FK、Jacobian、可操度、位姿变换 |
| `planning/planner.py` | `XArm7MotionPlanner`, `WorkspaceSafety` | 路径规划、路径验证、elbow flip、工作空间安全 |
| `planning/types.py` | `Pose`, `IKResult`, `PathResult`, `PlanningProfile`, `TeleopProfile` | 核心数据类型与配置 |
| `teleop/core/controller.py` | `TeleopController` | 状态机、质量标记、跳跃钳制、紧急停止 |
| `teleop/control/safety.py` | `check_arm_torque`, `check_hand_current`, … | Per-frame 安全检查（无状态） |
| `recording/quality_flags.py` | `QualityFlags` | 11-bit 质量标记位掩码 |
| `utils/rate_limiter.py` | `RateLimiter` | 频率控制 + 超时告警 |
