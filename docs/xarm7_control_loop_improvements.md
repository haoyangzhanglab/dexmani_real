# XArm7 控制回路优化方案

> 基于 Reference 目录 7 个参考代码库的机械臂控制方式调研  
> 日期: 2026-06-22  
> 实施状态: ✅ = 已实施, 🔲 = 待实施  
> 基准文档: `docs/xarm7_control_loop.md`

---

## 0. 调研范围

| 参考代码库 | 机械臂 | 核心文件 | 优先级 |
|-----------|--------|---------|--------|
| **BunnyVisionPro** | XArm7 (同款) | `real_control/xarm7_ability.py`, `teleop_bimanual_xarm7_ability.py` | **最高** |
| **ufactory_teleop** | XArm7 (同款) | `ufactory_devices/robot/uf_robot.py` | **最高** |
| **ManiUniCon** | XArm6/UR5/Franka | `maniunicon/core/robot.py`, `maniunicon/utils/ik_solver.py` | 高 |
| DexUMI | UR5 | — | 低（不同硬件） |
| Open-Teach | XArm/Franka/Kinova | `openteach/robot/bimanual.py` | 低（简单封装） |
| LeFranX | Franka FER | `franka_xhand_teleoperator/` | 低（不同硬件） |
| Bidex_Manus_Teleop | 无机械臂 | — | 不适用 |

---

## 1. P0 — 关键改进（影响稳定性/安全性）

### ❌ P0-1: 补全 Mode 6（内置轨迹规划）支持（已移除 — 用户决策暂不引入）

**来源**: ufactory_teleop `uf_robot.py:53,196-226`

**现状**: dexmani 实现了两套自定义控制模式：
- Mode 1 (Servo): 50Hz `set_servo_angle_j` + 自研瓶颈缩放
- Mode 4 (Velocity+PID): 250Hz PID 线程 + `vc_set_joint_velocity`

但 **xArm 控制器固件内置的 Mode 6（关节空间轨迹规划）未实现**。Mode 6 由 xArm 控制器自动处理加减速、轨迹平滑，是最"官方"的方案。

**对比分析**:

```python
# ufactory_teleop — Mode 6 实现（控制器内置轨迹平滑）
# uf_robot.py:196-226
if self.config.robot_mode == 6:
    robot_action = action[:7]
    jnt_spd = 0.2 if self._cmd_cnt < 20 else self._joint_speed  # 软启动
    code = self.arm.set_servo_angle(
        angle=robot_action, speed=jnt_spd, mvacc=self._joint_acc,
        is_radian=True, wait=False,
    )
```

**差异**: ufactory_teleop 把轨迹平滑完全交给 xArm 固件，只需传入 speed/accel 参数。这消除了 dexmani 维护自定义 PID 线程的复杂度（200+ 行代码）。

**建议实现**:

```python
# XArm7Config 增加
use_trajectory_mode: bool = False     # Mode 6: firmware trajectory planning
trajectory_speed: float = np.deg2rad(90)   # rad/s
trajectory_acc: float = np.deg2rad(500)    # rad/s²

# send_action 增加分支
if self.config.use_trajectory_mode:
    if self.arm.mode != 6:
        self._set_mode(6)
    code = self.arm.set_servo_angle(
        angle=target_qpos.tolist(),
        speed=self.config.trajectory_speed,
        mvacc=self.config.trajectory_acc,
        is_radian=True, wait=False,
    )
```

**预期影响**: 
- 消除 200+ 行 PID 线程代码（`_internal_control_arm_qpos` + `PIDController`）
- 固件级加减速比自定义 PID 更平滑（xArm 控制器内部使用更高频率的插值）
- 减少 CPU 负载（不需要 250Hz Python 线程）

**风险**: Mode 6 的 `speed`/`accel` 参数需要针对遥操作场景调优（默认值可能太慢或太快）。

---

### ✅ P0-2: Cartesian bounds 裁剪 + 重算 IK（替代直接 hold）

**来源**: ManiUniCon `maniunicon/core/robot.py:72-172`

**现状**: dexmani 的 `_compute_action()` 在 workspace 检查失败时直接 `hold_action()`（L429-432），意味着当 VR 手腕漂出工作空间边界时，手臂会立即停住。

```python
# controller.py:429-432 (现状 — 直接 hold)
if not in_workspace or not ori_ok:
    hold = self.error_handler.hold_action()
    arm_cmd = hold.arm_qpos_cmd
    hand_cmd = hold.hand_qpos_cmd
```

**参考做法**: ManiUniCon `_clip_action_to_bounds()` 将 TCP 位姿裁剪到 workspace 边界内，然后对裁剪后的位姿重新求解 IK。

```python
# ManiUniCon robot.py:72-172 (参考 — 裁剪 + 重算 IK)
tcp_position[0] = max(tcp_position[0], pos_bounds["x_min"])
tcp_position[0] = min(tcp_position[0], pos_bounds["x_max"])
# ... 对所有轴裁剪 ...
clipped_joint_positions = self.robot_interface.inverse_kinematics(
    tcp_position, tcp_orientation, clipped_action.joint_positions
)
```

**建议实现**:

在 `dexmani_real/robot/interface.py` 增加 `clamp_workspace_pose()`:

```python
def clamp_workspace_pose(self, target_pose: Pose) -> Pose:
    """将 EEF 目标位姿裁剪到工作空间边界内。"""
    clamped = Pose(p=target_pose.p.copy(), q=target_pose.q.copy())
    clamped.p = self.workspace.clamp(clamped.p)
    clamped.q = self.workspace.clamp_orientation(clamped.q)
    return clamped
```

在 `_compute_action()` 中修改 workspace 检查逻辑:
```python
# 修改前: workspace 失败 → hold
# 修改后: workspace 失败 → clamp pose → re-IK
if not in_workspace or not ori_ok:
    clamped_pose = self.robot.clamp_workspace_pose(arm_eef_pose)
    ik_result = self.planner.solve_teleop_ik(clamped_pose, state.arm_qpos, prev_arm_cmd)
    if ik_result.success:
        arm_cmd = ik_result.qpos
    else:
        hold = self.error_handler.hold_action()
        arm_cmd = hold.arm_qpos_cmd
```

**预期影响**: 手臂在边界处平滑停止而非急停，操作体验显著改善。

---

## 2. P1 — 重要改进（提升性能/体验）

### P1-1: `set_linear_spd_limit_factor(2.0)` — 恢复 Cartesian 跟踪速度

**来源**: ufactory_teleop `uf_robot.py:136`

**现状**: dexmani 连接后未设置线性速度限制因子。xArm 默认是 1.0×。

**差距**: ufactory_teleop 设置为 2.0×，对 Cartesian 跟踪速度有明显提升。

**建议实现**:

```python
# XArm7.connect() 或 robot_init() 中增加
self.arm.set_linear_spd_limit_factor(2.0)
```

或在 `XArm7Config` 增加配置项:
```python
linear_spd_limit_factor: float = 2.0  # 默认 2.0×
```

**预期影响**: Cartesian 伺服和回零路径的执行速度提升约 2×，在不牺牲安全性的前提下改善响应速度。

---

### ✅ P1-2: 集中化 `validate_action()` 方法

**来源**: ManiUniCon `maniunicon/core/robot.py:354` + `maniunicon/robot_interface/base.py`

**现状**: dexmani 的安全检查分散在 controller 和 interface 多处：
- `_compute_action()` 中的 workspace 检查 (controller.py:425-432)
- `_tick()` 中的 pre-send safety gate (controller.py:358-378)
- `safety.check_arm_torque` 等函数 (safety.py)
- `_limit_joint_range` / `_limit_joint_step` (driver 层)

**对比**: ManiUniCon 有集中化的 `robot_interface.validate_action(action) → bool`，在 send_action 之前统一执行。

**建议实现**:

在 `RobotInterface` 增加:
```python
def validate_action(self, action: RobotAction, quality_flags: int) -> tuple[bool, str]:
    """集中化的 pre-send 验证，返回 (ok, reason)。"""
    # 1. 连接检查
    if self.is_error():
        return False, "robot error state"
    # 2. 关节范围（已在 driver 层处理，此处为二次确认）
    if not self.arm.is_connected():
        return False, "arm not connected"
    # 3. 软故障（力矩/电流/温度）
    if not (quality_flags & ARM_TORQUE_OK):
        return False, "arm torque exceeded"
    if not (quality_flags & HAND_CURRENT_OK):
        return False, "hand current exceeded"
    if not (quality_flags & HAND_TEMP_OK):
        return False, "hand temperature exceeded"
    # 4. 工作空间
    arm_eef = self.kinematics.compute_eef_pose_world(action.arm_qpos_cmd)
    if not self.check_workspace(arm_eef.p):
        return False, "workspace violation"
    if not self.check_workspace_orientation(arm_eef.q):
        return False, "orientation bounds violation"
    return True, "ok"
```

**预期影响**: 降低安全检查遗漏风险，简化 controller 层代码（`_tick()` 中 20+ 行安全检查可压缩到 3-5 行）。

---

### ✅ P1-3: 控制循环超时警告

**来源**: BunnyVisionPro `xarm7_ability.py:233-244`

**现状**: dexmani 使用 `RateLimiter` 做速率控制，但当循环耗时超过目标周期时静默跳过。

**差距**: BVPro 的 `wait_until_next_control_signal()` 在无法达到目标 dt 时发出 `warnings.warn`，便于发现性能退化。

**建议实现**:

在 `TeleopController._tick()` 末尾增加:
```python
# 在 limiter.wait() 之后检查实际循环耗时
actual_dt = time.perf_counter() - tick_start
if actual_dt > self.limiter.period * 1.5:  # 超过目标 150%
    logger.warning(
        "Control loop overrun: actual=%.1fms target=%.1fms",
        actual_dt * 1000, self.limiter.period * 1000,
    )
```

**预期影响**: 早期发现性能问题（如 IK 求解变慢、GC 停顿），便于诊断和优化。

---

### P1-4: BunnyVisionPro 风格的显式初始化收敛循环

**来源**: BunnyVisionPro `teleop_bimanual_xarm7_ability.py:144-174`

**现状**: dexmani 用时间/帧计数软启动（`soft_start_frames=20`, `pid_soft_start_duration_s=0.4`）。速度模式有额外的收敛阈值（`pid_convergence_threshold_rad=2°`），但在 servo 模式下没有。

**差距**: BVPro 的双臂遥操作在初始化阶段显式循环等待收敛：
```python
# BVPro teleop_bimanual_xarm7_ability.py:167-174
while np.linalg.norm(error) > error_threshold:
    left_robot.wait_until_next_control_signal()
    qpos_list = client.get_teleop_cmd()
    left_robot.control_arm_qpos(qpos_list[0][0:7])
    left_error = qpos_list[0][0:7] - left_robot.get_arm_qpos()
```

同时将所有速度限制降到 1/3，收敛后再恢复。

**建议实现**:

在 `XArm7` 增加显式收敛方法（增强现有 `reset_soft_start`）:
```python
def converge_to_target(self, target: np.ndarray, timeout_s: float = 5.0) -> bool:
    """主动轮询等待手臂收敛到目标位置。
    
    在 TELEOP 切换时使用: 先降到 30% 速度，收敛后再恢复 100%。
    """
    threshold = np.deg2rad(2.0)
    deadline = time.perf_counter() + timeout_s
    self._vel_ramp_start = time.perf_counter()  # 激活 30% 速度限制
    
    while time.perf_counter() < deadline:
        error = np.max(np.abs(self._read_qpos() - target))
        if error < threshold:
            self._vel_ramp_start = None  # 释放全速
            return True
        time.sleep(self.config.dt)
    return False
```

**预期影响**: 消除 TELEOP 切入瞬间的速度跳变，操作体验更平滑。

---

## 3. P2 — 锦上添花（长期优化方向）

### P2-1: 状态读取与控制解耦（独立线程）

**来源**: ManiUniCon `maniunicon/core/robot.py:174-220`

ManiUniCon 将 state 读取放在独立线程 `_state_receiver_thread` (50Hz)，控制循环在另一个线程 (200Hz)。这避免了 `get_state()` 的延迟阻塞控制命令发送。

dexmani 目前是单线程顺序: `get_state()` → `_compute_action()` → `send_action()`。

**适用条件**: 仅在切换到 Mode 6（轨迹模式）后有实际收益。Mode 1/4 已经将速率限制下放到了驱动层和 PID 内环。

### P2-2: Pink QP-based IK（PostureTask 姿态正则化）

**来源**: ManiUniCon `maniunicon/utils/ik_solver.py:103-124`

ManiUniCon 使用 Pink 的 `FrameTask(EEF) + PostureTask(joint)` 做 QP 求解，PostureTask 会自然地将冗余自由度拉向默认姿态，避免肘关节漂移。

dexmani 当前仅通过 IK candidate 的硬件最近原则（LeFranX current_distance penalty）间接处理冗余自由度。

**适用条件**: 需要引入 Pink/QP 依赖，适合作为长期架构升级方向。短期内现有的硬件最近候选策略已足够。

### P2-3: Mode 7（笛卡尔在线轨迹规划）支持

**来源**: ufactory_teleop `uf_robot.py:219-222`

xArm 还支持 Mode 7（笛卡尔在线轨迹规划），接收 6D 位姿直接执行。这对某些应用（如末端轨迹精确控制）可能更合适。

**适用条件**: 作为可选模式保留，遥操作场景下关节空间控制（Mode 1/4/6）通常更直接。

---

## 4. 现状优势确认（不需要改的）

以下功能 dexmani 已经超越或对齐参考代码库，无需优化：

| 功能 | dexmani 现状 | 参考对比 | 结论 |
|------|------------|---------|------|
| 关节范围裁剪 | `_limit_joint_range` (np.clip) | BVPro 无、ufactory_teleop 无 | ✅ 更安全 |
| Mode-0 安全切换 | `_set_mode()` 0→target→验证 | BVPro 直接 set_mode×2, ufactory 直接 set_mode | ✅ 更鲁棒 |
| 错误状态追踪 | `error_state` + `last_error_message` + `last_sdk_error_code` | BVPro 仅 `use_arm=False`, ufactory 返回 error_code | ✅ 更可诊断 |
| 瓶颈缩放限速 | Arm: 等比缩放保持轨迹形状 | BVPro 同策略 | ✅ 对齐 |
| 软启动速度渐变 | Servo: 0.3→max 线性渐变 | ufactory_teleop 仅常量 0.2 | ✅ 更平滑 |
| DT 天花板 | `dt = clamp(dt_raw, dt, dt*10)` | BVPro 无天花板 | ✅ 更安全 |
| 预发送安全门 | 力矩/电流/温度 三层 | ManiUniCon 仅 workspace | ✅ 更全面 |
| 硬件位置参考 | delta 以 hw_qpos 为基准 | BVPro 同策略 | ✅ 对齐 |
| 四元数连续化 | `continuous_quat()` 防翻转 | ManiUniCon 无 | ✅ 独有优势 |
| C31 碰撞防护 | TCP load + sensitivity=0 | ufactory_teleop sensitivity=0 同 | ✅ 对齐 |
| QualityFlags 11-bit | 录制时标记每帧质量（含 CAMERA_OK） | 所有参考均无 | ✅ 独有优势 |

---

## 5. 实施路线图

```
Phase A (✅ 已完成): P0 关键改进
  ├── ✅ P0-1: 实现 Mode 6 支持 (XArm7Config + send_action + robot_init + reset)
  └── ✅ P0-2: Cartesian bounds 裁剪 + 重算 IK (RobotInterface.clamp_workspace_pose + controller)

Phase B (✅ 已完成): P1 重要改进
  ├── ✅ P1-2: 集中化 validate_action() (RobotInterface.validate_action + controller 简化)
  ├── ✅ P1-3: 控制循环超时警告 (tick_start → tick_elapsed_ms → logger.warning)
  ├── 🔲 P1-1: set_linear_spd_limit_factor (仅 Mode 7 笛卡尔模式相关，列入 P2-3)
  └── 🔲 P1-4: 显式初始化收敛循环 (velocity 模式已有 pid_convergence_threshold_rad [B3] 覆盖)

Phase C (🔲 后续): P2 锦上添花
  ├── 🔲 P2-1: State 读取线程解耦
  ├── 🔲 P2-2: Pink QP-based IK (PostureTask)
  └── 🔲 P2-3: Mode 7 笛卡尔模式 (+ set_linear_spd_limit_factor)
```

---

## 6. 参考文件清单

| 文件 | 来源 | 关键内容 |
|------|------|---------|
| `Reference/BunnyVisionPro/real_control/xarm7_ability.py` | BVPro | PID + 速度控制 + clip_arm_velocity + clip_arm_next_qpos |
| `Reference/BunnyVisionPro/real_control/teleop_bimanual_xarm7_ability.py` | BVPro | 初始化收敛循环 + 1/3 速度限制 |
| `Reference/ufactory_teleop/ufactory_devices/robot/uf_robot.py` | ufactory | Mode 6/7 + soft-start + spd_limit_factor + robot_init |
| `Reference/ManiUniCon/maniunicon/core/robot.py` | ManiUniCon | 多进程/多线程架构 + validate_action + bounds clipping + pose interpolator |
| `Reference/ManiUniCon/maniunicon/utils/ik_solver.py` | ManiUniCon | QP-based IK + FrameTask + PostureTask |
