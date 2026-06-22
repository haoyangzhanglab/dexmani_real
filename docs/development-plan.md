# VR 遥操作代码开发方案

> **日期**: 2026-06-22 | **基于**: 5 框架对比分析 + 专项对比 + 控制回路设计文档 + 源码审查报告
> **参考项目**: ManiUniCon, LeFranX, BunnyVision Pro, Open-Teach

---

## 目录

1. [方案总览](#1-方案总览)
2. [Phase 0: 文档修正 (立即)](#2-phase-0-文档修正)
3. [Phase 1: 安全关键修复 (P0)](#3-phase-1-安全关键修复)
4. [Phase 2: 运动质量提升 (P1)](#4-phase-2-运动质量提升)
5. [Phase 3: 架构增强 (P2)](#5-phase-3-架构增强)
6. [Phase 4: 工程优化 (P3)](#6-phase-4-工程优化)
7. [附录: 完整文件变更清单](#7-附录)

---

## 1. 方案总览

### 1.1 优先级矩阵

| Phase | 优先级 | 改进项数 | 预计工作量 | 风险等级 | 依赖 |
|-------|--------|---------|-----------|---------|------|
| **Phase 0** | 立即 | 6 | 1 天 | 低 | 无 |
| **Phase 1** | P0 | 4 | 1-2 天 | 低 | Phase 0 |
| **Phase 2** | P1 | 6 | 1-2 周 | 中 | Phase 1 |
| **Phase 3** | P2 | 4 | 2-4 周 | 中-高 | Phase 2 |
| **Phase 4** | P3 | 3 | 1-3 天 | 低 | Phase 2 |

### 1.2 关键设计原则

1. **向后兼容**: 所有新功能通过可选参数/配置项引入，默认行为不变
2. **渐进式集成**: 每项改进可独立启用/禁用，支持 A/B 测试
3. **参考优先**: 优先参考 Reference 项目中已验证的实现，再做适配
4. **安全不可妥协**: Phase 1 安全改进必须在其他 phase 之前完成
5. **测试驱动**: 每项改进先在 `dry_run=True` 模式验证，再上真机

### 1.3 依赖关系图

```
Phase 0 (文档修正)
  └─► Phase 1.1 (max_record_frames) ─── 独立，无依赖
  └─► Phase 1.2 (orientation bounds) ── 依赖 Phase 0 的 WorkspaceSafety 理解
  └─► Phase 1.3 (VR delta rot cap) ──── 独立，仅改 arm_mapper.py
  └─► Phase 1.4 (tracking soft decel) ─ 独立，仅改 controller.py

Phase 1
  └─► Phase 2.1 (adaptive damping) ──── 需要 Phase 1.2 的配置扩展经验
  └─► Phase 2.2 (Cartesian interpolator) ─ 需要 Phase 1.3 的 arm_mapper 理解
  └─► Phase 2.3 (validate_action) ───── 需要 Phase 1.2 的 orientation bounds
  └─► Phase 2.4 (REARM) ─────────────── 独立，改 controller + keyboard
  └─► Phase 2.5 (DLS-only mode) ─────── 独立，仅改 config
  └─► Phase 2.6 (manipulability scoring) ─ 需要 Phase 2.1 的 adaptive damping

Phase 2
  └─► Phase 3.1 (ZMQ process sep) ───── 独立，新模块
  └─► Phase 3.2 (camera daemon) ──────── 依赖 Phase 3.1 的进程模型
  └─► Phase 3.3 (LeRobot/Zarr export) ── 独立，新脚本
  └─► Phase 3.4 (Lock-Free IPC) ──────── 依赖 Phase 3.1，可选替代 ZMQ

Phase 3
  └─► Phase 4.1 (config round-trip) ──── 独立
  └─► Phase 4.2 (Docker retargeting) ─── 独立
  └─► Phase 4.3 (dual-hand support) ──── 独立，但影响面大
```

---

## 2. Phase 0: 文档修正

> **目标**: 修正 `docs/code-review-design-docs.md` 中发现的 6 个高/中严重度文档错误
> **工作量**: 1 天 | **风险**: 低

### 2.1 H1: Layer 2 错误处理描述修正

**文件**: `docs/vr-teleop-control-loop-design.md` §6.1

**错误**: 文档写 `TRACKING_OK=0` 时"继续管道（发送 hold cmd）"，实际代码 `controller.py:211` 是 `return` 立即退出 `_tick()`，不发送任何指令。

**修复**:
```markdown
# 修改前
TRACKING_OK=0 (stale, 非连续丢失):
  → 继续管道（发送 hold cmd）

# 修改后
TRACKING_OK=0 (stale, 非连续丢失):
  → return 立即退出 _tick()，本帧不发送任何指令
  → 机器人自然保持在之前发送的最后指令位置
```

### 2.2 H2: Record/Execute 顺序修正

**文件**: `docs/vr-teleop-control-loop-design.md` §3.1 图、§5.2 伪代码、§8.1 表

**错误**: 文档将 Execute 放在 Record 之前（Stage 6 → Stage 7），实际代码 `controller.py:257-276` Record 在 Execute **之前**执行。

**修复**:
```markdown
# §3.1 管道图修改
Stage 6 (Record) → Stage 7 (Execute)

# §5.2 伪代码修改
├─ 6. Record (RECORDING state only, 在 Execute 之前)
│     └─ recorder.add_frame(state, action, vr, flags, T_base_eef)
├─ 7. Execute action (non-IDLE states only)

# §8.1 表修改
| 6. Record Frame  | 0.5ms | 录制的是本帧计算出的 action（不等硬件执行结果） |
| 7. Execute Action| 1.0ms | 发送指令到硬件 |
```

### 2.3 M1: 阶段数量一致性

**文件**: `docs/vr-teleop-control-loop-design.md` §3.1

**修复**: 统一为 7 阶段（与 `_tick()` 代码中的 7 个注释块对应），删除"共 8 个阶段"的说法。

### 2.4 M2: Quality Flags 描述修正

**文件**: `docs/vr-teleop-control-loop-design.md` §3.6

**修复**: 改为描述性文字：
- bits 0-5 (TRACKING/IK/RETARGET/JUMP/WORKSPACE) 在 `_compute_action()` 内增量设置
- bits 7-10 (TORQUE/CURRENT/TEMP/COMM) 在 `_tick()` step 5 设置
- Quality Flags 不是独立的单一步骤，而是在管道各步骤增量设置后聚合

### 2.5 BM1/BM2: framework-comparison.md BVPro 修正

**文件**: `docs/vr-teleop-framework-comparison.md` §2.2/§6

**修复**:
- BM1: `XArm7AbilityRobot` → `XArm7Ability`
- BM2: `control_arm_qpos()` 行号 `:230` → `:196-198`；`:200-228` 标注为 `_internal_control_arm_qpos()`（PID 内环）；`:230` 改为 `control_hand_qpos()`

---

## 3. Phase 1: 安全关键修复

> **目标**: 修复 4 个 P0 级安全和可靠性缺陷
> **工作量**: 1-2 天 | **风险**: 低（均为增量修改，不影响现有流程）

---

### 3.1 P0-1: 录制帧数硬上限 max_record_frames

**参考**: ManiUniCon `max_record_steps=5000`

**问题**: `EpisodeRecorder` 无帧数上限，操作员忘记停止会导致磁盘耗尽。

**实施方案**:

#### Step 1: 修改 `EpisodeRecorder.__init__`

**文件**: `dexmani_real/recording/episode_recorder.py:37-58`

```python
class EpisodeRecorder:
    def __init__(
        self,
        data_dir: str,
        max_frames: int = 10000,  # NEW: hard cap per episode
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames  # NEW
        # ... rest unchanged
```

#### Step 2: 在 `add_frame()` 中添加 early-stop

**文件**: `dexmani_real/recording/episode_recorder.py:120` (add_frame 开头)

```python
def add_frame(self, state, action, vr_frame, quality_flags, camera_frame=None, T_base_eef=None) -> bool:
    if not self._recording or self._file is None:
        return False
    
    # NEW: Auto-stop on max frame count
    if self._frame_count >= self.max_frames:
        logger.warning("Episode reached max_frames=%d, auto-stopping.", self.max_frames)
        self._file.attrs["stopped_reason"] = "max_frames"
        self.stop_episode(success=True)
        return True
    
    # ... rest unchanged
```

#### Step 3: 在 PipelineConfig 中添加配置项

**文件**: `dexmani_real/config/pipeline_config.py`

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    max_record_frames: int = 10000  # NEW
```

**测试**:
- 单元测试: 创建 recorder, 设置 `max_frames=5`, add 6 帧, 验证自动停止且 `stopped_reason="max_frames"`
- 集成测试: dry_run 模式下录制, 验证 auto-stop 不崩溃

---

### 3.2 P0-2: EEF 方向工作空间边界 (Orientation Workspace Bounds)

**参考**: ManiUniCon `quest_controller.py:134-158 _check_safety_limits()` + `:160-190 _apply_safety_limits()`

**问题**: `WorkspaceSafety` 仅检查 EEF 位置 (x, y, z)，缺少方向检查。wrist 极值姿态可能导致自碰撞。

**实施方案**:

#### Step 1: 扩展 WorkspaceSafety 类

**文件**: `dexmani_real/planning/planner.py` (WorkspaceSafety 类, ~:638-665)

在现有 WorkspaceSafety 类中添加方向检查方法:

```python
class WorkspaceSafety:
    """EEF workspace bounds checking and clamping.
    
    workspace_bounds: (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters.
    orientation_bounds: (3, 2) array [[roll_min, roll_max], [pitch_min, pitch_max], [yaw_min, yaw_max]]
        in radians. None disables orientation checking (backward compatible).
    """

    def __init__(
        self,
        workspace_bounds: np.ndarray,
        orientation_bounds: np.ndarray | None = None,
    ) -> None:
        self.bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if self.bounds.shape != (3, 2):
            raise ValueError(f"workspace_bounds must have shape (3, 2), got {self.bounds.shape}.")
        self.ori_bounds = (
            None if orientation_bounds is None
            else np.asarray(orientation_bounds, dtype=np.float64)
        )

    # ... existing check() and clamp() methods unchanged ...

    def check_orientation(self, eef_quat_wxyz: np.ndarray) -> bool:
        """Check whether EEF orientation (as Euler XYZ) is within orientation bounds."""
        if self.ori_bounds is None:
            return True
        from dexmani_real.planning.pose_utils import wxyz_to_xyzw
        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        return bool(
            np.all(euler >= self.ori_bounds[:, 0])
            and np.all(euler <= self.ori_bounds[:, 1])
        )

    def clamp_orientation(self, eef_quat_wxyz: np.ndarray) -> np.ndarray:
        """Clip EEF orientation to orientation bounds. Returns clamped quat (wxyz)."""
        if self.ori_bounds is None:
            return np.asarray(eef_quat_wxyz, dtype=np.float64)
        from dexmani_real.planning.pose_utils import wxyz_to_xyzw, xyzw_to_wxyz
        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        euler = np.clip(euler, self.ori_bounds[:, 0], self.ori_bounds[:, 1])
        clamped_quat_xyzw = Rotation.from_euler('XYZ', euler).as_quat()
        return xyzw_to_wxyz(clamped_quat_xyzw)
```

#### Step 2: 在 RobotInterfaceConfig 中添加配置

**文件**: `dexmani_real/robot/types.py:111-151`

```python
@dataclass
class RobotInterfaceConfig:
    # ... existing fields ...
    # Orientation workspace bounds (Euler XYZ, radians).
    # None disables orientation checking (backward compatible).
    workspace_orientation_bounds: np.ndarray | None = None
```

#### Step 3: 在 RobotInterface 中添加方向检查方法

**文件**: `dexmani_real/robot/interface.py` (在 `check_workspace()` 附近)

```python
def check_workspace_orientation(self, eef_quat_wxyz: np.ndarray) -> bool:
    """Check EEF orientation against orientation bounds."""
    ws = getattr(self, '_workspace_safety', None)
    if ws is None:
        return True
    return ws.check_orientation(eef_quat_wxyz)
```

#### Step 4: 在 controller._compute_action() 中集成

**文件**: `dexmani_real/teleop/core/controller.py` (_compute_action 中 workspace check 之后)

```python
# 现有: workspace check
arm_eef_pos = self.planner.compute_eef_pose_world(arm_cmd).p
in_workspace = self.robot.check_workspace(arm_eef_pos)

# NEW: orientation workspace check
ori_ok = self.robot.check_workspace_orientation(arm_eef_quat)

quality.set(IN_WORKSPACE, in_workspace and ori_ok)
```

**测试**:
- 单元测试: 测试 `check_orientation()` 正常/越界/None bounds 三种情况
- 单元测试: 测试 `clamp_orientation()` 裁切到边界
- 集成测试: dry_run 模式下验证 flag 设置正确

---

### 3.3 P0-3: VR Per-Step 旋转 Delta 安全限制

**参考**: ManiUniCon `max_delta_rot=1.0rad` (`quest_controller.py:160-190`)

**问题**: `ArmWristMapper` 有 `eef_delta_bounds` 用于位置，但无 per-frame 旋转 delta cap。VR 跟踪 glitch 产生的大角度跳变直接送入 IK。

**实施方案**:

#### Step 1: 在 ArmWristMapper 中添加旋转 delta 限制

**文件**: `dexmani_real/teleop/vr/arm_mapper.py:16-28 (__init__) + :49-75 (map)`

```python
class ArmWristMapper:
    def __init__(
        self,
        pos_scale: float = 1.0,
        rot_scale: float = 1.0,
        vr_to_base_rot: np.ndarray | None = None,
        eef_delta_bounds: np.ndarray | None = None,
        max_delta_rot_rad: float = 1.0,  # NEW: ~57°, catches VR tracking glitches
    ) -> None:
        # ... existing fields ...
        self.max_delta_rot_rad = max_delta_rot_rad  # NEW

    def _clip_delta_rot(self, delta_rot: np.ndarray) -> np.ndarray:
        """Clamp per-frame rotation delta to prevent VR tracking glitches."""
        axis, angle = mat2axangle(delta_rot)
        if angle > self.max_delta_rot_rad:
            logger.debug(
                "clip_delta_rot: clamping %.3f rad -> %.3f rad",
                angle, self.max_delta_rot_rad,
            )
            return axangle2mat(axis, self.max_delta_rot_rad, is_normalized=True)
        return delta_rot

    def map(self, wrist_pos, wrist_quat_wxyz):
        # ... existing code up to scale_rot ...
        delta_rot_vr = self.scale_rot(delta_rot_vr)
        delta_rot_vr = self._clip_delta_rot(delta_rot_vr)  # NEW: add here
        delta_rot_base = self.vr_to_base_rot @ delta_rot_vr @ self.vr_to_base_rot.T
        # ... rest unchanged ...
```

**测试**:
- 单元测试: 构造超过 `max_delta_rot_rad` 的旋转, 验证被 clamp
- 单元测试: 正常旋转不受影响
- 集成测试: dry_run 模式下用 DummyTracker 模拟大跳变

---

### 3.4 P0-4: VR 跟踪丢失软减速保持

**参考**: BVPro `clip_arm_velocity()` 初始化降速策略

**问题**: 当前连续丢失 >1.0s 直接 E-Stop，但短暂丢失（<1s）会导致急停抖动。

**实施方案**:

#### Step 1: 修改 _tick() 中的丢失处理逻辑

**文件**: `dexmani_real/teleop/core/controller.py:_tick()` (~:207-211)

```python
# 当前代码:
if not tq_result.ok:
    self.error_handler.record_failure("vr_stale")
    if tq_result.tracking_lost:
        self._escalate_to_emergency("VR tracking lost > 1.0s")
    return

# 修改为:
if not tq_result.ok:
    self.error_handler.record_failure("vr_stale")
    if tq_result.tracking_lost:
        lost_dur = tq_result.lost_duration_s
        if lost_dur >= 1.0:
            self._escalate_to_emergency("VR tracking lost > 1.0s")
        else:
            # Soft deceleration: 指数衰减到 current_qpos
            decay = np.exp(-lost_dur * 3.0)  # ~5% remaining at 1s
            state = self.robot.get_state() if not self.dry_run else self._dummy_state()
            if self._last_arm_cmd is not None and state is not None:
                # Gradually pull toward current position
                arm_interp = (
                    decay * self._last_arm_cmd
                    + (1.0 - decay) * state.arm_qpos
                )
                # Send soft hold command (position hold, no new IK)
                action = RobotAction(
                    arm_qpos_cmd=arm_interp,
                    hand_qpos_cmd=(
                        self._last_hand_cmd if self._last_hand_cmd is not None
                        else state.hand_qpos
                    ),
                )
                if not self.dry_run:
                    self.robot.send_action(action)
    return
```

**测试**:
- 单元测试: 模拟 TrackingQuality 返回 `tracking_lost=True` 的各种 duration 值
- 集成测试: dry_run 模式下注入虚拟跟踪丢失事件，观察命令变化曲线
- 真机测试: 用手遮挡 Quest 摄像头 0.5s, 观察机器人是否平滑减速而非急停

---

## 4. Phase 2: 运动质量提升

> **目标**: 实现 6 个 P1 级功能增强
> **工作量**: 1-2 周 | **风险**: 中（涉及核心控制回路修改）

---

### 4.1 P1-1: 近奇点自适应 DLS 阻尼

**参考**: ManiUniCon QP 求解器近零阻尼 (1e-12) + dexmani 现有 `kin.compute_manipulability()`

**问题**: 固定 damping=0.02 在非奇异区域产生 ~1-2mm 持续跟踪偏差。`compute_manipulability()` 已实现但未在 teleop 热点使用。

**实施方案**:

#### Step 1: 在 TeleopProfile 中添加自适应阻尼配置

**文件**: `dexmani_real/planning/types.py:106-133`

```python
@dataclass(kw_only=True)
class TeleopProfile:
    # ... existing fields ...
    
    # Adaptive damping (NEW)
    adaptive_damping: bool = False  # 默认关闭，向后兼容
    differential_ik_min_damping: float = 0.001   # 非奇异区低阻尼
    differential_ik_max_damping: float = 0.05    # 近奇异区高阻尼
    manipulability_threshold: float = 0.005       # 可操作性阈值
```

#### Step 2: 修改 solve_differential_ik() 使用自适应阻尼

**文件**: `dexmani_real/planning/ik.py:solve_differential_ik()` (~:259-262)

```python
def solve_differential_ik(self, target, current, prev, profile):
    # ... existing code: compute jacobian, error ...
    
    # NEW: Adaptive damping
    if profile.adaptive_damping:
        mu = self.kin.compute_manipulability(current)
        threshold = profile.manipulability_threshold
        if mu > threshold:
            damping = profile.differential_ik_min_damping
        elif mu < 1e-6:  # effectively singular
            damping = profile.differential_ik_max_damping
        else:
            # Linear interpolation between min and max
            ratio = mu / threshold
            damping = (
                profile.differential_ik_min_damping
                + (profile.differential_ik_max_damping - profile.differential_ik_min_damping)
                * (1.0 - ratio)
            )
    else:
        damping = float(profile.differential_ik_damping)
    
    # Use adaptive damping in DLS formula
    lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(6)
    # ... rest unchanged ...
```

**参考实现**: ManiUniCon 的 QP 求解器自然处理秩亏 (damping=1e-12)，dexmani 无法直接搬用 QP 但可借鉴其"正常区域低阻尼、奇异区域高阻尼"的哲学。

**测试**:
- 单元测试: 在已知非奇异配置下验证 damping=min，奇异配置下 damping=max
- 仿真测试: SAPIEN 中对比固定 vs 自适应 damping 的跟踪精度
- 真机测试: 比较正常运动和近奇异运动下的平滑度

---

### 4.2 P1-2: Cartesian Pose 插值器

**参考**: ManiUniCon `pose_trajectory_interpolator.py:78-207`

**问题**: 50Hz 控制循环在 VR 更新 25-30Hz 时会重读同一 VR 帧（stale reuse），导致不流畅运动。

**实施方案**:

#### Step 1: 新建 CartPoseInterpolator 类

**新文件**: `dexmani_real/teleop/vr/pose_interpolator.py`

完整的 `CartPoseInterpolator` 类（~150 行），核心逻辑:

```python
"""Cartesian pose interpolator — smooths between discrete VR frames.

Ref: ManiUniCon PoseTrajectoryInterpolator (pose_trajectory_interpolator.py:78-207).
Key differences from ManiUniCon version:
  - Simplified: no future waypoint queue (VR is 50Hz native, not 30Hz)
  - Integrated with ArmWristMapper output (wxyz quat convention)
  - Optional: disabled by default, configurable via TeleopProfile
"""

class CartPoseInterpolator:
    """Interpolates between discrete VR-frame poses for smooth robot motion.
    
    Receives target poses at VR rate and produces interpolated poses at the
    controller's sampling rate via:
      - Linear interpolation for position
      - SLERP for rotation
      - Speed-limited temporal scheduling
    
    Usage in controller:
      # In _compute_arm_command(), after arm_mapper.map():
      interpolator.push_target_pose(target_pos, target_quat_wxyz)
      result = interpolator.get_interpolated_pose()
      if result is not None:
          target_pos, target_quat_wxyz = result
      # Then feed into IK as usual
    """

    def __init__(
        self,
        max_pos_speed: float = 0.25,    # m/s
        max_rot_speed: float = 0.5,     # rad/s
        max_history: int = 5,
    ) -> None:
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self._waypoints: deque[tuple[float, np.ndarray, np.ndarray]] = (
            deque(maxlen=max_history)
        )
        self._last_pos: np.ndarray | None = None
        self._last_rot: Rotation | None = None
        self._earliest_arrival_time: float = 0.0

    def push_target_pose(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Enqueue a new target waypoint (called at VR frame rate)."""
        # ... see maniunicon-comparison.md P1-1 for full implementation ...

    def get_interpolated_pose(
        self, now: float | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get interpolated pose at current time (called at controller rate).
        
        Returns:
            (pos, quat_wxyz) or None if no waypoints available.
        """
        # ... linear pos + SLERP rot interpolation ...

    def reset(self) -> None:
        """Clear all waypoints (on state transition)."""
        self._waypoints.clear()
        self._last_pos = None
        self._last_rot = None
        self._earliest_arrival_time = 0.0
```

> 完整代码见 `docs/maniunicon-comparison.md` §4 P1-1（已验证的 ManiUniCon 参考实现）。

#### Step 2: 在 TeleopProfile 中添加配置

**文件**: `dexmani_real/planning/types.py`

```python
@dataclass(kw_only=True)
class TeleopProfile:
    # ... existing fields ...
    use_cartesian_interpolation: bool = False  # NEW: 默认关闭
    interpolation_max_pos_speed: float = 0.25  # NEW: m/s
    interpolation_max_rot_speed: float = 0.5   # NEW: rad/s
```

#### Step 3: 在 TeleopController 中集成

**文件**: `dexmani_real/teleop/core/controller.py`

```python
class TeleopController:
    def __init__(self, ..., use_cartesian_interpolation: bool = False):
        # ... existing init ...
        self._pose_interpolator = (
            CartPoseInterpolator() if use_cartesian_interpolation else None
        )

    def _compute_arm_command(self, vr_frame, state, prev_arm_cmd, quality):
        # ... existing: arm_mapper.map() ...
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

            # ... rest: build Pose, solve IK ...
```

**测试**:
- 单元测试: 测试插值器线性位置 + SLERP 旋转正确性
- 单元测试: 测试 speed-limited scheduling（验证超过速度限制时 waypoint 的 earliest_arrival_time）
- 集成测试: dry_run 模式 + DummyTracker, 对比插值前后的命令序列平滑度
- 真机测试: A/B 对比（启用/禁用插值）的运动平滑度

---

### 4.3 P1-3: 集中化 validate_action() 安全检查门

**参考**: ManiUniCon `validate_action()` 统一门模式

**问题**: 安全检查分散在 controller.py 的 4 个位置，可能遗漏或重复。

**实施方案**:

#### Step 1: 在 RobotInterface 中添加 validate_action()

**文件**: `dexmani_real/robot/interface.py` (在 send_action 之前)

```python
def validate_action(
    self,
    action: RobotAction,
    state: RobotState,
    check_self_collision: bool = False,
) -> tuple[bool, dict[str, bool]]:
    """Centralized pre-send safety gate.
    
    Returns:
        (all_ok, details) where details maps check_name -> passed.
    
    Checks:
      1. arm_qpos within joint limits
      2. hand_qpos within joint limits
      3. EEF position within workspace bounds
      4. EEF orientation within orientation bounds (if configured)
      5. Arm torque within limits (from state observation)
      6. Hand current within limits (from state observation)
    """
    details: dict[str, bool] = {}
    
    # 1. Arm joint limits (hard check — E-Stop level)
    arm_joint_ok = safety.check_arm_joint_limits(
        state, self.arm.config.qpos_min, self.arm.config.qpos_max
    )
    details["arm_joint_limits"] = arm_joint_ok
    if not arm_joint_ok:
        return False, details
    
    # 2. Hand joint limits (warning level)
    hand_joint_ok = safety.check_hand_joint_limits(
        state, self.hand.config.qpos_min, self.hand.config.qpos_max
    )
    details["hand_joint_limits"] = hand_joint_ok
    
    # 3. Workspace position
    eef_pos = self._compute_eef_position(action.arm_qpos_cmd)
    in_ws = self.check_workspace(eef_pos)
    details["in_workspace"] = in_ws
    
    # 4. Workspace orientation (if configured)
    ori_ok = self.check_workspace_orientation(action.target_eef_rot6d)
    details["orientation_ok"] = ori_ok
    
    # 5-6. Hardware state checks (informational)
    details["arm_torque_ok"] = safety.check_arm_torque(state)
    details["hand_current_ok"] = safety.check_hand_current(state)
    
    all_ok = all(details.values())
    return all_ok, details
```

#### Step 2: 在 _tick() 中调用 validate_action

**文件**: `dexmani_real/teleop/core/controller.py:_tick()` (Step 5 之后)

```python
# 在 send_action 之前添加集中化验证
if not self.dry_run:
    all_ok, details = self.robot.validate_action(action, state)
    if not all_ok:
        logger.warning("validate_action failed: %s", 
                       {k: v for k, v in details.items() if not v})
        # Fall back to hold
        hold = self.error_handler.hold_action()
        action = RobotAction(
            arm_qpos_cmd=hold.arm_qpos_cmd,
            hand_qpos_cmd=hold.hand_qpos_cmd,
        )
        # Record failure but don't send
        return
```

**注意**: 此改动是**增量式的**——先在 _tick() 中添加集中化门，不删除现有的分散检查。经过充分验证后，后续迭代可逐步将分散检查迁移到 `validate_action()` 中。

**测试**:
- 单元测试: mock 各种失败场景, 验证 validate_action 返回正确的 (False, details)
- 集成测试: dry_run 模式验证全部通过的正常流程

---

### 4.4 P1-4: REARM 中间恢复路径

**参考**: ManiUniCon `reset_event` (键盘 'h' 或 Quest 'A' 按钮)

**问题**: EMERGENCY_STOP 后需要完全重启脚本。瞬态 IK 失败或手部通信恢复后仍需全流程重启。

**实施方案**:

#### Step 1: 添加 REARM 控制信号

**文件**: `dexmani_real/teleop/control/keyboard.py`

```python
class ControlSignal(Enum):
    TELEOP = "T"
    RECORD = "R"
    STOP = "S"
    HOME = "H"
    EMERGENCY_STOP = "ESC"
    REARM = "X"      # NEW: 'x' key re-arm
    QUIT = "Q"

_KEY_MAP: dict[str, ControlSignal] = {
    "t": ControlSignal.TELEOP,
    "r": ControlSignal.RECORD,
    "s": ControlSignal.STOP,
    "h": ControlSignal.HOME,
    "x": ControlSignal.REARM,   # NEW
    "q": ControlSignal.QUIT,
    "\x1b": ControlSignal.EMERGENCY_STOP,
}
```

#### Step 2: 添加 _rearm() 方法和状态转换

**文件**: `dexmani_real/teleop/core/controller.py`

```python
# 在 _transition() 中添加:
if signal == ControlSignal.REARM:
    self._rearm()
    return

def _rearm(self) -> None:
    """Re-arm from EMERGENCY_STOP without script restart.
    
    Clears errors, resets tracking, transitions to IDLE.
    No-op if not in EMERGENCY_STOP state.
    """
    if self.state != ControllerState.EMERGENCY_STOP:
        logger.info(
            "REARM ignored: not in EMERGENCY_STOP (current=%s)",
            self.state.value,
        )
        return
    
    logger.info("REARM: clearing errors and resetting...")
    self.running = True
    self.error_handler.clear()
    self.tracking_quality.reset()
    
    if not self.dry_run:
        try:
            self.robot.arm.clear_error()
            self.robot.reset_soft_start()
        except Exception as e:
            logger.warning("REARM: error clearing robot state: %s", e)
    
    self.state = ControllerState.IDLE
    self._last_arm_cmd = None
    self._last_hand_cmd = None
    logger.info("REARM complete. State: IDLE")
```

**测试**:
- 单元测试: 在 IDLE/TELEOP 状态调用 REARM, 验证 no-op
- 单元测试: 在 EMERGENCY_STOP 状态调用 REARM, 验证恢复到 IDLE
- 集成测试: dry_run 模式完整流程: E-Stop → REARM → TELEOP

---

### 4.5 P1-5: DLS-only 模式

**参考**: BVPro 纯 DLS 方案（无随机回退）

**问题**: MPlib position IK 回退是最大延迟波动源（~10ms），DLS-only 模式可消除此开销。

**实施方案**:

此改进是**纯配置项**——代码已支持（`TeleopProfile.use_position_ik`），只需在使用时显式设置:

```python
# scripts/real/test_quest_hand_teleop.py
teleop_profile = TeleopProfile(
    use_position_ik=False,  # DLS-only, 无 MPlib 回退
    use_differential_ik_fallback=True,
    # ... other params ...
)
```

无需代码修改。在文档和脚本中增加使用说明即可。

---

### 4.6 P1-6: IK 可操作性评分

**参考**: LeFranX `weighted_ik.cpp:71-76` Yoshikawa 可操作性 + dexmani 现有 `kin.compute_manipulability()`

**问题**: `kin.compute_manipulability()` 已实现但未在 teleop 热点使用。

**实施方案**:

#### Step 1: 在 solve_differential_ik() 中添加可操作性检查

**文件**: `dexmani_real/planning/ik.py:solve_differential_ik()` (迭代循环内)

```python
# After computing proposed qpos, before accepting:
if profile.min_manipulability > 0:  # NEW config, default 0 = disabled
    manip = self.kin.compute_manipulability(proposed_qpos)
    if manip < profile.min_manipulability:
        # Skip this iteration, increase damping
        damping = profile.differential_ik_damping * profile.singularity_damping_scale
        continue  # retry with higher damping
```

#### Step 2: 在 TeleopProfile 中添加配置

**文件**: `dexmani_real/planning/types.py`

```python
@dataclass(kw_only=True)
class TeleopProfile:
    # ... existing fields ...
    min_manipulability: float = 0.0         # NEW: 0=disabled
    singularity_damping_scale: float = 10.0  # NEW: damping multiplier near singularity
```

**测试**:
- 仿真测试: SAPIEN 中验证在奇异位姿附近 IK 不会产生异常解

---

## 5. Phase 3: 架构增强

> **目标**: 实现 4 个 P2 级架构改进
> **工作量**: 2-4 周 | **风险**: 中-高（涉及进程/通信模型变更）

---

### 5.1 P2-1: ZMQ 进程分离 — VR 与控制解耦

**参考**: Open-Teach 多进程 ZMQ PUB/SUB + ManiUniCon SharedMemory IPC

**问题**: 单进程架构导致 VR 解析、IK、retargeting 竞争 GIL。

**实施方案**:

#### 架构设计

```
┌──────────────────────┐     ZMQ PUB (5555)     ┌──────────────────────┐
│ VR Publisher Process │───────────────────────►│ TeleopController     │
│ (独立 mp.Process)    │   topic="vr_frame"     │ (主进程, 50Hz)       │
│                      │   JSON + numpy buffer  │                      │
│ QuestHandTracker     │                        │ ZMQ SUB → get_latest │
│ _receive_loop()      │                        │ → Arm IK → Hand     │
│ → zmq pub            │                        │ → Safety → Execute  │
└──────────────────────┘                        └──────────────────────┘
```

#### Step 1: 新建 VR Publisher

**新文件**: `dexmani_real/teleop/vr/vr_publisher.py`

```python
"""VR frame publisher process — decoupled from control loop via ZMQ PUB/SUB.

Ref: Open-Teach OculusVRHandDetector (openteach/components/detector/oculus.py:74-104)
     ManiUniCon SharedStorage write_action/read_all_action pattern

Design:
  - Runs as a separate multiprocessing.Process
  - Owns its own QuestHandTracker instance
  - Publishes vr_frame dict as JSON + numpy buffer via ZMQ PUB
  - Controller subscribes via ZMQ SUB, polls non-blocking
"""

import multiprocessing as mp
import zmq
import numpy as np
import json
import time


class VRFramePublisher:
    """Publishes VR frames over ZMQ PUB socket."""
    
    def __init__(
        self,
        tracker: QuestHandTracker,
        pub_port: int = 5555,
        hz: float = 60.0,
    ) -> None:
        self.tracker = tracker
        self.pub_port = pub_port
        self.hz = hz
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
    
    def start(self) -> None:
        self._process = mp.Process(target=self._run, daemon=True)
        self._process.start()
    
    def _run(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.PUB)
        socket.bind(f"tcp://127.0.0.1:{self.pub_port}")
        
        interval = 1.0 / self.hz
        while not self._stop_event.is_set():
            frame = self.tracker.get_latest()
            if frame is not None:
                # Serialize: JSON for metadata, raw bytes for numpy arrays
                meta = {k: v for k, v in frame.items() if not isinstance(v, np.ndarray)}
                arrays = {k: v.tobytes() for k, v in frame.items() if isinstance(v, np.ndarray)}
                socket.send_json({"meta": meta, "arrays": arrays})
            time.sleep(interval)
        
        socket.close()
        ctx.term()
    
    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
```

#### Step 2: 在 TeleopController 中集成 ZMQ SUB

**文件**: `dexmani_real/teleop/core/controller.py`

```python
class TeleopController:
    def __init__(self, ..., use_zmq_vr: bool = False, zmq_vr_port: int = 5555):
        # ... existing init ...
        self._use_zmq_vr = use_zmq_vr
        if use_zmq_vr:
            self._zmq_ctx = zmq.Context()
            self._zmq_sub = self._zmq_ctx.socket(zmq.SUB)
            self._zmq_sub.connect(f"tcp://127.0.0.1:{zmq_vr_port}")
            self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    
    def _read_vr_frame(self):
        if self._use_zmq_vr:
            return self._read_vr_frame_zmq()
        else:
            return self._read_vr_frame_direct()
    
    def _read_vr_frame_zmq(self):
        try:
            msg = self._zmq_sub.recv_json(flags=zmq.NOBLOCK)
            # Deserialize back to dict with numpy arrays
            # ...
            return frame
        except zmq.Again:
            return None
```

> **注意**: ZMQ 方案需要额外依赖 `pyzmq`。如果希望零外部依赖，可以参考 ManiUniCon 的 `SharedMemoryRingBuffer` 方案（纯 Python `multiprocessing.shared_memory`），但 ZMQ 的 PUB/SUB 模式更适合一对多广播。

**测试**:
- 集成测试: 启动 VR Publisher → Controller SUB 接收 → 验证帧数据一致
- 性能测试: 对比直接调用 vs ZMQ 的延迟增量

---

### 5.2 P2-2: 相机录制 Daemon 进程

**参考**: ManiUniCon 独立 Camera Process

**问题**: 相机录制在 `add_frame(camera_frame=...)` 中同步执行，RealSense USB 断开会导致控制循环崩溃。

**实施方案**:

#### Step 1: 新建 CameraProcess

**新文件**: `dexmani_real/sensor/camera_process.py`

```python
"""Camera recording daemon process.

Ref: ManiUniCon Camera Process (main.py:163-170 RobotControlSystem)

Design:
  - Independent mp.Process per camera
  - Writes latest frame to SharedMemory + sets mp.Event
  - Controller non-blocking polls Event, reads latest frame
  - Crash isolation: camera process crash sets error flag, controller continues
"""

import multiprocessing as mp
import numpy as np
import time


class CameraProcess:
    """Captures frames from RealSense in a separate process."""
    
    def __init__(
        self,
        camera_serial: str,
        hz: float = 30.0,
        buffer_size: int = 3,
    ) -> None:
        self._process: mp.Process | None = None
        self._stop_event = mp.Event()
        self._frame_ready = mp.Event()
        self._crash_flag = mp.Event()
        # Shared memory for latest frame
        self._shm = None  # allocated in child process
    
    def start(self) -> None:
        self._process = mp.Process(target=self._run, daemon=True)
        self._process.start()
    
    def get_latest_frame(self, timeout: float = 0.0) -> dict | None:
        """Non-blocking poll for latest camera frame."""
        if self._frame_ready.wait(timeout=timeout):
            self._frame_ready.clear()
            # Read from shared memory
            return self._read_from_shm()
        return None
    
    @property
    def crashed(self) -> bool:
        return self._crash_flag.is_set()
    
    def _run(self) -> None:
        try:
            import pyrealsense2 as rs
            # ... RealSense init and capture loop ...
            while not self._stop_event.is_set():
                # capture frame → write to shm → set _frame_ready
                pass
        except Exception:
            self._crash_flag.set()
    
    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=3.0)
```

#### Step 2: 在 controller._tick() 中集成

```python
# In _tick(), instead of:
#   recorder.add_frame(camera_frame=...)
# Do:
if self._camera_process is not None:
    camera_frame = self._camera_process.get_latest_frame(timeout=0.0)
    if self._camera_process.crashed:
        logger.warning("Camera process crashed, continuing without camera data")
else:
    camera_frame = None

if self.state == ControllerState.RECORDING:
    self.recorder.add_frame(..., camera_frame=camera_frame)
```

**测试**:
- 集成测试: 启动 CameraProcess → 验证帧消费
- 鲁棒性测试: kill CameraProcess → 验证 controller 不崩溃

---

### 5.3 P2-3: LeRobot/Zarr 格式导出器

**参考**: ManiUniCon Zarr + LeRobot v3.0 + LeFranX HuggingFace Dataset

**问题**: HDF5 原生格式与 Diffusion Policy 训练生态不兼容。

**实施方案**:

#### Step 1: 新建导出脚本

**新文件**: `scripts/tools/export_hdf5_to_zarr.py` (~200 行)

```python
#!/usr/bin/env python3
"""Export HDF5 teleop episodes to Zarr format (Diffusion Policy compatible).

Usage:
    python export_hdf5_to_zarr.py --data_dir ./recordings/ --output ./zarr_data/

Ref: ManiUniCon Zarr exporter + LeRobot v3.0 dataset format
"""

import argparse
import json
from pathlib import Path
import numpy as np
import h5py
import zarr
from numcodecs import Blosc


def load_episodes(data_dir: Path, quality_mask: int | None = None):
    """Load all episodes from data_dir, optionally filter by quality flags."""
    episodes = []
    for h5_path in sorted(data_dir.glob("episode_*.h5")):
        with h5py.File(h5_path, "r") as f:
            if quality_mask is not None:
                flags = f["quality_flags"][:]
                valid = (flags & quality_mask) == quality_mask
            else:
                valid = slice(None)
            
            episodes.append({
                "obs_arm_qpos": f["obs/arm_qpos"][:][valid],
                "obs_hand_qpos": f["obs/hand_qpos"][:][valid],
                "obs_eef_pos": f["obs/eef_pos"][:][valid],
                "obs_eef_quat": f["obs/eef_quat"][:][valid],
                "action_arm_qpos": f["action/arm_qpos"][:][valid],
                "action_hand_qpos": f["action/hand_qpos"][:][valid],
                "meta": dict(f["meta"].attrs),
            })
    return episodes


def compute_norm_stats(episodes: list[dict]) -> dict:
    """Compute mean/std for normalization (like ManiUniCon norm_stats)."""
    all_obs = []
    all_act = []
    for ep in episodes:
        # Concatenate all observation dimensions
        obs = np.concatenate([
            ep["obs_arm_qpos"],
            ep["obs_hand_qpos"],
            ep["obs_eef_pos"],
            ep["obs_eef_quat"],
        ], axis=1)
        act = np.concatenate([
            ep["action_arm_qpos"],
            ep["action_hand_qpos"],
        ], axis=1)
        all_obs.append(obs)
        all_act.append(act)
    
    all_obs = np.concatenate(all_obs, axis=0)
    all_act = np.concatenate(all_act, axis=0)
    
    return {
        "obs_mean": all_obs.mean(axis=0),
        "obs_std": all_obs.std(axis=0),
        "action_mean": all_act.mean(axis=0),
        "action_std": all_act.std(axis=0),
    }


def export_to_zarr(episodes, norm_stats, output_dir, compressor=None):
    """Write to Zarr format compatible with Diffusion Policy."""
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    root = zarr.open(str(output_dir / "data.zarr"), mode="w")
    
    for i, ep in enumerate(episodes):
        grp = root.create_group(f"episode_{i}")
        # Write obs and action arrays
        # ...
    
    # Write norm stats
    with open(output_dir / "norm_stats.json", "w") as f:
        json.dump({k: v.tolist() for k, v in norm_stats.items()}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality_mask", type=int, default=0x07BF)  # ALL_GOOD
    args = parser.parse_args()
    
    episodes = load_episodes(Path(args.data_dir), args.quality_mask)
    print(f"Loaded {len(episodes)} episodes, {sum(len(e['obs_arm_qpos']) for e in episodes)} total frames")
    
    norm_stats = compute_norm_stats(episodes)
    export_to_zarr(episodes, norm_stats, args.output)
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    main()
```

**测试**:
- 单元测试: 用小型 HDF5 fixture 测试导出 → 验证 Zarr 结构可被 Diffusion Policy 加载
- 端到端测试: 录制 → 导出 → 训练一个 epoch → 验证 loss 下降

---

### 5.4 P2-4: Lock-Free 共享内存 IPC

**参考**: ManiUniCon `SharedMemoryRingBuffer` + `SharedMemoryQueue`

**问题**: ZMQ 方案虽然功能完善，但引入额外依赖和数据拷贝开销。Lock-Free SharedMemory 是纯 Python 的高性能替代。

**实施方案**:

此改进是 Phase 3.1 ZMQ 方案的**更高级替代**。如果不是极限性能场景（>200Hz IPC），优先使用 ZMQ；仅在性能基准显示 ZMQ 成为瓶颈时才实施。

**参考代码**: ManiUniCon `utils/shared_memory/shared_memory_ring_buffer.py:125-166` (lock-free atomic counter increment) + `shared_memory_queue.py:88-107` (dual-counter FIFO)

如需要实施，参考 ManiUniCon 的完整实现移植（~500 行），核心组件:
1. `SharedMemoryRingBuffer` — 单生产者单消费者，FILO，单原子计数器
2. `SharedMemoryQueue` — 单生产者单消费者，FIFO，双原子计数器
3. `SharedStorage` — 封装 RingBuffer + Queue 的高级接口

> **注意**: 此改进为 Phase 3 可选项目，依赖 Python 3.8+ 的 `multiprocessing.shared_memory` 模块。

---

## 6. Phase 4: 工程优化

> **目标**: 实现 3 个 P3 级工程改进
> **工作量**: 1-3 天 | **风险**: 低

---

### 6.1 P3-1: PipelineConfig from_dict() / to_dict() Round-Trip

**参考**: ManiUniCon Hydra 自动 YAML 往返

**问题**: `PipelineConfig.to_dict()` 存在但 `from_dict()` 未实现，无法还原配置。

**实施方案**:

#### Step 1: 实现 from_dict() 类方法

**文件**: `dexmani_real/config/pipeline_config.py`

```python
@dataclass
class PipelineConfig:
    # ... existing fields ...
    
    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Reconstruct PipelineConfig from a dict (reverse of to_dict())."""
        import dataclasses
        
        kw = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            val = d[f.name]
            field_type = f.type
            
            # Handle nested dataclasses with from_dict
            if hasattr(field_type, 'from_dict'):
                kw[f.name] = field_type.from_dict(val)
            # Handle numpy arrays stored as lists
            elif isinstance(val, list) and f.name.endswith(('_bounds',)):
                kw[f.name] = np.array(val, dtype=np.float64)
            # Handle None
            elif val is None:
                kw[f.name] = None
            else:
                kw[f.name] = val
        
        return cls(**kw)
```

#### Step 2: 对子 dataclass 重复此模式

**文件**: `dexmani_real/robot/types.py` (RobotInterfaceConfig), `dexmani_real/planning/types.py` (PlanningProfile, TeleopProfile)

同样添加 `from_dict()` 方法。

#### Step 3: 添加 numpy 序列化辅助函数

**文件**: `dexmani_real/config/pipeline_config.py`

```python
def _list_to_ndarray(val: list, dtype=np.float64) -> np.ndarray:
    """Reverse of _ndarray_to_list — restore numpy array from list."""
    return np.array(val, dtype=dtype)
```

#### Step 4: 验证 round-trip

```python
# 在单元测试中:
config = PipelineConfig(...)
d = config.to_dict()
config2 = PipelineConfig.from_dict(d)
assert d == config2.to_dict(), "Round-trip failed!"
```

---

### 6.2 P3-2: Docker 化重定向服务

**参考**: BVPro `bunny_teleop_server` Docker 部署

**问题**: dex_retargeting 依赖项多（PyTorch, trimesh, 特定 CUDA 版本），部署困难。

**实施方案**:

创建 `Dockerfile.teleop` 和 `docker-compose.yml`，将重定向服务容器化:

```dockerfile
# Dockerfile.teleop
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install dex-retargeting numpy scipy trimesh pyzmq

COPY dexmani_real/teleop/vr/hand_retarget.py /app/
COPY dexmani_real/teleop/vr/ref_adapter.py /app/

CMD ["python", "-m", "dexmani_real.teleop.vr.hand_retarget_server"]
```

这是低优先级改进——当前 retargeting 性能在本地运行良好（2-5ms/帧），Docker 化主要收益在部署一致性而非性能。

---

### 6.3 P3-3: 双手（左手）支持

**参考**: BVPro 原生双手支持 + LeFranX dual config

**问题**: 当前仅支持右手。添加左手需要镜像配置和坐标变换。

**实施方案**:

非本次开发重点，提供路线图:

1. 创建左手配置: `configs/left_hand_config.json`
2. 镜像坐标变换矩阵: `OPERATOR2MANO_LEFT`
3. 在 `TeleopController` 中支持 `side="left"` 参数
4. 左手 XHand 关节角映射（镜像关节序）
5. SAPIEN 仿真中构建左手环境

---

## 7. 附录

### 7.1 完整文件变更清单

| Phase | 文件 | 操作 | 行数变化 |
|-------|------|------|---------|
| **0** | `docs/vr-teleop-control-loop-design.md` | 编辑 §3.1, §5.2, §6.1, §8.1 | ~30 行 |
| **0** | `docs/vr-teleop-framework-comparison.md` | 编辑 §2.2, §6 | ~10 行 |
| **1.1** | `recording/episode_recorder.py` | 编辑 __init__, add_frame | +15 行 |
| **1.1** | `config/pipeline_config.py` | 添加 max_record_frames | +3 行 |
| **1.2** | `planning/planner.py` | 扩展 WorkspaceSafety | +40 行 |
| **1.2** | `robot/types.py` | 添加 orientation_bounds | +3 行 |
| **1.2** | `robot/interface.py` | 添加 check_workspace_orientation | +10 行 |
| **1.2** | `teleop/core/controller.py` | 集成 orientation check | +5 行 |
| **1.3** | `teleop/vr/arm_mapper.py` | 添加 _clip_delta_rot | +15 行 |
| **1.4** | `teleop/core/controller.py` | 修改丢失处理逻辑 | +25 行 |
| **2.1** | `planning/types.py` | 添加自适应阻尼字段 | +5 行 |
| **2.1** | `planning/ik.py` | 修改 solve_differential_ik | +15 行 |
| **2.2** | `teleop/vr/pose_interpolator.py` | **新建** | +150 行 |
| **2.2** | `planning/types.py` | 添加插值配置 | +5 行 |
| **2.2** | `teleop/core/controller.py` | 集成插值器 | +20 行 |
| **2.3** | `robot/interface.py` | 添加 validate_action | +60 行 |
| **2.3** | `teleop/core/controller.py` | 调用 validate_action | +15 行 |
| **2.4** | `teleop/control/keyboard.py` | 添加 REARM 信号 | +2 行 |
| **2.4** | `teleop/core/controller.py` | 添加 _rearm | +30 行 |
| **2.6** | `planning/ik.py` | 添加可操作性检查 | +10 行 |
| **2.6** | `planning/types.py` | 添加可操作性配置 | +3 行 |
| **3.1** | `teleop/vr/vr_publisher.py` | **新建** | +100 行 |
| **3.1** | `teleop/core/controller.py` | 添加 ZMQ SUB 模式 | +40 行 |
| **3.2** | `sensor/camera_process.py` | **新建** | +120 行 |
| **3.2** | `teleop/core/controller.py` | 集成异步相机 | +15 行 |
| **3.3** | `scripts/tools/export_hdf5_to_zarr.py` | **新建** | +200 行 |
| **4.1** | `config/pipeline_config.py` | 添加 from_dict | +30 行 |
| **4.1** | `robot/types.py` | 添加 from_dict | +20 行 |
| **4.1** | `planning/types.py` | 添加 from_dict | +20 行 |

### 7.2 参考文件速查表

| 参考内容 | 参考项目 | 完整路径 |
|----------|---------|----------|
| PoseTrajectoryInterpolator | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/pose_trajectory_interpolator.py:78-207` |
| _check_safety_limits | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/quest_controller.py:134-158` |
| _apply_safety_limits | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/quest_controller.py:160-190` |
| JointSpaceSmoother | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/filter.py:77-138` |
| SharedMemoryRingBuffer | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/shared_memory/shared_memory_ring_buffer.py:125-166` |
| SharedMemoryQueue | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/shared_memory/shared_memory_queue.py:88-107` |
| IKSolver (Pink QP) | ManiUniCon | `Reference/ManiUniCon/maniunicon/utils/ik_solver.py:171-217` |
| WeightedIKSolver | LeFranX | `Reference/LeFranX/franka_xhand_teleoperator/src/weighted_ik.cpp:71-76` |
| clip_arm_velocity | BVPro | `Reference/BunnyVisionPro/real_control/xarm7_ability.py:185-194` |
| OculusVRHandDetector | Open-Teach | `Reference/Open-Teach/openteach/components/detector/oculus.py:74-104` |
| compute_manipulability | dexmani | `dexmani_real/planning/kinematics.py:69-74` |

### 7.3 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Cartesian 插值层位置 | IK 之前（Cartesian 空间） | 在源头消除噪声，比关节空间 EMA 更有效（ManiUniCon 经验） |
| validate_action 位置 | send_action 之前（集中化） | 统一安全门，不删除现有分散检查（增量式） |
| IPC 方案 | 优先 ZMQ，可选 SharedMemory | ZMQ 更成熟、调试方便、支持一对多；SharedMemory 是性能优化备选 |
| 自适应阻尼 | 默认关闭 | 向后兼容，需真机验证后启用 |
| DLS-only 模式 | 纯配置项 | TeleopProfile.use_position_ik=False 即可，无需代码 |
| 相机进程 | 独立 mp.Process + mp.Event | Crash 隔离，不影响控制回路（ManiUniCon 经验） |

---

> **方案版本**: v1.0 | **制定日期**: 2026-06-22
> **基于文档**: `vr-teleop-framework-comparison.md` v2.0, `maniunicon-comparison.md` v1.0, `vr-teleop-control-loop-design.md` v1.0, `code-review-design-docs.md`
