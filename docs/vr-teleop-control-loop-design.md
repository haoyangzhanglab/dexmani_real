# VR 遥操作控制回路设计

> **版本**: v1.0 | **日期**: 2026-06-22 | **基于**: dexmani_real main 分支当前代码

---

## 目录

1. [设计目标](#1-设计目标)
2. [架构分层](#2-架构分层)
3. [管道阶段](#3-管道阶段)
4. [状态机](#4-状态机)
5. [接口契约](#5-接口契约)
6. [错误处理策略](#6-错误处理策略)
7. [数据流与生命周期](#7-数据流与生命周期)
8. [性能预算](#8-性能预算)
9. [配置点](#9-配置点)
10. [测试策略](#10-测试策略)

---

## 1. 设计目标

### 1.1 核心原则

| 原则 | 说明 |
|------|------|
| **Safety-First** | 任何管道失败 → hold last_good，硬件异常 → E-Stop。安全不可妥协 |
| **Deterministic Primary** | IK 优先确定性方法（DLS），仅在必要时回退随机方法（MPlib） |
| **Fail-Safe Defaults** | 未连接的模块返回 hold，不发散；未知状态视为不安全 |
| **Observable** | 每帧产生 11-bit quality flags（含 CAMERA_OK），录制完整的 state×action×vr 三元组 |
| **Single-Thread Simplicity** | 当前单线程足够(7-12ms DLS路径满足 20ms 预算)；解耦是未来优化方向 |

### 1.2 设计约束

| 约束 | 值 | 来源 |
|------|-----|------|
| 目标频率 | 50 Hz (20ms/frame) | `RateLimiter(50.0)` |
| VR 帧新鲜度 | < 200ms | `TrackingQualityConfig.max_frame_age_s=0.2` |
| 跟踪丢失容忍 | < 1.0s 连续丢失 | `TrackingQualityConfig.ema_loss_timeout_s=1.0` |
| IK 位置精度 | < 8mm | `TeleopProfile.max_pose_error_pos_m=0.008` |
| IK 姿态精度 | < 0.08 rad (~4.6°) | `TeleopProfile.max_pose_error_rot_rad=0.08` |
| 关节跳变上限 | arm 5°/frame, hand 10°/frame | `_ARM_JUMP_LIMIT_RAD` / `_HAND_JUMP_LIMIT_RAD` |
| 速度限制 | [60,60,60,60,90,90,120] °/s | 驱动层 `XArm7._limit_joint_step()` |
| 力矩限制 | [50,50,30,30,30,20,20] Nm | `safety._ARM_TORQUE_LIMIT_NM` |

---

## 2. 架构分层

### 2.1 五层模型

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Orchestration                                          │
│   TeleopController: state machine, lifecycle, keyboard dispatch │
│   File: teleop/core/controller.py                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ owns & orchestrates
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Pipeline (per-frame computation)                       │
│   _tick() → _read_vr → _compute_action → _send → _record        │
│   Pipeline = VR Input Gate → Arm IK → Hand Retarget → Safety →  │
│              Quality Flags → Execute → Record                    │
│   File: teleop/core/controller.py :199-285                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ delegates to
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Computation                                            │
│   ArmWristMapper, TeleopIKSolver, XHandRetargeter,              │
│   Safety checks, QualityFlags                                   │
│   Files: teleop/vr/, planning/ik.py, teleop/control/safety.py   │
└────────────────────────────┬────────────────────────────────────┘
                             │ reads/writes via
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Abstraction                                            │
│   RobotInterface: unified arm+hand, workspace, collision         │
│   QuestHandTracker: VR data source                               │
│   EpisodeRecorder: data recording                                │
│   Files: robot/interface.py, teleop/vr/vr_tracker.py             │
└────────────────────────────┬────────────────────────────────────┘
                             │ wraps
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Hardware                                                │
│   XArm7 SDK, XHand SDK, HTS SDK                                 │
│   Files: robot/xarm7/, robot/xhand/, hand_tracking_sdk/         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖图

```
QuestHandTracker ──(vr_frame)──► TeleopController
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
            ArmWristMapper    XArm7MotionPlanner   XHandRetargeter
                    │                 │                  │
                    │     ┌───────────┤                  │
                    │     ▼           ▼                  │
                    │  TeleopIKSolver  │                  │
                    │     │           │                  │
                    │     ▼           │                  │
                    │  XArm7Kinematics│                  │
                    │     │           │                  │
                    ▼     ▼           ▼                  ▼
              ┌─────────────────────────────────────────────┐
              │              RobotInterface                  │
              │   get_state() / send_action() / check_*()   │
              └──────────┬────────────────────┬─────────────┘
                         ▼                    ▼
                      XArm7                XHand
```

---

## 3. 管道阶段

### 3.1 管道总览

`_tick()` 一帧内的数据流，共 8 个阶段（对应代码中 _tick() 的注释块）：

```
  Stage 1                    Stage 2         Stage 3         Stage 4
┌───────────────┐         ┌───────────┐  ┌───────────┐  ┌──────────┐
│ VR INPUT +    │────────►│ ARM IK    │─►│ HAND      │─►│ SAFETY   │
│ QUALITY GATE  │         │ SOLVER    │  │ RETARGET  │  │ CHECKS   │
└───────────────┘         └───────────┘  └───────────┘  └──────────┘
     │                          │               │              │
     ▼                          ▼               ▼              ▼
 tracker.get_latest()    TeleopIKSolver   XHandRetargeter  safety.*
 + TrackingQuality       .solve()         .retarget()      check_*()
   .check()

  Stage 5 (incremental)     Stage 6            Stage 7         Stage 8
┌──────────────────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────┐
│ QUALITY FLAGS        │  │ READ CAMERA  │  │ RECORD    │  │ EXECUTE  │
│ (set across stages)  │─►│ (multi/single)│─►│ FRAME     │─►│ ACTION   │
└──────────────────────┘  └──────────────┘  └───────────┘  └──────────┘
     │                           │                 │              │
     ▼                           ▼                 ▼              ▼
 QualityFlags.set()*10    MultiCameraManager  CollectionLoop  robot
 (in pipeline steps)      .read_all_latest()  .record_frame() .send_action()
                          or CameraProcess
                          .poll_latest_frame()
```

> **注意**: Quality Flags 不是独立的单一步骤 — bits 0-5（TRACKING/IK/RETARGET/JUMP/WORKSPACE）在 `_compute_action()` 内增量设置，bits 7-10（TORQUE/CURRENT/TEMP/COMM）在 `_tick()` step 5 设置。Stage 7 (Record) 在 Stage 8 (Execute) 之前执行 — 录制的是本帧计算出的 action，不等硬件执行结果。录制通过 `CollectionLoop.record_frame()` 委托执行（支持 auto_stop_on_quality_drop + 多相机帧）。

### 3.2 Stage 1: VR Input Gate

**职责**: 获取最新 VR 帧，检查跟踪质量，决定是否继续管道。

**输入**: 无（从 tracker 拉取）

**输出**: `dict | None` — VR frame with wrist_pos, wrist_quat_wxyz, landmarks, sequence_id, local_recv_ns

**接口**:
```python
# QuestHandTracker (teleop/vr/vr_tracker.py:142)
def get_latest(max_age_s: float | None = None) -> dict | None:
    """线程安全获取最新帧。max_age_s=None 使用 self.max_frame_age_s (0.2s)。"""

# VR Frame dict schema:
{
    "side": str,                    # "right"
    "wrist_pos": np.ndarray,        # (3,) float64, FLU frame
    "wrist_quat_wxyz": np.ndarray,  # (4,) float64
    "landmarks": np.ndarray,        # (21, 3) float64, FLU frame
    "recv_ts_ns": int,              # HTS 接收时间戳 (ns)
    "source_ts_ns": int,            # Quest 源时间戳 (ns)
    "sequence_id": int,             # 帧序号
    "source_frame_seq": int,        # 源帧序号
    "coordinate_frame": str,        # "flu"
    "local_recv_ns": int,           # 本地接收时间 (monotonic_ns)
}
```

**决策逻辑**:
```
vr_frame 为 None?
  ├─ Yes → hold + 标记 tracking_lost
  └─ No → age_s > 0.2s?
           ├─ Yes → hold + 标记 stale
           └─ No → 通过，继续管道
```

**失败处理**:
| 失败类型 | 行为 | 质量标记 |
|----------|------|----------|
| None (tracker 未连接) | hold current qpos | `TRACKING_OK=0` |
| age > 0.2s (stale) | hold current qpos | `TRACKING_OK=0` |
| 连续丢失 > 1.0s | **E-Stop** | `TRACKING_OK=0` |

### 3.3 Stage 2: Arm IK Solver

**职责**: 将 VR 手腕位姿映射为机器人末端位姿，求解 7-DOF 关节角。

**子阶段**:
```
ArmWristMapper.map(wrist_pos, wrist_quat) → target_eef_pose (Pose)
    │
    ▼
TeleopIKSolver.solve(target_pose, current_qpos, prev_cmd) → IKResult
    │
    ├─ Step 1: solve_differential_ik()  (DLS, deterministic, <1ms)
    │   └─ pose_error 达标? → 返回
    │   └─ 不达标? → 进入 Step 2
    │
    ├─ Step 2: solve_position_ik()  (MPlib, stochastic, ~10ms)
    │   └─ seed=prev_cmd(3次) → 成功? → 返回 hardware-closest
    │   └─ 失败? → seed=current_qpos(2次) → 成功? → 返回
    │   └─ 失败? → 进入 Step 3
    │
    └─ Step 3: hold prev_cmd (last_good)
```

**输入**:
```python
# TeleopIKSolver.solve() (planning/ik.py:46)
def solve(
    target_eef_pose_world: Pose,   # ArmWristMapper 输出
    current_qpos: np.ndarray,      # (7,) 当前关节角, from RobotState
    previous_qpos_cmd: np.ndarray, # (7,) 上帧命令
) -> IKResult:
```

**输出**:
```python
@dataclass
class IKResult:
    success: bool
    qpos: np.ndarray | None           # (7,) joint angles 或 prev_cmd
    reason: str                        # 失败原因
    report: dict[str, Any]             # 详细报告
    held: bool                         # 是否为 hold 结果
```

**Wrist → EEF 映射 (`ArmWristMapper`)**:
```python
# teleop/vr/arm_mapper.py:49
def map(wrist_pos, wrist_quat_wxyz) -> dict | None:
    """
    Reset-relative 差分映射:
      delta_pos_base = pos_scale * (vr_to_base_rot @ (wrist_pos - wrist_pos0))
      delta_rot_base = vr_to_base_rot @ (wrist_rot @ wrist_rot0.T) @ vr_to_base_rot.T
      target_pos = eef_pos0 + delta_pos_base (经 clip_delta_pos 裁剪)
      target_rot = delta_rot_base @ eef_rot0
    """
```

**EMA 平滑** (仅 arm):
```python
# controller.py:380
arm_cmd = ema_smooth(raw_arm, self._last_arm_cmd, self.ema_alpha_arm)
# ema_alpha_arm=1.0 默认关闭。设为 0.3-0.5 可平滑操作员手抖。
```

### 3.4 Stage 3: Hand Retarget

**职责**: 将 21 个 VR 手部关键点映射为 XHand 12-DOF 关节角。

**子阶段**:
```
estimate_frame_from_hand_points(landmarks)
    → wrist_rot (3×3): SVD 手掌帧估计
    │
    ▼
landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
    → mano_landmarks (21×3): 转换到 MANO 手模型坐标系
    │
    ▼
XHandRetargeter.retarget(mano_landmarks)
    → _build_ref_value(mano_landmarks): 向量差分
    → XHandRefAdapter.apply(): pinky 缩放 (1.2x-2.2x)
    → dex_retargeting.retarget(): 优化求解
    → _retargeted_joint_order: 重映射到 XHand 关节序
    │
    ▼
hand_cmd (12,) — 无 EMA，dex_retargeting 内置 low_pass_alpha
```

**输入**:
```python
# XHandRetargeter.retarget() (teleop/vr/hand_retarget.py:93)
def retarget(hand_joint_pos: np.ndarray | None) -> np.ndarray | None:
    """
    hand_joint_pos: (21, 3) landmarks in MANO frame
    returns: (12,) XHand joint angles or None on failure
    """
```

**Rretarget 验证**:
```python
# controller.py:431
quality.set(RETARGET_OK, retarget_ok)
quality.set(RETARGET_VALID, safety.check_retarget_valid(hand_cmd))
# check_retarget_valid: 检查所有关节在 [-0.5, 2.5] rad 生理范围内
```

### 3.5 Stage 4: Safety Checks

**职责**: 检查机器人硬件状态，确保不发送危险指令。

**检查清单**（按严重性排序）:

| 检查项 | 函数 | 失败行为 | 质量标记 |
|--------|------|----------|----------|
| 关节力矩 (arm) | `check_arm_torque()` | quality flag | `ARM_TORQUE_OK` |
| 关节电流 (hand) | `check_hand_current()` | quality flag | `HAND_CURRENT_OK` |
| 关节温度 (hand) | `check_hand_temperature()` | quality flag | `HAND_TEMP_OK` |
| 通信状态 (hand) | `check_hand_comm()` | quality flag | `HAND_COMM_OK` |
| 关节限位 (arm) | `check_arm_joint_limits()` | **E-Stop** | N/A (hard block) |
| 关节限位 (hand) | `check_hand_joint_limits()` | **Warning** | N/A |
| 工作空间 (EEF) | `robot.check_workspace()` | hold last_good | `IN_WORKSPACE` |
| 关节跳变 (arm) | 内联 `_apply_jump_clamp()` | clip + hold | `JOINT_JUMP_OK` |
| 关节跳变 (hand) | 内联 `_apply_jump_clamp()` | clip + hold | `JOINT_JUMP_OK` |
| 机器人错误 | `robot.is_error()` | **E-Stop** | N/A (hard block) |

**执行顺序**:
```
Stage 3 输出 arm_cmd, hand_cmd
    │
    ├─ workspace check → IN_WORKSPACE flag
    │   └─ fail: hold last_good (不执行 hand retarget)
    │
    ├─ joint jump clamp → JOINT_JUMP_OK flag
    │   └─ fail: clip delta to limit, still send clipped cmd
    │
    ├─ hardware safety checks → quality flags
    │   (non-blocking, recorded for post-hoc analysis)
    │
    └─ hard safety checks → E-Stop / Warning
```

### 3.6 Stage 5: Quality Flags Assembly

**职责**: 汇总所有检查结果到 11-bit flag（含 CAMERA_OK bit 6）。

**bit 定义** (`recording/quality_flags.py:25-34`):

```
Bit  0: TRACKING_OK      = VR tracking valid
Bit  1: IK_SUCCESS       = IK solve success
Bit  2: RETARGET_OK      = hand retargeting success
Bit  3: RETARGET_VALID   = retarget within physiological range [-0.5, 2.5] rad
Bit  4: JOINT_JUMP_OK    = joint jump within limits (5°/10° per frame)
Bit  5: IN_WORKSPACE     = EEF within workspace bounds
Bit  7: ARM_TORQUE_OK    = arm torque within per-joint limits
Bit  8: HAND_CURRENT_OK  = hand current < 500mA
Bit  9: HAND_TEMP_OK     = hand temperature < 70°C
Bit 10: HAND_COMM_OK     = hand communication normal (no board error)
```

**ALL_GOOD_MASK**: 0x07BF (bits 0-5, 7-10 all set)

**用途**: 
- 录制数据后处理时过滤低质量帧
- 实时状态显示：`controller._print_status()` 列出失败项
- `stop_episode()` 计算 `num_valid_frames`

### 3.7 Stage 6: Execute Action

**职责**: 发送关节指令到机器人硬件。

**接口**:
```python
# RobotInterface.send_action() (robot/interface.py:330)
def send_action(action: RobotAction) -> dict:
    """
    发送 arm + hand 指令。
    
    内部处理:
      - XArm7.send_action(): 重试 + _limit_joint_step() bottleneck scaling + 关节限位剪裁
      - XHand.send_action(): 关节限位剪裁
    
    Returns:
      {"arm_ok": bool, "hand_ok": bool,
       "arm_cmd": ndarray | None,   # (7,) post-clip 实际发送值
       "hand_cmd": ndarray | None}  # (12,) post-clip 实际发送值
    """
```

**RobotAction schema** (`robot/types.py:82-108`):
```python
@dataclass
class RobotAction:
    arm_qpos_cmd: np.ndarray        # (7,) float64 rad — final command after clipping
    hand_qpos_cmd: np.ndarray       # (12,) float64 rad
    target_eef_pos: np.ndarray | None = None   # (3,) m — EEF target before IK
    target_eef_rot6d: np.ndarray | None = None # (6,) — EEF target rotation
```

**执行条件**:
```python
if not self.dry_run:
    if self.robot.is_error():       # 发送前再次检查错误
        self._escalate_to_emergency(...)
        return
    if self.state != ControllerState.IDLE:  # IDLE 状态不发送
        result = self.robot.send_action(action)
```

### 3.8 Stage 7: Record Frame

**职责**: 将观测-动作-VR 三元组写入 HDF5 文件。

**接口**:
```python
# EpisodeRecorder.add_frame() (recording/episode_recorder.py:120)
def add_frame(
    state: RobotState,           # 完整 robot state
    action: RobotAction,         # 发送的 action
    vr_frame: dict[str, Any],    # VR 原始帧
    quality_flags: int,          # 11-bit quality (含 CAMERA_OK)
    camera_frame: dict | None,   # 相机帧（可选）
    T_base_eef: np.ndarray | None, # (4,4) EEF 位姿（用于相机外参）
) -> bool:
```

**HDF5 结构** (per episode_XXX.h5):
```
/meta:          task_label, operator, duration, fps, num_frames, 
                num_valid_frames, success, camera metadata
/obs:           arm_qpos(7), arm_qvel(7), arm_tau(7), eef_pos(3), eef_quat(4),
                hand_qpos(12), hand_current(12), hand_temperature(12)
/action:        arm_qpos(7), hand_qpos(12)
/vr:            wrist_pos(3), wrist_quat(4), landmarks(21,3)
/quality_flags: (T,) uint16
/camera:        rgb(T,H,W,3), depth(T,H,W), timestamps(T),
                extrinsics(T,4,4), K(3,3)
```

**录制条件**:
```python
if self.state == ControllerState.RECORDING and self.recorder is not None:
    self.recorder.add_frame(...)
# TELEOP 状态不录制，仅 RECORDING 状态录制
```

---

## 4. 状态机

### 4.1 状态定义

```
                    ┌──────────────┐
          ESC/Q ───►│ EMERGENCY    │◄─── ESC/Q (任意状态)
                    │ STOP         │
                    └──────────────┘
                          ▲
               arm_joint_limit violation
               robot.is_error() before send
               VR tracking lost > 1.0s
                          │
    ┌─────────┐    T     ┌──────────┐    R     ┌───────────┐
    │  IDLE   │─────────►│ TELEOP   │─────────►│ RECORDING │
    └─────────┘          └──────────┘          └───────────┘
         ▲                    │   ▲                  │
         │ S                  │   │ S                │ S
         │                    ▼   │                  │
         │              ┌──────────┐                 │
         └──────────────│   IDLE   │◄────────────────┘
         (TELEOP→IDLE)  └──────────┘  (RECORDING→TELEOP, not IDLE)
                              │
                        H (任意状态)
                              ▼
                        return_to_home()
                              │
                              ▼
                           IDLE
```

### 4.2 状态转换表

| 当前状态 | 信号 | 新状态 | 额外操作 |
|----------|------|--------|----------|
| IDLE | `TELEOP` (T) | TELEOP | `_reset_mapper()`, `robot.reset_soft_start()` |
| IDLE | `HOME` (H) | IDLE | `robot.return_to_home()` |
| TELEOP | `RECORD` (R) | RECORDING | `_reset_mapper()`, `recorder.start_episode()` |
| TELEOP | `STOP` (S) | IDLE | `self.state = ControllerState.IDLE` |
| TELEOP | `HOME` (H) | IDLE | `robot.return_to_home()`, `error_handler.clear()` |
| RECORDING | `STOP` (S) | TELEOP | `recorder.stop_episode(success=True)` |
| RECORDING | `HOME` (H) | IDLE | 同 TELEOP→HOME |
| ANY | `EMERGENCY_STOP` (ESC) | EMERGENCY_STOP | `robot.emergency_stop()`, `running=False` |
| ANY | `QUIT` (Q) | (exit) | `running=False`（`_shutdown()` 在 `run()` finally 块执行） |

### 4.3 每个状态的 _tick() 行为

| 状态 | _tick() 行为 |
|------|-------------|
| **IDLE** | VR 读取 + 质量检查(stale→E-Stop) + 安全检查(state 上的力矩/电流/温度 flag) + 键盘轮询 |
| **TELEOP** | 完整 7-stage pipeline (见§3.1) |
| **RECORDING** | 同 TELEOP + `recorder.add_frame()` |
| **EMERGENCY_STOP** | `running=False` → 主循环退出 |

---

## 5. 接口契约

### 5.1 TeleopController

**生命周期**:
```python
controller = TeleopController(
    robot: RobotInterface,          # 统一硬件接口
    arm_mapper: ArmWristMapper,     # VR wrist → EEF 位姿
    retargeter: XHandRetargeter,    # VR hand → 手部关节角
    planner: XArm7MotionPlanner,    # IK + 碰撞检测
    *,
    tracker: QuestHandTracker | None,  # VR 数据源
    keyboard_queue: object | None,     # 键盘事件队列
    target_hz: float = 50.0,           # 目标频率
    ema_alpha_arm: float = 1.0,        # arm 平滑因子 (1.0=关闭)
    dry_run: bool = False,             # 无硬件模式
    recorder: EpisodeRecorder | None,  # 数据录制器
)

controller.start()   # 启动键盘线程
controller.run()     # 进入主循环 (blocking)
controller.stop()    # 停止键盘线程
```

**主循环伪代码**:
```python
def run(self):
    self.start()
    if not self.dry_run and not self.robot.is_connected():
        self.robot.connect()
    
    try:
        while self.running:
            self._handle_keyboard()  # 非阻塞轮询，产生 ControlSignal
            self._tick()             # 单帧管道
            self.limiter.wait()      # 补偿计算时间，维持 50Hz
    except KeyboardInterrupt:
        ...  # 优雅退出
    except (RuntimeError, ConnectionError, ValueError):
        ...  # 异常退出
    finally:
        self._shutdown()  # 停止录制 + 键盘线程
```

### 5.2 _tick() 控制流

```
_tick()
  │
  ├─ 1. _read_vr_frame()              → vr_frame | None
  │
  ├─ 2. tracking_quality.check()      → ok, stale, tracking_lost
  │     ├─ not ok → record_failure("vr_stale")
  │     └─ tracking_lost → _escalate_to_emergency()
  │
  ├─ 3. robot.get_state()             → RobotState
  │     └─ dry_run? → _dummy_state()
  │
  ├─ 4. _compute_action(vr, state)    → (RobotAction, QualityFlags)
  │     │
  │     ├─ error_handler.init_fallback(current_qpos)
  │     │
  │     ├─ _compute_arm_command(vr, state, prev_cmd, quality)
  │     │   ├─ arm_mapper.map(wrist_pos, wrist_quat) → target Pose
  │     │   ├─ planner.solve_teleop_ik(target, current, prev) → IKResult
  │     │   ├─ ema_smooth(raw, last, alpha) → smoothed arm cmd
  │     │   └─ quality.set(IK_SUCCESS, ...)
  │     │
  │     ├─ workspace check on arm_cmd EEF
  │     │   ├─ pass → _compute_hand_command(vr, prev, quality)
  │     │   │   ├─ estimate_frame_from_hand_points(landmarks)
  │     │   │   ├─ landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
  │     │   │   ├─ retargeter.retarget(mano_landmarks) → hand cmd
  │     │   │   └─ quality.set(RETARGET_OK/VALID, ...)
  │     │   └─ fail → hold last_good (skip hand)
  │     │
  │     ├─ _apply_jump_clamp(arm, hand, prev_arm, prev_hand)
  │     │   └─ quality.set(JOINT_JUMP_OK, ...)
  │     │
  │     └─ if ik_ok and retarget_ok: error_handler.update_good_positions()
  │
  ├─ 5. Safety checks on state
  │     ├─ quality.set(ARM_TORQUE_OK, check_arm_torque(state))
  │     ├─ quality.set(HAND_CURRENT_OK, check_hand_current(state))
  │     ├─ quality.set(HAND_TEMP_OK, check_hand_temperature(state))
  │     ├─ quality.set(HAND_COMM_OK, check_hand_comm(state))
  │     ├─ check_arm_joint_limits → E-Stop on violation
  │     └─ check_hand_joint_limits → Warning on violation
  │
  ├─ 6. Record (RECORDING state only, 在 Execute 之前)
  │     └─ recorder.add_frame(state, action, vr, flags, T_base_eef)
  │
  ├─ 7. Execute action (non-IDLE states only)
  │     ├─ robot.is_error()? → E-Stop
  │     └─ robot.send_action(action) → {arm_ok, hand_ok}
  │
  └─ 8. Periodic status print (每 2s)
```

### 5.3 关键组件接口

#### ArmWristMapper

```python
# teleop/vr/arm_mapper.py
class ArmWristMapper:
    def reset(wrist_pos, wrist_quat_wxyz, eef_pos, eef_quat_wxyz) -> None
        """重置参考点（IDLE→TELEOP / TELEOP→RECORDING 时调用）"""
    
    def map(wrist_pos, wrist_quat_wxyz) -> dict | None
        """VR 手腕位姿 → target EEF 位姿。未 reset 返回 None。"""
        # returns: {"pos": (3,), "quat_wxyz": (4,)}
    
    def clear() -> None
        """清除参考点"""
    
    def is_ready() -> bool
        """reset 已被调用过?"""
```

#### TeleopIKSolver

```python
# planning/ik.py
class TeleopIKSolver:
    def solve(target_eef_pose_world, current_qpos, previous_qpos_cmd) -> IKResult
        """
        两阶段 IK:
          1. solve_differential_ik() — DLS, 确定性, <1ms
          2. solve_position_ik() — MPlib 随机 IK, ~10ms, 硬件最近候选
          3. 均失败 → prev_cmd (hold)
        """
    
    def solve_differential_ik(target, current, prev, profile) -> IKResult
        """DLS 微分 IK (ref: BVPro). gain=1.0, damping=0.02."""
    
    def solve_position_ik(target, current, prev, profile) -> tuple[qpos|None, report]
        """MPlib 位置 IK 回退。种子: prev_cmd(3次) → current_qpos(2次)"""
```

#### XHandRetargeter

```python
# teleop/vr/hand_retarget.py
class XHandRetargeter:
    def retarget(hand_joint_pos) -> np.ndarray | None
        """
        hand_joint_pos: (21, 3) landmarks in MANO frame
        returns: (12,) XHand joint angles
        内部调用: _build_ref_value → XHandRefAdapter.apply → dex_retargeting.retarget
        """
```

#### XHandRefAdapter

```python
# teleop/vr/ref_adapter.py
class XHandRefAdapter:
    def apply(ref_value, hand_joint_pos, origin_indices, task_indices) -> np.ndarray
        """
        粉红指自适应缩放:
          - 检测 pinky 伸展程度（指尖-指根距离）
          - 伸展时缩放 2.2x，卷曲时缩放 1.2x
          - pinky_blend=1.0 时完全启用适配
        """
```

#### RobotInterface

```python
# robot/interface.py
class RobotInterface:
    # ── Lifecycle ──
    def connect() -> dict[str, bool]       # {"arm": bool, "hand": bool}
    def disconnect() -> None
    def is_connected() -> bool
    
    # ── State ──
    def get_state() -> RobotState          # 完整状态，含 FK 计算
    def is_error() -> bool                 # arm 或 hand 有错误?
    def clear_error() -> bool
    
    # ── Action ──
    def send_action(action: RobotAction) -> dict
    def emergency_stop() -> None           # arm.stop() + hand.stop()
    def reset_soft_start() -> None         # TELEOP 入口重置
    
    # ── Workspace ──
    def check_workspace(pos: np.ndarray) -> bool
    
    # ── Home ──
    def return_to_home(use_planning=True, cancel_event=None) -> bool
```

#### ErrorHandler

```python
# teleop/core/error_handler.py
class TeleopErrorHandler:
    def init_fallback(arm_qpos, hand_qpos) -> None
        """初始化回退位置（仅设置未记录过的关节）"""
    
    def record_failure(stage, msg="") -> RobotAction
        """记录失败 + 返回 hold action"""
    
    def hold_action() -> RobotAction
        """返回 last_good 位置（无历史则返回零位）"""
    
    def update_good_positions(arm_qpos, hand_qpos) -> None
        """成功帧后更新 last_good"""
    
    def clear() -> None
        """状态转换时清除"""
```

---

## 6. 错误处理策略

### 6.1 三层错误响应

```
Layer 1: Per-Frame Soft Failures
  → quality flag = 0 (标记但不阻塞)
  → 继续管道
  - ARM_TORQUE_OK=0, HAND_CURRENT_OK=0, HAND_TEMP_OK=0, HAND_COMM_OK=0
  - RETARGET_VALID=0

Layer 2: Per-Frame Hard Failures
  → 根据失败类型不同，行为有差异：
  
  TRACKING_OK=0 (stale, 非连续丢失):
    → return 立即退出 _tick()，本帧不发送任何指令
    → 机器人自然保持在之前发送的最后指令位置
  
  IK_SUCCESS=0 / RETARGET_OK=0:
    → 在 _compute_action() 内部 fallback 到 prev_cmd
    → 继续管道，发送的是 hold 指令（= 上一帧的 cmd）
  
  JOINT_JUMP_OK=0:
    → clip delta 到上限 + 发送 clipped cmd
  
  IN_WORKSPACE=0:
    → hold last_good + 跳过 hand retarget

Layer 3: Escalation Failures
  → E-Stop (停止所有运动)
  → running = False
  → 管道终止
  - VR tracking lost > 1.0s
  - arm_joint_limits violation
  - robot.is_error() before send_action
  - Keyboard ESC
```

### 6.2 hold-last-good 机制

```python
# 每帧开始时初始化（从当前状态）
error_handler.init_fallback(current_arm_qpos, current_hand_qpos)

# 管道成功时更新
if ik_ok and retarget_ok:
    error_handler.update_good_positions(arm_cmd, hand_cmd)

# 管道失败时
# 自动返回 hold_action() = last_good qpos（零位仅在首次 fallback）
```

### 6.3 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| hand joint limits violation → Warning (not E-Stop) | Warning | XHand 有内部 commboard-level 错误保护，会在超限前故障停机 |
| arm joint limits violation → E-Stop | E-Stop | 关节限位是机械安全硬约束，不可妥协 |
| 累计 E-Stop 升级 | 已移除 | 逐帧 hold 足够安全；持久故障由 `robot.is_error()` 捕获 |
| NaN 在 safety check 中 | 视为 FAIL | `not np.all(np.isfinite(x))` → 返回 False（设计意图：宁可误报不可漏报） |
| ema_alpha_arm 默认 1.0 | 无平滑 | 当前 DLS 路径足够稳定（设计意图）；需要时通过参数配置 |

---

## 7. 数据流与生命周期

### 7.1 完整生命周期

```
                    connect()
                        │
                        ▼
                  ┌──────────┐
            ┌────►│  IDLE    │
            │     └────┬─────┘
            │          │ T (键盘)
            │          ▼
            │     ┌──────────┐
   return_to_home │ TELEOP   │◄───────────┐
            │     └────┬─────┘            │
            │          │ R (键盘)          │ S (键盘) → IDLE
            │          ▼                  │
            │     ┌──────────┐    S       │
            │     │RECORDING ├────────────┘
            │     └──────────┘
            │
            └────────────────── H (键盘) / HOME 完成
```

### 7.2 一帧内的数据流

```
┌──────────────────────────────────────────────────────────────────┐
│ Frame N                                                          │
│                                                                   │
│  VR Input ──────► vr_frame                                       │
│    tracker.get_latest()                                           │
│                                                                   │
│  Quality Gate ──► ok=True/False                                   │
│    tracking_quality.check(vr)                                     │
│                                                                   │
│  Robot State ────► RobotState                                     │
│    robot.get_state()                                              │
│    {arm_qpos(7), eef_pos(3), hand_qpos(12), torque(7), ...}      │
│                                                                   │
│  Arm IK ─────────► IKResult + arm_cmd(7)                          │
│    wrist → mapper.map() → target Pose                             │
│    target Pose → ik_solver.solve() → qpos                         │
│                                                                   │
│  Workspace ──────► in_workspace=True/False                        │
│    planner.compute_eef_pose_world(arm_cmd) → pos                  │
│    robot.check_workspace(pos)                                      │
│                                                                   │
│  Hand Retarget ──► hand_cmd(12)                                   │
│    landmarks → estimate_frame → OPERATOR2MANO                     │
│    → retargeter.retarget() → qpos                                 │
│                                                                   │
│  Jump Clamp ─────► arm_cmd', hand_cmd' (clipped)                  │
│                                                                   │
│  Safety ─────────► QualityFlags (11-bit)                            │
│    check_arm_torque, check_hand_current, etc.                     │
│                                                                   │
│  Record ─────────► HDF5 append (在 Execute 之前)                  │
│    recorder.add_frame(state, action, vr, flags, T_base_eef)       │
│                                                                   │
│  Execute ────────► {"arm_ok": bool, "hand_ok": bool}             │
│    robot.send_action(RobotAction(arm_cmd', hand_cmd'))            │
│                                                                   │
│  Wait ───────────► sleep(remaining)                                │
│    limiter.wait() → 补偿计算时间到 20ms budget                     │
└──────────────────────────────────────────────────────────────────┘
```

### 7.3 VR Frame 新鲜度时间线

```
Quest (50Hz)
  │ frame 1    frame 2    frame 3    frame 4    frame 5
  │ ts=0ms     ts=20ms    ts=40ms    ts=60ms    ts=80ms
  ▼
HTS SDK (TCP)
  │ recv=+5ms  recv=+3ms  recv=XXX   recv=+4ms  recv=+6ms
  ▼                  │                    │
Python get_latest()  │  frame 3 丢失      │
  │                   │                    │
  ▼                   ▼                    ▼
controller._tick()
  age=6ms       age=5ms        age=25ms       age=7ms
  OK             OK             STALE (>200ms) OK
                                hold last_good
                                
  tracking_lost timer:
    0s            0s            0.02s          0.0s (reset on recovery)
                                (started)
                                
  连续丢失 > 1.0s:
    → EMERGENCY_STOP
```

---

## 8. 性能预算

### 8.1 每帧时间预算 (Target: 20ms @ 50Hz)

```
┌────────────────────────────────────────────────────────────┐
│ Stage              │ Best Case │ Typical  │ Worst Case    │
├────────────────────┼───────────┼──────────┼───────────────┤
│ 1. VR Input        │   0.5ms   │   1.0ms  │   2.0ms       │
│ 2. Quality Gate    │   0.1ms   │   0.1ms  │   0.1ms       │
│ 3. Robot State     │   1.5ms   │   2.0ms  │   3.0ms       │
│ 4. Arm IK (DLS)    │   0.8ms   │   1.5ms  │   3.0ms       │
│ 4b. Arm IK (MPlib) │     —     │     —    │  10.0ms ⚠️    │
│ 5. Workspace Check │   0.1ms   │   0.1ms  │   0.2ms       │
│ 6. Hand Retarget   │   2.0ms   │   3.0ms  │   5.0ms       │
│ 7. Jump Clamp      │   0.05ms  │   0.05ms │   0.1ms       │
│ 8. Safety Checks   │   0.2ms   │   0.3ms  │   0.5ms       │
│ 9. Record Frame    │   0.3ms   │   0.5ms  │   1.0ms       │
│ 10. Execute Action │   0.8ms   │   1.0ms  │   2.0ms       │
├────────────────────┼───────────┼──────────┼───────────────┤
│ TOTAL (DLS path)   │   6.4ms   │   9.6ms  │  17.0ms ✓     │
│ TOTAL (MPlib path) │     —     │     —    │  27.0ms ⚠️    │
│ Budget             │  20.0ms   │  20.0ms  │  20.0ms       │
│ Margin (DLS)       │  68%      │  52%     │  15%          │
└────────────────────────────────────────────────────────────┘
```

### 8.2 延迟关键路径

```
最坏情况延迟链 (MPlib 回退触发时):
  VR(2ms) + State(3ms) + IK_MPlib(10ms) + Hand(5ms) + Send(2ms) = 22ms
  
  超出 20ms budget 2ms → RateLimiter 告警 "over budget"
  频率: DLS 成功率 ~95% → ~2.5 次/秒触发 MPlib 回退
```

### 8.3 缓存与资源

| 资源 | 用量 | 说明 |
|------|------|------|
| 主线程 CPU | ~40% (50Hz 下) | 单核，GIL 持有者 |
| VR 线程 CPU | ~5% | daemon thread, HTS SDK I/O |
| 内存 | ~50MB | HDF5 buffer + numpy arrays |
| HDF5 磁盘 IO | ~2MB/s | 50fps × ~40KB/frame |

---

## 9. 配置点

### 9.1 可调参数矩阵

| 参数 | 默认值 | 位置 | 调整建议 |
|------|--------|------|----------|
| `target_hz` | 50.0 | `TeleopController.__init__` | 降低到 30Hz 可放宽预算 |
| `ema_alpha_arm` | 1.0 (关闭) | `TeleopController.__init__` | 0.3-0.5 平滑手抖 |
| `max_frame_age_s` | 0.2s | `TrackingQualityConfig` | 增大容忍偶尔丢帧 |
| `ema_loss_timeout_s` | 1.0s | `TrackingQualityConfig` | 减小 → 更快触发 E-Stop |
| `differential_ik_damping` | 0.02 | `TeleopProfile` | 增大 → 更稳定但精度更低 |
| `differential_ik_gain` | 1.0 | `TeleopProfile` | 0.5-0.8 减速跟踪 |
| `use_position_ik` | True | `TeleopProfile` | False → DLS-only 模式 |
| `use_differential_ik_fallback` | True | `TeleopProfile` | False → 仅 position IK |
| `max_pose_error_pos_m` | 0.008 | `TeleopProfile` | 0.01 放宽 IK 精度要求 |
| `max_pose_error_rot_rad` | 0.08 | `TeleopProfile` | 0.1 放宽姿态精度 |
| `max_ik_jump_deg` | (30-60)° | `TeleopProfile` | 减小 → 更平滑但更慢 |
| `check_self_collision` | True | `TeleopProfile` | False → 更快但无碰撞保护 |
| `pos_scale` | 1.0 | `ArmWristMapper` | >1.0 放大运动 |
| `rot_scale` | 1.0 | `ArmWristMapper` | <1.0 减弱旋转 |
| `pinky_blend` | 1.0 | `XHandRefAdapter` | 0.0 关闭粉红指适配 |

### 9.2 配置文件层次

```
TeleopProfile (planning/types.py)          ← IK 参数
    │
    ▼
TeleopController.__init__ params           ← EMA, target_hz
    │
    ▼
RobotInterfaceConfig (robot/types.py)      ← workspace, desk, hand FK
    │
    ▼
XArm7PlannerConfig (planning/types.py)     ← URDF, vel limits, collision
    │
    ▼
CollisionConfig (planning/collision_config.py) ← 桌面几何参数
```

---

## 10. 测试策略

### 10.1 测试层级

| 层级 | 方式 | 覆盖 |
|------|------|------|
| **单元测试** | pytest, mock hardware | 每个 pipeline stage 的纯函数 |
| **集成测试** | `dry_run=True` + DummyTracker | 完整 pipeline，无硬件 |
| **仿真测试** | SAPIEN simulation | IK + 碰撞检测 + 工作空间 |
| **硬件测试** | 真机 (XArm7 + XHand) | 完整端到端 |

### 10.2 Dry-Run 模式

```python
# controller.py:example() — 无需硬件即可测试完整控制逻辑
controller = TeleopController(
    robot=robot,
    arm_mapper=arm_mapper,
    retargeter=retargeter,
    planner=planner,
    keyboard_queue=q,
    dry_run=True,           # 跳过所有硬件 I/O
    target_hz=50.0,
)
controller.tracker = DummyTracker()  # 提供虚拟 VR 数据
```

### 10.3 可注入接口

所有外部依赖通过构造函数注入，可被 mock 替换：

| 依赖 | mock 方式 |
|------|-----------|
| VR 数据源 | `DummyTracker` 或自定义 mock |
| 机器人硬件 | `dry_run=True` → `_dummy_state()` |
| 键盘输入 | `keyboard_queue` multiprocessing.Queue |
| 数据录制 | `recorder=None` 跳过录制 |

---

> **文档版本**: v1.0 | **最后更新**: 2026-06-22
> **与代码对齐**: `controller.py`, `ik.py`, `safety.py`, `tracking.py`, `quality_flags.py`, `interface.py`, `planner.py`, `kinematics.py`, `collision_config.py`, `types.py`
