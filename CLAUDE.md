# CLAUDE.md — DexMani Real

Dexterous manipulation teleoperation & data collection for **xArm7 (7-DOF arm) + XHand (12-DOF hand)** with VR control.

---

## LLM 行为准则 (Andrej Karpathy Skills)

以下准则偏向谨慎而非速度。对于简单任务，自行判断是否适用。

### 1. 先思考，再编码

**不假设。不隐藏困惑。明确权衡。**

- 明确陈述你的假设；如果不确定，主动询问。
- 如果存在多种解释，把它们都列出来，而不是默默选择一种。
- 如果存在更简单的方法，指出来，并在有充分理由时提出异议。
- 如果某些东西不清楚，停下来，明确指出困惑点，然后提问。

### 2. 简洁优先

**解决问题的代码越少越好。不要做推测性开发。**

- 不要添加超出需求的功能。
- 不要为单次使用的代码创建抽象层。
- 不要添加未被要求的"灵活性"或"可配置性"。
- 不要为不可能发生的场景添加错误处理。
- 如果 200 行可以变成 50 行，重写它。
- 自检：资深工程师会觉得这过度复杂吗？如果是，简化它。

### 3. 手术级修改

**只动必须动的。只清理你自己造成的混乱。**

编辑已有代码的规则：
- 不要"顺便"改进相邻代码、注释或格式。
- 不要重构没有坏的东西。
- 匹配现有风格，即使你的做法不同。
- 如果你注意到无关的死代码，提出来但不要删除。

针对你的修改产生的孤立产物的规则：
- 删除因你的修改而不再使用的 import/变量/函数。
- 不要删除已有的死代码，除非被要求。

测试：每一行改动都应直接追溯到用户的需求。

### 4. 目标驱动执行

**定义成功标准。循环直到验证通过。**

将任务转化为可验证的目标：
- "添加验证" → 先为无效输入编写测试，然后让它们通过
- "修复 bug" → 先写可复现的测试，然后让它通过
- "重构" → 确保重构前后测试都通过
- "实现功能 X" → 找到或编写测试，循环直到通过

多步骤任务使用编号计划 + 验证检查点。

强大的成功标准让你能独立循环迭代；模糊的标准（"让它跑起来"）需要不断向用户确认。

---

这些准则正在生效的标志：
- diff 中没有不必要的改动
- 没有因过度复杂而被要求重写
- 澄清性问题出现在实现之前，而非错误之后

---

## 调研规则

当用户说"调研某个项目"时，先在 `~/Desktop/Reference/` 目录下查找对应项目的代码仓库，基于实际代码进行分析，而非依赖记忆或猜测。

## Project overview

```
dexmani_real/          ← Python package root
├── robot/             ← Hardware drivers (XArm7, XHand) + unified RobotInterface
├── teleop/            ← VR teleop controller, state machine, pipeline, retargeting
│   ├── core/          ← TeleopController (state machine), TeleopPipeline, error handler
│   ├── vr/            ← ArmMapper, HandRetargeter, VRTracker (Quest), DummyTracker
│   └── control/       ← KeyboardHandler, safety checks (torque/current/temp/comm)
├── planning/          ← MPlib motion planner, IK, kinematics, collision detection
├── recording/         ← HDF5 episode recorder, CollectionLoop, frame buffer, validation
├── sensor/            ← RealSense camera driver, multi-camera manager, VR receiver
├── simulation/        ← SAPIEN-based simulation mirror of real hardware
├── shm/               ← SharedMemoryRingBuffer for cross-process camera data
├── config/            ← Camera extrinsics (cameras.json), PipelineConfig (serializable)
├── utils/             ← Shared utilities (log, serialization, rate limiting, signal)
├── tools/             ← CLI utilities (HDF5→Zarr export)
├── services/          ← Standalone services (retarget server)
├── assets/            ← URDF, SRDF, meshes, retargeting configs
├── examples/          ← Real/sim teleop entry points + motion planning tests
│   ├── real/          ← Real-hardware examples (keyboard_teleop, test_motion, quest_teleop)
│   └── sim/           ← Simulation examples (vr_teleop_sim, test_motion)
├── docs/              ← Architecture docs, analysis notes
```

## Key data flow (teleop loop @ 50 Hz)

```
VR Tracker → ArmWristMapper (wrist→EEF pose)  ──→ TeleopPipeline.compute_action()
            XHandRetargeter (landmarks→12-DOF) ──┘   ├─ arm: wrist pose → solve_teleop_ik
                                                      ├─ hand: MANO skeleton retargeting
                                                      ├─ EMA smoothing (fixed alpha, default 0.75)
                                                      └─ jump-limit safety gate (5°/10° arm/hand)
                                                                   │
RobotInterface.validate_action() ← pre-send gate (torque, current, temp, comm, workspace)
RobotInterface.send_action()    ← joint-limit + delta-limit clipping
         │
    ┌── XArm7 (SDK C++ binding)
    └── XHand (SDK C++ binding)
```

## Core types (robot/types.py)

- **`RobotState`** — complete state: `arm_qpos(7)`, `arm_qvel(7)`, `arm_tau(7)`, `eef_pos(3)`, `eef_quat_wxyz(4)`, `eef_rot6d(6)`, `hand_qpos(12)`, `hand_tactile_sum(5,3)`, `fingertip_pos(5,3)`
- **`RobotAction`** — command: `arm_qpos_cmd(7)`, `hand_qpos_cmd(12)`, optional `target_eef_pos`/`target_eef_rot6d`
- **`RobotInterfaceConfig`** — arm/hand configs, workspace bounds, collision config, hand URDF transforms
- **`RobotInterface`** — sole hardware access point; controllers NEVER call XArm7/XHand directly

## State machine (TeleopController)

```
IDLE ──T(teleop)──→ TELEOP ──R(record)──→ RECORDING
  ↑       │   S(stop)→IDLE      │   H(home)→IDLE
  ├───H(home)─────┘              │
  └──ESC / timeout: EMERGENCY_STOP
```
- **PAUSED** state: freeze IK, hold position (C key / foot pedal)
- **SAVE_PROMPT** state: confirm save/retry/discard after recording stop
- Return-to-home: 2-phase — Phase 1: EEF Cartesian path, Phase 2: joint-space redundant joint alignment

## Motion planning (planning/planner.py)

- **XArm7MotionPlanner** — MPlib backend, delegates to `kin` (FK/Jacobian), `ik_mgr` (IK candidate gen), `mp_planner` (raw MPlib calls) via `__getattr__` proxy
- **`plan_path()`** — screw → multi-RRT fallback, validated through 8 checks (limits, elbow consistency, start depth, waypoint delta, terminal pose, self/env collision, workspace, desk safety)
- **`solve_teleop_ik()`** — teleop-optimized IK (seed from prev qpos, elbow consistency, desk safety preferred)
- **Collision modes**: `geometric_fk` (FK fingertip Z vs desk, zero-cost) or `mplib_pointcloud` (octree point cloud)
- **Path scoring**: joint_length * 1.0 + waypoint_delta * 2.0 + eef_inefficiency * 3.0

## HDF5 recording format

```
episode_000.h5
  /meta: task_label, operator, tags, duration, fps, num_frames, success,
         camera_serial, camera_type, camera_K, T_base/eef_camera, pipeline_snapshot
  /obs: arm_qpos(7), arm_qvel(7), arm_tau(7), eef_pos(3), eef_quat(4),
        hand_qpos(12), hand_tactile_sum(5,3)
  /action: arm_qpos(7), hand_qpos(12)
  /vr: wrist_pos(3), wrist_quat(4), landmarks(21,3)
  /camera/<serial>/rgb(T,H,W,3), depth(T,H,W), timestamps(T)
  /timestamps(T), /vr_timestamps(T)
```
- **Batch mode** (default): `InMemoryFrameBuffer` flushes every 100 frames
- **CollectionLoop** orchestrates lifecycle: pre-record buffer (N seconds), start/stop, sidecar JSON, file routing (success_dir / failure_dir)

## Conventions

| Aspect | Convention |
|--------|-----------|
| Python | 3.10+, **conda env: `real`** |
| Formatting | black (line-length 120), isort (black profile), mypy (disallow_untyped_defs=false) |
| Imports | `from __future__ import annotations`; `TYPE_CHECKING` for circular deps |
| Data types | `dataclass` for config/state; `numpy` for all math |
| Naming | `snake_case` files/vars/funcs, `PascalCase` classes, `UPPER_SNAKE` module constants |
| Logging | `from dexmani_real.utils.log import get_logger` → `logger = get_logger(__name__)` |
| Error handling | fail-safe (NaN→neutral, errors→warning+fallback); try/except with `logger.warning` |
| Hardware access | ONLY via `RobotInterface`; never call XArm7/XHand directly |
| Thread safety | `threading.Event` for cancellation; `ExitStack` for cleanup; GIL-protected numpy ops |
| Cross-process | `SharedMemoryRingBuffer` (shm/) — camera/VR processes ↔ controller |

## Entry points

| Entry point | Purpose |
|-------------|---------|
| `examples/real/test_quest_hand_teleop.py` | Main real-hardware VR teleop |
| `examples/real/keyboard_teleop_real.py` | Keyboard-based arm control |
| `examples/sim/vr_teleop_sim.py` | VR teleop in SAPIEN simulation |
| `dexmani_real/services/retarget_server.py` | Standalone hand retargeting service (ZMQ REP) |
| `dexmani_real/tools/export_hdf5_to_zarr.py` | HDF5→Zarr format converter |

## Safety architecture

1. **Pre-send gate** (`validate_action()`): robot error, arm connection, arm torque, hand current/temp/comm, workspace bounds
2. **IK-level**: workspace clamping, IK anomaly jump limits (5° arm / 10° hand)
3. **Path execution**: torque monitoring per waypoint, collision verification (self + env + desk FK)
4. **Desk safety**: `FingertipDeskSafety` — FK-based fingertip Z check (complements MPlib point cloud)
5. **Emergency stop**: `RobotInterface.emergency_stop()` → arm.stop() + hand.stop()
6. **SlidingWindowMonitor**: trend tracking for hand temperature, current, IK miss counts

## Key dependencies

- `mplib` — motion planning library (IK, screw/RRT planning, collision detection)
- `pinocchio` — rigid body dynamics (FK, Jacobian)
- `h5py` — HDF5 serialization
- RealSense SDK — camera capture
- XArm7/XHand SDKs — hardware communication (C++ bindings)
- `sapien` — physics simulation
- `numpy` — all array math

## Anti-patterns to avoid

- ❌ Calling XArm7/XHand directly → use `RobotInterface`
- ❌ Blocking calls in 50Hz loop → use async patterns or offload to separate processes
- ❌ Ignoring validate_action() → always call before send_action()
- ❌ Circular imports → use `TYPE_CHECKING` + lazy imports
- ❌ Mutable defaults in dataclass fields → use `field(default_factory=...)`
- ❌ Skipping `__init__.py` re-exports → each subpackage has explicit `__all__`
