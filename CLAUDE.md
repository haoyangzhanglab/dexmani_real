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
├── recording/         ← HDF5 episode recorder, RecordingSession, CollectionLoop, aligned buffer, validation
├── sensor/            ← RealSense camera driver, multi-camera manager, VR receiver
├── simulation/        ← SAPIEN-based simulation mirror of real hardware
├── shm/               ← SharedMemoryRingBuffer for cross-process camera data
├── config/            ← Camera extrinsics (cameras.json, CameraCalib)
├── utils/             ← Shared utilities (log, serialization, rate limiting, signal)
├── tools/             ← CLI utilities (HDF5 episode viewer, HDF5→Zarr export)
├── assets/            ← URDF, SRDF, meshes, retargeting configs
├── examples/          ← Real/sim teleop entry points + motion planning tests
│   ├── real/          ← Real-hardware examples (keyboard_teleop, test_motion, quest_teleop)
│   └── sim/           ← Simulation examples (vr_teleop_sim, test_motion)
├── docs/              ← Architecture docs, analysis notes
```

## Key data flow (decision/recording loop @ 16 Hz; ArmInnerLoop stays @ 50 Hz)

```
VR Tracker → ArmWristMapper (wrist→EEF pose)  ──→ TeleopPipeline.compute_action()
            XHandRetargeter (landmarks→12-DOF) ──┘   ├─ arm: wrist pose → solve_teleop_ik
                                                      ├─ hand: MANO skeleton retargeting
                                                      │   → adaptive scaling → NLP optimize
                                                      │   → LPFilter EMA (dex_retargeting built-in; τ-invariant:
                                                      │     0.6@50Hz → 0.943@16Hz, set via XHandRetargeter ctor)
                                                      │   → delta clip (XHand E3; entry point derives
                                                      │     deg2rad(90)/CTRL_HZ ≈ 0.098 rad/send @16Hz)
                                                      ├─ arm: Cartesian EMA (production SHM path: 1.0/1.0
                                                      │   pass-through — smoothing is Mode 6 firmware's job)
                                                      └─ arm IK anomaly jump-limit (default 90°, planning/ik.py)
                                                                   │
RobotInterface.validate_action() ← pre-send gate (error + connection + torque + temp + workspace clamp
                                    + arm clip + hand clip; env_collision accepted but not wired)
ArmInnerLoop.set_target(arm_qpos_cmd) ← arm cmd → 50Hz inner loop (mode 6: passthrough, firmware trajectory planning)
RobotInterface.send_action(action)    ← hand only (arm handled by ArmInnerLoop)
        │
    ┌── XArm7 (SDK C++ binding)  ← driven by ArmInnerLoop @ 50Hz (mode 6: firmware online trajectory planning)
    └── XHand (SDK C++ binding)  ← joint-limit + delta clip(E3) + optional EMA(E2)
```

## Core types (robot/types.py)

- **`RobotState`** — complete state: `arm_qpos(7)`, `arm_qvel(7)`, `arm_tau(7)`, `eef_pos(3)`, `eef_quat_wxyz(4)`, `eef_rot6d(6)`, `hand_qpos(12)`, `hand_tactile_sum(5,3)`, `hand_tactile_force(5,120,3)`, `fingertip_pos(5,3)`
- **`RobotAction`** — command: `arm_qpos_cmd(7)`, `hand_qpos_cmd(12)`, optional `target_eef_pos`/`target_eef_rot6d`
- **`RobotInterfaceConfig`** — arm/hand configs, workspace bounds, collision config, hand URDF transforms
- **`RobotInterface`** — sole hardware access point; controllers NEVER call XArm7/XHand directly

## State machine (TeleopController)

```
IDLE ──B(begin+record)──→ TELEOP ⇄ C(pause) ⇄ PAUSED
  ↑                          │
  └── S(stop+save) / H(home) ┘
  ESC / VR-disconnect timeout → EMERGENCY_STOP
```
- States (`ControllerState` enum): **IDLE, TELEOP, PAUSED, EMERGENCY_STOP** only.
- **Recording is a bool flag, not a state**: set True on BEGIN (starts together with TELEOP),
  saved on STOP (S) / discarded on QUIT (Q). There is no RECORDING or SAVE_PROMPT state.
- **PAUSED**: freeze IK, hold position (C key)
- Return-to-home: 2-phase — Phase 1: EEF Cartesian path, Phase 2: joint-space redundant joint alignment

## Motion planning (planning/planner.py)

- **XArm7MotionPlanner** — MPlib backend, delegates to `kin` (FK/Jacobian), `ik_mgr` (IK candidate gen), `mp_planner` (raw MPlib calls) via `__getattr__` proxy
- **`plan_path()`** — screw → multi-RRT fallback, validated through 8 checks (limits, elbow consistency, start depth, waypoint delta, terminal pose, self/env collision, workspace, desk safety)
- **`solve_teleop_ik()`** — teleop-optimized IK (seed from prev qpos, elbow consistency, desk safety preferred)
- **Collision modes**: `geometric_fk` (FK fingertip Z vs desk, zero-cost) or `mplib_pointcloud` (octree point cloud)
- **Path scoring**: joint_length * 1.0 + waypoint_delta * 2.0 + eef_inefficiency * 3.0

## HDF5 recording format

All streams are aligned to one `dt=1/control_hz` time grid at record time (`TimestampAlignedBuffer`,
16 Hz in production entry points; library default 50), keyed by `state.timestamp`; camera frames
stream per-frame, index-aligned to the grid (per-slot forward-fill).

```
episode_YYYYMMDD_HHMMSS.h5   # timestamp-named; +_N suffix on same-second collision
  /meta (group)
    attrs: schema_version(=7), control_hz, task_label, operator, tags, duration, fps, num_frames,
           success, min_frames_met, has_camera, has_pointcloud, has_timestamps,
           camera_serial, camera_type, camera_K, camera_T_world_camera, camera_T_eef_camera,
           skip_initial_frames, control_mode, arm_mode, hand_mode,
           arm_delta_clip, hand_delta_clip, hand_max_qvel_deg_s, hand_ema_alpha, hand_low_pass_alpha,
           ema_alpha_pos, ema_alpha_rot,
           pc_num_points, pc_depth_min_m, pc_depth_max_m, pc_workspace, pc_voxel_size,
           pc_radius_outlier_min_points, pc_radius_outlier_radius, pc_fps_backend
  /arm_qpos(T,7)           arm joint positions (rad)
  /arm_ee(T,9)             EEF [pos(3), rot6d(6)]
  /arm_qvel(T,7)           arm joint velocities (rad/s)
  /arm_tau(T,7)            arm joint torques (Nm)
  /hand_qpos(T,12)         hand joint positions (rad)
  /hand_fingertip(T,5,3)   fingertip positions in base frame
  /hand_contact(T,5,3)     tactile force sum per finger
  /action_arm_joint(T,7)   arm joint command
  /action_arm_ee(T,9)      target EEF [pos(3), rot6d(6)] (NaN if not set)
  /action_hand_joint(T,12) hand joint command
  /flag_ik_ok(T,)          bool — IK solved successfully
  /flag_retarget_ok(T,)    bool — retargeting converged
  /flag_held(T,)           bool — command held (no fresh VR/IK result)
  /flag_camera_fresh(T,)   bool — a new camera frame arrived within 0.2s (False = frozen/forward-filled)
  /vr_wrist_pos(T,3)       VR wrist position in base frame
  /vr_wrist_rot6d(T,6)     VR wrist orientation as 6D rotation
  /vr_landmarks(T,21,3)    VR hand landmarks (MANO convention)
  /rgb(T,H,W,3)            uint8 camera frames (forward-filled to grid)
  /depth(T,H,W)            uint16 Z16 depth, L515 validity-gated (forward-filled to grid)
  /pointcloud(T,2048,6)    float32 world-frame [xyz, rgb 0-1] — computed online @30Hz in
                           CameraProcess (sensor/pointcloud_processor.py), forward-filled
  /timestamp(T)            raw sample timestamps on the dt grid (62.5ms spacing @16Hz;
                           forward-filled slots repeat the previous raw value)
```
- **RecordingSession** (`recording/recording_session.py`) — driver-agnostic: one writer thread
  serializes start/record/stop so the HDF5 file is touched by a single thread (no teardown race).
- **CollectionLoop** orchestrates lifecycle: start/stop, sidecar JSON, discard (unlink). Single
  `data_dir` — `success` is a `/meta` attr + sidecar `classification`, **not** directory routing.

## Conventions

| Aspect | Convention |
|--------|-----------|
| Python | 3.10+, **conda env: `real_robot`** |
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
| `examples/real/vr_teleop_shm.py` | Main real-hardware VR teleop (TeleopController + SHM VR) |
| `examples/real/vr_teleop_arm_only.py` | Arm-only VR teleop (direct recorder, no controller) |
| `examples/real/keyboard_teleop_real.py` | Keyboard-based arm control |
| `examples/real/test_quest_hand_teleop.py` | Standalone hand-retargeting test (no TeleopController) |
| `examples/sim/vr_teleop_sim.py` | VR teleop in SAPIEN simulation |
| `dexmani_real/tools/visualize_episode.py` | Rerun-based HDF5 episode viewer (3D + camera + time series) |
| `dexmani_real/tools/export_hdf5_to_zarr.py` | HDF5→Zarr format converter |

## Safety architecture

1. **Pre-send gate** (`validate_action()`): robot error gate, arm connection gate, **torque gate**,
   **temperature gate** (both fed from ArmInnerLoop's 50Hz dynamics readback via `get_dynamics()`),
   workspace clamp, arm joint-limit clip, hand joint-limit clip.
   `env_collision_check` param is accepted but **not yet wired** — hard prerequisite before
   autonomous policy rollouts.
2. **IK-level**: workspace clamping, IK anomaly jump-limit (arm: 90° default, `planning/ik.py:140`)
3. **Hand command-level**: delta clip (E3 per-send hard gate; production entry derives
   `deg2rad(90)/CTRL_HZ` ≈ 0.098 rad @16Hz, library default 0.3) + optional EMA (E2, `XHandConfig.ema_alpha`)
4. **Retargeting-level**: built-in LPFilter EMA (dex_retargeting `SeqRetargeting.retarget()`;
   τ-invariant `low_pass_alpha` 0.6@50Hz → 0.943@16Hz via ctor)
5. **Path execution**: torque monitoring per waypoint, collision verification (self + env + desk FK)
6. **Desk safety**: `FingertipDeskSafety` — FK-based fingertip Z check (complements MPlib point cloud)
7. **Emergency stop**: `RobotInterface.emergency_stop()` → arm.stop() + hand.stop()

## Key dependencies

- `mplib` — motion planning library (IK, screw/RRT planning, collision detection)
- `pinocchio` — rigid body dynamics (FK, Jacobian)
- `h5py` — HDF5 serialization
- RealSense SDK — camera capture
- XArm7/XHand SDKs — hardware communication (C++ bindings)
- `sapien` — physics simulation
- `numpy` — all array math

## Hardware notes

**Intel RealSense L515** — must be connected to a **direct motherboard USB 3.0 port** (no intermediate hub). Connecting through a USB hub causes isochronous packet loss and pipeline stall (frames stop arriving after ~3 s). Verify with `lsusb -t`: the L515 (8086:0b64) must appear directly under a root hub port, not indented under a `Hub` node.

## Anti-patterns to avoid

- ❌ Calling XArm7/XHand directly → use `RobotInterface`
- ❌ Blocking calls in 50Hz loop → use async patterns or offload to separate processes
- ❌ Ignoring validate_action() → always call before send_action()
- ❌ Circular imports → use `TYPE_CHECKING` + lazy imports
- ❌ Mutable defaults in dataclass fields → use `field(default_factory=...)`
- ❌ Skipping `__init__.py` re-exports → each subpackage has explicit `__all__`
