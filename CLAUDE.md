# CLAUDE.md — DexMani Real

Dexterous manipulation teleop & data collection for **xArm7 (7-DOF arm) + XHand (12-DOF hand)** with VR control.

---

## Quick Reference

| Task | Key File(s) |
|------|------------|
| Modify arm control / servo loop | `dexmani_real/robot/inner_loop.py` |
| Modify hand control | `dexmani_real/robot/xhand/xhand.py` |
| Modify IK / retargeting pipeline | `dexmani_real/teleop/core/pipeline.py` |
| Modify IK solver | `dexmani_real/planning/ik.py` |
| Modify safety checks (validate) | `dexmani_real/robot/validate.py` |
| Modify state machine | `dexmani_real/teleop/core/controller.py` |
| Modify HDF5 recording format | `dexmani_real/recording/episode_recorder.py` |
| Modify recording lifecycle | `dexmani_real/recording/collection_loop.py` |
| Modify VR arm mapping | `dexmani_real/teleop/vr/arm_mapper.py` |
| Modify VR hand retargeting | `dexmani_real/teleop/vr/hand_retarget.py` |
| Modify camera pipeline | `dexmani_real/sensor/camera_process.py` |
| Modify point cloud processing | `dexmani_real/sensor/pointcloud_processor.py` |
| Modify shared memory / SHM | `dexmani_real/shm/` |
| Add a new entry point | `examples/real/` (real) or `examples/sim/` (sim) |
| Add a CLI tool | `dexmani_real/tools/` |
| Visualize an episode | `dexmani_real/tools/visualize_episode.py` |
| Health-check an episode | `dexmani_real/tools/check_episode_health.py` |
| Run tests | `conda run -n real_robot python -m pytest tests/ -v` |
| Type-check | `conda run -n real_robot mypy dexmani_real/` |

**Environment:** conda env `real_robot` (Python 3.10). All scripts run with `PYTHONPATH=.` from repo root.
Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot`

---

## Karpathy Guidelines (always active)

Before coding: **state assumptions, surface tradeoffs, ask if unclear.**
When coding: **minimum code, no speculative features, match existing style.**
When done: **verify with a test/run, not just "looks right."**

Full guidelines: invoke `/karpathy-guidelines` skill or see `.claude/skills/karpathy-guidelines/SKILL.md`.

---

## Project Structure

```
./
├── dexmani_real/          ← Python package
│   ├── robot/             ← XArm7 + XHand drivers, inner loop, validation, interface
│   │   ├── xarm7/         ← XArm7 SDK wrapper (xarm7.py, error_codes.py)
│   │   └── xhand/         ← XHand SDK wrapper (xhand.py, motor_trajectory_interpolator.py)
│   ├── teleop/            ← VR teleop controller + pipeline + retargeting
│   │   ├── core/          ← TeleopController (state machine), TeleopPipeline
│   │   ├── vr/            ← ArmWristMapper, XHandRetargeter, QuestHandTracker, DummyTracker
│   │   └── control/       ← KeyboardHandler, safety checks, audio feedback
│   ├── planning/          ← MPlib planner, IK, FK kinematics, collision, desk safety
│   ├── recording/         ← HDF5 recorder, RecordingSession, CollectionLoop, aligned buffer
│   ├── sensor/            ← RealSense driver, multi-camera manager, point cloud, VR receiver
│   ├── simulation/        ← SAPIEN simulation mirror of real hardware
│   ├── shm/               ← SharedMemoryRingBuffer for cross-process data
│   ├── config/            ← Camera extrinsics (CameraCalib)
│   ├── utils/             ← Logging, signal utils, rate limiting, array utils
│   └── tools/             ← CLI tools: episode viewer, health check, HDF5→Zarr export
├── examples/
│   ├── real/              ← Real-hardware entry points
│   └── sim/               ← Simulation entry points
├── tests/                 ← pytest test suite
├── docs/                  ← Architecture docs, analysis reports
├── assets/                ← URDF, SRDF, meshes, retargeting configs, audio
└── .claude/               ← Claude Code config (settings, skills, workflows)
```

---

## Architecture

### Data Flow (recording loop @ 16 Hz; ArmInnerLoop @ 50 Hz)

```
VR Tracker ──→ ArmWristMapper (wrist → EEF pose)  ──→ TeleopPipeline.compute_action()
              XHandRetargeter (landmarks → 12-DOF) ──┘   ├─ arm:  wrist pose → solve_teleop_ik
                                                          ├─ hand: MANO → NLP optimize → LPFilter EMA
                                                          │        → delta clip (≈0.098 rad/send @16Hz)
                                                          ├─ arm:  Cartesian EMA (TeleopPipeline default: pos α=0.8, rot α=0.4; check actual entry point overrides)
                                                          └─ arm:  IK anomaly jump-limit (default 90°)
                                                                       │
RobotInterface.validate_action() ← pre-send gate (error + connection + torque + temp
                                    + workspace clamp + joint-limit clip)
ArmInnerLoop.set_target(arm_qpos_cmd) ← arm → 50Hz inner loop (mode 6: firmware trajectory planning)
RobotInterface.send_action(action)    ← hand only (arm handled by ArmInnerLoop)
```

**Key rates:** Control loop 16 Hz, ArmInnerLoop 50 Hz (mode 6 — firmware handles trajectory smoothing).
ArmInnerLoop provides per-step joint delta clamp + dynamics readback (torque, temperature) to validate_action.

### State Machine (TeleopController)

```
IDLE ──B(begin+record)──→ TELEOP ⇄ C(pause) ⇄ PAUSED
  ↑                          │
  └── S(stop+save) / H(home) ┘
  ESC / VR-disconnect timeout → EMERGENCY_STOP
```

States (`ControllerState` enum): **IDLE, TELEOP, PAUSED, EMERGENCY_STOP** only.
Recording is a bool flag (not a state): set True on B, saved on S, discarded on Q.

### Core Types (`dexmani_real/robot/types.py`)

- **`RobotState`** — `arm_qpos(7)`, `arm_qvel(7)`, `arm_tau(7)`, `eef_pos(3)`, `eef_quat_wxyz(4)`, `eef_rot6d(6)`, `hand_qpos(12)`, `hand_tactile_sum(5,3)`, `hand_tactile_force(5,120,3)`, `fingertip_pos(5,3)`, `arm_connected`, `hand_connected`, `timestamp`
- **`RobotAction`** — `arm_qpos_cmd(7)`, `hand_qpos_cmd(12)`, optional `target_eef_pos`/`target_eef_rot6d`
- **`RobotInterfaceConfig`** — arm/hand configs, workspace bounds, collision config, hand URDF transforms
- **`RobotInterface`** — sole hardware access point; controllers NEVER call XArm7/XHand directly

### Safety Architecture

1. **Pre-send gate** (`validate_action()`): error gate → connection gate → torque gate → temperature gate → workspace clamp → arm clip → hand clip
2. **IK-level**: workspace clamping, IK anomaly jump-limit (arm: 90° default)
3. **Hand command-level**: delta clip (≈0.098 rad/send @16Hz) + optional EMA
4. **Path execution**: torque monitoring per waypoint, collision verification
5. **Desk safety**: `FingertipDeskSafety` — FK-based fingertip Z check (zero-cost, complements MPlib point cloud)
6. **Emergency stop**: `RobotInterface.emergency_stop()` → arm.stop() + hand.stop()

Torque limits: J1-J2=50, J3-J5=30, J6-J7=20 Nm. Temperature limit: 70°C per joint.

---

## Recording Format (HDF5)

Schema version **8**. All streams aligned to `dt=1/control_hz` time grid at record time (16 Hz).
Camera frames stream per-frame, index-aligned to grid (per-slot forward-fill).

**Key datasets:** `/arm_qpos(T,7)`, `/arm_ee(T,9)`, `/arm_qvel(T,7)`, `/arm_tau(T,7)`, `/hand_qpos(T,12)`, `/hand_fingertip(T,5,3)`, `/hand_contact(T,5,3)`, `/action_arm_joint(T,7)`, `/action_arm_ee(T,9)`, `/action_hand_joint(T,12)`, `/flag_ik_ok(T,)`, `/flag_retarget_ok(T,)`, `/flag_held(T,)`, `/flag_camera_fresh(T,)`, `/vr_wrist_pos(T,3)`, `/vr_wrist_rot6d(T,6)`, `/vr_landmarks(T,21,3)`, `/rgb(T,H,W,3)`, `/depth(T,H,W)`, `/pointcloud(T,2048,6)`, `/timestamp(T)`

**Meta attrs:** schema_version, control_hz, task_label, operator, tags, duration, fps, num_frames, success, stop_reason, camera_* (serial, type, K, extrinsics), pc_* (num_points, depth range, voxel_size), hand_* (delta_clip, ema_alpha, low_pass_alpha), ema_alpha_pos/rot, truncated, cam_frames_dropped

**Recording pipeline:** `EpisodeRecorder` (HDF5 I/O) → `CollectionLoop` (lifecycle) → `RecordingSession` (single writer thread, queue-based). Driver feeds `start()` / `record()` / `stop()`.

---

## Entry Points

| Entry Point | Purpose |
|-------------|---------|
| `examples/real/vr_teleop_shm.py` | **Main** real-hardware VR teleop (TeleopController + SHM VR) |
| `examples/real/vr_teleop_arm_only.py` | Arm-only VR teleop (direct recorder, no controller) |
| `examples/real/vr_teleop_arm_only_record.py` | Arm-only with recording |
| `examples/real/vr_teleop_arm_only_record_plus.py` | Arm-only extended recording |
| `examples/real/keyboard_teleop_real.py` | Keyboard-based arm control |
| `examples/real/test_quest_hand_teleop.py` | Standalone hand-retargeting test |
| `examples/real/test_motion_planning_real.py` | Motion planning test on hardware |
| `examples/real/test_pointcloud_process.py` | Point cloud processing test |
| `examples/real/test_pointcloud_stream.py` | Point cloud streaming test |
| `examples/real/replay_traj.py` | Replay a recorded trajectory |
| `examples/real/calibrate_camera.py` | Camera extrinsic calibration |
| `examples/real/calibrate_l515_depth.py` | L515 depth calibration (sigma_poly) |
| `examples/real/test_realsense.py` | RealSense camera connection test |
| `examples/sim/vr_teleop_sim.py` | VR teleop in SAPIEN simulation |
| `examples/sim/test_motion_planning_sim.py` | Motion planning in simulation |
| `dexmani_real/tools/visualize_episode.py` | 3D + camera + time-series HDF5 viewer |
| `dexmani_real/tools/check_episode_health.py` | Episode health check (grid fill, camera freeze, tracking error) |
| `dexmani_real/tools/export_hdf5_to_zarr.py` | HDF5→Zarr converter |

---

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
| Thread safety | `threading.Lock` for shared state; `threading.Event` for cancellation; GIL-protected numpy ops |
| Cross-process | `SharedMemoryRingBuffer` (shm/) — camera/VR processes ↔ controller |
| Git | `feat/` branches off `main`; commit messages in English |
| PR body | End with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)` |

---

## Anti-Patterns

- ❌ Calling XArm7/XHand directly → use `RobotInterface`
- ❌ Blocking calls in 50Hz loop → use async patterns or offload to separate processes
- ❌ Ignoring validate_action() → always call before send_action()
- ❌ Circular imports → use `TYPE_CHECKING` + lazy imports
- ❌ Mutable defaults in dataclass fields → use `field(default_factory=...)`
- ❌ Skipping `__init__.py` — new subpackages must have a docstring; `planning/`, `recording/`, and `simulation/` define `__all__` (match their pattern when adding public API)
- ❌ Hardcoding 50Hz assumptions → use `control_hz` from config (recording is 16 Hz)
- ❌ Adding state enum variants → Recording is a bool, not a ControllerState; no RECORDING/SAVE_PROMPT state

---

## Hardware Notes

**Intel RealSense L515** — must connect to a **direct motherboard USB 3.0 port** (no hub). Connecting through a hub causes isochronous packet loss and pipeline stall (~3s until frames stop). Verify: `lsusb -t` — L515 (8086:0b64) must be under a root hub port, not indented under `Hub`.

**L515 known issues:** Depth intrinsics can enter a bad state (missing VGA/XGA parameters → `rs.align` crash); `hardware_reset()` fixes it (not a calibration defect). See memory: [[l515-depth-intrinsics-bad-state]], [[l515-midrun-stream-stall]].

**xArm7 Mode 6:** Firmware online trajectory planning — arm target is forwarded directly at 50Hz. Firmware respects speed/accel limits (default: 90°/s, 500°/s²). No inner-loop interpolation needed.

**Key dependencies:** `mplib` (motion planning), `pinocchio` (rigid body dynamics), `h5py` (HDF5), RealSense SDK, XArm7/XHand SDKs (C++ bindings), `sapien` (simulation), `numpy`.
