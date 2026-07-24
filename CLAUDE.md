# CLAUDE.md -- DexMani Real

Dexterous manipulation teleop & data collection for **xArm7 (7-DOF) + XHand (12-DOF)** with VR control.
**Env:** conda `real_robot` (Python 3.10). All scripts: `PYTHONPATH=.` from repo root.
Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot`

---

## Quick Reference

| Task | Key File(s) |
|------|------------|
| Arm servo loop | `dexmani_real/robot/inner_loop.py` |
| Hand control | `dexmani_real/robot/xhand/xhand.py` |
| Arm/hand process isolation | `robot/arm_process.py`, `hand_process.py` |
| IK / retargeting pipeline | `teleop/core/pipeline.py`, `planning/ik.py` |
| Safety checks (validate) | `robot/validate.py` |
| State machine | `examples/real/vr_teleop_arm_only_record_plus.py` (inline bool flags, not enum) |
| Recording format / lifecycle | `recording/episode_recorder.py`, `collection_loop.py`, `collection_config.py` |
| VR arm/hand mapping | `teleop/vr/arm_mapper.py`, `hand_retarget.py` |
| Camera / point cloud | `sensor/camera_process.py`, `pointcloud_processor.py` |
| Shared memory / SHM | `shm/robot_ring.py`, `robot_layouts.py`, `robot_rpc.py` |
| Episode tools (viz/health/export) | `tools/` |
| Type-check | `conda run -n real_robot mypy dexmani_real/` |

---

## Architecture

### Data Flow

```
VR Tracker --> ArmWristMapper (wrist -> EEF pose)  --> TeleopPipeline.compute_action()
              XHandRetargeter (landmarks -> 12-DOF) --+   +- arm:  solve_teleop_ik (MPlib)
                                                          +- hand: MANO -> NLP optimize
                                                          +- arm:  Cartesian EMA (pos a=0.5, rot a=0.25)
                                                          +- arm:  elbow-flip detection + IK fallback
                                                                       |
RobotInterface.validate_action() <- pre-send gate (error + connection + NaN + torque + temp)
ArmInnerLoop.set_target(arm_qpos_cmd) <- arm -> 16Hz forwarded (mode 6: firmware trajectory planning)
RobotInterface.send_action(action)    <- hand only (arm handled by ArmInnerLoop)
```

**Rates:** Control loop 16 Hz. Mode 6 handles all trajectory smoothing (120 deg/s, 500 deg/s^2). ArmInnerLoop: joint delta clamp (0.3 rad/step), velocity-adaptive tracking error, torque/temp readback.

### Process Isolation & State Machine

Arm/hand run in crash-isolated fork processes via SHM -- no fallback. `arm_process.py` (ArmControlProcess/ArmSHMFacade), `hand_process.py` (HandControlProcess/HandSHMFacade), `shm/robot_ring.py` (SeqlockRingBuffer, odd/even torn-read protocol), `shm/robot_layouts.py` (numpy dtypes).

State machine: inline bool flags per entry point. IDLE, TELEOP, PAUSED, EMERGENCY_STOP. Recording = bool.

### Core Types (`robot/types.py`) & RobotInterface (`robot/interface.py`)

- **`RobotState`** -- `arm_qpos(7) arm_qvel(7) arm_tau(7) eef_pos(3) eef_quat_wxyz(4) eef_rot6d(6) hand_qpos(12) hand_tactile_sum(5,3) hand_tactile_force(5,120,3) fingertip_pos(5,3) arm_connected hand_connected timestamp:float`
- **`RobotAction`** -- `arm_qpos_cmd(7) hand_qpos_cmd(12)` + optional `target_eef_pos/rot6d`
- **`RobotInterface`**: sole hardware access. `get_state()`, `send_action()` (hand only), `return_to_home()` (3-tier), `emergency_stop()`, `set_arm_servo()`.

---

## Key Invariants

1. All hardware through `RobotInterface`; never XArm7/XHand directly
2. Always `preflight_check()` before use + `validate_action()` before `send_action()`
3. Recording grid-aligned to 16 Hz (`dt=1/control_hz`) -- breaking alignment corrupts downstream
4. Mode 6 handles trajectory; do NOT interpolate arm commands (double-interpolation -> overshoot)
5. State = bool flags, recording = bool -- not an enum

---

## Known Footguns

- **C24 mid-motion:** IK spike -> hold-on-failure -> ramp reset -> overspeed trip (`c24-ramp-reset-midmotion.md`)
- **Frozen camera:** L515 mid-run silent stall ~35-60s; forward-fill masks it (`l515-midrun-stream-stall.md`)
- **ENOSPC false positive:** Disk check races with async writer (`arm-only-record-session-2026-07-18.md`)
- **Velocity tuning ineffective:** Mode 6 bottleneck is acc/jerk, not velocity (`mode6-tracking-error-root-cause.md`)
- **`sync_primitives.py` missing:** Imported under `if TYPE_CHECKING:` in `inner_loop.py:30`; safe but confusing

---

## Recording Format

HDF5 v8-10 (auto-selected). All streams grid-aligned to 16 Hz. Video sidecar: opt-in MP4. Pipeline: `TimestampAlignedBuffer` -> `EpisodeRecorder` (async writer) -> `CollectionLoop` -> `CollectionConfig`. Field catalog: `episode_recorder.py` docstring.

---

## Safety Architecture

1. **Pre-send gate:** error -> connection -> NaN -> torque (J1-J2=50, J3-J5=30, J6-J7=20 Nm) -> temperature (70 C)
2. **ArmInnerLoop:** delta clamp (0.3 rad/step) + velocity-adaptive tracking + mode-drift + reco (22=self-collision, 24=overspeed)
3. **IK-level:** workspace clamping + elbow-flip detection + hold-on-failure
4. **Pre-flight:** `preflight_check(robot)` before every entry point
5. **Desk safety + E-stop:** `FingertipDeskSafety` (FK-based Z check); `emergency_stop()` -> arm.stop() + hand.stop()

---

## Conventions

| Aspect | Convention |
|--------|-----------|
| Python | 3.10+, **conda: `real_robot`** |
| Formatting | black (line-length 120), isort (black profile), mypy |
| Imports | `import numpy as np` (universal); `from __future__ import annotations` (preferred); `if TYPE_CHECKING:` for circular deps |
| Logger | `logger = get_logger(__name__)` after ALL imports, before any class/function |
| Types / Naming | `dataclass` for config/state, `numpy` for math; `snake_case`, `PascalCase`, `UPPER_SNAKE` |
| Error handling | fail-safe (NaN->neutral); always `logger.warning("msg", exc_info=True)` |
| Hardware / Threading | ONLY `RobotInterface`; `Lock`/`Event` + SHM rings; `FromDictMixin` (`utils/serialization.py`) |

---

## Anti-Patterns

- Calling XArm7/XHand directly; using RobotInterface without `preflight_check()` or skipping `validate_action()`
- Blocking I/O in 16Hz loop (camera read, file write -> silent frame drop)
- Assuming hand is connected without `hand_connected` check
- Mutating RobotState/RobotAction arrays in-place (shape validation only at construction)
- Interpolating arm commands in app code (Mode 6 double-interpolation -> overshoot)
- `logger.warning(f"foo: {e}")` without `exc_info=True` (loses stack)
- Circular imports without `TYPE_CHECKING` + lazy imports
- Mutable defaults in dataclass fields -- use `field(default_factory=...)`
- Hardcoding rate assumptions (use `control_hz` from config) or adding state enum variants (recording = bool, not a state)

---

## Typical Edit Patterns

| When you... | Also update... |
|-------------|---------------|
| Add a field to `RobotState` | `interface.py` (`get_state()`) + `robot_layouts.py` (SHM dtype) + `episode_recorder.py` (dataset) |
| Add a recording dataset | `episode_recorder.py` + `episode_reader.py` + `check_episode_health.py` |
| Change IK solver | `planning/ik.py` + `teleop/core/pipeline.py` + `validate.py` (clamping) + `inner_loop.py` (delta may need retune) |
| New entry point | Include `preflight_check()` + follow pattern from `vr_teleop_arm_only_record_plus.py` |
| Tune arm dynamics | `inner_loop.py` (delta clamp) + `validate.py` (tracking threshold); velocity alone has near-zero impact |

---

## Entry Points

**Primary:** `examples/real/vr_teleop_arm_only_record_plus.py` (main: teleop+recording+audio), `vr_teleop_arm_only_record.py`, `vr_teleop_arm_only.py`, `replay_traj.py`.
**Diag:** `calibrate_camera.py`, `calibrate_l515_depth.py`, `keyboard_teleop_real.py`, `test_*.py`.
**Sim:** `examples/sim/vr_teleop_sim.py`, `test_motion_planning_sim.py`.

---

## Hardware Notes

**xArm7 Mode 6:** Firmware trajectory planning, targets at 16 Hz (120 deg/s, 500 deg/s^2). No inner-loop interpolation. Tracking error bottleneck is acc/jerk, not velocity.

**L515:** Direct motherboard USB 3.0 only (no hub; verify `lsusb -t`, 8086:0b64 under root hub). Depth intrinsics bad state: `hardware_reset()`. Mid-run stream stall ~35-60s. XU flaky: use `set_option` fallback.

**Deps:** `mplib`, `pinocchio`, `h5py`, RealSense SDK, XArm7/XHand SDKs, `sapien`, `numpy`, `pyav`.