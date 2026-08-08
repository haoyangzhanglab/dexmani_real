# CLAUDE.md -- DexMani Real

Dexterous manipulation teleop & data collection for **xArm7 (7-DOF) + XHand (12-DOF)** with VR control.
**Env:** conda `real_robot` (Python 3.10). All scripts: `PYTHONPATH=.` from repo root.
Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot`

---

## Quick Reference

| Task | Key File(s) |
|------|------------|
| **Main entry point** | `examples/real/vr_teleop_hand_record.py` (canonical, 5-process arch) |
| Keyboard teleop | `examples/real/keyboard_teleop_real.py` — optional XHand, SharedStorage, `defaults.keyboard_teleop` |
| Camera calibration | `examples/real/calibrate_camera.py` — ArUco eye-to-hand, `CameraCalibConfig` |
| Trajectory replay | `examples/real/replay_traj.py` — episode replay + consistency metrics |
| VR heading calib | `examples/real/calibrate_vr_heading.py` — one-shot T_vr_to_robot |
| **Policy process** | `policy/vr_teleop_policy.py` — `policy_loop(shared, config)`, reads rings, writes actions, owns recording |
| Planner factory | `planning/planner.py` — `XArm7MotionPlanner.create_default()` canonical setup |
| **SharedStorage (data plane)** | `shm/shared_storage.py` — all rings, queues, flags in one place |
| Arm servo loop | `robot/arm_loop.py` — `arm_loop(shared)` (canonical), Mode 6 |
| Hand control | `robot/hand_process.py` — `hand_loop(shared)` (canonical) |
| VR receiver | `sensor/vr_receiver_process.py` — `vr_loop(shared)` writes to `vr_ring` |
| Camera | `sensor/camera_process.py` — independent process, unchanged |
| IK / retargeting | `planning/ik.py`, `planning/kinematics.py`, `teleop/arm_mapper.py`, `teleop/hand_retarget.py` |
| Safety state machine | `robot/safety.py` — SafetyState enum (DISARMED/ARMED/RUNNING/FAULT) + transition helpers |
| Recording format / lifecycle | `recording/episode_recorder.py` |
| SHM primitives | `shm/ring_buffer.py` (CameraRingBuffer + SeqlockRingBuffer base), `shm/robot_ring.py` (SeqlockRingBuffer + `get_last_k(k)` multi-frame read) |
| Core types | `robot/types.py` — RobotState, RobotAction + ArmState/HandState/HandTactile (doc-only dataclasses; authoritative format = `*_DTYPE` in `shm/shared_storage.py`) |
| Episode tools (quality/viz) | `dexmani_real/tools/` — `episode_quality.py` (filter/health/assess/validate), `visualize_episode.py` (3D + tactile) |
| Type-check | `conda run -n real_robot mypy dexmani_real/` |

---

## Architecture

### 5-Process Model + Thin Main

```
Main (~150 lines) — spawns 5 processes, monitors is_running
  │
  ├─ Camera (camera_loop) ──camera_ring──┐
  ├─ VR (vr_loop) ──────vr_ring─────┤
  │                               ▼
  ├─ Policy (policy_loop) ──arm_action_q──→ Arm (arm_loop)
  │                ──hand_cmd_ring─→ Hand (hand_loop)
  │                ◄──arm_state_ring, hand_state_ring, hand_tactile_ring
  │                owns EpisodeRecorder (single-clock recording)
  │
  ├─ Arm (arm_loop, Mode 6, 30Hz) — reads arm_action_q, servos xArm7, writes arm_state_ring
  └─ Hand (hand_loop, 30Hz) — reads hand_cmd_ring, servos XHand, writes hand_state_ring + hand_tactile_ring
```

**Key principles (from ManiUniCon):**
- Main does NOT touch data plane — only orchestrates
- SharedStorage is the sole data plane — one class, all rings/queues/flags
- Processes exchange structured data only — no Python objects, no RPC
- Policy owns recording — single-process TimestampAlignedBuffer, natural alignment

### Data Flow

```
VR Tracker → ArmWristMapper → EMA → WorkspaceClamp → solve_teleop_ik → DeltaClamp
                                                                              │
                                                          ┌───────────────────┘
                                                          ▼
                                         shared.arm_action_q.put(ArmAction)
                                         shared.hand_cmd_ring.write(HandCmd)
```

**Rates:** Policy loop 16 Hz. Arm/Hand servo loops 30 Hz. Normal teleoperation sends
one target per policy tick and relies on Mode 6 firmware smoothing. Return-home
densely samples direct and staged joint-space candidates offline for collision
validation, with a bounded MPlib joint-space fallback when those heuristics are
blocked. It then temporarily switches to Mode 0 and sends only the validated
milestones as unblended MoveJoint targets. The arm worker restores Mode 6 on
healthy exits and acknowledges completion only after fresh controller feedback
converges.

### SharedStorage Data Plane (`shm/shared_storage.py`, ~340 lines)

| Transport | Type | Direction | Semantics |
|-----------|------|-----------|-----------|
| `arm_action_q` | mp.Queue(maxsize=2) | Policy → Arm | Ordered, bounded backpressure |
| `arm_home_result_q` | mp.Queue(maxsize=2) | Arm → requester | Correlated `HomeResult` ACK; success/failure is never inferred from stale state |
| `hand_cmd_ring` | SeqlockRingBuffer(8) | Policy → Hand | Latest-wins (position servo) |
| `arm_state_ring` | SeqlockRingBuffer(8) | Arm → Policy | Read-latest; `get_last_k(k)` k-帧历史 (~265B) |
| `hand_state_ring` | SeqlockRingBuffer(8) | Hand → Policy | Read-latest; `get_last_k(k)` k-帧历史 (~472B, no tactile_force) |
| `hand_tactile_ring` | SeqlockRingBuffer(8) | Hand → Policy | Sparse writes (~14.4KB, only on contact) |
| `vr_ring` | SeqlockRingBuffer(8) | VR → Policy | ~600B/frame |
| `camera_ring` | CameraRingBuffer(5) | Camera → Policy | ~1.5MB/frame |
| `is_running` | mp.Value | Main → all | Sole writer: Main |
| `is_recording` | mp.Value | Policy → Arm/Hand/Camera | Sole writer: Policy |
| `error_state` | mp.Value | Arm/Hand → all | Sticky latch (set-only) |
| `estop_request` | mp.Value | Policy → Arm/Hand | ESC key |
| `safety_state` | mp.Value('i') | Main + Policy → all | SafetyState enum (0-3). Main: DISARMED↔ARMED, →FAULT. Policy: ARMED↔RUNNING |
| `arm_heartbeat_s` | mp.Value('d') | Arm → Main | `time.monotonic()` per tick, timeout=1.0s |
| `hand_heartbeat_s` | mp.Value('d') | Hand → Main | `time.monotonic()` per tick, timeout=1.0s |
| `policy_heartbeat_s` | mp.Value('d') | Policy → Main | `time.monotonic()` per tick, timeout=1.0s |
| `vr_heartbeat_s` | mp.Value('d') | VR → Main | `time.monotonic()` per event, timeout=5.0s |
| `camera_heartbeat_s` | mp.Value('d') | Camera → Main | `time.monotonic()` per tick, timeout=2.0s |

### Process Entries

```python
# Each function is an mp.Process target, accepting SharedStorage + optional config:
arm_loop(shared, config)    # robot/arm_loop.py — Mode 6 servo, FK, tracking error
hand_loop(shared, config)   # robot/hand_process.py — XHand position servo, sets error_state
policy_loop(shared, config) # policy/vr_teleop_policy.py — VR→IK + recording, sets is_recording
vr_loop(shared)             # sensor/vr_receiver_process.py — HTS TCP
camera_loop(shared)         # Main — bridges frames from CameraSession → shared.camera_ring
```

### Core Types (`robot/types.py`)

- **`ArmState`** — joint/EEF/status fields plus last accepted command sequence, producer/receive/SDK-return monotonic timestamps, derived queue/apply/SDK latency, HOLD flag, and state timestamp (322B, from arm_state_ring; eef via Pinocchio ArmFK in arm_loop, tracking_err = max|qpos - last_target|)
- **`HandState`** — `qpos(12) current(12) tactile_sum(5,3) tactile_contact(5) error_state connected qpos_stale commboard_err(12) jointboard_err(12) tipboard_err(12) timestamp` (472B, from hand_state_ring)
- **`HandTactile`** — `tactile_force(5,120,3)` (14.4KB, from hand_tactile_ring, sparse)
- **`RobotState`** — legacy 22-field monolithic state (Policy assembles from ArmState+HandState+HandTactile for recording)
- **`RobotAction`** — `arm_qpos_cmd(7) hand_qpos_cmd(12)` + optional `target_eef_pos/rot6d`

---

## Key Invariants

1. **All cross-process data through SharedStorage** — never direct SDK calls across processes
2. **Policy owns recording** — single-clock domain, natural (state, action, camera) alignment
3. **Mode 6 handles trajectory** — do NOT interpolate arm commands (double-interpolation → overshoot)
4. **Arm Queue (maxsize=2)** — bounded backpressure; Policy blocks if Arm falls behind
5. **Hand Ring (latest-wins)** — position servo; old targets overwritten
6. **Recording grid-aligned to 16 Hz** (`dt=1/control_hz`) — breaking alignment corrupts downstream
7. **State = bool flags, recording = bool** — not an enum. **Safety state IS an enum** (SafetyState, 0-3), stored in `shared.safety_state`
8. **Seqlock on all control rings** — torn-read protection for arm_state and hand_cmd
9. **`get_last_k(k)` returns oldest-first, ≤k frames** — callers must handle `len(result) < k`; k > maxlen raises ValueError; each frame independently seqlock-verified; overwritten frames silently dropped

---

## Safety State Machine (ManiUniCon P0 — 2026-08-02)

Four-state machine per ManiUniCon §13.2:

```
DISARMED(0) --[Main: all ready]--> ARMED(1) --[Policy: B key]--> RUNNING(2)
     ^                                |  ^                           |  |
     |                                v  |                           v  |
     +---[Main: Q/shutdown]----------+  +---[Policy: C/S/D/H]------+  |
                                                                       |
     FAULT(3) <--[Main: error_state | proc death | heartbeat timeout]-- ANY
       |
       +--[Main: shutdown only]--> DISARMED(0)
```

- **Main** owns: DISARMED↔ARMED, →FAULT, →DISARMED
- **Policy** owns: ARMED↔RUNNING (teleop start/stop)
- **Arm/Hand** read-only: gate servo on `safety_state in (ARMED, RUNNING)`
- **5 process heartbeats** (`time.monotonic()` per tick) monitored by Main at 10Hz
- **Heartbeat timeouts** (from `config/defaults.py` `safety.heartbeat_timeouts`): arm/hand/policy=1.0s, vr=5.0s, camera=2.0s
- **Existing bool flags preserved** (`is_running`, `error_state`, `estop_request`) — state machine is additive
- **See**: `robot/safety.py` for SafetyState enum + transition validation

---

## Simplified Safety Architecture

### Design principle: firmware is the safety backstop

xArm7 Mode 6 firmware already enforces: C22 (self-collision), C24 (velocity), C31 (collision-induced current),
collision detection, torque limit. Application-level collision checks reject invalid
commands before they reach firmware; firmware remains the final safety backstop.

### Coordinated safety layers (single-writer)

1. **Arm-level:** NaN guard (protects `last_target`)
   + Mode 6 error handling (C22/C31 → immediate sticky FAULT; C24 has bounded
   `clean_error+set_state+set_mode` recovery; repeated C24 failures → FAULT)
   + `except Exception` path also escalates to FAULT after `_RECOVERY_MAX` consecutive failures
2. **Policy-level:** arm connected gate + NaN guard + workspace clamp + conservative
   asynchronous arm-hand transition envelope + downward contact-stall pose resync
   near the tabletop (context only; not a table exclusion zone) +
   safety_state gate (ARMED required for B, FAULT blocks send) + hand_qpos_stale hold
3. **IK-level:** workspace clamping + elbow-flip detection + hold-on-failure + delta clamp
4. **E-stop:** Policy sets `estop_request=True` → Arm/Hand detect flag → `set_state(4)`
5. **Error state:** sticky latch (`error_state` mp.Value) — Arm/Hand set, Main detects → FAULT
6. **FK zero-pose guard:** throttled warning on FK failure (code≠0 or exception) — consumers
   see zero EEF with log trail
7. **Heartbeat supervisor:** Main monitors 5 process heartbeats at 10Hz → FAULT on timeout
8. **Safety state machine:** formal DISARMED/ARMED/RUNNING/FAULT states with validated
   transitions (Main owns DISARMED↔ARMED/→FAULT, Policy owns ARMED↔RUNNING)
9. **Return-home:** the caller holding the collision planner densely validates
   self-collision, arm-hand, workspace and table clearance along every segment,
   then sends only safe milestones. `arm_loop` temporarily uses firmware Mode 0
   MoveJoint point-to-point planning, restores Mode 6, waits for real joint feedback
   at each milestone, and replies on
   `arm_home_result_q` using the request ID. VR homing is policy-owned;
   Main never tries to move workers after `is_running=False` or `DISARMED`.

### Key safety features

- **Hand qpos_stale hold**: prevents gap jump on driver board lockout recovery
- **Recovery counter FAULT escalation**: `_RECOVERY_MAX=30` — persistent errors trigger FAULT instead of silent infinite retry. Separate counters for servo and state-read errors.
- Archived safety simplifications: arm joint-limit clip, policy error-code gate, VR quat gate, startup_error, tactile gate, stale target timeout — all removed in favor of firmware backstop + heartbeat supervisor.

---

## Known Footguns

- **C24 mid-motion**: ~~IK spike → hold-on-failure → ramp reset → overspeed trip~~ (fixed: hold-on-failure preserves last valid target)
- **Frozen camera**: L515 mid-run silent stall ~35-60s; forward-fill masks it
- **ENOSPC false positive**: Disk check races with async writer
- **Velocity tuning ineffective**: Mode 6 bottleneck is acc/jerk, not velocity
- **Arm Queue backpressure**: `maxsize=2` means Policy blocks if Arm falls >125ms behind — monitor with status print

---

## Recording Format

HDF5 v13. All streams are grid-aligned (normally 16 Hz). Pipeline: `TimestampAlignedBuffer` → `EpisodeRecorder` (accumulate-then-dump, async writer). `flag_sample_valid` distinguishes source samples from grid back-fills. `/meta/fps` and `control_hz` denote the nominal grid rate; `duration` retains its legacy wall-clock meaning, while `wall_duration_s`, `grid_duration_s`, `grid_dt_s`, `non_sampled_duration_s`, and `wall_fps` make pauses and prompt gates explicit. Raw arm/hand targets and policy-stage timing datasets support latency diagnosis.

Key hand-related datasets in `data.h5` (full catalog: `episode_recorder.py:add_frame()`):
- `hand_qpos` (T,12), `hand_fingertip` (T,5,3), `hand_contact` (T,5,3), `hand_tactile_force` (T,5,120,3), `hand_tactile_contact` (T,5)
- `hand_current` (T,12), `hand_connected` (T,), `hand_qpos_stale` (T,), `hand_error_state` (T,)
- `hand_tipboard_err` / `hand_commboard_err` / `hand_jointboard_err` (T,12)
- `action_hand_joint` (T,12), `flag_retarget_ok` (T,), `flag_frame_status` (T,)

`episode_quality health` reports measured `hand_qpos` excursions outside the strict SDK command bounds. The independent `hand_feedback_bound_tolerance_rad` metadata/config value classifies sub-degree settling error without widening command or optimizer bounds. The XHand driver counts every finite 30 Hz feedback read, throttles only over-tolerance warnings, and logs the aggregate/per-joint totals at worker exit; the episode report computes the analogous statistics on valid recorded source frames.

`hand_tactile_ring` publishes sparsely (contact-only); `hand_state_ring` publishes every tick (30 Hz).

---

## Conventions

| Aspect | Convention |
|--------|-----------|
| Python | 3.10+, **conda: `real_robot`** |
| Formatting | black (line-length 120), isort (black profile), mypy |
| Imports | `import numpy as np` (universal); `from __future__ import annotations` (preferred); `if TYPE_CHECKING:` for circular deps |
| Logger | `logger = get_logger(__name__)` after ALL imports, before any class/function |
| Types / Naming | `dataclass` for config/state, `numpy` for math; `snake_case`, `PascalCase`, `UPPER_SNAKE` |
| Error handling | fail-safe (NaN→neutral); always `logger.warning("msg", exc_info=True)` |
| Process isolation | mp.Process targets are plain functions (`*_loop(shared)`), not class methods |
| Lazy SDK imports | SDK imports inside process functions (not at module level) — avoids import errors in Main |

---

## Anti-Patterns

- Calling XArm7/XHand SDK from Policy or Main (SDK imports only in arm_loop/hand_loop)
- Creating SHM rings outside SharedStorage (use `shared.xxx_ring`)
- Blocking I/O in 16Hz loop (camera read, file write → silent frame drop)
- Assuming hand is connected without checking `hand_state.connected`
- Mutating RobotState/RobotAction arrays in-place (shape validation only at construction)
- Interpolating arm commands in app code (Mode 6 double-interpolation → overshoot)
- `logger.warning(f"foo: {e}")` without `exc_info=True` (loses stack)
- Circular imports without `TYPE_CHECKING` + lazy imports
- Mutable defaults in dataclass fields — use `field(default_factory=...)`
- Hardcoding rate assumptions (use `control_hz` from config)
- Silently swallowing exceptions without logging (`pass` in except — always `logger.warning(..., exc_info=True)`)
- Putting business logic in Main (Main = spawn + monitor + shutdown, nothing else)

---

## Typical Edit Patterns

| When you... | Also update... |
|-------------|---------------|
| Add a field to ArmState/HandState | `shared_storage.py` (dtype) + `types.py` (dataclass) + arm_loop/hand_loop (write) + policy (read) |
| Add a recording dataset | `episode_recorder.py` (add_frame data dict) + `episode_reader.py` + `episode_quality.py` |
| Add a hand health flag (bool) | `types.py` (RobotState, default False after hand_current) + `policy/vr_teleop_policy.py` (_build_robot_state read + else-branch) + `episode_recorder.py` (add_frame data dict) |
| Change IK solver | `planning/ik.py` + `policy/vr_teleop_policy.py` |
| Add a new ring to SharedStorage | `shared_storage.py` + producer process + consumer process |
| New entry point (new architecture) | Follow Main pattern: `SharedStorage.create()` → spawn `*_loop(shared)` → monitor |
| Tune arm dynamics | `arm_loop.py` (ArmLoopConfig) + Mode 6 acc/jerk; velocity alone has near-zero impact |

---

## Entry Points

- **Primary**: `examples/real/vr_teleop_hand_record.py` — canonical 5-process entry, `--task`/`--operator`/`--acc`/`--speed`/`--no-hand`/`--config`/`--print-config`
- **Keyboard teleop**: `examples/real/keyboard_teleop_real.py` — optional XHand, SharedStorage; 8 mm maximum translation step, continuous EMA state across key-release edges, no redundant release command in the ordered arm FIFO, and throttled/buffered `RELTRACE` pre/post-release motion diagnostics
- **Camera calibration**: `examples/real/calibrate_camera.py` — ArUco eye-to-hand
- **Trajectory replay**: `examples/real/replay_traj.py` — episode replay + consistency metrics
- **VR heading calib**: `examples/real/calibrate_vr_heading.py`

---

## Hardware Notes

**xArm7 motion modes:** Mode 6 online replanning handles normal 16 Hz teleoperation;
do not add interpolation to that stream. Its per-joint velocity profiles need not be
synchronous, so homing densely validates joint-space segments, temporarily enters
Mode 0, and sends sparse unblended MoveJoint targets. Firmware still owns trajectory
generation; the worker restores Mode 6 on healthy exits and stops on E-stop/FAULT.
Homing defaults to 30°/s and
acknowledges completion only from fresh encoder feedback. See UFACTORY's official
[Mode 0/6 description](https://github.com/xarm-developer/xarm_ros#57-xarm_apixarm_msgs-online-planning-modes-added).

**XHand:** 12-DOF EtherCAT position servo. Latest-wins semantics (hand_cmd_ring). Tactile: 5 fingers × 120 taxels × 3 axes. Board errors auto-logged.
`keyboard_teleop_real.py` probes XHand optionally by default (`--require-hand`
for strict startup, `--no-hand` to skip probing); `calibrate_camera.py` is
arm-only. Both seed the collision model with configured open-hand geometry when
measured hand state is unavailable. Canonical data collection remains fail-closed.

**L515:** Direct motherboard USB 3.0 only (no hub; verify `lsusb -t`, 8086:0b64 under root hub). Depth intrinsics bad state: `hardware_reset()`. Mid-run stream stall ~35-60s. XU flaky: use `set_option` fallback.

**Quest VR:** HTS TCP on port 8000. `adb reverse tcp:8000 tcp:8000` for USB. `vr_loop` handles coordinate conversion (Unity left-hand → FLU).

**Deps:** `mplib`, `pinocchio`, `h5py`, RealSense SDK, XArm7/XHand SDKs, `numpy`, `pyav`.

---

## Changelog (key milestones)

All items below are resolved — see git log for full history.

- **P0**: Safety state machine (DISARMED/ARMED/RUNNING/FAULT) + 5-process heartbeats + Main supervisor
- **P0**: `SeqlockRingBuffer.get_last_k(k)` multi-frame read + `read_arm_state_k`/`read_hand_state_k`
- **P1**: Code dedup — `hand_home_converge()`, `_seed_hand_retargeter()`, shared ring helpers
- **P1**: Episode quality toolkit — held-frame filter, tracking-error filter, health/validate/assess
- **P2**: camera_loop extracted to `sensor/camera_process.py`
- **P3-P7**: All entry points migrated to SharedStorage architecture; old entry points deleted
- **Config**: `SharedStorageConfig` centralized; defaults.py frozen singletons + JSON override
- **Teleop**: Directory flattened (control/vr/core removed); dead code deleted (~690 lines)
- **Robot**: `xarm7/` subpackage deleted; inner_loop→arm_loop rename; hand legacy classes removed
- **Safety simplification**: Redundant gates removed (firmware is the backstop)
- **Ultracode review**: 23 fixes (arm_loop FAULT escalation, NaN guards, camera metadata race, etc.)
- **Code review rounds 1-2**: NameError fixes, CLI parameter plumbing, resource leak fixes, import cleanup
- **Dead code**: `robot/interface.py`, `validate.py`, `preflight.py`, `arm_process.py`, `collision_config.py`, deprecation classes, unused imports — all removed
