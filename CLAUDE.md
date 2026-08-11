# CLAUDE.md -- DexMani Real

Dexterous manipulation teleop & data collection for **xArm7 (7-DOF) + XHand (12-DOF)** with VR control.
**Env:** conda `real_robot` (Python 3.10). All scripts: `PYTHONPATH=.` from repo root.
Activate: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate real_robot`

---

## Quick Reference

| Task | Key File(s) |
|------|------------|
| **Main entry point** | `examples/real/vr_teleop_hand_record.py` (spawn-only canonical runtime) |
| Learned policy entry | `examples/real/learned_policy_real.py` — manifest/hash validated, isolated inference, explicit B-to-run |
| Keyboard teleop | `examples/real/keyboard_teleop_real.py` — optional XHand, SharedStorage, `defaults.keyboard_teleop` |
| Camera calibration | `examples/real/calibrate_camera.py` — ArUco eye-to-hand, `CameraCalibConfig` |
| Trajectory replay | `examples/real/replay_traj.py` — dry-run by default; certified v15 `sent`-stream live replay |
| VR heading calib | `examples/real/calibrate_vr_heading.py` — one-shot T_vr_to_robot |
| **Policy coordinator** | `policy/vr_teleop_policy.py` — owns grid/episode decisions and the only safe-command path |
| Action protocol | `policy/action_protocol.py` — SafetyGate, prepare/commit, ACKs, chunk scheduler, epoch quiesce |
| Optional inference | `policy/runtime.py`, `policy/inference_process.py`, `policy/tensor_block.py` — backend-neutral contracts and isolated worker |
| Planner factory | `planning/planner.py` — `XArm7MotionPlanner.create_default()` canonical setup |
| IPC schemas | `ipc/schema.py` — dependency-neutral NumPy dtypes for state, action, inference, recording, VR, and camera headers |
| **SharedStorage (data plane)** | `shm/shared_storage.py` — all rings, queues, flags in one place |
| Arm servo loop | `robot/arm_loop.py` — `arm_loop(shared)` (canonical), Mode 6 |
| Hand control | `robot/hand_process.py` — `hand_loop(shared)` (canonical) |
| VR receiver | `sensor/vr_receiver_process.py` — `vr_loop(shared)` writes to `vr_ring` |
| Camera | `sensor/camera_process.py` — independent capture/pointcloud process with source-freshness metadata |
| IK / retargeting | `planning/ik.py`, `planning/kinematics.py`, `teleop/arm_mapper.py`, `teleop/hand_retarget.py` |
| Safety state machine | `robot/safety.py` — SafetyState enum (DISARMED/ARMED/RUNNING/FAULT) + transition helpers |
| Recording format / lifecycle | `recording/io_process.py`, `recording/episode_recorder.py` — dedicated RecorderIO + schema v15 |
| SHM primitives | `shm/ring_buffer.py` (CameraRingBuffer + SeqlockRingBuffer base), `shm/robot_ring.py` (SeqlockRingBuffer + `get_last_k(k)` multi-frame read) |
| Core types | `robot/types.py` — RobotState, RobotAction + ArmState/HandState/HandTactile (doc-only dataclasses; authoritative IPC format = `*_DTYPE` in `ipc/schema.py`) |
| Episode tools (quality/viz) | `dexmani_real/tools/` — `episode_quality.py` (filter/health/assess/validate), `visualize_episode.py` (3D + tactile) |
| Type-check | `conda run -n real_robot mypy dexmani_real/` |

---

## Architecture

### Spawn-only capability process model + thin Main

```
Main — resolves immutable config, creates SharedStorage, spawns capabilities,
supervises health, and performs verified shutdown
  │
  ├─ Camera (camera_loop) ──camera_ring──┐
  ├─ VR (vr_loop) ──────vr_ring─────┤
  │                               ▼
  ├─ PolicyCoordinator ──SafetyGate/prepare/commit──→ Arm + optional Hand
  │                ◄──arm_state_ring, hand_state_ring, hand_tactile_ring
  │                ──bounded record_sample_ring──→ optional RecorderIO
  │
  ├─ Arm (arm_loop, Mode 6, 30Hz) — reads arm_action_q, servos xArm7, writes arm_state_ring
  ├─ Hand (hand_loop, 30Hz) — reads prepared/committed endpoints, writes state/tactile/ACK
  └─ RecorderIO (recording capability only) — writes, verifies, fsyncs, and atomically publishes episodes
```

The learned-policy entry adds an isolated Inference worker and a two-slot
seqlock tensor block. PolicyCoordinator alone reads SharedStorage, constructs
causal 16 Hz snapshots, normalizes only backend identity into the current
session/epoch/action-ID domain, preserves backend-created target/expiry times,
and schedules prepare/commit work at 64 Hz. Expired backend chunks are never
retimed or revived.
Model imports occur only in Inference after spawn; the ordinary VR entry never
loads an unselected model. A camera clock reset increments the camera
generation, forces RUNNING back to ARMED, invalidates the old policy epoch,
performs a fresh backend warmup/finite-output check, and requires the operator
to press B again; it is never treated as a seamless hot swap.

An inference manifest names `backend_entrypoint` (`module:Class`), `actuators`,
the 16 Hz `modalities`/history contract, the joint-position `action` contract,
and every model/preprocessor resource as a relative path plus SHA-256. Startup
fails while DISARMED on an unknown modality, hash mismatch, insufficient ring
capacity or memory budget, warmup exception/NaN, invalid output shape, or a
missed benchmark deadline. Camera payload publication is enabled only when a
selected modality requests it. `camera_ready` is published only after the
worker has captured and published one verified RGB-D frame (with either a valid
point cloud or the configured invalid point-cloud placeholder).

**Key principles (from ManiUniCon):**
- Main does NOT touch data plane — only orchestrates
- SharedStorage is the sole data plane — one class, all rings/queues/flags
- Processes exchange structured data only — no Python objects, no RPC
- Policy owns episode/grid/sample decisions; RecorderIO owns only serialization and transactional publication
- Every process and multiprocessing primitive comes from `mp.get_context("spawn")`

### Data Flow

```
VR Tracker → ArmWristMapper → shortest-arc EMA → solve_teleop_ik → raw candidate
                                                                              │
                                                          ┌───────────────────┘
                                                          ▼
                       ActionSafetyGate (freshness/limits/delta/workspace/table/collision)
                                                          ↓
                          correlated arm+hand prepare → ACK → commit → apply ACK
```

**Rates:** Policy coordination (heartbeat, operator input, RecorderIO status) runs at
64 Hz; causal observations, actions, and recording stay on the 16 Hz grid. Arm/Hand
servo loops run at 30 Hz. Normal teleoperation sends one target per grid tick and
relies on Mode 6 firmware smoothing. Return-home
densely samples direct and staged joint-space candidates offline for collision
validation, with a bounded MPlib joint-space fallback when those heuristics are
blocked. It then temporarily switches to Mode 0 and sends only the validated
milestones as unblended MoveJoint targets. The arm worker restores Mode 6 on
healthy exits and acknowledges completion only after fresh controller feedback
converges.

### SharedStorage Data Plane (`shm/shared_storage.py`)

All payload layouts live in `ipc/schema.py`; SharedStorage allocates and owns
their transports but does not import policy, inference, or recording
implementations merely to discover a dtype.

| Transport | Type | Direction | Semantics |
|-----------|------|-----------|-----------|
| `arm_action_q` | mp.Queue(maxsize=2) | Policy → Arm | Ordered, bounded backpressure |
| `arm_home_result_q` | mp.Queue(maxsize=2) | Arm → requester | Correlated `HomeResult` ACK; success/failure is never inferred from stale state |
| `hand_cmd_ring` | SeqlockRingBuffer(8) | Policy → Hand | Latest-wins (position servo) |
| `action_commit_ring` | SeqlockRingBuffer | Policy → Arm/Hand | Correlated commit after all enabled workers PREPARED |
| `arm_ack_ring`, `hand_ack_ring` | SeqlockRingBuffer | Workers → Policy | RECEIVED/PREPARED/APPLIED/REJECTED/SDK_FAILED plus confirmed STOPPED |
| `arm_state_ring` | SeqlockRingBuffer(8) | Arm → Policy | Read-latest; `get_last_k(k)` k-帧历史 (~265B) |
| `hand_state_ring` | SeqlockRingBuffer(8) | Hand → Policy | Read-latest; `get_last_k(k)` k-帧历史 (~472B, no tactile_force) |
| `hand_tactile_ring` | SeqlockRingBuffer(8) | Hand → Policy | Every successful read, including release; freshness/calibration/unit metadata |
| `vr_ring` | SeqlockRingBuffer(8) | VR → Policy | ~600B/frame |
| `camera_ring` | CameraRingBuffer(5) | Camera → Policy | Strict 640×480 RGB/depth + 2048×6 pointcloud; device/capture timestamps and validity |
| `record_sample_ring` | SeqlockRingBuffer(4) | Policy → RecorderIO | Fixed camera/control payload; overflow aborts the episode |
| `arm_metrics_ring`, `hand_metrics_ring`, `camera_metrics_ring`, `policy_metrics_ring` | dedicated SeqlockRingBuffer(8) | single worker → Main | ~1 Hz loop health; actual Hz, work-time max, overruns, missed slots, long-block reanchors; warning-only |
| `is_running` | mp.Value | Main → all | Sole writer: Main |
| `is_recording` | mp.Value | Policy → Arm/Hand/Camera | Sole writer: Policy |
| `error_state` | mp.Value | Safety-critical workers → all | Sticky latch (set-only) |
| `estop_request` | mp.Value | Policy → Arm/Hand | ESC key |
| `safety_state` | mp.Value('i') | Main + Policy → all | SafetyState enum (0-3). Main: DISARMED↔ARMED, →FAULT. Policy: ARMED↔RUNNING |
| `arm_heartbeat_s` | mp.Value('d') | Arm → Main | `time.monotonic()` per tick, timeout=1.0s |
| `hand_heartbeat_s` | mp.Value('d') | Hand → Main | `time.monotonic()` per tick, timeout=1.0s |
| `policy_heartbeat_s` | mp.Value('d') | Policy → Main | `time.monotonic()` per tick, timeout=1.0s |
| `vr_heartbeat_s` | mp.Value('d') | VR → Main | `time.monotonic()` per event, timeout=5.0s |
| `camera_heartbeat_s` | mp.Value('d') | Camera → Main | Worker-liveness heartbeat; source freshness comes from frame capture monotonic time |
| `recorder_heartbeat_s` | mp.Value('d') | RecorderIO → Main | Writer-process liveness, distinct from episode health |

### Process Entries

```python
# Each function is an mp.Process target, accepting SharedStorage + optional config:
arm_loop(shared, config)    # robot/arm_loop.py — Mode 6 servo, FK, tracking error
hand_loop(shared, config)   # robot/hand_process.py — XHand position servo, sets error_state
policy_loop(shared, config) # policy/vr_teleop_policy.py — VR→IK + recording, sets is_recording
recorder_io_loop(shared, config) # recording/io_process.py — HDF5/video transaction worker
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
2. **Policy owns recording decisions/grid** — RecorderIO may write bytes but never chooses samples
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
- **Enabled-capability heartbeats** (`time.monotonic()` per tick) monitored by Main at 10Hz
- **Heartbeat timeouts** (from `config/defaults.py` `safety.heartbeat_timeouts`): arm/hand/policy=1.0s, vr=5.0s, camera=2.0s
- **Loop metrics are diagnostic-only.** Heartbeats remain the liveness/fault signal; performance overruns never transition safety state by themselves.
- **Existing bool flags preserved** (`is_running`, `error_state`, `estop_request`) — state machine is additive
- **See**: `robot/safety.py` for SafetyState enum + transition validation

---

## Simplified Safety Architecture

### Design principle: firmware is the safety backstop

xArm7 Mode 6 firmware already enforces: C22 (self-collision), C24 (velocity), C31 (collision-induced current),
collision detection, torque limit. Application-level collision checks reject invalid
commands before they reach firmware; firmware remains the final safety backstop.

### Collision detection layers and static scene

The Pinocchio collision validator owns two deliberately separate geometry
models. The SRDF-filtered self model preserves the meaning of
`check_self_collision*`: robot↔robot only. A second model contains only
robot↔static-obstacle pairs. Combined point, detail, dense segment, and
asynchronous arm×hand transition queries are used by teleop IK candidate
filtering/final review, `ActionSafetyGate`, planned paths, return-home,
band-alignment, and replay preflight. MPlib still proposes paths; the combined
model performs dense post-validation and does not add arm-side interpolation.

Static boxes are optional and default to an empty scene:

```json
{
  "environment": {
    "static_boxes": [
      {
        "name": "fixture_left",
        "center_xyz_m": [0.35, -0.42, 0.20],
        "size_xyz_m": [0.20, 0.05, 0.40],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
      }
    ]
  }
}
```

Centers and full side lengths are metres in the xArm base frame; quaternions
rotate box-local axes into that frame. Names are unique and `table` is reserved
for the existing independent table-clearance layer. Non-finite values,
non-positive sizes, and non-unit quaternions fail configuration resolution.
Collision-query exceptions fail closed at command/path boundaries. Replay
preflight certificate v2 binds URDF/SRDF contents plus the ordered normalized
scene; v1 remains readable but is accepted only for an empty static scene.

### Coordinated safety layers (single-writer)

1. **Arm-level:** NaN guard (protects `last_target`)
   + Mode 6 error handling (C22/C31 → immediate sticky FAULT; C24 has bounded
   fresh-feedback measured-hold recovery; a second C24 in the bounded window → FAULT)
   + `except Exception` path also escalates to FAULT after `_RECOVERY_MAX` consecutive failures
2. **Policy-level:** mandatory ActionSafetyGate applies finite/shape, freshness,
   epoch/TTL, joint limits, dt-aware delta, workspace, table clearance, and a
   conservative asynchronous arm-hand transition envelope before raw IPC; plus
   safety-state gating and hand tracking-stall hold. Pause and stale-VR edges
   increment `policy_epoch`, publish one coordinated measured arm/hand hold,
   require the exact APPLIED ACKs, then re-anchor on fresh measured FK/VR before
   active mapping resumes; audio playback is not part of this correctness path.
3. **IK-level:** workspace clamping + elbow-flip detection + hold-on-failure + delta clamp
4. **E-stop:** Policy sets `estop_request=True` → Arm/Hand detect flag → `set_state(4)`
5. **Error state:** sticky latch (`error_state` mp.Value) — Arm/Hand set, Main detects → FAULT
6. **FK zero-pose guard:** throttled warning on FK failure (code≠0 or exception) — consumers
   see zero EEF with log trail
7. **Heartbeat supervisor:** Main monitors every enabled capability at 10Hz → FAULT on timeout
8. **Safety state machine:** formal DISARMED/ARMED/RUNNING/FAULT states with validated
   transitions (Main owns DISARMED↔ARMED/→FAULT, Policy owns ARMED↔RUNNING)
9. **Return-home:** the caller holding the collision planner densely validates
   self-collision, arm-hand, static environment, workspace and table clearance along every segment,
   then sends only safe milestones. `arm_loop` temporarily uses firmware Mode 0
   MoveJoint point-to-point planning, restores Mode 6, waits for real joint feedback
   at each milestone, and replies on
   `arm_home_result_q` using the request ID. VR homing is policy-owned;
   Main never tries to move workers after `is_running=False` or `DISARMED`.

### Key safety features

- **Hand qpos_stale hold**: prevents gap jump on driver board lockout recovery
- **Recovery counter FAULT escalation**: `_RECOVERY_MAX=30` — persistent errors trigger FAULT instead of silent infinite retry. Separate counters for servo and state-read errors.
- **VR pose validity gate:** the receiver publishes only finite, exact-shape poses/landmarks with normalized nonzero quaternions; the mapper independently fails closed and clears an invalid reset anchor.
- Archived safety simplifications: policy error-code gate, startup_error, tactile gate, and stale target timeout were removed in favor of the coordinated safety protocol and heartbeat supervisor.

---

## Known Footguns

- **C24 mid-motion**: ~~IK spike → hold-on-failure → ramp reset → overspeed trip~~ (fixed: hold-on-failure preserves last valid target)
- **Frozen camera**: L515 may silently stall; v15 marks duplicate/old slots stale and discards the active episode after 2s without changing teleop safety state
- **Camera writer failure**: queue saturation, codec crash, and ENOSPC are episode-fatal; only the temp directory is removed, never a partial published episode
- **Velocity tuning ineffective**: Mode 6 bottleneck is acc/jerk, not velocity
- **Arm Queue backpressure**: `maxsize=2` means Policy blocks if Arm falls >125ms behind — monitor with status print

---

## Recording Format

HDF5 v15 is additive; the high-level reader keeps v12–v14 raw-readable but marks
their semantic validity `UNKNOWN`. Training and live replay require `VALID` v15
episodes. The pre-training validator and filter reject `UNKNOWN` by default;
`--allow-legacy-offline` is limited to explicit migration/diagnostic work. Policy
emits a fixed 16 Hz sample stream to the bounded RecorderIO
ring; RecorderIO owns `data.h5`, `rgb.mp4`, `depth.h5`, and `pointcloud.h5`,
drains and reopens every output, verifies all lengths/shapes/dtypes and decoded
video length, fsyncs file and directory, then atomically renames the hidden temp
directory. Overflow, codec, ENOSPC, or camera-source faults abort only the
episode and leave teleoperation available.

`source_sample_index`, `source_timestamp`, `fill_reason`, and validity flags make
causal hold-last/leading placeholders explicit. Observation source/receive/
publish times, age/skew/history masks, raw and safe actions, commit/ACK
identities, tactile source time/freshness/calibration/unit, pointcloud
source/padding statistics, the canonical resolved config, and hashed
robot/device/calibration resources provide v15 provenance. Arm and hand workers
publish canonical device-identity JSON from values already exposed by their
SDK connection; missing vendor fields remain `unavailable` rather than being
guessed. A held source row that intentionally emits no new command uses
`action_id=0` with queued/committed both false; a published hold is bound to its
exact action candidate and ACKs. Old dataset meanings are unchanged.

Key hand-related datasets in `data.h5` (full catalog: `episode_recorder.py:add_frame()`):
- `hand_qpos` (T,12), `hand_fingertip` (T,5,3), `hand_contact` (T,5,3), `hand_tactile_force` (T,5,120,3), `hand_tactile_contact` (T,5)
- `hand_current` (T,12), `hand_connected` (T,), `hand_qpos_stale` (T,), `hand_error_state` (T,)
- `hand_tipboard_err` / `hand_commboard_err` / `hand_jointboard_err` (T,12)
- `action_hand_joint` (T,12), `flag_retarget_ok` (T,), `flag_frame_status` (T,)

`episode_quality health` reports measured `hand_qpos` excursions outside the strict SDK command bounds. The independent `hand_feedback_bound_tolerance_rad` metadata/config value classifies sub-degree settling error without widening command or optimizer bounds. The XHand driver counts every finite 30 Hz feedback read, throttles only over-tolerance warnings, and logs the aggregate/per-joint totals at worker exit; the episode report computes the analogous statistics on valid recorded source frames.

`hand_tactile_ring` publishes every successful device read, including release;
`hand_state_ring` also publishes every tick (30 Hz).
`hand.ethercat_slave_position=-1` means unverified/unknown: shutdown skips the
vendor INIT request and relies on close plus watchdog. Configure a non-negative
position only after an installation-specific hardware check; it is then covered
by the resolved-config hash.

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
| Add a field to ArmState/HandState | `ipc/schema.py` (dtype) + `types.py` (dataclass) + arm_loop/hand_loop (write) + policy (read) |
| Add a recording dataset | `episode_recorder.py` (add_frame data dict) + `episode_reader.py` + `episode_quality.py` |
| Add a hand health flag (bool) | `types.py` (RobotState, default False after hand_current) + `policy/vr_teleop_policy.py` (_build_robot_state read + else-branch) + `episode_recorder.py` (add_frame data dict) |
| Change IK solver | `planning/ik.py` + `policy/vr_teleop_policy.py` |
| Add a new ring to SharedStorage | `shared_storage.py` + producer process + consumer process |
| New entry point (new architecture) | Follow Main pattern: `SharedStorage.create()` → spawn `*_loop(shared)` → monitor |
| Tune arm dynamics | `arm_loop.py` (ArmLoopConfig) + Mode 6 acc/jerk; velocity alone has near-zero impact |

---

## Entry Points

- **Primary**: `examples/real/vr_teleop_hand_record.py` — spawn-only VR runtime with camera, VR, policy, arm, optional hand, and optional RecorderIO; recording remains enabled by default and `--no-record` removes that process; `--task`/`--operator`/`--acc`/`--speed`/`--no-hand`/`--no-record`/`--config`/`--print-config`
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

**L515:** The device is retired; librealsense 2.50.0 is the last project
baseline known to support it. Do not blindly upgrade the SDK. Every v15 episode
records the observed SDK/firmware, actual stream profile, alignment mode,
intrinsics/distortion, depth scale, and encoding settings. Use a direct
motherboard USB 3.0 connection (no hub; verify `lsusb -t`, 8086:0b64 under root
hub) and verify the active profile is 640×480@30 Hz. Depth intrinsics bad state:
`hardware_reset()`. XU is flaky, so use the existing `set_option` path. A source
stall marks short gaps stale and discards the current recording after 2s;
teleoperation remains RUNNING and no camera-only FAULT is raised.

**Quest VR:** HTS TCP on port 8000. `adb reverse tcp:8000 tcp:8000` for USB. `vr_loop` handles coordinate conversion (Unity left-hand → FLU).

**TAG hand retargeting:** policy hand FK and TAG optimization use the same resolved
`hand_urdf_path`. Startup requires five unique existing fingertip frames in
thumb/index/middle/ring/pinky order; configuration drift fails before the first
retargeting frame instead of shortening or shifting the optimizer target list.

**Deps:** `mplib`, `pinocchio`, `h5py`, RealSense SDK, XArm7/XHand SDKs, `numpy`, `pyav`.

---

## Changelog (key milestones)

All items below are resolved — see git log for full history.

- **P0**: Safety state machine (DISARMED/ARMED/RUNNING/FAULT) + capability heartbeats + Main supervisor
- **P0**: `SeqlockRingBuffer.get_last_k(k)` multi-frame read + `read_arm_state_k`/`read_hand_state_k`
- **P1**: Code dedup — `hand_home_converge()`, `_seed_hand_retargeter()`, shared ring helpers
- **P1**: Episode quality toolkit — held-frame filter, tracking-error filter, health/validate/assess
- **P2**: camera_loop extracted to `sensor/camera_process.py`
- **P3-P7**: All entry points migrated to SharedStorage architecture; old entry points deleted
- **Config**: immutable `ResolvedRuntimeConfig`; `CLI > JSON > defaults`, canonical JSON + SHA-256
- **Teleop**: Directory flattened (control/vr/core removed); dead code deleted (~690 lines)
- **Robot**: `xarm7/` subpackage deleted; inner_loop→arm_loop rename; hand legacy classes removed
- **Safety protocol**: mandatory geometry-aware ActionSafetyGate + correlated prepare/commit/ACK + epoch/TTL rejection
- **Ultracode review**: 23 fixes (arm_loop FAULT escalation, NaN guards, camera metadata race, etc.)
- **Code review rounds 1-2**: NameError fixes, CLI parameter plumbing, resource leak fixes, import cleanup
- **Replay**: hash-bound preflight certificate and dry-run default; live replay requires v15 validity and the `sent` stream
