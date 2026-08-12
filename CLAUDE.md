# CLAUDE.md — DexMani Real Implementation Guide

`AGENTS.md` is the binding working contract. This document is the fast path for
understanding a behavior change: choose the correct owner, trace the control or
data path, and make the smallest safe edit. [README.md](README.md) is the
complete file-by-file map.

## 1. Sixty-second orientation

DexMani Real controls an xArm7 (7 DoF) and XHand (12 DoF) from Quest VR,
records HDF5 episodes, can deploy an experimental learned policy, and can replay an
episode. Python 3.10 runs in conda `real_robot`; commands run from the repo root
with `PYTHONPATH=.`.

```bash
git status --short
rg -n "<symbol-or-config-key>" dexmani_real examples
conda run -n real_robot python -m compileall -q dexmani_real examples
```

Do not run `examples/` without explicit hardware authorization. They are
entry points, not test fixtures.

### Task-to-code router

| Goal | Start with | Follow next |
|---|---|---|
| Change a runtime value | `config/defaults.py` | `config/runtime.py`, every derived duration/capacity/metadata field |
| Change a shared field or command | `utils/schema.py` | `shm/shared_storage.py`, producer, consumer, recorder/reader |
| Change VR behavior | `teleop/loop.py` | snapshot → mapper/retargeting → planner → safety gate → samples |
| Change learned deployment | `examples/deploy_policy.py`, `policy/spec.py` | inference process → coordinator → safety gate |
| Change an arm/hand action | `policy/safety.py` | gate validation → send_command → worker apply, `robot/safety.py`, supervisor |
| Change FK/IK/collision | `planning/` | teleop fallback/hold, replay preflight, safety gate |
| Change episode I/O | `recording/io_process.py` | recorder → reader → analysis → replay |
| Change replay | `examples/replay_episode.py` | — | Self-contained script; preflight → session → runner → metrics |
| Change calibration | `examples/calibrate_camera.py`, `examples/calibrate_vr_heading.py` | explicit confirmation/write paths and calibration JSON contract |

## 2. Ownership map

The system is spawn-only: workers receive `SharedStorage` plus an optional
config, never a live SDK object or another worker object.

```text
Main / domain lifecycle
  resolve immutable config → create SharedStorage → spawn → readiness → supervise → verified shutdown

Camera worker ─┐
VR worker ─────┼── shared state rings ──► Teleop or learned-policy coordinator
Arm worker ────┤                                  │
Hand worker ───┘                         fire-and-forget command publication
                                                    ├──► bounded arm endpoint/HOME queue → Arm worker
                                                    ├──► priority arm STOP/RESUME ring ───┘
                                                    └──► latest-wins hand ring → Hand worker

Teleop / coordinator ── aligned sample ring ──► RecorderIO ──► HDF5 episode v16
```

| Owner | Owns | Must not own |
|---|---|---|
| Main / lifecycle module | Config snapshot, storage, process creation, read-only readiness, health, shutdown | Mapping, actions, sample selection |
| `teleop/loop.py` | VR mapping, IK, action candidates, safety gating, recording/grid decisions | HDF5 serialization or direct SDK use |
| `policy/learned_coordinator.py` | Causal observations, current-tick inference result publication, run generation/action IDs | Direct SDK use or recording I/O |
| `recording/io_process.py` | Record consumption, HDF5/video write, verification, fsync, atomic publish | Choosing what or when to sample |
| `robot/arm_loop.py` / `robot/hand_process.py` | Vendor-device I/O, measured feedback, command application | Policy decisions or cross-worker calls |
| `sensor/` workers | Device acquisition and source-freshness metadata | Motion/control decisions |
| `runtime/` | Readiness, heartbeat/process supervision, verified teardown | Domain mapping or command production |

### Canonical process targets

```python
arm_loop(shared, config)          # xArm Mode 6 servo and FK state
hand_loop(shared, config)         # XHand servo, state/tactile feedback
camera_loop(shared, config)       # RealSense frames → camera_ring
vr_loop(shared, config)           # Quest/HTS frames → vr_ring
teleop_loop(shared, config)       # VR → candidate → safe arm/hand commands
recorder_io_loop(shared, config)  # aligned samples → transactional episode
```

Learned deployment adds `inference_loop` in an isolated child. The ordinary VR
path does not load a model.

## 3. Data contracts that matter

### Shared storage and IPC

`utils/schema.py` is the authority for cross-process NumPy dtypes and fixed
shapes. `SharedStorage` owns allocation, names, close/unlink, flags, events,
and transport instances; it must not import policy or recorder code merely to
discover a dtype.

| Transport | Direction | Semantics |
|---|---|---|
| `arm_action_q` | controller → arm | Ordered `mp.Queue(maxsize=2)`; carries fixed endpoints plus correlated HOME requests; endpoint backpressure is intentional |
| `arm_control_ring` | controller → arm | Fixed-dtype latest-wins STOP/RESUME request; read before the ordered queue so braking never competes with endpoints for capacity |
| `hand_cmd_ring` | controller → hand | Seqlock, latest-wins servo target |
| arm/hand/VR/camera rings | worker → controller | Seqlock state; source and publish freshness are distinct |
| `record_control_ring` | controller → RecorderIO | Latest immutable fixed-field START/STOP boundary; no JSON payload |
| `record_sample_ring` | controller → RecorderIO | Fixed-grid aligned sample; overflow aborts the episode |
| `record_status_ring` | RecorderIO → controller/main | Latest lifecycle outcome and bounded error/path text; no progress protocol |
| `inference_candidate_ring` | inference → learned coordinator | Latest current-tick candidate; only allocated for policy deployment |
| flags and heartbeats | lifecycle/workers | Flags are simple values; heartbeats use `time.monotonic()` |

Read a ring with its documented seqlock API. `get_last_k(k)` returns verified
frames oldest-first, may be shorter than `k`, and raises for `k > maxlen`.
Never store arbitrary mutable Python graphs in shared memory.

### Safety state and command lifetime

```text
DISARMED -- Main readiness --> ARMED -- policy/teleop operator action --> RUNNING
    ^                               ^                                  |
    └------- Main shutdown ---------┴-------------- Main fault ---------┘
                                                     ▼
                                                   FAULT
```

- Main owns `DISARMED ↔ ARMED`, `→ FAULT`, and shutdown.
- Teleop/policy owns `ARMED ↔ RUNNING`.
- Arm and hand workers only gate command behavior on the state.
- `error_state` is sticky. A process death or enabled-worker heartbeat timeout
  becomes a main-owned fault.
- `SafetyGate` (in `policy/safety.py`) is the single validation boundary:
  well-formed → joint limits → whole-action velocity-envelope accept/reject →
  collision → workspace. It never clips individual joints of a motion action.
  Hand velocity is command-to-command, never target-to-measured; workers
  additionally reject stale-generation, expired, operational-limit, and rated
  mechanical-limit violations without changing an endpoint. Runtime config may
  narrow, but cannot widen, the bundled rated mechanical envelope.
- `run_generation` tags both policy observations and candidates. Begin, pause,
  home, feedback fault, and camera re-warm advance it; the coordinator drops a
  mailbox result unless it, its observation, and shared state have the same
  generation. Those paths publish a coordinated hold when fresh feedback is
  available and require re-anchor/re-warm before motion resumes.

Do not turn a simple flag into an enum, add a second state writer, or bypass
the SafetyGate validation boundary.

## 4. Critical behavior paths

### VR teleoperation and recording

```text
VR frame → causal snapshot → ArmWristMapper / hand retargeter → IK candidate
         → SafetyGate.validate() → send_command() → workers apply immediately
         → grid-aligned state/action/VR/camera sample → RecorderIO
```

- `teleop/snapshot.py` creates a causal snapshot; do not mix arrivals from
  unrelated times.
- `teleop/arm_mapper.py` applies frame transforms and workspace/rotation bounds.
- `planning/ik.py` and `planning/ik_candidates.py` solve and filter; failure
  holds rather than inventing a new command.
- `teleop/episode_samples.py` owns recording sample construction. One sample is
  emitted per `control_hz` grid tick (normally 16 Hz), not per sensor arrival.
- Mode 6 firmware smooths arm targets. Application-side interpolation is unsafe.
- A moving keyboard arm stops through a correlated firmware State 6 request,
  not a delayed measured/predicted endpoint. State 6 is the request; deployed
  firmware may report non-ready State 5 (or briefly State 6). The worker
  accepts only that error-free post-request state set and ACKs through
  `last_cmd_seq`; the caller independently waits for two fresh low-velocity
  frames before re-anchoring to measured qpos. Before the next motion endpoint,
  the controller sends an explicit RESUME request through the same priority
  ring. The worker reads joints directly from the SDK, establishes that exact
  measured hold while entering Mode 6/ready State 2, then ACKs; an ordinary
  endpoint can neither resume the arm nor fill the queue during this handshake.

### Experimental learned-policy deployment

```text
PolicySpec YAML + resource hashes → deployment preflight → isolated inference
                                  → causal coordinator → SafetyGate → workers
```

`PolicySpec` binds adapter module, explicit observation history, action contract,
resources and SHA-256s. `action.dt_s` must equal `1 / control_hz`; a live run
also requires `hardware_deployable: true`. Keep model import in the inference
child. The adapter must return one current-tick `ActionCandidate`; adapters for
chunk-producing models choose or expand their next step locally. A single
`run_generation` invalidates candidates from a prior start, pause, home, or
camera re-warm, while the control side assigns action IDs and command targets.
`ActionCandidate.target_monotonic_ns` remains proposal/episode-provenance
timing and `valid_until_monotonic_ns` is the freshness guard; `send_command`
creates the actual worker delivery target after queueing begins.

The default teleoperation `SharedStorage` does not allocate inference rings.
Only `examples/deploy_policy.py` (via
`SharedStorageConfig.from_runtime(..., enable_inference=True)`) may opt into
that experimental capability.

### Episode write, read, analyse, replay

```text
aligned samples → RecorderIO → temporary episode + stream verification
                → fsync + atomic publish → EpisodeReader / visualize / replay
```

- HDF5 schema v16 is the only runtime episode format. Readers, visualization,
  and replay accept only a published v16 directory; migrate historical data
  outside the runtime before using it.
- Recorder control and the shared sample payload are fixed NumPy dtypes. The
  START boundary snapshots only task/operator and essential device/calibration
  metadata plus the required resolved-config SHA-256; per-grid fields are typed
  rather than JSON-encoded. The
  fire-and-forget worker protocol records no fabricated ACK or apply status.
- Writer failure, stream mismatch, overflow, codec failure, or ENOSPC aborts
  the episode rather than silently publishing partial data.
- `examples/visualize_episode.py` is an offline episode consumer (Rerun 3D visualization).
- `examples/replay_episode.py` defaults to dry-run. Live replay reruns fail-closed
  provenance and dense geometry checks immediately before worker startup.

## 5. Edit recipes

### Adding a shared state field

1. Define shape/dtype in `utils/schema.py`; add the corresponding explanatory
   dataclass field in `robot/types.py`.
2. Allocate it through `SharedStorage`; write it in the owning worker and read
   it in each consumer.
3. Decide whether recording persists it. If yes, update recorder, reader,
   quality/visualization/replay consumers and the v16 schema contract together.
4. Check finite values, units, initial/invalid values, and process cleanup.

### Changing control, IK, or collision behavior

1. Change the pure planning/mapper helper first, with explicit unit/shape
   validation.
2. Trace candidate rejection to hold-on-failure, upstream teleop delta shaping,
   whole-action gate rejection, frame-quality flags, and replay preflight.
3. Keep arm behavior as one Mode 6 endpoint per grid tick; never insert an
   interpolation layer.
4. Exercise invalid target, stale feedback, collision/rejection, and normal
   hold recovery with fakes before any hardware validation.

### Changing recording or replay

1. Keep v16 dataset meanings stable. Coordinate any format change across
   recorder, reader, visualization, and replay in the same change.
2. Update producer, reader, offline quality/visualization, replay loader, and
   provenance/schema marker together.
3. For replay, verify provenance, dense geometry and explicit hand mode before
   worker startup. Do not weaken a check to make dry-run or a fixture pass.

### Adding a worker or CLI

1. Put the worker target in its domain package as a plain `*_loop(shared, config)`
   function.
2. Extend the owning lifecycle module for storage, spawn, readiness, heartbeat,
   shutdown, and failure behavior.
3. Keep the `examples/` script as a one-import `main()` wrapper.
4. Update README’s project map and this guide when ownership/routing changes.

## 6. High-value conventions and footguns

| Do | Avoid |
|---|---|
| Validate finite shapes at module/process boundaries | Letting malformed numeric commands reach a worker |
| Use `field(default_factory=...)` and copy before publication | Mutable dataclass defaults or in-place mutation after publication |
| Put `logger = get_logger(__name__)` after imports; log operational exceptions with `exc_info=True` | `except: pass` or context-free warning logs |
| Use `control_hz` and derived config values | Hard-coded timing/buffer assumptions |
| Keep file/network/UI work out of control loops | Blocking I/O in servo or control loops |
| Use `TYPE_CHECKING`/lazy imports to break cycles and isolate SDKs | Circular imports or SDK objects crossing process boundaries |
| Check measured hand feedback and SDK return codes | Assuming hand connection or command application |
| Keep new logic in the domain owner | Business logic in a thin main/CLI wrapper |

### Hardware facts that change engineering decisions

- xArm Mode 6 is the normal servo mode; firmware is the final collision/current
  safety backstop. C22/C31 are immediate faults; C24 has bounded measured-hold
  recovery. Homing uses a separately validated Mode 0 milestone path.
- Successful vendor SDK startup chatter may be captured only within the owning
  device worker's bounded initialization calls. Project readiness logs remain
  visible, and failed SDK calls replay their captured native diagnostics.
- `config/desk_plane.json` is shared calibration input for point-cloud filtering,
  online collision checks, replay preflight, and homing. The collision model
  represents it as a tilted finite box, excludes only configured mounting-link
  contact, and uses mesh distance for soft clearance; fixed-Z frame padding is
  a compatibility fallback only.
- XHand is a 12-DoF EtherCAT position servo; its command ring is latest-wins.
  The default keyboard/data-collection path requires measured hand feedback;
  `--no-hand` is an explicit secured/open-pose assumption.
- XHand contact, backlash, and torque-limited steady-state error are valid
  execution outcomes. Unchanged qpos or failure to converge to a requested
  angle is diagnostic data, not a freshness fault; use forced-read success,
  source timestamps, worker heartbeat, board error registers, and SDK return
  codes for health. The v16 `qpos_stale` bit is retained as a reserved false
  compatibility field.
- Return-home creates explicit bounded milestones from the worker's last
  accepted hand command to the configured hand-home endpoint. Every complete
  endpoint is published unchanged and matched to a successful SDK action ID;
  no measured-angle convergence is tested. After the exact final-home ACK, arm
  homing uses configured hand-home geometry and the existing validated path.
- A stalled L515 can appear healthy when frames are forward-filled. Source
  freshness, not just worker heartbeat, determines frame quality.
- Quest HTS is TCP port 8000; USB commonly needs `adb reverse tcp:8000 tcp:8000`.
- Calibration JSON carries physical coordinate-frame conventions. Do not
  regenerate or delete it casually.

## 7. Current command surface

| Entry point | Domain owner | Default safety posture |
|---|---|---|
| `examples/collect_teleop.py` | — | Hardware control; self-contained script; explicit authorization required |
| `examples/deploy_policy.py` | — | Hardware control; self-contained script; spec/hash/preflight gated |
| `examples/keyboard_teleop.py` | — | Hardware control; self-contained script; measured hand feedback by default |
| `examples/replay_episode.py` | — | Dry-run by default; `--live` reruns dense preflight; self-contained script |
| `examples/calibrate_camera.py` | — | Hardware/data-writing operation; self-contained ArUco hand-eye calibration |
| `examples/calibrate_vr_heading.py` | — | Hardware read; transform write is gated; self-contained VR heading calibration |
| `examples/realsense_record_example.py` | — | Interactive RealSense RGB-D + point-cloud test; hardware read-only by default |
| `examples/pointcloud_process_example.py` | `sensor/pointcloud_processor.py` | Production point-cloud pipeline diagnostic; explicit confirmation for desk-plane write |
| `examples/xhand_control_example.py` | — | Standalone XHand SDK diagnostic; requires explicit hardware authorization for motion commands |
| `examples/visualize_episode.py` | — | Offline Rerun 3D visualization; no hardware control; self-contained script |

## 8. Completion checklist

1. Review the focused diff and `git status --short`; preserve unrelated edits.
2. Verify every referenced path and configuration key with `rg`.
3. Run the smallest safe offline validation, plus compilation for Python-source
   changes when appropriate.
4. State what was not run—especially hardware validation—and give a manual
   checklist instead of claiming device behavior from static analysis.
