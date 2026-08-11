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
| Change a shared field or command | `ipc/schema.py` | `shm/shared_storage.py`, producer, consumer, recorder/reader |
| Change VR behavior | `teleop/loop.py` | snapshot → mapper/retargeting → planner → action protocol → samples |
| Change learned deployment | `policy/deployment.py`, `policy/spec.py` | inference process → coordinator → action protocol |
| Change an arm/hand action | `policy/action_protocol.py` | arm/hand worker ACKs, `robot/safety.py`, supervisor |
| Change FK/IK/collision | `planning/` | teleop fallback/hold, replay preflight, safety gate |
| Change episode I/O | `recording/io_process.py` | recorder → reader → analysis → replay |
| Change replay | `replay/episode.py` | preflight → session → runner → metrics |
| Change calibration | `examples/calibrate_camera.py`, `examples/calibrate_vr_heading.py` | explicit confirmation/write paths and JSON compatibility |

## 2. Ownership map

The system is spawn-only: workers receive `SharedStorage` plus an optional
config, never a live SDK object or another worker object.

```text
Main / domain lifecycle
  resolve immutable config → create SharedStorage → spawn → readiness → supervise → verified shutdown

Camera worker ─┐
VR worker ─────┼── shared state rings ──► Teleop or learned-policy coordinator
Arm worker ────┤                                  │
Hand worker ───┘                         prepare/commit action protocol
                                                    ├──► bounded arm queue → Arm worker
                                                    └──► latest-wins hand ring → Hand worker

Teleop / coordinator ── aligned sample ring ──► RecorderIO ──► HDF5 episode v15
```

| Owner | Owns | Must not own |
|---|---|---|
| Main / lifecycle module | Config snapshot, storage, process creation, read-only readiness, health, shutdown | Mapping, actions, sample selection |
| `teleop/loop.py` | VR mapping, IK, action candidates, safety gating, recording/grid decisions | HDF5 serialization or direct SDK use |
| `policy/learned_coordinator.py` | Causal observations, inference result scheduling, policy epoch/action IDs | Direct SDK use or recording I/O |
| `recording/io_process.py` | Record consumption, HDF5/video write, verification, fsync, atomic publish | Choosing what or when to sample |
| `robot/arm_loop.py` / `robot/hand_process.py` | Vendor-device I/O, measured feedback, command application, ACKs | Policy decisions or cross-worker calls |
| `sensor/` workers | Device acquisition and source-freshness metadata | Motion/control decisions |
| `runtime/` | Readiness, heartbeat/process supervision, verified teardown | Domain mapping or command production |

### Canonical process targets

```python
arm_loop(shared, config)          # xArm Mode 6 servo and FK state
hand_loop(shared, config)         # XHand servo, state/tactile and ACKs
camera_loop(shared, config)       # RealSense frames → camera_ring
vr_loop(shared, config)           # Quest/HTS frames → vr_ring
teleop_loop(shared, config)       # VR → candidate → safe arm/hand commands
recorder_io_loop(shared, config)  # aligned samples → transactional episode
```

Learned deployment adds `inference_loop` in an isolated child. The ordinary VR
path does not load a model.

## 3. Data contracts that matter

### Shared storage and IPC

`ipc/schema.py` is the authority for cross-process NumPy dtypes and fixed
shapes. `SharedStorage` owns allocation, names, close/unlink, flags, events,
and transport instances; it must not import policy or recorder code merely to
discover a dtype.

| Transport | Direction | Semantics |
|---|---|---|
| `arm_action_q` | controller → arm | Ordered `mp.Queue(maxsize=2)`; backpressure is intentional |
| `hand_cmd_ring` | controller → hand | Seqlock, latest-wins servo target |
| commit + ACK rings | controller ⇄ workers | Correlated prepare/commit/apply/reject protocol |
| arm/hand/VR/camera rings | worker → controller | Seqlock state; source and publish freshness are distinct |
| `record_sample_ring` | controller → RecorderIO | Fixed-grid aligned sample; overflow aborts the episode |
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
- `ActionSafetyGate` validates shape/finite values, freshness, epoch/TTL,
  limits, delta, workspace, table/collision constraints before raw command IPC.
- Pause, stale VR, or a camera generation reset invalidates the active policy
  epoch, publishes a coordinated hold, and requires fresh feedback/re-anchor.

Do not turn a simple flag into an enum, add a second state writer, or make a
recovery path bypass prepare/commit/ACK correlation.

## 4. Critical behavior paths

### VR teleoperation and recording

```text
VR frame → causal snapshot → ArmWristMapper / hand retargeter → IK candidate
         → ActionSafetyGate → prepare + ACK → commit + apply ACK
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

### Experimental learned-policy deployment

```text
PolicySpec YAML + resource hashes → deployment preflight → isolated inference
                                  → causal coordinator → ActionSafetyGate → workers
```

`PolicySpec` binds adapter module, explicit observation history, action contract,
resources and SHA-256s. `action.dt_s` must equal `1 / control_hz`; a live run
also requires `hardware_deployable: true`. Keep model import in the inference
child and retain backend-created action target/expiry times—do not retime stale
chunks into validity.

The default teleoperation `SharedStorage` does not allocate inference rings.
Only `policy/deployment.py` may opt into that experimental capability.

### Episode write, read, analyse, replay

```text
aligned samples → RecorderIO → temporary episode + stream verification
                → fsync + atomic publish → EpisodeReader / analysis / replay
```

- HDF5 schema v15 is additive. Readers keep v12–v14 raw-readable but treat
  semantic validity conservatively; never silently repurpose an old dataset.
- Writer failure, stream mismatch, overflow, codec failure, or ENOSPC aborts
  the episode rather than silently publishing partial data.
- `recording/analysis/episode_quality.py` and
  `recording/analysis/visualize_episode.py` are
  offline consumers under `recording/analysis/`.
- `replay/episode.py` defaults to dry-run. Live replay reruns fail-closed
  provenance and dense geometry checks immediately before worker startup.

## 5. Edit recipes

### Adding a shared state field

1. Define shape/dtype in `ipc/schema.py`; add the corresponding explanatory
   dataclass field in `robot/types.py`.
2. Allocate it through `SharedStorage`; write it in the owning worker and read
   it in each consumer.
3. Decide whether recording persists it. If yes, update recorder, reader,
   quality/visualization/replay consumers and preserve old episodes.
4. Check finite values, units, initial/invalid values, and process cleanup.

### Changing control, IK, or collision behavior

1. Change the pure planning/mapper helper first, with explicit unit/shape
   validation.
2. Trace candidate rejection to hold-on-failure, delta clamp, frame-quality
   flags, safety gate, and replay preflight.
3. Keep arm behavior as one Mode 6 endpoint per grid tick; never insert an
   interpolation layer.
4. Exercise invalid target, stale feedback, collision/rejection, and normal
   hold recovery with fakes before any hardware validation.

### Changing recording or replay

1. Keep v15 dataset meanings stable; add optional fields compatibly.
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
| Check measured hand feedback and ACK identity | Assuming hand connection or command application |
| Keep new logic in the domain owner | Business logic in a thin main/CLI wrapper |

### Hardware facts that change engineering decisions

- xArm Mode 6 is the normal servo mode; firmware is the final collision/current
  safety backstop. C22/C31 are immediate faults; C24 has bounded measured-hold
  recovery. Homing uses a separately validated Mode 0 milestone path.
- XHand is a 12-DoF EtherCAT position servo; its command ring is latest-wins.
  The default keyboard/data-collection path requires measured hand feedback;
  `--no-hand` is an explicit secured/open-pose assumption.
- A stalled L515 can appear healthy when frames are forward-filled. Source
  freshness, not just worker heartbeat, determines frame quality.
- Quest HTS is TCP port 8000; USB commonly needs `adb reverse tcp:8000 tcp:8000`.
- Calibration JSON carries physical coordinate-frame conventions. Do not
  regenerate or delete it casually.

## 7. Current command surface

| Entry point | Domain owner | Default safety posture |
|---|---|---|
| `examples/collect_teleop.py` | `teleop/experiment.py` | Hardware control; explicit authorization required |
| `examples/deploy_policy.py` | `policy/deployment.py` | Hardware control; spec/hash/preflight gated |
| `examples/keyboard_teleop_real.py` | `teleop/keyboard_experiment.py` | Hardware control; measured hand feedback by default |
| `examples/replay_episode.py` | `replay/episode.py` | Dry-run by default; `--live` reruns dense preflight |
| `examples/calibrate_camera.py` | — | Hardware/data-writing operation; self-contained ArUco hand-eye calibration |
| `examples/calibrate_vr_heading.py` | — | Hardware read; transform write is gated; self-contained VR heading calibration |
| `examples/realsense_record_example.py` | — | Interactive RealSense RGB-D + point-cloud test; hardware read-only by default |
| `examples/pointcloud_process_example.py` | `sensor/pointcloud_processor.py` | Production point-cloud pipeline diagnostic; explicit confirmation for desk-plane write |
| `examples/xhand_control_example.py` | — | Standalone XHand SDK diagnostic; requires explicit hardware authorization for motion commands |

## 8. Completion checklist

1. Review the focused diff and `git status --short`; preserve unrelated edits.
2. Verify every referenced path and configuration key with `rg`.
3. Run the smallest safe offline validation, plus compilation for Python-source
   changes when appropriate.
4. State what was not run—especially hardware validation—and give a manual
   checklist instead of claiming device behavior from static analysis.
