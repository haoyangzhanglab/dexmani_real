# AGENTS.md — DexMani Real Working Contract

This is the repository-wide contract for coding agents. It applies below the
repository root unless a deeper `AGENTS.md` overrides it. Read this file before
editing; use [README.md](README.md) as the file-by-file project map and
`CLAUDE.md` as the implementation navigation guide.

## 1. Fast start

Work from the repository root. Before editing, inspect the current worktree and
the smallest relevant call path; do not assume a clean tree or reformat unrelated
files.

```bash
git status --short
rg -n "<symbol-or-config-key>" dexmani_real examples
conda run -n real_robot python -m compileall -q dexmani_real examples
```

- Target environment: conda `real_robot`, Python 3.10, `PYTHONPATH=.`.
- `pyproject.toml` is the only packaging configuration. Native planning, CUDA,
  device-SDK, and robot dependencies are managed by the conda environment.
- This repository has no conventional unit-test suite. Treat any `test_*.py`
  under `examples/` as an interactive hardware program, never as a test.
- Prefer a small deterministic offline check that exercises the changed pure
  helper, dtype, reader, or lifecycle branch.

## 2. System model

DexMani Real is a VR teleoperation, data collection, and replay system for an xArm7
(7 DoF), XHand (12 DoF), Quest, and RealSense L515.

```text
camera / VR / arm / hand ──shared-memory state──► teleop
                                                     │
                       arm queue ◄──────────────────┼──► hand command ring
                                                     │
                                         aligned sample ring
                                                     ▼
                                            RecorderIO ──► HDF5 episode v16
```

The main/lifecycle process resolves immutable configuration, creates
`SharedStorage`, performs bounded read-only readiness checks, supervises worker
health, and shuts down. It does not map VR poses, publish actions, or choose
recording samples.

### Task router

Start with the source of truth in this table, then follow only its direct
producers and consumers.

| If the task changes… | Start here | Then audit |
|---|---|---|
| Numeric default or runtime override | `config/defaults.py`, `config/runtime.py` | Derived rates, buffer capacities, timeouts, metadata, CLI overrides |
| Cross-process state/action layout | `utils/schema.py` | `robot/types.py`, `SharedStorage`, producer, consumer, recording reader/writer |
| Ring, queue, flag, event, or metric | `shm/shared_storage.py` | Allocation/cleanup, all writers/readers, readiness, heartbeat, shutdown |
| VR teleoperation behavior | `teleop/loop.py` | mapper, snapshot, hand control, IK fallback, action protocol, recording samples |
| Arm/hand safety or servo behavior | `robot/arm_loop.py`, `robot/hand_process.py` | `robot/safety.py`, action protocol, supervisor, homing and e-stop paths |
| FK, IK, collision, or a joint path | `planning/` | teleop hold/fallback/delta clamp and replay dense preflight |
| Episode schema or quality rule | `recording/` | reader, analysis, visualization, replay consumers, v16 schema contract |
| Replay behavior | `examples/replay_episode.py` | provenance/dense preflight, session/runner, metrics, live safety path |
| Episode visualization | `examples/visualize_episode.py` | Rerun integration, EpisodeReader, point cloud, time-series views |
| Calibration | `examples/calibrate_camera.py`, `examples/calibrate_vr_heading.py` | explicit write/confirmation path and calibration JSON contract |
| CLI surface | `examples/` | Keep the wrapper thin; put lifecycle and behavior in a domain module |

## 3. Non-negotiable architecture

Preserve these unless the user explicitly requests an architectural redesign.

1. `SharedStorage` is the only cross-process data plane. Processes do not call
   each other or exchange live SDK objects/mutable object graphs.
2. `utils/schema.py` owns fixed-shape NumPy payload definitions. Cross-process
   values must stay structured and shape/finite-validated at their boundary.
3. Hardware SDK instances are local to their owning device worker/driver. Do
   not add xArm/XHand SDK calls to main, policy, recorder, or replay code; never
   share a live SDK instance across processes.
4. Teleoperation owns control-grid, action, and episode/sample decisions.
   `RecorderIO` only serializes, verifies, and transactionally publishes what
   it receives.
5. Recording is grid-aligned to `1 / control_hz` (normally 16 Hz), never
   arrival-time sampled.
6. The arm queue is ordered and intentionally bounded (`maxsize=2`). The hand
   command ring is intentionally latest-wins.
7. Shared control/state rings use seqlocks. `get_last_k(k)` is verified,
   oldest-first, may return fewer than `k`, and rejects `k > maxlen`.
8. `is_running`, `is_recording`, `error_state`, and `estop_request` are simple
   shared flags. Only `safety_state` stores the `SafetyState` enum.
9. `error_state` is sticky; heartbeats use `time.monotonic()`, never wall time.
10. Main owns `DISARMED ↔ ARMED`, transition to `FAULT`, and shutdown. Policy
    owns `ARMED ↔ RUNNING`; arm/hand workers only gate behavior on state.
11. xArm Mode 6 firmware performs arm trajectory smoothing. Do not add
    application-side arm interpolation; it can cause overshoot and C24 faults.
12. Firmware is the final safety backstop. Application checks protect command
    validity, recovery, data quality, and coordinated stop—they do not replace
    firmware limits.
13. `run_generation` invalidates stale queued/ring commands across begin, pause,
    home, feedback fault, and camera re-warm. Workers reject a command from an
    older generation before it crosses the command boundary.
14. Recorder START/STOP boundaries, recorder status, and aligned samples are
    fixed NumPy dtypes. Do not put JSON or an acknowledgement/apply protocol in
    the shared-memory control path.

## 4. Hardware and operational safety

Every program below `examples/` can affect hardware. Do **not** run
teleoperation, replay, homing, calibration, or RealSense
without explicit user authorization and confirmation that the
workspace is clear and the hardware is ready. Do not use a module import as a
shortcut when it might initialize a device SDK.

Allowed without that authorization: static inspection, compilation, and focused
offline tests with fakes/mocks. Do not weaken safety checks merely to make an
offline check pass.

Operational data is not disposable test data:

- `dexmani_real/config/*.json` carries calibration/frame conventions.
- Recordings, logs, videos, and temporary episode directories may be large and
  safety-relevant. Do not delete or rewrite them without an explicit target.
- The xArm default address is `192.168.1.111`; never probe it without approval.
- Quest HTS uses TCP 8000; the L515 should use a direct USB 3 connection.

## 5. Change playbooks

Use the smallest vertical slice that fully preserves a contract.

| Change | Required impact check |
|---|---|
| Add/change arm or hand state | dtype → documentation dataclass → worker write → policy read → recording path → reader/analysis if persisted |
| Add/change a recording dataset | schema dtype → recorder → reader → analysis/visualization → replay consumer → v16 schema contract |
| Add/change a ring or queue | `SharedStorage` create/close → producer → consumer → readiness/heartbeat → failure/shutdown behavior |
| Change IK/collision logic | planner + candidate/fallback behavior + hold-on-failure + delta clamp + frame-quality flags + replay preflight |
| Change a rate/default | `config/defaults.py` first → all derived durations/capacities/timeouts → metadata and CLI help |
| Change safety/fault transition | supervisor + policy + arm + hand + shutdown + e-stop, including sticky-fault behavior |
| Add an entry point | Thin `examples/` forwarding CLI → domain lifecycle that owns storage, spawn, readiness, supervision, and shutdown |

Do not silently change HDF5 meaning in place. Runtime episodes use schema v16
only: coordinate a format change across writer, reader, visualization, replay, and the
schema marker. Migrate historical episodes outside the runtime before
using them.

## 6. Implementation rules

- Use Python 3.10+ syntax and `from __future__ import annotations` in new
  modules. Use dataclasses for configuration/state and NumPy arrays for numeric
  payloads.
- Keep units in names (`_rad`, `_deg`, `_m`, `_s`, `_hz`); unsuffixed angles are
  radians. Validate shapes and finite values at module/process boundaries.
- Put `logger = get_logger(__name__)` after imports. Log caught operational
  exceptions with context and `exc_info=True`; never silently swallow a control,
  hardware, IPC, or recording fault.
- Use `field(default_factory=...)` for mutable dataclass fields. Do not mutate a
  published state/action array in place.
- Keep control loops free of blocking file, camera, network, or UI work. Put
  pure math and transforms in small helpers that can be tested offline.
- Avoid broad refactors while addressing a focused issue. Preserve unrelated
  changes in a dirty worktree.

## 7. Finish definition

Before handing off:

1. Inspect the focused diff and `git status --short`; do not overwrite or
   include unrelated modifications.
2. Verify changed paths/commands and run the least risky relevant offline check.
3. Run `conda run -n real_robot python -m compileall -q dexmani_real examples`
   for Python-source changes unless a narrower check is more appropriate.
4. For safety/process changes, cover startup failure, normal shutdown, worker
   death, heartbeat timeout, sticky fault, and e-stop with fakes/mocks where
   feasible.
5. Report exactly what was and was not run. Hardware validation is a separate,
   explicitly authorized manual step.

Update `README.md` when the file map or user-facing architecture changes, and
update `CLAUDE.md` when implementation routing, ownership, or a key invariant
changes.
