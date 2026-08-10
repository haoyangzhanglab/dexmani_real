# AGENTS.md — DexMani Real

This file defines the repository-wide working agreement for coding agents. It
applies to every file below the repository root. If a deeper `AGENTS.md` is
added later, its instructions take precedence for that subtree.

## Project purpose

DexMani Real is a Python 3.10 robotics system for VR teleoperation and data
collection with an xArm7 (7 DoF), an XHand (12 DoF), a Quest headset, and an
Intel RealSense L515. The canonical runtime is a five-process architecture:

```text
camera ───────────────┐
VR ──────────────────┤
                     v
policy ──arm queue──> arm
   │
   └──hand ring─────> hand

arm/hand/camera/VR ──shared-memory state──> policy
policy ──aligned samples──> HDF5 episode (schema v15)
```

The thin main process only creates shared storage, starts workers, supervises
heartbeats and process health, and shuts workers down. Business logic belongs
in the worker or domain module, not in the entry point.

## Source map

- `examples/real/vr_teleop_hand_record.py`: canonical five-process entry point.
- `examples/real/keyboard_teleop_real.py`: arm-oriented keyboard teleoperation.
- `examples/real/calibrate_camera.py`: ArUco eye-to-hand camera calibration.
- `examples/real/calibrate_vr_heading.py`: VR-to-robot heading calibration.
- `examples/real/replay_traj.py`: recorded-episode replay and consistency checks.
- `dexmani_real/config/defaults.py`: source of truth for numeric defaults.
- `dexmani_real/ipc/schema.py`: dependency-neutral source of truth for every
  cross-process NumPy dtype and fixed-size protocol payload.
- `dexmani_real/shm/shared_storage.py`: cross-process data plane, ring/queue allocation,
  flags, readiness events, and supervisor helpers.
- `dexmani_real/shm/ring_buffer.py`, `robot_ring.py`: seqlock ring primitives.
- `dexmani_real/policy/vr_teleop_policy.py`: VR mapping, IK, command production,
  safety gating, and ownership of recording.
- `dexmani_real/robot/arm_loop.py`: xArm Mode 6 servo loop and arm state producer.
- `dexmani_real/robot/hand_process.py`: XHand servo loop and hand state producer.
- `dexmani_real/robot/safety.py`: `DISARMED/ARMED/RUNNING/FAULT` transitions.
- `dexmani_real/planning/`: FK, IK, collision checking, pose and path utilities.
- `dexmani_real/recording/`: timestamp alignment and HDF5 schema v15 I/O.
- `dexmani_real/sensor/`: RealSense, point-cloud, and VR receiver processes.
- `dexmani_real/teleop/`: arm mapping, hand retargeting, keyboard and audio UX.
- `dexmani_real/tools/`: episode quality analysis and visualization CLIs.
- `assets/`: URDF/SRDF meshes, retargeting configuration, and audio prompts.
- `CLAUDE.md`: detailed architecture notes and operational background. Keep it
  aligned when an architectural change makes its statements stale.

## Environment and commands

Run commands from the repository root. The expected environment is conda
`real_robot`, Python 3.10, with the repository on `PYTHONPATH`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate real_robot
export PYTHONPATH=.
```

Packaging metadata is intentionally minimal; `setup.py` does not declare the
hardware and planning dependencies. Do not assume `pip install -e .` creates a
working robot environment. Important external dependencies include NumPy,
SciPy, h5py, PyAV, mplib, Pinocchio, nlopt, pyrealsense2, OpenCV, Open3D,
PyTorch, rerun, dex-retargeting, and the vendor xArm/XHand SDKs.

Useful non-hardware validation commands:

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples/real
conda run -n real_robot mypy dexmani_real/
conda run -n real_robot black --check dexmani_real examples/real
conda run -n real_robot isort --check-only dexmani_real examples/real
```

Formatting is Black-compatible with a 120-character line length and isort's
Black profile, as configured in `pyproject.toml`. Prefer a focused check of the
changed modules when optional SDK imports prevent a repository-wide check.

There is currently no conventional unit-test suite. Files named
`examples/real/test_*.py` are interactive hardware diagnostics, not safe
pytest tests. Never run them merely because their names begin with `test_`.

## Hardware safety boundary

Treat every script under `examples/real/` as potentially hardware-affecting.
Do not run teleoperation, replay, calibration, RealSense, or XHand examples
without explicit user authorization and confirmation that the workspace is
clear and the hardware is ready. In particular, do not execute:

- the canonical or keyboard teleoperation entry points;
- trajectory replay or homing routines;
- camera/VR calibration routines that write calibration JSON;
- direct vendor SDK examples or commands that connect to the robot;
- ad-hoc imports when module import itself may initialize a device SDK.

Static inspection, compilation, formatting checks, type checking, and focused
tests with mocked hardware are acceptable. Do not weaken a safety check merely
to make an offline test pass.

## Architectural invariants

Preserve these unless the user explicitly requests an architectural redesign:

1. All inter-process payloads travel through `SharedStorage`; processes do not
   call one another or share live SDK objects.
2. Only arm and hand worker processes import/use their respective vendor SDKs.
   Keep SDK imports lazy where needed so the main process can import offline.
3. Policy owns the episode/grid/sample decisions and one coordinator clock
   domain for state, action, VR, and camera alignment. `RecorderIO` owns only
   serialization, verification, and transactional publication.
4. Policy recording is grid-aligned at `1 / control_hz` (normally 16 Hz). Do
   not replace this with arrival-time sampling.
5. The arm action queue is ordered and bounded (`maxsize=2`); its backpressure
   is intentional. The hand command ring is latest-wins by design.
6. xArm Mode 6 firmware performs trajectory smoothing. Do not add arm-side
   interpolation; double interpolation can produce overshoot and C24 faults.
7. Shared control/state rings use seqlocks. `get_last_k(k)` returns verified
   frames oldest-first, may return fewer than `k`, and rejects `k > maxlen`.
8. `is_running`, `is_recording`, `error_state`, and `estop_request` remain
   simple shared flags. `safety_state` alone is the `SafetyState` enum.
9. `error_state` is a sticky latch. Heartbeat timestamps use
   `time.monotonic()`, not wall-clock time.
10. Main owns `DISARMED <-> ARMED`, transitions to `FAULT`, and shutdown;
    policy owns `ARMED <-> RUNNING`; arm and hand only gate behavior on state.
11. The firmware is the final safety backstop. Application checks protect
    command validity, recovery behavior, data quality, and coordinated stop.
12. Cross-process structures are NumPy dtypes/scalars or other explicitly
    structured values—not arbitrary mutable Python object graphs.

## Cross-module change checklist

Changes to shared formats have a wider blast radius than their defining file:

- Arm/hand state field: update the dtype in `ipc/schema.py`, documentation
  dataclass in `robot/types.py`, producer write, policy read, and recording path.
- Ring or queue: update `SharedStorage` creation/cleanup, producer, consumer,
  readiness/heartbeat handling where applicable, and architecture docs.
- Recording dataset/schema: update recorder, reader, quality tool,
  visualization/replay consumers, schema marker, and backward compatibility.
- IK or collision behavior: inspect both planning code and policy fallback,
  hold-on-failure, delta-clamp, and frame-quality flags.
- Rate/default: change `config/defaults.py` first, then audit all derived
  durations, buffer sizes, heartbeat thresholds, recorder metadata, and CLIs.
- Robot state transition or fault behavior: audit main supervisor, policy,
  arm loop, hand loop, shutdown, and e-stop paths together.
- New entry point: follow the thin-main pattern: create `SharedStorage`, spawn
  plain `*_loop(shared, config)` functions, await readiness, supervise, shut down.

Do not silently change HDF5 meanings in place. Add fields compatibly and keep
old episodes readable. Readers should tolerate older optional datasets.

## Coding conventions

- Use Python 3.10+ syntax and `from __future__ import annotations` in new modules.
- Use dataclasses for configuration/state and NumPy arrays for numeric payloads.
- Use `snake_case`, `PascalCase`, and `UPPER_SNAKE_CASE` conventionally.
- Put `logger = get_logger(__name__)` after imports and before definitions.
- Log caught operational exceptions with context and `exc_info=True`; do not
  silently swallow faults in control, hardware, IPC, or recording paths.
- Use `field(default_factory=...)` for mutable dataclass values.
- Include units in names (`_rad`, `_deg`, `_m`, `_s`, `_hz`) and preserve the
  repository convention that unsuffixed angles are radians.
- Validate array shapes and finite values at process/module boundaries. Avoid
  in-place mutation of state/action arrays after publication.
- Keep control loops free of blocking file, camera, network, and UI operations.
- Prefer small pure helpers for math and transformation logic; they are easier
  to validate offline than process loops.
- Avoid broad refactors while fixing a focused issue, especially in safety and
  timing code. Preserve unrelated user changes in a dirty worktree.

## Validation strategy

Choose checks according to risk and report exactly what was and was not run:

1. For documentation/config-only edits, inspect the diff and verify referenced
   paths and commands.
2. For pure Python helpers, add or run focused deterministic tests without SDKs.
3. For dtype, ring, timing, or recording changes, test round trips, shapes,
   units, boundary values, old-format reads, and cleanup of shared memory/files.
4. For process changes, validate startup failure, normal shutdown, worker death,
   heartbeat timeout, sticky fault, and e-stop paths with mocks where possible.
5. Hardware validation is a separate, explicitly authorized step. Provide a
   concise manual checklist rather than claiming hardware behavior from static
   or mocked tests.

Before finishing, inspect `git diff` and `git status --short`. Do not overwrite,
revert, format, or include unrelated pre-existing modifications.

## Operational notes

- xArm default address is `192.168.1.111`; do not probe it without permission.
- Quest HTS traffic uses TCP port 8000; USB operation commonly needs
  `adb reverse tcp:8000 tcp:8000`.
- The L515 should be directly attached over USB 3.0. A stalled stream can look
  healthy if frames are forward-filled, so freshness must be measured at source.
- Calibration files under `dexmani_real/config/*.json` are operational data.
  Preserve coordinate-frame conventions and do not regenerate them casually.
- Generated recordings, logs, videos, and temporary episode directories can be
  large or safety-relevant. Do not delete or rewrite them without explicit scope.
