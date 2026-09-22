# Codex Task — Tighten DexMani Real Runtime Semantics Without Growing the Runtime

## Status and execution rule

This task was re-reviewed against current `main` at:

`15cd48dc01bc00fae513d2ca5769de45a9c67183`

The implementation code beneath this task is still the behavior fact-checked in parent commit `46ba86710a62cda1233d522d62fc20627ad1dd4d`.

Before editing:

1. read `AGENTS.md`;
2. inspect `git status --short`;
3. inspect current definitions/usages rather than trusting line numbers in this document;
4. preserve unrelated user changes;
5. do not run hardware-affecting code.

If `main` moved, reconcile this task against the new code while preserving its intent. Do not mechanically restore mechanisms that newer code already removed.

This is one focused implementation pass. Do not stop after partial edits if the remaining work is offline and within scope.

---

# 1. Repository intent

`dexmani_real` is a personal PhD research repository for:

- real-robot dexterous data collection;
- VR/HTS and keyboard teleoperation;
- xArm7 + XHand control;
- RGB-D / point-cloud / proprioception / tactile observations;
- physical replay;
- learned-policy real-robot evaluation.

It is **not** a generic robotics platform, production serving stack, distributed command transaction layer, or generic dataset framework.

Priorities:

**physical safety > experiment correctness > research iteration speed > readability > generic extensibility > enterprise robustness**

Prefer delete, then inline, then merge. Add abstractions only for a real hardware/resource boundary or demonstrated duplication.

---

# 2. Verified baseline — preserve exactly

The latest code already corrected earlier review findings. Do not regress them.

## 2.1 Point-cloud observation semantics

Preserve:

- point-cloud-only policy:
  - latest fresh published point cloud;
  - no lookup of its historical source RGB-D frame;
- RGB-only policy:
  - latest fresh RGB;
- RGB + point-cloud policy:
  - exact source identity via `source_camera_sequence`.

`source_camera_sequence` is scientifically useful provenance. Keep it.

Do not make point-cloud-only rollout depend on camera-ring history.

## 2.2 Cartesian IK collision semantics

Preserve:

- final prepared hand target is installed into the planner before teleop / `action_ee` IK;
- online Cartesian IK rejects self-colliding target configurations;
- return-home planning uses the intended measured/final hand geometry.

Do not add current-to-target transition, environment, workspace, or table path collision checking to normal teleop/eval/replay.

Full path planning remains owned by planned `return_home`.

## 2.3 Command transport and motion authority

Preserve latest-target semantics:

```text
high-level controller
  -> latest RobotCommand mailbox
  -> worker observes newest sequence
  -> run_id / safety check at SDK boundary
  -> vendor SDK
```

Keep `run_id` as the stale-work lifecycle fence.

Do not add:

- command ACK/adoption;
- public/scientific command IDs;
- ordered action FIFO;
- single-inflight transactions;
- adoption/reached ledgers;
- command leases.

## 2.4 Hardware control philosophy

Preserve:

- xArm Mode 6 absolute targets and controller-side online replanning;
- XHand validated absolute position targets;
- relaxed XHand HOME completion on accepted SDK send;
- one owning process per live hardware SDK.

Do not add:

- host-side xArm trajectory interpolation;
- XHand target interpolation;
- 200/300 Hz repeated hand sends;
- hidden software slew filters;
- future timestamp waypoint scheduling.

## 2.5 Data architecture

Preserve:

- raw episodes as experiment source of truth;
- actual monotonic experiment timestamps;
- strict whole-episode raw -> canonical Zarr admission;
- no row repair, resampling, interpolation, splitting, or silent salvage;
- canonical Zarr as the full fixed training cache;
- recorder transaction / atomic publication behavior unless a concrete bug is found.

Do not migrate raw schema v32 merely for field cleanliness.

---

# 3. Scope

Implement only these changes:

1. enforce service-rate compatibility for every research-semantic latest-target command producer;
2. make normal observation freshness producer-owned and cadence-derived;
3. move XHand feedback-read failure timeout to XHand hardware config;
4. immediately terminate a recorded teleop demo after its first permanently inadmissible control row;
5. remove the arbitrary point-count whitelist;
6. make arm/hand 30 Hz worker timing sleep-only;
7. remove dead/self-proof `camera_requested` and `pointcloud_requested` state;
8. add lightweight, accurately named policy timing summaries.

No runtime redesign.

---

# 4. Change A — command producer rate vs worker service rate

## Why

`robot_command_ring` is a latest-target mailbox with `maxlen=1`.

Publishing faster than the relevant actuator worker service cadence can systematically overwrite intended high-level targets before the SDK observes them.

The validation here is deliberately modest:

> prevent a configured high-level producer cadence from exceeding its relevant worker cadence.

This is **not** an ACK guarantee that every target reaches hardware. Latest-target semantics remain intentional.

Do not introduce a 2x margin, scheduler, phase lock, or delivery accounting.

Use a tiny floating-point tolerance so mathematically equal rates are accepted.

## 4.1 VR teleop

At the central cross-section config boundary, currently `validate_config(cfg)` in `dexmani_real/config/experiment.py`:

```python
limiting_hz = cfg.arm.loop_hz
if cfg.policy.hand_enabled:
    limiting_hz = min(limiting_hz, cfg.hand.loop_hz)
```

Reject when:

```text
cfg.teleop.control_hz > limiting_hz
```

within a small numerical tolerance.

Do not use the stale variable name `runtime` inside `validate_config`; the argument is `cfg`.

## 4.2 Keyboard jog / camera calibration control cadence

`keyboard_teleop.control_hz` drives arm-only latest-target command production in keyboard jog and camera-calibration motion.

Validate centrally:

```text
cfg.keyboard_teleop.control_hz <= cfg.arm.loop_hz
```

The hand worker does not limit this arm-only command path.

## 4.3 Learned policy

In `validate_policy_runtime_compatibility(policy_spec, runtime)`:

```python
policy_hz = 1.0 / float(policy_spec.control_dt_s)
limiting_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
```

Current Real policy deployment requires arm7 + hand12, so both worker rates matter.

Reject if policy cadence exceeds the limiting worker cadence.

Report both rates in the error.

## 4.4 Physical replay

Replay publishes every recorded arm+hand target once at `trajectory.fps`.

In the existing replay preflight boundary, reject:

```text
trajectory.fps > min(runtime.arm.loop_hz, runtime.hand.loop_hz)
```

before motion starts.

Do not change replay scheduling itself.

## Acceptance

- default VR teleop 16 Hz + workers 30 Hz: pass;
- keyboard 30 Hz + arm worker 30 Hz: pass;
- learned policy 16 Hz + workers 30 Hz: pass;
- replay 16 Hz + workers 30 Hz: pass;
- any relevant producer configured above its limiting worker rate: fail before motion/startup;
- no new delivery protocol.

---

# 5. Change B — producer-owned normal observation freshness

## Current problem

Current normal observation freshness is both overly permissive and partly owned by `PolicyParams`:

```text
arm state     0.5 s
hand state    1.0 s
camera        0.25 s
```

At 30 Hz that is roughly 15, 30, and 7.5 producer periods.

A controller should not treat nearly one-second-old hand/tactile feedback as current.

## Target rule

Use one internal rule for **normal runtime observation consumption**:

```text
max usable age = 4 nominal producer periods
```

At 30 Hz:

```text
4 / 30 ~= 0.133 s
```

This is an implementation constant, not a YAML research knob.

Use a small private helper/constant in the owning config module if that avoids repetition. Do not add a generic timing framework.

## 5.1 Arm

Remove:

```text
PolicyParams.arm_state_stale_threshold_s
```

Expose a derived, non-dataclass-field property on `ArmParams`, preferably named clearly to distinguish it from homing-specific freshness, e.g.:

```python
ArmParams.feedback_max_age_s
```

computed from `loop_hz`.

Update all normal arm-feedback consumers currently reading the policy field, including camera calibration paths.

### Important

Do **not** replace or reinterpret:

```text
ArmParams.homing.state_max_age_s
```

That is a separate planned-homing contract and remains unchanged.

## 5.2 Hand

Remove:

```text
PolicyParams.hand_state_stale_threshold_s
```

Expose:

```python
HandParams.feedback_max_age_s
```

derived from `loop_hz`.

Update:

- normal observation assembly;
- return-home measured-hand freshness;
- all other actual hand-feedback freshness consumers.

## 5.3 Camera and derived point cloud

Remove `CameraParams.max_frame_age_s` as a user-configurable dataclass field.

Retain the same public property name if convenient, but derive it from:

```text
4 / fps
```

Point-cloud freshness remains tied to the source camera cadence because the published cloud keeps the source camera acquisition timestamp.

Preserve:

```text
CameraParams.source_stall_timeout_s
```

as an independent producer-failure timeout.

Keep the invariant:

```text
source_stall_timeout_s > derived max_frame_age_s
```

## 5.4 VR

Do not change VR/HTS freshness in this task.

Its producer cadence and source semantics differ from the fixed-rate arm/hand/camera producers.

## Configuration semantics

Derived freshness should not become another persisted/user-editable field.

Do not add backward-compatibility aliases for removed external config keys. Repository config loading intentionally rejects stale unknown fields.

Search and update repository-owned docs/examples if any reference the removed keys.

---

# 6. Change C — XHand feedback failure timeout belongs to HandParams

## Current problem

`PolicyParams.hand_disconnect_timeout_s` is actually passed to `hand_loop()` as the duration for repeated `get_state()` failure before worker failure.

That is XHand hardware producer behavior.

## Implementation

Move/rename it to `HandParams`, preferably:

```python
state_read_failure_timeout_s: float = 1.0
```

Validate finite and positive in `HandParams.validate()`.

Prefer simplifying the worker signature to:

```python
def hand_loop(shared, config):
```

and read the timeout from `config` internally.

Update **all** current process constructors, including:

- teleop;
- learned-policy deployment;
- keyboard teleop;
- physical replay;
- any other repository search hit.

Remove the old `PolicyParams` field and validation.

## Preserve distinct meanings

Do not merge:

```text
~133 ms normal feedback freshness
1.0 s repeated hardware feedback-read failure timeout
```

A stale sample should stop being usable well before the worker declares the hardware path failed.

Do not add reconnect machinery.

---

# 7. Change D — recorded teleop fails fast on the first inadmissible control row

## Verified behavior

`run_control_grid_tick()` already:

1. computes the target/status;
2. records the row when recording is active;
3. returns `FRAME_OK`, `FRAME_IK_FAIL`, or `FRAME_RETARGET_FAIL`.

Offline admission rejects the whole episode on the first non-`FRAME_OK` row.

Therefore, after the first failed recorded row, continuing that recording cannot produce a canonical training episode.

## Required recording behavior

Immediately after `run_control_grid_tick(...)` returns:

- if recording is active and `status != FRAME_OK`:
  1. keep the failed row already written by `run_control_grid_tick()`;
  2. call the existing teleop `stop(...)` path with:
     - `save=True`;
     - concrete reason (`ik_failure` or `retarget_failure`);
     - `incomplete=True`;
  3. continue the outer loop.

The existing `stop(..., incomplete=True)` already:

- revokes motion;
- marks recorder technical status invalid;
- requests recorder finalization with partial retention intent;
- sets recording state false.

**Do not call `recorder.join_stop()` synchronously from the 16 Hz control-tick path.**

Let the existing `poll_stop()` / recorder transaction finish asynchronously.

Do not change recorder transaction semantics.

## Unrecorded/debug teleop

Keep the useful existing behavior that repeated control failures eventually pause debug teleop, but remove it from user-facing experiment config.

Replace `PolicyParams.max_consecutive_errors` with a small private teleop-loop implementation constant (retain current value 10 unless current code gives a concrete reason otherwise).

Reset the local failure counter on a fresh begin/resume.

For recorded teleop, the first failed row always wins; the debug threshold must never delay invalid-demo termination.

---

# 8. Change E — remove arbitrary point-count whitelist

## Current problem

Realtime/deployment code currently restricts point counts to:

```python
{1024, 2048, 4096, 8192}
```

But:

- `PointCloudConfig` already validates `num_points` as a positive integer;
- realtime IPC dtype is dynamically built per deployment;
- policy/runtime compatibility already compares runtime point count to PolicySpec shape.

The whitelist blocks legitimate research ablations.

## Final contract

Require only:

```text
N is a positive non-bool integer
point cloud is float32 [N, 6]
runtime PointCloudConfig.num_points == PolicySpec point_cloud.shape[0]
```

## Remove

Delete `SUPPORTED_POINT_CLOUD_COUNTS` and all imports/usages.

Current known locations include:

- `dexmani_real/ipc/schema.py`;
- `dexmani_real/ipc/channels.py`;
- `dexmani_real/sensor/pointcloud_worker.py`;
- `dexmani_real/deployment/config.py`;
- `examples/export_policy_zarr.py`;
- `examples/visualize_episode.py`.

Search current `main` for any others.

## Specific behavior

`make_pointcloud_frame_dtype(num_points)`:
- reject bool;
- reject non-integer values;
- reject `<= 0`;
- build dynamic dtype otherwise.

`RuntimeChannelsConfig.pointcloud_num_points`:
- positive integer only.

Policy point-cloud capability:
- dtype `float32`;
- rank 2;
- second dimension exactly 6;
- first dimension positive;
- runtime N must separately equal policy N.

CLI tools:
- remove enumerated `choices`;
- preserve simple integer parsing;
- validate at the owning config/processing boundary.

Do not replace the whitelist with another arbitrary maximum or memory planner.

---

# 9. Change F — arm/hand worker LoopRate should sleep, not busy-spin

`LoopRate` already supports owner-selected `busy_wait=False`.

Do not rewrite `LoopRate`.

Change only the SDK service loops:

```python
LoopRate(config.loop_hz, label="arm", busy_wait=False)
LoopRate(config.loop_hz, label="hand", busy_wait=False)
```

These are 30 Hz command-admission / feedback-update service loops, not low-level hard-real-time servo clocks.

Do not change control-grid timing mechanisms in teleop/policy/calibration as part of this item.

Fix stale comments/docs that call `arm.loop_hz` or `hand.loop_hz` a physical "servo rate".

Preferred wording:

```text
worker command-admission / feedback-update rate
```

---

# 10. Change G — delete dead/self-proof requested flags

## `camera_requested`

There is no meaningful runtime consumer of `shared.camera_requested.value`.

Process construction already determines whether the camera process exists.

Remove it.

## `pointcloud_requested`

Its current runtime purpose is a point-cloud worker startup self-check, even though the parent explicitly decides whether to spawn that worker.

Remove it and the startup assertion.

## Remove full wiring

Delete requested fields/arguments from:

- `RuntimeChannelsConfig`;
- `RuntimeChannelsConfig.from_runtime()`;
- validation;
- `RuntimeChannels`;
- shared resource allocation;
- teleop/deployment/calibration session construction;
- point-cloud worker startup;
- docs/comments.

## Do not over-optimize IPC allocation

Keep the current standard rings allocated.

Do **not** make camera/point-cloud rings optional and do not add workflow-specific `None` channel shapes just to save small amounts of shared memory.

---

# 11. Change H — lightweight policy timing summary with accurate semantics

## Goal

Determine whether synchronous policy inference materially reduces the achieved action cadence without building a tracing system.

## Inference

Continue collecting inference latency and report:

- mean;
- p95;
- max.

## Action-step interval

Track the interval between successive **successful PolicyRunner action steps** using the existing `stamp`:

- when `execute=True`, `stamp` is the actual host publication timestamp returned by `publish_command()`;
- when `execute=False`, `stamp` is the local monotonic step timestamp used by the dry-run path.

Therefore name/log this metric accurately as an action-step interval/effective action-step Hz, and include `execute=<bool>` in the summary.

Do not claim dry-run intervals are hardware publication intervals.

Reset the previous-step timestamp in `_begin()` so inter-episode HOME/operator gaps are excluded.

If a failed IK step produces no action publication, the larger interval before the next successful action step should remain visible; it reflects real runner behavior.

Report:

- configured action rate `1 / control_dt_s`;
- action-step interval mean / p95 / max;
- effective action-step Hz from mean interval.

Handle empty/one-sample cases without NumPy warnings or divide-by-zero.

## Do not add

- HDF5/raw timing fields;
- IPC telemetry;
- generic profiler/tracer;
- history-reset counters;
- worker timing instrumentation.

---

# 12. Explicit non-goals

Do not implement:

- XHand command-before-feedback reordering;
- separate tactile worker/thread;
- arm worker 30 -> 60 Hz;
- hand worker rate increase;
- xArm host-side interpolation;
- XHand interpolation;
- async policy inference;
- inference/actuation overlap;
- future timestamp scheduling;
- stale-prefix action pruning;
- actuator latency compensation;
- online sensor resampling;
- missing-frame repetition/interpolation;
- generic tracing/observability infrastructure;
- new committed tests directory;
- raw schema migration;
- recorder protocol redesign;
- collision architecture redesign.

### XHand ordering

Current hand worker reads feedback before normal command admission.

That may add latency if `get_state()` is slow, but it also means a successful feedback read precedes each newly admitted target.

Do not change that ordering without hardware timing evidence.

---

# 13. Expected files and dependency search

Likely files include:

```text
dexmani_real/config/defaults.py
dexmani_real/config/experiment.py
dexmani_real/deployment/config.py
dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py

dexmani_real/runtime/observation.py

dexmani_real/robot/arm_worker.py
dexmani_real/robot/hand_worker.py
dexmani_real/robot/arm_homing.py

dexmani_real/teleop/loop.py
dexmani_real/teleop/session.py
dexmani_real/teleop/keyboard_session.py

dexmani_real/replay/session.py
dexmani_real/replay/trajectory.py

dexmani_real/calibration/camera/motion.py
dexmani_real/calibration/camera/session.py

dexmani_real/ipc/schema.py
dexmani_real/ipc/channels.py

dexmani_real/sensor/pointcloud_worker.py

examples/export_policy_zarr.py
examples/visualize_episode.py
```

Search actual current definitions/usages before editing. The list is guidance, not a mandate to touch every file.

Avoid touching unless required by a real dependency:

```text
planning/collision.py
planning/kinematics/ik.py
point-cloud algorithm internals
recording/storage raw schema
recording transaction internals
dataset whole-episode admission semantics
```

---

# 14. Implementation order

Use this order to reduce churn and make failures local:

1. config ownership changes:
   - derived freshness;
   - XHand failure timeout move;
   - remove max-consecutive-errors knob;
   - central teleop/keyboard rate validation;
2. update all consumers/call sites of removed config;
3. policy and replay rate compatibility;
4. point-count whitelist removal end-to-end;
5. dead requested-state removal end-to-end;
6. teleop recorded-failure fail-fast;
7. arm/hand sleep-only rate limiter;
8. policy timing summary;
9. repository-wide stale-reference cleanup;
10. offline validation.

Do not create temporary compatibility layers just to stage the refactor.

---

# 15. Documentation cleanup

Update repository-owned comments/docs only where changed semantics require it.

Make clear that:

- VR teleop rate is owned by `teleop.control_hz`;
- keyboard/calibration jog rate is owned by `keyboard_teleop.control_hz`;
- learned-policy action spacing is owned by `PolicySpec.control_dt_s`;
- replay cadence comes from the recorded trajectory;
- arm/hand `loop_hz` is worker service cadence, not hardware physical servo frequency;
- normal arm/hand/camera freshness is derived from producer cadence.

Do not add long design-manifesto docstrings to source.

Do not add backward-compatibility aliases for removed config keys.

---

# 16. Validation

No hardware execution.

## Required low-cost checks

Run when available:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If Ruff is unavailable, report it. Do not install/upgrade the experiment environment.

## Focused one-off offline smoke checks

Do not add a committed `tests/` directory.

### Rate contracts

Verify:

- VR teleop 16, workers 30 -> pass;
- keyboard 30, arm worker 30 -> pass;
- policy 16, workers 30 -> pass;
- replay 16, workers 30 -> pass;
- each producer above its limiting relevant worker -> reject.

### Freshness

For 30 Hz producer:

```text
normal max age ~= 4 / 30 s
```

Verify:

- arm uses `ArmParams` derived feedback freshness;
- hand uses `HandParams` derived feedback freshness;
- camera/cloud use camera-derived freshness;
- homing-specific `ArmParams.homing.state_max_age_s` remains unchanged;
- VR freshness remains unchanged.

### XHand timeout ownership

Repository runtime code should no longer reference:

```text
PolicyParams.hand_disconnect_timeout_s
```

and all hand-worker constructors should use the simplified hardware-owned config path.

### Point-cloud counts

Verify representative counts:

```text
512
1536
3072
```

work through relevant config / IPC dtype / policy compatibility paths when shapes match.

Verify bool, zero, negative, and non-integer counts fail at an owning boundary.

### Recorded teleop failure

With pure logic/mocks only:

- first recorded `FRAME_IK_FAIL` row is submitted before stop;
- first recorded `FRAME_RETARGET_FAIL` row is submitted before stop;
- the existing non-blocking `stop(save=True, incomplete=True)` path is requested immediately;
- no control-loop `join_stop()` was added;
- unrecorded repeated-failure debug behavior still pauses after the private threshold.

### Dead state / stale config search

No runtime source definition/use of:

```text
camera_requested
pointcloud_requested
SUPPORTED_POINT_CLOUD_COUNTS
arm_state_stale_threshold_s
hand_state_stale_threshold_s
hand_disconnect_timeout_s
max_consecutive_errors
```

Occurrences in `codex_task.md` are expected.

### Timing summary

Exercise the pure summary/statistic path with:

- no samples;
- one sample;
- multiple synthetic samples.

No warnings/divide-by-zero.

---

# 17. Final diff review

Before handoff:

1. inspect `git diff --stat`;
2. inspect `git diff`;
3. search for deleted symbols/stale terminology;
4. confirm no unrelated architecture changes;
5. confirm no hardware code was executed;
6. confirm no raw schema change;
7. confirm no new worker/thread/process/protocol was introduced.

Prefer deleting obsolete imports/validation branches over leaving compatibility debris.

---

# 18. Hardware commissioning — report only

Do not execute these measurements in this task.

Recommend for a later authorized hardware session:

1. XHand `get_state()` latency: mean / p95 / max;
2. XHand `send_action()` latency: mean / p95 / max;
3. high-level arm target publication -> xArm SDK send latency: mean / p95 / max;
4. policy inference latency: mean / p95 / max;
5. real successful action publication interval / effective action Hz.

Only those measurements should motivate later decisions about:

- XHand feedback/command decoupling;
- arm worker 30 -> 60 Hz;
- async policy inference.

Do not pre-implement them.

---

# 19. Final acceptance criteria

Complete only when:

- latest-target transport and run_id fencing are unchanged;
- xArm Mode 6 and XHand absolute-target semantics are unchanged;
- relaxed XHand HOME semantics are unchanged;
- current Cartesian IK + final-hand self-collision semantics are unchanged;
- point-cloud-only and RGB+point-cloud semantics are unchanged;
- VR teleop, keyboard/calibration jog, learned policy, and physical replay cannot be configured to publish faster than their relevant worker service cadence;
- normal arm/hand/camera freshness is producer-owned and cadence-derived;
- homing-specific and VR freshness semantics are not accidentally changed;
- XHand state-read failure timeout is hardware-owned;
- a recorded teleop episode immediately terminates after the first recorded IK/retarget failure row without blocking recorder finalization in the control tick;
- arbitrary positive point counts are supported;
- arm/hand workers no longer busy-spin for 30 Hz service timing;
- `camera_requested` / `pointcloud_requested` are gone;
- policy timing summary is accurate for execute and dry-run modes;
- no scheduler, ACK layer, compatibility layer, raw schema version, worker, thread, or process was added;
- offline checks pass or unavailable optional tooling is explicitly reported;
- hardware validation remains pending.

---

# 20. Handoff

Report concisely:

1. files changed;
2. semantic changes;
3. deleted config/state/validation;
4. offline checks and results;
5. anything not validated because it requires hardware;
6. the five commissioning measurements above;
7. any deviation from this task and exact reason.

Do not claim hardware correctness from offline checks.
