# Codex Task — Simplify and Tighten DexMani Real Runtime Semantics

## Task status

This task is based on the current `main` fact-checked at commit:

`46ba86710a62cda1233d522d62fc20627ad1dd4d`

Before editing, re-read the current repository state and `AGENTS.md`. If `main` has moved, preserve the intent below and reconcile against the new code instead of mechanically applying stale line-level assumptions.

## Repository intent

`dexmani_real` is a personal PhD research repository for:

- dexterous real-robot data collection;
- VR/HTS teleoperation;
- xArm7 + XHand control;
- RGB-D / point-cloud / proprioception / tactile observations;
- learned-policy real-robot evaluation.

It is **not** a generic robotics runtime, production serving system, distributed command transaction layer, or generic dataset framework.

Priorities remain:

**physical safety > experiment correctness > research iteration speed > readability > generic extensibility > enterprise robustness**

Prefer the smallest implementation that preserves the actual research and hardware guarantees.

Do not replace deleted mechanisms with equivalent machinery under new names.

---

# 1. Verified baseline — do not regress

The latest code already fixed several previous review findings. Treat these as correct baseline behavior.

## 1.1 Point-cloud observation semantics

Preserve the current split:

- point-cloud-only policy:
  - consumes the latest fresh published point cloud;
  - does **not** look up the source RGB-D frame;
- RGB-only policy:
  - consumes the latest fresh RGB frame;
- RGB + point-cloud policy:
  - uses `source_camera_sequence` to require exact source-frame identity.

`source_camera_sequence` is useful scientific provenance. Do not remove it.

Do not reintroduce a point-cloud-only dependency on historical camera-ring retention.

## 1.2 Cartesian IK self-collision

Preserve the current behavior:

- teleop hand target is prepared first;
- policy `action_ee` hand target is prepared first;
- `planner.set_hand_qpos(final_prepared_hand)` is applied before Cartesian IK;
- online IK rejects self-colliding target configurations;
- return-home planning uses appropriate measured/final hand geometry.

Do not add normal-runtime current-to-target path/environment collision checking.

Full path collision/workspace/table planning remains owned by planned `return_home`.

## 1.3 Command transport

Preserve latest-target semantics:

```text
controller
  -> latest RobotCommand mailbox
  -> worker observes newest sequence
  -> final run_id/safety check
  -> SDK
```

Do not introduce:

- command ACK/adoption;
- command IDs as public/scientific identity;
- actuator ledgers;
- ordered command FIFO;
- single-inflight transactions;
- command lease machinery.

Keep `run_id` as the lifecycle stale-work fence.

## 1.4 xArm / XHand control philosophy

Preserve:

- xArm Mode 6 absolute targets and controller-side online replanning;
- XHand validated absolute position targets;
- relaxed XHand HOME completion on accepted send rather than strict measured convergence;
- arm/hand worker processes as the sole SDK owners.

Do not add:

- host-side xArm trajectory interpolation;
- 200/300 Hz XHand target resend;
- hidden actuator slew filters;
- timestamp waypoint scheduling.

## 1.5 Data architecture

Preserve:

- raw episodes as experiment source of truth;
- actual monotonic timestamps;
- strict whole-episode raw -> canonical Zarr admission;
- no row dropping, interpolation, repair, or silent salvage;
- canonical Zarr as a full fixed training cache;
- current recorder transaction / atomic publication architecture unless a concrete bug is found.

Do not change raw schema v32 merely for field-cleanliness.

---

# 2. Goal of this task

Make one narrow cleanup pass that improves correctness while reducing runtime/configuration complexity.

The task should accomplish exactly these high-value changes:

1. enforce control-rate compatibility with latest-target worker service rates;
2. make observation freshness producer-owned and cadence-derived;
3. move XHand read-failure timeout to the XHand hardware config owner;
4. fail fast when a recorded teleop demonstration becomes permanently inadmissible;
5. remove the arbitrary point-count whitelist;
6. stop busy-spinning the 30 Hz arm/hand worker service loops;
7. remove dead/self-proof `camera_requested` / `pointcloud_requested` shared state;
8. add lightweight policy timing summary statistics.

Do **not** expand scope into a runtime redesign.

---

# 3. Required change A — control-rate compatibility

## Problem

`robot_command_ring` is a latest-target mailbox with `maxlen=1`.

Arm and hand workers currently service commands at `runtime.arm.loop_hz` and `runtime.hand.loop_hz`.

If a controller publishes targets faster than the relevant worker can observe them, intermediate intended control targets can be overwritten before reaching the SDK.

This is valid latest-target transport behavior, but it violates the meaning of a configured research control grid if allowed silently.

The current default is fine:

```text
teleop = 16 Hz
arm worker = 30 Hz
hand worker = 30 Hz
```

The missing piece is a boundary validation that prevents future invalid configs/checkpoints.

## Implementation

### Teleop configuration

In the central experiment cross-section validation boundary (currently `validate_config()` in `dexmani_real/config/experiment.py`), validate:

```python
worker_hz = runtime.arm.loop_hz
if runtime.policy.hand_enabled:
    worker_hz = min(worker_hz, runtime.hand.loop_hz)

runtime.teleop.control_hz <= worker_hz
```

Reject a teleop rate that exceeds the service rate of any actuator participating in teleop.

Keep the validation simple. No 2x safety factor, Nyquist-style rule, phase synchronization, or scheduling abstraction is needed.

### Learned policy configuration

In `validate_policy_runtime_compatibility()`:

```python
policy_hz = 1.0 / policy_spec.control_dt_s
worker_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
```

Reject if `policy_hz > worker_hz`.

Current Real learned-policy deployment requires arm7 + hand12, so both worker rates matter.

Use a numerically reasonable comparison; do not reject equality because of floating-point representation noise.

Error messages should report the policy/teleop rate and limiting worker rate.

## Acceptance

- 16 Hz teleop + 30 Hz workers: pass.
- 30 Hz controller + 30 Hz workers: pass.
- controller > 30 Hz with 30 Hz relevant worker: reject before startup.
- no new runtime scheduler or transport state.

---

# 4. Required change B — producer-owned freshness

## Problem

Current normal observation freshness is overly permissive and owned by the wrong config section:

```text
PolicyParams.arm_state_stale_threshold_s  = 0.5 s
PolicyParams.hand_state_stale_threshold_s = 1.0 s
CameraParams.max_frame_age_s              = 0.25 s
```

At 30 Hz these correspond to roughly 15, 30, and 7.5 producer periods.

A dexterous policy should not continue treating hand/tactile state from almost one second ago as current.

Freshness is a property of the producer cadence, not of a learned policy.

## Target semantics

Use one internal research-runtime rule:

```text
normal observation freshness = 4 nominal producer periods
```

At 30 Hz this is approximately 133 ms.

This provides a scheduling cushion while keeping stale data bounded to only a small number of physical updates.

## Implementation

Use one private/internal constant, e.g.:

```python
_OBSERVATION_FRESHNESS_PERIODS = 4.0
```

Avoid adding a new user-facing YAML knob.

### Arm

Remove normal arm observation freshness from `PolicyParams`.

Expose it from the arm producer config as a derived property:

```python
ArmParams.state_max_age_s
    = _OBSERVATION_FRESHNESS_PERIODS / loop_hz
```

Update normal observation consumers and camera calibration paths that currently read `runtime.policy.arm_state_stale_threshold_s`.

### Hand

Remove normal hand observation freshness from `PolicyParams`.

Expose:

```python
HandParams.state_max_age_s
    = _OBSERVATION_FRESHNESS_PERIODS / loop_hz
```

Update:

- normal observation assembly;
- return-home measured-hand freshness use;
- any other actual hand-state freshness consumer.

### Camera

Remove user-configurable normal `max_frame_age_s` as a dataclass field and make it a cadence-derived property:

```python
CameraParams.max_frame_age_s
    = _OBSERVATION_FRESHNESS_PERIODS / fps
```

Preserve `source_stall_timeout_s` as an independent hardware/source failure timeout.

Its meaning is different:

- `max_frame_age_s`: whether a consumer may still use the latest sample;
- `source_stall_timeout_s`: when the camera producer declares source failure.

Keep validation that the stall timeout is greater than normal frame freshness.

### Important exclusions

Do **not** conflate this task with:

- `HomingParams.state_max_age_s`, which belongs to the dedicated planned homing procedure;
- VR/HTS freshness, whose producer cadence is not being normalized in this task;
- device failure/reconnect timeouts.

Do not add adaptive frequency estimation.

## Config migration philosophy

Do not add backward-compatibility aliases for removed freshness keys.

This repository intentionally validates external config strictly. Update repository-owned docs/examples/config references instead.

---

# 5. Required change C — move XHand read-failure timeout to HandParams

## Problem

`PolicyParams.hand_disconnect_timeout_s` is passed to the XHand worker as the state-read failure timeout.

This is hardware producer behavior, not learned-policy behavior.

## Implementation

Move/rename the setting into `HandParams`, preferably with semantics matching actual use, e.g.:

```python
state_read_failure_timeout_s: float = 1.0
```

Validate it in `HandParams.validate()`.

Simplify the worker boundary where practical:

```python
def hand_loop(shared, config):
    ...
    timeout = config.state_read_failure_timeout_s
```

Update all process construction call sites.

Remove the old field from `PolicyParams`.

Do not introduce a generic reconnect state machine.

## Preserve two different time scales

Normal sample freshness and device failure timeout must remain distinct:

```text
~133 ms:
latest sample is no longer usable by controller

1.0 s:
worker has failed to obtain usable feedback long enough to declare hardware/runtime failure
```

---

# 6. Required change D — fail fast on invalid recorded teleop demonstrations

## Verified current mismatch

During teleop, `FRAME_IK_FAIL` / `FRAME_RETARGET_FAIL` rows are recorded, but the loop waits for `max_consecutive_errors=10` before pausing.

Offline `validate_episode()` rejects the entire episode on the **first** non-`FRAME_OK` row.

Therefore once the first failed row occurs during recording, that demonstration is permanently inadmissible for canonical training export.

Continuing to record it wastes operator time and produces data known to be unusable for training.

## Required behavior

### Recording teleop

On the first non-`FRAME_OK` control row:

1. preserve/write that failed raw row exactly as today;
2. mark the episode technically invalid/incomplete;
3. immediately stop the recording episode / revoke active motion through existing lifecycle helpers;
4. retain the partial raw episode so the failure reason remains inspectable;
5. give a concrete stop reason such as `ik_failure` or `retarget_failure`.

Do not silently drop the failed row.

Do not salvage the previous rows into canonical training data.

### Unrecorded/debug teleop

A failed control tick may continue to behave as a skipped command.

If automatic pause after repeated failures is useful for debug UX, keep a small module-local implementation constant rather than a user-facing experiment parameter.

## Simplification

Remove `PolicyParams.max_consecutive_errors`.

Do not create a generic episode-validity state machine.

Use the existing recorder/lifecycle functions.

---

# 7. Required change E — remove the point-count whitelist

## Problem

The current runtime unnecessarily restricts point counts to:

```python
{1024, 2048, 4096, 8192}
```

The actual infrastructure already supports dynamically sized fixed deployment buffers and `PointCloudConfig` already validates `num_points` as a positive integer.

The whitelist blocks legitimate 3D research ablations such as 512, 768, 1536, 3072, etc.

## Required contract

Only require:

```text
N is a positive integer
point cloud is float32 [N, 6]
runtime PointCloudConfig.num_points == PolicySpec point_cloud.shape[0]
```

## Remove

Remove `SUPPORTED_POINT_CLOUD_COUNTS` and all imports/usages from at least:

- `dexmani_real/ipc/schema.py`;
- `dexmani_real/ipc/channels.py`;
- `dexmani_real/sensor/pointcloud_worker.py`;
- `dexmani_real/deployment/config.py`;
- `examples/export_policy_zarr.py`;
- `examples/visualize_episode.py`;
- any other current source location found by repository search.

### Specific replacements

`make_pointcloud_frame_dtype(num_points)`:
- validate positive integer;
- create the dynamic dtype.

`RuntimeChannelsConfig.pointcloud_num_points`:
- validate positive integer;
- no enumerated choices.

Policy compatibility:
- require `shape == [positive N, 6]`, dtype `float32`;
- separately require runtime N to equal policy N.

CLI tools:
- use `type=int`;
- remove `choices=...`;
- rely on the owning config/processing boundary to reject non-positive values.

Do not add a maximum-memory planner or a replacement whitelist.

---

# 8. Required change F — make arm/hand worker rate limiting sleep-only

## Verified fact

`LoopRate` already supports owner-selected `busy_wait=False`.

Recorder already uses this correctly.

Arm and hand currently construct:

```python
LoopRate(config.loop_hz, label="arm")
LoopRate(config.loop_hz, label="hand")
```

Production default therefore busy-spins in the final timing window.

These 30 Hz loops are SDK command-admission / feedback-update service loops, not hard-real-time low-level servo clocks.

## Implementation

Use:

```python
LoopRate(config.loop_hz, label="arm", busy_wait=False)
LoopRate(config.loop_hz, label="hand", busy_wait=False)
```

Do not rewrite `LoopRate` in this task.

Do not remove its deterministic-test clock/sleep injection or other utility behavior merely for cleanup.

Fix stale wording such as:

```python
loop_hz: float = 30.0  # arm_loop servo rate
```

Use accurate terminology:

```text
worker command-admission / feedback-update rate
```

Do the same wherever documentation implies worker Hz is the hardware's physical servo frequency.

---

# 9. Required change G — remove dead requested flags

## `camera_requested`

Current source has no meaningful consumer of `shared.camera_requested.value`.

Process construction already determines whether a camera worker exists.

Remove the dead state.

## `pointcloud_requested`

Its only meaningful use is currently a point-cloud worker startup self-check:

```python
if not shared.pointcloud_requested.value:
    raise RuntimeError(...)
```

But the process exists only because the parent explicitly spawned it.

This check proves an already-established fact and adds no useful safety.

Remove it.

## Remove the full wiring

Delete the requested fields from:

- `RuntimeChannelsConfig`;
- validation;
- `from_runtime()`;
- `RuntimeChannels`;
- shared resource allocation;
- teleop/deployment/calibration call sites;
- point-cloud startup assertion;
- stale comments/docs.

## Do not over-optimize allocation

Continue allocating the current standard rings even when a workflow does not use every ring.

Do **not** make camera/point-cloud rings Optional merely to save a small amount of shared memory.

Avoid introducing `None` checks and workflow-specific channel shapes.

---

# 10. Required change H — lightweight policy timing summary

## Goal

Measure whether synchronous inference actually limits real execution without building a tracing framework.

Current summary reports only mean inference time.

Add lightweight in-process summary statistics.

## Track

Across policy execution:

- inference latency: mean / p95 / max;
- successful action publish interval: mean / p95 / max;
- effective action rate derived from mean successful publish interval;
- configured control rate for comparison.

Use actual publication timestamps.

Reset the "previous publication" timestamp at the start of each episode so inter-episode HOME/operator gaps do not contaminate publish-interval statistics.

It is fine for aggregate arrays/statistics to span multiple episodes as long as cross-episode intervals are excluded.

If an IK failure causes a skipped publication, the later larger successful-publication interval should remain visible; that is real execution behavior.

## Keep it simple

- log the summary at worker shutdown;
- do not add HDF5/raw fields;
- do not add IPC telemetry;
- do not add a generic profiler;
- do not add history-reset counters in this task.

A small local helper for mean/p95/max is acceptable if it reduces duplication.

---

# 11. Explicit non-goals / forbidden scope expansion

Do not implement any of the following in this task:

- XHand command-before-feedback reordering;
- a separate tactile thread/process;
- arm worker 30 -> 60 Hz;
- hand worker frequency increase;
- xArm host-side interpolation;
- XHand command interpolation;
- async policy inference;
- inference/actuation overlap;
- future timestamp scheduling;
- stale-prefix action pruning;
- actuator latency compensation;
- online sensor resampling;
- missing-frame repeat/interpolation;
- generic tracing/telemetry framework;
- new tests directory;
- raw schema migration;
- recorder protocol redesign;
- collision architecture redesign.

These require either hardware measurements or a separate research requirement.

### XHand ordering note

Current XHand worker reads feedback before admitting the next normal command.

Although moving command admission first might reduce latency if `get_state()` is slow, the current order also ensures a successful feedback read before a new target is sent.

Do **not** change this without hardware latency evidence.

Commissioning measurement belongs after this software task, not inside it.

---

# 12. File-level guidance

Expected files to inspect/change include, but are not limited to:

```text
dexmani_real/config/defaults.py
dexmani_real/config/experiment.py
dexmani_real/deployment/config.py
dexmani_real/runtime/observation.py

dexmani_real/robot/arm_worker.py
dexmani_real/robot/hand_worker.py
dexmani_real/robot/arm_homing.py

dexmani_real/teleop/loop.py
dexmani_real/teleop/session.py

dexmani_real/ipc/schema.py
dexmani_real/ipc/channels.py

dexmani_real/sensor/pointcloud_worker.py

dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py

examples/export_policy_zarr.py
examples/visualize_episode.py
```

Search actual current definitions/usages before editing; do not assume this list is exhaustive.

Avoid touching these areas unless required by an actual dependency of the changes above:

```text
planning/collision.py
planning/kinematics/ik.py
point-cloud algorithm internals
recording/storage raw schema
recorder transaction implementation
dataset whole-episode admission semantics
```

The latest versions of the collision/point-cloud observation fixes are intentional.

---

# 13. Documentation/config cleanup

After code changes:

- update README/config comments if they reference removed freshness fields, requested flags, supported point counts, or worker "servo rate";
- make it clear that:
  - teleop control frequency is owned by teleop config;
  - learned-policy action spacing is owned by `PolicySpec.control_dt_s`;
  - arm/hand `loop_hz` is worker service cadence, not physical low-level servo frequency;
  - normal observation freshness is cadence-derived;
- do not add lengthy design-manifesto docstrings to source files;
- keep durable rationale in `AGENTS.md` / README only when needed.

Do not add compatibility aliases for removed config fields.

---

# 14. Validation strategy

Do not run hardware-affecting code.

Start by inspecting:

```bash
git status --short
```

Preserve unrelated changes.

## Required low-cost checks

Run if available:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If Ruff is unavailable, report it; do not install or upgrade the experiment environment.

## Focused offline smoke checks

Use small one-off pure-Python checks, not a new committed tests directory.

Verify at minimum:

### Rate compatibility

- teleop 16 Hz, workers 30 Hz -> pass;
- controller 30 Hz, workers 30 Hz -> pass;
- controller > limiting worker Hz -> fail;
- policy `control_dt_s = 1/16`, workers 30 Hz -> pass;
- policy > worker rate -> fail.

### Freshness

At 30 Hz:

```text
state/frame max age ~= 4 / 30 s
```

Verify:

- arm observation uses arm producer-derived freshness;
- hand observation uses hand producer-derived freshness;
- camera/cloud use camera producer-derived freshness;
- homing-specific timeout/freshness semantics were not accidentally replaced.

### Point-cloud count

Verify arbitrary legitimate positive counts such as:

```text
512
1536
3072
```

can construct the point-cloud config / IPC dtype / runtime-policy compatibility path when shapes match.

Verify zero/negative/bool counts are rejected at the owning boundary.

### Teleop invalid demonstration behavior

Using pure logic/mocks only:

- first recorded `FRAME_IK_FAIL` row is retained and immediately ends/invalidates the episode;
- first recorded `FRAME_RETARGET_FAIL` behaves likewise;
- no row is silently discarded or converted to `FRAME_OK`;
- unrecorded debug control does not require recorder lifecycle behavior.

### Dead-state cleanup

Repository search should show no source-code runtime use/definition of:

```text
camera_requested
pointcloud_requested
SUPPORTED_POINT_CLOUD_COUNTS
PolicyParams.arm_state_stale_threshold_s
PolicyParams.hand_state_stale_threshold_s
PolicyParams.hand_disconnect_timeout_s
PolicyParams.max_consecutive_errors
```

Occurrences inside this task document are expected and should not count as stale runtime code.

### Timing summary

Exercise the pure summary logic with synthetic timing samples or by directly instantiating the minimal relevant helper state where possible.

Verify no divide-by-zero / empty-array warnings when no inference/publication samples exist.

---

# 15. Hardware commissioning — report only, do not execute

This task must **not** perform hardware validation.

At handoff, explicitly list the following recommended commissioning measurements for a later authorized real-robot session:

1. XHand `get_state()` duration:
   - mean / p95 / max;
2. XHand `send_action()` duration:
   - mean / p95 / max;
3. high-level arm command publication -> actual xArm SDK send latency:
   - mean / p95 / max;
4. policy inference:
   - mean / p95 / max;
5. actual successful action publication interval / effective action Hz.

These measurements decide later, in a separate task, whether there is evidence for:

- XHand command/read decoupling;
- arm worker 30 -> 60 Hz;
- async policy inference.

Do not pre-implement those mechanisms.

---

# 16. Final acceptance criteria

The task is complete only when all of the following are true:

- latest-target command semantics remain unchanged;
- xArm Mode 6 control semantics remain unchanged;
- XHand normal absolute-target and relaxed HOME semantics remain unchanged;
- current online IK self-collision behavior with final prepared hand target remains unchanged;
- point-cloud-only and RGB+point-cloud semantics remain unchanged;
- teleop and learned-policy rates cannot silently exceed relevant worker service rates;
- normal arm/hand/camera freshness is producer-owned and cadence-derived;
- XHand state-read failure timeout is hardware-owned, not policy-owned;
- a recorded teleop episode stops immediately after its first permanently inadmissible control row, while preserving that raw failure row;
- arbitrary positive point counts are supported;
- arm/hand 30 Hz worker loops no longer busy-spin for deadline precision;
- dead `camera_requested` / `pointcloud_requested` shared state is gone;
- policy shutdown summary reports useful inference and actual publication timing;
- no new scheduler, ACK protocol, worker, thread, process, compatibility layer, or raw schema version was introduced;
- source/docs contain no stale references caused by the removed config/state;
- offline checks pass, or unavailable optional tooling is clearly reported;
- hardware validation is explicitly left pending.

---

# 17. Handoff format

When finished, report concisely:

1. files changed;
2. semantic changes;
3. code/config deleted;
4. offline checks run and results;
5. anything not validated because it requires hardware;
6. the five commissioning measurements listed above;
7. any deviation from this task, with the exact reason.

Do not claim hardware correctness from offline smoke checks.
