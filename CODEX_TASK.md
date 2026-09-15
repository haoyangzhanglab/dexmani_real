# CODEX TASK — Restore Camera-Calibration Jog Responsiveness with ACK-Anchored Bounded Lookahead

> This is a **temporary implementation task file** for Codex. Read `AGENTS.md` first and obey it throughout the task. This repository controls physical robotics hardware. **Do not connect to xArm/XHand/RealSense and do not execute any hardware-affecting program for this task.** Work from source inspection and offline validation only.
>
> This task was authored against `main` at commit `383b76ca4a320c740b1710b4d28cf1fb20a069df`. Before editing, inspect the actual current `HEAD` and `git status --short`. If `HEAD` has advanced, re-trace the relevant code paths from current source and adapt the implementation to the current contracts; do not blindly apply stale line-level assumptions.
>
> If implementation and all safe offline acceptance checks succeed, **delete `CODEX_TASK.md` before final handoff** and re-run the final low-cost checks. If the task is incomplete or validation fails, keep this file so the work can be resumed.

---

## 1. Goal

Restore responsive, practical Cartesian jog speed in `examples/calibrate_camera.py` without reintroducing the command-reference drift and command-jump failure mode repaired on 2026-09-14.

The current camera-calibration motion law is intentionally safe but too conservative:

```text
current measured EEF pose
        + one jog increment
        -> IK
        -> publish
        -> synchronous arm SDK acceptance
```

With the current defaults, a held translation key keeps the commanded endpoint only about one configured jog step ahead of measured motion (`8 mm`), and a held rotation key only about one rotation step ahead (`0.03 rad`). In xArm Mode 6 online trajectory planning this behaves like a low-lead position follower and is substantially slower than keyboard teleoperation.

The target behavior is **ACK-anchored bounded lookahead**:

```text
last SDK-accepted arm command qpos
        -> FK = accepted command EEF anchor
        + one jog increment
        -> desired EEF target
        -> workspace clamp
        -> bound target lead relative to latest measured EEF pose
        -> IK(current measured qpos, previous accepted command)
        -> existing canonicalization + early jump guard + SafetyGate
        -> publish
        -> wait for arm SDK acceptance
        -> only then commit previous_command
```

Use a default calibration lookahead of **3 jog frames**:

```text
position step       = 0.008 m
rotation step       = 0.03 rad
lookahead frames    = 3
max position lead   = 0.024 m
max rotation lead   = 0.09 rad  (~5.16 deg)
```

This should make a held jog build useful command lead while preserving precise single-step control and a bounded distance from measured robot state.

---

## 2. Why this design

### 2.1 Current behavior is too slow

Current `dexmani_real/calibration/camera/motion.py` derives each moving target from the latest measured EEF pose:

```python
measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
desired_pos = measured_pose.p + dx
...
proposed_quat = measured_pose.q.copy()
```

Therefore, during sustained motion, command lead does not accumulate beyond approximately one jog increment. The next target mostly advances by however far the physical robot already moved.

### 2.2 Do not restore the old unbounded Cartesian virtual target

Before the 2026-09-14 command-jump recovery change, calibration persisted independent `target_pos` / `target_quat` state and advanced it every control tick. That could run ahead of physical tracking, especially in Mode 6, and contributed to producer/worker command-reference divergence and IK branch/jump failures.

Do **not** restore an unbounded open-loop Cartesian target.

### 2.3 Calibration already has a stronger anchor than keyboard teleoperation

Calibration synchronously calls `wait_command_accepted()` and advances `state.previous_command` only after arm SDK acceptance. This gives calibration a valuable invariant:

```text
state.previous_command ~= last producer command confirmed to have crossed the arm SDK boundary
```

Use that accepted joint endpoint as the persistent command anchor. Do not add persistent `target_pos` / `target_quat` back to `CalibrationLoopState`.

### 2.4 Why bounded lookahead is still necessary

Using `FK(previous_command) + jog_delta` without a measured-state lead cap could still let a sequence of SDK-accepted Mode-6 endpoints run arbitrarily far ahead of physical tracking. The final worker jump guard protects command-to-command continuity, but it is not a tracking-lead controller.

The new target must therefore be bounded relative to the latest measured EEF pose in both translation and rotation.

---

## 3. Current source facts to preserve

Re-read current source before editing. At the task-authoring commit, the important contracts are:

### 3.1 Calibration control cadence and jog step

- `dexmani_real/calibration/camera/session.py` runs calibration control with `LoopRate(runtime.keyboard_teleop.control_hz)`.
- The default keyboard/calibration control frequency is `30 Hz`.
- `CalibrationConfig.delta_pos_m = 0.008`.
- `CalibrationConfig.delta_rpy_rad = 0.03`.

Do not change the arm loop frequency or these jog increments in this task.

### 3.2 Arm worker remains the final command-continuity authority

`dexmani_real/robot/arm_worker.py` owns final raw command continuity:

```text
same run generation -> reference = last SDK-accepted target
new run generation  -> reference = latest measured qpos
```

A raw joint jump over `runtime.arm.max_servo_command_jump_rad` is recoverably rejected and establishes an ARMED boundary rather than latching a global fault.

Do not weaken, replace, or bypass this worker guard.

### 3.3 Calibration producer currently has synchronous arm acceptance

Normal calibration motion currently follows:

```text
prepare_joint_command
publish_command(required_safety_state=RUNNING)
wait_command_accepted(wait_for_arm=True, wait_for_hand=False)
```

`state.previous_command` is updated only after acceptance.

Preserve this synchronous ACK behavior in this task.

### 3.4 Existing calibration rejection lifecycle is intentional

The current calibration path:

- converts recoverable rejection into a command-silent `ARMED` boundary;
- resets `previous_command` to current measured qpos;
- sets `blocked_until_release=True`;
- requires all physical Cartesian jog keys to be released before motion may restart.

Preserve this behavior.

### 3.5 Keyboard teleoperation is a behavioral reference, not the edit target

`dexmani_real/teleop/keyboard_session.py` already implements bounded Cartesian virtual-target lead using:

```text
max_position_lead = command_lookahead_frames * position_step
max_rotation_lead = command_lookahead_frames * rotation_step
```

and limits both translation and shortest-arc rotational lead relative to measured EEF pose.

Use its math as a reference for lead bounding, but **do not refactor or behaviorally modify keyboard teleoperation in this task**. The goal is a minimal calibration-specific responsiveness repair.

---

## 4. Hard constraints / non-goals

### 4.1 No hardware execution

Do not run anything that may open, discover, or command real devices, including but not limited to:

```text
examples/calibrate_camera.py
examples/keyboard_teleop.py
examples/collect_teleop.py
physical replay
policy rollout with execution
robot/device discovery
xArm SDK construction through a live path
RealSense startup
XHand startup
homing
```

Do not claim hardware validation.

Allowed validation is source inspection and proven device-free offline work such as pure-function checks, imports already inspected to be side-effect-free, compile checks, and deterministic numerical simulations.

### 4.2 Do not change hardware/controller limits

Do not change:

```text
arm.loop_hz
max_joint_velocity_deg_per_s
max_joint_acceleration_deg_per_s2
max_servo_command_jump_rad
xArm Mode 6 behavior
collision sensitivity
homing parameters
```

The speed regression is a command-generation issue, not a reason to raise hardware limits.

### 4.3 Do not remove synchronous arm ACK

Do not convert calibration normal motion to non-blocking last-published command state. The ACK is part of the chosen design because it gives `previous_command` a strong accepted-endpoint meaning.

### 4.4 Do not restore persistent open-loop Cartesian target state

Do not add the old calibration fields back:

```text
target_pos
target_quat
motion_active
blocked_keys
```

The only persistent arm command anchor needed for normal calibration jog should remain `previous_command`.

### 4.5 Do not redesign IPC, worker lifecycle, or command validation

Do not change coupled-command schemas, arm-state schemas, ticket identity, generation semantics, worker acceptance reporting, safety-state transitions, or persisted dataset semantics.

Do not modify `arm_worker.py`, `runtime/safety.py`, or `control/publication.py` unless current source has independently changed and an actual compatibility issue is discovered. If such a change appears necessary, stop and report why instead of widening scope silently.

### 4.6 Do not alter keyboard/VR/policy behavior

Do not modify the behavior of:

```text
dexmani_real/teleop/keyboard_session.py
dexmani_real/teleop/control_loop/*
dexmani_real/deployment/*
```

unless current-source re-tracing proves a strictly necessary shared compatibility edit. A cleanup/refactor of keyboard target logic is explicitly out of scope.

### 4.7 No persisted-data or schema changes

Do not modify recording/dataset schemas, field meanings, processing behavior, policy data representation, or normalization semantics.

---

## 5. Target control law and invariants

### 5.1 Persistent state ownership

Keep these meanings distinct:

```text
state.current_qpos       = latest validated measured arm qpos
state.previous_command   = last calibration arm endpoint confirmed accepted by SDK
arm_worker.last_target   = final worker-owned last SDK-accepted target
```

The producer may use `previous_command` to generate its next proposal, but `arm_worker.last_target` remains the final safety authority.

### 5.2 Start of a fresh motion epoch

When calibration is `ARMED` and a jog begins, preserve the existing re-anchor:

```python
state.previous_command = state.current_qpos.copy()
begin_motion(...)
```

Therefore the first command after idle/rejection/home starts from measured state.

### 5.3 Desired pose generation

For each moving tick after the lifecycle checks:

```text
measured_pose = FK(state.current_qpos)
anchor_pose   = FK(state.previous_command)
```

Translation target:

```text
desired_position = anchor_pose.position + one current jog increment
```

Rotation target:

```text
desired_rotation = delta_rotation(current jog increment) * anchor_pose.rotation
```

Match the existing left-multiplication convention used by calibration/keyboard jog rotation; do not silently change coordinate semantics.

### 5.4 Workspace clamp

Apply the existing calibration workspace command margin and axis-aligned workspace clamp to the desired position before measured-lead limiting.

Preserve current workspace-boundary warning behavior. The warning should describe actual workspace clipping, not measured-lead limiting.

### 5.5 Measured-state lead bound

Add one calibration tuning parameter:

```python
CalibrationConfig.command_lookahead_frames: int = 3
```

Validate it as a positive integer in `CalibrationConfig.__post_init__`.

Derive:

```python
max_position_lead_m = (
    calibration_config.command_lookahead_frames
    * calibration_config.delta_pos_m
)
max_rotation_lead_rad = (
    calibration_config.command_lookahead_frames
    * calibration_config.delta_rpy_rad
)
```

Position lead:

```text
lead = workspace_clipped_target_position - measured_position
if ||lead|| > max_position_lead:
    target_position = measured_position + lead / ||lead|| * max_position_lead
```

Rotation lead must use the shortest relative rotation, matching the existing keyboard semantics:

```text
R_rel = R_target * inverse(R_measured)
rotvec = log(R_rel)
if ||rotvec|| > max_rotation_lead:
    R_target = Exp(rotvec / ||rotvec|| * max_rotation_lead) * R_measured
```

Use the repository's existing quaternion convention (`wxyz`) and `scipy.spatial.transform.Rotation(..., scalar_first=True)` consistently.

The bounded target is the pose passed to IK.

### 5.6 IK and publication path remains unchanged after target construction

After the bounded pose is constructed, preserve the existing sequence:

```text
planner.solve_teleop_ik(
    bounded_target_pose,
    state.current_qpos,
    state.previous_command,
)

nearest_equivalent_qpos(..., state.previous_command)
check_worker_arm_target(... previous_target = state.previous_command ...)
prepare_joint_command(...)
publish_command(... required RUNNING ...)
wait_command_accepted(... wait_for_arm=True ...)
```

Only after successful acceptance:

```python
state.previous_command = candidate.arm_qpos.copy()
```

### 5.7 No lead accumulation after rejection or idle

On ordinary rejection, worker-created ARMED boundary, idle release, or home completion, preserve the existing measured re-anchor. The next motion epoch must not resume from an old ahead-of-measured command anchor.

### 5.8 Single-step precision

If a fresh motion epoch starts with `previous_command == current_qpos`, and the target is not workspace-clipped, the first translation target must be one configured translation increment from the measured pose. Likewise, the first rotation target must be one configured rotation increment.

The new mechanism must not turn one short key press into a 3-step jump.

### 5.9 Sustained-motion responsiveness

With an idealized stationary measured pose and successively accepted endpoints, translation lead should build approximately:

```text
8 mm -> 16 mm -> 24 mm -> capped at 24 mm
```

for the default parameters. Rotation lead should analogously build:

```text
0.03 rad -> 0.06 rad -> 0.09 rad -> capped at 0.09 rad
```

In real motion, measured feedback advances, so the bounded lead should remain dynamic but never exceed the configured cap (up to numerical tolerance).

---

## 6. Required implementation

Use the smallest coherent edit that implements the above behavior.

### 6.1 `dexmani_real/calibration/camera/solver.py`

Extend `CalibrationConfig` with:

```python
command_lookahead_frames: int = 3
```

Add validation:

```text
must be an int (not bool) and >= 1
```

Do not move calibration configuration ownership into general runtime YAML as part of this task.

### 6.2 Lead-bounding math

Prefer a small pure helper for the measured-state lead limit if it makes the control code materially clearer and offline validation easier.

A suitable location is `dexmani_real/control/jog.py`, because that module already owns pure Cartesian jog semantics. A suitable conceptual API is:

```python
limit_cartesian_pose_lead(
    measured_pos,
    measured_quat_wxyz,
    target_pos,
    target_quat_wxyz,
    *,
    max_position_lead_m,
    max_rotation_lead_rad,
) -> tuple[np.ndarray, np.ndarray]
```

Naming may follow repository style, but semantics must match Section 5.5.

Keep it pure:

- no shared memory;
- no planner calls;
- no device IO;
- no lifecycle state;
- no logging side effects;
- no hidden mutable state.

Do not over-engineer a new class or generic motion framework.

If current source makes a local private helper in calibration materially smaller/clearer than a shared pure helper, that is acceptable, but do not duplicate large chunks of unrelated keyboard-session logic.

### 6.3 `dexmani_real/calibration/camera/motion.py`

Change only the moving target-generation portion of `run_calibration_motion_tick()`.

Replace the current one-step measured-relative target source:

```text
measured_pose + jog_delta
```

with:

```text
FK(previous_command) + jog_delta
        -> workspace clamp
        -> measured-pose lead bound
```

Do not add persistent Cartesian target fields to `CalibrationLoopState`.

The intended high-level moving branch is:

```python
# Existing lifecycle / release-latch logic remains.

if safety_state == ARMED:
    state.previous_command = state.current_qpos.copy()
    begin_motion(...)

measured_pose = FK(state.current_qpos)
anchor_pose = FK(state.previous_command)

# Advance one operator step from the accepted command anchor.
desired_pos = anchor_pose.p + dx
desired_quat = apply_delta_rotation(anchor_pose.q, drpy)

# Existing workspace margin/clipping.
workspace_target_pos = clip(desired_pos, command_low, command_high)

# New measured-state lead cap.
proposed_pos, proposed_quat = limit_pose_lead(
    measured_pose,
    workspace_target_pos,
    desired_quat,
    max_position_lead = frames * delta_pos,
    max_rotation_lead = frames * delta_rpy,
)

# Existing IK / canonicalization / jump guard / SafetyGate / publish / ACK.
...

# Commit only after successful SDK acceptance.
state.previous_command = accepted_arm_endpoint.copy()
```

Preserve the existing status line; `target=` should report the actual bounded position target sent into IK, not the pre-limit desired target.

### 6.4 Keep current failure behavior

Do not change the semantics of:

```text
_reject_calibration_motion
blocked_until_release
any_jog_key_held
worker-created ARMED boundary detection
fatal vs recoverable rejection
quit hold
return-home behavior
```

An IK failure, ordinary recoverable command rejection, publication failure, or ACK rejection must still quiesce motion and require full physical jog-key release before restart when the runtime is otherwise healthy.

---

## 7. Numerical and semantic edge cases

The implementation must handle the following without introducing special-case state machines:

### 7.1 Zero rotation increment

If `drpy == 0`, the desired orientation should be the accepted anchor orientation before measured-lead limiting.

### 7.2 Target already inside lead bound

The limiter must leave position/orientation unchanged apart from harmless numerical normalization.

### 7.3 Exact lead boundary

Do not perturb a target merely because it is numerically equal to the cap. Use stable comparisons/tolerances consistent with normal floating-point code in the repository.

### 7.4 Workspace boundary plus lead bound

Workspace clipping happens first. Because measured position is already a validated workspace observation and the workspace is axis-aligned/convex, scaling the clipped target back toward measured position must remain inside the workspace.

### 7.5 Opposite held keys

Opposite keys can produce zero net `dx/drpy` while physical jog keys are still held. Do not change the existing full-release rejection latch; it intentionally distinguishes physical held keys from net motion.

### 7.6 Quaternion sign

Quaternion `q` and `-q` represent the same orientation. Use `Rotation`/shortest-relative-rotation semantics so the lead limiter does not create a false ~2π rotation because of sign choice.

### 7.7 Accepted command may lead measured feedback

This is expected in Mode 6. Do not treat `previous_command != current_qpos` as a tracking fault. The measured-state lead cap exists specifically to make this lag bounded at the Cartesian target-generation layer.

---

## 8. Expected edit surface

Expected files are approximately:

```text
dexmani_real/calibration/camera/solver.py
dexmani_real/calibration/camera/motion.py
```

and, only if using the preferred pure helper:

```text
dexmani_real/control/jog.py
```

Do not modify `examples/calibrate_camera.py`; it is only the CLI entry point and is not the source of the speed regression.

Do not change README/permanent documentation unless current source has independently evolved such that a stable user-visible contract actually requires documentation. This task file itself is temporary and should be deleted on successful completion.

---

## 9. Offline validation

### 9.1 Mandatory repository checks

After the focused edit, run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Do not run hardware examples.

### 9.2 Pure lead-limiter checks

If the lead bound is factored into a pure helper, run deterministic device-free numerical checks covering at least:

1. **Position within bound**: target lead below cap remains unchanged.
2. **Position above bound**: target lead is clipped to exactly the configured norm.
3. **Rotation within bound**: shortest relative rotation below cap remains unchanged.
4. **Rotation above bound**: relative rotvec norm is clipped to the configured cap.
5. **Quaternion sign equivalence**: equivalent `q`/`-q` inputs do not create a large spurious rotation.
6. **Combined translation + rotation**: both bounds apply independently.

Use only pure modules/functions whose import path has been inspected and proven not to acquire hardware.

### 9.3 Deterministic lookahead progression check

Perform an offline Cartesian surrogate simulation of the intended accepted-anchor law. With:

```text
measured position fixed at 0
position step = 0.008 m along +X
lookahead_frames = 3
accepted anchor updated to each bounded target
```

verify target lead progresses:

```text
0.008, 0.016, 0.024, 0.024, ... m
```

Do the analogous rotation check:

```text
0.03, 0.06, 0.09, 0.09, ... rad
```

This is a control-law simulation only; it is not hardware validation.

### 9.4 Source-level invariant review

Inspect the final diff and verify all of the following directly from source:

- no persistent `target_pos` / `target_quat` was added to calibration state;
- `previous_command` is still committed only after successful arm acceptance;
- a fresh motion epoch re-anchors `previous_command` to measured qpos;
- idle/rejection/home still remove stale command lead;
- worker jump guard and safety lifecycle were not weakened;
- hardware speed/acceleration/loop parameters were not changed;
- keyboard/VR/policy behavior was not changed;
- no persisted data/schema semantics changed.

---

## 10. Acceptance criteria

The task is complete only if all of the following are true.

### A. Control behavior

- Calibration no longer generates every held-key target as `measured_pose + one step`.
- The next desired target is advanced from `FK(previous_command)`, where `previous_command` is the last producer command confirmed accepted by the arm worker/SDK.
- Translation and rotation target lead are bounded relative to latest measured EEF pose.
- Default lead cap is `3 * delta_pos_m` and `3 * delta_rpy_rad`.
- A fresh single-step jog still produces only one jog increment, not three increments.
- Sustained idealized motion can build to the configured lead cap instead of remaining at one-step lead.

### B. Safety/correctness

- `arm_worker` remains final command-continuity authority.
- `wait_command_accepted()` remains in the normal calibration command path.
- `previous_command` advances only after successful arm acceptance.
- Recoverable rejection still closes the motion epoch and requires full jog-key release.
- No unbounded Cartesian target state is reintroduced.
- No hardware limits, IPC schemas, persisted schemas, or dataset semantics are changed.

### C. Scope/quality

- The implementation is small and explicit; no generic control framework is introduced.
- Keyboard/VR/policy behavior remains unchanged.
- Pure math is separated enough to be validated offline without hardware if doing so reduces complexity.
- Final code/comments describe the current mechanism rather than the historical incident.

### D. Validation

- Mandatory low-cost checks pass.
- Pure numerical lead-bound checks pass.
- Deterministic 3-frame lookahead progression simulation passes.
- No hardware-affecting program was executed.

---

## 11. Handoff requirements

Before final handoff:

1. Inspect `git diff` for the complete focused change.
2. Inspect `git status --short` and preserve unrelated user changes.
3. If all implementation and offline checks pass, delete `CODEX_TASK.md` and rerun:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

4. Do **not** commit or push unless the parent/user separately instructs you to do so.
5. Report concisely:
   - changed files and mechanism;
   - preserved safety/lifecycle invariants;
   - exact offline validation performed and results;
   - explicit statement that no hardware validation was performed;
   - any residual tuning question (principally whether real-hardware feel later prefers lookahead `2`, `3`, or `4`).

The implementation default for this task is **3**. Do not preemptively tune it higher without hardware evidence.
