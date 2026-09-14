# CODEX TASK — Arm Command-Jump Recovery + Manual Jog Stabilization (Offline-Only)

> This is a **temporary implementation task file** for Codex. Read `AGENTS.md` first and obey it throughout the task. This repository controls physical robotics hardware. **Do not connect to xArm/XHand/RealSense or execute any hardware-affecting program in this task.** Work from source inspection and offline validation only.
>
> The task file was authored against `main` at commit `38715b4ab71da6d6583daa53d37e8f0fd5dbc6af`. Before editing, inspect the actual current `HEAD` and `git status --short`. If `HEAD` has advanced, re-trace the relevant code paths and adapt the implementation to current source; do not blindly apply stale line-level assumptions.
>
> If implementation and all offline acceptance checks succeed, **delete `CODEX_TASK.md` before final handoff** and re-run the final low-cost checks. If the task is incomplete or validation fails, keep this file so the work can be resumed.

---

## 1. Goal

Fix the repository-wide arm command-jump failure mode without over-engineering the runtime and without changing persisted dataset semantics.

The concrete failure already observed in camera calibration is:

```text
OnlineIK rejects / large target discontinuity
    -> local producer state and worker command reference diverge
    -> next arm endpoint is safe relative to producer state but > worker 20° raw-jump limit
    -> arm_worker raises RuntimeError
    -> sticky error_state
    -> global FAULT
```

Related reference drift exists in keyboard / VR / policy / replay because producers may reason about a last-published endpoint while `arm_worker` owns the true last SDK-accepted endpoint.

The target architecture for this task is deliberately small:

```text
Producer owns intent / proposal generation
        |
        v
latest-wins coupled command IPC
        |
        v
arm_worker owns final command continuity
        |
        +-- same generation: reference = last SDK-accepted target
        +-- new generation:  reference = latest measured qpos
        |
        +-- command jump > limit: recoverable command rejection
        |      -> atomically invalidate the rejected current ticket
        |      -> RUNNING -> ARMED when applicable
        |      -> keep worker alive
        |      -> do NOT latch error_state
        |
        +-- malformed target / joint limit / SDK/controller failure:
               fail-fast as today
```

At the same time, remove the camera-calibration-specific open-loop Cartesian target accumulation that makes the original failure easy to trigger.

The implementation must preserve normal VR / policy / replay behavior as much as possible. This is a **control-correctness repair**, not a global control-stack redesign.

---

## 2. Hard constraints / non-negotiable scope

### 2.1 Absolutely no hardware execution

Do **not** run anything that can open or command real devices, including but not limited to:

```text
examples/calibrate_camera.py
real keyboard teleop entry points
VR teleop with workers
physical replay
policy rollout with execute enabled
homing
robot/device discovery that opens SDK connections
RealSense startup
XArmAPI construction through a live entry point
```

Do not claim hardware validation in the handoff.

Allowed validation is pure/offline only: imports that do not acquire devices, pure functions, `RuntimeChannels` shared-memory construction, fake/mock arm objects, deterministic state-machine simulations, compile checks, diff checks, and existing offline utilities whose path has been inspected and proven device-free.

### 2.2 Do not change persisted data semantics

This task must **not** change:

```text
dexmani_real/recording/storage/schema.py
EPISODE_SCHEMA_VERSION
/action_arm_joint_sent meaning
action_hand_joint meaning
processed HDF5 schema
dataset processing semantics
training action representation
normalization semantics
recorded frame cadence
```

Do not rename or reinterpret `action_arm_joint_sent`. It currently means the producer-submitted arm target recorded by the control path; it is not a guaranteed SDK-accepted trajectory. Keep that historical contract unchanged.

Do not modify `dexmani_real/dataset/*` as part of this task.

### 2.3 Do not redesign IPC

Do **not** add any of the following in this task:

```text
reference_action_id CAS protocol
command_ref_qpos fields in ARM_STATE_DTYPE
last_rejected_action_id fields in ARM_STATE_DTYPE
new command-result IPC channels
new request/response queues
FIFO replacement for latest-wins command transport
```

Keep `COUPLED_COMMAND_DTYPE` and `ARM_STATE_DTYPE` unchanged unless current source has independently changed before implementation and a strictly necessary compatibility adjustment is discovered. Any schema change requires explicit re-review before proceeding.

### 2.4 Do not change physical controller behavior

Keep unchanged:

```text
xArm Mode 6
runtime arm loop frequency
max joint velocity
max joint acceleration
20° max_servo_command_jump_rad default
SafetyGate measured-state geometry semantics
XHand worker behavior
```

Do not switch to Mode 1, Pink, a new interpolator, a new servo frequency, or a tracking-error fault.

Mode 6 is online trajectory planning: measured position can legitimately lag the accepted command endpoint. Do not treat `measured != commanded` as a fault.

### 2.5 Preserve normal VR / policy trajectory generation

Do not broadly refactor or replace:

```text
TeleopController.prev_qpos_cmd
VR mapper
EMA pose smoothing
teleop 8° endpoint shaping
policy previous_arm_command_qpos
policy action decoding
policy model-spike guard
replay trajectory timing
```

Only add the minimal lifecycle handling needed when the arm worker establishes a recoverable ARMED boundary.

---

## 3. Current source facts that must be preserved

Re-read the current source before editing, but the implementation should be based on these current contracts.

### 3.1 `arm_worker` already owns the real command reference

Current `dexmani_real/robot/arm_worker.py` computes:

```python
jump_reference = (
    st.last_target
    if command_generation == st.last_command_generation
    else st.last_measured_qpos
)
```

`st.last_target` advances only after `st.arm.servo(target)` returns SDK code `0`.

This is the correct final command-continuity authority. Keep it.

The worker currently validates with `check_worker_arm_target(...)`, checks that the ticket still permits execution, and raises `RuntimeError` for any validation issue. That makes a command jump a global runtime fault. This task changes only the **jump** classification.

### 3.2 Worker raw jump check is intentionally raw

`dexmani_real/robot/command_validation.py` currently rejects:

```python
abs(target - previous_target_qpos_rad) > max_command_jump_rad
```

Do not change the final worker guard to wrapped/equivalent delta semantics. The worker final fence must continue to validate the exact qpos representation about to cross the SDK boundary.

### 3.3 `revoke_motion()` is not safe after a separate stale-ticket check

`runtime/safety.py` protects lifecycle state, generation, and latest command ownership with `motion_lock`.

A naive worker implementation such as:

```text
check ticket A is current
unlock
new ticket B publishes
worker calls revoke_motion()
```

can incorrectly revoke B.

The worker rejection path therefore requires one **atomic conditional rejection primitive** under `motion_lock`.

### 3.4 Camera calibration currently accumulates a virtual target

`CalibrationLoopState` currently carries:

```text
current_qpos
previous_command
target_pos
target_quat
motion_active
blocked_keys
```

Held jog keys mutate `target_pos/target_quat` every control tick, independent of physical tracking. On Mode 6 tracking lag, the virtual target can run ahead of measured pose and push IK across branch/jump boundaries.

IK failure currently re-anchors Cartesian target to measured pose but leaves the same RUNNING epoch and command reference in place. This is the direct bug to remove.

### 3.5 Camera calibration has a useful synchronous ACK property

Calibration publishes one arm command then calls `wait_command_accepted()` before advancing `previous_command`.

Therefore, within an active calibration motion epoch, `previous_command` can legitimately remain the last arm command known to have crossed the arm SDK boundary.

Preserve that useful property.

### 3.6 Keyboard is less strict

Keyboard publishes non-blocking while motion is held. Its `previous_command` is last-published, not guaranteed last SDK-accepted. Do not attempt a large architecture migration in this task. Instead:

- narrow OnlineIK early jump admission to the worker limit;
- canonicalize/check the final endpoint before publication;
- make rejection close the motion epoch and require full jog-key release;
- rely on the worker final guard for races / skipped publications.

### 3.7 VR and policy already have their own continuity shaping

VR has tighter per-tick arm shaping (default about 8°) and policy has an explicit arm jump guard. Their normal action generation is not the main failure trigger and is data-distribution-sensitive.

Do not rewrite those paths. Only make sure an unexpected worker-established ARMED boundary ends or pauses the active run cleanly instead of being misclassified as a hardware fault.

### 3.8 Replay already distinguishes REJECTED from FAULT

`EpisodeReplayer` has `ReplayStatus.REJECTED` and `_reject(...)` for valid-runtime safety rejections. Reuse that concept if a worker-established ARMED boundary currently falls into a fault path.

Do not change replay to per-frame ACK pacing in this task.

---

## 4. Target invariants

The final implementation must satisfy all of the following.

### 4.1 Single final arm-continuity authority

The final safety decision is always made by `arm_worker` using:

```text
same generation -> last SDK-accepted arm target
new generation  -> latest measured arm qpos
```

No producer-side check may weaken or replace this worker check.

### 4.2 Jump rejection is recoverable

For the exact command-validation reason `command jump limit violation`:

```text
- do not call xArm SDK
- do not update st.last_target
- do not update st.last_cmd
- do not set error_state
- atomically invalidate the rejected ticket
- leave lifecycle at ARMED (RUNNING -> ARMED; ARMED may remain ARMED with generation invalidation)
- keep arm worker alive
```

### 4.3 Other worker validation failures remain fail-fast

A current executable command that reaches the worker with:

```text
non-finite target
joint limit violation
```

must still raise into the existing worker fail-fast boundary.

SDK non-zero return, controller error, disconnect, or invalid hardware feedback also remain fatal as today.

### 4.4 Stale rejected command must never revoke a newer command

If ticket A is rejected by the jump guard, but ticket B became current before the worker establishes the rejection boundary, A must become a no-op. B must remain valid.

This invariant is mandatory and is the main reason for the new atomic safety helper.

### 4.5 Calibration jog is measured-relative, not open-loop cumulative

For each moving calibration tick:

```text
proposed_pose_k = FK(measured_qpos_k) + one jog increment
```

not:

```text
proposed_pose_k = previous_virtual_target + one jog increment
```

No cross-tick `target_pos/target_quat` accumulation should remain in calibration motion state.

### 4.6 Rejection requires true jog-key release

After calibration/keyboard motion rejection, motion may not restart until **all physical Cartesian jog keys** are released.

Do not use net `moving == False` as the release test: opposite keys such as `W+S` can cancel to zero while jog keys are still physically held.

### 4.7 Old data contract remains unchanged

No raw/processed dataset schema or recorded field semantics may change.

---

## 5. Required implementation

Use the smallest coherent implementation that satisfies the contracts below. Naming may be adjusted to repository style, but do not change semantics.

### 5.1 `runtime/safety.py` — add one atomic conditional rejection primitive

Add a narrow public helper, preferably named along the lines of:

```python
reject_coupled_command_if_current(
    shared,
    *,
    ticket: CoupledCommandTicket,
) -> bool
```

Required semantics:

1. Acquire `shared.motion_lock` once.
2. Verify the ticket still owns the latest coupled-command slot using the existing locked ticket identity logic.
3. Verify runtime execution conditions that matter at the SDK boundary:
   - runtime still running;
   - no sticky error;
   - no e-stop;
   - ticket not expired.
4. If any check fails, return `False` and change nothing.
5. If the ticket is still current/executable, invalidate it atomically by establishing an ARMED motion boundary under the same lock:
   - if current state is RUNNING, transition RUNNING -> ARMED;
   - if current state is already ARMED and the ticket is current, retain ARMED while advancing generation / invalidating the ticket using the existing revocation machinery;
   - never transition a faulted/stopped state back to ARMED.
6. Clear `run_started_monotonic_ns` through the canonical revocation path.
7. Return `True` only when this exact ticket was successfully invalidated.
8. Do not perform hardware IO while holding the lock.
9. Do not set `error_state` here.

Do **not** implement this as:

```python
if coupled_command_ticket_allows_execution(...):
    revoke_motion(...)
```

because that contains a TOCTOU race.

Keep existing `cancel_coupled_command_if_current()` semantics unchanged; ACK timeout cancellation deliberately invalidates generation without necessarily changing lifecycle state and is a separate contract.

### 5.2 `robot/command_validation.py` — remove magic-string duplication only

Introduce a small constant for the existing jump rejection string, for example:

```python
ARM_COMMAND_JUMP_REJECTION = "command jump limit violation"
```

Return that constant from `check_worker_arm_target()`.

Do not introduce a new error hierarchy or enum unless current code has independently evolved and a typed result is already canonical.

Do not change numerical validation semantics.

### 5.3 `robot/arm_worker.py` — recover only command-jump rejection

Update `_handle_servo_command()` while preserving current latest-wins and SDK ordering.

Required control flow:

```text
read current command/ticket
mark ring sequence processed (existing behavior)
compute command_generation
compute actual worker jump_reference
compute target
run check_worker_arm_target()

if issue == command-jump rejection:
    atomically reject this exact ticket with the new safety helper
    if helper succeeds:
        log one warning with useful diagnostics
    return regardless (never execute a stale rejected command)

for every other validation issue:
    preserve current stale-ticket behavior:
        if ticket no longer allows execution -> return silently
        if current executable ticket -> raise RuntimeError as today

if no issue:
    preserve current final ticket execution check
    call SDK
    on SDK code 0 only:
        update last_target / generation / last_cmd
```

Important details:

- A superseded jump-rejected command must not fault and must not revoke the newer ticket.
- A current jump-rejected command must not cross the SDK boundary.
- Do not update `st.last_target` on rejection.
- Do not change the raw worker 20° comparison.
- Do not add 2π canonicalization inside the worker in this task.
- Do not change Mode 6 or driver parameters.

Warning log should be concise but diagnostically useful. Prefer fields equivalent to:

```text
action_id
generation
joint index (1-based)
raw max delta in degrees
configured limit in degrees
```

Do not spam normal-path logs.

### 5.4 `control/jog.py` — make jog-key ownership explicit

This module already owns the mapping from held keys to Cartesian jog increments. Put the physical jog-key vocabulary here rather than duplicating it in calibration and keyboard.

Define one shared immutable key set / tuple containing exactly the keys consumed by `compute_cartesian_jog_delta()`:

```text
w s a d up down left right i k j l
```

Provide a tiny pure helper if useful, e.g. testing whether an `active_keys` iterable contains any Cartesian jog key.

The helper must distinguish:

```text
W + S held -> jog keys are still held even though net dx == 0
W + SPACE  -> jog key is still held
SPACE only -> no jog key held
```

Do not expand keyboard mappings or user-facing controls.

---

## 6. Camera calibration refactor

Files:

```text
dexmani_real/calibration/camera/motion.py
dexmani_real/calibration/camera/session.py
```

This is the main producer-side correctness fix.

### 6.1 Simplify `CalibrationLoopState`

Remove cross-tick state that exists only for the open-loop virtual Cartesian target:

```text
target_pos
target_quat
motion_active
blocked_keys tuple
```

Keep:

```text
current_qpos
previous_command
blocked_until_release: bool
samples / home / diagnostics fields
```

`previous_command` remains valuable in calibration because it advances only after synchronous arm SDK acceptance.

On construction, initialize:

```python
previous_command = current_qpos.copy()
blocked_until_release = False
```

If removing the now-unused `planner` argument from `from_arm_state()` is a small and clean change, do so and update the one caller. Do not preserve a meaningless argument solely to minimize line count.

### 6.2 Use authoritative shared lifecycle, not local `motion_active`

Do not maintain a second local RUNNING/ARMED truth.

For each tick, inspect `shared.safety_state`.

Expected states during normal calibration motion are ARMED or RUNNING. FAULT/e-stop/sticky error remain terminal through existing session handling.

### 6.3 Release latch

At the start of each motion tick:

```text
active_keys = keys.pressed_keys()
jog_key_held = any shared Cartesian jog key in active_keys
```

If `blocked_until_release`:

```text
jog_key_held == True  -> return
jog_key_held == False -> clear block, previous_command=current_qpos, return
```

The release tick itself must not immediately start motion. A fresh press is required.

Do not use net `moving` to clear the latch.

### 6.4 Idle behavior

Compute `dx, drpy` and the effective `moving` value as today.

If there is no effective jog motion:

- if shared safety state is RUNNING, revoke to ARMED;
- set `previous_command = current_qpos.copy()`;
- keep existing low-rate status output based on measured pose;
- return.

This also allows measured state to continue settling after the last Mode-6 endpoint without inventing a tracking-error fault.

### 6.5 New motion epoch

When effective jog motion is requested:

- if state is ARMED:
  - set `previous_command = current_qpos.copy()`;
  - call `begin_motion(shared)`;
  - failure to enter the intended lifecycle remains a calibration fault.
- if state is RUNNING, continue the current epoch.
- if state is neither ARMED nor RUNNING, do not try to repair it locally; use existing terminal/session fault semantics.

### 6.6 Measured-relative one-step target

Every moving tick must start from the latest measured arm state:

```python
measured_pose = planner.kin.compute_eef_pose_world(state.current_qpos)
```

Position proposal:

```python
desired_pos = measured_pose.p + dx
proposed_pos = np.clip(desired_pos, command_low, command_high)
```

Keep existing workspace-boundary diagnostics, but report clipping of this local proposal.

Rotation proposal:

```python
proposed_quat = measured_pose.q.copy()
if any(drpy):
    delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
    proposed_quat = quat_multiply(delta_quat, proposed_quat)
```

Do not persist either proposed pose across ticks.

### 6.7 IK

Call:

```python
planner.solve_teleop_ik(
    Pose(p=proposed_pos, q=proposed_quat),
    state.current_qpos,
    state.previous_command,
)
```

On IK failure:

- keep throttled warning behavior;
- establish a normal quiescent boundary (`RUNNING -> ARMED` if needed);
- `previous_command = current_qpos.copy()`;
- set `blocked_until_release = True`;
- return;
- do not mutate a persistent Cartesian target because none should exist.

A small helper such as `_quiesce_calibration_motion(...)` is encouraged if it removes repeated lifecycle/reset logic. Keep it narrow.

### 6.8 Final command canonicalization and exact early worker-equivalent guard

OnlineIK currently canonicalizes primarily around measured state. Before publication, canonicalize the final IK result to the nearest limit-valid equivalent around `state.previous_command`:

```python
q_cmd = planner.ik_mgr.nearest_equivalent_qpos(
    np.asarray(ik_result.qpos, dtype=np.float64),
    state.previous_command,
)
```

Then call the existing worker arm validator using the exact configured worker limit and `state.previous_command` as the reference.

This final producer-side guard is important because:

- OnlineIK candidate filtering may use wrapped/equivalent deltas;
- nullspace/post-processing can modify the final qpos;
- the worker uses a raw qpos difference.

If the exact early guard reports the command-jump rejection:

- quiesce to ARMED;
- block until full jog-key release;
- do not publish.

If it reports `non-finite target` or `joint limit violation`, treat that as a software invariant failure and use the existing calibration fault path; do not silently recover from malformed output.

### 6.9 Publication / ACK is transactional

Continue using:

```text
prepare_joint_command
publish_command(required RUNNING)
wait_command_accepted(wait_for_arm=True)
```

On normal `PreparedCommand` / publication / ACK rejection:

- if runtime is otherwise healthy, quiesce to ARMED and block until full jog-key release;
- if `prepared.fatal`, sticky error, FAULT, or e-stop is present, use existing terminal calibration fault behavior.

Remember that `wait_command_accepted()` may cancel a ticket by advancing generation while lifecycle remains RUNNING. The calibration rejection path must explicitly revoke/quiesce afterward; do not rely on generation cancellation alone.

Only after successful arm acceptance:

```python
state.previous_command = candidate.arm_qpos.copy()
```

“No ACK, no commit.”

### 6.10 Home / quit behavior

Update home re-anchor logic to match the simplified state:

```text
current_qpos = fresh measured feedback
previous_command = current_qpos
blocked_until_release = False
```

Remove old Cartesian-target and local-motion-active resets.

Keep the existing measured quit hold protocol unless a directly related compile/runtime inconsistency appears. Do not redesign quit behavior.

### 6.11 Calibration OnlineIK profile

In `calibration/camera/session.py`, set the calibration-specific OnlineIK jump envelope to the worker hard limit on all seven joints:

```python
jump_deg = float(np.rad2deg(runtime.arm.max_servo_command_jump_rad))
OnlineIKConfig(
    max_ik_jump_deg=(jump_deg,) * 7,
    max_pose_error_pos_m=...,
    max_pose_error_rot_rad=...,
)
```

This is an early search-space constraint, not the final safety guarantee.

Do not globally change `OnlineIKConfig` defaults.

---

## 7. Keyboard teleop stabilization

File:

```text
dexmani_real/teleop/keyboard_session.py
```

Keep normal keyboard trajectory behavior intact. Do not replace its bounded virtual lookahead with calibration-style measured-relative one-step motion.

### 7.1 OnlineIK profile

Set keyboard-specific OnlineIK `max_ik_jump_deg` to the worker hard limit on all joints, analogous to calibration.

Keep existing pose-error limits and lookahead parameters unchanged.

### 7.2 Full-release latch

Replace `blocked_keys: tuple | None` with `blocked_until_release: bool`.

Use the shared Cartesian jog-key vocabulary from `control/jog.py`.

While blocked:

```text
any jog key still held -> remain blocked
all jog keys released  -> clear block, rebuild anchors from current measured qpos, return/continue this release tick
```

`W+S`, `W+SPACE`, or any other combination containing a physical jog key must not clear the latch.

### 7.3 Final qpos canonicalization / early raw guard

In `_publish_keyboard_target()` after successful IK:

1. canonicalize the final IK qpos to the nearest equivalent around `previous_command_qpos_rad`;
2. run `check_worker_arm_target()` using:
   - that canonicalized qpos;
   - `previous_command_qpos_rad`;
   - runtime arm joint limits;
   - `runtime.arm.max_servo_command_jump_rad`;
3. map a jump failure to the existing ordinary keyboard safety-rejected result;
4. malformed/non-finite/joint-limit results should not be converted into a hardware SDK call.

Do not remove the worker final guard; keyboard `previous_command` is still last-published and can race the true worker reference.

### 7.4 Rejection closes the motion epoch

Current keyboard IK/safety rejection blocks keys but can leave the same RUNNING epoch active. Change this.

On IK rejection or ordinary safety rejection while runtime is otherwise healthy:

```text
if RUNNING -> revoke to ARMED
motion_active = False
release debounce state = reset
last_motion_action_id = 0
release ACK timer = reset
rebuild previous_command / target_pos / target_quat from current measured qpos
blocked_until_release = True
```

Then require full jog-key release before a fresh press can call `begin_motion()` again.

Use a narrow helper if it avoids duplicating this reset sequence.

Preserve the existing normal release behavior:

- debounce key release;
- wait boundedly for the final normal published action to cross the arm SDK boundary;
- then revoke to ARMED;
- re-anchor from measured feedback.

Do not turn every moving tick into a blocking ACK path.

---

## 8. VR teleop lifecycle handling — minimal only

Primary file to inspect:

```text
dexmani_real/teleop/loop.py
```

Do **not** broadly modify `teleop/control_loop/grid.py` or `TeleopController` unless a small compile-consistency change is strictly required.

The worker may now establish an ARMED boundary on a final command-jump rejection without setting `error_state`.

Add the smallest lifecycle handling so that if:

```text
teleop_active == True
and shared.safety_state is unexpectedly ARMED
and this was not just caused by the operator event being processed in the same loop
```

then the current teleop run is terminated safely as a recoverable control rejection.

Preferred behavior:

```text
- enter a command-silent pause/rejection boundary using existing helpers
- teleop_active = False
- if recording_active:
      stop_recording(save=False, reason="arm_command_rejected" or equally clear stable reason)
      recording_active = False
      shared.is_recording = False
- clear/re-anchor controller temporal reference through existing pause machinery
- do not set error_state
- require a fresh B to start a new run; C should not resume this rejection boundary
```

Do not change normal VR proposals, 8° arm shaping, EMA, hand ramping, recording action values, or grid cadence.

If current loop behavior already handles this exact case correctly after source re-trace, do not add duplicate logic. Document that finding in the final handoff and leave the file unchanged.

---

## 9. Learned-policy executor — prefer no production change

Inspect:

```text
dexmani_real/deployment/executor.py
```

The existing executor already detects motion being revoked outside the normal stop request and terminates the current episode. Re-trace this path against the new worker behavior.

Preferred outcome for this task:

- no changes to policy action generation;
- no changes to policy `previous_arm_command_qpos` semantics;
- no changes to policy recording schema;
- no changes to policy model-spike guard;
- unexpected worker ARMED boundary ends the current physical episode;
- global `error_state` remains false unless there is an independent actual fault.

The existing stop reason may be historically named `hardware_fault`. If that label would become materially false for recoverable command rejection and can be corrected locally without changing dataset schema or downstream interfaces, use a narrow neutral stop reason. Otherwise do not broaden this task into a stop-reason migration; report the residual naming issue in handoff.

Do not modify this file just for architectural symmetry.

---

## 10. Replay — classify worker rejection, do not change timing

Inspect:

```text
dexmani_real/replay/replayer.py
```

Do not change:

```text
replay_hz
target schedule
send mask
trajectory representation
recorded action semantics
per-frame pacing policy
```

If a current physical replay is RUNNING and the arm worker establishes ARMED because of a recoverable command-jump rejection, the replay should terminate as:

```text
ReplayStatus.REJECTED
```

not as `FAULT`, provided:

```text
error_state == False
estop_request == False
controller feedback remains healthy
```

Use the existing `_reject(...)` / terminal quiescence concepts.

If current replay code already reaches REJECTED under this state transition after re-tracing the exact loop, do not add duplicate logic.

Do not add per-frame arm ACK pacing in this task.

---

## 11. Files expected in scope

Expected production scope is approximately:

```text
dexmani_real/runtime/safety.py
dexmani_real/robot/command_validation.py
dexmani_real/robot/arm_worker.py
dexmani_real/control/jog.py
dexmani_real/calibration/camera/motion.py
dexmani_real/calibration/camera/session.py
dexmani_real/teleop/keyboard_session.py
dexmani_real/teleop/loop.py          # only if needed after re-trace
dexmani_real/replay/replayer.py      # only if needed after re-trace
```

`deployment/executor.py` should preferably remain unchanged unless source re-trace proves a small lifecycle classification fix is necessary.

The following are **out of scope and should not change**:

```text
dexmani_real/ipc/schema.py
dexmani_real/ipc/channels.py         # unless only comments/docs must reflect a changed source contract; prefer no change
dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py
dexmani_real/recording/**
dexmani_real/dataset/**
policy/model code
camera/pointcloud data pipeline
XHand worker
```

If implementation appears to require a persisted-data or IPC-schema change, stop and re-evaluate the design rather than silently expanding scope.

---

## 12. Offline validation strategy — no hardware required

This task must be accepted entirely offline.

Do not introduce a new test-framework dependency merely for this task. The repository currently does not define a permanent generic test harness. Prefer:

```text
- existing pure functions
- small `python - <<'PY'` assertion scripts
- `unittest.mock`
- fake arm objects
- `RuntimeChannels.create()` with unique temporary prefixes
- temporary helper scripts outside the repository or untracked during development, removed before handoff
```

If a current repository test harness exists by the time this task runs, use it. Do not restore historical test infrastructure just to satisfy this document.

### 12.1 Core safety atomicity checks

Exercise the new safety helper directly with `RuntimeChannels` and synthetic tickets/ring writes.

Required cases:

1. **Current RUNNING ticket rejection**
   - current ticket A exists;
   - helper returns true;
   - safety RUNNING -> ARMED;
   - generation advances;
   - no sticky error is set.

2. **Current ARMED ticket rejection**
   - current ARMED command ticket exists where the current publication contract allows it;
   - helper invalidates the ticket while retaining ARMED;
   - generation advances.

3. **Stale ticket must not revoke newer ticket**
   - publish A;
   - publish B so B is latest;
   - attempt to reject A;
   - helper returns false;
   - generation and lifecycle are not changed by rejection of A;
   - B remains current.

4. **Expired ticket**
   - rejection helper must not resurrect/change lifecycle for an already non-executable expired ticket.

5. **FAULT/e-stop/stopped runtime**
   - helper must not transition those states back to ARMED.

This stale-A/newer-B test is a hard acceptance criterion.

### 12.2 Fake-arm worker checks

Do not call `arm_loop()` because it performs startup and device connection.

Test `_handle_servo_command()` (and related pure/loop-local helpers) with a fake arm object implementing the minimal methods/properties used by the function.

Required cases:

1. same-generation delta below limit -> fake `servo()` called once;
2. delta exactly at configured limit -> accepted;
3. delta just above limit -> fake `servo()` not called;
4. current jump rejection -> ARMED boundary, no `error_state`, `last_target` unchanged;
5. stale jump-rejected ticket after newer publication -> newer command not revoked;
6. current non-finite target -> fail-fast / exception path remains;
7. current joint-limit target -> fail-fast / exception path remains;
8. fake SDK non-zero return -> fail-fast path remains;
9. successful fake SDK call -> `last_target`, command generation, and accepted command metadata update exactly as before.

Also verify a rejected arm ticket cannot later pass `coupled_command_ticket_allows_execution()` on the hand side because the generation/ticket ownership has been invalidated.

### 12.3 Calibration pure target checks

Prefer factoring the measured-relative proposal math into a small pure helper only if that makes testing materially simpler; do not create a large abstraction.

Required deterministic checks:

1. fixed measured pose + repeated synthetic `W` ticks:
   - every proposed position remains exactly one configured step from the same measured pose;
   - no cross-tick accumulation occurs.
2. one rotation tick:
   - proposal is one configured rotation increment from measured orientation.
3. workspace clipping still behaves correctly.
4. final qpos canonicalization selects the nearest valid equivalent around `previous_command`.
5. a final raw delta below the worker limit passes the producer early guard.
6. a final raw delta above the worker limit is rejected before publication.

### 12.4 Calibration lifecycle checks

Use fake keys / planner / publication results as needed. No device worker should start.

Required state sequences:

```text
IK reject
    -> RUNNING becomes ARMED
    -> previous_command re-anchors to measured
    -> blocked_until_release=True
```

Then verify:

```text
W still held      -> stays blocked
W + SPACE held    -> stays blocked
W + S held        -> stays blocked even though net motion can be zero
all jog keys up   -> block clears, no motion starts on this release tick
fresh W press     -> new RUNNING epoch may begin
```

Also simulate:

```text
publish succeeds
wait_command_accepted returns False
```

and verify the same ARMED + release-latch recovery.

Simulate `wait_command_accepted()` generation cancellation while lifecycle remains RUNNING and verify calibration explicitly quiesces to ARMED rather than assuming cancellation is enough.

### 12.5 Keyboard regression checks

Validate the changed helper/state behavior without starting a live keyboard listener or robot worker.

Required checks:

- existing `_compute_keyboard_target_update()` normal lookahead math is unchanged;
- IK/safety rejection closes RUNNING -> ARMED;
- rejection rebuilds command anchors from measured feedback;
- `blocked_until_release` has the same physical-key behavior as calibration, including W+SPACE and W+S;
- release debounce / final-action ACK path for a normal successful motion is preserved.

### 12.6 VR/policy/replay lifecycle checks

For any changed production path, simulate shared lifecycle values directly.

VR expected behavior if changed:

```text
teleop_active + recording_active + unexpected ARMED
    -> command-silent/end run
    -> current episode discarded
    -> error_state stays false
    -> fresh B required
```

Policy expected behavior:

```text
active physical episode + unexpected ARMED
    -> episode terminates
    -> no global sticky fault unless independent fault exists
```

Replay expected behavior if changed:

```text
running replay + unexpected healthy ARMED
    -> ReplayStatus.REJECTED
    -> not FAULT
```

### 12.7 Persisted-data compatibility checks

Because this task must not change persisted-data code, the strongest low-cost guard is scope inspection:

```bash
git diff --name-only
```

Confirm it contains no modifications under:

```text
dexmani_real/recording/
dexmani_real/dataset/
dexmani_real/ipc/schema.py
```

If local historical episodes are available, an optional offline smoke check may load them before/after the change and compare key arrays, but the task must not depend on private local datasets being present.

---

## 13. Required low-cost repository checks

Run at minimum:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Also run the focused offline assertion scripts/checks described above.

Do not run example programs as tests.

Do not run CUDA/device checks unless they are clearly offline and already part of the existing environment; they are not required for this control task.

A skipped or failed check is not a passing check. Report it clearly.

---

## 14. Implementation order

Use this order to minimize rework and prevent scope drift.

1. Read `AGENTS.md` and inspect `git status --short`; preserve unrelated user changes.
2. Confirm current `HEAD`; if it differs from the authoring baseline, re-read the relevant current files.
3. Re-trace:
   - `runtime/safety.py` ticket/generation locking;
   - `robot/command_validation.py`;
   - `robot/arm_worker.py`;
   - `control/jog.py`;
   - calibration motion/session;
   - keyboard session;
   - VR loop unexpected lifecycle handling;
   - policy executor motion-revoked behavior;
   - replay lifecycle classification.
4. Implement the atomic current-ticket rejection primitive first.
5. Implement worker jump-only recoverable rejection.
6. Run focused offline core atomicity + fake-arm checks before touching producers.
7. Refactor camera calibration to measured-relative transactional jog.
8. Run calibration pure/lifecycle offline checks.
9. Stabilize keyboard rejection/release behavior and local OnlineIK profile.
10. Run keyboard offline regression checks.
11. Inspect VR/policy/replay with the new worker behavior and make only the minimal lifecycle changes actually required.
12. Run focused lifecycle simulations for every changed path.
13. Run repository low-cost checks.
14. Inspect `git diff --stat`, `git diff`, and `git diff --name-only` for accidental scope expansion.
15. Confirm recording/dataset/IPC schemas are untouched.
16. Remove all temporary local validation scripts/artifacts.
17. If and only if all acceptance criteria below pass, delete `CODEX_TASK.md`.
18. Re-run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

19. Final handoff must state exactly what changed, all offline validation performed, and explicitly state that **real hardware was not exercised**.

---

## 15. Acceptance criteria

All of the following are mandatory.

### A. Worker correctness

- Final worker reference semantics remain unchanged: last SDK-accepted target in the same generation, measured qpos in a new generation.
- A current arm command jump above the configured limit never crosses the SDK boundary.
- A current arm command jump above the limit no longer crashes the arm worker or sets sticky `error_state`.
- A current jump rejection invalidates the command and establishes an ARMED boundary.
- `st.last_target` and accepted command metadata do not advance on rejection.
- Non-finite target, joint-limit violation, SDK failure, controller error, and feedback failure remain fail-fast.

### B. Atomicity

- Rejection of stale ticket A cannot revoke newer ticket B.
- The current-ticket check and generation/lifecycle invalidation occur under one `motion_lock` critical section.
- No hardware IO occurs while holding `motion_lock`.

### C. Calibration

- No persistent `target_pos/target_quat` accumulation remains in calibration motion state.
- Every jog proposal is derived from the latest measured EEF pose plus one configured increment.
- `previous_command` advances only after arm SDK acceptance during RUNNING.
- IK / ordinary safety / publication / ACK rejection quiesces to ARMED and requires full jog-key release.
- W+SPACE and W+S cannot bypass the release latch.
- Calibration-specific OnlineIK jump bounds are aligned to the worker hard limit.
- Final qpos is re-canonicalized around the calibration accepted command reference and raw-checked before publication.

### D. Keyboard

- Existing measured-bounded Cartesian lookahead and normal successful motion behavior are preserved.
- Keyboard-specific OnlineIK jump bounds are aligned to the worker hard limit.
- Final endpoint receives a local canonicalization/raw early guard.
- IK/safety rejection closes the current motion epoch, re-anchors from measured feedback, and requires full jog-key release.
- Normal release debounce and final normal action ACK semantics remain intact.

### E. VR / policy / replay

- Normal VR action generation is unchanged.
- A healthy worker-established ARMED boundary does not become a global hardware fault merely because a jump was rejected.
- An active VR recording affected by that rejection is not silently retained as a normal demonstration.
- Policy normal action decoding/generation remains unchanged unless a minimal lifecycle classification fix is strictly necessary.
- Replay timing / cadence is unchanged in this task.
- Healthy replay command rejection resolves to REJECTED rather than FAULT if current code otherwise misclassifies it.

### F. Data / compatibility

- `EPISODE_SCHEMA_VERSION` is unchanged.
- `action_arm_joint_sent` semantics are unchanged.
- No recording or dataset schema change is introduced.
- No processed/training representation change is introduced.
- No IPC dtype field is added for this task.

### G. Offline-only validation

- No xArm/XHand/RealSense hardware connection or hardware-affecting entry point is executed.
- Core stale-ticket atomicity is demonstrated offline.
- Core fake-arm accept/reject/fault behavior is demonstrated offline.
- Calibration and keyboard release-latch behavior is demonstrated offline.
- Repository compile/diff checks pass.

---

## 16. Explicit non-goals

Do not use this task to implement or clean up any of the following:

```text
CAS / reference-action protocol
new ARM_STATE telemetry fields
new recording fields
new processed-data fields
schema migrations
Mode 1 migration
Pink IK migration
high-rate software servo interpolation
tracking-error fault gates
changing 20° to a larger worker limit
clipping a solved IK target to 20°
policy temporal smoothing redesign
VR hand transactional-state cleanup
replay resampling redesign
per-frame replay ACK pacing
new metrics subsystem
broad config cleanup
README architecture rewrite
unrelated typing/style cleanup
```

If one of these becomes necessary to make the requested behavior correct, stop and document why rather than silently expanding scope.

---

## 17. Final review checklist

Before deleting this task file, inspect the implementation as a reviewer, not only as the author.

Ask explicitly:

```text
1. Can a stale rejected ticket revoke a newer command?
2. Can jump rejection still reach xArm SDK?
3. Can jump rejection still set sticky error_state indirectly?
4. Did any path accidentally turn measured tracking lag into a fault?
5. Did calibration retain any cross-tick virtual Cartesian target?
6. Can W+S or W+SPACE bypass the release latch?
7. Does calibration commit previous_command before ACK anywhere?
8. Did keyboard normal successful trajectory generation change unnecessarily?
9. Did VR/policy/replay normal cadence or action values change?
10. Did any recording/dataset/IPC schema file change?
11. Did any validation script or temporary artifact remain in the repository?
12. Was any real hardware touched despite the offline-only constraint?
```

If any answer is uncertain, re-inspect the exact code path before handoff.

---

## 18. Final handoff format

Keep the final Codex handoff concise but concrete. It should contain:

```text
- implementation summary by subsystem;
- important lifecycle/failure-semantics changes;
- files intentionally left unchanged to protect data compatibility;
- offline validation performed and results;
- any checks not run / limitations;
- explicit statement: no real hardware was connected or exercised;
- final git status / remaining user changes, if any.
```

Do not claim physical validation. Do not keep this task file after successful completion.
