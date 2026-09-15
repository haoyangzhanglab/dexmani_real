# Codex Task — Simplify and Fix Policy Deployment Action Clip / Reject Semantics

> **Temporary implementation artifact.** Read and execute this task, then remove this file in the implementation cleanup once the work is complete and verified. Do not move this plan into README/AGENTS/permanent architecture docs.

## 0. Repository contract

Before editing, read and obey `AGENTS.md`.

This repository controls real robot hardware. **Do not run hardware-affecting programs** for this task. Validation must be source inspection + offline checks only unless the user separately authorizes real hardware execution.

Before changes:

```bash
git status --short
```

Preserve unrelated user changes.

---

# 1. Goal

Fix the learned-policy deployment action safety semantics in `dexmani_real` with the **smallest coherent change**.

Current bad behavior is conceptually:

```text
policy action
  -> validate
  -> safety/IK reject one waypoint
  -> robot does not execute that waypoint
  -> executor still advances the policy trajectory
  -> physical state and logical chunk diverge
```

The new behavior must be:

```text
policy action
  -> decode
  -> deterministic physical-space projection
  -> final validation
  -> publish + advance
```

with only four runtime outcomes:

```text
1. projectable action         -> PROJECT -> PUBLISH -> ADVANCE
2. timing-stale action        -> SKIP STALE
3. feedback temporarily absent-> WAIT (no publish, no advance)
4. non-projectable unsafe action -> ABORT CURRENT ROLLOUT
```

A hardware/runtime invariant failure remains the existing `FAULT` path.

The core invariant is:

```text
Physical-safety rejection must never mean “skip this waypoint and execute a later waypoint”.
```

The **only** allowed non-published trajectory advancement is explicit time-domain stale dropping, as already implemented for elapsed/expired policy targets.

---

# 2. Scope / non-goals

## In scope

Primarily:

- `dexmani_real/deployment/executor.py`
- minimal related cleanup in `dexmani_real/config/defaults.py`
- minimal diagnostic/comment updates if needed

Inspect but do not redesign:

- `dexmani_real/control/publication.py`
- `dexmani_real/control/safety_gate.py`
- `dexmani_real/robot/arm_worker.py`
- `dexmani_real/robot/hand_worker.py`
- `dexmani_real/robot/command_validation.py`

## Explicit non-goals

Do **not** add:

- QP / CBF / optimization safety filters
- Ruckig inside the policy executor
- a recovery FSM
- reject-triggered immediate re-inference machinery
- Cartesian FK -> workspace projection -> IK for joint policies
- collision planning/checking beyond current behavior
- blocking per-command arm+hand ACK transactions
- new IPC schemas
- new hardware-limit hierarchies/config knobs
- normalized Flow-Matching output clipping as the physical safety mechanism
- changes to teleoperation behavior
- changes to data collection semantics
- changes to ManiFlow / `dexmani_policy`

Do not weaken final worker safety guards.

---

# 3. Source facts that the implementation must preserve

Re-read current source before editing; do not rely only on this task text.

## 3.1 Current executor bug

`PolicyExecutor._commit_terminal_step()` currently advances both published and rejected steps. `_reject_due_step()` therefore creates physical/logical trajectory divergence.

This semantics must disappear.

## 3.2 Arm worker contract

`robot/arm_worker.py` validates every arm target against:

- finite values
- arm joint limits
- `ArmParams.max_servo_command_jump_rad`

The jump reference is the worker's **last SDK-accepted target** for the active generation.

The worker guard stays unchanged and remains the final fail-safe before the SDK.

## 3.3 Hand worker contract

`robot/hand_worker.py` already owns temporal hand shaping:

```text
target
 -> mechanical bound validation
 -> per-tick delta limiting from last SDK-accepted setpoint while RUNNING
 -> bound validation again
 -> SDK send
```

Therefore the policy executor must not duplicate a hand endpoint-delta reject policy.

## 3.4 Hand limits

`HandParams.qpos_min_rad/qpos_max_rad` are command/operational bounds and are validated as nested inside the mechanical/rated bounds. They are the correct policy-side clipping range.

The hand worker still uses mechanical bounds as the final SDK invariant.

## 3.5 Timing stale behavior

Current executor intentionally skips actions/chunks that are no longer future-valid according to wall-clock scheduling. Preserve this.

Examples that may still call `_advance_prediction()` without publication:

- a wholly stale chunk / skipped stale prefix
- candidate/target that expired due to timing
- `PUBLISH_REASON_EXPIRED`

Do not conflate timing stale dropping with physical safety rejection.

## 3.6 Temporary feedback unavailability

`prepare_command()` distinguishes temporary unavailable/stale feedback from fatal feedback.

Temporary unavailability must continue to mean:

```text
WAIT
no publication
no prediction advance
```

Do not add a new WAIT state machine.

---

# 4. Reviewed design

## 4.1 Policy execution pipeline

Implement this pipeline for a decoded physical joint target:

```text
raw policy prediction
        |
        v
decode representation / optional EE IK
        |
        v
policy target projector
        |-- arm: wrap nearest equivalent
        |-- arm: absolute joint clip
        |-- arm: command-continuity clip
        `-- hand: operational absolute joint clip
        |
        v
existing SafetyGate
        |
        +-- valid ----------------> publish
        |                              |
        |                              v
        |                           advance
        |
        `-- actual unsafe target --> abort rollout
```

The projector belongs in the **policy executor**, not in generic `prepare_command()`. `prepare_command()` is a shared final validation boundary used by other command paths and should stay validation-oriented.

---

# 5. Required arm projection

Replace the current reject-only `_validate_policy_arm_action()` behavior with a pure/simple projection helper, e.g. `_project_policy_targets(...)` or equivalent.

Use physical joint radians.

## 5.1 Reference

For arm command continuity:

```python
reference_arm_qpos = (
    current_measured_arm_qpos
    if previous_arm_command_qpos is None
    else previous_arm_command_qpos
)
```

This reference is a **command-continuity reference**, not a tracking-error constraint.

Workspace validation continues to start from measured feedback via `SafetyGate`.

## 5.2 Canonicalize periodic joints

Keep:

```python
wrap_nearest_equivalent(...)
```

before numeric clipping.

## 5.3 Absolute joint clipping

After canonicalization:

```python
arm = np.clip(
    canonical_arm,
    runtime.arm.joint_limit_lower,
    runtime.arm.joint_limit_upper,
)
```

Do not reject an otherwise finite arm proposal merely because it exceeded a joint endpoint before projection.

The post-project `SafetyGate` and arm worker still validate the result.

## 5.4 Command-continuity clipping

Use the existing single source of truth:

```python
runtime.arm.max_servo_command_jump_rad
```

Do not keep a second policy-specific arm jump threshold.

Semantics:

```python
delta = arm - reference_arm_qpos
arm = reference_arm_qpos + np.clip(delta, -limit, +limit)
```

This is **command continuity shaping**, not a physical velocity model. Do not derive a new per-policy-step velocity limit from `max_joint_velocity_deg_per_s`; the arm controller/Mode 6 owns dynamic trajectory shaping.

Implementation must be numerically safe at the worker threshold. The projected result must satisfy the exact worker-side `<= max_servo_command_jump_rad` predicate under float64 arithmetic. Prefer the smallest simple implementation; add only minimal numerical slack if required by focused tests.

Because both the reference and clipped endpoint are inside the arm joint box, the continuity projection must remain inside the joint box.

---

# 6. Required hand projection

For policy deployment, hand shaping is only:

```python
hand = np.clip(
    raw_hand,
    runtime.hand.qpos_min_rad,
    runtime.hand.qpos_max_rad,
)
```

Remove the policy deployment use of:

```text
policy.hand_max_action_jump_rad
SafetyGate(max_hand_delta_rad=...)
```

Build the policy `SafetyGate` with `max_hand_delta_rad=None`.

Do not add a second hand slew/rate limiter in the executor. `hand_worker.py` already owns the per-tick actuator-side rate limiting.

Once the policy target is clipped to command bounds, do not rely on the policy-specific endpoint roundoff canonicalizer. Stop passing `canonicalize_policy_hand_roundoff=True` from this executor path. Do not delete generic/shared roundoff helpers unless they become globally unused and removal is obviously safe; avoid unrelated cleanup.

---

# 7. Critical reliability fix: arm-only non-blocking publication backpressure

This is required.

A previous design that only projected relative to `previous_arm_command_qpos` was insufficient because:

```text
executor reference = previous published arm target
worker reference   = last SDK-accepted arm target
```

With a latest-wins command ring, the worker could theoretically miss an intermediate arm command. Then two individually valid executor steps could compose into a >20 degree jump at the worker.

Do **not** solve this with blocking ACK waits.

Instead, before publishing a new arm command, require the prior published arm command to have been accepted by the arm worker.

Use the already existing `_CommandProgress` state.

Conceptually:

```python
latest = self.progress.latest_published_action_id
arm_accepted = self.progress.arm_accepted_action_id

if self.execute and latest is not None and arm_accepted < latest:
    return  # wait; do not publish and do not advance
```

Important details:

- This is non-blocking; just return from the current executor poll.
- Keep ingesting fresh inference/predictions while possible; place the gate late enough that inference freshness/trace handling is not unnecessarily frozen.
- Do not consume a control slot when waiting for arm acceptance.
- When time advances enough that a pending future action becomes stale, the existing timing logic may skip it later. That is correct.
- Do **not** require the hand `accepted_target_action_id` to catch up before the next publication. The hand worker deliberately rate-limits toward endpoints and may not exactly reach every intermediate policy endpoint. Blocking on exact hand endpoint acceptance would over-serialize the controller and duplicate hand-worker ownership.

With this gate, before every publication after the first:

```text
previous_arm_command_qpos == prior arm target accepted by the worker
```

under the normal generation contract. This makes executor and worker jump references consistent without new IPC fields or blocking transactions.

If the arm worker unexpectedly rejects a projected command, the existing progress/safety machinery should stop/fault rather than allowing later commands to hide the violation. Do not weaken the worker guard to make the test pass.

---

# 8. Reject / abort semantics

## 8.1 Delete “reject this step then continue” behavior

`_reject_due_step()` must no longer exist with semantics that call `_commit_terminal_step()`.

Replace that control flow with a small terminal-rollout helper if useful, e.g. `_abort_due_action(...)`.

A terminal policy abort may:

- capture the event timestamp
- update rejection/IK diagnostics
- record the existing rejection frame status / policy trace decision if useful
- call existing `_finish_episode(..., aborted=True)`

It must **not**:

- call `_advance_prediction()`
- call `_commit_terminal_step()`
- update previous command baselines as if a command was executed

Reuse the existing episode fence into `SafetyState.ARMED`; do not convert ordinary policy failure into a sticky hardware `FAULT`.

## 8.2 IK failure

For `action_ee`:

```text
IK no usable solution -> abort current rollout
```

Do not skip to a later future action.

Preserve existing diagnostic recording if it can be kept simply.

## 8.3 Workspace violation

After projection, if the existing policy workspace check reports a real workspace violation:

```text
abort current rollout
```

Do not add joint->FK->Cartesian clip->IK projection in this task.

## 8.4 Post-project invariant failures

After the projector, these should be impossible during normal operation:

- arm joint limit rejection
- hand joint limit rejection
- hand delta rejection (disabled for policy gate)

If the post-project shared gate reports an impossible invariant violation, treat it as an implementation/configuration/runtime invariant failure rather than silently continuing. Prefer the existing fail-closed fault path for impossible post-project states.

Keep existing `prepared.fatal` semantics for check failures such as workspace-check exceptions.

## 8.5 Malformed IPC / non-finite predictions

Do not weaken existing inference/IPC finite and shape validation. If malformed prediction IPC reaches the executor, preserve the existing fail-closed behavior rather than adding a second permissive fallback.

---

# 9. Publication / timeline semantics

After the change, `_commit_terminal_step()` should mean only a successfully published policy control decision.

Update its comment/docstring accordingly.

Expected table:

| Event | Publish? | Advance prediction? | End rollout? |
|---|---:|---:|---:|
| projected safe action | yes | yes | no |
| temporary feedback unavailable | no | no | no |
| waiting for prior arm ACK/backpressure | no | no | no |
| explicit stale/expired action | no | yes | no |
| IK failure | no | no | yes |
| actual workspace safety violation | no | no | yes |
| impossible post-project invariant / checker failure | no | no | FAULT |
| lifecycle stop / generation fence | existing semantics | existing semantics | existing boundary |

Do not change `PUBLISH_REASON_EXPIRED` stale-drop behavior.

---

# 10. Recording and trace behavior

Do not change persisted schemas.

Keep the useful existing split:

- policy prediction / `PolicyTrace` contains raw model actions
- rollout `EpisodeAction.arm_qpos_cmd` / `hand_qpos_cmd` contains the actual projected submitted target

For EE policies, preserving raw Cartesian intent in the existing Cartesian fields while recording projected submitted arm qpos is acceptable and already matches the `EpisodeAction` intent/command split.

If preserving `_RejectKind` is the smallest way to retain existing IK/safety failure recording, keep it. **Do not delete abstractions merely for cosmetic cleanup.** Remove only code made genuinely obsolete by removing per-step continuation after rejection.

Clip/projection must remain diagnosable. Prefer existing trace + recorded submitted command over a new persisted telemetry schema. A small in-memory counter/log is acceptable only if it is trivial and clearly useful; do not build a telemetry subsystem.

---

# 11. Configuration cleanup

After confirming repository-wide references:

- remove `PolicyParams.arm_max_action_jump_rad`; use `ArmParams.max_servo_command_jump_rad`
- remove `PolicyParams.hand_max_action_jump_rad`; hand temporal shaping belongs to `HandParams.hand_max_delta_rad_per_tick`
- remove their validation branches/comments

Do **not** remove or relocate `PolicyParams.endpoint_delta_tolerance_rad` in this task if shared `SafetyGate` still uses it as its generic default. Avoid broad shared-safety refactors.

Do not introduce replacement policy-specific clipping thresholds.

---

# 12. Reference-project design alignment

The implementation should reflect the common useful pattern found in the reference projects, without copying their weaknesses:

## Stanford Diffusion Policy

Borrow:

- correctable geometric violations are projected/clipped
- timing-stale actions are dropped as a timing concern
- large motion is shaped/retimed rather than represented as “drop waypoint and continue”

## ManiUniCon

Borrow:

- configuration-space projection for correctable limits
- uncorrectable safety failures terminate/fault instead of skipping a single waypoint

## Standard LeRobot / LeFranX

Borrow:

- relative target clipping / command shaping as a producer-side safety mechanism
- XHand-style absolute target clipping
- leave trajectory dynamics to the controller/worker layer

Do not copy:

- weak send-failure propagation
- silent logical progress after failed physical application

## PI-R2

Borrow:

- do not advance into later trajectory content when a trustworthy command cannot currently be executed
- its measured-state/delta-clip idea supports projection rather than binary per-waypoint rejection

Do not copy:

- the current PI-R2 production path's missing clip hookup
- warning-only SDK failure handling

The resulting DexMani implementation may include the small arm acceptance backpressure because DexMani's own latest-wins IPC + final worker jump guard creates a repository-specific correctness requirement not present in the same form in those projects.

---

# 13. Suggested executor pseudocode

Use existing names/structure where possible; do not force a rewrite.

```python
def _project_policy_targets(arm_qpos, hand_qpos, reference_arm_qpos, runtime):
    arm = wrap_nearest_equivalent(
        arm_qpos,
        reference_arm_qpos,
        runtime.arm.joint_limit_lower,
        runtime.arm.joint_limit_upper,
    )

    arm = np.clip(
        arm,
        np.asarray(runtime.arm.joint_limit_lower, dtype=np.float64),
        np.asarray(runtime.arm.joint_limit_upper, dtype=np.float64),
    )

    jump = float(runtime.arm.max_servo_command_jump_rad)
    arm = reference_arm_qpos + np.clip(
        arm - reference_arm_qpos,
        -jump,
        jump,
    )

    hand = np.clip(
        hand_qpos,
        np.asarray(runtime.hand.qpos_min_rad, dtype=np.float64),
        np.asarray(runtime.hand.qpos_max_rad, dtype=np.float64),
    )

    return arm, hand
```

Then conceptually:

```text
decode / IK
  -> failure: record + finish aborted episode
  -> success: project targets
  -> build candidate
  -> prepare_command with policy gate (no hand delta gate, no hand roundoff shaping)
      -> unavailable: return/wait
      -> real workspace violation: record + abort episode
      -> impossible/fatal invariant: fault
  -> ensure target has not become stale
  -> publish
      -> expired: stale skip
      -> lifecycle fence: existing boundary
      -> success: record publication, update previous commands, commit/advance
```

And in the active loop, after worker progress observation / fresh prediction ingest and before a new physical publication:

```text
if prior published arm action has not yet been accepted:
    return
```

---

# 14. Focused verification requirements

Do not run robot hardware.

Use the smallest offline checks available.

At minimum run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Also perform focused offline verification of the changed pure/control semantics. Do not add a new testing framework/dependency solely for this task if the repository has none.

Required cases to verify, via existing tests if available or a small offline Python assertion harness if imports permit:

## Projection identity

Safe joint target remains unchanged, except legitimate periodic canonicalization.

## Hand incident regression

For the problematic index-bend joint:

```text
raw = -0.176450133 rad
command lower = -0.174 rad
```

Expected:

```text
projected = -0.174 rad
no policy SafetyGate hand-joint rejection
```

## Arm jump projection

Example:

```text
reference = 0 deg
raw target = 25 deg
worker guard = 20 deg
```

Expected projected command is within 20 deg of reference and passes the same worker jump predicate.

Test the exact-threshold numerical case too.

## Arm publication backpressure

Simulate/prove from control flow:

```text
latest_published_action_id = N
arm_accepted_action_id = N-1
```

Expected:

```text
no new command publication
no prediction commit/advance due to safety/backpressure itself
```

After acceptance reaches `N`, the executor may publish a future-valid action.

## Workspace failure

A post-project target that fails the real workspace predicate must terminate the current policy rollout without `_advance_prediction()`.

## IK failure

No usable IK solution must terminate the current rollout without advancing to the next action in that chunk.

## Feedback unavailable

Temporary unavailable feedback must return/wait and must not advance.

## Stale action

Existing elapsed / `PUBLISH_REASON_EXPIRED` behavior must continue to increment stale diagnostics and skip the stale action.

## Final worker guards

Do not modify tests/logic to bypass `check_worker_arm_target` or `check_worker_hand_target`. Projected normal commands should make those guards unreachable in normal policy operation, not weaker.

---

# 15. Incident-level acceptance criteria

The redesign is accepted only if its semantics eliminate the original failure chain:

```text
small generative hand bound overshoot
 -> hand command clip
 -> coupled command remains publishable
 -> no policy-level hand endpoint rejection
 -> no reject-induced physical/logical divergence
 -> no secondary arm jump cascade caused by skipped physical progress
```

For the known trace, the previous selected hand mechanical violations should become projection events rather than repeated policy-step safety rejects.

Do not claim real-hardware success unless hardware is actually tested later under explicit authorization.

---

# 16. Code-quality constraints

- Prefer one small pure projector helper in `executor.py`; do not create a framework.
- Keep generic `SafetyGate` and publication ownership intact.
- Keep workers as final hardware-authority boundaries.
- Keep one source of truth for thresholds.
- No duplicated hand rate limiting.
- No blocking loops waiting for ACK.
- No silent safety rejection + trajectory advancement.
- Preserve current stale-timing semantics.
- Update comments/docstrings that explicitly describe rejected steps as terminal/advanceable.
- Do not perform unrelated cleanup.

---

# 17. Final handoff checklist for Codex

Before completion:

1. Re-read the focused diff.
2. Confirm no teleop behavior changed.
3. Confirm no worker safety guard was weakened.
4. Confirm policy hand delta rejection is disabled.
5. Confirm arm uses `ArmParams.max_servo_command_jump_rad` as the single jump threshold.
6. Confirm a safety/IK abort cannot call `_commit_terminal_step()` / `_advance_prediction()`.
7. Confirm stale/expired timing paths still can skip stale actions.
8. Confirm arm-only non-blocking acceptance backpressure is present.
9. Confirm temporary feedback unavailability still waits.
10. Run the offline checks listed above.
11. Report what was not hardware-validated.
12. **Delete `CODEX_TASK_POLICY_ACTION_SAFETY_REDESIGN.md` as cleanup once implementation and verification are complete.**
