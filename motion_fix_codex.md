# Motion Control Fix — Codex Task

## 0. Task status and reviewed baseline

This task was finally reviewed against the latest `main` control behavior after commit:

- reviewed repository head before this task file: `2d17fe8af11cbb8b7ef2d3e3e35ad22a30496208`
- repository: `haoyangzhanglab/dexmani_real`

The commits after the earlier motion review primarily changed Raw/Canonical export, point-cloud processing, calibration state, and recording behavior. The motion-control defects described below are still present in the current keyboard/calibration/xArm control paths.

Before editing, re-read the current branch and `AGENTS.md`. If `main` has moved since this task was committed, reconcile only relevant control-path changes. Do not overwrite unrelated work.

This task is **implementation work**, not another architecture redesign.

---

## 1. Intent

Eliminate the observed xArm motion discontinuities/jitter in:

- `examples/keyboard_teleop.py`
- `examples/calibrate_camera.py`
- shared return-home behavior used by keyboard, camera calibration, and `examples/collect_teleop.py`

The reported symptoms are:

1. keyboard/calibration Cartesian jog can visibly jerk on short press/release;
2. repeated press/release causes unnecessary controller stop/restart transitions;
3. pressing outward against the workspace boundary can produce small repeated motion/jitter;
4. recoverable calibration IK rejection can unnecessarily stop and restart Mode 6;
5. after return-home visually reaches the canonical pose, a small residual movement can still occur around the Mode-0 → Mode-6 restoration.

Fix the **control semantics**, not the tuning.

The desired architecture is:

```text
operator jog intent
      ↓
propose Cartesian target
      ↓
workspace projection
      ↓
effective target changed?
      ├─ no  → no IK / no publication
      └─ yes
           ↓
        online IK
           ├─ recoverable reject → keep last committed target
           └─ success
                ↓
          establish RUNNING epoch if needed
                ↓
             publish
                ├─ rejected lifecycle → clear local reference
                └─ success
                     ↓
                  commit target
                     ↓
          Mode 6 remains active across ordinary jog idle
```

Return-home must finish as:

```text
revoke streaming authority
      ↓
Mode 0 validated planned HOME
      ↓
Mode-0 convergence + dwell
      ↓
restore Mode 6 / State 0
      ↓
feedback-driven post-Mode6 convergence + dwell
      ↓
HOME success
```

---

## 2. External implementation references and how to use them

Use these projects only as behavioral references; do not copy their architectures wholesale.

### ManiUniCon

Reference:

- `Universal-Control/ManiUniCon/maniunicon/policies/keyboard.py`

Relevant idea:

- keyboard/operator targets are persistent;
- releasing motion keys stops target integration rather than destroying the command reference;
- explicit reset/re-anchor boundaries synchronize target state back to measured robot state.

Do **not** port ManiUniCon's Mode-1/high-rate interpolation architecture.

### UFACTORY LeRobot

Reference:

- `xArm-Developer/lerobot_robot_ufactory/src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py`

Relevant Mode-6 behavior:

- joint streaming uses Mode 6 + State 0;
- targets are absolute joint positions;
- normal streaming uses `set_servo_angle(..., wait=False)`;
- ordinary lack of a new action does not inherently require `set_state(4)`;
- `set_state(4)` is a true stop/lifecycle action;
- after setup/home-like blocking positioning, Mode 6 is restored and the implementation allows a short settling period.

DexMani should keep its stronger run-id fencing, freshness checks, process ownership, IK/collision logic, and planned HOME.

---

## 3. Non-negotiable invariants

Do **not** change the following values or mechanisms as part of this task.

### Numerical parameters

Keep exactly the current resolved/default behavior:

- keyboard Cartesian translation step: `0.008 m`
- keyboard Cartesian rotation step: `0.03 rad`
- keyboard/calibration control rate: `30 Hz`
- arm worker rate: `30 Hz`
- VR teleop rate: current `16 Hz`
- xArm Mode-6 max joint velocity: `135 deg/s`
- xArm max joint acceleration: `810 deg/s^2`
- all existing homing tolerances/timing values

Do not silently retune any of them.

### Control architecture

Keep:

- xArm normal streaming in **Mode 6**;
- `XArm7.servo()` using `set_servo_angle(..., wait=False)`;
- latest-target `robot_command_ring` semantics;
- `run_id` as the lifecycle authority epoch;
- `SafetyState = DISARMED / ARMED / RUNNING / FAULT`;
- current xArm/XHand worker ownership;
- current emergency-stop/fault/shutdown fencing;
- current VR EMA behavior;
- current planned HOME collision/path checks.

Do not introduce:

- Mode 1;
- Mode 4/5 velocity control;
- State-6 deceleration-stop as the ordinary jog release mechanism;
- a 100/200 Hz software trajectory interpolator;
- generic EMA/low-pass/joint-slew filters for the arm;
- command ACK/adoption/reached ledgers;
- new command identities;
- new SafetyState values;
- `command_lookahead_frames`;
- a short-tap latch;
- new committed test directories.

---

## 4. Final lifecycle decision

The central distinction is:

```text
ordinary jog idle
    !=
motion authority revocation
```

### Ordinary keyboard/calibration jog idle

When no effective jog delta exists after a valid streaming epoch has begun:

- do not call `revoke_motion()`;
- do not clear the committed Cartesian target;
- do not clear the committed joint target;
- do not increment `run_id`;
- do not cause `arm_worker` to call `arm.stop()`;
- simply publish no new target for that tick.

`SafetyState.RUNNING` means the current streaming authority/session remains valid. It does **not** mean the joints must have nonzero velocity at every instant.

### True lifecycle/safety boundaries

These still revoke/fence motion exactly as today:

- HOME;
- e-stop;
- hardware/controller fault;
- required feedback becoming unavailable/stale;
- explicit stop/quit/shutdown lifecycle;
- calibration `stop_request`;
- VR manual C pause;
- VR stale/unavailable required observation;
- other existing explicit episode/lifecycle boundaries.

Do not weaken these paths.

---

## 5. Scope of production changes

The intended production diff should remain concentrated in:

1. `dexmani_real/teleop/jog.py`
2. `dexmani_real/teleop/keyboard_session.py`
3. `dexmani_real/calibration/camera/motion.py`
4. `dexmani_real/robot/drivers/xarm7.py`

### Files that should normally remain unchanged

Do not change these unless inspection proves a small change is strictly required for correctness:

- `dexmani_real/robot/arm_worker.py`
- `dexmani_real/runtime/safety.py`
- `dexmani_real/robot/commands.py`
- `dexmani_real/teleop/runner.py`
- `dexmani_real/teleop/control/controller.py`
- `dexmani_real/config/control.py`
- `dexmani_real/config/hardware.py`

The current worker behavior is intentional: once an epoch is truly revoked, the arm worker must stop that authority. The bug is producer-side overuse of revocation for ordinary jog idle.

---

## 6. Change A — one small pure Cartesian jog proposal helper

### File

`dexmani_real/teleop/jog.py`

Keyboard teleop and camera calibration currently duplicate:

- incremental Cartesian translation;
- workspace clipping;
- incremental XYZ Euler rotation composition;
- effective target-change detection.

Create one **small pure helper** for this demonstrated duplication.

Suggested semantic interface (exact naming may be improved if clearer):

```python
propose_cartesian_jog_pose(
    command_pose,
    dx,
    drpy,
    position_lower,
    position_upper,
) -> tuple[Pose, np.ndarray, bool]
```

The return values should provide:

1. the proposed `Pose`;
2. the unclipped desired position, needed by calibration boundary logging;
3. whether the projected Cartesian target actually changed.

Requirements:

- no hardware access;
- no SafetyState access;
- no run-id access;
- no IK;
- no publication;
- no logging;
- no mutable controller object/class;
- translation and rotation composition must preserve current semantics;
- use the existing wxyz quaternion convention correctly.

### Effective no-op semantics

A target is unchanged when:

- projected position is exactly the current committed position, and
- there is no rotation increment.

Do not add a user-visible deadband or hysteresis.

Workspace clipping uses deterministic configured boundary values, so exact equality is appropriate for repeated outward presses at an already committed boundary.

If translation is clipped but rotation changes, the proposal is still changed.

If one axis is clipped but another axis changes, the proposal is still changed.

---

## 7. Change B — keyboard jog uses a persistent Mode-6 epoch

### File

`dexmani_real/teleop/keyboard_session.py`

### Current defect

Current ordinary key release does all of:

```text
command_qpos = None
command_pose = None
revoke_motion()
```

That invalidates `run_id`, and `arm_worker` correctly turns the invalidated streaming epoch into a State-4 stop. The next jog begins a new epoch and causes Mode 6 to be re-entered.

This creates unnecessary:

```text
Mode6 → State4 → Mode6
```

cycles during ordinary short press/release.

### Required behavior

#### 7.1 Persistent committed references

Treat:

- `command_pose` as the last successfully published Cartesian target;
- `command_qpos` as the last successfully published arm joint target.

Once established, ordinary key release must preserve both.

#### 7.2 Idle

For:

```python
not moving
```

do not revoke and do not clear the references.

Just skip producing a new command for that tick.

If the system has never published a jog target and remains ARMED, the references may naturally remain `None`.

#### 7.3 True invalid observation

If required robot feedback is unavailable/stale (`read_observation(...) is None`):

- clear local command references;
- revoke RUNNING authority as today;
- do not keep a stale epoch alive.

#### 7.4 HOME

R remains a true lifecycle boundary:

- clear local references;
- revoke the streaming epoch;
- run planned HOME;
- after HOME, leave local command references unset;
- the next valid jog must re-anchor from fresh measured state.

Do not change HOME key edge semantics.

---

## 8. Change C — keyboard proposal/commit is transactional

### File

`dexmani_real/teleop/keyboard_session.py`

### Current defect

The current code assigns the new `command_pose` before IK succeeds.

That allows unreachable Cartesian intent to accumulate across rejected ticks.

### Required transaction

For each effective jog:

1. ensure a local baseline exists:
   - if no committed command exists, use fresh measured qpos and its FK pose as the local baseline;
2. compute `proposed_pose` without changing committed state;
3. apply workspace projection;
4. if the proposal is an effective Cartesian no-op:
   - skip IK;
   - skip publication;
   - keep committed references unchanged;
5. set current measured hand geometry in the planner when hand feedback is available;
6. solve IK using:
   - `proposed_pose`;
   - fresh measured arm qpos;
   - last committed joint command/baseline;
7. `IKFailureKind.INVALID_OUTPUT` remains fatal;
8. ordinary IK no-solution:
   - publish nothing;
   - keep committed pose/qpos unchanged;
   - do not revoke;
9. only after a valid IK candidate exists, establish RUNNING authority if currently ARMED;
10. use the resulting/current `run_id` for publication;
11. only when `publish_command()` succeeds:
    - commit `command_pose = proposed_pose`;
    - commit `command_qpos = target.copy()`;
12. if publication is rejected because lifecycle authority changed:
    - clear local command references;
    - do not pretend the proposal was committed.

### Why `begin_motion()` should be after a valid proposal/IK

Do not enter RUNNING merely because a key is physically held if:

- the workspace projection is a no-op;
- IK rejected the first proposal.

A new RUNNING epoch should be created only when there is an actually publishable first target.

Once RUNNING exists, ordinary idle keeps it alive.

---

## 9. Change D — camera calibration adopts the same committed-target semantics

### File

`dexmani_real/calibration/camera/motion.py`

Camera calibration should share the same Cartesian proposal helper and command semantics as keyboard jog.

### Required behavior

#### 9.1 Ordinary idle

When no effective jog delta is requested:

- keep `state.command_pose`;
- keep `state.command_qpos`;
- do not revoke RUNNING;
- preserve existing periodic idle status output.

#### 9.2 First target

If no committed command reference exists:

- initialize a local baseline from `state.current_qpos` and FK;
- create/solve a proposal first;
- call `begin_motion()` only when an effective valid IK target is ready to publish.

#### 9.3 Effective workspace no-op

If clipping produces exactly the already committed pose and rotation is unchanged:

- keep boundary warning behavior;
- skip IK;
- skip publication;
- keep command state and epoch unchanged.

This is important for avoiding repeated redundant 7-DOF IK solves at a hard workspace boundary.

#### 9.4 Transactional commit

Do not assign `state.command_pose = proposed_pose` before successful IK/publication.

Only a successfully published target becomes the next committed reference.

---

## 10. Change E — split recoverable calibration rejection from authority revocation

### File

`dexmani_real/calibration/camera/motion.py`

The current `_reject_calibration_motion()` conflates:

1. recoverable target rejection; and
2. motion-authority/lifecycle revocation.

Separate these semantics with the smallest direct code possible.

Do not build a new state machine or controller class.

### Recoverable IK rejection

For an ordinary finite IK target that has no valid solution:

- keep the last committed `command_pose`;
- keep the last committed `command_qpos`;
- keep the existing RUNNING epoch if one exists;
- publish no new target;
- set `blocked_until_release = True`;
- require physical jog-key release before accepting another jog proposal;
- do not call `revoke_motion()`.

This lets Mode 6 finish/hold the last valid endpoint instead of hard-stopping the controller.

### Hard/lifecycle rejection

The following remain hard boundaries and must still revoke/fault as appropriate:

- `shared.stop_request`;
- e-stop;
- sticky error;
- runtime shutdown;
- unexpected SafetyState;
- HOME;
- session exit.

In particular, **do not route `shared.stop_request` through the new soft IK-reject path**. It is an explicit command-admission/lifecycle signal and must retain hard authority semantics.

After consuming/resetting `stop_request`, local command references must not survive as if they were still authoritative.

Suggested organization:

- one tiny helper for “block jog until physical release, preserve committed target”;
- one tiny helper/path for “clear reference and revoke/fault authority”.

Avoid an abstraction hierarchy.

---

## 11. Change F — shared HOME must settle after restoring Mode 6

### File

`dexmani_real/robot/drivers/xarm7.py`

### Current behavior

Normal `XArm7.home()` paths validate Mode-0 convergence/dwell and then call:

```python
self.enter_mode6()
```

and return immediately.

Thus HOME success currently proves the Mode-0 trajectory settled, but does not prove the robot remains settled after restoring Mode 6.

### Required postcondition

On every successful HOME path:

1. complete the existing Mode-0 path/convergence/dwell exactly as today;
2. call `enter_mode6()`;
3. wait, using live feedback, until the canonical final target satisfies the existing:
   - `homing.convergence_rad`;
   - `homing.velocity_convergence_rad_s`;
4. require those conditions continuously for the existing `homing.dwell_s`;
5. only then return HOME success.

### No new tuning

Do not add a new fixed sleep configuration.

Do not change existing homing parameter values.

Use existing values to bound the post-Mode6 settle. A reasonable implementation uses:

```text
post-mode6 timeout = target_timeout_s + dwell_s
```

or an equivalently simple bounded expression derived only from existing homing values.

### Transient handling

Unlike the existing strict Mode-0 dwell helper, the post-Mode6 settle should tolerate a brief mode-switch transient:

- if position/velocity leave the convergence window, reset `stable_since`;
- continue until the bounded deadline;
- require one full continuous dwell period before success.

Do not immediately fail on the first transient non-converged feedback sample.

Still fail immediately on:

- abort request;
- controller error;
- invalid SDK feedback/read failure.

On timeout, raise a useful error including the largest joint position error and/or velocity if practical.

### Feedback callback

Continue invoking the existing HOME feedback callback during post-Mode6 settling so shared arm feedback reflects the actual post-switch state.

### Simplify the three success exits

Current `home()` has multiple normal paths that independently call `enter_mode6()`.

Refactor only enough to funnel successful HOME completion through one small “restore Mode6 and settle” tail/helper, avoiding duplicate postcondition logic.

Do not redesign the HOME planner or worker protocol.

---

## 12. VR teleop — explicit non-scope after final review

### Files

- `dexmani_real/teleop/runner.py`
- `dexmani_real/teleop/control/controller.py`

Do **not** apply the keyboard idle rule mechanically to VR.

### Why

Normal active VR teleop already keeps one RUNNING epoch and does not revoke just because the human wrist is momentarily stationary.

C pause is different from keyboard key release:

- C is an explicit operator lifecycle boundary;
- DexMani has asynchronous producer → latest-target ring → hardware-worker execution;
- revoking `run_id` at C prevents a target prepared before the pause from crossing the SDK fence after the pause.

Therefore preserve:

```text
C pause
→ revoke
→ worker stop/hold
→ clear reference
→ fresh post-pause re-anchor
→ new epoch
```

Also preserve hard revoke for stale required VR/robot observations.

Do not change VR EMA, VR workspace handling, or hand-retarget temporal behavior in this task.

VR receives the HOME improvement automatically through the shared `XArm7.home()` fix.

---

## 13. Do not modify `arm_worker` to hide producer bugs

### File

`dexmani_real/robot/arm_worker.py`

The current behavior:

```text
streaming epoch becomes invalid
→ arm.stop()
→ stopped=True
→ next authorized arm target lazily enter_mode6()
```

is correct for a **true revoked authority epoch**.

Do not introduce:

- “soft revoke” exceptions in the worker;
- special run-id classes;
- controller-mode heuristics;
- ignored revocations.

The fix is to stop revoking ordinary keyboard/calibration idle in the producer.

---

## 14. Required behavior matrix

After implementation, this matrix must be true.

| Event | Local target ref | run_id / SafetyState | State-4 consequence |
|---|---|---|---|
| keyboard ordinary key release | preserve | preserve RUNNING | no |
| calibration ordinary key release | preserve | preserve RUNNING | no |
| keyboard workspace outward no-op | preserve | preserve | no |
| calibration workspace outward no-op | preserve | preserve | no |
| keyboard ordinary IK reject | preserve | preserve | no |
| calibration ordinary IK reject | preserve + block until release | preserve | no |
| keyboard required feedback stale | clear | revoke if RUNNING | yes |
| calibration runtime unhealthy | clear | revoke/fault | yes |
| calibration `stop_request` | clear | hard lifecycle revoke | yes if epoch active |
| keyboard/calibration HOME | clear | revoke to ARMED HOME authority | yes before HOME |
| VR C pause | current behavior | revoke | yes/hold as current |
| VR stale | current behavior | revoke | yes/hold as current |
| e-stop / hardware fault | clear/irrelevant | revoke/fault | yes |
| process shutdown | irrelevant | revoke/shutdown | yes |

---

## 15. Recording/data semantics must not regress

The repository's latest recording contract is unrelated to the arm-motion fix and must be preserved.

In particular:

- do not modify Raw/Canonical data contracts;
- do not change the clean-teleop capture rules;
- do not change C pause marking capture discard-only;
- do not fabricate previous actions for rows where no command was published;
- do not make keyboard/calibration changes leak into recording code.

The VR runner/controller were recently changed for recording semantics. Preserve those changes.

---

## 16. No unrelated payload retuning

Camera calibration supports:

- `--hand-geometry absent`
- `--hand-geometry secured-home`

The current xArm TCP load is configured globally.

Do not invent or change TCP payload values in this task.

If hardware validation later shows that only the physically-absent-hand case still jitters after the control fix, treat payload identification as a separate task backed by actual physical configuration data.

---

## 17. Implementation quality constraints

The implementation should remain direct and small.

Prefer:

- pure helper + local control flow;
- explicit proposal vs committed variable naming where useful;
- one Mode6 post-HOME settle helper/tail;
- no new framework;
- no new generic controller abstraction.

Comments should explain only non-obvious semantics, especially:

- why ordinary jog idle intentionally keeps RUNNING;
- why a Cartesian proposal is committed only after publication;
- why post-HOME Mode6 settling is part of the HOME success postcondition.

Do not leave historical debugging prose in production source.

---

## 18. Offline verification

Do not run hardware-affecting code without explicit authorization.

Do not add a committed `tests/` directory.

Use focused one-off pure/offline smoke checks for the changed semantics.

At minimum verify the following cases with pure helpers/fakes or inspection-level harnesses that do not connect hardware:

### Cartesian proposal helper

1. already at +X workspace boundary + outward X jog:
   - proposed position equals committed position;
   - `changed == False`;
2. at +X boundary + inward X jog:
   - `changed == True`;
3. translation clipped but nonzero rotation:
   - `changed == True`;
4. one clipped axis plus another changing translation axis:
   - `changed == True`;
5. quaternion output remains finite and normalized according to existing Pose expectations.

### Keyboard lifecycle/commit

6. ordinary idle after a committed command:
   - does not call/reach `revoke_motion`;
   - committed pose/qpos remain unchanged;
7. recoverable IK rejection:
   - committed pose/qpos remain unchanged;
   - run epoch remains unchanged;
8. publication rejection:
   - local references are cleared;
9. first workspace no-op / first IK rejection while ARMED:
   - does not create a useless RUNNING epoch;
10. successful first proposal:
    - establishes RUNNING and publishes with the resulting run_id.

### Calibration lifecycle/commit

11. ordinary idle:
    - keeps committed target and epoch;
12. ordinary IK rejection:
    - keeps committed target and epoch;
    - sets `blocked_until_release`;
13. blocked state clears only after physical jog keys are released;
14. `stop_request` remains a hard lifecycle path and is not treated as a soft IK rejection.

### HOME

15. after `enter_mode6()`, a transient non-converged sample resets the stable timer rather than reporting success;
16. HOME returns only after one full continuous existing dwell period inside position/velocity convergence;
17. post-Mode6 timeout remains bounded;
18. abort/controller error still fails promptly.

Then run the repository-required checks:

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
git diff --check
```

If optional tooling is unavailable, report it rather than installing/upgrading the experiment environment.

---

## 19. Hardware acceptance plan — document only, do not execute automatically

After offline review passes, report that hardware validation is still required.

Do not run these without explicit authorization.

### Keyboard

- repeated short taps in one direction;
- press → release → press;
- direction reversal;
- hold outward at each relevant workspace boundary;
- rotation while translation is boundary-clipped;
- approach an IK-rejecting/singular region under supervision.

Expected:

- ordinary release does not produce State4/Mode6 restart;
- fixed 8 mm / 0.03 rad behavior is unchanged;
- boundary hold stops producing new IK/publication once the projected pose is unchanged;
- rejected target does not advance the committed Cartesian reference.

### Camera calibration

Repeat the keyboard-style jog tests.

Expected:

- release does not hard-stop the arm;
- recoverable IK reject blocks until physical release but does not revoke the valid Mode6 epoch;
- SPACE capture still requires fresh stationary arm feedback exactly as before.

### HOME

Run HOME from:

- keyboard teleop;
- camera calibration;
- VR collect teleop.

Expected:

- completion is reported only after Mode6 has been restored and the arm has remained within the existing position/velocity convergence bounds for the existing dwell duration;
- no visible “HOME finished, then small extra twitch” should remain from an unobserved mode-switch settling transient.

---

## 20. Definition of done

This task is complete only when all of the following are true:

1. ordinary keyboard and camera-calibration jog idle no longer revoke the active Mode6 epoch;
2. no ordinary key release causes `arm.stop()` through run-id invalidation;
3. keyboard and calibration Cartesian targets are proposal/validate/publish/commit, not pre-committed before IK;
4. repeated outward input at an already committed workspace boundary performs no redundant IK/publication;
5. recoverable calibration IK rejection preserves the last valid command epoch and requires physical release before retry;
6. calibration `stop_request` and all real safety/lifecycle boundaries still revoke correctly;
7. `arm_worker.py` safety semantics remain intact;
8. all current movement/timing parameters remain unchanged;
9. HOME success includes post-Mode6 feedback stability using existing convergence/dwell parameters;
10. VR C pause/stale semantics remain unchanged;
11. recent recording/data-contract behavior remains unchanged;
12. no unnecessary control abstraction, interpolation layer, filter, config knob, or compatibility shim is added;
13. offline checks pass or unavailable tooling is explicitly reported;
14. final diff is reviewed for unrelated edits, dead code, stale comments, and accidental parameter changes.

---

## 21. Final implementation principle

Keep this distinction explicit throughout the patch:

```text
no new valid motion proposal
    ≠
motion authority revoked
```

and:

```text
proposed target
    ≠
committed actuator target
```

For keyboard/calibration, Mode 6 should remain the stable firmware trajectory owner across ordinary jog idle.

For true lifecycle/safety boundaries, `run_id` revocation must remain the authoritative fence.

For HOME, success means the robot is stable **after** Mode 6 is restored, not merely that the Mode-0 path once reached the target.
