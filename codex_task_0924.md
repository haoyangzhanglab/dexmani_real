# Codex Task 0924 — DexMani XHand Safety and Runtime Lifecycle Tightening

## 0. Task intent

This task applies a **small, targeted safety/correctness patch** to `dexmani_real`.

The repository is a personal PhD research codebase, not a production robotics framework. Keep the current architecture. Do **not** introduce new middleware, new IPC message types, new safety states, command acknowledgements, generic lifecycle abstractions, strict timing frameworks, or unrelated cleanup.

Implement only the following reviewed items:

1. **P0-1 — XHand physical revoke semantics**
   - Normal motion revocation must actively stop the previous XHand position target by holding the latest measured pose.
   - E-stop / final shutdown must put XHand into vendor passive mode.
2. **P0-2 — XHand HOME measured convergence**
   - HOME success must require measured joint convergence.
   - Use a deliberately loose **5 degree maximum per-joint error tolerance** because XHand positioning accuracy is limited.
3. **P0-3 — XHand `CRC_UNCONFIRMED` semantics**
   - A CRC-unconfirmed send is not equivalent to a rejected command and must not kill the hand worker.
4. **P1-1 — Single owner for policy episode timeout**
   - `PolicyRunner` owns `max_running_s`.
   - Main/session supervision must not independently time out a RUNNING policy episode.
5. **P2-1 — Camera timestamp semantics**
   - The realtime camera timestamp used for freshness must be the host-monotonic frame-return timestamp already captured immediately after `wait_for_frame()`, not the later post-processing completion time.
6. **P3 — Only cleanup directly caused by the changes above**
   - Remove fields/parameters/comments/imports that become dead specifically because of this task.
   - Do not perform a general cleanup sweep.

Everything else from prior review is explicitly **out of scope**.

---

## 1. Architectural constraints

Preserve the existing core architecture:

- `SafetyState = DISARMED / ARMED / RUNNING / FAULT`.
- `run_id` remains the stale-command epoch fence.
- Arm, hand, sensor, policy, and recorder worker separation remains unchanged.
- Shared-memory latest-value rings remain unchanged.
- Recorder transaction semantics remain unchanged.
- `RLock` / `motion_lock` behavior remains unchanged.
- Do not add a robot-ready/policy-ready action handshake.
- Do not add per-command ACK/completion tracking.
- Do not add a STOP command to `RobotCommand`.
- Do not add another shared-memory state for XHand stop/hold.
- Do not change point-cloud processing, IK behavior, replay behavior, dataset provenance, timing admission, packaging, or unrelated P3 cleanup.

The implementation should follow the existing xArm worker pattern where useful: the worker remembers which motion authority may still be physically active and reacts locally when that authority is revoked.

Reference design intuition:
- `arm_worker.py` already tracks a streaming epoch and calls `arm.stop()` after authority loss.
- PI-R2's XHand implementation uses vendor modes `0=passive`, `3=position`; use the same hardware-level concept without importing its broader architecture.

---

## 2. P0-1 — XHand physical revoke semantics

### 2.1 Driver primitives

Target file:

- `dexmani_real/robot/drivers/xhand.py`

Add:

```python
_PASSIVE_MODE = 0
_POSITION_MODE = 3  # existing
```

Provide two explicit driver-level semantics:

```python
def hold_current(self, qpos: np.ndarray) -> XHandSendStatus:
    """Actively hold the supplied measured joint pose in position mode."""
    ...

def set_passive(self) -> XHandSendStatus:
    """Put every XHand joint into vendor passive mode."""
    ...
```

Keep these operations thin. Do not create a stop protocol or extra state machine.

For `hold_current(qpos)`, treat the input as measured feedback rather than a new research/control target:

- require the normal finite `(12,)` shape;
- clip the measured pose to configured mechanical limits before sending;
- then send it in POSITION mode.

This prevents a tiny feedback/calibration overshoot beyond a rated limit from turning a safety hold into a rejected command. Do not loosen validation for normal `send_action()`; the clipping exception is only for the worker-local measured-pose hold primitive.

### 2.2 Centralize SDK send-status decoding

The existing `send_action()` already maps XHand SDK return codes into:

- `ACCEPTED`
- `CRC_UNCONFIRMED`
- `REJECTED`

Refactor only as much as needed so `send_action()`, `hold_current()`, and `set_passive()` share the same return-code interpretation.

A suitable shape is a small private helper that submits the current `self._command` and returns `XHandSendStatus`.

Important:

- `send_action(target)` must explicitly set every commanded joint back to `_POSITION_MODE`.
- `hold_current(qpos)` must also result in position mode.
- This is required because `set_passive()` mutates joint modes; a later resumed command must not leave the hand in passive mode while only changing position fields.

Do not duplicate CRC/error-code interpretation in the worker.

### 2.3 Worker ownership of physical revoke

Target file:

- `dexmani_real/robot/hand_worker.py`

Do **not** add a stop queue or new IPC message.

Track the authority that may still have a physically active XHand target:

```python
active_motion_authority: tuple[int, SafetyState] | None = None
```

The tuple is required. Tracking only `run_id` is incorrect because:

- normal streaming targets require `SafetyState.RUNNING`;
- HOME targets require `SafetyState.ARMED`.

When a target is actually submitted to the SDK and the result is either:

- `ACCEPTED`, or
- `CRC_UNCONFIRMED`

record:

```python
active_motion_authority = (run_id, required_safety_state)
```

A CRC-unconfirmed command must be treated as **possibly delivered**, therefore it may still be physically active.

### 2.4 Revoke behavior

On each hand-worker iteration, after attempting the state read but **before processing HOME requests or new RobotCommands**, detect whether the stored authority is still valid using the existing:

```python
command_may_cross_sdk(
    shared,
    run_id=epoch,
    required_safety_state=required_state,
)
```

If authority has been revoked:

1. Use the latest valid measured XHand `qpos`.
2. Call `hand.hold_current(qpos)`.
3. If hold returns `ACCEPTED`:
   - clear `active_motion_authority`;
   - skip normal target processing for that iteration.
4. If hold returns `CRC_UNCONFIRMED`:
   - **do not clear** `active_motion_authority`;
   - allow the normal worker loop to retry hold on the next tick using the newest available feedback.
5. If hold returns `REJECTED`:
   - raise a runtime error so the existing worker fault path handles it.

Do not add retry counters or retry queues. The existing ~30 Hz worker loop is the retry mechanism.

Maintain the latest valid measured pose locally together with its host-monotonic feedback timestamp, for example:

```python
last_valid_qpos: np.ndarray | None
last_valid_qpos_timestamp_ns: int | None
```

The revoke check must run even when the current `get_state()` call returns `None`. Do not structure the loop so that a transient read failure `continue` skips physical revoke handling.

When authority is revoked:

- prefer the qpos from the current successful state read;
- otherwise use the cached qpos only if it is still fresh under the existing `config.feedback_max_age_s` semantics;
- do not perform a new blocking SDK read solely for revoke handling.

If no fresh-enough measured pose is available, fall back to `set_passive()` rather than commanding an arbitrarily stale pose.

For this normal-revoke passive fallback:

- `ACCEPTED`: clear `active_motion_authority`;
- `CRC_UNCONFIRMED`: keep `active_motion_authority` so the next worker tick retries the safety action;
- `REJECTED`: raise into the existing worker fault path.

The cached-pose path is specifically for brief read dropouts; it must not convert a long sensor outage into a command toward an old pose.

### 2.5 E-stop and shutdown

Behavior distinction:

- Normal pause/stop/timeout/authority revocation: **hold current position** when fresh-enough measured feedback is available; otherwise use passive as the safety fallback.
- E-stop: **set passive immediately**, then exit the worker loop.
- Final worker teardown: perform best-effort passive before closing the SDK handle.

Keep lifecycle actuation in the **hand worker**, not hidden inside `XHand.disconnect()`. Leave `disconnect()` as resource/device teardown so partial-connect error handling stays simple.

A small worker-local best-effort passive helper is acceptable. For E-stop/final cleanup, one best-effort passive call at each lifecycle site is enough:

- `ACCEPTED`: continue teardown;
- `CRC_UNCONFIRMED`: log the uncertainty and continue teardown;
- `REJECTED` or exception: log the cleanup failure and continue teardown.

Do not add passive retry counters or a teardown protocol. Do not allow a passive-send failure during `finally` to mask an already-active exception.

### 2.6 Important behavioral invariant

After a normal revoke:

```text
old endpoint may have been moving
        ↓
run_id / required SafetyState becomes invalid
        ↓
hand worker sends current measured qpos as POSITION target
        ↓
old endpoint is replaced by an approximate hold target
```

No global safety-state extension is required.

---

## 3. P0-3 — Correct `CRC_UNCONFIRMED` semantics

Target files:

- `dexmani_real/robot/drivers/xhand.py`
- `dexmani_real/robot/hand_worker.py`

The current driver contract says CRC delivery uncertainty is nonfatal, but the worker currently treats every status other than `ACCEPTED` as a fatal send failure. Fix this contradiction.

Simplify `_send_target()`:

- remove the `accepted` argument;
- return `XHandSendStatus | None`;
- return `None` when motion authority was lost before the SDK call;
- raise only for unsafe target validation or `REJECTED`;
- return `ACCEPTED` or `CRC_UNCONFIRMED` unchanged.

Conceptually:

```python
status = hand.send_action(target)

if status is XHandSendStatus.REJECTED:
    raise RuntimeError(...)

return status
```

Do not add another warning in the worker for `CRC_UNCONFIRMED`; the driver already logs the SDK-level diagnostic.

### HOME + CRC behavior

For HOME submission:

- `REJECTED` => HOME command submission fails.
- `ACCEPTED` => proceed to measured convergence.
- `CRC_UNCONFIRMED` => also proceed to measured convergence.

This is intentional: measured convergence becomes the ground truth. If the uncertain command was not actually delivered, HOME will simply fail to converge before the existing timeout.

---

## 4. P0-2 — XHand HOME measured convergence

Target files:

- `dexmani_real/config/defaults.py`
- `dexmani_real/robot/hand_homing.py`

### 4.1 Configuration

Add one HandParams field only:

```python
home_tolerance_deg: float = 5.0
```

Validate that it is finite and strictly positive.

Do not add configurable convergence-frame count, velocity threshold, settle time, or a new home state machine.

Use a module constant:

```python
_HOME_CONSECUTIVE_SAMPLES = 3
```

### 4.2 Semantics

Change the meaning of successful `home_hand()` from:

> SDK command submission succeeded.

to:

> The HOME command was not rejected and measured XHand feedback converged within 5 degrees on every joint for three consecutive new fresh samples.

The 5-degree tolerance is deliberate because XHand position-control accuracy is limited.

### 4.3 Where convergence belongs

Keep convergence orchestration in `home_hand()` / Main-side code.

Do **not** block the hand worker in a convergence loop. The worker should remain an SDK owner that continuously publishes feedback.

Flow:

```text
home_hand()
  → revoke prior authority
  → enqueue HOME target under an ARMED epoch
  → worker submits target
  → worker reports submission result
  → home_hand() waits on shared hand_state_ring
  → measured convergence decides final HOME success
```

### 4.4 Convergence algorithm

After the worker reports a non-rejected HOME submission:

1. Take a local `boundary_ns = time.monotonic_ns()`.
2. Poll `shared.hand_state_ring.read_latest()`.
3. Count only **new** samples:
   - timestamp must be greater than `boundary_ns`;
   - timestamp must differ from the previously counted sample;
   - qpos must be finite;
   - sample should satisfy the existing hand feedback freshness semantics.
4. Compute:

```python
max_error_rad = np.max(np.abs(qpos - target))
```

5. Require:

```python
max_error_rad <= np.deg2rad(cfg.home_tolerance_deg)
```

for **3 consecutive new samples**.
6. Any sample outside tolerance resets the consecutive counter.
7. Abort immediately if:
   - the caller's `abort_requested()` becomes true;
   - the HOME epoch loses `SafetyState.ARMED` authority;
   - runtime/estop/error semantics already used by the current HOME path indicate interruption.
8. Use the **existing total `home_timeout_s` budget**. Do not silently create one timeout for submission plus another full timeout for convergence.

On convergence timeout or abort, revoke the HOME epoch using the existing lifecycle helper and return a failed `HomeResult`.

### 4.5 Timeout

Keep the existing:

```python
home_timeout_s = 1.0
```

for this task.

Do not loosen the 5-degree tolerance just to compensate for slow movement. If real hardware proves 1 second is too short while the hand is still converging normally, that is a separate follow-up tuning change.

### 4.6 Interaction with P0-1

Do not add separate HOME-stop logic.

If HOME is aborted or times out, the normal `run_id`/authority revocation must make the hand worker detect the stale ARMED authority and execute `hold_current()`.

This is an important integration invariant.

---

## 5. P1-1 — Make PolicyRunner the sole policy episode timeout owner

Target files:

- `dexmani_real/deployment/session.py`
- `dexmani_real/deployment/runner.py`
- `dexmani_real/runtime/safety.py`
- `dexmani_real/ipc/channels.py`

### 5.1 Remove Main/session timeout enforcement

Delete the `max_running_s` timeout block from the Main/session supervision loop in `deployment/session.py`.

Main remains responsible for:

- child-process health;
- operator-thread health;
- E-stop;
- quit;
- verified shutdown.

`PolicyRunner` remains responsible for:

- episode begin;
- `max_running_s`;
- action execution;
- episode finish;
- recorder episode finish.

Do not replace the deleted Main timeout logic with another watchdog.

### 5.2 Preserve the correct termination reason

Extend `PolicyRunner._finish()` minimally so the caller may provide a `RunEndReason` for revocation.

For the timeout paths, call it with:

```python
run_end_reason=RunEndReason.TIMEOUT
```

so that a runner-owned timeout records:

```text
shared.run_ended_reason == TIMEOUT
```

instead of the current default `EXECUTOR_BOUNDARY`.

Do not redesign the rest of the `RunEndReason` taxonomy in this task.

### 5.3 Remove dead shared start timestamp

After Main no longer consumes it, `shared.run_started_monotonic_ns` has no remaining reader.

Remove it from:

- `RuntimeChannels` declarations/comments;
- shared-memory/value allocation;
- `_begin_motion_locked()` shared write;
- `revoke_motion()` shared reset;
- affected smoke fixtures/assertions.

Keep the local monotonic timestamp returned from `_begin_motion_locked()`:

```python
return run_id, started_ns
```

because `PolicyRunner` already stores it in:

```python
self.started_ns
```

and should continue to use that local value for `max_running_s`.

---

## 6. P2-1 — Camera realtime timestamp semantics

Target file:

- `dexmani_real/sensor/camera/realsense.py`

The RealSense path already captures:

```python
wait_return_monotonic_ns
```

immediately after the frame wait returns.

Later, after alignment and RGB/depth copies, it currently takes a second:

```python
timestamp_ns = time.monotonic_ns()
```

and publishes that as the realtime camera timestamp.

Change the `RGBDFrame.timestamp_ns` used by the camera worker/ring to:

```python
timestamp_ns = wait_return_monotonic_ns
```

Rationale:

- arm/hand/VR freshness is expressed in host monotonic time;
- the camera freshness timestamp should approximate host frame availability, not post-processing completion;
- device timestamps remain diagnostic metadata and should not replace the host-monotonic timestamp;
- do not add capture/receive/process/publish timestamp fields.

Keep `wait_return_monotonic_ns` itself if it is already useful diagnostic metadata.

---

## 7. P3 — Scope-limited cleanup only

Perform only cleanup directly caused by Sections 2–6.

Expected cleanup includes:

- remove `accepted` from `_send_target()`;
- remove `run_started_monotonic_ns`;
- update the obsolete `hand_homing.py` module docstring that currently says success only means SDK-send success;
- remove imports/comments/assertions that become unused because of these exact changes;
- update focused smoke tests for changed semantics.

Do **not** touch unrelated cleanup candidates, including:

- `camera_present`;
- `StopRequest`;
- `EefTargetProposal`;
- EMA clipping;
- validator naming;
- legacy episode sidecar handling;
- duplicate raw metadata;
- point-cloud code;
- IK code;
- replay code;
- dataset provenance;
- packaging;
- CI layout.

---

## 8. Required software verification

Follow the repository's existing testing convention. Do **not** create a committed `tests/` directory.

Update the existing focused offline smoke checks only where needed.

At minimum verify the following invariants with mocks/fakes where practical:

### XHand send semantics

1. `ACCEPTED` normal send succeeds.
2. `CRC_UNCONFIRMED` normal send does **not** raise or kill the worker path.
3. `REJECTED` remains fatal.
4. A command that may have been delivered under CRC uncertainty still records active motion authority.
5. `send_action()` after `set_passive()` restores POSITION mode.

### Physical revoke

6. A RUNNING target creates active authority `(run_id, RUNNING)`.
7. A HOME target creates active authority `(run_id, ARMED)`.
8. Revoking the stored authority causes `hold_current()`.
9. Successful hold clears active authority.
10. CRC-unconfirmed hold leaves active authority set so the next worker tick retries.
11. Rejected hold enters the existing failure path.
12. A transient state-read failure does not skip revoke handling when a fresh-enough cached qpos exists.
13. A stale/missing cached qpos uses passive fallback rather than commanding an old hold pose.
14. `hold_current()` clips measured feedback to mechanical limits while normal `send_action()` remains strict.
15. E-stop invokes passive behavior rather than trying to continue normal motion.

### HOME convergence

16. Three distinct fresh samples with maximum joint error <= 5 degrees succeed.
17. A sample with maximum joint error > 5 degrees resets the consecutive counter.
18. Re-reading the same ring sample must not count as multiple convergence samples.
19. Abort/authority loss fails HOME.
20. Timeout fails HOME.
21. CRC-unconfirmed HOME submission may still succeed if measured feedback actually converges.

### Policy timeout

22. PolicyRunner timeout revokes motion with `RunEndReason.TIMEOUT`.
23. Main/session no longer independently enforces `max_running_s`.
24. `run_started_monotonic_ns` no longer exists.
25. Runner still measures episode duration from its local `self.started_ns`.

### Camera timestamp

26. The realtime `RGBDFrame.timestamp_ns` equals the captured host `wait_return_monotonic_ns`, not a later processing timestamp.

Do not build elaborate test infrastructure solely to test one assignment if doing so would increase complexity more than the implementation.

---

## 9. Required real-hardware validation

Software checks cannot prove XHand physical behavior. After implementation, manually validate on the real hand.

Required scenarios:

1. **RUNNING → normal Stop/Pause**
   - command a visible finger motion;
   - revoke while the hand is still travelling;
   - verify the old endpoint is replaced by an approximate hold near the current pose.
2. **HOME → abort**
   - start HOME;
   - abort before arrival;
   - verify the hand stops progressing toward the old HOME target and holds.
3. **HOME convergence**
   - verify HOME is accepted when all joints settle within approximately 5 degrees for three feedback samples;
   - inspect whether `home_timeout_s=1.0` is practically sufficient, but do not change it as part of this task unless implementation cannot function at all.
4. **E-stop / final shutdown**
   - verify XHand enters passive mode.
5. **Resume after passive**
   - after a new valid session/command, verify normal position control is restored; this specifically validates that `send_action()` resets joint modes to POSITION.
6. **CRC behavior if reproducible**
   - verify a single CRC-unconfirmed response does not kill the worker.

If passive mode creates an unacceptable physical hazard for the actual experimental setup (for example, dropping an object is more dangerous than holding it), report that hardware observation rather than adding a new lifecycle framework. The normal software structure should remain unchanged.

---

## 10. Definition of done

The task is complete when all of the following are true:

- XHand has explicit, minimal `hold_current()` and passive hardware semantics.
- Normal authority loss physically replaces the prior XHand endpoint with a hold target when fresh-enough measured feedback exists, with passive fallback when it does not.
- Revoke handling still runs across transient state-read failures using only fresh-enough cached feedback.
- E-stop/shutdown requests passive behavior.
- Resumed position commands explicitly restore position mode.
- `CRC_UNCONFIRMED` no longer crashes the hand worker.
- HOME success means measured convergence within **5 degrees per joint for 3 distinct fresh samples**.
- HOME abort/timeout composes with the same physical-revoke logic rather than introducing separate stop machinery.
- `PolicyRunner` is the only owner of policy episode timeout.
- Runner-owned timeout records `RunEndReason.TIMEOUT`.
- `run_started_monotonic_ns` has been removed.
- Camera realtime freshness timestamp uses `wait_return_monotonic_ns`.
- No unrelated architecture, validation, point-cloud, IK, replay, provenance, packaging, or cleanup work is included.
- The resulting implementation is smaller or only minimally larger than the current one and preserves the repository's PhD-research-code character.

## 11. Implementation style

Prefer:

```text
delete → inline → small private helper → minimal new state
```

Avoid:

```text
new framework → new protocol → generic abstraction → future-proofing
```

Make the patch easy to audit against the concrete invariants above.
