# VibeCode Task: Finish Synchronous Policy Deployment Semantics

> **Audience:** Claude Code / coding agents implementing this task in `dexmani_real`.
>
> **Status:** one-off implementation task contract. This belongs under `.codex/tasks/`, not permanent architecture documentation. Remove or archive it after the implementation is merged and stable behavior is reflected in source/README where appropriate.
>
> **Repository safety:** this repository controls physical robots. This task is **source/offline work only**. Do not run policy rollout, teleoperation, replay, homing, calibration, device discovery, camera acquisition, or any command that opens live robot/camera SDKs unless the user separately and explicitly authorizes supervised hardware validation.

---

## 0. Read this first

Before editing anything:

1. Read repository-root `AGENTS.md` completely.
2. Read repository-root `CLAUDE.md`.
3. Run `git status --short` and preserve unrelated user changes.
4. Confirm the current branch/HEAD. The design below was reviewed against commit `72ac8d6d4ece9fff675a1f15fb268d1dfa57ca44`, but **current source is authoritative** if the branch has moved.
5. Inspect the actual producer → transformation → consumer → side-effect path before changing it.

Do not treat this task file as a substitute for source inspection. If current source has already solved part of the task, do not reimplement it.

---

# 1. Objective

The synchronous learned-policy architecture is already fundamentally correct:

```text
current causal observation
        ↓
blocking policy inference
        ↓
PolicySpec.n_action_steps
        ↓
process-local action deque
        ↓
execute one action per policy control tick
        ↓
queue empty → observe and infer again
```

Do **not** redesign that architecture.

This task fixes four remaining semantic problems at their roots:

1. **Policy action admission uses subsystem-heartbeat freshness instead of policy-control freshness.**
2. **A stale/unavailable control state can pause a queued chunk and later resume old actions.**
3. **Recording/evidence failures are promoted to global physical `FAULT`.**
4. **One policy action dispatch reads robot feedback twice, so IK/projection and SafetyGate can use different physical states.**

The final system must optimize for:

```text
simple > clever
explicit ownership > implicit coupling
one source of truth > parallel thresholds
whole-chunk invalidation > partial stale-action scheduling
verified safe shutdown > misclassified physical FAULT
```

The target is a **single-researcher PhD experimental runtime**, not a production policy-serving platform.

---

# 2. Design authority and reference lessons

Use the following reference mechanisms as design guidance, without copying their unrelated complexity.

## 2.1 LeRobot synchronous inference / ACT

Relevant lessons:

- inline blocking inference is valid and simple;
- an action chunk is consumed from a local queue;
- semantic discontinuities such as task/reset changes discard queued actions rather than introducing a timestamp scheduler;
- robot safety may remain below the policy layer.

DexMani consequence:

> Preserve queue-empty blocking inference. Treat loss of trustworthy control feedback as a chunk-continuity break: discard the remaining chunk and replan from a fresh observation.

## 2.2 ManiUniCon synchronized execution

Relevant lesson:

- in synchronized mode, inference latency is a chunk-boundary pause;
- the new chunk is re-anchored after inference and executed as a sequence;
- stale-prefix timestamp pruning belongs to its asynchronous path, not the synchronized baseline.

DexMani consequence:

> Keep `rate.reset()` after blocking inference. Do not add stale-prefix skipping, future timestamps, catch-up, action replacement, or robot-ready/policy-ready handshakes.

## 2.3 StarVLA deployment boundary

Relevant lesson:

- policy/model service and real-time robot/safety infrastructure are separate ownership domains;
- data/recording/model-service failure is not automatically a physical robot failure.

DexMani consequence:

> Distinguish **physical/control safety failure** from **experiment/session failure**. Recording may invalidate the rollout and terminate the session without claiming the robot entered a hardware fault.

---

# 3. Non-goals — do not expand scope

Do **not** implement any of the following:

```text
async/background inference
RTC / real-time chunking
robot_ready / policy_ready Events
per-action ACK barriers
physical convergence waits
future-action timestamp scheduling
stale-prefix skipping
catch-up execution
action replacement / overlapping predictions
new action queue processes
new policy threads
new freshness configuration knobs
new recording transaction protocol
new persisted recording schema
new telemetry framework
changes to dexmani_policy
```

Do not remove or weaken existing:

```text
generation fences
command validity windows
coupled command latest-ticket checks
worker final SDK fences
joint/workspace safety checks
arm/hand command shaping already intentionally owned by the runtime/workers
causal observation construction
fixed wall-clock observation grid
```

Do not redesign action clipping/projection in this task.

---

# 4. Existing behavior that must remain invariant

Preserve these current semantics unless a section below explicitly changes them:

1. `PolicyRunner` is the sole learned-policy scheduling/model owner.
2. New inference occurs only when `self.actions` is empty.
3. `model_runtime.predict(...)` is blocking.
4. A post-inference generation/liveness check discards stale inference results.
5. `self.rate.reset()` occurs after a newly predicted chunk is accepted.
6. At most one action is dispatched per policy control tick.
7. Queue head is popped **only after successful publication / dry-run publishability**.
8. No per-action physical convergence wait exists.
9. Arm/Hand workers retain final SDK authority and final ticket/generation/safety fencing.
10. Observation history remains causal and wall-clock based.
11. `PolicySpec.n_action_steps` remains Policy-owned; Real must not add a second execution horizon.
12. Existing action projection/clipping/SafetyGate behavior remains unchanged in this task.

---

# 5. Root cause A: one dispatch currently consumes two physical states

Current policy dispatch effectively does:

```text
_dispatch_action(action)
    ↓
_decode_action(action)
    ↓
read_arm_state_dict()
diagnose_arm_feedback(...)
    ↓
EE IK / joint decode
    ↓
_project_policy_targets(...)
    ↓
build_action_candidate(...)
    ↓
prepare_command(...)
    ↓
read arm feedback AGAIN
read hand feedback
    ↓
SafetyGate.validate(...)
```

This creates a TOCTOU semantic mismatch:

```text
IK/projection uses arm state at t0
SafetyGate uses arm state at t1
```

It also makes it easy for the two reads to use different freshness policy.

The fix is **not** a lock or cross-worker handshake. The fix is one immutable feedback selection per action dispatch.

---

# 6. Patch A — Policy Feedback Transaction + Chunk Continuity

Implement Patch A first and keep it independently reviewable.

Expected primary files:

```text
dexmani_real/control/publication.py
dexmani_real/deployment/executor.py
```

Do not modify teleop/replay/calibration callers unless compilation proves a minimal compatibility edit is required.

## 6.1 Add `CommandFeedbackSnapshot`

Place it in `dexmani_real/control/publication.py`, close to the existing feedback snapshot/result helpers.

Keep it minimal:

```python
@dataclass(frozen=True)
class CommandFeedbackSnapshot:
    captured_monotonic_ns: int
    arm_qpos: np.ndarray
    arm_source_monotonic_ns: int
    hand_qpos: np.ndarray | None = None
    hand_source_monotonic_ns: int | None = None
```

Do **not** add unrelated fields such as:

```text
accepted_action_id
accepted_monotonic_ns
command progress
worker ACK state
ring sequence diagnostics
```

Those are not required by decode/IK/SafetyGate and would blur ownership.

### Snapshot semantics

`CommandFeedbackSnapshot` does **not** claim arm and hand were sampled atomically.

Its contract is:

> During one command dispatch, select at most one arm sample and one required hand sample, evaluate both against one common `now_ns`, ownership-copy the values, and use those exact samples throughout decode and safety admission.

Do not add a global lock around the two rings.

## 6.2 Add `read_command_feedback(...)`

In `control/publication.py`, create one reusable helper around the existing `_read_arm_feedback` / `read_hand_feedback` logic.

Preferred shape:

```python
def read_command_feedback(
    shared: Any,
    *,
    require_hand: bool,
    arm_max_age_s: float,
    hand_max_age_s: float,
) -> tuple[CommandFeedbackSnapshot | None, str, FeedbackIssue | None]:
    ...
```

Requirements:

- capture `now_ns = time.monotonic_ns()` exactly once;
- read arm ring once;
- validate arm shape/finite/connected/controller/state/timestamp/freshness against that `now_ns`;
- when hand is required, read hand ring once and validate against the same `now_ns`;
- copy qpos arrays into the snapshot so later ring writes cannot mutate the selected state;
- return enough information for the caller to distinguish:
  - temporary/unavailable (`issue is None`),
  - `FeedbackIssueCode.STALE`,
  - fatal feedback issues.

Do not create a second large result hierarchy if the existing `FeedbackIssue` tuple is sufficient.

## 6.3 Learned-policy freshness source of truth

For the **learned-policy execution path only**, use:

```python
runtime.policy.max_input_age_s
```

for both arm and required hand command-feedback freshness.

Do not use:

```python
runtime.safety.heartbeat_timeouts["arm"]
runtime.safety.heartbeat_timeouts["hand"]
```

for policy action admission.

Rationale:

```text
heartbeat timeout
    = subsystem/process liveness budget

policy.max_input_age_s
    = whether physical state is fresh enough for learned-policy control
```

Do not globally replace `runtime.policy.arm_state_stale_threshold_s`; it is used by teleop/replay/calibration and is outside this task.

Do not add another policy feedback freshness config.

## 6.4 `_decode_action()` must become side-effect free with respect to feedback I/O

Change the PolicyRunner helper from conceptually:

```python
_decode_action(action)
```

to:

```python
_decode_action(action, feedback: CommandFeedbackSnapshot)
```

Inside it:

- use `feedback.arm_qpos` as current measured arm state;
- preserve existing action/action_ee semantics;
- preserve current IK behavior;
- preserve current projection behavior;
- preserve current policy-vs-runtime failure classification;
- remove `read_arm_state_dict(...)` and the first `diagnose_arm_feedback(...)` from this helper.

After this change, `_decode_action()` must not read shared state.

## 6.5 Keep `prepare_command()` backward compatible

Do **not** broadly redesign `prepare_joint_command()` or all its teleop/replay/calibration callers.

Instead, extend `prepare_command()` with one optional argument, e.g.:

```python
feedback_snapshot: CommandFeedbackSnapshot | None = None
```

Behavior:

```text
feedback_snapshot is None
    → existing generic behavior: read/validate feedback internally

feedback_snapshot supplied
    → do NOT read arm/hand rings again
    → use the supplied snapshot for SafetyGate current state
```

This allows PolicyRunner to use one physical snapshot while existing generic callers remain unchanged.

Do not change public behavior of `prepare_joint_command()` unless required for compilation.

## 6.6 Re-check freshness of the SAME snapshot before publication

Removing the second ring read must not accidentally weaken publication-time freshness.

A snapshot can be fresh at selection and become stale while EE IK / FK / workspace checks run.

Add a small helper in `control/publication.py`, for example:

```python
def command_feedback_is_fresh(
    snapshot: CommandFeedbackSnapshot,
    *,
    now_monotonic_ns: int,
    arm_max_age_s: float,
    hand_max_age_s: float,
) -> bool:
    ...
```

or an equivalent compact function returning a reason.

Requirements:

- do not read the rings again;
- evaluate `snapshot.arm_source_monotonic_ns` against the new `now_ns`;
- if a hand sample is present, evaluate its source timestamp too;
- malformed/future timestamps should not be silently accepted;
- PolicyRunner must call this after decode/preparation and immediately before physical publication / dry-run publishability.

Target semantics:

```text
read feedback ONCE
    ↓
decode / IK / projection
    ↓
SafetyGate with SAME snapshot
    ↓
re-check SAME snapshot age
    ↓
publish
```

## 6.7 Add a narrow `_invalidate_chunk()`

In `PolicyRunner` add exactly the execution-state invalidation needed for a feedback continuity break:

```python
def _invalidate_chunk(self) -> None:
    self.actions.clear()
    self.previous_arm_command_qpos = None
```

Do not call `_clear_execution()` for this case.

Do not reset:

```text
run_generation
observation_id
model RNG / model_runtime.reset_episode()
recording state
last_recorded_action
run_started_ns
```

This is a replanning boundary, not an episode boundary.

Why clear `previous_arm_command_qpos`:

> After a feedback discontinuity, the next fresh chunk's first action must anchor continuity to the newly measured arm state, not a pre-gap command target.

## 6.8 Feedback failure disposition

Use the existing `FeedbackIssueCode` taxonomy. Do not create a new taxonomy.

Required behavior:

| Feedback result | PolicyRunner behavior |
|---|---|
| healthy | decode → prepare → freshness recheck → publish |
| `read_latest()` unavailable (`issue is None`) | `_invalidate_chunk()`; return; no FAULT |
| `FeedbackIssueCode.STALE` | `_invalidate_chunk()`; return; no FAULT |
| same snapshot becomes stale before publish | `_invalidate_chunk()`; return; no FAULT |
| DISCONNECTED | existing `_fault(...)` |
| CONTROLLER_ERROR | existing `_fault(...)` |
| STATE_INVALID | existing `_fault(...)` |
| MISSING_TIMESTAMP | existing `_fault(...)` |
| FUTURE_TIMESTAMP | existing `_fault(...)` |
| MALFORMED_SHAPE | existing `_fault(...)` |
| NONFINITE | existing `_fault(...)` |

Do not introduce a retry timer, grace counter, action age scheduler, or partial-prefix reuse.

A chunk invalidation is cheap: the next queue-empty iteration naturally builds a fresh causal observation and performs a fresh blocking inference.

## 6.9 Target `_dispatch_action()` shape

The final control flow should be approximately:

```python
def _dispatch_action(self, action: np.ndarray) -> None:
    max_age_s = float(self.runtime.policy.max_input_age_s)

    feedback, reason, issue = read_command_feedback(
        self.shared,
        require_hand=True,  # derive from actual action/policy contract if current source requires
        arm_max_age_s=max_age_s,
        hand_max_age_s=max_age_s,
    )

    if feedback is None:
        if issue is None or issue.code is FeedbackIssueCode.STALE:
            self._invalidate_chunk()
            return
        self._fault(f"fatal command feedback: {issue.code.value}")
        return

    decoded, reject_kind, decode_rejection = self._decode_action(action, feedback)
    if decoded is None:
        ...  # preserve existing IK/rejection semantics
        return

    candidate = build_action_candidate(...)
    ...

    prepared = prepare_command(
        self.shared,
        candidate,
        gate=self.gate,
        arm_feedback_max_age_s=max_age_s,
        hand_feedback_max_age_s=max_age_s,
        feedback_snapshot=feedback,
    )
    ...

    if not command_feedback_is_fresh(
        feedback,
        now_monotonic_ns=time.monotonic_ns(),
        arm_max_age_s=max_age_s,
        hand_max_age_s=max_age_s,
    ):
        self._invalidate_chunk()
        return

    result = publish_command(...)
    ...
    if result.published:
        self.actions.popleft()
```

Adapt names to current source style. Do not mechanically copy this pseudocode if a smaller implementation fits the existing code better.

---

# 7. Patch A acceptance invariants

After Patch A, all of these must be true:

1. One policy action dispatch selects robot feedback only once per required modality.
2. EE IK/projection and SafetyGate consume the same arm state.
3. Learned-policy action admission uses `runtime.policy.max_input_age_s`, not heartbeat liveness timeout.
4. A stale/unavailable control state never leaves the old queue head waiting for later execution.
5. Stale/unavailable control state does **not** become a hardware FAULT by itself.
6. Fatal feedback remains fail-closed.
7. The same selected snapshot is freshness-checked immediately before publication without re-reading the ring.
8. Chunk invalidation also clears `previous_arm_command_qpos`.
9. New inference still happens only when the deque is empty.
10. Publish success still precedes `popleft()`.
11. Existing action shaping/SafetyGate/worker safety remains unchanged.

---

# 8. Root cause B: experiment evidence failure shares the physical fault channel

Current recording-related code can set global `shared.error_state` from multiple layers:

```text
RecorderClient transport failure
PolicyRunner recording failure handling
RecorderIO finalization/process failure
recording-only Camera worker failures
Supervisor worker death / heartbeat timeout
shutdown finalizer nonzero/escalated worker exit
```

This means:

```text
Disk / RecorderIO / recording camera failure
        ↓
error_state = True
        ↓
SafetyState.FAULT
```

That classification is wrong for a research rollout when robot/control integrity is still intact.

The correct semantic is:

```text
experiment evidence invalid
        ↓
fence motion
        ↓
end session with failure status
        ↓
verified shutdown / DISARMED
```

while preserving:

```text
physical/control integrity failure
        ↓
FAULT
```

---

# 9. Patch B — Separate Session Failure from Physical Fault

Implement Patch B separately from Patch A.

Patch B must **not** redesign RecorderIO START/STOP transactions, queues, ring schema, saved episode schema, or finalization protocol.

Expected files after current-source verification:

```text
dexmani_real/ipc/channels.py
dexmani_real/recording/client.py
dexmani_real/recording/io_worker.py
dexmani_real/sensor/camera/worker.py
dexmani_real/deployment/executor.py
dexmani_real/runtime/status.py
dexmani_real/runtime/supervisor.py
dexmani_real/runtime/processes.py
dexmani_real/deployment/lifecycle.py
```

Touch only what is necessary for one coherent failure-domain change.

## 9.1 Add exactly one shared session-result flag

In `RuntimeChannels` add:

```python
session_failed: Any
```

Allocate:

```python
storage.session_failed = ctx.Value("b", False)
```

Semantics:

```text
error_state
    = physical/control/runtime safety integrity is lost

session_failed
    = requested experiment cannot produce a valid result,
      but verified safe shutdown is still possible
```

Do not add separate flags for recorder/camera/storage/etc.

Update comments around `error_state` so its ownership/meaning is no longer described as a generic error sink.

## 9.2 `RecorderClient` must stop writing global `error_state`

`RecorderClient` is a transaction/transport client. Its `_fail_transport()` should:

```text
mark client unavailable
mark recording false
store RecorderStopResult(error=...)
```

It must **not** write:

```python
shared.error_state.value = True
```

Do not change its queues or public transaction protocol.

The PolicyRunner remains responsible for deciding what a recording error means for the policy workflow.

## 9.3 RecorderIO failures are session failures, not physical faults

Inspect all `shared.error_state.value = True` writes inside `recording/io_worker.py`.

For recording/finalization/process failures owned by RecorderIO, replace the global physical-fault side effect with the session-failure domain:

```python
shared.session_failed.value = True
```

The worker may still:

- mark its local `fatal=True`;
- emit/retain the appropriate `RecordingFinished(error=...)` when possible;
- exit nonzero / raise `RuntimeError` on fatal RecorderIO failure.

Do not weaken its storage correctness or resource-release checks.

Important exception:

> Do not weaken generic verified-shutdown protection when a child cannot be confirmed stopped and shared IPC cannot safely be reclaimed. That catastrophic runtime condition may remain fail-closed.

## 9.4 Replace PolicyRunner `_latch_recording_fault()` with failed-session shutdown

Do not create a second recording state machine.

Use one tiny helper, for example:

```python
def _request_failed_session_shutdown(self) -> None:
    self.shared.session_failed.value = True
    self.shared.quit_requested.value = True
```

Keep existing episode invalidation/fencing at the caller that already knows the recording error context.

Examples:

```text
_record_rollout_tick exception
    → existing _invalidate_rollout(... save=False)
    → _request_failed_session_shutdown()

_poll_recorder terminal error
    → existing _invalidate_rollout(... save=False) when active
    → _request_failed_session_shutdown()

terminal recorder START failure before motion
    → _request_failed_session_shutdown()

_complete_recording result.error
    → _request_failed_session_shutdown()
```

Avoid double-calling `_finish_episode()`.

After this patch, recording failure must not set `error_state` in PolicyRunner.

## 9.5 Camera failure is workflow-dependent

Current policy lifecycle starts Camera when either:

```text
Policy observation requires RGB/point_cloud
OR
formal rollout recording requires camera evidence
```

These cases are different:

### Control-critical Camera

Policy uses RGB or point cloud.

```text
Camera loss → policy observation unavailable → control workflow cannot remain valid
```

Treat this as critical/fail-closed under existing semantics.

### Recording-only Camera

Policy does not use camera, but camera is started solely because `recording_config is not None`.

```text
Camera loss → evidence/session failure
           ≠ physical robot failure
```

`camera_loop` currently contains direct `shared.error_state` writes on persistent read/publication failures. Add the smallest workflow-selectable behavior, for example an optional third argument:

```python
def camera_loop(
    shared,
    cfg,
    latch_runtime_fault: bool = True,
) -> None:
    ...
```

Default must remain `True` so teleop and other current callers preserve behavior.

Policy lifecycle passes:

```python
latch_runtime_fault = _requires_camera(policy_spec)
```

When `False`, camera-owned failures that currently latch `error_state` should instead mark:

```python
shared.session_failed.value = True
```

and exit so the supervisor can terminate the failed experiment session.

Do not make Camera silently continue after a persistent failure.

Do not weaken Camera when it is part of the policy observation contract.

## 9.6 Do not encode criticality globally in `ProcessSpec`

Do **not** add a permanent `safety_critical` field to every `ProcessSpec` unless current source makes that strictly necessary.

Criticality is workflow-dependent, especially for Camera.

Keep this policy-deployment-specific by computing:

```python
service_process_names: set[str] = set()

if recording_config is not None:
    service_process_names.add("recorder")
    if not _requires_camera(policy_spec):
        service_process_names.add("camera")
```

Thus:

```text
state-only policy + recording:
    arm/hand/policy = critical
    camera/recorder = experiment services

RGB policy + recording:
    arm/hand/policy/camera = critical
    recorder = experiment service

point-cloud policy + recording:
    arm/hand/policy/camera/pointcloud = critical
    recorder = experiment service
```

## 9.7 Add `ExitReason.SERVICE_FAILURE`

Extend `runtime/status.py` minimally:

```python
SERVICE_FAILURE = ...
```

Do not change meanings of existing enum values unnecessarily.

## 9.8 Make supervisor service-aware with one optional set

Extend `run_supervisor(...)` and/or its pure helper with:

```python
service_process_names: Collection[str] = ()
```

Default empty preserves teleop/replay/calibration behavior.

Required priority:

```text
1. E-stop
2. sticky physical error_state
3. critical worker death
4. critical heartbeat timeout
5. session_failed
6. service worker death / service heartbeat timeout
7. explicit quit
8. none
```

If critical and service failures happen together, critical failure wins.

### Critical failure

Preserve existing behavior:

```text
→ SafetyState.FAULT
→ normal_exit = False
```

### Service failure

Required behavior:

```text
shared.session_failed = True
exit_reason = SERVICE_FAILURE
normal_exit = True
break supervisor loop
```

`normal_exit=True` means only:

> use the verified normal shutdown path rather than forcing SafetyState.FAULT.

It does **not** mean experiment success.

The lifecycle's final success predicate will inspect `session_failed`.

## 9.9 Startup readiness must distinguish critical workers and services

Current policy startup is approximately:

```text
policy READY
↓
start all remaining workers
↓
one readiness wait
↓
any failure → error_state + FAULT
```

Change only the grouping, not the overall architecture:

```text
policy READY
↓
start critical workers
↓
wait critical readiness
↓
start experiment services
↓
wait service readiness
↓
ARMED
```

Rules:

- critical readiness failure → existing `error_state / FAULT` path;
- service readiness failure → `session_failed=True`, clean verified shutdown, return exit code 1;
- never transition to ARMED if requested recording services failed readiness;
- during service readiness wait, if an already-ready critical worker dies or `error_state` becomes true, classify it as critical failure, not service failure.

Keep PolicyRunner-first model warmup behavior unchanged.

## 9.10 Make verified shutdown aware of service processes

Current shutdown finalizer treats any nonzero/escalated child as a physical fault.

Extend policy deployment's shutdown call path with an optional:

```python
service_process_names: Collection[str] = ()
```

Default empty preserves all other workflows.

After processes have been **confirmed stopped**:

```text
failed critical process
    → error_state=True / FAULT

failed service process
    → session_failed=True
    → do not convert to physical FAULT
```

Do not weaken the existing fail-closed path where a child process cannot be confirmed stopped and RuntimeChannels therefore cannot be safely closed/unlinked.

A service process that required terminate/kill but was then confirmed dead may fail the experiment without claiming a physical robot FAULT.

## 9.11 Recording finalization timeout is a session failure

In `deployment/lifecycle.py`, current failure of `_wait_for_rollout_recording(...)` must no longer do:

```python
shared.error_state.value = True
```

It should set:

```python
shared.session_failed.value = True
```

while preserving safe shutdown and exit code 1.

## 9.12 Final session result

The final policy deployment success predicate must include:

```python
not bool(shared.session_failed.value)
```

Desired outcomes:

### Successful rollout / normal Q

```text
session_failed = False
error_state = False
SafetyState = DISARMED
exit code = 0
```

### Recorder / recording-only camera failure

```text
session_failed = True
error_state = False
SafetyState = DISARMED
exit code = 1
```

### Arm / Hand / control-critical Camera / policy-control failure

```text
error_state = True (or equivalent critical fault evidence)
SafetyState = FAULT
exit code = 1
```

---

# 10. Patch B acceptance invariants

After Patch B:

1. `RecorderClient` never independently promotes transport failure to physical `error_state`.
2. RecorderIO storage/finalization failures invalidate the session but do not claim a robot hardware fault.
3. PolicyRunner recording failures request failed-session shutdown, not FAULT.
4. Recording-only camera failure is a session failure.
5. Policy-input camera failure remains critical.
6. Recorder/service worker death or heartbeat timeout exits via `SERVICE_FAILURE`, not FAULT.
7. Critical worker death/heartbeat timeout remains FAULT.
8. Recording service readiness failure prevents ARMED but does not enter FAULT.
9. Recording finalization timeout yields `session_failed=True`.
10. Service-process nonzero shutdown yields failed session, not physical FAULT, once process termination is verified.
11. Unverified child shutdown / unsafe IPC cleanup remains fail-closed.
12. A failed session returns nonzero even when the robot safely reaches DISARMED.
13. Teleop/replay/calibration behavior is unchanged by default.
14. RecorderIO protocol, queues, sample ring, saved schema, and episode transaction semantics are unchanged.

---

# 11. Files expected to change

Use source truth; do not change a file only because it appears in this list.

## Patch A

### `dexmani_real/control/publication.py`

Expected:

- `CommandFeedbackSnapshot`;
- `read_command_feedback(...)`;
- compact same-snapshot freshness helper;
- optional `feedback_snapshot` support in `prepare_command(...)`;
- no broad `prepare_joint_command` redesign.

### `dexmani_real/deployment/executor.py`

Expected:

- `_decode_action(..., feedback)` no longer reads rings;
- policy action admission uses `runtime.policy.max_input_age_s`;
- `_invalidate_chunk()`;
- stale/unavailable feedback invalidates whole pending chunk;
- fatal feedback still faults;
- same-snapshot freshness checked before publication;
- healthy queue/publication behavior remains unchanged.

## Patch B

### `dexmani_real/ipc/channels.py`

- add `session_failed` shared flag;
- clarify `error_state` comment/semantics if needed.

### `dexmani_real/recording/client.py`

- remove global physical-fault side effect from `_fail_transport()`.

### `dexmani_real/recording/io_worker.py`

- recorder-owned failures mark `session_failed`, not `error_state`;
- preserve fatal exit and resource correctness.

### `dexmani_real/sensor/camera/worker.py`

- add a minimal workflow-selectable fault-latching mode with default preserving existing behavior;
- recording-only Policy deployment passes noncritical/service mode.

### `dexmani_real/deployment/executor.py`

- replace recording-fault latch with failed-session shutdown request;
- do not double-finish episodes.

### `dexmani_real/runtime/status.py`

- add `SERVICE_FAILURE`.

### `dexmani_real/runtime/supervisor.py`

- distinguish service vs critical process death/heartbeat timeout;
- service failure uses safe normal shutdown path.

### `dexmani_real/runtime/processes.py`

- verified finalizer distinguishes confirmed service failure from critical failure;
- unverified shutdown remains fail-closed.

### `dexmani_real/deployment/lifecycle.py`

- compute workflow-specific `service_process_names`;
- stage critical/service readiness;
- pass service set to supervisor/shutdown;
- recording finalization failure → `session_failed`;
- final success predicate includes `not session_failed`.

---

# 12. Files/mechanisms that should normally remain untouched

Unless current-source dependencies make a tiny compatibility edit necessary, do not modify:

```text
dexmani_real/deployment/inference/observation.py
dexmani_real/control/safety_gate.py
dexmani_real/robot/arm_worker.py
dexmani_real/robot/hand_worker.py
dexmani_real/ipc/schema.py coupled-command fields
RecorderIO queue/ring transaction schema
Policy API / dexmani_policy integration
PolicySpec ownership
```

Do not turn this into a repository-wide naming/typing cleanup.

---

# 13. Implementation sequence

Follow this order. Do not mix both patches into one unreviewable edit.

## Phase A1 — trace and lock down current contracts

Before editing, inspect:

```text
executor._dispatch_action
executor._decode_action
executor._project_policy_targets
publication._read_arm_feedback
publication.read_hand_feedback
publication.prepare_command
publication.prepare_joint_command
SafetyGate.validate
```

Search all callers of:

```text
prepare_command
prepare_joint_command
read_hand_feedback
```

Confirm teleop/replay/calibration compatibility requirements.

## Phase A2 — implement feedback snapshot

1. Add minimal snapshot type/helper.
2. Make `prepare_command` optionally consume it.
3. Modify PolicyRunner to read once and pass snapshot through.
4. Add final same-snapshot freshness recheck.
5. Add narrow chunk invalidation.
6. Re-read diff specifically looking for any remaining second PolicyRunner feedback read.

## Phase A3 — validate Patch A before touching recording

Run only offline checks described below.

Then inspect the diff and confirm no recording/lifecycle semantics changed accidentally.

## Phase B1 — introduce session failure domain

1. Add `session_failed`.
2. Remove RecorderClient global fault side effect.
3. Convert RecorderIO-owned errors to session failure.
4. Convert PolicyRunner recording failure to failed-session shutdown.

## Phase B2 — make policy lifecycle service-aware

1. Add `SERVICE_FAILURE`.
2. Compute `service_process_names` in policy lifecycle.
3. Make supervisor classify service death/heartbeat separately.
4. Split critical/service readiness.
5. Make shutdown finalization service-aware.
6. Make recording-only Camera use service failure semantics.
7. Update final success predicate.

## Phase B3 — final cross-boundary review

Trace both complete paths:

```text
RecorderIO failure
→ session_failed
→ supervisor/lifecycle
→ motion fence
→ verified shutdown
→ DISARMED
→ exit 1
```

and:

```text
Arm/Hand/control-critical Camera failure
→ error_state / critical worker failure
→ supervisor
→ FAULT
→ exit 1
```

If either path can silently return 0 or misclassify safety state, the task is not complete.

---

# 14. Offline validation requirements

Follow `AGENTS.md`: do not invent a test framework merely to satisfy a checklist.

First run the repository-wide low-cost checks:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Before running any Python snippet that imports repository modules, inspect the import path for hardware side effects. Prefer static/source validation when uncertain.

If focused offline test infrastructure already exists, add/run the smallest relevant tests. If no test infrastructure exists, do not create a broad new harness; use minimal pure/offline checks only where safe.

## Patch A scenarios to verify

At minimum reason through or test these cases with fake/shared in-memory state if an existing safe pattern exists:

### A. Healthy action

```text
fresh arm + fresh hand
→ one snapshot
→ decode/IK
→ SafetyGate same snapshot
→ final freshness true
→ publish
→ popleft exactly once
```

### B. Arm ring changes between decode and prepare

```text
snapshot q0 selected
ring later becomes q1
→ decode uses q0
→ SafetyGate uses q0
→ no second ring read
```

### C. Snapshot ages out during expensive decode/IK

```text
snapshot initially fresh
→ work takes long enough
→ final same-snapshot freshness fails
→ no publish
→ actions cleared
→ previous_arm_command_qpos = None
→ no FAULT
```

### D. Feedback unavailable

```text
no readable required feedback
→ invalidate whole pending chunk
→ no publish
→ no FAULT
```

### E. Feedback stale

```text
FeedbackIssueCode.STALE
→ invalidate whole pending chunk
→ no publish
→ no FAULT
```

### F. Fatal feedback

For disconnected/controller-error/state-invalid/future-timestamp/nonfinite/malformed:

```text
→ existing fault path
→ no attempt to replan through an unsafe hardware condition
```

### G. Episode/generation semantics

Confirm chunk invalidation does not reset:

```text
run_generation
observation_id
model episode RNG
recording transaction
```

## Patch B scenarios to verify

### H. RecorderClient transport failure

```text
client unavailable/error result
→ error_state remains false
→ PolicyRunner eventually marks session_failed / requests shutdown
```

### I. RecorderIO crash/finalization failure

```text
session_failed = True
→ recorder may exit nonzero
→ supervisor classifies service failure
→ no SafetyState.FAULT solely because recorder failed
```

### J. Recording-only camera failure

For a state-only policy:

```text
camera service failure
→ session_failed
→ SERVICE_FAILURE
→ verified shutdown
→ DISARMED
→ exit 1
```

### K. Policy-input camera failure

For RGB/point-cloud policy:

```text
camera failure
→ critical failure
→ FAULT
→ exit 1
```

### L. Critical worker failure during service readiness

```text
service startup in progress
arm/hand/policy dies
→ must classify critical failure
→ FAULT
```

### M. Normal user quit

```text
session_failed = False
error_state = False
→ verified shutdown
→ DISARMED
→ exit 0
```

### N. Failed session but physically safe

```text
session_failed = True
error_state = False
all critical workers stop safely
→ DISARMED
→ exit 1
```

### O. Unverified process shutdown

```text
child cannot be confirmed stopped
→ preserve existing fail-closed resource/safety behavior
```

---

# 15. Code quality constraints

Claude Code should actively reject its own implementation if it introduces any of these smells:

```text
new duplicate freshness thresholds
new per-action timestamps
new retry state machine for feedback
new ACK protocol
new background threads for policy scheduling
new global ProcessSpec policy that affects unrelated workflows
service/critical classification duplicated in many modules
Recorder protocol changes unrelated to failure classification
broad refactors around otherwise-stable code
```

Prefer:

```text
one small immutable snapshot
one existing freshness source of truth
one narrow chunk invalidation helper
one shared session_failed flag
one workflow-local service_process_names set
optional parameters with backward-compatible defaults
```

Comments should explain invariants and ownership, not narrate implementation history.

---

# 16. Final review checklist before handoff

Before finishing, inspect the focused diff and answer these questions from source:

## Feedback/control

- Does PolicyRunner read arm feedback more than once per action dispatch?
- Does PolicyRunner read hand feedback more than once per action dispatch?
- Can SafetyGate receive a different arm qpos than EE IK used?
- Can a stale queue head survive a feedback continuity break?
- Can a fresh-at-read snapshot become stale before publish without being caught?
- Does stale/unavailable feedback incorrectly become FAULT?
- Do fatal hardware feedback failures still fail closed?
- Is queue head still popped only after successful publication?

## Recording/lifecycle

- Can `RecorderClient` still set `error_state`?
- Can `RecorderIO` storage failure still set physical `error_state`?
- Can a recording-only Camera directly latch physical `error_state`?
- Can recorder death still be classified as generic critical `WORKER_DEATH`?
- Can service heartbeat timeout still become physical FAULT?
- Can recording readiness failure still enter FAULT before motion begins?
- Can recording finalization timeout still set `error_state`?
- Can a failed recording session accidentally return exit code 0?
- Can a service worker shutdown failure be safely distinguished only after process termination is verified?
- Are policy-input Camera/PointCloud failures still critical?

## Scope

- Did the patch change action clipping/projection? It should not.
- Did the patch change observation-grid logic? It should not.
- Did the patch add async inference/ACK/timestamp scheduling? It must not.
- Did the patch modify `dexmani_policy`? It must not.
- Did the patch alter teleop/replay/calibration behavior unintentionally? It must not.

---

# 17. Required handoff report

At completion, report:

1. exact files changed;
2. concise description of Patch A behavior;
3. concise description of Patch B behavior;
4. which offline checks were run and their actual results;
5. anything not validated because it requires CUDA/hardware/live devices;
6. any deviation from this task contract and the source-level reason;
7. final `git status --short`.

Do not claim real-robot validation unless real hardware was explicitly authorized and actually exercised.

---

# 18. Definition of done

The task is complete only when the code expresses these three invariants directly:

```text
ONE ACTION DISPATCH
    → ONE IMMUTABLE CONTROL-FEEDBACK SELECTION

FEEDBACK CONTINUITY BREAK
    → DISCARD THE WHOLE UNEXECUTED CHUNK
    → FRESH OBSERVATION + FRESH INFERENCE

EXPERIMENT EVIDENCE FAILURE
    → FAILED SESSION + SAFE DISARM
    ≠ PHYSICAL ROBOT FAULT
```

Everything else in the synchronous policy architecture should remain as simple as it is today.
