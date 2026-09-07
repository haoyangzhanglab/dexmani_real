# DexMani Real — Formal Policy Eval Follow-up Repair Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Reviewed code baseline: `cfa446fe536c95078e28a663ef919e13f1fc0b07`  
> Review-hardened guide baseline: `dd2f3d6ac4304b93415fcfc43b07ae3890e8839c`  
> Intended executor: Claude Code / Codex  
> Scope: finish the remaining **formal real-eval transaction correctness** work after the canonical rollout cleanup. Keep the patch local, deterministic, non-blocking, and hardware-independent.
>
> This guide supersedes the formal-eval follow-up details in `docs/policy_rollout_simplification_guide.md`. The earlier guide remains authoritative for the already-stable Policy contract, causal observation path, ActionChunk scheduling, and learned-policy reject-only arm semantics.

---

## 1. Freeze the already-correct rollout semantics

The previous cleanup correctly stabilized these boundaries. **Do not redesign them in this task**:

```text
Policy public contract
→ current PolicySpec only
→ no temporal_ensemble_coeff

Observation freshness
→ max_input_age_s / skew / grid lag before inference

Action temporal validity
→ logical timestamps
→ stale-prefix skip
→ whole-stale discard
→ no catch-up

Command delivery validity
→ action_validity_s
→ publication + worker final guards

Learned-policy arm
→ nearest-equivalent canonicalization
→ joint bounds
→ per-joint jump reject
→ never silent clip

Failure class
→ true EE IK failure = IK_FAIL
→ post-IK / joint / workspace / hand rejection = SAFETY_REJECT
```

Preserve:

```text
shared-memory rings
causal observation alignment
inference worker separation
Prediction latest-wins ownership
run_generation
sync + async modes
executor_poll_hz
coupled arm+hand publication
SafetyGate
physical_home_completed / home sequence
workspace/collision checks
command-progress watchdog
first-command / silence / action-apply timeouts
RecorderIO process architecture
supervisor / heartbeat / process-death handling
robot workers
IPC schemas
raw / processed / Policy-Zarr / calibration schema versions
```

Do not add:

```text
Temporal Ensemble
RTC
steps_per_inference
chunk blending
new scheduler
blocking realtime acceptance wait
new recorder/event framework
compatibility/version machinery
```

---

## 2. Verified remaining issues

The following are confirmed against the reviewed implementation.

| Priority | Issue | Required correction |
|---|---|---|
| P0 | eval/recording failure can revoke RUNNING after IPC publication but before arm+hand SDK acceptance | defer eval-only motion fence until the outstanding published command is accepted |
| P1 | `_build_evaluation_frame_inputs()` has only a `fatal` bool and mixes arm/hand control faults with camera/FK recording issues | use a small typed evidence-failure disposition; preserve control-critical fault ownership |
| P1 | single-frame camera reset/duplicate/delivery-quality problems can turn initial eval startup into global `_fault()` | treat recording-only/transient camera evidence as retryable or INVALID, not robot FAULT |
| P1 | `PreparedCommand.unavailable` is non-fatal in normal run but becomes `_fault()` only in formal eval | make normal/eval behavior identical |
| P1 | EE IK-success followed by arm admission rejection records `ik_ok=False` | fix audit flags without changing control classification |
| P1 | rejected terminal step may finish at `max_action_steps` before rejection evidence is recorded | separate step commit from episode finish |
| P1 | formal `max_frames` is renamed only after RecorderIO already began finalization with `reason="max_frames"` | set the formal reason at RecorderIO's original auto-finalization boundary |
| P1 | pending eval INVALID has no fully specified interaction with operator stop, task timeout, recorder integrity, and an already-pending action-step termination | separate motion precedence, evaluation disposition, and sticky storage-integrity decision |
| P1 | wall-clock timeout while the initial sample is still pending is currently reported as task `FAILURE` although no Policy command has run | classify pre-first-command evidence timeout as `INVALID` |
| P2 | post-control evidence uses a fresh wall-clock anchor after publication/rejection | use the control-event timestamp for deterministic causal audit rows |

The P2 item is an audit-quality optimization. Complete P0/P1 first if it becomes unexpectedly invasive.

---

## 3. Formal-eval transaction model

Keep four boundaries distinct:

```text
1. CONTROL DECISION
   decode / IK / safety → publish or reject

2. IPC PUBLICATION
   coupled command enters the latest-wins ring

3. ACTUATOR ACCEPTANCE
   arm worker accepts its exact endpoint
   hand worker accepts the exact logical target action_id

4. EVALUATION EVIDENCE
   the audit row enters RecorderIO
```

Important existing fact:

- arm acceptance is the successful xArm SDK servo call for the endpoint;
- hand `accepted_target_action_id` advances only when the exact logical hand target has been accepted by the SDK;
- intermediate hand slew setpoints update `last_sdk_setpoint_accepted_monotonic_ns`, so the existing command-progress watchdog observes continued hand progress while waiting for the exact target.

### Successful action

```text
control decision
→ publish coupled command
→ commit scheduler/control bookkeeping
→ attempt evaluation evidence
→ evidence healthy: continue normal rollout
→ evidence invalid: publish no later Policy action
→ keep normal executor polling / progress watchdog alive
→ wait until arm+hand acceptance covers this action_id
→ then fence to ARMED and finish INVALID
```

### Rejected action

```text
control decision = IK_FAIL or SAFETY_REJECT
→ consume the control slot
→ commit step/prediction bookkeeping
→ record rejection evidence
→ only then apply action-step or eval-invalid termination
```

A rejected action has no actuator acceptance to wait for.

### Startup

```text
B
→ Recorder START ACK
→ RUNNING generation
→ initial causal held sample
→ first Policy inference/command
```

No Policy command may run before the initial sample succeeds.

---

## 4. P0 — non-blocking acceptance-fenced eval invalidation

### Why immediate invalidation is still wrong

`publish_command()` intentionally does not wait for workers. Arm and hand perform their final `coupled_command_ticket_allows_execution(...)` check immediately before SDK I/O. Immediate `_finish_episode()` revokes RUNNING/generation and can therefore invalidate a command that was published but not yet accepted by both workers.

Do **not** call blocking `wait_command_accepted()` from `_publish_due_action()`.

### Minimal pending state

Use one small executor-local state, for example:

```python
@dataclass
class _PendingEvaluationTermination:
    reason: str
    stop_reason: str
    recorder_save: bool
    wait_for_action_id: int
```

Use `None` for the whole pending object when no acceptance-fenced termination exists. Initial/rejected paths that have no physical action may still terminate immediately and do not need a pending object.

```python
self.pending_evaluation_termination: _PendingEvaluationTermination | None
```

Reset it in `_clear_execution()`.

### Request helper

Use a helper conceptually like:

```python
def _request_evaluation_invalid(
    self,
    reason: str,
    *,
    stop_reason: str,
    recorder_save: bool,
    wait_for_action_id: int | None,
) -> None:
    ...
```

Required behavior:

1. `wait_for_action_id is None` → finish INVALID immediately.
2. `progress.covers(wait_for_action_id)` → finish immediately.
3. Otherwise latch pending termination and leave RUNNING intact.
4. While pending, never ingest or publish another Policy action.
5. Do not busy-wait; `_observe_worker_progress()` remains the only progress polling mechanism.
6. No new timeout. Existing command-progress timeout, worker health, ticket TTL, e-stop and supervisor remain authoritative.
7. Repeated calls must be idempotent.
8. Once pending exists, no later call may change `wait_for_action_id` to a newer action; publishing a later action while pending is an implementation error.
9. `recorder_save=False` is sticky. A later branch must never upgrade corrupted/unsound storage back to `True`.
10. If a later unsaved RecorderIO failure arrives while a saved-prefix evidence INVALID is pending, update the pending reason/stop reason to the recorder-integrity cause and set `recorder_save=False`.

A simple merge rule is sufficient:

```text
pending save=True + incoming save=False
→ incoming recorder-integrity reason wins
→ save=False

otherwise
→ keep the first root-cause reason
```

Do not create a generic priority framework.

### Placement in `_run_active_tick()`

After `_observe_worker_progress(now_ns)` succeeds, check pending eval termination **before** normal task timeout/silence/action scheduling:

```text
observe worker progress
→ real progress fault? existing _fault()
→ pending eval invalid?
     covered → finish INVALID
     not covered → return; publish nothing else
→ normal max_running/watchdogs/prediction scheduling
```

This creates a bounded termination-drain phase. It may extend the nominal task wall-clock slightly while the already-published command is accepted, but publishes no new Policy action. The existing command-progress timeout and command validity window bound this drain.

### Recorder status polling during pending termination

`_poll_evaluation_recorder()` must route ERROR/unexpected FINALIZING/capacity termination through the same pending mechanism when a published action is outstanding.

Important control-flow rule:

> If recorder polling merely **latches** a pending eval invalidation and does not actually end the run, `_poll_evaluation_recorder()` must not prevent `_handle_run_boundary()` from processing higher-authority e-stop / shared fault / operator stop in the same executor tick.

Return/branch semantics should distinguish:

```text
pending latched, run still active
→ continue boundary processing

run actually finished/cleared
→ stop this tick
```

Do not let recorder polling delay safety/operator authority unnecessarily.

---

## 5. Motion precedence vs evaluation classification vs storage integrity

These are three different questions. Do not encode them as one overloaded priority number.

### A. Motion authority

The following may revoke motion immediately even if an eval-invalid action is awaiting acceptance:

```text
e-stop
real hardware/worker fault
shared error_state / supervisor failure
explicit operator S/C/D/Q stop
```

Human/safety authority is allowed to cancel the outstanding command. The acceptance fence exists only to prevent **recorder/evidence logic itself** from cancelling the current physical command.

### B. Evaluation classification

If an automatic evidence/recorder INVALID was already detected, a later operator SUCCESS/FAILURE must not sanitize the trial into a valid task outcome.

When an operator stop arrives while `pending_evaluation_termination` exists:

```text
motion stop happens immediately
but formal outcome remains INVALID
and the pending automatic stop_reason remains the evaluation reason
```

Do not let `eval:success:operator` or `eval:failure:operator` overwrite a previously detected recorder/evidence INVALID.

You may keep `RuntimeChannels.evaluation_outcome` operator-owned; the executor-local pending INVALID can remain authoritative when consuming the subsequent operator stop. No new IPC outcome state is required.

### C. Storage integrity

`recorder_save=False` is sticky across **all** later termination paths.

Example:

```text
record construction/storage failure
→ pending INVALID(save=False)
→ operator S or hardware fault arrives before acceptance
→ motion may stop immediately
→ recorder must still not be asked to save a transaction already known untrustworthy
```

The cleanest implementation is to ensure `_finish_evaluation_episode()` never upgrades an existing pending `recorder_save=False` decision.

If RecorderIO has already independently entered terminal/finalizing state, do not attempt to rewrite its owner-controlled terminal reason or storage result.

### Interaction with `pending_truncation_action_id`

A successful final action may simultaneously hit `max_action_steps` and then suffer eval evidence failure.

```text
pending eval INVALID
> pending action_step_limit
```

Check pending eval termination before `pending_truncation_action_id` after progress observation. When INVALID finishes, `_clear_execution()` clears the truncation state.

---

## 6. P1 — typed evaluation-evidence failure scope

The existing `(inputs, reason, fatal_bool)` contract is too coarse. `_build_evaluation_frame_inputs()` reads both control-critical robot feedback and evaluation-only camera/FK evidence.

Use one small explicit disposition enum instead of string parsing, for example:

```python
class _EvaluationEvidenceIssueKind(Enum):
    RETRYABLE = auto()
    EVAL_INVALID = auto()
    CONTROL_FAULT = auto()
```

Return conceptually:

```python
(inputs, reason, issue_kind)
```

when inputs are unavailable.

### CONTROL_FAULT

Use for failures that are genuinely part of robot/control health, e.g.:

```text
arm/hand ring read corruption/exception
arm disconnected
arm controller error
arm/hand state invalid
arm/hand malformed/non-finite feedback
non-stale fatal arm/hand FeedbackIssue
executor invariant failure such as evidence called with no active RUNNING epoch
```

These may continue to call `_fault()` because they are not recording-only failures.

### RETRYABLE

Use for source conditions that can become valid on a later startup tick, e.g.:

```text
no causal arm/hand sample yet
post-RUNNING source not available yet
arm/hand STALE while waiting for fresh data
no causal camera frame yet
camera frame older than freshness budget
transient camera CLOCK_RESET / DUPLICATE / DELIVERY_DELAY frame
```

For the **initial sample**, retry until the existing startup evidence deadline.

For a **post-control evidence row**, the event timestamp is already fixed: do not wait and shift the row into the future. Any non-control evidence failure after a committed control result invalidates the formal trial.

### EVAL_INVALID

Use for non-control audit/evidence integrity failures that should not become robot FAULT, e.g.:

```text
camera evidence metadata malformed/internally inconsistent
camera causal ring/payload copy failure while camera lifecycle is otherwise alive
evaluation-only hand/FK construction unavailable or failing
EpisodeState/evidence assembly failure after control feedback itself passed health checks
non-finite evaluation-only derived fields
```

### Caller behavior

Initial sample:

```text
RETRYABLE
→ wait until deadline

EVAL_INVALID
→ INVALID / discard startup transaction / no Policy command

CONTROL_FAULT
→ existing _fault()
```

Post-control evidence:

```text
CONTROL_FAULT
→ existing safety fault path; may revoke immediately

RETRYABLE or EVAL_INVALID
→ formal INVALID
→ if a successful command was just published, use acceptance-fenced invalidation
```

Do not use `reason.startswith("camera")` to decide fault ownership. If stop-reason taxonomy needs source identity, derive it from the typed issue kind/domain, not diagnostic string parsing.

---

## 7. P1 — initial-sample timeout taxonomy

The task has not begun until the mandatory initial held sample succeeds.

Therefore, while `evaluation_initial_sample_pending` is true:

```text
initial evidence deadline reached
or
max_running_s reached before first Policy command
→ INVALID
→ recorder_save=False
→ no Policy command executed
```

Use one stable startup/evidence invalid reason, for example:

```text
eval:invalid:initial_evidence_timeout
```

Do not report this pre-command condition as `eval:failure:timeout`.

After the initial sample succeeds, keep the existing task wall-clock timeout semantics unchanged in this patch.

---

## 8. P1 — `PreparedCommand.unavailable` parity

`prepare_command()` already distinguishes:

```text
unavailable=True
→ feedback missing or STALE
→ not a fatal safety condition

fatal=True
→ fatal feedback/checker condition
```

Remove the formal-eval-only escalation.

Required behavior in both normal and eval:

```python
if prepared.unavailable:
    return
```

Do not consume the control slot and do not fabricate IK/SAFETY rejection evidence because a complete control decision was not made.

Persistent unavailability remains bounded by existing timestamp staleness, first-command/silence, command-progress, feedback-health and supervisor logic.

Keep genuine `prepared.fatal` handling unchanged.

---

## 9. P1 — correct EE IK metadata

The control rejection kind is already explicit. Fix only the recorded truth table.

For decode-stage rejection:

```python
is_ee = self.policy_spec.action_key == "action_ee"
ik_attempted = is_ee
ik_ok = bool(is_ee and reject_kind is _RejectKind.SAFETY)
```

Required rows:

| action space | reject kind | `ik_attempted` | `ik_ok` | frame status |
|---|---|---:|---:|---|
| joint | SAFETY | false | false | SAFETY_REJECT |
| EE | IK | true | false | IK_FAIL |
| EE | SAFETY after successful IK | true | true | SAFETY_REJECT |

Preparation/SafetyGate rejection after an EE decode also means IK succeeded, so keep `ik_ok=True` there.

Do not infer these values from human-readable reasons.

The rejected EE IK joint candidate does not need a new persisted field in this patch; current raw-action validity flags already prevent the held fallback arm value from masquerading as a valid raw joint candidate.

---

## 10. P1 — terminal rejected step ordering

### Current bug

A rejected action currently commits `_record_terminal_step(successful=False)` before rejection evidence. At `max_action_steps`, that helper can clear the active RUNNING episode before the row is built.

### Required refactor

Separate **step commit** from **episode finish**.

A compact helper is enough, for example:

```python
def _commit_terminal_step(
    self,
    *,
    successful: bool,
    candidate: ActionCandidate | None,
) -> bool:
    """Commit step/prediction state and return limit_reached."""
```

Rules:

1. increment `episode_steps` exactly once;
2. compute `limit_reached`;
3. if not terminal, advance prediction exactly as today;
4. successful physical terminal action keeps existing `pending_truncation_action_id` acceptance behavior;
5. do not finish a rejected formal-eval episode from inside the commit helper;
6. ordinary non-eval rejected terminal step may finish immediately in its caller because there is no eval evidence to preserve.

Formal rejected path:

```text
consume control slot + capture terminal_ns
→ commit step/prediction bookkeeping
→ attempt rejection evidence at terminal_ns
→ evidence caused INVALID / run ended / pending termination? stop here
→ otherwise if limit reached: finish action_step_limit
```

Do not create a synthetic held row to compensate for an ordering bug.

---

## 11. P1 — persist formal `max_frames` reason at the RecorderIO owner boundary

### Current ownership bug

RecorderIO automatically starts finalization with:

```text
reason="max_frames"
```

before PolicyExecutor learns about FINALIZING. Once `RecorderClient.poll_stop()` sees FINALIZING, a later executor STOP is a no-op and cannot rewrite the reason already captured by RecorderIO/EpisodeRecorder.

### Minimal fix

Keep RecorderIO generic and make only its automatic capacity reason configurable:

```python
@dataclass(frozen=True)
class RecorderIOConfig:
    ...
    max_frames_stop_reason: str = "max_frames"
```

Validate it as a non-empty bounded string using the existing `RECORD_STOP_REASON_BYTES` contract / `bounded_control_text` helper.

Define a formal-eval constant in `deployment/evaluation.py`:

```python
EVALUATION_MAX_FRAMES_STOP_REASON = "eval:invalid:max_frames"
```

`lifecycle._evaluation_recorder_config(...)` passes this value.

RecorderIO auto-cap path uses:

```python
reason=self.config.max_frames_stop_reason
```

Ordinary teleop/recording callers keep the default `"max_frames"` unchanged.

PolicyExecutor recognizes the exact formal-eval constant when polling FINALIZING. Do not send a second STOP to rewrite an already-running finalization.

Healthy max-frame capacity termination remains `save=True`: INVALID describes evaluation validity, not storage corruption.

No raw schema/version change is required.

---

## 12. P2 — causal control-event evidence anchors

Current post-control helpers create a new `time.monotonic_ns()` after the control result. This makes state/action pairing depend on whether a newer feedback frame appears before evidence assembly.

Prefer deterministic event anchors.

### Successful command

Pass the already-validated:

```python
publication_ns = publish_result.ticket.published_monotonic_ns
```

to the evidence helper and use it as the causal anchor.

### Rejected action

Capture:

```python
terminal_ns = time.monotonic_ns()
```

when the rejection/control slot is committed and use that as the evidence anchor.

### Timing metrics

Measure code cost from a separate clock:

```python
build_start_ns = time.monotonic_ns()
inputs = _build_evaluation_frame_inputs(event_anchor_ns)
build_ms = (time.monotonic_ns() - build_start_ns) / 1e6
```

Do not use the historical event timestamp itself to compute processing latency.

This preserves control-first ordering while making audit cuts reproducible.

---

## 13. Recommended production patch scope

Primary:

```text
dexmani_real/deployment/executor.py
```

Small supporting changes:

```text
dexmani_real/deployment/evaluation.py
dexmani_real/deployment/lifecycle.py
dexmani_real/recording/io_worker.py
```

Tests/docs:

```text
tests/test_policy_rollout.py
optionally one focused RecorderIO-boundary test module
repo_map.md
docs/policy_formal_eval_followup_guide.md
```

Normally do not modify:

```text
dexmani_real/deployment/config.py
dexmani_real/deployment/inference/*
dexmani_real/deployment/timing.py
dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py
dexmani_real/recording/client.py
dexmani_real/recording/recorder.py
dexmani_real/runtime/safety.py
dexmani_real/runtime/supervisor.py
robot/arm_worker.py
robot/hand_worker.py
sensor drivers
IPC schemas
teleop
raw/processed dataset schemas
```

`recording/client.py` or `recording/recorder.py` should only be touched if source inspection proves the RecorderIO-config solution cannot be completed at the existing owner boundary.

Do not perform unrelated cleanup.

---

## 14. Implementation order

### Phase A — typed evidence ownership

1. replace the ambiguous evidence `fatal` bool with a tiny typed disposition;
2. preserve arm/hand control-fault behavior;
3. make camera/FK recording-only issues retryable/INVALID rather than robot FAULT;
4. fix initial-sample wall-clock timeout to INVALID.

### Phase B — acceptance-fenced eval termination

1. add pending termination state;
2. make request/merge idempotent;
3. keep `recorder_save=False` sticky;
4. route post-publication evidence/RecorderIO invalidation through it;
5. process pending state immediately after worker-progress observation;
6. do not publish new actions while pending;
7. preserve immediate e-stop/hardware/operator motion authority;
8. preserve pending INVALID classification against later operator SUCCESS/FAILURE.

### Phase C — control parity + metadata

1. remove eval-only `PreparedCommand.unavailable` fault;
2. fix EE `ik_ok` truth table.

### Phase D — terminal-step ordering

Refactor step commit vs finish and preserve terminal rejection evidence.

### Phase E — max-frames owner reason

1. add default generic `max_frames_stop_reason` to RecorderIOConfig;
2. formal eval overrides it with the constant;
3. auto-finalization uses that reason;
4. executor recognizes but never rewrites finalization reason.

### Phase F — event-anchor optimization

Use publication/rejection event timestamps if the change remains local.

### Phase G — deterministic tests and durable docs

No trained Policy or hardware is required.

---

## 15. Required deterministic regressions

Use lightweight `unittest`/fakes. Do not build a hardware simulator framework.

### A. Evidence issue ownership

Cover at least:

```text
arm/hand non-stale fatal feedback issue
→ CONTROL_FAULT
→ existing _fault()
```

```text
initial camera duplicate/reset/delivery-delay or missing fresh frame
→ retry within initial deadline
→ no _fault()
```

```text
terminal evaluation-only camera/FK metadata error
→ INVALID
→ no _fault()
```

```text
max_running_s expires while initial sample still pending
→ eval:invalid:initial_evidence_timeout
→ recorder_save=False
→ no Policy command
```

### B. `PreparedCommand.unavailable`

With recorder absent and present:

```text
prepared.unavailable=True
→ no _fault()
→ no control-slot consumption
→ no IK/SAFETY row
```

### C. EE rejection metadata

```text
EE IK failure
→ ik_attempted=True
→ ik_ok=False
→ IK_FAIL
```

```text
EE IK success + arm/joint admission failure
→ ik_attempted=True
→ ik_ok=True
→ SAFETY_REJECT
```

```text
joint safety failure
→ ik_attempted=False
→ ik_ok=False
→ SAFETY_REJECT
```

### D. Publish→evidence-failure acceptance fence

Fake `action_id=42`:

```text
publish succeeds
evidence fails
progress.covers(42)=False
```

Assert:

```text
no immediate _finish_evaluation_episode
no new Policy action/prediction publication
pending waits for action_id=42
```

Then advance arm+hand accepted IDs to cover 42 and assert INVALID finishes exactly once.

### E. Hand intermediate progress remains valid drain progress

While pending action 42:

```text
hand exact accepted_target_action_id < 42
but last_sdk_setpoint_accepted_monotonic_ns advances
```

Assert the existing command-progress watchdog does not falsely fault while genuine intermediate hand SDK progress continues.

Then exact acceptance covers 42 and termination completes.

### F. Real progress timeout still wins safety

Pending eval INVALID must not hide a genuine command-progress timeout. Assert the existing `_fault()` path remains reachable.

### G. Pending INVALID + operator stop

```text
pending automatic eval INVALID exists
operator later requests SUCCESS/FAILURE stop
```

Assert:

```text
motion can stop immediately
automatic INVALID classification/stop_reason is not replaced by SUCCESS/FAILURE
```

### H. Sticky unsaved storage decision

```text
pending recorder-integrity INVALID(save=False)
then operator stop or hardware-fault finish path runs
```

Assert final recorder request never upgrades save to True.

### I. Pending INVALID beats pending action-step limit

Successful final action:

```text
max_action_steps reached
evidence fails
action later accepted
```

Final evaluation disposition must be INVALID, not action-step failure.

### J. Terminal rejected step ordering

Set next rejected step to hit `max_action_steps` and assert call order:

```text
consume slot
→ commit step
→ rejection evidence
→ finish
```

Evidence must see an active RUNNING epoch.

### K. Persisted max-frames reason at owner boundary

Test RecorderIO auto-finalization itself:

```text
default RecorderIOConfig
→ reason == "max_frames"

formal-eval RecorderIOConfig
→ reason == "eval:invalid:max_frames"
```

Do not merely mock executor `_finish_evaluation_episode()`.

### L. Recorder polling while pending

If recorder poll latches a pending invalidation but the run remains active, higher-authority e-stop/operator boundary handling must still be reachable in the same logical boundary pass.

### M. Event anchor, if implemented

```text
published evidence anchor == ticket.published_monotonic_ns
rejection evidence anchor == captured terminal_ns
```

Timing metrics must still use separate processing clocks.

### N. Preserve scheduler regressions

Keep/strengthen:

```text
partial stale → prefix skip
whole stale → discard
old run_generation → ignored
no catch-up
old source_monotonic_ns alone does not expire a timestamp-valid action
```

---

## 16. Validation commands

Before editing:

```bash
git status --short
git rev-parse HEAD
```

Inspect current call graph:

```bash
rg -n "_build_evaluation_frame_inputs|_invalidate_evaluation|_finish_evaluation_episode|pending_truncation_action_id|_record_terminal_step|_record_evaluation_(command|rejection)_evidence|prepared\.unavailable|max_frames|ik_ok|evaluation_outcome" \
  dexmani_real tests repo_map.md docs
```

After editing:

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_policy_rollout.py'
```

If a second focused recorder test is added, run it explicitly or include it in a narrow deterministic discovery command.

Then:

```bash
git diff --check
git diff --stat
```

Search stale semantics:

```bash
rg -n "eval:failure:max_frames|arm_action_delta_clip_rad|max_source_to_command_age_s|temporal_ensemble_coeff" \
  dexmani_real examples README.md repo_map.md docs tests
```

Historical explanatory mentions in implementation guides are allowed; active runtime/current docs must match implementation.

Do not run Policy training, checkpoint selection, real Policy export/restore, long simulator evaluation, or robot hardware during this repair phase.

---

## 17. Definition of Done

### Control/recording independence

```text
published action + eval-only failure
→ no recorder-caused revocation before outstanding arm+hand acceptance
```

```text
pending eval INVALID
→ no later Policy action
→ progress watchdog remains active
→ accepted outstanding action then fences to ARMED/INVALID
```

```text
operator/e-stop/hardware authority
→ may still stop immediately
```

### Failure ownership

```text
control-critical arm/hand failure
→ real control fault remains possible
```

```text
recording-only camera/FK/evidence issue
→ retry/INVALID
→ not global FAULT by itself
```

```text
PreparedCommand.unavailable
→ same behavior in run and eval
```

### Outcome/storage correctness

```text
automatic eval INVALID
→ cannot be overwritten by later operator SUCCESS/FAILURE
```

```text
recorder_save=False
→ sticky across all later termination paths
```

```text
pre-first-command evidence timeout
→ INVALID, not task FAILURE
```

### Audit correctness

```text
EE IK_FAIL → ik_ok=False
EE post-IK SAFETY_REJECT → ik_ok=True
```

```text
terminal rejected action
→ evidence attempted before episode finish
```

```text
formal max_frames
→ RecorderIO begins finalization with persisted eval:invalid:max_frames reason
```

### Preserved invariants

```text
no blocking realtime acceptance wait
no new scheduler
no temporal blending / RTC
no source-age action deadline
no learned-policy arm clipping
no compatibility/version machinery
no RecorderIO architecture rewrite
no data schema version change
no hardware-worker authority change
```

---

## 18. Staged integration / first real-robot eval validation

Only after a trained Policy checkpoint exists and all offline checks pass:

### Stage 1 — Policy integration only

```text
current Policy checkpoint
→ export/restore validation
→ Real check/shadow
→ no physical command
```

Confirm public `PolicySpec`, observation shapes, inference latency and action chunk shape.

### Stage 2 — short physical ordinary run

Use a very small `max_action_steps` first.

Verify:

```text
published action_id
arm last_cmd/action acceptance
hand exact accepted_target_action_id
run_generation
SafetyState transitions
command-progress telemetry
```

Do not start with large trial batches.

### Stage 3 — one formal eval trial

Inspect the raw artifact before increasing trial count:

```text
stop_reason
success/truncated metadata
frame_status
ik_attempted / ik_ok
flag_safety_reject
action_id / action_target timestamps
action_arm_joint_sent
hand_accepted_target_action_id
control_run_generation
RecorderIO saved/error/failure_count
```

Also inspect:

```text
evaluation_state_build_ms / max
evaluation_record_ms / max
publication interval / schedule lateness
```

The remaining hardware-only risk is post-control camera/evidence copying load on the real control machine. Measure it before large-scale evaluation.

### Stage 4 — expand evaluation

Only after Stage 3 confirms transaction ordering and artifact semantics:

```text
increase max_action_steps / wall-clock budget
run multiple SUCCESS/FAILURE/INVALID examples
then start task success-rate evaluation
```

Do not interpret task success rate before the evaluation transaction itself is validated.

---

## 19. Final Claude Code / Codex report

### Changed

Report exact files and these mandatory repairs:

```text
typed evidence failure ownership
acceptance-fenced eval invalidation
pending termination precedence/storage merge
initial timeout taxonomy
PreparedCommand.unavailable parity
EE ik_ok metadata
terminal rejection ordering
persisted formal max_frames reason
```

Mention event-anchor optimization separately if implemented.

### Preserved

Confirm:

```text
Policy public contract
causal observation alignment
sync/async
run_generation
timestamp scheduler
command TTL/watchdogs
SafetyGate
home/workspace/collision
RecorderIO architecture
supervisor/lifecycle
Real data schema versions
```

### Verified

Only list commands/tests actually run.

### Explicitly Not Run

```text
no Policy training
no checkpoint selection
no real checkpoint restore/export
no long simulator evaluation
no robot hardware
```

### Remaining Risk

List concrete unverified risks only. Before physical validation, the principal one should be timing/load of post-control evaluation state/camera copying on the real control machine.

Do not propose RTC, temporal smoothing, compatibility layers, a new scheduler, or a recorder rewrite as generic follow-up work.
