# DexMani Real — Formal Policy Eval Follow-up Repair Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Reviewed baseline: `cfa446fe536c95078e28a663ef919e13f1fc0b07`  
> Upstream Policy boundary: current `dexmani_policy.deployment.PolicySpec` without `temporal_ensemble_coeff`  
> Intended executor: Claude Code / Codex  
> Scope: finish the remaining **formal real-eval correctness** work after the canonical rollout cleanup. Keep the patch local, deterministic, and hardware-independent.
>
> This guide supersedes the formal-eval follow-up details in `docs/policy_rollout_simplification_guide.md`. The earlier cleanup remains authoritative for the already-stable Policy contract, ActionChunk scheduling, and reject-only learned-policy arm semantics.

---

## 1. Current status

The previous rollout repair correctly stabilized these parts. **Do not redesign them again**:

```text
Policy public contract
→ no temporal_ensemble_coeff

Observation freshness
→ max_input_age_s / skew / grid lag before inference

Action temporal validity
→ logical timestamps / stale-prefix / whole-stale / no-catch-up

Command delivery validity
→ action_validity_s + existing publication/worker guards

Learned-policy arm
→ nearest-equivalent canonicalization
→ joint bounds
→ per-joint jump reject
→ never silent clip

Failure class
→ true EE IK failure = IK_FAIL
→ post-IK/joint/workspace/hand safety rejection = SAFETY_REJECT
```

Also preserve:

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
command progress watchdog
first-command / silence / apply timeouts
RecorderIO process architecture
supervisor / heartbeat / process-death handling
robot workers
IPC schemas
raw/processed/Policy-Zarr/calibration schema versions
```

This follow-up is **not** another deployment redesign.

---

## 2. Verified remaining problems

The following issues are confirmed against the reviewed baseline.

| Priority | Issue | Why it matters |
|---|---|---|
| P0 | evaluation failure can fence RUNNING after IPC publish but before arm+hand SDK acceptance | recorder behavior can still change whether the current physical command crosses the actuator boundary |
| P1 | initial-sample recording-only camera/evidence errors can still call `_fault()` | a single eval-evidence problem can become robot global FAULT even when the lifecycle itself is healthy |
| P1 | `PreparedCommand.unavailable` is non-fatal in normal run but becomes `_fault()` only in formal eval | formal eval changes control semantics |
| P1 | EE IK-success followed by arm admission rejection records `ik_ok=False` | audit metadata contradicts the actual rejection class |
| P1 | `max_frames` is mapped to `eval:invalid:max_frames` only at executor intent level; RecorderIO has already started finalization with reason `max_frames` | persisted episode stop reason does not match the formal-eval taxonomy |
| P1 | rejected terminal step at `max_action_steps` may finish the episode before its rejection evidence is recorded | the last control result can disappear from the formal eval record |
| P2 | per-command evidence anchor uses a fresh `time.monotonic_ns()` after publication | state/action pairing depends unnecessarily on scheduling; publication/rejection event timestamps give a cleaner causal audit cut |

Do not expand this patch into unrelated timeout taxonomy, simulator behavior, Policy training, or data-format migration.

---

## 3. Target transaction semantics

Formal evaluation must distinguish four boundaries:

```text
1. CONTROL DECISION
   decode / IK / safety → publish or reject

2. IPC PUBLICATION
   coupled command enters latest-wins ring

3. PHYSICAL ACCEPTANCE
   arm + hand workers report accepted action_id

4. EVALUATION EVIDENCE
   RecorderIO receives the audit row
```

The required ordering is:

### Successful physical action

```text
control decision
→ publish coupled command
→ commit scheduler/control bookkeeping
→ attempt evaluation evidence
→ if evidence is healthy: continue
→ if evidence fails: stop scheduling new actions
→ wait non-blockingly until arm+hand acceptance covers this published action
→ then fence the trial and mark INVALID
```

### Rejected action

```text
control decision = IK_FAIL or SAFETY_REJECT
→ consume/commit control slot
→ commit episode-step bookkeeping
→ record rejection evidence
→ then apply action-step-limit or eval-invalid termination
```

A rejected action has no actuator acceptance to wait for.

### Startup

```text
B
→ Recorder START ACK
→ RUNNING epoch
→ initial causal held sample
→ first Policy command
```

The initial sample remains mandatory, but an evaluation-only evidence problem must end the trial as INVALID rather than create robot FAULT by itself.

---

## 4. P0 repair — defer eval termination until the current published command is accepted

### Problem

`publish_command()` is intentionally non-blocking. The arm and hand workers perform a final `coupled_command_ticket_allows_execution(...)` check immediately before their SDK call. If the executor calls `_invalidate_evaluation()` immediately after publication, `_finish_episode()` revokes RUNNING/generation, so a worker that has not yet reached its final guard can drop the published command.

Do **not** solve this by adding a blocking `wait_command_accepted()` inside `_publish_due_action()`; that would put worker latency directly into the 16 Hz execution path.

### Recommended minimal state

Add one small executor-owned pending eval termination state, for example:

```python
@dataclass
class _PendingEvaluationTermination:
    reason: str
    stop_reason: str
    recorder_save: bool
    wait_for_action_id: int | None
```

Store:

```python
self.pending_evaluation_termination: _PendingEvaluationTermination | None
```

Reset it in `_clear_execution()`.

### Helper 1 — request invalid termination

Replace immediate `_invalidate_evaluation()` semantics with a helper conceptually like:

```python
def _request_evaluation_invalid(
    self,
    reason: str,
    *,
    stop_reason: str,
    recorder_save: bool,
    wait_for_action_id: int | None = None,
) -> None:
    ...
```

Rules:

1. If `wait_for_action_id is None`, finish INVALID immediately.
2. If `self.progress.covers(wait_for_action_id)` is already true, finish immediately.
3. Otherwise latch the pending termination and **do not revoke motion yet**.
4. While a pending termination exists, do not ingest/publish any later Policy action.
5. Never busy-wait; normal executor polling + `_observe_worker_progress()` owns progress.
6. If a later recorder-integrity failure occurs while an evidence-only invalid is pending, `recorder_save=False` must dominate. Do not accidentally turn corrupted storage back into a saved prefix.
7. Existing e-stop, worker fault, shared `error_state`, operator S/Q and supervisor failure always override this pending eval state through the existing lifecycle paths.

### Helper 2 — finish pending invalid after acceptance

After `_observe_worker_progress(now_ns)` updates acceptance watermarks:

```text
pending eval termination?
    no → normal scheduling
    yes and progress.covers(wait_for_action_id)
        → _finish_evaluation_episode(... INVALID ...)
    yes and not covered
        → return without scheduling another action
```

If acceptance never arrives, the existing `command_progress_timeout` remains the authority and may escalate to the existing real control fault path. Do not invent another timeout.

### Where to pass `wait_for_action_id`

For an evidence failure associated with a successfully published command:

```python
wait_for_action_id = published_candidate.action_id
```

For a rejection row or initial sample:

```python
wait_for_action_id = None
```

For asynchronous RecorderIO status/error detected between control ticks, use the latest published action only when it is still outstanding:

```text
progress.latest_published_action_id exists
and not progress.covers(latest_published_action_id)
→ defer eval fence until that action is covered
```

This closes the publish→SDK race without adding blocking synchronization.

---

## 5. Interaction with existing `pending_truncation_action_id`

Do not create ambiguous competing episode endings.

A successful final action may simultaneously:

```text
reach max_action_steps
and
fail evaluation evidence
```

In that case the evaluation evidence failure must take precedence:

```text
INVALID recording/evidence outcome
> action_step_limit task termination
```

The smallest acceptable implementation is:

1. keep existing `pending_truncation_action_id`;
2. check `pending_evaluation_termination` **before** `pending_truncation_action_id` after progress observation;
3. when both refer to the same action, finish INVALID after acceptance and clear the truncation state through `_clear_execution()`.

Do not add a generalized workflow/state-machine framework just to merge these two fields.

---

## 6. P1 repair — initial evidence failure must not own robot FAULT

### Current problem

`_record_initial_evaluation_sample()` still interprets the `fatal` flag returned by `_build_evaluation_frame_inputs()` as permission to call `_fault()`.

However `_build_evaluation_frame_inputs()` is an **evaluation evidence builder**. It can return terminal evidence errors for a single camera frame such as reset/duplicate/invalid audit metadata while the camera worker continues running. Camera worker lifecycle failure is independently represented by worker crash, heartbeat, or `shared.error_state`.

### Target ownership

Inside the evaluation evidence path:

```text
retryable evidence miss
→ wait until startup evidence deadline

terminal evidence-quality error
→ INVALID trial
→ no Policy command
→ recorder STOP/discard as appropriate
→ no executor-owned global FAULT
```

Actual control/lifecycle faults remain owned by:

```text
shared.error_state / e-stop
arm/hand worker fail-fast
camera/recorder process crash
heartbeat/supervisor
command-progress health
```

### Implementation recommendation

Keep the existing evidence-builder classification, but rename the local meaning if useful:

```text
fatal
→ terminal_for_evaluation
```

or

```text
retryable: bool
```

Do not parse source strings to decide robot fault ownership.

In `_record_initial_evaluation_sample()`:

```text
inputs unavailable + retryable
→ keep waiting until initial deadline

inputs unavailable + terminal-for-eval
→ _request_evaluation_invalid(..., wait_for_action_id=None)
```

If the real lifecycle independently sets `error_state` concurrently, the normal boundary/supervisor path will still fault the system.

---

## 7. P1 repair — `PreparedCommand.unavailable` must behave the same in run and eval

`prepare_command()` intentionally distinguishes:

```text
unavailable=True
→ feedback missing or stale
→ not a fatal safety condition

fatal=True
→ actual fatal feedback/checker condition
```

Current formal eval adds an extra `_fault()` only when `self.recorder is not None`.

Remove that eval-only escalation.

Required behavior:

```python
if prepared.unavailable:
    return
```

for both normal run and formal eval.

Do not consume the current control slot and do not record IK/SAFETY rejection, because no valid control decision was completed.

The existing timestamp staleness, first-command/silence watchdogs, worker feedback checks and supervisor remain responsible for persistent unavailability.

Keep:

```python
if prepared.fatal:
    self._fault(...)
```

unchanged.

---

## 8. P1 repair — correct EE `ik_ok` metadata after post-IK safety rejection

The control classification is already correct. Only the recording flags are inconsistent.

For decode-stage rejection use:

```python
is_ee = self.policy_spec.action_key == "action_ee"
ik_attempted = is_ee
ik_ok = bool(is_ee and reject_kind is _RejectKind.SAFETY)
```

Required truth table:

| action space | reject kind | `ik_attempted` | `ik_ok` | frame status |
|---|---|---:|---:|---|
| joint | SAFETY | false | false | SAFETY_REJECT |
| EE | IK | true | false | IK_FAIL |
| EE | SAFETY after successful IK | true | true | SAFETY_REJECT |

Do not infer this from rejection strings.

Preparation/SafetyGate rejection after an EE decode already implies IK succeeded; preserve that same truth table.

---

## 9. P1 repair — terminal rejected step must record evidence before episode finalization

### Current bug

A rejected action currently calls `_record_terminal_step(successful=False)` before `_record_evaluation_rejection_evidence()`. When that step reaches `max_action_steps`, `_record_terminal_step()` can end the episode immediately, clearing the RUNNING epoch before the rejection evidence helper runs.

### Required ordering

Separate **step commit** from **episode finish**.

A compact refactor is preferred:

```python
def _commit_terminal_step(
    self,
    *,
    successful: bool,
    candidate: ActionCandidate | None,
) -> bool:
    """Commit episode-step/prediction state and return whether the step limit was reached."""
```

Suggested semantics:

1. increment `episode_steps`;
2. determine `limit_reached`;
3. if not terminal, advance the active prediction as today;
4. if successful + physical + terminal, retain the current acceptance-fenced `pending_truncation_action_id` behavior;
5. **do not directly finish a rejected formal-eval episode inside this helper**;
6. return `limit_reached` to the caller.

Rejected path:

```text
consume control slot
→ commit episode step / scheduler state
→ record rejection evidence
→ if evidence invalidated trial: stop here
→ if action-step limit reached: finish action_step_limit
```

This preserves the final rejection row.

Do not add synthetic “held” rows after termination to compensate for a lost rejection row; fix the ordering instead.

---

## 10. P1 repair — make formal-eval `max_frames` reason persist correctly

### Current problem

RecorderIO begins finalization itself when `EpisodeRecorder.max_frames_reached` becomes true:

```text
_begin_finalization(save=True, reason="max_frames")
```

By the time PolicyExecutor sees `RecorderPhase.FINALIZING`, `RecorderClient` has already marked the transaction as stop-pending. A later executor call using `stop_reason="eval:invalid:max_frames"` cannot overwrite the reason already handed to the EpisodeRecorder finalization thread.

### Recommended minimal fix

Keep RecorderIO generic, but make its automatic capacity reason configurable.

Add to `RecorderIOConfig`:

```python
max_frames_stop_reason: str = "max_frames"
```

Validate it as a non-empty bounded control/status string using the existing `RECORD_STOP_REASON_BYTES` boundary.

In RecorderIO capacity auto-finalization use:

```python
reason=self.config.max_frames_stop_reason
```

Define one formal-eval constant, preferably in `deployment/evaluation.py`:

```python
EVALUATION_MAX_FRAMES_STOP_REASON = "eval:invalid:max_frames"
```

`lifecycle._evaluation_recorder_config(...)` sets:

```python
max_frames_stop_reason=EVALUATION_MAX_FRAMES_STOP_REASON
```

Ordinary teleop/recording callers keep the default `"max_frames"` unchanged.

PolicyExecutor then recognizes the exact formal-eval constant when RecorderIO reports FINALIZING. Do **not** try to send a second STOP to rewrite an already-running finalization.

This changes no raw schema and introduces no version/migration mechanism; it only makes the already-existing `stop_reason` correct at its owner boundary.

---

## 11. P2 audit-quality optimization — anchor evidence to the control event

This is an audit-quality improvement, not the main control bug.

Current evidence helpers create a new `time.monotonic_ns()` after publication/rejection. That means the chosen causal state may depend on whether a worker publishes a newer feedback sample before the evidence builder runs.

Prefer explicit event anchors:

### Published action

Use:

```python
anchor_ns = publication_ns
```

where `publication_ns` is the `CoupledCommandTicket.published_monotonic_ns` already validated by `_publish_due_action()`.

### Rejected action

Capture the rejection/control-slot terminal timestamp:

```python
terminal_ns = time.monotonic_ns()
```

when the reject is committed, and pass it to the rejection evidence helper.

### Timing metrics

Do not use the event anchor to measure code runtime. Use separate measurement timestamps:

```python
build_start_ns = time.monotonic_ns()
inputs = _build_evaluation_frame_inputs(anchor_ns)
build_ms = (time.monotonic_ns() - build_start_ns) / 1e6
```

This yields deterministic causal audit cuts without moving recording work back in front of control publication.

If this change becomes unexpectedly invasive, complete P0/P1 first; do not block the correctness patch on P2.

---

## 12. Recorder/eval termination precedence

Keep precedence simple and explicit:

```text
1. e-stop / lifecycle hardware fault / worker death
   → existing global FAULT path

2. recorder integrity/storage failure
   → INVALID eval
   → recorder_save=False
   → wait for outstanding published action acceptance before eval fence

3. evaluation evidence-quality failure
   → INVALID eval
   → save healthy prefix when recorder is trustworthy
   → wait for outstanding published action acceptance before eval fence

4. operator SUCCESS / FAILURE / INVALID
   → existing explicit operator outcome path

5. action-step / wall-clock task boundary
   → existing task termination semantics
```

If an eval-invalid reason appears while an action-step-limit termination is waiting for the same published action, INVALID takes precedence.

Do not add priority numbers to IPC or persist a second outcome state machine.

---

## 13. Expected production patch scope

Primary:

```text
dexmani_real/deployment/executor.py
```

Small supporting changes likely required for the persisted max-frames reason:

```text
dexmani_real/deployment/evaluation.py
dexmani_real/deployment/lifecycle.py
dexmani_real/recording/io_worker.py
```

Tests/docs:

```text
tests/test_policy_rollout.py
possibly one small recorder-boundary test module if cleaner
repo_map.md
this guide
```

Normally **do not modify**:

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

`recording/client.py` or `recording/recorder.py` should only be touched if source inspection proves the `RecorderIOConfig.max_frames_stop_reason` solution cannot be completed at the current RecorderIO owner boundary.

Do not add schema versions, compatibility readers, event buses, or a new recorder architecture.

---

## 14. Implementation order for Claude Code / Codex

Implement in this order:

### Phase A — acceptance-fenced eval termination

1. add pending eval termination state;
2. reset it in `_clear_execution()`;
3. route post-publication evidence/RecorderIO invalidation through it;
4. stop scheduling later Policy actions while pending;
5. finish only after `_CommandProgress.covers(action_id)`;
6. keep existing command-progress timeout and lifecycle fault ownership.

### Phase B — remove eval-only fault divergence

1. initial evidence terminal error → INVALID, not `_fault()`;
2. `PreparedCommand.unavailable` → same no-command behavior in normal and eval;
3. keep truly fatal preparation/lifecycle conditions unchanged.

### Phase C — metadata and terminal-step correctness

1. fix EE post-IK `ik_ok` flag;
2. refactor step commit vs episode finish;
3. ensure terminal rejected step evidence exists before action-step termination.

### Phase D — persisted max-frames taxonomy

1. add generic `RecorderIOConfig.max_frames_stop_reason="max_frames"`;
2. formal eval overrides it with `eval:invalid:max_frames`;
3. RecorderIO uses it at the original `_begin_finalization()` call;
4. executor does not attempt reason rewriting after FINALIZING has begun.

### Phase E — causal event anchor optimization

Use publication/rejection event timestamps for evidence if the change remains local.

### Phase F — offline regressions + durable docs

No hardware or trained Policy is required for this patch.

---

## 15. Required deterministic regressions

Extend the current lightweight tests; do not introduce a heavy framework.

### A. EE rejection metadata

```text
EE IK failure
→ kind=IK
→ ik_attempted=True
→ ik_ok=False
→ frame_status=IK_FAIL
```

```text
EE IK success + arm jump/joint admission failure
→ kind=SAFETY
→ ik_attempted=True
→ ik_ok=True
→ frame_status=SAFETY_REJECT
```

```text
joint admission failure
→ ik_attempted=False
→ ik_ok=False
→ SAFETY_REJECT
```

### B. Initial eval evidence ownership

Fake a terminal camera/evidence error while:

```text
shared.error_state=False
estop=False
workers otherwise healthy
```

Assert:

```text
trial INVALID / episode finish requested
_fault() not called
first Policy inference/command not requested
```

A separately latched `shared.error_state` path must still follow the existing global fault/lifecycle handling.

### C. `PreparedCommand.unavailable`

For both recorder absent and recorder present:

```text
prepared.unavailable=True
→ no _fault()
→ no control-slot consumption
→ no SAFETY_REJECT/IK_FAIL row
```

### D. Publish→evidence-failure acceptance fence

Use fake `_CommandProgress` / shared state:

```text
publish action_id=42
evidence failure occurs
progress does not cover 42
```

Assert:

```text
no _finish_evaluation_episode yet
no new Policy action scheduled
```

Then update acceptance watermarks so:

```text
progress.covers(42) == True
```

Assert INVALID termination happens exactly once.

Also test that an actual progress timeout still reaches the existing fault path instead of being hidden by pending eval invalidation.

### E. Pending invalid beats pending action-step limit

For the same successful terminal action:

```text
max_action_steps reached
evidence failure occurs
arm+hand later accept
```

Assert final reason is eval INVALID, not `action_step_limit`.

### F. Terminal rejected step evidence ordering

Set `episode_steps` so the next rejected action reaches `max_action_steps`.

Record call order and assert:

```text
control-slot commit
→ rejection evidence attempt
→ episode finish
```

The rejection row must not observe `run_started_ns=None` before it is built.

### G. Persisted max-frames reason

Without real disk/video/hardware, test the RecorderIO owner boundary with fakes/mocks:

```text
ordinary RecorderIOConfig
→ auto-cap finalization reason == "max_frames"

formal-eval RecorderIOConfig
→ auto-cap finalization reason == "eval:invalid:max_frames"
```

Test the reason passed to `_begin_finalization()`, not merely the later executor `_finish_evaluation_episode()` call.

### H. Event anchor (if Phase E implemented)

Assert:

```text
published evidence builder anchor == ticket.published_monotonic_ns
rejection evidence builder anchor == rejection terminal timestamp
```

and timing metrics still measure local build/record duration from independent measurement clocks.

### I. Preserve existing scheduling regressions

Keep or strengthen:

```text
partial stale → prefix skip
whole stale → discard
old run_generation → ignored
no catch-up
source_monotonic_ns alone does not expire timestamp-valid action
```

Do not weaken existing tests while adding formal-eval coverage.

---

## 16. Validation commands

Before editing:

```bash
git status --short
git rev-parse HEAD
```

Inspect the actual current call graph before patching:

```bash
rg -n "_invalidate_evaluation|_finish_evaluation_episode|pending_truncation_action_id|_record_terminal_step|_record_evaluation_(command|rejection)_evidence|prepared\.unavailable|max_frames|ik_ok" \
  dexmani_real tests repo_map.md docs
```

After editing:

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_policy_rollout.py'
```

If a second focused recorder-boundary test module is added, run it explicitly or use a narrow discovery pattern covering both.

Then:

```bash
git diff --check
git diff --stat
```

Search for stale semantics:

```bash
rg -n "eval:failure:max_frames|arm_action_delta_clip_rad|max_source_to_command_age_s|temporal_ensemble_coeff" \
  dexmani_real examples README.md repo_map.md docs tests
```

Historical descriptions in implementation guides may mention removed names; active runtime/current docs must match the final implementation.

Do not run Policy training, checkpoint selection, real Policy export/restore, or robot hardware in this code-repair phase.

---

## 17. Definition of Done

The patch is complete only when all of the following are true.

### Control / recorder independence

```text
published coupled action
+ later evidence/recorder failure
→ no immediate motion revocation while that action is still awaiting arm+hand acceptance
```

```text
pending eval INVALID
→ no later Policy action is published
→ existing progress watchdog stays active
→ after arm+hand acceptance covers the outstanding action, trial fences to ARMED/INVALID
```

### Evaluation ownership

```text
initial evaluation-only evidence failure
→ INVALID
→ not executor-owned global FAULT
```

```text
PreparedCommand.unavailable
→ same control behavior with and without formal eval
```

### Audit correctness

```text
EE IK_FAIL
→ ik_ok=False

EE post-IK SAFETY_REJECT
→ ik_ok=True
```

```text
terminal rejected control step
→ evidence attempted before episode finalization
```

```text
formal eval max_frames
→ RecorderIO begins finalization with persisted reason eval:invalid:max_frames
```

### Preserved invariants

```text
no blocking actuator acceptance wait in realtime publish path
no temporal blending / RTC / new scheduler
no source-age action deadline
no learned-policy arm clipping
no compatibility/version machinery
no RecorderIO redesign
no data schema version change
no hardware-worker authority change
```

---

## 18. Deferred integration / real-robot validation

Only after a real trained Policy checkpoint exists and the offline patch is clean:

```text
Policy checkpoint
→ current Policy export/restore validation
→ Real shadow/check
→ inspect timing metrics
→ short physical run with small max_action_steps
→ formal eval trial
→ inspect raw stop_reason / flags / action ids / acceptance provenance
→ then expand trial count
```

For the first physical formal-eval validation, explicitly inspect:

```text
published action_id
arm accepted action_id
hand accepted target action_id
run_generation
SafetyState transitions
recorded frame_status / ik flags / safety flags
stop_reason
truncated
RecorderIO error/failure_count
```

Do not begin large-scale task evaluation until these transaction boundaries are confirmed on hardware.

---

## 19. Final Claude Code / Codex report

Return a concise report with:

### Changed

Exact files and the six mandatory fixes:

```text
acceptance-fenced eval invalidation
initial evidence ownership
PreparedCommand.unavailable parity
EE ik_ok metadata
terminal rejection ordering
persisted max_frames reason
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

The principal remaining risk before physical validation should be timing/load on the real control machine, especially post-control evaluation state/camera copying. Do not propose RTC, temporal smoothing, compatibility layers, or a recorder rewrite as generic follow-up work.
