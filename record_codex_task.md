# DexMani Real Recording Simplification Task

## 0. Scope and repository

Work only in:

~~~text
~/Desktop/dexmani_real
~~~

This task file belongs at:

~~~text
~/Desktop/dexmani_real/record_codex_task.md
~~~

Reviewed remote head when this task was finalized:

~~~text
dexmani_real main
c9bbf5eed7e5b775c7c56237759aa8e68fc74fd3
~~~

Inspect the local repository first. Local HEAD is authoritative if newer. Do not patch by stale line numbers.

This is a personal PhD real-robot research repository. It is not a production recorder, distributed transaction system, generic dataset framework, or multi-user service.

Optimize for:

1. physical safety;
2. correct experiment semantics;
3. raw-data integrity;
4. fast research iteration;
5. simple ownership and readable failure paths;
6. minimal duplicated state and validation.

Prefer delete -> inline -> merge -> rewrite. Do not replace removed complexity with a new abstraction carrying the same concept.

Reference projects are conceptual references only:

- LeRobot: borrow simple episode UX and lightweight timing diagnostics; do not copy silent video-frame dropping.
- ManiUniCon: borrow raw/offline separation; do not copy independent per-modality recording or online timestamp-grid resampling.

---

# 1. Core design rule

The recording stack should preserve facts, not prove its own correctness through layers of status flags.

The final model is:

~~~text
sensor workers
    ↓
latest fresh samples
    ↓
ObservationRow
    ↓
controller / IK / retarget
    ↓
publish valid RobotCommand
    ↓
build EpisodeFrame from the SAME ObservationRow
    ↓
record_sample_ring
    ↓
RecorderIO
    ↓
exact sequence drain through STOP cutoff
    ↓
staging files
    ↓
close + structural validate
    ↓
atomic publish
    ↓
episode_*
~~~

Raw data is experiment truth.

Training admission is an offline responsibility.

Do not introduce:

- command ACK/adoption protocols;
- per-sensor recording transactions;
- online modality resampling;
- synthetic repaired rows;
- generic validity flags;
- quality scores;
- temporal-contract frameworks;
- compatibility layers for deleted recording mechanisms.

---

# 2. Non-negotiable invariants

Preserve these behaviors.

## 2.1 One control row is one causal unit

The controller and recorder must use the same immutable ObservationRow.

Never re-read camera/state after control computation to construct the recorded row.

Keep:

- robot state;
- RGB-D;
- raw VR;
- tactile/current;
- final high-level command target;
- frame status;
- real source timestamps;
- observation timestamp;
- action publication timestamp.

## 2.2 RecorderIO remains a separate process

Do not move HDF5/video writes into the control loop.

RecorderIO remains the single owner of episode serialization.

## 2.3 No silent sample loss

Keep exact sequence-based draining.

If a source sequence has been overwritten or is unavailable:

~~~text
recording failure
→ workflow_failed
→ immediately revoke RUNNING motion
→ retain diagnostic incomplete staging when possible
~~~

Never silently skip or truncate rows.

## 2.4 STOP cutoff remains explicit

STOP must retain:

~~~text
through_sequence
~~~

RecorderIO must drain exactly through that sequence before final save.

Do not replace this with a shared boolean-only recording state.

## 2.5 Atomic publication remains

A normal published episode must still use:

~~~text
.tmp_episode_*
    ↓ close
    ↓ validate
atomic rename
    ↓
episode_*
~~~

Do not weaken this.

## 2.6 Tactile validity remains explicit

XHand aggregate and dense tactile can fail independently while joint state remains usable.

Keep the existing tactile-validity fields.

## 2.7 Pause semantics remain explicit

Keep had_pause.

Pause/resume currently performs:

~~~text
revoke motion
clear controller reference
wait for fresh post-pause state
re-anchor
resume
~~~

This is a real control-reference discontinuity, not merely a large timestamp gap.

Whole-episode teleop training export may continue to reject had_pause=True.

---

# 3. First correctness fix: unify recoverable IK/retarget behavior

Current policy evaluation already treats Cartesian IK failure as recoverable:

~~~text
IK failure
→ do not publish a command
→ record FRAME_IK_FAIL
→ clear remaining action chunk
→ schedule next control step
→ continue
~~~

Teleop recording currently behaves differently:

~~~text
first non-OK frame
→ stop recording immediately
→ mark incomplete
~~~

Remove this inconsistency.

## Required teleop behavior

After run_control_grid_tick():

~~~text
FRAME_OK
  → publish valid command
  → record normal row

FRAME_IK_FAIL / FRAME_RETARGET_FAIL
  → no invalid command publication
  → record diagnostic row
  → keep episode/runtime alive
  → increment existing consecutive-failure counter
~~~

Use the existing failure counter.

When consecutive failures reach the existing threshold:

~~~text
pause
→ revoke motion
→ clear controller reference
→ fresh re-anchor before resume
~~~

Do not turn ordinary IK/retarget failure into a recorder/storage failure.

Do not invent hold-action rows.

Do not delete failed rows from raw data.

Raw -> canonical training export must continue to reject any episode containing non-OK frame status.

Update stale README text that currently says the first IK/retarget failure immediately stops a recorded teleop episode.

---

# 4. Fix the current retain_partial semantic bug by removing the abstraction

Current code can call:

~~~text
save=True
retain_partial=True
~~~

and then atomic-publish the staging directory before the later incomplete-preservation branch sees it.

Do not patch this with more conditionals.

Delete retain_partial from the recording protocol.

Remove it from:

- RecorderClient.stop_episode();
- StopRecording;
- RecorderIO finalization helpers;
- EpisodeRecorder failure_note logic where it exists only to model interrupted-but-structurally-valid episodes;
- teleop/policy callers.

Use only two storage outcomes.

## 4.1 Structurally complete raw episode

If RecorderIO can:

- drain the required rows;
- close writers;
- validate files/layout;
- atomic-publish;

then the result is a normal:

~~~text
episode_*
~~~

Its experiment outcome is described by actual data such as:

- termination_reason;
- frame status;
- had_pause;
- timestamps;
- workflow provenance.

An unsuccessful experiment does not automatically mean storage-incomplete data.

## 4.2 Storage-incomplete episode

Only storage/transport failures create:

~~~text
incomplete_episode_*
    failure_note.json
~~~

Examples:

- record-ring overwrite;
- missing expected source sequence;
- sample decode failure;
- HDF5/video write failure;
- final structural-validation failure;
- RecorderIO crash during active serialization;
- cleanup failure leaving uncertain files.

Do not use incomplete_* merely because IK failed, a policy rollout timed out, a required observation became stale, or the operator ended an abnormal experiment.

---

# 5. Keep required-recording failure fail-fast

Do not remove the current immediate motion revoke when the required recorder becomes unusable.

A real recording/transport failure means the experiment can no longer satisfy its required evidence capture.

Required behavior:

~~~text
recording/storage transport failure
→ workflow_failed = True
→ if currently RUNNING, revoke motion to ARMED
→ do not classify healthy arm/hand hardware as FAULT solely because recorder failed
~~~

Keep the hardware-fault vs workflow-failure distinction.

Do not wait for a later supervisor poll before revoking motion.

---

# 6. Simplify the recorder control/result protocol

The current recorder protocol contains distributed-system-style proof state that is not justified for this single-workstation research stack.

## 6.1 Target control messages

Keep StartRecording conceptually small:

~~~python
StartRecording(
    task,
    operator,
    start_sequence,
    episode_name=None,
)
~~~

Target StopRecording:

~~~python
StopRecording(
    save,
    reason,
    through_sequence,
    had_pause,
)
~~~

Delete from StopRecording:

- retain_partial;
- technical_status;
- deadline_monotonic_ns.

## 6.2 Target start result

RecordingStarted should only return what the caller actually needs, ideally:

~~~python
RecordingStarted(path)
~~~

Do not return recorder-owned episode-budget fields.

## 6.3 Target final result

Use a plain final result, conceptually:

~~~python
@dataclass(frozen=True)
class RecordingResult:
    saved: bool
    path: str | None
    frame_count: int
    reason: str
    error: str | None = None
~~~

poll_stop():

~~~text
None              -> still pending
RecordingResult   -> finished
~~~

Do not represent "not finished yet" by constructing RecordingResult(done=False).

Delete exactly-once result-delivery machinery such as:

- done;
- _terminal_result_delivered;
- branches whose only job is preventing the same cached final result from being returned twice.

A single owner may safely receive the same cached immutable final result again.

---

# 7. Remove recorder finalization deadline proof state

Delete the cross-process deadline/completion proof framework:

- StopRecording.deadline_monotonic_ns;
- RecorderClient._finish_deadline_ns;
- shared.recorder_finish_deadline_ns;
- shared.recorder_completed_ns;
- EpisodeRecorder finalization deadline parameter/checks;
- "completed before deadline but queue delivery was late" logic.

Keep one caller-side bounded wait:

~~~python
recorder.join_stop(timeout=...)
~~~

If the wait expires:

~~~text
workflow_failed
→ revoke if still RUNNING
→ supervisor shutdown handles the stuck recorder
~~~

A slow but eventually successful close is not scientific-data corruption.

Do not mark a structurally valid episode invalid merely because finalization exceeded an arbitrary storage deadline.

---

# 8. Move experiment-duration ownership out of Recorder

Recorder should serialize until STOP. It should not decide experiment duration.

Delete recorder-owned:

- max_frames;
- _max_frames_reached;
- max_frames_stop_reason;
- RecorderIOConfig.max_frames;
- RecordingStarted.max_frames;
- RecorderClient._max_frames;
- truncation logic used only by recorder capacity policy;
- rollout recorder frame margin.

## 8.1 Policy evaluation

PolicyRunner already owns max_running_s.

Keep it as the sole rollout episode-duration owner.

Delete:

~~~text
_ROLLOUT_RECORDER_FRAME_MARGIN
ceil(max_running_s * control_hz) + margin
~~~

from recorder configuration.

## 8.2 Teleop

Preserve current row-budget semantics for this task rather than silently switching to wall-clock semantics.

The teleop owner may derive:

~~~python
max_rows = round(max_record_duration_s * control_hz)
~~~

After accepted raw-row submission reaches that budget:

~~~text
stop(save=True, reason="max_record_duration")
~~~

RecorderClient may keep a simple submitted frame_count because it is useful to the owner.

The recorder backend itself must not auto-stop at capacity.

Do not combine this cleanup with an unrelated redesign of teleop duration semantics.

---

# 9. Remove the min-frames runtime/storage chain

Current min_frames/min_frames_met is only a persisted quality warning, not an actual admission rule.

Delete:

- min_record_duration_s if it has no remaining real behavior after this cleanup;
- RecorderIOConfig.min_frames;
- EpisodeRecorder.min_frames;
- RecordingResult.min_frames_met;
- meta min_frames_met;
- EpisodeReader.min_frames_met;
- visualizer warnings that exist only for this field.

If a future training pipeline requires a minimum episode length, implement the rule directly in offline dataset admission.

Do not propagate a warning through config -> runtime -> storage -> reader.

---

# 10. Increase real buffering instead of increasing protocol complexity

Current record_sample_ring_maxlen is 4.

At 16 Hz this gives only about 250 ms of recorder backlog tolerance while RecorderIO performs video encoding and HDF5/filesystem writes.

Change the default to:

~~~python
record_sample_ring_maxlen = 16
~~~

This gives about 1 s of transient backlog tolerance at 16 Hz and costs only tens of MiB of shared memory for current 640x480 RGB-D payloads.

Do not copy LeRobot's behavior of dropping video frames when a queue is full.

Overflow remains a hard recording failure.

---

# 11. Remove shared recorder_consumed_sequence

Current overwrite protection is duplicated:

- producer pre-check using shared recorder_consumed_sequence;
- RecorderIO backlog check using last_sample_sequence;
- exact read_sequence(expected) verification.

Keep the latter two inside RecorderIO.

Delete:

- shared.recorder_consumed_sequence;
- producer-side overflow pre-check;
- RecorderIO writes to shared consumed sequence.

RecorderIO keeps only its local:

~~~text
last_sample_sequence
~~~

Correctness requirement:

~~~text
if latest - last_sample_sequence > ring.maxlen
or read_sequence(expected) fails
→ loud recording failure
~~~

No silent loss is allowed.

---

# 12. Make DISCARD cheap and clean

Operator DISCARD and start-cancel are explicit statements that the data is not wanted.

Do not fully finalize/validate/publish discarded data.

For save=False:

~~~text
stop production / establish stop
→ close active writers
→ delete staging
→ return saved=False
~~~

It is acceptable to skip processing remaining unconsumed sample-ring rows because the entire episode is intentionally discarded.

Do not convert a normal discard into incomplete_*.

Do not write a clean-discard aborted manifest.

Delete the normal .aborted.json path if it has no remaining real failure-diagnostic role.

Storage failures should continue to use incomplete_* + failure_note.json.

---

# 13. Remove technical-status/result-sidecar self-proof

The current system has parallel validity sources:

~~~text
HDF5 technical_status
episode_*.result.json technical_status
directory publication state
frame_status
termination_reason
had_pause
~~~

This is too many truths.

The target meaning should be:

~~~text
episode_*           = structurally complete raw storage
incomplete_episode* = storage transaction incomplete/failed
~~~

Training eligibility belongs to dataset processing.

Remove the late-invalidating sidecar mechanism:

- write_recording_failure();
- episode_*.result.json;
- record_episode_path shared state;
- EpisodeReader.require_valid() sidecar lookup.

Remove technical_status once its remaining consumers are replaced by concrete facts.

Do not silently reinterpret technical_status with a new meaning. If removing persisted technical_status changes the current raw metadata contract, perform one intentional schema bump and update all consumers together rather than keeping a compatibility branch.

After sidecar removal, re-check whether shared.is_recording has any real consumer. If not, delete it.

---

# 14. Dataset admission after technical_status removal

Raw -> canonical Zarr remains whole-episode all-or-nothing.

Do not repair, split, resample, interpolate or silently salvage a trajectory.

Teleop admission should use concrete facts:

1. published/current supported raw layout;
2. workflow == teleop;
3. had_pause == False;
4. termination_reason is a normal teleop ending;
5. every flag_frame_status == FRAME_OK;
6. required tactile-valid fields are true;
7. required floating arrays are finite;
8. observation/action timestamps are positive and strictly increasing;
9. no existing gross control-gap rule is violated;
10. action timestamp does not precede observation completion;
11. required source timestamps are present;
12. RGB-D/calibration/layout checks pass.

Define a small explicit normal-termination allowlist from actual current teleop reasons, for example the real equivalents of:

~~~text
STOP
HOME
max_record_duration
~~~

Do not guess strings. Inspect OperatorCommand and the current stop() call sites and normalize the reason names once.

Abnormal reasons such as hardware/resource failure, interrupted shutdown, camera unavailable, etc. must be rejected explicitly.

Policy-eval raw episodes remain diagnostic/evaluation data and must not accidentally enter fixed-dt teleop training export.

---

# 15. Keep had_pause; do not create more persisted event flags

had_pause is justified because it denotes a controller-reference reset.

Do not add analogous persisted flags such as:

- had_ik_failure;
- had_camera_stale;
- had_workflow_failure;
- had_timeout.

Those events are already represented by:

- frame_status;
- termination_reason;
- timestamps.

Keep one fact per concept.

---

# 16. Schema cleanup: do it once, not piecemeal

The current raw schema is v32.

Do not create several schema versions during this task.

After runtime/lifecycle changes are stable, inspect whether a single next-version cleanup is worthwhile.

Candidates include:

- removing technical_status metadata;
- removing task_success="unknown";
- removing duplicated stop_reason vs termination_reason;
- removing duplicated fps vs control_hz;
- removing wall_fps if unused;
- removing has_camera / has_timestamps / camera_stream_frames if structural validation already proves them;
- removing truncated after recorder-owned max-frame logic disappears;
- removing raw float timestamp if observation_timestamp_ns is the sole control-row time source;
- removing IPC camera_present if recording rows always require RGB-D.

Only remove persisted fields when all current consumers are updated in the same change.

Do not add runtime compatibility branches for old schemas. Important old raw data can be migrated once if needed.

If the schema cleanup creates too much unrelated risk, leave v32 persisted-field cleanup for a separate change after the runtime simplification is complete.

Correct runtime behavior has higher priority than cosmetic HDF5 cleanup.

---

# 17. EpisodeTiming cleanup

Current EpisodeTiming.grid_duration_s is actually populated from recorded timestamp span, not a nominal grid duration.

Only retain timing abstractions with real consumers.

Prefer eventually reducing to something like:

~~~python
EpisodeTiming(
    rate_hz,
    dt_s,
)
~~~

Compute actual raw timestamp span on demand.

Do not maintain grid/wall/non_sampled timing ontology unless a concrete consumer requires it.

Do not combine this with runtime behavior changes if doing so expands the blast radius unnecessarily.

---

# 18. Keep provenance validation

Do not delete provenance merely because other validators are being removed.

Current policy-eval provenance carries concrete experimental context including policy/checkpoint/inference and point-cloud configuration.

Keep the existing lightweight normalization unless inspection proves a check has no actual purpose.

This cleanup is about duplicated lifecycle/validity state, not deleting every validator.

---

# 19. Keep EpisodeFinalizationError and local lifecycle guards that still separate recovery

Do not mechanically delete:

- EpisodeFinalizationError;
- _finishing;
- resources_released;
- local immutable result caching;

if they still distinguish expected storage-finalization failure from an unexpected RecorderIO programming/process crash.

Delete only machinery with no independent behavioral value.

---

# 20. Expected ownership after the refactor

## Teleop / PolicyRunner

Own:

- when an episode starts/stops;
- experiment time/row budget;
- operator commands;
- IK retry/pause behavior;
- termination reason.

## RecorderClient

Own:

- START/STOP request lifecycle;
- frame submission;
- simple local frame count;
- waiting for recorder results;
- fail-fast workflow response when required recording becomes unavailable.

## RecorderIO

Own:

- record-sample sequence integrity;
- backlog/overwrite detection;
- draining through STOP cutoff;
- serialization process failure.

## EpisodeRecorder

Own only storage:

- staging directory;
- file writers;
- close;
- structural validation;
- atomic publication;
- incomplete storage preservation.

## Dataset processing

Own:

- training eligibility;
- whole-episode rejection;
- numerical transforms;
- canonical Zarr generation.

No other layer should invent a second validity system.

---

# 21. Likely files

Inspect actual current code before editing. Likely relevant files include:

~~~text
dexmani_real/recording/client.py
dexmani_real/recording/io_worker.py
dexmani_real/recording/recorder.py
dexmani_real/recording/frame.py
dexmani_real/recording/storage/schema.py
dexmani_real/recording/storage/reader.py
dexmani_real/ipc/channels.py
dexmani_real/ipc/schema.py
dexmani_real/teleop/loop.py
dexmani_real/teleop/session.py
dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py
dexmani_real/runtime/processes.py
dexmani_real/dataset/processing.py
examples/visualize_episode.py
README.md
AGENTS.md
~~~

Do not perform unrelated deployment, calibration, point-cloud, IK-algorithm or style refactors.

Preserve the recently simplified deployment/action/calibration ownership from the current main branch.

---

# 22. Required implementation order

Use this order to minimize debugging cost.

1. Inspect local git status and preserve unrelated work.
2. Re-read actual current definitions and all consumers before editing.
3. Make teleop IK/retarget failures recoverable and align them with policy-eval semantics.
4. Remove retain_partial and fix episode/incomplete storage meaning.
5. Increase record_sample_ring_maxlen from 4 to 16.
6. Implement simple/fast DISCARD behavior and remove normal aborted manifests.
7. Move max episode budget out of Recorder and remove recorder capacity auto-stop.
8. Remove min-frames runtime/storage propagation.
9. Simplify RecordingStarted / RecordingResult / poll_stop / join_stop.
10. Remove shared finalization deadline/completed state.
11. Remove shared recorder_consumed_sequence and producer-side duplicate overflow proof.
12. Remove result-sidecar/record_episode_path machinery.
13. Replace training dependence on technical_status with concrete termination/frame/pause checks.
14. Remove technical_status cleanly; bump raw schema once only if required by persisted-contract policy.
15. Re-check shared.is_recording and delete it if no independent consumer remains.
16. Update README/AGENTS comments only where behavior changed.
17. Run focused offline smoke checks.
18. Search globally for stale fields/concepts before final handoff.

Do not use git reset --hard.

Do not run hardware-affecting examples.

---

# 23. Focused offline smoke checks

The repository intentionally has no general committed tests directory. Use focused pure/offline checks and existing smoke-test style where appropriate.

At minimum validate these cases.

## Lifecycle

### normal save

~~~text
START
→ N rows
→ STOP(save=True, cutoff=N)
→ exact N rows
→ episode_* published
~~~

### discard

~~~text
START
→ rows
→ DISCARD
→ no episode_*
→ no incomplete_*
→ no .aborted.json
~~~

### start cancellation

~~~text
START
→ cancel before real run
→ clean cleanup
→ no persisted episode
~~~

## Control failures

### transient teleop IK failure

~~~text
FRAME_IK_FAIL recorded
→ no invalid RobotCommand publish
→ episode remains active
→ next control attempts continue
~~~

### repeated teleop failures

~~~text
existing failure threshold reached
→ pause
→ controller reference cleared
→ resume requires fresh post-pause re-anchor
~~~

### policy-eval IK failure

Verify existing behavior remains:

~~~text
diagnostic row
→ action chunk cleared
→ runtime continues
~~~

## Recorder integrity

### STOP with backlog

RecorderIO must drain every sequence through cutoff.

### ring overwrite

Must fail loudly, set workflow_failed and revoke motion.

### missing expected sequence

Must fail loudly.

### write/finalization failure

Must not publish normal episode_*.

Preserve incomplete_* when useful diagnostic staging exists.

### recorder result timeout

Caller reports workflow failure and shutdown proceeds without shared deadline-proof state.

## Dataset

### normal teleop episode

Exports successfully.

### episode with any non-OK frame

Whole episode rejected.

### had_pause=True

Whole episode rejected.

### abnormal termination_reason

Whole episode rejected.

### policy_eval provenance

Rejected by fixed-dt teleop exporter.

---

# 24. Low-cost repository checks

Run:

~~~bash
python -m compileall -q dexmani_real examples

ruff format --check dexmani_real examples

ruff check --select F401,F821,F822,F823,I dexmani_real examples

git diff --check
~~~

If Ruff is unavailable, report it. Do not install or upgrade packages merely to run the check.

Also run relevant existing pure Python smoke tests already present in the repository if they do not connect hardware.

Never execute:

- teleoperation;
- policy rollout against hardware;
- homing;
- camera capture;
- XHand/xArm SDK discovery;
- calibration writes.

---

# 25. Final stale-reference audit

Before finishing, search at least for:

~~~text
retain_partial
technical_status
result.json
write_recording_failure
record_episode_path
recorder_finish_deadline_ns
recorder_completed_ns
recorder_consumed_sequence
max_frames_stop_reason
max_frames_reached
_ROLLOUT_RECORDER_FRAME_MARGIN
min_frames_met
min_frames
aborted.json
shared.is_recording
FRAME_IK_FAIL
FRAME_RETARGET_FAIL
had_pause
termination_reason
~~~

Not every occurrence must become zero.

Expected remaining examples:

- FRAME_IK_FAIL / FRAME_RETARGET_FAIL in frame diagnostics and dataset rejection;
- had_pause in teleop recording metadata/admission;
- termination_reason in raw metadata/admission.

Every remaining removed-concept occurrence must have a concrete current purpose.

---

# 26. Acceptance criteria

The task is complete only when all of the following are true.

## Runtime semantics

- teleop and action_ee policy evaluation treat isolated IK failure as recoverable;
- failed IK never publishes an invalid new command;
- repeated teleop failures use the existing pause/re-anchor path;
- recorder/storage failure still revokes active motion immediately.

## Recording semantics

- controller and recorder reuse the same ObservationRow;
- no silent row drop exists;
- STOP cutoff is exact;
- RecorderIO remains separate;
- normal save still structurally validates before atomic publish;
- episode_* means storage-complete raw data;
- incomplete_* means storage/serialization failure, not merely experiment failure;
- normal DISCARD leaves no dataset/manifest clutter.

## Simplicity

- retain_partial is gone;
- deadline/completed cross-process proof state is gone;
- exactly-once final-result delivery state is gone;
- recorder no longer owns experiment max/min frame policy;
- shared consumed-sequence proof is gone;
- result-sidecar invalidation is gone;
- no replacement framework with equivalent complexity was introduced.

## Data admission

- teleop training admission uses concrete experiment facts;
- had_pause remains meaningful and enforced;
- non-OK frame status rejects the whole episode;
- abnormal termination rejects the whole episode;
- policy_eval raw data does not enter teleop fixed-dt training;
- no row repair/splitting/resampling is introduced.

## Performance

- record_sample_ring default is 16;
- no new blocking disk/video work is added to the control loop;
- DISCARD is cheaper than SAVE;
- no unnecessary polling/IPC state is added.

## Scope

No regression or unrelated redesign in:

- arm/hand hardware worker behavior;
- SafetyState;
- run_id stale-command fencing;
- robot-command latest-target semantics;
- camera/point-cloud calibration ownership;
- online IK algorithm itself;
- tactile semantics;
- raw->Zarr action semantics;
- deployment action ownership.

---

# 27. Final decision rule

For every recording mechanism kept or added, ask:

> What concrete bad experiment, lost row, corrupt file, unsafe continuation, or ambiguous scientific datum does this prevent?

If the answer is only:

> it proves another internal state is correct

delete it.

For every stored field, ask:

> Is this a real experiment fact that cannot be unambiguously derived from already stored facts?

If not, do not add another persisted truth.

The target is not the smallest possible recorder.

The target is the smallest recorder that is still reliable for repeated real-robot PhD experiments.
