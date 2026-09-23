# DexMani Real Recording Runtime Simplification Task

## 0. Scope and reviewed baseline

Work only in:

~~~text
~/Desktop/dexmani_real
~~~

This task file belongs at:

~~~text
~/Desktop/dexmani_real/record_codex_task.md
~~~

Reviewed remote baseline:

~~~text
dexmani_real main
aab1982e5a83bf48d25e0a5e202f3087b402254b
~~~

The code baseline immediately before this task file was added was:

~~~text
c9bbf5eed7e5b775c7c56237759aa8e68fc74fd3
~~~

Inspect local `git status --short` and local HEAD first. Local code is authoritative if newer. Preserve unrelated work. Do not patch by stale line number.

This is a personal PhD real-robot research repository, not a production recorder, distributed transaction system, generic dataset framework, or multi-user service.

Optimize for:

1. physical safety;
2. correct experiment/control semantics;
3. raw-data integrity;
4. fast research iteration;
5. simple ownership and readable failure paths;
6. minimal duplicated IPC/protocol state.

Prefer delete -> inline -> merge -> rewrite. Do not remove a cheap mechanism that prevents a concrete bad experiment merely to reduce LOC.

Reference projects are conceptual references only:

- LeRobot: borrow simple episode UX and lightweight diagnostics; do not copy silent frame dropping.
- ManiUniCon: borrow raw/offline separation; do not copy independent per-modality recording, online timestamp-grid filling, or min-length truncation.

---

# 1. Task boundary: runtime/protocol cleanup, not raw-schema migration

This task MUST simplify recording runtime/control flow while keeping the current raw v32 persisted contract readable.

Do not bump the raw episode schema in this task.

Do not delete or reinterpret current persisted v32 fields such as:

- `technical_status`;
- `had_pause`;
- `timestamp`;
- `min_frames_met`;
- current timing/meta attrs;
- current camera/calibration attrs.

Do not redesign `EpisodeTiming` in this task.

Do not remove the current min-frame metadata path in this task.

Do not perform a broad HDF5 metadata cleanup in this task.

Those are valid future cleanup topics, but combining them with recorder-runtime simplification would unnecessarily create historical-data migration risk.

The main target here is:

> Fix behavior and remove unnecessary cross-process/lifecycle machinery without changing the scientific raw layout.

---

# 2. Core recording architecture to preserve

The intended architecture remains:

~~~text
sensor workers
    ↓
latest required fresh samples
    ↓
one immutable ObservationRow
    ↓
controller / IK / retarget
    ↓
publish valid RobotCommand only
    ↓
build EpisodeFrame from the SAME ObservationRow
    ↓
record_sample_ring
    ↓
RecorderIO
    ↓
sequence-aware serialization
    ↓
staging files
    ↓
close + structural validation
    ↓
atomic publish
    ↓
episode_*
~~~

Raw episodes are experiment source of truth.

Canonical training admission remains offline and whole-episode all-or-nothing.

Do not introduce:

- command ACK/adoption protocols;
- per-modality recording transactions;
- runtime sensor resampling;
- bad-row repair;
- synthetic hold-action rows;
- generic observation-valid flags;
- quality scores;
- temporal-contract frameworks;
- a replacement state machine with different names.

---

# 3. Non-negotiable invariants

## 3.1 Same control row

Controller and recorder must use the same ObservationRow.

Never re-read camera/state after control computation to manufacture the recorded observation for that action.

Keep real experiment facts:

- robot state;
- RGB-D;
- VR data;
- hand current/tactile and tactile validity;
- final high-level command target;
- pre-IK intent where currently defined;
- frame status;
- observation/action/source timestamps.

## 3.2 RecorderIO remains separate

Do not move MP4/HDF5/filesystem work into the control loop.

## 3.3 No silent row loss

Keep exact sequence checks.

Missing/overwritten rows must remain a loud recording failure.

## 3.4 STOP cutoff remains explicit

A save STOP must retain `through_sequence`.

RecorderIO must drain exactly through that sequence before normal finalization.

Do not replace this with a boolean-only recording protocol.

## 3.5 Atomic publication remains

Normal saved raw data still follows:

~~~text
.tmp_episode_*
→ close writers
→ validate
→ atomic rename
→ episode_*
~~~

## 3.6 Pause remains meaningful

Keep `had_pause`.

Pause/resume currently revokes motion, clears controller reference and requires fresh re-anchor. That is a real control discontinuity.

Raw -> Zarr may continue rejecting every `had_pause=True` teleop episode.

## 3.7 Required recorder failure remains fail-fast

If required recording becomes unavailable while RUNNING:

~~~text
workflow_failed = True
→ immediately revoke active motion to ARMED
~~~

Do not classify healthy arm/hand hardware as FAULT solely because RecorderIO failed.

Do not delay this revoke until a later supervisor poll.

---

# 4. P0 correctness fix: make teleop IK/retarget failures recoverable

Current policy evaluation already treats Cartesian IK failure as recoverable:

~~~text
IK fail
→ no invalid command publication
→ record FRAME_IK_FAIL
→ clear remaining action chunk
→ schedule next step
→ continue
~~~

Current teleop recording instead terminates on the first non-OK frame.

Unify the semantics.

## Required teleop behavior

`run_control_grid_tick()` already returns `target=None` for retarget/IK failure, so no partial arm/hand command is published. Preserve this.

After a control tick:

~~~text
FRAME_OK
  → command may publish
  → normal row recorded

FRAME_IK_FAIL / FRAME_RETARGET_FAIL
  → no new command published
  → diagnostic row recorded
  → keep teleop episode active
  → increment existing consecutive-failure counter
~~~

Use the existing `_DEBUG_FAILURE_LIMIT` behavior:

~~~text
repeated failures
→ pause
→ revoke motion
→ clear reference
→ require fresh post-pause re-anchor
~~~

Do not:

- auto-stop the episode on the first IK/retarget failure;
- publish a hold command as fake action;
- delete the failed raw row;
- mark the recorder/storage path failed merely because control IK failed.

Canonical teleop export must continue rejecting any episode containing a non-OK frame status.

Update README text that currently says the first recorded IK/retarget failure immediately terminates the episode.

---

# 5. P0 correctness fix: remove `retain_partial`

Current `save=True + retain_partial=True` semantics are inconsistent with atomic publication: a staging directory may already have been published before the later incomplete-preservation branch tries to rename it.

Do not patch this with more branches.

Delete `retain_partial` from:

- `RecorderClient.stop_episode()`;
- `StopRecording`;
- RecorderIO finalization helpers;
- EpisodeRecorder finalization arguments used only for this concept;
- teleop/policy callers.

Use the following storage distinction.

## 5.1 Structurally complete raw capture

If RecorderIO successfully:

- owns all required source rows;
- closes writers;
- validates storage;
- atomic-publishes;

then it is a normal `episode_*` directory.

The experiment itself may still be abnormal, and current raw v32 `technical_status`, `had_pause`, frame status and termination metadata may describe that.

Do not call a structurally complete hardware/policy/observation failure an `incomplete_*` storage transaction.

## 5.2 Storage/recording transaction failure

Use `incomplete_episode_*` only for real recording transaction failure, for example:

- ring overwrite before a required saved row was consumed;
- missing expected sequence;
- sample decode failure;
- HDF5/video write failure;
- structural finalization/validation failure;
- RecorderIO crash/cleanup failure leaving uncertain staging.

Keep `failure_note.json` only for this real storage-failure path.

---

# 6. Preserve raw v32 `technical_status` for now; remove only the sidecar proof

Do NOT remove the persisted `technical_status` contract in this task.

Do not reinterpret it under the same schema version.

Keep the existing v32 field and existing training-side `require_valid()` meaning where required.

However, remove the extra late-invalidating sidecar mechanism:

- `write_recording_failure()`;
- `episode_*.result.json`;
- `record_episode_path` shared state;
- `runtime/processes.py` logic that writes that sidecar.

After sidecar removal, search all consumers of `shared.is_recording`.

If it has no independent purpose, delete `shared.is_recording` and all writes to it.

Do not replace the deleted sidecar with another marker file or shared validity flag.

## 6.1 Ownerless runtime shutdown

If the runtime ends while RecorderIO still has an active recording and no explicit STOP was received, do not classify it as a storage-write failure merely because the owner disappeared.

Use the latest committed sample sequence as the emergency cutoff, close the transaction normally if possible, and mark the current v32 experiment status/termination as abnormal (for example `runtime_shutdown` + invalid technical status).

If storage itself fails during that close, then preserve `incomplete_*`.

The resulting abnormal but structurally complete raw episode must remain rejected by current training validity rules.

---

# 7. Simplify STOP finalization timing: remove cross-process deadline proof

Delete the shared deadline/completion proof framework:

- `StopRecording.deadline_monotonic_ns`;
- `RecorderClient._finish_deadline_ns`;
- `shared.recorder_finish_deadline_ns`;
- `shared.recorder_completed_ns`;
- EpisodeRecorder deadline parameter/checks;
- "completed before deadline but queue delivery was late" branches.

Keep ordinary caller-side timeouts:

- START acknowledgement timeout;
- bounded STOP/join wait.

A conceptual finalization flow should be:

~~~text
STOP
→ RecorderIO finishes
→ result queue
→ client join_stop(timeout)
~~~

If the bounded client wait expires:

~~~text
recording transport considered unavailable
→ workflow_failed
→ revoke if still RUNNING
→ supervisor handles shutdown
~~~

Do not declare already-written scientific data corrupt solely because close/result delivery was slow.

Do not remove all recorder timeouts indiscriminately; remove only the duplicated cross-process proof state.

---

# 8. Simplify `RecordingResult` lifecycle

Current `done=False` placeholder results and `_terminal_result_delivered` implement exactly-once message semantics inside a single-owner client.

Remove that complexity.

Target behavior:

~~~python
poll_stop() -> RecordingResult | None
~~~

where:

~~~text
None             = still pending
RecordingResult  = final result
~~~

A final immutable result may be cached and returned again safely.

Delete state/branches whose only purpose is ensuring a final result is delivered exactly once.

Keep real lifecycle state:

- recording active;
- stop pending;
- recorder unavailable;
- last final result;
- episode path;
- local submitted frame count.

Keep START acknowledgement as a real boundary.

---

# 9. Move normal episode-length ownership out of Recorder, but retain a hard resource guard

Recorder must not own normal experiment termination.

## 9.1 Policy evaluation

`PolicyRunner.max_running_s` remains the sole normal rollout-duration owner.

Remove rollout normal-budget duplication such as:

~~~text
_ROLLOUT_RECORDER_FRAME_MARGIN
ceil(max_running_s * control_hz) + margin
RecordingStarted.max_frames
max_frames_stop_reason
client auto-stop at recorder capacity
~~~

## 9.2 Teleop

Preserve current row-budget semantics in this task.

The teleop owner may derive:

~~~python
max_rows = round(max_record_duration_s * control_hz)
~~~

After successful raw-row submission reaches this owner budget:

~~~text
stop(save=True, reason="max_record_duration")
~~~

Do not silently change this task into a wall-clock-duration redesign.

## 9.3 Keep one simple hard recorder safety ceiling

Do NOT remove all frame-capacity protection.

A simple high hard cap inside the storage writer is useful against runaway resource consumption.

The hard cap:

- is not a normal experiment stop reason;
- is not sent to RecorderClient in `RecordingStarted`;
- does not use `max_frames_stop_reason`;
- must not create a valid truncated episode;
- if exceeded, behaves as a recording/resource failure.

The existing `DEFAULT_MAX_RECORD_FRAMES` may remain or be simplified into this role.

Prefer a direct exception/error over the current "accepted-last-row + False return + special-case" capacity protocol.

Normal teleop/policy limits should stop well before the hard guard.

---

# 10. Increase actual buffering: record ring 4 -> 16

Change default:

~~~python
record_sample_ring_maxlen = 16
~~~

At 16 Hz this provides about one second of transient RecorderIO backlog tolerance.

This is more useful than elaborate deadline/status machinery.

Keep overflow as a hard recording failure.

Do not copy LeRobot-style silent frame dropping.

---

# 11. Remove shared `recorder_consumed_sequence`

Current loss detection is duplicated:

1. producer pre-check against shared consumed sequence;
2. RecorderIO `latest - last_sample_sequence > ring.maxlen`;
3. exact `read_sequence(expected)` check.

Keep 2 and 3 inside RecorderIO.

Delete:

- `shared.recorder_consumed_sequence`;
- producer-side pre-overwrite proof;
- RecorderIO writes to shared consumed sequence.

RecorderIO keeps a local `last_sample_sequence`.

If a required saved row has been overwritten or cannot be read exactly, fail loudly.

A producer overwrite may occur before detection; that is acceptable because the episode fails instead of silently corrupting data.

---

# 12. Make DISCARD cheap, but preserve sequence continuity

For `save=False`, the episode is intentionally unwanted.

Do not spend time finishing/validating data that will be deleted.

A discard path may:

~~~text
stop further producer submission
→ capture through_sequence
→ close writers
→ delete staging
→ return saved=False
~~~

It may skip decoding/writing remaining unconsumed ring rows.

However, because the record ring uses a global monotonically increasing sequence across episodes, a fast discard MUST advance RecorderIO's local sequence cursor to the discard cutoff before the next START.

Conceptually:

~~~text
last_sample_sequence = through_sequence
~~~

after the discard is accepted/cleaned.

This is required so the next START begins at the correct `latest_sequence + 1` boundary.

Do not let fast DISCARD create a false missing-sequence error in the next episode.

Do not:

- create `incomplete_*` for a normal discard;
- write a normal `.aborted.json`;
- structurally validate data the user explicitly discarded.

If an actual writer/cleanup failure occurs while discarding, report that real failure normally.

---

# 13. Remove normal `.aborted.json` clutter

Delete the clean-discard aborted-manifest behavior.

Normal:

- DISCARD;
- start-cancelled;
- empty unwanted capture;

should leave no dataset artifact after successful cleanup.

Keep failure notes only with real `incomplete_*` storage failures.

Do not replace `.aborted.json` with a differently named normal-discard manifest.

---

# 14. Keep current training admission; do not add a second new validity framework

Because raw v32 `technical_status` remains in this task, do not simultaneously invent a new termination-reason validity ontology.

Keep current whole-episode admission based on the existing concrete checks:

- current supported raw schema/layout;
- `technical_status` validity;
- `had_pause == False`;
- teleop workflow provenance;
- all frame statuses OK;
- tactile validity;
- finite required arrays;
- timestamp ordering/gross-gap checks;
- action timestamp after observation;
- source timestamps present;
- RGB-D/calibration/geometry validity.

The important new effect from this task is:

- recoverable IK/retarget rows may now exist in otherwise continued raw teleop captures;
- those rows are already rejected by the existing non-OK frame-status rule.

Do not add row salvage, splitting, repair or resampling.

A later raw-schema cleanup can replace `technical_status` with a more direct termination-based contract after migration is planned.

---

# 15. Explicitly out of scope for this task

Do not modify these unless required to fix a direct regression from the changes above:

- raw schema version;
- raw float `timestamp` removal;
- `camera_present` schema cleanup;
- `min_frames` / `min_frames_met` persisted semantics;
- duplicated legacy meta attrs such as `fps`, `wall_fps`, `stop_reason`, etc.;
- `EpisodeTiming` redesign;
- provenance normalization;
- point-cloud/calibration ownership;
- deployment artifact contracts;
- IK solver algorithm/tolerances;
- action semantics;
- tactile semantics.

Do not turn this task into a general recording-schema rewrite.

---

# 16. Keep useful local lifecycle guards

Do not mechanically delete:

- `EpisodeFinalizationError`;
- `_finishing`;
- `resources_released`;
- START acknowledgement;
- bounded client wait;
- local final-result caching;

if they still prevent or classify a concrete failure.

Simplification means removing redundant cross-process proof state, not removing every check.

---

# 17. Target ownership after refactor

## Teleop / PolicyRunner

Own:

- episode begin/end;
- operator/control termination;
- normal episode budget;
- IK/retarget recovery and pause behavior;
- experiment-level technical status under current v32 semantics.

## RecorderClient

Own:

- START/STOP request lifecycle;
- frame publication to the record ring;
- local submitted frame count;
- bounded wait for recorder result;
- fail-fast reaction when required recorder transport is unavailable.

## RecorderIO

Own:

- local sequence cursor;
- saved-row continuity;
- STOP cutoff draining;
- serialization worker health;
- emergency abnormal close when owner/runtime disappears.

## EpisodeRecorder

Own storage only:

- staging;
- camera/HDF5 writers;
- hard resource guard;
- close;
- structural validation;
- atomic publish;
- incomplete storage preservation.

## Dataset processing

Own training eligibility and canonicalization.

No other layer should add another validity system.

---

# 18. Likely files

Inspect actual code before editing. Likely relevant:

~~~text
dexmani_real/recording/client.py
dexmani_real/recording/io_worker.py
dexmani_real/recording/recorder.py
dexmani_real/ipc/channels.py
dexmani_real/teleop/loop.py
dexmani_real/teleop/session.py
dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py
dexmani_real/runtime/processes.py
dexmani_real/dataset/processing.py
README.md
AGENTS.md
~~~

Touch `recording/frame.py`, raw schema, reader, visualizer, etc. only if required by a concrete compile/runtime dependency from the above changes.

Do not perform unrelated deployment, calibration, point-cloud, IK-algorithm, schema or style refactors.

---

# 19. Required implementation order

Use this order.

1. Inspect `git status --short`, local HEAD and actual consumers.
2. Trace the current recording path end-to-end:
   ~~~text
   teleop/policy owner
   -> RecorderClient
   -> record ring
   -> RecorderIO
   -> EpisodeRecorder
   -> EpisodeReader/dataset admission
   ~~~
3. Make teleop IK/retarget failure recoverable.
4. Remove `retain_partial` and correct complete-vs-incomplete storage semantics.
5. Implement abnormal owner/runtime shutdown close without misclassifying it as storage failure.
6. Change record ring default from 4 to 16.
7. Move normal max-duration/max-row ownership out of Recorder; retain only a high hard guard.
8. Add the sequence-safe fast DISCARD path and remove normal aborted manifests.
9. Simplify RecordingStarted/RecordingResult/poll/join semantics.
10. Remove shared finalization deadline/completed proof state.
11. Remove shared consumed-sequence proof.
12. Remove result sidecar + `record_episode_path`; delete `shared.is_recording` if now dead.
13. Update stale README/comments.
14. Run focused offline smoke checks.
15. Search for dead references and inspect the full final diff.

Do not bump raw schema.

Do not use `git reset --hard`.

Do not execute hardware-affecting commands.

---

# 20. Required focused offline checks

Use existing pure/offline smoke-test style or focused one-off tests. Do not restore a broad committed tests directory.

## 20.1 Normal save

~~~text
START
→ N exact source rows
→ STOP(save=True, cutoff=N)
→ exact rows serialized
→ validation succeeds
→ episode_* published
~~~

## 20.2 STOP with RecorderIO backlog

Ensure every sequence through cutoff is serialized exactly once.

## 20.3 Transient teleop IK failure

Verify:

~~~text
FRAME_IK_FAIL row recorded
no RobotCommand published for failed tick
episode remains active
next control tick is allowed
~~~

Do the equivalent for retarget failure if practical without hardware.

## 20.4 Repeated teleop failure

Existing threshold still causes pause/re-anchor behavior.

## 20.5 Policy-eval IK failure regression

Existing behavior remains:

~~~text
failed diagnostic row
action chunk cleared
runtime continues
~~~

## 20.6 Training rejection

A raw teleop episode containing any non-OK frame status is still rejected whole.

A `had_pause=True` episode is still rejected whole.

## 20.7 Normal discard

~~~text
START
→ some consumed and some unconsumed rows
→ DISCARD
→ writers close
→ staging deleted
→ local recorder sequence cursor advances to cutoff
→ next START accepts the next global sequence
→ no episode_*
→ no incomplete_*
→ no .aborted.json
~~~

This check is important.

## 20.8 Ring overwrite / missing sequence

Must:

- fail loudly;
- latch workflow failure;
- revoke active motion;
- not publish a normal episode.

## 20.9 Real storage-write/finalization failure

Must not publish a normal `episode_*`.

Preserve useful `incomplete_*` staging after resources close when possible.

## 20.10 Runtime/owner disappearance

With an active capture and no explicit STOP:

- use a bounded emergency cutoff;
- attempt normal close of structurally usable raw data;
- mark current v32 experiment status abnormal;
- reject it via existing training validity;
- only use `incomplete_*` if storage close itself fails.

## 20.11 Result wait timeout

Client fails the workflow and shutdown remains bounded without the removed shared deadline/completion proof fields.

## 20.12 Hard recorder frame ceiling

Exceeding the defensive hard cap is a recording/resource failure, not a valid auto-stopped/truncated episode.

---

# 21. Low-cost repository checks

Run:

~~~bash
python -m compileall -q dexmani_real examples

ruff format --check dexmani_real examples

ruff check --select F401,F821,F822,F823,I dexmani_real examples

git diff --check
~~~

If Ruff is unavailable, report it; do not install/upgrade the experiment environment merely for linting.

Run relevant existing offline smoke tests that do not connect hardware.

Never run:

- teleoperation;
- real policy rollout;
- homing;
- physical replay;
- camera capture;
- live xArm/XHand/RealSense discovery;
- calibration writes.

---

# 22. Final stale-reference audit

Search at minimum for:

~~~text
retain_partial
deadline_monotonic_ns
recorder_finish_deadline_ns
recorder_completed_ns
recorder_consumed_sequence
_terminal_result_delivered
max_frames_stop_reason
max_frames_reached
_ROLLOUT_RECORDER_FRAME_MARGIN
aborted.json
write_recording_failure
record_episode_path
shared.is_recording
FRAME_IK_FAIL
FRAME_RETARGET_FAIL
technical_status
had_pause
~~~

Expected outcome:

- `retain_partial`: zero;
- shared deadline/completed proof: zero;
- shared consumed sequence: zero;
- result-sidecar path: zero;
- normal aborted manifest: zero;
- rollout frame-margin/budget protocol: zero;
- `shared.is_recording`: zero if no independent consumer remains;
- `FRAME_IK_FAIL/FRAME_RETARGET_FAIL`: remain in diagnostics/control/data rejection;
- `technical_status`: remains only because raw v32 contract is intentionally preserved;
- `had_pause`: remains intentionally.

Inspect every remaining occurrence rather than mechanically forcing all searches to zero.

---

# 23. Acceptance criteria

The task is complete only when all of the following are true.

## Control behavior

- isolated teleop IK/retarget failures are recoverable;
- failed control ticks publish no invalid new command;
- repeated failures still use existing pause/fresh-reanchor behavior;
- action_ee policy-eval failure semantics are not regressed.

## Required-recorder safety

- real recorder/transport failure still immediately revokes active motion;
- recorder failure remains workflow failure, not automatic arm/hand FAULT.

## Recording integrity

- same ObservationRow is used for control and raw recording;
- no silent row loss;
- SAVE drains exact STOP cutoff;
- normal saved data structurally validates before atomic publish;
- storage failure cannot become normal `episode_*`;
- abnormal-but-structurally-complete experiment data is not mislabeled as storage-incomplete.

## DISCARD

- discard does not unnecessarily serialize backlog;
- discard advances the local sequence cursor correctly;
- the next episode starts at the correct global sequence;
- normal discard leaves no manifest/dataset clutter.

## Complexity

- `retain_partial` is gone;
- shared finalization deadline/completed proof is gone;
- exactly-once final-result delivery machinery is gone;
- shared consumed-sequence proof is gone;
- recorder no longer owns normal experiment duration;
- result-sidecar invalidation is gone;
- no equivalent replacement framework is introduced.

## Resource robustness

- record ring default is 16;
- a simple high hard frame/resource guard remains;
- exceeding it fails recording rather than producing a normal truncated episode.

## Persisted-data scope

- raw schema remains v32;
- existing v32 persisted semantics are not silently reinterpreted;
- no historical-data migration is required by this task.

## Scope safety

No unrelated redesign/regression in:

- SafetyState;
- run_id stale-command fencing;
- latest-target RobotCommand semantics;
- arm/hand workers;
- camera/calibration;
- point cloud;
- online IK algorithm/tuning;
- tactile validity;
- action/action_ee semantics;
- raw -> Zarr whole-episode policy.

---

# 24. Final decision rule

For every mechanism kept or added, ask:

> What concrete bad experiment, lost row, corrupt file, unsafe continuation, or wrong episode boundary does this prevent?

If the answer is only:

> it proves another internal state is correct

remove it.

For every proposed deletion, also ask:

> Does this cheap mechanism currently protect a real hardware, storage, sequence, or historical-data invariant?

If yes, keep the invariant and simplify only its implementation.

The target is not the shortest recorder.

The target is the smallest reliable runtime recorder for repeated real-robot PhD experiments, without forcing a raw-data migration in the same change.
