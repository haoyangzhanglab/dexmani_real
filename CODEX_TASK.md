# CODEX_TASK — Minimal Policy Rollout Trace + Rerun Visualization

> Temporary implementation task for Codex. Follow `AGENTS.md` first. After the
> implementation is complete, independently reviewed, and verified, **delete this
> file in the same final change set** so the repository does not retain a one-off
> implementation plan.
>
> This task was authored against parent source commit
> `d7afc8f8a292708311460109942df7eb490f6817`. If the worktree has moved, current
> source/schema/configuration remain the source of truth. Re-check the relevant
> paths before editing and adapt only as needed to preserve the intent below.

## 1. Goal

Add a small offline debugging path for recorded learned-policy rollouts that answers,
without changing policy/control semantics:

```text
What did the Policy predict?
        ↓
Which chunk step did PolicyExecutor actually process?
        ↓
Was it published, IK-rejected, or safety-rejected?
        ↓
How did the physical robot state respond?
```

Implement this as two complementary evidence sources:

```text
raw rollout episode                    policy trace
physical evidence                      proposal/scheduling evidence
(data.h5/depth.h5/rgb.mp4)             (<episode>.policy_trace.npz)
        │                                      │
        └────────────────┬─────────────────────┘
                         ▼
              visualize_policy_rollout.py
```

The resulting viewer is an offline **prediction-to-execution debugger**, not a new
runtime telemetry system and not a replacement for the existing raw/processed
episode visualizers.

The design priority is:

```text
correct semantics > simple ownership > low runtime overhead > extra features
```

## 2. Required user-visible behavior

A recorded policy session should remain structurally simple:

```text
rollouts/<policy>/<task>/<experiment>/session_*/
  run_config.yaml
  episode_001/
    data.h5
    depth.h5
    rgb.mp4
  episode_001.policy_trace.npz
  episode_002/
    ...
  episode_002.policy_trace.npz
```

For each **successfully published raw episode**, save one sibling trace file:

```text
<episode-dir-name>.policy_trace.npz
```

Example:

```text
session_20260914_170000/episode_001/
session_20260914_170000/episode_001.policy_trace.npz
```

Add the offline entry point:

```bash
python examples/visualize_policy_rollout.py <published-rollout-episode>
python examples/visualize_policy_rollout.py <published-rollout-episode> --info
```

The script must automatically resolve the sibling trace from the episode directory;
do not require the normal user to pass a second path.

The viewer must clearly separate:

```text
Exact evidence:
  - saved Policy prediction chunks
  - prediction timing diagnostics
  - Executor terminal decisions recorded by the trace
  - raw recorded robot state / commands / RGB-D

Reconstructed visualization:
  - current production point-cloud reconstruction from the raw episode
  - FK-derived Cartesian visualization for joint-action predictions

Not provided by v1:
  - bit-exact PolicyObservation tensors/history used at inference time
  - attention maps / hidden states
  - online/live visualization
```

Never label reconstructed RGB/point-cloud state as the exact model input.

---

## 3. Hard constraints

### 3.1 Hardware safety

Do **not** run hardware-affecting programs while implementing or validating this
change. In particular, do not run:

```text
examples/run_policy.py
teleoperation
replay
homing
camera/device acquisition
calibration writes
robot SDK/device constructors that connect to hardware
```

Use source inspection and offline checks only. Do not claim hardware validation.

### 3.2 Do not change runtime authority

Keep these ownership boundaries unchanged:

```text
Inference worker   owns model execution and Prediction production.
PolicyExecutor     owns scheduling, stale-prefix skipping, decode/IK, safety,
                   and command publication.
RecorderIO         owns the transactional raw episode.
Viewer             owns offline correlation/rendering only.
```

Do not move scheduling logic into the trace or viewer.

### 3.3 Do not change IPC or raw schemas

Do **not** modify:

```text
Prediction dataclass fields
PREDICTION_DTYPE
RuntimeChannels / shared-memory topology
raw episode DATASET_SPECS / schema version
processed episode schema / version
SafetyGate semantics
command publication semantics
dexmani_policy public contract
```

The raw episode schema intentionally rejects unexpected datasets. Do not put policy
trace datasets inside `data.h5`.

### 3.4 Trace is diagnostics, never execution authority

Trace collection and trace persistence are **fail-open diagnostics**:

- a trace collection failure must not reject a policy action;
- a trace save failure must not fault the robot/session;
- a trace failure must not set `error_state`, request e-stop, revoke motion, or
  change RecorderIO save/discard behavior;
- log a warning and disable/discard that episode's trace if necessary.

Raw recording keeps its existing fail-closed behavior. Do not weaken it.

### 3.5 No disk I/O on the RUNNING control path

During a motion epoch the trace is process-local memory only. Do not write `.npz`,
HDF5, JSON, logs-as-data, or other files per prediction/action.

Persist the trace only after RecorderIO reports the episode transaction finished.

---

## 4. Current source facts to preserve

Re-check these before editing; they describe the current intended mechanism.

### Prediction transport

`inference/worker.py` publishes one validated `Prediction` to the latest-wins
`prediction_ring`. `SharedMemoryRingBuffer.read_latest()` already returns:

```text
(data copy, ring commit monotonic timestamp, logical sequence)
```

The logical sequence is already the correct transport identity. Do not invent a
second `prediction_id` counter.

`read_latest_prediction()` in `deployment/executor.py` currently drops the ring
commit timestamp and returns only `(Prediction, sequence)`. For this task it should
preserve the timestamp as well, without changing the IPC record.

### Prediction scheduling

`PolicyExecutor` is the scheduling owner. It uses:

```text
prediction.logical_step_monotonic_ns
step_dt_ns
first_future_step_index(...)
```

to skip stale prefix actions and process future actions. Skips can also occur later
while advancing an active prediction. There is no single canonical per-action
"stale terminal event" in the current scheduler.

Therefore **do not fabricate `STALE_DROP` decision rows** in v1.

### Raw rollout recording

For a terminal control decision, the executor already records a raw row with the
same terminal `now_ns` used for that outcome:

```text
frame_status = 0  published / OK
frame_status = 2  IK reject
frame_status = 3  safety reject
```

Idle/held raw rows use status `1`; they are not policy terminal decision rows.

The raw episode timestamp is monotonic seconds (`now_ns / 1e9`), which allows exact
offline alignment with trace monotonic-ns fields.

### Recorder lifecycle

RecorderIO publishes the raw episode transaction atomically. `RecorderClient`
returns a final `RecorderStopResult` containing `saved` and the final episode
`path`. Use that returned final path to name the trace sidecar; do not guess paths.

---

## 5. Expected edit surface

Expected production/user-facing files:

```text
NEW     dexmani_real/deployment/policy_trace.py
MODIFY  dexmani_real/deployment/executor.py
NEW     examples/visualize_policy_rollout.py
MODIFY  README.md
```

`CODEX_TASK.md` itself must be deleted after implementation/verification.

A tiny change to `examples/visualize_episode.py` is acceptable **only if strictly
necessary to reuse existing raw Rerun rendering cleanly**. Prefer reuse/subclassing
or a very small shared helper over a generic visualization framework. Do not perform
unrelated cleanup.

No changes are expected in:

```text
examples/run_policy.py
inference/worker.py
ipc/schema.py
ipc/channels.py
recording/storage/schema.py
dataset/processed.py
control/safety_gate.py
```

If current source proves a listed file must change for the same contract, keep the
change minimal and explain it in the final handoff.

---

# 6. Task A — Add `policy_trace.py`

Create a small pure/offline-friendly module:

```text
dexmani_real/deployment/policy_trace.py
```

It should own only:

1. the in-memory trace buffer for one motion epoch;
2. strict conversion to/from the v1 NPZ payload;
3. atomic sibling-file persistence;
4. structural/semantic validation for viewer admission.

It must not import hardware SDKs, Torch, `dexmani_policy`, Rerun, or shared-memory
runtime objects.

## 6.1 Trace version and exact NPZ keys

Use a small explicit version constant, initially `1`.

The v1 NPZ must contain exactly these arrays/scalars:

```text
trace_version
run_started_monotonic_ns
chunk_size
action_dim

prediction_sequence
prediction_publish_monotonic_ns
prediction_ingest_monotonic_ns
prediction_source_monotonic_ns
prediction_logical_step_monotonic_ns
prediction_first_index
prediction_inference_latency_ms
prediction_observation_age_ms
prediction_observation_skew_ms
actions

decision_prediction_sequence
decision_chunk_index
decision_due_monotonic_ns
decision_event_monotonic_ns
decision_frame_status
```

Recommended exact dtypes/shapes:

```text
trace_version                         int32   scalar
run_started_monotonic_ns              uint64  scalar
chunk_size                            int32   scalar
action_dim                            int32   scalar

prediction_sequence                   uint64  [P]
prediction_publish_monotonic_ns       uint64  [P]
prediction_ingest_monotonic_ns        uint64  [P]
prediction_source_monotonic_ns        uint64  [P]
prediction_logical_step_monotonic_ns  uint64  [P]
prediction_first_index                int32   [P]
prediction_inference_latency_ms       float64 [P]
prediction_observation_age_ms         float64 [P]
prediction_observation_skew_ms        float64 [P]
actions                               float64 [P, H, A]

decision_prediction_sequence          uint64  [E]
decision_chunk_index                  int32   [E]
decision_due_monotonic_ns             uint64  [E]
decision_event_monotonic_ns           uint64  [E]
decision_frame_status                 uint8   [E]
```

where:

```text
H = PolicySpec.chunk_size
A = PolicySpec.control_action_dim (currently 19 or 21)
P = number of new same-generation predictions observed by the Executor
E = number of traced terminal Executor decisions
```

For an episode with zero predictions, `actions` must still have a well-defined empty
shape `(0, H, A)`. Initialize the trace with `chunk_size` and `action_dim`; do not
infer them only at save time.

Do not use object arrays or pickle. The loader must use `allow_pickle=False`.

## 6.2 Prediction row semantics

One prediction row represents one **new same-generation prediction sequence first
observed by `PolicyExecutor`**.

Fields mean:

```text
prediction_sequence
  prediction_ring logical sequence; canonical trace identity.

prediction_publish_monotonic_ns
  commit timestamp returned by prediction_ring.read_latest(); not Executor read
  time and not model-start time.

prediction_ingest_monotonic_ns
  the `now_ns` used by PolicyExecutor when it evaluates
  first_future_step_index() for this newly observed prediction.

prediction_source_monotonic_ns
prediction_logical_step_monotonic_ns
prediction_inference_latency_ms
prediction_observation_age_ms
prediction_observation_skew_ms
actions
  copied directly from the validated Prediction semantics already admitted at the
  existing inference/IPC boundaries.

prediction_first_index
  0..H-1 : first action still strictly future at initial Executor ingest
  -1     : the whole chunk was already stale at initial Executor ingest
```

Record wholly stale newly observed chunks too. They explain latency/discard behavior.

Do not create one prediction trace row for repeated reads of the same ring sequence.
Do not record old-generation predictions that the Executor ignores.

## 6.3 Decision row semantics

A decision row represents only an action that reaches one of the existing terminal
rollout outcomes:

```text
0 = _RECORD_FRAME_OK            (published)
2 = _RECORD_FRAME_IK_FAIL       (IK rejection)
3 = _RECORD_FRAME_SAFETY_REJECT (safety rejection)
```

Do not add a new enum if the existing status vocabulary can be reused safely.

Do not add decision rows for:

```text
idle/held grid samples
initial stale prefix skips
later scheduler skips
prediction replacement by a newer chunk
transient unavailable feedback paths that do not commit a terminal step
expired/stale paths that currently only advance/drop scheduler state
```

The absence of a decision row is not itself assigned a new semantic meaning.

`decision_prediction_sequence + decision_chunk_index` identifies the raw Policy
action from `actions`. Do **not** redundantly store `raw_action` in the decision
table.

Published arm/hand commands already live in the raw episode. Do **not** redundantly
store published joint targets in the trace.

`decision_due_monotonic_ns` is the due time used by the Executor control slot.
`decision_event_monotonic_ns` is the actual terminal publication/rejection time used
for the corresponding raw rollout row.

The scheduled logical target is intentionally not persisted because it is exactly
reconstructible as:

```python
scheduled_target_ns = (
    prediction_logical_step_monotonic_ns
    + decision_chunk_index * step_dt_ns
)
```

The viewer must compute `step_dt_ns` from the raw episode's canonical `control_hz`
with the same rounding rule used by the Executor.

## 6.4 In-memory validation

Keep validation focused and cheap. At minimum enforce:

```text
trace started before rows are added
positive run-start / timestamp fields where required
chunk_size > 0, action_dim > 0
Prediction.actions shape == (H, A)
Prediction.actions finite
prediction sequence strictly increases within one trace
publish_ns <= ingest_ns
first_index is -1 or within [0, H)
inference/age/skew metrics finite and non-negative
decision sequence refers to a recorded prediction row
decision chunk index is within [0, H)
decision due/event timestamps are positive and due_ns <= event_ns
decision status is one of {0, 2, 3}
no duplicate terminal decision for the same (prediction_sequence, chunk_index)
```

The loader should validate the same cross-array shape/key/dtype invariants before a
viewer accepts the trace.

## 6.5 Atomic save

Use `np.savez_compressed`, but do not expose a partially written final sidecar.
Write to a temporary file in the same directory and then `os.replace()` it into the
final path after the file has closed successfully. Clean up the temp file on failure.

Do not overwrite an already existing final trace sidecar silently; treat that as a
trace-save error and let the Executor's fail-open wrapper log it.

No trace file should be written before the raw episode has been successfully
published.

---

# 7. Task B — Integrate trace into `PolicyExecutor`

Keep the Executor as the single owner of scheduling semantics. Add only passive
trace hooks.

## 7.1 Preserve prediction ring commit time

Change the local helper contract from conceptually:

```python
read_latest_prediction(...) -> (Prediction, sequence) | None
```

to:

```python
read_latest_prediction(...) -> (Prediction, publish_monotonic_ns, sequence) | None
```

using the timestamp already returned by `prediction_ring.read_latest()`.

Do not change `Prediction`, `PREDICTION_DTYPE`, or the inference worker.

## 7.2 Track active prediction sequence separately

Add process-local Executor state equivalent to:

```python
self.active_prediction_sequence: int | None
```

Do not insert the ring sequence into the `Prediction` dataclass; transport identity
belongs to the Real ring boundary, not the inference result contract.

Whenever `active_prediction` is cleared/replaced, keep
`active_prediction_sequence` coherent. A tiny local helper to clear both is
acceptable if it reduces drift, but do not refactor the scheduler architecture.

## 7.3 Start trace only after the motion epoch exists

The trace corresponds to a real RUNNING episode, not merely a Recorder START
reservation.

Lifecycle must be:

```text
Recorder START acknowledged
        ↓
begin_requested_motion() succeeds
        ↓
run_started_ns / run_generation established
        ↓
start in-memory PolicyTrace for that motion epoch
        ↓
RUNNING
```

If Recorder START succeeds but the motion epoch is cancelled before RUNNING, no
trace should be started/published.

Only recorded rollouts (`recording_config` / `RecorderClient` present) need the trace
in v1. Do not create trace artifacts for non-recorded deployment mode.

## 7.4 Record predictions at initial ingest

In `_ingest_latest_prediction()`:

1. read `(prediction, publish_ns, sequence)`;
2. ignore repeated sequence exactly as today;
3. ignore wrong generation exactly as today;
4. compute `first_index` using the existing scheduler;
5. append one prediction trace row **before returning on fully stale chunks**;
6. use `prediction_first_index = -1` when `first_future_step_index()` returns None;
7. preserve all existing scheduler/statistics behavior.

Trace collection must not modify deadlines, cadence, active prediction choice, or
stale counters.

## 7.5 Critical invariant: capture sequence/index before advancing

Several current terminal paths call `_commit_terminal_step()` /
`_advance_prediction()` before recording finishes, which can mutate `self.step_index`
and clear/replace active state.

At the very beginning of processing one due action, before any function that may
advance scheduler state, capture local immutable values:

```text
prediction_sequence = current active_prediction_sequence
chunk_index          = current self.step_index
```

Use those captured local values for the terminal decision trace.

**Never read `self.step_index` after `_reject_due_step()`, `_commit_terminal_step()`,
or `_advance_prediction()` to identify the action that just terminated.** That would
attribute the decision to the next chunk step.

This invariant is required for published, IK-rejected, and safety-rejected paths.

## 7.6 Record only existing terminal decision outcomes

Add one decision row for:

```text
successful publication    status 0
IK rejection              status 2
safety rejection          status 3
```

Use exactly the terminal timestamp already used for the corresponding raw rollout
record:

```text
publication_ns for successful publication
terminal_ns from _reject_due_step() for rejection
```

Use the same `due_ns` already owned by the Executor.

Do not create synthetic stale/replacement events solely for the trace.

## 7.7 Fail-open trace wrappers

Do not let `PolicyTrace` exceptions propagate through the RUNNING scheduler.

Use a small Executor-owned fail-open boundary (one or two narrow helper methods are
fine) so that an unexpected trace error:

```text
logs warning
marks/disables the current trace
continues existing policy execution/recording behavior unchanged
```

Do not spam a warning every control tick after one trace failure; disable that
trace for the remainder of the episode.

## 7.8 Finalize trace only after RecorderIO finishes

`_finish_episode()` currently requests Recorder stop and may clear execution state
before the transaction completes. The trace buffer must survive that pending-stop
period; do **not** reset it from generic `_clear_execution()`.

When `_complete_recording(result)` receives the terminal Recorder result:

```text
result.saved == True and no recorder error
    → derive final sidecar path from result.path
    → atomically save the buffered trace
    → trace-save failure: warning only
    → clear trace state for next episode

result.saved == False or result.error is not None
    → do not publish a trace sidecar
    → discard/clear buffered trace
```

Expected final trace path:

```python
episode_path = Path(result.path)
trace_path = episode_path.parent / f"{episode_path.name}.policy_trace.npz"
```

If `saved=True` but `result.path` is unexpectedly absent/invalid, warn and discard
the diagnostics trace; do not alter the already completed raw transaction.

The same completion path must work for normal operator stop, timeout, saved fault or
e-stop episodes, max-frames finalization, and lifecycle shutdown whenever RecorderIO
ultimately publishes the raw episode.

---

# 8. Task C — Add `visualize_policy_rollout.py`

Create:

```text
examples/visualize_policy_rollout.py
```

It is **offline only**. It must not import/load a Policy model or connect to hardware.

## 8.1 CLI

Required interface:

```bash
python examples/visualize_policy_rollout.py EPISODE
python examples/visualize_policy_rollout.py EPISODE --info
python examples/visualize_policy_rollout.py EPISODE --max-frames N
```

Support the same practical point-cloud controls as `visualize_episode.py` when
possible:

```text
--point-cloud / --no-point-cloud
--pointcloud-num-points N
```

Resolve trace path automatically:

```python
episode = Path(args.episode).resolve()
trace = episode.parent / f"{episode.name}.policy_trace.npz"
```

Fail clearly if either the published raw episode or sibling trace is missing or
invalid.

## 8.2 Reuse current raw visualization

Do not copy the entire raw visualizer.

Prefer a small reuse strategy around the existing `EpisodeVisualizer` and its
production point-cloud/FK rendering. A lightweight subclass is acceptable. If
Rerun blueprint construction makes that impractical, extract only genuinely shared
stateless visualization helpers; do not build a generic base-viewer framework.

The rollout viewer should retain useful raw views:

```text
RGB
Depth
current-production reconstructed point cloud
actual EEF
actual fingertips
raw state/action/flags as appropriate
```

and add policy-specific views.

## 8.3 Time semantics

Use the existing Rerun `time` timeline in monotonic seconds:

```python
time_seconds = monotonic_ns / 1e9
```

Raw rollout `timestamp` already uses the same clock domain.

Keep raw `step` as a secondary sequence timeline for raw rows, but policy trace events
must be meaningful on the absolute `time` timeline. A simple implementation is to
log trace events before logging raw step-indexed rows so no stale `step` timeline is
accidentally attached to trace events.

When `--max-frames` truncates raw display, log only trace events whose event/ingest
time falls within the displayed raw timestamp interval. `--info` should still report
full trace counts.

## 8.4 Policy 3D proposal visualization

Render each newly ingested prediction as a future Cartesian EEF proposal trajectory.
The trace stores the raw Policy action chunk; interpret it exactly according to the
current Real action contract:

### Joint action (`A == 19`)

```text
actions[:, :7] = arm joint targets
        ↓ canonical Arm FK used by the repository
future EEF XYZ proposal trajectory
```

Use the existing FK implementation (`compute_eef_pose_history_xarm_base` or the
current canonical equivalent). This is a visualization derivation from the exact
Policy output; do not mutate the saved action.

### EE action (`A == 21`)

```text
actions[:, :3] = EEF XYZ in xArm base
```

Use those positions directly.

A `Points3D` trajectory is sufficient and uses an already-supported Rerun primitive;
do not add fragile visualization dependencies merely to draw connecting lines.

At each terminal decision, render/highlight the selected proposal target obtained by:

```text
prediction_sequence → prediction row
chunk_index          → action within that row
```

For rejection rows this is the rejected Policy target, not a command claimed to have
been published.

## 8.5 Policy timing views

At each prediction ingest time, log at least:

```text
inference_latency_ms
observation_age_ms
observation_skew_ms
first_index
ring_to_executor_ms = (ingest_ns - publish_ns) / 1e6
```

At each terminal decision event time, log at least:

```text
chunk_index
decision_frame_status
schedule_lateness_ms = (event_ns - due_ns) / 1e6
```

Do not create dozens of per-joint policy plots by default. Existing raw state/action
series plus 3D proposal visualization and these timing traces are enough for v1.

## 8.6 Raw command/response comparison

Use the raw episode as the sole source of physical response and submitted commands.
Do not duplicate them into `policy_trace.npz`.

The viewer may derive concise tracking diagnostics from raw data, e.g. norms of:

```text
actual arm_qpos - recorded arm command
actual hand_qpos - recorded hand command
```

only if this stays small and unambiguous. Do not turn v1 into a metrics framework.

## 8.7 `--info`

`--info` should validate both artifacts and print a concise summary without opening
Rerun. Include at least:

```text
episode path
trace path
raw frame count / control_hz
prediction count P
decision count E
actions shape (P, H, A)
number of fully stale-at-ingest predictions (first_index == -1)
decision counts for status 0 / 2 / 3
```

Optionally include concise mean/max timing diagnostics if already trivial from the
loaded arrays; do not add a reporting subsystem.

---

# 9. Task D — Minimal README update

Update the existing learned-policy rollout / visualization documentation just enough
to expose the supported workflow.

Add a command equivalent to:

```bash
python examples/visualize_policy_rollout.py \
  rollouts/<policy>/<task>/<experiment>/session_*/episode_001 --info
```

and state concisely that recorded rollout sessions include a sibling
`episode_XXX.policy_trace.npz` used to correlate exact Policy predictions/Executor
decisions with the raw physical episode.

If documenting the output tree, show the sidecar as a sibling of `episode_XXX/`, not
inside the raw episode directory.

Do not put the detailed NPZ schema, status implementation, or this task plan into
README; those volatile details belong in source.

---

# 10. Explicit non-goals

Do not add any of the following in this task:

```text
exact PolicyObservation snapshots
RGB/point-cloud/tactile inference-history duplication
attention/activation logging
online Rerun streaming
new telemetry process/thread/queue/shared-memory ring
new IPC dtype fields
new raw/processed schema fields
new scheduler abstraction
new stale-event taxonomy
SQLite/database storage
HDF5 trace writer
GPU/CUDA/environment snapshots
Git SHA or checkpoint hashing
outcome/success evaluator
policy comparison dashboard
large visualization framework/refactor
```

Do not "clean up" unrelated viewer, recorder, scheduler, or configuration code while
you are nearby.

---

# 11. Focused offline verification

Before running any Python entry point, inspect it for hardware side effects as
required by `AGENTS.md`. Do not run hardware-affecting examples.

## 11.1 Trace round-trip check

Using a temporary directory and synthetic NumPy data only, verify:

1. a started trace accepts multiple prediction rows and decision rows;
2. one prediction can have `first_index = -1`;
3. actions persist exactly with shape `(P, H, A)` and float64 values;
4. decision sequence/index resolves to the expected action;
5. `save()` + strict loader round-trip all fields;
6. `np.load(..., allow_pickle=False)` succeeds;
7. no temporary file remains after successful atomic save;
8. an empty-prediction trace can serialize with shape `(0, H, A)` if such save is
   supported by the final lifecycle helper;
9. invalid status, duplicate decision identity, bad shape, or invalid timestamp is
   rejected by the trace boundary.

Do not instantiate hardware channels for this check.

## 11.2 Scheduler identity check

Perform a focused pure/offline check of the current timing function showing that:

```text
logical_start + k * step_dt
```

maps to the expected `first_future_step_index()` around boundary timestamps. Confirm
that the trace stores the initial returned index and does not invent per-action stale
rows.

## 11.3 Executor source review

Manually inspect the changed terminal paths and verify the key invariant:

```text
captured prediction_sequence/chunk_index
    happen before any _commit_terminal_step() / _advance_prediction()
```

Confirm published, IK-reject, and safety-reject rows all use the correct captured
identity and terminal timestamp.

Confirm trace failures cannot propagate into command publication or RecorderIO
failure state.

## 11.4 Viewer checks

After confirming the script is offline-only, run at least:

```bash
python examples/visualize_policy_rollout.py --help
```

If a real already-recorded local rollout artifact is available, `--info` may be run
offline. Do not require one for acceptance and do not launch a hardware policy run to
create test data.

A small temporary synthetic trace can be used to exercise trace loading/helpers, but
do not invent a second fake raw-episode framework solely for this task.

## 11.5 Repository checks

Run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Inspect the focused diff before handoff.

If the repository already has a focused existing offline test location for these
symbols, use it. Do not introduce or restore a generic test framework solely to
satisfy this task.

---

# 12. Independent review before completion

If Codex sub-agents are available, use the repository roles rather than inventing new
permanent prompts:

1. optionally use `luna-max` before editing to re-map the smallest current
   producer → scheduler → recorder → viewer path;
2. after implementation and initial checks, ask `sol-high` for an **independent
   no-edit review** of the actual diff and affected current source, emphasizing:

```text
scheduler identity correctness
trace lifecycle across RecorderIO pending stop
fail-open diagnostics isolation
no IPC/raw-schema drift
no duplicate/misleading persisted semantics
Rerun time alignment
hardware-safety regressions
```

Do not let the reviewer edit files. Address concrete findings yourself, then repeat
relevant offline checks.

If sub-agents are unavailable, perform the same review manually and state that in the
handoff.

---

# 13. Acceptance criteria

All of the following must be true before cleanup:

- [ ] Existing raw episode structure/schema/version are unchanged.
- [ ] Existing processed schema/version are unchanged.
- [ ] `Prediction`, `PREDICTION_DTYPE`, and inference worker publication contract are
      unchanged.
- [ ] `read_latest_prediction()` preserves prediction-ring commit timestamp and
      logical sequence for Executor-local use.
- [ ] The trace uses prediction-ring logical sequence as its prediction identity;
      no redundant custom prediction counter is introduced.
- [ ] Trace starts only after a real motion epoch successfully begins.
- [ ] New same-generation predictions are traced exactly once at initial Executor
      ingest, including wholly stale chunks (`first_index = -1`).
- [ ] No synthetic per-action `STALE_DROP` event taxonomy is introduced.
- [ ] Active prediction sequence remains coherent whenever active prediction state is
      replaced or cleared.
- [ ] Terminal decision identity captures `(prediction_sequence, chunk_index)` before
      scheduler advancement.
- [ ] Published decisions use raw status 0; IK rejects use 2; safety rejects use 3.
- [ ] Decision terminal timestamps match the existing raw rollout terminal event
      timestamps.
- [ ] Trace does not duplicate raw Policy action per decision or raw published joint
      commands.
- [ ] No trace disk I/O occurs on the RUNNING control path.
- [ ] Trace exceptions cannot affect safety state, command publication, raw recording,
      or lifecycle decisions.
- [ ] A trace sidecar is published only after a raw episode is successfully published.
- [ ] Discarded/failed raw episodes do not publish a trace sidecar.
- [ ] Trace persistence is atomic enough that the final `.npz` is never intentionally
      exposed half-written.
- [ ] Viewer loads trace with `allow_pickle=False` and validates v1 keys/dtypes/shapes.
- [ ] Viewer automatically finds the sibling trace from the episode directory.
- [ ] Viewer aligns trace and raw evidence on the same monotonic `time` timeline.
- [ ] Viewer renders exact Policy proposal chunks and exact terminal selected chunk
      indices.
- [ ] Joint-action Cartesian proposal visualization uses canonical repository FK;
      EE-action visualization uses saved XYZ directly.
- [ ] Viewer clearly distinguishes reconstructed scene/point cloud from exact model
      input and does not claim bit-exact PolicyObservation support.
- [ ] `--info` works without opening Rerun and summarizes raw + trace structure.
- [ ] README exposes the concise rollout visualization workflow and sibling sidecar.
- [ ] Focused offline verification passes.
- [ ] `python -m compileall -q dexmani_real examples` passes.
- [ ] `git diff --check` passes.
- [ ] Final diff contains no unrelated cleanup or architecture expansion.
- [ ] No hardware-affecting validation was performed.

---

# 14. Final cleanup

After implementation, independent review, fixes, and final offline verification:

1. delete this `CODEX_TASK.md` file;
2. run `git diff --check` and `git status --short` again;
3. inspect the final diff to confirm the temporary task is gone and no unrelated files
   remain changed;
4. do not leave generated `.npz`, temporary viewer output, caches, or review artifacts
   in the repository.

The finished repository should contain the implementation and concise permanent
README documentation only, not this task plan.

---

# 15. Expected final handoff from Codex

Report concisely:

1. **Files changed** and the exact contract/behavior added in each.
2. **Trace semantics**: identity, lifecycle, persisted fields, and fail-open behavior.
3. **Viewer behavior**: what is exact vs reconstructed and how to invoke it.
4. **Offline checks run** and their results.
5. **Independent review findings** and how any findings were resolved (or note that
   sub-agents were unavailable and manual review was used).
6. **Anything not verified**, especially hardware/Rerun rendering with a real rollout
   if no suitable local artifact existed.
7. Explicit confirmation that **no hardware-affecting program was executed**.
8. Explicit confirmation that **`CODEX_TASK.md` was deleted as final cleanup**.
