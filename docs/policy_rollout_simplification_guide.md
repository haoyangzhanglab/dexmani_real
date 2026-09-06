# DexMani Real — Policy Rollout Semantic Repair Guide

> Repository: `dexmani_real`  
> Intended executor: Claude Code / Codex  
> Upstream Policy baseline reviewed: `haoyangzhanglab/dexmani_policy@c748776421fbb85575d052482c4b7ad6bb140dde`  
> Scope: personal PhD real-robot research. Fix rollout semantics with the smallest reliable patch. Do not turn this task into a scheduler, recorder, lifecycle, or data-format rewrite.
>
> **Current project constraint:** there is still no trained Policy checkpoint available for integration testing. This task must be implemented and verified with source-level/synthetic checks only. No policy training, checkpoint selection, real artifact export/restore, or physical robot execution belongs in this phase.

---

## 1. Upstream Policy contract is now fixed

The current `dexmani_policy.deployment` public boundary is:

```text
PolicySpec
├── action_key
├── action_dim
├── control_action_dim
├── horizon
├── n_obs_steps
├── n_action_steps
├── observation_fields
├── control_dt_s
├── requires_hand
└── rgb_preprocessing

LoadedPolicy.predict(obs)
→ finite float64 [n_action_steps, control_action_dim]
→ canonical validated control_action
```

Important facts:

```text
PolicySpec.temporal_ensemble_coeff no longer exists
LoadedPolicy.predict() does not blend chunks
Policy artifact format is Policy-owned and unversioned at the Real boundary
Real must not parse Policy artifact internals
```

Therefore Real-side Policy adaptation is now deterministic: remove the obsolete temporal guard and keep all other actually-used compatibility checks.

---

## 2. Goal

The learned-policy execution path must remain:

```text
Timestamped sensors
        ↓
Causal Observation Builder
        ↓
Policy inference worker
        ↓
Prediction / ActionChunk [N,D]
        ↓
timestamp-based action selection
        ↓
joint decode / EE→IK
        ↓
reject-only policy admission + SafetyGate
        ↓
coupled arm + hand command
        ↓
robot workers
```

Formal evaluation must use the same action-control semantics after its startup recording barrier:

```text
Recorder START / initial evidence
        ↓
normal rollout control path
        ↓
record rollout result as side evidence
```

The key principle is:

> Observation validity decides whether Policy may infer. Action timestamps decide whether a chunk action is still executable. Command TTL/watchdogs decide whether a prepared command can still cross the hardware boundary. Evaluation recording does not add a fourth action-validity rule.

---

## 3. Hard execution constraints

Do not run or initiate:

```text
policy training
resume/fine-tuning
DDP/NCCL
checkpoint selection sweep
long simulator evaluation
real Policy checkpoint restore
real Policy deployment export
GPU policy integration requiring trained weights
xArm/XHand physical commands
formal real-robot eval
```

Do not create/train a dummy neural policy to satisfy tests.

Use:

```text
fake PolicySpec
synthetic Prediction/ActionChunk
synthetic monotonic timestamps
fake RuntimeChannels / recorder / planner
NumPy arrays
deterministic CPU-only checks
compile/static checks
```

Any real-checkpoint/hardware integration is Deferred.

---

## 4. Preserve these mechanisms

Do not redesign or remove unless a direct regression proves necessary:

```text
shared-memory sensor rings
causal multimodal observation alignment
camera-source state alignment
inference worker separation
Prediction latest-wins ring
run_generation stale-result protection
sync + async modes
logical/timestamped action grid
first_future_step_index stale-prefix filtering
whole-stale discard
no catch-up
executor_poll_hz
coupled arm+hand publication
command progress watchdog
first-command / command-silence / action-apply timeouts
hardware feedback health checks
e-stop
physical_home_completed + home sequence
workspace segment check
collision hooks
SafetyGate
Recorder START acknowledgement
RecorderIO process/state machine
supervisor/heartbeat/process-death handling
robot SDK workers
IPC schemas
```

Do not add:

```text
Temporal Ensemble
ChunkOverlapBlender
RTC
steps_per_inference
chunk merge/blending
adaptive horizon
latency compensation
new scheduler
new recorder process/event bus
new compatibility framework
```

---

## 5. Important scope clarification: do not remove Real data schema versions

The Policy cleanup removed **Policy deployment/best-checkpoint compatibility versioning**.

It does **not** mean this Real task should remove legitimate data/calibration schema versions such as:

```text
EPISODE_SCHEMA_VERSION
PROCESSED_SCHEMA_VERSION
POLICY_ZARR_SCHEMA_VERSION
VR transform / calibration schema versions
```

Those protect stored dataset/calibration contracts and are outside this rollout cleanup.

In particular, current Real Policy Zarr is already `POLICY_ZARR_SCHEMA_VERSION = 7`, matching the current Policy Real-data boundary. Leave this alone.

---

## 6. Phase A — remove the obsolete Policy temporal guard

Current `validate_policy_runtime_compatibility()` still requires:

```text
policy_spec.temporal_ensemble_coeff
```

although current PolicySpec no longer contains that field.

Required change in `dexmani_real/deployment/config.py`:

Remove:

```text
_GENERIC_TEMPORAL_ENSEMBLE_ERROR
try/getattr of temporal_ensemble_coeff
non-null rejection
related error text
```

Preserve all other compatibility checks:

```text
observation field names/shapes/dtypes
point-cloud semantics
fingertip semantics
requires_hand
n_action_steps <= IPC capacity
action_key
control_action_dim
control_dt_s vs Real control_hz
```

Do not make Real parse `_format`, `contract`, `best_ckpt.json`, or any Policy artifact metadata.

---

## 7. Phase B — remove the hidden source-age action deadline

This is no longer an open audit item; remove it.

### Why

Observation Builder already checks source freshness before Policy inference:

```text
anchor_ns - latest_source_ns <= max_input_age_s
```

and enforces causal alignment/skew/grid-lag constraints.

After Policy produces a chunk, Executor currently adds another validity rule:

```text
prediction.source_monotonic_ns + max_source_to_command_age_s
```

which is used both to discard an active prediction and as `ActionCandidate.valid_until`.

That duplicates observation freshness and can silently truncate the future tail of an otherwise timestamp-valid chunk. The Policy contract does not know this extra horizon truncation.

### Required removal

Remove:

```text
PolicyParams.max_source_to_command_age_s
PolicyExecutor.max_source_age_ns
_prediction_source_deadline_ns()
_UINT64_MAX if then unused
_next_due_action() source-age invalidation
source_deadline_ns in _publish_due_action()
valid_until_monotonic_ns=source_deadline_ns
```

Build normal Policy candidates using the existing command delivery TTL only:

```python
build_action_candidate(
    ...,
    scheduled_target_monotonic_ns=scheduled_target_ns,
    action_validity_s=runtime.policy.action_validity_s,
)
```

### Keep responsibilities separate

```text
Observation freshness
→ max_input_age_s / skew / grid lag before inference

Action temporal validity
→ logical target timestamp + stale-prefix/whole-stale rules

Command delivery validity
→ action_validity_s + publication/worker final guards

Hardware liveness
→ existing progress/silence/apply watchdogs
```

Do not remove or merge the latter watchdogs.

---

## 8. Phase C — explicit IK vs SAFETY rejection semantics

Current code conflates some post-IK arm-admission failures with IK failure.

Use the smallest explicit internal classification. For example:

```python
class _RejectKind(Enum):
    IK = auto()
    SAFETY = auto()
```

or an equivalent tiny typed result.

Do not classify by parsing reason strings.

Required semantics:

### Joint policy

```text
joint action
→ arm canonicalization/admission
→ SafetyGate
→ OK or SAFETY_REJECT
```

Joint policy must never emit `IK_FAIL`.

### EE policy

```text
EE action
→ rot6d/target decode
→ IK fail
    → IK_FAIL

IK success
→ arm canonicalization/admission
→ SafetyGate
→ any failure here
    → SAFETY_REJECT
```

Examples that must be SAFETY_REJECT, not IK_FAIL:

```text
arm joint limit violation
arm per-step jump violation
workspace rejection
hand joint/delta rejection
other non-fatal SafetyGate rejection
```

Ordinary IK/safety step rejection must not become a global runtime FAULT.

---

## 9. Phase D — make learned-policy arm jump protection reject-only

Current arm behavior silently clips a learned action while hand behavior rejects a large jump. This creates asymmetric and hidden action semantics.

Current behavior:

```text
raw arm target
→ wrap_nearest_equivalent
→ joint limit admission
→ np.clip(delta, ±arm_action_delta_clip_rad)
→ execute modified target
```

Required behavior:

```text
raw arm target
→ wrap_nearest_equivalent
→ joint limit admission
→ abs(delta) <= threshold ?
      yes → use canonical target
      no  → SAFETY_REJECT
```

### Recommended local rename

Rename the learned-policy-only configuration:

```text
arm_action_delta_clip_rad
→ arm_max_action_jump_rad
```

and helper:

```text
_clip_policy_arm_action
→ _validate_policy_arm_action
```

or equivalent names reflecting reject-only behavior.

Keep `wrap_nearest_equivalent()`; that is representation canonicalization, not learned-action shaping.

Do not move this check into generic SafetyGate if doing so would spread changes into teleop/replay. A local learned-policy admission helper is sufficient.

### Metrics

Remove `arm_action_clip_count` semantics. Prefer the existing generic `safety_rejection_count`; add a narrow arm-jump rejection counter only if it materially helps debugging.

Do not preserve a metric named “clip” when no clip occurs.

---

## 10. Phase E — formal eval: keep startup barrier, remove per-action recording gate

The current problematic order is approximately:

```text
action due
→ build evaluation frame inputs
→ evidence unavailable
→ _fault()
→ action never executes
```

This changes policy control behavior only because formal recording is enabled.

### Keep the startup protocol

Retain:

```text
B request
→ physical start-pose checks
→ Recorder START
→ RECORDING ACK
→ RUNNING
→ initial causal evaluation sample
→ policy commands
```

The initial sample is the one allowed evaluation startup barrier. If it cannot be recorded before the first Policy command within the configured startup budget, end the trial INVALID without executing Policy commands.

Do not remove the initial-sample concept in this patch.

### After initial sample succeeds

Every normal control slot must follow control-first semantics:

#### Successful action

```text
action due
→ decode / IK
→ arm admission
→ SafetyGate / prepare_command
→ publish command
→ update command/control-slot state
→ attempt evaluation evidence + RecorderClient.add_frame
```

#### Rejected action

```text
action due
→ decode / IK / safety result
→ reject + consume the control slot
→ attempt to record rejection evidence
```

Evaluation evidence must not decide whether the already-valid current action is published/rejected.

### Important ordering rule

Once a command has been published, a later recording/evidence failure must never retroactively reclassify that command as “not executed”.

Ensure scheduler state (`next_command_due_ns`, previous accepted targets, episode step accounting, publication metrics) reflects the control outcome before any post-control evaluation termination path runs.

---

## 11. Recording/evaluation failure handling

Do not call `_fault()` merely because the evaluation side cannot produce/store one evidence row.

Introduce a small evaluation-invalid termination helper if useful, conceptually:

```text
mark EvaluationOutcome.INVALID
revoke/end current episode to ARMED
request Recorder STOP
save healthy prefix when storage is still trustworthy
log explicit eval invalid reason
```

Do not set global `error_state=True` solely for an evaluation metadata failure.

### Evidence unavailable but Recorder storage is healthy

Examples:

```text
causal evaluation camera row temporarily unavailable/stale
evaluation-only FK/state assembly failed after the control result
recording-only metadata unavailable
```

After the current control result is complete:

```text
trial → INVALID
stop rollout
save the valid recording prefix when possible
```

Do not retry indefinitely and do not continue a formally scored trial with a missing middle evidence row.

### Recorder integrity/storage failure

Examples:

```text
sample ring overflow
RecorderClient.add_frame() returns false because recording transaction failed
RecorderIO reports writer/finalization error
```

Then:

```text
trial → INVALID
stop rollout
recording may remain unsaved/failed
reason must be explicit
```

Do not pretend corrupted storage is a normal saved INVALID episode.

### `max_frames`

Recorder `max_frames` is infrastructure/capacity exhaustion, not task failure.

Change formal outcome semantics from:

```text
eval:failure:max_frames
```

to:

```text
eval:invalid:max_frames
```

or an equivalent INVALID reason.

---

## 12. Do not suppress independent lifecycle faults

This patch separates **executor-owned recording semantics** from robot-control health. It does not make recording workers invisible to the lifecycle.

If existing independent infrastructure detects:

```text
arm/hand worker death
camera/recorder process death
shared error_state
heartbeat timeout
e-stop
other supervisor-level failure
```

keep the existing lifecycle/supervisor FAULT behavior.

In particular, do not redesign `run_supervisor`, camera worker process-death policy, or global worker supervision just to make a recording-only source non-fatal.

The narrow guarantee for this patch is:

> A per-action evaluation evidence miss or RecorderIO business/status failure should not be converted into `_fault()` by PolicyExecutor when the underlying lifecycle has not independently faulted.

This keeps scope bounded and ownership clear.

---

## 13. Preserve home, workspace, collision, and data semantics

Do not remove:

```text
physical_home_completed
home sequence
arm measured start-pose recheck
workspace segment FK check
collision hooks
hand limits
hardware SDK final guards
```

The current home latch represents a complete home procedure; arm qpos alone does not replace it.

Do not redesign raw/processed HDF5 or Policy Zarr schemas in this task.

---

## 14. Expected patch scope

Search first; keep the final diff small.

Expected primary files:

```text
dexmani_real/deployment/config.py
dexmani_real/deployment/executor.py
dexmani_real/config/defaults.py
dexmani_real/deployment/metrics.py
```

Likely durable docs:

```text
docs/action_clip_mechanisms.md
README.md
repo_map.md
docs/policy_rollout_simplification_guide.md
```

Likely new lightweight check:

```text
tests/test_policy_rollout.py
```

Files that should normally not need production changes:

```text
dexmani_real/deployment/inference/observation.py
dexmani_real/deployment/inference/worker.py
dexmani_real/deployment/timing.py
dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py
dexmani_real/deployment/evaluation.py
dexmani_real/recording/client.py
dexmani_real/recording/io_worker.py
dexmani_real/runtime/safety.py
dexmani_real/runtime/supervisor.py
dexmani_real/deployment/lifecycle.py
dexmani_real/deployment/operator.py
sensor drivers
robot workers
IPC schema
```

Modify one of those only if a direct dependency is demonstrated; explain why in the final report.

---

## 15. Implementation order

Implement in this order so failures stay attributable:

### P0 — Policy boundary

```text
remove temporal_ensemble_coeff Real guard
```

### P1 — Timing semantics

```text
remove max_source_to_command_age_s and source-deadline path
preserve observation freshness + timestamp scheduling + command TTL/watchdogs
```

### P2 — Action semantics

```text
explicit IK vs SAFETY rejection kind
arm jump clip → reject-only
metrics/docs follow the new semantics
```

### P3 — Formal evaluation

```text
keep START ACK + initial sample barrier
remove per-action evaluation evidence from action admission
post-control evidence failure → INVALID episode, not executor _fault()
Recorder integrity failure → INVALID + possibly unsaved
max_frames → INVALID
```

### P4 — lightweight deterministic checks + durable docs

Do not mix additional dead-code cleanup or unrelated robot infrastructure changes into this patch.

---

## 16. Deterministic offline checks

Do not require a trained Policy or hardware.

Use fake/synthetic objects and standard-library `unittest` if practical.

Required coverage:

### Policy boundary

```text
current fake PolicySpec without temporal_ensemble_coeff
→ validate_policy_runtime_compatibility succeeds when all actual fields match
```

### Timestamp scheduling

```text
partial stale async chunk → skip prefix
whole stale async chunk → discard
old run_generation → ignore
no catch-up behavior preserved
```

Add a regression proving an action is **not** dropped solely because `source_monotonic_ns` is older than the removed `max_source_to_command_age_s`, provided its timestamp scheduling still makes it executable.

### Arm admission

```text
within arm jump threshold
→ canonical target unchanged (except nearest-equivalent wrapping)

beyond arm jump threshold
→ SAFETY_REJECT
→ no clipped command is produced
```

### Failure attribution

```text
joint-policy arm admission failure → SAFETY_REJECT
true EE IK failure → IK_FAIL
EE IK success + arm admission failure → SAFETY_REJECT
prepare_command / SafetyGate reject → SAFETY_REJECT
```

### Formal evaluation ordering

Use fake publication/evidence/recorder hooks to prove:

```text
valid control action
→ publication/control state happens first
→ evaluation evidence failure happens second
→ trial becomes INVALID
→ publication is not undone
```

For a rejected action, prove the control slot/rejection is committed before evaluation recording failure terminates the trial.

### Startup barrier

Verify formal eval still requires:

```text
Recorder START ACK
initial sample success
```

before the first Policy command is allowed.

### Recorder outcome

Verify:

```text
healthy SUCCESS/FAILURE/INVALID → save requested
max_frames → INVALID
recorder integrity failure → INVALID, save not falsely reported
```

No camera/arm/hand worker or real Policy model should be started.

---

## 17. Validation commands

Before editing:

```bash
git status --short
git rev-parse HEAD
rg -n "temporal_ensemble_coeff|max_source_to_command_age_s|arm_action_delta_clip_rad|arm_action_clip_count" \
  dexmani_real examples README.md repo_map.md docs
```

After editing:

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_policy_rollout.py'

git diff --check
git diff --stat

rg -n "temporal_ensemble_coeff|max_source_to_command_age_s|arm_action_delta_clip_rad|arm_action_clip_count" \
  dexmani_real examples README.md repo_map.md docs
```

If a different lightweight deterministic test filename is chosen, run and report the actual command.

Expected final search:

```text
no active temporal guard
no source-age action deadline
no learned-policy arm clipping semantic
no clip metric/name
```

Historical discussion inside this implementation guide may mention removed names.

Do not run training, real checkpoint restore/export, long sim eval, or hardware.

---

## 18. Definition of Done for this phase

All must hold without trained Policy weights or robot hardware:

```text
Real accepts the current PolicySpec public contract
→ no temporal_ensemble dependency
```

```text
Observation freshness
→ checked before inference
```

```text
Action validity
→ timestamp stale filtering only
```

```text
Command delivery validity
→ action_validity_s + existing hardware/watchdog boundaries
```

```text
arm learned-policy jump
→ execute canonical target or SAFETY_REJECT
→ never silently clip
```

```text
true IK failure
→ IK_FAIL

post-IK / joint / workspace / hand safety failure
→ SAFETY_REJECT
```

```text
formal eval after initial sample
→ same control action path as normal rollout
→ recording cannot gate a valid current action
```

```text
recording/evidence failure
→ current trial INVALID
→ not PolicyExecutor global FAULT by itself
```

and all of these remain intact:

```text
causal observation alignment
sync/async
run_generation
stale-prefix / whole-stale / no-catch-up
executor polling
SafetyGate
home lifecycle
workspace/collision checks
worker watchdogs
RecorderIO architecture
supervisor/lifecycle fault handling
Real raw/processed/calibration schema versions
```

---

## 19. Deferred integration validation

After the first real Policy checkpoint exists, run a separate integration phase:

```text
Policy checkpoint
→ Policy deployment export
→ Policy restore/inference
→ Real check
→ Real shadow
→ timing profiling
→ physical rollout
→ formal real eval
```

This is not part of the current cleanup and must not block completion.

---

## 20. Final Claude Code / Codex report

### Changed
List exact files and the four semantic repairs:

```text
Policy boundary
source-age deadline removal
IK/safety + arm reject-only semantics
formal-eval recording ordering/failure semantics
```

### Preserved
Explicitly confirm:

```text
causal observation alignment
sync/async
run_generation
timestamp scheduler
command TTL/watchdogs
SafetyGate
home/workspace/collision behavior
RecorderIO architecture
supervisor/lifecycle
Real data schema versions
```

### Verified
Report only commands actually run: compile, deterministic tests, diff check, stale-reference search.

### Explicitly Not Run
Must state:

```text
no Policy training
no checkpoint selection
no real checkpoint restore/export
no long simulator evaluation
no robot hardware
```

### Remaining risk
State concrete unverified risks only. In particular, until hardware profiling exists, report that post-control formal-eval evidence copying/recording overhead has not been measured on the real control machine.

Do not propose RTC, temporal smoothing, new scheduler, compatibility framework, or recorder rewrite as generic follow-up work.
