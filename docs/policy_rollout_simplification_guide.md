# DexMani Real — Policy Rollout Correctness Cleanup Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD real-robot research. Make the learned-policy rollout path logically correct and easier to reason about. Preserve working robot infrastructure; do not redesign the runtime into a general framework.

---

## 1. Goal

The target learned-policy execution path is:

```text
Timestamped sensors
        ↓
Causal Observation Builder
        ↓
Policy inference worker
        ↓
Prediction / ActionChunk
        ↓
Timestamp-based stale action filtering
        ↓
Joint decode or EE→IK
        ↓
Safety validation
        ↓
Coupled arm + hand command
        ↓
Robot workers
```

The goal is semantic cleanup, not a new scheduler.

Preserve:

```text
causal observation alignment
shared-memory rings
inference worker separation
prediction ownership
run_generation protection
timestamped action execution
stale-prefix skipping
no catch-up
coupled arm+hand publication
e-stop boundary
worker supervision
```

---

## 2. Do not redesign existing infrastructure

Do not rewrite:

```text
IPC implementation
sensor drivers
teleoperation
RecorderIO process architecture
worker supervision
heartbeat system
robot SDK workers
```

The current process separation and safety boundaries are useful. Simplify only the learned-policy semantics.

---

## 3. Remove obsolete Policy compatibility fields

Current Real deployment validates obsolete generic temporal semantics.

Remove:

```text
temporal_ensemble_coeff
Generic temporal blending assumptions
```

from:

```text
deployment/config.py
Policy compatibility checks
CLI/config plumbing where only this field is used
```

Do not add:

```text
Temporal Ensemble
ChunkOverlapBlender
RTC
previous-action conditioning
```

If a future experiment needs these, they belong to an explicit Policy algorithm variant, not generic Real runtime behavior.

---

## 4. Keep current timestamped ActionChunk execution

Do not replace the current scheduling design.

The executor should keep the standard semantics:

For:

```text
A=[a0,...,aN]
```

with:

```text
t_i=t_chunk+i*control_dt
```

execute:

```text
future action
    wait until target time

past action
    skip

whole chunk stale
    discard
```

Do not:

```text
retime stale actions
execute catch-up bursts
blend chunks
```

Current mechanisms such as prediction ring, run generation, and executor polling are implementation details; keep them unless a concrete bug requires change.

---

## 5. Timing cleanup: remove only duplicated action validity

Separate:

### Observation validity

Used before inference:

```text
sensor age
sensor skew
causal alignment
```

Keep.

### Command delivery validity

Used between executor and robot workers:

```text
command TTL / worker acceptance timeout
```

Keep.

### Remove duplicated prediction invalidation

Review:

```text
max_source_to_command_age_s
```

If it only re-checks the age of the observation after a Policy chunk has already been produced, remove that dependency from action execution.

Do not remove worker safety timeouts:

```text
first command timeout
command silence timeout
command progress timeout
action apply timeout
```

Those protect physical execution, not Policy semantics.

---

## 6. Safety semantics: make failures interpretable

The goal is not to build a planner. Keep simple validation.

Keep:

```text
joint limits
hand limits
per-step delta limits
EEF workspace limits
hardware feedback health
e-stop
```

Do not add:

```text
online trajectory optimization
adaptive safety correction
new collision planner
```

### Action modification

Do not silently change learned actions.

Current learned-policy arm spike clipping should be reviewed:

Preferred semantics:

```text
unsafe action
    ↓
SAFETY_REJECT
    ↓
record rejection
```

not:

```text
unsafe action
    ↓
modified action
    ↓
execute silently
```

If a future experiment studies command limiting, make it an explicit experiment variable and record both raw/executed actions.

---

## 7. Separate IK failure and safety rejection

Maintain clear attribution.

### Joint action policy

```text
joint action
    ↓
safety
        OK
        SAFETY_REJECT
```

### EE action policy

```text
EE action
    ↓
IK
        IK_FAIL
    ↓
safety
        SAFETY_REJECT
```

Do not label:

```text
joint limit
workspace rejection
large action jump
```

as IK failure.

Modify only the metadata path; do not change the raw data schema unless unavoidable.

---

## 8. Formal evaluation: decouple recording metadata from command validity

Current formal evaluation has valuable concepts:

```text
B
Recorder START
RUNNING
SUCCESS/FAILURE/INVALID
Recorder STOP
```

Keep them.

However, evaluation-only evidence collection must not decide whether a valid Policy command can be published.

Required behavior:

```text
Policy observation valid
        ↓
execute normal control path
        ↓
record rollout metadata/evidence when available
```

### Initial episode boundary

Keep the initial recording/start handshake. It is reasonable that recording must be ready before an episode begins.

### During RUNNING

Do not require every action publication to wait for:

```text
camera read
FK reconstruction
EpisodeState assembly
Recorder write
```

A missing recording-only source should become a recording/evaluation issue, not silently change Policy control semantics.

Do not create a new recorder architecture. Modify the existing executor/evaluation path minimally.

---

## 9. Recorder outcome semantics

Separate:

```text
episode outcome
```

from:

```text
storage integrity
```

Recommended:

```text
SUCCESS
FAILURE
INVALID
```

are saved normally when the rollout reached a valid recording state.

However, unrecoverable recorder/storage failures such as:

```text
sample ring overflow
writer corruption
cannot finalize artifact
```

may abort saving. They must be explicitly reported as recorder failures.

Do not silently discard ordinary failed/invalid trials.

---

## 10. Home/start lifecycle

Do not remove `physical_home_completed` blindly.

It currently represents that a complete home procedure occurred, not just a convenience flag.

If simplifying it:

1. First ensure B checks the complete measured start condition:

```text
arm state near home
hand state near home
feedback healthy
```

2. Then remove redundant historical authorization state.

Do not replace a known-safe check with arm-only qpos checking.

---

## 11. PolicySpec boundary

Real should consume only the public Policy API.

Current Real compatibility should continue validating the fields it genuinely uses:

```text
action_key
control_action_dim
n_obs_steps
n_action_steps
observation_fields
control timing
```

Do not perform a broad PolicySpec redesign in this task.

---

## 12. Tests / checks

The repository currently does not establish a pytest test suite. Do not create a large test framework only for this patch.

Prefer:

```text
small deterministic unittest/script checks
```

covering:

```text
partial stale chunk
whole stale chunk
old run_generation ignored
IK_FAIL attribution
SAFETY_REJECT attribution
recording-only evidence missing does not change command semantics
```

No hardware tests.

---

## 13. Expected files

Search first. Expected areas:

```text
dexmani_real/deployment/config.py
dexmani_real/deployment/executor.py

dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py

dexmani_real/deployment/evaluation.py

dexmani_real/examples/run_policy.py
README.md
repo_map.md
```

Do not modify:

```text
teleop
sensor drivers
robot workers
IPC implementation
```

unless required by a direct dependency.

---

## 14. Implementation order

### Phase A — Policy compatibility cleanup

1. Remove obsolete temporal field checks.
2. Verify Real loads the new Policy public contract.

### Phase B — executor semantics cleanup

1. Remove duplicated prediction source-age invalidation if it duplicates observation validity.
2. Fix IK vs safety attribution.
3. Replace hidden action modification with explicit rejection where applicable.

### Phase C — evaluation cleanup

1. Keep episode boundaries.
2. Stop evaluation-only evidence from gating every action.
3. Preserve normal rollout artifacts and explicit recorder failures.

### Phase D — docs/checks

Update durable docs only after behavior is verified.

---

## 15. Validation

Before editing:

```bash
git status --short
git rev-parse HEAD
```

After editing:

```bash
python -m compileall -q dexmani_real

# use repository's existing lightweight check style if present


git diff --check
git diff --stat
```

Do not run real robot commands without explicit authorization.

---

## 16. Definition of Done

Complete when:

```text
Policy rollout uses canonical timestamped action chunks
```

```text
Normal deployment and formal evaluation share control semantics
```

```text
IK failure != safety rejection
```

```text
Recording does not silently alter valid Policy commands
```

```text
No generic temporal blending exists in Real runtime
```

and:

- existing robot infrastructure remains intact;
- focused offline checks pass;
- no RTC/Temporal Ensemble/scheduler framework was added.

---

## 17. Final Claude Code / Codex report

Return:

### Changed
- files changed;
- semantic contracts modified.

### Preserved
Confirm:

```text
causal observations
prediction ownership
timestamp scheduling
robot safety boundary
worker architecture
```

### Verified
- compile result;
- focused checks;
- diff check.

### Not Verified
Explicitly list:

```text
real robot execution
hardware timing
task success rate
```

### Remaining risk
Only concrete discovered risks. Do not propose generic temporal smoothing or new deployment frameworks.
