# DexMani Real — Canonical Policy Rollout Cleanup Guide

> Repository: `dexmani_real`  
> Intended executor: Claude Code / Codex  
> Scope: personal PhD real-robot research. Keep one current runtime contract. Do not add compatibility layers, schema versions, or historical migration logic.

---

## 1. Goal

Make the learned-policy rollout path logically correct and easy to reason about:

```text
Timestamped sensors
        ↓
Causal Observation Builder
        ↓
Policy inference worker
        ↓
ActionChunk [N,D]
        ↓
timestamp-based action selection
        ↓
joint decode / EE→IK
        ↓
safety validation
        ↓
arm + hand command
        ↓
robot workers
```

This is a semantic cleanup, not a runtime rewrite.

---

## 2. Preserve existing infrastructure

Do not redesign:

```text
shared-memory rings
sensor workers
causal observation alignment
inference worker separation
Prediction ownership
run_generation
stale-prefix skipping
whole-stale discard
no catch-up behavior
coupled arm+hand publication
sync/async modes
executor polling
worker heartbeat
supervisor
home lifecycle
RecorderIO
SafetyGate
robot SDK workers
```

Do not add:

```text
Temporal Ensemble
RTC
steps_per_inference
new scheduler
chunk blending
adaptive horizon
latency compensation
new recorder framework
```

---

## 3. Remove obsolete Policy compatibility dependency

After `dexmani_policy` cleanup, Real should consume only the current public Policy API.

Remove all Real-side use of:

```text
temporal_ensemble_coeff
```

Do not add:

```text
artifact version parser
legacy config loader
v1/v2/v3 compatibility branches
migration framework
```

If old metadata exists, update or regenerate it once before use. Runtime code should support only the current contract.

---

## 4. Preserve timestamped ActionChunk semantics

Keep current execution semantics:

```text
future action
    execute at target time

past action
    skip

whole chunk stale
    discard
```

Do not:

```text
retime stale actions
catch-up execute old actions
blend old/new chunks
```

Keep:

```text
prediction ring
run_generation
executor polling
```

unless a concrete correctness bug requires change.

---

## 5. Timing cleanup

Separate three concepts.

### Observation validity

Responsible for:

```text
sensor age
sensor skew
causal alignment
```

Keep in observation construction.

### Command delivery validity

Responsible for:

```text
worker acceptance timeout
command watchdog
hardware delivery safety
```

Keep.

### Remove duplicated action invalidation

Audit:

```text
max_source_to_command_age_s
```

Remove it only if it duplicates observation freshness after Policy already produced a valid ActionChunk.

Do not remove:

```text
first command timeout
command silence timeout
command progress timeout
action apply timeout
```

---

## 6. IK and safety attribution

Keep failure categories minimal and correct.

### Joint policy

```text
joint action
    ↓
safety
        OK
        SAFETY_REJECT
```

### EE policy

```text
EE action
    ↓
IK
    ↓
    IK_FAIL

or

IK success
    ↓
safety
    ↓
    SAFETY_REJECT
```

Do not classify:

```text
joint limit
workspace rejection
large action jump
```

as IK failure.

Do not create a large exception hierarchy.

---

## 7. Evaluation recording must not change control semantics

Keep the episode lifecycle:

```text
B
↓
Recorder START
↓
RUNNING
↓
rollout
↓
outcome
↓
Recorder STOP/save
```

Recorder remains an observer, not a controller.

During RUNNING:

```text
Policy observation validity
```

may affect control.

But evaluation-only evidence should not block a valid action.

Example:

```text
state-only policy
camera recording problem
```

should become an evaluation/recording issue, not a policy action rejection.

Do not create a new recorder architecture. Modify the current executor/evaluation path minimally.

---

## 8. Episode storage

Separate:

```text
outcome
```

from:

```text
storage integrity
```

Normal outcomes:

```text
SUCCESS
FAILURE
INVALID
```

should be preserved when recorder data is healthy.

Unrecoverable recorder/storage corruption may fail saving, but must have an explicit reason.

Do not delete ordinary failed rollouts.

---

## 9. Do not redesign safety or lifecycle

Do not remove:

```text
physical_home_completed
home sequence
workspace checks
collision hooks
supervisor
heartbeat
```

unless a direct bug is found.

A cleanup task should not become a robot infrastructure rewrite.

---

## 10. Patch scope

Search first. Expected areas:

```text
dexmani_real/deployment/config.py
dexmani_real/deployment/executor.py
dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py
dexmani_real/deployment/evaluation.py
examples/run_policy.py
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

unless directly required.

---

## 11. Validation

Do not build a large test framework.

Use small deterministic offline checks if appropriate:

```text
partial stale chunk → skip prefix
whole stale chunk → discard
old run_generation ignored
joint rejection → SAFETY_REJECT
EE IK failure → IK_FAIL
EE IK success + safety failure → SAFETY_REJECT
evaluation-only evidence missing does not alter valid action scheduling
```

No hardware tests.

Run:

```bash
python -m compileall -q dexmani_real

git diff --check
```

---

## 12. Implementation order

### Phase A

Remove obsolete Policy compatibility checks.

### Phase B

Fix executor semantics:

```text
IK vs safety attribution
recording not gating action publication
```

### Phase C

Update durable docs.

Do not mix new research algorithms into this cleanup.

---

## 13. Definition of Done

The final runtime should satisfy:

```text
canonical timestamped ActionChunk execution
```

```text
normal deployment == evaluation control semantics
```

```text
IK_FAIL != SAFETY_REJECT
```

```text
no generic temporal blending mechanism
```

```text
one current runtime contract
no compatibility/version framework
```

---

## 14. Final report

Return:

### Changed
Exact files and semantic changes.

### Preserved
Confirm:

```text
sensor alignment
prediction ownership
timestamp scheduler
SafetyGate
RecorderIO
lifecycle
```

### Verified
Compile/check commands and offline checks.

### Not Verified
Hardware execution and real task success.

### Remaining Risk
Only concrete discovered risks. Do not propose RTC, temporal smoothing, or a new deployment framework.
