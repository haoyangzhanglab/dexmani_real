# DexMani Real — Policy Eval Correctness Repair Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Reviewed baseline: `0a03b57fdf5c9ea3ca0103a313313a50bdf82ba3`  
> Audience: Claude Code / Codex  
> Scope: personal PhD real-robot research. Fix confirmed correctness issues only. Do not turn deployment into a generic framework.

---

# 1. Scope

Current formal eval architecture is correct and should be preserved:

```text
PolicyObservation
    ↓
Inference worker
    ↓
Prediction
    ↓
PolicyExecutor
    ↓
sync/async scheduling
    ↓
decode / optional EE IK
    ↓
SafetyGate
    ↓
one coupled arm+hand command
    ↓
robot workers
```

Formal evaluation additionally owns:

```text
B
↓
Recorder START
↓
RECORDING ACK
↓
RUNNING
↓
initial evidence sample
↓
rollout
↓
SUCCESS / FAILURE / INVALID / timeout
↓
motion fence
↓
Recorder STOP
```

Do not redesign:

- async logical-grid scheduling;
- stale-prefix skipping;
- no catch-up behavior;
- coupled arm-hand command ownership;
- SafetyGate boundary;
- Recorder ownership;
- raw schema version.

---

# 2. Confirmed Bug A — retryable evaluation evidence is treated as fatal

`PolicyExecutor._build_evaluation_frame_inputs()` returns:

```python
(inputs, reason, fatal)
```

Semantics:

```text
fatal=False
    evidence temporarily unavailable
    retry on later executor tick

fatal=True
    malformed/unhealthy source
    fail closed
```

Current bug:

```python
evaluation_inputs, reason, _fatal = build_evaluation_frame_inputs(...)

if evaluation_inputs is None:
    self._fault(...)
```

The caller ignores the helper contract.

---

## Required Fix

Use the existing `fatal` decision:

```python
evaluation_inputs, reason, fatal = build_evaluation_frame_inputs(...)

if evaluation_inputs is None:
    if fatal:
        self._fault(...)
    return
```

Do not add:

```text
retry manager
evidence state machine
camera retry queue
scheduler changes
```

A missing source in one tick should simply produce no command for that tick. Existing async stale handling remains responsible for expired slots.

---

# 3. Confirmed Bug B — rejection metadata confuses IK failure and action admission failure

Current action path:

```text
policy action
    ↓
joint passthrough or EE→IK
    ↓
arm action admission
    ↓
SafetyGate/publication
```

A rejection can mean:

```text
A. true EE IK failure
B. joint limit / action admission failure
```

Current recording can incorrectly map both to:

```text
FRAME_IK_FAIL
```

Example bad state:

```text
ik_attempted = false
frame_status = FRAME_IK_FAIL
```

---

## Required Fix

Return a minimal rejection reason from action decode/admission:

```text
"ik"
"action_safety"
```

Required semantics:

### True EE IK failure

```text
ik_attempted = true
ik_ok = false
safety_reject = false
frame_status = FRAME_IK_FAIL
```

### Joint/bounds/action admission failure

```text
safety_reject = true
frame_status = FRAME_SAFETY_REJECT
```

For joint-space policies:

```text
ik_attempted must not become true
frame_status must not become FRAME_IK_FAIL
```

Do not modify raw schema.

---

# 4. Do not change camera contracts

Previous camera concerns were not confirmed bugs.

Do not modify:

- frame-gap telemetry semantics;
- RGB-D timestamp contract;
- camera schema.

Camera admission remains owned by the camera subsystem.

---

# 5. Do not redesign Recorder failure behavior

Current formal-eval fail-closed behavior is intentional:

```text
recording integrity failure
    ↓
motion stops
    ↓
operator investigates
```

Do not change this into silent continuation.

---

# 6. Recording latency is a deferred measurement, not this patch

Formal eval currently performs evidence collection before publication, including:

```text
causal state reads
RGB-D reads
FK
EpisodeState assembly
metadata generation
```

This may add latency, but there is currently no proof it harms control.

Do not move recording logic during this repair.

Future profiling should compare:

```text
run --async
vs
formal eval --async
```

using:

```text
schedule_lateness
publication interval
skipped_prefix_steps
stale_prediction_count
```

---

# 7. Required focused tests

Do not restore a large test suite.

Add only:

```text
tests/test_policy_executor_timing.py
tests/test_policy_evaluation.py
```

## test_policy_executor_timing.py

Cover:

```text
partial stale prediction
all stale prediction
no catch-up
sync regression
```

## test_policy_evaluation.py

Cover:

```text
retryable evidence:
    fatal=False
    no _fault()

fatal evidence:
    fatal=True
    _fault()

joint rejection:
    FRAME_SAFETY_REJECT

IK failure:
    FRAME_IK_FAIL

Recorder before RUNNING
Failure trial saved
Outcome written before stop
Finalizing blocks next episode
```

Use fakes only.

No hardware.

---

# 8. Preserve runtime invariants

The patch must not change:

```text
first_future_step_index
logical timestamp scheduling
prediction_ring ownership
latest-wins prediction
no catch-up
SafetyGate
IK ownership
coupled arm-hand ActionCandidate
worker final guards
command watchdog
e-stop behavior
```

---

# 9. Suggested patch scope

Expected files:

```text
dexmani_real/deployment/executor.py
(optional) dexmani_real/deployment/evaluation.py
repo_map.md

tests/test_policy_executor_timing.py
tests/test_policy_evaluation.py
```

Do not add:

```text
RTC
ACT temporal ensemble
latency scheduler
second recorder
schema migration
metrics dashboard
```

---

# 10. Validation

Run:

```bash
git status --short
git rev-parse HEAD

python -m compileall -q dexmani_real examples

python -m pytest -q \
 tests/test_policy_executor_timing.py \
 tests/test_policy_evaluation.py

git diff --check
git diff --stat
```

Do not run real robot commands without explicit authorization.

---

# 11. Definition of Done

Complete when:

```text
fatal=False evidence miss
→ retryable
→ no false FAULT
```

```text
true IK failure
→ FRAME_IK_FAIL
```

```text
joint/action admission failure
→ FRAME_SAFETY_REJECT
```

and:

- scheduler semantics unchanged;
- coupled command semantics unchanged;
- focused tests pass;
- docs match actual repository tree.

---

# 12. Final Codex report

Return only:

## Changed
Files and contracts changed.

## Verified
Compile/tests/diff checks.

## Not Verified
Explicitly list:

```text
xArm
XHand
RealSense
hardware timing
real task success rate
```

## Deferred
Only:

```text
Whether formal-eval recording overhead measurably affects hardware scheduling latency.
```
