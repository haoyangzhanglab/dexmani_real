# DexMani Real — Policy Rollout Simplification Guide

> Repository: `haoyangzhanglab/dexmani_real`
> Intended executor: Claude Code / Codex
> Scope: personal PhD real-robot research.

## 1. Goal

Simplify the learned-policy deployment path while preserving the mechanisms that are directly required for correct robot-learning experiments:

```text
timestamped observation
        ↓
Policy inference
        ↓
ActionChunk [N, D]
        ↓
stale-action filtering
        ↓
joint / EE decode
        ↓
IK when required
        ↓
simple safety validation
        ↓
coupled arm + hand command
        ↓
robot
```

Do not turn this repository into a general robotics serving framework.

The implementation must optimize for:

- correct experiment semantics;
- identical deployment and evaluation control paths;
- simple debugging;
- minimal hidden runtime behavior.

---

## 2. Preserve these invariants

Do not redesign:

```text
causal observation alignment
shared-memory sensor rings
inference worker separation
Prediction / ActionChunk ownership
run_generation stale-result protection
logical timestamped action execution
stale-prefix skipping
no catch-up behavior
arm+hand coupled command publication
hardware e-stop boundary
```

These are core correctness mechanisms.

The following are implementation details, not new architecture concepts:

```text
run_generation
prediction ring
executor polling
worker heartbeat
RecorderIO process
```

---

## 3. Policy boundary

Real should depend only on the public Policy contract:

```text
PolicySpec
    action_key
    control_action_dim
    n_obs_steps
    n_action_steps
    observation_fields
    control_dt_s

predict(observation)
    -> float64 ActionChunk [N, D]
```

Real must not understand:

```text
Diffusion
Flow Matching
solver details
normalizer internals
model preprocessing
EMA
```

Remove all Real-side assumptions about generic temporal blending.

---

## 4. Remove obsolete generic temporal semantics

Current code contains compatibility checks for:

```text
temporal_ensemble_coeff
```

This is not a supported Real runtime feature.

Required:

- remove active `temporal_ensemble_coeff` dependency from Real deployment config;
- remove runtime validation whose only purpose is checking this obsolete field;
- keep canonical Policy action execution only.

Do not implement:

```text
Temporal Ensemble
ChunkOverlapBlender
RTC
previous-action conditioning
```

If a future experiment studies these methods, they must be explicit Policy algorithms, not hidden Real runtime behavior.

---

## 5. Action execution semantics

The executor should implement standard timestamped action chunk execution:

For chunk:

```text
A = [a0, a1, ..., aN]
```

with:

```text
t_i = t_chunk + i * control_dt
```

Rules:

```text
future action
    execute at target time

past action
    skip

entire chunk stale
    discard chunk
```

Do not:

```text
retime stale actions
execute old actions faster to catch up
blend old/new chunks
```

---

## 6. Remove hidden duplicated timing semantics

Observation freshness and command freshness are different concepts.

Keep:

```text
observation age
observation modality skew
```

for building valid Policy input.

Keep one command delivery timeout for:

```text
executor → robot worker
```

Avoid using sensor source age again after Policy already produced an ActionChunk.

Remove or simplify duplicated concepts such as:

```text
max_source_to_command_age_s
```

if they only revalidate the already-created action chunk.

---

## 7. Safety policy

Safety should be reject-oriented.

Recommended learned-policy checks:

```text
arm joint limits
hand joint limits
per-step joint delta limits
EEF workspace limits
hardware feedback health
```

Do not silently modify learned actions.

Change:

```text
arm spike clipping
```

into:

```text
SAFETY_REJECT
```

unless a later experiment explicitly studies a command limiter.

Do not make PolicyExecutor an online motion planner.

Avoid adding:

```text
online collision planner
trajectory optimizer
adaptive safety correction
```

---

## 8. IK and safety attribution

Maintain clear failure semantics:

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
        IK_FAIL
    ↓
safety
        SAFETY_REJECT
```

Do not record joint-limit/action-admission failures as IK failures.

---

## 9. Formal evaluation redesign

Formal evaluation must execute the same control path as normal deployment.

Current desired structure:

```text
B
↓
Recorder START
↓
RUNNING
↓
normal policy rollout
↓
outcome
↓
Recorder STOP/save
```

Recorder is an observer, not a controller.

Do not allow evaluation-only evidence collection to block action publication.

Example:

```text
state-only policy
camera recording failure
```

should not change policy control semantics.

Recording failure may produce:

```text
INVALID episode
```

but should not automatically become a robot control fault.

---

## 10. Episode storage semantics

Every episode that enters RUNNING should be preserved.

Separate:

```text
save episode
```

from:

```text
outcome
```

Store:

```text
SUCCESS
FAILURE
INVALID
reason
```

Invalid episodes are valuable debugging data and should not disappear.

---

## 11. Home/start lifecycle

Avoid redundant authorization state.

Prefer:

```text
H
↓
robot moves home

B
↓
check current robot state
↓
start if valid
```

The current measured robot state is the source of truth.

Do not rely on a long-lived flag whose meaning is "home was completed previously" unless required by a specific hardware constraint.

---

## 12. Files expected to change

Expected areas:

```text
dexmani_real/deployment/config.py
dexmani_real/deployment/inference/*
dexmani_real/deployment/executor.py

dexmani_real/control/publication.py
dexmani_real/control/safety_gate.py

dexmani_real/runtime/safety.py

dexmani_real/deployment/evaluation.py

dexmani_real/recording/*

examples/run_policy.py
README.md
repo_map.md
```

Do not rewrite:

```text
teleop
robot workers
IPC implementation
sensor drivers
```

unless a direct dependency requires it.

---

## 13. Focused tests

Add only deterministic tests for semantics:

```text
partial stale chunk
whole stale chunk
old run_generation discarded
IK failure classification
safety rejection classification
evaluation recording does not alter action scheduling
```

No hardware tests.

---

## 14. Explicit non-goals

Do not implement:

```text
RTC
Temporal Ensemble
adaptive horizon
latency compensation
steps_per_inference
new scheduler framework
new recorder architecture
metrics dashboard
industrial deployment framework
```

If later experiments require these, add them as explicit research variants.

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
python -m pytest -q <focused tests>
git diff --check
git diff --stat
```

Do not run real robot commands unless explicitly authorized.

---

## 16. Definition of Done

Complete when:

```text
Policy execution path is simple and timestamp-correct
```

```text
Normal deployment and eval share the same control semantics
```

```text
IK failures and safety failures are distinguishable
```

```text
Recording no longer silently changes policy behavior
```

```text
No generic temporal blending mechanism exists in Real runtime
```

Report changed files, verified tests, and remaining unverified hardware items only.
