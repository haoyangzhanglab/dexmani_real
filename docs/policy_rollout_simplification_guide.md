# DexMani Real — Canonical Policy Rollout Simplification Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Intended executor: Claude Code / Codex  
> Cross-repository Policy baseline: `haoyangzhanglab/dexmani_policy@8fb7bac7a898433f290404e2a0567ab215c0e4ba`
> Scope: simplify and correct learned-policy real deployment for personal PhD robot-learning experiments.
>
> This guide supersedes previous rollout/formal-evaluation cleanup plans. It is the current implementation contract.

Implementation status: the coordinated API, periodic full-chunk rollout, single
physical rollout, shared run/eval recording, separate task outcome, and Formal
Eval transaction removal are implemented. Validation is offline only. Verify
hardware in order: check, shadow, low-risk run, raw inspection, then eval.
Retired mechanisms below are removal history, not compatibility requirements.

---

# 1. Objective

`dexmani_real` is a research rollout and data-collection system, not a generic robot deployment framework.

The goal is:

```text
checkpoint
    ↓
Policy rollout
    ↓
one reliable physical episode
    ↓
raw recording + reproducible metadata
```

The repository should prioritize:

```text
simple semantics
correct control behavior
paper reproducibility
low maintenance
```

It should not introduce:

```text
new scheduler research
new action aggregation
new VLA serving framework
new evaluation framework
```

---

# 2. Final rollout semantics

The production rollout model is:

```text
synchronous chunk inference
+
asynchronous timestamped robot execution
```

This matches the common Diffusion Policy / UMI style.

Meaning:

```text
Inference worker
    observation
        ↓
    Policy.predict_action_chunk()
        ↓
    future action chunk
        ↓
    publish timestamped prediction

Executor
    consumes valid future actions
    skips stale prefix
    replaces old future plan with newer valid plan
    sends commands to robot
```

Do NOT expose this as a user-selectable:

```text
--inference-mode sync/async
```

There is one production rollout semantics.

---

# 3. Cross-repository ownership

## dexmani_policy owns

```text
model
checkpoint
training seed
runtime eval seed
horizon
n_obs_steps
n_action_steps
full future action chunk
stochastic inference
```

Public boundary:

```python
predict_action_chunk(obs)
    -> float64 [chunk_size, control_action_dim]
```

where:

```python
chunk_size = horizon - n_obs_steps + 1
```

---

## dexmani_real owns

```text
camera/sensor acquisition
causal observation assembly
timestamp scheduling
stale action filtering
future plan replacement
IK
physical safety
arm/hand execution
recording
rollout outcome
```

Real must not parse Policy artifact internals.

---

# 4. Policy API migration

Policy exposes `PolicySpec.chunk_size = horizon - n_obs_steps + 1` and
`LoadedPolicy.predict_action_chunk(observation)` returning finite
`float64[chunk_size, control_action_dim]`. Real's runtime protocol and NumPy
adapter use this public method directly. There is no old-API fallback.

Real-side contract:

```text
prediction length = chunk_size
query period      = n_action_steps
```

Do not add:

```text
replan_steps
steps_per_inference
execution_horizon
queue threshold
```

Those duplicate Policy semantics.

---

# 5. Inference scheduling (implemented)

Production has one scheduling path; the former mode labels are retired:

```text
sync rollout / async rollout  →  one periodic rollout path
```

The inference worker runs one periodic schedule:

```text
period = n_action_steps * control_dt
```

At each inference point:

```text
build causal observation
        ↓
Policy.predict_action_chunk()
        ↓
publish Prediction
```

The executor continues consuming previous future actions while inference is running.

---

# 6. Action timestamp semantics

Every prediction action has a target control-grid timestamp.

When a new prediction arrives:

```text
past actions
    ↓
discard

future actions
    ↓
valid candidate plan
```

Rules:

## Keep

```text
timestamp-based stale filtering
first_future_step_index
latest valid future plan wins
whole stale discard
no catch-up execution
```

## Do not add

```text
Temporal Ensemble
chunk blending
weighted averaging
RTC
adaptive horizon
```

Policy produces actions. Real schedules them.

---

# 7. Prediction buffer requirements

Because Policy returns full future chunk:

```text
chunk_size >= n_action_steps
```

is the ideal latency-hiding case.

Example:

```text
chunk_size = 15
n_action_steps = 8
control_dt = 62.5ms
```

Future tail:

```text
7 steps
= 437.5ms
```

Inference latency does not need to be below one control period.

The important metric is:

```text
did the executor run out of future valid actions?
```

---

# 8. Compatibility checks

Keep strict checks for physical correctness:

```text
observation fields
shape
dtype
point cloud semantics
fingertip semantics
eef/tactile semantics
requires_hand
action_key
control dimension
control_dt
```

Do not weaken semantic validation during this refactor.

Current EEF/tactile support is a Real capability; do not mix adding new Policy observation fields into this rollout refactor.

---

# 9. Eval seed support

Training seed and evaluation seed are different concepts.

Required:

```bash
run_policy.py eval EXPERIMENT --eval-seed 0
```

Meaning:

```text
training seed
    checkpoint identity

 eval seed
    rollout stochastic state
```

Remove any Real-side restriction that forces:

```text
seed == 0
```

The Policy runtime already supports arbitrary non-negative seeds.

---

# 10. run_policy.py command contract

## Available commands

```text
list
check
shadow
run
eval
```

---

## Final intent

### list

Policy discovery only.

### check

```text
restore
warmup
full action-chunk smoke test
```

No hardware.

### shadow

Full lifecycle, no actuator publication.

### run

One physical debugging rollout.

### eval

One physical paper rollout.

---

# 11. CLI boundary

The following implementation controls are intentionally absent:

```text
--inference-mode
--runtime-config
--max-action-steps
--output-dir
```

Reason:

They expose implementation details rather than experiment identity.

The experiment identity is:

```text
Policy selector
checkpoint
Eval seed
task
```

---

# 12. Single rollout per invocation

`run_policy.py` executes:

```text
process start
    ↓
robot ready
    ↓
H home
    ↓
B begin
    ↓
one rollout
    ↓
result
    ↓
H return home
    ↓
Q exit
```

The invocation does not support:

```text
trial 1
trial 2
trial 3
```

inside one process.

A second B after completion should be rejected.

---

# 13. Keyboard interface

Keep operator control.

Before rollout:

```text
H home
B begin
Q quit
ESC emergency stop
```

During eval:

```text
S success
C failure
D invalid
Q abort
ESC emergency stop
```

After rollout:

```text
H home
Q quit
ESC emergency stop
```

---

# 14. Recording policy (implemented)

Physical rollout recording is explicit:

```text
shadow
    no formal recording

run
    record raw episode

eval
    record raw episode
```

Recording is an observer of rollout, not a controller of action execution.

---

# 15. Output layout

Use:

```text
rollouts/
```

Example:

```text
rollouts/
└── policy/
    └── task/
        └── experiment/
            └── eval/
                └── seed_001/
                    └── episode_timestamp/
                        ├── data.h5
                        ├── rgb.mp4
                        ├── depth.h5
                        └── result.json
```

Do not require users to manually select output directories.

---

# 16. Recorder outcome semantics

Separate:

```text
storage success
```

from:

```text
robot task success
```

Do not use one boolean for both.

Recommended API:

```python
stop_episode(save=True, reason=...)
```

Task outcome belongs in:

```text
result.json
```

Example:

```json
{
  "outcome": "success",
  "eval_seed": 1,
  "duration_s": 18.4,
  "stop_reason": "operator"
}
```

Keep raw-v24 compatibility. Do not migrate dataset schema in this task.

---

# 17. Formal evaluation simplification

The Formal Eval transaction cleanup is implemented. The following removal list
describes the retired design, not mechanisms to restore.

## Retired mechanisms

```text
per-action evaluation evidence gate
initial evidence ownership inside executor
pending evaluation termination transaction
evaluation-specific observation builder
multi-episode evaluation session
```

## Retained mechanisms

```text
Recorder START / RECORDING ACK as the sole startup recording barrier
Recorder transactional finalization
supervisor/lifecycle fault handling
hardware safety
```

After startup barrier:

The first ordinary recorded control-grid sample starts the episode evidence.
There is no mandatory initial held frame, second observation builder, or
per-command evidence admission transaction. A recording failure marks the
rollout INVALID and immediately revokes future motion generation; it does not
wait for arm/hand acceptance. CommandProgress remains a hardware liveness guard.
Normal STOP/finalization and result.json completion remain bounded, with motion
already fenced. Task FAILURE can still save raw data; the recorder's `save`
argument describes storage commitment, not task success.

Control order:

```text
action due
    ↓
policy decode
    ↓
IK/safety
    ↓
publish or reject
    ↓
record outcome
```

Recording failure should not retroactively change control semantics.

---

# 18. Keep safety mechanisms

Do not remove:

```text
E-stop
run_generation
worker generation fence
joint limits
workspace checks
SafetyGate
SDK final checks
command liveness watchdog
```

These are physical correctness mechanisms, not deployment complexity.

---

# 19. Metrics

Keep only useful rollout diagnostics:

```text
inference_latency
observation_age
observation_skew
schedule_lateness
skipped_prefix_steps
stale_prediction_count
ik_rejection_count
safety_rejection_count
command_progress_timeout
```

Do not add scheduler research metrics.

Live logging does not reset rollout rejection counts. The current result stores
executor counts and latest timing samples; it does not claim episode-wide
inference-latency percentiles. The optional action-gap metric remains deferred.

---

# 20. Completed migration milestones

The coordinated migration is complete for the offline implementation:

```text
PolicySpec.chunk_size + predict_action_chunk()
absolute n_action_steps cadence
full future-chunk transport and timestamped execution
single rollout CLI and automatic rollouts/ output
shared run/eval recording with separate result.json outcome
Formal Eval transaction removal
```

Hardware-only validation remains a separate operator procedure: `check`,
`shadow`, a low-risk `run`, raw inspection, then `eval`.

---

# 21. Non-goals

Do not implement:

```text
RTC
LeRobot async server
ActionQueue
Temporal Ensemble
chunk aggregation
adaptive scheduler
remote inference
Hydra deployment configs
new dataset schema
EEF/tactile Policy training support
model changes
```

---

# 22. Definition of Done

The current system satisfies:

```text
one rollout path
one action scheduling semantics
one Policy deployment boundary
```

A researcher can run:

```bash
python examples/run_policy.py eval policy/task/experiment --eval-seed 1 --max-duration 60
```

and obtain:

```text
checkpoint identity
seed identity
raw episode
result metadata
latency diagnostics
```

without understanding internal scheduler details.

The deployment layer should support research, not become the research topic.
