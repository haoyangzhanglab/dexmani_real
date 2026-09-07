# DexMani Real — Canonical Policy Rollout Simplification Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Intended executor: Claude Code / Codex  
> Cross-repository Policy baseline: `haoyangzhanglab/dexmani_policy@main`  
> Scope: simplify and correct learned-policy real deployment for personal PhD robot-learning experiments.
>
> This guide supersedes previous rollout/formal-evaluation cleanup plans. It is the current implementation contract.

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

Current Policy already computes:

```text
pred_action [horizon, action_dim]
```

and exposes only:

```text
control_action [n_action_steps, control_action_dim]
```

The migration should expose the complete future chunk.

Required Real-side assumption after migration:

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

# 5. Inference scheduling

Remove the conceptual distinction:

```text
sync rollout
async rollout
```

The worker should run one periodic schedule:

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

# 10. run_policy.py redesign

## Keep commands

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
synthetic prediction
```

No hardware.

### shadow

Full lifecycle, no actuator publication.

### run

One physical debugging rollout.

### eval

One physical paper rollout.

---

# 11. Remove unnecessary CLI complexity

Remove:

```text
--inference-mode
--runtime-config
--max-action-steps
--output-dir
```

Reason:

They expose implementation details rather than experiment identity.

The experiment identity should be:

```text
Policy selector
checkpoint
Eval seed
task
```

---

# 12. Single rollout per invocation

`run_policy.py` should execute:

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

Do not support:

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

# 14. Recording policy

Physical execution is expensive.

Therefore:

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

Current Formal Eval contains mechanisms for production audit.

For personal PhD experiments, simplify.

## Remove

```text
per-action evaluation evidence gate
initial evidence ownership inside executor
pending evaluation termination transaction
evaluation-specific observation builder
multi-episode evaluation session
```

## Keep

```text
Recorder START acknowledgement
initial startup barrier
Recorder transactional finalization
supervisor/lifecycle fault handling
hardware safety
```

After startup barrier:

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

---

# 20. Implementation order

## Phase 0

Fix obvious correctness:

```text
eval seed support
Recorder outcome semantics
```

## Phase 1

Policy chunk API migration:

```text
dexmani_policy.predict_action_chunk()
```

## Phase 2

Real rollout migration:

```text
remove sync/async public mode
consume full future chunk
unify inference schedule
```

## Phase 3

run_policy simplification:

```text
single rollout
simpler CLI
automatic output
```

## Phase 4

Formal Eval simplification:

```text
remove per-action evidence transaction
keep startup/recording safety
```

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

The final system should satisfy:

```text
one rollout path
one action scheduling semantics
one Policy deployment boundary
```

A researcher can run:

```bash
python examples/run_policy.py eval policy/task/experiment --eval-seed 1
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
