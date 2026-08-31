# DexMani Real Learned-Policy Deployment Refactor Plan

> **Audience**: Codex / repository maintainers  
> **Scope**: `dexmani_real` only.  
> **Execution order**: start this plan only after the Policy handoff defined below is available.  
> **Baseline reviewed**: `main` at `f76384789fce5c2f35dd55dcfdc6f7dc276098a6` (2026-08-31).  
> **Rule**: before every phase, re-read current `HEAD`, `AGENTS.md`, `code_style.md`, and the touched source files. If `main` has moved, re-evaluate facts instead of applying this snapshot mechanically.

---

## 1. Goal

Simplify learned-policy deployment while preserving the safety properties that are materially stronger than ManiUniCon and Stanford Diffusion Policy real-world runtimes.

Target runtime:

```text
verified deployment-v2 artifact
        ↓
strict Policy restore
        ↓
causal ObservationBatch
        ↓
agent.predict_action()
        ├── pred_action      validate full model output
        └── control_action   only executable output
                ↓
immutable logical targets
                ↓
expired-prefix drop
                ↓
latest due endpoint
                ↓
EE→IK when required
                ↓
SafetyGate
                ↓
atomic coupled command ticket
                ↓
worker final permit / bounds
                ↓
SDK
```

The intended simplification is **fewer duplicate concepts**, not weaker causality or weaker hardware fences.

---

## 2. Required Policy handoff

Do not begin Real control-semantic changes until `dexmani_policy` has delivered:

```text
Policy commit SHA
representative deployment-v2 fixture
fixture checkpoint SHA-256
schema-v2 sidecar
supported-policy matrix
strict-restore result
direct/export control_action parity result
no-network/no-external-file result
```

Policy implementation plan:

```text
dexmani_policy/docs/real_deployment_implementation_plan.md
```

The handoff must establish:

```text
pred_action is the full model prediction
control_action is the only default executable slice
n_action_steps defines executable length
tail is optional and is not a Real execution contract
```

If the Policy artifact contract changes, stop and resolve the producer contract first. Do not patch Real around a moving producer.

---

## 3. Non-goals

Do **not** do the following as part of this refactor:

- Do not design `deployment.v3` just to rename existing fields.
- Do not make Real read Policy `simple.v1` training checkpoints.
- Do not add general registries/factories/manager-service-controller hierarchies.
- Do not weaken no-follow/hash/TOCTOU/provenance checks at the artifact boundary.
- Do not remove run generations.
- Do not retime stale policy actions to a new future slot.
- Do not clip learned arm/workspace actions and continue as if the original proposal were valid.
- Do not remove `SafetyGate`, atomic ticketing, or worker SDK-boundary validation.
- Do not replace `ActionBuffer` before differential evidence exists.
- Do not merge H4 and task execution bounds into one generic physical-bounds class.
- Do not claim `shadow` is zero hardware side effect while hand startup can still move/reset the hand.
- Do not run any real-hardware command through Codex/tooling.

---

## 4. Safety and causality invariants to preserve

### 4.1 Observation causality

Every selected state/sensor record must satisfy:

```text
0 < source_monotonic_ns <= publish_monotonic_ns <= anchor_monotonic_ns
```

Camera paths additionally preserve their receive/payload-ready ordering and clock generation.

For a visual policy grid:

```text
visual.source_ns <= desired_logical_step_ns
logical_step_ns - visual.source_ns <= max_grid_lag_ns
```

Robot state paired to the visual sample:

```text
state.source_ns <= visual.source_ns
visual.source_ns - state.source_ns <= max_observation_skew_ns
```

Never use future state, previous-run state, or a repeated old visual frame to fabricate a complete history.

### 4.2 Run isolation

Each `ARMED → RUNNING` must:

- advance `run_generation`;
- create a new `run_started_monotonic_ns`;
- invalidate prior coupled commands;
- reset inference episode state;
- reset scheduler state;
- prevent old plans/commands/ACKs from executing.

### 4.3 Immutable action timing

Canonical target grid:

```text
target_i = observation_logical_step_ns + i * control_dt_ns
```

Allowed:

```text
expired prefix → drop
all expired    → drop whole prediction
```

Forbidden:

```text
expired action → move to next future slot
shift entire chunk to ensure execution
```

### 4.4 Three command fences

Keep all three:

1. **Coordinator SafetyGate** for policy-semantic motion safety.
2. **Atomic publication permit** for validation→publication STOP/generation races.
3. **Worker final permit** for shared-memory→SDK STOP/generation/expiry races.

They protect different boundaries.

---

## 5. Current reviewed issues

### P0 semantic bugs

1. Real currently derives executable future actions from full `pred_action` instead of using Policy `control_action`.
2. Reference `horizon=16`, `n_obs_steps=2`, `n_action_steps=8` therefore means Policy intends 8 executable steps while current Real can treat 15 future prediction steps as actionable.
3. Warmup/readiness uses a raw horizon-derived viable window and can report ready when the true source-to-command deadline leaves too few executable endpoints.
4. Physical seed can be inherited too implicitly; physical rollout must be explicitly reproducible.

### P1 complexity/accuracy problems

5. `deployment/worker.py` reimplements causal history/alignment semantics that already belong in `ipc/causal.py`.
6. `run_policy.py` mixes inspect/preflight/shadow/H4/task concerns into one flat parser.
7. Business/runtime failures go through parser-style usage errors.
8. Default console output is too noisy: periodic metrics, per-endpoint logs, large JSON/config dumps.
9. Multi-process log filenames can collide because they are second-resolution without PID.
10. `shadow` docs overstate lack of hardware side effects; hand startup may still call `reset_home()`.
11. Authoritative docs describe 15 future prediction steps as actionable instead of distinguishing 15 predicted vs 8 executable.
12. `AGENTS.md` contains a stale statement about lacking a general unit-test suite despite current focused tests.

### P2 simplification candidates

13. `valid_mask` currently represents only an expired prefix on a strictly increasing target grid; arbitrary hole masks are unnecessary generality.
14. Current default `max_source_to_command_age_s` is tighter than `max_plan_age_s`, so the latter may be redundant after evidence.
15. `InferenceContext.step_dt_ns` appears not to be needed across the plan IPC boundary.
16. Current candidate builder makes delivery `target_monotonic_ns` equal to creation time, suggesting a redundant timing identity.
17. `plan_id`, `observation_id`, and ring sequence may carry overlapping identity semantics.
18. `ActionBuffer` is correct but complex; a target-indexed endpoint buffer may express the same latest-wins semantics more directly.

---

## 6. Phase R1 — execute `control_action`

**Status**: BLOCKED on Policy handoff  
**This is the first Real semantic change.**

### R1.1 Policy adapter

Primary files to inspect/update:

```text
dexmani_real/integrations/dexmani_policy.py
dexmani_real/deployment/manifest.py
dexmani_real/deployment/contracts.py
relevant tests
docs/policy_deployment.md
```

Inference result:

```python
with torch.inference_mode():
    result = agent.predict_action(obs_dict, denoise_timesteps=...)

pred = require_tensor(result, "pred_action")
control = require_tensor(result, "control_action")
```

Validate full output:

```text
pred.shape == [1, horizon, model_action_dim]
pred finite
```

Validate executable output:

```text
control.shape == [1, n_action_steps, control_action_dim]
control finite
```

Only `control_action` is decoded into the Real `PolicyPrediction`/plan path.

### R1.2 Preflight consistency

During isolated check/warmup, verify:

```python
start = n_obs_steps - 1
expected = pred[
    :,
    start:start + n_action_steps,
    :control_action_dim,
]
```

and compare with `control_action` exactly when dtype/implementation permits, or with a narrowly justified tolerance.

Do not force this equality synchronization every online tick if it adds GPU sync/latency. The hot path needs shape/finite checks; the full semantic equivalence belongs in preflight/tests.

### R1.3 Model-specific cases

#### R3D auxiliary EE

- Validate full 28-D prediction if that is the model action dimension.
- Execute only 19-D `control_action`.
- Do not truncate full prediction before validating it.

#### DQ-RISE current default

Current Policy default is `action_ee`.

Real must decode 21-D control action as:

```text
EE position + EE rot6d + hand qpos
```

and route arm motion through collision-aware IK.

### R1.4 `tail`

Do not require `tail` in the Real adapter. DQ-RISE does not return it.

Temporal ensembling is a separate algorithmic ablation and is not enabled by default in Real.

### R1.5 Documentation

Replace:

```text
15 actionable future steps
```

with:

```text
prediction future steps = horizon - (n_obs_steps - 1) = 15
executable control steps = n_action_steps = 8
```

Keep sidecar `required_action_steps` for backward-compatible serialization, but document it as the legacy prediction-future/allocation length, not the execution length.

### R1 tests

Must include:

- sentinel values in prediction tail never enter Real plan/scheduler;
- `control_action` exact execution shape;
- full `pred_action` shape/finite still validated;
- R3D full-vs-control dimension case;
- DQ-RISE EE decode;
- existing joint DP3 path unchanged except for shortened executable horizon.

**Important**: old H4/task evidence was produced under old full-future execution semantics and cannot authorize the new behavior. New physical promotion is required later.

---

## 7. Phase R2 — real deadline readiness/timing

**Status**: BLOCKED on R1

### R2.1 One pure timing helper

Add a small pure module, e.g.:

```text
dexmani_real/deployment/timing.py
```

Core function:

```python
def usable_target_mask(
    *,
    logical_step_ns: int,
    steps: int,
    step_dt_ns: int,
    inference_finished_ns: int,
    observation_source_ns: int,
    command_lead_ns: int,
    max_plan_age_ns: int,
    max_source_to_command_age_ns: int,
) -> np.ndarray:
    earliest = inference_finished_ns + command_lead_ns
    deadline = min(
        inference_finished_ns + max_plan_age_ns,
        observation_source_ns + max_source_to_command_age_ns,
    )
    targets = logical_step_ns + np.arange(steps, dtype=np.int64) * step_dt_ns
    return (targets > earliest) & (targets < deadline)
```

Use the same timing semantics in warmup/stamping/deadline tests instead of separate formulas.

### R2.2 Startup gate

Keep startup lightweight:

```text
warmup samples = 5
stable samples = last 3
```

Require a small minimum number of executable endpoints in each stable sample, e.g. 2 for normal shadow/task qualification and 1 for one-endpoint H4 if the explicit H4 path requires it.

The exact threshold is an engineering constant; document it and test the boundary.

### R2.3 Isolated benchmark

Detailed latency profiling belongs in an explicit offline command:

```bash
python examples/run_policy.py check EXP --benchmark-samples 100
```

Report:

```text
warmup samples
p50/p95/max inference
usable endpoint counts
GPU peak memory if available
```

Do not impose a 100-sample benchmark on every hardware startup.

### R2.4 Inference mode

Use `torch.inference_mode()` for deployment inference unless a specific policy requires otherwise and proves it.

---

## 8. Phase R3 — CLI, logging, and research ergonomics

**Status**: can follow R1/R2; do not mix with scheduler changes.

### R3.1 Thin example entry

Keep:

```text
examples/run_policy.py
```

as a thin research-friendly wrapper around reusable package code.

Suggested UX:

```bash
python examples/run_policy.py inspect EXP
python examples/run_policy.py check EXP --device cuda:0 --seed 1066
python examples/run_policy.py shadow EXP --device cuda:0 --seed 1066
python examples/run_policy.py h4 PROFILE.yaml
python examples/run_policy.py run PROFILE.yaml
```

No subcommand → print help only, no Torch/hardware side effects.

### R3.2 Side-effect contract

| Command | Torch/GPU | Hardware connection | learned coupled writes |
|---|---:|---:|---:|
| `inspect` | no | no | no |
| `check` | yes | no | no |
| `shadow` | yes | yes | 0 |
| `h4` | yes | yes | max 1 endpoint |
| `run` | yes | yes | bounded by profile |

`shadow` must explicitly print/document current hand startup behavior.

### R3.3 Physical profiles

Profile owns only Real run inputs:

```text
experiment/artifact path
runtime config
deployment config
device
seed
expected checkpoint SHA
max runtime
endpoint bound
ACK timeout
```

Do not duplicate artifact-owned fields such as action dimension, horizon, modalities, EMA, point count, or NFE.

Keep `H4ExecuteBounds` and `TaskExecuteBounds` separate.

### R3.4 Physical seed

Physical `h4/run` must have an explicit seed in profile or CLI. Do not silently inherit an incidental default.

### R3.5 Typed errors

Use parser errors only for CLI syntax.

Runtime/artifact failures should be concise typed errors, not full usage dumps.

Suggested categories:

```text
ArtifactError
PolicyCompatibilityError
PolicyLoadError
HardwareStartupError
RuntimeFault
```

### R3.6 Console vs file logging

Set logger to support:

```text
console = INFO
file = DEBUG
```

Move to DEBUG:

- periodic metrics snapshots;
- per-endpoint publication logs;
- full point-cloud/config JSON;
- full receipt JSON;
- repetitive low-level worker lifecycle diagnostics.

Keep at INFO:

- policy summary;
- aggregated readiness;
- ARMED/RUNNING/STOPPED;
- operator prompts;
- compact periodic status;
- final result and receipt path.

Log filename should include PID:

```text
dexmani_YYYYMMDD_HHMMSS_<pid>.log
```

Do not add a queue/event-bus/observability framework.

### R3.7 Offline check vs online load

`check` is an optional offline diagnostic, not a mandatory duplicate load before every online run.

A normal `shadow/h4/run` should still perform the authoritative verified load in the inference child before hardware workers are promoted ready.

---

## 9. Phase R4 — consolidate causal readers

**Status**: after R1 baseline is stable  
**Behavior target**: no observation-selection semantic change.

Current repository already has `ipc/causal.py` defining source/publish/anchor semantics, while deployment worker contains a second history/alignment implementation.

### R4.1 Extend `ipc/causal.py`

Add small pure helpers, e.g.:

```python
read_causal_structured_history(...)
align_history_to_reference_sources(...)
```

They should own:

```text
source <= publish <= anchor
not-before run epoch
max age
required health flags
oldest-first history
latest state at-or-before each visual source
max source skew
```

### R4.2 Deployment worker retains only domain-specific work

Keep in `deployment/worker.py`:

- point-cloud/RGB payload validation;
- camera generation checks;
- visual control-grid selection;
- requested modality assembly;
- metrics.

Remove or delegate duplicate causal predicates/history state alignment.

### R4.3 Visual-grid wrapper cleanup

Unify point-cloud/RGB visual-grid selection when the algorithm is identical. Remove a typed wrapper that merely calls the generic implementation and rechecks type if it provides no independent semantic boundary.

### R4.4 Differential tests

Before deleting the old implementation, run old/new pure selection against identical synthetic histories.

Cover:

```text
future state
stale state
B-before-run sample
old camera generation
clock reset
duplicate visual sequence
missing history
max-grid-lag boundary
max-skew boundary
```

Outputs must match exactly for behavior-preserving cases.

---

## 10. Phase R5 — wire/scheduler simplification

**Status**: only after R1–R4 establish a new tested baseline.

Each item below should be a separate small PR.

### R5.1 Replace arbitrary `valid_mask` with expired-prefix slicing

Targets are strictly increasing. The inference-latency validity pattern therefore only needs to represent:

```text
0000011111
```

not arbitrary holes.

Use:

```python
first_valid = np.searchsorted(
    target_ns,
    inference_finished_ns + command_lead_ns,
    side="right",
)
if first_valid == len(target_ns):
    return None
```

Then slice actions/targets and transport only remaining endpoints.

After tests prove equivalence, remove:

```text
JointActionChunk.valid_mask
policy plan IPC valid_mask
ActionBuffer valid-mask branches
mask validation/copy logic
```

Never retime the remaining target timestamps.

### R5.2 Evaluate deadline convergence

Current plan deadline:

```text
min(
  inference_finished + max_plan_age,
  observation_source + max_source_to_command_age,
)
```

With current defaults, source-to-command is the tighter bound.

Before deleting `max_plan_age_s`:

- prove the dominance condition for current supported profiles;
- add differential tests;
- check H4/task profiles for overrides.

If truly redundant, converge on a single clearly named:

```text
max_observation_to_command_age_s
```

### R5.3 Remove redundant timing/identity fields one at a time

Candidates, in order:

1. `InferenceContext.step_dt_ns` if not consumed across IPC;
2. candidate delivery `target_monotonic_ns` if it is always equal to creation time and no caller depends on the distinction;
3. explicit `plan_id` if observation identity + ring sequence fully cover semantics.

For each field deletion update all:

```text
dataclass
IPC dtype
writer
reader
receipt/logging
tests
```

Do not batch these removals.

### R5.4 Experimental EndpointBuffer

Do not immediately replace `ActionBuffer`.

Implement a test-only/experimental target-indexed buffer concept:

```text
target_ns → newest endpoint
finalized_through_ns
```

New plans permanently replace older endpoints at identical target timestamps; expiry must not resurrect the older endpoint.

Run recorded/synthetic plan streams through:

```text
legacy ActionBuffer
experimental EndpointBuffer
```

Compare:

- selected target;
- observation identity;
- endpoint tensor;
- commit/discard sequence;
- stale/fallback behavior;
- command silence.

Only after equivalence:

```text
offline replay
→ multiprocess shadow
→ live shadow
→ fresh H4
```

If maintenance/readability gain is not material, keep the existing ActionBuffer.

---

## 11. Phase R6 — safety code local cleanup

**Status**: after scheduler/timing semantics are stable.

### R6.1 Keep

Do not remove:

```text
raw policy SafetyGate
shaped-hand second SafetyGate
atomic publication permit
worker final permit
worker actuator-local bounds/jump checks
arm/hand ACK
generation invalidation
H4/task bounds
```

### R6.2 Consolidate only redundant coordinator permit reads

Desired conceptual pipeline:

```text
early cheap runtime rejection
→ fresh feedback + SafetyGate
→ final atomic state/generation/expiry/ring publication
→ worker final gate
```

The early check avoids expensive work on an already-stopped run; the atomic check closes the race before shared-memory publication; worker check closes the race before SDK. Do not collapse these into one check.

Never hold `motion_lock` across hardware IO.

### R6.3 Hand bounds preflight

The hand mechanical/rated preflight after the second gate is partly redundant mathematically but cheap and explicit at the coupled-publication boundary. Keep it unless a focused proof/test demonstrates removal improves clarity without losing defense-in-depth.

Do not add more equivalent layers.

---

## 12. Phase R7 — hand observe-only startup

**Status**: DEFERRED, separate hardware-qualified PR

Current shadow guarantee should be phrased as:

```text
learned-policy coupled command writes = 0
```

not:

```text
zero hardware side effects
```

because XHand startup may perform connect/calibration/home-reset behavior.

Promotion path:

```text
fake driver
→ read-only XHand diagnostic
→ verify qpos/current/tactile stability
→ live shadow
→ introduce OBSERVE_ONLY startup mode
→ dedicated hardware evidence
```

Do not bundle with control-action/timing/scheduler work.

---

## 13. Phase R8 — RGB/text deployment

**Status**: DEFERRED until Policy contract exists.

Real must reproduce deterministic Policy evaluation preprocessing, not invent a “reasonable” resize.

When Policy artifact provides explicit RGB preprocessing metadata, Real may implement it and prove replay parity.

Dynamic text conditioning remains unsupported until an explicit task-text contract exists.

---

## 14. Performance work after correctness

Only optimize measured hot paths.

First low-risk steps:

- `torch.inference_mode()`;
- quiet console logging;
- benchmark/report inference and observation assembly separately.

Consider fixed-shape input-buffer reuse only if allocation/copy profiling shows material latency/jitter impact.

`torch.compile` must remain policy/profile-specific and opt-in. Do not make it a universal deployment default because ActionFlow/R3D/point-cloud custom ops have different graph-break/numerical characteristics.

---

## 15. Validation and promotion ladder

After R1 changes executable semantics, physical evidence must be rebuilt:

```text
Policy exported fixture
→ Real inspect
→ isolated check
→ recorded observation replay
→ multiprocess shadow
→ live shadow
→ fresh H4 one endpoint
→ bounded task transport
→ task capability evaluation
```

Transport success is not task success.

### Offline/focused tests

At minimum inspect/run relevant tests including:

```bash
python -m compileall -q dexmani_real examples tests
pytest -q tests/test_deployment_timing.py
pytest -q tests/test_action_buffer.py
pytest -q tests/test_coupled_command_publication.py
pytest -q tests/test_worker_command_validation.py
pytest -q tests/test_safety_gate_command_delta.py
git diff --check
```

Add focused tests near changed modules rather than relying only on broad smoke tests.

No Codex/tool run may connect to or command real hardware.

---

## 16. Documentation updates

### `docs/policy_deployment.md`

Make this the long-lived **consumer/operator truth**, not a refactor todo list.

Update:

- current main vs last hardware-evidenced revision;
- 15 predicted vs 8 executable;
- `control_action` execution;
- source/deadline readiness;
- shadow learned-write guarantee + hand startup caveat;
- qualified policy support matrix;
- DQ-RISE current EE action behavior;
- RGB/text deferred status;
- final CLI flow.

### `README.md`

Show concise user commands and side effects.

### `AGENTS.md`

Correct stale testing guidance; current repository has a substantial focused test suite.

This file (`policy_deployment_refactor_plan.md`) is implementation tracking and may be removed after the refactor. Do not turn it into the permanent runtime specification.

---

## 17. Codex execution rules

Before each PR:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Then read repository instructions.

Each PR must:

- modify one primary semantic variable;
- preserve unrelated user work;
- avoid generic abstractions without demonstrated duplication/boundary need;
- include focused tests;
- update authoritative documentation affected by the change;
- report environment limitations precisely.

Stop rather than guess if:

- Policy handoff is incomplete or incompatible;
- Real main materially moved from reviewed code;
- full/output control semantics conflict;
- causal differential tests differ unexpectedly;
- strict artifact restore fails;
- shadow writes a learned coupled command;
- generation/ticket/worker fence tests fail;
- SafetyGate would need to be bypassed;
- validation requires actual hardware to establish truth.

---

## 18. Recommended Real PR sequence

```text
Real PR-1  control_action semantics
    - full-output validation
    - n_action_steps execution only
    - model-specific tests

Real PR-2  timing/readiness
    - shared usable-target helper
    - real source deadline
    - inference_mode

Real PR-3  CLI/logging/research ergonomics
    - inspect/check/shadow/h4/run
    - explicit seed/profile
    - typed errors
    - quiet console

Real PR-4  causal-reader consolidation
    - ipc/causal shared history/alignment
    - differential parity

Real PR-5a valid_mask → prefix slicing
Real PR-5b deadline convergence if proven
Real PR-5c timing/identity field deletions
Real PR-5d EndpointBuffer experiment

Real PR-6  safety permit-read cleanup
Real PR-7  hand observe-only (hardware-qualified)
Real PR-8  RGB/text after Policy contract
```

Do not merge multiple PR-5 semantic removals into one large refactor.

---

## 19. Definition of Done

Core Real refactor is complete when:

- [ ] Real executes only Policy `control_action` with `n_action_steps` endpoints.
- [ ] Full `pred_action` is still validated and auxiliary dimensions are not silently discarded before validation.
- [ ] Prediction tail never executes by default.
- [ ] Warmup/readiness uses true executable endpoints under source-owned deadlines.
- [ ] Stale actions are dropped, never retimed.
- [ ] Causal state/history semantics have one shared implementation.
- [ ] Run generation, atomic ticketing, worker final permit, SafetyGate, collision, and ACK all remain enforced.
- [ ] CLI exposes clear inspect/check/shadow/h4/run side effects.
- [ ] Physical seed and bounds are explicit.
- [ ] Default console output is concise; detailed evidence remains in DEBUG/file logs and receipts.
- [ ] Shadow still demonstrates zero **learned coupled-command writes**.
- [ ] Old physical evidence is not reused after control-action semantics change; fresh H4 is completed before task rollout.
- [ ] Any ActionBuffer replacement is supported by differential + shadow + H4 evidence, or the existing scheduler is retained.
