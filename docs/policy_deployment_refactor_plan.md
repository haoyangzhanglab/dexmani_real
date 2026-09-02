# DexMani Real Learned-Policy Deployment Refactor Plan

> **Audience**: Codex / repository maintainers  
> **Scope**: `dexmani_real` only.  
> **Execution order**: start this plan only after the Policy handoff defined below is available.  
> **Reviewed baseline**: `main` at `f76384789fce5c2f35dd55dcfdc6f7dc276098a6` (2026-08-31).  
> **Review status**: v2 — cross-checked against current Real timing, preflight, artifact, SafetyGate and Policy adapter code.  
> **Rule**: before every PR, re-read current `HEAD`, `AGENTS.md`, `code_style.md`, and touched source. If `main` moved, re-evaluate facts instead of applying this snapshot mechanically.

---

## 1. Goal

Simplify learned-policy deployment without weakening the causality and hardware fences that are materially stronger than ManiUniCon and Stanford Diffusion Policy real-world runtimes.

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
plan/source deadline
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

The simplification target is **fewer duplicate concepts**, not weaker safety.

---

## 2. Required Policy handoff

Do not begin Real semantic changes until Policy delivers:

```text
Policy commit SHA
clean-source/provenance result
representative deployment-v2 artifact or deterministic fixture generator
artifact SHA-256
schema-v2 sidecar
supported-policy matrix
strict-restore result
direct/export pred_action/control_action parity result
no-network/no-external-file result
```

Policy plan:

```text
dexmani_policy/docs/real_deployment_implementation_plan.md
```

Handoff invariants:

```text
pred_action      = full model prediction
control_action   = only default executable slice
n_action_steps   = executable length
tail             = optional, not a Real contract
required_action_steps = legacy prediction-future/allocation length
```

If producer contract moves, stop and fix Policy first. Do not patch Real around a moving producer.

---

## 3. Non-goals

Do not:

- design deployment-v3 just to rename fields;
- make Real read Policy `simple.v1`;
- add generic registries/factories/manager-service-controller layers;
- weaken artifact no-follow/hash/TOCTOU/provenance checks;
- remove run generations;
- retime stale policy actions;
- clip learned arm/workspace proposals and continue as if unchanged;
- remove SafetyGate, atomic ticketing or worker SDK-boundary checks;
- replace ActionBuffer before differential evidence;
- merge H4 and task bounds;
- claim shadow is zero hardware side effect while hand startup may move/reset;
- make `torch.compile` a universal deployment default;
- run any real-hardware command through Codex/tooling.

---

## 4. Safety and causality invariants

### 4.1 Observation causality

Every selected record:

```text
0 < source_monotonic_ns <= publish_monotonic_ns <= anchor_monotonic_ns
```

Camera paths additionally preserve receive/payload-ready ordering and camera generation.

Visual grid:

```text
visual.source_ns <= desired_logical_step_ns
desired_logical_step_ns - visual.source_ns <= max_grid_lag_ns
```

State aligned to visual source:

```text
state.source_ns <= visual.source_ns
visual.source_ns - state.source_ns <= max_observation_skew_ns
```

Never use future state, previous-run state, cross-generation camera frames, or repeated old visuals to fabricate history.

### 4.2 Run isolation

Every `ARMED → RUNNING`:

- advances `run_generation`;
- creates new `run_started_monotonic_ns`;
- invalidates prior coupled commands;
- resets inference episode state;
- resets scheduler state;
- prevents old plans/commands/ACKs from executing.

### 4.3 Immutable action timing

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
expired action → new future slot
shift chunk to ensure execution
```

### 4.4 Three command fences

Keep all three:

1. coordinator SafetyGate;
2. atomic state/generation/publication ticket;
3. worker final state/generation/expiry/bounds permit before SDK.

They close different race windows.

---

## 5. Reviewed issues

### P0

1. Real currently decodes the full future interval from `pred_action`, not Policy `control_action`.
2. With `horizon=16`, `n_obs_steps=2`, `n_action_steps=8`, current Real can treat 15 predicted future steps as executable although Policy intends 8.
3. Preflight receipt and prediction validation also use `required_action_steps=15`; fixing only the adapter would leave the isolated contract inconsistent.
4. Startup warmup uses the 15-step allocation window, producing an overly permissive latency qualification.
5. Physical inference seed is too easy to inherit implicitly.

### P1

6. `deployment/worker.py` duplicates causal history/alignment semantics already represented in `ipc/causal.py`.
7. Flat `run_policy.py` CLI mixes inspect/check/shadow/H4/task.
8. Business failures use parser-style usage output.
9. Default console is noisy; process log filenames can collide.
10. Shadow docs overstate lack of side effects; XHand startup may reset/home.
11. Docs conflate 15 predicted steps with 8 executable steps.
12. `AGENTS.md` has stale test-suite guidance.

### P2 candidates

13. Transport `valid_mask` currently represents only an inference-late prefix; arbitrary holes are unnecessary unless another producer is introduced.
14. Current default source-to-command age is tighter than plan-age, so plan-age may be removable after profile proof.
15. Some timing/identity fields may be redundant.
16. ActionBuffer is correct but complex; a target-indexed endpoint buffer may be simpler if differential equivalence is proven.

---

# Phase R1 — Execute Policy `control_action`

**Status**: MERGED / REVIEWED — Real `main` at
`fd8195f757f341f99a50f232bf59820a0fb15ec6`; Policy handoff accepted.
**This is the first Real semantic change.**

## R1.1 Adapter contract

Primary files:

```text
dexmani_real/integrations/dexmani_policy.py
dexmani_real/deployment/preflight.py
dexmani_real/deployment/manifest.py
dexmani_real/deployment/contracts.py
related tests
docs/policy_deployment.md
```

Inference:

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

Only `control_action` is decoded into Real `PolicyPrediction`.

Do not require `tail`.

## R1.2 Preflight/warmup semantic consistency

During isolated preflight and startup warmup, verify the declared slice:

```python
start = n_obs_steps - 1
expected = pred[
    :,
    start:start + n_action_steps,
    :control_action_dim,
]
```

Compare with `control_action` exactly where deterministic dtype permits, otherwise with a narrow justified tolerance.

Hot path does not need an extra exact equality synchronization every tick; shape/finite checks remain mandatory.

## R1.3 Fix prediction-future → executable preflight contracts

Current preflight uses sidecar `required_action_steps` as returned action length. R1 must update all executable-length checks to `allocation.n_action_steps`:

```text
PreflightResult.action_steps
run_isolated_preflight receipt comparison
_run_preflight_child action_steps
_validate_prediction expected_steps
related tests/receipts
```

`required_action_steps` stays in the artifact/sidecar for compatibility; it no longer means Real executable length.

This step is mandatory. Do not leave two contradictory action-length contracts.

## R1.4 Model-specific cases

### R3D aux EE

- full model prediction may be 28-D;
- validate all 28 dimensions;
- execute only 19-D `control_action`.

### DQ-RISE current default

Current Policy default is `action_ee`.

Decode 21-D control action as:

```text
EE pos3 + EE rot6d6 + hand12
```

then route arm through collision-aware IK.

## R1.5 Docs

Canonical wording is formula-first:

```text
required_action_steps
= prediction future steps
= horizon - (n_obs_steps - 1)

executable control steps
= n_action_steps
```

For the representative DP3 artifact only, these values are 15 prediction-future
steps and 8 executable steps.

Old H4/task evidence was generated under old full-future semantics and cannot authorize R1.

## R1 tests

- sentinel tail values never enter Real plan/scheduler;
- control shape is exactly `n_action_steps`;
- full pred shape/finite still checked;
- preflight receipt says artifact `n_action_steps`, not `required_action_steps`;
- R3D full/control dimension case;
- DQ-RISE EE decode;
- DP3 joint path unchanged except executable horizon.

---

# Phase R2 — Timing and readiness

**Status**: MERGED / REVIEWED — Real `main` =
`effe745c68847a4b32ed1e4680041a350da4f4fe`.

## R2.1 Separate three timing concepts

Do **not** use one combined mask for both lower latency expiry and upper plan deadline. That would create masks such as `00011100` and contradict the later prefix-only simplification.

Add a small pure module, e.g. `deployment/timing.py`, with separate helpers:

```python
def build_target_grid(logical_step_ns, steps, step_dt_ns) -> np.ndarray:
    ...

def first_deliverable_index(target_ns, inference_finished_ns, command_lead_ns) -> int:
    # lower bound only: target > finish + lead
    ...

def compute_plan_deadline_ns(
    inference_finished_ns,
    observation_source_ns,
    max_plan_age_ns,
    max_source_to_command_age_ns,
) -> int:
    return min(
        inference_finished_ns + max_plan_age_ns,
        observation_source_ns + max_source_to_command_age_ns,
    )

def usable_target_mask(target_ns, first_index, deadline_ns) -> np.ndarray:
    # diagnostics/qualification only; may include a trailing deadline suffix
    ...
```

Ownership:

```text
inference stamping  → target grid + expired-prefix lower bound only
BufferedPlan        → immutable upper deadline
scheduler           → target < deadline
check/metrics       → compose both to report truly usable endpoints
```

This preserves the invariant that the **transport valid prefix state has no arbitrary holes**.

## R2.2 Startup warmup is a model-latency qualification, not a proof of source schedulability

`runtime.warmup()` currently returns model-path latencies but has no physical observation source timestamp. Do not fabricate an “exact real source deadline” from that data.

Keep:

```text
warmup samples = 5
stable samples = last 3
```

Use `n_action_steps` and the executable target grid to reject clearly impossible model latency (for example require at least a small number of theoretical post-inference targets after `command_lead`). Document this as a **qualification guard**, not an end-to-end readiness proof.

Actual online source-aware schedulability is authoritative because every real plan carries the real `observation_latest_source_ns` and coordinator deadline.

If a source-aware synthetic/advisory benchmark is added to `check`, its assumed source/logical/anchor relationship must be explicit in the receipt; do not present a synthetic assumption as measured hardware timing.

## R2.3 Online exact deadline

For each real plan:

```text
deadline = min(
    inference_finished + max_plan_age,
    observation_latest_source + max_source_to_command_age,
)
```

No endpoint with `target >= deadline` may execute.

No source-aware suffix invalidity needs to be serialized as `valid_mask`; scheduler deadline already owns it.

## R2.4 Offline benchmark

```bash
python examples/run_policy.py check EXP --device cuda:0 --seed 1066 --benchmark-samples 100
```

Report separately:

```text
model-path latency p50/p95/max
control-horizon theoretical remaining targets
source-aware usable targets only when source assumptions/recorded data exist
GPU peak memory if available
```

Do not impose 100 samples on hardware startup.

## R2.5 Inference mode

Use `torch.inference_mode()` unless a policy proves a need otherwise.

## R2 tests

- 8-step latency qualification catches cases that the old 15-step window passed;
- lower-bound prefix logic matches current stamping semantics;
- upper deadline is enforced separately;
- a deadline cutting the tail does not mutate transport into an arbitrary-hole mask;
- zero usable online endpoints safely produce no command and eventually hit existing first-command/silence watchdog semantics.

---

# Promotion Gate A — establish the new physical semantic baseline

After R1 + R2, before scheduler/wire semantic simplification:

```text
Policy fixture
→ Real inspect
→ isolated check
→ recorded observation replay
→ multiprocess shadow
→ live shadow
→ fresh H4 one endpoint
```

This creates a clean physical baseline for “8-step control_action + corrected timing”.

Do not stack R5 scheduler changes on top of R1/R2 without first obtaining this baseline. Codex does not run the live steps; the operator performs them separately.

---

# Phase R3 — CLI and logging ergonomics

**Status**: implementation on `codex/real-r3-cli-logging` — **READY FOR REVIEW**;
behavior-preserving with respect to control data flow.

```text
Gate A offline: PASS on separate evidence branch a70df31aae6d00a004f927a679ece813efc1a4d7
Gate A live and one-endpoint H4: PASS on frozen Real cc28cda511f58134c5566f7da85d65f0c1a86aac
(`gate-a-h4-validated-20260902`). The R3 candidate is a different Real revision;
it requires fresh Gate A before any physical authorization.
```

R4/R5 remain unchanged and are not part of this implementation.

## R3.1 Thin entry

Keep:

```text
examples/run_policy.py
```

as a thin wrapper around package CLI code.

Suggested UX:

```bash
python examples/run_policy.py inspect EXP
python examples/run_policy.py check EXP --device cuda:0 --seed 1066
python examples/run_policy.py shadow EXP --device cuda:0 --seed 1066 --hand --max-running-seconds 10
python examples/run_policy.py h4 PROFILE.yaml
python examples/run_policy.py run PROFILE.yaml
```

No subcommand → help only, no Torch/hardware side effect.

## R3.2 Side-effect table

| command | Torch/GPU | hardware connection | learned coupled writes |
|---|---:|---:|---:|
| inspect | no | no | no |
| check | yes | no | no |
| shadow | yes | yes | 0 |
| h4 | yes | yes | max 1 endpoint |
| run | yes | yes | bounded profile |

Shadow output/docs must state current hand startup behavior.

## R3.3 Physical profiles

Profile owns only Real run intent:

```text
experiment/artifact path
runtime config
deployment config
device
seed
expected checkpoint SHA
max running time
endpoint bound
ACK timeout
```

Do not duplicate artifact-owned model dimensions/modalities/horizon/EMA/NFE.

Keep H4 and task bounds separate.

## R3.4 Explicit physical seed

`h4/run` require explicit seed in profile or CLI. No incidental default.

## R3.5 Typed concise errors

Parser errors only for CLI syntax. Runtime/artifact failures should not reprint full usage.

Use the smallest useful typed exception set; do not create a new exception hierarchy merely for naming symmetry.

## R3.6 Logging

Logger level must allow DEBUG records while handlers filter output:

```text
logger = DEBUG
console handler = INFO
file handler = DEBUG
```

Move periodic metrics, per-endpoint logs and full JSON/config dumps to DEBUG.

Keep policy summary, readiness, ARMED/RUNNING/STOPPED, operator prompts, compact status and receipt path at INFO.

Log filename includes PID:

```text
dexmani_YYYYMMDD_HHMMSS_<pid>.log
```

Do not add queue/event-bus/observability infrastructure.

## R3.7 Check vs online load

`check` is optional offline diagnostics, not a mandatory duplicate model load before every online run.

`shadow/h4/run` still perform their own authoritative verified load in the inference child before hardware workers are promoted ready.

---

# Phase R4 — Consolidate causal readers

**Status**: after Promotion Gate A  
**Behavior target**: no observation-selection semantic change.

Current `ipc/causal.py` and deployment worker duplicate causality logic.

## R4.1 Keep layer direction clean

`ipc/causal.py` is shared low-level infrastructure. It must **not import `deployment.observation.FrameWindow` or other deployment-layer types**.

Add IPC-neutral pure helpers/data, e.g. arrays/records or a small IPC-local dataclass:

```text
read_causal_structured_history(...)
align_history_to_reference_sources(...)
```

Deployment worker converts the neutral result to `FrameWindow`.

Do not create an `ipc → deployment` dependency.

## R4.2 Shared semantics

Helpers own:

```text
source <= publish <= anchor
publish-field fallback to ring publish timestamp
not-before run epoch
max age
required health flags
oldest-first history
latest state source <= each visual source
max source skew
```

Deployment worker retains:

- point-cloud/RGB payload validation;
- camera generation;
- visual control-grid selection;
- modality assembly;
- deployment metrics.

## R4.3 Visual-grid cleanup

Unify point-cloud/RGB grid selection only where algorithm is genuinely identical. Remove a typed wrapper only if it provides no semantic boundary.

## R4.4 Differential validation

Before deleting old code, run old/new selection over identical synthetic **and recorded** histories.

Cover:

```text
future state
stale state
B-before-run sample
publish-field fallback
old camera generation
clock reset
duplicate visual sequence
missing history
max-grid-lag boundary
max-skew boundary
```

Require exact selection parity for behavior-preserving cases.

After R4, repeat offline replay and multiprocess/live shadow before relying on the new reader in further semantic refactors.

---

# Phase R5 — Wire/scheduler simplification

**Status**: only after R1–R4 and Promotion Gate A establish a stable baseline.  
Each subsection is a separate PR.

## R5.1 Replace transport `valid_mask` with expired-prefix slicing

With R2 preserving the upper deadline outside the transport mask, the inference-latency validity pattern is only:

```text
0000011111
```

Use:

```python
first_valid = np.searchsorted(
    target_ns,
    inference_finished_ns + command_lead_ns,
    side="right",
)
if first_valid == len(target_ns):
    return None

actions = actions[first_valid:]
target_ns = target_ns[first_valid:]
```

Never retime targets.

After equivalence tests remove:

```text
JointActionChunk.valid_mask
policy plan IPC valid_mask
ActionBuffer valid-mask branches
mask validation/copy logic
```

Before removal, code-search all plan producers and prove no producer creates non-prefix holes.

## R5.2 Evaluate deadline convergence

Current:

```text
min(
  inference_finished + max_plan_age,
  observation_source + max_source_to_command_age,
)
```

Before removing `max_plan_age_s`:

- prove dominance for every supported runtime/H4/task profile;
- check overrides;
- add differential tests.

Only then converge to `max_observation_to_command_age_s` if materially clearer.

## R5.3 Remove timing/identity fields one at a time

Candidates:

1. `InferenceContext.step_dt_ns` if no consumer requires it;
2. candidate delivery `target_monotonic_ns` if it is provably equal to creation and the distinction has no diagnostic/scheduling value;
3. explicit `plan_id` if observation identity + ring sequence fully cover all scheduler semantics.

These are candidates, not mandatory deletions. Keep a field if it improves conceptual clarity more than it costs.

For each deletion update dataclass, IPC, writer, reader, receipt/logs and tests in one PR.

## R5.4 Experimental EndpointBuffer

Do not replace ActionBuffer directly.

Test a target-indexed concept:

```text
target_ns → newest endpoint
finalized_through_ns
```

Newer endpoint permanently wins an identical target; newer expiry must not resurrect older target state.

Differentially compare against ActionBuffer:

- selected target;
- observation/plan provenance;
- endpoint tensor;
- commit/discard sequence;
- stale/fallback behavior;
- command silence.

Only after exact offline equivalence:

```text
recorded replay
→ multiprocess shadow
→ live shadow
→ fresh H4
```

If readability/maintenance gain is not material, keep ActionBuffer.

---

# Phase R6 — Safety code local cleanup

**Status**: after scheduler/timing semantics are stable.

Keep:

```text
raw policy SafetyGate
shaped-hand second SafetyGate
atomic publication permit
worker final permit
worker bounds/jump checks
arm/hand ACK
generation invalidation
H4/task bounds
```

Only consolidate redundant coordinator permit reads into:

```text
early cheap runtime rejection
→ fresh feedback + SafetyGate
→ final atomic state/generation/expiry/ring publication
→ worker final gate
```

Early check, atomic check and worker check remain distinct.

Never hold `motion_lock` across hardware IO.

Hand mechanical/rated preflight after the second gate is cheap intentional defense-in-depth; keep it unless a focused proof makes removal clearly better.

---

# Phase R7 — Hand observe-only startup

**Status**: DEFERRED, separate hardware-qualified PR.

Current shadow guarantee:

```text
learned-policy coupled command writes = 0
```

not:

```text
zero hardware side effects
```

Promotion:

```text
fake driver
→ read-only XHand diagnostic
→ verify qpos/current/tactile stability
→ live shadow
→ OBSERVE_ONLY startup mode
→ dedicated hardware evidence
```

Do not bundle with control/timing/scheduler changes.

---

# Phase R8 — RGB/text deployment

**Status**: DEFERRED until Policy contract exists.

Real reproduces deterministic Policy evaluation preprocessing exactly. Do not invent a plausible resize/crop.

Dynamic text remains unsupported until explicit task-text semantics exist.

---

## 6. Performance work after correctness

Low-risk first:

- `torch.inference_mode()`;
- quiet console logging;
- measure observation assembly and model inference separately.

Reuse fixed-shape input buffers only if profiling shows allocation/copy materially affects latency/jitter.

`torch.compile` remains policy/profile-specific opt-in.

---

## 7. Promotion ladder

### After R1/R2

Use Promotion Gate A and rebuild H4 because executable semantics changed.

### After behavior-preserving R3/R4

Require offline/replay and shadow evidence. A new H4 is optional if differential + shadow prove no control semantic change, but use operator judgment for any observation-path modification.

### After R5 semantic changes

Each scheduler/wire semantic change gets its own regression evidence. EndpointBuffer replacement requires fresh H4 before task rollout.

### Final task promotion

```text
fresh qualified artifact
→ inspect
→ isolated check
→ recorded replay
→ multiprocess shadow
→ live shadow
→ H4
→ bounded task transport
→ task capability evaluation
```

Transport success is not task success.

Codex/tooling never performs live hardware steps.

---

## 8. Focused tests

At minimum inspect/run relevant subsets:

```bash
python -m compileall -q dexmani_real examples tests
pytest -q tests/test_deployment_preflight.py
pytest -q tests/test_deployment_timing.py
pytest -q tests/test_action_buffer.py
pytest -q tests/test_coupled_command_publication.py
pytest -q tests/test_worker_command_validation.py
pytest -q tests/test_safety_gate_command_delta.py
git diff --check
```

Add focused tests near changed modules rather than relying only on broad smoke tests.

---

## 9. Long-lived documentation

### `docs/policy_deployment.md`

This is the permanent consumer/operator truth. Update:

- current main vs last hardware-evidenced revision;
- 15 predicted vs 8 executable;
- control_action execution;
- model-latency warmup vs exact online source deadline;
- shadow learned-write guarantee + hand startup caveat;
- qualified policy matrix;
- DQ-RISE EE action;
- RGB/text deferred status;
- final CLI flow.

### README

Keep concise commands and side effects.

### AGENTS

Correct stale testing guidance.

This refactor plan is temporary implementation tracking and may be deleted after completion.

---

## 10. Codex execution rules

Before every PR:

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

Each PR:

- one primary semantic variable;
- preserve unrelated user work;
- no generic abstraction without demonstrated need;
- focused tests;
- update authoritative docs;
- report environment limitations precisely.

Stop instead of guessing when:

- Policy handoff is incomplete/incompatible;
- Real main materially moved;
- pred/control semantics conflict;
- preflight/action-length contracts disagree;
- causal differential tests differ unexpectedly;
- strict artifact restore fails;
- shadow writes a learned coupled command;
- generation/ticket/worker fence tests fail;
- SafetyGate would need bypassing;
- truth requires actual hardware.

---

## 11. Recommended Real PR sequence

```text
Real PR-1  control_action semantics + preflight 15→8
    - full-output validation
    - n_action_steps execution only
    - PreflightResult / _validate_prediction consistency
    - model-specific tests

Real PR-2  timing/readiness
    - separate prefix timing from plan deadline
    - 8-step model-latency qualification
    - exact online source deadline
    - inference_mode

----- operator Promotion Gate A: replay → shadow → fresh H4 -----

Real PR-3  CLI/logging ergonomics
    - inspect/check/shadow/h4/run
    - explicit seed/profile
    - concise errors
    - quiet console

Real PR-4  causal-reader consolidation
    - IPC-neutral shared history/alignment
    - synthetic + recorded differential parity
    - replay/shadow validation

Real PR-5a valid_mask → prefix slicing
Real PR-5b deadline convergence if proven
Real PR-5c optional timing/identity field deletions
Real PR-5d EndpointBuffer experiment

Real PR-6  safety permit-read cleanup
Real PR-7  hand observe-only (hardware-qualified)
Real PR-8  RGB/text after Policy contract
```

Do not merge PR-5 semantic removals into one refactor.

---

## 12. Definition of Done

Core Real work is complete when:

- [ ] Real executes only `control_action` with `n_action_steps` endpoints.
- [ ] Preflight, warmup and receipts use the same executable length.
- [ ] Full `pred_action` is still validated before auxiliary dimensions are discarded.
- [ ] Tail never executes by default.
- [ ] Lower inference-latency prefix and upper plan/source deadline remain distinct concepts.
- [ ] Warmup is described accurately as model-latency qualification, while online source-aware deadline remains authoritative.
- [ ] Stale actions are dropped, never retimed.
- [ ] Causal state/history semantics have one shared IPC-neutral implementation.
- [ ] Run generation, atomic ticket, worker final permit, SafetyGate, collision and ACK remain enforced.
- [ ] A fresh H4 baseline exists after the 15→8 semantic change before scheduler simplification.
- [ ] CLI exposes clear inspect/check/shadow/h4/run side effects.
- [ ] Physical seed/bounds are explicit.
- [ ] Console is concise; detailed evidence remains in debug/file logs and receipts.
- [ ] Shadow proves zero learned coupled-command writes.
- [ ] Any ActionBuffer replacement has differential + shadow + fresh H4 evidence, or ActionBuffer is retained.
