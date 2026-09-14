# CODEX_TASK — DexMani Real Minimal Fixes

> Temporary implementation task for local Codex. After the requested code changes are complete and verified, delete this file in the same change set so the repository does not retain a one-off implementation plan.

## Goal

Apply three minimal fixes without expanding architecture or introducing production-oriented infrastructure:

1. Fix the `scheduled_target_monotonic_ns` future-time contract conflict.
2. Remove the ambiguous `meta.success = save` recording metadata.
3. Persist the actual inference `device` in `run_config.yaml`.

The project is a personal research codebase. Prefer the smallest coherent code change that preserves existing ownership and behavior.

---

## Repository constraints

Before editing:

```bash
git status --short
```

Preserve unrelated user changes.

Do **not** run hardware-affecting programs. Do not connect to xArm/XHand/cameras, home the robot, run teleoperation, replay, or policy rollout.

Do not weaken lifecycle, generation, freshness, command-validity, worker, or safety boundaries.

Do not add new scheduler abstractions, outcome evaluators, schema migrations, GPU environment snapshots, telemetry systems, or test frameworks.

Expected source files to modify:

```text
dexmani_real/control/publication.py
dexmani_real/recording/recorder.py
examples/run_policy.py
```

No IPC dtype or `ActionCandidate` field changes are expected.

---

# Task 1 — Fix `scheduled_target_monotonic_ns` future-time contract

## Problem

`ActionCandidate` already defines three distinct timing concepts:

```text
scheduled_target_monotonic_ns
    policy-grid logical endpoint / provenance

target_monotonic_ns
    worker delivery target

valid_until_monotonic_ns
    hard command delivery expiry
```

`PolicyExecutor` intentionally selects an upcoming action during the control interval preceding its logical target, so a valid policy command must allow:

```text
scheduled_target_monotonic_ns > now_ns
```

However, `build_action_candidate()` currently rejects that case with:

```python
if not 0 < scheduled_ns <= now_ns:
    return None
```

This conflicts with the Executor's future-target scheduling and can cause valid policy actions to be discarded before publication.

## Required change

In `dexmani_real/control/publication.py`, change only the structural validation of `scheduled_ns`:

```diff
- if not 0 < scheduled_ns <= now_ns:
+ if scheduled_ns <= 0:
      return None
```

Add a concise nearby comment clarifying that `scheduled_target_monotonic_ns` is logical/provenance time and may be before or after candidate creation time.

## Required ownership after the fix

Keep stale-policy semantics in `PolicyExecutor`:

```text
Prediction arrival
    -> first_future_step_index() skips stale prefix

Due action
    -> decode / IK / safety
    -> final check that now < scheduled_target_monotonic_ns
    -> publish
```

Do not move policy stale checks into generic publication code.

## Explicitly do not change

Keep all of the following unchanged:

```text
target_monotonic_ns = now_ns
valid_until_monotonic_ns computation
minimum_delivery_window_s behavior
generation fencing
latest-wins command ownership
coupled_command_ticket_allows_execution()
arm worker scheduling
hand worker scheduling
```

Do **not** set:

```python
target_monotonic_ns = scheduled_target_monotonic_ns
```

The current runtime has one scheduling owner: `PolicyExecutor`. Turning the worker delivery target into a future target would introduce a second scheduling layer.

## Safety expectation

This change must not weaken physical execution authority. Worker execution should still require:

```text
active motion state
matching run_generation
latest/current command ticket
valid_until_monotonic_ns not expired
no fault / no e-stop
```

---

# Task 2 — Remove ambiguous `meta.success = save`

## Problem

`dexmani_real/recording/recorder.py` currently writes:

```python
meta.attrs["success"] = save
```

This value describes recording transaction intent/state, not manipulation task success.

The policy deployment contract already states that task success is judged offline from the published raw episode. Keeping a metadata field named `success` creates a high-risk semantic trap for later evaluation scripts.

## Required change

Remove the metadata assignment:

```diff
 meta.attrs["duration"] = duration
 meta.attrs["wall_duration_s"] = duration
 meta.attrs["num_frames"] = self._frame_count
- meta.attrs["success"] = save
 meta.attrs["fps"] = self.control_hz
```

Add a short comment near final metadata writing:

```python
# Task success is intentionally judged offline from the published episode.
```

## Do not replace it with another field

Do not add any of the following:

```text
episode_saved
task_success
task_success_valid
success_unknown
```

Reason: a formally published episode directory is already evidence that the recorder transaction succeeded. `save=False` does not publish the episode payload; it produces an aborted manifest instead. A second `episode_saved=True` attribute would be redundant.

Task outcome remains an offline research/evaluation concern.

## Schema handling

Do **not** bump `EPISODE_SCHEMA_VERSION` for this change.

Current structural validation does not require `meta.success`, and repository readers/dataset processing do not consume it. A schema bump would unnecessarily invalidate existing v29 episodes and create migration work.

Do not modify old episode files.

---

# Task 3 — Persist inference `device` in `run_config.yaml`

## Problem

`examples/run_policy.py` accepts:

```text
--device
```

and passes `args.device` directly into `InferenceWorkerConfig.device`, but `_write_run_config()` does not persist it.

This leaves one explicit experimental condition missing from the session record, which matters especially for inference-latency/NFE comparisons.

## Required change

Extend `_write_run_config()` with a `device: str` keyword argument.

Expected signature shape:

```python
def _write_run_config(
    session_dir: Path,
    *,
    experiment: str,
    artifact: str,
    inference_steps: int,
    seed: int,
    device: str,
    num_episodes: int,
    max_duration_s: float,
) -> None:
```

Add the exact string to the YAML payload:

```python
payload = {
    "experiment": experiment,
    "artifact": artifact,
    "inference_steps": inference_steps,
    "seed": seed,
    "device": device,
    "num_episodes": num_episodes,
    "max_duration_s": float(max_duration_s),
}
```

Update the caller to pass:

```python
device=args.device
```

The persisted value must be the same string that is passed to `InferenceWorkerConfig.device`.

## Explicitly do not add

Do not inspect or persist:

```text
GPU model
GPU UUID
CUDA driver
Torch/CUDA versions
hostname
Git SHA
checkpoint SHA256
```

Do not import Torch or initialize CUDA in the parent process for metadata collection. The parent should remain lightweight; model/CUDA ownership stays in the inference child.

---

# Focused verification

Use offline checks only.

## A. Future target contract

Perform a focused check of `build_action_candidate()` using deterministic timestamps equivalent to:

```text
now_ns = 1_000_000_000
scheduled_target_monotonic_ns = 1_062_500_000
```

Verify:

```text
candidate is not None
candidate.scheduled_target_monotonic_ns == 1_062_500_000
candidate.target_monotonic_ns == 1_000_000_000
```

Also verify `scheduled_target_monotonic_ns <= 0` remains rejected.

Do not create a new test framework solely for this task. If a focused existing test location exists, use it; otherwise a small offline Python check is sufficient.

## B. Recorder metadata

Verify new finalized metadata no longer includes:

```text
success
```

and still includes the existing technical metadata such as:

```text
stop_reason
schema_version
num_frames
```

Confirm structural episode validation logic is unchanged.

## C. Run config

Call `_write_run_config()` with a temporary directory and verify the resulting YAML contains, for example:

```yaml
device: cuda:0
```

Confirm the value is unchanged from the input string.

## D. Repository checks

Run:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Inspect the focused diff before handoff.

Do not claim hardware validation.

---

# Acceptance criteria

- [ ] A future positive `scheduled_target_monotonic_ns` can be represented by `ActionCandidate`.
- [ ] Non-positive scheduled target timestamps remain rejected.
- [ ] `PolicyExecutor` remains the owner of stale/future policy timing semantics.
- [ ] `target_monotonic_ns`, delivery expiry, generation fencing, latest-ticket ownership, and worker safety checks are unchanged.
- [ ] Newly published episodes no longer write `meta.success`.
- [ ] No replacement task-success/save metadata field is introduced.
- [ ] `EPISODE_SCHEMA_VERSION` remains unchanged.
- [ ] `run_config.yaml` records the exact inference `device` string.
- [ ] Parent process behavior does not add Torch/CUDA introspection.
- [ ] No new scheduler, evaluator, telemetry, migration, or generic testing infrastructure is introduced.
- [ ] Offline compile/diff checks pass.
- [ ] No hardware-affecting validation is performed.
- [ ] This temporary `CODEX_TASK.md` is deleted after implementation and verification.

---

# Expected final handoff from Codex

Report only:

1. files changed and the exact contract change in each;
2. focused offline checks run and their results;
3. anything not verified;
4. confirmation that no hardware was exercised;
5. confirmation that `CODEX_TASK.md` was removed as final cleanup.
