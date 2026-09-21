# Codex Task — Final Research-Workflow Cleanup and Contract Convergence

## Purpose

Perform one final, bounded cleanup of `dexmani_real` so the repository is simpler, more internally consistent, and easier to use for personal PhD real-robot research without weakening real safety or experiment semantics.

This is **not** another architecture rewrite. The task is to:

1. fix two confirmed configuration/ownership problems;
2. remove dead or unreachable compatibility mechanisms with no current consumer;
3. make the current raw-v30 / xArm7+XHand replay contract explicit;
4. remove stale timing/test/migration documentation;
5. keep the remaining runtime/safety mechanisms that still have real ownership or hazard-prevention value.

The desired outcome is a **net simplification**: fewer concepts, fewer hidden configuration paths, less optionality, and documentation that describes only supported workflows.

At the time this task was authored, the reviewed repository head was:

```text
4823794a1563a3dd4e20a49508c4b25cc95b7b42
```

If the local HEAD differs, first re-audit the affected paths and preserve unrelated user changes. Do not reset or discard user work to force this task onto an older snapshot.

---

## Repository intent and priorities

Read `AGENTS.md` before editing and follow it.

This repository is a personal PhD real-robot research codebase for xArm7 + XHand + RealSense + VR/HTS data collection and learned-policy evaluation. It is not a generic robotics platform or production service.

Priority order:

```text
physical safety
> experiment correctness
> iteration speed
> readability
> generic extensibility
> enterprise robustness
```

Prefer:

```text
delete > inline > merge > rewrite
```

Add an abstraction only for a real hardware/resource boundary or demonstrated duplication whose shared owner is clear.

---

## Explicit user decisions — do not reverse

These are intentional current-state decisions, not regressions:

- `tests/` was intentionally deleted. **Do not restore it.**
- `examples/convert_raw_v29.py` completed its migration role and was intentionally deleted. **Do not restore it.**
- the previous `CODEX_TASK.md` was intentionally deleted after its task completed. **Do not restore it.**
- `assets/` is out of scope. **Do not modify assets or asset packaging.**
- do not add `--config` to `examples/run_policy.py` as part of this task.
- do not create a new test framework.
- do not run hardware-affecting examples or connect/discover devices.
- do not perform CUDA/checkpoint/hardware validation.
- do not redesign RuntimeChannels, process topology, recorder ownership, command FIFO, ACK/watermark logic, or generation semantics.

This new `codex_task.md` is explicitly user-requested. Keep it through the task. Do not delete it automatically at the end.

---

# Phase 0 — Preflight and scope control

Before editing:

1. Run:
   ```bash
   git status --short
   git rev-parse HEAD
   ```
2. Preserve unrelated local changes.
3. Read the current versions of at least:
   - `AGENTS.md`
   - `README.md`
   - `examples/calibrate_camera.py`
   - `dexmani_real/calibration/camera/session.py`
   - `dexmani_real/planning/paths.py`
   - `dexmani_real/planning/planner.py`
   - `dexmani_real/robot/arm_homing.py`
   - `dexmani_real/config/defaults.py`
   - `dexmani_real/config/experiment.py`
   - `dexmani_real/robot/commands.py`
   - `dexmani_real/replay/trajectory.py`
   - `dexmani_real/replay/session.py`
   - `dexmani_real/replay/replayer.py`
   - `dexmani_real/ipc/channels.py`
4. Trace every symbol being removed from definition -> caller -> downstream side effect before deleting it.
5. If the current source differs materially from the facts below, adapt the implementation to current truth rather than mechanically applying old line-level edits.

Do not ask for confirmation for ordinary source edits. Stop only if:
- current code contradicts a task assumption in a way that changes physical behavior;
- the only way forward would require hardware access;
- unrelated user changes create a real merge/ownership ambiguity.

---

# Phase 1 — Fix camera serial ownership

## Confirmed problem

`examples/calibrate_camera.py` correctly resolves:

```text
CLI --serial
    > YAML camera.serial
    > Python default
```

into:

```python
runtime.camera.serial
```

but then passes the original CLI value again as:

```python
camera_serial=args.serial
```

This bypasses a YAML-only serial selection when `args.serial is None`.

## Required design

There must be exactly one selection owner after config resolution:

```text
CLI / YAML / defaults
        |
resolve_experiment_config()
        |
runtime.camera.serial
        |
_start_camera(...)
        |
actual device serial returned by RealSense
        |
saved calibration provenance
```

## Required changes

- Remove the redundant `camera_serial` argument from `run_camera_calibration(...)`.
- Remove the same redundant parameter from internal calibration-session call chains if present.
- Consume `runtime.camera.serial` at the camera-start boundary.
- Preserve the **actual connected device serial** returned by RealSense for written calibration provenance.
- Keep `--serial` as a CLI override; do not remove the user-facing option.

## Acceptance

No remaining source path should pass:

```text
camera_serial=args.serial
```

after `resolve_experiment_config()`.

Do not introduce a wrapper or adapter to preserve the old internal parameter.

---

# Phase 2 — Make arm-home configuration ownership explicit

## Confirmed problem

`dexmani_real/planning/paths.py` currently imports the global canonical arm defaults and uses them to decide live home-path behavior, including:

- fallback `hand_safety_margin_m`;
- joint-limit fallback when planner is absent;
- equivalent-joint mask computation in band alignment.

This bypasses resolved runtime configuration and preserves planner-less behavior that current real home callers do not use.

## Target ownership

Use:

```text
ExperimentConfig
      |
ArmHomeConfig.from_runtime()
      |
execute_arm_home()
      |
planning/paths.py
      |
XArm7MotionPlanner / runtime-derived home parameters
```

The planning helpers must not independently read `config.defaults.arm` to decide live motion behavior.

## 2A. Tighten `ArmHomeConfig`

All real repo callers currently construct home config through:

```python
ArmHomeConfig.from_runtime(runtime, ...)
```

Use that existing owner rather than widening `execute_arm_home()` with more unrelated scalar arguments.

Required:

- add the home-planning values actually needed by `paths.py` to `ArmHomeConfig`, including:
  - `table_z_surface_m`
  - `hand_safety_margin_m`
- populate them in `ArmHomeConfig.from_runtime()` from the resolved runtime.
- remove class-field defaults that read the global `config.defaults.arm` singleton when those fields are always constructed via `from_runtime()`.
- after doing so, remove the now-unnecessary global `arm` import from `robot/arm_homing.py` if no real use remains.
- validate newly owned numeric values at the same config boundary as the rest of `ArmHomeConfig`.

Prefer required dataclass fields over maintaining a second implicit default source.

## 2B. Make home planners real requirements

`compute_joint_home_path()` and `compute_band_alignment_path()` currently accept an optional planner, but current real callers route through `arm_homing.py` and already require a planner.

Required:

- make `XArm7MotionPlanner` mandatory for these home planning functions;
- remove planner-less joint-limit wrapping fallback;
- remove `planner is not None` branches that become unreachable;
- remove `hasattr(planner, "plan_joint_qpos_path")` if the concrete planner API already guarantees that method;
- use planner-owned kinematic facts for equivalent-joint logic:
  - planner / IK-manager joint limits;
  - planner / IK-manager equivalent-joint mask;
  - `nearest_equivalent_qpos()` / `compute_qpos_delta()` as appropriate.
- pass `ArmHomeConfig.table_z_surface_m` and `ArmHomeConfig.hand_safety_margin_m` into the pure path checks.
- remove the `config.defaults.arm as _arm_cfg` dependency from `planning/paths.py` when no remaining behavior needs it.

Do not create a new planning-context/config class.

## 2C. Preserve planner diagnostic cross-check

Do **not** delete the existing low-cost check in `planning/planner.py` that compares the MPlib/URDF joint limits against the canonical Python hardware limits.

That check is diagnostic, not live command authority. The planner must continue to use the URDF/MPlib limits for kinematic authority.

If useful, clarify the comment to make the ownership explicit:

```text
diagnose accidental drift between canonical Python hardware limits and the URDF;
planner authority remains the URDF/MPlib model
```

Do not convert this warning into a new framework or runtime config coupling.

## 2D. Remove dead planner-unavailable home state only if truly unreachable

After making planner mandatory, re-trace `ArmHomeStatus.PLANNER_UNAVAILABLE`.

Delete it and its branch only if there is no remaining public/internal caller that can reach `execute_arm_home()` without a planner.

Do not delete the status merely because the new type annotation is non-optional; confirm runtime callers first.

---

# Phase 3 — Simplify dispatch-delay configuration without removing expiry

## Required semantic decision

Keep the command-expiry mechanism.

Keep:

- `max_dispatch_delay_s`;
- `expires_monotonic_ns`;
- immutable command deadlines;
- worker-side final deadline/admission fence;
- generation revocation on command expiry;
- `COMMAND_EXPIRED` semantics;
- existing strict ordered coupled FIFO / ACK behavior.

The current runtime still needs a stale-backlog time bound. Do not replace the scheduler in this task.

## Simplify the config type

`max_dispatch_delay_s=None` is not a real supported disabled mode: current hardware-affecting entry points require a finite positive dispatch budget before startup.

Therefore:

- change `SafetyParams.max_dispatch_delay_s` from `float | None` to `float`;
- keep default `0.1`;
- validate it unconditionally in `SafetyParams.validate()`;
- preserve finite / positive / representable-nanosecond checks;
- make YAML `null` fail during normal config resolution, not later at a hardware entry point.

Use a short semantic comment only, for example:

```python
# Maximum lateness allowed after a command becomes eligible.
max_dispatch_delay_s: float = 0.1
```

Do not justify 0.1 via a particular fixed-pose measurement.

## Explicitly delete stale timing narrative

Remove repository documentation/comments that depend on:

- `<36 ms`;
- the removed `examples/measure_dispatch_timing.py`;
- fixed-pose timing as proof of the 100 ms default;
- wording that says this default is pending a specific timing-tool commissioning pass.

The default is a runtime stale-command budget, not a benchmark result.

Do not restore the timing script.

---

# Phase 4 — Remove the unused SafetyGate hand-delta mechanism

## Confirmed current state

Repo-wide caller tracing found no runtime caller that enables `max_hand_delta_rad` with a non-`None` value. Policy currently passes it explicitly as `None`.

The associated endpoint tolerance exists only for this inactive branch.

## Required deletion

After one final caller check, remove the complete unused mechanism:

- `SafetyGate.max_hand_delta_rad`;
- `SafetyGate.endpoint_delta_tolerance_rad`;
- `SafetyGate._coerce_delta()` if unused after removal;
- `_joint_delta_limit_detail()`;
- `GateRejectCode.HAND_DELTA_LIMIT`;
- `hand_delta_reference_qpos` from command preparation / gate validation if it has no remaining independent use;
- `max_hand_delta_rad` and `endpoint_delta_tolerance_rad` parameters from `planner_action_safety_gate()`;
- explicit `max_hand_delta_rad=None` and endpoint-tolerance arguments in deployment;
- `PolicyParams.endpoint_delta_tolerance_rad` and its validation if it has no other consumer;
- the `policy_defaults` import in `robot/commands.py` if that import becomes unused.

Important: **do not remove** `_JOINT_LIMIT_TOLERANCE_RAD` if it remains used by hand joint-limit validation. The joint-limit tolerance and the dead hand-delta tolerance are different mechanisms.

Preserve the active SafetyGate responsibilities:

```text
finite/shape
joint bounds
workspace transition
optional collision transition
```

Do not move hand-delta shaping into another layer as part of this task.

---

# Phase 5 — Make physical replay explicitly raw-v30 + xArm7/XHand

## Confirmed storage contract

`EpisodeReader` currently:

1. requires the current schema version;
2. calls `require_valid()`;
3. validates every dataset in `DATASET_SPECS` for presence, shape, and dtype.

Therefore replay code does not need a second layer of “dataset may be absent” compatibility for current raw v30.

## Required replay simplification

Tighten `TrajectoryData` and replay loading to the only supported physical replay contract.

After confirming current callers:

- make current required trajectory arrays non-optional where the raw-v30 reader already guarantees them, including:
  - arm sent targets;
  - hand targets;
  - arm measured qpos;
  - hand measured qpos;
  - send mask;
  - computed arm EEF history.
- remove `if key in h5 else None` branches for mandatory v30 datasets.
- remove redundant “required dataset exists” checks already guaranteed by `EpisodeReader.require_valid()`.
- use the validated metadata frame count directly rather than recomputing a minimum across required dataset lengths when the reader already guarantees equal length.
- remove `action_source`, `has_hand`, `has_hand_actions`, `require_hand_actions()`, or equivalent compatibility helpers only if final caller tracing confirms they become redundant.
- simplify physical replay worker startup so XHand is not treated as optional after preflight if current preflight and runtime requirements already guarantee hand data and `runtime.policy.hand_enabled=true`.
- remove dead `send_mask is None` branches.
- update `examples/replay_episode.py` wording from “xArm7 and optional XHand” to the actual supported physical replay contract.

Do not change persisted raw schema semantics.
Do not reinterpret old episode data.
Do not add runtime compatibility for v29 or older formats.

---

# Phase 6 — Remove confirmed dead helpers and only low-risk duplication

## Delete confirmed dead APIs

After one final repo-wide search, delete without compatibility wrappers:

- `canonicalize_policy_hand_endpoint_roundoff()`;
- `POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD` if only used by that helper;
- `required_dataset_names()`.

Git history is the archive. Do not add deprecation aliases.

## Quaternion multiply

It is acceptable to replace the private duplicate quaternion multiply in:

```text
teleop/control_loop/action_proposal.py
```

with the existing public/local project primitive in:

```text
planning/kinematics/pose.py
```

only if the ordering/normalization semantics are exactly equivalent at the call site.

Do **not** also merge quaternion normalization helpers if their failure behavior differs.

This cleanup is optional if it would create a wider diff than the saved code justifies.

## Keep local startup helpers local

Do **not** extract the duplicated `read_initial_arm()` startup loops into `ipc/channels.py` or a new service/helper module.

The duplication is small; moving timeout/retry/health policy into IPC would increase cross-layer responsibility.

Keep those local unless a smaller same-module merge naturally appears during editing.

## Keep tiny local validators local

Do not create shared utility abstractions merely to deduplicate:

- `_finite_vector()`;
- CLI `_positive_float()`.

---

# Phase 7 — Mechanical workspace-bound cleanup

`WorkspaceBounds.as_array()` already returns a mutable float64 `(3, 2)` copy.

Where semantically identical, replace manual forms such as:

```python
np.asarray(runtime.policy.workspace.as_tuple(), dtype=np.float64)
```

or hand-built `[[x_min, x_max], ...]` arrays with:

```python
runtime.policy.workspace.as_array()
```

Candidate paths include current uses in teleop, deployment, calibration, replay, and homing.

Do this only where the array semantics are identical. Do not introduce a workspace manager/helper.

---

# Phase 8 — Documentation truth pass

Documentation must describe current supported workflows, not the history of how the refactor was validated.

## README.md

Remove stale references to:

- `examples/measure_dispatch_timing.py`;
- `<36 ms` fixed-pose timing evidence;
- fixed-pose timing workflow and flags;
- `examples/convert_raw_v29.py`;
- v29 conversion instructions;
- `tests/`;
- `pytest` commands;
- wording that camera calibration is only for an absent XHand.

Keep camera calibration concise and accurate:

```text
absent       = no XHand mounted
secured-home = mounted XHand physically secured at configured home
```

Both current calibration assertions use the conservative fixed-home XHand collision envelope; preserve that truth if still current.

Keep README focused on:

- repository purpose;
- install/environment;
- configuration precedence;
- actual research entry points;
- collection/recording;
- policy rollout;
- raw-v30 -> policy-Zarr data flow;
- concise safety/ownership facts useful to an operator;
- low-cost offline checks that actually exist.

Do not turn README into an internal concurrency design document.

## AGENTS.md

The repository intentionally has no committed `tests/` directory.

Update the check/handoff guidance so future agents are not instructed to rely on or restore a deleted test suite.

Keep guidance such as:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If Ruff is unavailable locally, report that clearly rather than installing/upgrading the experiment environment without permission.

State that focused one-off offline smoke checks for changed pure logic are appropriate, but do not introduce a new committed test framework unless explicitly requested.

Never use a hardware example as a test.

## IPC docstring

Correct the stale `read_hand_state_dict()` docstring so it names only fields actually returned.

Do not claim board telemetry or an accepted command sequence if the returned dict does not contain them.

---

# Phase 9 — Verification

This task intentionally does not restore the deleted test suite.

## 9A. Required low-cost checks

Run on the exact final working tree:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

If a command is unavailable because the local environment lacks the tool/dependency, report it as not run/blocked. Do not modify the global experiment environment merely to satisfy the check.

## 9B. Required targeted offline smoke checks

Without importing or invoking live SDK behavior, verify at least:

1. config precedence:
   - YAML `camera.serial` resolves into `runtime.camera.serial`;
   - CLI serial override wins over YAML.
2. dispatch delay:
   - default `0.1` resolves to `100_000_000` ns;
   - `max_dispatch_delay_s: null` is rejected by normal config resolution.
3. replay strictness:
   - current loader assumes/uses the validated v30 required fields directly;
   - no fallback to older/missing hand/send-mask datasets remains.
4. home path imports:
   - `planning/paths.py` no longer reads `config.defaults.arm` to decide live path behavior.

Use the smallest pure-Python snippets possible. Do not connect hardware.

## 9C. Residual source audit

The following should be absent from active source/docs after completion unless a final caller trace demonstrates a justified remaining use:

```text
camera_serial=args.serial
measure_dispatch_timing
convert_raw_v29
<36 ms
max_hand_delta_rad
endpoint_delta_tolerance_rad
HAND_DELTA_LIMIT
canonicalize_policy_hand_endpoint_roundoff
POLICY_HAND_ENDPOINT_ROUNDOFF_TOLERANCE_RAD
required_dataset_names
xArm7 and optional XHand
```

`planning/paths.py` should not retain:

```text
config.defaults import arm as _arm_cfg
```

README/AGENTS should not instruct users to run:

```text
python -m pytest
... examples tests
```

The following must **remain** because they are intentionally preserved:

```text
max_dispatch_delay_s
expires_monotonic_ns
run_generation
coupled FIFO / per-consumer ACK semantics
recorder-process finalization ownership
URDF-vs-canonical joint-limit diagnostic in planner.py
```

---

# Phase 10 — Final diff review

Before finishing:

1. inspect `git diff --stat`;
2. inspect the full diff;
3. check for dead imports and stale comments;
4. ensure the change is net-simple rather than framework-expanding;
5. ensure no assets changed;
6. ensure no tests were restored;
7. ensure no hardware-affecting command was run;
8. ensure no persisted raw field meaning changed;
9. ensure no public `dexmani_policy` interface was modified or bypassed.

Do not create commits, branches, or push unless the user separately asks for Git operations.

---

# Explicit anti-goals

Do not:

- redesign process supervision;
- replace multiprocessing/SHM/FIFO infrastructure;
- replace the coupled FIFO with a new scheduler;
- remove command expiry;
- remove generation fencing;
- remove final SDK-admission checks;
- remove arm/hand worker ownership;
- inline recorder I/O into a control process;
- create manager/factory/backend/registry abstractions;
- productionize wheel/assets packaging;
- restore v29 runtime support;
- restore `tests/`;
- modify `assets/`;
- add a new monitoring framework;
- add a new configuration layer;
- make `run_policy.py` accept YAML as part of this task;
- run hardware, CUDA, or real checkpoints.

---

# Completion report

When implementation is complete, report concisely:

1. **Correctness fixes**
   - camera serial ownership;
   - arm-home runtime-config ownership.
2. **Deleted mechanisms**
   - exact dead branches/helpers/config fields removed.
3. **Contract tightening**
   - replay/raw-v30/XHand simplifications.
4. **Kept intentionally**
   - command expiry;
   - generation/FIFO/ACK;
   - recorder ownership;
   - planner URDF drift diagnostic.
5. **Documentation**
   - README/AGENTS/docstring changes.
6. **Verification**
   - each command/smoke check: pass / fail / not run and why.
7. **Hardware**
   - explicitly state: NOT RUN.
8. **Remaining limitations**
   - only concrete unresolved issues, not speculative future architecture work.

The task is complete when current supported behavior is simpler and more explicit, the two confirmed ownership bugs are fixed, dead compatibility code is removed, documentation matches the tree, and no new framework has been introduced.
