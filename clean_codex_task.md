# Codex Task — Final Style-Only Cleanup for DexMani Real

## Baseline

Fact-checked against:

~~~text
main = ea8a4d7f5e1bcd8feba8919eb9ec32b08940c0e4
~~~

If main moved, inspect current source before editing and preserve this task's intent. Do not restore findings that newer code already fixed.

This is a **style-only final polish** covering only P0, P1, and P2.

Desired result:

- consistent Python formatting;
- shorter, factual source prose;
- fewer redundant comments/docstrings;
- simpler example imports;
- no runtime redesign;
- no behavioral change.

Target style:

> ManiUniCon-like prose restraint + pi-r2-flow-like implementation directness + DexMani's existing robotics/geometry rigor.

## 1. Hard scope boundary

Allowed:

1. formatting and import layout;
2. comment/docstring cleanup;
3. deletion of comments/docstrings that merely restate code;
4. concise package/module prose;
5. minimal Ruff configuration;
6. removal of repository-root sys.path injection from examples that already rely on editable installation;
7. trivial type-comment cleanup when behavior is unchanged;
8. a concise source-style rule in AGENTS.md.

Forbidden:

- algorithm changes;
- robot-control or safety changes;
- motion-authority or homing changes;
- collision, IK/FK, retargeting or observation-semantic changes;
- worker/process/thread topology changes;
- IPC, raw episode or policy-Zarr schema changes;
- config-default or research-hyperparameter changes;
- hardware SDK call-order changes;
- recording-transaction changes;
- public import-path changes;
- file/directory moves;
- module split/merge;
- new wrappers, protocols, base classes, validators or abstractions;
- Pydantic, mypy, pre-commit, complex CI or a committed tests directory;
- full-project type annotation or docstring standardization;
- renaming meaningful robotics variables just to shorten lines;
- broad print-to-logger conversion;
- adjacent cleanup unrelated to style.

The arbitrary point-count whitelist has already been removed. Do not reintroduce or re-clean it.

## 2. P0 — mechanical formatting

### 2.1 Minimal Ruff configuration

Add to pyproject.toml:

~~~toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
~~~

Keep the configuration intentionally small. Do not enable broad rule families such as ANN, D, SIM, TRY, PLR, C90, ARG or PERF.

Do not use linting to redesign research code.

### 2.2 Format source

When Ruff is available:

~~~bash
ruff format dexmani_real examples
ruff check --select I --fix dexmani_real examples
~~~

If Ruff is unavailable, do not install or upgrade the experiment environment. Manually format only the affected files and report that Ruff was unavailable.

Highest-priority formatting targets:

~~~text
dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py
dexmani_real/runtime/observation.py
dexmani_real/runtime/safety.py
dexmani_real/replay/session.py
dexmani_real/recording/frame.py
dexmani_real/teleop/control_loop/grid.py
dexmani_real/robot/arm_homing.py
dexmani_real/replay/replayer.py
dexmani_real/robot/arm_worker.py
dexmani_real/robot/hand_worker.py
dexmani_real/recording/client.py
dexmani_real/sensor/camera/worker.py
dexmani_real/teleop/retargeting/dexpilot.py
dexmani_real/teleop/retargeting/tag_optimizer.py
dexmani_real/teleop/retargeting/pin_grad.py
~~~

Do not compress independent operations onto one line merely to reduce LOC.

Preserve descriptive robotics names. Wrap long expressions instead of abbreviating names such as handbase_quat_eef_wxyz, observation_timestamp_ns or source_camera_sequence.

## 3. P1 — source prose cleanup

The repository is not globally over-commented. Do not perform blanket comment deletion.

### 3.1 Preserve high-value comments

Keep comments/docstrings that explain non-obvious:

- mathematics;
- FK/IK assumptions;
- coordinate frames and transforms;
- quaternion conventions;
- units;
- numerical approximations;
- vendor SDK or firmware behavior;
- hardware limitations;
- sensor semantics;
- joint/finger ordering;
- experimental thresholds;
- real safety-critical reasons;
- external algorithm/source attribution.

Specifically preserve the substance of:

- XHand SDK vs Pinocchio joint ordering in planning/kinematics/hand_fk.py;
- FreeFlyer/Jacobian convention in teleop/retargeting/pin_grad.py;
- SO(3)/quaternion smoothing rationale in teleop/control_loop/action_proposal.py;
- equivalent-angle and staged-homing rationale in planning/paths.py;
- depth-edge/same-surface/filtering rationale in sensor/pointcloud.py;
- RealSense alignment/timestamp/firmware semantics in sensor/camera/realsense.py;
- manipulability/null-space/branch-continuity reasoning in planning/kinematics/ik.py;
- xArm firmware EEF vs URDF EEF distinction in planning/kinematics/arm_fk.py.

Long comments are acceptable when they contain real scientific or hardware information.

### 3.2 Reduce architecture-governance prose

Prefer direct descriptions of what code does.

Review prose centered on words such as:

~~~text
owns
ownership
boundary
contract
single source of truth
validated
canonical
transactional
provenance
explicit
fail-closed
~~~

Do not delete these mechanically. Keep them when they describe a real persisted-data convention, physical-safety/SDK boundary or scientifically meaningful provenance.

Remove or simplify them when they only narrate software governance.

Priority examples:

- deployment/__init__.py: reduce the package manifesto to a one-line package description.
- ipc/schema.py: remove "single source of truth" prose; describe the schema directly.
- ipc/channels.py: simplify "centralized data plane", ownership and Usage prose.
- dataset/pointcloud.py: replace persisted-boundary / does-not-own prose with a direct point-cloud description.
- recording/io_worker.py: remove repeated "Policy owns / ownership-copies / then owns" narration while keeping operational facts.

### 3.3 Remove development-history prose

Source should describe current behavior, not implementation history.

Clean historical wording such as:

~~~text
no longer
existing implementation
previous design
restores
dropped
upstream's
without touching site-packages
retained for parity
~~~

when it describes development history rather than current semantics.

Examples:

- planning/kinematics/arm_fk.py: replace "the arm worker no longer computes EEF" with a statement of how EEF is computed now.
- teleop/retargeting/dexpilot.py: keep the current human-flexion prior and its mathematics, but sharply reduce dex-retargeting version history, commented upstream parameters, site-packages discussion, "vanilla optimizer" wording and parity/fallback archaeology.

Do not change the DexPilot algorithm while editing prose.

### 3.4 Preserve concise attribution

Keep concise attribution such as:

~~~text
Adapted from TAG/Retargeting/Hand_Retargeting/utils/pin_grad.py.
~~~

Prefer current algorithm + concise attribution over upstream version history + patch history.

### 3.5 Shorten README-like internal module docstrings

Highest-priority target:

~~~text
dexmani_real/calibration/camera/session.py
~~~

Its module docstring currently includes ownership, algorithm overview, output path, hardware preparation, environment setup, CLI usage, controls, XHand physical-state explanation and collision semantics.

Reduce the internal module docstring to roughly 1–3 concise sentences. Leave user workflow/hardware instructions to README, CLI help or the example entry point.

Other high-value cleanup targets:

~~~text
dexmani_real/recording/io_worker.py
dexmani_real/calibration/camera/solver.py
dexmani_real/sensor/camera/geometry.py
dexmani_real/dataset/pointcloud.py
dexmani_real/ipc/channels.py
dexmani_real/ipc/schema.py
dexmani_real/robot/drivers/xhand.py
dexmani_real/robot/model.py
dexmani_real/planning/kinematics/arm_fk.py
dexmani_real/utils/rate.py
~~~

### 3.6 Simplify package docstrings

Package __init__.py files should normally be export-only, empty, or have a one-line description.

Priority targets:

~~~text
dexmani_real/__init__.py
dexmani_real/deployment/__init__.py
dexmani_real/recording/__init__.py
~~~

Remove subsystem inventories and architecture manifestos. Do not change exported names.

### 3.7 Remove restatement/decorative comments

Delete comments that only restate the next line or function name.

Remove decorative section headings such as long dashed "Atomic file finalisation" comments.

Compress long control-flow explanations when one factual line is sufficient, for example:

~~~text
Preserve failed partial episodes for offline diagnosis.
~~~

Do not delete comments that explain why unusual behavior is necessary.

### 3.8 Private helper docstrings

Private helpers normally do not need docstrings when the name and implementation are clear.

Keep them only when they explain non-obvious math, frames/units, SDK/hardware caveats, safety reasoning or subtle behavior.

Do not mechanically delete all private docstrings.

### 3.9 Avoid duplicate dynamic facts

Do not duplicate mutable code facts such as schema version numbers in prose when a code constant already defines them.

## 4. P1 — AGENTS.md source-style guardrail

Add a concise Code style section to AGENTS.md without weakening existing hardware/safety/research rules.

It should communicate:

- Keep research code direct and readable.
- Do not compress independent operations onto one line merely to reduce line count.
- Comments explain non-obvious robotics, math, coordinate frames, units, vendor SDK behavior, experimental thresholds, attribution or real safety-critical reasons.
- Do not narrate architecture, ownership, boundaries, contracts, validation or obvious control flow when code already makes them clear.
- Describe current behavior rather than implementation history.
- Private helpers normally do not need docstrings.
- Run Ruff formatting and import sorting on touched Python files when Ruff is available.
- Do not introduce abstractions solely to satisfy style tooling.

Keep this section short.

## 5. P2 — small packaging and annotation cleanup

P2 is subordinate to P0/P1. Do not let it expand scope.

### 5.1 Remove repository-root sys.path injection

At the fact-checked baseline, repository-root sys.path.insert exists in these 9 examples:

~~~text
examples/keyboard_teleop.py
examples/calibrate_camera.py
examples/collect_teleop.py
examples/replay_episode.py
examples/export_policy_zarr.py
examples/xhand_diagnostics.py
examples/calibrate_vr_heading.py
examples/visualize_episode.py
examples/pointcloud_process_example.py
~~~

examples/realsense_record_example.py no longer contains this hack; do not add it.

README already documents editable installation:

~~~bash
python -m pip install -e .
~~~

Remove only the path-injection boilerplate and imports that become unused solely because of its removal. Preserve independent uses of sys or pathlib.Path.

Do not change example CLI behavior otherwise.

### 5.2 Clean awkward double-comment type ignores conservatively

Where a line contains both a type-ignore and a long prose justification, prefer a local assert, cast or clearer control flow only when the edit is trivial and runtime behavior remains identical.

Do not chase all type ignores.

Keep reasonable external-package ignores such as type: ignore[import-untyped].

### 5.3 Trim mechanical Args/Returns prose selectively

Do not standardize Google vs NumPy docstrings.

Remove parameter prose that merely repeats a parameter name, type annotation or obvious shape. Keep explanations of algorithmic relationships, non-obvious shapes, units, coordinate frames and semantics.

Be conservative in:

~~~text
teleop/retargeting/tag_optimizer.py
teleop/retargeting/pin_grad.py
teleop/control_loop/action_proposal.py
planning/paths.py
~~~

Scientific explanation has higher priority than brevity.

## 6. Files that must not be structurally refactored

Do not split these merely because they are long:

~~~text
dexmani_real/config/defaults.py
dexmani_real/sensor/pointcloud.py
dexmani_real/planning/kinematics/ik.py
dexmani_real/recording/recorder.py
dexmani_real/sensor/camera/realsense.py
~~~

Do not merge small modules merely to reduce file count.

Do not move examples into tools/ in this task.

## 7. Execution order

1. Read AGENTS.md.
2. Inspect git status --short and preserve unrelated changes.
3. Confirm current main and reconcile only if it moved.
4. Add minimal Ruff config.
5. Run Ruff formatter/import sorting if available.
6. Inspect formatter diff before manual edits.
7. Perform P1 module/package prose cleanup.
8. Remove P1 restatement/development-history prose.
9. Add the short AGENTS.md Code style section.
10. Perform the P2 sys.path cleanup.
11. Perform only obvious P2 annotation/docstring cleanup.
12. Run low-cost validation.
13. Inspect the full final diff for accidental semantic changes.

Do not intermingle architecture redesign with this pass.

## 8. Validation

No hardware execution.

Run:

~~~bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
~~~

If Ruff is available, also run:

~~~bash
ruff format --check dexmani_real examples
ruff check --select I dexmani_real examples
~~~

If Ruff is unavailable:

- do not install it;
- report that limitation;
- still run compileall and git diff --check.

Useful final searches:

~~~bash
git grep -n "single source of truth" -- dexmani_real
git grep -n "no longer" -- dexmani_real
git grep -n "site-packages" -- dexmani_real
git grep -n "sys.path.insert" -- examples
~~~

Search hits are not an acceptance metric. Review context; some words may legitimately remain.

## 9. Final diff review

Before finishing, confirm:

- no algorithm changed;
- no config default or schema changed;
- no hardware/safety behavior changed;
- no public API/import path changed;
- no file moved;
- no process/thread/worker added or removed;
- no validation/safety mechanism removed merely because it looked verbose;
- scientific/math/hardware comments remain;
- source prose is shorter and more timeless;
- imports and formatting are consistent;
- the 9 listed path-injection hacks are gone;
- package docstrings are concise;
- AGENTS.md contains the source-style guardrail.

Inspect:

~~~bash
git diff --stat
git diff
~~~

The diff should be dominated by line wrapping, import organization, prose deletion/rewording, small example import cleanup, Ruff config and AGENTS style guidance.

If program structure starts changing, revert those changes.

## 10. Acceptance criteria

Complete only when:

1. P0 formatting drift is removed without semantic changes.
2. Long lines are wrapped instead of shortening meaningful robotics names.
3. Architecture/governance prose is substantially reduced in the identified P1 files.
4. Development-history prose is replaced by descriptions of current behavior.
5. Math, geometry, hardware, SDK, attribution and experimental-rationale comments remain.
6. README-like internal module docstrings are shortened.
7. Package docstrings are minimal.
8. AGENTS.md contains concise source-style guidance.
9. The 9 current repository-root sys.path.insert hacks are removed.
10. Type-ignore/docstring cleanup stays conservative and behavior-neutral.
11. No P3 work is performed.
12. Offline checks pass, or unavailable Ruff tooling is explicitly reported.
13. No hardware code is executed.

## 11. Handoff

Report concisely:

1. files changed;
2. P0 formatting changes;
3. P1 prose/comment cleanup;
4. P2 packaging/annotation cleanup;
5. validation commands and results;
6. Ruff availability;
7. confirmation that no runtime/algorithm/schema/safety behavior was intentionally changed.

Do not claim hardware validation from this style-only task.
