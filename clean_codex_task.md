# Codex Task — Final Style-Only Cleanup for DexMani Real

## 0. Baseline and goal

Production code was fact-checked at:

~~~text
ea8a4d7f5e1bcd8feba8919eb9ec32b08940c0e4
~~~

This task document was added afterward and does not itself change production code.

If main moved, inspect the new diff first. Reconcile only real source changes; do not revive stale findings.

This pass covers **P0 + P1 + P2 only**. The intended result is a boring final-polish diff:

- mechanical Python formatting;
- concise, factual source prose;
- removal of redundant architecture/history narration;
- removal of the known example path-bootstrap hacks;
- no robot/runtime redesign.

Target style:

> ManiUniCon-like prose restraint + pi-r2-flow-like implementation directness + DexMani's existing robotics/geometry rigor.

## 1. Non-negotiable scope

### Allowed

- Python formatting and import organization;
- concise comment/docstring edits;
- deletion of comments that restate code;
- minimal Ruff configuration;
- a short source-style guardrail in AGENTS.md;
- removal of the 9 known repository-root sys.path injections from examples;
- deletion of imports made unused by that removal.

### Forbidden

Do not change:

- algorithms or numerical formulas;
- robot-control, safety, motion-authority or homing behavior;
- collision, FK/IK, retargeting or observation semantics;
- SDK call order;
- worker/process/thread topology;
- IPC/runtime protocols;
- raw or Zarr schemas and persisted meanings;
- config defaults or research hyperparameters;
- recorder transaction behavior;
- public module/import paths;
- repository layout.

Do not:

- split, merge, move or rename modules;
- add wrappers, protocols, base classes, validators or abstraction layers;
- add Pydantic, mypy, pre-commit, complex CI or a tests directory;
- standardize all typing or docstring formats;
- shorten meaningful robotics names to satisfy line length;
- broadly replace print with logging;
- perform adjacent cleanup.

The arbitrary point-count whitelist is already gone. Do not touch that topic.

One intentional P2 behavior change is allowed: the 9 example scripts listed below will rely on the documented editable install instead of mutating sys.path. No robot/runtime behavior may change.

## 2. Editing strategy

Use two distinct kinds of edits.

### Mechanical edits

Ruff formatting/import sorting may touch the full Python tree.

### Manual semantic-prose edits

Manual prose/comment edits are restricted to this allowlist unless a directly adjacent line in the same file must be adjusted for formatting/import cleanup:

~~~text
AGENTS.md
pyproject.toml

dexmani_real/__init__.py
dexmani_real/deployment/__init__.py
dexmani_real/recording/__init__.py

dexmani_real/calibration/camera/session.py
dexmani_real/calibration/camera/solver.py
dexmani_real/recording/io_worker.py
dexmani_real/recording/recorder.py
dexmani_real/sensor/camera/geometry.py
dexmani_real/dataset/pointcloud.py
dexmani_real/ipc/channels.py
dexmani_real/ipc/schema.py
dexmani_real/robot/drivers/xhand.py
dexmani_real/robot/model.py
dexmani_real/planning/kinematics/arm_fk.py
dexmani_real/utils/rate.py
dexmani_real/teleop/retargeting/dexpilot.py
~~~

P2 path-bootstrap edits are additionally allowed in the 9 listed example files in section 5.

Do not expand manual prose cleanup repository-wide merely because a grep finds similar words.

## 3. P0 — formatting

### 3.1 Minimal Ruff config

Add:

~~~toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F", "I"]
~~~

Keep the existing tool.isort configuration unless there is a concrete conflict. Do not add broad lint families such as ANN, D, SIM, TRY, PLR, C90, ARG or PERF.

### 3.2 Ruff execution order

If Ruff is available, prefer:

~~~bash
ruff check --select I --fix dexmani_real examples
ruff format dexmani_real examples
~~~

Run import sorting before the final formatter pass.

After all manual P1/P2 edits, run the formatter once more:

~~~bash
ruff format dexmani_real examples
~~~

Do not use unsafe fixes.

If the ruff executable is absent, try only an already-installed module form:

~~~bash
python -m ruff --version
~~~

Do not install or upgrade anything.

If Ruff is unavailable in both forms, do **not** emulate a whole-repository formatter pass by hand. Manually clean only the known P0 hotspots below plus files changed for P1/P2, and report that full mechanical formatting remains unavailable.

### 3.3 Known P0 hotspots

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

Formatting rule:

> readability beats LOC minimization.

Split independent assignments/arguments onto readable lines. Preserve descriptive names such as handbase_quat_eef_wxyz, observation_timestamp_ns and source_camera_sequence.

Do not manually rewrite expressions merely because Ruff leaves a long string/comment unchanged.

## 4. P1 — prose and comments

The repository is **not** globally over-commented. Do not do blanket comment deletion.

### 4.1 Preserve scientific and physical information

Keep the substance of comments/docstrings that explain:

- math and numerical approximations;
- FK/IK assumptions;
- frames, transforms, quaternion conventions and units;
- vendor SDK/firmware behavior;
- hardware limitations;
- sensor semantics;
- joint/finger ordering;
- experimental thresholds/rationale;
- actual safety-critical reasons;
- concise source attribution.

In particular, preserve:

- SDK ↔ Pinocchio joint ordering in planning/kinematics/hand_fk.py;
- FreeFlyer/Jacobian convention in teleop/retargeting/pin_grad.py;
- SO(3)/quaternion smoothing rationale in teleop/control_loop/action_proposal.py;
- equivalent-angle/staged-homing rationale in planning/paths.py;
- point-cloud filtering rationale in sensor/pointcloud.py;
- RealSense alignment/timestamp/firmware details in sensor/camera/realsense.py;
- manipulability/null-space/branch reasoning in planning/kinematics/ik.py;
- xArm firmware EEF vs URDF EEF distinction in planning/kinematics/arm_fk.py.

These files may be mechanically formatted, but do not broaden manual prose cleanup into them unless they are on the manual allowlist above.

### 4.2 Reduce software-governance narration

Within the manual allowlist, simplify prose built around terms such as:

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

Do not delete these words mechanically. Keep them when they express a real persisted-data convention, scientific provenance, physical-safety rule or actual SDK/process boundary.

Delete/rewrite them when they only narrate software governance.

Concrete targets:

- deployment/__init__.py → one-line package description;
- ipc/schema.py → remove "single source of truth" narration;
- ipc/channels.py → shorten "centralized data plane", ownership and Usage prose;
- dataset/pointcloud.py → describe point-cloud reconstruction directly;
- recording/io_worker.py → remove repeated "owns/ownership-copies/then owns" narration while retaining the actual shared-memory/serialization facts;
- recording/recorder.py → compress long transaction/control-flow commentary, but retain real persistence semantics.

### 4.3 Remove implementation-history prose

Describe the current implementation, not how it evolved.

Within the allowlist, rewrite history-oriented prose such as:

~~~text
no longer
previous implementation
existing implementation
restores
dropped
upstream's
without touching site-packages
retained for parity
~~~

when those phrases describe development history.

Examples:

- arm_fk.py: state how EEF is derived now; do not say what the arm worker "no longer" does.
- dexpilot.py: retain the current human-flexion prior, its mathematical purpose, and concise attribution; remove detailed dex-retargeting version archaeology, commented-upstream-parameter discussion, site-packages discussion, "vanilla optimizer" wording and parity/fallback history.

Do not alter the objective, gradient or solver behavior.

### 4.4 Keep attribution

Keep concise attribution such as:

~~~text
Adapted from TAG/Retargeting/Hand_Retargeting/utils/pin_grad.py.
~~~

Current algorithm + concise attribution is preferred over upstream patch history.

### 4.5 Shorten README-like module prose

Highest priority:

~~~text
dexmani_real/calibration/camera/session.py
~~~

Reduce the module docstring to roughly 1–3 sentences describing the current calibration role. Remove embedded hardware-preparation checklist, CLI usage, keyboard-control table and architecture narration from the internal module docstring.

Do not delete code or operator safety checks.

Also simplify docstrings/comments in the other manual-allowlist files where they merely restate module boundaries.

### 4.6 Package docstrings

Make these minimal without changing exports:

~~~text
dexmani_real/__init__.py
dexmani_real/deployment/__init__.py
dexmani_real/recording/__init__.py
~~~

Remove subsystem inventories/manifestos. A one-line description is sufficient.

### 4.7 Restatement/decorative comments

Within the manual allowlist, remove comments that simply restate the next line/function name and decorative section separators.

Compress verbose local commentary when one factual line is enough, e.g.:

~~~text
Preserve failed partial episodes for offline diagnosis.
~~~

Keep "why" comments for unusual behavior.

### 4.8 Private helper docstrings

Do not run a repository-wide private-docstring purge.

Within the allowlist, delete a private helper docstring only when it is clearly a restatement of the function name/body. Keep any math/frame/hardware/safety caveat.

### 4.9 Dynamic facts

Avoid repeating mutable constants such as schema version numbers in prose when the code constant already defines them.

## 5. P1 — AGENTS.md guardrail

Add a short "Code style" section to AGENTS.md. Do not rewrite the existing safety/research guidance.

It should state, concisely:

- Keep research code direct and readable.
- Do not compress independent operations merely to reduce LOC.
- Comment non-obvious robotics, math, frames, units, SDK behavior, experimental rationale, attribution and real safety reasons.
- Do not narrate architecture/ownership/contracts/validation/obvious control flow when code already expresses them.
- Describe current behavior rather than implementation history.
- Private helpers normally do not need docstrings.
- Use Ruff formatting/import sorting when available.
- Do not add abstractions solely for style tooling.

## 6. P2 — example import bootstrap cleanup

Remove the repository-root sys.path injection from exactly these currently verified examples:

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

Do not add it to examples/realsense_record_example.py.

README already establishes:

~~~bash
python -m pip install -e .
~~~

Remove only:

- repository-root calculation used solely for sys.path mutation;
- sys.path.insert boilerplate;
- imports made unused solely by that deletion.

Preserve independent uses of sys and pathlib.Path.

Do not otherwise change example CLI behavior.

## 7. P2 — conservative annotation/docstring cleanup

Do **not** replace type-ignore comments with assert statements or new runtime checks. That would exceed style-only scope.

Do not proactively chase type ignores.

Only if a touched line contains a redundant prose comment after a necessary type-ignore may that prose be shortened/removed while preserving the ignore.

Keep external-package ignores such as:

~~~python
# type: ignore[import-untyped]
~~~

Do not standardize Google vs NumPy docstrings.

Do not proactively edit scientific Args/Returns documentation outside the manual allowlist. Within the allowlist, trim only text that plainly repeats the signature; keep units, shapes, frames and algorithmic relationships.

## 8. Files explicitly not to restructure

Do not split or merge these:

~~~text
dexmani_real/config/defaults.py
dexmani_real/sensor/pointcloud.py
dexmani_real/planning/kinematics/ik.py
dexmani_real/recording/recorder.py
dexmani_real/sensor/camera/realsense.py
~~~

Do not move examples into tools/.

## 9. Execution order

1. Read AGENTS.md and this task.
2. Run git status --short; preserve unrelated user changes.
3. Confirm current HEAD and inspect changes since the baseline if any.
4. Add the minimal Ruff config.
5. Run Ruff import sorting + formatting if already available.
6. Inspect git diff --stat and git diff after the mechanical pass.
7. Make only the allowlisted P1 manual prose edits.
8. Add the short AGENTS.md Code style section.
9. Remove the 9 P2 sys.path bootstraps.
10. Apply only incidental P2 type/docstring cleanup allowed above.
11. Run the final Ruff format pass if available.
12. Run validation.
13. Inspect the complete diff and revert anything that changes program semantics or expands scope.

## 10. Validation

No hardware execution.

Always run:

~~~bash
python -m compileall -q dexmani_real examples
git diff --check
~~~

If Ruff is available, run:

~~~bash
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
~~~

If only python -m ruff is available, use the equivalent module commands.

Do not install missing tooling.

Final searches:

~~~bash
git grep -n "single source of truth" -- dexmani_real
git grep -n "no longer" -- dexmani_real
git grep -n "site-packages" -- dexmani_real
git grep -n "sys.path.insert" -- examples
~~~

Search counts are not goals. Review context. The sys.path search should be empty for the verified repository-root bootstrap pattern.

Also inspect:

~~~bash
git diff --stat
git diff
~~~

## 11. Acceptance criteria

Finish only when:

1. P0 mechanical formatting/import organization is complete when Ruff is available; otherwise the limitation is reported and manual formatting stayed restricted to the known hotspots/touched files.
2. Long code expressions are wrapped without abbreviating meaningful robotics names.
3. P1 architecture/history prose is reduced only in the manual allowlist.
4. Scientific, geometric, hardware, SDK, attribution and experimental-rationale information is preserved.
5. calibration/camera/session.py no longer embeds README/CLI material in its module docstring.
6. the three targeted package docstrings are minimal.
7. AGENTS.md has the concise Code style guardrail.
8. all 9 verified repository-root sys.path bootstraps are removed.
9. no assert/control-flow/runtime checks were added for type-cleanup purposes.
10. no P3 or structural work was performed.
11. compileall and diff-check pass.
12. Ruff checks pass when Ruff is available.
13. no hardware code was executed.
14. no robot/runtime/algorithm/schema/safety behavior was intentionally changed, except the explicit example import-bootstrap cleanup.

## 12. Handoff

Report concisely:

1. files changed;
2. P0 mechanical changes;
3. P1 prose/comment changes;
4. P2 example-bootstrap changes;
5. validation commands/results;
6. Ruff availability;
7. confirmation that no robot/runtime/algorithm/schema/safety behavior was intentionally changed;
8. any deviation from this task and the exact reason.

Do not claim hardware validation.
