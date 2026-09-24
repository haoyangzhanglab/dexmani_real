# Clean configuration tree and centralize calibration state

## Context

Repository: `haoyangzhanglab/dexmani_real`

Implementation baseline reviewed against `main` at commit:

```text
5e21b3df72cb956e00f7d8dd70e3acfc9ba10d0f
```

The only later commit at the time of the final design review added this task document; re-check current HEAD before editing as required below.

This task is a focused repository-structure cleanup. Do not broaden it into a configuration-framework rewrite.

The current `dexmani_real/config/` directory mixes two different concepts:

- Python runtime configuration code:
  - `defaults.py`
  - `experiment.py`
  - `pointcloud.py`
- Mutable calibration state:
  - `cameras.json`
  - `desk_plane.json`
  - `vr_transform.json`

Separate those responsibilities while keeping everything inside the `dexmani_real` package.

The intended distinction is:

```text
dexmani_real/config/
    defines and resolves runtime configuration

dexmani_real/calibration/
    implements calibration workflows and contracts

dexmani_real/calibration/state/
    stores the currently accepted persistent calibration state
```

Keep the mechanism simple. Do not introduce Hydra, Draccus, Pydantic settings, registries, profile systems, rig IDs, a new calibration config dataclass, or another generic path abstraction.

Follow `AGENTS.md`: physical safety first; preserve unrelated work; prefer delete/inline/merge over new abstractions; do not run hardware-affecting code.

---

## Required final tree

The relevant package structure should end up as:

```text
dexmani_real/
├── config/
│   ├── __init__.py
│   ├── defaults.py
│   ├── experiment.py
│   └── pointcloud.py
│
├── calibration/
│   ├── __init__.py
│   ├── table.py
│   ├── camera/
│   │   ├── extrinsics.py
│   │   ├── motion.py
│   │   ├── session.py
│   │   └── solver.py
│   └── state/
│       ├── cameras.json
│       ├── table_plane.json
│       └── vr_transform.json
│
└── ...
```

Important:

- `calibration/state/` is a package-data directory, not a Python package.
- Do **not** add `calibration/state/__init__.py`.
- Do **not** add a second directory named `calibration` elsewhere.
- Do **not** move configuration outside the `dexmani_real` package.

---

## 0. Pre-edit inspection

Before changing files:

1. Run `git status --short`.
2. Preserve all unrelated changes.
3. Confirm the current HEAD and inspect the actual current contents/references before editing.
4. Search the full repository, including `examples/`, for:
   - `cameras.json`
   - `desk_plane.json`
   - `vr_transform.json`
   - `PACKAGE_DIR / "config"`
   - `dexmani_real/config/`
   - `CAMERA_CALIBRATION_PATH`
   - `vr_transform_path`
5. Trace each changed calibration artifact from producer to consumer:
   - camera calibration writer -> `CameraExtrinsics` / recording / pointcloud / deployment
   - table calibration writer -> `resolve_table_plane` / pointcloud / collision geometry
   - VR heading writer -> teleop startup / teleop control loop

Do not assume the baseline file contents are unchanged if HEAD has advanced.

---

## 1. Move the three persistent calibration files

Move, preserving their contents:

```text
dexmani_real/config/cameras.json
-> dexmani_real/calibration/state/cameras.json

dexmani_real/config/desk_plane.json
-> dexmani_real/calibration/state/table_plane.json

dexmani_real/config/vr_transform.json
-> dexmani_real/calibration/state/vr_transform.json
```

Rename `desk_plane.json` to `table_plane.json` intentionally. The current code already uses table-oriented terminology such as:

- `TableCollisionConfig`
- `TablePlaneFit`
- `fit_table_plane`
- `publish_table_plane`
- `resolve_table_plane`

Do not retain duplicate old JSON files and do not add old-path fallback logic.

After the migration, `dexmani_real/config/` should contain only Python configuration code.

---

## 2. Keep fixed calibration resource paths in `dexmani_real/calibration/__init__.py`

Do **not** create `calibration/paths.py`.

The existing `dexmani_real/calibration/__init__.py` is intentionally tiny and is an appropriate resource boundary for calibration resources that are not runtime-configurable. Keep it side-effect free and dependent only on the standard library.

Define the state directory and fixed camera/VR paths there, for example:

```python
from pathlib import Path

CALIBRATION_STATE_DIR = Path(__file__).resolve().parent / "state"

CAMERAS_PATH = CALIBRATION_STATE_DIR / "cameras.json"
VR_TRANSFORM_PATH = CALIBRATION_STATE_DIR / "vr_transform.json"
```

Do **not** force the table plane into the same absolute-path mechanism. `TableCollisionConfig.plane_path` is an actual serializable runtime configuration field with supported YAML/data overrides, so its default should remain a stable package-relative string and be resolved through the runtime resolver.

Requirements:

- Importing `dexmani_real.calibration` must not connect hardware.
- Do not import camera SDKs, NumPy, calibration algorithms, config modules, or teleop code from `calibration/__init__.py`.
- `CAMERAS_PATH` and `VR_TRANSFORM_PATH` are the fixed canonical package-internal locations for camera and VR state.
- Table-state location is owned by `TableCollisionConfig.plane_path` plus `resolve_table_plane_path()`.
- Explicit custom paths used by tests or supported overrides must continue to work.

---

## 3. Update table calibration configuration and path resolution

### `dexmani_real/config/defaults.py`

Change the default for `TableCollisionConfig.plane_path` from:

```python
"dexmani_real/config/desk_plane.json"
```

to the stable package-relative string:

```python
"dexmani_real/calibration/state/table_plane.json"
```

Do **not** replace this serializable config value with an absolute `Path` or `str(TABLE_PLANE_PATH)`. `config_as_dict()`, `--print-config`, and persisted run configuration should not contain machine-specific checkout/site-packages prefixes.

Preserve the current ability to override `plane_path` through YAML/data/CLI and preserve `None` semantics for inline `plane_abcd`.

Do not redesign `TableCollisionConfig`.

### `dexmani_real/config/experiment.py`

There is currently duplicated relative-path resolution logic between runtime table loading and the pointcloud example.

Add one small public helper owned by this module:

```python
def resolve_table_plane_path(table: TableCollisionConfig) -> Path:
    ...
```

Expected semantics:

- reject/raise if `table.plane_path is None`
- `expanduser()`
- if absolute, use it directly
- if relative, resolve it relative to the repository root exactly as current runtime semantics do
- return a resolved `Path`

Refactor `resolve_table_plane()` to use that helper.

Do not add a generic resource resolver or path registry. This helper exists because the exact table-plane resolution behavior is already duplicated and is part of the table configuration contract.

Preserve all existing validation and error behavior as much as possible.

---

## 4. Update camera calibration ownership

### `dexmani_real/calibration/camera/solver.py`

Remove the old ownership of:

```python
CAMERA_CALIBRATION_PATH = PACKAGE_DIR / "config" / "cameras.json"
```

and remove the now-unneeded `PACKAGE_DIR` import if nothing else uses it.

The solver should remain calibration math plus persistence helpers. It should not define the canonical repository location.

### `dexmani_real/calibration/camera/session.py`

Import:

```python
from dexmani_real.calibration import CAMERAS_PATH
```

Use `CAMERAS_PATH` when publishing an accepted camera calibration.

Do not import the canonical path indirectly from `solver.py`.

Update stale module documentation that says accepted results are written to `dexmani_real/config/cameras.json`.

### `dexmani_real/calibration/camera/extrinsics.py`

Import `CAMERAS_PATH`.

Delete the private/default-path method that reconstructs the old path, if it is still present.

Keep explicit custom-path support. The constructor should continue to permit code such as:

```python
CameraExtrinsics("/tmp/cameras.json")
```

while:

```python
CameraExtrinsics()
```

uses `CAMERAS_PATH`.

Prefer accepting `str | Path | None` if consistent with the current implementation.

Update all stale error messages and docstrings that tell users to create `dexmani_real/config/cameras.json`.

Do not change the JSON schema or camera extrinsics math.

---

## 5. Simplify VR transform path handling

### `dexmani_real/teleop/config.py`

`TeleopConfig.vr_transform_path` is currently a fixed package resource masquerading as a session configuration field. It is not exposed as a meaningful runtime/YAML/CLI option.

Delete `vr_transform_path` from `TeleopConfig`.

Do not replace it with another config field or a calibration config dataclass.

### `dexmani_real/teleop/loop.py`

Import `VR_TRANSFORM_PATH` from `dexmani_real.calibration`.

Replace repository-root/path concatenation with:

```python
load_vr_transform(VR_TRANSFORM_PATH)
```

Remove `pathlib.Path` if it becomes unused.

Do not move `dexmani_real/teleop/vr_transform.py` in this task. It contains the consumer-side validated VR transform contract and is also used by VR mapping. Moving it would enlarge the change without improving runtime behavior.

### `dexmani_real/teleop/session.py`

Import the canonical calibration paths needed here.

Replace the startup check:

```python
load_vr_transform(repo_root / "dexmani_real/config/vr_transform.json")
```

with the canonical `VR_TRANSFORM_PATH`.

Review `_validate_recording_resources()`:

- VR transform is already validated unconditionally before process startup.
- It is not specifically a recording resource.
- Remove the duplicate VR-transform entry from recording-resource validation.
- Keep camera calibration in recording-resource validation if the current recording path requires it.
- Use `CAMERAS_PATH` rather than reconstructing the old path.
- After those changes the helper should no longer need a `repo_root` argument; remove that now-unused parameter and update its call site.

Keep the `repo_root` local in `run_teleop_experiment()` because the current process-building/data-directory path still legitimately uses it.

---

## 6. Synchronize all relevant examples

Examples are part of this task. Do not update only package code.

### `examples/calibrate_vr_heading.py`

This is a direct producer of the VR state file.

- Import `VR_TRANSFORM_PATH` from `dexmani_real.calibration`.
- Remove `PACKAGE_DIR` if it is no longer used.
- Remove the local `_OUTPUT_PATH = PACKAGE_DIR / ...` indirection.
- Use `VR_TRANSFORM_PATH` directly for:
  - existing-file checks
  - backup creation
  - atomic JSON publish
  - user-visible saved-path output
- Preserve current quality gates, backup behavior, and atomic write semantics.
- Update the module docstring so it no longer says `config/vr_transform.json`.

Do not add `--output`, `--calibration-dir`, rig/profile selection, or another path option.

### `examples/calibrate_camera.py`

The actual write is performed by the calibration session, so do not duplicate persistence logic here.

Synchronize only the stale user-facing wording about the camera calibration output.

Do not add another output-path print: `save_camera_calibration()` already reports the actual path after a successful write. Do not import `CAMERAS_PATH` into this example solely for duplicate display.

Do not add a new output-path option.

### `examples/pointcloud_process_example.py`

This example can both consume and publish the table plane, so it must use exactly the same path semantics as runtime.

- import `resolve_table_plane_path` from `dexmani_real.config.experiment`
- delete the local duplicated `_resolve_table_plane_path()`
- use the shared resolver when deciding where a newly accepted table fit is published
- preserve current explicit confirmation, backup, atomic publication, and round-trip verification
- update all `desk_plane.json` wording to `table_plane.json`

Do not alter point-cloud processing behavior or calibration math.

### Other examples

Audit at least:

- `examples/realsense_record_example.py`
- `examples/visualize_episode.py`
- `examples/export_policy_zarr.py`
- `examples/run_policy.py`
- `examples/collect_teleop.py`
- `examples/keyboard_teleop.py`

These should normally require no structural path edits because they should consume public APIs such as `CameraExtrinsics()`, `resolve_table_plane()`, or `ExperimentConfig`.

If they contain stale direct old-path references in the current HEAD, update those references. Otherwise leave them unchanged.

A successful design means most consumers do not need to know the physical calibration-state directory.

---

## 7. Packaging

Update `pyproject.toml`.

Replace:

```toml
[tool.setuptools.package-data]
"dexmani_real.config" = ["*.json"]
```

with package data for the calibration state directory:

```toml
[tool.setuptools.package-data]
"dexmani_real.calibration" = ["state/*.json"]
```

Do not make `state/` a Python package.

Ensure all three canonical JSON files remain available in editable installs and built packages.

---

## 8. Ignore generated calibration backups

Current camera/table/VR calibration workflows create timestamped backups next to the canonical JSON files, e.g. `*.json.bak.*`.

Add a narrow rule to `.gitignore`:

```gitignore
dexmani_real/calibration/state/*.json.bak.*
```

Do not ignore the three canonical JSON state files themselves.

Do not add a broad repository-wide `*.json.bak.*` rule unless the current repository clearly requires it.

---

## 9. Stable README update

Update `README.md` only with the stable user-facing convention, not an implementation inventory.

Near installation/configuration guidance, state succinctly that:

- runtime parameters resolve via CLI > YAML > Python defaults
- accepted camera/table/VR calibration state lives under `dexmani_real/calibration/state/`
- corresponding calibration entry points explicitly update that state

Do not turn README into a migration log.

---

## 10. Explicit non-goals

Do **not** do any of the following in this task:

- do not introduce Hydra, Draccus, Pydantic, OmegaConf, or another config framework
- do not split `defaults.py` into multiple domain files
- do not rename `experiment.py`
- do not move `pointcloud.py`
- do not create `config/paths.py`
- do not create `calibration/paths.py`
- do not create `CalibrationConfig`, `CalibrationPaths`, profile, rig, or registry abstractions
- do not move configuration outside the `dexmani_real` package
- do not move `teleop/vr_transform.py`
- do not change calibration JSON schemas
- do not change camera/table/VR calibration math
- do not change keyboard teleop, IK, timing, policy, or control-loop semantics
- do not keep old calibration JSON copies for compatibility
- do not add fallback loading from the old `dexmani_real/config/*.json` paths
- do not run real-hardware examples or any calibration write workflow as a test

---

## 11. Repository-wide cleanup checks

After editing, search the full repository again.

The final tree must have no stale production/example references to:

```text
dexmani_real/config/cameras.json
dexmani_real/config/desk_plane.json
dexmani_real/config/vr_transform.json
desk_plane.json
PACKAGE_DIR / "config" / "cameras.json"
PACKAGE_DIR / "config" / "vr_transform.json"
```

There should be no obsolete:

```text
CAMERA_CALIBRATION_PATH
TeleopConfig.vr_transform_path
```

unless current HEAD contains a demonstrably different use that still has valid semantics; if so, explain rather than silently retaining duplication.

Also review all comments/docstrings/error messages/user prompts for old paths.

---

## 12. Required offline validation

Do not connect hardware.

Run the repository's normal offline checks:

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
git diff --check
```

If the environment has the required optional dependencies, also run:

```bash
python -m dexmani_real.deployment.smoke_test
```

Do not install or upgrade dependencies just to make optional checks available.

Add/run a focused one-off pure-Python smoke check where possible that verifies the canonical files resolve and existing loaders can read them without hardware access, conceptually:

```python
from dexmani_real.calibration import CAMERAS_PATH, VR_TRANSFORM_PATH
from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.config.experiment import resolve_experiment_config, resolve_table_plane_path
from dexmani_real.teleop.vr_transform import load_vr_transform

assert CAMERAS_PATH.is_file()
assert VR_TRANSFORM_PATH.is_file()

runtime = resolve_experiment_config()
assert resolve_table_plane_path(runtime.environment.table).is_file()

CameraExtrinsics()
load_vr_transform(VR_TRANSFORM_PATH)
```

Only run this if inspection confirms these constructors/functions are hardware-free, as required by repository policy.

Also inspect package-data behavior if practical without modifying the environment.

---

## 13. Definition of done

The task is complete only when all of the following hold:

1. `dexmani_real/config/` contains only Python configuration code.
2. The three accepted calibration JSON files live under `dexmani_real/calibration/state/`.
3. `desk_plane.json` has been renamed to `table_plane.json`.
4. `dexmani_real/calibration/__init__.py` owns the fixed canonical camera/VR paths, while the table path remains a stable serializable runtime-config value.
5. Camera calibration writer and `CameraExtrinsics()` use the same canonical camera path.
6. Runtime table loading and `pointcloud_process_example.py` use the same table-path resolver, and default config serialization contains the stable package-relative table path rather than a machine-specific absolute path.
7. VR calibration writer, teleop preflight, and teleop control-loop loader use the same canonical VR path.
8. `TeleopConfig` no longer carries a fixed `vr_transform_path`.
9. No compatibility fallback to the old `config/*.json` locations remains.
10. Required `examples/` documentation/code has been synchronized.
11. Package data includes `calibration/state/*.json`.
12. Generated timestamped calibration backups are ignored without ignoring canonical state.
13. All applicable offline checks pass.
14. The final diff contains no unrelated control, timing, IK, policy, or hardware behavior changes.

---

## 14. Final report

When finished, provide a concise report containing:

- final tree for `config/` and `calibration/`
- changed files grouped by purpose
- any deviations from this task caused by newer HEAD state
- repository-wide stale-reference search results
- offline checks run and their results
- checks not run and the concrete reason
- explicit confirmation that no hardware-affecting command was executed
