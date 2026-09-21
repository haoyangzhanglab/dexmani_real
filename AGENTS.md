# Working on DexMani Real

This is a personal PhD research repository for real-robot dexterous data collection
and learned-policy evaluation, using xArm7, XHand, RealSense and VR/HTS.
It is not a generic robotics platform or production policy-serving service.

## Priorities

Physical safety > experiment correctness > iteration speed > readability >
generic extensibility > enterprise robustness.

Prefer delete, then inline, then merge, then rewrite. Add an abstraction only for
an actual hardware/resource boundary or demonstrated duplication. A single
implementation does not need an interface for hypothetical future backends.
Do not preserve internal wrappers, states, validators or tests merely because
they already exist. Git history preserves old implementations.

## Hardware

Do not execute hardware-affecting code without explicit authorization. This
includes connection/discovery through a live SDK, robot motion, homing, physical
replay, teleoperation, policy rollout and calibration writes. Inspect imports,
constructors and examples before executing them.

Preserve the actual physical guarantees: mechanical limits, finite SDK inputs,
actuator speed/step limits, emergency stop, robot error handling, safe disconnect,
rejection of revoked commands and experiment-required collision/workspace checks.
Each guarantee needs a clear owner, not repeated checks in every layer.
Generation/freshness mechanisms may be simplified only after tracing the race or
physical hazard they currently prevent, not merely because their names look complex.
Keep live SDK objects in their owning process. Imports and ordinary constructors
must not connect devices. Never report offline checks as hardware validation.

## Changes

Before editing, inspect `git status --short` and preserve unrelated changes.
Trace definition → producer → transformation → consumer → side effect from actual
entry points. Read both sides of changed process, policy, robot and storage boundaries.
Source and resolved configuration define current behavior; update stale docs.

Validate external inputs at their owning boundaries. Do not revalidate structures
just constructed internally or wrap exceptions without a real recovery decision.
Preserve units, coordinate frames, action timing, sensor validity and actual
scientific data. Do not silently reinterpret an existing persisted field.
When old data needs conversion, prefer an explicit offline conversion over runtime
compatibility branches. Use `dexmani_policy` public interfaces.

## Checks and handoff

The repository intentionally has no committed `tests/` directory; do not restore it.
Start with focused one-off offline smoke checks for changed pure logic. Do not add
a committed test framework unless explicitly requested. Useful low-cost checks are:

```bash
python -m compileall -q dexmani_real examples
ruff check --select F401,F821,F822,F823 dexmani_real examples
git diff --check
```

Protect transforms, FK/IK, retargeting math, dataset conversion, clipping and real
regressions with focused offline checks. If Ruff or another optional dependency is
unavailable, report it rather than installing or upgrading the experiment
environment. Never run hardware examples as tests.

Inspect the final diff for dead imports, configuration, docs and terminology.
Report measured changes, passing/failed/skipped checks and remaining limitations.
Keep one-off audits and plans in review artifacts or PRs, not permanent runtime docs.
README contains setup, workflows and stable ownership; implementation details belong
in source. Avoid new compatibility adapters and half-migrated internal interfaces.
