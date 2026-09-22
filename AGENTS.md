# Working on DexMani Real

This is a personal PhD research repository for real-robot dexterous data collection and learned-policy evaluation using xArm7, XHand, RealSense, and VR/HTS.

It is not a generic robotics platform, production policy-serving system, distributed command-transaction layer, or generic dataset framework.

## Priorities

Physical safety > experiment correctness > iteration speed > readability > generic extensibility > enterprise robustness.

Prefer delete, then inline, then merge, then rewrite. Add an abstraction only for an actual hardware/resource boundary or demonstrated duplication.

Do not preserve wrappers, validators, state machines, or protocol fields merely because they already exist. Git history preserves old implementations.

Do not replace a deleted mechanism with the same concept under a new name.

## Hardware

Do not execute hardware-affecting code without explicit authorization. This includes live SDK discovery/connection, robot motion, homing, physical replay, teleoperation, policy rollout, camera capture, XHand diagnostics, and calibration writes.

Inspect imports, constructors, and examples before executing them.

Keep live SDK objects in their owning processes:

- xArm SDK in arm worker;
- XHand SDK in hand worker;
- RealSense SDK in camera worker;
- model/CUDA runtime in policy worker.

Imports and ordinary constructors must not connect devices.

Never report offline checks as hardware validation.

## Real physical guarantees

Preserve the actual physical guarantees that matter for this research stack:

- finite SDK inputs;
- arm/hand physical or rated hard limits;
- emergency stop;
- xArm SDK/controller speed and acceleration limits;
- xArm/XHand SDK error handling;
- safe disconnect;
- run_id stale-command fencing at the last software boundary before SDK admission;
- sensor freshness for required observations;
- XHand tactile aggregate/dense partial validity;
- child-process shutdown before shared-memory release.

Normal teleop, learned-policy evaluation, and ordinary supervised streaming do not require generic software collision checking.

Collision/workspace/table path planning is owned by planned return_home.

Cartesian teleop or EE policies may use a simple EEF workspace clip before IK. Joint policies do not run normal-runtime FK merely to impose a generic workspace gate.

XHand normal streaming sends validated absolute position targets directly. Do not add producer delta clipping, worker software slew, adoption/reached identities, or a command ACK protocol unless the user explicitly changes the design.

xArm normal streaming relies on absolute target limits plus SDK/controller speed and acceleration controls. Do not add a hidden high-level delta filter merely for symmetry.

## Runtime semantics

Keep SafetyState simple: DISARMED / ARMED / RUNNING / FAULT.

Keep run_id as the lifecycle epoch that prevents stale delayed work from regaining motion authority.

Do not introduce command_id, actuator adoption ledgers, partial-adoption accounting, or single-inflight command transactions.

The command transport is latest-target semantics. Ring sequence may remain an internal transport detail, but it is not scientific data or public command identity.

C remains operator-facing pause/resume. Preserve fresh post-pause re-anchor behavior.

## Observation and validity

For arm, hand joint state, camera, VR, and point cloud:

- publish only structurally usable samples;
- on acquisition/derivation failure, do not publish a new fake sample;
- consumers use latest-sample freshness;
- do not add generic connected/state_valid/qpos_stale/observation_valid flags.

XHand tactile is the important exception: aggregate and dense tactile may fail independently while joint state remains usable, so explicit tactile validity remains in IPC/raw.

Do not build source -> receive -> publish -> commit causal proof frameworks.

One control step should assemble one current observation snapshot from latest required fresh samples. Teleop control and raw recording should reuse that same snapshot.

Policy temporal history belongs in a local control-row deque, not in historical sensor-ring reconstruction.

## Data

Raw episodes are the experiment source of truth.

Preserve scientifically useful information such as:

- robot state;
- RGB-D and calibration;
- XHand current/tactile and tactile validity;
- raw VR source data;
- real experiment timestamps;
- high-level absolute control targets;
- small diagnostic frame status.

Do not persist runtime self-proof such as command adoption, generic validity, or collision-gate audit fields.

Canonical Zarr is a full fixed training cache, not policy-specific. It should contain all canonical learning-relevant modalities and let dexmani_policy select sensor_modalities at load time.

Zarr must not contain runtime timing arrays or validity masks.

Raw -> Zarr admission is whole-episode all-or-nothing. Do not repair, split, resample, drop individual bad rows, or silently salvage an abnormal demonstration. Reject the whole raw episode and report the concrete reason.

Keep action and action_ee semantics consistent: both represent the same final high-level target in different action spaces.

## Changes

Before editing, inspect git status --short and preserve unrelated changes.

Trace definition -> producer -> transformation -> consumer -> side effect from actual entry points.

Read both sides of changed process, policy, robot, and storage boundaries.

Source and resolved configuration define current behavior; update stale docs and comments.

Validate external inputs at the owning boundary. Do not repeatedly revalidate structures just constructed internally or wrap exceptions without a real recovery decision.

Preserve units, coordinate frames, joint/finger ordering, action semantics, sensor semantics, and actual research data.

Do not silently reinterpret an existing persisted field. For a real schema semantic change, use a new schema version instead of runtime compatibility branches.

Use dexmani_policy public interfaces.

## Checks and handoff

The repository intentionally has no committed tests directory; do not restore it.

Start with focused one-off offline smoke checks for changed pure logic.

Useful low-cost checks:

    python -m compileall -q dexmani_real examples
    ruff check --select F401,F821,F822,F823 dexmani_real examples
    git diff --check

Protect transforms, FK/IK, retargeting, dataset conversion, simple target clipping, observation history, and real regressions with focused offline checks.

If Ruff or another optional dependency is unavailable, report it rather than installing/upgrading the experiment environment.

Never run hardware examples as tests.

Inspect the final diff/searches for dead imports, deleted config, stale docs, and obsolete terminology.

README contains stable setup/workflow semantics. codex_task.md contains the current implementation migration specification. Implementation details belong in source after the migration is complete.
