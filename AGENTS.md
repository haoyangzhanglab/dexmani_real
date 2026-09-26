# Working on DexMani Real

This is a personal PhD research repository for real-robot dexterous data collection and learned-policy evaluation using xArm7, XHand, RealSense, and VR/HTS. Keep changes focused on these experiments.

## Priorities

Physical safety > experiment correctness > iteration speed > readability > generic extensibility.

Prefer delete, then inline, then merge, then rewrite. Add abstractions only for actual hardware/resource boundaries or demonstrated duplication. Do not retain obsolete mechanisms or recreate them under new names.

## Hardware and safety

Do not execute hardware-affecting code without explicit authorization, including SDK discovery/connection, motion, homing, replay, teleoperation, rollout, camera capture, diagnostics, and calibration writes. Inspect relevant imports, constructors, and entry points before execution. Ordinary imports and constructors must not connect devices.

Keep xArm, XHand, and RealSense SDK objects in their respective workers; keep model/CUDA runtime in the policy worker. Shut down child processes before releasing shared memory.

Preserve finite SDK inputs, physical/rated joint limits, emergency stop, xArm speed/acceleration controls, SDK error handling, safe disconnect, and required sensor freshness. Fence stale commands with `run_id` at the last software boundary before SDK admission.

Cartesian IK checks target self-collision using the final prepared hand target. Arm-only Cartesian control uses fresh measured hand geometry when available; hand-disabled operation assumes the hand is absent or secured at configured home. Normal teleop/eval/replay do not check transition paths or environment collisions; full collision/workspace/table path planning belongs to planned return-home.

Cartesian control may clip EEF workspace before IK. Do not add a generic FK workspace gate to joint policies. Normal arm/hand streaming uses validated absolute targets and device controls; do not add software delta/slew filters without an explicit design change.

## Runtime and observations

Keep `SafetyState` simple: DISARMED / ARMED / RUNNING / FAULT. Commands use latest-target semantics with `run_id` as the lifecycle epoch. Do not add command identities, adoption/reached ledgers, ACKs, or command transactions. Ring sequence is an internal transport detail.

C remains operator-facing pause/resume, with fresh post-pause re-anchoring.

Publish only usable sensor samples; acquisition/derivation failures must not publish fake replacements or generic validity flags. Consumers use latest-sample freshness. XHand aggregate and dense tactile validity remain explicit because either can fail independently of joint state.

Assemble one current observation snapshot per control step and reuse it for teleop and recording. Policy history belongs in a local control-row deque, not sensor-ring reconstruction.

A point cloud is self-contained. Match its source camera frame only when the policy consumes RGB and point cloud together; pointcloud-only rollout recording uses independent latest fresh camera telemetry.

## Research data

Raw v34 episodes are the experiment source of truth. Preserve measured robot state, RGB-D, current/tactile payloads, final absolute joint targets, frame_valid, monotonic-false episode_valid, and physical camera/hand-mount calibration. Invalid tactile payloads contain NaN; runtime aggregate/dense validity remains explicit. Keep freshness, causal selection, publication clocks and fixed-dt continuity checks in runtime; do not persist timestamps, raw VR, detailed status, software provenance, or transport bookkeeping. Snapshot the resolved physical hand mount at recording START; Raw is its sole offline truth source. New recording requires complete eye-to-hand aligned RGB-D calibration before START.

Canonical Zarr is a full training cache containing all learning-relevant modalities; `dexmani_policy` selects model inputs at load time. Do not include runtime timing arrays or validity masks.

Raw-to-Zarr admission is whole-episode all-or-nothing. Reject abnormal episodes with concrete reasons; do not repair, split, resample, drop bad rows, or silently salvage them. `action` and `action_ee` must describe the same final high-level target in different action spaces.

Preserve units, frames, joint/finger ordering, and sensor/action semantics. Persisted semantic changes require a new schema version, not silent reinterpretation or compatibility branches.

## Making changes

Check `git status --short` before editing and preserve unrelated changes. Trace the relevant definition, producer, transformation, consumer, and side effect from actual entry points; read both sides of changed process, policy, robot, and storage boundaries.

Source and resolved configuration define current behavior. Use `dexmani_policy` public interfaces. Validate external inputs at their owning boundary; avoid redundant internal validation or exception wrapping without recovery.

Keep code direct and readable. Comments should explain non-obvious robotics, math, frames, units, SDK behavior, experimental rationale, attribution, and safety reasons, rather than repeat control flow or implementation history. Use Ruff formatting and import sorting when available.

README documents stable workflows and operating conventions. Keep implementation inventories and historical task plans out of standing instructions.

## Offline checks

Use focused one-off smoke checks for changed pure logic, especially transforms, FK/IK, retargeting, conversion, target clipping, and observation history. Do not add a committed tests directory or run hardware examples as tests. Offline checks are not hardware validation.

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

If optional tools or dependencies are unavailable, report that rather than installing or upgrading the experiment environment. Review the final diff for unrelated changes, dead code/config, and stale documentation.
