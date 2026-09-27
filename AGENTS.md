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

C remains operator-facing pause/resume. A clean operator pause freezes the pending capture; an unresolved clean capture defaults to save. Resume intentionally starts a new Raw episode before fresh post-START re-anchoring and motion authorization.

Publish only usable sensor samples; acquisition/derivation failures must not publish fake replacements or generic validity flags. Consumers use latest-sample freshness. XHand aggregate and dense tactile validity remain explicit because either can fail independently of joint state.

Assemble one current observation snapshot per control step and reuse it for teleop and recording. Policy history belongs in a local control-row deque, not sensor-ring reconstruction.

A point cloud is self-contained. Match its source camera frame only when the policy consumes RGB and point cloud together; pointcloud-only rollout recording uses independent latest fresh camera telemetry.

## Research data

Current Raw episodes use `format="dexmani.raw"`, with control rows and aligned depth in `data.h5` plus `rgb.mp4`. Normal runtime/export/replay code supports only this current format; do not add legacy Raw compatibility branches. Legacy data is outside the supported repository contract. Raw is immutable experiment evidence and is not rewritten in place. Validate known fields that are present, tolerate unknown additive fields, and let consumers require the capabilities they need. The current writer remains strict about its emitted fields.

Preserve measured robot state, RGB-D, current/tactile payloads, actual published absolute joint targets, physical camera/hand-mount calibration, and final `termination_reason`. New Raw has no persisted `frame_valid`, `episode_valid`, or global incremental schema version. Keep freshness, causal selection and cadence checks in runtime; do not persist per-row timestamps, raw VR, detailed status, software provenance, or transport bookkeeping.

Finite action targets mean that target was published for that row; NaN means no corresponding action was published. Never substitute a previous target. Preserve an actual published hold. Publish only clean, temporally continuous fixed-dt teleop demonstrations: operator pause/resume is a Raw episode boundary, never an append across a wall-clock or control-state discontinuity. S/D finalize only the current capture, H only performs planned HOME, and Q uses explicit two-stage quit with no timeout. Technical control/publication failure, cadence discontinuity, recording failure or abnormal lifecycle invalidates the current capture; revoke motion promptly and discard it rather than silently streaming with a discard-only recorder. Recoverable failures require operator C to start a new segment with fresh resources. Abnormal cleanup never defaults to saving an unfinished teleop capture. Policy rollouts retain ordinary unexecutable steps as NaN-action rows and may save a healthy recorded prefix on abnormal termination with its reason. Do not delete policy-failure evidence to select successful rollouts.

Use a bounded local recorder FIFO with non-blocking submission and no additional full RGB-D copy. A single writer thread owns HDF5/PyAV from open through close. START readiness precedes motion authorization; blocking finalization follows motion revocation. Queue Full and writer errors are experiment failures, never silent drops.

Tactile payloads use XHand SDK-native values with software bias removed; their SI force unit is unverified. Native recording and Raw-to-Zarr apply no additional tactile scaling. Missing tactile payloads contain NaN; runtime aggregate/dense validity remains explicit and independent of joint state.

Snapshot the resolved physical hand mount at recording START; Raw is its sole offline truth source. New recording requires complete eye-to-hand aligned RGB-D calibration before START. The aligned color-grid distortion model must have a validated canonical depth-deprojection path. Keep this lightweight compatibility check shared by recording, Raw loading, and numerical deprojection; reject unsupported models before recording.

Canonical Zarr uses `format="dexmani.real.canonical"` and remains a general multimodal learning cache. Root attrs contain only format, task_name, dt, depth_scale_m_per_unit, complete pointcloud_config and fingertip_link_names. Preserve a single aligned-Z16 depth scale per store. Document array meanings in README; do not duplicate semantic dictionaries across export, training and deployment. The export report stores the resolved processing snapshot, including the audited training table plane. Deployment uses saved numerical point-cloud parameters and fingertip link choices with current Real calibration and physical hand mount. No runtime timing arrays or generic validity masks belong in Canonical.

Raw-to-Canonical export preflights all candidate episodes before writing. Reject and skip whole episodes with no rows, missing required files/fields, malformed Raw layouts/metadata, or NaN/Inf in any required floating array, including tactile/current/effort telemetry. Report rejection reasons and persist accepted/rejected/explicitly excluded episode lists in `export_report.json`; write no Zarr if none pass. Unsupported formats, incompatible task/dt/depth scale/array layouts, I/O errors and conversion failures remain fatal to the entire export. Do not repair, split, resample, drop bad rows or salvage partial episodes. Raw may retain missing telemetry as NaN; every floating output of the exporter must be finite. `action` and `action_ee` describe the same final target. `dexmani_policy` datasets remain domain-agnostic and load only selected arrays; canonical finiteness is the exporter’s responsibility. Deployment metadata capture saves only dt and consumed point-cloud numerical configuration/fingertip links. Canonical availability does not imply live Real availability: unsupported live modalities must fail before motion.

Canonical caches are regenerated from current Raw and only the current canonical format is supported. Published field names never silently change meaning; source code defines the numerical representation and README documents it. Persist numerical preprocessing choices without adding a semantic registry or deployment ABI. Additive fields do not invalidate consumers that do not request them. Do not introduce global schema-version compatibility chains or fabricate modalities that were never recorded.

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
