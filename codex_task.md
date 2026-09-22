# Codex Task — Simplify DexMani Real into a Research-First Real-Robot Experiment Stack

Status (2026-09-22): the architectural migration and offline verification are complete in the local working tree. Physical commissioning remains pending; offline checks do not validate hardware behavior. This file remains the design and acceptance specification. Sections 26–27 retain the original migration map and phase checklist, including names of modules that have since been deleted; they are not a current source inventory or outstanding implementation plan.

## 0. Purpose and execution contract

This file is the implementation specification for the local Codex CLI.

The repository is a personal PhD real-robot manipulation endpoint for:

1. collecting dexterous manipulation demonstrations;
2. deploying and evaluating policies trained in dexmani_policy or other external repositories.

It is not a production robot platform, distributed transaction system, generic robotics middleware, policy-serving service, or dataset framework.

The implementation target is intentionally research-first:

physical safety > experiment correctness > iteration speed > readability > generic extensibility > enterprise robustness.

Prefer:

delete > inline > merge > rewrite > add abstraction.

A mechanism earns its complexity only if it protects a current physical guarantee, a scientific-data guarantee, or an active experiment requirement.

Before editing:

1. Read AGENTS.md and README.md in full.
2. Run git status --short and git rev-parse HEAD.
3. Preserve unrelated local changes. Never reset, checkout, stash, clean, or overwrite user work.
4. Trace actual current definitions, producers, transformations, consumers, and side effects before editing.
5. Source code and resolved configuration are authoritative when documentation is stale.
6. Do not execute any hardware-affecting code. No live SDK discovery, device connection, motion, home, replay, teleoperation, camera capture, calibration writes, policy rollout, or physical XHand diagnostics without explicit user authorization.
7. Offline imports and constructors must not connect hardware.
8. Do not add or restore a committed tests directory. Use focused one-off offline smoke checks.
9. Do not install or upgrade the experiment environment merely to make checks pass.
10. Do not create generic replacement frameworks for concepts this task removes.
11. Complete the refactor instead of stopping after a plan unless blocked by missing source/dependencies or a required physical validation.
12. Do not claim hardware validation from offline checks.
13. Keep codex_task.md at repository root until the user explicitly asks to remove it.

The design below was reviewed against:

- dexmani_real implementation at cda8d201b00eed9018587aea42b4077b8d613645;
- dexmani_policy at 6ba48d7566bd565839a8cee9c14df32e402f8a65;
- xArm-Developer/lerobot_robot_ufactory at 5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82;
- Universal-Control/ManiUniCon at 85c6f2e32ecf9f2bed62d202b058c39623444686;
- pi-r2-flow/pi-r2-flow at 3af52ca400a6ec7d141416879aa531a6f62f697a;
- wengmister/LeFranX at a39906e6629f39490950fe8bd20f4f992ed74fd7.

Reference projects are simplicity references, not safety authorities. Preserve stronger hardware-boundary checks when they have clear physical value.

---

# 1. Repository mission

## 1.1 Teleoperation collection

Target conceptual flow:

    Quest / VR
        |
        v
    wrist mapping + hand retargeting
        |
        v
    optional EE workspace clip
        |
        v
    IK when needed
        |
        v
    absolute joint target preparation
        |
        v
    xArm7 + XHand
        |
        v
    timestamped robot state + tactile + RGB-D + VR + target
        |
        v
    raw research episode

## 1.2 Learned-policy evaluation

Target conceptual flow:

    external policy artifact
        |
        v
    PolicySpec
        |
        v
    current observation row + local observation history
        |
        v
    synchronous predict + local action chunk
        |
        v
    decode / optional EE workspace clip / IK
        |
        v
    absolute joint target preparation
        |
        v
    xArm7 + XHand
        |
        v
    evaluation telemetry + result

## 1.3 Explicit non-goals

Do not build:

- generic robot plugin systems;
- generic Sensor or Modality interfaces;
- distributed command transactions;
- command ACK/adoption ledgers;
- async inference or RTC;
- remote inference;
- future-timestamp action queues;
- generic safety-gate frameworks;
- generic validity frameworks;
- automatic data repair pipelines;
- policy-specific Zarr variants;
- runtime compatibility layers for old schemas unless explicitly requested.

---

# 2. Final architecture

The target runtime has asynchronous hardware producers, one high-level command mailbox, current-step observation assembly, and simple workflow-specific control.

    Teleop                         Policy
      |                              |
      | proposal                     | raw action
      +---------------+--------------+
                      |
                      v
             target interpretation
       map / retarget / optional IK
                      |
                      v
           absolute target preparation
           - finite
           - arm nearest 2pi equivalent
           - operational joint limits
           - EE workspace clip before IK only
                      |
                immutable target
                      |
          +-----------+-----------+
          |                       |
          v                       v
      raw recording          RobotCommand mailbox
                                  |
                     +------------+------------+
                     |                         |
                     v                         v
                Arm worker                Hand worker
                 xArm SDK                  XHand SDK
                     |                         |
                     +------------+------------+
                                  |
                                  v
                          state / sensor rings
                                  |
                                  v
                         current observation row
                                  |
                                  v
                        local observation deque

Offline:

    raw episode
        |
        +-- structural/training admission
        |
        +-- offline FK
        +-- offline fingertip geometry
        +-- offline point cloud
        |
        v
    one canonical full Zarr dataset

The main simplification principle is:

hardware stream -> control row -> raw episode -> canonical training cache.

Do not mix responsibilities across these layers.

---

# 3. Hard physical/runtime invariants

## 3.1 Resource ownership

Keep live SDK/runtime objects inside their owning processes:

- xArm SDK -> arm worker;
- XHand SDK -> hand worker;
- RealSense SDK -> camera worker;
- model/CUDA runtime -> policy worker.

Do not merge arm and hand into one worker merely for API symmetry.

## 3.2 Lifecycle state

Keep the simple SafetyState model:

- DISARMED;
- ARMED;
- RUNNING;
- FAULT.

Normal streaming RobotCommand admission is RUNNING-only.

Return-home is a dedicated ARMED operation, not a normal streaming RobotCommand.

Keep run_id as the stale-command lifecycle epoch.

run_id exists only to ensure old delayed work cannot regain motion authority after pause, stop, timeout, restart, or a new run.

Do not replace command_id with another identity under a new name.

## 3.3 Final software fence before SDK calls

Before a streaming SDK send, a worker must confirm:

- runtime is active;
- no estop/fault blocks motion;
- SafetyState is RUNNING;
- command run_id equals current run_id;
- target shape is correct;
- target is finite;
- target is inside the worker-owned physical/rated hard limits.

Then call the SDK.

Do not hold global lifecycle locks across SDK IO.

A vendor call that already crossed the software fence cannot be retracted. Do not create event ledgers to pretend otherwise.

## 3.4 Emergency stop and fault

Keep emergency stop, SDK error handling, safe disconnect, and fail-closed hardware behavior.

Worker-owned unexpected SDK errors remain runtime/hardware failures.

Recording or camera failures are not automatically hardware FAULT. They terminate or invalidate the experiment workflow while leaving safe operator recovery/home/quit available when hardware itself is healthy.

## 3.5 Shared-memory cleanup

Do not close/unlink shared memory while child processes may still access it.

Shutdown order remains:

    request shutdown
    join children
    terminate stragglers
    kill only if necessary
    confirm children stopped
    close/unlink shared memory

Do not add a large shutdown-report taxonomy.

---

# 4. Delete the command adoption system

The current command/adoption architecture is obsolete for this project.

Delete the following concepts throughout IPC, workers, workflows, recording, raw schema, replay, docs, and configuration:

- command_id;
- next_command_id;
- issued_monotonic_ns as command identity;
- single-inflight command admission;
- CommandAdoption;
- read_command_adoption;
- arm adoption identity;
- hand adoption identity;
- hand reached command identity;
- adoption timestamps;
- partial adoption;
- adoption unknown;
- adoption accounting timeout/barrier;
- pending_record_command_id;
- pending adoption/revocation bookkeeping;
- CommandHistory;
- command send masks;
- normal-row requirements based on actuator adoption;
- replay logic based on adoption identity.

Do not reintroduce the same concepts as:

- target_sequence_id;
- applied_id;
- accepted generation;
- execution identity;
- snapshot epoch;
- command generation.

The shared-memory ring sequence may remain as a private transport implementation detail, for example to let a worker know whether a newer mailbox value exists. It must not become scientific data or a public command identity.

---

# 5. RobotCommand semantics

Use one small current-run high-level target.

Conceptually:

    RobotCommand:
        run_id
        arm_present
        hand_present
        arm_qpos[7]
        hand_qpos[12]

arm_present and hand_present may remain if current workflows genuinely require arm-only/hand-disabled operation. They are runtime transport details and must not enter the raw training schema.

The mailbox semantic is latest target, not FIFO and not transaction.

A worker consumes the newest current-run target available to it.

If a newer target supersedes an older target before the worker consumes the older value, that is normal latest-target control behavior. Do not rebuild ACK infrastructure to prove every intermediate target crossed every SDK.

A lightweight skipped-mailbox counter is acceptable only as a local commissioning/debug statistic:

    if new_sequence > last_sequence + 1:
        skipped += new_sequence - last_sequence - 1

Such counters:

- do not block motion;
- do not enter raw episodes;
- do not enter Zarr;
- do not become command identity;
- may be printed in session/evaluation summaries.

---

# 6. Action semantics

## 6.1 Definition

The recorded high-level action is:

the absolute executable control target produced by the controller after high-level interpretation and simple target preparation, and published to the robot command boundary for that control step.

It is not:

- proof that both SDKs consumed it;
- proof that the hardware reached it;
- a physical actuation onset timestamp;
- a low-level XHand interpolated setpoint.

Prefer field names such as:

- action_arm_joint_target;
- action_hand_joint_target;

and Zarr canonical action for the concatenated 19-DoF target.

Do not use names that imply execution proof, such as sent, adopted, applied, or executed.

## 6.2 Target preparation is intentionally simple

Normal xArm target path:

    proposal
      -> finite
      -> nearest 2pi equivalent
      -> absolute operational joint-limit clip
      -> publish
      -> worker physical/rated hard-limit check
      -> xArm SDK velocity/acceleration controlled servo

Normal XHand target path:

    proposal / retarget output
      -> finite
      -> absolute operational joint-limit clip
      -> publish
      -> worker mechanical hard-limit check
      -> direct XHand absolute position SDK command

No high-level arm delta clip.

No high-level hand delta clip.

No XHand software slew.

No worker-side target interpolation.

Delete or retire configuration and logic such as:

- teleop_arm_max_delta_rad_per_tick;
- max_servo_command_jump_rad when used as normal streaming delta limiting;
- hand_max_delta_rad_per_tick;
- hand_max_sdk_step_rad if introduced;
- limit_hand_target_delta in the streaming path;
- previous-command delta shaping;
- last SDK accepted setpoint tracking used only for slew.

Keep xArm firmware speed/acceleration limits because they are actual actuator motion controls.

## 6.3 EE workspace

For teleop or EE-action policies only, keep a cheap Cartesian input-domain clip before IK:

    eef_position = clip(eef_position, workspace_min, workspace_max)

This is intent shaping, not a generic safety gate.

Joint-action policies do not run FK solely to impose a normal-runtime workspace gate.

## 6.4 Joint limits

Keep two semantic levels only where they truly differ:

- operational/controller limits may clip a proposed target into the intended experiment range;
- worker-owned physical/rated limits are a hard final fence and reject violations.

Do not create typed result frameworks around simple clipping.

---

# 7. Delete SafetyGate from normal runtime

Delete the generic normal-runtime SafetyGate abstraction and related machinery:

- SafetyGate;
- GateResult;
- GateRejectCode;
- planner_action_safety_gate;
- normal streaming workspace path checks;
- normal streaming collision transition checks;
- collision callback plumbing through teleop/eval/replay command dispatch.

Simple helpers for finite/shape/limit preparation are acceptable, but prefer plain functions over class hierarchies.

Do not replace SafetyGate with another generic gate under a different name.

---

# 8. Collision checking belongs only to return_home

Normal teleoperation, learned-policy evaluation, ordinary replay, and ordinary supervised manual operation do not run generic software collision checking.

Reason:

- the project is operator-supervised research;
- the hardware emergency stop is available;
- targets remain inside explicit actuator limits;
- xArm retains firmware velocity/acceleration controls;
- normal runtime should not be burdened by home-level geometric proof.

Collision checking is retained for planned return_home only.

Return-home may preserve the existing meaningful geometric complexity:

- nearest equivalent arm configuration;
- planned waypoints;
- self collision;
- arm-hand collision;
- table/static obstacle clearance;
- workspace constraints;
- safe path selection;
- Mode 0 execution;
- abort;
- final measured convergence;
- Mode 6 restoration.

Do not simplify away true return-home geometry merely because normal runtime no longer uses collision checks.

Calibration motion must not run collision checking. It may keep only simple procedure-specific non-collision bounds or known-geometry assumptions required by the calibration procedure. Do not reuse generic SafetyGate.

---

# 9. xArm worker and driver target

## 9.1 Keep

Preserve:

- xArm SDK ownership in arm worker;
- axis/DOF verification;
- SDK return-code handling;
- refusal of existing controller errors unless an explicit maintenance workflow says otherwise;
- collision sensitivity firmware setting if currently required;
- TCP load configuration;
- max joint velocity and acceleration configuration;
- Mode 6 streaming setup;
- Mode 0 return-home execution;
- finite hard-bound checks at worker boundary;
- best-effort stop/disconnect;
- measured qpos/qvel/effort publication.

Do not blindly copy UFACTORY clean_error startup behavior if current explicit refusal is safer for research commissioning.

## 9.2 Remove

Remove from arm state/worker integration when no longer used:

- command adoption state;
- command IDs;
- tracking-error raw fields that can be derived;
- connected bit;
- generic state_valid bit;
- publish timestamp used only for causal proof;
- generic heartbeat callbacks inside driver waits/home.

Arm state should approach:

    qpos
    qvel
    effort
    timestamp_ns

If the third get_joint_states output is not conclusively verified as physical Nm torque, prefer arm_effort over arm_tau in the new schema/documentation.

---

# 10. XHand worker and driver target

## 10.1 Keep

Preserve:

- XHand SDK ownership in hand worker;
- serial/EtherCAT support if currently used;
- fresh joint/current/tactile reads;
- explicit current_ma semantics;
- 12-joint completeness and finite parsing;
- aggregate tactile;
- dense tactile;
- independent aggregate/dense tactile validity;
- software no-contact tactile bias calibration;
- SDK error handling;
- mechanical hard-limit validation;
- board error decoding for diagnostics/logging where useful.

## 10.2 Remove

Delete:

- producer hand delta clipping;
- worker hand delta clipping;
- software slew;
- active-target interpolation;
- last_sdk_setpoint tracking used for slew;
- hand adoption/reach identity;
- connected;
- qpos_stale;
- generic state_valid;
- per-state board error arrays when they exist only for normal policy/raw transport;
- publish timestamp used only for causal proof.

The hand worker sends each new current-run absolute target directly through the XHand SDK, matching the simple behavior of the reference research implementations.

Hand state should approach:

    qpos
    current
    tactile_aggregate
    tactile_aggregate_valid
    tactile_dense
    tactile_dense_valid
    timestamp_ns

A whole hand joint read failure does not republish an old qpos with qpos_stale. Instead, publish no new hand sample. The old timestamp naturally becomes stale. Persistent failure transitions to workflow/runtime failure according to existing hardware-error handling.

Partial tactile failure is different and remains explicit because qpos/current may be valid while one tactile representation is not.

---

# 11. Validity and freshness model

Do not build a generic valid/invalid framework.

## 11.1 Whole samples

For arm, hand joint state, camera, VR, and point cloud:

- producer publishes only a structurally usable sample;
- producer does not publish a new sample when acquisition/derivation fails;
- consumer checks existence and freshness;
- persistent required-source failure ends or blocks the workflow as appropriate.

Delete generic flags such as:

- connected;
- state_valid;
- qpos_stale;
- camera_valid;
- camera_health;
- camera_generation;
- pointcloud_valid;
- observation_valid.

## 11.2 Freshness

Keep one host monotonic timestamp per produced sample.

A consumer uses a simple age check:

    now_ns - sample.timestamp_ns <= max_age_ns

Do not require exact cross-modal synchronization.

Delete normal-runtime max_observation_skew hard gating.

Do not recreate source -> receive -> publish -> commit -> anchor proof chains.

## 11.3 Tactile partial validity

Keep only the explicit XHand partial-modality validity that has real sensor meaning:

- tactile aggregate valid;
- tactile dense valid.

Invalid tactile is not zero contact.

IPC may use fixed payload buffers plus validity bits.

Raw may store NaN for invalid tactile payloads plus validity.

Canonical Zarr contains no validity mask because only fully valid episodes enter it.

---

# 12. Camera and point cloud simplification

## 12.1 Camera

Camera worker publishes aligned usable RGB-D frames only.

Use a host monotonic receive/acquisition-completion timestamp for runtime freshness.

Remove DeviceClockMapper from online correctness logic.

Remove camera_generation and CameraHealth protocol from normal runtime.

Raw camera device timestamp/frame number may be retained only if it is cheap and clearly useful for offline sensor analysis, but it must not drive runtime gating.

RGB and depth in one sample must come from the same aligned frame.

## 12.2 Point cloud

Keep point cloud online only when a deployed policy actually requires it.

Point-cloud worker publishes only valid derived clouds.

No generic pointcloud_valid field.

When a policy simultaneously consumes RGB and point cloud, preserve one simple source_camera_sequence identity so both can be selected from the same RGB-D source frame.

Do not generalize this identity into a multimodal synchronization framework.

For demonstration storage, raw episodes store RGB-D and calibration. Point clouds are derived offline for Zarr.

---

# 13. Observation assembly

Replace historical causal ring reconstruction with a current-step snapshot plus a local temporal deque.

## 13.1 Current ObservationRow

At each real control step:

1. copy the latest required arm sample;
2. copy the latest required hand sample;
3. copy the latest required camera sample when required;
4. copy the latest required VR sample for teleop;
5. copy the latest required point cloud when the current policy requires it;
6. verify each required sample is fresh;
7. if RGB and point cloud are both required, ensure source_camera_sequence matches;
8. after the copies are complete, set observation_timestamp_ns = time.monotonic_ns().

Because all samples were copied before observation_timestamp_ns, they were available when the observation snapshot was completed. No sequence-ceiling or publish-timestamp causal proof is required.

If a required current sample is unavailable/stale, return no ObservationRow and handle it at the workflow level. Do not return ObservationRow(valid=False).

## 13.2 One snapshot, reused

For teleop, the same immutable current observation snapshot must be reused by:

- controller/retarget/IK;
- target preparation;
- raw recording.

Do not reread arm/hand/camera separately for recording.

For policy evaluation, distinguish:

- the model query observation history;
- the later current execution telemetry while consuming an already predicted action chunk.

Policy rollout is evaluation telemetry by default, not behavior-cloning source data.

## 13.3 Observation history

Use a local deque with maxlen equal to n_obs_steps.

Every real control step with a valid current observation appends one row.

Policy input stacks deque rows.

Do not reconstruct historical rows by searching sensor rings backward at inference time.

Warm-up uses edge padding consistent with dexmani_policy sampling:

    first row -> [r0, r0, ..., r0]
    next row  -> [..., r0, r1]

On pause/resume or a clearly large control gap, clear history and edge-pad from the new current row.

Blocking synchronous inference does not synthesize missed control rows.

---

# 14. Teleoperation behavior

Keep current operator controls, especially C.

Expected controls remain conceptually:

- B begin;
- C pause/resume;
- S stop/save;
- D discard;
- H return-home;
- Q quit;
- ESC emergency stop.

## 14.1 C pause/resume

C must not be deleted or reinterpreted.

Pause:

- RUNNING -> ARMED;
- advance/fence run_id so old delayed commands become stale;
- clear unpublished local intent/chunks;
- clear observation history;
- retain the same raw episode if the user later resumes.

Resume:

- require fresh post-pause robot and VR feedback;
- re-anchor teleop state from current measured robot/VR state;
- start a new motion epoch/run_id;
- restart observation history from the fresh current row.

Do not create adoption/revocation evidence.

A paused-and-resumed raw episode is allowed to exist for diagnosis/review, but it is not a clean training episode because its timing contains a large gap. Canonical Zarr export must reject the entire episode and report that reason.

## 14.2 Recording requirements

In normal recording mode:

- camera and recorder are required before B;
- never silently downgrade B into unrecorded teleop;
- only explicit --no-record allows unrecorded teleop.

If camera or recorder fails during a recording:

- stop/revoke further streaming motion;
- close the recording safely when possible;
- mark/store it as incomplete/diagnostic according to existing storage conventions;
- do not claim hardware FAULT unless hardware itself failed;
- block further recorded B in the same broken recording session if that is the simplest current behavior;
- H/Q may remain available if hardware is healthy.

---

# 15. Learned-policy evaluation

Keep synchronous inference and local action chunks.

Do not add:

- async prediction;
- background policy queries;
- RTC;
- remote inference;
- action merging;
- future action timestamp queues;
- catch-up bursts.

At each execution step:

1. read a fresh current ObservationRow;
2. append it to local history;
3. when action chunk is empty, snapshot history and captured run_id;
4. call synchronous predict;
5. if run_id changed while predict blocked, discard the returned chunk;
6. validate chunk shape/finiteness/action representation;
7. pop the next raw policy action;
8. if EE action, apply simple EE workspace clip then IK;
9. if joint action, use joint values directly;
10. apply absolute operational joint limit preparation;
11. publish latest RobotCommand;
12. schedule the next high-level action relative to actual successful publication/control timing; never burst to catch up.

Do not wait for actuator adoption before the next control period.

For chunked policies, each future chunk action is interpreted/prepared at its own execution step. Do not add hidden high-level delta filters.

Evaluation summaries may contain small useful metrics such as:

- inference latency;
- number/fraction of absolute-limit clips;
- maximum clip magnitude;
- worker skipped-mailbox target counts;
- termination reason;
- task success annotation.

Do not build per-row runtime audit schemas for these metrics.

---

# 16. Raw episode v32

Create a new raw schema version rather than reinterpreting v31.

Old v31 adoption semantics are obsolete. No runtime backward-compatibility branch is required unless explicitly requested.

Raw is the experimental source of truth. It should describe what the experiment observed and commanded, not how the Python runtime proved its own protocol.

## 16.1 Recommended core dynamic fields

Use names that preserve physical meaning. Final exact naming may follow existing conventions, but semantics must match:

Timing:

- observation_timestamp_ns;
- action_timestamp_ns;
- arm_timestamp_ns;
- hand_timestamp_ns;
- camera_timestamp_ns;
- vr_timestamp_ns.

Arm:

- arm_qpos [7];
- arm_qvel [7];
- arm_effort [7] unless torque units are conclusively verified.

Hand:

- hand_qpos [12];
- hand_current [12].

Tactile:

- hand_contact [5,3];
- hand_contact_valid;
- hand_tactile_force [5,120,3];
- hand_tactile_force_valid.

Action:

- action_arm_joint_target [7];
- action_hand_joint_target [12].

Optional controller intent:

- arm_eef_intent [9] for teleop or EE policy pre-IK Cartesian intent;
- absent/NaN when not applicable.

Teleop source:

- VR wrist pose;
- VR hand landmarks;
- only additional VR fields that are genuinely needed for reconstruction/analysis.

Frame status:

Keep a very small diagnostic enum such as:

- FRAME_OK;
- FRAME_IK_FAIL;
- FRAME_RETARGET_FAIL;
- optionally one invalid-policy-output status if current policy evaluation needs it.

Delete:

- FRAME_PARTIAL_ADOPTION;
- FRAME_ADOPTION_UNKNOWN;
- command-adoption status;
- generic observation_valid;
- action_valid duplicate masks;
- camera health/generation fields;
- worker connected/state_valid fields.

For a failed controller row that has no valid target, write a diagnostic status and do not pretend the previous action was the current target. Prefer NaN/absent action payload semantics that make accidental training use fail visibly.

## 16.2 Camera storage

Keep simple raw storage, for example current RGB video plus depth HDF5 if that remains easiest.

Every normal recorded control row must have a usable camera sample.

Do not use zero-image placeholders to make a broken recording look complete.

Static calibration and camera identity/depth scale remain episode metadata.

When multiple raw episodes are exported into one canonical Zarr, batch-level static semantics must agree: image geometry/resolution, camera intrinsics/extrinsics, depth scale, nominal control dt, joint ordering, and other fixed representation-defining metadata. Reject a mixed batch with a clear reason rather than broadcasting per-episode calibration to every row or silently combining incompatible semantics.

## 16.3 Timing

Raw timestamps are actual monotonic experiment timing.

Do not require exact fixed-dt equality. Normal scheduler jitter is allowed.

Raw time exists for analysis and export admission.

---

# 17. Canonical full Zarr dataset

Do not create policy-specific Zarr exports.

Build one fixed, full canonical training dataset so different dexmani_policy experiments can select sensor_modalities at load time.

Zarr is a disposable/rebuildable training cache. Raw episodes remain the source.

## 17.1 Canonical dynamic arrays

The full dataset should include all learning-relevant dynamic fields, not runtime bookkeeping.

Recommended data arrays:

- joint_state [T,19];
- arm_qvel [T,7];
- arm_effort [T,7];
- hand_current [T,12];
- contact_force [T,5,3];
- tactile_force [T,5,120,3];
- eef_pose [T,9];
- fingertip_points [T,5,3];
- rgb [T,H,W,3];
- depth [T,H,W];
- point_cloud [T,N,6];
- action [T,19];
- action_ee [T,21].

Do not add VR human-source arrays to canonical policy Zarr unless a real current policy consumes them.

## 17.2 Canonical action semantics

action:

    [arm executable joint target 7,
     hand executable joint target 12]

action_ee:

    [FK of executable arm joint target as pos3 + rot6d6,
     hand executable joint target 12]

Thus action and action_ee are equivalent representations of the same high-level executable target.

Do not use pre-IK EE intent as action_ee.

Raw arm_eef_intent, when present, is separate research provenance.

This preserves compatibility with dexmani_policy, which currently treats joint_state/action/action_ee as canonical data concepts.

## 17.3 Static metadata

Do not broadcast static calibration per frame.

Store once in metadata/attrs where needed. This assumes batch-level static semantics were checked consistent during export:

- camera intrinsic;
- camera extrinsic;
- depth scale;
- joint order;
- tactile finger/point ordering;
- coordinate frames;
- action semantics;
- point-cloud frame;
- nominal dt;
- task/schema metadata.

## 17.4 No time arrays

Canonical Zarr must not contain runtime timing arrays such as:

- observation timestamp;
- source timestamps;
- publish timestamps;
- camera device time.

Keep timing in raw only.

Use nominal dt as dataset metadata.

## 17.5 No validity masks

Canonical Zarr must not contain:

- contact_force_valid;
- tactile_force_valid;
- observation_valid;
- action_valid;
- frame status.

Reason: only completely training-eligible raw episodes enter canonical Zarr.

Every Zarr row is assumed usable for all canonical modalities.

---

# 18. Raw episode admission to Zarr is all-or-nothing

This is a hard requirement.

The exporter must never repair, split, crop around, resample, mask, or silently salvage an abnormal raw episode.

A raw episode either enters Zarr in full, preserving row count/order, or the whole episode is rejected with a concrete reason.

Do not:

- drop individual invalid rows;
- split one raw episode into multiple training episodes;
- remove pause regions;
- stitch rows around a failed frame;
- resample irregular timing;
- fill invalid tactile with zeros;
- silently ignore camera gaps;
- convert incomplete recordings into valid training trajectories.

## 18.1 Episode admission checks

Keep the admission checks minimal and directly tied to training semantics.

A complete teleop raw episode is accepted only if, at minimum:

1. storage is structurally readable and current-schema;
2. recording completed normally, not incomplete;
3. workflow is training-eligible teleop, not policy_eval telemetry;
4. there is at least one row;
5. all required dynamic arrays have correct shape/dtype/frame count;
6. all required numeric state/action values are finite;
7. every frame_status is FRAME_OK;
8. all aggregate tactile rows are valid;
9. all dense tactile rows are valid;
10. RGB/depth frame count and geometry match the row count;
11. observation/action timing is strictly increasing where applicable;
12. there is no pause/missing-control gap large enough to invalidate fixed-rate trajectory semantics;
13. derived offline point cloud/FK/fingertip processing succeeds for all rows.

Normal OS jitter is allowed.

Do not require exact interval equality to nominal dt.

Use one simple, documented gap threshold, preferably approximately 2 times nominal dt unless existing measured workloads justify a different simple constant.

Do not build a timing-quality framework.

## 18.2 Rejection reporting

When exporting a task directory containing multiple episodes:

- accepted episodes may be appended;
- rejected episodes are skipped as whole episodes;
- every rejection must be visible with an explicit reason;
- final summary must report accepted/rejected counts and reasons.

A small export report file is acceptable if it is simpler than relying only on terminal logs.

Do not put rejection status inside training data arrays.

When exporting one explicit episode, rejection should fail clearly/nonzero.

## 18.3 Whole-episode annotations

Existing explicit operator annotations may still exclude an entire episode or override task metadata if that is already useful.

Do not add per-frame curation machinery.

---

# 19. Zarr validation should be lightweight

After writing, validate only low-cost structural facts:

- required arrays exist;
- dtype/shape match the canonical contract;
- first dimension N is identical;
- episode_ends is strictly increasing;
- episode_ends[-1] == N;
- metadata required by dexmani_policy is present.

Do not reopen and re-prove every point-cloud bound, every timestamp relation, every tactile invariant, or every transform that the exporter just computed.

Transformation functions should validate their own direct inputs/outputs once at the owning boundary.

Deployment/model compatibility should focus on true type/semantic errors:

- required field names;
- shapes;
- dtypes;
- joint ordering;
- units;
- coordinate frames;
- point-cloud point count/frame;
- action representation;
- tactile ordering when used.

Do not make exact serialized processing-config equality a generic runtime blocker.

Full processing config belongs in experiment/checkpoint provenance.

---

# 20. Recorder simplification

Keep a separate recorder process because RGB-D/video/HDF5 IO should not block the control loop.

Target recording protocol is conceptually:

    START
    sample stream
    STOP
    flush / close / atomic finalize
    RESULT

Keep staging -> final rename because it cheaply prevents incomplete data from masquerading as complete data.

Delete recorder dependencies on command adoption/accounting.

Recorder failures are workflow failures, not automatically hardware FAULT.

Review CameraStreamWriter or equivalent nested asynchronous writer. The recorder process already provides asynchronous isolation from control. If a simple offline throughput benchmark shows synchronous RGB/depth writing in RecorderIO comfortably exceeds collection rate, delete the second internal writer thread/queue and its queue-full/finalization complexity. If benchmark evidence shows it is required, keep it. Do not decide this by aesthetics alone.

---

# 21. Return-home

Return-home is the one intentionally conservative motion path.

Keep it separate from streaming RobotCommand.

Conceptual flow:

    operator H
       |
       v
    ARMED
       |
       v
    latest fresh arm + hand state
       |
       v
    plan safe path
       |
       v
    arm home request
       |
       v
    arm worker Mode 0 execution
       |
       v
    final measured qpos/qvel convergence
       |
       v
    restore Mode 6
       |
       v
    HomeResult(ok, reason)

Delete transaction-style home proof:

- arm_home_completed_run_id;
- command adoption/reach identity;
- duplicate caller-side proof if the owning worker already verified measured final convergence;
- oversized HomeStatus taxonomies when callers only need success/failure + reason.

A small HomeResult is enough.

XHand home sends the validated absolute home target directly. It succeeds when the owning worker receives SDK send success; queue publication alone is not success. Do not wait for measured convergence or add a position tolerance. SDK errors still fail the operation. Do not reintroduce slew or adoption/reach identity.

---

# 22. Replay and calibration

## 22.1 Replay

Replay uses recorded absolute high-level targets.

Delete adoption/send-mask semantics.

Replay should retain only genuinely useful protections:

- compatible schema/workflow;
- current measured start-state sanity;
- action finiteness/limits;
- lifecycle/run_id fencing;
- operator supervision/estop;
- simple schema/start-state/absolute-limit preflight only. Do not run collision preflight; collision checking belongs to return_home only.

Normal per-step generic collision SafetyGate is not used.

Do not reproduce recorded operator pause gaps by default unless the replay mode explicitly promises wall-time-faithful playback.

## 22.2 Calibration

Calibration should use narrow procedure-specific motion constraints.

Do not retain generic SafetyGate merely because calibration once consumed it.

Keep only the checks that calibration genuinely requires, such as explicit workspace target bounds or a fixed known hand-geometry assumption.

---

# 23. Process health and supervisor simplification

Remove generic heartbeat/health taxonomies where actual process liveness plus latest-sample age is sufficient.

For sensor/robot workers, use:

- process.is_alive();
- latest required sample timestamp/freshness;
- explicit hardware error/fault state.

Replace string-indexed generic readiness registries with explicit Events if that is simpler:

- arm_ready;
- hand_ready;
- camera_ready;
- vr_ready;
- recorder_ready;
- policy_ready;
- pointcloud_ready only when process exists.

Do not preserve:

- generic heartbeat registry;
- camera health state machine;
- evidence_failed/session_failed/recorder_transport_failed layers when one local workflow failure state is enough;
- large generic failure taxonomies.

A child worker that catches a fatal unexpected exception should log, signal fatal error if needed, and exit nonzero rather than silently returning success after setting multiple state flags.

---

# 24. Configuration cleanup

Only remove/move configuration that corresponds to deleted semantics or a real source-of-truth problem.

Delete obsolete normal-runtime configuration such as:

- adoption/accounting timeouts;
- hand_max_delta_rad_per_tick;
- teleop_arm_max_delta_rad_per_tick;
- normal streaming max_servo_command_jump_rad when it only exists for software delta clipping;
- max_observation_skew_s hard-gate config;
- generic collision enable/check config for normal teleop/eval;
- heartbeat fields rendered redundant by process liveness/sample freshness.

Keep actual hardware settings:

- xArm IP;
- xArm firmware speed/acceleration;
- collision sensitivity firmware parameter if still intentionally configured;
- TCP load;
- hard/operational joint limits;
- XHand connection settings;
- tactile calibration settings;
- camera settings;
- freshness thresholds;
- return-home planning parameters;
- teleop control_hz;
- policy control_dt from PolicySpec;
- recording paths/settings.

Avoid broad cosmetic config churn unrelated to deleted semantics.

---

# 25. Documentation cleanup

README.md, AGENTS.md, examples/help text, source docstrings, comments, and config comments must be consistent with the final design.

Remove stale claims about:

- command_id;
- single-inflight;
- actuator adoption;
- partial/unknown adoption;
- hand reached IDs;
- XHand bounded slew;
- normal-runtime collision checking;
- generic SafetyGate;
- observation causal-proof chains;
- raw v31 adoption semantics;
- Zarr timing arrays;
- Zarr validity masks;
- automatic row segmentation/salvage.

README should describe stable user-facing workflows, not internal implementation audit details.

AGENTS should state the new real physical guarantees clearly:

- finite SDK inputs;
- physical/rated actuator limits;
- emergency stop;
- xArm firmware speed/acceleration controls;
- SDK error handling;
- run_id stale-command fencing;
- operator supervision;
- collision planning for return_home;
- EE workspace clip for Cartesian control;
- child-stop-before-SHM-release;
- sensor freshness and tactile partial validity.

---

# 26. File-level migration map

Historical migration map, retained to explain the original scope. Check current source and dependencies before further edits; several listed modules no longer exist.

Original major files:

## IPC/runtime

dexmani_real/ipc/schema.py

- shrink RobotCommand dtype;
- simplify arm/hand state dtypes;
- remove adoption/reached/state_valid/connected/qpos_stale/publish/generation fields no longer needed.

dexmani_real/ipc/channels.py

- remove shared command-ID allocator/accounting fields;
- remove redundant health/heartbeat control fields;
- retain latest rings and explicit lifecycle primitives.

dexmani_real/runtime/safety.py

- keep SafetyState/run_id/estop/lifecycle fencing;
- remove coupled-command/adoption-specific helpers.

dexmani_real/runtime/supervisor.py and process startup code

- simplify readiness/health/failure handling;
- preserve safe shutdown ordering.

## Robot command path

dexmani_real/robot/commands.py

- remove CommandAdoption, AcceptanceResult, SafetyGate, GateResult, GateRejectCode, planner gate, adoption waits;
- reduce to small target preparation/publication/freshness helpers or split/delete the file if simpler.

dexmani_real/robot/projection.py

- keep only simple arm nearest-equivalent + absolute limit preparation and hand absolute limit preparation;
- remove normal high-level delta limiting/report frameworks.

dexmani_real/utils/feedback.py

- delete or shrink heavily; generic feedback issue taxonomy should not survive if simple latest+fresh checks replace it.

dexmani_real/robot/arm_worker.py

- remove adoption metadata;
- publish compact arm state;
- consume latest current-run target directly.

dexmani_real/robot/hand_worker.py

- remove adoption/reached metadata and software slew;
- direct-send new absolute current-run target;
- publish compact hand/tactile state.

dexmani_real/robot/drivers/xarm7.py

- preserve real hardware-boundary checks;
- remove runtime heartbeat callbacks/protocol coupling.

dexmani_real/robot/drivers/xhand.py

- preserve sensor/error/tactile capabilities;
- do not add normal command slew.

## Teleop

dexmani_real/teleop/control_loop/action_proposal.py

- keep VR mapping/EMA/retarget/IK intent logic;
- remove high-level arm/hand delta clipping;
- keep simple EE workspace clip where appropriate.

dexmani_real/teleop/control_loop/grid.py

- assemble one current observation snapshot;
- prepare absolute target;
- publish latest command;
- record same snapshot/target;
- keep C behavior;
- remove adoption wait/SafetyGate/complex observation validity.

dexmani_real/teleop/episode_samples.py

- remove generic observation_valid/cross-modal skew proof;
- make raw frame construction direct.

dexmani_real/teleop/homing.py and robot home modules

- simplify request/result semantics while preserving real geometry and measured convergence.

## Policy deployment

dexmani_real/deployment/observation.py

- rewrite around current observation row + deque;
- remove historical causal ring reconstruction, generation and timestamp-order proof machinery.

dexmani_real/deployment/runner.py

- remove adoption gating and SafetyGate;
- keep synchronous inference, local chunk, run_id fence, cadence, simple clip metrics.

dexmani_real/deployment/session.py

- simplify health/failure orchestration accordingly.

## Recording/raw

dexmani_real/recording/frame.py

- define raw v32 data semantics.

dexmani_real/recording/storage/schema.py

- remove adoption statuses/fields;
- define compact frame status and raw v32.

dexmani_real/recording/client.py

- remove adoption accounting and pending-command logic.

dexmani_real/recording/io_worker.py and recorder/storage writers

- simplify transaction protocol;
- benchmark nested camera writer thread before deciding whether to delete it.

dexmani_real/recording/storage/reader.py

- structural raw reader; no command-history reconstruction.

## Dataset

dexmani_real/dataset/contracts.py

- fixed full canonical Zarr schema;
- remove timing arrays and validity masks;
- static calibration stored once.

dexmani_real/dataset/processing.py

- whole-episode admission;
- no row repair/split;
- offline FK/fingertips/pointcloud;
- all rows preserved for accepted episodes.

dexmani_real/dataset/export.py

- all-or-nothing episode append;
- explicit rejection reporting;
- lightweight final validation.

dexmani_real/dataset/provenance.py

- keep only provenance with actual experimental value.

## Replay/calibration

dexmani_real/replay/*

- remove adoption/send-mask/SafetyGate dependencies;
- use recorded absolute targets and simple lifecycle/limit checks.

dexmani_real/calibration/*

- remove generic SafetyGate dependencies;
- keep procedure-specific bounds/protections.

## Docs/examples

README.md
AGENTS.md
examples/collect_teleop.py
examples/run_policy.py
examples/replay_episode.py
examples/export_policy_zarr.py

- remove obsolete terminology and behavior claims.

---

# 27. Migration phases

Completed migration checklist, retained in dependency order for acceptance and regression review. Hardware commissioning is tracked separately in section 31.

## Phase 0 — Freeze the new contracts

Before broad code edits:

- update task/docs terminology;
- define final RobotCommand semantic;
- define compact arm/hand state semantics;
- define current ObservationRow semantics;
- define raw v32;
- define canonical full Zarr;
- define all-or-nothing raw admission.

Do not write compatibility adapters for v31 unless explicitly requested.

## Phase 1 — Remove adoption and simplify command transport

Modify both producer and worker sides together.

- shrink RobotCommand;
- remove command IDs and adoption/reach state;
- latest-target mailbox;
- run_id stale fence;
- optional skipped-target debug counters;
- remove single-inflight gating.

At end of phase, teleop/policy/replay may still have old higher-level code, but the core command transport must no longer depend on adoption.

## Phase 2 — Simplify target preparation and remove SafetyGate

- delete generic SafetyGate;
- remove normal collision/workspace path checks;
- remove arm delta clip;
- remove hand delta clip;
- remove XHand slew;
- retain absolute limits and EE workspace clip before IK;
- keep worker hard physical fences.

Trace all call sites before deleting configuration.

## Phase 3 — Simplify arm/hand state and health

- compact state dtypes;
- no republished stale hand qpos;
- no connected/state_valid/qpos_stale;
- freshness replaces generic validity;
- tactile partial validity retained;
- remove adoption/reached fields.

## Phase 4 — Rebuild teleop current-step observation

- one immutable current snapshot per control step;
- same snapshot drives control and recording;
- per-modality freshness only;
- C pause/resume re-anchor preserved;
- remove cross-modal skew hard gate and observation_valid.

## Phase 5 — Raw v32

- migrate recorder/frame/schema/reader;
- action target names/semantics;
- observation/action/source timestamps;
- small frame status;
- no command-adoption fields;
- no fake previous action on failed controller rows;
- RGB-D remains raw source.

## Phase 6 — Policy observation and runner

- current row + deque;
- edge padding;
- history reset after pause/gap;
- synchronous inference only;
- run_id stale-return discard;
- no adoption wait;
- no catch-up bursts;
- evaluation summary metrics only.

## Phase 7 — Canonical full Zarr

- full fixed schema;
- no timing arrays;
- no validity masks;
- static calibration once;
- canonical action/action_ee semantics;
- preserve dexmani_policy loading compatibility.

## Phase 8 — All-or-nothing episode admission

- reject entire abnormal raw episode;
- no row split/salvage/resample;
- explicit rejection reason;
- accepted episodes preserve every row;
- lightweight export report.

## Phase 9 — Return-home, replay, calibration

- home retains geometry but loses transaction proof;
- replay loses adoption/send masks;
- calibration loses generic SafetyGate.

## Phase 10 — Supervisor/recorder cleanup

- remove obsolete health/heartbeat/accounting state;
- simplify recorder protocol;
- benchmark nested camera writer async layer and delete it if unnecessary.

## Phase 11 — Dead-code/config/doc cleanup

Repository-wide search and remove stale terms/config/imports.

No old compatibility layer should remain solely to make obsolete tests/docs pass.

---

# 28. Anti-regression rules

The refactor is not complete if any removed concept reappears under a new name.

Do not introduce a new field/class whose primary purpose is to prove an internal runtime event rather than describe:

- physical robot state;
- real sensor state;
- lifecycle authority;
- experimental timing;
- training data;
- a true hardware safety boundary.

Specifically prohibit reintroduction of:

- command ACK ledgers;
- target application IDs;
- generic validity masks;
- generic safety gates;
- normal-runtime collision frameworks;
- XHand software slew;
- arm/hand high-level delta clipping;
- policy-specific Zarr variants;
- automatic raw row salvage.

When uncertain, prefer fewer states that need checking.

---

# 29. Offline validation

Do not add a permanent test framework.

Use focused one-off pure-logic checks for changed code, especially:

- arm nearest-equivalent angle handling;
- arm/hand absolute operational limit clipping;
- hard-limit rejection helpers;
- VR mapping;
- retargeting;
- IK/FK;
- ObservationHistory edge padding/reset;
- latest+fresh observation selection;
- RGB/pointcloud source identity when both required;
- raw v32 shape/dtype/read/write;
- all-or-nothing episode rejection reasons;
- canonical Zarr full schema;
- action/action_ee equivalence;
- no Zarr timing arrays or validity masks;
- point-cloud/FK/fingertip offline derivation;
- return-home planning pure geometry.

Run low-cost repository checks:

    python -m compileall -q dexmani_real examples
    ruff check --select F401,F821,F822,F823 dexmani_real examples
    git diff --check

If Ruff or optional dependencies are unavailable, report them instead of modifying the environment.

Do not execute hardware entry points as tests.

---

# 30. Repository-wide static acceptance searches

Before handoff, search for stale concepts.

Expected to disappear from normal runtime/docs, except migration notes if absolutely necessary:

- command_id;
- CommandAdoption;
- last_adopted;
- last_reached;
- PARTIAL_ADOPTION;
- ADOPTION_UNKNOWN;
- single-inflight;
- hand_max_delta_rad_per_tick;
- teleop_arm_max_delta_rad_per_tick;
- normal-runtime bounded slew;
- SafetyGate;
- GateRejectCode;
- normal-runtime collision_check;
- observation_valid;
- qpos_stale;
- generic state_valid;
- camera_generation;
- DeviceClockMapper online use;
- Zarr contact_force_valid;
- Zarr tactile_force_valid;
- Zarr observation/source timing arrays.

Do not mechanically delete a symbol that remains necessary for a different, clearly justified purpose. Trace each occurrence.

---

# 31. Physical commissioning after offline refactor

Do not perform these checks without explicit user authorization.

The refactor should leave a short commissioning list rather than trying to encode all behavior in software proofs.

Measure/verify:

- xArm streaming smoothness using SDK velocity/acceleration controls after removing software delta clip;
- XHand direct absolute-target behavior after removing all software slew;
- current spikes/jerk/tracking under representative hand targets;
- C pause/resume stale-command fencing and re-anchor;
- fresh-state behavior when arm/hand/camera/VR pauses or disconnects;
- return-home collision-safe path and final convergence;
- S/D/Q/H/ESC behavior;
- slow policy inference and run_id stale-return discard;
- no action catch-up bursts;
- RealSense/recorder failure handling;
- tactile aggregate/dense partial failure behavior;
- raw v32 recording;
- full Zarr export and whole-episode rejection reporting;
- representative dexmani_policy training load;
- representative real policy rollout;
- physical emergency stop and safe shutdown.

Offline correctness is not physical validation.

---

# 32. Definition of done

The task is complete only when all of the following are true:

1. command/adoption accounting is removed end-to-end;
2. normal RobotCommand is latest-target + run_id only;
3. xArm has no high-level software delta clip;
4. XHand has no software delta clip or slew;
5. normal teleop/eval/replay do not run generic collision checks;
6. collision planning remains for return_home;
7. generic SafetyGate is removed;
8. whole-sample validity flags are replaced by publish-only-usable + freshness;
9. tactile aggregate/dense validity remains in IPC/raw only;
10. observation history uses current control rows + deque;
11. teleop control and recording reuse the same current observation snapshot;
12. C pause/resume remains intact and uses run_id + fresh re-anchor;
13. raw v32 contains scientific state/action/timing without command-adoption audit fields;
14. raw action means high-level absolute executable control target, not SDK execution proof;
15. canonical Zarr is full and fixed, not policy-specific;
16. Zarr contains no timing arrays or validity masks;
17. Zarr action/action_ee describe equivalent final target representations;
18. abnormal raw episodes are rejected whole with explicit reasons;
19. exporter performs no silent row repair, split, resample, or salvage;
20. policy_eval telemetry is not silently exported as teleop BC data;
21. recorder remains isolated from control, with no adoption transaction protocol;
22. README/AGENTS/examples/comments no longer assert obsolete behavior;
23. obsolete config/state/imports are deleted;
24. offline checks pass or missing optional tooling is clearly reported;
25. no hardware validation is claimed.

The intended final character of the repository is:

simple enough to understand in one pass,
strong at real hardware boundaries,
honest about asynchronous sensors and commanded targets,
and optimized for rapid, reproducible PhD real-robot experiments rather than production-style protocol proof.
