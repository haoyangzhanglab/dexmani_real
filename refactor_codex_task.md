# Codex Task — DexMani Real Architecture and Readability Refactor

## 0. Task intent

Refactor dexmani_real into a clearer, more direct, paper-quality real-robot research codebase while preserving current physical-safety, experiment, data, and runtime behavior.

This task is based on four full repository reviews plus comparison with three reference codebases:

- ManiUniCon: borrow clear resource/lifecycle ownership, but do not copy its generic Base*/factory/framework style.
- RISE: borrow direct paper-code organization and easily traceable experiment entry points.
- pi-r2-flow: borrow the separation of pure control helpers from orchestration, but avoid giant stateful runtime scripts.

The target style is:

> RISE-like directness + ManiUniCon-like resource ownership + pi-r2-flow-like pure helpers, without a generic robotics framework.

dexmani_real is a personal PhD real-robot research repository. It is not a production robotics platform. The desired result is a codebase where a new robotics researcher can trace:

    CLI -> experiment session -> process worker -> runner/controller -> driver/numerical logic

without having to infer hidden state, dynamic delegation, duplicated configuration, or framework machinery.

This is primarily a behavior-preserving architecture/readability refactor. Do not mix it with algorithm changes, new control policies, new safety semantics, or new data schemas.

The task was reviewed against main commit:

    268d9979bb4c725444b1632c52be646fb03e32ce

Before editing, inspect the current tree and adapt to any newer commits without reverting unrelated work.

### Execution scope

Treat Phases 1-8 as the core refactor. Complete them in order unless the current source proves that a proposed change no longer reduces complexity or would violate an invariant. If that happens, keep the simpler current design and report the specific reason instead of forcing the target shape.

Phases 9-10 are conditional cleanup. Perform them only when they clearly reduce complexity with a small, mechanically reviewable diff. By default, do **not** relocate the examples directory or the offline smoke suite merely to match the illustrative target tree.

The reference repositories are design references only. Do not copy source code, add their dependencies, or reproduce their framework layers.

Keep the repository importable and internally consistent at the end of every phase. Do not carry temporary old/new APIs across phases. Update all in-repository call sites before running that phase's checks.

Unless explicitly requested by the caller, do not create commits, branches, or pull requests. Leave the implementation in the current working tree for review.

---

## 1. Repository priorities

Follow AGENTS.md exactly.

Priority order:

    Physical safety
    > experiment correctness
    > iteration speed
    > readability
    > generic extensibility

When simplifying code, prefer:

    delete -> inline -> merge -> rewrite -> abstraction

Add a class only when it has a real reason to own resources or persistent state.

Do not add abstraction just because a file is long.

---

## 2. Hard safety and experiment invariants

The refactor must preserve all of the following.

### 2.1 Runtime safety

Keep:

- SafetyState exactly DISARMED / ARMED / RUNNING / FAULT.
- run_id as the stale-command lifecycle epoch.
- motion_lock semantics.
- last-software-boundary run_id checks before SDK admission.
- no command ACK protocol.
- no command transaction protocol.
- no new command IDs.
- no new safety states.
- no hidden software motion-completion model.

Keep the current distinction between:

- normal authority revocation;
- stop/pause/timeout;
- E-stop;
- hardware/runtime fault;
- planned return-home.

Do not hold motion_lock while a hardware SDK operation may block.

### 2.2 Hardware ownership

Preserve process-local ownership:

- xArm SDK object stays in the arm worker.
- XHand SDK object stays in the hand worker.
- RealSense SDK object stays in the camera worker.
- HTS/VR client stays in the VR worker.
- model/CUDA runtime stays in the policy worker.

Ordinary imports and configuration construction must not connect hardware.

Do not execute hardware-affecting code during this task.

### 2.3 Shutdown

Preserve:

- children stop before shared memory is released;
- verified graceful/terminate/kill behavior;
- no shared-memory unlink while a child may still be alive;
- current critical-worker versus service-worker failure semantics;
- safe driver disconnect/passive/stop behavior.

### 2.4 Current robot behavior

Preserve:

- xArm limits, speed and acceleration semantics;
- XHand rated/mechanical limits;
- XHand 5 degree HOME convergence tolerance;
- XHand 2 second HOME timeout;
- XHand normal revoke hold-current behavior;
- passive fallback when appropriate;
- tactile aggregate/dense validity semantics;
- arm/hand HOME measured convergence;
- IK behavior;
- target collision checks;
- planned return-home collision/environment checks;
- normal teleop/eval/replay collision scope;
- workspace behavior.

Do not add software slew/delta filters.

### 2.5 Operator behavior

Preserve teleop and policy operator semantics exactly.

Policy deployment must preserve:

- immediate S/Q motion fencing even while H is blocking;
- same-batch S/Q suppressing H/B;
- H requiring a fresh later B before a new rollout;
- physical_home_completed semantics;
- sticky ESC/E-stop behavior;
- current event ordering and lifecycle transitions.

Teleop must preserve:

- B/C/S/D/H/Q/ESC meanings;
- pause/resume;
- post-pause fresh re-anchoring;
- recording boundaries;
- HOME behavior;
- control cadence;
- abnormal episode semantics.

### 2.6 Observations and data

Preserve:

- one current observation snapshot reused for control and recording;
- freshness semantics;
- policy-local history;
- point-cloud source-camera matching only when RGB and point cloud are jointly consumed;
- raw episode schemas;
- units;
- frames;
- joint ordering;
- finger ordering;
- tactile ordering;
- action semantics;
- canonical Zarr behavior;
- whole-episode admission/rejection;
- provenance semantics.

No persisted semantic change is allowed in this task.

---

## 3. Architecture rules for this refactor

Use the following rules consistently.

### 3.1 Use a class when

Use a class for:

- hardware or external-resource ownership;
- filesystem transaction ownership;
- a persistent cross-tick control state machine;
- a persistent operator state machine;
- a session with substantial shared mutable context.

Examples that are already correct:

- XArm7
- XHand
- RealSenseCamera
- EpisodeRecorder
- PolicyRunner
- TeleopController
- KeyboardInput
- _RecorderIOSession

New justified classes in this task:

- TeleopRunner
- PolicyOperator
- RuntimeSupervisor
- CameraCalibrationSession
- RolloutStats as a dataclass

### 3.2 Keep functions when

Keep functions for:

- geometry;
- transforms;
- FK/IK math helpers;
- collision math;
- point-cloud numerical processing;
- validation;
- target projection;
- dataset processing;
- replay transaction flow when no persistent state object is needed;
- multiprocessing target adapters.

### 3.3 Keep workflow topology explicit

Experiment/session functions should visibly construct the processes they use.

A reader should be able to see which processes exist and the order in which they start.

Do not hide process topology behind:

- RuntimeManager
- DeviceManager
- ProcessRegistry
- WorkerRegistry
- generic start_everything()
- plugin/factory machinery

RuntimeSupervisor may abstract lifecycle mechanics, but not workflow topology.

### 3.4 Do not introduce framework abstractions

Do not add:

- BaseRobot
- BaseSensor
- BaseWorker
- BaseRunner
- BaseSession
- generic RobotInterface hierarchies
- dependency-injection containers
- Hydra-style runtime object factories
- service locators
- plugin registries
- generic lifecycle graphs
- Protocol/Generic layers without a concrete current need

---

## 4. High-level target organization

The intended direction is:

~~~text
dexmani_real/
├── config/
│   ├── hardware.py
│   ├── control.py
│   ├── environment.py
│   ├── pointcloud.py
│   └── experiment.py
├── ipc/
├── runtime/
│   ├── safety.py
│   ├── processes.py
│   ├── supervisor.py
│   ├── observation.py
│   └── operator_input.py
├── robot/
│   ├── drivers/
│   ├── arm_worker.py
│   ├── hand_worker.py
│   ├── arm_homing.py
│   ├── hand_homing.py
│   └── ...
├── sensor/
│   ├── camera/
│   ├── pointcloud.py
│   ├── pointcloud_worker.py
│   └── vr_worker.py
├── teleop/
│   ├── session.py
│   ├── runner.py
│   ├── keyboard_session.py
│   ├── config.py
│   ├── control/
│   │   ├── controller.py
│   │   ├── action_proposal.py
│   │   ├── hand_retargeting.py
│   │   └── vr_mapping.py
│   └── retargeting/
├── deployment/
│   ├── session.py
│   ├── operator.py
│   ├── runner.py
│   ├── action.py
│   ├── observation.py
│   └── config.py
├── calibration/
│   ├── camera/
│   │   ├── session.py
│   │   ├── motion.py
│   │   ├── solver.py
│   │   └── extrinsics.py
│   ├── vr_heading.py
│   └── table.py
├── planning/
├── recording/
├── replay/
├── dataset/
└── utils/
~~~

Do not force file moves that add large import churn without improving ownership or traceability. The structure above is a direction, not a reason for cosmetic movement.

---

## 5. Phase 1 — Normalize worker vocabulary and delete duplicate config wrappers

This phase must be behavior-preserving and mostly mechanical.

### 5.1 Rename multiprocessing targets

Rename process entry points consistently:

- arm_loop -> run_arm_worker
- hand_loop -> run_hand_worker
- camera_loop -> run_camera_worker
- pointcloud_loop -> run_pointcloud_worker
- vr_loop -> run_vr_worker
- recorder_io_loop -> run_recorder_worker
- policy_runner_loop -> run_policy_worker
- teleop_loop -> run_teleop_worker

Update all imports and Process(target=...) call sites.

Do not retain compatibility aliases unless a real external API in this repository requires them. Prefer deleting obsolete names.

Do not rename ordinary internal control loops merely because they contain the word loop.

### 5.2 Simplify camera worker configuration

Delete CameraLoopConfig.

It duplicates CameraParams and reconstructs CameraParams only to validate the same fields.

The camera worker should accept the canonical immutable CameraParams directly:

~~~python
def run_camera_worker(shared, config: CameraParams) -> None:
    ...
~~~

Production call sites should pass:

    runtime.camera

Do not create CameraWorkerConfig as a replacement unless the worker later needs a genuinely narrower/composed boundary object. It does not need one now.

### 5.3 Simplify VR worker configuration

Delete VRReceiverConfig.

It duplicates VRParams one-to-one.

The VR worker should accept VRParams directly:

~~~python
def run_vr_worker(shared, config: VRParams) -> None:
    ...
~~~

For VR calibration, create the required VRParams value explicitly, for example by dataclasses.replace(runtime.vr, port=...) or by constructing a validated VRParams.

Do not create a replacement VRWorkerConfig.

### 5.4 Keep real boundary configs

Keep configuration objects that genuinely project or combine workflow state across a process boundary:

- PolicyRuntimeConfig
- FingertipAssemblerConfig
- RuntimeChannelsConfig
- TeleopConfig
- point-cloud worker configuration
- recorder worker configuration

Rename, where practical:

- PointCloudLoopConfig -> PointCloudWorkerConfig
- RecorderIOConfig -> RecorderWorkerConfig

Only perform these renames if all call sites can be updated cleanly in the same change. Do not leave parallel old/new APIs.

### 5.5 Teleop control module vocabulary

Move/rename:

- teleop/control_loop -> teleop/control
- grid.py -> controller.py
- hand_control.py -> hand_retargeting.py
- run_control_grid_tick -> execute_control_step, if that name still accurately describes the function after the TeleopRunner refactor

Keep:

- action_proposal.py
- vr_mapping.py

Do not change their algorithms.

### Phase 1 acceptance

Repository searches should show:

- no CameraLoopConfig;
- no VRReceiverConfig;
- no multiprocessing target using the old *_loop names listed above;
- imports resolve cleanly;
- no behavior/config value changes.

---

## 6. Phase 2 — Introduce TeleopRunner

The current teleop worker has persistent state encoded as local variables and closures. Make that state explicit.

Create:

    dexmani_real/teleop/runner.py

with:

~~~python
class TeleopRunner:
    def run(self) -> None: ...
    def _begin_episode(self) -> ...: ...
    def _stop_episode(self, *, save: bool, reason: str, abnormal: bool = False) -> None: ...
    def _pause(self, *, manual: bool, mark_episode: bool = True) -> None: ...
    def _try_resume(self, observation) -> bool: ...
    def _handle_operator_command(self, command) -> None: ...
    def _execute_control_step(self, observation) -> None: ...
    def _home_abort_requested(self) -> bool: ...
~~~

The exact private method boundaries may differ slightly if the current source suggests a clearer split, but the ownership must be explicit.

### 6.1 TeleopRunner should own

- operator input lifetime used by the worker;
- AudioFeedback lifetime if currently worker-owned;
- RecorderClient lifetime;
- TeleopController lifetime;
- active/paused/resume state;
- episode state;
- quit-pending state;
- timing/cadence state;
- per-session failure counters;
- recording row budget;
- local lifecycle needed across control ticks.

### 6.2 TeleopRunner must not absorb

Do not move these algorithms into TeleopRunner:

- IK math;
- path planning;
- VR geometry;
- retargeting optimization;
- point-cloud processing;
- recording storage implementation;
- hardware SDK calls.

Those remain in their current domain owners.

### 6.3 Thin worker adapter

run_teleop_worker must be a thin multiprocessing target:

~~~python
def run_teleop_worker(shared, config: TeleopConfig) -> None:
    TeleopRunner(shared, config).run()
~~~

### 6.4 Preserve TeleopController

TeleopController is already a justified stateful control object.

Keep it.

Prefer clearer names where the current names are too abbreviated:

- cache -> hand_observation_cache
- prev_qpos_cmd -> previous_arm_command
- ema_pos -> smoothed_eef_position
- ema_quat -> smoothed_eef_quaternion
- compute -> compute_command, if this improves call-site clarity

Do not rename merely for style if a name is already clear.

### Phase 2 acceptance

The teleop worker should no longer be a large function whose persistent runtime state is held by unrelated local variables and closures.

The behavior of B/C/S/D/H/Q/ESC must be unchanged.

---

## 7. Phase 3 — Extract PolicyOperator from deployment/session.py

Create:

    dexmani_real/deployment/operator.py

with a state owner such as:

~~~python
class PolicyOperator:
    def run(self) -> None: ...
    def _handle_command_batch(self, commands) -> None: ...
    def _request_stop(self) -> None: ...
    def _request_quit(self) -> None: ...
    def _run_home(self) -> None: ...
    def _home_abort_requested(self) -> bool: ...
~~~

Use it from the existing operator thread. It does not need to subclass Thread.

### 7.1 PolicyOperator owns

- KeyboardInput interaction;
- B/S/H/Q/ESC interpretation;
- same-batch command ordering;
- HOME blocking interaction;
- fresh authorization requirement after HOME;
- immediate stop/quit callbacks needed while HOME is blocked;
- operator-local state.

### 7.2 deployment/session.py owns

After extraction, deployment/session.py should focus on:

- validation;
- modality requirements;
- RuntimeChannels creation;
- explicit process construction;
- start ordering;
- ARMED transition;
- PolicyOperator thread lifecycle;
- supervision;
- shutdown.

### 7.3 Preserve event semantics exactly

Do not simplify the operator flow by changing event ordering.

In particular preserve:

- S/Q immediate motion fencing;
- S/Q suppressing H/B in the same batch;
- H blocking behavior;
- clearing stale H/B after HOME where current behavior requires it;
- retaining S/Q/ESC semantics;
- physical_home_completed;
- fresh B after successful HOME.

### Phase 3 acceptance

deployment/session.py should read as an experiment topology/lifecycle file rather than a combined topology + human state-machine implementation.

---

## 8. Phase 4 — Introduce a thin RuntimeSupervisor

runtime/supervisor.py currently exposes process readiness/liveness helper functions.

Evolve it into a small state owner:

~~~python
class RuntimeSupervisor:
    def __init__(self, shared, readiness_timeouts):
        self.shared = shared
        self.readiness_timeouts = readiness_timeouts
        self.started_processes = []

    def start(self, processes) -> None: ...
    def check(self) -> bool: ...
    def run(self) -> None: ...
    def shutdown(
        self,
        *,
        disarm_if_clean: bool = False,
        service_process_names=(),
    ) -> ShutdownReport: ...
~~~

The exact method surface may be smaller if that is clearer.

### 8.1 RuntimeSupervisor may own

- the list of already-started child processes;
- readiness waits;
- liveness checks;
- repeated supervision loop;
- invocation of verified shutdown helpers.

### 8.2 RuntimeSupervisor must not own

Do not move into it:

- process construction;
- sensor selection;
- policy modality decisions;
- robot creation;
- camera creation;
- recorder creation;
- task semantics;
- HOME planning;
- operator semantics.

### 8.3 Keep runtime/processes.py

Do not absorb stop_processes_verified or shutdown_processes_verified into a generic manager.

They remain explicit low-level safety primitives.

RuntimeSupervisor may call them.

### 8.4 Update workflows carefully

Update teleop, deployment, replay, keyboard control, and camera calibration only where the supervisor removes repeated lifecycle mechanics.

Keep their process topology visible.

### Phase 4 acceptance

A session file should still visibly show which processes are created and in what order they start, while repeated started-list/readiness/check/shutdown boilerplate is reduced.

---

## 9. Phase 5 — Remove planner dynamic delegation and clarify safety locking

### 9.1 Remove XArm7MotionPlanner.__getattr__

Delete the dynamic delegation across:

- kin
- IK geometry
- MPlib planner

Update internal and external call sites to use explicit owners.

Examples:

~~~python
self.kin.compute_eef_pose_world(...)
self.ik_geometry.canonicalize_qpos(...)
~~~

Do not preserve a magic fallback alias.

### 9.2 Rename ik_mgr

Rename the IKGeometry owner from:

    ik_mgr

to:

    ik_geometry

across planner, online IK, path/home helpers, and all call sites where it refers to IKGeometry.

Do not rename unrelated manager variables that are not this object.

### 9.3 Keep a narrow planner public API

Explicit wrappers should exist only when the operation is genuinely part of the planner abstraction, such as:

- online IK;
- joint path planning;
- planner hand-state update;
- collision-checked planning;
- planner-level target operations.

Do not wrap every lower-level geometry method simply to make call sites shorter.

### 9.4 Clarify nested motion-lock logic

runtime/safety.py already has _begin_motion_locked.

Add the symmetric internal revoke primitive:

~~~python
def _revoke_motion_locked(...):
    ...
~~~

Then implement revoke_motion by acquiring motion_lock once and calling that helper.

When revoke_motion_if_run_id or request_policy_stop already hold motion_lock, call _revoke_motion_locked directly.

Current code uses ctx.RLock, so this is not a deadlock fix. It is a readability/safety-contract improvement that removes reliance on implicit recursive locking.

Do not change state-transition behavior.

### Phase 5 acceptance

Repository searches should show:

- no XArm7MotionPlanner.__getattr__;
- no dynamic planner API forwarding;
- no ik_mgr name for the IKGeometry owner;
- nested revoke logic is explicit;
- safety transition behavior is unchanged.

---

## 10. Phase 6 — Clean PolicyRunner state and CameraCalibrationSession

### 10.1 PolicyRunner

Keep PolicyRunner as the long-lived policy execution object.

Improve low-information names when safe:

- spec -> policy_spec
- fk -> fingertip_runtime, if the object is specifically the deployment fingertip/FK runtime
- actions -> action_queue
- _row -> _read_observation
- _live -> _has_motion_authority
- _begin -> _begin_episode
- _finish -> _finish_episode

Avoid tuple-packed initialization when separate assignments improve readability.

### 10.2 Add RolloutStats

Move rollout diagnostics out of general execution state:

~~~python
@dataclass
class RolloutStats:
    ...
~~~

It should own the existing statistics, such as:

- inference timing;
- action interval timing;
- clip counters/magnitudes;
- IK failure counts;
- publication counts.

Do not change what is measured or how it is interpreted.

### 10.3 CameraCalibrationSession

calibration/camera/session.py currently has substantial persistent shared session context.

Introduce:

~~~python
class CameraCalibrationSession:
    def run(self) -> int: ...
    def _capture_sample(self, ...) -> ...: ...
    def _handle_sample_events(self, ...) -> ...: ...
    def _show_preview(self, ...) -> ...: ...
    def _check_runtime(self, ...) -> ...: ...
    def _run_control_loop(self, ...) -> ...: ...
    def _solve_and_save(self, ...) -> ...: ...
~~~

Method boundaries should follow the actual source rather than this exact list if another split is clearer.

Do not move calibration mathematics or motion primitives into this class.

Keep:

- solver.py for calibration math/detection/persistence logic;
- motion.py for motion/safety helpers;
- extrinsics.py for calibration loading and transform representation.

### Phase 6 acceptance

Persistent rollout/calibration state should be explicit, while numerical code remains independently readable and testable.

---

## 11. Phase 7 — Make resource/config ownership explicit

### 11.1 TeleopConfig runtime is required

Remove the implicit:

    default_factory=resolve_experiment_config

from TeleopConfig.runtime.

TeleopConfig should require an explicit ExperimentConfig.

Production code already resolves runtime before building the teleop session.

Configuration construction should not unexpectedly resolve a whole experiment.

### 11.2 CameraExtrinsics loading must be explicit

CameraExtrinsics() reads and validates cameras.json.

Do not hide this file/resource acquisition in worker-config default factories.

Remove implicit/default construction from:

- recorder worker configuration;
- point-cloud worker configuration paths where the session can provide the already-loaded snapshot.

Load the calibration snapshot at the workflow/session boundary:

~~~python
camera_calibration = CameraExtrinsics()
~~~

and inject it into the relevant config objects.

Reuse the same immutable snapshot when both recorder and point-cloud worker need it.

### 11.3 RuntimeChannelsConfig point count

RuntimeChannelsConfig.from_runtime(runtime) should use:

    runtime.pointcloud.num_points

as its default point-cloud point count.

Policy deployment may still explicitly override this from the policy artifact when the policy declares a different point count.

Do not silently hard-code 1024 as the from_runtime default.

### 11.4 Avoid hidden filesystem work in dataclass defaults

Review config dataclasses touched by this task.

A simple dataclass construction should not unexpectedly:

- connect hardware;
- read calibration files;
- load models;
- perform other external resource acquisition.

### Phase 7 acceptance

At workflow boundaries, a reader can see required calibration/resource snapshots being acquired and passed into workers.

---

## 12. Phase 8 — Split config responsibilities and remove module singleton defaults

Do this only after the previous phases are stable because it creates broad import churn.

### 12.1 Target files

Create a moderate split, not one file per dataclass.

hardware.py should contain hardware-facing parameter types, for example:

- HomingParams
- ArmParams
- HandParams
- CameraParams
- VRParams

control.py should contain control/policy/runtime parameter types, for example:

- EMAParams
- VRMappingParams
- PolicyParams
- TeleopTimingParams
- KeyboardTeleopParams
- SafetyParams
- TAGRetargetingParams
- DexPilotRetargetingParams

environment.py should contain environment/spatial parameter types, for example:

- WorkspaceBounds
- StaticCollisionBox
- TableCollisionConfig
- EnvironmentConfig

Keep pointcloud.py and experiment.py.

If the current dependency graph suggests a slightly different grouping that reduces circular imports, prefer the simpler dependency graph over the exact grouping above.

### 12.2 ExperimentConfig owns defaults

Prefer:

~~~python
@dataclass(frozen=True)
class ExperimentConfig:
    arm: ArmParams = field(default_factory=ArmParams)
    hand: HandParams = field(default_factory=HandParams)
    ...
~~~

Then resolve YAML/CLI changes over a fresh ExperimentConfig rather than deep-copying module-level singleton objects.

### 12.3 Remove module singleton defaults

Remove global singleton instances such as:

- arm = ArmParams()
- hand = HandParams()
- camera = CameraParams()
- vr = VRParams()
- policy = PolicyParams()
- and analogous default objects

Update low-level modules to use explicit runtime/config values or type defaults as appropriate.

Do not keep compatibility singleton aliases after all internal call sites are migrated.

### 12.4 Do not mass rename Params to Config

Do not create broad churn by changing every Params suffix.

Names should reflect domain meaning, not artificial suffix uniformity.

### 12.5 Clean pseudo-field docstrings

Replace class-body string literals following dataclass fields, such as:

    prior_weight: float = ...
    """..."""

with either:

- a concise comment;
- or class-level documentation.

Those literals are not real field docstrings.

### Phase 8 acceptance

There is one clear source of runtime configuration: ExperimentConfig plus explicit external overrides.

No production behavior depends on mutable/global config singleton instances.

---

## 13. Phase 9 — Finish module-boundary cleanup where it is already half-done

### 13.1 Retargeting

retargeting/retargeter.py currently mixes shared landmark geometry with backend-specific runtime classes while backend modules already exist.

Prefer a final structure such as:

~~~text
retargeting/
├── geometry.py
├── dexpilot.py
├── tag.py
├── tag_optimizer.py
└── pin_grad.py
~~~

Possible ownership:

geometry.py:

- landmark validation;
- human flexion features;
- adaptive XHand landmark scaling/geometry helpers.

dexpilot.py:

- DexPilotHandRetargeter;
- DexPilot builder/runtime.

tag.py:

- TAGHandRetargeter.

Keep optimizer/math behavior unchanged.

Do not create a generic retargeter interface hierarchy.

### 13.2 VR heading calibration

Move reusable VR heading calibration implementation out of the long example script into:

    dexmani_real/calibration/vr_heading.py

Keep the CLI thin.

Prefer functions and a small config dataclass; do not create a class unless persistent state actually justifies it.

### 13.3 Canonical quaternion helper

If the duplicate xyzw_to_wxyz helper still exists, reuse the canonical pose/kinematics conversion helper instead of maintaining two copies.

Do not create a generic validation utility module merely to deduplicate a few-line local validator.

### Phase 9 acceptance

Algorithms are easier to locate by domain name, without introducing new framework layers.

---

## 14. Phase 10 — Repository-level paper-code polish

Perform this only after the core source refactor is stable.

### 14.1 deployment/smoke_test.py

It currently acts as a broad offline regression suite rather than deployment-only smoke coverage.

**Default for this task: leave its path unchanged.** Only move/split it if the core refactor makes the current location actively misleading and the move remains small and mechanical.

A later cleanup may move/split it under something like:

    dexmani_real/offline_checks/

while preserving a simple single entry point.

Do not create a conventional committed tests/ tree.

If moving this suite would create disproportionate path/import churn, leave it in place for this task and only improve naming/documentation where necessary. Core source clarity is more important than path aesthetics.

### 14.2 examples/scripts/tools

The repository currently mixes formal workflows and diagnostics in examples/.

**Default for this task: keep existing CLI paths stable.** Naming/documentation cleanup is preferred over directory migration unless a move has a clear, immediate readability benefit and all user-facing commands can be updated mechanically.

Formal workflows include:

- collect_teleop
- keyboard_teleop
- replay_episode
- run_policy
- calibrate_camera
- calibrate_vr_heading
- export_policy_zarr

Diagnostics/inspection include:

- pointcloud processing diagnostic
- RealSense recording diagnostic
- episode visualization
- XHand diagnostics

Do not force a directory migration if it creates large documentation/packaging churn.

If a clean mechanical move is performed, use a simple distinction such as scripts/ for formal workflow CLIs and tools/ for diagnostics.

Otherwise, leave paths stable and ensure file/module names and README descriptions make the distinction obvious.

### 14.3 pyproject

Ruff is the canonical formatter/import sorter.

Remove stale standalone isort configuration if it is no longer used.

Do not add additional formatting/lint toolchains.

### 14.4 README

Update README only for stable user-visible changes caused by this task:

- renamed CLI paths if any;
- stable source organization if useful;
- calibration workflow location if user-facing commands change.

Do not copy this implementation plan into README.

README should remain workflow-oriented.

---

## 15. Explicit no-change areas

Do not structurally refactor these merely because they are large.

### 15.1 Drivers

Keep as high-cohesion resource owners:

- robot/drivers/xarm7.py
- robot/drivers/xhand.py
- sensor/camera/realsense.py

Do not split them into connection/command/state helper classes.

### 15.2 Numerical pipelines

Keep functional unless an actual persistent state requirement emerges:

- sensor/pointcloud.py
- planning/collision.py
- planning/paths.py
- planning/kinematics/*
- dataset processing modules
- replay numerical/trajectory logic

Do not create PointCloudProcessor or similar classes just to reduce LOC.

### 15.3 Recording

The current structure is a good internal reference:

    thin worker target
        -> _RecorderIOSession
        -> EpisodeRecorder

Preserve recording transaction and storage behavior.

### 15.4 Replay

Keep replay workflow procedural.

Do not add ReplaySession simply for architectural symmetry.

### 15.5 KeyboardInput

Keep KeyboardInput as one resource owner.

Do not split keyboard listener, held-key state, terminal echo, callbacks, and listener health into multiple tiny classes.

---

## 16. Naming rules after refactor

Use names that communicate ownership.

Preferred patterns:

- long-lived execution state: *Runner
- human operator state: *Operator
- physical device: concrete hardware name
- multiprocessing target: run_*_worker
- experiment workflow: run_*
- one control iteration: *_step
- math/geometry: compute_*, transform_*, project_*, solve_*
- validation: validate_*
- diagnostics: *_diagnostics

Avoid vague new names such as:

- Manager
- Handler
- System
- Helper
- Grid
- Engine

unless they describe a real domain concept.

Do not rename already-clear names just for consistency.

---

## 17. Documentation and comments

Follow AGENTS.md.

Comments should explain non-obvious:

- robotics semantics;
- frames;
- units;
- SDK behavior;
- concurrency/safety ordering;
- experiment rationale;
- numerical assumptions.

Do not mechanically delete comments.

Do not add comments that simply narrate Python control flow.

Public workflow and process boundaries may receive useful type annotations, but do not turn this into a repository-wide typing project.

Any remains acceptable at multiprocessing/native-SDK boundaries where precise typing would add complexity without clarity.

---

## 18. Import and dependency direction

While editing, avoid new circular dependencies.

Preferred direction:

    config/data definitions
        -> runtime/domain logic
        -> workflow/session
        -> CLI

Hardware drivers should not import experiment/session modules.

Numerical planning/sensor algorithms should not depend on teleop/deployment orchestration.

robot.model remains the canonical source of robot joint/model constants.

Do not move constants merely to satisfy a cosmetic layering rule.

---

## 19. Execution strategy

Implement this as ordered, reviewable stages.

At the start, run `git status --short` and preserve unrelated local changes. Do not reset, clean, checkout over, or reformat unrelated files.

Do not make one giant mechanical rewrite before checking intermediate correctness. Finish one phase, update every affected call site, remove phase-local dead code, inspect the diff, and run the relevant offline checks before starting the next phase. A phase must not intentionally leave the repository in a half-renamed or dual-API state.

Recommended sequence:

1. worker/module naming + duplicate Camera/VR config deletion;
2. TeleopRunner;
3. PolicyOperator;
4. RuntimeSupervisor;
5. planner explicit ownership + safety locked helper;
6. PolicyRunner cleanup + RolloutStats + CameraCalibrationSession;
7. explicit calibration/config ownership;
8. config responsibility split and singleton removal;
9. retargeting/VR calibration cleanup, only when the resulting split is clearly simpler;
10. repository-level polish only when low-risk; default to leaving existing CLI/offline-check paths stable.

After each major phase:

- remove newly dead imports/functions/config;
- do not preserve obsolete compatibility mechanisms;
- inspect the diff for accidental behavior change;
- run focused offline checks relevant to that phase.

If a later phase becomes significantly riskier because the current source has diverged, keep the stable completed phases and report the exact blocker rather than introducing a generic workaround.

---

## 20. Offline validation

Do not run hardware-affecting code.

Before executing any Python module beyond compile/lint checks, inspect its imports, constructors, and entry point and confirm it cannot discover/connect hardware, open a camera/VR stream, write calibration state, home, replay, teleoperate, or execute policy motion. If there is doubt, do not run it.

At minimum run the repository checks from AGENTS.md:

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

If example paths are intentionally moved, update these commands to the actual source/CLI directories and update AGENTS.md accordingly.

When installed optional dependencies are available, also run the existing offline regression entry point:

    python -m dexmani_real.deployment.smoke_test

or its new equivalent if the suite is intentionally relocated.

Use focused one-off offline checks for changed pure logic, especially:

- configuration resolution;
- RuntimeChannels create/close;
- RuntimeSupervisor readiness/shutdown;
- SafetyState transitions;
- run_id fencing;
- policy operator event ordering;
- teleop pause/resume state transitions;
- camera frame packing;
- point-cloud numerical behavior;
- planner/IK numerical behavior;
- retargeting numerical behavior;
- calibration pure math;
- recording start/stop/finalization boundaries.

Do not install or upgrade dependencies just to make a check available. Report unavailable checks.

Offline checks do not constitute hardware validation.

---

## 21. Hardware validation checklist to report, not execute

At the end of the task, report that the operator should separately validate:

- xArm connect/read/stop;
- XHand connect/read/passive;
- XHand HOME with current 5 degree tolerance and 2 second timeout;
- arm HOME;
- VR readiness;
- camera readiness;
- point-cloud readiness;
- recorder startup/finalization;
- teleop B/C/S/D/H/Q/ESC;
- teleop pause/resume re-anchoring;
- policy B/S/H/Q/ESC;
- stop/quit while HOME is blocking;
- E-stop;
- abnormal worker shutdown;
- verified child termination before shared-memory release.

Do not perform these checks without explicit authorization.

---

## 22. Final repository acceptance criteria

The task is complete only when the following are true.

### 22.1 Readability

A reader can trace the main workflows without hidden ownership:

    CLI
    -> session
    -> worker
    -> runner/controller
    -> driver or numerical algorithm

### 22.2 No hidden state-machine functions

The major persistent state machines are explicit objects:

- TeleopRunner
- PolicyRunner
- PolicyOperator
- _RecorderIOSession
- CameraCalibrationSession where appropriate

### 22.3 No unnecessary config duplication

There is no CameraLoopConfig or VRReceiverConfig copy of canonical runtime config.

Worker-specific config objects exist only where they actually narrow/combine boundary inputs.

### 22.4 No planner magic forwarding

XArm7MotionPlanner does not use __getattr__ to expose unrelated owner methods.

Ownership is visible at call sites.

### 22.5 Lifecycle stays explicit

RuntimeSupervisor removes repeated lifecycle mechanics without hiding experiment topology.

### 22.6 Config has one clear source

ExperimentConfig and explicit overrides define runtime configuration.

Module-level default singleton instances are removed after migration.

### 22.7 Resource acquisition is visible

Calibration/model/hardware resources are not unexpectedly loaded by innocent dataclass defaults.

### 22.8 Research behavior is unchanged

No change to:

- safety-state semantics;
- run_id fencing;
- hardware limits;
- control actions;
- observation semantics;
- calibration meaning;
- dataset schema;
- recording schema;
- action semantics;
- IK/collision algorithms;
- retargeting algorithms;
- point-cloud algorithms.

### 22.9 No over-engineering

The final diff must not introduce generic framework layers that are not required by the current experiments.

---

## 23. Final review before completion

Before declaring the task complete, perform one final coherence review across definition -> producer -> transformation -> consumer -> side effect for every changed cross-process or persisted-data boundary.

Then:

1. Inspect git status and the complete diff.
2. Check for unrelated changes.
3. Check for dead compatibility aliases.
4. Check for dead config fields.
5. Search for stale old process-target names.
6. Search for CameraLoopConfig and VRReceiverConfig.
7. Search for XArm7MotionPlanner.__getattr__.
8. Search for IKGeometry still being called ik_mgr.
9. Search for hidden CameraExtrinsics default construction in worker configs.
10. Search for imports from old config/default singleton objects.
11. Confirm README and AGENTS.md are updated only where stable user-facing paths/checks changed.
12. Run all available offline checks.
13. Report any check that could not run because of optional dependency availability.
14. Report the separate hardware-validation checklist without executing it.
15. Report which conditional Phase 9-10 cleanups were intentionally skipped and why; skipped cosmetic moves are not task failures.
16. Summarize the final architecture in terms of ownership changes, deleted abstractions, and preserved behavior rather than raw file-count changes.

The desired final result is a direct, explicit, research-oriented codebase: important state is visible, important ownership is visible, important safety ordering is visible, and numerical research logic remains simple and easy to inspect.
