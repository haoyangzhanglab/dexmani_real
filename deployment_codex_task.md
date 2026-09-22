# DexMani Real ↔ Policy Deployment Cleanup Task

## 0. Scope and repositories

Work on both sibling local repositories:

~~~text
~/Desktop/dexmani_real
~/Desktop/dexmani_policy
~~~

This task file belongs in:

~~~text
~/Desktop/dexmani_real/deployment_codex_task.md
~~~

Reviewed remote heads when this task was finalized:

~~~text
dexmani_real   ca950c1b147d98677fa723318995b84d6947874c
dexmani_policy 6ba48d7566bd565839a8cee9c14df32e402f8a65
~~~

Inspect both local HEADs first. Local code is authoritative if newer. Do not patch by stale line number.

These are personal PhD research repositories. Optimize for:

1. correct experiments and correct robot behavior;
2. simple ownership boundaries;
3. fast research iteration;
4. useful reproducibility;
5. minimal duplicated state and minimal compatibility machinery.

Do not optimize for untrusted third-party artifacts, multi-tenant security, enterprise schema evolution, arbitrary future embodiments, or hypothetical plugin systems.

---

# 1. Non-negotiable design rule

> Freeze how the model sees data; do not freeze what the physical world currently is.

There are only three relevant categories.

## 1.1 Model-dependent preprocessing — Policy/checkpoint owned

These change the model input distribution or inference behavior and therefore follow the trained policy:

- observation modalities;
- tensor layout/order for non-image observations;
- ordered robot joint names;
- n_obs_steps;
- n_action_steps;
- control_dt_s;
- physical action mode: joint or eef;
- point-cloud algorithm parameters:
  - num_points;
  - depth gates;
  - edge filters;
  - workspace crop;
  - voxel size;
  - table-removal enabled/disabled;
  - table-removal thresholds;
  - outlier filtering;
  - sampling parameters;
- Policy-owned RGB preprocessing;
- normalization state/modes;
- model architecture and weights;
- Policy-private inference configuration.

## 1.2 Current physical calibration — Real owned

These may legitimately change after training. Old policies must use the current correct values and must not become incompatible merely because calibration changed:

- camera extrinsic calibration;
- camera intrinsics/native RGB-D geometry;
- depth scale;
- camera serial/physical identity;
- hand/end-effector mounting calibration used for fingertip geometry;
- other current robot calibration values.

## 1.3 Current environment geometry — Real owned

These describe the current world and are not policy identity:

- current table-plane coefficients;
- current table pose/height;
- other environment geometry used to map the current world into the policy canonical frame.

Historical calibration/environment values may stay in raw/Zarr/rollout metadata as provenance only. They are not deployment compatibility gates.

---

# 2. Core target architecture

The final boundary should be thin:

~~~text
dexmani_real
  hardware / sensors / current calibration
  -> canonical raw observation
  -> minimal PolicySpec
  -> dexmani_policy

dexmani_policy
  preprocessing
  normalization
  model
  unnormalization
  -> physical action chunk

dexmani_real
  physical action adapter
  safety / limits / IK
  -> hardware
~~~

Do not create a third shared package such as dexmani_contract.

---

# 3. Minimal Real-facing PolicySpec

Keep the Real-facing public spec small. A conceptual target is:

~~~python
@dataclass(frozen=True)
class ObservationSpec:
    dtype: str
    shape: tuple[int | None, ...]

@dataclass(frozen=True)
class PolicySpec:
    observations: dict[str, ObservationSpec]
    n_obs_steps: int
    n_action_steps: int
    control_dt_s: float
    action_mode: Literal["joint", "eef"]
    joint_names: tuple[str, ...]
    pointcloud_config: dict[str, Any] | None = None
~~~

Exact file/type names may follow current project style.

Persist this spec in the deployment artifact as plain dict/list/string/number metadata, then parse to dataclasses at runtime.

Do not pickle custom metadata classes just to persist the spec.

The public spec must not expose Policy-private internals such as:

- model action_dim when auxiliary outputs exist;
- horizon;
- use_aux_ee;
- tcp_dim / hand_dim;
- denoise/inference steps;
- normalizer state/grammar;
- Hydra agent config;
- RGB resize/crop/mean/std internals;
- dataset paths;
- calibration snapshots;
- descriptive algorithm/derivation/version strings.

Keep default inference steps outside the compatibility spec, for example in ExperimentInfo or Policy runtime metadata.

---

# 4. Required correctness fixes

## 4.1 Explicit ordered joint names

In the Real robot-model owner, define the canonical order already used by hardware/data paths:

~~~python
XARM7_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)

ROBOT_JOINT_NAMES = (
    *XARM7_JOINT_NAMES,
    *XHAND_SDK_JOINT_NAMES,
)
~~~

Use actual local names if they differ.

New Real Policy Zarr metadata must store the ordered list, e.g. joint_names.

Checkpoint/artifact metadata must propagate exactly this order.

Real deployment must require exact tuple equality, not set equality or fuzzy aliases.

Do not add permanent runtime guessing for old checkpoints. If important old artifacts need support, use one-time conversion.

## 4.2 Remove Real-side model RGB resize

In dexmani_real deployment observation construction:

- remove model-facing resize/crop;
- pass raw camera RGB unchanged;
- preserve uint8 HWC;
- require valid array, HWC layout, 3 channels and stackable history.

Policy owns resize/crop/scale/normalize/model-specific processing.

Do not add an unconditional training-H/W == runtime-H/W gate in Real.

Current Policy preprocessing already supports resizing. A different raw resolution should remain usable when the Policy preprocessing supports it.

If a specific Policy path truly requires fixed raw H/W, reject inside Policy, not duplicated in Real.

## 4.3 Unified online IK profile

Create one small helper for online Cartesian control behavior, conceptually:

~~~python
def make_online_ik_config(runtime, *, control_dt_s: float) -> OnlineIKConfig:
    return OnlineIKConfig(
        max_pose_error_pos_m=runtime.policy.ik_max_pose_error_pos_m,
        max_pose_error_rot_rad=runtime.policy.ik_max_pose_error_rot_rad,
        nullspace_step_size_deg=(
            runtime.policy.ik_nullspace_step_rate_deg_s * control_dt_s
        ),
    )
~~~

Use it for paths intended to share behavior:

- teleop/VR Cartesian control;
- EEF policy deployment.

Do not force intentionally different calibration/homing/keyboard profiles through this helper.

## 4.4 Isolate physical action decoding

Keep action_ee because current research uses it.

But only the Real physical action adapter should branch on control representation.

Prefer one small module/function, e.g.:

~~~text
dexmani_real/deployment/action.py
~~~

Behavior:

~~~text
joint:
  physical 19D chunk -> arm7 + hand12

eef:
  physical 21D chunk -> pose9 + hand12
                     -> IK for arm7
                     -> arm7 + hand12
~~~

Policy may internally predict 28D or other auxiliary layouts. LoadedPolicy.predict() must return only the physical control chunk.

Real-facing API should use action_mode = joint | eef, not Policy-internal action_key semantics.

---

# 5. Point-cloud ownership: highest-priority data-path fix

## 5.1 Separate table-removal policy from table calibration

Currently table_plane_abcd being None/non-None implicitly decides whether table removal occurs. That mixes preprocessing policy with current environment geometry.

Add an explicit model-owned boolean in PointCloudConfig, e.g.:

~~~python
remove_table: bool = True
~~~

The boolean follows the trained policy.

The numeric table_plane_abcd never belongs to PointCloudConfig.

Required behavior:

~~~text
remove_table=True
  -> use current Real calibrated table plane
  -> if no valid current plane exists, fail clearly

remove_table=False
  -> do not remove the table
  -> ignore current table plane for perception
~~~

Do not let collision-planning table.enabled accidentally control policy perception.

## 5.2 Deployment must use the trained PointCloudConfig directly

Do not keep two point-cloud configs and compare them.

Target flow:

~~~text
training PointCloudConfig
  -> checkpoint-owned metadata
  -> deployment artifact PolicySpec
  -> Real PointCloudLoopConfig
~~~

Online policy deployment must use the artifact config, not current runtime.pointcloud defaults.

Conceptually:

~~~python
PointCloudLoopConfig(
    pointcloud=PointCloudConfig.from_dict(policy_spec.pointcloud_config),
    camera_calibration=current_camera_calibration,
    table_plane_abcd=(
        current_table_plane
        if policy_spec.pointcloud_config["remove_table"]
        else None
    ),
)
~~~

A small PointCloudConfig.from_dict() helper is fine. Do not create a registry/factory framework.

runtime.pointcloud remains valid for new data collection, diagnostics, visualization and future experiments. It must not redefine preprocessing for an already-trained policy.

## 5.3 Current calibration always wins

Deployment compatibility must never compare training-time and current:

- camera extrinsic;
- camera intrinsic;
- camera serial;
- depth scale;
- table-plane coefficients;
- hand-mount calibration.

Old policies must use current correct calibration.

Changing camera serial is allowed if the currently connected camera has valid current calibration.

A missing current calibration is a Real setup error, not a policy incompatibility.

Do not add:

- calibration hashes;
- camera fingerprints;
- table-plane equality/tolerance gates against training;
- URDF hashes.

## 5.4 Clean point-cloud metadata

For new Real Policy Zarr output, prefer one direct algorithm config attr such as:

~~~text
pointcloud_config_json
~~~

It contains only PointCloudConfig.to_dict().

Keep table plane and camera calibration separately as provenance.

The current mixed processing_config_json may be used only as a migration source for old data. Do not make it the new model-facing contract.

Stop using these as deployment compatibility gates:

- POINT_CLOUD_POLICY_ID;
- POINT_CLOUD_COLOR_SOURCE;
- POINT_CLOUD_SAMPLING;
- POINT_CLOUD_TRANSFORM.

They may remain as optional debug/provenance text only.

Correctness must come from actual config, direct tensor order, and numerical regression tests.

---

# 6. Keep direct tensor meaning, remove descriptive prose

Do not delete all semantics blindly.

Keep concrete ordering facts that can cause same-shape silent errors:

- ordered joint_names;
- RGB channel order/layout;
- point feature order, e.g. [x, y, z, r, g, b];
- when tactile/contact is actually consumed, concrete finger/sensor/axis order if required.

Prefer lists/tuples over opaque descriptive strings.

Delete runtime compatibility checks based only on algorithm IDs, derivation IDs, policy IDs or version prose.

---

# 7. Deployment artifact/export redesign

## 7.1 Export must not reopen the training Zarr

Current deployment export roughly does:

~~~text
checkpoint
+ selected/relocated Zarr
-> build_real_policy_data_semantics
-> semantics_mismatch
-> artifact
~~~

Remove that deployment dependency.

Target:

~~~text
selected checkpoint
-> derive minimal public spec from checkpoint-owned saved information
-> build self-contained deployment artifact
~~~

A deleted, moved or unavailable training Zarr must not prevent deployment of a valid checkpoint.

Remove --zarr-path from deployment export if its only purpose is semantic-equivalence proof.

## 7.2 Do not bump the training checkpoint format just for this task

Do not introduce simple.v4 or broadly redesign TrainCheckpoint.

Use current checkpoint-owned information to derive deployment metadata.

For new checkpoints, add only missing concrete facts in the existing saved metadata path:

- ordered joint_names;
- clean point-cloud algorithm config when relevant.

Keep broad training-resume format churn out of scope.

## 7.3 Training resume and deployment are different

Training resume may remain strict enough to detect changed effective training inputs.

Deployment must not depend on historical calibration equality or reopening the dataset.

Do not weaken useful training-resume checks merely because deployment is being simplified.

## 7.4 Keep artifact concept, simplify bureaucracy

Keep a self-contained deployment artifact and preserve useful public APIs where practical:

- inspect_experiment;
- load_experiment;
- export_deployment_artifact.

inspect_experiment should inspect plain metadata without constructing the model.

load_experiment should perform authoritative strict restore and return the NumPy-facing runtime.

A simple loaded.spec == inspected.spec sanity check is fine.

Keep:

- torch.load(..., weights_only=True) if already working;
- strict state_dict restore;
- actual normalizer restore and dimension checks;
- basic format/version + required-key checks;
- simple atomic publication if already cheap/reliable.

Remove when dead:

- FrozenMetadata recursive immutability;
- recursive Hydra _target_ namespace allowlist;
- exact artifact root-key equality that rejects harmless extra provenance;
- deployment semantic deep-diff machinery;
- runtime checks of descriptive algorithm/derivation/version prose.

Do not replace them with a new generic contract/security framework.

## 7.5 Remove duplicate export-time full model prediction

Today export performs strict restore + synthetic prediction, and Real runtime later performs strict restore + warmup.

Keep runtime restore/warmup authoritative before policy_ready.

Export should do cheap structural validation and optional cheap reload/parse only.

Remove export-time full model inference if runtime still performs strict restore/warmup.

---

# 8. Real observation construction

Keep dexmani_real/deployment/observation.py direct.

Conceptually:

~~~python
def build_policy_observation(rows, spec, ...):
    # joint_state
    # point_cloud if requested
    # raw RGB if requested
    # contact/tactile if requested and valid
    # EEF/fingertip from current Real geometry
    # direct shape/dtype/finite checks where meaningful
~~~

Do not create an observation registry/factory.

Fixed numerical modalities should keep exact shape/dtype checks.

Raw RGB should stay raw; Policy owns model preprocessing.

---

# 9. Calibration/provenance

Historical raw episodes/Zarr may keep:

- camera serial;
- intrinsics/native geometry;
- extrinsics;
- depth scale;
- table plane;
- task/recording provenance.

These are scientific provenance, not deployment compatibility.

Rollout recording should also store the current table plane actually used for point-cloud preprocessing if not already present.

Do not compare rollout calibration with training calibration.

---

# 10. Timing and performance

Keep synchronous chunk inference as the baseline.

Do not add in this task:

- async inference;
- continuous inference;
- RTC;
- action inpainting;
- complex chunk splicing.

Keep useful telemetry such as:

- inference_ms;
- action_step_interval_ms.

Add point-cloud/camera age only if timestamp clocks are clearly compatible and the metric is cheap and unambiguous.

Do not create a temporal schema/contract framework.

Performance changes beyond this should be driven by measurements.

---

# 11. Things that must remain strict

Simplification must not weaken real correctness or safety.

Keep or strengthen:

- required observation modalities;
- fixed non-image tensor shape/dtype;
- finite floating inputs;
- ordered joint_names;
- point feature/channel ordering where relevant;
- physical action chunk shape;
- finite physical actions;
- n_obs_steps / n_action_steps / control_dt validity;
- policy rate <= actuator worker rate;
- strict state_dict restore;
- actual normalizer restore/field dimensions;
- sensor freshness;
- tactile validity flags;
- RGB/point-cloud source identity when both are requested;
- whole-episode validation;
- bad-frame/missing-row rejection;
- joint limits;
- workspace/safety projection;
- homing;
- E-stop;
- IK failure handling;
- worker/process health.

Do not simplify safety/lifecycle code merely because it is long.

---

# 12. Explicit non-goals / forbidden additions

Do not introduce:

- CalibrationSpec hierarchy;
- camera calibration hash/fingerprint;
- camera-serial policy compatibility gate;
- table-plane equality/tolerance gate against training;
- URDF SHA256 compatibility gate;
- PhysicalSemanticsVersion;
- closed-world semantic-dictionary equality;
- generic ProducerConfig registry;
- generic TemporalContract;
- shared dexmani_contract package;
- observation/action plugin registry;
- generic compatibility framework;
- permanent v1/v2/v3 runtime branches for old artifacts.

Do not solve a two-case branch with a factory or registry.

Do not preserve dead code "just in case" after the new path is proven. Git history is the archive.

---

# 13. Required tests

Prioritize numerical and end-to-end tests over metadata-parser tests.

## 13.1 Joint ordering

Verify exact propagation:

~~~text
Real Zarr joint_names
-> checkpoint-owned metadata
-> deployment PolicySpec
-> Real ROBOT_JOINT_NAMES
~~~

A deliberate permutation must fail while dimension remains 19.

## 13.2 Calibration changes must not invalidate deployment

An old policy must remain deployable when current:

- camera extrinsic changes;
- camera intrinsic changes;
- table-plane coefficients change;
- camera serial changes with valid current calibration;
- hand-mount calibration changes for fingertip derivation.

Do not assert identical outputs across changed calibration.

Assert:

- no policy compatibility gate rejects the policy;
- current Real calibration values are used.

## 13.3 Policy-owned point-cloud config

Export/build with one point-cloud config, then create Real runtime with a different runtime.pointcloud default.

Deployment must use the artifact config.

Verify a nontrivial parameter such as voxel size or workspace comes from the policy artifact.

## 13.4 Table-removal policy vs current plane

Test:

~~~text
remove_table=True
  -> current valid plane is used
  -> changed numeric plane remains compatible
  -> missing current plane fails clearly

remove_table=False
  -> current plane is ignored for perception
~~~

## 13.5 Offline/online point-cloud numerical equivalence

For identical:

- RGB;
- raw depth;
- intrinsics;
- depth scale;
- extrinsic;
- current table plane;
- PointCloudConfig;

require deterministic/numerically equivalent offline and online point-cloud production.

This replaces confidence previously placed in algorithm-ID strings.

## 13.6 RGB ownership and variable raw resolution

Verify Real passes raw RGB unchanged and never performs model-facing resize.

For a Policy preprocessing path with fixed resize output, feed at least two valid raw H/W values and verify both reach the same expected model-facing output shape.

Still reject:

- non-uint8 RGB;
- wrong channel/layout;
- inconsistent unstackable history.

If a specific Policy truly requires fixed raw H/W, rejection belongs in Policy preprocessing/runtime.

## 13.7 EEF/FK/fingertip numerical regression

For identical joint states/current Real geometry:

~~~text
offline EEF ~= online EEF
offline fingertip ~= online fingertip
~~~

Do not test algorithm-ID strings.

## 13.8 Normalizer

Keep/implement:

~~~text
x -> normalize -> unnormalize ~= x
~~~

and strict restore/field-dimension tests, especially for physical actions.

## 13.9 Physical action adapter

Joint mode:

~~~text
19D -> exact arm7 + hand12 split/order
~~~

EEF mode:

~~~text
21D -> pose9 + hand12
pose -> IK -> FK ~= pose target
~~~

Do not require IK(FK(q)) == q for the redundant arm.

## 13.10 Shared IK profile

For equal control period/runtime settings, teleop and EEF deployment must construct the same intended online IK/null-space step.

## 13.11 Export independence

After a valid checkpoint exists, deployment artifact export must succeed without opening the training Zarr.

Make the original Zarr unavailable/relocated in the test and confirm export only uses checkpoint-owned information.

## 13.12 End-to-end deployment smoke

Use a mock/tiny artifact:

~~~text
raw observation history
-> LoadedPolicy preprocessing/predict
-> physical action chunk
-> Real action adapter
~~~

Verify keys, shapes, finite values and ordering without hardware.

---

# 14. Tests to remove or rewrite

Delete/rewrite tests whose only purpose is enforcing removed deployment bureaucracy:

- semantic-dict exact key sets for deployment;
- semantics_mismatch deployment equality;
- training table plane == deployment table plane;
- point-cloud algorithm/version prose equality;
- EEF/fingertip derivation/version equality;
- FrozenMetadata recursive immutability;
- recursive Hydra target allowlist;
- exact artifact root-key set;
- export-time Zarr relocation/equivalence proof;
- Real-side RGB resize behavior.

Do not recreate equivalent bureaucracy under different names.

Do not delete training-resume tests merely because deployment no longer uses the same metadata. Evaluate those separately by whether they prevent resuming on changed training inputs.

---

# 15. Migration policy

Prefer a clean current design over permanent compatibility layers.

- New Real Policy Zarr writes ordered joint_names and clean point-cloud algorithm config.
- New deployment artifacts use the new minimal public spec.
- Important old artifacts/checkpoints may be converted once using an old revision or a focused migration tool.
- A migration tool may recognize only known legacy formats and emit the new artifact.
- Do not add long-lived "if old artifact then guess" logic to normal deployment.
- Avoid changing the training checkpoint root format unless truly unavoidable.

---

# 16. Likely files

## dexmani_real

Likely relevant:

~~~text
dexmani_real/robot/model.py
dexmani_real/config/pointcloud.py
dexmani_real/dataset/contracts.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/pointcloud.py
dexmani_real/deployment/config.py
dexmani_real/deployment/observation.py
dexmani_real/deployment/runner.py
dexmani_real/deployment/session.py
dexmani_real/sensor/pointcloud_worker.py
dexmani_real/planning/kinematics/ik.py
dexmani_real/teleop/loop.py
dexmani_real/recording/*
examples/run_policy.py
~~~

Do not perform unrelated style/safety refactors.

## dexmani_policy

Likely relevant:

~~~text
dexmani_policy/datasets/real_policy_contract.py
dexmani_policy/training/resume.py
dexmani_policy/common/checkpoint_io.py
dexmani_policy/deployment/contract.py
dexmani_policy/deployment/export.py
dexmani_policy/deployment/restore.py
dexmani_policy/deployment/runtime.py
relevant tests / smoke_test.py
~~~

Do not modify checkpoint_io.py broadly unless required. In particular, do not bump the training checkpoint format just for deployment cleanup.

---

# 17. Required execution order

Follow this sequence.

1. Inspect both local repositories and map the current training -> checkpoint -> export -> inspect/load -> Real deployment paths.
2. Record current tests covering those paths.
3. Add focused tests for new invariants before large deletion where practical.
4. Implement canonical ordered joint names.
5. Remove Real-side RGB resize and make Policy accept/validate raw RGB correctly.
6. Unify online IK behavior and isolate physical action decoding.
7. Add explicit point-cloud table-removal policy and make policy-owned PointCloudConfig drive online production with current calibration/table plane.
8. Derive the minimal public artifact spec from checkpoint-owned saved information; do not reopen Zarr during export.
9. Separate runtime defaults such as inference steps from compatibility spec.
10. Switch Real deployment to the minimal spec.
11. Remove duplicate export-time full model prediction if runtime strict restore/warmup remains authoritative.
12. Run focused tests and end-to-end smoke tests.
13. Only then delete dead semantic deployment machinery.
14. Run both repositories' relevant full test suites.
15. Search for stale concepts and inspect every remaining occurrence.

Do not use git reset --hard or discard unrelated local work.

---

# 18. Final stale-reference audit

Before finishing, search at least for:

~~~text
semantics_mismatch
deployment_data_semantics
processing_config_json
point_cloud_table_plane_abcd_json
POINT_CLOUD_POLICY_ID
POINT_CLOUD_TRANSFORM
POINT_CLOUD_SAMPLING
FrozenMetadata
validate_agent_targets
requires_hand
resize_rgb(
action_key
OnlineIKConfig(
verify_deployment_prediction
~~~

Not every occurrence must become zero.

Acceptable remaining uses include:

- training-resume-only historical dataset checks;
- debug/provenance logging;
- Policy-internal action_key training config;
- unrelated image utilities/examples;
- intentionally different IK profiles.

Every remaining deployment-path occurrence must have a concrete current purpose.

---

# 19. Acceptance criteria

The task is complete only when all conditions below are true.

## Interface

- Real-facing Policy spec is small.
- Persisted spec metadata is plain.
- Public spec contains observation interface, timing, physical action mode, exact joint ordering and point-cloud algorithm config when needed.
- inference defaults and Policy-private architecture/normalization/preprocessing do not pollute compatibility spec.

## Calibration

- recalibrating camera extrinsics does not invalidate an old policy;
- changing camera intrinsics/resolution does not invalidate an RGB policy when Policy preprocessing supports the new raw size;
- recalibrating the table plane does not invalidate an old point-cloud policy;
- changing camera serial with valid current calibration does not invalidate the policy;
- current calibration/environment values are actually used;
- historical calibration values remain provenance only.

## Point cloud

- deployment uses the policy's trained algorithm config, not runtime.pointcloud defaults;
- table-removal enabled/disabled is model-owned;
- numeric table plane is current Real environment data;
- perception table removal is not accidentally controlled by collision enablement;
- descriptive point-cloud IDs are not deployment gates.

## RGB

- Real no longer performs model-facing RGB resize/crop;
- raw RGB is passed unchanged;
- variable raw H/W is allowed when Policy preprocessing supports it;
- Policy owns and validates model-facing spatial preprocessing.

## Actions

- exact ordered 19-joint convention is explicit;
- Real only sees physical joint or eef mode;
- auxiliary output layout remains Policy-private;
- teleop and EEF deployment use consistent intended online IK configuration.

## Artifact/export

- deployment export no longer requires training Zarr;
- no unnecessary training-checkpoint format bump is introduced;
- artifact remains self-contained and inspectable;
- strict runtime restore and normalizer checks remain;
- runtime warmup remains before policy_ready;
- duplicate export-time full model prediction is removed;
- FrozenMetadata, recursive target allowlist and deployment semantic deep-diff disappear when no longer live.

## Safety/data integrity

No regression in:

- freshness;
- source identity;
- homing;
- E-stop;
- joint/workspace limits;
- IK failure handling;
- worker health;
- whole-episode validation;
- recording integrity.

## Complexity

- no generic contract/registry/plugin/calibration-hash/temporal framework is introduced;
- normal deployment has no permanent legacy-guessing branch;
- final code is visibly simpler than the current deployment semantic-contract system;
- every remaining deployment compatibility check prevents a concrete wrong tensor, wrong action, wrong preprocessing mode, or unsafe execution.

---

# 20. Final decision tests

Before adding or keeping any deployment validation, ask:

> If this check fails, can we name the concrete wrong tensor, wrong action, wrong preprocessing behavior, or unsafe robot behavior it prevents?

If not, do not add it.

Before putting any value in the public Policy artifact spec, ask:

> Does Real need this value to construct the raw observation/control boundary?

If not, keep it Policy-private.

Before freezing any physical value into a policy, ask:

> If this value changes, should an old policy use the new current-world measurement, or must it reproduce training-time preprocessing?

- current-world measurement -> Real owns it and recalibration is allowed;
- preprocessing definition -> Policy owns it and remains stable.

That distinction must remain obvious in the final code.
