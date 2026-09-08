# DexMani Real — Raw v25 & Runtime Simplification Implementation Guide

> Repository: `haoyangzhanglab/dexmani_real`  
> Intended executor: Codex / coding agent  
> Design baseline: `main@c258f52b16fd80fc8ba773e4dcc79825da3548ea`  
> Scope: solve issues **#3, #4, #6, #7, #13, #15, #17, #18** with minimal mechanism and minimal maintenance cost.  
> Repository role: personal PhD research code for real-robot data collection, policy deployment, and evaluation.

This document is an implementation contract. Prefer deleting mechanisms over replacing them with new frameworks. If current `main` has drifted from the design baseline, re-check call sites before editing; preserve the invariants in this guide rather than stale line-level details.

---

# 1. Objective

The target repository is intentionally narrow:

```text
dexmani_real
= hardware / sensors
+ calibration
+ teleoperation
+ raw episode recording
+ causal observation assembly
+ policy rollout scheduling
+ physical safety
+ real-robot evaluation
```

The data path remains:

```text
raw v25 episode
      ↓  dexmani_real offline processing
processed v14 episode
      ↓  dexmani_real export
Policy Zarr v7
      ↓
dexmani_policy
```

**Do not modify `dexmani_policy`.**

**Do not change processed v14 or Policy Zarr v7 unless an existing bug makes exact preservation impossible.** The expected outcome of this refactor is that `dexmani_policy` sees the same training contract as before.

Historical raw v24 data is migrated once:

```text
raw v24
   ↓  tools/convert_raw_v24_to_v25.py
raw v25
   ↓
processed v14
   ↓
Zarr v7
```

After migration there is one supported raw reader: **v25 only**. Do not add a v24 compatibility reader, schema registry, migration graph, or runtime version-dispatch framework.

---

# 2. Problems in scope

| Issue | Problem | Target |
|---|---|---|
| #3 | Config mini-framework and trusted-default validation | Simple dataclasses + YAML patch + one explicit config validation boundary |
| #4 | Raw v24 mixes research data, derived features, runtime audit, diagnostics, and provenance | Minimal raw v25; preserve only useful/non-reconstructable signals and a few cheap fields needed by current processing |
| #6 | Recorder START/STOP/status are implemented as fixed SHM rings + generation/FSM protocol | Keep high-rate sample SHM; move low-rate control/result to `multiprocessing.Queue` |
| #7 | Observation intermediate dataclasses repeatedly revalidate causality, shape, dtype, contiguity, and ownership | Causal/freshness checks owned by builder; process-local dataclasses become simple containers |
| #13 | Action safety is checked repeatedly across ActionCandidate/publication/SafetyGate/worker | Two intentional layers: semantic SafetyGate + final hardware-worker guard |
| #15 | Prediction/metrics/internal dataclasses perform public-SDK-style type proof | Boundary validation only; simple internal objects and scalar metrics |
| #17 | Camera IPC/raw schema carries large provenance and diagnostic telemetry surface | Keep online causality/admission data and calibration; remove recording-only provenance/diagnostics |
| #18 | Generic serialization helper is dead abstraction | Delete directly |

---

# 3. Non-goals

Do **not** introduce any of the following while implementing this guide:

```text
Hydra / OmegaConf migration
Pydantic model layer
new generic config library
schema registry
versioned Reader hierarchy
raw-v24 runtime compatibility
new dataset format
new recorder transaction framework
new event bus
new IPC abstraction layer
new metrics framework
new safety framework
new policy API
cross-repository preprocessing migration
```

Do not redesign working robot-learning algorithms during this refactor. In particular, do not change:

```text
policy action semantics
policy inference cadence
stale-prefix scheduling
IK algorithm
retargeting algorithm
point-cloud algorithm
processed-v14 feature definitions
Zarr-v7 feature definitions
```

unless required to fix a concrete bug discovered by tests.

---

# 4. Red-line invariants

These mechanisms are real safety/correctness boundaries and must survive the refactor.

## 4.1 Physical / concurrency safety

Keep:

```text
run_generation / command authority
latest command ticket check
command expiry
worker process death detection
heartbeat timeout
final SDK-boundary finite check
final mechanical joint-limit check
arm command-jump guard
workspace / collision / operational-limit SafetyGate
```

The intended safety structure is:

```text
policy / teleop target
        ↓
controller SafetyGate
  operational limits
  hand delta
  workspace
  collision
        ↓
coupled command publication
        ↓
arm / hand worker
  current authority / ticket
  expiry
  finite
  mechanical limits
  arm jump where applicable
        ↓
SDK
```

Two layers are intentional. More than two layers for the same numeric bound are not.

## 4.2 Temporal correctness

Keep causal source selection:

```text
source_monotonic_ns > 0
source_monotonic_ns <= publish_monotonic_ns <= observation_anchor
```

For camera, keep the stronger online order currently required by the causal reader:

```text
source <= receive <= publish <= anchor
```

Keep freshness / max-skew admission in the observation builder.

## 4.3 Recording correctness

Keep:

```text
single RecorderIO owner
sample FIFO ownership / overwrite protection
STOP drains through a known final sample sequence
temp episode directory
writer/camera errors fail the episode
close files before publication
atomic temp -> final rename
```

Do not keep full post-write semantic revalidation merely as proof that the writer wrote what it already owned.

## 4.4 Camera / calibration correctness

Keep metadata required to reconstruct metric RGB-D / point cloud data:

```text
camera serial
camera type
native depth/color intrinsics
native distortion metadata
T_color_from_depth
T_xarm_base_from_color (eye-to-hand)
depth_scale
camera payload mode / aligned-depth semantics
```

Do not remove rigid-transform or serial checks from calibration.

---

# 5. Validation policy for the final codebase

Use this rule everywhere:

```text
External input  -> validate once
Policy output   -> validate once
Physical safety -> semantic controller check
SDK boundary    -> final mechanical + authority guard

Trusted process-local dataclass/helper -> type hints + tests, no runtime proof framework
```

Examples:

```text
YAML unknown key                 KEEP
invalid calibration transform    KEEP
policy output NaN                KEEP / fail closed
policy action wrong shape        KEEP / fail closed
worker mechanical-limit check    KEEP
worker generation/expiry check   KEEP

Prediction exact Python int      DELETE
Prediction reject bool-as-Real   DELETE
internal ndarray readonly copy   DELETE
internal C-contiguous proof      DELETE
config default tuple-type proof  DELETE
metric NaN raising rollout error DELETE
```

Diagnostic data must not be allowed to stop a valid robot rollout. If a metric is invalid, ignore it or leave it unset. If an action is invalid, fail closed.

---

# 6. Target raw v25

Raw v25 should minimize **code contract**, not merely bytes per row. A cheap field should remain if removing it forces a larger compatibility or quality-processing mechanism.

Keep the published episode layout:

```text
episode_xxx/
├── data.h5
├── depth.h5
└── rgb.mp4
```

This avoids unnecessary camera-storage and visualization migration.

## 6.1 Required datasets

The following is the target required set for `data.h5`.

### Timeline

```text
timestamp                    float64 [N]
source_sample_index          int64   [N]
fill_reason                  uint8   [N]
flag_sample_valid            bool    [N]
```

Reason for retaining these three legacy-looking fields:

1. existing valid v24 episodes can be projected to v25 without filtering/re-encoding RGB;
2. existing cleaning semantics already understand SOURCE/HOLD/PLACEHOLDER;
3. new v25 collection can trivially write:

```text
source_sample_index = 0,1,2,...
fill_reason         = SOURCE
flag_sample_valid   = True
```

They cost little and remove the need for a compatibility Reader or optional legacy branch.

### Robot state

```text
arm_qpos                    float64 [N,7]
arm_qvel                    float64 [N,7]
arm_tau                     float64 [N,7]

hand_qpos                   float64 [N,12]
hand_current                float64 [N,12]
hand_contact                float64 [N,5,3]
hand_tactile_force          float64 [N,5,120,3]

arm_connected               bool    [N]
hand_connected              bool    [N]
hand_qpos_stale             bool    [N]

tracking_error              float64 [N]
arm_last_cmd_seq            int64   [N]
```

`tracking_error` and `arm_last_cmd_seq` remain because the current temporal-quality detector uses them and their storage cost is negligible relative to rewriting that subsystem solely to remove two fields.

### Action

```text
action_arm_joint_sent       float64 [N,7]
action_hand_joint           float64 [N,12]
action_arm_ee               float64 [N,9]

flag_action_queued          bool    [N]
flag_frame_status           uint8   [N]
```

`action_arm_joint_sent` is the authoritative arm joint action for training and is required by current processed-v14 generation. Never fabricate it from `action_arm_joint` when migrating an episode that does not contain a sent stream.

`action_arm_ee` remains because it is control intent and cannot be reconstructed exactly from the final joint target.

Frame status values remain compact and sufficient:

```text
0 OK
1 HELD
2 IK_FAIL
3 SAFETY_REJECT
4 RETARGET_FAIL
```

Do not keep separate `flag_ik_ok`, `flag_ik_attempted`, `flag_retarget_ok`, `flag_held`, and `flag_safety_reject` in v25. Update cleaner logic to use `flag_frame_status`.

### Source / observation validity

```text
observation_anchor_monotonic_ns     uint64 [N]
observation_valid                   bool   [N]

arm_source_monotonic_ns             uint64 [N]
hand_source_monotonic_ns            uint64 [N]
tactile_source_monotonic_ns         uint64 [N]
vr_source_monotonic_ns              uint64 [N]
camera_source_monotonic_ns          uint64 [N]

tactile_fresh                       bool   [N]
tactile_calibrated                  bool   [N]
tactile_unit_code                   uint8  [N]

flag_camera_fresh                   bool   [N]
camera_depth_frame_number           uint64 [N]
camera_color_frame_number           uint64 [N]
```

Do not persist source/publish/receive proof vectors, observation-age vectors, observation-skew vectors, history masks, or source sequence IDs.

### Visual-policy aligned robot state

Keep only:

```text
policy_observation_arm_qpos         float64 [N,7]
policy_observation_hand_qpos        float64 [N,12]
policy_observation_valid            bool    [N]
```

These fields deliberately remain redundant with control-grid state because existing visual processed-v14 logic consumes the robot state causal to camera exposure. Keeping 19 floats plus one bool per row is cheaper and safer than redesigning that alignment during this refactor.

Delete all provenance used only to prove these three fields after construction.

### Raw teleoperation signal

Keep:

```text
vr_wrist_pos                        float64 [N,3]
vr_wrist_rot6d                      float64 [N,6]
vr_landmarks                        float64 [N,21,3]
head_quat_wxyz                      float64 [N,4]
```

These signals are not reconstructable from robot action and may be required for future retargeting studies.

## 6.2 Required metadata

Keep the metadata needed by collection, task labeling, RGB-D reconstruction, and reproducibility:

```text
schema_version = 25
num_frames
control_hz

task_label
operator

camera_name
camera_serial
camera_type
camera_payload_mode
depth_scale

camera_depth_intrinsics
camera_depth_width
camera_depth_height
camera_depth_distortion_model
camera_depth_distortion_coeffs

camera_color_intrinsics
camera_color_width
camera_color_height
camera_color_distortion_model
camera_color_distortion_coeffs

camera_T_color_from_depth
camera_T_xarm_base_from_color          # when applicable
camera_T_xarm_base_from_depth          # optional if already used by tools
camera_T_eef_from_depth                # when applicable

real_git_commit / existing compact code provenance if already cheap to provide
```

Do not create a new metadata framework. If a current metadata attribute is harmless and removing it creates more code than it deletes, it may remain. The priority is deleting metadata whose only purpose is runtime audit/proof and whose production requires dedicated IPC channels.

## 6.3 Delete from raw v25

Remove these categories from schema, frame construction, recorder sample IPC, raw semantic validators, docs, and tests.

### Derived geometry

```text
arm_ee
hand_fingertip
hand_tactile_contact
```

Processed v14 already recomputes fingertip FK; state EEF is reconstructable from arm qpos. `hand_contact` stays because it is the compact raw tactile sum used by current processed contact-force output.

### Policy observation proof

```text
policy_observation_reference_monotonic_ns
policy_observation_arm_source_sequence
policy_observation_hand_source_sequence
policy_observation_arm_source_monotonic_ns
policy_observation_hand_source_monotonic_ns
policy_observation_arm_publish_monotonic_ns
policy_observation_hand_publish_monotonic_ns
policy_observation_skew_s
```

### Observation audit/provenance

```text
observation_id
arm_source_sequence
hand_source_sequence
vr_source_sequence
camera_source_sequence

arm_publish_monotonic_ns
hand_publish_monotonic_ns
vr_publish_monotonic_ns
camera_publish_monotonic_ns

observation_source_receive_monotonic_ns
observation_source_age_s
observation_source_skew_s
observation_history_valid_mask
observation_skew_s
```

### Action runtime proof / ACK

```text
action_id
action_created_monotonic_ns
action_target_monotonic_ns
action_valid_until_monotonic_ns
hand_accepted_target_action_id
```

### Duplicate action/debug representations

```text
action_arm_joint
action_arm_joint_raw
action_hand_joint_raw
target_pos_before_clamp
target_eef_pos_raw
target_eef_rot6d_raw
```

The only arm joint action persisted by v25 is `action_arm_joint_sent`.

### Camera diagnostic/audit telemetry

```text
camera_health                     # runtime may keep; raw does not need it
camera_generation                 # runtime may keep; raw does not need it
camera_clock_reset
camera_duplicate
camera_frame_gap
camera_backlog_s
camera_delivery_delay_above_floor_s
camera_age_s
camera_ring_sequence
camera_depth_device_timestamp_s
camera_color_device_timestamp_s
camera_wait_return_monotonic_ns
camera_payload_ready_monotonic_ns
camera_depth_timestamp_domain
camera_color_timestamp_domain
pointcloud_valid_depth_ratio
```

### Profiling

```text
ik_solve_time_ms
policy_map_time_ms
hand_retarget_time_ms
transition_check_time_ms
policy_compute_time_ms
```

Profiling belongs in logs/result files, not the permanent raw schema.

---

# 7. New v25 collection timing semantics

New v25 collection must **not** use `recording/timeline.py::TimestampAlignedBuffer`.

Current control code already produces a complete controller-grid recording sample with an authoritative observation anchor. Record the sample that was actually emitted.

Target:

```text
controller emits sample @ t0  -> store row t0
controller emits sample @ t1  -> store row t1
controller misses / pauses t2  -> store nothing
controller emits sample @ t3  -> store row t3
```

Do not synthesize a hold row for t2.

For every newly collected v25 row:

```text
source_sample_index = sequential persisted row index
fill_reason         = SOURCE
flag_sample_valid   = True
```

`timestamp` is the actual controller/observation anchor used for that row. `grid_dt_s = 1/control_hz` remains a nominal period in metadata. Offline processing identifies a segment boundary when timestamp spacing is not approximately one nominal `grid_dt_s`.

Converted v24 episodes retain their historical HOLD/PLACEHOLDER rows. The cleaner drops them using the same `fill_reason/flag_sample_valid` rule. This permits `depth.h5` and `rgb.mp4` to remain byte-identical/hard-linked during migration.

---

# 8. One-shot v24 -> v25 converter

Create:

```text
tools/convert_raw_v24_to_v25.py
```

The converter is a **frozen historical utility**, not part of the runtime architecture.

## 8.1 Rules

It must:

1. import only stable generic libraries (`argparse`, `pathlib`, `os`, `shutil`, `h5py`, optionally `numpy`);
2. **not import `EpisodeReader`, current raw schema, dataset processing, or recorder code**;
3. open raw v24 directly with `h5py`;
4. require `meta.schema_version == 24`;
5. require `action_arm_joint_sent`; if missing, fail that episode instead of fabricating an action;
6. copy only the v25 dataset set from section 6;
7. copy small useful metadata, set `schema_version=25`, `converted_from_schema=24`, and preserve `num_frames`;
8. hard-link `depth.h5` and `rgb.mp4` by default when possible; fall back to `copy2` on cross-filesystem `EXDEV`;
9. write to a temporary destination and rename only after verification;
10. never modify or overwrite source v24 data;
11. support one episode and a directory of episodes;
12. print per-episode success/failure and a final summary.

Do not add:

```text
checksums
migration manifests
rollback journals
full RGB decode
video re-encode
schema graph
in-place mode
```

## 8.2 Conversion mapping

Most retained fields are exact same-name copies. The converter should use a single constant tuple/list such as `KEEP_DATASETS` and `src.copy(name, dst)` so HDF5 data does not need to materialize in Python.

The v25-required fields copied from v24 are:

```text
timestamp
source_sample_index
fill_reason
flag_sample_valid

arm_qpos
arm_qvel
arm_tau
hand_qpos
hand_current
hand_contact
hand_tactile_force
arm_connected
hand_connected
hand_qpos_stale
tracking_error
arm_last_cmd_seq

action_arm_joint_sent
action_hand_joint
action_arm_ee
flag_action_queued
flag_frame_status

observation_anchor_monotonic_ns
observation_valid
arm_source_monotonic_ns
hand_source_monotonic_ns
tactile_source_monotonic_ns
vr_source_monotonic_ns
camera_source_monotonic_ns
tactile_fresh
tactile_calibrated
tactile_unit_code
flag_camera_fresh
camera_depth_frame_number
camera_color_frame_number

policy_observation_arm_qpos
policy_observation_hand_qpos
policy_observation_valid

vr_wrist_pos
vr_wrist_rot6d
vr_landmarks
head_quat_wxyz
```

If a real v24 episode reveals that a listed field is not actually mandatory in the v24 writer, stop and classify it before changing the converter. Do not silently invent missing values.

## 8.3 Converter verification

Verify only cheap structural facts:

```text
output data.h5 exists
schema_version == 25
all required datasets exist
all required datasets have first dimension N
depth.h5 exists and /depth first dimension is N
rgb.mp4 exists and is non-empty
```

Do not full-decode RGB as part of every conversion. Source v24 is assumed to have already been a valid published episode.

---

# 9. Golden regression strategy

Before deleting v24 logic, create a local regression baseline from representative real episodes if available.

Use at least:

```text
1 normal teleop episode
1 episode containing camera / tactile data used by visual profile
1 episode with at least one short pause / nontrivial frame status if available
```

Before refactor:

```text
raw v24
  ↓ old processing
processed-v14-before
  ↓ old export
zarr-v7-before
```

After converter + v25 refactor:

```text
same raw v24
  ↓ converter
raw v25
  ↓ new processing
processed-v14-after
  ↓ same export contract
zarr-v7-after
```

Compare **policy-relevant output**, not removed audit metadata.

Required equality / numerical equivalence:

```text
episode count
episode lengths / segment boundaries
state
action
eef_pose
fingertip_points
contact_force
tactile_force
RGB
depth or generated pointcloud, depending profile
task labels
```

Use `array_equal` for discrete data and `allclose` for derived floating-point arrays. If a difference occurs, identify whether it is:

```text
A. accidental semantic regression -> fix
B. deliberate removal of an old over-defensive row rejection -> document and inspect
```

Do not force old quality/debug JSON to be byte-identical.

If real episode data is not available in the Codex workspace, create synthetic v24 fixtures for automated tests and leave a clearly documented command for the user to run the real-data regression before deleting the v24 archive.

---

# 10. Phased implementation

Each phase should be independently reviewable. Prefer one commit per phase or per tightly coupled sub-phase. Do not mix raw-schema migration with safety/control changes in one commit.

---

## Phase 0 — Freeze migration path and baseline

### Goal

Make it impossible for later cleanup to strand historical data.

### Add

```text
tools/convert_raw_v24_to_v25.py
tests/test_raw_v24_to_v25_conversion.py
```

### Actions

1. Implement converter exactly as section 8.
2. Build a small synthetic v24 fixture using the current schema names.
3. Verify retained arrays are exact copies.
4. Verify removed arrays are absent.
5. Verify wrong schema is rejected.
6. Verify missing `action_arm_joint_sent` is rejected.
7. Verify source episode is never changed.
8. Verify media hard-link/copy path.
9. Produce local real-data golden processed/Zarr outputs if data is available.

### Exit criteria

```text
converter tests pass
converter has no dexmani_real imports
source v24 remains untouched
at least one v24 fixture converts to structurally valid planned-v25 output
```

Do not change the production Reader or writer yet.

---

## Phase 1 — #18 Delete generic serialization

### Goal

Remove dead abstraction before simplifying config.

### Inspect / edit

```text
dexmani_real/utils/serialization.py
planning configuration classes containing from_dict()
all imports of from_dict_helper
```

### Actions

Delete:

```text
utils/serialization.py
MotionPlanningConfig.from_dict()
OnlineIKConfig.from_dict()
associated imports
```

Do not replace the helper with another generic deserializer.

### Exit criteria

Repository search returns zero hits for:

```text
from_dict_helper
utils.serialization
MotionPlanningConfig.from_dict
OnlineIKConfig.from_dict
```

Run affected planning/config tests.

---

## Phase 2 — #3 Simplify config

### Goal

Retain convenient nested config access and YAML override while deleting the generic typed configuration framework.

### Main files

```text
dexmani_real/config/experiment.py
dexmani_real/config/defaults.py
dexmani_real/config/pointcloud.py
callers of canonical_json / canonical_yaml / generic config rebuild
```

### Target API

Conceptually:

```python
cfg = load_config(yaml_path=None, overrides=None)
validate_config(cfg)
```

Implementation may keep `ExperimentConfig` / nested dataclasses if renaming would create churn.

### Keep

```text
nested dataclasses
source-code defaults
YAML loading
CLI dotted override if actively used
unknown YAML key rejection
calibration resolution
cross-field safety invariants
```

### Required explicit invariants

At minimum keep validation for:

```text
lower < upper
home within robot joint limits
operational hand limits inside mechanical hand limits
valid workspace bounds
valid collision/table geometry
positive control frequencies / periods where required
```

This phase must establish:

```text
hand operational lower >= hand mechanical lower
hand operational upper <= hand mechanical upper
```

so Phase 5 can remove duplicate controller mechanical checking safely.

### Delete

```text
generic get_type_hints/get_origin/get_args reconstruction
recursive type-proof framework
canonical_json if no real consumer
stored canonical_yaml property if only printing needs it
MappingProxyType / pickle machinery used only for config immutability
trusted-default bool/int/tuple/string proof
most __post_init__ type pedantry
```

For `--print-config`, serialize the actual dataclass with a small direct conversion rather than maintaining a canonical configuration subsystem.

### Tests

Keep tests for behavior:

```text
valid defaults load
valid YAML override changes expected field
unknown key fails
invalid nested safety relation fails
operational limits outside mechanical limits fail
```

Delete tests whose only purpose is proving source-code literals have exact Python runtime types.

### Exit criteria

```text
one obvious config load path
one obvious config validation boundary
no generic typed deserializer
robot startup/config tests pass
```

---

## Phase 3 — #15 Simplify Prediction and metrics

### Goal

Move validation to real boundaries and make internal telemetry passive.

### Main files

```text
dexmani_real/deployment/prediction.py
dexmani_real/deployment/metrics.py
dexmani_real/deployment/inference/worker.py
prediction IPC serialization/deserialization sites
tests/test_policy_rollout.py
```

### Prediction target

Use a plain dataclass/slots object containing the existing fields needed by scheduling and metrics:

```text
run_generation
source_monotonic_ns
logical_step_monotonic_ns
actions
inference_latency_ms
observation_age_ms
observation_skew_ms
```

The exact field set may follow current code, but `Prediction` itself should not prove exact scalar classes, uint64 ranges, C-contiguity, readonly ownership, etc.

### Boundary validation

Immediately after policy inference:

```text
cast action output to float64 once
check expected 2-D shape / action_dim
check finite
```

Then publish.

Do not repeat the same proof in `Prediction`, metrics, executor, and IPC packer.

### Metrics target

Replace deques and `observe_*()` methods with latest scalar fields plus counters.

Conceptually:

```python
@dataclass
class PolicyStats:
    inference_latency_ms: float | None = None
    observation_age_ms: float | None = None
    observation_skew_ms: float | None = None
    schedule_lateness_ms: float | None = None
    publication_interval_ms: float | None = None
    skipped_prefix_steps: int | None = None

    safety_rejection_count: int = 0
    ik_rejection_count: int = 0
    stale_prediction_count: int = 0
```

If a diagnostic scalar is non-finite, leave it unset; do not raise from the robot rollout path.

### Tests

Keep:

```text
wrong policy action shape rejected at inference boundary
NaN/Inf policy action rejected at inference boundary
valid prediction schedules identically
metric snapshot contains expected latest values/counters
```

Delete tests for exact Python scalar type acceptance/rejection or readonly flags.

---

## Phase 4 — #7 Simplify observation internals

### Goal

Keep temporal correctness while removing redundant process-local value-object validation.

### Main file

```text
dexmani_real/deployment/inference/observation.py
```

### Ownership

`ipc/causal.py` owns raw causal source ordering.

Observation builder owns:

```text
freshness
history availability
max observation skew
camera generation consistency where still required online
requested modality availability
state/camera temporal pairing
```

Policy input boundary owns only final model-facing requirements that are not already guaranteed by construction.

### Delete / simplify

Turn these into simple containers:

```text
FrameWindow
PointCloudFrame
RgbFrame
ObservationBatch
PolicyObservation
```

Remove their defensive `__post_init__` proof code where the builder already guarantees the invariant.

Do not add a replacement validator class.

### Keep algorithmic checks

Do not delete checks that cause the builder to reject/return no observation for:

```text
future source
stale source
insufficient history
cross-generation camera history
excessive observation skew
missing requested modality
```

### Tests

Replace constructor-exception tests with behavior tests:

```text
future frame is never selected
stale frame is rejected
insufficient history yields no observation
camera generation boundary does not leak old frames
valid history produces correct T x D arrays
valid visual/pointcloud modalities preserve shapes
```

Do not change the `dexmani_policy` API.

---

## Phase 5 — #13 Collapse safety to two layers

### Goal

Preserve real physical safety while deleting repeated same-layer numerical validation.

### Main files

```text
dexmani_real/control/action.py
dexmani_real/control/safety_gate.py
dexmani_real/control/publication.py
dexmani_real/robot/command_validation.py
dexmani_real/robot/arm_worker.py
dexmani_real/robot/hand_worker.py
related tests
```

### Controller layer keeps

```text
operational arm/hand limits
hand endpoint delta / slew semantics if controller-owned
workspace
collision / transition safety
```

### Worker layer keeps

```text
current command authority / generation / ticket
expiry / latest-command check
finite
mechanical limits
arm jump guard
SDK error -> fault
```

### Delete

1. `ActionCandidate` array copying/readonly/type-proof where producer owns the arrays.
2. Controller mechanical hand-limit check performed immediately after an operational-limit SafetyGate, **only after Phase 2 guarantees operational limits are inside mechanical limits**.
3. Repeated shape/finite proof in intermediate publication objects when a boundary already guarantees it.
4. Hyper-detailed 17-digit per-joint diagnostic renderers.

Use compact diagnostics such as:

```text
hand_joint_limit:j6
hand_delta_limit:j3:0.142>0.120
workspace
collision
```

### Never delete

```text
worker final mechanical limit guard
worker final finite guard
worker authority/ticket/expiry guard
arm jump guard
```

### Tests

Behavior tests must prove:

```text
semantic operational-limit violation rejected before publication
workspace/collision rejection still works
stale generation/ticket cannot reach SDK
expired command cannot reach SDK
worker independently rejects mechanical-limit violation
worker independently rejects NaN/Inf
arm jump guard still faults/rejects as before
```

---

## Phase 6 — #4 Introduce the only supported raw schema: v25

This is the main vertical refactor. Do not combine it with Phase 7 camera-IPC cleanup or Phase 8 recorder control-plane cleanup until v25 roundtrip and processed-v14 output are stable.

### 6A — Replace schema/reader contract

Main files:

```text
dexmani_real/recording/storage/schema.py
dexmani_real/recording/storage/reader.py
dexmani_real/recording/storage/hdf5_writer.py
docs/data_schema.md
raw-schema tests
```

Actions:

1. set `EPISODE_SCHEMA_VERSION = 25`;
2. replace v24 dataset specs with section 6 required datasets;
3. remove semantic constants/IDs whose only purpose was validating deleted fields;
4. reduce `validate_data_layout()` to required names, shapes/dtypes, and consistent first dimension;
5. delete large `validate_raw_semantics()` proof machinery unless a remaining check protects a real downstream invariant;
6. make `EpisodeReader` v25-only;
7. Reader checks only structural requirements needed to safely read:

```text
episode directory exists
data.h5 / depth.h5 / rgb.mp4 exist
schema_version == 25
required dataset names exist
first dimensions == num_frames
depth first dimension == num_frames
```

Do not full-decode RGB on Reader construction or episode finalization.

### 6B — Simplify frame/sample construction

Main files:

```text
dexmani_real/recording/sample.py
dexmani_real/recording/frame.py
dexmani_real/recording/client.py
dexmani_real/ipc/schema.py::make_record_sample_dtype
teleop/episode_samples.py
deployment/executor.py recording signal construction
```

Actions:

1. stop computing/storing `arm_ee` and `hand_fingertip` solely for raw recording;
2. retain raw tactile sum (`hand_contact`) and full tactile force;
3. collapse frame flags to `flag_frame_status` + `flag_action_queued`;
4. stop constructing deleted observation/action/camera/profiling metadata;
5. shrink record sample dtype to values needed by raw v25 plus camera payload;
6. make `EpisodeFrame` a plain owned container; remove MappingProxyType/readonly proof machinery;
7. keep copies only where required to cross mutable SHM ownership safely.

Do not delete kinematics used by control/deployment; only stop using derived kinematics as raw recording fields.

### 6C — Remove second recorder time grid

Main files:

```text
dexmani_real/recording/timeline.py
dexmani_real/recording/recorder.py
dexmani_real/recording/storage/hdf5_writer.py
related tests
```

Actions:

1. stop routing new source frames through `TimestampAlignedBuffer`;
2. write each accepted controller frame exactly once;
3. assign `source_sample_index` sequentially;
4. set `fill_reason=SOURCE`, `flag_sample_valid=True` for every newly collected row;
5. preserve actual frame timestamp; do not manufacture hold rows;
6. batch disk appends in a small implementation-only buffer (e.g. 32 rows) if useful for IO efficiency;
7. flush remaining rows at episode stop.

The batch buffer has **no temporal semantics**.

Once no production/test code needs it, delete `recording/timeline.py`.

### 6D — Simplify offline cleaner for v25

Main files:

```text
dexmani_real/dataset/clean.py
dexmani_real/dataset/quality.py only if signatures need minor cleanup
dexmani_real/dataset/processing.py
tests/test_processed_v14.py
```

The cleaner should become a research-data cleaner, not a runtime audit replayer.

Base row validity should be approximately:

```text
SOURCE sample
AND flag_sample_valid
AND flag_action_queued
AND acceptable flag_frame_status
AND arm_connected
AND hand_connected
AND not hand_qpos_stale
AND finite state/action
AND action inside mechanical limits
AND valid tactile source for profiles that require it
AND observation_valid
```

Visual profile additionally requires:

```text
flag_camera_fresh
policy_observation_valid
valid depth payload
```

Delete offline re-proofs based on removed data:

```text
action timing proof
camera health taxonomy replay
camera generation/clock-reset/duplicate revalidation
device timestamp monotonic proof
recorded observation skew consistency
policy observation source/publish sequence proof
flag_ik_ok / flag_retarget_ok consistency warnings
```

Use `flag_frame_status` directly:

```text
OK -> normal candidate
IK_FAIL -> existing short-transient-vs-long-run policy may remain, based only on status run length
HELD / SAFETY_REJECT / RETARGET_FAIL -> reject unless an existing explicit behavior is required
```

Keep `select_tactile_rows_to_references()` exact and O(N log N); do not rescan each persisted prefix (O(N²)) or assume source timestamps are monotonic. A small coordinate-compressed Fenwick prefix-max can track the largest active source coordinate, with a separate latest-row array for duplicate timestamps. Activate only rows reached in persisted order, then query coordinates at or before the reference. Do not introduce occupancy/rank machinery. Preserve:

```text
chosen tactile row <= current persisted row
largest proven source timestamp; ties choose the latest persisted row
source > 0 and hand_source == tactile_source
source <= reference
fresh
calibrated
unit_code == 0
max skew
```

Do not let a future persisted row repair an earlier observation.

### 6E — Preserve processed v14

`dataset/processing.py` may change how it obtains raw inputs, but its output schema/semantics remain processed v14.

Keep/recompute:

```text
fingertip_points from joint state
EEF pose from joint state
contact_force from retained hand_contact source
tactile_force from retained full tactile source
visual policy-aligned state from policy_observation_*_qpos
RGB/depth/pointcloud behavior
```

### Phase 6 tests

At minimum:

```text
v25 writer -> v25 reader roundtrip
new v25 collection writes no synthetic HOLD/PLACEHOLDER rows
timestamp gaps remain gaps
converted v24 HOLD/PLACEHOLDER rows are rejected by cleaner
v25 -> processed v14 fixture passes
visual processed fixture passes
pointcloud processed fixture passes if environment supports it
fingertip output is recomputed and matches prior behavior
```

Then run the golden regression in section 9.

### Exit criteria

Do not proceed to Phase 7 until:

```text
v25-only Reader works
new writer produces v25
converted v24 produces v25
processed-v14 tests pass
Zarr export contract is unchanged
golden real-data comparison is accepted or clearly documented for user verification
```

---

## Phase 7 — #17 Reduce camera/provenance telemetry

Only perform this after raw v25 no longer consumes deleted camera/provenance fields.

### Main files

```text
dexmani_real/ipc/schema.py
dexmani_real/ipc/camera_ring.py
dexmani_real/ipc/channels.py
dexmani_real/ipc/causal.py
dexmani_real/sensor/camera/worker.py
dexmani_real/sensor/pointcloud_worker.py
dexmani_real/teleop/session.py
dexmani_real/teleop/control_loop/camera_freshness.py
dexmani_real/deployment/inference/observation.py
dexmani_real/deployment/executor.py
recording/client.py / io_worker.py metadata paths
```

### Online camera header target

Keep only values with a live consumer:

```text
source_monotonic_ns
receive_monotonic_ns
publish_monotonic_ns
camera_generation
depth_frame_number
color_frame_number
camera_health
```

`camera_health` may encode reset/duplicate/gap/delay conditions. Before deleting separate `clock_reset`/`duplicate` flags, add/keep a test that proves such a frame is never reported as `camera_health == OK` if online admission relied on the separate flag before.

The exact final field list may retain one extra field if a real online consumer requires it. Do not retain fields solely because raw v24 used them.

### Delete from runtime IPC when no online consumer remains

```text
device timestamp values / domains
payload_ready timestamp
frame_gap flag/counter if health is authoritative
clock_reset bool if health+generation are authoritative
duplicate bool if health is authoritative
backlog / delivery delay telemetry
pc_valid_depth_ratio
recording-only camera firmware / SDK / profile shared arrays
recording-only arm/hand device identity shared arrays
```

Keep camera serial, geometry, and depth scale through the simplest existing static configuration/shared path required by calibration and raw metadata.

### Camera ring payload metadata

RGB/depth resolution is fixed when the ring is constructed. If all attach paths already know the configured shapes, remove duplicate per-frame `rgb_size`, `depth_size`, and shape fields and use ring configuration as the capacity/layout contract. Do this only after confirming there is no supported name-only attach path that needs to recover dimensions from a frame header.

Do not weaken seqlock/torn-read correctness.

### Tests

```text
healthy camera frame admitted
future camera frame rejected
stale camera frame rejected
reset/generation boundary not admitted as healthy
camera ring torn read still rejected
pointcloud worker still rejects unhealthy/noncausal source
camera calibration metadata still reaches v25 episode
```

---

## Phase 8 — #6 Replace recorder control/status SHM with queues

Do this last so control-plane changes are not mixed with raw-schema changes.

### Keep data plane

```text
record_sample_ring
recorder_consumed_sequence / equivalent consumer progress
```

This path is high-rate and large because it carries camera payload and state/action data.

### Delete control/status data plane misuse

Delete:

```text
RECORD_CONTROL_DTYPE
RECORD_STATUS_DTYPE
record_control_ring
record_status_ring
RecorderPhase wire FSM
generation transaction IDs used only by recorder lifecycle
fixed byte capacities for task/operator/reason/path/status
late START cancellation protocol
READY/RECORDING/FINALIZING/COMPLETED/ERROR/STOPPED phase protocol
```

### Add simple low-rate queues

Use standard spawn-compatible `multiprocessing.Queue`, with small bounded capacity if desired.

Conceptual messages:

```python
@dataclass
class StartRecording:
    task: str
    operator: str
    start_sequence: int

@dataclass
class StopRecording:
    save: bool
    reason: str
    through_sequence: int

@dataclass
class RecordingStarted:
    path: str

@dataclass
class RecordingFinished:
    saved: bool
    path: str | None
    frame_count: int
    reason: str
    error: str | None = None
```

Do not add request IDs/generation IDs unless a concrete concurrent-command bug proves they are necessary. The client supports one recording transaction at a time.

### START semantics

1. client snapshots `record_sample_ring.latest_sequence + 1` as `start_sequence`;
2. send `StartRecording`;
3. RecorderIO initializes temp episode and returns `RecordingStarted`;
4. producer begins sending recording samples only after start succeeds.

If start acknowledgment times out or RecorderIO dies, mark recording unavailable / abort the session through existing supervisor behavior. Do not build a cancellation transaction protocol for a worker that failed to acknowledge START.

### STOP semantics

1. stop producer from publishing new recording samples;
2. snapshot `through_sequence = record_sample_ring.latest_sequence`;
3. send `StopRecording(save, reason, through_sequence)`;
4. RecorderIO drains all committed samples through that sequence;
5. flush pending rows;
6. close camera/data writers;
7. atomically publish or discard;
8. return one `RecordingFinished`.

`through_sequence` is the important concurrency invariant. Preserve it.

### Recorder local state

Client may use only:

```text
IDLE
RECORDING
STOPPING
```

or equivalent booleans. Worker can often use simply `active_recorder is None / not None`.

Do not expose internal finalization phases over IPC.

### Tests

```text
START returns started event
STOP drains exactly through final committed sequence
samples published after producer stop are not expected
no sample <= through_sequence is lost
full sample ring / overwritten unconsumed row still fails the episode
writer error produces RecordingFinished(error=...)
discard removes temp episode
save closes files then atomically renames
next episode can start after previous finish
RecorderIO death is surfaced by existing supervisor/session behavior
```

---

# 11. Test policy during refactor

Prefer behavioral tests over defensive-value-object tests.

## Keep / add

```text
causal source selection
stale/future observation rejection
policy output NaN/shape rejection
SafetyGate behavior
worker final safety behavior
record sample FIFO ownership
record stop through_sequence drain
atomic episode publication
v24 -> v25 projection
v25 raw -> processed v14
processed -> Zarr contract
```

## Delete / rewrite

Tests whose main purpose is:

```text
internal dataclass rejects bool/int subtype
readonly ndarray flag set
C-contiguous proof
trusted config literal raises exact TypeError
Prediction exact uint64 proof
ObservationBatch constructor proves builder invariant
raw v24 audit telemetry semantic equality
RecorderPhase intermediate status ordering
```

A test should survive if an internal dataclass is replaced by a tuple without changing system behavior.

---

# 12. File-action checklist for Codex

This is a guide, not an instruction to edit all files blindly. Search call sites before deletion.

| Area | Likely files | Action |
|---|---|---|
| #18 serialization | `utils/serialization.py`, planning config | DELETE dead helper/from_dict |
| #3 config | `config/experiment.py`, `config/defaults.py`, callers | SIMPLIFY |
| #15 prediction | `deployment/prediction.py`, `deployment/metrics.py`, inference worker | SIMPLIFY |
| #7 observation | `deployment/inference/observation.py`, tests | SIMPLIFY internal contracts; KEEP causal logic |
| #13 safety | `control/action.py`, `control/safety_gate.py`, `control/publication.py`, worker validation | MERGE repeated checks; KEEP two safety layers |
| #4 converter | `tools/convert_raw_v24_to_v25.py` | ADD frozen tool |
| #4 raw schema | `recording/storage/schema.py`, `reader.py`, `hdf5_writer.py` | REPLACE v24 with v25 |
| #4 frame/sample | `recording/sample.py`, `recording/frame.py`, `recording/client.py`, `ipc/schema.py` | SHRINK |
| #4 timeline | `recording/timeline.py`, recorder call sites | DELETE after direct-row recording works |
| #4 processing | `dataset/clean.py`, `dataset/processing.py`, tests | SIMPLIFY input validation; KEEP processed-v14 output |
| #17 camera IPC | `ipc/schema.py`, `camera_ring.py`, `channels.py`, camera worker, causal consumers | SHRINK after raw v25 |
| #6 recorder protocol | `recording/client.py`, `recording/io_worker.py`, `ipc/channels.py`, `ipc/schema.py` | SHM control/status -> Queue |
| docs | `docs/data_schema.md`, obsolete migration guides | UPDATE/DELETE after implementation |

---

# 13. Execution discipline for Codex

For every phase:

1. inspect all direct call sites before editing;
2. make the smallest coherent change;
3. run focused tests first;
4. run the full repository test suite before moving to the next phase;
5. do not preserve old APIs unless an active current-main caller requires them;
6. do not add compatibility shims "just in case";
7. if a deleted validation was physical/race safety rather than programmer-error proof, restore it at the correct boundary;
8. remove newly dead imports, helpers, constants, tests, and docs in the same phase;
9. report code reduction and any semantic change at the end of each phase.

Recommended generic checks:

```bash
python -m compileall dexmani_real
python -m pytest -q
```

Use narrower test files while iterating, then run full `pytest` before phase completion.

Do not require hardware for unit tests. After all software tests pass, perform hardware verification in increasing-risk order:

```text
1. config/load startup without motion
2. camera/sensor health
3. teleop shadow/no-motion path if available
4. short low-risk recording episode
5. inspect raw v25
6. process to v14 and export Zarr
7. policy shadow
8. short low-risk policy run
9. normal teleop/policy evaluation
```

---

# 14. Final acceptance criteria

The refactor is complete only when all of the following are true.

## Repository boundary

```text
dexmani_policy modified: NO
processed v14 schema changed: NO
Policy Zarr v7 contract changed: NO
raw runtime-supported versions: v25 only
v24 compatibility Reader: NONE
```

## #18 / #3

```text
utils/serialization.py removed
generic from_dict helper removed
one simple config loading path
unknown external config keys still rejected
safety cross-field config invariants retained
```

## #15 / #7

```text
Prediction is a simple internal carrier
metrics are scalar/counter state, not deque framework
policy action checked once at inference boundary
observation process-local dataclasses have no duplicated proof machinery
causal/freshness/history behavior remains correct
```

## #13

```text
controller semantic SafetyGate remains
worker final mechanical/authority/expiry checks remain
middle duplicate bounds/type checks removed
```

## #4

```text
raw schema == v25
v24 converter exists and is self-contained
new v25 writer emits direct source rows, no recorder synthetic grid
TimestampAlignedBuffer removed from production recording
Reader validates structure, not a large runtime-audit contract
cleaner no longer replays deployment audit proofs
processed v14 output remains usable and equivalent on golden data
```

## #17

```text
camera online IPC contains only fields with live online consumers
recording-only firmware/SDK/profile/device identity channels removed
raw v25 no longer persists camera diagnostic telemetry
camera causal admission and calibration remain correct
```

## #6

```text
high-rate record samples still use SHM
START/STOP/result use multiprocessing.Queue
record control/status SHM dtypes/rings removed
no recorder generation/FSM compatibility protocol
STOP drains through final committed sample sequence
atomic episode publication remains
```

## Data migration

Before deleting/archiving old v24 data, the user must have successfully run:

```text
v24 -> converter -> v25 -> processed v14 -> Zarr v7
```

on representative real episodes and verified policy-relevant arrays/episode boundaries.

---

# 15. Desired end-state mental model

After this work, a new contributor/Codex run should be able to understand the system as five simple boundaries:

```text
Config
YAML -> load/patch -> validate once -> trusted config

Observation
sensor SHM -> causal/fresh builder -> existing policy input

Control
policy/teleop -> SafetyGate -> command IPC -> worker final guard -> SDK

Recording
controller sample -> sample SHM -> RecorderIO -> raw v25
START/STOP -----------------------> Queue

Data
raw v25 -> processed v14 -> Zarr v7 -> dexmani_policy
```

If a proposed implementation adds a sixth framework-like boundary to solve one of these eight issues, it is probably moving in the wrong direction.
