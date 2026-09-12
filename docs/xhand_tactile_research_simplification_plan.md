# XHand tactile research pipeline simplification plan

> Canonical implementation plan for Codex / Claude Code.
>
> Baseline reviewed: `haoyangzhanglab/dexmani_real` `main` on 2026-09-11, with raw v28 / processed v19 / Policy Zarr v12.
> Source code and current schemas remain authoritative if this document becomes stale.
>
> **Status: IMPLEMENTED.**
> This document defines the tactile acquisition/data-path simplification, which has since been implemented. Source and current schemas (raw v29 / processed v20 / Policy Zarr v13) are authoritative; this document does not claim hardware validation.

---

## 0. Executive decision

The next XHand tactile path should be a research-oriented clean break, not another incremental extension of the current provenance machinery.

The target is:

```text
one XHand read_state()
        ↓
one hand observation
        ├── qpos / current
        ├── aggregate tactile  [5,3]
        ├── dense tactile      [5,120,3]
        ├── aggregate_valid
        └── dense_valid
        ↓
one hand_state_ring publication
        ↓
latest causal hand observation at the control-step anchor
        ↓
raw episode
        ↓
direct representation/dtype transform
        ↓
processed dataset
```

The core rules are:

1. **One SDK read is one logical hand sample.** Do not split aggregate and dense tactile into separately selected observations.
2. **Tactile validity only answers whether the numeric payload is usable.** It must not encode freshness, contact state, physical units, data-quality heuristics, or historical provenance.
3. **Keep exactly two public tactile validity bits:** aggregate and dense. The only reason for two bits is the real XHand RS485 partial-availability behavior.
4. **Recording never searches backward to repair tactile.** If the selected hand sample has invalid tactile, record it as invalid.
5. **Raw validity is authoritative.** Processing copies validity; it does not reconstruct it from calibration/unit/freshness fields.
6. **Software zero/bias correction stays in the driver.** This task is about robot-learning data, not sensor-calibration research.
7. **Remove frame-level tactile metadata that is constant, derivable, or policy-dependent:** `fresh`, `calibrated`, `unit_code`, binary contact, and independent tactile timestamps.
8. **Do not weaken robot safety.** Command validation, joint limits, worker failure timeout, generation/ticket checks, estop, heartbeat, and shutdown remain unchanged.

If implemented immediately against the reviewed baseline, persisted schema changes are expected to be:

```text
raw v28       -> raw v29
processed v19 -> processed v20
Policy Zarr v12 -> v13  # mechanical Real export mirror only
```

If schema versions advance before implementation, bump each current schema exactly once rather than reusing these integers mechanically.

---

## 1. Scope

### 1.1 In scope

This task owns the complete `dexmani_real` tactile path:

```text
robot/drivers/xhand.py
    ↓
robot/hand_worker.py
    ↓
ipc/schema.py + ipc/channels.py + ipc/causal.py
    ↓
teleop / rollout recording boundary
    ↓
recording/sample.py + recording/frame.py + recording/storage/schema.py
    ↓
dataset/processing.py + dataset/processed.py
    ↓
dataset/export.py
    ↓
Real deployment observation assembly
```

The goal is to make that path direct, readable, and robust before designing how a policy encodes tactile.

### 1.2 Out of scope

Do not implement in this task:

```text
tactile CNN / Transformer / taxel tokenization
aggregate-vs-dense policy architecture
normalization inside dexmani_policy
modality dropout / mask tokens
high-rate tactile sidecar recording
slip-specific temporal tactile models
known-load SI/Newton calibration
vendor-taxel spatial geometry reconstruction
generic SensorManager / ModalityRegistry / status framework
legacy runtime compatibility layers
```

Do not modify `dexmani_policy` model behavior in this task. Real Policy Zarr changes are limited to mechanically mirroring the new Real processed artifact; model-side tactile consumption is a separate follow-up.

---

## 2. Current problems

The current driver already receives aggregate and dense tactile from the same XHand SDK call:

```text
read_state(..., True)
    ├── finger_state
    │    ├── qpos
    │    └── current
    └── sensor_data
         ├── calc_force [5,3]
         └── raw_force  [5,120,3]
```

However the downstream path currently separates them:

```text
HAND_STATE_DTYPE
    ├── qpos/current
    ├── tactile_sum
    └── tactile_sum_valid

HAND_TACTILE_DTYPE
    ├── tactile_force
    ├── fresh
    ├── calibrated
    └── unit_code
```

Recording then independently selects aggregate and dense tactile and persists multiple tactile-specific timestamps/flags. Processing later reconstructs validity from several of those fields.

This creates unnecessary concepts:

```text
aggregate source identity
dense source identity
aggregate freshness
dense freshness
calibration propagation
per-frame unit propagation
aggregate/dense source matching
backward search for an older valid aggregate
processed validity reconstruction
```

Most of those concepts do not represent additional sensor information. They are consequences of splitting one SDK sample into multiple logical streams.

The simplification should remove that accidental complexity rather than optimize it.

---

## 3. Reference-project conclusions

The design is intentionally no more complicated than the three reviewed reference projects except where XHand partial availability requires it.

### 3.1 PI-R2 Flow

Reference:

```text
pi-r2-flow/pi-r2-flow
  deployment/mindex/robots/xhand_robot.py
```

Useful ideas:

```text
fresh SDK read
native calc_force/raw_force values
software zero/bias subtraction
simple observation dictionary
simple recording path
```

Do not copy:

```text
any-nonzero-error => discard all tactile
unverified Newton naming
```

DexMani should remain slightly more precise only because RS485 can expose aggregate force while dense force is unavailable.

### 3.2 DexUMI

Reference:

```text
real-stanford/DexUMI
  dexumi/hand_sdk/xhand/hand_api_cls.py
```

Useful idea:

```text
failed read does not create a new successful sensor sample
```

Do not copy:

```text
returning latest queued state without explicit age semantics
linear interpolation of tactile/FSR for causal policy observations
```

DexMani already has a control-step causal timeline. Keep latest-causal selection and do not interpolate tactile.

### 3.3 KineDex

Reference:

```text
DinoMini00/KineDex_code
  diffusion_policy/real_world/xhand_interpolation_controller.py
```

Useful idea:

```text
dense tactile [5,120,3] is ordinary robot-learning observation data
```

Do not copy:

```text
failed read => return cached previous state as if current
aggregate /255 magic scaling
```

### 3.4 Final complexity target

```text
PI-R2 / DexUMI:
    effectively one tactile-valid observation

Target DexMani:
    aggregate_valid
    dense_valid
```

The extra boolean is justified only by the XHand partial-status semantics. Do not add validity enums, reason bitmasks, health classes, status registries, or quality scores.

---

## 4. Canonical tactile terminology

Use the following meanings consistently in runtime code:

```text
tactile_aggregate
    XHand sensor_data[*].calc_force
    shape [5,3]

tactile_dense
    XHand sensor_data[*].raw_force
    shape [5,120,3]

tactile_aggregate_valid
    this SDK read contains a usable canonical aggregate payload

tactile_dense_valid
    this SDK read contains a usable canonical dense payload
```

`aggregate` and `dense` describe representation, not physical units.

Persisted raw/processed field names may retain the existing public vocabulary where doing so avoids gratuitous cross-repository churn:

```text
raw hand_contact          = runtime tactile_aggregate
raw hand_tactile_force    = runtime tactile_dense
processed contact_force   = raw hand_contact
processed tactile_force   = raw hand_tactile_force
```

Do not interpret `hand_contact` / `contact_force` as a binary contact flag. They are continuous `[5,3]` aggregate force-like sensor vectors.

Binary contact is a derived representation and is not canonical sensor data.

---

## 5. Validity: one small rule, two booleans

### 5.1 Definition

The canonical definition is:

```text
valid
= SDK status allows this representation
  AND payload parses with the expected shape
  AND payload is finite
  AND session software-bias initialization succeeded
```

Nothing else belongs in tactile validity.

Validity explicitly does **not** encode:

```text
freshness / age
contact vs no-contact
force magnitude thresholds
Newton/SI calibration
unit identity
camera/state synchronization
temporal skew
data quality scores
historical schema provenance
```

Recommended code comment:

```python
# Tactile validity only means that this SDK read provides a usable,
# finite payload for the canonical bias-corrected representation.
# It does not encode freshness, contact state, physical units,
# temporal alignment, or data-quality heuristics.
```

### 5.2 SDK availability helper

Keep one pure helper in `robot/drivers/xhand.py`:

```python
def _tactile_validity(code: int | None, *, comm_type: str) -> tuple[bool, bool]:
    """Return (aggregate_allowed, dense_allowed) for one SDK read status."""
```

Required serial / RS485 behavior:

| SDK code | Meaning | aggregate | dense |
|---|---|---:|---:|
| `0` | OK | true | true |
| `1501018` | combined force unavailable | false | false |
| `1501019` | distributed force unavailable | true | false |
| `1501020` | temperature unavailable | true | true |
| CRC / any other nonzero | unavailable / uncertain | false | false |

EtherCAT remains conservative:

```text
0       -> (true, true)
nonzero -> (false, false)
```

Do not add a registry or class hierarchy around these four known statuses.

### 5.3 Parse validation

Aggregate parse succeeds only when:

```text
shape == (5,3)
all finite
```

Dense parse succeeds only when:

```text
shape == (5,120,3)
all finite
exactly 120 raw_force entries per finger
```

A parse failure invalidates only the affected representation.

### 5.4 Session zero/bias readiness

The driver keeps software bias private. `calibrate_tactile()` / future `zero_tactile()` returns one startup success boolean.

The hand worker combines startup readiness with per-read payload validity:

```python
aggregate_valid = tactile_ready and state.tactile_aggregate_valid
dense_valid = tactile_ready and state.tactile_dense_valid
```

Do not publish `tactile_calibrated` as a frame field.

If bias initialization fails:

```text
joint control continues
tactile aggregate_valid = false
tactile dense_valid = false
```

Tactile failure must not unnecessarily disable safe joint control.

---

## 6. Software bias / zeroing

Keep the current software bias concept because it materially stabilizes tactile observations and is already simpler than building calibration into the dataset.

Canonical numeric representation remains:

```text
SDK native numeric value - software no-contact bias
```

Do not reintroduce:

```text
0.1 scaling
/255 scaling
unverified SI conversion
```

### 6.1 Default startup path

Keep the current compact behavior unless hardware evidence motivates a change:

```text
collect 5 fresh valid no-contact samples
    ↓
mean aggregate bias + mean dense bias
    ↓
publish candidate biases
    ↓
3 fresh verification reads
```

The existing small aggregate residual check may remain a startup zeroing sanity check. It must not become a frame-level validity rule.

Do not add retry state machines, temperature compensation, robust estimators, or vendor-reset orchestration in this task.

### 6.2 Naming

`calibrate_tactile()` currently means software no-contact zeroing, not physical SI calibration. A later local rename to `zero_tactile()` / `estimate_tactile_bias()` is reasonable if it falls naturally out of the refactor, but do not create a rename-only compatibility wrapper.

---

## 7. Driver target

Primary file:

```text
dexmani_real/robot/drivers/xhand.py
```

Target state object:

```python
@dataclass
class XHandState:
    qpos: np.ndarray
    current_ma: np.ndarray

    tactile_aggregate: np.ndarray      # [5,3]
    tactile_dense: np.ndarray          # [5,120,3]
    tactile_aggregate_valid: bool
    tactile_dense_valid: bool

    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray
```

Delete from canonical driver state:

```text
tactile_contact
_TACTILE_CONTACT_THRESHOLD
```

The driver must not convert continuous tactile into binary contact.

### 7.1 `get_state()` flow

Target flow:

```text
read_state
    ↓
validate raw_state / usable read code
    ↓
parse joints
    ↓
(aggregate_allowed, dense_allowed) = _tactile_validity(...)
    ↓
parse aggregate independently if allowed
    ↓
parse dense independently if allowed
    ↓
subtract software bias when available
    ↓
return XHandState
```

Driver parsing should stay single-shot: no runtime retry/backoff/recovery mechanism is added.

---

## 8. IPC and hand worker: one hand observation ring

Primary files:

```text
dexmani_real/ipc/schema.py
dexmani_real/ipc/channels.py
dexmani_real/robot/hand_worker.py
```

### 8.1 Merge dense tactile into `HAND_STATE_DTYPE`

Target hand state wire record:

```text
qpos                         float64 [12]
current                      float64 [12]

tactile_aggregate            float32 [5,3]
tactile_dense                float32 [5,120,3]
tactile_aggregate_valid      uint8
tactile_dense_valid          uint8

connected                    uint8
qpos_stale                   uint8
accepted_target_action_id    uint64
accepted_target_monotonic_ns uint64
last_sdk_setpoint_accepted_monotonic_ns uint64
commboard_err                int32 [12]
jointboard_err               int32 [12]
tipboard_err                 int32 [12]
source_monotonic_ns          uint64
publish_monotonic_ns         uint64
state_valid                  uint8
```

Keep qpos/current precision unchanged in this task. Tactile payloads should use `float32` because the processed/model boundary already uses float32 and the sensor does not benefit from double-precision transport/storage.

### 8.2 Delete the second tactile ring

Delete:

```text
HAND_TACTILE_DTYPE
RuntimeChannels.hand_tactile_ring
hand_tactile_ring_maxlen
_build_tactile_frame()
```

One successful XHand read produces one `HAND_STATE_DTYPE` publication.

This restores the natural invariant:

```text
qpos source
== current source
== aggregate source
== dense source
```

No source-identity matching is required downstream.

### 8.3 Read failure behavior

Keep the existing worker health semantics rather than redesigning worker scheduling:

```text
single read failure
    -> publish stale qpos/current with state_valid=false / qpos_stale=true
    -> aggregate_valid=false
    -> dense_valid=false
    -> keep failure-duration tracking

persistent failure timeout
    -> existing worker fault path
```

The ring may zero-fill invalid tactile for fixed-size realtime transport. Dataset recording must never treat those zeros as valid no-contact data.

### 8.4 Expected payload cost

Dense tactile is:

```text
5 * 120 * 3 * 4 bytes = 7.2 KB per hand sample
```

At a 30 Hz hand loop, the raw dense shared-memory write rate is roughly 216 KB/s, which is small for local shared memory.

Do not keep a second ring merely to optimize this hypothetical cost. After implementation, run an offline synthetic ring benchmark / runtime timing inspection. Only reintroduce an optimization if a measured regression exists.

---

## 9. Causal selection: select one hand sample, never repair tactile

Primary files:

```text
dexmani_real/ipc/causal.py
dexmani_real/teleop/episode_samples.py
dexmani_real/deployment/inference/observation.py
```

The control-step observation already has a causal anchor `T[t]`.

Select:

```text
latest hand state with
0 < source_monotonic_ns <= publish_monotonic_ns <= anchor_monotonic_ns
```

Then use tactile from that exact hand sample.

### 9.1 Delete tactile-specific backward search

Delete the canonical use of:

```text
read_hand_contact_causal()
read_hand_tactile_causal()
_dense_tactile_fresh()
```

Do not produce:

```text
qpos_t + aggregate_{t-1}
qpos_t + dense_{t-2}
```

merely to maximize complete rows.

If the selected hand sample says:

```text
aggregate_valid=false
```

then the control-step aggregate is invalid.

If it says:

```text
dense_valid=false
```

then the control-step dense tactile is invalid.

This makes sensor dropout directly observable instead of silently repairing it with older values.

### 9.2 Freshness is separate from measurement validity

Do not fold age into the two tactile validity bits.

The dataset already owns:

```text
observation_anchor_monotonic_ns
hand_source_monotonic_ns
```

Therefore tactile/hand age is always reconstructible:

```text
age_ns = observation_anchor_monotonic_ns - hand_source_monotonic_ns
```

No persisted `tactile_fresh` flag is needed.

Runtime deployment may still impose its own age/skew gates at the observation boundary. Those are temporal admission rules, not sensor measurement validity.

---

## 10. Recording boundary

Primary files:

```text
dexmani_real/recording/sample.py
dexmani_real/recording/frame.py
dexmani_real/teleop/episode_samples.py
```

### 10.1 `EpisodeState`

Replace separate hand-contact/tactile arguments with one hand-state extraction.

Conceptually:

```python
EpisodeState(
    ...,
    hand_qpos=...,
    hand_current=...,
    hand_contact=...,                  # aggregate payload, existing persisted name
    hand_contact_valid=...,
    hand_tactile_force=...,            # dense payload, existing persisted name
    hand_tactile_force_valid=...,
    hand_source_monotonic_ns=...,
)
```

`record_frame()` / `record_held()` no longer accept a separate `hand_tactile` frame.

### 10.2 Invalid payload representation

At the recording boundary enforce:

```text
valid=true
    -> payload must be finite

valid=false
    -> persisted payload is NaN
```

Examples:

```text
valid no-contact:
    payload approximately zero
    valid=true

invalid tactile:
    payload=NaN
    valid=false
```

This prevents an invalid worker zero-fill from being mistaken for a real no-contact measurement.

Aggregate and dense invalidity remain independent.

### 10.3 Remove tactile-specific recording provenance

Delete from the new raw schema:

```text
hand_contact_source_monotonic_ns
tactile_source_monotonic_ns
tactile_sum_fresh
tactile_fresh
tactile_calibrated
tactile_unit_code
```

Use the single:

```text
hand_source_monotonic_ns
```

for qpos/current/aggregate/dense from the selected hand sample.

---

## 11. Raw vNext schema

If executed immediately, bump raw v28 -> v29.

### 11.1 Tactile-related target fields

Keep the existing persisted payload vocabulary to avoid gratuitous downstream rename churn, but add direct validity and remove derived provenance:

| field | shape | dtype | meaning |
|---|---:|---|---|
| `hand_qpos` | `(N,12)` | float64 | selected causal hand qpos |
| `hand_current` | `(N,12)` | float64 | same XHand sample current |
| `hand_contact` | `(N,5,3)` | **float32** | bias-corrected XHand `calc_force` |
| `hand_contact_valid` | `(N,)` | bool | aggregate payload usable |
| `hand_tactile_force` | `(N,5,120,3)` | **float32** | bias-corrected XHand `raw_force` |
| `hand_tactile_force_valid` | `(N,)` | bool | dense payload usable |
| `hand_source_monotonic_ns` | `(N,)` | uint64 | host-monotonic source/availability proxy for the whole hand sample |
| `observation_anchor_monotonic_ns` | `(N,)` | uint64 | control-step causal cut |

### 11.2 Remove

```text
hand_contact_source_monotonic_ns
tactile_source_monotonic_ns
tactile_sum_fresh
tactile_fresh
tactile_calibrated
tactile_unit_code
```

The raw schema should not persist a binary `tactile_contact` signal.

### 11.3 Raw validation

Raw storage validation should prove only structural invariants:

```text
shape / dtype
row count
valid row -> finite payload
invalid row -> NaN payload
0 < hand_source_monotonic_ns <= observation_anchor_monotonic_ns for normal recorded rows
```

Do not infer validity from timestamps or payload magnitude.

---

## 12. Processed vNext

If executed immediately, bump processed v19 -> v20.

Keep the current external payload names unless a separate cross-repository rename is deliberately scheduled:

```text
contact_force       [N,5,3]      float32
contact_force_valid [N]          bool

tactile_force       [N,5,120,3]  float32
tactile_force_valid [N]          bool
```

### 12.1 Processing is intentionally dumb

Canonical transform:

```python
contact_force = raw.hand_contact
contact_force_valid = raw.hand_contact_valid

tactile_force = raw.hand_tactile_force
tactile_force_valid = raw.hand_tactile_force_valid
```

Delete validity reconstruction such as:

```text
finite
AND calibrated
AND unit_code == native
AND source timestamp > 0
AND legacy freshness
```

Processing may assert the raw invariant, but it must not make a new validity decision.

### 12.2 Remove processed per-row tactile telemetry

Delete from the canonical processed artifact:

```text
contact_force_fresh
tactile_force_fresh
tactile_calibrated
tactile_unit_code
contact_source_monotonic_ns
tactile_source_monotonic_ns
```

Keep:

```text
observation_anchor_monotonic_ns
hand_source_monotonic_ns
```

because they are cheap raw facts and allow any future temporal-age analysis without freezing a freshness threshold into the dataset.

### 12.3 Static semantic metadata

Static representation metadata may remain at the processed/Zarr root because it is cheap and useful for interpretation:

```text
contact_force_representation = xhand_sdk_calc_force_fx_fy_fz_bias_corrected
tactile_force_representation = xhand_sdk_raw_force_fx_fy_fz_bias_corrected
finger order
sensor order
raw_force point order
axis labels
```

Current unit/SI attrs may be retained as **static compatibility/self-description metadata** during this Real-only refactor, but they must not participate in row validity or runtime tactile health.

Do not spend this task proving or changing Newton/SI semantics.

---

## 13. Policy Zarr mirror

`dexmani_real/dataset/export.py` should continue to be a mechanical processed -> Zarr mirror.

If the processed key set changes, bump the Real Policy Zarr schema once (v12 -> v13 if executed now) and export the new superset exactly.

No additional tactile repair, validity reconstruction, or model-specific pruning belongs in the exporter.

This task does not change how `dexmani_policy` encodes tactile. A separate follow-up owns model consumers and any eventual cross-repository naming cleanup.

---

## 14. Real deployment observation path

Even though policy-side tactile encoding is out of scope, `dexmani_real` must remain internally coherent after removing `hand_tactile_ring`.

Update `deployment/inference/observation.py` mechanically:

```text
joint_state
    -> read hand qpos from hand_state_ring

contact_force
    -> read tactile_aggregate from the same aligned hand_state history
    -> require tactile_aggregate_valid

tactile_force
    -> read tactile_dense from the same aligned hand_state history
    -> require tactile_dense_valid
```

Delete:

```text
hand_tactile_provenance_history
_read_tactile_provenance_history()
_read_tactile_force_history() over a separate ring
aggregate/provenance source timestamp equality check
calibrated/unit_code runtime tactile gates
```

Temporal deployment gates remain where they already belong:

```text
run-start
causal source/publish ordering
max input age
control-grid lag
cross-modality skew
```

These are observation-timing constraints, not tactile measurement validity.

Contact-only deployment must not require dense-valid tactile.

---

## 15. Legacy data strategy

Do not build a compatibility framework.

### 15.1 Existing v26 / v28 data

Existing raw data was recorded under different semantics:

```text
aggregate and dense may have different selected source timestamps
fresh/calibration/unit fields exist
historical v26 contains legacy conversion semantics
```

Therefore do **not** convert old raw data into vNext and claim atomic hand-sample semantics.

That would be a false semantic migration.

### 15.2 Freeze old artifacts

Recommended policy:

```text
existing validated processed/Zarr artifacts
    -> keep frozen for existing experiments

new recordings after the change
    -> use only vNext
```

Do not add:

```text
v28/v29 runtime compatibility reader
a schema migration graph
deprecated tactile wrappers
dual hand ring implementations
```

Existing one-shot historical conversion tools and forensic documents may remain as frozen evidence; they must not be imported by the new runtime path.

If old raw must be reprocessed in the future, use the historical repository revision that owns its semantics rather than complicating the current research pipeline.

---

## 16. File-level implementation plan

### Phase A — Driver semantics

`dexmani_real/robot/drivers/xhand.py`

- keep `_tactile_validity()` as one pure helper;
- rename runtime values toward aggregate/dense terminology;
- keep direct SDK-native parsing and software bias;
- delete `tactile_contact` and contact threshold;
- keep aggregate/dense parse failures independent;
- keep calibration/zeroing private to the driver.

Focused tests:

```text
0 -> aggregate=true dense=true
1501018 -> false,false
1501019 -> true,false
1501020 -> true,true
CRC/other -> false,false
parse/nonfinite aggregate invalidates aggregate only
parse/nonfinite dense invalidates dense only
```

### Phase B — One hand IPC record

`dexmani_real/ipc/schema.py`
`dexmani_real/ipc/channels.py`
`dexmani_real/robot/hand_worker.py`

- add aggregate/dense float32 payloads and two valid bits to `HAND_STATE_DTYPE`;
- delete `HAND_TACTILE_DTYPE` and `hand_tactile_ring`;
- one worker read -> one ring write;
- combine per-read validity with session tactile-ready at the worker boundary;
- preserve all command/safety fields unchanged.

Focused tests:

```text
one publication carries qpos + aggregate + dense with one source timestamp
read failure => state invalid + both tactile invalid
1501019 => state valid + aggregate valid + dense invalid
calibration failure => joints continue + tactile invalid
```

### Phase C — Causal/recording simplification

`dexmani_real/ipc/causal.py`
`dexmani_real/teleop/episode_samples.py`
`dexmani_real/recording/sample.py`
`dexmani_real/recording/frame.py`

- use one selected causal hand state;
- remove tactile backward search and separate tactile frame argument;
- map invalid tactile to NaN at recording boundary;
- remove tactile fresh/calibration/unit signal construction.

Focused tests:

```text
latest hand sample invalid tactile does NOT fall back to previous tactile
valid zero remains finite zero + valid=true
invalid payload becomes NaN + valid=false
aggregate/dense partial invalidity is preserved independently
```

### Phase D — Raw schema vNext

`dexmani_real/recording/storage/schema.py`
reader/writer/tests

- bump raw schema once;
- add direct aggregate/dense validity fields;
- change tactile payload storage to float32;
- remove tactile-specific derived telemetry/timestamps;
- no compatibility branch in the current writer/reader.

### Phase E — Processed vNext

`dexmani_real/dataset/processing.py`
`dexmani_real/dataset/processed.py`

- bump processed schema once;
- copy tactile validity directly from raw;
- remove legacy v26/current-v28 tactile validity branches from the canonical vNext path;
- remove per-row tactile fresh/calibrated/unit/source datasets;
- keep static representation metadata;
- validator checks mask/payload consistency, not validity policy.

### Phase F — Real export and deployment plumbing

`dexmani_real/dataset/export.py`
`dexmani_real/deployment/inference/observation.py`
`dexmani_real/deployment/lifecycle.py`
related tests

- mechanically mirror processed vNext to Zarr;
- remove hand-tactile-ring capacity/configuration;
- read aggregate/dense from aligned hand-state history;
- keep temporal/freshness admission at deployment observation boundary;
- do not change model-side tactile consumption.

### Phase G — Documentation cleanup

After implementation is green:

- update `docs/data_schema.md` to the new canonical schema;
- update `repo_map.md` tactile topology statements;
- update README only if user-facing recording/export workflows change;
- mark obsolete tactile design documents historical rather than editing them into fake current-state guides.

---

## 17. Delete list

After source search confirms no independent consumer, delete the following mechanisms rather than keeping wrappers:

```text
XHandState.tactile_contact
_TACTILE_CONTACT_THRESHOLD
HAND_TACTILE_DTYPE
RuntimeChannels.hand_tactile_ring
hand_tactile_ring_maxlen
_build_tactile_frame()
read_hand_contact_causal()
read_hand_tactile_causal()
_dense_tactile_fresh()
tactile_sum_fresh persisted field
tactile_fresh persisted field
tactile_calibrated persisted field
tactile_unit_code persisted field
hand_contact_source_monotonic_ns persisted field
tactile_source_monotonic_ns persisted field
processed contact_force_fresh
processed tactile_force_fresh
processed tactile_calibrated
processed tactile_unit_code
processed contact_source_monotonic_ns
processed tactile_source_monotonic_ns
```

Delete tests whose only purpose is to prove those removed mechanisms. Replace them with smaller tests for the new invariants; do not recreate removed behavior through compatibility helpers.

---

## 18. Protected invariants

This tactile simplification must not change:

```text
arm/hand command safety
mechanical joint limits
command finite checks
per-tick slew/jump limits
CRC command-delivery uncertainty handling
state-read failure timeout
heartbeat and readiness
estop behavior
SafetyState / run_generation / ticket semantics
collision/workspace checks
verified shutdown
exact sent-action recording
control-step obs-before-action ordering
camera/arm causality rules
```

Do not loosen a safety or freshness check merely because tactile transport becomes simpler.

---

## 19. Verification plan

No hardware-affecting validation is allowed unless explicitly requested separately.

### 19.1 Static/offline checks

At minimum:

```bash
python -m compileall -q dexmani_real examples
pytest -q tests/test_xhand_driver_tactile.py
pytest -q tests/test_recording_control_contact.py
pytest -q tests/test_control_step_dataset.py
pytest -q tests/test_observation_builder.py
pytest -q tests/test_deployment_eef_tactile.py
git diff --check
git diff --stat
```

Adjust test filenames when removed mechanisms make old tests obsolete; the replacement tests must cover equivalent safety/correctness boundaries without preserving obsolete architecture.

### 19.2 Synthetic IPC performance check

Use a pure shared-memory benchmark with the new `HAND_STATE_DTYPE` to measure:

```text
write throughput
read throughput
copy latency
ring capacity memory
```

No hardware SDK import or device connection is needed.

Acceptance criterion is not an arbitrary microbenchmark number. The purpose is to verify that merging the 7.2 KB dense payload does not produce a meaningful regression relative to the hand loop/control rates. If it does, optimize based on measured evidence rather than restoring the full old provenance system.

### 19.3 Hardware follow-up — NOT part of this implementation task

When the code is later exercised on real XHand hardware, record:

```text
effective hand read Hz
read latency/jitter
SDK status histogram
tactile aggregate/dense invalid rate
frequency of RS485 1501019 partial failures
no-contact residual after zeroing
```

Hardware validation must be reported explicitly; do not infer it from unit tests.

---

## 20. Acceptance criteria

The refactor is complete only when all of the following are true.

### Runtime

```text
one XHand read -> one hand_state_ring record
no separate tactile ring
aggregate/dense share one hand source timestamp
exactly two tactile validity bits
```

### Validity

```text
validity = SDK availability + parse/shape/finite + session bias readiness
no freshness/unit/contact/quality gate in tactile validity
```

### Recording

```text
one selected causal hand sample per control row
no backward tactile repair
invalid tactile -> NaN + valid=false
valid zero tactile remains distinguishable from invalid
```

### Raw

```text
direct aggregate/dense validity persisted
no per-row tactile fresh/calibrated/unit fields
no independent aggregate/dense timestamps
```

### Processing

```text
validity copied, not reconstructed
no legacy tactile validity policy in canonical vNext processing
```

### Complexity

The implementation must not introduce:

```text
validity enum
reason bitmask
tactile health object
sensor manager
status registry
schema compatibility framework
repair/fill logic
interpolation
```

### Safety

All existing robot command and lifecycle safety invariants remain unchanged.

---

## 21. Final architecture

The intended final tactile path should be readable in one pass:

```text
XHand SDK read_state
    ↓
parse qpos/current
parse calc_force/raw_force independently
    ↓
software bias subtraction
    ↓
aggregate_valid / dense_valid
    ↓
ONE hand_state_ring record
    ↓
latest causal hand state for control step
    ↓
RAW
    hand_contact + hand_contact_valid
    hand_tactile_force + hand_tactile_force_valid
    hand_source_monotonic_ns
    observation_anchor_monotonic_ns
    ↓
PROCESSED
    contact_force + contact_force_valid
    tactile_force + tactile_force_valid
    hand_source_monotonic_ns
    observation_anchor_monotonic_ns
    ↓
Zarr mechanical mirror
```

The design intentionally stops there.

How aggregate or dense tactile is normalized, encoded, fused, or ablated belongs to policy research, not the Real acquisition pipeline.

The Real layer should preserve one simple scientific fact:

> **At each control step, this is the latest causal XHand hand sample, and these two booleans say whether its aggregate and dense tactile payloads are usable.**
