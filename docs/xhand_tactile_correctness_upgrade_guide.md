# XHand tactile correctness upgrade guide

> Implementation guide for Codex / Claude Code.
>
> Baseline verified against `main` at commit `bc14cb00bd67c7d9e7c4bcef373e3c5ed5fa7a16` (2026-09-09).
> Source code is authoritative if this document becomes stale.

## 1. Goal and scope

This task fixes the XHand tactile correctness path while keeping the system small and research-oriented.

The target architecture is:

```text
XHand SDK read_state
        |
        +------------------------------+
        |                              |
  calc_force [5,3]               raw_force [5,120,3]
        |                              |
        | independent validity         | independent validity
        |                              |
        +-------- software no-contact bias --------+
                               |
                    SDK-native numeric scale
                               |
                   raw episode stores BOTH
                               |
                  processed artifact stores BOTH
                               |
              Policy decides what to consume later
```

This guide focuses only on the five confirmed/high-priority problems:

1. `calc_force` and dense `raw_force` validity are incorrectly collapsed into one boolean.
2. A contact-only policy is unnecessarily coupled to dense-tactile freshness.
3. `calibrated=True` currently means only “bias arrays exist”; there is no post-bias verification.
4. The pre-calibration load check uses dense raw taxels instead of checking both aggregate and dense signals.
5. The driver incorrectly multiplies both tactile representations by `0.1`; canonical Real data should use the XHand SDK-native numeric scale.

The following are deliberately **out of scope** for this task:

- choosing whether `dexmani_policy` should consume contact bits, magnitude, 15-D XYZ, or dense tactile;
- tactile CNN / Transformer / taxel-token architecture;
- proving the 120-taxel spatial geometry (`10x12`, flips, rotations, etc.);
- a high-rate tactile sidecar stream;
- changing causal alignment to interpolation;
- enabling vendor `reset_sensor()` by default;
- claiming Newton/SI units before physical calibration;
- fixing host read-completion timestamp vs true sensor-acquisition timestamp.

The Real repository must preserve both `calc_force [5,3]` and `raw_force [5,120,3]` so later policy ablations do not require recollecting data.

---

## 2. Fact-check: current repository behavior

### 2.1 Current scale is wrong for the intended canonical representation

Current `dexmani_real/robot/drivers/xhand.py` defines:

```python
_TACTILE_SCALE = 0.1
_TACTILE_CONTACT_THRESHOLD = 1.0
_RAW_FORCE_CONTACT_THRESHOLD = 1.0
```

and `_parse_tactile()` applies:

```python
tactile_force *= _TACTILE_SCALE
force_sum *= _TACTILE_SCALE
```

before subtracting software biases.

This means current canonical values are:

```text
stored = 0.1 * SDK_value - bias_in_scaled_space
```

The repository previously removed the same `0.1` scale at commit
`aa084a9b3a64fe38970154ea4595a6c3c56cbb67`, explicitly because it had no reference-project justification; the following tactile implementation at commit
`16d3121dc5fd0d9d7ebe233ee03f75b5e6084fcc` returned SDK values without scaling.
The `0.1` convention was later reintroduced into the current driver lineage.

### 2.2 PI-R2 reference behavior

Reference:

```text
pi-r2-flow/pi-r2-flow
  deployment/mindex/robots/xhand_robot.py
```

PI-R2 parses:

```text
sensor_data[k].calc_force   -> [5,3]
sensor_data[k].raw_force    -> [5,120,3]
```

and does **not** multiply either representation by `0.1` (or any other sensor scale).
Its canonical form is effectively:

```text
SDK numeric value - software bias
```

PI-R2 also performs a post-bias residual check after its reset/bias sequence.
Its code labels aggregate force as Newtons, but this repository must **not** copy that SI claim because we do not have independent calibration evidence for this installation.

### 2.3 DexUMI reference behavior

References:

```text
real-stanford/DexUMI
  dexumi/hand_sdk/xhand/hand_api_cls.py
  real_script/eval_policy/eval_xhand.py
```

DexUMI's native XHand deployment path also forwards SDK `calc_force` / `raw_force` values without `0.1` scaling.
For policy input it computes aggregate magnitude and uses an empirical threshold of `10`:

```text
contact = ||calc_force_xyz|| >= 10
```

Therefore the current DexMani Real pair:

```text
scale = 0.1
threshold = 1
```

is numerically equivalent **only for binary thresholding**:

```text
0.1 * F > 1  <=>  F > 10
```

It does **not** justify rescaling the continuous sensor representation.

### 2.4 Current partial-status handling is inconsistent with the repository's own hardware diagnostic

Current runtime `get_state()` does:

```python
tactile_valid = code == 0
...
tactile_sum_valid = tactile_valid
tactile_valid = tactile_valid
```

so error codes `1501018`, `1501019`, and `1501020` invalidate both aggregate and dense force.

But `examples/xhand_control_example.py` already encodes the more precise, RS485-verified semantics:

| SDK code | Meaning | `calc_force` | `raw_force` |
|---|---|---:|---:|
| `0` | OK | valid | valid |
| `1501018` | combined force unavailable | invalid | invalid (conservative) |
| `1501019` | distributed force unavailable | valid | invalid |
| `1501020` | temperature unavailable | valid | valid |
| `1501070` | CRC uncertainty | invalid | invalid |

The production driver should match this table for **serial / RS485**. Do not assume the same partial-status semantics for EtherCAT unless independently verified; for EtherCAT, keep tactile fail-closed on nonzero read status while still preserving usable joint feedback where the existing runtime allows it.

### 2.5 Current contact-only deployment still depends on dense tactile

The design already has separate streams:

```text
hand_state_ring:
    tactile_sum [5,3]
    tactile_sum_valid

hand_tactile_ring:
    tactile_force [5,120,3]
    fresh
    calibrated
    unit_code
```

However `_build_tactile_frame()` currently writes:

```python
fresh = dense_valid
calibrated = dense_valid and calibration_state
```

and contact-only deployment reads `fresh/calibrated/unit_code` from `hand_tactile_ring` as provenance.
As a result, a `1501019` distributed-force failure prevents a contact-only policy from using otherwise-valid `calc_force`.

### 2.6 Current calibration is not verified

Current calibration:

```text
clear biases
-> require full tactile at startup
-> capture 5 no-contact samples
-> mean into aggregate/raw bias
-> bias arrays exist == calibrated
```

There is no independent fresh read after the bias is applied.

### 2.7 Current raw recording has only dense tactile freshness

Raw v25 stores both:

```text
hand_contact        [N,5,3]       # actually SDK calc_force
hand_tactile_force  [N,5,120,3]   # SDK raw_force
```

but has only one `tactile_fresh` field, sourced from `hand_tactile_ring.fresh`, i.e. dense validity.
Therefore it cannot represent:

```text
calc_force valid + raw_force invalid
```

without losing provenance.

---

## 3. Target invariants

Implementations must preserve all of these invariants.

### I1. Canonical Real tactile values use SDK-native numeric scale

After this change:

```text
aggregate = SDK calc_force - software aggregate bias
dense     = SDK raw_force  - software dense bias
```

There is no arbitrary `0.1` sensor scaling in the driver.

Use the unit identity:

```text
xhand_sdk_native_unknown_si
```

until physical calibration proves an SI conversion.

### I2. Aggregate and dense tactile remain separate canonical observations

Keep both:

```text
calc_force  [5,3]
raw_force   [5,120,3]
```

Do not derive one from the other and do not assume:

```text
calc_force == raw_force.sum(axis=taxel)
```

### I3. Aggregate and dense validity are independent

The runtime must be able to represent:

```text
tactile_sum_valid=True
tactile_valid=False
```

for RS485 `1501019`.

### I4. Calibration is one shared state, but sample validity remains modality-specific

The software calibration estimates both biases together. `calibrated=True` means the calibration procedure completed and passed verification.

It does **not** mean a particular dense frame is valid.

Therefore:

```text
calibrated   = calibration state
fresh        = this dense sample is valid
sum_fresh    = this aggregate sample is valid
```

### I5. Contact-only deployment must not require dense tactile freshness

If aggregate force is valid, calibrated, causal, and within skew/age constraints, a policy that requests only `contact_force` may run even when dense tactile is unavailable.

A policy requesting `tactile_force` must still fail closed when dense force is unavailable.

### I6. Raw recording preserves enough information for either future policy choice

Raw data must persist both payloads and their modality-specific validity.

### I7. Existing causal alignment stays unchanged

Do not replace the current newest-source-before-reference selection with linear interpolation.

### I8. Persisted semantic changes require version changes

Do not silently reinterpret existing raw v25, processed v14, or Policy Zarr v7 numeric values.

---

## 4. Implementation design

## 4.1 Phase A — fix the XHand driver first

Primary file:

```text
dexmani_real/robot/drivers/xhand.py
```

### A1. Remove `_TACTILE_SCALE`

Delete:

```python
_TACTILE_SCALE = 0.1
```

and delete both multiplications in tactile parsing.

Do **not** replace it with `_TACTILE_SCALE = 1.0`; an identity scale is unnecessary state and invites future confusion.

### A2. Restore thresholds to the native SDK numeric scale

Use:

```python
_TACTILE_CONTACT_THRESHOLD = 10.0
_RAW_FORCE_CONTACT_THRESHOLD = 10.0
```

These are explicitly **SDK numeric units**, not Newtons.

This preserves the previous effective binary boundary:

```text
old: 0.1 * F > 1
new:       F > 10
```

Do not introduce a new physical threshold in this task.

### A3. Add one small modality-validity helper

Use one pure helper owned by the driver, conceptually:

```python
def _tactile_validity(code: int | None, *, comm_type: str) -> tuple[bool, bool]:
    """Return (calc_force_valid, raw_force_valid)."""
```

Required behavior:

```text
serial / RS485:
  0       -> (True,  True)
  1501018 -> (False, False)
  1501019 -> (True,  False)
  1501020 -> (True,  True)
  1501070 -> (False, False)

EtherCAT:
  0       -> (True, True)
  nonzero -> (False, False) for tactile
```

Do not build a generic status-policy class or registry.

### A4. Split aggregate and dense parsing

Current `_parse_tactile()` parses both together, so it cannot retain aggregate force when dense force is unavailable.

Replace it with two narrow helpers:

```text
_parse_tactile_sum(state)   -> [5,3]
_parse_tactile_force(state) -> [5,120,3]
```

Both should reuse `_sensor_data(state)` and `_force_xyz(...)`.

Do not duplicate sensor-list validation.

### A5. Parse each modality independently in `get_state()`

Pseudo-flow:

```text
read_state
  |
  +-> parse joints (existing behavior)
  |
  +-> (sum_allowed, dense_allowed) = _tactile_validity(...)
          |
          +-> parse sum independently; parser failure => sum_valid=False
          |
          +-> parse dense independently; parser failure => dense_valid=False
```

The resulting state must be able to contain:

```text
sum payload valid + dense payload zero-filled invalid
```

Compute `tactile_contact` from aggregate force whenever `tactile_sum_valid` is true, not when dense force is valid.

Do not weaken joint-state handling for partial tactile errors.

---

## 4.2 Phase B — make calibration mean “verified calibration”

Keep the mechanism software-only for this task. PI-R2's vendor `reset_sensor()` path is useful reference evidence but should remain a later A/B experiment rather than being introduced automatically now.

### B1. Keep one calibration for both representations

Calibration should still estimate:

```text
aggregate bias [5,3]
dense bias     [5,120,3]
```

from the same five fresh no-contact reads.

### B2. Load detection checks aggregate OR dense

Current code checks dense taxels whenever dense is available and ignores aggregate force.

Replace with:

```text
aggregate_load = any(||calc_force[finger]|| > 10)  if sum valid
dense_load     = any(||raw_force[taxel]|| > 10)    if dense valid
load_present   = aggregate_load OR dense_load
```

The value `10` preserves the previous effective numerical boundary after removing `0.1`.
It is a conservative startup heuristic in unknown SDK units, not a physical contact calibration.

The calibration entry point should require both aggregate and dense payloads to be valid because this repository intentionally calibrates and records both canonical representations.

### B3. `_capture_tactile_bias()` should return candidate biases

Prefer:

```python
bias_sum, bias_raw = _capture_tactile_bias()
```

instead of publishing member state inside the helper.
This makes the lifecycle explicit:

```text
capture candidate -> publish candidate -> verify -> keep or clear
```

No new class is needed.

### B4. Add post-bias verification

After publishing the candidate biases, perform a small independent verification window:

```text
3 fresh reads, ~20 ms apart
```

Require on every verification read:

```text
tactile_sum_valid == True
tactile_valid == True
```

Compute:

```text
aggregate_peak = max over reads/fingers ||calc_force||
dense_peak     = max over reads/fingers/taxels ||raw_force||
```

Initial acceptance boundary:

```text
aggregate_peak <= 10 SDK-native units
dense_peak     <= 10 SDK-native units
```

This intentionally reuses the already-established effective threshold rather than inventing a new unverified number.
Later hardware characterization may tighten these limits.

If verification fails or raises:

```text
clear both biases
return False / raise the existing XHand calibration error as appropriate
```

Only after verification succeeds may `tactile_calibrated` become externally observable as true.

Do not add retry state machines. One calibration attempt plus a clear failure is sufficient; the operator can correct contact/load and restart.

### B5. Calibration logging

One concise success line is enough. Include aggregate/dense residual peaks if available.
Do not add a health manager or long-running calibration telemetry.

---

## 4.3 Phase C — decouple contact-only policy from dense tactile

Files:

```text
dexmani_real/robot/hand_worker.py
dexmani_real/deployment/inference/observation.py
```

### C1. Fix `HAND_TACTILE_DTYPE.calibrated` semantics at the producer

Current `_build_tactile_frame()` writes:

```text
fresh      = dense_valid
calibrated = dense_valid AND calibration_state
```

Change it to:

```text
fresh      = dense_valid
calibrated = calibration_state
```

`calibrated` is a property of the calibration state; `fresh` is a property of this dense payload.

This is the smallest change that preserves the existing separate ring while allowing its tiny metadata to prove calibration even on a frame where dense data is unavailable.

### C2. Contact-only provenance reader must not gate on dense `fresh`

`_read_tactile_provenance_history()` is used only to avoid copying `[5,120,3]` for contact-only policies.
For this path require:

```text
calibrated == True
unit_code == native-unit code
causal source/publish timestamps
age bound
```

Do **not** require dense `fresh` there.

Aggregate freshness is already independently enforced by the `hand_state_ring` reader through:

```text
state_valid
tactile_sum_valid
qpos_stale == False
```

Keep the final source-timestamp identity check between aggregate history and tactile calibration/unit provenance.

### C3. Dense-policy path remains strict

`_read_tactile_force_history()` must continue to require:

```text
fresh == True
calibrated == True
unit code matches
causal and recent
finite dense payload
```

### C4. Required deployment behavior

| Runtime situation | contact-only policy | dense-tactile policy |
|---|---:|---:|
| code `0`, calibrated | run | run |
| RS485 `1501019`, aggregate valid / dense invalid | **run** | reject |
| RS485 `1501018` | reject | reject |
| CRC | reject | reject |
| uncalibrated | reject | reject |

No padding or zero substitution is allowed for a requested invalid tactile modality.

---

## 4.4 Phase D — make raw storage represent both modalities correctly

Because numeric semantics change and aggregate/dense validity must be distinct, raw v25 must not be silently reused.

### D1. Bump raw schema v25 -> v26

File:

```text
dexmani_real/recording/storage/schema.py
```

Set:

```python
EPISODE_SCHEMA_VERSION = 26
```

Keep both existing payload datasets unchanged in shape and name:

```text
hand_contact        [N,5,3]       float64  # calc_force, historical name retained
hand_tactile_force  [N,5,120,3]   float64  # raw_force
```

Do **not** rename `hand_contact` in this task; a field rename adds no correctness value and would expand the migration surface.

### D2. Add exactly one new row-level validity field

Add:

```text
tactile_sum_fresh  [N] bool
```

Meaning:

```text
hand_contact on this row is a valid aggregate tactile sample from its source read
```

Keep existing:

```text
tactile_fresh [N] bool
```

with the clarified meaning:

```text
hand_tactile_force on this row is a valid dense tactile sample
```

Keep the common fields:

```text
tactile_source_monotonic_ns
tactile_calibrated
tactile_unit_code
```

because aggregate and dense are sampled from the same XHand `read_state` call and share the same calibration state and numeric unit convention.

### D3. Recording provenance

Update `_recording_provenance()` in:

```text
dexmani_real/teleop/episode_samples.py
```

Compute aggregate freshness from `hand_state`:

```text
hand_state.tactile_sum_valid
+ valid/causal hand source timestamp
+ recording tactile age bound
```

Compute dense freshness from `hand_tactile.fresh` as today.

Record both flags independently.

### D4. Recording-start policy

Keep the current recording-start gate strict on **dense** tactile freshness and successful calibration.
The explicit collection intent is to preserve both representations, so starting a new official episode while dense tactile is unavailable should still be rejected.

This is different from learned-policy deployment: a contact-only policy may continue under `1501019`, but a new data-collection episode intended to preserve both modalities should start only when both are healthy.

### D5. Unit semantics

For raw v26 define code `0` as the single current canonical identity:

```text
xhand_sdk_native_unknown_si
```

Do not call it Newtons.

Prefer one named constant at the IPC/storage contract boundary rather than scattering magic `0` checks, for example:

```python
TACTILE_UNIT_CODE_XHAND_SDK_NATIVE = 0
```

Do not introduce an enum hierarchy unless the codebase actually needs multiple units.

---

## 4.5 Phase E — processed and Policy Zarr semantic versioning

Changing the numeric scale propagates to persisted downstream artifacts.
Following `AGENTS.md`, do not silently change their meaning.

### E1. Bump processed v14 -> v15

Files include:

```text
dexmani_real/dataset/processed.py
dexmani_real/dataset/processing.py
docs/data_schema.md
relevant processed tests
```

Keep both:

```text
contact_force [T,5,3]
tactile_force [T,5,120,3]
```

in processed data.

Update semantic identities to make the stored transform explicit:

```text
contact_force representation:
  xhand_sdk_calc_force_fx_fy_fz_bias_corrected

tactile_force representation:
  xhand_sdk_raw_force_fx_fy_fz_bias_corrected

unit:
  xhand_sdk_native_unknown_si

si_verified:
  False

spatial_geometry_verified for dense tactile:
  False
```

Do not add magnitude/contact-bit datasets; those remain downstream derived features.

For now processed v15 may continue to require both tactile payloads for each retained row, preserving the existing same-source paired research artifact. Raw v26 retains independent validity, so a future contact-only processing profile can be added without recollecting data if experiments justify it.

### E2. Keep same-source invariant

Do not change the existing selector invariant:

```text
contact_force[t] and tactile_force[t]
come from the identical selected raw tactile source row
```

Do not introduce separate nearest-neighbor matching for the two payloads.

### E3. Policy Zarr must also get a semantic version bump

Current Zarr v7 contains `contact_force`, so its numeric meaning also changes.
Do not emit native-scale values while still labeling the artifact v7.

Create Policy Zarr v8 with the same current legacy key projection unless another task explicitly changes policy modalities:

```text
joint_state
action
action_ee
contact_force
fingertip_points
+ profile-dependent visual fields
```

Do **not** add dense `tactile_force` to Zarr v8 in this task; the user has intentionally not selected the policy representation yet.

The only reason for the v8 bump is the corrected `contact_force` numeric semantics.

Any `dexmani_policy` loader/checkpoint contract update is a separate cross-repository task. Do not move Real acquisition or processing behavior into `dexmani_policy`.

---

## 5. Historical raw episode repair

Add a dedicated offline conversion tool:

```text
tools/convert_raw_v25_to_v26_tactile.py
```

Model its transactional structure after the existing:

```text
tools/convert_raw_v24_to_v25.py
```

### 5.1 Never mutate the source by default

Use the same source/destination workflow:

```bash
python tools/convert_raw_v25_to_v26_tactile.py SOURCE DESTINATION
```

Support:

```text
one episode -> one destination episode
episode root -> destination root
```

Create a process-owned staging directory, verify the result, then rename atomically.
Preserve/hard-link `depth.h5` and `rgb.mp4` using the same EXDEV fallback as the existing converter.

Do not make in-place mutation the default; interrupted `*=10` writes can leave an episode partially converted.

### 5.2 Exact scale correction for directly-recorded v25 episodes

Direct raw v25 recording was introduced after the current `0.1`-scaled driver behavior was already present.
For a direct v25 episode (`converted_from_schema` absent), convert:

```python
hand_contact[...] *= 10.0
hand_tactile_force[...] *= 10.0
```

This is mathematically correct even with software bias:

```text
old bias = mean(0.1 * SDK_zero)
old saved = 0.1 * SDK - old bias
          = 0.1 * (SDK - mean(SDK_zero))

new target = SDK - mean(SDK_zero)
           = 10 * old saved
```

For uncalibrated samples the same scale conversion still holds, and invalid zero-filled payloads remain zero.

### 5.3 Populate the new aggregate-validity field

The legacy runtime collapsed aggregate/dense validity, so directly-recorded v25 episodes can be migrated conservatively with:

```text
tactile_sum_fresh = tactile_fresh
```

This does not invent information that v25 did not preserve.

### 5.4 Converted v24 -> v25 episodes require explicit scale classification

The existing `convert_raw_v24_to_v25.py` copies tactile payloads unchanged and writes:

```text
converted_from_schema = 24
```

Historical raw v24 episodes span different code generations, so the v26 converter must **not** blindly multiply every converted v25 episode by 10.

Default behavior for `converted_from_schema == 24`:

```text
REFUSE with a clear message requiring explicit source-scale selection
```

Provide one explicit option, for example:

```text
--converted-v24-scale legacy-0.1
--converted-v24-scale native
```

Behavior:

```text
legacy-0.1 -> multiply tactile payloads by 10
native     -> copy tactile payloads unchanged
```

In both cases populate `tactile_sum_fresh = tactile_fresh` conservatively because old artifacts do not prove independent aggregate freshness.

Do not guess from payload magnitude.
Do not infer scale from task type or threshold statistics.

### 5.5 Derived artifacts are regenerated, not patched

Do not write migration tools for processed v14 or Policy Zarr v7.
After raw conversion:

```text
raw v26
  -> regenerate processed v15
  -> regenerate Policy Zarr v8 if needed
```

This keeps one source of truth and avoids repairing duplicated provenance/semantic metadata in multiple artifact layers.

---

## 6. Recommended file-level work plan

Implement in this order so each phase has a narrow invariant.

### Phase 1 — driver correctness

Edit:

```text
dexmani_real/robot/drivers/xhand.py
```

Tasks:

1. delete `_TACTILE_SCALE` and both multiplications;
2. move thresholds `1.0 -> 10.0`;
3. add RS485-aware `(sum_valid, dense_valid)` status helper;
4. split aggregate and dense parsing;
5. compute contact from `tactile_sum_valid`;
6. change load detection to aggregate OR dense;
7. make bias capture return candidate arrays;
8. add 3-read post-bias verification;
9. clear candidate biases on verification failure.

Do not touch IPC/schema in this phase.

### Phase 2 — runtime provenance decoupling

Edit:

```text
dexmani_real/robot/hand_worker.py
dexmani_real/deployment/inference/observation.py
```

Tasks:

1. make `calibrated` independent of dense `fresh` in tactile-ring publication;
2. contact-only metadata path ignores dense `fresh`;
3. full-dense path remains unchanged/strict;
4. preserve exact source-timestamp equality check.

### Phase 3 — raw v26 and migration

Edit/add:

```text
dexmani_real/recording/storage/schema.py
dexmani_real/teleop/episode_samples.py
dexmani_real/recording/frame.py     # only if needed by the new field
dexmani_real/recording/storage/*    # only boundary code required by schema v26
tools/convert_raw_v25_to_v26_tactile.py
docs/data_schema.md
```

Tasks:

1. bump raw schema to 26;
2. add `tactile_sum_fresh`;
3. define native unknown unit semantics;
4. implement transactional v25 -> v26 converter;
5. keep both tactile payloads.

### Phase 4 — processed v15 / Zarr v8 semantic propagation

Edit:

```text
dexmani_real/dataset/processed.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/export.py
docs/data_schema.md
relevant tests
```

Tasks:

1. bump processed to v15;
2. bump Policy Zarr to v8;
3. update aggregate/dense representation strings and unit attrs;
4. do not add dense tactile to Zarr yet;
5. retain same-source tactile pairing.

### Phase 5 — documentation cleanup

After implementation, update stable user-facing references (`README.md`, `repo_map.md`) only where schema versions / supported workflow text actually changed.
Do not keep this guide's baseline snapshot as a competing source of truth.

---

## 7. Required tests

Do not rely on hardware for normal CI/unit validation.
Use fake SDK state/error objects for the driver paths.

### 7.1 Driver validity matrix

Add focused tests that prove:

| Comm | code | joints | sum | dense |
|---|---:|---:|---:|---:|
| serial | `0` | valid | valid | valid |
| serial | `1501018` | valid | invalid | invalid |
| serial | `1501019` | valid | valid | invalid |
| serial | `1501020` | valid | valid | valid |
| serial | `1501070` | valid when joint payload is valid | invalid | invalid |
| ethercat | `0` | valid | valid | valid |
| ethercat | nonzero tactile-status code | preserve existing joint policy | invalid | invalid |

Also test parser-level failure independence:

```text
malformed raw_force does not erase a valid calc_force
malformed calc_force does not fabricate aggregate validity
```

### 7.2 Scale test

For a fake SDK payload with known values:

```text
calc_force = [10,20,30]
raw_force taxel = [4,5,6]
```

with zero bias, assert exact native numeric output (no factor `0.1`).

With known candidate bias, assert:

```text
output = SDK - bias
```

### 7.3 Threshold-equivalence test

Prove the behavior that is intentionally preserved:

```text
old scaled threshold: 0.1*F > 1
new native threshold: F > 10
```

for representative values below, at, and above the boundary.

### 7.4 Calibration tests

Using deterministic fake fresh reads, test:

1. no-load capture + low residual -> calibrated;
2. aggregate startup load -> refused;
3. localized dense startup load -> refused;
4. verification aggregate residual > threshold -> biases cleared;
5. verification dense residual > threshold -> biases cleared;
6. incomplete aggregate/dense payload during capture -> failure;
7. incomplete payload during verification -> failure and biases cleared.

### 7.5 Contact-only deployment regression

Extend `tests/test_deployment_eef_tactile.py` with the critical case:

```text
hand_state aggregate:
    tactile_sum_valid=True
    source=t

hand_tactile metadata:
    fresh=False           # dense unavailable
    calibrated=True
    unit=native
    source=t

Policy requests contact_force only
=> observation succeeds
```

Then prove:

```text
same inputs + Policy requests tactile_force
=> observation rejected
```

### 7.6 Raw v26 migration tests

Add a focused converter test using a synthetic episode:

- direct v25: both tactile payloads become exactly `old * 10`;
- all non-tactile datasets remain bit-identical;
- `tactile_sum_fresh == old tactile_fresh`;
- schema becomes 26;
- source episode remains unchanged;
- converted-from-v24 input is refused without explicit scale option;
- `legacy-0.1` option scales by 10;
- `native` option preserves values;
- staging cleanup occurs on failure.

### 7.7 Processed / Zarr tests

Update existing processed and Zarr projection tests to prove:

```text
processed v15 contact_force == selected raw v26 hand_contact
processed v15 tactile_force == selected raw v26 hand_tactile_force
Policy Zarr v8 contact_force == processed v15 contact_force
```

and unit semantics are exactly:

```text
xhand_sdk_native_unknown_si
si_verified=False
```

Do not add an assertion that `contact_force == tactile_force.sum(...)`.

---

## 8. Validation commands

Before editing:

```bash
git status --short
```

At minimum after the corresponding phases:

```bash
python -m compileall -q dexmani_real tools examples
python -m unittest tests.test_deployment_eef_tactile
python -m unittest tests.test_tactile_selector
python -m unittest tests.test_processed_v14   # rename/update as appropriate for v15
python -m unittest tests.test_zarr_v7_projection  # rename/update as appropriate for v8
```

Run the new focused driver/calibration/migration tests explicitly.
Then run the repository's current offline unit suite if practical:

```bash
python -m unittest discover -s tests
```

Finish with:

```bash
git diff --check
git diff --stat
git status --short
```

Do not run XHand hardware-affecting examples automatically.

---

## 9. Manual hardware verification after code review

This is not part of automated Codex/Claude execution unless the user explicitly asks to run hardware.

When performed manually, use a stationary/no-motion tactile read workflow and verify:

1. no-contact values are near zero after calibration;
2. pressing one finger changes the corresponding aggregate force and dense taxels;
3. `1501019` (if reproducible) leaves aggregate force readable while dense force is marked invalid;
4. contact threshold `10` behaves approximately like the previous `scale=0.1, threshold=1` configuration;
5. no code or documentation labels the value as Newtons without calibration evidence.

Known-load SI calibration and taxel geometry characterization are separate experiments.

---

## 10. Explicit non-goals / reject these implementation directions

Codex / Claude Code should reject the following unless a new task explicitly requests them:

- Do not keep `0.1` in the driver as “normalization”. ML normalization belongs downstream.
- Do not reduce canonical Real data to binary contact.
- Do not drop dense tactile because the current policy does not yet consume it.
- Do not make dense tactile mandatory for a contact-only deployment observation.
- Do not linear-interpolate tactile values across time.
- Do not add `reset_sensor()` as a new public runtime API in this task.
- Do not hardcode `10x12` taxel geometry.
- Do not call threshold `10` “10 N”.
- Do not rename `hand_contact` while fixing scale/validity; document its `calc_force [5,3]` semantics instead.
- Do not patch processed/Zarr artifacts in place; regenerate them from corrected raw v26.
- Do not introduce managers, registries, plugin layers, or generic sensor abstractions for one XHand device.

---

## 11. Completion criteria

The task is complete only when all of the following are true:

```text
[ ] driver emits SDK-native bias-corrected calc_force and raw_force
[ ] no _TACTILE_SCALE=0.1 remains in the production tactile path
[ ] native threshold 10 preserves the previous binary decision boundary
[ ] RS485 1501019 yields sum-valid / dense-invalid
[ ] contact-only deployment survives dense-invalid / sum-valid frames
[ ] dense-tactile deployment still rejects dense-invalid frames
[ ] calibration performs independent post-bias verification
[ ] calibration failure clears both biases
[ ] raw v26 stores both payloads + separate aggregate/dense freshness
[ ] direct raw v25 -> v26 conversion scales both payloads by exactly 10
[ ] converted v24-derived episodes require explicit scale classification
[ ] processed v15 preserves both tactile representations in native SDK scale
[ ] Policy Zarr v8 does not silently reuse v7 tactile numeric semantics
[ ] no SI/Newton claim is introduced
[ ] same-source causal pairing remains unchanged
[ ] focused offline tests pass
```

The intended final principle is simple:

```text
Real owns faithful, bias-corrected sensor observations.
Policy owns representation choice and ML normalization.
```
