# XHand tactile correctness upgrade guide

> Implementation guide for Codex / Claude Code.
>
> Baseline verified against `main` at commit `bc14cb00bd67c7d9e7c4bcef373e3c5ed5fa7a16` (2026-09-09).
> Source code is authoritative if this document becomes stale.
>
> **Historical note (2026-09-10):** this guide documents the frozen raw **v25 → v26** tactile
> migration. Its references to raw v26 / processed v15 / Policy Zarr v8 describe that legacy
> contract; the current schemas are raw **v28** / processed **v17** / Policy Zarr **v10** (see
> [`data_schema.md`](data_schema.md) and
> [`control_step_dataset_simplification_plan.md`](control_step_dataset_simplification_plan.md)).
> Do not read this guide as the current schema.

## 1. Goal and scope

This task fixes the XHand tactile correctness path while keeping the mechanism small, explicit, and suitable for a personal research codebase.

Target architecture:

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

The Real layer preserves sensor information. It must not prematurely choose whether a future policy consumes aggregate XYZ, magnitude, binary contact, dense tactile, or a learned tactile embedding.

This guide focuses on five issues:

1. `calc_force` and dense `raw_force` validity are incorrectly collapsed into one boolean.
2. A contact-only policy is unnecessarily coupled to dense-tactile freshness.
3. `calibrated=True` currently means only “bias arrays exist”; there is no independent post-bias verification.
4. The current pre-calibration load heuristic uses uncalibrated force magnitude as if it were a reliable contact detector.
5. The driver incorrectly multiplies both tactile representations by `0.1`; canonical Real data should use the XHand SDK-native numeric scale.

Out of scope:

- choosing the final tactile representation in `dexmani_policy`;
- tactile CNN / Transformer / taxel tokenization;
- proving the 120-taxel spatial geometry (`10x12`, flips, rotations, etc.);
- high-rate tactile sidecar recording;
- changing causal alignment to interpolation;
- claiming Newton/SI units before physical calibration;
- fixing host read-completion timestamp vs true sensor-acquisition timestamp;
- adding a general sensor abstraction/framework.

---

## 2. Fact-check and reference-project conclusions

### 2.1 Current DexMani Real scale

Current `dexmani_real/robot/drivers/xhand.py` defines:

```python
_TACTILE_SCALE = 0.1
_TACTILE_BIAS_SAMPLE_COUNT = 5
_TACTILE_BIAS_SAMPLE_INTERVAL_S = 0.02
_TACTILE_CONTACT_THRESHOLD = 1.0
_RAW_FORCE_CONTACT_THRESHOLD = 1.0
```

and `_parse_tactile()` does:

```python
tactile_force *= _TACTILE_SCALE
force_sum *= _TACTILE_SCALE
```

before subtracting software biases.

Therefore current recorded values are mathematically:

```text
old_value = 0.1 * SDK_value - old_bias
          = 0.1 * (SDK_value - mean(SDK_zero_samples))
```

The `0.1` scale is not supported by the current repository's verified hardware semantics.

The repository previously removed the same scale at commit
`aa084a9b3a64fe38970154ea4595a6c3c56cbb67` because it lacked reference-project justification. The following tactile implementation at commit
`16d3121dc5fd0d9d7ebe233ee03f75b5e6084fcc` returned SDK tactile values without scaling.

### 2.2 PI-R2: primary reference for this change

Reference:

```text
pi-r2-flow/pi-r2-flow
  deployment/mindex/robots/xhand_robot.py
```

PI-R2 parses directly:

```text
sensor_data[k].calc_force -> [5,3]
sensor_data[k].raw_force  -> [5,120,3]
```

and does not multiply either representation by `0.1` or another numeric scale.
Its effective canonical form is:

```text
SDK numeric value - software bias
```

Its tactile zeroing path uses:

```text
vendor reset_sensor()
-> 5 fresh reads
-> software mean bias for calc_force and raw_force
-> fresh post-bias read
-> verify per-finger ||calc_force|| <= 2.0
```

PI-R2 names the verification argument `verify_thresh_n=2.0` and comments it as Newtons. DexMani Real must **not** copy the SI/Newton claim because we do not have independent known-load calibration for this installation.

What we do copy from PI-R2 is the engineering pattern:

```text
native SDK scale
+ software bias
+ small post-bias aggregate residual threshold
```

### 2.3 DexUMI: secondary reference only

DexUMI's native XHand deployment path also forwards SDK `calc_force` / `raw_force` without `0.1` scaling.

Its policy code computes:

```text
magnitude = ||calc_force_xyz||
contact = magnitude >= 10
```

and its configuration documents `fsr_binary_cutoff=[10,10,10]` for XHand.

This is only an empirical policy cutoff. DexUMI provides no evidence that:

```text
10 SDK units == 10 N
```

and no evidence that `10` is a good threshold for DexMani Real's light-contact manipulation.

Therefore:

- use DexUMI as evidence that native SDK scale is viable;
- do **not** copy its contact cutoff as a hardware constant;
- do **not** preserve the old `0.1*F > 1 <=> F > 10` boundary just for compatibility.

Correctness and sensitivity to light contact are more important than preserving an arbitrary historical binary flag.

### 2.4 Partial RS485 tactile status semantics

Current production `get_state()` does:

```python
tactile_valid = code == 0
...
tactile_sum_valid = tactile_valid
tactile_valid = tactile_valid
```

so `1501018`, `1501019`, and `1501020` currently invalidate both aggregate and dense force.

But `examples/xhand_control_example.py` already implements the more precise RS485 semantics:

| SDK code | Meaning | `calc_force` | `raw_force` |
|---|---|---:|---:|
| `0` | OK | valid | valid |
| `1501018` | combined force unavailable | invalid | invalid (conservative) |
| `1501019` | distributed force unavailable | valid | invalid |
| `1501020` | temperature unavailable | valid | valid |
| `1501070` | CRC uncertainty | invalid | invalid |

The production driver should match this table for serial / RS485.

Do not assume identical partial-status semantics for EtherCAT unless independently verified. For EtherCAT, keep tactile fail-closed on a nonzero read status while preserving usable joint feedback according to the existing runtime policy.

### 2.5 Contact-only deployment is still coupled to dense tactile

Current rings already separate values:

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

But `_build_tactile_frame()` currently writes:

```text
fresh      = dense_valid
calibrated = dense_valid AND calibration_state
```

and contact-only deployment reads these metadata from `hand_tactile_ring`.

So RS485 `1501019` still prevents an aggregate-only policy from consuming otherwise-valid `calc_force`.

### 2.6 Current calibration state is weak

Current calibration is:

```text
clear biases
-> check startup force using an uncalibrated magnitude threshold
-> capture 5 samples
-> mean into aggregate/raw bias
-> bias arrays exist == calibrated
```

There is no independent fresh read after applying the candidate bias.

The pre-bias magnitude check is also conceptually weak:

```text
uncalibrated reading = unknown static sensor offset + external load
```

From a single absolute magnitude, software cannot reliably distinguish these two terms. Lowering that pre-bias threshold simply makes false rejection more likely.

The better simple rule is procedural:

```text
operator ensures the hand is free / unloaded at startup
-> capture bias
-> verify post-bias residual with a small threshold
```

### 2.7 Current raw recording cannot represent aggregate-valid / dense-invalid

Raw v25 stores both:

```text
hand_contact        [N,5,3]       # SDK calc_force representation
hand_tactile_force  [N,5,120,3]   # SDK raw_force representation
```

but only one `tactile_fresh`, sourced from dense tactile validity.

It cannot represent:

```text
calc_force valid + raw_force invalid
```

without losing provenance.

---

## 3. Target invariants

### I1. Canonical tactile values use SDK-native numeric scale

After the fix:

```text
aggregate = SDK calc_force - software aggregate bias
dense     = SDK raw_force  - software dense bias
```

No arbitrary sensor scaling is applied in the driver.

Persist unit identity as:

```text
xhand_sdk_native_unknown_si
```

until a known-load experiment proves an SI conversion.

### I2. Store both representations

Keep both:

```text
calc_force [5,3]
raw_force  [5,120,3]
```

Do not derive one from the other and do not assume:

```text
calc_force == raw_force.sum(axis=taxel)
```

### I3. Aggregate and dense validity are independent

The runtime must represent:

```text
tactile_sum_valid=True
tactile_valid=False
```

for RS485 `1501019`.

### I4. Calibration state and frame validity are different concepts

```text
calibrated = startup calibration completed and passed post-bias verification
sum_fresh  = aggregate payload from this source read is valid
fresh      = dense payload from this source read is valid
```

### I5. Contact-only deployment must not require dense freshness

A policy requesting only `contact_force` may run when aggregate force is valid, calibrated, causal, and recent even if dense tactile is unavailable.

A policy requesting `tactile_force` still rejects an invalid dense frame.

### I6. Thresholds are derived heuristics, not physical units

Use two separate constants even if both initially equal `2.0`:

```python
_TACTILE_CONTACT_THRESHOLD = 2.0
_TACTILE_CALIBRATION_RESIDUAL_THRESHOLD = 2.0
```

Both are in **XHand SDK-native unknown units**.

They answer different questions:

```text
contact threshold:
    is aggregate force large enough to expose a convenience contact bit?

calibration residual threshold:
    after bias subtraction, is aggregate no-contact residual still too large?
```

Do not create a dense-taxel contact threshold in this task.

### I7. Existing causal alignment remains unchanged

Do not replace newest-source-before-reference selection with interpolation.

### I8. Persisted semantic changes require version changes

Do not silently reinterpret raw v25, processed v14, or Policy Zarr v7 numeric values.

---

## 4. Implementation design

## 4.1 Phase A — driver scale, validity, and contact signal

Primary file:

```text
dexmani_real/robot/drivers/xhand.py
```

### A1. Remove `_TACTILE_SCALE`

Delete:

```python
_TACTILE_SCALE = 0.1
```

and remove both multiplications in tactile parsing.

Do not replace it with `_TACTILE_SCALE = 1.0`.

### A2. Use a smaller PI-R2-inspired aggregate contact threshold

Use:

```python
_TACTILE_CONTACT_THRESHOLD = 2.0
```

Interpretation:

```text
2.0 XHand SDK-native units
```

not `2 N`.

Rationale:

- PI-R2 uses a small aggregate post-bias boundary of `2.0` on `||calc_force||`;
- DexUMI's `10` is only an empirical binary policy cutoff and is too coarse as the default reference for light dexterous contact;
- the runtime `tactile_contact` flag is derived convenience information, not canonical force data;
- downstream policy experiments remain free to choose a different threshold or ignore the bit entirely.

Do **not** add `_RAW_FORCE_CONTACT_THRESHOLD`.
Dense raw taxels remain continuous sensor data, not a binary contact oracle.

### A3. Add one small modality-validity helper

Use one pure helper, conceptually:

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

Do not introduce a status registry/class hierarchy.

### A4. Split aggregate and dense parsing

Replace the current combined parser with two narrow helpers:

```text
_parse_tactile_sum(state)   -> [5,3]
_parse_tactile_force(state) -> [5,120,3]
```

Both should reuse `_sensor_data(state)` and `_force_xyz(...)`.

### A5. Parse both modalities independently

Pseudo-flow:

```text
read_state
  |
  +-> parse joints
  |
  +-> (sum_allowed, dense_allowed) = _tactile_validity(...)
          |
          +-> parse aggregate independently
          |
          +-> parse dense independently
```

A malformed dense payload must not erase a valid aggregate payload.

Compute `tactile_contact` whenever `tactile_sum_valid=True`:

```python
np.linalg.norm(tactile_sum, axis=1) > _TACTILE_CONTACT_THRESHOLD
```

---

## 4.2 Phase B — simplify and strengthen calibration

Primary file:

```text
dexmani_real/robot/drivers/xhand.py
```

The calibration mechanism should mirror the useful PI-R2 structure without importing unnecessary complexity.

### B1. Explicit no-contact startup precondition

Calibration is only valid when the operator starts the system with the fingertips unloaded.

Document/log this requirement clearly.

Do **not** attempt to prove no-contact using a small threshold on **uncalibrated** absolute force. Before bias estimation:

```text
measured = unknown offset + external force
```

so a low numeric cutoff cannot reliably distinguish sensor offset from real contact.

Therefore remove `_tactile_load_present()` as a hard pre-bias admission gate rather than replacing `10` with another arbitrary small value.

The startup calibration still requires:

```text
aggregate payload valid
dense payload valid
all values finite
```

because this repository deliberately calibrates and records both representations.

### B2. Capture one shared five-read bias window

Keep the existing compact sampling pattern:

```text
5 fresh reads
~20 ms spacing
```

Estimate:

```text
bias_sum [5,3]
bias_raw [5,120,3]
```

from the same reads.

Prefer:

```python
bias_sum, bias_raw = _capture_tactile_bias()
```

so candidate biases are not published as successful calibration inside the capture helper.

### B3. Publish candidate biases, then verify independently

After candidate biases are assigned, collect:

```text
3 fresh verification reads
~20 ms spacing
```

Every verification read must have:

```text
tactile_sum_valid == True
tactile_valid == True
finite aggregate payload
finite dense payload
```

### B4. Verify aggregate residual with a small independent threshold

Define:

```python
_TACTILE_CALIBRATION_RESIDUAL_THRESHOLD = 2.0
```

For the verification window compute:

```text
aggregate_peak = max over reads/fingers ||calc_force_bias_corrected||
```

Require:

```text
aggregate_peak <= 2.0 SDK-native units
```

This follows PI-R2's use of a `2.0` aggregate post-bias residual boundary, but DexMani Real must keep the unit labeled unknown.

Do not reuse `_TACTILE_CONTACT_THRESHOLD` by name even though both initial values are `2.0`; they are separate semantics and should be independently tunable later.

### B5. Dense tactile is verified structurally, not by an invented per-taxel threshold

For dense `raw_force` during verification require:

```text
valid
finite
correct shape
```

Optionally log diagnostic statistics such as:

```text
abs max
p99 of per-taxel ||xyz||
```

but do **not** reject calibration based on a hard dense-taxel magnitude cutoff in this task.

Reason:

- PI-R2's hard verification is aggregate `calc_force`, not per-taxel raw force;
- aggregate and dense representations may have different internal scaling/aggregation behavior;
- a per-taxel threshold would be another unverified hardware assumption.

### B6. Failure semantics

If any verification read is invalid/non-finite or aggregate residual exceeds `2.0`:

```text
clear both candidate biases
calibrated=False
return False / raise existing calibration error as appropriate
```

Do not add retry state machines.
One startup attempt with a clear failure is enough for this personal research system.

### B7. `reset_sensor()` remains optional, not default

PI-R2 calls vendor `reset_sensor()` before software biasing.
DexUMI exposes the method but does not demonstrate it as a mandatory deployment path.

For this task:

- do not add a public runtime reset API;
- do not require vendor reset for correctness;
- keep software bias + post-bias verification as the default minimal path;
- if later hardware tests show materially better repeatability with `reset_sensor()`, add it as a private startup calibration step in a separate change.

---

## 4.3 Phase C — decouple contact-only policy from dense tactile

Files:

```text
dexmani_real/robot/hand_worker.py
dexmani_real/deployment/inference/observation.py
```

### C1. Fix tactile-ring `calibrated` semantics

Current publication:

```text
fresh      = dense_valid
calibrated = dense_valid AND calibration_state
```

Change to:

```text
fresh      = dense_valid
calibrated = calibration_state
```

Calibration state and sample validity are independent concepts.

### C2. Contact-only provenance reader ignores dense `fresh`

`_read_tactile_provenance_history()` exists to avoid copying `[5,120,3]` for contact-only policies.

For this path require:

```text
calibrated == True
unit_code == native-unit code
causal source/publish timestamps
age bound
```

Do not require dense `fresh`.

Aggregate sample validity is already enforced from `hand_state_ring` by:

```text
state_valid
tactile_sum_valid
qpos_stale == False
```

Keep exact aggregate/provenance source-timestamp identity.

### C3. Dense path stays strict

`_read_tactile_force_history()` continues to require:

```text
fresh == True
calibrated == True
unit matches
causal and recent
finite dense payload
```

### C4. Required runtime behavior

| Runtime situation | contact-only policy | dense-tactile policy |
|---|---:|---:|
| code `0`, calibrated | run | run |
| RS485 `1501019`, aggregate valid / dense invalid | **run** | reject |
| RS485 `1501018` | reject | reject |
| CRC | reject | reject |
| uncalibrated | reject | reject |

No padding or zero substitution is allowed for a requested invalid tactile modality.

---

## 4.4 Phase D — raw v26 stores both payloads and both validity states

Changing the numeric representation and validity contract must not silently reuse raw v25.

### D1. Bump raw v25 -> v26

File:

```text
dexmani_real/recording/storage/schema.py
```

Set:

```python
EPISODE_SCHEMA_VERSION = 26
```

Keep payload names/shapes:

```text
hand_contact        [N,5,3]       float64  # historical name; payload is calc_force
hand_tactile_force  [N,5,120,3]   float64  # raw_force
```

Do not rename `hand_contact` in this task.

### D2. Add one aggregate freshness field

Add:

```text
tactile_sum_fresh [N] bool
```

Meaning:

```text
hand_contact on this row is a valid aggregate tactile sample
```

Clarify existing:

```text
tactile_fresh [N] bool
```

as:

```text
hand_tactile_force on this row is a valid dense tactile sample
```

Keep common:

```text
tactile_source_monotonic_ns
tactile_calibrated
tactile_unit_code
```

because both payloads come from the same XHand `read_state` source and share one calibration state/unit convention.

### D3. Recording provenance

Update `_recording_provenance()` in:

```text
dexmani_real/teleop/episode_samples.py
```

Aggregate freshness comes from:

```text
hand_state.tactile_sum_valid
+ valid/causal hand source timestamp
+ recording tactile age bound
```

Dense freshness continues to come from `hand_tactile.fresh`.

### D4. Recording-start policy stays strict on both modalities

The collection intent is to preserve both aggregate and dense tactile.

Therefore a new official recording session should still require:

```text
calibrated
aggregate healthy
dense healthy
```

This is intentionally stricter than contact-only policy deployment.

### D5. Unit code

Define code `0` as:

```text
xhand_sdk_native_unknown_si
```

Prefer one named constant, e.g.:

```python
TACTILE_UNIT_CODE_XHAND_SDK_NATIVE = 0
```

Do not add an enum hierarchy for one value.

---

## 4.5 Phase E — processed v15 and Policy Zarr v8

### E1. Bump processed v14 -> v15

Keep both:

```text
contact_force [T,5,3]
tactile_force [T,5,120,3]
```

Update semantics:

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

Do not add magnitude or binary-contact datasets.

For now processed v15 may continue to require both tactile payloads for retained rows, preserving the current paired research artifact. Raw v26 retains independent validity so a future contact-only processed profile can be added without recollection if needed.

### E2. Preserve same-source pairing

Keep:

```text
contact_force[t] and tactile_force[t]
come from the identical selected raw tactile source row
```

Do not independently nearest-match the two signals.

### E3. Bump Policy Zarr v7 -> v8

Zarr v7 contains `contact_force`, whose numeric semantics change after removing `0.1`.
Do not silently keep schema version 7.

Zarr v8 keeps the current key projection unless another task explicitly changes policy modalities:

```text
joint_state
action
action_ee
contact_force
fingertip_points
+ profile-dependent visual fields
```

Do not add dense `tactile_force` to Zarr v8 yet.
The user has intentionally not selected the policy tactile representation.

---

## 5. Historical raw episode repair

Add:

```text
tools/convert_raw_v25_to_v26_tactile.py
```

Model its transaction structure after:

```text
tools/convert_raw_v24_to_v25.py
```

### 5.1 Source is immutable by default

CLI:

```bash
python tools/convert_raw_v25_to_v26_tactile.py SOURCE DESTINATION
```

Support one episode or an episode root.

Use a process-owned staging directory, verify the result, then publish by rename.
Preserve/hard-link `depth.h5` and `rgb.mp4` with the existing EXDEV copy fallback.

Do not default to in-place mutation.

### 5.2 Direct v25 scale correction

For directly-recorded v25 episodes (`converted_from_schema` absent):

```python
hand_contact[...] *= 10.0
hand_tactile_force[...] *= 10.0
```

This is exact despite software bias:

```text
old_bias  = mean(0.1 * SDK_zero)
old_value = 0.1 * SDK - old_bias
          = 0.1 * (SDK - mean(SDK_zero))

new_value = SDK - mean(SDK_zero)
          = 10 * old_value
```

This migration factor is a representation correction only. It has nothing to do with the new contact/residual threshold of `2.0`.

### 5.3 Populate aggregate freshness conservatively

Legacy v25 collapsed aggregate/dense validity.
Therefore migrate:

```text
tactile_sum_fresh = tactile_fresh
```

Do not invent aggregate-valid frames that v25 did not prove.

### 5.4 v24-derived v25 requires explicit source-scale classification

The existing v24 -> v25 tool copies tactile payloads unchanged and writes:

```text
converted_from_schema = 24
```

Historical v24 data spans different code generations. Do not guess scale from payload magnitude.

Default:

```text
REFUSE converted_from_schema == 24
unless source scale is explicitly supplied
```

Suggested options:

```text
--converted-v24-scale legacy-0.1
--converted-v24-scale native
```

Behavior:

```text
legacy-0.1 -> tactile payloads * 10
native     -> tactile payloads unchanged
```

In both cases:

```text
tactile_sum_fresh = tactile_fresh
```

### 5.5 Regenerate derived artifacts

Do not patch processed v14 or Zarr v7 in place.

Use:

```text
raw v26
  -> regenerate processed v15
  -> regenerate Policy Zarr v8 if needed
```

---

## 6. Recommended file-level work plan

### Phase 1 — driver correctness

Edit:

```text
dexmani_real/robot/drivers/xhand.py
```

Tasks:

1. delete `_TACTILE_SCALE` and both multiplications;
2. set `_TACTILE_CONTACT_THRESHOLD = 2.0` SDK-native units;
3. remove `_RAW_FORCE_CONTACT_THRESHOLD`;
4. add RS485-aware `(sum_valid, dense_valid)` helper;
5. split aggregate and dense parsing;
6. compute contact from aggregate validity;
7. remove the hard pre-bias magnitude load gate;
8. keep explicit no-contact startup requirement;
9. make bias capture return candidate arrays;
10. add 3-read post-bias aggregate verification at independent threshold `2.0`;
11. require dense verification frames to be valid/finite but do not invent a dense hard threshold;
12. clear both candidate biases on verification failure.

### Phase 2 — runtime provenance decoupling

Edit:

```text
dexmani_real/robot/hand_worker.py
dexmani_real/deployment/inference/observation.py
```

Tasks:

1. make `calibrated` independent of dense `fresh`;
2. contact-only metadata path ignores dense `fresh`;
3. dense path remains strict;
4. preserve source-timestamp identity checks.

### Phase 3 — raw v26 and migration

Edit/add:

```text
dexmani_real/recording/storage/schema.py
dexmani_real/teleop/episode_samples.py
dexmani_real/recording/frame.py             # only if required by field plumbing
dexmani_real/recording/storage/*            # only owning boundary code
tools/convert_raw_v25_to_v26_tactile.py
docs/data_schema.md
```

Tasks:

1. raw schema 26;
2. `tactile_sum_fresh`;
3. native unknown unit semantics;
4. transactional v25 -> v26 converter;
5. keep both tactile payloads.

### Phase 4 — processed v15 / Zarr v8

Edit:

```text
dexmani_real/dataset/processed.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/export.py
docs/data_schema.md
relevant tests
```

Tasks:

1. processed v15;
2. Policy Zarr v8;
3. updated native-scale representation/unit attrs;
4. no dense tactile in Zarr unless a later policy task requests it;
5. retain same-source pairing.

---

## 7. Required tests

Use fake SDK state/error objects. Do not require hardware for unit validation.

### 7.1 Driver validity matrix

| Comm | code | joints | sum | dense |
|---|---:|---:|---:|---:|
| serial | `0` | valid | valid | valid |
| serial | `1501018` | valid | invalid | invalid |
| serial | `1501019` | valid | valid | invalid |
| serial | `1501020` | valid | valid | valid |
| serial | `1501070` | valid when joint payload is valid | invalid | invalid |
| ethercat | `0` | valid | valid | valid |
| ethercat | nonzero tactile-status code | preserve existing joint policy | invalid | invalid |

Also prove:

```text
malformed raw_force does not erase valid calc_force
malformed calc_force does not fabricate aggregate validity
```

### 7.2 Native-scale test

With zero bias, fake SDK values must emerge unchanged numerically.

Example:

```text
calc_force = [10,20,30]
raw taxel  = [4,5,6]
```

With known bias:

```text
output = SDK - bias
```

No `0.1` factor remains.

### 7.3 Aggregate contact threshold test

Test the new provisional threshold independently:

```text
||calc_force|| < 2.0  -> no contact
||calc_force|| = 2.0  -> follow the implementation's explicit > / >= convention
||calc_force|| > 2.0  -> contact
```

Do not test compatibility with the old effective cutoff `10`; behavior is intentionally allowed to become more sensitive.

### 7.4 Calibration tests

Using deterministic fake reads, prove:

1. valid five-read bias capture + three low-residual verification reads -> calibrated;
2. verification aggregate residual > `2.0` -> biases cleared;
3. invalid aggregate payload during capture -> failure;
4. invalid dense payload during capture -> failure;
5. invalid/non-finite aggregate verification payload -> failure and biases cleared;
6. invalid/non-finite dense verification payload -> failure and biases cleared;
7. dense verification magnitude alone does not fail calibration when it is valid/finite;
8. no pre-bias absolute-magnitude contact gate remains.

### 7.5 Contact-only deployment regression

Critical case:

```text
hand_state:
    tactile_sum_valid=True
    source=t

hand_tactile metadata:
    fresh=False
    calibrated=True
    unit=native
    source=t

contact_force-only policy
=> succeeds

tactile_force policy
=> rejects
```

### 7.6 Raw migration tests

Synthetic direct v25 episode:

- both tactile payloads exactly `old * 10`;
- all non-tactile datasets bit-identical;
- `tactile_sum_fresh == old tactile_fresh`;
- schema becomes 26;
- source remains unchanged;
- v24-derived input refuses without explicit scale option;
- `legacy-0.1` scales;
- `native` preserves values;
- failed conversion removes staging output.

### 7.7 Processed / Zarr tests

Prove:

```text
processed v15 contact_force == selected raw v26 hand_contact
processed v15 tactile_force == selected raw v26 hand_tactile_force
Policy Zarr v8 contact_force == processed v15 contact_force
```

and:

```text
unit = xhand_sdk_native_unknown_si
si_verified = False
```

Never assert:

```text
contact_force == tactile_force.sum(...)
```

---

## 8. Validation commands

Before editing:

```bash
git status --short
```

After relevant phases:

```bash
python -m compileall -q dexmani_real tools examples
python -m unittest tests.test_deployment_eef_tactile
python -m unittest tests.test_tactile_selector
```

Run the new focused driver/calibration/migration tests explicitly.
Update/rename processed and Zarr tests with their schema versions as appropriate.
Then, if practical:

```bash
python -m unittest discover -s tests
```

Finish with:

```bash
git diff --check
git diff --stat
git status --short
```

Do not automatically run XHand hardware-affecting examples.

---

## 9. Manual hardware verification after code review

Do this only when explicitly requested.

Use a stationary tactile-read workflow; no robot motion is needed.

### 9.1 Startup calibration

With fingertips free/unloaded:

1. run multiple startup calibrations;
2. inspect post-bias per-finger aggregate norms;
3. verify the `2.0` residual threshold is not producing routine false failures;
4. record aggregate residual mean/max and dense p99/max for later characterization.

### 9.2 Light-contact sensitivity

After calibration, apply very light fingertip contact and inspect:

```text
||calc_force||
tactile_contact at threshold 2.0
dense taxel activation
```

The purpose is not to prove Newton units. It is to verify that `2.0` is a useful provisional sensitivity for this hardware rather than the much coarser DexUMI cutoff `10`.

### 9.3 Threshold adjustment rule

If no-contact residuals frequently exceed `2.0`, do not silently increase all thresholds together.
Measure the distribution first and adjust:

```text
contact threshold
calibration residual threshold
```

independently.

A later characterization task can replace fixed defaults with evidence from repeated no-contact and light-contact trials.

---

## 10. Explicit non-goals / reject these implementation directions

- Do not keep `0.1` in the driver as ML normalization.
- Do not call SDK numeric values Newtons without physical calibration.
- Do not copy DexUMI's `10` cutoff as the default contact threshold.
- Do not introduce a dense raw-force contact threshold merely because the old driver had one.
- Do not reduce canonical Real data to binary contact.
- Do not drop dense tactile because the current policy may not consume it yet.
- Do not make dense tactile mandatory for contact-only deployment.
- Do not linear-interpolate tactile across time.
- Do not add `reset_sensor()` as a public runtime API in this task.
- Do not hardcode `10x12` taxel geometry.
- Do not rename `hand_contact` while fixing scale/validity; document its `calc_force [5,3]` semantics.
- Do not patch processed/Zarr artifacts in place; regenerate from raw v26.
- Do not add managers, registries, plugin layers, or generic sensor abstractions.

---

## 11. Completion criteria

```text
[ ] driver emits SDK-native bias-corrected calc_force and raw_force
[ ] no _TACTILE_SCALE=0.1 remains in production tactile code
[ ] aggregate convenience contact threshold is 2.0 SDK-native units, not labeled N
[ ] calibration residual threshold is a separate 2.0 SDK-native constant
[ ] no dense raw-force hard contact/residual threshold is introduced
[ ] no hard pre-bias magnitude gate is used to pretend uncalibrated load can be identified reliably
[ ] RS485 1501019 yields sum-valid / dense-invalid
[ ] contact-only deployment survives dense-invalid / sum-valid frames
[ ] dense-tactile deployment rejects dense-invalid frames
[ ] calibration performs independent post-bias verification
[ ] calibration failure clears both biases
[ ] raw v26 stores both payloads + separate aggregate/dense freshness
[ ] direct raw v25 -> v26 conversion scales both payloads by exactly 10
[ ] converted v24-derived episodes require explicit scale classification
[ ] processed v15 preserves both tactile representations in SDK-native scale
[ ] Policy Zarr v8 does not silently reuse v7 tactile numeric semantics
[ ] no SI/Newton claim is introduced
[ ] same-source causal pairing remains unchanged
[ ] focused offline tests pass
```

Final principle:

```text
Real owns faithful, bias-corrected sensor observations.
Small thresholds are provisional derived heuristics, not sensor units.
Policy owns representation choice and ML normalization.
```
