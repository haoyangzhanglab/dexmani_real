# XHand tactile atomicity and dataset invariant follow-up fix plan

> Canonical follow-up implementation plan for Codex / Claude Code.
>
> Reviewed GitHub baseline: `main` at `832c2ffff5b4774177ea45cbaa257c947c75bc86` on 2026-09-11.
> Source code is authoritative if `main` advances before implementation.
>
> **Status: PROPOSED — NOT IMPLEMENTED.**
>
> This is a narrow follow-up to the completed raw-v29 / processed-v20 / Zarr-v13 tactile simplification. It does **not** reopen the acquisition architecture. The current driver, one-hand-ring IPC, recording schema, direct raw validity masks, and software-bias ownership are retained.

---

## 0. Executive decision

The current tactile refactor is structurally correct for data acquisition and recording, but two follow-up issues should be closed before the Real tactile pipeline is considered frozen:

1. **Deployment atomicity:** deployment currently reads qpos, aggregate tactile, and dense tactile from the same `hand_state_ring`, but filters each field independently before control-grid alignment. A newest tactile-invalid hand record can therefore be removed before alignment and an older tactile-valid record can be selected while qpos comes from the newer hand sample. This silently recreates `qpos_t + tactile_{t-1}` backward repair.
2. **Dataset mask/payload invariant:** the canonical recorder writes invalid tactile as `NaN + valid=False`, and the schema/docs already describe that contract, but raw processing and processed validation do not consistently reject the contradictory state `finite payload + valid=False`.

The repair is intentionally small:

```text
A. align one hand sample once, then project qpos/aggregate/dense
B. validate tactile mask/payload consistency; never repair it
```

No schema version bump is required. These changes make the existing v29/v20/v13 semantics match their already-declared contract.

---

## 1. Non-goals

Do not change or redesign:

```text
XHand driver parsing
_tactile_validity() status matrix
software bias / zeroing
HAND_STATE_DTYPE
raw v29 field set
processed v20 field set
Zarr v13 field set
command safety
worker failure timeout
tactile units / SI semantics
policy tactile encoder / normalization
historical raw migration
```

Do not reintroduce:

```text
hand_tactile_ring
backward tactile repair
fresh/calibrated/unit per-row fields
validity reason enums
quality scores
sensor-health frameworks
interpolation
```

---

## 2. Fact-checked issue A — deployment can still mix hand sample identities

### 2.1 Current behavior

`deployment/inference/observation.py::_build_observation()` currently constructs separate histories from the same `hand_state_ring`:

```text
hand qpos history:
    required state_valid, !qpos_stale

aggregate history:
    required state_valid, tactile_aggregate_valid, !qpos_stale

dense history:
    required state_valid, tactile_dense_valid, !qpos_stale
```

Each history is then aligned independently to the same policy control-grid references.

This is still a hidden repair mechanism because validity filtering happens **before** sample selection.

Example at ~30 Hz:

```text
sample k-1 @ reference-53ms:
    qpos valid
    aggregate valid
    dense valid

sample k @ reference-20ms:
    qpos valid
    aggregate valid
    dense invalid
```

With `max_observation_skew_s = 0.10`, independent filtering can produce:

```text
qpos      -> sample k
aggregate -> sample k
dense     -> sample k-1
```

The values are all causal and within skew, but they are not one XHand sample.

### 2.2 Required invariant

For every aligned policy hand timestep `j`:

```text
hand sample identity(j)
    == qpos identity(j)
    == aggregate identity(j)
    == dense identity(j)
```

Validity is checked **after** that sample identity is chosen.

Therefore:

```text
selected sample aggregate_valid=false + contact_force requested
    -> observation unavailable
    -> DO NOT search an older aggregate sample

selected sample dense_valid=false + tactile_force requested
    -> observation unavailable
    -> DO NOT search an older dense sample
```

A modality that is not requested must not gate the observation.

---

## 3. Fix A — read and align one hand history once

Primary file:

```text
dexmani_real/deployment/inference/observation.py
```

### 3.1 Preferred implementation

Introduce one narrow, domain-specific hand history container in this file. Do not create a general sensor abstraction.

Conceptually:

```python
@dataclass(frozen=True)
class HandFrameWindow:
    qpos: np.ndarray                    # [T,12] float64
    tactile_aggregate: np.ndarray       # [T,5,3] float32
    tactile_dense: np.ndarray           # [T,5,120,3] float32
    tactile_aggregate_valid: np.ndarray # [T] bool/uint8
    tactile_dense_valid: np.ndarray     # [T] bool/uint8

    source_sequence: np.ndarray         # [T]
    source_monotonic_ns: np.ndarray     # [T]
    publish_monotonic_ns: np.ndarray    # [T]
    valid_mask: np.ndarray              # [T]
```

The exact class name may differ, but the ownership must stay local to deployment observation assembly.

### 3.2 Read once

Add a helper similar to:

```python
def _read_hand_history(... ) -> HandFrameWindow | None:
```

It scans `shared.hand_state_ring` once and filters only the **hand sample itself**:

```text
state_valid == true
qpos_stale == false
source/publish causality valid
not before current run
within state max-age bound
qpos finite
```

Do **not** filter records by:

```text
tactile_aggregate_valid
tactile_dense_valid
```

Do not reject a hand record merely because one tactile representation is unavailable.

Copy the aggregate/dense payload and both validity bits from the same structured record. Invalid IPC tactile may be zero-filled; the validity bits are authoritative at this runtime boundary.

### 3.3 Align once

Align the `HandFrameWindow` to `reference_ns` once using the same latest-causal / skew semantics already used for state history.

Conceptually:

```text
raw hand history
    ↓
selected indices = align once(reference_ns)
    ↓
aligned hand history
```

All hand-derived fields must use those exact indices.

Do not run `_align_state_history_to_reference_ns()` independently for qpos / aggregate / dense.

### 3.4 Gate requested tactile after alignment

After `aligned_hand` exists:

```python
if contact_force_requested and not np.all(aligned_hand.tactile_aggregate_valid):
    return None

if tactile_force_requested and not np.all(aligned_hand.tactile_dense_valid):
    return None
```

Only requested modalities gate inference.

Examples:

```text
joint_state only + dense invalid
    -> valid observation

contact_force requested + aggregate valid + dense invalid
    -> valid observation

tactile_force requested + dense invalid
    -> no observation

contact_force + tactile_force requested
    -> both valid on every selected hand sample
```

### 3.5 Project without re-selection

`_to_policy_observation()` should build arrays directly from the already-aligned hand sample window:

```text
joint_state:
    arm_history.qpos + aligned_hand.qpos

contact_force:
    aligned_hand.tactile_aggregate

tactile_force:
    aligned_hand.tactile_dense
```

Do not create a second source-selection or repair path here.

### 3.6 Observation timing

`observation_timing_ms()` should count the aligned hand source once, not separately count qpos/contact/dense source timestamps that are now guaranteed identical.

This makes reported cross-modality skew represent actual sensor-modality skew rather than duplicate projections of the same XHand sample.

### 3.7 Preferred cleanup

If straightforward, simplify `ObservationBatch` by replacing separate:

```text
hand_history
hand_contact_history
hand_tactile_force_history
```

with one aligned hand window.

If that would create disproportionate unrelated churn, retaining projected `FrameWindow` fields is acceptable only if they are built from the same selected indices and no field performs independent history selection.

Correct sample identity is mandatory; the exact local representation is not.

---

## 4. Fix A tests — prove no fallback, not merely eventual failure

Primary file:

```text
tests/test_deployment_eef_tactile.py
```

The critical regression tests must use a **single newest invalid sample with an older valid sample still inside the skew bound**.

### 4.1 Dense no-fallback regression

Construct:

```text
k-1: dense_valid=true
k:   dense_valid=false
reference >= sample k
sample k-1 still within max_observation_skew_s
```

Expected:

```text
tactile_force requested -> _build_observation(...) is None
```

This test fails under the current independently filtered history behavior and passes only when validity is checked after sample selection.

### 4.2 Aggregate no-fallback regression

Same shape:

```text
k-1: aggregate_valid=true
k:   aggregate_valid=false
```

Expected:

```text
contact_force requested -> None
```

### 4.3 Partial-validity behavior

Selected newest sample:

```text
aggregate_valid=true
dense_valid=false
```

Expected:

```text
joint_state only      -> pass
contact_force only    -> pass
tactile_force only    -> fail
contact + dense       -> fail
```

### 4.4 Identity regression

When both aggregate and dense are valid, directly assert that all hand-derived policy modalities use the same selected hand `source_sequence` / `source_monotonic_ns` as hand qpos for every history slot.

Do not rely only on equal numeric values; pin source identity.

---

## 5. Fact-checked issue B — tactile mask/payload contract is not fully enforced

### 5.1 Declared and producer behavior

The canonical recording boundary already produces:

```text
valid tactile:
    finite payload + valid=true

invalid tactile:
    all-NaN payload + valid=false
```

Raw v29 documentation describes the same semantics.

### 5.2 Current validation gap

The current processing/processed validators guarantee at most:

```text
valid=true -> payload finite
```

They do not consistently reject:

```text
valid=false + finite payload
```

That state should not be admitted by the canonical v29/v20 pipeline because it reintroduces ambiguity about whether an invalid finite payload is archival sensor data, a placeholder, or usable information.

---

## 6. Fix B — one tiny tactile mask/payload invariant

The canonical invariant is:

```text
valid == true  <=> payload row is fully finite
valid == false <=> payload row is all NaN
```

For the canonical recorder this is intentionally stronger than merely saying the mask is authoritative.

It distinguishes exactly two states:

```text
real zero/no-contact: finite zero + valid=true
invalid measurement:  NaN + valid=false
```

Do not add reason enums or partial payload semantics.

### 6.1 Raw processing admission

Primary file:

```text
dexmani_real/dataset/processing.py
```

`analyze_episode()` already reads:

```text
hand_contact
hand_contact_valid
hand_tactile_force
hand_tactile_force_valid
```

Add a small pure helper or local check that enforces for each representation:

```python
rows_finite = np.all(np.isfinite(payload), axis=payload_axes)
rows_all_nan = np.all(np.isnan(payload), axis=payload_axes)

if np.any(valid & ~rows_finite):
    raise ValueError(...)
if np.any(~valid & ~rows_all_nan):
    raise ValueError(...)
```

Do not repair contradictory rows in processing.

A corrupted/third-party raw row is a technical contract error and must fail loudly.

### 6.2 Processed validation

Primary file:

```text
dexmani_real/dataset/processed.py
```

Extend the existing chunked `_VALIDITY_MASKED_KEYS` validation:

```text
valid row   -> fully finite
invalid row -> all NaN
```

Keep it chunked so dense tactile validation does not require an unnecessary whole-dataset temporary copy.

### 6.3 No schema bump

Do not bump:

```text
raw v29
processed v20
Zarr v13
```

The documented semantics already say invalid payloads are NaN. This patch brings validators into compliance with the existing schema contract; it does not define a new persisted representation.

---

## 7. Fix B tests

Primary files:

```text
tests/test_control_step_dataset.py
tests/test_recording_control_contact.py
```

Add/adjust tests:

### Raw processing

```text
valid=true + finite       -> pass
valid=true + NaN          -> fail
valid=false + all NaN     -> pass
valid=false + finite      -> fail
```

Test aggregate and dense independently.

Replace any existing test that expects `valid=false + finite payload` to survive processing; that expectation contradicts the canonical v29 contract.

### Processed validator

After producing a valid processed artifact, tamper one row:

```text
contact_force_valid=false
contact_force=row finite
```

Expected:

```text
validate_processed_hdf5 -> ValueError
```

Repeat for dense if useful; one parameterized test is preferable to duplicated machinery.

### Recording boundary

Retain existing tests proving:

```text
valid zero -> finite zero + true
invalid aggregate -> NaN + false
invalid dense -> NaN + false
partial validity remains independent
```

---

## 8. Data usability decision

### 8.1 New v29 -> v20 -> v13 data

The new data path is already suitable for a hardware pilot because recording itself is atomic:

```text
one selected causal HAND_STATE record
    -> qpos/current/aggregate/dense
    -> direct validity masks
```

Issue A affects deployment observation assembly, not raw recording identity.

After Fix B, data corruption involving mask/payload contradictions also fails loudly.

Recommended workflow after this patch:

```text
5-10 short hardware pilot episodes
    ↓
offline inspect validity ratios + sample ages + zero baselines
    ↓
if healthy, freeze Real tactile acquisition/data pipeline
    ↓
start main tactile dataset collection
```

Suggested pilot diagnostics:

```text
aggregate_valid ratio
dense_valid ratio
aggregate-valid / dense-invalid partial-status frequency
hand age = observation_anchor - hand_source
no-contact aggregate residual statistics
dense residual statistics
session-to-session zero/bias stability
```

Do not add these diagnostics as runtime validity gates.

### 8.2 Historical salvage data

Historical v26-derived / processed-v18 / Zarr-v11 salvage artifacts remain useful for:

```text
vision/proprio baselines
3D policy development
training-pipeline smoke tests
careful single-tactile-modality baselines respecting old masks
```

They are not the canonical source for experiments whose scientific claim depends on exact same-read hand/tactile identity, including:

```text
aggregate-vs-dense synchronization claims
calc_force vs raw_force same-read regression
contact-transition timing analysis
dense tactile final paper experiments
```

Do not migrate legacy raw into v29 and claim the new atomic semantics.

---

## 9. Files expected to change

The follow-up should stay small. Expected runtime/data changes:

```text
dexmani_real/deployment/inference/observation.py
dexmani_real/dataset/processing.py
dexmani_real/dataset/processed.py
```

Expected tests/docs:

```text
tests/test_deployment_eef_tactile.py
tests/test_control_step_dataset.py
possibly tests/test_recording_control_contact.py
docs/xhand_tactile_research_simplification_plan.md  # mark implemented if appropriate
docs/data_schema.md                                 # only if wording needs sync
```

Do not touch driver/worker/schema unless source inspection reveals a concrete violation of this plan.

---

## 10. Verification

No hardware commands are part of this fix.

Run focused checks first:

```bash
python -m compileall -q dexmani_real examples
pytest -q tests/test_deployment_eef_tactile.py
pytest -q tests/test_recording_control_contact.py
pytest -q tests/test_control_step_dataset.py
pytest -q tests/test_observation_builder.py
pytest -q tests/test_control_step_export.py
pytest -q tests/test_recorder_io_boundary.py
git diff --check
```

Then, if the focused suite passes, run the repository's ordinary offline test suite if practical.

No test may be made green by weakening:

```text
causal source/publish ordering
max input age
max observation skew
run generation isolation
state_valid / qpos_stale handling
robot command safety
```

---

## 11. Acceptance criteria

The follow-up is complete when all are true:

### Deployment atomicity

```text
hand history scanned once for aligned hand samples
sample identity selected before tactile validity gating
no requested tactile modality can fall back to an older XHand sample
qpos/aggregate/dense for one policy timestep share exact hand source identity
unrequested invalid tactile never blocks unrelated modalities
```

### Dataset contract

```text
valid tactile row   -> finite payload
invalid tactile row -> all-NaN payload
raw processing rejects contradictory rows
processed validator rejects contradictory rows
processing does not repair mask/payload contradictions
```

### Scope

```text
no schema bump
no second tactile ring
no validity framework
no unit/freshness reintroduction
no historical migration layer
no policy tactile architecture work
```

### Safety

All existing hardware-command and lifecycle safety behavior is unchanged.

---

## 12. Final desired state

After this patch the complete semantics are simple:

```text
XHand read
    ↓
one HAND_STATE record
    ↓
recording:
    select one causal hand sample
    persist qpos + aggregate + dense + two validity bits

processing:
    verify mask/payload invariant
    copy tactile + validity directly

deployment:
    select one causal hand sample per policy grid slot
    then check requested tactile validity
    never substitute an older tactile sample
```

The Real tactile pipeline should then be frozen unless a real hardware pilot reveals a measured acquisition problem.
