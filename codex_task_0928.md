# Codex Task 0928 — Simplify the DexMani Real ↔ Policy Boundary

## 0. Task intent

Refactor `dexmani_real` and `dexmani_policy` toward a **research-oriented, paper-code architecture** comparable in spirit to Diffusion Policy, DexUMI and ManiUniCon.

The goal is **not** to build a production-grade schema/ABI/compatibility system. The goal is to keep the parts that directly affect:

1. training tensor correctness,
2. experiment reproducibility,
3. real-robot numerical consistency,
4. physical runtime safety,

while deleting duplicated Real semantic-contract machinery that does not materially improve a personal research codebase.

The intended end state is:

```text
strict Raw
   ↓
strict offline export
   ↓
simple multimodal Zarr
   ↓
generic Policy Dataset
   ↓
Policy + normal checkpoint/config
   ↓
minimal numerical runtime metadata
   ↓
Real observation/action adapters
   ↓
strict physical runtime safety
```

Prefer **deletion and local validation** over new abstractions.

Do **not** replace the current contract system with another schema/capability/representation framework.

---

## 1. Repository/environment assumptions

Local checkout layout:

```text
<workspace>/
├── dexmani_real/      # current working repository
└── dexmani_policy/    # sibling repository: ../dexmani_policy
```

Conda environments:

```text
dexmani_real   → conda env: real_robot
dexmani_policy → conda env: policy
```

Use the environment corresponding to the repository whose code is being executed, for example:

```bash
# from dexmani_real/
conda run --no-capture-output -n real_robot python ...

# for the sibling policy repo
cd ../dexmani_policy
conda run --no-capture-output -n policy python ...
```

Before editing, inspect both worktrees:

```bash
git -C . status --short
git -C ../dexmani_policy status --short
```

Do not overwrite or revert unrelated local changes.

---

## 2. Reviewed baselines

This task was designed against:

### DexMani

- `haoyangzhanglab/dexmani_policy@acf11884a65205e40fb1a427c0e15c312a634abe`
- `haoyangzhanglab/dexmani_real@83d273d1ba625d39bdb5e8e5a7be2b91d6ee6abf`

Re-check the current local code before applying edits. If the relevant code has moved, preserve the intent below rather than mechanically applying paths.

### Reference repositories used for fact-checking

- Diffusion Policy: `real-stanford/diffusion_policy@5ba07ac6661db573af695b419a7947ecb704690f`
- 3D Diffusion Policy / DP3: `YanjieZe/3D-Diffusion-Policy@47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97`
- SAT: `XiaohanLei/SAT@cd7c0a8877d6090a9a85ebee0ceca961830b3654`
- R3D-Policy: `Wushr-Lance/R3D-Policy@e637c0148376ddc4b5e667fa8f8e108cb8ff7a85`
- DexUMI: `real-stanford/DexUMI@acddb8f8a89a8f0186868bbec44306eb7808114a`
- ManiUniCon: `Universal-Control/ManiUniCon@85c6f2e32ecf9f2bed62d202b058c39623444686`

---

## 3. Decisions already fact-checked — DO NOT revisit in this task

These decisions are intentional and are **out of scope for modification**.

### 3.1 Keep full-buffer normalization statistics

`BaseDataset.iter_normalization_data()` currently uses the complete ReplayBuffer rather than `train_mask`.

Keep this behavior.

This is consistent with the common Diffusion Policy / DP3 / SAT-style dataset normalization pattern and keeps DexMani aligned with its reference implementations.

Do not introduce train-only normalization in this task.

### 3.2 Keep canonical export strict for contact/tactile and all training modalities

Raw recording may preserve unavailable sensor evidence as NaN. That is acceptable at the Raw layer.

However, the processed/canonical Zarr is a **training-ready multimodal dataset**. Keep the current policy that canonical export rejects non-finite floating values, including:

- contact force,
- dense tactile force,
- arm velocity,
- arm effort,
- hand current,
- state/action/derived geometry,
- point cloud.

Do not defer these failures to training.

A failed/invalid optional sensor in Raw should therefore make that episode ineligible for the canonical multimodal training export unless explicitly excluded/fixed upstream.

Do not replace invalid values with zeros.

### 3.3 Keep `use_aux_ee` normalization behavior

Do not change `use_aux_ee` normalization in this task.

Fact-check:

- Official SAT does not contain the same joint-action + auxiliary-EE target mechanism.
- DexMani's `use_aux_ee` is modeled after official R3D-Policy `use_target_ee`.
- Official R3D concatenates joint action and target EE and applies the same `limits` normalizer to the combined action tensor.

Therefore the current DexMani behavior is consistent with the actual R3D reference design.

Do not reinterpret the auxiliary rot6d block as requiring identity normalization as part of this cleanup.

### 3.4 Do not refactor MultiTask sampling yet

Leave the following behavior untouched:

- `MultiTaskDataset.sampling_strategy`
- balanced/proportional/weighted logic
- `mp.Manager` epoch synchronization
- current persistent-worker behavior
- current outer `ResumableDistributedSampler`

It remains a possible future cleanup, but is not part of 0928.

### 3.5 Keep ReplayBuffer min/max logging

Do not remove the current full-array min/max logging.

Fact-check:

- official Diffusion Policy does not print the ranges,
- official DP3 does print `shape / dtype / min~max` for ReplayBuffer fields,
- DexMani's current logging follows the DP3 lineage and is useful for 3D research/debugging.

Optimize only if it becomes a measured bottleneck later.

### 3.6 Keep the following mechanisms unchanged

Do not redesign or simplify:

- `SequenceSampler` core behavior,
- PC/RGB thin dataset wrappers,
- full-buffer normalizer semantics,
- strict training resume contract,
- strict checkpoint restoration,
- raw/EMA explicit selection,
- RGB preprocessing ownership,
- point-cloud numerical algorithm,
- current physical safety state machine,
- HOME/run authorization,
- observation freshness checks,
- stale inference discard,
- joint/hand/workspace clipping,
- IK validation,
- worker readiness/shutdown,
- control-rate validation,
- Raw immutability,
- Raw strict camera/calibration validation,
- MultiTask sampling.

---

# 4. Core architecture change

The current duplicated Real semantic path is approximately:

```text
dexmani_real
CANONICAL_MODALITY_SEMANTICS
        ↓
per-array Zarr semantic attrs
        ↓
dexmani_policy/datasets/real_policy_contract.py
_SEMANTICS + validate_modality_contract()
        ↓
real_runtime.modality_contracts
        ↓
dexmani_real/deployment/config.py
CANONICAL_MODALITY_SEMANTICS comparison again
```

Delete this duplicated semantic-contract path.

After this task, the two repository interfaces should be:

## Training interface

```text
dexmani_real processed Zarr
        ↓
dexmani_policy ReplayBuffer / Dataset
```

## Deployment interface

```text
saved config.yaml + normal checkpoint
        ↓
minimal Real numerical metadata
        ↓
dexmani_real observation/action adapters
```

The executable code that constructs observations/actions is the operational definition of the representation:

- `dexmani_real/deployment/observation.py::build_policy_observation`
- `dexmani_real/deployment/action.py::decode_policy_action`

Do not create a second metadata language describing those functions.

---

# 5. Target canonical Zarr metadata

Keep the canonical Zarr multimodal arrays and `meta/episode_ends`.

Remove per-array semantic-contract attrs from the runtime path.

The root attrs should contain only compact dataset/numerical facts:

```text
format
task_name
dt
depth_scale_m_per_unit
pointcloud_config
fingertip_link_names
```

Precise intent:

### `format`

Keep:

```text
dexmani.real.canonical
```

It is only a marker that this dataset was generated by the current DexMani Real canonical exporter.

Do not add:

- schema version,
- compatibility version,
- migration version,
- version negotiation.

If an old processed cache no longer satisfies the current format expectations, regenerate it from immutable Raw.

### `task_name`

Keep and validate.

It prevents accidental training with the wrong task dataset.

### `dt`

Keep and require finite positive value.

It is a real training/deployment temporal parameter.

### `depth_scale_m_per_unit`

Move the depth-scale numerical fact to root attrs.

Because canonical depth remains `uint16`, this value is needed to interpret it physically.

Require one uniform depth scale across all episodes in one canonical store.

### `pointcloud_config`

Persist `ProcessingConfig.pointcloud.to_dict()` at the root.

This is not a semantic contract; it is the numerical preprocessing configuration that changes the actual point-cloud tensor distribution.

It must remain available so a Real deployment of a point-cloud policy can reconstruct the same numerical preprocessing with current physical calibration.

Do **not** put the historical table plane into this runtime config.

### `fingertip_link_names`

Persist the actual five fingertip link names used to derive `fingertip_points`.

They are a numerical geometry choice: changing the selected links changes the resulting observation.

Do not persist a larger fingertip semantic contract.

The current hand mount remains per-Raw-episode physical evidence during offline derivation; live deployment uses the current physical hand mount.

---

# 6. Preserve export provenance without putting it in runtime contracts

The current point-cloud semantic contract contains `export_provenance.table_plane_abcd`.

Remove that semantic-contract mechanism, but do not silently lose useful provenance.

Use the existing `export_report.json` for provenance.

Add the resolved processing snapshot to the report, for example:

```json
{
  "processing": {
    "pointcloud": { "...": "..." },
    "table_plane_abcd": [ ... ],
    "fingertip_link_names": [ ... ]
  }
}
```

`ProcessingConfig.to_dict()` already exists and is appropriate for this purpose.

The policy runtime must **not** read `export_report.json`.

Deployment continues to use:

- current camera calibration,
- current table calibration,
- current hand physical mount,

combined with the saved numerical preprocessing parameters.

---

# 7. dexmani_real changes

## 7.1 `dexmani_real/dataset/contracts.py`

Keep this file for now to avoid low-value file churn.

Do not rename it and do not introduce a replacement schema module.

Keep only the useful compact pieces:

- `validate_task_name`,
- `ProcessingConfig`,
- canonical array shape/dtype definitions,
- `canonical_array_specs`,
- `CANONICAL_FORMAT`.

Remove runtime semantic machinery:

- `FINGERTIP_KINEMATIC_MODEL`,
- `CANONICAL_MODALITY_SEMANTICS`,
- `canonical_modality_contracts()`,
- imports used only by those semantic tables.

Keep `ProcessingConfig.to_dict()`; use it in the export report.

Do not alter the actual canonical array names, shapes, or dtypes as part of this task.

## 7.2 `dexmani_real/dataset/export.py`

Remove `canonical_modality_contracts`.

Do not write per-array semantic attrs.

When creating the root store, persist the compact root attrs defined above.

Required checks across episodes in one store:

- task name is uniform,
- `dt` is uniform,
- depth scale is uniform,
- array tail shapes/dtypes are uniform.

Point-cloud config and fingertip link names are supplied once by the resolved `ProcessingConfig` for the export and therefore should be written once to root attrs.

Continue to:

- refuse overwrite,
- write atomically via staging,
- preflight complete episodes before publishing output,
- reject non-finite training data,
- reject transformed non-finite values,
- validate rot6d outputs,
- require complete canonical modalities,
- preserve `export_report.json`.

Add `processing.to_dict()` to the export report.

Do not add a migration path for old Zarr caches.

## 7.3 `dexmani_real/dataset/processing.py`

Do not weaken `validate_episode()`.

Keep strict finite checks for all floating Raw fields required by the canonical multimodal export.

Keep transformed block finite checks.

Keep point-cloud, EEF and fingertip derivation unchanged.

Only update imports if semantic-contract symbols are removed.

## 7.4 `dexmani_real/deployment/config.py`

Delete dependency on:

- `CANONICAL_MODALITY_SEMANTICS`,
- `FINGERTIP_KINEMATIC_MODEL`,
- saved `modality_contracts`.

Keep the actual Real capability/safety checks:

- hand-enabled requirement,
- positive finite policy `dt`,
- supported live observation fields,
- `joint_state` requirement,
- supported action mode,
- policy rate <= worker rate,
- point-cloud config validation via `PointCloudConfig.from_dict`,
- recording budget checks.

For point-cloud policies:

- require `info.pointcloud_config`,
- parse it with `PointCloudConfig.from_dict()`,
- return that validated config exactly as the current session code expects.

For fingertip policies:

- require `info.fingertip_link_names`,
- require exactly five distinct non-empty strings.

Do not compare semantic labels, units, frames, axis names, joint-order metadata, or alignment strings.

## 7.5 `dexmani_real/deployment/observation.py`

Replace:

```python
policy_info.modality_contracts["fingertip_points"]["recipe"]["fingertip_link_names"]
```

with the compact saved field:

```python
policy_info.fingertip_link_names
```

Keep all FK computations and live observation construction unchanged.

## 7.6 Other Real deployment files

Keep `session.py`, `runner.py`, `action.py` and runtime safety behavior unchanged except for field-name plumbing required by the new `PolicyInfo`.

Do not redesign process ownership or shared memory.

---

# 8. dexmani_policy changes

## 8.1 Delete `dexmani_policy/datasets/real_policy_contract.py`

Delete the file after all imports/callers are removed.

Do not move it to another directory.

Do not create:

- `schema.py`,
- `contract.py`,
- `capabilities.py`,
- `representation.py`,
- a registry abstraction.

## 8.2 `dexmani_policy/datasets/base_dataset.py`

Remove all Real-specific behavior:

- import of `real_policy_contract`,
- `self.real_contract`,
- `read_real_contract(...)`,
- `validate_real_finiteness(...)`.

The dataset loading path should reduce to the normal selected-key path:

```python
load_keys = list(sensor_modalities) + [action_key]
if use_aux_ee:
    load_keys.append("action_ee")

self.replay_buffer = ReplayBuffer.copy_from_path(
    self.zarr_path,
    keys=load_keys,
)
```

Preserve:

- split logic,
- SequenceSampler,
- augmentation,
- RGB processing,
- `obs_horizon`,
- action composition,
- full-buffer normalization data,
- validation determinism.

After this change, `datasets/` should contain no Real/canonical/hardware semantic logic.

## 8.3 `dexmani_policy/datasets/replay_buffer.py`

Do not change behavior in this task except what is strictly necessary for removed Real plumbing.

In particular:

- keep in-memory loading,
- keep float casting behavior,
- keep current full min/max logging,
- do not add a new full-array finiteness scan,
- do not add lazy Zarr,
- do not re-import the full Diffusion Policy ReplayBuffer implementation.

Canonical finiteness is an exporter responsibility for this research workflow.

## 8.4 Add a tiny Real metadata capture helper under deployment, not datasets

Place the Real-specific training/deployment metadata reader in:

```text
dexmani_policy/deployment/runtime.py
```

Do not create a new subsystem unless necessary.

A helper such as:

```python
capture_real_runtime(dataset, cfg) -> dict | None
```

is sufficient.

It should inspect the already-loaded dataset's ReplayBuffer root attrs; do not reopen and rescan the Zarr.

Expected behavior:

### Simulation / ordinary dataset

If it is not a DexMani Real dataset:

```python
return None
```

Do not add `real_runtime: null` to every simulation config.

### Unsupported old Real cache

If metadata clearly identifies a Real dataset but `format != "dexmani.real.canonical"`, fail explicitly.

A lightweight guard is enough, e.g. current Real domain marker / `dexmani.real.*` format marker.

Do not scan array semantic attrs to identify legacy formats.

Do not implement migration or fallback reading.

### Current canonical Real

Validate only:

- `task_name == cfg.task_name`,
- finite positive `dt`.

Construct:

```python
runtime = {
    "dt": float(dt),
}
```

If the policy consumes `point_cloud`:

- require root `pointcloud_config` mapping,
- require positive integer `num_points`,
- verify the stored point-cloud tensor N agrees with `pointcloud_config["num_points"]`,
- preserve the existing Agent `num_points` / encoder `num_points` consistency checks where those fields are explicitly configured,
- save the full point-cloud numerical dict:

```python
runtime["pointcloud"] = pointcloud_config
```

If the policy consumes `fingertip_points`:

- require root `fingertip_link_names`,
- require exactly five distinct non-empty strings,
- save them:

```python
runtime["fingertip_link_names"] = [...]
```

Do not save modality semantic dictionaries.

For MultiTask datasets that do not expose one single `replay_buffer`, preserve current effective behavior: no direct Real runtime capture / no direct Real deployment artifact.

Do not invent a multi-task Real deployment contract in this task.

## 8.5 `dexmani_policy/training/build_utils.py`

Keep current dataset/normalizer behavior.

After dataset construction:

- call the tiny deployment metadata capture helper,
- only add `cfg.real_runtime` when a current canonical Real dataset returns non-null runtime metadata.

Do not add `real_runtime: null` for simulation.

Then build the normalizer exactly as today.

Do not modify:

- full-buffer stats,
- action normalization,
- `use_aux_ee`,
- MultiTask normalization,
- EMA/model construction.

## 8.6 `dexmani_policy/deployment/runtime.py`

Simplify `PolicyInfo`.

Replace:

```text
control_dt_s
modality_contracts
```

with explicit minimal fields:

```text
control_dt_s: float | None
pointcloud_config: dict | None
fingertip_link_names: tuple[str, ...] | None
```

`inspect_policy()` should:

- continue resolving experiment/checkpoint/EMA/NFE exactly as today,
- continue deriving observation fields from saved dataset config,
- continue deriving action mode from `action_key`,
- read `cfg.get("real_runtime")`,
- expose `dt`, optional point-cloud config, optional fingertip links,
- not validate Real hardware live support,
- not validate semantic contracts.

Real live capability belongs in `dexmani_real`.

Keep `LoadedPolicy` behavior unchanged.

## 8.7 `dexmani_policy/common/inference.py`

Make `load_experiment_config()` generic again.

Remove the requirement that every saved config contain `real_runtime`.

It should only require a valid mapping loaded from the saved resolved `config.yaml`.

Real deployment checks belong in the Real deployment path.

Do not alter strict checkpoint restoration.

## 8.8 `dexmani_policy/smoke_test.py`

Update assumptions that every config contains `real_runtime`.

For example, replace direct:

```python
saved["real_runtime"]
```

with logic compatible with simulation configs that omit the key.

Keep existing strict checkpoint/EMA smoke coverage.

If a canonical Real smoke dataset is available, continue exercising `inspect_policy/load_policy/warmup`.

---

# 9. Canonical data meaning: documentation, not runtime ABI

The following information remains important for humans and papers:

- joint-state order,
- units,
- EEF frame,
- rot6d convention,
- fingertip ordering,
- tactile sensor ordering,
- tactile axes,
- point-cloud XYZRGB convention,
- action meaning and timing alignment.

Do not maintain it as duplicated executable dictionaries across repositories.

Document it in repository documentation (existing README and/or one concise data-format section).

The source code that produces/consumes the tensors remains the executable truth.

Do not create a formal versioned specification.

---

# 10. Required documentation updates

Update relevant README / AGENTS text in both repositories so it no longer claims that:

- Policy validates complete modality semantic contracts,
- saved experiments require `modality_contracts`,
- deployment replays semantic contracts.

Document the new rule accurately:

### dexmani_real

- Raw remains strict and immutable.
- Canonical export rejects malformed/non-finite training data.
- Canonical Zarr stores compact root numerical metadata.
- Point-cloud deployment reuses the saved numerical point-cloud config with current calibration.
- Fingertip deployment reuses the saved fingertip link selection with current hand mount.
- No legacy cache migration path is supported; regenerate processed data from Raw.

### dexmani_policy

- Dataset code is source/domain agnostic.
- Real canonical detection/capture lives only in deployment metadata plumbing.
- Saved config/checkpoint remain the experiment artifact.
- `real_runtime` exists only for Real-trained experiments and contains only numerical deployment requirements.
- Simulation experiments need not contain `real_runtime`.

---

# 11. Explicit non-goals

Do **not** add or implement:

- DatasetSchema,
- array schema framework,
- semantic-ID registry,
- representation registry,
- capability registry,
- schema versions,
- migration tools,
- compatibility negotiation,
- generic third-party Real format support,
- separate deployment export artifact,
- a new ABI,
- lazy Zarr,
- key-first-k optimization,
- new missing-data masks,
- new tactile imputation,
- MultiTask sampling refactor,
- train-only normalization,
- new action normalization semantics,
- new safety policy,
- new hardware abstraction.

If simplification can be achieved by deleting code, prefer deletion.

---

# 12. Behavioral invariants that must remain true

The refactor is accepted only if all of these remain true.

## Data/export invariants

1. Raw evidence remains immutable.
2. Existing output Zarr is never overwritten.
3. Export remains atomic.
4. Invalid/incomplete Raw episodes are rejected before publishing a canonical dataset.
5. Any non-finite floating modality required by the canonical multimodal export causes rejection.
6. Canonical array names/shapes/dtypes do not change in this task.
7. All arrays remain temporally aligned to `meta/episode_ends`.
8. One canonical store requires uniform task, control `dt`, depth scale, shape and dtype.
9. Export report remains available and gains the resolved processing provenance.

## Training invariants

1. Simulation and Real use the same `BaseDataset` path.
2. Only configured `sensor_modalities` plus action fields are loaded.
3. Normalizer still uses full ReplayBuffer statistics.
4. ReplayBuffer still prints current range diagnostics.
5. `use_aux_ee` behavior is unchanged.
6. Sequence sampling/padding is unchanged.
7. MultiTask behavior is unchanged.
8. Resume behavior is unchanged.

## Deployment invariants

1. A simulation experiment cannot accidentally execute on Real: absence of usable Real runtime metadata must fail in `dexmani_real` preflight.
2. Real deployment still rejects unsupported observation fields.
3. `joint_state` remains required by the current Real adapter.
4. Point-cloud policies require the saved training point-cloud numerical config.
5. Live point-cloud generation uses current calibration + saved numerical processing config.
6. Fingertip policies use the saved training fingertip link selection + current physical hand mount.
7. Policy control rate must not exceed worker rate.
8. Action mode and physical action dimensions remain unchanged.
9. `decode_policy_action()` safety behavior remains unchanged.
10. No hardware workers should start before compatibility/preflight succeeds.

---

# 13. Validation / test plan

Use both conda environments correctly.

## 13.1 Static/import validation

From `dexmani_real`:

```bash
conda run --no-capture-output -n real_robot   python -m compileall dexmani_real examples
```

From `../dexmani_policy`:

```bash
conda run --no-capture-output -n policy   python -m compileall dexmani_policy
```

## 13.2 Policy config validation

From `../dexmani_policy`:

```bash
conda run --no-capture-output -n policy   python dexmani_policy/smoke_test.py --config-only   dp dp3 r3d sat maniflow dqrise multitask_dit
```

All existing policy configs should remain structurally valid.

## 13.3 Policy smoke tests

If the corresponding local datasets/GPU are available, run representative full smoke tests, at minimum:

```bash
conda run --no-capture-output -n policy   python dexmani_policy/smoke_test.py dp3 r3d
```

Also run RGB `dp` when its local dataset is available.

Do not fabricate datasets merely to make the smoke test pass.

## 13.4 Canonical export validation

If local Raw episodes are available:

1. export to a **new temporary output path**,
2. never overwrite an existing canonical dataset,
3. inspect root attrs,
4. verify there are no per-array semantic-contract attrs,
5. verify root contains the required compact metadata,
6. verify `export_report.json` contains processing provenance.

Do not run any hardware-dependent collection or motion command for this validation.

## 13.5 Cross-repo Real metadata validation

Using a newly exported canonical Zarr if available, verify in the `policy` environment that:

- `BaseDataset` loads without any Real-specific dataset code,
- `cfg.real_runtime` is captured only for canonical Real,
- RGB-only/joint-only Real policies save only `dt`,
- point-cloud policies additionally save `pointcloud`,
- fingertip policies additionally save `fingertip_link_names`,
- simulation configs do not gain `real_runtime: null`.

Then, in `real_robot`, verify pure preflight/config code can consume the resulting `PolicyInfo` without starting robot workers.

## 13.6 Search-based cleanup validation

At completion, run searches to confirm the removed concepts are gone from runtime code:

```bash
grep -R "modality_contracts\|CANONICAL_MODALITY_SEMANTICS\|validate_modality_contract\|real_policy_contract"   dexmani_real ../dexmani_policy
```

Expected: no active runtime-code references. Documentation/history wording should also be updated.

Do not blindly delete unrelated text matches without understanding them.

---

# 14. Implementation order

Use this order to keep the cross-repo change understandable.

## Phase A — simplify producer metadata in dexmani_real

1. Remove semantic tables/functions from `dataset/contracts.py`.
2. Write compact root attrs in `dataset/export.py`.
3. Preserve strict export validation.
4. Add processing provenance to `export_report.json`.
5. Keep all array layouts unchanged.

## Phase B — make dexmani_policy datasets generic

1. Remove Real imports/state from `BaseDataset`.
2. Delete `datasets/real_policy_contract.py`.
3. Add tiny metadata capture helper to deployment runtime.
4. Update `build_dataset_and_normalizer()`.
5. Make common inference independent of Real.
6. Simplify `PolicyInfo/inspect_policy`.
7. Update smoke-test assumptions.

## Phase C — simplify dexmani_real deployment consumer

1. Remove semantic-contract comparisons.
2. Consume `pointcloud_config` and `fingertip_link_names` directly.
3. Keep live capability/rate/safety checks.
4. Update fingertip runtime construction.
5. Leave worker/process/safety logic otherwise unchanged.

## Phase D — docs and validation

1. Update README/AGENTS in both repos.
2. Run compile/config checks.
3. Run available smoke/export checks.
4. Inspect final diffs in both repositories.
5. Confirm no unrelated changes were introduced.

---

# 15. Expected cleanup result

The final architecture should be easy to explain in a paper-code README:

```text
dexmani_real
Raw → validate/process → multimodal training Zarr

dexmani_policy
Zarr → ReplayBuffer → SequenceSampler → Dataset → Normalizer → Agent

training artifact
resolved config + normal checkpoint
+ minimal Real numerical metadata when applicable

dexmani_real deployment
live sensors → build_policy_observation()
             → LoadedPolicy
             → decode_policy_action()
             → runtime safety → robot
```

No semantic-contract layer should sit between these steps.

---

# 16. Completion criteria

The task is complete when:

- [ ] `dexmani_policy/datasets/real_policy_contract.py` is deleted.
- [ ] `BaseDataset` contains no Real-specific contract logic.
- [ ] `CANONICAL_MODALITY_SEMANTICS` is removed from active code.
- [ ] canonical arrays no longer carry duplicated semantic-contract attrs.
- [ ] canonical root metadata contains the compact numerical facts required above.
- [ ] canonical export still rejects non-finite contact/tactile and other training modalities.
- [ ] `real_runtime.modality_contracts` no longer exists.
- [ ] Real runtime metadata is limited to `dt`, optional point-cloud numerical config, and optional fingertip link selection.
- [ ] simulation saved configs no longer require `real_runtime: null`.
- [ ] `common.inference` is not Real-specific.
- [ ] Real deployment still rejects unsupported modalities and unsafe/incompatible runtime rates.
- [ ] point-cloud live preprocessing still uses the training numerical config.
- [ ] fingertip live geometry still uses the training link selection when that modality is consumed.
- [ ] normalizer semantics are unchanged.
- [ ] `use_aux_ee` semantics are unchanged.
- [ ] MultiTask sampling is unchanged.
- [ ] ReplayBuffer min/max logging is unchanged.
- [ ] SequenceSampler/resume/checkpoint/safety behavior is unchanged.
- [ ] documentation matches the simplified architecture.
- [ ] both repositories compile and config validation passes in their respective conda environments.

The implementation should end here. Do not use this task as an opportunity for additional architecture redesign.
