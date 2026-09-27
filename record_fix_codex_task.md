# Codex Task: Simplify Recording and Make Research Data Contracts Capability-Based

## Status and intent

This task is an **approved architectural refactor** for `dexmani_real`, coordinated with the real-data consumer path in `dexmani_policy`.

The repository is a **personal PhD research codebase**, not a production data platform. Optimize for:

1. physical safety,
2. experiment/data correctness,
3. real-time control integrity,
4. simple research iteration,
5. readability,
6. only then generic extensibility.

The design direction is deliberately aligned with the useful parts of:

- official Diffusion Policy: simple episode lifecycle, replay-buffer / `episode_ends`, fixed-dt policy data;
- ManiUniCon: raw-vs-processed separation and field-driven processing;
- LeRobot: feature/self-describing metadata and threaded streaming encoding;

while **not** copying their unnecessary constraints:

- do not resample independently recorded streams like Diffusion Policy / ManiUniCon when DexMani already has one causal control-row snapshot;
- do not use LeRobot as the internal source-of-truth format;
- do not inherit LeRobot's public-ecosystem `CODEBASE_VERSION` / migration burden;
- do not allow recording queues to block the control loop or silently drop frames.

### Reference implementations inspected

Use these as behavioral references only; do not vendor their implementations.

- `real-stanford/diffusion_policy`
  - `diffusion_policy/real_world/real_env.py`
  - `diffusion_policy/common/timestamp_accumulator.py`
  - `diffusion_policy/common/replay_buffer.py`
- `Universal-Control/ManiUniCon`
  - `maniunicon/utils/timestamp_accumulator.py`
  - `maniunicon/core/robot.py`
  - `tools/process_demo_data.py`
- `huggingface/lerobot`
  - `src/lerobot/datasets/dataset_writer.py`
  - `src/lerobot/datasets/video_utils.py`
  - `src/lerobot/datasets/feature_utils.py`
  - `src/lerobot/datasets/dataset_metadata.py`

The reviewed DexMani baseline is:

- `dexmani_real@cb2b5f7382c1c9d577385dc88a16be2fd0753a0c`
- `dexmani_policy@0ee770fb5bc659196449dd06f9ff3c284526db7a`

Before editing, re-read the current code and account for any newer changes.

---

## Explicit task-authorized changes to current AGENTS.md

This task intentionally supersedes only the following current standing data/recording statements. Update `AGENTS.md` and stable README documentation after the implementation is accepted.

Replace the old requirements that:

- the normal Raw reader accepts only Raw v34;
- new Raw must preserve persisted `frame_valid` and `episode_valid`;
- Canonical Policy Zarr must remain globally versioned as v15;
- every persisted semantic change requires incrementing a global schema version;
- recording must remain a dedicated runtime worker/process.

The following existing instructions remain in force and must not be weakened:

- physical safety and finite SDK inputs;
- `SafetyState` and `run_id` command fencing;
- xArm/XHand/RealSense SDK ownership in their hardware workers;
- CUDA/model ownership in the policy worker;
- sensor freshness and causal observation assembly;
- current operator-facing C pause/resume behavior unless this task explicitly requires a recording-only adjustment;
- actual published action semantics;
- XHand aggregate/dense tactile runtime validity being independent of joint-state freshness;
- current calibration ownership rules;
- no committed `tests/` directory;
- no hardware-affecting execution without explicit user authorization.

Do not use this task as justification for unrelated runtime, safety, planner, action-space, training-resume, or model changes.

---

# 1. Target architecture

The target runtime data path is:

```text
Camera / Arm / Hand / VR workers
             |
             v
      read_observation()
             |
       ObservationRow
   (owns copied RGB-D payload)
             |
      +------+------+
      |             |
      v             v
 control/policy   AsyncEpisodeRecorder
      |             |
      |        queue.put_nowait(frame)
      |             |
      v             v
 robot command   bounded local Queue
                    |
               writer thread
               +----+----+
               |         |
            data.h5   rgb.mp4

completed staging
      |
structural reopen smoke
      |
.tmp -> published Raw
      |
Raw -> Canonical exporter
      |
canonical multimodal Zarr
      |
dexmani_policy loads only requested modalities
```

The core principle is:

> **The control owner owns recording semantics, but the control thread never performs recording I/O.**

---

# 2. Phase A — replace the Recorder process with a local asynchronous sink

## 2.1 Required deletion

After the replacement is proven, delete the dedicated recording transport rather than keeping compatibility wrappers.

Delete:

- `dexmani_real/recording/client.py`
- `dexmani_real/recording/io_worker.py`
- `RecorderClient`
- `RecorderIO` / `RecorderWorkerConfig`
- `StartRecording`
- `StopRecording`
- `RecordingStarted`
- recorder-process `RecordingResult` transport if no longer needed
- `record_sample_ring`
- `record_control_q`
- `record_result_q`
- `recorder_ready`
- `record_sample_ring_maxlen`
- `make_record_sample_dtype()`
- recorder process construction/readiness/supervision/cleanup wiring
- sequence-fence machinery that exists only for the cross-process recorder:
  - `start_sequence`
  - `through_sequence`
  - `last_sample_sequence`
  - recorder-side sequence draining
  - `pending_stop`
  - recorder transport polling/join protocol

Also remove stale comments/config/docs that describe RecorderIO or the second recording SHM path.

Do not leave dead compatibility shims for this internal transport.

## 2.2 Replacement API

Prefer a direct class in the existing recording package, with no new framework layer. It may live in `recording/recorder.py` unless splitting one small private writer helper materially improves readability.

The public API should be approximately:

```python
start_episode(...)
add_frame(frame, step_timestamp_ns=...)
mark_discard(reason)        # teleop-only local lifecycle helper if needed
save_episode(reason=...)
discard_episode(reason=...)
close()
```

Do not recreate `RecorderClient` under a new name.

The recorder owns at most one active episode.

## 2.3 Start and stop are allowed to block

Preserve the current safe lifecycle ordering.

### START

For both teleop and policy evaluation:

1. validate/snapshot required recording metadata;
2. create the staging directory;
3. initialize writer resources;
4. start the writer thread and prove it is ready;
5. return from `start_episode()`;
6. only then reacquire a fresh observation / reset references as currently required;
7. only then grant motion authority.

The existing design already intentionally allows recorder START to block before motion. Preserve that property.

### STOP / SAVE / DISCARD

Motion authority must be revoked before any potentially blocking finalization.

Required order:

1. revoke/end motion authority;
2. stop accepting new recorder frames;
3. enqueue a writer sentinel;
4. drain queued frames in FIFO order;
5. flush/close video and HDF5;
6. perform the closed-file structural reopen smoke;
7. publish the completed staging directory or remove it;
8. return to the caller.

Do not join a writer while the robot still has RUNNING authority.

## 2.4 Producer hot path is a hard real-time-adjacent contract

`add_frame()` must remain extremely small.

Allowed on the producer/control thread:

- check stored writer failure;
- optionally perform one lightweight cadence check;
- `queue.put_nowait(frame)`;
- increment simple counters.

Forbidden on the producer/control thread:

- PyAV encode/mux;
- HDF5 access;
- compression;
- file-system reads/writes/stat;
- JSON/YAML parsing;
- calibration lookup;
- queue operations that can block;
- retry/sleep loops;
- copies of the full RGB-D payload.

Do not use `queue.put(..., timeout=...)`.

Use a bounded local `queue.Queue` with a default capacity of 16 frames, matching the current one-second-ish backlog budget at the default 16 Hz teleop rate.

If the queue is full:

```text
raise RecordingBackpressureError
```

Do not:

- block;
- drop the oldest frame;
- drop the newest frame;
- silently continue;
- increase the queue without evidence.

A full queue means the recording sink cannot sustain the configured experiment rate.

## 2.5 Do not copy RGB-D again

Verify the ownership path before implementation.

Current `CameraRingBuffer.read_latest()` / exact-sequence reads return ownership copies. `ObservationRow` freezes those arrays but does not expose mutable SHM views.

Therefore the recorder queue should pass the existing `EpisodeFrame` / RGB-D array references directly to the writer thread.

Do **not** copy the ~1.5 MiB RGB-D payload again merely for thread safety unless the ownership assumptions in the current code have changed.

This is an intentional difference from LeRobot's generic streaming encoder, which copies camera frames because arbitrary camera drivers may reuse buffers.

## 2.6 Writer thread

Use one writer thread per active recorder/episode unless measurements prove a need for more.

The writer thread owns all file handles.

It may:

- encode RGB with the existing PyAV/H.264 path;
- batch low-dimensional rows;
- batch depth writes;
- write HDF5;
- record its first exception for propagation to the control owner.

Retain or improve batching rather than issuing many tiny HDF5 calls.

Suggested policy:

- RGB: stream frame-by-frame;
- low-dimensional numeric rows: append in small batches (e.g. existing 32-row scale);
- depth: append in small batches if the storage writer supports it cleanly.

Do not add a thread pool or per-modality worker graph.

### Compression

Avoid spending CPU on tiny low-dimensional telemetry.

It is acceptable, and preferable if it simplifies/accelerates the local writer, to use:

- no compression for small low-dimensional Raw arrays;
- lightweight compression for depth (current gzip level 1 or a simpler supported equivalent);
- current fast H.264 settings for RGB.

Do not change visual/depth semantics in this task.

## 2.7 Writer failure

The local recorder needs only a local error slot, e.g.:

```python
_error: BaseException | None
```

The writer thread stores the first failure.

The runner checks it:

- at `add_frame()`;
- and at a natural point in each active control loop.

A required recording failure remains an experiment failure and must flow through the existing safety/lifecycle path.

Do not add a new shared Event, result queue, recorder health process, ACK protocol, or command identity.

## 2.8 Preserve policy startup ordering

Do not regress the current deployment property that model restore / CUDA initialization / warmup completes before hardware workers are unnecessarily engaged.

Constructing a recorder object must be cheap and must not open camera/HDF5/video resources.

Episode resources are created only on actual `start_episode()`.

---

# 3. Phase B — simplify Raw semantics without losing policy-evaluation evidence

## 3.1 Remove persisted validity fields

For newly recorded Raw episodes, remove:

- `frame_valid`
- `episode_valid`

and the cross-process/lifecycle machinery whose only purpose was maintaining those persisted fields.

Do not replace them with generic persisted validity masks.

## 3.2 Action is the row-level publication fact

For new Raw:

- finite action targets mean that target was actually published for that row;
- NaN action targets mean no corresponding action was published.

Never substitute the previous target for an unpublished action.

This keeps the existing core action truth while deleting the duplicated `frame_valid` signal.

If the safety/runtime path actually publishes a hold target, record that actual published target.

## 3.3 Teleop and policy-rollout publication rules are intentionally different

### Teleop demonstrations

Teleop Raw is training evidence.

Only clean demonstrations may be published.

A teleop capture becomes non-publishable when its fixed-dt/control-row semantics are broken, including at minimum:

- control/retarget/IK failure that prevents the required action publication;
- publication rejection;
- recording backpressure or writer failure;
- cadence discontinuity;
- any existing lifecycle condition that currently makes the episode invalid for training.

Do not persist an invalid teleop episode merely for audit.

A **minimal transient local discard reason** is allowed inside the recorder/teleop owner to preserve the existing C pause/resume control behavior without reintroducing persisted validity state or a cross-process protocol.

Important:

- C must remain an operator-facing control pause/resume feature unless a separate UX change is approved.
- If C or another lifecycle event makes the current dataset episode unusable, mark the current staging capture for discard locally.
- Stop accepting/writing additional useless recording rows when practical, but do not let recorder bookkeeping alter physical safety or re-anchoring behavior.
- Do not invent automatic hidden episode segmentation.

The final published teleop dataset directory must contain only clean demonstrations.

### Policy evaluation rollouts

Policy rollouts are evaluation evidence, not training demonstrations.

Ordinary policy failures such as an EEF target with no IK solution must remain observable in the saved rollout. A row may therefore contain:

- a valid observation;
- NaN published-action fields because no action was actually published.

Do not discard an entire policy rollout just because the policy produced an unexecutable step.

Likewise, abnormal rollout termination may still save the successfully recorded prefix, with an explicit termination reason, if the recording subsystem itself remained healthy.

This avoids selection bias that would otherwise delete precisely the policy failures evaluation is meant to capture.

## 3.4 Persist final episode facts, not validity state

Persist:

```text
termination_reason
```

as a simple final metadata value where an episode is published.

Do not recreate a multi-stage episode-status state machine.

The runtime already knows why an episode ended; store that final fact.

## 3.5 Cadence guard

Do not copy Diffusion Policy / ManiUniCon timestamp resampling into Raw.

DexMani already assembles one current causal control-row snapshot and reuses it for control + recording.

For clean teleop demonstrations, retain a minimal fixed-dt continuity guard using one monotonic step/publication clock.

A sufficient form is:

```text
0 < t[i] - t[i-1] <= 2 * dt
```

or the existing justified equivalent.

On violation, the teleop capture becomes non-publishable.

Do not maintain separate observation/publication clocks unless actual semantics require both.

Policy rollout timing remains evaluation telemetry; do not reject a rollout merely because inference was slow.

---

# 4. Phase B2 — Raw layout and forward-compatible reader

## 4.1 New Raw format identity, no new global version chain

New Raw episodes should identify themselves with a stable format identity such as:

```text
format = "dexmani.raw"
```

Do not introduce:

```text
Raw v35 / v36 / v37 ...
```

The rule is:

> A published Raw field name has immutable meaning.

If a future representation has different semantics, create a new descriptive field instead of silently reinterpreting an existing field.

Examples:

- keep native tactile representation stable;
- a future calibrated SI tactile representation gets a different field/semantic name.

A radically different top-level storage protocol may use a different `format` identity; that is not a continuous integer migration chain.

## 4.2 Raw is immutable and is never migrated in place

Formal project rule:

> **Raw episodes are immutable experimental evidence. Never rewrite or migrate Raw episodes in place.**

Old Raw v34 remains untouched.

## 4.3 Reader is capability-based, writer is strict

Separate current-writer validation from reader compatibility.

### Current writer

The current writer remains strict about exactly the fields it promises to emit. Typos or missing current fields must fail immediately.

### Reader

The reader must be forward tolerant:

- validate known fields that are present;
- do not reject an episode merely because it contains an additional unknown future field;
- expose capability checks / `require_fields(...)` to consumers;
- consumers validate only fields they actually need.

Do not make the generic Raw reader decide training eligibility.

## 4.4 Legacy Raw v34 adapter

Do not migrate old v34 episodes.

Add one thin read adapter:

- recognize legacy `schema_version == 34`;
- read the existing `data.h5 + depth.h5 + rgb.mp4` layout;
- expose the same logical field access to downstream code;
- preserve legacy `frame_valid/episode_valid` only as legacy admission evidence where needed.

Do not create a v34 -> new-Raw conversion script.

## 4.5 Merge depth into data.h5 for new Raw if this still deletes code cleanly

For newly recorded Raw, prefer:

```text
episode/
  data.h5
    meta/...
    arm_qpos
    ...
    depth
  rgb.mp4
```

instead of:

```text
data.h5
depth.h5
rgb.mp4
```

This should delete the artificial `MergedH5File` / sidecar abstraction and give the writer one HDF5 owner.

Keep the legacy reader fallback for old `depth.h5`.

Do not perform this merge if the current storage library/runtime makes it less reliable; the primary goal remains recorder simplification and real-time integrity.

---

# 5. Phase C — generic, versionless Canonical Zarr

## 5.1 Canonical is a multimodal research cache, not a policy-specific format

Keep a stable multimodal superset suitable for future dexterous policies.

The canonical arrays should continue to cover learning-relevant stable modalities such as:

```text
joint_state
arm_qvel
arm_effort
hand_current

eef_pose
fingertip_points

contact_force
tactile_force

rgb
depth
point_cloud

action
action_ee
```

Do not shrink canonical data to only the modalities consumed by today's DP/DP3/R3D/etc.

The existing `dexmani_policy.BaseDataset` already loads only:

```text
sensor_modalities + action_key + optional action_ee
```

so a multimodal superset does not force all modalities into each policy's RAM.

## 5.2 Rename policy-specific dataset concepts

Where practical and not excessively disruptive, rename policy-specific canonical helpers to reflect their actual role, e.g.:

- `policy_array_specs` -> `canonical_array_specs`
- `policy_semantics` -> remove/restructure as modality contracts
- `POLICY_ZARR_*` -> canonical format naming

Avoid a gratuitous public-API rename cascade outside this data path.

## 5.3 No global Canonical schema version

New canonical stores should use a stable identity such as:

```text
format = "dexmani.real.canonical"
```

with root metadata limited to true dataset-wide facts, e.g.:

```text
format
task_name
dt
```

Do not add `schema_version = 16/17/18`.

Canonical Zarr is a derived artifact:

> **Regenerate it from Raw when the derived representation/recipe changes; never migrate it in place.**

## 5.4 Per-modality self-description

Replace the current flat root-level `policy_semantics()` attrs with attrs attached to each array.

Do not create one giant new framework/manifest unless Zarr constraints force it.

Each modality should carry only the semantics that cannot be inferred from the array itself.

Examples:

### `data/joint_state`

```text
semantic_id = "dexmani.joint_state"
unit = "rad"
components / ordering
```

### `data/fingertip_points`

```text
semantic_id = "dexmani.fingertip_points.xarm_base"
frame = "xarm_base"
unit = "m"
finger_order = [...]
recipe = {
  kinematic_model,
  fingertip_link_names,
  mount_source: "raw_episode"
}
```

### `data/tactile_force`

```text
semantic_id = "dexmani.xhand.tactile_dense.native"
frame
unit
finger_order
point_order
axis_order
missing = "nan"
```

### `data/point_cloud`

```text
semantic_id = "dexmani.point_cloud.xyzrgb.xarm_base"
frame = "xarm_base"
features = ["x","y","z","r","g","b"]
xyz_unit = "m"
rgb_range = [0, 1]
recipe = { resolved PointCloudConfig numerical recipe }
```

### Actions

Keep explicit stable semantics, e.g.:

```text
dexmani.action.joint_absolute_published
dexmani.action.ee_target
```

Do not duplicate array `shape` or `dtype` in attrs; the array already owns those facts.

Use normal JSON-serializable Zarr attrs rather than JSON strings when supported cleanly by the currently pinned Zarr version.

## 5.5 Semantic IDs are identities, not versions

Do not use `semantic_id = "...v2"` as a substitute for global schema versions.

A semantic ID names a representation.

If the meaning truly changes, use a new descriptive representation/field.

If only an algorithmic recipe changes while the representation remains the same, keep the semantic ID and persist the actual recipe.

## 5.6 Derived recipes

Only derived modalities need algorithmic recipes.

Required:

- point cloud recipe;
- fingertip FK representation recipe.

Do not create recipe machinery for direct/raw telemetry unless there is an actual transform to reproduce.

### Point cloud

Persist the resolved training-time point-cloud numerical recipe with the point-cloud array.

Also preserve enough export provenance to reproduce the derived array, including the audited table plane used by the offline derivation when table removal is enabled.

However:

- current deployment calibration/table state remains Real-owned;
- do not deploy using a historical training table plane merely because it was persisted;
- the saved policy runtime contract should extract only the numerical representation recipe that must match training.

### Fingertip points

Persist representation choices such as:

- kinematic model identity;
- fingertip link names/order;
- mount source = Raw episode metadata.

Do not persist a workstation-specific absolute URDF path as the representation identity.

---

# 6. Canonical missing-data policy

Current Raw intentionally stores NaN for unavailable tactile samples. Canonical export must not reject an entire otherwise usable teleop episode merely because an optional research modality contains NaN.

Define two classes:

### Core / required-to-derive canonical data

Must be finite for a training-admitted teleop episode:

- required arm/hand state for the current dexterous embodiment;
- published action targets used to construct `action` / `action_ee`;
- inputs required to derive EEF/fingertip/point-cloud outputs;
- derived EEF/fingertip/point-cloud outputs themselves.

### Optional telemetry

May contain NaN as an explicit missing-sample marker:

- tactile/contact;
- current/effort/other telemetry where the runtime can legitimately have missing measurements.

Do not add persisted generic validity masks merely to mirror NaN.

When a future policy **selects** an optional modality such as `tactile_force`, its training dataset initialization must fail clearly if that selected modality contains unsupported NaN and the policy has no explicit mask/missing-data mechanism.

This is consumer-specific capability validation.

---

# 7. Simplify the exporter

After Raw publication is clean and the new reader exists, simplify Raw -> Canonical export.

Desired flow:

```text
discover selected episodes
-> validate required teleop capabilities
-> establish uniform task/dt/static modality representation
-> transform
-> append
-> episode_ends
-> atomic staging rename
```

Prefer fail-fast for unexpected structural/semantic failures.

Keep only a very small non-destructive episode exclusion mechanism if manual demonstration curation is still required, e.g. a simple exclude list.

Remove:

- per-episode task-name rewriting;
- batch task override complexity;
- rejected-and-continue array rollback machinery when clean Raw + explicit exclusion makes it unnecessary;
- `--dry-run` if real export into owned staging already provides the same validation safely.

Do not silently repair, resample, split, drop, or salvage bad teleop rows.

---

# 8. Downstream dexmani_policy contract

This phase is required before declaring the new canonical format complete.

If `dexmani_policy` is available in the workspace, update it in the coordinated change. Otherwise, do not silently break it: stop Phase C at a clear compatibility boundary and document the exact companion patch.

## 8.1 BaseDataset / ReplayBuffer behavior

Preserve the good current behavior:

```python
load_keys = sensor_modalities + [action_key]
if use_aux_ee:
    load action_ee
```

Do not make BaseDataset load the whole multimodal superset.

Do not rewrite ReplayBuffer to lazy loading as part of this task.

## 8.2 Replace global version checks with requested-capability checks

Current `real_policy_contract.py` must stop requiring:

```text
schema_version == 15
```

For new canonical stores:

1. require `format == "dexmani.real.canonical"`;
2. validate `task_name`;
3. validate finite positive `dt`;
4. determine required modalities from:
   - `sensor_modalities`;
   - `action_key`;
   - optional aux action requirements;
5. for each required modality:
   - require the array;
   - validate shape/dtype expected by the actual policy path;
   - validate the expected `semantic_id`;
   - validate selected modality finiteness where the policy does not support missing data;
   - extract only runtime recipes actually required for deployment.

Error messages should identify the missing/incompatible capability, e.g.:

```text
Required modality 'tactile_force' is missing.
```

not:

```text
Expected Zarr v18, got v16.
```

## 8.3 Generalize PolicyInfo runtime contracts

Avoid growing:

```text
pointcloud_config
fingertip_config
tactile_config
...
```

Prefer one compact mapping such as:

```python
modality_contracts: dict[str, dict]
```

or an equally small typed equivalent if the existing code clearly benefits.

Point-cloud deployment reads its numerical recipe from:

```text
modality_contracts["point_cloud"]
```

Fingertip deployment reads its representation choices from:

```text
modality_contracts["fingertip_points"]
```

Tactile/contact direct modalities normally need semantic/shape validation but no numerical preprocessing recipe.

Do not turn this into a generic plugin framework.

## 8.4 Fingertip train/deploy consistency

Current real deployment builds fingertip FK using the current runtime's configured fingertip link names.

For a policy trained on canonical `fingertip_points`, the representation definition used at training must be restored from the saved experiment runtime contract.

Use:

- training-time representation choices from the saved modality contract;
- current physical hand mount/calibration from the current Real runtime.

This mirrors the existing point-cloud principle:

> training representation recipe is checkpoint/dataset-owned; current physical calibration is Real-owned.

---

# 9. Legacy Canonical Zarr v15

Do not migrate v15 stores in place.

New canonical stores use `format = "dexmani.real.canonical"`.

For existing v15 stores, either:

- regenerate from Raw when needed; or
- keep one thin legacy read branch in `dexmani_policy` while old experiments still need it.

Do not create:

```text
migrate_zarr_v15_to_v16.py
migrate_zarr_v16_to_v17.py
...
```

Canonical Zarr is reproducible derived materialization, not immutable evidence.

---

# 10. Explicit evolution policy

Document and enforce these rules:

1. **Raw episodes are immutable and never migrated in place.**
2. **Canonical datasets are derived artifacts and are regenerated, never migrated in place.**
3. **New Raw/Canonical formats do not use a monotonically increasing global schema version.**
4. **Published field names and semantic IDs never silently change meaning.**
5. **New fields/modalities are additive and do not invalidate old consumers that do not request them.**
6. **Unknown additional fields do not invalidate an otherwise usable dataset.**
7. **Consumers validate only the capabilities they require.**
8. **A truly different representation gets a new descriptive field/semantic identity.**
9. **Derived modalities persist the actual representation recipe needed to reproduce or deploy them.**
10. **No migration script may fabricate a modality that was never physically recorded.**

A content fingerprint may be added later for provenance, but do not introduce it unless there is a current consumer. It must never become another hidden global-version equality check.

---

# 11. Preserve these non-goals

Do not change as part of this task:

- robot SDK worker ownership;
- motion safety architecture;
- `SafetyState` model;
- `run_id` fencing;
- command mailbox semantics;
- homing/planned-home behavior;
- point-cloud algorithm mathematics;
- tactile bias algorithm;
- policy action-space definitions;
- policy architecture/training behavior;
- checkpoint resume/RNG/EMA behavior;
- simulation datasets;
- replay safety logic except the minimum needed to consume the new Raw reader;
- current calibration procedures;
- public LeRobot integration beyond an optional later converter.

Do not add a production-style event bus, transaction service, status ledger, migration framework, schema registry service, or generic dataset plugin system.

---

# 12. Implementation sequence and gates

Do not perform this as one unreviewable rewrite.

## Milestone 1 — recording transport only

Goal:

- local asynchronous writer;
- real-time producer path;
- remove recorder process plumbing.

Keep Raw file/field semantics as close to current behavior as practical during this milestone so timing regressions can be isolated.

Required offline checks plus manual real-time validation plan.

Only after this passes, proceed.

## Milestone 2 — Raw semantic cleanup

Goal:

- remove persisted validity fields for new Raw;
- finite/NaN published-action semantics;
- teleop clean publication vs policy-rollout evidence;
- `termination_reason`;
- new `format = "dexmani.raw"`;
- capability-based reader;
- legacy v34 read adapter;
- optionally merge depth into `data.h5`.

## Milestone 3 — Canonical contract

Goal:

- generic multimodal canonical format;
- per-array semantic attrs / recipes;
- no global canonical version;
- exporter simplification;
- legacy v15 handling;
- coordinated `dexmani_policy` capability extraction.

Do not proceed to the next milestone while the current milestone has unresolved correctness or timing regressions.

---

# 13. Validation

## 13.1 Offline checks

Follow repository policy: do not add a committed `tests/` directory.

Use focused temporary/offline smoke checks for changed pure logic.

At minimum verify:

### Recorder

- FIFO ordering;
- bounded queue;
- Queue Full raises immediately;
- writer exception propagates;
- start/save/discard lifecycle;
- no publish of incomplete staging;
- writer drains all accepted rows before save;
- discard removes owned staging;
- structural reopen after close;
- no additional full RGB-D copy in the producer path.

### Raw

- new-format reader;
- legacy v34 reader;
- new `data.h5/depth` path if implemented;
- legacy `depth.h5` fallback;
- unknown additive field does not break reader;
- missing required capability gives a clear error;
- teleop publish rejects/discards non-clean capture;
- policy rollout may retain NaN-action failure rows;
- termination reason roundtrip.

### Canonical

- multimodal arrays keep correct shape/dtype;
- optional tactile/contact NaN is preserved;
- selected policy modality with unsupported NaN fails clearly downstream;
- per-array semantic attrs roundtrip;
- point-cloud recipe roundtrip;
- fingertip recipe roundtrip;
- `episode_ends` correctness;
- exporter fails atomically without a partial published Zarr.

Run:

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check --select F401,F821,F822,F823,I dexmani_real examples
git diff --check
```

If `dexmani_policy` is changed, run its equivalent existing offline smoke/compile/lint path without installing/upgrading dependencies.

## 13.2 Real-time acceptance gate

Do **not** execute hardware tests automatically. Provide the user with the exact manual validation procedure after the offline implementation passes.

Compare recording OFF vs recording ON.

### Teleop

Measure control-step interval distribution and failures.

### Policy evaluation

Use the existing:

- `RolloutStats.inference_ms`
- `RolloutStats.action_step_intervals_ms`
- effective action-step Hz

Add only lightweight recorder diagnostics needed for the validation, e.g.:

- submit duration;
- peak queue occupancy;
- backpressure count.

Acceptance conditions:

- no Queue Full in representative recording;
- no writer failure;
- no recorder-induced control-cycle miss;
- `add_frame()` remains far below the control period with no long blocking tail;
- recording ON does not materially degrade policy inference/action-step timing relative to recording OFF.

Do not keep verbose benchmark-only instrumentation permanently unless it is useful experiment telemetry.

---

# 14. Final review checklist

Before considering the task complete, explicitly verify all of the following.

### Recording

- [ ] no dedicated recorder process remains;
- [ ] no record sample SHM ring remains;
- [ ] no recorder control/result queues remain;
- [ ] no recorder readiness Event remains;
- [ ] producer path does no I/O/compression/blocking;
- [ ] writer is bounded and single-owner;
- [ ] start occurs before motion;
- [ ] finalization occurs after motion authority is revoked;
- [ ] queue backpressure is a hard recording failure, never a silent drop.

### Raw

- [ ] new Raw has no global incremental schema version;
- [ ] old v34 data is read without migration;
- [ ] new Raw does not persist `frame_valid` / `episode_valid`;
- [ ] actual published action is still the only stored action truth;
- [ ] policy-rollout failure evidence is not selected away;
- [ ] teleop published datasets remain clean training demonstrations;
- [ ] tactile missingness remains distinguishable;
- [ ] calibration and hand mount remain preserved.

### Canonical

- [ ] remains a general multimodal dexterous-learning cache;
- [ ] no global incremental canonical schema version;
- [ ] root attrs contain only dataset-wide facts;
- [ ] modality semantics live with the modality;
- [ ] derived recipes live with the derived modality;
- [ ] future tactile/fingertip/proprioception policies can consume existing canonical fields;
- [ ] current policies still load only requested arrays.

### Evolution

- [ ] no Raw migration script added;
- [ ] no Canonical migration chain added;
- [ ] unknown additive fields are forward compatible;
- [ ] old consumers fail only on missing/incompatible required capabilities;
- [ ] semantic changes use a new descriptive representation instead of reinterpretation.

### Scope

- [ ] no unrelated safety/model/planning refactor;
- [ ] no committed tests directory;
- [ ] README and AGENTS.md updated only after behavior is implemented and verified;
- [ ] final diff contains no obsolete RecorderIO/version/validity documentation.

---

# 15. Expected outcome

The finished system should be easier to reason about than the current implementation:

```text
one causal control row
    -> non-blocking local recorder submission
    -> one asynchronous writer
    -> immutable Raw evidence
    -> reproducible generic Canonical Zarr
    -> policy selects required capabilities
```

The refactor succeeds only if it simultaneously achieves:

- less code and fewer runtime resources;
- no regression in teleop/policy-eval real-time behavior;
- stronger, simpler data semantics;
- no recurring schema-migration burden;
- continued support for future dexterous modalities such as tactile, contact, fingertip geometry, proprioception, RGB-D, and point clouds.

Prefer deletion and direct contracts over replacement frameworks.
