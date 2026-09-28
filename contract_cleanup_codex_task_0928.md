# DexMani Real Contract / Runtime Cleanup Task

## 0. Scope

Primary repository:

```text
~/Desktop/dexmani_real
```

Sibling repository used only to verify the existing Real ↔ Policy numerical interface:

```text
~/Desktop/dexmani_policy
```

This task file belongs at:

```text
~/Desktop/dexmani_real/contract_cleanup_codex_task_0928.md
```

Reviewed remote heads when this task was finalized:

```text
dexmani_real   2082782068fc280b7b5d894c7f78a97268dc006d
dexmani_policy 141b0337c148d4f2dd09ade11f4db660a8ab9c4d
```

Inspect local HEADs first. Local code is authoritative if newer. Do not apply patches by stale line number.

This is a personal PhD real-robot dexterous-manipulation repository. Optimize in this order:

1. hardware safety;
2. scientific correctness;
3. data traceability / reproducibility;
4. research iteration speed;
5. code simplicity;
6. generic extensibility.

The purpose of this task is **not** to build a generic robotics framework. It is to remove duplicated or hidden experiment semantics while preserving strict hardware boundaries and the user's intentional data-selection rules.

The central architectural goal is:

> **Source-specific code may decide what action is intended; only one source-agnostic Real path may decide how that intent becomes a physical target.**

---

# 1. Explicit decisions that must not be changed

Two previously reviewed behaviors are intentional research choices.

## 1.1 Teleop technical failure may discard the whole capture

Keep the current policy that a teleop capture marked discard-only by control/publication/cadence failure is not published as Raw.

Do not reinterpret this task as a request to preserve every technical-failure prefix.

Do not change:

- `mark_discard(...)` semantics merely to preserve failure evidence;
- explicit DISCARD behavior;
- current teleop whole-capture discard policy.

## 1.2 Canonical admission remains strict whole-episode full-modality admission

Keep the current choice that canonical training data accepts only complete teleop demonstrations whose required current Raw numeric fields are finite for the whole episode.

In particular, do **not** make canonical admission model-modality-aware in this task.

Do not weaken:

- whole-episode finite validation;
- rejection caused by invalid tactile/contact values;
- `collection_source == "teleop"` as the current canonical training-source policy;
- whole-episode rejection rather than silent frame deletion/interpolation.

The cleanup may make this ownership and naming clearer, but behavior stays strict.

---

# 2. Non-negotiable design rules

## 2.1 Human / Policy source code owns intent, not physical realization

Source-specific layers may own:

```text
Teleop:
    VR calibration / coordinate mapping
    reset-relative mapping
    explicit teleop EMA
    hand retargeting

Policy:
    tensor output interpretation
    joint-vs-EEF physical action mode
    action-chunk extraction
```

They must not independently own:

```text
workspace projection
hand operational projection
joint periodic-equivalence projection
EEF -> joint online IK behavior
source-specific IK fallback policy
physical-target diagnostics
```

Those belong to one shared online action-realization path.

## 2.2 Validation does not invent plausible values

A validator may:

```text
accept
reject
raise
return unavailable
```

A validator must not silently turn invalid data into:

```text
identity quaternion
current joint state
clipped tracking sample
zero action
"safe-looking" replacement observation
```

If smoothing, clipping or projection is intentionally part of the experiment, it must be named and owned as a transform, not hidden inside validation.

## 2.3 One semantic has one mathematical owner

The following must not have independent implementations with independent tolerances:

```text
SO(3) validity
SE(3) validity
unit-quaternion validity / normalization
```

Use one small mathematical helper surface. Do not build a frame graph or geometry framework.

## 2.4 Artifact lifecycle follows artifact meaning

Treat the three artifact classes differently:

```text
Raw         = experiment evidence selected by the current recording policy
Canonical   = rebuildable training cache
Calibration = current physical experiment state
```

Therefore:

- Raw remains create-only and unpublished staging is distinct from published Raw;
- Canonical may be explicitly rebuilt/replaced;
- Calibration is current state and may be replaced after recalibration;
- do not apply Raw-style durability/versioning ceremony to every artifact.

## 2.5 Hardware safety remains defense-in-depth

Do not remove or weaken:

- `run_id` motion epochs;
- `SafetyState` authorization;
- motion revocation before blocking shutdown/finalization;
- worker/driver finite and mechanical-limit checks;
- observation freshness;
- RGB / point-cloud source identity when requested;
- seqlock/torn-read protection;
- collision checks;
- E-stop behavior;
- homing safety;
- physical-worker failure -> FAULT;
- "never unlink shared memory while a worker may still access it".

A duplicated hardware-boundary check is not automatically overengineering.

---

# 3. Problem ledger and required disposition

Do not skip items in this table. Some entries require code changes; some require explicit preservation or clearer ownership.

| ID | Current issue | Root cause | Required disposition |
|---|---|---|---|
| P1-1 | EEF Policy deployment enables random IK fallback while Teleop does not | action source owns realization policy | remove source-specific fallback; shared realizer owns one deterministic online IK behavior |
| P1-2 | VR has hard-coded per-frame rotation clamp | validity/filtering/control transform are mixed | remove the hidden per-frame clamp; do not replace it with another hidden heuristic |
| P1-3 | `validate_episode()` sounds generic while enforcing strict teleop canonical admission | Raw validity and current canonical-policy naming are blurred | keep behavior; rename/reframe as strict canonical teleop admission |
| P1-4 | Teleop and Policy each perform workspace projection | duplicated physical action semantics | move workspace projection into one shared action realizer |
| P1-5 | Teleop and Policy each perform hand operational clipping | duplicated physical action semantics | move hand operational projection into one shared action realizer |
| P1-6 | joint and EEF realization logic is split across source-specific modules | no unique owner for physical action semantics | introduce a small source-neutral intent/realization boundary |
| P1-7 | validation helpers may silently transform invalid values | validator and policy decision are mixed | invalid math must raise/reject; explicit transforms remain explicit |
| P1-8 | `normalize_quat_wxyz()` returns identity for degenerate input | generic math helper guesses a fallback | make degenerate/nonfinite quaternion invalid; audit callers |
| P1-9 | Teleop-specific Human→Intent logic is mixed with Intent→Robot logic | controller owns too much | Teleop ends at an action intent |
| P1-10 | Policy bridge mixes tensor interpretation with robot realization | deployment adapter owns too much | Policy bridge ends at an action intent |
| P2-1 | SE(3) validation is copied across modules | no mathematical truth owner | centralize one strict helper |
| P2-2 | SE(3) tolerance has already drifted (`1e-5` vs `1e-6`) | copied logic diverged | one tolerance definition and one implementation |
| P2-3 | SO(3) validation has independent implementations | same root cause | centralize one helper |
| P2-4 | quaternion validation/normalization semantics are scattered | same root cause | centralize strict validation/normalization behavior |
| P2-5 | two `validate_task_name` functions mean different things | path safety and task identity share one vague name | keep two concepts but rename them explicitly |
| P2-6 | Canonical behaves close to an immutable artifact | Raw lifecycle leaked into cache lifecycle | add explicit rebuild/overwrite path after successful staging validation |
| P2-7 | one-task-per-store is treated like a universal contract | dataset organization and numerical semantics are conflated | keep current one-task store design; document it as organization, not generic schema |
| P2-8 | `dt` uniformity is mixed with weaker metadata checks | invariants have different scientific weight | keep `dt` strict; identify it as Policy/runtime numerical semantics |
| P2-9 | `depth_scale` uniformity has unclear status | storage metadata is confused with deployment ABI | keep current fixed-store behavior, but do not promote it to Policy compatibility metadata |
| P2-10 | canonical metadata could grow into a generic schema | cross-repo interface lacks a minimality rule | keep only current concrete deployment-relevant numerical facts; no schema framework |
| P2-11 | clipping/IK diagnostics are computed in several caller layers | transform and telemetry ownership are split | shared realizer returns physical target plus diagnostics |
| P2-12 | IK solver configuration can become source-dependent | solver profile is configured by action origin | solver knows control profile, never "teleop vs policy" |
| P3-1 | Raw publication recursively fsyncs the whole tree | atomicity and power-loss durability are conflated | keep staging/close/validate/rename; remove recursive tree durability |
| P3-2 | calibration writers create timestamped backup files | calibration is treated like a long-lived versioned ABI | remove automatic backups; Git/history owns version history |
| P3-3 | calibration JSON writes use fsync durability | same durability overreach | keep temp + replace; remove database-style fsync ceremony |
| P3-4 | table calibration performs post-write numerical readback | serialization is revalidated after already-valid in-memory state | validate before write; remove redundant readback |
| P3-5 | table calibration writes an unused `schema_version` | schema ceremony without a consumer | remove the unused field |
| P3-6 | VR calibration persists both `theta_deg` and derived matrix | two sources of truth | persist irreducible heading value; derive matrix at load |
| P3-7 | VR calibration persists both quality statistics and derived grade | derived state is serialized | persist measurements; derive grade |
| P3-8 | VR calibration uses schema/convention protocol machinery | current calibration is treated as cross-version ABI | simplify persisted current-state representation; convention lives in code |
| P3-9 | camera/table/VR calibration have different persistence rituals | no shared lifecycle principle | converge on validate -> confirm -> temp -> replace, without frameworkization |
| P3-10 | detailed shutdown report types leak into runtime control flow | diagnostics became a public lifecycle protocol | keep safety sequence; make supervisor own interpretation, keep details internal/logging |
| P3-11 | sessions reason about `report.clean` | same leakage | sessions should receive simple clean success/failure, not escalation internals |
| P3-12 | config requires readiness timeout for every possible subsystem | config describes platform superset, not active runtime graph | validate only configured/started subsystems |
| P3-13 | RuntimeChannels represents more capability than every run uses | one global storage object serves several workflows | do not redesign now; prevent further registry/capability growth |
| P3-14 | internal helpers repeat config validation after resolved config is trusted | every layer treats internal inputs as untrusted | remove only clear redundant/silent repair while touched; keep external/hardware boundaries strict |
| P3-15 | EMA helper silently clips already-validated alpha | internal repair hides propagation bugs | consume validated alpha directly; invalid alpha should fail upstream |
| P3-16 | XHand startup retries look production-like | lab hardware has real pre-run startup transients | keep; do not convert into runtime auto-recovery |
| P3-17 | shared-memory seqlock code is complex | large multi-process payloads can tear | keep unchanged |
| P3-18 | observation freshness/source identity is strict | asynchronous sensors need temporal semantics | keep unchanged |
| P3-19 | motion authority is checked repeatedly with `run_id` | stale commands can survive blocking operations | keep unchanged |
| P3-20 | worker and driver both validate hard limits | SDK boundary must not trust upstream | keep unchanged |

---

# 4. Target online action interface

Do not create a registry or generic action framework.

Use at most a few small immutable data objects. Exact names may follow local conventions.

Conceptually:

```python
@dataclass(frozen=True)
class ActionIntent:
    mode: Literal["joint", "eef"]
    arm: np.ndarray
    hand: np.ndarray | None


@dataclass(frozen=True)
class ActionRealization:
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None

    workspace_clip_m: float = 0.0
    arm_clip_rad: float = 0.0
    hand_clip_rad: float = 0.0

    ik_result: IKResult | None = None
```

A single source-neutral function/object performs:

```text
joint intent:
    finite/shape check
    nearest mechanical-equivalent representation
    operational arm projection
    hand operational projection

EEF intent:
    finite/shape check
    workspace projection
    Rot6D -> rotation/quaternion
    one deterministic online IK profile
    hand operational projection
```

It must not know whether the input came from:

```text
teleop
policy
```

The existing `RobotCommand(run_id, ...)` remains the publication object after realization.

Do not move `run_id`, SDK authority, hard mechanical limits or worker checks into this new layer.

## 4.1 What remains Teleop-specific

Teleop owns only Human→Intent:

```text
VR frame validity/freshness
VR calibration transform
relative reset anchor
position/rotation scale
explicit configured EMA
hand retargeting
-> ActionIntent
```

The existing configured total-from-reset rotation limit may remain if it is intentionally part of teleop mapping.

The hard-coded per-frame `0.52 rad` clamp must not remain hidden.

## 4.2 What remains Policy-specific

Policy deployment owns only model-output→Intent:

```text
prediction chunk shape
joint mode: 19D -> arm7 + hand12 intent
eef mode:   21D -> pose9 + hand12 intent
-> ActionIntent
```

It does not own IK, workspace projection or hand projection.

## 4.3 Paths intentionally outside this unification

Do not force unrelated operations through `ActionIntent` merely for architectural purity.

Keep intentionally separate where appropriate:

- calibration motion;
- planned homing;
- replay of recorded physical joint targets;
- keyboard jog/manual diagnostics with distinct semantics.

Only unify paths that are intended to share the same online learned/teleop action semantics.

---

# 5. Phase A — establish one action realization path

Do this before deleting old helpers.

## A1. Add the minimal intent/realization representation

Choose the smallest location consistent with current ownership, for example within `robot/` or a small existing action module.

Do not add:

```text
ActionRegistry
ActionFactory
ProducerCapabilities
plugin dispatch
source enums for teleop/policy
```

The realizer should infer behavior only from `intent.mode` and current runtime state/config.

## A2. Move common joint/hand projection into the realizer

Consolidate the current semantics from:

```text
robot/projection.py
teleop/control/controller.py
deployment/action.py
```

Preserve current operational limits and periodic-equivalence behavior.

Worker/driver mechanical limit checks remain separate and unchanged.

## A3. Move EEF workspace projection and EEF->joint IK into the realizer

Use one online IK profile for Teleop and learned EEF control.

Remove deployment-only:

```python
enable_random_fallback=True
```

Do not replace it with another source-specific toggle.

If random fallback is genuinely needed in future research, it must be an explicit shared IK-profile choice affecting every path that claims the same online EEF semantics.

## A4. Convert TeleopController into an intent producer

After Human→Intent mapping, Teleop should call the shared realizer and publish the returned physical target.

Do not change P0-1 discard behavior.

Preserve:

- recorder correspondence with the command actually published;
- previous-command continuity semantics;
- current `run_id` checks;
- current failure categories unless a rename is necessary for correctness.

## A5. Convert deployment action code into a Policy-output bridge

`deployment/action.py` should become thin:

```text
physical policy vector
-> validate shape/finite
-> ActionIntent
```

The shared realizer performs the physical conversion.

Existing rollout statistics must consume realization diagnostics rather than recomputing clip differences in `runner.py`.

## A6. Keep IK solver pure with respect to source

`planning/kinematics/ik.py` may expose search/profile configuration, but no logic may branch on action origin.

Do not add `source="policy"` or `source="teleop"`.

---

# 6. Phase B — remove hidden transforms and repair mathematical ownership

## B1. Remove the hard-coded VR per-frame rotation clamp

Delete the implicit `max_per_frame_rot_rad=0.52` stateful clamp from normal VR mapping.

Do not replace it in this task with a new guessed threshold.

Existing runtime VR freshness remains the authoritative stale-sample gate.

If a future experiment needs angular-velocity rejection, implement it explicitly from `ΔR / Δt` as sensor validity with a researched threshold; that is out of scope here.

## B2. Keep explicit Teleop transforms explicit

Configured EMA, reset-relative mapping, position/rotation scale and the existing configured total-delta limit are Teleop experiment transforms.

Do not mislabel them as generic safety validators.

## B3. Remove redundant EMA alpha repair

If resolved config already guarantees `alpha in [0, 1]`, do not silently `np.clip` it again inside the EMA helper.

The helper may assert/raise if called directly with invalid values.

## B4. Make quaternion normalization strict

Change the generic quaternion normalization helper so:

```text
non-finite quaternion -> error
near-zero norm        -> error
valid quaternion      -> normalized quaternion
```

Do not return identity for degenerate input.

Audit all callers before changing behavior. Where a caller genuinely wants identity, the caller must state that choice explicitly.

## B5. Add one small geometry-validation owner

Create one small shared module, e.g. `dexmani_real/utils/geometry.py`, containing only strict reusable mathematical validation such as:

```python
validate_rotation_matrix(...)
validate_rigid_transform(...)
normalize_unit_quaternion_wxyz(...)
```

Do not move unrelated FK/IK/Rot6D logic into it.

Use one tolerance policy, with the current strict runtime/calibration behavior as the reference. Avoid the current `1e-5` vs `1e-6` split.

## B6. Replace copied validators

Audit and replace duplicated rigid/SO(3) checks in at least:

```text
recording/recorder.py
recording/storage/reader.py
dataset/pointcloud.py
sensor/pointcloud.py
sensor/pointcloud_worker.py
calibration/camera/extrinsics.py
calibration/camera/solver.py
teleop/vr_transform.py
```

Boundary-specific exception types are fine:

```python
try:
    validate_rigid_transform(...)
except ValueError as exc:
    raise RawDataError(...) from exc
```

Do not copy the math again merely to produce a different error class.

## B7. Disambiguate task validators

Keep the two distinct concepts but name them explicitly, for example:

```text
validate_task_dir_name     # safe directory component
validate_task_identity     # research/policy task label
```

Update imports and examples.

Do not force one function to satisfy both semantics.

---

# 7. Phase C — clarify strict canonical semantics and cache lifecycle

## C1. Keep strict canonical admission behavior

Do not weaken the user's P0-2 choice.

Rename/reframe the generic function so the contract is obvious, conceptually:

```python
validate_canonical_teleop_episode(reader)
```

Its intended semantics remain:

```text
nonempty episode
valid task identity
collection_source == teleop
all current canonical Raw fields present
RGB/depth present
required mount metadata present
all floating current canonical Raw fields finite for every row
```

No modality-aware policy framework.

## C2. Keep one-task-per-store as current organization

Do not build multi-task schema support in Real.

Document/encode the current rule as:

```text
one canonical export store == one task identity
```

Future multi-task training belongs in `dexmani_policy` dataset composition.

## C3. Keep `dt` strict

`dt` is a real training/deployment numerical semantic currently captured by Policy.

Do not weaken uniform `dt` checks.

## C4. Keep current depth-scale uniformity, but do not promote it to Policy ABI

The canonical cache stores depth whose interpretation needs a depth scale.

Keep the current fixed-store rule unless a concrete experiment needs mixed scales.

Do not add depth scale to the public Policy/runtime compatibility contract merely because the exporter checks it.

## C5. Add explicit canonical rebuild/overwrite

Canonical is a rebuildable cache.

Target behavior:

```text
default:
    existing output -> refuse

explicit --overwrite / rebuild:
    build complete staging output first
    validate staging
    only then remove/replace the previous cache
```

Do not overwrite or modify Raw.

Do not build a versioned-cache migration framework.

Because Canonical is rebuildable, do not add backup/rollback machinery just to emulate a database transaction.

## C6. Preserve the current minimal Real↔Policy numerical metadata

Do not remove the currently meaningful cross-repo facts:

```text
format = dexmani.real.canonical
task_name
dt
pointcloud_config when consumed
fingertip_link_names when consumed
```

The Policy repository currently consumes these concrete facts.

Do not expand the interface into a generic runtime snapshot.

Policy changes are unnecessary unless this task finds a concrete mismatch introduced by the Real cleanup.

---

# 8. Phase D — simplify persistence according to artifact type

## D1. Raw publication: keep atomicity, remove recursive durability

Keep:

```text
staging
writer close/finalization
row-count / readable-episode validation
target must not exist
same-parent rename
```

Remove the recursive `fsync_tree` durability pass and parent fsync requirement from normal Raw publication.

The requirement is "never publish a partially finalized episode", not "survive arbitrary power loss like a database".

Do not weaken recorder ownership or incomplete-vs-published distinction.

## D2. Simplify `atomic_json_dump`

For repository-local calibration/metrics JSON, use:

```text
mkstemp
write
close
os.replace
```

Remove explicit file/parent fsync unless another concrete current consumer demonstrably requires it.

Keep cleanup of an uncommitted temp file on failure.

## D3. Remove calibration backup files

Remove automatic timestamped backup creation from current calibration writers/examples, including camera/table/VR paths where present.

Git/history and experiment provenance are the archive.

Do not delete already-existing historical backup files unless they are generated artifacts clearly safe to remove; this task is about stopping future backup proliferation.

## D4. Table calibration persistence

Keep:

- validated plane;
- quality metrics;
- explicit operator confirmation;
- current timestamp if useful;
- temp + replace.

Remove:

- unused `schema_version`;
- fsync ceremony;
- post-write reopen/readback comparison;
- automatic backup.

Do not change calibrated numeric values as part of this cleanup.

## D5. VR calibration persistence

Persist irreducible current-state facts only.

A suitable minimal payload is conceptually:

```text
ref
theta_deg
quality:
    std_deg
    max_deviation_deg
    frames
calibrated_at
```

Derive at load/runtime:

```text
T_vr_to_robot
quality grade
```

Do not persist both source and derived value.

The coordinate convention belongs in code. If that convention changes materially in the future, recalibrate rather than build a calibration migration framework.

If the checked-in current calibration JSON must be structurally migrated, preserve its numeric calibration measurements exactly; this task must not invent/re-fit calibration.

## D6. Camera calibration persistence

Apply the same lifecycle principle where safe:

```text
validate current result
preserve other camera entries
temp + replace
no timestamped backup ceremony
```

Do not weaken physical serial identity, SE(3) checks or calibration quality.

---

# 9. Phase E — reduce runtime/config protocol leakage without touching safety

Do this only after A-D are green.

## E1. Readiness timeout follows the active process graph

In config validation:

- validate every timeout entry that is provided;
- do not require keys for every possible subsystem globally.

At actual process startup:

- every process that is actually started and requires readiness must have a timeout;
- missing timeout for an active process fails before/at startup with a clear error.

Do not introduce a capability registry or process graph framework.

## E2. Keep shutdown escalation safety; reduce report leakage

The following sequence is non-negotiable:

```text
revoke motion
stop request / is_running false
bounded join
terminate if needed
kill if needed
confirm worker is dead
only then close/unlink shared IPC
physical abnormal stop -> FAULT
```

The cleanup target is only that session-level code should not depend on a growing escalation-report protocol.

Prefer:

```text
RuntimeSupervisor.shutdown()
    -> simple clean success/failure to caller
    -> detailed ProcessExit/ShutdownReport, if retained, stays internal/logging
```

Do not replace clear safety states with swallowed exceptions.

Do not weaken the distinction between graceful and escalated physical-worker shutdown; it remains diagnostic/fault information.

## E3. Do not redesign RuntimeChannels

The current shared runtime object may contain channels unused by some workflows.

That is not sufficient reason to build:

```text
ChannelRegistry
CapabilityGraph
dynamic plugin IPC
```

Only remove obviously dead members discovered during this task.

## E4. Internal validation cleanup is opportunistic, not a sweep

Do not run a repository-wide "remove duplicate validation" refactor.

Remove only obvious redundant silent repair in files already changed.

External inputs, persisted data, process boundaries and hardware/SDK boundaries remain strict.

---

# 10. Files likely to change

Inspect current code first; this list is directional, not a requirement to touch every file.

## Action semantics

```text
dexmani_real/teleop/control/controller.py
dexmani_real/teleop/control/action_proposal.py
dexmani_real/teleop/control/vr_mapping.py
dexmani_real/deployment/action.py
dexmani_real/deployment/runner.py
dexmani_real/robot/projection.py
dexmani_real/robot/commands.py
dexmani_real/planning/kinematics/ik.py
```

Potential new small file:

```text
dexmani_real/robot/action_realization.py
```

Use another local name if ownership is clearer. Do not create a package hierarchy for one helper.

## Geometry

```text
dexmani_real/utils/geometry.py
dexmani_real/planning/kinematics/pose.py
dexmani_real/recording/recorder.py
dexmani_real/recording/storage/reader.py
dexmani_real/dataset/pointcloud.py
dexmani_real/sensor/pointcloud.py
dexmani_real/sensor/pointcloud_worker.py
dexmani_real/calibration/camera/extrinsics.py
dexmani_real/calibration/camera/solver.py
dexmani_real/teleop/vr_transform.py
```

## Dataset/cache

```text
dexmani_real/dataset/processing.py
dexmani_real/dataset/export.py
dexmani_real/dataset/contracts.py
examples/export_policy_zarr.py
dexmani_real/teleop/session.py
examples/collect_teleop.py
```

## Persistence/calibration

```text
dexmani_real/utils/atomic_io.py
dexmani_real/calibration/table.py
dexmani_real/calibration/camera/solver.py
dexmani_real/teleop/vr_transform.py
examples/calibrate_vr_heading.py
dexmani_real/calibration/state/table_plane.json      # structural migration only if required
dexmani_real/calibration/state/vr_transform.json     # structural migration only; preserve numbers
```

## Runtime/config

```text
dexmani_real/config/control.py
dexmani_real/runtime/processes.py
dexmani_real/runtime/supervisor.py
teleop/deployment/replay sessions that consume shutdown result
```

## Read-only/verification sibling paths

Check, but do not modify unless the Real cleanup creates a concrete interface mismatch:

```text
dexmani_policy/training/build_utils.py
dexmani_policy/deployment/runtime.py
```

---

# 11. Things that must remain strict and unchanged

The cleanup is incomplete if it makes any of these weaker:

## Motion authorization

```text
SafetyState
run_id epoch invalidation
publish-time authority
worker pre-SDK authority recheck
mode-switch / blocking-operation recheck
```

## Hardware boundaries

```text
arm hard joint limits
hand mechanical limits
finite command targets
collision checks
homing path validation
E-stop
fresh measured state before physical operations
```

## IPC / observation

```text
seqlock torn-read prevention
camera payload shape/dtype checks
point-cloud fixed realtime schema
arm/hand/camera/cloud/VR freshness
RGB-cloud source-camera identity
```

## Calibration correctness

```text
camera serial identity
current calibration lookup
SE(3) correctness
stationary-arm capture
calibration residual/quality gate
table-plane quality
```

## Data policy

```text
P0-1 teleop whole-capture discard choice
P0-2 strict complete finite canonical admission
Raw no-overwrite after publication
missing sensor values are not fabricated as zeros
```

---

# 12. Explicit non-goals / forbidden additions

Do not introduce:

```text
ActionRegistry
ActionFactory framework
Producer/Capability registry
generic RobotAction plugin system
generic geometry/frame graph
SchemaManager
CalibrationSchema hierarchy
MigrationManager
Raw v1/v2/v3 compatibility branches
Canonical migration framework
ContractRegistry
shared dexmani_contract package
generic active-subsystem graph framework
automatic runtime recovery after motion faults
new smoothing/fallback/control compensation
new guessed VR angular-velocity threshold
calibration hash/fingerprint compatibility
URDF hash compatibility
```

Do not refactor replay, homing, calibration motion or keyboard jog into the new online action realizer unless a concrete duplicated semantic on the active path requires it.

Do not use `git reset --hard` or discard unrelated local work.

---

# 13. Required tests

Prefer focused pure-logic tests. Do not connect to real hardware for this task.

## 13.1 Source-independent joint realization

For the same current arm state and the same joint/hand intent:

```text
Teleop-produced intent
Policy-produced intent
```

must produce the same realized physical target.

The realizer API must not accept a source identifier.

## 13.2 Source-independent EEF realization

For identical:

```text
EEF pose intent
hand intent
current arm qpos
previous arm command
runtime limits/workspace
```

Teleop and Policy paths must use the same realization code and produce the same success/failure/result.

There must be no Policy-only random fallback path.

## 13.3 Workspace projection

Test:

- in-workspace EEF intent is unchanged;
- out-of-workspace intent is projected exactly once;
- diagnostic magnitude is returned by the realizer;
- runner/controller do not recompute a second independent clip.

## 13.4 Hand operational projection

Test:

- in-range hand intent unchanged;
- out-of-range intent projected exactly once;
- diagnostic clip magnitude returned;
- worker/driver mechanical hard-limit tests remain.

## 13.5 Joint periodic equivalence

Preserve current nearest-equivalent behavior for joints supporting periodic representations.

Verify no source-specific difference.

## 13.6 IK failure classification

Preserve distinction between:

```text
no solution / no valid candidate
invalid numerical/model output
```

Do not turn expected IK rejection into silent current-joint fallback.

## 13.7 VR mapping

Verify:

- the hard-coded per-frame rotation clamp no longer exists;
- valid reset-relative mapping remains numerically correct;
- configured total-delta behavior, if retained, remains explicit;
- VR stale/freshness rejection remains outside the mapper and still works.

## 13.8 Quaternion validity

Verify:

```text
valid finite quaternion -> normalized
zero quaternion         -> error
nonfinite quaternion    -> error
```

Audit callers so no new identity fallback appears elsewhere.

## 13.9 Shared SO(3)/SE(3) semantics

Use the same test matrices/transforms against all consumers.

Include at least one transform between the former tolerance regimes so the repository no longer has one path accepting while another rejects solely because of copied tolerance values.

## 13.10 Task validation names

Verify path-safe task directory validation and task-identity validation remain distinct and callers use the correct function.

## 13.11 Strict canonical admission regression

Explicitly prove P0-2 remains unchanged:

- non-teleop source rejected;
- any current canonical floating field with NaN/Inf rejects the episode;
- tactile NaN still rejects the episode;
- no frame dropping/interpolation.

## 13.12 Canonical overwrite/rebuild

Verify:

```text
existing output + default
    -> reject

existing output + explicit overwrite
    -> fully build/validate staging first
    -> replace cache
```

Raw input remains unchanged.

## 13.13 Policy metadata regression

Verify the current Real canonical metadata still supplies the Policy-side facts actually consumed:

```text
format
task_name
dt
pointcloud_config if used
fingertip_link_names if used
```

Do not add a new metadata registry.

## 13.14 Raw publication regression

Verify staging/validation/no-overwrite/rename semantics remain.

Do not require recursive filesystem fsync in the test.

## 13.15 Minimal calibration persistence

Table:

- quality and plane values survive round-trip;
- no unused schema version is required;
- no post-write readback step is needed.

VR:

- load derives matrix from `theta_deg`;
- quality grade derives from stored measurements;
- numeric calibration measurements are preserved across structural migration.

Camera:

- serial identity and rigid transform validation remain strict.

## 13.16 Active readiness timeout

Test:

- unused subsystem timeout may be absent;
- a process actually started with no required timeout fails clearly;
- provided timeout values remain finite/positive.

## 13.17 Shutdown safety regression

Test with process doubles only:

- motion revoked before blocking stop;
- graceful worker stop succeeds;
- terminate/kill escalation is still detectable;
- a worker not confirmed dead prevents shared-memory close;
- physical worker abnormal stop causes FAULT;
- session caller no longer needs detailed escalation structure if supervisor internalizes it.

## 13.18 Existing safety/data regressions

Run existing focused tests for:

```text
run_id authority
command hard limits
freshness
seqlock/ring reads
camera ring
point-cloud schema
homing/planning pure logic
Raw reader/writer
deployment observation/action
```

---

# 14. Tests that should not be added

Do not add tests whose only purpose is to preserve new bureaucracy, such as:

```text
exact giant metadata dictionaries
generic action registry membership
generic subsystem capability sets
calibration schema-version matrices
legacy calibration migration chains
source-specific action-realizer behavior
deep equality of diagnostic report dataclasses
```

Tests should assert concrete numerical semantics, data policy, or safety behavior.

---

# 15. Offline validation

Do not run commands that connect to hardware.

At minimum:

```bash
python -m compileall -q dexmani_real examples
ruff check dexmani_real examples
git diff --check
```

Then run focused pure-logic tests affected by each phase.

If the repository has no existing focused test for a changed high-risk pure function, add a small deterministic test rather than relying on import success.

Do not treat offline tests as proof of real-hardware safety.

---

# 16. Implementation boundaries and efficiency rules

## 16.1 Prefer move-and-delete over duplicate-and-deprecate

When the shared realizer is proven:

- delete dead source-specific realization logic;
- do not keep permanent wrappers around the old path;
- do not maintain both new and legacy implementations.

Git history is the archive.

## 16.2 Do not change five subsystems in one unvalidated step

After each major phase:

1. run targeted tests;
2. inspect diff;
3. only then delete old code.

In particular, establish the new action realizer before removing old Teleop/Deployment realization helpers.

## 16.3 Keep module ownership obvious

A reviewer should be able to answer these questions from file locations alone:

```text
Who maps a human to an intent?        Teleop
Who maps model output to an intent?   Deployment Policy bridge
Who maps intent to physical targets?  One Real action realizer
Who authorizes motion?                Runtime safety
Who enforces final physical limits?   Workers/drivers
Who owns calibration current state?   Calibration/Real
Who owns canonical training cache?    Dataset export
```

If the refactor makes these answers harder, simplify it.

---

# 17. Expected code reduction

This task should reduce or consolidate code in these categories:

```text
deployment-only EEF IK setup
duplicate workspace clipping
duplicate hand projection
duplicate clip-stat recomputation
hard-coded per-frame VR rotation clamp
degenerate quaternion identity fallback
duplicate rigid-transform validators
duplicate rotation validators
ambiguous task validator names
recursive filesystem durability
calibration backup/readback/schema ceremony
inactive-subsystem timeout completeness requirement
session-level shutdown report interpretation
```

The final repository should have fewer independent behavior definitions than before.

Do not count safety checks at independent hardware/process boundaries as unwanted duplication.

---

# 18. Recommended execution phases

Use these commit-sized phases locally. They do not have to become separate Git commits, but each should be testable before the next.

## Phase 1 — action semantics

```text
ActionIntent / ActionRealization
shared realizer
Teleop migration
Policy deployment migration
diagnostic migration
remove Policy-only random IK fallback
```

Gate:

- Teleop and Policy no longer own duplicate physical realization;
- focused action tests pass;
- no safety boundary changed.

## Phase 2 — hidden transform + geometry cleanup

```text
remove per-frame VR clamp
strict quaternion normalization
shared SO(3)/SE(3) validation
rename task validators
remove redundant EMA alpha clamp
```

Gate:

- all geometry consumers agree;
- no silent invalid->identity behavior;
- teleop mapping tests pass.

## Phase 3 — canonical/cache cleanup

```text
clarify strict teleop canonical naming
explicit overwrite/rebuild
document task/dt/depth-scale ownership
verify existing Policy metadata interface
```

Gate:

- P0-2 unchanged;
- existing Policy integration still passes;
- Raw never overwritten.

## Phase 4 — persistence/calibration cleanup

```text
remove recursive fsync
simplify atomic JSON replace
remove calibration backups
remove table readback/schema ceremony
simplify VR persisted state
apply same persistence principle to camera calibration
```

Gate:

- numeric calibration values preserved;
- current calibration still validates;
- no hardware operation was executed.

## Phase 5 — runtime/config leakage cleanup

```text
active-only readiness timeout validation
supervisor owns shutdown-result interpretation
reduce session dependence on report structure
```

Gate:

- shutdown safety invariants pass process-double tests;
- no SHM ownership regression;
- no capability framework added.

---

# 19. Execution order for Codex CLI

Follow this sequence.

1. Inspect current local `dexmani_real` HEAD and diff against the reviewed head if newer.
2. Inspect local `dexmani_policy` only to confirm current Real canonical metadata consumers; do not broaden scope.
3. Map the live Teleop and Policy action call paths from producer to SDK boundary.
4. Record focused tests covering those paths.
5. Add the minimal source-neutral action intent/realization types and tests.
6. Move operational workspace/arm/hand projection and online EEF realization into the shared realizer.
7. Migrate Teleop to Human→Intent→Realizer without changing P0-1 discard policy.
8. Migrate Policy deployment to prediction→Intent→Realizer and remove Policy-only random IK fallback.
9. Move clip/IK telemetry ownership to realization results; delete duplicate caller computations.
10. Run targeted action/teleop/deployment tests before deleting old realization code.
11. Remove the hidden VR per-frame clamp and redundant EMA alpha repair; make quaternion normalization strict.
12. Centralize SO(3)/SE(3) validation and replace copied validators; rename the two task validators.
13. Run focused geometry/calibration/point-cloud/Raw tests.
14. Clarify strict canonical teleop admission naming and add explicit canonical overwrite/rebuild while preserving P0-2 exactly.
15. Verify `dexmani_policy` still consumes the same minimal Real metadata; avoid Policy code changes unless required.
16. Simplify Raw/calibration persistence: remove recursive fsync, automatic calibration backups, redundant readback and unused/derived persisted fields.
17. Run calibration/data tests and confirm checked-in calibration numeric values were not changed except deterministic representation migration.
18. Make readiness timeout validation active-process-only.
19. Internalize detailed shutdown-report interpretation in Supervisor without changing stop/escalation/FAULT/SHM safety.
20. Run all relevant offline tests, compile/ruff/diff checks.
21. Only now delete dead helpers/imports/comments from old paths.
22. Perform the final stale-reference audit below and inspect every remaining match.

Do not use `git reset --hard` or discard unrelated local work.

---

# 20. Final stale-reference audit

Before finishing, search at least for:

```text
enable_random_fallback=True
max_per_frame_rot_rad
project_arm_command(
project_hand_command(
decode_policy_action(
workspace_clip_m
hand_clip_rad
arm_clip_count
_validate_rigid_transform
rigid homogeneous
validate_rotation_matrix
normalize_quat_wxyz
return np.array([1.0, 0.0, 0.0, 0.0]
def validate_task_name
fsync_tree
os.fsync
.json.bak
schema_version
T_vr_to_robot
quality.grade
readback does not match published fit
_READINESS_SUBSYSTEMS
readiness_timeouts_s is missing a runtime subsystem
ShutdownReport
ProcessExit
report.clean
```

Do not blindly require zero matches.

Every remaining occurrence must be classified as one of:

1. intentionally preserved hardware/safety behavior;
2. explicit current data/calibration semantics;
3. dead/stale reference that must be removed.

Also search for accidental source-aware realization concepts:

```text
source == "policy"
source == "teleop"
policy_ik
teleop_ik
policy_workspace
teleop_workspace
```

The shared physical realizer must not depend on action origin.

---

# 21. Final acceptance criteria

The task is complete only when all of the following are true.

## Data-policy decisions

- P0-1 remains unchanged: current teleop technical-failure discard policy is preserved.
- P0-2 remains unchanged: canonical admission is whole-episode, teleop-only and strict finite full-modality under the current schema.
- no bad-frame interpolation/fill/drop framework was introduced.

## Action semantics

- Teleop and Policy produce a shared source-neutral action intent.
- one Real action-realization path owns workspace projection, operational hand projection and online EEF->joint behavior.
- Teleop and Policy do not each maintain independent copies of these transforms.
- Policy-only random IK fallback is gone.
- the realizer does not receive an action-source identity.
- `RobotCommand(run_id,...)`, motion authorization and worker/driver hard-limit gates remain downstream and strict.
- expected IK rejection is not silently turned into current-joint fallback.
- action clipping/IK diagnostics come from the realizer rather than duplicate recomputation.

## Teleop

- Human→Intent logic remains explicit: VR calibration/mapping, reset anchor, configured scale/EMA and hand retargeting.
- the hidden hard-coded per-frame rotation clamp is gone.
- no replacement hidden smoothing/fallback was added.
- runtime VR freshness remains strict.

## Geometry

- one SO(3)/SE(3) validation implementation is used across active Real paths.
- the previous `1e-5` vs `1e-6` semantic split is gone.
- degenerate/nonfinite quaternion normalization fails instead of returning identity.
- boundary-specific error types wrap shared math rather than duplicate it.
- task-directory safety and task identity have distinct names.

## Canonical/cache

- strict canonical admission behavior is unchanged but clearly named/owned.
- one-task-per-store remains a current organization choice, not a generic multi-task framework.
- `dt` remains strict.
- current depth-scale store consistency remains unless a concrete current need requires otherwise.
- explicit overwrite/rebuild is supported only for Canonical, after staging validation.
- Raw is never overwritten by this workflow.
- current minimal Real↔Policy metadata continues to work.

## Persistence / calibration

- Raw still uses unpublished staging, writer finalization, validation, no-overwrite and rename publication.
- recursive tree fsync is gone unless a concrete current requirement is documented.
- calibration writes use simple temp + replace semantics.
- automatic timestamped calibration backups are no longer generated.
- table calibration no longer requires an unused schema version or redundant write-readback proof.
- VR calibration persists irreducible measurements rather than duplicated matrix/grade state.
- current camera serial/SE(3)/quality validation remains strict.
- no calibration numeric value was invented or silently altered during representation cleanup.

## Runtime / lifecycle

- an unused subsystem does not require a readiness-timeout config entry.
- every actually started readiness-gated process still requires a valid timeout.
- motion is revoked before blocking shutdown.
- physical workers are still bounded-join/terminate/kill managed and must be confirmed dead before shared-memory close.
- physical abnormal stop still produces FAULT.
- session code no longer depends unnecessarily on a detailed shutdown-report protocol.
- seqlock, freshness, source identity and `run_id` protections remain unchanged.

## Complexity

- no action registry/factory, capability framework, schema framework, migration framework or shared contract package was introduced.
- no permanent legacy runtime branch was added for old implementation details.
- no unrelated replay/homing/calibration-motion refactor was performed.
- the number of independent definitions of physical action semantics and rigid-transform validity is lower than before.
- every new abstraction corresponds to a real boundary and is smaller than the duplicate logic it replaces.

---

# 22. Final decision tests

Before adding or keeping an action transform, ask:

> **Does this transform define Human/Model intent, or does it define how any intent becomes a physical target?**

- Human/Model-specific -> source adapter.
- physical realization -> one shared Real owner.

Before adding a validator, ask:

> **Does it reject an invalid value, or silently manufacture a different valid-looking value?**

If it manufactures a replacement, it is not a validator. Make the transform explicit or remove it.

Before duplicating geometry logic, ask:

> **Would two modules be allowed to disagree about whether the same rotation/transform is valid?**

If no, there must be one mathematical owner.

Before adding persistence machinery, ask:

> **Is this Raw evidence, a rebuildable cache, or current calibration state?**

- Raw evidence -> create-only publication discipline.
- rebuildable cache -> explicit replace is allowed.
- current calibration -> simple validated replacement.

Before keeping a runtime/config protocol, ask:

> **Does it prevent a concrete unsafe robot state or incorrect experiment, or does it only describe hypothetical platform completeness?**

If it only describes hypothetical completeness, remove or localize it.

Before changing any of the strict safety/data paths, ask:

> **Was this behavior explicitly selected by the user or required by a real hardware/process boundary?**

If yes, preserve it even when the code looks more defensive than a typical research repository.
