# DexMani Real removal and simplification task

## Intent

Simplify dexmani_real for its actual role: a personal PhD / RAL real-robot experiment repository, not a production robotics platform.

The goal is not to minimize code mechanically. Remove duplicated diagnostics, rebuildable-artifact self-proof, forensic residue, and the secondary workflow-failure lifecycle while preserving the boundaries that protect physical safety, experiment correctness, and irreplaceable raw data.

This task was designed from code state 143ff2fcdfd2f66776b2dd1c2676101463f24ec2, before this task file was added. If implementation code has changed since then, re-trace the affected definition/producer/consumer paths before editing rather than applying this plan mechanically.

Follow AGENTS.md. In particular:

- physical safety > experiment correctness > iteration speed > readability > generic extensibility;
- prefer delete, then inline, then merge, then rewrite;
- do not execute hardware-affecting code without explicit authorization;
- do not replace removed mechanisms with renamed equivalents;
- do not add a committed test suite;
- persisted semantic changes require an exact new schema version rather than compatibility branches.

## Non-goals and invariants

Do not simplify or redesign the following as part of this task:

- SafetyState = DISARMED / ARMED / RUNNING / FAULT;
- run_id lifecycle fencing at the SDK boundary;
- arm/hand finite and physical-limit validation;
- ESTOP and verified child shutdown before shared-memory release;
- XHand dropout / passive fail-safe behavior;
- sensor freshness checks;
- shared-memory rings and the separate recorder I/O process;
- raw episode staging, structural admission, and durable publication;
- whole-episode raw-to-Zarr admission;
- dexmani_real <-> dexmani_policy semantic compatibility checks;
- keyboard teleop, replay, audio feedback, HOME protocol, or HOME planning;
- RealSense backend support, hand-retargeting backends, or replay evaluation;
- the useful experimental functions of either diagnostic script in item 02.

Do not add a new shared session_failed, experiment_failed, failure ledger, ACK protocol, or other replacement for workflow_failed.

## Required implementation order

Implement the work in the following order. Keep runtime-lifecycle changes separate from data-format changes so failures remain easy to review and bisect.

1. Items 02 + 03: diagnostic readability and point-cloud instrumentation.
2. Items 04 + 05: Zarr publication simplification.
3. Items 06 + 07: raw metadata/schema cleanup and failed-staging cleanup.
4. Item 08: remove policy-session result artifact.
5. Item 09: remove dynamic process taxonomy while temporarily retaining current workflow-failure behavior.
6. Item 10: remove workflow_failed completely.
7. Item 12: only opportunistic cleanup; do not start a config-validation refactor.
8. Item 01: delete the broad smoke suite last.

Before editing, run git status --short and preserve unrelated changes.

---

## 02. Keep both diagnostic scripts; remove only temporary statistics and improve readability

### Files

- examples/realsense_record_example.py
- examples/pointcloud_process_example.py

### Intent

Both files remain separate and keep their current experiment-facing roles.

realsense_record_example.py remains the live RealSense RGB-D / point-cloud diagnostic. Preserve, unless a small implementation detail is proven unused:

- camera discovery / connection behavior;
- connect/disconnect lifecycle inspection;
- RGB-D live display;
- depth visualization and colormap switching;
- point-cloud display;
- raw/processed point-cloud switching;
- point-cloud freeze/reset controls;
- production point-cloud configuration display;
- auto-exposure-priority read/toggle/restore behavior;
- keyboard controls and current live performance information.

pointcloud_process_example.py remains the production point-cloud commissioning tool. Preserve:

- camera connection and device/sensor readback;
- RGB-D inspection;
- calibration/extrinsics loading;
- optional table fitting and explicit publication;
- raw point-cloud construction;
- production point-cloud construction;
- workspace and coordinate-frame visualization;
- diagnostic snapshot export;
- Open3D visualization;
- aggregate production-pipeline benchmarking.

Do not merge or rename these two scripts.

### Remove / simplify

Remove only temporary profiling/statistics machinery that obscures the main data flow.

For realsense_record_example.py:

- remove statistics-only dataclasses / aggregate dictionaries if they exist solely to feed rolling/final performance summaries;
- keep direct per-frame values needed by the live HUD, such as current FPS/read latency/point-cloud latency/depth-valid ratio/point count;
- use the minimum state required for a stable FPS display;
- remove end-of-run arrays and detailed aggregate performance summaries that do not affect diagnostics or control behavior.

For pointcloud_process_example.py:

- remove per-stage point-cloud timing plumbing and stage-percentile reports;
- remove timing bookkeeping for setup operations such as calibration-file loading or table fitting;
- keep simple wall-clock point-cloud build timing and a compact aggregate benchmark, e.g. pipeline p50/p95 and capture-to-cloud p50/p95;
- keep failure/output reporting, snapshot metadata, table fit quality, and all geometry-relevant information.

Improve readability through function order, narrower local state, direct values, and clear names. Do not add comments or docstrings to explain the refactor. Existing useful comments/docstrings may remain; readability must come from code structure.

Do not introduce a shared diagnostics framework or diagnostics/common.py.

### Acceptance

- both scripts still exist at the same paths;
- their operator-visible capabilities above remain available;
- no hardware command is executed during validation;
- temporary profiling/state plumbing is materially smaller;
- no new explanatory comments/docstrings are added for this cleanup.

---

## 03. Remove point-cloud timing/statistics from the production implementation

### Files

- dexmani_real/sensor/pointcloud.py
- update examples/pointcloud_process_example.py as required by item 02.

### Intent

build_point_cloud() must be the direct production algorithm, not a wrapper around a diagnostic implementation whose statistics are discarded.

### Required changes

Remove production-only diagnostic API/state when no real consumer remains:

- PointCloudBuildTimings;
- PointCloudBuildStats;
- build_point_cloud_with_stats;
- perf_counter_ns calls used only for stage profiling;
- count calculations used only for diagnostic reporting.

Move no algorithmic work into the example script. There must remain one canonical point-cloud algorithm.

Preserve all calculations required for algorithmic decisions, including counts/masks needed to decide whether a processing stage has enough points to continue. Preserve point-cloud output semantics, sampling, table removal, workspace filtering, outlier filtering, XYZRGB representation, dtype, and failure behavior.

The example benchmark should time the public build_point_cloud() call externally.

### Acceptance

- repository search finds no PointCloudBuildTimings, PointCloudBuildStats, or build_point_cloud_with_stats;
- production build_point_cloud() returns the same type/shape/semantics as before;
- point-cloud worker behavior and validate_point_cloud_array boundary remain unchanged unless a change is strictly required by removal of the stats API.

---

## 04. Delete staged-Zarr self-validation

### File

- dexmani_real/dataset/export.py

Delete _validate_staged_policy_zarr() and its call.

Do not replace it with another post-write re-open verifier.

Retain the existing producer-side checks during transformation:

- uniform static semantics across episodes;
- modality key completeness;
- transformed shape/dtype checks;
- transformed row-count checks;
- whole-episode rejection.

Retain the strict consumer-side dexmani_policy contract.

The intended trust chain is:

raw reader/admission -> transform checks -> write cache -> publish -> dexmani_policy semantic contract.

---

## 05. Use lightweight publication for rebuildable Zarr only

### Files

- dexmani_real/dataset/export.py
- dexmani_real/utils/atomic_io.py only if a tiny helper materially improves clarity.

### Intent

Raw episodes are irreplaceable source data and keep durable publication. Policy Zarr is a rebuildable training cache and does not need recursive fsync of every chunk.

### Required behavior

For Zarr:

- create staging under the final target parent, as today;
- refuse an occupied final target;
- publish by same-filesystem rename;
- clean owned staging on failure;
- do not recursively fsync the Zarr tree.

Prefer a small explicit helper such as publish_staging() only if it keeps no-overwrite/same-parent semantics clearer than inline code. Do not add a configurable durable=False flag to atomic_publish().

For raw episodes:

- keep _validate_temp_episode();
- keep atomic_publish();
- keep recursive durability/fsync behavior.

For small calibration/config JSON:

- keep atomic_json_dump() unchanged.

### Acceptance

- dataset/export.py no longer calls durable atomic_publish() for policy Zarr;
- recording/recorder.py still uses durable atomic_publish();
- no-overwrite behavior is preserved.

---

## 06. Compress raw episode metadata with an exact schema bump

### Files

- dexmani_real/recording/recorder.py
- dexmani_real/recording/storage/schema.py
- dexmani_real/recording/storage/reader.py
- update only real consumers/documentation that break because of the schema change.

### Required schema change

Bump EPISODE_SCHEMA_VERSION from 32 to 33.

Keep exact-version reading. Do not add a v32 compatibility branch.

Remove final metadata that is duplicated, constant, or derivable:

- task_success;
- duration (keep wall_duration_s);
- fps (keep canonical control_hz);
- wall_fps;
- has_camera;
- has_timestamps;
- camera_stream_frames;
- truncated;
- stop_reason (keep termination_reason).

Keep at least:

- schema_version;
- technical_status;
- had_pause;
- termination_reason;
- control_hz;
- num_frames;
- wall_duration_s;
- min_frames_met for now;
- task/operator/provenance and camera/calibration semantics already required downstream.

Also remove the historical episode_xxx.result.json invalidation compatibility check from EpisodeReader.require_valid(). Exact schema versioning is the current contract; do not retain dead compatibility paths.

Do not weaken _validate_temp_episode().

### Acceptance

- newly written raw episodes use schema 33;
- reader accepts exactly schema 33;
- no compatibility branch is introduced;
- training admission still rejects technical_status != valid and paused episodes through the existing processing path;
- min_frames_met remains readable by visualize_episode.py.

---

## 07. Delete incomplete-episode forensic publication

### Files

- dexmani_real/recording/recorder.py
- related imports/callers only.

### Intent

A failed partial recording is not a second artifact type. Preserve the concrete exception in logs; remove owned staging after all writers release resources.

### Required changes

Delete:

- _preserve_incomplete_staging();
- incomplete_* naming/allocation;
- failure_note.json;
- failure-artifact calls to atomic_json_dump().

On recording failure or clean discard:

1. preserve the original failure as the primary error;
2. close/release camera and HDF5 writers;
3. remove the owned temporary episode directory;
4. if cleanup itself fails, log the staging path and cleanup error and leave the .tmp_* directory in place.

A cleanup failure must not replace or hide the original recording/storage exception.

Keep the startup stale-staging warning if it is useful for naturally retained .tmp_* directories.

---

## 08. Remove session_result.json

### Files

- examples/run_policy.py
- README.md

Delete the final session_result.json write and any now-unused import.

Policy-session success/failure is represented by the process/CLI exit code. Task success remains an offline evaluation/annotation concern.

Update README so a rollout session is documented as containing run_config.yaml plus saved episode directories, without session_result.json.

Do not replace this file with a marker or renamed session-status file.

---

## 09. Remove dynamic service_process_names classification

### Files

- dexmani_real/runtime/processes.py
- dexmani_real/runtime/supervisor.py
- dexmani_real/teleop/session.py
- dexmani_real/deployment/session.py
- any other direct caller.

### Intent

There is one fixed physical process boundary in this repository: arm and hand directly own motion hardware. Workflows must not pass their own service/critical taxonomies.

### Required changes

Remove the service_process_names parameter from shutdown APIs and all callers.

Use one private fixed classification such as:

_PHYSICAL_PROCESS_NAMES = frozenset({"arm", "hand"}).

During verified shutdown:

- abnormal/non-graceful arm or hand exit -> physical fault / error_state / SafetyState.FAULT;
- abnormal/non-graceful nonphysical worker exit -> session failure but not physical FAULT;
- any child that cannot be confirmed stopped -> physical fail-closed FAULT and do not unlink shared memory;
- fully clean shutdown may transition to DISARMED.

Add a small derived property such as ShutdownReport.clean if it removes repeated shutdown-success calculations. It should be true only when shared memory closed and every process exited code 0 without escalation.

At this stage, workflow_failed may still be set for a confirmed nonphysical worker failure so item 09 can be reviewed independently. Do not combine item 09 and item 10 into one opaque change.

### Acceptance

- repository search finds no service_process_names;
- arm/hand abnormal exit remains a physical FAULT;
- nonphysical abnormal exit causes a failed session without falsely claiming a hardware fault;
- unverified child termination still prevents shared-memory release.

---

## 10. Remove workflow_failed and use failure ownership instead

### Files likely affected

- dexmani_real/ipc/channels.py
- dexmani_real/runtime/safety.py
- dexmani_real/runtime/supervisor.py
- dexmani_real/runtime/processes.py
- dexmani_real/recording/client.py
- dexmani_real/recording/io_worker.py
- dexmani_real/sensor/camera/worker.py
- dexmani_real/sensor/pointcloud_worker.py
- dexmani_real/teleop/runner.py
- dexmani_real/teleop/session.py
- dexmani_real/deployment/operator.py
- dexmani_real/deployment/runner.py
- dexmani_real/deployment/session.py
- dexmani_real/calibration/camera/session.py
- remove obsolete smoke-test references later with item 01.

Re-trace actual current references before editing.

### Target model

Keep only two failure channels.

Physical/control fault:

- arm/hand failure, ESTOP, or unverified shutdown;
- represented by error_state and SafetyState.FAULT.

Experiment/session failure:

- camera, VR, point-cloud, recorder, policy, teleop, or operator failure;
- represented by the owning exception/nonzero process or local session result;
- ends the current session safely;
- does not set a shared sticky failure bit and does not claim a physical fault.

Do not add a replacement shared failure flag.

### Supervisor behavior

A started required child that exits unexpectedly ends the session.

RuntimeSupervisor.check() should:

- detect any dead started child;
- clear its ready event;
- if arm/hand died, immediately latch physical fault and revoke to FAULT;
- otherwise revoke active motion to a non-FAULT state if needed;
- report failure to the owning session through return/exception/local result, not shared memory.

RuntimeSupervisor.run() should return a simple clean/failed outcome or otherwise make the result available locally to run_teleop_experiment().

The session function should use local state plus ShutdownReport.clean to choose exit code.

### Recording failure behavior

Remove workflow_failed writes/checks from RecorderClient and recorder I/O.

Required recording transport/storage failures are exceptional:

- broken control/result queue;
- recorder START timeout;
- recorder process unavailable;
- sample submission failure;
- RecordingResult.error;
- recorder STOP/result timeout.

Mark the current recorder state technically invalid, revoke active motion with the existing recording-failure reason where appropriate, then raise RuntimeError to the owner. Do not silently convert required recorder failure into a false return plus shared flag.

A normal local precondition such as already recording or stop pending may remain a non-exceptional refusal if needed by existing callers.

If changing RecorderClient.add_frame(), prefer success-by-return and failure-by-exception rather than a boolean nobody meaningfully consumes. Update the real callers only.

The recorder worker may revoke current motion on a fatal recording failure, but should then raise and exit nonzero rather than set a workflow flag.

### Teleop behavior

Remove all workflow_failed branches.

Preserve recoverable teleop semantics that are not currently workflow failures.

When a required recording camera/recorder resource is truly lost in a path that currently sets workflow_failed:

- mark/stop the active episode as abnormal as appropriate;
- raise so the teleop worker exits nonzero;
- let the main supervisor end the session.

Do not keep the process alive in an unusable degraded state.

Teleop finalization must continue marking an interrupted active recording invalid.

### Policy behavior

Remove all workflow_failed branches/writes.

Use local exception state inside PolicyRunner.run() / run_policy_worker() so a local policy or recorder exception finalizes an active episode as abnormal before re-raising.

Also handle external nonphysical worker failure without a shared flag: when the main session shuts the runtime down while a policy episode is still active, the policy runner must treat that unfinished active run as abnormal. A normal completed/STOPped episode already clears run_id; use that local distinction rather than adding shared state.

Preserve existing per-episode recoverable behavior such as an observation becoming stale where the current code already ends only that episode instead of setting workflow_failed.

### Camera / point-cloud / other workers

Camera and point-cloud worker exceptions should log as appropriate and re-raise. Process exit is the failure signal.

Do not set a replacement shared status.

### Camera calibration

CameraCalibrationSession._runtime_issue() currently infers camera failure from workflow_failed. Replace this with the actual owned process state.

Pass/retain the camera process handle in CameraCalibrationSession and explicitly report camera worker exited when it is no longer alive. Keep the arm-process and freshness checks.

### Safety request functions

Remove workflow_failed from request_policy_start(). A failed session should already be ending/ended via local supervision; existing is_running, ESTOP, error state, stop request, HOME authorization, and SafetyState gates remain.

### Session exit status

Use local session failure state and shutdown report data:

- physical fault / ESTOP -> failure;
- required worker/teleop/operator failure -> failure;
- non-clean process shutdown or failed shared-memory close -> failure;
- normal operator/episode completion with clean shutdown -> success.

Do not persist this state into shared memory merely so the final return statement can read it.

### Acceptance

Repository search must find no workflow_failed after item 01 is also complete.

Before deleting the smoke file, all production references must already be gone.

The resulting lifecycle must not contain another shared session-failure equivalent.

---

## 12. Keep config validation; only remove proven cosmetic ceremony opportunistically

This item is primarily a guardrail, not a requested cleanup.

Do not redesign validate_config() or remove hardware/algorithm/integration invariants.

Preserve validation for:

- arm/hand hard, rated, operational and HOME limits;
- xArm SDK speed/acceleration/collision settings;
- camera geometry/rate/stall/freshness relationships;
- workspace/table/static collision geometry;
- PointCloudConfig semantics;
- EMA/VR/retargeting mathematical domains;
- recording capacity relationships;
- policy/runtime joint/finger/tactile/point-cloud/action semantics;
- rollout filesystem safety and pinned artifact identity.

Only when touching the exact surrounding code for another required change may trivial cosmetic checks such as surrounding-whitespace rejection be normalized or removed. Do not create a standalone config-validation diff for a handful of lines.

Most importantly, keep generic configuration validation at the external/config ownership boundary. Do not move it into control loops or duplicate it in workers.

---

## 01. Delete the broad deployment smoke suite last

### File

- dexmani_real/deployment/smoke_test.py

The final repository should not keep this broad committed regression suite.

However, do not delete it before the refactor starts.

### Efficient migration use

At the beginning, while the baseline still matches its assumptions, it may be run once as an offline baseline. It may also be used selectively after early low-risk items if it still applies.

Do not spend time expanding or redesigning the suite for the new architecture. In particular, when items 06/09/10 intentionally remove old schema/lifecycle mechanisms, delete obsolete expectations instead of building a new regression framework that will itself be removed.

After all production changes and focused checks pass:

- delete dexmani_real/deployment/smoke_test.py;
- remove any imports/references that exist at that time;
- do not replace it with a tests/ directory or another committed mega-smoke file.

Use focused one-off checks for the final architecture.

---

## Review constraints

Keep changes narrowly scoped. Do not use this task as an opportunity to refactor:

- HOME planning;
- keyboard input;
- audio implementation;
- replay;
- retargeting;
- camera-family support;
- policy action/statistics code unrelated to workflow_failed;
- general naming/style outside touched areas.

Delete dead imports, fields, helpers, docs, and config only when they become unused because of this task.

README should describe stable user-facing behavior only. Do not copy this task plan into README.

## Offline verification

No hardware-affecting program may be executed.

At minimum run:

    python -m compileall -q dexmani_real examples
    ruff format --check dexmani_real examples
    ruff check --select F401,F821,F822,F823,I dexmani_real examples
    git diff --check

If Ruff is available, format changed files before the final checks.

Use focused temporary/offline checks, not committed tests, for the changed boundaries.

### Point cloud

- use synthetic RGB/depth/geometry input;
- verify build_point_cloud() output dtype/shape and deterministic semantics needed by existing code;
- verify no stats API remains.

### Zarr

- exercise an offline export fixture or temporary synthetic raw episode if available without hardware;
- verify staged output renames successfully;
- verify occupied targets are refused;
- verify dexmani_policy contract can still open the result.

### Raw schema

- create/read a temporary schema-33 episode entirely offline;
- verify removed attrs are absent;
- verify technical_status, had_pause, timing, camera sidecars, and min_frames_met still work;
- verify schema 32 is rejected rather than compatibility-read.

### Runtime lifecycle

Use dummy multiprocessing children only.

Verify:

- normal children + clean shutdown -> DISARMED and clean report;
- nonphysical child crash -> failed session, no physical FAULT;
- arm or hand child crash -> error_state + FAULT;
- confirmed nonphysical worker failure -> failed session with DISARMED after safe terminal shutdown;
- physical worker failure -> FAULT; unverified child shutdown -> FAULT and no unsafe shared-memory release;
- no shared replacement for workflow_failed.

### Recording failure propagation

Use mocked/offline recorder queues/results.

Verify:

- required recorder failure raises to its owner and revokes active motion as needed;
- active episode becomes technically invalid;
- clean discard remains non-failure.

Do not install or upgrade dependencies to make optional checks pass. Report unavailable optional tools.

## Final repository assertions

Before finishing, search the repository and verify:

- no PointCloudBuildTimings;
- no PointCloudBuildStats;
- no build_point_cloud_with_stats;
- no _validate_staged_policy_zarr;
- no service_process_names;
- no workflow_failed;
- no session_result.json;
- no failure_note.json or incomplete_* protocol;
- no historical raw .result.json compatibility read;
- EPISODE_SCHEMA_VERSION == 33;
- raw recorder still calls durable atomic_publish();
- policy Zarr does not recursively fsync its staging tree;
- both diagnostic scripts still exist and retain their major functionality;
- dexmani_real/deployment/smoke_test.py is deleted;
- no replacement committed test suite was added;
- no hardware-affecting validation was run.

Review the final diff for unrelated changes and for any mechanism removed here that has been recreated under another name.
