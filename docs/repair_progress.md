# DexMani Real 修复进度

对应 [repair_workflow.md](repair_workflow.md)。仅主 agent 维护。

## 当前状态

- 2026-09-06：已制定 workflow，复用现有 sol-high / terra-xhigh / luna-max。
- Baseline、Phase 0、完整 Phase 1（A → B → C）、Phase 2、完整 Phase 3（A → B → C）、
  Phase 4、Phase 5、完整 Phase 6（A → B → C）和 Phase 7 已完成。
- 所有未进入 red-test gate 的计划 finding 仍不视为已证实的当前缺陷。

## Execution baseline — 2026-09-06

| Repository | HEAD | Worktree before Phase 0 | Software baseline |
| --- | --- | --- | --- |
| dexmani_real | `3cffc047d660ab72beb3f9d1f8d325efba690ee7` | 本 workflow 的 docs/navigation 未提交 | `conda run --no-capture-output -n real_robot python -m pytest -q`: 199 passed, 106 subtests passed |
| dexmani_policy | `31b697cb6c3d4ba598c2a3b396f99393ad5ffc9d` | clean | `conda run --no-capture-output -n policy python -m unittest discover -s tests -p 'test_*.py' -v`: 77 tests OK |

测试均使用 temporary artifact、fake SDK 或 mock；没有运行 physical example、连接真实
xArm/XHand/RealSense，或启动训练、DDP、simulator rollout。

## Phase 0 — complete

- Current source evidence: Policy planner uses `check_self_collision=False`; its SafetyGate has
  no collision callback. VR/keyboard teleop retain `OnlineIKConfig` endpoint collision checks,
  while their realtime SafetyGate has no transition collision callback.
- Changed only `user_design.md`: recorded the accepted learned-policy and teleop collision
  boundaries. Runtime code and tests were untouched.
- Validation: `conda run --no-capture-output -n real_robot python -m pytest -q
  tests/test_policy_executor.py` → 41 passed; `git diff --check` → PASS.
- Preserved: learned arm spike clip, PolicySpec hand ownership, endpoint collision behavior,
  no realtime policy collision query, and no teleop swept collision query.

## Plan reference and current source

| Repository | Plan baseline | Executed baseline | Initial worktree |
| --- | --- | --- | --- |
| dexmani_real | `2af49e14a78d1a0977735a48d06f2917b3c4d53c` | `3cffc047d660ab72beb3f9d1f8d325efba690ee7` | 本 workflow 的 docs/navigation 未提交 |
| dexmani_policy | `9e9597f4e3a06702f23cc88084b821217053a580` | `31b697cb6c3d4ba598c2a3b396f99393ad5ffc9d` | clean |

两仓库都已偏离计划提供的 baseline，因此每个跨仓库 boundary 都先按当前源码重新追踪。至此尚未
修改 Policy；进入 Phase 1B 前会再次确认其 HEAD、worktree 与导出调用链。

## Phase 1A — complete (Real processed/export)

- Current source evidence: visual profiles previously generated fingertips from camera-aligned qpos,
  whereas `joint` copied raw `hand_fingertip`; that made one persisted field have two derivations.
- Red gate: `test_joint_writer_recomputes_fingertips_from_processed_joint_state` failed on the prior
  implementation because output retained the deliberately wrong raw `-7` points. The new missing
  fingerprint export test also failed because the v12 exporter accepted the artifact.
- Implementation: bumped processed HDF5 v12 → v13 and Policy Zarr v6 → v7 without a legacy reader;
  every writer now materializes `joint_state` once and derives all fingertips with the same arm+hand FK.
  Processing freezes derivation (`fk_from_processed_joint_state`), algorithm ID
  (`arm_hand_fk_from_joint_state_v1`), and a canonical geometry SHA-256 covering both FK URDF hashes,
  arm EEF, SDK→URDF mapping, ordered fingertip links, and normalized hand mount rotation/position.
  Export validates and propagates these attrs, and rejects non-uniform multi-episode identity.
- Tests: all four profile writer paths are checked for their exact FK `joint_state` inputs; JOINT output
  is checked against intentionally wrong raw fingertips. Geometry tests cover every dependency plus
  `q`, `-q`, and scaled-quaternion equivalence. Missing/non-uniform fingerprint artifacts reject.
- Green: `tests/test_policy_offline_multimodal.py tests/test_policy_multimodal_observation.py` →
  `33 passed, 53 subtests passed`; `python -m compileall -q dexmani_real examples` → PASS;
  complete Real suite → `203 passed, 116 subtests passed`; `git diff --check` → PASS.
- Preserved: raw schema v24; deployment schema v3; point-cloud generation algorithm; no legacy v12/v6
  fallback; no hardware process or physical device was started.

## Phase 1B — complete (latest Policy export)

- Re-traced the latest `dexmani_policy` at
  `31b697cb6c3d4ba598c2a3b396f99393ad5ffc9d`, rather than applying the plan's old
  baseline mechanically. Its worktree also contained unrelated changes outside the allowed
  export/test/progress files; they were preserved and not reverted.
- Red gate: before the implementation,
  `conda run --no-capture-output -n policy python -m unittest tests.test_deployment_export -v`
  produced four failures and three errors: the old exporter rejected v7, still accepted v6,
  and omitted the required field semantics.
- Implementation: the Policy exporter now accepts only Real Zarr v7. It validates the existing
  Zarr attrs once, returns validated point-cloud semantics (frame, units, color source,
  policy/config/table/sampling/transform identity), and writes them to the existing
  `ObservationFieldSpec.semantics`. A requested fingertip field now requires and propagates
  its frozen derivation, policy ID, and geometry SHA-256. Deployment artifact schema remains v3;
  no v6 reader or metadata inference was added.
- Green: affected exporter tests → `10 tests OK`; final latest-Policy full unit suite →
  `82 tests OK`; changed Policy files compile with `py_compile`; `git diff --check` → PASS.

## Phase 1C — complete (Real startup compatibility)

- Current-source trace confirms `run_policy_deployment()` calls
  `validate_policy_runtime_compatibility()` before `RuntimeChannels.create()`, hence before
  any deployment worker is constructed.
- Red gate: the new Real identity tests against the old compatibility function produced
  `12 failed, 3 passed`; old code only checked point-cloud tensor shape/count and did not
  compare either semantic fingerprint.
- Implementation: Real constructs expected point-cloud identity from the resolved runtime
  point-cloud config, table plane canonical JSON, and point-cloud constants. It constructs the
  fingertip fingerprint from current arm/hand URDF bytes, EEF frame, SDK→URDF mapping, ordered
  fingertip links, and mount transform. Every requested semantic key is compared explicitly;
  absent values and mismatches fail with a field-specific error before channels or workers.
- Tests: same `N` with changed voxel, workspace, or table plane rejects; all point-cloud field
  semantics and selected missing fields reject; fingertip derivation/policy/geometry mismatch
  and omission reject. A lifecycle spy proves identity rejection occurs before channel creation.
  The pure integration smoke creates a temporary Real Zarr v7, asks the latest Policy exporter
  to build its contract, parses its unchanged v3 deployment artifact, and validates it in Real.
- Green: focused Real Phase 1 tests plus the cross-repo smoke →
  `33 passed, 63 subtests passed`; final complete Real suite →
  `208 passed, 138 subtests passed`; `python -m compileall -q dexmani_real examples` and
  `git diff --check` → PASS. No physical example, SDK connection, hardware process, training,
  or DDP was started; Policy suite demo/eval paths use temporary files and fake runners.

## Phase 2 — complete (stationary camera-calibration sampling)

- Current source evidence: the old capture path collected the N-frame ArUco median and then read
  only one arm state. It therefore could not establish that the arm was stationary during the
  camera window, and `cameras.json` persisted no diagnostic capture provenance.
- Red gate: new offline calibration tests against that implementation produced `8 failed`:
  it captured before validating arm feedback, never performed the post-capture arm read/drift
  check, and did not accept the provenance arguments.
- Implementation: capture now validates fresh/healthy arm feedback and
  `max(abs(qvel)) <= runtime.arm.homing.velocity_convergence_rad_s` before and after the
  camera window. It accepts only when `max(abs(q_after - q_before)) <=
  runtime.arm.homing.convergence_rad`; rejected samples print their quality reason and do not
  mutate `error_state`, `estop_request`, or `SafetyState`. No calibration-specific threshold,
  state, motion revoke, or cross-device interpolation was added.
- Accepted solves persist a finite diagnostic-only `calibration_capture` object alongside the
  existing camera pose: stream geometry, intrinsics, distortion, selected method, sample count,
  position/rotation residual mean/std/max, and UTC timestamp. `CameraExtrinsics` continues to
  ignore this field when loading runtime extrinsics/intrinsics.
- Green: new calibration tests → `8 passed`; focused calibration/lifecycle/executor tests →
  `55 passed, 8 subtests passed`; final complete Real suite → `216 passed, 138 subtests passed`;
  `python -m compileall -q dexmani_real examples` → PASS. All tests used mocks, fake feedback,
  and temporary `cameras.json`; no RealSense, GUI, worker, robot motion, or calibration file in
  the repository was opened for writing.

## Phase 3 — complete (teleop lifecycle and explicit publication state)

- Current source evidence: the teleop child set `quit_requested` and immediately returned on its
  local second-Q/timeout paths. The unchanged supervisor correctly prioritizes e-stop and sticky
  fault over worker death, but a clean child exit before the parent polls Q can otherwise become
  `WORKER_DEATH`. `run_teleop_experiment()` also allowed `recording_enabled` with the hand
  disabled, despite the raw recording contract requiring arm7 + hand12. Seven of eight production
  `publish_command()` callsites relied on its optional lifecycle default.
- Red gates: the original loop returned after two fake ticks after a real Q path rather than
  waiting for parent `is_running=False`; no-hand recording reached startup preflight; grid and
  hand-home publication omitted their state; and the publication signature remained optional.
  The focused pre-fix run reported the expected five failures.
- 3A implementation: after Q has completed its recording decision and set `quit_requested`, the
  child produces no grid work or recording start, keeps the `policy` heartbeat alive through the
  existing outer loop, and returns only after parent shutdown clears `is_running`. The direct
  second-Q and timeout paths now enter that same wait. E-stop/error behavior and the supervisor
  priority code were not changed.
- 3B implementation: `run_teleop_experiment()` now rejects `recording_enabled and not hand_enabled`
  before task/preflight work, `RuntimeChannels.create()`, or any camera/recorder/worker startup.
  `--no-hand` remains usable for non-recording arm bring-up/debug; the CLI help and README state
  this explicitly.
- 3C audit and implementation: every actual publication side effect names its lifecycle state:
  calibration quit hold, replay warm-up, and hand homing use `ARMED`; calibration jog, keyboard
  jog, teleop grid, replay stream, and policy executor use `RUNNING`. `publish_command()` now
  requires `required_safety_state` as a keyword. The pure check APIs retain their optional state
  parameters unchanged.
- Tests: new `tests/test_teleop_lifecycle.py` drives a real Q control path through fake loop ticks,
  proves it stays alive until parent shutdown, and confirms `EXPLICIT_QUIT`; it also proves
  no-hand recording never reaches preflight/channel creation. Grid and hand-home tests assert
  their `RUNNING`/`ARMED` publications, and the publication signature test rejects omitted state.
- Green: relevant lifecycle/publication/calibration/replay/executor tests → `111 passed, 34 subtests
  passed`; complete Real suite → `220 passed, 138 subtests passed`; `python -m compileall -q
  dexmani_real examples` and `git diff --check` → PASS. All paths were fakes, mocks, or temporary
  artifacts; no physical example, SDK connection, camera, worker process, or robot motion ran.

## Phase 4 — complete (visual temporal consistency)

- Current source evidence: offline `ProcessingConfig.from_runtime()` admitted visual camera frames
  up to `runtime.camera.max_frame_age_s`, while the learned-policy observation path can impose the
  stricter `runtime.policy.max_input_age_s`. The deployment-only point-cloud producer also used the
  camera limit alone, even though it discards stale input before calling `build_point_cloud()`.
- Red gate: with camera/policy limits `(0.25, 0.15)`, new tests observed the old visual processing
  limit of `0.25`; a visual explicit override of `0.20` was accepted instead of rejected. The
  pre-fix focused test command reported `2 failed`.
- Implementation: visual `RGB`, `POINTCLOUD`, and `RGB_PC` processing now project their default
  camera-age limit as `min(camera.max_frame_age_s, policy.max_input_age_s)`. An explicit visual
  `max_camera_age_s` above the policy limit raises `ValueError` rather than being silently clamped.
  `JOINT` retains its camera-only value and has no added visual dependency. The realtime point-cloud
  worker now uses the same minimum before its existing pre-build stale-frame rejection; its geometry,
  sampling, IPC, and post-compute freshness behavior are unchanged.
- Tests: regressions cover all visual profiles, both `(0.25, 0.15) → 0.15` and
  `(0.10, 0.15) → 0.10`, over-wide visual override rejection, and the unchanged JOINT override.
- Green: focused timing tests → `3 passed`; related offline multimodal/observation tests →
  `38 passed, 75 subtests passed`; complete Real suite → `222 passed, 138 subtests passed`;
  `python -m compileall -q dexmani_real examples`, Black check of changed Phase 4 files, and
  `git diff --check` → PASS. No example, hardware SDK, device, worker process, or robot motion ran.

## Phase 5 — complete (minimal experiment provenance)

- Current source evidence: recorder already preserved arbitrary `provenance_<key>` attrs and teleop
  startup already passed resource hashes through `RecorderIOConfig`, but it did not preserve the
  canonical resolved config text or DexMani Real source revision/dirty state.
- Red gate: new provenance regressions on the prior implementation reported `3 failed, 1 passed`:
  there was no startup provenance constructor, so neither canonical config nor Git identity could be
  added to the existing recorder namespace.
- Implementation: recording startup now validates `sha256(runtime.canonical_json) == runtime.sha256`,
  takes one bounded Git snapshot (`rev-parse HEAD` plus `status --porcelain`), requires a 40-character
  lowercase commit, and supplies `resolved_config_json`, `dexmani_real_git_commit`, and
  `dexmani_real_git_dirty` (`"0"`/`"1"`) alongside unchanged resource hashes. It runs before channel
  creation, never in a realtime loop, and recorder writes the existing `provenance_*` attrs. Dirty
  records only identify their base revision; no source patch, untracked-file snapshot, sidecar, or raw
  schema change was added.
- Tests: canonical JSON/hash equality, clean/dirty status, invalid config hash/commit rejection,
  one startup snapshot before channel creation, HDF5 persistence, and existing resource provenance
  retention are covered with mocks and temporary HDF5 files.
- Green: recording/reader/teleop lifecycle tests → `11 passed`; complete Real suite →
  `227 passed, 138 subtests passed`; `python -m compileall -q dexmani_real examples`, Black check of
  the new test, and `git diff --check` → PASS. No example, hardware SDK, device, worker process, or
  robot motion ran.

## Phase 6 — complete (deterministic replay correctness)

- Current source evidence: tracking-lag selection used strict RMSE improvement, so an equal-RMSE
  constant signal retained the first scanned `-max_lag`. `replay_episode()` reached physical
  preflight before admitting its output target. Replay wording also described a generic exact command
  stream even though the current hand worker derives bounded SDK setpoints from logical hand targets.
- Red gate: on the prior evaluator, constant identical data selected `-7` and a period-three exact
  RMSE tie selected `-5`, rather than the minimum absolute lag. Occupied-output regressions reached
  the mocked physical preflight and had no admission helper, proving the rejection boundary was absent.
- Implementation: each lag candidate is now selected lexicographically by
  `(rmse, abs(lag), lag)`, with no excitation threshold or confidence model. Before preflight,
  channel allocation, or worker creation, replay admits only a missing or empty output directory;
  a file, nonempty directory, or dangling symlink returns `REJECTED` with the explicit
  alternate-`--output` message.
  Documentation now states that arm replay uses recorded published targets, while hand replay uses
  recorded logical targets that the current hand worker converts into bounded intermediate SDK
  setpoints; it does not claim exact actuator-setpoint replay.
- Tests: pure NumPy regressions cover constant zero, known delayed, and exact RMSE tie trajectories.
  Session mocks prove a file, nonempty directory, and dangling symlink reject before preflight or
  `RuntimeChannels.create()`, while missing and empty directories remain admissible.
- Green: focused replay tests → `15 passed, 2 subtests passed`; complete Real suite →
  `232 passed, 140 subtests passed`; `compileall`, Black check of the fresh Phase 6 files, and
  `git diff --check` → PASS. Black still suggests unrelated existing formatting changes in
  `replay/session.py` and `replay/replayer.py`, which were intentionally not reformatted. No
  hardware entry point, SDK connection, worker process, or robot motion ran.

## Phase 7 — complete (low-risk documentation cleanup)

- Current source evidence: active replay docstrings/comments still named the pre-refactor
  `replay_controller` and `replay_capture` modules, although the current owners are
  `EpisodeReplayer` and `ReplayRecorder`.
- Implementation: updated only those two active source descriptions. The two tracked completed
  namespace/integration progress checkpoints were retained: one is linked by its historical phase
  brief, and deleting either would require broader historical-link maintenance outside this cleanup.
- Green: active-source stale-name scan is empty; replay focused tests → `15 passed, 2 subtests
  passed`; replay package `compileall` and `git diff --check` → PASS. No runtime behavior, schema,
  safety policy, hardware entry point, SDK connection, worker process, or robot motion changed.

## Post-Phase 1–7 review — complete

- Reviewed current Real changes by schema/identity, calibration, lifecycle/publication, freshness,
  recording provenance, replay, and documentation boundary; reviewed the latest clean Policy
  exporter at `d2a5aecf6f08196458c0f517ba9c158ba3860c05` for its v7-only semantic propagation.
- Fixed one fail-closed replay admission gap: a dangling output symlink previously looked missing to
  `Path.exists()` and could reach physical preflight. It now rejects with the same occupied-output
  error before preflight, channel allocation, or worker creation.
- Removed stale active v12/v6 strings from the processing/export/processed-visualization examples,
  and replaced the remaining generic processed-replay "exact raw commands" description with the
  arm-target/logical-hand-target contract. Raw `hand_fingertip` fields remain documented because
  they are valid raw-recording provenance, not processed fingertip output. Historical completed
  refactor checkpoints remain because their phase briefs still link to them.
- Green: Real `compileall` plus full suite → `232 passed, 141 subtests passed`; latest Policy full
  suite → `82 tests OK`; active stale-term scan and `git diff --check` → PASS. All checks were
  offline; targeted Black/isort checks for cleanup files also passed. No real robot, camera, worker
  process, or physical replay ran.

## Phase 状态

| 单元 | 状态 | 证据/下一步 |
| --- | --- | --- |
| Baseline | 完成 | Execution baseline above |
| 0 | 完成 | `user_design.md` guardrail; focused test and diff check passed |
| 1A | 完成 | Real processed v13 / Zarr v7, unified fingertip derivation and frozen identity |
| 1B | 完成 | latest Policy export accepts only Zarr v7 and propagates validated semantics |
| 1C | 完成 | Real startup validates runtime point-cloud/fingertip identity before channels |
| 2 | 完成 | stationary before/after capture gate and diagnostic-only calibration provenance |
| 3 A/B/C | 完成 | Q parent-owned shutdown, no-hand recording rejection, explicit publication states |
| 4 | 完成 | visual and realtime point-cloud freshness share the deployment envelope |
| 5 | 完成 | canonical config and Git base/dirty provenance in existing raw namespace |
| 6 A/B/C | 完成 | deterministic lag tie, preflight output admission, accurate arm/hand replay wording |
| 7 | 完成 | active replay ownership names synchronized; historical completed checkpoints retained |
| Final | 软件回归完成 | manual hardware H1–H6 与最终 acceptance 仍待执行 |
| Hardware H1–H6 | 未执行 | 需要操作者另行明确启动 |

## 单元证据模板

```text
Phase / substep:
Owner / allowed files:
Current source evidence / invariant:
Red command / expected behavior failure / actual result:
Implementation / changed files:
Green and related commands / exact outcomes:
Diff review / preserved non-goals:
Unvalidated / blockers:
Parent acceptance / next step:
```

## 最终 acceptance（尚未执行）

| Invariant | Expected | Result / evidence |
| --- | --- | --- |
| Policy realtime software collision | unchanged / disabled | 未验证 |
| Teleop endpoint collision | unchanged / enabled | 未验证 |
| Teleop swept realtime collision | unchanged / disabled | 未验证 |
| Learned arm spike clip | unchanged | 未验证 |
| Policy XHand requirement | PolicySpec-owned | 未验证 |
| Point-cloud config mismatch | FAIL before motion | 未验证 |
| Point-cloud table mismatch | FAIL before motion | 未验证 |
| Fingertip derivation | all profiles from processed joint_state | 未验证 |
| Fingertip geometry mismatch | FAIL before motion | 未验证 |
| Processed schema | v13 | 未验证 |
| Policy Zarr schema | v7 | 未验证 |
| Deployment schema | v3 unchanged | 未验证 |
| Calibration moving sample | reject | 未验证 |
| Calibration stable sample | accept | 未验证 |
| Calibration rejection | no global FAULT | 未验证 |
| Teleop Q | clean parent-owned shutdown | PASS（纯软件 fake-loop + supervisor classification） |
| no-hand + recording | reject before worker startup | PASS（纯软件 spy） |
| Motion publication | explicit ARMED/RUNNING | PASS（production callsite audit + regressions） |
| Offline visual freshness | no looser than deployment | PASS（pure config regressions） |
| Pointcloud producer freshness | no looser than deployment | PASS（pure config regressions） |
| Raw canonical config | recoverable | PASS（canonical JSON/hash + HDF5 attr） |
| Raw git commit | persisted | PASS（40-char lowercase SHA + HDF5 attr） |
| Dirty source | explicitly marked | PASS（clean/dirty mock Git snapshots） |
| Replay lag exact tie | minimum absolute lag | PASS（pure NumPy regression） |
| Existing replay output | reject before hardware | PASS（preflight/channel spy） |
| read_latest semantics | unchanged | 未验证 |
| RealSense clock mapping | unchanged | 未验证 |

Deferred finding 仅保留为 follow-up：read_latest stale fallback、RealSense depth/color clock，
以及 workflow 列出的其他 deferred 项；本轮不实施。
