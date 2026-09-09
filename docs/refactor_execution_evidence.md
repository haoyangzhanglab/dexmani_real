# Refactor execution evidence

Execution follows the repository-local canonical plan
[`docs/refactor_execution_plan.md`](refactor_execution_plan.md), with
the user's 2026-09-09 updated agent assignments and Recorder/admission rules
taking precedence. Hardware execution is excluded from this software run.

## Final correctness follow-up — 2026-09-09

- `BASE_SHA`: `537c8e9f723c4038fc7790bd7649b7f7dea8f397`; fetch/prune and
  ff-only pull confirmed this was the latest `origin/main`. Initial worktree clean.
- Branch: `fix/final-refactor-followup`. Implementation: `terra-max`;
  independent source review: `luna-max`.
- Baseline in the `real_robot` Python environment: **344 passed, 94 subtests
  passed**; package/examples/tests compileall and diff check passed. The default
  shell has no `python` command; checks use the absolute interpreter below.
- Scope: restore per-episode analysis failure isolation; enforce one valid
  processed task identity before publication across producer, structural gate,
  full validator and Zarr inspection; repair the two stale current-format v13
  references; publish the canonical plan locally.
- The plan was restored from
  `origin/docs/refactor-plan-final-20260909:docs/refactor_execution_plan.md`.
  Its content is unchanged; Markdown hard breaks use backslashes instead of
  trailing spaces so the complete branch diff passes `git diff --check`.
  Superseded implementation guides were not restored. The planning branch's
  Codex workflow and bugfix policy duplicate rules already covered by the plan;
  neither was added.

### Fixes and final software gate

- Analysis isolation enumerates `FileNotFoundError`, `OSError`, `ValueError`,
  `KeyError`, `RuntimeError`, and `IndexError` only around the source analysis
  boundary. `recording/storage/reader.py` exposes missing files/datasets and
  invalid source structure; `recording/storage/video.py` exposes decoder
  runtime failures and frame-index errors; `dataset/clean.py` consumes HDF5
  keys and NumPy shapes/indices. Each rejection retains exception type/message.
  Program-control exceptions propagate. CLI auto-skip, direct-library blocking,
  explicit include blocking and explicit exclude-without-open remain distinct.
- Dataset task validation rejects non-strings, empty/untrimmed names, `unknown`
  and ASCII C0/DEL. Accepted task identities are resolved before writing; invalid
  raw identities, mixed accepted tasks and CLI output-task mismatch fail before
  staging. Writers and the invalid-frame report use the resolved identity.
  Library callers retain arbitrary output directory names. The export CLI's
  existing `.`/`..` path guard remains in place, with a regression test.
- Added regressions cover RuntimeError isolation and disposition, termination
  exception propagation, invalid global/annotation/raw task identity, mixed
  batches, expected-root mismatch, structural/full/export rejection, and both
  CLI task-path boundaries. Existing valid artifact/golden checks remain green.
- Final targeted checks: `test_dataset_admission.py` **31 passed**;
  `test_processed_v14.py` **26 passed, 4 subtests passed**;
  `test_zarr_v7_projection.py` **17 passed**. Combined: **74 passed, 4 subtests**.
  Export CLI regression is included in dataset admission tests.
- Final full suite: **368 passed, 98 subtests passed**. `compileall -q
  dexmani_real examples tests`, worktree diff check and complete BASE-to-final
  branch diff check: passed.
- Independent `luna-max` review: **APPROVE**, no blocker or major findings.
  Reviewer re-traced the source boundaries, reran the targeted/full gates and
  independently reopened the real Zarr described below.
- Protected areas unchanged: Safety, Freshness, Recorder lifecycle/protocol,
  IPC, Zarr gap tolerance and `dexmani_policy`; all explicit KEEP paths retained.
- Deferred findings: none. **MANUAL-HARDWARE-GATE PENDING**.

### Real Zarr actual-write gate

- Read-only source: `episodes_processed/pick_place_toy`, **61** processed-v14
  HDF5 files, all carrying task identity `pick_place_toy`.
- Existing `export_processed_hdf5_to_zarr` wrote an actual temporary store at
  `/tmp/dexmani-final-followup-zarr-ymk4454t/pick_place_toy.zarr`; the formal
  `datasets/pick_place_toy.zarr` was not modified.
- **60 accepted / 1 rejected**, **14,063 frames**. The only rejection remains
  `episode_20260827_224527`, under the unchanged whole-episode gap policy.
- Reopened the published store and ran `_validate_zarr`: passed. Its 60
  `episode_ends` are strictly increasing and exactly equal the cumulative
  accepted source lengths; final end is 14,063. Task identity is `pick_place_toy`.
- All ten expected arrays have length 14,063: `action`, `action_ee`,
  `camera_extrinsic`, `camera_intrinsic`, `contact_force`, `depth`,
  `fingertip_points`, `joint_state`, `point_cloud`, `rgb`. Shapes/dtypes and
  frozen v7 keys/attrs passed the existing validator.
- Local detailed evidence: `/tmp/dexmani-final-followup-zarr-ymk4454t/verification.json`.
  Temporary artifacts are software verification output, not committed datasets.
- No hardware was exercised: **MANUAL-HARDWARE-GATE PENDING**.

## Baseline

- Fetch, checkout main, and ff-only pull completed on 2026-09-09.
- `BASE_MAIN_SHA`: `668e1e9065d51fd85b52c3e0a8ce6a59db469f0d`.
- Initial worktree: clean.
- Python: `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`.
- Full offline suite: **293 passed, 94 subtests passed**.
- `compileall -q dexmani_real examples` and `git diff --check`: passed.
- Production package: 129 Python files / 40,587 physical lines.
- Examples: 13 Python files / 6,081 physical lines.
- Tests: 21 Python files / 5,990 physical lines.
- CI: **not configured at baseline HEAD**. GitHub retains an `offline-ci`
  workflow registration, but its source path returns 404 at main, no `.github`
  files are tracked at HEAD, and main has zero check runs. This is not CI green.

## Stage records

### Phase 0 — guardrails

- Base: `668e1e9065d51fd85b52c3e0a8ce6a59db469f0d`.
- Branch: `codex/refactor-phase0-guardrails`.
- Commit: `65c32d24e97ca0cc71e62f7841e546e1dcdae7c0`.
- PR: https://github.com/haoyangzhanglab/dexmani_real/pull/2 (merged).
- Integrated SHA: `5f0417639a5f4aa8c4e91633228fdec88669f721`.
- Ordinary implementation: luna-max; independent review: terra-max, approved.
- Ordinary focused tests: **12 passed** (keyboard, dataset admission, pointcloud).
- Full gate: **310 passed, 94 subtests passed**; compileall and diff check passed.
- Production and examples LOC unchanged. Four test files added/extended; no hardware run.
- Existing processed/Zarr golden regression coverage reused. No schema, IPC,
  thresholds, algorithms, or protected production paths changed.
- No separate bugfix commit in this stage; unsafe lifecycle repairs remain PR4.

Recorder characterization was implemented by a `gpt-6-astra / medium` agent
and approved by a separate `gpt-6-astra / medium` reviewer. Four requested
Recorder test files: **43 passed, 2 subtests passed** independently. Review
requested three extra checks (result transport failure, unreaped shutdown
without historical failure, live-finalizer handle retention/START refusal);
all were added and approved. Existing failure-history nonzero assertions remain
until PR4.

PR4 repair requirements confirmed during review: failed camera cleanup must
retain any live writer handle; a live finalizer must block START even when an
error has already been recorded. Phase 0 does not lock these unsafe old paths
as desired behavior.

### PR1 — leaf cleanup

- Base: `5f0417639a5f4aa8c4e91633228fdec88669f721`.
- Branch: `codex/refactor-pr1-leaf`.
- Implementation: luna-max; independent reviewer: terra-max.
- Independent review: approved; reviewer targeted suite **38 passed** plus
  compile/import and diff checks. PR: https://github.com/haoyangzhanglab/dexmani_real/pull/3
  (merged), integrated SHA `d2494cdf2840ec3a6ac120203f00eeb55040040d`.
- Commits: `ae16888` keyboard diagnostics; `9094855` rate stats;
  `efeaae6` unused profile; `4a0d323` EMA ownership; `f0ab11d` grid ownership.
- Targeted boundary suite: **119 passed, 26 subtests passed**.
- Full gate: **310 passed, 94 subtests passed**; compileall and diff check passed.
- Differential checks: EMA bitwise identical for 100 seeded inputs; all resolved
  command-limit fields identical; rate sleep/clock/deadline identical over 30
  normal/overdue/long-block/reset steps at 16/60/128 Hz.
- Production: 129 → 127 Python files; 40,587 → 40,286 physical lines (−301).
  Examples unchanged (13 files / 6,081 lines).
- Deleted mechanisms: passive keyboard motion diagnostics, rate-stat snapshots,
  unused arm device profile, health/smoothing modules, public limits cache DTO.
- Preserved: keyboard dispositions/action IDs, deadline behavior, feedback
  validation/count thresholds, TeleopConfig process DTO, startup-only numeric
  cache construction, EMA computation, all protected source/schema/IPC boundaries.
- No bugfix or design-policy change; hardware validation pending.

### PR2 — dataset single-pass

- Base: `d2494cdf2840ec3a6ac120203f00eeb55040040d`.
- Branch: `codex/refactor-pr2-dataset`.
- Implementation: terra-max; independent reviewer: astra-medium.
- Commits: `32f8d2e` implementation/tests; `235b277` docs/evidence.
- PR: https://github.com/haoyangzhanglab/dexmani_real/pull/4 (merged).
- Integrated SHA: `918a1719b92df43e0311cd68d8cbd74d1c92302e`.
- Independent review: approved. Targeted suite **50 passed**; full suite
  **318 passed, 94 subtests passed**; compileall and diff check passed.
- Numeric golden: 32 clean/stale-tactile AUDIT processed datasets match baseline
  exactly in dtype, shape and values. Existing processed/Zarr tests pass.
- Production: 127 files, 40,286 → 40,301 lines (+15); examples: 13 files,
  6,081 → 5,822 lines (−259). Combined production/examples decrease: 244 lines.
- Library defaults still block unannotated rejection; canonical CLI skips it.
  YAML membership (including omitted include) remains independent of task-name
  override; explicit excludes bypass unreadable raw sources.
- Tests cover default/skip/include/exclude/task-name admission, conflicting names,
  one real analysis per episode, one batch call per CLI/profile, user annotation
  filtering, AUDIT row retention, removed CLI flags, and report-before-publish.
- Review findings resolved: restore skipped names/reasons from the report;
  retain compare invalid-config exit 2; make the analysis spy reject duplicates.
- Removed STRICT and repeated audit/exclusion/report orchestration. All reports
  are produced in staging; Zarr export source and gap tests are unchanged.
- No independent pre-existing bugfix; no protected schema/version/IPC/threshold,
  research-array or durability changes. Hardware not exercised.

### PR3 — workflow

- Base: `918a1719b92df43e0311cd68d8cbd74d1c92302e`.
- Branch: `codex/refactor-pr3-workflow`.
- Implementation: terra-max; independent reviewer: astra-medium.
- PR: https://github.com/haoyangzhanglab/dexmani_real/pull/5 (merged).
- Integrated SHA: `afe387f3f7a0fb017b25f0216e06b49082f373fa`.
- Homing → pointcloud → keyboard completed in separate commits.
- Homing commit: `6088496`; stable configured caller API preserved. Independent
  review approved with **35 targeted tests passed**. Hand-first SDK acceptance,
  config projection, fresh feedback, cancellation, reset, fallback and audio
  ordering are covered; protected control/arm_homing implementation unchanged.
- Pointcloud source audit refined the implementation plan: reported builds and
  fresh-frame benchmarks already share `build_point_cloud_with_stats`;
  calibration deliberately uses `aligned_depth_points_in_base` before table
  cropping/outlier filtering/sampling. Diagnostic capture needs RGB/raw/metric
  depth, benchmark capture needs RGB/raw, calibration needs depth only. A
  forwarding helper or mode-based capture abstraction would add complexity or
  change validation/timing. Production remains unchanged; stronger offline
  characterization verifies these existing shared boundaries and distinct
  capture contracts. This is an already-satisfied design target, not an
  unfinished refactor or a deferred defect.
- Pointcloud independent review approved; **8 targeted tests passed**. New
  checks pin warmup/build arguments, calibration depth-only admission, both
  calibration branches, post-calibration frame/plane snapshot identity,
  disconnect and exact benchmark timing ranges. Production diff is zero.
- Pointcloud tests/audit commit: `1b82b0e`.
- Keyboard owner merge commit: `608850b`; independent refactor review approved,
  **9 targeted tests passed**, full checkpoint **328 passed, 94 subtests passed**.
  Command edges, held chars/arrows, repeatable raw events, sticky ESC/callback
  variants, listener health, release/quiesce and echo handling are covered.
  Command and raw-event capture are enabled only by consumers that drain them.
  `KeyboardState` is removed; camera and teleop callers share `KeyboardInput`.
- Review found a directly touched, pre-existing B1 bug: bounded stop/rollback
  could retain a live listener, but a subsequent start could overwrite its
  handle. Separate fix commit `5ea5615` retains ownership, refuses replacement,
  and gates subsequent callbacks after shutdown/rollback. In-flight external
  callbacks already dispatched before stop are not synchronously cancelled.
- B1 independent review approved: **11 targeted tests passed**, including
  explicit reap and same-owner restart; workflow target **30 passed**.
- Final full gate: **330 passed, 94 subtests passed**; compileall and diff check
  passed. Production: 127 files, 40,301 → 40,129 lines (−172); examples unchanged
  at 13 files / 5,822 lines. No protected safety/IPC/schema/threshold, pointcloud
  algorithm or action/observation research semantics changed; no hardware run.

### PR4 — Recorder ownership

- Base: `afe387f3f7a0fb017b25f0216e06b49082f373fa`.
- Branch: `codex/refactor-pr4-recorder`.
- Implementation: astra-medium; independent reviewer: a separate astra-medium.
- PR: https://github.com/haoyangzhanglab/dexmani_real/pull/6 (merged).
- Integrated SHA: `2316bf0c129c145716a954664f9791cbae28a715`.
- 4A commit `eb0b053`: additive synchronous `finish_episode` returns reserved
  paths for save/discard, returns None without active work, and raises after
  failure cleanup. Legacy async API remains for this migration checkpoint.
- Independent 4A review approved; **46 targeted tests, 2 subtests passed**;
  diff check passed. Unsafe old cleanup paths are not treated as recoverable
  until 4B establishes explicit resource-release evidence.
- Resource audit: camera thread termination alone cannot prove closure;
  encoder/depth close errors must retain failure/handles, including an encoder
  whose old close method can return early after a prior failed close.
- 4B commit `65d9d0d`: RecorderIO-owned local-Queue finalizer with pending state
  before launch, main-thread quarantine/heartbeat/control polling, explicit reap
  before terminal publication, and one shutdown owner. Typed transaction errors
  are recoverable only after resource release; timeout/unexpected errors remain
  fatal. Failed camera/HDF5/directory cleanup retains ownership and paths.
- Independent 4B review approved: **56 targeted tests, 2 subtests passed**;
  full checkpoint **343 passed, 94 subtests passed**, compileall/diff check passed.
  Real Event-blocked threads verify queued-result-before-exit and explicit join;
  missing results, thread-start failure, unsafe cleanup and original-deadline
  shutdown are covered. Legacy async remains only until the next checkpoint.
- 4C commit `601d468`: removed serializer async lifecycle, previous-stop state,
  registry/atexit and obsolete StopResult export; migrated direct tests. Final
  transaction is camera close/count → flush → final metadata → HDF5 close →
  one artifact validation → atomic publication. RecorderClient poll/join remain.
- Final independent review approved: **57 targeted tests, 2 subtests passed**;
  full **344 passed, 94 subtests passed**, compileall/diff check passed. Tests
  cover each missing/overflow/decode/write/camera failure → next START/save →
  clean worker exit, terminal transport failure with one send attempt/no second
  finalization, and partial START allocation failure remaining fatal.
- Production: 127 files, 40,129 → 40,036 lines (−93). Examples unchanged.
- Protected source diffs are empty for RecorderClient, IPC/raw schema,
  supervisor/process shutdown, dataset and atomic I/O; thresholds unchanged.
  Reserved failed/discard paths remain available for evaluation result.json.
- Review findings resolved: pending-first shutdown checks, unexpected-error
  classification, directory cleanup failure retention, real-thread test entry
  synchronization, explicit-join proof and terminal transport failure coverage.
  No separate unrelated bugfix; no hardware run.

### PR5 — contract audit

- Base: `2316bf0c129c145716a954664f9791cbae28a715`.
- Branch: `codex/refactor-pr5-contract-audit`.
- Commit: `5747fdf`; PR: https://github.com/haoyangzhanglab/dexmani_real/pull/7 (merged).
- Integrated SHA: `34291eb68d5677c28f2c9b9bdb7650e13d518f6d`.
- Implementation/audit: astra-medium; independent reviewer: a separate astra-medium.
- Decision: KEEP all inspected validators; production diff is zero. Full
  five-category check-level classification and producer/consumer evidence:
  [refactor_contract_audit.md](refactor_contract_audit.md).
- README/repo map explicitly identify applicable PolicySpec field semantics as
  Real public compatibility contract; fingertip uses policy ID, EEF algorithm ID.
- Independent review approved; 14 local audit links resolve. Author targeted
  **183 passed, 48 subtests passed**; independent focused **40 passed, 22 subtests**;
  full **344 passed, 94 subtests passed**, compileall/diff check passed.
- No FIXED_SCHEMA_IDENTITY candidate met all five deletion conditions. No
  production/examples/test changes, size change, bugfix or new validator;
  gap tolerance and all protected boundaries unchanged. Hardware not exercised.

### PR6 — unused telemetry and historical documentation

- Base: `34291eb68d5677c28f2c9b9bdb7650e13d518f6d`.
- Branch: `codex/refactor-pr6-cleanup`.
- Production commit: `d4cd915`; documentation commit: `a7710cc`.
- PR: https://github.com/haoyangzhanglab/dexmani_real/pull/8 (merged).
- Integrated SHA: `174170b703feed42dd4740368da16c58f6ddd7c9`.
- Implementation: luna-max; independent reviewer: terra-max.
- Removed StageTimer and its module/wiring, arm/hand log-only counters,
  CameraStreamWriter percentile/high-watermark collection, and pointcloud
  worker rolling performance buffers/counters. Whole-repository consumer
  searches found only those collectors/logs; no formal result consumers.
- Kept arm endpoint sequence fences and hand generation-change resets of
  accepted command references. Kept camera frame counts, errors, retained close
  resources and repeated-close joins; pointcloud source/sequence/freshness,
  publication and readiness; standalone builder/benchmark/snapshot statistics.
- Kept PolicyStats and result.json.metrics, feedback taxonomy (STALE has a
  distinct control disposition), status_print_interval (live grid consumer),
  all safety/IPC/schema/thresholds and fsync_tree/durability without profiling.
- Deleted four superseded implementation guides (3,931 lines):
  eef_tactile_observation_upgrade_guide.md, policy_rollout_simplification_guide.md,
  policy_formal_eval_followup_guide.md, raw_v25_runtime_simplification_guide.md.
  Their only filename references were within that group. Canonical schema,
  raw v24 migration, action-clipping and pointcloud docs and both incident
  records remain. Removed a pre-existing README link to an absent evaluation
  guide; current run_policy navigation/workflow remains.
- Updated active pointcloud documentation to remove claims that the deleted
  rolling worker percentile is available. Pure-build and capture-to-cloud
  benchmark scopes/numbers remain; total deployment latency requires a separate
  measurement and is not inferred from those benchmarks.
- Review findings resolved: restored unconditional repeated camera-writer join
  after an intermediate indentation mistake; corrected obsolete pointcloud
  telemetry documentation. No final lifecycle behavior change or separate code bugfix.
- Independent review approved: **111 passed, 28 subtests**; local documentation
  links resolve. Author targeted **109 passed, 2 subtests**; full **344 passed, 94 subtests**;
  compileall and diff check passed. No tests changed or hardware run.
- Production: 127 → 126 files; 40,036 → 39,797 lines (−239).
  Examples unchanged: 13 files / 5,822 lines. All conditional Phase 7 items DEFER.

## Integrated delivery

- Integrated software revision: `174170b703feed42dd4740368da16c58f6ddd7c9`.
  Final evidence changes after this revision affect only this document.
- Phase 0 and PR1–PR6 are merged sequentially into main: PRs #2–#8. Each branch
  was based on the preceding merged SHA recorded above; no stacked/unmerged
  dependency remains. GitHub reported CLEAN merge state and no status checks.
  CI remains **not configured**, not CI green.
- Final main offline gate: **344 passed, 94 subtests passed** (baseline 293/94);
  compileall for package/examples/tests and diff check passed. Source worktree
  was clean before this evidence update. No hardware or example program run.

| Scope | Baseline files / physical lines | Integrated files / physical lines | Line change |
|---|---:|---:|---:|
| Production package | 129 / 40,587 | 126 / 39,797 | −790 |
| Examples | 13 / 6,081 | 13 / 5,822 | −259 |
| Package + examples | 142 / 46,668 | 139 / 45,619 | −1,049 |
| Offline tests | 21 / 5,990 | 25 / 8,017 | +2,027 |

Counts use tracked Python files at baseline and integrated revisions; historical
Markdown deletions are excluded from production savings. Four obsolete guides
account for 3,931 separately removed documentation lines.

### Commit sequence

Merge commits and stage base SHAs are recorded above. Changes within each stage
are listed in order here, including evidence commits:

| Stage | Commits |
|---|---|
| Phase 0 | `65c32d2` |
| PR1 | `ae16888`, `9094855`, `efeaae6`, `4a0d323`, `f0ab11d`, `13d5639` |
| PR2 | `32f8d2e`, `235b277` |
| PR3 | `6088496`, `1b82b0e`, `608850b`, `5ea5615`, `840ca49` |
| PR4 | `eb0b053`, `65d9d0d`, `601d468`, `ab13bc4` |
| PR5 | `5747fdf` |
| PR6 | `d4cd915`, `a7710cc` |

### Removed and retained mechanisms

Removed passive diagnostics, unused profile/health/smoothing/timing wrappers,
public teleop limit projection, STRICT-only analysis, duplicate CLI audit/batch
orchestration and report rewrite, homing passthrough, duplicate keyboard state
owner, serializer async lifecycle/registry/atexit/previous-stop ceremony,
historical-failure exit poisoning and duplicate final artifact existence gate.
RecorderIO owns the one pending finalizer and retains fatal resource failures.

Intentionally retained TeleopConfig, grid startup numeric cache, KeyboardInput
held/event/repeat/release/ESC behavior, RecorderClient poll/join and failed/discard
reserved paths, AUDIT/HARD_ONLY temporal analysis and whole-episode Zarr gap
admission, formal result metrics, effective incidents, validators, feedback
failure distinctions and fsync durability. Existing shared pointcloud builder
and calibration preprocessing already met the Phase 3 target; additional
forwarding wrappers were rejected and numeric/CLI/benchmark tests expanded.

A full baseline-to-integrated diff is empty for protected control action,
publication/SafetyGate, deployment (including adapter/prediction/timing), IPC,
safety/supervisor/process shutdown, planning/assets, replay, AudioFeedback,
camera clock/source code, pointcloud algorithms, Zarr export, RecorderClient,
atomic I/O, historical v24 converter, `.claude/settings.local.json` and
`examples/xhand_control_example.py`. Changed worker/grid paths retain their
safety, generation, sequence and freshness decisions. No external dexmani_policy
repository was modified; no schema/version/IPC/threshold change was made.

No new design blocker was deferred. Conditional Phase 7 backend/hardware/legacy
compatibility cleanup remains DEFER because its prerequisite decisions were
not made; it is outside the software completion scope. The one local B1 listener
ownership fix is isolated in `5ea5615`. Hardware smoke, trained-policy execution
and deployment latency measurement remain outside the offline evidence.

### Final independent integration review

- **APPROVED** for `174170b703feed42dd4740368da16c58f6ddd7c9`; no blocking findings.
  **SOFTWARE COMPLETE**; **MANUAL-HARDWARE-GATE PENDING**.
- Reviewer: independent astra-medium, separate from the Recorder/contract
  implementation agent. Reviewed actual integrated source at `174170b`,
  aggregate baseline diff and final evidence; no production edits by reviewer.
- Independent offline boundary suite: **239 passed, 50 subtests passed**.
  Covers control safety, policy rollout/prediction/observation/tactile/producers,
  Recorder actual-thread and retained-resource guards, dataset admission/Zarr/
  deployment semantics, keyboard/homing and pointcloud.
- Re-traced publication → worker final SDK fence, generation resets, visual
  freshness/causal history, Recorder pending → thread → join → result and
  close → validate → atomic publication, and dataset single analysis → staged
  reports. Replay/audio/deployment protected behavior is retained across changed
  callers. PR6 hand reset and camera resource-release proofs remain intact.
- Reviewer independently confirmed the protected-path diff is empty and moved
  smoothing functions are AST-identical to baseline.
- Root final gate and protected-path comparison are recorded above. All 74 local
  links across README, repo map, evidence, contract audit and pointcloud docs
  resolve. CI and hardware limitations remain explicit.

### Changed files by stage

`A` = added, `M` = modified, `D` = removed. Git PR diffs retain full content.

**Phase 0**

```text
A	docs/refactor_execution_evidence.md
M	repo_map.md
A	tests/test_dataset_admission.py
A	tests/test_keyboard_input.py
A	tests/test_pointcloud_characterization.py
M	tests/test_recorder_queue_io.py
```

**PR1**

```text
M	dexmani_real/config/defaults.py
M	dexmani_real/teleop/config.py
M	dexmani_real/teleop/control_loop/action_proposal.py
M	dexmani_real/teleop/control_loop/grid.py
D	dexmani_real/teleop/control_loop/smoothing.py
D	dexmani_real/teleop/health.py
M	dexmani_real/teleop/keyboard_session.py
M	dexmani_real/teleop/loop.py
M	dexmani_real/utils/rate.py
M	docs/refactor_execution_evidence.md
M	repo_map.md
```

**PR2**

```text
M	README.md
M	dexmani_real/dataset/clean.py
M	dexmani_real/dataset/contracts.py
M	dexmani_real/dataset/processing.py
M	dexmani_real/dataset/quality.py
M	docs/data_schema.md
M	docs/refactor_execution_evidence.md
M	examples/process_episodes.py
M	repo_map.md
M	tests/test_dataset_admission.py
```

**PR3**

```text
M	dexmani_real/calibration/camera/motion.py
M	dexmani_real/calibration/camera/session.py
M	dexmani_real/control/jog.py
M	dexmani_real/runtime/operator_input.py
M	dexmani_real/teleop/homing.py
M	dexmani_real/teleop/keyboard_session.py
M	docs/refactor_execution_evidence.md
M	repo_map.md
M	tests/test_keyboard_input.py
M	tests/test_pointcloud_characterization.py
A	tests/test_teleop_homing.py
```

**PR4**

```text
M	README.md
M	dexmani_real/recording/__init__.py
M	dexmani_real/recording/io_worker.py
M	dexmani_real/recording/recorder.py
M	dexmani_real/recording/storage/camera_writer.py
M	docs/refactor_execution_evidence.md
M	repo_map.md
M	tests/test_raw_v25_recording.py
M	tests/test_recorder_io_boundary.py
M	tests/test_recorder_queue_io.py
```

**PR5**

```text
M	README.md
A	docs/refactor_contract_audit.md
M	docs/refactor_execution_evidence.md
M	repo_map.md
```

**PR6**

```text
M	README.md
M	dexmani_real/recording/storage/camera_writer.py
M	dexmani_real/robot/arm_worker.py
M	dexmani_real/robot/hand_worker.py
M	dexmani_real/sensor/pointcloud_worker.py
M	dexmani_real/teleop/control_loop/grid.py
D	dexmani_real/teleop/control_loop/timing.py
M	dexmani_real/teleop/loop.py
D	docs/eef_tactile_observation_upgrade_guide.md
M	docs/pointcloud_pipeline.md
D	docs/policy_formal_eval_followup_guide.md
D	docs/policy_rollout_simplification_guide.md
D	docs/raw_v25_runtime_simplification_guide.md
M	docs/refactor_execution_evidence.md
M	repo_map.md
```

## Hardware gate

MANUAL-HARDWARE-GATE PENDING. No hardware was exercised. The manual checklist
remains startup/readiness, H home/audio, VR and keyboard teleop, ESC/e-stop,
record/save/discard, raw inspection, raw → processed → Zarr, replay, policy
shadow, and controlled run/eval.
