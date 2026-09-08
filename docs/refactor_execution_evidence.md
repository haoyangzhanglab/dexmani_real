# Refactor execution evidence

Execution follows the canonical plan on
`origin/docs/refactor-plan-final-20260909:docs/refactor_execution_plan.md`, with
the user's 2026-09-09 updated agent assignments and Recorder/admission rules
taking precedence. Hardware execution is excluded from this software run.

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

## Hardware gate

MANUAL-HARDWARE-GATE PENDING. No hardware was exercised. The manual checklist
remains startup/readiness, H home/audio, VR and keyboard teleop, ESC/e-stop,
record/save/discard, raw inspection, raw → processed → Zarr, replay, policy
shadow, and controlled run/eval.
