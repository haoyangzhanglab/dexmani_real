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

## Hardware gate

MANUAL-HARDWARE-GATE PENDING. No hardware was exercised. The manual checklist
remains startup/readiness, H home/audio, VR and keyboard teleop, ESC/e-stop,
record/save/discard, raw inspection, raw → processed → Zarr, replay, policy
shadow, and controlled run/eval.
