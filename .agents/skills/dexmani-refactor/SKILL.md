---
name: dexmani-refactor
description: Execute or resume the staged dexmani_real simplification plan with tiered subagents, phase acceptance, compact checkpoints, and automatic progression.
---

# DexMani staged refactor

Use this workflow only when the user asks to execute or resume
`docs/dexmani_real 删简重构执行方案.md`. The main agent owns phase order,
integration, acceptance, and the final safety judgment. Subagents receive bounded
tasks; they do not redefine the plan.

## Start or resume

1. Read the plan and the applicable `AGENTS.md` completely.
2. Run `git status --short` and `git rev-parse HEAD`. Preserve unrelated changes.
3. Inspect the focused source path and both sides of each changed boundary. Treat
   source and schemas as authoritative when they disagree with the plan.
4. If goal tools are available, create or continue one goal covering the whole
   requested phase range. Do not set a token budget unless the user supplied one.
5. Infer the next phase from the current source, diff, tests, prior checkpoint,
   and commits. Never redo an accepted phase merely because the chat was compacted.

Use the repository at `/home/zhanghaoyang/Desktop/dexmani_policy` to inspect the
Policy public contract. Do not modify it unless the user explicitly puts that
repository in the write scope.

Run Python only through the `real_robot` Conda environment. Use commands such as:

```bash
conda run -n real_robot pytest -q ...
conda run -n real_robot python -m compileall -q dexmani_real
```

A CUDA failure inside the sandbox means the environment cannot expose the local
GPU; it is not evidence that the machine or code lacks CUDA. Do not add CPU
fallbacks, compatibility branches, or temporary code for that condition. Request
the narrow sandbox escalation needed for a required GPU check, or record the check
as not run.

## Choose an agent

Choose by the actual task, not by file count. Use the least expensive tier that
can make the decision reliably:

- `sol-high`: safety-critical or ambiguous work across policy/control, IPC,
  concurrency, lifecycle, command authority, hardware fences, or several owners.
- `terra-xhigh`: a clear multi-file refactor, observation/data ownership change,
  configuration cleanup, or focused integration task with known invariants.
- `luna-max`: narrow mechanical deletion, call-site inventory, documentation,
  focused test execution, or an audit with no architectural decision.

The default phase owner is:

| Phase | Owner | Reason |
|---|---|---|
| 0 | `sol-high` | Freeze safety behavior before deletion. |
| 1 | `sol-high` | Cross-repository contract ownership. |
| 2 | `terra-xhigh` | Clear array-ownership and copy reduction. |
| 3 | `sol-high` | Publication races and final SDK authority. |
| 4 | `sol-high` | Process lifecycle, health, and fail-closed behavior. |
| 5 / 5B | `sol-high` | Command identity and actuator progress semantics. |
| 6 | `sol-high` | Learned-hand physical safety ordering. |
| 7 | `terra-xhigh` | Runtime/config/cleanup ownership. |
| 8 | `terra-xhigh` | Data integrity versus audit separation. |
| 9 | `luna-max` or `terra-xhigh` | Only justified, explicitly selected reductions. |

Upgrade the tier when investigation reveals a harder boundary. Do not downgrade a
safety decision merely to save tokens.

Use one write-owning agent at a time. Parallelize only independent read-heavy
inspection, focused tests, or audits whose output can be summarized. Never let two
agents edit overlapping files. The main agent reviews the focused diff and resolves
conflicts; a subagent's completion is not phase acceptance.

## Phase loop

Process phases strictly in plan order. Do not mix runtime and persisted-schema
changes. For each phase:

1. Define one bounded outcome and the invariants that must remain true.
2. Search every symbol being removed across production, tests, docs, and examples.
3. Assign the smallest coherent task to the selected tier.
4. Prefer deletion, direct data flow, and local functions. Add no dependency,
   framework, compatibility shim, temporary path, speculative abstraction, or
   defensive check without a demonstrated owner-level invariant.
5. Inspect the focused diff. Reject changes that add duplicate ownership, hidden
   state, unrelated formatting, or more mechanism than they remove.
6. Run the phase's focused tests once, then one full
   `conda run -n real_robot pytest -q`. Repeat or broaden checks only after a
   relevant failure or new change. Run `git diff --check` for every phase;
   compile or type-check only when the changed boundary needs it.
7. Compare the offline result with the phase acceptance criteria and the permanent
   safety, research-correctness, and reliability invariants in the plan. Add any
   required live-robot cases to the consolidated hardware queue; do not run them
   between phases.
8. If offline-accepted, create the compact checkpoint below and proceed to the next
   phase without asking for confirmation. Keep each phase independently committable,
   but create commits only when the invoking request authorizes commits.

Do not add tests that only pin the new implementation shape. Add or retain tests
for observable safety or data semantics. The full suite is intentionally run once
per phase because it is small; do not rerun it after an unchanged result.

## Execution clarifications

Apply these source-audited clarifications when the corresponding phase is reached:

- Phase 1: call Real/Policy compatibility validation exactly once in
  `run_policy_deployment`, before shared-memory/process creation. Downstream worker
  specs, configs, and loops trust that boundary. Policy-owned invariants remain in
  the Policy runtime.
- Phase 4: do not blindly remove child-side readiness waits. The parent needs a
  bounded wait that combines ready, fault, liveness, and timeout. A child that has
  no process handles may use a flag-only wait while the parent supervises it.
- Phase 5B: first audit `_CommandProgress`. If it already expresses the target
  actuator-progress semantics, keep it and remove only demonstrated duplication.
- Phase 7: do not delete partial-allocation cleanup bookkeeping until an offline
  idempotent cleanup check proves it obsolete.
- Phase 8: ordinary episode reads may move expensive artifact hashing to explicit
  audit, but processed replay must continue to hard-reject a raw `data.h5` identity
  mismatch. URDF, SRDF, or config hash mismatches may become provenance warnings
  only after the current geometry and full physical preflight validate the complete
  trajectory.
- Phase 9 is not automatic cleanup. Execute an item only when current evidence
  shows a net simplification without schema migration, IPC replacement, or merging
  inference with execution; otherwise record it as intentionally deferred.

## Acceptance and compact boundary

After each phase, produce exactly one concise checkpoint in this shape:

```text
PHASE_CHECKPOINT
phase: <number and name>
state: accepted | offline_accepted_hardware_pending | blocked
base_head: <sha>
scope: <files and owning boundary>
removed: <state, branch, copy, or duplicate owner removed>
preserved: <safety and research invariants>
validation: <commands and exact result>
not_validated: <hardware, GPU, or external checks not run>
hardware_queue: <new live-robot cases contributed by this phase, or none>
worktree: <unrelated user changes still present>
next: <next phase and selected tier>
```

This checkpoint is the only phase detail that needs to survive context compaction.
If the host exposes agent-callable compaction, compact immediately after writing the
checkpoint. Otherwise continue automatically and let the host compact when needed;
interactive `/compact` is a user command, so do not fake it with a shell command,
session restart, state file, or summary code. On resume, trust the checkpoint only
after reconciling it with source, Git state, and test results.

## Consolidated hardware gate

From Phase 3 onward, defer planned live-robot checks until all requested offline
phases and the final offline suite have passed. Each phase contributes only the
smallest new cases needed to a cumulative hardware queue. Deduplicate that queue
against the plan's hardware regression gate, then execute it in one final hardware
session so setup, homing, operator attention, and risk exposure are not repeated.

Never run hardware-affecting code without explicit user authorization. Without that
authorization, complete the offline workflow, leave the affected phases as
`offline_accepted_hardware_pending`, and report one consolidated pending hardware
campaign. Do not claim merge acceptance or hardware validation.

Run an earlier hardware check only when offline evidence shows that its result is
required before the next code change can be reasoned safe. Treat that as a stop
condition, explain the exact dependency, and request explicit authorization instead
of continuing speculatively. Do not add compatibility or temporary code to bypass
the gate.

## Stop conditions

Continue autonomously through ordinary failures: diagnose, make the smallest
in-scope correction, and rerun only the affected check. Stop the current phase and
report the smallest next investigation when the plan's stop conditions apply,
including an unexpected baseline failure, an unknown production writer, removal of
the only safety guard, unproved causal parity, required schema migration or new
dependency, unknown vendor behavior, or a hardware-dependent semantic decision.

At final completion, inspect `git status --short`, the focused diff, and the final
test result. Report accepted phases, the deduplicated final hardware campaign,
deferred Phase 9 items, and any validation that was not run.
