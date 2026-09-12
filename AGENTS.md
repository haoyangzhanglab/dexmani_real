# AGENTS.md — DexMani Real Engineering Contract

This file is the repository-wide contract for coding agents. It applies throughout
the repository unless a deeper `AGENTS.md` explicitly overrides it.

Keep this file stable. It defines how to investigate, modify, and validate the
repository; it is not an inventory of current modules or runtime parameters.

## 1. Sources of truth

Separate current system behavior from agent policy:

```text
Current behavior: source code + schemas + resolved configuration → README
Agent behavior:   AGENTS.md → tool-specific adapters
```

Source defines what the system currently does. This file defines how an agent may
investigate, modify, or execute it. When documentation and implementation disagree,
verify the current behavior from source/configuration and fix stale documentation
only when it is in scope.

Use `README.md` for supported user-facing workflows and repository orientation, not
as an implementation specification.

## 2. Hardware safety

DexMani Real controls physical robotics hardware. Do not execute hardware-affecting
code unless the user explicitly authorizes that operation.

Hardware-affecting behavior includes device connection or discovery that opens a
live SDK/device, robot motion, homing, teleoperation, physical replay, policy
rollout, and calibration writes. Treat `examples/`, imports, constructors, and
helper scripts as potentially hardware-affecting until their execution path is
inspected.

- Prefer source inspection and offline checks during normal development.
- Do not bypass or weaken existing safety, collision, freshness, lifecycle,
  generation, command-validation, or fail-closed boundaries to make a change pass.
- Keep live SDK objects inside their owning driver/worker boundary; do not move them
  across process boundaries.
- Never claim hardware validation unless real hardware was actually exercised.
- Report hardware/CUDA/device validation separately from offline validation.

## 3. Working method

Before editing, inspect the smallest relevant entry point and trace important values
through the real call/data path:

```text
definition → producer → transformation → consumer → side effect
```

For a changed boundary, inspect both sides. Relevant boundaries include process/IPC,
runtime/storage, policy/control, control/robot, configuration/runtime, and
user-input/physical side effects.

Confirm behavior from current source, schemas, and resolved configuration rather
than filenames, comments, historical plans, or stale documentation.

While editing:

- make the smallest coherent change that solves the assigned problem;
- preserve unrelated user changes and externally visible behavior unless the task
  explicitly changes that behavior;
- avoid speculative features, abstractions, configurability, and nearby cleanup;
- follow the existing ownership and dependency direction;
- remove only artifacts made obsolete by the current change;
- prefer the simplest design that makes ownership, data flow, and failure behavior
  explicit.

Scope is defined by the requested behavior, not by an arbitrary file count. A small
vertical change may touch the producer and consumer required to keep one contract
coherent, but every changed file must be necessary for that same goal.

## 4. Boundaries and ownership

Important mutable resources need one clear owner, especially hardware SDK state,
process lifecycle, shared state, robot command publication, recording output, and
model runtime resources.

Validate contracts at their owning boundaries. For robotics/data interfaces, check
relevant shape, dtype, units, coordinate frame, freshness/lifecycle state, and
persisted semantics.

Maintain one source of truth for each contract. Do not add fallback defaults or
parallel interpretations when a canonical configuration/schema already exists.

Treat IPC and persisted-data changes as cross-boundary changes: inspect writers,
representations, readers, persistence, and downstream consumers. Never silently
change the meaning of an existing persisted field.

Keep pure computation separate from device/IPC/file side effects when that improves
ownership and testability, but do not introduce abstraction layers without a real
boundary or demonstrated duplication.

For adjacent repositories such as `dexmani_policy`, consume public interfaces and
contracts rather than reaching into private implementation details.

## 5. Verification and handoff

Start with the smallest safe checks relevant to the change. Repository-wide low-cost
checks are:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git status --short
```

Run focused offline validation that actually exists for the changed subsystem when
useful. Do not run example programs as tests, and do not invent or restore test
infrastructure merely to satisfy a generic checklist.

Before handoff:

1. inspect the focused diff and final worktree status;
2. check for accidental scope expansion, duplicated logic, hidden ownership, and
   stale comments/docs introduced by the change;
3. report what changed, what was validated, and what remains unverified.

A skipped or failed check is not a passing check.

## 6. Documentation discipline

Permanent documentation should contain information with the same lifetime as the
file that owns it.

- `README.md`: project orientation, setup, canonical user workflows, stable
  architecture, and user-visible side effects/outputs.
- `AGENTS.md`: repository-wide agent safety and engineering contract.
- `CLAUDE.md`: Claude-specific adapter only.
- `.codex/agents/`: Codex role behavior only.
- source/config/schema: current implementation details and volatile contracts.

Do not use permanent repository docs or agent prompts as storage for one-off
implementation plans, incident narratives, generated evidence, migration history,
experiment statistics, cleanup checklists, or snapshots of schema versions, error
codes, queue sizes, CLI internals, and other fast-changing implementation facts.
Use source, Git history, issues/PRs, or experiment artifacts for those records.
