# CLAUDE.md

This file gives Claude Code repository-level working guidance.

Keep this file **small and stable**. It should describe how to work in the repository, not mirror the current implementation.

For detailed coding conventions, read `docs/CODE_STYLE.md`.  
For the agent-wide repository contract, read `AGENTS.md`.  
Use the source code, configuration and domain documentation as the authority for current implementation details.

## Working Principles

When modifying this repository:

1. Understand the relevant data flow before editing.
2. Make the smallest coherent change that solves the requested problem.
3. Prefer explicit code over framework-like abstractions.
4. Preserve existing behavior unless the task requires changing it.
5. Keep hardware ownership, process ownership and side effects obvious.
6. Do not mix unrelated refactoring into a focused change.
7. Treat real-robot execution as safety-sensitive.

The repository is research software. Optimize primarily for:

```text
readability
→ correctness
→ debuggability
→ reproducibility
→ extensibility
```

Do not optimize for hypothetical generality.

## Before Editing

Inspect the current worktree and the smallest relevant call path.

Useful operations include:

```bash
git status --short
rg -n "<symbol-or-key>" dexmani_real examples
```

Read neighboring producers and consumers when changing:

- shared data
- robot commands
- configuration
- coordinate transforms
- dataset fields
- lifecycle behavior

Do not assume documentation is more current than the implementation.

## Implementation Style

Follow `docs/CODE_STYLE.md`.

In particular:

- use precise domain names
- make units and coordinate frames explicit
- prefer small pure functions for math and transforms
- use classes for genuine state, ownership or lifecycle
- keep constructors cheap
- keep scripts and CLI entry points thin
- avoid pass-through wrappers
- avoid speculative interfaces, factories and registries
- centralize schemas and configuration defaults
- keep control loops free of unrelated blocking work

When existing code is complicated, simplify the local data flow before introducing another abstraction.

## Architecture Changes

Do not perform architectural redesign implicitly.

If a task can be solved with a local implementation, prefer that first.

Before introducing a new:

- manager
- service
- controller layer
- abstract base class
- protocol
- registry
- factory
- generic framework

verify that it represents a real boundary or removes demonstrated duplication.

One implementation normally does not require an abstraction layer.

## Hardware Safety

Do not run hardware-affecting programs unless the user explicitly requests it.

This includes operations that may:

- connect to a robot
- command motion
- home a device
- start teleoperation
- replay trajectories
- modify calibration
- initialize hardware through an SDK

Static inspection, compilation and focused offline checks are preferred during normal code editing.

Do not weaken a safety check to make an offline test pass.

## Verification

Use the least risky validation appropriate to the change.

Typical checks include:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
```

Run focused tests or small offline reproductions where useful.

Before finishing:

1. inspect the focused diff
2. check for unrelated modifications
3. check for unnecessary abstractions
4. verify important failure paths when relevant
5. report exactly what was and was not tested

Hardware validation should be reported separately from offline validation.

## Documentation Policy

Do **not** update this file simply because implementation details change.

Information that does not belong here includes:

- current schema version numbers
- exact queue or ring capacities
- hardware IP addresses or ports
- complete module/file maps
- individual configuration values
- current CLI names
- dataset field lists
- detailed state-machine internals

Put such information in the source code, README, configuration or focused documents under `docs/`.

Update `CLAUDE.md` only when the **repository-wide way Claude should work** materially changes.