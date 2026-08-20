# AGENTS.md — DexMani Real Agent Contract

This file defines the repository-wide working contract for coding agents.

It applies to the repository unless a more specific `AGENTS.md` exists deeper in the tree.

Keep this document stable. It defines **how agents should modify the repository**, not a snapshot of the current implementation.

Current implementation details belong in source code, configuration, README, and focused documents under `docs/`.

## 1. Source of Truth

When information conflicts, use this priority:

```text
runtime behavior and source code
→ schema / configuration definitions
→ focused domain documentation
→ README
→ agent guidance
```

Do not preserve outdated documentation behavior merely because it is described here.

For code style, follow `docs/CODE_STYLE.md`.

## 2. Repository Intent

DexMani Real is safety-sensitive robotics research software.

Changes should prioritize:

1. correctness
2. safety
3. readable data flow
4. debuggability
5. reproducibility
6. minimal conceptual complexity

General-purpose extensibility is not a goal by itself.

Prefer the simplest design that clearly represents the current system.

## 3. Before Making Changes

Always inspect the current worktree first.

Do not assume:

- the tree is clean
- adjacent files are unchanged
- documentation matches current behavior
- a named component still has the same implementation

Identify the smallest relevant call path.

For a value or behavior, determine:

```text
definition
→ producer
→ transformation
→ consumer
→ side effect
```

When changing a boundary, inspect both sides of that boundary.

Do not overwrite unrelated user changes.

## 4. Change Scope

Prefer a small vertical change over a broad horizontal refactor.

A good change normally modifies only the components necessary to preserve one coherent behavior.

Do not:

- reformat unrelated files
- rename unrelated APIs
- reorganize directories during a bug fix
- replace working code merely for stylistic uniformity
- introduce infrastructure unrelated to the requested task

If local cleanup is required to make the requested change understandable, keep it tightly scoped.

## 5. Architectural Principles

### 5.1 Explicit ownership

Every important mutable resource should have a clear owner.

This especially applies to:

- hardware SDK instances
- worker/process lifecycle
- shared state
- robot command publication
- recording output
- model runtime resources

Do not create multiple competing owners for the same resource.

### 5.2 Explicit boundaries

Treat these as important system boundaries:

```text
hardware → software
process → process
model → control
control → robot
runtime → storage
storage → runtime
user input → robot behavior
```

Validate important shape, dtype, unit, frame, freshness and state assumptions at these boundaries.

Avoid duplicating identical validation throughout internal helpers.

### 5.3 One source of truth

Do not create parallel definitions of:

- configuration defaults
- shared data layouts
- persisted schemas
- state semantics
- safety constraints

Extend the existing canonical definition.

### 5.4 Simple dependency direction

Dependencies should follow domain ownership.

Avoid circular imports and mutually dependent subsystems.

Do not solve dependency problems with hidden imports unless the dependency is genuinely optional.

### 5.5 Separate computation from side effects

Whenever practical, isolate pure:

- geometry
- transformations
- filtering
- validation
- trajectory computation
- data conversion

from:

- hardware IO
- process communication
- file IO
- visualization
- logging

This keeps important algorithms testable without hardware.

## 6. Hardware and Process Safety

Never run hardware-affecting code unless explicitly requested by the user.

Do not assume that a Python script is harmless merely because it is under `examples/`.

Imports and constructors should not introduce new hidden hardware side effects.

Prefer explicit lifecycle APIs such as:

```text
construct
→ start/connect
→ operate
→ stop/close
```

A live hardware SDK object should remain local to its owning component or process.

Do not pass live SDK objects through multiprocessing boundaries.

Do not bypass an existing safety boundary in order to simplify an implementation or make a test pass.

## 7. Coding Style

Follow `docs/CODE_STYLE.md`.

Important defaults:

- precise `snake_case` names
- `PascalCase` classes
- units in physical-value names when ambiguous
- coordinate frames visible in robotics data names
- grouped standard / third-party / project imports
- no wildcard imports
- one semantic responsibility per function
- pure helpers for mathematical operations
- early returns instead of deep nesting
- classes only when state or ownership justifies them
- cheap constructors
- thin CLI and example scripts
- explicit side effects
- concise comments explaining why, not what

Avoid generic names such as `Manager`, `Handler`, `Processor`, `Data`, or `Utils` when a domain-specific name exists.

## 8. Avoid Overengineering

Before adding any of the following, establish a concrete need:

- abstract base class
- protocol/interface
- factory
- registry
- plugin system
- generic event bus
- dependency-injection layer
- manager/service/controller hierarchy
- adapter that only forwards calls

Prefer direct composition.

Prefer three obvious functions over a generic subsystem that requires significant navigation to understand.

An abstraction should either:

1. establish a meaningful system boundary, or
2. remove demonstrated duplication.

Do not abstract hypothetical future implementations.

## 9. Configuration and Schemas

Configuration should be traceable to one canonical default.

Avoid independent fallback defaults scattered across workers and scripts.

Pass components the configuration they actually own rather than an unnecessarily large global configuration object.

Changes to shared or persisted data must be treated as boundary changes.

When modifying such data, identify all relevant:

```text
writers
→ representation
→ readers
→ persistence
→ downstream consumers
```

Do not silently change the meaning of an existing persisted field.

## 10. Concurrency

Concurrency should remain explicit.

For worker loops:

- keep the critical loop narrow
- avoid blocking unrelated IO
- make startup and shutdown bounded
- preserve failure visibility
- use appropriate monotonic timing for elapsed-time logic
- avoid silent exception swallowing

Do not add multiprocessing merely to isolate ordinary synchronous code.

Do not add synchronization without first identifying the actual shared mutable state.

## 11. Refactoring Policy

When simplifying complicated existing code, use this order:

1. clarify names
2. reveal data flow
3. isolate pure computation
4. remove duplication
5. simplify ownership
6. remove unnecessary wrappers
7. split genuinely independent responsibilities

Do not begin a cleanup by building a new abstraction framework around the existing code.

Preserve externally visible behavior unless the task explicitly changes it.

## 12. Verification

Use the smallest relevant offline validation first.

Typical safe checks include:

```bash
git status --short
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
```

Use focused unit-style scripts, mocks, fakes, or deterministic examples when appropriate.

For changes involving lifecycle, IPC, recording or safety-sensitive logic, consider relevant failure paths in addition to the normal path.

Do not claim hardware validation unless hardware execution was actually performed.

## 13. Finishing a Task

Before handing off:

1. inspect the final diff
2. preserve unrelated worktree changes
3. remove debugging artifacts
4. check for duplicated logic
5. check for unnecessary abstractions
6. run appropriate safe validation
7. report what changed
8. report what was tested
9. report important validation that was not performed

The final implementation should be understandable without requiring the agent conversation that produced it.

## 14. Documentation Maintenance

Do **not** modify this file for ordinary implementation changes.

The following should normally live elsewhere:

- module inventories
- exact filenames for every subsystem
- current schema version
- queue/ring capacities
- device addresses
- ports
- concrete dataset field lists
- current hardware parameters
- individual CLI commands
- temporary architectural migration state

Use:

- source code for executable truth
- configuration for tunable values
- README for user-facing navigation
- `docs/` for domain contracts and architecture
- `CODE_STYLE.md` for coding conventions

Update `AGENTS.md` only when the repository-wide **agent working contract, safety policy, or engineering philosophy** changes.