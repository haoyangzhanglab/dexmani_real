# AGENTS.md — DexMani Real Agent Contract

This is the repository-wide contract for coding agents. It applies everywhere
unless a deeper `AGENTS.md` explicitly overrides it.

Keep this file stable. It defines how to work in the repository, not the
current module inventory or a snapshot of runtime parameters.

## 1. Authority and document ownership

When information conflicts, use this order:

```text
runtime behavior and source code
→ schemas and configuration definitions
→ README and focused implementation documentation
→ repo_map.md
→ agent guidance
```

Each top-level document has one job:

- `AGENTS.md`: repository-wide engineering and safety contract.
- `CLAUDE.md`: concise Claude Code entry point; it must defer to this file.
- `code_style.md`: concrete coding conventions for this personal research codebase.
- `README.md`: user-facing setup, architecture, workflows, and commands.
- `repo_map.md`: current file inventory and ownership map.

Do not preserve stale documentation behavior when the implementation disagrees.

## 2. Engineering priorities

DexMani Real is safety-sensitive robotics research software. Prioritize:

1. correctness and fail-closed behavior
2. hardware and operator safety
3. explicit ownership and readable data flow
4. debuggability and reproducibility
5. minimal conceptual complexity

General-purpose extensibility is not a goal by itself. Prefer the simplest
design that accurately represents the current system.

## 3. Required workflow

Before editing:

1. Run `git status --short` and preserve unrelated user changes.
2. Read the smallest relevant entry point and call path.
3. Trace important values through:

   ```text
   definition → producer → transformation → consumer → side effect
   ```

4. Inspect both sides of every changed boundary: hardware/software,
   process/process, model/control, control/robot, runtime/storage, and
   user-input/robot-behavior.
5. Confirm current behavior from source and configuration rather than filenames,
   comments, or old documentation.

While editing:

- Make the smallest coherent vertical change.
- Preserve externally visible behavior unless the task explicitly changes it.
- Do not reformat, rename, relocate, or refactor unrelated code.
- Keep entry scripts thin; put reusable behavior in the owning package.
- Remove only artifacts made obsolete by the current change.

Before handoff:

1. Inspect the focused diff and final worktree status.
2. Check for duplicated logic, hidden ownership, and unnecessary abstractions.
3. Run the smallest safe validation proportional to the risk.
4. Report what changed, what was validated, and what was not validated.

## 4. Safety and side effects

Never run hardware-affecting code unless the user explicitly requests it. Treat
all scripts under `examples/` as potentially hardware-affecting until inspected.

Hardware-affecting behavior includes connecting to a device, commanding motion,
homing, teleoperation, physical replay, calibration writes, and constructors or
imports that initialize an SDK.

Required ownership rules:

- A live hardware SDK object stays inside its owning driver or worker process.
- Do not pass SDK objects across multiprocessing boundaries.
- Robot commands must pass through the existing safety and lifecycle boundaries.
- Do not weaken validation, freshness checks, collision checks, generation
  checks, or fail-closed behavior to make a test pass.
- Constructors should be cheap; use explicit `start`/`connect` and `stop`/`close`
  lifecycle operations.

Prefer offline inspection, compilation, deterministic pure-function checks,
fakes, and mocks during normal development. Never claim hardware validation
unless real hardware was actually exercised.

## 5. Architecture and ownership

Every important mutable resource must have one clear owner, especially:

- hardware SDK instances
- process lifecycle and shutdown
- shared-memory state
- robot command publication
- recording output
- model runtime resources

Keep dependency direction aligned with domain ownership. Avoid circular imports,
hidden imports used only to escape dependency problems, and competing owners for
the same state.

Separate pure computation—geometry, transforms, filtering, validation,
trajectory generation, and data conversion—from hardware IO, IPC, file IO,
visualization, and logging whenever practical.

For worker loops:

- keep the critical loop narrow
- use monotonic time for elapsed-time and freshness logic
- avoid unrelated blocking IO
- keep startup and shutdown bounded
- expose failures; do not silently swallow exceptions
- add synchronization only for identified shared mutable state

Do not add multiprocessing merely to isolate ordinary synchronous code.

## 6. Boundaries, configuration, and schemas

Validate shape, dtype, units, coordinate frame, freshness, lifecycle state, and
provenance at the boundary that owns the contract. Avoid repeating identical
validation throughout internal helpers.

Maintain one source of truth for:

- runtime defaults and resolved configuration
- shared-memory layouts
- persisted episode and processed-data schemas
- safety state semantics and command limits
- robot model and calibration resource paths

Pass components the configuration they own rather than an unnecessarily broad
global object. Do not add fallback defaults in workers or scripts when a
canonical definition already exists.

Treat shared-memory and persisted-data changes as boundary changes. Inspect all
writers, representations, readers, persistence paths, and downstream consumers.
Never silently change the meaning of an existing persisted field.

## 7. Implementation style

Follow [`code_style.md`](code_style.md). Its central rule is to keep the two
research paths—real-data collection and model deployment—direct, readable, and
safe rather than turning the repository into a general robotics platform.

Required defaults include:

- precise `snake_case` functions and values; `PascalCase` classes
- units in physical-value names when ambiguous
- coordinate frames visible in robotics data names
- standard, third-party, then project import groups; no wildcard imports
- one semantic responsibility per function
- pure helpers for mathematical operations
- early returns instead of deep nesting
- classes only when state, ownership, or lifecycle justifies them
- concise comments explaining why, invariants, or safety rationale
- explicit side effects and bounded failure behavior

Prefer domain-specific names over generic `Manager`, `Handler`, `Processor`,
`Data`, or `Utils` names when a precise alternative exists.

Before adding an abstract base class, protocol, factory, registry, plugin system,
event bus, dependency-injection layer, or manager/service/controller hierarchy,
show that it establishes a real boundary or removes demonstrated duplication.
Do not abstract hypothetical future implementations.

When simplifying existing code, prefer this order:

1. clarify names
2. reveal data flow
3. isolate pure computation
4. remove duplication
5. simplify ownership
6. remove unnecessary wrappers
7. split genuinely independent responsibilities

## 8. Verification

Start with the least risky relevant checks:

```bash
git status --short
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
```

Add focused offline checks for changed pure functions, schemas, readers,
lifecycle branches, IPC contracts, recording transactions, and safety failure
paths as appropriate. The repository does not currently have a general unit
test suite, so do not treat example programs as tests.

## 9. Documentation maintenance

Documentation changes must follow their ownership boundary:

- Update `README.md` when supported user workflows, setup, or stable architecture
  navigation changes.
- Update `code_style.md` only when concrete repository-wide coding conventions
  change.
- Update `repo_map.md` when tracked files are added, removed, moved, or change
  primary responsibility.
- Update `CLAUDE.md` only for Claude-specific working guidance.
- Update this file only when the repository-wide agent contract, safety policy,
  or engineering philosophy changes.

Implementation snapshots—schema versions, queue capacities, device addresses,
ports, dataset fields, CLI flags, and current hardware parameters—belong in
source, configuration, README, or `repo_map.md`, not in agent/style guidance.
