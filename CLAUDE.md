# CLAUDE.md

Follow [`AGENTS.md`](AGENTS.md) first. It is the repository-wide authority for
safety, engineering discipline, scope, and verification.

Use [`README.md`](README.md) for supported user-facing workflows and repository
orientation. Confirm current implementation behavior from source, schemas, and
resolved configuration before changing code.

Before executing Python, examples, imports, constructors, or helper scripts, inspect
the relevant path when hardware side effects are possible. Do not treat an entry
point as safe merely because it is convenient for verification.

For non-trivial changes, trace the smallest relevant
`producer → transformation → consumer → side effect` path, make one coherent change,
and validate offline first.

Keep this file Claude-specific and small. Do not duplicate repository contracts,
module inventories, implementation snapshots, or historical plans here.
