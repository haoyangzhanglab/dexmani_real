# CLAUDE.md

Claude Code must read and follow [`AGENTS.md`](AGENTS.md) before modifying this
repository. `AGENTS.md` is authoritative for safety, scope, style, verification,
and documentation ownership; this file only adds a concise Claude-specific
workflow.

## Start here

- Use [`README.md`](README.md) for supported workflows and architecture.
- Use [`repo_map.md`](repo_map.md) to find the current owner of a file or behavior.
- Use [`code_style.md`](code_style.md) for concrete Python and research-code style.
- Use source and resolved configuration as the final authority.
- When writing, reviewing, or refactoring code, apply the project-local
  `.claude/skills/karpathy-guidelines/SKILL.md` guidance where available.

## Working loop

1. Inspect `git status --short`; never overwrite unrelated user changes.
2. Identify the smallest relevant call path and trace producer → representation
   → consumer → side effect.
3. State material assumptions and success criteria before a non-trivial edit.
4. Make the smallest coherent change; avoid speculative abstractions and nearby
   cleanup.
5. Validate offline first, inspect the final diff, and report unperformed checks.

Useful read-only discovery commands:

```bash
rg -n "<symbol-or-key>" dexmani_real examples
rg --files
git diff -- <focused-paths>
```

Typical safe validation:

```bash
python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
git status --short
```

## Safety reminders

Do not run anything that may connect to xArm7, XHand, RealSense, or Quest/HTS,
command motion, home hardware, replay an episode, or write calibration unless
the user explicitly asks. An `examples/` script, import, or constructor is not
assumed safe until inspected.

Never bypass `SharedStorage`, the unified safety gate, worker-side validation,
generation/freshness checks, collision checks, or verified shutdown to simplify
an implementation or make a check pass.

Keep this file small and stable. Current filenames and module responsibilities
belong in `repo_map.md`; coding conventions belong in `code_style.md`; commands
and supported workflows belong in `README.md`.
