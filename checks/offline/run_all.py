"""Run every offline pass-item check and summarize pass/fail.

Each ``check_*.py`` runs in its own subprocess so a crash in one cannot
corrupt another's shared-memory state, and so shared-memory names are released
between checks.  Exit status is non-zero when any check fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

# Export the repo root so subprocess checks can import ``dexmani_real``.
_ENV = dict(os.environ)
_env_pythonpath = _ENV.get("PYTHONPATH", "")
_ENV["PYTHONPATH"] = (
    str(_REPO_ROOT) + (os.pathsep + _env_pythonpath if _env_pythonpath else "")
)


def _checks() -> list[Path]:
    return sorted(_HERE.glob("check_*.py"))


def main() -> int:
    checks = _checks()
    if not checks:
        print("run_all: no check_*.py scripts found", file=sys.stderr)
        return 2

    failures: list[tuple[str, int]] = []
    for check in checks:
        proc = subprocess.run(
            [sys.executable, str(check)],
            cwd=str(_REPO_ROOT),
            env=_ENV,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            print(f"PASS  {check.name}")
        else:
            failures.append((check.name, proc.returncode))
            print(f"FAIL  {check.name} (exit {proc.returncode})")
            if proc.stdout.strip():
                print("  stdout: " + proc.stdout.strip().replace("\n", "\n  "))
            if proc.stderr.strip():
                tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
                print("  stderr: " + tail.replace("\n", "\n  "))

    print("-" * 60)
    print(f"{len(checks) - len(failures)}/{len(checks)} checks passed")
    if failures:
        print("failed:", ", ".join(name for name, _ in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
