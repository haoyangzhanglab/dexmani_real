"""Environment for isolated Python model processes, without global path changes."""

import os
from pathlib import Path


def module_environment():
    """Let a fresh Python find this package, including a source-only checkout.

    Parent sys.path edits are not inherited by subprocesses. Only this child's
    PYTHONPATH is extended; installed packages and the caller environment stay.
    """
    environment = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source + (os.pathsep + current if current else "")
    return environment
