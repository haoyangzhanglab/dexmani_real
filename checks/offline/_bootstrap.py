"""Ensure the repo root is importable when a check runs standalone.

``checks/offline/run_all.py`` already exports the repo root through
``PYTHONPATH``; this module is the equivalent bootstrap for a check executed
directly (``python checks/offline/check_*.py``) so it never depends on the
caller's working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
