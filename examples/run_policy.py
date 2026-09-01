#!/usr/bin/env python3
"""Thin entry point for offline and physical learned-policy deployment commands."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dexmani_real.deployment.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
