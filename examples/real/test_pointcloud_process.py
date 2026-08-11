#!/usr/bin/env python3
"""Compatibility entry point for :mod:`examples.real.diagnose_pointcloud`.

This module intentionally defines no pytest tests and imports no device SDK.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.diagnostics.pointcloud import main

__test__ = False


if __name__ == "__main__":
    raise SystemExit(main())
