#!/usr/bin/env python3
"""CLI entry point for bounded RealSense diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.diagnostics.realsense import main

if __name__ == "__main__":
    raise SystemExit(main())
