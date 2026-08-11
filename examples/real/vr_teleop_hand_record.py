#!/usr/bin/env python3
"""Compatibility entry point for VR teleoperation data collection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.teleop.experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
