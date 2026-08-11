#!/usr/bin/env python3
"""Keyboard teleoperation entry point with measured XHand feedback by default."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.teleop.keyboard_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
