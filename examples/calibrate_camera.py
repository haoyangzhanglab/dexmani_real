#!/usr/bin/env python3
"""Run the xArm7/RealSense eye-to-hand camera calibration workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.sensor.camera_calibration import main

if __name__ == "__main__":
    raise SystemExit(main())
