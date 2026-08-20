#!/usr/bin/env python3
"""Run the RealSense point-cloud and desk-plane diagnostic workflow."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.sensor.pointcloud_diagnostic import main

if __name__ == "__main__":
    raise SystemExit(main())
