#!/usr/bin/env python3
"""Usage: ``python examples/realsense_record_example.py``.

Run the interactive RealSense RGB-D and point-cloud diagnostic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.sensor.realsense_diagnostic import main

if __name__ == "__main__":
    raise SystemExit(main())
