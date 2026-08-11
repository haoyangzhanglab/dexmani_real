#!/usr/bin/env python3
"""Experimental learned-policy deployment entry point.

Unlike the teleoperation path, this capability requires an external adapter,
PolicySpec YAML, model resources, and its own offline validation fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.policy.deployment import main

if __name__ == "__main__":
    raise SystemExit(main())
