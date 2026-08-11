#!/usr/bin/env python3
"""Inspect or replay one DexMani HDF5 episode.

Dry-run inspection is the default. Passing ``--live`` crosses the hardware
safety boundary and additionally requires a preflight certificate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.replay.episode import main

if __name__ == "__main__":
    raise SystemExit(main())
