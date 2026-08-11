#!/usr/bin/env python3
"""Compatibility entry point for the read-only XHand diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.diagnostics.xhand import main

if __name__ == "__main__":
    raise SystemExit(main())
