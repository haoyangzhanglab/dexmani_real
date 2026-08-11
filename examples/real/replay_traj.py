#!/usr/bin/env python3
"""Legacy filename alias for ``replay_episode.py``.

Existing ``--h5 PATH`` commands remain accepted; new commands should pass the
episode path positionally to ``replay_episode.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.replay.episode import main

if __name__ == "__main__":
    raise SystemExit(main())
