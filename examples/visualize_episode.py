#!/usr/bin/env python3
"""Usage: ``python examples/visualize_episode.py EPISODE``.

Visualize a recorded DexMani episode with Rerun.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.recording.episode_visualizer import main

if __name__ == "__main__":
    raise SystemExit(main())
