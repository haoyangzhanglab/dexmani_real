"""按 Raw 行顺序离线追踪场景，保存位姿与点云结果。"""
# Usage: python scripts/cordvip/track_object.py --episode PATH --scene SCENE.json --headless --output NEW_DIR

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

if __name__ == "__main__":
    from cordvip_geometry.cli.track import main

    raise SystemExit(main(workspace=ROOT))
