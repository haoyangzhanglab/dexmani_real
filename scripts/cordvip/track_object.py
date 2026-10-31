"""按 Raw 行顺序离线追踪场景，保存位姿与点云结果。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.scene_tracking import main

if __name__ == "__main__":
    raise SystemExit(main(workspace=ROOT))
