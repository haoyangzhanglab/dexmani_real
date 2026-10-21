"""RGB-D 位姿与点云可视化；默认连接 RealSense，--rgbd-dir 使用离线源。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.scene_tracking import main

if __name__ == "__main__":
    raise SystemExit(main(workspace=ROOT))
