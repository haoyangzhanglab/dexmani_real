"""RGB-D 视频流 → 物体位姿、逐帧实测点云与模型点云可视化。

读取米制资产；默认连接 RealSense，--rgbd-dir 可使用离线 RGB-D。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.live_object_tracking import main

if __name__ == "__main__":
    raise SystemExit(main(workspace=ROOT))
