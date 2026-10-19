"""RGB 分割与 mesh 重建；提供实测尺寸后显式生成米制追踪资产。"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.build_object import main

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]], workspace=ROOT))
