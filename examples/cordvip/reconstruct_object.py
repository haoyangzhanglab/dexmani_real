"""RGB 图像 → Grounded-SAM → Hunyuan3D mesh 与固定表面点云。

提供实测尺寸后额外生成米制追踪资产；模型和工作流实现在独立库中。
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]
from cordvip_geometry.cli.build_object import image_main

if __name__ == "__main__":
    raise SystemExit(image_main(workspace=ROOT))
