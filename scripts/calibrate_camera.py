"""在同一窗口点动 xArm7、采集 RealSense 样本并标定相机外参。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.calibration_data import CAMERAS_PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, metavar="PATH", help="导出 CONFIG: RealConfig 的实验配置文件")
    parser.add_argument(
        "--output", type=Path, default=CAMERAS_PATH, metavar="JSON",
        help=f"标定通过后更新的外参 JSON（默认共享 {CAMERAS_PATH.name}）",
    )
    args = parser.parse_args(argv)

    from dexmani_real.config.real import load_config
    from dexmani_real.utils.camera_calibrate.interactive import CameraCalibrator

    return CameraCalibrator(load_config(args.config), output_path=args.output).run()


if __name__ == "__main__":
    raise SystemExit(main())
