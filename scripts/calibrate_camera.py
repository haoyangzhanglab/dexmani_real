#!/usr/bin/env python3
"""移动 xArm7 并采集 RealSense 图像标定相机外参，通过质量检查后保存。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dexmani_real.config.calibration_data import CAMERAS_PATH
from dexmani_real.config.real import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ArUco eye-to-hand camera calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Trusted Python experiment file exporting CONFIG: RealConfig",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Camera calibration JSON (default: {CAMERAS_PATH})",
    )
    args = parser.parse_args(argv)

    from dexmani_real.utils.camera_calibrate.aruco import (
        ARUCO_DICT_NAME,
        MARKER_ID,
        MARKER_SIZE_M,
    )
    from dexmani_real.utils.camera_calibrate.calibrate import CameraCalibrator

    print("=" * 60)
    print("  ArUco Hand-Eye Calibration — xArm7 + RealSense (eye-to-hand)")
    print(
        f"  ArUco: {ARUCO_DICT_NAME} ID={MARKER_ID} size={MARKER_SIZE_M * 1000:.1f}mm"
    )
    print("=" * 60)

    runtime = load_config(args.config)
    runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))

    return CameraCalibrator(
        runtime,
        output_path=args.output or CAMERAS_PATH,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
