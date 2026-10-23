#!/usr/bin/env python3
"""移动 xArm7 并采集 RealSense 图像标定相机外参，通过质量检查后保存。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dexmani_real.config.experiment import load_experiment_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ArUco eye-to-hand camera calibration")
    parser.add_argument(
        "--serial",
        default=None,
        help="RealSense serial (required with multiple devices)",
    )
    parser.add_argument(
        "--hand-geometry",
        choices=("absent", "secured-home"),
        required=True,
        help=(
            "required physical-state assertion: absent means no XHand is mounted; "
            "secured-home means a mounted XHand is physically fixed at configured "
            "home. Collision checks use the fixed-home XHand envelope in both cases."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment YAML overrides; --serial takes precedence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Camera calibration JSON (default: checkout calibration/state/cameras.json)",
    )
    args = parser.parse_args(argv)

    from dexmani_real.calibration import CAMERAS_PATH
    from dexmani_real.calibration.camera.session import CameraCalibrationSession
    from dexmani_real.calibration.camera.aruco import ARUCO_DICT_NAME, ArucoConfig

    aruco = ArucoConfig()

    print("=" * 60)
    print("  ArUco Hand-Eye Calibration — xArm7 + RealSense (eye-to-hand)")
    print(
        f"  ArUco: {ARUCO_DICT_NAME} ID={aruco.target_id} size={aruco.marker_size_m * 1000:.1f}mm"
    )
    print(f"  hand physical-state assertion: {args.hand_geometry}")
    print("=" * 60)

    runtime = load_experiment_config(
        yaml_path=args.config,
        cli_overrides={"camera.serial": args.serial, "hand.enabled": False},
    )

    return CameraCalibrationSession(
        runtime,
        hand_geometry=args.hand_geometry,
        aruco_config=aruco,
        output_path=args.output or CAMERAS_PATH,
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
