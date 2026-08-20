#!/usr/bin/env python3
"""Run the hardware-affecting xArm7/RealSense eye-to-hand calibration workflow.

The interactive session can command the arm and writes ``cameras.json`` after
ENTER.  ``--hand-geometry`` is a physical collision-model assertion: use
``absent`` only without a mounted XHand, or ``secured-home`` only when the
mounted hand is physically fixed at its configured home pose.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.sensor.camera_calibration import ARUCO_DICT_NAME, ArucoConfig
from dexmani_real.sensor.camera_calibration_session import run_camera_calibration


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
        default="secured-home",
        help="physical assertion for arm-only procedure (default: secured-home)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="experiment YAML; --serial takes precedence",
    )
    args = parser.parse_args(argv)
    aruco = ArucoConfig()

    print("=" * 60)
    print("  ArUco Hand-Eye Calibration — xArm7 + RealSense (eye-to-hand)")
    print(
        f"  ArUco: {ARUCO_DICT_NAME} ID={aruco.target_id} "
        f"size={aruco.marker_size_m * 1000:.1f}mm"
    )
    print(f"  hand geometry assertion: {args.hand_geometry}")
    print("=" * 60)

    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={"camera.serial": args.serial},
        )
    except (
        KeyError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        print(f"Invalid calibration config: {exc}", file=sys.stderr)
        return 2

    return run_camera_calibration(
        runtime,
        camera_serial=args.serial,
        hand_geometry=args.hand_geometry,
        aruco_config=aruco,
    )


if __name__ == "__main__":
    raise SystemExit(main())
