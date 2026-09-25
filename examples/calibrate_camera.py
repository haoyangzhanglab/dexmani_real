#!/usr/bin/env python3
"""Usage: python examples/calibrate_camera.py --hand-geometry {absent,secured-home}

Moves xArm7; ENTER solves calibration and saves cameras.json only if quality checks pass.
Use absent only without XHand; secured-home requires a hand fixed at home.
Both use the fixed-home collision envelope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from dexmani_real.calibration.camera.session import run_camera_calibration
from dexmani_real.calibration.camera.solver import ARUCO_DICT_NAME, ArucoConfig
from dexmani_real.config.experiment import resolve_experiment_config


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
        help="experiment YAML; --serial takes precedence",
    )
    args = parser.parse_args(argv)
    aruco = ArucoConfig()

    print("=" * 60)
    print("  ArUco Hand-Eye Calibration — xArm7 + RealSense (eye-to-hand)")
    print(
        f"  ArUco: {ARUCO_DICT_NAME} ID={aruco.target_id} size={aruco.marker_size_m * 1000:.1f}mm"
    )
    print(f"  hand physical-state assertion: {args.hand_geometry}")
    print("=" * 60)

    try:
        runtime = resolve_experiment_config(
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
        hand_geometry=args.hand_geometry,
        aruco_config=aruco,
    )


if __name__ == "__main__":
    raise SystemExit(main())
