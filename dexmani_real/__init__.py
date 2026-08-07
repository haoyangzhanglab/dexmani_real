"""DexMani Real — dexterous manipulation robot teleoperation & data collection.

Subsystems:
    robot/      — Hardware drivers (XArm7 7-DOF arm, XHand 12-DOF hand)
    teleop/     — VR teleoperation (tracking, retargeting, control, safety)
    planning/   — Motion planning (IK, kinematics, collision detection)

    recording/  — HDF5 episode recording
    sensor/     — RealSense camera driver
    config/     — Camera calibration, pipeline config
    utils/      — Shared utilities (arrays, signals, coordinate transforms, point clouds)
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
ASSET_DIR = (PACKAGE_DIR / "assets") if (PACKAGE_DIR / "assets").exists() else (PACKAGE_DIR.parent / "assets")
