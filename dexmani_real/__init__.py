"""DexMani Real — dexterous manipulation robot teleoperation & data collection.

Subsystems:
    robot/        — Hardware drivers (xArm7 arm, XHand) and execution workers
    teleop/       — VR teleoperation (mapping, retargeting, control loop)
    planning/     — FK/IK, kinematics, collision and path planning
    policy/       — Action protocol and safety validation
    recording/    — HDF5 episode recording and offline readers
    deployment/   — Learned-policy deployment runtime
    integrations/ — Model-repository adapters
    runtime/      — Process lifecycle, supervision, shutdown
    shm/          — Shared-memory data plane (rings, queues, storage)
    sensor/       — RealSense camera, VR receiver, point-cloud processing
    config/       — Defaults, runtime snapshot, calibration
    utils/        — Shared utilities (schema, limits, logging, arrays)
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
ASSET_DIR = (PACKAGE_DIR / "assets") if (PACKAGE_DIR / "assets").exists() else (PACKAGE_DIR.parent / "assets")
