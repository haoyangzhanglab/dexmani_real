"""DexMani Real — real-robot dexterous manipulation runtime.

Subsystems:
    calibration/  — Camera and table calibration behavior
    config/       — Defaults, validated runtime snapshots, and calibration data
    dataset/      — Offline episode processing and dataset export
    deployment/   — Learned-policy deployment runtime
    ipc/          — Shared-memory schemas, rings, queues, and channels
    planning/     — FK/IK, collision checking, and path planning
    recording/    — Transactional episode recording and reading
    replay/       — Safety-gated physical replay
    robot/        — Target preparation, publication, homing, drivers and workers
    runtime/      — Process lifecycle, supervision, and shutdown
    sensor/       — Camera, VR, and point-cloud acquisition
    teleop/       — VR/keyboard teleoperation and hand retargeting
    utils/        — Small shared helpers without domain ownership
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).parent.resolve()
ASSET_DIR = PACKAGE_DIR.parent / "assets"
