"""Small teleoperation view over the canonical typed runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH


@dataclass(frozen=True)
class TeleopConfig:
    """Episode metadata and hand model path alongside the resolved runtime."""

    runtime: ExperimentConfig
    task_label: str = ""
    hand_urdf_path: str = str(XHAND_RIGHT_URDF_PATH)

    def __post_init__(self) -> None:
        if not self.hand_urdf_path:
            raise ValueError("hand_urdf_path must be non-empty")
