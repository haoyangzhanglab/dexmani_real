"""Pipeline-level configuration aggregating all subsystem configs.

Single-point serializable config source for HDF5 /meta pipeline_snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dexmani_real.robot.planner.planner_types import PlanningProfile, TeleopProfile
from dexmani_real.robot.robot_interface import RobotInterfaceConfig


def _ndarray_to_list(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ndarray_to_list(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


@dataclass
class PipelineConfig:
    """Aggregated configuration for the entire teleoperation pipeline.

    Serialized into HDF5 /meta as pipeline_snapshot for full reproducibility.
    """

    robot: RobotInterfaceConfig = field(default_factory=RobotInterfaceConfig)
    planning_profile: PlanningProfile = field(default_factory=PlanningProfile)
    teleop_profile: TeleopProfile = field(default_factory=TeleopProfile)

    control_rate_hz: float = 50.0

    hand_ema_alpha_teleop: float = 0.3
    arm_ema_alpha_teleop: float = 0.3
    arm_ema_alpha_deploy: float = 0.5
    hand_ema_alpha_deploy: float = 0.5

    data_dir: str = "data"

    pipeline_name: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (numpy arrays converted to lists)."""
        return {
            "pipeline_name": self.pipeline_name,
            "description": self.description,
            "control_rate_hz": self.control_rate_hz,
            "hand_ema_alpha_teleop": self.hand_ema_alpha_teleop,
            "arm_ema_alpha_teleop": self.arm_ema_alpha_teleop,
            "arm_ema_alpha_deploy": self.arm_ema_alpha_deploy,
            "hand_ema_alpha_deploy": self.hand_ema_alpha_deploy,
            "data_dir": self.data_dir,
            "robot": _ndarray_to_list(self._config_to_dict(self.robot)),
            "planning_profile": _ndarray_to_list(self._config_to_dict(self.planning_profile)),
            "teleop_profile": _ndarray_to_list(self._config_to_dict(self.teleop_profile)),
        }

    @staticmethod
    def _config_to_dict(config) -> dict:
        """Convert a @dataclass config to a plain dict."""
        result = {}
        for fld in config.__dataclass_fields__:
            result[fld] = getattr(config, fld)
        return result
