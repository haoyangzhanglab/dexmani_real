"""Pipeline-level configuration aggregating all subsystem configs.

Single-point serializable config source for HDF5 /meta pipeline_snapshot.
"""

from __future__ import annotations

__all__ = ["PipelineConfig"]

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.planning.types import PlanningProfile, TeleopProfile
from dexmani_real.robot.interface import RobotInterfaceConfig

# Default max frames per recording episode — single source of truth
# used by both PipelineConfig and EpisodeRecorder.
DEFAULT_MAX_RECORD_FRAMES: int = 10000
from dexmani_real.utils.serialization import FromDictMixin, from_dict_helper


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
class PipelineConfig(FromDictMixin):
    """Aggregated configuration for the entire teleoperation pipeline.

    Serialized into HDF5 /meta as pipeline_snapshot for full reproducibility.
    """

    robot: RobotInterfaceConfig = field(default_factory=RobotInterfaceConfig)
    planning_profile: PlanningProfile = field(default_factory=PlanningProfile)
    teleop_profile: TeleopProfile = field(default_factory=TeleopProfile)

    control_rate_hz: float = 50.0

    cartesian_ema_alpha_teleop: float = 0.3   # Cartesian EEF EMA before IK
    cartesian_ema_alpha_deploy: float = 0.5

    data_dir: str = "episodes"
    max_record_frames: int = DEFAULT_MAX_RECORD_FRAMES

    pipeline_name: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (numpy arrays converted to lists)."""
        return _ndarray_to_list(dataclasses.asdict(self))

    # from_dict inherited from FromDictMixin.
    # Handles: list→ndarray, list→tuple, dict→nested dataclass, None for Optional.
