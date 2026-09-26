"""Validate PolicySpec cadence against actuator worker service rates.

Policy defines tensor shapes, modalities and action spacing. Worker settings
are pickle-safe; this module imports neither Policy nor Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.ipc.schema import POINT_CLOUD_FEATURE_DIM
from dexmani_real.robot.model import (
    HAND_FINGER_NAMES,
    ROBOT_JOINT_NAMES,
    XHAND_RIGHT_URDF_PATH,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)

_SUPPORTED_OBSERVATION_FIELDS = frozenset(
    {
        "joint_state",
        "point_cloud",
        "rgb",
        "contact_force",
        "fingertip_points",
        "eef_pose",
        "tactile_force",
    }
)


def validate_max_running_s(max_running_s: float | None) -> float | None:
    """Validate the runner-owned B-relative episode duration limit."""
    if max_running_s is None:
        return None
    if isinstance(max_running_s, bool):
        raise TypeError("max_running_s must be a finite positive number or None")
    value = float(max_running_s)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("max_running_s must be finite and positive")
    return value


def _validate_real_observation_capability(policy_spec: Any) -> tuple[Any, ...]:
    """Validate the Policy observation projection that Real can produce."""
    fields = policy_spec.observation_fields
    names = tuple(field.name for field in fields)
    if not set(names) <= _SUPPORTED_OBSERVATION_FIELDS or "joint_state" not in names:
        raise ValueError("Real supports only configured modalities including joint_state")

    fixed_fields = {
        "joint_state": ((19,), "float32"),
        "contact_force": ((5, 3), "float32"),
        "fingertip_points": ((5, 3), "float32"),
        "eef_pose": ((9,), "float32"),
        "tactile_force": ((5, 120, 3), "float32"),
    }
    for field, name in zip(fields, names, strict=True):
        shape = field.shape
        dtype = field.dtype
        if name in fixed_fields:
            if (shape, dtype) != fixed_fields[name]:
                raise ValueError(
                    f"Policy observation field {name!r} does not match Real's raw tensor"
                )
        elif name == "point_cloud":
            if (
                len(shape) != 2
                or shape[1] != POINT_CLOUD_FEATURE_DIM
                or dtype != "float32"
                or isinstance(shape[0], bool)
                or not isinstance(shape[0], Integral)
                or shape[0] <= 0
            ):
                raise ValueError(
                    "Policy point_cloud must be float32 [N, 6] with positive integer N"
                )
        elif name == "rgb" and (shape != (None, None, 3) or dtype != "uint8"):
            raise ValueError("Policy rgb must declare raw uint8 HWC with variable H/W")
    return fields


def validate_policy_runtime_compatibility(policy_spec: Any, runtime: Any) -> None:
    """Validate only whether Real can run the Policy-owned public contract."""
    policy_hz = 1.0 / float(policy_spec.control_dt_s)
    limiting_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
    if policy_hz > limiting_hz and not math.isclose(
        policy_hz, limiting_hz, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            f"Policy action rate {policy_hz:g} Hz exceeds limiting worker rate {limiting_hz:g} Hz"
        )
    fields = _validate_real_observation_capability(policy_spec)
    if tuple(policy_spec.joint_names) != ROBOT_JOINT_NAMES:
        raise ValueError("Policy ordered joint_names do not match Real robot joints")
    if policy_spec.action_mode not in {"joint", "eef"}:
        raise ValueError("Policy action_mode must be joint or eef")
    for field in fields:
        expected = {}
        if field.name == "point_cloud":
            expected["features"] = ("x", "y", "z", "r", "g", "b")
            config = PointCloudConfig.from_dict(policy_spec.pointcloud_config)
            if config.num_points != field.shape[0]:
                raise ValueError("Policy pointcloud config disagrees with its tensor shape")
        if field.name == "rgb":
            expected["channels"] = ("r", "g", "b")
        if field.name in {"fingertip_points", "contact_force", "tactile_force"}:
            expected["fingers"] = HAND_FINGER_NAMES
        if field.name in {"contact_force", "tactile_force"}:
            expected["sensors"] = XHAND_TACTILE_SENSOR_FINGER_IDS
            expected["axes"] = ("fx", "fy", "fz")
        if field.name == "tactile_force":
            expected["points"] = tuple(range(120))
        for axis, order in expected.items():
            if tuple(field.ordering.get(axis, ())) != order:
                raise ValueError(f"Policy {field.name} {axis} order does not match Real")


@dataclass(frozen=True)
class PolicyRuntimeConfig:
    """Narrow Policy-owned inputs required by the policy child.

    ``artifact`` pins the resolved deployment filename so the child cannot
    silently switch checkpoints after the parent inspected them; Policy owns
    the filesystem layout and only the basename crosses this boundary.
    """

    experiment: str
    device: str
    spec: Any
    seed: int
    artifact: str
    inference_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, str) or not self.experiment.strip():
            raise ValueError("experiment must be a non-empty Policy selector")
        if self.experiment != self.experiment.strip():
            raise ValueError("experiment must not have surrounding whitespace")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        if self.device != self.device.strip():
            raise ValueError("device must not have leading or trailing whitespace")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("policy inference seed must be a non-negative integer")
        if (
            not isinstance(self.artifact, str)
            or not self.artifact
            or self.artifact != Path(self.artifact).name
        ):
            raise ValueError("artifact must be a plain checkpoint filename (no path separators)")
        if type(self.inference_steps) is not int or self.inference_steps < 1:
            raise ValueError("inference_steps must be a positive integer")


def validate_num_episodes(num_episodes: Any) -> int:
    """Validate the run-owner episode budget (episodes to run, not saves)."""
    if isinstance(num_episodes, bool) or type(num_episodes) is not int:
        raise TypeError("num_episodes must be a positive integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    return int(num_episodes)


@dataclass(frozen=True)
class RolloutRecordingConfig:
    """Recording inputs for a physical rollout, with an absolute session data_dir.

    The CLI validates paths before workers start. Episode counts and budgets
    belong to lifecycle configuration. Runtime records technical stop reasons;
    task success is judged offline from raw episodes.
    """

    data_dir: str
    task_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, str) or not self.data_dir.strip():
            raise ValueError("rollout data_dir must be a non-empty path")
        raw_data_dir = Path(self.data_dir)
        if not raw_data_dir.is_absolute():
            raise ValueError("rollout data_dir must be absolute")
        resolved_data_dir = raw_data_dir.resolve(strict=False)
        if resolved_data_dir == resolved_data_dir.parent:
            raise ValueError("rollout data_dir must not be a filesystem root")
        if not isinstance(self.task_label, str) or not self.task_label.strip():
            raise ValueError("rollout task_label must be non-empty")
        if self.task_label != self.task_label.strip():
            raise ValueError("rollout task_label must not have surrounding whitespace")
        object.__setattr__(self, "data_dir", str(resolved_data_dir))


@dataclass(frozen=True)
class FingertipAssemblerConfig:
    """Explicit pickle-safe geometry inputs for deployment-local FK."""

    hand_urdf_path: str
    fingertip_link_names: tuple[str, ...]
    handbase_position_eef_m: tuple[float, float, float]
    handbase_quat_eef_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_runtime(cls, runtime: Any) -> "FingertipAssemblerConfig":
        hand = runtime.hand
        return cls(
            hand_urdf_path=str(XHAND_RIGHT_URDF_PATH),
            fingertip_link_names=tuple(hand.fingertip_link_names),
            handbase_position_eef_m=tuple(hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        if not self.hand_urdf_path or len(self.fingertip_link_names) != 5:
            raise ValueError("fingertip FK requires one URDF and five link names")
        values = (*self.handbase_position_eef_m, *self.handbase_quat_eef_wxyz)
        if (
            len(self.handbase_position_eef_m) != 3
            or len(self.handbase_quat_eef_wxyz) != 4
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise ValueError("fingertip mount transform is malformed")


__all__ = [
    "FingertipAssemblerConfig",
    "PolicyRuntimeConfig",
    "RolloutRecordingConfig",
    "validate_policy_runtime_compatibility",
    "validate_max_running_s",
    "validate_num_episodes",
]
