"""Real-side deployment manifest — the validated model <-> deployment contract.

A :class:`DeploymentManifest` is the frozen summary of "what this checkpoint
expects" that a new model must satisfy to run under the deployment runtime. It
is assembled by the Real side after verified checkpoint restore from three
sources:

* the checkpoint's ``train_params`` (authoritative model hyper-parameters),
* the embedded model config ``sensor_modalities`` / ``num_points`` / ``pc_dim``,
* the real control/deployment configuration (action-key expectation, hand
  presence, point-cloud count).

``domain`` is a construct-time constant (``"real"``). The adapter separately
requires a Real Policy Zarr v5 training-data contract before constructing this
manifest; the constant alone is not evidence of checkpoint provenance.

``dt`` remains outside the model-shape manifest, but the checkpoint training-data
contract must match ``1 / action_control_hz`` before the agent is constructed.
Per-inference action timestamps then use ``InferenceContext.step_dt_ns``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.ipc.schema import (
    ARM_DOF,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    HAND_DOF,
    MAX_POLICY_CHUNK_STEPS,
    POINT_CLOUD_FEATURE_DIM,
    SUPPORTED_POINT_CLOUD_COUNTS,
)

SUPPORTED_ACTION_KEYS = frozenset({"action", "action_ee"})
SUPPORTED_SENSOR_MODALITIES = frozenset({"joint_state", "point_cloud", "rgb"})

# Native control-action dimensions (what the runtime hands the coordinator).
# joint: arm(7) + hand(12); ee: position(3) + rot6d(6) + hand(12).
JOINT_CONTROL_ACTION_DIM = ARM_DOF + HAND_DOF
EE_CONTROL_ACTION_DIM = EE_POS_DIM + EE_ROT6D_DIM + HAND_DOF


@dataclass(frozen=True)
class DeploymentManifest:
    """Frozen, validated summary of the model-facing deployment contract."""

    domain: str = "real"
    action_key: str = "action"
    n_obs_steps: int = 2
    n_action_steps: int = 8
    action_dim: int = 19
    horizon: int = 16
    hand_dim: int | None = None
    tcp_dim: int | None = None
    control_action_dim: int = JOINT_CONTROL_ACTION_DIM
    auxiliary_action_layout: str = "none"
    sensor_modalities: tuple[str, ...] = ("joint_state", "point_cloud")
    point_cloud_num_points: int | None = 1024
    point_cloud_feature_dim: int | None = POINT_CLOUD_FEATURE_DIM
    rgb_shape: tuple[int, int, int] | None = None
    rgb_color_order: str | None = None
    rgb_value_range: str | None = None

    def __post_init__(self) -> None:
        if self.domain != "real":
            raise ValueError(
                f"DeploymentManifest.domain must be 'real', got {self.domain!r}"
            )
        if self.action_key not in SUPPORTED_ACTION_KEYS:
            raise ValueError(
                f"action_key must be one of {sorted(SUPPORTED_ACTION_KEYS)}, "
                f"got {self.action_key!r}"
            )
        for name in (
            "n_obs_steps",
            "n_action_steps",
            "action_dim",
            "horizon",
            "control_action_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        # control_action slice bound (mirrors dexmani_policy validate_eval_config).
        if self.n_obs_steps - 1 + self.n_action_steps > self.horizon:
            raise ValueError(
                "n_obs_steps-1+n_action_steps exceeds horizon: the control_action "
                f"slice [{self.n_obs_steps - 1}:{self.n_obs_steps - 1 + self.n_action_steps}] "
                "would be out of bounds"
            )
        if self.n_action_steps > MAX_POLICY_CHUNK_STEPS:
            raise ValueError(
                f"n_action_steps={self.n_action_steps} exceeds the plan transport "
                f"capacity {MAX_POLICY_CHUNK_STEPS}"
            )
        for name in ("hand_dim", "tcp_dim"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer or None, got {value!r}"
                )
        expected_control = (
            JOINT_CONTROL_ACTION_DIM
            if self.action_key == "action"
            else EE_CONTROL_ACTION_DIM
        )
        if self.control_action_dim != expected_control:
            raise ValueError(
                f"control_action_dim={self.control_action_dim} is inconsistent with "
                f"action_key={self.action_key!r} (expected {expected_control})"
            )
        if self.auxiliary_action_layout == "none":
            if self.action_dim != self.control_action_dim:
                raise ValueError(
                    "action_dim must equal control_action_dim without an auxiliary layout"
                )
        elif self.auxiliary_action_layout == "joint19_ee9":
            if (
                self.action_key != "action"
                or self.control_action_dim != JOINT_CONTROL_ACTION_DIM
                or self.action_dim
                != JOINT_CONTROL_ACTION_DIM + EE_POS_DIM + EE_ROT6D_DIM
            ):
                raise ValueError("joint19_ee9 requires action=28 and control_action=19")
        else:
            raise ValueError("auxiliary_action_layout must be 'none' or 'joint19_ee9'")
        modalities = self.sensor_modalities
        if not isinstance(modalities, tuple) or not modalities:
            raise ValueError("sensor_modalities must be a non-empty tuple")
        if len(set(modalities)) != len(modalities):
            raise ValueError("sensor_modalities must not contain duplicates")
        unknown = set(modalities) - SUPPORTED_SENSOR_MODALITIES
        if unknown:
            raise ValueError(f"unsupported sensor modalit(ies): {sorted(unknown)}")
        if "joint_state" not in modalities or not (
            {"point_cloud", "rgb"} & set(modalities)
        ):
            raise ValueError(
                "sensor_modalities must contain joint_state and at least one visual modality"
            )
        if "point_cloud" in modalities:
            if (
                isinstance(self.point_cloud_num_points, bool)
                or self.point_cloud_num_points not in SUPPORTED_POINT_CLOUD_COUNTS
            ):
                raise ValueError(
                    "point_cloud_num_points must be one of "
                    f"{sorted(SUPPORTED_POINT_CLOUD_COUNTS)}"
                )
            if self.point_cloud_feature_dim != POINT_CLOUD_FEATURE_DIM:
                raise ValueError(
                    f"point_cloud_feature_dim must be {POINT_CLOUD_FEATURE_DIM}"
                )
        elif (
            self.point_cloud_num_points is not None
            or self.point_cloud_feature_dim is not None
        ):
            raise ValueError(
                "point-cloud metadata must be None when point_cloud is absent"
            )
        if "rgb" in modalities:
            if (
                not isinstance(self.rgb_shape, tuple)
                or len(self.rgb_shape) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in self.rgb_shape
                )
                or self.rgb_shape[2] != 3
            ):
                raise ValueError(
                    "rgb_shape must be a positive (height, width, 3) tuple"
                )
            if self.rgb_color_order != "rgb" or self.rgb_value_range != "uint8_0_255":
                raise ValueError("RGB metadata must describe raw RGB uint8 [0,255]")
        elif any(
            value is not None
            for value in (self.rgb_shape, self.rgb_color_order, self.rgb_value_range)
        ):
            raise ValueError("RGB metadata must be None when rgb is absent")

    @property
    def uses_point_cloud(self) -> bool:
        return "point_cloud" in self.sensor_modalities

    @property
    def uses_rgb(self) -> bool:
        return "rgb" in self.sensor_modalities

    @property
    def is_ee(self) -> bool:
        return self.action_key == "action_ee"


def manifest_from_sources(
    *,
    action_key: str,
    n_obs_steps: int,
    n_action_steps: int,
    action_dim: int,
    horizon: int,
    tcp_dim: int | None,
    hand_dim: int | None,
    control_action_dim: int,
    auxiliary_action_layout: str,
    sensor_modalities: Iterable[str],
    point_cloud_num_points: int | None,
    point_cloud_feature_dim: int | None,
    rgb_shape: tuple[int, int, int] | None,
    rgb_color_order: str | None,
    rgb_value_range: str | None,
) -> DeploymentManifest:
    """Assemble and validate a manifest from model + config sources.

    Every input shape is artifact-owned. Missing metadata is invalid when its
    modality is requested rather than replaced with a deployment fallback.
    """
    modalities = tuple(str(m) for m in sensor_modalities)
    if len(set(modalities)) != len(modalities):
        raise ValueError("sensor_modalities must not contain duplicates")
    modalities = tuple(sorted(modalities))
    return DeploymentManifest(
        domain="real",
        action_key=action_key,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        action_dim=action_dim,
        horizon=horizon,
        hand_dim=hand_dim,
        tcp_dim=tcp_dim,
        control_action_dim=control_action_dim,
        auxiliary_action_layout=auxiliary_action_layout,
        sensor_modalities=modalities,
        point_cloud_num_points=(
            None if point_cloud_num_points is None else int(point_cloud_num_points)
        ),
        point_cloud_feature_dim=(
            None if point_cloud_feature_dim is None else int(point_cloud_feature_dim)
        ),
        rgb_shape=rgb_shape,
        rgb_color_order=rgb_color_order,
        rgb_value_range=rgb_value_range,
    )


def validate_manifest_against_deployment(
    manifest: DeploymentManifest,
    deployment: DeploymentConfig,
) -> None:
    """Cross-validate a loaded manifest against the resolved deployment config.

    This is the startup gate that turns a config/checkpoint mismatch into a
    fail-closed load error instead of a silent runtime hang (all plans dropped,
    silence watchdog exempt until the first command).  Raises ``ValueError`` on
    the first inconsistency.
    """
    if manifest.action_key != deployment.action_key:
        raise ValueError(
            f"manifest action_key={manifest.action_key!r} does not match "
            f"deployment action_key={deployment.action_key!r}"
        )
    if manifest.n_obs_steps != deployment.observation_horizon:
        raise ValueError(
            f"manifest n_obs_steps={manifest.n_obs_steps} does not match deployment "
            f"observation_horizon={deployment.observation_horizon}"
        )
    # Both action keys command hand12 (19-DoF joint or 21-DoF EE), so the hand
    # must be enabled and present in the observation contract.
    if not deployment.hand_enabled:
        raise ValueError(
            "policy commands hand12 (joint 19D or EE 21D) and requires "
            "hand_enabled=True"
        )
    from dexmani_real.deployment.observation import parse_observation_fields

    requested = set(parse_observation_fields(deployment.observation_fields))
    expected_fields = {"arm_qpos", "hand_qpos"}
    if manifest.uses_point_cloud:
        expected_fields.add("point_cloud")
    if manifest.uses_rgb:
        expected_fields.add("rgb")
    if requested != expected_fields:
        raise ValueError(
            "DexMani real deployment observation_fields must match the artifact "
            f"modalities {sorted(expected_fields)!r}; got "
            f"{deployment.observation_fields!r}"
        )
    hand_fields = requested & {
        "hand_qpos",
        "hand_joint_position",
        "hand_current",
        "hand_joint_torque",
        "hand_tactile_sum",
        "fingertip_force",
    }
    if not hand_fields:
        raise ValueError(
            "policy commands the hand but observation_fields="
            f"{deployment.observation_fields!r} requests no hand field"
        )
    modalities = frozenset(manifest.sensor_modalities)
    if "joint_state" not in modalities or not (modalities & {"point_cloud", "rgb"}):
        raise ValueError(
            "DexMani real deployment requires joint_state plus at least one "
            "supported visual modality; "
            f"got manifest sensor_modalities={manifest.sensor_modalities!r}"
        )
    if (
        manifest.uses_point_cloud
        and manifest.point_cloud_num_points != deployment.pointcloud_num_points
    ):
        raise ValueError(
            f"manifest point_cloud_num_points={manifest.point_cloud_num_points} "
            f"does not match deployment pointcloud_num_points="
            f"{deployment.pointcloud_num_points}"
        )
    if manifest.uses_point_cloud and "point_cloud" not in requested:
        raise ValueError(
            "manifest requires point_cloud but deployment observation_fields "
            f"={deployment.observation_fields!r} does not request it"
        )
    if manifest.uses_rgb and "rgb" not in requested:
        raise ValueError(
            "manifest requires rgb but deployment observation_fields "
            f"={deployment.observation_fields!r} does not request it"
        )
