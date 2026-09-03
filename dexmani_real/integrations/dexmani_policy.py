"""NumPy adapter from Real observations to the public DexMani Policy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.config import validate_policy_spec
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.planning.poses import validate_rot6d_geometry
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_ARM_DOF = ARM_JOINT_SHAPE[0]
_HAND_DOF = HAND_JOINT_SHAPE[0]
_JOINT_ACTION_DIM = _ARM_DOF + _HAND_DOF
_EE_POS_DIM = 3
_EE_ROT6D_DIM = 6
_EE_ACTION_DIM = _EE_POS_DIM + _EE_ROT6D_DIM + _HAND_DOF


class DexManiPolicyRuntime:
    """Adapt one already-loaded public Policy runtime to Real's typed contract.

    This class owns no checkpoint, Hydra, EMA, normalizer, Torch, or device
    behavior. The inference worker loads those through the Policy public API.
    """

    def __init__(self, loaded_policy: Any, expected_spec: Any) -> None:
        validate_policy_spec(expected_spec)
        for name in ("info", "spec", "warmup", "reset_episode", "predict", "close"):
            if not hasattr(loaded_policy, name):
                raise TypeError(f"loaded Policy runtime is missing {name!r}")
        self._policy: Any | None = loaded_policy
        self.spec = expected_spec
        self._validate_loaded_spec(loaded_policy.spec)

    def _validate_loaded_spec(self, spec: Any) -> None:
        """Reject inspect/load drift and unsupported Real embodiment contracts."""
        expected = {
            name: getattr(self.spec, name)
            for name in (
                "action_key",
                "action_dim",
                "control_action_dim",
                "horizon",
                "n_obs_steps",
                "n_action_steps",
                "control_dt_s",
                "point_cloud_num_points",
                "point_cloud_feature_dim",
                "sensor_modalities",
                "requires_hand",
            )
        }
        mismatches = {
            name: (getattr(spec, name, None), value)
            for name, value in expected.items()
            if getattr(spec, name, None) != value
        }
        if mismatches:
            raise ValueError(
                "loaded PolicySpec does not match the validated Real projection: "
                f"{mismatches}"
            )

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        return self._require_open().warmup(samples=samples)

    def reset_episode(self) -> None:
        self._require_open().reset_episode()

    def predict(self, observation: ObservationBatch) -> PolicyPrediction:
        arrays = self._encode_observation(observation)
        control_action = self._require_open().predict(arrays)
        return self._decode_control_action(control_action)

    def close(self) -> None:
        policy = self._policy
        if policy is None:
            return
        self._policy = None
        policy.close()

    def _require_open(self) -> Any:
        if self._policy is None:
            raise RuntimeError("DexMani Policy adapter is closed")
        return self._policy

    def _encode_observation(
        self, observation: ObservationBatch
    ) -> dict[str, np.ndarray]:
        if not isinstance(observation, ObservationBatch):
            raise TypeError("policy observation must be an ObservationBatch")
        if observation.arm_history is None or observation.hand_history is None:
            raise ValueError("DexMani Policy requires arm and hand joint history")
        n_obs = self.spec.n_obs_steps
        arm = observation.arm_history.values
        hand = observation.hand_history.values
        if arm.shape != (n_obs, _ARM_DOF) or hand.shape != (n_obs, _HAND_DOF):
            raise ValueError(
                f"joint history must be arm {(n_obs, _ARM_DOF)} and "
                f"hand {(n_obs, _HAND_DOF)}"
            )
        if arm.dtype.kind != "f" or hand.dtype.kind != "f":
            raise TypeError("joint history must use floating-point arrays")
        if not np.isfinite(arm).all() or not np.isfinite(hand).all():
            raise ValueError("joint history contains NaN/Inf")

        frames = tuple(observation.pointcloud_history)
        if len(frames) != n_obs:
            raise ValueError(
                f"need exactly {n_obs} causal point-cloud frames, got {len(frames)}"
            )
        if len({frame.camera_generation for frame in frames}) != 1:
            raise ValueError("point-cloud history crosses a camera generation boundary")
        expected_point_shape = (
            self.spec.point_cloud_num_points,
            self.spec.point_cloud_feature_dim,
        )
        for frame in frames:
            if frame.values.shape != expected_point_shape:
                raise ValueError(
                    "point-cloud frame shape does not match the PolicySpec projection"
                )
            if frame.values.dtype != np.float32:
                raise TypeError("point-cloud frames must be float32")
            if not np.isfinite(frame.values).all():
                raise ValueError("point-cloud history contains NaN/Inf")

        joint_state = np.ascontiguousarray(
            np.concatenate((arm, hand), axis=1), dtype=np.float32
        )
        point_cloud = np.ascontiguousarray(
            np.stack([frame.values for frame in frames]), dtype=np.float32
        )
        return {"joint_state": joint_state, "point_cloud": point_cloud}

    def _decode_control_action(self, value: Any) -> PolicyPrediction:
        if not isinstance(value, np.ndarray):
            raise TypeError("Policy predict must return a NumPy array")
        expected_shape = (
            self.spec.n_action_steps,
            self.spec.control_action_dim,
        )
        if value.shape != expected_shape or value.dtype != np.float64:
            raise ValueError(
                "Policy control action must be float64 "
                f"{expected_shape}, got shape={value.shape} dtype={value.dtype}"
            )
        if not np.isfinite(value).all():
            raise ValueError("Policy control action contains NaN/Inf")

        control = np.array(value, dtype=np.float64, copy=True, order="C")
        if self.spec.action_key == "action":
            if control.shape[1] != _JOINT_ACTION_DIM:
                raise ValueError("joint control action must be arm7 + hand12")
            return PolicyPrediction(
                arm_qpos=control[:, :_ARM_DOF],
                hand_qpos=control[:, _ARM_DOF:],
            )

        if control.shape[1] != _EE_ACTION_DIM:
            raise ValueError("EE control action must be pos3 + rot6d6 + hand12")
        ee_pos = control[:, :_EE_POS_DIM]
        ee_rot6d = control[:, _EE_POS_DIM : _EE_POS_DIM + _EE_ROT6D_DIM]
        validate_rot6d_geometry(ee_rot6d, label="ee_rot6d")
        return PolicyPrediction(
            arm_qpos=None,
            hand_qpos=control[:, _EE_POS_DIM + _EE_ROT6D_DIM :],
            ee_pos=ee_pos,
            ee_rot6d=ee_rot6d,
        )
