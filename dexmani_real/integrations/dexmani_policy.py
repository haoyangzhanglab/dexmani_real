"""NumPy adapter from Real observations to the public DexMani Policy runtime."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.config import (
    policy_observation_fields,
    validate_policy_spec,
)
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.observation import PolicyObservation
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
        """Reject inspect/load drift that would change Real's tensor or action boundary."""
        validate_policy_spec(spec)
        names = [
            "action_key",
            "action_dim",
            "control_action_dim",
            "horizon",
            "n_obs_steps",
            "n_action_steps",
            "control_dt_s",
            "requires_hand",
        ]
        expected = {name: getattr(self.spec, name) for name in names}
        mismatches = {
            name: (getattr(spec, name, None), value)
            for name, value in expected.items()
            if getattr(spec, name, None) != value
        }
        expected_fields = tuple(
            (field.name, field.shape, field.dtype)
            for field in policy_observation_fields(self.spec)
        )
        loaded_fields = tuple(
            (field.name, field.shape, field.dtype)
            for field in policy_observation_fields(spec)
        )
        if loaded_fields != expected_fields:
            mismatches["observation_fields"] = (loaded_fields, expected_fields)
        if mismatches:
            raise ValueError(
                "loaded PolicySpec does not match the Real tensor/action boundary: "
                f"{mismatches}"
            )

    def warmup(self, *, samples: int) -> tuple[float, ...]:
        return self._require_open().warmup(samples=samples)

    def reset_episode(self) -> None:
        self._require_open().reset_episode()

    def predict(self, observation: PolicyObservation) -> PolicyPrediction:
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
        self, observation: PolicyObservation
    ) -> dict[str, np.ndarray]:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("policy observation must be a PolicyObservation")
        fields = policy_observation_fields(self.spec)
        field_names = tuple(field.name for field in fields)
        if tuple(observation.arrays) != field_names:
            raise ValueError("PolicyObservation fields do not match PolicySpec")
        n_obs = int(self.spec.n_obs_steps)
        for field in fields:
            array = observation.arrays[field.name]
            expected_dtype = np.dtype(field.dtype)
            expected_shape = (n_obs, *field.shape)
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(
                    f"PolicyObservation {field.name} must be {expected_dtype} "
                    f"{expected_shape}, got dtype={array.dtype} shape={array.shape}"
                )
        return dict(observation.arrays)

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
