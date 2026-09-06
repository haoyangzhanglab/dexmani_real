"""Pure cross-repository deployment-contract smoke test."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from dexmani_policy.deployment.contract import (
    DEPLOYMENT_FORMAT,
    DEPLOYMENT_SCHEMA_VERSION,
    parse_deployment_contract,
)
from dexmani_policy.deployment.runtime import PolicySpec

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.config.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.deployment.config import validate_policy_runtime_compatibility
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
)


class PolicyCrossRepositoryContractTest(unittest.TestCase):
    def test_real_zarr_policy_contract_and_real_runtime_agree(self) -> None:
        runtime = resolve_experiment_config()
        table = runtime.environment.table
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = zarr.open_group(str(path), mode="w")
            root.attrs.update(
                {
                    "schema_name": "dexmani-real-policy-zarr",
                    "schema_version": 7,
                    "domain": "real",
                    "profile": "pointcloud",
                    "task_name": "task",
                    "dt": 1.0 / runtime.policy.control_hz,
                    "episode_start_policy": "full_history",
                    "obs_alignment": "obs[t]_before_action[t]",
                    "observation_reference": "camera_source_monotonic_ns",
                    "state_alignment": "camera_source_aligned_state",
                    "action_semantics": "teleop_published_joint_target",
                    "point_cloud_frame": "xarm_base",
                    "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
                    "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
                    "point_cloud_table_plane_abcd_json": json.dumps(
                        table.plane_abcd if table.enabled else None,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "point_cloud_sampling": POINT_CLOUD_SAMPLING,
                    "point_cloud_transform": POINT_CLOUD_TRANSFORM,
                    "fingertip_points_frame": "xarm_base",
                    "fingertip_points_unit": "m",
                    "fingertip_points_derivation": FINGERTIP_POINTS_DERIVATION,
                    "fingertip_points_policy_id": FINGERTIP_POLICY_ID,
                }
            )
            data = root.create_group("data")
            data.create_dataset("joint_state", data=np.zeros((1, 19), np.float32))
            data.create_dataset("action", data=np.zeros((1, 19), np.float32))
            data.create_dataset("action_ee", data=np.zeros((1, 21), np.float32))
            data.create_dataset(
                "point_cloud",
                data=np.zeros((1, runtime.pointcloud.num_points, 6), np.float32),
            )
            data.create_dataset(
                "fingertip_points", data=np.zeros((1, 5, 3), np.float32)
            )
            data_contract = {
                "schema_name": "dexmani-real-policy-zarr",
                "schema_version": 7,
                "domain": "real",
                "profile": "pointcloud",
                "task_name": "task",
                "dt": 1.0 / runtime.policy.control_hz,
                "obs_alignment": "obs[t]_before_action[t]",
                "observation_reference": "camera_source_monotonic_ns",
                "state_alignment": "camera_source_aligned_state",
                "action_semantics": "teleop_published_joint_target",
                "requires_hand": True,
                "observation_fields": {
                    "joint_state": {
                        "shape": [19],
                        "dtype": "float32",
                        "semantics": {
                            "representation": "joint_position",
                            "frame": "robot_joint",
                            "units": "rad",
                            "joint_order": "xarm7_xhand12",
                        },
                    },
                    "point_cloud": {
                        "shape": [runtime.pointcloud.num_points, 6],
                        "dtype": "float32",
                        "semantics": {
                            "representation": "xyzrgb",
                            "frame": "xarm_base",
                            "position_units": "m",
                            "color_order": "rgb",
                            "color_source": POINT_CLOUD_COLOR_SOURCE,
                            "policy_id": POINT_CLOUD_POLICY_ID,
                            "table_plane_abcd_json": json.dumps(
                                table.plane_abcd if table.enabled else None,
                                separators=(",", ":"),
                                allow_nan=False,
                            ),
                            "sampling": POINT_CLOUD_SAMPLING,
                            "transform": POINT_CLOUD_TRANSFORM,
                        },
                    },
                    "fingertip_points": {
                        "shape": [5, 3],
                        "dtype": "float32",
                        "semantics": {
                            "representation": "point_xyz",
                            "frame": "xarm_base",
                            "units": "m",
                            "finger_order": "thumb_index_mid_ring_pinky",
                            "derivation": FINGERTIP_POINTS_DERIVATION,
                            "policy_id": FINGERTIP_POLICY_ID,
                        },
                    },
                },
            }

        deployment_spec = parse_deployment_contract(
            {
                "_format": DEPLOYMENT_FORMAT,
                "contract": {
                    "schema_version": DEPLOYMENT_SCHEMA_VERSION,
                    "inference_config": {
                        "action_key": "action",
                        "action_dim": 19,
                        "horizon": 16,
                        "n_obs_steps": 2,
                        "n_action_steps": 8,
                        "eval": {
                            "denoise_steps": 1,
                            "temporal_ensemble_coeff": None,
                        },
                    },
                    "data_contract": data_contract,
                    "producer": {},
                },
                "weights": {"synthetic": True},
            }
        )
        policy_spec = PolicySpec(
            action_key=deployment_spec.action_key,
            action_dim=deployment_spec.action_dim,
            control_action_dim=deployment_spec.control_action_dim,
            horizon=deployment_spec.horizon,
            n_obs_steps=deployment_spec.n_obs_steps,
            n_action_steps=deployment_spec.n_action_steps,
            temporal_ensemble_coeff=deployment_spec.temporal_ensemble_coeff,
            observation_fields=deployment_spec.observation_fields,
            control_dt_s=deployment_spec.control_dt_s,
            requires_hand=deployment_spec.requires_hand,
            rgb_preprocessing=deployment_spec.rgb_preprocessing,
        )

        validate_policy_runtime_compatibility(policy_spec, runtime)


if __name__ == "__main__":
    unittest.main()
