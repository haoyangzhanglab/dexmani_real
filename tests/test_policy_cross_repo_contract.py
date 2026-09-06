"""Pure cross-repository deployment-contract smoke test."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dexmani_policy.deployment import export as policy_export
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
    compute_fingertip_geometry_sha256,
)
from dexmani_real.robot.model import (
    HAND_SDK_TO_URDF_IDX,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XHAND_RIGHT_URDF_PATH,
)
from dexmani_real.utils.atomic_io import sha256_file


class PolicyCrossRepositoryContractTest(unittest.TestCase):
    def test_real_zarr_v7_policy_contract_and_real_runtime_agree(self) -> None:
        runtime = resolve_experiment_config()
        table = runtime.environment.table
        geometry_sha256 = compute_fingertip_geometry_sha256(
            arm_fk_urdf_sha256=sha256_file(XARM7_XHAND_COLLISION_URDF_PATH),
            arm_eef_frame="custom_eef_link",
            hand_fk_urdf_sha256=sha256_file(XHAND_RIGHT_URDF_PATH),
            hand_sdk_to_urdf_idx=HAND_SDK_TO_URDF_IDX,
            fingertip_link_names=runtime.hand.fingertip_link_names,
            handbase_position_eef_m=runtime.hand.T_eef_handbase_pos_xyz,
            handbase_quat_eef_wxyz=runtime.hand.T_eef_handbase_quat_wxyz,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.zarr"
            root = policy_export.zarr.open_group(str(path), mode="w")
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
                    "point_cloud_config_sha256": runtime.pointcloud.sha256,
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
                    "fingertip_points_geometry_sha256": geometry_sha256,
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
            data_contract = policy_export._build_observation_contract(
                path,
                {
                    "task_name": "task",
                    "action_key": "action",
                    "dt": 1.0 / runtime.policy.control_hz,
                    "agent": {
                        "num_points": runtime.pointcloud.num_points,
                        "pc_dim": 6,
                    },
                },
                ["joint_state", "point_cloud", "fingertip_points"],
            )

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
