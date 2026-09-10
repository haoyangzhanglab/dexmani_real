"""Point-cloud and fingertip numeric train/deploy contract regressions.

Offline only: no hardware, no shared memory, no policy checkpoint.  Pins that
the deployment expected semantics carry the full numeric derivation config
(``processing_config_json``) and fingertip FK geometry (``fingertip_config_json``)
as canonical JSON, byte-identical to what processing persists into the artifact,
and that any physics-changing drift fails the exact semantics compare.

Run with:

    python -m pytest -q tests/test_deployment_numeric_contracts.py
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import h5py
import pytest

from dexmani_real.config.defaults import hand as HAND_DEFAULTS
from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.dataset.contracts import ProcessingConfig, canonical_json
from dexmani_real.dataset.processing import process_episode_root
from dexmani_real.deployment.config import (
    _expected_fingertip_semantics,
    _expected_pointcloud_semantics,
    validate_policy_runtime_compatibility,
)
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from test_control_step_dataset import write_control_episode

# The canonical outlier radius rejects the flat synthetic plane entirely; the
# permissive policy mirrors test_control_step_dataset.permissive_test_config.
_PERMISSIVE_POINTCLOUD = dataclasses.replace(
    PointCloudConfig(),
    outlier_min_neighbors=0,
    outlier_min_component_points=1,
)


def _contract_runtime(
    pointcloud: PointCloudConfig | None = None,
    *,
    table_enabled: bool = False,
    table_plane: tuple[float, float, float, float] | None = None,
    hand_config=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        policy=SimpleNamespace(control_hz=16.0),
        pointcloud=_PERMISSIVE_POINTCLOUD if pointcloud is None else pointcloud,
        environment=SimpleNamespace(
            table=SimpleNamespace(enabled=table_enabled, plane_abcd=table_plane)
        ),
        hand=HAND_DEFAULTS if hand_config is None else hand_config,
    )


def _hand_config(
    *,
    link_names=XHAND_FINGERTIP_LINK_NAMES,
    position=HAND_DEFAULTS.T_eef_handbase_pos_xyz,
    quaternion=HAND_DEFAULTS.T_eef_handbase_quat_wxyz,
) -> SimpleNamespace:
    return SimpleNamespace(
        fingertip_link_names=tuple(link_names),
        T_eef_handbase_pos_xyz=tuple(position),
        T_eef_handbase_quat_wxyz=tuple(quaternion),
    )


class _Field:
    def __init__(self, name, shape, dtype, semantics=None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.semantics = semantics if semantics is not None else {}


def _policy_spec(*fields: _Field) -> SimpleNamespace:
    names = tuple(field.name for field in fields)
    if "joint_state" not in names:
        fields = (_Field("joint_state", (19,), "float32"),) + fields
    return SimpleNamespace(
        observation_fields=fields,
        n_obs_steps=2,
        n_action_steps=4,
        chunk_size=15,
        requires_hand=True,
        action_key="action",
        control_action_dim=19,
        control_dt_s=1.0 / 16.0,
    )


def _pointcloud_spec(semantics) -> SimpleNamespace:
    return _policy_spec(_Field("point_cloud", (1024, 6), "float32", dict(semantics)))


def _fingertip_spec(semantics) -> SimpleNamespace:
    return _policy_spec(_Field("fingertip_points", (5, 3), "float32", dict(semantics)))


class TestProcessingConfigJsonContract:
    """The full numeric point-cloud config must survive train -> deploy."""

    def test_identical_numeric_config_passes(self):
        runtime = _contract_runtime()
        semantics = _expected_pointcloud_semantics(runtime)
        validate_policy_runtime_compatibility(_pointcloud_spec(semantics), runtime)

    @pytest.mark.parametrize(
        "drift",
        [
            {"voxel_size_m": 0.006},
            {"outlier_radius_m": 0.013},
            {"workspace": (0.0, -0.5, 0.0, 0.85, 0.5, 0.8)},
            {"depth_max_m": 1.45},
        ],
    )
    def test_physics_changing_drift_fails(self, drift):
        runtime = _contract_runtime()
        drifted_runtime = _contract_runtime(
            dataclasses.replace(runtime.pointcloud, **drift)
        )
        artifact_semantics = _expected_pointcloud_semantics(drifted_runtime)
        with pytest.raises(
            ValueError, match="point_cloud processing_config_json mismatch"
        ):
            validate_policy_runtime_compatibility(
                _pointcloud_spec(artifact_semantics), runtime
            )

    def test_table_plane_drift_fails(self):
        plane = (0.0, 0.0, 1.0, -0.022)
        runtime = _contract_runtime(table_enabled=True, table_plane=plane)
        drifted_runtime = _contract_runtime(
            table_enabled=True, table_plane=(0.0, 0.0, 1.0, -0.030)
        )
        artifact_semantics = _expected_pointcloud_semantics(drifted_runtime)
        # Either plane key proving the drift is acceptable; both carry it.
        with pytest.raises(
            ValueError,
            match=r"point_cloud (table_plane_abcd_json|processing_config_json) mismatch",
        ):
            validate_policy_runtime_compatibility(
                _pointcloud_spec(artifact_semantics), runtime
            )

    def test_persisted_attr_bytes_match_deployment_expectation(self, tmp_path):
        """End-to-end parity pin: the processed artifact attr and the runtime
        expectation are produced by different modules from one config source and
        must serialize byte-identically."""
        runtime = _contract_runtime()
        config = ProcessingConfig.from_runtime(runtime)
        episode = write_control_episode(tmp_path / "raw")
        output = tmp_path / "processed"
        process_episode_root(episode, output, config)
        with h5py.File(output / f"{episode.name}.h5", "r") as artifact:
            persisted = artifact.attrs["processing_config_json"]
        expected = _expected_pointcloud_semantics(runtime)["processing_config_json"]
        assert persisted == expected
        assert canonical_json(
            {
                "pointcloud": config.pointcloud.to_dict(),
                "table_plane_abcd": None,
            }
        ) == expected


class TestFingertipConfigJsonContract:
    """Fingertip FK numeric inputs must survive train -> deploy."""

    def test_identical_geometry_config_passes(self):
        runtime = _contract_runtime()
        semantics = _expected_fingertip_semantics(runtime)
        validate_policy_runtime_compatibility(_fingertip_spec(semantics), runtime)

    def test_link_order_mismatch_fails(self):
        runtime = _contract_runtime()
        reordered = _hand_config(
            link_names=(
                XHAND_FINGERTIP_LINK_NAMES[1],
                XHAND_FINGERTIP_LINK_NAMES[0],
                *XHAND_FINGERTIP_LINK_NAMES[2:],
            )
        )
        artifact_semantics = _expected_fingertip_semantics(
            _contract_runtime(hand_config=reordered)
        )
        with pytest.raises(
            ValueError, match="fingertip_points fingertip_config_json mismatch"
        ):
            validate_policy_runtime_compatibility(
                _fingertip_spec(artifact_semantics), runtime
            )

    def test_mount_translation_mismatch_fails(self):
        runtime = _contract_runtime()
        shifted = _hand_config(position=(-0.014, 0.0, 0.0))
        artifact_semantics = _expected_fingertip_semantics(
            _contract_runtime(hand_config=shifted)
        )
        with pytest.raises(
            ValueError, match="fingertip_points fingertip_config_json mismatch"
        ):
            validate_policy_runtime_compatibility(
                _fingertip_spec(artifact_semantics), runtime
            )

    def test_mount_quaternion_mismatch_fails(self):
        runtime = _contract_runtime()
        rotated = _hand_config(quaternion=(1.0, 0.0, 0.0, 0.0))
        artifact_semantics = _expected_fingertip_semantics(
            _contract_runtime(hand_config=rotated)
        )
        with pytest.raises(
            ValueError, match="fingertip_points fingertip_config_json mismatch"
        ):
            validate_policy_runtime_compatibility(
                _fingertip_spec(artifact_semantics), runtime
            )

    def test_persisted_attr_bytes_match_deployment_expectation(self, tmp_path):
        runtime = _contract_runtime()
        config = ProcessingConfig.from_runtime(runtime)
        episode = write_control_episode(tmp_path / "raw")
        output = tmp_path / "processed"
        process_episode_root(episode, output, config)
        with h5py.File(output / f"{episode.name}.h5", "r") as artifact:
            persisted = artifact.attrs["fingertip_config_json"]
        assert persisted == _expected_fingertip_semantics(runtime)[
            "fingertip_config_json"
        ]


def test_resolved_experiment_config_parity():
    """The canonical CLI/deployment config source serializes identically on
    both sides of the contract without any processing run."""
    runtime = resolve_experiment_config()
    config = ProcessingConfig.from_runtime(runtime)
    assert _expected_pointcloud_semantics(runtime)["processing_config_json"] == (
        canonical_json(
            {
                "pointcloud": config.pointcloud.to_dict(),
                "table_plane_abcd": (
                    None
                    if config.table_plane_abcd is None
                    else list(config.table_plane_abcd)
                ),
            }
        )
    )
    assert _expected_fingertip_semantics(runtime)["fingertip_config_json"] == (
        canonical_json(
            {
                "fingertip_link_names": list(config.fingertip_link_names),
                "handbase_position_eef_m": list(config.handbase_position_eef_m),
                "handbase_quat_eef_wxyz": list(config.handbase_quat_eef_wxyz),
            }
        )
    )
