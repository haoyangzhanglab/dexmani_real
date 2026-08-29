"""Offline contract checks for order-independent sensor modalities."""

from __future__ import annotations

import unittest

from dexmani_real.deployment.manifest import manifest_from_sources


def _manifest(*, modalities: tuple[str, ...]):
    return manifest_from_sources(
        action_key="action",
        n_obs_steps=4,
        n_action_steps=8,
        action_dim=19,
        horizon=16,
        tcp_dim=None,
        hand_dim=None,
        control_action_dim=19,
        sensor_modalities=modalities,
        point_cloud_num_points=1024,
        point_cloud_feature_dim=6,
    )


class DeploymentManifestTest(unittest.TestCase):
    def test_sensor_modality_order_is_canonicalized(self) -> None:
        manifest = _manifest(modalities=("point_cloud", "joint_state"))
        self.assertEqual(manifest.sensor_modalities, ("joint_state", "point_cloud"))

    def test_duplicate_sensor_modality_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            _manifest(modalities=("joint_state", "point_cloud", "joint_state"))
