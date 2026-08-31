"""Offline contract checks for order-independent sensor modalities."""

from __future__ import annotations

import unittest

from dexmani_real.deployment.manifest import manifest_from_sources


def _manifest(*, modalities: tuple[str, ...]):
    uses_point_cloud = "point_cloud" in modalities
    return manifest_from_sources(
        action_key="action",
        n_obs_steps=4,
        n_action_steps=8,
        action_dim=19,
        horizon=16,
        tcp_dim=None,
        hand_dim=None,
        control_action_dim=19,
        auxiliary_action_layout="none",
        sensor_modalities=modalities,
        point_cloud_num_points=1024 if uses_point_cloud else None,
        point_cloud_feature_dim=6 if uses_point_cloud else None,
        rgb_shape=(480, 640, 3) if "rgb" in modalities else None,
        rgb_color_order="rgb" if "rgb" in modalities else None,
        rgb_value_range="uint8_0_255" if "rgb" in modalities else None,
    )


class DeploymentManifestTest(unittest.TestCase):
    def test_sensor_modality_order_is_canonicalized(self) -> None:
        manifest = _manifest(modalities=("point_cloud", "joint_state"))
        self.assertEqual(manifest.sensor_modalities, ("joint_state", "point_cloud"))

    def test_duplicate_sensor_modality_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            _manifest(modalities=("joint_state", "point_cloud", "joint_state"))

    def test_rgb_only_contract_is_supported(self) -> None:
        manifest = _manifest(modalities=("joint_state", "rgb"))
        self.assertTrue(manifest.uses_rgb)
        self.assertFalse(manifest.uses_point_cloud)
        self.assertEqual(manifest.rgb_shape, (480, 640, 3))
