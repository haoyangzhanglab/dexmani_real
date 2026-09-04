"""Offline tests for the narrow Policy deployment value object."""

from __future__ import annotations

import unittest

from dexmani_real.deployment.config import PolicyDeploymentConfig


class PolicyDeploymentConfigTest(unittest.TestCase):
    def test_defaults_are_sync_and_unbounded(self) -> None:
        config = PolicyDeploymentConfig()

        self.assertEqual(config.inference_mode, "sync")
        self.assertIsNone(config.max_action_steps)

    def test_explicit_values_are_preserved(self) -> None:
        config = PolicyDeploymentConfig(
            inference_mode="async", max_action_steps=12
        )
        self.assertEqual(config.inference_mode, "async")
        self.assertEqual(config.max_action_steps, 12)

    def test_invalid_values_fail_closed(self) -> None:
        for kwargs in (
            {"inference_mode": "invalid"},
            {"max_action_steps": True},
            {"max_action_steps": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PolicyDeploymentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
