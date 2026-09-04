"""Offline tests for the narrow Policy deployment configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dexmani_real.deployment.config import (
    PolicyDeploymentConfig,
    resolve_policy_deployment_config,
)


class PolicyDeploymentConfigTest(unittest.TestCase):
    def test_defaults_are_sync_and_unbounded(self) -> None:
        config = resolve_policy_deployment_config()

        self.assertEqual(config, PolicyDeploymentConfig())
        self.assertEqual(config.inference_mode, "sync")
        self.assertIsNone(config.max_action_steps)

    def test_yaml_values_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy_deployment.yaml"
            path.write_text(
                "inference:\n  mode: async\nepisode:\n  max_action_steps: 12\n",
                encoding="utf-8",
            )

            config = resolve_policy_deployment_config(yaml_path=path)

        self.assertEqual(config.inference_mode, "async")
        self.assertEqual(config.max_action_steps, 12)

    def test_cli_values_override_yaml_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy_deployment.yaml"
            path.write_text(
                "inference:\n  mode: async\nepisode:\n  max_action_steps: 12\n",
                encoding="utf-8",
            )

            config = resolve_policy_deployment_config(
                yaml_path=path,
                inference_mode="sync",
                max_action_steps=3,
            )

        self.assertEqual(config.inference_mode, "sync")
        self.assertEqual(config.max_action_steps, 3)

    def test_unknown_fields_and_invalid_values_fail_closed(self) -> None:
        invalid_documents = (
            "extra: true\n",
            "inference:\n  extra: true\n",
            "episode:\n  extra: true\n",
            "inference:\n  mode: invalid\n",
            "episode:\n  max_action_steps: true\n",
            "episode:\n  max_action_steps: 0\n",
        )
        for document in invalid_documents:
            with self.subTest(document=document):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "policy_deployment.yaml"
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises((TypeError, ValueError)):
                        resolve_policy_deployment_config(yaml_path=path)

    def test_constructor_rejects_bool_as_action_limit(self) -> None:
        with self.assertRaises(ValueError):
            PolicyDeploymentConfig(max_action_steps=True)


if __name__ == "__main__":
    unittest.main()
