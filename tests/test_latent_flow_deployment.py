"""Offline future-window contract; no devices or policy rollout are constructed."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from dexmani_policy.deployment.metadata import PolicyInfo
from dexmani_real.deployment.preflight import validate_execution_config


class FutureWindowTests(unittest.TestCase):
    def test_full_latent_flow_horizon_and_original_policy_limit(self):
        cfg = OmegaConf.create({
            "policy_name": "latent_flow", "dataset": {"sensor_modalities": ["joint_state"]},
            "real_runtime": {"dt": 0.0625},
            "agent": {
                "_target_": "dexmani_policy.agents.policies.latent_flow.LatentFlowPolicy",
                "horizon": 32, "n_obs_steps": 2, "n_action_steps": 32,
            },
        })
        execution = SimpleNamespace(
            execution_mode="sync", action_steps=32, start_arm_home_tolerance_deg=5,
            max_decision_age_s=1, max_wait_s=1,
        )
        info = PolicyInfo(Path("."), Path("unused.pt"), cfg, "ema", 10)
        self.assertEqual(info.future_steps, 32)
        validate_execution_config(info, execution)
        cfg.agent._target_ = "dexmani_policy.agents.policies.dp3.DP3Policy"
        old = PolicyInfo(Path("."), Path("unused.pt"), cfg, "raw", 10)
        self.assertEqual(old.future_steps, 31)
        with self.assertRaisesRegex(ValueError, "future_steps"):
            validate_execution_config(old, execution)
        execution.action_steps = 31
        validate_execution_config(old, execution)


if __name__ == "__main__":
    unittest.main()
