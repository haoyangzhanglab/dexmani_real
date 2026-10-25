#!/usr/bin/env python3
"""离线检查 Policy/Real 名称与入口契约；不加载模型或连接设备。"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from dexmani_policy.common.config import PROJECT_ROOT, load_policy_config
from dexmani_policy.deployment import PolicyInfo, inspect_policy
from omegaconf import OmegaConf

from dexmani_real.config import ExperimentConfig
from dexmani_real.deployment.config import ExecutionConfig
from dexmani_real.deployment.preflight import validate_execution_config
from dexmani_real.deployment.run import run_policy, session_directory


class DeploymentContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiment = self.root / "experiments/dp/offline_task/run_seed7"
        self.checkpoint = self.experiment / "checkpoints/milestone.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()  # Inspection resolves paths without reading weights.
        (self.checkpoint.parent / "latest.pt").symlink_to(self.checkpoint.name)
        self.config = {
            "policy_name": "dp", "task_name": "offline_task", "action_key": "action",
            "agent": {"horizon": 8, "n_obs_steps": 2, "n_action_steps": 2, "action_dim": 19},
            "dataset": {"sensor_modalities": ["joint_state"]},
            "normalization": {"joint_state": "limits", "action": "limits"},
            "policy_metadata": {
                "action_start_index": 1, "control_action_dim": 19,
                "supported_execution_modes": ["sync", "async", "rtc"],
                "action_context_shapes": {},
            },
            "real_runtime": {"dt": 0.0625},
            "eval": {"use_ema": True, "inference_steps": 10, "action_steps": 4},
        }
        (self.experiment / "config.yaml").write_text(yaml.safe_dump(self.config))
        (self.experiment / "best_ckpt.json").write_text(json.dumps({
            "global_step": 1, "ckpt_relpath": "checkpoints/milestone.pt",
            "selection_result": "eval_ckpt_selector/selection_result.json",
            "inference": {"use_ema": False, "inference_steps": 3, "action_steps": 6},
        }))
        self.info = inspect_policy(
            self.experiment, checkpoint=self.checkpoint, weights="ema", inference_steps=10,
        )
        runtime = ExperimentConfig()
        self.runtime = replace(runtime, environment=replace(
            runtime.environment, table=replace(runtime.environment.table, enabled=False),
        ))

    def test_ddp_policy_info_uses_method_task_and_run_for_output(self):
        overlays = sorted((PROJECT_ROOT / "dexmani_policy/configs/ddp").glob("*.yaml"))
        self.assertTrue(overlays)
        for overlay in overlays:
            with self.subTest(selector=f"ddp/{overlay.stem}"):
                base = load_policy_config(overlay.stem)
                cfg = load_policy_config(f"ddp/{overlay.stem}")
                self.assertEqual(cfg.policy_name, base.policy_name)
                self.assertGreater(cfg.training.num_gpus, 1)
                # Only identity crosses here; this fixture does not assert model deployability.
                saved = OmegaConf.merge(self.info.saved_config, {"policy_name": cfg.policy_name})
                info = replace(self.info, saved_config=saved)
                self.assertIsInstance(info, PolicyInfo)
                with patch("dexmani_real.deployment.run.time.strftime", return_value="20000101_000000"):
                    first = session_directory(self.root / "rollouts", info)
                    second = session_directory(self.root / "rollouts", info)
                self.assertEqual(first.relative_to(self.root / "rollouts"), Path(
                    base.policy_name, "offline_task", "run_seed7", "session_20000101_000000",
                ))
                self.assertTrue(first.is_dir())
                self.assertTrue(second.is_dir())
                self.assertEqual(second.name, "session_20000101_000000_01")

    def test_invalid_directory_components_are_rejected_before_output(self):
        valid = dict(policy_name="dp", task_name="task", experiment_dir=Path("run"))
        for field in valid:
            for value in (None, 7, "", ".", "..", "../escape", "a/b", "a\\b", "/absolute"):
                with self.subTest(field=field, value=value):
                    parts = {**valid, field: (
                        SimpleNamespace(name=value) if field == "experiment_dir" else value
                    )}
                    with self.assertRaisesRegex(ValueError, "single directory components"):
                        session_directory(self.root / "rejected", SimpleNamespace(**parts))
                    self.assertFalse((self.root / "rejected").exists())

    def test_symlink_escape_is_rejected(self):
        output = self.root / "rollouts"
        outside = self.root / "outside"
        output.mkdir()
        outside.mkdir()
        (output / self.info.policy_name).symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside its root"):
            session_directory(output, self.info)
        self.assertEqual(list(outside.iterdir()), [])

    def test_checkpoint_inference_defaults_and_real_execution_stay_separate(self):
        cases = (
            ("latest", None, None, ExecutionConfig(), "ema", 10, 2),
            ("best", None, None, ExecutionConfig(), "raw", 3, 2),
            ("best", "ema", 5, ExecutionConfig(
                execution_mode="async", action_steps=3, prefetch_steps=1,
            ), "ema", 5, 3),
        )
        for checkpoint, weights, steps, execution, expected_weights, expected_nfe, expected_actions in cases:
            with self.subTest(checkpoint=checkpoint, execution=execution):
                with (patch("dexmani_real.deployment.run.PolicySession") as session,
                      redirect_stdout(io.StringIO())):
                    session.return_value.run.return_value = 0
                    result = run_policy(
                        self.experiment, replace(self.runtime, execution=execution),
                        checkpoint=checkpoint, weights=weights, inference_steps=steps,
                        record_raw=False, output=self.root / "rollouts",
                    )
                self.assertEqual(result, 0)
                runtime, info = session.call_args.args
                self.assertEqual(info.checkpoint_path, self.checkpoint)
                self.assertEqual((info.weights, info.inference_steps), (expected_weights, expected_nfe))
                self.assertEqual(runtime.execution, replace(execution, action_steps=expected_actions))
                payload = yaml.safe_load((session.call_args.kwargs["results_dir"] / "run_config.yaml").read_text())
                self.assertEqual(payload["runtime"]["execution"]["action_steps"], expected_actions)
                self.assertEqual(payload["policy"]["policy_name"], "dp")

    def test_deployment_restrictions_remain_enforced(self):
        with self.assertRaisesRegex(ValueError, "async/rtc require"):
            validate_execution_config(self.info, ExecutionConfig(execution_mode="async", action_steps=2))
        saved = OmegaConf.merge(self.info.saved_config, {
            "policy_metadata": {"supported_execution_modes": ["sync"]},
        })
        with self.assertRaisesRegex(NotImplementedError, "does not support rtc"):
            validate_execution_config(replace(self.info, saved_config=saved), ExecutionConfig(
                execution_mode="rtc", action_steps=2, prefetch_steps=1, rtc_guidance_cap=0,
            ))
        for changes, exception, message in (
            ({"dataset": {"sensor_modalities": ["tactile_valid"]}}, ValueError, "no live producer"),
            ({"dataset": {"datasets": []}}, NotImplementedError, "MultiTask Real deployment"),
        ):
            with self.subTest(changes=changes):
                (self.experiment / "config.yaml").write_text(yaml.safe_dump({
                    **self.config, "dataset": changes["dataset"],
                }))
                with (patch("dexmani_real.deployment.run.PolicySession") as session,
                      self.assertRaisesRegex(exception, message)):
                    run_policy(self.experiment, self.runtime, record_raw=False, output=self.root / "rejected")
                session.assert_not_called()
                self.assertFalse((self.root / "rejected").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
