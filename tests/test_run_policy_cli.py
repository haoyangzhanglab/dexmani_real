"""Offline routing contracts for the learned-policy CLI."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "run_policy.py"
_MODULE_NAME = "_test_run_policy_cli_target"


def _load_cli_module() -> object:
    sys.modules.pop(_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run_policy.py for testing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _experiment_info() -> SimpleNamespace:
    spec = SimpleNamespace(
        observation_fields=(
            SimpleNamespace(name="joint_state", shape=(19,), dtype="float32"),
            SimpleNamespace(name="point_cloud", shape=(1024, 6), dtype="float32"),
        ),
        action_dim=19,
        horizon=16,
        n_obs_steps=2,
        action_key="action",
        control_action_dim=19,
        n_action_steps=8,
        control_dt_s=0.0625,
        requires_hand=True,
    )
    return SimpleNamespace(
        selector="dp3/pick/seed0",
        task_name="pick",
        policy_name="DP3",
        checkpoint_name="deployment_latest.pt",
        spec=spec,
    )


class RunPolicyParserTest(unittest.TestCase):
    def test_check_exposes_only_its_device_option(self) -> None:
        cli = _load_cli_module()
        args = cli._parser().parse_args(
            [
                "check",
                "dp3/pick/seed0",
                "--device",
                "cpu",
            ]
        )

        self.assertEqual(args.command, "check")
        self.assertEqual(args.experiment, "dp3/pick/seed0")
        self.assertEqual(args.device, "cpu")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli._parser().parse_args(
                ["check", "dp3/pick/seed0", "--runtime-config", "real.yaml"]
            )

    def test_deployment_options_work_before_and_after_subcommand(self) -> None:
        cli = _load_cli_module()
        parser = cli._parser()

        before = parser.parse_args(
            [
                "--deployment-config",
                "policy.yaml",
                "--inference-mode",
                "async",
                "--max-action-steps",
                "12",
                "run",
                "dp3/pick/seed0",
            ]
        )
        self.assertEqual(before.deployment_config, "policy.yaml")
        self.assertEqual(before.inference_mode, "async")
        self.assertEqual(before.max_action_steps, 12)

        after = parser.parse_args(
            [
                "shadow",
                "dp3/pick/seed0",
                "--deployment-config",
                "policy.yaml",
                "--inference-mode",
                "sync",
                "--max-action-steps",
                "3",
                "--runtime-config",
                "runtime.yaml",
            ]
        )
        self.assertEqual(after.deployment_config, "policy.yaml")
        self.assertEqual(after.inference_mode, "sync")
        self.assertEqual(after.max_action_steps, 3)
        self.assertEqual(after.runtime_config, "runtime.yaml")

    def test_precommand_options_are_rejected_when_unowned(self) -> None:
        cli = _load_cli_module()
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main(["--deployment-config", "policy.yaml", "check", "dp3/pick/seed0"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "[CLI] check does not accept --deployment-config", stderr.getvalue()
        )

        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.main(["--device", "cpu", "list"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("[CLI] list does not accept --device", stderr.getvalue())


class RunPolicyCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cli = _load_cli_module()

    def test_list_calls_only_policy_list_api(self) -> None:
        stdout = io.StringIO()
        with (
            patch.object(
                self.cli,
                "_list_policy_experiments",
                return_value=("dp3/pick/seed0",),
            ) as list_experiments,
            patch.object(self.cli, "_inspect_policy_experiment") as inspect,
            patch.object(self.cli, "_load_policy_experiment") as load,
            patch.object(self.cli, "_prepare_lifecycle_inputs") as prepare,
            contextlib.redirect_stdout(stdout),
        ):
            result = self.cli.main(["list", "DP3"])

        self.assertEqual(result, 0)
        list_experiments.assert_called_once_with("DP3")
        inspect.assert_not_called()
        load.assert_not_called()
        prepare.assert_not_called()
        self.assertIn("dp3/pick/seed0", stdout.getvalue())

    def test_check_loads_warms_and_closes_without_lifecycle_projection(self) -> None:
        info = _experiment_info()
        policy = SimpleNamespace(warmup=Mock(return_value=(0.010, 0.012, 0.011)))
        policy.close = Mock()
        stdout = io.StringIO()
        with (
            patch.object(
                self.cli, "_inspect_policy_experiment", return_value=info
            ) as inspect,
            patch.object(
                self.cli, "_load_policy_experiment", return_value=policy
            ) as load,
            patch.object(self.cli, "_prepare_lifecycle_inputs") as prepare,
            contextlib.redirect_stdout(stdout),
        ):
            result = self.cli.main(["check", "dp3/pick/seed0", "--device", "cpu"])

        self.assertEqual(result, 0)
        inspect.assert_called_once_with("dp3/pick/seed0")
        load.assert_called_once_with("dp3/pick/seed0", "cpu")
        policy.warmup.assert_called_once_with(samples=3)
        policy.close.assert_called_once_with()
        prepare.assert_not_called()
        self.assertIn("prediction ....... OK", stdout.getvalue())
        self.assertIn("READY", stdout.getvalue())

    def test_lifecycle_projection_carries_only_execute_intent(self) -> None:
        info = _experiment_info()
        args = self.cli._parser().parse_args(["run", info.selector])
        runtime = object()
        with (
            patch(
                "dexmani_real.config.runtime.resolve_runtime_config",
                return_value=runtime,
            ) as resolve_runtime,
            patch(
                "dexmani_real.deployment.config.validate_policy_runtime_compatibility"
            ) as validate_compatibility,
        ):
            inputs = self.cli._prepare_lifecycle_inputs(args, info, execute=True)

        self.assertIs(inputs.execute, True)
        self.assertIs(inputs.runtime, runtime)
        self.assertIs(inputs.policy_spec, info.spec)
        self.assertEqual(inputs.worker_config.experiment, info.selector)
        self.assertEqual(inputs.worker_config.device, args.device)
        self.assertEqual(inputs.worker_config.seed, 0)
        self.assertIs(inputs.worker_config.spec, info.spec)
        resolve_runtime.assert_called_once_with(yaml_path=args.runtime_config)
        validate_compatibility.assert_called_once_with(info.spec, runtime)
        for retired_name in (
            "execution_mode",
            "hand_acknowledged",
            "h4_execute_bounds",
            "task_execute_bounds",
            "inference_seed",
        ):
            self.assertFalse(hasattr(inputs.worker_config, retired_name))

    def test_shadow_projects_validate_only_and_starts_through_seam(self) -> None:
        info = _experiment_info()
        inputs = SimpleNamespace(execute=False)
        with (
            patch.object(self.cli, "_inspect_policy_experiment", return_value=info),
            patch.object(
                self.cli, "_prepare_lifecycle_inputs", return_value=inputs
            ) as prepare,
            patch.object(self.cli, "_start_lifecycle", return_value=0) as start,
        ):
            result = self.cli.main(["shadow", "dp3/pick/seed0"])

        self.assertEqual(result, 0)
        prepare.assert_called_once()
        self.assertIs(prepare.call_args.kwargs["execute"], False)
        start.assert_called_once_with(inputs)

    def test_start_lifecycle_forwards_deployment_config(self) -> None:
        deployment_config = object()
        inputs = SimpleNamespace(
            runtime=object(),
            policy_spec=object(),
            worker_config=object(),
            execute=False,
            deployment_config=deployment_config,
        )
        with patch(
            "dexmani_real.deployment.lifecycle.run_policy_deployment",
            return_value=0,
        ) as run:
            result = self.cli._start_lifecycle(inputs)

        self.assertEqual(result, 0)
        run.assert_called_once_with(
            inputs.runtime,
            inputs.policy_spec,
            inputs.worker_config,
            inputs.execute,
            deployment_config=deployment_config,
        )

    def test_run_projects_physical_publication_and_starts_through_seam(self) -> None:
        info = _experiment_info()
        inputs = SimpleNamespace(execute=True)
        with (
            patch.object(self.cli, "_inspect_policy_experiment", return_value=info),
            patch.object(
                self.cli, "_prepare_lifecycle_inputs", return_value=inputs
            ) as prepare,
            patch.object(self.cli, "_start_lifecycle", return_value=0) as start,
        ):
            result = self.cli.main(["run", "dp3/pick/seed0"])

        self.assertEqual(result, 0)
        prepare.assert_called_once()
        self.assertIs(prepare.call_args.kwargs["execute"], True)
        start.assert_called_once_with(inputs)

    def test_lifecycle_failures_are_not_reported_as_compatibility_failures(
        self,
    ) -> None:
        info = _experiment_info()
        inputs = SimpleNamespace(execute=False)
        stderr = io.StringIO()
        with (
            patch.object(self.cli, "_inspect_policy_experiment", return_value=info),
            patch.object(self.cli, "_prepare_lifecycle_inputs", return_value=inputs),
            patch.object(
                self.cli, "_start_lifecycle", side_effect=RuntimeError("worker lost")
            ),
            contextlib.redirect_stderr(stderr),
        ):
            result = self.cli.main(["shadow", "dp3/pick/seed0"])

        self.assertEqual(result, 1)
        self.assertIn("[LIFECYCLE] lifecycle failed: worker lost", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
