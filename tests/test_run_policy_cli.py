"""Offline routing tests for the Phase 2 policy CLI."""

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
        sensor_modalities=("joint_state", "point_cloud"),
        point_cloud_num_points=1024,
        point_cloud_feature_dim=6,
        n_obs_steps=2,
        action_key="action",
        control_action_dim=19,
        n_action_steps=8,
        control_dt_s=0.0625,
    )
    return SimpleNamespace(
        selector="dp3/pick/seed0",
        task_name="pick",
        policy_name="DP3",
        checkpoint_name="deployment_latest.pt",
        spec=spec,
    )


class RunPolicyParserTest(unittest.TestCase):
    def test_subcommands_only_expose_supported_advanced_options(self) -> None:
        cli = _load_cli_module()
        args = cli._parser().parse_args(
            ["check", "dp3/pick/seed0", "--device", "cpu", "--config", "real.yaml"]
        )

        self.assertEqual(args.command, "check")
        self.assertEqual(args.experiment, "dp3/pick/seed0")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.config, "real.yaml")
        help_text = cli._parser().format_help()
        for retired_flag in (
            "--execution-mode",
            "--print-config",
            "--preflight-only",
            "--execute-",
            "--task-",
            "--hand",
        ):
            self.assertNotIn(retired_flag, help_text)


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
        projection = object()
        with (
            patch(
                "dexmani_real.config.runtime.resolve_runtime_config",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.config.resolve_policy_runtime_config",
                return_value=projection,
            ) as resolve_projection,
        ):
            inputs = self.cli._prepare_lifecycle_inputs(args, info, execute=True)

        self.assertIs(inputs.execute, True)
        self.assertIs(inputs.runtime, runtime)
        self.assertIs(inputs.projection, projection)
        self.assertIs(resolve_projection.call_args.kwargs["execute"], True)
        for retired_name in (
            "execution_mode",
            "hand_acknowledged",
            "h4_execute_bounds",
            "task_execute_bounds",
            "inference_seed",
        ):
            self.assertNotIn(retired_name, resolve_projection.call_args.kwargs)

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


if __name__ == "__main__":
    unittest.main()
