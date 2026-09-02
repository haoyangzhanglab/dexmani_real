"""Offline regression tests for the R3 policy CLI, profile, check, and logging."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import dexmani_real.deployment.cli as cli
import dexmani_real.deployment.preflight as preflight
from dexmani_real.deployment.config import H4ExecuteBounds, TaskExecuteBounds
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.profile import load_physical_run_profile


def _write_experiment(root: Path) -> Path:
    root.mkdir()
    (root / "config.yaml").write_text("name: synthetic\n", encoding="utf-8")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "epoch=0001-deployment-v1.pt"
    checkpoint.write_bytes(b"offline-cli-fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = {
        "schema_version": 1,
        "checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": digest,
        },
        "embedded_contract_sha256": "b" * 64,
        "allocation": {
            "task_name": "pick_place_toy",
            "action_key": "action",
            "action_dim": 19,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "horizon": 16,
            "required_action_steps": 15,
            "control_dt_s": 0.0625,
            "sensor_modalities": ["joint_state", "point_cloud"],
            "observation_fields": ["arm_qpos", "hand_qpos", "point_cloud"],
            "requires_hand": True,
            "point_cloud_num_points": 1024,
            "point_cloud_feature_dim": 6,
        },
        "producer": {
            "repository": "haoyangzhanglab/dexmani_policy",
            "commit": "c" * 40,
            "metadata_provenance": "retrofitted",
        },
    }
    sidecar_path = checkpoint.with_name(f"{checkpoint.name}.deployment.json")
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (checkpoints / "deployment_latest.pt").symlink_to(checkpoint.name)
    return root


def _profile_payload(*, endpoints: int = 1) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_dir": "experiment",
        "runtime_config": "configs/runtime.yaml",
        "deployment_config": None,
        "device": "cpu",
        "seed": 1066,
        "hand_acknowledged": True,
        "expected_checkpoint_sha256": "a" * 64,
        "max_running_seconds": 10.0,
        "acknowledgement_timeout_seconds": 2.0,
        "max_published_endpoints": endpoints,
    }


def _write_profile(root: Path, payload: dict[str, object]) -> Path:
    import yaml

    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_task_scene_card(root: Path) -> Path:
    path = root / "scene_card.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_name": "pick_place_toy",
                "object_description": "fixture block",
                "object_start_description": "fixture start pose",
                "target_description": "fixture target",
                "success_criterion": "fixture success",
                "phase_endpoint_indices": {
                    "approach": 1,
                    "grasp": 2,
                    "lift": 3,
                    "place": 4,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


class PolicyCliTest(unittest.TestCase):
    def test_no_command_prints_help_without_heavy_import_or_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "logs"
            script = r"""
import sys
from dexmani_real.deployment.cli import main
if main([]) != 0:
    raise SystemExit("nonzero")
try:
    main(["--help"])
except SystemExit as exc:
    if exc.code != 0:
        raise
forbidden = (
    "torch", "dexmani_policy", "dexmani_real.deployment.lifecycle",
    "dexmani_real.robot.arm_worker", "dexmani_real.robot.hand_worker",
    "dexmani_real.robot.xhand", "dexmani_real.sensor.camera_worker",
    "dexmani_real.sensor.pointcloud_worker", "xarm", "xhand_controller",
    "pyrealsense2",
)
prefixes = ("torch", "dexmani_policy", "xarm", "xhand_controller", "pyrealsense2")
loaded = [name for name in sys.modules if name in forbidden or
          any(name.startswith(prefix + ".") for prefix in prefixes)]
if loaded:
    raise SystemExit(",".join(sorted(loaded)))
"""
            env = os.environ.copy()
            env["DEXMANI_LOG_DIR"] = str(log_dir)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("{inspect,check,shadow,h4,run}", result.stdout)
            self.assertFalse(log_dir.exists())

    def test_inspect_is_hardware_free_and_emits_canonical_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            experiment = _write_experiment(Path(directory) / "experiment")
            script = r"""
import json
import sys
from dexmani_real.deployment.cli import main
if main(["inspect", sys.argv[1], "--device", "cpu"]) != 0:
    raise SystemExit("inspect failed")
forbidden = (
    "torch", "dexmani_policy", "dexmani_real.deployment.lifecycle",
    "dexmani_real.robot.arm_worker", "dexmani_real.robot.hand_worker",
    "dexmani_real.robot.xhand", "dexmani_real.sensor.camera_worker",
    "dexmani_real.sensor.pointcloud_worker", "xarm", "xhand_controller",
    "pyrealsense2",
)
prefixes = ("torch", "dexmani_policy", "xarm", "xhand_controller", "pyrealsense2")
loaded = [name for name in sys.modules if name in forbidden or
          any(name.startswith(prefix + ".") for prefix in prefixes)]
if loaded:
    raise SystemExit(",".join(sorted(loaded)))
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(experiment)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["schema_version"], 1)
            self.assertEqual(receipt["artifact"]["allocation"]["n_action_steps"], 8)
            self.assertFalse(receipt["artifact"]["checkpoint_sha256_verified"])
            self.assertIn("projection_sha256", receipt["runtime"])
            self.assertIn("real_source", receipt)

    def test_check_calls_offline_api_without_hardware_owner_import(self) -> None:
        modules_before = set(sys.modules)
        artifact = SimpleNamespace()
        runtime = SimpleNamespace(sha256="r" * 64)
        projection = SimpleNamespace(runtime=object(), sha256="p" * 64)
        result = SimpleNamespace()
        offline_check = Mock(return_value=result)
        preflight_module = ModuleType("dexmani_real.deployment.preflight")
        preflight_module.run_isolated_policy_check = offline_check
        identity_module = ModuleType("dexmani_real.deployment.run_identity")
        identity_module.resolve_real_source_identity = Mock(return_value=object())
        identity_module.canonical_run_receipt_json = Mock(return_value='{"ok":true}')
        with (
            patch.object(
                cli,
                "_resolve_artifact_projection",
                return_value=(artifact, runtime, projection),
            ) as resolve,
            patch.dict(
                sys.modules,
                {
                    "dexmani_real.deployment.preflight": preflight_module,
                    "dexmani_real.deployment.run_identity": identity_module,
                },
            ),
            redirect_stdout(io.StringIO()),
        ):
            code = cli.main(
                [
                    "check",
                    "experiment",
                    "--device",
                    "cpu",
                    "--seed",
                    "1066",
                    "--benchmark-samples",
                    "3",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(resolve.call_args.kwargs["seed"], 1066)
        offline_check.assert_called_once_with(projection.runtime, benchmark_samples=3)
        new_modules = set(sys.modules) - modules_before
        for module_name in (
            "dexmani_real.robot.arm_worker",
            "dexmani_real.robot.hand_worker",
            "dexmani_real.robot.xhand",
            "dexmani_real.sensor.camera_worker",
            "dexmani_real.sensor.pointcloud_worker",
        ):
            self.assertNotIn(module_name, new_modules)
        self.assertFalse(
            any(
                name == "xarm"
                or name.startswith("xarm.")
                or name == "xhand_controller"
                or name.startswith("xhand_controller.")
                or name == "pyrealsense2"
                or name.startswith("pyrealsense2.")
                for name in new_modules
            )
        )

    def test_check_is_hardware_free_in_fresh_process(self) -> None:
        script = r"""
import io
import sys
from contextlib import redirect_stdout
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from dexmani_real.deployment import cli
artifact = SimpleNamespace()
runtime = SimpleNamespace(sha256="r" * 64)
projection = SimpleNamespace(runtime=object(), sha256="p" * 64)
preflight_module = ModuleType("dexmani_real.deployment.preflight")
preflight_module.run_isolated_policy_check = Mock(return_value=SimpleNamespace())
identity_module = ModuleType("dexmani_real.deployment.run_identity")
identity_module.resolve_real_source_identity = Mock(return_value=object())
identity_module.canonical_run_receipt_json = Mock(return_value='{"ok":true}')
with (patch.object(cli, "_resolve_artifact_projection",
                   return_value=(artifact, runtime, projection)),
      patch.dict(sys.modules, {
          "dexmani_real.deployment.preflight": preflight_module,
          "dexmani_real.deployment.run_identity": identity_module,
      }), redirect_stdout(io.StringIO())):
    if cli.main(["check", "experiment", "--device", "cpu",
                 "--seed", "1066", "--benchmark-samples", "1"]) != 0:
        raise SystemExit("check failed")
forbidden = (
    "dexmani_real.deployment.lifecycle", "dexmani_real.robot.arm_worker",
    "dexmani_real.robot.hand_worker", "dexmani_real.robot.xhand",
    "dexmani_real.sensor.camera_worker", "dexmani_real.sensor.pointcloud_worker",
    "xarm", "xhand_controller", "pyrealsense2",
)
prefixes = ("xarm", "xhand_controller", "pyrealsense2")
loaded = [name for name in sys.modules if name in forbidden or
          any(name.startswith(prefix + ".") for prefix in prefixes)]
if loaded:
    raise SystemExit(",".join(sorted(loaded)))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_check_spawn_child_does_not_inherit_parent_hardware_modules(self) -> None:
        """Exercise the real spawn child while its parent has a hardware import."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = _write_experiment(root / "experiment")
            child_site = root / "child_site"
            child_site.mkdir()
            (child_site / "sitecustomize.py").write_text(
                """
import os
import sys
from types import ModuleType

import numpy as np
from dexmani_real.deployment.contracts import PolicyPrediction
import dexmani_real.deployment.preflight as preflight


class _FakePolicyRuntime:
    def warmup(self, *, samples):
        return (0.001,) * samples

    def predict(self, _observation):
        return PolicyPrediction(
            arm_qpos=np.zeros((8, 7), dtype=np.float64),
            hand_qpos=np.zeros((8, 12), dtype=np.float64),
        )

    def close(self):
        pass


def _fake_loader(config):
    artifact = config.artifact
    return (
        _FakePolicyRuntime(),
        artifact.checkpoint_sha256_from_index,
        {
            "origin": "/policy/__init__.py",
            "commit": "c" * 40,
            "dirty": "false",
            "source_tree_sha256": "d" * 64,
            "version": "0.1.0",
        },
    )


preflight._load_verified_policy_runtime = _fake_loader
if os.environ.get("DEXMANI_POLICY_CHECK_TEST_INJECT_HARDWARE") == "1":
    sys.modules["xhand_controller"] = ModuleType("xhand_controller")
""".lstrip(),
                encoding="utf-8",
            )
            from dexmani_real.config.runtime import resolve_runtime_config
            from dexmani_real.deployment.artifact import resolve_policy_artifact
            from dexmani_real.deployment.config import resolve_policy_runtime_config

            projection = resolve_policy_runtime_config(
                artifact=resolve_policy_artifact(experiment),
                runtime_config=resolve_runtime_config(),
                device="cpu",
                inference_seed=1066,
                execution_mode="shadow",
                hand_acknowledged=False,
            )
            candidate_root = Path(__file__).resolve().parents[1]
            python_path = os.pathsep.join(
                value
                for value in (
                    str(child_site),
                    str(candidate_root),
                    os.environ.get("PYTHONPATH"),
                )
                if value
            )
            parent_hardware_module = ModuleType("xhand_controller")
            with (
                patch.dict(os.environ, {"PYTHONPATH": python_path}),
                patch.dict(
                    sys.modules,
                    {"xhand_controller": parent_hardware_module},
                ),
            ):
                result = preflight.run_isolated_policy_check(
                    projection.runtime,
                    benchmark_samples=1,
                    timeout_s=15.0,
                )
                with (
                    patch.dict(
                        os.environ,
                        {"DEXMANI_POLICY_CHECK_TEST_INJECT_HARDWARE": "1"},
                    ),
                    self.assertRaisesRegex(RuntimeError, "xhand_controller"),
                ):
                    preflight.run_isolated_policy_check(
                        projection.runtime,
                        benchmark_samples=1,
                        timeout_s=15.0,
                    )

        self.assertEqual(result.benchmark_samples, 1)
        self.assertTrue(result.checkpoint_sha256_verified)
        self.assertEqual(result.device, "cpu")

    def test_shadow_propagates_projection_seed_bound_and_calls_lifecycle_once(self):
        artifact = SimpleNamespace(
            allocation_contract=SimpleNamespace(requires_hand=True)
        )
        runtime = object()
        projection = SimpleNamespace(runtime=object())
        lifecycle = ModuleType("dexmani_real.deployment.lifecycle")
        lifecycle.run_policy_deployment = Mock(return_value=0)
        identity = ModuleType("dexmani_real.deployment.run_identity")
        real_source = object()
        identity.resolve_real_source_identity = Mock(return_value=real_source)
        stderr = io.StringIO()
        with (
            patch.object(
                cli,
                "_resolve_artifact_projection",
                return_value=(artifact, runtime, projection),
            ) as resolve,
            patch.dict(
                sys.modules,
                {
                    "dexmani_real.deployment.lifecycle": lifecycle,
                    "dexmani_real.deployment.run_identity": identity,
                },
            ),
            redirect_stderr(stderr),
        ):
            code = cli.main(
                [
                    "shadow",
                    "experiment",
                    "--device",
                    "cpu",
                    "--seed",
                    "1066",
                    "--hand",
                    "--max-running-seconds",
                    "10",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(resolve.call_args.kwargs["execution_mode"], "shadow")
        self.assertEqual(resolve.call_args.kwargs["seed"], 1066)
        self.assertTrue(resolve.call_args.kwargs["hand_acknowledged"])
        lifecycle.run_policy_deployment.assert_called_once_with(
            runtime,
            projection.runtime,
            max_running_s=10.0,
            real_source=real_source,
        )
        self.assertIn("hand startup may reset/home", stderr.getvalue())

    def test_profiles_are_strict_and_paths_are_profile_relative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = _profile_payload()
            profile_path = _write_profile(root, payload)
            profile = load_physical_run_profile(profile_path)
            self.assertEqual(profile.experiment_dir, root / "experiment")
            self.assertEqual(profile.runtime_config, root / "configs" / "runtime.yaml")
            self.assertIsNone(profile.deployment_config)
            self.assertEqual(profile.seed, 1066)

            for key in ("seed",):
                invalid = dict(payload)
                invalid.pop(key)
                _write_profile(root, invalid)
                with self.assertRaisesRegex(ValueError, "missing"):
                    load_physical_run_profile(profile_path)
            invalid = dict(payload)
            invalid["horizon"] = 16
            _write_profile(root, invalid)
            with self.assertRaisesRegex(ValueError, "unknown=.*horizon"):
                load_physical_run_profile(profile_path)

    def test_task_run_rejects_a_schema_v1_profile_before_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile_path = _write_profile(
                Path(directory), _profile_payload(endpoints=4)
            )
            with (
                patch.object(cli, "_resolve_artifact_projection") as resolve,
                redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(cli.main(["run", str(profile_path)]), 1)
            resolve.assert_not_called()
            self.assertIn("schema-v2 profile", stderr.getvalue())

    def test_h4_and_run_build_distinct_exact_bounds(self) -> None:
        for command, endpoints, bounds_type, execution_mode in (
            ("h4", 1, H4ExecuteBounds, "execute"),
            ("run", 4, TaskExecuteBounds, "task"),
        ):
            with (
                self.subTest(command=command),
                tempfile.TemporaryDirectory() as directory,
            ):
                profile_path = _write_profile(
                    Path(directory),
                    (
                        _profile_payload(endpoints=endpoints)
                        if command == "h4"
                        else {
                            **_profile_payload(endpoints=endpoints),
                            "schema_version": 2,
                            "task_scene_card": "scene_card.json",
                        }
                    ),
                )
                if command == "run":
                    _write_task_scene_card(Path(directory))
                artifact = SimpleNamespace(
                    checkpoint_sha256_from_index="a" * 64,
                    allocation_contract=SimpleNamespace(task_name="pick_place_toy"),
                )
                runtime = object()
                projection = SimpleNamespace(runtime=object(), sha256="d" * 64)
                lifecycle = ModuleType("dexmani_real.deployment.lifecycle")
                lifecycle.run_policy_deployment = Mock(return_value=0)
                identity = ModuleType("dexmani_real.deployment.run_identity")
                real_source = SimpleNamespace(availability="available", dirty="false")
                identity.resolve_real_source_identity = Mock(return_value=real_source)
                with (
                    patch.object(
                        cli,
                        "_resolve_artifact_projection",
                        return_value=(artifact, runtime, projection),
                    ) as resolve,
                    patch.dict(
                        sys.modules,
                        {
                            "dexmani_real.deployment.lifecycle": lifecycle,
                            "dexmani_real.deployment.run_identity": identity,
                        },
                    ),
                ):
                    code = cli.main([command, str(profile_path)])
                self.assertEqual(code, 0)
                kwargs = resolve.call_args.kwargs
                self.assertEqual(kwargs["seed"], 1066)
                self.assertEqual(kwargs["execution_mode"], execution_mode)
                bounds = kwargs[
                    "h4_execute_bounds" if command == "h4" else "task_execute_bounds"
                ]
                self.assertIsInstance(bounds, bounds_type)
                self.assertEqual(bounds.max_published_endpoints, endpoints)
                lifecycle.run_policy_deployment.assert_called_once()

    def test_physical_endpoint_and_checkpoint_contracts_fail_closed(self) -> None:
        cases = (("h4", 2, "exactly 1"), ("run", 1, "greater than 1"))
        for command, endpoints, message in cases:
            with (
                self.subTest(command=command),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = _write_profile(
                    Path(directory), _profile_payload(endpoints=endpoints)
                )
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertEqual(cli.main([command, str(path)]), 1)
                self.assertIn(message, stderr.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            path = _write_profile(Path(directory), _profile_payload())
            artifact = SimpleNamespace(checkpoint_sha256_from_index="b" * 64)
            with (
                patch.object(
                    cli,
                    "_resolve_artifact_projection",
                    return_value=(artifact, object(), SimpleNamespace()),
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(cli.main(["h4", str(path)]), 1)

    def test_runtime_failure_is_concise_and_legacy_flat_flags_are_rejected(self):
        stderr = io.StringIO()
        with (
            patch.object(
                cli, "_resolve_artifact_projection", side_effect=ValueError("bad hash")
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(cli.main(["inspect", "experiment"]), 1)
        self.assertEqual(stderr.getvalue().strip(), "error: bad hash")
        self.assertNotIn("usage:", stderr.getvalue())
        with self.assertRaises(SystemExit):
            cli.main(["--experiment-dir", "experiment", "--print-config"])


class PolicyCheckTest(unittest.TestCase):
    def _runtime_config(self) -> SimpleNamespace:
        allocation = SimpleNamespace(
            n_action_steps=3,
            action_dim=19,
            action_key="action",
            control_dt_s=0.1,
        )
        artifact = SimpleNamespace(allocation_contract=allocation)
        deployment = SimpleNamespace(
            device="cpu", inference_seed=1066, command_lead_s=0.01
        )
        return SimpleNamespace(artifact=artifact, deployment=deployment)

    def test_check_import_sentinel_rejects_xhand_native_sdk_modules(self) -> None:
        for module_name in ("xhand_controller", "xhand_controller.fake"):
            with (
                self.subTest(module_name=module_name),
                patch.dict(sys.modules, {module_name: ModuleType(module_name)}),
                self.assertRaisesRegex(RuntimeError, module_name.replace(".", r"\.")),
            ):
                preflight._require_hardware_free_check_imports()

    def test_tiny_cpu_check_loads_once_predicts_three_times_and_closes(self) -> None:
        runtime_config = self._runtime_config()
        prediction = PolicyPrediction(
            arm_qpos=np.zeros((3, 7), dtype=np.float64),
            hand_qpos=np.zeros((3, 12), dtype=np.float64),
        )
        runtime = SimpleNamespace(
            warmup=Mock(return_value=(0.01,) * 5),
            predict=Mock(return_value=prediction),
            close=Mock(),
        )
        provenance = {
            "origin": "/policy/__init__.py",
            "commit": "c" * 40,
            "dirty": "false",
            "source_tree_sha256": "d" * 64,
            "version": "0.1.0",
        }
        clock = iter((0, 10_000_000, 20_000_000, 40_000_000, 50_000_000, 80_000_000))
        with (
            patch.object(
                preflight,
                "_load_verified_policy_runtime",
                return_value=(runtime, "a" * 64, provenance),
            ) as load,
            patch.object(preflight, "_synthetic_observation", return_value=object()),
            patch.object(
                preflight.time, "perf_counter_ns", side_effect=lambda: next(clock)
            ),
            patch.object(
                preflight, "_require_hardware_free_check_imports"
            ) as check_imports,
        ):
            result = preflight._run_policy_check_child(runtime_config, 3)
        load.assert_called_once_with(runtime_config)
        runtime.warmup.assert_called_once_with(samples=5)
        self.assertEqual(runtime.predict.call_count, 3)
        runtime.close.assert_called_once_with()
        check_imports.assert_called_once_with()
        self.assertEqual(result.benchmark_samples, 3)
        self.assertEqual(result.latency_p50_ms, 20.0)
        self.assertEqual(result.latency_p95_ms, 30.0)
        self.assertGreaterEqual(result.remaining_targets_min, 0)
        self.assertEqual(result.source_aware_schedulability, "NOT_MEASURED")
        self.assertIsNone(result.gpu_peak_memory_bytes)

    def test_check_rejects_invalid_sample_counts_and_prediction(self) -> None:
        for value in (0, True, 1001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preflight._validate_benchmark_samples(value)
        runtime = SimpleNamespace(
            warmup=Mock(return_value=(0.01,) * 5),
            predict=Mock(return_value=object()),
            close=Mock(),
        )
        provenance = {
            "origin": "/policy/__init__.py",
            "commit": "c" * 40,
            "dirty": "false",
            "source_tree_sha256": "d" * 64,
            "version": "0.1.0",
        }
        with (
            patch.object(
                preflight,
                "_load_verified_policy_runtime",
                return_value=(runtime, "a" * 64, provenance),
            ),
            patch.object(preflight, "_synthetic_observation", return_value=object()),
            self.assertRaises(TypeError),
        ):
            preflight._run_policy_check_child(self._runtime_config(), 1)
        runtime.close.assert_called_once_with()


class DeploymentLoggingTest(unittest.TestCase):
    def test_handler_levels_pid_filename_no_duplicates_and_metrics_debug(self):
        with tempfile.TemporaryDirectory() as directory:
            script = r"""
import json
import logging
import os
from unittest.mock import Mock
from dexmani_real.utils.log import get_logger
logger = get_logger("r3.logging.test")
same = get_logger("r3.logging.test")
from dexmani_real.deployment import metrics as metrics_module
mock_logger = Mock()
metrics_module.logger = mock_logger
metrics = metrics_module.Metrics()
metrics.increment("count")
metrics.flush()
file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)
                    and not isinstance(h, logging.FileHandler)]
print(json.dumps({
    "logger_level": logger.level,
    "same": logger is same,
    "handler_count": len(logger.handlers),
    "console_levels": [h.level for h in console_handlers],
    "file_levels": [h.level for h in file_handlers],
    "filenames": [os.path.basename(h.baseFilename) for h in file_handlers],
    "pid": os.getpid(),
    "debug_called": mock_logger.debug.call_count,
    "info_called": mock_logger.info.call_count,
}))
"""
            env = os.environ.copy()
            env["DEXMANI_LOG_DIR"] = directory
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["logger_level"], logging.DEBUG)
            self.assertTrue(value["same"])
            self.assertEqual(value["handler_count"], 2)
            self.assertEqual(value["console_levels"], [logging.INFO])
            self.assertEqual(value["file_levels"], [logging.DEBUG])
            self.assertRegex(
                value["filenames"][0],
                rf"^dexmani_\d{{8}}_\d{{6}}_{value['pid']}\.log$",
            )
            self.assertEqual(value["debug_called"], 1)
            self.assertEqual(value["info_called"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
