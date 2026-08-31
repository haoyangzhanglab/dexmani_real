"""Focused offline checks for the Batch-2 experiment preflight boundary."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import dexmani_real.deployment.preflight as preflight_module
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.control.arm_home import ArmHomeStatus, execute_arm_home
from dexmani_real.deployment.artifact import resolve_policy_artifact
from dexmani_real.deployment.config import (
    DeploymentConfig,
    H4ExecuteBounds,
    PolicyRuntimeConfig,
    TaskExecuteBounds,
    resolve_policy_runtime_config,
)
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.lifecycle import (
    _physical_execute_provenance_json,
    build_policy_worker_specs,
    run_policy_deployment,
)
from dexmani_real.deployment.manifest import DeploymentManifest
from dexmani_real.deployment.operator import (
    _request_immediate_stop,
    run_operator_control,
)
from dexmani_real.deployment.preflight import (
    PreflightResult,
    _decode_child_message,
    _encode_child_message,
    _preflight_child,
    _run_preflight_child,
    _terminate_join_kill_close,
    _validate_prediction,
    run_isolated_preflight,
)
from dexmani_real.deployment.run_identity import RealSourceIdentity
from dexmani_real.deployment.worker import (
    _load_inference_runtime,
    inference_loop,
    stamp_prediction_timing,
)
from dexmani_real.integrations.dexmani_policy import (
    DexManiPolicyRuntime,
    _canonical_json_sha256,
    _validate_embedded_targets,
    _validate_inference_device,
    precheck_policy_package_provenance,
    verify_imported_policy_provenance,
)
from dexmani_real.runtime.safety import SafetyState, StopRequest
from dexmani_real.teleop.keyboard import ControlSignal


def _sidecar(checkpoint: Path, *, checkpoint_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
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


def _write_experiment(root: Path, *, matching_hash: bool = False) -> Path:
    root.mkdir()
    (root / "config.yaml").write_text("name: synthetic\n", encoding="utf-8")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "epoch=0001-deployment-v1.pt"
    checkpoint.write_bytes(b"not-a-torch-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    sidecar = _sidecar(
        checkpoint, checkpoint_sha256=digest if matching_hash else "a" * 64
    )
    sidecar_path = checkpoint.with_name(f"{checkpoint.name}.deployment.json")
    sidecar_path.write_text(
        json.dumps(sidecar, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    (checkpoints / "deployment_latest.pt").symlink_to(checkpoint.name)
    return root


class DeploymentPreflightTest(unittest.TestCase):
    def _projection(self, root: Path):
        return resolve_policy_runtime_config(
            artifact=resolve_policy_artifact(root),
            runtime_config=resolve_runtime_config(),
        )

    def test_cuda_device_never_silently_falls_back_to_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA, but CUDA is unavailable"):
                _validate_inference_device("cuda:0")
        self.assertEqual(str(_validate_inference_device("cpu")), "cpu")

    def test_projection_is_pickle_safe_and_artifact_owned_values_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            projection = self._projection(root)
            restored = pickle.loads(pickle.dumps(projection.runtime))
            self.assertEqual(
                restored.artifact.checkpoint_path,
                projection.runtime.artifact.checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "artifact-owned"):
                resolve_policy_runtime_config(
                    artifact=resolve_policy_artifact(root),
                    runtime_config=resolve_runtime_config(),
                    data={"action_key": "action_ee"},
                )
            permitted = resolve_policy_runtime_config(
                artifact=resolve_policy_artifact(root),
                runtime_config=resolve_runtime_config(),
                data={"inference_hz": 9.0},
            )
            self.assertEqual(permitted.runtime.deployment.inference_hz, 9.0)
            with self.assertRaisesRegex(TypeError, "unknown"):
                resolve_policy_runtime_config(
                    artifact=resolve_policy_artifact(root),
                    runtime_config=resolve_runtime_config(),
                    data={"device": "cpu"},
                )

    def test_worker_specs_preserve_the_frozen_shadow_execution_mode(self):
        runtime = resolve_runtime_config()
        policy_runtime = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                observation_fields="arm_qpos",
            ),
            control_dt_s=1.0 / float(runtime.policy.control_hz),
            execution_mode="shadow",
        )

        specs = build_policy_worker_specs(object(), runtime, policy_runtime)

        inference = next(spec for spec in specs if spec.name == "inference")
        coordinator = next(spec for spec in specs if spec.name == "policy")
        self.assertIs(inference.args[1], policy_runtime)
        self.assertEqual(coordinator.args[1].execution_mode, "shadow")

    def test_inference_runtime_requires_a_resolved_artifact(self):
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(observation_fields="arm_qpos"),
            control_dt_s=0.0625,
        )

        with self.assertRaisesRegex(ValueError, "resolved artifact"):
            _load_inference_runtime(config)

    def test_artifact_bound_inference_uses_verified_stream_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment", matching_hash=True)
            config = self._projection(root).runtime
            runtime = SimpleNamespace(
                warmup=Mock(return_value=(0.1,) * 5),
                close=Mock(),
            )
            shared = SimpleNamespace(
                is_running=SimpleNamespace(value=0),
                action_control_hz=16.0,
                set_heartbeat=Mock(),
                set_ready=Mock(),
            )
            with patch(
                "dexmani_real.deployment.preflight.load_verified_policy_runtime",
                return_value=runtime,
            ) as verified_loader:
                inference_loop(shared, config)

        verified_loader.assert_called_once_with(config)
        runtime.warmup.assert_called_once_with(samples=5)
        runtime.close.assert_called_once_with()
        shared.set_ready.assert_called_once_with("inference")

    def test_artifact_warmup_latency_failure_prevents_inference_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment", matching_hash=True)
            config = self._projection(root).runtime
            runtime = SimpleNamespace(
                warmup=Mock(return_value=(0.1, 0.1, 0.9, 0.9, 0.9)),
                close=Mock(),
            )
            shared = SimpleNamespace(
                is_running=SimpleNamespace(value=0),
                action_control_hz=16.0,
                set_heartbeat=Mock(),
                set_ready=Mock(),
            )
            with patch(
                "dexmani_real.deployment.preflight.load_verified_policy_runtime",
                return_value=runtime,
            ):
                with self.assertRaisesRegex(RuntimeError, "viable action window"):
                    inference_loop(shared, config)

        shared.set_ready.assert_not_called()
        runtime.close.assert_called_once_with()

    def test_preflight_does_not_replace_checkpoint_bound_rng_streams(self):
        prediction = object()
        runtime = SimpleNamespace(predict=Mock(return_value=prediction), close=Mock())
        runtime_config = SimpleNamespace(
            artifact=SimpleNamespace(
                allocation_contract=SimpleNamespace(
                    required_action_steps=15,
                    action_dim=19,
                )
            ),
            inference_seed=1066,
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
                preflight_module,
                "_load_verified_policy_runtime",
                return_value=(runtime, "a" * 64, provenance),
            ),
            patch.object(
                preflight_module,
                "_synthetic_observation",
                return_value=object(),
            ),
            patch.object(preflight_module, "_validate_prediction") as validate,
            patch.object(torch, "manual_seed") as manual_seed,
            patch.object(np.random, "seed") as numpy_seed,
        ):
            result = _run_preflight_child(runtime_config)

        manual_seed.assert_not_called()
        numpy_seed.assert_not_called()
        runtime.predict.assert_called_once()
        validate.assert_called_once_with(prediction, runtime_config)
        runtime.close.assert_called_once_with()
        self.assertEqual(result.action_steps, 15)
        self.assertEqual(result.package_commit, "c" * 40)

    def test_physical_receipt_provenance_is_structured_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            runtime = resolve_runtime_config()
            projection = resolve_policy_runtime_config(
                artifact=resolve_policy_artifact(root),
                runtime_config=runtime,
                inference_seed=1066,
                execution_mode="task",
                hand_acknowledged=True,
                task_execute_bounds=TaskExecuteBounds(
                    max_published_endpoints=331,
                    acknowledgement_timeout_s=2.0,
                    max_running_s=25.0,
                ),
            )
            provenance = json.loads(
                _physical_execute_provenance_json(
                    projection.runtime,
                    runtime_sha256=runtime.sha256,
                    real_source=RealSourceIdentity(
                        availability="available",
                        commit="e" * 40,
                        dirty="false",
                        python_tree_sha256="f" * 64,
                    ),
                    invocation_argv=("run_policy.py", "--execution-mode", "task"),
                    projection_sha256=projection.sha256,
                )
            )

        self.assertEqual(provenance["policy_projection"]["sha256"], projection.sha256)
        self.assertEqual(provenance["policy_projection"]["device"], "cuda:0")
        self.assertEqual(provenance["policy_projection"]["inference_seed"], 1066)
        self.assertEqual(
            provenance["policy_projection"]["bounds"]["max_published_endpoints"],
            331,
        )
        self.assertEqual(
            provenance["policy_package_contract"]["commit"],
            projection.runtime.artifact.producer.commit,
        )

    def test_hand_shadow_requires_acknowledgement_before_runtime_channels(self):
        runtime = resolve_runtime_config()
        policy_runtime = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                observation_fields="arm_qpos",
                hand_enabled=True,
            ),
            control_dt_s=1.0 / float(runtime.policy.control_hz),
            execution_mode="shadow",
            hand_acknowledged=False,
        )

        with patch(
            "dexmani_real.deployment.lifecycle.RuntimeChannels.create"
        ) as create_channels:
            with self.assertRaisesRegex(ValueError, "requires --hand"):
                run_policy_deployment(runtime, policy_runtime)

        create_channels.assert_not_called()

    def test_h4_execute_requires_hand_and_explicit_operator_acknowledgement(self):
        runtime = resolve_runtime_config()
        bounds = H4ExecuteBounds(
            max_published_endpoints=1,
            acknowledgement_timeout_s=2.0,
            max_running_s=30.0,
        )
        with self.assertRaisesRegex(ValueError, "hand-enabled"):
            PolicyRuntimeConfig(
                deployment=DeploymentConfig(
                    observation_fields="arm_qpos",
                ),
                control_dt_s=1.0 / float(runtime.policy.control_hz),
                execution_mode="execute",
                hand_acknowledged=True,
                h4_execute_bounds=bounds,
            )
        with self.assertRaisesRegex(ValueError, "explicit --hand"):
            PolicyRuntimeConfig(
                deployment=DeploymentConfig(
                    observation_fields="arm_qpos",
                    hand_enabled=True,
                ),
                control_dt_s=1.0 / float(runtime.policy.control_hz),
                execution_mode="execute",
                hand_acknowledged=False,
                h4_execute_bounds=bounds,
            )

    def test_task_execute_is_a_separate_multi_endpoint_profile(self):
        runtime = resolve_runtime_config()
        bounds = TaskExecuteBounds(
            max_published_endpoints=320,
            acknowledgement_timeout_s=2.0,
            max_running_s=25.0,
        )
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                observation_fields="arm_qpos",
                hand_enabled=True,
            ),
            control_dt_s=1.0 / float(runtime.policy.control_hz),
            execution_mode="task",
            hand_acknowledged=True,
            task_execute_bounds=bounds,
        )
        self.assertIs(config.physical_execute_bounds, bounds)
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            TaskExecuteBounds(
                max_published_endpoints=1,
                acknowledgement_timeout_s=2.0,
                max_running_s=25.0,
            )

    def test_run_limit_is_validated_before_runtime_channels(self):
        runtime = resolve_runtime_config()
        policy_runtime = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                observation_fields="arm_qpos",
            ),
            control_dt_s=1.0 / float(runtime.policy.control_hz),
            execution_mode="shadow",
            hand_acknowledged=False,
        )

        with patch(
            "dexmani_real.deployment.lifecycle.RuntimeChannels.create"
        ) as create_channels:
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                run_policy_deployment(runtime, policy_runtime, max_running_s=0.0)

        create_channels.assert_not_called()

    def test_shadow_operator_home_key_never_writes_or_requests_motion(self):
        coupled_ring = SimpleNamespace(write=Mock())
        shared = SimpleNamespace(
            is_running=SimpleNamespace(value=1),
            start_request=SimpleNamespace(value=False),
            stop_request=SimpleNamespace(value=False),
            quit_requested=SimpleNamespace(value=False),
            error_state=SimpleNamespace(value=False),
            estop_request=SimpleNamespace(value=False),
            physical_home_completed=SimpleNamespace(value=False),
            coupled_cmd_ring=coupled_ring,
        )

        class _Keyboard:
            estop_latched = False
            healthy = True

            @staticmethod
            def start():
                return None

            @staticmethod
            def stop():
                return None

            @staticmethod
            def poll(*, timeout):
                del timeout
                shared.is_running.value = 0
                return (ControlSignal.HOME,)

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=_Keyboard(),
            ),
            patch("dexmani_real.deployment.operator._home") as home,
        ):
            run_operator_control(
                shared,
                object(),
                object(),
                None,
                stop_event=threading.Event(),
                execution_mode="shadow",
            )

        home.assert_not_called()
        coupled_ring.write.assert_not_called()
        self.assertFalse(shared.start_request.value)
        self.assertFalse(shared.stop_request.value)

    def test_immediate_stop_fences_an_armed_or_running_runtime(self):
        shared = SimpleNamespace(
            stop_request=SimpleNamespace(value=False),
            safety_state=SimpleNamespace(value=int(SafetyState.RUNNING)),
            error_state=SimpleNamespace(value=False),
        )
        with patch(
            "dexmani_real.deployment.operator.revoke_motion", return_value=True
        ) as revoke:
            _request_immediate_stop(shared)

        self.assertEqual(shared.stop_request.value, int(StopRequest.OPERATOR))
        revoke.assert_called_once_with(shared, SafetyState.ARMED)

    def test_immediate_stop_never_arms_a_disarmed_runtime(self):
        shared = SimpleNamespace(
            stop_request=SimpleNamespace(value=False),
            safety_state=SimpleNamespace(value=int(SafetyState.DISARMED)),
            error_state=SimpleNamespace(value=False),
        )
        with patch("dexmani_real.deployment.operator.revoke_motion") as revoke:
            _request_immediate_stop(shared)

        self.assertEqual(shared.stop_request.value, int(StopRequest.OPERATOR))
        revoke.assert_not_called()

    def test_arm_home_honors_operator_cancellation_before_motion_boundary(self):
        shared = SimpleNamespace(estop_request=SimpleNamespace(value=False))
        result = execute_arm_home(
            shared,
            np.zeros(7),
            planner=None,
            config=object(),
            estop_requested=lambda: False,
            cancel_requested=lambda: True,
        )

        self.assertEqual(result.status, ArmHomeStatus.CANCELLED)

    def test_physical_operator_drains_home_events_accumulated_while_homing(self):
        shared = SimpleNamespace(
            is_running=SimpleNamespace(value=1),
            start_request=SimpleNamespace(value=False),
            stop_request=SimpleNamespace(value=False),
            quit_requested=SimpleNamespace(value=False),
            error_state=SimpleNamespace(value=False),
            estop_request=SimpleNamespace(value=False),
            physical_home_completed=SimpleNamespace(value=False),
        )

        class _Keyboard:
            estop_latched = False
            healthy = True

            def __init__(self):
                self.signals = [ControlSignal.HOME]
                self.drained = 0
                self.later_home_pending = True

            @staticmethod
            def start():
                return None

            @staticmethod
            def stop():
                return None

            def poll(self, *, timeout):
                del timeout
                if self.signals:
                    signals = tuple(self.signals)
                    self.signals.clear()
                    return signals
                if self.later_home_pending:
                    self.later_home_pending = False
                    return (ControlSignal.HOME,)
                shared.is_running.value = 0
                return ()

            def drain_signal(self, target):
                retained = [signal for signal in self.signals if signal is not target]
                removed = len(self.signals) - len(retained)
                self.signals = retained
                self.drained += removed
                return removed

        keyboard = _Keyboard()

        def home(*_args, **_kwargs):
            keyboard.signals.extend([ControlSignal.HOME] * 7)
            return True

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch(
                "dexmani_real.deployment.operator._home", side_effect=home
            ) as run_home,
        ):
            run_operator_control(
                shared,
                object(),
                object(),
                object(),
                stop_event=threading.Event(),
                execution_mode="execute",
            )

        run_home.assert_called_once()
        self.assertEqual(keyboard.drained, 7)
        self.assertTrue(shared.physical_home_completed.value)

    def test_physical_operator_failed_home_keeps_begin_gate_closed(self):
        shared = SimpleNamespace(
            is_running=SimpleNamespace(value=1),
            start_request=SimpleNamespace(value=False),
            stop_request=SimpleNamespace(value=False),
            quit_requested=SimpleNamespace(value=False),
            error_state=SimpleNamespace(value=False),
            estop_request=SimpleNamespace(value=False),
            physical_home_completed=SimpleNamespace(value=False),
        )

        class _Keyboard:
            estop_latched = False
            healthy = True

            @staticmethod
            def start():
                return None

            @staticmethod
            def stop():
                return None

            @staticmethod
            def poll(*, timeout):
                del timeout
                shared.is_running.value = 0
                return (ControlSignal.HOME, ControlSignal.BEGIN)

            @staticmethod
            def drain_signal(_target):
                return 0

        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=_Keyboard(),
            ),
            patch("dexmani_real.deployment.operator._home", return_value=False),
        ):
            run_operator_control(
                shared,
                object(),
                object(),
                object(),
                stop_event=threading.Event(),
                execution_mode="execute",
            )

        self.assertFalse(shared.physical_home_completed.value)
        self.assertFalse(shared.start_request.value)

    def test_physical_operator_requires_a_fresh_begin_after_home(self):
        shared = SimpleNamespace(
            is_running=SimpleNamespace(value=1),
            start_request=SimpleNamespace(value=False),
            stop_request=SimpleNamespace(value=False),
            quit_requested=SimpleNamespace(value=False),
            error_state=SimpleNamespace(value=False),
            estop_request=SimpleNamespace(value=False),
            physical_home_completed=SimpleNamespace(value=False),
        )

        class _Keyboard:
            estop_latched = False
            healthy = True

            def __init__(self) -> None:
                self.polls = 0
                self.drained: list[ControlSignal] = []

            @staticmethod
            def start():
                return None

            @staticmethod
            def stop():
                return None

            def poll(self, *, timeout):
                del timeout
                self.polls += 1
                if self.polls == 1:
                    # Both orderings are stale: B before H and B queued while
                    # H is blocking must never authorize the later run.
                    return (
                        ControlSignal.BEGIN,
                        ControlSignal.HOME,
                        ControlSignal.BEGIN,
                    )
                shared.is_running.value = 0
                return ()

            def drain_signal(self, target):
                self.drained.append(target)
                return 0

        keyboard = _Keyboard()
        with (
            patch(
                "dexmani_real.deployment.operator.KeyboardHandler",
                return_value=keyboard,
            ),
            patch("dexmani_real.deployment.operator._home", return_value=True),
        ):
            run_operator_control(
                shared,
                object(),
                object(),
                object(),
                stop_event=threading.Event(),
                execution_mode="execute",
            )

        self.assertTrue(shared.physical_home_completed.value)
        self.assertFalse(shared.start_request.value)
        self.assertEqual(
            keyboard.drained,
            [ControlSignal.HOME, ControlSignal.BEGIN],
        )

    def test_table_plane_projection_is_null_when_disabled_and_exact_when_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            artifact = resolve_policy_artifact(root)
            disabled_runtime = resolve_runtime_config(
                data={"environment": {"table": {"enabled": False}}}
            )
            disabled = resolve_policy_runtime_config(
                artifact=artifact, runtime_config=disabled_runtime
            )
            self.assertEqual(disabled.runtime.point_cloud_table_plane_abcd_json, "null")
            enabled_runtime = resolve_runtime_config()
            enabled = resolve_policy_runtime_config(
                artifact=resolve_policy_artifact(root), runtime_config=enabled_runtime
            )
            self.assertEqual(
                json.loads(enabled.runtime.point_cloud_table_plane_abcd_json),
                list(enabled_runtime.environment.table.plane_abcd),
            )

    def test_hash_mismatch_prevents_policy_import_or_checkpoint_deserialize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            projection = self._projection(root)
            with patch.dict(sys.modules, {"dexmani_policy": None}):
                with patch.object(torch, "load", side_effect=AssertionError):
                    with self.assertRaisesRegex(ValueError, "SHA-256"):
                        _run_preflight_child(projection.runtime)

    def test_parent_propagates_child_hash_failure_without_a_hardware_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            projection = self._projection(root)
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                run_isolated_preflight(projection.runtime, timeout_s=10.0)

    def test_preflight_rechecks_checkpoint_and_directory_identity_after_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment", matching_hash=True)
            projection = self._projection(root)
            original_hash = preflight_module._sha256_stream

            def mutate_checkpoint(stream):
                result = original_hash(stream)
                projection.runtime.artifact.checkpoint_path.write_bytes(b"changed")
                return result

            with patch.object(
                preflight_module, "_sha256_stream", side_effect=mutate_checkpoint
            ):
                with self.assertRaisesRegex(ValueError, "checkpoint identity changed"):
                    _run_preflight_child(projection.runtime)

        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment", matching_hash=True)
            projection = self._projection(root)
            original_hash = preflight_module._sha256_stream

            def mutate_directory(stream):
                result = original_hash(stream)
                marker = projection.runtime.artifact.checkpoint_path.parent / "race"
                marker.write_text("changed", encoding="utf-8")
                return result

            with patch.object(
                preflight_module, "_sha256_stream", side_effect=mutate_directory
            ):
                with self.assertRaisesRegex(
                    ValueError, "checkpoints directory identity"
                ):
                    _run_preflight_child(projection.runtime)

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires Linux procfs")
    def test_rejected_preflight_closes_its_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            projection = self._projection(root)
            before = len(os.listdir("/proc/self/fd"))
            for _ in range(20):
                with self.assertRaises(ValueError):
                    _run_preflight_child(projection.runtime)
            self.assertEqual(before, len(os.listdir("/proc/self/fd")))

    def test_adapter_receipt_cross_checks_producer_allocation_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            checkpoints = root / "checkpoints"
            producer = {
                "repository": "haoyangzhanglab/dexmani_policy",
                "commit": "c" * 40,
                "metadata_provenance": "retrofitted",
                "retrofitted_train_params_fields": [],
            }
            sidecar_path = next(checkpoints.glob("*.deployment.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            projection = self._projection(root)
            allocation = projection.runtime.artifact.allocation_contract
            inference = {
                "task_name": allocation.task_name,
                "action_key": allocation.action_key,
                "action_dim": allocation.action_dim,
                "n_obs_steps": allocation.n_obs_steps,
                "n_action_steps": allocation.n_action_steps,
                "horizon": allocation.horizon,
                "use_aux_ee": False,
            }
            train = {
                "action_key": allocation.action_key,
                "action_dim": allocation.action_dim,
                "n_obs_steps": allocation.n_obs_steps,
                "n_action_steps": allocation.n_action_steps,
                "horizon": allocation.horizon,
                "use_aux_ee": False,
            }
            data = {
                "task_name": allocation.task_name,
                "action_key": allocation.action_key,
                "model_action_dim": allocation.action_dim,
                "n_obs_steps": allocation.n_obs_steps,
                "n_action_steps": allocation.n_action_steps,
                "horizon": allocation.horizon,
                "dt": allocation.control_dt_s,
                "sensor_modalities": list(allocation.sensor_modalities),
                "point_cloud_num_points": allocation.point_cloud_num_points,
                "point_cloud_feature_dim": allocation.point_cloud_feature_dim,
                "pad_before": 1,
                "pad_after": 7,
                "padding_semantics": "repeat_edge",
                "use_aux_ee": False,
            }
            contract = {
                "schema_version": 1,
                "inference_config": inference,
                "data_contract": data,
                "train_params": train,
                "producer": producer,
                "retrofitted_train_params_fields": [],
            }
            sidecar["embedded_contract_sha256"] = _canonical_json_sha256(contract)
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            projection = self._projection(root)
            checkpoint = SimpleNamespace(
                deployment_contract=contract,
                producer=producer,
                inference_config={
                    "task_name": allocation.task_name,
                    "action_key": allocation.action_key,
                    "action_dim": allocation.action_dim,
                    "n_obs_steps": allocation.n_obs_steps,
                    "n_action_steps": allocation.n_action_steps,
                    "horizon": allocation.horizon,
                    "use_aux_ee": False,
                },
                train_params={
                    "action_key": allocation.action_key,
                    "action_dim": allocation.action_dim,
                    "n_obs_steps": allocation.n_obs_steps,
                    "n_action_steps": allocation.n_action_steps,
                    "horizon": allocation.horizon,
                    "use_aux_ee": False,
                },
                data_contract={
                    "task_name": allocation.task_name,
                    "action_key": allocation.action_key,
                    "model_action_dim": allocation.action_dim,
                    "n_obs_steps": allocation.n_obs_steps,
                    "n_action_steps": allocation.n_action_steps,
                    "horizon": allocation.horizon,
                    "dt": allocation.control_dt_s,
                    "sensor_modalities": list(allocation.sensor_modalities),
                    "point_cloud_num_points": allocation.point_cloud_num_points,
                    "point_cloud_feature_dim": allocation.point_cloud_feature_dim,
                    "pad_before": 1,
                    "pad_after": 7,
                    "padding_semantics": "repeat_edge",
                    "use_aux_ee": False,
                },
            )
            runtime = DexManiPolicyRuntime(projection.runtime)
            runtime._validate_artifact_receipt(checkpoint)
            checkpoint.deployment_contract["schema_version"] = True
            with self.assertRaisesRegex(ValueError, "unsupported version"):
                runtime._validate_artifact_receipt(checkpoint)
            checkpoint.deployment_contract["schema_version"] = 1
            checkpoint.data_contract["pad_before"] = 0
            checkpoint.deployment_contract["data_contract"]["pad_before"] = 0
            sidecar["embedded_contract_sha256"] = _canonical_json_sha256(contract)
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            runtime = DexManiPolicyRuntime(self._projection(root).runtime)
            with self.assertRaisesRegex(ValueError, "padding"):
                runtime._validate_artifact_receipt(checkpoint)
            checkpoint.data_contract["pad_before"] = 1
            checkpoint.deployment_contract["data_contract"]["pad_before"] = 1
            checkpoint.producer = checkpoint.producer | {"commit": "d" * 40}
            checkpoint.deployment_contract["producer"] = checkpoint.producer
            sidecar["embedded_contract_sha256"] = _canonical_json_sha256(contract)
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            runtime = DexManiPolicyRuntime(self._projection(root).runtime)
            with self.assertRaisesRegex(ValueError, "producer.commit"):
                runtime._validate_artifact_receipt(checkpoint)

    def test_embedded_target_and_experiment_package_origin_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "_target_"):
            _validate_embedded_targets({"agent": {"_target_": "outside.Agent"}})
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            projection = self._projection(root)
            module_path = root / "fake_policy.py"
            module_path.write_text("", encoding="utf-8")
            fake_module = SimpleNamespace(__file__=str(module_path))
            with patch.dict(sys.modules, {"dexmani_policy": fake_module}):
                with self.assertRaisesRegex(ValueError, "must not be imported"):
                    precheck_policy_package_provenance(projection.runtime)

    def test_provenance_gate_never_executes_an_experiment_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment", matching_hash=True)
            projection = self._projection(root)
            package = root / "dexmani_policy"
            package.mkdir()
            marker = root / "marker"
            (package / "__init__.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            previous = {
                name: value
                for name, value in sys.modules.items()
                if name == "dexmani_policy" or name.startswith("dexmani_policy.")
            }
            for name in previous:
                sys.modules.pop(name)
            try:
                with patch.object(sys, "path", [str(root), *sys.path]):
                    with self.assertRaisesRegex(ValueError, "inside experiment_dir"):
                        precheck_policy_package_provenance(projection.runtime)
                self.assertFalse(marker.exists())
            finally:
                for name in list(sys.modules):
                    if name == "dexmani_policy" or name.startswith("dexmani_policy."):
                        sys.modules.pop(name)
                sys.modules.update(previous)

    def test_post_gate_import_must_match_the_prechecked_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = _write_experiment(base / "experiment")
            projection = self._projection(root)
            package_root = base / "installed" / "dexmani_policy"
            package_root.mkdir(parents=True)
            marker = base / "imported-marker"
            (package_root / "__init__.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                encoding="utf-8",
            )
            previous = {
                name: value
                for name, value in sys.modules.items()
                if name == "dexmani_policy" or name.startswith("dexmani_policy.")
            }
            for name in previous:
                sys.modules.pop(name)
            try:
                with patch.object(sys, "path", [str(package_root.parent), *sys.path]):
                    with patch(
                        "dexmani_real.integrations.dexmani_policy._git_package_provenance",
                        return_value=("c" * 40, "false"),
                    ):
                        provenance = precheck_policy_package_provenance(
                            projection.runtime
                        )
                    self.assertFalse(marker.exists())
                    importlib.import_module("dexmani_policy")
                    verify_imported_policy_provenance(provenance)
                self.assertTrue(marker.exists())
            finally:
                for name in list(sys.modules):
                    if name == "dexmani_policy" or name.startswith("dexmani_policy."):
                        sys.modules.pop(name)
                sys.modules.update(previous)

    def test_preflight_message_is_bounded_and_child_errors_are_truncated(self):
        oversized = {"ok": False, "receipt": None, "error": "x" * (1024 * 1024)}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _encode_child_message(oversized)

        class _Connection:
            payload: bytes | None = None

            def send_bytes(self, value: bytes) -> None:
                self.payload = value

            def close(self) -> None:
                pass

        connection = _Connection()
        with patch.object(
            preflight_module,
            "_run_preflight_child",
            side_effect=RuntimeError("x" * (1024 * 1024)),
        ):
            _preflight_child(connection, object())
        assert connection.payload is not None
        message = _decode_child_message(connection.payload)
        self.assertLessEqual(len(message["error"]), 2 * 1024)
        receipt = PreflightResult(
            checkpoint_sha256="a" * 64,
            checkpoint_sha256_verified=True,
            action_steps=15,
            action_dim=19,
            package_origin="/opt/dexmani_policy/__init__.py",
            package_commit="c" * 40,
            package_dirty="false",
            package_source_tree_sha256="d" * 64,
            package_version="0.1.0",
        )
        self.assertEqual(PreflightResult.from_wire(receipt.to_wire()), receipt)

    def test_process_cleanup_kills_a_stubborn_child_before_closing(self):
        class _Process:
            def __init__(self) -> None:
                self.alive = True
                self.terminated = 0
                self.killed = 0
                self.closed = 0

            def is_alive(self) -> bool:
                return self.alive

            def terminate(self) -> None:
                self.terminated += 1

            def kill(self) -> None:
                self.killed += 1
                self.alive = False

            def join(self, timeout: float) -> None:
                assert timeout > 0.0

            def close(self) -> None:
                self.closed += 1

        process = _Process()
        _terminate_join_kill_close(process)
        self.assertEqual(
            (process.terminated, process.killed, process.closed), (1, 1, 1)
        )

    def test_prediction_validation_requires_prediction_shape_hand_and_ee_geometry(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            joint_runtime = self._projection(root).runtime
            with self.assertRaisesRegex(TypeError, "PolicyPrediction"):
                _validate_prediction(object(), joint_runtime)
            steps = joint_runtime.artifact.allocation_contract.required_action_steps
            joint = PolicyPrediction(
                arm_qpos=np.zeros((steps, 7)),
                hand_qpos=np.zeros((steps, 12)),
            )
            _validate_prediction(joint, joint_runtime)
            valid_ee = PolicyPrediction(
                arm_qpos=None,
                hand_qpos=np.zeros((steps, 12)),
                ee_pos=np.zeros((steps, 3)),
                ee_rot6d=np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), (steps, 1)),
            )
            with self.assertRaisesRegex(ValueError, "joint action artifact"):
                _validate_prediction(valid_ee, joint_runtime)

            sidecar_path = next((root / "checkpoints").glob("*.deployment.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["allocation"]["action_key"] = "action_ee"
            sidecar["allocation"]["action_dim"] = 21
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            ee_runtime = self._projection(root).runtime
            self.assertEqual(ee_runtime.deployment.action_key, "action_ee")
            with self.assertRaisesRegex(ValueError, "EE action artifact"):
                _validate_prediction(joint, ee_runtime)
            _validate_prediction(valid_ee, ee_runtime)
            invalid_ee = PolicyPrediction(
                arm_qpos=None,
                hand_qpos=np.zeros((steps, 12)),
                ee_pos=np.zeros((steps, 3)),
                ee_rot6d=np.zeros((steps, 6)),
            )
            with self.assertRaisesRegex(ValueError, "geometry"):
                _validate_prediction(invalid_ee, ee_runtime)

    def test_adapter_uses_complete_future_pred_action_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            runtime_config = resolve_policy_runtime_config(
                artifact=resolve_policy_artifact(root),
                runtime_config=resolve_runtime_config(),
                inference_seed=1066,
            ).runtime
            allocation = runtime_config.artifact.allocation_contract
            runtime = DexManiPolicyRuntime(runtime_config)
            runtime._manifest = DeploymentManifest(
                action_key=allocation.action_key,
                n_obs_steps=allocation.n_obs_steps,
                n_action_steps=allocation.n_action_steps,
                action_dim=allocation.action_dim,
                horizon=allocation.horizon,
                hand_dim=12,
                control_action_dim=19,
                sensor_modalities=allocation.sensor_modalities,
                point_cloud_num_points=allocation.point_cloud_num_points,
                point_cloud_feature_dim=allocation.point_cloud_feature_dim,
            )
            pred_action = torch.arange(
                allocation.horizon * allocation.action_dim, dtype=torch.float32
            ).reshape(1, allocation.horizon, allocation.action_dim)
            control_action = pred_action[
                :,
                allocation.n_obs_steps
                - 1 : allocation.n_obs_steps
                - 1
                + allocation.n_action_steps,
                :19,
            ]
            tail = pred_action[
                :, allocation.n_obs_steps - 1 + allocation.n_action_steps :, :19
            ]
            runtime._agent = SimpleNamespace(
                predict_action=lambda *_args, **_kwargs: {
                    "pred_action": pred_action,
                    "control_action": control_action,
                    "tail": tail,
                }
            )

            self.assertEqual(runtime_config.deployment.inference_seed, 1066)
            prediction = runtime.predict(
                preflight_module._synthetic_observation(runtime_config)
            )

            expected = pred_action[
                :, allocation.n_obs_steps - 1 : allocation.horizon, :19
            ].numpy()[0]
            expected_from_policy_outputs = torch.cat(
                (control_action, tail), dim=1
            ).numpy()[0]
            np.testing.assert_array_equal(expected, expected_from_policy_outputs)
            self.assertEqual(prediction.arm_qpos.shape, (15, 7))
            self.assertEqual(prediction.hand_qpos.shape, (15, 12))
            np.testing.assert_array_equal(
                prediction.arm_qpos, expected[:, :7].astype(np.float64)
            )
            np.testing.assert_array_equal(
                prediction.hand_qpos, expected[:, 7:].astype(np.float64)
            )
            np.testing.assert_array_equal(
                prediction.arm_qpos[0], pred_action.numpy()[0, 1, :7]
            )
            np.testing.assert_array_equal(
                prediction.arm_qpos[-1], pred_action.numpy()[0, 15, :7]
            )
            self.assertFalse(
                np.array_equal(prediction.arm_qpos[0], pred_action.numpy()[0, 0, :7])
            )

            logical_step_ns = 1_000_000_000
            step_dt_ns = 62_500_000
            chunk = stamp_prediction_timing(
                prediction,
                logical_step_ns=logical_step_ns,
                step_dt_ns=step_dt_ns,
                inference_finished_ns=logical_step_ns - 1,
                command_lead_ns=0,
            )
            assert chunk is not None
            np.testing.assert_array_equal(chunk.arm_qpos[:8], expected[:8, :7])
            np.testing.assert_array_equal(chunk.arm_qpos[8:], expected[8:, :7])
            np.testing.assert_array_equal(
                chunk.target_monotonic_ns,
                logical_step_ns
                + np.arange(allocation.required_action_steps, dtype=np.uint64)
                * step_dt_ns,
            )

            invalid_pred_action = pred_action.clone()
            invalid_pred_action[:, -1, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                runtime._decode(invalid_pred_action)

    def test_adapter_warmup_restores_python_numpy_and_torch_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            runtime_config = self._projection(root).runtime
            allocation = runtime_config.artifact.allocation_contract
            runtime = DexManiPolicyRuntime(runtime_config)
            runtime._manifest = DeploymentManifest(
                action_key=allocation.action_key,
                n_obs_steps=allocation.n_obs_steps,
                n_action_steps=allocation.n_action_steps,
                action_dim=allocation.action_dim,
                horizon=allocation.horizon,
                hand_dim=12,
                control_action_dim=19,
                sensor_modalities=allocation.sensor_modalities,
                point_cloud_num_points=allocation.point_cloud_num_points,
                point_cloud_feature_dim=allocation.point_cloud_feature_dim,
            )

            def predict_action(*_args, **_kwargs):
                random.random()
                np.random.random()
                torch.rand(1)
                return {
                    "pred_action": torch.zeros(
                        (1, allocation.horizon, allocation.action_dim),
                        dtype=torch.float32,
                    )
                }

            runtime._agent = SimpleNamespace(predict_action=predict_action)
            random.seed(17)
            np.random.seed(19)
            torch.manual_seed(23)
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state().clone()

            timings_s = runtime.warmup(samples=3)

            self.assertEqual(len(timings_s), 3)
            self.assertEqual(random.getstate(), python_state)
            restored_numpy_state = np.random.get_state()
            self.assertEqual(restored_numpy_state[0], numpy_state[0])
            np.testing.assert_array_equal(restored_numpy_state[1], numpy_state[1])
            self.assertEqual(restored_numpy_state[2:], numpy_state[2:])
            torch.testing.assert_close(torch.random.get_rng_state(), torch_state)

    def test_adapter_discards_r3d_auxiliary_ee_tail_after_full_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            sidecar_path = next((root / "checkpoints").glob("*.deployment.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["schema_version"] = 2
            sidecar["allocation"].update(
                {
                    "action_dim": 28,
                    "control_action_dim": 19,
                    "auxiliary_action_layout": "joint19_ee9",
                    "rgb_shape": None,
                    "rgb_color_order": None,
                    "rgb_value_range": None,
                }
            )
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            runtime_config = self._projection(root).runtime
            allocation = runtime_config.artifact.allocation_contract
            runtime = DexManiPolicyRuntime(runtime_config)
            runtime._manifest = DeploymentManifest(
                action_key="action",
                n_obs_steps=allocation.n_obs_steps,
                n_action_steps=allocation.n_action_steps,
                action_dim=28,
                horizon=allocation.horizon,
                hand_dim=12,
                control_action_dim=19,
                auxiliary_action_layout="joint19_ee9",
                sensor_modalities=allocation.sensor_modalities,
                point_cloud_num_points=allocation.point_cloud_num_points,
                point_cloud_feature_dim=allocation.point_cloud_feature_dim,
            )
            pred_action = torch.arange(
                allocation.horizon * allocation.action_dim, dtype=torch.float32
            ).reshape(1, allocation.horizon, allocation.action_dim)

            prediction = runtime._decode(pred_action)

            expected = pred_action[:, 1:16, :19].numpy()[0]
            np.testing.assert_array_equal(prediction.arm_qpos, expected[:, :7])
            np.testing.assert_array_equal(prediction.hand_qpos, expected[:, 7:])

    def test_adapter_encodes_causal_rgb_history_in_model_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            sidecar_path = next((root / "checkpoints").glob("*.deployment.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["schema_version"] = 2
            sidecar["allocation"].update(
                {
                    "control_action_dim": 19,
                    "auxiliary_action_layout": "none",
                    "sensor_modalities": ["joint_state", "rgb"],
                    "observation_fields": ["arm_qpos", "hand_qpos", "rgb"],
                    "point_cloud_num_points": None,
                    "point_cloud_feature_dim": None,
                    "rgb_shape": [2, 3, 3],
                    "rgb_color_order": "rgb",
                    "rgb_value_range": "uint8_0_255",
                }
            )
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            artifact = resolve_policy_artifact(root)
            with self.assertRaisesRegex(TypeError, "unknown"):
                resolve_policy_runtime_config(
                    artifact=artifact,
                    runtime_config=resolve_runtime_config(),
                    data={"pointcloud_num_points": 1024},
                )
            projection = resolve_policy_runtime_config(
                artifact=artifact,
                runtime_config=resolve_runtime_config(),
                device="cpu",
            )
            runtime = DexManiPolicyRuntime(projection.runtime)
            allocation = projection.runtime.artifact.allocation_contract
            runtime._manifest = DeploymentManifest(
                action_key="action",
                n_obs_steps=allocation.n_obs_steps,
                n_action_steps=allocation.n_action_steps,
                action_dim=allocation.action_dim,
                horizon=allocation.horizon,
                hand_dim=12,
                control_action_dim=19,
                sensor_modalities=allocation.sensor_modalities,
                point_cloud_num_points=None,
                point_cloud_feature_dim=None,
                rgb_shape=allocation.rgb_shape,
                rgb_color_order="rgb",
                rgb_value_range="uint8_0_255",
            )
            runtime._device = "cpu"
            observation = preflight_module._synthetic_observation(projection.runtime)
            encoded = runtime._encode(observation)

            self.assertEqual(tuple(encoded["rgb"].shape), (1, 2, 3, 2, 3))
            self.assertEqual(encoded["rgb"].dtype, torch.float32)
            self.assertEqual(float(encoded["rgb"].max()), 0.0)

    def test_adapter_decodes_complete_ee_future_and_rejects_degenerate_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            sidecar_path = next((root / "checkpoints").glob("*.deployment.json"))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["allocation"]["action_key"] = "action_ee"
            sidecar["allocation"]["action_dim"] = 21
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            runtime_config = self._projection(root).runtime
            allocation = runtime_config.artifact.allocation_contract
            runtime = DexManiPolicyRuntime(runtime_config)
            runtime._manifest = DeploymentManifest(
                action_key="action_ee",
                n_obs_steps=allocation.n_obs_steps,
                n_action_steps=allocation.n_action_steps,
                action_dim=allocation.action_dim,
                horizon=allocation.horizon,
                hand_dim=12,
                tcp_dim=7,
                control_action_dim=21,
                sensor_modalities=allocation.sensor_modalities,
                point_cloud_num_points=allocation.point_cloud_num_points,
                point_cloud_feature_dim=allocation.point_cloud_feature_dim,
            )
            runtime._action_key = "action_ee"
            pred_action = torch.arange(
                allocation.horizon * allocation.action_dim, dtype=torch.float32
            ).reshape(1, allocation.horizon, allocation.action_dim)
            pred_action[:, :, 3:9] = torch.tensor(
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=torch.float32
            )

            prediction = runtime._decode(pred_action)

            expected = pred_action[:, 1:16, :21].numpy()[0]
            np.testing.assert_array_equal(prediction.ee_pos, expected[:, :3])
            np.testing.assert_array_equal(prediction.ee_rot6d, expected[:, 3:9])
            np.testing.assert_array_equal(prediction.hand_qpos, expected[:, 9:21])
            np.testing.assert_array_equal(prediction.ee_pos[0], pred_action[0, 1, :3])
            np.testing.assert_array_equal(
                prediction.hand_qpos[-1], pred_action[0, 15, 9:21]
            )

            invalid_tail = pred_action.clone()
            invalid_tail[:, 15, 3:9] = 0.0
            with self.assertRaisesRegex(ValueError, "zero or near-collinear"):
                runtime._decode(invalid_tail)

    def test_print_config_isolated_from_policy_torch_and_hardware_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            script = r"""
import runpy
import sys
namespace = runpy.run_path("examples/run_policy.py")
code = namespace["main"](
    ["--experiment-dir", sys.argv[1], "--device", "cpu", "--print-config"]
)
if code != 0:
    raise SystemExit(code)
forbidden = (
    "torch",
    "dexmani_policy",
    "dexmani_real.deployment.lifecycle",
    "dexmani_real.deployment.worker",
    "dexmani_real.robot.",
    "dexmani_real.sensor.",
)
loaded = [
    name for name in sys.modules
    if name == forbidden[0] or name.startswith(forbidden[1])
    or name in forbidden[2:4] or name.startswith(forbidden[4])
    or name.startswith(forbidden[5])
]
if loaded:
    raise SystemExit(",".join(sorted(loaded)))
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            receipt = json.loads(result.stdout)
            self.assertFalse(receipt["artifact"]["checkpoint_sha256_verified"])
            self.assertIsNone(receipt["artifact"]["actual_checkpoint_sha256"])
            self.assertEqual(
                receipt["runtime"]["fixed_runtime_target"],
                "dexmani_real.integrations.dexmani_policy:DexManiPolicyRuntime",
            )
            self.assertIn("real_source", receipt)
            self.assertIsNone(receipt["policy_package"])

    def test_preflight_module_import_isolated_from_sensor_policy_and_torch(self):
        script = r"""
import sys
import dexmani_real.deployment.preflight
forbidden = ("torch", "dexmani_policy", "dexmani_real.sensor.")
loaded = [
    name for name in sys.modules
    if name == forbidden[0] or name.startswith(forbidden[1])
    or name.startswith(forbidden[2])
]
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

    def test_cli_shadow_forwards_b_relative_limit_to_lifecycle(self):
        import runpy

        namespace = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/run_policy.py")
        )
        main = namespace["main"]
        module_globals = main.__globals__
        events: list[str] = []
        artifact = object()
        runtime = object()
        projection = SimpleNamespace(runtime=object(), sha256="d" * 64)
        module_globals["resolve_policy_artifact"] = lambda _path: (
            events.append("artifact") or artifact
        )
        module_globals["resolve_runtime_config"] = lambda *, yaml_path: (
            events.append("runtime") or runtime
        )

        def resolve_projection(**kwargs):
            self.assertIs(kwargs["artifact"], artifact)
            self.assertIs(kwargs["runtime_config"], runtime)
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["execution_mode"], "shadow")
            events.append("projection")
            return projection

        module_globals["resolve_policy_runtime_config"] = resolve_projection
        real_source = object()
        module_globals["resolve_real_source_identity"] = lambda: real_source
        lifecycle = ModuleType("dexmani_real.deployment.lifecycle")

        def run_lifecycle(
            actual_runtime,
            actual_projection,
            *,
            max_running_s,
            real_source: object,
        ):
            self.assertIs(actual_runtime, runtime)
            self.assertIs(actual_projection, projection.runtime)
            self.assertEqual(max_running_s, 120.0)
            self.assertIs(real_source, module_globals["resolve_real_source_identity"]())
            events.append("lifecycle")
            return 0

        lifecycle.run_policy_deployment = run_lifecycle
        with patch.dict(
            sys.modules,
            {"dexmani_real.deployment.lifecycle": lifecycle},
        ):
            code = main(
                [
                    "--experiment-dir",
                    "synthetic",
                    "--device",
                    "cpu",
                    "--max-running-seconds",
                    "120",
                ]
            )

        self.assertEqual(code, 0)
        self.assertEqual(events, ["artifact", "runtime", "projection", "lifecycle"])

    def test_cli_h4_execute_requires_and_forwards_immutable_bounds(self):
        import runpy

        namespace = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/run_policy.py")
        )
        main = namespace["main"]
        module_globals = main.__globals__
        artifact = SimpleNamespace(checkpoint_sha256_from_index="a" * 64)
        runtime = object()
        projection = SimpleNamespace(runtime=object(), sha256="d" * 64)
        module_globals["resolve_policy_artifact"] = lambda _path: artifact
        module_globals["resolve_runtime_config"] = lambda *, yaml_path: runtime

        def resolve_projection(**kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["execution_mode"], "execute")
            self.assertTrue(kwargs["hand_acknowledged"])
            bounds = kwargs["h4_execute_bounds"]
            self.assertEqual(bounds.max_published_endpoints, 1)
            self.assertEqual(bounds.acknowledgement_timeout_s, 2.0)
            self.assertEqual(bounds.max_running_s, 30.0)
            return projection

        module_globals["resolve_policy_runtime_config"] = resolve_projection
        module_globals["resolve_real_source_identity"] = lambda: SimpleNamespace(
            availability="available", dirty="false"
        )
        lifecycle = ModuleType("dexmani_real.deployment.lifecycle")

        def run_lifecycle(
            actual_runtime,
            actual_projection,
            *,
            max_running_s,
            real_source,
            invocation_argv,
            projection_sha256,
        ):
            self.assertIs(actual_runtime, runtime)
            self.assertIs(actual_projection, projection.runtime)
            self.assertIsNone(max_running_s)
            self.assertEqual(real_source.dirty, "false")
            self.assertIn("--execution-mode", invocation_argv)
            self.assertEqual(projection_sha256, projection.sha256)
            return 0

        lifecycle.run_policy_deployment = run_lifecycle
        with patch.dict(
            sys.modules,
            {"dexmani_real.deployment.lifecycle": lifecycle},
        ):
            code = main(
                [
                    "--experiment-dir",
                    "synthetic",
                    "--device",
                    "cpu",
                    "--execution-mode",
                    "execute",
                    "--hand",
                    "--max-running-seconds",
                    "30",
                    "--execute-max-published-endpoints",
                    "1",
                    "--execute-ack-timeout-seconds",
                    "2",
                    "--execute-expected-checkpoint-sha256",
                    "a" * 64,
                ]
            )

        self.assertEqual(code, 0)

    def test_cli_task_forwards_distinct_bounds_and_seed(self):
        import runpy

        namespace = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/run_policy.py")
        )
        main = namespace["main"]
        module_globals = main.__globals__
        artifact = SimpleNamespace(checkpoint_sha256_from_index="a" * 64)
        runtime = object()
        projection = SimpleNamespace(runtime=object(), sha256="d" * 64)
        module_globals["resolve_policy_artifact"] = lambda _path: artifact
        module_globals["resolve_runtime_config"] = lambda *, yaml_path: runtime

        def resolve_projection(**kwargs):
            self.assertEqual(kwargs["device"], "cpu")
            self.assertEqual(kwargs["execution_mode"], "task")
            self.assertEqual(kwargs["inference_seed"], 1066)
            self.assertIsNone(kwargs["h4_execute_bounds"])
            bounds = kwargs["task_execute_bounds"]
            self.assertEqual(bounds.max_published_endpoints, 320)
            self.assertEqual(bounds.acknowledgement_timeout_s, 2.0)
            self.assertEqual(bounds.max_running_s, 25.0)
            return projection

        module_globals["resolve_policy_runtime_config"] = resolve_projection
        module_globals["resolve_real_source_identity"] = lambda: SimpleNamespace(
            availability="available", dirty="false"
        )
        lifecycle = ModuleType("dexmani_real.deployment.lifecycle")
        lifecycle.run_policy_deployment = Mock(return_value=0)
        with patch.dict(
            sys.modules,
            {"dexmani_real.deployment.lifecycle": lifecycle},
        ):
            code = main(
                [
                    "--experiment-dir",
                    "synthetic",
                    "--device",
                    "cpu",
                    "--execution-mode",
                    "task",
                    "--hand",
                    "--inference-seed",
                    "1066",
                    "--max-running-seconds",
                    "25",
                    "--task-max-published-endpoints",
                    "320",
                    "--task-ack-timeout-seconds",
                    "2",
                    "--task-expected-checkpoint-sha256",
                    "a" * 64,
                ]
            )

        self.assertEqual(code, 0)
        lifecycle.run_policy_deployment.assert_called_once()
        self.assertEqual(
            lifecycle.run_policy_deployment.call_args.kwargs["projection_sha256"],
            projection.sha256,
        )

    def test_cli_defaults_to_cuda_and_rejects_removed_legacy_parameters(self):
        import runpy

        main = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "examples/run_policy.py")
        )["main"]

        with tempfile.TemporaryDirectory() as directory:
            root = _write_experiment(Path(directory) / "experiment")
            self.assertEqual(main(["--experiment-dir", str(root), "--print-config"]), 0)
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--experiment-dir",
                        str(root),
                        "--print-config",
                        "--checkpoint",
                        "old.pt",
                    ]
                )
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--experiment-dir",
                        str(root),
                        "--preflight-only",
                        "--max-running-seconds",
                        "120",
                    ]
                )
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--experiment-dir",
                        str(root),
                        "--print-config",
                        "--execution-mode",
                        "execute",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
