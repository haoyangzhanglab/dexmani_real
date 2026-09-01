"""Offline-only tests for the Promotion Gate A qualification harness."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from gate_a_offline import (
    FORBIDDEN_HARDWARE_MODULES,
    ReplayPayload,
    _inspect_fixture,
    _operator_commands,
    _replay_feed_segment,
    build_observation,
    discover_recorded_episode,
)

from dexmani_real.deployment.artifact import ResolvedPolicyArtifact


class _FakeReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.validity: Any = SimpleNamespace(value="VALID")
        self.h5f = {"meta": SimpleNamespace(attrs={"task_label": "pick_place_toy"})}

    def __enter__(self) -> "_FakeReader":
        from dexmani_real.recording.reader import ValidityState

        self.validity = ValidityState.VALID
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class GateAOfflineTest(unittest.TestCase):
    @staticmethod
    def _inspect_stdout(*, producer_commit: str, n_action_steps: int = 8) -> str:
        return json.dumps(
            {
                "artifact": {
                    "producer": {"commit": producer_commit},
                    "allocation": {
                        "n_obs_steps": 2,
                        "n_action_steps": n_action_steps,
                        "horizon": 16,
                        "required_action_steps": 15,
                        "action_dim": 19,
                    },
                }
            }
        )

    def _payload(self) -> ReplayPayload:
        return ReplayPayload(
            episode_name="episode_recorded",
            schema_version=24,
            task_name="pick_place_toy",
            control_hz=16.0,
            grid_dt_s=0.0625,
            resolved_config_sha256="a" * 64,
            window_start_index=0,
            source_rows=np.array([0, 1], dtype=np.int64),
            arm_qpos=np.zeros((2, 7), dtype=np.float64),
            arm_qvel=np.zeros((2, 7), dtype=np.float64),
            arm_source_sequence=np.array([1, 2], dtype=np.uint64),
            arm_source_ns=np.array([1_010, 1_070], dtype=np.int64),
            arm_publish_ns=np.array([1_011, 1_071], dtype=np.int64),
            hand_qpos=np.zeros((2, 12), dtype=np.float64),
            hand_source_sequence=np.array([3, 4], dtype=np.uint64),
            hand_source_ns=np.array([1_000, 1_060], dtype=np.int64),
            hand_publish_ns=np.array([1_001, 1_061], dtype=np.int64),
            camera_source_sequence=np.array([10, 11], dtype=np.int64),
            camera_source_ns=np.array([1_020, 1_080], dtype=np.int64),
            camera_publish_ns=np.array([1_025, 1_085], dtype=np.int64),
            camera_generation=np.array([1, 1], dtype=np.int64),
            depth_frame_number=np.array([20, 21], dtype=np.int64),
            color_frame_number=np.array([30, 31], dtype=np.int64),
            logical_step_ns=np.array([1_030, 1_090], dtype=np.int64),
            point_cloud=np.zeros((2, 1024, 6), dtype=np.float32),
        )

    def test_recorded_rebase_preserves_every_relative_delta(self) -> None:
        payload = self._payload()
        observation = build_observation(
            payload,
            replay_epoch_ns=10_000_000_000,
            run_generation=2,
            observation_id=1,
        )
        arm_history = observation.arm_history
        hand_history = observation.hand_history
        self.assertIsNotNone(arm_history)
        self.assertIsNotNone(hand_history)
        assert arm_history is not None and hand_history is not None

        self.assertEqual(
            np.diff(arm_history.source_monotonic_ns).tolist(),
            np.diff(payload.arm_source_ns).tolist(),
        )
        self.assertEqual(
            np.diff(hand_history.source_monotonic_ns).tolist(),
            np.diff(payload.hand_source_ns).tolist(),
        )
        self.assertEqual(
            observation.pointcloud_history[0].publish_monotonic_ns
            - observation.pointcloud_history[0].source_monotonic_ns,
            int(payload.camera_publish_ns[0] - payload.camera_source_ns[0]),
        )
        self.assertEqual(len(observation.pointcloud_history), 2)
        self.assertEqual(arm_history.values.shape, (2, 7))
        self.assertEqual(hand_history.values.shape, (2, 12))

    def test_discovery_selects_newest_valid_matching_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "pick_place_toy" / "episode_old"
            newer = root / "pick_place_toy" / "episode_new"
            for episode in (older, newer):
                episode.mkdir(parents=True)
                for name in ("data.h5", "depth.h5", "rgb.mp4"):
                    (episode / name).touch()
            older_time = 1_000_000_000
            newer_time = 2_000_000_000
            older.touch()
            newer.touch()
            import os

            os.utime(older, ns=(older_time, older_time))
            os.utime(newer, ns=(newer_time, newer_time))
            with patch("gate_a_offline.EpisodeReader", _FakeReader):
                selected, counts = discover_recorded_episode(
                    root, task_name="pick_place_toy"
                )

        self.assertEqual(selected.name, "episode_new")
        self.assertEqual(counts, {"candidate_count": 2, "valid_matching_count": 2})

    def test_current_fixture_inspect_executes_real_cli_print_config(self) -> None:
        commit = "a" * 40
        completed = SimpleNamespace(
            returncode=0,
            stdout=self._inspect_stdout(producer_commit=commit),
            stderr="",
        )
        with patch("gate_a_offline.subprocess.run", return_value=completed) as run:
            receipt = _inspect_fixture(
                fixture_experiment=Path("/fixture"),
                policy_commit=commit,
                policy_root=Path("/policy"),
                real_root=Path("/real"),
                python=Path("/python"),
            )

        command = run.call_args.args[0]
        self.assertEqual(command[1], "/real/examples/run_policy.py")
        self.assertIn("--experiment-dir", command)
        self.assertIn("--device", command)
        self.assertIn("cpu", command)
        self.assertIn("--print-config", command)
        self.assertNotIn("--execution-mode", command)
        self.assertNotIn("--hand", command)
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], "/real:/policy")
        self.assertEqual(receipt["control_action_dim"], 19)

    def test_current_fixture_inspect_rejects_wrong_producer(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=self._inspect_stdout(producer_commit="b" * 40),
            stderr="",
        )
        with patch("gate_a_offline.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "producer mismatch"):
                _inspect_fixture(
                    fixture_experiment=Path("/fixture"),
                    policy_commit="a" * 40,
                    policy_root=Path("/policy"),
                    real_root=Path("/real"),
                    python=Path("/python"),
                )

    def test_replay_feeder_stays_inside_selected_source_segment(self) -> None:
        source_rows = np.array([10, 11, 12, 30, 31, 32, 33], dtype=np.int64)
        segment_ends = np.array([3, 7], dtype=np.int64)

        segment_start, segment_end, feed_start, feed_end = _replay_feed_segment(
            source_rows,
            segment_ends,
            window_start=0,
            window_steps=2,
            max_frames=20,
        )

        self.assertEqual((segment_start, segment_end), (0, 3))
        self.assertEqual((feed_start, feed_end), (0, 3))
        self.assertNotIn(30, source_rows[feed_start:feed_end])

    def test_replay_feeder_rejects_cross_segment_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one source segment"):
            _replay_feed_segment(
                np.array([10, 11, 12, 30, 31, 32, 33], dtype=np.int64),
                np.array([3, 7], dtype=np.int64),
                window_start=2,
                window_steps=2,
                max_frames=20,
            )

    def test_harness_has_no_hardware_owner_imports_or_lifecycle_call(self) -> None:
        source_path = Path(__file__).with_name("gate_a_offline.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)

        self.assertTrue(FORBIDDEN_HARDWARE_MODULES.isdisjoint(imported))
        self.assertNotIn("run_policy_deployment", calls)
        self.assertNotIn("build_policy_worker_specs", calls)

    def test_operator_commands_freeze_identity_and_one_endpoint_h4(self) -> None:
        artifact = cast(
            ResolvedPolicyArtifact,
            SimpleNamespace(
                producer=SimpleNamespace(commit="b" * 40),
                checkpoint_sha256_from_index="c" * 64,
            ),
        )
        commands = _operator_commands(
            artifact,
            real_commit="a" * 40,
            device="cuda:0",
            seed=7,
        )

        self.assertIn("--execution-mode shadow", commands["live_shadow"])
        self.assertIn("--inference-seed 7", commands["live_shadow"])
        self.assertIn("--execute-max-published-endpoints 1", commands["fresh_h4"])
        self.assertIn("--execute-ack-timeout-seconds 2", commands["fresh_h4"])
        self.assertIn("c" * 64, commands["fresh_h4"])


if __name__ == "__main__":
    unittest.main()
