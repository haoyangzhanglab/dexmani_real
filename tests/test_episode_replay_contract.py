"""Offline contract checks for the physical episode replay boundary."""

from __future__ import annotations

import contextlib
import inspect
import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dexmani_real.robot.episode_replay import (
    EpisodeReplayConfig,
    EpisodeReplayer,
    ReplayStatus,
    TrajectoryData,
    load_trajectory,
    modeled_hand_actions,
    replay_episode,
    require_hand_actions,
)
from examples.replay_episode import _parse_args


def _trajectory(hand_actions: np.ndarray | None) -> TrajectoryData:
    frame_count = 2
    return TrajectoryData(
        episode_path="fixture",
        num_frames=frame_count,
        fps=16.0,
        task_label="test",
        action_arm_joint=np.zeros((frame_count, 7), dtype=np.float64),
        action_hand_joint=hand_actions,
        arm_qpos=np.zeros((frame_count, 7), dtype=np.float64),
        hand_qpos=None,
        arm_ee=None,
        action_source="sent",
    )


def _episode_reader(
    *, include_sent_stream: bool = True
) -> tuple[mock.MagicMock, mock.MagicMock]:
    frame_count = 2
    h5: dict[str, object] = {
        "meta": SimpleNamespace(
            attrs={
                "num_frames": frame_count,
                "task_label": "test",
                "resolved_config_sha256": "0" * 64,
            }
        ),
        "arm_qpos": np.zeros((frame_count, 7), dtype=np.float64),
        "action_arm_joint": np.ones((frame_count, 7), dtype=np.float64),
        "action_hand_joint": np.zeros((frame_count, 12), dtype=np.float64),
    }
    if include_sent_stream:
        h5["action_arm_joint_sent"] = np.full((frame_count, 7), 2.0, dtype=np.float64)

    reader = mock.MagicMock()
    reader.meets_min_duration = True
    reader.timing = SimpleNamespace(rate_hz=16.0)
    reader.h5f = h5
    context = mock.MagicMock()
    context.__enter__.return_value = reader
    return reader, context


class EpisodeReplayContractTest(unittest.TestCase):
    def test_loader_has_no_alternate_source_or_dry_run_controls(self) -> None:
        self.assertEqual(
            tuple(inspect.signature(load_trajectory).parameters), ("episode_path",)
        )
        parameters = inspect.signature(EpisodeReplayer).parameters
        self.assertNotIn("dry_run", parameters)
        self.assertIs(parameters["shared"].default, inspect.Parameter.empty)
        self.assertIs(parameters["runtime"].default, inspect.Parameter.empty)

    def test_cli_rejects_removed_dry_run_option(self) -> None:
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            _parse_args(["episode", "--dry-run"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --dry-run", stderr.getvalue())

    def test_loader_requires_valid_episode_and_uses_exact_sent_stream(self) -> None:
        reader, context = _episode_reader()
        with (
            mock.patch(
                "dexmani_real.robot.episode_replay.resolve_episode_path",
                return_value=(".", "fixture"),
            ),
            mock.patch(
                "dexmani_real.robot.episode_replay.EpisodeReader",
                return_value=context,
            ),
        ):
            trajectory = load_trajectory("fixture")

        reader.require_valid.assert_called_once_with(purpose="physical replay")
        self.assertEqual(trajectory.action_source, "sent")
        np.testing.assert_array_equal(trajectory.action_arm_joint, 2.0)

    def test_loader_does_not_fall_back_to_command_stream(self) -> None:
        _reader, context = _episode_reader(include_sent_stream=False)
        with (
            mock.patch(
                "dexmani_real.robot.episode_replay.resolve_episode_path",
                return_value=(".", "fixture"),
            ),
            mock.patch(
                "dexmani_real.robot.episode_replay.EpisodeReader",
                return_value=context,
            ),
            self.assertRaisesRegex(ValueError, "requires /action_arm_joint_sent"),
        ):
            load_trajectory("fixture")

    def test_hand_actions_are_required(self) -> None:
        trajectory = _trajectory(None)
        with self.assertRaisesRegex(ValueError, "requires recorded hand data"):
            require_hand_actions(trajectory)

    def test_hand_actions_must_be_finite_and_fixed_shape(self) -> None:
        invalid_actions = np.zeros((2, 12), dtype=np.float64)
        invalid_actions[1, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "must be finite shape"):
            modeled_hand_actions(_trajectory(invalid_actions))

    def test_valid_hand_actions_are_returned_as_float64(self) -> None:
        actions = np.zeros((2, 12), dtype=np.float32)
        result = modeled_hand_actions(_trajectory(actions))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, actions)

    def test_preflight_rejection_happens_before_shared_storage_creation(self) -> None:
        config = EpisodeReplayConfig(
            output_dir="unused",
            evaluate_consistency=False,
            config_sha256="0" * 64,
        )
        with (
            mock.patch(
                "dexmani_real.robot.episode_replay.verify_replay_preflight",
                side_effect=ValueError("blocked"),
            ),
            mock.patch(
                "dexmani_real.robot.episode_replay.SharedStorage.create"
            ) as create,
        ):
            outcome = replay_episode(
                _trajectory(np.zeros((2, 12), dtype=np.float64)),
                mock.Mock(),
                config,
            )

        self.assertEqual(outcome.status, ReplayStatus.REJECTED)
        self.assertIn("blocked", outcome.reason)
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
