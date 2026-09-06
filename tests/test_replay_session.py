"""Offline terminal-state tests for replay session shutdown."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from dexmani_real.dataset.processed import PROCESSED_SCHEMA_VERSION
from dexmani_real.replay import session as replay_session
from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus
from dexmani_real.replay.session import _post_shutdown_outcome
from dexmani_real.replay.trajectory import (
    TrajectoryData,
    _processed_replay_source,
    load_processed_trajectory,
    verify_replay_preflight,
)
from dexmani_real.runtime.processes import ProcessExit, ShutdownReport
from dexmani_real.runtime.safety import SafetyState


class _Value:
    def __init__(self, value: bool | int) -> None:
        self.value = value


class ReplayShutdownOutcomeTest(unittest.TestCase):
    def test_cleanup_error_is_not_reclassified_as_physical_fault(self) -> None:
        shared = SimpleNamespace(
            estop_request=_Value(False),
            error_state=_Value(False),
            safety_state=_Value(int(SafetyState.DISARMED)),
        )
        report = ShutdownReport(
            exits=(ProcessExit("arm", 0, "graceful"),),
            shared_closed=False,
        )

        outcome = _post_shutdown_outcome(
            shared,
            ReplayOutcome(ReplayStatus.COMPLETED),
            report,
        )

        self.assertIs(outcome.status, ReplayStatus.CLEANUP_FAILED)
        self.assertFalse(outcome.successful)
        self.assertFalse(shared.error_state.value)
        self.assertEqual(shared.safety_state.value, int(SafetyState.DISARMED))

    def test_worker_failure_remains_a_physical_runtime_fault(self) -> None:
        shared = SimpleNamespace(
            estop_request=_Value(False),
            error_state=_Value(False),
            safety_state=_Value(int(SafetyState.DISARMED)),
        )
        report = ShutdownReport(
            exits=(ProcessExit("arm", -15, "terminate"),),
            shared_closed=False,
        )

        outcome = _post_shutdown_outcome(
            shared,
            ReplayOutcome(ReplayStatus.COMPLETED),
            report,
        )

        self.assertIs(outcome.status, ReplayStatus.FAULT)


class ReplayOutputAdmissionTest(unittest.TestCase):
    @staticmethod
    def _config(output_dir: Path) -> replay_session.EpisodeReplayConfig:
        return replay_session.EpisodeReplayConfig(
            output_dir=str(output_dir),
            evaluate_consistency=True,
        )

    def test_existing_output_rejects_before_preflight_or_channel_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_file = root / "replay.npz"
            output_file.write_bytes(b"prior-result")
            output_dir = root / "replay"
            output_dir.mkdir()
            (output_dir / "metrics.json").write_text("{}", encoding="utf-8")
            dangling_output = root / "dangling-replay"
            dangling_output.symlink_to(
                root / "missing-target", target_is_directory=True
            )

            for occupied_path in (output_file, output_dir, dangling_output):
                with self.subTest(path=occupied_path):
                    with (
                        patch(
                            "dexmani_real.replay.session.verify_replay_preflight",
                            side_effect=AssertionError("preflight must not run"),
                        ) as preflight,
                        patch(
                            "dexmani_real.replay.session.RuntimeChannels.create"
                        ) as create_channels,
                    ):
                        outcome = replay_session.replay_episode(
                            object(),
                            object(),
                            self._config(occupied_path),
                        )

                    self.assertIs(outcome.status, ReplayStatus.REJECTED)
                    self.assertEqual(
                        outcome.reason,
                        "Replay output already exists; choose another --output directory.",
                    )
                    preflight.assert_not_called()
                    create_channels.assert_not_called()

    def test_missing_and_empty_output_directories_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_output = root / "missing"
            empty_output = root / "empty"
            empty_output.mkdir()

            replay_session._validate_replay_output_dir(missing_output)
            replay_session._validate_replay_output_dir(empty_output)

            self.assertFalse(missing_output.exists())
            self.assertTrue(empty_output.is_dir())


class ReplayPhysicalPreflightTest(unittest.TestCase):
    def _trajectory(self) -> TrajectoryData:
        return TrajectoryData(
            episode_path="episode",
            num_frames=2,
            fps=16.0,
            task_label="pick",
            action_arm_joint=np.zeros((2, 7), dtype=np.float64),
            action_hand_joint=np.zeros((2, 12), dtype=np.float64),
            arm_qpos=np.zeros((2, 7), dtype=np.float64),
            hand_qpos=np.zeros((2, 12), dtype=np.float64),
            arm_ee=None,
            action_source="sent",
        )

    @staticmethod
    def _runtime() -> SimpleNamespace:
        return SimpleNamespace(
            policy=SimpleNamespace(
                hand_enabled=True,
                workspace=SimpleNamespace(
                    x_min=-1.0,
                    x_max=1.0,
                    y_min=-1.0,
                    y_max=1.0,
                    z_min=-1.0,
                    z_max=1.0,
                ),
            ),
            arm=SimpleNamespace(
                joint_limit_lower=(-1.0,) * 7,
                joint_limit_upper=(1.0,) * 7,
            ),
            hand=SimpleNamespace(
                qpos_min_rad=(-1.0,) * 12,
                qpos_max_rad=(1.0,) * 12,
                mechanical_qpos_min_rad=(-1.5,) * 12,
                mechanical_qpos_max_rad=(1.5,) * 12,
            ),
            environment=SimpleNamespace(static_boxes=()),
        )

    def _three_frame_trajectory(self) -> TrajectoryData:
        trajectory = self._trajectory()
        trajectory.num_frames = 3
        trajectory.action_arm_joint = np.zeros((3, 7), dtype=np.float64)
        trajectory.action_hand_joint = np.zeros((3, 12), dtype=np.float64)
        trajectory.arm_qpos = np.zeros((3, 7), dtype=np.float64)
        trajectory.hand_qpos = np.zeros((3, 12), dtype=np.float64)
        return trajectory

    def test_failed_physical_preflight_rejects_before_warning(self) -> None:
        planner = MagicMock()
        planner.is_workspace_segment_safe.return_value = False
        with (
            patch(
                "dexmani_real.replay.trajectory.XArm7MotionPlanner",
                return_value=planner,
            ),
            patch("dexmani_real.replay.trajectory.logger.warning") as warning,
        ):
            with self.assertRaisesRegex(ValueError, "recorded start->0"):
                verify_replay_preflight(
                    self._trajectory(),
                    self._runtime(),
                )
        warning.assert_not_called()

    def test_intermediate_arm_limit_violation_rejects_before_planner_or_warning(
        self,
    ) -> None:
        trajectory = self._three_frame_trajectory()
        trajectory.action_arm_joint[1, 1] = 1.01
        with (
            patch("dexmani_real.replay.trajectory.XArm7MotionPlanner") as planner,
            patch("dexmani_real.replay.trajectory.logger.warning") as warning,
        ):
            with self.assertRaisesRegex(
                ValueError, "arm action at frame 1 violates joint limits"
            ):
                verify_replay_preflight(
                    trajectory,
                    self._runtime(),
                )
        planner.assert_not_called()
        warning.assert_not_called()

    def test_intermediate_hand_limit_violation_rejects_before_planner_or_warning(
        self,
    ) -> None:
        trajectory = self._three_frame_trajectory()
        trajectory.action_hand_joint[1, 0] = 1.01
        with (
            patch("dexmani_real.replay.trajectory.XArm7MotionPlanner") as planner,
            patch("dexmani_real.replay.trajectory.logger.warning") as warning,
        ):
            with self.assertRaisesRegex(
                ValueError, "hand action at frame 1 violates command joint limits"
            ):
                verify_replay_preflight(
                    trajectory,
                    self._runtime(),
                )
        planner.assert_not_called()
        warning.assert_not_called()

    def test_equivalent_arm_angles_use_canonical_stream_for_geometry(self) -> None:
        trajectory = self._three_frame_trajectory()
        trajectory.action_arm_joint[1, 0] = 2.0 * np.pi + 0.2
        trajectory.action_arm_joint[2, 0] = 0.3
        runtime = self._runtime()
        runtime.arm.joint_limit_lower = (-2.0 * np.pi,) + (-1.0,) * 6
        runtime.arm.joint_limit_upper = (2.0 * np.pi,) + (1.0,) * 6
        planner = MagicMock()
        planner.is_workspace_segment_safe.return_value = True
        planner.collision_model.check_transition_collision_free.return_value = True
        with (
            patch(
                "dexmani_real.replay.trajectory.XArm7MotionPlanner",
                return_value=planner,
            ),
        ):
            verify_replay_preflight(trajectory, runtime)

        canonical_middle = planner.is_workspace_segment_safe.call_args_list[1].args[1]
        self.assertAlmostEqual(float(canonical_middle[0]), 0.2)
        collision_middle = (
            planner.collision_model.check_transition_collision_free.call_args_list[
                1
            ].args[1]
        )
        self.assertAlmostEqual(float(collision_middle[0]), 0.2)

    def test_processed_replay_rejects_tampered_sample_provenance_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "selection.h5"
            with h5py.File(artifact_path, "w") as artifact:
                artifact.attrs.update(
                    {
                        "schema_name": "dexmani-real-processed-hdf5",
                        "schema_version": PROCESSED_SCHEMA_VERSION,
                        "domain": "real",
                        "profile": "state_only",
                        "episode_steps": 1,
                        "source_frames": 2,
                        "dt": 0.1,
                        "source_contiguity_tolerance_s": 1e-6,
                        "source_decision_json": json.dumps(
                            {
                                "profile": "state_only",
                                "source_frames": 2,
                                "selected_frames": 1,
                                "dropped_frames": 1,
                                "accepted": True,
                                "rejected_reason": None,
                                "selected_source_ranges": [[0, 1]],
                                "selected_segment_ends": [1],
                                "hard_invalid_reason_names": [],
                                "source_path": directory,
                            }
                        ),
                    }
                )
                provenance = artifact.create_group("provenance")
                provenance.attrs["drop_reason_bit_names_json"] = json.dumps(
                    {"0": "dropped"}
                )
                provenance.create_dataset(
                    "source_row_index", data=np.asarray([0], dtype=np.int64)
                )
                provenance.create_dataset(
                    "source_sample_index", data=np.asarray([10.0], dtype=np.float64)
                )
                provenance.create_dataset(
                    "source_timestamp_s", data=np.asarray([1.0], dtype=np.float64)
                )
                provenance.create_dataset(
                    "source_segment_ends", data=np.asarray([1], dtype=np.int64)
                )
                provenance.create_dataset(
                    "source_keep_mask", data=np.asarray([True, False], dtype=np.bool_)
                )
                provenance.create_dataset(
                    "source_drop_reason_bits",
                    data=np.asarray([0, 1], dtype=np.uint64),
                )

            with self.assertRaisesRegex(
                ValueError, "source_sample_index dtype must be int64"
            ):
                _processed_replay_source(artifact_path)


if __name__ == "__main__":
    unittest.main()
