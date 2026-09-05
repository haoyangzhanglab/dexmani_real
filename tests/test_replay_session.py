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

from dexmani_real.replay.controller import ReplayOutcome, ReplayStatus
from dexmani_real.replay.session import _post_shutdown_outcome
from dexmani_real.replay.trajectory import (
    TrajectoryData,
    _processed_replay_source,
    load_processed_trajectory,
    verify_replay_preflight,
)
from dexmani_real.runtime.safety import SafetyState
from dexmani_real.runtime.workers import ProcessExit, ShutdownReport


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


class ReplayIntegritySeparationTest(unittest.TestCase):
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
            resolved_config_sha256="a" * 64,
            model_provenance=(
                ("arm_hand_collision_urdf_sha256", "b" * 64),
                ("arm_hand_urdf_sha256", "b" * 64),
                ("arm_hand_srdf_sha256", "b" * 64),
            ),
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

    def test_model_hash_mismatch_warns_only_after_complete_physical_preflight(
        self,
    ) -> None:
        planner = MagicMock()
        planner.is_workspace_segment_safe.return_value = True
        planner.collision_model.check_transition_collision_free.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            model_paths = []
            for name in ("collision.urdf", "hand.urdf", "robot.srdf"):
                path = Path(directory) / name
                path.write_text(name, encoding="utf-8")
                model_paths.append(path)
            with (
                patch(
                    "dexmani_real.replay.trajectory.XArm7MotionPlanner",
                    return_value=planner,
                ),
                patch(
                    "dexmani_real.replay.trajectory.preflight_model_paths",
                    return_value=tuple(model_paths),
                ),
                patch("dexmani_real.replay.trajectory.logger.warning") as warning,
            ):
                verify_replay_preflight(
                    self._trajectory(),
                    self._runtime(),
                    provenance_sha256="a" * 64,
                )
        self.assertEqual(planner.is_workspace_segment_safe.call_count, 2)
        self.assertEqual(
            planner.collision_model.check_transition_collision_free.call_count, 2
        )
        warning.assert_called_once()
        self.assertIn("reproducibility warning", warning.call_args.args[0])

    def test_failed_physical_preflight_never_downgrades_hash_mismatch_to_warning(
        self,
    ) -> None:
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
                    provenance_sha256="different" * 8,
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
                    provenance_sha256="different" * 8,
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
                    provenance_sha256="different" * 8,
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
            patch(
                "dexmani_real.replay.trajectory._reproducibility_warnings",
                return_value=(),
            ),
        ):
            verify_replay_preflight(
                trajectory,
                runtime,
                provenance_sha256="a" * 64,
            )

        canonical_middle = planner.is_workspace_segment_safe.call_args_list[1].args[1]
        self.assertAlmostEqual(float(canonical_middle[0]), 0.2)
        collision_middle = (
            planner.collision_model.check_transition_collision_free.call_args_list[
                1
            ].args[1]
        )
        self.assertAlmostEqual(float(collision_middle[0]), 0.2)

    def test_processed_raw_data_hash_mismatch_remains_a_hard_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory)
            (source_path / "data.h5").touch()
            with (
                patch(
                    "dexmani_real.replay.trajectory._processed_replay_source",
                    return_value=(
                        source_path,
                        np.asarray((0,), dtype=np.int64),
                        "a" * 64,
                        1,
                        "b" * 64,
                    ),
                ),
                patch(
                    "dexmani_real.replay.trajectory.sha256_file",
                    return_value="c" * 64,
                ),
                patch("dexmani_real.replay.trajectory.load_trajectory") as load_raw,
            ):
                with self.assertRaisesRegex(
                    ValueError, "raw source data.h5 hash mismatch"
                ):
                    load_processed_trajectory("selection.h5")
                load_raw.assert_not_called()

    def test_processed_replay_rejects_tampered_sample_provenance_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "selection.h5"
            with h5py.File(artifact_path, "w") as artifact:
                artifact.attrs.update(
                    {
                        "schema_name": "dexmani-real-processed-hdf5",
                        "schema_version": 11,
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
                        "source_member_sha256_json": json.dumps(
                            {
                                "data.h5": "a" * 64,
                                "depth.h5": "b" * 64,
                                "rgb.mp4": "c" * 64,
                            }
                        ),
                        "source_resolved_config_sha256": "d" * 64,
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

    def test_processed_config_hash_mismatch_is_deferred_to_post_preflight_warning(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory)
            (source_path / "data.h5").touch()
            with (
                patch(
                    "dexmani_real.replay.trajectory._processed_replay_source",
                    return_value=(
                        source_path,
                        np.asarray((0, 1), dtype=np.int64),
                        "d" * 64,
                        2,
                        "b" * 64,
                    ),
                ),
                patch(
                    "dexmani_real.replay.trajectory.sha256_file",
                    return_value="b" * 64,
                ),
                patch(
                    "dexmani_real.replay.trajectory.load_trajectory",
                    return_value=self._trajectory(),
                ),
            ):
                trajectory = load_processed_trajectory("selection.h5")
        self.assertEqual(
            trajectory.provenance_warnings,
            ("processed source config hash does not match the raw source",),
        )


if __name__ == "__main__":
    unittest.main()
