"""临时 Raw 的真实编解码、严格准入和共享几何；不读取历史实验或连接硬件。"""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import zarr

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.data.canonical.export import CanonicalExporter
from dexmani_real.data.canonical.schema import ProcessingConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.raw.schema import ARM_ACTION_SEMANTICS, RAW_FORMAT, validate_fixed_rate_rows
from dexmani_real.data.recorder import AsyncEpisodeRecorder
from dexmani_real.deployment.observation import build_policy_observation, project_policy_row
from dexmani_real.deployment.session import build_observation_kinematics
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.sensor.pointcloud.config import PointCloudConfig
from dexmani_real.sensor.pointcloud.hand_config import HandPointCloudConfig
from tests.offline_fixtures import make_row, recording_metadata


class RawContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dexmani-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = ExperimentConfig()
        self.processing = ProcessingConfig(
            pointcloud=PointCloudConfig(
                remove_table=False, num_points=16, voxel_size_m=.001,
                workspace=(-1., -1., 0., 1., 1., 2.),
            ),
            hand_pointcloud=HandPointCloudConfig(num_points=64),
        )
        self.rows = [make_row(1_000_000_000 + i * 62_500_000) for i in range(3)]
        for i, row in enumerate(self.rows):
            row.arm["qpos"] = np.asarray(self.runtime.arm.home_qpos) + i * .001
            row.hand["qpos"] = np.deg2rad(self.runtime.hand.home_qpos_deg)
            row.hand["tactile_aggregate"] = 2.5
            row.hand["tactile_dense"] = -1.5
            # Repeated camera and held tactile retain their original source times.
            row.camera["timestamp_ns"] = 1_000_000_000
            row.hand["tactile_aggregate_timestamp_ns"] = 1_000_000_000
        self.command = RobotCommand(
            1, np.asarray(self.runtime.arm.home_qpos) + .02,
            np.deg2rad(self.runtime.hand.home_qpos_deg) + .01,
        )

    def capture(self, name, *, wait=False, failed=False, cleanup_error=False):
        metadata = dict(
            recording_metadata(), camera_payload_mode="depth_to_color_aligned_rgbd",
            camera_color_intrinsics=np.array([[500., 0, 4], [0, 500., 4], [0, 0, 1]]),
            camera_color_distortion_model="none", camera_color_distortion_coeffs=np.zeros(5),
            camera_T_xarm_base_from_color=np.eye(4),
            hand_tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
        )
        recorder = AsyncEpisodeRecorder(self.root / name, control_hz=16)
        recorder.start_episode(task_label="offline", collection_source="teleop", metadata=metadata)
        for i, row in enumerate(self.rows):
            recorder.add_frame(row, None if wait and i == 1 else self.command)
        return recorder.save_episode(
            "hardware_fault" if failed else "operator",
            technical_status="failed" if failed else "normal",
            details=[dict(stage="robot_stop", message="stop unconfirmed")] if cleanup_error else None,
        )

    def test_new_raw_roundtrip_export_and_online_geometry(self):
        path = self.capture("valid", cleanup_error=True)
        original = (path / "data.h5").read_bytes()
        with EpisodeReader(path) as reader:
            self.assertEqual(reader.meta["format"], RAW_FORMAT)
            self.assertEqual(reader.meta["arm_action_semantics"], ARM_ACTION_SEMANTICS)
            self.assertNotIn("stop_result", reader.meta)
            self.assertFalse(any("dispatch" in name for name in reader.fields))
            reader.validate_completion()
            reader.validate_rows(chunk_frames=1)
            np.testing.assert_array_equal(reader.read_timing_rows()["camera_sequence"], [1, 1, 1])
        trajectory = load_trajectory(path)
        np.testing.assert_array_equal(trajectory.action_arm_joint[0], self.command.arm_qpos)
        self.assertFalse(np.array_equal(trajectory.action_arm_joint, trajectory.arm_qpos))
        output = self.root / "canonical.zarr"
        result = CanonicalExporter(self.processing, chunk_frames=1).export(path, output)
        self.assertEqual(result["total_frames"], 3)
        store = zarr.open_group(str(output), mode="r")["data"]
        specs = canonical_array_specs(16, 8, 8, 64)
        self.assertEqual(set(store), set(specs))
        self.assertEqual(len(specs), 13)
        for name, (tail, dtype) in specs.items():
            self.assertEqual(store[name].shape, (3, *tail))
            self.assertEqual(store[name].dtype, dtype)
            self.assertTrue(np.isfinite(store[name][:]).all())
        np.testing.assert_allclose(store["action"][0, :7], self.command.arm_qpos)
        np.testing.assert_array_equal(store["finger_tactile_force"][:], 2.5)
        np.testing.assert_array_equal(store["taxel_tactile_force"][:], -1.5)
        fields = ("joint_state", "eef_pose", "fingertip_xyz", "hand_pointcloud")
        info = SimpleNamespace(
            observation_fields=fields, n_obs_steps=3,
            fingertip_link_names=self.runtime.hand.fingertip_link_names,
        )
        runtime = replace(
            self.runtime, hand_pointcloud=self.processing.hand_pointcloud,
            execution=replace(self.runtime.execution, action_steps=1),
        )
        online = build_policy_observation(
            [project_policy_row(row, fields) for row in self.rows], info,
            kinematics=build_observation_kinematics(info, runtime),
        )
        for name in fields:
            np.testing.assert_allclose(online[name], store[name][:], rtol=1e-6, atol=1e-7)
        self.assertEqual((path / "data.h5").read_bytes(), original)

    def test_wait_and_failed_attempt_rejected_without_rewriting_raw(self):
        for wait, failed in ((True, False), (False, True)):
            with self.subTest(wait=wait):
                path = self.capture(f"invalid-{wait}", wait=wait, failed=failed)
                original = (path / "data.h5").read_bytes()
                with EpisodeReader(path) as reader:
                    np.testing.assert_allclose(reader["timestamp"][:], [0, .0625, .125])
                    targets = reader["action_arm_joint_target"][:]
                    if wait:
                        self.assertTrue(np.isnan(targets[1]).all())
                    else:
                        np.testing.assert_array_equal(targets[1], self.command.arm_qpos)
                with self.assertRaises(ValueError):
                    load_trajectory(path)
                with self.assertLogs("dexmani_real.data.canonical.export", level="WARNING"):
                    with self.assertRaisesRegex(ValueError, "all selected episodes were rejected"):
                        CanonicalExporter(self.processing).export(path, self.root / f"bad-{wait}.zarr")
                self.assertEqual((path / "data.h5").read_bytes(), original)

    def test_old_format_explicitly_rejected_read_only(self):
        path = self.capture("old-fixture")
        # This is a newly constructed temporary fixture, never a historical Raw.
        with h5py.File(path / "data.h5", "r+") as data:
            data["meta"].attrs["format"] = "dexmani.raw"
        original = (path / "data.h5").read_bytes()
        with self.assertRaisesRegex(RawDataError, "unsupported Raw format"):
            EpisodeReader(path)
        self.assertEqual((path / "data.h5").read_bytes(), original)

    def test_cumulative_grid_drift_keeps_original_quarter_period_bound(self):
        self.assertEqual(validate_fixed_rate_rows(np.arange(8) * .0645, 16), ())
        errors = validate_fixed_rate_rows(np.arange(9) * .0645, 16)
        self.assertIn("timing row 8", errors[0])
        self.assertIn("grid_deviation_ns=16000000", errors[0])
        self.assertIn("tolerance_ns=15625000", errors[0])
