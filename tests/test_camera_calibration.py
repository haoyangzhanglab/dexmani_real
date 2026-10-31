"""相机标定的只读 SDK 边界、五帧采样与真实离线文件读写。"""

import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.config.real import RealConfig
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.camera_calibrate.aruco import ARUCO_DICT, MARKER_ID, ArucoSampler
from dexmani_real.utils.camera_calibrate.calibrate import collect_samples, solve_samples


def marker_rgb():
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), MARKER_ID, 80)
    rgb = np.full((120, 160, 3), 255, np.uint8)
    rgb[20:100, 40:120] = marker[..., None]
    return rgb


def arm_feedback(qpos=None, qvel=None, timestamp_ns=None):
    return dict(qpos=np.zeros(7) if qpos is None else qpos,
                qvel=np.zeros(7) if qvel is None else qvel,
                timestamp_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns)


class CalibrationCollectionTests(unittest.TestCase):
    def test_two_groups_only_connect_read_close_and_save_real_npz(self):
        runtime = replace(RealConfig(), camera=CameraConfig(width=160, height=120))
        intrinsics = CameraIntrinsics(160, 120, 150., 150., 80., 60., "none", (0.,) * 5)
        image, sequence = marker_rgb(), [0]

        def read_camera():
            sequence[0] += 1
            return SimpleNamespace(rgb=image, timestamp_ns=time.monotonic_ns(),
                                   color_frame_number=sequence[0])

        camera = SimpleNamespace(connect=Mock(), disconnect=Mock(), read=read_camera,
                                 active_serial="test-camera", color_intrinsics=intrinsics, depth_scale_m=.001)
        methods = ("motion_enable", "set_mode", "set_state", "set_collision_sensitivity",
                   "set_tcp_load", "set_joint_maxacc", "set_servo_angle_j", "set_servo_angle")
        apis = []
        for _ in range(2):
            api = Mock()
            api.axis = 7
            api.get_err_warn_code.return_value = (0, [0, 0])
            api.get_joint_states.return_value = (0, [np.zeros(7).tolist(), np.zeros(7).tolist()])
            apis.append(api)
        constructor = Mock(side_effect=apis)
        wrapper = SimpleNamespace(XArmAPI=constructor)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"xarm": SimpleNamespace(wrapper=wrapper), "xarm.wrapper": wrapper},
        ), patch("dexmani_real.sensor.realsense.RealSenseCamera", return_value=camera), patch(
            "dexmani_real.robot.xarm7.driver.XArm7.initialize", side_effect=AssertionError("must remain read-only"),
        ) as initialize, patch("builtins.input", side_effect=["", "", "q"]):
            output = Path(directory) / "capture"
            self.assertEqual(collect_samples(runtime, output), 2)
            initialize.assert_not_called()
            self.assertEqual(constructor.call_count, 2)
            for api in apis:
                self.assertGreaterEqual(api.get_joint_states.call_count, 7)
                api.disconnect.assert_called_once()
                for name in methods:
                    getattr(api, name).assert_not_called()
            metadata = json.loads((output / "capture.json").read_text())
            self.assertEqual(metadata["camera"]["serial"], "test-camera")
            self.assertEqual(metadata["criteria"]["capture_frames"], 5)
            for identity in range(2):
                with np.load(output / "samples" / f"{identity:06d}.npz", allow_pickle=False) as sample:
                    self.assertEqual(int(sample["sample_id"]), identity)
                    self.assertEqual(sample["rgb"].shape, (5, 120, 160, 3))
                    self.assertEqual(sample["arm_qpos"].shape, (5, 7))
                    self.assertTrue(sample["marker_detected"].all())
                    self.assertTrue(np.all(np.diff(sample["camera_timestamp_ns"]) > 0))
                    self.assertTrue(np.all(np.diff(sample["color_frame_number"]) > 0))
                    self.assertTrue(np.all(np.diff(sample["arm_timestamp_ns"]) > 0))
                    self.assertLess(int(sample["arm_before_timestamp_ns"]), int(sample["arm_after_timestamp_ns"]))
                    self.assertTrue(np.isfinite(sample["T_base_from_eef"]).all())
            with self.assertRaises(FileExistsError):
                collect_samples(runtime, output)
        camera.connect.assert_called_once()
        camera.disconnect.assert_called_once()

    def test_sdk_read_failure_closes_devices_and_does_not_save_a_sample(self):
        runtime = replace(RealConfig(), camera=CameraConfig(width=160, height=120))
        color = CameraIntrinsics(160, 120, 150., 150., 80., 60., "none", (0.,) * 5)
        camera = SimpleNamespace(connect=Mock(), disconnect=Mock(), read=Mock(),
                                 active_serial="test-camera", color_intrinsics=color, depth_scale_m=.001)
        arm = SimpleNamespace(connect=Mock(), close=Mock(), read=Mock(side_effect=OSError("SDK read failed")))
        with tempfile.TemporaryDirectory() as directory, patch(
            "dexmani_real.sensor.realsense.RealSenseCamera", return_value=camera,
        ), patch("dexmani_real.robot.xarm7.driver.XArm7", return_value=arm), patch(
            "builtins.input", return_value="",
        ):
            output = Path(directory) / "capture"
            with self.assertRaisesRegex(OSError, "SDK read failed"):
                collect_samples(runtime, output)
            self.assertEqual(list((output / "samples").iterdir()), [])
        arm.close.assert_called_once()
        camera.disconnect.assert_called_once()


class ArucoCaptureTests(unittest.TestCase):
    def setUp(self):
        self.runtime = replace(RealConfig(), camera=CameraConfig(width=160, height=120))
        self.sampler = ArucoSampler(np.array([[150., 0., 80.], [0., 150., 60.], [0., 0., 1.]]), np.zeros(5))
        self.image = marker_rgb()

    def test_five_fresh_frames_skip_stale_and_duplicate_sources(self):
        now, calls, stamps = [1_000_000_000], [0], {}

        def tick():
            now[0] += 1_000_000
            return now[0]

        def sleep(seconds):
            now[0] += int(seconds * 1e9)

        def read_camera():
            calls[0] += 1
            if calls[0] == 1:
                stamp = tick()
                now[0] += 500_000_000
                return SimpleNamespace(rgb=self.image, timestamp_ns=stamp, color_frame_number=99)
            identity = calls[0] // 2
            stamps.setdefault(identity, tick())
            return SimpleNamespace(rgb=self.image, timestamp_ns=stamps[identity], color_frame_number=identity)

        with patch("dexmani_real.utils.camera_calibrate.aruco.time.monotonic_ns", side_effect=tick), patch(
            "dexmani_real.utils.camera_calibrate.aruco.time.monotonic", side_effect=lambda: now[0] / 1e9,
        ), patch("dexmani_real.utils.camera_calibrate.aruco.time.sleep", side_effect=sleep):
            sample = self.sampler.capture(read_camera, lambda: arm_feedback(timestamp_ns=tick()),
                                          camera_config=self.runtime.camera, arm_config=self.runtime.arm)
        self.assertEqual([frame["color_frame_number"] for frame in sample["frames"]], [1, 2, 3, 4, 5])
        self.assertEqual([frame["timestamp_ns"] for frame in sample["frames"]], [stamps[i] for i in range(1, 6)])
        self.assertGreater(calls[0], 5)

    def test_motion_and_drift_reject_capture(self):
        read_camera = Mock()
        with self.assertRaisesRegex(ValueError, "arm is moving"):
            self.sampler.capture(read_camera, lambda: arm_feedback(qvel=np.ones(7)),
                                 camera_config=self.runtime.camera, arm_config=self.runtime.arm)
        read_camera.assert_not_called()
        reads, sequence = [0], [0]

        def read_arm():
            reads[0] += 1
            return arm_feedback(qpos=np.full(7, .2 if reads[0] >= 7 else 0.))

        def stationary_camera():
            sequence[0] += 1
            return SimpleNamespace(rgb=self.image, timestamp_ns=time.monotonic_ns(), color_frame_number=sequence[0])

        with self.assertRaisesRegex(ValueError, "arm drifted"):
            self.sampler.capture(stationary_camera, read_arm,
                                 camera_config=self.runtime.camera, arm_config=self.runtime.arm)

    def test_bad_rgb_shape_and_dtype_fail_clearly(self):
        for image in (np.zeros((120, 160, 3), np.float32), np.zeros((60, 80, 3), np.uint8)):
            with self.subTest(shape=image.shape, dtype=image.dtype):
                def read_camera():
                    return SimpleNamespace(rgb=image, timestamp_ns=time.monotonic_ns(), color_frame_number=1)
                with self.assertRaisesRegex(ValueError, "RGB"):
                    self.sampler.capture(read_camera, arm_feedback,
                                         camera_config=self.runtime.camera, arm_config=self.runtime.arm)

    def test_no_new_color_frame_reaches_the_capture_deadline(self):
        now = [1_000_000_000]

        def tick():
            now[0] += 1_000_000
            return now[0]

        def sleep(seconds):
            now[0] += int(seconds * 1e9)

        def repeated_camera():
            return SimpleNamespace(rgb=self.image, timestamp_ns=tick(), color_frame_number=1)

        with patch("dexmani_real.utils.camera_calibrate.aruco.time.monotonic_ns", side_effect=tick), patch(
            "dexmani_real.utils.camera_calibrate.aruco.time.monotonic", side_effect=lambda: now[0] / 1e9,
        ), patch("dexmani_real.utils.camera_calibrate.aruco.time.sleep", side_effect=sleep):
            with self.assertRaisesRegex(TimeoutError, "fresh calibration image unavailable"):
                self.sampler.capture(repeated_camera, lambda: arm_feedback(timestamp_ns=tick()),
                                     camera_config=self.runtime.camera, arm_config=self.runtime.arm)


class CalibrationSolveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "capture"
        (self.source / "samples").mkdir(parents=True)
        color = CameraIntrinsics(160, 120, 150., 150., 80., 60., "none", (0.,) * 5)
        (self.source / "capture.json").write_text(json.dumps(dict(camera=dict(serial="test-camera", color=color.to_dict()))))
        rng = np.random.default_rng(31)
        self.transform = np.eye(4)
        self.transform[:3, :3] = Rotation.from_rotvec([.17, -.23, .31]).as_matrix()
        self.transform[:3, 3] = [.35, -.2, .5]
        mount = np.eye(4)
        mount[:3, 3] = [.015, .025, .11]
        self.identities = list(range(10, 58, 3))
        for index, identity in enumerate(self.identities):
            eef = np.eye(4)
            eef[:3, :3] = Rotation.from_rotvec(rng.uniform(-.8, .8, 3)).as_matrix()
            eef[:3, 3] = rng.uniform(-.25, .25, 3)
            np.savez(self.source / "samples" / f"{index:06d}.npz", sample_id=identity,
                     T_base_from_eef=eef, T_camera_from_marker=np.linalg.inv(self.transform) @ eef @ mount)

    def test_real_npz_solve_and_original_id_exclusion_stay_offline(self):
        with patch("dexmani_real.robot.xarm7.driver.XArm7.connect", side_effect=AssertionError("offline")), patch(
            "dexmani_real.sensor.realsense.RealSenseCamera.connect", side_effect=AssertionError("offline"),
        ):
            for excluded in ((), (self.identities[3],)):
                with self.subTest(excluded=excluded):
                    output = self.root / ("all.json" if not excluded else "excluded.json")
                    result = solve_samples(self.source, output, exclude=excluded)
                    np.testing.assert_allclose(result.T_base_from_camera, self.transform, atol=1e-10, rtol=0)
                    report = json.loads(output.with_suffix(".result.json").read_text())
                    self.assertEqual(report["excluded_sample_ids"], list(excluded))
                    self.assertEqual(report["used_sample_ids"], [i for i in self.identities if i not in excluded])
                    self.assertIn(report["worst_sample_id"], report["used_sample_ids"])
                    saved = json.loads(output.read_text())["test-camera"]
                    self.assertEqual(saved["calibration_capture"]["sample_count"], len(self.identities) - len(excluded))

    def test_unknown_and_invalid_sample_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown excluded sample IDs"):
            solve_samples(self.source, self.root / "unknown.json", exclude=[999])
        invalid = self.source / "samples" / "invalid.npz"
        for identity in (1.5, -1, self.identities[0]):
            with self.subTest(identity=identity):
                np.savez(invalid, sample_id=identity, T_base_from_eef=np.eye(4), T_camera_from_marker=np.eye(4))
                with self.assertRaisesRegex(ValueError, "sample_id"):
                    solve_samples(self.source, self.root / "invalid.json")
        self.assertFalse((self.root / "invalid.json").exists())

    def test_actual_io_failures_are_not_converted_to_success(self):
        with self.assertRaises(FileNotFoundError):
            solve_samples(self.root / "missing", self.root / "missing.json")
        blocked = self.root / "not-a-directory"
        blocked.write_text("keep")
        with self.assertRaises(OSError):
            solve_samples(self.source, blocked / "camera.json")
        self.assertEqual(blocked.read_text(), "keep")
        (self.source / "samples" / "broken.npz").write_bytes(b"not an npz")
        with self.assertRaises(ValueError):
            solve_samples(self.source, self.root / "broken.json")


if __name__ == "__main__":
    unittest.main()
