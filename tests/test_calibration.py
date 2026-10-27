import itertools
import json
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.calibration.camera import aruco, extrinsics, hand_eye
from dexmani_real.calibration.camera.run import CameraCalibration
from dexmani_real.calibration.table_plane import resolve_table_plane, save_table_plane
from dexmani_real.calibration.vr_heading import VRCalibration, estimate_heading, load_vr_transform
from dexmani_real.geometry.plane import fit_table_plane
from scripts.calibrate_vr_heading import publish_vr_transform


def pose(rotvec, position):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    transform[:3, 3] = position
    return transform


def hand_eye_samples():
    rng = np.random.default_rng(7)
    camera = pose([0.2, -0.4, 0.3], [0.5, -0.2, 0.7])
    mount = pose([-0.1, 0.2, 0.05], [0.02, 0.01, 0.08])
    eef = np.array([pose(rng.normal(0, 0.5, 3), rng.uniform(-0.2, 0.2, 3)) for _ in range(16)])
    return camera, eef, np.linalg.inv(camera) @ eef @ mount


class HandEyeTests(unittest.TestCase):
    def test_known_transform_and_noisy_samples(self):
        camera, eef, marker = hand_eye_samples()
        result = hand_eye.solve_hand_eye(eef, marker)
        np.testing.assert_allclose(result.T_base_from_camera, camera, atol=1e-10)
        self.assertLess(result.position_rms_mm, 1e-8)
        self.assertLess(result.rotation_rms_deg, 1e-8)
        marker[:, :3, 3] += np.random.default_rng(3).normal(0, 0.0001, (len(marker), 3))
        result = hand_eye.solve_hand_eye(eef, marker)
        np.testing.assert_allclose(result.T_base_from_camera, camera, atol=0.001)
        self.assertLess(result.position_rms_mm, 1)

    def test_insufficient_samples_and_single_axis_are_rejected(self):
        _, eef, marker = hand_eye_samples()
        with self.assertRaisesRegex(ValueError, "at least"):
            hand_eye.solve_hand_eye(eef[:3], marker[:3])
        for rotations in (
            np.repeat(np.eye(3)[None], 16, axis=0),
            Rotation.from_euler("z", np.linspace(-1, 1, 16)).as_matrix(),
        ):
            with self.subTest(rotations=rotations.shape):
                eef[:, :3, :3] = rotations
                with self.assertRaisesRegex(ValueError, "excitation"):
                    hand_eye.solve_hand_eye(eef, marker)

    def test_invalid_candidate_is_reported_and_other_methods_still_run(self):
        camera, eef, marker = hand_eye_samples()
        solve = cv2.calibrateHandEye

        def candidate(*args, method):
            if method == cv2.CALIB_HAND_EYE_TSAI:
                return np.full((3, 3), np.nan), np.zeros((3, 1))
            return solve(*args, method=method)

        with (
            patch.object(hand_eye.cv2, "calibrateHandEye", side_effect=candidate),
            self.assertLogs(hand_eye.logger, "WARNING"),
        ):
            result = hand_eye.solve_hand_eye(eef, marker)
        self.assertNotEqual(result.method, "TSAI")
        np.testing.assert_allclose(result.T_base_from_camera, camera, atol=1e-10)

    def test_quality_gates_save_but_keeps_residuals_for_sample_removal(self):
        _, eef, marker = hand_eye_samples()
        for position, rotation in ((10, 0), (0, 10)):
            session = CameraCalibration(SimpleNamespace(), hand_geometry="absent")
            session.samples = list(zip(eef, marker))
            session.runtime_state = SimpleNamespace(run_id=1)
            session.stop_jog = Mock(return_value=True)
            session.cancel_requested = Mock(return_value=False)
            session.robot = Mock()
            errors = np.full(16, position), np.full(16, rotation)
            errors[0][-1] *= 2
            errors[1][-1] *= 2
            with (
                self.subTest(position=position, rotation=rotation),
                patch.object(hand_eye, "closed_loop_errors", return_value=errors),
                patch("dexmani_real.calibration.camera.run.save_camera_calibration") as save,
                patch("builtins.print"),
            ):
                session.solve_and_save()
                save.assert_not_called()
                self.assertFalse(session.calibration_saved)
                self.assertFalse(session.result.quality_ok)
                self.assertEqual(session.result.worst_sample, 15)
                session.handle_sample_events(["x"])
                self.assertEqual(len(session.samples), 15)
                np.testing.assert_array_equal(session.samples[-1][0], eef[14])
                self.assertIsNone(session.result)

    def test_qualified_candidate_precedes_smaller_position_error(self):
        _, eef, marker = hand_eye_samples()
        errors = [(np.zeros(16), np.full(16, 4)), (np.ones(16), np.zeros(16))]
        with patch.object(
            hand_eye, "closed_loop_errors", side_effect=[errors[0], *[errors[1]] * 4]
        ):
            result = hand_eye.solve_hand_eye(eef, marker)
        self.assertEqual(result.method, "PARK")
        self.assertTrue(result.quality_ok)

    def test_qualified_result_is_saved_with_capture_metadata(self):
        _, eef, marker = hand_eye_samples()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            session = CameraCalibration(SimpleNamespace(), hand_geometry="absent", output_path=path)
            session.samples = list(zip(eef, marker))
            session.runtime_state = SimpleNamespace(run_id=1)
            session.stop_jog = Mock(return_value=True)
            session.cancel_requested = Mock(return_value=False)
            session.serial = "test"
            session.camera_width, session.camera_height = 640, 480
            session.intrinsics, session.distortion = np.eye(3), np.zeros(5)
            with patch("builtins.print"):
                session.solve_and_save()
            self.assertTrue(session.calibration_saved)
            saved = json.loads(path.read_text())["camera_0"]
            self.assertEqual(saved["calibration_capture"]["sample_count"], len(eef))
            np.testing.assert_allclose(
                extrinsics.load_camera_extrinsics(path)["test"], session.result.T_base_from_camera
            )

    def test_all_solver_failures_remain_visible(self):
        _, eef, marker = hand_eye_samples()
        with (
            patch.object(hand_eye.cv2, "calibrateHandEye", side_effect=cv2.error("solver failed")),
            self.assertLogs(hand_eye.logger, "WARNING") as logs,
        ):
            with self.assertRaisesRegex(ValueError, "no hand-eye candidate"):
                hand_eye.solve_hand_eye(eef, marker)
        self.assertEqual(len(logs.output), 5)
        self.assertTrue(all("solver failed" in message for message in logs.output))

    def test_cancel_during_solve_does_not_publish(self):
        camera, eef, marker = hand_eye_samples()
        session = CameraCalibration(SimpleNamespace(), hand_geometry="absent")
        session.samples = list(zip(eef, marker))
        session.runtime_state = SimpleNamespace(run_id=1)
        session.stop_jog = Mock(return_value=True)
        session.cancel_requested = Mock(return_value=False)

        def solve(*args):
            session.runtime_state.run_id += 1
            return hand_eye.HandEyeResult(camera, "PARK", np.zeros(16), np.zeros(16))

        with (
            patch("dexmani_real.calibration.camera.run.solve_hand_eye", side_effect=solve),
            patch("dexmani_real.calibration.camera.run.save_camera_calibration") as save,
        ):
            session.solve_and_save()
        save.assert_not_called()
        self.assertIsNone(session.result)

    def test_sample_removal_invalidates_residuals(self):
        camera, eef, marker = hand_eye_samples()
        session = CameraCalibration(SimpleNamespace(), hand_geometry="absent")
        session.samples = list(zip(eef, marker))
        session.result = hand_eye.HandEyeResult(camera, "PARK", np.arange(16), np.zeros(16))
        session.cancel_requested = Mock(return_value=False)
        session.robot = Mock()
        with patch("builtins.print"):
            session.handle_sample_events(["x", "backspace"])
        self.assertEqual(len(session.samples), 14)
        self.assertIsNone(session.result)
        np.testing.assert_array_equal(session.samples[-1][0], eef[13])


class MarkerTests(unittest.TestCase):
    def setUp(self):
        self.sampler = aruco.MarkerSampler(
            np.array([[600.0, 0, 200], [0, 600, 200], [0, 0, 1]]), np.zeros(5)
        )
        self.arm_config = SimpleNamespace(homing=SimpleNamespace(convergence_rad=0.01))
        self.camera_config = SimpleNamespace(max_frame_age_s=0.1)

    def test_real_marker_detection(self):
        image = np.full((400, 400), 255, dtype=np.uint8)
        image[120:280, 120:280] = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT),
            aruco.MARKER_ID,
            160,
        )
        result = self.sampler.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        self.assertIsNotNone(result)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(result[1][2], 600 * aruco.MARKER_SIZE_M / 160, delta=0.001)

    def test_capture_uses_distinct_frames_and_so3_mean(self):
        frames = [
            {
                "timestamp_ns": index + 1,
                "ring_sequence": index + 1,
                "color_frame_number": number,
                "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
            }
            for index, number in enumerate([1, 1, 2, 3, 4, 5])
        ]
        stationary = {"qpos": np.zeros(7)}
        self.sampler.stationary_arm = Mock(return_value=stationary)
        self.sampler.detect = Mock(
            side_effect=[
                (np.array([0, 0, angle]), np.array([0, 0, 1.0]))
                for angle in [np.pi - 0.01, -np.pi + 0.01, np.pi, np.pi, np.pi]
            ]
        )
        with (
            patch.object(aruco, "read_camera_frame", side_effect=frames) as read,
            patch.object(aruco, "sample_is_fresh", return_value=True),
            patch.object(aruco.time, "monotonic_ns", return_value=0),
            patch.object(aruco.time, "sleep"),
        ):
            joints, transform = self.sampler.capture(
                None, Mock(), camera_config=self.camera_config, arm_config=self.arm_config
            )
        self.assertEqual(read.call_count, 6)
        self.assertEqual(self.sampler.detect.call_count, 5)
        np.testing.assert_allclose(
            transform[:3, :3], Rotation.from_euler("z", np.pi).as_matrix(), atol=1e-10
        )
        np.testing.assert_array_equal(joints, np.zeros(7))

    def test_capture_rejects_drift(self):
        self.sampler.stationary_arm = Mock(
            side_effect=[{"qpos": np.zeros(7)}, {"qpos": np.ones(7)}]
        )
        self.sampler.detect = Mock(return_value=(np.zeros(3), np.zeros(3)))
        frame_number = itertools.count(1)

        def frame(*args, **kwargs):
            n = next(frame_number)
            return dict(
                timestamp_ns=n,
                ring_sequence=n,
                color_frame_number=n,
                rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            )

        with (
            patch.object(aruco, "read_camera_frame", side_effect=frame),
            patch.object(aruco, "sample_is_fresh", return_value=True),
            patch.object(aruco.time, "monotonic_ns", return_value=0),
        ):
            with self.assertRaisesRegex(ValueError, "drifted"):
                self.sampler.capture(
                    None, Mock(), camera_config=self.camera_config, arm_config=self.arm_config
                )

    def test_capture_times_out_and_honors_cancellation(self):
        self.sampler.stationary_arm = Mock(return_value={"qpos": np.zeros(7)})
        with (
            patch.object(aruco, "read_camera_frame", return_value=None),
            patch.object(aruco.time, "monotonic", side_effect=itertools.count(0, 0.5)),
            patch.object(aruco.time, "sleep"),
        ):
            with self.assertRaises(TimeoutError):
                self.sampler.capture(
                    None, Mock(), camera_config=self.camera_config, arm_config=self.arm_config
                )
        with patch.object(aruco, "read_camera_frame") as read:
            with self.assertRaises(CancelledError):
                self.sampler.capture(
                    None,
                    Mock(side_effect=CancelledError),
                    camera_config=self.camera_config,
                    arm_config=self.arm_config,
                )
        read.assert_not_called()


class CalibrationFileTests(unittest.TestCase):
    def test_optional_default_missing_is_distinct_from_explicit_or_invalid(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            with patch.object(extrinsics, "CAMERAS_PATH", path):
                self.assertIsNone(extrinsics.load_optional_camera_extrinsics())
                with self.assertRaises(FileNotFoundError):
                    extrinsics.load_optional_camera_extrinsics(path)
                path.write_text("{")
                with self.assertRaises(json.JSONDecodeError):
                    extrinsics.load_optional_camera_extrinsics()
                path.write_text(
                    json.dumps(
                        {
                            "camera": {
                                "serial": "one",
                                "type": "eye_to_hand",
                                "pose": {
                                    "position": [float("nan"), 0, 0],
                                    "orientation": [1, 0, 0, 0],
                                },
                            }
                        }
                    )
                )
                with self.assertRaisesRegex(ValueError, "finite"):
                    extrinsics.load_optional_camera_extrinsics()

    def test_serial_update_preserves_other_cameras_and_cancel_preserves_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            first = pose([0.1, 0.2, 0.3], [0.5, 0.2, 0.1])
            extrinsics.save_camera_calibration(first, "one", path)
            extrinsics.save_camera_calibration(np.eye(4), "two", path)
            original_other = json.loads(path.read_text())["camera_1"]
            extrinsics.save_camera_calibration(np.eye(4), "one", path)
            values = extrinsics.load_camera_extrinsics(path)
            np.testing.assert_allclose(values["one"], np.eye(4))
            self.assertEqual(json.loads(path.read_text())["camera_1"], original_other)
            before = path.read_bytes()
            with self.assertRaises(CancelledError):
                extrinsics.save_camera_calibration(first, "one", path, cancelled=lambda: True)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_duplicate_serial_and_invalid_pose_fail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            extrinsics.save_camera_calibration(np.eye(4), "one", path)
            data = json.loads(path.read_text())
            data["camera_1"] = data["camera_0"]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                extrinsics.load_camera_extrinsics(path)
            with self.assertRaises(ValueError):
                extrinsics.pose_to_matrix([0, 0, 0], [0, 0, 0, 0])

    def test_table_roundtrip_and_invalid_normal(self):
        rng = np.random.default_rng(1)
        points = np.column_stack((rng.uniform(-0.5, 0.5, (1000, 2)), np.zeros(1000)))
        fit = fit_table_plane(points)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "table.json"
            save_table_plane(path, fit)
            config = SimpleNamespace(plane_path=path)
            np.testing.assert_allclose(resolve_table_plane(config), fit.plane_abcd)
            path.write_text(json.dumps(dict(a=0, b=0, c=-1, d=0)))
            with self.assertRaisesRegex(ValueError, "upward"):
                resolve_table_plane(config)


class VRHeadingTests(unittest.TestCase):
    def test_wraparound_outliers_and_transform(self):
        angles = np.deg2rad([179, -179, 180] * 15 + [0])
        forwards = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        result = estimate_heading(forwards, "head")
        self.assertEqual(result.frames, 45)
        self.assertAlmostEqual(abs(result.theta_deg), 180)
        np.testing.assert_allclose(result.transform @ [-1, 0, 0], [1, 0, 0], atol=1e-10)
        self.assertEqual(result.grade, "excellent")

    def test_degenerate_and_insufficient_samples_fail(self):
        for forwards in (
            np.tile([0, 0, 1], (40, 1)),
            np.tile([1, 0, 0], (5, 1)),
            np.tile([[1, 0, 0], [-1, 0, 0]], (20, 1)),
        ):
            with self.subTest(shape=forwards.shape), self.assertRaises(ValueError):
                estimate_heading(np.asarray(forwards), "head")

    def test_roundtrip_and_poor_quality_gate(self):
        good = VRCalibration(74, "head", 1, 2, 100)
        poor = VRCalibration(74, "head", 6, 10, 100)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vr.json"
            publish_vr_transform(good, path)
            self.assertEqual(load_vr_transform(path), good)
            with self.assertRaises(ValueError):
                publish_vr_transform(poor, path)
            self.assertEqual(load_vr_transform(path), good)
            diagnostic = Path(directory) / "diagnostic.json"
            publish_vr_transform(poor, diagnostic, force=True)
            with self.assertRaisesRegex(ValueError, "recalibration"):
                load_vr_transform(diagnostic)
            with self.assertRaises(ValueError):
                VRCalibration(float("nan"), "head", 1, 2, 100)


if __name__ == "__main__":
    unittest.main()
