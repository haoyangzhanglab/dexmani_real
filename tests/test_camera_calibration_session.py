"""Offline contracts for camera-calibration sampling and provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.calibration.camera import session
from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
from dexmani_real.calibration.camera.solver import (
    ArucoConfig,
    CalibrationConfig,
    CalibrationSamples,
    save_camera_calibration,
)
from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.runtime.safety import SafetyState


class _Value:
    def __init__(self, value: object) -> None:
        self.value = value


class _Shared:
    def __init__(self) -> None:
        self.error_state = _Value(False)
        self.safety_state = _Value(int(SafetyState.ARMED))


class _ArmFk:
    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(qpos[:3], dtype=np.float64).copy(),
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )


def _arm_state(
    *,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
    source_monotonic_ns: int = 999_000_000,
) -> dict[str, object]:
    return {
        "connected": True,
        "error_code": 0,
        "state_valid": True,
        "source_monotonic_ns": source_monotonic_ns,
        "qpos": np.zeros(7, dtype=np.float64) if qpos is None else qpos,
        "qvel": np.zeros(7, dtype=np.float64) if qvel is None else qvel,
    }


class CameraCalibrationSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = resolve_experiment_config()
        self.shared = _Shared()
        self.samples = CalibrationSamples()
        self.intrinsics = np.eye(3, dtype=np.float64)
        self.distortion = np.zeros(5, dtype=np.float64)
        self.aruco = ArucoConfig(capture_frames=3)

    def _capture(
        self,
        arm_states: list[dict[str, object]],
        *,
        marker_found: bool = True,
    ) -> tuple[Mock, Mock]:
        detect = Mock(
            return_value=(
                (
                    np.zeros(3, dtype=np.float64),
                    np.array([0.1, 0.2, 0.3], dtype=np.float64),
                )
                if marker_found
                else None
            )
        )
        read_state = Mock(side_effect=arm_states)
        with (
            patch(
                "dexmani_real.calibration.camera.session.read_arm_state_dict",
                read_state,
            ),
            patch(
                "dexmani_real.calibration.camera.session._detect_aruco_stable",
                detect,
            ),
            patch(
                "dexmani_real.calibration.camera.session.make_arm_fk",
                return_value=_ArmFk(),
            ),
            patch(
                "dexmani_real.calibration.camera.session.eef_rpy_from_rot6d",
                return_value=np.zeros(3, dtype=np.float64),
            ),
            patch(
                "dexmani_real.calibration.camera.session.time.monotonic_ns",
                return_value=1_000_000_000,
            ),
        ):
            session._capture_calibration_sample(
                self.shared,
                self.runtime,
                object(),
                self.intrinsics,
                self.distortion,
                self.samples,
                self.aruco,
            )
        return detect, read_state

    def _assert_sample_rejection_preserves_armed_state(self) -> None:
        self.assertEqual(len(self.samples), 0)
        self.assertFalse(self.shared.error_state.value)
        self.assertEqual(self.shared.safety_state.value, int(SafetyState.ARMED))

    def test_stationary_before_and_after_capture_is_accepted(self) -> None:
        before = _arm_state()
        after_qpos = np.zeros(7, dtype=np.float64)
        after_qpos[0] = self.runtime.arm.homing.convergence_rad * 0.5
        after = _arm_state(qpos=after_qpos)

        detect, read_state = self._capture([before, after])

        self.assertEqual(detect.call_count, 1)
        self.assertEqual(read_state.call_count, 2)
        self.assertEqual(len(self.samples), 1)
        np.testing.assert_allclose(self.samples.tvec_ee2base[0], after_qpos[:3])
        self.assertFalse(self.shared.error_state.value)
        self.assertEqual(self.shared.safety_state.value, int(SafetyState.ARMED))

    def test_high_velocity_before_capture_rejects_without_camera_capture(self) -> None:
        qvel = np.zeros(7, dtype=np.float64)
        qvel[0] = self.runtime.arm.homing.velocity_convergence_rad_s * 1.1

        detect, read_state = self._capture([_arm_state(qvel=qvel)])

        detect.assert_not_called()
        self.assertEqual(read_state.call_count, 1)
        self._assert_sample_rejection_preserves_armed_state()

    def test_high_velocity_after_capture_rejects_without_fault(self) -> None:
        qvel = np.zeros(7, dtype=np.float64)
        qvel[0] = self.runtime.arm.homing.velocity_convergence_rad_s * 1.1

        detect, read_state = self._capture([_arm_state(), _arm_state(qvel=qvel)])

        self.assertEqual(detect.call_count, 1)
        self.assertEqual(read_state.call_count, 2)
        self._assert_sample_rejection_preserves_armed_state()

    def test_large_joint_drift_rejects_without_fault(self) -> None:
        after_qpos = np.zeros(7, dtype=np.float64)
        after_qpos[2] = self.runtime.arm.homing.convergence_rad * 1.1

        detect, read_state = self._capture([_arm_state(), _arm_state(qpos=after_qpos)])

        self.assertEqual(detect.call_count, 1)
        self.assertEqual(read_state.call_count, 2)
        self._assert_sample_rejection_preserves_armed_state()

    def test_stale_arm_feedback_rejects_without_camera_capture(self) -> None:
        stale = _arm_state(source_monotonic_ns=1)

        detect, read_state = self._capture([stale])

        detect.assert_not_called()
        self.assertEqual(read_state.call_count, 1)
        self._assert_sample_rejection_preserves_armed_state()

    def test_missing_marker_rejects_after_the_capture_window_without_fault(
        self,
    ) -> None:
        detect, read_state = self._capture(
            [_arm_state(), _arm_state()],
            marker_found=False,
        )

        self.assertEqual(detect.call_count, 1)
        self.assertEqual(read_state.call_count, 2)
        self._assert_sample_rejection_preserves_armed_state()


class CameraCalibrationProvenanceTest(unittest.TestCase):
    def test_solve_records_finite_capture_provenance(self) -> None:
        samples = CalibrationSamples()
        for index in range(3):
            samples.append(
                np.array([0.01 * index, 0.0, 0.0]),
                np.zeros(3),
                np.zeros(3),
                np.array([0.0, 0.0, 0.5]),
            )
        planner = SimpleNamespace(
            kin=SimpleNamespace(
                base_pose_world=SimpleNamespace(
                    q=np.array([1.0, 0.0, 0.0, 0.0]),
                    p=np.zeros(3),
                )
            )
        )
        intrinsics = np.array(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )
        distortion = np.array([0.1, -0.1, 0.0, 0.0, 0.0])
        saved = Mock()
        with (
            patch(
                "dexmani_real.calibration.camera.session.calibrate_and_select",
                return_value=(
                    np.eye(4),
                    "PARK",
                    np.array([1.0, 2.0, 3.0]),
                    np.array([0.1, 0.2, 0.3]),
                    [("PARK", 0.1)],
                ),
            ),
            patch(
                "dexmani_real.calibration.camera.session.save_camera_calibration",
                saved,
            ),
        ):
            transform = session._solve_and_save_calibration(
                samples,
                planner,
                "serial-1",
                CalibrationConfig(min_samples=3),
                intrinsics=intrinsics,
                distortion=distortion,
            )

        self.assertIsNotNone(transform)
        capture = saved.call_args.kwargs["calibration_capture"]
        self.assertEqual(capture["width"], 640)
        self.assertEqual(capture["height"], 480)
        self.assertEqual(capture["fps"], 30)
        self.assertEqual(capture["method"], "PARK")
        self.assertEqual(capture["sample_count"], 3)
        np.testing.assert_allclose(
            tuple(capture["position_error_mm"].values()),
            (2.0, 0.816496580927726, 3.0),
        )
        np.testing.assert_allclose(
            tuple(capture["rotation_error_deg"].values()),
            (0.2, 0.0816496580927726, 0.3),
        )
        self.assertTrue(capture["calibrated_at_utc"].endswith("+00:00"))
        json.dumps(capture, allow_nan=False)

    def test_save_preserves_other_entries_and_capture_metadata(self) -> None:
        capture = {
            "width": 640,
            "height": 480,
            "fps": 30,
            "intrinsics": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "method": "PARK",
            "sample_count": 3,
            "position_error_mm": {"mean": 1.0, "std": 0.5, "max": 2.0},
            "rotation_error_deg": {"mean": 0.1, "std": 0.05, "max": 0.2},
            "calibrated_at_utc": "2026-09-06T00:00:00Z",
        }
        other = {
            "serial": "other-serial",
            "type": "eye_to_hand",
            "pose": {
                "position": [0.0, 0.0, 0.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps({"other": other}), encoding="utf-8")

            save_camera_calibration(
                np.eye(4),
                "new-serial",
                path,
                calibration_capture=capture,
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            extrinsics = CameraExtrinsics(str(path))
            new_camera = extrinsics.resolve_name_by_serial("new-serial")
            new_extrinsics = extrinsics.get_extrinsics(new_camera)
        self.assertEqual(payload["other"], other)
        new_entry = next(
            entry for entry in payload.values() if entry["serial"] == "new-serial"
        )
        self.assertEqual(new_entry["calibration_capture"], capture)
        json.dumps(payload, allow_nan=False)
        np.testing.assert_allclose(new_extrinsics, np.eye(4))


if __name__ == "__main__":
    unittest.main()
