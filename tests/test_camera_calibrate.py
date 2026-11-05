import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.utils.camera_calibrate import extrinsics
from dexmani_real.utils.camera_calibrate.calibrate import save_sample, solve_samples
from dexmani_real.utils.camera_calibrate.extrinsics import (
    load_camera_extrinsics,
    load_optional_camera_extrinsics,
    save_camera_calibration,
)
from dexmani_real.utils.camera_calibrate.hand_eye import HandEyeResult

SERIAL = "218622271441"
SAMPLE_IDS = tuple(reversed(range(10, 46, 3)))


@pytest.fixture
def capture(tmp_path):
    source = tmp_path / "capture"
    (source / "samples").mkdir(parents=True)
    (source / "capture.json").write_text(json.dumps({"camera": {"serial": SERIAL}}))
    rng = np.random.default_rng(17)
    transforms = np.tile(np.eye(4), (len(SAMPLE_IDS) + 2, 1, 1))
    transforms[:, :3, :3] = Rotation.from_rotvec(rng.uniform(-.7, .7, (len(transforms), 3))).as_matrix()
    transforms[:, :3, 3] = rng.uniform(-.3, .3, (len(transforms), 3))
    base_from_camera, eef_from_marker = transforms[:2]
    eef = transforms[2:]
    markers = np.linalg.inv(base_from_camera) @ eef @ eef_from_marker
    for index, sample_id in enumerate(SAMPLE_IDS):
        save_sample(source / "samples" / f"{index:06d}.npz", sample_id,
                    {"T_camera_from_marker": markers[index]}, eef[index])
    return source, base_from_camera, eef, markers


def test_serial_only_solve_saves_transform_and_preserves_other_cameras(capture, tmp_path, capsys):
    source, expected, _, _ = capture
    output = tmp_path / "cameras.json"
    legacy = {"T_base_from_camera": np.eye(4).tolist(), "calibration_capture": {"legacy": True}}
    output.write_text(json.dumps({SERIAL: legacy, "other-camera": legacy}))
    # 部署读取器继续忽略旧条目的附加字段。
    np.testing.assert_array_equal(load_camera_extrinsics(SERIAL, output), np.eye(4))
    np.testing.assert_array_equal(load_optional_camera_extrinsics(SERIAL, output), np.eye(4))

    result = solve_samples(source, output)

    assert isinstance(result, HandEyeResult) and result.quality_ok
    np.testing.assert_allclose(result.T_base_from_camera, expected, atol=1e-10)
    assert result.position_errors_mm.shape == result.rotation_errors_deg.shape == (len(SAMPLE_IDS),)
    cameras = json.loads(output.read_text())
    assert cameras[SERIAL] == {"T_base_from_camera": result.T_base_from_camera.tolist()}
    assert cameras["other-camera"] == legacy
    np.testing.assert_allclose(load_camera_extrinsics(SERIAL, output), expected, atol=1e-10)
    assert not output.with_suffix(".result.json").exists()
    text = capsys.readouterr().out
    for value in ("HORAUD", f"samples={len(SAMPLE_IDS)}", "RMS=", " mm / ", " deg;",
                  f"worst sample ID={sorted(SAMPLE_IDS)[result.worst_sample]}", str(output)):
        assert value in text


@pytest.mark.parametrize("report_state", ["existing", "unwritable", "same_suffix"])
def test_old_report_path_does_not_gate_saving(capture, tmp_path, report_state):
    source, expected, _, _ = capture
    output = tmp_path / ("cameras.result.json" if report_state == "same_suffix" else "cameras.json")
    old_report = output.with_suffix(".result.json")
    if report_state == "existing":
        old_report.write_text("historical report")
    elif report_state == "unwritable":
        old_report.mkdir()

    solve_samples(source, output)

    np.testing.assert_allclose(load_camera_extrinsics(SERIAL, output), expected, atol=1e-10)
    if report_state == "existing":
        assert old_report.read_text() == "historical report"
    elif report_state == "unwritable":
        assert old_report.is_dir() and not list(old_report.iterdir())


def test_exclusions_ignore_missing_ids_and_remove_present_samples(capture, tmp_path, capsys):
    source, expected, eef, markers = capture
    baseline = solve_samples(source, tmp_path / "baseline.json")
    missing = solve_samples(source, tmp_path / "missing.json", exclude=(999, 999))
    np.testing.assert_array_equal(missing.T_base_from_camera, baseline.T_base_from_camera)
    np.testing.assert_array_equal(missing.position_errors_mm, baseline.position_errors_mm)
    np.testing.assert_array_equal(missing.rotation_errors_deg, baseline.rotation_errors_deg)

    markers[0, :3, 3] += [.2, -.1, .1]
    np.savez(source / "samples" / "000000.npz", sample_id=SAMPLE_IDS[0],
             T_base_from_eef=eef[0], T_camera_from_marker=markers[0])
    capsys.readouterr()
    result = solve_samples(source, tmp_path / "filtered.json", exclude=(SAMPLE_IDS[0], SAMPLE_IDS[0], 999))

    np.testing.assert_allclose(result.T_base_from_camera, expected, atol=1e-10)
    assert result.position_errors_mm.shape == result.rotation_errors_deg.shape == (len(SAMPLE_IDS) - 1,)
    text = capsys.readouterr().out
    assert f"samples={len(SAMPLE_IDS) - 1}" in text
    assert f"worst sample ID={sorted(SAMPLE_IDS[1:])[result.worst_sample]}" in text


@pytest.mark.parametrize("sample_id", [SAMPLE_IDS[0], -1, 1.5, [10]])
def test_invalid_or_duplicate_sample_ids_rejected_even_when_excluded(capture, tmp_path, sample_id):
    source, _, eef, markers = capture
    np.savez(source / "samples" / "extra.npz", sample_id=sample_id,
             T_base_from_eef=eef[0], T_camera_from_marker=markers[0])
    output = tmp_path / "cameras.json"

    with pytest.raises(ValueError, match="sample_id"):
        solve_samples(source, output, exclude=(SAMPLE_IDS[0], 999))

    assert not output.exists()


@pytest.mark.parametrize("output_exists", [False, True])
@pytest.mark.parametrize("failure,error", [
    ("empty", "no calibration samples remain"),
    ("few", "need at least"),
    ("excitation", "insufficient hand-eye excitation"),
    ("nonfinite", "finite .* transform"),
    ("position", "hand-eye quality failed: .*worst sample ID"),
    ("rotation", "hand-eye quality failed: .*worst sample ID"),
])
def test_failed_solve_leaves_calibration_untouched(capture, tmp_path, failure, error, output_exists):
    source, _, eef, markers = capture
    excluded = ()
    if failure == "empty":
        excluded = (*SAMPLE_IDS, 999)
    elif failure == "few":
        excluded = SAMPLE_IDS[1:]
    else:
        if failure == "excitation":
            eef[:, :3, :3] = np.eye(3)
        elif failure == "nonfinite":
            markers[0, 0, 3] = np.nan
        elif failure == "position":
            markers[0, :3, 3] += [.2, -.1, .1]
        elif failure == "rotation":
            markers[0, :3, :3] = markers[0, :3, :3] @ Rotation.from_euler("x", 45, degrees=True).as_matrix()
        for index, sample_id in enumerate(SAMPLE_IDS):
            np.savez(source / "samples" / f"{index:06d}.npz", sample_id=sample_id,
                     T_base_from_eef=eef[index], T_camera_from_marker=markers[index])
    output = tmp_path / "cameras.json"
    if output_exists:
        output.write_text(json.dumps({SERIAL: {"T_base_from_camera": np.eye(4).tolist()}}))
    before = output.read_bytes() if output_exists else None

    with pytest.raises(ValueError, match=error):
        solve_samples(source, output, exclude=excluded)

    assert (output.read_bytes() if output.exists() else None) == before
    assert not output.with_suffix(".result.json").exists()


@pytest.mark.parametrize("target", ["capture.json", "samples/000000.npz"])
def test_output_cannot_overwrite_capture_inputs(capture, tmp_path, target):
    source, _, _, _ = capture
    target = source / target
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    before = target.read_bytes()

    for output in (target, alias):
        with pytest.raises(ValueError, match="must not overwrite capture inputs"):
            solve_samples(source, output)
        assert target.read_bytes() == before


def test_cli_solve_does_not_import_collection_modules(capture, tmp_path):
    source, expected, _, _ = capture
    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_camera.py"
    output = tmp_path / "cli.json"
    code = """
import runpy
import sys

for name in (
    "dexmani_real.utils.camera_calibrate.aruco",
    "dexmani_real.geometry.kinematics",
    "dexmani_real.sensor.realsense",
    "dexmani_real.robot.xarm7.driver",
):
    sys.modules[name] = None
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code, str(script), "solve", str(source),
         "--exclude", "999", "--output", str(output)],
        capture_output=True, text=True, check=True,
    )
    np.testing.assert_allclose(load_camera_extrinsics(SERIAL, output), expected, atol=1e-10)
    assert f"samples={len(SAMPLE_IDS)}" in completed.stdout


@pytest.mark.parametrize("camera", [{}, {"serial": None}])
def test_solve_requires_capture_serial(capture, tmp_path, camera):
    source, _, _, _ = capture
    (source / "capture.json").write_text(json.dumps({"camera": camera}))
    output = tmp_path / "cameras.json"
    save_camera_calibration(np.eye(4), SERIAL, output)
    before = output.read_bytes()

    with pytest.raises((KeyError, ValueError), match="serial"):
        solve_samples(source, output)

    assert output.read_bytes() == before


def test_calibration_still_matches_real_serial(tmp_path):
    path = tmp_path / "cameras.json"
    transform = np.eye(4)
    transform[0, 3] = .25
    save_camera_calibration(transform, "camera-A", path)
    np.testing.assert_array_equal(load_camera_extrinsics("camera-A", path), transform)
    with pytest.raises(KeyError):
        load_camera_extrinsics("camera-B", path)
    for serial in (None, "", "  "):
        for loader in (load_camera_extrinsics, load_optional_camera_extrinsics):
            with pytest.raises(ValueError, match="serial"):
                loader(serial, path)
        with pytest.raises(ValueError, match="serial"):
            save_camera_calibration(transform, serial, path)


def test_optional_extrinsics_only_omit_missing_default(tmp_path, monkeypatch):
    path = tmp_path / "cameras.json"
    monkeypatch.setattr(extrinsics, "CAMERAS_PATH", path)
    assert load_optional_camera_extrinsics(SERIAL) is None
    with pytest.raises(FileNotFoundError):
        load_optional_camera_extrinsics(SERIAL, path)
    save_camera_calibration(np.eye(4), "other-camera", path)
    assert load_optional_camera_extrinsics(SERIAL) is None
    with pytest.raises(KeyError):
        load_optional_camera_extrinsics(SERIAL, path)
    path.write_text(json.dumps({SERIAL: {"T_base_from_camera": np.zeros((4, 4)).tolist()}}))
    for selected_path in (None, path):
        with pytest.raises(ValueError, match="homogeneous"):
            load_optional_camera_extrinsics(SERIAL, selected_path)


@pytest.mark.parametrize("contents", ["[]", "null"])
def test_invalid_calibration_mapping_is_rejected_without_overwrite(tmp_path, contents):
    path = tmp_path / "cameras.json"
    path.write_text(contents)
    for loader in (load_camera_extrinsics, load_optional_camera_extrinsics):
        with pytest.raises(TypeError, match="keyed by serial"):
            loader(SERIAL, path)
    with pytest.raises(TypeError, match="keyed by serial"):
        save_camera_calibration(np.eye(4), SERIAL, path)
    assert path.read_text() == contents


@pytest.mark.parametrize("model", ["none", "brown_conrady"])
def test_sampling_pnp_uses_camera_intrinsics_and_distortion(model):
    from dexmani_real.utils.camera_calibrate.aruco import ArucoSampler

    coefficients = (-.25, .1, .002, -.001, .01)
    intrinsics = CameraIntrinsics(640, 480, 610., 620., 320., 240., model, coefficients)
    sampler = ArucoSampler(intrinsics)
    rvec, tvec = np.array([.2, -.1, .05]), np.array([.15, -.08, .5])
    distortion = np.zeros(5) if model == "none" else np.asarray(coefficients)
    corners = cv2.projectPoints(sampler.corners, rvec, tvec, intrinsics.matrix(), distortion)[0]

    estimated_rvec, estimated_tvec = sampler.estimate_pose(corners.reshape(4, 2))

    np.testing.assert_allclose(estimated_rvec, rvec, atol=1e-5)
    np.testing.assert_allclose(estimated_tvec, tvec, atol=1e-6)


def test_collect_records_actual_camera_without_requested_config(tmp_path, monkeypatch):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.geometry import kinematics
    from dexmani_real.sensor import realsense
    from dexmani_real.utils.camera_calibrate.calibrate import collect_samples

    intrinsics = CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5)
    camera = Mock(active_serial="actual-camera", color_intrinsics=intrinsics)
    camera_factory = Mock(return_value=camera)
    monkeypatch.setattr(realsense, "RealSenseCamera", camera_factory)
    monkeypatch.setattr(kinematics, "ArmFK", Mock())
    monkeypatch.setattr("builtins.input", lambda _: "q")
    runtime = RealConfig()
    output = tmp_path / "capture"

    assert collect_samples(runtime, output) == 0

    metadata = json.loads((output / "capture.json").read_text())
    assert set(metadata) == {"camera", "marker", "criteria"}
    assert metadata["camera"] == {"serial": "actual-camera", "color": intrinsics.to_dict()}
    assert metadata["marker"]["size_m"] > 0
    assert metadata["criteria"]["camera_max_age_s"] == runtime.camera.max_frame_age_s
    camera_factory.assert_called_once_with(
        runtime.camera, depth_enabled=False, auto_exposure_priority=None,
    )
    camera.disconnect.assert_called_once()
