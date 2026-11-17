"""离线 hand-eye 数值求解、候选保存和显式输出路径。"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.utils.camera_calibrate.calibrate import save_sample, solve_samples
from dexmani_real.utils.camera_calibrate.hand_eye import MIN_SAMPLES, solve_hand_eye


def calibration_poses():
    rng = np.random.default_rng(17)
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("xyz", [.3, -.6, .4]).as_matrix()
    camera[:3, 3] = [.4, -.2, .8]
    marker_in_eef = np.eye(4)
    marker_in_eef[:3, :3] = Rotation.from_euler("xyz", [-.2, .1, .3]).as_matrix()
    marker_in_eef[:3, 3] = [.02, -.04, .1]
    eef = np.tile(np.eye(4), (MIN_SAMPLES + 3, 1, 1))
    eef[:, :3, :3] = Rotation.from_rotvec(rng.normal(0., .5, (len(eef), 3))).as_matrix()
    eef[:, :3, 3] = rng.uniform([.3, -.2, .2], [.6, .2, .5], (len(eef), 3))
    markers = np.linalg.inv(camera) @ eef @ marker_in_eef
    return eef, markers, camera


@pytest.mark.parametrize("noisy", [False, True])
def test_solvable_candidate_is_saved_even_above_empirical_quality(tmp_path, capsys, noisy):
    eef, markers, expected = calibration_poses()
    if noisy:
        markers[:, :3, 3] += np.random.default_rng(19).normal(0., .025, (len(markers), 3))
    source = tmp_path / "capture"
    samples = source / "samples"
    samples.mkdir(parents=True)
    (source / "capture.json").write_text(json.dumps({"camera": {"serial": "test-camera"}}))
    for sample_id, (eef_pose, marker_pose) in enumerate(zip(eef, markers)):
        save_sample(samples / f"{sample_id:06d}.npz", sample_id,
                    {"T_camera_from_marker": marker_pose}, eef_pose)
    before = {path.name: path.read_bytes() for path in samples.iterdir()}
    output = tmp_path / "candidate.json"
    result = solve_samples(source, output)
    saved = np.asarray(json.loads(output.read_text())["test-camera"]["T_base_from_camera"])
    np.testing.assert_allclose(saved, result.T_base_from_camera)
    assert {path.name: path.read_bytes() for path in samples.iterdir()} == before
    log = capsys.readouterr().out
    if noisy:
        assert not result.quality_ok
        assert "警告" in log and "worst sample ID" in log
    else:
        assert result.quality_ok
        np.testing.assert_allclose(saved, expected, atol=1e-10)
    assert "sample 0: residual" in log


def test_unsolvable_samples_remain_rejected():
    eef, markers, _ = calibration_poses()
    with pytest.raises(ValueError, match="at least"):
        solve_hand_eye(eef[:MIN_SAMPLES - 1], markers[:MIN_SAMPLES - 1])
    eef[:, :3, :3] = np.eye(3)
    with pytest.raises(ValueError, match="excitation"):
        solve_hand_eye(eef, markers)


def test_solve_cli_requires_explicit_candidate_path(tmp_path):
    script = Path(__file__).parents[1] / "scripts/calibrate_camera.py"
    spec = importlib.util.spec_from_file_location("calibrate_camera_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit) as error:
        module.main(["solve", str(tmp_path)])
    assert error.value.code == 2
