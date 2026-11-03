"""相机标定的数学、采样及文件边界；不连接硬件。"""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.config.real import RealConfig
from dexmani_real.geometry.camera import CameraIntrinsics, camera_parameters_cv
from dexmani_real.utils.camera_calibrate import aruco, calibrate
from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
from dexmani_real.utils.camera_calibrate.hand_eye import solve_hand_eye


@pytest.mark.parametrize('contents,error', [
    (None, FileNotFoundError),
    ('{}', KeyError),
    ('{"other": {"T_base_from_camera": []}}', KeyError),
])
def test_optional_default_extrinsics_only_allow_missing_file_or_serial(monkeypatch, tmp_path, contents, error):
    from dexmani_real.utils.camera_calibrate import extrinsics

    path = tmp_path / 'cameras.json'
    monkeypatch.setattr(extrinsics, 'CAMERAS_PATH', path)
    if contents is not None:
        path.write_text(contents)
    assert extrinsics.load_optional_camera_extrinsics('actual-camera') is None
    with pytest.raises(error):
        extrinsics.load_camera_extrinsics('actual-camera')
    # 显式选择同一文件时不能把缺失当作可选。
    with pytest.raises(error):
        extrinsics.load_optional_camera_extrinsics('actual-camera', path)


@pytest.mark.parametrize('serial', ['', ' ', None])
def test_optional_extrinsics_reject_invalid_serial(monkeypatch, tmp_path, serial):
    from dexmani_real.utils.camera_calibrate import extrinsics

    path = tmp_path / 'cameras.json'
    path.write_text('{}')
    monkeypatch.setattr(extrinsics, 'CAMERAS_PATH', path)
    with pytest.raises(ValueError, match='serial must be non-empty'):
        extrinsics.load_optional_camera_extrinsics(serial)


@pytest.mark.parametrize('contents,error', [
    ('{', json.JSONDecodeError),
    ('[]', ValueError),
    ('{"actual-camera": {}}', KeyError),
    ('{"actual-camera": null}', TypeError),
    ('{"actual-camera": {"T_base_from_camera": []}}', ValueError),
    (json.dumps({'actual-camera': {'T_base_from_camera': np.zeros((4, 4)).tolist()}}), ValueError),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_optional_extrinsics_preserve_invalid_data_errors(monkeypatch, tmp_path, contents, error, explicit):
    from dexmani_real.utils.camera_calibrate import extrinsics

    path = tmp_path / 'cameras.json'
    path.write_text(contents)
    monkeypatch.setattr(extrinsics, 'CAMERAS_PATH', path)
    with pytest.raises(error):
        extrinsics.load_optional_camera_extrinsics('actual-camera', path if explicit else None)


@pytest.fixture
def poses():
    rng = np.random.default_rng(42)
    def transform(r, t):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_rotvec(r).as_matrix()
        result[:3, 3] = t
        return result
    camera = transform([.6, -.4, .3], [.9, .2, .6])
    mount = transform([-.3, .2, .1], [.03, -.06, .12])
    eef = np.array([transform(rng.uniform(-1, 1, 3), rng.uniform(-.4, .4, 3)) for _ in range(16)])
    return camera, mount, eef, np.linalg.inv(camera) @ eef @ mount


@pytest.fixture
def intrinsics():
    return CameraIntrinsics(640, 480, 600., 600., 320., 240., 'brown_conrady',
                            (.14, -.47, .0006, -.0003, .42))


def test_eye_to_hand_direction(poses, intrinsics):
    camera, mount, eef, markers = poses
    result = solve_hand_eye(eef, markers)
    np.testing.assert_allclose(result.T_base_from_camera, camera, atol=1e-10)
    np.testing.assert_allclose(eef @ mount, result.T_base_from_camera @ markers, atol=1e-10)
    assert result.quality_ok
    external = camera_parameters_cv(intrinsics, result.T_base_from_camera)['camera_extrinsic']
    np.testing.assert_allclose(external.reshape(3, 4), np.linalg.inv(camera)[:3], atol=1e-7)


@pytest.mark.parametrize('motion', ['none', 'small', 'single_axis', 'too_few'])
def test_degenerate_motion(poses, motion):
    _, _, eef, markers = poses
    if motion == 'too_few':
        eef, markers = eef[:9], markers[:9]
    elif motion == 'single_axis':
        eef[:, :3, :3] = Rotation.from_rotvec(np.linspace(-1, 1, len(eef))[:, None] * [0, 0, 1]).as_matrix()
    elif motion == 'small':
        eef[:, :3, :3] = Rotation.from_rotvec(np.random.default_rng(0).normal(0, .001, (len(eef), 3))).as_matrix()
    else:
        eef[:, :3, :3] = np.eye(3)
    with pytest.raises(ValueError, match='excitation|at least'):
        solve_hand_eye(eef, markers)


@pytest.fixture
def sampler(intrinsics):
    return aruco.ArucoSampler(intrinsics)


@pytest.fixture
def observation(sampler):
    r = Rotation.from_euler('xyz', [155, 8, 12], degrees=True).as_rotvec()
    t = np.array([.02, -.01, .6])
    corners = cv2.projectPoints(sampler.corners, r, t, sampler.intrinsics, sampler.distortion)[0].reshape(4, 2)
    return corners, (r, t)


def test_pnp_noise_and_bad_corner(sampler, observation):
    corners, (r, t) = observation
    rng = np.random.default_rng(1)
    for _ in range(20):
        estimated_r, estimated_t = sampler.estimate_pose(corners + rng.normal(0, .15, (4, 2)))
        assert np.linalg.norm(estimated_t - t) < .005
        assert (Rotation.from_rotvec(estimated_r).inv() * Rotation.from_rotvec(r)).magnitude() < np.deg2rad(3)
    corners[0] += [10, -10]
    assert sampler.estimate_pose(corners) is None


@pytest.mark.parametrize('model', ['inverse_brown_conrady', 'modified_brown_conrady', 'ftheta', 'kannala_brandt4'])
def test_unsupported_distortion(intrinsics, model):
    from dataclasses import replace
    with pytest.raises(ValueError, match='distortion model'):
        aruco.ArucoSampler(replace(intrinsics, distortion_model='distortion.' + model))


@pytest.mark.parametrize('model', ['none', 'brown_conrady', 'distortion.none', 'distortion.brown_conrady'])
def test_supported_distortion(intrinsics, model):
    from dataclasses import replace
    sampler = aruco.ArucoSampler(replace(intrinsics, distortion_model=model))
    np.testing.assert_array_equal(sampler.distortion, np.zeros(5) if model.endswith('none') else intrinsics.distortion_coeffs)


@pytest.fixture
def capture(monkeypatch, sampler, observation):
    clock = [1_000_000_000]
    numbers = iter([1, 1, 2, 3, 4, 5])  # 新的 depth/时间不能让重复 Color 成为新样本。
    rgb = np.zeros((480, 640, 3), np.uint8)
    def camera():
        clock[0] += 1_000_000
        return SimpleNamespace(rgb=rgb, timestamp_ns=clock[0], color_frame_number=next(numbers))
    def arm():
        return {'qpos': np.zeros(7), 'qvel': np.zeros(7), 'timestamp_ns': clock[0]}
    monkeypatch.setattr(aruco.time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(aruco.time, 'sleep', lambda _: None)
    monkeypatch.setattr(sampler, 'detect', lambda _: observation)
    config = RealConfig()
    def run(read_camera=camera, read_arm=arm):
        return sampler.capture(read_camera, read_arm, camera_config=config.camera, arm_config=config.arm)
    return run


@pytest.mark.parametrize('valid_count', [2, 3, 4, 5])
def test_capture_detection_count_and_schema(monkeypatch, sampler, observation, capture, tmp_path, valid_count):
    detections = iter([observation] * valid_count + [(observation[0], None)] * (5 - valid_count))
    monkeypatch.setattr(sampler, 'detect', lambda _: next(detections))
    if valid_count < 4:
        with pytest.raises(ValueError, match='valid PnP'):
            capture()
        return
    sample = capture()
    assert sample['rgb'].shape == (5, 480, 640, 3)
    assert sample['marker_detected'].sum() == valid_count
    np.testing.assert_array_equal(sample['color_frame_number'], [1, 2, 3, 4, 5])
    path = tmp_path / 'sample.npz'
    calibrate.save_sample(path, 7, sample, np.eye(4))
    with np.load(path, allow_pickle=False) as data:
        expected = {'sample_id', 'T_base_from_eef', 'T_camera_from_marker', 'rgb',
                    'camera_timestamp_ns', 'color_frame_number', 'marker_corners_xy',
                    'marker_rvec', 'marker_tvec', 'marker_detected', 'arm_qpos', 'arm_qvel',
                    'arm_timestamp_ns', 'arm_before_qpos', 'arm_before_qvel', 'arm_before_timestamp_ns',
                    'arm_after_qpos', 'arm_after_qvel', 'arm_after_timestamp_ns'}
        assert set(data.files) == expected
        for name, value in sample.items():
            np.testing.assert_array_equal(data[name], value)
        assert data['sample_id'] == 7
        assert data['camera_timestamp_ns'].dtype == np.int64
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        calibrate.save_sample(path, 8, sample, np.eye(4))
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('fault', ['flip', 'translation'])
def test_pose_inconsistency(monkeypatch, sampler, observation, capture, fault):
    corners, (r, t) = observation
    bad_r = (Rotation.from_rotvec(r) * Rotation.from_euler('x', 40, degrees=True)).as_rotvec()
    bad = (bad_r, t) if fault == 'flip' else (r, t + [.02, 0, 0])
    detections = iter([observation] * 4 + [(corners, bad)])
    monkeypatch.setattr(sampler, 'detect', lambda _: next(detections))
    with pytest.raises(ValueError, match='inconsistent marker'):
        capture()


@pytest.mark.parametrize('fault', ['velocity', 'drift', 'timeout'])
def test_capture_motion_and_timeout(monkeypatch, capture, fault):
    if fault == 'timeout':
        ticks = iter(range(100))
        monkeypatch.setattr(aruco.time, 'monotonic', lambda: next(ticks))
        with pytest.raises(TimeoutError, match='fresh'):
            capture(read_camera=lambda: None)
    else:
        states = [0, 0, .1, 0, 0, 0, 0]
        def read_arm():
            value = states.pop(0)
            return {'qpos': np.full(7, value if fault == 'drift' else 0),
                        'qvel': np.full(7, value if fault == 'velocity' else 0), 'timestamp_ns': 1}
        with pytest.raises(ValueError, match='moving|drifted'):
            capture(read_arm=read_arm)


def test_partial_sample_write(monkeypatch, tmp_path):
    def fail(stream, **arrays):
        stream.write(b'incomplete')
        raise OSError('disk full')
    monkeypatch.setattr(calibrate.np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='disk full'):
        calibrate.save_sample(tmp_path / 'sample.npz', 1, {}, np.eye(4))
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def saved_samples(tmp_path, poses, intrinsics):
    source = tmp_path / 'capture'
    (source / 'samples').mkdir(parents=True)
    (source / 'capture.json').write_text(json.dumps({'camera': {'serial': 'test', 'color': intrinsics.to_dict()}}))
    for i, (eef, marker) in enumerate(zip(poses[2], poses[3])):
        np.savez(source / 'samples' / f'{i:06d}.npz', sample_id=10 + 2 * i,
                 T_base_from_eef=eef, T_camera_from_marker=marker)
    return source


def test_saved_poses_and_exclude(saved_samples, tmp_path, poses):
    output = tmp_path / 'cameras.json'
    output.write_text(json.dumps({'other': {'T_base_from_camera': np.eye(4).tolist()}}))
    result = calibrate.solve_samples(saved_samples, output, exclude=[12])
    assert result.method == 'HORAUD'
    np.testing.assert_allclose(load_camera_extrinsics('test', output), poses[0], atol=1e-10)
    np.testing.assert_array_equal(load_camera_extrinsics('other', output), np.eye(4))
    report = json.loads(output.with_suffix('.result.json').read_text())
    assert report['excluded_sample_ids'] == [12]
    assert report['used_sample_ids'] == [i for i in range(10, 42, 2) if i != 12]
    assert report['worst_sample_id'] == report['used_sample_ids'][result.worst_sample]
    before = output.read_bytes()
    with pytest.raises(ValueError, match='unknown excluded'):
        calibrate.solve_samples(saved_samples, output, exclude=[1])
    assert output.read_bytes() == before


@pytest.mark.parametrize('fault', ['bad_pose', 'quality', 'report_alias', 'input_alias'])
def test_failed_solve_preserves_files(saved_samples, tmp_path, poses, fault):
    output = tmp_path / 'cameras.json'
    output.write_text(json.dumps({'test': {'T_base_from_camera': poses[0].tolist()}}))
    if fault == 'report_alias':
        output.with_suffix('.result.json').symlink_to(output)
    if fault in {'bad_pose', 'quality'}:
        marker = poses[3][0].copy()
        if fault == 'bad_pose':
            marker[0, 0] = np.nan
        else:
            marker[:3, 3] += [.2, -.2, .1]
        np.savez(saved_samples / 'samples' / '000000.npz', sample_id=10,
                 T_base_from_eef=poses[2][0], T_camera_from_marker=marker)
    if fault == 'input_alias':
        output = saved_samples / 'capture.json'
    before = {p: p.read_bytes() for p in [output, *saved_samples.rglob('*.npz')]}
    with pytest.raises(ValueError):
        calibrate.solve_samples(saved_samples, output)
    assert all(p.read_bytes() == data for p, data in before.items())
    if fault == 'quality':
        report = json.loads(output.with_suffix('.result.json').read_text())
        assert not report['quality_ok']
        assert report['worst_sample_id'] == 10


@pytest.mark.parametrize("close_failure", [False, True])
def test_collect_rejection_retry_and_close(monkeypatch, tmp_path, intrinsics, capsys, close_failure):
    events = []
    class Camera:
        active_serial = 'test'
        color_intrinsics = intrinsics
        depth_scale_m = .001
        def __init__(self, *a, **kw): pass
        def connect(self): events.append('camera connect')
        def disconnect(self): events.append('camera close')
        def read(self): raise AssertionError('capture is substituted')
    class Arm:
        def __init__(self, *a): pass
        def connect(self, **kw): events.append('arm connect')
        def close(self):
            events.append('arm close')
            if close_failure:
                raise OSError('disconnect failed')
    outcomes = iter([ValueError('unstable pose'), {'arm_after_qpos': np.zeros(7)}])
    def capture(*a, **kw):
        result = next(outcomes)
        if isinstance(result, Exception):
            raise result
        return result
    commands = iter(['x', '', '', 'Q'])
    monkeypatch.setattr('builtins.input', lambda _: next(commands))
    monkeypatch.setattr('dexmani_real.sensor.realsense.RealSenseCamera', Camera)
    monkeypatch.setattr('dexmani_real.robot.xarm7.driver.XArm7', Arm)
    monkeypatch.setattr('dexmani_real.geometry.kinematics.ArmFK', lambda: SimpleNamespace(forward=lambda q: np.eye(4)))
    monkeypatch.setattr(aruco.ArucoSampler, 'capture', capture)
    monkeypatch.setattr(calibrate, 'save_sample', lambda *a: events.append('save'))
    if close_failure:
        with pytest.raises(OSError, match='disconnect failed'):
            calibrate.collect_samples(RealConfig(), tmp_path / 'run')
        assert events == ['camera connect', 'arm connect', 'arm close', 'camera close']
        return
    assert calibrate.collect_samples(RealConfig(), tmp_path / 'run') == 1
    assert events == ['camera connect', 'arm connect', 'arm close', 'arm connect', 'arm close', 'save', 'camera close']
    assert '采样拒绝：unstable pose' in capsys.readouterr().out


def test_cli_defaults_and_exclude(monkeypatch):
    from pathlib import Path

    from dexmani_real.config.calibration_data import CAMERAS_PATH
    from scripts.calibrate_camera import main
    calls = []
    monkeypatch.setattr(calibrate, 'solve_samples', lambda *a, **kw: calls.append((a, kw)))
    assert main(['solve', 'old', '--exclude', '12', '--exclude', '18']) == 0
    assert calls == [((Path('old'), CAMERAS_PATH), {'exclude': [12, 18]})]


@pytest.mark.parametrize('fault', ['behind', 'nonfinite', 'failed'])
def test_invalid_pnp_result(monkeypatch, sampler, observation, fault):
    r, t = observation[1]
    t = t.copy()
    if fault == 'behind':
        t[2] = -1
    if fault == 'nonfinite':
        t[2] = np.nan
    monkeypatch.setattr(cv2, 'solvePnP', lambda *a, **kw: (fault != 'failed', r[:, None], t[:, None]))
    assert sampler.estimate_pose(observation[0]) is None


def test_actual_aruco_detection(sampler):
    rgb = np.full((480, 640, 3), 255, np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(aruco.ARUCO_DICT), aruco.MARKER_ID, 160)
    rgb[160:320, 240:400] = marker[..., None]
    corners, pose = sampler.detect(rgb)
    assert corners.shape == (4, 2)
    assert pose is not None
    missing, pose = sampler.detect(np.full_like(rgb, 255))
    assert np.isnan(missing).all() and pose is None


def test_extrinsic_loader_stays_lightweight(tmp_path):
    import subprocess
    import sys
    path = tmp_path / 'cameras.json'
    path.write_text(json.dumps({'test': {'T_base_from_camera': np.eye(4).tolist()}}))
    code = """
import sys
from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
load_camera_extrinsics('test', sys.argv[1])
assert 'cv2' not in sys.modules
"""
    subprocess.run([sys.executable, '-B', '-c', code, str(path)], check=True)



def test_noisy_horaud_leave_one_out(poses):
    camera, _, eef, markers = poses
    rng = np.random.default_rng(4)
    markers[:, :3, 3] += rng.normal(0, .0003, (len(markers), 3))
    markers[:, :3, :3] = markers[:, :3, :3] @ Rotation.from_rotvec(
        rng.normal(0, np.deg2rad(.03), (len(markers), 3))).as_matrix()
    for omitted in range(len(eef)):
        keep = np.arange(len(eef)) != omitted
        result = solve_hand_eye(eef[keep], markers[keep])
        assert result.quality_ok
        assert np.linalg.norm(result.T_base_from_camera[:3, 3] - camera[:3, 3]) < .002
        error = Rotation.from_matrix(result.T_base_from_camera[:3, :3].T @ camera[:3, :3])
        assert np.rad2deg(error.magnitude()) < .2


@pytest.mark.parametrize('stamp', [1_100_000_000, 3_000_000_000])
def test_stale_or_future_frame_is_rejected(monkeypatch, capture, sampler, stamp):
    from itertools import chain, repeat
    stamps = chain([1_000_000_000], repeat(2_000_000_000))
    ticks = iter(range(10))
    monkeypatch.setattr(aruco.time, 'monotonic_ns', lambda: next(stamps))
    monkeypatch.setattr(aruco.time, 'monotonic', lambda: next(ticks))
    def detect(_):
        pytest.fail('invalid timestamp reached marker detection')
    monkeypatch.setattr(sampler, 'detect', detect)
    with pytest.raises(TimeoutError, match='fresh'):
        capture(read_camera=lambda: SimpleNamespace(timestamp_ns=stamp, color_frame_number=1))


def test_slow_arm_feedback_rejected(monkeypatch):
    ticks = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(calibrate.time, 'monotonic_ns', lambda: next(ticks))
    arm = SimpleNamespace(read=lambda: (np.zeros(7), np.zeros(7)))
    with pytest.raises(TimeoutError, match='freshness budget'):
        calibrate.read_arm_feedback(arm, .1)


def test_formal_json_publish_failure_preserves_calibration(monkeypatch, saved_samples, tmp_path):
    from dexmani_real.utils import io
    output = tmp_path / 'cameras.json'
    original = json.dumps({'test': {'T_base_from_camera': np.eye(4).tolist()}})
    output.write_text(original)
    replace = io.os.replace
    def fail_formal(source, target):
        if target == output:
            raise OSError('publish failed')
        return replace(source, target)
    monkeypatch.setattr(io.os, 'replace', fail_formal)
    with pytest.raises(OSError, match='publish failed'):
        calibrate.solve_samples(saved_samples, output)
    assert output.read_text() == original
    assert json.loads(output.with_suffix('.result.json').read_text())['quality_ok']
    assert not list(tmp_path.glob('.*.tmp-*'))
