"""VR 运行朝向读取；不连接设备或改变标定策略。"""

import json

import numpy as np
import pytest

from dexmani_real.utils.vr_heading import load_vr_transform


@pytest.mark.parametrize('angle', [-90, 270, 630, '270', 0, 360, -360])
def test_finite_heading_keeps_rotation_direction_and_input(tmp_path, angle):
    path = tmp_path / 'vr.json'
    contents = json.dumps({'theta_deg': angle})
    path.write_text(contents)
    expected = (
        np.eye(3) if float(angle) in (0, 360, -360)
        else np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    )
    np.testing.assert_allclose(load_vr_transform(path), expected, rtol=0, atol=1e-14)
    assert path.read_text() == contents


@pytest.mark.parametrize('angle', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_heading_is_rejected(tmp_path, angle):
    path = tmp_path / 'vr.json'
    path.write_text(json.dumps({'theta_deg': angle}))
    with pytest.raises(ValueError, match='VR heading must be finite'):
        load_vr_transform(path)


@pytest.mark.parametrize('contents,error', [
    (None, FileNotFoundError), ('{}', KeyError), ('{', json.JSONDecodeError),
    ('{"theta_deg": "invalid"}', ValueError),
])
def test_heading_input_errors_propagate(tmp_path, contents, error):
    path = tmp_path / 'vr.json'
    if contents is not None:
        path.write_text(contents)
    with pytest.raises(error):
        load_vr_transform(path)


@pytest.mark.parametrize('argument', [None, '', 'missing.json'])
def test_teleop_cli_only_defaults_an_omitted_vr_path(monkeypatch, argument):
    from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
    from dexmani_real.teleop import run
    from scripts.collect_teleop import main

    paths = []
    def run_teleop(runtime, **kwargs):
        paths.append(kwargs['vr_transform_path'])
        return 0
    monkeypatch.setattr(run, 'run_teleop', run_teleop)
    arguments = [] if argument is None else ['--vr-transform', argument]
    assert main(['--no-record', *arguments]) == 0
    assert paths == [VR_TRANSFORM_PATH if argument is None else argument]
