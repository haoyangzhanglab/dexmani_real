"""录制、时效与恢复测试共用的纯内存输入。"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now_ns=1_000_000_000)
    monkeypatch.setattr("time.monotonic_ns", lambda: value.now_ns)
    return value


@pytest.fixture
def runtime():
    config = RealConfig()
    return replace(config, execution=replace(
        config.execution, control_dt_s=0.05, action_steps=2, prefetch_steps=1,
    ))


@pytest.fixture
def camera_snapshot():
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot

    return CameraSnapshot(
        "test-camera",
        CameraIntrinsics(
            width=4, height=4, fx=10, fy=10, ppx=2, ppy=2,
            distortion_model="none", distortion_coeffs=(0, 0, 0, 0, 0),
        ),
        0.001, np.eye(4),
    )
