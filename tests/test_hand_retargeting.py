"""Native offline method/reset checks, separate from source-sample caching."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.teleop.control.hand_retargeting import (
    HandRetargetObservationCache,
    compute_hand_command,
)


@pytest.mark.parametrize("method", ["tag", "dexpilot"])
def test_native_retarget_and_reset_preserve_sdk_output(method):
    from dexmani_real.teleop.config import TeleopConfig
    from dexmani_real.teleop.session import _build_hand_retargeter

    cfg = ExperimentConfig()
    cfg = replace(cfg, policy=replace(cfg.policy, hand_retargeting_type=method))
    retargeter = _build_hand_retargeter(TeleopConfig(cfg, task_label="synthetic"))
    landmarks = np.zeros((21, 3))
    for finger, (x, y) in enumerate(
        ((-0.045, 0.02), (-0.022, 0.055), (0, 0.065), (0.02, 0.06), (0.04, 0.048))
    ):
        for joint in range(4):
            landmarks[1 + finger * 4 + joint] = (x, y + joint * 0.02, 0)
    home = np.deg2rad(cfg.hand.home_qpos_deg)
    retargeter.reset(home)
    first = retargeter.retarget(landmarks)
    assert first is not None and first.shape == (12,) and np.isfinite(first).all()
    retargeter.retarget(landmarks * 1.1)
    retargeter.reset(home)
    np.testing.assert_allclose(retargeter.retarget(landmarks), first, rtol=0, atol=1e-7)
    assert retargeter.retarget(np.zeros((21, 3))) is None


@pytest.mark.parametrize("success", [True, False])
def test_same_vr_source_caches_success_and_failure(success):
    calls = []

    def solve(landmarks):
        calls.append(1)
        return np.arange(12, dtype=float) if success else None

    retargeter = NS(retarget=solve)
    cache = HandRetargetObservationCache()
    frame = dict(ring_sequence=4, landmarks=np.zeros((21, 3)))
    first = compute_hand_command(retargeter, frame, cache)
    if first is not None:
        first[:] = -1
    second = compute_hand_command(retargeter, frame, cache)
    assert calls == [1]
    if success:
        np.testing.assert_array_equal(second, np.arange(12))
    else:
        assert second is None
    cache.reset()
    compute_hand_command(retargeter, frame, cache)
    assert calls == [1, 1]


def test_geometry_import_has_no_optimizer_backend():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'nlopt', 'pinocchio', 'dex_retargeting'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from dexmani_real.teleop.retargeting import retargeter
assert callable(retargeter.validate_landmarks)
""",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
