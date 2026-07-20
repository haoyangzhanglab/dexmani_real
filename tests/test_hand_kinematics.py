"""Regression tests for HandKinematics fail-safe construction (bug #2 in 077ce36).

pinocchio.buildModelFromUrdf raises ValueError for a nonexistent/invalid URDF,
but __init__ only caught (ImportError, RuntimeError) — the escaped ValueError
crashed RobotInterface construction instead of degrading to _ready=False.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from dexmani_real.robot.hand_kinematics import HandKinematics


def _fake_pinocchio(exc: Exception) -> types.ModuleType:
    mod = types.ModuleType("pinocchio")

    def buildModelFromUrdf(path):
        raise exc

    mod.buildModelFromUrdf = buildModelFromUrdf
    return mod


@pytest.mark.parametrize("exc", [ValueError("bad urdf"), RuntimeError("bad urdf")], ids=["ValueError", "RuntimeError"])
def test_build_failure_degrades_not_crashes(monkeypatch, exc):
    """A raising buildModelFromUrdf must leave the object inert, not raise."""
    monkeypatch.setitem(sys.modules, "pinocchio", _fake_pinocchio(exc))
    hk = HandKinematics("/any/hand.urdf")
    assert hk.is_ready() is False
    tips = hk.compute_tip_positions_in_handbase(np.zeros(12))
    assert tips.shape == (5, 3)
    assert np.isnan(tips).all()


def test_real_pinocchio_bad_path_degrades():
    """Integration: real pinocchio raises ValueError on a missing URDF file."""
    pytest.importorskip("pinocchio")
    hk = HandKinematics("/nonexistent/hand.urdf")
    assert hk.is_ready() is False
