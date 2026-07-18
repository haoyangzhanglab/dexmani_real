"""Hand per-send delta clip: velocity semantics, rollback pin, stub tracking.

The 16Hz migration re-semantized the hand E3 clip: entry points derive
max_delta_rad = deg2rad(HAND_MAX_QVEL_DEG_S) / CTRL_HZ (velocity semantics),
while the library default stays 0.3 rad/step (spike gate).  These are NOT the
same quantity — see the rollback note in examples/real/vr_teleop_shm.py.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.xhand import XHand, XHandConfig


def test_per_send_clip_16hz_velocity_semantics():
    clip = float(np.deg2rad(90.0)) / 16.0
    assert abs(clip - 0.098175) < 1e-5
    # Round trip: per-send clip × rate == the intended joint speed limit
    assert abs(np.degrees(clip) * 16.0 - 90.0) < 1e-9


def test_rollback_does_not_restore_library_default():
    """CTRL_HZ back to 50 gives 0.031 rad, NOT the old default 0.3 (9.6x tighter)."""
    clip_at_50 = float(np.deg2rad(90.0)) / 50.0
    assert abs(clip_at_50 - 0.0314) < 1e-3
    assert abs(clip_at_50 - XHandConfig().max_delta_rad) > 0.25  # library default unchanged: 0.3


def test_stub_send_action_tracks_request():
    """Stub mode must track the request — recorded actions would otherwise
    freeze at home_qpos (silent action-stream replacement)."""
    hand = XHand(XHandConfig())
    hand._stub_mode = True
    hand.last_qpos_cmd = np.zeros(12, dtype=np.float64)

    target = np.full(12, 0.2, dtype=np.float64)
    assert hand.send_action(target)
    assert hand.last_qpos_cmd is not None
    # Joint-limit clipped request, not the stale previous command
    expected = np.clip(target, hand.config.qpos_min, hand.config.qpos_max)
    np.testing.assert_allclose(hand.last_qpos_cmd, expected, atol=1e-12)
