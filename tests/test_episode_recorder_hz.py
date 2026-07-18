"""EpisodeRecorder control_hz grid tests (synthetic feed, no hardware)."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dexmani_real.recording.episode_recorder import EpisodeRecorder

HZ = 16.0
DT = 1.0 / HZ


def _fake_state(ts: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=ts,
        arm_qpos=np.zeros(7),
        arm_qvel=np.zeros(7),
        arm_tau=np.zeros(7),
        eef_pos=np.zeros(3),
        eef_rot6d=np.array([1, 0, 0, 0, 1, 0], dtype=np.float64),
        hand_qpos=np.zeros(12),
        fingertip_pos=np.zeros((5, 3)),
        hand_tactile_sum=np.zeros((5, 3)),
    )


def _fake_action() -> SimpleNamespace:
    return SimpleNamespace(
        arm_qpos_cmd=np.zeros(7),
        hand_qpos_cmd=np.zeros(12),
        target_eef_pos=None,
        target_eef_rot6d=None,
    )


def _fake_vr() -> dict:
    return {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
    }


def _record_episode(tmp_path, n_frames: int, skip: int = 0) -> str:
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=int(round(1.0 * HZ)))
    assert rec.start_episode(task_label="test", skip_initial_frames=skip)
    rng = np.random.default_rng(0)
    t0 = 1000.0
    accepted = 0
    for k in range(n_frames):
        ts = t0 + k * DT + float(rng.uniform(-0.005, 0.005))  # jitter << dt/2
        ok = rec.add_frame(_fake_state(ts), _fake_action(), _fake_vr(), signals={"ik_ok": True})
        if k < skip:
            assert not ok, "skip_initial_frames must reject the first frames"
        else:
            assert ok
            accepted += 1
    path = rec.stop_episode(success=True)
    rec.join_stop()
    assert path is not None
    return path


def test_grid_dt_and_meta(tmp_path):
    path = _record_episode(tmp_path, n_frames=40)
    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        assert meta["schema_version"] == 7
        assert meta["control_hz"] == pytest.approx(HZ)
        assert meta["num_frames"] == 40
        assert bool(meta["min_frames_met"])  # 40 >= 16
        ts = f["timestamp"][:]
        assert len(ts) == 40
        # One frame per slot (no dup-drop/forward-fill under ±5ms jitter);
        # stored timestamps are raw sample times, so steps = dt ± jitter.
        steps = np.diff(ts)
        assert np.all(steps > 0)
        assert np.allclose(steps, DT, atol=0.011)  # |jitter_k - jitter_{k+1}| <= 10ms
        slots = np.round((ts - ts[0]) / DT).astype(int)
        assert np.array_equal(slots, np.arange(40))
        assert f["arm_qpos"].shape == (40, 7)
        assert f["action_hand_joint"].shape == (40, 12)


def test_skip_initial_frames_no_gap(tmp_path):
    """Skipped frames re-anchor the grid — no forward-filled hole at the start."""
    path = _record_episode(tmp_path, n_frames=20, skip=3)
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["num_frames"] == 17
        ts = f["timestamp"][:]
        assert np.allclose(np.diff(ts), DT, atol=0.011)


def test_min_frames_not_met(tmp_path):
    path = _record_episode(tmp_path, n_frames=10)  # < 16
    with h5py.File(path, "r") as f:
        assert not bool(f["meta"].attrs["min_frames_met"])
