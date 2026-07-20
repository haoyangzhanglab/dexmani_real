"""EpisodeRecorder opt-in sent-command stream (schema v9, arm_sent_stream).

When ``arm_sent_stream=True`` the recorder must add ``/action_arm_joint_sent(T,7)``
fed through the existing per-frame ``arm_qpos_sent`` kwarg and stamp
``schema_version=9`` + ``arm_sent_stream=True`` into /meta; with the flag off
(default) the behavior must be byte-identical v8 — no dataset, no meta attr —
even if the kwarg is passed. Harness mirrors test_episode_recorder_hz.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

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


def test_sent_stream_records_fed_values(tmp_path):
    """arm_sent_stream=True → /action_arm_joint_sent(T,7) ≡ fed values, schema 9."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4, arm_sent_stream=True)
    assert rec.start_episode(task_label="sent-stream")
    rng = np.random.default_rng(7)
    sent = rng.uniform(-1.0, 1.0, size=(30, 7))
    t0 = 5000.0
    for k in range(30):
        assert rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), arm_qpos_sent=sent[k])
    path = rec.stop_episode(success=True)
    assert rec.join_stop()

    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        assert meta["schema_version"] == 9
        assert bool(meta["arm_sent_stream"]) is True
        assert meta["num_frames"] == 30

        ds = f["action_arm_joint_sent"]
        assert ds.shape == (30, 7)
        assert ds.dtype == np.float64
        assert np.array_equal(ds[:], sent)

        # The IK-target stream is untouched and remains a SEPARATE dataset
        # (action_arm_joint carries the _fake_action zeros, not the sent values).
        assert f["action_arm_joint"].shape == (30, 7)
        assert not np.array_equal(f["action_arm_joint"][:], sent)

        # Index-aligned with the rest of the grid.
        assert f["timestamp"].shape == (30,)
        assert f["hand_qpos"].shape == (30, 12)


def test_sent_stream_default_off_unchanged(tmp_path):
    """Flag off (default): no dataset, no meta attr, schema stays 8 — even when
    the kwarg is explicitly supplied (the ctor flag alone gates the stream)."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    assert rec.start_episode(task_label="sent-stream-off")
    t0 = 5100.0
    for k in range(12):
        assert rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), arm_qpos_sent=np.full(7, float(k)))
    path = rec.stop_episode(success=True)
    assert rec.join_stop()

    with h5py.File(path, "r") as f:
        assert "action_arm_joint_sent" not in f
        assert "arm_sent_stream" not in f["meta"].attrs
        assert f["meta"].attrs["schema_version"] == 8
        assert f["meta"].attrs["num_frames"] == 12


def test_sent_stream_none_rows_are_zeros(tmp_path):
    """Unset kwarg rows record zeros (optional-action-stream convention),
    interleaved with fed values, at the correct grid indices."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4, arm_sent_stream=True)
    assert rec.start_episode()
    t0 = 5200.0
    for k in range(16):
        sent = None if k % 2 == 0 else np.full(7, float(k))
        assert rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), arm_qpos_sent=sent)
    path = rec.stop_episode(success=True)
    assert rec.join_stop()

    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["schema_version"] == 9
        ds = f["action_arm_joint_sent"][:]
        assert ds.shape == (16, 7)
        for k in range(16):
            expected = np.zeros(7) if k % 2 == 0 else np.full(7, float(k))
            assert np.array_equal(ds[k], expected), f"row {k}"
