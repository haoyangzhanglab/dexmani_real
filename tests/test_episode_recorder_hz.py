"""EpisodeRecorder control_hz grid tests (synthetic feed, no hardware)."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
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
        assert meta["schema_version"] == 8
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


def test_stop_reason_truncated(tmp_path):
    """max_frames auto-stop must be visible in /meta (truncated + stop_reason)."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=8, control_hz=HZ, min_frames=4)
    assert rec.start_episode()
    t0 = 1000.0
    for k in range(10):
        ok = rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), signals={"ik_ok": True})
        assert ok == (k < 8)  # 9th/10th rejected at the cap
    assert rec.max_frames_reached
    path = rec.stop_episode(success=True)
    assert rec.join_stop()
    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        assert bool(meta["truncated"])
        assert meta["stop_reason"] == "max_frames"
        assert meta["num_frames"] == 8


def test_stop_reason_manual_and_explicit(tmp_path):
    path = _record_episode(tmp_path, n_frames=20)
    with h5py.File(path, "r") as f:
        assert not bool(f["meta"].attrs["truncated"])
        assert f["meta"].attrs["stop_reason"] == "manual"

    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ)
    assert rec.start_episode()
    for k in range(5):
        rec.add_frame(_fake_state(2000.0 + k * DT), _fake_action(), _fake_vr())
    path2 = rec.stop_episode(success=False, reason="estop")
    assert rec.join_stop()
    with h5py.File(path2, "r") as f:
        assert f["meta"].attrs["stop_reason"] == "estop"
        assert not bool(f["meta"].attrs["success"])


def test_cam_drop_counter_and_lzf(tmp_path):
    """Enqueue-side camera drops are counted in /meta; rgb dataset is lzf.

    No cam-writer thread + a tiny queue → deterministic backlog: the headroom
    check (qsize < maxsize-1) admits only the first frame, drops the other 9;
    the stop-time safety drain writes the one queued item and the tail-pad
    heals the dataset length — exactly the masking the counters must expose.
    """
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    rec._start_cam_writer = lambda: None  # no consumer
    assert rec.start_episode()
    rec._cam_queue = queue.Queue(maxsize=2)
    t0 = 3000.0
    for k in range(10):
        cam = {"rgb": np.zeros((8, 8, 3), dtype=np.uint8), "frame_number": k}
        assert rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), camera_frame=cam)
    path = rec.stop_episode(success=True)
    assert rec.join_stop()
    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        assert meta["cam_frames_dropped"] == 9
        assert meta["cam_items_written"] == 1  # safety-drain accounting
        assert f["rgb"].shape == (10, 8, 8, 3)  # length healed by tail-pad, content lost
        assert f["rgb"].compression == "lzf"
        assert f["rgb"][:].shape == (10, 8, 8, 3)  # decompresses cleanly


def test_join_stop_keep_handle(tmp_path):
    """join_stop keeps the handle on timeout so the overlap guard stays armed."""
    rec = EpisodeRecorder(data_dir=str(tmp_path))
    t = threading.Thread(target=time.sleep, args=(0.5,), daemon=True)
    t.start()
    rec._stop_thread = t
    assert rec.join_stop(timeout=0.05) is False
    assert rec._stop_thread is t
    assert rec.join_stop(timeout=2.0) is True
    assert rec._stop_thread is None
    assert rec.join_stop() is True  # reentrant no-op


def test_start_refuses_pending_stop(tmp_path):
    rec = EpisodeRecorder(data_dir=str(tmp_path))
    rec.join_stop = lambda timeout=10.0: False  # pending flush that never finishes
    assert rec.start_episode() is False
    assert not rec.is_recording


def test_backfill_camera_meta_after_lazy_write(tmp_path):
    """B-before-camera-connect: camera attrs missing at the initial lazy meta
    write are re-written at stop from backfill_camera_meta() values."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    assert rec.start_episode()  # camera child not connected → K/depth_scale snapshot None
    t0 = 2000.0
    for k in range(8):
        assert rec.add_frame(_fake_state(t0 + k * DT), _fake_action(), _fake_vr(), signals={"ik_ok": True})
    rec._ensure_hdf5()  # initial meta write happens with camera keys still None
    rec.backfill_camera_meta(depth_scale=0.00025, camera_K=np.arange(9, dtype=np.float64).reshape(3, 3))
    path = rec.stop_episode(success=True)
    assert rec.join_stop()
    with h5py.File(path, "r") as f:
        meta = f["meta"].attrs
        assert meta["depth_scale"] == pytest.approx(0.00025)
        assert np.allclose(meta["camera_K"], np.arange(9.0))


def test_backfill_does_not_override_start_values(tmp_path):
    """Keys already supplied at start_episode() win over late backfill."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    assert rec.start_episode(depth_scale=0.001)
    assert rec.add_frame(_fake_state(2000.0), _fake_action(), _fake_vr(), signals={"ik_ok": True})
    rec.backfill_camera_meta(depth_scale=0.00025)
    path = rec.stop_episode(success=True)
    assert rec.join_stop()
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["depth_scale"] == pytest.approx(0.001)


def test_stop_impl_enospc_sets_stop_error(tmp_path):
    """ENOSPC mid-flush → stop_error set, join_stop False, state reset for restart."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    assert rec.start_episode()
    for k in range(8):
        assert rec.add_frame(_fake_state(2000.0 + k * DT), _fake_action(), _fake_vr(), signals={"ik_ok": True})
    # Inject disk-full error into the flush path called by _stop_episode_impl_inner.
    def _raise_enospc():
        raise OSError(28, "No space left on device")
    rec._flush_buffered = _raise_enospc
    path = rec.stop_episode(success=True)
    assert path is not None
    ok = rec.join_stop(timeout=5.0)
    assert not ok, "join_stop must return False when stop thread crashed"
    assert rec.stop_error is not None
    assert "OSError" in rec.stop_error
    assert "No space" in rec.stop_error
    assert not rec.is_recording


def test_restart_after_stop_error(tmp_path):
    """State is clean after a crashed stop — a new episode can start."""
    rec = EpisodeRecorder(data_dir=str(tmp_path), max_frames=960, control_hz=HZ, min_frames=4)
    assert rec.start_episode()
    assert rec.add_frame(_fake_state(2000.0), _fake_action(), _fake_vr(), signals={"ik_ok": True})
    _orig_flush = rec._flush_buffered
    rec._flush_buffered = lambda: (_ for _ in ()).throw(OSError(28, "No space left on device"))
    rec.stop_episode(success=True)
    rec.join_stop(timeout=5.0)
    assert rec.stop_error is not None
    # Restore the real _flush_buffered so the second episode writes cleanly.
    rec._flush_buffered = _orig_flush
    # start_episode must succeed — state was reset by the wrapper's except block.
    assert rec.start_episode()
    assert rec.is_recording
    assert rec.add_frame(_fake_state(3000.0), _fake_action(), _fake_vr(), signals={"ik_ok": True})
    path = rec.stop_episode(success=True)
    assert rec.join_stop()
    assert rec.stop_error is None  # clean this time
    assert Path(path).exists()
