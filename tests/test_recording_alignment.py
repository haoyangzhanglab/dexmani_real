"""Verification for the refactored recording path (RecordingSession + aligned buffer).

Covers the bugs the refactor fixes:
  NEW-1  no trailing-frame loss on stop (all enqueued frames land)
  NEW-2  target_eef_pos / target_eef_rot6d recorded
  NEW-3  ik_ok / retarget_ok / held flags recorded
  NEW-4  single aligned grid: every dataset length N, strictly-increasing dt grid
  NEW-7  eye_to_hand extrinsics stored as the static base→camera pose
"""

from __future__ import annotations

import time

import h5py
import numpy as np
import pytest

from dexmani_real.robot.types import RobotState, RobotAction
from dexmani_real.recording import (
    CollectionConfig,
    CollectionLoop,
    DataValidator,
    EpisodeRecorder,
    RecordingSession,
)

N = 200
DT = 1.0 / 50.0


class _FakeCalib:
    """Duck-typed CameraCalib: only to_meta_dict(cam_name) is used by the recorder."""

    def __init__(self, T_base_camera: np.ndarray) -> None:
        self._T = T_base_camera

    def to_meta_dict(self, cam_name: str) -> dict:
        return {
            "camera_serial": "TEST",
            "camera_type": "eye_to_hand",
            "camera_T_base_camera": self._T.flatten().tolist(),
        }


def _state(ts: float, i: int) -> RobotState:
    f = 0.001 * i  # per-frame variation → nonzero variance, no consecutive duplicates
    return RobotState(
        arm_qpos=np.arange(7, dtype=np.float64) * 0.1 + f,
        arm_qvel=np.zeros(7) + f,
        arm_tau=np.zeros(7) + f,
        eef_pos=np.array([0.3, 0.0, 0.2]) + f,
        eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        hand_qpos=np.arange(12, dtype=np.float64) * 0.05 + f,
        hand_tactile_sum=np.zeros((5, 3)) + f,
        hand_tactile_force=np.zeros((5, 120, 3)) + f,
        fingertip_pos=np.zeros((5, 3)) + f,
        arm_connected=True,
        hand_connected=True,
        timestamp=ts,
    )


def _action(i: int) -> RobotAction:
    # Even frames carry an EEF target; odd frames leave it None (→ NaN row).
    has_tgt = i % 2 == 0
    return RobotAction(
        arm_qpos_cmd=np.arange(7, dtype=np.float64) * 0.1 + 0.001 * i,
        hand_qpos_cmd=np.arange(12, dtype=np.float64) * 0.05 + 0.001 * i,
        target_eef_pos=(np.array([0.3, 0.0, 0.2]) + 0.001 * i) if has_tgt else None,
        target_eef_rot6d=(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])) if has_tgt else None,
    )


def _vr(i: int) -> dict:
    return {
        "wrist_pos": np.array([0.1, 0.2, 0.3]) + 0.001 * i,
        "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)) + 0.001 * i,
        "local_recv_ns": int(1e9 * (i * DT)),
    }


def _cam(i: int) -> dict:
    return {
        "rgb": np.ones((4, 4, 3), dtype=np.uint8) * ((i % 254) + 1),
        "depth": np.ones((4, 4), dtype=np.uint16) * (i + 1),
        "timestamp": float(i * DT),
    }


def _Tbe(i: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [0.3 + 0.001 * i, 0.0, 0.2]
    return T


def _run_session(tmp_path, with_camera: bool):
    rec = EpisodeRecorder(str(tmp_path), max_frames=5000)
    loop = CollectionLoop(rec, CollectionConfig(save_sidecar_json=True))
    sess = RecordingSession(loop, validate=False)

    T_bc = np.array(
        [[0.0, -1.0, 0.0, 0.5], [1.0, 0.0, 0.0, -0.2], [0.0, 0.0, 1.0, 0.8], [0.0, 0.0, 0.0, 1.0]]
    )
    meta: dict = {"task_label": "test", "operator": "tester"}
    if with_camera:
        meta["calib"] = _FakeCalib(T_bc)
        meta["camera_name"] = "camera"

    sess.start(meta)

    # Wait until the writer thread captured the buffer start_time, then align our
    # synthetic capture timestamps to that grid (S + (i+0.5)*dt → grid slot i).
    t0 = time.time()
    while rec._start_time is None and time.time() - t0 < 5.0:
        time.sleep(0.001)
    assert rec._start_time is not None, "start_episode never ran"
    S = rec._start_time

    for i in range(N):
        ts = S + (i + 0.5) * DT
        sess.record(
            dict(
                state=_state(ts, i),
                action=_action(i),
                vr_frame=_vr(i),
                camera_frame=_cam(i) if with_camera else None,
                camera_frames=None,
                T_base_eef=_Tbe(i) if with_camera else None,
                signals={"ik_ok": True, "retarget_ok": (i % 3 != 0), "held": (i % 5 == 0)},
            )
        )

    path = sess.stop(save=True)
    sess.shutdown()
    assert path is not None, "stop() returned no path (timed out or failed)"
    return path, T_bc


def test_no_tail_loss_and_alignment_no_camera(tmp_path):
    path, _ = _run_session(tmp_path, with_camera=False)
    with h5py.File(path, "r") as f:
        # (a) no tail loss — every enqueued frame landed on the grid
        assert f["meta"].attrs["num_frames"] == N
        assert f["meta"].attrs["schema_version"] == 2

        # (b) index alignment — all streams length N
        keys = [
            "obs/arm_qpos", "obs/eef_pos", "obs/hand_qpos", "obs/hand_tactile_force",
            "action/arm_qpos", "action/hand_qpos",
            "action/target_eef_pos", "action/target_eef_rot6d",
            "action/ik_ok", "action/retarget_ok", "action/held",
            "vr/wrist_pos", "vr/landmarks", "timestamps", "vr_timestamps",
        ]
        for k in keys:
            assert f[k].shape[0] == N, (k, f[k].shape)

        # strictly-increasing grid with ~dt spacing
        ts = f["timestamps"][:]
        assert np.all(np.diff(ts) > 0)
        assert abs(float(np.median(np.diff(ts))) - DT) < 1e-6

        # (c) signals present + populated
        assert f["action/ik_ok"].dtype == np.bool_
        held = f["action/held"][:]
        assert held.any() and (~held).any()  # both True and False recorded

        # (NEW-2) target present: even frames finite, odd frames NaN
        tp = f["action/target_eef_pos"][:]
        assert np.isfinite(tp[0]).all()
        assert np.isnan(tp[1]).all()
        assert f["action/target_eef_rot6d"].shape[1] == 6


def test_camera_alignment_and_extrinsics(tmp_path):
    path, T_bc = _run_session(tmp_path, with_camera=True)
    with h5py.File(path, "r") as f:
        assert f["meta"].attrs["num_frames"] == N
        for k in [
            "obs/arm_qpos", "action/arm_qpos",
            "camera/rgb", "camera/depth", "camera/timestamps", "camera/extrinsics",
            "timestamps",
        ]:
            assert f[k].shape[0] == N, (k, f[k].shape)

        # (NEW-7) eye_to_hand → static base→camera extrinsics every frame
        extr = f["camera/extrinsics"][:]
        for i in range(0, N, 50):
            assert np.allclose(extr[i], T_bc), (i, extr[i])

        # camera not all-zero
        assert f["camera/rgb"][0].any()


def test_data_validator_passes(tmp_path):
    path, _ = _run_session(tmp_path, with_camera=False)
    report = DataValidator().validate(path)
    failed = [(c.name, c.detail) for c in report.checks if not c.passed]
    assert report.is_valid, failed
