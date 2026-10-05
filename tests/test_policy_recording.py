import json

import h5py
import numpy as np

from dexmani_real.recording.frame import EpisodeFrame
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.recording.storage.schema import DATASET_SPECS
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry


def test_actual_raw_writer_preserves_mode_trace_and_missing_auxiliary(tmp_path):
    camera = CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(camera, camera, np.eye(4))
    recorder = AsyncEpisodeRecorder(
        tmp_path, control_hz=10, rgb_shape=(16, 16, 3), execution_path="worker_grid_rtc_v1"
    )
    recorder.start_episode(
        task_label="synthetic",
        collection_source="policy_rollout",
        camera_geometry=geometry,
        camera_T_xarm_base_from_color=None,
        depth_scale=0.001,
        handbase_position_eef_m=np.zeros(3),
        handbase_quat_eef_wxyz=np.array([1.0, 0, 0, 0]),
    )
    data = {
        name: np.zeros(spec.tail_shape, dtype=spec.dtype) for name, spec in DATASET_SPECS.items()
    }
    data["hand_contact"][:] = np.nan
    data["hand_tactile_force"][:] = np.nan
    data["dispatch_status"][:] = 1
    recorder.add_frame(
        EpisodeFrame(data, np.zeros((16, 16, 3), np.uint8), np.ones((16, 16), np.uint16))
    )
    recorder.policy_trace = {
        "execute": True,
        "events": [{"event": "end", "reason": "operator", "detail": "after_predict"}],
    }
    saved = recorder.save_episode(reason="operator")
    recorder.close()
    with h5py.File(saved / "data.h5") as raw:
        meta = raw["meta"]
        assert meta.attrs["execution_path"] == "worker_grid_rtc_v1"
        assert meta.attrs["termination_reason"] == "operator"
        assert json.loads(meta["policy_trace"][()])["events"][0]["detail"] == "after_predict"
    with h5py.File(saved / "data.h5") as raw:
        assert np.isnan(raw["hand_contact"][:]).all()
        np.testing.assert_array_equal(raw["dispatch_status"][:], [[1, 1]])
