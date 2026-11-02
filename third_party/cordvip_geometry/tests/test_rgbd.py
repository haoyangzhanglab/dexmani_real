from dataclasses import replace

import numpy as np
import pytest
from cordvip_geometry.rgbd import RGBDFrame, load_frame, save_frame

from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot


def camera_and_row():
    camera = CameraSnapshot(
        serial="test_camera",
        color=CameraIntrinsics(10, 8, 10, 10, 5, 4, "none", (0.0,) * 5),
        depth_scale_m=0.001,
        T_xarm_base_from_color=None,
    )
    row = {
        "rgb": np.zeros((8, 10, 3), dtype=np.uint8),
        "depth": np.full((8, 10), 500, dtype=np.uint16),
        "timestamp_ns": np.int64(100),
        "ring_sequence": np.uint64(7),
        "color_frame_number": 12,
    }
    return camera, row


def test_from_camera_preserves_calibration_time_and_input_arrays():
    camera, row = camera_and_row()
    frame = RGBDFrame.from_camera(camera, row)
    np.testing.assert_allclose(frame.depth_m, 0.5)
    np.testing.assert_array_equal(frame.K, camera.color.matrix())
    assert frame.depth_m.dtype == np.float32
    assert frame.source_id == "camera:test_camera:7"
    assert frame.timestamp_ns == 100
    assert frame.clock == "host_monotonic_oldest_rgbd_receive"
    assert frame.row_index is None
    assert frame.distortion_model == camera.color.distortion_model
    assert frame.distortion_coeffs == camera.color.distortion_coeffs
    assert frame.rgb is row["rgb"]
    assert np.shares_memory(frame.rgb, row["rgb"])
    assert not np.shares_memory(frame.depth_m, row["depth"])
    assert row["rgb"].flags.writeable and row["depth"].flags.writeable
    np.testing.assert_array_equal(row["depth"], 500)


@pytest.mark.parametrize(
    "key,value",
    [
        ("rgb", np.zeros((8, 10, 3), dtype=np.float32)),
        ("rgb", np.zeros((8, 9, 3), dtype=np.uint8)),
        ("depth", np.ones((8, 10), dtype=np.float32)),
        ("depth", np.ones((9, 10), dtype=np.uint16)),
        ("timestamp_ns", 0),
        ("timestamp_ns", 1.5),
        ("ring_sequence", -1),
        ("ring_sequence", True),
    ],
)
def test_from_camera_rejects_malformed_dynamic_rows(key, value):
    camera, row = camera_and_row()
    row[key] = value
    with pytest.raises(ValueError):
        RGBDFrame.from_camera(camera, row)


@pytest.mark.parametrize(
    "field,value",
    [
        ("depth_m", np.full((8, 10), -1, np.float32)),
        ("depth_m", np.full((8, 10), np.nan, np.float32)),
        ("depth_m", np.ones((8, 10), np.float64)),
        ("rgb", np.zeros((8, 10, 3), np.float32)),
        ("rgb", np.zeros((8, 9, 3), np.uint8)),
        ("K", np.zeros((3, 3))),
        ("distortion_model", "inverse_brown_conrady"),
        ("source_id", ""),
        ("clock", ""),
        ("timestamp_ns", 0),
        ("timestamp_ns", 1.5),
        ("row_index", -1),
    ],
)
def test_direct_frame_construction_admits_only_valid_source_data(field, value):
    camera, row = camera_and_row()
    source = RGBDFrame.from_camera(camera, row)
    with pytest.raises(ValueError):
        replace(source, **{field: value})


def test_load_frame_uses_same_rgbd_boundary(tmp_path):
    camera, row = camera_and_row()
    path = tmp_path / "source.npz"
    save_frame(path, RGBDFrame.from_camera(camera, row))
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: saved[key] for key in saved.files}
    payload["depth_m"][0, 0] = np.nan
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="depth_m"):
        load_frame(path)
