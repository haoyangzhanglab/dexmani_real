from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cordvip_geometry import track_raw


def raw_reader():
    timing = np.zeros(
        3,
        dtype=[
            (key, "i8")
            for key in (
                "observation_ns",
                "arm_ns",
                "hand_ns",
                "camera_sequence",
                "camera_ns",
            )
        ],
    )
    timing["observation_ns"] = [100, 200, 300]
    timing["camera_ns"] = [110, 110, 310]
    timing["camera_sequence"] = [4, 4, 5]
    reader = MagicMock(
        spec=[
            "path",
            "num_frames",
            "__getitem__",
            "require_fields",
            "iter_rgb_frames",
            "read_camera",
            "close",
        ]
    )
    reader.path, reader.num_frames = Path("/tmp/recorded_episode"), 3
    arrays = {"depth": np.full((3, 2, 3), 500, np.uint16), "timing/rows": timing}
    reader.__getitem__.side_effect = arrays.__getitem__
    reader.iter_rgb_frames.return_value = iter(
        [np.full((2, 3, 3), i, np.uint8) for i in (0, 0, 2)]
    )
    reader.read_camera.return_value = SimpleNamespace(
        color=SimpleNamespace(
            height=2,
            width=3,
            matrix=lambda: np.eye(3),
            distortion_model="none",
            distortion_coeffs=(0, 0, 0, 0, 0),
        ),
        depth_scale_m=0.001,
    )
    tracker = MagicMock(spec=["step", "close"])
    tracker.step.side_effect = lambda frame: {"frame": frame}
    return reader, tracker


def test_track_raw_is_lazy_and_preserves_every_source_row():
    reader, tracker = raw_reader()
    with patch("cordvip_geometry.raw.RawHandCloud") as hand:
        results = track_raw(reader, tracker)
        reader.read_camera.assert_not_called()
        frames = [result["frame"] for result in results]
    hand.assert_not_called()
    assert [frame.row_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_ns for frame in frames] == [110, 110, 310]
    assert frames[0].source_id == frames[1].source_id
    assert [int(frame.rgb[0, 0, 0]) for frame in frames] == [0, 0, 2]
    assert tracker.step.call_count == 3
    reader.close.assert_not_called()
    tracker.close.assert_not_called()


def test_track_raw_attaches_hand_from_the_same_row():
    reader, tracker = raw_reader()
    with patch("cordvip_geometry.raw.RawHandCloud") as hand:
        hand.return_value.for_row.side_effect = lambda frame, timing: {
            "row": frame.row_index,
            "timestamp_ns": int(timing["observation_ns"]),
        }
        results = list(track_raw(reader, tracker, with_hand=True, hand_points=64))
    hand.assert_called_once_with(reader, reader.read_camera.return_value, num_points=64)
    assert [result["hand"]["row"] for result in results] == [0, 1, 2]
    assert [result["hand"]["timestamp_ns"] for result in results] == [100, 200, 300]
    for result, call in zip(results, hand.return_value.for_row.call_args_list):
        assert call.args[0] is result["frame"]


def test_track_raw_propagates_failure_without_closing_owned_resources():
    reader, tracker = raw_reader()
    tracker.step.side_effect = [{"frame": "first"}, RuntimeError("tracking failed")]
    results = track_raw(reader, tracker)
    assert next(results) == {"frame": "first"}
    with pytest.raises(RuntimeError, match="tracking failed"):
        next(results)
    reader.close.assert_not_called()
    tracker.close.assert_not_called()
