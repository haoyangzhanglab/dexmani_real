from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from cordvip_geometry.geometry import RectificationMaps
from cordvip_geometry.rgbd import RGBDFrame
from cordvip_geometry.tracking import SceneTracker


def frame(row_index=0):
    return RGBDFrame(
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        depth_m=np.full((8, 10), 0.5, dtype=np.float32),
        K=np.array([[10.0, 0, 5.0], [0, 10.0, 4.0], [0, 0, 1.0]]),
        distortion_model="none",
        distortion_coeffs=(0.0,) * 5,
        source_id="camera:test:1",
        timestamp_ns=100,
        clock="test",
        row_index=row_index,
    )


def segmenter(mask):
    return SimpleNamespace(
        initialize=MagicMock(side_effect=lambda rgb, masks: masks),
        track=MagicMock(return_value={"object": mask}),
    )


def tracker_with_masks(mask):
    sam = segmenter(mask)
    tracker = SceneTracker(sam, {}, foundationpose_root="/unused", debug_dir="/unused")
    return tracker, sam


def test_open_loads_assets_once_and_closes_context(tmp_path):
    source = frame()
    mask = np.ones(source.depth_m.shape, dtype=bool)
    sam = segmenter(mask)
    context = MagicMock()
    context.__enter__.return_value = sam
    context.__exit__.return_value = False
    mesh, points, normals = (
        object(),
        np.zeros((2, 3), np.float32),
        np.ones((2, 3), np.float32),
    )
    pose_tracker = SimpleNamespace(
        mesh=mesh,
        register=MagicMock(return_value=np.eye(4)),
        track=MagicMock(return_value=np.eye(4)),
    )
    renderer = SimpleNamespace(
        render=MagicMock(
            side_effect=lambda pixels, pose, **kwargs: np.full(len(pixels), 0.5)
        )
    )
    with (
        patch(
            "cordvip_geometry.assets.load_asset",
            return_value=(mesh, points, normals, {"source": "asset"}),
        ) as load,
        patch("cordvip_geometry.sam2.SAM2Client", return_value=context),
        patch("cordvip_geometry.tracking.PoseTracker", return_value=pose_tracker),
        patch("cordvip_geometry.tracking.ModelDepthRenderer", return_value=renderer),
        patch("cordvip_geometry.tracking.prepare_tracking_frame", return_value={}),
    ):
        with SceneTracker.open(
            {"object": tmp_path / "asset"}, workspace=tmp_path, min_matched_pixels=4
        ) as tracker:
            tracker.step(source, initial_masks={"object": mask})
            tracker.step(replace(source, timestamp_ns=101, row_index=1))
            assert tracker.assets["object"][1] is points
            assert tracker.asset_metadata == {"object": {"source": "asset"}}
            load.assert_called_once_with(tmp_path / "asset")
            context.__enter__.assert_called_once()
            context.__exit__.assert_not_called()
        context.__exit__.assert_called_once_with(None, None, None)
        assert tracker.trackers == {} and tracker.renderers == {}
    assert list(tmp_path.iterdir()) == []


def test_mask_only_open_closes_after_body_error(tmp_path):
    source = frame()
    mask = np.ones(source.depth_m.shape, dtype=bool)
    context = MagicMock()
    context.__enter__.return_value = segmenter(mask)
    context.__exit__.return_value = False
    error = RuntimeError("consumer failed")
    with (
        patch("cordvip_geometry.assets.load_asset") as load,
        patch("cordvip_geometry.sam2.SAM2Client", return_value=context),
        patch("cordvip_geometry.tracking.PoseTracker") as pose,
        pytest.raises(RuntimeError) as caught,
        SceneTracker.open({"object": None}, workspace=tmp_path) as tracker,
    ):
        result = tracker.step(source, initial_masks={"object": mask})
        assert result["objects"]["object"]["state"] == "OBSERVED_ONLY"
        raise error
    assert caught.value is error
    load.assert_not_called()
    pose.assert_not_called()
    assert context.__exit__.call_count == 1
    assert context.__exit__.call_args.args[:2] == (RuntimeError, error)


def test_repeated_live_frame_reuses_in_memory_result():
    source = frame(row_index=None)
    mask = np.ones(source.depth_m.shape, dtype=bool)
    tracker, sam = tracker_with_masks(mask)
    first = tracker.step(source, initial_masks={"object": mask})
    assert tracker.step(replace(source)) is first
    sam.initialize.assert_called_once()
    sam.track.assert_not_called()
    following = tracker.step(
        replace(source, source_id="camera:test:2", timestamp_ns=101)
    )
    assert following is not first
    sam.track.assert_called_once()


def test_initial_selections_are_loaded_only_for_first_frame(tmp_path):
    source = frame()
    mask = np.ones(source.depth_m.shape, dtype=bool)
    sam = segmenter(mask)
    selected = tmp_path / "selection"
    tracker = SceneTracker(
        sam,
        {},
        foundationpose_root="/unused",
        debug_dir="/unused",
        initial_selections={"object": selected},
    )
    with patch("cordvip_geometry.io.load_initial_selection", return_value=mask) as load:
        tracker.step(source)
        tracker.step(replace(source, timestamp_ns=101, row_index=1))
    load.assert_called_once_with(selected, source)
    sam.initialize.assert_called_once()
    sam.track.assert_called_once()


def test_raw_rows_process_repeated_camera_frame_independently():
    source = frame()
    mask = np.ones(source.depth_m.shape, dtype=bool)
    tracker, sam = tracker_with_masks(mask)
    first = tracker.step(source, initial_masks={"object": mask})
    following = tracker.step(replace(source, row_index=1))
    assert following is not first
    assert following["frame"].row_index == 1
    sam.track.assert_called_once()


@pytest.mark.parametrize(
    "change", ["K", "grid", "distortion_coeffs", "distortion_model"]
)
def test_step_rejects_changed_calibration_or_grid(change):
    source = frame()
    mask = np.ones(source.depth_m.shape, dtype=bool)
    tracker, sam = tracker_with_masks(mask)
    tracker.step(source, initial_masks={"object": mask})
    values = {
        "K": {"K": np.array([[11.0, 0, 5.0], [0, 10.0, 4.0], [0, 0, 1.0]])},
        "grid": {"rgb": source.rgb[:, :-1], "depth_m": source.depth_m[:, :-1]},
        "distortion_coeffs": {
            "distortion_model": "brown_conrady",
            "distortion_coeffs": (0.1, 0, 0, 0, 0),
        },
        "distortion_model": {"distortion_model": "brown_conrady"},
    }
    with pytest.raises(ValueError, match="fixed camera"):
        tracker.step(replace(source, timestamp_ns=101, **values[change]))
    sam.track.assert_not_called()


def test_step_rectifies_original_initial_mask_once():
    source = replace(
        frame(), distortion_model="brown_conrady", distortion_coeffs=(3.0, 0, 0, 0, 0)
    )
    mask = np.zeros(source.depth_m.shape, dtype=bool)
    mask[1:4, 1:4] = True
    original = mask.copy()
    maps = RectificationMaps(source.depth_m.shape, source.K, source.distortion_coeffs)
    expected = maps.apply_mask(mask)
    assert not np.array_equal(mask, expected)
    assert not np.array_equal(expected, maps.apply_mask(expected))
    maps.apply_mask = MagicMock(wraps=maps.apply_mask)
    tracker, sam = tracker_with_masks(expected)
    with patch(
        "cordvip_geometry.tracking.RectificationMaps", return_value=maps
    ) as build:
        result = tracker.step(source, initial_masks={"object": mask})
        tracker.step(replace(source, timestamp_ns=101, row_index=1))
    build.assert_called_once()
    maps.apply_mask.assert_called_once_with(mask)
    np.testing.assert_array_equal(sam.initialize.call_args.args[1]["object"], expected)
    np.testing.assert_array_equal(
        result["objects"]["object"]["cloud"].semantic_mask, expected
    )
    np.testing.assert_array_equal(mask, original)


def test_model_points_composes_pose_and_preserves_template():
    tracker, _ = tracker_with_masks(np.ones((8, 10), dtype=bool))
    points = np.array([[1, 0, 0], [0, 2, 0]], dtype=np.float32)
    original = points.copy()
    points.setflags(write=False)
    tracker.assets = {name: (None, points, None) for name in ("object", "lost")}
    pose = np.array([[0, -1, 0, 0.1], [1, 0, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]])
    result = {
        "objects": {
            "object": {"pose_valid": True, "pose": pose},
            "lost": {"pose_valid": False, "pose": None, "candidate_pose": pose},
            "mask": {"pose_valid": False, "pose": None},
        }
    }
    camera_points = tracker.model_points(result, "object")
    np.testing.assert_allclose(
        camera_points, [[0.1, 1.2, 0.3], [-1.9, 0.2, 0.3]], atol=1e-6
    )
    target = np.array([[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
    target_points = tracker.model_points(result, "object", T_target_camera=target)
    np.testing.assert_allclose(
        target_points, [[0.8, 3.1, 4.3], [1.8, 1.1, 4.3]], atol=1e-6
    )
    assert camera_points.dtype == target_points.dtype == np.float32
    for name in ("lost", "mask"):
        empty = tracker.model_points(result, name, T_target_camera=target)
        assert empty.shape == (0, 3) and empty.dtype == np.float32
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize(
    "name,value",
    [
        ("pixel_stride", 0),
        ("pixel_stride", 1.5),
        ("pixel_stride", True),
        ("min_matched_pixels", 2.5),
        ("min_matched_pixels", False),
        ("depth_tolerance_m", 0),
        ("depth_tolerance_m", np.nan),
        ("depth_tolerance_m", True),
        ("min_matched_fraction", np.nan),
        ("min_matched_fraction", 1.1),
    ],
)
def test_constructor_rejects_invalid_tracking_parameters_before_model_loading(
    name, value
):
    with (
        patch("cordvip_geometry.tracking.PoseTracker") as pose,
        pytest.raises(ValueError, match=name),
    ):
        SceneTracker(
            None,
            {},
            foundationpose_root="/unused",
            debug_dir="/unused",
            **{name: value},
        )
    pose.assert_not_called()


@pytest.mark.parametrize(
    "masks",
    [
        None,
        {},
        {"object": np.zeros((8, 10), bool)},
        {"object": np.ones((8, 10), np.uint8)},
    ],
)
def test_invalid_initial_mask_can_be_corrected_without_consuming_tracker(masks):
    source = frame()
    mask = np.ones(source.depth_m.shape, bool)
    tracker, sam = tracker_with_masks(mask)
    with pytest.raises(ValueError):
        tracker.step(source, initial_masks=masks)
    assert tracker.maps is None and tracker.directions is None
    sam.initialize.assert_not_called()
    result = tracker.step(source, initial_masks={"object": mask})
    assert result["objects"]["object"]["observed_valid"]
    sam.initialize.assert_called_once()


@pytest.mark.parametrize(
    "changes", [{"clock": "another_clock", "timestamp_ns": 101}, {"timestamp_ns": 99}]
)
def test_clock_change_or_time_regression_is_rejected_before_inference(changes):
    source = frame()
    mask = np.ones(source.depth_m.shape, bool)
    tracker, sam = tracker_with_masks(mask)
    tracker.step(source, initial_masks={"object": mask})
    with pytest.raises(ValueError, match="chronological"):
        tracker.step(replace(source, **changes))
    sam.track.assert_not_called()


def test_model_points_rejects_nonrigid_target_transform():
    tracker, _ = tracker_with_masks(np.ones((8, 10), bool))
    tracker.assets = {"object": (None, np.ones((2, 3), np.float32), None)}
    result = {"objects": {"object": {"pose_valid": True, "pose": np.eye(4)}}}
    with pytest.raises(ValueError, match="rotation"):
        tracker.model_points(result, "object", T_target_camera=np.diag([2.0, 1, 1, 1]))


def test_asset_name_cannot_escape_debug_directory():
    with (
        patch("cordvip_geometry.tracking.PoseTracker") as pose,
        pytest.raises(ValueError, match="Asset names"),
    ):
        SceneTracker(
            None,
            {"../outside": (None, None, None)},
            foundationpose_root="/unused",
            debug_dir="/unused",
        )
    pose.assert_not_called()


def test_fixed_camera_does_not_follow_edits_to_a_returned_frame():
    source = frame()
    mask = np.ones(source.depth_m.shape, bool)
    tracker, sam = tracker_with_masks(mask)
    result = tracker.step(source, initial_masks={"object": mask})
    result["frame"].K[0, 0] += 1
    with pytest.raises(ValueError, match="fixed camera"):
        tracker.step(replace(source, timestamp_ns=101, row_index=1))
    sam.track.assert_not_called()
