import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from cordvip_geometry import tracking
from cordvip_geometry.cli.scene import choose_masks, load_scene
from cordvip_geometry.cli.view import view_results
from cordvip_geometry.io import scene_arrays, write_npz
from cordvip_geometry.rgbd import RGBDFrame
from cordvip_geometry.viewer import scene_clouds


class Segmenter:
    def initialize(self, rgb, masks):
        self.masks = masks
        return masks

    def track(self, rgb):
        return self.masks


def frame(depth):
    return RGBDFrame(
        np.zeros((32, 32, 3), np.uint8), np.full((32, 32), depth, np.float32),
        np.array([[100., 0., 16.], [0., 100., 16.], [0., 0., 1.]]), "none", (0.,) * 5,
    )


def test_observed_only_accepts_unused_pose_parameters_and_zero_depth(tmp_path):
    tracker = tracking.SceneTracker(
        Segmenter(), {}, foundationpose_root=None, debug_dir=tmp_path,
        min_matched_pixels="unused", depth_tolerance_m=np.nan, min_matched_fraction=None,
    )
    for result in (tracker.step(frame(0), initial_masks={"observed": np.ones((32, 32), bool)}),
                   tracker.step(frame(0))):
        item = result["objects"]["observed"]
        assert item["state"] == "OBSERVED_ONLY"
        assert not item["observed_valid"] and not item["pose_valid"]


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_stride_is_required_without_assets(tmp_path, stride):
    with pytest.raises(ValueError, match="pixel_stride"):
        tracking.SceneTracker(None, {}, foundationpose_root=None, debug_dir=tmp_path,
                              pixel_stride=stride)


@pytest.mark.parametrize("parameters", [
    {"min_matched_pixels": 0}, {"min_matched_pixels": True}, {"min_matched_pixels": 1.5},
    {"depth_tolerance_m": 0}, {"depth_tolerance_m": np.nan},
    {"min_matched_fraction": 0}, {"min_matched_fraction": 1.1},
    {"min_matched_fraction": np.inf},
])
def test_pose_parameters_required_with_assets(tmp_path, parameters):
    with pytest.raises(ValueError):
        tracking.SceneTracker(None, {"model": (None, None, None)},
                              foundationpose_root=None, debug_dir=tmp_path, **parameters)


def test_mixed_scene_and_pose_depth_requirement(tmp_path, monkeypatch):
    pose = SimpleNamespace(mesh=None, register=Mock(return_value=np.eye(4)))
    monkeypatch.setattr(tracking, "PoseTracker", lambda *args, **kwargs: pose)
    monkeypatch.setattr(tracking, "ModelDepthRenderer", lambda *args, **kwargs: SimpleNamespace(
        render=lambda pixels, *args, **kwargs: np.full(len(pixels), .5, np.float32)))
    tracker = tracking.SceneTracker(Segmenter(), {"model": (None, None, None)},
                                    foundationpose_root=None, debug_dir=tmp_path)
    masks = {"model": np.ones((32, 32), bool), "observed": np.ones((32, 32), bool)}
    with pytest.raises(ValueError, match="pose initialization"):
        tracker.step(frame(0), initial_masks=masks)
    pose.register.assert_not_called()
    tracker = tracking.SceneTracker(Segmenter(), {"model": (None, None, None)},
                                    foundationpose_root=None, debug_dir=tmp_path)
    current = replace(frame(.5), row_index=0, timestamp_ns=0, source_id="")
    result = tracker.step(current, initial_masks=masks)
    assert result["objects"]["model"]["state"] == "TRACKING"
    assert result["objects"]["observed"]["state"] == "OBSERVED_ONLY"
    frames = tmp_path / "frames"
    frames.mkdir()
    write_npz(frames / "000000.npz", scene_arrays(result))
    with np.load(frames / "000000.npz", allow_pickle=False) as saved:
        row = dict(saved)
    assert row["timestamp_ns"] == row["row_index"] == 0
    assert row["source_id"] == "" and "clock" not in row
    for name in masks:
        cloud = result["objects"][name]["cloud"]
        np.testing.assert_array_equal(row[name + "__observed_points_camera"],
                                      cloud.observed_points_camera)
        assert row[name + "__observed_points_camera"].dtype == np.float32
    assert np.isnan(row["observed__T_camera_object"]).all()
    colors = {"model": (255, 0, 0), "observed": (0, 255, 0)}
    points, _ = scene_clouds(row, masks, colors)
    assert len(points) == 512
    template = np.array([[0., 0., .5], [.1, 0., .5]], np.float32)
    points, _ = scene_clouds(row, masks, colors, model_templates={"model": template})
    np.testing.assert_array_equal(points, template)
    assert view_results(tmp_path, headless=True) == 0


def test_initial_mask_bypasses_unused_prompt(tmp_path):
    mask = np.ones((32, 32), bool)
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"objects": [
        {"id": "observed", "initial_selection": str(path), "prompt": None},
    ]}))
    objects = load_scene(scene, headless=True)
    segmenter = SimpleNamespace(predict=Mock(side_effect=AssertionError("unused prompt")))
    masks, _ = choose_masks(objects, frame(0), None, tmp_path, segmenter)
    np.testing.assert_array_equal(masks["observed"], mask)
    segmenter.predict.assert_not_called()
