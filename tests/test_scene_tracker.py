import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from cordvip_geometry import tracking
from cordvip_geometry.cli.scene import choose_masks, load_scene
from cordvip_geometry.rgbd import RGBDFrame


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
    result = tracker.step(frame(.5), initial_masks=masks)
    assert result["objects"]["model"]["state"] == "TRACKING"
    assert result["objects"]["observed"]["state"] == "OBSERVED_ONLY"


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
