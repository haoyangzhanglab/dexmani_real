from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh
from cordvip_geometry.foundationpose import guide_centered_pose
from cordvip_geometry.geometry import simplify_mesh
from cordvip_geometry.hand import HandCloud
from cordvip_geometry.model_depth import ModelDepthRenderer, camera_rays
from cordvip_geometry.raw import iter_raw_rows
from cordvip_geometry.rgbd import RGBDFrame
from cordvip_geometry.tracking import SceneTracker, scene_arrays


def test_model_depth_preserves_camera_z_and_pixel_order():
    mesh = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    renderer = ModelDepthRenderer(mesh, pixel_stride=2)
    K = np.array([[100, 0, 30], [0, 100, 30], [0, 0, 1]], dtype=float)
    directions = camera_rays((62, 62), K, 2)
    pixels = np.array([[30, 40], [0, 0], [30, 30]], dtype=np.int32)
    pose = np.eye(4)
    pose[2, 3] = 1.2
    depth = renderer.render(pixels, pose, directions=directions)
    np.testing.assert_allclose(depth, [1, 0, 1], atol=1e-6)
    assert depth.dtype == np.float32
    pose[2, 3] = 0.1
    np.testing.assert_allclose(
        renderer.render(pixels, pose, directions=directions), 0.3, atol=1e-6
    )
    empty = renderer.render(
        np.empty((0, 2), dtype=np.int32), pose, directions=directions
    )
    assert empty.shape == (0,) and empty.dtype == np.float32


@pytest.mark.parametrize("textured", [False, True])
def test_simplify_mesh_preserves_frame_color_and_source(textured):
    mesh = trimesh.creation.icosphere(subdivisions=6, radius=0.05)
    mesh.apply_translation([0.3, -0.2, 0.1])
    color = [71, 151, 231, 255]
    if textured:
        from PIL import Image

        mesh.visual = trimesh.visual.texture.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            image=Image.new("RGBA", (4, 4), tuple(color)),
        )
    else:
        mesh.visual.vertex_colors = color
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()
    reduced = simplify_mesh(mesh)
    assert len(reduced.faces) <= 20_000
    np.testing.assert_allclose(reduced.bounds, mesh.bounds, atol=0.0001)
    np.testing.assert_allclose(
        reduced.visual.vertex_colors, np.tile(color, (len(reduced.vertices), 1))
    )
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_small_mesh_is_unchanged():
    mesh = trimesh.creation.box()
    assert simplify_mesh(mesh) is mesh


def test_bbox_guidance_preserves_rotation_and_depth():
    K = np.array([[100, 0, 10], [0, 100, 10], [0, 0, 1]], dtype=float)
    pose = trimesh.transformations.rotation_matrix(0.4, [0, 0, 1])
    pose[:3, 3] = [0.5, 0.3, 0.8]
    mask = np.zeros((30, 30), dtype=bool)
    mask[12:17, 16:21] = True
    guided = guide_centered_pose(pose, K, mask)
    np.testing.assert_allclose(guided[:3, :3], pose[:3, :3])
    np.testing.assert_allclose(guided[:3, 3], [0.064, 0.032, 0.8])
    np.testing.assert_allclose(pose[:3, 3], [0.5, 0.3, 0.8])


def test_raw_rows_read_recorded_timing_dataset():
    timing = np.zeros(
        2,
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
    timing["observation_ns"] = [100, 200]
    timing["camera_ns"] = [110, 210]
    timing["camera_sequence"] = [10, 11]
    reader = MagicMock(
        spec=["path", "num_frames", "__getitem__", "require_fields", "iter_rgb_frames"]
    )
    reader.path, reader.num_frames = Path("/tmp/recorded_episode"), 2
    arrays = {"depth": np.full((2, 2, 3), 500, dtype=np.uint16), "timing/rows": timing}
    reader.__getitem__.side_effect = arrays.__getitem__
    reader.iter_rgb_frames.return_value = iter(
        [np.zeros((2, 3, 3), dtype=np.uint8)] * 2
    )
    intrinsics = SimpleNamespace(
        height=2,
        width=3,
        matrix=lambda: np.eye(3),
        distortion_model="none",
        distortion_coeffs=(0, 0, 0, 0, 0),
    )
    frames = list(
        iter_raw_rows(reader, SimpleNamespace(color=intrinsics, depth_scale_m=0.001))
    )
    assert [frame.timestamp_ns for frame, _ in frames] == [110, 210]
    assert [frame.row_index for frame, _ in frames] == [0, 1]
    np.testing.assert_allclose(frames[0][0].depth_m, 0.5)
    assert frames[0][0].source_id.endswith(":camera:10")


def make_scene():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.full((8, 8), 0.5, dtype=np.float32)
    mask = np.ones((8, 8), dtype=bool)
    segmenter = SimpleNamespace(
        initialize=lambda rgb, masks: masks, track=lambda rgb: {"object": mask}
    )
    tracker = SceneTracker(
        segmenter,
        {},
        foundationpose_root="/unused",
        debug_dir="/unused",
        min_matched_pixels=4,
    )
    frame = RGBDFrame(
        rgb=rgb,
        depth_m=depth,
        K=np.eye(3),
        distortion_model="none",
        distortion_coeffs=(0.0,) * 5,
        source_id="row:0",
        row_index=0,
        timestamp_ns=1,
        clock="test",
    )
    return tracker, frame, mask


def test_mask_only_requires_no_pose_model():
    tracker, frame, mask = make_scene()
    result = tracker.step(frame, initial_masks={"object": mask})
    arrays = scene_arrays(result)
    assert arrays["object__state"] == "OBSERVED_ONLY"
    assert arrays["object__observed_valid"]
    assert not arrays["object__pose_valid"]
    assert np.isnan(arrays["object__T_camera_object"]).all()
    assert arrays["object__observed_points_camera"].shape == (16, 3)


def test_depth_disagreement_loses_pose_and_retains_evidence():
    tracker, frame, mask = make_scene()
    model = MagicMock()
    model.register.return_value = model.track.return_value = np.eye(4)
    renderer = MagicMock()
    renderer.render.side_effect = [np.full(16, 0.5), np.full(16, 0.7)]
    tracker.trackers["object"], tracker.renderers["object"] = model, renderer
    first = tracker.step(frame, initial_masks={"object": mask})
    assert first["objects"]["object"]["pose_valid"]
    with patch(
        "cordvip_geometry.tracking.prepare_tracking_frame", return_value={}
    ) as prepare:
        lost = tracker.step(
            replace(frame, row_index=1, source_id="row:1", timestamp_ns=2)
        )
        still_lost = tracker.step(
            replace(frame, row_index=2, source_id="row:2", timestamp_ns=3)
        )
    assert prepare.call_count == 1
    assert model.track.call_count == 1
    arrays = scene_arrays(lost)
    assert arrays["object__state"] == "POSE_LOST"
    assert arrays["object__observed_valid"]
    assert np.isnan(arrays["object__T_camera_object"]).all()
    np.testing.assert_array_equal(
        arrays["object__candidate_T_camera_object"], np.eye(4)
    )
    np.testing.assert_allclose(
        arrays["object__observed_model_depth_residual_m"], -0.2, atol=1e-6
    )
    assert arrays["object__model_supported_point_count"] == 0
    assert still_lost["objects"]["object"]["observed_valid"]


def test_insufficient_initial_depth_fails_before_registration():
    tracker, frame, mask = make_scene()
    model = MagicMock()
    tracker.trackers["object"] = model
    with pytest.raises(ValueError, match="valid depth pixels"):
        tracker.step(
            replace(frame, depth_m=frame.depth_m * 0), initial_masks={"object": mask}
        )
    model.register.assert_not_called()


def test_model_failure_propagates():
    tracker, frame, mask = make_scene()
    tracker.trackers["object"] = SimpleNamespace(
        register=MagicMock(side_effect=RuntimeError("GPU failure"))
    )
    with pytest.raises(RuntimeError, match="GPU failure"):
        tracker.step(frame, initial_masks={"object": mask})


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_points", 1.5),
        ("num_points", True),
        ("num_points", 0),
        ("seed", 0.5),
        ("seed", True),
        ("seed", -1),
    ],
)
def test_hand_sampling_parameters_are_not_silently_truncated(field, value):
    with pytest.raises(ValueError, match=field):
        HandCloud("/unused", ["joint"], T_camera_from_root=np.eye(4), **{field: value})
