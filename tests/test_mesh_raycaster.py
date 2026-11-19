import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("open3d")

GEOMETRY_SOURCE = (
    Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
)
with patch.object(sys, "path", [str(GEOMETRY_SOURCE), *sys.path]):
    from cordvip_geometry.camera import camera_rays
    from cordvip_geometry.mesh import MeshRaycaster

PIXEL_STRIDE = 4
IMAGE_SHAPE = (480, 640)
INTRINSICS = np.array([[610, 0, 320], [0, 608, 240], [0, 0, 1]], dtype=float)


@pytest.fixture(scope="module")
def box_renderer():
    return MeshRaycaster(trimesh.creation.box(extents=(0.12, 0.078, 0.085)))


@pytest.mark.parametrize(
    ("translation", "rotvec", "image_shape", "casting"),
    [
        pytest.param((0, 0, 0.65), (0, 0, 0), IMAGE_SHAPE, "roi", id="center"),
        pytest.param((0.248, 0, 0.65), (0, 0, 0), IMAGE_SHAPE, "roi", id="edge"),
        pytest.param((0.29, 0, 0.65), (0, 0, 0), IMAGE_SHAPE, "roi", id="partial"),
        pytest.param((0, 0, 0.55), (0.35, 0.8, 0.5), IMAGE_SHAPE, "roi", id="rotate"),
        pytest.param((0, 0, 0.02), (0, 0, 0), IMAGE_SHAPE, "full", id="near-plane"),
        pytest.param((0, 0, -0.12), (0, 0, 0), IMAGE_SHAPE, "empty", id="behind"),
        # 盒子覆盖右下角，检验奇数尺寸的最后一行和一列。
        pytest.param((0.29, 0.23, 0.65), (0, 0, 0), (481, 643), "roi", id="odd-grid"),
    ],
)
def test_roi_matches_full_render(
    box_renderer, translation, rotvec, image_shape, casting
):
    directions = camera_rays(image_shape, INTRINSICS, PIXEL_STRIDE)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = translation
    full = box_renderer.render(directions, pose)
    with patch.object(box_renderer, "render", wraps=box_renderer.render) as render:
        roi = box_renderer.render_roi(
            directions, pose, INTRINSICS, PIXEL_STRIDE, 2 * PIXEL_STRIDE
        )

    assert roi.shape == tuple(
        (size + PIXEL_STRIDE - 1) // PIXEL_STRIDE for size in image_shape
    )
    assert roi.dtype == full.dtype == np.float32
    np.testing.assert_allclose(roi, full, rtol=0, atol=1e-5)
    if casting == "empty":
        assert not roi.any()
        render.assert_not_called()
    else:
        assert full.any()
        render.assert_called_once()
        cast_directions = render.call_args.args[0]
        if casting == "full":
            assert cast_directions is directions
        else:
            assert cast_directions.size < directions.size


def test_roi_uses_original_sample_positions_and_zero_fills(box_renderer):
    directions = camera_rays(IMAGE_SHAPE, INTRINSICS, PIXEL_STRIDE)
    pose = np.eye(4)
    pose[2, 3] = 0.65
    window = np.s_[49:72, 63:98]
    expected_rays = directions[window]
    with patch.object(
        box_renderer,
        "render",
        return_value=np.ones(expected_rays.shape[:2], np.float32),
    ) as render:
        roi = box_renderer.render_roi(
            directions, pose, INTRINSICS, PIXEL_STRIDE, 2 * PIXEL_STRIDE
        )

    render.assert_called_once()
    np.testing.assert_array_equal(render.call_args.args[0], expected_rays)
    expected = np.zeros(directions.shape[:2], dtype=np.float32)
    expected[window] = 1
    np.testing.assert_array_equal(roi, expected)


def test_roi_matches_full_for_near_plane_with_rotation_error():
    directions = camera_rays(IMAGE_SHAPE, INTRINSICS, PIXEL_STRIDE)
    mesh = trimesh.creation.box(extents=(0.00016, 0.00017, 0.00012))
    mesh.apply_translation((0, 0, 0.052))
    renderer = MeshRaycaster(mesh)
    pose = np.eye(4)
    # 容差内的旋转误差足以让近处包围盒的两个投影分支不同。
    pose[2, 2] = 1.0004
    pose[:3, 3] = (0.00038, -0.00025, 0.001002 - mesh.bounds[0, 2] * pose[2, 2])
    full = renderer.render(directions, pose)
    roi = renderer.render_roi(
        directions, pose, INTRINSICS, PIXEL_STRIDE, 2 * PIXEL_STRIDE
    )

    assert full.any()
    np.testing.assert_allclose(roi, full, rtol=0, atol=1e-5)


def test_roi_returns_zero_when_aabb_is_entirely_below_near_plane():
    directions = camera_rays(IMAGE_SHAPE, INTRINSICS, PIXEL_STRIDE)
    renderer = MeshRaycaster(trimesh.creation.box(extents=(0.0002, 0.0002, 0.0002)))
    pose = np.eye(4)
    pose[2, 3] = 0.0005
    assert renderer.render(directions, pose).any()
    with patch.object(renderer, "render", wraps=renderer.render) as render:
        roi = renderer.render_roi(
            directions, pose, INTRINSICS, PIXEL_STRIDE, 2 * PIXEL_STRIDE
        )

    render.assert_not_called()
    np.testing.assert_array_equal(roi, np.zeros(directions.shape[:2], dtype=np.float32))
