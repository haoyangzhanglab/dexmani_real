"""合成 RGB-D 的几何、滤波和稀疏采样契约；不依赖设备或 GPU。"""

import numpy as np
import pytest

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.geometry.camera import CameraIntrinsics, deproject_depth
from dexmani_real.perception.pointcloud import (
    PointCloudProcessor,
    build_raw_point_cloud,
    filter_depth,
    radius_inliers,
    sample_indices,
    voxel_downsample,
)


def intrinsics(size=32, focal=100.0):
    return CameraIntrinsics(
        size,
        size,
        focal,
        focal,
        (size - 1) / 2,
        (size - 1) / 2,
        "none",
        (0, 0, 0, 0, 0),
    )


def processor(*, size=32, transform=None, **config):
    return PointCloudProcessor(
        depth_scale_m=0.001,
        intrinsics=intrinsics(size),
        T_xarm_base_from_color=np.eye(4) if transform is None else transform,
        table_plane_abcd=(0, 0, 1, -0.18) if config.get("remove_table") else None,
        config=PointCloudConfig(
            **{
                "remove_table": False,
                "num_points": 64,
                "workspace": (-2, -2, -2, 2, 2, 2),
                **config,
            }
        ),
    )


def test_deprojection_plane_and_optical_axes():
    depth = np.ones((3, 3), np.float32)
    points, rows, columns = deproject_depth(depth, depth > 0, intrinsics(3, 2))
    expected = np.column_stack(((columns - 1) / 2, (rows - 1) / 2, np.ones(9)))
    np.testing.assert_array_equal(points, expected)
    # Optical RDF：右侧像素 +X、下方像素 +Y，平面深度为 +Z。
    np.testing.assert_array_equal(points[[4, 5, 7]], [[0, 0, 1], [0.5, 0, 1], [0, 0.5, 1]])
    assert points.dtype == np.float32


def test_raw_cloud_applies_camera_to_base_once_and_keeps_rgb():
    depth = np.full((3, 3), 600, np.uint16)
    depth[0, 0] = 0
    rgb = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    transform = np.array([[0, 0, 1, 0.2], [1, 0, 0, -0.1], [0, 1, 0, 0.3], [0, 0, 0, 1]])
    cloud = build_raw_point_cloud(
        depth_raw=depth,
        color=rgb,
        depth_scale_m=0.001,
        intrinsics=intrinsics(3, 2),
        T_xarm_base_from_color=transform,
    )
    rows, columns = np.nonzero(depth)
    expected = np.column_stack(
        (
            np.full(8, 0.8),
            (columns - 1) * 0.3 - 0.1,
            (rows - 1) * 0.3 + 0.3,
        )
    )
    np.testing.assert_allclose(cloud[:, :3], expected, atol=1e-7)
    np.testing.assert_allclose(cloud[:, 3:], rgb[rows, columns] / 255, atol=1e-7)
    assert cloud.shape == (8, 6) and cloud.dtype == np.float32


def test_depth_filter_keeps_plane_and_removes_edge_flyer():
    depth = np.full((30, 30), 0.6, np.float32)
    assert filter_depth(depth).all()
    depth[:, 15:] = 0.9
    depth[15, 14] = 0.75
    keep = filter_depth(depth)
    assert not keep[15, 14]
    assert keep[15, 8] and keep[15, 22]
    isolated = np.zeros_like(depth)
    isolated[15, 15] = 0.6
    assert not filter_depth(isolated).any()


@pytest.mark.parametrize("cardinal", range(5))
@pytest.mark.parametrize("diagonal", range(5))
@pytest.mark.parametrize("edge", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_depth_filter_neighbor_support_thresholds(cardinal, diagonal, edge, dtype):
    depth = np.full((5, 5), 0.9 if edge else 0, dtype=dtype)
    depth[2, 2] = 0.6
    for row, column in ((1, 2), (3, 2), (2, 1), (2, 3))[:cardinal]:
        depth[row, column] = 0.6
    for row, column in ((1, 1), (1, 3), (3, 1), (3, 3))[:diagonal]:
        depth[row, column] = 0.6
    before = depth.copy()
    expected = (cardinal == 4 and diagonal >= 1) if edge else cardinal + diagonal >= 2
    actual = filter_depth(depth)
    assert actual.dtype == bool and actual[2, 2] == expected
    np.testing.assert_array_equal(depth, before)


def test_voxel_downsample_averages_xyz_and_rgb_across_negative_cells():
    xyz = np.array([[-0.004, 0, 0], [-0.002, 0, 0], [0.001, 0, 0], [0.003, 0, 0]], np.float32)
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32)
    points, colors = voxel_downsample(xyz, rgb)
    np.testing.assert_allclose(points, [[-0.003, 0, 0], [0.002, 0, 0]], atol=1e-9)
    np.testing.assert_array_equal(colors, [[0.5, 0.5, 0], [0.5, 0.5, 1]])
    assert points.dtype == colors.dtype == np.float32


def test_radius_removal_rejects_isolated_points_and_small_dense_fragments():
    x, y = np.meshgrid(np.arange(5) * 0.004, np.arange(5) * 0.004)
    cluster = np.column_stack((x.ravel(), y.ravel(), np.zeros(25)))
    fragment = np.column_stack((0.3 + np.arange(7) * 0.0001, np.zeros((7, 2))))
    points = np.vstack((cluster, fragment, [1, 1, 1])).astype(np.float32)
    np.testing.assert_array_equal(radius_inliers(points), [True] * 25 + [False] * 8)


@pytest.mark.parametrize("num_points", [64, 2048])
def test_processor_plane_fps_and_sparse_fill(num_points):
    depth = np.full((32, 32), 600, np.uint16)
    rgb = np.full((32, 32, 3), [255, 128, 0], np.uint8)
    cloud = processor(num_points=num_points).process(depth, rgb)
    assert cloud.shape == (num_points, 6) and cloud.dtype == np.float32
    np.testing.assert_allclose(cloud[:, 2], 0.6, atol=1e-7)
    np.testing.assert_allclose(cloud[:, 3:], np.broadcast_to([1, 128 / 255, 0], (num_points, 3)))
    unique = np.unique(cloud[:, :3], axis=0)
    if num_points == 2048:
        # 重复填充只满足输出点数；独立几何点数没有随之增加。
        assert 0 < len(unique) <= depth.size < num_points
        np.testing.assert_array_equal(cloud[len(unique) :], cloud[: num_points - len(unique)])
    else:
        assert len(unique) == num_points


def test_processor_crops_workspace_and_table_in_base_frame():
    transform = np.array(
        [
            [1, 0, 0, 0.4],
            [0, 0, -1, 1.2],
            [0, 1, 0, 0.2],
            [0, 0, 0, 1],
        ]
    )
    depth = np.full((32, 32), 600, np.uint16)
    rgb = np.zeros((32, 32, 3), np.uint8)
    cloud = processor(
        transform=transform,
        remove_table=True,
        workspace=(0.35, 0.59, 0.10, 0.45, 0.61, 0.28),
    ).process(depth, rgb)
    assert cloud.shape == (64, 6)
    assert np.all((cloud[:, 0] >= 0.35) & (cloud[:, 0] <= 0.45))
    np.testing.assert_allclose(cloud[:, 1], 0.6, atol=1e-7)
    assert np.all((cloud[:, 2] > 0.19) & (cloud[:, 2] <= 0.28))


@pytest.mark.parametrize("case", ["zero", "isolated", "outside", "after_radius"])
def test_processor_returns_empty_when_no_valid_geometry(case):
    depth = np.zeros((32, 32), np.uint16)
    rgb = np.zeros((32, 32, 3), np.uint8)
    config = {}
    if case == "isolated":
        depth[16, 16] = 600
    elif case == "outside":
        depth[:] = 600
        config["workspace"] = (0, 0, 1, 1, 1, 2)
    elif case == "after_radius":
        depth[15:18, 15:18] = 600
    cloud = processor(**config).process(depth, rgb)
    assert cloud.shape == (0, 6) and cloud.dtype == np.float32


def test_sample_indices_preserves_deterministic_fps_and_repeat_semantics():
    xyz = np.random.default_rng(42).normal(size=(128, 3)).astype(np.float32)
    selected = sample_indices(xyz, 32)
    assert selected.shape == (32,) and len(np.unique(selected)) == 32
    np.testing.assert_array_equal(selected, sample_indices(xyz, 32))
    np.testing.assert_array_equal(sample_indices(xyz[:3], 8), [0, 1, 2, 0, 1, 2, 0, 1])


@pytest.mark.parametrize("model,coeffs", [
    ("none", (0.,) * 5), ("brown_conrady", (0.,) * 5),
    ("distortion.none", (.1, 0, 0, 0, 0)),
])
def test_identity_image_grid_reuses_input(model, coeffs):
    import cv2
    from dexmani_real.geometry.camera import init_undistort_maps, remap_image

    camera = CameraIntrinsics(4, 3, 4., 4., 2., 1., model, coeffs)
    maps = init_undistort_maps(camera)
    assert maps is None
    for image, method in (
        (np.zeros((3, 4, 3), np.uint8), cv2.INTER_LINEAR),
        (np.zeros((3, 4), np.uint16), cv2.INTER_NEAREST),
    ):
        assert remap_image(image, maps, interpolation=method) is image


@pytest.mark.parametrize("distorted", [False, True])
@pytest.mark.parametrize("output_hw", [(480, 640), (224, 320)])
def test_real_cloud_policy_projection_on_pinhole_grid(distorted, output_hw):
    import cv2
    import torch
    from scipy.spatial.transform import Rotation
    from dexmani_real.geometry.camera import camera_extrinsic_cv, init_undistort_maps, remap_image
    from dexmani_policy.agents.obs_encoder.pointcloud.camera_projection import project_to_image
    from dexmani_policy.agents.obs_encoder.rgb.image_processor import ImageProcessor

    K = np.array([[610., 0, 319.5], [0, 612., 239.5], [0, 0, 1]])
    D = np.array([.05, .01, .001, -.001, .002]) if distorted else np.zeros(5)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [.2, -.4, .3]).as_matrix()
    T[:3, 3] = [.25, -.35, .8]
    camera = CameraIntrinsics(640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2], "brown_conrady", tuple(D))
    y, x = np.indices((480, 640))
    rgb = np.stack((x // 4, y // 3, (x + y) // 5), axis=-1).astype(np.uint8)
    before = rgb.copy()
    maps = init_undistort_maps(camera)
    image = remap_image(rgb, maps, interpolation=cv2.INTER_LINEAR)
    processor = ImageProcessor(image_size=output_hw, image_mean=(0, 0, 0), image_std=(1, 1, 1))
    _, affine = processor.process_images(torch.from_numpy(image).permute(2, 0, 1)[None], return_transform=True)
    pixels = np.array([[0, 0], [639, 0], [320, 240], [639, 240], [0, 479], [639, 479]])
    for depth_value in (350, 800, 1400):
        depth = np.zeros((480, 640), np.uint16)
        depth[pixels[:, 1], pixels[:, 0]] = depth_value
        cloud = build_raw_point_cloud(
            depth_raw=depth, color=rgb, depth_scale_m=.001,
            intrinsics=camera, T_xarm_base_from_color=T,
        )
        rows, columns = np.nonzero(depth)
        expected = cv2.undistortPointsIter(
            np.stack((columns, rows), axis=-1).astype(np.float64)[:, None], K, D, np.eye(3), K,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-12),
        )[:, 0]
        expected = (expected + .5) * np.array(output_hw[::-1]) / [640, 480] - .5
        projection = project_to_image(
            torch.from_numpy(cloud[None, :, :3].copy()), torch.tensor(K, dtype=torch.float32)[None],
            torch.from_numpy(camera_extrinsic_cv(T).reshape(1, 3, 4)), affine, output_hw,
        )
        actual = (projection.grid_xy.numpy()[0] + 1) * np.array(output_hw[::-1]) / 2 - .5
        assert projection.valid.all()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)
        np.testing.assert_allclose(cloud[:, 3:], rgb[rows, columns] / 255, atol=1e-7)
    if distorted:
        # 单独判断 OpenCV 双线性插值表的量化误差，不混入投影误差。
        ramp = np.stack((x, y), axis=-1).astype(np.float32)
        sampled = remap_image(ramp, maps, interpolation=cv2.INTER_LINEAR)
        valid = (maps[0] >= 0) & (maps[0] < 639) & (maps[1] >= 0) & (maps[1] < 479)
        # 从针孔目标像素独立正投影至原始畸变网格，检查映射的 K、尺寸和光轴。
        rays = np.column_stack((
            (x[valid] - K[0, 2]) / K[0, 0],
            (y[valid] - K[1, 2]) / K[1, 1],
            np.ones(valid.sum()),
        ))
        source_pixels = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, D)[0][:, 0]
        np.testing.assert_allclose(sampled[valid], source_pixels, rtol=0, atol=1/64 + 1e-4)
    np.testing.assert_array_equal(rgb, before)
