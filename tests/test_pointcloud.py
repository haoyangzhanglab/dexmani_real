"""Public point-cloud filtering and sampling on synthetic RGB-D; no sensor I/O."""

from dataclasses import replace

import numpy as np
import pytest

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry
from dexmani_real.sensor.pointcloud import build_point_cloud


@pytest.fixture
def scene():
    camera = CameraIntrinsics(32, 32, 100.0, 100.0, 16.0, 16.0, "none", (0.0,) * 5)
    transform = np.eye(4)
    transform[0, 3] = 0.4
    rows, columns = np.indices((32, 32))
    color = np.stack((columns * 7, rows * 7, np.full_like(rows, 180)), axis=-1).astype(np.uint8)
    return dict(
        depth_raw=np.full((32, 32), 500, np.uint16),
        color=color,
        depth_scale_m=0.001,
        geometry=RGBDGeometry(camera, camera, np.eye(4)),
        T_xarm_base_from_color=transform,
        table_plane_abcd=(0.0, 0.0, 1.0, -0.022),
    )


@pytest.mark.parametrize("remove_table", [False, True])
@pytest.mark.parametrize("num_points", [8, 2048])
@pytest.mark.parametrize("filtering", [False, True])
def test_planar_cloud_geometry_color_and_deterministic_sampling(
    scene, remove_table, num_points, filtering
):
    config = PointCloudConfig(
        remove_table=remove_table,
        num_points=num_points,
        voxel_size_m=0.001,
        outlier_min_neighbors=6 if filtering else 0,
        outlier_min_component_points=10 if filtering else 1,
    )
    result = build_point_cloud(**scene, config=config)
    assert result.shape == (num_points, 6)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    np.testing.assert_array_equal(result, build_point_cloud(**scene, config=config))
    # Analytic pinhole projection ties sampled XYZ to its original RGB pixel.
    columns = np.rint((result[:, 0] - 0.4) * 200 + 16).astype(int)
    rows = np.rint(result[:, 1] * 200 + 16).astype(int)
    assert ((columns >= 0) & (columns < 32) & (rows >= 0) & (rows < 32)).all()
    expected = np.column_stack(((columns - 16) * 0.005 + 0.4, (rows - 16) * 0.005))
    np.testing.assert_allclose(result[:, :2], expected, atol=1e-7)
    np.testing.assert_allclose(result[:, 2], 0.5)
    np.testing.assert_allclose(result[:, 3:], scene["color"][rows, columns] / 255, atol=1e-7)


@pytest.mark.parametrize("stage", ["depth", "workspace", "density", "component", "table"])
def test_empty_cloud_at_each_filter_returns_none(scene, stage):
    config = PointCloudConfig()
    if stage == "depth":
        scene["depth_raw"].fill(0)
    elif stage == "workspace":
        config = replace(config, workspace=(1.0, 1.0, 1.0, 2.0, 2.0, 2.0))
    elif stage == "density":
        config = replace(config, outlier_min_neighbors=10000)
    elif stage == "component":
        config = replace(config, outlier_min_component_points=10000)
    else:
        scene["table_plane_abcd"] = (0.0, 0.0, 1.0, -0.5)
    assert build_point_cloud(**scene, config=config) is None
    if stage == "table":
        assert build_point_cloud(**scene, config=replace(config, remove_table=False)) is not None


@pytest.mark.parametrize(
    "overrides",
    [
        {"table_core_height_m": float("nan")},
        {"table_core_height_m": -0.001},
        {"table_object_seed_height_m": 0.001},
        {"table_object_seed_min_pixels": 0},
    ],
)
def test_public_boundary_rejects_invalid_table_recipe_even_without_depth(scene, overrides):
    scene["depth_raw"].fill(0)
    with pytest.raises(ValueError):
        build_point_cloud(**scene, config=PointCloudConfig(**overrides))


@pytest.mark.parametrize("plane", [None, (0, 0, 1), (0, 0, -1, 0), (0, 0, 1, np.nan)])
def test_public_boundary_rejects_invalid_plane(scene, plane):
    scene["table_plane_abcd"] = plane
    with pytest.raises(ValueError):
        build_point_cloud(**scene, config=PointCloudConfig())
