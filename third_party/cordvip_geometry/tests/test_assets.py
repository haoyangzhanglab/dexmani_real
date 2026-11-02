import json

import numpy as np
import pytest
import trimesh
from cordvip_geometry.assets import load_asset, prepare_asset
from cordvip_geometry.geometry import sample_mesh_surface


@pytest.fixture
def source_mesh(tmp_path):
    mesh = trimesh.creation.box(extents=(2, 3, 4))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.43, [1, 2, 3]))
    mesh.apply_translation([4, -3, 2])
    path = tmp_path / "source.obj"
    mesh.export(path)
    return path, mesh


@pytest.fixture
def asset(tmp_path, source_mesh):
    source, _ = source_mesh
    return prepare_asset(
        source, tmp_path / "asset", workspace=tmp_path, scale_to_meters=0.1
    )


def test_prepare_preserves_origin_orientation_and_metric_scale(asset, source_mesh):
    _, original = source_mesh
    mesh, points, normals, metadata = load_asset(asset)
    np.testing.assert_allclose(mesh.vertices, original.vertices * 0.1, atol=1e-8)
    np.testing.assert_allclose(mesh.bounds, original.bounds * 0.1, atol=1e-8)
    assert metadata["units"] == "m" and metadata["frame"] == "object"
    assert points.shape == normals.shape == (1024, 3)
    assert points.dtype == normals.dtype == np.float32
    assert not points.flags.writeable and not normals.flags.writeable


def test_extent_and_scale_share_reproducible_surface_samples(tmp_path, source_mesh):
    source, original = source_mesh
    scaled = prepare_asset(
        source,
        tmp_path / "scaled",
        workspace=tmp_path,
        scale_to_meters=0.1,
        num_points=512,
        seed=7,
    )
    measured = prepare_asset(
        source,
        tmp_path / "measured",
        workspace=tmp_path,
        size_axis="z",
        size_meters=np.ptp(original.vertices[:, 2]) * 0.1,
        num_points=512,
        seed=7,
    )
    _, points_a, normals_a, _ = load_asset(scaled)
    _, points_b, normals_b, _ = load_asset(measured)
    np.testing.assert_allclose(points_a, points_b, atol=1e-7)
    np.testing.assert_array_equal(normals_a, normals_b)
    again = sample_mesh_surface(load_asset(scaled)[0], 512, seed=7)
    np.testing.assert_allclose(points_a, again[0], atol=1e-7)
    np.testing.assert_allclose(normals_a, again[1], atol=1e-7)


def test_surface_normals_correspond_to_the_sampled_faces():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [0, 1, 0], [5, 0, 0], [5, 2, 0], [5, 0, 1]],
        faces=[[0, 1, 2], [3, 4, 5]],
        process=False,
    )
    points, normals = sample_mesh_surface(mesh, 4096, seed=3)
    vertical = points[:, 0] > 4
    assert 0.45 < vertical.mean() < 0.55
    np.testing.assert_array_equal(
        normals[vertical], np.tile([1, 0, 0], (vertical.sum(), 1))
    )
    np.testing.assert_array_equal(
        normals[~vertical], np.tile([0, 0, 1], ((~vertical).sum(), 1))
    )
    assert np.all(points[vertical, 1] / 2 + points[vertical, 2] <= 1)
    assert np.all(points[~vertical, 0] / 2 + points[~vertical, 1] <= 1)
    np.testing.assert_array_equal(points[~vertical, 2], 0)
    np.testing.assert_array_equal(points[vertical, 0], 5)


@pytest.mark.parametrize(
    "key,value",
    [("units", "mm"), ("frame", "camera"), ("points_file", "../outside.npy")],
)
def test_asset_metadata_rejects_incompatible_geometry(asset, key, value):
    path = asset / "asset.json"
    metadata = json.loads(path.read_text())
    metadata[key] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_asset(asset)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite", "nonunit"])
def test_asset_rejects_invalid_corresponding_normals(asset, invalid):
    path = asset / "normals.npy"
    normals = np.load(path, allow_pickle=False)
    if invalid == "shape":
        normals = normals[:-1]
    elif invalid == "nonfinite":
        normals[0, 0] = np.nan
    else:
        normals *= 2
    np.save(path, normals, allow_pickle=False)
    with pytest.raises(ValueError):
        load_asset(asset)


def test_prepare_never_overwrites_an_existing_asset(asset, source_mesh, tmp_path):
    source, _ = source_mesh
    original = {
        path.relative_to(asset): path.read_bytes()
        for path in asset.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        prepare_asset(source, asset, workspace=tmp_path, scale_to_meters=0.2, seed=9)
    for relative, contents in original.items():
        assert (asset / relative).read_bytes() == contents


@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {"scale_to_meters": 0},
        {"scale_to_meters": -1},
        {"scale_to_meters": np.nan},
        {"scale_to_meters": "1"},
        {"size_axis": "x"},
        {"size_meters": 0.1},
        {"size_axis": "xy", "size_meters": 0.1},
        {"size_axis": "x", "size_meters": -0.1},
        {"scale_to_meters": 1, "size_axis": "x", "size_meters": 0.1},
        {"scale_to_meters": 1, "num_points": 0},
        {"scale_to_meters": 1, "num_points": 1.5},
        {"scale_to_meters": 1, "num_points": np.nan},
        {"scale_to_meters": 1, "seed": -1},
        {"scale_to_meters": 1, "seed": 1.5},
    ],
)
def test_prepare_rejects_invalid_public_parameters_before_writing(
    tmp_path, source_mesh, parameters
):
    source, _ = source_mesh
    output = tmp_path / "invalid_asset"
    with pytest.raises(ValueError):
        prepare_asset(source, output, workspace=tmp_path, **parameters)
    assert not output.exists()


def test_prepare_accepts_numpy_numeric_parameters(tmp_path, source_mesh):
    source, _ = source_mesh
    output = prepare_asset(
        source,
        tmp_path / "numpy_parameters",
        workspace=tmp_path,
        scale_to_meters=np.float64(0.1),
        num_points=np.int64(32),
        seed=np.int64(2),
    )
    _, points, _, metadata = load_asset(output)
    assert points.shape == (32, 3)
    assert metadata["num_points"] == 32 and metadata["seed"] == 2
