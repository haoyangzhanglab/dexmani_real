import json

import numpy as np
import pytest
import trimesh
from cordvip_geometry.assets import load_asset, prepare_asset

from scripts.cordvip.allocate_points import allocate_counts, allocate_task


@pytest.fixture
def source_task(tmp_path):
    task = tmp_path / "boxes"
    task.mkdir()
    for name, side in (("small", 1), ("large", 2)):
        mesh = trimesh.creation.box(extents=(side, side, side))
        mesh.apply_translation([4, -2, 3])
        source = tmp_path / f"{name}.obj"
        mesh.export(source)
        directory = prepare_asset(
            source,
            task / name,
            workspace=tmp_path,
            scale_to_meters=1,
            num_points=17,
            seed=11,
        )
        metadata = json.loads((directory / "asset.json").read_text())
        metadata.update(
            pointcloud_file="pointcloud_17.ply",
            source_image="reference.jpg",
            calibration={"source": "fixture", "measured_side_m": side},
            research_note=f"keep {name}",
        )
        (directory / "asset.json").write_text(json.dumps(metadata))
        (directory / "pointcloud_17.ply").write_bytes(b"old sampling must be replaced")
        (directory / "reference.jpg").write_bytes(b"reference image bytes")
        (directory / "notes.txt").write_text(name)
    return task


def test_counts_preserve_total_and_minimum_representation():
    counts = allocate_counts([1, 8])
    np.testing.assert_array_equal(counts, [115, 909])
    assert np.issubdtype(counts.dtype, np.integer)
    assert counts.sum() == 1024
    np.testing.assert_array_equal(allocate_counts([1, 100_000], total=3), [1, 2])


def test_counts_break_equal_remainders_in_input_order():
    np.testing.assert_array_equal(allocate_counts([1, 1, 1], total=5), [2, 2, 1])
    np.testing.assert_array_equal(allocate_counts([1, 8], total=2), [1, 1])


@pytest.mark.parametrize("volumes", [[], [0, 1], [-1, 1], [np.nan, 1], [np.inf, 1]])
def test_counts_reject_invalid_volumes(volumes):
    with pytest.raises(ValueError):
        allocate_counts(volumes)


@pytest.mark.parametrize("total", [2, 3.5, np.nan, True])
def test_counts_reject_invalid_budget(total):
    with pytest.raises(ValueError):
        allocate_counts([1, 2, 3], total=total)


def test_task_outputs_are_complete_consistent_assets(source_task):
    original = {
        (name, filename): (source_task / name / filename).read_bytes()
        for name in ("small", "large")
        for filename in (
            "asset.json",
            "points.npy",
            "normals.npy",
            "mesh/mesh_m.obj",
            "reference.jpg",
        )
    }
    output = allocate_task(source_task)
    assert output == source_task / "sampled_1024"
    allocation = json.loads((output / "allocation.json").read_text())
    assert allocation["source_task"] == str(source_task.resolve())
    assert allocation["total_points"] == 1024
    assert allocation["volume_method"] == "mesh"
    assert list(allocation["objects"]) == ["large", "small"]
    for name, volume, count in (("small", 1, 115), ("large", 8, 909)):
        source, destination = source_task / name, output / name
        metadata = json.loads((source / "asset.json").read_text())
        _, points, normals, sampled = load_asset(destination)
        assert allocation["objects"][name] == {"volume_m3": volume, "num_points": count}
        assert sampled["num_points"] == count and sampled["seed"] == 0
        assert sampled["calibration"] == metadata["calibration"]
        assert sampled["research_note"] == metadata["research_note"]
        assert sampled["source_image"] == "reference.jpg"
        assert sampled["source_mesh"] == str(
            (source / metadata["source_mesh"]).resolve()
        )
        assert sampled["points_file"] == "points.npy"
        assert sampled["normals_file"] == "normals.npy"
        assert sampled["pointcloud_file"] == "pointcloud.ply"
        assert points.shape == normals.shape == (count, 3)
        assert points.dtype == normals.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
        for relative in (metadata["mesh_file"], "reference.jpg", "notes.txt"):
            assert (destination / relative).read_bytes() == (
                source / relative
            ).read_bytes()
        assert not (destination / "pointcloud_17.ply").exists()
        ply = destination / "pointcloud.ply"
        lines = ply.read_text().splitlines()
        header_end = lines.index("end_header")
        assert lines[:3] == ["ply", "format ascii 1.0", f"element vertex {count}"]
        assert [line.split()[-1] for line in lines[3:header_end]] == [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
        ]
        values = np.loadtxt(ply, skiprows=header_end + 1)
        np.testing.assert_allclose(values[:, :3], points, atol=1e-6)
        np.testing.assert_allclose(values[:, 3:], normals, atol=1e-6)
    assert sum(item["num_points"] for item in allocation["objects"].values()) == 1024
    for (name, filename), contents in original.items():
        assert (source_task / name / filename).read_bytes() == contents


def test_task_rejects_overwrite_and_output_inside_source_assets(source_task):
    output = allocate_task(source_task)
    metadata = (output / "allocation.json").read_bytes()
    points = np.load(output / "small/points.npy", allow_pickle=False)
    with pytest.raises(FileExistsError):
        allocate_task(source_task)
    assert (output / "allocation.json").read_bytes() == metadata
    np.testing.assert_array_equal(
        np.load(output / "small/points.npy", allow_pickle=False), points
    )
    for source_asset in (source_task / "small", source_task / "small/resampled"):
        with pytest.raises(ValueError):
            allocate_task(source_task, source_asset)
    assert not (source_task / "small/resampled").exists()


def test_task_seed_is_reproducible_with_an_explicit_output(source_task):
    first = allocate_task(source_task)
    second = allocate_task(source_task, source_task / "other_sampling")
    assert list(json.loads((second / "allocation.json").read_text())["objects"]) == [
        "large",
        "small",
    ]
    for name in ("small", "large"):
        for filename in ("points.npy", "normals.npy"):
            np.testing.assert_array_equal(
                np.load(first / name / filename, allow_pickle=False),
                np.load(second / name / filename, allow_pickle=False),
            )


def test_task_rejects_open_mesh_without_box_volume_fallback(source_task):
    path = source_task / "small/mesh/mesh_m.obj"
    mesh = trimesh.load(path, process=False)
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    mesh.export(path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        allocate_task(source_task)
    assert not (source_task / "sampled_1024").exists()
    assert path.read_bytes() == original
