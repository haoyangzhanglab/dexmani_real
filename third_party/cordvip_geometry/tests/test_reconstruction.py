import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh
from cordvip_geometry.geometry import load_mesh
from cordvip_geometry.reconstruction import generate_mesh
from PIL import Image


@pytest.mark.parametrize(
    "parameters",
    [
        {"seed": -1},
        {"seed": 0.5},
        {"steps": 0},
        {"steps": np.nan},
        {"octree_resolution": 15},
        {"num_chunks": 0},
        {"num_chunks": "8000"},
    ],
)
def test_generation_rejects_invalid_parameters_before_loading_models(
    tmp_path, parameters
):
    output = tmp_path / "mesh_unscaled.obj"
    with pytest.raises(ValueError):
        generate_mesh(
            "missing.png", "missing_source", "missing_weights", output, **parameters
        )
    assert not output.exists()


@pytest.mark.parametrize("dense", [False, True])
def test_generation_exports_reduced_mesh_and_reports_actual_geometry(
    tmp_path, monkeypatch, dense
):
    mesh = trimesh.creation.icosphere(subdivisions=6 if dense else 1)
    mesh.apply_translation([0.3, -0.2, 0.1])
    original_vertices, original_faces = mesh.vertices.copy(), mesh.faces.copy()
    source = tmp_path / "source"
    pipeline_source = source / "hy3dshape/hy3dshape/pipelines.py"
    pipeline_source.parent.mkdir(parents=True)
    pipeline_source.write_text("stub")
    weights = tmp_path / "weights/hunyuan3d-dit-v2-1"
    weights.mkdir(parents=True)
    (weights / "model.fp16.ckpt").write_bytes(b"stub")
    (weights / "config.yaml").write_text(
        "conditioner: {params: {main_image_encoder: {kwargs: {config: {}}}}}"
    )
    rgba = np.zeros((12, 12, 4), np.uint8)
    rgba[3:9, 3:9] = 255
    image = tmp_path / "foreground.png"
    Image.fromarray(rgba).save(image)
    pipeline = MagicMock(return_value=[mesh])
    factory = MagicMock()
    factory.from_single_file.return_value = pipeline
    torch = SimpleNamespace(
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: True, init=lambda: None),
        float16="float16",
        Generator=MagicMock(),
        inference_mode=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "hy3dshape",
        SimpleNamespace(Hunyuan3DDiTFlowMatchingPipeline=factory),
    )
    output = tmp_path / "mesh_unscaled.obj"
    with patch.dict(os.environ), patch.object(sys, "path", sys.path.copy()):
        result = generate_mesh(image, source, weights.parent, output)
    exported = load_mesh(output)
    assert result["source_faces"] == len(original_faces)
    assert result["faces"] == len(exported.faces) <= 20_000
    assert result["vertices"] == len(exported.vertices)
    assert result["mesh_simplification"] == ("quadric_decimation" if dense else "none")
    np.testing.assert_allclose(result["bounds_unscaled"], exported.bounds, atol=1e-7)
    np.testing.assert_allclose(exported.bounds, mesh.bounds, atol=0.002)
    np.testing.assert_array_equal(mesh.vertices, original_vertices)
    np.testing.assert_array_equal(mesh.faces, original_faces)
