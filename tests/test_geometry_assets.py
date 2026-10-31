"""几何共享光线、米制采样与资产读写；无需 GPU 模型。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"))

from cordvip_geometry.cli.prepare_object import prepare_mesh
from cordvip_geometry.geometry import load_metric_mesh, sample_mesh_surface
from cordvip_geometry.io import load_asset
from cordvip_geometry.model_depth import ModelDepthRenderer, camera_rays


class ModelDepthTests(unittest.TestCase):
    def test_shared_rays_keep_camera_z_and_global_stride_grid(self):
        shape = (40, 41)
        intrinsics = np.array([[60., 0., 19.5], [0., 60., 19.5], [0., 0., 1.]])
        for stride in (1, 2, 3):
            with self.subTest(stride=stride):
                directions = camera_rays(shape, intrinsics, stride)
                self.assertFalse(directions.flags.writeable)
                original = directions.copy()
                for extent, z in ((.2, .6), (.1, .8)):
                    mesh = trimesh.creation.box(extents=[extent] * 3)
                    renderer = ModelDepthRenderer(mesh, pixel_stride=stride)
                    pose = np.eye(4)
                    pose[2, 3] = z
                    depth = renderer.render(shape, intrinsics, pose, directions=directions)
                    # 对准光轴的立方体只显示前表面；t_hit 应为相机 Z，而非射线长度。
                    front = z - extent / 2
                    v, u = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
                    expected = np.zeros(shape, np.float32)
                    hits = ((np.abs((u - 19.5) * front / 60) < extent / 2)
                            & (np.abs((v - 19.5) * front / 60) < extent / 2))
                    expected[::stride, ::stride] = np.where(hits, front, 0)
                    np.testing.assert_allclose(depth, expected, atol=1e-7, rtol=0)
                    np.testing.assert_array_equal(directions, original)


class MetricAssetTests(unittest.TestCase):
    def test_final_metric_sampling_is_once_and_preserves_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "unscaled.obj"
            mesh = trimesh.creation.box(extents=[2., 3., 4.])
            mesh.apply_translation([1., -2., .5])
            mesh.export(source)
            original_bytes = source.read_bytes()
            expected_mesh = load_metric_mesh(source)
            expected_mesh.apply_scale(.01)
            expected_points, expected_normals = sample_mesh_surface(expected_mesh, 64, 17)
            for name, scaling in (("factor", dict(scale_to_meters=.01)),
                                  ("extent", dict(size_axis="x", size_meters=.02))):
                with self.subTest(name=name), patch(
                    "cordvip_geometry.geometry.sample_mesh_surface", wraps=sample_mesh_surface,
                ) as sampling:
                    output = prepare_mesh(source, root / name, workspace=root,
                                          num_points=64, seed=17, **scaling)
                    sampling.assert_called_once()
                    final_mesh, points, normals, metadata = load_asset(output)
                    np.testing.assert_array_equal(points, expected_points)
                    np.testing.assert_array_equal(normals, expected_normals)
                    np.testing.assert_allclose(final_mesh.bounds, expected_mesh.bounds, atol=1e-10, rtol=0)
                    self.assertEqual(metadata["units"], "m")
                    self.assertEqual(metadata["frame"], "object")
                    self.assertEqual(metadata["seed"], 17)
                    self.assertEqual(metadata["num_points"], 64)
                    self.assertEqual(json.loads((output / "asset.json").read_text())["scale_to_meters"], .01)
                    self.assertEqual(source.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
