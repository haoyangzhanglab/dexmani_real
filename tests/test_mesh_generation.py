import contextlib
import importlib.metadata
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"))
from cordvip_geometry.reconstruction import generate_mesh


class MeshGenerationTests(unittest.TestCase):
    def generate_with_fake_model(self, directory, *, export_error=None):
        root = Path(directory)
        package = root / "source" / "hy3dshape" / "hy3dshape"
        package.mkdir(parents=True)
        (package / "pipelines.py").write_text("# fake model module\n")
        weights = root / "weights"
        weights.mkdir()
        (weights / "model.fp16.ckpt").write_bytes(b"fake model, never loaded")
        (weights / "config.yaml").write_text(
            "conditioner:\n  params:\n    main_image_encoder:\n      kwargs:\n        config: {}\n"
        )
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[2:6, 2:6] = 255
        image = root / "image.png"
        Image.fromarray(rgba).save(image)
        mesh = SimpleNamespace(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            faces=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
            extents=np.ones(3), bounds=np.array([[0, 0, 0], [1, 1, 1]]),
            export=mock.Mock(return_value="mesh payload", side_effect=export_error),
        )
        component = mock.Mock()
        pipeline = mock.Mock(return_value=[mesh])
        pipeline.model = pipeline.vae = pipeline.conditioner = component
        hy3dshape = ModuleType("hy3dshape")
        hy3dshape.__file__ = str(package / "__init__.py")
        hy3dshape.Hunyuan3DDiTFlowMatchingPipeline = SimpleNamespace(
            from_single_file=mock.Mock(return_value=pipeline)
        )
        target = SimpleNamespace(type="cuda", index=0)
        torch = ModuleType("torch")
        torch.device = mock.Mock(return_value=target)
        torch.float16 = "float16"
        torch.inference_mode = contextlib.nullcontext
        torch.Generator = mock.Mock()
        forbidden = mock.Mock(side_effect=RuntimeError("optional memory report unavailable"))
        torch.cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            init=mock.Mock(), empty_cache=mock.Mock(), device=lambda _: contextlib.nullcontext(),
            reset_peak_memory_stats=forbidden, max_memory_allocated=forbidden,
            max_memory_reserved=forbidden,
        )
        output = root / "mesh.obj"
        with mock.patch.dict(sys.modules, {"torch": torch, "hy3dshape": hy3dshape}), mock.patch.object(
            importlib.metadata, "version", side_effect=RuntimeError("optional version report unavailable")
        ) as version:
            result = generate_mesh(image, root / "source", weights, output)
            forbidden.assert_not_called()
            version.assert_not_called()
        return result, output

    def test_optional_reports_cannot_break_mesh_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self.generate_with_fake_model(directory)
            self.assertEqual(output.read_text(), "mesh payload")
            self.assertEqual(result["seed"], 0)
            self.assertEqual(result["steps"], 50)
            self.assertEqual(result["weights_directory"], str(Path(directory) / "weights"))
            self.assertNotIn("versions", result)
            self.assertNotIn("peak_allocated_bytes", result)

    def test_real_mesh_write_error_remains_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, "mesh export failed"):
                self.generate_with_fake_model(directory, export_error=OSError("mesh export failed"))
            self.assertFalse((Path(directory) / "mesh.obj").exists())


if __name__ == "__main__":
    unittest.main()
