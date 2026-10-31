import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"))
from cordvip_geometry.cli import build_object
from cordvip_geometry.cli.arguments import parse_object_args
from cordvip_geometry.sam2_config import SAM_MODELS, add_sam_arguments, resolve_sam_arguments


class ObjectBuildTests(unittest.TestCase):
    def sam_args(self, *argv):
        parser = argparse.ArgumentParser()
        add_sam_arguments(parser, Path("/workspace"))
        return resolve_sam_arguments(parser.parse_args(argv), Path("/workspace"))

    def test_official_variant_selects_one_matching_pair(self):
        for variant in SAM_MODELS:
            with self.subTest(variant=variant):
                args = self.sam_args("--sam-variant", variant)
                self.assertEqual(args.sam_config, SAM_MODELS[variant]["config"])
                self.assertEqual(args.sam_checkpoint.name, SAM_MODELS[variant]["weight"])
        self.assertEqual(self.sam_args().sam_variant, "small")

    def test_custom_pair_must_be_complete_and_has_no_variant_source(self):
        args = self.sam_args("--sam-config", "custom.yaml", "--sam-checkpoint", "custom.pt")
        self.assertEqual((args.sam_config, args.sam_checkpoint), ("custom.yaml", Path("custom.pt")))
        for argv in (("--sam-config", "custom.yaml"), ("--sam-checkpoint", "custom.pt"),
                     ("--sam-variant", "small", "--sam-config", "custom.yaml", "--sam-checkpoint", "custom.pt")):
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError):
                    self.sam_args(*argv)

    def build_args(self, directory, *, scale=True):
        argv = ["build", "--image", "unused.png", "--text", "cup.",
                "--headless", "--instance", "0", "--output-dir", str(Path(directory) / "build")]
        if scale:
            argv += ["--scale-to-meters", ".01"]
        return parse_object_args(argv, workspace=directory)

    def test_build_calls_stages_directly_and_isolates_only_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_args(directory)
            order = []
            process = Mock(pid=1, exitcode=0)
            process.start.side_effect = lambda: order.append("generate")
            process.join.side_effect = lambda: order.append("join")
            context = Mock()
            context.Process.return_value = process
            with patch.object(build_object, "segment", side_effect=lambda *a, **k: order.append("segment")), \
                    patch.object(build_object.mp, "get_context", return_value=context), \
                    patch("cordvip_geometry.cli.prepare_object.prepare_mesh", side_effect=lambda *a, **k: order.append("prepare")) as prepare, \
                    contextlib.redirect_stdout(io.StringIO()):
                build_object.build(args)
            self.assertEqual(order, ["segment", "generate", "join", "prepare"])
            self.assertIs(context.Process.call_args.kwargs["target"], build_object.generate)
            parameters = context.Process.call_args.kwargs["kwargs"]
            self.assertNotIn("num_points", parameters)
            self.assertFalse(any(key.startswith("sam") for key in parameters))
            self.assertNotIn("scale_to_meters", parameters)
            self.assertEqual(prepare.call_args.kwargs["scale_to_meters"], .01)
            self.assertEqual(prepare.call_args.kwargs["num_points"], args.num_points)

    def test_failed_generation_does_not_prepare_an_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_args(directory)
            process = Mock(pid=1, exitcode=7)
            context = Mock()
            context.Process.return_value = process
            with patch.object(build_object, "segment"), \
                    patch.object(build_object.mp, "get_context", return_value=context), \
                    patch("cordvip_geometry.cli.prepare_object.prepare_mesh") as prepare:
                with self.assertRaisesRegex(RuntimeError, "exitcode=7"):
                    build_object.build(args)
            prepare.assert_not_called()

    def test_build_without_scale_has_no_tracking_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.build_args(directory, scale=False)
            context = Mock()
            context.Process.return_value = Mock(pid=1, exitcode=0)
            with patch.object(build_object, "segment"), \
                    patch.object(build_object.mp, "get_context", return_value=context), \
                    patch("cordvip_geometry.cli.prepare_object.prepare_mesh") as prepare, \
                    contextlib.redirect_stdout(io.StringIO()):
                build_object.build(args)
            prepare.assert_not_called()
            self.assertFalse((args.output_dir / "asset").exists())

    def test_generation_outputs_mesh_without_unscaled_sampling_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection"
            selection.mkdir()
            (selection / "selection.json").write_text(json.dumps(dict(
                selection="explicit_headless_instance", selected_instance=0,
            )))
            Image.fromarray(np.zeros((4, 4, 4), dtype=np.uint8)).save(selection / "foreground_rgba.png")

            def generate_mesh(**kwargs):
                kwargs["output_mesh_path"].write_text("model mesh")
                return dict(units="unscaled", seed=kwargs["seed"])

            output = root / "generated"
            with patch("cordvip_geometry.reconstruction.generate_mesh", side_effect=generate_mesh), \
                    patch("cordvip_geometry.geometry.sample_mesh_surface") as sample, \
                    contextlib.redirect_stdout(io.StringIO()):
                build_object.generate(selection, output, workspace=root,
                                      hunyuan_root=root / "model", weights=root / "weights", seed=7)
            sample.assert_not_called()
            self.assertEqual({path.name for path in output.iterdir()}, {
                "mesh_unscaled.obj", "generation_request.json", "generation_result.json",
            })
            metadata = json.loads((output / "generation_result.json").read_text())
            self.assertEqual(metadata["seed"], 7)
            for removed in ("points_file", "normals_file", "num_points", "sampling", "normals_method"):
                self.assertNotIn(removed, metadata)


if __name__ == "__main__":
    unittest.main()
