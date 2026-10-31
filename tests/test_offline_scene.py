"""Raw 顺序离线流程、首帧来源与 observed/pose 独立语义。"""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dexmani_real.data.raw import RawReader, RawWriter
from test_recorder import raw_metadata, raw_row

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"))
from cordvip_geometry.cli.arguments import parse_tracking_args
from cordvip_geometry.cli.scene_tracking import run
from cordvip_geometry.integrations.dexmani import iter_raw_rows
from cordvip_geometry.io import load_initial_selection, write_confirmed_selection
from cordvip_geometry.rgbd import Frame, save_frame
from cordvip_geometry.scene_tracking import SceneTracker, scene_arrays
import cordvip_geometry.cli.check_tracking
import cordvip_geometry.sam2_config
import cordvip_geometry.segmentation


class Segmenter:
    """固定 mask 的局部模型边界 fake；Raw、编码、点云与 FK 走真实实现。"""

    provenance = {"model": "local test fake"}
    last_metrics = {}

    def __init__(self, *args, **kwargs):
        self.rows = []

    def start(self):
        return self

    def initialize(self, rgb, masks, *, frame_key):
        self.masks = masks
        self.rows.append(frame_key["row_index"])
        return masks

    def track(self, rgb, *, frame_key):
        self.rows.append(frame_key["row_index"])
        return self.masks

    def close(self):
        pass


def record_episode(path):
    path.mkdir()
    writer = RawWriter(path, control_hz=16)
    try:
        writer.open(dict(raw_metadata(), task_label="test", collection_source="policy_rollout"))
        for index in range(3):
            writer.append_frame(raw_row(index), None)
        writer.finish("operator", (), expected_frames=3, technical_status="normal")
    finally:
        writer.close()


def save_selection(directory, frame, mask):
    directory.mkdir()
    save_frame(directory / "source_rgbd.npz", frame)
    write_confirmed_selection(directory, mask, dict(
        selection="gui_confirmed", selected_instance=None,
        mask_grid="original_rgb_grid; not_automatically_rectified",
        source_id=frame.source_id, timestamp_ns=frame.timestamp_ns, clock=frame.clock,
    ), mask_file="mask.npy")


class OfflineSceneTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.episode = self.root / "episode"
        record_episode(self.episode)
        with RawReader(self.episode) as reader:
            self.first, _ = next(iter_raw_rows(reader, reader.read_camera()))
        self.mask = np.ones((16, 16), bool)
        self.selection = self.root / "selected"
        save_selection(self.selection, self.first, self.mask)

    def run_scene(self, *, extra=(), segmenter=None):
        output = self.root / "scene_result"
        args = parse_tracking_args([
            "--episode", str(self.episode), "--initial-selection", str(self.selection),
            "--output", str(output), "--headless", *extra,
        ], workspace=self.root)
        segmenter = segmenter or Segmenter()
        with patch("cordvip_geometry.cli.check_tracking.check_tracking_environment", return_value={"ok": True, "checks": []}) as environment, patch(
            "cordvip_geometry.sam2_config.check_sam_environment", return_value=[]
        ), patch("cordvip_geometry.segmentation.GroundedSAM2Segmenter", return_value=segmenter), patch(
            "dexmani_real.data.raw.RawReader", wraps=RawReader
        ) as reader, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
            self.assertFalse(environment.call_args.kwargs["need_pose"])
            reader.assert_called_once()
        return output, segmenter

    def test_default_processes_every_raw_row_and_mask_only_needs_no_fp(self):
        with patch.dict(sys.modules, {"torch": None, "open3d": None}):
            output, segmenter = self.run_scene()
        self.assertEqual(segmenter.rows, [0, 1, 2])
        self.assertEqual(sorted(file.name for file in (output / "frames").iterdir()), [
            "000000.npz", "000001.npz", "000002.npz",
        ])
        for index in range(3):
            with np.load(output / "frames" / f"{index:06d}.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["row_index"], index)
                self.assertEqual(saved["timestamp_ns"], raw_row(index).camera["timestamp_ns"])
                self.assertEqual(str(saved["clock"]), "host_monotonic_oldest_rgbd_receive")
                self.assertFalse(saved["object__pose_valid"])
                self.assertTrue(saved["object__observed_valid"])
                self.assertEqual(saved["object__model_normals_camera"].shape, (0, 3))
                self.assertEqual(saved["object__observed_points_camera"].shape, (64, 3))
                self.assertNotIn("object_roles", saved.files)
                self.assertNotIn("published_ns", saved.files)
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "completed")
        self.assertFalse((output / "latest.npz").exists())
        from cordvip_geometry.cli.object_tracking_demo import view_results

        with patch.dict(sys.modules, {"torch": None, "open3d": None}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(view_results(output, headless=True), 0)

    def test_explicit_limit_is_partial_and_hand_uses_measured_current_row(self):
        output, segmenter = self.run_scene(extra=["--max-frames", "2", "--with-hand", "--hand-points", "64"])
        self.assertEqual(segmenter.rows, [0, 1])
        summary = json.loads((output / "result.json").read_text())
        self.assertEqual((summary["status"], summary["processed_frames"], summary["source_frames"]), ("partial", 2, 3))
        with np.load(output / "frames" / "000001.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["joint_state"][:7], np.full(7, .01))
            np.testing.assert_array_equal(saved["joint_state"][7:], np.full(12, .02))
            self.assertEqual(saved["hand_points_camera"].shape, (64, 3))
            self.assertEqual(saved["arm_timestamp_ns"], raw_row(1).arm["timestamp_ns"][0])
            self.assertEqual(saved["camera_timestamp_ns"], raw_row(1).camera["timestamp_ns"])
            self.assertTrue(saved["hand_timing_available"])

    def test_multiobject_selection_is_bound_to_the_same_first_frame(self):
        left = self.mask.copy()
        left[:, 8:] = False
        right = ~left
        second = self.root / "second_selection"
        save_selection(second, self.first, right)
        input_scene = self.root / "scene.json"
        input_scene.write_text(json.dumps({"objects": [
            {"id": "left", "role": "object", "initial_selection": str(self.selection)},
            {"id": "right", "role": "reference", "initial_selection": str(second)},
        ]}))
        # 给左实例使用互斥 mask，避免有意构造的重叠消掉所有实测点。
        np.save(self.selection / "mask.npy", left)
        args = parse_tracking_args([
            "--episode", str(self.episode), "--scene", str(input_scene), "--headless",
            "--output", str(self.root / "multi_result"),
        ], workspace=self.root)
        with patch("cordvip_geometry.cli.check_tracking.check_tracking_environment", return_value={"ok": True, "checks": []}), patch(
            "cordvip_geometry.sam2_config.check_sam_environment", return_value=[]
        ), patch("cordvip_geometry.segmentation.GroundedSAM2Segmenter", return_value=Segmenter()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        with np.load(self.root / "multi_result/frames/000000.npz", allow_pickle=False) as saved:
            self.assertEqual(len(saved["left__observed_points_camera"]), 32)
            self.assertEqual(len(saved["right__observed_points_camera"]), 32)
        roles = [item["role"] for item in json.loads((self.root / "multi_result/scene.json").read_text())["objects"]]
        self.assertEqual(roles, ["object", "reference"])

    def test_changed_initial_source_is_rejected(self):
        altered_rgb = self.first.rgb.copy()
        altered_rgb[0, 0] = 255
        altered_K = self.first.K.copy()
        altered_K[0, 0] += 1
        for frame in (
            replace(self.first, rgb=altered_rgb),
            replace(self.first, depth_m=self.first.depth_m + .01),
            replace(self.first, K=altered_K),
            replace(self.first, row_index=1),
            replace(self.first, timestamp_ns=self.first.timestamp_ns + 1),
        ):
            with self.subTest(frame=frame.source_id):
                with self.assertRaisesRegex(ValueError, "differs"):
                    load_initial_selection(self.selection, frame)

    def test_failure_keeps_original_cause_and_only_completed_prefix(self):
        segmenter = Segmenter()
        segmenter.track = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model failed"))
        segmenter.close = lambda: (_ for _ in ()).throw(OSError("model close failed"))
        with self.assertLogs("cordvip_geometry.cli.scene_tracking", level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.run_scene(segmenter=segmenter)
        output = self.root / "scene_result"
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["processed_frames"], 1)
        self.assertIn("model failed", result["error"])
        self.assertIn("model close failed", result["cleanup_errors"][0])
        self.assertEqual(len(list((output / "frames").iterdir())), 1)


class ScenePoseTests(unittest.TestCase):
    def test_shared_rays_and_pose_loss_preserve_measured_points(self):
        shape = (8, 8)
        K = np.array([[16., 0., 4.], [0., 16., 4.], [0., 0., 1.]])
        rgb, depth = np.zeros((*shape, 3), np.uint8), np.full(shape, .5, np.float32)
        frame = Frame(rgb, depth, K, "none", (0.,) * 5, "test:0", 100, "recorded", 0)
        left = np.ones(shape, bool)
        left[:, 4:] = False
        masks = {"left": left, "right": ~left}
        pose = np.eye(4)
        pose[2, 3] = .5
        fake_tracker = unittest.mock.Mock()
        fake_tracker.register.return_value = pose
        fake_tracker.track.return_value = pose
        rays = []
        rendered_depth = [depth]

        class Renderer:
            def __init__(self, *args, **kwargs):
                pass

            def render(self, *args, directions):
                rays.append(directions)
                return rendered_depth[0]

        assets = {name: (None, np.zeros((1, 3)), np.array([[1., 0., 0.]])) for name in masks}
        with patch("cordvip_geometry.foundationpose.FoundationPosePlusTracker", return_value=fake_tracker), patch(
            "cordvip_geometry.scene_tracking.ModelDepthRenderer", Renderer
        ), patch("cordvip_geometry.scene_tracking.prepare_tracking_frame", return_value={}), patch.object(SceneTracker, "sync"):
            tracker = SceneTracker(Segmenter(), assets, foundationpose_root=Path("unused"), debug_dir=Path("unused"), min_matched_pixels=1)
            result = tracker.step(frame, rgb, depth, K, initial_masks=masks)
            self.assertIs(rays[0], rays[1])
            self.assertFalse(rays[0].flags.writeable)
            arrays = scene_arrays(result)
            self.assertTrue(arrays["left__pose_valid"])
            np.testing.assert_array_equal(arrays["left__model_normals_camera"], [[1., 0., 0.]])
            rendered_depth[0] = depth + .1
            following = replace(frame, source_id="test:1", timestamp_ns=200, row_index=1)
            lost = scene_arrays(tracker.step(following, rgb, depth, K))
            self.assertFalse(lost["left__pose_valid"])
            self.assertTrue(lost["left__observed_valid"])
            self.assertEqual(len(lost["left__observed_points_camera"]), 8)
            self.assertEqual(len(lost["left__model_points_camera"]), 0)
            self.assertEqual(str(lost["left__state"]), "POSE_LOST")


if __name__ == "__main__":
    unittest.main()
