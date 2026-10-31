"""离线结果可独立读取；无需 Raw、模型、相机或 3D 环境。"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"))
from cordvip_geometry.cli.object_tracking_demo import view_results
from cordvip_geometry.viewer import scene_display


def result_frame(row):
    return dict(
        rgb=np.zeros((8, 8, 3), np.uint8), K=np.eye(3),
        row_index=row, timestamp_ns=1_000_000 + row, clock="host_monotonic_ns",
        object__semantic_mask=np.ones((8, 8), bool), object__pose_valid=False,
        object__observed_valid=True, object__state="OBSERVED_ONLY",
        object__T_camera_object=np.full((4, 4), np.nan),
        object__observed_points_camera=np.array([[0., 0., .7]], np.float32),
        object__model_points_camera=np.empty((0, 3), np.float32),
        hand_points_camera=np.array([[.1, 0., .7]], np.float32),
    )


class OfflineViewerTests(unittest.TestCase):
    def test_saved_scene_reads_without_raw_models_or_open3d(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "scene.json").write_text(json.dumps({"objects": [{"id": "object", "role": "object"}]}))
            (root / "run.json").write_text(json.dumps({"colors_rgb": {"object": [0, 255, 255]}}))
            for row in range(2):
                np.savez(root / "frames" / f"{row:06d}.npz", **result_frame(row))
            with patch.dict(sys.modules, {"open3d": None, "torch": None, "pyrealsense2": None}), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(view_results(root, headless=True), 0)
            self.assertIn("已读取 2 行", output.getvalue())
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["frames", "run.json", "scene.json"])

    def test_observed_points_survive_missing_pose_and_rgb_must_be_valid(self):
        frame = result_frame(0)
        colors = {"object": [0, 255, 255]}
        panel, observed, _ = scene_display(frame, ["object"], colors, include_clouds=True)
        self.assertEqual(panel.shape, (8, 8, 3))
        self.assertEqual(len(observed), 2)  # 独立实测点与手点仍可见。
        _, model, _ = scene_display(frame, ["object"], colors, show_model=True, include_clouds=True)
        np.testing.assert_array_equal(model, frame["hand_points_camera"])
        frame["rgb"] = frame["rgb"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "RGB"):
            scene_display(frame, ["object"], colors)


if __name__ == "__main__":
    unittest.main()
