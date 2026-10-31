"""Raw viewer 的显式 RGB 行关联；不启动 Rerun GUI。"""

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from scripts import visualize_episode as viewer


class VisualizeEpisodeTests(unittest.TestCase):
    def visualizer(self, images, count=2):
        viz = viewer.EpisodeVisualizer.__new__(viewer.EpisodeVisualizer)
        viz.rr = Mock()
        viz.reader = MagicMock()
        depth = np.stack([np.full((2, 3), index + 1, np.uint16) for index in range(count)])
        viz.reader.__getitem__.side_effect = {"depth": depth}.__getitem__
        viz.reader.iter_rgb_frames.return_value = iter(images)
        viz.camera_model = SimpleNamespace(color=SimpleNamespace(height=2, width=3), depth_scale_m=.001)
        viz.frame_count = count
        viz.timestamps_s = np.arange(count) / 16
        viz.state = dict(hand_fingertip=np.zeros((count, 5, 3)), arm_ee=np.zeros((count, 9)))
        viz.pointcloud_processor = Mock()
        viz.pointcloud_processor.process.return_value = np.empty((0, 6), np.float32)
        viz.empty_cloud_count = viz.pointcloud_processing_ns = viz.pointcloud_processed_frames = 0
        return viz, depth

    def test_sequential_rgb_matches_depth_and_respects_max_frames(self):
        rgb = [np.full((2, 3, 3), index * 50, np.uint8) for index in range(2)]

        def images():
            yield from rgb
            raise AssertionError("max_frames must not decode beyond the requested rows")

        viz, depth = self.visualizer(images())
        with tempfile.TemporaryDirectory() as directory, patch.object(
            viewer, "EpisodeVisualizer", return_value=viz,
        ), patch.object(viewer, "log_rgbd") as log_rgbd, patch.object(
            viewer, "log_points",
        ) as log_points, patch("dexmani_real.utils.log.configure_logging"):
            self.assertEqual(viewer.main([directory, "--max-frames", "2"]), 0)
        for index, call in enumerate(log_rgbd.call_args_list):
            self.assertIs(call.args[0], rgb[index])
            np.testing.assert_array_equal(call.args[1], depth[index])
        self.assertEqual(log_rgbd.call_count, 2)
        self.assertEqual(viz.empty_cloud_count, 2)
        cloud_calls = [call for call in log_points.call_args_list if call.args[0] == "pcd"]
        self.assertEqual(len(cloud_calls), 2)
        self.assertTrue(all(call.args[1].shape == (0, 6) for call in cloud_calls))
        viz.reader.close.assert_called_once()
        viz.rr.disconnect.assert_called_once()

    def test_short_rgb_stream_fails_and_closes(self):
        viz, _ = self.visualizer([np.zeros((2, 3, 3), np.uint8)])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            viewer, "EpisodeVisualizer", return_value=viz,
        ), patch.object(viewer, "log_rgbd"), patch.object(viewer, "log_points"), patch(
            "dexmani_real.utils.log.configure_logging",
        ):
            with self.assertRaisesRegex(ValueError, "RGB missing at frame 1"):
                viewer.main([directory])
        viz.reader.close.assert_called_once()
        viz.rr.disconnect.assert_called_once()

    def test_bad_rgb_fails_before_display(self):
        viz, _ = self.visualizer([])
        for rgb in (np.zeros((2, 3, 3), np.float32), np.zeros((3, 2, 3), np.uint8)):
            with self.subTest(shape=rgb.shape, dtype=rgb.dtype), patch.object(viewer, "log_rgbd") as log:
                with self.assertRaisesRegex(ValueError, "RGB shape or dtype invalid at frame 1"):
                    viz.log_camera(1, rgb)
                log.assert_not_called()

    def test_explicit_temporary_alias_is_rejected_by_cli_and_direct_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            published = Path(directory) / "episode_published"
            published.mkdir()
            alias = Path(directory) / ".tmp_alias"
            alias.symlink_to(published, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unpublished"):
                viewer.main([str(alias), "--info"])
            rerun = Mock()
            with patch.dict("sys.modules", {"rerun": rerun}):
                with self.assertRaisesRegex(ValueError, "unpublished"):
                    viewer.EpisodeVisualizer(str(alias), table_plane_path=None)
            rerun.init.assert_not_called()


if __name__ == "__main__":
    unittest.main()
