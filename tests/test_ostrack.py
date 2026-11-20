"""覆盖 OSTrack-256 框约定和共享状态，测试无需 GPU。"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
with patch.object(sys, "path", [str(BACKEND), *sys.path]):
    from cordvip_geometry.ostrack import (
        OSTRACK_CONFIG,
        OSTrackAdapter,
        mask_to_xyxy,
    )


class MockUpstreamTracker:
    def __init__(self, params, dataset_name):
        self.network = object()
        self.params = params

    def initialize(self, rgb, info):
        self.template = object()
        self.state = info["init_bbox"]

    def track(self, rgb):
        if rgb[0, 0, 0]:
            return {"target_bbox": None}
        x, y, w, h = self.state
        self.state = [x + 1, y, w, h]
        return {"target_bbox": self.state}


class OSTrackTest(unittest.TestCase):
    def test_mask_bbox_includes_boundary_pixels(self):
        mask = np.zeros((12, 20), dtype=bool)
        self.assertIsNone(mask_to_xyxy(mask))
        mask[5:12, 9:20] = True
        self.assertEqual(mask_to_xyxy(mask), (9, 5, 20, 12))

    def test_objects_share_network_and_keep_independent_templates(self):
        test_cfg = SimpleNamespace(
            TEMPLATE_FACTOR=2.0, TEMPLATE_SIZE=128, SEARCH_FACTOR=4.0, SEARCH_SIZE=256
        )
        upstream = {
            "lib.config.ostrack.config": SimpleNamespace(
                cfg=SimpleNamespace(TEST=test_cfg),
                update_config_from_file=lambda path, base_cfg: None,
            ),
            "lib.test.tracker.ostrack": SimpleNamespace(OSTrack=MockUpstreamTracker),
        }
        rgb = np.zeros((40, 50, 3), dtype=np.uint8)
        with (
            TemporaryDirectory() as directory,
            patch.dict(sys.modules, upstream),
            patch.object(sys, "path", sys.path.copy()),
        ):
            root = Path(directory)
            checkpoint = root / "OSTrack_ep0300.pth.tar"
            with self.assertRaisesRegex(FileNotFoundError, "OSTrack-256 checkpoint"):
                OSTrackAdapter(root, checkpoint)
            checkpoint.touch()
            config = root / OSTRACK_CONFIG
            config.parent.mkdir(parents=True)
            config.touch()
            adapter = OSTrackAdapter(root, checkpoint)

        adapter.initialize("first", rgb, (5, 3, 15, 15))
        adapter.initialize("second", rgb, (20, 13, 30, 25))
        first, second = adapter._objects.values()
        self.assertIs(first.network, second.network)
        self.assertIsNot(first.template, second.template)
        self.assertEqual(first.state, [5, 3, 10, 12])
        self.assertEqual(adapter.track("first", rgb), (6, 3, 16, 15))
        self.assertEqual(second.state, [20, 13, 10, 12])

        rgb[0, 0, 0] = 1
        self.assertIsNone(adapter.track("first", rgb))
        self.assertEqual(first.state, [6, 3, 10, 12])
        with self.assertRaisesRegex(ValueError, "Invalid OSTrack initialization box"):
            adapter.initialize("first", rgb, (2, 5, float("inf"), 17))
        self.assertIs(adapter._objects["first"], first)
        adapter.initialize("first", rgb, (-2, 5, 10, 17))
        self.assertIs(adapter._objects["first"].network, second.network)
        self.assertIsNot(adapter._objects["first"].template, first.template)
        self.assertIs(adapter._objects["second"], second)
        self.assertEqual(adapter._objects["first"].state, [0, 5, 10, 12])
