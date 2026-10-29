"""CordVIP 首帧身份、支持点与关闭竞态，不加载模型或打开窗口。"""

from concurrent.futures import CancelledError
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cordvip_geometry.cli import build_object, capture_object
from cordvip_geometry.io import load_initial_selection, write_confirmed_selection
from cordvip_geometry.live_workers import (
    CloseStatus, SceneWorker, SourceSnapshot, wait_worker,
)
from cordvip_geometry.rgbd import Frame, save_frame
from cordvip_geometry.scene_tracking import SceneTracker
from cordvip_geometry.segmentation import GroundedSAM2Segmenter


def frame():
    return Frame(
        rgb=np.ones((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32), K=np.eye(3),
        distortion_model="none", distortion_coeffs=(0, 0, 0, 0, 0),
        source_id="raw:/old/episode:camera:1", timestamp_ns=10,
        clock="host_monotonic_oldest_rgbd_receive", row_index=0,
    )


class CordVIPReliabilityTests(unittest.TestCase):
    def test_close_discards_results_and_preserves_independent_failures(self):
        for failure, completed in (
            (ValueError("model failed"), False),
            (CancelledError("expected close"), False),
            (ValueError("model failed"), True),
        ):
            with self.subTest(failure=type(failure).__name__, completed=completed):
                entered, release = threading.Event(), threading.Event()

                def step(*args, **kwargs):
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("test did not release model")
                    raise failure

                segmenter = SimpleNamespace(
                    start=lambda: None, cancel=lambda: None, last_metrics={}, provenance={},
                )
                tracker = SimpleNamespace(trackers={}, step=step, close=lambda: None)
                worker = SceneWorker(segmenter, lambda segmenter: tracker).start()
                deadline = time.monotonic() + 2

                def heartbeat():
                    if time.monotonic() >= deadline:
                        raise TimeoutError("scene worker did not become ready")

                try:
                    wait_worker(worker, heartbeat, loading=True)
                    self.assertTrue(worker.submit("step", frame()))
                    self.assertTrue(entered.wait(1))
                    if completed:
                        release.set()
                        while worker.snapshot().busy:
                            heartbeat()
                            time.sleep(0.001)
                    else:
                        self.assertTrue(worker.close(timeout_s=0).pending)
                finally:
                    release.set()
                    status = worker.close(timeout_s=2)
                self.assertTrue(status.stopped)
                self.assertIsNone(worker.poll())
                if isinstance(failure, CancelledError):
                    self.assertIsNone(status.error)
                else:
                    self.assertIn("ValueError: model failed", status.error)

    def test_loading_failure_after_close_remains_visible(self):
        entered, release = threading.Event(), threading.Event()

        def start():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release loader")
            raise ValueError("load failed")

        segmenter = SimpleNamespace(start=start, cancel=lambda: None, close=lambda: None)
        worker = SceneWorker(segmenter, lambda segmenter: None).start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(worker.close(timeout_s=0).pending)
        finally:
            release.set()
            status = worker.close(timeout_s=2)
        self.assertIn("ValueError: load failed", status.error)
        self.assertIsNone(worker.poll())

    def test_sam_cancel_does_not_hide_received_model_error(self):
        for model_error in (False, True):
            with self.subTest(model_error=model_error):
                segmenter = GroundedSAM2Segmenter(
                    "python", ".", "sam.pt", sam_config="sam", dino_weights="dino",
                    log_path="sam.log",
                )
                segmenter._process = SimpleNamespace(stdin=None, stdout=None)
                segmenter.signal_group = Mock()
                sent = {}

                def receive(stream):
                    segmenter._cancelled.set()
                    if model_error:
                        return {**sent, "error": "independent model failure"}, {}
                    raise EOFError("closed by cancel")

                with (
                    patch("cordvip_geometry.segmentation.send",
                          side_effect=lambda stream, metadata, arrays: sent.update(metadata)),
                    patch("cordvip_geometry.segmentation.receive", side_effect=receive),
                ):
                    expected = RuntimeError if model_error else CancelledError
                    with self.assertRaises(expected) as failure:
                        segmenter.call("load")
                if model_error:
                    self.assertIn("independent model failure", str(failure.exception))

    def test_builder_owns_failed_sam_start_and_preserves_the_load_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            source.write_bytes(b"source is copied after the RGB has been loaded")
            args = SimpleNamespace(
                headless=True, instance=0, list_only=False, rgbd=None,
                output_dir=root / "output", workspace=root, text="object",
                sam_python=root / "python", sam_root=root / "sam",
                sam_checkpoint=root / "sam.pt", sam_config="sam",
                dino_weights=root / "dino", box_threshold=0.3, text_threshold=0.25,
            )
            segmenter = GroundedSAM2Segmenter(
                args.sam_python, args.sam_root, args.sam_checkpoint,
                sam_config=args.sam_config, dino_weights=args.dino_weights,
                log_path=root / "sam.log",
            )
            try:
                with (
                    patch.object(build_object, "load_rgb", return_value=(frame().rgb, source, {})),
                    patch("cordvip_geometry.segmentation.GroundedSAM2Segmenter", return_value=segmenter),
                    patch("cordvip_geometry.segmentation.subprocess.Popen"),
                    patch.object(segmenter, "call", side_effect=ValueError("load failed")),
                    patch.object(segmenter, "close", side_effect=RuntimeError("close failed")) as close,
                    self.assertLogs(build_object.__name__, level="ERROR") as logs,
                ):
                    with self.assertRaisesRegex(ValueError, "load failed"):
                        build_object.ObjectModelBuilder(args).segment()
                    close.assert_called_once()
                self.assertIn("close failed", "\n".join(logs.output))
            finally:
                if segmenter._log is not None:
                    segmenter._log.close()

    def test_selection_relocation_keeps_content_and_recorded_identity_checks(self):
        original = frame()
        moved = replace(original, source_id="raw:/new/episode:camera:1")
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            save_frame(directory / "source_rgbd.npz", original)
            metadata = dict(
                selection="gui_confirmed", selected_instance=0,
                mask_grid="original_rgb_grid; not_automatically_rectified",
                source_id=original.source_id, timestamp_ns=original.timestamp_ns,
                clock=original.clock,
            )
            write_confirmed_selection(
                directory, np.ones((8, 8), dtype=bool), metadata, mask_file="mask.npy",
            )
            self.assertTrue(load_initial_selection(directory, moved)[0].all())
            for changes in (
                {"rgb": np.zeros_like(moved.rgb)},
                {"depth_m": moved.depth_m * 2},
                {"K": np.diag([2, 1, 1])},
                {"distortion_coeffs": (0.1, 0, 0, 0, 0)},
                {"row_index": 1}, {"timestamp_ns": 11}, {"clock": "other"},
                {"source_id": "raw:/new/episode:camera:2"},
                {"source_id": "realsense:camera:1"},
            ):
                with self.subTest(changes=tuple(changes)):
                    with self.assertRaises(ValueError):
                        load_initial_selection(directory, replace(moved, **changes))
            saved_metadata = json.loads((directory / "selection.json").read_text())
            saved_metadata["source_id"] = moved.source_id
            (directory / "selection.json").write_text(json.dumps(saved_metadata))
            with self.assertRaisesRegex(ValueError, "metadata source_id"):
                load_initial_selection(directory, moved)

    def test_pose_initialization_uses_actual_stride_support(self):
        source = frame()
        masks = {"object": np.ones((8, 8), dtype=bool)}

        def new_segmenter():
            return SimpleNamespace(
                initialize=lambda *args, **kwargs: masks, track=lambda *args, **kwargs: masks,
                last_metrics={},
            )

        points, normals = np.zeros((1, 3)), np.array([[0, 0, 1]])
        pose_tracker = SimpleNamespace(register=Mock(return_value=np.eye(4)))
        renderer = SimpleNamespace(render=lambda *args: np.ones((8, 8), dtype=np.float32))
        with (
            patch("cordvip_geometry.foundationpose.FoundationPosePlusTracker",
                  return_value=pose_tracker),
            patch("cordvip_geometry.scene_tracking.ModelDepthRenderer", return_value=renderer),
        ):
            tracker = SceneTracker(
                new_segmenter(), {"object": (None, points, normals)},
                foundationpose_root="unused", debug_dir="unused", pixel_stride=2,
            )
            with self.assertRaisesRegex(ValueError, "16.*requires 30"):
                tracker.step(source, source.rgb, source.depth_m, source.K, initial_masks=masks)
            self.assertIsNone(tracker.instance_ids)
            pose_tracker.register.assert_not_called()

            # 首次初始化失败后须开启新会话，不在已初始化的 SAM 上重试。
            tracker = SceneTracker(
                new_segmenter(), {"object": (None, points, normals)},
                foundationpose_root="unused", debug_dir="unused", pixel_stride=2,
                min_matched_pixels=16,
            )
            with patch.object(tracker, "sync"):
                first = tracker.step(
                    source, source.rgb, source.depth_m, source.K, initial_masks=masks,
                )
            self.assertEqual(first["objects"]["object"]["state"], "TRACKING")
            later = replace(source, timestamp_ns=11, row_index=1)
            depth = np.zeros_like(source.depth_m)
            later_result = tracker.step(later, later.rgb, depth, later.K)
            self.assertEqual(later_result["objects"]["object"]["state"], "POSE_LOST")

        mask_only = SceneTracker(new_segmenter(), {}, foundationpose_root="unused", debug_dir="unused")
        observed = mask_only.step(
            source, source.rgb, source.depth_m, source.K, initial_masks=masks,
        )
        self.assertEqual(observed["objects"]["object"]["state"], "OBSERVED_ONLY")

    def test_frozen_copy_can_be_confirmed_after_source_failure(self):
        import cv2

        for freeze in (False, True):
            with self.subTest(freeze=freeze), TemporaryDirectory() as directory:
                root = Path(directory)
                source = frame()
                original_rgb = source.rgb.copy()
                snapshots = [SourceSnapshot(source, 1, False, None)] if freeze else []
                snapshots.append(SourceSnapshot(None, 1, True, "source failed"))

                def snapshot():
                    if len(snapshots) == 1:
                        source.rgb[:] = 9
                    return snapshots.pop(0)

                reader = SimpleNamespace(
                    start=Mock(), snapshot=snapshot,
                    close=lambda timeout_s: CloseStatus(True, "source failed"),
                )
                args = SimpleNamespace(
                    rgbd_dir=root / "input", output_dir=root / "output", workspace=root,
                    replay_fps=10,
                )
                with (
                    patch.dict("os.environ", {"DISPLAY": "test"}),
                    patch("cordvip_geometry.live_workers.LatestFrameReader", return_value=reader),
                    patch.object(capture_object, "panel", return_value=np.zeros((2, 2, 3))),
                    patch.object(cv2, "namedWindow"), patch.object(cv2, "resizeWindow"),
                    patch.object(cv2, "imshow"), patch.object(cv2, "destroyWindow"),
                    patch.object(cv2, "getWindowProperty", return_value=1),
                    patch.object(cv2, "waitKey", side_effect=[32, 13]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "source failed"):
                        capture_object.run(args)
                result = json.loads((args.output_dir / "result.json").read_text())
                self.assertEqual(result["capture_saved"], freeze)
                self.assertIn("source failed", result["error"])
                self.assertEqual((args.output_dir / "frame.npz").exists(), freeze)
                if freeze:
                    with np.load(args.output_dir / "frame.npz") as saved:
                        np.testing.assert_array_equal(saved["rgb"], original_rgb)


if __name__ == "__main__":
    unittest.main()
