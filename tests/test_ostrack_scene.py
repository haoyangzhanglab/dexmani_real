"""验证 OSTrack 场景跟踪、恢复和居中位姿先验。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.perception.hoi import track_hoi_frame

BACKEND = Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
with patch.object(sys, "path", [str(BACKEND), *sys.path]):
    from cordvip_geometry import foundationpose
    from cordvip_geometry.camera import RGBDFrame
    from cordvip_geometry.tracking import SceneTracker, TrackingConfig


class OSTrackSceneTest(unittest.TestCase):
    def setUp(self):
        self.names = ("cup", "tray")
        self.prompts = dict(zip(self.names, self.names))
        self.template = {"object_names": np.array(self.names)}
        self.image_shape = (40, 64)
        self.K = np.array([[100, 0, 30], [0, 100, 20], [0, 0, 1]])
        self.rgb = np.zeros((*self.image_shape, 3), np.uint8)
        self.depth = np.ones(self.image_shape, np.float32)
        self.sequence = 0
        self.timestamp_ns = 1_000_000_000

        self.mask_boxes = {"cup": (14, 6, 34, 26), "tray": (38, 12, 58, 34)}
        self.boxes = self.mask_boxes.copy()
        self.masks = {}
        self.poses = {}
        for name, (x1, y1, x2, y2) in self.mask_boxes.items():
            mask = np.zeros(self.image_shape, bool)
            mask[y1:y2, x1:x2] = True
            self.masks[name] = mask
            pose = np.eye(4, dtype=np.float32)
            pose[:3, 3] = [(x1 + x2 - 60) / 200, (y1 + y2 - 40) / 200, 1]
            self.poses[name] = pose

        self.events = []
        self.reused_images = []
        self.register_snapshots = {name: [] for name in self.names}
        self.register_depths = {name: [] for name in self.names}
        self.track_depths = dict.fromkeys(self.names, 1.0)
        self.scene = SceneTracker.__new__(SceneTracker)
        self.scene.config = TrackingConfig(segmentation_interval_frames=4)
        self.scene.segmenter = Mock(spec_set=["detect", "segment_boxes"])
        self.scene.segmenter.detect.side_effect = self.detect
        self.scene.segmenter.segment_boxes.side_effect = self.segment
        self.scene.ostrack = Mock()
        self.scene.ostrack.track.side_effect = lambda name, rgb: self.boxes[name]
        self.scene.ostrack.initialize.side_effect = (
            lambda name, rgb, box: self.events.append(("initialize", name))
        )
        self.scene.trackers = {}
        self.scene.renderers = {}
        render_masks = self.masks.copy()
        for name in self.names:
            self.scene.trackers[name] = SimpleNamespace(
                estimator=SimpleNamespace(pose_last=None, diameter=1),
                object_from_center=np.eye(4),
                register=Mock(side_effect=lambda *args, name=name: self.register(name)),
            )
            self.scene.renderers[name] = SimpleNamespace(
                render_roi=Mock(side_effect=lambda directions, pose, *args, name=name: (
                    render_masks[name][::4, ::4] * pose[2, 3]
                )),
            )
        self.scene.trackers["tray"].object_from_center[0, 3] = 0.01
        tracker_patch = patch(
            "cordvip_geometry.tracking.track_objects", side_effect=self.track_objects
        )
        self.track_objects_mock = tracker_patch.start()
        self.addCleanup(tracker_patch.stop)

    def detect(self, rgb, text, *args):
        self.events.append(("detect", text))
        return [
            {"label": name, "box_xyxy": box}
            for name, box in self.mask_boxes.items()
        ]

    def segment(self, rgb, boxes, *, reuse_image=False):
        self.events.append(("segment", tuple(boxes)))
        self.reused_images.append(reuse_image)
        return {name: self.masks[name] for name in boxes}

    def register(self, name):
        self.events.append(("register", name))
        pose = self.poses[name].copy()
        if self.register_depths[name]:
            pose[2, 3] = self.register_depths[name].pop(0)
        tracker = self.scene.trackers[name]
        self.register_snapshots[name].append(tracker.estimator.pose_last)
        tracker.estimator.pose_last = pose @ tracker.object_from_center
        return pose

    def track_objects(self, active, *args, **kwargs):
        poses = {}
        for name, tracker in active.items():
            pose = self.poses[name].copy()
            pose[2, 3] = self.track_depths[name]
            tracker.estimator.pose_last = pose @ tracker.object_from_center
            poses[name] = pose
        return poses

    def frame(self, step_ms=100):
        self.sequence += 1
        self.timestamp_ns += step_ms * 1_000_000
        return RGBDFrame(
            self.rgb, self.depth, self.K, "none", (0.0,) * 5,
            self.sequence, self.timestamp_ns,
        )

    def initialize(self):
        return self.scene.initialize(self.frame(), self.prompts)

    def enter_reacquire(self):
        self.track_depths["cup"] = 1.1
        reliable_pose = self.scene.trackers["cup"].estimator.pose_last
        self.boxes["cup"] = (2, 2, 12, 12)
        for step_ms in (10, 80, 220):
            result = track_hoi_frame(self.scene, self.frame(step_ms), self.template, self.prompts)
            self.assertFalse(result.valid)
            np.testing.assert_array_equal(result.object_valid, [False, True])
            self.assertTrue(np.isnan(result.object_pose_camera[0]).all())
            self.assertIs(self.scene.trackers["cup"].estimator.pose_last, reliable_pose)
        self.assertEqual(tuple(self.scene.next_retry), ("cup",))
        self.assertEqual(self.scene.last_good_bbox["cup"], self.mask_boxes["cup"])
        return reliable_pose

    def hide_cup_until_reacquire(self, observed_depth):
        visible_mask = self.masks["cup"]
        self.masks["cup"] = np.zeros_like(visible_mask)
        self.depth[visible_mask] = observed_depth
        self.boxes["cup"] = None
        reliable_pose = self.scene.trackers["cup"].estimator.pose_last
        for _ in range(4):
            result = self.scene.track(self.frame())
            self.assertFalse(result.valid["cup"])
            self.assertNotIn("cup", result.poses)
            self.assertIs(self.scene.trackers["cup"].estimator.pose_last, reliable_pose)
        self.assertIn("cup", self.scene.next_retry)
        return visible_mask

    def test_normal_tracking_uses_image_masks_only_at_interval(self):
        result = self.initialize()
        self.assertEqual(result.valid, dict.fromkeys(self.names, True))
        self.scene.ostrack.clear.assert_called_once()
        for name, rgb, box in (args for args, _ in self.scene.ostrack.initialize.call_args_list):
            self.assertIs(rgb, self.rgb)
            self.assertEqual(box, self.boxes[name])
        for index in range(1, 5):
            result = self.scene.track(self.frame())
            self.assertTrue(all(result.valid.values()))
            self.assertEqual(result.quality["tray"]["mask_source"], int(index == 4))
        self.assertEqual(self.track_objects_mock.call_count, 4)
        self.assertEqual(self.scene.ostrack.track.call_count, 8)
        self.assertEqual(self.scene.segmenter.segment_boxes.call_count, 2)
        self.assertEqual(self.scene.segmenter.segment_boxes.call_args.args[1], self.mask_boxes)
        self.assertEqual(self.scene.segmenter.detect.call_count, 1)
        self.assertEqual(self.scene.ostrack.initialize.call_count, len(self.names))
        priors = self.track_objects_mock.call_args.kwargs["bbox_priors"]
        np.testing.assert_allclose(
            priors["tray"][0],
            self.poses["tray"] @ self.scene.trackers["tray"].object_from_center,
        )

    def test_last_good_box_recovery_is_valid_immediately_and_shares_segmentation(self):
        self.scene.config = TrackingConfig(segmentation_interval_frames=1)
        self.initialize()
        self.enter_reacquire()
        self.events.clear()
        self.poses["cup"][0, 3] += 0.01
        result = self.scene.track(self.frame())
        self.assertEqual(self.events, [
            ("segment", self.names), ("register", "cup"), ("initialize", "cup")
        ])
        self.assertEqual(self.scene.segmenter.segment_boxes.call_args.args[1], self.mask_boxes)
        self.scene.ostrack.initialize.assert_called_with("cup", self.rgb, self.mask_boxes["cup"])
        self.assertEqual(self.scene.ostrack.initialize.call_count, len(self.names) + 1)
        self.assertNotIn("cup", self.scene.next_retry)
        self.assertTrue(result.valid["cup"])
        self.assertEqual(tuple(result.poses), self.names)
        np.testing.assert_array_equal(result.poses["cup"], self.poses["cup"])

    def test_dino_runs_only_after_rejected_last_good_box_registration(self):
        self.scene.config = TrackingConfig(segmentation_interval_frames=1)
        self.initialize()
        previous = self.enter_reacquire()
        self.events.clear()
        self.reused_images.clear()
        self.register_depths["cup"] = [1.1, 1.0]
        result = self.scene.track(self.frame())
        self.assertEqual(self.events, [
            ("segment", self.names), ("register", "cup"),
            ("detect", "cup."), ("segment", ("cup",)),
            ("register", "cup"), ("initialize", "cup"),
        ])
        self.assertIs(self.register_snapshots["cup"][-2], previous)
        self.assertIs(self.register_snapshots["cup"][-1], previous)
        self.assertEqual(self.reused_images, [False, True])
        self.assertTrue(all(result.valid.values()))
        self.assertEqual(result.failures, {})

    def test_failed_reacquisition_stays_invalid_until_processed_frame_retry(self):
        self.initialize()
        previous = self.enter_reacquire()
        self.register_depths["cup"] = [1.1, 1.1]
        for _ in range(8):
            result = self.scene.track(self.frame())
            self.assertFalse(result.valid["cup"])
            self.assertNotIn("cup", result.poses)
            self.assertIs(self.scene.trackers["cup"].estimator.pose_last, previous)
        self.assertEqual(self.scene.ostrack.initialize.call_count, len(self.names))
        self.assertEqual(self.scene.segmenter.detect.call_count, 2)
        result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        self.assertEqual(self.scene.ostrack.initialize.call_count, len(self.names) + 1)

    def test_disabled_recovery_reports_no_retry_and_keeps_object_invalid(self):
        self.scene.config = TrackingConfig(recovery_enabled=False)
        self.initialize()
        self.enter_reacquire()
        self.events.clear()

        result = self.scene.track(self.frame())

        self.assertFalse(result.valid["cup"])
        self.assertNotIn("cup", result.poses)
        self.assertEqual(result.failures["cup"], "REACQUIRE; recovery disabled")
        self.assertFalse(any(kind in {"detect", "register"} for kind, _ in self.events))

    def test_complete_foreground_occlusion_recovers_in_last_good_box(self):
        self.initialize()
        clear_mask = self.hide_cup_until_reacquire(0.7)
        self.depth[clear_mask] = 1.0
        self.masks["cup"] = clear_mask

        result = self.scene.track(self.frame())

        self.assertTrue(result.valid["cup"])
        self.assertEqual(self.scene.segmenter.detect.call_count, 1)
        self.scene.ostrack.initialize.assert_called_with("cup", self.rgb, self.mask_boxes["cup"])

    def test_out_of_view_target_reenters_outside_last_good_box(self):
        self.initialize()
        self.hide_cup_until_reacquire(1.3)
        old_box = self.mask_boxes["cup"]
        new_box = (2, 18, 14, 38)
        mask = self.masks["cup"]
        mask[18:38, 2:14] = True
        self.mask_boxes["cup"] = new_box
        self.poses["cup"][:2, 3] = [-0.22, 0.1]
        self.scene.renderers["cup"].render_roi.side_effect = None
        self.scene.renderers["cup"].render_roi.return_value = mask[::4, ::4]

        def segment(rgb, boxes, *, reuse_image=False):
            masks = self.segment(rgb, boxes, reuse_image=reuse_image)
            if boxes.get("cup") == old_box:
                masks["cup"] = np.zeros_like(mask)
            return masks

        self.scene.segmenter.segment_boxes.side_effect = segment
        self.scene.segmenter.segment_boxes.reset_mock()
        result = self.scene.track(self.frame())

        self.assertTrue(result.valid["cup"])
        np.testing.assert_array_equal(result.poses["cup"], self.poses["cup"])
        self.assertEqual(self.scene.segmenter.detect.call_count, 2)
        self.assertEqual(self.scene.segmenter.segment_boxes.call_args_list[0].args[1]["cup"], old_box)
        self.scene.segmenter.segment_boxes.assert_called_with(self.rgb, {"cup": new_box}, reuse_image=True)
        self.scene.ostrack.initialize.assert_called_with("cup", self.rgb, new_box)

    def test_invalid_bbox_skips_prior_and_keeps_depth_validation(self):
        self.initialize()
        self.boxes["cup"] = None
        for _ in range(4):
            result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        self.assertEqual(result.quality["cup"]["mask_source"], 0)
        self.assertNotIn("cup", self.track_objects_mock.call_args.kwargs["bbox_priors"])
        self.assertEqual(self.scene.last_good_bbox["cup"], self.mask_boxes["cup"])

    def test_drifted_bbox_does_not_replace_last_good_box_on_depth_only_frame(self):
        self.initialize()
        self.boxes["cup"] = (2, 2, 12, 12)
        result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        self.assertEqual(result.quality["cup"]["mask_source"], 0)
        self.assertEqual(self.scene.last_good_bbox["cup"], self.mask_boxes["cup"])
        self.boxes["cup"] = (16, 6, 36, 26)
        self.scene.track(self.frame())
        self.assertEqual(self.scene.last_good_bbox["cup"], (16, 6, 36, 26))

    def test_early_segmentation_can_validate_occlusion_and_reuses_cad_depth(self):
        self.initialize()
        offset = self.scene.bbox_offsets["cup"].copy()
        self.masks["cup"] = self.masks["cup"].copy()
        self.masks["cup"][:, :26] = False
        self.depth[6:26, 14:26] = 0.7
        self.boxes["cup"] = None
        result = self.scene.track(self.frame())
        self.assertFalse(result.valid["cup"])
        self.assertEqual(self.scene.segmenter.segment_boxes.call_count, 1)
        result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        self.assertGreater(result.quality["cup"]["occluded_fraction"], 0)
        self.assertEqual(self.scene.segmenter.segment_boxes.call_args.args[1], self.mask_boxes)
        self.assertNotIn("cup", self.track_objects_mock.call_args.kwargs["bbox_priors"])
        np.testing.assert_array_equal(self.scene.bbox_offsets["cup"], offset)
        self.assertEqual(self.scene.renderers["cup"].render_roi.call_count, 3)
        result = self.scene.track(self.frame())
        self.assertFalse(result.valid["cup"])
        self.assertEqual(self.scene.segmenter.segment_boxes.call_count, 2)
        self.assertEqual(self.scene.segmenter.detect.call_count, 1)

    def test_clear_mask_recalibrates_in_tracker_box_coordinates(self):
        self.initialize()
        original = self.scene.bbox_offsets["cup"].copy()
        mask = np.zeros(self.image_shape, bool)
        mask[7:25, 16:38] = True
        self.masks["cup"] = mask
        self.boxes["cup"] = (13, 4, 39, 30)
        self.poses["cup"][:2, 3] += [0.02, -0.01]
        for _ in range(4):
            result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        offset = self.scene.bbox_offsets["cup"]
        self.assertFalse(np.allclose(offset, original))
        np.testing.assert_allclose(offset, [0, 2 / 26], atol=1e-7)
        self.assertEqual(self.scene.last_good_bbox["cup"], (16, 7, 38, 25))
        self.scene.track(self.frame())
        pose, box, offset = self.track_objects_mock.call_args.kwargs["bbox_priors"]["cup"]
        np.testing.assert_allclose(
            foundationpose.xy_prior_delta(pose, self.K, box, offset, self.image_shape, 0.2),
            [0, 0], atol=1e-8,
        )
        self.boxes["cup"] = (0, 0, 10, 10)
        for _ in range(3):
            result = self.scene.track(self.frame())
        self.assertTrue(result.valid["cup"])
        self.assertNotIn("cup", self.scene.bbox_offsets)
        self.assertEqual(self.scene.ostrack.initialize.call_count, len(self.names))


class XYPriorTest(unittest.TestCase):
    def test_centered_candidate_is_corrected_before_refinement(self):
        import torch

        centered = np.array([
            [0, -1, 0, 0.1], [1, 0, 0, -0.05], [0, 0, 1, 2], [0, 0, 0, 1],
        ], dtype=np.float32)
        tf_to_centered = np.eye(4, dtype=np.float32)
        tf_to_centered[:3, 3] = [-0.05, 0.1, -0.15]
        K = np.array([[100, 0, 30], [0, 80, 20], [0, 0, 1]])
        shape = (40, 64)
        offset = foundationpose.bbox_center_offset(centered, K, (20, 10, 48, 30), shape)
        np.testing.assert_allclose(offset, [-1 / 28, 2 / 20])
        shifted_box = (24, 4, 52, 24)

        prior_candidates = []

        def render_crops(size, poses, **kwargs):
            prior_candidates.append(poses.clone())
            return (torch.zeros((1, 1)),) * 4

        refiner = SimpleNamespace(
            cfg=SimpleNamespace(input_resize=(16, 16), crop_ratio=1.2),
            refine=lambda poses: poses,
        )
        original_internal = torch.tensor(centered)
        estimator = SimpleNamespace(
            pose_last=original_internal, tf_to_centered_mesh=torch.tensor(tf_to_centered),
            refiner=refiner, diameter=1, glctx=None, mesh_tensors=None,
        )
        backend = SimpleNamespace(
            render_crops=render_crops, sample_observed_crops=lambda poses, *args, **kwargs: poses,
        )
        frame_cache = {"K": K, "projection": None, "crop_grid": None}
        with (
            patch.dict(sys.modules, {"learning.training.predict_pose_refine": backend}),
            patch.object(foundationpose, "prepare_tracking_frame", return_value=(None, None)),
        ):
            result = foundationpose.track_objects(
                {"cup": SimpleNamespace(estimator=estimator)},
                np.zeros((*shape, 3), np.uint8), np.ones(shape), K, frame_cache,
                bbox_priors={"cup": (centered, shifted_box, offset)}, xy_prior_gain=0.2,
            )

        after = prior_candidates[0][0].numpy()
        np.testing.assert_array_equal(original_internal.numpy(), centered)
        np.testing.assert_array_equal(after[:3, :3], centered[:3, :3])
        self.assertEqual(after[2, 3], centered[2, 3])
        np.testing.assert_allclose(after[:2, 3] - centered[:2, 3], [0.016, -0.03], atol=1e-8)
        projected = after[:2, 3] / after[2, 3] * [100, 80] + [30, 20]
        np.testing.assert_allclose([39, 12] - projected, 0.8 * np.array([4, -6]), atol=1e-6)
        np.testing.assert_allclose(result["cup"], after @ tf_to_centered)
        np.testing.assert_allclose(result["cup"] @ np.linalg.inv(tf_to_centered), after)

    def test_invalid_projection_is_skipped_and_far_box_is_bounded(self):
        K = np.array([[100, 0, 30], [0, 100, 20], [0, 0, 1]])
        centered = np.eye(4)
        centered[2, 3] = 2
        shape = (40, 64)
        offset = np.zeros(2)
        self.assertIsNone(foundationpose.bbox_center_offset(centered, K, (0, 0, 5, 5), shape))
        np.testing.assert_allclose(
            foundationpose.xy_prior_delta(centered, K, (53, 30, 63, 40), offset, shape, 0.2), [0.02, 0.02],
        )
        centered[2, 3] = 0
        np.testing.assert_array_equal(
            foundationpose.xy_prior_delta(centered, K, (53, 30, 63, 40), offset, shape, 0.2), [0, 0],
        )
        centered[:3, 3] = [100, 0, 2]
        np.testing.assert_array_equal(
            foundationpose.xy_prior_delta(centered, K, (53, 30, 63, 40), offset, shape, 0.2), [0, 0],
        )
