"""几何语义和整段标注行为回归，无模型或设备依赖。"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/cordvip_geometry/src")]
from cordvip_geometry.camera import RGBDFrame, RGBDRectifier, camera_rays
from cordvip_geometry.mesh import MeshRaycaster, scale_mesh
from dexmani_real.geometry.hoi import sample_mesh_surface
from cordvip_geometry.tracking import TrackingFailure, SceneResult, TrackingConfig

def script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts/cordvip" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRawReader:
    num_frames = 3

    def __init__(self, path=None):
        self.data_file = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read_camera(self):
        color = SimpleNamespace(
            height=2,
            width=2,
            matrix=lambda: np.eye(3),
            distortion_model="none",
            distortion_coeffs=(0, 0, 0, 0, 0),
        )
        return SimpleNamespace(
            color=color, depth_scale_m=0.001, T_xarm_base_from_color=np.eye(4)
        )

    def require_fields(self, *names):
        pass

    def __getitem__(self, name):
        if name == "depth":
            return np.ones((3, 2, 2), np.uint16)
        return np.array(
            [(1, 1), (1, 1), (2, 2)],
            dtype=[("camera_sequence", "i8"), ("camera_ns", "i8")],
        )

    def iter_rgb_frames(self):
        yield from [np.zeros((2, 2, 3), np.uint8)] * 3


@pytest.fixture
def grounded_segmenter():
    import torch
    from cordvip_geometry.segmentation import SAM2Segmenter

    class Inputs(dict):
        input_ids = None

        def to(self, device):
            return self

    class Processor:
        def __call__(self, **kwargs):
            assert kwargs["text"] == "apple. banana."
            return Inputs()

        def post_process_grounded_object_detection(self, *args, **kwargs):
            return [self.detections]

    segmenter = SAM2Segmenter.__new__(SAM2Segmenter)
    segmenter.device = torch.device("cpu")
    segmenter.detector = lambda **kwargs: None
    segmenter.processor = Processor()
    return segmenter


def test_scale_preserves_origin_and_source():
    mesh = trimesh.creation.box()
    mesh.apply_translation([2, 3, 4])
    before = mesh.vertices.copy()
    scaled = scale_mesh(mesh, size_axis="x", size_meters=0.1)
    np.testing.assert_array_equal(mesh.vertices, before)
    np.testing.assert_allclose(scaled.vertices, before * 0.1)
    with pytest.raises(ValueError):
        scale_mesh(mesh, scale_to_meters=-1)


def test_fixed_surface_and_quota():
    module = script("sample_template")
    np.testing.assert_array_equal(
        module.allocate_counts(
            [0.020131764981252655, 0.021525464350244293, 0.099356038613691]
        ),
        [183, 191, 650],
    )
    np.testing.assert_array_equal(module.allocate_counts(np.ones(16)), np.full(16, 64))
    mesh = trimesh.creation.box()
    points = sample_mesh_surface(mesh)
    np.testing.assert_array_equal(points, sample_mesh_surface(mesh))
    np.testing.assert_allclose(np.abs(points).max(axis=1), 0.5)
    assert points.dtype == np.float32


def test_rectification_and_camera_z():
    K = np.array([[100.0, 0, 16], [0, 100.0, 16], [0, 0, 1.0]])
    rgb = np.zeros((32, 32, 3), np.uint8)
    depth = np.ones((32, 32), np.float32)
    r, d = RGBDRectifier(depth.shape, K, np.zeros(5)).apply(rgb, depth)
    assert r is rgb and d is depth
    pose = np.eye(4)
    pose[2, 3] = 1
    rays = camera_rays(depth.shape, K, 2)[[8, 7], [8, 8]]
    z = MeshRaycaster(trimesh.creation.box(extents=(0.08, 0.1, 0.12))).render(
        rays, pose
    )
    np.testing.assert_allclose(z, [0.94, 0.94], atol=1e-6)


def test_duplicate_camera_frame_and_tracking_failure():
    module = script("annotate_episodes")

    class Tracker:
        count = 0

        def initialize(self, frame, prompts):
            assert isinstance(frame, RGBDFrame)
            self.count += 1
            return SceneResult(
                {"apple": np.eye(4)}, {"apple": {"iou": 1.0}}, {"apple": True}
            )

        def track(self, frame):
            self.count += 1
            pose = np.eye(4)
            pose[0, 3] = 1
            return SceneResult(
                {"apple": pose}, {"apple": {"iou": 1.0}}, {"apple": True}
            )

    tracker = Tracker()
    poses, _, quality, valid = module.track_episode(
        FakeRawReader(), tracker, ["apple"], {"apple": "apple"}
    )
    assert tracker.count == 2
    assert quality[0] == quality[1]
    np.testing.assert_array_equal(poses[0], poses[1])
    assert poses[2, 0, 0, 3] == 1

    def fail(frame):
        raise TrackingFailure("lost")

    tracker.track = fail
    poses, _, _, valid = module.track_episode(
        FakeRawReader(), tracker, ["apple"], {"apple": "apple"}
    )
    assert not valid[2, 0] and np.isnan(poses[2]).all()


def test_offline_requires_camera_extrinsics(monkeypatch):
    reader = FakeRawReader()
    camera = reader.read_camera()
    camera.T_xarm_base_from_color = None
    monkeypatch.setattr(reader, "read_camera", lambda: camera)

    with pytest.raises(ValueError, match="相机外参"):
        script("annotate_episodes").track_episode(
            reader, None, ["apple"], {"apple": "apple"}
        )


@pytest.mark.parametrize("has_hoi", [False, True])
def test_existing_or_invalid_episode_never_writes(monkeypatch, tmp_path, has_hoi):
    module = script("annotate_episodes")
    reader = FakeRawReader()
    if has_hoi:
        reader.data_file["hoi"] = object()

    monkeypatch.setattr(module, "RawReader", lambda path: reader)
    monkeypatch.setattr(
        module, "load_task_template", lambda task: {"object_names": ["apple"]}
    )
    monkeypatch.setattr(
        module, "task_objects", lambda task: [{"name": "apple", "prompt": "apple"}]
    )

    def load_tracker(*args, **kwargs):
        assert not has_hoi, "existing HOI loaded tracking models"

    monkeypatch.setattr(module, "SceneTracker", load_tracker)

    def fail(*args):
        raise ValueError("invalid raw input")

    monkeypatch.setattr(module, "track_episode", fail)

    def write(*args):
        pytest.fail("existing or failed episode wrote HOI")

    monkeypatch.setattr(module, "write_hoi", write)
    if has_hoi:
        monkeypatch.setattr(
            module, "read_hoi_validity", lambda reader: np.ones(3, bool)
        )
        module.annotate_episodes([tmp_path / "episode"], tmp_path)
    else:
        with pytest.raises(ValueError, match="invalid raw input"):
            module.annotate_episodes([tmp_path / "episode"], tmp_path)


def test_template_and_raw_roundtrip(tmp_path):
    import json

    import h5py

    from dexmani_real.data.hoi import write_hoi
    from dexmani_real.geometry.hoi import load_task_template

    task = tmp_path / "task"
    mesh_dir = task / "apple/mesh"
    mesh_dir.mkdir(parents=True)
    (task / "task.json").write_text(json.dumps({"objects": [{"name": "apple"}]}))
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(mesh_dir / "mesh_m.obj")
    module = script("sample_template")
    module.sample_template(task)
    template = load_task_template(task)
    original = (task / "hoi/template.npz").read_bytes()
    module.sample_template(task)
    assert (task / "hoi/template.npz").read_bytes() == original
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset("timing/rows", data=np.arange(129))
    poses = np.tile(np.eye(4), (129, 1, 1, 1))
    poses[1, 0, 0, 3] = 1
    poses[128, 0, 1, 3] = 2
    write_hoi(
        path,
        template,
        poses,
        np.eye(4),
        quality=[[{"iou": 1.0}]] * len(poses),
        tracking_config={},
        object_valid=np.ones((len(poses), 1), bool),
    )
    with h5py.File(path, "r") as data:
        assert "_hoi_staging" not in data
        points = data["hoi/object_pointcloud"][:]
        np.testing.assert_allclose(points[0], template["points_local"])
        np.testing.assert_allclose(
            points[1], template["points_local"] + [1, 0, 0], atol=1e-7
        )
        np.testing.assert_allclose(
            points[128], template["points_local"] + [0, 2, 0], atol=2e-7
        )


def test_batched_template_transform():
    from dexmani_real.geometry.hoi import transform_task_points

    rng = np.random.default_rng(0)
    template = {
        "points_local": rng.uniform(-0.1, 0.1, (1024, 3)).astype(np.float32),
        "offsets": np.array([0, 183, 374, 1024]),
    }
    poses = np.tile(np.eye(4, dtype=np.float32), (5, 3, 1, 1))
    poses[:, :, :3, 3] = rng.random((5, 3, 3))
    poses[:, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    base = np.eye(4)
    base[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    base[:3, 3] = [0.2, -0.3, 0.1]
    actual = transform_task_points(template, poses, base)
    single = np.stack([transform_task_points(template, pose, base) for pose in poses])
    np.testing.assert_array_equal(actual, single)
    assert actual.shape == (5, 1024, 3) and actual.dtype == np.float32
    for frame in range(5):
        for obj, (start, end) in enumerate(
            zip(template["offsets"][:-1], template["offsets"][1:])
        ):
            camera = (
                template["points_local"][start:end] @ poses[frame, obj, :3, :3].T
                + poses[frame, obj, :3, 3]
            )
            expected = camera @ base[:3, :3].T + base[:3, 3]
            np.testing.assert_allclose(actual[frame, start:end], expected, atol=2e-7)


def test_scene_batch_keeps_per_object_depth_gate(monkeypatch):
    from cordvip_geometry import tracking

    shape = (24, 24)
    mask = np.ones(shape, bool)
    scene = tracking.SceneTracker.__new__(tracking.SceneTracker)
    scene.rectifier = RGBDRectifier(shape, np.eye(3), np.zeros(5))
    scene.directions = camera_rays(shape, np.eye(3), tracking.PIXEL_STRIDE)
    scene.segmenter = SimpleNamespace(track=lambda rgb: {"apple": mask, "banana": mask})
    scene.trackers = {
        n: SimpleNamespace(estimator=SimpleNamespace(pose_last=np.eye(4)))
        for n in ("apple", "banana")
    }
    scene.states = dict.fromkeys(scene.trackers, "TRACKING")
    scene.next_retry = dict.fromkeys(scene.trackers, 0)
    scene.frame_index = 0
    scene.last_sam_frame = 0
    scene.config = TrackingConfig(recovery_enabled=False, segmentation_interval_frames=1)
    scene.frame_signature = (shape, np.eye(3), "none", np.zeros(5))
    scene.frame_cache = {}
    scene.renderers = {
        "apple": SimpleNamespace(render=lambda rays, pose: np.ones(len(rays))),
        "banana": SimpleNamespace(render=lambda rays, pose: np.full(len(rays), 2.0)),
    }
    calls = []

    def batch(trackers, rgb, depth, K, frame_cache):
        assert frame_cache is scene.frame_cache
        calls.append(list(trackers))
        return {name: np.eye(4) for name in trackers}

    monkeypatch.setattr(tracking, "track_objects", batch)
    frame = RGBDFrame(
        rgb=np.zeros((*shape, 3), np.uint8),
        depth_m=np.ones(shape, np.float32),
        K=np.eye(3),
        distortion_model="none",
        distortion_coeffs=np.zeros(5),
    )
    result = scene.track(frame)
    assert result.valid == {"apple": True, "banana": False}
    assert list(result.poses) == ["apple"]
    assert calls == [["apple", "banana"]]
    mask[:] = False
    assert not any(scene.track(frame).valid.values())
    assert len(calls) == 1


@pytest.mark.parametrize("fails", [False, True])
def test_registration_computation_settings_are_local(fails):
    import torch
    from cordvip_geometry.foundationpose import FoundationPoseTracker

    tracker = FoundationPoseTracker.__new__(FoundationPoseTracker)

    def register(**kwargs):
        assert torch.backends.cudnn.enabled
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert torch.zeros(1).device.type == "meta"
        if fails:
            raise TrackingFailure("lost")
        tracker.estimator.pose_last = np.eye(4)
        return np.eye(4)

    tracker.estimator = SimpleNamespace(register=register)

    with (
        torch.device("meta"),
        torch.backends.cudnn.flags(enabled=False, deterministic=False, benchmark=True),
    ):
        if fails:
            with pytest.raises(TrackingFailure, match="lost"):
                tracker.register(None, None, None, None)
        else:
            np.testing.assert_array_equal(
                tracker.register(None, None, None, None), np.eye(4)
            )
        assert torch.zeros(1).device.type == "meta"
        assert not torch.backends.cudnn.enabled
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark


@pytest.mark.parametrize(
    "options",
    [
        ["--size-meters", "0.1"],
        ["--size-axis", "x"],
        ["--scale-to-meters", "0.1", "--size-axis", "x", "--size-meters", "0.1"],
        ["--scale-to-meters", "nan"],
        ["--scale-to-meters", "-1"],
    ],
)
def test_cli_rejects_invalid_scale_before_inference(options):
    from cordvip_geometry.cli import parse_args

    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "reconstruct",
                "--image",
                "unused.jpg",
                "--prompt",
                "apple",
                "--output",
                "unused",
                *options,
            ]
        )
    assert error.value.code == 2


def test_cli_mesh_scale_roundtrip(tmp_path):
    from cordvip_geometry.cli import main
    from cordvip_geometry.mesh import load_mesh

    source = tmp_path / "source.obj"
    mesh = trimesh.creation.box()
    mesh.apply_translation([2, 3, 4])
    mesh.export(source)
    assert (
        main(
            [
                "scale",
                "--mesh",
                str(source),
                "--scale-to-meters",
                "0.1",
                "--output",
                str(tmp_path / "metric"),
            ]
        )
        == 0
    )
    np.testing.assert_allclose(
        load_mesh(tmp_path / "metric/mesh_m.obj").vertices, mesh.vertices * 0.1
    )


def test_prepare_task_requires_scale_before_models(tmp_path):
    import json

    (tmp_path / "task.json").write_text(json.dumps({"objects": [{"name": "apple"}]}))
    with pytest.raises(ValueError, match="完整实测尺寸"):
        script("prepare_task").prepare_task(str(tmp_path))


def test_offline_uses_template_order(monkeypatch, tmp_path):
    module = script("annotate_episodes")
    template = {"object_names": ["banana", "apple"]}

    monkeypatch.setattr(module, "RawReader", FakeRawReader)
    monkeypatch.setattr(module, "load_task_template", lambda task: template)
    monkeypatch.setattr(
        module,
        "task_objects",
        lambda task: [{"name": n, "prompt": n} for n in ["apple", "banana"]],
    )

    def tracker(paths, **kwargs):
        assert list(paths) == ["banana", "apple"]
        return SimpleNamespace(config=TrackingConfig())

    def track(reader, tracker, names, prompts):
        assert list(names) == list(prompts) == ["banana", "apple"]
        return np.zeros((3, 2, 4, 4)), np.eye(4), [], np.ones((3, 2), bool)

    monkeypatch.setattr(module, "SceneTracker", tracker)
    monkeypatch.setattr(module, "track_episode", track)
    monkeypatch.setattr(module, "write_hoi", lambda *args, **kwargs: None)
    module.annotate_episodes([tmp_path / "episode"], str(tmp_path))


def test_shape_loader_preserves_environment(monkeypatch, tmp_path):
    import os

    import torch
    from cordvip_geometry.reconstruction import load_shape_model

    weights = tmp_path / "hunyuan3d-dit-v2-1"
    weights.mkdir()
    (weights / "config.yaml").write_text(
        "conditioner:\n  params:\n    main_image_encoder:\n"
        "      kwargs:\n        config: {}\n"
    )

    class Component:
        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def enable_flashvdm_decoder(self, **kwargs):
            pass

    pipeline = SimpleNamespace(
        **{n: Component() for n in ("model", "vae", "conditioner")}
    )
    module = SimpleNamespace(
        Hunyuan3DDiTFlowMatchingPipeline=SimpleNamespace(
            from_single_file=lambda *args, **kwargs: pipeline
        )
    )
    monkeypatch.setitem(sys.modules, "hy3dshape", module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    assert load_shape_model(tmp_path, tmp_path) is pipeline
    assert all(key not in os.environ for key in keys)


@pytest.mark.parametrize("empty", [False, True])
def test_reconstruction_removes_degenerate_faces(monkeypatch, empty):
    import torch
    from cordvip_geometry.reconstruction import reconstruct_mesh
    from PIL import Image

    source = trimesh.creation.box()
    faces = [[0, 0, 1]] if empty else np.vstack((source.faces, [0, 0, 1]))
    mesh = trimesh.Trimesh(
        vertices=np.vstack((source.vertices, [10, 10, 10])),
        faces=faces,
        process=False,
    )
    rgba = np.full((2, 2, 4), 255, np.uint8)
    rgba[0, 0, 3] = 0
    generator = torch.Generator()
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: generator)

    def pipeline(**kwargs):
        return [mesh]

    if empty:
        with pytest.raises(RuntimeError, match="degenerate geometry"):
            reconstruct_mesh(pipeline, Image.fromarray(rgba))
    else:
        result = reconstruct_mesh(pipeline, Image.fromarray(rgba))
        np.testing.assert_array_equal(result.vertices, source.vertices)
        np.testing.assert_array_equal(result.faces, source.faces)


@pytest.mark.parametrize(
    "scale",
    [
        {"scale_to_meters": 0},
        {"scale_to_meters": float("nan")},
        {"size_axis": "w", "size_meters": 0.1},
    ],
)
def test_prepare_task_rejects_invalid_measurement(tmp_path, scale):
    import json

    (tmp_path / "task.json").write_text(
        json.dumps({"objects": [{"name": "apple", **scale}]})
    )
    with pytest.raises(ValueError):
        script("prepare_task").prepare_task(tmp_path)


def test_npz_rgb_checked_before_model(monkeypatch, tmp_path):
    from cordvip_geometry import segmentation
    from cordvip_geometry.cli import main

    def unexpected(*args, **kwargs):
        pytest.fail("invalid input loaded the model")

    monkeypatch.setattr(segmentation, "SAM2Segmenter", unexpected)
    source = tmp_path / "rgbd.npz"
    np.savez(source, rgb=np.ones((4, 4, 3), np.float32))
    with pytest.raises(ValueError, match="uint8"):
        main(
            [
                "segment",
                "--rgbd",
                str(source),
                "--prompt",
                "apple",
                "--headless",
                "--output",
                str(tmp_path / "out"),
            ]
        )


def test_detection_nms_groups_labels_and_preserves_instances(grounded_segmenter):
    import torch

    grounded_segmenter.processor.detections = {
        "boxes": torch.tensor(
            [
                [-1, -1, 20, 20],
                [0, 0, 20, 20],
                [0, 0, 20, 20],
                [10, 0, 30, 20],
                [40, 0, 45, 20],
                [2, 2, 2, 8],
            ],
            dtype=torch.float32,
        ),
        "scores": torch.tensor([0.9, 0.7, 0.8, 0.6, 0.99, 0.95]),
        "labels": ["apple", " Apple. ", "banana", "apple", "apple", "banana"],
    }
    candidates = grounded_segmenter.detect(
        np.zeros((32, 32, 3), np.uint8), "Apple. banana"
    )
    assert [item["label"] for item in candidates] == ["apple", "banana", "apple"]
    np.testing.assert_allclose([item["score"] for item in candidates], [0.9, 0.8, 0.6])
    np.testing.assert_array_equal(
        [item["box_xyxy"] for item in candidates],
        [[0, 0, 20, 20], [0, 0, 20, 20], [10, 0, 30, 20]],
    )
    assert all(set(item) == {"box_xyxy", "score", "label"} for item in candidates)


@pytest.mark.parametrize("boxes", [[], [[-3, 0, -1, 2], [2, 2, 2, 3]]])
def test_detection_empty_after_clipping(grounded_segmenter, boxes):
    import torch

    grounded_segmenter.processor.detections = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.ones(len(boxes)),
        "labels": ["apple"] * len(boxes),
    }
    assert (
        grounded_segmenter.detect(np.zeros((32, 32, 3), np.uint8), "apple. banana.")
        == []
    )


def test_single_image_segmentation_keeps_candidate_interface(monkeypatch):
    from contextlib import nullcontext

    import torch
    from cordvip_geometry.segmentation import SAM2Segmenter

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    segmenter = SAM2Segmenter.__new__(SAM2Segmenter)
    rgb = np.zeros((8, 8, 3), np.uint8)
    candidates = [
        {"box_xyxy": np.array([0, 0, 4, 4]), "score": 0.9, "label": "apple"},
        {"box_xyxy": np.array([4, 4, 8, 8]), "score": 0.8, "label": "apple"},
    ]
    segmenter.detect = lambda *args: candidates
    calls = []

    def predict(**kwargs):
        np.testing.assert_array_equal(
            kwargs["box"], [item["box_xyxy"] for item in candidates]
        )
        masks = np.zeros((2, 1, 8, 8), np.float32)
        masks[1, 0, 4:, 4:] = 1
        return masks, None, None

    segmenter.image_predictor = SimpleNamespace(set_image=calls.append, predict=predict)
    result = segmenter.predict(rgb, "apple")
    assert len(calls) == 1 and calls[0] is rgb
    assert len(result) == 1 and result[0]["score"] == 0.8
    assert result[0]["label"] == "apple" and result[0]["mask"].dtype == bool
    np.testing.assert_array_equal(result[0]["box_xyxy"], candidates[1]["box_xyxy"])
    candidates.clear()
    assert segmenter.predict(rgb, "apple") == [] and len(calls) == 1


def test_scene_initialization_rectifies_before_detection_and_resets_state():
    import cv2
    from cordvip_geometry.tracking import SceneTracker

    shape = (32, 32)
    K = np.array([[20.0, 0, 16], [0, 20.0, 16], [0, 0, 1]])
    rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(*shape, 3)
    depth = np.ones(shape, np.float32)
    boxes = {"fruit": np.array([0, 0, 16, 32]), "dish": np.array([16, 0, 32, 32])}
    masks = {"fruit": np.indices(shape)[1] < 16, "dish": np.indices(shape)[1] >= 16}
    scene = SceneTracker.__new__(SceneTracker)
    scene.config = TrackingConfig()
    scene.trackers = {
        name: SimpleNamespace(estimator=SimpleNamespace(pose_last=np.eye(4)))
        for name in boxes
    }
    scene.renderers = {
        name: SimpleNamespace(render=lambda rays, pose: np.ones(len(rays)))
        for name in boxes
    }
    received = []

    def detect(image, text):
        assert all(item.estimator.pose_last is None for item in scene.trackers.values())
        assert scene.frame_cache == {}
        np.testing.assert_array_equal(image, expected_rgb)
        return [
            {"box_xyxy": boxes["dish"], "label": "bowl", "score": 0.9},
            {"box_xyxy": boxes["fruit"], "label": "apple.", "score": 0.8},
        ]

    def initialize(image, detected_boxes):
        np.testing.assert_array_equal(image, expected_rgb)
        assert list(detected_boxes) == ["fruit", "dish"]
        for name, box in boxes.items():
            np.testing.assert_array_equal(detected_boxes[name], box)
        return masks

    def register(image, observed, intrinsics, mask):
        np.testing.assert_array_equal(image, expected_rgb)
        np.testing.assert_array_equal(observed, expected_depth)
        np.testing.assert_array_equal(intrinsics, K)
        received.append(mask)
        return np.eye(4)

    scene.segmenter = SimpleNamespace(detect=detect, initialize=initialize)
    for tracker in scene.trackers.values():
        tracker.register = register
    for distortion in [(0.3, 0, 0, 0, 0), (0, 0, 0, 0, 0)]:
        maps = cv2.initUndistortRectifyMap(
            K, np.array(distortion, dtype=float), None, K, shape[::-1], cv2.CV_32FC1
        )
        expected_rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
        expected_depth = cv2.remap(depth, *maps, cv2.INTER_NEAREST)
        for tracker in scene.trackers.values():
            tracker.estimator.pose_last = np.eye(4)
        scene.frame_cache = {"K": K, "rgb": rgb}
        frame = RGBDFrame(rgb, depth, K, "brown_conrady", distortion)
        result = scene.initialize(frame, {"fruit": " Apple. ", "dish": "bowl"})
        assert list(result.poses) == ["fruit", "dish"]
        assert received[-2] is masks["fruit"] and received[-1] is masks["dish"]
    assert scene.rectifier.maps is None


@pytest.mark.parametrize("count", [0, 2])
def test_scene_rejects_missing_or_ambiguous_detection(count):
    from cordvip_geometry.tracking import SceneTracker

    scene = SceneTracker.__new__(SceneTracker)
    scene.config = TrackingConfig()
    scene.trackers = {
        "fruit": SimpleNamespace(estimator=SimpleNamespace(pose_last=None))
    }
    scene.segmenter = SimpleNamespace(detect=lambda *args: [{"label": "apple"}] * count)
    frame = RGBDFrame(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), np.eye(3), "none", (0,) * 5
    )
    with pytest.raises(
        TrackingFailure, match=f"fruit: expected one detection for 'apple', got {count}"
    ):
        scene.initialize(frame, {"fruit": "apple"})


def test_pose_quality_requires_two_sided_evidence():
    from cordvip_geometry.tracking import pose_quality

    config = TrackingConfig()
    target = np.zeros((40, 40), bool)
    target[10:30, 10:30] = True
    rendered = target.astype(float)
    observed = np.where(target, 1.0, 2.0)
    assert pose_quality(target, observed, rendered, config)[1]
    # mask 内完全匹配仍不能接受轮廓过大的模型。
    assert not pose_quality(target, observed, np.ones_like(rendered), config)[1]
    visible = target.copy()
    visible[:, 20:] = False
    occluded_depth = observed.copy()
    occluded_depth[target & ~visible] = 0.8
    metrics, valid = pose_quality(visible, occluded_depth, rendered, config)
    assert valid and metrics["occluded_fraction"] == 0.5
    occluded_depth[target & ~visible] = 0
    # 缺失深度不能被计为遮挡。
    assert pose_quality(visible, occluded_depth, rendered, config)[0]["occluded_fraction"] == 0
    assert not pose_quality(target, observed, rendered * 1.1, config)[1]


def test_joint_interpolation_boundaries_and_conflicts():
    from dexmani_real.utils.timing import interpolate_joints

    stamps = np.array([100, 150, 150, 200]) * 1_000_000
    q = np.array([[0.0], [1.0], [1.0], [2.0]])
    target = np.array([90, 100, 125, 200, 210]) * 1_000_000
    values, valid = interpolate_joints(stamps, q, target, 0.1)
    np.testing.assert_array_equal(valid, [False, True, True, True, False])
    np.testing.assert_allclose(values[valid, 0], [0, 0.5, 2])
    assert not interpolate_joints(stamps, q, [125_000_000], 0.01)[1][0]
    q[2] = 3
    with pytest.raises(ValueError, match="Conflicting"):
        interpolate_joints(stamps, q, target, 0.1)


def test_fps_surface_is_fixed_and_improves_coverage():
    from dexmani_real.geometry.hoi import sample_surface_points
    from scipy.spatial import cKDTree

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    points = sample_surface_points(mesh, 256)
    np.testing.assert_array_equal(points, sample_surface_points(mesh, 256))
    np.testing.assert_allclose(np.max(np.abs(points), axis=1), 0.05)
    reference = sample_mesh_surface(mesh, 20000, seed=19)
    old = cKDTree(sample_mesh_surface(mesh, 256)).query(reference)[0]
    new = cKDTree(points).query(reference)[0]
    assert np.percentile(new, 95) < 0.8 * np.percentile(old, 95)


@pytest.mark.parametrize("devices", [("arm",), ("arm", "hand")])
def test_live_joint_history_matches_offline_interpolation(devices):
    from collections import deque
    from dexmani_real.robot.system import Robot

    robot = Robot.__new__(Robot)
    robot.enable_joint_history(0.35, devices=devices)
    for name in devices:
        n = 7 if name == "arm" else 12
        robot.joint_history[name] = deque(
            [(1_000_000_000, np.zeros(n)), (1_080_000_000, np.ones(n))]
        )
    aligned = robot.aligned_joints(1_020_000_000, 0.1)
    assert set(aligned) == {name + "_qpos" for name in devices}
    for positions in aligned.values():
        np.testing.assert_allclose(positions, 0.25)
    assert robot.aligned_joints(1_090_000_000, 0.1) is None
    robot.enable_joint_history(0.35, devices=devices)
    assert robot.aligned_joints(1_020_000_000, 0.1) is None


@pytest.fixture
def recovery_scene(monkeypatch):
    from cordvip_geometry import tracking

    shape = (24, 24)
    frame = RGBDFrame(
        np.zeros((*shape, 3), np.uint8),
        np.ones(shape, np.float32),
        np.eye(3),
        "none",
        np.zeros(5),
    )
    scene = tracking.SceneTracker.__new__(tracking.SceneTracker)
    scene.config = TrackingConfig(recovery_retry_interval_frames=3, segmentation_interval_frames=1)
    masks = {n: np.ones(shape, bool) for n in ("apple", "banana")}
    scene.trackers = {}
    registrations = []
    for name in masks:
        estimator = SimpleNamespace(pose_last=None)

        def register(rgb, depth, K, mask, name=name, estimator=estimator):
            registrations.append(name)
            estimator.pose_last = np.eye(4)
            return np.eye(4)

        scene.trackers[name] = SimpleNamespace(estimator=estimator, register=register)
    scene.renderers = {
        n: SimpleNamespace(render=lambda rays, pose: np.full(len(rays), pose[2, 2]))
        for n in masks
    }
    detections = []

    def detect(*args):
        detections.append(1)
        return [{"label": n, "box_xyxy": np.array([0, 0, 24, 24])} for n in masks]

    scene.segmenter = SimpleNamespace(
        detect=detect,
        initialize=lambda *args: masks.copy(),
        track=lambda rgb: masks.copy(),
        segment_boxes=lambda rgb, boxes: {n: np.ones(shape, bool) for n in boxes},
        correct_masks=lambda new: {**masks, **new},
    )

    def batch(trackers, *args):
        result = {}
        for n, t in trackers.items():
            t.estimator.pose_last = np.eye(4)
            result[n] = np.eye(4)
        return result

    monkeypatch.setattr(tracking, "track_objects", batch)
    scene.initialize(frame, {name: name for name in masks})
    return scene, frame, masks, registrations, detections


def test_object_recovery_and_identity(recovery_scene):
    scene, frame, masks, registrations, detections = recovery_scene
    masks["apple"][:] = False
    lost = scene.track(frame)
    assert lost.valid == {"apple": False, "banana": True}
    assert list(lost.poses) == ["banana"]
    recovered = scene.track(frame)
    assert all(recovered.valid.values())
    assert list(recovered.poses) == ["apple", "banana"]
    assert registrations == ["apple", "banana", "apple"]
    assert len(detections) == 2


def test_failed_recovery_throttles_and_rolls_back(recovery_scene):
    scene, frame, masks, registrations, detections = recovery_scene
    previous = scene.trackers["apple"].estimator.pose_last.copy()
    masks["apple"][:] = False
    scene.track(frame)

    def failed(*args):
        scene.trackers["apple"].estimator.pose_last = np.full((4, 4), 99.0)
        raise TrackingFailure("registration failed")

    scene.trackers["apple"].register = failed
    for _ in range(7):
        result = scene.track(frame)
        assert not result.valid["apple"] and result.valid["banana"]
        assert "apple" not in result.poses
        np.testing.assert_array_equal(
            scene.trackers["apple"].estimator.pose_last, previous
        )
    assert len(detections) == 4  # 初始化，以及第 2、5、8 帧的恢复尝试。


def test_quality_rejection_restores_internal_pose(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, _, _, _ = recovery_scene

    def batch(trackers, *args):
        poses = {n: np.eye(4) for n in trackers}
        poses["apple"][2, 2] = 2
        for n in trackers:
            trackers[n].estimator.pose_last = poses[n].copy()
        return poses

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    np.testing.assert_array_equal(
        scene.trackers["apple"].estimator.pose_last, np.eye(4)
    )


def test_invalid_hoi_storage_requires_explicit_validity(tmp_path):
    import h5py
    from dexmani_real.data.hoi import write_hoi, read_hoi_validity

    path = tmp_path / "data.h5"
    template = dict(
        points_local=np.ones((1024, 3), np.float32),
        offsets=[0, 1024],
        object_names=["apple"],
    )
    with h5py.File(path, "w") as data:
        data.create_dataset("timing/rows", data=np.arange(3))
    poses = np.tile(np.eye(4), (3, 1, 1, 1))
    write_hoi(
        path,
        template,
        poses,
        np.eye(4),
        quality=[[{"iou": 1}], [{}], [{"iou": 1}]],
        tracking_config={},
        object_valid=np.array([[True], [False], [True]]),
    )
    with h5py.File(path, "r+") as data:
        reader = SimpleNamespace(data_file=data, num_frames=3)
        np.testing.assert_array_equal(read_hoi_validity(reader), [True, False, True])
        assert np.isnan(data["hoi/object_pose_camera"][1]).all()
        assert not data["hoi/object_pointcloud"][1].any()
        assert data["hoi/object_pointcloud"][2].all()
        del data["hoi/valid"]
        del data["hoi/object_valid"]
        with pytest.raises(KeyError, match="valid"):
            read_hoi_validity(reader)


def test_initialization_failure_retries_offline():
    module = script("annotate_episodes")
    calls = []

    def initialize(*args):
        calls.append(1)
        if len(calls) == 1:
            raise TrackingFailure("initial registration")
        return SceneResult({"apple": np.eye(4)}, {"apple": {"iou": 1}}, {"apple": True})

    tracker = SimpleNamespace(
        initialize=initialize, track=lambda frame: pytest.fail("premature track")
    )
    poses, _, _, valid = module.track_episode(
        FakeRawReader(), tracker, ["apple"], {"apple": "apple"}
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(valid[:, 0], [False, False, True])
    assert np.isnan(poses[:2]).all()


def test_worker_retries_initial_registration(monkeypatch, tmp_path):
    from cordvip_geometry import tracking
    from dexmani_real.perception.hoi_worker import run_hoi_worker
    from dexmani_real.geometry import hoi

    monkeypatch.setattr(
        hoi, "task_objects", lambda task: [{"name": "apple", "prompt": "apple"}]
    )
    monkeypatch.setattr(RGBDFrame, "from_camera", lambda *args: object())
    monkeypatch.setattr("signal.signal", lambda *args: None)
    calls, written = [], []

    def initialize(*args):
        calls.append("initialize")
        if len(calls) == 1:
            raise TrackingFailure("initial registration")
        return SceneResult({"apple": np.eye(4)}, {"apple": {}}, {"apple": True})

    monkeypatch.setattr(
        tracking,
        "SceneTracker",
        lambda *args, **kwargs: SimpleNamespace(
            initialize=initialize, track=lambda frame: pytest.fail("premature track")
        ),
    )
    shared = SimpleNamespace(
        is_running=SimpleNamespace(value=True), hoi_ready=SimpleNamespace(value=False)
    )
    shared.hoi_request = SimpleNamespace(
        read=lambda: ({"episode_id": 1, "after_sequence": 0, "after_ns": 0}, 1)
    )
    frames = iter([({"timestamp_ns": 1}, 1), ({"timestamp_ns": 2}, 2)])
    shared.camera_ring = SimpleNamespace(read=lambda: next(frames))

    def write(**kwargs):
        written.append(kwargs)
        if len(written) == 2:
            shared.is_running.value = False

    shared.hoi_ring = SimpleNamespace(field_names=(), write=write)
    template = dict(
        object_names=["apple"],
        points_local=np.ones((1024, 3), np.float32),
        offsets=[0, 1024],
    )
    run_hoi_worker(
        shared, SimpleNamespace(T_xarm_base_from_color=np.eye(4)), tmp_path, template
    )
    assert calls == ["initialize", "initialize"]
    assert [r["valid"] for r in written] == [False, True]
    assert not written[0]["object_pointcloud"].any()
    assert [r["source_camera_sequence"] for r in written] == [1, 2]


def test_correction_history_prunes_inputs_with_outputs():
    from cordvip_geometry.segmentation import prune_state

    def outputs():
        return {
            "cond_frame_outputs": {0: object()},
            "non_cond_frame_outputs": {i: object() for i in range(1, 10)},
        }

    state = dict(
        output_dict=outputs(),
        output_dict_per_obj={0: outputs()},
        temp_output_dict_per_obj={0: outputs()},
        frames_already_tracked=dict.fromkeys(range(10)),
        consolidated_frame_inds={
            "cond_frame_outputs": {0},
            "non_cond_frame_outputs": {2, 8},
        },
        point_inputs_per_obj={0: {0: object()}},
        mask_inputs_per_obj={0: {2: object(), 8: object()}},
    )
    prune_state(state, 9, 3)
    assert set(state["mask_inputs_per_obj"][0]) == {8}
    assert set(state["point_inputs_per_obj"][0]) == {0}
    assert state["consolidated_frame_inds"]["non_cond_frame_outputs"] == {8}
    assert set(state["output_dict"]["non_cond_frame_outputs"]) == {7, 8, 9}


def test_ambiguous_recovery_is_not_assigned(recovery_scene):
    scene, frame, masks, registrations, _ = recovery_scene
    masks["apple"][:] = False
    scene.track(frame)
    scene.segmenter.detect = lambda *args: [{"label": "apple"}, {"label": " Apple. "}]
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    assert registrations == ["apple", "banana"]


def test_batch_exception_isolates_object(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, _, _, _ = recovery_scene

    def batch(trackers, *args):
        for tracker in trackers.values():
            tracker.estimator.pose_last = np.full((4, 4), 9.0)
        if "apple" in trackers:
            raise TrackingFailure("apple failed")
        return {"banana": np.eye(4)}

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    np.testing.assert_array_equal(
        scene.trackers["apple"].estimator.pose_last, np.eye(4)
    )


def test_offline_temporary_loss_preserves_frame_alignment():
    module = script("annotate_episodes")

    class Reader(FakeRawReader):
        def __getitem__(self, name):
            if name == "depth":
                return super().__getitem__(name)
            return np.array(
                [(1, 1), (2, 2), (3, 3)],
                dtype=[("camera_sequence", "i8"), ("camera_ns", "i8")],
            )

    recovered_pose = np.eye(4)
    recovered_pose[0, 3] = 2
    results = iter(
        [
            SceneResult({}, {"apple": {}}, {"apple": False}),
            SceneResult(
                {"apple": recovered_pose}, {"apple": {"iou": 1}}, {"apple": True}
            ),
        ]
    )
    tracker = SimpleNamespace(
        initialize=lambda *args: SceneResult(
            {"apple": np.eye(4)}, {"apple": {"iou": 1}}, {"apple": True}
        ),
        track=lambda *args: next(results),
    )
    poses, _, quality, valid = module.track_episode(
        Reader(), tracker, ["apple"], {"apple": "apple"}
    )
    np.testing.assert_array_equal(valid[:, 0], [True, False, True])
    assert poses.shape == (3, 1, 4, 4)
    assert np.isnan(poses[1]).all() and poses[2, 0, 0, 3] == 2
    assert quality[1] == [{}]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(recovery_enabled=1),
        dict(recovery_retry_interval_frames=0),
        dict(recovery_retry_interval_frames=True),
        dict(recovery_box_threshold=float("nan")),
        dict(recovery_text_threshold=1.1),
    ],
)
def test_invalid_recovery_config(kwargs):
    with pytest.raises(ValueError):
        TrackingConfig(**kwargs)


def test_recovery_rejection_after_sam_merge_keeps_retry_interval(recovery_scene):
    scene, frame, masks, registrations, detections = recovery_scene
    masks["apple"][:] = False
    previous = scene.trackers["apple"].estimator.pose_last
    scene.track(frame)

    def merged_masks(corrections):
        assert scene.states["apple"] == "LOST"
        return masks.copy()  # 合并后的 Apple mask 仍为空，必须拒绝恢复。

    scene.segmenter.correct_masks = merged_masks
    for _ in range(3):  # 第 2 帧尝试；第 3、4 帧等待。
        result = scene.track(frame)
        assert result.valid == {"apple": False, "banana": True}
        assert scene.trackers["apple"].estimator.pose_last is previous
    assert len(detections) == 2
    scene.track(frame)  # 第 5 帧重试。
    assert len(detections) == 3
    assert registrations == ["apple", "banana", "apple", "apple"]


@pytest.mark.parametrize("stage", ["batch", "detection", "registration"])
def test_programming_errors_are_not_treated_as_tracking_loss(
    recovery_scene, monkeypatch, stage
):
    from cordvip_geometry import tracking

    scene, frame, masks, _, _ = recovery_scene

    def malformed(*args):
        raise KeyError("malformed model output")

    if stage == "batch":
        monkeypatch.setattr(tracking, "track_objects", malformed)
    else:
        masks["apple"][:] = False
        scene.track(frame)
        if stage == "detection":
            scene.segmenter.detect = malformed
        else:
            scene.trackers["apple"].register = malformed
    with pytest.raises(KeyError, match="malformed model output"):
        scene.track(frame)


@pytest.mark.parametrize("stage", ["track", "correction"])
def test_shared_sam_failure_is_not_silently_retried(recovery_scene, stage):
    scene, frame, masks, _, _ = recovery_scene
    previous = {
        name: tracker.estimator.pose_last for name, tracker in scene.trackers.items()
    }

    def broken(*args):
        raise RuntimeError("shared SAM state failure")

    if stage == "track":
        scene.segmenter.track = broken
    else:
        masks["apple"][:] = False
        scene.track(frame)
        previous = {
            name: tracker.estimator.pose_last
            for name, tracker in scene.trackers.items()
        }
        scene.segmenter.correct_masks = broken
    with pytest.raises(RuntimeError, match="shared SAM state failure") as error:
        scene.track(frame)
    assert type(error.value) is RuntimeError
    for name, tracker in scene.trackers.items():
        assert tracker.estimator.pose_last is previous[name]


def test_nonfinite_candidate_only_invalidates_its_object(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, _, _, detections = recovery_scene
    original = tracking.track_objects
    calls = []
    previous = scene.trackers["apple"].estimator.pose_last

    def nonfinite(trackers, *args):
        calls.append(list(trackers))
        poses = original(trackers, *args)
        poses["apple"][:] = np.nan
        trackers["apple"].estimator.pose_last = poses["apple"]
        return poses

    monkeypatch.setattr(tracking, "track_objects", nonfinite)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    assert scene.trackers["apple"].estimator.pose_last is previous
    assert calls == [["apple", "banana"]] and len(detections) == 1


def test_annotator_rejects_existing_hoi_without_validity(monkeypatch, tmp_path):
    import h5py

    module = script("annotate_episodes")
    with h5py.File(tmp_path / "data.h5", "w") as data:
        data.create_dataset("timing/rows", data=np.arange(3))
        data.create_dataset(
            "hoi/object_pointcloud", data=np.zeros((3, 1024, 3), np.float32)
        )
    monkeypatch.setattr(
        module, "load_task_template", lambda task: {"object_names": ["apple"]}
    )
    monkeypatch.setattr(
        module, "task_objects", lambda task: [{"name": "apple", "prompt": "apple"}]
    )
    monkeypatch.setattr(
        module,
        "SceneTracker",
        lambda *args, **kwargs: pytest.fail("unexpected model load"),
    )
    with pytest.raises(KeyError, match="valid"):
        module.annotate_episodes([tmp_path], tmp_path)


def test_viewer_keeps_joint_alignment_separate_from_object_validity(monkeypatch):
    from scripts import visualize_episode as module

    viewer = module.EpisodeVisualizer.__new__(module.EpisodeVisualizer)
    logged, clouds = {}, []
    viewer.rr = SimpleNamespace(
        Scalar=lambda value: value,
        log=lambda name, value: logged.update({name: value}),
        set_time_sequence=lambda *args: None,
        set_time_seconds=lambda *args: None,
    )
    viewer.hoi_alignment = {
        "keep": np.array([False]),
        "arm_qpos": np.zeros((1, 7)),
        "hand_qpos": np.zeros((1, 12)),
    }
    viewer.reader = {"timing/rows": [{"camera_ns": 1, "arm_ns": 1, "hand_ns": 1}]}

    class Reader(dict):
        data_file = {"hoi/quality": {}}

    viewer.reader = Reader(viewer.reader)
    viewer.log_hoi_diagnostics(0, viewer.hoi_alignment)
    assert logged["hoi/alignment_available"] == 1
    viewer.has_hoi = True
    viewer.hoi_valid = [False]
    viewer.timestamps_s = [0]
    viewer.log_camera = lambda *args: None
    viewer.log_pointcloud = lambda *args: None
    viewer.log_hand_geometry = lambda *args: None
    monkeypatch.setattr(
        module, "log_points", lambda name, points, **kwargs: clouds.append(points)
    )
    # 无效帧无需读取物体点云，且须清除上一帧显示。
    viewer.log_step(0, np.zeros((1, 1, 3), np.uint8))
    assert clouds == [None] and logged["hoi/valid"] == 0

@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_segmentation_interval_validation(value):
    with pytest.raises(ValueError):
        TrackingConfig(segmentation_interval_frames=value)


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_depth_gate_counts_both_directions(offset):
    from cordvip_geometry.tracking import pose_depth_quality

    rendered = np.ones((10, 10))
    observed = rendered.copy()
    observed[:6] += offset
    metrics, accepted = pose_depth_quality(observed, rendered, TrackingConfig())
    assert not accepted and metrics["depth_pixels"] == 40
    assert metrics["depth_fraction"] == 0.4
    assert metrics["occluded_fraction"] == -1


@pytest.mark.parametrize(
    "interval,expected", [(1, [1, 2, 3, 4]), (2, [2, 4]), (3, [3])]
)
def test_segmentation_schedule(recovery_scene, interval, expected):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=interval)
    calls = []
    def track(rgb):
        calls.append(scene.frame_index)
        return masks.copy()

    scene.segmenter.track = track
    for index in range(1, 5):
        result = scene.track(frame)
        assert all(result.valid.values())
        for metrics in result.quality.values():
            assert all(np.isfinite(v) for v in metrics.values())
            assert metrics["mask_source"] == int(index in expected)
            assert metrics["mask_age_frames"] == index - scene.last_sam_frame
    assert calls == expected


@pytest.mark.parametrize("broken", [False, True])
def test_event_refresh_rechecks_all_and_rolls_back(recovery_scene, broken):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=3)
    # 60% 投影被更近深度覆盖；当前 mask 提供真实遮挡证据。
    frame.depth_m[:16] = 0.8
    for mask in masks.values():
        mask[:16] = False
    calls, renders = [], []
    for name in scene.renderers:
        scene.renderers[name].render = lambda rays, pose, n=name: renders.append(
            n
        ) or np.ones(len(rays))
    previous = {n: t.estimator.pose_last for n, t in scene.trackers.items()}

    def refresh(rgb):
        calls.append(scene.frame_index)
        if broken:
            raise RuntimeError("shared state broken")
        return masks.copy()

    scene.segmenter.track = refresh
    if broken:
        with pytest.raises(RuntimeError, match="shared state"):
            scene.track(frame)
        assert all(
            t.estimator.pose_last is previous[n] for n, t in scene.trackers.items()
        )
    else:
        result = scene.track(frame)
        assert all(result.valid.values())
        assert all(q["mask_source"] == 1 for q in result.quality.values())
    assert calls == [1]
    assert sorted(renders) == ["apple", "banana"]


def test_event_rejection_and_due_recovery(recovery_scene):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=10)
    scene.renderers["apple"].render = lambda rays, pose: np.full(len(rays), 2.0)
    events = []
    original_detect = scene.segmenter.detect

    def track(rgb):
        events.append(("track", scene.frame_index))
        return masks.copy()

    def detect(*args):
        events.append(("detect", scene.frame_index))
        return original_detect(*args)

    scene.segmenter.track = track
    scene.segmenter.detect = detect
    assert scene.track(frame).valid == {"apple": False, "banana": True}
    scene.renderers["apple"].render = lambda rays, pose: np.ones(len(rays))
    assert all(scene.track(frame).valid.values())
    assert events == [("track", 1), ("track", 2), ("detect", 2)]


def test_mixed_quality_hdf5_is_finite(recovery_scene, tmp_path):
    import h5py
    from dataclasses import asdict, replace
    from dexmani_real.data.hoi import write_hoi

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=2)
    frame.depth_m[18:] = 0
    results = [
        scene.initialize(frame, scene.prompts),
        scene.track(frame),
        scene.track(frame),
    ]
    names = list(masks)
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as data:
        data.create_dataset("timing/rows", data=np.arange(3))
    write_hoi(
        path,
        dict(
            points_local=np.ones((1024, 3), np.float32),
            offsets=[0, 512, 1024],
            object_names=names,
        ),
        np.array([[r.poses[n] for n in names] for r in results]),
        np.eye(4),
        quality=[[r.quality[n] for n in names] for r in results],
        tracking_config=asdict(scene.config),
        object_valid=np.ones((3, 2), bool),
    )
    with h5py.File(path, "r") as data:
        for values in data["hoi/quality"].values():
            assert np.isfinite(values[:]).all()
        np.testing.assert_array_equal(data["hoi/quality/mask_source"][:, 0], [1, 0, 1])
        assert data["hoi/object_pointcloud"].shape == (3, 1024, 3)
        np.testing.assert_array_equal(
            data["hoi/quality/depth_coverage"][:], np.full((3, 2), 0.75)
        )


def test_correction_uses_video_index(monkeypatch):
    import torch
    from contextlib import nullcontext
    from cordvip_geometry.segmentation import SAM2Segmenter

    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    segmenter = SAM2Segmenter.__new__(SAM2Segmenter)
    segmenter.state = {"num_frames": 3}
    calls = []
    segmenter.tracker = SimpleNamespace(
        add_new_mask=lambda state, **kw: calls.append(kw["frame_idx"]),
        infer_single_frame=lambda state, index: index,
    )
    segmenter._masks = lambda result: result
    assert segmenter.correct_masks({"apple": np.ones((2, 2), bool)}) == 2
    assert calls == [2] and segmenter.state["num_frames"] == 3


def test_online_offline_event_schedule_matches(recovery_scene, monkeypatch, tmp_path):
    from dataclasses import replace
    from dexmani_real.perception.hoi_worker import run_hoi_worker
    from dexmani_real.geometry import hoi
    from cordvip_geometry import tracking

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=2)
    names = list(masks)
    calls = []
    def track(rgb):
        calls.append(scene.frame_index)
        return masks.copy()

    scene.segmenter.track = track

    class Reader(FakeRawReader):
        num_frames = 4

        def read_camera(self):
            return SimpleNamespace(
                color=SimpleNamespace(
                    height=24,
                    width=24,
                    matrix=lambda: frame.K,
                    distortion_model="none",
                    distortion_coeffs=np.zeros(5),
                ),
                depth_scale_m=0.001,
                T_xarm_base_from_color=np.eye(4),
            )

        def __getitem__(self, name):
            if name == "depth":
                return np.full((4, 24, 24), 1000, np.uint16)
            return np.array(
                [(1, 1), (1, 1), (2, 2), (3, 3)],
                dtype=[("camera_sequence", "i8"), ("camera_ns", "i8")],
            )

        def iter_rgb_frames(self):
            yield from [frame.rgb] * 4

    _, _, quality, validity = script("annotate_episodes").track_episode(
        Reader(), scene, names, scene.prompts
    )
    assert calls == [2] and validity.all() and quality[0] == quality[1]
    calls.clear()
    monkeypatch.setattr(tracking, "SceneTracker", lambda *a, **kw: scene)
    monkeypatch.setattr(
        hoi, "task_objects", lambda task: [{"name": n, "prompt": n} for n in names]
    )
    monkeypatch.setattr(RGBDFrame, "from_camera", lambda *a: frame)
    monkeypatch.setattr("signal.signal", lambda *a: None)
    shared = SimpleNamespace(
        is_running=SimpleNamespace(value=True), hoi_ready=SimpleNamespace(value=False)
    )
    shared.hoi_request = SimpleNamespace(
        read=lambda: ({"episode_id": 1, "after_sequence": 0, "after_ns": 0}, 1)
    )
    frames = iter([({"timestamp_ns": n}, n) for n in (1, 1, 2, 3)])
    shared.camera_ring = SimpleNamespace(read=lambda: next(frames))
    written = []

    def write(**row):
        written.append(row)
        if len(written) == 3:
            shared.is_running.value = False

    shared.hoi_ring = SimpleNamespace(field_names=(), write=write)
    template = dict(
        object_names=names,
        points_local=np.ones((1024, 3), np.float32),
        offsets=[0, 512, 1024],
    )
    run_hoi_worker(
        shared, Reader().read_camera(), tmp_path, template, tracking_config=scene.config
    )
    assert calls == [2] and all(r["valid"] for r in written)
    assert [r["source_camera_sequence"] for r in written] == [1, 2, 3]


@pytest.mark.parametrize(
    "observed,rendered", [(0.0, 1.0), (np.nan, 1.0), (1.0, 0.0), (1.0, np.inf)]
)
def test_depth_gate_requires_support(observed, rendered):
    from cordvip_geometry.tracking import pose_depth_quality

    metrics, accepted = pose_depth_quality(
        np.full((10, 10), observed), np.full((10, 10), rendered), TrackingConfig()
    )
    assert (
        not accepted
        and metrics["depth_pixels"] == 0
        and metrics["depth_fraction"] == 0
        and metrics["depth_coverage"] == 0
    )


def test_depth_gate_boundary():
    from cordvip_geometry.tracking import pose_depth_quality

    observed = np.ones((10, 10))
    observed[:5] = 2
    metrics, accepted = pose_depth_quality(
        observed, np.ones_like(observed), TrackingConfig(min_depth_pixels=50)
    )
    assert accepted and metrics["depth_fraction"] == 0.5


def test_event_refresh_can_invalidate_other_candidate(recovery_scene):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=3)
    scene.renderers["banana"].render = lambda rays, pose: np.full(len(rays), 2.0)
    masks["apple"][:] = False
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": False}
    assert scene.last_sam_frame == 1


def test_event_resets_keyframe_interval_and_initialization(recovery_scene):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=3)
    calls = []
    def track(rgb):
        calls.append(scene.frame_index)
        return masks.copy()

    scene.segmenter.track = track
    frame.depth_m[:16] = 0.8
    for mask in masks.values():
        mask[:16] = False
    assert all(scene.track(frame).valid.values())
    frame.depth_m[:] = 1
    for mask in masks.values():
        mask[:] = True
    for _ in range(3):
        assert all(scene.track(frame).valid.values())
    assert calls == [1, 4]
    scene.initialize(frame, scene.prompts)
    assert scene.frame_index == scene.last_sam_frame == 0
    assert not scene.frame_cache


def test_viewer_marks_unmeasured_quality_as_missing():
    from scripts.visualize_episode import EpisodeVisualizer

    logged = {}
    viewer = EpisodeVisualizer.__new__(EpisodeVisualizer)
    viewer.rr = SimpleNamespace(
        Scalar=float, log=lambda name, value: logged.update({name: value})
    )

    class Reader(dict):
        data_file = {
            "hoi/quality": {
                "iou": np.array([[-1.0, 0.75]]),
                "mask_source": np.array([[0.0, 1.0]]),
                "depth_fraction": np.array([[0.8, 0.9]]),
                "depth_coverage": np.array([[0.0, 0.7]]),
            }
        }

    viewer.reader = Reader(
        {"timing/rows": [{"arm_ns": 1, "hand_ns": 1, "camera_ns": 1}]}
    )
    viewer.log_hoi_diagnostics(
        0, {"arm_qpos": np.zeros((1, 7)), "hand_qpos": np.zeros((1, 12))}
    )
    assert np.isnan(logged["hoi/object_0/iou"])
    assert logged["hoi/object_1/iou"] == 0.75
    assert logged["hoi/object_0/mask_source"] == 0
    assert logged["hoi/object_0/depth_fraction"] == 0.8
    assert logged["hoi/object_0/depth_coverage"] == 0.0
    assert logged["hoi/object_1/depth_coverage"] == 0.7


@pytest.mark.parametrize("truncated_reference", [False, True])
@pytest.mark.parametrize("low_coverage", [False, True])
def test_benchmark_reference_and_cleanup(
    recovery_scene, monkeypatch, tmp_path, truncated_reference, low_coverage
):
    import json
    import torch
    from cordvip_geometry import tracking

    module = script("benchmark_tracking")
    scene, frame, masks, _, _ = recovery_scene
    original_batch = tracking.track_objects
    original_depth_quality = tracking.pose_depth_quality
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "test GPU")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "test")
    monkeypatch.setattr(module.tracking, "SceneTracker", lambda *a, **kw: scene)
    monkeypatch.setattr(
        module, "task_objects", lambda task: [{"name": n, "prompt": n} for n in masks]
    )
    monkeypatch.setattr(module, "RawReader", FakeRawReader)
    calls = 0

    def replay(reader, tracker, names, prompts):
        nonlocal calls
        calls += 1
        tracker.initialize(frame, prompts)
        if low_coverage:
            frame.depth_m[10:] = 0
        # 三次预热后是逐帧基线；缺失参考必须报错，不能静默略过复核。
        if not (truncated_reference and calls == 4):
            tracker.track(frame)
            tracker.track(frame)

    monkeypatch.setattr(module, "track_episode", replay)
    output = tmp_path / "results"
    args = [
        "--episode",
        str(tmp_path / "episode_test"),
        "--repeats",
        "1",
        "--output",
        str(output),
    ]
    if truncated_reference:
        with pytest.raises(IndexError):
            module.main(args)
    else:
        module.main(args)
        runs = json.loads((output / "summary.json").read_text())
        assert [r["frames"] for r in runs] == [3, 3, 3]
        assert [r["counts"]["sam_track"] for r in runs] == (
            [2, 2, 2] if low_coverage else [2, 1, 0]
        )
        expected_coverage_events = [0, 2, 2] if low_coverage else [0, 0, 0]
        for key in ("coverage_rejected_frames", "coverage_triggered_sam_frames"):
            assert [r[key] for r in runs] == expected_coverage_events
        assert all(r["reference_rejections"] == 0 for r in runs)
        assert (output / "comparison.json").is_file()
    assert tracking.track_objects is original_batch
    assert tracking.pose_depth_quality is original_depth_quality


@pytest.mark.parametrize(
    "value", [True, False, 0, -0.1, 1.1, np.nan, np.inf, "bad", "0.5"]
)
def test_depth_coverage_config(value):
    with pytest.raises(ValueError, match="min_depth_coverage"):
        TrackingConfig(min_depth_coverage=value)


@pytest.mark.parametrize("missing", [np.nan, 0.0])
def test_sparse_depth_requires_coverage(missing):
    from cordvip_geometry.tracking import pose_depth_quality, pose_quality

    rendered = np.ones((100, 100))
    observed = np.full_like(rendered, missing)
    observed.flat[:30] = 1
    config = TrackingConfig()
    depth, accepted = pose_depth_quality(observed, rendered, config)
    full, mask_accepted = pose_quality(
        np.ones_like(rendered, bool), observed, rendered, config
    )
    assert not accepted and mask_accepted
    assert depth["depth_pixels"] == 30 and depth["depth_fraction"] == 1
    assert depth["depth_coverage"] == full["depth_coverage"] == 0.003


@pytest.mark.parametrize("support", [49, 50, 51, 100])
@pytest.mark.parametrize("fraction", [0.49, 0.5, 0.51, 1.0])
def test_depth_coverage_and_match_boundaries(support, fraction):
    from cordvip_geometry.tracking import pose_depth_quality

    rendered = np.ones((100, 100))
    observed = np.zeros_like(rendered)
    observed[:support] = 2
    matched = int(support * 100 * fraction)
    observed.flat[:matched] = 1
    metrics, accepted = pose_depth_quality(observed, rendered, TrackingConfig())
    assert metrics["depth_coverage"] == support / 100
    assert metrics["depth_fraction"] == matched / (support * 100)
    assert accepted == (support >= 50 and fraction >= 0.5)


def test_depth_coverage_does_not_replace_minimum_count():
    from cordvip_geometry.tracking import pose_depth_quality

    metrics, accepted = pose_depth_quality(
        np.ones((5, 5)), np.ones((5, 5)), TrackingConfig()
    )
    assert not accepted and metrics["depth_pixels"] == 25
    assert metrics["depth_coverage"] == metrics["depth_fraction"] == 1


@pytest.mark.parametrize("interval,invalid_mask", [(3, False), (3, True), (1, False)])
def test_low_coverage_refresh(recovery_scene, monkeypatch, interval, invalid_mask):
    from dataclasses import replace
    from cordvip_geometry import tracking

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=interval)
    frame.depth_m[10:] = 0
    masks["banana"][10:] = False
    if invalid_mask:
        masks["apple"][:] = False
        masks["apple"][:6] = True  # 深度点数足够，但轮廓 precision 不足。
    renders, calls, refinements = [], [], []
    for name in masks:

        def render(rays, pose, name=name):
            renders.append(name)
            depth = np.ones((12, 12))
            if name == "banana":
                depth[5:] = 0
            return depth.ravel()

        scene.renderers[name].render = render
    original = tracking.track_objects

    def batch(*args):
        refinements.append(1)
        return original(*args)

    monkeypatch.setattr(tracking, "track_objects", batch)
    scene.segmenter.track = lambda rgb: calls.append(scene.frame_index) or masks.copy()
    result = scene.track(frame)
    assert calls == [1] and refinements == [1]
    assert result.valid == {"apple": not invalid_mask, "banana": True}
    assert scene.states["apple"] == ("LOST" if invalid_mask else "TRACKING")
    assert sorted(renders) == ["apple", "banana"]
    assert result.quality["banana"]["depth_coverage"] == 1
    if not invalid_mask:
        assert result.quality["apple"]["depth_coverage"] == 5 / 12
        assert result.quality["apple"]["mask_source"] == 1


def test_benchmark_coverage_counts_shared_frames():
    module = script("benchmark_tracking")
    check = dict(low_coverage=True, coverage_only_rejection=True)
    rows = [
        dict(
            initial=False,
            total_ms=1.0,
            valid={"apple": True, "banana": True},
            stages={"sam_track": [1.0]},
            reference_rejected=[],
            depth_checks=[check, check],
        ),
        dict(
            initial=False,
            total_ms=1.0,
            valid={"apple": False, "banana": True},
            stages={"sam_track": [1.0]},
            reference_rejected=[],
            depth_checks=[dict(low_coverage=True, coverage_only_rejection=False)],
        ),
    ]
    stats = module.tracking_statistics(rows)
    assert stats["coverage_rejected_frames"] == 1
    assert stats["coverage_triggered_sam_frames"] == 2
    assert stats["sam_calls"] == 2


def test_coverage_is_independent_of_mask_and_occlusion():
    from cordvip_geometry.tracking import pose_depth_quality, pose_quality

    rendered = np.ones((10, 10))
    rendered[:2] = 0
    observed = np.ones_like(rendered)
    observed[2:4] = 0
    observed[4:6] = 0.8
    mask = observed == 1
    full, _ = pose_quality(mask, observed, rendered, TrackingConfig())
    depth, _ = pose_depth_quality(observed, rendered, TrackingConfig())
    assert full["depth_coverage"] == depth["depth_coverage"] == 0.75
    assert full["occluded_fraction"] == 0.25
