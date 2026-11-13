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
from cordvip_geometry.mesh import MeshRaycaster, sample_mesh_surface, scale_mesh
from cordvip_geometry.tracking import TrackingFailure


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
            return {"apple": np.eye(4)}

        def track(self, frame):
            self.count += 1
            pose = np.eye(4)
            pose[0, 3] = 1
            return {"apple": pose}

    tracker = Tracker()
    poses, _ = module.track_episode(
        FakeRawReader(), tracker, ["apple"], {"apple": "apple"}
    )
    assert tracker.count == 2
    np.testing.assert_array_equal(poses[0], poses[1])
    assert poses[2, 0, 0, 3] == 1

    def fail(frame):
        raise TrackingFailure("lost")

    tracker.track = fail
    with pytest.raises(TrackingFailure, match="行 2"):
        module.track_episode(FakeRawReader(), tracker, ["apple"], {"apple": "apple"})


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
def test_existing_or_failed_episode_never_writes(monkeypatch, tmp_path, has_hoi):
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
        raise TrackingFailure("lost")

    monkeypatch.setattr(module, "track_episode", fail)

    def write(*args):
        pytest.fail("existing or failed episode wrote HOI")

    monkeypatch.setattr(module, "write_hoi", write)
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
    write_hoi(path, template, poses, np.eye(4))
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
    scene.trackers = {"apple": object(), "banana": object()}
    scene.renderers = {
        "apple": SimpleNamespace(render=lambda rays, pose: np.ones(len(rays))),
        "banana": SimpleNamespace(render=lambda rays, pose: np.full(len(rays), 2.0)),
    }
    calls = []

    def batch(trackers, rgb, depth, K, masks):
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
    with pytest.raises(
        TrackingFailure, match="banana: Semantic depth/model disagreement"
    ):
        scene.track(frame)
    assert calls == [["apple", "banana"]]
    mask[:] = False
    with pytest.raises(TrackingFailure, match="apple: insufficient segmented depth"):
        scene.track(frame)
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
        return object()

    def track(reader, tracker, names, prompts):
        assert list(names) == list(prompts) == ["banana", "apple"]
        return np.zeros((3, 2, 4, 4)), np.eye(4)

    monkeypatch.setattr(module, "SceneTracker", tracker)
    monkeypatch.setattr(module, "track_episode", track)
    monkeypatch.setattr(module, "write_hoi", lambda *args: None)
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


def test_scene_initialization_rectifies_before_detection_and_resets_poses():
    import cv2
    from cordvip_geometry.tracking import SceneTracker

    shape = (32, 32)
    K = np.array([[20.0, 0, 16], [0, 20.0, 16], [0, 0, 1]])
    rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(*shape, 3)
    depth = np.ones(shape, np.float32)
    boxes = {"fruit": np.array([0, 0, 16, 32]), "dish": np.array([16, 0, 32, 32])}
    masks = {"fruit": np.indices(shape)[1] < 16, "dish": np.indices(shape)[1] >= 16}
    scene = SceneTracker.__new__(SceneTracker)
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
        frame = RGBDFrame(rgb, depth, K, "brown_conrady", distortion)
        result = scene.initialize(frame, {"fruit": " Apple. ", "dish": "bowl"})
        assert list(result) == ["fruit", "dish"]
        assert received[-2] is masks["fruit"] and received[-1] is masks["dish"]
    assert scene.rectifier.maps is None


@pytest.mark.parametrize("count", [0, 2])
def test_scene_rejects_missing_or_ambiguous_detection(count):
    from cordvip_geometry.tracking import SceneTracker

    scene = SceneTracker.__new__(SceneTracker)
    scene.trackers = {
        "fruit": SimpleNamespace(estimator=SimpleNamespace(pose_last=None))
    }
    scene.segmenter = SimpleNamespace(detect=lambda *args: [{"label": "apple"}] * count)
    frame = RGBDFrame(
        np.zeros((8, 8, 3), np.uint8), np.ones((8, 8)), np.eye(3), "none", (0,) * 5
    )
    with pytest.raises(
        TrackingFailure, match=f"fruit: expected one first-frame detection, got {count}"
    ):
        scene.initialize(frame, {"fruit": "apple"})
