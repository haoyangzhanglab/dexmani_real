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
from cordvip_geometry.tracking import SceneResult, TrackingConfig, TrackingFailure

from dexmani_real.geometry.hoi import sample_mesh_surface


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


def front_pose():
    pose = np.eye(4)
    pose[2, 3] = 1.0
    return pose


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


@pytest.mark.parametrize("missing,reason", [
    ("T_xarm_base_from_color", "相机外参"), ("depth_scale_m", "深度单位"),
])
def test_offline_requires_rgbd_geometry(monkeypatch, missing, reason):
    reader = FakeRawReader()
    camera = reader.read_camera()
    setattr(camera, missing, None)
    monkeypatch.setattr(reader, "read_camera", lambda: camera)

    with pytest.raises(ValueError, match=reason):
        script("annotate_episodes").track_episode(
            reader, None, ["apple"], {"apple": "apple"}
        )


@pytest.mark.parametrize("has_hoi", [False, True])
def test_existing_or_invalid_episode_never_writes(monkeypatch, tmp_path, has_hoi):
    module = script("annotate_episodes")
    reader = FakeRawReader()
    if has_hoi:
        (tmp_path / "episode").mkdir()
        (tmp_path / "episode/hoi.h5").touch()

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
    path = tmp_path / "hoi.h5"
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
        assert set(data["template"]) == {"points_local", "offsets", "object_names"}
        np.testing.assert_array_equal(
            data["template/object_names"].asstr()[:], template["object_names"],
        )
        np.testing.assert_array_equal(data["template/offsets"][:], template["offsets"])
        points = data["object_pointcloud"][:]
        np.testing.assert_allclose(points[0], template["points_local"])
        np.testing.assert_allclose(
            points[1], template["points_local"] + [1, 0, 0], atol=1e-7
        )
        np.testing.assert_allclose(
            points[128], template["points_local"] + [0, 2, 0], atol=2e-7
        )


@pytest.fixture
def annotated_episode(task_template):
    from dexmani_real.data.hoi import write_hoi
    from dexmani_real.data.raw import RawWriter, camera_metadata
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.geometry.hoi import load_task_template
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE

    episode = task_template / "episode"
    episode.mkdir()
    camera = CameraSnapshot(
        None, CameraIntrinsics(32, 32, 500, 500, 16, 16, "none", (0.0,) * 5),
        0.001, np.eye(4),
    )
    writer = RawWriter(episode, control_hz=20)
    writer.open(dict(collection_source="policy", task_label="test", **camera_metadata(camera)))
    try:
        for index in range(3):
            stamp = 1_000_000_000 + index * 50_000_000
            arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
            arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
            hand["tactile_aggregate_timestamp_ns"] = hand["tactile_dense_timestamp_ns"] = stamp
            row = SimpleNamespace(
                arm=arm, hand=hand, control_timestamp_ns=stamp, observation_timestamp_ns=stamp,
                camera=dict(timestamp_ns=stamp, ring_sequence=index + 1,
                            rgb=np.full((32, 32, 3), 90, np.uint8),
                            depth=np.full((32, 32), 600, np.uint16)),
            )
            writer.append_frame(row, SimpleNamespace(arm_qpos=np.zeros(7), hand_qpos=np.zeros(12)))
        writer.finish("user_stop")
    finally:
        writer.close()
    template = load_task_template(task_template)
    kwargs = dict(
        quality=[[{"iou": 1.0}, {"iou": None}], [{}, {}], [{"iou": 0.5}, {}]],
        object_valid=np.ones((3, 2), bool), tracking_config={},
    )
    poses = np.tile(np.eye(4), (3, 2, 1, 1))
    write_hoi(episode / "hoi.h5", template, poses, np.eye(4), **kwargs)
    return episode, template, poses, kwargs


@pytest.mark.parametrize("failure", ["compute", "write", "close", "replace"])
def test_hoi_replacement_failure_keeps_published_files(annotated_episode, monkeypatch, failure):
    import h5py
    from dexmani_real.data import hoi

    episode, template, poses, kwargs = annotated_episode
    before = {name: (episode / name).read_bytes() for name in ("data.h5", "rgb.mp4", "hoi.h5")}

    def fail(*args, **kwargs):
        raise OSError(f"failed {failure}")

    if failure == "compute":
        monkeypatch.setattr(hoi, "transform_task_points", fail)
    elif failure == "write":
        monkeypatch.setattr(h5py.Group, "create_dataset", fail)
    elif failure == "close":
        original_close = h5py.File.close

        def close(file):
            temporary = file.id.valid and ".hoi.h5.tmp-" in file.filename
            original_close(file)
            if temporary:
                fail()

        monkeypatch.setattr(h5py.File, "close", close)
    else:
        monkeypatch.setattr(hoi.os, "replace", fail)
    with pytest.raises(OSError, match=f"failed {failure}"):
        hoi.write_hoi(episode / "hoi.h5", template, poses, np.eye(4), overwrite=True, **kwargs)
    assert not list(episode.glob(".hoi.h5.tmp-*"))
    assert {name: (episode / name).read_bytes() for name in before} == before


def test_hoi_explicit_recompute_preserves_raw_and_reader_snapshot(annotated_episode, monkeypatch):
    from dexmani_real.data.hoi import write_hoi
    from dexmani_real.data.raw import RawReader

    episode, template, poses, kwargs = annotated_episode
    raw_before = {name: (episode / name).read_bytes() for name in ("data.h5", "rgb.mp4")}
    with pytest.raises(FileExistsError):
        write_hoi(episode / "hoi.h5", template, poses, np.eye(4), **kwargs)
    assert not list(episode.glob(".hoi.h5.tmp-*"))
    module = script("annotate_episodes")
    pose = np.eye(4)
    pose[0, 3] = 0.25
    result = SceneResult(
        {name: pose for name in template["object_names"]},
        {name: {} for name in template["object_names"]},
        {name: True for name in template["object_names"]},
    )
    tracker = SimpleNamespace(config=TrackingConfig(), initialize=lambda *a: result, track=lambda *a: result)
    monkeypatch.setattr(module, "SceneTracker", lambda *a, **kw: tracker)
    with RawReader(episode) as old_reader:
        old_cloud = old_reader["object_pointcloud"][:]
        quality = old_reader.hoi_file["quality/iou"][:]
        assert np.isnan(quality[0, 1]) and np.isnan(quality[1]).all()
        module.annotate_episodes([episode], episode.parent, overwrite=True)
        np.testing.assert_array_equal(old_reader["object_pointcloud"][:], old_cloud)
        with RawReader(episode) as new_reader:
            np.testing.assert_allclose(new_reader["object_pointcloud"][:], old_cloud + [0.25, 0, 0], atol=2e-7)
            assert new_reader.hoi_file["object_valid"][:].all()
    assert {name: (episode / name).read_bytes() for name in raw_before} == raw_before


@pytest.mark.parametrize("kind", ["scalar", "group"])
def test_hoi_validity_reports_malformed_object_names(annotated_episode, kind):
    import h5py
    from dexmani_real.data.hoi import read_hoi_validity
    from dexmani_real.data.raw import RawDataError, RawReader

    episode, *_ = annotated_episode
    with h5py.File(episode / "hoi.h5", "r+") as data:
        del data["template/object_names"]
        if kind == "scalar":
            data["template/object_names"] = "apple"
        else:
            group = data.create_group("template/object_names")
            group.create_group("apple")
            group.create_group("banana")
    with RawReader(episode) as reader, pytest.raises(RawDataError, match="template/object_names"):
        read_hoi_validity(reader)


def test_viewer_rejects_missing_depth_units_before_starting_display(annotated_episode, monkeypatch):
    import h5py
    import rerun as rr
    from scripts.visualize_episode import EpisodeVisualizer

    episode, *_ = annotated_episode
    with h5py.File(episode / "data.h5", "r+") as data:
        data["meta"].attrs["camera_payload_mode"] = "rgb"
    monkeypatch.setattr(rr, "init", lambda *a, **kw: pytest.fail("started display without depth units"))
    with pytest.raises(ValueError, match="深度单位"):
        EpisodeVisualizer(str(episode), table_plane_path=None)


def test_geometry_digest_uses_ordered_numeric_values():
    from dexmani_real.geometry.hoi import geometry_digest

    points = np.arange(24, dtype=np.float32).reshape(8, 3)
    offsets = np.array([0, 4, 8], np.int64)
    expected = geometry_digest(points, offsets)
    assert expected == geometry_digest(points.astype(">f4"), offsets.astype(">i8"))
    assert expected == geometry_digest(np.asfortranarray(points), offsets)
    assert expected != geometry_digest(points[::-1], offsets)
    assert expected != geometry_digest(points + 1, offsets)
    assert expected != geometry_digest(points, [0, 3, 8])


@pytest.mark.parametrize("prompts", [[None, "apple"], ["...", "apple"], [" APPLE. ", "apple"]])
def test_prompt_checks_belong_to_tracker(task_template, prompts):
    import json
    from cordvip_geometry.tracking import SceneTracker
    from dexmani_real.geometry.hoi import load_task_template

    names = ["banana", "apple"]
    (task_template / "task.json").write_text(json.dumps({
        "objects": [{"name": name, "prompt": prompt} for name, prompt in zip(names, prompts)],
    }))
    assert list(load_task_template(task_template)["object_names"]) == names
    tracker = SceneTracker.__new__(SceneTracker)
    tracker.trackers = dict.fromkeys(names)
    with pytest.raises(ValueError, match="Prompts"):
        tracker.initialize(None, dict(zip(names, prompts)))


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


@pytest.fixture
def task_template(tmp_path):
    import json

    objects = [{"name": name} for name in ("banana", "apple")]
    (tmp_path / "task.json").write_text(json.dumps({"objects": objects}))
    for obj in objects:
        directory = tmp_path / obj["name"] / "mesh"
        directory.mkdir(parents=True)
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(directory / "mesh_m.obj")
    script("sample_template").sample_template(tmp_path)
    return tmp_path


def test_template_reuse_and_explicit_overwrite(task_template):
    import json
    from dexmani_real.geometry.hoi import load_task_template

    task = task_template
    template = load_task_template(task)
    np.testing.assert_array_equal(template["object_names"], ["banana", "apple"])
    assert set(template) == {"points_local", "offsets", "object_names"}
    # task.json 重排不改变已有模板的对象顺序和点序。
    (task / "task.json").write_text(json.dumps({"objects": [{"name": "apple"}, {"name": "banana"}]}))
    np.testing.assert_array_equal(load_task_template(task)["object_names"], ["banana", "apple"])
    mesh_path = task / "apple/mesh/mesh_m.obj"
    trimesh.creation.box(extents=(0.2, 0.1, 0.1)).export(mesh_path)
    original = (task / "hoi/template.npz").read_bytes()
    script("sample_template").sample_template(task)
    assert (task / "hoi/template.npz").read_bytes() == original
    script("sample_template").sample_template(task, overwrite=True)
    updated = load_task_template(task)
    np.testing.assert_array_equal(updated["object_names"], ["apple", "banana"])
    apple = list(updated["object_names"]).index("apple")
    segment = slice(updated["offsets"][apple], updated["offsets"][apple + 1])
    np.testing.assert_allclose(np.max(np.abs(updated["points_local"][segment, 0])), 0.1)


@pytest.mark.parametrize("single", [False, True])
def test_external_directory_workflow(task_template, monkeypatch, single):
    from dexmani_real.deployment.preflight import resolve_hoi_inputs
    from dexmani_real.geometry.hoi import load_task_template

    checkout = task_template / "checkout"
    (checkout / "episodes/pick/episode_unselected").mkdir(parents=True)
    source = task_template / "external/pick"
    episodes = [source / name for name in ("episode_001", "episode_002")]
    for episode in episodes:
        episode.mkdir(parents=True)
        (episode / "data.h5").touch()
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("HOME", str(task_template))
    assert script("prepare_task").main(["--task-dir", "~"]) == 0
    original = (task_template / "hoi/template.npz").read_bytes()
    assert script("sample_template").main(["--task-dir", ".."]) == 0
    assert (task_template / "hoi/template.npz").read_bytes() == original

    module = script("annotate_episodes")
    reads, meshes = [], []
    monkeypatch.setattr(module, "ROOT", checkout)

    def reader(path):
        reads.append(path)
        return FakeRawReader()

    def tracker(paths, *, workspace, config):
        assert workspace == checkout
        meshes.append(paths)
        return SimpleNamespace(config=config)

    monkeypatch.setattr(module, "RawReader", reader)
    monkeypatch.setattr(module, "SceneTracker", tracker)
    monkeypatch.setattr(module, "track_episode", lambda *args: (
        np.tile(np.eye(4), (3, 2, 1, 1)), np.eye(4), [[{}, {}]] * 3, np.ones((3, 2), bool),
    ))
    input_root = "../external/pick/episode_001" if single else "~/external/pick"
    assert module.main([input_root, "--task-dir", ".."]) == 0
    assert reads == (episodes[:1] if single else episodes)
    assert meshes == [{name: task_template / name / "mesh/mesh_m.obj" for name in ("banana", "apple")}]
    # 已有 HOI 的默认跳过保持惰性，不加载错误资产或模型。
    assert module.main([input_root, "--task-dir", "../missing"]) == 0
    assert len(reads) == (1 if single else 2)
    resolved, template = resolve_hoi_inputs(
        SimpleNamespace(observation_fields=("object_pointcloud",), task_name="unrelated/experiment label"),
        "..",
    )
    assert resolved == task_template
    np.testing.assert_array_equal(template["points_local"], load_task_template(task_template)["points_local"])


@pytest.mark.parametrize("name", ["prepare_task", "sample_template", "annotate_episodes"])
def test_task_clis_remove_replaced_options(name):
    argv = ["input"] if name == "annotate_episodes" else []
    with pytest.raises(SystemExit) as error:
        script(name).main([*argv, "--task", "pick"])
    assert error.value.code == 2


def test_annotation_cli_removes_episode_option():
    with pytest.raises(SystemExit) as error:
        script("annotate_episodes").main(["input", "--task-dir", "task", "--episode", "episode"])
    assert error.value.code == 2


def test_template_overwrite_failure_keeps_existing_asset(task_template, monkeypatch):
    module = script("sample_template")
    path = task_template / "hoi/template.npz"
    before = path.read_bytes()

    def fail(stream, **values):
        stream.write(b"incomplete template")
        raise OSError("simulated disk write failure")

    monkeypatch.setattr(module.np, "savez", fail)
    with pytest.raises(OSError, match="disk write"):
        module.sample_template(task_template, overwrite=True)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".template.npz.*"))


def test_mesh_surface_checks_preserve_partial_degenerate_geometry(tmp_path, caplog):
    from cordvip_geometry.mesh import load_mesh

    box = trimesh.creation.box()
    mesh = trimesh.Trimesh(vertices=box.vertices, faces=np.vstack((box.faces, [0, 0, 1])), process=False)
    path = tmp_path / "partial.obj"
    mesh.export(path)
    loaded = load_mesh(path)
    np.testing.assert_array_equal(loaded.vertices, mesh.vertices)
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
    assert "1/13 degenerate faces" in caplog.text
    trimesh.Trimesh(vertices=box.vertices, faces=[[0, 0, 1]], process=False).export(path)
    with pytest.raises(ValueError, match="surface area"):
        load_mesh(path)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_scale_rejects_unrepresentable_surface(scale):
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError, match="surface area"):
            scale_mesh(trimesh.creation.box(), scale_to_meters=scale)


def test_prepare_reused_mesh_validates_measurement_without_models(tmp_path):
    import json

    directory = tmp_path / "apple/mesh"
    directory.mkdir(parents=True)
    trimesh.creation.box(extents=(0.1, 0.2, 0.3)).export(directory / "mesh_m.obj")
    module = script("prepare_task")
    for measurement in ({}, {"size_axis": "y", "size_meters": 0.2}):
        (tmp_path / "task.json").write_text(json.dumps({"objects": [{"name": "apple", **measurement}]}))
        module.prepare_task(tmp_path)
    (tmp_path / "task.json").write_text(json.dumps({
        "objects": [{"name": "apple", "size_axis": "y", "size_meters": 0.3}],
    }))
    with pytest.raises(ValueError, match="尺寸与实测"):
        module.prepare_task(tmp_path)


def test_scene_batch_does_not_gate_on_mask_depth(recovery_scene, monkeypatch):
    from dataclasses import replace
    from cordvip_geometry import tracking

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, recovery_enabled=False)
    scene.renderers["banana"].render = lambda rays, pose: np.full(len(rays), 2.0)
    calls = []
    original = tracking.track_objects

    def batch(trackers, *args):
        calls.append(list(trackers))
        return original(trackers, *args)

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert result.valid == {"apple": True, "banana": False}
    assert list(result.poses) == ["apple"]
    masks["apple"][:] = False
    assert not any(scene.track(frame).valid.values())
    assert calls == [["apple", "banana"], ["apple"]]


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


def test_registration_rejects_uninitialized_backend():
    from cordvip_geometry.foundationpose import FoundationPoseTracker

    tracker = FoundationPoseTracker.__new__(FoundationPoseTracker)
    tracker.estimator = SimpleNamespace(
        pose_last=np.eye(4), register=lambda **kwargs: np.eye(4)
    )
    with pytest.raises(TrackingFailure, match="did not initialize tracking"):
        tracker.register(None, None, None, None)
    assert tracker.estimator.pose_last is None


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
        name: SimpleNamespace(
            estimator=SimpleNamespace(pose_last=front_pose()),
            object_from_center=np.eye(4),
        )
        for name in boxes
    }
    scene.renderers = {
        name: SimpleNamespace(
            render=lambda rays, pose, name=name: masks[name][::2, ::2]
            .ravel()
            .astype(float)
        )
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
        return front_pose()

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
            tracker.estimator.pose_last = front_pose()
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
    assert pose_quality(target, observed, rendered, config)[1] == "VALID"
    # mask 内完全匹配仍不能接受轮廓过大的模型。
    assert pose_quality(target, observed, np.ones_like(rendered), config)[1] == "LOST"
    visible = target.copy()
    visible[:, 20:] = False
    occluded_depth = observed.copy()
    occluded_depth[target & ~visible] = 0.8
    metrics, verdict = pose_quality(visible, occluded_depth, rendered, config)
    assert verdict == "VALID" and metrics["occluded_fraction"] == 0.5
    occluded_depth[target & ~visible] = 0
    # 缺失深度不能被计为遮挡。
    assert (
        pose_quality(visible, occluded_depth, rendered, config)[0]["occluded_fraction"]
        == 0
    )
    assert pose_quality(target, observed, rendered * 1.1, config)[1] == "LOST"


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
    from scipy.spatial import cKDTree

    from dexmani_real.geometry.hoi import sample_surface_points

    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    points = sample_surface_points(mesh, 256)
    np.testing.assert_array_equal(points, sample_surface_points(mesh, 256))
    np.testing.assert_allclose(np.max(np.abs(points), axis=1), 0.05)
    reference = sample_mesh_surface(mesh, 20000, seed=19)
    old = cKDTree(sample_mesh_surface(mesh, 256)).query(reference)[0]
    new = cKDTree(points).query(reference)[0]
    assert np.percentile(new, 95) < 0.8 * np.percentile(old, 95)


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
        estimator = SimpleNamespace(pose_last=None, diameter=0.1)

        def register(rgb, depth, K, mask, name=name, estimator=estimator):
            registrations.append(name)
            estimator.pose_last = front_pose()
            return front_pose()

        scene.trackers[name] = SimpleNamespace(
            estimator=estimator,
            register=register,
            object_from_center=np.eye(4),
            symmetry_tfs=np.eye(4)[None],
        )
    scene.renderers = {
        n: SimpleNamespace(render=lambda rays, pose: np.full(len(rays), pose[2, 2]))
        for n in masks
    }
    detections = []

    def detect(*args):
        detections.append(1)
        return [{"label": n, "box_xyxy": np.array([0, 0, 24, 24])} for n in masks]

    def correct_masks(new):
        masks.update(new)
        return masks.copy()

    scene.segmenter = SimpleNamespace(
        detect=detect,
        initialize=lambda *args: masks.copy(),
        track=lambda rgb: masks.copy(),
        segment_boxes=lambda rgb, boxes: {n: np.ones(shape, bool) for n in boxes},
        correct_masks=correct_masks,
    )

    def batch(trackers, *args):
        result = {}
        for n, t in trackers.items():
            t.estimator.pose_last = front_pose()
            result[n] = front_pose()
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
    pending = scene.track(frame)
    assert pending.valid == {"apple": False, "banana": True}
    assert set(scene.pending_pose) == {"apple"}
    recovered = scene.track(frame)
    assert all(recovered.valid.values())
    assert not scene.pending_pose
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
        poses = {n: front_pose() for n in trackers}
        poses["apple"][2, 2] = 2
        for n in trackers:
            trackers[n].estimator.pose_last = poses[n].copy()
        return poses

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    np.testing.assert_array_equal(
        scene.trackers["apple"].estimator.pose_last, front_pose()
    )


def test_invalid_hoi_storage_requires_explicit_validity(tmp_path):
    import h5py

    from dexmani_real.data.hoi import read_hoi_validity, write_hoi

    path = tmp_path / "hoi.h5"
    template = dict(
        points_local=np.ones((1024, 3), np.float32),
        offsets=[0, 512, 1024],
        object_names=["apple", "pear"],
    )
    poses = np.tile(np.eye(4), (3, 2, 1, 1))
    write_hoi(
        path,
        template,
        poses,
        np.eye(4),
        quality=[[{"iou": 1}, {}], [{}, {}], [{"iou": 1}, {}]],
        tracking_config={},
        object_valid=np.array([[True, True], [True, False], [True, True]]),
    )
    with h5py.File(path, "r+") as data:
        reader = SimpleNamespace(hoi_file=data, num_frames=3)
        np.testing.assert_array_equal(read_hoi_validity(reader), [True, False, True])
        assert np.isnan(data["object_pose_camera"][1, 1]).all()
        assert np.isfinite(data["object_pose_camera"][1, 0]).all()
        np.testing.assert_array_equal(read_hoi_validity(reader, stop=2), [True, False])
        assert not data["object_pointcloud"][1].any()
        assert data["object_pointcloud"][2].all()
        assert "valid" not in data
        del data["object_valid"]
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

    from dexmani_real.geometry import hoi
    from dexmani_real.perception.hoi_worker import run_hoi_worker

    monkeypatch.setattr(
        hoi, "task_objects", lambda task: [{"name": "apple", "prompt": "apple"}]
    )
    monkeypatch.setattr(RGBDFrame, "from_camera", lambda *args, **kwargs: object())
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
        return {"banana": front_pose()}

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    np.testing.assert_array_equal(
        scene.trackers["apple"].estimator.pose_last, front_pose()
    )


def test_single_object_failure_is_not_retried(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, _, _, detections = recovery_scene
    scene.states["banana"] = "LOST"
    scene.next_retry["banana"] = 100
    previous = scene.trackers["apple"].estimator.pose_last
    calls = []

    def fail(trackers, *args):
        calls.append(list(trackers))
        trackers["apple"].estimator.pose_last = np.full((4, 4), 99.0)
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(tracking, "track_objects", fail)
    result = scene.track(frame)
    assert calls == [["apple"]]
    assert not any(result.valid.values())
    assert scene.trackers["apple"].estimator.pose_last is previous
    assert "refinement failed" in result.failures["apple"]
    assert len(detections) == 1


def test_all_lost_waits_without_inference(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, masks, _, detections = recovery_scene
    original_batch = tracking.track_objects
    original_detect = scene.segmenter.detect
    sam_frames, pose_frames = [], []

    def sam(rgb):
        sam_frames.append(scene.frame_index)
        return masks.copy()

    def batch(*args):
        pose_frames.append(scene.frame_index)
        return original_batch(*args)

    monkeypatch.setattr(scene.segmenter, "track", sam)
    monkeypatch.setattr(tracking, "track_objects", batch)
    for mask in masks.values():
        mask[:] = False
    assert not any(scene.track(frame).valid.values())  # 1: 明确丢失。
    monkeypatch.setattr(scene.segmenter, "detect", lambda *args: [])
    for _ in range(3):
        assert not any(scene.track(frame).valid.values())  # 2: 检测失败；3、4: 等待。
    assert sam_frames == [1, 2] and pose_frames == [1]

    monkeypatch.setattr(scene.segmenter, "detect", original_detect)
    assert not any(scene.track(frame).valid.values())  # 5: 注册为 Pending。
    assert set(scene.pending_pose) == set(masks)
    assert all(scene.track(frame).valid.values())  # 6: 确认恢复。
    assert sam_frames == [1, 2, 5, 6] and pose_frames == [1, 6]
    assert len(detections) == 2


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
    "name,value",
    [
        ("recovery_enabled", 1),
        ("min_depth_pixels", 0),
        ("min_depth_pixels", 1.5),
        ("segmentation_interval_frames", 0),
        ("segmentation_interval_frames", True),
        ("segmentation_interval_frames", 1.5),
        ("recovery_retry_interval_frames", 0),
        ("recovery_retry_interval_frames", True),
        ("depth_tolerance_m", -1),
        ("depth_tolerance_m", np.nan),
        ("occlusion_timeout_s", 0),
        ("occlusion_timeout_s", np.inf),
        ("max_translation_speed_m_s", -1),
        ("max_translation_speed_m_s", True),
        ("max_rotation_speed_rad_s", np.nan),
        ("max_rotation_speed_rad_s", "0.5"),
        ("min_depth_coverage", 0),
        ("min_depth_coverage", 1.1),
        ("min_depth_coverage", np.nan),
        ("min_depth_coverage", True),
        ("min_depth_fraction", 0),
        ("min_precision", np.nan),
        ("min_recall", 1.1),
        ("recovery_box_threshold", np.nan),
        ("recovery_text_threshold", 1.1),
    ],
)
def test_invalid_tracking_config(name, value):
    with pytest.raises(ValueError, match=name):
        TrackingConfig(**{name: value})


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


@pytest.mark.parametrize(
    "invalid", ["scale", "reflection", "shear", "last_row", "behind", "camera_plane"]
)
@pytest.mark.parametrize(
    "stage", ["initialize", "track_mask", "track_depth", "recover"]
)
def test_invalid_pose_is_rejected_before_rendering(
    recovery_scene, monkeypatch, invalid, stage
):
    from dataclasses import replace
    from cordvip_geometry import tracking

    scene, frame, _, _, _ = recovery_scene
    pose = front_pose()
    if invalid == "scale":
        pose[0, 0] = 2
    elif invalid == "reflection":
        pose[0, 0] = -1
    elif invalid == "shear":
        pose[0, 1] = 0.2
    elif invalid == "last_row":
        pose[3, 0] = 0.1
    else:
        pose[2, 3] = -1 if invalid == "behind" else 0

    def must_not_render(*args):
        pytest.fail("invalid candidate reached the renderer")

    monkeypatch.setattr(scene.renderers["apple"], "render", must_not_render)
    previous = scene.trackers["apple"].estimator.pose_last
    if stage in ("initialize", "recover"):
        def register(*args):
            scene.trackers["apple"].estimator.pose_last = pose
            return pose
        monkeypatch.setattr(scene.trackers["apple"], "register", register)
        if stage == "initialize":
            with pytest.raises(TrackingFailure, match="invalid|in front"):
                scene.initialize(frame, {"apple": "apple", "banana": "banana"})
            assert scene.trackers["apple"].estimator.pose_last is None
            assert scene.states == {"apple": "LOST", "banana": "TRACKING"}
            return
        scene.states["apple"] = "LOST"
        scene.next_retry["apple"] = 0
    else:
        original = tracking.track_objects
        def batch(*args):
            poses = original(*args)
            poses["apple"] = pose
            scene.trackers["apple"].estimator.pose_last = pose
            return poses
        monkeypatch.setattr(tracking, "track_objects", batch)
        if stage == "track_depth":
            scene.config = replace(scene.config, segmentation_interval_frames=2)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    assert list(result.poses) == ["banana"]
    assert scene.states["apple"] == "LOST"
    assert scene.trackers["apple"].estimator.pose_last is previous


def test_pose_front_check_uses_mesh_center(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, _, _, _ = recovery_scene
    original_batch = tracking.track_objects
    scene.trackers["apple"].object_from_center[2, 3] = 2

    def batch(*args):
        poses = original_batch(*args)
        poses["apple"][2, 3] = -1  # 原点在相机后方，居中网格仍在前方。
        return poses

    monkeypatch.setattr(tracking, "track_objects", batch)
    result = scene.track(frame)
    assert all(result.valid.values())
    assert result.poses["apple"][2, 3] == -1
    assert scene.trackers["apple"].estimator.pose_last[2, 3] == 1


def test_pose_gate_tolerates_float32_rotation_noise(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking
    from scipy.spatial.transform import Rotation

    scene, frame, _, _, _ = recovery_scene
    original = tracking.track_objects
    def batch(*args):
        poses = original(*args)
        poses["apple"] = poses["apple"].astype(np.float32)
        poses["apple"][:3, :3] = Rotation.from_euler("z", 0.3).as_matrix().astype(np.float32)
        poses["apple"][0, 0] += 1e-5
        scene.trackers["apple"].estimator.pose_last = poses["apple"]
        return poses
    monkeypatch.setattr(tracking, "track_objects", batch)
    assert all(scene.track(frame).valid.values())


def test_annotator_skips_existing_file_without_loading_inputs(monkeypatch, tmp_path):
    module = script("annotate_episodes")
    (tmp_path / "hoi.h5").write_bytes(b"do not inspect or change")
    monkeypatch.setattr(
        module,
        "RawReader",
        lambda *args, **kwargs: pytest.fail("unexpected Raw read"),
    )
    module.annotate_episodes([tmp_path], tmp_path / "unused_task")
    assert (tmp_path / "hoi.h5").read_bytes() == b"do not inspect or change"


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
        "arm_qpos": np.zeros((1, 7)),
        "hand_qpos": np.zeros((1, 12)),
    }
    viewer.reader = {"timing/rows": [{"camera_ns": 1, "arm_ns": 1, "hand_ns": 1}]}

    class Reader(dict):
        hoi_file = {"quality": {}}

    viewer.reader = Reader(viewer.reader)
    viewer.log_hoi_diagnostics(0, viewer.hoi_alignment)
    assert logged["hoi/alignment_available"] == 1
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


@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_depth_gate_counts_both_directions(offset):
    from cordvip_geometry.tracking import pose_depth_quality

    rendered = np.ones((10, 10))
    observed = rendered.copy()
    observed[:6] += offset
    metrics, verdict = pose_depth_quality(observed, rendered, TrackingConfig())
    assert verdict == "LOST" and metrics["depth_pixels"] == 40
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
    assert scene.track(frame).valid == {"apple": False, "banana": True}
    assert all(scene.track(frame).valid.values())
    assert events == [("track", 1), ("track", 2), ("detect", 2), ("track", 3)]


def test_mixed_quality_hdf5_is_finite(recovery_scene, tmp_path):
    import json
    from dataclasses import asdict, replace

    import h5py

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
    path = tmp_path / "hoi.h5"
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
        assert json.loads(data.attrs["tracking_config"]) == asdict(scene.config)
        for values in data["quality"].values():
            assert np.isfinite(values[:]).all()
        np.testing.assert_array_equal(data["quality/mask_source"][:, 0], [1, 0, 1])
        assert data["object_pointcloud"].shape == (3, 1024, 3)
        np.testing.assert_array_equal(
            data["quality/depth_coverage"][:], np.full((3, 2), 0.75)
        )


def test_correction_uses_video_index(monkeypatch):
    from contextlib import nullcontext

    import torch
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
    import multiprocessing as mp
    from dataclasses import replace

    import h5py
    from cordvip_geometry import tracking

    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.export import ExportConfig, export_zarr
    from dexmani_real.data.hoi import write_hoi
    from dexmani_real.data.raw import TIMING_ROW_DTYPE
    from dexmani_real.geometry import hoi
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.perception.hoi_worker import run_hoi_worker
    from dexmani_real.runtime.observation import check_hoi_status

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=2)
    names = list(masks)
    calls = []

    def track(rgb):
        calls.append(scene.frame_index)
        if rgb[0, 0, 0] == 4:
            masks["apple"][:] = False
        return masks.copy()

    scene.segmenter.track = track
    sequences = np.array([1, 2, 3, 3, 4, 5, 6])
    stamps = 1_000_000_000 + (sequences - 1) * 100_000_000
    depth = np.full((len(sequences), 24, 24), 1000, np.uint16)
    depth[1] = 0  # 短暂完全遮挡，随后正常观测恢复。
    images = [np.full_like(frame.rgb, sequence) for sequence in sequences]
    timing = np.zeros(len(sequences), dtype=TIMING_ROW_DTYPE)
    timing["control_ns"] = 1_000_000_000 + np.arange(len(sequences)) * 100_000_000
    timing["observation_ns"] = timing["control_ns"] + 50_000_000
    timing["camera_sequence"], timing["camera_ns"] = sequences, stamps

    class Reader(FakeRawReader):
        num_frames = len(sequences)

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
            return depth if name == "depth" else timing

        def iter_rgb_frames(self):
            yield from images

    poses, base, quality, validity = script("annotate_episodes").track_episode(
        Reader(), scene, names, scene.prompts
    )
    expected_valid = [True, False, True, True, False, False, True]
    np.testing.assert_array_equal(validity.all(axis=1), expected_valid)
    assert np.isnan(poses[~validity]).all()
    assert calls == [1, 3, 4, 5] and quality[2] == quality[3]
    template = dict(
        object_names=names,
        points_local=np.ones((1024, 3), np.float32),
        offsets=[0, 512, 1024],
    )
    episode = tmp_path / "episode_test"
    episode.mkdir()
    with h5py.File(episode / "data.h5", "w") as data:
        data.create_group("meta").attrs.update(task_label="test", control_hz=10.0)
        data.create_dataset("timing/rows", data=timing)
        data.create_dataset("timing/segment_starts", data=np.array([0], np.int64))
    write_hoi(
        episode / "hoi.h5",
        template,
        poses,
        base,
        quality=quality,
        object_valid=validity,
        tracking_config={},
    )
    with h5py.File(episode / "hoi.h5", "r") as data:
        offline_cloud = data["object_pointcloud"][:]
        np.testing.assert_array_equal(data["object_valid"][:].all(axis=1), expected_valid)
        assert not offline_cloud[~np.array(expected_valid)].any()
    output = tmp_path / "train.zarr"
    with pytest.raises(ValueError, match="all Raw episodes rejected"):
        export_zarr(episode, output, processing=ExportConfig(), fields=("object_pointcloud",))
    assert not output.exists()

    calls.clear()
    for mask in masks.values():
        mask[:] = True
    monkeypatch.setattr(tracking, "SceneTracker", lambda *a, **kw: scene)
    monkeypatch.setattr(
        hoi, "task_objects", lambda task: [{"name": n, "prompt": n} for n in names]
    )
    monkeypatch.setattr("signal.signal", lambda *a: None)
    shared = SensorChannels(mp.get_context("spawn"), hoi=True)
    shared.hoi_request.write(episode_id=1, after_sequence=0, after_ns=0)
    frames = iter(
        [
            ({"timestamp_ns": stamp, "rgb": rgb, "depth": observed}, int(sequence))
            for stamp, rgb, observed, sequence in zip(stamps, images, depth, sequences)
        ]
    )
    shared.camera_ring = SimpleNamespace(read=lambda: next(frames))
    written = []
    states = []
    ring_write = shared.hoi_ring.write

    def write(**row):
        ring_write(**row)
        written.append(row)
        states.append(scene.states.copy())
        assert check_hoi_status(
            shared, RealConfig(), now_ns=int(row["timestamp_ns"])
        ) == ([] if row["valid"] else ["hoi_pose_invalid"])
        if len(written) == 6:
            shared.is_running.value = False

    monkeypatch.setattr(shared.hoi_ring, "write", write)
    run_hoi_worker(
        shared, Reader().read_camera(), tmp_path, template, tracking_config=scene.config
    )
    assert calls == [1, 3, 4, 5]
    assert [r["source_camera_sequence"] for r in written] == [1, 2, 3, 4, 5, 6]
    assert [state["apple"] for state in states] == [
        "TRACKING",
        "OCCLUDED",
        "TRACKING",
        "LOST",
        "LOST",
        "TRACKING",
    ]
    unique = [0, 1, 2, 4, 5, 6]
    np.testing.assert_array_equal(
        [r["valid"] for r in written], np.array(expected_valid)[unique]
    )
    np.testing.assert_array_equal(
        [r["object_pointcloud"] for r in written], offline_cloud[unique]
    )
    assert all(r["object_pointcloud"].dtype == np.float32 for r in written)
    assert scene.timestamp_ns == stamps[-1] and scene.source_sequence == 6
    assert scene.frame_dt_s == 0.1


@pytest.mark.parametrize(
    "observed,rendered", [(0.0, 1.0), (np.nan, 1.0), (1.0, 0.0), (1.0, np.inf)]
)
def test_depth_gate_requires_support(observed, rendered):
    from cordvip_geometry.tracking import pose_depth_quality

    metrics, verdict = pose_depth_quality(
        np.full((10, 10), observed), np.full((10, 10), rendered), TrackingConfig()
    )
    assert (
        verdict == "OCCLUDED"
        and metrics["depth_pixels"] == 0
        and metrics["depth_fraction"] == 0
        and metrics["depth_coverage"] == 0
    )


def test_depth_gate_boundary():
    from cordvip_geometry.tracking import pose_depth_quality

    observed = np.ones((10, 10))
    observed[:5] = 2
    metrics, verdict = pose_depth_quality(
        observed, np.ones_like(observed), TrackingConfig(min_depth_pixels=50)
    )
    assert verdict == "VALID" and metrics["depth_fraction"] == 0.5


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


def test_viewer_reads_current_quality_and_marks_unmeasured_values():
    from scripts.visualize_episode import EpisodeVisualizer

    logged = {}
    viewer = EpisodeVisualizer.__new__(EpisodeVisualizer)
    viewer.rr = SimpleNamespace(
        Scalar=float, log=lambda name, value: logged.update({name: value})
    )

    class Reader(dict):
        hoi_file = {
            "quality": {
                "iou": np.array([[-1.0, 0.75]]),
                "mask_source": np.array([[0.0, 1.0]]),
                "depth_fraction": np.array([[0.8, 0.9]]),
                "depth_coverage": np.array([[0.0, 0.7]]),
            }
        }

    viewer.reader = Reader(
        {"timing/rows": [{"arm_ns": 1, "hand_ns": 1, "camera_ns": 1}]}
    )
    alignment = {"arm_qpos": np.zeros((1, 7)), "hand_qpos": np.zeros((1, 12))}
    viewer.log_hoi_diagnostics(0, alignment)
    assert np.isnan(logged["hoi/object_0/iou"])
    assert logged["hoi/object_1/iou"] == 0.75
    assert logged["hoi/object_0/mask_source"] == 0
    assert logged["hoi/object_0/depth_fraction"] == 0.8
    assert logged["hoi/object_0/depth_coverage"] == 0.0
    assert logged["hoi/object_1/depth_coverage"] == 0.7
    del viewer.reader.hoi_file["quality"]
    with pytest.raises(KeyError, match="quality"):
        viewer.log_hoi_diagnostics(0, alignment)


@pytest.mark.parametrize("missing", [np.nan, 0.0])
def test_sparse_depth_requires_coverage(missing):
    from cordvip_geometry.tracking import pose_depth_quality, pose_quality

    rendered = np.ones((100, 100))
    observed = np.full_like(rendered, missing)
    observed.flat[:30] = 1
    config = TrackingConfig()
    depth, verdict = pose_depth_quality(observed, rendered, config)
    full, mask_verdict = pose_quality(
        np.ones_like(rendered, bool), observed, rendered, config
    )
    assert verdict == mask_verdict == "OCCLUDED"
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
    metrics, verdict = pose_depth_quality(observed, rendered, TrackingConfig())
    assert metrics["depth_coverage"] == support / 100
    assert metrics["depth_fraction"] == matched / (support * 100)
    expected = "OCCLUDED" if support < 50 else ("VALID" if fraction >= 0.5 else "LOST")
    assert verdict == expected


def test_depth_coverage_does_not_replace_minimum_count():
    from cordvip_geometry.tracking import pose_depth_quality

    metrics, verdict = pose_depth_quality(
        np.ones((5, 5)), np.ones((5, 5)), TrackingConfig()
    )
    assert verdict == "OCCLUDED" and metrics["depth_pixels"] == 25
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
    assert result.valid == {"apple": False, "banana": True}
    assert scene.states["apple"] == "OCCLUDED"
    assert sorted(renders) == ["apple", "banana"]
    assert result.quality["banana"]["depth_coverage"] == 1
    if not invalid_mask:
        assert result.quality["apple"]["depth_coverage"] == 5 / 12
        assert result.quality["apple"]["mask_source"] == 1


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
    assert full.keys() == depth.keys()
    assert full["depth_coverage"] == depth["depth_coverage"] == 0.75
    assert full["occluded_fraction"] == 0.25


@pytest.mark.parametrize("min_coverage", [0.25, 0.75])
@pytest.mark.parametrize("min_fraction", [0.4, 0.8])
def test_depth_thresholds_are_independent(min_coverage, min_fraction):
    from cordvip_geometry.tracking import pose_depth_quality

    observed = np.zeros((10, 10))
    observed[:5] = 2
    observed.flat[:30] = 1
    config = TrackingConfig(
        min_depth_coverage=min_coverage, min_depth_fraction=min_fraction
    )
    metrics, verdict = pose_depth_quality(observed, np.ones_like(observed), config)
    assert metrics["depth_coverage"] == 0.5
    assert metrics["depth_fraction"] == 0.6
    expected = (
        "OCCLUDED"
        if min_coverage > 0.5
        else ("VALID" if min_fraction <= 0.6 else "LOST")
    )
    assert verdict == expected


@pytest.mark.parametrize(
    "case,expected",
    [
        ("normal", "VALID"),
        ("partial_occlusion", "VALID"),
        ("large_occlusion", "OCCLUDED"),
        ("complete_occlusion", "OCCLUDED"),
        ("missing", "OCCLUDED"),
        ("nan", "OCCLUDED"),
        ("sparse_match", "OCCLUDED"),
        ("sparse_mismatch", "OCCLUDED"),
        ("depth_contradiction", "LOST"),
        ("empty_mask", "LOST"),
        ("oversized_cad", "LOST"),
        ("disjoint_mask", "LOST"),
    ],
)
def test_geometric_evidence_classes(case, expected):
    from cordvip_geometry.tracking import pose_quality

    config = TrackingConfig()
    mask = np.ones((20, 20), bool)
    rendered = np.ones(mask.shape)
    observed = rendered.copy()
    if case in ("partial_occlusion", "large_occlusion", "complete_occlusion"):
        hidden_rows = {
            "partial_occlusion": 10,
            "large_occlusion": 19,
            "complete_occlusion": 20,
        }[case]
        mask[:hidden_rows] = False
        observed[:hidden_rows] = 0.7
    elif case in ("missing", "nan", "sparse_match", "sparse_mismatch"):
        observed[:] = np.nan if case == "nan" else 0
        if case.startswith("sparse"):
            observed.flat[:30] = 1 if case == "sparse_match" else 2
    elif case == "depth_contradiction":
        observed[:] = 2
    elif case == "empty_mask":
        mask[:] = False
    elif case == "oversized_cad":
        mask[5:] = False
        observed[5:] = 2  # 背景更远，不能解释为前景遮挡。
    elif case == "disjoint_mask":
        mask[10:] = False
        rendered[:10] = 0
    metrics, verdict = pose_quality(mask, observed, rendered, config)
    assert verdict == expected


def test_sufficient_contour_contradiction_is_not_hidden_by_missing_mask_depth():
    from cordvip_geometry.tracking import pose_quality

    mask = np.zeros((20, 20), bool)
    mask[:5] = True
    observed = np.full(mask.shape, 2.0)
    observed[:5] = 0
    metrics, verdict = pose_quality(
        mask, observed, np.ones(mask.shape), TrackingConfig()
    )
    assert metrics["mask_depth_coverage"] == 0
    assert metrics["visible_depth_coverage"] == 0.75
    assert verdict == "LOST"


def test_mask_depth_absence_still_tracks_and_rolls_back(recovery_scene, monkeypatch):
    from cordvip_geometry import tracking

    scene, frame, masks, _, detections = recovery_scene
    masks["apple"][12:] = False
    scene.renderers["apple"].render = (
        lambda rays, pose: masks["apple"][::2, ::2].ravel().astype(float)
    )
    frame.depth_m[:12] = 0
    original = tracking.track_objects
    previous = scene.trackers["apple"].estimator.pose_last
    seeds = []

    def track(trackers, *args):
        seeds.append(trackers["apple"].estimator.pose_last)
        poses = original(trackers, *args)
        poses["apple"][0, 3] = 0.01
        trackers["apple"].estimator.pose_last = poses["apple"]
        return poses

    monkeypatch.setattr(tracking, "track_objects", track)
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": True}
    assert scene.states["apple"] == "OCCLUDED" and "apple" not in result.poses
    assert scene.trackers["apple"].estimator.pose_last is previous
    frame.depth_m[:] = 1
    assert all(scene.track(frame).valid.values())
    assert seeds == [previous, previous] and len(detections) == 1


def test_missing_current_mask_uses_depth_quality(recovery_scene):
    scene, frame, masks, _, _ = recovery_scene
    scene.segmenter.track = lambda rgb: {"banana": masks["banana"]}
    result = scene.track(frame)
    assert all(result.valid.values())
    assert result.quality["apple"]["mask_source"] == 0


@pytest.mark.parametrize("long_occlusion", [False, True])
def test_complete_occlusion_waits_then_tracks_or_recovers(
    recovery_scene, long_occlusion
):
    from dataclasses import replace

    scene, frame, masks, registrations, detections = recovery_scene
    frame = replace(frame, source_sequence=10, timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    detections.clear()
    registrations.clear()
    frame.depth_m[:] = 0
    first = scene.track(replace(frame, source_sequence=11, timestamp_ns=1_100_000_000))
    assert not any(first.valid.values()) and set(scene.states.values()) == {"OCCLUDED"}
    if long_occlusion:
        scene.track(replace(frame, source_sequence=25, timestamp_ns=1_590_000_000))
        assert set(scene.states.values()) == {"OCCLUDED"}
        scene.track(replace(frame, source_sequence=26, timestamp_ns=1_600_000_000))
        assert set(scene.states.values()) == {"LOST"} and not detections
        frame.depth_m[:] = 1
        pending = scene.track(
            replace(frame, source_sequence=29, timestamp_ns=1_700_000_000)
        )
        assert not any(pending.valid.values()) and set(scene.pending_pose) == {
            "apple",
            "banana",
        }
        result = scene.track(
            replace(frame, source_sequence=30, timestamp_ns=1_733_000_000)
        )
        assert len(detections) == 1 and registrations == ["apple", "banana"]
    else:
        frame.depth_m[:] = 1
        result = scene.track(
            replace(frame, source_sequence=17, timestamp_ns=1_300_000_000)
        )
        assert not detections and not registrations
    assert all(result.valid.values()) and not scene.occluded_since


def test_source_gaps_do_not_advance_sam_video_indices(recovery_scene):
    from dataclasses import replace

    scene, frame, masks, _, _ = recovery_scene
    scene.config = replace(scene.config, segmentation_interval_frames=2)
    frame = replace(frame, source_sequence=100, timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    scene.segmenter.state = {"num_frames": 1}
    calls = []

    def sam(rgb):
        calls.append(scene.segmenter.state["num_frames"])
        scene.segmenter.state["num_frames"] += 1
        return masks.copy()

    scene.segmenter.track = sam
    for sequence, stamp, age, dt, gap in (
        (105, 1_100_000_000, 5, 0.1, 5),
        (109, 1_200_000_000, 0, 0.1, 4),
        (120, 1_450_000_000, 11, 0.25, 11),
    ):
        result = scene.track(
            replace(frame, source_sequence=sequence, timestamp_ns=stamp)
        )
        assert all(result.valid.values())
        assert result.quality["apple"]["mask_age_frames"] == age
        assert result.quality["apple"]["dt_s"] == dt
        assert result.quality["apple"]["source_frame_gap"] == gap
    assert calls == [1] and scene.frame_index == 3
    assert scene.segmenter.state["num_frames"] == 2


@pytest.mark.parametrize("bounded", [False, True])
def test_pose_jump_checks_use_real_dt_and_preserve_geometry(
    recovery_scene, monkeypatch, bounded
):
    from dataclasses import replace
    from cordvip_geometry import tracking

    scene, frame, _, _, detections = recovery_scene
    scene.config = replace(
        scene.config, max_translation_speed_m_s=1.0 if bounded else None
    )
    frame = replace(frame, source_sequence=1, timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    previous = scene.trackers["apple"].estimator.pose_last
    original = tracking.track_objects
    renders = []
    scene.renderers["apple"].render = lambda rays, pose: renders.append(1) or np.ones(
        len(rays)
    )

    def jump(trackers, *args):
        poses = original(trackers, *args)
        poses["apple"][0, 3] = 1
        trackers["apple"].estimator.pose_last = poses["apple"]
        return poses

    monkeypatch.setattr(tracking, "track_objects", jump)
    result = scene.track(replace(frame, source_sequence=2, timestamp_ns=1_033_000_000))
    assert result.valid == {"apple": not bounded, "banana": True}
    assert result.quality["apple"]["depth_fraction"] == 1
    assert result.quality["apple"]["translation_delta_m"] == 1
    assert result.quality["apple"]["pose_dt_s"] == 0.033
    assert renders == [1] and len(detections) == 2  # 两次初始化；跳变帧不检测。
    if bounded:
        assert scene.states["apple"] == "LOST"
        assert scene.trackers["apple"].estimator.pose_last is previous


@pytest.mark.parametrize("offset, continuous", [(0.0, True), (1.0, False)])
def test_motion_limits_use_one_symmetry_equivalent_pose(offset, continuous):
    from cordvip_geometry.tracking import pose_motion

    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    symmetry = half_turn.copy()
    symmetry[0, 3] = offset
    tracker = SimpleNamespace(
        estimator=SimpleNamespace(diameter=0.1),
        object_from_center=np.eye(4),
        symmetry_tfs=np.stack([np.eye(4), symmetry]),
    )
    config = TrackingConfig(max_translation_speed_m_s=0.1, max_rotation_speed_rad_s=0.1)
    metrics, allowed = pose_motion(np.eye(4), half_turn, tracker, 0.01, config)
    assert allowed is continuous
    assert metrics["rotation_delta_rad"] == 0
    assert metrics["translation_delta_m"] == offset


def test_motion_uses_mesh_center_instead_of_object_origin():
    from cordvip_geometry.tracking import pose_motion

    object_from_center = np.eye(4)
    object_from_center[0, 3] = 2
    tracker = SimpleNamespace(
        estimator=SimpleNamespace(diameter=0.1),
        object_from_center=object_from_center,
        symmetry_tfs=np.eye(4)[None],
    )
    center_from_object = np.linalg.inv(object_from_center)
    half_turn = np.diag([-1.0, -1.0, 1.0, 1.0])
    metrics, allowed = pose_motion(
        center_from_object,
        half_turn @ center_from_object,
        tracker,
        0.1,
        TrackingConfig(max_translation_speed_m_s=0.1),
    )
    assert allowed and metrics["translation_delta_m"] == 0


@pytest.mark.parametrize(
    "failure", ["none", "depth", "motion", "inference", "batch_retry"]
)
def test_pending_pose_is_temporary_and_confirmation_is_independent(
    recovery_scene, monkeypatch, failure
):
    from dataclasses import replace
    from cordvip_geometry import tracking

    scene, frame, masks, _, detections = recovery_scene
    scene.config = replace(scene.config, max_translation_speed_m_s=1.0)
    frame = replace(frame, source_sequence=10, timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    detections.clear()
    masks["apple"][:] = False
    previous = scene.trackers["apple"].estimator.pose_last
    scene.track(replace(frame, source_sequence=11, timestamp_ns=1_010_000_000))
    pending_internal = front_pose()
    pending_internal[0, 3] = 0.7
    pending_public = front_pose()
    pending_public[0, 3] = 0.2

    def register(*args):
        scene.trackers["apple"].estimator.pose_last = pending_internal
        return pending_public

    scene.trackers["apple"].register = register
    pending = scene.track(
        replace(frame, source_sequence=12, timestamp_ns=1_020_000_000)
    )
    assert pending.valid == {"apple": False, "banana": True}
    assert "apple" not in pending.poses and "apple" in scene.pending_pose
    assert scene.trackers["apple"].estimator.pose_last is previous
    retry = scene.next_retry["apple"]
    original = tracking.track_objects
    seeds = []

    def confirm(trackers, *args):
        if failure == "batch_retry" and len(trackers) > 1:
            for tracker in trackers.values():
                tracker.estimator.pose_last = np.full((4, 4), 99.0)
            raise RuntimeError("batch failed")
        if "apple" in trackers:
            seeds.append(trackers["apple"].estimator.pose_last)
            if failure == "inference":
                raise TrackingFailure("apple inference failed")
        poses = original(trackers, *args)
        if "apple" in trackers:
            poses["apple"] = pending_public.copy()
            poses["apple"][0, 3] += 1 if failure == "motion" else 0.005
            trackers["apple"].estimator.pose_last = poses["apple"]
        return poses

    monkeypatch.setattr(tracking, "track_objects", confirm)
    if failure == "depth":
        scene.renderers["apple"].render = lambda rays, pose: np.full(len(rays), 2.0)
    result = scene.track(replace(frame, source_sequence=13, timestamp_ns=1_030_000_000))
    succeeds = failure in ("none", "batch_retry")
    assert result.valid == {"apple": succeeds, "banana": True}
    assert all(seed is pending_internal for seed in seeds)
    assert not scene.pending_pose
    assert len(detections) == 1 and scene.next_retry["apple"] == retry
    if not succeeds:
        assert scene.states["apple"] == "LOST" and "apple" not in result.poses
        assert scene.trackers["apple"].estimator.pose_last is previous
        scene.track(replace(frame, source_sequence=14, timestamp_ns=1_040_000_000))
        assert len(detections) == 1  # 失败确认仍遵守原定第 5 处理帧重试时间。


def test_recovery_mask_merge_rechecks_healthy_object_once(recovery_scene):
    scene, frame, masks, _, _ = recovery_scene
    masks["apple"][:] = False
    scene.track(frame)
    renders = []
    for name in masks:
        scene.renderers[name].render = lambda rays, pose, name=name: renders.append(
            name
        ) or np.ones(len(rays))

    def merge(corrections):
        masks.update(corrections)
        masks["banana"][:] = False
        return masks.copy()

    scene.segmenter.correct_masks = merge
    result = scene.track(frame)
    assert result.valid == {"apple": False, "banana": False}
    assert set(scene.pending_pose) == {"apple"} and scene.states["banana"] == "LOST"
    assert sorted(renders) == ["apple", "banana"]


@pytest.mark.parametrize(
    "sequence,stamp,reason",
    [
        (np.uint64(9), 1_010_000_000, "new camera source sequence"),
        (np.uint64(10), 1_010_000_000, "new camera source sequence"),
        (np.uint64(11), 999_000_000, "moved backwards"),
        (np.uint64(0), 1_010_000_000, "must be positive"),
    ],
)
def test_scene_rejects_invalid_source_metadata(recovery_scene, sequence, stamp, reason):
    from dataclasses import replace

    scene, frame, _, _, _ = recovery_scene
    frame = replace(frame, source_sequence=np.uint64(10), timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    with pytest.raises(ValueError, match=reason):
        scene.track(replace(frame, source_sequence=sequence, timestamp_ns=stamp))


def test_rgbd_frame_carries_source_metadata():
    camera = FakeRawReader().read_camera()
    row = dict(
        rgb=np.zeros((2, 2, 3), np.uint8),
        depth=np.full((2, 2), 1000, np.uint16),
        timestamp_ns=np.uint64(1_000_000_000),
    )
    frame = RGBDFrame.from_camera(camera, row, source_sequence=11)
    assert frame.timestamp_ns == 1_000_000_000 and frame.source_sequence == 11
    assert frame.depth_m.dtype == np.float32 and np.all(frame.depth_m == 1)
    assert RGBDFrame.from_camera(camera, row).source_sequence is None


def test_multiple_moving_objects_share_one_batch_per_frame(recovery_scene, monkeypatch):
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from cordvip_geometry import tracking

    scene, frame, _, _, _ = recovery_scene
    scene.config = replace(
        scene.config,
        segmentation_interval_frames=2,
        max_translation_speed_m_s=1,
        max_rotation_speed_rad_s=5,
    )
    frame = replace(frame, source_sequence=1, timestamp_ns=1_000_000_000)
    scene.initialize(frame, scene.prompts)
    calls = []

    def move(trackers, rgb, depth, K, cache):
        calls.append(list(trackers))
        poses = {}
        for index, (name, tracker) in enumerate(trackers.items(), start=1):
            pose = front_pose()
            pose[0, 3] = index * scene.frame_index * 0.01
            pose[:3, :3] = Rotation.from_euler(
                "z", index * scene.frame_index * 0.1
            ).as_matrix()
            tracker.estimator.pose_last = pose
            poses[name] = pose
        return poses

    monkeypatch.setattr(tracking, "track_objects", move)
    for step in range(1, 4):
        result = scene.track(
            replace(
                frame,
                source_sequence=step + 1,
                timestamp_ns=1_000_000_000 + step * 100_000_000,
            )
        )
        assert all(result.valid.values()) and list(result.poses) == ["apple", "banana"]
        assert result.poses["apple"][0, 3] == step * 0.01
        assert result.poses["banana"][0, 3] == step * 0.02
    assert calls == [["apple", "banana"]] * 3


@pytest.mark.parametrize("info", [False, True])
def test_visualizer_can_bypass_corrupt_hoi(annotated_episode, monkeypatch, info):
    import h5py

    from scripts import visualize_episode as visual
    from dexmani_real.utils import log

    monkeypatch.setattr(log, "configure_logging", lambda: None)

    episode, *_ = annotated_episode
    (episode / "hoi.h5").write_bytes(b"corrupt annotation")
    _disable_viewer(monkeypatch, visual)

    def unexpected_hoi(*args, **kwargs):
        raise AssertionError("Raw viewing must not consume HOI")

    monkeypatch.setattr(visual, "align_hoi_episode", unexpected_hoi)
    monkeypatch.setattr(visual, "read_hoi_validity", unexpected_hoi)
    monkeypatch.setattr(visual.EpisodeVisualizer, "log_hoi_diagnostics", unexpected_hoi)
    opened = []
    original = h5py.File

    def open_file(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", open_file)
    args = [str(episode), "--no-hoi", "--max-frames", "1", "--no-point-cloud"]
    if info:
        args.append("--info")
    assert visual.main(args) == 0
    assert "hoi.h5" not in opened
    with pytest.raises(OSError):
        visual.main([arg for arg in args if arg != "--no-hoi"])


def _disable_viewer(monkeypatch, visual):
    import rerun as rr
    for name in ("init", "send_blueprint", "disconnect", "log", "set_time_sequence", "set_time_seconds"):
        monkeypatch.setattr(rr, name, lambda *a, **kw: None)
    for name in ("log_rgbd", "log_series", "log_camera_geometry", "log_points"):
        monkeypatch.setattr(visual, name, lambda *a, **kw: None)


def test_visualizer_reads_only_consumed_prefix(annotated_episode, monkeypatch):
    import h5py

    from scripts import visualize_episode as visual

    episode, *_ = annotated_episode
    with h5py.File(episode / "data.h5", "r+") as data:
        data["arm_qpos"][2] = np.nan
        rows = data["timing/rows"][:]
        rows["control_ns"][2] = -1
        data["timing/rows"][:] = rows
    _disable_viewer(monkeypatch, visual)
    original = h5py.Dataset.__getitem__
    reads = []

    def read(dataset, key, **kwargs):
        if dataset.name in {"/arm_qpos", "/hand_qpos", "/object_valid", "/timing/rows"}:
            reads.append((dataset.name, key))
            assert (isinstance(key, slice) and key.stop == 1) or key == 0
        return original(dataset, key, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", read)
    viewer = visual.EpisodeVisualizer(str(episode), table_plane_path=None, max_frames=1)
    try:
        assert len(viewer.timestamps_s) == len(viewer.hoi_valid) == 1
        assert all(len(q) == 1 for q in viewer.hoi_alignment.values())
        viewer.log_step(0, next(viewer.reader.iter_rgb_frames()))
    finally:
        viewer.close()
    assert {name for name, _ in reads} == {"/arm_qpos", "/hand_qpos", "/object_valid", "/timing/rows"}


@pytest.mark.parametrize("devices", [("arm",), ("arm", "hand")])
@pytest.mark.parametrize("stamps", [[], [100_000_000], [100_000_000, 180_000_000, 300_000_000]])
@pytest.mark.parametrize("target", [90_000_000, 100_000_000, 120_000_000, 180_000_000, 240_000_000, 310_000_000])
@pytest.mark.parametrize("gap", [.08, .079999999])
def test_scalar_alignment_matches_offline(devices, stamps, target, gap):
    from collections import deque

    from dexmani_real.robot.system import Robot
    from dexmani_real.utils.timing import interpolate_joints

    robot = Robot.__new__(Robot)
    robot.enable_joint_history(.35, devices=devices)
    expected = {}
    for name in devices:
        width = 7 if name == "arm" else 12
        positions = np.arange(len(stamps) * width, dtype=float).reshape(-1, width)
        robot.joint_history[name] = deque(zip(stamps, positions.copy()))
        if stamps:
            q, valid = interpolate_joints(stamps, positions, [target], gap)
            if valid[0]:
                expected[name + "_qpos"] = q[0]
    actual = robot.aligned_joints(target, gap)
    if not expected:
        assert actual is None
    else:
        assert set(actual) == set(expected)
        for key, values in expected.items():
            np.testing.assert_allclose(actual[key], values)
            actual[key][:] = -999
        again = robot.aligned_joints(target, gap)
        for key, values in expected.items():
            np.testing.assert_allclose(again[key], values)
    if "hand" in devices:
        robot.joint_history["hand"].clear()
        assert robot.aligned_joints(target, gap) is None
    robot.enable_joint_history(.35, devices=devices)
    assert robot.aligned_joints(target, gap) is None


@pytest.fixture
def feedback_robot():
    from threading import Event

    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE

    robot = Robot.__new__(Robot)
    robot.runtime_state = SimpleNamespace(stop_signal=Event())
    robot.check = lambda: None
    robot.read_hand = lambda: None
    robot._check_hand_feedback = lambda: None
    frame = np.zeros(1, ARM_STATE_DTYPE)
    robot.read_state = lambda: SimpleNamespace(arm=frame, hand=None)
    robot.enable_joint_history(.1, devices=("arm",))

    def feed(stamp, value=0):
        frame["timestamp_ns"] = stamp
        frame["qpos"] = value
        robot.feedback_next_ns = 0
        robot.service_feedback()

    return robot, frame, feed


@pytest.mark.parametrize("fault", ["zero", "reverse", "conflict"])
def test_joint_feedback_rejects_invalid_history(feedback_robot, fault):
    robot, _, feed = feedback_robot
    feed(1_000_000_000)
    with pytest.raises(ValueError, match="timestamp|Conflicting"):
        feed(0 if fault == "zero" else 900_000_000 if fault == "reverse" else 1_000_000_000,
             1 if fault == "conflict" else 0)
    assert robot.runtime_state.stop_signal.is_set()
    assert len(robot.joint_history["arm"]) == 1


def test_joint_feedback_duplicates_cropping_and_segments(feedback_robot):
    robot, frame, feed = feedback_robot
    feed(1_000_000_000)
    feed(1_000_000_000)
    assert len(robot.joint_history["arm"]) == 1
    assert robot.joint_history["arm"][0][0] == int(frame["timestamp_ns"][0])
    for index in range(1, 6):
        feed(1_000_000_000 + index * 40_000_000, index)
    # 1.1 s 窗口边界仍可使用 1.08 s 的左邻点。
    assert robot.joint_history["arm"][0][0] == 1_080_000_000
    np.testing.assert_allclose(robot.aligned_joints(1_100_000_000, .04)["arm_qpos"], 2.5)
    result = robot.aligned_joints(1_200_000_000, .04)
    result["arm_qpos"][:] = -10
    np.testing.assert_array_equal(frame["qpos"], 5)
    frame["qpos"] = -20
    np.testing.assert_array_equal(robot.aligned_joints(1_200_000_000, .04)["arm_qpos"], 5)
    robot.enable_joint_history(.1, devices=("arm",))
    assert robot.aligned_joints(1_100_000_000, .04) is None
