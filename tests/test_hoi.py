import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/cordvip_geometry/src")]
from cordvip_geometry.assets import prepare_asset
from dexmani_real.geometry.hoi import load_task_template, task_objects, transform_task_points
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import RawReader, TIMING_ROW_DTYPE
from dexmani_real.data.export import resolve_fields, export_zarr, ExportConfig
from scripts.cordvip import annotate_episodes as annotation

spec = importlib.util.spec_from_file_location(
    "allocate", ROOT / "scripts/cordvip/allocate_points.py"
)
allocate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocate)


@pytest.fixture
def task(tmp_path):
    directory = tmp_path / "task"
    directory.mkdir()
    for index, name in enumerate(("apple", "bowl")):
        mesh = trimesh.creation.box(extents=(1 + index, 2, 3))
        mesh.apply_translation((2, -1, 4))
        source = tmp_path / f"{name}.obj"
        mesh.export(source)
        prepare_asset(source, directory / name, scale_to_meters=0.1)
    allocate.allocate_task(directory)
    return directory


@pytest.mark.parametrize("objects,error", [
    ([{"name": "apple"}, {"name": "bowl"}], None),
    ([{"name": "bowl", "prompt": "  Blue bowl. "}, {"name": "apple", "prompt": "red apple"}], None),
    *[([{"name": "apple", "prompt": prompt}], "apple.*Prompt")
      for prompt in ("", " \t\n", ". .", "，。！", None, 123, ["apple"])],
    *[([{"name": "apple", "prompt": "Red apple"}, {"name": "bowl", "prompt": prompt}], "apple.*bowl")
      for prompt in ("red APPLE", " Red apple... ")],
    ([{"name": "apple"}, {"name": "bowl", "prompt": "APPLE."}], "apple.*bowl"),
])
def test_task_prompts(tmp_path, objects, error):
    source = json.dumps({"objects": objects})
    (tmp_path / "task.json").write_text(source)
    if error:
        with pytest.raises(ValueError, match=error):
            task_objects(tmp_path)
    else:
        expected = [dict(obj, prompt=obj.get("prompt", obj["name"])) for obj in objects]
        assert task_objects(tmp_path) == expected
    assert (tmp_path / "task.json").read_text() == source


@pytest.mark.parametrize("name", [None, 123, ["apple"], {"id": "apple"}, "a\0b", "../apple", ""])
def test_task_rejects_invalid_object_names(tmp_path, name):
    (tmp_path / "task.json").write_text(json.dumps({"objects": [{"name": name}]}))
    with pytest.raises(ValueError, match="非法对象名"):
        task_objects(tmp_path)


def test_task_rejects_duplicate_names_with_distinct_prompts(tmp_path):
    (tmp_path / "task.json").write_text(json.dumps({"objects": [
        {"name": "apple", "prompt": "red apple"},
        {"name": "apple", "prompt": "green apple"},
    ]}))
    with pytest.raises(ValueError, match="重复对象名"):
        task_objects(tmp_path)


def test_template(task):
    template, manifest = load_task_template(task)
    before = (task / "hoi/template.npz").read_bytes()
    allocate.allocate_task(task)
    assert before == (task / "hoi/template.npz").read_bytes()
    import shutil

    original_manifest = (task / "hoi/manifest.json").read_bytes()
    shutil.rmtree(task / "hoi")
    allocate.allocate_task(task)
    assert before == (task / "hoi/template.npz").read_bytes()
    assert original_manifest == (task / "hoi/manifest.json").read_bytes()
    assert manifest["object_names"] == ["apple", "bowl"]
    assert np.diff(template["offsets"]).sum() == 1024
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[:, :3, 3] = [0.2, -0.4, 0.7]
    base = np.eye(4)
    base[:3, 3] = [-0.3, 0.1, 0.8]
    points = transform_task_points(template, poses, base)
    for index in range(2):
        segment = slice(*template["offsets"][index : index + 2])
        homogeneous = np.c_[template["points_local"][segment], np.ones(len(points[segment]))]
        np.testing.assert_allclose(
            points[segment], (base @ poses[index] @ homogeneous.T).T[:, :3], atol=1e-6
        )
    (task / "apple/mesh/mesh_m.obj").unlink()
    with pytest.raises(ValueError, match="mesh_m.obj"):
        load_task_template(task)


@pytest.mark.parametrize("changed", ["mesh", "template"])
def test_template_identity_covers_used_geometry(task, changed):
    if changed == "mesh":
        path = task / "apple/mesh/mesh_m.obj"
        mesh = trimesh.load(path, process=False)
        mesh.apply_translation((0.01, 0, 0))
        mesh.export(path)
    else:
        template, _ = load_task_template(task)
        template["points_local"][0, 0] += 0.01
        np.savez(task / "hoi/template.npz", **template)
    with pytest.raises(ValueError, match="已改变"):
        load_task_template(task)


def test_template_rejects_task_name_mismatch(task):
    path = task / "hoi/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["task_name"] = "other_task"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="task_name.*目录名"):
        load_task_template(task)


def make_raw(path, *, segment_starts=(0,)):
    path.mkdir()
    with h5py.File(path / "data.h5", "w") as f:
        rows = np.zeros(2, dtype=TIMING_ROW_DTYPE)
        rows["control_ns"] = [1000000000, 1100000000]
        rows["observation_ns"] = rows["control_ns"] + 1000
        f.create_dataset("timing/rows", data=rows)
        f.create_dataset("timing/segment_starts", data=np.asarray(segment_starts, dtype=np.int64))
        f.create_group("meta").attrs["control_hz"] = 10
        f["meta"].attrs["task_label"] = "task"


@pytest.mark.parametrize("with_hoi", [False, True])
@pytest.mark.parametrize("task_label", ["task", "other_task", None])
def test_raw_task_identity(task, tmp_path, with_hoi, task_label):
    episode = tmp_path / "episode_1"
    make_raw(episode)
    if with_hoi:
        template, manifest = load_task_template(task)
        write_hoi(
            episode / "data.h5", template, manifest,
            np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1)),
            np.ones((2, 2), bool), np.ones((2, 2), np.float32), np.eye(4),
        )
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["meta"].attrs["task_label"]
        if task_label is not None:
            data["meta"].attrs["task_label"] = task_label
    if with_hoi and task_label != "task":
        with pytest.raises(ValueError, match="task_label"):
            RawReader(episode)
        with pytest.raises(ValueError, match="task_label"):
            annotation.annotate_episodes([episode], task)
        with pytest.raises(ValueError, match="task_label"):
            export_zarr(episode, tmp_path / "bad.zarr", processing=ExportConfig(), fields=["object_pointcloud"])
        assert not (tmp_path / "bad.zarr").exists()
    else:
        with RawReader(episode) as reader:
            assert (reader.hoi_metadata is not None) == with_hoi


def test_raw_export(task, tmp_path):
    import zarr

    template, manifest = load_task_template(task)
    episode = tmp_path / "episode_1"
    make_raw(episode, segment_starts=(0, 1))
    with RawReader(episode) as reader:
        assert "object_pointcloud" not in reader.fields
    with pytest.raises(ValueError, match="missing fields"):
        export_zarr(
            episode,
            tmp_path / "missing.zarr",
            processing=ExportConfig(),
            fields=["object_pointcloud"],
        )
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
    valid = np.ones((2, 2), bool)
    quality = np.ones((2, 2), np.float32)
    write_hoi(episode / "data.h5", template, manifest, poses, valid, quality, np.eye(4))
    with RawReader(episode) as reader:
        assert reader["object_pointcloud"].shape == (2, 1024, 3)
        assert reader.hoi_metadata["template_sha256"] == manifest["template_sha256"]
    assert len(resolve_fields(None)) == 13
    assert "object_pointcloud" not in resolve_fields(None)
    result = export_zarr(
        episode, tmp_path / "out.zarr", processing=ExportConfig(), fields=["object_pointcloud"]
    )
    assert result["raw_episode_count"] == 1
    assert result["episode_count"] == 2
    exported = zarr.open_group(str(tmp_path / "out.zarr"), mode="r")
    assert exported["meta/episode_ends"][:].tolist() == [1, 2]
    with RawReader(episode) as reader:
        np.testing.assert_array_equal(exported["data/object_pointcloud"][:], reader["object_pointcloud"][:])
    with h5py.File(episode / "data.h5", "r+") as f:
        f["hoi/object_pointcloud"][1, 0, 0] = np.nan
    with RawReader(episode) as reader:
        assert reader.hoi_metadata == manifest
    with pytest.raises(ValueError, match="invalid at row 1"):
        export_zarr(
            episode, tmp_path / "nan.zarr", processing=ExportConfig(), fields=["object_pointcloud"]
        )
    assert not (tmp_path / "nan.zarr").exists()
    with h5py.File(episode / "data.h5", "r+") as f:
        f["hoi/object_pose_valid"][1] = False
        f["hoi/object_pose_camera"][1] = np.nan
        f["hoi/object_pointcloud_valid"][1] = False
        f["hoi/object_pointcloud"][1] = 0
    with pytest.raises(ValueError, match="invalid at row 1"):
        export_zarr(
            episode, tmp_path / "bad.zarr", processing=ExportConfig(), fields=["object_pointcloud"]
        )
    assert not (tmp_path / "bad.zarr").exists()
    with h5py.File(episode / "data.h5", "r+") as f:
        f["hoi"].attrs["complete"] = False
    with pytest.raises(ValueError, match="未完成"):
        RawReader(episode)


def test_staging_ignored(tmp_path):
    episode = tmp_path / "episode_1"
    make_raw(episode)
    with h5py.File(episode / "data.h5", "r+") as f:
        f.create_group("_hoi_staging")
        f.create_dataset("object_pointcloud", data=np.zeros((2, 1024, 3), np.float32))
    with RawReader(episode) as reader:
        assert reader.hoi_metadata is None
        assert "object_pointcloud" not in reader.fields
        with pytest.raises(KeyError):
            reader["object_pointcloud"]


def test_export_rejects_different_templates(task, tmp_path):
    template, manifest = load_task_template(task)
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for i in range(2):
        episode = episodes / f"episode_{i}"
        make_raw(episode)
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
        write_hoi(
            episode / "data.h5",
            template,
            manifest,
            poses,
            np.ones((2, 2), bool),
            np.ones((2, 2), np.float32),
            np.eye(4),
        )
    with h5py.File(episode / "data.h5", "r+") as data:
        data["hoi"].attrs["template_sha256"] = "0" * 64
    output = tmp_path / "mixed.zarr"
    with pytest.raises(ValueError, match="HOI task/template mismatch"):
        export_zarr(episodes, output, processing=ExportConfig(), fields=["object_pointcloud"])
    assert not output.exists()


@pytest.mark.parametrize("scale", [None, 0, -1, float("nan")])
def test_asset_requires_real_scale(task, scale):
    from cordvip_geometry.assets import load_asset

    path = task / "apple/asset.json"
    metadata = json.loads(path.read_text())
    metadata["scale_to_meters"] = scale
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_asset(path.parent)


def test_allocation():
    assert allocate.allocate_counts([1, 1, 1]).tolist() == [342, 341, 341]
    assert allocate.allocate_counts(np.ones(16)).tolist() == [64] * 16
    with pytest.raises(ValueError):
        allocate.allocate_counts(np.ones(17))


def test_hoi_admission_and_clock():
    from types import SimpleNamespace
    from dexmani_real.runtime.observation import ObservationNeeds, check_observation
    from dexmani_real.deployment.preflight import validate_hoi_deployment

    names = SimpleNamespace(observation_fields=("joint_state",), hoi_pcd=None)
    assert validate_hoi_deployment(names, False) is None
    with pytest.raises(ValueError):
        validate_hoi_deployment(names, True)
    names.observation_fields = ("object_pointcloud",)
    with pytest.raises(ValueError):
        validate_hoi_deployment(names, False)
    with pytest.raises(ValueError, match="缺少"):
        validate_hoi_deployment(names, True)
    runtime = SimpleNamespace(camera=SimpleNamespace(max_frame_age_s=0.1), execution=SimpleNamespace(hoi_max_age_s=0.25))
    needs = ObservationNeeds(hoi=True)
    row = SimpleNamespace(hoi={"valid": True, "timestamp_ns": 1000000000})
    assert check_observation(row, runtime, needs, now_ns=1050000000) == []
    assert check_observation(row, runtime, needs, now_ns=1250000000) == ["hoi_stale"]
    row.hoi["valid"] = False
    assert check_observation(row, runtime, needs, now_ns=1050000000) == ["hoi_pose_invalid"]


@pytest.mark.parametrize("camera_fields", [(), ("rgb", "depth")])
def test_hoi_reads_only_requested_same_source_images(camera_fields, monkeypatch):
    import multiprocessing as mp
    from types import SimpleNamespace
    from unittest.mock import Mock
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.runtime.observation import ObservationNeeds, check_observation, read_observation

    channels = SensorChannels(
        mp.get_context("spawn"),
        camera=CameraConfig(width=4, height=4),
        hoi=True,
        camera_fields=camera_fields,
    )
    channels.hoi_request.write(episode_id=1, after_ns=10, after_sequence=0)
    rgb = np.full((4, 4, 3), 7, np.uint8)
    depth = np.full((4, 4), 8, np.uint16)
    channels.hoi_ring.write(
        episode_id=1,
        timestamp_ns=20,
        source_camera_sequence=9,
        valid=True,
        object_pointcloud=np.ones((1024, 3), np.float32),
        **{
            name: value for name, value in (("rgb", rgb), ("depth", depth)) if name in camera_fields
        },
    )
    assert set(channels.hoi_ring.dtype.names) & {"rgb", "depth"} == set(camera_fields)
    channels.camera_ring.write(timestamp_ns=30, rgb=rgb + 1, depth=depth + 1)
    robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=None, hand=None))
    runtime = SimpleNamespace(camera=CameraConfig(), execution=SimpleNamespace(hoi_max_age_s=0.25))
    with monkeypatch.context() as patch:
        patch.setattr(
            channels.camera_ring, "read", Mock(side_effect=AssertionError("异步相机帧不应被读取"))
        )
        needs = ObservationNeeds(hoi=True, camera_fields=camera_fields)
        row = read_observation(channels, robot, needs=needs)
        assert row.hoi["timestamp_ns"] == 20
        assert check_observation(row, runtime, needs, now_ns=30) == []
        assert set(row.hoi) & {"rgb", "depth"} == set(camera_fields)
        if camera_fields:
            assert row.camera["timestamp_ns"] == 20 and row.camera["ring_sequence"] == 9
            np.testing.assert_array_equal(row.camera["rgb"], rgb)
        else:
            assert row.camera is None
        for episode_id in (0, 2):
            channels.hoi_request.write(episode_id=episode_id, after_ns=30, after_sequence=1)
            row = read_observation(channels, robot, needs=needs)
            assert row.hoi is None and row.camera is None
            assert check_observation(row, runtime, needs, now_ns=30) == ["hoi_unavailable"]
    # 普通策略仍直接读取最新相机帧。
    row = read_observation(channels, robot, needs=ObservationNeeds(camera_fields=camera_fields))
    if camera_fields:
        assert row.camera["timestamp_ns"] == 30
        np.testing.assert_array_equal(row.camera["rgb"], rgb + 1)
    else:
        assert row.camera is None


def test_stale_hoi_does_not_enter_history(monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    from unittest.mock import Mock
    from dexmani_real.deployment.rollout import Rollout
    from dexmani_real.runtime.observation import ObservationNeeds

    row = SimpleNamespace(
        hoi={"timestamp_ns": 1_000_000_000, "valid": True},
    )
    rollout = Rollout.__new__(Rollout)
    rollout.model_needs = ObservationNeeds(hoi=True)
    rollout.runtime = SimpleNamespace(camera=SimpleNamespace(max_frame_age_s=0.133), execution=SimpleNamespace(hoi_max_age_s=0.25))
    rollout.policy_info = SimpleNamespace(n_obs_steps=2)
    rollout.observations = SimpleNamespace(
        project=Mock(return_value={"points": np.ones((1024, 3))})
    )
    rollout.history = deque(maxlen=2)
    rollout.model_input_issues = ()
    monkeypatch.setattr("time.monotonic_ns", lambda: 1_100_000_000)
    assert rollout.update_history(row)
    monkeypatch.setattr("time.monotonic_ns", lambda: 1_120_000_000)
    assert rollout.update_history(row)
    monkeypatch.setattr("time.monotonic_ns", lambda: 1_250_000_000)
    assert not rollout.update_history(row)
    assert not rollout.history
    assert rollout.observations.project.call_count == 2


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("failure", ["invalid_pose", "stale", "previous_episode", "worker_exit"])
def test_invalid_hoi_discards_plan(mode, failure):
    from types import SimpleNamespace
    from collections import deque
    from unittest.mock import Mock
    from dexmani_real.deployment.rollout import Rollout, EpisodeResult
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.runtime.state import SessionState

    rollout = Rollout.__new__(Rollout)
    rollout.result = EpisodeResult(1)
    rollout.runtime_state = SessionState()
    rollout.cancel_epoch = rollout.runtime_state.cancel_epoch
    rollout.robot = Mock()
    rollout.robot.read_state.return_value = SimpleNamespace(arm=None, hand=None)
    rollout.sensors = SimpleNamespace(channels=SimpleNamespace(
        hoi_ring=Mock(), hoi_request=Mock(), camera_ring=SimpleNamespace(latest_sequence=1),
    ))
    rollout.sensors.channels.hoi_ring.read.return_value = (
        {
            "episode_id": 0 if failure == "previous_episode" else 1,
            "valid": failure != "invalid_pose",
            "timestamp_ns": 0,
        },
        1,
    )
    rollout.sensors.channels.hoi_request.read.return_value = ({"episode_id": 1}, 1)
    rollout.runtime = SimpleNamespace(camera=SimpleNamespace(max_frame_age_s=0.1), execution=SimpleNamespace(hoi_max_age_s=0.25))
    rollout.model_needs = ObservationNeeds(hoi=True)
    rollout.execution = SimpleNamespace(mode=mode)
    rollout.recorder = None
    rollout.started_ns = 0
    rollout.run_deadline_ns = None
    rollout.plan = object()
    rollout.query = object()
    rollout.history = deque([{}])
    rollout.preparation_issue = None
    if failure == "worker_exit":
        from dexmani_real.deployment.session import PolicySession
        from dexmani_real.robot.commands import RobotError

        session = SimpleNamespace(keyboard=Mock(), sensors=Mock(), model_needs=rollout.model_needs)
        session.keyboard.check_health.return_value = True
        session.sensors.check.return_value = False
        rollout.robot.check.side_effect = lambda: PolicySession.check_services(session)
        rollout.prepare = Mock()
        with pytest.raises(RobotError, match="sensor process exited"):
            rollout.run(poll_operator=lambda: None)
        rollout.robot.stop.assert_called_once()
    else:
        rollout.step()
    assert rollout.finished
    assert rollout.plan is rollout.query is None
    assert not rollout.history
    assert rollout.runtime_state.stop_signal.is_set()
    rollout.robot.send_action.assert_not_called()


@pytest.mark.parametrize("camera_fields", [(), ("rgb", "depth")])
@pytest.mark.parametrize("transition", ["none", "next_episode", "inactive"])
def test_worker_geometry_and_source(task, monkeypatch, camera_fields, transition):
    import multiprocessing as mp
    import signal
    from contextlib import contextmanager
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.perception.hoi_worker import run_hoi_worker
    from cordvip_geometry.tracking import SceneTracker

    template, manifest = load_task_template(task)
    channels = SensorChannels(
        mp.get_context("spawn"),
        camera=CameraConfig(width=4, height=4),
        hoi=True,
        camera_fields=camera_fields,
    )
    channels.hoi_request.write(episode_id=2, after_ns=10, after_sequence=0)
    channels.camera_ring.write(
        timestamp_ns=20, rgb=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.uint16)
    )
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = [0.1, 0.2, 0.3]
    calls = []
    opened_trackers = []
    restart_episode = transition != "none"

    class Tracker:
        def reset(self):
            calls.append("reset")

        def detect_masks(self, frame, prompts):
            calls.append("detect")
            return {}

        def step(self, frame, initial_masks=None):
            calls.append("step")
            if restart_episode and calls.count("step") == 1:
                channels.hoi_request.write(
                    episode_id=0 if transition == "inactive" else 3, after_ns=30, after_sequence=1,
                )
                channels.camera_ring.write(
                    timestamp_ns=40,
                    rgb=np.ones((4, 4, 3), np.uint8),
                    depth=np.ones((4, 4), np.uint16),
                )
            return {
                "objects": {
                    name: {"pose_valid": True, "pose": pose}
                    for name, pose in zip(manifest["object_names"], poses)
                }
            }

    @contextmanager
    def opened(*args, **kwargs):
        tracker = Tracker()
        opened_trackers.append(tracker)
        yield tracker

    camera_read = channels.camera_ring.read

    def read_active_camera(*args, **kwargs):
        assert channels.hoi_request.read()[0]["episode_id"] != 0
        return camera_read(*args, **kwargs)

    sleeps = []

    def idle(seconds):
        assert transition == "inactive" and seconds == 0.005
        assert calls == ["reset", "detect", "step"]
        assert channels.hoi_ring.read() is None
        sleeps.append(seconds)
        assert len(sleeps) <= 2
        if len(sleeps) == 2:
            channels.hoi_request.write(
                episode_id=3, after_ns=50, after_sequence=channels.camera_ring.latest_sequence,
            )
        channels.camera_ring.write(
            timestamp_ns=40 + 10 * len(sleeps), rgb=np.ones((4, 4, 3), np.uint8),
            depth=np.ones((4, 4), np.uint16),
        )

    monkeypatch.setattr(SceneTracker, "open", opened)
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    monkeypatch.setattr(channels.camera_ring, "read", read_active_camera)
    monkeypatch.setattr("time.sleep", idle)
    write = channels.hoi_ring.write

    def finish(**payload):
        write(**payload)
        channels.is_running.value = False

    monkeypatch.setattr(channels.hoi_ring, "write", finish)
    camera = CameraSnapshot(
        None,
        CameraIntrinsics(
            width=4,
            height=4,
            fx=10,
            fy=10,
            ppx=2,
            ppy=2,
            distortion_model="none",
            distortion_coeffs=(0, 0, 0, 0, 0),
        ),
        0.001,
        np.eye(4),
    )
    run_hoi_worker(channels, camera, task, template)
    result = channels.hoi_ring.read()[0]
    assert len(opened_trackers) == 1
    assert len(sleeps) == (2 if transition == "inactive" else 0)
    assert result["episode_id"] == (3 if restart_episode else 2)
    assert result["timestamp_ns"] == (60 if transition == "inactive" else 40 if restart_episode else 20)
    assert result["source_camera_sequence"] == (4 if transition == "inactive" else 2 if restart_episode else 1)
    assert calls == ["reset", "detect", "step"] * (2 if restart_episode else 1)
    assert set(result.dtype.names) & {"rgb", "depth"} == set(camera_fields)
    if camera_fields:
        assert np.all(result["rgb"] == int(restart_episode))
    np.testing.assert_array_equal(
        result["object_pointcloud"], transform_task_points(template, poses, np.eye(4))
    )


def test_viewer_valid_invalid_and_absent_hoi(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import rerun as rr

    spec = importlib.util.spec_from_file_location(
        "hoi_viewer", ROOT / "scripts/visualize_episode.py"
    )
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    visualizer = viewer.EpisodeVisualizer.__new__(viewer.EpisodeVisualizer)
    visualizer.rr = Mock()
    visualizer.timestamps_s = [0, 0.1]
    visualizer.log_camera = Mock(return_value=None)
    visualizer.log_pointcloud = Mock()
    visualizer.log_hand_geometry = Mock()
    points = np.ones((2, 1024, 3), np.float32)

    class Reader:
        hoi_metadata = {}
        data_file = {"hoi": SimpleNamespace(attrs={"offsets": [0, 512, 1024]})}

        def __getitem__(self, key):
            return {
                "object_pointcloud_valid": [True, False],
                "object_pointcloud": points,
            }[key]

    visualizer.reader = Reader()
    visualizer.object_colors = np.zeros((1024, 3), np.uint8)
    logged = []
    monkeypatch.setattr(rr, "log", lambda path, value: logged.append((path, value)))
    visualizer.log_step(0, None)
    assert isinstance(logged[-1][1], rr.Points3D)
    visualizer.log_step(1, None)
    assert isinstance(logged[-1][1], rr.Clear)
    points[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        visualizer.log_step(0, None)
    visualizer.reader.hoi_metadata = None
    visualizer.log_step(0, None)
    assert len(logged) == 2


def test_interrupted_publish(task, tmp_path, monkeypatch):
    import dexmani_real.data.hoi as storage

    template, manifest = load_task_template(task)
    ep = tmp_path / "episode_1"
    make_raw(ep)
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
    valid = np.ones((2, 2), bool)
    original = storage.transform_task_points

    def fail(*args):
        raise RuntimeError("injected interruption")

    monkeypatch.setattr(storage, "transform_task_points", fail)
    with pytest.raises(RuntimeError, match="interruption"):
        write_hoi(
            ep / "data.h5",
            template,
            manifest,
            poses,
            valid,
            np.ones((2, 2), np.float32),
            np.eye(4),
        )
    with RawReader(ep) as reader:
        assert reader.hoi_metadata is None
        assert "_hoi_staging" in reader.data_file
    monkeypatch.setattr(storage, "transform_task_points", original)
    write_hoi(
        ep / "data.h5", template, manifest, poses, valid, np.ones((2, 2), np.float32), np.eye(4)
    )
    with RawReader(ep) as reader:
        assert reader.hoi_metadata is not None
        assert "_hoi_staging" not in reader.data_file


def test_nonfinite_transform_is_not_published(task, tmp_path):
    template, manifest = load_task_template(task)
    episode = tmp_path / "episode_1"
    make_raw(episode)
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 2, 1, 1))
    poses[1, 0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        write_hoi(
            episode / "data.h5",
            template,
            manifest,
            poses,
            np.ones((2, 2), bool),
            np.ones((2, 2), np.float32),
            np.eye(4),
        )
    with RawReader(episode) as reader:
        assert reader.hoi_metadata is None


def test_build_reuses_published_assets_without_models(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "build_task", ROOT / "scripts/cordvip/build_task.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    task = tmp_path / "task"
    objects = [{"name": name, "scale_to_meters": 0.1} for name in ("apple", "bowl")]
    for obj in objects:
        directory = task / obj["name"]
        directory.mkdir(parents=True)
        (directory / "reference.jpg").write_bytes(b"mock image")
    (task / "task.json").write_text(json.dumps({"objects": objects}))
    calls = []

    def build(args):
        calls.append(args)
        generation = args.output / "generation"
        generation.mkdir(parents=True)
        mesh = generation / "mesh_unscaled.obj"
        trimesh.creation.box().export(mesh)
        prepare_asset(mesh, args.output / "asset", scale_to_meters=args.scale_to_meters)

    monkeypatch.setattr(builder, "build", build)
    builder.build_task(task)
    for obj in objects:
        (task / obj["name"] / "reference.jpg").unlink()
    builder.build_task(task)
    assert len(calls) == 2
    assert [args.prompt for args in calls] == ["apple", "bowl"]
    assert all((task / obj["name"] / "mesh/mesh_m.obj").is_file() for obj in objects)
    (task / "apple/mesh/mesh_m.obj").unlink()
    with pytest.raises(ValueError, match="mesh_m.obj"):
        builder.build_task(task)


@pytest.mark.parametrize("timings,error", [
    ([(7, 10), (7, 10)], None),
    ([(7, 10), (8, 11)], None),
    ([(0, 10), (7, 10)], "非法相机时钟"),
    ([(7, 0), (8, 10)], "非法相机时钟"),
    ([(-1, 10), (7, 10)], "非法相机时钟"),
    ([(7, -1), (8, 10)], "非法相机时钟"),
    ([(7, 10), (6, 11)], "非法相机时钟"),
    ([(7, 10), (8, 9)], "非法相机时钟"),
    ([(7, 10), (7, 11)], "来源时间冲突"),
])
def test_batch_annotation_reuses_models_and_deduplicates_frames(task, tmp_path, monkeypatch, capsys, timings, error):
    from contextlib import contextmanager
    from types import SimpleNamespace

    episodes = [tmp_path / f"episode_{i}" for i in range(2)]
    for episode in episodes:
        make_raw(episode)
        with h5py.File(episode / "data.h5", "r+") as f:
            f["meta"].attrs["task_label"] = task.name
            rows = f["timing/rows"][:]
            rows["camera_sequence"], rows["camera_ns"] = np.asarray(timings).T
            rows["arm_ns"] = [0, -1]
            f["timing/rows"][:] = rows
    monkeypatch.setattr(
        RawReader, "read_camera", lambda self: SimpleNamespace(T_xarm_base_from_color=np.eye(4))
    )
    monkeypatch.setattr(
        annotation,
        "iter_raw_rows",
        lambda reader, camera: iter((None, row) for row in reader["timing/rows"][:]),
    )
    calls = []

    class Tracker:
        registration_attempts = {}
        max_reregistrations = 3

        def reset(self):
            calls.append("reset")

        def detect_masks(self, frame, prompts):
            assert prompts == {"apple": "apple", "bowl": "bowl"}
            calls.append("detect")
            return {}

        def step(self, *args, **kwargs):
            calls.append("step")
            return {
                "objects": {
                    name: {
                        "pose_valid": True,
                        "pose": np.eye(4),
                        "state": "TRACKING",
                        "error": None,
                        "cloud": SimpleNamespace(model_supported_fraction=1.0),
                    }
                    for name in ("apple", "bowl")
                }
            }

    @contextmanager
    def opened(*args, **kwargs):
        calls.append("open")
        yield Tracker()
        calls.append("close")

    monkeypatch.setattr(annotation.SceneTracker, "open", opened)
    if error:
        with pytest.raises(ValueError, match=error):
            annotation.annotate_episodes(episodes, task)
        with RawReader(episodes[0]) as reader:
            assert reader.hoi_metadata is None
        return
    annotation.annotate_episodes(episodes, task)
    steps = ["step"] * len({seq for seq, stamp in timings})
    assert calls == ["open", "detect", *steps, "reset", "detect", *steps, "close"]
    output = capsys.readouterr().out
    assert output.count("TRACKING=2, POSE_LOST=0") == 4
    assert output.count("耗尽=False") == 4
    assert output.count("P50/P95/Max(ms)=不可用, 有效样本=0/2") == 4
    for episode in episodes:
        with RawReader(episode) as reader:
            np.testing.assert_array_equal(
                reader["object_pointcloud"][0], reader["object_pointcloud"][1]
            )
    calls.clear()
    annotation.annotate_episodes(episodes, task)
    assert calls == []


@pytest.mark.parametrize("attempts,recovered", [(1, False), (3, False), (0, True)])
def test_annotation_diagnostics(task, tmp_path, monkeypatch, capsys, attempts, recovered):
    from contextlib import contextmanager
    from types import SimpleNamespace

    episode = tmp_path / "episode"
    make_raw(episode)
    timing = np.zeros(6, dtype=TIMING_ROW_DTYPE)
    timing["camera_sequence"] = [1, 2, 2, 3, 4, 4]
    timing["camera_ns"] = np.array([10, 20, 20, 30, 40, 40]) * 1_000_000
    timing["arm_ns"] = timing["camera_ns"] - np.arange(1, 7) * 1_000_000
    timing["arm_ns"][-2:] = [0, -1]
    timing["hand_ns"] = -1
    timing["hand_ns"][0] = 0
    with h5py.File(episode / "data.h5", "r+") as data:
        del data["timing/rows"]
        data.create_dataset("timing/rows", data=timing)
        data["meta"].attrs["task_label"] = task.name
    monkeypatch.setattr(RawReader, "read_camera", lambda self: SimpleNamespace(T_xarm_base_from_color=np.eye(4)))
    monkeypatch.setattr(annotation, "iter_raw_rows", lambda *args: iter((None, row) for row in timing))

    class Tracker:
        steps = 0
        max_reregistrations = 3
        registration_attempts = {"apple": attempts}

        def detect_masks(self, *args):
            return {}

        def step(self, *args, **kwargs):
            self.steps += 1
            objects = {}
            for name in ("apple", "bowl"):
                valid = name == "bowl" or self.steps == 1 or (recovered and self.steps == 4)
                objects[name] = dict(
                    pose_valid=valid, pose=np.eye(4) if valid else None,
                    state="TRACKING" if valid else "POSE_LOST",
                    error=None if valid else "empty mask" if self.steps == 2 else "depth gate",
                    cloud=SimpleNamespace(model_supported_fraction=float(valid)),
                )
            return {"objects": objects}

    tracker = Tracker()

    @contextmanager
    def opened(*args, **kwargs):
        assert kwargs["max_reregistrations"] == 3
        yield tracker

    monkeypatch.setattr(annotation.SceneTracker, "open", opened)
    annotation.annotate_episodes([episode], task)
    assert tracker.steps == 4
    output = capsys.readouterr().out
    apple_line = next(line for line in output.splitlines() if "apple:" in line)
    assert f"有效 {3 if recovered else 1}/6" in apple_line
    assert "首次丢失 Raw 行(0-based)=1" in apple_line
    assert f"TRACKING={3 if recovered else 1}, POSE_LOST={3 if recovered else 5}" in apple_line
    assert f"末态={'TRACKING' if recovered else 'POSE_LOST'}" in apple_line
    assert "最近失败原因=depth gate" in apple_line
    assert f"当前连续重注册={attempts}/3, 耗尽={attempts == 3}" in apple_line
    assert "bowl: 有效 6/6; 首次丢失 Raw 行(0-based)=None" in output
    assert "P50/P95/Max(ms)=2.500/3.850/4.000, 有效样本=4/6" in output
    assert "P50/P95/Max(ms)=不可用, 有效样本=0/6（非硬件同步误差）" in output
    with RawReader(episode) as reader:
        np.testing.assert_array_equal(reader["object_pose_valid"][:, 0], [True, False, False, False, recovered, recovered])
        np.testing.assert_array_equal(reader["timing/rows"][:], timing)


def test_manifest_order_change_rejected(task):
    (task / "task.json").write_text(json.dumps({"objects": [{"name": "bowl"}, {"name": "apple"}]}))
    with pytest.raises(ValueError, match="顺序"):
        load_task_template(task)


@pytest.mark.parametrize("field", ["camera_ns", "arm_ns", "hand_ns"])
@pytest.mark.parametrize("stamp", [-1, 0, 1_000_000])
def test_raw_hand_timing_requires_positive_sources(field, stamp):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from cordvip_geometry.raw import RawHandCloud

    source = RawHandCloud.__new__(RawHandCloud)
    source.reader = MagicMock(num_frames=1)
    source.reader.__getitem__.return_value = np.zeros((1, 3))
    source.hand_model = SimpleNamespace(transform=lambda joints: {})
    source.T_base_camera = np.eye(4)
    timing = dict(camera_ns=1_000_000, arm_ns=2_000_000, hand_ns=3_000_000)
    timing[field] = stamp
    result = source.for_row(SimpleNamespace(row_index=0), timing)
    assert result["timing_available"] == (stamp > 0)
    expected = (max(timing.values()) - min(timing.values())) / 1e6 if stamp > 0 else None
    assert result["skew_ms"] == expected


def test_checkpoint_asset_validation(task, tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    import dexmani_real.deployment.preflight as preflight

    template, manifest = load_task_template(task)
    root = tmp_path / "checkout"
    target = root / "assets/pointcloud_template/object/task"
    target.parent.mkdir(parents=True)
    shutil.copytree(task, target)
    monkeypatch.setattr(preflight, "__file__", str(root / "dexmani_real/deployment/preflight.py"))
    info = SimpleNamespace(
        observation_fields=("object_pointcloud",), task_name="task", hoi_pcd=manifest
    )
    path, loaded = preflight.validate_hoi_deployment(info, True)
    assert path == target
    np.testing.assert_array_equal(loaded["points_local"], template["points_local"])
    info.hoi_pcd = dict(info.hoi_pcd, template_sha256="0" * 64)
    with pytest.raises(ValueError, match="哈希"):
        preflight.validate_hoi_deployment(info, True)


@pytest.mark.parametrize("entry", ["preflight", "worker", "offline"])
def test_prompt_validation_before_models(task, monkeypatch, entry):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cordvip_geometry.tracking import SceneTracker
    from dexmani_real.deployment import preflight
    from dexmani_real.perception.hoi_worker import run_hoi_worker

    template, manifest = load_task_template(task)
    (task / "task.json").write_text(json.dumps({"objects": [
        {"name": "apple"}, {"name": "bowl", "prompt": "APPLE."},
    ]}))
    opened = Mock()
    monkeypatch.setattr(SceneTracker, "open", opened)
    monkeypatch.setattr("signal.signal", lambda *args: None)
    if entry == "preflight":
        root = task.parent
        destination = root / "assets/pointcloud_template/object/task"
        destination.parent.mkdir(parents=True)
        task.rename(destination)
        monkeypatch.setattr(preflight, "__file__", str(root / "dexmani_real/deployment/preflight.py"))
        info = SimpleNamespace(observation_fields=("object_pointcloud",), task_name="task", hoi_pcd=manifest)
        call = lambda: preflight.validate_hoi_deployment(info, True)
    elif entry == "worker":
        call = lambda: run_hoi_worker(None, None, task, template)
    else:
        call = lambda: annotation.annotate_episodes([], task)
    with pytest.raises(ValueError, match="apple.*bowl"):
        call()
    opened.assert_not_called()


def test_sam_history_keeps_conditioning_and_recent_frames():
    from cordvip_geometry.sam2_worker import prune_state

    outputs = [
        {
            "cond_frame_outputs": {0: "initial"},
            "non_cond_frame_outputs": dict.fromkeys(range(1, 1000)),
        }
        for _ in range(3)
    ]
    state = {
        "output_dict": outputs[0],
        "output_dict_per_obj": {0: outputs[1]},
        "temp_output_dict_per_obj": {0: outputs[2]},
        "frames_already_tracked": dict.fromkeys((0, 1, 980, 999)),
        "consolidated_frame_inds": {"non_cond_frame_outputs": set(range(1, 1000))},
    }
    prune_state(state, 999, 15)
    for output in outputs:
        assert output["cond_frame_outputs"] == {0: "initial"}
        assert set(output["non_cond_frame_outputs"]) == set(range(985, 1000))
    assert set(state["frames_already_tracked"]) == {0, 999}
    assert state["consolidated_frame_inds"]["non_cond_frame_outputs"] == set(range(985, 1000))


@pytest.fixture
def training_data(task, tmp_path):
    import zarr
    from omegaconf import OmegaConf

    _, manifest = load_task_template(task)
    path = tmp_path / "training.zarr"
    root = zarr.open_group(str(path), mode="w")
    data = root.create_group("data")
    for name, shape in (("object_pointcloud", (8, 1024, 3)), ("joint_state", (8, 19)), ("action", (8, 19))):
        data.create_dataset(name, data=np.zeros(shape, np.float32))
    root.create_group("meta").create_dataset("episode_ends", data=np.array([8], np.int64))
    root.attrs["hoi_pcd"] = manifest
    cfg = OmegaConf.create({
        "task_name": "task", "action_key": "action", "agent": {"action_dim": 19},
        "dataset": {
            "_target_": "dexmani_policy.datasets.demonstration.DemonstrationDataset",
            "zarr_path": str(path), "sensor_modalities": ["object_pointcloud"],
            "obs_horizon": 3, "horizon": 4, "pc_dim": 3,
        },
    })
    return cfg, root, manifest


@pytest.mark.policy_integration
@pytest.mark.parametrize("variant", ["canonical", "uppercase_hash", "extra_metadata"])
def test_training_metadata_reaches_deployment(task, training_data, tmp_path, monkeypatch, variant):
    import shutil
    from types import SimpleNamespace
    from dexmani_policy.deployment.metadata import policy_info
    from dexmani_policy.training.common.config import assemble_effective_config
    from dexmani_policy.training.common.data import build_dataset
    import dexmani_real.deployment.preflight as preflight

    cfg, root, manifest = training_data
    metadata = dict(manifest)
    if variant == "uppercase_hash":
        metadata["template_sha256"] = metadata["template_sha256"].upper()
    elif variant == "extra_metadata":
        metadata["note"] = "dataset annotation"
    root.attrs["hoi_pcd"] = metadata
    dataset, description = build_dataset(cfg)
    assert description["deployment_metadata"]["hoi_pcd"] == manifest
    assert root.attrs["hoi_pcd"] == metadata
    assert dataset[0]["obs"]["object_pointcloud"].shape == (3, 1024, 3)
    agent = SimpleNamespace(
        normalization_dims=description["field_dims"], n_obs_steps=3, n_action_steps=8,
        future_steps=14, control_action_dim=19, supports_rtc=False, effective_config={},
    )
    cfg = assemble_effective_config(cfg, description, agent)
    info = policy_info(tmp_path / "policy.pt", cfg, agent, weights="raw", inference_steps=10)
    assert info.to_dict()["hoi_pcd"] == manifest
    checkout = tmp_path / "checkout"
    target = checkout / "assets/pointcloud_template/object/task"
    shutil.copytree(task, target)
    monkeypatch.setattr(preflight, "__file__", str(checkout / "dexmani_real/deployment/preflight.py"))
    loaded_path, template = preflight.validate_hoi_deployment(info, True)
    assert loaded_path == target
    assert template["points_local"].shape == (1024, 3)


@pytest.mark.policy_integration
@pytest.mark.parametrize("field,value", [
    ("hoi_pcd", ...), ("hoi_pcd", None),
    ("task_name", ...), ("task_name", "wrong"), ("task_name", "  "), ("task_name", 123),
    ("object_names", ...), ("object_names", []), ("object_names", ["apple", "apple"]),
    ("object_names", "apple"), ("object_names", [" "]), ("object_names", [["apple"]]),
    ("template_sha256", ...), ("template_sha256", "a" * 63),
    ("template_sha256", "g" * 64), ("template_sha256", 123),
])
def test_training_rejects_invalid_manifest(training_data, field, value):
    from dexmani_policy.training.common.data import build_dataset

    cfg, root, manifest = training_data
    attrs = root.attrs if field == "hoi_pcd" else manifest
    if value is ...:
        del attrs[field]
    else:
        attrs[field] = value
    if field != "hoi_pcd":
        root.attrs["hoi_pcd"] = attrs
    with pytest.raises(ValueError, match=field):
        build_dataset(cfg)


@pytest.mark.policy_integration
@pytest.mark.parametrize("shape,dtype,error", [
    ((8, 512, 3), np.float32, "float32"), ((8, 1024, 6), np.float32, "float32"),
    ((8, 3), np.float32, "shape"), ((8, 1024, 3), np.float64, "float32"),
])
def test_training_rejects_invalid_object_points(training_data, shape, dtype, error):
    from hydra.errors import InstantiationException
    from dexmani_policy.training.common.data import build_dataset

    cfg, root, _ = training_data
    del root["data/object_pointcloud"]
    root["data"].create_dataset("object_pointcloud", data=np.zeros(shape, dtype))
    with pytest.raises(InstantiationException if len(shape) == 2 else ValueError, match=error):
        build_dataset(cfg)


@pytest.mark.policy_integration
@pytest.mark.parametrize("metadata", [..., None, {"task_name": "wrong"}])
def test_non_hoi_training_ignores_manifest(training_data, metadata):
    from dexmani_policy.training.common.data import build_dataset

    cfg, root, _ = training_data
    cfg.dataset.sensor_modalities = ["joint_state"]
    if metadata is ...:
        del root.attrs["hoi_pcd"]
    else:
        root.attrs["hoi_pcd"] = metadata
    dataset, description = build_dataset(cfg)
    assert "hoi_pcd" not in description["deployment_metadata"]
    assert dataset[0]["obs"]["joint_state"].shape == (3, 19)


@pytest.mark.policy_integration
@pytest.mark.parametrize("field,value", [
    ("task_name", "other"), ("object_names", ["bowl", "apple"]), ("template_sha256", "0" * 64),
])
def test_resume_rejects_changed_manifest(training_data, field, value):
    from types import SimpleNamespace
    from dexmani_policy.training.common.data import build_dataset
    from dexmani_policy.training.trainer import PolicyTrainer

    cfg, _, manifest = training_data
    dataset, description = build_dataset(cfg)
    checkpoint = SimpleNamespace(effective_config={
        "deployment_metadata": {"hoi_pcd": dict(manifest, **{field: value})},
    })
    with pytest.raises(ValueError, match="Resume data/model mismatch: HOI"):
        PolicyTrainer(cfg, dataset, description, checkpoint=checkpoint)


@pytest.mark.parametrize('failure', ['missing', 'ambiguous', 'invalid', 'program', 'cuda', 'ipc'])
def test_worker_episode_failure_and_retry(task, monkeypatch, caplog, failure):
    import multiprocessing as mp
    from contextlib import contextmanager
    from unittest.mock import Mock
    from types import SimpleNamespace
    from cordvip_geometry.tracking import SceneTracker
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.perception.hoi_worker import run_hoi_worker

    template, manifest = load_task_template(task)
    channels = SensorChannels(mp.get_context('spawn'), camera=CameraConfig(width=4, height=4), hoi=True)
    camera = CameraSnapshot(None, CameraIntrinsics(width=4, height=4, fx=10, fy=10, ppx=2, ppy=2,
        distortion_model='none', distortion_coeffs=(0, 0, 0, 0, 0)), .001, np.eye(4))
    def request(epoch):
        channels.hoi_request.write(episode_id=epoch, after_ns=10, after_sequence=channels.camera_ring.latest_sequence)
        channels.camera_ring.write(timestamp_ns=20 * epoch, rgb=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.uint16))
    request(1)
    tracker = SimpleNamespace(reset=Mock())
    def detect(frame, prompts):
        if tracker.reset.call_count == 1:
            if failure in ('program', 'cuda', 'ipc'):
                raise {'program': ValueError, 'cuda': RuntimeError, 'ipc': BrokenPipeError}[failure](failure)
            if failure in ('missing', 'ambiguous'):
                tracker.segmenter = SimpleNamespace(predict=lambda *a: [dict(label=next(iter(prompts.values())), mask=np.ones((4, 4), bool))] * (0 if failure == 'missing' else 2))
                return SceneTracker.detect_masks(tracker, frame, prompts)
        return {}
    tracker.detect_masks = Mock(side_effect=detect)
    tracker.step = Mock(side_effect=lambda *a, **k: {'objects': {
        name: dict(pose_valid=tracker.reset.call_count > 1, pose=np.eye(4), error='gate failed')
        for name in manifest['object_names']}})
    @contextmanager
    def opened(*a, **k):
        yield tracker
    monkeypatch.setattr(SceneTracker, 'open', opened)
    monkeypatch.setattr('signal.signal', lambda *a: None)
    writes = []
    write = channels.hoi_ring.write
    def publish(**payload):
        writes.append(payload)
        write(**payload)
        if payload['valid']:
            channels.is_running.value = False
    monkeypatch.setattr(channels.hoi_ring, 'write', publish)
    sleeps = []
    def idle(seconds):
        sleeps.append(seconds)
        assert len(writes) == 1 and not writes[0]['valid']
        assert tracker.detect_masks.call_count == 1
        if len(sleeps) == 1:
            channels.camera_ring.write(timestamp_ns=30, rgb=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.uint16))
        else:
            request(2)
    monkeypatch.setattr('time.sleep', idle)
    if failure in ('program', 'cuda', 'ipc'):
        with pytest.raises({'program': ValueError, 'cuda': RuntimeError, 'ipc': BrokenPipeError}[failure], match=failure):
            run_hoi_worker(channels, camera, task, template)
        assert not writes
    else:
        run_hoi_worker(channels, camera, task, template)
        assert tracker.reset.call_count == 2
        assert [p['episode_id'] for p in writes] == [1, 2]
        assert [int(p['timestamp_ns']) for p in writes] == [20, 40]
        assert not writes[0]['object_pointcloud'].any()
        assert len([r for r in caplog.records if r.name.endswith('hoi_worker')]) == 1


@pytest.mark.parametrize('failure', ['empty', 'overlap', 'depth', 'shape'])
def test_scene_initialization_error_scope(monkeypatch, failure):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cordvip_geometry.rgbd import RGBDFrame
    from cordvip_geometry.tracking import SceneTracker, SceneInitializationError
    frame = RGBDFrame(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.float32),
                      np.eye(3), 'none', (0, 0, 0, 0, 0))
    tracker = SceneTracker.__new__(SceneTracker)
    tracker.maps = None
    tracker.initial_selections = None
    tracker.assets = {'apple': None}
    tracker.segmenter = SimpleNamespace(initialize=Mock(side_effect=lambda rgb, masks: masks))
    tracker.pixel_stride = 1
    tracker.min_matched_pixels = 1
    tracker.trackers = {'apple': Mock()}
    tracker.lost_objects = {}
    mask = np.ones((8, 8), bool)
    masks = {'apple': mask}
    if failure == 'empty':
        masks['apple'] = np.zeros_like(mask)
    elif failure == 'overlap':
        masks['bowl'] = mask.copy()
    elif failure == 'shape':
        masks['apple'] = np.ones((4, 4), bool)
    with pytest.raises(ValueError) as raised:
        tracker.step(frame, initial_masks=masks)
    assert isinstance(raised.value, SceneInitializationError) == (failure != 'shape')
    tracker.trackers['apple'].register.assert_not_called()


@pytest.fixture
def pose_tracker():
    import torch
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cordvip_geometry.foundationpose import PoseTracker

    tracker = PoseTracker.__new__(PoseTracker)
    pose = torch.eye(4)
    pose[2, 3] = 1
    tracker.estimator = SimpleNamespace(
        pose_last=pose.clone(), mesh=None, mesh_tensors=None, diameter=1, glctx=None,
        refiner=SimpleNamespace(predict=Mock(return_value=(pose.clone(), None))),
        get_tf_to_centered_mesh=lambda: torch.eye(4),
    )

    def register(**kwargs):
        tracker.estimator.pose_last = pose.clone()
        return tracker.estimator.register.return_value

    tracker.estimator.register = Mock(side_effect=register, return_value=pose.numpy().copy())
    return tracker


@pytest.mark.parametrize("operation", ["register", "track"])
@pytest.mark.parametrize("output", ["valid", "nan", "homogeneous", "rotation"])
def test_foundationpose_output_validation(pose_tracker, operation, output):
    from cordvip_geometry.foundationpose import PoseRegistrationError

    estimator = pose_tracker.estimator
    pose = estimator.register.return_value if operation == "register" else estimator.refiner.predict.return_value[0]
    if output == "nan":
        pose[0, 3] = float("nan")
    elif output == "homogeneous":
        pose[3, 0] = 1
    elif output == "rotation":
        pose[0, 0] = 2
    mask = np.ones((4, 4), bool)
    if operation == "register":
        call = lambda: pose_tracker.register(None, None, np.eye(3), mask)
    else:
        call = lambda: pose_tracker.track(np.eye(3), mask, prepared_frame={})
    if output == "valid":
        np.testing.assert_array_equal(call(), pose)
    else:
        with pytest.raises(PoseRegistrationError, match="invalid FoundationPose output") as raised:
            call()
        assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize("operation", ["register", "track"])
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, BrokenPipeError])
def test_foundationpose_estimator_errors_propagate(pose_tracker, operation, error_type):
    error = error_type("estimator failed")
    estimator = pose_tracker.estimator
    method = estimator.register if operation == "register" else estimator.refiner.predict
    method.side_effect = error
    with pytest.raises(error_type) as raised:
        if operation == "register":
            pose_tracker.register(None, None, np.eye(3), np.ones((4, 4), bool))
        else:
            pose_tracker.track(np.eye(3), np.ones((4, 4), bool), prepared_frame={})
    assert raised.value is error


@pytest.mark.parametrize("operation", ["register", "track"])
def test_scene_marks_invalid_foundationpose_output_lost(pose_tracker, monkeypatch, operation):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cordvip_geometry import tracking
    from cordvip_geometry.rgbd import RGBDFrame

    mask = np.ones((4, 4), bool)
    frame = RGBDFrame(np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.float32),
                     np.eye(3), "none", (0, 0, 0, 0, 0))
    segmenter = SimpleNamespace(initialize=lambda rgb, masks: masks, track=lambda rgb: {"apple": mask})
    pose_tracker.mesh = None
    renderer = SimpleNamespace(render=Mock(return_value=np.ones(16)))
    monkeypatch.setattr(tracking, "PoseTracker", lambda *args, **kwargs: pose_tracker)
    monkeypatch.setattr(tracking, "ModelDepthRenderer", lambda *args, **kwargs: renderer)
    monkeypatch.setattr(tracking, "prepare_tracking_frame", lambda *args: {})
    scene = tracking.SceneTracker(segmenter, {"apple": (None, None, None)},
                                  foundationpose_root=".", debug_dir=".",
                                  pixel_stride=1, min_matched_pixels=1)
    if operation == "register":
        pose_tracker.estimator.register.return_value[0, 3] = np.nan
        result = scene.step(frame, initial_masks={"apple": mask})
    else:
        assert scene.step(frame, initial_masks={"apple": mask})["objects"]["apple"]["pose_valid"]
        renderer.render.reset_mock()
        pose_tracker.estimator.refiner.predict.return_value[0][0, 0] = 2
        result = scene.step(frame)
    item = result["objects"]["apple"]
    assert not item["pose_valid"] and item["pose"] is None
    assert item["state"] == "POSE_LOST"
    assert item["error"] == scene.lost_objects["apple"]
    assert "invalid FoundationPose output" in item["error"]
    renderer.render.assert_not_called()
