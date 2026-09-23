"""Offline deployment regressions: python -m dexmani_real.deployment.smoke_test."""

import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.deployment.observation import build_policy_observation
from dexmani_real.planning.kinematics.ik import make_online_ik_config
from dexmani_real.robot.model import ROBOT_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


def observation_spec(*names):
    shapes = {"joint_state": (19,), "rgb": (None, None, 3)}
    return SimpleNamespace(
        observation_fields=tuple(
            SimpleNamespace(
                name=name, shape=shapes[name], dtype="uint8" if name == "rgb" else "float32"
            )
            for name in names
        )
    )


def row(image):
    return SimpleNamespace(
        arm={"qpos": np.zeros((1, 7), dtype=np.float64)},
        hand={"qpos": np.zeros((1, 12), dtype=np.float64)},
        camera={"rgb": image},
    )


class DeploymentSmoke(unittest.TestCase):
    def test_joint_order(self):
        self.assertEqual(
            ROBOT_JOINT_NAMES, (*tuple(f"joint{i}" for i in range(1, 8)), *XHAND_SDK_JOINT_NAMES)
        )

    def test_raw_rgb(self):
        spec = observation_spec("joint_state", "rgb")
        for h, w in ((12, 16), (24, 32)):
            image = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
            result = build_policy_observation([row(image), row(image)], spec)
            np.testing.assert_array_equal(result["rgb"], np.stack([image, image]))
        for image in (np.zeros((12, 16, 3), np.float32), np.zeros((3, 12, 16), np.uint8)):
            with self.assertRaises(ValueError):
                build_policy_observation([row(image)], spec)
        with self.assertRaises(ValueError):
            build_policy_observation(
                [row(np.zeros((12, 16, 3), np.uint8)), row(np.zeros((24, 32, 3), np.uint8))], spec
            )

    def test_online_ik_profile(self):
        runtime = resolve_experiment_config()
        dt = 1.0 / runtime.teleop.control_hz
        profile = make_online_ik_config(runtime, control_dt_s=dt)
        self.assertEqual(
            profile.nullspace_step_size_deg, runtime.policy.ik_nullspace_step_rate_deg_s * dt
        )
        self.assertEqual(profile, make_online_ik_config(runtime, control_dt_s=dt))


class TablePlaneSmoke(unittest.TestCase):
    def test_default_export_records_current_plane(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import zarr

        with tempfile.TemporaryDirectory() as directory:
            plane_path = Path(directory) / "plane.json"
            plane_path.write_text(json.dumps(dict(a=0, b=0, c=1, d=-0.1)))
            runtime = resolve_experiment_config(
                data={"environment": {"table": {"enabled": False, "plane_path": str(plane_path)}}}
            )
            with patch(
                "dexmani_real.dataset.export.resolve_experiment_config", return_value=runtime
            ):
                target = export_fixture(directory)
            root = zarr.open_group(str(target), mode="r")
            self.assertEqual(
                json.loads(root.attrs["point_cloud_table_plane_abcd_json"]), [0.0, 0.0, 1.0, -0.1]
            )

    def test_unused_plane_file_is_not_required(self):
        import tempfile
        from pathlib import Path

        from dexmani_real.config.pointcloud import PointCloudConfig
        from dexmani_real.dataset.contracts import ProcessingConfig
        from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig

        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing-plane.json")
            for runtime_removal in (False, True):
                runtime = resolve_experiment_config(
                    data={
                        "environment": {"table": {"enabled": False, "plane_path": missing}},
                        "pointcloud": {"remove_table": runtime_removal},
                    }
                )
                disabled = PointCloudConfig(remove_table=False)
                self.assertIsNone(
                    PointCloudLoopConfig.from_runtime(runtime, pointcloud=disabled).table_plane_abcd
                )
                self.assertIsNone(
                    ProcessingConfig.from_runtime(runtime, pointcloud=disabled).table_plane_abcd
                )
                required = PointCloudConfig(remove_table=True)
                with self.assertRaisesRegex(ValueError, "failed to load calibrated table plane"):
                    PointCloudLoopConfig.from_runtime(runtime, pointcloud=required)
                with self.assertRaisesRegex(ValueError, "failed to load calibrated table plane"):
                    ProcessingConfig.from_runtime(runtime, pointcloud=required)
            with self.assertRaisesRegex(ValueError, "failed to load calibrated table plane"):
                resolve_experiment_config(
                    data={"environment": {"table": {"enabled": True, "plane_path": missing}}}
                )

    def test_current_plane_and_invalid_required_geometry(self):
        import json
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        from dexmani_real.config.pointcloud import PointCloudConfig
        from dexmani_real.dataset.contracts import ProcessingConfig
        from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plane.json"
            runtime = resolve_experiment_config(
                data={"environment": {"table": {"enabled": False, "plane_path": str(path)}}}
            )
            required = PointCloudConfig()
            for height in (0.1, 0.2):
                path.write_text(json.dumps(dict(a=0, b=0, c=1, d=-height)))
                expected = (0.0, 0.0, 1.0, -height)
                self.assertEqual(
                    PointCloudLoopConfig.from_runtime(
                        runtime, pointcloud=required
                    ).table_plane_abcd,
                    expected,
                )
                self.assertEqual(ProcessingConfig.from_runtime(runtime).table_plane_abcd, expected)
                collision = resolve_experiment_config(
                    data={"environment": {"table": {"enabled": True, "plane_path": str(path)}}}
                )
                self.assertEqual(collision.environment.table.plane_abcd, expected)
            for contents in (
                "not json",
                "{}",
                '{"a":0,"b":0,"c":-1,"d":0}',
                '{"a":0,"b":0,"c":1,"d":NaN}',
            ):
                path.write_text(contents)
                with self.subTest(contents=contents):
                    with self.assertRaisesRegex(ValueError, "table plane"):
                        PointCloudLoopConfig.from_runtime(runtime, pointcloud=required)
                    with self.assertRaisesRegex(ValueError, "table plane"):
                        ProcessingConfig.from_runtime(runtime)
                    with self.assertRaisesRegex(ValueError, "table plane"):
                        resolve_experiment_config(
                            data={"environment": {"table": {"plane_path": str(path)}}}
                        )
                    self.assertIsNone(
                        PointCloudLoopConfig.from_runtime(
                            runtime, pointcloud=replace(required, remove_table=False)
                        ).table_plane_abcd
                    )
            inline = replace(
                runtime,
                environment=replace(
                    runtime.environment,
                    table=replace(
                        runtime.environment.table, plane_path=None, plane_abcd=(0, 0, 1, -0.3)
                    ),
                ),
            )
            self.assertEqual(
                PointCloudLoopConfig.from_runtime(inline, pointcloud=required).table_plane_abcd,
                (0.0, 0.0, 1.0, -0.3),
            )
            with self.assertRaisesRegex(ValueError, "requires a current calibrated table plane"):
                ProcessingConfig()


class RawFixture:
    """In-memory recording reader; numerical export is the production path."""

    def __init__(self, frames=4):
        from dexmani_real.recording.storage.schema import DATASET_SPECS

        self.runtime = resolve_experiment_config()
        self.images = np.random.default_rng(2).integers(0, 256, (frames, 32, 40, 3), dtype=np.uint8)
        meta = {
            "num_frames": frames,
            "provenance_workflow": "teleop",
            "task_label": "smoke",
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "depth_scale": 0.001,
            "camera_type": "eye_to_hand",
            "camera_T_color_from_depth": np.eye(4),
            "camera_T_xarm_base_from_color": np.eye(4),
        }
        meta["camera_T_xarm_base_from_color"][0, 3] = 0.4
        for stream in ("color", "depth"):
            prefix = f"camera_{stream}"
            meta.update(
                {
                    f"{prefix}_width": 40,
                    f"{prefix}_height": 32,
                    f"{prefix}_intrinsics": np.array([[100.0, 0, 20], [0, 100, 16], [0, 0, 1]]),
                    f"{prefix}_distortion_model": "none",
                    f"{prefix}_distortion_coeffs": [0.0] * 5,
                }
            )
        self.h5f = {
            name: np.zeros((frames, *spec.tail_shape), spec.dtype)
            for name, spec in DATASET_SPECS.items()
        }
        self.h5f["meta"] = SimpleNamespace(attrs=meta)
        self.h5f["depth"] = np.full((frames, 32, 40), 500, np.uint16)
        for name in DATASET_SPECS:
            if name.endswith("timestamp_ns"):
                self.h5f[name][:] = 1_000_000_000 + np.arange(frames, dtype=np.uint64) * 33_333_333
        self.h5f["timestamp"][:] = np.arange(frames) / 30
        self.h5f["hand_contact_valid"][:] = True
        self.h5f["hand_tactile_force_valid"][:] = True
        self.h5f["arm_qpos"][:] = self.runtime.arm.home_qpos
        self.h5f["hand_qpos"][:] = np.deg2rad(self.runtime.hand.home_qpos_deg)
        self.h5f["action_arm_joint_target"][:] = self.h5f["arm_qpos"]
        self.h5f["action_hand_joint_target"][:] = self.h5f["hand_qpos"]
        self.timing = SimpleNamespace(grid_dt_s=1 / 30)

    def require_valid(self, **kwargs):
        pass

    def iter_camera_frames(self, key):
        return iter(self.images)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def export_fixture(directory, frames=4):
    from pathlib import Path
    from unittest.mock import patch

    from dexmani_real.dataset.export import export_raw_to_zarr

    raw = Path(directory) / "raw"
    raw.mkdir()
    (raw / "data.h5").touch()
    target = Path(directory) / "training.zarr"
    reader = RawFixture(frames)
    with patch("dexmani_real.dataset.export.EpisodeReader", return_value=reader):
        export_raw_to_zarr(raw, target)
    return target


class NumericalDeploymentSmoke(unittest.TestCase):
    def test_export_and_geometry(self):
        import tempfile

        import zarr
        from dexmani_policy.deployment import ObservationFieldSpec, PolicySpec

        from dexmani_real.deployment.config import (
            FingertipAssemblerConfig,
            validate_policy_runtime_compatibility,
        )
        from dexmani_real.deployment.observation import build_fingertip_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = zarr.open_group(str(export_fixture(directory)), mode="r")
            self.assertEqual(tuple(root.attrs["joint_names"]), ROBOT_JOINT_NAMES)
            self.assertIn("pointcloud_config_json", root.attrs)
            self.assertNotIn("processing_config_json", root.attrs)
            raw = RawFixture()
            fields = tuple(
                ObservationFieldSpec(
                    name,
                    shape,
                    "float32",
                    {"fingers": ("thumb", "index", "middle", "ring", "pinky")}
                    if name == "fingertip_points"
                    else {},
                )
                for name, shape in (
                    ("joint_state", (19,)),
                    ("eef_pose", (9,)),
                    ("fingertip_points", (5, 3)),
                )
            )
            spec = PolicySpec(fields, 4, 1, 1 / 30, "joint", ROBOT_JOINT_NAMES)
            fk = build_fingertip_runtime(spec, FingertipAssemblerConfig.from_runtime(raw.runtime))
            rows = [
                SimpleNamespace(
                    arm={"qpos": raw.h5f["arm_qpos"][i : i + 1]},
                    hand={"qpos": raw.h5f["hand_qpos"][i : i + 1]},
                )
                for i in range(4)
            ]
            online = build_policy_observation(rows, spec, fingertip_runtime=fk)
            for name in ("eef_pose", "fingertip_points"):
                np.testing.assert_allclose(online[name], root["data"][name][:], atol=1e-6)
            from dataclasses import replace

            current = replace(
                raw.runtime,
                hand=replace(raw.runtime.hand, T_eef_handbase_pos_xyz=(0.05, 0.02, 0.10)),
            )
            validate_policy_runtime_compatibility(spec, current)
            changed = FingertipAssemblerConfig.from_runtime(current)
            moved = build_policy_observation(
                rows, spec, fingertip_runtime=build_fingertip_runtime(spec, changed)
            )
            self.assertFalse(np.allclose(moved["fingertip_points"], online["fingertip_points"]))

    def test_pointcloud_and_current_calibration(self):
        from dataclasses import replace

        from dexmani_policy.deployment import ObservationFieldSpec, PolicySpec

        from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
        from dexmani_real.config.pointcloud import PointCloudConfig
        from dexmani_real.dataset.pointcloud import (
            RawEpisodePointCloudDeriver,
            load_raw_episode_base_from_color,
            load_raw_episode_camera_model,
        )
        from dexmani_real.deployment.config import validate_policy_runtime_compatibility
        from dexmani_real.sensor.pointcloud import build_point_cloud
        from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig, _load_static_inputs

        raw = RawFixture()
        trained = PointCloudConfig(voxel_size_m=0.006)
        spec = PolicySpec(
            (
                ObservationFieldSpec("joint_state", (19,), "float32"),
                ObservationFieldSpec(
                    "point_cloud",
                    (1024, 6),
                    "float32",
                    {"features": ("x", "y", "z", "r", "g", "b")},
                ),
            ),
            2,
            1,
            1 / 30,
            "joint",
            ROBOT_JOINT_NAMES,
            trained.to_dict(),
        )
        runtime = replace(raw.runtime, pointcloud=replace(trained, voxel_size_m=0.011))
        validate_policy_runtime_compatibility(spec, runtime)
        with self.assertRaisesRegex(ValueError, "joint_names"):
            validate_policy_runtime_compatibility(
                replace(spec, joint_names=tuple(reversed(ROBOT_JOINT_NAMES))), runtime
            )
        camera = load_raw_episode_camera_model(raw)
        transform = load_raw_episode_base_from_color(raw)
        for plane in ((0.0, 0.0, 1.0, -0.1), (0.0, 0.0, 1.0, -0.2)):
            runtime = replace(
                runtime,
                environment=replace(
                    runtime.environment,
                    table=replace(
                        runtime.environment.table, enabled=False, plane_path=None, plane_abcd=plane
                    ),
                ),
            )
            cfg = PointCloudLoopConfig.from_runtime(
                runtime, pointcloud=PointCloudConfig.from_dict(spec.pointcloud_config)
            )
            self.assertEqual(cfg.pointcloud.voxel_size_m, 0.006)
            self.assertEqual(cfg.table_plane_abcd, plane)
            kwargs = dict(
                depth_raw=raw.h5f["depth"][0],
                color=raw.images[0],
                depth_scale_m=camera.depth_scale_m,
                geometry=camera.geometry,
                T_xarm_base_from_color=transform,
                table_plane_abcd=plane,
                config=trained,
            )
            online = build_point_cloud(**kwargs)
            offline = RawEpisodePointCloudDeriver(raw, camera, transform, trained, plane).derive(
                0, raw.images[0]
            )
            self.assertIsNotNone(online)
            np.testing.assert_array_equal(online, offline)
            moved = transform.copy()
            moved[0, 3] += 0.03
            changed = build_point_cloud(**{**kwargs, "T_xarm_base_from_color": moved})
            self.assertFalse(np.allclose(changed, online))
        with self.assertRaisesRegex(ValueError, "table plane"):
            PointCloudLoopConfig(trained, CameraExtrinsics(), None)
        disabled = replace(trained, remove_table=False)
        self.assertIsNone(
            PointCloudLoopConfig(disabled, CameraExtrinsics(), (0, 0, 1, -0.6)).table_plane_abcd
        )
        a = build_point_cloud(**{**kwargs, "config": disabled, "table_plane_abcd": None})
        b = build_point_cloud(**{**kwargs, "config": disabled, "table_plane_abcd": (0, 0, 1, -0.6)})
        np.testing.assert_array_equal(a, b)
        with self.assertRaisesRegex(ValueError, "table plane"):
            build_point_cloud(**{**kwargs, "table_plane_abcd": None})
        # Camera serial selects current extrinsics; intrinsics and scale come from the camera worker.
        import json
        import tempfile
        from pathlib import Path

        for serial, shift, fx, scale in (
            ("new-A", 0.2, 110.0, 0.001),
            ("new-B", 0.3, 120.0, 0.002),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "cameras.json"
                path.write_text(
                    json.dumps(
                        {
                            "camera_0": {
                                "serial": serial,
                                "type": "eye_to_hand",
                                "pose": {"position": [shift, 0, 0], "orientation": [1, 0, 0, 0]},
                            }
                        }
                    )
                )
                calibration = CameraExtrinsics(path)
                native = replace(
                    camera.geometry,
                    depth=replace(camera.geometry.depth, fx=fx),
                    color=replace(camera.geometry.color, fx=fx),
                )
                shared = SimpleNamespace(
                    is_running=SimpleNamespace(value=True),
                    camera_ready=SimpleNamespace(is_set=lambda: True),
                    camera_geometry=SimpleNamespace(value=json.dumps(native.to_dict()).encode()),
                    camera_depth_scale=SimpleNamespace(value=scale),
                    camera_serial=SimpleNamespace(value=serial.encode()),
                )
                geometry, actual_scale, actual_transform = _load_static_inputs(shared, calibration)
                self.assertEqual(geometry.color.fx, fx)
                self.assertEqual(actual_scale, scale)
                self.assertAlmostEqual(actual_transform[0, 3], shift)
                validate_policy_runtime_compatibility(spec, runtime)

    def test_actions(self):
        from dexmani_real.deployment.action import decode_policy_action, make_action_planner
        from dexmani_real.planning.kinematics.arm_fk import make_arm_fk

        runtime = resolve_experiment_config()
        kwargs = dict(
            current_arm_qpos=runtime.arm.home_qpos,
            previous_arm_command_qpos=None,
            workspace=runtime.policy.workspace.as_array(),
            hand_qpos_min_rad=runtime.hand.qpos_min_rad,
            hand_qpos_max_rad=runtime.hand.qpos_max_rad,
        )
        arm = np.asarray(runtime.arm.home_qpos)
        hand = np.deg2rad(runtime.hand.home_qpos_deg)
        result = decode_policy_action(np.r_[arm, hand], "joint", planner=None, **kwargs)
        np.testing.assert_array_equal(result[0], arm)
        np.testing.assert_array_equal(result[1], hand)
        planner = make_action_planner("eef", runtime, control_dt_s=1 / 30)
        self.assertEqual(
            planner.teleop_profile, make_online_ik_config(runtime, control_dt_s=1 / 30)
        )
        position, rotation = make_arm_fk().compute(arm)
        result = decode_policy_action(
            np.r_[position, rotation, hand], "eef", planner=planner, **kwargs
        )
        self.assertIsNotNone(result[0])
        actual_p, actual_r = make_arm_fk().compute(result[0])
        np.testing.assert_allclose(
            actual_p,
            np.clip(position, kwargs["workspace"][:, 0], kwargs["workspace"][:, 1]),
            atol=0.008,
        )
        np.testing.assert_allclose(actual_r, rotation, atol=0.08)
        from unittest.mock import Mock

        failed = Mock()
        failed.solve_teleop_ik.return_value = SimpleNamespace(success=False, qpos=None)
        extreme_hand = hand + 100
        result = decode_policy_action(
            np.r_[position, rotation, extreme_hand], "eef", planner=failed, **kwargs
        )
        self.assertIsNone(result[0])
        np.testing.assert_array_equal(failed.set_hand_qpos.call_args.args[0], result[1])
        for bad in (np.zeros(18), np.full(19, np.nan)):
            with self.assertRaises(ValueError):
                decode_policy_action(bad, "joint", planner=None, **kwargs)

    def test_whole_episode_rejection(self):
        from dexmani_real.dataset.processing import validate_episode

        for mutation in (
            "flag_frame_status",
            "hand_contact_valid",
            "hand_tactile_force_valid",
            "action_timestamp_ns",
        ):
            reader = RawFixture()
            if mutation == "flag_frame_status":
                reader.h5f[mutation][2] = 1
            else:
                reader.h5f[mutation][2] = 0
            with self.assertRaises(ValueError):
                validate_episode(reader)


class RuntimeBoundarySmoke(unittest.TestCase):
    def test_freshness_source_identity_and_tactile(self):
        import time
        from unittest.mock import Mock

        from dexmani_policy.deployment import ObservationFieldSpec, PolicySpec

        from dexmani_real.ipc.schema import (
            ARM_STATE_DTYPE,
            CAMERA_FRAME_HEADER_DTYPE,
            HAND_STATE_DTYPE,
            make_pointcloud_frame_dtype,
        )
        from dexmani_real.runtime.observation import ObservationHistory, read_observation

        now = time.monotonic_ns()
        runtime = resolve_experiment_config()
        arm = np.zeros(1, ARM_STATE_DTYPE)
        arm["timestamp_ns"] = now
        hand = np.zeros(1, HAND_STATE_DTYPE)
        hand["timestamp_ns"] = now
        cloud = np.zeros(1, make_pointcloud_frame_dtype(1024))
        cloud["timestamp_ns"] = now
        cloud["source_camera_sequence"] = 17
        header = np.zeros(1, CAMERA_FRAME_HEADER_DTYPE)
        header["timestamp_ns"] = now
        shared = SimpleNamespace(
            arm_state_ring=Mock(),
            hand_state_ring=Mock(),
            pointcloud_ring=Mock(),
            camera_ring=Mock(),
        )
        shared.arm_state_ring.read_latest.return_value = (arm, 0, 1)
        shared.hand_state_ring.read_latest.return_value = (hand, 0, 1)
        shared.pointcloud_ring.read_latest.return_value = (cloud, 0, 1)
        shared.camera_ring.read_sequence.return_value = {
            "header": header,
            "rgb": np.zeros((12, 16, 3), np.uint8),
            "depth": np.zeros((12, 16), np.uint16),
        }
        both = read_observation(
            shared, runtime, require_pointcloud=True, require_rgb_cloud_identity=True
        )
        self.assertIsNotNone(both)
        shared.camera_ring.read_sequence.assert_called_once_with(17)
        shared.camera_ring.read_sequence.return_value = None
        self.assertIsNone(
            read_observation(
                shared, runtime, require_pointcloud=True, require_rgb_cloud_identity=True
            )
        )
        shared.camera_ring.reset_mock()
        only_cloud = read_observation(shared, runtime, require_pointcloud=True)
        self.assertIsNotNone(only_cloud)
        shared.camera_ring.read_sequence.assert_not_called()
        shared.camera_ring.read_latest.assert_not_called()
        spec = PolicySpec(
            (
                ObservationFieldSpec("joint_state", (19,), "float32"),
                ObservationFieldSpec("contact_force", (5, 3), "float32"),
            ),
            1,
            1,
            1 / 30,
            "joint",
            ROBOT_JOINT_NAMES,
        )
        self.assertIsNone(build_policy_observation([only_cloud], spec))
        stale = cloud.copy()
        stale["timestamp_ns"] = 1
        shared.pointcloud_ring.read_latest.return_value = (stale, 0, 2)
        self.assertIsNone(read_observation(shared, runtime, require_pointcloud=True))
        history = ObservationHistory(2, 1 / 30)
        history.append(only_cloud)
        from dataclasses import replace

        newer = replace(
            only_cloud, observation_timestamp_ns=only_cloud.observation_timestamp_ns + 1_000_000_000
        )
        history.append(newer)
        self.assertEqual(len(history.rows), 1)
        self.assertIs(history.padded()[0], newer)

    def test_delayed_prediction_loses_motion_authority(self):
        import time
        from unittest.mock import Mock, patch

        from dexmani_policy.deployment import ObservationFieldSpec, PolicySpec

        from dexmani_real.deployment.runner import PolicyRunner
        from dexmani_real.runtime.safety import SafetyState

        spec = PolicySpec(
            (ObservationFieldSpec("joint_state", (19,), "float32"),),
            1,
            1,
            1 / 30,
            "joint",
            ROBOT_JOINT_NAMES,
        )
        for fence in ("run_id", "estop_request"):
            shared = SimpleNamespace(
                **{
                    name: SimpleNamespace(value=value)
                    for name, value in {
                        "is_running": True,
                        "error_state": False,
                        "estop_request": False,
                        "run_id": 1,
                        "safety_state": int(SafetyState.RUNNING),
                        "workflow_failed": False,
                        "quit_requested": False,
                    }.items()
                }
            )
            model = Mock()

            def delayed_prediction(obs):
                getattr(shared, fence).value = 2 if fence == "run_id" else True
                return np.zeros((1, 19))

            model.predict.side_effect = delayed_prediction
            runner = PolicyRunner(
                shared,
                resolve_experiment_config(),
                spec,
                model_runtime=model,
                fingertip_runtime=None,
                execute=True,
                max_running_s=None,
            )
            runner.run_id = 1
            sample = row(np.zeros((12, 16, 3), np.uint8))
            sample.observation_timestamp_ns = time.monotonic_ns()
            runner._row = Mock(return_value=sample)
            with patch("dexmani_real.deployment.runner.publish_command") as publish:
                runner.step()
                publish.assert_not_called()
            self.assertFalse(runner.actions)

    def test_warmup_before_ready(self):
        import threading
        from unittest.mock import Mock, patch

        from dexmani_real.deployment.runner import policy_runner_loop

        spec = observation_spec("joint_state")
        config = SimpleNamespace(
            experiment="fixture",
            device="cpu",
            seed=0,
            artifact="artifact.pt",
            inference_steps=1,
            spec=spec,
        )
        for fail in (False, True):
            shared = SimpleNamespace(
                policy_ready=threading.Event(), workflow_failed=SimpleNamespace(value=False)
            )
            model = Mock()
            model.spec = spec

            def warmup(**kwargs):
                self.assertFalse(shared.policy_ready.is_set())
                if fail:
                    raise RuntimeError("intentional warmup failure")

            model.warmup.side_effect = warmup
            with (
                patch("dexmani_policy.deployment.load_experiment", return_value=model),
                patch("dexmani_real.deployment.runner.build_fingertip_runtime", return_value=None),
                patch("dexmani_real.deployment.runner.PolicyRunner") as runner,
            ):
                if fail:
                    with self.assertRaises(RuntimeError):
                        policy_runner_loop(shared, None, config, False)
                    runner.assert_not_called()
                    self.assertFalse(shared.policy_ready.is_set())
                    self.assertTrue(shared.workflow_failed.value)
                else:
                    policy_runner_loop(shared, None, config, False)
                    self.assertTrue(shared.policy_ready.is_set())
                model.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
