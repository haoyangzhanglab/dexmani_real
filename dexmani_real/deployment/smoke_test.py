"""Offline data, deployment and mocked device regressions.

Run with: python -m dexmani_real.deployment.smoke_test
"""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import numpy as np

from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
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
        profile = make_online_ik_config(runtime)
        self.assertEqual(profile.operational_joint_lower_rad, tuple(runtime.arm.joint_limit_lower))
        self.assertEqual(profile.operational_joint_upper_rad, tuple(runtime.arm.joint_limit_upper))
        self.assertFalse(profile.enable_random_fallback)
        self.assertEqual(profile, make_online_ik_config(runtime))


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
        from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig

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
                    PointCloudWorkerConfig.from_runtime(
                        runtime, camera_calibration=CameraExtrinsics(), pointcloud=disabled
                    ).table_plane_abcd
                )
                self.assertIsNone(
                    ProcessingConfig.from_runtime(runtime, pointcloud=disabled).table_plane_abcd
                )
                required = PointCloudConfig(remove_table=True)
                with self.assertRaisesRegex(ValueError, "failed to load calibrated table plane"):
                    PointCloudWorkerConfig.from_runtime(
                        runtime, camera_calibration=CameraExtrinsics(), pointcloud=required
                    )
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
        from dexmani_real.sensor.pointcloud_worker import PointCloudWorkerConfig

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
                    PointCloudWorkerConfig.from_runtime(
                        runtime, camera_calibration=CameraExtrinsics(), pointcloud=required
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
                        PointCloudWorkerConfig.from_runtime(
                            runtime, camera_calibration=CameraExtrinsics(), pointcloud=required
                        )
                    with self.assertRaisesRegex(ValueError, "table plane"):
                        ProcessingConfig.from_runtime(runtime)
                    with self.assertRaisesRegex(ValueError, "table plane"):
                        resolve_experiment_config(
                            data={"environment": {"table": {"plane_path": str(path)}}}
                        )
                    self.assertIsNone(
                        PointCloudWorkerConfig.from_runtime(
                            runtime,
                            camera_calibration=CameraExtrinsics(),
                            pointcloud=replace(required, remove_table=False),
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
                PointCloudWorkerConfig.from_runtime(
                    inline, camera_calibration=CameraExtrinsics(), pointcloud=required
                ).table_plane_abcd,
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
        from dexmani_real.sensor.pointcloud_worker import (
            PointCloudWorkerConfig,
            _load_static_inputs,
        )

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
            cfg = PointCloudWorkerConfig.from_runtime(
                runtime,
                camera_calibration=CameraExtrinsics(),
                pointcloud=PointCloudConfig.from_dict(spec.pointcloud_config),
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
            PointCloudWorkerConfig(trained, CameraExtrinsics(), None)
        disabled = replace(trained, remove_table=False)
        self.assertIsNone(
            PointCloudWorkerConfig(disabled, CameraExtrinsics(), (0, 0, 1, -0.6)).table_plane_abcd
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
        np.testing.assert_array_equal(result.arm_qpos, arm)
        np.testing.assert_array_equal(result.hand_qpos, hand)
        planner = make_action_planner("eef", runtime)
        self.assertEqual(
            planner.online_ik_profile, make_online_ik_config(runtime, enable_random_fallback=True)
        )
        position, rotation = make_arm_fk().compute(arm)
        result = decode_policy_action(
            np.r_[position, rotation, hand], "eef", planner=planner, **kwargs
        )
        self.assertIsNotNone(result.arm_qpos)
        self.assertIs(result.arm_qpos, result.ik_result.qpos)
        actual_p, actual_r = make_arm_fk().compute(result.arm_qpos)
        np.testing.assert_allclose(
            actual_p,
            np.clip(position, kwargs["workspace"][:, 0], kwargs["workspace"][:, 1]),
            atol=0.008,
        )
        np.testing.assert_allclose(actual_r, rotation, atol=0.08)
        from unittest.mock import Mock

        failed = Mock()
        failed.solve_online_ik.return_value = SimpleNamespace(success=False, qpos=None)
        extreme_hand = hand + 100
        result = decode_policy_action(
            np.r_[position, rotation, extreme_hand], "eef", planner=failed, **kwargs
        )
        self.assertIsNone(result.arm_qpos)
        self.assertIs(result.ik_result, failed.solve_online_ik.return_value)
        failed.solve_online_ik.assert_called_once()
        np.testing.assert_array_equal(failed.set_hand_qpos.call_args.args[0], result.hand_qpos)
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
            runner._read_observation = Mock(return_value=sample)
            with patch("dexmani_real.deployment.runner.publish_command") as publish:
                runner.step()
                publish.assert_not_called()
            self.assertFalse(runner.action_queue)

    def test_warmup_before_ready(self):
        import threading
        from unittest.mock import Mock, patch

        from dexmani_real.deployment.runner import run_policy_worker

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
                        run_policy_worker(shared, None, config, False)
                    runner.assert_not_called()
                    self.assertFalse(shared.policy_ready.is_set())
                    self.assertTrue(shared.workflow_failed.value)
                else:
                    run_policy_worker(shared, None, config, False)
                    self.assertTrue(shared.policy_ready.is_set())
                model.close.assert_called_once()


@contextmanager
def fake_sdk_driver(relative_path, module_name, sdk_name, sdk):
    """Load driver code against a fake SDK without importing native device libraries."""
    import importlib.util
    import sys
    from pathlib import Path
    from unittest.mock import patch

    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {sdk_name: sdk, module_name: module}):
        spec.loader.exec_module(module)
        yield module


def motion_fixture(state):
    from queue import Queue
    from threading import Event, RLock
    from unittest.mock import Mock

    return SimpleNamespace(
        **{
            name: SimpleNamespace(value=value)
            for name, value in dict(
                is_running=True,
                error_state=False,
                estop_request=False,
                workflow_failed=False,
                quit_requested=False,
                physical_home_completed=False,
                stop_request=0,
                start_request=False,
                run_id=1,
                run_ended_reason=0,
                safety_state=int(state),
            ).items()
        },
        motion_lock=RLock(),
        hand_home_q=Queue(),
        hand_home_result_q=Queue(),
        hand_state_ring=Mock(),
        hand_ready=Event(),
    )


class XHandSafetySmoke(unittest.TestCase):
    def setUp(self):
        from dexmani_real.config.hardware import HandParams

        stack = ExitStack()
        self.addCleanup(stack.close)
        self.driver = stack.enter_context(
            fake_sdk_driver(
                "robot/drivers/xhand.py",
                "dexmani_real.robot.drivers.xhand",
                "xhand_controller",
                SimpleNamespace(xhand_control=SimpleNamespace()),
            )
        )
        self.cfg = HandParams()
        self.target = np.deg2rad(self.cfg.home_qpos_deg)
        self.status = self.driver.XHandSendStatus

    def test_driver_modes_limits_and_statuses(self):
        from unittest.mock import Mock

        hand = self.driver.XHand(self.cfg)
        hand.connected_flag = True
        hand._command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)])
        hand._control = Mock()
        for code, expected in (
            (0, self.status.ACCEPTED),
            (1_501_070, self.status.CRC_UNCONFIRMED),
            (-1, self.status.REJECTED),
        ):
            hand._control.send_command.return_value = SimpleNamespace(error_code=code)
            self.assertIs(hand.set_passive(), expected)
            self.assertTrue(all(j.mode == 0 for j in hand._command.finger_command))
            self.assertIs(hand.send_action(self.target), expected)
            self.assertTrue(all(j.mode == 3 for j in hand._command.finger_command))
            self.assertIs(hand.hold_current(self.target), expected)
        hand._control.send_command.return_value = SimpleNamespace(error_code=0)
        for bound, offset in (
            (self.cfg.mechanical_qpos_max_rad, 0.001),
            (self.cfg.mechanical_qpos_min_rad, -0.001),
        ):
            measured = np.asarray(bound) + offset
            self.assertIs(hand.send_action(measured), self.status.REJECTED)
            self.assertIs(hand.hold_current(measured), self.status.ACCEPTED)
            np.testing.assert_array_equal([j.position for j in hand._command.finger_command], bound)
        for invalid in (np.zeros(11), np.full(12, np.nan)):
            with self.assertRaises(ValueError):
                hand.hold_current(invalid)

    def test_send_boundary(self):
        from unittest.mock import Mock

        from dexmani_real.robot.hand_worker import _send_target
        from dexmani_real.runtime.safety import SafetyState, revoke_motion

        shared = motion_fixture(SafetyState.RUNNING)
        hand = Mock()
        for status in (self.status.ACCEPTED, self.status.CRC_UNCONFIRMED):
            hand.send_action.return_value = status
            self.assertIs(_send_target(shared, hand, self.target, 1, self.cfg), status)
        hand.send_action.return_value = self.status.REJECTED
        with self.assertRaises(RuntimeError):
            _send_target(shared, hand, self.target, 1, self.cfg)
        revoke_motion(shared)
        hand.send_action.reset_mock()
        self.assertIsNone(_send_target(shared, hand, self.target, 1, self.cfg))
        hand.send_action.assert_not_called()

    def test_worker_revocation(self):
        from unittest.mock import Mock, call, patch

        from dexmani_real.robot.commands import RobotCommand
        from dexmani_real.robot.hand_worker import run_hand_worker
        from dexmani_real.runtime.safety import SafetyState, revoke_motion

        for home in (False, True):
            for scenario in (
                "fresh",
                "replacement",
                "dropout",
                "stale",
                "rejected",
                "estop",
                "read_timeout",
            ):
                with self.subTest(home=home, scenario=scenario):
                    shared = motion_fixture(SafetyState.ARMED if home else SafetyState.RUNNING)
                    clock = [10.0]
                    tick = [0]
                    events = []
                    state = SimpleNamespace(
                        qpos=self.target.copy(),
                        current_ma=np.zeros(12),
                        tactile_aggregate_valid=False,
                        tactile_dense_valid=False,
                        commboard_err=(0,),
                        jointboard_err=(0,),
                        tipboard_err=(0,),
                    )
                    hand = Mock(is_connected=True, tactile_calibrated=False)
                    send_ticks = []

                    def send(target):
                        send_ticks.append(tick[0])
                        return self.status.CRC_UNCONFIRMED

                    hand.send_action.side_effect = send

                    def safety_send(kind):
                        events.append(kind)
                        if scenario == "rejected" and kind == "hold":
                            return self.status.REJECTED
                        return (
                            self.status.CRC_UNCONFIRMED
                            if len(events) == 1
                            else self.status.ACCEPTED
                        )

                    hand.hold_current.side_effect = lambda q: safety_send("hold")
                    hand.set_passive.side_effect = lambda: safety_send("passive")
                    hand.get_state.side_effect = lambda: (
                        None
                        if tick[0] and scenario in ("dropout", "stale", "read_timeout")
                        else state
                    )
                    command = RobotCommand(1, hand_qpos=self.target)
                    if home:
                        shared.hand_home_q.put((self.target, 1, 20_000_000_000))

                    def wait():
                        tick[0] += 1
                        clock[0] += 0.2 if scenario in ("stale", "read_timeout") else 0.01
                        if tick[0] == 1:
                            if scenario == "estop":
                                shared.estop_request.value = True
                            elif home:
                                # Same epoch, changed state must revoke ARMED HOME authority.
                                shared.safety_state.value = int(SafetyState.RUNNING)
                            else:
                                revoke_motion(shared)
                                if scenario == "replacement":
                                    shared.hand_home_q.put(
                                        (self.target, shared.run_id.value, 20_000_000_000)
                                    )
                        if tick[0] == (8 if scenario == "read_timeout" else 4):
                            shared.is_running.value = False

                    with (
                        patch.object(self.driver, "XHand", return_value=hand),
                        patch("dexmani_real.robot.hand_worker.LoopRate") as rate,
                        patch(
                            "dexmani_real.robot.hand_worker.read_robot_command",
                            return_value=(command, 1)
                            if not home or scenario == "replacement"
                            else None,
                        ),
                        patch("time.monotonic", side_effect=lambda: clock[0]),
                        patch("time.monotonic_ns", side_effect=lambda: int(clock[0] * 1e9)),
                    ):
                        rate.return_value.wait.side_effect = wait
                        if scenario in ("rejected", "read_timeout"):
                            with self.assertRaisesRegex(
                                RuntimeError, "revoke failed|feedback timed out"
                            ):
                                run_hand_worker(shared, self.cfg)
                            self.assertTrue(shared.error_state.value)
                        else:
                            run_hand_worker(shared, self.cfg)
                            self.assertFalse(shared.error_state.value)
                    self.assertEqual(send_ticks, [0, 3] if scenario == "replacement" else [0])
                    if home:
                        self.assertTrue(shared.hand_home_result_q.get_nowait()[1].ok)
                    if scenario in ("fresh", "replacement", "dropout"):
                        self.assertEqual(events, ["hold", "hold", "passive"])
                        np.testing.assert_array_equal(
                            hand.hold_current.call_args.args[0], self.target
                        )
                    elif scenario in ("stale", "read_timeout"):
                        self.assertEqual(events, ["passive", "passive", "passive"])
                    elif scenario == "estop":
                        self.assertEqual(events, ["passive"])
                        self.assertEqual(
                            hand.mock_calls[-2:], [call.set_passive(), call.disconnect()]
                        )
                    else:
                        self.assertEqual(events, ["hold", "passive"])
                    hand.disconnect.assert_called_once()

    def test_passive_cleanup_does_not_mask_fault(self):
        from unittest.mock import Mock, patch

        from dexmani_real.robot.hand_worker import run_hand_worker
        from dexmani_real.runtime.safety import SafetyState

        shared = motion_fixture(SafetyState.ARMED)
        hand = Mock()
        hand.get_state.side_effect = RuntimeError("original read failure")
        hand.set_passive.side_effect = RuntimeError("cleanup failure")
        with patch.object(self.driver, "XHand", return_value=hand):
            with self.assertRaisesRegex(RuntimeError, "original read failure"):
                run_hand_worker(shared, self.cfg)
        hand.disconnect.assert_called_once()

    def test_crc_home_converges(self):
        from unittest.mock import Mock, patch

        from dexmani_real.ipc.schema import HAND_STATE_DTYPE
        from dexmani_real.robot.hand_homing import home_hand
        from dexmani_real.robot.hand_worker import _send_target
        from dexmani_real.robot.home import HomeResult
        from dexmani_real.runtime.safety import SafetyState

        shared = motion_fixture(SafetyState.ARMED)
        clock = [10.0]
        hand = Mock()
        hand.send_action.return_value = self.status.CRC_UNCONFIRMED

        def submit(request):
            target, epoch, _ = request
            status = _send_target(shared, hand, target, epoch, self.cfg, home=True)
            shared.hand_home_result_q.put((epoch, HomeResult(status is not None)))

        def sample():
            clock[0] += 0.01
            frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
            frame["qpos"] = self.target
            frame["timestamp_ns"] = int(clock[0] * 1e9)
            return frame, 0, 1

        shared.hand_home_q = Mock()
        shared.hand_home_q.put_nowait.side_effect = submit
        shared.hand_state_ring.read_latest.side_effect = sample
        with (
            patch("time.monotonic", side_effect=lambda: clock[0]),
            patch("time.monotonic_ns", side_effect=lambda: int(clock[0] * 1e9)),
            patch("time.sleep"),
        ):
            result = home_hand(shared, resolve_experiment_config())
        self.assertTrue(result.ok)
        self.assertEqual(shared.hand_state_ring.read_latest.call_count, 3)
        hand.send_action.assert_called_once()


class HandHomeSmoke(unittest.TestCase):
    def test_convergence_and_total_budget(self):
        from unittest.mock import patch

        from dexmani_real.ipc.schema import HAND_STATE_DTYPE
        from dexmani_real.robot.hand_homing import home_hand
        from dexmani_real.robot.home import HomeResult
        from dexmani_real.runtime.safety import SafetyState, revoke_motion

        runtime = resolve_experiment_config()
        target = np.deg2rad(runtime.hand.home_qpos_deg)
        for scenario in (
            "converged",
            "reset",
            "duplicate",
            "stale",
            "nan",
            "timeout",
            "abort",
            "authority",
            "estop",
            "error",
            "shutdown",
            "budget",
        ):
            with self.subTest(scenario=scenario):
                shared = motion_fixture(SafetyState.ARMED)
                clock = [10.0]
                reads = [0]
                first_stamp = [None]

                def submitted(*args):
                    self.assertLessEqual(args[3], runtime.hand.home_timeout_s)
                    if scenario == "budget":
                        clock[0] += runtime.hand.home_timeout_s - 0.01
                    return HomeResult(True)

                def sample():
                    reads[0] += 1
                    clock[0] += 0.001
                    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
                    frame["qpos"] = target
                    stamp = int(clock[0] * 1e9)
                    first_stamp[0] = first_stamp[0] or stamp
                    if scenario == "duplicate":
                        stamp = first_stamp[0]
                    elif scenario == "stale":
                        stamp -= 1_000_000_000
                    elif scenario == "nan":
                        frame["qpos"][0, 0] = np.nan
                    elif scenario == "timeout" or (scenario == "reset" and reads[0] == 3):
                        frame["qpos"][0, 0] += np.deg2rad(5.1)
                    else:
                        frame["qpos"][0, 0] += np.deg2rad(4.9)
                    frame["timestamp_ns"] = stamp
                    if scenario == "authority":
                        revoke_motion(shared)
                    elif scenario in ("estop", "error", "shutdown"):
                        field = {
                            "estop": "estop_request",
                            "error": "error_state",
                            "shutdown": "is_running",
                        }[scenario]
                        getattr(shared, field).value = scenario != "shutdown"
                    return frame, 0, reads[0]

                shared.hand_state_ring.read_latest.side_effect = sample
                with (
                    patch("dexmani_real.robot.hand_homing.wait_home_result", side_effect=submitted),
                    patch("time.monotonic_ns", side_effect=lambda: int(clock[0] * 1e9)),
                    patch(
                        "time.sleep",
                        side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay),
                    ),
                ):
                    result = home_hand(
                        shared,
                        runtime,
                        abort_requested=lambda: scenario == "abort" and reads[0] > 0,
                    )
                self.assertEqual(result.ok, scenario in ("converged", "reset"))
                if result.ok:
                    self.assertEqual(reads[0], 6 if scenario == "reset" else 3)
                else:
                    self.assertEqual(shared.run_id.value, 3)
                self.assertLess(clock[0], 10.0 + runtime.hand.home_timeout_s + 0.02)

    def test_tolerance_validation(self):
        from dataclasses import replace

        from dexmani_real.config.hardware import HandParams

        cfg = HandParams()
        self.assertEqual(cfg.home_tolerance_deg, 5.0)
        self.assertEqual(cfg.home_timeout_s, 2.0)
        for invalid in (0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "home_tolerance_deg"):
                replace(cfg, home_tolerance_deg=invalid).validate()


class TimeoutAndCameraSmoke(unittest.TestCase):
    def test_runner_timeout_before_and_after_prediction(self):
        from unittest.mock import Mock, patch

        from dexmani_real.deployment.runner import PolicyRunner
        from dexmani_real.runtime.safety import RunEndReason, SafetyState, begin_requested_motion

        spec = observation_spec("joint_state")
        spec.n_obs_steps, spec.n_action_steps, spec.control_dt_s, spec.action_mode = (
            1,
            1,
            1 / 30,
            "joint",
        )
        for during_prediction in (False, True):
            shared = motion_fixture(SafetyState.ARMED)
            clock = [10_000_000_000]
            with patch("time.monotonic_ns", side_effect=lambda: clock[0]):
                shared.start_request.value = True
                epoch, started = begin_requested_motion(shared)
                runner = PolicyRunner(
                    shared,
                    resolve_experiment_config(),
                    spec,
                    model_runtime=Mock(),
                    fingertip_runtime=None,
                    execute=True,
                    max_running_s=1.0,
                )
                runner.run_id, runner.started_ns = epoch, started
                sample = row(None)
                sample.observation_timestamp_ns = clock[0]
                runner._read_observation = Mock(return_value=sample)
                runner.recorder = Mock()

                def predict(obs):
                    clock[0] += 1_000_000_000
                    return np.zeros((1, 19))

                runner.model.predict.side_effect = predict
                if not during_prediction:
                    clock[0] += 1_000_000_000
                with patch("dexmani_real.deployment.runner.publish_command") as publish:
                    runner.step()
                    publish.assert_not_called()
                self.assertEqual(shared.run_ended_reason.value, RunEndReason.TIMEOUT)
                self.assertEqual(shared.safety_state.value, SafetyState.ARMED)
                self.assertIsNone(runner.run_id)
                runner.recorder.stop_episode.assert_called_once_with(save=True, reason="timeout")

    def test_camera_frame_return_timestamp(self):
        from unittest.mock import Mock, patch

        from dexmani_real.sensor.camera.worker import pack_camera_frame

        with fake_sdk_driver(
            "sensor/camera/realsense.py", "_smoke_realsense", "pyrealsense2", SimpleNamespace()
        ) as driver:
            camera = driver.RealSenseCamera()
            camera.pipeline, camera.frame_queue, camera.depth_scale = Mock(), Mock(), 0.001
            depth, color = Mock(), Mock()
            depth.get_data.return_value = np.zeros((2, 2), np.uint16)
            color.get_data.return_value = np.zeros((2, 2, 3), np.uint8)
            for frame in (depth, color):
                frame.get_frame_number.return_value = 7
                frame.get_timestamp.return_value = 1000
                frame.get_frame_timestamp_domain.return_value = 0
            frames = camera.frame_queue.wait_for_frame.return_value.as_frameset.return_value
            frames.get_depth_frame.return_value = depth
            frames.get_color_frame.return_value = color
            camera._depth_to_color_aligner = Mock()
            camera._depth_to_color_aligner.process.return_value.get_depth_frame.return_value = depth
            with patch("time.monotonic_ns", side_effect=[100, 200, 300, 400]):
                result = camera.read(compute_depth=False)
            self.assertEqual(result.timestamp_ns, 100)
            self.assertEqual(result.timestamp_ns, result.wait_return_monotonic_ns)
            header, _, _ = pack_camera_frame(
                result.rgb,
                result.depth_aligned_to_color_raw,
                timestamp_ns=result.timestamp_ns,
                depth_frame_number=7,
                color_frame_number=7,
            )
            self.assertEqual(header["timestamp_ns"][0], 100)


if __name__ == "__main__":
    unittest.main()
