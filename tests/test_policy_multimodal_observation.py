"""Focused offline tests for the Real multimodal Policy boundary."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import (
    FingertipAssemblerConfig,
    PolicyWorkerConfig,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.lifecycle import (
    _compute_policy_observation_ring_capacities,
    build_policy_worker_specs,
)
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
    PolicyObservation,
    RgbFrame,
)
from dexmani_real.deployment.worker import (
    _align_state_history_to_reference_ns,
    _read_tactile_provenance_history,
    _select_control_grid_reference_ns,
    _to_policy_observation,
)
from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime
from dexmani_real.ipc.channels import RuntimeChannelsConfig
from dexmani_real.ipc.schema import HAND_TACTILE_DTYPE
from dexmani_real.planning.fingertip import compute_fingertip_points_xarm_base


def _field(name: str, runtime) -> SimpleNamespace:
    values = {
        "joint_state": ((19,), "float32"),
        "point_cloud": ((1024, 6), "float32"),
        "rgb": (
            (runtime.camera.height, runtime.camera.width, 3),
            "uint8",
        ),
        "contact_force": ((5, 3), "float32"),
        "fingertip_points": ((5, 3), "float32"),
    }
    shape, dtype = values[name]
    return SimpleNamespace(name=name, shape=shape, dtype=dtype)


def _policy_spec(
    fields: tuple[str, ...], runtime, *, n_obs_steps: int = 2
) -> SimpleNamespace:
    return SimpleNamespace(
        action_key="action",
        action_dim=19,
        control_action_dim=19,
        horizon=16,
        n_obs_steps=n_obs_steps,
        n_action_steps=8,
        control_dt_s=1.0 / runtime.policy.control_hz,
        requires_hand=True,
        observation_fields=tuple(_field(name, runtime) for name in fields),
    )


def _policy_channels_config(
    runtime,
    policy_spec: SimpleNamespace,
    *,
    camera_requested: bool = False,
    pointcloud_requested: bool = False,
) -> RuntimeChannelsConfig:
    base = RuntimeChannelsConfig.from_runtime(
        runtime,
        camera_requested=camera_requested,
        pointcloud_requested=pointcloud_requested,
    )
    return replace(
        base,
        **_compute_policy_observation_ring_capacities(runtime, policy_spec, base),
    )


def _window(values: np.ndarray, sources=(90, 190)) -> FrameWindow:
    return FrameWindow(
        values=values,
        source_sequence=np.array([1, 2], dtype=np.uint64),
        source_monotonic_ns=np.array(sources, dtype=np.uint64),
        publish_monotonic_ns=np.array([95, 195], dtype=np.uint64),
        valid_mask=np.ones(2, dtype=np.uint8),
    )


class _ArmFk:
    def __init__(self):
        self.calls = 0

    def compute(self, _qpos):
        self.calls += 1
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class _HandFk:
    def is_ready(self):
        return True

    def compute_tip_positions_in_handbase(self, _qpos):
        return np.arange(15, dtype=np.float64).reshape(5, 3) / 100.0


class MultimodalContractTest(unittest.TestCase):
    def test_adapter_rejects_rgb_shape_drift(self) -> None:
        runtime = resolve_runtime_config()
        spec = _policy_spec(("joint_state", "rgb"), runtime)

        class Loaded:
            info = object()

            def __init__(self):
                self.spec = spec

            def warmup(self, *, samples):
                return (0.0,) * samples

            def reset_episode(self):
                pass

            def predict(self, _observation):
                return np.zeros((spec.n_action_steps, 19), dtype=np.float64)

            def close(self):
                pass

        adapter = DexManiPolicyRuntime(Loaded(), spec)
        observation = PolicyObservation(
            observation_id=1,
            run_generation=1,
            anchor_monotonic_ns=30,
            latest_source_monotonic_ns=10,
            logical_step_monotonic_ns=20,
            arrays={
                "joint_state": np.zeros((2, 19), dtype=np.float32),
                "rgb": np.zeros((2, 1, 1, 3), dtype=np.uint8),
            },
        )
        with self.assertRaises(ValueError):
            adapter.predict(observation)

    def test_fingertip_eef_pose_reuse_matches_without_second_arm_fk(self) -> None:
        arm_fk = _ArmFk()
        kwargs = {
            "arm_fk": arm_fk,
            "hand_fk": _HandFk(),
            "handbase_position_eef_m": np.array([1.0, 0.0, 0.0]),
            "handbase_quat_eef_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        direct = compute_fingertip_points_xarm_base(np.zeros(7), np.zeros(12), **kwargs)
        self.assertEqual(arm_fk.calls, 1)
        reused = compute_fingertip_points_xarm_base(
            np.zeros(7),
            np.zeros(12),
            **kwargs,
            eef_position_xarm_base_m=np.zeros(3),
            eef_rot6d_xarm_base=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        self.assertEqual(arm_fk.calls, 1)
        np.testing.assert_allclose(reused, direct)

    def test_contact_requires_fresh_calibrated_tactile_proof(self) -> None:
        class Ring:
            maxlen = 2

            def __init__(self, frame):
                self.frame = frame

            def get_last_k(self, _count):
                return [(self.frame, 110, 1)]

        frame = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
        frame["source_monotonic_ns"][0] = 100
        frame["fresh"][0] = 1
        self.assertIsNone(
            _read_tactile_provenance_history(
                Ring(frame),
                anchor_ns=120,
                history_len=2,
                max_age_ns=100,
                not_before_ns=50,
            )
        )
        frame["calibrated"][0] = 1
        proof = _read_tactile_provenance_history(
            Ring(frame),
            anchor_ns=120,
            history_len=2,
            max_age_ns=100,
            not_before_ns=50,
        )
        self.assertIsNotNone(proof)
        np.testing.assert_array_equal(proof.source_monotonic_ns, [100])

    def test_seven_supported_modality_matrices_and_workers(self) -> None:
        runtime = resolve_runtime_config()
        matrices = (
            ("joint_state",),
            ("joint_state", "point_cloud"),
            ("joint_state", "rgb"),
            ("joint_state", "contact_force"),
            ("joint_state", "fingertip_points"),
            ("joint_state", "point_cloud", "rgb"),
            ("joint_state", "point_cloud", "rgb", "contact_force", "fingertip_points"),
        )
        for modalities in matrices:
            with self.subTest(modalities=modalities):
                spec = _policy_spec(modalities, runtime)
                validate_policy_runtime_compatibility(spec, runtime)
                workers = build_policy_worker_specs(
                    object(),
                    runtime,
                    spec,
                    PolicyWorkerConfig(
                        experiment="fake/model", device="cpu", spec=spec
                    ),
                    execute=False,
                )
                names = {worker.name for worker in workers}
                self.assertEqual(
                    "camera" in names, bool(set(modalities) & {"rgb", "point_cloud"})
                )
                self.assertEqual("pointcloud" in names, "point_cloud" in modalities)
                self.assertIn("hand", names)

    def test_field_projection_checks_shape_and_dtype(self) -> None:
        runtime = resolve_runtime_config()
        for attribute, value in (
            ("shape", (18,)),
            ("dtype", "float64"),
        ):
            with self.subTest(attribute=attribute):
                spec = _policy_spec(("joint_state",), runtime)
                setattr(spec.observation_fields[0], attribute, value)
                with self.assertRaises(ValueError):
                    validate_policy_runtime_compatibility(spec, runtime)

    def test_policy_observation_exact_arrays_are_owned_and_read_only(self) -> None:
        source = np.zeros((2, 19), dtype=np.float32)
        observation = PolicyObservation(
            observation_id=1,
            run_generation=2,
            anchor_monotonic_ns=30,
            latest_source_monotonic_ns=10,
            logical_step_monotonic_ns=20,
            arrays={"joint_state": source},
        )
        source[0, 0] = 1.0
        self.assertEqual(observation.arrays["joint_state"][0, 0], 0.0)
        self.assertTrue(observation.arrays["joint_state"].flags.c_contiguous)
        self.assertFalse(observation.arrays["joint_state"].flags.writeable)
        with self.assertRaises(ValueError):
            PolicyObservation(
                observation_id=1,
                run_generation=1,
                anchor_monotonic_ns=30,
                latest_source_monotonic_ns=10,
                logical_step_monotonic_ns=20,
                arrays={"joint_state": source, "unknown": source},
            )

    def test_nonvisual_grid_is_causal_and_skew_bounded(self) -> None:
        references, logical = _select_control_grid_reference_ns(
            run_started_ns=100, anchor_ns=355, history_len=3, step_dt_ns=100
        )
        np.testing.assert_array_equal(references, [100, 200, 300])
        self.assertEqual(logical, 300)
        history = FrameWindow(
            values=np.arange(12, dtype=np.float64).reshape(4, 3),
            source_sequence=np.arange(1, 5, dtype=np.uint64),
            source_monotonic_ns=np.array([90, 180, 205, 295], dtype=np.uint64),
            publish_monotonic_ns=np.array([95, 185, 210, 300], dtype=np.uint64),
            valid_mask=np.ones(4, dtype=np.uint8),
        )
        aligned = _align_state_history_to_reference_ns(
            history, references, max_skew_ns=20
        )
        self.assertIsNotNone(aligned)
        np.testing.assert_array_equal(aligned.source_monotonic_ns, [90, 180, 295])
        self.assertIsNone(
            _align_state_history_to_reference_ns(history, references, max_skew_ns=5)
        )

    def test_exact_multimodal_projection_and_fingertip_frame(self) -> None:
        runtime = resolve_runtime_config()
        modalities = (
            "joint_state",
            "point_cloud",
            "rgb",
            "contact_force",
            "fingertip_points",
        )
        spec = _policy_spec(modalities, runtime)
        clouds = tuple(
            PointCloudFrame(np.zeros((1024, 6), np.float32), i, s, p, 1)
            for i, s, p in ((1, 100, 110), (2, 200, 210))
        )
        rgbs = tuple(
            RgbFrame(
                np.zeros((runtime.camera.height, runtime.camera.width, 3), np.uint8),
                i,
                s,
                p,
                1,
            )
            for i, s, p in ((1, 100, 110), (2, 200, 210))
        )
        batch = ObservationBatch(
            observation_id=1,
            run_generation=1,
            run_started_monotonic_ns=50,
            anchor_monotonic_ns=220,
            latest_source_monotonic_ns=200,
            logical_step_monotonic_ns=200,
            arm_history=_window(np.zeros((2, 7))),
            hand_history=_window(np.zeros((2, 12))),
            hand_tactile_sum_history=_window(np.ones((2, 5, 3))),
            hand_tactile_provenance_history=_window(np.zeros((2, 1), dtype=np.uint8)),
            pointcloud=clouds[-1],
            pointcloud_history=clouds,
            rgb_history=rgbs,
        )
        config = FingertipAssemblerConfig(
            hand_urdf_path="fake.urdf",
            fingertip_link_names=("a", "b", "c", "d", "e"),
            handbase_position_eef_m=(1.0, 0.0, 0.0),
            handbase_quat_eef_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        result = _to_policy_observation(
            batch, spec, fingertip_runtime=(_ArmFk(), _HandFk(), config)
        )
        self.assertEqual(tuple(result.arrays), modalities)
        self.assertEqual(result.arrays["joint_state"].shape, (2, 19))
        self.assertEqual(result.arrays["contact_force"].shape, (2, 5, 3))
        np.testing.assert_allclose(
            result.arrays["fingertip_points"][0, 0], [1.0, 0.01, 0.02]
        )
        with self.assertRaises(RuntimeError):
            _to_policy_observation(batch, spec, fingertip_runtime=None)

        mismatched_rgb = (
            rgbs[0],
            RgbFrame(rgbs[1].values, 2, 199, 210, 1),
        )
        with self.assertRaises(ValueError):
            ObservationBatch(
                observation_id=1,
                run_generation=1,
                run_started_monotonic_ns=50,
                anchor_monotonic_ns=220,
                latest_source_monotonic_ns=200,
                logical_step_monotonic_ns=200,
                arm_history=_window(np.zeros((2, 7))),
                hand_history=_window(np.zeros((2, 12))),
                pointcloud=clouds[-1],
                pointcloud_history=clouds,
                rgb_history=mismatched_rgb,
            )

    def test_policy_observation_ring_capacities(self) -> None:
        runtime = resolve_runtime_config()
        large_windows_runtime = replace(
            runtime,
            policy=replace(
                runtime.policy,
                max_input_age_s=0.8,
                max_observation_skew_s=0.4,
                max_grid_lag_s=0.3,
            ),
        )
        cases = (
            (runtime, 1, False, False, (10, 10, 10, 5, 8)),
            (runtime, 8, True, False, (26, 26, 26, 23, 8)),
            (runtime, 8, True, True, (26, 26, 26, 23, 23)),
            (large_windows_runtime, 8, True, True, (61, 61, 61, 49, 49)),
        )
        for (
            case_runtime,
            n_obs_steps,
            camera_requested,
            pointcloud_requested,
            expected,
        ) in cases:
            with self.subTest(expected=expected):
                config = _policy_channels_config(
                    case_runtime,
                    _policy_spec((), case_runtime, n_obs_steps=n_obs_steps),
                    camera_requested=camera_requested,
                    pointcloud_requested=pointcloud_requested,
                )
                self.assertEqual(
                    (
                        config.arm_state_ring_maxlen,
                        config.hand_state_ring_maxlen,
                        config.hand_tactile_ring_maxlen,
                        config.camera_ring_maxlen,
                        config.pointcloud_ring_maxlen,
                    ),
                    expected,
                )

    def test_pointcloud_request_requires_explicit_camera_request(self) -> None:
        runtime = resolve_runtime_config()
        with self.assertRaisesRegex(ValueError, "requires camera_requested"):
            RuntimeChannelsConfig.from_runtime(
                runtime,
                pointcloud_requested=True,
            )


if __name__ == "__main__":
    unittest.main()
