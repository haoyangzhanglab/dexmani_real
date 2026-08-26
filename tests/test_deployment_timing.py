"""Offline deployment checks for run epochs and causal scheduling."""

from __future__ import annotations

import pickle
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.deployment.config import (
    DeploymentConfig,
    PolicyRuntimeConfig,
    resolve_deployment_config,
)
from dexmani_real.deployment.contracts import InferenceContext
from dexmani_real.deployment.coordinator import _adoptable, _plan_deadline_ns
from dexmani_real.deployment.manifest import DeploymentManifest
from dexmani_real.deployment.observation import PointCloudFrame, freeze_array
from dexmani_real.deployment.worker import (
    _build_observation,
    _select_pointcloud_control_grid,
)
from dexmani_real.integrations.dexmani_policy import (
    _expected_normalizer_dims,
    _validate_training_data_contract,
)
from dexmani_real.ipc.channels import RuntimeChannelsConfig
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    POLICY_PLAN_DTYPE,
    make_pointcloud_frame_dtype,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    begin_motion,
    read_run_epoch,
    revoke_motion,
)


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value


class _Ring:
    def __init__(self, records: list[tuple[np.ndarray, int, int]]) -> None:
        self.records = records
        self.maxlen = max(8, len(records))

    def get_last_k(self, count: int):
        return self.records[-count:]


def _shared_epoch() -> SimpleNamespace:
    return SimpleNamespace(
        safety_state=_Value(int(SafetyState.ARMED)),
        run_generation=_Value(4),
        run_started_monotonic_ns=_Value(0),
        active_coupled_command_sequence=_Value(9),
        motion_lock=threading.RLock(),
        is_running=_Value(1),
        error_state=_Value(0),
        estop_request=_Value(0),
    )


def _pointcloud(sequence: int, source_ns: int) -> PointCloudFrame:
    return PointCloudFrame(
        values=np.zeros((1, 6), dtype=np.float32),
        source_camera_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=source_ns + 1,
        camera_generation=1,
    )


def _plan() -> np.void:
    frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
    frame["plan_id"][0] = 1
    frame["run_generation"][0] = 5
    frame["observation_id"][0] = 2
    frame["observation_latest_source_monotonic_ns"][0] = 1_000_000_000
    frame["observation_logical_step_monotonic_ns"][0] = 1_050_000_000
    frame["observation_anchor_monotonic_ns"][0] = 1_060_000_000
    frame["inference_started_monotonic_ns"][0] = 1_070_000_000
    frame["inference_finished_monotonic_ns"][0] = 1_080_000_000
    frame["num_steps"][0] = 3
    frame["arm_present"][0] = 1
    frame["hand_present"][0] = 1
    frame["target_monotonic_ns"][0, :3] = (
        1_050_000_000,
        1_200_000_000,
        1_300_000_000,
    )
    frame["valid_mask"][0, :3] = 1
    return frame[0]


class DeploymentTimingTest(unittest.TestCase):
    def test_deployment_wrapper_rejects_ignored_sibling_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "sibling"):
            resolve_deployment_config(
                data={
                    "deployment": {"runtime_target": "tests:fake"},
                    "task_nmae": "typo",
                }
            )

    def test_observation_freshness_allows_the_required_history_span(self) -> None:
        run_ns = 1_000_000_000
        step_ns = 62_500_000
        anchor_ns = run_ns + 260_000_000
        cloud_records: list[tuple[np.ndarray, int, int]] = []
        arm_records: list[tuple[np.ndarray, int, int]] = []
        point_dtype = make_pointcloud_frame_dtype(1024)
        for index in range(1, 5):
            source_ns = run_ns + index * step_ns - 10_000_000
            cloud = np.zeros(1, dtype=point_dtype)
            cloud["source_camera_sequence"][0] = index
            cloud["source_monotonic_ns"][0] = source_ns
            cloud["camera_publish_monotonic_ns"][0] = source_ns + 1
            cloud["publish_monotonic_ns"][0] = source_ns + 2
            cloud["camera_generation"][0] = 1
            cloud_records.append((cloud, source_ns + 3, index))

            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            arm["source_monotonic_ns"][0] = source_ns - 1
            arm["publish_monotonic_ns"][0] = source_ns
            arm["state_valid"][0] = 1
            arm_records.append((arm, source_ns, index))

        hand_records: list[tuple[np.ndarray, int, int]] = []
        offsets_ns = range(40_000_000, 251_000_000, 10_000_000)
        for index, offset_ns in enumerate(offsets_ns, 1):
            source_ns = run_ns + offset_ns
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            hand["source_monotonic_ns"][0] = source_ns
            hand["publish_monotonic_ns"][0] = source_ns + 1
            hand["state_valid"][0] = 1
            hand_records.append((hand, source_ns + 1, index))

        shared = SimpleNamespace(
            arm_state_ring=_Ring(arm_records),
            hand_state_ring=_Ring(hand_records),
            hand_tactile_ring=_Ring([]),
            pointcloud_ring=_Ring(cloud_records),
        )
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=4,
            max_input_age_s=0.15,
            hand_enabled=True,
            observation_fields="arm_qpos,hand_qpos,hand_current,point_cloud",
        )
        observation = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=run_ns,
            anchor_ns=anchor_ns,
            step_dt_ns=step_ns,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(len(observation.pointcloud_history), 4)
        assert observation.hand_current_history is not None
        self.assertEqual(observation.hand_current_history.values.shape[0], 4)

    def test_state_history_budget_includes_grid_lag_and_cross_modal_skew(self) -> None:
        run_ns = 1_000_000_000
        anchor_ns = run_ns + 350_000_000
        point_dtype = make_pointcloud_frame_dtype(1024)
        cloud_records: list[tuple[np.ndarray, int, int]] = []
        arm_records: list[tuple[np.ndarray, int, int]] = []
        hand_records: list[tuple[np.ndarray, int, int]] = []
        for sequence, cloud_source_ns in enumerate(
            (run_ns + 120_000_000, run_ns + 220_000_000), start=1
        ):
            cloud = np.zeros(1, dtype=point_dtype)
            cloud["source_camera_sequence"][0] = sequence
            cloud["source_monotonic_ns"][0] = cloud_source_ns
            cloud["camera_publish_monotonic_ns"][0] = cloud_source_ns + 1
            cloud["publish_monotonic_ns"][0] = cloud_source_ns + 2
            cloud["camera_generation"][0] = 1
            cloud_records.append((cloud, cloud_source_ns + 3, sequence))

            state_source_ns = cloud_source_ns - 90_000_000
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            arm["source_monotonic_ns"][0] = state_source_ns
            arm["publish_monotonic_ns"][0] = state_source_ns + 1
            arm["state_valid"][0] = 1
            arm_records.append((arm, state_source_ns + 1, sequence))
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            hand["source_monotonic_ns"][0] = state_source_ns
            hand["publish_monotonic_ns"][0] = state_source_ns + 1
            hand["state_valid"][0] = 1
            hand_records.append((hand, state_source_ns + 1, sequence))

        shared = SimpleNamespace(
            arm_state_ring=_Ring(arm_records),
            hand_state_ring=_Ring(hand_records),
            hand_tactile_ring=_Ring([]),
            pointcloud_ring=_Ring(cloud_records),
        )
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=2,
            max_input_age_s=0.15,
            max_grid_lag_s=0.08,
            max_observation_skew_s=0.10,
            hand_enabled=True,
            observation_fields="arm_qpos,hand_qpos,point_cloud",
        )

        observation = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=run_ns,
            anchor_ns=anchor_ns,
            step_dt_ns=100_000_000,
        )

        self.assertIsNotNone(observation)

    def test_sensor_ring_capacity_covers_the_complete_freshness_budget(self) -> None:
        runtime = resolve_runtime_config()
        config = RuntimeChannelsConfig.from_runtime(
            runtime,
            pointcloud_requested=True,
            observation_horizon=4,
            observation_dt_s=0.0625,
            max_input_age_s=0.15,
            max_observation_skew_s=0.10,
            max_grid_lag_s=0.08,
        )
        required_span_s = 3 * 0.0625 + 0.15 + 0.10 + 0.08
        pointcloud_span_s = 3 * 0.0625 + 0.15 + 0.08
        self.assertGreaterEqual(
            config.hand_state_ring_maxlen,
            int(np.ceil(runtime.hand.loop_hz * required_span_s)) + 2,
        )
        self.assertEqual(
            config.hand_tactile_ring_maxlen,
            config.hand_state_ring_maxlen,
        )
        self.assertGreaterEqual(
            config.pointcloud_ring_maxlen,
            int(np.ceil(runtime.camera.fps * pointcloud_span_s)) + 2,
        )

    def test_policy_runtime_config_and_training_data_contract_roundtrip(self) -> None:
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                hand_enabled=True,
                task_name="pick",
            ),
            control_dt_s=0.0625,
            point_cloud_frame="xarm_base",
            point_cloud_color_source="aligned_rgb",
            point_cloud_policy_id="policy-v1",
            point_cloud_config_sha256="a" * 64,
            point_cloud_table_plane_abcd_json="null",
            point_cloud_sampling="sample-v1",
            point_cloud_transform="transform-v1",
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.runtime_target, "tests:fake")
        contract = {
            "domain": "real",
            "schema_name": "dexmani-real-policy-zarr",
            "schema_version": 4,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "profile": "pointcloud",
            "task_name": "pick",
            "dt": 0.0625,
            "sensor_modalities": ["joint_state", "point_cloud"],
            "point_cloud_frame": config.point_cloud_frame,
            "point_cloud_color_source": config.point_cloud_color_source,
            "point_cloud_policy_id": config.point_cloud_policy_id,
            "point_cloud_config_sha256": config.point_cloud_config_sha256,
            "point_cloud_table_plane_abcd_json": (
                config.point_cloud_table_plane_abcd_json
            ),
            "point_cloud_sampling": config.point_cloud_sampling,
            "point_cloud_transform": config.point_cloud_transform,
            "point_cloud_num_points": 1024,
            "point_cloud_feature_dim": 6,
        }
        _validate_training_data_contract(contract, restored)
        contract["domain"] = "sim"
        with self.assertRaises(ValueError):
            _validate_training_data_contract(contract, restored)

    def test_faas_normalizer_dimensions_are_model_space_not_native_space(self) -> None:
        manifest = DeploymentManifest(
            action_dim=39,
            use_faas=True,
            hand_dim=12,
            tcp_dim=7,
        )
        agent = SimpleNamespace(faas_mapper=SimpleNamespace(MAPPED_JOINT_DIM=32))

        self.assertEqual(
            _expected_normalizer_dims(manifest, agent),
            {"action": 39, "joint_state": 39, "point_cloud": 6},
        )

    def test_integer_wire_arrays_reject_lossy_casts(self) -> None:
        with self.assertRaises(TypeError):
            freeze_array(np.asarray([1.9]), name="timestamp", dtype=np.uint64)
        with self.assertRaises(ValueError):
            freeze_array(
                np.asarray([-1], dtype=np.int64), name="timestamp", dtype=np.uint64
            )

    def test_run_epoch_is_atomic_and_revocation_clears_it(self) -> None:
        shared = _shared_epoch()

        self.assertTrue(begin_motion(shared))
        epoch = read_run_epoch(shared)
        self.assertEqual(epoch.generation, 5)
        self.assertGreater(epoch.started_monotonic_ns, 0)
        self.assertEqual(shared.active_coupled_command_sequence.value, 0)

        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        revoked = read_run_epoch(shared)
        self.assertEqual(revoked.generation, 6)
        self.assertEqual(revoked.started_monotonic_ns, 0)

    def test_pointcloud_history_uses_policy_grid_not_producer_cadence(self) -> None:
        run_ns = 1_000_000_000
        frames = tuple(
            _pointcloud(sequence, source_ns)
            for sequence, source_ns in enumerate(
                (1_080_000_000, 1_180_000_000, 1_290_000_000, 1_390_000_000),
                start=1,
            )
        )

        selected, logical_ns = _select_pointcloud_control_grid(
            frames,
            run_started_ns=run_ns,
            anchor_ns=1_390_000_000,
            history_len=3,
            step_dt_ns=100_000_000,
            max_grid_lag_ns=30_000_000,
        )

        self.assertEqual(logical_ns, 1_300_000_000)
        self.assertEqual(
            [frame.source_camera_sequence for frame in selected], [1, 2, 3]
        )

    def test_pointcloud_grid_never_reuses_one_source_frame(self) -> None:
        frames = tuple(
            _pointcloud(sequence, source_ns)
            for sequence, source_ns in enumerate(
                (1_090_000_000, 1_290_000_000, 1_390_000_000), start=1
            )
        )

        selected, logical_ns = _select_pointcloud_control_grid(
            frames,
            run_started_ns=1_000_000_000,
            anchor_ns=1_390_000_000,
            history_len=3,
            step_dt_ns=100_000_000,
            max_grid_lag_ns=120_000_000,
        )

        self.assertEqual(selected, ())
        self.assertEqual(logical_ns, 0)

    def test_adoption_skips_expired_prefix_without_retiming(self) -> None:
        ok, reason, first_future = _adoptable(
            _plan(),
            current_generation=5,
            last_observation_id=1,
            now_ns=1_100_000_000,
            max_plan_age_ns=500_000_000,
            max_source_to_command_age_ns=400_000_000,
            command_lead_ns=10_000_000,
        )

        self.assertTrue(ok, reason)
        self.assertEqual(first_future, 1)

    def test_plan_deadline_rejects_noncausal_inference_timestamps(self) -> None:
        plan = _plan()
        plan["inference_started_monotonic_ns"] = 1_040_000_000

        deadline = _plan_deadline_ns(
            plan,
            max_plan_age_ns=500_000_000,
            max_source_to_command_age_ns=400_000_000,
        )

        self.assertIsNone(deadline)

    def test_inference_context_rejects_start_before_causal_cut(self) -> None:
        with self.assertRaisesRegex(ValueError, "causal cut"):
            InferenceContext(
                run_generation=5,
                observation_id=1,
                observation_anchor_monotonic_ns=1_100,
                observation_latest_source_monotonic_ns=1_000,
                observation_logical_step_monotonic_ns=1_050,
                inference_started_monotonic_ns=1_090,
                inference_finished_monotonic_ns=1_200,
                step_dt_ns=50,
            )


if __name__ == "__main__":
    unittest.main()
