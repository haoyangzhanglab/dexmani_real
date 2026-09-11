"""Offline regressions for deployment eef_pose / tactile_force capability.

No hardware, no shared memory, no policy checkpoint.  Fake rings and fake FK
resources pin: EEF derived from the causally aligned arm history with one Arm
FK per timestep, the contact-only fast path never copying the full tactile
tensor, the full-tactile gates and source-identity fail-closed checks, and the
PolicyObservation boundary contract for the new tails.  Run with:

    python -m unittest discover -s tests -p 'test_deployment_eef_tactile.py'
"""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np

import dexmani_real.deployment.inference.observation as observation_mod
from dexmani_real.config.defaults import PolicyParams
from dexmani_real.deployment.config import (
    _SUPPORTED_OBSERVATION_FIELDS,
    _expected_contact_force_semantics,
    _expected_tactile_force_semantics,
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.inference.observation import (
    ObservationBatch,
    _build_observation,
    _to_policy_observation,
    build_fingertip_runtime,
)
from dexmani_real.deployment.lifecycle import _requires_hand_sensor
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_DERIVATION,
    compute_eef_pose_history_xarm_base,
    make_arm_fk,
)
from dexmani_real.planning.kinematics.fingertip import compute_fingertip_history_xarm_base

_T0_NS = 10**15
_DT_NS = 62_500_000  # 16 Hz control grid
_HORIZON = 3
# Anchor sits 10ms past the tick-5 reference so each record's
# publish = source + 1ms stays inside the causal cut.
_ANCHOR_NS = _T0_NS + 5 * _DT_NS + 10**7
_REF_TICKS = (3, 4, 5)
_MOUNT_P = (0.0, 0.0, 0.1)
_MOUNT_Q = (1.0, 0.0, 0.0, 0.0)


def _ref_ns(tick: int) -> int:
    return _T0_NS + tick * _DT_NS


class _Field:
    def __init__(self, name, shape, dtype, semantics=None):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.semantics = semantics if semantics is not None else {}


def _fake_policy_spec(*fields: _Field, n_obs_steps: int = _HORIZON):
    names = tuple(field.name for field in fields)
    if "joint_state" not in names:
        fields = (_Field("joint_state", (19,), "float32"),) + fields
    return types.SimpleNamespace(
        observation_fields=fields,
        n_obs_steps=n_obs_steps,
        n_action_steps=4,
        chunk_size=15,
        requires_hand=True,
        action_key="action",
        control_action_dim=19,
        control_dt_s=_DT_NS / 1e9,
    )


class _FakeRing:
    """Oldest-first ring stub exposing maxlen + get_last_k."""

    def __init__(self, records: list[tuple[np.ndarray, int, int]]):
        self._records = records
        self.maxlen = 8

    def get_last_k(self, k: int):
        return self._records[-int(k):]

    def get_last_k_fields(self, k: int, fields: tuple[str, ...]):
        return [
            ({name: data[0][name].copy() for name in fields}, timestamp, sequence)
            for data, timestamp, sequence in self._records[-int(k):]
        ]


def _arm_record(tick: int, qpos: np.ndarray) -> tuple[np.ndarray, int, int]:
    data = np.zeros(1, dtype=ARM_STATE_DTYPE)
    data["qpos"][0] = qpos
    data["state_valid"][0] = 1
    data["source_monotonic_ns"][0] = _ref_ns(tick)
    data["publish_monotonic_ns"][0] = _ref_ns(tick) + 10**6
    return data, data["publish_monotonic_ns"][0], tick + 1


def _hand_state_record(
    tick: int,
    *,
    aggregate_valid: bool = True,
    dense_valid: bool = True,
) -> tuple[np.ndarray, int, int]:
    data = np.zeros(1, dtype=HAND_STATE_DTYPE)
    data["qpos"][0] = 0.1
    data["tactile_aggregate"][0] = float(tick)
    data["tactile_aggregate_valid"][0] = int(aggregate_valid)
    data["tactile_dense"][0] = float(tick)
    data["tactile_dense_valid"][0] = int(dense_valid)
    data["state_valid"][0] = 1
    data["qpos_stale"][0] = 0
    data["source_monotonic_ns"][0] = _ref_ns(tick)
    data["publish_monotonic_ns"][0] = _ref_ns(tick) + 10**6
    return data, data["publish_monotonic_ns"][0], tick + 1


def _fake_shared(
    *,
    arm_qpos: np.ndarray | None = None,
    aggregate_invalid_ticks: tuple[int, ...] = (),
    dense_invalid_ticks: tuple[int, ...] = (),
) -> types.SimpleNamespace:
    ticks = range(6)
    qpos_history = (
        np.zeros((6, 7)) if arm_qpos is None else np.asarray(arm_qpos)
    )
    return types.SimpleNamespace(
        arm_state_ring=_FakeRing(
            [_arm_record(tick, qpos_history[tick]) for tick in ticks]
        ),
        hand_state_ring=_FakeRing(
            [
                _hand_state_record(
                    tick,
                    aggregate_valid=tick not in aggregate_invalid_ticks,
                    dense_valid=tick not in dense_invalid_ticks,
                )
                for tick in ticks
            ]
        ),
    )


def _build(shared, spec) -> ObservationBatch | None:
    return _build_observation(
        shared,
        PolicyParams(),
        spec,
        observation_id=7,
        run_generation=0,
        run_started_ns=_T0_NS,
        anchor_ns=_ANCHOR_NS,
        step_dt_ns=_DT_NS,
    )


class _CountingFakeArmFK:
    def __init__(self) -> None:
        self.calls = 0

    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.asarray(qpos, dtype=np.float64)
        self.calls += 1
        return qpos[:3].copy(), np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class _FakeHandFK:
    def is_ready(self) -> bool:
        return True

    def compute_tip_positions_in_handbase(self, hand_qpos: np.ndarray) -> np.ndarray:
        hand = np.asarray(hand_qpos, dtype=np.float64)
        base = np.arange(15, dtype=np.float64).reshape(5, 3) * 0.01
        return base + float(hand[0]) * 0.001


def _fake_fingertip_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        handbase_position_eef_m=_MOUNT_P, handbase_quat_eef_wxyz=_MOUNT_Q
    )


def _distinct_arm_qpos() -> np.ndarray:
    qpos = np.zeros((6, 7))
    for tick in range(6):
        qpos[tick] = 0.01 * (tick + 1)
    return qpos


class TestFieldGates(unittest.TestCase):
    def test_supported_fields_include_new_modalities(self) -> None:
        self.assertIn("eef_pose", _SUPPORTED_OBSERVATION_FIELDS)
        self.assertIn("tactile_force", _SUPPORTED_OBSERVATION_FIELDS)

    def test_requires_hand_sensor(self) -> None:
        self.assertTrue(
            _requires_hand_sensor(
                _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
            )
        )
        # eef_pose alone must not pull in the hand sensor set.
        spec = types.SimpleNamespace(
            observation_fields=(_Field("eef_pose", (9,), "float32"),)
        )
        self.assertFalse(_requires_hand_sensor(spec))


class TestBuildObservation(unittest.TestCase):
    def test_joint_state_only_does_not_require_tactile_validity(self) -> None:
        spec = _fake_policy_spec(_Field("eef_pose", (9,), "float32"))
        observation = _build(_fake_shared(dense_invalid_ticks=(5,)), spec)
        self.assertIsNotNone(observation)
        self.assertIsNotNone(observation.hand_history)
        np.testing.assert_array_equal(
            observation.arm_history.source_monotonic_ns,
            np.asarray([_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64),
        )

    def test_contact_only_does_not_require_dense_valid(self) -> None:
        spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        observation = _build(_fake_shared(dense_invalid_ticks=(5,)), spec)
        self.assertIsNotNone(observation)
        self.assertIsNotNone(observation.hand_history)

    def test_single_source_identity(self) -> None:
        spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32"),
            _Field("tactile_force", (5, 120, 3), "float32"),
        )
        observation = _build(_fake_shared(), spec)
        self.assertIsNotNone(observation)
        hand = observation.hand_history
        self.assertIsNotNone(hand)
        self.assertEqual(hand.tactile_dense.shape, (_HORIZON, 5, 120, 3))
        # One hand window carries qpos, aggregate, and dense under a single
        # source axis; the aligned source matches the policy grid, and the
        # aggregate/dense values encode the very tick their source points at.
        np.testing.assert_array_equal(
            hand.source_monotonic_ns,
            np.asarray([_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64),
        )
        for index, tick in enumerate(_REF_TICKS):
            self.assertTrue(np.all(hand.tactile_aggregate[index] == float(tick)))
            self.assertTrue(np.all(hand.tactile_dense[index] == float(tick)))

    def test_contact_only_survives_dense_invalid_but_dense_policy_fails(self) -> None:
        shared = _fake_shared(dense_invalid_ticks=(0, 1, 2, 3, 4, 5))
        contact_spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        self.assertIsNotNone(_build(shared, contact_spec))
        dense_spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        self.assertIsNone(_build(shared, dense_spec))

    def test_dense_requires_dense_valid(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        shared = _fake_shared(dense_invalid_ticks=(4, 5))
        self.assertIsNone(_build(shared, spec))

    def test_aggregate_requires_aggregate_valid(self) -> None:
        spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        shared = _fake_shared(aggregate_invalid_ticks=(4, 5))
        self.assertIsNone(_build(shared, spec))

    def test_force_window_respects_causal_cut(self) -> None:
        shared = _fake_shared()
        # The hand record carries an in-array publish timestamp; a publish after
        # the anchor must fail the causal cut regardless of the ring slot time.
        for data, _ring_publish_ns, _sequence in shared.hand_state_ring._records:
            data["publish_monotonic_ns"][0] = _ANCHOR_NS + 1
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        self.assertIsNone(_build(shared, spec))

    def test_dense_no_fallback_newest_invalid(self) -> None:
        # Newest sample (tick 5) has invalid dense; the older valid sample
        # (tick 4) is only 62.5 ms back, still inside the 0.10 s skew bound.
        # Validity is checked after selection, so the observation must fail
        # rather than fall back to tick 4's dense payload.
        shared = _fake_shared(dense_invalid_ticks=(5,))
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        self.assertIsNone(_build(shared, spec))

    def test_aggregate_no_fallback_newest_invalid(self) -> None:
        shared = _fake_shared(aggregate_invalid_ticks=(5,))
        spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        self.assertIsNone(_build(shared, spec))

    def test_partial_validity(self) -> None:
        # Selected newest sample: aggregate valid, dense invalid.
        shared = _fake_shared(dense_invalid_ticks=(5,))
        joint_spec = _fake_policy_spec(_Field("eef_pose", (9,), "float32"))
        contact_spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        dense_spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        both_spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32"),
            _Field("tactile_force", (5, 120, 3), "float32"),
        )
        self.assertIsNotNone(_build(shared, joint_spec))
        self.assertIsNotNone(_build(shared, contact_spec))
        self.assertIsNone(_build(shared, dense_spec))
        self.assertIsNone(_build(shared, both_spec))


class TestToPolicyObservation(unittest.TestCase):
    def test_geometry_uses_policy_visible_float32_joints(self):
        spec = _fake_policy_spec(
            _Field("eef_pose", (9,), "float32"),
            _Field("fingertip_points", (5, 3), "float32"),
        )
        arm64 = np.random.default_rng(29).uniform(-0.5, 0.5, (6, 7))
        self.assertFalse(np.array_equal(arm64, arm64.astype(np.float32).astype(np.float64)))
        observation = _build(_fake_shared(arm_qpos=arm64), spec)
        arm_fk = mock.Mock(wraps=make_arm_fk())
        hand_fk = mock.Mock(wraps=_FakeHandFK())
        with mock.patch.object(
            observation_mod, "compute_fingertip_history_xarm_base",
            wraps=compute_fingertip_history_xarm_base,
        ) as fingertips:
            result = _to_policy_observation(
                observation, spec,
                fingertip_runtime=(arm_fk, hand_fk, _fake_fingertip_config()),
            )
        joint = result.arrays["joint_state"]
        expected_eef = compute_eef_pose_history_xarm_base(joint[:, :7])
        np.testing.assert_array_equal(result.arrays["eef_pose"], expected_eef.astype(np.float32))
        original_eef = compute_eef_pose_history_xarm_base(observation.arm_history.values)
        self.assertFalse(np.array_equal(original_eef.astype(np.float32), result.arrays["eef_pose"]))
        self.assertEqual(arm_fk.compute.call_count, _HORIZON)
        for index, call in enumerate(arm_fk.compute.call_args_list):
            np.testing.assert_array_equal(call.args[0], joint[index, :7])
        for index, call in enumerate(hand_fk.compute_tip_positions_in_handbase.call_args_list):
            np.testing.assert_array_equal(call.args[0], joint[index, 7:19])
        np.testing.assert_array_equal(fingertips.call_args.kwargs["eef_pose_history"], expected_eef)
        expected_tips = compute_fingertip_history_xarm_base(
            joint[:, :7], joint[:, 7:19], hand_fk=_FakeHandFK(),
            handbase_position_eef_m=np.asarray(_MOUNT_P),
            handbase_quat_eef_wxyz=np.asarray(_MOUNT_Q),
            eef_pose_history=expected_eef,
        )
        np.testing.assert_array_equal(result.arrays["fingertip_points"], expected_tips)

    def _observation(self, spec) -> ObservationBatch:
        observation = _build(_fake_shared(arm_qpos=_distinct_arm_qpos()), spec)
        self.assertIsNotNone(observation)
        return observation

    def test_eef_pose_values_and_single_shared_fk(self) -> None:
        spec = _fake_policy_spec(
            _Field("eef_pose", (9,), "float32"),
            _Field("fingertip_points", (5, 3), "float32"),
        )
        observation = self._observation(spec)
        arm_fk = _CountingFakeArmFK()
        runtime = (arm_fk, _FakeHandFK(), _fake_fingertip_config())
        policy_observation = _to_policy_observation(
            observation, spec, fingertip_runtime=runtime
        )
        eef = policy_observation.arrays["eef_pose"]
        self.assertEqual(eef.shape, (_HORIZON, 9))
        self.assertEqual(eef.dtype, np.float32)
        self.assertTrue(eef.flags.c_contiguous)
        for index, tick in enumerate(_REF_TICKS):
            expected_pos = np.float32(0.01 * (tick + 1))
            np.testing.assert_allclose(eef[index, :3], expected_pos, rtol=1e-6)
            np.testing.assert_allclose(
                eef[index, 3:], np.float32([1, 0, 0, 0, 1, 0]), rtol=0.0, atol=0.0
            )
        self.assertEqual(arm_fk.calls, _HORIZON)
        fingertips = policy_observation.arrays["fingertip_points"]
        self.assertEqual(fingertips.shape, (_HORIZON, 5, 3))
        self.assertEqual(fingertips.dtype, np.float32)

    def test_fingertip_only_still_runs_one_fk_per_step(self) -> None:
        spec = _fake_policy_spec(_Field("fingertip_points", (5, 3), "float32"))
        observation = self._observation(spec)
        arm_fk = _CountingFakeArmFK()
        runtime = (arm_fk, _FakeHandFK(), _fake_fingertip_config())
        policy_observation = _to_policy_observation(
            observation, spec, fingertip_runtime=runtime
        )
        self.assertEqual(arm_fk.calls, _HORIZON)
        self.assertNotIn("eef_pose", policy_observation.arrays)

    def test_tactile_force_array_contract(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        observation = self._observation(spec)
        policy_observation = _to_policy_observation(observation, spec)
        tactile = policy_observation.arrays["tactile_force"]
        self.assertEqual(tactile.shape, (_HORIZON, 5, 120, 3))
        self.assertEqual(tactile.dtype, np.float32)
        self.assertTrue(tactile.flags.c_contiguous)
        for index, tick in enumerate(_REF_TICKS):
            self.assertTrue(np.all(tactile[index] == np.float32(float(tick))))

    def test_to_policy_observation_requires_hand_history(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        batch = ObservationBatch(
            observation_id=1,
            run_generation=0,
            run_started_monotonic_ns=_T0_NS,
            anchor_monotonic_ns=_ANCHOR_NS,
            latest_source_monotonic_ns=_ANCHOR_NS,
            logical_step_monotonic_ns=_ANCHOR_NS,
        )
        with self.assertRaises(ValueError):
            _to_policy_observation(batch, spec)

    def test_eef_pose_without_runtime_raises(self) -> None:
        spec = _fake_policy_spec(_Field("eef_pose", (9,), "float32"))
        observation = self._observation(spec)
        with self.assertRaises(RuntimeError):
            _to_policy_observation(observation, spec, fingertip_runtime=None)


class TestBuildFingertipRuntime(unittest.TestCase):
    def test_unrequested_fields_return_none(self) -> None:
        spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        self.assertIsNone(build_fingertip_runtime(spec, None))

    def test_eef_only_returns_arm_fk_without_hand_fk(self) -> None:
        spec = _fake_policy_spec(_Field("eef_pose", (9,), "float32"))
        runtime = build_fingertip_runtime(spec, None)
        self.assertIsNotNone(runtime)
        arm_fk, hand_fk, config = runtime
        self.assertTrue(hasattr(arm_fk, "compute"))
        self.assertIsNone(hand_fk)
        self.assertIsNone(config)

    def test_fingertip_without_config_raises(self) -> None:
        spec = _fake_policy_spec(_Field("fingertip_points", (5, 3), "float32"))
        with self.assertRaises(TypeError):
            build_fingertip_runtime(spec, None)


class TestRuntimeCompatibility(unittest.TestCase):
    def _runtime(self) -> types.SimpleNamespace:
        runtime = types.SimpleNamespace()
        runtime.policy = types.SimpleNamespace(control_hz=1e9 / _DT_NS)
        runtime.pointcloud = types.SimpleNamespace(num_points=1024)
        runtime.environment = types.SimpleNamespace(
            table=types.SimpleNamespace(enabled=False, plane_abcd=None)
        )
        return runtime

    def test_eef_pose_semantics_accepted(self) -> None:
        spec = _fake_policy_spec(
            _Field(
                "eef_pose",
                (9,),
                "float32",
                semantics={
                    "representation": "position_m_rot6d",
                    "frame": "xarm_base",
                    "position_units": "m",
                    "rotation_representation": "rot6d",
                    "derivation": EEF_POSE_DERIVATION,
                    "algorithm_id": EEF_POSE_ALGORITHM_ID,
                },
            ),
            _Field("tactile_force", (5, 120, 3), "float32",
                   semantics=_expected_tactile_force_semantics()),
        )
        validate_policy_runtime_compatibility(spec, self._runtime())

    def test_tactile_semantic_contract(self) -> None:
        expected = {
            "representation": "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
            "finger_order": "thumb_index_mid_ring_pinky",
            "sensor_order": "xhand_sdk_sensor_data_order",
            "point_order": "xhand_sdk_sensor_data_raw_force_order",
            "axis_labels": "fx_fy_fz",
            "unit": "xhand_sdk_native_unknown_si",
            "si_verified": False,
            "spatial_geometry_verified": False,
        }
        def validate(semantics):
            spec = _fake_policy_spec(_Field(
                "tactile_force", (5, 120, 3), "float32", semantics=semantics
            ))
            validate_policy_runtime_compatibility(spec, self._runtime())

        validate(expected)
        with self.assertRaises(ValueError):
            validate({})
        for key, value in expected.items():
            for replacement in ([True, 0, "false"] if isinstance(value, bool)
                                else ["wrong"]):
                with self.subTest(key=key, replacement=replacement):
                    with self.assertRaises(ValueError):
                        validate({**expected, key: replacement})
            with self.subTest(missing=key):
                with self.assertRaises(ValueError):
                    validate({name: item for name, item in expected.items() if name != key})

    def test_contact_force_native_semantics_accepted(self) -> None:
        spec = _fake_policy_spec(
            _Field(
                "contact_force",
                (5, 3),
                "float32",
                semantics=_expected_contact_force_semantics(),
            )
        )
        validate_policy_runtime_compatibility(spec, self._runtime())

    def test_contact_force_legacy_scaled_units_rejected(self) -> None:
        semantics = _expected_contact_force_semantics()
        semantics["units"] = "sdk_scaled_unknown_si"
        spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32", semantics=semantics)
        )
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, self._runtime())

    def test_contact_force_wrong_representation_rejected(self) -> None:
        semantics = _expected_contact_force_semantics()
        semantics["representation"] = "sdk_scaled_calc_force"
        spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32", semantics=semantics)
        )
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, self._runtime())

    def test_eef_pose_wrong_semantics_rejected(self) -> None:
        spec = _fake_policy_spec(
            _Field(
                "eef_pose",
                (9,),
                "float32",
                semantics={"representation": "position_quat"},
            )
        )
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, self._runtime())

    def test_tactile_force_wrong_shape_rejected(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 3), "float32"))
        with self.assertRaises(ValueError):
            validate_policy_runtime_compatibility(spec, self._runtime())


if __name__ == "__main__":
    unittest.main()
