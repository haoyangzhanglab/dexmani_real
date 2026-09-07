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
    validate_policy_runtime_compatibility,
)
from dexmani_real.deployment.inference.observation import (
    FrameWindow,
    ObservationBatch,
    PolicyObservation,
    _align_state_history_to_camera_frames,
    _build_observation,
    _to_policy_observation,
    build_fingertip_runtime,
)
from dexmani_real.deployment.lifecycle import _requires_hand_sensor
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
)
from dexmani_real.planning.kinematics.arm_fk import (
    EEF_POSE_ALGORITHM_ID,
    EEF_POSE_DERIVATION,
)

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


def _arm_record(tick: int, qpos: np.ndarray) -> tuple[np.ndarray, int, int]:
    data = np.zeros(1, dtype=ARM_STATE_DTYPE)
    data["qpos"][0] = qpos
    data["state_valid"][0] = 1
    data["source_monotonic_ns"][0] = _ref_ns(tick)
    data["publish_monotonic_ns"][0] = _ref_ns(tick) + 10**6
    return data, data["publish_monotonic_ns"][0], tick + 1


def _hand_state_record(tick: int) -> tuple[np.ndarray, int, int]:
    data = np.zeros(1, dtype=HAND_STATE_DTYPE)
    data["qpos"][0] = 0.1
    data["tactile_sum"][0] = float(tick)
    data["state_valid"][0] = 1
    data["tactile_sum_valid"][0] = 1
    data["qpos_stale"][0] = 0
    data["source_monotonic_ns"][0] = _ref_ns(tick)
    data["publish_monotonic_ns"][0] = _ref_ns(tick) + 10**6
    return data, data["publish_monotonic_ns"][0], tick + 1


def _tactile_record(
    tick: int,
    *,
    fresh: bool = True,
    calibrated: bool = True,
    unit_code: int = 0,
) -> tuple[np.ndarray, int, int]:
    data = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
    data["tactile_force"][0] = float(tick)
    data["source_monotonic_ns"][0] = _ref_ns(tick)
    data["fresh"][0] = int(fresh)
    data["calibrated"][0] = int(calibrated)
    data["unit_code"][0] = unit_code
    return data, _ref_ns(tick) + 10**6, tick + 1


def _fake_shared(
    *,
    arm_qpos: np.ndarray | None = None,
    tactile_uncalibrated_ticks: tuple[int, ...] = (),
) -> types.SimpleNamespace:
    ticks = range(6)
    qpos_history = (
        np.zeros((6, 7)) if arm_qpos is None else np.asarray(arm_qpos)
    )
    return types.SimpleNamespace(
        arm_state_ring=_FakeRing(
            [_arm_record(tick, qpos_history[tick]) for tick in ticks]
        ),
        hand_state_ring=_FakeRing([_hand_state_record(tick) for tick in ticks]),
        hand_tactile_ring=_FakeRing(
            [
                _tactile_record(
                    tick, calibrated=tick not in tactile_uncalibrated_ticks
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

    def test_policy_observation_accepts_new_tails(self) -> None:
        arrays = {
            "joint_state": np.zeros((_HORIZON, 19), dtype=np.float32),
            "eef_pose": np.zeros((_HORIZON, 9), dtype=np.float32),
            "tactile_force": np.zeros((_HORIZON, 5, 120, 3), dtype=np.float32),
        }
        observation = PolicyObservation(
            observation_id=1,
            run_generation=0,
            anchor_monotonic_ns=_ANCHOR_NS,
            latest_source_monotonic_ns=_ref_ns(5),
            logical_step_monotonic_ns=_ref_ns(5),
            arrays=arrays,
        )
        self.assertEqual(
            tuple(observation.arrays), ("joint_state", "eef_pose", "tactile_force")
        )

    def test_policy_observation_rejects_wrong_tail_and_dtype(self) -> None:
        base = {
            "observation_id": 1,
            "run_generation": 0,
            "anchor_monotonic_ns": _ANCHOR_NS,
            "latest_source_monotonic_ns": _ref_ns(5),
            "logical_step_monotonic_ns": _ref_ns(5),
        }
        with self.assertRaises(ValueError):
            PolicyObservation(
                arrays={
                    "joint_state": np.zeros((_HORIZON, 19), dtype=np.float32),
                    "eef_pose": np.zeros((_HORIZON, 8), dtype=np.float32),
                },
                **base,
            )
        with self.assertRaises(TypeError):
            PolicyObservation(
                arrays={
                    "joint_state": np.zeros((_HORIZON, 19), dtype=np.float32),
                    "tactile_force": np.zeros((_HORIZON, 5, 120, 3), dtype=np.float64),
                },
                **base,
            )

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
    def test_joint_state_only_skips_full_tactile_read(self) -> None:
        spec = _fake_policy_spec(_Field("eef_pose", (9,), "float32"))
        with mock.patch.object(
            observation_mod, "_read_tactile_force_history"
        ) as force_reader:
            observation = _build(_fake_shared(), spec)
        force_reader.assert_not_called()
        self.assertIsNotNone(observation)
        self.assertIsNone(observation.hand_tactile_force_history)
        np.testing.assert_array_equal(
            observation.arm_history.source_monotonic_ns,
            np.asarray([_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64),
        )

    def test_contact_only_does_not_copy_full_tactile(self) -> None:
        spec = _fake_policy_spec(_Field("contact_force", (5, 3), "float32"))
        with mock.patch.object(
            observation_mod, "_read_tactile_force_history"
        ) as force_reader:
            observation = _build(_fake_shared(), spec)
        force_reader.assert_not_called()
        self.assertIsNotNone(observation)
        self.assertIsNotNone(observation.hand_tactile_sum_history)
        self.assertIsNotNone(observation.hand_tactile_provenance_history)
        self.assertIsNone(observation.hand_tactile_force_history)

    def test_tactile_force_requested_reads_aligned_full_tensor(self) -> None:
        spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32"),
            _Field("tactile_force", (5, 120, 3), "float32"),
        )
        observation = _build(_fake_shared(), spec)
        self.assertIsNotNone(observation)
        force_history = observation.hand_tactile_force_history
        self.assertEqual(force_history.values.shape, (_HORIZON, 5, 120, 3))
        np.testing.assert_array_equal(
            force_history.source_monotonic_ns,
            observation.hand_tactile_sum_history.source_monotonic_ns,
        )
        np.testing.assert_array_equal(
            force_history.source_monotonic_ns,
            observation.hand_tactile_provenance_history.source_monotonic_ns,
        )
        for index, tick in enumerate(_REF_TICKS):
            self.assertTrue(np.all(force_history.values[index] == float(tick)))

    def test_contact_and_force_source_identity_is_enforced(self) -> None:
        spec = _fake_policy_spec(
            _Field("contact_force", (5, 3), "float32"),
            _Field("tactile_force", (5, 120, 3), "float32"),
        )
        shifted_sources = np.asarray(
            [_ref_ns(tick) - _DT_NS for tick in _REF_TICKS], dtype=np.uint64
        )
        shifted = FrameWindow(
            values=np.zeros((_HORIZON, 5, 120, 3)),
            source_sequence=np.ones(_HORIZON, dtype=np.uint64),
            source_monotonic_ns=shifted_sources,
            publish_monotonic_ns=shifted_sources + 10**6,
            valid_mask=np.ones(_HORIZON, dtype=np.uint8),
        )
        with mock.patch.object(
            observation_mod, "_read_tactile_force_history", return_value=shifted
        ):
            self.assertIsNone(_build(_fake_shared(), spec))

    def test_uncalibrated_tactile_fails_closed(self) -> None:
        # Ticks 4-5 uncalibrated: the newest calibrated source for ref5 is
        # ref3, which exceeds the 100ms skew budget, so no observation is
        # assembled (a single stale sample would legally forward-align).
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        shared = _fake_shared(tactile_uncalibrated_ticks=(4, 5))
        self.assertIsNone(_build(shared, spec))

    def test_force_window_respects_causal_cut(self) -> None:
        window = FrameWindow(
            values=np.zeros((_HORIZON, 5, 120, 3)),
            source_sequence=np.ones(_HORIZON, dtype=np.uint64),
            source_monotonic_ns=np.asarray(
                [_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64
            ),
            publish_monotonic_ns=np.asarray(
                [_ANCHOR_NS + 1] * _HORIZON, dtype=np.uint64
            ),
            valid_mask=np.ones(_HORIZON, dtype=np.uint8),
        )
        arm = FrameWindow(
            values=np.zeros((_HORIZON, 7)),
            source_sequence=np.ones(_HORIZON, dtype=np.uint64),
            source_monotonic_ns=np.asarray(
                [_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64
            ),
            publish_monotonic_ns=np.asarray(
                [_ref_ns(tick) for tick in _REF_TICKS], dtype=np.uint64
            ),
            valid_mask=np.ones(_HORIZON, dtype=np.uint8),
        )
        with self.assertRaises(ValueError):
            ObservationBatch(
                observation_id=1,
                run_generation=0,
                run_started_monotonic_ns=_T0_NS,
                anchor_monotonic_ns=_ANCHOR_NS,
                latest_source_monotonic_ns=_ref_ns(5),
                logical_step_monotonic_ns=_ref_ns(5),
                arm_history=arm,
                hand_tactile_force_history=window,
            )

    def test_camera_alignment_parity_for_force_history(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        observation = _build(_fake_shared(), spec)
        force_history = observation.hand_tactile_force_history
        camera_frames = tuple(
            types.SimpleNamespace(source_monotonic_ns=_ref_ns(tick))
            for tick in _REF_TICKS
        )
        aligned = _align_state_history_to_camera_frames(
            force_history, camera_frames, max_skew_ns=int(0.1 * 1e9)
        )
        self.assertIsNotNone(aligned)
        np.testing.assert_array_equal(
            aligned.source_monotonic_ns, force_history.source_monotonic_ns
        )


class TestToPolicyObservation(unittest.TestCase):
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

    def test_tactile_force_without_history_raises(self) -> None:
        spec = _fake_policy_spec(_Field("tactile_force", (5, 120, 3), "float32"))
        observation = _build(
            _fake_shared(arm_qpos=_distinct_arm_qpos()),
            _fake_policy_spec(),
        )
        with self.assertRaises(ValueError):
            _to_policy_observation(observation, spec)

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
            _Field("tactile_force", (5, 120, 3), "float32"),
        )
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
