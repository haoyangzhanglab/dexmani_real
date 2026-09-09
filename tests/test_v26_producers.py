"""Offline checks of real v26 producer inputs, without SDKs or workers."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dexmani_real.deployment.executor import PolicyExecutor
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    make_record_sample_dtype,
)
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.frame import build_episode_frame, decode_record_sample
from dexmani_real.recording.sample import EpisodeAction, build_episode_state
from dexmani_real.teleop.episode_samples import record_frame, record_held
from dexmani_real.teleop.control_loop.grid import _recording_policy_observation_signals


def _state():
    return build_episode_state(
        np.zeros(1, dtype=ARM_STATE_DTYPE),
        np.zeros(1, dtype=HAND_STATE_DTYPE),
        timestamp_s=1.0,
    )


def test_first_hold_action_derives_intent_from_physical_joint_state():
    state = _state()
    executor = SimpleNamespace(last_recorded_action=None)
    action = PolicyExecutor._recorded_hold_action(executor, state)
    position, rot6d = make_arm_fk().compute(state.arm_qpos)
    np.testing.assert_array_equal(action.target_eef_pos, position)
    np.testing.assert_array_equal(action.target_eef_rot6d, rot6d)


class _SampleRing:
    dtype = make_record_sample_dtype((2, 2, 3), (2, 2))
    maxlen = 4
    latest_sequence = 0

    def write(self, frame):
        self.frame = frame.copy()
        self.latest_sequence += 1


def _client():
    client = object.__new__(RecorderClient)
    client.shared = SimpleNamespace(
        record_sample_ring=_SampleRing(),
        recorder_consumed_sequence=SimpleNamespace(value=0),
    )
    client._recording = True
    client._max_frames = 0
    client._frame_count = 0
    return client


def _inputs():
    return (
        _state(),
        EpisodeAction(
            np.zeros(7), np.zeros(12), np.zeros(3), np.array([1, 0, 0, 0, 1, 0])
        ),
        {
            "wrist_pos": np.zeros(3),
            "wrist_quat_wxyz": np.array([1, 0, 0, 0]),
            "landmarks": np.zeros((21, 3)),
            "head_quat_wxyz": np.array([0, 1, 0, 0]),
        },
    )


def test_client_and_direct_frame_share_mapping_and_own_serialized_values():
    inputs = _inputs()
    sent = np.arange(7) / 10
    kwargs = {
        "arm_qpos_sent": sent,
        "signals": {"observation_anchor_monotonic_ns": 10**9, "action_queued": True},
        "camera_frame": {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "camera_fresh": True,
        },
    }
    expected = build_episode_frame(*inputs, **kwargs)
    client = _client()
    assert client.add_frame(*inputs, **kwargs)
    decoded = decode_record_sample(client.shared.record_sample_ring.frame[0])
    assert decoded.timestamp_s == expected.timestamp_s
    assert decoded.data.keys() == expected.data.keys()
    for name in expected.data:
        np.testing.assert_array_equal(decoded.data[name], expected.data[name])
    np.testing.assert_array_equal(
        decoded.data["head_quat_wxyz"], inputs[2]["head_quat_wxyz"]
    )
    sent[:] = -99
    np.testing.assert_array_equal(
        decoded.data["action_arm_joint_sent"], np.arange(7) / 10
    )


def test_client_requires_explicit_submitted_arm_target():
    with pytest.raises(ValueError, match="submitted arm target"):
        _client().add_frame(*_inputs())


def _dense_tactile_frame(*, fresh: bool, source_ns: int, value: float = 7.0):
    tactile = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
    tactile["fresh"][0] = int(fresh)
    tactile["source_monotonic_ns"][0] = source_ns
    tactile["tactile_force"][0] = value
    return tactile


def _record_dense_frame(client, hand_tactile, *, anchor_ns):
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    vr = _inputs()[2]
    record_frame(
        client,
        arm,
        hand,
        np.zeros(7),
        np.zeros(12),
        np.zeros(3),
        np.array([1, 0, 0, 0]),
        vr,
        None,
        hand_tactile,
        observation_anchor_monotonic_ns=anchor_ns,
        max_observation_skew_s=0.1,
    )
    return decode_record_sample(client.shared.record_sample_ring.frame[0])


def test_non_fresh_dense_tactile_persists_nan():
    frame = _record_dense_frame(
        _client(),
        _dense_tactile_frame(fresh=False, source_ns=10**9),
        anchor_ns=2 * 10**9,
    )
    assert np.isnan(frame.data["hand_tactile_force"]).all()


def test_age_stale_dense_tactile_persists_nan():
    # fresh=1 but the source is older than the 250ms recording age bound.
    frame = _record_dense_frame(
        _client(),
        _dense_tactile_frame(fresh=True, source_ns=10**9),
        anchor_ns=2 * 10**9,
    )
    assert np.isnan(frame.data["hand_tactile_force"]).all()


def test_fresh_dense_tactile_persists_payload():
    frame = _record_dense_frame(
        _client(),
        _dense_tactile_frame(fresh=True, source_ns=10**9, value=7.0),
        anchor_ns=1_200_000_000,
    )
    assert np.all(frame.data["hand_tactile_force"] == 7.0)


def test_teleop_active_and_ik_hold_reach_real_client_with_v26_fields():
    client = _client()
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["tracking_err"] = 0.125
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    vr = _inputs()[2]
    target = np.arange(7) / 10
    record_frame(
        client,
        arm,
        hand,
        target,
        np.zeros(12),
        np.zeros(3),
        np.array([1, 0, 0, 0]),
        vr,
        None,
        observation_anchor_monotonic_ns=10**9,
        max_observation_skew_s=0.1,
    )
    first = decode_record_sample(client.shared.record_sample_ring.frame[0])
    assert first.timestamp_s == 1.0
    assert first.data["flag_frame_status"] == 0
    assert first.data["flag_action_queued"]
    np.testing.assert_array_equal(first.data["action_arm_joint_sent"], target)
    record_held(
        client,
        arm,
        target,
        np.zeros(12),
        vr,
        None,
        hand_state=hand,
        frame_status=2,
        arm_qpos_sent=target,
        action_queued=True,
        observation_anchor_monotonic_ns=2 * 10**9,
        max_observation_skew_s=0.1,
    )
    held = decode_record_sample(client.shared.record_sample_ring.frame[0])
    assert held.timestamp_s == 2.0
    assert held.data["flag_frame_status"] == 2
    assert held.data["flag_action_queued"]
    np.testing.assert_array_equal(held.data["action_arm_joint_sent"], target)
    assert held.data["tracking_error"] == 0.125


def test_published_retarget_failure_remains_queued():
    client = _client()
    record_frame(
        client,
        np.zeros(1, dtype=ARM_STATE_DTYPE),
        np.zeros(1, dtype=HAND_STATE_DTYPE),
        np.zeros(7),
        np.zeros(12),
        np.zeros(3),
        np.array([1, 0, 0, 0]),
        _inputs()[2],
        None,
        frame_status=4,
        observation_anchor_monotonic_ns=10**9,
        max_observation_skew_s=0.1,
    )
    frame = decode_record_sample(client.shared.record_sample_ring.frame[0])
    assert frame.data["flag_frame_status"] == 4
    assert frame.data["flag_action_queued"]


@pytest.mark.parametrize(
    "source_ns, valid", [(950_000_000, True), (800_000_000, False)]
)
def test_visual_record_builder_owns_causal_feedback_freshness(source_ns, valid):
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    for frame in (arm, hand):
        frame["source_monotonic_ns"] = source_ns
        frame["state_valid"] = True
    shared = SimpleNamespace(
        arm_state_ring=object(),
        hand_state_ring=object(),
        hand_tactile_ring=object(),
    )
    with mock.patch(
        "dexmani_real.teleop.control_loop.grid.read_valid_structured_frame_aligned_to_source",
        side_effect=[(arm, source_ns, 1), (hand, source_ns, 1), None, None],
    ):
        signals = _recording_policy_observation_signals(
            shared,
            {"source_monotonic_ns": 10**9},
            anchor_monotonic_ns=1_010_000_000,
            max_observation_skew_s=0.1,
        )
    assert signals["policy_observation_valid"] == valid
