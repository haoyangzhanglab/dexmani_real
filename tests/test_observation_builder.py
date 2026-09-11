"""Offline sensor admission checks at the observation builder boundary."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dexmani_real.deployment.inference import observation as obs
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    CAMERA_FRAME_HEADER_DTYPE,
    HAND_STATE_DTYPE,
)


@pytest.mark.parametrize(
    "source,publish,value",
    [
        (0, 90, 0),
        (101, 101, 0),
        (90, 101, 0),
        (1, 2, 0),
        (90, 95, np.nan),
    ],
)
def test_invalid_state_never_enters_history(source, publish, value):
    record = np.zeros(1, dtype=ARM_STATE_DTYPE)
    record["source_monotonic_ns"] = source
    record["publish_monotonic_ns"] = publish
    record["qpos"] = value
    ring = SimpleNamespace(maxlen=4, get_last_k=lambda k: [(record, publish, 1)])
    assert (
        obs._read_state_history(
            ring, history_len=4, anchor_ns=100, values_field="qpos", max_age_ns=20
        )
        is None
    )


def test_nonfinite_tactile_payload_is_rejected_at_read():
    record = np.zeros(1, dtype=HAND_STATE_DTYPE)
    record["source_monotonic_ns"] = 90
    record["publish_monotonic_ns"] = 95
    record["state_valid"] = 1
    record["tactile_dense_valid"] = 1
    record["tactile_dense"][0, 0, 0] = np.inf
    ring = SimpleNamespace(maxlen=4, get_last_k=lambda k: [(record, 95, 1)])
    assert (
        obs._read_state_history(
            ring,
            history_len=4,
            anchor_ns=100,
            values_field="tactile_dense",
            required_true_fields=("state_valid", "tactile_dense_valid"),
            max_age_ns=20,
            not_before_ns=1,
        )
        is None
    )


def _camera_header(source=90, receive=92, publish=95, generation=1):
    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["source_monotonic_ns"] = source
    header["receive_monotonic_ns"] = receive
    header["publish_monotonic_ns"] = publish
    header["camera_generation"] = generation
    return header


@pytest.mark.parametrize(
    "source,receive,publish,health,shape,dtype,admitted",
    [
        (90, 92, 95, 0, (2, 2, 3), np.uint8, True),
        (90, 89, 95, 0, (2, 2, 3), np.uint8, False),
        (90, 96, 95, 0, (2, 2, 3), np.uint8, False),
        (90, 92, 101, 0, (2, 2, 3), np.uint8, False),
        (1, 2, 3, 0, (2, 2, 3), np.uint8, False),
        (90, 92, 95, 1, (2, 2, 3), np.uint8, False),
        (90, 92, 95, 0, (2, 2, 4), np.uint8, False),
        (90, 92, 95, 0, (2, 2, 3), np.float32, False),
    ],
)
def test_rgb_requires_causal_receive_order_freshness_health_and_payload(
    source, receive, publish, health, shape, dtype, admitted
):
    header = _camera_header(source, receive, publish)
    header["camera_health"] = health
    ring = mock.Mock()
    ring.read_sequence.return_value = {
        "header": header,
        "rgb": np.zeros(shape, dtype=dtype),
    }
    frame = obs._rgb_frame_from_camera_record(
        ring, header, publish, 1, anchor_ns=100, max_age_ns=20, not_before_ns=1
    )
    assert (frame is not None) == admitted


def test_camera_restart_does_not_supply_old_history():
    headers = {1: _camera_header(80, 81, 82, 1), 2: _camera_header(90, 91, 92, 2)}
    ring = mock.Mock(maxlen=4)
    ring.get_last_metadata.return_value = [(headers[1], 82, 1), (headers[2], 92, 2)]
    ring.read_sequence.side_effect = lambda sequence, **kwargs: {
        "header": headers[sequence],
        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    frames = obs._read_rgb_history(
        SimpleNamespace(camera_ring=ring),
        anchor_ns=100,
        max_age_ns=30,
        history_len=4,
        not_before_ns=1,
    )
    assert len(frames) == 1 and frames[0].camera_generation == 2
    selected, _ = obs._select_camera_control_grid(
        frames,
        run_started_ns=70,
        anchor_ns=100,
        history_len=2,
        step_dt_ns=10,
        max_grid_lag_ns=20,
    )
    # Camera reuse fills both slots from the single post-restart frame; no
    # old-generation frame is present to leak through.
    assert len(selected) == 2
    assert all(frame.camera_generation == 2 for frame in selected)


def test_alignment_rejects_future_and_excessive_skew():
    for source in (110, 50):
        window = obs.FrameWindow(
            np.zeros((1, 7)),
            np.array([1]),
            np.array([source]),
            np.array([source]),
            np.ones(1),
        )
        assert (
            obs._align_state_history_to_reference_ns(
                window,
                np.array([100]),
                max_skew_ns=20,
                run_started_ns=1,
            )
            is None
        )


def test_float32_overflow_cannot_reach_model():
    def window(width, value):
        return obs.FrameWindow(
            np.full((1, width), value),
            np.array([1]),
            np.array([90]),
            np.array([95]),
            np.ones(1),
        )

    batch = obs.ObservationBatch(
        1, 1, 1, 100, 90, 100, arm_history=window(7, 1e100), hand_history=window(12, 0)
    )
    spec = SimpleNamespace(observation_fields=(SimpleNamespace(name="joint_state"),))
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="NaN/Inf"):
        obs._to_policy_observation(batch, spec)
