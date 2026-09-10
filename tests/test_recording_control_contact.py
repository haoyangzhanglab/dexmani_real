"""Recording contact has its own causal selection, without command-side effects."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dexmani_real.dataset.contracts import OutputProfile, ProcessingConfig
from dexmani_real.dataset.processing import _open_processing_episode, analyze_episode
from dexmani_real.deployment import executor as executor_module
from dexmani_real.ipc.causal import read_hand_contact_causal
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.recording.frame import decode_record_sample
from dexmani_real.teleop.episode_samples import record_frame, record_held
from test_control_step_dataset import write_control_episode
from test_raw_v26_recording import _frame, _recorder
from test_v26_producers import _client


class HandRing:
    def __init__(self, frames):
        self.frames = frames
        self.latest_sequence = len(frames)
        self.maxlen = max(1, len(frames))

    def read_sequence(self, sequence):
        frame, published = self.frames[sequence - 1]
        return frame.copy(), published, sequence


def hand_frame(source_ns, *, valid=True, value=3.0):
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand["source_monotonic_ns"] = source_ns
    hand["publish_monotonic_ns"] = source_ns + 1
    hand["state_valid"] = 1
    hand["tactile_sum_valid"] = int(valid)
    hand["tactile_sum"] = value
    hand["qpos"] = source_ns / 1e9
    return hand


@pytest.mark.parametrize(
    "fault",
    [
        "state",
        "sum",
        "stale",
        "nan",
        "future_payload",
        "future_ring",
        "publish_order",
        "zero_source",
    ],
)
def test_contact_reader_skips_invalid_newest(fault):
    old = hand_frame(800_000_000)
    new = hand_frame(900_000_000, value=9)
    published = 900_000_002
    if fault == "state":
        new["state_valid"] = 0
    elif fault == "sum":
        new["tactile_sum_valid"] = 0
    elif fault == "stale":
        new["qpos_stale"] = 1
    elif fault == "nan":
        new["tactile_sum"] = np.nan
    elif fault == "future_payload":
        new["publish_monotonic_ns"] = 1_000_000_001
    elif fault == "future_ring":
        published = 1_000_000_001
    elif fault == "publish_order":
        published = 899_999_999
    else:
        new["source_monotonic_ns"] = 0
    selected = read_hand_contact_causal(
        HandRing([(old, 800_000_002), (new, published)]),
        anchor_monotonic_ns=1_000_000_000,
    )
    np.testing.assert_array_equal(selected, old)


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("case", ["current", "fallback", "old", "missing"])
def test_teleop_records_selected_contact_and_keeps_latest_hand(held, case):
    anchor = 1_000_000_000
    latest = hand_frame(950_000_000, valid=case == "current", value=9)
    frames = []
    expected_source = 0
    if case in {"fallback", "old"}:
        expected_source = 900_000_000 if case == "fallback" else 500_000_000
        frames.append((hand_frame(expected_source), expected_source + 2))
    frames.append((latest, 950_000_002))
    if case == "current":
        expected_source = 950_000_000
    client = _client()
    shared = SimpleNamespace(hand_state_ring=HandRing(frames))
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    common = dict(
        observation_anchor_monotonic_ns=anchor,
        shared=shared,
        max_observation_skew_s=0.1,
    )
    if held:
        record_held(
            client,
            arm,
            np.zeros(7),
            np.zeros(12),
            None,
            None,
            hand_state=latest,
            arm_qpos_sent=np.zeros(7),
            **common,
        )
    else:
        record_frame(
            client,
            arm,
            latest,
            np.zeros(7),
            np.zeros(12),
            np.zeros(3),
            np.array([1, 0, 0, 0]),
            None,
            None,
            **common,
        )
    row = decode_record_sample(client.shared.record_sample_ring.frame[0]).data
    np.testing.assert_array_equal(row["hand_qpos"], latest["qpos"][0])
    assert row["hand_source_monotonic_ns"] == 950_000_000
    assert row["hand_contact_source_monotonic_ns"] == expected_source
    assert bool(row["tactile_sum_fresh"]) == (case in {"current", "fallback"})
    if case == "missing":
        assert np.isnan(row["hand_contact"]).all()
    else:
        np.testing.assert_array_equal(
            row["hand_contact"], 9 if case == "current" else 3
        )
    assert np.isnan(row["hand_tactile_force"]).all()


@pytest.mark.parametrize("source_ns", [0, 10_000_000_000])
def test_current_raw_contact_source_is_required_and_legacy26_uses_hand_source(
    tmp_path, source_ns
):
    episode = write_control_episode(tmp_path)
    config = ProcessingConfig(profile=OutputProfile.JOINT)
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["hand_contact_source_monotonic_ns"][5] = source_ns
    with _open_processing_episode(episode) as reader:
        with pytest.raises(ValueError, match="hand_contact_source"):
            analyze_episode(reader, config)
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["meta"].attrs["schema_version"] = 26
        del raw["hand_contact_source_monotonic_ns"]
    with _open_processing_episode(episode) as reader:
        assert analyze_episode(reader, config).accepted


@pytest.mark.parametrize("missing", [False, True])
def test_rollout_records_contact_independently_without_admitting_commands(
    monkeypatch, missing
):
    now = 1_000_000_000
    latest = hand_frame(950_000_000, valid=False, value=0)
    frames = [] if missing else [(hand_frame(500_000_000), 500_000_002)]
    frames.append((latest, 950_000_002))
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["source_monotonic_ns"] = 950_000_000
    shared = SimpleNamespace(
        arm_state_ring=object(),
        hand_state_ring=HandRing(frames),
        hand_tactile_ring=object(),
    )
    monkeypatch.setattr(
        executor_module,
        "read_causal_structured_frame",
        lambda ring, **kwargs: (
            (arm, 950_000_001, 1)
            if ring is shared.arm_state_ring
            else (latest, 950_000_001, 1) if ring is shared.hand_state_ring else None
        ),
    )
    monkeypatch.setattr(
        executor_module,
        "read_camera_frame_causal",
        lambda *args, **kwargs: {
            "source_monotonic_ns": 940_000_000,
            "camera_health": 0,
        },
    )
    recorded = []
    executor = SimpleNamespace(
        recorder=object(),
        run_started_ns=1,
        next_record_ns=now,
        step_dt_ns=62_500_000,
        shared=shared,
        runtime=SimpleNamespace(camera=SimpleNamespace(max_frame_age_s=0.2)),
        _recorded_hold_action=lambda state: SimpleNamespace(arm_qpos_cmd=np.zeros(7)),
        _record_frame=lambda inputs, *args, **kwargs: recorded.append(inputs) or True,
    )
    executor_module.PolicyExecutor._record_rollout_tick(executor, now)
    state, camera, signals = recorded[0]
    np.testing.assert_array_equal(state.hand_qpos, latest["qpos"][0])
    assert signals["hand_source_monotonic_ns"] == 950_000_000
    assert not signals["tactile_sum_fresh"]
    assert state.hand_contact_source_monotonic_ns == (0 if missing else 500_000_000)
    if missing:
        assert np.isnan(state.hand_tactile_sum).all()
    else:
        np.testing.assert_array_equal(state.hand_tactile_sum, 3)


def test_contact_source_survives_writer_without_repair(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start_episode()
    frame = _frame(1.0)
    frame.data["hand_contact_source_monotonic_ns"] = np.uint64(500_000_000)
    frame.data["hand_contact"][:] = 7
    recorder.add_episode_frame(frame)
    result = Path(recorder.finish_episode(save=True))
    with h5py.File(result / "data.h5", "r") as raw:
        assert raw["hand_contact_source_monotonic_ns"].dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(
            raw["hand_contact_source_monotonic_ns"][:], [500_000_000]
        )
        np.testing.assert_array_equal(raw["hand_contact"][:], 7)
