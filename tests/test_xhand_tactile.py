import json
import time
from dataclasses import replace
from itertools import chain, repeat
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import RawDataError, RawReader, episode_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.model import XHAND_TACTILE_SENSOR_FINGER_IDS
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand import driver
from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState, expire_tactile
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_observation,
)
from dexmani_real.runtime.state import SessionState


def sdk_state(aggregate=2, dense=3, *, fault_index=None, tip_code=0):
    joints = [SimpleNamespace(id=i, position=0.25, commboard_err=0, jonitboard_err=0, tipboard_err=0)
              for i in range(12)]
    if fault_index is not None:
        joints[XHAND_TACTILE_SENSOR_FINGER_IDS[fault_index]].tipboard_err = tip_code
    sensors = [SimpleNamespace(
        calc_force=SimpleNamespace(fx=aggregate, fy=0, fz=0),
        raw_force=[SimpleNamespace(fx=dense, fy=0, fz=0)] * 120,
    ) for _ in range(5)]
    return SimpleNamespace(finger_state=joints, sensor_data=sensors)


def reply(state=None, *, code=0):
    return SimpleNamespace(error_code=code, error_message="mock reply"), state


@pytest.fixture
def hand_driver(monkeypatch):
    cfg = replace(HandConfig(), tactile_calibration_samples=3, tactile_calibration_timeout_s=0.22)
    hand = driver.XHand(cfg)
    hand.connected_flag = True
    hand.control = Mock()
    hand.control.reset_sensor.return_value = reply()[0]
    clock = SimpleNamespace(now_ns=1_000_000_000)

    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)

    monkeypatch.setattr(driver, "time", SimpleNamespace(
        monotonic_ns=lambda: clock.now_ns, time_ns=lambda: clock.now_ns + 10**15, sleep=sleep,
    ))
    return hand, clock


@pytest.mark.parametrize("inject_crc", [False, True])
def test_calibration_means_only_valid_frames_and_applies_bias(hand_driver, inject_crc):
    hand, _ = hand_driver
    before = reply(sdk_state())
    failed = [reply(sdk_state(100, 100), code=driver.COMMUNICATION_CRC_ERROR_CODE)] if inject_crc else []
    hand.control.read_state.side_effect = [
        before, *failed, reply(sdk_state(1, 4)), reply(sdk_state(2, 5)), reply(sdk_state(3, 6)),
        reply(sdk_state(2, 5)), reply(sdk_state(3, 7)),
    ]

    assert hand.calibrate_tactile(no_contact=True)
    assert [call.args[1] for call in hand.control.reset_sensor.call_args_list] == list(driver.TACTILE_SENSOR_IDS)
    metadata = hand.tactile_calibration_metadata
    assert metadata["calibrated"] and metadata["bias_applied"] and metadata["bias_time_ns"] > 0
    np.testing.assert_array_equal(np.array(metadata["bias_sdk"]["aggregate"])[:, 0], 2)
    np.testing.assert_array_equal(np.array(metadata["bias_sdk"]["dense"])[:, :, 0], 5)
    state = hand.get_state()
    np.testing.assert_array_equal(state.tactile_aggregate[:, 0], 1)
    np.testing.assert_array_equal(state.tactile_dense[:, :, 0], 2)
    assert hand.crc_counts["read_errors"] == int(inject_crc)


def test_insufficient_samples_preserve_prior_bias_after_unverified_reset(hand_driver):
    hand, _ = hand_driver
    hand.control.read_state.return_value = reply(sdk_state())
    assert hand.calibrate_tactile(no_contact=True)
    previous = hand.tactile_calibration_metadata
    partial = reply(sdk_state(fault_index=2, tip_code=207), code=1_501_024)
    hand.control.read_state.side_effect = chain([reply(sdk_state())], repeat(partial))

    assert not hand.calibrate_tactile(no_contact=True)
    current = hand.tactile_calibration_metadata
    assert not current["calibrated"] and not current["bias_applied"]
    assert "insufficient fresh" in current["issue"]
    assert current["bias_sdk"] == previous["bias_sdk"]
    assert current["bias_time_ns"] == previous["bias_time_ns"]


def test_calibration_checks_contact_before_reset_and_held_out_residual(hand_driver):
    hand, _ = hand_driver
    hand.control.read_state.return_value = reply(sdk_state(60))
    assert not hand.calibrate_tactile(no_contact=True)
    hand.control.reset_sensor.assert_not_called()
    assert "contact" in hand.tactile_calibration_metadata["issue"]

    hand.control.read_state.side_effect = [reply(sdk_state())] * 4 + [reply(sdk_state(15))]
    assert not hand.calibrate_tactile(no_contact=True)
    assert "residual" in hand.tactile_calibration_metadata["issue"]
    assert hand.tactile_calibration_metadata["bias_sdk"] is None


@pytest.mark.parametrize("tip_code,sdk_code,valid_channels", [
    (207, 1_501_024, (False, False)),
    (200, driver.COMBINED_FORCE_UNAVAILABLE_CODE, (False, True)),
    (201, driver.DISTRIBUTED_FORCE_UNAVAILABLE_CODE, (True, False)),
])
def test_one_finger_sdk_error_is_local_to_its_affected_modalities(hand_driver, tip_code, sdk_code, valid_channels):
    hand, _ = hand_driver
    hand.control.read_state.return_value = reply(sdk_state(fault_index=1, tip_code=tip_code), code=sdk_code)
    state = hand.get_state()
    np.testing.assert_array_equal(state.qpos, 0.25)
    for values, valid in zip((state.tactile_aggregate, state.tactile_dense), valid_channels):
        mask = np.isfinite(values).reshape(5, -1).all(axis=1)
        np.testing.assert_array_equal(mask, [True, valid, True, True, True])


@pytest.mark.parametrize("sdk_code,missing,retained", [
    (driver.COMBINED_FORCE_UNAVAILABLE_CODE, "tactile_aggregate", "tactile_dense"),
    (driver.DISTRIBUTED_FORCE_UNAVAILABLE_CODE, "tactile_dense", "tactile_aggregate"),
])
def test_unlocated_sdk_error_only_invalidates_the_reported_modality(hand_driver, sdk_code, missing, retained):
    hand, _ = hand_driver
    hand.control.read_state.return_value = reply(sdk_state(), code=sdk_code)
    state = hand.get_state()
    assert getattr(state, missing) is None
    assert np.isfinite(getattr(state, retained)).all()


@pytest.mark.parametrize("code", [202, driver.TEMPERATURE_UNAVAILABLE_CODE])
def test_temperature_errors_preserve_force_feedback_and_are_recorded_on_reset(hand_driver, code):
    hand, _ = hand_driver
    hand.control.read_state.return_value = reply(sdk_state(), code=code)
    hand.control.reset_sensor.return_value = reply(code=code)[0]

    state = hand.get_state()
    assert np.isfinite(state.tactile_aggregate).all()
    assert np.isfinite(state.tactile_dense).all()
    assert hand.reset_tactile_sensors() == dict.fromkeys(driver.TACTILE_SENSOR_IDS, code)


def test_nonfinite_finger_reuses_only_its_own_recent_good_value(hand_driver):
    hand, clock = hand_driver
    feedback = XHandFeedback()
    first_ns = clock.now_ns
    hand.control.read_state.return_value = reply(sdk_state())
    feedback.consume(hand.get_state())

    clock.now_ns += 70_000_000
    state = sdk_state(6, 7)
    state.sensor_data[2].calc_force.fx = np.inf
    hand.control.read_state.return_value = reply(state)
    frame = feedback.consume(hand.get_state())
    assert frame["tactile_aggregate_timestamp_ns"][0, 2] == first_ns
    assert frame["tactile_dense_timestamp_ns"][0, 2] == clock.now_ns
    assert np.isfinite(frame["tactile_aggregate"]).all()

    clock.now_ns += 30_000_000
    frame = expire_tactile(frame, max_age_s=hand.cfg.tactile_max_age_s, now_ns=clock.now_ns)
    assert np.isnan(frame["tactile_aggregate"][0, 2]).all()
    assert np.isfinite(frame["tactile_aggregate"][0, [0, 1, 3, 4]]).all()
    assert np.isfinite(frame["tactile_dense"]).all()
    row = observation(frame, clock.now_ns)
    runtime = SimpleNamespace(hand=hand.cfg)
    assert check_observation(row, runtime, ObservationNeeds(required_tactile=("dense",)), now_ns=clock.now_ns) == []
    issues = check_observation(row, runtime, ObservationNeeds(required_tactile=("aggregate",)), now_ns=clock.now_ns)
    assert len(issues) == 1 and "fingers=[2]" in issues[0]


def test_no_tactile_payload_keeps_joint_feedback_and_no_zero_tactile(hand_driver):
    hand, _ = hand_driver
    state = sdk_state()
    state.sensor_data = []
    hand.control.read_state.return_value = reply(state)
    sample = hand.get_state()
    assert sample.tactile_aggregate is None and sample.tactile_dense is None
    frame = XHandFeedback().consume(sample)
    np.testing.assert_array_equal(frame["qpos"], 0.25)
    assert np.isnan(frame["tactile_aggregate"]).all() and np.isnan(frame["tactile_dense"]).all()


def observation(frame, stamp):
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["timestamp_ns"] = stamp
    return Observation(
        arm=arm, hand=frame, vr=None, point_cloud=None, observation_timestamp_ns=stamp, control_timestamp_ns=stamp,
        camera={"timestamp_ns": stamp, "ring_sequence": 1, "rgb": np.zeros((24, 32, 3), dtype=np.uint8),
                "depth": np.zeros((24, 32), dtype=np.uint16)},
    )


@pytest.mark.parametrize("requires_tactile", [False, True])
def test_rollout_only_requires_tactile_when_trained_fields_need_it(requires_tactile):
    now = time.monotonic_ns()
    state = SessionState()
    runtime = RealConfig(execution=ExecutionConfig(control_dt_s=0.05, action_steps=1))
    fields = ("joint_state", "finger_tactile_force") if requires_tactile else ("joint_state",)
    info = SimpleNamespace(observation_fields=fields, n_obs_steps=1, requires_prev_action=True)
    observations = ObservationBuilder(info, runtime)
    row = observation(XHandFeedback().consume(XHandState(np.zeros(12), None, None, now)), now)
    model = Mock(future=None)
    robot = Mock(hand_feedback_pending=False)
    rollout = Rollout(
        state, runtime, info, episode_cancel_epoch=state.cancel_epoch, robot=robot,
        observations=observations, sensors=None, action_mapper=None, model=model,
        max_duration_s=None, index=1, output_dir=None, record_raw=True, effective_parameters={},
    )
    rollout.started_ns = now
    rollout.recorder = Mock()
    rollout.recorder.append.return_value = None
    rollout.read_observations = Mock(return_value=(row, row))
    assert rollout.recording_issue(row) is None

    rollout.step()

    if requires_tactile:
        assert rollout.finished
        model.predict.assert_not_called()
        assert "hand_tactile_aggregate_unavailable_or_stale" in rollout.result.termination_details
    else:
        assert not rollout.finished and not state.stop_signal.is_set()
        model.predict.assert_called_once()
        assert set(observations.build(model.predict.call_args.args[0])) == {"joint_state"}
        rollout.recorder.append.assert_called_once()


def test_raw_keeps_bias_validity_and_source_times_with_partial_tactile(hand_driver, tmp_path):
    hand, clock = hand_driver
    hand.control.read_state.return_value = reply(sdk_state())
    assert hand.calibrate_tactile(no_contact=True)
    hand.control.read_state.return_value = reply(sdk_state(fault_index=3, tip_code=201))
    row = observation(XHandFeedback().consume(hand.get_state()), clock.now_ns)
    camera = CameraSnapshot(None, CameraIntrinsics(32, 24, 30, 30, 16, 12, "none", (0,) * 5), None, None)
    metadata = episode_recording_metadata(
        camera, {}, tactile_reset=hand.tactile_reset_metadata,
        tactile_calibration=hand.tactile_calibration_metadata,
    )
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.start(task_label="tactile", collection_source="policy_rollout", metadata=metadata)
    try:
        assert recorder.append(row, action_accepted=False) is None
    finally:
        recorder.finish(save=True, reason="test")

    with RawReader(recorder.published_path) as raw:
        parameters = json.loads(raw.meta["effective_parameters"])
        assert parameters["tactile_calibration"] == hand.tactile_calibration_metadata
        assert set(raw.data_file.keys()) == {
            "meta", "timing", "depth", "arm_qpos", "hand_qpos", "hand_contact", "hand_tactile_force",
            "action_arm_joint_target", "action_hand_joint_target", "action_send_accepted",
        }
        assert np.isfinite(raw["hand_contact"][0]).all()
        valid = np.isfinite(raw["hand_tactile_force"][0]).all(axis=(1, 2))
        np.testing.assert_array_equal(valid, [True, True, True, False, True])
        assert np.isnan(raw["hand_tactile_force"][0, 3]).all()
        assert raw.read_timing(source_clocks=("arm_ns", "hand_ns", "tactile_aggregate_ns"))["tactile_dense_ns"][0] == 0
        with pytest.raises(RawDataError, match="tactile_dense_ns"):
            raw.read_timing(source_clocks=("tactile_dense_ns",))

    source = recorder.published_path
    fields = ("joint_state", "finger_tactile_force")
    assert export_zarr(source, tmp_path / "valid.zarr", processing=ExportConfig(), fields=fields)["total_frames"] == 1
    with pytest.raises(ValueError, match="all Raw episodes rejected.*tactile_dense_ns"):
        export_zarr(source, tmp_path / "invalid.zarr", processing=ExportConfig(), fields=("taxel_tactile_force",))
