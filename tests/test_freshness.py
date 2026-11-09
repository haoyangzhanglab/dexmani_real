import multiprocessing as mp
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_observation,
    read_observation,
    sources_available,
)


def hand_sample(stamp):
    return XHandState(
        qpos=np.deg2rad(RealConfig().hand.home_qpos_deg),
        tactile_aggregate=np.ones((5, 3), np.float32),
        tactile_dense=np.full((5, 120, 3), 2, np.float32),
        timestamp_ns=stamp,
    )


def observation(stamp):
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["qpos"] = RealConfig().arm.home_qpos
    arm["timestamp_ns"] = stamp
    return Observation(
        arm=arm, hand=XHandFeedback().consume(hand_sample(stamp)),
        camera={
            "timestamp_ns": stamp, "ring_sequence": 1,
            "rgb": np.ones((4, 4, 3), np.uint8),
            "depth": np.ones((4, 4), np.uint16),
        },
        vr={"recv_ts_ns": stamp},
        point_cloud=np.ones((2, 6), np.float32), pointcloud_timestamp_ns=stamp,
        control_timestamp_ns=stamp, observation_timestamp_ns=stamp + 1,
    )


def test_defaults_and_feedback_period_validation(runtime):
    from dexmani_real.robot.system import FEEDBACK_PERIOD_NS
    from dexmani_real.runtime.preflight import validate_robot_config

    assert runtime.arm.feedback_max_age_s == 4 / 25
    assert runtime.hand.feedback_max_age_s == 0.20
    assert runtime.teleop.target_max_age_s == runtime.keyboard_teleop.target_max_age_s == 0.25
    validate_robot_config(runtime)
    runtime.teleop.validate()
    runtime.keyboard_teleop.validate()
    with pytest.raises(ValueError, match="feedback period"):
        replace(runtime.arm, feedback_max_age_s=1 / runtime.arm.feedback_hz).validate()
    with pytest.raises(ValueError, match="30 Hz"):
        validate_robot_config(replace(runtime, hand=replace(
            runtime.hand, feedback_max_age_s=FEEDBACK_PERIOD_NS / 1e9,
        )))


@pytest.mark.parametrize("ttl", [4 / 25, 0.20, 4 / 30, 0.5])
def test_freshness_boundaries(ttl):
    from dexmani_real.utils.timing import sample_is_fresh

    stamp = 1_000_000_000
    budget = int(ttl * 1e9)
    assert sample_is_fresh(stamp, ttl, stamp)
    assert sample_is_fresh(stamp, ttl, stamp + budget - 1)
    assert not sample_is_fresh(stamp, ttl, stamp + budget)
    assert not sample_is_fresh(stamp, ttl, stamp - 1)
    assert not sample_is_fresh(0, ttl, stamp)


def test_camera_first_receipt_and_duplicate_callbacks(clock, camera_snapshot):
    from dexmani_real.sensor.realsense import RealSenseCamera

    camera = RealSenseCamera(CameraConfig(width=4, height=4))
    camera.color_intrinsics = camera_snapshot.color
    camera.aligner = SimpleNamespace(process=lambda frames: frames)

    def frameset(color_number, depth_number):
        frames = SimpleNamespace(
            get_color_frame=lambda: SimpleNamespace(
                get_frame_number=lambda: color_number,
                get_data=lambda: np.full((4, 4, 3), color_number, np.uint8),
            ),
            get_depth_frame=lambda: SimpleNamespace(
                get_frame_number=lambda: depth_number,
                get_data=lambda: np.full((4, 4), depth_number, np.uint16),
            ),
            keep=Mock(),
        )
        frames.as_frameset = lambda: frames
        return frames

    source_ns = clock.now_ns
    camera.receive_frame(frameset(1, 1))
    clock.now_ns += 20_000_000
    first = camera.read()
    assert first.timestamp_ns == source_ns
    clock.now_ns += 20_000_000
    camera.receive_frame(frameset(1, 1))
    assert camera.pending_frame is None
    assert camera.received_ns == [source_ns, source_ns]
    camera.receive_frame(frameset(2, 1))
    second = camera.read()
    assert second.timestamp_ns == source_ns
    assert (second.color_frame_number, second.depth_frame_number) == (2, 1)
    assert np.all(second.rgb == 2) and np.all(second.depth == 1)
    clock.now_ns += 20_000_000
    camera.receive_frame(frameset(2, 2))
    assert camera.read().timestamp_ns == source_ns + 40_000_000


def test_camera_cache_reads_and_consumer_needs(clock, runtime):
    from dexmani_real.ipc.sensors import SensorChannels

    channels = SensorChannels(mp.get_context("spawn"), camera=CameraConfig(width=4, height=4))
    original = observation(clock.now_ns)
    channels.camera_ring.write(**{k: v for k, v in original.camera.items() if k != "ring_sequence"})
    robot = SimpleNamespace(read_state=lambda: original)
    needs = ObservationNeeds(camera_fields=("rgb", "depth"))
    for age_ns in (0, 100_000_000, 200_000_000):
        clock.now_ns = original.camera["timestamp_ns"] + age_ns
        row = read_observation(channels, robot, needs=needs)
        assert row.camera["timestamp_ns"] == original.camera["timestamp_ns"]
        assert row.camera["ring_sequence"] == 1
        assert sources_available(row, needs)
        issues = check_observation(row, runtime, needs)
        assert issues == (["camera_source_stale"] if age_ns == 200_000_000 else [])
        assert check_observation(row, runtime, ObservationNeeds()) == []
    assert not sources_available(replace(row, camera=None), needs)
    # 没有相机需求时不读取相机通道，即使通道完全不存在也能构造观测。
    assert read_observation(None, robot).camera is None


def test_pointcloud_deduplicates_and_keeps_camera_source(clock, runtime, camera_snapshot, monkeypatch):
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.perception import worker

    channels = SensorChannels(
        mp.get_context("spawn"), camera=CameraConfig(width=4, height=4),
        pointcloud_num_points=2, camera_fields=("rgb", "depth"),
    )
    robot = SimpleNamespace(read_state=lambda: observation(clock.now_ns))
    needs = ObservationNeeds(point_cloud=True, camera_fields=("rgb", "depth"))
    row = read_observation(channels, robot, needs=needs)
    assert row.point_cloud is row.camera is None
    assert check_observation(row, runtime, needs) == ["pointcloud_unavailable"]
    source_ns = clock.now_ns

    def publish_camera(value):
        channels.camera_ring.write(
            timestamp_ns=clock.now_ns,
            rgb=np.full((4, 4, 3), value, np.uint8),
            depth=np.full((4, 4), value, np.uint16),
        )

    def process(depth, rgb):
        np.testing.assert_array_equal(rgb[..., 0], depth)
        assert depth[0, 0] == len(published) + 1 <= 2
        # 模拟计算消耗相机预算；不把这段假时钟耗时当作性能测量。
        clock.now_ns += 150_000_000
        return np.full((2, 6), depth[0, 0], np.float32)

    processor = SimpleNamespace(process=Mock(side_effect=process))
    monkeypatch.setattr(worker, "PointCloudProcessor", Mock(return_value=processor))
    published = []
    write = channels.pointcloud_ring.write

    def capture(**payload):
        published.append(payload)
        return write(**payload)

    monkeypatch.setattr(channels.pointcloud_ring, "write", capture)

    def idle(_):
        # 同一相机序列必须进入等待，而不能再次计算。
        if len(published) == 1:
            publish_camera(2)
        else:
            channels.is_running.value = False

    monkeypatch.setattr(worker.time, "sleep", idle)
    publish_camera(1)
    worker.run_pointcloud_worker(channels, runtime.pointcloud, camera_snapshot, None)
    assert processor.process.call_count == 2
    assert [p["source_camera_sequence"] for p in published] == [1, 2]
    assert [int(p["timestamp_ns"]) for p in published] == [source_ns, source_ns + 150_000_000]
    publish_camera(3)
    row = read_observation(channels, robot, needs=needs)
    assert row.pointcloud_timestamp_ns == row.camera["timestamp_ns"] == source_ns + 150_000_000
    assert row.camera["ring_sequence"] == 2
    assert np.all(row.point_cloud == 2) and np.all(row.camera["rgb"] == 2)
    assert check_observation(row, runtime, needs) == ["pointcloud_stale"]


@pytest.mark.parametrize("invalid_channel", ["aggregate", "dense"])
def test_xhand_crc_and_invalid_tactile_keep_source(clock, runtime, invalid_channel):
    from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand

    raw = SimpleNamespace(
        finger_state=[SimpleNamespace(
            id=i, position=q, commboard_err=0, jonitboard_err=0, tipboard_err=0,
        ) for i, q in enumerate(hand_sample(clock.now_ns).qpos)],
        sensor_data=[SimpleNamespace(
            calc_force=SimpleNamespace(fx=1, fy=2, fz=3),
            raw_force=[SimpleNamespace(fx=4, fy=5, fz=6) for _ in range(120)],
        ) for _ in range(5)],
    )
    reply = SimpleNamespace(error_code=0, error_message="test")

    def read_state(device_id, query):
        assert device_id == runtime.hand.device_id and query is True
        clock.now_ns += 10_000_000
        return reply, raw

    hand = XHand(runtime.hand)
    hand.connected_flag = True
    hand.control = SimpleNamespace(read_state=read_state)
    feedback = XHandFeedback()
    started = clock.now_ns
    initial = feedback.consume(hand.get_state())
    for field in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
        assert initial[field][0] == started
    clock.now_ns += 30_000_000
    reply.error_code = COMMUNICATION_CRC_ERROR_CODE
    assert feedback.consume(hand.get_state()) is initial
    reply.error_code = 0
    if invalid_channel == "aggregate":
        raw.sensor_data[0].calc_force.fx = float("nan")
    else:
        raw.sensor_data[0].raw_force[0].fx = float("nan")
    started = clock.now_ns
    updated = feedback.consume(hand.get_state())
    assert updated["timestamp_ns"][0] == started
    field = f"tactile_{invalid_channel}"
    np.testing.assert_array_equal(updated[field], initial[field])
    np.testing.assert_array_equal(updated[f"{field}_timestamp_ns"], initial[f"{field}_timestamp_ns"])
    other = "dense" if invalid_channel == "aggregate" else "aggregate"
    assert updated[f"tactile_{other}_timestamp_ns"][0] == started
    feedback.clear_tactile()
    cleared = feedback.snapshot()
    assert cleared["timestamp_ns"][0] == started
    for channel in ("aggregate", "dense"):
        assert cleared[f"tactile_{channel}_timestamp_ns"][0] == 0
        assert np.isnan(cleared[f"tactile_{channel}"]).all()


def test_tactile_first_sample_and_runtime_vs_recording_ttl(clock, runtime):
    feedback = XHandFeedback()
    empty = replace(hand_sample(clock.now_ns), tactile_aggregate=None, tactile_dense=None)
    row = replace(observation(clock.now_ns), hand=feedback.consume(empty))
    needs = ObservationNeeds(required_tactile=("aggregate", "dense"))
    assert check_observation(row, runtime, needs, allow_tactile_hold=True) == [
        "hand_tactile_aggregate_unavailable", "hand_tactile_dense_unavailable",
    ]
    feedback.consume(hand_sample(clock.now_ns))
    clock.now_ns += 200_000_000
    row = replace(observation(clock.now_ns), hand=feedback.consume(replace(empty, timestamp_ns=clock.now_ns)))
    assert len(check_observation(row, runtime, needs)) == 2
    assert check_observation(row, runtime, needs, allow_tactile_hold=True) == []
    assert check_observation(row, runtime, ObservationNeeds()) == []


@pytest.mark.parametrize("expiry", ["feedback", "command"])
@pytest.mark.parametrize("stage", ["before", "during", "fresh"])
def test_arm_expiry_blocks_servo(clock, runtime, expiry, stage):
    from dexmani_real.robot.xarm7.protocol import ArmChannels
    from dexmani_real.robot.xarm7.worker import ArmController
    from dexmani_real.runtime.state import SessionState

    state = SessionState()
    io = ArmChannels(mp.get_context("spawn"), stop_signal=state.stop_signal, emergency_signal=state.emergency_signal)
    driver = Mock()
    controller = ArmController(runtime.arm, io, Mock(), driver)
    deadline = clock.now_ns + (160_000_000 if expiry == "feedback" else 250_000_000)
    controller.status["timestamp_ns"] = clock.now_ns if expiry == "feedback" else deadline - 50_000_000
    controller.active = {
        "q_goal": np.asarray(runtime.arm.home_qpos), "hold": False,
        "expires_ns": deadline if expiry == "command" else deadline + 1_000_000_000,
    }

    def trajectory_step(*args, **kwargs):
        clock.now_ns += 10_000_000
        return np.asarray(runtime.arm.home_qpos)

    trajectory = SimpleNamespace(step=Mock(side_effect=trajectory_step))
    controller.reference = trajectory
    clock.now_ns = deadline - {"before": 0, "during": 10_000_000, "fresh": 10_000_001}[stage]
    controller.tick()
    if stage == "fresh":
        driver.servo.assert_called_once()
        assert not state.stop_signal.is_set()
    else:
        driver.servo.assert_not_called()
        assert trajectory.step.call_count == int(stage == "during")
        assert state.stop_signal.is_set()
        assert io.stop_reason.value.decode() == ("arm_feedback_expired" if expiry == "feedback" else "command_expired")
        assert controller.active is controller.reference is None
        controller.tick()
        driver.servo.assert_not_called()


def test_arm_feedback_uses_query_start(clock, runtime):
    from dexmani_real.robot.xarm7.worker import ArmController

    started = clock.now_ns

    def query():
        clock.now_ns += 20_000_000
        return np.asarray(runtime.arm.home_qpos), np.zeros(7)

    controller = ArmController(runtime.arm, Mock(), Mock(), SimpleNamespace(read=query))
    controller.read_feedback()
    assert controller.status["timestamp_ns"] == started


@pytest.mark.parametrize("after_arm_publish", [False, True])
def test_hand_expiry_blocks_new_positions(clock, runtime, after_arm_publish):
    from dexmani_real.robot.commands import FeedbackUnavailable, RobotCommand
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.state import SessionState

    robot = Robot(runtime.arm, runtime.hand, runtime_state=SessionState())
    robot.connected = True
    robot.arm = Mock(is_connected=True)
    robot.hand = Mock(is_connected=True)
    robot.hand_feedback.consume(hand_sample(clock.now_ns))

    def expire(*args, **kwargs):
        clock.now_ns += 200_000_000

    if after_arm_publish:
        robot.arm.submit.side_effect = expire
    else:
        expire()
    with pytest.raises(FeedbackUnavailable, match="hand feedback"):
        robot.send_action(
            RobotCommand(np.asarray(runtime.arm.home_qpos), hand_sample(clock.now_ns).qpos),
            expires_ns=clock.now_ns + 1_000_000_000,
        )
    assert robot.arm.submit.call_count == int(after_arm_publish)
    robot.hand._send_action.assert_not_called()
    assert robot.runtime_state.stop_signal.is_set()


def make_rollout(runtime, needs, clock, *, record_raw=False):
    from dexmani_real.deployment.rollout import Rollout
    from dexmani_real.runtime.state import SessionState

    state = SessionState()
    rollout = Rollout(
        state, runtime, SimpleNamespace(n_obs_steps=2, action_mode="joint"),
        cancel_epoch=state.cancel_epoch, robot=Mock(),
        observations=SimpleNamespace(needs=needs, project=Mock(return_value={"test": np.ones(1)})),
        record_needs=ObservationNeeds(camera_fields=("rgb", "depth")) if record_raw else ObservationNeeds(),
        sensors=Mock(), action_mapper=Mock(),
        model=Mock(future=None), max_running_s=None, index=1, output_dir=None, record_raw=record_raw,
    )
    rollout.model.poll.return_value = None
    rollout.action_mapper.map_joint.return_value = SimpleNamespace(
        arm_qpos=np.asarray(runtime.arm.home_qpos),
        hand_qpos=hand_sample(clock.now_ns).qpos, rejection_reason=None,
    )
    rollout.started_ns = rollout.next_step_ns = clock.now_ns
    return rollout


@pytest.mark.parametrize("modality,needs", [
    ("camera", ObservationNeeds(camera_fields=("rgb",))),
    ("pointcloud", ObservationNeeds(point_cloud=True)),
    ("tactile_aggregate", ObservationNeeds(required_tactile=("aggregate",))),
    ("tactile_dense", ObservationNeeds(required_tactile=("dense",))),
])
def test_history_admission_and_rejection_logging(clock, runtime, caplog, modality, needs):
    rollout = make_rollout(runtime, needs, clock)
    rollout.observations.project.side_effect = lambda row: {"stamp": row.observation_timestamp_ns}
    row = observation(clock.now_ns)
    ttl = runtime.hand.feedback_max_age_s if needs.required_tactile else runtime.camera.max_frame_age_s
    assert rollout.update_history(row)
    clock.now_ns += int(ttl * 1e9) - 1
    assert rollout.update_history(row)
    clock.now_ns += 1
    assert not rollout.update_history(row)
    assert not rollout.history
    assert rollout.observations.project.call_count == 2
    assert len(caplog.records) == 1
    assert modality in caplog.text
    clock.now_ns += 1_000_000
    assert not rollout.update_history(row)
    assert len(caplog.records) == 1
    row = observation(clock.now_ns)
    assert rollout.update_history(row)
    assert list(rollout.history) == [{"stamp": row.observation_timestamp_ns}] * 2
    assert rollout.model_input_issues == ()
    clock.now_ns += int(ttl * 1e9)
    assert not rollout.update_history(row)
    assert len(caplog.records) == 2


@pytest.mark.parametrize("hoi", [False, True])
def test_rollout_stop_publishes_inactive_once(clock, runtime, hoi):
    from dexmani_real.runtime.state import RunEndReason

    rollout = make_rollout(runtime, ObservationNeeds(hoi=hoi), clock)
    channels = rollout.sensors.channels
    channels.camera_ring.latest_sequence = 7
    rollout.stop("duration_timeout", run_end_reason=RunEndReason.DURATION_COMPLETE)
    clock.now_ns += 1
    rollout.stop("repeated stop")
    assert rollout.result.termination_reason == "duration_complete"
    if hoi:
        channels.hoi_request.write.assert_called_once_with(
            episode_id=0, after_ns=clock.now_ns - 1, after_sequence=7,
        )
    else:
        channels.hoi_request.write.assert_not_called()


def test_joint_policy_does_not_require_camera_or_tactile(clock, runtime):
    from dexmani_real.deployment.observation import ObservationBuilder

    builder = ObservationBuilder(SimpleNamespace(observation_fields=("joint_state",)), runtime)
    assert builder.needs == ObservationNeeds()
    sample = replace(hand_sample(clock.now_ns), tactile_aggregate=None, tactile_dense=None)
    state = replace(observation(clock.now_ns), hand=XHandFeedback().consume(sample))
    row = read_observation(None, SimpleNamespace(read_state=lambda: state), needs=builder.needs)
    rollout = make_rollout(runtime, builder.needs, clock)
    rollout.observations = builder
    assert rollout.update_history(row)
    assert builder.build(rollout.history)["joint_state"].shape == (2, 19)


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("modality", ["camera", "tactile"])
def test_stale_inputs_preserve_finite_chunks_without_new_inference(clock, runtime, monkeypatch, mode, modality):
    from dexmani_real.deployment.schedule import ActionChunk
    from dexmani_real.robot.commands import COMMAND_LEASE_NS

    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode))
    needs = (ObservationNeeds(camera_fields=("rgb",)) if modality == "camera"
             else ObservationNeeds(required_tactile=("aggregate",)))
    rollout = make_rollout(runtime, needs, clock)
    assert rollout.update_history(observation(clock.now_ns))
    rollout.first_prediction = False
    rollout.plan = ActionChunk(np.ones((2, 19)), 0)
    source = observation(clock.now_ns - 500_000_000)

    def sample():
        row = observation(clock.now_ns)
        if modality == "camera":
            sample = replace(row, camera=source.camera)
            return sample, sample
        # 关节仍新鲜，只有模型必需的触觉来源过期。
        hand = row.hand.copy()
        hand["tactile_aggregate_timestamp_ns"] = source.hand["tactile_aggregate_timestamp_ns"]
        sample = replace(row, hand=hand)
        return sample, sample

    monkeypatch.setattr(rollout, "sample_observation", sample)
    for _ in range(2):
        rollout.step()
        assert rollout.robot.send_action.call_args.kwargs["expires_ns"] == clock.now_ns + COMMAND_LEASE_NS
        clock.now_ns += rollout.dt_ns
    assert rollout.robot.send_action.call_count == 2
    assert not rollout.history
    rollout.step()
    rollout.model.predict.assert_not_called()
    assert rollout.robot.send_action.call_count == 2
    if mode == "sync":
        assert rollout.plan is None and not rollout.finished
    else:
        assert rollout.finished
        assert rollout.result.termination_details == "prediction exhausted"


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
def test_recovered_prediction_uses_only_new_history(clock, runtime, monkeypatch, mode):
    from dexmani_real.deployment.schedule import ActionChunk

    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode))
    rollout = make_rollout(runtime, ObservationNeeds(camera_fields=("rgb",)), clock)
    rollout.observations.project.side_effect = lambda row: {"stamp": row.camera["timestamp_ns"]}
    old_stamp = clock.now_ns
    assert rollout.update_history(observation(old_stamp))
    rollout.first_prediction = False
    plan = rollout.plan = ActionChunk(np.ones((4, 19)), 0)
    stale = observation(old_stamp - 500_000_000).camera
    monkeypatch.setattr(rollout, "sample_observation", lambda: (
        replace(observation(clock.now_ns), camera=stale), observation(clock.now_ns),
    ))
    rollout.step()
    assert not rollout.history and rollout.plan is plan
    rollout.model.predict.assert_not_called()

    monkeypatch.setattr(rollout, "sample_observation", lambda: (
        observation(clock.now_ns), observation(clock.now_ns),
    ))
    recovered_stamps = []
    for _ in range(3):
        clock.now_ns += rollout.dt_ns
        recovered_stamps.append(clock.now_ns)
        rollout.step()
        if rollout.model.predict.called:
            break
    rollout.model.predict.assert_called_once()
    rows = rollout.model.predict.call_args.args[0]
    expected = ([recovered_stamps[0]] + recovered_stamps)[-2:]
    assert [row["stamp"] for row in rows] == expected
    assert old_stamp not in [row["stamp"] for row in rows]


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
def test_inflight_prediction_keeps_existing_handoff_rules(clock, runtime, monkeypatch, mode):
    from dexmani_real.deployment.schedule import Prediction

    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode))
    rollout = make_rollout(runtime, ObservationNeeds(camera_fields=("rgb",)), clock)
    assert rollout.update_history(observation(clock.now_ns))
    rollout.first_prediction = False
    rollout.query = None if mode == "sync" else Prediction(0, None)
    future = Mock()
    future.result.return_value = np.ones((2, 19))
    rollout.model.future = future
    rollout.model.poll.return_value = future
    row = replace(observation(clock.now_ns), camera=observation(clock.now_ns - 500_000_000).camera)
    monkeypatch.setattr(rollout, "sample_observation", lambda: (row, row))
    rollout.step()
    rollout.robot.send_action.assert_called_once()
    rollout.model.predict.assert_not_called()


def make_teleop(runtime, clock):
    from dexmani_real.ipc.sensors import SensorChannels
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop.loop import TeleopLoop

    channels = SensorChannels(mp.get_context("spawn"), vr=True)
    channels.vr_ring.write(
        wrist_pos=np.zeros(3), wrist_quat_wxyz=np.array([1, 0, 0, 0]),
        landmarks=np.zeros((21, 3)), recv_ts_ns=clock.now_ns, sequence_id=1,
    )
    robot = Mock()
    robot.read_state.side_effect = lambda: observation(clock.now_ns)
    controller = Mock()
    controller.compute_target.return_value = SimpleNamespace(
        arm_qpos=np.asarray(runtime.arm.home_qpos), hand_qpos=hand_sample(clock.now_ns).qpos,
        rejection_reason=None, eef_pose=None,
    )
    keyboard = Mock()
    keyboard.poll.return_value = ([], False)
    state = SessionState()
    loop = TeleopLoop(
        state, runtime, robot, task_label="freshness", sensors=SimpleNamespace(channels=channels),
        controller=controller, recording_dir=None, max_duration_s=None,
        keyboard=keyboard, audio=Mock(),
    )
    robot.stop.side_effect = state.stop_signal.set
    loop.cancel_epoch = state.cancel_epoch
    loop.try_begin()
    return loop


def test_vr_repeated_sample_renews_targets_until_source_expires(clock, runtime):
    from dexmani_real.teleop.loop import TeleopPhase

    loop = make_teleop(runtime, clock)
    period_ns = round(1e9 / runtime.teleop.control_hz)
    for tick in range(8):
        loop.run_control_tick()
        assert loop.pending_termination_reason is None
        assert loop.target_deadline_ns == clock.now_ns + 250_000_000
        assert loop.robot.send_action.call_count == tick + 1
        clock.now_ns += period_ns
    # 500 ms 时目标租约尚未到期，但同一 VR 来源已失效，不能再产生目标。
    assert clock.now_ns < loop.target_deadline_ns
    loop.run_control_tick()
    assert loop.robot.send_action.call_count == loop.controller.compute_target.call_count == 8
    assert loop.runtime_state.stop_signal.is_set()
    assert loop.pending_termination_reason is None
    assert loop.phase is TeleopPhase.PAUSED
    loop.robot.stop.assert_called_once()
    assert loop.last_command is None


@pytest.mark.parametrize("expiry", ["source", "target"])
def test_vr_expiry_during_mapping_prevents_submission(clock, runtime, expiry):
    loop = make_teleop(runtime, clock)
    if expiry == "source":
        ring = loop.sensors.channels.vr_ring
        payload = ring.read()[0]
        ring.write(**{name: (clock.now_ns - 490_000_000 if name == "recv_ts_ns" else payload[name])
                      for name in ring.field_names})
        compute_ns = 10_000_000
    else:
        loop.run_control_tick()
        clock.now_ns += 200_000_000
        compute_ns = 50_000_000
    candidate = loop.controller.compute_target.return_value

    def compute(*args, **kwargs):
        clock.now_ns += compute_ns
        return candidate

    loop.controller.compute_target.side_effect = compute
    before = loop.robot.send_action.call_count
    loop.run_control_tick()
    assert loop.robot.send_action.call_count == before
    from dexmani_real.teleop.loop import TeleopPhase
    assert loop.pending_termination_reason is None
    assert loop.phase is TeleopPhase.PAUSED
    assert loop.last_command is None


def test_vr_rejected_targets_do_not_extend_lease(clock, runtime):
    loop = make_teleop(runtime, clock)
    loop.run_control_tick()
    deadline = loop.target_deadline_ns
    loop.controller.compute_target.return_value = SimpleNamespace(
        arm_qpos=None, rejection_reason="geometry_rejected",
    )
    for _ in range(3):
        clock.now_ns += 62_500_000
        loop.run_control_tick()
        assert loop.target_deadline_ns == deadline
        assert loop.robot.send_action.call_args.kwargs["expires_ns"] == deadline
    assert loop.robot.send_action.call_count == 4
    clock.now_ns = deadline
    loop.run_control_tick()
    from dexmani_real.teleop.loop import TeleopPhase
    assert loop.pending_termination_reason is None
    assert loop.phase is TeleopPhase.PAUSED
    assert loop.target_deadline_ns is None
    assert loop.robot.send_action.call_count == 4


def test_keyboard_jog_braking_and_hold_renew_250ms_lease(clock, runtime):
    from dexmani_real.runtime.jog import CartesianJog
    from dexmani_real.runtime.state import SessionState

    limits = np.array([runtime.arm.joint_limit_lower, runtime.arm.joint_limit_upper]).T
    jog = CartesianJog(SimpleNamespace(limits=limits), runtime.keyboard_teleop, (np.zeros(3), np.ones(3)))
    jog.propose = Mock(return_value=SimpleNamespace(qpos=np.asarray(runtime.arm.home_qpos), reason=None))
    robot = Mock()
    robot.read_state.side_effect = lambda: observation(clock.now_ns)
    keys = Mock()
    state = SessionState()
    deadline = clock.now_ns + 250_000_000
    for motion, rejected in ((True, False), (True, False), (False, False), (False, False), (True, True)):
        clock.now_ns += round(1e9 / runtime.keyboard_teleop.control_hz)
        assert clock.now_ns < deadline
        keys.snapshot.return_value = {"w"} if motion else set()
        if rejected:
            jog.propose.return_value = SimpleNamespace(qpos=None, reason="geometry_rejected")
        jog.step(robot, keys, state, runtime)
        command = robot.send_action.call_args.args[0]
        assert command.arm_hold is (not motion or rejected)
        deadline = robot.send_action.call_args.kwargs["expires_ns"]
        assert deadline == clock.now_ns + 250_000_000
    assert not state.stop_signal.is_set()


def test_recording_and_export_keep_held_tactile_times(clock, runtime, camera_snapshot, tmp_path, monkeypatch, caplog):
    import h5py

    from dexmani_real.data.export import validate_export_episode
    from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
    from dexmani_real.data.recorder import EpisodeRecorder

    # 只替换视频编码，实际走 recorder 快照、Raw 编码和 HDF5 写出。
    monkeypatch.setattr("dexmani_real.data.raw._VideoWriter", Mock())
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.is_recording = True
    source_ns = clock.now_ns
    feedback = XHandFeedback()
    initial = feedback.consume(hand_sample(source_ns))
    needs = ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
    for age in (600_000_000, 650_000_000):
        clock.now_ns = source_ns + age
        hand = feedback.consume(replace(
            hand_sample(clock.now_ns), tactile_aggregate=None, tactile_dense=None,
        ))
        row = replace(observation(clock.now_ns), hand=hand)
        assert check_observation(row, runtime, needs, allow_tactile_hold=True) == []
        assert recorder.append(row) is None
    warnings = [r for r in caplog.records if r.name == "dexmani_real.data.recorder"]
    assert len(warnings) == 2  # 每个通道只告警一次。
    assert recorder.recording_truncated_reason is None
    metadata = dict(
        snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}),
        collection_source="policy", task_label="freshness",
    )
    writer = RawWriter(tmp_path, control_hz=20)
    try:
        writer.open(metadata)
        for _ in range(2):
            writer.append_frame(*recorder.queue.get_nowait())
        writer.finish("test", [])
    finally:
        writer.close()
    with RawReader(tmp_path) as reader:
        timing = reader["timing/rows"][:]
        assert timing["tactile_aggregate_ns"].tolist() == [source_ns, source_ns]
        assert timing["tactile_dense_ns"].tolist() == [source_ns, source_ns]
        assert timing["hand_ns"].tolist() == [source_ns + 600_000_000, source_ns + 650_000_000]
        for field, raw_name in (("tactile_aggregate", "hand_contact"), ("tactile_dense", "hand_tactile_force")):
            np.testing.assert_array_equal(reader[raw_name][:], np.repeat(initial[field], 2, axis=0))
        validate_export_episode(reader, ("finger_tactile_force", "taxel_tactile_force"))
        np.testing.assert_array_equal(reader["timing/rows"][:], timing)
    assert "held beyond 0.5s" in caplog.text
    # 外部 Raw 的无效来源只能诊断；不能把从未有效的触觉误报为 Hold。
    caplog.clear()
    with h5py.File(tmp_path / "data.h5", "r+") as data:
        timing["tactile_aggregate_ns"] = 0
        timing["tactile_dense_ns"] = -1
        timing["camera_ns"] = 0
        timing["camera_sequence"] = 0
        data["timing/rows"][:] = timing
    with RawReader(tmp_path) as reader:
        validate_export_episode(reader, ("finger_tactile_force", "taxel_tactile_force", "depth"))
        np.testing.assert_array_equal(reader["timing/rows"][:], timing)
    for field in ("tactile_aggregate_ns", "tactile_dense_ns", "camera_ns", "camera_sequence"):
        assert f"{field} invalid/regressed at row 0" in caplog.text
    assert "held beyond" not in caplog.text


@pytest.mark.parametrize("stale_device", [None, "arm", "hand"])
def test_replay_start_uses_current_feedback_budgets(clock, runtime, monkeypatch, stale_device):
    from dexmani_real.replay import run as replay
    from dexmani_real.replay.trajectory import Trajectory
    from dexmani_real.robot.commands import RobotError

    row = observation(clock.now_ns)
    if stale_device is not None:
        frame = getattr(row, stale_device).copy()
        frame["timestamp_ns"] = clock.now_ns - int(getattr(runtime, stale_device).feedback_max_age_s * 1e9)
        row = replace(row, **{stale_device: frame})
    trajectory = Trajectory(20, row.arm["qpos"].copy(), row.hand["qpos"].copy())
    robot, keyboard = Mock(), Mock()
    robot.read_state.return_value = row
    keyboard.poll.return_value = ([], False)
    monkeypatch.setattr(replay, "load_trajectory", lambda *args, **kwargs: trajectory)
    monkeypatch.setattr(replay, "Robot", Mock(return_value=robot))
    monkeypatch.setattr(replay, "Keyboard", Mock(return_value=keyboard))

    def advance(seconds):
        clock.now_ns += round(seconds * 1e9)

    monkeypatch.setattr(replay.time, "sleep", advance)
    if stale_device is None:
        replay.run_replay("unused", runtime)
        robot.send_action.assert_called_once()
    else:
        with pytest.raises(RobotError, match=f"replay start feedback: {stale_device}_"):
            replay.run_replay("unused", runtime)
        robot.send_action.assert_not_called()
    robot.close.assert_called_once()
    keyboard.stop.assert_called_once()


@pytest.mark.parametrize("case", ["fresh", "slow", "crc", "action_deadline"])
def test_xhand_tool_uses_one_read_deadline(clock, runtime, monkeypatch, case):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "control_xhand", Path(__file__).resolve().parents[1] / "scripts/tools/control_xhand.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hand = module.XHandDiagnostic(runtime.hand)
    budget_ns = int(runtime.hand.feedback_max_age_s * 1e9)
    duration_ns = {"fresh": budget_ns - 1, "slow": budget_ns, "crc": budget_ns // 2,
                   "action_deadline": 50_000_000}[case]

    def query():
        started = clock.now_ns
        clock.now_ns += duration_ns
        return None if case == "crc" else hand_sample(started)

    def advance(seconds):
        clock.now_ns += round(seconds * 1e9)

    hand.get_state = Mock(side_effect=query)
    monkeypatch.setattr(module.time, "sleep", advance)
    deadline = clock.now_ns + duration_ns if case == "action_deadline" else None
    if case == "fresh":
        assert hand.read_fresh_state().timestamp_ns == clock.now_ns - duration_ns
    else:
        with pytest.raises(TimeoutError, match="deadline expired"):
            hand.read_fresh_state(action_deadline_ns=deadline)
    assert hand.get_state.call_count == (2 if case == "crc" else 1)


@pytest.mark.parametrize('valid', [True, False])
def test_hoi_initial_observation_finishes_immediately(clock, runtime, monkeypatch, valid):
    rollout = make_rollout(runtime, ObservationNeeds(hoi=True), clock)
    row = replace(observation(clock.now_ns), hoi={'valid': valid, 'timestamp_ns': clock.now_ns})
    monkeypatch.setattr('dexmani_real.deployment.rollout.read_observation', lambda *a, **k: row)
    monkeypatch.setattr('time.sleep', Mock(side_effect=AssertionError('明确结果不应继续等待')))
    assert (rollout.start_observation() is not None) == valid
    if not valid:
        assert 'hoi_pose_invalid' in rollout.preparation_issue


@pytest.mark.parametrize('initial_hoi,budget', [(False, 1), (True, 120)])
def test_hoi_observation_wait_budget(clock, runtime, monkeypatch, initial_hoi, budget):
    rollout = make_rollout(runtime, ObservationNeeds(hoi=True), clock)
    monkeypatch.setattr('dexmani_real.deployment.rollout.read_observation',
                        lambda *a, **k: observation(clock.now_ns))
    start = clock.now_ns
    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)
    monkeypatch.setattr('time.sleep', sleep)
    assert rollout.start_observation(initial_hoi=initial_hoi) is None
    assert clock.now_ns - start == budget * 1_000_000_000


@pytest.mark.parametrize('age,valid,expected', [
    (180_000_000, True, []), (250_000_000, True, ['hoi_stale']),
    (260_000_000, True, ['hoi_stale']), (20_000_000, False, ['hoi_pose_invalid']),
    (250_000_000, False, ['hoi_pose_invalid', 'hoi_stale']),
])
def test_separate_hoi_age_budget(clock, runtime, age, valid, expected):
    stamp = clock.now_ns - age
    row = replace(observation(stamp), hoi={'valid': valid, 'timestamp_ns': stamp})
    assert check_observation(row, runtime, ObservationNeeds(hoi=True)) == expected
    if age >= 180_000_000:
        assert check_observation(row, runtime, ObservationNeeds(camera_fields=('rgb',))) == ['camera_source_stale']
        assert check_observation(row, runtime, ObservationNeeds(point_cloud=True)) == ['pointcloud_stale']
    overridden = replace(runtime, execution=replace(runtime.execution, hoi_max_age_s=.1))
    if valid:
        assert check_observation(row, overridden, ObservationNeeds(hoi=True)) == ['hoi_stale']


@pytest.mark.parametrize("hoi", [False, True])
@pytest.mark.parametrize('age', [0, -1, float('nan'), float('inf'), True, None, '0.25'])
def test_invalid_hoi_age_config(runtime, age, hoi):
    from dexmani_real.deployment.preflight import validate_execution_config

    info = SimpleNamespace(future_steps=4, observation_fields=("object_pointcloud",) if hoi else ("joint_state",))
    execution = replace(runtime.execution, hoi_max_age_s=age)
    if hoi:
        with pytest.raises(ValueError, match='hoi_max_age_s'):
            validate_execution_config(info, execution)
    else:
        validate_execution_config(info, execution)


@pytest.mark.parametrize('hoi,images,record', [
    (True, False, True), (True, True, True), (True, False, False),
    (True, True, False), (False, True, True), (False, True, False),
])
def test_model_and_raw_camera_sources(clock, runtime, monkeypatch, tmp_path, camera_snapshot, hoi, images, record):
    from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
    from dexmani_real.data.recorder import EpisodeRecorder
    from dexmani_real.ipc.sensors import SensorChannels
    fields = ('rgb', 'depth') if images else ()
    needs = ObservationNeeds(hoi=hoi, camera_fields=fields)
    rollout = make_rollout(runtime, needs, clock, record_raw=record)
    channels = SensorChannels(mp.get_context('spawn'), camera=CameraConfig(width=4, height=4),
                              hoi=hoi, camera_fields=fields)
    rollout.sensors.channels = channels
    rollout.robot.read_state.side_effect = lambda: observation(clock.now_ns)
    source = clock.now_ns - 180_000_000
    def pixels(value):
        return dict(rgb=np.full((4, 4, 3), value, np.uint8), depth=np.full((4, 4), value, np.uint16))
    if hoi:
        channels.hoi_request.write(episode_id=1, after_ns=1, after_sequence=0)
        channels.hoi_ring.write(episode_id=1, timestamp_ns=source, source_camera_sequence=10,
            valid=True, object_pointcloud=np.ones((1024, 3), np.float32),
            **{k: v for k, v in pixels(10).items() if k in fields})
    for _ in range(12):
        channels.camera_ring.write(timestamp_ns=clock.now_ns - 20_000_000, **pixels(12))
    read = channels.camera_ring.read
    completed = []
    def camera_read(**kwargs):
        result = read(**kwargs)
        clock.now_ns += 1_000_000
        completed.append(clock.now_ns)
        clock.now_ns += 1
        return result
    spy = Mock(side_effect=camera_read)
    monkeypatch.setattr(channels.camera_ring, 'read', spy)
    row, raw = rollout.read_observations()
    assert spy.call_count == int(record or not hoi)
    assert check_observation(row, runtime, needs) == []
    if hoi:
        assert row.hoi['timestamp_ns'] == source
        assert row.hoi['source_camera_sequence'] == 10
        assert set(channels.hoi_ring.dtype.names) & {'rgb', 'depth'} == set(fields)
    if images:
        assert row.camera['ring_sequence'] == (10 if hoi else 12)
        assert np.all(row.camera['rgb'] == (10 if hoi else 12))
    else:
        assert row.camera is None
    if not record:
        assert raw is row
        return
    assert raw.camera['ring_sequence'] == 12
    assert raw.observation_timestamp_ns > completed[-1]
    assert raw.observation_timestamp_ns >= raw.camera['timestamp_ns']
    assert rollout.recording_issue(raw) is None
    assert raw.hoi is None
    monkeypatch.setattr('dexmani_real.data.raw._VideoWriter', Mock())
    recorder = EpisodeRecorder(tmp_path, control_hz=20)
    recorder.is_recording = True
    assert recorder.append(raw) is None
    writer = RawWriter(tmp_path, control_hz=20)
    try:
        writer.open(dict(snapshot_recording_metadata(camera_snapshot, tactile_reset_metadata={}),
                         collection_source='policy', task_label='hoi'))
        writer.append_frame(*recorder.queue.get_nowait())
        writer.finish('test', [])
    finally:
        writer.close()
    with RawReader(tmp_path) as reader:
        timing = reader['timing/rows'][0]
        assert timing['camera_sequence'] == 12
        assert timing['camera_ns'] == raw.camera['timestamp_ns']
        assert np.all(reader['depth'][0] == 12)


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
@pytest.mark.parametrize('stage', ['between_ticks', 'mapping'])
def test_hoi_loss_stops_before_submission(clock, runtime, monkeypatch, mode, stage):
    from dexmani_real.deployment.schedule import ActionChunk
    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode))
    rollout = make_rollout(runtime, ObservationNeeds(hoi=True), clock)
    row = replace(observation(clock.now_ns), hoi={'valid': True, 'timestamp_ns': clock.now_ns})
    latest = replace(row, hoi=dict(row.hoi))
    rollout.plan = ActionChunk(np.ones((2, 19)), 0)
    rollout.first_prediction = False
    monkeypatch.setattr('dexmani_real.deployment.rollout.read_observation', lambda *a, **k: latest)
    monkeypatch.setattr(rollout, 'sample_observation', lambda: (row, row))
    if stage == 'between_ticks':
        rollout.next_step_ns += rollout.dt_ns
        latest.hoi['valid'] = False
    else:
        map_command = rollout.map_command
        def mapping(action, sample):
            result = map_command(action, sample)
            latest.hoi['valid'] = False
            return result
        monkeypatch.setattr(rollout, 'map_command', mapping)
    rollout.step()
    assert rollout.finished
    assert rollout.plan is rollout.query is None
    rollout.robot.send_action.assert_not_called()


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
@pytest.mark.parametrize('raw_stale', [False, True])
def test_rollout_records_independent_raw_snapshot(clock, runtime, monkeypatch, mode, raw_stale):
    from dexmani_real.deployment.schedule import ActionChunk
    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode))
    rollout = make_rollout(runtime, ObservationNeeds(hoi=True), clock, record_raw=True)
    model = replace(observation(clock.now_ns), camera=None,
                    hoi={'valid': True, 'timestamp_ns': clock.now_ns - 180_000_000})
    raw = observation(clock.now_ns - (180_000_000 if raw_stale else 20_000_000))
    raw = replace(raw, observation_timestamp_ns=clock.now_ns)
    monkeypatch.setattr(rollout, 'read_observations', lambda: (model, raw))
    monkeypatch.setattr('dexmani_real.deployment.rollout.read_observation', lambda *a, **k: model)
    rollout.recorder = Mock()
    rollout.recorder.append.return_value = None
    rollout.first_prediction = False
    rollout.plan = ActionChunk(np.ones((2, 19)), 0)
    rollout.step()
    rollout.recorder.append.assert_called_once()
    assert rollout.recorder.append.call_args.args[0] is raw
    assert not rollout.finished
    rollout.robot.send_action.assert_called_once()
    assert rollout.recorder.append.call_args.args[1] is rollout.robot.send_action.call_args.args[0]
    assert rollout.recorder.append.call_args.kwargs["quality_issue"] == (
        "camera_source_stale" if raw_stale else None
    )
