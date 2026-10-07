"""Offline device-fault, data-integrity and calibration regressions; device I/O is fake."""

import json
import threading
from concurrent.futures import CancelledError
from fractions import Fraction
from types import SimpleNamespace as NS

import av
import numpy as np
import pytest

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import (
    DexManiRobot,
    DispatchError,
    DispatchInterrupted,
    DispatchStatus,
)
from dexmani_real.runtime.safety import RunEndReason, SafetyState


def shared_state():
    return NS(
        motion_lock=threading.RLock(),
        **{
            k: NS(value=v)
            for k, v in dict(
                run_id=1,
                run_ended_id=0,
                run_ended_reason=0,
                is_running=True,
                error_state=False,
                estop_request=False,
                quit_requested=False,
                stop_request=0,
                safety_state=int(SafetyState.RUNNING),
            ).items()
        },
    )


def fake_robot(monkeypatch):
    shared, runtime = shared_state(), ExperimentConfig()
    robot = DexManiRobot(shared, runtime)
    clock = NS(now=100, mode_delay=0, arm_delay=0, hand_delay=0, interrupt=None, calls=[])
    monkeypatch.setattr("time.monotonic_ns", lambda: clock.now)
    robot._connected = True
    robot._owner = threading.get_ident()

    def call(name, delay):
        clock.calls.append(name)
        clock.now += delay
        if clock.interrupt == name:
            raise KeyboardInterrupt()
        if name == "arm" and clock.interrupt == "revoke":
            shared.run_id.value += 1
        return 0 if name == "arm" else DispatchStatus.ACCEPTED

    robot.arm = NS(
        is_connected=True,
        servo=lambda q: call("arm", clock.arm_delay),
        enter_mode6=lambda: call("mode", clock.mode_delay),
    )
    robot.hand = NS(is_connected=True, send_action=lambda q: call("hand", clock.hand_delay))
    command = RobotCommand(
        1, np.array(runtime.arm.home_qpos), np.deg2rad(runtime.hand.home_qpos_deg)
    )
    return robot, clock, command


@pytest.mark.parametrize("where,expected", [("before", (0, 0)), ("arm", (4, 0)), ("hand", (1, 4))])
def test_dispatch_interrupt_evidence(monkeypatch, where, expected):
    robot, clock, command = fake_robot(monkeypatch)
    clock.interrupt = where
    if where == "before":

        def check():
            raise KeyboardInterrupt()

        robot.check_services = check
    with pytest.raises(DispatchInterrupted) as caught:
        robot.send_action(command, valid_until_ns=200)
    result = caught.value.result
    assert (result.arm, result.hand) == expected
    assert result.timestamp_ns == clock.now


@pytest.mark.parametrize(
    "stage,expected_calls,expected_status",
    [
        ("at_deadline", [], (0, 0)),
        ("service", [], (0, 0)),
        ("mode", ["mode"], (0, 0)),
        ("arm", ["arm"], (1, 0)),
        ("revoke", ["arm"], (1, 0)),
    ],
)
def test_dispatch_deadline_at_last_boundary(monkeypatch, stage, expected_calls, expected_status):
    robot, clock, command = fake_robot(monkeypatch)
    if stage == "at_deadline":
        clock.now = 200
    if stage == "service":

        def check():
            clock.now += 500
            return True

        robot.check_services = check
    if stage == "mode":
        robot._arm_stopped, clock.mode_delay = True, 100
    if stage == "arm":
        clock.arm_delay = 100
    if stage == "revoke":
        clock.interrupt = "revoke"
    with pytest.raises(DispatchError) as caught:
        robot.send_action(command, valid_until_ns=200)
    assert clock.calls == expected_calls
    assert (caught.value.result.arm, caught.value.result.hand) == expected_status


def test_last_sdk_late_return_keeps_accepted(monkeypatch):
    robot, clock, command = fake_robot(monkeypatch)
    clock.hand_delay = 200
    result = robot.send_action(command, valid_until_ns=200)
    assert (result.arm, result.hand) == (1, 1)
    assert result.timestamp_ns == 300


def test_hand_home_uses_armed_deadline(monkeypatch):
    robot, clock, command = fake_robot(monkeypatch)
    robot.shared.safety_state.value = int(SafetyState.ARMED)
    command = RobotCommand(1, hand_qpos=command.hand_qpos)
    result = robot.send_hand_home(command, valid_until_ns=200)
    assert result.hand == 1 and clock.calls == ["hand"]
    clock.now = 200
    with pytest.raises(DispatchError):
        robot.send_hand_home(command, valid_until_ns=200)
    assert clock.calls == ["hand"]


@pytest.mark.parametrize("available", [(True, False), (False, True), (True, True), (False, False)])
def test_tactile_independent_candidates(monkeypatch, available):
    from dexmani_real.robot.drivers.xhand import XHand, _tactile_validity

    monkeypatch.setattr("time.sleep", lambda _: None)
    hand = XHand(ExperimentConfig().hand)
    raw = NS(
        tactile_aggregate=np.ones((5, 3)),
        tactile_dense=np.ones((5, 120, 3)),
        tactile_aggregate_valid=available[0],
        tactile_dense_valid=available[1],
    )
    hand._read_state = lambda **kw: raw
    assert hand.tare_tactile() == available
    assert (
        hand._tactile_bias_aggregate is not None,
        hand._tactile_bias_dense is not None,
    ) == available
    assert _tactile_validity(1501019, comm_type="rs485") == (True, False)


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_tactile_verify_never_exposes_candidate(monkeypatch, error):
    from dexmani_real.robot.drivers.xhand import XHand

    hand = XHand(ExperimentConfig().hand)
    hand._tactile_bias_aggregate = np.ones((5, 3))
    hand._tactile_bias_dense = np.ones((5, 120, 3))
    hand._capture_tactile_bias = lambda **kwargs: (np.zeros((5, 3)), np.zeros((5, 120, 3)))

    def verify(*args, **kwargs):
        assert hand._tactile_bias_aggregate is hand._tactile_bias_dense is None
        raise error()

    hand._verify_tactile_bias = verify
    with pytest.raises(error):
        hand.tare_tactile()
    assert hand._tactile_bias_aggregate is hand._tactile_bias_dense is None


def test_absolute_hw_distance():
    from dexmani_real.planning.kinematics.ik import OnlineIKSolver

    solver = OnlineIKSolver.__new__(OnlineIKSolver)
    solver.operational_limits = np.tile([-np.pi, np.pi], (7, 1))
    solver._jump_limit = np.full(7, np.deg2rad(20))

    def q(angle):
        return np.deg2rad([angle, 0, 0, 0, 0, 0, 0])

    assert solver.dynamic_rejection(q(106), q(-106), q(101)) == "hw_dist"
    assert solver.dynamic_rejection(q(106), q(100), q(101)) is None
    assert solver.dynamic_rejection(q(179), q(179), q(-179)) == "jump"


@pytest.mark.parametrize(
    "clearance,safe",
    [
        ([-0.001 - 0.0004 * i for i in range(16)] + [0], False),
        ([-0.001, -0.0014, -0.0005, -0.0008, 0], True),
        ([-0.001, 0, -0.0001, 0], False),
        ([-0.001, -0.0005], False),
    ],
)
def test_soft_escape_against_best(monkeypatch, clearance, safe):
    from dexmani_real.planning import paths

    planner = NS(
        planning_profile=NS(check_self_collision=False), is_workspace_segment_safe=lambda a, b: True
    )
    monkeypatch.setattr(
        paths, "_table_clearance_m", lambda p, q, **kw: (clearance[int(q[0])], 0, "fake")
    )
    path = np.zeros((len(clearance), 7))
    path[:, 0] = np.arange(len(clearance))
    result = paths._check_home_path_candidate(
        path,
        "fake",
        planner,
        table_z_surface_m=0,
        hand_safety_margin_m=0,
        allow_table_soft_escape=True,
    )
    assert result.safe == safe


def test_atomic_cancel_and_commit(tmp_path, monkeypatch):
    from dexmani_real.utils import atomic_io

    path = tmp_path / "calibration.json"
    path.write_text('{"old": true}')
    with pytest.raises(CancelledError):
        atomic_io.atomic_json_dump({"new": True}, path, cancelled=lambda: True)
    assert json.loads(path.read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [path]
    cancelled = NS(value=False)
    original = atomic_io.os.replace

    def replace(source, destination):
        cancelled.value = True
        original(source, destination)

    monkeypatch.setattr(atomic_io.os, "replace", replace)
    atomic_io.atomic_json_dump({"new": True}, path, cancelled=lambda: cancelled.value)
    assert json.loads(path.read_text()) == {"new": True}


def test_handeye_selects_qualified_candidate(monkeypatch):
    from dexmani_real.calibration.camera import solver

    monkeypatch.setattr(solver, "_HAND_EYE_METHODS", {"A": 0, "B": 1})

    def candidate(*args, method):
        t = np.eye(4)
        t[0, 3] = method
        return t

    monkeypatch.setattr(solver, "_calibrate_eye_to_hand", candidate)
    monkeypatch.setattr(
        solver,
        "_compute_closed_loop_errors",
        lambda t, *args: (
            (np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]))
            if t[0, 3] == 0
            else (np.array([2.0, 2.0, 2.0]), np.array([1.0, 1.0, 1.0]))
        ),
    )
    result = solver.calibrate_and_select(
        np.zeros((3, 3)),
        np.array([[0, 0, 0], [0.3, 0, 0], [0, 0.4, 0]]),
        np.zeros((3, 3)),
        np.zeros((3, 3)),
        max_position_rms_mm=5,
        max_rotation_rms_deg=3,
    )
    assert result[1] == "B"


def test_table_final_refit_support(monkeypatch):
    from dexmani_real.calibration import table

    rng = np.random.default_rng(7)
    points = np.column_stack((rng.uniform(-1, 1, (1500, 2)), np.zeros(1500)))
    calls = []

    def refit(points):
        calls.append(len(points))
        return np.array([0.0, 0.0, 1.0]), 0.02 if len(calls) == 3 else 0.0

    monkeypatch.setattr(table, "_least_squares_plane", refit)
    with pytest.raises(ValueError, match="final.*support"):
        table.fit_table_plane(points, min_inlier_ratio=0.561)


@pytest.mark.parametrize(
    "rate,origin,preset",
    [
        (Fraction(30), 0, "ultrafast"),
        (Fraction(30000, 1001), 0, "medium"),
        (Fraction(30), 15, "medium"),
    ],
)
def test_native_cfr_random_matches_sequential(tmp_path, rate, origin, preset):
    from dexmani_real.recording.storage.video import VideoDecoder

    path = tmp_path / "test.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv444p"
        stream.options = {"preset": preset, "g": "12"}
        for i in range(140):
            pixels = np.full((16, 32, 3), i, np.uint8)
            pixels[:, i % 32, 0] = 255
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = origin + i, 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    with VideoDecoder(path) as decoder:
        oracle = list(decoder.iter_frames())
        for index in [123, 0, 139, 12, 11, 13, 123, 50, 49, 1]:
            np.testing.assert_array_equal(decoder.read_frame(index), oracle[index])


@pytest.mark.parametrize("frozen", ["rgb", "depth", "both", "neither"])
def test_camera_oldest_channel_clock(monkeypatch, frozen):
    from dexmani_real.config.hardware import CameraParams
    from dexmani_real.sensor.camera import realsense, worker

    shared = NS(
        is_running=NS(value=True),
        camera_depth_scale=NS(value=0),
        camera_serial=NS(value=b""),
        camera_geometry=NS(value=b""),
        camera_ready=threading.Event(),
    )
    frames = []
    clock = NS(now=1_000_000_000, index=0, closed=False)
    monkeypatch.setattr("time.monotonic_ns", lambda: clock.now)

    class Camera:
        active_serial = "fake"

        def __init__(self, cfg):
            pass

        def connect(self):
            return True

        def get_depth_scale(self):
            return 0.001

        def get_geometry(self):
            return NS(to_dict=lambda: {})

        def disconnect(self):
            clock.closed = True

        def read(self, **kwargs):
            clock.now += 100_000_000
            clock.index += 1
            if clock.index == 24:
                shared.is_running.value = False
            return NS(
                depth_frame_number=1 if frozen in ("depth", "both") else clock.index,
                color_frame_number=1 if frozen in ("rgb", "both") else clock.index,
                timestamp_ns=clock.now,
                rgb=np.zeros((2, 2, 3), np.uint8),
                depth_aligned_to_color_raw=np.zeros((2, 2), np.uint16),
            )

    shared.camera_ring = NS(
        write=lambda header, *args: frames.append((clock.now, int(header["timestamp_ns"][0])))
    )
    monkeypatch.setattr(realsense, "RealSenseCamera", Camera)
    if frozen == "neither":
        worker.run_camera_worker(shared, CameraParams())
    else:
        with pytest.raises(RuntimeError, match="stopped producing"):
            worker.run_camera_worker(shared, CameraParams())
    assert clock.closed
    if frozen in ("rgb", "depth"):
        assert len(frames) > 2 and frames[-1][1] == frames[0][1]
        assert frames[-1][0] - frames[-1][1] > 1_000_000_000
    elif frozen == "both":
        assert len(frames) == 1
    else:
        assert all(now == source for now, source in frames)


def test_calibration_capture_cancel_is_prompt(monkeypatch):
    from dexmani_real.calibration.camera import session

    clock = NS(now=0.0)
    monkeypatch.setattr(session.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(session.time, "sleep", lambda dt: setattr(clock, "now", clock.now + dt))
    monkeypatch.setattr(session, "read_camera_frame", lambda _: None)
    with pytest.raises(CancelledError):
        session._detect_aruco_stable(
            None,
            np.eye(3),
            np.zeros(5),
            marker_size_m=0.1,
            target_id=1,
            max_frame_age_s=0.1,
            cancel_requested=lambda: clock.now >= 0.01,
        )
    assert clock.now == 0.01


@pytest.mark.parametrize(
    "field,value",
    [("control_hz", 0), ("idle_interval_frames", 0), ("workspace_command_margin_m", 1.0)],
)
def test_invalid_calibration_config_before_connect(monkeypatch, field, value):
    from dataclasses import replace

    from dexmani_real.calibration.camera import session

    cfg = ExperimentConfig()
    cfg = replace(
        cfg,
        policy=replace(cfg.policy, hand_enabled=False),
        keyboard_teleop=replace(cfg.keyboard_teleop, **{field: value}),
    )

    def connect(_):
        raise AssertionError("connected hardware")

    monkeypatch.setattr(session.DexManiRobot, "connect", connect)
    with pytest.raises(ValueError):
        session.run_camera_calibration(cfg, hand_geometry="absent")


def test_calibration_cancelled_queue_never_solves(monkeypatch):
    from dexmani_real.calibration.camera import session

    instance = session.CameraCalibrationSession.__new__(session.CameraCalibrationSession)
    instance.shared = shared_state()
    instance.shared.quit_requested.value = True
    instance.keys = NS(pop_event=lambda: "enter", healthy=True, is_pressed=lambda _: False)
    instance.camera_process = NS(is_alive=lambda: True)
    instance.robot = NS(check=lambda: None, stop=lambda: pytest.fail("queued solve stopped robot"))
    instance._handle_sample_events()


@pytest.mark.parametrize("stop_fails", [False, True])
def test_teleop_cancel_records_once_before_propagation(monkeypatch, stop_fails):
    from dexmani_real.teleop.control import controller

    robot, clock, command = fake_robot(monkeypatch)
    clock.interrupt = "hand"
    recorded, stops = [], []
    details = []

    def stop():
        stops.append(True)
        if stop_fails:
            raise TimeoutError("STOP_TIMEOUT")

    robot.stop = stop
    from dexmani_real.robot.action import ActionRealization

    ctl = NS(
        runtime=robot.runtime,
        compute_target=lambda *a: ActionRealization(
            command.arm_qpos, command.hand_qpos, eef_pose=np.zeros(9)
        ),
        commit_dispatched_target=lambda target: None,
    )
    row = NS(arm={"timestamp_ns": [100]}, hand={"timestamp_ns": [100]}, vr={"recv_ts_ns": 100})
    recorder = NS(check_error=lambda: None, accepting_frames=True, add_frame=recorded.append)
    monkeypatch.setattr(controller, "build_episode_frame", lambda row, cmd, res: res)
    with pytest.raises(DispatchInterrupted):
        controller.execute_control_step(
            ctl, robot.shared, robot, row, recorder, termination_details=details
        )
    assert details == (
        [dict(stage="dispatch_stop", exception_type="TimeoutError", message="STOP_TIMEOUT")]
        if stop_fails
        else []
    )
    assert len(recorded) == len(stops) == 1
    assert (recorded[0].arm, recorded[0].hand) == (1, 4)
    assert robot.shared.run_ended_reason.value == RunEndReason.ESTOP


@pytest.mark.parametrize("cause", [RunEndReason.OPERATOR, RunEndReason.QUIT])
def test_teleop_first_reason_matches_run(cause):
    from dexmani_real.teleop.runner import TeleopRunner

    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared = shared_state()
    runner.control_run_id = 1
    runner.shared.run_ended_id.value = 1
    runner.shared.run_ended_reason.value = int(cause)
    assert runner._end_reason("motion_revoked") == cause.name.lower()
    runner.control_run_id = 2
    assert runner._end_reason("motion_revoked") == "motion_revoked"


@pytest.mark.parametrize("where", ["arm", "hand", "wait"])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_replay_cancel_returns_actual_prefix(monkeypatch, where, stop_fails):
    from dexmani_real.replay import replayer

    robot, clock, command = fake_robot(monkeypatch)
    robot.shared.safety_state.value = int(SafetyState.ARMED)
    # begin_motion increments the epoch; the real robot owner uses that epoch.
    stops = []

    def stop():
        stops.append(True)
        if stop_fails and len(stops) == 1:
            raise TimeoutError("STOP_TIMEOUT")

    robot.stop = stop
    row = NS(
        arm={"qpos": np.array([command.arm_qpos]), "timestamp_ns": [100]},
        hand={"qpos": np.array([command.hand_qpos]), "timestamp_ns": [100]},
    )
    monkeypatch.setattr(replayer, "read_observation", lambda *a, **kw: row)
    monkeypatch.setattr(
        replayer,
        "make_arm_fk",
        lambda: NS(compute=lambda _: (np.zeros(3), np.array([1, 0, 0, 0, 1, 0]))),
    )
    if where == "wait":
        calls = []

        def wait(*args):
            calls.append(True)
            if len(calls) > 1:
                raise KeyboardInterrupt()

        monkeypatch.setattr(replayer, "_wait_replay", wait)
    else:
        clock.interrupt = where
        monkeypatch.setattr(replayer, "_wait_replay", lambda *a: None)
    trajectory = NS(
        num_frames=3,
        fps=30,
        arm_qpos=np.tile(command.arm_qpos, (3, 1)),
        hand_qpos=np.tile(command.hand_qpos, (3, 1)),
        action_arm_joint=np.tile(command.arm_qpos, (3, 1)),
        action_hand_joint=np.tile(command.hand_qpos, (3, 1)),
    )
    outcome = replayer.replay_targets(
        robot.shared, robot.runtime, trajectory, None, robot=robot, hand_start_duration_s=0
    )
    assert outcome.status == (
        replayer.ReplayStatus.FAULT if stop_fails else replayer.ReplayStatus.ESTOP
    )
    if stop_fails:
        assert "STOP_TIMEOUT" in outcome.reason
    assert len(outcome.replay_data["dispatch_status"]) == 1
    expected = {"arm": [4, 0], "hand": [1, 4], "wait": [1, 1]}[where]
    np.testing.assert_array_equal(outcome.replay_data["dispatch_status"][0], expected)
    assert "KeyboardInterrupt" in str(outcome.replay_data["termination_reason"])


def test_tactile_bad_aggregate_does_not_drop_dense(monkeypatch):
    from dexmani_real.robot.drivers import xhand

    monkeypatch.setattr(xhand.time, "sleep", lambda _: None)
    hand = xhand.XHand(ExperimentConfig().hand)
    count = NS(n=0)

    def read(**kw):
        count.n += 1
        value = 100.0 if count.n > xhand._TACTILE_BIAS_SAMPLE_COUNT else 1.0
        return NS(
            tactile_aggregate=np.full((5, 3), value),
            tactile_dense=np.ones((5, 120, 3)),
            tactile_aggregate_valid=True,
            tactile_dense_valid=True,
        )

    hand._read_state = read
    assert hand.tare_tactile() == (False, True)


def test_tactile_actual_get_state_has_independent_usable_flags(monkeypatch):
    from dexmani_real.robot.drivers import xhand

    hand = xhand.XHand(ExperimentConfig().hand)
    hand.connected_flag = True
    hand._control = NS(read_state=lambda *args: (NS(error_code=1501019), object()))
    monkeypatch.setattr(xhand, "_error_code", lambda _: 1501019)
    hand._parse_joints = lambda raw: (
        np.zeros(12),
        np.zeros(12),
        dict(commboard_err=[], jointboard_err=[], tipboard_err=[]),
    )
    hand._parse_tactile_aggregate = lambda raw: np.full((5, 3), 3.0)
    hand._tactile_bias_aggregate = np.ones((5, 3))
    state = hand.get_state()
    assert state.tactile_aggregate_valid and not state.tactile_dense_valid
    np.testing.assert_array_equal(state.tactile_aggregate, np.full((5, 3), 2.0))
    assert np.isnan(state.tactile_dense).all()
    raw = hand._read_state(apply_bias=False)
    np.testing.assert_array_equal(raw.tactile_aggregate, np.full((5, 3), 3.0))


def test_diagnostic_failure_is_nonzero_and_disconnects(monkeypatch, capsys):
    import examples.realsense_record_example as diag

    closed = []

    class Camera:
        active_serial = "fake"

        def __init__(self, *args):
            pass

        def connect(self):
            return True

        def disconnect(self):
            closed.append(True)

        def get_device_info(self):
            return "fake"

        def get_depth_scale(self):
            return 0.001

        def get_geometry(self):
            return NS(aligned_depth_to_color=lambda: None)

        def read(self, **kwargs):
            raise RuntimeError("injected read failure")

    monkeypatch.setattr("dexmani_real.sensor.camera.realsense.RealSenseCamera", Camera)
    monkeypatch.setattr(diag, "_list_cameras", lambda: [True])
    monkeypatch.setattr(diag, "_test_lifecycle", lambda cfg: True)
    monkeypatch.setattr(diag, "_get_ae_priority", lambda _: None)
    monkeypatch.setattr(diag, "CameraExtrinsics", lambda path=None: None)
    monkeypatch.setattr(diag, "_compute_base_from_color", lambda *args: np.eye(4))
    monkeypatch.setattr(
        diag,
        "load_experiment_config",
        lambda **kw: NS(
            pointcloud=NS(
                validate=lambda: None,
                remove_table=False,
                num_points=10,
                depth_min_m=0.1,
                depth_max_m=1,
                voxel_size_m=0.01,
                workspace=None,
            ),
            environment=NS(table=None),
        ),
    )
    monkeypatch.setattr(diag, "NonBlockingPCDViewer", lambda **kw: NS(close=lambda: None))
    monkeypatch.setattr(diag.cv2, "destroyAllWindows", lambda: None)
    assert diag.main([]) == 1
    assert closed == [True]
    assert "Test complete" not in capsys.readouterr().out


def test_unknown_retargeting_config_field_is_rejected(tmp_path):
    from dexmani_real.config.experiment import load_experiment_config

    path = tmp_path / "old.yaml"
    path.write_text("tag_retargeting:\n  prior_weight: 0.0\n")
    with pytest.raises(TypeError, match="prior_weight"):
        load_experiment_config(yaml_path=str(path))


def test_session_camera_snapshot_survives_file_change(tmp_path, monkeypatch):
    from dexmani_real.calibration.camera.extrinsics import CameraExtrinsics
    from dexmani_real.recording import recorder

    path = tmp_path / "cameras.json"

    def payload(x):
        return {
            "camera": {
                "serial": "serial",
                "type": "eye_to_hand",
                "pose": {"position": [x, 0, 0], "orientation": [1, 0, 0, 0]},
            }
        }

    path.write_text(json.dumps(payload(1)))
    snapshot = CameraExtrinsics(path)
    path.write_text(json.dumps(payload(2)))
    shared = NS(
        camera_serial=NS(value=b"serial"),
        camera_geometry=NS(value=b"{}"),
        camera_depth_scale=NS(value=0.001),
    )
    monkeypatch.setattr(recorder.RGBDGeometry, "from_dict", lambda _: object())
    metadata = recorder.snapshot_recording_metadata(
        shared, ExperimentConfig(), collection_source="teleop", camera_calibration=snapshot
    )
    assert metadata["camera_T_xarm_base_from_color"][0, 3] == 1
    missing = recorder.snapshot_recording_metadata(
        shared, ExperimentConfig(), collection_source="teleop", camera_calibration=None
    )
    assert missing["camera_T_xarm_base_from_color"] is None


def test_calibration_cancel_after_solve_does_not_publish(monkeypatch):
    from dexmani_real.calibration.camera import session

    instance = session.CameraCalibrationSession.__new__(session.CameraCalibrationSession)
    instance.shared = shared_state()
    instance.shared.safety_state.value = int(SafetyState.ARMED)
    events = iter(["enter", None])
    instance.keys = NS(pop_event=lambda: next(events), healthy=True, is_pressed=lambda _: False)
    instance.camera_process = NS(is_alive=lambda: True)
    instance.robot = NS(check=lambda: None, stop=lambda: None)
    instance.state = NS(command_qpos=None, command_pose=None, samples=[], calibration_saved=False)
    instance.planner = None
    instance.serial = "fake"
    instance.calibration_config = None
    instance.intrinsics = np.eye(3)
    instance.distortion = np.zeros(5)

    def solve(*args, **kw):
        instance.shared.quit_requested.value = True
        return np.eye(4), {}

    monkeypatch.setattr(session, "_solve_calibration", solve)
    monkeypatch.setattr(
        session,
        "save_camera_calibration",
        lambda *a, **kw: pytest.fail("published cancelled candidate"),
    )
    instance._handle_sample_events()
    assert not instance.state.calibration_saved


def test_calibration_cancel_during_post_capture_feedback_does_not_append(monkeypatch):
    from dexmani_real.calibration.camera import session

    instance = session.CameraCalibrationSession.__new__(session.CameraCalibrationSession)
    instance.runtime = ExperimentConfig()
    instance.robot = None
    instance.shared = shared_state()
    instance.keys = NS(healthy=True, is_pressed=lambda _: False)
    instance.camera_process = NS(is_alive=lambda: True)
    instance.state = NS(samples=[])
    instance.intrinsics = np.eye(3)
    instance.distortion = np.zeros(5)
    instance.aruco_config = NS(marker_size_m=0.05, target_id=0, capture_frames=1)
    calls = []

    def read(*args):
        calls.append(True)
        if len(calls) == 2:
            instance.shared.quit_requested.value = True
        return {"qpos": np.zeros(7)}, ""

    monkeypatch.setattr(session, "_read_stationary_calibration_arm_state", read)
    monkeypatch.setattr(session, "_detect_aruco_stable", lambda *a, **kw: (np.zeros(3), np.ones(3)))
    instance._capture_sample()
    assert len(calls) == 2 and instance.state.samples == []


def test_encoder_explicit_frame_clock(tmp_path):
    from dexmani_real.recording.storage.video import VideoDecoder, VideoEncoder

    path = tmp_path / "encoded.mp4"
    with VideoEncoder(path, fps=29.97, width=32, height=16) as encoder:
        for i in range(8):
            encoder.write_frame(np.full((16, 32, 3), i, np.uint8))
    with VideoDecoder(path) as decoder:
        frames = list(decoder.iter_frames())
        assert len(frames) == 8
        np.testing.assert_array_equal(decoder.read_frame(7), frames[7])


@pytest.mark.parametrize("cause", [RunEndReason.OPERATOR, RunEndReason.QUIT])
def test_teleop_stop_failure_retains_first_reason(cause):
    from dexmani_real.teleop.runner import TeleopRunner

    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared = shared_state()
    runner.control_run_id = 1
    runner.pending_termination_reason = None
    runner.termination_details = []
    runner.robot = NS(stop=lambda: (_ for _ in ()).throw(RuntimeError("stop failed")))
    with pytest.raises(RuntimeError, match="stop failed"):
        runner._pause_control("motion_revoked", run_end_reason=cause)
    saved = []
    runner.recorder = NS(is_recording=True, save_episode=lambda **kw: saved.append(kw))
    runner._finish_capture(True, "cleanup_failure", announce=False)
    assert saved == [
        {
            "reason": cause.name.lower(),
            "details": [
                {"stage": "pause_stop", "exception_type": "RuntimeError", "message": "stop failed"}
            ],
        }
    ]


def test_tag_native_gradient_without_prior():
    from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
    from dexmani_real.teleop.retargeting.tag_optimizer import TAGHandRetargeter

    cfg = ExperimentConfig()
    optimizer = TAGHandRetargeter(
        cfg.hand.fingertip_link_names, cfg.tag_retargeting, str(XHAND_RIGHT_URDF_PATH)
    )._optimizer
    q = (optimizer.joint_limits_lower + optimizer.joint_limits_upper) / 2
    optimizer.qpos_floating[7:] = q
    optimizer.pin_grad.update_kinematics(optimizer.qpos_floating)
    optimizer._current_target = 0.95 * np.array(
        [
            optimizer.pin_grad.data.oMf[i].translation.copy()
            for i in optimizer.pin_grad.tip_frame_ids
        ]
    )
    optimizer.qpos_stage1 = q.copy()
    optimizer.pinch_factors[:] = 0.2
    for objective in (optimizer._obj_s1, optimizer._obj_s2):
        gradient = np.zeros_like(q)
        assert np.isfinite(objective(q, gradient))
        numeric = np.zeros_like(q)
        for i in range(len(q)):
            delta = np.zeros_like(q)
            delta[i] = 1e-6
            numeric[i] = (
                objective(q + delta, np.empty(0)) - objective(q - delta, np.empty(0))
            ) / 2e-6
        np.testing.assert_allclose(gradient, numeric, rtol=1e-4, atol=1e-6)


def test_replay_rmse_names_preserve_distinct_statistics():
    from dexmani_real.replay.evaluation import compute_metrics

    arm = np.deg2rad([[7, 0, 0, 0, 0, 0, 0]])
    hand = np.deg2rad([[7] + [0] * 11])
    metrics = compute_metrics(
        np.zeros_like(arm),
        arm,
        None,
        np.zeros((1, 3)),
        np.zeros((1, 6)),
        30,
        np.zeros_like(hand),
        hand,
    )
    assert metrics.arm_mean_joint_rmse_deg == pytest.approx(1)
    assert metrics.hand_pooled_rmse_deg == pytest.approx(7 / np.sqrt(12))
