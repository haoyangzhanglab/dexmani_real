import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.control.actions import ActionResult
from dexmani_real.control.ik import IKReferenceError
from dexmani_real.robot.commands import RobotCommand, RobotError
from dexmani_real.teleop.loop import TeleopLoop


@pytest.fixture
def loop(monkeypatch):
    clock = [1000000000]
    monkeypatch.setattr("dexmani_real.teleop.loop.time.monotonic_ns", lambda: clock[0])
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.cancel_epoch = 1
    loop.runtime_state = SimpleNamespace(
        cancel_epoch=1,
        requests_allow_motion=lambda: True,
        stop_signal=threading.Event(),
        motion_lock=threading.RLock(),
    )
    loop.runtime = SimpleNamespace(teleop=SimpleNamespace(target_max_age_s=0.5))
    loop.controller = Mock(previous_eef_target="old_pose")
    loop.controller.compute_target.return_value = ActionResult(
        None, np.ones(12), "geometry"
    )
    loop.robot = Mock()
    loop.recorder = Mock()
    loop.recorder.append.return_value = None
    loop.admit_observation = lambda row: []
    loop.audio = Mock()
    loop.last_command = RobotCommand(np.zeros(7), np.zeros(12))
    loop.target_deadline_ns = clock[0] + 100000000
    loop.hold_reason = None
    loop.request_episode_end = Mock()
    loop.pending_termination_reason = None
    loop.poll_blocking_commands = Mock(return_value=False)
    loop.preparation_interrupted = Mock(return_value=False)
    loop.max_duration_s = 10
    return loop, clock


def row():
    return SimpleNamespace(arm={"qpos": [np.zeros(7)]})


def test_paired_resend_keeps_deadline_and_record(loop):
    obj, clock = loop
    previous = obj.last_command
    deadline = obj.target_deadline_ns
    observation = row()
    obj.execute_control_step(observation)
    obj.robot.send_action.assert_called_once_with(previous, expires_ns=deadline)
    obj.recorder.append.assert_called_once_with(observation, previous)
    assert obj.controller.commit.call_args.kwargs["eef_pose"] == "old_pose"
    assert obj.target_deadline_ns == deadline
    clock[0] = deadline
    obj.execute_control_step(observation)
    assert obj.robot.send_action.call_count == 1


@pytest.mark.parametrize(
    "failure", [RobotError("hand failed"), IKReferenceError("lead mismatch")]
)
def test_failures_do_not_commit(loop, failure):
    obj, _ = loop
    if isinstance(failure, IKReferenceError):
        obj.controller.compute_target.side_effect = failure
        obj.execute_control_step(row())
        obj.robot.send_action.assert_not_called()
        obj.recorder.append.assert_not_called()
        obj.request_episode_end.assert_called_once()
    else:
        obj.robot.send_action.side_effect = failure
        with pytest.raises(RobotError):
            obj.execute_control_step(row())
        assert obj.recorder.append.call_args.args[1] is obj.last_command
    obj.controller.commit.assert_not_called()


def test_cancel_does_not_commit(loop):
    obj, _ = loop
    obj.runtime_state.cancel_epoch += 1
    obj.execute_control_step(row())
    obj.robot.send_action.assert_not_called()
    obj.controller.commit.assert_not_called()


def test_vr_reference_begins_only_on_commit(monkeypatch):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.teleop.vr import VRController

    cfg = RealConfig()
    mapper = ActionMapper.for_mode(
        "eef",
        arm=cfg.arm,
        hand=cfg.hand,
        motion=cfg.motion,
        environment=cfg.environment,
    )
    retarget = Mock()
    retarget.retarget.return_value = None
    controller = VRController(mapper, retarget, cfg.teleop, np.eye(3))
    q = np.array(cfg.arm.home_qpos)
    obs = SimpleNamespace(
        arm={"qpos": [q], "timestamp_ns": [1]},
        hand={"qpos": [np.deg2rad(cfg.hand.home_qpos_deg)]},
        observation_timestamp_ns=1,
        vr={
            "wrist_pos": np.zeros(3),
            "wrist_quat_wxyz": np.array([1, 0, 0, 0]),
            "landmarks": np.zeros((21, 3)),
        },
    )
    controller.reset(obs)
    bounds = Mock(wraps=mapper.solver.bounds)
    monkeypatch.setattr(mapper.solver, "bounds", bounds)
    controller.compute_target(obs, control_ns=10000000)
    assert controller.ik_reference is None
    assert bounds.call_args.args[-1] == 1 / cfg.teleop.control_hz
    command = RobotCommand(q.copy(), np.deg2rad(cfg.hand.home_qpos_deg))
    controller.commit(
        command,
        eef_pose=controller.previous_eef_target,
        submit_end_ns=20000000,
        measured_qpos=q,
    )
    np.testing.assert_array_equal(controller.ik_reference.anchor, q)
    obs.arm["qpos"] = [q + 0.3]
    with pytest.raises(IKReferenceError):
        controller.compute_target(obs, control_ns=30000000)
    controller.clear()
    assert controller.ik_reference is None
    obs.arm["qpos"] = [q]
    controller.reset(obs)
    controller.compute_target(obs, control_ns=20000000)
    assert bounds.call_args.args[-1] == 1 / cfg.teleop.control_hz


def test_policy_eef_preparation_budget_and_rejection(monkeypatch):
    from dexmani_real.deployment.rollout import Rollout
    from dexmani_real.runtime.ik_reference import IKReference

    obj = Rollout.__new__(Rollout)
    obj.policy_info = SimpleNamespace(action_mode="eef")
    obj.execution = SimpleNamespace(control_dt_s=0.1)
    q = np.zeros(7)
    obj.action_mapper = Mock()
    obj.action_mapper.solver.measured_reference.return_value = q
    obj.action_mapper.map_eef.return_value = ActionResult(None, None, "collision")
    now_ns = 100_000_000
    monkeypatch.setattr("dexmani_real.deployment.rollout.time.monotonic_ns", lambda: now_ns)
    action = np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], np.zeros(12)]
    observation = row()
    for ref, expected in [
        (None, 0.1),
        (IKReference(q + 0.01, q, now_ns - 1_000_000), 0.001),
        (None, 0.1),
    ]:
        obj.ik_reference = ref
        command, reason = obj.map_command(action, observation)
        assert command is None and reason == "action_rejected: collision"
        assert obj.action_mapper.map_eef.call_args.args[2][-1] == expected
        assert obj.ik_reference is ref


@pytest.mark.parametrize("age_ns,valid", [(0, True), (1, True), (-1, False), (500_000_000, False)])
def test_device_feedback_time_boundary(age_ns, valid):
    from dexmani_real.runtime.observation import check_device_feedback

    now = 1_000_000_000
    runtime = SimpleNamespace(
        arm=SimpleNamespace(feedback_max_age_s=0.5),
        hand=SimpleNamespace(enabled=False),
    )
    observation = SimpleNamespace(arm={"timestamp_ns": [now - age_ns]})
    assert (check_device_feedback(observation, runtime, now_ns=now) == []) == valid


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("fail", [False, True])
def test_policy_submission_progress(mode, fail, monkeypatch):
    from dexmani_real.deployment.rollout import Rollout

    obj = Rollout.__new__(Rollout)
    obj.execution = SimpleNamespace(mode=mode, action_steps=2)
    obj.next_step_ns = 0
    obj.dt_ns = 100000000
    obj.started_ns = 1000000000
    obj.last_slot = -1
    obj.result = SimpleNamespace(skipped_slots=0)
    obj.plan = SimpleNamespace(actions=np.zeros((20, 21)), anchor_slot=0)
    obj.cursor = 0
    obj.has_submitted = False
    obj.ik_reference = None
    obj.model = SimpleNamespace(future=None)
    obj.sample_observation = Mock(return_value=row())
    obj.update_history = Mock(return_value=True)
    obj.check_motion = Mock()
    obj.duration_expired = Mock(return_value=False)
    obj.poll_model = Mock()
    obj.select_async_action = Mock(return_value=np.zeros(21))
    command = RobotCommand(np.zeros(7), np.zeros(12))
    obj.prepare_command = Mock(return_value=command)
    obj.robot = Mock()
    obj.record_row = Mock()
    obj.policy_info = SimpleNamespace(action_mode="eef")
    obj.action_mapper = SimpleNamespace(
        solver=SimpleNamespace(limits=np.tile([-6, 6], (7, 1)))
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.time.monotonic_ns", lambda: 1000000001
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.control_slot", lambda **kw: (0, 1000000100, 0)
    )
    if fail:
        obj.robot.send_action.side_effect = RobotError("partial")
        with pytest.raises(RobotError):
            (obj.step_sync if mode == "sync" else obj.step_async)()
        assert obj.ik_reference is None and obj.cursor == 0
    else:
        (obj.step_sync if mode == "sync" else obj.step_async)()
        assert obj.ik_reference is not None
        assert obj.cursor == (0 if mode == "async" else 1)
    assert obj.record_row.call_args.args[1] is command


def test_raw_and_export_use_submitted_targets():
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.export import iter_canonical_blocks
    from dexmani_real.data.raw import encode_raw_row
    from dexmani_real.geometry.kinematics import ArmFK
    from dexmani_real.geometry.pose import encode_eef_pose

    cfg = RealConfig()
    q = np.array(cfg.arm.home_qpos)
    h = np.deg2rad(cfg.hand.home_qpos_deg)
    observation = SimpleNamespace(
        arm=[{"qpos": q + 0.01}],
        hand=[
            {
                "qpos": h + 0.01,
                "tactile_aggregate": np.zeros((5, 3)),
                "tactile_dense": np.zeros((5, 120, 3)),
            }
        ],
    )
    raw = encode_raw_row(observation, RobotCommand(q, h))

    class Reader(dict):
        num_frames = 1

    reader = Reader({k: np.array([v]) for k, v in raw.items()})
    fk = ArmFK()
    blocks = list(
        iter_canonical_blocks(
            reader,
            cfg,
            fields=("action", "action_ee"),
            arm_fk=fk,
            hand_fk=None,
            hand_cloud=None,
            camera={"parameters": {}},
        )
    )
    np.testing.assert_allclose(blocks[0]["action"][0], np.r_[q, h], atol=1e-6)
    np.testing.assert_allclose(
        blocks[0]["action_ee"][0, :9],
        encode_eef_pose(np.array([fk.forward(q)]))[0],
        atol=1e-6,
    )


def test_async_expired_candidate_does_not_commit(monkeypatch):
    from dexmani_real.deployment.rollout import Rollout

    obj = Rollout.__new__(Rollout)
    obj.execution = SimpleNamespace(mode="async")
    obj.started_ns = 100
    obj.dt_ns = 100
    obj.next_step_ns = 100
    obj.last_slot = -1
    obj.result = SimpleNamespace(skipped_slots=0)
    obj.plan = SimpleNamespace(anchor_slot=0, actions=np.zeros((1, 21)))
    obj.model = SimpleNamespace(future=None)
    obj.query = None
    obj.sample_observation = Mock(return_value=row())
    obj.update_history = Mock(return_value=True)
    obj.check_motion = Mock()
    obj.duration_expired = Mock(return_value=False)
    obj.poll_model = Mock()
    obj.select_async_action = Mock(return_value=np.zeros(21))
    obj.prepare_command = Mock(return_value=RobotCommand(np.zeros(7), np.zeros(12)))
    obj.robot = Mock()
    obj.commit_arm = Mock()
    obj.record_row = Mock()
    obj.stop = Mock()
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.time.monotonic_ns", lambda: 300
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.control_slot", lambda **kw: (0, 200, 0)
    )
    obj.step_async()
    obj.robot.send_action.assert_not_called()
    obj.commit_arm.assert_not_called()
    assert obj.record_row.call_args.args[1] is None
    obj.stop.assert_called_once()


@pytest.mark.parametrize("invalid", ["outside", "nan"])
def test_policy_start_reference_classification(invalid, monkeypatch):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.deployment.rollout import Rollout

    cfg = RealConfig()
    obj = Rollout.__new__(Rollout)
    obj.runtime = cfg
    obj.policy_info = SimpleNamespace(action_mode="eef")
    obj.action_mapper = ActionMapper.for_mode(
        "eef",
        arm=cfg.arm,
        hand=cfg.hand,
        motion=cfg.motion,
        environment=cfg.environment,
    )
    obj.preparation_valid = lambda: True
    obj.robot = Mock()
    obj.sensors = SimpleNamespace(channels=None)
    obj.capture_needs = obj.model_needs = SimpleNamespace(required_tactile=())
    obj.record_raw = False
    q = np.array(cfg.arm.home_qpos)
    q[2] = cfg.arm.joint_limit_upper[2] + 0.01 if invalid == "outside" else np.nan
    obs = SimpleNamespace(arm={"qpos": [q]})
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.read_observation", lambda *a, **kw: obs
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.check_device_feedback", lambda *a: []
    )
    monkeypatch.setattr(
        "dexmani_real.deployment.rollout.check_observation", lambda *a: []
    )
    if invalid == "outside":
        assert obj.start_observation() is None
        assert "outside operational limits" in obj.preparation_issue
    else:
        with pytest.raises(ValueError):
            obj.start_observation()
