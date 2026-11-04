"""实测起点经真实重定向、mapper 和 RobotCommand 后仍满足首帧范围与速率。"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.runtime.observation import Observation
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter
from dexmani_real.teleop.retargeting.landmarks import hand_lengths, prepare_landmarks
from dexmani_real.teleop.vr import VRController


def make_hand_loop(monkeypatch, *, smoothing_tau_s=.02, custom_limits=False):
    now = [1_000_000_000]
    monkeypatch.setattr('time.monotonic_ns', lambda: now[0])
    runtime = RealConfig()
    if custom_limits:
        lower = np.array(runtime.hand.qpos_min_rad)
        lower[[2, 5, 7, 9, 11]] += .1
        runtime = replace(runtime, hand=replace(runtime.hand, qpos_min_rad=tuple(lower)))
    config = replace(runtime.hand_retargeting, smoothing_tau_s=smoothing_tau_s)
    retargeter = AdaptiveHandRetargeter(config, runtime.hand)
    # 合法、固定骨长的弯曲人手；本测试实际运行 FK、解析梯度和 NLopt。
    landmarks = np.zeros((21, 3))
    for i, (side, forward) in enumerate(((.05, .03), (.03, .07), (0., .08), (-.02, .07), (-.04, .06))):
        base = np.array([0., side, forward])
        landmarks[1 + 4 * i:5 + 4 * i] = base + np.arange(4)[:, None] * np.array([.02, 0., .02])
    measurements = hand_lengths(prepare_landmarks(landmarks))
    retargeter.calibrate([measurements] * config.calibration_samples)
    solver = SimpleNamespace(
        compute_eef_pose_base=lambda q: Pose(np.zeros(3), np.array([1., 0., 0., 0.])),
        solve=lambda pose, current, previous, **kwargs: SimpleNamespace(qpos=current.copy()),
    )
    mapper = ActionMapper(
        SimpleNamespace(workspace=np.array([[-1., 1.]] * 3)), solver=solver,
        hand_enabled=True, hand_lower_rad=runtime.hand.qpos_min_rad,
        hand_upper_rad=runtime.hand.qpos_max_rad,
        home_hand_qpos=np.deg2rad(runtime.hand.home_qpos_deg),
    )
    controller = VRController(mapper, retargeter, runtime.teleop, np.eye(3))
    state = SessionState()
    commands, hand_targets = [], []
    robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    robot.connected = True
    robot.arm = SimpleNamespace(
        is_connected=True, check_error=lambda: None,
        submit=lambda command, **kwargs: commands.append(command),
        request_stop=lambda: None, wait_stopped=lambda: None,
    )

    def send_hand(target):
        hand_targets.append(target.copy())
        return False  # 普通写 CRC 正常返回仍推进提交参考。

    robot.hand = SimpleNamespace(is_connected=True, _send_action=send_hand, stop=lambda *a, **kw: None)
    loop = TeleopLoop(
        state, runtime, robot, task_label='test', sensors=None, controller=controller,
        recording_dir=None, max_duration_s=60, runtime_config_json=None,
        keyboard=None, audio=SimpleNamespace(play=lambda sound: None),
    )
    loop.poll_blocking_commands = lambda: False
    loop.cancel_epoch = state.cancel_epoch
    loop.phase = TeleopPhase.PREPARING

    def observation(qpos):
        robot.hand_feedback.consume(XHandState(qpos, None, None, now[0]))
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        arm['timestamp_ns'] = now[0]
        arm['qpos'] = runtime.arm.home_qpos
        # 独立可写输入也不能通过 reset 的别名改变控制参考。
        return Observation(
            arm, robot.hand_feedback.snapshot().copy(), None,
            dict(wrist_pos=np.zeros(3), wrist_quat_wxyz=np.array([1., 0., 0., 0.]),
                 landmarks=landmarks, recv_ts_ns=now[0]),
            None, now[0], control_timestamp_ns=now[0],
        )

    return loop, observation, now, commands, hand_targets


@pytest.mark.parametrize('start', ['lower', 'interior', 'custom_lower'])
@pytest.mark.parametrize('interval_ns', [0, 1, 10_000_000])
@pytest.mark.parametrize('tau', [0., .02])
def test_first_submitted_target_respects_measured_start_after_mapper(monkeypatch, start, interval_ns, tau):
    loop, observe, now, commands, sent = make_hand_loop(
        monkeypatch, smoothing_tau_s=tau, custom_limits=start == 'custom_lower',
    )
    initial = np.array(loop.runtime.hand.qpos_min_rad)
    if start == 'interior':
        initial = np.deg2rad(loop.runtime.hand.home_qpos_deg)
    row = observe(initial)
    loop.try_begin(row)
    assert loop.phase is TeleopPhase.ACTIVE
    # reset 不共享输入；未提交的候选也不改变首次参考。
    row.hand['qpos'] = loop.runtime.hand.qpos_max_rad
    now[0] += interval_ns
    row = observe(initial)
    candidate = loop.controller.compute_target(row, control_ns=now[0])
    assert candidate.rejection_reason is None
    loop.execute_control_step(row)
    assert len(commands) == len(sent) == 1
    final = commands[0].hand_qpos
    np.testing.assert_array_equal(final, sent[0])
    speed = loop.runtime.hand_retargeting.max_joint_speed_rad_s
    assert np.max(np.abs(final - initial)) <= speed * interval_ns / 1e9 + 1e-12
    assert np.all(final >= loop.runtime.hand.qpos_min_rad)
    assert np.all(final <= loop.runtime.hand.qpos_max_rad)
    if interval_ns == 10_000_000:
        assert np.max(np.abs(final - initial)) > 1e-5
    # CRC 未确认也按正常提交尝试推进；下一帧以实际 RobotCommand 为参考。
    now[0] += 10_000_000
    loop.execute_control_step(observe(initial))
    assert len(commands) == len(sent) == 2
    assert np.max(np.abs(sent[1] - final)) <= speed * .01 + 1e-12


@pytest.mark.parametrize('outside', ['zero', 'custom_lower', 'upper'])
def test_outside_start_rejects_only_episode_before_any_target(monkeypatch, outside, caplog):
    loop, observe, now, commands, sent = make_hand_loop(monkeypatch, custom_limits=outside == 'custom_lower')
    initial = np.array(loop.runtime.hand.qpos_min_rad)
    if outside == 'zero':
        initial[:] = 0.
    elif outside == 'custom_lower':
        initial[2] -= .01
    else:
        initial[2] = loop.runtime.hand.qpos_max_rad[2] + .01
    loop.try_begin(observe(initial))
    assert not commands and not sent
    assert loop.pending_termination_reason is not None
    assert loop.phase is not TeleopPhase.ACTIVE
    assert 'right_hand_thumb_rota_joint2' in caplog.text
    assert '就位' in caplog.text
    loop.end_episode()
    assert loop.phase is TeleopPhase.IDLE
    assert not loop.runtime_state.quit_requested
    # 无自动入域或 HOME；操作者就位后可再次接纳。
    loop.runtime_state.stop_signal.clear()
    loop.pending_termination_reason = None
    loop.cancel_epoch = loop.runtime_state.cancel_epoch
    loop.try_begin(observe(np.deg2rad(loop.runtime.hand.home_qpos_deg)))
    assert loop.phase is TeleopPhase.ACTIVE
    assert not commands and not sent


@pytest.mark.parametrize('error_sign', [-1, 1])
def test_default_home_arrival_tolerance_stays_inside_start_range(monkeypatch, error_sign):
    loop, observe, now, commands, sent = make_hand_loop(monkeypatch)
    hand = loop.runtime.hand
    hand.validate_home()
    measured = np.deg2rad(np.array(hand.home_qpos_deg) + error_sign * hand.home_tolerance_deg)
    loop.try_begin(observe(measured))
    assert loop.phase is TeleopPhase.ACTIVE
    now[0] += 10_000_000
    loop.execute_control_step(observe(measured))
    assert len(commands) == len(sent) == 1
    assert np.max(np.abs(sent[0] - measured)) <= .04 + 1e-12
