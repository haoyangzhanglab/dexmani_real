"""最小 HDF5 与公共设备 send/close 的离线 demo 回归。"""

import json
import threading
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.replay import run
from dexmani_real.replay.trajectory import load_trajectory
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import SessionState


def demo(path, n=300):
    path.mkdir()
    cfg = RealConfig()
    with h5py.File(path / "data.h5", "w") as data:
        data.create_dataset(
            "action_arm_joint_target",
            data=(
                np.tile(cfg.arm.home_qpos, (n, 1)) + np.arange(n)[:, None] * 0.00001
            ).astype("f4"),
        )
        data.create_dataset(
            "action_hand_joint_target",
            data=np.tile(np.deg2rad(cfg.hand.home_qpos_deg) + 0.001, (n, 1)).astype(
                "f4"
            ),
        )
        data.create_group("meta").attrs.update(
            control_hz=16,
            joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
        )
    return cfg


def fake_devices(monkeypatch, cfg, delay, *, failure=None, close_failure=False):
    now = [1_000_000_000]
    sent, closed, feedback_calls = [], [], []
    monkeypatch.setattr(
        run,
        "SessionState",
        lambda: SessionState(
            stop_signal=threading.Event(), emergency_signal=threading.Event()
        ),
    )
    keyboard = Keyboard()

    def make_keyboard(**kwargs):
        keyboard.bindings = kwargs["bindings"]
        keyboard.estop_callback = kwargs["estop_callback"]
        return keyboard

    monkeypatch.setattr(run, "Keyboard", make_keyboard)
    monkeypatch.setattr(
        Keyboard,
        "start",
        lambda self: (
            setattr(self, "active", True),
            setattr(self, "listener", SimpleNamespace(is_alive=lambda: True)),
        ),
    )
    monkeypatch.setattr(Keyboard, "stop", lambda self: closed.append("keyboard"))
    monkeypatch.setattr("time.monotonic_ns", lambda: now[0])
    monkeypatch.setattr("time.monotonic", lambda: now[0] / 1e9)
    monkeypatch.setattr(
        "time.sleep", lambda seconds: now.__setitem__(0, now[0] + round(seconds * 1e9))
    )

    def connect(self):
        def snapshot():
            row = np.zeros(1, dtype=ARM_STATE_DTYPE)
            row["qpos"] = cfg.arm.home_qpos
            row["timestamp_ns"] = now[0]
            return row

        def submit(command, **kw):
            if failure:
                raise failure
            sent.append(command.arm_qpos.copy())

        def close():
            closed.append("arm")
            if close_failure:
                raise OSError("close failed")

        self.arm = SimpleNamespace(
            is_connected=True,
            check_error=lambda: None,
            snapshot=snapshot,
            submit=submit,
            close=close,
            is_closed=True,
        )

        def get_state():
            feedback_calls.append(now[0])
            return XHandState(np.deg2rad(cfg.hand.home_qpos_deg), None, None, now[0])

        def send(q):
            now[0] += delay

        self.hand = SimpleNamespace(
            is_connected=True,
            control=True,
            get_state=get_state,
            _send_action=send,
            stop=lambda: None,
        )

        def disconnect(**kwargs):
            closed.append("hand")
            self.hand.control = None

        self.hand.disconnect = disconnect
        self.connected = True
        self.service_feedback()

    monkeypatch.setattr(Robot, "connect", connect)
    monkeypatch.setattr(
        Robot,
        "prepare_streaming",
        lambda self, epoch: self.runtime_state.stop_signal.clear(),
    )
    return now, sent, closed, feedback_calls, keyboard


@pytest.mark.parametrize("delay", [0, 63_000_000])
def test_demo_sequential_timing_and_repeat(monkeypatch, tmp_path, delay):
    path = tmp_path / ".tmp_demo"
    cfg = demo(path)
    now, sent, closed, feedback, _ = fake_devices(monkeypatch, cfg, delay)
    for _ in range(2):
        before = now[0]
        run.run_replay(path, cfg)
        assert now[0] - before == 299 * max(62_500_000, delay) + delay + 500_000_000
    assert len(sent) == 600 and len(feedback) > 600
    np.testing.assert_array_equal(
        sent[:300], load_trajectory(path, cfg).action_arm_joint
    )
    np.testing.assert_array_equal(sent[300:], sent[:300])
    assert closed == ["arm", "hand", "keyboard"] * 2
    assert sorted(p.name for p in path.iterdir()) == ["data.h5"]


@pytest.mark.parametrize(
    "bad",
    [
        "nan",
        "complex",
        "string",
        "shape",
        "empty",
        "length",
        "arm_limit",
        "hand_limit",
        "units",
        "order",
        "frequency",
        "hand_rate",
    ],
)
def test_invalid_demo_rejected_before_devices(monkeypatch, tmp_path, bad):
    path = tmp_path / "demo"
    cfg = demo(path, 2)
    with h5py.File(path / "data.h5", "r+") as data:
        if bad in {"complex", "string", "shape", "empty"}:
            del data["action_arm_joint_target"]
            data["action_arm_joint_target"] = np.zeros(
                (0 if bad == "empty" else 2, 6 if bad == "shape" else 7),
                dtype={"complex": "c16", "string": "S1"}.get(bad, "f8"),
            )
        elif bad == "nan":
            data["action_arm_joint_target"][0, 0] = np.nan
        elif bad == "length":
            values = data["action_hand_joint_target"][:1]
            del data["action_hand_joint_target"]
            data["action_hand_joint_target"] = values
        elif bad == "arm_limit":
            data["action_arm_joint_target"][0, 0] = 100
        elif bad == "hand_limit":
            data["action_hand_joint_target"][0, 2] = 0
        elif bad in {"frequency", "hand_rate"}:
            data["meta"].attrs["control_hz"] = np.inf if bad == "frequency" else 84
        else:
            data["meta"].attrs[
                "joint_units" if bad == "units" else "arm_joint_names"
            ] = "deg" if bad == "units" else "[]"

    def unexpected_robot(*args, **kwargs):
        pytest.fail("invalid input must fail before creating devices")

    monkeypatch.setattr(run, "Robot", unexpected_robot)
    with pytest.raises(ValueError):
        run.run_replay(path, cfg)


@pytest.mark.parametrize(
    "primary,closing", [(True, False), (False, True), (True, True)]
)
def test_replay_errors_close_all_resources(
    monkeypatch, tmp_path, primary, closing, caplog
):
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, _, closed, _, _ = fake_devices(
        monkeypatch,
        cfg,
        0,
        failure=ValueError("send failed") if primary else None,
        close_failure=closing,
    )
    with pytest.raises(
        ValueError if primary else OSError,
        match="send failed" if primary else "close failed",
    ):
        run.run_replay(path, cfg)
    assert closed == ["arm", "hand", "keyboard"]
    if primary and closing:
        assert "replay close also failed" in caplog.text


@pytest.mark.parametrize("key", ["s", "q", "esc"])
@pytest.mark.parametrize("stage", ["connect", "prepare", "rows", "tail"])
def test_replay_cancellation_closes_once(monkeypatch, tmp_path, key, stage):
    from dexmani_real.robot.commands import OperationCancelled

    path = tmp_path / "demo"
    cfg = demo(path, 2)
    now, sent, closed, _, keyboard = fake_devices(monkeypatch, cfg, 0)

    def cancel():
        keyboard.handle_key(key, True)

    if stage in {"connect", "prepare"}:
        name = "connect" if stage == "connect" else "prepare_streaming"
        original = getattr(Robot, name)

        def operation(self, *args):
            original(self, *args)
            cancel()

        monkeypatch.setattr(Robot, name, operation)
    else:
        original = Robot.service_feedback

        def service(self):
            original(self)
            if len(sent) == (1 if stage == "rows" else 2):
                cancel()

        monkeypatch.setattr(Robot, "service_feedback", service)
    if key == "esc":
        with pytest.raises(OperationCancelled):
            run.run_replay(path, cfg)
    else:
        run.run_replay(path, cfg)
    assert closed == ["arm", "hand", "keyboard"]
    assert now[0] < 1_563_000_000


@pytest.mark.parametrize("kind", ["start", "feedback", "device_stop"])
def test_replay_rejects_real_boundaries(monkeypatch, tmp_path, kind):
    path = tmp_path / "demo"
    cfg = demo(path, 2)
    _, _, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    original = Robot.read_state

    def read(self):
        state = original(self)
        if kind == "start":
            state.arm["qpos"] += 0.3
        elif kind == "feedback":
            state.arm["timestamp_ns"] = 1
        return state

    monkeypatch.setattr(Robot, "read_state", read)
    if kind == "device_stop":
        original_send = Robot.send_action

        def send(self, *args, **kwargs):
            original_send(self, *args, **kwargs)
            self.runtime_state.stop_signal.set()

        monkeypatch.setattr(Robot, "send_action", send)
    with pytest.raises((ValueError, RuntimeError)):
        run.run_replay(path, cfg)
    assert closed == ["arm", "hand", "keyboard"]


def test_replay_imports_without_geometry_or_video():
    import subprocess
    import sys

    code = """
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'pinocchio', 'hppfcl', 'av', 'cv2', 'scipy'}:
            raise RuntimeError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from dexmani_real.replay.run import run_replay
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_cli_can_repeat_without_output_directory(monkeypatch, tmp_path):
    from scripts.replay_episode import main

    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, sent, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    assert main([str(path)]) == 0
    assert main([str(path)]) == 0
    assert len(sent) == 2 and closed == ["arm", "hand", "keyboard"] * 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["demo"]


def test_device_stop_after_connect_cannot_be_cleared_by_prepare(monkeypatch, tmp_path):
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, sent, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    connect = Robot.connect

    def stopped_connect(self):
        connect(self)
        self.runtime_state.stop_signal.set()

    monkeypatch.setattr(Robot, "connect", stopped_connect)

    def unexpected_prepare(*args):
        pytest.fail("unexplained device stop must not be resumed")

    monkeypatch.setattr(Robot, "prepare_streaming", unexpected_prepare)
    with pytest.raises(RuntimeError, match="device stopped without operator input"):
        run.run_replay(path, cfg)
    assert not sent
    assert closed == ["arm", "hand", "keyboard"]


def test_emergency_during_shutdown_is_nonzero(monkeypatch, tmp_path):
    from scripts.replay_episode import main

    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, sent, closed, _, keyboard = fake_devices(monkeypatch, cfg, 0)
    close = Robot.close

    def close_and_cancel(self):
        close(self)
        keyboard.handle_key("esc", True)

    monkeypatch.setattr(Robot, "close", close_and_cancel)
    assert main([str(path)]) == 1
    assert len(sent) == 1
    assert closed == ["arm", "hand", "keyboard"]


def test_close_timeout_retains_pending_arm(monkeypatch, tmp_path):
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, _, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    connect = Robot.connect
    robots = []

    def pending_connect(self):
        connect(self)
        self.arm.is_closed = False

        def close():
            closed.append("arm")
            raise TimeoutError("owner still active")

        self.arm.close = close
        robots.append(self)

    monkeypatch.setattr(Robot, "connect", pending_connect)
    with pytest.raises(TimeoutError, match="owner still active"):
        run.run_replay(path, cfg)
    assert robots[0].arm is not None and robots[0].stop_required
    assert closed == ["arm", "hand", "keyboard"]


@pytest.mark.parametrize("key", ["s", "q"])
@pytest.mark.parametrize("failure", [None, "owner", "close"])
def test_slow_prepare_cancel_does_not_revalidate_old_feedback(monkeypatch, tmp_path, key, failure):
    from dexmani_real.robot.commands import OperationCancelled, RobotError
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    now, sent, closed, _, keyboard = fake_devices(
        monkeypatch, cfg, 0, close_failure=failure == "close")
    def prepare(self, epoch):
        now[0] += 200_000_000
        keyboard.handle_key(key, True)
        # The callback's cancellation fact survives consumption of its event.
        keyboard.poll(self.runtime_state, timeout=0)
        if failure == "owner":
            def fault():
                raise RobotError("SDK fault")
            self.arm.check_error = fault
        raise OperationCancelled("prepare cancelled")
    monkeypatch.setattr(Robot, "prepare_streaming", prepare)
    if failure:
        with pytest.raises(RobotError if failure == "owner" else OSError,
                           match="SDK fault" if failure == "owner" else "close failed"):
            run.run_replay(path, cfg)
    else:
        run.run_replay(path, cfg)
    assert not sent and closed == ["arm", "hand", "keyboard"]


@pytest.mark.parametrize("key", ["s", "q", "esc"])
def test_cancel_during_start_snapshot_precedes_feedback_age(monkeypatch, tmp_path, key):
    from dexmani_real.robot.commands import OperationCancelled

    path = tmp_path / "demo"
    cfg = demo(path, 1)
    now, sent, closed, _, keyboard = fake_devices(monkeypatch, cfg, 0)
    read_state = Robot.read_state

    def delayed_snapshot(self):
        snapshot = read_state(self)
        now[0] += 200_000_000
        keyboard.handle_key(key, True)
        return snapshot

    monkeypatch.setattr(Robot, "read_state", delayed_snapshot)
    if key == "esc":
        with pytest.raises(OperationCancelled):
            run.run_replay(path, cfg)
    else:
        run.run_replay(path, cfg)
    assert not sent and closed == ["arm", "hand", "keyboard"]


@pytest.mark.parametrize("stage", ["prepare", "cancel_check"])
def test_interrupt_latches_emergency_before_close(monkeypatch, tmp_path, stage):
    from dexmani_real.robot.commands import OperationCancelled

    path = tmp_path / "demo"
    cfg = demo(path, 1)
    _, sent, closed, _, keyboard = fake_devices(monkeypatch, cfg, 0)
    check, close = Robot.check, Robot.close

    def prepare(self, epoch):
        if stage == "prepare":
            raise KeyboardInterrupt()
        keyboard.handle_key("q", True)
        raise OperationCancelled("prepare cancelled")

    def interrupted_check(self):
        check(self)
        if self.runtime_state.quit_requested:
            raise KeyboardInterrupt()

    def emergency_close(self):
        assert self.runtime_state.estop_request
        assert self.runtime_state.stop_signal.is_set()
        close(self)

    monkeypatch.setattr(Robot, "prepare_streaming", prepare)
    monkeypatch.setattr(Robot, "check", interrupted_check)
    monkeypatch.setattr(Robot, "close", emergency_close)
    with pytest.raises(KeyboardInterrupt):
        run.run_replay(path, cfg)
    assert not sent and closed == ["arm", "hand", "keyboard"]


@pytest.mark.parametrize("fault", [False, True])
def test_tail_without_hand_sample_only_fails_on_io_error(monkeypatch, tmp_path, fault):
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    now, sent, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    connect = Robot.connect
    def setup(self):
        connect(self)
        get = self.hand.get_state
        def sample():
            if not sent:
                return get()
            if fault:
                raise OSError("hand I/O failed")
            return None
        self.hand.get_state = sample
    monkeypatch.setattr(Robot, "connect", setup)
    if fault:
        with pytest.raises(OSError, match="hand I/O failed"):
            run.run_replay(path, cfg)
    else:
        run.run_replay(path, cfg)
        assert now[0] == 1_500_000_000
    assert closed == ["arm", "hand", "keyboard"]


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("edge", ["lower", "upper"])
@pytest.mark.parametrize("offset", [0, 0.5e-6, 2e-6])
def test_file_limits_normalize_then_pass_public_send(monkeypatch, tmp_path, dtype, edge, offset):
    from dexmani_real.robot.commands import RobotCommand
    path = tmp_path / "demo"
    cfg = demo(path, 1)
    arm = np.asarray(cfg.arm.joint_limit_lower if edge == "lower" else cfg.arm.joint_limit_upper)
    hand = np.asarray(cfg.hand.qpos_min_rad if edge == "lower" else cfg.hand.qpos_max_rad)
    sign = -1 if edge == "lower" else 1
    with h5py.File(path / "data.h5", "r+") as data:
        for name, values in (("action_arm_joint_target", arm), ("action_hand_joint_target", hand)):
            del data[name]
            data[name] = np.asarray([values + sign * offset], dtype=dtype)
    if offset > 1e-6:
        with pytest.raises(ValueError, match="absolute limits"):
            load_trajectory(path, cfg)
        return
    trajectory = load_trajectory(path, cfg)
    now, sent, _, _, _ = fake_devices(monkeypatch, cfg, 0)
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    robot = Robot(cfg.arm, cfg.hand, runtime_state=state)
    robot.connect()
    try:
        robot.send_action(RobotCommand(trajectory.action_arm_joint[0],
                                       trajectory.action_hand_joint[0]),
                          expires_ns=now[0] + 2_000_000_000)
        assert len(sent) == 1
    finally:
        robot.close()


def test_hand_feedback_still_checked_at_public_send(monkeypatch, tmp_path):
    from dexmani_real.robot.commands import OperationCancelled
    path = tmp_path / "demo"
    cfg = demo(path, 5)
    _, sent, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    connect = Robot.connect
    def setup(self):
        connect(self)
        get = self.hand.get_state
        self.hand.get_state = lambda: None if sent else get()
    monkeypatch.setattr(Robot, "connect", setup)
    with pytest.raises(OperationCancelled, match="hand feedback unavailable/stale"):
        run.run_replay(path, cfg)
    # arm 发布在 hand 检查之前；整体失败仍不能继续提交第五行。
    assert len(sent) == 4
    assert closed == ["arm", "hand", "keyboard"]
