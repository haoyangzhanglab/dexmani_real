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


@pytest.mark.parametrize("kind", ["start", "feedback", "lease", "device_stop"])
def test_replay_rejects_real_boundaries(monkeypatch, tmp_path, kind):
    path = tmp_path / "demo"
    cfg = demo(path, 2)
    now, sent, closed, _, _ = fake_devices(monkeypatch, cfg, 0)
    original = Robot.read_state

    def read(self):
        state = original(self)
        if kind == "start":
            state.arm["qpos"] += 0.3
        elif sent:
            if kind == "feedback":
                state.arm["timestamp_ns"] = 1
            elif kind == "device_stop":
                self.runtime_state.stop_signal.set()
        return state

    monkeypatch.setattr(Robot, "read_state", read)
    if kind == "lease":
        original_send = Robot.send_action

        def send(self, *args, **kwargs):
            original_send(self, *args, **kwargs)
            now[0] += 2_000_000_000

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
