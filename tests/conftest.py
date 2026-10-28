import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config import RealConfig
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        origin = now = 1_000_000_000

        def at(self, ms):
            self.now = self.origin + round(ms * 1e6)

        def sleep(self, seconds):
            self.now += round(seconds * 1e9)

    clock = Clock()
    monkeypatch.setattr(time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now / 1e9)
    monkeypatch.setattr(time, "perf_counter", lambda: clock.now / 1e9)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


@pytest.fixture
def robot_fixture(clock):
    runtime = RealConfig()
    state = SessionState()
    state.arm()
    robot = Robot(runtime.arm, runtime.hand, run_signal=state.run_signal,
                  stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
                  signal_lock=state.motion_lock)
    robot.owner = threading.get_ident()
    robot.connected = True
    sent, hand_sent = [], []
    health = []

    class Arm:
        is_connected = True

        def check_error(self):
            pass

        def snapshot(self):
            # 模拟独立机械臂进程每 40 ms 发布的来源时钟。
            frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
            frame["qpos"] = runtime.arm.home_qpos
            frame["timestamp_ns"] = clock.origin + (clock.now - clock.origin) // 40_000_000 * 40_000_000
            return frame

        def update_health(self, *, health_until_ns):
            health.append(health_until_ns)

        def prepare(self, run_id, *, service):
            service()
            state.stop_signal.clear()

        def begin_operation(self, run_id):
            state.stop_signal.clear()

        def submit(self, command, **kwargs):
            sent.append((clock.now, command, kwargs))

        def limit_execution_until(self, deadline):
            self.execution_limit = deadline

        def request_stop(self):
            state.stop_signal.set()

        def wait_stopped(self):
            pass

        def close(self):
            self.is_connected = False

    class Hand:
        is_connected = True
        reads = 0
        crc = False

        def get_state(self):
            self.reads += 1
            if self.crc:
                return None
            return XHandState(np.deg2rad(runtime.hand.home_qpos_deg),
                              np.zeros((5, 3), np.float32), np.zeros((5, 120, 3), np.float32))

        def _send_action(self, target):
            hand_sent.append(target.copy())

        def stop(self, *args, **kwargs):
            pass

        def disconnect(self):
            self.is_connected = False

    robot.arm, robot.hand = Arm(), Hand()
    robot.reset_hand_feedback()
    robot.service_feedback()
    return SimpleNamespace(runtime=runtime, state=state, robot=robot, sent=sent,
                           hand_sent=hand_sent, health=health)
