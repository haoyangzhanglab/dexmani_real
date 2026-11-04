import multiprocessing as mp
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.runtime.home import home_arm
from dexmani_real.runtime.state import SessionState


def test_snapshot_contention_reuses_complete_snapshot_without_renewal():
    class CountedRing(SharedRing):
        reads = 0
        @property
        def slots(self):
            self.reads += 1
            return super().slots

    ring = CountedRing(mp.get_context('spawn'), np.dtype([('timestamp_ns', 'i8'), ('value', 'i8')]), 2)
    ring.write(timestamp_ns=123, value=7)
    first = ring.read(fields=('timestamp_ns', 'value'))
    ring.slots[ring.latest_sequence % ring.capacity]['sequence'] = 3
    ring.reads = 0
    stale = ring.read(fields=('value', 'timestamp_ns'))
    assert ring.reads == 2
    assert stale[1] == first[1]
    assert stale[0]['timestamp_ns'] == 123 and stale[0]['value'] == 7
    assert ring.read(fields=('timestamp_ns',)) is None

def test_home_cancellation_after_submission_stops_confirmation():
    import threading
    import time

    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.robot.xarm7.worker import ArmController
    config = RealConfig().arm
    target = np.asarray(config.home_qpos)
    measured = target + .01
    moves = []
    cancelled = threading.Event()
    def move(qpos):
        moves.append(qpos)
        cancelled.set()
    def check_abort():
        if cancelled.is_set():
            raise OperationCancelled('operator stop')
    io = SimpleNamespace(stop=threading.Event(), publish_status=lambda status: None)
    driver = SimpleNamespace(prepare_motion=lambda *a, **kw: None,
                             move_home=move, read=lambda: (measured, np.zeros(7)))
    owner = ArmController(config, io, None, driver)
    with pytest.raises(OperationCancelled, match='operator stop'):
        owner.home(
                   check_abort=check_abort, deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert len(moves) == 1


def test_pointcloud_worker_does_not_reprocess_cached_camera_frame(monkeypatch):
    from dexmani_real.perception import worker
    running = SimpleNamespace(value=True)
    processed, published = [], []
    frames = [dict(timestamp_ns=stamp, rgb=np.array([stamp]), depth=np.array([stamp]))
              for stamp in (100, 200)]

    class CameraRing:
        reads = 0
        @property
        def latest_sequence(self):
            return 1 if self.reads == 0 else 2
        def read(self):
            self.reads += 1
            # 新帧已发布，第一次复制竞争退回上一份缓存，下一次才取到新帧。
            sequence = 1 if self.reads < 3 else 2
            return frames[sequence - 1], sequence

    def process(depth, rgb):
        stamp = int(rgb[0])
        processed.append(stamp)
        if stamp == 200:
            running.value = False
        return np.zeros((2, 6))

    monkeypatch.setattr(worker, 'PointCloudProcessor', lambda **kwargs: SimpleNamespace(process=process))
    monkeypatch.setattr(worker.time, 'sleep', lambda delay: None)
    shared = SimpleNamespace(is_running=running, camera_ring=CameraRing(),
                             pointcloud_ring=SimpleNamespace(field_names={'rgb', 'depth'},
                                                             write=lambda **row: published.append(row)))
    camera = SimpleNamespace(depth_scale_m=.001, color=None, T_xarm_base_from_color=None)
    worker.run_pointcloud_worker(shared, RealConfig().pointcloud, camera, None)
    assert processed == [100, 200]
    assert [(row['source_camera_sequence'], row['timestamp_ns']) for row in published] == [(1, 100), (2, 200)]
    assert [int(row['rgb'][0]) for row in published] == [100, 200]


@pytest.mark.parametrize('part', ['arm_target', 'arm_timeout', 'hand_target', 'hand_timeout'])
def test_unused_home_config_does_not_gate_streaming_but_home_rejects(part):
    from dexmani_real.config.control import HomeConfigurationError
    from dexmani_real.runtime.home import home_hand
    from dexmani_real.runtime.preflight import validate_robot_config
    runtime = RealConfig()
    if part == 'arm_target':
        runtime = replace(runtime, arm=replace(runtime.arm, home_qpos=(float('nan'),) * 7))
    elif part == 'arm_timeout':
        runtime = replace(runtime, arm=replace(runtime.arm, homing=replace(runtime.arm.homing, timeout_s=0)))
    elif part == 'hand_target':
        runtime = replace(runtime, hand=replace(runtime.hand, home_qpos_deg=(float('nan'),) * 12))
    else:
        runtime = replace(runtime, hand=replace(runtime.hand, home_timeout_s=0))
    validate_robot_config(runtime)
    state = SessionState()
    robot = SimpleNamespace(arm_config=runtime.arm, hand_config=runtime.hand)
    # invalid HOME 必须在请求规划或发送任何目标前失败。
    with pytest.raises(HomeConfigurationError):
        if part.startswith('arm'):
            home_arm(state, robot,
                       cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
        else:
            home_hand(state, robot, cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)


def test_streaming_validates_standstill_without_unused_hand_home():
    from dexmani_real.runtime.preflight import validate_robot_config
    runtime = RealConfig()
    bad_arm = replace(runtime.arm, homing=replace(runtime.arm.homing, velocity_convergence_rad_s=0))
    with pytest.raises(ValueError, match='standstill'):
        validate_robot_config(replace(runtime, arm=bad_arm))
    fixed_hand = replace(runtime.hand, enabled=False, home_qpos_deg=(float('nan'),) * 12)
    validate_robot_config(replace(runtime, hand=fixed_hand))


@pytest.mark.parametrize('outcome', ['arrived', 'timeout', 'cancelled'])
@pytest.mark.parametrize('crc', [False, True])
def test_hand_home_keeps_arm_stopped_and_confirms_new_samples(monkeypatch, outcome, crc):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE
    from dexmani_real.robot.xhand.feedback import XHandState
    from dexmani_real.runtime.home import home_hand

    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.runtime.home.time.monotonic_ns', lambda: now[0])
    monkeypatch.setattr('dexmani_real.runtime.home.time.sleep',
                        lambda delay: now.__setitem__(0, now[0] + round(delay * 1e9)))
    cfg, state = RealConfig(), SessionState()
    hand_cfg = replace(cfg.hand, home_timeout_s=.2)
    robot = Robot(cfg.arm, hand_cfg, runtime_state=state)
    robot.connected = True
    stops, sends = [], []
    robot.arm = SimpleNamespace(is_connected=True, check_error=lambda: None,
                                wait_stopped=lambda: None,
                                resume=lambda *a, **kw: pytest.fail('arm resumed for hand HOME'))
    target = np.deg2rad(hand_cfg.home_qpos_deg)
    initial = target + .15
    def send(qpos):
        assert state.stop_signal.is_set()
        sends.append(qpos.copy())
        if outcome == 'cancelled':
            state.request_stop()
        return COMMUNICATION_CRC_ERROR_CODE if crc else 0
    def sample():
        assert state.stop_signal.is_set()
        qpos = target if outcome == 'arrived' else initial
        return XHandState(qpos, None, None, now[0])
    robot.hand = SimpleNamespace(is_connected=True, _send_position=send, get_state=sample,
                                stop=lambda *a, **kw: stops.append('hand'))
    robot.hand_feedback.consume(XHandState(initial, None, None, now[0]))
    robot.hand_stop_required = True
    epoch = state.cancel_epoch
    run = lambda: home_hand(state, robot, cancel_epoch=epoch, cancel_requested=lambda: False)
    if outcome == 'arrived':
        run()
        assert now[0] > 1_000_000_000
    elif outcome == 'timeout':
        with pytest.raises(TimeoutError, match='convergence'):
            run()
    else:
        with pytest.raises(OperationCancelled):
            run()
    assert len(sends) == (2 if crc and outcome != 'cancelled' else 1)
    assert state.stop_signal.is_set() and stops.count('hand') == 2
    assert state.stop_signal.is_set() and not robot.hand_stop_required


@pytest.mark.parametrize('already_home', [False, True])
@pytest.mark.parametrize('arrives', [False, True])
def test_home_sdk_return_does_not_mean_measured_arrival(monkeypatch, already_home, arrives):
    from dexmani_real.robot.xarm7.worker import ArmController
    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.robot.xarm7.worker.time.monotonic_ns', lambda: now[0])
    cfg = RealConfig().arm
    target = np.asarray(cfg.home_qpos)
    moves, samples = [], []
    def read():
        # First settling interval determines whether a move is necessary;
        # a later transient must reset dwell for either path.
        elapsed = now[0] - 1_000_000_000
        q = target if already_home or moves else target + .01
        if elapsed >= 400_000_000 and (not arrives or elapsed < 500_000_000):
            q = target + .01
        if not already_home and not arrives:
            q = target + .01
        samples.append((now[0], q.copy()))
        now[0] += 1_000_000
        return q, np.zeros(7)
    io = SimpleNamespace(publish_status=lambda row: None,
                         stop=SimpleNamespace(wait=lambda dt: now.__setitem__(0, now[0]+round(dt*1e9))))
    owner = ArmController(cfg, io, None, SimpleNamespace(
        read=read, prepare_motion=lambda *a, **kw: None, move_home=lambda q: moves.append(q.copy())))
    if arrives:
        owner.home(check_abort=lambda: None, deadline_ns=now[0]+2_000_000_000)
        np.testing.assert_allclose(owner.status['qpos'], target)
    else:
        with pytest.raises(TimeoutError):
            owner.home(check_abort=lambda: None, deadline_ns=now[0]+2_000_000_000)
    assert len(moves) == int(not already_home)
    assert now[0] >= 1_600_000_000


@pytest.mark.parametrize('hand_pending', [False, True])
def test_arm_home_does_not_touch_hand_or_geometry(hand_pending):
    from dexmani_real.robot.system import Robot
    cfg, state = RealConfig(), SessionState()
    robot = Robot(cfg.arm, replace(cfg.hand, home_qpos_deg=(float('nan'),)*12), runtime_state=state)
    calls = []
    def arm_home(epoch, *, service, abort, deadline_ns):
        service()
        assert not abort()
        assert deadline_ns > 0
        calls.append(epoch)
        state.stop_signal.set()
    robot.arm = SimpleNamespace(home=arm_home, check_error=lambda: None, wait_stopped=lambda: None)
    robot.hand = SimpleNamespace(stop=lambda *a, **kw: pytest.fail('hand write'))
    robot.hand_stop_required = hand_pending
    home_arm(state, robot, cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
    assert calls == [state.cancel_epoch]
    assert robot.hand_stop_required == hand_pending
    # arm HOME 本身不写手；显式整机停止仍须完成此前手部命令的收尾。
    stopped = []
    robot.hand.stop = lambda *a, **kw: stopped.append(True)
    robot.stop()
    robot.service_idle()
    assert stopped == ([True] if hand_pending else [])
    assert not robot.hand_stop_required
