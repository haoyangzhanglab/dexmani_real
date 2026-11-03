import multiprocessing as mp
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.runtime.home import home_robot
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


def test_home_planner_construction_uses_same_total_deadline(monkeypatch):
    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.runtime.home.time.monotonic_ns', lambda: now[0])
    monkeypatch.setattr('dexmani_real.runtime.home.time.monotonic', lambda: now[0] / 1e9)
    runtime = RealConfig()
    state = SessionState()
    robot = SimpleNamespace(arm_config=runtime.arm, hand_config=replace(runtime.hand, enabled=False),
                            stop=lambda: None)
    def planner():
        now[0] += 61_000_000_000
        return SimpleNamespace()
    with pytest.raises(TimeoutError):
        home_robot(state, robot, planner, cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)


def test_home_sdk_return_does_not_mean_measured_arrival():
    import threading
    import time

    from dexmani_real.robot.xarm7.worker import ArmController
    config = RealConfig().arm
    target = np.asarray(config.home_qpos)
    measured = target + .01
    moves = []
    io = SimpleNamespace(stop=threading.Event(), publish_status=lambda status: None)
    driver = SimpleNamespace(enter_mode=lambda *a, **kw: None,
                             move_home_milestone=moves.append,
                             read=lambda: (measured, np.zeros(7)))
    owner = ArmController(config, io, None, driver)
    owner.status['qpos'] = measured
    owner.status['timestamp_ns'] = time.monotonic_ns()
    with pytest.raises(TimeoutError, match='not confirmed'):
        owner.home(np.stack((measured, target)),
                   check_abort=lambda: None, deadline_ns=time.monotonic_ns() + 20_000_000)
    assert len(moves) == 1


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
    driver = SimpleNamespace(enter_mode=lambda *a, **kw: None,
                             move_home_milestone=move, read=lambda: (measured, np.zeros(7)))
    owner = ArmController(config, io, None, driver)
    owner.status['qpos'] = measured
    owner.status['timestamp_ns'] = time.monotonic_ns()
    with pytest.raises(OperationCancelled, match='operator stop'):
        owner.home(np.stack((measured, target)),
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
            home_robot(state, robot, lambda: pytest.fail('invalid HOME reached planner'),
                       cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
        else:
            home_hand(state, robot, cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)


def test_streaming_keeps_standstill_thresholds_and_disabled_hand_geometry_checks():
    from dexmani_real.runtime.preflight import validate_robot_config
    runtime = RealConfig()
    bad_arm = replace(runtime.arm, homing=replace(runtime.arm.homing, velocity_convergence_rad_s=0))
    with pytest.raises(ValueError, match='standstill'):
        validate_robot_config(replace(runtime, arm=bad_arm))
    fixed_hand = replace(runtime.hand, enabled=False, home_qpos_deg=(float('nan'),) * 12)
    with pytest.raises(ValueError, match='finite'):
        validate_robot_config(replace(runtime, hand=fixed_hand))


@pytest.mark.parametrize('bad_sample', [None, 'middle', 'start'])
def test_home_checks_only_direct_path_without_soft_escape(bad_sample):
    from dexmani_real.control.homing import HomePlanner, HomePathRejected
    planner = HomePlanner.__new__(HomePlanner)
    planner.arm = RealConfig().arm
    planner.limits = np.tile((-4., 4.), (7, 1))
    current, target = np.zeros(7), np.full(7, .1)
    visited = []
    def check(q):
        visited.append(np.array(q))
        if bad_sample == 'middle' and .04 < q[0] < .06:
            raise HomePathRejected('collision')
        return (-.01 if bad_sample == 'start' and q[0] == 0 else .01), .1
    planner._check_state = check
    if bad_sample:
        with pytest.raises(HomePathRejected):
            planner.plan(current, target)
    else:
        np.testing.assert_array_equal(planner.plan(current, target), [current, target])
        assert len(visited) > 2
    assert all(np.allclose(q, q[0]) for q in visited)


def test_owner_rejects_obsolete_multi_waypoint_home_before_motion():
    from dexmani_real.robot.xarm7.worker import ArmController
    config = RealConfig().arm
    target = np.asarray(config.home_qpos)
    owner = ArmController(config, None, None, SimpleNamespace())
    with pytest.raises(ValueError, match='endpoints'):
        owner.home(np.stack((target + .02, target + .01, target)),
                   check_abort=lambda: None, deadline_ns=1)


@pytest.mark.parametrize('changed', ['position', 'velocity'])
def test_home_rechecks_measured_start_after_mode_change(changed):
    import time
    from dexmani_real.robot.xarm7.worker import ArmController
    config = RealConfig().arm
    target = np.asarray(config.home_qpos)
    planned_start = target + .02
    measured = [planned_start.copy()]
    velocity = np.zeros(7)
    def enter_mode(*args, **kwargs):
        if changed == 'position':
            measured[0] = planned_start + .01
        else:
            velocity[:] = 1
    driver = SimpleNamespace(
        enter_mode=enter_mode,
        read=lambda: (measured[0], velocity),
        move_home_milestone=lambda q: pytest.fail('moved from an unchecked start'),
    )
    owner = ArmController(config, SimpleNamespace(publish_status=lambda row: None), None, driver)
    owner.status['qpos'] = planned_start
    owner.status['timestamp_ns'] = time.monotonic_ns()
    with pytest.raises(RuntimeError, match='planned path start|HOME preparation'):
        owner.home(np.stack((planned_start, target)), check_abort=lambda: None,
                   deadline_ns=time.monotonic_ns() + 1_000_000_000)
