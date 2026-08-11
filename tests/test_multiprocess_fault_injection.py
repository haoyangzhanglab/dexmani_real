from __future__ import annotations

import threading
import time
import uuid
from multiprocessing import shared_memory
from queue import Full

import numpy as np
import pytest

from dexmani_real.policy.action_protocol import SafeCommandPublisher, make_command_frame
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.runtime.processes import shutdown_processes_verified, spawn_context
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig, run_supervisor, wait_subsystem_ready
from dexmani_real.testing.fake_workers import (
    FakeWorkerConfig,
    fake_arm_worker,
    fake_camera_worker,
    fake_hand_worker,
    fake_policy_worker,
    fake_vr_worker,
)


def _shared() -> SharedStorage:
    return SharedStorage.create(
        prefix=f"fault_injection_{uuid.uuid4().hex}",
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
            camera_pc_shape=(1, 6),
        ),
        mp_context=spawn_context(),
    )


def _candidate(shared: SharedStorage, *, action_id: int = 1) -> ActionCandidate:
    now_ns = time.monotonic_ns()
    return ActionCandidate(
        observation_id=action_id,
        session_generation=int(shared.session_generation.value),
        policy_epoch=int(shared.policy_epoch.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + 50_000_000,
        valid_until_monotonic_ns=now_ns + 500_000_000,
        arm_qpos=np.full(7, 0.01 * action_id),
        hand_qpos=np.full(12, 0.01 * action_id),
        chunk_id=action_id,
    )


def _stop(shared: SharedStorage, processes: list[object]) -> None:
    if not bool(getattr(shared, "_closed", False)):
        shutdown_processes_verified(shared, processes, graceful_timeout_s=1.0)


def test_spawn_runtime_normal_start_supervise_exit_and_shared_memory_cleanup() -> None:
    shared = _shared()
    process = spawn_context().Process(target=fake_policy_worker, args=(shared,), name="policy")
    process.start()
    ring_name = shared.policy_metrics_ring.name
    try:
        assert wait_subsystem_ready(shared, [("policy", shared.policy_ready, 2.0)], [process])
        timer = threading.Timer(0.1, lambda: setattr(shared.quit_requested, "value", True))
        timer.start()
        reason, normal = run_supervisor(
            shared,
            [process],
            ["policy"],
            {"policy": shared.policy_heartbeat_s},
            status_interval_s=0.05,
            heartbeat_timeouts_s={"policy": 0.5},
            supervisor_hz=100.0,
        )
        timer.join()
        assert normal and reason == "explicit quit"
        report = shutdown_processes_verified(shared, [process], graceful_timeout_s=1.0)
        assert report.all_stopped and report.shared_closed
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=ring_name)
    finally:
        _stop(shared, [process])


def test_spawn_worker_startup_failure_sets_sticky_fault_and_stops_cleanly() -> None:
    shared = _shared()
    process = spawn_context().Process(
        target=fake_arm_worker,
        args=(shared, FakeWorkerConfig(behavior="startup_fail")),
        name="arm",
    )
    process.start()
    try:
        assert not wait_subsystem_ready(shared, [("arm", shared.arm_ready, 1.0)], [process])
        assert shared.error_state.value
    finally:
        _stop(shared, [process])


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("die_after_ready", "process died"),
        ("heartbeat_stall", "heartbeat timeout"),
        ("sticky_fault", "error_state set"),
    ],
)
def test_supervisor_fault_injection_priorities_coordinate_stop(behavior: str, expected: str) -> None:
    shared = _shared()
    config = FakeWorkerConfig(behavior=behavior, trigger_after_s=0.03)
    process = spawn_context().Process(target=fake_policy_worker, args=(shared, config), name="policy")
    process.start()
    try:
        assert wait_subsystem_ready(shared, [("policy", shared.policy_ready, 2.0)], [process])
        reason, normal = run_supervisor(
            shared,
            [process],
            ["policy"],
            {"policy": shared.policy_heartbeat_s},
            heartbeat_timeouts_s={"policy": 0.06},
            supervisor_hz=200.0,
        )
        assert not normal and expected in reason
        assert int(shared.safety_state.value) == 3
        if behavior == "sticky_fault":
            assert shared.error_state.value
    finally:
        _stop(shared, [process])


def test_real_prepare_commit_and_applied_ack_with_fake_arm_and_hand() -> None:
    shared = _shared()
    context = spawn_context()
    processes = [
        context.Process(target=fake_arm_worker, args=(shared,), name="arm"),
        context.Process(target=fake_hand_worker, args=(shared,), name="hand"),
    ]
    for process in processes:
        process.start()
    try:
        assert wait_subsystem_ready(
            shared,
            [("arm", shared.arm_ready, 2.0), ("hand", shared.hand_ready, 2.0)],
            processes,
        )
        candidate = _candidate(shared)
        publisher = SafeCommandPublisher(shared)
        assert publisher.publish(candidate, prepare_timeout_s=0.5)
        assert publisher.wait_applied(candidate, timeout_s=1.0)
        arm = shared.arm_state_ring.read_latest()
        hand = shared.hand_state_ring.read_latest()
        assert arm is not None and hand is not None
        np.testing.assert_allclose(arm[0]["qpos"][0], candidate.arm_qpos)
        np.testing.assert_allclose(hand[0]["qpos"][0], candidate.hand_qpos)
    finally:
        _stop(shared, processes)


@pytest.mark.parametrize(("arm_behavior", "hand_behavior"), [("no_prepare", "normal"), ("normal", "reject_prepare")])
def test_prepare_timeout_and_unilateral_rejection_never_commit(
    arm_behavior: str,
    hand_behavior: str,
) -> None:
    shared = _shared()
    context = spawn_context()
    processes = [
        context.Process(
            target=fake_arm_worker,
            args=(shared, FakeWorkerConfig(behavior=arm_behavior)),
            name="arm",
        ),
        context.Process(
            target=fake_hand_worker,
            args=(shared, FakeWorkerConfig(behavior=hand_behavior)),
            name="hand",
        ),
    ]
    for process in processes:
        process.start()
    try:
        assert wait_subsystem_ready(
            shared,
            [("arm", shared.arm_ready, 2.0), ("hand", shared.hand_ready, 2.0)],
            processes,
        )
        assert not SafeCommandPublisher(shared).publish(_candidate(shared), prepare_timeout_s=0.1)
        assert shared.action_commit_ring.read_latest() is None
    finally:
        _stop(shared, processes)


def test_bounded_arm_queue_backpressure_is_preserved() -> None:
    shared = _shared()
    process = spawn_context().Process(
        target=fake_arm_worker,
        args=(shared, FakeWorkerConfig(behavior="no_prepare")),
        name="arm",
    )
    process.start()
    try:
        assert wait_subsystem_ready(shared, [("arm", shared.arm_ready, 2.0)], [process])
        shared.arm_action_q.put(make_command_frame(_candidate(shared, action_id=1), actuator="arm"), timeout=0.1)
        shared.arm_action_q.put(make_command_frame(_candidate(shared, action_id=2), actuator="arm"), timeout=0.1)
        with pytest.raises(Full):
            shared.arm_action_q.put(make_command_frame(_candidate(shared, action_id=3), actuator="arm"), timeout=0.05)
    finally:
        _stop(shared, [process])


def test_vr_staleness_and_camera_generation_switch_use_real_rings() -> None:
    shared = _shared()
    context = spawn_context()
    processes = [
        context.Process(
            target=fake_vr_worker,
            args=(shared, FakeWorkerConfig(stale_vr=True)),
            name="vr",
        ),
        context.Process(target=fake_camera_worker, args=(shared,), name="camera"),
    ]
    for process in processes:
        process.start()
    try:
        assert wait_subsystem_ready(
            shared,
            [("vr", shared.vr_ready, 2.0), ("camera", shared.camera_ready, 2.0)],
            processes,
        )
        deadline = time.monotonic() + 2.0
        camera_result = None
        while time.monotonic() < deadline:
            camera_result = shared.camera_ring.read_latest()
            if camera_result is not None and int(camera_result[0]["camera_generation"][0]) >= 2:
                break
            time.sleep(0.01)
        vr_result = shared.vr_ring.read_latest()
        assert vr_result is not None
        assert time.monotonic_ns() - int(vr_result[0]["local_recv_ns"][0]) > 5_000_000_000
        assert camera_result is not None
        assert int(camera_result[0]["camera_generation"][0]) == 2
        assert int(camera_result[0]["frame_number"][0]) >= 3
    finally:
        _stop(shared, processes)
