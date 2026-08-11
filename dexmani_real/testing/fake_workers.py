"""Spawn-safe fake workers implementing the production SharedStorage protocols.

These targets are intentionally test-only. They import no device SDK and are
not selected by runtime configuration or any production entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Empty
from typing import Any, Literal

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_COMMAND_DTYPE,
    ARM_STATE_DTYPE,
    CAMERA_FRAME_HEADER_DTYPE,
    HAND_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
)
from dexmani_real.policy.action_protocol import (
    AckStatus,
    RejectReason,
    command_matches_commit,
    make_ack,
    make_stopped_ack,
)
from dexmani_real.shm.shared_storage import new_frame

FakeBehavior = Literal[
    "normal",
    "startup_fail",
    "die_after_ready",
    "heartbeat_stall",
    "sticky_fault",
    "no_prepare",
    "reject_prepare",
]


@dataclass(frozen=True)
class FakeWorkerConfig:
    behavior: FakeBehavior = "normal"
    tick_s: float = 0.005
    trigger_after_s: float = 0.10
    camera_generation: int = 1
    stale_vr: bool = False


def _startup(shared: object, component: str, config: FakeWorkerConfig) -> float | None:
    if config.behavior == "startup_fail":
        shared.error_state.value = True  # type: ignore[attr-defined]
        return None
    now = time.monotonic()
    getattr(shared, f"{component}_heartbeat_s").value = now
    getattr(shared, f"{component}_ready").set()
    return now


def _tick_faults(shared: object, component: str, config: FakeWorkerConfig, started_s: float) -> bool:
    elapsed = time.monotonic() - started_s
    if config.behavior != "heartbeat_stall" or elapsed < config.trigger_after_s:
        getattr(shared, f"{component}_heartbeat_s").value = time.monotonic()
    if elapsed >= config.trigger_after_s:
        if config.behavior == "die_after_ready":
            return False
        if config.behavior == "sticky_fault":
            shared.error_state.value = True  # type: ignore[attr-defined]
    return bool(shared.is_running.value)  # type: ignore[attr-defined]


def fake_arm_worker(shared: object, config: FakeWorkerConfig = FakeWorkerConfig()) -> None:
    started = _startup(shared, "arm", config)
    if started is None:
        return
    state = new_frame(ARM_STATE_DTYPE)
    state["connected"][0] = 1
    state["state_valid"][0] = 1
    pending: np.ndarray | None = None
    while _tick_faults(shared, "arm", config, started):
        source_ns = time.monotonic_ns()
        state["source_monotonic_ns"][0] = source_ns
        state["publish_monotonic_ns"][0] = source_ns
        state["timestamp"][0] = source_ns / 1e9
        shared.arm_state_ring.write(state)  # type: ignore[attr-defined]

        if config.behavior != "no_prepare" and pending is None:
            try:
                command = shared.arm_action_q.get_nowait()  # type: ignore[attr-defined]
            except Empty:
                command = None
            if isinstance(command, np.ndarray) and command.shape == (1,) and command.dtype == ARM_COMMAND_DTYPE:
                if config.behavior == "reject_prepare":
                    shared.arm_ack_ring.write(  # type: ignore[attr-defined]
                        make_ack(command, AckStatus.REJECTED, reject_reason=RejectReason.JOINT_LIMIT)
                    )
                else:
                    shared.arm_ack_ring.write(make_ack(command, AckStatus.RECEIVED))  # type: ignore[attr-defined]
                    shared.arm_ack_ring.write(make_ack(command, AckStatus.PREPARED))  # type: ignore[attr-defined]
                    pending = command.copy()
        if pending is not None:
            commit_result = shared.action_commit_ring.read_latest()  # type: ignore[attr-defined]
            commit = None if commit_result is None else commit_result[0]
            if (
                commit is not None
                and command_matches_commit(pending, commit)
                and time.monotonic_ns() >= int(pending["target_monotonic_ns"][0])
            ):
                state["qpos"][0] = pending["qpos_cmd"][0]
                source_ns = time.monotonic_ns()
                state["source_monotonic_ns"][0] = source_ns
                state["publish_monotonic_ns"][0] = source_ns
                state["timestamp"][0] = source_ns / 1e9
                shared.arm_state_ring.write(state)  # type: ignore[attr-defined]
                shared.arm_ack_ring.write(  # type: ignore[attr-defined]
                    make_ack(pending, AckStatus.APPLIED, applied_monotonic_ns=time.monotonic_ns())
                )
                pending = None
        time.sleep(config.tick_s)
    shared.arm_ack_ring.write(make_stopped_ack())  # type: ignore[attr-defined]


def fake_hand_worker(shared: object, config: FakeWorkerConfig = FakeWorkerConfig()) -> None:
    started = _startup(shared, "hand", config)
    if started is None:
        return
    state = new_frame(HAND_STATE_DTYPE)
    state["connected"][0] = 1
    state["state_valid"][0] = 1
    state["send_healthy"][0] = 1
    state["read_healthy"][0] = 1
    pending: np.ndarray | None = None
    last_sequence = 0
    while _tick_faults(shared, "hand", config, started):
        source_ns = time.monotonic_ns()
        state["source_monotonic_ns"][0] = source_ns
        state["publish_monotonic_ns"][0] = source_ns
        state["timestamp"][0] = source_ns / 1e9
        shared.hand_state_ring.write(state)  # type: ignore[attr-defined]

        if config.behavior != "no_prepare" and pending is None:
            result = shared.hand_cmd_ring.read_latest()  # type: ignore[attr-defined]
            if result is not None and result[2] != last_sequence:
                command, _timestamp_ns, last_sequence = result
                if command.shape == (1,) and command.dtype == HAND_COMMAND_DTYPE:
                    if config.behavior == "reject_prepare":
                        shared.hand_ack_ring.write(  # type: ignore[attr-defined]
                            make_ack(command, AckStatus.REJECTED, reject_reason=RejectReason.JOINT_LIMIT)
                        )
                    else:
                        shared.hand_ack_ring.write(make_ack(command, AckStatus.RECEIVED))  # type: ignore[attr-defined]
                        shared.hand_ack_ring.write(make_ack(command, AckStatus.PREPARED))  # type: ignore[attr-defined]
                        pending = command.copy()
        if pending is not None:
            commit_result = shared.action_commit_ring.read_latest()  # type: ignore[attr-defined]
            commit = None if commit_result is None else commit_result[0]
            if (
                commit is not None
                and command_matches_commit(pending, commit)
                and time.monotonic_ns() >= int(pending["target_monotonic_ns"][0])
            ):
                state["qpos"][0] = pending["qpos_cmd"][0]
                source_ns = time.monotonic_ns()
                state["source_monotonic_ns"][0] = source_ns
                state["publish_monotonic_ns"][0] = source_ns
                state["timestamp"][0] = source_ns / 1e9
                shared.hand_state_ring.write(state)  # type: ignore[attr-defined]
                shared.hand_ack_ring.write(  # type: ignore[attr-defined]
                    make_ack(pending, AckStatus.APPLIED, applied_monotonic_ns=time.monotonic_ns())
                )
                pending = None
        time.sleep(config.tick_s)
    shared.hand_ack_ring.write(make_stopped_ack())  # type: ignore[attr-defined]


def fake_camera_worker(shared: object, config: FakeWorkerConfig = FakeWorkerConfig()) -> None:
    started = _startup(shared, "camera", config)
    if started is None:
        return
    generation = int(config.camera_generation)
    frame_number = 0
    rgb_shape = shared.camera_ring._rgb_shape  # type: ignore[attr-defined]
    depth_shape = shared.camera_ring._depth_shape  # type: ignore[attr-defined]
    pc_shape = shared.camera_ring._pc_shape  # type: ignore[attr-defined]
    assert rgb_shape is not None and depth_shape is not None
    while _tick_faults(shared, "camera", config, started):
        frame_number += 1
        if frame_number == 3:
            generation += 1
        now_s = time.monotonic()
        header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
        header["timestamp"][0] = now_s
        header["capture_monotonic_s"][0] = now_s
        header["source_monotonic_ns"][0] = time.monotonic_ns()
        header["camera_generation"][0] = generation
        header["frame_number"][0] = frame_number
        header["rgb_size"][0] = int(np.prod(rgb_shape))
        header["depth_size"][0] = int(np.prod(depth_shape) * 2)
        header["rgb_shape_h"][0], header["rgb_shape_w"][0], header["rgb_shape_c"][0] = rgb_shape
        header["depth_shape_h"][0], header["depth_shape_w"][0] = depth_shape
        rgb = np.full(rgb_shape, frame_number % 255, dtype=np.uint8)
        depth = np.full(depth_shape, frame_number, dtype=np.uint16)
        pointcloud = None if pc_shape is None else np.zeros(pc_shape, dtype=np.float32)
        shared.camera_ring.write(header, rgb, depth, pointcloud=pointcloud)  # type: ignore[attr-defined]
        time.sleep(config.tick_s)


def fake_vr_worker(shared: object, config: FakeWorkerConfig = FakeWorkerConfig()) -> None:
    started = _startup(shared, "vr", config)
    if started is None:
        return
    sequence = 0
    while _tick_faults(shared, "vr", config, started):
        sequence += 1
        frame: Any = np.zeros(1, dtype=shared.vr_ring.dtype)  # type: ignore[attr-defined]
        frame["wrist_quat_wxyz"][0] = (1.0, 0.0, 0.0, 0.0)
        frame["head_quat_wxyz"][0] = (1.0, 0.0, 0.0, 0.0)
        frame["sequence_id"][0] = sequence
        now_ns = time.monotonic_ns()
        source_ns = now_ns - 10_000_000_000 if config.stale_vr else now_ns
        frame["head_sequence_id"][0] = sequence
        frame["head_recv_ts_ns"][0] = source_ns
        frame["local_recv_ns"][0] = source_ns
        shared.vr_ring.write(frame)  # type: ignore[attr-defined]
        time.sleep(config.tick_s)


def fake_policy_worker(shared: object, config: FakeWorkerConfig = FakeWorkerConfig()) -> None:
    started = _startup(shared, "policy", config)
    if started is None:
        return
    while _tick_faults(shared, "policy", config, started):
        time.sleep(config.tick_s)
