from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dexmani_real.config import defaults
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.planning.preflight import PreflightCertificate, create_preflight_certificate, verify_preflight_binding
from dexmani_real.policy.action_protocol import (
    ARM_COMMAND_DTYPE,
    COMMIT_DTYPE,
    AckStatus,
    ActionSafetyGate,
    ActionSafetyGateConfig,
    JointActionScheduler,
    RejectReason,
    SafeCommandPublisher,
    command_matches_commit,
    make_ack,
    make_command_frame,
    make_stopped_ack,
    publish_joint_targets,
    validate_worker_command,
)
from dexmani_real.policy.inference_process import InferenceConfig, encode_candidate
from dexmani_real.policy.learned_coordinator import CoordinatorTick, LearnedCoordinatorConfig, LearnedPolicyCoordinator
from dexmani_real.policy.observation_sources import SharedObservationSource
from dexmani_real.policy.runtime import (
    ActionCandidate,
    ActionChunk,
    ActionSpec,
    FrozenArrayMap,
    ModalitySpec,
    ObservationSnapshot,
    ObservationSpec,
)
from dexmani_real.policy.tensor_block import ObservationTensorBlock
from dexmani_real.policy.vr_teleop_policy import (
    PolicyConfig,
    _matching_source_sequence,
    _read_causal_structured_frame,
    _recording_provenance,
)
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.hand_process import HandProcessConfig
from dexmani_real.runtime.processes import shutdown_processes_verified, supervisor_exit_reason
from dexmani_real.runtime.status import ExitReason
from dexmani_real.sensor.camera_process import pack_camera_frame
from dexmani_real.sensor.clock_sync import DeviceClockMapper
from dexmani_real.shm.shared_storage import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    SharedStorage,
    SharedStorageConfig,
    new_frame,
)


def _candidate(*, now_ns: int, action_id: int = 1, epoch: int = 3, step_index: int = 0) -> ActionCandidate:
    return ActionCandidate(
        observation_id=11,
        session_generation=7,
        policy_epoch=epoch,
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + 10_000_000 * (step_index + 1),
        valid_until_monotonic_ns=now_ns + 100_000_000,
        arm_qpos=np.zeros(7),
        hand_qpos=np.zeros(12),
        chunk_id=5,
        step_index=step_index,
    )


def _snapshot(now_ns: int) -> ObservationSnapshot:
    empty = FrozenArrayMap(())
    return ObservationSnapshot(11, now_ns, empty, empty, empty, empty, session_generation=7)


def _gate(*, geometry: bool = False) -> ActionSafetyGate:
    return ActionSafetyGate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=(-2.0,) * 7,
            arm_joint_upper_rad=(2.0,) * 7,
            hand_joint_lower_rad=(-2.0,) * 12,
            hand_joint_upper_rad=(2.0,) * 12,
            arm_max_velocity_rad_s=1.0,
            hand_max_velocity_rad_s=1.0,
            require_geometry_checks=geometry,
        ),
        workspace_check=(lambda _a, _b: True) if geometry else None,
        transition_collision_check=(lambda _a, _b, _c, _d: True) if geometry else None,
        table_clearance_check=(lambda _a, _b, _c, _d: True) if geometry else None,
    )


def test_runtime_config_precedence_is_immutable_and_hash_is_canonical(tmp_path: Path) -> None:
    original_speed = defaults.arm.max_joint_velocity_deg_per_s
    config_path = tmp_path / "runtime.json"
    config_path.write_text(json.dumps({"arm": {"max_joint_velocity_deg_per_s": 80.0}}), encoding="utf-8")

    resolved = resolve_runtime_config(
        json_path=config_path,
        cli_overrides={"arm.max_joint_velocity_deg_per_s": 90.0, "arm.ip": None},
    )
    identical = resolve_runtime_config(json_data=resolved.to_dict())

    assert resolved.arm.max_joint_velocity_deg_per_s == 90.0
    assert resolved.arm.ip == defaults.arm.ip
    assert defaults.arm.max_joint_velocity_deg_per_s == original_speed
    assert identical.canonical_json == resolved.canonical_json
    assert identical.sha256 == resolved.sha256
    with pytest.raises((AttributeError, TypeError)):
        resolved.arm.max_joint_velocity_deg_per_s = 1.0


def test_runtime_config_rejects_unknown_and_cross_invalid_values() -> None:
    with pytest.raises(TypeError, match="unknown runtime config"):
        resolve_runtime_config(json_data={"arm": {"mystery": 1}})
    with pytest.raises(ValueError, match="stall abort"):
        resolve_runtime_config(
            cli_overrides={
                "camera.max_frame_age_s": 1.0,
                "camera.recording_stall_abort_s": 0.5,
            }
        )
    with pytest.raises(TypeError, match="finite number"):
        resolve_runtime_config(cli_overrides={"arm.max_joint_velocity_deg_per_s": "fast"})
    with pytest.raises(TypeError, match="string or null"):
        resolve_runtime_config(cli_overrides={"camera.serial": 1234})
    with pytest.raises(ValueError, match="workspace.*ordered"):
        resolve_runtime_config(cli_overrides={"policy.workspace.x_min": 0.8, "policy.workspace.x_max": 0.7})

    disabled_delta = resolve_runtime_config(json_data={"hand": {"max_delta_rad": None}})
    assert disabled_delta.hand.max_delta_rad is None


def test_resolved_safety_and_hand_values_reach_worker_and_ipc_configs() -> None:
    home_values = list(defaults.hand.home_qpos_deg)
    home_values[0] = 1.0
    home_deg = tuple(home_values)
    runtime = resolve_runtime_config(
        cli_overrides={
            "hand.home_qpos_deg": home_deg,
            "hand.max_delta_rad": 0.05,
            "safety.max_consecutive_recoveries": 7,
        }
    )

    arm_cfg = ArmLoopConfig.from_runtime(runtime)
    hand_cfg = HandProcessConfig.from_runtime(runtime)
    shared_cfg = SharedStorageConfig.from_runtime(runtime)

    assert arm_cfg.max_consecutive_recoveries == 7
    assert hand_cfg.max_delta_rad == pytest.approx(0.05)
    np.testing.assert_allclose(hand_cfg.home_qpos_rad, np.deg2rad(home_deg))
    np.testing.assert_allclose(shared_cfg.hand_home_qpos_rad, hand_cfg.home_qpos_rad)


def test_recording_capability_is_resolved_without_mutating_defaults() -> None:
    assert defaults.policy.recording_enabled
    runtime = resolve_runtime_config(cli_overrides={"policy.recording_enabled": False})

    assert not runtime.policy.recording_enabled
    assert not PolicyConfig.from_runtime(runtime).recording_enabled
    assert defaults.policy.recording_enabled


def test_action_gate_clamps_by_dt_and_fails_closed_without_geometry() -> None:
    now_ns = time.monotonic_ns()
    candidate = _candidate(now_ns=now_ns)
    candidate = ActionCandidate(
        **{
            **candidate.__dict__,
            "arm_qpos": np.full(7, 0.5),
            "hand_qpos": np.full(12, 0.5),
        }
    )
    result = _gate().evaluate(
        candidate,
        snapshot=_snapshot(now_ns),
        current_arm_qpos=np.zeros(7),
        current_hand_qpos=np.zeros(12),
        expected_session_generation=7,
        expected_policy_epoch=3,
        now_monotonic_ns=now_ns,
        dt_s=0.1,
    )
    assert result.accepted and result.delta_clamped and result.candidate is not None
    np.testing.assert_allclose(result.candidate.arm_qpos, 0.1)
    np.testing.assert_allclose(result.candidate.hand_qpos, 0.1)

    missing_geometry = ActionSafetyGate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=(-1.0,) * 7,
            arm_joint_upper_rad=(1.0,) * 7,
            hand_joint_lower_rad=(-1.0,) * 12,
            hand_joint_upper_rad=(1.0,) * 12,
            arm_max_velocity_rad_s=1.0,
            hand_max_velocity_rad_s=1.0,
            require_geometry_checks=True,
        )
    )
    rejected = missing_geometry.evaluate(
        candidate,
        snapshot=_snapshot(now_ns),
        current_arm_qpos=np.zeros(7),
        current_hand_qpos=np.zeros(12),
        expected_session_generation=7,
        expected_policy_epoch=3,
        now_monotonic_ns=now_ns,
        dt_s=0.1,
    )
    assert not rejected.accepted
    assert "geometry" in rejected.reason


def test_joint_publisher_rejects_missing_or_reduced_geometry_gate() -> None:
    shared = Mock()

    assert not publish_joint_targets(shared, np.zeros(7))
    assert not publish_joint_targets(shared, np.zeros(7), safety_gate=_gate(geometry=False))
    assert not shared.arm_command_seq.get_lock.called


def test_joint_publisher_returns_the_committed_dt_clamped_candidate() -> None:
    prefix = f"publisher_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 2, 3),
            camera_depth_shape=(2, 2),
            camera_pc_shape=(1, 6),
        ),
    )
    try:
        arm_state = new_frame(ARM_STATE_DTYPE)
        arm_state["qpos"][0] = 0.0
        arm_state["state_valid"][0] = 1
        arm_state["source_monotonic_ns"][0] = time.monotonic_ns()
        shared.arm_state_ring.write(arm_state)
        with patch("dexmani_real.policy.action_protocol.SafeCommandPublisher.publish", return_value=True):
            candidate = publish_joint_targets(
                shared,
                np.full(7, 0.5),
                dt_s=0.1,
                safety_gate=_gate(geometry=True),
            )

        assert candidate is not None and candidate.arm_qpos is not None
        np.testing.assert_allclose(candidate.arm_qpos, 0.1)

        with patch("dexmani_real.policy.action_protocol.SafeCommandPublisher.publish") as publish:
            assert (
                publish_joint_targets(
                    shared,
                    np.zeros(7),
                    observation_anchor_monotonic_ns=time.monotonic_ns() - 1_000_000_000,
                    safety_gate=_gate(geometry=True),
                )
                is None
            )
        publish.assert_not_called()

        arm_state["source_monotonic_ns"][0] = time.monotonic_ns() - 1_000_000_000
        shared.arm_state_ring.write(arm_state)
        with patch("dexmani_real.policy.action_protocol.SafeCommandPublisher.publish") as publish:
            assert publish_joint_targets(shared, np.zeros(7), safety_gate=_gate(geometry=True)) is None
        publish.assert_not_called()
    finally:
        shared.close()


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda frame, now: frame["policy_epoch"].__setitem__(0, 2), RejectReason.OLD_EPOCH),
        (lambda frame, now: frame["action_id"].__setitem__(0, 4), RejectReason.OUT_OF_ORDER),
        (lambda frame, now: frame["valid_until_monotonic_ns"].__setitem__(0, now - 1), RejectReason.EXPIRED),
        (lambda frame, now: frame["qpos_cmd"][0].__setitem__(0, np.nan), RejectReason.NONFINITE),
    ],
)
def test_worker_rejects_stale_or_invalid_commands(mutate, expected: RejectReason) -> None:
    now_ns = time.monotonic_ns()
    frame = make_command_frame(_candidate(now_ns=now_ns, action_id=5), actuator="arm")
    mutate(frame, now_ns)
    reason = validate_worker_command(
        frame,
        dtype=ARM_COMMAND_DTYPE,
        expected_session_generation=7,
        minimum_policy_epoch=3,
        last_action_id=4,
        now_monotonic_ns=now_ns,
        joint_lower_rad=np.full(7, -1.0),
        joint_upper_rad=np.full(7, 1.0),
    )
    assert reason is expected


def test_stop_ack_is_explicit_and_has_no_action_identity() -> None:
    applied_ns = time.monotonic_ns()
    ack = make_stopped_ack(applied_monotonic_ns=applied_ns)
    assert int(ack["status"][0]) == 6
    assert int(ack["action_id"][0]) == 0
    assert int(ack["applied_monotonic_ns"][0]) == applied_ns


def test_commit_must_match_the_complete_prepared_command_identity() -> None:
    now_ns = time.monotonic_ns()
    command = make_command_frame(_candidate(now_ns=now_ns), actuator="arm")
    commit = np.zeros(1, dtype=COMMIT_DTYPE)
    for name in (
        "session_generation",
        "policy_epoch",
        "observation_id",
        "action_id",
        "chunk_id",
        "step_index",
        "created_monotonic_ns",
        "target_monotonic_ns",
        "valid_until_monotonic_ns",
        "is_hold",
    ):
        commit[name][0] = command[name][0]
    commit["committed_monotonic_ns"][0] = now_ns + 1

    assert command_matches_commit(command, commit)
    commit["observation_id"][0] += 1
    assert not command_matches_commit(command, commit)
    commit["observation_id"][0] = command["observation_id"][0]
    commit["committed_monotonic_ns"][0] = command["target_monotonic_ns"][0] + 1
    assert not command_matches_commit(command, commit)


def test_chunk_scheduler_preserves_committed_and_replaces_uncommitted() -> None:
    now_ns = time.monotonic_ns()
    scheduler = JointActionScheduler(ActionSpec(chunk_length=2, dt_s=0.01))
    first = (_candidate(now_ns=now_ns, action_id=1, step_index=0), _candidate(now_ns=now_ns, action_id=2, step_index=1))
    scheduler.submit(ActionChunk(5, first), now_monotonic_ns=now_ns)
    scheduler.mark_committed(1)
    replacement = (
        _candidate(now_ns=now_ns, action_id=3, step_index=0),
        _candidate(now_ns=now_ns, action_id=4, step_index=1),
    )
    scheduler.submit(ActionChunk(5, replacement), now_monotonic_ns=now_ns)

    assert [step.action_id for step in scheduler.pending] == [1, 3, 4]

    late_now = now_ns + 200_000_000
    scheduler.submit(ActionChunk(5, replacement), now_monotonic_ns=late_now)
    assert scheduler.all_late
    hold = scheduler.make_coordinated_hold(
        template=replacement[0],
        arm_qpos=np.zeros(7),
        hand_qpos=np.zeros(12),
        action_id=10,
        now_monotonic_ns=late_now,
    )
    assert hold.is_hold and hold.target_monotonic_ns > late_now


def test_chunk_scheduler_opens_prepare_window_before_target() -> None:
    now_ns = time.monotonic_ns()
    scheduler = JointActionScheduler(ActionSpec(chunk_length=1, dt_s=0.01))
    candidate = _candidate(now_ns=now_ns, action_id=1)
    scheduler.submit(ActionChunk(5, (candidate,)), now_monotonic_ns=now_ns)

    assert scheduler.pop_ready(lead_time_s=0.005, now_monotonic_ns=now_ns) is None
    assert scheduler.pop_ready(lead_time_s=0.010, now_monotonic_ns=now_ns) == candidate


def test_chunk_scheduler_keeps_action_ids_in_order_when_multiple_steps_are_ready() -> None:
    now_ns = time.monotonic_ns()
    scheduler = JointActionScheduler(ActionSpec(chunk_length=2, dt_s=0.01))
    first = _candidate(now_ns=now_ns, action_id=1, step_index=0)
    second = _candidate(now_ns=now_ns, action_id=2, step_index=1)
    scheduler.submit(ActionChunk(5, (first, second)), now_monotonic_ns=now_ns)

    stalled_tick_ns = now_ns + 15_000_000
    assert scheduler.pop_ready(lead_time_s=0.010, now_monotonic_ns=stalled_tick_ns) == first
    assert scheduler.pop_ready(lead_time_s=0.010, now_monotonic_ns=stalled_tick_ns) == second


def test_learned_coordinator_normalizes_backend_protocol_metadata() -> None:
    prefix = f"learned_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    spec = ObservationSpec((ModalitySpec("arm_qpos", (7,), "float64"),))
    block = ObservationTensorBlock.create(f"{prefix}_tensor", spec)
    try:
        source_ns = time.monotonic_ns()
        arm_frame = new_frame(ARM_STATE_DTYPE)
        arm_frame["qpos"][0] = np.zeros(7)
        arm_frame["source_monotonic_ns"][0] = source_ns
        arm_frame["publish_monotonic_ns"][0] = source_ns + 1
        arm_frame["state_valid"][0] = 1
        shared.arm_state_ring.write(arm_frame)
        hand_frame = new_frame(HAND_STATE_DTYPE)
        hand_frame["qpos"][0] = np.zeros(12)
        hand_frame["source_monotonic_ns"][0] = source_ns
        hand_frame["publish_monotonic_ns"][0] = source_ns + 1
        hand_frame["state_valid"][0] = 1
        shared.hand_state_ring.write(hand_frame)

        inference = InferenceConfig("unused.module:Backend", spec, ActionSpec(chunk_length=1))
        coordinator = LearnedPolicyCoordinator(
            shared,
            inference,
            block,
            _gate(),
            config=LearnedCoordinatorConfig(candidate_timeout_s=1.0),
        )
        snapshot = coordinator.publish_snapshot(anchor_monotonic_ns=source_ns + 2)
        assert snapshot is not None
        raw = ActionCandidate(
            observation_id=snapshot.observation_id,
            session_generation=999,
            policy_epoch=888,
            action_id=777,
            created_monotonic_ns=source_ns,
            target_monotonic_ns=source_ns + 1,
            valid_until_monotonic_ns=source_ns + 2,
            arm_qpos=np.zeros(7),
            chunk_id=44,
        )
        shared.inference_candidate_ring.write(encode_candidate(raw))

        chunk = coordinator.consume_candidate_chunk(now_monotonic_ns=source_ns + 3)

        assert chunk is not None
        normalized = chunk.steps[0]
        assert normalized.session_generation == int(shared.session_generation.value)
        assert normalized.policy_epoch == int(shared.policy_epoch.value)
        assert normalized.action_id != raw.action_id
        assert normalized.target_monotonic_ns > source_ns + 3
    finally:
        block.close()
        block.unlink()
        shared.close()


def test_camera_generation_change_forces_armed_epoch_and_rewarm() -> None:
    prefix = f"learned_reset_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    spec = ObservationSpec((ModalitySpec("arm_qpos", (7,), "float64"),))
    block = ObservationTensorBlock.create(f"{prefix}_tensor", spec)
    try:
        source_ns = time.monotonic_ns()
        arm_frame = new_frame(ARM_STATE_DTYPE)
        arm_frame["qpos"][0] = np.zeros(7)
        arm_frame["source_monotonic_ns"][0] = source_ns
        arm_frame["publish_monotonic_ns"][0] = source_ns + 1
        arm_frame["state_valid"][0] = 1
        shared.arm_state_ring.write(arm_frame)
        hand_frame = new_frame(HAND_STATE_DTYPE)
        hand_frame["qpos"][0] = np.zeros(12)
        hand_frame["source_monotonic_ns"][0] = source_ns
        hand_frame["publish_monotonic_ns"][0] = source_ns + 1
        hand_frame["state_valid"][0] = 1
        shared.hand_state_ring.write(hand_frame)
        shared.inference_ready.set()
        shared.safety_state.value = 2
        coordinator = LearnedPolicyCoordinator(
            shared,
            InferenceConfig("unused.module:Backend", spec, ActionSpec(chunk_length=1)),
            block,
            _gate(),
        )
        coordinator.publish_snapshot(anchor_monotonic_ns=source_ns + 2)
        coordinator._publish_coordinated_hold = Mock()  # type: ignore[method-assign]
        coordinator.sources._builder.camera_generation = 1

        result = coordinator.tick(now_monotonic_ns=source_ns + 100_000_000)

        assert result is CoordinatorTick.REWARMING
        assert shared.safety_state.value == 1
        assert shared.policy_epoch.value == 2
        assert not shared.inference_ready.is_set()
        assert coordinator.rewarm_pending
        coordinator._publish_coordinated_hold.assert_called_once()  # type: ignore[attr-defined]
    finally:
        block.close()
        block.unlink()
        shared.close()


def test_camera_snapshot_never_mixes_payload_generations() -> None:
    prefix = f"camera_generation_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    spec = ObservationSpec(
        (
            ModalitySpec("camera_rgb", (2, 3, 3), "uint8"),
            ModalitySpec("camera_pointcloud", (4, 6), "float32"),
        )
    )
    try:
        source = SharedObservationSource(shared, spec)

        def write_frame(*, generation: int, value: int, pointcloud_valid: bool) -> None:
            receive_ns = time.monotonic_ns()
            rgb = np.full((2, 3, 3), value, dtype=np.uint8)
            depth = np.full((2, 3), value, dtype=np.uint16)
            pointcloud = np.full((4, 6), value, dtype=np.float32)
            header, _, _ = pack_camera_frame(
                rgb,
                depth,
                timestamp=float(generation),
                capture_monotonic_s=receive_ns / 1e9,
                frame_id=generation,
                pc_num_points=4 if pointcloud_valid else 0,
                source_monotonic_ns=receive_ns - 1_000_000,
                camera_generation=generation,
            )
            shared.camera_ring.write(header, rgb, depth, pointcloud)

        write_frame(generation=1, value=1, pointcloud_valid=True)
        write_frame(generation=2, value=2, pointcloud_valid=False)
        snapshot = source.build(anchor_monotonic_ns=time.monotonic_ns())

        assert snapshot.camera_generation == 2
        assert bool(snapshot.valid_history_mask["camera_rgb"][0])
        np.testing.assert_array_equal(snapshot.values["camera_rgb"][0], np.full((2, 3, 3), 2, dtype=np.uint8))
        assert not bool(snapshot.valid_history_mask["camera_pointcloud"][0])
        np.testing.assert_array_equal(snapshot.values["camera_pointcloud"][0], np.zeros((4, 6), dtype=np.float32))
    finally:
        shared.close()


def test_device_clock_mapping_detects_duplicate_gap_and_reset() -> None:
    mapper = DeviceClockMapper(reset_jump_ns=100_000_000)
    first = mapper.map(device_time_s=1.0, host_receive_ns=2_000_000_000, frame_number=10)
    duplicate = mapper.map(device_time_s=1.0, host_receive_ns=2_010_000_000, frame_number=10)
    gap = mapper.map(device_time_s=1.03, host_receive_ns=2_030_000_000, frame_number=13)
    reset = mapper.map(device_time_s=0.1, host_receive_ns=2_040_000_000, frame_number=1)

    assert first.source_monotonic_ns <= 2_000_000_000
    assert duplicate.duplicate
    assert gap.frame_gap == 2
    assert reset.clock_reset and reset.generation == 1


def test_base_policy_causal_reader_never_selects_a_post_deadline_frame() -> None:
    dtype = np.dtype([("source_monotonic_ns", "<u8"), ("value", "<f8")])

    def frame(source_ns: int, value: float) -> np.ndarray:
        result = np.zeros(1, dtype=dtype)
        result["source_monotonic_ns"][0] = source_ns
        result["value"][0] = value
        return result

    ring = SimpleNamespace(
        maxlen=4,
        get_last_k=Mock(
            return_value=[
                (frame(100, 1.0), 101, 1),
                (frame(200, 2.0), 201, 2),
                (frame(220, 9.0), 260, 4),
                (frame(300, 3.0), 301, 3),
            ]
        ),
    )

    selected = _read_causal_structured_frame(
        ring,
        source_field="source_monotonic_ns",
        anchor_monotonic_ns=250,
    )

    assert selected is not None
    assert selected[2] == 2
    assert float(selected[0]["value"][0]) == 2.0


def test_recording_provenance_recovers_exact_source_sequence_not_latest() -> None:
    dtype = np.dtype([("source_monotonic_ns", "<u8"), ("publish_monotonic_ns", "<u8")])

    def frame(source_ns: int, publish_ns: int) -> np.ndarray:
        result = np.zeros(1, dtype=dtype)
        result["source_monotonic_ns"][0] = source_ns
        result["publish_monotonic_ns"][0] = publish_ns
        return result

    selected = frame(100, 110)
    ring = SimpleNamespace(
        maxlen=4,
        get_last_k=Mock(
            return_value=[
                (selected.copy(), 110, 7),
                (frame(200, 210), 210, 8),
            ]
        ),
    )

    assert _matching_source_sequence(ring, selected) == 7
    assert _matching_source_sequence(ring, frame(50, 60)) == 0


def test_recording_provenance_uses_explicit_candidate_not_latest_global_commit() -> None:
    prefix = f"record_provenance_{uuid.uuid4().hex}"
    shared = SharedStorage.create(
        prefix=prefix,
        config=SharedStorageConfig(
            camera_rgb_shape=(2, 3, 3),
            camera_depth_shape=(2, 3),
            camera_pc_shape=(4, 6),
        ),
    )
    try:
        now_ns = time.monotonic_ns()
        source_ns = now_ns - 3_000_000
        publish_ns = now_ns - 2_000_000
        arm_frame = new_frame(ARM_STATE_DTYPE)
        arm_frame["source_monotonic_ns"][0] = source_ns
        arm_frame["publish_monotonic_ns"][0] = publish_ns
        arm_frame["state_valid"][0] = 1
        shared.arm_state_ring.write(arm_frame)
        hand_frame = new_frame(HAND_STATE_DTYPE)
        hand_frame["source_monotonic_ns"][0] = source_ns
        hand_frame["publish_monotonic_ns"][0] = publish_ns
        hand_frame["state_valid"][0] = 1
        shared.hand_state_ring.write(hand_frame)

        candidate = ActionCandidate(
            observation_id=44,
            session_generation=int(shared.session_generation.value),
            policy_epoch=int(shared.policy_epoch.value),
            action_id=7,
            created_monotonic_ns=now_ns - 1_000_000,
            target_monotonic_ns=now_ns + 100_000_000,
            valid_until_monotonic_ns=now_ns + 200_000_000,
            arm_qpos=np.zeros(7),
            hand_qpos=np.zeros(12),
            chunk_id=7,
        )
        shared.arm_ack_ring.write(make_ack(make_command_frame(candidate, actuator="arm"), AckStatus.PREPARED))
        shared.hand_ack_ring.write(make_ack(make_command_frame(candidate, actuator="hand"), AckStatus.PREPARED))
        unrelated = ActionCandidate(
            observation_id=99,
            session_generation=int(shared.session_generation.value),
            policy_epoch=int(shared.policy_epoch.value),
            action_id=99,
            created_monotonic_ns=now_ns,
            target_monotonic_ns=now_ns + 300_000_000,
            valid_until_monotonic_ns=now_ns + 400_000_000,
            arm_qpos=np.zeros(7),
            chunk_id=99,
        )
        SafeCommandPublisher(shared).commit(unrelated)
        vr_frame = {
            "ring_sequence": 44,
            "local_recv_ns": source_ns,
            "publish_monotonic_ns": publish_ns,
        }

        signals = _recording_provenance(
            shared,
            arm_frame,
            hand_frame,
            None,
            vr_frame,
            None,
            anchor_monotonic_ns=time.monotonic_ns(),
            action_candidate=candidate,
        )
        assert signals["observation_id"] == 44
        assert signals["action_id"] == 7
        assert signals["action_chunk_id"] == 7
        assert signals["action_queued"] is True
        assert signals["action_committed"] is True

        held = _recording_provenance(
            shared,
            arm_frame,
            hand_frame,
            None,
            vr_frame,
            None,
            anchor_monotonic_ns=time.monotonic_ns(),
        )
        assert held["observation_id"] == 44
        assert held["action_id"] == 0
        assert held["action_queued"] is False
        assert held["action_committed"] is False
    finally:
        shared.close()


@pytest.mark.parametrize(
    ("estop", "fault", "exitcode", "heartbeat_age_s", "quit_requested", "expected"),
    [
        (True, True, 1, 10.0, True, ExitReason.ESTOP),
        (False, True, 1, 10.0, True, ExitReason.STICKY_FAULT),
        (False, False, 1, 10.0, True, ExitReason.WORKER_DEATH),
        (False, False, None, 10.0, True, ExitReason.HEARTBEAT_TIMEOUT),
        (False, False, None, 0.1, True, ExitReason.EXPLICIT_QUIT),
        (False, False, None, 0.1, False, ExitReason.NONE),
    ],
)
def test_supervisor_exit_priority_is_safety_first(
    estop: bool,
    fault: bool,
    exitcode: int | None,
    heartbeat_age_s: float,
    quit_requested: bool,
    expected: ExitReason,
) -> None:
    shared = SimpleNamespace(
        estop_request=SimpleNamespace(value=estop),
        error_state=SimpleNamespace(value=fault),
        quit_requested=SimpleNamespace(value=quit_requested),
        is_running=SimpleNamespace(value=True),
    )
    process = SimpleNamespace(exitcode=exitcode)

    assert supervisor_exit_reason(shared, [process], {"worker": heartbeat_age_s}, {"worker": 1.0}) is expected


class _ShutdownProcess:
    def __init__(self, *, remains_alive: bool = False) -> None:
        self.name = "stubborn"
        self.exitcode: int | None = None
        self.remains_alive = remains_alive
        self.events: list[str] = []
        self._phase = "running"

    def join(self, timeout: float) -> None:
        assert timeout >= 0
        self.events.append("join")

    def is_alive(self) -> bool:
        return self.remains_alive or self.exitcode is None

    def terminate(self) -> None:
        self.events.append("terminate")
        self._phase = "terminated"

    def kill(self) -> None:
        self.events.append("kill")
        self._phase = "killed"
        if not self.remains_alive:
            self.exitcode = -9


def test_shutdown_escalates_to_kill_before_shared_cleanup() -> None:
    process = _ShutdownProcess()
    shared = SimpleNamespace(is_running=SimpleNamespace(value=True), close=Mock())

    report = shutdown_processes_verified(
        shared,
        [process],
        graceful_timeout_s=0.0,
        terminate_timeout_s=0.0,
        kill_timeout_s=0.0,
    )

    assert process.events == ["join", "terminate", "join", "kill", "join"]
    assert report.all_stopped and report.shared_closed
    assert report.exits[0].escalation == "kill"
    assert shared.is_running.value is False
    shared.close.assert_called_once_with()


def test_shutdown_keeps_shared_memory_when_exit_is_unconfirmed() -> None:
    process = _ShutdownProcess(remains_alive=True)
    shared = SimpleNamespace(is_running=SimpleNamespace(value=True), close=Mock())

    with pytest.raises(RuntimeError, match="could not be confirmed stopped"):
        shutdown_processes_verified(
            shared,
            [process],
            graceful_timeout_s=0.0,
            terminate_timeout_s=0.0,
            kill_timeout_s=0.0,
        )

    shared.close.assert_not_called()


def test_preflight_certificate_detects_tampering_and_binding_changes(tmp_path: Path) -> None:
    model = tmp_path / "robot.urdf"
    model.write_text("robot", encoding="utf-8")
    arm_actions = np.zeros((2, 7))
    hand_actions = np.zeros((2, 12))
    workspace = np.array([[0.2, 0.7], [-0.5, 0.5], [0.05, 0.5]])
    certificate = create_preflight_certificate(
        source_episode=str(tmp_path / "episode"),
        arm_actions=arm_actions,
        hand_actions=hand_actions,
        collision_model_paths=[model],
        workspace_bounds_m=workspace,
        resolved_config_sha256="a" * 64,
        transition_check=lambda *_: True,
        workspace_check=lambda *_: True,
        table_check=lambda *_: True,
    )
    path = certificate.write(tmp_path / "certificate.json")
    loaded = PreflightCertificate.read(path)
    verify_preflight_binding(
        loaded,
        source_episode=str(tmp_path / "episode"),
        arm_actions=arm_actions,
        hand_actions=hand_actions,
        collision_model_paths=[model],
        workspace_bounds_m=workspace,
        resolved_config_sha256="a" * 64,
    )

    changed = arm_actions.copy()
    changed[1, 0] = 0.01
    with pytest.raises(ValueError, match="trajectory"):
        verify_preflight_binding(
            loaded,
            source_episode=str(tmp_path / "episode"),
            arm_actions=changed,
            hand_actions=hand_actions,
            collision_model_paths=[model],
            workspace_bounds_m=workspace,
            resolved_config_sha256="a" * 64,
        )
