"""Offline tests for the bounded ordered command FIFO transport (task T2).

Real shared-memory ring, real cross-process ``Value`` watermarks, and the
real publication/safety functions; fake clocks are avoided in favor of the
transport's own sequence identities. Coverage maps to the acceptance matrix:

* V04 single/dual actuator, absent consumer, ordered commit→SDK delivery;
* V05 FULL is recoverable backpressure committing the IDENTICAL candidate
  (no pop, no rebuild, no duplicate) once capacity frees;
* V06 a lost committed current-epoch sequence is corruption, EMPTY is a wait;
* V07 hand intermediate slew / CRC / exact-endpoint cursor distinction;
* V08 generation cancel batch-invalidates the backlog, stale-generation ACKs
  never satisfy an acceptance wait, and ordered same-generation watermarks
  prove predecessor acceptance without action-ID supersession.

Never opens hardware or an SDK: the arm/hand workers are exercised through
their consumption functions with fake SDK objects.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    publish_command,
    wait_command_accepted,
)
from dexmani_real.ipc.command_stream import (
    CommandStreamConsumer,
    CommandStreamCorruption,
)
from dexmani_real.ipc.ring import SeqlockSlot, SharedMemoryRingBuffer
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    COUPLED_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    invalidate_coupled_commands,
)

_GEN = 7
_COUNTER = 0


def _next_ring_name(prefix: str) -> str:
    global _COUNTER
    _COUNTER += 1
    return f"t2_{prefix}_{os.getpid()}_{_COUNTER}_{time.monotonic_ns()}"


class _StateRing:
    """Fake latest-only feedback ring returning one synthetic record."""

    def __init__(self, record: np.ndarray | None = None) -> None:
        self._record = record

    def set(self, record: np.ndarray | None) -> None:
        self._record = record

    def read_latest(self):
        if self._record is None:
            return None
        return (self._record, time.monotonic_ns(), 1)


class _FakeShared:
    """Transport-level RuntimeChannels stand-in.

    Real shared-memory command ring and real cross-process Values; no
    processes are spawned. Safety state defaults to RUNNING with generation
    ``_GEN`` so publication is permitted.
    """

    def __init__(self, maxlen: int = 4, *, generation: int = _GEN) -> None:
        ctx = mp.get_context("spawn")
        self.coupled_cmd_ring = SharedMemoryRingBuffer(
            _next_ring_name("cmd"),
            COUPLED_COMMAND_DTYPE,
            maxlen=maxlen,
            create=True,
        )
        self.motion_lock = ctx.RLock()
        self.run_generation = ctx.Value("Q", generation)
        self.run_generation_base_sequence = ctx.Value("Q", 0)
        self.arm_cmd_consumed_sequence = ctx.Value("q", -1)
        self.hand_cmd_consumed_sequence = ctx.Value("q", -1)
        self.arm_command_seq = ctx.Value("Q", 0)
        self.is_running = ctx.Value("b", True)
        self.error_state = ctx.Value("b", False)
        self.estop_request = ctx.Value("b", False)
        self.safety_state = ctx.Value("i", int(SafetyState.RUNNING))
        self.run_started_monotonic_ns = ctx.Value("Q", 0)
        self.stop_request = ctx.Value("b", 0)
        self.arm_state_ring = _StateRing()
        self.hand_state_ring = _StateRing()

    def close(self) -> None:
        self.coupled_cmd_ring.close()
        try:
            self.coupled_cmd_ring.unlink()
        except FileNotFoundError:
            pass


def _arm_qpos(value: float) -> np.ndarray:
    return np.full(7, value, dtype=np.float64)


def _hand_qpos(value: float) -> np.ndarray:
    return np.full(12, value, dtype=np.float64)


def _publish(
    shared: _FakeShared,
    action_id: int,
    *,
    arm: float | None = None,
    hand: float | None = None,
    is_hold: bool = False,
    generation: int = _GEN,
):
    candidate = ActionCandidate(
        run_generation=generation,
        action_id=action_id,
        arm_qpos=None if arm is None else _arm_qpos(arm),
        hand_qpos=None if hand is None else _hand_qpos(hand),
        is_hold=is_hold,
    )
    return candidate, publish_command(
        shared, candidate, required_safety_state=SafetyState.RUNNING
    )


class _TransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.shared = _FakeShared(maxlen=4)
        self.addCleanup(self.shared.close)


class OrderedDeliveryTest(_TransportTest):
    def test_single_consumer_receives_records_in_commit_order(self):
        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        sent = []
        for action_id in (11, 12, 13):
            _candidate, result = _publish(self.shared, action_id, arm=0.1 * action_id)
            self.assertTrue(result.published, result.reason)
            sent.append((action_id, result.command.sequence))
        # Sequences are commit-produced and strictly ordered.
        self.assertEqual([seq for _aid, seq in sent], [1, 2, 3])

        received = []
        for expected_action_id, expected_seq in sent:
            record = consumer.next_record()
            self.assertIsNotNone(record)
            command, sequence = record
            self.assertEqual(sequence, expected_seq)
            self.assertEqual(int(command["action_id"][0]), expected_action_id)
            self.assertEqual(
                float(command["arm_qpos"][0][0]), 0.1 * expected_action_id
            )
            self.assertEqual(int(command["run_generation"][0]), _GEN)
            consumer.advance()
            received.append(sequence)
        self.assertEqual(received, [1, 2, 3])
        # EMPTY after the backlog: a wait, not a fault.
        self.assertIsNone(consumer.next_record())
        self.assertEqual(int(self.shared.arm_cmd_consumed_sequence.value), 3)

    def test_absent_actuator_record_only_advances_that_consumer(self):
        arm = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        hand = CommandStreamConsumer(
            self.shared, self.shared.hand_cmd_consumed_sequence
        )
        _candidate, result = _publish(self.shared, 21, arm=0.5)  # hand absent
        self.assertTrue(result.published, result.reason)

        arm_record = arm.next_record()
        self.assertIsNotNone(arm_record)
        self.assertEqual(int(arm_record[0]["arm_present"][0]), 1)
        self.assertEqual(int(arm_record[0]["hand_present"][0]), 0)
        arm.advance()

        hand_record = hand.next_record()
        self.assertIsNotNone(hand_record)
        # The hand consumer skips the record without any SDK semantics; its
        # watermark still advances so capacity is released.
        self.assertEqual(int(hand_record[0]["hand_present"][0]), 0)
        hand.advance()
        self.assertEqual(int(self.shared.hand_cmd_consumed_sequence.value), 1)


class FullBackpressureTest(_TransportTest):
    def test_full_is_recoverable_and_retries_the_identical_candidate(self):
        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        for action_id in (31, 32, 33, 34):
            _candidate, result = _publish(self.shared, action_id, arm=0.01 * action_id)
            self.assertTrue(result.published, result.reason)

        blocked_candidate, blocked = _publish(self.shared, 35, arm=0.35)
        self.assertFalse(blocked.published)
        self.assertEqual(blocked.reason, "command fifo full")
        self.assertEqual(blocked.fifo_depth, 4)

        # Free exactly one slot: the SAME candidate object commits unchanged.
        record = consumer.next_record()
        self.assertIsNotNone(record)
        self.assertEqual(int(record[0]["action_id"][0]), 31)
        consumer.advance()
        retry = publish_command(
            self.shared, blocked_candidate, required_safety_state=SafetyState.RUNNING
        )
        self.assertTrue(retry.published, retry.reason)
        self.assertEqual(retry.command.sequence, 5)

        # Drain everything: exactly 31..35 in order, no loss, no duplication.
        seen = [31]
        while True:
            record = consumer.next_record()
            if record is None:
                break
            command, _sequence = record
            seen.append(int(command["action_id"][0]))
            consumer.advance()
        self.assertEqual(seen, [31, 32, 33, 34, 35])

    def test_slow_hand_governs_dual_consumer_capacity(self):
        arm = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        _hand = CommandStreamConsumer(
            self.shared, self.shared.hand_cmd_consumed_sequence
        )  # attached but never advances: the slow consumer
        for action_id in (41, 42, 43, 44):
            _candidate, result = _publish(
                self.shared, action_id, arm=0.1, hand=0.2
            )
            self.assertTrue(result.published, result.reason)
            record = arm.next_record()
            self.assertIsNotNone(record)
            arm.advance()  # arm keeps up fully
        # Arm consumed everything, yet the lagging hand watermark blocks.
        _candidate, blocked = _publish(self.shared, 45, arm=0.1, hand=0.2)
        self.assertFalse(blocked.published)
        self.assertEqual(blocked.reason, "command fifo full")

    def test_unattached_consumer_does_not_enter_watermark(self):
        arm = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        # hand watermark stays -1 (never attached): excluded from capacity.
        for round_index in range(3):
            for action_id in range(51 + 10 * round_index, 55 + 10 * round_index):
                _candidate, result = _publish(self.shared, action_id, arm=0.1)
                self.assertTrue(result.published, result.reason)
                record = arm.next_record()
                self.assertIsNotNone(record)
                arm.advance()
        self.assertEqual(int(self.shared.hand_cmd_consumed_sequence.value), -1)

    def test_no_attached_consumer_fails_closed(self):
        _candidate, result = _publish(self.shared, 61, arm=0.1)
        self.assertFalse(result.published)
        self.assertEqual(result.reason, "no attached command consumer")


class GenerationEpochTest(_TransportTest):
    def test_generation_cancel_batch_skips_invalidated_backlog(self):
        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        for action_id in (71, 72):
            _candidate, result = _publish(self.shared, action_id, arm=0.1)
            self.assertTrue(result.published, result.reason)

        new_generation = invalidate_coupled_commands(self.shared)
        self.assertEqual(new_generation, _GEN + 1)
        self.assertEqual(
            int(self.shared.run_generation_base_sequence.value), 2
        )  # new epoch starts at L+1 = 3

        self.assertTrue(consumer.resync_if_stale_generation(new_generation))
        # The invalidated backlog is never delivered after the resync.
        self.assertIsNone(consumer.next_record())

        _candidate, result = _publish(
            self.shared, 73, arm=0.3, generation=new_generation
        )
        self.assertTrue(result.published, result.reason)
        self.assertEqual(result.command.sequence, 3)
        record = consumer.next_record()
        self.assertIsNotNone(record)
        command, sequence = record
        self.assertEqual(sequence, 3)
        self.assertEqual(int(command["action_id"][0]), 73)
        self.assertEqual(int(command["run_generation"][0]), new_generation)

    def test_stale_watermark_cannot_hold_back_new_epoch(self):
        _consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        for action_id in (81, 82, 83, 84):
            _candidate, result = _publish(self.shared, action_id, arm=0.1)
            self.assertTrue(result.published, result.reason)
        # Watermark stuck at 0 with a full backlog of the old epoch.
        new_generation = invalidate_coupled_commands(self.shared)
        # The clamp max(c, b-1) frees the whole capacity for the new epoch.
        for action_id in (85, 86, 87, 88):
            _candidate, result = _publish(
                self.shared, action_id, arm=0.1, generation=new_generation
            )
            self.assertTrue(result.published, result.reason)
        _candidate, blocked = _publish(
            self.shared, 89, arm=0.1, generation=new_generation
        )
        self.assertFalse(blocked.published)
        self.assertEqual(blocked.reason, "command fifo full")


class CorruptionTest(_TransportTest):
    def test_empty_is_wait_not_fault(self):
        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        self.assertIsNone(consumer.next_record())
        self.assertEqual(int(self.shared.arm_cmd_consumed_sequence.value), 0)

    def test_lost_committed_sequence_is_transport_corruption(self):
        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        _candidate, result = _publish(self.shared, 91, arm=0.1)
        self.assertTrue(result.published, result.reason)
        # Destroy the resident slot's seqlock marker behind the consumer's
        # back: a committed current-epoch sequence becomes unreadable.
        ring = self.shared.coupled_cmd_ring
        slot_index = result.command.sequence % ring.maxlen
        SeqlockSlot(
            ring._shm.buf,
            ring._HEADER_SIZE + slot_index * ring._slot_size,
        ).begin_write(12345, 0)
        with self.assertRaises(CommandStreamCorruption):
            consumer.next_record()


def _arm_feedback_record(
    *, action_id: int, generation: int, sequence: int
) -> np.ndarray:
    record = np.zeros(1, dtype=ARM_STATE_DTYPE)
    record["connected"] = 1
    record["state_valid"] = 1
    record["error_code"] = 0
    now_ns = time.monotonic_ns()
    record["source_monotonic_ns"] = now_ns
    record["qpos"] = 0.1
    record["last_cmd_seq"] = action_id
    record["last_cmd_generation"] = generation
    record["last_cmd_accepted_sequence"] = sequence
    record["last_cmd_accepted_monotonic_ns"] = now_ns if action_id else 0
    return record


def _hand_feedback_record(
    *, action_id: int, generation: int, sequence: int
) -> np.ndarray:
    record = np.zeros(1, dtype=HAND_STATE_DTYPE)
    record["connected"] = 1
    record["state_valid"] = 1
    now_ns = time.monotonic_ns()
    record["source_monotonic_ns"] = now_ns
    record["qpos"] = 0.1
    record["accepted_target_action_id"] = action_id
    record["accepted_target_generation"] = generation
    record["accepted_target_sequence"] = sequence
    record["accepted_target_monotonic_ns"] = now_ns if action_id else 0
    return record


class WaitCommandAcceptedTest(_TransportTest):
    """Ordered same-generation acceptance; no action-ID supersession."""

    def _wait(self, *, command, action_id, wait_for_arm=True, wait_for_hand=False):
        return wait_command_accepted(
            self.shared,
            command=command,
            action_id=action_id,
            wait_for_arm=wait_for_arm,
            wait_for_hand=wait_for_hand,
            timeout_s=0.2,
            arm_feedback_max_age_s=1.0,
            hand_feedback_max_age_s=1.0,
        )

    def test_exact_acceptance_same_generation(self):
        from dexmani_real.runtime.safety import CommittedCommand

        self.shared.arm_state_ring.set(
            _arm_feedback_record(action_id=42, generation=_GEN, sequence=3)
        )
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3), action_id=42
        )
        self.assertTrue(result.accepted, result.reason)

    def test_larger_same_generation_watermark_proves_predecessor(self):
        from dexmani_real.runtime.safety import CommittedCommand

        # A later arm-targeted record was accepted (id 50 > 42, sequence 5 >= 3):
        # ordered consumption proves our command was SDK-accepted first.
        self.shared.arm_state_ring.set(
            _arm_feedback_record(action_id=50, generation=_GEN, sequence=5)
        )
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3), action_id=42
        )
        self.assertTrue(result.accepted, result.reason)

    def test_stale_generation_ack_never_satisfies_wait(self):
        from dexmani_real.runtime.safety import CommittedCommand

        # A large watermark from the PREVIOUS generation is not evidence.
        self.shared.arm_state_ring.set(
            _arm_feedback_record(action_id=99, generation=_GEN - 1, sequence=99)
        )
        generation_before = int(self.shared.run_generation.value)
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3), action_id=42
        )
        self.assertFalse(result.accepted)
        self.assertIn("not accepted", result.reason)
        # The timed-out wait cancelled only its own still-current epoch.
        self.assertEqual(
            int(self.shared.run_generation.value), generation_before + 1
        )

    def test_earlier_sequence_watermark_keeps_waiting(self):
        from dexmani_real.runtime.safety import CommittedCommand

        self.shared.arm_state_ring.set(
            _arm_feedback_record(action_id=40, generation=_GEN, sequence=2)
        )
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3), action_id=42
        )
        self.assertFalse(result.accepted)

    def test_hand_exact_endpoint_acceptance(self):
        from dexmani_real.runtime.safety import CommittedCommand

        self.shared.hand_state_ring.set(
            _hand_feedback_record(action_id=42, generation=_GEN, sequence=3)
        )
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3),
            action_id=42,
            wait_for_arm=False,
            wait_for_hand=True,
        )
        self.assertTrue(result.accepted, result.reason)

    def test_runtime_stop_aborts_wait(self):
        from dexmani_real.runtime.safety import CommittedCommand

        self.shared.arm_state_ring.set(
            _arm_feedback_record(action_id=40, generation=_GEN, sequence=2)
        )
        self.shared.safety_state.value = int(SafetyState.DISARMED)
        result = self._wait(
            command=CommittedCommand(run_generation=_GEN, sequence=3), action_id=42
        )
        self.assertFalse(result.accepted)
        self.assertIn("safety state", result.reason)


class _FakeArmSdk:
    def __init__(self) -> None:
        self.servo_calls: list[np.ndarray] = []
        self.error_code = 0

    def servo(self, target: np.ndarray) -> int:
        self.servo_calls.append(np.asarray(target, dtype=np.float64).copy())
        return 0

    def read_live_error_code(self) -> int:
        return 0


class ArmWorkerConsumptionTest(_TransportTest):
    """V04 worker side: ordered commit→SDK delivery through the real consumer."""

    def _make_state(self, consumer):
        from dexmani_real.config.defaults import ArmParams
        from dexmani_real.ipc.channels import new_frame
        from dexmani_real.robot.arm_worker import _CmdState, _LoopState

        arm = _FakeArmSdk()
        st = _LoopState(
            cfg=ArmParams(),
            arm=arm,
            frame=new_frame(ARM_STATE_DTYPE),
            last_target=np.zeros(7, dtype=np.float64),
            last_measured_qpos=np.zeros(7, dtype=np.float64),
            last_command_generation=_GEN,
            consumer=consumer,
        )
        return st, arm

    def test_records_reach_sdk_in_commit_order(self):
        from dexmani_real.robot.arm_worker import _consume_one_arm_command

        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        st, arm = self._make_state(consumer)
        targets = []
        for action_id, value in ((101, 0.05), (102, 0.09)):
            _candidate, result = _publish(self.shared, action_id, arm=value)
            self.assertTrue(result.published, result.reason)
            targets.append(value)
            _consume_one_arm_command(st, self.shared, _GEN)
        self.assertEqual(len(arm.servo_calls), 2)
        self.assertAlmostEqual(float(arm.servo_calls[0][0]), targets[0])
        self.assertAlmostEqual(float(arm.servo_calls[1][0]), targets[1])
        # The acceptance watermark advanced only after SDK acceptance.
        self.assertEqual(int(self.shared.arm_cmd_consumed_sequence.value), 2)
        self.assertEqual(st.last_cmd.seq, 102)
        self.assertEqual(st.last_cmd.generation, _GEN)
        self.assertEqual(st.last_cmd.accepted_sequence, 2)

    def test_absent_record_never_touches_sdk(self):
        from dexmani_real.robot.arm_worker import _consume_one_arm_command

        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        st, arm = self._make_state(consumer)
        _candidate, result = _publish(self.shared, 103, hand=0.2)
        self.assertTrue(result.published, result.reason)
        _consume_one_arm_command(st, self.shared, _GEN)
        self.assertEqual(arm.servo_calls, [])
        self.assertEqual(int(self.shared.arm_cmd_consumed_sequence.value), 1)

    def test_jump_rejection_pauses_epoch_without_sdk_send(self):
        from dexmani_real.robot.arm_worker import _consume_one_arm_command

        consumer = CommandStreamConsumer(
            self.shared, self.shared.arm_cmd_consumed_sequence
        )
        st, arm = self._make_state(consumer)
        # Inside joint limits but beyond max_servo_command_jump_rad (0.349)
        # from the zero reference.
        _candidate, result = _publish(self.shared, 104, arm=0.5)
        self.assertTrue(result.published, result.reason)
        generation_before = int(self.shared.run_generation.value)
        _consume_one_arm_command(st, self.shared, _GEN)
        self.assertEqual(arm.servo_calls, [])
        # The rejection revoked motion and batch-invalidated the epoch.
        self.assertEqual(
            int(self.shared.run_generation.value), generation_before + 1
        )
        self.assertEqual(
            int(self.shared.safety_state.value), int(SafetyState.ARMED)
        )


class _FakeSendStatus:
    ACCEPTED = object()
    REJECTED = object()
    CRC_UNCONFIRMED = object()


class _FakeHandSdk:
    def __init__(self, statuses) -> None:
        self._statuses = list(statuses)
        self.sent: list[np.ndarray] = []

    def send_action(self, setpoint: np.ndarray):
        self.sent.append(np.asarray(setpoint, dtype=np.float64).copy())
        return self._statuses.pop(0)


class HandWorkerConsumptionTest(_TransportTest):
    """V07: intermediate slew / CRC / exact-endpoint cursor semantics."""

    def _setup(self, statuses, *, max_delta=0.1):
        from dexmani_real.robot.hand_worker import (
            _HandCommandAck,
            _consume_one_hand_command,
        )

        consumer = CommandStreamConsumer(
            self.shared, self.shared.hand_cmd_consumed_sequence
        )
        ack = _HandCommandAck(last_sdk_accepted_qpos=np.zeros(12))
        hand = _FakeHandSdk(statuses)
        permit = SimpleNamespace(
            run_generation=_GEN, state=SafetyState.RUNNING, allows_motion=True
        )

        def consume(measured=None):
            _consume_one_hand_command(
                self.shared,
                hand,
                _FakeSendStatus,
                consumer,
                permit,
                np.zeros(12) if measured is None else measured,
                ack,
                mechanical_lower=np.full(12, -1.0),
                mechanical_upper=np.full(12, 2.0),
                max_delta_rad_per_tick=max_delta,
            )

        return consumer, ack, hand, consume

    def test_intermediate_setpoints_do_not_advance_endpoint_cursor(self):
        statuses = [_FakeSendStatus.ACCEPTED] * 10
        _consumer, ack, hand, consume = self._setup(statuses)
        target_value = 0.25  # 2.5 slew steps of 0.1 from zero
        _candidate, result = _publish(self.shared, 111, hand=target_value)
        self.assertTrue(result.published, result.reason)

        consume()  # bounded = 0.1: intermediate
        self.assertEqual(len(hand.sent), 1)
        self.assertAlmostEqual(float(hand.sent[0][0]), 0.1)
        self.assertEqual(ack.accepted_target_action_id, 0)
        self.assertEqual(
            int(self.shared.hand_cmd_consumed_sequence.value), 0
        )  # cursor holds

        consume()  # bounded = 0.2: intermediate
        self.assertAlmostEqual(float(hand.sent[1][0]), 0.2)
        self.assertEqual(ack.accepted_target_action_id, 0)

        consume()  # bounded == target: exact endpoint ACCEPTED
        self.assertAlmostEqual(float(hand.sent[2][0]), target_value)
        self.assertEqual(ack.accepted_target_action_id, 111)
        self.assertEqual(ack.accepted_target_generation, _GEN)
        self.assertEqual(ack.accepted_target_sequence, 1)
        self.assertEqual(
            int(self.shared.hand_cmd_consumed_sequence.value), 1
        )  # released

    def test_crc_unconfirmed_leaves_everything_unacknowledged(self):
        _consumer, ack, hand, consume = self._setup(
            [_FakeSendStatus.CRC_UNCONFIRMED, _FakeSendStatus.ACCEPTED]
        )
        _candidate, result = _publish(self.shared, 112, hand=0.05)
        self.assertTrue(result.published, result.reason)

        consume()  # CRC: no reference update, no ACK, no cursor move
        self.assertEqual(len(hand.sent), 1)
        self.assertEqual(ack.accepted_target_action_id, 0)
        self.assertEqual(ack.last_sdk_setpoint_accepted_monotonic_ns, 0)
        self.assertTrue(np.all(ack.last_sdk_accepted_qpos == 0.0))
        self.assertEqual(int(self.shared.hand_cmd_consumed_sequence.value), 0)

        consume()  # retry of the SAME record is accepted exactly
        self.assertEqual(len(hand.sent), 2)
        self.assertEqual(ack.accepted_target_action_id, 112)
        self.assertEqual(int(self.shared.hand_cmd_consumed_sequence.value), 1)

    def test_sdk_rejection_raises_worker_fault(self):
        from dexmani_real.robot.hand_worker import _HandWorkerFault

        _consumer, _ack, _hand, consume = self._setup([_FakeSendStatus.REJECTED])
        _candidate, result = _publish(self.shared, 113, hand=0.05)
        self.assertTrue(result.published, result.reason)
        with self.assertRaises(_HandWorkerFault):
            consume()

    def test_absent_hand_record_advances_without_send(self):
        _consumer, _ack, hand, consume = self._setup([])
        _candidate, result = _publish(self.shared, 114, arm=0.1)
        self.assertTrue(result.published, result.reason)
        consume()
        self.assertEqual(hand.sent, [])
        self.assertEqual(int(self.shared.hand_cmd_consumed_sequence.value), 1)


def _spawn_consume_one(shared) -> None:
    """Child entry: attach as the arm consumer and commit-consume one record.

    Validates the real spawn transport: RuntimeChannels-style pickling of the
    new watermark values, cross-process ring reads, and cross-process
    watermark visibility for the publisher's capacity check.
    """
    consumer = CommandStreamConsumer(shared, shared.arm_cmd_consumed_sequence)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        record = consumer.next_record()
        if record is not None:
            command, _sequence = record
            assert int(command["action_id"][0]) == 12345
            assert float(command["arm_qpos"][0][0]) == 0.25
            consumer.advance()
            return
        time.sleep(0.01)
    raise TimeoutError("no command record reached the spawned consumer")


class SpawnTransportTest(unittest.TestCase):
    """One real spawn round-trip through the ordered command FIFO."""

    def test_record_crosses_process_boundary_and_watermark_returns(self):
        ctx = mp.get_context("spawn")
        shared = _FakeShared(maxlen=4)
        self.addCleanup(shared.close)
        process = ctx.Process(target=_spawn_consume_one, args=(shared,))
        process.start()
        try:
            # Publish only after the child attached its watermark, mirroring
            # the readiness-gated production order.
            deadline = time.monotonic() + 20.0
            while (
                int(shared.arm_cmd_consumed_sequence.value) < 0
                and time.monotonic() < deadline
            ):
                self.assertEqual(process.exitcode, None, "consumer child died")
                time.sleep(0.01)
            self.assertGreaterEqual(int(shared.arm_cmd_consumed_sequence.value), 0)
            _candidate, result = _publish(shared, 12345, arm=0.25)
            self.assertTrue(result.published, result.reason)
            process.join(timeout=30.0)
            self.assertEqual(process.exitcode, 0)
            # The child's watermark advance is visible to the publisher side.
            self.assertEqual(int(shared.arm_cmd_consumed_sequence.value), 1)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
