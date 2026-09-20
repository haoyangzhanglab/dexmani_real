"""Bounded ordered command-FIFO mechanics shared by the publisher and workers.

The coupled command ring is consumed as a lossless ordered FIFO: one committed
record per sequence, each attached worker consumer advances its own watermark
in commit order, and the publisher commits a new record only while the shared
capacity invariant holds:

    L - min(max(c_i, b - 1)) < C

with ``L`` the last committed sequence, ``C`` the ring capacity, ``c_i`` each
attached consumer's consumed sequence, and ``b`` the first sequence of the
current run generation. FULL is non-blocking recoverable backpressure for the
producer; it never drops queued records and never faults by itself.

Consumption watermarks live in cross-process values owned by RuntimeChannels.
A consumer that never attached (``-1``) does not enter the capacity watermark,
so an absent actuator cannot stall the stream. A late old-generation watermark
update is clamped by ``b - 1`` and cannot hold back a new epoch.

This module keeps the existing shared-memory platform assumptions of
``ipc/ring.py``; it adds no new locking protocol and no scheduling framework.
"""

from __future__ import annotations

from typing import Any

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Watermark value meaning "this consumer never attached"; excluded from the
# capacity minimum. Signed because sequence watermarks are never negative.
CONSUMER_NOT_ATTACHED = -1


class CommandStreamCorruption(RuntimeError):
    """A committed current-generation FIFO sequence was lost.

    This is a transport contract violation, not an EMPTY wait: with the
    capacity invariant enforced at commit, a resident committed sequence can
    only become unreadable through a real IPC fault. Workers fail fast on it.
    """


def command_stream_capacity_locked(shared: Any) -> tuple[bool, int, bool]:
    """Capacity verdict for one commit; the caller must own ``motion_lock``.

    Returns ``(has_capacity, backlog_depth, any_consumer_attached)`` where
    ``backlog_depth = L - min(max(c_i, b - 1))`` over attached consumers.
    With no attached consumer the stream fails closed (no capacity).
    """
    ring = shared.coupled_cmd_ring
    latest = int(ring.latest_sequence)
    base = int(shared.run_generation_base_sequence.value)
    watermarks = []
    for watermark in (
        shared.arm_cmd_consumed_sequence,
        shared.hand_cmd_consumed_sequence,
    ):
        value = int(watermark.value)
        if value != CONSUMER_NOT_ATTACHED:
            watermarks.append(max(value, base))
    if not watermarks:
        return False, max(0, latest - base), False
    backlog = latest - min(watermarks)
    return backlog < int(ring.maxlen), backlog, True


class CommandStreamConsumer:
    """One worker's ordered cursor over the coupled command FIFO.

    EMPTY is a wait, never a fault. A committed sequence of the current epoch
    that cannot be read is rechecked inside the short motion lock and only
    then reported as :class:`CommandStreamCorruption`; the cursor never jumps
    to the latest record. A run-generation change resyncs the cursor to the
    new epoch's first sequence, batch-skipping the invalidated backlog.
    """

    def __init__(self, shared: Any, watermark: Any) -> None:
        self._shared = shared
        self._watermark = watermark
        with shared.motion_lock:
            base = int(shared.coupled_cmd_ring.latest_sequence)
            if int(watermark.value) < base:
                watermark.value = base
            self._generation = int(shared.run_generation.value)
        self._next_sequence = base + 1

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def resync_if_stale_generation(self, permit_generation: int) -> bool:
        """Skip the invalidated backlog after an epoch change.

        Returns ``True`` when the cursor was resynced, so callers can rebuild
        epoch-local command references (e.g. the hand command-space anchor).
        """
        if permit_generation == self._generation:
            return False
        shared = self._shared
        with shared.motion_lock:
            base = int(shared.run_generation_base_sequence.value)
            self._generation = int(shared.run_generation.value)
            self._next_sequence = base + 1
            if int(self._watermark.value) < base:
                self._watermark.value = base
        logger.debug(
            "command stream: consumer resynced to generation=%d sequence=%d",
            self._generation,
            self._next_sequence,
        )
        return True

    def next_record(self) -> tuple[Any, int] | None:
        """Return ``(record, sequence)`` for the next committed record.

        ``None`` means EMPTY (nothing committed at the cursor yet) or that an
        epoch change invalidated the target sequence mid-read; the next tick's
        :meth:`resync_if_stale_generation` repositions the cursor. Raises
        :class:`CommandStreamCorruption` only for a genuinely lost committed
        current-epoch sequence.
        """
        shared = self._shared
        ring = shared.coupled_cmd_ring
        sequence = self._next_sequence
        if sequence > int(ring.latest_sequence):
            return None
        result = ring.read_sequence(sequence)
        if result is None or int(result[2]) != sequence:
            # Recheck inside the short lock before classifying the miss.
            with shared.motion_lock:
                base = int(shared.run_generation_base_sequence.value)
                latest = int(ring.latest_sequence)
                result = ring.read_sequence(sequence)
            if result is not None and int(result[2]) == sequence:
                pass  # recovered on the locked recheck
            elif sequence <= base:
                # An epoch advance invalidated this record during the read.
                return None
            elif sequence <= latest:
                raise CommandStreamCorruption(
                    f"committed command sequence {sequence} was lost "
                    f"(latest={latest}, generation={self._generation})"
                )
            else:
                return None
        with shared.motion_lock:
            if int(shared.run_generation.value) != self._generation:
                return None
            if int(result[0]["run_generation"][0]) != self._generation:
                raise CommandStreamCorruption(
                    f"command sequence {sequence} has an impossible generation "
                    f"in stable epoch {self._generation}"
                )
        return result[0], sequence

    def advance(self) -> bool:
        """Release this worker's record only while its epoch is still current.

        SDK acceptance or an absent actuator may advance; intermediate hand
        setpoints and CRC-unconfirmed sends may not. A late old SDK return
        leaves both the cursor and shared watermark unchanged until resync.
        The worker is the sole owner of this cursor, including across SDK IO.
        """
        with self._shared.motion_lock:
            if int(self._shared.run_generation.value) != self._generation:
                return False
            sequence = self._next_sequence
            self._next_sequence = sequence + 1
            if int(self._watermark.value) < sequence:
                self._watermark.value = sequence
            return True
