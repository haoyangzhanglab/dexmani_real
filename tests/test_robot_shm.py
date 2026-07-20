"""SHM foundation tests: robot_layouts / robot_ring / robot_rpc.

Pins the arm-hand-process-isolation SHM layer (plan §4):
  1. Layout roundtrip through SeqlockRingBuffer + schema-drift alarm
     (explicit itemsize constants, byte-for-byte vs plan §4.1-4.5/§4.10).
  2. is_fresh fake-clock boundaries (injected now_ns, ts_ns<=0 never fresh).
  3. Seqlock torn-read injection: a zeroed slot sequence + scribbled data must
     yield the last-good cached frame — never garbage — and never raise.
  4. Stale SHM cleanup on create (FileExistsError → unlink + recreate).
  5. RPC roundtrip (server thread echoes ok=1 + final_qpos; cmd_seq matched).
  6. RPC timeout with no server raises RpcTimeoutError promptly.
  7. ARM_TARGET is_hold / producer_id survive roundtrip.

All tests use unique per-test SHM names (test suffix + PID) and unlink in
finally; total runtime <5 s.
"""

from __future__ import annotations

import os
import threading
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from dexmani_real.shm.robot_layouts import (
    ARM_CMD_CLEAR_ERROR,
    ARM_CMD_DTYPE,
    ARM_CMD_RESET_BLOCKING,
    ARM_CMD_RESULT_DTYPE,
    ARM_STATE_DTYPE,
    ARM_TARGET_DTYPE,
    HAND_CMD_DTYPE,
    HAND_STATE_DTYPE,
    POLICY_CHUNK_DTYPE,
    PRODUCER_POLICY,
    PRODUCER_TELEOP,
    new_frame,
)
from dexmani_real.shm.robot_ring import SeqlockRingBuffer, is_fresh
from dexmani_real.shm.robot_rpc import RpcClient, RpcServer, RpcTimeoutError

# Schema-drift alarm: packed little-endian byte sizes from plan §4.1-4.5, §4.10.
# If any field name/order/kind changes, these break loudly.
EXPECTED_ITEMSIZE = {
    "arm_state": 298,
    "arm_target": 61,
    "arm_cmd": 114776,  # includes MAX_ARM_WAYPOINTS(2048)x7 f8 waypoint block
    "arm_cmd_result": 73,
    "hand_state": 14735,  # includes tactile_force (5,120,3) f8
    "hand_cmd": 100,
    "policy_chunk": 2444,
}

ALL_DTYPES = {
    "arm_state": ARM_STATE_DTYPE,
    "arm_target": ARM_TARGET_DTYPE,
    "arm_cmd": ARM_CMD_DTYPE,
    "arm_cmd_result": ARM_CMD_RESULT_DTYPE,
    "hand_state": HAND_STATE_DTYPE,
    "hand_cmd": HAND_CMD_DTYPE,
    "policy_chunk": POLICY_CHUNK_DTYPE,
}

# Documented ring block layout (robot_ring.py): 64B header, then slots of
# [timestamp_ns u64 @0, sequence u64 @8, data @16].
_HEADER_SIZE = 64
_OFF_WRITE_IDX = 0


def shm_name(test_id: str) -> str:
    """Unique SHM name per test + per run (avoids cross-run leftovers)."""
    return f"dexmani_test_{test_id}_{os.getpid()}"


def fill_known(frame: np.ndarray) -> np.ndarray:
    """Fill every field of a 1-record frame with deterministic known values."""
    for i, name in enumerate(frame.dtype.names):
        if frame.dtype[name].kind == "f":
            frame[name] = 0.25 * (i + 1)
        else:  # u1/u4/u8/i4/i8 — small ints fit every width
            frame[name] = i + 1
    return frame


def assert_records_equal(got: np.ndarray, want: np.ndarray) -> None:
    """Field-by-field equality (robust across numpy versions, names the field)."""
    assert got.shape == want.shape == (1,)
    for name in want.dtype.names:
        assert np.array_equal(got[name], want[name]), f"field {name!r} mismatch"


# ----------------------------------------------------------------------
# 1. Layout roundtrip + schema-drift alarm
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_DTYPES))
def test_layout_roundtrip(name: str) -> None:
    """new_frame → known values → write/read_latest → identical values + shape."""
    dtype = ALL_DTYPES[name]
    assert dtype.itemsize == EXPECTED_ITEMSIZE[name], (
        f"{name} itemsize {dtype.itemsize} != expected {EXPECTED_ITEMSIZE[name]} — "
        "SHM schema drifted from plan §4 (byte layout must not change)"
    )

    ring = SeqlockRingBuffer(shm_name(f"roundtrip_{name}"), dtype, maxlen=2, create=True)
    try:
        # Empty ring: no frame, age -1, sequence 0.
        assert ring.read_latest() is None
        assert ring.frame_age_ns() == -1
        assert ring.latest_sequence == 0

        frame = fill_known(new_frame(dtype))
        seq = ring.write(frame)
        assert seq == 1 == ring.latest_sequence

        got = ring.read_latest()
        assert got is not None
        data, ts_ns, rseq = got
        assert rseq == seq
        assert ts_ns > 0
        assert data.dtype == dtype
        assert_records_equal(data, frame)
        assert ring.frame_age_ns() >= 0
    finally:
        ring.close()
        ring.unlink()


# ----------------------------------------------------------------------
# 2. is_fresh fake clock
# ----------------------------------------------------------------------


def test_is_fresh_fake_clock_boundaries() -> None:
    """Injected now_ns: exactly-at-timeout is fresh, 1 ns beyond is not; ts<=0 never."""
    now = 10_000_000_000  # 10 s in ns
    timeout_s = 0.2  # 200 ms budget

    assert is_fresh(now - 200_000_000, timeout_s, now_ns=now) is True  # boundary: exactly at timeout
    assert is_fresh(now - 200_000_001, timeout_s, now_ns=now) is False  # 1 ns beyond
    assert is_fresh(now - 100_000_000, timeout_s, now_ns=now) is True  # comfortably fresh
    assert is_fresh(now - 1_000_000_000, timeout_s, now_ns=now) is False  # stale

    # Never-written slot timestamps are never fresh.
    assert is_fresh(0, timeout_s, now_ns=now) is False
    assert is_fresh(-5, timeout_s, now_ns=now) is False

    # Default-clock path: a just-taken timestamp is fresh.
    assert is_fresh(time.monotonic_ns(), 1.0) is True


# ----------------------------------------------------------------------
# 3. Torn-read injection → last-good fallback, never garbage, never raises
# ----------------------------------------------------------------------


def test_torn_read_returns_last_good_never_garbage() -> None:
    """A zeroed slot sequence (writer mid-overwrite) must yield the cached
    last-good frame — even when the slot's data region holds garbage."""
    dtype = ARM_TARGET_DTYPE
    maxlen = 2
    slot_dtype = np.dtype([("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", dtype)])

    ring = SeqlockRingBuffer(shm_name("torn"), dtype, maxlen=maxlen, create=True)
    try:
        # Good frame A → read succeeds and populates the last-good cache.
        frame_a = new_frame(dtype)
        frame_a["target"] = np.arange(1, 8, dtype=np.float64)
        frame_a["is_hold"] = 0
        frame_a["producer_id"] = PRODUCER_TELEOP
        seq_a = ring.write(frame_a)  # seq 1 → slot 1

        got = ring.read_latest()
        assert got is not None
        assert_records_equal(got[0], frame_a)

        # Simulate a mid-write slot via a raw attach to the same SHM block:
        # zero the latest slot's sequence (seqlock mid-write marker) and
        # scribble its data, leaving write_idx untouched.
        raw = shared_memory.SharedMemory(name=ring.name)
        try:
            write_idx = int(np.ndarray((1,), dtype="<u8", buffer=raw.buf, offset=_OFF_WRITE_IDX)[0])
            slots = np.ndarray((maxlen,), dtype=slot_dtype, buffer=raw.buf, offset=_HEADER_SIZE)
            slots[write_idx]["sequence"] = 0
            slots[write_idx]["data"]["target"] = -999.0  # garbage a naive reader would return
        finally:
            raw.close()

        # Reader must fall back to last-good (frame A), never garbage, never raise.
        got2 = ring.read_latest()
        assert got2 is not None, "torn read must fall back to the last-good cache"
        data2, _ts2, seq2 = got2
        assert seq2 == seq_a, "last-good cache must be frame A"
        assert_records_equal(data2, frame_a)
        assert not np.any(data2["target"] == -999.0)

        # A repeat read is stable (still last-good, no raise).
        got2b = ring.read_latest()
        assert got2b is not None and got2b[2] == seq_a

        # Recovery: a proper write restores consistent reads.
        frame_b = new_frame(dtype)
        frame_b["target"] = np.arange(11, 18, dtype=np.float64)
        frame_b["producer_id"] = PRODUCER_POLICY
        seq_b = ring.write(frame_b)
        got3 = ring.read_latest()
        assert got3 is not None
        assert got3[2] == seq_b
        assert_records_equal(got3[0], frame_b)
    finally:
        ring.close()
        ring.unlink()


def test_torn_read_odd_sequence_mid_write_never_accepted() -> None:
    """An ODD slot sequence (reader sampled between the writer's two seqlock
    stores) must be treated as mid-write → last-good — even when seq1 == seq2
    (both samples land on the same odd value, the interleaving a single-store
    seqlock cannot distinguish)."""
    dtype = ARM_TARGET_DTYPE
    maxlen = 2
    slot_dtype = np.dtype([("timestamp_ns", "<u8"), ("sequence", "<u8"), ("data", dtype)])

    ring = SeqlockRingBuffer(shm_name("torn_odd"), dtype, maxlen=maxlen, create=True)
    try:
        frame_a = new_frame(dtype)
        frame_a["target"] = np.arange(1, 8, dtype=np.float64)
        frame_a["producer_id"] = PRODUCER_TELEOP
        seq_a = ring.write(frame_a)

        got = ring.read_latest()
        assert got is not None
        assert got[2] == seq_a

        # Simulate the writer's opening store: ODD marker (2*next-1) + garbage
        # data, exactly what a reader racing the writer would sample.
        raw = shared_memory.SharedMemory(name=ring.name)
        try:
            write_idx = int(np.ndarray((1,), dtype="<u8", buffer=raw.buf, offset=_OFF_WRITE_IDX)[0])
            slots = np.ndarray((maxlen,), dtype=slot_dtype, buffer=raw.buf, offset=_HEADER_SIZE)
            slots[write_idx]["sequence"] = 2 * seq_a + 1  # odd: write in progress
            slots[write_idx]["data"]["target"] = -999.0
        finally:
            raw.close()

        # Both samples agree on the ODD value — must still fall back to
        # last-good, never the garbage.
        got2 = ring.read_latest()
        assert got2 is not None, "odd-sequence read must fall back to the last-good cache"
        data2, _ts2, seq2 = got2
        assert seq2 == seq_a, "last-good cache must be frame A"
        assert_records_equal(data2, frame_a)
        assert not np.any(data2["target"] == -999.0)

        # A completed write (even marker) restores consistent reads, and the
        # sequence handed back is the LOGICAL one (not the doubled marker).
        frame_b = new_frame(dtype)
        frame_b["target"] = np.arange(21, 28, dtype=np.float64)
        frame_b["producer_id"] = PRODUCER_POLICY
        seq_b = ring.write(frame_b)
        assert seq_b == seq_a + 1
        got3 = ring.read_latest()
        assert got3 is not None
        assert got3[2] == seq_b
        assert_records_equal(got3[0], frame_b)
    finally:
        ring.close()
        ring.unlink()


def test_logical_sequence_across_wraps() -> None:
    """write() returns and read_latest() reports the LOGICAL sequence across
    ring wraps (slot markers are 2*seq-1/2*seq internally; 0 stays unwritten)
    — the RPC cmd_seq correlation and hand echo rely on this."""
    ring = SeqlockRingBuffer(shm_name("wrap"), ARM_TARGET_DTYPE, maxlen=2, create=True)
    try:
        for i in range(1, 8):  # several wraps of a 2-slot ring
            frame = new_frame(ARM_TARGET_DTYPE)
            frame["target"] = np.full(7, float(i))
            assert ring.write(frame) == i
            got = ring.read_latest()
            assert got is not None
            assert got[2] == i, f"read seq {got[2]} != logical {i}"
            assert float(got[0]["target"][0][0]) == float(i)
        assert ring.latest_sequence == 7
    finally:
        ring.close()
        ring.unlink()


# ----------------------------------------------------------------------
# 4. Stale SHM cleanup on create
# ----------------------------------------------------------------------


def test_stale_shm_cleanup_on_create() -> None:
    """A leftover block (wrong size) from a crashed run is unlinked and
    recreated when stale_cleanup=True; without it, FileExistsError propagates."""
    name = shm_name("stale")

    # Leftover from a "crashed" run: smaller than the ring needs, never unlinked.
    stale = shared_memory.SharedMemory(name=name, create=True, size=128)
    stale.close()  # leave the block linked

    try:
        with pytest.raises(FileExistsError):
            SeqlockRingBuffer(name, ARM_TARGET_DTYPE, maxlen=2, create=True, stale_cleanup=False)

        ring = SeqlockRingBuffer(name, ARM_TARGET_DTYPE, maxlen=2, create=True, stale_cleanup=True)
        try:
            frame = new_frame(ARM_TARGET_DTYPE)
            frame["target"] = np.arange(1, 8, dtype=np.float64)
            ring.write(frame)
            got = ring.read_latest()
            assert got is not None
            assert_records_equal(got[0], frame)
        finally:
            ring.close()
            ring.unlink()
    finally:
        # If the ring never got created (unexpected failure), drop the stale block.
        try:
            leftover = shared_memory.SharedMemory(name=name)
            leftover.close()
            leftover.unlink()
        except FileNotFoundError:
            pass


# ----------------------------------------------------------------------
# 5. RPC roundtrip
# ----------------------------------------------------------------------


def test_rpc_roundtrip_echo() -> None:
    """Server thread (handle_pending every 1 ms) echoes ok=1 + final_qpos;
    client.call returns the matching result with cmd_seq == written seq."""
    cmd_ring = SeqlockRingBuffer(shm_name("rpc_cmd"), ARM_CMD_DTYPE, maxlen=2, create=True)
    result_ring = SeqlockRingBuffer(shm_name("rpc_result"), ARM_CMD_RESULT_DTYPE, maxlen=2, create=True)
    final_qpos = np.linspace(0.1, 0.7, 7)
    served_seqs: list[int] = []
    stop = threading.Event()

    def handler(request: np.ndarray, seq: int) -> np.ndarray:
        served_seqs.append(seq)
        result = new_frame(ARM_CMD_RESULT_DTYPE)
        result["ok"] = 1  # cmd_seq left zero — server stamps it
        result["final_qpos"] = final_qpos
        return result

    server = RpcServer(cmd_ring, result_ring, handler)

    def serve() -> None:
        while not stop.is_set():
            server.handle_pending()
            time.sleep(0.001)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        client = RpcClient(cmd_ring, result_ring, timeout_s=5.0, poll_s=0.001)
        request = new_frame(ARM_CMD_DTYPE)
        request["cmd"] = ARM_CMD_RESET_BLOCKING
        request["target"] = final_qpos

        result = client.call(request)

        assert int(result["ok"][0]) == 1
        assert np.allclose(result["final_qpos"][0], final_qpos)
        assert served_seqs, "server must have dispatched the command"
        assert int(result["cmd_seq"][0]) == served_seqs[-1] == cmd_ring.latest_sequence
    finally:
        stop.set()
        thread.join(timeout=2.0)
        cmd_ring.close()
        cmd_ring.unlink()
        result_ring.close()
        result_ring.unlink()


# ----------------------------------------------------------------------
# 6. RPC timeout without a server
# ----------------------------------------------------------------------


def test_rpc_timeout_without_server() -> None:
    """No server: call raises RpcTimeoutError promptly, naming cmd + timeout."""
    cmd_ring = SeqlockRingBuffer(shm_name("to_cmd"), ARM_CMD_DTYPE, maxlen=2, create=True)
    result_ring = SeqlockRingBuffer(shm_name("to_result"), ARM_CMD_RESULT_DTYPE, maxlen=2, create=True)
    try:
        client = RpcClient(cmd_ring, result_ring, timeout_s=0.05, poll_s=0.005)
        request = new_frame(ARM_CMD_DTYPE)
        request["cmd"] = ARM_CMD_CLEAR_ERROR

        t0 = time.monotonic()
        with pytest.raises(RpcTimeoutError) as exc_info:
            client.call(request)
        elapsed = time.monotonic() - t0

        assert elapsed < 1.0, f"timeout took {elapsed:.3f}s — must fire near timeout_s=0.05"
        msg = str(exc_info.value)
        assert "timed out" in msg
        assert f"cmd={ARM_CMD_CLEAR_ERROR}" in msg  # error names the cmd code
    finally:
        cmd_ring.close()
        cmd_ring.unlink()
        result_ring.close()
        result_ring.unlink()


# ----------------------------------------------------------------------
# 7. ARM_TARGET is_hold / producer_id roundtrip
# ----------------------------------------------------------------------


def test_arm_target_hold_and_producer_roundtrip() -> None:
    """Hold sentinel + producer_id (D9 mismatch gate input) survive the ring."""
    ring = SeqlockRingBuffer(shm_name("target"), ARM_TARGET_DTYPE, maxlen=2, create=True)
    try:
        # Hold sentinel: target ignored, is_hold=1.
        hold = new_frame(ARM_TARGET_DTYPE)
        hold["is_hold"] = 1
        hold["producer_id"] = PRODUCER_POLICY
        ring.write(hold)

        got = ring.read_latest()
        assert got is not None
        assert int(got[0]["is_hold"][0]) == 1
        assert int(got[0]["producer_id"][0]) == PRODUCER_POLICY

        # Live target from the teleop producer.
        target_vals = np.linspace(-1.0, 1.0, 7)
        tgt = new_frame(ARM_TARGET_DTYPE)
        tgt["target"] = target_vals
        tgt["is_hold"] = 0
        tgt["producer_id"] = PRODUCER_TELEOP
        ring.write(tgt)

        got2 = ring.read_latest()
        assert got2 is not None
        assert int(got2[0]["is_hold"][0]) == 0
        assert int(got2[0]["producer_id"][0]) == PRODUCER_TELEOP
        assert np.allclose(got2[0]["target"][0], target_vals)
    finally:
        ring.close()
        ring.unlink()
