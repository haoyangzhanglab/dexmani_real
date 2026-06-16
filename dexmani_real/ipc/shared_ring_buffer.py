"""seq_num protocol ring buffer over multiprocessing.shared_memory.

Memory layout per slot:
    [seq: int64 (8 bytes)][payload_len: int64 (8 bytes)][payload: slot_size bytes]

seq_num is a monotonically increasing write counter. Readers scan all slots
for the largest seq > last_seq and return the corresponding payload.
"""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class RingBufferConfig:
    slot_count: int = 64
    slot_size: int = 1_048_576  # 1 MB per slot
    create: bool = True


class SharedRingBuffer:
    """Single-producer single-consumer ring buffer over shared memory.

    Write protocol:
        seq = next_seq()
        write(payload, seq)

    Read protocol:
        data, seq = read(last_seq)
        if data is None: no new frame available
    """

    def __init__(self, name: str, config: RingBufferConfig) -> None:
        import multiprocessing.shared_memory as sm

        self._name = name
        self._config = config
        self._slot_bytes = 8 + 8 + config.slot_size  # seq + len + payload
        self._total_bytes = self._slot_bytes * config.slot_count

        try:
            self._shm = sm.SharedMemory(name=name, create=config.create, size=self._total_bytes)
        except Exception:
            if config.create:
                raise
            self._shm = sm.SharedMemory(name=name, create=False)

        self._buf = np.ndarray(self._total_bytes, dtype=np.uint8, buffer=self._shm.buf)
        self._seq_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, data: bytes, seq: int | None = None) -> int:
        """Write one frame. Returns the seq_num written."""
        if seq is None:
            self._seq_counter += 1
            seq = self._seq_counter
        else:
            self._seq_counter = max(self._seq_counter, seq)

        slot_idx = seq % self._config.slot_count
        offset = slot_idx * self._slot_bytes

        payload_len = min(len(data), self._config.slot_size)
        self._buf[offset : offset + 8] = np.frombuffer(
            np.int64(seq).tobytes(), dtype=np.uint8
        )
        self._buf[offset + 8 : offset + 16] = np.frombuffer(
            np.int64(payload_len).tobytes(), dtype=np.uint8
        )
        self._buf[offset + 16 : offset + 16 + payload_len] = np.frombuffer(
            data[:payload_len], dtype=np.uint8
        )
        return seq

    def read(self, last_seq: int = -1) -> tuple[bytes | None, int]:
        """Read the latest frame with seq > last_seq. Returns (data, seq)."""
        best_seq = last_seq
        best_slot = -1

        for i in range(self._config.slot_count):
            offset = i * self._slot_bytes
            seq = int(np.frombuffer(self._buf[offset : offset + 8], dtype=np.int64)[0])
            if seq > best_seq:
                best_seq = seq
                best_slot = i

        if best_slot < 0:
            return None, last_seq

        offset = best_slot * self._slot_bytes
        payload_len = int(
            np.frombuffer(self._buf[offset + 8 : offset + 16], dtype=np.int64)[0]
        )
        if payload_len <= 0 or payload_len > self._config.slot_size:
            return None, last_seq

        payload = bytes(self._buf[offset + 16 : offset + 16 + payload_len])
        return payload, best_seq

    def close(self) -> None:
        self._shm.close()
        try:
            self._shm.unlink()
        except Exception:
            pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def slot_count(self) -> int:
        return self._config.slot_count

    @property
    def slot_size(self) -> int:
        return self._config.slot_size


def example() -> None:
    import pickle

    config = RingBufferConfig(slot_count=4, slot_size=1024, create=True)
    buf = SharedRingBuffer("test_ring_buffer_example", config)

    data = {"test": [1, 2, 3], "ts": time.time()}
    seq = buf.write(pickle.dumps(data))
    print(f"wrote seq={seq}")

    payload, read_seq = buf.read(last_seq=-1)
    if payload is not None:
        restored = pickle.loads(payload)
        print(f"read seq={read_seq} data={restored}")

    buf.close()


if __name__ == "__main__":
    example()
