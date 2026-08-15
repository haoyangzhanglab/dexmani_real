"""A10: ``get_last_k`` returns verified frames oldest-first and bounds k.

Covers the drop-oldest FILO history contract: overwritten slots are skipped
without discarding already-collected older history, results are oldest-first,
``k <= 0`` is empty, and ``k > maxlen`` raises.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.shm.ring_buffer import SharedMemoryRingBuffer


def _values(frames: list[tuple[np.ndarray, int, int]]) -> list[float]:
    return [float(frame["v"][0]) for frame, _, _ in frames]


def main() -> int:
    dtype = np.dtype([("v", "<f8")])
    name = "check_ring_history_test"
    ring = SharedMemoryRingBuffer.create_or_replace(name, dtype, maxlen=3)
    try:
        # Empty ring.
        assert ring.get_last_k(3) == [], "empty ring must return []"
        assert ring.get_last_k(0) == [], "k<=0 must return []"

        # Write 5 frames into a 3-slot ring (two overwrites).
        for i in range(5):
            frame = np.zeros(1, dtype=dtype)
            frame["v"][0] = float(i)
            ring.write(frame)

        # Oldest-first: the 3 survivors are [2, 3, 4].
        last3 = ring.get_last_k(3)
        assert _values(last3) == [2.0, 3.0, 4.0], _values(last3)
        # A shorter window returns the newest k, still oldest-first.
        assert _values(ring.get_last_k(2)) == [3.0, 4.0], _values(ring.get_last_k(2))
        assert _values(ring.get_last_k(1)) == [4.0], _values(ring.get_last_k(1))

        # k exceeding capacity is a programming error, not a silent clamp.
        try:
            ring.get_last_k(4)
        except ValueError:
            pass
        else:
            raise AssertionError("get_last_k(k>maxlen) must raise ValueError")
    finally:
        ring.close()
        ring.unlink()

    print("check_ring_history: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
