"""PID process communication channels via shared memory.

Two single-slot channels for Main↔PID process communication:
  - PIDTargetChannel: Main → PID (target qpos or None-sentinel)
  - PIDStateChannel: PID → Main (current qpos + error flag)

Pattern follows SharedMemoryRingBuffer (ring_buffer.py) — shared_memory.SharedMemory
+ numpy array views, lock-free single-producer/single-consumer.

Ref: ManiUniCon lock-free shared memory pattern (main.py:163-170).
"""

from __future__ import annotations

import time
from multiprocessing import shared_memory

import numpy as np

from dexmani_real.log import get_logger

logger = get_logger(__name__)

# Channel layout: 9 × float64 = 72 bytes
#   [0:7)   data[0:7]  (arm_qpos or target_qpos)
#   [7]     flag        (valid_flag or error_flag)
#   [8]     timestamp   (perf_counter)
_CHANNEL_SIZE = 9 * 8  # 9 float64 × 8 bytes


class PIDTargetChannel:
    """Main → PID process: target joint positions or None-sentinel.

    Layout (9 float64):
        [0:7)  target_qpos  — target joint positions (rad)
        [7]    valid_flag   — 1.0 = valid target, 0.0 = None-sentinel (decelerate)
        [8]    timestamp    — perf_counter when written
    """

    def __init__(self, name: str = "pid_target", create: bool = True) -> None:
        self.name = name
        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=_CHANNEL_SIZE)
        else:
            self._shm = shared_memory.SharedMemory(name=name)

        self._data = np.ndarray((9,), dtype=np.float64, buffer=self._shm.buf, offset=0)
        if create:
            self._data[:] = 0.0

    def write(self, target: np.ndarray | None, ts: float | None = None) -> None:
        """Write target qpos or None-sentinel.

        Args:
            target: (7,) array of target joint positions, or None to signal deceleration.
            ts: Timestamp (perf_counter). If None, uses time.perf_counter().
        """
        if target is not None:
            t = np.asarray(target, dtype=np.float64).ravel()
            self._data[0:7] = t[:7]
            self._data[7] = 1.0
        else:
            self._data[0:7] = 0.0
            self._data[7] = 0.0
        self._data[8] = ts if ts is not None else time.perf_counter()

    def read(self) -> tuple[np.ndarray | None, float]:
        """Read latest target.

        Returns:
            (target_qpos, timestamp) — target_qpos is None when valid_flag == 0.
        """
        flag = float(self._data[7])
        ts = float(self._data[8])
        if flag == 0.0:
            return None, ts
        return self._data[0:7].copy(), ts

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()



class PIDStateChannel:
    """PID → Main process: current arm state for teleop feedback.

    Layout (9 float64):
        [0:7)  arm_qpos    — current joint positions (rad)
        [7]    error_flag  — 1.0 = error, 0.0 = ok
        [8]    timestamp   — perf_counter when written
    """

    def __init__(self, name: str = "pid_state", create: bool = True) -> None:
        self.name = name
        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=_CHANNEL_SIZE)
        else:
            self._shm = shared_memory.SharedMemory(name=name)

        self._data = np.ndarray((9,), dtype=np.float64, buffer=self._shm.buf, offset=0)
        if create:
            self._data[:] = 0.0

    def write(self, qpos: np.ndarray, error_state: bool, ts: float | None = None) -> None:
        """Write current arm state.

        Args:
            qpos: (7,) current joint positions.
            error_state: True if PID process has an error.
            ts: Timestamp (perf_counter). If None, uses time.perf_counter().
        """
        q = np.asarray(qpos, dtype=np.float64).ravel()
        self._data[0:7] = q[:7]
        self._data[7] = 1.0 if error_state else 0.0
        self._data[8] = ts if ts is not None else time.perf_counter()

    def read(self) -> tuple[np.ndarray, bool, float]:
        """Read latest state.

        Returns:
            (arm_qpos, error_flag, timestamp).
        """
        qpos = self._data[0:7].copy()
        error_flag = float(self._data[7]) != 0.0
        ts = float(self._data[8])
        return qpos, error_flag, ts

    def close(self) -> None:
        self._shm.close()

    def unlink(self) -> None:
        self._shm.unlink()
