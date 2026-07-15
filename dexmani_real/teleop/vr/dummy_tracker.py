"""Fake VR tracker for dry-run/testing — emits static frames."""

from __future__ import annotations

__all__ = ["DummyTracker"]

import time
from typing import Any

import numpy as np


class DummyTracker:
    """Fake VR tracker that emits static frames for dry-run / testing.

    Each call to get_latest() bumps the sequence_id and refreshes local_recv_ns
    so that frame-age staleness checks (inlined in controller.py) always report a fresh frame.
    """

    def __init__(self, wrist_pos: np.ndarray | None = None) -> None:
        self._seq = 0
        self.started = True
        self._base_wrist = (
            np.asarray(wrist_pos, dtype=np.float64)
            if wrist_pos is not None
            else np.zeros(3, dtype=np.float64)
        )

    def get_latest(self, max_age_s: float | None = None) -> dict[str, Any]:
        self._seq += 1
        return {
            "side": "right",
            "wrist_pos": self._base_wrist.copy(),
            "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "landmarks": np.zeros((21, 3), dtype=np.float64),
            "recv_ts_ns": time.monotonic_ns(),
            "source_ts_ns": time.monotonic_ns(),
            "sequence_id": self._seq,
            "source_frame_seq": self._seq,
            "coordinate_frame": "flu",
            "local_recv_ns": time.monotonic_ns(),
        }

    def connect(self) -> bool:
        return True

    def disconnect(self) -> None:
        pass
