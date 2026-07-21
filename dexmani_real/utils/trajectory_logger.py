"""Per-frame VR wrist + EEF trajectory logger for offline motion debug.

Stores per-tick data in append lists, flushes to an .npz file on save().
Frame data is recorded regardless of teleop/recording state so the full
session motion can be analysed.
"""

from __future__ import annotations

import numpy as np


class TrajectoryLogger:
    """Record VR wrist + EEF trajectories per-frame for offline debug.

    Stores per-tick data in append lists, flushes to an .npz file on save().
    Frame data is recorded regardless of teleop/recording state so the full
    session motion can be analysed.
    """

    def __init__(self) -> None:
        self._records: list[dict[str, object]] = []

    def append(
        self,
        t: float,
        wrist_pos: np.ndarray,
        wrist_quat_wxyz: np.ndarray,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        actual_eef_pos: np.ndarray,
        actual_eef_quat_wxyz: np.ndarray,
        arm_qpos_actual: np.ndarray,
        ik_ok: bool,
        *,
        wrist_delta: np.ndarray | None = None,
        eef_delta: np.ndarray | None = None,
        target_pos_before_clamp: np.ndarray | None = None,
    ) -> None:
        self._records.append(
            {
                "t": float(t),
                "wrist_pos": np.asarray(wrist_pos, dtype=np.float64).copy(),
                "wrist_quat_wxyz": np.asarray(wrist_quat_wxyz, dtype=np.float64).copy(),
                "target_pos": np.asarray(target_pos, dtype=np.float64).copy(),
                "target_quat_wxyz": np.asarray(target_quat_wxyz, dtype=np.float64).copy(),
                "actual_eef_pos": np.asarray(actual_eef_pos, dtype=np.float64).copy(),
                "actual_eef_quat_wxyz": np.asarray(actual_eef_quat_wxyz, dtype=np.float64).copy(),
                "arm_qpos_actual": np.asarray(arm_qpos_actual, dtype=np.float64).copy(),
                "ik_ok": bool(ik_ok),
                "wrist_delta": (
                    np.asarray(wrist_delta, dtype=np.float64).copy()
                    if wrist_delta is not None
                    else np.full(3, np.nan, dtype=np.float64)
                ),
                "eef_delta": (
                    np.asarray(eef_delta, dtype=np.float64).copy()
                    if eef_delta is not None
                    else np.full(3, np.nan, dtype=np.float64)
                ),
                "target_pos_before_clamp": (
                    np.asarray(target_pos_before_clamp, dtype=np.float64).copy()
                    if target_pos_before_clamp is not None
                    else np.full(3, np.nan, dtype=np.float64)
                ),
            }
        )

    def __len__(self) -> int:
        return len(self._records)

    def save(self, path: str) -> str:
        """Stack all records into arrays and write to .npz. Returns path."""
        if not self._records:
            raise ValueError("No trajectory data to save")

        data: dict[str, np.ndarray] = {}
        keys = list(self._records[0].keys())
        for key in keys:
            stacked: np.ndarray = np.stack([r[key] for r in self._records])  # type: ignore[arg-type,misc]
            data[key] = stacked

        np.savez_compressed(path, **data)
        return path
