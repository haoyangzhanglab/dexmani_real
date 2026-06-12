"""Simple in-memory trajectory buffer for recording and playback.

NOTE: This is a transitional .npz-based implementation. The long-term plan is to
replace it with HDF5-based EpisodeRecorder (see CLAUDE.md Section 5).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


DEFAULT_RECORD_DIR = Path.cwd() / "recordings"


class TrajectoryBuffer:
    """Stores per-frame data and supports save/load to .npz files."""

    def __init__(self, max_frames: int | None = None) -> None:
        self.frames: list[dict[str, np.ndarray]] = []
        self.max_frames = max_frames

    def add_frame(self, **kwargs: float | np.ndarray) -> bool:
        if self.max_frames is not None and len(self.frames) >= self.max_frames:
            return False
        self.frames.append({
            key: np.asarray(value, dtype=np.float64)
            for key, value in kwargs.items()
        })
        return True

    def __len__(self) -> int:
        return len(self.frames)

    def __bool__(self) -> bool:
        return len(self.frames) > 0

    def clear(self) -> None:
        self.frames.clear()

    def get_frame(self, index: int) -> dict[str, np.ndarray]:
        return self.frames[index]

    def get_array(self, key: str) -> np.ndarray:
        return np.stack([f[key] for f in self.frames], axis=0)

    @property
    def keys(self) -> list[str]:
        return list(self.frames[0].keys()) if self.frames else []

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            raise ValueError("TrajectoryBuffer is empty, nothing to save.")
        data: dict[str, np.ndarray] = {}
        for key in self.frames[0]:
            data[key] = self.get_array(key)
        np.savez_compressed(str(path), **data)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryBuffer":
        data = np.load(str(path))
        buffer = cls()
        keys = sorted(data.files)
        if not keys:
            return buffer
        n = len(data[keys[0]])
        for i in range(n):
            frame = {key: data[key][i].copy() for key in keys}
            buffer.frames.append(frame)
        data.close()
        return buffer

    def iter_frames(self) -> Iterator[dict[str, np.ndarray]]:
        yield from self.frames

    @property
    def duration(self) -> float:
        if not self.frames or "timestamp" not in self.frames[0]:
            return 0.0
        ts = self.get_array("timestamp")
        return float(ts[-1] - ts[0])

    def summary(self) -> str:
        if not self.frames:
            return "TrajectoryBuffer(empty)"
        keys_str = ", ".join(self.keys[:6])
        if len(self.keys) > 6:
            keys_str += f", ... (+{len(self.keys) - 6})"
        return (
            f"TrajectoryBuffer(frames={len(self.frames)}, "
            f"duration={self.duration:.2f}s, keys=[{keys_str}])"
        )


def get_next_episode_path(base_dir: str | Path | None = None) -> Path:
    base_dir = Path(base_dir or DEFAULT_RECORD_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    episode_num = 0
    while True:
        path = base_dir / f"episode_{episode_num:03d}" / "trajectory.npz"
        if not path.exists():
            return path
        episode_num += 1


def list_episodes(base_dir: str | Path | None = None) -> list[Path]:
    base_dir = Path(base_dir or DEFAULT_RECORD_DIR)
    if not base_dir.exists():
        return []
    episodes: list[Path] = []
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("episode_"):
            traj = entry / "trajectory.npz"
            if traj.exists():
                episodes.append(traj)
    return episodes


def get_latest_episode(base_dir: str | Path | None = None) -> Path | None:
    episodes = list_episodes(base_dir)
    return episodes[-1] if episodes else None
