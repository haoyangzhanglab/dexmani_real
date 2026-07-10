"""ReplayBuffer — load recorded teleop episodes for training and inspection.

Provides episode-level and step-level access to HDF5 recordings and Zarr
exports.  Read-only by design — writing is handled by EpisodeRecorder.

Usage::

    from dexmani_real.recording import ReplayBuffer, DataLoadConfig

    # From raw HDF5 recordings
    config = DataLoadConfig(filter_success=True, min_frames=50)
    buffer = ReplayBuffer.from_hdf5("./episodes/", config=config)
    print(buffer.summary())

    # From pre-exported Zarr
    buffer = ReplayBuffer.from_zarr("./export/data.zarr")

    # Iterate episodes
    for ep in buffer.iter_episodes():
        print(f"{ep['meta']['task_label']}: {len(ep['obs'])} frames")

    # Training windows (never cross episode boundaries)
    buffer.compute_norm_stats()
    buffer.normalize()
    for window in buffer.iter_steps(window_size=100, stride=50):
        obs = window["obs"]       # (100, obs_dim)
        action = window["action"] # (100, action_dim)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import zarr

from dexmani_real.tools.export_hdf5_to_zarr import _read_episode_meta, compute_norm_stats, load_episodes
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Default observation / action keys — must match export_hdf5_to_zarr.py.
_DEFAULT_OBS_KEYS: list[tuple[str, int]] = [
    ("arm_qpos", 7),
    ("arm_ee", 9),
    ("hand_qpos", 12),
]
_DEFAULT_ACTION_KEYS: list[tuple[str, int]] = [
    ("action_arm_joint", 7),
    ("action_hand_joint", 12),
]


@dataclass
class DataLoadConfig:
    """Which HDF5 keys to load and optional episode-level filters."""

    obs_keys: list[tuple[str, int]] = field(default_factory=lambda: _DEFAULT_OBS_KEYS)
    action_keys: list[tuple[str, int]] = field(default_factory=lambda: _DEFAULT_ACTION_KEYS)
    filter_task: str | None = None
    filter_success: bool | None = None
    filter_tags: str | None = None
    min_frames: int | None = None


class ReplayBuffer:
    """Load and iterate over recorded teleop episodes from HDF5 or Zarr.

    Internal representation (all in-memory, float32)::

        _obs_list:    list[np.ndarray]   each (T_i, obs_dim)
        _action_list: list[np.ndarray]   each (T_i, action_dim)
        _meta_list:   list[dict]         per-episode metadata
        _episode_ends: np.ndarray        cumulative frame counts (int64)
        _episode_paths: list[Path]       source file paths
        _norm_stats:  dict | None        computed or loaded norm stats
    """

    def __init__(self) -> None:
        self._obs_list: list[np.ndarray] = []
        self._action_list: list[np.ndarray] = []
        self._meta_list: list[dict] = []
        self._episode_ends: np.ndarray = np.array([], dtype=np.int64)
        self._episode_paths: list[Path] = []
        self._norm_stats: dict[str, np.ndarray] | None = None
        self._obs_dim: int = 0
        self._action_dim: int = 0

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_hdf5(
        cls,
        data_dir: str | Path,
        config: DataLoadConfig | None = None,
    ) -> "ReplayBuffer":
        """Load all HDF5 episodes from *data_dir*.

        Args:
            data_dir: Directory containing ``episode_*.h5`` files.
            config:  Optional load configuration (keys, filters).  Defaults to
                     the standard obs/action key sets.
        """
        data_dir = Path(data_dir)
        cfg = config or DataLoadConfig()

        # Temporarily patch the module-level key lists so load_episodes()
        # reads the caller's key configuration.
        import dexmani_real.tools.export_hdf5_to_zarr as _export_mod

        orig_obs = _export_mod._OBS_KEYS
        orig_act = _export_mod._ACTION_KEYS
        _export_mod._OBS_KEYS = cfg.obs_keys
        _export_mod._ACTION_KEYS = cfg.action_keys
        try:
            obs_list, action_list, episode_lengths, episode_paths = load_episodes(
                data_dir,
                filter_task=cfg.filter_task,
                filter_success=cfg.filter_success,
                filter_tags=cfg.filter_tags,
                min_frames=cfg.min_frames,
            )
        finally:
            _export_mod._OBS_KEYS = orig_obs
            _export_mod._ACTION_KEYS = orig_act

        self = cls()
        self._obs_list = obs_list
        self._action_list = action_list
        self._episode_paths = episode_paths
        self._episode_ends = np.cumsum(episode_lengths, dtype=np.int64)

        # Read per-episode metadata
        for p in episode_paths:
            self._meta_list.append(_read_episode_meta(p))

        if obs_list:
            self._obs_dim = obs_list[0].shape[1]
            self._action_dim = action_list[0].shape[1]

        logger.info(
            "Loaded %d episodes, %d total frames (obs_dim=%d, action_dim=%d)",
            len(obs_list),
            int(self._episode_ends[-1]) if len(self._episode_ends) > 0 else 0,
            self._obs_dim,
            self._action_dim,
        )
        return self

    @classmethod
    def from_zarr(cls, zarr_path: str | Path) -> "ReplayBuffer":
        """Load from a Zarr store produced by ``export_hdf5_to_zarr``.

        Expects the standard layout::

            data.zarr/
              data/obs      (T, D) float32
              data/action   (T, D) float32
              meta/episode_ends  (E,) int64
              meta/norm_stats/   (optional)
        """
        zarr_path = Path(zarr_path)
        root = zarr.open_group(str(zarr_path), mode="r")

        if "data" not in root:
            raise KeyError(f"Zarr store {zarr_path} missing 'data' group")
        if "meta" not in root:
            raise KeyError(f"Zarr store {zarr_path} missing 'meta' group")

        all_obs = np.asarray(root["data"]["obs"], dtype=np.float32)
        all_action = np.asarray(root["data"]["action"], dtype=np.float32)
        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)

        # Split flat arrays back into per-episode lists
        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []
        prev = 0
        for end in episode_ends:
            obs_list.append(all_obs[prev:end])
            action_list.append(all_action[prev:end])
            prev = int(end)

        # Reconstruct minimal metadata
        episode_paths: list[Path] = []
        meta_list: list[dict] = []
        for i, length in enumerate(np.diff(episode_ends, prepend=0)):
            episode_paths.append(zarr_path / f"episode_{i}")
            meta_list.append({"num_frames": int(length), "source": str(zarr_path)})

        self = cls()
        self._obs_list = obs_list
        self._action_list = action_list
        self._meta_list = meta_list
        self._episode_paths = episode_paths
        self._episode_ends = episode_ends
        self._obs_dim = int(all_obs.shape[1])
        self._action_dim = int(all_action.shape[1])

        # Load norm stats if present
        norm_stats_path = "meta/norm_stats"
        if norm_stats_path in root:
            stats = root[norm_stats_path]
            self._norm_stats = {
                "obs_mean": np.asarray(stats["obs_mean"], dtype=np.float32),
                "obs_std": np.asarray(stats["obs_std"], dtype=np.float32),
                "action_mean": np.asarray(stats["action_mean"], dtype=np.float32),
                "action_std": np.asarray(stats["action_std"], dtype=np.float32),
            }

        logger.info(
            "Loaded Zarr %s: %d episodes, %d total frames (obs_dim=%d, action_dim=%d)",
            zarr_path,
            len(obs_list),
            int(episode_ends[-1]) if len(episode_ends) > 0 else 0,
            self._obs_dim,
            self._action_dim,
        )
        return self

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_episodes(self) -> int:
        """Number of loaded episodes."""
        return len(self._obs_list)

    @property
    def n_steps(self) -> int:
        """Total number of frames across all episodes."""
        if len(self._episode_ends) == 0:
            return 0
        return int(self._episode_ends[-1])

    @property
    def obs_dim(self) -> int:
        """Observation vector dimension."""
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        """Action vector dimension."""
        return self._action_dim

    @property
    def episode_ends(self) -> np.ndarray:
        """Cumulative frame counts (int64, shape (n_episodes,))."""
        return self._episode_ends

    @property
    def norm_stats(self) -> dict[str, np.ndarray] | None:
        """Normalization statistics dict or ``None`` if not yet computed/loaded."""
        return self._norm_stats

    # ── Episode-level access ─────────────────────────────────────────────────

    def get_episode(self, idx: int) -> dict:
        """Return episode *idx* as a dict with keys ``obs``, ``action``, ``meta``, ``path``."""
        if idx < 0:
            idx += len(self._obs_list)
        if idx < 0 or idx >= len(self._obs_list):
            raise IndexError(f"Episode index {idx} out of range [0, {len(self._obs_list)})")
        return {
            "obs": self._obs_list[idx],
            "action": self._action_list[idx],
            "meta": self._meta_list[idx],
            "path": self._episode_paths[idx],
        }

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, idx: int) -> dict:
        return self.get_episode(idx)

    # ── Step-level access ────────────────────────────────────────────────────

    def get_step_slice(self, start: int, stop: int) -> dict:
        """Return ``{obs, action}`` for global step range ``[start, stop)``.

        Spans episode boundaries transparently — concatenates slices from
        all episodes that fall within the requested range.
        """
        if len(self._episode_ends) == 0:
            return {"obs": np.empty((0, self._obs_dim), dtype=np.float32),
                    "action": np.empty((0, self._action_dim), dtype=np.float32)}

        total = int(self._episode_ends[-1])
        if start < 0:
            start = max(0, total + start)
        if stop < 0:
            stop = max(0, total + stop)
        start = max(0, min(start, total))
        stop = max(0, min(stop, total))
        if start >= stop:
            return {"obs": np.empty((0, self._obs_dim), dtype=np.float32),
                    "action": np.empty((0, self._action_dim), dtype=np.float32)}

        first_ep = int(np.searchsorted(self._episode_ends, start, side="right"))
        last_ep = int(np.searchsorted(self._episode_ends, stop, side="right"))

        obs_parts: list[np.ndarray] = []
        act_parts: list[np.ndarray] = []

        ep_start_global = 0 if first_ep == 0 else int(self._episode_ends[first_ep - 1])
        for ep_idx in range(first_ep, min(last_ep + 1, len(self._obs_list))):
            ep_end_global = int(self._episode_ends[ep_idx])
            local_start = max(0, start - ep_start_global)
            local_stop = min(len(self._obs_list[ep_idx]), stop - ep_start_global)
            if local_start < local_stop:
                obs_parts.append(self._obs_list[ep_idx][local_start:local_stop])
                act_parts.append(self._action_list[ep_idx][local_start:local_stop])
            ep_start_global = ep_end_global

        if not obs_parts:
            return {"obs": np.empty((0, self._obs_dim), dtype=np.float32),
                    "action": np.empty((0, self._action_dim), dtype=np.float32)}

        return {
            "obs": np.concatenate(obs_parts, axis=0),
            "action": np.concatenate(act_parts, axis=0),
        }

    # ── Iterators ────────────────────────────────────────────────────────────

    def iter_episodes(self) -> Iterator[dict]:
        """Yield ``{obs, action, meta, path}`` for each episode."""
        for i in range(len(self._obs_list)):
            yield self.get_episode(i)

    def iter_steps(self, window_size: int, stride: int = 1) -> Iterator[dict]:
        """Yield ``{obs, action}`` windows of fixed length.

        Windows **never** cross episode boundaries.  Episodes shorter than
        *window_size* are skipped.
        """
        for obs, action in zip(self._obs_list, self._action_list):
            t = obs.shape[0]
            if t < window_size:
                continue
            for start in range(0, t - window_size + 1, stride):
                yield {
                    "obs": obs[start : start + window_size],
                    "action": action[start : start + window_size],
                }

    # ── Normalization ────────────────────────────────────────────────────────

    def compute_norm_stats(self) -> dict[str, np.ndarray]:
        """Compute per-dimension mean/std across all frames and store internally."""
        self._norm_stats = compute_norm_stats(self._obs_list, self._action_list)
        return self._norm_stats

    def normalize(self) -> None:
        """Normalize all obs and action data in-place using stored stats.

        Requires :meth:`compute_norm_stats` or :meth:`from_zarr` (with stats)
        to have been called first.
        """
        if self._norm_stats is None:
            raise RuntimeError(
                "No normalization stats available. Call compute_norm_stats() first, "
                "or load from a Zarr that contains meta/norm_stats."
            )
        obs_mean = self._norm_stats["obs_mean"]
        obs_std = self._norm_stats["obs_std"]
        act_mean = self._norm_stats["action_mean"]
        act_std = self._norm_stats["action_std"]

        for i in range(len(self._obs_list)):
            self._obs_list[i] = (self._obs_list[i] - obs_mean) / obs_std
            self._action_list[i] = (self._action_list[i] - act_mean) / act_std

    # ── Utility ──────────────────────────────────────────────────────────────

    def episode_lengths(self) -> np.ndarray:
        """Return per-episode frame counts."""
        if len(self._episode_ends) == 0:
            return np.array([], dtype=np.int64)
        return np.diff(self._episode_ends, prepend=0)

    def summary(self) -> str:
        """Human-readable summary of the loaded data."""
        n = self.n_episodes
        if n == 0:
            return "ReplayBuffer: empty (0 episodes)"

        lengths = self.episode_lengths()
        total_frames = int(self._episode_ends[-1])
        tasks = set()
        for m in self._meta_list:
            task = m.get("task_label", "")
            if task:
                tasks.add(task)

        lines = [
            f"ReplayBuffer: {n} episodes, {total_frames} total frames",
            f"  obs_dim={self._obs_dim}, action_dim={self._action_dim}",
            f"  frames/episode: min={lengths.min()}, max={lengths.max()}, "
            f"mean={lengths.mean():.0f}, median={int(np.median(lengths))}",
        ]
        if tasks:
            lines.append(f"  tasks: {', '.join(sorted(tasks))}")
        if self._norm_stats is not None:
            lines.append("  norm_stats: computed")
        return "\n".join(lines)
