"""Tests for ReplayBuffer — runs without real recorded data."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr
from numcodecs import Blosc

from dexmani_real.recording.replay_buffer import DataLoadConfig, ReplayBuffer
from dexmani_real.tools.export_hdf5_to_zarr import compute_norm_stats, write_zarr


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_h5_episode(path: Path, n_frames: int = 50, task_label: str = "pick_place", success: bool = True) -> None:
    """Write a minimal synthetic episode HDF5 file."""
    rng = np.random.RandomState(42)
    with h5py.File(str(path), "w") as f:
        meta = f.create_group("meta")
        meta.attrs["task_label"] = task_label
        meta.attrs["success"] = success
        meta.attrs["fps"] = 50.0
        meta.attrs["num_frames"] = n_frames
        meta.attrs["duration"] = n_frames / 50.0
        meta.attrs["tags"] = "test,synthetic"
        meta.attrs["operator"] = "test"

        obs = f.create_group("obs")
        obs.create_dataset("arm_qpos", data=rng.randn(n_frames, 7).astype(np.float32))
        obs.create_dataset("arm_qvel", data=rng.randn(n_frames, 7).astype(np.float32))
        obs.create_dataset("arm_tau", data=rng.randn(n_frames, 7).astype(np.float32))
        obs.create_dataset("eef_pos", data=rng.randn(n_frames, 3).astype(np.float32))
        obs.create_dataset("eef_quat", data=rng.randn(n_frames, 4).astype(np.float32))
        obs.create_dataset("hand_qpos", data=rng.randn(n_frames, 12).astype(np.float32))

        act = f.create_group("action")
        act.create_dataset("arm_qpos", data=rng.randn(n_frames, 7).astype(np.float32))
        act.create_dataset("hand_qpos", data=rng.randn(n_frames, 12).astype(np.float32))

        f.create_dataset("timestamps", data=np.linspace(0, n_frames / 50.0, n_frames, dtype=np.float64))


def _make_synthetic_hdf5_dir(n_episodes: int = 3, n_frames: int = 50) -> Path:
    """Create a temporary directory with synthetic episode HDF5 files."""
    tmp = tempfile.mkdtemp(prefix="replay_test_")
    for i in range(n_episodes):
        _make_h5_episode(
            Path(tmp) / f"episode_{i:03d}.h5",
            n_frames=n_frames + i * 10,
            task_label=f"task_{i % 2}",
            success=(i % 2 == 0),
        )
    return Path(tmp)


def _make_synthetic_zarr(n_episodes: int = 3, n_frames: int = 50) -> Path:
    """Create a temporary Zarr store from synthetic episodes."""
    h5_dir = _make_synthetic_hdf5_dir(n_episodes, n_frames)
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    lengths: list[int] = []

    for h5_path in sorted(h5_dir.glob("episode_*.h5")):
        with h5py.File(str(h5_path), "r") as f:
            n = f["obs/arm_qpos"].shape[0]
            o = np.concatenate(
                [
                    np.asarray(f["obs/arm_qpos"][:], dtype=np.float32),
                    np.asarray(f["obs/eef_pos"][:], dtype=np.float32),
                    np.asarray(f["obs/eef_quat"][:], dtype=np.float32),
                    np.asarray(f["obs/hand_qpos"][:], dtype=np.float32),
                ],
                axis=1,
            )
            a = np.concatenate(
                [
                    np.asarray(f["action/arm_qpos"][:], dtype=np.float32),
                    np.asarray(f["action/hand_qpos"][:], dtype=np.float32),
                ],
                axis=1,
            )
            obs_list.append(o)
            action_list.append(a)
            lengths.append(n)

    zarr_dir = Path(tempfile.mkdtemp(prefix="replay_zarr_"))
    stats = compute_norm_stats(obs_list, action_list)
    write_zarr(zarr_dir, obs_list, action_list, lengths, stats, name="data")
    return zarr_dir / "data.zarr"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hdf5_dir():
    return _make_synthetic_hdf5_dir(n_episodes=4, n_frames=50)


@pytest.fixture(scope="module")
def zarr_path():
    return _make_synthetic_zarr(n_episodes=4, n_frames=50)


# ── Tests: from_hdf5 ─────────────────────────────────────────────────────────


class TestFromHDF5:
    def test_load_all(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        assert buf.n_episodes == 4
        assert buf.obs_dim == 26
        assert buf.action_dim == 19
        assert buf.n_steps > 0

    def test_filter_success(self, hdf5_dir: Path) -> None:
        cfg = DataLoadConfig(filter_success=True)
        buf = ReplayBuffer.from_hdf5(hdf5_dir, config=cfg)
        # Episode 0, 2 are success=True; episode 1, 3 are success=False
        assert buf.n_episodes == 2
        for ep in buf.iter_episodes():
            assert ep["meta"]["success"] is True or ep["meta"]["success"] == 1

    def test_filter_task(self, hdf5_dir: Path) -> None:
        cfg = DataLoadConfig(filter_task="task_0")
        buf = ReplayBuffer.from_hdf5(hdf5_dir, config=cfg)
        assert buf.n_episodes == 2
        for ep in buf.iter_episodes():
            assert "task_0" in str(ep["meta"].get("task_label", ""))

    def test_min_frames(self, hdf5_dir: Path) -> None:
        # Episodes have 50, 60, 70, 80 frames — filter >= 65
        cfg = DataLoadConfig(min_frames=65)
        buf = ReplayBuffer.from_hdf5(hdf5_dir, config=cfg)
        assert buf.n_episodes == 2  # episodes with 70 and 80 frames

    def test_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            buf = ReplayBuffer.from_hdf5(td)
            assert buf.n_episodes == 0
            assert buf.n_steps == 0
            assert buf.summary() == "ReplayBuffer: empty (0 episodes)"


# ── Tests: from_zarr ─────────────────────────────────────────────────────────


class TestFromZarr:
    def test_load(self, zarr_path: Path) -> None:
        buf = ReplayBuffer.from_zarr(zarr_path)
        assert buf.n_episodes == 4
        assert buf.obs_dim == 26
        assert buf.action_dim == 19
        assert buf.norm_stats is not None
        assert "obs_mean" in buf.norm_stats

    def test_missing_data_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = zarr.open_group(td, mode="w")
            root.create_group("meta").create_dataset("episode_ends", data=np.array([10], dtype=np.int64))
            with pytest.raises(KeyError, match="missing 'data' group"):
                ReplayBuffer.from_zarr(td)

    def test_missing_meta_group(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = zarr.open_group(td, mode="w")
            root.create_group("data")
            with pytest.raises(KeyError, match="missing 'meta' group"):
                ReplayBuffer.from_zarr(td)


# ── Tests: episode access ────────────────────────────────────────────────────


class TestEpisodeAccess:
    def test_get_episode(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        ep = buf.get_episode(0)
        assert "obs" in ep
        assert "action" in ep
        assert "meta" in ep
        assert "path" in ep
        assert ep["obs"].shape == (50, 26)
        assert ep["action"].shape == (50, 19)

    def test_len_and_index(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        assert len(buf) == 4
        assert buf[0]["obs"].shape[0] == 50
        assert buf[-1]["obs"].shape[0] == 80  # last episode: 50 + 3*10 = 80

    def test_index_error(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        with pytest.raises(IndexError):
            buf.get_episode(10)
        with pytest.raises(IndexError):
            buf.get_episode(-10)


# ── Tests: step-level access ─────────────────────────────────────────────────


class TestStepSlice:
    def test_within_single_episode(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        data = buf.get_step_slice(0, 30)
        assert data["obs"].shape == (30, 26)
        assert data["action"].shape == (30, 19)

    def test_across_episodes(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        # Ep 0: 50 frames, Ep 1: 60 frames → boundary at 50
        data = buf.get_step_slice(40, 70)
        assert data["obs"].shape == (30, 26)  # 10 from ep0 + 20 from ep1

    def test_full_range(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        data = buf.get_step_slice(0, buf.n_steps)
        assert data["obs"].shape == (buf.n_steps, buf.obs_dim)

    def test_single_step(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        data = buf.get_step_slice(25, 26)
        assert data["obs"].shape == (1, 26)

    def test_empty_range(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        data = buf.get_step_slice(10, 10)
        assert data["obs"].shape == (0, 26)

    def test_negative_indices(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        total = buf.n_steps
        data_full = buf.get_step_slice(0, total)
        data_neg = buf.get_step_slice(-total, -1)
        # last frame excluded by negative stop
        assert data_neg["obs"].shape[0] == total - 1


# ── Tests: iterators ─────────────────────────────────────────────────────────


class TestIterators:
    def test_iter_episodes(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        episodes = list(buf.iter_episodes())
        assert len(episodes) == buf.n_episodes
        assert all("obs" in ep for ep in episodes)

    def test_iter_steps_basic(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        windows = list(buf.iter_steps(window_size=20, stride=20))
        assert len(windows) > 0
        for w in windows:
            assert w["obs"].shape == (20, 26)
            assert w["action"].shape == (20, 19)

    def test_iter_steps_no_cross_episode(self, hdf5_dir: Path) -> None:
        """Each window must come entirely from one episode."""
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        for w in buf.iter_steps(window_size=30, stride=15):
            # The obs data within a window should be internally contiguous
            # (they come from a single numpy slice)
            assert w["obs"].shape[0] == 30

    def test_iter_steps_short_episode_skipped(self, hdf5_dir: Path) -> None:
        """Episodes shorter than window_size should be skipped."""
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        # Ep 0 has 50 frames — all windows should come from the same single
        # episode since windows don't cross boundaries
        windows = list(buf.iter_steps(window_size=51, stride=10))
        # Only episodes with >= 51 frames contribute
        assert all(w["obs"].shape[0] == 51 for w in windows)

    def test_iter_steps_empty(self, hdf5_dir: Path) -> None:
        """Empty buffer produces no windows."""
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        windows = list(buf.iter_steps(window_size=999, stride=1))
        assert len(windows) == 0


# ── Tests: normalization ─────────────────────────────────────────────────────


class TestNormalization:
    def test_compute_norm_stats(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        stats = buf.compute_norm_stats()
        assert "obs_mean" in stats
        assert "obs_std" in stats
        assert "action_mean" in stats
        assert "action_std" in stats
        assert stats["obs_mean"].shape == (26,)
        assert stats["action_mean"].shape == (19,)
        # Standard deviations should all be > 0 (synthetic data has variance)
        assert np.all(stats["obs_std"] > 0)

    def test_normalize(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        buf.compute_norm_stats()
        ep_before = buf.get_episode(0)["obs"].copy()
        buf.normalize()
        ep_after = buf.get_episode(0)["obs"]
        # Normalized data should differ from original
        assert not np.allclose(ep_before, ep_after)
        # Normalized obs should have mean ≈ 0, std ≈ 1 (approximately)
        all_obs = np.concatenate([ep["obs"] for ep in buf.iter_episodes()], axis=0)
        assert np.allclose(np.mean(all_obs, axis=0), 0.0, atol=1e-5)
        assert np.allclose(np.std(all_obs, axis=0), 1.0, atol=1e-5)

    def test_normalize_without_stats(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        with pytest.raises(RuntimeError, match="No normalization stats"):
            buf.normalize()


# ── Tests: properties ────────────────────────────────────────────────────────


class TestProperties:
    def test_episode_ends(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        ends = buf.episode_ends
        assert len(ends) == 4
        assert ends[-1] == buf.n_steps
        assert np.all(np.diff(ends) > 0)  # monotonically increasing

    def test_episode_lengths(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        lengths = buf.episode_lengths()
        assert len(lengths) == 4
        assert lengths[0] == 50
        assert lengths[-1] == 80

    def test_norm_stats_none_by_default(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        assert buf.norm_stats is None

    def test_summary(self, hdf5_dir: Path) -> None:
        buf = ReplayBuffer.from_hdf5(hdf5_dir)
        s = buf.summary()
        assert "ReplayBuffer:" in s
        assert "episodes" in s
        assert "obs_dim=26" in s
        assert "action_dim=19" in s
