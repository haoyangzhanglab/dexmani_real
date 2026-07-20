"""Regression tests for ReplayBuffer.from_hdf5 (flagged bug #1 in 077ce36).

from_hdf5 crashed on every call: load_episodes() returns a 7-tuple
(obs, action, lengths, paths, rgb_list, depth_list, camera_meta) but only
4 names were unpacked. These tests pin the 7-tuple contract — both via a
stubbed load_episodes and end-to-end against synthetic episode_*.h5 files.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

import dexmani_real.recording.replay_buffer as rb_mod
import dexmani_real.tools.export_hdf5_to_zarr as export_mod
from dexmani_real.recording.replay_buffer import DataLoadConfig, ReplayBuffer


def test_from_hdf5_unpacks_seven_tuple(tmp_path, monkeypatch):
    """A stubbed load_episodes returning the real 7-tuple must load cleanly."""
    obs = [np.zeros((3, 4), dtype=np.float32), np.ones((2, 4), dtype=np.float32)]
    action = [np.zeros((3, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]
    paths = [tmp_path / "episode_0.h5", tmp_path / "episode_1.h5"]
    seven_tuple = (obs, action, [3, 2], paths, [None, None], [None, None], None)

    calls = {}

    def fake_load_episodes(data_dir, **kwargs):
        calls["kwargs"] = kwargs
        return seven_tuple

    monkeypatch.setattr(rb_mod, "load_episodes", fake_load_episodes)
    monkeypatch.setattr(rb_mod, "_read_episode_meta", lambda p: {"task_label": "stub"})

    orig_obs_keys = list(export_mod._OBS_KEYS)
    orig_act_keys = list(export_mod._ACTION_KEYS)

    buf = ReplayBuffer.from_hdf5(tmp_path)

    assert buf.n_episodes == 2
    assert buf.n_steps == 5
    assert buf.obs_dim == 4
    assert buf.action_dim == 2
    assert list(buf.episode_ends) == [3, 5]
    ep = buf.get_episode(0)
    assert ep["obs"].shape == (3, 4)
    assert ep["action"].shape == (3, 2)
    assert ep["meta"] == {"task_label": "stub"}
    assert ep["path"] == paths[0]
    # filter kwargs forwarded to load_episodes
    assert calls["kwargs"]["filter_task"] is None
    assert calls["kwargs"]["min_frames"] is None
    # module-level key lists restored after the config-driven patch
    assert list(export_mod._OBS_KEYS) == orig_obs_keys
    assert list(export_mod._ACTION_KEYS) == orig_act_keys


def _write_episode(path: Path, n: int) -> Path:
    """Minimal episode file with the datasets load_episodes reads directly."""
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["task_label"] = "test"
        meta.attrs["num_frames"] = n
        f.create_dataset("arm_qpos", data=np.tile(np.arange(7, dtype=np.float32), (n, 1)))
        f.create_dataset("action_arm_joint", data=np.tile(np.arange(7, dtype=np.float32) + 100, (n, 1)))
    return path


def test_from_hdf5_end_to_end_synthetic_h5(tmp_path):
    """Real load_episodes path: glob, per-key read, 7-tuple unpack, meta read."""
    _write_episode(tmp_path / "episode_a.h5", n=5)
    _write_episode(tmp_path / "episode_b.h5", n=3)

    cfg = DataLoadConfig(obs_keys=[("arm_qpos", 7)], action_keys=[("action_arm_joint", 7)])
    buf = ReplayBuffer.from_hdf5(tmp_path, config=cfg)

    assert buf.n_episodes == 2
    assert buf.n_steps == 8
    assert buf.obs_dim == 7
    assert buf.action_dim == 7
    assert list(buf.episode_ends) == [5, 8]
    assert list(buf.episode_lengths()) == [5, 3]

    ep0 = buf.get_episode(0)
    assert ep0["obs"].shape == (5, 7)
    np.testing.assert_array_equal(ep0["obs"][0], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(ep0["action"][0], np.arange(7, dtype=np.float32) + 100)
    assert ep0["meta"].get("task_label") == "test"
    assert Path(ep0["path"]).name == "episode_a.h5"

    ep1 = buf.get_episode(1)
    assert ep1["obs"].shape == (3, 7)
    assert Path(ep1["path"]).name == "episode_b.h5"
