"""Consumers of the nominal grid rate (meta control_hz → fps → 50 fallback).

Covers design principle 5 of the 16Hz migration: consumers prefer the nominal
grid rate (/meta control_hz, schema v7) over the stop-time achieved fps, and
clamp implausible values before they drive real hardware.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def replay_mod():
    spec = importlib.util.spec_from_file_location("replay_traj", REPO / "examples" / "real" / "replay_traj.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_traj"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_episode(path: Path, attrs: dict, n: int = 8) -> Path:
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        for k, v in attrs.items():
            meta.attrs[k] = v
        f.create_dataset("action_arm_joint", data=np.zeros((n, 7)))
        f.create_dataset("arm_qpos", data=np.zeros((n, 7)))
    return path


def test_replay_prefers_control_hz_over_fps(tmp_path, replay_mod):
    """v7 file: stop-time fps (diluted/inflated) must not win over control_hz."""
    p = _write_episode(tmp_path / "v7.h5", {"control_hz": 16.0, "fps": 793.2})
    traj = replay_mod.load_trajectory(str(p))
    assert traj.fps == 16.0


def test_replay_falls_back_to_fps_then_default(tmp_path, replay_mod):
    p = _write_episode(tmp_path / "v6.h5", {"fps": 50.0})
    assert replay_mod.load_trajectory(str(p)).fps == 50.0
    p2 = _write_episode(tmp_path / "bare.h5", {})
    assert replay_mod.load_trajectory(str(p2)).fps == 50.0


def test_replay_clamps_implausible_rates(tmp_path, replay_mod):
    """replay_hz drives the real arm — fps=0 or a paused-episode fps must not."""
    for bad in (0.0, 0.4, 793.2):
        p = _write_episode(tmp_path / f"bad_{bad}.h5", {"fps": bad})
        assert replay_mod.load_trajectory(str(p)).fps == 50.0


def test_export_episode_control_rates(tmp_path):
    from dexmani_real.tools.export_hdf5_to_zarr import _episode_control_rates

    p16 = _write_episode(tmp_path / "episode_a.h5", {"control_hz": 16.0, "fps": 15.7})
    p50 = _write_episode(tmp_path / "episode_b.h5", {"fps": 50.0})
    assert _episode_control_rates([p16, p50]) == [16.0, 50.0]
