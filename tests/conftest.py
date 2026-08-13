"""Shared pytest fixtures for the headless hardware-loop harness.

The harness runs the production loop bodies in daemon threads against a real
``SharedStorage``, injecting fakes only at the device/operator seams that would
otherwise require hardware: the xArm SDK (via ``sys.modules``), the pynput
keyboard and audio player (via monkeypatch), and the SIGTERM handler (which
``signal.signal`` can only install on the main thread).
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time

import pytest

from dexmani_real.shm.shared_storage import SharedStorage

from tests.fakes.audio import FakeAudioFeedback
from tests.fakes.keyboard import FakeKeyboardHandler


@pytest.fixture
def shared():
    """A fresh, fully-allocated SharedStorage per test."""
    storage = SharedStorage.create(
        prefix=f"harness_{mp.current_process().pid}_{time.monotonic_ns()}"
    )
    yield storage
    storage.close()


@pytest.fixture
def teleop_fakes(monkeypatch):
    """Patch the teleop loop's operator/audio/SIGTERM seams to headless doubles."""
    import dexmani_real.teleop.loop as loop_module

    FakeKeyboardHandler.last_instance = None
    FakeAudioFeedback.last_instance = None
    monkeypatch.setattr(loop_module, "KeyboardHandler", FakeKeyboardHandler)
    monkeypatch.setattr(loop_module, "AudioFeedback", FakeAudioFeedback)
    monkeypatch.setattr(loop_module, "_END_AUDIO_GRACE_S", 0.0)
    # teleop_loop installs a SIGTERM handler; signal.signal only works on the
    # main thread.  The loop exits via is_running/quit_requested, so the
    # registration is a no-op in the harness.
    monkeypatch.setattr(signal, "signal", lambda *args, **kwargs: None)
    return FakeKeyboardHandler


@pytest.fixture
def arm_fakes(monkeypatch):
    """Install the fake xArm SDK and yield the FakeXArmAPI class."""
    from tests.fakes.xarm_sdk import FakeXArmAPI, install_xarm_fake, remove_xarm_fake

    install_xarm_fake()
    yield FakeXArmAPI
    remove_xarm_fake()
