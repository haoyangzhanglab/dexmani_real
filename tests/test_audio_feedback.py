"""Player order, cancellation and child cleanup without launching audio."""

import threading

import pytest

from dexmani_real.teleop import audio_feedback as audio


@pytest.mark.parametrize("failure", [None, "returncode", "communicate", "launch"])
def test_audio_order_and_disable_on_failure(monkeypatch, failure):
    calls, processes = [], []
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)
    monkeypatch.setattr(audio.os.path, "isfile", lambda path: True)

    class Player:
        returncode = 0

        def __init__(self, command, **kw):
            calls.append(command)
            self.killed = self.reaped = False
            self.communications = 0
            processes.append(self)
            if failure == "launch":
                raise OSError("launch failed")

        def poll(self):
            return None if not self.reaped else self.returncode

        def kill(self):
            self.killed = True

        def communicate(self):
            self.communications += 1
            if failure == "communicate" and self.communications == 1:
                raise OSError("read failed")
            self.reaped = True
            self.returncode = 1 if failure == "returncode" else 0
            return b"", b"failure" if failure else b""

    monkeypatch.setattr(audio.subprocess, "Popen", Player)
    player = audio.AudioFeedback()
    try:
        player.play("home")
        player.queue("home_done")
        player.queue("begin")
        assert player.wait_until_idle(2)
        expected = 1 if failure else 3
        assert len(calls) == expected
        if failure:
            monkeypatch.setattr(
                audio.os.path, "isfile", lambda _: pytest.fail("disabled file lookup")
            )
            player.play("begin")
            player.queue("home")
            assert player.wait_until_idle(1)
            assert len(calls) == 1 and player._disabled
        if failure == "communicate":
            assert processes[0].killed and processes[0].reaped
        if not failure:
            assert [c[-1].split("/")[-1] for c in calls] == [
                audio._EVENT_MAP[x] for x in ("home", "home_done", "begin")
            ]
    finally:
        player.close()
        player.close()
    assert not player._worker.is_alive()


@pytest.mark.parametrize("state", ["no_player", "closed"])
def test_audio_inactive_skips_files_and_repeated_warnings(monkeypatch, caplog, state):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None if state == "no_player" else name)
    monkeypatch.setattr(audio.os.path, "isfile", lambda _: pytest.fail("inactive file lookup"))
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: pytest.fail("inactive launch"))
    player = audio.AudioFeedback()
    if state == "no_player":
        assert "No audio player" in caplog.text
    else:
        player.close()
    caplog.clear()
    try:
        for _ in range(3):
            player.play("begin")
            player.queue("unknown")
        assert not caplog.records
    finally:
        player.close()


@pytest.mark.parametrize("operation", ["play", "queue"])
def test_close_during_path_lookup_prevents_enqueue(monkeypatch, operation):
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)
    monkeypatch.setattr(audio.subprocess, "Popen", lambda *a, **kw: pytest.fail("closed launch"))
    player = audio.AudioFeedback()

    def lookup(_):
        player.close()
        return True

    monkeypatch.setattr(audio.os.path, "isfile", lookup)
    try:
        getattr(player, operation)("begin")
        assert player._closed and not player._pending
    finally:
        player.close()


def test_audio_preemption_during_launch_reaps_outside_lock(monkeypatch):
    launched, release = threading.Event(), threading.Event()
    processes = []
    monkeypatch.setattr(audio.shutil, "which", lambda name: name)
    monkeypatch.setattr(audio.os.path, "isfile", lambda path: True)

    class Player:
        returncode = 0

        def __init__(self, *a, **kw):
            self.killed = False
            processes.append(self)
            if len(processes) == 1:
                launched.set()
                assert release.wait(2)

        def kill(self):
            self.killed = True

        def poll(self):
            return None

        def communicate(self):
            assert player._condition.acquire(blocking=False)
            player._condition.release()
            return b"", b""

    monkeypatch.setattr(audio.subprocess, "Popen", Player)
    player = audio.AudioFeedback()
    try:
        player.play("home")
        assert launched.wait(2)
        player.play("emergency")
        release.set()
        assert player.wait_until_idle(2)
        assert len(processes) == 2 and processes[0].killed
    finally:
        release.set()
        player.close()
