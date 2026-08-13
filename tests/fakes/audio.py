"""Headless AudioFeedback double for teleop_loop tests.

The production player shells out to aplay/paplay in a daemon thread.  This
double records prompted events instead, exposing the same surface the loop uses
(``play`` / ``queue`` / ``is_playing``).
"""

from __future__ import annotations


class FakeAudioFeedback:
    """Record prompted events instead of playing audio."""

    last_instance: "FakeAudioFeedback | None" = None

    def __init__(self, audio_dir: str | None = None) -> None:
        del audio_dir
        self.events: list[str] = []
        self.queued: list[str] = []
        FakeAudioFeedback.last_instance = self

    @property
    def is_playing(self) -> bool:
        return False

    def play(self, event: str) -> None:
        self.events.append(event)

    def queue(self, event: str) -> None:
        self.queued.append(event)
