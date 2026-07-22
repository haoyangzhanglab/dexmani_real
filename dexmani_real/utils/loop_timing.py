"""Per-stage timing for the 16 Hz teleop main loop.

Usage (one ``tick()`` at the loop top, ``mark()`` after each stage):

    timer = StageTimer(window=50)
    while running:
        line = timer.tick()
        if line:
            print(f"  [timing] {line}")
        limiter.wait()
        timer.mark("wait")
        ...
        timer.mark("ik")

``tick()`` closes the previous iteration (total = time between ticks) and,
every ``window`` iterations, returns an aggregate report line (else None).
Early ``continue`` paths simply skip later marks — those stages accumulate
fewer samples that window, per-stage means stay correct.  "other" is the
total not covered by any mark (prints, keyboard poll, un-marked branches).
"""

from __future__ import annotations

import time


class StageTimer:
    """Aggregate per-stage wall-clock timing over a fixed iteration window."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._stage_sums: dict[str, float] = {}
        self._total_sum = 0.0
        self._total_max = 0.0
        self._n = 0
        self._tick_t: float | None = None
        self._mark_t: float = 0.0

    def tick(self) -> str | None:
        """Close the previous iteration; every ``window`` ticks return a report."""
        now = time.perf_counter()
        line: str | None = None
        if self._tick_t is not None:
            total = now - self._tick_t
            self._total_sum += total
            self._total_max = max(self._total_max, total)
            self._n += 1
            if self._n >= self.window:
                line = self._report()
                self._stage_sums.clear()
                self._total_sum = self._total_max = 0.0
                self._n = 0
        self._tick_t = now
        self._mark_t = now
        return line

    def mark(self, name: str) -> None:
        """Attribute the time since the previous mark (or tick) to ``name``."""
        now = time.perf_counter()
        self._stage_sums[name] = self._stage_sums.get(name, 0.0) + (now - self._mark_t)
        self._mark_t = now

    def _report(self) -> str:
        n = self._n
        marked = sum(self._stage_sums.values())
        parts = [f"total={self._total_sum / n * 1e3:.1f}ms(max {self._total_max * 1e3:.0f})"]
        parts += [f"{k}={v / n * 1e3:.1f}" for k, v in self._stage_sums.items()]
        parts.append(f"other={(self._total_sum - marked) / n * 1e3:.1f}")
        parts.append(f"[n={n}]")
        return " ".join(parts)
