"""Conservative device-clock to host-monotonic mapping."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ClockMapping:
    source_monotonic_ns: int
    generation: int
    duplicate: bool
    frame_gap: int
    clock_reset: bool
    # Delay above the trailing lower-envelope estimate. This is a relative
    # delivery-delay diagnostic, not an SDK queue depth or absolute latency.
    backlog_ns: int


class DeviceClockMapper:
    """Map a device clock onto the host monotonic clock.

    The L515 depth timestamp is not reliably synchronized to host monotonic
    (``global_time_enabled`` has no confirmed effect on the L515), so the
    device clock can drift relative to the host.  The offset is therefore
    estimated as the minimum ``host_receive - device`` over a bounded trailing
    window instead of the all-time minimum.
    The window minimum still cannot move a sample into the future and remains
    insensitive to a transient USB scheduling delay (an inflated offset), but
    it is free to rise and track slow drift.  An all-time minimum would pin
    the offset at its initial value and let ``source_monotonic_ns`` lag the
    host unboundedly as the device clock drifts.  A device rollback or a large
    disagreement between host/device deltas starts a new generation and clears
    the window.
    """

    def __init__(
        self, *, reset_jump_ns: int = 2_000_000_000, window_ns: int = 2_000_000_000
    ) -> None:
        if reset_jump_ns <= 0:
            raise ValueError("reset_jump_ns must be positive")
        if window_ns <= 0:
            raise ValueError("window_ns must be positive")
        self.reset_jump_ns = int(reset_jump_ns)
        self.window_ns = int(window_ns)
        self.generation = 0
        # (host_ns, offset_ns) pairs, oldest first, bounded to window_ns.
        self._offset_window: deque[tuple[int, int]] = deque()
        self._last_device_ns: int | None = None
        self._last_host_ns: int | None = None
        self._last_frame_number: int | None = None

    def reset(self) -> None:
        self.generation += 1
        self._offset_window.clear()
        self._last_device_ns = None
        self._last_host_ns = None
        self._last_frame_number = None

    def map(
        self, *, device_time_s: float, host_receive_ns: int, frame_number: int
    ) -> ClockMapping:
        device_ns = int(round(float(device_time_s) * 1e9))
        host_ns = int(host_receive_ns)
        frame = int(frame_number)
        if device_ns < 0 or host_ns <= 0 or frame < 0:
            raise ValueError(
                "device/host timestamps and frame number must be non-negative"
            )

        duplicate = (
            self._last_frame_number == frame and self._last_device_ns == device_ns
        )
        frame_gap = 0
        clock_reset = False
        if self._last_device_ns is not None and self._last_host_ns is not None:
            device_delta = device_ns - self._last_device_ns
            host_delta = host_ns - self._last_host_ns
            rollback = device_delta < 0 or (
                self._last_frame_number is not None
                and frame < self._last_frame_number
                and not duplicate
            )
            jump = abs(device_delta - host_delta) > self.reset_jump_ns
            if rollback or jump:
                self.reset()
                clock_reset = True
            elif (
                self._last_frame_number is not None
                and frame > self._last_frame_number + 1
            ):
                frame_gap = frame - self._last_frame_number - 1

        offset = host_ns - device_ns
        # Use the trailing-window minimum to ignore transient USB-delivery spikes.
        while (
            self._offset_window and host_ns - self._offset_window[0][0] > self.window_ns
        ):
            self._offset_window.popleft()
        self._offset_window.append((host_ns, offset))
        offset_lower_bound = min(off for _, off in self._offset_window)
        source_ns = device_ns + offset_lower_bound
        source_ns = min(source_ns, host_ns)

        self._last_device_ns = device_ns
        self._last_host_ns = host_ns
        self._last_frame_number = frame
        return ClockMapping(
            source_monotonic_ns=source_ns,
            generation=self.generation,
            duplicate=duplicate,
            frame_gap=frame_gap,
            clock_reset=clock_reset,
            backlog_ns=max(0, host_ns - source_ns),
        )
