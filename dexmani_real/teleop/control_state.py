"""Small, deterministic state transitions used by the teleoperation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.teleop.hand_control import HandRetargetObservationCache


@dataclass
class TeleopLoopState:
    """Mutable state owned by one teleoperation coordinator run."""

    hand_retargeter: Any = None
    prev_qpos_cmd: np.ndarray | None = None
    prev_hand_qpos: np.ndarray | None = None
    ema_prev_pos: np.ndarray | None = None
    ema_prev_quat: np.ndarray | None = None
    hand_ramp_start: np.ndarray | None = None
    hand_ramp_step: int = 0
    hand_retarget_cache: HandRetargetObservationCache = field(
        default_factory=HandRetargetObservationCache
    )
    hand_available: bool = False
    hand_disconnected_at_s: float | None = None
    teleop_active: bool = False
    recording_active: bool = False
    begin_audio_gate_deadline_s: float | None = None
    ignore_begin_audio_until_silent: bool = False
    quit_pending: bool = False
    quit_after_recording: bool = False
    quit_recording_deadline_s: float = 0.0
    post_teleop_deadline_s: float = 0.0
    arm_feedback_error_count: int = 0
    consecutive_ik_hold_frames: int = 0
    ik_hold_started_s: float = 0.0
    last_target_eef_pos: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    last_target_eef_rot6d: np.ndarray = field(
        default_factory=lambda: np.full(6, np.nan)
    )
    sigterm_requested: bool = False


class CoordinatorDirective(str, Enum):
    """Control-flow decision returned by one coordinator state handler."""

    NORMAL = "normal"
    CONTINUE = "continue"
    REANCHOR_GRID = "reanchor_grid"
    BREAK = "break"


@dataclass
class CommandQuiescence:
    """Track one command-silent pause and its feedback freshness boundary.

    This object never carries an actuator target. The owner advances
    ``run_generation`` on the first entry, preserves that boundary across
    repeated pause observations, may reclassify the current reason for an
    explicit operator transition without a second generation advance, and
    replaces the boundary when an explicit BEGIN opens a distinct run. The
    boundary is used only to decide when feedback is safe to re-anchor from.
    """

    reason: str | None = None
    entered_monotonic_ns: int = 0

    @property
    def active(self) -> bool:
        return self.reason is not None

    def enter(self, reason: str, *, entered_monotonic_ns: int) -> bool:
        """Enter quiescence; preserve the first reason/time on repeated calls."""
        self._validate_reason(reason)
        entered_ns = int(entered_monotonic_ns)
        if entered_ns <= 0:
            raise ValueError("quiescence entry time must be positive")
        if self.active:
            return False
        self.reason = reason
        self.entered_monotonic_ns = entered_ns
        return True

    def relabel(self, reason: str) -> None:
        """Reclassify an active pause without moving its freshness boundary.

        An explicit C can turn an automatic gate into a user-resumable pause;
        STOP, DISCARD, maximum duration, and QUIT can make that same pause
        non-resumable. In both cases its generation is already invalidated.
        """
        self._validate_reason(reason)
        if not self.active:
            raise RuntimeError("cannot relabel inactive command quiescence")
        self.reason = reason

    def feedback_is_newer(
        self,
        *,
        arm_source_monotonic_ns: int,
        vr_receive_monotonic_ns: int,
        hand_source_monotonic_ns: int | None,
    ) -> bool:
        """Require every enabled feedback source to strictly postdate entry."""
        if not self.active or self.entered_monotonic_ns <= 0:
            return False
        boundary_ns = self.entered_monotonic_ns
        hand_is_newer = (
            hand_source_monotonic_ns is None
            or int(hand_source_monotonic_ns) > boundary_ns
        )
        return bool(
            int(arm_source_monotonic_ns) > boundary_ns
            and int(vr_receive_monotonic_ns) > boundary_ns
            and hand_is_newer
        )

    def clear(self) -> tuple[str | None, int]:
        """Leave quiescence and return the reason and original entry time."""
        reason = self.reason
        entered_ns = self.entered_monotonic_ns
        self.reason = None
        self.entered_monotonic_ns = 0
        return reason, entered_ns

    @staticmethod
    def _validate_reason(reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quiescence reason must be a non-empty string")
