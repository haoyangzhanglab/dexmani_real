"""H1: the remaining hand health gate is the shared predicate.

``supervisor._hand_feedback_issue`` must produce the same fail-closed verdict as
``validate_hand_feedback`` (the single source of truth in
``utils/hand_health.py``): a stale, future, disconnected, hardware-errored,
invalid, or I/O-unhealthy frame is reported identically in the pre-flight
health summary.  The coordinator hand-reference seed was removed with the hand
delta clip (D3); the publication boundary reaches the same predicate through
``_hand_feedback_snapshot``.

The fixed rejection strings are locked verbatim; the timestamp branches are
compared by keyword because their numeric age suffix depends on
``time.monotonic_ns`` and is not stable across two independent calls.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import make_hand_state_frame

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.runtime.supervisor import _hand_feedback_issue
from dexmani_real.utils.hand_health import validate_hand_feedback


def _mid(low: tuple[float, ...], high: tuple[float, ...]) -> np.ndarray:
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    return (lo + hi) / 2.0


def _predicate_kwargs(frame: np.ndarray, *, now_ns: int, max_age_s: float) -> dict:
    return dict(
        connected=bool(frame["connected"][0]),
        error_state=bool(frame["error_state"][0]),
        state_valid=bool(frame["state_valid"][0]),
        send_healthy=bool(frame["send_healthy"][0]),
        read_healthy=bool(frame["read_healthy"][0]),
        source_monotonic_ns=int(frame["source_monotonic_ns"][0]),
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=np.asarray(frame["qpos"][0], dtype=np.float64),
    )


def main() -> int:
    hand_mid = _mid(hand_defaults.qpos_min_rad, hand_defaults.qpos_max_rad)
    max_age_s = 1.0
    now_ns = time.monotonic_ns()

    # ── golden strings (fixed branches, verbatim) ──
    base = dict(
        connected=True,
        error_state=False,
        state_valid=True,
        send_healthy=True,
        read_healthy=True,
        source_monotonic_ns=now_ns,
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=hand_mid,
    )
    golden = [
        (dict(base, connected=False), "hand disconnected"),
        (dict(base, error_state=True), "hand reported a hardware error"),
        (dict(base, state_valid=False), "hand state marked invalid"),
        (dict(base, send_healthy=False), "hand command/state I/O is unhealthy"),
        (dict(base, source_monotonic_ns=0), "hand state has no source timestamp"),
    ]
    for kwargs, expected in golden:
        got = validate_hand_feedback(**kwargs)
        assert got == expected, (got, expected)

    # One mutated signal each; the last is healthy.
    variants = {
        "fresh": make_hand_state_frame(hand_mid),
        "stale": make_hand_state_frame(hand_mid, source_monotonic_ns=now_ns - int(2.0 * 1e9)),
        "future": make_hand_state_frame(hand_mid, source_monotonic_ns=now_ns + int(5.0 * 1e9)),
        "disconnected": make_hand_state_frame(hand_mid, connected=0),
        "invalid": make_hand_state_frame(hand_mid, state_valid=0),
        "error": make_hand_state_frame(hand_mid, error_state=1),
        "io": make_hand_state_frame(hand_mid, send_healthy=0),
    }

    # ── supervisor helper delegates to the same predicate ──
    for name, frame in variants.items():
        expected = validate_hand_feedback(
            **_predicate_kwargs(frame, now_ns=time.monotonic_ns(), max_age_s=max_age_s)
        )
        got = _hand_feedback_issue(frame, max_age_s=max_age_s)
        if name in ("stale", "future"):
            # Numeric age suffix is not stable across calls; compare the verdict.
            assert (got is None) == (expected is None), (name, got, expected)
            if got is not None:
                assert name in got and name in expected, (name, got, expected)
        else:
            assert got == expected, (name, got, expected)

    print("check_hand_health_unification: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
