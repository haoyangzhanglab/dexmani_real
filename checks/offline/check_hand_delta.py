"""A12: shared hand preflight validates operational + mechanical + delta.

``validate_hand_command_delta`` is the single reject-whole boundary for every
coupled hand path.  It must accept an in-envelope command (returning a copy,
never clipping), and raise on an operational-limit, mechanical-envelope,
command-delta, or limit-configuration violation.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.safety import validate_hand_command_delta


def _raises(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {label}")


def main() -> int:
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
    op_lower = np.asarray(hand_defaults.qpos_min_rad, dtype=np.float64)
    op_upper = np.asarray(hand_defaults.qpos_max_rad, dtype=np.float64)
    mid = (op_lower + op_upper) / 2.0
    prev = mid.copy()

    # ── Accept, return a copy, never clip ──────────────────────────────
    out = validate_hand_command_delta(
        mid, prev, op_lower, op_upper, rated_lower, rated_upper, None
    )
    np.testing.assert_allclose(out, mid)
    assert out is not mid, "must return a copy, not the caller's array"
    out = validate_hand_command_delta(
        mid, prev, op_lower, op_upper, rated_lower, rated_upper, 0.20
    )
    np.testing.assert_allclose(out, mid)

    # ── Operational limit violation ────────────────────────────────────
    op_violation = mid.copy()
    op_violation[0] = op_upper[0] + 0.1
    _raises(
        lambda: validate_hand_command_delta(
            op_violation, prev, op_lower, op_upper, rated_lower, rated_upper, None
        ),
        "operational limit violation",
    )

    # ── Command-to-command delta violation ─────────────────────────────
    delta_violation = mid.copy()
    delta_violation[0] = mid[0] + 0.5  # > 0.20, still inside operational box
    _raises(
        lambda: validate_hand_command_delta(
            delta_violation, prev, op_lower, op_upper, rated_lower, rated_upper, 0.20
        ),
        "command delta violation",
    )

    # ── Mechanical envelope exceeding the rated device envelope ────────
    too_wide = rated_upper.copy()
    too_wide[0] = rated_upper[0] + 1.0
    _raises(
        lambda: validate_hand_command_delta(
            mid, prev, op_lower, op_upper, rated_lower, too_wide, None
        ),
        "mechanical > rated",
    )

    # ── Operational limits outside the mechanical envelope ─────────────
    narrow_mech = rated_upper.copy()
    narrow_mech[0] = op_upper[0] - 0.5
    _raises(
        lambda: validate_hand_command_delta(
            mid, prev, op_lower, op_upper, rated_lower, narrow_mech, None
        ),
        "operational > mechanical",
    )

    # ── Delta requested without a previous accepted command ────────────
    _raises(
        lambda: validate_hand_command_delta(
            mid, None, op_lower, op_upper, rated_lower, rated_upper, 0.20
        ),
        "delta without previous",
    )

    # ── Non-finite / malformed command ─────────────────────────────────
    nan_cmd = mid.copy()
    nan_cmd[3] = np.nan
    _raises(
        lambda: validate_hand_command_delta(
            nan_cmd, prev, op_lower, op_upper, rated_lower, rated_upper, None
        ),
        "non-finite command",
    )

    print("check_hand_delta: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
