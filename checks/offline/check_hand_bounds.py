"""A12: shared hand preflight validates operational + mechanical bounds.

``validate_hand_command_bounds`` is the single reject-whole boundary for every
coupled hand path.  It must accept an in-envelope command (returning a copy,
never clipping), and raise on an operational-limit, mechanical-envelope, or
limit-configuration violation.
"""

from __future__ import annotations

import sys

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.policy.safety import validate_hand_command_bounds


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

    # ── Accept, return a copy, never clip ──────────────────────────────
    out = validate_hand_command_bounds(mid, op_lower, op_upper, rated_lower, rated_upper)
    np.testing.assert_allclose(out, mid)
    assert out is not mid, "must return a copy, not the caller's array"

    # ── Operational limit violation (isolated from the mechanical check) ─
    # The operational box is conservative only on the min side of some joints
    # (op_max == mech_max for every joint), so a command above op_upper would
    # also trip the mechanical check.  Command below a conservative joint's
    # operational *min* but inside the rated mechanical envelope instead, so
    # this case exercises the operational check alone.
    j = int(np.argmax(op_lower - rated_lower))
    assert op_lower[j] > rated_lower[j] + 1e-9, "need a conservative-min joint"
    op_violation = mid.copy()
    op_violation[j] = rated_lower[j]  # inside mechanical, below operational min
    _raises(
        lambda: validate_hand_command_bounds(
            op_violation, op_lower, op_upper, rated_lower, rated_upper
        ),
        "operational limit violation",
    )

    # ── Mechanical envelope exceeding the rated device envelope ────────
    too_wide = rated_upper.copy()
    too_wide[0] = rated_upper[0] + 1.0
    _raises(
        lambda: validate_hand_command_bounds(
            mid, op_lower, op_upper, rated_lower, too_wide
        ),
        "mechanical > rated",
    )

    # ── Operational limits outside the mechanical envelope ─────────────
    narrow_mech = rated_upper.copy()
    narrow_mech[0] = op_upper[0] - 0.5
    _raises(
        lambda: validate_hand_command_bounds(
            mid, op_lower, op_upper, rated_lower, narrow_mech
        ),
        "operational > mechanical",
    )

    # ── Non-finite / malformed command ─────────────────────────────────
    nan_cmd = mid.copy()
    nan_cmd[3] = np.nan
    _raises(
        lambda: validate_hand_command_bounds(
            nan_cmd, op_lower, op_upper, rated_lower, rated_upper
        ),
        "non-finite command",
    )

    print("check_hand_bounds: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
