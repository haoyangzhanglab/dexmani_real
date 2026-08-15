"""A13: per-frame rotation gate tracks the RAW wrist baseline.

After a clamped spike frame, ``_last_wrist_rot`` must store the raw incoming
wrist orientation — not the clamped/gated output — so the recovery frame's
delta is measured against truth rather than a drifted baseline.
"""

from __future__ import annotations

import sys

import numpy as np
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import mat2quat, quat2mat

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.teleop.arm_mapper import ArmWristMapper


def main() -> int:
    mapper = ArmWristMapper(
        pos_scale=1.0,
        rot_scale=1.0,
        max_delta_rot_rad=1.0,       # total-from-reset cap: leave un-clipped
        max_per_frame_rot_rad=0.1,   # per-frame cap: small, to force clamping
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    zero = np.zeros(3)
    mapper.reset(zero, identity, zero, identity)

    # A ~0.5 rad wrist rotation in one frame exceeds the 0.1 rad per-frame cap.
    raw_rot = axangle2mat([0.0, 0.0, 1.0], 0.5)
    wrist_quat = mat2quat(raw_rot)
    mapped = mapper.map(zero, np.asarray(wrist_quat, dtype=np.float64))
    assert mapped is not None, "map() should produce a valid (clamped) target"

    # Baseline must be the RAW orientation, not the clamped output.
    assert mapper._last_wrist_rot is not None
    assert np.allclose(mapper._last_wrist_rot, quat2mat(wrist_quat), atol=1e-12), (
        "_last_wrist_rot drifted away from the raw wrist orientation"
    )
    gated = axangle2mat([0.0, 0.0, 1.0], 0.1)
    assert not np.allclose(mapper._last_wrist_rot, gated, atol=1e-12), (
        "_last_wrist_rot must not track the clamped output"
    )

    print("check_vr_rotation_recovery: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
