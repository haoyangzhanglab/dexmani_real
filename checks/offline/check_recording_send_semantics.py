"""A17: recording provenance clears send-event fields on a synthetic hold.

When a grid tick emits no action (``action_candidate is None``), the recorded
observation must clear ``action_queued``, ``action_id`` and every
``action_*_monotonic_ns`` field, so a command-silent pause is never replayed
as an actuator send.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.teleop.episode_samples import _recording_provenance


def _context() -> tuple[SimpleNamespace, dict, dict]:
    shared = SimpleNamespace(arm_state_ring=None, hand_state_ring=None)
    vr_frame = {"local_recv_ns": 0, "ring_sequence": 0, "publish_monotonic_ns": 0}
    cam = {
        "source_monotonic_ns": 0,
        "receive_monotonic_ns": 0,
        "publish_monotonic_ns": 0,
        "camera_fresh": False,
        "ring_sequence": 0,
    }
    return shared, vr_frame, cam


def main() -> int:
    shared, vr_frame, cam = _context()

    cand = ActionCandidate(
        observation_id=100,
        run_generation=1,
        created_monotonic_ns=1000,
        target_monotonic_ns=2000,
        valid_until_monotonic_ns=3000,
        action_id=42,
        arm_qpos=np.zeros(7, dtype=np.float64),
    )
    prov = _recording_provenance(shared, None, None, None, vr_frame, cam, action_candidate=cand)
    assert prov["action_queued"] is True
    assert prov["action_id"] == 42
    assert prov["action_created_monotonic_ns"] == 1000
    assert prov["action_target_monotonic_ns"] == 2000
    assert prov["action_valid_until_monotonic_ns"] == 3000

    # Synthetic hold: no action candidate → every send-event field cleared.
    prov = _recording_provenance(shared, None, None, None, vr_frame, cam, action_candidate=None)
    assert prov["action_queued"] is False
    assert prov["action_id"] == 0
    assert prov["action_created_monotonic_ns"] == 0
    assert prov["action_target_monotonic_ns"] == 0
    assert prov["action_valid_until_monotonic_ns"] == 0

    print("check_recording_send_semantics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
