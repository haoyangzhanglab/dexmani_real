"""Offline tests for causal observation admission and the run budget (task T4).

Covers the acceptance matrix:
* V11 old-but-causal frames are admitted; future frames, ordering violations,
  wrong camera generation, and unavailable required history are not; warm-up
  edge repeat keeps the ORIGINAL source time; low-rate reuse is normal.
* V12 no generic age/skew/grid-lag gate remains in the observation path;
  camera health separates finite-but-slow delivery (admitted) from an invalid
  clock (rejected).
* V14 the parent supervisor enforces the run budget with a generation-checked
  revocation, so a blocking predict is bounded without a loop-heartbeat veto
  and an expired timeout can never revoke a newer trial.

These exercise the real admission functions against synthetic ring records;
they never open hardware, an SDK, or a camera.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

import numpy as np

# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    from dexmani_real.deployment.observation import (
        _pointcloud_frame_from_record,
        _read_state_history,
        _rgb_identity_from_header,
        _select_camera_control_grid,
        _select_history_indices,
    )
    from dexmani_real.ipc.schema import ARM_STATE_DTYPE, CAMERA_FRAME_HEADER_DTYPE
    from dexmani_real.runtime.safety import (
        SafetyState,
        revoke_motion_if_generation,
    )
    from dexmani_real.sensor.camera.worker import CameraHealth

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    _IMPORT_ERROR = exc

_MS = 1_000_000
_S = 1_000_000_000


class _FakeRing:
    """get_last_k over pre-built (record, publish_ns, sequence) frames."""

    def __init__(self, frames, maxlen: int = 16) -> None:
        self._frames = list(frames)
        self.maxlen = maxlen

    def get_last_k(self, k: int):
        return self._frames[-k:]


def _arm_frame(source_ns: int, publish_ns: int, sequence: int, qpos: float = 0.1):
    record = np.zeros(1, dtype=ARM_STATE_DTYPE)
    record["state_valid"] = 1
    record["connected"] = 1
    record["source_monotonic_ns"] = source_ns
    record["publish_monotonic_ns"] = publish_ns
    record["qpos"] = qpos
    return (record, publish_ns, sequence)


def _camera_header(
    *,
    source_ns: int,
    receive_ns: int,
    publish_ns: int,
    generation: int = 3,
    health: int = int(CameraHealth.OK),
):
    header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
    header["source_monotonic_ns"] = source_ns
    header["receive_monotonic_ns"] = receive_ns
    header["publish_monotonic_ns"] = publish_ns
    header["camera_generation"] = generation
    header["camera_health"] = health
    return header


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class StateHistoryAdmissionTest(unittest.TestCase):
    def test_old_but_causal_frames_are_admitted(self):
        """A frame far older than any removed age gate stays a valid slot."""
        anchor = 100 * _S
        frames = [
            # Source is 30 s old — previously dropped by generic age gates.
            _arm_frame(70 * _S, 70 * _S + 5 * _MS, 1),
            _arm_frame(anchor - 10 * _MS, anchor - 5 * _MS, 2),
        ]
        window = _read_state_history(
            _FakeRing(frames),
            history_len=8,
            anchor_ns=anchor,
            values_field="qpos",
            required_true_fields=("state_valid",),
            not_before_ns=1,
        )
        self.assertIsNotNone(window)
        self.assertEqual(window.values.shape[0], 2)
        self.assertEqual(int(window.source_monotonic_ns[0]), 70 * _S)

    def test_future_and_order_violations_are_rejected(self):
        anchor = 100 * _S
        frames = [
            _arm_frame(anchor + 1, anchor + 2, 1),  # future source
            _arm_frame(anchor - 1, anchor + 1, 2),  # commit after the anchor
            _arm_frame(0, 5, 3),  # missing source timestamp
            _arm_frame(anchor - 9 * _MS, anchor - 8 * _MS, 4),  # valid
        ]
        window = _read_state_history(
            _FakeRing(frames),
            history_len=8,
            anchor_ns=anchor,
            values_field="qpos",
            required_true_fields=("state_valid",),
        )
        self.assertIsNotNone(window)
        self.assertEqual(list(window.source_sequence), [4])

    def test_not_before_run_start(self):
        anchor = 100 * _S
        frames = [
            _arm_frame(50 * _S, 50 * _S, 1),  # before run start
            _arm_frame(90 * _S, 90 * _S, 2),
        ]
        window = _read_state_history(
            _FakeRing(frames),
            history_len=8,
            anchor_ns=anchor,
            values_field="qpos",
            not_before_ns=60 * _S,
        )
        self.assertEqual(list(window.source_sequence), [2])


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class RingCommitAdmissionTest(unittest.TestCase):
    def test_real_commit_and_payload_order_for_arm_and_hand(self):
        from dexmani_real.deployment.observation import _read_hand_history
        from dexmani_real.ipc.schema import HAND_STATE_DTYPE
        for name, dtype in (("arm", ARM_STATE_DTYPE), ("hand", HAND_STATE_DTYPE)):
            for publish, commit, anchor, admitted in (
                (95, 105, 100, False), (95, 99, 100, True),
                (98, 96, 100, False), (89, 99, 100, False),
                (95, 99, 10000, True), (0, 99, 100, True),
            ):
                with self.subTest(name=name, publish=publish, commit=commit, anchor=anchor):
                    record = np.zeros(1, dtype=dtype)
                    record["state_valid"] = 1
                    record["source_monotonic_ns"] = 90
                    record["publish_monotonic_ns"] = publish
                    ring = _FakeRing([(record, commit, 1)])
                    kwargs = dict(history_len=1, anchor_ns=anchor)
                    if name == "arm":
                        window = _read_state_history(ring, values_field="qpos", **kwargs)
                    else:
                        window = _read_hand_history(ring, **kwargs)
                    self.assertEqual(window is not None, admitted)
                    if admitted:
                        self.assertEqual(int(window.publish_monotonic_ns[0]), publish or commit)


class GridSelectionTest(unittest.TestCase):
    def test_low_rate_reuse_and_newest_before_reference(self):
        run_started = 10 * _S
        refs = np.array([10 * _S, 10 * _S + 200 * _MS, 11 * _S], dtype=np.int64)
        sources = np.array([10 * _S, 10 * _S + 500 * _MS], dtype=np.uint64)
        valid = np.ones(2, dtype=np.uint8)
        indices = _select_history_indices(
            sources, valid, refs, run_started_ns=run_started
        )
        # No skew gate: the 10.0 s source legitimately serves the 10.2 s
        # reference (a 200 ms gap the removed 0.1 s skew bound would have
        # rejected); newest-source <= reference wins per slot.
        self.assertEqual(list(indices), [0, 0, 1])

    def test_warmup_edge_repeat_keeps_original_source_time(self):
        run_started = 10 * _S
        # Leading references at/before run start repeat the oldest real frame.
        refs = np.array(
            [9 * _S, 9 * _S + 62_500_000, 10 * _S + 200 * _MS], dtype=np.int64
        )
        sources = np.array([10 * _S + 100 * _MS, 10 * _S + 150 * _MS], dtype=np.uint64)
        valid = np.ones(2, dtype=np.uint8)
        indices = _select_history_indices(
            sources, valid, refs, run_started_ns=run_started
        )
        self.assertEqual(list(indices), [0, 0, 1])
        # The repeated slot keeps the ORIGINAL source time — nothing fabricates
        # a run-start frame.
        self.assertEqual(int(sources[indices[0]]), 10 * _S + 100 * _MS)

    def test_missing_required_history_waits(self):
        refs = np.array([10 * _S, 11 * _S], dtype=np.int64)
        sources = np.array([10 * _S + 500 * _MS], dtype=np.uint64)
        valid = np.ones(1, dtype=np.uint8)
        # The 10 s reference is after run start and the only source (10.5 s)
        # is in its future: an explicit wait (None), never a future or
        # substituted frame.
        self.assertIsNone(
            _select_history_indices(
                sources, valid, refs, run_started_ns=9 * _S
            )
        )

    def test_camera_grid_reuse_without_lag_bound(self):
        frames = [
            SimpleNamespace(source_monotonic_ns=10 * _S, camera_generation=3),
            SimpleNamespace(source_monotonic_ns=10 * _S + 300 * _MS, camera_generation=3),
        ]
        # A reference 5 s after the newest frame still reuses it: the removed
        # grid-lag bound no longer rejects slow-but-causal visual sources.
        refs = np.array([10 * _S, 15 * _S], dtype=np.int64)
        selected, logical = _select_camera_control_grid(
            tuple(frames), run_started_ns=9 * _S, reference_ns=refs
        )
        self.assertEqual(len(selected), 2)
        self.assertIs(selected[1], frames[1])
        self.assertEqual(logical, 15 * _S)


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class CameraIdentityAdmissionTest(unittest.TestCase):
    def test_slow_but_valid_delivery_is_admitted(self):
        anchor = 100 * _S
        for health in (
            int(CameraHealth.OK),
            int(CameraHealth.FRAME_GAP),
            int(CameraHealth.DELIVERY_DELAY),
        ):
            identity = _rgb_identity_from_header(
                _camera_header(
                    # Source is 30 s old: finite-but-slow, still causal.
                    source_ns=70 * _S,
                    receive_ns=70 * _S + 10 * _MS,
                    publish_ns=70 * _S + 20 * _MS,
                    health=health,
                ),
                70 * _S + 25 * _MS,
                5,
                anchor_ns=anchor,
                not_before_ns=1,
            )
            self.assertIsNotNone(identity, f"health={health} must stay usable")
            self.assertEqual(identity.source_camera_sequence, 5)

    def test_invalid_clock_and_ordering_are_rejected(self):
        anchor = 100 * _S
        cases = {
            "clock_reset": _camera_header(
                source_ns=99 * _S,
                receive_ns=99 * _S,
                publish_ns=99 * _S,
                health=int(CameraHealth.CLOCK_RESET),
            ),
            "duplicate": _camera_header(
                source_ns=99 * _S,
                receive_ns=99 * _S,
                publish_ns=99 * _S,
                health=int(CameraHealth.DUPLICATE),
            ),
            "future_source": _camera_header(
                source_ns=anchor + 1, receive_ns=anchor + 2, publish_ns=anchor + 3
            ),
            "receive_before_source": _camera_header(
                source_ns=90 * _S, receive_ns=89 * _S, publish_ns=90 * _S
            ),
            "commit_after_anchor": _camera_header(
                source_ns=90 * _S, receive_ns=90 * _S, publish_ns=90 * _S
            ),
        }
        for name, header in cases.items():
            ring_publish = anchor + 1 if name == "commit_after_anchor" else 95 * _S
            with self.subTest(case=name):
                self.assertIsNone(
                    _rgb_identity_from_header(
                        header,
                        ring_publish,
                        7,
                        anchor_ns=anchor,
                        not_before_ns=1,
                    )
                )

    def test_pointcloud_provenance_chain_without_age_drop(self):
        from dexmani_real.ipc.schema import make_pointcloud_frame_dtype

        anchor = 100 * _S
        record = np.zeros(1, dtype=make_pointcloud_frame_dtype(1024))[0]
        record["source_camera_sequence"] = 9
        record["source_monotonic_ns"] = 40 * _S  # very old but fully causal
        record["camera_publish_monotonic_ns"] = 40 * _S + _MS
        record["publish_monotonic_ns"] = 40 * _S + 2 * _MS
        record["camera_generation"] = 2
        cloud = np.zeros((1024, 6), dtype=np.float32)
        cloud[:, 3:] = 0.5
        record["point_cloud"] = cloud
        frame = _pointcloud_frame_from_record(
            record,
            40 * _S + 3 * _MS,
            anchor_ns=anchor,
            num_points=1024,
            not_before_ns=1,
        )
        self.assertIsNotNone(frame)
        self.assertEqual(frame.source_monotonic_ns, 40 * _S)
        # An ordering violation still never passes.
        record["camera_publish_monotonic_ns"] = 39 * _S
        self.assertIsNone(
            _pointcloud_frame_from_record(
                record,
                40 * _S + 3 * _MS,
                anchor_ns=anchor,
                num_points=1024,
                not_before_ns=1,
            )
        )


class _BudgetShared:
    """Minimal motion-lock shared state for generation-checked revocation."""

    def __init__(self, generation: int, state: SafetyState) -> None:
        import threading

        self.motion_lock = threading.RLock()
        self.run_generation = SimpleNamespace(value=generation)
        self.run_generation_base_sequence = SimpleNamespace(value=0)
        self.arm_cmd_consumed_sequence = SimpleNamespace(value=-1)
        self.hand_cmd_consumed_sequence = SimpleNamespace(value=-1)
        self.is_running = SimpleNamespace(value=True)
        self.error_state = SimpleNamespace(value=False)
        self.estop_request = SimpleNamespace(value=False)
        self.safety_state = SimpleNamespace(value=int(state))
        self.run_started_monotonic_ns = SimpleNamespace(value=0)
        self.run_started_generation = SimpleNamespace(value=0)
        self.run_ended_generation = SimpleNamespace(value=0)
        self.run_ended_started_monotonic_ns = SimpleNamespace(value=0)
        self.run_ended_monotonic_ns = SimpleNamespace(value=0)
        self.run_ended_reason = SimpleNamespace(value=0)

        self.stop_request = SimpleNamespace(value=0)
        self.coupled_cmd_ring = SimpleNamespace(latest_sequence=0)


@unittest.skipIf(_IMPORT_ERROR is not None, f"dependencies unavailable: {_IMPORT_ERROR}")
class RunBudgetRevocationTest(unittest.TestCase):
    """V14: generation-checked parent revocation of a blocking-predict epoch."""

    def test_matching_generation_revokes(self):
        shared = _BudgetShared(generation=5, state=SafetyState.RUNNING)
        self.assertTrue(revoke_motion_if_generation(shared, 5))
        self.assertEqual(int(shared.safety_state.value), int(SafetyState.ARMED))
        self.assertEqual(int(shared.run_generation.value), 6)

    def test_expired_timeout_never_revokes_a_newer_trial(self):
        shared = _BudgetShared(generation=7, state=SafetyState.RUNNING)
        # The snapshot generation (6) belongs to an already-finished trial.
        self.assertFalse(revoke_motion_if_generation(shared, 6))
        self.assertEqual(int(shared.safety_state.value), int(SafetyState.RUNNING))
        self.assertEqual(int(shared.run_generation.value), 7)

    def test_supervisor_loop_enforces_the_run_budget(self):
        """The real supervisor loop revokes an expired RUNNING epoch.

        Driven through ``run_supervisor`` itself, not by re-implementing its
        branch: the budget must fire while the control owner is blocked, and
        the epoch must be fenced exactly once without a physical fault.
        """
        import threading

        from dexmani_real.runtime.supervisor import run_supervisor

        shared = _BudgetShared(generation=5, state=SafetyState.RUNNING)
        shared.physical_home_completed = SimpleNamespace(value=True)
        shared.start_request = SimpleNamespace(value=False)
        shared.quit_requested = SimpleNamespace(value=False)
        shared.session_failed = SimpleNamespace(value=False)
        shared.run_started_monotonic_ns.value = time.monotonic_ns()
        result: dict = {}

        def target():
            result["outcome"] = run_supervisor(
                shared,
                [],
                status_interval_s=3600.0,
                heartbeat_timeouts_s={},
                supervisor_hz=200.0,
                max_running_s=0.05,
            )

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5.0
            while (
                time.monotonic() < deadline
                and int(shared.safety_state.value) != int(SafetyState.ARMED)
            ):
                time.sleep(0.01)
            self.assertEqual(int(shared.safety_state.value), int(SafetyState.ARMED))
            self.assertEqual(int(shared.run_generation.value), 6)
            self.assertFalse(bool(shared.error_state.value))
            # Only then does the control owner end the trial.
            shared.quit_requested.value = True
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())
            exit_reason, normal_exit = result["outcome"]
            self.assertEqual(exit_reason, "shutdown requested")
            self.assertTrue(normal_exit)
        finally:
            shared.is_running.value = False
            thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
