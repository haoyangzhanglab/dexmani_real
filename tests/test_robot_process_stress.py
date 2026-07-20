"""Adversarial runtime stress tests for the arm/hand process-isolation layers.

Red-team coverage for plan §8 C3 (child crash injection), C4 (stale SHM at
startup) and §5.2 (lifecycle): real fork children with fake hardware driven
through the SHM rings, exactly like production.

  1. Restart cycles — 5× start → set_target → stop of ArmSHMFaçade; every stop
     must reap the child (recorded pid dead post-join) and unlink all four SHM
     segments (attaching by name raises FileNotFoundError).
  2. Arm SIGKILL — crash mid target-stream: ``crashed`` within ~1 s,
     get_state fabricates an error record validate_action would reject
     (error_state=1 / connected=0), ensure_running restarts and a fresh target
     reaches the fake inner loop within 0.3 s.
  3. Stale SHM at startup — pre-created raw blocks named {prefix}_state /
     _target with a WRONG size are unlinked + recreated by start(); targets
     flow afterwards.
  4. Hand SIGKILL — degrades to connected=0 with NO escalation (estop event
     untouched); send_action still returns the clip/EMA expected_cmd
     immediately (F1 state machine in the main process survives child death).
  5. RPC under death — an RPC convenience call against a SIGKILLed child
     raises RpcTimeoutError promptly (no hang); after ensure_running the same
     RPC succeeds.
  6. set_target flood — 200 rapid writes: the ring drops the oldest, never
     raises, and the latest read is exactly the last write (seqlock
     consistency, logical sequence survives maxlen=2 wraps).

All SHM names carry the unique ``stress_`` prefix so this file can never
collide with other test files' rings; the whole file runs in < 15 s.
Plan ref: docs/arm-hand-process-isolation-plan.md §4-5, §8 C3/C4, F1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from dexmani_real.robot.arm_process import ArmProcessConfig, ArmSHMFaçade
from dexmani_real.robot.hand_process import HandProcessConfig, HandSHMFaçade
from dexmani_real.shm.robot_layouts import ARM_CMD_CLEAR_ERROR
from dexmani_real.shm.robot_rpc import RpcTimeoutError

# Reuse the module-level fakes from the functional test file (they run inside
# the forked child; the parent observes them only through SHM state records).
try:  # pytest prepend import mode: tests/ is on sys.path
    from test_robot_process import _arm_last_sent, _FakeHandFactory, _FakeInnerFactory, _hand_config, _wait_for
except ImportError:  # pragma: no cover — package-style invocation fallback
    from tests.test_robot_process import (  # type: ignore[no-redef]
        _arm_last_sent,
        _FakeHandFactory,
        _FakeInnerFactory,
        _hand_config,
        _wait_for,
    )

# The four arm ring segments (names: f"{shm_prefix}{suffix}").
_ARM_SUFFIXES = ("_state", "_target", "_cmd", "_cmd_result")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _stress_prefix(kind: str) -> str:
    """Unique per-test SHM prefix; 'stress_' never collides with other files."""
    return f"stress_{kind}_{uuid.uuid4().hex[:10]}"


def _pid_dead(pid: int) -> bool:
    """True once the pid is fully reaped (os.kill raises ProcessLookupError)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _make_stress_arm(prefix: str | None = None, rpc_timeout_s: float = 2.0) -> tuple[ArmSHMFaçade, str]:
    """ArmSHMFaçade over a fake inner loop on stress_-prefixed rings.

    Caller must stop() the façade.
    """
    ctx = mp.get_context("fork")
    prefix = prefix if prefix is not None else _stress_prefix("arm")
    config = ArmProcessConfig(
        loop_hz=50.0,
        shm_prefix=prefix,
        target_timeout_s=0.2,
        state_stale_mult=3.0,
        ready_timeout_s=5.0,
        rpc_timeout_s=rpc_timeout_s,
    )
    façade = ArmSHMFaçade(config, None, ctx.Event(), _FakeInnerFactory())
    return façade, prefix


# ═══════════════════════════════════════════════════════════════════
# 1. Restart cycles — no pid / SHM leaks across 5 start→target→stop rounds
# ═══════════════════════════════════════════════════════════════════


def test_arm_restart_cycles_no_pid_or_shm_leaks() -> None:
    """§5.2 lifecycle hygiene: 5× start → set_target → stop. After every stop
    the child pid must be reaped and all four SHM segments unlinked."""
    façade, prefix = _make_stress_arm()
    try:
        for cycle in range(5):
            assert façade.start() is True, f"cycle {cycle}: start failed"
            assert façade.wait_ready(3.0), f"cycle {cycle}: child not ready"
            pid = façade._proc._process.pid
            assert pid is not None and pid > 0

            q = np.full(7, 0.05 * (cycle + 1))
            façade.set_target(q)
            assert _wait_for(
                lambda q=q: np.allclose(_arm_last_sent(façade), q), timeout=1.0
            ), f"cycle {cycle}: target not echoed (last_sent={_arm_last_sent(façade)})"

            façade.stop(timeout=3.0)
            assert not façade.running, f"cycle {cycle}: child still alive after stop"
            assert _wait_for(
                lambda: _pid_dead(pid), timeout=1.0
            ), f"cycle {cycle}: child pid {pid} leaked — alive after join"
            for suffix in _ARM_SUFFIXES:
                with pytest.raises(FileNotFoundError):
                    shared_memory.SharedMemory(name=f"{prefix}{suffix}")
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 2. Arm SIGKILL — crash detection, fabricated error record, restart
# ═══════════════════════════════════════════════════════════════════


def test_arm_sigkill_crash_detected_and_restarts() -> None:
    """§8 C3: SIGKILL the arm child mid target-stream. ``crashed`` must be
    True within ~1 s, get_state must fabricate an error record (error_state=1,
    connected=0 — validate_action trips), and ensure_running must restart the
    child so a new target reaches the fake within 0.3 s."""
    façade, _prefix = _make_stress_arm()
    stop_writer = threading.Event()

    def _writer() -> None:
        q = np.full(7, 0.12)
        while not stop_writer.is_set():
            try:
                façade.set_target(q)
            except Exception:
                return
            time.sleep(0.005)

    try:
        assert façade.start() and façade.wait_ready(3.0)
        writer = threading.Thread(target=_writer, daemon=True)
        writer.start()
        assert _wait_for(
            lambda: np.allclose(_arm_last_sent(façade), np.full(7, 0.12)), timeout=1.0
        ), "streamed targets never echoed before the crash"

        pid = façade._proc._process.pid
        t_kill = time.monotonic()
        os.kill(pid, signal.SIGKILL)
        assert _wait_for(lambda: façade.crashed, timeout=1.0), "crash not detected within ~1s"
        assert time.monotonic() - t_kill < 1.0
        stop_writer.set()
        writer.join(timeout=1.0)
        assert not writer.is_alive()

        # Last published state ages past the freshness gate (3/50 = 60 ms).
        time.sleep(0.1)
        rec, _age = façade.get_state()
        assert rec is not None, "get_state must fabricate, never return None, after a crash"
        assert int(rec["error_state"][0]) == 1
        assert int(rec["connected"][0]) == 0, "post-crash record must trip validate_action"

        # Restart and prove the new pipeline is live within 0.3 s.
        assert façade.ensure_running() is True
        assert façade.running and not façade.crashed
        q2 = np.full(7, -0.35)
        t0 = time.monotonic()
        façade.set_target(q2)
        assert _wait_for(
            lambda: np.allclose(_arm_last_sent(façade), q2), timeout=0.3
        ), f"restarted child did not echo a new target within 0.3s (last={_arm_last_sent(façade)})"
        assert time.monotonic() - t0 < 0.3
    finally:
        stop_writer.set()
        try:
            façade.stop(timeout=3.0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 3. Stale wrong-size SHM at startup — cleaned, recreated, targets flow
# ═══════════════════════════════════════════════════════════════════


def test_arm_stale_shm_at_startup_cleaned_and_flows() -> None:
    """§8 C4: raw SharedMemory blocks named {prefix}_state / _target with a
    WRONG size pre-exist → start() unlinks + recreates them and targets flow
    (a stale arm_target from a dead run must never be chased)."""
    prefix = _stress_prefix("arm")
    stale_blocks: list[shared_memory.SharedMemory] = []
    for suffix in ("_state", "_target"):
        stale_blocks.append(shared_memory.SharedMemory(name=f"{prefix}{suffix}", create=True, size=64))
    façade, prefix = _make_stress_arm(prefix=prefix)
    try:
        assert façade.start() is True, "start must survive stale wrong-size SHM blocks"
        assert façade.wait_ready(3.0)

        # The wrong-size blocks were replaced by real rings (larger than 64 B).
        # (The probe's tracker registration is balanced by the façade's unlink
        # at stop() — set-based tracker: duplicate registrations collapse.)
        probe = shared_memory.SharedMemory(name=f"{prefix}_state")
        try:
            assert probe.size > 64, "stale 64-byte block was not recreated"
        finally:
            probe.close()

        q = np.full(7, 0.21)
        façade.set_target(q)
        assert _wait_for(
            lambda: np.allclose(_arm_last_sent(façade), q), timeout=1.0
        ), f"targets do not flow after stale-SHM cleanup (last_sent={_arm_last_sent(façade)})"
    finally:
        # close() only: the façade's stale-cleanup unlink already balanced the
        # tracker registration of these blocks (a manual unregister would hit
        # a KeyError on the set-based tracker and untrack the recreated ring).
        for sm in stale_blocks:
            try:
                sm.close()
            except Exception:
                pass
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 4. Hand SIGKILL — degraded mode, NO escalation, send_action survives
# ═══════════════════════════════════════════════════════════════════


def test_hand_sigkill_degrades_without_escalation() -> None:
    """§8 C3 (hand): SIGKILL the hand child → get_state returns connected=0
    (degraded, NO escalation — the estop event stays clear) and send_action
    still returns the F1 clip/EMA expected_cmd immediately with zero wait."""
    ctx = mp.get_context("fork")
    prefix = _stress_prefix("hand")
    estop = ctx.Event()
    config = HandProcessConfig(
        hz=30.0,
        shm_prefix=prefix,
        cmd_stale_hold_s=0.5,
        state_stale_s=0.2,
        rpc_timeout_s=1.0,
    )
    façade = HandSHMFaçade(config, None, estop, _hand_config(), _FakeHandFactory())
    try:
        assert façade.start() is True and façade.wait_ready(3.0)
        pid = façade._proc._process.pid

        os.kill(pid, signal.SIGKILL)
        assert _wait_for(lambda: not façade.running, timeout=1.0), "hand child death not detected"
        assert façade.crashed, "SIGKILLed hand child must be flagged crashed"

        time.sleep(0.3)  # > state_stale_s (0.2) → freshness gate fabricates
        rec, _age = façade.get_state()
        assert int(rec["connected"][0]) == 0, "dead hand child must read as disconnected"
        assert int(rec["error_state"][0]) == 1
        assert not estop.is_set(), "hand staleness must DEGRADE, never escalate to estop"

        # F1: the clip/EMA state machine lives in the main process — it
        # survives child death and answers immediately (ok=False: nothing can
        # be written to a dead child, but there is no raise and no wait).
        t0 = time.monotonic()
        ok, expected = façade.send_action(np.full(12, 0.02))
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, f"send_action blocked {elapsed:.3f}s after child death — must be immediate"
        np.testing.assert_allclose(expected, np.full(12, 0.02), atol=1e-12)  # baseline 0 + delta 0.02 ≤ max 0.1
        assert ok is False
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 5. RPC against a dead child — prompt timeout, then recovery
# ═══════════════════════════════════════════════════════════════════


def test_arm_rpc_times_out_promptly_on_dead_child_then_recovers() -> None:
    """§8 C3: an RPC convenience call against a SIGKILLed child must raise
    RpcTimeoutError promptly (no hang); after ensure_running the same RPC
    round-trips through the restarted child."""
    façade, _prefix = _make_stress_arm(rpc_timeout_s=0.5)
    try:
        assert façade.start() and façade.wait_ready(3.0)
        pid = façade._proc._process.pid
        os.kill(pid, signal.SIGKILL)
        assert _wait_for(lambda: façade.crashed, timeout=1.0)

        t0 = time.monotonic()
        with pytest.raises(RpcTimeoutError):
            façade.clear_error()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5, f"RPC to a dead child hung {elapsed:.1f}s (timeout=0.5s) — must fail promptly"

        assert façade.ensure_running() is True
        result = façade.clear_error()
        assert int(result["ok"][0]) == 1
        final = np.asarray(result["final_qpos"][0], dtype=np.float64)
        # The fake encodes the RPC code in final_qpos[0] — proves the round trip.
        assert final[0] == pytest.approx(float(ARM_CMD_CLEAR_ERROR))
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 6. set_target flood — drop-oldest ring, seqlock-consistent latest read
# ═══════════════════════════════════════════════════════════════════


def test_arm_set_target_flood_drops_oldest_and_reads_last() -> None:
    """200 rapid set_target writes into the maxlen=2 arm_target ring: never
    raises, drops the oldest, and read_latest returns exactly the last write
    (seqlock consistency; the logical sequence survives 100 wraps)."""
    façade, _prefix = _make_stress_arm()
    try:
        assert façade.start() and façade.wait_ready(3.0)

        n = 200
        q_last = None
        for i in range(n):
            q_last = np.full(7, 0.001 * i)
            façade.set_target(q_last)  # must never raise under flood

        latest = façade._target_ring.read_latest()
        assert latest is not None, "flooded ring lost its latest frame"
        data, _ts_ns, seq = latest
        assert seq == n, f"logical sequence broken by maxlen=2 wraps (got {seq}, want {n})"
        np.testing.assert_allclose(np.asarray(data["target"][0], dtype=np.float64), q_last, atol=1e-12)

        # The child converges on the final target once the flood settles.
        assert _wait_for(
            lambda: np.allclose(_arm_last_sent(façade), q_last), timeout=1.0
        ), f"child never applied the final flooded target (last_sent={_arm_last_sent(façade)})"
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass
