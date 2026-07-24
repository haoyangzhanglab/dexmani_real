"""Arm + hand process layers through REAL mp.Process IPC with fake hardware.

The fakes implement the exact protocols the children drive (inner-loop protocol
for the arm child, stub-like XHand for the hand child) and run inside the
forked child — the parent therefore observes them ONLY through the SHM state
records, exactly like production:

  * arm: ``set_target`` targets are copied into ``last_sent_cmd`` (simulating
    delta-clip passthrough) and echoed via ARM_STATE["last_sent"]; a hold
    (``set_target(None)``) reverts ``last_sent_cmd`` to the fake's canned qpos.
  * arm: ``exec_macro`` encodes (code, n_waypoints, dt, wp[-1,0], wp[-1,6],
    speed, acc) into the result's ``final_qpos`` so the parent can verify the
    RPC roundtrip contents.
  * arm: shared ``kill_event`` / ``stale_event`` (fork-inherited mp.Events)
    make the fake raise SystemExit (crash test) or stop state publishing
    (stale-fabrication test) from within the child.
  * hand: send count / stop count are published via
    HAND_STATE["tactile_sum"][0,0] / [1,0]; a shared ``bias`` mp.Value offsets
    the echoed ``last_qpos_cmd`` joint 0 for echo-resync tests.

Each test uses a unique SHM prefix and short config timings (< 2 s each).
Plan ref: docs/arm-hand-process-isolation-plan.md §4-5, F1.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from dexmani_real.robot.arm_process import ArmProcessConfig, ArmSHMFaçade
from dexmani_real.robot.hand_process import HandProcessConfig, HandSHMFaçade
from dexmani_real.robot.xhand.xhand import XHandConfig
from dexmani_real.shm.robot_layouts import (
    ARM_CMD_EXEC_WAYPOINTS,
    ARM_TARGET_DTYPE,
    PRODUCER_POLICY,
    PRODUCER_TELEOP,
    new_frame,
)


def _wait_for(predicate, timeout: float = 1.0, interval: float = 0.005) -> bool:
    """Poll ``predicate`` until True or the deadline; returns the final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ═══════════════════════════════════════════════════════════════════
# Arm fakes (module-level — the fork child inherits them)
# ═══════════════════════════════════════════════════════════════════

_FAKE_ARM_QPOS = np.linspace(0.10, 0.70, 7)  # canned get_state qpos / hold position


class FakeInnerLoop:
    """In-memory inner-loop protocol implementation (see module docstring)."""

    def __init__(
        self, config: ArmProcessConfig, kill_event=None, stale_event=None, estop_ack=None, macro_release=None
    ) -> None:
        self._lock = threading.Lock()
        self.received: list = []  # thread-safe record of set_target calls (child-side only)
        self._qpos = _FAKE_ARM_QPOS.copy()
        self._last_sent = self._qpos.copy()  # starts holding position
        self._kill_event = kill_event
        self._stale_event = stale_event
        self._estop_ack = estop_ack  # mp.Event set by emergency_stop (parent observes)
        self._macro_release = macro_release  # mp.Event: hold exec_macro until set
        self._started = False
        # NOTE: deliberately no ``_cfg`` attribute — the child's getattr() sync
        # injection then sees None and skips the two-phase handshake wiring.

    # ── lifecycle ──

    def start(self) -> None:
        self._started = True

    def stop(self, timeout: float = 3.0) -> None:
        self._started = False

    @property
    def is_alive(self) -> bool:
        return self._started

    def wait_ready(self, timeout: float) -> bool:
        return True

    def emergency_stop(self) -> bool:
        """Fast-path estop mirror (ArmInnerLoop.emergency_stop): ack to the
        parent and 'stop the loop' — must complete WITHOUT any macro lock."""
        if self._estop_ack is not None:
            self._estop_ack.set()
        self._started = False
        return True

    # ── protocol ──

    def set_target(self, target) -> None:
        with self._lock:
            if target is None:
                self.received.append(None)  # is_hold path — hold current position
                self._last_sent = self._qpos.copy()
            else:
                q = np.asarray(target, dtype=np.float64).reshape(7).copy()
                self.received.append(q)
                self._last_sent = q  # delta-clip passthrough

    def get_state(self):
        # Kill hook FIRST: SystemExit is a BaseException — it escapes both
        # _publish_arm_state's and the child main's ``except Exception`` so the
        # child process actually dies (crash test).
        if self._kill_event is not None and self._kill_event.is_set():
            raise SystemExit(42)
        # Stale hook: RuntimeError is caught by _publish_arm_state → the child
        # keeps ticking but publishes nothing (façade freshness-gate test).
        if self._stale_event is not None and self._stale_event.is_set():
            raise RuntimeError("fail_stale: state publishing suspended")
        with self._lock:
            return self._qpos.copy(), False, time.perf_counter()

    def get_dynamics(self):
        zeros = np.zeros(7, dtype=np.float64)
        temps = np.full(7, 35.0, dtype=np.float64)
        return zeros.copy(), zeros.copy(), temps

    @property
    def tracking_error(self) -> float:
        return 0.0

    @property
    def last_sent_cmd(self) -> np.ndarray:
        with self._lock:
            return self._last_sent.copy()

    @property
    def ramp_step(self) -> int:
        return 0

    @property
    def mode(self) -> int:
        return 6

    @property
    def connected(self) -> bool:
        return True

    def exec_macro(self, code: int, fields: dict) -> dict:
        """Record the call and echo its contents back via final_qpos:

        ``[code, n_waypoints, dt, waypoints[-1, 0], waypoints[-1, 6], speed, acc]``
        so the parent can verify the RPC roundtrip through the result record.
        """
        if self._macro_release is not None:
            # Simulate a long-running macro: the RPC thread sits on this (and
            # on macro_lock child-side) until the test releases it.
            deadline = time.monotonic() + 30.0
            while not self._macro_release.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
        with self._lock:
            wp = np.asarray(fields.get("waypoints", np.zeros((0, 7), dtype=np.float64)), dtype=np.float64)
            n = int(wp.shape[0]) if wp.size else 0
            last = wp[-1] if n else np.zeros(7, dtype=np.float64)
            final = np.array(
                [
                    float(code),
                    float(n),
                    float(fields.get("dt", 0.0)),
                    float(last[0]),
                    float(last[6]),
                    float(fields.get("speed", 0.0)),
                    float(fields.get("acc", 0.0)),
                ],
                dtype=np.float64,
            )
        return {"ok": True, "arm_err": 0, "sdk_ret": 0, "final_qpos": final}


class _FakeInnerFactory:
    """Module-level callable factory; kill/stale events are fork-inherited."""

    def __init__(self, kill_event=None, stale_event=None, estop_ack=None, macro_release=None) -> None:
        self.kill_event = kill_event
        self.stale_event = stale_event
        self.estop_ack = estop_ack
        self.macro_release = macro_release

    def __call__(self, config: ArmProcessConfig) -> FakeInnerLoop:
        return FakeInnerLoop(
            config,
            kill_event=self.kill_event,
            stale_event=self.stale_event,
            estop_ack=self.estop_ack,
            macro_release=self.macro_release,
        )


def _make_arm_env(target_timeout_s: float = 0.2):
    """Build an ArmSHMFaçade over fresh rings with a fake inner loop.

    Returns (façade, kill_event, stale_event, prefix); caller must stop().
    """
    ctx = mp.get_context("fork")
    prefix = f"t_arm_{uuid.uuid4().hex[:10]}"
    kill_event = ctx.Event()
    stale_event = ctx.Event()
    config = ArmProcessConfig(
        loop_hz=50.0,
        shm_prefix=prefix,
        target_timeout_s=target_timeout_s,
        state_stale_mult=3.0,
        ready_timeout_s=5.0,
        rpc_timeout_s=2.0,
    )
    façade = ArmSHMFaçade(config, None, ctx.Event(), _FakeInnerFactory(kill_event, stale_event))
    return façade, kill_event, stale_event, prefix


def _arm_last_sent(façade: ArmSHMFaçade):
    rec, _age = façade.get_state()
    return None if rec is None else np.asarray(rec["last_sent"][0], dtype=np.float64)


@pytest.fixture()
def arm_env():
    façade, kill_event, stale_event, prefix = _make_arm_env()
    yield façade, kill_event, stale_event, prefix
    try:
        façade.stop(timeout=2.0)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Arm tests
# ═══════════════════════════════════════════════════════════════════


def test_arm_start_ready_and_target_passthrough(arm_env):
    façade, _kill, _stale, _prefix = arm_env
    assert façade.start() is True
    assert façade.wait_ready(2.0) is True
    assert façade.running

    q = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    façade.set_target(q)
    # The fake copies the target into last_sent_cmd → ARM_STATE["last_sent"] echo.
    assert _wait_for(
        lambda: np.allclose(_arm_last_sent(façade), q), timeout=1.0
    ), f"target not echoed within 1s (last_sent={_arm_last_sent(façade)})"
    rec, age = façade.get_state()
    assert rec is not None and age >= 0
    assert int(rec["error_state"][0]) == 0
    assert int(rec["connected"][0]) == 1


def test_arm_set_target_none_holds(arm_env):
    façade, _kill, _stale, _prefix = arm_env
    assert façade.start() and façade.wait_ready(2.0)

    q = np.full(7, -0.42)
    façade.set_target(q)
    assert _wait_for(lambda: np.allclose(_arm_last_sent(façade), q), timeout=1.0)

    façade.set_target(None)  # is_hold=1 → child calls inner.set_target(None)
    # The fake records None and reverts last_sent_cmd to its held position.
    assert _wait_for(
        lambda: np.allclose(_arm_last_sent(façade), _FAKE_ARM_QPOS), timeout=1.0
    ), f"hold not echoed (last_sent={_arm_last_sent(façade)})"


def test_arm_stop_unlinks_shm(arm_env):
    façade, _kill, _stale, prefix = arm_env
    assert façade.start() and façade.wait_ready(2.0)
    assert façade.running

    façade.stop(timeout=3.0)
    assert not façade.running
    for suffix in ("_state", "_target", "_cmd", "_cmd_result"):
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=f"{prefix}{suffix}")


def test_arm_stale_state_fabricates_error_record(arm_env):
    """Fake stops publishing (child stays alive) → façade freshness gate must
    fabricate an error record so validate_action trips — no silent staleness."""
    façade, _kill, stale_event, _prefix = arm_env
    assert façade.start() and façade.wait_ready(2.0)
    # ready_event lands before the first tick publishes — wait for a real frame.
    assert _wait_for(
        lambda: (lambda r: r is not None and int(r["connected"][0]) == 1)(façade.get_state()[0]),
        timeout=1.0,
    ), "child never published a healthy arm_state frame"

    stale_event.set()  # fake get_state raises → _publish_arm_state catches, no writes

    def _fabricated() -> bool:
        rec, _age = façade.get_state()
        return rec is not None and (int(rec["error_state"][0]) == 1 or int(rec["connected"][0]) == 0)

    assert _wait_for(_fabricated, timeout=1.0), "stale arm_state must produce a fabricated error record"
    rec, age = façade.get_state()
    assert int(rec["error_state"][0]) == 1
    assert int(rec["connected"][0]) == 0
    # Freshness gate = state_stale_mult / loop_hz = 3/50 = 60 ms.
    assert age > int(3.0 / 50.0 * 1e9)
    assert façade.running  # child is alive — only the state stream died
    assert not façade.crashed


def test_arm_crash_stale_semantics_and_ensure_running(arm_env):
    façade, kill_event, _stale, _prefix = arm_env
    assert façade.start() and façade.wait_ready(2.0)

    # ── crash the child (fake raises SystemExit inside the tick loop) ──
    kill_event.set()
    assert _wait_for(lambda: not façade.running, timeout=2.0), "child did not die after kill flag"
    assert façade.crashed

    time.sleep(0.1)  # let the last published state go stale (60 ms gate)
    rec, _age = façade.get_state()
    assert rec is not None, "get_state must fabricate, never return None, after a crash"
    assert int(rec["error_state"][0]) == 1 or int(rec["connected"][0]) == 0

    # ── ensure_running restarts the child (rings recreated) ──
    kill_event.clear()
    assert façade.ensure_running() is True
    assert façade.running
    assert not façade.crashed

    q2 = np.full(7, 0.25)
    façade.set_target(q2)
    assert _wait_for(
        lambda: np.allclose(_arm_last_sent(façade), q2), timeout=1.5
    ), f"restarted child does not echo targets (last_sent={_arm_last_sent(façade)})"


def test_arm_exec_waypoints_rpc_roundtrip(arm_env):
    façade, _kill, _stale, _prefix = arm_env
    assert façade.start() and façade.wait_ready(2.0)

    wp = np.arange(3 * 7, dtype=np.float64).reshape(3, 7) * 0.01
    result = façade.exec_waypoints(wp, dt=0.05)
    assert int(result["ok"][0]) == 1
    assert int(result["sdk_ret"][0]) == 0
    final = np.asarray(result["final_qpos"][0], dtype=np.float64)
    # Fake encodes (code, n_waypoints, dt, wp[-1,0], wp[-1,6], speed, acc).
    assert final[0] == pytest.approx(float(ARM_CMD_EXEC_WAYPOINTS))
    assert final[1] == pytest.approx(3.0)
    assert final[2] == pytest.approx(0.05)
    assert final[3] == pytest.approx(wp[-1, 0])
    assert final[4] == pytest.approx(wp[-1, 6])


def test_arm_producer_id_mismatch_ignored():
    """D9: a nonzero producer_id mismatch is dropped; producer_id=0 is accepted."""
    # target_timeout_s=2.0 so the stale-hold path cannot interfere with the
    # producer_id assertion window.
    façade, _kill, _stale, _prefix = _make_arm_env(target_timeout_s=2.0)
    try:
        assert façade.start() and façade.wait_ready(2.0)

        q1 = np.full(7, 0.11)
        façade.set_target(q1, producer_id=PRODUCER_TELEOP)
        assert _wait_for(lambda: np.allclose(_arm_last_sent(façade), q1), timeout=1.0)

        # Write a foreign-producer target straight onto the ring.
        q2 = np.full(7, 0.99)
        frame = new_frame(ARM_TARGET_DTYPE)
        frame["target"][0] = q2
        frame["producer_id"][0] = PRODUCER_POLICY
        façade._target_ring.write(frame)

        # The child must ignore it for the whole 0.3 s observation window.
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            last = _arm_last_sent(façade)
            assert last is not None
            assert not np.allclose(last, q2), "child applied a foreign-producer target"
            time.sleep(0.02)

        # producer_id=0 (unset) passes the check.
        q3 = np.full(7, 0.33)
        frame0 = new_frame(ARM_TARGET_DTYPE)
        frame0["target"][0] = q3
        frame0["producer_id"][0] = 0
        façade._target_ring.write(frame0)
        assert _wait_for(lambda: np.allclose(_arm_last_sent(façade), q3), timeout=1.0)
    finally:
        façade.stop(timeout=2.0)


# ═══════════════════════════════════════════════════════════════════
# Hand fakes (module-level — the fork child inherits them)
# ═══════════════════════════════════════════════════════════════════


class FakeHand:
    """Stub-like XHand replacement for the hand child.

    * ``send_action`` joint-limit-clips (safety net, like the real child),
      optionally offsets joint 0 by a shared ``bias`` value (echo-resync test)
      and stores the result as ``last_qpos_cmd`` (the echoed value).
    * ``get_state`` returns zero tactile, EXCEPT ``tactile_force_sum[0, 0]`` =
      send count and ``[1, 0]`` = stop() count — the parent observes both
      through HAND_STATE["tactile_sum"].
    """

    def __init__(self, config: XHandConfig, bias_value=None, traj_started=None) -> None:
        self.config = config
        self.connected_flag = False
        self.error_state = False
        self.last_error_code = None
        self.last_joint_limit_clipped = False
        self.consecutive_send_errors = 0
        self.last_qpos_cmd: np.ndarray | None = None
        self._send_count = 0
        self._stop_count = 0
        self._bias = bias_value  # fork-inherited mp.Value("d") or None
        self._traj_started = traj_started  # fork-inherited mp.Event or None

    def connect(self) -> bool:
        self.connected_flag = True
        # Mirror XHand._init_hand_state: anchor last_qpos_cmd to "hardware" (home).
        self.last_qpos_cmd = np.asarray(self.config.home_qpos, dtype=np.float64).copy()
        return True

    def send_action(self, qpos_cmd) -> bool:
        q = np.clip(
            np.asarray(qpos_cmd, dtype=np.float64).reshape(12),
            np.asarray(self.config.qpos_min, dtype=np.float64),
            np.asarray(self.config.qpos_max, dtype=np.float64),
        )
        if self._bias is not None and self._bias.value != 0.0:
            q = q.copy()
            q[0] += self._bias.value
        self.last_qpos_cmd = q
        self._send_count += 1
        return True

    def get_state(self, full: bool = False, force_update: bool | None = None) -> dict:
        qpos = self.last_qpos_cmd if self.last_qpos_cmd is not None else np.zeros(12, dtype=np.float64)
        tactile_sum = np.zeros((5, 3), dtype=np.float64)
        tactile_sum[0, 0] = float(self._send_count)  # send-count side channel
        tactile_sum[1, 0] = float(self._stop_count)  # detorque-count side channel
        return {
            "qpos": np.asarray(qpos, dtype=np.float64).copy(),
            "tactile_force_sum": tactile_sum,
            "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
        }

    def stop(self) -> bool:
        self._stop_count += 1
        self.error_state = True
        return True

    def reset(self, qpos=None) -> bool:
        return self.send_action(np.zeros(12) if qpos is None else np.asarray(qpos, dtype=np.float64))

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_code = None
        return True

    def send_trajectory(
        self,
        waypoints,
        duration_s: float,
        max_speed: float | None = None,
        abort_event=None,
    ) -> bool:
        """Simulate the real step loop: sleep out ``duration_s`` in 20 ms
        steps, aborting at a step boundary when ``abort_event`` is set —
        exactly like XHand.send_trajectory (returns False on abort)."""
        if self._traj_started is not None:
            self._traj_started.set()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            if abort_event is not None and abort_event.is_set():
                return False  # aborted mid-trajectory (estop preemption)
            time.sleep(0.02)
        return True

    def reset_connection(self) -> bool:
        self.consecutive_send_errors = 0
        return True


class _FakeHandFactory:
    """Module-level callable factory; bias mp.Value is fork-inherited."""

    def __init__(self, bias_value=None, traj_started=None) -> None:
        self.bias_value = bias_value
        self.traj_started = traj_started

    def __call__(self, config: XHandConfig) -> FakeHand:
        return FakeHand(config, bias_value=self.bias_value, traj_started=self.traj_started)


def _hand_config(ema_alpha: float = 0.0, max_delta_rad: float = 0.1) -> XHandConfig:
    return XHandConfig(
        home_qpos=np.zeros(12, dtype=np.float64),
        qpos_min=np.full(12, -1.0, dtype=np.float64),
        qpos_max=np.full(12, 1.0, dtype=np.float64),
        ema_alpha=ema_alpha,
        max_delta_rad=max_delta_rad,
    )


def _make_hand_env(ema_alpha: float = 0.0, max_delta_rad: float = 0.1):
    """Build a HandSHMFaçade over fresh rings with a FakeHand child.

    Returns (façade, bias_value, hand_config); caller must stop().
    """
    ctx = mp.get_context("fork")
    prefix = f"t_hand_{uuid.uuid4().hex[:10]}"
    config = HandProcessConfig(
        hz=30.0,
        shm_prefix=prefix,
        cmd_stale_hold_s=0.5,
        state_stale_s=0.2,
        rpc_timeout_s=2.0,
    )
    bias = ctx.Value("d", 0.0)
    hand_cfg = _hand_config(ema_alpha=ema_alpha, max_delta_rad=max_delta_rad)
    façade = HandSHMFaçade(config, None, ctx.Event(), hand_cfg, _FakeHandFactory(bias))
    return façade, bias, hand_cfg


def _hand_state(façade: HandSHMFaçade):
    rec, _age = façade.get_state()
    return rec


def _hand_send_count(façade: HandSHMFaçade) -> int:
    return int(round(float(_hand_state(façade)["tactile_sum"][0][0, 0])))


def _hand_stop_count(façade: HandSHMFaçade) -> int:
    return int(round(float(_hand_state(façade)["tactile_sum"][0][1, 0])))


@pytest.fixture()
def hand_env():
    façade, bias, hand_cfg = _make_hand_env()
    yield façade, bias, hand_cfg
    try:
        façade.stop(timeout=2.0)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# Hand tests (F1: clip/EMA state machine in the façade, stateless child)
# ═══════════════════════════════════════════════════════════════════


def test_hand_send_action_returns_expected_immediately(hand_env):
    façade, _bias, _cfg = hand_env
    assert façade.start() is True
    assert façade.wait_ready(3.0) is True
    assert façade.running

    target = np.full(12, 0.02)
    t0 = time.monotonic()
    ok, expected = façade.send_action(target)
    elapsed = time.monotonic() - t0
    assert ok is True
    assert elapsed < 0.1, f"send_action blocked for {elapsed:.3f}s — must return immediately"
    # Baseline seeded from the child echo (home=zeros); 0.02 < max_delta 0.1 → passthrough.
    np.testing.assert_allclose(expected, target, atol=1e-12)
    np.testing.assert_allclose(façade.last_qpos_cmd, target, atol=1e-12)


def test_hand_e3_delta_clip_steps_exactly_max_delta(hand_env):
    façade, _bias, _cfg = hand_env  # max_delta_rad=0.1, ema_alpha=0
    assert façade.start() and façade.wait_ready(3.0)

    big = np.full(12, 0.5)  # 5× the per-step limit
    ok, expected = façade.send_action(big)
    assert ok
    np.testing.assert_allclose(expected, np.full(12, 0.1), atol=1e-12, rtol=0)

    ok, expected2 = façade.send_action(big)
    assert ok
    np.testing.assert_allclose(expected2, np.full(12, 0.2), atol=1e-12, rtol=0)

    # Negative direction as well.
    ok, expected3 = façade.send_action(np.full(12, -0.5))
    assert ok
    np.testing.assert_allclose(expected3, np.full(12, 0.1), atol=1e-12, rtol=0)  # 0.2 → 0.1


def test_hand_e2_ema_matches_closed_form():
    façade, _bias, cfg = _make_hand_env(ema_alpha=0.3, max_delta_rad=0.1)
    try:
        assert façade.start() and façade.wait_ready(3.0)

        rng = np.random.default_rng(1)
        cmds = rng.uniform(-0.5, 0.5, size=(10, 12))

        # Closed-form re-derivation of limit → E3 → E2 (verbatim XHand order).
        last = np.zeros(12, dtype=np.float64)  # baseline seeded from child echo (home)
        ema = None  # _ema_qpos: None until the first command (verbatim)
        alpha = cfg.ema_alpha
        limit = cfg.max_delta_rad
        for cmd in cmds:
            x = np.clip(cmd, -1.0, 1.0)
            x = last + np.clip(x - last, -limit, limit)  # E3
            if ema is not None:  # E2 skips (and seeds on) the first command
                x = (1.0 - alpha) * ema + alpha * x
            ema = x.copy()
            last = x.copy()  # ok=True → _last_qpos_cmd advances
            ok, expected = façade.send_action(cmd)
            assert ok
            np.testing.assert_allclose(expected, x, atol=1e-12, rtol=0)
    finally:
        façade.stop(timeout=2.0)


def test_hand_joint_limit_clip_in_facade():
    façade, _bias, _cfg = _make_hand_env(ema_alpha=0.0, max_delta_rad=0.0)  # E3 disabled
    try:
        assert façade.start() and façade.wait_ready(3.0)

        ok, expected = façade.send_action(np.full(12, 5.0))
        assert ok
        np.testing.assert_allclose(expected, np.full(12, 1.0), atol=1e-12, rtol=0)  # clipped to qpos_max
        assert façade._last_joint_limit_clipped is True

        ok, expected = façade.send_action(np.full(12, -5.0))
        assert ok
        np.testing.assert_allclose(expected, np.full(12, -1.0), atol=1e-12, rtol=0)  # clipped to qpos_min
    finally:
        façade.stop(timeout=2.0)


def test_hand_echo_resync_on_value_mismatch(hand_env, caplog):
    façade, bias, _cfg = hand_env  # max_delta=0.1; bias offsets the child's echo joint 0
    assert façade.start() and façade.wait_ready(3.0)

    # Step 1: healthy echo — baseline 0 → expected 0.02, child echoes 0.02.
    c1 = np.zeros(12)
    c1[0] = 0.02
    ok, e1 = façade.send_action(c1)
    assert ok and abs(e1[0] - 0.02) < 1e-12
    assert _wait_for(lambda: abs(float(_hand_state(façade)["last_qpos_cmd"][0][0]) - 0.02) < 1e-12, timeout=1.0)
    façade.check_echo()

    # Step 2: bias the child's echo by +0.05 on joint 0 → value mismatch.
    bias.value = 0.05
    c2 = c1.copy()
    c2[0] += 0.02  # baseline 0.02 → expected 0.04
    ok, e2 = façade.send_action(c2)
    assert ok and abs(e2[0] - 0.04) < 1e-12
    assert _wait_for(lambda: abs(float(_hand_state(façade)["last_qpos_cmd"][0][0]) - 0.09) < 1e-12, timeout=1.0)

    with caplog.at_level(logging.WARNING):
        façade.check_echo()  # mismatch → throttled warn + resync baseline ← echo value

    resynced = façade.last_qpos_cmd
    assert resynced is not None
    np.testing.assert_allclose(resynced[0], 0.09, atol=1e-12)
    assert "resyncing baseline" in caplog.text

    # Step 3: the next command continues from the ECHOED value (0.09), not the
    # stale façade baseline (0.04) — delta +0.02 → 0.11 (vs 0.06 if unresynced).
    c3 = np.zeros(12)
    c3[0] = 0.11
    ok, e3 = façade.send_action(c3)
    assert ok
    assert abs(e3[0] - 0.11) < 1e-9, f"baseline not resynced (got {e3[0]:.6f}, want 0.11)"


def test_hand_sends_only_on_new_seq(hand_env):
    façade, _bias, _cfg = hand_env
    assert façade.start() and façade.wait_ready(3.0)

    ok, _e = façade.send_action(np.full(12, 0.03))
    assert ok
    assert _wait_for(lambda: _hand_send_count(façade) == 1, timeout=1.0)

    # ~9 child ticks at 30 Hz with the ring untouched: the same seq must NOT
    # be re-sent (position servo holds without refresh).
    time.sleep(0.3)
    assert _hand_send_count(façade) == 1, "child re-sent an already-processed seq"
    assert façade.running and not façade.crashed


def test_hand_cmd_stale_holds_and_never_detorques(hand_env):
    """cmd ring stale > cmd_stale_hold_s (0.5 s) → child stops sending new
    commands (hold position) and NEVER calls stop() (detorque)."""
    façade, _bias, _cfg = hand_env
    assert façade.start() and façade.wait_ready(3.0)

    ok, _e = façade.send_action(np.full(12, 0.04))
    assert ok
    assert _wait_for(lambda: _hand_send_count(façade) == 1, timeout=1.0)

    time.sleep(0.7)  # > cmd_stale_hold_s, ring untouched

    rec = _hand_state(façade)
    assert int(round(float(rec["tactile_sum"][0][0, 0]))) == 1, "send count must freeze while cmd is stale"
    assert int(round(float(rec["tactile_sum"][0][1, 0]))) == 0, "child must NEVER detorque on stale cmd"
    assert int(rec["connected"][0]) == 1  # child still alive and publishing state
    assert façade.running and not façade.crashed


# ═══════════════════════════════════════════════════════════════════
# E-stop preemption + orphan-exit regression tests (plan §4.8, §5.2)
# ═══════════════════════════════════════════════════════════════════


def test_arm_estop_preempts_in_flight_macro():
    """Estop while a long RPC macro holds the RPC thread: the child must fire
    the fast-path emergency_stop WITHOUT waiting on the macro (the old
    macro_lock path blocked here for the whole macro duration)."""
    ctx = mp.get_context("fork")
    prefix = f"t_arm_{uuid.uuid4().hex[:10]}"
    estop_ack = ctx.Event()
    macro_release = ctx.Event()  # held clear → the fake macro blocks
    config = ArmProcessConfig(
        loop_hz=50.0,
        shm_prefix=prefix,
        ready_timeout_s=5.0,
        rpc_timeout_s=15.0,
    )
    façade = ArmSHMFaçade(
        config,
        None,
        ctx.Event(),
        _FakeInnerFactory(estop_ack=estop_ack, macro_release=macro_release),
    )
    try:
        assert façade.start() and façade.wait_ready(2.0)

        wp = np.arange(3 * 7, dtype=np.float64).reshape(3, 7) * 0.01
        macro_result: dict = {}

        def _run_macro() -> None:
            try:
                macro_result["res"] = façade.exec_waypoints(wp, dt=0.05)
            except Exception as e:  # late result / timeout — irrelevant here
                macro_result["err"] = repr(e)

        thread = threading.Thread(target=_run_macro, daemon=True)
        thread.start()
        time.sleep(0.2)  # let the RPC thread pick up the macro and block on it

        # Raise estop while the macro is in flight.
        façade._proc._estop_event.set()
        assert estop_ack.wait(timeout=1.0), (
            "estop was deferred behind the in-flight macro — the child must take "
            "the fast path (no macro_lock) per plan §4.8"
        )

        macro_release.set()  # let the macro finish
        thread.join(timeout=3.0)
        assert not thread.is_alive()
    finally:
        macro_release.set()
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


def test_hand_estop_preempts_in_flight_trajectory():
    """Estop mid-SEND_TRAJECTORY: the trajectory must abort at a step boundary
    and the tick loop must detorque (hand.stop) far before the trajectory's
    natural end (the old tick-thread RPC blocked estop for the full duration)."""
    ctx = mp.get_context("fork")
    prefix = f"t_hand_{uuid.uuid4().hex[:10]}"
    traj_started = ctx.Event()
    config = HandProcessConfig(hz=30.0, shm_prefix=prefix, rpc_timeout_s=5.0)
    hand_cfg = _hand_config()
    façade = HandSHMFaçade(config, None, ctx.Event(), hand_cfg, _FakeHandFactory(traj_started=traj_started))
    try:
        assert façade.start() and façade.wait_ready(3.0)

        result_box: dict = {}

        def _run_macro() -> None:
            result_box["res"] = façade.send_trajectory(np.zeros((3, 12)), duration_s=5.0)

        thread = threading.Thread(target=_run_macro, daemon=True)
        t0 = time.monotonic()
        thread.start()
        assert traj_started.wait(timeout=2.0), "trajectory macro never started"
        time.sleep(0.1)  # settle into the step loop

        façade._proc._estop_event.set()
        thread.join(timeout=3.0)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive(), "estop did not preempt the in-flight trajectory"
        assert elapsed < 4.0, f"macro ran {elapsed:.1f}s of its 5s duration despite estop"
        res = result_box.get("res")
        assert res is not None and int(res["ok"][0]) == 0, "aborted trajectory must report ok=0"
        assert _hand_stop_count(façade) >= 1, "estop detorque (hand.stop) must have run"
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass


def test_hand_orphan_exits_cleanly_after_silence():
    """daemon=False child with a main that 'exited without stop()': after
    orphan_exit_s with zero new cmd seqs the child holds position (never
    detorques) and exits CLEANLY — so multiprocessing's atexit join cannot
    hang interpreter shutdown."""
    ctx = mp.get_context("fork")
    prefix = f"t_hand_{uuid.uuid4().hex[:10]}"
    config = HandProcessConfig(hz=30.0, shm_prefix=prefix, orphan_exit_s=0.5)
    hand_cfg = _hand_config()
    façade = HandSHMFaçade(config, None, ctx.Event(), hand_cfg, _FakeHandFactory())
    try:
        assert façade.start() is True and façade.wait_ready(3.0) is True
        assert façade.running

        # No commands at all — simulate a main process that exited without
        # calling stop(). The child must exit autonomously after the budget.
        assert _wait_for(lambda: not façade.running, timeout=5.0), "orphaned child did not exit autonomously"
        assert not façade.crashed, "autonomous orphan exit (code 0) must not be flagged as a crash"
    finally:
        try:
            façade.stop(timeout=2.0)
        except Exception:
            pass
