"""D1 BIT-EXACT — façade F1 state machine ≡ real XHand.send_action (plan §8 D1).

Same deterministic command sequence (np.random.default_rng(0), 50 commands,
some exceeding the delta clip, some beyond the joint limits) through two paths:

  path A — real ``XHand.send_action`` on a real XHand object, instantiated
           WITHOUT hardware: the SDK-facing ``control.send_command`` is
           monkeypatched to a capture stub (the value written into the
           HandCommand struct — what the firmware would receive — is the
           comparison target).  NOTE: ``_stub_mode`` cannot be used for D1 —
           its send_action deliberately applies ONLY the joint-limit clip and
           skips E2/E3 (xhand.py:508-514), so bit-exactness requires the
           normal (non-stub) pipeline with the SDK send captured.
  path B — ``HandSHMFaçade.send_action``'s returned ``expected_cmd`` with the
           façade constructed but never start()ed (state machine only, no
           child, no rings → ok=False).  XHand advances ``last_qpos_cmd``
           only on a successful send, so the test advances the façade's E3
           baseline the same way after each call — mirroring a successful
           delivery while exercising the exact lifted formulas.

Assertion: max abs diff < 1e-9 per step (the lifts are identical float64
arithmetic, so the diff is in fact exactly 0).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from dexmani_real.robot.hand_process import HandProcessConfig, HandSHMFaçade
from dexmani_real.robot.xhand.xhand import XHand, XHandConfig

# ── SDK-facing capture fakes (path A) ──


class _FakeFingerCommand:
    def __init__(self) -> None:
        self.position = 0.0


class _FakeHandCommand:
    """Minimal HandCommand_t stand-in: 12 finger commands with a position slot."""

    def __init__(self) -> None:
        self.finger_command = [_FakeFingerCommand() for _ in range(12)]


class _FakeControl:
    """Captures every SDK send_command as the (12,) position vector; always OK."""

    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []

    def send_command(self, device_id: int, command: _FakeHandCommand) -> SimpleNamespace:
        self.sent.append(np.array([fc.position for fc in command.finger_command], dtype=np.float64))
        return SimpleNamespace(error_code=0, error_message="")  # error_ok() → True


def _make_xhand_path(config: XHandConfig) -> XHand:
    """Real XHand in the normal (non-stub) mode with the SDK send captured."""
    hand = XHand(config)
    hand.control = _FakeControl()
    hand.hand_command = _FakeHandCommand()
    hand.connected_flag = True
    hand.error_state = False
    # Mirror connect()'s post-_init_hand_state baseline (no hardware read here).
    hand.last_qpos_cmd = np.asarray(config.home_qpos, dtype=np.float64).copy()
    hand._ema_qpos = None
    return hand


def _make_facade_path(config: XHandConfig) -> HandSHMFaçade:
    """HandSHMFaçade state machine only — never start()ed (no child, no rings)."""
    façade = HandSHMFaçade(
        HandProcessConfig(shm_prefix="t_f1_never_started"),
        None,
        None,
        config,
        None,
    )
    façade._last_qpos_cmd = np.asarray(config.home_qpos, dtype=np.float64).copy()
    façade._ema_qpos = None
    return façade


def _command_sequence(config: XHandConfig) -> np.ndarray:
    """50 deterministic commands: some beyond joint limits, some with >max_delta steps."""
    rng = np.random.default_rng(0)
    qmin = np.asarray(config.qpos_min, dtype=np.float64)
    qmax = np.asarray(config.qpos_max, dtype=np.float64)
    cmds = rng.uniform(qmin - 0.6, qmax + 0.6, size=(50, 12))
    # The sequence must exercise both gates or the comparison proves nothing.
    assert np.any(cmds > qmax) and np.any(cmds < qmin), "sequence never hits a joint limit"
    assert np.any(np.abs(np.diff(cmds, axis=0)) > float(config.max_delta_rad)), "sequence never exceeds the delta clip"
    return cmds


def test_d1_facade_expected_cmd_bitexact_vs_xhand():
    """limit → E3 delta → E2 EMA: façade expected_cmd ≡ XHand's sent value."""
    config = XHandConfig(ema_alpha=0.3, max_delta_rad=0.1)
    cmds = _command_sequence(config)

    hand = _make_xhand_path(config)
    façade = _make_facade_path(config)

    max_diff = 0.0
    for k, cmd in enumerate(cmds):
        # ── path A: real XHand pipeline, value captured at the SDK send ──
        assert hand.send_action(cmd) is True
        sent = hand.control.sent[-1]
        np.testing.assert_array_equal(hand.last_qpos_cmd, sent)  # success ⇒ identical

        # ── path B: façade state machine, zero-wait expected_cmd ──
        ok, expected = façade.send_action(cmd)
        assert ok is False  # no child / rings — expected_cmd is still produced

        diff = float(np.max(np.abs(sent - expected)))
        max_diff = max(max_diff, diff)
        assert diff < 1e-9, f"step {k}: XHand sent {sent} vs façade expected {expected} (diff {diff:.3g})"

        # XHand advances last_qpos_cmd only on a successful send; mirror that so
        # the next step's E3 baseline matches on both paths (EMA bookkeeping
        # already advanced inside send_action on both paths, verbatim).
        façade._last_qpos_cmd = expected.copy()

    assert max_diff < 1e-9
    assert len(hand.control.sent) == 50


def test_d1_bitexact_with_e2_e3_disabled():
    """alpha=0 / max_delta=0 (the child-side no-op config): joint-limit clip only."""
    config = XHandConfig(ema_alpha=0.0, max_delta_rad=0.0)
    cmds = _command_sequence(config)

    hand = _make_xhand_path(config)
    façade = _make_facade_path(config)

    qmin = np.asarray(config.qpos_min, dtype=np.float64)
    qmax = np.asarray(config.qpos_max, dtype=np.float64)
    for k, cmd in enumerate(cmds):
        assert hand.send_action(cmd) is True
        sent = hand.control.sent[-1]
        # With both gates disabled, both paths degenerate to the joint clip.
        np.testing.assert_array_equal(sent, np.clip(cmd, qmin, qmax))

        ok, expected = façade.send_action(cmd)
        assert ok is False
        np.testing.assert_array_equal(expected, sent)  # bit-exact (diff == 0.0)
        façade._last_qpos_cmd = expected.copy()
