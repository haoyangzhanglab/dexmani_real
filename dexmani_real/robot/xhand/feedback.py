"""Bounded XHand feedback and tactile-vector hold, without device I/O."""

import logging

import numpy as np

from dexmani_real.robot.feedback import HAND_STATE_DTYPE

logger = logging.getLogger(__name__)


def hand_feedback_time_ns(hand, required_tactile=("aggregate", "dense")):
    """Oldest required hand source, without conflating tactile age with joint time."""
    return min(
        int(hand[name][0])
        for name in ("timestamp_ns", *(f"tactile_{name}_timestamp_ns" for name in required_tactile))
    )


def update_tactile_cache(raw, values, valid_ns, now_ns):
    """Hold whole three-axis vectors; only finite new vectors advance their clocks."""
    # Validate in the observation dtype too: finite float64 may overflow float32.
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(raw, dtype=values.dtype)
    if raw.shape != values.shape:
        raise RuntimeError(f"invalid tactile shape: {raw.shape}, expected {values.shape}")
    valid = np.isfinite(raw).all(axis=-1)
    np.copyto(values, raw, where=valid[..., None])
    np.copyto(valid_ns, now_ns, where=valid)
    return not valid.all()


class XHandFeedback:
    """Own consumed samples and their validity clocks within one robot connection."""

    def __init__(self, config, *, required_tactile=("aggregate", "dense")):
        self.config = config
        self.required_tactile = tuple(required_tactile)
        if not set(self.required_tactile) <= {"aggregate", "dense"}:
            raise ValueError("unknown required tactile channel")
        self.latest = None
        self.tactile_cache = {}
        self.tactile_held = set()
        self.read_crc = False
        self.initializing_until_ns = None
        self.previous_errors = None

    def reset(self, now_ns, *, deadline_ns=None):
        self.latest = None
        self.tactile_cache = {
            name: (
                np.full(HAND_STATE_DTYPE[f"tactile_{name}"].shape, np.nan, dtype=np.float32),
                np.zeros(HAND_STATE_DTYPE[f"tactile_{name}"].shape[:-1], dtype=np.int64),
            )
            for name in ("aggregate", "dense")
        }
        self.tactile_held.clear()
        self.read_crc = False
        self.initializing_until_ns = (
            now_ns + int(self.config.initial_state_timeout_s * 1e9)
            if deadline_ns is None
            else deadline_ns
        )

    def consume(self, state, now_ns):
        if state is None:
            if not self.read_crc:
                logger.warning("XHand read CRC; holding previous feedback within age budget")
            self.read_crc = True
            # Do not feed cached values back into the update function: no clock advances.
            return self.checked_feedback(now_ns)
        if self.read_crc:
            logger.info("XHand trustworthy reads recovered")
        self.read_crc = False
        errors = tuple(
            tuple(getattr(state, name))
            for name in ("commboard_err", "jointboard_err", "tipboard_err")
        )
        if errors != self.previous_errors and any(any(row) for row in errors):
            logger.warning(
                "XHand board errors: %s",
                ", ".join(
                    f"{board}[{index}]={code}"
                    for board, row in zip(("comm", "joint", "tip"), errors)
                    for index, code in enumerate(row)
                    if code
                ),
            )
        self.previous_errors = errors
        frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
        frame["qpos"], frame["current"] = state.qpos, state.current_ma
        for name, (values, valid_ns) in self.tactile_cache.items():
            held = update_tactile_cache(getattr(state, f"tactile_{name}"), values, valid_ns, now_ns)
            if held and name not in self.tactile_held:
                logger.warning("XHand tactile %s incomplete; holding previous finite vectors", name)
                self.tactile_held.add(name)
            elif not held and name in self.tactile_held:
                logger.info("XHand tactile %s recovered", name)
                self.tactile_held.remove(name)
            # Assignment copies into an independent snapshot for recording/model workers.
            frame[f"tactile_{name}"] = values
            frame[f"tactile_{name}_timestamp_ns"] = valid_ns.min()
        frame["timestamp_ns"] = now_ns
        frame.flags.writeable = False
        self.latest = frame
        return self.checked_feedback(now_ns)

    def checked_feedback(self, now_ns):
        frame = self.latest
        max_age_ns = int(self.config.feedback_max_age_s * 1e9)
        issue = "hand joint feedback has no valid initial value"
        if frame is not None:
            if not np.isfinite(frame["qpos"]).all():
                raise RuntimeError("hand joint feedback is nonfinite")
            issue = None
            for name in (
                "timestamp_ns",
                *(f"tactile_{channel}_timestamp_ns" for channel in self.required_tactile),
            ):
                stamp = int(frame[name][0])
                if not 0 < stamp <= now_ns < stamp + max_age_ns:
                    issue = (
                        f"hand {name}: no valid initial value or feedback age reached "
                        f"{self.config.feedback_max_age_s:.3f}s"
                    )
                    break
        # Idle feedback may recover after a cancelled wait; the deadline only bounds missing values.
        if issue is not None and self.initializing_until_ns is not None:
            if now_ns >= self.initializing_until_ns:
                raise RuntimeError(
                    f"XHand initial feedback timed out after {self.config.initial_state_timeout_s:.3f}s: "
                    f"{issue}"
                )
            return None
        if issue is not None:
            raise RuntimeError(issue)
        self.initializing_until_ns = None
        return frame
