"""XHand SDK payload decoding, bounded feedback and tactile-vector hold; no device I/O."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    HAND_DOF,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        # Oldest original valid vector per modality; individual taxel clocks remain local.
        ("tactile_aggregate_timestamp_ns", "<u8"),
        ("tactile_dense_timestamp_ns", "<u8"),
        # Last trustworthy joint read, independent of held tactile values.
        ("timestamp_ns", "<u8"),
    ]
)

logger = logging.getLogger(__name__)

COMBINED_FORCE_UNAVAILABLE_CODE = 1_501_018
DISTRIBUTED_FORCE_UNAVAILABLE_CODE = 1_501_019
TEMPERATURE_UNAVAILABLE_CODE = 1_501_020


def tactile_validity(code: int, *, comm_type: str) -> tuple[bool, bool]:
    """RS485 1501019/201 drops dense only; 1501020/202 keeps both; EtherCAT rejects nonzero status."""
    if comm_type == "ethercat":
        return (code == 0, code == 0)
    if code == 0:
        return (True, True)
    if code in (COMBINED_FORCE_UNAVAILABLE_CODE, 200):
        return (False, False)
    if code in (DISTRIBUTED_FORCE_UNAVAILABLE_CODE, 201):
        return (True, False)
    if code in (TEMPERATURE_UNAVAILABLE_CODE, 202):
        return (True, True)
    return (False, False)


@dataclass
class XHandState:
    """One trustworthy joint read; tactile may contain NaN/Inf or status-masked NaN."""

    qpos: np.ndarray
    current_ma: np.ndarray
    tactile_aggregate: np.ndarray
    tactile_dense: np.ndarray
    commboard_err: np.ndarray
    jointboard_err: np.ndarray
    tipboard_err: np.ndarray


def parse_joints(
    state: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    qpos = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
    current = np.full(HAND_JOINT_SHAPE, np.nan, dtype=np.float64)
    errors = {
        name: np.zeros(HAND_JOINT_SHAPE, dtype=np.int32)
        for name in ("commboard_err", "jointboard_err", "tipboard_err")
    }
    seen: set[int] = set()
    for joint in state.finger_state:
        index = int(joint.id)
        if index < 0 or index >= HAND_DOF or index in seen:
            raise ValueError(f"invalid or duplicate joint id {index}")
        seen.add(index)
        qpos[index] = float(joint.position)
        current[index] = float(getattr(joint, "torque", np.nan))
        errors["commboard_err"][index] = int(joint.commboard_err)
        # The vendor SDK spells this field jonitboard_err.
        errors["jointboard_err"][index] = int(joint.jonitboard_err)
        errors["tipboard_err"][index] = int(joint.tipboard_err)
    if len(seen) != HAND_DOF:
        raise ValueError(f"{len(seen)}/{HAND_DOF} joints reported")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("non-finite joint position feedback")
    current[~np.isfinite(current)] = np.nan
    return qpos, current, errors


def parse_tactile_aggregate(state: Any) -> np.ndarray:
    """Return the SDK-native ``[5,3]`` aggregate ``calc_force`` payload."""
    force_aggregate = np.asarray(
        [
            [sensor.calc_force.fx, sensor.calc_force.fy, sensor.calc_force.fz]
            for sensor in state.sensor_data
        ],
        dtype=np.float64,
    )
    if force_aggregate.shape != HAND_TACTILE_SUM_SHAPE:
        raise ValueError("aggregate tactile must have shape [5,3]")
    return force_aggregate


def parse_tactile_dense(state: Any) -> np.ndarray:
    """Return the SDK-native ``[5,120,3]`` dense ``raw_force`` payload."""
    tactile_dense = np.asarray(
        [
            [[force.fx, force.fy, force.fz] for force in sensor.raw_force]
            for sensor in state.sensor_data
        ],
        dtype=np.float64,
    )
    if tactile_dense.shape != HAND_TACTILE_FORCE_SHAPE:
        raise ValueError("dense tactile must have shape [5,120,3]")
    return tactile_dense


def update_tactile_cache(raw, values, valid_ns, source_ns):
    """Hold whole three-axis vectors; only finite new vectors advance their clocks."""
    # Validate in the observation dtype too: finite float64 may overflow float32.
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(raw, dtype=values.dtype)
    if raw.shape != values.shape:
        raise RuntimeError(f"invalid tactile shape: {raw.shape}, expected {values.shape}")
    valid = np.isfinite(raw).all(axis=-1)
    np.copyto(values, raw, where=valid[..., None])
    np.copyto(valid_ns, source_ns, where=valid)
    return not valid.all()


class XHandFeedback:
    """Own consumed samples and their validity clocks within one robot connection."""

    def __init__(self, config):
        self.config = config
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

    def consume(self, state, *, source_ns, now_ns):
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
        frame["qpos"] = state.qpos
        for name, (values, valid_ns) in self.tactile_cache.items():
            held = update_tactile_cache(getattr(state, f"tactile_{name}"), values, valid_ns, source_ns)
            if held and name not in self.tactile_held:
                logger.warning("XHand tactile %s incomplete; holding previous finite vectors", name)
                self.tactile_held.add(name)
            elif not held and name in self.tactile_held:
                logger.info("XHand tactile %s recovered", name)
                self.tactile_held.remove(name)
            # Assignment copies into an independent snapshot for recording/model workers.
            frame[f"tactile_{name}"] = values
            frame[f"tactile_{name}_timestamp_ns"] = valid_ns.min()
        frame["timestamp_ns"] = source_ns
        frame.flags.writeable = False
        self.latest = frame
        return self.checked_feedback(now_ns)

    def checked_feedback(self, now_ns):
        frame = self.latest
        max_age_ns = int(self.config.feedback_max_age_s * 1e9)
        issue = "hand joint feedback has no valid initial value"
        if frame is not None:
            issue = None
            stamp = int(frame["timestamp_ns"][0])
            if not 0 < stamp <= now_ns < stamp + max_age_ns:
                issue = (
                    "hand timestamp_ns: no valid initial value or feedback age reached "
                    f"{self.config.feedback_max_age_s:.3f}s"
                )
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
