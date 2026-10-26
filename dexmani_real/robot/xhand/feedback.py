"""XHand 载荷解析及保留原始有效时间的触觉缓存。"""

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
from dexmani_real.robot.xhand.config import (
    INITIAL_STATE_TIMEOUT_S,
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        # 每种模态取最早的原始有效时间，逐 taxel 时间仅在本地保存。
        ("tactile_aggregate_timestamp_ns", "<u8"),
        ("tactile_dense_timestamp_ns", "<u8"),
        # 关节来源时间独立于触觉缓存。
        ("timestamp_ns", "<u8"),
    ]
)

logger = logging.getLogger(__name__)

COMBINED_FORCE_UNAVAILABLE_CODE = 1_501_018
DISTRIBUTED_FORCE_UNAVAILABLE_CODE = 1_501_019
TEMPERATURE_UNAVAILABLE_CODE = 1_501_020


def tactile_validity(code: int, *, comm_type: str) -> tuple[bool, bool]:
    """RS485 201 仅禁用 dense、202 不禁用触觉；EtherCAT 要求状态为零。"""
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
    """关节数据可信，触觉仍可能包含无效值。"""

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
        # SDK 字段名实际拼写为 jonitboard_err。
        errors["jointboard_err"][index] = int(joint.jonitboard_err)
        errors["tipboard_err"][index] = int(joint.tipboard_err)
    if len(seen) != HAND_DOF:
        raise ValueError(f"{len(seen)}/{HAND_DOF} joints reported")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("non-finite joint position feedback")
    current[~np.isfinite(current)] = np.nan
    return qpos, current, errors


def parse_tactile_aggregate(state: Any) -> np.ndarray:
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
    """按三轴向量整体更新，仅新鲜有限值推进来源时间。"""
    # 有限 float64 仍可能溢出 float32，按最终观测 dtype 判定有效性。
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(raw, dtype=values.dtype)
    if raw.shape != values.shape:
        raise RuntimeError(f"invalid tactile shape: {raw.shape}, expected {values.shape}")
    valid = np.isfinite(raw).all(axis=-1)
    np.copyto(values, raw, where=valid[..., None])
    np.copyto(valid_ns, source_ns, where=valid)
    return not valid.all()


class XHandFeedback:
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
            now_ns + int(INITIAL_STATE_TIMEOUT_S * 1e9) if deadline_ns is None else deadline_ns
        )

    def consume(self, state, *, source_ns, now_ns):
        if state is None:
            if not self.read_crc:
                logger.warning("XHand read CRC; holding previous feedback within age budget")
            self.read_crc = True
            # 缓存数据不重新参与更新，避免推进来源时间。
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
            held = update_tactile_cache(
                getattr(state, f"tactile_{name}"), values, valid_ns, source_ns
            )
            if held and name not in self.tactile_held:
                logger.warning("XHand tactile %s incomplete; holding previous finite vectors", name)
                self.tactile_held.add(name)
            elif not held and name in self.tactile_held:
                logger.info("XHand tactile %s recovered", name)
                self.tactile_held.remove(name)

            frame[f"tactile_{name}"] = values
            frame[f"tactile_{name}_timestamp_ns"] = valid_ns.min()
        frame["timestamp_ns"] = source_ns
        frame.flags.writeable = False
        self.latest = frame
        return self.checked_feedback(now_ns)

    def checked_feedback(self, now_ns):
        frame = self.latest
        max_age_ns = int(self.config.feedback_max_age_s * 1e9)
        issue: str | None = "hand joint feedback has no valid initial value"
        if frame is not None:
            issue = None
            stamp = int(frame["timestamp_ns"][0])
            if not 0 < stamp <= now_ns < stamp + max_age_ns:
                issue = (
                    "hand timestamp_ns: no valid initial value or feedback age reached "
                    f"{self.config.feedback_max_age_s:.3f}s"
                )
        # 取消初始化后仍允许空闲反馈恢复，截止时间只约束缺失值。
        if issue is not None and self.initializing_until_ns is not None:
            if now_ns >= self.initializing_until_ns:
                raise RuntimeError(
                    f"XHand initial feedback timed out after {INITIAL_STATE_TIMEOUT_S:.3f}s: "
                    f"{issue}"
                )
            return None
        if issue is not None:
            raise RuntimeError(issue)
        self.initializing_until_ns = None
        return frame
