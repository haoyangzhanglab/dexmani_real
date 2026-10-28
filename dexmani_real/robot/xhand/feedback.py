"""XHand 标准化状态、触觉缓存与反馈时效。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import RobotError
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
)
from dexmani_real.robot.xhand.config import INITIAL_STATE_TIMEOUT_S

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


@dataclass
class XHandState:
    """关节数据可信，触觉仍可能包含无效值。"""

    qpos: np.ndarray
    tactile_aggregate: np.ndarray
    tactile_dense: np.ndarray


def update_tactile_cache(raw, values, valid_ns, source_ns):
    """按三轴向量整体更新，仅新鲜有限值推进来源时间。"""
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
        self.initializing_until_ns = None

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
        self.initializing_until_ns = (
            now_ns + int(INITIAL_STATE_TIMEOUT_S * 1e9) if deadline_ns is None else deadline_ns
        )

    def consume(self, state, *, source_ns, now_ns):
        if state is None:
            return self.checked_feedback(now_ns)

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
        if frame is not None:
            stamp = int(frame["timestamp_ns"][0])
            if 0 < stamp <= now_ns < stamp + max_age_ns:
                self.initializing_until_ns = None
                return frame

        if self.initializing_until_ns is not None:
            if now_ns < self.initializing_until_ns:
                return None
            raise RobotError("XHand initial feedback timed out: no fresh joint state")
        raise RobotError("XHand joint feedback unavailable/stale")
