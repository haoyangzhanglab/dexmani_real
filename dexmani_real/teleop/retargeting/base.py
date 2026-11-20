"""手部重定向共用的标定、时间步与已确认命令参考。"""

from abc import ABC, abstractmethod
from math import expm1

import numpy as np

from dexmani_real.control.actions import StartPoseRejected
from dexmani_real.robot.commands import (
    HandTargetRejected,
    hand_reference_from_feedback,
    validate_hand_target,
)
from dexmani_real.teleop.retargeting.landmarks import (
    HandCalibration,
    HandGeometryRejected,
    hand_lengths,
    prepare_landmarks,
)


class HandRetargeterBase(ABC):
    prepare_landmarks = staticmethod(prepare_landmarks)

    def __init__(self, config, hand_config):
        config.validate()
        self.config = config
        self.limits = np.column_stack((hand_config.qpos_min_rad, hand_config.qpos_max_rad))
        self.calibration: HandCalibration | None = None
        self.reference: tuple[np.ndarray, int] | None = None
        self.rejection_reason = None

    def calibration_measurement(self, landmarks: np.ndarray) -> np.ndarray:
        return hand_lengths(self.prepare_landmarks(landmarks))

    def calibrate(self, measurements) -> None:
        self.calibration = HandCalibration.from_measurements(measurements, self.config)

    def reset(self, initial_qpos: np.ndarray, *, timestamp_ns: int) -> None:
        if timestamp_ns <= 0:
            raise ValueError("hand reset timestamp must be positive")
        try:
            qpos = hand_reference_from_feedback(initial_qpos, *self.limits.T)
        except HandTargetRejected as exc:
            raise StartPoseRejected(f"手部实测参考不满足要求，请先就位后再次按 B。{exc}") from exc
        self.reference = qpos, int(timestamp_ns)
        self.rejection_reason = None

    def retarget(self, landmarks: np.ndarray, *, timestamp_ns: int) -> np.ndarray | None:
        if self.calibration is None or self.reference is None:
            raise RuntimeError("hand retargeting requires calibration and a measured reset pose")
        qpos, reference_ns = self.reference
        if timestamp_ns < reference_ns:
            raise ValueError("hand computation time moved backwards")
        dt_s = (timestamp_ns - reference_ns) * 1e-9
        self.rejection_reason = None
        if dt_s == 0:
            return qpos.copy()
        tau = self.config.smoothing_tau_s
        alpha = 1.0 if tau == 0 else -expm1(-dt_s / tau)
        try:
            points = self.prepare_landmarks(landmarks)
            command = self._retarget(points, qpos, dt_s, alpha)
            if command is not None:
                return validate_hand_target(command, *self.limits.T)
        except (HandGeometryRejected, HandTargetRejected) as exc:
            self.rejection_reason = str(exc)
        return None

    @abstractmethod
    def _retarget(
        self, points: np.ndarray, qpos: np.ndarray, dt_s: float, alpha: float,
    ) -> np.ndarray | None:
        """从当前掌系关键点生成以确认参考为起点的目标。"""

    def commit(self, qpos: np.ndarray, *, timestamp_ns: int) -> None:
        # 控制线程仅在发送正常返回后更新最后提交参考。
        if self.reference is None or timestamp_ns < self.reference[1]:
            raise ValueError("committed hand time moved backwards or reset is missing")
        self.reference = qpos.copy(), int(timestamp_ns)
