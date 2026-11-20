"""居中 Mesh 原点在相机系下的短时 XY 运动预测。"""

import numpy as np

_MAX_GAP_NS = 150_000_000
_POSITION_STD = 0.005
_VELOCITY_STD = 0.3
_ACCELERATION_STD = 1.5


class KalmanXY:
    """对 x、y 分别使用同一匀速模型，状态按位置、速度两行排列。"""

    def __init__(self, xy: np.ndarray, timestamp_ns: int):
        self._reset(xy, timestamp_ns)

    def _reset(self, xy: np.ndarray, timestamp_ns: int) -> None:
        self._state = np.stack((xy, np.zeros(2, dtype=np.float64)))
        self._covariance = np.diag([_POSITION_STD**2, _VELOCITY_STD**2])
        self._timestamp_ns = timestamp_ns
        self._has_velocity = False

    def predict(self, timestamp_ns: int) -> np.ndarray | None:
        """返回候选初值，已验证的滤波状态只由 update 推进。"""
        elapsed_ns = timestamp_ns - self._timestamp_ns
        if not self._has_velocity or not 0 < elapsed_ns <= _MAX_GAP_NS:
            return None
        dt = elapsed_ns / 1e9
        position, velocity = self._state
        return position + dt * velocity

    def update(self, xy: np.ndarray, timestamp_ns: int) -> None:
        elapsed_ns = timestamp_ns - self._timestamp_ns
        if elapsed_ns <= 0:
            return
        if elapsed_ns > _MAX_GAP_NS:
            self._reset(xy, timestamp_ns)
            return

        dt = elapsed_ns / 1e9
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        acceleration = _ACCELERATION_STD * np.array([0.5 * dt**2, dt])
        covariance = (
            transition @ self._covariance @ transition.T + np.outer(acceleration, acceleration)
        )
        predicted = transition @ self._state
        gain = covariance[:, 0] / (covariance[0, 0] + _POSITION_STD**2)
        self._state = predicted + np.outer(gain, xy - predicted[0])
        # 位置是唯一观测，两个轴使用相同的噪声和增益。
        residual = np.eye(2)
        residual[:, 0] -= gain
        self._covariance = (
            residual @ covariance @ residual.T + _POSITION_STD**2 * np.outer(gain, gain)
        )
        self._timestamp_ns = timestamp_ns
        self._has_velocity = True
