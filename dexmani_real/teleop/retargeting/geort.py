"""GeoRT 推理及 XHand 遥操作目标约束。"""

import numpy as np

from dexmani_real.robot.commands import HandTargetRejected, validate_hand_qpos
from dexmani_real.teleop.retargeting.base import HandRetargeterBase
from dexmani_real.teleop.retargeting.geort_model import GeoRTModel
from dexmani_real.teleop.retargeting.landmarks import hand_lengths, prepare_geort_landmarks


class GeoRTRetargeter(HandRetargeterBase):
    prepare_landmarks = staticmethod(prepare_geort_landmarks)

    def __init__(self, config, hand_config):
        super().__init__(config, hand_config)
        self.model = GeoRTModel(config.geort_checkpoint_dir)
        self.checkpoint_metadata = {
            "weights_sha256": self.model.weights_sha256,
            "config_sha256": self.model.config_sha256,
        }
        self.last_prediction_rad = None

    def reset(self, initial_qpos: np.ndarray, *, timestamp_ns: int) -> None:
        self.last_prediction_rad = None
        super().reset(initial_qpos, timestamp_ns=timestamp_ns)

    def retarget(self, landmarks: np.ndarray, *, timestamp_ns: int) -> np.ndarray | None:
        self.last_prediction_rad = None
        return super().retarget(landmarks, timestamp_ns=timestamp_ns)

    def _retarget(self, points: np.ndarray, qpos: np.ndarray, dt_s: float, alpha: float) -> np.ndarray:
        self.calibration.validate_lengths(hand_lengths(points), tolerance=self.config.bone_length_tolerance)
        prediction = self.model.predict(points)
        try:
            self.last_prediction_rad = validate_hand_qpos(prediction)
        except (TypeError, ValueError) as exc:
            raise HandTargetRejected(f"GeoRT prediction rejected: {exc}") from exc
        lower, upper = self.limits.T
        target = np.clip(self.last_prediction_rad, lower, upper)
        smoothed = target if alpha == 1.0 else qpos + alpha * (target - qpos)
        step = self.config.max_joint_speed_rad_s * dt_s
        return np.clip(smoothed, np.maximum(lower, qpos - step), np.minimum(upper, qpos + step))
