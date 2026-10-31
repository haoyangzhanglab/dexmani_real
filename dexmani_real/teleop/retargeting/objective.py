"""XHand 近段/末段联合方向拟合，位置与有意对指作为辅助。"""

import numpy as np

POSITION_SCALE_M = 0.10  # 固定量纲尺度，避免与位置权重形成重复调参自由度。


def huber(error, delta):
    """各行向量的径向 Huber；返回逐行损失与对 error 的导数。"""
    norm = np.linalg.norm(error, axis=-1)
    factor = delta / np.maximum(norm, delta)
    value = np.where(norm <= delta, 0.5 * norm**2, delta * (norm - 0.5 * delta))
    return value, factor[..., None] * error


class HandObjective:
    def __init__(self, fk, frame_ids, targets, previous, dt_s, config):
        self.fk, self.frame_ids = fk, frame_ids
        self.targets = targets
        self.previous, self.config = previous, config
        self.damping = config.damping_s / dt_s
        self.direction_weights = (
            np.array([config.proximal_direction_weight, config.direction_weight]) / config.direction_huber_delta
        )
        self.pinch_weights = config.pinch_weight * targets.activation**2 / 4.0
        self.position_weights = np.zeros((5, 3))
        self.position_weights[:, :2] = config.shape_weight
        self.position_weights[:, 2] = config.tip_weight
        # 非拇指 MCP 大多固定；仅拟合可表达的 PIP/TIP，避免常数项影响停止判据。
        self.position_weights[1:, 0] = 0.0
        if not targets.thumb_base_observable:
            self.position_weights[0] = 0.0

    def __call__(self, qpos, gradient):
        cfg, targets = self.config, self.targets
        if gradient.size:
            points, jac = self.fk.positions_and_jacobians(qpos, self.frame_ids)
            jac = jac.reshape(5, 3, 3, 12)
        else:
            points = self.fk.forward(qpos, self.frame_ids)[:, :3, 3]
        points = points.reshape(5, 3, 3)
        scale, delta = POSITION_SCALE_M, cfg.huber_delta_m / POSITION_SCALE_M
        error = (points - targets.positions) / scale
        losses, derivatives = huber(error, delta)
        value = np.sum(self.position_weights * losses) / 5.0
        if gradient.size:
            grad = np.einsum(
                "fp,fpc,fpcq->q", self.position_weights, derivatives, jac,
            ) / (5.0 * scale)

        segments = np.diff(points, axis=1)
        length = np.linalg.norm(segments, axis=2)
        direction = segments / length[:, :, None]
        direction_error = direction - targets.directions
        # 机构无法表达的侧摆/末节弯曲不应持续增大对其他关节的牵引。
        # 按弦长做径向 Huber，除以 delta 后大残差的斜率为 1；参考方向不变。
        losses, derivatives = huber(direction_error, cfg.direction_huber_delta)
        value += np.dot(self.direction_weights, np.sum(losses, axis=0)) / 5.0
        if gradient.size:
            # d(v/|v|)/dq = (I-dd^T) Jv / |v|，含方向归一化的完整链式法则。
            direction_grad = (
                derivatives - direction * np.sum(direction * derivatives, axis=2)[:, :, None]
            ) / length[:, :, None]
            grad += np.einsum("s,fsc,fscq->q", self.direction_weights, direction_grad, np.diff(jac, axis=1)) / 5.0

        pair = points[1:, 2] - points[0, 2]
        # 可微范数只用于优化器，不把 epsilon 注入人手角度或尺度估计。
        distance = np.sqrt(np.sum(pair**2, axis=1) + 1e-12)
        error = ((distance - targets.pinch_distances) / scale)[:, None]
        losses, derivatives = huber(error, delta)
        value += np.dot(self.pinch_weights, losses)
        if gradient.size:
            grad += np.einsum(
                "f,fc,fcq->q", self.pinch_weights * derivatives[:, 0] / scale,
                pair / distance[:, None], jac[1:, 2] - jac[0, 2],
            )

        difference = qpos - self.previous
        value += 0.5 * self.damping * np.dot(difference, difference)
        if gradient.size:
            gradient[:] = grad + self.damping * difference
        return float(value)
