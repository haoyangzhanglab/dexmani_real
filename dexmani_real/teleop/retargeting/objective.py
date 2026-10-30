"""Wuji 式手形/末节方向联合拟合，加连续对指距离和时间阻尼。"""

import numpy as np


def huber(error, delta):
    """各行向量的径向 Huber；返回逐行损失与对 error 的导数。"""
    norm = np.linalg.norm(error, axis=-1)
    factor = np.minimum(1.0, delta / np.maximum(norm, 1e-15))
    value = np.where(norm <= delta, 0.5 * norm**2, delta * (norm - 0.5 * delta))
    return value, factor[..., None] * error


class HandObjective:
    def __init__(self, fk, frame_ids, targets, previous, dt_s, config):
        self.fk, self.frame_ids = fk, frame_ids
        self.positions, self.directions, self.activation, self.pinch_distances = targets
        self.previous, self.dt_s, self.config = previous, dt_s, config
        # Thumb 同时服务多个对指目标，不采用非光滑的 argmax 伙伴切换。
        finger_activation = np.r_[1.0 - np.prod(1.0 - self.activation), self.activation]
        self.position_weights = np.zeros((5, 3))
        self.position_weights[:, :2] = (
            config.shape_weight * (1.0 - 0.8 * finger_activation[:, None])
        )
        self.position_weights[:, 2] = config.tip_weight
        # 非拇指 MCP 大多固定；仅拟合可表达的 PIP/TIP，避免常数项影响停止判据。
        self.position_weights[1:, 0] = 0.0

    def __call__(self, qpos, gradient):
        cfg = self.config
        if gradient.size:
            points, jac = self.fk.positions_and_jacobians(qpos, self.frame_ids)
            jac = jac.reshape(5, 3, 3, 12)
        else:
            points = self.fk.forward(qpos, self.frame_ids)[:, :3, 3]
        points = points.reshape(5, 3, 3)
        scale, delta = cfg.position_scale_m, cfg.huber_delta_m / cfg.position_scale_m
        error = (points - self.positions) / scale
        losses, derivatives = huber(error, delta)
        value = np.sum(self.position_weights * losses) / 5.0
        if gradient.size:
            grad = np.einsum(
                "fp,fpc,fpcq->q", self.position_weights, derivatives, jac,
            ) / (5.0 * scale)

        terminal = points[:, 2] - points[:, 1]
        length = np.linalg.norm(terminal, axis=1)
        direction = terminal / length[:, None]
        direction_error = direction - self.directions
        value += 0.5 * cfg.direction_weight * np.sum(direction_error**2) / 5.0
        if gradient.size:
            # d(v/|v|)/dq = (I-dd^T) Jv / |v|，含方向归一化的完整链式法则。
            direction_grad = (
                direction_error - direction * np.sum(direction * direction_error, axis=1)[:, None]
            ) / length[:, None]
            grad += cfg.direction_weight * np.einsum(
                "fc,fcq->q", direction_grad, jac[:, 2] - jac[:, 1],
            ) / 5.0

        pair = points[1:, 2] - points[0, 2]
        # 可微范数只用于优化器，不把 epsilon 注入人手角度或尺度估计。
        distance = np.sqrt(np.sum(pair**2, axis=1) + 1e-12)
        error = ((distance - self.pinch_distances) / scale)[:, None]
        losses, derivatives = huber(error, delta)
        weights = cfg.pinch_weight * self.activation**2 / 4.0
        value += np.dot(weights, losses)
        if gradient.size:
            grad += np.einsum(
                "f,fc,fcq->q", weights * derivatives[:, 0] / scale,
                pair / distance[:, None], jac[1:, 2] - jac[0, 2],
            )

        difference = qpos - self.previous
        damping = cfg.damping_s / self.dt_s
        value += 0.5 * damping * np.dot(difference, difference)
        if gradient.size:
            gradient[:] = grad + damping * difference
        return float(value)
