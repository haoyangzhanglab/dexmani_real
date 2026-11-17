"""XHand 近段/末段联合方向拟合，位置与有意对指作为辅助。"""

import numpy as np

POSITION_SCALE_M = 0.10  # 固定量纲尺度，避免与位置权重形成重复调参自由度。


def huber(error, delta, *, compute_gradient=True):
    """各行向量的径向 Huber；未请求导数时返回 (loss, None)。"""
    # 直接归约平方范数，供二次分支复用，避免先开方再平方。
    squared = np.einsum("...i,...i->...", error, error)
    norm = np.sqrt(squared)
    value = np.where(norm <= delta, 0.5 * squared, delta * (norm - 0.5 * delta))
    derivative = None
    if compute_gradient:
        derivative = (delta / np.maximum(norm, delta))[..., None] * error
    return value, derivative


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
        self.pinch_enabled = bool(np.any(self.pinch_weights))
        self.position_weights = np.zeros((5, 3))
        self.position_weights[:, :2] = config.shape_weight
        self.position_weights[:, 2] = config.tip_weight
        # Thumb/Index 基部可动；其余固定基座不产生常数损失。
        self.position_weights[2:, 0] = 0.0

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
        losses, derivatives = huber(error, delta, compute_gradient=bool(gradient.size))
        value = np.einsum("fp,fp->", self.position_weights, losses) / 5.0
        if gradient.size:
            grad = np.einsum(
                "fp,fpc,fpcq->q", self.position_weights, derivatives, jac,
            ) / (5.0 * scale)

        segments = points[:, 1:] - points[:, :-1]
        length = np.sqrt(np.einsum("fsc,fsc->fs", segments, segments))
        direction = segments / length[:, :, None]
        direction_error = direction - targets.directions
        # 速率边界和时间阻尼仍会产生跟踪残差；保留既有稳健损失。
        # 按弦长做径向 Huber，除以 delta 后大残差的斜率为 1；参考方向不变。
        losses, derivatives = huber(
            direction_error, cfg.direction_huber_delta, compute_gradient=bool(gradient.size),
        )
        value += np.einsum("s,fs->", self.direction_weights, losses) / 5.0
        if gradient.size:
            # d(v/|v|)/dq = (I-dd^T) Jv / |v|，含方向归一化的完整链式法则。
            direction_grad = (
                derivatives - direction * np.einsum("fsc,fsc->fs", direction, derivatives)[:, :, None]
            ) / length[:, :, None]
            grad += np.einsum("s,fsc,fscq->q", self.direction_weights, direction_grad,
                              jac[:, 1:] - jac[:, :-1]) / 5.0

        if self.pinch_enabled:
            pair = points[1:, 2] - points[0, 2]
            # 可微范数只用于优化器，不把 epsilon 注入人手角度或尺度估计。
            distance = np.sqrt(np.einsum("fc,fc->f", pair, pair) + 1e-12)
            error = ((distance - targets.pinch_distances) / scale)[:, None]
            losses, derivatives = huber(error, delta, compute_gradient=bool(gradient.size))
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
