"""Null-space optimization for xArm7 redundant DOF resolution.

The xArm7 is a 7-DOF arm with a 6-DOF EEF task, leaving 1 redundant DOF.
The null-space projector N = I - J⁺J maps joint adjustments into the
self-motion manifold — EEF pose unchanged by construction (J · dq_null = 0).

Reference: Liegeois 1977 gradient projection method.
"""

from __future__ import annotations

import numpy as np


def nullspace_projector(J: np.ndarray) -> np.ndarray:
    """Compute null-space projector N = I - J⁺J via SVD.

    For the xArm7 6×7 Jacobian (rank 6): N is 7×7, symmetric, idempotent,
    with one eigenvalue ≈ 1 (null-space direction) and six ≈ 0 (range-space).

    Args:
        J: 6×dof end-effector Jacobian.  Must have full row rank (6).

    Returns:
        7×7 null-space projector matrix.
    """
    # SVD: J = U S V^T  →  V is 7×6 (right singular vectors for non-zero singular values)
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    V = Vt.T  # 7×6
    # N = I - V V^T — numerically stable; eigenvalues exactly 0 or 1.
    return np.eye(J.shape[1]) - V @ V.T


def joint_limit_gradient(
    qpos: np.ndarray,
    joint_limits: np.ndarray,
    margin_deg: float = 15.0,
) -> np.ndarray:
    """Quadratic repulsive gradient from joint limits.

    Gradient is zero when distance ≥ margin (joint is safely away from limit).
    Within the margin, the gradient increases linearly, pushing the joint
    toward centre.  C¹ continuous — no discontinuous joint motion at the
    margin boundary.

    Potential: V(q) = ((margin - d) / margin)²  for d < margin, else 0.
    Gradient: ∂V/∂q = -2(margin - d)/margin² · sign(direction).

    Args:
        qpos: current joint positions [rad], shape (dof,).
        joint_limits: (dof, 2) array [low, high] per joint [rad].
        margin_deg: distance from limit [deg] below which repulsion activates.

    Returns:
        Gradient vector ∇V(q), shape (dof,).  NaN-safe: returns zeros on
        non-finite input.
    """
    if not np.all(np.isfinite(qpos)):
        return np.zeros_like(qpos)

    margin = np.deg2rad(margin_deg)
    low = joint_limits[:, 0]
    high = joint_limits[:, 1]
    grad = np.zeros(qpos.shape[0], dtype=np.float64)

    for i in range(qpos.shape[0]):
        d_low = qpos[i] - low[i]
        d_high = high[i] - qpos[i]

        if d_low < margin:
            # Push positive — joint too close to lower limit
            grad[i] = 2.0 * (margin - d_low) / (margin * margin)
        elif d_high < margin:
            # Push negative — joint too close to upper limit
            grad[i] = -2.0 * (margin - d_high) / (margin * margin)

    return grad


def apply_nullspace_optimization(
    qpos: np.ndarray,
    jacobian: np.ndarray,
    joint_limits: np.ndarray,
    step_size_rad: float = np.deg2rad(1.0),
    margin_deg: float = 15.0,
) -> np.ndarray:
    """Apply null-space joint-limit repulsion to an IK solution.

    Computes the null-space projector from the Jacobian, projects the
    joint-limit repulsive gradient into the self-motion manifold, clips
    the step to ``step_size_rad``, and returns the adjusted qpos.

    The EEF pose is unchanged by construction: J @ (qpos' - qpos) ≈ 0.

    Args:
        qpos: IK solution to refine [rad], shape (7,).
        jacobian: 6×7 EEF Jacobian at qpos.
        joint_limits: (7, 2) array [low, high] per joint [rad].
        step_size_rad: max per-frame null-space step [rad] (default 1°).
        margin_deg: joint-limit margin for repulsion [deg].

    Returns:
        Refined qpos with null-space adjustment applied.
    """
    grad = joint_limit_gradient(qpos, joint_limits, margin_deg)
    if not np.any(grad):
        return qpos  # no joint near limit — skip SVD (~0.13 ms saved)

    N = nullspace_projector(jacobian)
    dq = N @ grad
    dq_max = float(np.max(np.abs(dq)))

    if dq_max > step_size_rad and dq_max > 1e-12:
        dq *= step_size_rad / dq_max

    return qpos + dq
