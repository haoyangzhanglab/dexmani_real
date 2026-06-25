#!/usr/bin/env python3
"""Internal test utility module — shared math helpers, dataclasses, and sim exec utils.

Consolidates duplicated functions that were scattered across examples/real/
and examples/sim/.  All callers import from here to avoid drift.

Usage:
    from examples._test_utils import quat_multiply, angular_dist_rad, ...
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sapien.core as sapien

from dexmani_real.planning import Pose
from dexmani_real.simulation import SimRobotInterface


# ═══════════════════════════════════════════════════════════════════
#  Quaternion / rotation helpers
# ═══════════════════════════════════════════════════════════════════


def angular_dist_rad(q1: np.ndarray, q2: np.ndarray) -> float:
    """Angular distance between two wxyz quaternions (radians)."""
    return float(2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0)))


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions: q1 ⊗ q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3×3 rotation matrix.

    Delegates to scipy.spatial.transform.Rotation for numerical stability.
    """
    from scipy.spatial.transform import Rotation

    # wxyz → xyzw (scipy convention)
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(q_xyzw).as_matrix().astype(np.float64)


def random_quat_within_angle(rng: np.random.RandomState, max_deg: float) -> np.ndarray:
    """Uniform random rotation quaternion (wxyz) with angle ≤ max_deg."""
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, np.deg2rad(max_deg))
    half = angle / 2
    return np.array([np.cos(half), axis[0] * np.sin(half),
                     axis[1] * np.sin(half), axis[2] * np.sin(half)])


def random_quat_full_so3(rng: np.random.RandomState) -> np.ndarray:
    """Uniformly sample SO(3) full-space random quaternion (wxyz).

    Uses the Marsaglia method (uniform distribution on the unit sphere).
    """
    u = rng.uniform(0, 1, 3)
    q = np.array([
        np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
        np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
        np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
        np.sqrt(u[0]) * np.cos(2 * np.pi * u[2]),
    ])
    q /= np.linalg.norm(q)
    return q


def random_quat_multi_axis(
    rng: np.random.RandomState, max_deg1: float = 45.0, max_deg2: float = 30.0,
) -> np.ndarray:
    """Two successive rotations around independent random axes.

    Produces richer attitude distribution than single-axis rotation.
    Composite rotation = R2 * R1 (applied in that order).
    """
    # Axis 1: random direction
    a1 = rng.randn(3)
    a1 /= np.linalg.norm(a1)
    angle1 = rng.uniform(0, np.deg2rad(max_deg1))
    half1 = angle1 / 2
    q1 = np.array([np.cos(half1), a1[0] * np.sin(half1),
                   a1[1] * np.sin(half1), a1[2] * np.sin(half1)])

    # Axis 2: orthogonal to axis 1
    a2 = rng.randn(3)
    a2 -= a1 * np.dot(a2, a1)
    norm = np.linalg.norm(a2)
    if norm < 1e-10:
        a2 = np.array([-a1[1], a1[0], 0.0]) if abs(a1[0]) > 1e-10 else np.array([1.0, 0.0, 0.0])
        a2 -= a1 * np.dot(a2, a1)
    a2 /= np.linalg.norm(a2)
    angle2 = rng.uniform(0, np.deg2rad(max_deg2))
    half2 = angle2 / 2
    q2 = np.array([np.cos(half2), a2[0] * np.sin(half2),
                   a2[1] * np.sin(half2), a2[2] * np.sin(half2)])

    return quat_multiply(q2, q1)  # R2 * R1


# ═══════════════════════════════════════════════════════════════════
#  Target pose builder
# ═══════════════════════════════════════════════════════════════════


_ROT_MODE_DOC = """
    Rotation modes:
      "fixed"        — keep home orientation unchanged
      "single_axis"  — rotate around one random axis ≤ rot_max_deg
      "multi_axis"   — two independent axes (≤ rot_ax1_deg, ≤ rot_ax2_deg)
      "full_so3"     — uniform SO(3) sampling (IK success rate will be lower)
"""


def build_target_pose(
    pos: np.ndarray,
    home_quat: np.ndarray,
    rng: "np.random.RandomState | None" = None,
    *,
    rot_mode: str = "single_axis",
    rot_max_deg: float = 30.0,
    rot_axis1_deg: float = 45.0,
    rot_axis2_deg: float = 30.0,
) -> Pose:
    """Build a target EEF pose with optional random rotation.

    Args:
        pos:        (3,) target position in world frame
        home_quat:  (4,) wxyz quaternion of the home orientation
        rng:        RandomState for reproducibility; None = no rotation
        rot_mode:   one of "fixed" / "single_axis" / "multi_axis" / "full_so3"
        rot_max_deg:      max angle for single_axis mode
        rot_axis1_deg:    max angle for first axis in multi_axis mode
        rot_axis2_deg:    max angle for second axis in multi_axis mode

    Returns: Pose(p=pos, q=<rotated quat>)
    """ + _ROT_MODE_DOC
    quat = home_quat
    if rng is None:
        return Pose(p=pos, q=quat)

    if rot_mode == "full_so3":
        quat = random_quat_full_so3(rng)
    elif rot_mode == "multi_axis":
        delta_q = random_quat_multi_axis(rng, rot_axis1_deg, rot_axis2_deg)
        quat = quat_multiply(delta_q, home_quat)
    elif rot_mode == "single_axis" and rot_max_deg > 0:
        quat = quat_multiply(random_quat_within_angle(rng, rot_max_deg), home_quat)
    # else: "fixed" — keep home quat

    return Pose(p=pos, q=quat)


# ═══════════════════════════════════════════════════════════════════
#  Joint path interpolation
# ═══════════════════════════════════════════════════════════════════


def interpolate_waypoints(path: np.ndarray, max_step: float) -> np.ndarray:
    """Linearly densify a sparse joint path so each step ≤ max_step rad.

    Args:
        path:     (W, J) array of joint-space waypoints
        max_step: maximum joint change per step (radians)

    Returns: (D, J) dense path (D ≥ W)
    """
    if len(path) <= 1:
        return path.astype(np.float64)
    dense = [path[0]]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        n = int(np.ceil(float(np.max(np.abs(b - a))) / max_step))
        for k in range(1, n + 1):
            dense.append(a + (k / n) * (b - a))
    return np.array(dense, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════
#  Simulation execution helpers (SAPIEN only)
# ═══════════════════════════════════════════════════════════════════


def execute_dense_path(
    sim: SimRobotInterface,
    dense: np.ndarray,
    viewer: "sapien.Viewer | None" = None,
    physics_steps_per_wp: int = 20,
) -> bool:
    """Execute a pre-densified arm joint path (N, 7) in simulation.

    Hand joints are held constant throughout.

    Returns:
        True if the entire path was executed (False if viewer closed).
    """
    assert dense.ndim == 2 and dense.shape[1] == 7
    hand = sim.get_full_qpos()[7:]
    for wp in dense:
        if viewer is not None and viewer.closed:
            return False
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([wp, hand]))
        sim._step_physics(n=physics_steps_per_wp)
        if viewer is not None:
            sim.scene.update_render()
            viewer.render()
    return True


def settle_at_target(
    sim: SimRobotInterface,
    target_arm: np.ndarray,
    hand_qpos: np.ndarray,
    max_iter: int = 30,
    converge_threshold_rad: float = np.deg2rad(0.05),
    physics_steps_per_wp: int = 20,
) -> float:
    """Closed-loop PD convergence to target arm joint angles.

    Drives the PD controller iteratively until max joint error falls below
    converge_threshold_rad (analogous to the real arm's blocking
    ``arm.reset(wait=True)``).

    Args:
        sim:                    simulation interface handle
        target_arm:             (7,) target arm joint angles
        hand_qpos:              (12,) hand joint angles (held constant)
        max_iter:               max PD iterations before giving up
        converge_threshold_rad: stop when max |joint_error| < this
        physics_steps_per_wp:   physics steps per PD iteration

    Returns:
        Final max joint error (radians).
    """
    for _ in range(max_iter):
        sim.robot.balance_passive_force()
        sim.robot.apply_action(np.concatenate([target_arm, hand_qpos]))
        sim._step_physics(n=physics_steps_per_wp)
        current = sim.get_full_qpos()[:7]
        err = float(np.max(np.abs(current - target_arm)))
        if err < converge_threshold_rad:
            return err
    current = sim.get_full_qpos()[:7]
    return float(np.max(np.abs(current - target_arm)))


# ═══════════════════════════════════════════════════════════════════
#  Shared data classes
# ═══════════════════════════════════════════════════════════════════


@dataclass
class IKStats:
    """Aggregate IK test statistics (shared by real & sim test scripts).

    Fields:
        ok:          number of successful IK solves
        total:       total number of IK targets attempted (optional)
        pos_errs_mm: list of position errors in mm (FK round-trip)
        rot_errs_deg: list of rotation errors in degrees
        max_dq_deg:  list of max joint displacement from seed (degrees, optional)
    """
    ok: int
    total: int = 0
    pos_errs_mm: list[float] = field(default_factory=list)
    rot_errs_deg: list[float] = field(default_factory=list)
    max_dq_deg: list[float] = field(default_factory=list)


def ik_stats_empty() -> IKStats:
    """Return a zeroed IKStats instance."""
    return IKStats(ok=0, total=0, pos_errs_mm=[], rot_errs_deg=[], max_dq_deg=[])
