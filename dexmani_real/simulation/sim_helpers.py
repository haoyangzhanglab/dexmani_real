"""Simulation execution helpers — dense path execution, PD convergence.

Functions in this module are SAPIEN-specific and only usable when
a simulation environment is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import sapien.core as sapien

    from dexmani_real.simulation import SimRobotInterface


def execute_dense_path(
    sim: SimRobotInterface,
    dense: np.ndarray,
    viewer: "sapien.Viewer | None" = None,
    physics_steps_per_wp: int = 20,
) -> bool:
    """Execute a pre-densified arm joint path (N, 7) in simulation.

    Hand joints are held constant throughout.

    Args:
        sim: simulation interface handle.
        dense: (N, 7) arm joint waypoints.
        viewer: optional SAPIEN viewer for rendering.
        physics_steps_per_wp: physics steps per waypoint.

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
    ``converge_threshold_rad``.

    Args:
        sim: simulation interface handle.
        target_arm: (7,) target arm joint angles.
        hand_qpos: (12,) hand joint angles (held constant).
        max_iter: max PD iterations before giving up.
        converge_threshold_rad: stop when max |joint_error| < this.
        physics_steps_per_wp: physics steps per PD iteration.

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
