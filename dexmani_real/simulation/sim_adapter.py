"""Simulation-to-RobotInterface adapter — wraps SAPIEN with hardware-compatible API.

Enables seamless switching between simulation and real hardware for controllers
and tests. No hardware dependencies — SAPIEN headless mode runs in CI
(no GPU/display required).
"""

from __future__ import annotations

__all__ = ["SimRobotConfig", "SimRobotInterface"]

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dexmani_real.robot._connection_state import ConnectionStateMixin
from dexmani_real.simulation.constructor import add_base_components, setup_scene
from dexmani_real.simulation.xarm7_xhand import XArm7XHand
from dexmani_real.utils.array_utils import nan_array


@dataclass
class SimRobotConfig:
    time_step: float = 1.0 / 240.0  # SAPIEN physics step (typically > control rate)
    headless: bool = True
    arm_home_qpos: np.ndarray = field(default_factory=lambda: np.deg2rad([-30.0, -1.9, 0.0, 13.5, -180.0, 74.7, 0.0]))
    # SAPIEN PD gains for the implicit joint-level position controller.
    # stiffness=1000, damping=100 → default SAPIEN values for bare xArm7
    # (no payload).  Increase damping if oscillation appears under load.
    arm_pd_gains: dict | None = field(
        default_factory=lambda: {
            "stiffness": 1000,
            "damping": 120,
            "force_limit": 200,
        }
    )
    hand_pd_gains: dict | None = field(
        default_factory=lambda: {
            "stiffness": 500,
            "damping": 100,
            "force_limit": 80,
        }
    )


class SimRobotInterface(ConnectionStateMixin):
    """SAPIEN simulation interface for independent testing.

    Provides get_state()/send_action()/reset() for simulation-internal
    validation without real hardware. Note: parameter and return types
    differ from RobotInterface — not a drop-in replacement.
    """

    def __init__(self, config: SimRobotConfig | None = None):
        super().__init__()
        self.config = config or SimRobotConfig()
        # Runtime type is sapien.Scene | None; annotated Any because scene is always
        # set/unset together with self.robot (connect/disconnect), but _step_physics
        # relies on the caller's robot-None guard, which mypy cannot see. The precise
        # Optional type would require type:ignore at every unguarded use site.
        self.scene: Any = None
        self.robot: XArm7XHand | None = None
        self.step_count = 0

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            self.scene = setup_scene(time_step=self.config.time_step)
            if not self.config.headless:
                add_base_components(self.scene)
            self.robot = XArm7XHand(
                self.scene,
                disable_self_collision=True,
                arm_home_qpos=self.config.arm_home_qpos.copy(),
                arm_pd_gains=self.config.arm_pd_gains,
                hand_pd_gains=self.config.hand_pd_gains,
            )
        except (OSError, ConnectionError, RuntimeError) as e:
            self.error_state = True
            self.last_error_message = f"sim setup failed: {e}"
            return False

        self.last_qpos_cmd = self.robot.get_qpos().copy()
        self.last_cmd_time = time.time()
        self.connected_flag = True
        self.error_state = False
        return True

    def disconnect(self) -> None:
        self.connected_flag = False
        self.scene = None
        self.robot = None

    def is_connected(self) -> bool:
        return self.robot is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.robot is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        return False

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_message = ""
        return self.is_connected()

    def stop(self) -> bool:
        """Soft-stop — send current position as target (hold in place)."""
        if self.robot is None:
            return False
        qpos = self.robot.get_qpos()
        self.robot.apply_action(qpos)
        return True

    # ------------------------------------------------------------------
    # State retrieval
    # ------------------------------------------------------------------

    def get_state(self, full: bool = False) -> dict[str, Any]:
        if self.robot is None:
            return {
                "qpos": nan_array(19),
                "qvel": nan_array(19),
                "timestamp": time.time(),
            }

        qpos = self.robot.get_qpos()  # (19,) [arm7 + hand12]
        eef_pose = self.robot.get_eef_pose()  # sapien.Pose

        state: dict[str, Any] = {
            "arm_qpos": qpos[:7].copy(),  # rad
            "hand_qpos": qpos[7:].copy(),  # rad
            "eef_pos": np.array(eef_pose.p, dtype=np.float64),  # m
            "eef_quat_wxyz": np.array(eef_pose.q, dtype=np.float64),  # w,x,y,z
            "qvel": np.zeros(19, dtype=np.float64),
            "timestamp": time.time(),
        }

        if full:
            state.update(
                {
                    "qpos_full": qpos.copy(),
                    "qlimits": self.robot.qlimits.copy() if self.robot.qlimits is not None else None,
                    "connected_flag": self.connected_flag,
                    "error_state": self.error_state,
                    "last_error_message": self.last_error_message,
                    "step_count": self.step_count,
                }
            )
        return state

    # ------------------------------------------------------------------
    # Action sending
    # ------------------------------------------------------------------

    def send_action(self, action: np.ndarray) -> bool:
        if self.robot is None:
            return False

        target_qpos = np.asarray(action, dtype=np.float64).reshape(19)

        # joint limit clip (using simulation URDF limits)
        if self.robot.qlimits is not None:
            qmin = self.robot.qlimits[:, 0]
            qmax = self.robot.qlimits[:, 1]
            clipped = np.clip(target_qpos, qmin, qmax)
            self.last_joint_limit_clipped = not np.allclose(target_qpos, clipped)
            target_qpos = clipped

        self.robot.apply_action(target_qpos)
        self._step_physics()

        self.last_qpos_cmd = target_qpos.copy()
        self.last_cmd_time = time.time()
        self.last_action_code = 0
        self.step_count += 1
        return True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, target: np.ndarray | None = None) -> bool:
        if self.robot is None:
            return False

        if target is not None:
            self.robot.set_qpos(target)
            self.robot.balance_passive_force()
            self.robot.apply_action(target)
        else:
            self.robot.reset(random_init=False)

        self._step_physics()
        self.last_qpos_cmd = self.robot.get_qpos().copy()
        self.last_cmd_time = time.time()
        return True

    def return_to_home(self) -> bool:
        return self.reset()

    def emergency_stop(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step_physics(self, n: int = 5):
        """Advance physics simulation. n=5: 240Hz → 48Hz effective control rate."""
        for _ in range(n):
            self.scene.step()

    def get_full_qpos(self) -> np.ndarray:
        """Return full 19-DOF qpos [arm7, hand12]."""
        if self.robot is None:
            return nan_array(19)
        return self.robot.get_qpos().copy()

    # ------------------------------------------------------------------
    # Simulation-specific validation
    # ------------------------------------------------------------------
