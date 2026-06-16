"""PolicyRunner — execute a trained policy on the real robot.

LeFranX-style action chunk execution with EMA smoothing and safety monitoring.
"""

from __future__ import annotations

import time
import traceback
import warnings
from typing import Any

import numpy as np

from dexmani_real.deploy.action_parser import ActionParser
from dexmani_real.deploy.observation_builder import ObservationBuilder
from dexmani_real.deploy.safety_monitor import SafetyMonitor
from dexmani_real.robot.robot_interface import RobotAction, RobotInterface, RobotState
from dexmani_real.utils.rate_limiter import RateLimiter


class PolicyRunner:
    """Main policy deployment loop.

    Executes a trained policy model on the real robot with:
    - Action chunk temporal execution
    - EMA smoothing (arm + hand, alpha=0.5 for stronger smoothing than teleop)
    - Safety monitoring every step
    """

    def __init__(
        self,
        robot: RobotInterface,
        model: Any,                    # must implement predict(obs: dict) -> np.ndarray
        norm_stats: dict,
        *,
        chunk_size: int = 1,           # length of action chunk model outputs
        n_action_steps: int = 1,       # execute this many steps per inference
        query_freq: int = 1,           # re-infer every N steps
        action_mode: str = "full",     # "full" | "arm_only" | "hand_only"
        hand_smooth_alpha: float = 0.5,
        arm_smooth_alpha: float = 0.5,
        safety_monitor: SafetyMonitor | None = None,
        max_steps: int = 1000,
        target_hz: float = 50.0,
        camera_recorder: Any | None = None,
    ) -> None:
        self.robot = robot
        self.model = model
        self.max_steps = max_steps

        self.obs_builder = ObservationBuilder(norm_stats)
        self.action_parser = ActionParser(action_mode)
        self.safety_monitor = safety_monitor
        self.camera_recorder = camera_recorder

        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.query_freq = query_freq
        self.hand_smooth_alpha = hand_smooth_alpha
        self.arm_smooth_alpha = arm_smooth_alpha

        self.limiter = RateLimiter(target_hz)
        self._ema_arm: np.ndarray | None = None
        self._ema_hand: np.ndarray | None = None
        self._action_buffer: np.ndarray | None = None

    def run(self) -> None:
        print(f"[PolicyRunner] Starting deployment loop, max_steps={self.max_steps}")
        print(f"  chunk_size={self.chunk_size} n_action_steps={self.n_action_steps}")
        print(f"  action_mode={self.action_parser.action_mode}")
        print(f"  EMA: arm_alpha={self.arm_smooth_alpha} hand_alpha={self.hand_smooth_alpha}")

        try:
            for step in range(self.max_steps):
                # 1. Read state
                state = self.robot.get_state()

                # 2. Camera frame
                camera_frame = None
                if self.camera_recorder is not None:
                    try:
                        camera_frame = self.camera_recorder.read_frame()
                    except Exception:
                        pass

                # 3. Inference (every query_freq steps, or first step / buffer empty)
                if step % self.query_freq == 0 or self._action_buffer is None:
                    obs = self.obs_builder.build(state, camera_frame)
                    try:
                        model_output = self.model.predict(obs)
                        model_output = np.asarray(model_output, dtype=np.float64)
                        self._action_buffer = model_output.reshape(-1, 19)
                    except Exception:
                        traceback.print_exc()
                        break

                # 4. Extract action from chunk buffer
                chunk_idx = (step // self.n_action_steps) % max(
                    self._action_buffer.shape[0] // self.n_action_steps, 1
                )
                raw_action = self._action_buffer[chunk_idx]

                action = self.action_parser.parse(raw_action, state)

                # 5. EMA smoothing
                if self._ema_arm is not None:
                    action.arm_qpos_cmd = (
                        self.arm_smooth_alpha * action.arm_qpos_cmd
                        + (1.0 - self.arm_smooth_alpha) * self._ema_arm
                    )
                if self._ema_hand is not None:
                    action.hand_qpos_cmd = (
                        self.hand_smooth_alpha * action.hand_qpos_cmd
                        + (1.0 - self.hand_smooth_alpha) * self._ema_hand
                    )
                self._ema_arm = action.arm_qpos_cmd.copy()
                self._ema_hand = action.hand_qpos_cmd.copy()

                # 6. Safety check
                if self.safety_monitor is not None:
                    status = self.safety_monitor.check(state, action)
                    if not status.ok:
                        print(f"[PolicyRunner] SAFETY STOP: {status.message}")
                        self.robot.emergency_stop()
                        break

                # 7. Execute
                if self.robot.is_error():
                    print("[PolicyRunner] Robot error state detected, stopping.")
                    break

                result = self.robot.send_action(action)
                if not result.get("arm_ok", False) or not result.get("hand_ok", False):
                    warnings.warn(
                        f"send_action: arm_ok={result.get('arm_ok')} "
                        f"hand_ok={result.get('hand_ok')}"
                    )

                self.limiter.wait()

        except KeyboardInterrupt:
            print("\n[PolicyRunner] KeyboardInterrupt — stopping.")
        except Exception:
            traceback.print_exc()
        finally:
            print(f"[PolicyRunner] Finished after {step + 1} steps.")
