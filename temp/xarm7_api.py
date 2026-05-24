import time
from typing import Optional

import numpy as np
from xarm.wrapper import XArmAPI


XARM7_Q_MIN = np.deg2rad(
    np.array([-360.0, -117.0, -360.0, -6.0, -360.0, -97.0, -360.0], dtype=np.float64)
)
XARM7_Q_MAX = np.deg2rad(
    np.array([360.0, 116.0, 360.0, 225.0, 360.0, 180.0, 360.0], dtype=np.float64)
)

XARM7_DEFAULT_MAX_JOINT_DELTA = np.deg2rad(
    np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5], dtype=np.float64)
)
XARM7_DEFAULT_MAX_SEGMENT_DELTA = np.deg2rad(
    np.array([3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 8.0], dtype=np.float64)
)

DEFAULT_JOINT_SPEED = np.deg2rad(20.0)
DEFAULT_JOINT_MVACC = np.deg2rad(100.0)

MODE_POSITION = 0
MODE_SERVO = 1
SET_STATE_READY = 0
STATE_STOP = 4
STATE_DECELERATION_STOP = 6
MOTION_OK_STATES = (1, 2)
STATE_SETTLE_TIME = 0.1


def first_bad_code(*codes: Optional[int]) -> int:
    for code in codes:
        if code is not None and code != 0:
            return int(code)
    return 0


def to_xarm7_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] != 7:
        raise ValueError(f"q shape must be (7,), got {tuple(q.shape)}")
    if not np.isfinite(q).all():
        raise ValueError("q contains nan or inf")
    return q


def to_xarm7_delta(delta, name: str = "delta") -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.ndim == 0:
        delta = np.full(7, float(delta), dtype=np.float64)
    else:
        delta = delta.reshape(-1)

    if delta.shape[0] != 7:
        raise ValueError(f"{name} shape must be scalar or (7,), got {tuple(delta.shape)}")
    if not np.isfinite(delta).all():
        raise ValueError(f"{name} contains nan or inf")
    if np.any(delta <= 0):
        raise ValueError(f"{name} must be positive")
    return delta


def to_xarm7_vec(x, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] < 7:
        raise RuntimeError(f"{name} length must be >= 7, got {x.shape[0]}")
    return x[:7]


def to_xarm7_path(path) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 7:
        raise ValueError(f"path shape must be (T, 7), got {tuple(path.shape)}")
    if path.shape[0] == 0:
        raise ValueError("path must contain at least one waypoint")
    if not np.isfinite(path).all():
        raise ValueError("path contains nan or inf")
    return path


def check_joint_limits(q_or_path) -> None:
    x = np.asarray(q_or_path, dtype=np.float64)
    if np.any((x < XARM7_Q_MIN) | (x > XARM7_Q_MAX)):
        raise ValueError("joint target is outside xArm7 joint limits")


def densify_joint_path(path, max_segment_delta=XARM7_DEFAULT_MAX_SEGMENT_DELTA) -> np.ndarray:
    path = to_xarm7_path(path)
    check_joint_limits(path)
    max_segment_delta = to_xarm7_delta(max_segment_delta, "max_segment_delta")

    dense_path = [path[0].copy()]
    for q0, q1 in zip(path[:-1], path[1:]):
        delta = q1 - q0
        if np.allclose(delta, 0.0):
            continue

        num_steps = int(np.ceil(np.max(np.abs(delta) / max_segment_delta)))
        num_steps = max(num_steps, 1)
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            dense_path.append(q0 + alpha * delta)

    return np.asarray(dense_path, dtype=np.float64)


class Xarm7Controller:
    def __init__(
        self,
        ip: str,
        use_self_collision_detection: bool = True,
        max_joint_delta=XARM7_DEFAULT_MAX_JOINT_DELTA,
    ):
        self.ip = ip
        self.use_self_collision_detection = bool(use_self_collision_detection)
        self.max_joint_delta = to_xarm7_delta(max_joint_delta, "max_joint_delta")

        self.arm = None
        self.last_api_code = None
        self.last_q_cmd = None
        self.last_action_limited = False

    def is_connected(self) -> bool:
        return self.arm is not None

    def connect(self) -> bool:
        if self.is_connected():
            return True

        arm = XArmAPI(self.ip, do_not_open=True)
        code = arm.connect()
        self.last_api_code = code
        if code != 0:
            return False

        self.arm = arm
        if not self.clear_error():
            self.disconnect()
            return False

        if not self.apply_safety_settings():
            self.disconnect()
            return False

        return self.get_error_status()["servo_ready"]

    def disconnect(self) -> bool:
        if self.arm is None:
            return True

        try:
            self.arm.disconnect()
        finally:
            self.arm = None
            self.last_q_cmd = None
            self.last_action_limited = False

        return True

    def set_mode(self, mode: int) -> bool:
        if not self.is_connected():
            return False

        code = first_bad_code(
            self.arm.set_mode(int(mode)),
            self.arm.set_state(SET_STATE_READY),
        )
        self.last_api_code = code
        if code == 0:
            time.sleep(STATE_SETTLE_TIME)
        return code == 0

    def apply_safety_settings(self) -> bool:
        if not self.is_connected():
            return False

        code = first_bad_code(
            self.arm.set_self_collision_detection(
                1 if self.use_self_collision_detection else 0
            ),
            self.arm.set_state(SET_STATE_READY),
        )
        self.last_api_code = code
        if code == 0:
            time.sleep(STATE_SETTLE_TIME)
        return code == 0

    def clear_error(self) -> bool:
        if not self.is_connected():
            return False

        code = first_bad_code(
            self.arm.clean_error(),
            self.arm.clean_warn(),
            self.arm.motion_enable(True),
            self.arm.set_mode(MODE_SERVO),
            self.arm.set_state(SET_STATE_READY),
        )
        self.last_api_code = code
        if code != 0:
            return False

        time.sleep(STATE_SETTLE_TIME)
        return self.get_error_status()["servo_ready"]

    def reset(self) -> bool:
        return self.clear_error()

    def return_home(self, *args, **kwargs):
        raise NotImplementedError("return_home() is not implemented yet")

    def stop(self, decelerate: bool = False) -> bool:
        if not self.is_connected():
            return False

        mode = int(getattr(self.arm, "mode", -1))
        state = STATE_DECELERATION_STOP if decelerate and mode == MODE_POSITION else STATE_STOP
        code = self.arm.set_state(state)
        self.last_api_code = code
        return code == 0

    def get_error_status(self) -> dict:
        if not self.is_connected():
            return {
                "mode": -1,
                "state": -1,
                "err": -1,
                "warn": -1,
                "connected": False,
                "ready": False,
                "servo_ready": False,
                "position_ready": False,
                "api_code": self.last_api_code,
            }

        code_state, state = self.arm.get_state()
        code_err, err_warn = self.arm.get_err_warn_code()
        api_code = first_bad_code(code_state, code_err)
        self.last_api_code = api_code

        state = int(state) if code_state == 0 else -1
        err = int(err_warn[0]) if code_err == 0 else -1
        warn = int(err_warn[1]) if code_err == 0 else -1
        mode = int(getattr(self.arm, "mode", -1))

        ready = api_code == 0 and state in MOTION_OK_STATES and err == 0 and warn == 0
        servo_ready = ready and mode == MODE_SERVO
        position_ready = ready and mode == MODE_POSITION

        return {
            "mode": mode,
            "state": state,
            "err": err,
            "warn": warn,
            "connected": api_code != -1,
            "ready": ready,
            "servo_ready": servo_ready,
            "position_ready": position_ready,
            "api_code": api_code,
        }

    def ensure_mode_ready(self, mode: int) -> dict:
        if not self.is_connected():
            raise RuntimeError("xarm is not connected")

        status = self.get_error_status()
        if status["api_code"] != 0 or status["err"] != 0 or status["warn"] != 0:
            raise RuntimeError(f"xarm has api/error/warn status: {status}")

        if status["ready"] and status["mode"] == int(mode):
            return status

        if not self.set_mode(int(mode)):
            raise RuntimeError(f"failed to enter mode {mode}")

        status = self.get_error_status()
        if not status["ready"] or status["mode"] != int(mode):
            raise RuntimeError(f"xarm failed to enter mode-ready state: {status}")
        return status

    def ensure_servo_ready(self) -> dict:
        return self.ensure_mode_ready(MODE_SERVO)

    def ensure_position_ready(self) -> dict:
        return self.ensure_mode_ready(MODE_POSITION)

    def read_joint_state(self):
        if not self.is_connected():
            raise RuntimeError("xarm is not connected")

        try:
            code, data = self.arm.get_joint_states(is_radian=True, num=3)
        except TypeError:
            code, data = self.arm.get_joint_states(is_radian=True)

        if code == 0 and len(data) >= 3:
            self.last_api_code = 0
            q = to_xarm7_vec(data[0], "q")
            dq = to_xarm7_vec(data[1], "dq")
            effort = to_xarm7_vec(data[2], "effort")
            return q, dq, effort, True, True

        code_q, q = self.arm.get_servo_angle(is_radian=True)
        self.last_api_code = first_bad_code(code, code_q)
        if code_q != 0:
            raise RuntimeError("failed to read joint positions")

        q = to_xarm7_vec(q, "q")
        dq = np.zeros(7, dtype=np.float64)
        effort = None
        return q, dq, effort, False, False

    def get_observation(self) -> dict:
        q, dq, effort, has_joint_velocity, has_effort = self.read_joint_state()
        return {
            "q": q,
            "dq": dq,
            "effort": effort,
            "has_joint_velocity": has_joint_velocity,
            "has_effort": has_effort,
            "timestamp": time.time(),
        }

    def get_observation_full(self) -> dict:
        obs = self.get_observation()
        obs.update(self.get_error_status())
        return obs

    def reset_action_limiter(self, q=None) -> None:
        self.last_q_cmd = None if q is None else to_xarm7_q(q).copy()
        self.last_action_limited = False

    def limit_joint_step(self, q: np.ndarray) -> np.ndarray:
        q = to_xarm7_q(q)
        if self.last_q_cmd is None:
            self.last_q_cmd = q.copy()
            self.last_action_limited = False
            return q

        q_limited = np.clip(
            q,
            self.last_q_cmd - self.max_joint_delta,
            self.last_q_cmd + self.max_joint_delta,
        )
        self.last_action_limited = not np.allclose(q_limited, q)
        self.last_q_cmd = q_limited.copy()
        return q_limited

    def send_action(self, q, strict_limit: bool = False) -> np.ndarray:
        self.ensure_servo_ready()

        q = to_xarm7_q(q)
        outside_limit = np.any((q < XARM7_Q_MIN) | (q > XARM7_Q_MAX))
        if strict_limit and outside_limit:
            raise ValueError("q is outside xArm7 joint limits")

        q_sent = np.clip(q, XARM7_Q_MIN, XARM7_Q_MAX)
        if self.last_q_cmd is None:
            q_now, _, _, _, _ = self.read_joint_state()
            self.reset_action_limiter(q_now)
        q_sent = self.limit_joint_step(q_sent)

        code = self.arm.set_servo_angle_j(
            angles=q_sent.tolist(),
            is_radian=True,
        )
        self.last_api_code = code
        if code != 0:
            raise RuntimeError(f"set_servo_angle_j failed with api code {code}")

        return q_sent

    def move_joint(
        self,
        q,
        speed: float = DEFAULT_JOINT_SPEED,
        mvacc: float = DEFAULT_JOINT_MVACC,
        wait: bool = True,
        radius: float = -1.0,
    ) -> np.ndarray:
        q = to_xarm7_q(q)
        check_joint_limits(q)
        self.ensure_position_ready()

        code = self.arm.set_servo_angle(
            angle=q.tolist(),
            speed=float(speed),
            mvacc=float(mvacc),
            is_radian=True,
            wait=bool(wait),
            radius=float(radius),
        )
        self.last_api_code = code
        if code != 0:
            raise RuntimeError(f"set_servo_angle failed with api code {code}")

        self.reset_action_limiter(q)
        return q

    def execute_joint_path(
        self,
        path,
        speed: float = DEFAULT_JOINT_SPEED,
        mvacc: float = DEFAULT_JOINT_MVACC,
        wait_each: bool = True,
        radius: float = -1.0,
        max_segment_delta=XARM7_DEFAULT_MAX_SEGMENT_DELTA,
        densify: bool = True,
        stop_on_error: bool = True,
    ) -> dict:
        input_path = to_xarm7_path(path)
        check_joint_limits(input_path)
        path_to_execute = (
            densify_joint_path(input_path, max_segment_delta)
            if densify
            else input_path.copy()
        )
        check_joint_limits(path_to_execute)
        self.ensure_position_ready()

        last_index = -1
        last_code = 0
        for i, q in enumerate(path_to_execute):
            code = self.arm.set_servo_angle(
                angle=q.tolist(),
                speed=float(speed),
                mvacc=float(mvacc),
                is_radian=True,
                wait=bool(wait_each),
                radius=float(radius),
            )
            self.last_api_code = code
            last_code = code
            last_index = i

            status = self.get_error_status()
            if code != 0 or status["api_code"] != 0 or status["err"] != 0 or status["warn"] != 0:
                if stop_on_error:
                    self.stop()
                break

        if last_index >= 0:
            self.reset_action_limiter(path_to_execute[last_index])

        final_status = self.get_error_status()
        ok = (
            last_code == 0
            and last_index == len(path_to_execute) - 1
            and final_status["api_code"] == 0
            and final_status["err"] == 0
            and final_status["warn"] == 0
        )
        return {
            "ok": ok,
            "num_input_points": int(len(input_path)),
            "num_dense_points": int(len(path_to_execute)),
            "num_executed_points": int(last_index + 1),
            "last_index": int(last_index),
            "last_code": int(last_code),
            "status": final_status,
        }


def example():
    """Minimal smoke test.

    This example connects to the robot, reads one observation, sends the
    current joint position back once, then stops and disconnects. It is a
    wrapper smoke test, not a motion demo.
    """
    ip = "192.168.1.xxx"
    ctrl = Xarm7Controller(ip=ip, use_self_collision_detection=True)

    if not ctrl.connect():
        print("connect failed")
        print(ctrl.get_error_status())
        return

    try:
        status = ctrl.get_error_status()
        print("status:", status)

        obs = ctrl.get_observation()
        print("q:", obs["q"])
        print("dq:", obs["dq"])
        print("has_joint_velocity:", obs["has_joint_velocity"])
        print("has_effort:", obs["has_effort"])

        q_sent = ctrl.send_action(obs["q"], strict_limit=True)
        print("q_sent:", q_sent)

    finally:
        ctrl.stop()
        ctrl.disconnect()
        print("disconnected")


if __name__ == "__main__":
    example()
