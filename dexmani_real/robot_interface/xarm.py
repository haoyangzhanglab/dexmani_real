import math
import time
import threading
import numpy as np
from typing import Callable, Sequence

from xarm.wrapper import XArmAPI
from dexmani_real.robot_interface.xarm_planner import XArm7Planner

JOINT_VEL_LIMITS_DEG: tuple[float, ...] = (45.0, 45.0, 45.0, 45.0, 60.0, 60.0, 90.0)
JOINT_ACC_LIMITS_DEG: tuple[float, ...] = tuple(v * 1.5 for v in JOINT_VEL_LIMITS_DEG)
JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-360.0, 360.0),
    (-117.0, 116.0),
    (-360.0, 360.0),
    (-6.0, 225.0),
    (-360.0, 360.0),
    (-97.0, 180.0),
    (-360.0, 360.0),
)
DEFAULT_SPEED_DEG = 30.0
DEFAULT_ACC_DEG = 67.5
DEFAULT_HOME_QPOS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def clip_angles(angles: Sequence[float], limits: tuple[tuple[float, float], ...] = JOINT_LIMITS) -> list[float]:
    return [max(lo, min(hi, float(angle))) for angle, (lo, hi) in zip(angles, limits)]


def clip_speed(speed: float) -> float:
    return max(1.0, min(float(speed), min(JOINT_VEL_LIMITS_DEG)))


def clip_acc(acc: float) -> float:
    return max(1.0, min(float(acc), min(JOINT_ACC_LIMITS_DEG)))


class XArm7Controller:
    MODE_POSITION = 0
    MODE_SERVO = 1
    MODE_TEACH = 2
    MODE_VELOCITY = 4
    MODE_TRAJECTORY = 6

    def __init__(self, ip: str, is_radian: bool = False):
        self.arm = XArmAPI(ip, is_radian=is_radian, do_not_open=True)
        self.is_connected = False
        self.mode = self.MODE_POSITION
        self.lock = threading.Lock()
        self.error_latched = False
        self.user_error_callback: Callable | None = None

    def error_stop_callback(self, data):
        err = 0
        warn = 0
        if isinstance(data, dict):
            err = int(data.get("error_code", 0) or 0)
            warn = int(data.get("warn_code", 0) or 0)
        elif isinstance(data, (list, tuple)):
            if len(data) > 0:
                err = int(data[0] or 0)
            if len(data) > 1:
                warn = int(data[1] or 0)
        if err != 0:
            self.error_latched = True
            self.emergency_stop()
        if self.user_error_callback is not None:
            self.user_error_callback(err, warn)

    def connect(self) -> bool:
        if self.arm.connect() != 0:
            return False
        self.is_connected = True
        self.error_latched = False
        if self.arm.motion_enable(enable=True) != 0:
            self.disconnect()
            return False
        if self.set_mode_raw(self.MODE_POSITION) != 0:
            self.disconnect()
            return False
        self.arm.register_error_warn_changed_callback(self.error_stop_callback)
        err, _ = self.get_err_warn()
        if err != 0:
            self.disconnect()
            return False
        return True

    def disconnect(self):
        with self.lock:
            if not self.is_connected:
                return
            self.arm.set_state(state=4)
            self.arm.disconnect()
            self.is_connected = False

    def set_mode_raw(self, mode: int) -> int:
        if self.error_latched:
            return -1
        with self.lock:
            code = self.arm.set_mode(mode)
            if code == 0:
                code = self.arm.set_state(state=0)
            if code == 0:
                self.mode = mode
            return code

    def set_mode(self, mode: int) -> int:
        return self.set_mode_raw(mode)

    def set_angles(
        self,
        angles: Sequence[float],
        speed: float = DEFAULT_SPEED_DEG,
        wait: bool = True,
        timeout: float | None = 10.0,
        acc: float = DEFAULT_ACC_DEG,
    ) -> int:
        if self.error_latched:
            return -1
        kwargs: dict[str, object] = {
            "angle": clip_angles(angles),
            "speed": clip_speed(speed),
            "mvacc": clip_acc(acc),
            "wait": wait,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout
        with self.lock:
            return self.arm.set_servo_angle(**kwargs)

    def set_servo(self, angles: Sequence[float]) -> int:
        if self.error_latched:
            return -1
        if self.mode != self.MODE_SERVO and self.set_mode(self.MODE_SERVO) != 0:
            return -1
        with self.lock:
            return self.arm.set_servo_angle_j(clip_angles(angles))

    def go_home(
        self,
        speed: float = DEFAULT_SPEED_DEG,
        wait: bool = True,
        timeout: float | None = 10.0,
        acc: float = DEFAULT_ACC_DEG,
    ) -> int:
        return self.set_angles(DEFAULT_HOME_QPOS, speed=speed, wait=wait, timeout=timeout, acc=acc)

    def get_angles(self) -> list[float] | None:
        code, angles = self.arm.get_servo_angle()
        return list(angles) if code == 0 else None

    def get_pose(self) -> list[float]:
        return list(self.arm.position)

    def get_torques(self) -> list[float]:
        return list(self.arm.joints_torque)

    def get_speeds(self) -> list[float]:
        return list(self.arm.realtime_joint_speeds)

    def get_temperatures(self) -> list[int]:
        return list(self.arm.temperatures)

    def get_state(self) -> dict:
        return {
            "mode": self.mode,
            "angles": self.get_angles(),
            "tcp_pose": self.get_pose(),
            "torques": self.get_torques(),
            "speeds": self.get_speeds(),
            "temperatures": self.get_temperatures(),
            "error_latched": self.error_latched,
        }

    def on_report_location(self, callback: Callable):
        return self.arm.register_report_location_callback(callback)

    def on_state_change(self, callback: Callable):
        return self.arm.register_state_changed_callback(callback)

    def on_error(self, callback: Callable):
        self.user_error_callback = callback

    def emergency_stop(self):
        with self.lock:
            self.arm.set_state(state=4)

    def recover(self):
        with self.lock:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
        self.error_latched = False
        self.set_mode_raw(self.mode)

    def get_err_warn(self) -> tuple[int, int]:
        code, err_warn = self.arm.get_err_warn_code()
        if code != 0:
            return code, 0
        return int(err_warn[0]), int(err_warn[1])

    def check_error(self) -> int:
        err, _ = self.get_err_warn()
        return err

    def check_warn(self) -> int:
        _, warn = self.get_err_warn()
        return warn


class XArm7(XArm7Controller):
    def __init__(
        self,
        ip: str,
        urdf_path: str,
        srdf_path: str | None = None,
        is_radian: bool = False,
        home_qpos: Sequence[float] | None = None,
        root_position: Sequence[float] | None = None,
        root_orientation_wxyz: Sequence[float] | None = None,
    ):
        super().__init__(ip, is_radian=is_radian)
        self.home_qpos = list(DEFAULT_HOME_QPOS if home_qpos is None else home_qpos)
        self.planner = XArm7Planner(
            urdf_path,
            srdf_path or urdf_path.replace('.urdf', '_mplib.srdf'),
            root_position=[0.0, 0.0, 0.0] if root_position is None else root_position,
            root_orientation_wxyz=[1.0, 0.0, 0.0, 0.0] if root_orientation_wxyz is None else root_orientation_wxyz,
        )

    def set_home_qpos(self, home_qpos: Sequence[float]):
        self.home_qpos = clip_angles(home_qpos)

    def set_root_pose(self, position: Sequence[float], orientation_wxyz: Sequence[float]):
        self.planner.set_root_pose(position, orientation_wxyz)

    def current_qpos_rad(self):
        angles = self.get_angles()
        return None if angles is None else np.deg2rad(angles)

    def go_home(
        self,
        speed: float = DEFAULT_SPEED_DEG,
        wait: bool = True,
        timeout: float | None = 10.0,
        acc: float = DEFAULT_ACC_DEG,
    ) -> int:
        return self.set_angles(self.home_qpos, speed=speed, wait=wait, timeout=timeout, acc=acc)

    def stream_path_rad(self, path) -> int:
        if self.mode != self.MODE_SERVO and self.set_mode(self.MODE_SERVO) != 0:
            return -1
        dt = max(float(self.planner.mp_dt), 1e-3)
        for qpos in np.asarray(path, dtype=float):
            if self.error_latched:
                return -1
            if self.set_servo(np.rad2deg(qpos).tolist()) != 0:
                return -1
            time.sleep(dt)
        return 0

    def forward_kinematics(self, qpos: Sequence[float] | None = None) -> tuple[list[float], list[float]] | None:
        qpos_deg = self.get_angles() if qpos is None else list(qpos)
        if qpos_deg is None:
            return None
        position, quat_wxyz = self.planner.forward_kinematics(np.deg2rad(qpos_deg))
        return position.tolist(), quat_wxyz.tolist()

    def inverse_kinematics(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        qpos: Sequence[float] | None = None,
    ) -> list[float] | None:
        ref_deg = self.get_angles() if qpos is None else list(qpos)
        if ref_deg is None:
            return None
        target = self.planner.inverse_kinematics(position, orientation_wxyz, np.deg2rad(ref_deg))
        return None if target is None else np.rad2deg(target).tolist()

    def inverse_kinematics_multi(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        qpos: Sequence[float] | None = None,
        n_perturb: int = 15,
        verbose: bool = False,
    ) -> list[list[float]]:
        ref_deg = self.get_angles() if qpos is None else list(qpos)
        if ref_deg is None:
            return []
        candidates = self.planner.inverse_kinematics_multi(
            np.asarray(position, dtype=float),
            np.asarray(orientation_wxyz, dtype=float),
            np.deg2rad(ref_deg),
            n_perturb=n_perturb,
            verbose=verbose,
        )
        return [np.rad2deg(q).tolist() for q in candidates]

    def set_planner_vel_limits(self, limits_deg: Sequence[float]):
        self.planner.set_vel_limits(np.deg2rad(np.asarray(limits_deg, dtype=float)))

    def set_planner_acc_limits(self, limits_deg: Sequence[float]):
        self.planner.set_acc_limits(np.deg2rad(np.asarray(limits_deg, dtype=float)))

    def set_pose(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        speed: float = DEFAULT_SPEED_DEG,
        wait: bool = True,
    ) -> int:
        current = self.current_qpos_rad()
        if current is None:
            return -1
        target = self.planner.inverse_kinematics(
            np.asarray(position, dtype=float),
            np.asarray(orientation_wxyz, dtype=float),
            current,
        )
        if target is None:
            return -1
        return self.set_angles(np.rad2deg(target).tolist(), speed=speed, wait=wait)

    def move_to_pose(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        plan_mode: str = 'joint_first',
        verbose: bool = False,
    ) -> int:
        current = self.current_qpos_rad()
        if current is None:
            return -1
        path = self.planner.plan_path(
            np.asarray(position, dtype=float),
            np.asarray(orientation_wxyz, dtype=float),
            current,
            plan_mode=plan_mode,
            verbose=verbose,
        )
        if path is None:
            return -1
        return self.stream_path_rad(path)

    def plan_path(
        self,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        current_qpos: Sequence[float] | None = None,
        plan_mode: str = 'joint_first',
        verbose: bool = False,
    ) -> list[list[float]] | None:
        qpos_deg = self.get_angles() if current_qpos is None else list(current_qpos)
        if qpos_deg is None:
            return None
        path = self.planner.plan_path(
            np.asarray(position, dtype=float),
            np.asarray(orientation_wxyz, dtype=float),
            np.deg2rad(qpos_deg),
            plan_mode=plan_mode,
            verbose=verbose,
        )
        return None if path is None else np.rad2deg(path).tolist()

    def check_collision(self, qpos: Sequence[float] | None = None) -> bool:
        qpos_deg = self.get_angles() if qpos is None else list(qpos)
        if qpos_deg is None:
            return False
        return self.planner.check_collision(np.deg2rad(qpos_deg))


XArm7MotionController = XArm7


def example():
    robot_ip = '192.168.1.xxx'
    ctrl = XArm7Controller(robot_ip)
    if not ctrl.connect():
        print('连接失败')
        return

    stopped = threading.Event()

    def on_error(err, warn):
        print(f'error={err}, warn={warn}')
        stopped.set()

    ctrl.on_error(on_error)

    try:
        print('回零位...')
        ctrl.go_home()
        print('移动到工作姿态...')
        ctrl.set_angles([0, -30, 0, 60, 0, 30, 0], speed=30)
        print('切换到伺服模式...')
        ctrl.set_mode(ctrl.MODE_SERVO)
        print('伺服模式运行 3 秒...')
        base = [0, -30, 0, 60, 0, 30, 0]
        t0 = time.perf_counter()
        while not stopped.is_set() and time.perf_counter() - t0 < 3.0:
            delta = math.sin((time.perf_counter() - t0) * 2.0) * 10.0
            ctrl.set_servo([base[0] + delta, *base[1:]])
            time.sleep(0.01)
        if not stopped.is_set():
            print('回到零位...')
            ctrl.set_mode(ctrl.MODE_POSITION)
            ctrl.go_home()
    finally:
        ctrl.disconnect()
        print('已断开连接')


if __name__ == '__main__':
    example()