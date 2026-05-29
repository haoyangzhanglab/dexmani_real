import time
import threading
import numpy as np
from typing import List
from threading import Lock
from pynput import keyboard


class GlobalKeyState:

    def __init__(self):
        self._pressed = set()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _on_press(self, key):
        ch = getattr(key, "char", None)
        if not ch:
            return
        with self._lock:
            self._pressed.add(ch)

    def _on_release(self, key):
        ch = getattr(key, "char", None)
        if not ch:
            return
        with self._lock:
            self._pressed.discard(ch)

    def is_pressed(self, key_char: str) -> bool:
        if not isinstance(key_char, str) or len(key_char) != 1:
            raise ValueError("key_char must be a single character, e.g. 'r'")
        with self._lock:
            return key_char in self._pressed

    def stop(self):
        try:
            self._listener.stop()
        except Exception:
            pass



class KeyBoardListener:

    def __init__(
            self,
            offset_pos_m: List[float] = [0.0, 0.0, 0.0],
            delta_pos: float = 0.01,
            delta_rpy: float = 0.01,
            delta_width: float = 0.1,
            max_mode_num: int = 8,
            trigger_cooldown: float = 1.0,
    ):
        self.init_state = np.zeros(7)
        self.buffer = np.zeros(7)  # delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, delta_width
        self.qmin = np.array([-0.04, -0.35, -0.3, -np.pi, -np.pi, -np.pi, 0.0])
        self.qmax = np.array([0.36, 0.35, 0.3, np.pi, np.pi, np.pi, 1.0])

        self.init_state[:3] = np.array(offset_pos_m)  # EEF初始坐标原点

        self.mode_buffer = 0  # 灵巧手模式（当前所在段）
        self.max_mode_num = max_mode_num
        self.last_mode_change_time = 0.0

        self.delta_pos = delta_pos
        self.delta_rpy = delta_rpy
        self.delta_width = delta_width

        self.record_flag = False
        self.last_record_time = 0.0
        self.trigger_cooldown = trigger_cooldown

        self.exit_flag = False

        self._lock = Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()


    def _move_hand_path(self, delta_alpha: float):
        """
        沿着整条路径连续移动：
        home -> mode[0] -> mode[1] -> ... -> mode[n]
        """
        alpha = self.buffer[6] + delta_alpha
        mode = self.mode_buffer

        while alpha > 1.0 and mode < self.max_mode_num - 1:
            alpha -= 1.0
            mode += 1

        while alpha < 0.0 and mode > 0:
            alpha += 1.0
            mode -= 1

        self.mode_buffer = mode
        self.buffer[6] = float(np.clip(alpha, 0.0, 1.0))


    def _on_press(self, key):
        with self._lock:
            if key == keyboard.Key.esc:
                self.exit_flag = True
                self._listener.stop()
                return
            try:
                k = key.char.lower()
            except AttributeError:
                k = key

            # 平移
            if k == 'w':
                self.buffer[0] += self.delta_pos
            elif k == 's':
                self.buffer[0] -= self.delta_pos
            elif k == 'a':
                self.buffer[1] -= self.delta_pos
            elif k == 'd':
                self.buffer[1] += self.delta_pos
            elif k == keyboard.Key.up:
                self.buffer[2] += self.delta_pos
            elif k == keyboard.Key.down:
                self.buffer[2] -= self.delta_pos

            # 旋转
            elif k == keyboard.Key.left:
                self.buffer[3] -= self.delta_rpy
            elif k == keyboard.Key.right:
                self.buffer[3] += self.delta_rpy
            elif k == 'i':
                self.buffer[4] -= self.delta_rpy
            elif k == 'k':
                self.buffer[4] += self.delta_rpy
            elif k == 'j':
                self.buffer[5] += self.delta_rpy
            elif k == 'l':
                self.buffer[5] -= self.delta_rpy

            # 手指沿整条路径连续前后移动
            elif k == 'o':
                self._move_hand_path(+self.delta_width)
            elif k == 'u':
                self._move_hand_path(-self.delta_width)

            # 记录
            elif k == 'r':
                now = time.monotonic()
                if now - self.last_record_time > self.trigger_cooldown:
                    self.record_flag = True
                    self.last_record_time = now
                    print("Record triggered")

            # 限幅，更新状态
            self.buffer = np.clip(self.buffer, self.qmin, self.qmax)

    def get_control(self):
        with self._lock:
            rec = self.record_flag
            self.record_flag = False
            
            return {
                "state": self.init_state + self.buffer,
                "record": rec,
                "mode": self.mode_buffer,
                "exit": self.exit_flag,
            }



def teleop_example():
    import time
    import numpy as np
    from transforms3d import euler
    from dexmani_real import ASSET_DIR
    from dexmani_real.robot.xarm7 import XArm7, XArm7Config
    from dexmani_real.robot.planner import (
        XArm7MotionPlanner,
        XArm7PlannerConfig,
        Pose,
        PlanningProfile,
        TeleopProfile,
    )

    robot = XArm7(
        config=XArm7Config(
            ip="192.168.1.111",
            use_delta_limit=False
        )
    ).connect()
    robot.reset()


    planner_config = XArm7PlannerConfig(
        urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
        srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf"),
        base_pose_world=Pose(p=[0.0, 0.0, 0.0], q=euler.euler2quat(0, 0, np.pi/6)),
    )
    arm_planner = XArm7MotionPlanner(
        config=planner_config,
        planning_profile=PlanningProfile(path_dt=0.05),
        teleop_profile=TeleopProfile(
            teleop_dt=0.04,
            max_qpos_cmd_speed_deg=(90, 90, 90, 90, 120, 120, 150),
            max_ik_jump_deg=(90, 90, 90, 90, 120, 120, 180),
        ),
    )

    init_qpos = robot.get_obs()["qpos"]
    previous_qpos_cmd = robot.last_qpos_cmd.copy()
    init_eef_pose = arm_planner.compute_eef_pose_world(qpos=init_qpos[:7])  # 初始化内部状态
    print(f"Initial EEF pose: {init_eef_pose}")

    eef_teleop = KeyBoardListener(
        offset_pos_m=[0.3181, 0.0, 0.3608],
        delta_pos = 0.006,
        delta_rpy = 0.0314,
    )

    while True:
        t1 = time.monotonic()
        msg = eef_teleop.get_control()

        if msg["exit"]:
            print("Teleop ended, stopping teleoperation.")
            break

        current_arm_qpos = robot.get_obs()["qpos"]

        target_eef_pos = np.asarray(msg["state"][:3], dtype=np.float64)
        target_eef_quat = np.asarray(
            euler.euler2quat(*msg["state"][3:6], axes="sxyz"),
            dtype=np.float64,
        )
        target_eef_pose = Pose(p=target_eef_pos, q=target_eef_quat)

        ik_result = arm_planner.solve_teleop_ik(
            target_eef_pose_world=target_eef_pose,
            current_qpos=current_arm_qpos,
            previous_qpos_cmd=previous_qpos_cmd,
        )

        if not ik_result.success:
            print(ik_result.brief())

        if ik_result.qpos is None:
            target_arm_qpos = previous_qpos_cmd.copy()
        else:
            target_arm_qpos = ik_result.qpos.copy()

        
        robot.send_action(target_arm_qpos)
        previous_qpos_cmd = target_arm_qpos.copy()

        time.sleep(0.012)  # 模拟控制周期

        t2 = time.monotonic()
        print(f"Loop time: {(t2 - t1) * 1000:.1f} ms")
    
    robot.reset()


if __name__ == "__main__":
    teleop_example()