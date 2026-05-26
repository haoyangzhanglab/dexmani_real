from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from xhand_controller import xhand_control as xh


JOINT_NAMES = [
    "thumb_abduction",
    "thumb_joint1",
    "thumb_joint2",
    "index_abduction",
    "index_joint1",
    "index_joint2",
    "middle_joint1",
    "middle_joint2",
    "ring_joint1",
    "ring_joint2",
    "little_joint1",
    "little_joint2",
]

SENSOR_IDS = [0x11, 0x12, 0x13, 0x14, 0x15]
SENSOR_NAMES = ["thumb", "index", "middle", "ring", "little"]


@dataclass
class XHandConfig:
    comm_type: str = "RS485"
    device_name: str | None = None
    baudrate: int = 3_000_000
    device_id: int = 0

    dt: float = 1.0 / 83.0
    num_joints: int = 12
    force_update_state: bool = False

    home_qpos: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float64))
    qpos_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -1.57, -1.05, 0.0,
                -0.087, 0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
            ],
            dtype=np.float64,
        )
    )
    qpos_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                1.57, 1.57, 1.57,
                0.297, 1.92, 1.92,
                1.92, 1.92,
                1.92, 1.92,
                1.92, 1.92,
            ],
            dtype=np.float64,
        )
    )
    max_qvel: np.ndarray = field(default_factory=lambda: np.deg2rad(np.ones(12) * 180.0))

    kp: int = 100
    ki: int = 0
    kd: int = 1
    tor_max: int = 100
    mode: int = 3

    use_delta_limit: bool = True
    clip_joint_limit: bool = True

    tactile_scale: float = 0.1


class XHand:
    def __init__(self, config: XHandConfig):
        self.config = config
        self.control = None
        self.device_name: str | None = None
        self.hand_command = None

        self.connected_flag = False
        self.error_state = False

        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_action_code: int | None = None
        self.last_error_code: int | None = None
        self.last_error_message = ""
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

        self.last_commboard_err = np.zeros(12, dtype=np.int32)
        self.last_jointboard_err = np.zeros(12, dtype=np.int32)
        self.last_tipboard_err = np.zeros(12, dtype=np.int32)
        self.last_hand_ids: list[int] = []

    def connect(self) -> bool:
        if xh is None:
            self.error_state = True
            self.last_error_code = -1
            self.last_error_message = "xhand_controller is not installed or cannot be imported"
            return False

        self.control = xh.XHandControl()
        comm_type = self.comm_type()

        if self.config.device_name is None:
            devices = self.control.enumerate_devices(comm_type)
            if devices is None or len(devices) == 0:
                self.error_state = True
                self.last_error_code = -2
                self.last_error_message = f"no XHand device found for {comm_type}"
                return False
            self.device_name = devices[0]
        else:
            self.device_name = self.config.device_name

        if comm_type == "RS485":
            err = self.control.open_serial(self.device_name, int(self.config.baudrate))
        elif comm_type == "EtherCAT":
            err = self.control.open_ethercat(self.device_name)
        else:
            self.error_state = True
            self.last_error_code = -3
            self.last_error_message = f"unsupported comm_type: {self.config.comm_type}"
            return False

        if not self.error_ok(err):
            self.save_error(err)
            self.error_state = True
            return False

        try:
            self.last_hand_ids = list(self.control.list_hands_id())
        except Exception:
            self.last_hand_ids = []

        self.connected_flag = True
        self.error_state = False
        self.build_command()

        state = self.get_state(full=False)
        if np.all(np.isfinite(state["qpos"])):
            self.last_qpos_cmd = state["qpos"].copy()
        else:
            self.last_qpos_cmd = self.array12(self.config.home_qpos)
        self.last_cmd_time = time.time()
        return True

    def disconnect(self):
        if self.control is not None:
            self.control.close_device()
        self.connected_flag = False

    def is_connected(self) -> bool:
        return self.control is not None and self.connected_flag and not self.error_state

    def is_error(self) -> bool:
        if self.control is None:
            return True
        if not self.connected_flag:
            return True
        if self.error_state:
            return True
        if self.last_error_code not in [None, 0]:
            return True
        if np.any(self.last_commboard_err != 0):
            return True
        if np.any(self.last_jointboard_err != 0):
            return True
        if np.any(self.last_tipboard_err != 0):
            return True
        return False

    def clear_error(self) -> bool:
        self.error_state = False
        self.last_error_code = None
        self.last_error_message = ""
        self.last_commboard_err[:] = 0
        self.last_jointboard_err[:] = 0
        self.last_tipboard_err[:] = 0
        return self.control is not None and self.connected_flag

    def stop(self) -> bool:
        if self.control is None:
            return False
        command = self.make_command(self.array12(self.config.home_qpos), mode=0, tor_max=0, kp=0, ki=0, kd=0)
        err = self.control.send_command(self.config.device_id, command)
        self.last_action_code = self.error_code(err)
        self.error_state = True
        if not self.error_ok(err):
            self.save_error(err)
            return False
        return True

    def reset(self, qpos: np.ndarray | None = None) -> bool:
        target = self.array12(self.config.home_qpos if qpos is None else qpos)
        return self.send_action(target)

    def move_to_joint_positions(self, qpos: np.ndarray) -> bool:
        return self.reset(qpos)

    def get_state(self, full: bool = False) -> dict[str, Any]:
        err, hand_state = self.read_raw_state(force_update=self.config.force_update_state)
        if not self.error_ok(err) or hand_state is None:
            self.save_error(err)
            self.error_state = True
            state = {
                "qpos": np.full(12, np.nan, dtype=np.float64),
                "current": np.full(12, np.nan, dtype=np.float64),
                "timestamp": time.time(),
            }
            if full:
                state.update(self.empty_full_state())
            return state

        state = self.parse_state(hand_state, full=full)
        self.last_error_code = 0
        self.last_error_message = ""
        return state

    def send_action(self, action: np.ndarray) -> bool:
        if self.control is None or self.hand_command is None:
            return False

        target_qpos = self.array12(action)
        qpos_after_limit = self.limit_joint_range(target_qpos)
        qpos_cmd = self.limit_joint_step(qpos_after_limit)

        self.write_command_positions(qpos_cmd)
        err = self.control.send_command(self.config.device_id, self.hand_command)
        self.last_action_code = self.error_code(err)

        if self.error_ok(err):
            self.last_qpos_cmd = qpos_cmd.copy()
            self.last_cmd_time = time.time()
            return True

        self.save_error(err)
        self.error_state = True
        return False

    def reset_sensor(self, sensor_id: int | None = None) -> bool:
        if self.control is None:
            return False
        sensor_ids = SENSOR_IDS if sensor_id is None else [int(sensor_id)]
        ok = True
        for sid in sensor_ids:
            err = self.control.reset_sensor(self.config.device_id, sid)
            if not self.error_ok(err):
                self.save_error(err)
                ok = False
        return ok

    def build_command(self):
        self.hand_command = self.make_command(self.array12(self.config.home_qpos))

    def make_command(
        self,
        qpos: np.ndarray,
        mode: int | None = None,
        tor_max: int | None = None,
        kp: int | None = None,
        ki: int | None = None,
        kd: int | None = None,
    ):
        command = xh.HandCommand_t()
        mode = self.config.mode if mode is None else mode
        tor_max = self.config.tor_max if tor_max is None else tor_max
        kp = self.config.kp if kp is None else kp
        ki = self.config.ki if ki is None else ki
        kd = self.config.kd if kd is None else kd

        for i in range(12):
            cmd = command.finger_command[i]
            cmd.id = i
            cmd.kp = int(kp)
            cmd.ki = int(ki)
            cmd.kd = int(kd)
            cmd.position = float(qpos[i])
            cmd.tor_max = int(tor_max)
            cmd.mode = int(mode)
            cmd.res0 = 0
            cmd.res1 = 0
            cmd.res2 = 0
            cmd.res3 = 0
        return command

    def write_command_positions(self, qpos: np.ndarray):
        for i in range(12):
            self.hand_command.finger_command[i].position = float(qpos[i])
            self.hand_command.finger_command[i].mode = int(self.config.mode)
            self.hand_command.finger_command[i].tor_max = int(self.config.tor_max)

    def read_raw_state(self, force_update: bool = False):
        if self.control is None:
            return None, None
        result = self.control.read_state(self.config.device_id, force_update)
        return self.unpack_result(result)

    def parse_state(self, hand_state, full: bool = False) -> dict[str, Any]:
        qpos = np.full(12, np.nan, dtype=np.float64)
        current = np.full(12, np.nan, dtype=np.float64)
        finger_ids = np.full(12, -1, dtype=np.int32)
        sensor_ids = np.full(12, -1, dtype=np.int32)
        raw_position = np.full(12, np.nan, dtype=np.float64)
        temperature = np.full(12, np.nan, dtype=np.float64)
        commboard_err = np.zeros(12, dtype=np.int32)
        jointboard_err = np.zeros(12, dtype=np.int32)
        tipboard_err = np.zeros(12, dtype=np.int32)

        finger_state = getattr(hand_state, "finger_state", [])
        for item in finger_state:
            idx = int(getattr(item, "id", -1))
            if idx < 0 or idx >= 12:
                continue
            finger_ids[idx] = idx
            sensor_ids[idx] = int(getattr(item, "sensor_id", -1))
            qpos[idx] = float(getattr(item, "position", np.nan))
            current[idx] = float(getattr(item, "torque", np.nan))
            raw_position[idx] = float(getattr(item, "raw_position", np.nan))
            temperature[idx] = float(getattr(item, "temperature", np.nan))
            commboard_err[idx] = int(getattr(item, "commboard_err", 0))
            jointboard_err[idx] = int(getattr(item, "jonitboard_err", getattr(item, "jointboard_err", 0)))
            tipboard_err[idx] = int(getattr(item, "tipboard_err", 0))

        self.last_commboard_err = commboard_err.copy()
        self.last_jointboard_err = jointboard_err.copy()
        self.last_tipboard_err = tipboard_err.copy()

        state = {
            "qpos": qpos,
            "current": current,
            "timestamp": time.time(),
        }

        if full:
            state.update(
                {
                    "finger_ids": finger_ids,
                    "sensor_ids": sensor_ids,
                    "raw_position": raw_position,
                    "temperature": temperature,
                    "commboard_err": commboard_err,
                    "jointboard_err": jointboard_err,
                    "tipboard_err": tipboard_err,
                    "tactile_force": self.parse_tactile(hand_state, scaled=True),
                    "tactile_force_raw": self.parse_tactile(hand_state, scaled=False),
                    "tactile_force_sum": self.parse_tactile_sum(hand_state, scaled=True),
                    "tactile_force_sum_raw": self.parse_tactile_sum(hand_state, scaled=False),
                    "tactile_temperature": self.parse_tactile_temperature(hand_state),
                    "connected_flag": self.connected_flag,
                    "error_state": self.error_state,
                    "last_action_code": self.last_action_code,
                    "last_error_code": self.last_error_code,
                    "last_error_message": self.last_error_message,
                    "last_joint_limit_clipped": self.last_joint_limit_clipped,
                    "last_delta_limited": self.last_delta_limited,
                    "last_hand_ids": self.last_hand_ids,
                    "comm_type": self.comm_type(),
                    "device_name": self.device_name,
                    "joint_names": JOINT_NAMES,
                    "sensor_names": SENSOR_NAMES,
                }
            )
        return state

    def parse_tactile(self, hand_state, scaled: bool = True) -> np.ndarray:
        tactile = np.zeros((5, 120, 3), dtype=np.float64)
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            sensor_data = getattr(hand_state, "senser_data", [])

        for i, sensor in enumerate(list(sensor_data)[:5]):
            raw_force = getattr(sensor, "raw_force", [])
            for j, force in enumerate(list(raw_force)[:120]):
                tactile[i, j, 0] = float(getattr(force, "fx", 0.0))
                tactile[i, j, 1] = float(getattr(force, "fy", 0.0))
                tactile[i, j, 2] = float(getattr(force, "fz", 0.0))
        if scaled:
            tactile *= self.config.tactile_scale
        return tactile

    def parse_tactile_sum(self, hand_state, scaled: bool = True) -> np.ndarray:
        force_sum = np.zeros((5, 3), dtype=np.float64)
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            sensor_data = getattr(hand_state, "senser_data", [])

        for i, sensor in enumerate(list(sensor_data)[:5]):
            calc_force = getattr(sensor, "calc_force", None)
            if calc_force is None:
                continue
            force_sum[i, 0] = float(getattr(calc_force, "fx", 0.0))
            force_sum[i, 1] = float(getattr(calc_force, "fy", 0.0))
            force_sum[i, 2] = float(getattr(calc_force, "fz", 0.0))
        if scaled:
            force_sum *= self.config.tactile_scale
        return force_sum

    def parse_tactile_temperature(self, hand_state) -> np.ndarray:
        temperature = np.full((5, 20), np.nan, dtype=np.float64)
        sensor_data = getattr(hand_state, "sensor_data", None)
        if sensor_data is None:
            sensor_data = getattr(hand_state, "senser_data", [])

        for i, sensor in enumerate(list(sensor_data)[:5]):
            temp = np.asarray(getattr(sensor, "temperature", []), dtype=np.float64).reshape(-1)
            if temp.size > 0:
                temperature[i, : min(20, temp.size)] = temp[:20]
        return temperature

    def limit_joint_range(self, qpos: np.ndarray) -> np.ndarray:
        if not self.config.clip_joint_limit:
            self.last_joint_limit_clipped = False
            return qpos
        clipped = np.clip(qpos, self.config.qpos_min, self.config.qpos_max)
        self.last_joint_limit_clipped = not np.allclose(qpos, clipped)
        return clipped

    def limit_joint_step(self, target_qpos: np.ndarray) -> np.ndarray:
        if not self.config.use_delta_limit:
            self.last_delta_limited = False
            return target_qpos

        now = time.time()
        if self.last_qpos_cmd is None:
            self.last_qpos_cmd = self.get_state()["qpos"].copy()
            if not np.all(np.isfinite(self.last_qpos_cmd)):
                self.last_qpos_cmd = self.array12(self.config.home_qpos)
        if self.last_cmd_time is None:
            self.last_cmd_time = now

        dt = max(now - self.last_cmd_time, self.config.dt)
        max_step = self.config.max_qvel * dt
        raw_step = target_qpos - self.last_qpos_cmd
        step = np.clip(raw_step, -max_step, max_step)
        self.last_delta_limited = not np.allclose(raw_step, step)
        return self.last_qpos_cmd + step

    def array12(self, value) -> np.ndarray:
        if value is None:
            return np.full(12, np.nan, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size >= 12:
            return arr[:12]
        out = np.full(12, np.nan, dtype=np.float64)
        out[: arr.size] = arr
        return out

    def empty_full_state(self) -> dict[str, Any]:
        return {
            "finger_ids": np.full(12, -1, dtype=np.int32),
            "sensor_ids": np.full(12, -1, dtype=np.int32),
            "raw_position": np.full(12, np.nan, dtype=np.float64),
            "temperature": np.full(12, np.nan, dtype=np.float64),
            "commboard_err": np.zeros(12, dtype=np.int32),
            "jointboard_err": np.zeros(12, dtype=np.int32),
            "tipboard_err": np.zeros(12, dtype=np.int32),
            "tactile_force": np.zeros((5, 120, 3), dtype=np.float64),
            "tactile_force_raw": np.zeros((5, 120, 3), dtype=np.float64),
            "tactile_force_sum": np.zeros((5, 3), dtype=np.float64),
            "tactile_force_sum_raw": np.zeros((5, 3), dtype=np.float64),
            "tactile_temperature": np.full((5, 20), np.nan, dtype=np.float64),
            "connected_flag": self.connected_flag,
            "error_state": self.error_state,
            "last_action_code": self.last_action_code,
            "last_error_code": self.last_error_code,
            "last_error_message": self.last_error_message,
            "last_joint_limit_clipped": self.last_joint_limit_clipped,
            "last_delta_limited": self.last_delta_limited,
            "last_hand_ids": self.last_hand_ids,
            "comm_type": self.comm_type(),
            "device_name": self.device_name,
            "joint_names": JOINT_NAMES,
            "sensor_names": SENSOR_NAMES,
        }

    def comm_type(self) -> str:
        name = str(self.config.comm_type).strip().lower()
        if name in ["rs485", "serial", "usb"]:
            return "RS485"
        if name in ["ethercat", "ethernet", "eth", "ecat"]:
            return "EtherCAT"
        return self.config.comm_type

    def unpack_result(self, result):
        if isinstance(result, tuple) or isinstance(result, list):
            if len(result) >= 2:
                return result[0], result[1]
        if isinstance(result, dict):
            items = list(result.items())
            if len(items) > 0:
                return items[0][0], items[0][1]
        return None, None

    def error_code(self, err) -> int | None:
        if err is None:
            return None
        return int(getattr(err, "error_code", -1))

    def error_ok(self, err) -> bool:
        return err is not None and self.error_code(err) == 0

    def save_error(self, err):
        if err is None:
            self.last_error_code = -1
            self.last_error_message = "empty error object"
            return
        self.last_error_code = self.error_code(err)
        self.last_error_message = str(getattr(err, "error_message", ""))


def print_state(state: dict[str, Any]):
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}, value={np.round(value, 6)}")
        else:
            print(f"{key}: {value}")


def example():
    config = XHandConfig(
        comm_type="RS485",
        device_name="/dev/ttyUSB0",
    )
    hand = XHand(config)

    if not hand.connect():
        raise RuntimeError(f"Failed to connect XHand: {hand.last_error_message}")

    try:
        state = hand.get_state(full=True)
        print_state(state)

        ok = hand.send_action(config.home_qpos)
        print(f"send_action home_qpos ok: {ok}")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    example()