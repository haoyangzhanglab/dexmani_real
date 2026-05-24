import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from xhand_controller import xhand_control


@dataclass
class XHandConfig:
    comm_type: str = "EtherCAT"          # "EtherCAT" or "RS485"
    ifname: str | None = None             # EtherCAT interface; auto-pick first if None
    serial_port: str | None = None        # RS485 port; auto-pick first if None
    baudrate: int = 3_000_000
    hand_id: int | None = None            # auto-pick first connected hand if None

    mode: int = 3                         # 0: powerless, 3: position control
    kp: int = 100
    ki: int = 0
    kd: int = 0
    tor_max: int = 300                    # SDK range is typically 0~400

    q_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, -0.698, 0.0, -0.087, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.832, 1.57, 1.57, 0.174, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92],
            dtype=np.float32,
        )
    )
    q_home: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))

    force_update: bool = False            # usually only needed for RS485 explicit refresh


class XHand:
    num_joint = 12
    num_finger = 5
    num_tactile_point = 120
    sensor_ids = (17, 18, 19, 20, 21)

    def __init__(self, cfg: XHandConfig):
        self.cfg = cfg
        self.device = xhand_control.XHandControl()
        self.command = xhand_control.HandCommand_t()
        self.finger_command = self.command.finger_command

        self.connected = False
        self.hand_id = cfg.hand_id
        self.mode = int(cfg.mode)

        self.q_cmd = np.zeros(self.num_joint, dtype=np.float32)
        self.step_idx = 0
        self.last_obs_t_ns: int | None = None
        self.last_obs_q: np.ndarray | None = None
        self.obs_cache: dict[str, Any] | None = None
        self.meta_cache: dict[str, Any] | None = None

        self.init_command()

    def init_command(self):
        q = self.clip_q(self.cfg.q_home)
        self.q_cmd = q.copy()

        for j, cmd in enumerate(self.finger_command[: self.num_joint]):
            cmd.id = j
            cmd.kp = int(self.cfg.kp)
            cmd.ki = int(self.cfg.ki)
            cmd.kd = int(self.cfg.kd)
            cmd.position = float(q[j])
            cmd.tor_max = int(self.cfg.tor_max)
            cmd.mode = int(self.mode)

    def require_connected(self, name: str):
        if not self.connected or self.hand_id is None:
            raise RuntimeError(f"{name}: XHand is not connected. Call connect() first.")

    def first_or_raise(self, items, name: str):
        items = list(items)
        if len(items) == 0:
            raise RuntimeError(f"{name}: not found")
        return items[0]

    def check_error(self, err, name: str):
        if int(err.error_code) != 0:
            raise RuntimeError(f"{name}: error_code={err.error_code} message={err.error_message}")

    def connect(self):
        if self.connected:
            return self

        comm_type = str(self.cfg.comm_type).strip()
        if comm_type == "EtherCAT":
            ifname = self.cfg.ifname
            if ifname is None:
                ifname = self.first_or_raise(
                    self.device.enumerate_devices("EtherCAT"),
                    "EtherCAT device",
                )
                self.cfg.ifname = ifname

            err = self.device.open_ethercat(ifname)
            self.check_error(err, f"open_ethercat(ifname={ifname})")

        elif comm_type == "RS485":
            port = self.cfg.serial_port
            if port is None:
                port = self.first_or_raise(
                    self.device.enumerate_devices("RS485"),
                    "RS485 device",
                )
                self.cfg.serial_port = port

            err = self.device.open_serial(port, int(self.cfg.baudrate))
            self.check_error(err, f"open_serial(port={port}, baudrate={self.cfg.baudrate})")

        else:
            raise ValueError(f"unsupported comm_type: {self.cfg.comm_type}")

        hand_ids = [int(x) for x in self.device.list_hands_id()]
        if self.hand_id is None:
            self.hand_id = int(self.first_or_raise(hand_ids, "XHand hand_id"))
        elif int(self.hand_id) not in hand_ids:
            raise RuntimeError(f"hand_id={self.hand_id} not found, available hand_ids={hand_ids}")

        self.connected = True
        return self

    def close(self):
        if self.connected:
            self.device.close_device()
        self.connected = False
        return self

    def set_mode(self, mode: int, kp=None, ki=None, kd=None, tor_max=None):
        self.require_connected("set_mode")

        self.mode = int(mode)
        self.cfg.kp = int(self.cfg.kp if kp is None else kp)
        self.cfg.ki = int(self.cfg.ki if ki is None else ki)
        self.cfg.kd = int(self.cfg.kd if kd is None else kd)
        self.cfg.tor_max = int(self.cfg.tor_max if tor_max is None else tor_max)

        for j, cmd in enumerate(self.finger_command[: self.num_joint]):
            cmd.kp = self.cfg.kp
            cmd.ki = self.cfg.ki
            cmd.kd = self.cfg.kd
            cmd.tor_max = self.cfg.tor_max
            cmd.mode = self.mode
            cmd.position = float(self.q_cmd[j])

        err = self.device.send_command(self.hand_id, self.command)
        self.check_error(err, "set_mode")
        return self

    def send_action(self, q: np.ndarray):
        self.require_connected("send_action")

        q = self.clip_q(q)
        for j, cmd in enumerate(self.finger_command[: self.num_joint]):
            cmd.position = float(q[j])

        err = self.device.send_command(self.hand_id, self.command)
        self.check_error(err, "send_command")

        self.q_cmd = q.copy()
        self.step_idx += 1
        return q

    def get_observation(self, force_update: bool | None = None, full: bool = False):
        self.require_connected("get_observation")

        if force_update is None:
            force_update = bool(self.cfg.force_update and self.cfg.comm_type == "RS485")

        err, state = self.device.read_state(self.hand_id, force_update)
        self.check_error(err, "read_state")

        obs = self.build_observation(state, full=full)
        self.obs_cache = obs
        return obs

    def build_observation(self, state: Any, full: bool = False):
        t_ns = time.monotonic_ns()
        dt = 0.0 if self.last_obs_t_ns is None else max((t_ns - self.last_obs_t_ns) * 1e-9, 0.0)

        q = np.zeros(self.num_joint, dtype=np.float32)
        dq = np.zeros(self.num_joint, dtype=np.float32)
        current = np.zeros(self.num_joint, dtype=np.float32)

        joint_temp = np.zeros(self.num_joint, dtype=np.int32)
        palm_temp = np.zeros(self.num_joint, dtype=np.int32)
        comm_err = np.zeros(self.num_joint, dtype=np.int32)
        joint_err = np.zeros(self.num_joint, dtype=np.int32)
        tip_err = np.zeros(self.num_joint, dtype=np.int32)

        has_velocity = False
        for fs in state.finger_state:
            j = int(fs.id)
            if not 0 <= j < self.num_joint:
                continue

            q[j] = float(fs.position)
            current[j] = float(fs.torque)  # SDK field name is torque; expose as current.

            velocity = getattr(fs, "velocity", None)
            if velocity is not None:
                dq[j] = float(velocity)
                has_velocity = True

            if full:
                temp = int(fs.temperature)
                joint_temp[j] = temp & 0xFF
                palm_temp[j] = (temp >> 8) & 0xFF
                comm_err[j] = int(fs.commboard_err)
                joint_err[j] = int(fs.jonitboard_err)
                tip_err[j] = int(fs.tipboard_err)

        if not has_velocity and dt > 1e-6 and self.last_obs_q is not None:
            dq = (q - self.last_obs_q) / dt

        self.last_obs_t_ns = t_ns
        self.last_obs_q = q.copy()

        sensor_data = list(getattr(state, "sensor_data", []))
        tactile_force = np.zeros((self.num_finger, 3), dtype=np.float32)
        tactile_raw = np.zeros((self.num_finger, self.num_tactile_point, 3), dtype=np.float32)
        tactile_raw_count = np.zeros(self.num_finger, dtype=np.int32)
        tactile_temp = np.zeros(self.num_finger, dtype=np.int32)
        tactile_temp_raw = []

        for i, sd in enumerate(sensor_data[: self.num_finger]):
            tactile_force[i, 0] = float(sd.calc_force.fx) / 10.0
            tactile_force[i, 1] = float(sd.calc_force.fy) / 10.0
            tactile_force[i, 2] = float(sd.calc_force.fz) / 10.0

            raw = np.asarray(
                [[f.fx, f.fy, f.fz] for f in getattr(sd, "raw_force", [])],
                dtype=np.float32,
            )
            if raw.size > 0:
                raw = raw.reshape(-1, 3) / 10.0
                n = min(raw.shape[0], self.num_tactile_point)
                tactile_raw[i, :n] = raw[:n]
                tactile_raw_count[i] = n

            if full:
                tactile_temp[i] = int(sd.calc_temperature)
                tactile_temp_raw.append(np.asarray(list(getattr(sd, "temperature", [])), dtype=np.int32))

        obs = {
            "t_ns": t_ns,
            "dt": dt,
            "step_idx": self.step_idx,
            "q": q,
            "dq": dq,
            "current": current,
            "tactile_force": tactile_force,
            "tactile_raw": tactile_raw,
        }

        if full:
            obs.update(
                joint_temp=joint_temp,
                palm_temp=palm_temp,
                tactile_temp=tactile_temp,
                tactile_raw_count=tactile_raw_count,
                tactile_temp_raw=tactile_temp_raw,
                comm_err=comm_err,
                joint_err=joint_err,
                tip_err=tip_err,
                state=state,
            )

        return obs

    def get_meta_info(self, refresh: bool = False):
        self.require_connected("get_meta_info")

        if self.meta_cache is not None and not refresh:
            return self.meta_cache

        sdk_version = self.device.get_sdk_version()

        err_dev, dev = self.device.read_device_info(self.hand_id)
        self.check_error(err_dev, "read_device_info")

        hand_component_id = int(self.hand_id) | 0x80
        err_ver, hardware_version = self.device.read_version(self.hand_id, hand_component_id)
        self.check_error(err_ver, f"read_version(component_id={hand_component_id})")

        err_type, hand_type = self.device.get_hand_type(self.hand_id)
        self.check_error(err_type, "get_hand_type")

        err_sn, serial_number = self.device.get_serial_number(self.hand_id)
        self.check_error(err_sn, "get_serial_number")

        err_name, hand_name = self.device.get_hand_name(self.hand_id)
        self.check_error(err_name, "get_hand_name")

        self.meta_cache = {
            "sdk_version": sdk_version,
            "hardware_version": hardware_version,
            "hand_id": self.hand_id,
            "hand_type": hand_type,
            "serial_number": serial_number,
            "hand_name": hand_name,
            "ev_hand": getattr(dev, "ev_hand", None),
            "is_calibrated": getattr(dev, "is_calibrated", None),
        }
        return self.meta_cache

    def reset_sensor(self):
        self.require_connected("reset_sensor")

        for sensor_id in self.sensor_ids:
            err = self.device.reset_sensor(self.hand_id, sensor_id)
            self.check_error(err, f"reset_sensor[{sensor_id}]")
        return self

    def reset(self, q: np.ndarray | None = None, sensor: bool = False):
        self.require_connected("reset")

        self.send_action(self.cfg.q_home if q is None else q)
        if sensor:
            self.reset_sensor()
        return self.get_observation(force_update=False)

    def stop(self):
        self.require_connected("stop")
        return self.set_mode(0)

    def clip_q(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (self.num_joint,):
            raise ValueError(f"q must have shape ({self.num_joint},), got {q.shape}")
        return np.clip(q, self.cfg.q_min, self.cfg.q_max)


def example():
    cfg = XHandConfig(comm_type="EtherCAT", ifname=None)
    hand = XHand(cfg)

    try:
        hand.connect()
        print(hand.get_meta_info())

        obs = hand.reset(sensor=False)
        print(obs["q"])
        print(obs["tactile_force"])

        hand.send_action(np.zeros(12, dtype=np.float32))
    finally:
        hand.close()


if __name__ == "__main__":
    example()
