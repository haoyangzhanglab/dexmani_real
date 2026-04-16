import time
import numpy as np
from typing import Any
from dataclasses import dataclass, field
from xhand_controller import xhand_control


@dataclass
class XHandConfig:
    comm_type: str = "EtherCAT"          # "EtherCAT" or "RS485"
    ifname: str | None = None            # EtherCAT网卡名；None时自动枚举
    serial_port: str | None = None       # RS485串口；None时自动枚举
    baudrate: int = 3_000_000
    hand_id: int | None = None

    default_mode: int = 3
    default_kp: int = 100
    default_ki: int = 0
    default_kd: int = 0
    default_tor_max: int = 300

    safe_joint_lower: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, -0.698, 0.0, -0.087, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )
    safe_joint_upper: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.832, 1.57, 1.57, 0.174, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92, 1.92],
            dtype=np.float32,
        )
    )
    home_qpos: np.ndarray = field(
        default_factory=lambda: np.zeros(12, dtype=np.float32)
    )

    default_force_update: bool = False
    verbose: bool = True


class XHand:
    NUM_JOINTS = 12
    NUM_FINGERS = 5
    SENSOR_IDS = [17, 18, 19, 20, 21]

    def __init__(self, config: XHandConfig):
        self.config = config
        self.device = xhand_control.XHandControl()
        self.command = xhand_control.HandCommand_t()

        self.connected = False
        self.started = False

        self.hand_id = config.hand_id
        self.mode = int(config.default_mode)

        self.last_qpos: np.ndarray | None = None
        self.last_obs: dict[str, Any] | None = None
        self.last_state: Any | None = None
        self.meta_info: dict[str, Any] | None = None

        self.init_command_template()


    def init_command_template(self) -> None:
        home_qpos = self.normalize_qpos(self.config.home_qpos)
        for i in range(self.NUM_JOINTS):
            cmd = self.command.finger_command[i]
            cmd.id = i
            cmd.kp = int(self.config.default_kp)
            cmd.ki = int(self.config.default_ki)
            cmd.kd = int(self.config.default_kd)
            cmd.position = float(home_qpos[i])
            cmd.tor_max = int(self.config.default_tor_max)
            cmd.mode = int(self.mode)


    def connect(self) -> None:
        if self.connected:
            return

        if self.config.comm_type == "EtherCAT":
            ifname = self.config.ifname
            if ifname is None:
                devices = self.device.enumerate_devices("EtherCAT")
                if not devices:
                    raise RuntimeError("No EtherCAT interface found")
                if len(devices) > 1:
                    raise RuntimeError(
                        f"Multiple EtherCAT interfaces found: {devices}. "
                        "Please set config.ifname explicitly."
                    )
                ifname = devices[0]
                self.config.ifname = ifname

            err = self.device.open_ethercat(ifname)
            self.raise_on_error(err, "connect")

        elif self.config.comm_type == "RS485":
            port = self.config.serial_port
            if port is None:
                devices = self.device.enumerate_devices("RS485")
                if not devices:
                    raise RuntimeError("No RS485 serial port found")
                if len(devices) > 1:
                    raise RuntimeError(
                        f"Multiple RS485 ports found: {devices}. "
                        "Please set config.serial_port explicitly."
                    )
                port = devices[0]
                self.config.serial_port = port

            err = self.device.open_serial(port, int(self.config.baudrate))
            self.raise_on_error(err, "connect")

        else:
            raise ValueError(f"Unsupported comm_type: {self.config.comm_type}")

        if self.hand_id is None:
            hand_ids = self.device.list_hands_id()
            if not hand_ids:
                raise RuntimeError("No XHand device found after open")
            self.hand_id = int(hand_ids[0])

        self.connected = True

        if self.config.verbose:
            if self.config.comm_type == "EtherCAT":
                print(f"[XHand] connected via EtherCAT, ifname={self.config.ifname}, hand_id={self.hand_id}")
            else:
                print(f"[XHand] connected via RS485, port={self.config.serial_port}, hand_id={self.hand_id}")


    def start(self) -> None:
        if self.started:
            return

        self.connect()
        self.reset()
        self.meta_info = self.get_meta_info(refresh=True)
        self.last_obs = self.get_observation()
        self.started = True

        if self.config.verbose:
            print("[XHand] started")


    def close(self) -> None:
        if not self.connected:
            return
        self.device.close_device()
        self.connected = False
        self.started = False
        if self.config.verbose:
            print("[XHand] closed")


    def disconnect(self) -> None:
        self.close()


    def is_connected(self) -> bool:
        return self.connected


    def set_mode(
        self,
        mode: int,
        kp: int | None = None,
        ki: int | None = None,
        kd: int | None = None,
        tor_max: int | None = None,
    ) -> None:
        self.mode = int(mode)

        if kp is not None:
            self.config.default_kp = int(kp)
        if ki is not None:
            self.config.default_ki = int(ki)
        if kd is not None:
            self.config.default_kd = int(kd)
        if tor_max is not None:
            self.config.default_tor_max = int(tor_max)

        self.apply_mode_template()

        if self.hand_id is None:
            raise RuntimeError("Device is not connected")

        err = self.device.send_command(self.hand_id, self.command)
        self.raise_on_error(err, "set_mode")


    def apply_mode_template(self) -> None:
        for i in range(self.NUM_JOINTS):
            cmd = self.command.finger_command[i]
            cmd.kp = int(self.config.default_kp)
            cmd.ki = int(self.config.default_ki)
            cmd.kd = int(self.config.default_kd)
            cmd.tor_max = int(self.config.default_tor_max)
            cmd.mode = int(self.mode)


    def send_action(self, qpos: np.ndarray) -> None:
        if self.hand_id is None:
            raise RuntimeError("Device is not connected")

        qpos = self.normalize_qpos(qpos)
        qpos = self.clip_qpos(qpos)

        for i in range(self.NUM_JOINTS):
            self.command.finger_command[i].position = float(qpos[i])

        err = self.device.send_command(self.hand_id, self.command)
        self.raise_on_error(err, "send_action")
        self.last_qpos = qpos.copy()


    def get_observation(self, force_update: bool | None = None) -> dict[str, Any]:
        if self.hand_id is None:
            raise RuntimeError("Device is not connected")

        if force_update is None:
            force_update = self.config.default_force_update

        err, state = self.device.read_state(self.hand_id, force_update)
        self.raise_on_error(err, "get_observation")

        obs = self.parse_state(state, err)
        self.last_state = state
        self.last_obs = obs
        return obs


    def get_last_observation(self) -> dict[str, Any] | None:
        return self.last_obs


    def is_error(self) -> bool:
        if self.last_obs is None:
            return False
        return not self.last_obs["debug_info"]["error"]["ok"]


    def stop(self) -> None:
        self.set_mode(0)


    def get_meta_info(self, refresh: bool = False) -> dict[str, Any]:
        if self.meta_info is not None and not refresh:
            return self.meta_info

        if self.hand_id is None:
            raise RuntimeError("Device is not connected")

        sdk_version = self.device.get_sdk_version()

        err_dev, dev = self.device.read_device_info(self.hand_id)
        self.raise_on_error(err_dev, "read_device_info")

        err_ver, hw_version = self.device.read_version(self.hand_id, 0)
        self.raise_on_error(err_ver, "read_version")

        err_type, hand_type = self.device.get_hand_type(self.hand_id)
        self.raise_on_error(err_type, "get_hand_type")

        err_sn, serial_number = self.device.get_serial_number(self.hand_id)
        self.raise_on_error(err_sn, "get_serial_number")

        err_name, hand_name = self.device.get_hand_name(self.hand_id)
        self.raise_on_error(err_name, "get_hand_name")

        self.meta_info = {
            "sdk_version": sdk_version,
            "hardware_version": hw_version,
            "hand_id": self.hand_id,
            "hand_type": hand_type,
            "serial_number": serial_number,
            "hand_name": hand_name,
            "ev_hand": getattr(dev, "ev_hand", None),
            "is_calibrated": getattr(dev, "is_calibrated", None),
        }
        return self.meta_info


    def reset_sensor(self) -> None:
        if self.hand_id is None:
            raise RuntimeError("Device is not connected")

        for sensor_id in self.SENSOR_IDS:
            err = self.device.reset_sensor(self.hand_id, sensor_id)
            self.raise_on_error(err, f"reset_sensor[{sensor_id}]")


    def reset(self) -> None:
        self.send_action(self.config.home_qpos)
        self.reset_sensor()
        self.last_obs = self.get_observation(force_update=True)

        if self.config.verbose:
            print("[XHand] reset done")


    def handle_exception(self) -> None:
        if self.config.verbose:
            print("[XHand] recover from exception")
        try:
            self.close()
        except Exception:
            pass
        time.sleep(0.2)
        self.connect()
        self.start()
        self.reset()

    # 输入规范化和安全检查
    def normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float32)
        if qpos.shape != (self.NUM_JOINTS,):
            raise ValueError(f"qpos shape must be ({self.NUM_JOINTS},), got {qpos.shape}")
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains NaN or Inf")
        return qpos


    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.maximum(qpos, self.config.safe_joint_lower)
        qpos = np.minimum(qpos, self.config.safe_joint_upper)
        return qpos


    def parse_state(self, state: Any, err: Any) -> dict[str, Any]:
        qpos = np.zeros(self.NUM_JOINTS, dtype=np.float32)
        torque = np.zeros(self.NUM_JOINTS, dtype=np.float32)  # 实际为实时电流，单位 mA

        joint_temp = np.zeros(self.NUM_JOINTS, dtype=np.int32)
        palm_temp = np.zeros(self.NUM_JOINTS, dtype=np.int32)

        commboard_err = np.zeros(self.NUM_JOINTS, dtype=np.int32)
        jointboard_err = np.zeros(self.NUM_JOINTS, dtype=np.int32)
        tipboard_err = np.zeros(self.NUM_JOINTS, dtype=np.int32)

        raw_finger_state = []

        for i, fs in enumerate(state.finger_state):
            qpos[i] = float(fs.position)
            torque[i] = float(fs.torque)

            temp = int(fs.temperature)
            joint_temp[i] = temp & 0xFF
            palm_temp[i] = (temp >> 8) & 0xFF

            commboard_err[i] = int(fs.commboard_err)
            jointboard_err[i] = int(fs.jonitboard_err)
            tipboard_err[i] = int(fs.tipboard_err)

            raw_finger_state.append(
                {
                    "id": int(fs.id),
                    "sensor_id": int(fs.sensor_id),
                    "position": float(fs.position),
                    "torque": int(fs.torque),
                    "raw_position": int(fs.raw_position),
                    "temperature": int(fs.temperature),
                    "commboard_err": int(fs.commboard_err),
                    "jointboard_err": int(fs.jonitboard_err),
                    "tipboard_err": int(fs.tipboard_err),
                }
            )

        sensor_data = getattr(state, "sensor_data", None)
        if sensor_data is None:
            raise RuntimeError("state.sensor_data not found in current SDK binding")

        fingertip_net_force = np.zeros((self.NUM_FINGERS, 3), dtype=np.float32)
        fingertip_contact_array = np.zeros((self.NUM_FINGERS, 120, 3), dtype=np.float32)
        fingertip_temp = np.zeros(self.NUM_FINGERS, dtype=np.int32)
        fingertip_temp_array = np.zeros((self.NUM_FINGERS, 20), dtype=np.int32)

        raw_sensor_data = []

        # 假设顺序为 thumb, index, middle, ring, little
        for i in range(min(self.NUM_FINGERS, len(sensor_data))):
            sd = sensor_data[i]

            fingertip_net_force[i, 0] = float(sd.calc_force.fx) / 10.0
            fingertip_net_force[i, 1] = float(sd.calc_force.fy) / 10.0
            fingertip_net_force[i, 2] = float(sd.calc_force.fz) / 10.0

            raw_force = np.asarray(
                [[float(f.fx), float(f.fy), float(f.fz)] for f in sd.raw_force],
                dtype=np.float32,
            )
            n = min(raw_force.shape[0], 120)
            if n > 0:
                fingertip_contact_array[i, :n] = raw_force[:n] / 10.0

            if hasattr(sd, "temperature"):
                temp_array = np.asarray(list(sd.temperature), dtype=np.int32)
                n_temp = min(temp_array.shape[0], 20)
                if n_temp > 0:
                    fingertip_temp_array[i, :n_temp] = temp_array[:n_temp]

            fingertip_temp[i] = int(sd.calc_temperature)

            raw_sensor_data.append(
                {
                    "calc_force": [
                        float(sd.calc_force.fx),
                        float(sd.calc_force.fy),
                        float(sd.calc_force.fz),
                    ],
                    "raw_force": raw_force.tolist(),
                    "temperature": fingertip_temp_array[i].tolist(),
                    "calc_temperature": int(sd.calc_temperature),
                }
            )

        error_info = {
            "ok": bool(
                int(err.error_code) == 0
                and np.all(commboard_err == 0)
                and np.all(jointboard_err == 0)
                and np.all(tipboard_err == 0)
            ),
            "error_code": int(err.error_code),
            "error_message": str(err.error_message),
            "commboard_err": commboard_err,
            "jointboard_err": jointboard_err,
            "tipboard_err": tipboard_err,
        }

        return {
            "timestamp": time.time(),
            "qpos": qpos,
            "torque": torque,
            "fingertip_net_force": fingertip_net_force,
            "fingertip_contact_array": fingertip_contact_array,
            "temperature": {
                "joint": joint_temp,
                "palm": palm_temp,
                "fingertip": fingertip_temp,
                "fingertip_array": fingertip_temp_array,
            },
            "debug_info": {
                "raw_obs": {
                    "finger_state": raw_finger_state,
                    "sensor_data": raw_sensor_data,
                },
                "error": error_info,
            },
        }


    def raise_on_error(self, err: Any, where: str) -> None:
        if int(err.error_code) != 0:
            raise RuntimeError(f"[{where}] error_code={err.error_code}, message={err.error_message}")


if __name__ == "__main__":
    cfg = XHandConfig(
        comm_type="EtherCAT",
        ifname=None,   # None时自动枚举；多张网卡时会提示你显式指定
        verbose=True,
    )

    # RS485的话这样写：
    # cfg = XHandConfig(
    #     comm_type="RS485",
    #     serial_port=None,   # None时自动枚举；多串口时会提示你显式指定
    #     verbose=True,
    # )

    hand = XHand(cfg)
    hand.start()

    print(hand.get_meta_info())

    qpos = np.zeros(12, dtype=np.float32)
    hand.send_action(qpos)

    obs = hand.get_observation()
    print(obs["qpos"])
    print(obs["fingertip_net_force"])

    hand.reset()
    hand.close()