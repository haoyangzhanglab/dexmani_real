#!/usr/bin/env python3
"""连接 XHand1，顺序执行 HOME → fist → palm → V → OK → HOME，到位后各停留 1 秒。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import OperationCancelled, validate_hand_target
from dexmani_real.robot.model import (
    HAND_DOF,
    XHAND_TACTILE_SENSOR_FINGER_IDS,
)
from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE, XHand
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import sample_is_fresh

logger = logging.getLogger("dexmani_real.tools.control_xhand")

HARDWARE_WORKER_ARG = "--_xhand-hardware-worker"
FEEDBACK_POLL_INTERVAL_S = 0.01
CONVERGENCE_SAMPLES = 3
ACTION_DWELL_S = 1.0

# SDK 关节顺序，单位为度；HOME/palm 沿用当前配置，不裁剪超限姿态。
PRESET_QPOS_DEG = {
    "fist": (
        11.85, 74.58, 40, -3.08, 106.02, 109.5,
        109.75, 107.56, 107.66, 109.5, 109.1, 109.15,
    ),
    "v": (38.32, 90, 52.08, 6.21, 2.6, 5.0, 2.1, 5.0, 109.5, 109.5, 109.5, 109.23),
    # 简化 OK：拇指、食指轻弯，其余手指张开，远端关节保留配置要求的 5° 下限。
    "ok": (49.0, 43.0, 48.0, 0.0, 48.0, 48.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
}


def action_sequence(config: HandConfig) -> list[tuple[str, np.ndarray]]:
    actions = (
        ("HOME", config.home_qpos_deg),
        ("fist", PRESET_QPOS_DEG["fist"]),
        ("palm", config.home_qpos_deg),
        ("V", PRESET_QPOS_DEG["v"]),
        ("OK", PRESET_QPOS_DEG["ok"]),
        ("HOME", config.home_qpos_deg),
    )
    result = []
    for name, values in actions:
        try:
            target = validate_hand_target(
                np.deg2rad(values), config.qpos_min_rad, config.qpos_max_rad,
            )
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        result.append((name, target))
    return result


class XHandDiagnostic(XHand):
    """只编排诊断与姿态序列；通信、状态解析和停止复用包内驱动。"""

    def __init__(self, config: HandConfig, *, cancel_requested=lambda: False) -> None:
        super().__init__(config)
        self.cancel_requested = cancel_requested
        self.motion_pending = False

    def check_active(self, deadline_ns: int | None = None) -> int:
        """检查取消及固定期限，并返回本次检查的时刻。"""
        if self.cancel_requested():
            raise OperationCancelled("XHand action sequence interrupted")
        now_ns = time.monotonic_ns()
        if deadline_ns is not None and now_ns >= deadline_ns:
            raise TimeoutError("XHand feedback/action deadline expired")
        return now_ns

    def print_device_info(self) -> None:
        print("\nSDK versions")
        print(f"  Software SDK: {self.control.get_sdk_version()}")

        error_struct, version = self.control.read_version(self.cfg.device_id, 0)
        if error_struct.error_code == 0:
            print(f"  Hardware SDK: {version}")
        else:
            print(f"  Hardware SDK unavailable (error_code={error_struct.error_code})")

        print("\nDevice info")
        error_struct, info = self.control.read_device_info(self.cfg.device_id)
        if error_struct.error_code != 0 or info is None:
            print(f"  Device info unavailable (error_code={error_struct.error_code})")
            return
        print(f"  serial_number: {''.join(info.serial_number[:16])}")
        print(f"  hand_id:       {info.hand_id}")
        print(f"  ev_hand:       {info.ev_hand}")

        error_struct, hand_type = self.control.get_hand_type(self.cfg.device_id)
        if error_struct.error_code == 0:
            print(f"  hand_type:     {hand_type}")
        else:
            print(f"  hand_type unavailable (error_code={error_struct.error_code})")

    def print_state(self, finger_id: int) -> None:
        print(f"\nRead state (finger {finger_id})")
        state = self.read_fresh_state()
        print("  joint positions (deg, SDK ids 0..11):", np.rad2deg(state.qpos).round(3).tolist())
        if finger_id in XHAND_TACTILE_SENSOR_FINGER_IDS:
            index = XHAND_TACTILE_SENSOR_FINGER_IDS.index(finger_id)
            force = state.tactile_aggregate
            print("  tactile aggregate (SDK units):", force[index].tolist() if force is not None else "unavailable")
            if state.tactile_dense is not None:
                dense = state.tactile_dense[index]
                print(f"  tactile dense (SDK units): points={len(dense)} max_abs={np.abs(dense).max():.3f}")
            else:
                print("  tactile dense (SDK units): unavailable")
        else:
            print(f"  joint {finger_id} has no tactile sensor")

    def read_fresh_state(self, *, action_deadline_ns: int | None = None) -> XHandState:
        """CRC 仅重读，所有尝试共用反馈与动作的固定截止时间。"""
        deadline_ns = time.monotonic_ns() + int(self.cfg.feedback_max_age_s * 1e9)
        if action_deadline_ns is not None:
            deadline_ns = min(deadline_ns, action_deadline_ns)
        while True:
            self.check_active(deadline_ns)
            state = self.get_state()
            now_ns = self.check_active(deadline_ns)
            if state is not None and sample_is_fresh(
                state.timestamp_ns, self.cfg.feedback_max_age_s, now_ns,
            ):
                return state
            time.sleep(min(FEEDBACK_POLL_INTERVAL_S, (deadline_ns - now_ns) / 1e9))

    def execute_action(self, target: np.ndarray) -> None:
        """执行 action_sequence 已校验的内部目标，提交后以实测反馈确认到位。"""
        deadline_ns = time.monotonic_ns() + int(self.cfg.home_timeout_s * 1e9)
        self.read_fresh_state(action_deadline_ns=deadline_ns)
        self.check_active(deadline_ns)
        self.motion_pending = True  # SDK 抛错也不能证明目标未送达，仍须停止收尾。
        retry_pending = self._send_position(target) == COMMUNICATION_CRC_ERROR_CODE
        last_sample_ns = time.monotonic_ns()
        print("  Position submission attempted; checking measured convergence")
        tolerance_rad = np.deg2rad(self.cfg.home_tolerance_deg)
        consecutive = 0
        while True:
            state = self.read_fresh_state(action_deadline_ns=deadline_ns)
            if retry_pending:
                self.check_active(deadline_ns)
                self._send_position(target)
                retry_pending = False
                last_sample_ns = time.monotonic_ns()
            elif state.timestamp_ns > last_sample_ns:
                last_sample_ns = state.timestamp_ns
                error_rad = float(np.max(np.abs(state.qpos - target)))
                consecutive = consecutive + 1 if error_rad <= tolerance_rad else 0
                if consecutive >= CONVERGENCE_SAMPLES:
                    self.check_active(deadline_ns)
                    print(f"  Measured target reached; max error={np.rad2deg(error_rad):.3f} deg")
                    return
            time.sleep(FEEDBACK_POLL_INTERVAL_S)

    def dwell(self) -> None:
        """到位后停留期间继续读取反馈，及时暴露通信故障。"""
        deadline_ns = time.monotonic_ns() + int(ACTION_DWELL_S * 1e9)
        while time.monotonic_ns() < deadline_ns:
            self.read_fresh_state()
            remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
            if remaining_s > 0:
                time.sleep(min(FEEDBACK_POLL_INTERVAL_S, remaining_s))
        self.check_active()

    def stop_action(self) -> None:
        if not self.motion_pending:
            return
        state = None
        try:
            # 取消后仍须尝试停止；读取失败时以 passive 收尾并保留读取异常。
            state = self.get_state()
        finally:
            self.stop(
                None if state is None else state.qpos,
                timestamp_ns=0 if state is None else state.timestamp_ns,
            )
            self.motion_pending = False
        print("  Stop command accepted by SDK")


def validate_serial_port(serial_port: str) -> None:
    """在调用原生 SDK 前检查串口；文件系统错误保留原始原因。"""
    port = Path(serial_port)
    if not stat.S_ISCHR(port.stat().st_mode):
        raise ValueError(f"serial port {serial_port} is not a character device")
    if not os.access(port, os.R_OK | os.W_OK):
        raise PermissionError(f"serial port {serial_port} is not readable and writable by the current user")


def run_hardware_session(config: HandConfig, *, finger_id: int, read_only: bool) -> int:
    config.validate()
    if not config.enabled:
        raise ValueError("hand.enabled=False; enable the hand in CONFIG.hand first")
    if not read_only:
        config.validate_home()
    # 连接前验证全部姿态，避免执行到中途才发现实验范围不允许后续目标。
    actions = [] if read_only else action_sequence(config)
    configure_logging()
    interrupted_signal = None
    cleaning_up = False

    def request_interrupt(signum, _):
        nonlocal interrupted_signal
        interrupted_signal = signum
        if not cleaning_up:
            raise OperationCancelled("XHand action sequence interrupted")

    hand = XHandDiagnostic(config, cancel_requested=lambda: interrupted_signal is not None)
    previous_handlers = {
        signum: signal.signal(signum, request_interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    result = 0
    try:
        if config.comm_type == "serial" and config.device_name is not None:
            validate_serial_port(config.device_name)
        hand.check_active()
        hand.connect(cancel_requested=hand.cancel_requested)
        hand.check_active()
        hand.print_device_info()
        hand.print_state(finger_id)
        for name, target in actions:
            print(f"\n  -> {name} (degrees): {np.rad2deg(target).round(3).tolist()}")
            hand.execute_action(target)
            hand.dwell()
    except OperationCancelled as exc:
        print("\nSequence interrupted; stopping any attempted action before closing")
        if interrupted_signal is None:
            print(str(exc), file=sys.stderr)
            result = 2
    except Exception:
        logger.exception("XHand diagnostic failed")
        result = 2
    finally:
        cleaning_up = True
        try:
            for stage, operation in (("stop", hand.stop_action), ("disconnect", hand.disconnect)):
                try:
                    operation()
                except Exception:
                    logger.exception("XHand %s failed; device cleanup unconfirmed", stage)
                    result = 2
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    return result or (128 + interrupted_signal if interrupted_signal is not None else 0)


def run_isolated_hardware_session(argv: list[str]) -> int:
    command = [sys.executable, str(Path(__file__).resolve()), *argv, HARDWARE_WORKER_ARG]
    # 终端信号只由父进程转发一次，避免同组子进程重复接收 Ctrl+C。
    worker = subprocess.Popen(command, start_new_session=True)

    def forward_interrupt(signum, _):
        try:
            worker.send_signal(signum)
        except ProcessLookupError:
            pass

    previous_handlers = {
        signum: signal.signal(signum, forward_interrupt)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        returncode = worker.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if returncode < 0:
        signal_name = signal.strsignal(-returncode)
        print(
            f"\nXHand SDK worker terminated by {signal_name}; hardware stop is unconfirmed.\n"
            "Check hand power, wiring, device permissions and port ownership.",
            file=sys.stderr,
        )
        return 1
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="可信 Python 实验配置，仅使用 CONFIG.hand"
    )
    parser.add_argument(
        "--finger-id", type=int, choices=range(HAND_DOF), default=5,
        help="显示指定 SDK 关节的触觉；带传感器的关节为 2、5、7、9、11",
    )
    parser.add_argument("--read-only", action="store_true", help="只读取设备与状态，不发送姿态或停止命令")
    parser.add_argument(
        "--print-config", action="store_true", help="打印手部配置，不连接硬件"
    )
    parser.add_argument(
        HARDWARE_WORKER_ARG,
        dest="xhand_hardware_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    # 普通运行只在硬件子进程加载一次 Python 配置，避免重复执行配置文件。
    if not args.print_config and not args.xhand_hardware_worker:
        return run_isolated_hardware_session(argv)
    if args.xhand_hardware_worker:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        config = load_config(args.config).hand
        if args.print_config:
            print(json.dumps({"hand": asdict(config)}, ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        return run_hardware_session(config, finger_id=args.finger_id, read_only=args.read_only)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
