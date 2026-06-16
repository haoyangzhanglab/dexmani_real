#!/usr/bin/env python3
"""Teleoperation entry script.

Assembles all components (robot, tracker, planner, retargeter, controller)
and runs the teleop control loop.

Usage:
    # Dry-run (no hardware, test control logic)
    python scripts/run_teleop.py --dry-run

    # Direct VR (no IPC)
    python scripts/run_teleop.py --arm-ip 192.168.1.113 --hand-device /dev/ttyUSB0

    # IPC mode (VR frames from SharedRingBuffer)
    python scripts/run_teleop.py --ipc --arm-ip 192.168.1.113

    # VR + hand only
    python scripts/run_teleop.py --vr-only --hand-device /dev/ttyUSB0

    # Custom arm EMA
    python scripts/run_teleop.py --ema-arm 0.5
"""

from __future__ import annotations

import argparse
import multiprocessing
import signal

import numpy as np

from dexmani_real.controller.teleop_controller import TeleopController
from dexmani_real.planner.arm_planner import XArm7MotionPlanner
from dexmani_real.planner.planner_types import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7PlannerConfig,
)
from dexmani_real.robot.robot_interface import RobotInterface, RobotInterfaceConfig
from dexmani_real.teleop.arm_wrist_mapper import ArmWristMapper
from dexmani_real.teleop.hand_retarget import XHandRetargeter
from dexmani_real.teleop.quest_hand_tracker import QuestHandTracker

# ── Paths ─────────────────────────────────────────────────────────
from dexmani_real import ASSET_DIR

URDF_PATH = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
SRDF_PATH = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf")
HAND_URDF_PATH = str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf")
EEF_LINK_NAME = "custom_eef_link"

# Robot base relative to world: +30° yaw about Z
import math

BASE_YAW_DEG = 30.0
_HALF_YAW = math.radians(BASE_YAW_DEG / 2.0)
BASE_POSE_WORLD = Pose(
    p=np.zeros(3, dtype=np.float64),
    q=np.array([math.cos(_HALF_YAW), 0.0, 0.0, math.sin(_HALF_YAW)], dtype=np.float64),
)


def build_planner() -> XArm7MotionPlanner:
    config = XArm7PlannerConfig(
        urdf_path=URDF_PATH,
        srdf_path=SRDF_PATH,
        eef_link_name=EEF_LINK_NAME,
        base_pose_world=BASE_POSE_WORLD,
    )
    return XArm7MotionPlanner(
        config=config,
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(),
    )


def build_controller(args: argparse.Namespace) -> TeleopController:
    """Factory: assemble all components from CLI args."""
    planner = build_planner()

    # Robot
    from dexmani_real.robot.xarm7 import XArm7Config
    from dexmani_real.robot.xhand import XHandConfig

    arm_config = XArm7Config(ip=args.arm_ip) if not args.arm_only_vr else XArm7Config()
    hand_config = XHandConfig(
        comm_type="RS485",
        device_name=args.hand_device,
    )

    robot_config = RobotInterfaceConfig(
        arm=arm_config,
        hand=hand_config,
        hand_urdf_path=HAND_URDF_PATH,
    )

    robot = RobotInterface(
        config=robot_config,
        kinematics=planner.kin,
        planner=planner,
    )

    # Tracker
    tracker = None
    if args.dry_run or args.arm_only_vr:
        # In dry-run, provide a dummy tracker that always returns a fresh frame
        import time as _time

        class _DummyTracker:
            def __init__(self):
                self._seq = 0
                self.started = True
            def get_latest(self, max_age_s=None):
                self._seq += 1
                return {
                    "side": "right",
                    "wrist_pos": np.zeros(3, dtype=np.float64),
                    "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    "landmarks": np.zeros((21, 3), dtype=np.float64),
                    "recv_ts_ns": _time.monotonic_ns(),
                    "source_ts_ns": _time.monotonic_ns(),
                    "sequence_id": self._seq,
                    "source_frame_seq": self._seq,
                    "coordinate_frame": "flu",
                    "local_recv_ns": _time.monotonic_ns(),
                }
            def connect(self): pass
            def disconnect(self): pass
        tracker = _DummyTracker()
    elif args.ipc:
        tracker = None  # VR frames come from IPC
    else:
        tracker = QuestHandTracker(
            transport=args.vr_transport,
            host=args.vr_host,
            port=args.vr_port,
            hand_side=args.vr_hand_side,
            output_frame="flu",
            max_frame_age_s=0.20,
            verbose=True,
        )

    # IPC buffer
    ipc_buffer = None
    if args.ipc:
        from dexmani_real.ipc.shared_ring_buffer import RingBufferConfig, SharedRingBuffer

        ipc_config = RingBufferConfig(slot_count=64, slot_size=1_048_576, create=False)
        ipc_buffer = SharedRingBuffer(args.ipc_name, ipc_config)

    # Arm mapper
    arm_mapper = ArmWristMapper(
        pos_scale=args.mapper_pos_scale,
        rot_scale=args.mapper_rot_scale,
    )

    # Retargeter
    retargeter = XHandRetargeter()

    # Keyboard queue
    keyboard_queue: multiprocessing.Queue = multiprocessing.Queue()

    controller = TeleopController(
        robot=robot,
        arm_mapper=arm_mapper,
        retargeter=retargeter,
        planner=planner,
        tracker=tracker,
        ipc_buffer=ipc_buffer,
        keyboard_queue=keyboard_queue,
        target_hz=float(args.rate),
        ema_alpha_arm=float(args.ema_arm),
        dry_run=args.dry_run,
    )
    return controller


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teleoperation controller for xArm7 + XHand via Quest VR"
    )
    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        help="No hardware — dummy state, test control logic")
    parser.add_argument("--ipc", action="store_true",
                        help="Read VR frames from SharedRingBuffer")
    parser.add_argument("--ipc-name", default="vr_frame",
                        help="SharedRingBuffer name (default: vr_frame)")
    parser.add_argument("--vr-only", action="store_true",
                        help="VR + hand only (no arm hardware)")

    # Hardware
    parser.add_argument("--arm-ip", default="192.168.1.113",
                        help="xArm IP address")
    parser.add_argument("--hand-device", default="/dev/ttyUSB0",
                        help="XHand serial device")
    parser.add_argument("--arm-only-vr", action="store_true",
                        help="No arm connection (use with --ipc + VR process)")

    # VR
    parser.add_argument("--vr-transport", default="tcp_server",
                        choices=["tcp_client", "tcp_server", "udp"],
                        help="VR transport mode")
    parser.add_argument("--vr-host", default="0.0.0.0",
                        help="VR host/interface")
    parser.add_argument("--vr-port", type=int, default=8000,
                        help="VR port")
    parser.add_argument("--vr-hand-side", default="right",
                        choices=["left", "right"],
                        help="Which hand to track")

    # Control
    parser.add_argument("--rate", type=float, default=50.0,
                        help="Control loop frequency (Hz)")
    parser.add_argument("--ema-arm", type=float, default=0.3,
                        help="EMA alpha for arm smoothing")
    # Mapper
    parser.add_argument("--mapper-pos-scale", type=float, default=1.0,
                        help="Position scale for wrist mapping")
    parser.add_argument("--mapper-rot-scale", type=float, default=1.0,
                        help="Rotation scale for wrist mapping")

    args = parser.parse_args()

    controller = build_controller(args)

    # Start tracker if direct mode
    if controller.tracker is not None and not args.dry_run:
        print(f"[run_teleop] Connecting VR tracker ({args.vr_transport}://{args.vr_host}:{args.vr_port})...")
        if not controller.tracker.connect():
            print(f"[run_teleop] ERROR: Tracker connect failed: {controller.tracker.last_error}")
            return

    def shutdown(sig=None, frame=None):
        print("\n[run_teleop] Shutdown signal received.")
        controller.stop()
        if controller.tracker is not None:
            controller.tracker.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        controller.run()
    finally:
        if controller.tracker is not None:
            controller.tracker.disconnect()
        print("[run_teleop] Done.")


if __name__ == "__main__":
    main()
