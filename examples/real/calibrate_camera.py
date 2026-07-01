#!/usr/bin/env python3
"""ARUCO 手眼标定 — xArm7 + RealSense 相机 (eye-to-hand only).

基于 keyboard_teleop 的机械臂移动逻辑，结合 ArUco 标记检测和
cv2.calibrateHandEye (Tsai 方法) 自动求解相机外参。

结果写入 dexmani_real/config/cameras.json，与 CameraCalib 兼容。

硬件准备:
  1. 打印 ArUco 4x4_50 标记 (ID=0)，尺寸 50mm×50mm
  2. 将标记贴在 xArm7 末端（手底座侧面，平整、对相机可见）
  3. RealSense 相机固定在三脚架上，视野覆盖操作空间
  4. 确保 conda 环境已安装: pyrealsense2, opencv-python, scipy

用法:
  conda activate real
  cd examples/real
  python calibrate_camera.py

操作:
  WASD / ↑↓←→    移动 EEF（与 keyboard_teleop 一致）
  SPACE          采集标定样本（需检测到 ArUco 标记）
  BACKSPACE      删除上一次采集的样本
  ENTER          计算标定并写入 cameras.json（至少 4 组样本）
  Q              退出（丢弃数据）
  ESC            急停退出
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import ASSET_DIR
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.robot.inner_loop import ArmInnerLoop
from dexmani_real.robot.interface import RobotAction, RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.utils.rate_limiter import RateLimiter

# ═══════════════════════════════════════════════ 配置

CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
DELTA_POS = 0.005        # 每次按键平移量 (m)
DELTA_RPY = 0.02         # 每次按键旋转量 (rad)
TARGET_LEAD_MAX = 0.03   # target 领先 arm 的最大距离 (m)

WORKSPACE_BOUNDS = np.array([
    [0.28, 0.72],    # x [min, max] m
    [-0.45, 0.45],   # y [min, max] m
    [0.05, 0.5],     # z [min, max] m
], dtype=np.float64)

# ArUco 标记参数
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.05   # 标记边长 (m)
MARKER_ID = 0          # 期望的标记 ID（None = 接受任意 ID）

# 相机外参保存路径（与 CameraCalib 默认路径一致）
CAMERAS_JSON_PATH = ASSET_DIR.parent / "config" / "cameras.json"

# 用于位姿采集的 ArUco 检测帧数（取中值以提高稳定性）
ARUCO_CAPTURE_FRAMES = 5


# ═══════════════════════════════════════════════ 键盘输入


class KeyState:
    """非阻塞键盘状态追踪（pynput, 线程安全）。"""

    def __init__(self):
        self._keys: set[str] = set()
        self._events: list[str] = []  # 一次性事件: 'space', 'enter', 'backspace'
        self._lock = threading.Lock()
        self._running = True
        self._thread: threading.Thread | None = None

    def _run(self):
        from pynput import keyboard

        def on_press(key):
            try:
                with self._lock:
                    if hasattr(key, "char") and key.char is not None:
                        self._keys.add(key.char.lower())
                    elif key == keyboard.Key.esc:
                        self._keys.add("esc")
                    elif key == keyboard.Key.up:
                        self._keys.add("up")
                    elif key == keyboard.Key.down:
                        self._keys.add("down")
                    elif key == keyboard.Key.left:
                        self._keys.add("left")
                    elif key == keyboard.Key.right:
                        self._keys.add("right")
                    elif key == keyboard.Key.space:
                        if "space" not in self._events:
                            self._events.append("space")
                    elif key == keyboard.Key.enter:
                        if "enter" not in self._events:
                            self._events.append("enter")
                    elif key == keyboard.Key.backspace:
                        if "backspace" not in self._events:
                            self._events.append("backspace")
            except Exception:
                pass

        def on_release(key):
            try:
                with self._lock:
                    if hasattr(key, "char") and key.char is not None:
                        self._keys.discard(key.char.lower())
                    elif key == keyboard.Key.esc:
                        self._keys.discard("esc")
                    elif key == keyboard.Key.up:
                        self._keys.discard("up")
                    elif key == keyboard.Key.down:
                        self._keys.discard("down")
                    elif key == keyboard.Key.left:
                        self._keys.discard("left")
                    elif key == keyboard.Key.right:
                        self._keys.discard("right")
            except Exception:
                pass

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        while self._running:
            time.sleep(0.1)
        self._listener.stop()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def is_pressed(self, key: str) -> bool:
        with self._lock:
            return key in self._keys

    def pop_event(self) -> str | None:
        """取走一个一次性事件，没有则返回 None。"""
        with self._lock:
            if self._events:
                return self._events.pop(0)
            return None


# ═══════════════════════════════════════════════ 姿态工具


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY (rad) → WXYZ 四元数。"""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


# ═══════════════════════════════════════════════ ArUco 检测


def detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    marker_size: float = MARKER_SIZE_M,
    target_id: int | None = MARKER_ID,
) -> tuple[np.ndarray, np.ndarray] | None:
    """检测 ArUco 标记，返回相机系下的 (rvec, tvec) 或 None。

    Returns:
        (rvec, tvec): 旋转向量 (Rodrigues) 和平移向量 (m)，均为 camera→marker。
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    # 如果指定了 target_id，只接受该 ID
    if target_id is not None:
        mask = ids.flatten() == target_id
        if not mask.any():
            return None
        corners = [c for c, m in zip(corners, mask) if m]

    marker_points = np.array([
        [-marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2,  marker_size / 2, 0],
        [ marker_size / 2, -marker_size / 2, 0],
        [-marker_size / 2, -marker_size / 2, 0],
    ], dtype=np.float32)

    rvecs = []
    tvecs = []
    for c in corners:
        _, rv, tv = cv2.solvePnP(
            marker_points, c, intrinsics, distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        rvecs.append(rv)
        tvecs.append(tv)

    # 返回第一个检测到的标记
    return rvecs[0].flatten().astype(np.float64), tvecs[0].flatten().astype(np.float64)


def detect_aruco_stable(
    pipeline: rs.pipeline,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    n_frames: int = ARUCO_CAPTURE_FRAMES,
) -> tuple[np.ndarray, np.ndarray] | None:
    """连续采集 n_frames 帧，返回 ArUco 位姿的中值（更稳定）。"""
    rvecs_all = []
    tvecs_all = []
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        result = detect_aruco_pose(image, intrinsics, distortion)
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return (
        np.median(rvecs_all, axis=0),
        np.median(tvecs_all, axis=0),
    )


# ═══════════════════════════════════════════════ 手眼标定


def calibrate_eye_to_hand(
    tvec_ee2base_list: list[np.ndarray],   # (N, 3)  EE 位置 (m) base 系
    rpy_ee2base_list: list[np.ndarray],    # (N, 3)  EE RPY (rad) base 系
    rvec_marker2camera_list: list[np.ndarray],  # (N, 3)  Rodrigues
    tvec_marker2camera_list: list[np.ndarray],  # (N, 3)  m
) -> np.ndarray:
    """执行 eye-to-hand 手眼标定，返回 T_base_camera (4x4)。

    cv2.calibrateHandEye 输入:
      - R_gripper2base, t_gripper2base: 末端→基座
      - R_target2cam,    t_target2cam:    标记→相机

    输出 (eye-to-hand): R_cam2base, t_cam2base → T_base_camera
    """
    rmat_ee2base = []
    for rpy in rpy_ee2base_list:
        rmat = R.from_euler("xyz", rpy, degrees=False).as_matrix()
        rmat_ee2base.append(rmat)

    rmat_marker2cam = [cv2.Rodrigues(rv)[0] for rv in rvec_marker2camera_list]
    tvec_marker2cam = [tv.reshape(3, 1) for tv in tvec_marker2camera_list]
    tvec_ee2base = [tv.reshape(3, 1) for tv in tvec_ee2base_list]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        rmat_ee2base,
        tvec_ee2base,
        rmat_marker2cam,
        tvec_marker2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_base_camera = np.eye(4, dtype=np.float64)
    T_base_camera[:3, :3] = R_cam2base
    T_base_camera[:3, 3] = t_cam2base.flatten()
    return T_base_camera


def compute_reprojection_error(
    T_base_camera: np.ndarray,
    tvec_ee2base_list: list[np.ndarray],
    rpy_ee2base_list: list[np.ndarray],
    rvec_marker2camera_list: list[np.ndarray],
    tvec_marker2camera_list: list[np.ndarray],
) -> np.ndarray:
    """计算每组样本的重投影误差（mm），返回 (N,) 数组。

    对于 eye-to-hand:
        T_base_camera * T_camera_marker ≈ T_base_ee * T_ee_marker
    由于 T_ee_marker 未知，这里用闭环误差替代:
        inv(T_base_ee) * T_base_camera * T_camera_marker
    应该在所有样本间一致——我们计算其位置部分的标准差作为质量指标。
    """
    T_cam2base_inv = np.linalg.inv(T_base_camera)
    errors = []
    for tvec_ee, rpy_ee, rvec_m2c, tvec_m2c in zip(
        tvec_ee2base_list, rpy_ee2base_list,
        rvec_marker2camera_list, tvec_marker2camera_list,
    ):
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R.from_euler("xyz", rpy_ee, degrees=False).as_matrix()
        T_base_ee[:3, 3] = tvec_ee

        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = cv2.Rodrigues(rvec_m2c)[0]
        T_camera_marker[:3, 3] = tvec_m2c

        # 闭环: T_ee_marker = inv(T_base_ee) * T_base_camera * T_camera_marker
        T_ee_marker = np.linalg.inv(T_base_ee) @ T_base_camera @ T_camera_marker
        errors.append(T_ee_marker[:3, 3].copy())

    errors = np.array(errors)  # (N, 3)
    # 每个样本的 T_ee_marker 位置与均值的偏差 (mm)
    mean_pos = errors.mean(axis=0)
    residuals_mm = np.linalg.norm(errors - mean_pos, axis=1) * 1000.0
    return residuals_mm


# ═══════════════════════════════════════════════ cameras.json 写入


def save_cameras_json(T_base_camera: np.ndarray, serial: str, json_path: Path) -> None:
    """将标定结果写入 cameras.json（pose 格式，与 CameraCalib 兼容）。

    - position: XYZ (m), camera 在 base 系中的位置
    - orientation: WXYZ quaternion, camera→base 的旋转
    """
    rot = R.from_matrix(T_base_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]  # scipy xyzw → wxyz
    pos = T_base_camera[:3, 3]

    entry = {
        "serial": serial,
        "type": "eye_to_hand",
        "pose": {
            "position": [round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6)],
            "orientation": [round(float(quat_wxyz[0]), 6), round(float(quat_wxyz[1]), 6),
                            round(float(quat_wxyz[2]), 6), round(float(quat_wxyz[3]), 6)],
        },
    }

    # 保留现有文件中的其他相机条目
    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    # 自动分配 camera key: camera_0, camera_1, ...
    cam_name = "camera_0"
    for name, cam_data in existing.items():
        if cam_data.get("serial") == serial:
            cam_name = name  # 覆盖同序列号相机的条目
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    # 备份旧文件
    if json_path.exists():
        backup = json_path.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        json_path.rename(backup)
        print(f"  已备份旧文件 → {backup.name}")

    existing[cam_name] = entry
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  标定结果已写入 → {json_path} (camera: {cam_name})")


# ═══════════════════════════════════════════════ 主程序


def main():
    print("=" * 60)
    print("  ArUco 手眼标定 — xArm7 + RealSense (eye-to-hand)")
    print(f"  ArUco: 4x4_50 ID={MARKER_ID} size={MARKER_SIZE_M*1000:.0f}mm")
    print(f"  移动: WASD/↑↓←→  采集: SPACE  标定: ENTER  撤销: BACKSPACE  退出: Q")
    print("=" * 60)

    # ── 1. 连接 xArm7 ──
    arm_config = XArm7Config()
    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf")

    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            base_pose_world=Pose(
                p=np.array([0.0, 0.0, 0.0]),
                q=np.array([np.cos(np.pi / 12), 0.0, 0.0, np.sin(np.pi / 12)]),
            ),
        ),
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(
            teleop_dt=CTRL_DT,
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
            differential_ik_max_pos_step_m=0.05,
        ),
    )

    robot = RobotInterface(
        RobotInterfaceConfig(arm=arm_config),
        kinematics=planner.kin,
        planner=planner,
    )

    print("\n连接 xArm7...")
    result = robot.connect()
    if not result.get("arm"):
        print("❌ arm 连接失败，退出")
        return
    print("  ✓ arm 已连接")

    # ── 2. 启动 ArmInnerLoop ──
    arm_inner = ArmInnerLoop()
    arm_inner.start()
    if not arm_inner.wait_ready(timeout=30.0):
        print("❌ Arm 内环线程启动超时")
        robot.disconnect()
        return
    print("  ✓ Arm 内环就绪 (250Hz)")

    arm_qpos, error_state, _ = arm_inner.get_state()
    state = robot.get_state(arm_qpos=arm_qpos if np.all(np.isfinite(arm_qpos)) else None)
    prev_qpos_cmd = state.arm_qpos.copy()
    target_pos = state.eef_pos.copy()
    target_quat = state.eef_quat_wxyz.copy()

    print(f"\n  当前 EEF: pos={np.round(target_pos, 4)}m  q={np.round(target_quat, 4)}")

    # ── 3. 启动 RealSense ──
    print("\n启动 RealSense 相机...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = rs_config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # 获取相机序列号
    serial = device.get_info(rs.camera_info.serial_number)
    print(f"  序列号: {serial}")

    rs_config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    profile = pipeline.start(rs_config)

    # 获取内参
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    INTRINSICS = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1],
    ])
    DISTORTION = np.array(intr.coeffs)
    print(f"  内参 fx={intr.fx:.1f} fy={intr.fy:.1f} ({intr.width}x{intr.height})")

    # 预热相机（前几帧可能曝光不稳定）
    for _ in range(30):
        pipeline.wait_for_frames()

    # ── 4. 键盘输入 ──
    keys = KeyState()
    keys.start()
    print("\n  键盘控制就绪 ←")

    # ── 5. 标定数据 ──
    samples_tvec_ee2base: list[np.ndarray] = []
    samples_rpy_ee2base: list[np.ndarray] = []
    samples_rvec_marker2cam: list[np.ndarray] = []
    samples_tvec_marker2cam: list[np.ndarray] = []

    # ── 6. 主循环 ──
    limiter = RateLimiter(CTRL_HZ)
    running = True
    loop_count = 0
    wall_warned = [False, False, False]
    last_wall_time = 0.0
    T_base_camera: np.ndarray | None = None

    def _get_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        """获取当前末端位姿 (pos_m, rpy_rad)。"""
        code, pos_list = robot.arm.arm.get_position(is_radian=True)
        if code != 0:
            return np.full(3, np.nan), np.full(3, np.nan)
        pos_arr = np.asarray(pos_list, dtype=np.float64)
        return pos_arr[:3] / 1000.0, pos_arr[3:]  # mm→m

    def _emergency_stop():
        nonlocal running
        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.emergency_stop()
        running = False

    try:
        while running:
            limiter.wait()
            loop_count += 1

            # ── 事件处理 ──
            event = keys.pop_event()
            while event is not None:
                if event == "space":
                    # 采集标定样本
                    print(f"\n  [{len(samples_tvec_ee2base)+1}] 采集 ArUco 位姿...", end=" ", flush=True)
                    ar_result = detect_aruco_stable(pipeline, INTRINSICS, DISTORTION)
                    if ar_result is None:
                        print("❌ 未检测到标记 — 跳过")
                    else:
                        rvec, tvec = ar_result
                        pos_ee, rpy_ee = _get_ee_pose()
                        if np.any(np.isnan(pos_ee)):
                            print("❌ 无法读取末端位姿 — 跳过")
                        else:
                            samples_rvec_marker2cam.append(rvec)
                            samples_tvec_marker2cam.append(tvec)
                            samples_tvec_ee2base.append(pos_ee)
                            samples_rpy_ee2base.append(rpy_ee)
                            print(
                                f"✓ (共 {len(samples_tvec_ee2base)} 组) "
                                f"EE={np.round(pos_ee, 3)}m  "
                                f"marker_dist={np.linalg.norm(tvec):.3f}m"
                            )
                elif event == "backspace":
                    if samples_tvec_ee2base:
                        samples_tvec_ee2base.pop()
                        samples_rpy_ee2base.pop()
                        samples_rvec_marker2cam.pop()
                        samples_tvec_marker2cam.pop()
                        print(f"  ↺ 已撤销，剩余 {len(samples_tvec_ee2base)} 组")
                    else:
                        print("  (无样本可撤销)")
                elif event == "enter":
                    n = len(samples_tvec_ee2base)
                    if n < 4:
                        print(f"  ❌ 至少需要 4 组样本，当前 {n} 组")
                    else:
                        print(f"\n  计算手眼标定 ({n} 组样本)...")
                        T_base_camera = calibrate_eye_to_hand(
                            samples_tvec_ee2base,
                            samples_rpy_ee2base,
                            samples_rvec_marker2cam,
                            samples_tvec_marker2cam,
                        )
                        errors_mm = compute_reprojection_error(
                            T_base_camera,
                            samples_tvec_ee2base,
                            samples_rpy_ee2base,
                            samples_rvec_marker2cam,
                            samples_tvec_marker2cam,
                        )
                        print(f"  重投影误差 (T_ee_marker 一致性):")
                        print(f"    mean={errors_mm.mean():.1f}mm  "
                              f"max={errors_mm.max():.1f}mm  "
                              f"std={errors_mm.std():.1f}mm")
                        print(f"  T_base_camera:")
                        print(f"    pos (m):     {np.round(T_base_camera[:3, 3], 4)}")
                        quat_wxyz = R.from_matrix(T_base_camera[:3, :3]).as_quat()[[3, 0, 1, 2]]
                        print(f"    quat (wxyz): {np.round(quat_wxyz, 4)}")

                        save_cameras_json(T_base_camera, serial, CAMERAS_JSON_PATH)
                        print("  ✓ 标定完成")
                event = keys.pop_event()

            # ── 退出/急停 ──
            if keys.is_pressed("esc"):
                print("\nESC: emergency_stop")
                _emergency_stop()
                break
            if keys.is_pressed("q"):
                print("\nQ: 退出")
                running = False
                break

            # ── 读取状态 ──
            try:
                arm_qpos, error_state, _ = arm_inner.get_state()
                if error_state:
                    print("  ⚠ Arm 内环异常")
                    continue
                state = robot.get_state(arm_qpos=arm_qpos)
            except Exception as e:
                print(f"  ⚠ get_state 异常: {e}")
                continue

            if not np.all(np.isfinite(state.arm_qpos)):
                continue

            # ── EEF target delta from keys ──
            dx = np.zeros(3)
            if keys.is_pressed("w"):     dx[0] += DELTA_POS
            if keys.is_pressed("s"):     dx[0] -= DELTA_POS
            if keys.is_pressed("a"):     dx[1] -= DELTA_POS
            if keys.is_pressed("d"):     dx[1] += DELTA_POS
            if keys.is_pressed("up"):    dx[2] += DELTA_POS
            if keys.is_pressed("down"):  dx[2] -= DELTA_POS

            drpy = np.zeros(3)
            if keys.is_pressed("left"):   drpy[0] += DELTA_RPY
            if keys.is_pressed("right"):  drpy[0] -= DELTA_RPY
            if keys.is_pressed("i"):      drpy[1] += DELTA_RPY
            if keys.is_pressed("k"):      drpy[1] -= DELTA_RPY
            if keys.is_pressed("j"):      drpy[2] -= DELTA_RPY
            if keys.is_pressed("l"):      drpy[2] += DELTA_RPY

            # 周期性状态打印
            if loop_count % 50 == 0:
                n = len(samples_tvec_ee2base)
                print(
                    f"[{loop_count:5d}] "
                    f"eef={np.round(state.eef_pos, 3)}m  "
                    f"samples={n}  "
                    f"{'← 按 SPACE 采集' if n < 4 else '← 按 ENTER 标定'}",
                    flush=True,
                )

            # 无输入 → snap target
            if np.all(dx == 0) and np.all(drpy == 0):
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                prev_qpos_cmd = state.arm_qpos.copy()
                continue

            # ── Target lead limit ──
            lead = np.linalg.norm(target_pos - state.eef_pos)
            if lead > TARGET_LEAD_MAX:
                target_pos = state.eef_pos + (target_pos - state.eef_pos) * (TARGET_LEAD_MAX / lead)

            # ── Workspace soft-wall ──
            new_pos = target_pos + dx
            for axis in range(3):
                lo, hi = WORKSPACE_BOUNDS[axis]
                if lo <= new_pos[axis] <= hi:
                    if dx[axis] != 0:
                        target_pos[axis] = new_pos[axis]
                else:
                    now = time.perf_counter()
                    if not wall_warned[axis] or now - last_wall_time > 3.0:
                        names = ["x", "y", "z"]
                        print(f"  ⚠ {names[axis]} 边界 [{lo:.2f}, {hi:.2f}]")
                        wall_warned[axis] = True
                        last_wall_time = now

            if np.any(drpy != 0):
                dq = _rpy_to_quat_wxyz(drpy[0], drpy[1], drpy[2])
                target_quat = quat_multiply(dq, target_quat)

            # ── IK ──
            target_pose = Pose(p=target_pos, q=target_quat)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)
            if not ik_result.success or ik_result.qpos is None:
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                continue

            prev_qpos_cmd = ik_result.qpos.copy()
            arm_inner.set_target(ik_result.qpos)
            robot.send_action(RobotAction(
                arm_qpos_cmd=ik_result.qpos,
                hand_qpos_cmd=np.zeros(12),
            ))

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — 退出")
    finally:
        keys.stop()
        pipeline.stop()

        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.disconnect()

        n = len(samples_tvec_ee2base)
        if n >= 4 and T_base_camera is None:
            print(f"\n  已采集 {n} 组样本但未执行标定。")
            print("  按 ENTER 可在退出前完成标定，或重新运行脚本。")
        elif T_base_camera is not None:
            print(f"\n  标定已完成 ({n} 组样本)")
        elif n > 0:
            print(f"\n  已丢弃 {n} 组未使用样本。")
        print("  Done.")


if __name__ == "__main__":
    main()
