#!/usr/bin/env python3
"""ARUCO 手眼标定 — xArm7 + RealSense 相机 (eye-to-hand only).

基于 keyboard_teleop 的机械臂移动逻辑，结合 ArUco 标记检测和
cv2.calibrateHandEye（5 种算法比选）自动求解相机外参。

手眼解在 xArm 基座系得到 T_base_camera，脚本再用 base_pose_world 转到 **world 系**
后写盘（与下游 eef_pos/arm_ee = compute_eef_pose_world 保持同一坐标系）。

结果写入 dexmani_real/config/cameras.json，与 CameraCalib 兼容。

硬件准备:
  1. 打印 ArUco 7x7_50 标记 (ID=1)，尺寸 98.2mm×98.2mm
  2. 将标记贴在 xArm7 末端（手底座侧面，平整、对相机可见）
  3. RealSense 相机固定在三脚架上，视野覆盖操作空间
  4. 确保 conda 环境已安装: pyrealsense2, opencv-python, scipy

用法:
  conda activate real
  cd examples/real
  python calibrate_camera.py

操作:
  WASD / ↑↓       移动 EEF（与 keyboard_teleop 一致）
  ←→             绕 X 轴旋转 (roll)
  I / K          绕 Y 轴旋转 (pitch)
  J / L          绕 Z 轴旋转 (yaw)
  SPACE          采集标定样本（需检测到 ArUco 标记）
  BACKSPACE      删除上一次采集的样本
  X              删除残差最大的最差帧（需先按 ENTER 评估质量）
  ENTER          计算标定并写入 cameras.json（至少 10 组，推荐 10~20 组）
  R              归位（return_home，2 段式路径）
  Q              退出（丢弃数据）
  ESC            急停退出

注意: 手眼标定必须有姿态变化才能解出旋转，请务必用旋转键在各样本间改变末端
      朝向、覆盖不共线的旋转轴；纯平移采样会导致标定退化。
      标定质量不达标时: ENTER 查看逐帧残差 → X 删最差帧 → 再 ENTER 复算，迭代提质。
"""

from __future__ import annotations

import json
import sys
import termios
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

from dexmani_real import ASSET_DIR, PACKAGE_DIR
from dexmani_real.planning import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.pose_utils import quat_multiply
from dexmani_real.robot.inner_loop import ArmInnerLoop

try:
    from pynput import keyboard  # type: ignore[import-untyped]
except ImportError:
    raise ImportError(
        "pynput is required for keyboard input. Install with: pip install pynput"
    )
from dexmani_real.robot.interface import RobotInterface, RobotInterfaceConfig
from dexmani_real.robot.xarm7 import XArm7Config
from dexmani_real.utils.rate_limiter import RateLimiter

# ═══════════════════════════════════════════════ 配置

CTRL_HZ = 50.0
CTRL_DT = 1.0 / CTRL_HZ
DELTA_POS = 0.005        # 每次按键平移量 (m)
DELTA_RPY = 0.02         # 每次按键旋转量 (rad)
TARGET_LEAD_MAX = 0.03   # target 领先 arm 的最大距离 (m)
HOME_DT = 0.04           # 归位 waypoint 间隔 (s): ~25°/s（保守，避免归位过快）

WORKSPACE_BOUNDS = np.array([
    [0.28, 0.72],    # x [min, max] m
    [-0.45, 0.45],   # y [min, max] m
    [0.05, 0.5],     # z [min, max] m
], dtype=np.float64)

# ArUco 标记参数
ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"
MARKER_SIZE_M = 0.0982   # 标记边长 (m)
MARKER_ID = 1          # 期望的标记 ID（None = 接受任意 ID）

# 相机外参保存路径（包内 config/，与 CameraCalib 默认读取路径一致）
CAMERAS_JSON_PATH = PACKAGE_DIR / "config" / "cameras.json"

# 用于位姿采集的 ArUco 检测帧数（取中值以提高稳定性）
ARUCO_CAPTURE_FRAMES = 5

# 进入标定计算所需的最少样本数
MIN_SAMPLES = 10

# 标定质量门槛：T_ee_marker 位置一致性 std 超过此值(mm)则拒绝写盘
MAX_CONSISTENCY_STD_MM = 5.0

# 标定质量门槛(旋转)：T_ee_marker 旋转一致性 std 超过此值(deg)则拒绝写盘。
# ArUco 单标记朝向本身较嘈杂(平面翻转歧义)，故阈值偏宽松，可按需收紧。
MAX_CONSISTENCY_ROT_STD_DEG = 3.0


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
        def on_press(key):
            try:
                with self._lock:
                    if hasattr(key, "char") and key.char is not None:
                        ch = key.char.lower()
                        if ch == "x":
                            # X = 一次性事件（删除残差最大的最差帧），不作为按住键
                            if "x" not in self._events:
                                self._events.append("x")
                        else:
                            self._keys.add(ch)
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
        # 立即停止监听器并 join 线程，避免退出后仍捕获按键、污染终端
        listener = getattr(self, "_listener", None)
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

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


def _pose_wxyz_to_matrix(p: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """(position, WXYZ 四元数) → 4x4 齐次矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
    T[:3, 3] = p
    return T


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
    # 单个 ArUco marker 开启角点子像素精修以提升位姿精度（对单 marker 有益；
    # ChArUco 板相反，应保持 CORNER_REFINE_NONE）。
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
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


# marker 四角在标记坐标系下的 3D 坐标（用于预览时画坐标轴）
_MARKER_CORNERS_3D = np.array([
    [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0],
    [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
    [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0],
], dtype=np.float32)


def draw_overlay(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    n_samples: int,
    target_id: int | None = MARKER_ID,
) -> tuple[np.ndarray, bool]:
    """在彩色帧上叠加 ArUco 检测框/坐标轴与状态文字（原地绘制）。

    Returns:
        (annotated_bgr, detected): detected 为是否检测到目标 ID 的标记。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    detected = False
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        for c, mid in zip(corners, ids.flatten()):
            if target_id is not None and mid != target_id:
                continue
            ok, rv, tv = cv2.solvePnP(
                _MARKER_CORNERS_3D, c, intrinsics, distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                cv2.drawFrameAxes(image, intrinsics, distortion, rv, tv, MARKER_SIZE_M * 0.5)
                detected = True

    color = (0, 200, 0) if detected else (0, 0, 255)
    status = f"MARKER {'OK' if detected else 'NOT FOUND'}"
    cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    hint = f"samples={n_samples}/{MIN_SAMPLES}  " + (
        "ENTER=calibrate" if n_samples >= MIN_SAMPLES else f"SPACE=capture (need >={MIN_SAMPLES})"
    )
    cv2.putText(image, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image, detected


# ═══════════════════════════════════════════════ 手眼标定

# OpenCV 的 5 种手眼算法。文献(PLOS ONE 2022)指出 TSAI 对旋转噪声最敏感，PARK 最稳；
# 这里全部跑一遍、取一致性最优的结果（见 calibrate_and_select）。
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def calibrate_eye_to_hand(
    tvec_ee2base_list: list[np.ndarray],   # (N, 3)  EE 位置 (m) base 系
    rpy_ee2base_list: list[np.ndarray],    # (N, 3)  EE RPY (rad) base 系
    rvec_marker2camera_list: list[np.ndarray],  # (N, 3)  Rodrigues
    tvec_marker2camera_list: list[np.ndarray],  # (N, 3)  m
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> np.ndarray:
    """执行 eye-to-hand 手眼标定，返回 T_base_camera (4x4)。

    cv2.calibrateHandEye 本身面向 eye-in-hand（相机装在末端），输入 gripper2base +
    target2cam 时返回的是 T_cam2gripper。本场景是 eye-to-hand（相机固定、marker 贴
    在末端），必须把机械臂位姿求逆（传 base2gripper 而非 gripper2base），返回值才是
    T_cam2base == T_base_camera。

    cv2.calibrateHandEye 输入（eye-to-hand 适配后）:
      - R_gripper2base, t_gripper2base: 传入 base→末端（已求逆）
      - R_target2cam,    t_target2cam:    标记→相机

    输出: R_cam2base, t_cam2base → T_base_camera
    """
    rmat_ee2base = []
    for rpy in rpy_ee2base_list:
        rmat = R.from_euler("xyz", rpy, degrees=False).as_matrix()
        rmat_ee2base.append(rmat)

    # eye-to-hand: gripper2base → base2gripper (求逆)
    rmat_base2ee = [rmat.T for rmat in rmat_ee2base]
    tvec_base2ee = [(-rmat.T @ tv).reshape(3, 1) for rmat, tv in zip(rmat_ee2base, tvec_ee2base_list)]

    rmat_marker2cam = [cv2.Rodrigues(rv)[0] for rv in rvec_marker2camera_list]
    tvec_marker2cam = [tv.reshape(3, 1) for tv in tvec_marker2camera_list]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        rmat_base2ee,
        tvec_base2ee,
        rmat_marker2cam,
        tvec_marker2cam,
        method=method,
    )

    T_base_camera = np.eye(4, dtype=np.float64)
    T_base_camera[:3, :3] = R_cam2base
    T_base_camera[:3, 3] = t_cam2base.flatten()
    return T_base_camera


def compute_marker_consistency(
    T_base_camera: np.ndarray,
    tvec_ee2base_list: list[np.ndarray],
    rpy_ee2base_list: list[np.ndarray],
    rvec_marker2camera_list: list[np.ndarray],
    tvec_marker2camera_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """计算标定质量指标：各样本 T_ee_marker 的位置/旋转一致性偏差，返回 (residuals_mm, residuals_deg)，均为 (N,)。

    注意：这不是像素重投影误差。对于 eye-to-hand:
        T_ee_marker = inv(T_base_ee) * T_base_camera * T_camera_marker
    由于 marker 刚性固定在末端，理论上该量在所有样本间应完全一致；标定越准，
    各样本的 T_ee_marker 越接近。这里取其位置部分与均值的偏差 (mm)，以及旋转部分
    与平均旋转的夹角 (deg)，作为质量指标。
    """
    positions = []
    rotations = []
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
        positions.append(T_ee_marker[:3, 3].copy())
        rotations.append(T_ee_marker[:3, :3].copy())

    # 位置：每个样本的 T_ee_marker 位置与均值的偏差 (mm)
    positions = np.array(positions)  # (N, 3)
    mean_pos = positions.mean(axis=0)
    residuals_mm = np.linalg.norm(positions - mean_pos, axis=1) * 1000.0

    # 旋转：每个样本的 T_ee_marker 旋转与平均旋转的夹角 (deg)
    rots = R.from_matrix(np.array(rotations))
    mean_rot = rots.mean()
    residuals_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return residuals_mm, residuals_deg


def calibrate_and_select(
    tvec_ee2base_list: list[np.ndarray],
    rpy_ee2base_list: list[np.ndarray],
    rvec_marker2camera_list: list[np.ndarray],
    tvec_marker2camera_list: list[np.ndarray],
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    """跑全部 5 种手眼算法，返回位置一致性 std 最小的那组结果。

    Returns:
        (T_best, name_best, errors_mm_best, errors_deg_best, table)
        table: [(method_name, std_mm), ...]，供打印对比（失败的算法 std 记为 nan）。
    """
    best: tuple[float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    table: list[tuple[float, float]] = []
    for name, m in HAND_EYE_METHODS.items():
        try:
            T = calibrate_eye_to_hand(
                tvec_ee2base_list, rpy_ee2base_list,
                rvec_marker2camera_list, tvec_marker2camera_list, method=m,
            )
            errors_mm, errors_deg = compute_marker_consistency(
                T, tvec_ee2base_list, rpy_ee2base_list,
                rvec_marker2camera_list, tvec_marker2camera_list,
            )
            std_mm = float(errors_mm.std())
        except cv2.error:
            table.append((name, float("nan")))
            continue
        table.append((name, std_mm))
        if best is None or std_mm < best[0]:
            best = (std_mm, name, T, errors_mm, errors_deg)

    if best is None:
        raise RuntimeError("所有手眼算法均失败")
    _, name_best, T_best, errors_mm_best, errors_deg_best = best
    return T_best, name_best, errors_mm_best, errors_deg_best, table


# ═══════════════════════════════════════════════ cameras.json 写入


def save_cameras_json(T_world_camera: np.ndarray, serial: str, json_path: Path) -> None:
    """将标定结果写入 cameras.json（pose 格式，与 CameraCalib 兼容）。

    - position: XYZ (m), camera 在 world 系中的位置
    - orientation: WXYZ quaternion, camera→world 的旋转
    """
    rot = R.from_matrix(T_world_camera[:3, :3])
    quat_wxyz = rot.as_quat()[[3, 0, 1, 2]]  # scipy xyzw → wxyz
    pos = T_world_camera[:3, 3]

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
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  标定结果已写入 → {json_path} (camera: {cam_name})")


# ═══════════════════════════════════════════════ 主程序


def do_return_home(
    robot: RobotInterface,
    arm_inner: ArmInnerLoop,
) -> ArmInnerLoop:
    """执行 return_home（停止内环线程 → 归位 → 重启内环线程）。返回新的内环实例。"""
    print("\nR: return_home ...", flush=True)
    try:
        # 停止内环，避免与 return_to_home 双重 XArmAPI 连接
        arm_inner.set_target(None)
        arm_inner.stop()
        print("  Arm 内环线程已停止")

        ok = robot.return_to_home(home_dt=HOME_DT)
        print(f"  {'OK' if ok else 'FAIL'}")

        new_inner = ArmInnerLoop()
        new_inner.start()
        print("  Arm 内环线程已重启")
        return new_inner
    except Exception:
        traceback.print_exc()
        print("  return_to_home 异常，尝试 emergency_stop")
        arm_inner.set_target(None)
        arm_inner.stop()
        robot.emergency_stop()
        raise


def main():
    print("=" * 60)
    print("  ArUco 手眼标定 — xArm7 + RealSense (eye-to-hand)")
    print(f"  ArUco: {ARUCO_DICT_NAME} ID={MARKER_ID} size={MARKER_SIZE_M*1000:.1f}mm")
    print(f"  移动: WASD/↑↓  旋转: ←→(roll) IK(pitch) JL(yaw)")
    print(f"  采集: SPACE  标定: ENTER(≥10,推荐10~20)  撤销: BACKSPACE  删最差帧: X  归位: R  退出: Q")
    print("=" * 60)

    # 保存终端 tty 设置，退出时恢复（pynput 监听退出后常残留 echo/规范模式异常）
    tty_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    tty_attrs = termios.tcgetattr(tty_fd) if tty_fd is not None else None

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
            use_position_ik=True,
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
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
    print("  ✓ Arm 内环就绪 (50Hz, mode 6)")

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

    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
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

    # 实时预览窗口（叠加 marker 检测框/坐标轴/状态）
    cv2.namedWindow("ArUco Calibration", cv2.WINDOW_AUTOSIZE)
    preview_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
    )
    display_img: np.ndarray | None = None
    print("  预览窗口已打开（绿=已检测，红=未检测）")

    # ── 6. 主循环 ──
    limiter = RateLimiter(CTRL_HZ)
    running = True
    loop_count = 0
    wall_warned = [False, False, False]
    last_wall_time = 0.0
    T_world_camera: np.ndarray | None = None
    # 最近一次 ENTER 计算出的逐帧残差 (mm)，与样本列表同序；样本变动即作废
    last_residuals: np.ndarray | None = None

    def _get_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        """获取当前末端位姿 (pos_m, rpy_rad)。

        必须用 is_radian=True: xArm Python SDK 默认返回角度(度)，仅此参数下返回弧度。
        本函数的 rpy 单位(弧度)必须与 calibrate_eye_to_hand 中 R.from_euler(..., degrees=False)
        保持一致——两处耦合，改动其一务必同步，否则会引入 ~57x 的角度错误。
        """
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

            # ── 实时预览（非阻塞取帧，相机 30fps < 50Hz 循环，无帧则沿用上一帧）──
            frames = pipeline.poll_for_frames()
            color_frame = frames.get_color_frame() if frames else None
            if color_frame:
                img = np.asanyarray(color_frame.get_data()).copy()
                display_img, _ = draw_overlay(
                    img, preview_detector, INTRINSICS, DISTORTION, len(samples_tvec_ee2base)
                )
            if display_img is not None:
                cv2.imshow("ArUco Calibration", display_img)
            cv2.waitKey(1)  # 泵 GUI 事件；键盘输入仍由 pynput 处理

            # ── 事件处理 ──
            event = keys.pop_event()
            while event is not None:
                if event == "space":
                    # 采集标定样本
                    print(f"\n  [{len(samples_tvec_ee2base)+1}] 采集 ArUco 位姿...", end=" ", flush=True)
                    try:
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
                                last_residuals = None  # 样本变动，作废旧残差
                                print(
                                    f"✓ (共 {len(samples_tvec_ee2base)} 组) "
                                    f"EE={np.round(pos_ee, 3)}m  "
                                    f"marker_dist={np.linalg.norm(tvec):.3f}m"
                                )
                    except Exception as e:
                        # 相机取帧异常（如 L515 卡死）——保住会话与已采集样本，不整场崩溃
                        print(f"❌ 采集异常，跳过本次: {e}")
                elif event == "backspace":
                    if samples_tvec_ee2base:
                        samples_tvec_ee2base.pop()
                        samples_rpy_ee2base.pop()
                        samples_rvec_marker2cam.pop()
                        samples_tvec_marker2cam.pop()
                        last_residuals = None  # 样本变动，作废旧残差
                        print(f"  ↺ 已撤销，剩余 {len(samples_tvec_ee2base)} 组")
                    else:
                        print("  (无样本可撤销)")
                elif event == "x":
                    # 定向删除残差最大的最差帧（需先按 ENTER 评估过质量）
                    if last_residuals is None or len(last_residuals) != len(samples_tvec_ee2base):
                        print("  (请先按 ENTER 计算/评估每帧质量，再按 X 删除最差帧)")
                    elif not samples_tvec_ee2base:
                        print("  (无样本)")
                    else:
                        worst = int(np.argmax(last_residuals))
                        r = float(last_residuals[worst])
                        for lst in (samples_tvec_ee2base, samples_rpy_ee2base,
                                    samples_rvec_marker2cam, samples_tvec_marker2cam):
                            lst.pop(worst)
                        last_residuals = None  # 样本变动，作废
                        print(f"  ✂ 删除最差帧 #{worst+1} (残差 {r:.1f}mm)，"
                              f"剩余 {len(samples_tvec_ee2base)} 组 — 按 ENTER 复算")
                elif event == "enter":
                    n = len(samples_tvec_ee2base)
                    if n < MIN_SAMPLES:
                        print(f"  ❌ 至少需要 {MIN_SAMPLES} 组样本，当前 {n} 组 — 请继续采集")
                    else:
                        print(f"\n  计算手眼标定 ({n} 组样本, 5 种算法比选)...")
                        T_candidate, method_best, errors_mm, errors_deg, method_table = calibrate_and_select(
                            samples_tvec_ee2base,
                            samples_rpy_ee2base,
                            samples_rvec_marker2cam,
                            samples_tvec_marker2cam,
                        )
                        # 手眼解出的是 xArm 基座系 T_base_camera（末端位姿取自 get_position）。
                        # 转到 world 系，与下游 eef_pos/arm_ee(compute_eef_pose_world) 一致：
                        #   T_world_camera = T_world_base @ T_base_camera
                        T_world_base = _pose_wxyz_to_matrix(
                            planner.kin.base_pose_world.p, planner.kin.base_pose_world.q
                        )
                        T_candidate = T_world_base @ T_candidate
                        std_mm = float(errors_mm.std())
                        std_deg = float(errors_deg.std())
                        last_residuals = errors_mm  # 逐帧残差，供 X 键定向删除
                        print(f"  各算法一致性 std (mm, 越小越好):")
                        for name, s in method_table:
                            mark = "  ← 选用" if name == method_best else ""
                            s_txt = "  失败" if np.isnan(s) else f"{s:7.1f}"
                            print(f"    {name:11s} {s_txt}{mark}")
                        print(f"  标定质量 (选用 {method_best}, T_ee_marker 一致性):")
                        print(f"    位置 mean={errors_mm.mean():.1f}mm  "
                              f"max={errors_mm.max():.1f}mm  "
                              f"std={std_mm:.1f}mm")
                        print(f"    旋转 mean={errors_deg.mean():.2f}°  "
                              f"max={errors_deg.max():.2f}°  "
                              f"std={std_deg:.2f}°")
                        worst = int(np.argmax(errors_mm))
                        print(f"  逐帧残差 (mm, 越大越可疑):")
                        for i, r in enumerate(errors_mm):
                            bar = "█" * min(30, int(r / max(errors_mm.max(), 1e-9) * 30))
                            flag = "  ← 最差, 按 X 删除" if i == worst else ""
                            print(f"    #{i+1:2d} {r:6.1f} {bar}{flag}")
                        print(f"  T_world_camera (world 系):")
                        print(f"    pos (m):     {np.round(T_candidate[:3, 3], 4)}")
                        quat_wxyz = R.from_matrix(T_candidate[:3, :3]).as_quat()[[3, 0, 1, 2]]
                        print(f"    quat (wxyz): {np.round(quat_wxyz, 4)}")

                        pos_bad = std_mm > MAX_CONSISTENCY_STD_MM
                        rot_bad = std_deg > MAX_CONSISTENCY_ROT_STD_DEG
                        if pos_bad or rot_bad:
                            reasons = []
                            if pos_bad:
                                reasons.append(f"位置 std={std_mm:.1f}mm > {MAX_CONSISTENCY_STD_MM:.1f}mm")
                            if rot_bad:
                                reasons.append(f"旋转 std={std_deg:.2f}° > {MAX_CONSISTENCY_ROT_STD_DEG:.1f}°")
                            print(f"  ❌ 一致性不达标（{'；'.join(reasons)}）— 拒绝写盘")
                            print(f"     请增大末端姿态变化(I/K/J/L、←→)后重采；")
                            print(f"     BACKSPACE 可逐个撤销可疑样本，再按 ENTER 重新计算。")
                        else:
                            T_world_camera = T_candidate
                            save_cameras_json(T_world_camera, serial, CAMERAS_JSON_PATH)
                            print(f"  ✓ 标定完成 (选用 {method_best}, 位置 std={std_mm:.1f}mm, 旋转 std={std_deg:.2f}°)")
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

            # ── R: 归位 ──
            if keys.is_pressed("r"):
                arm_inner = do_return_home(robot, arm_inner)
                if arm_inner.wait_ready(timeout=30.0):
                    arm_qpos, error_state, _ = arm_inner.get_state()
                    if not error_state and np.all(np.isfinite(arm_qpos)) and not np.all(arm_qpos == 0):
                        state = robot.get_state(arm_qpos=arm_qpos)
                        prev_qpos_cmd = state.arm_qpos.copy()
                        target_pos = state.eef_pos.copy()
                        target_quat = state.eef_quat_wxyz.copy()
                        print("  Arm 内环线程重启就绪")
                else:
                    print("  ⚠ Arm 内环重启超时")
                continue

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
                    f"{'← 按 SPACE 采集' if n < MIN_SAMPLES else '← 按 ENTER 标定'}",
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

            # ── Workspace soft-wall: directional — allow moving back into workspace ──
            new_pos = target_pos + dx
            for axis in range(3):
                lo, hi = WORKSPACE_BOUNDS[axis]
                cur = target_pos[axis]
                new = new_pos[axis]
                if dx[axis] == 0:
                    continue
                if lo <= new <= hi:
                    target_pos[axis] = new
                elif (cur < lo and dx[axis] > 0) or (cur > hi and dx[axis] < 0):
                    # Moving back toward workspace → allow, clamp to boundary
                    target_pos[axis] = float(np.clip(new, lo, hi))
                else:
                    # Moving further outside → reject
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
            if np.all(np.isfinite(state.hand_qpos)):
                planner.set_hand_qpos(state.hand_qpos)
            ik_result = planner.solve_teleop_ik(target_pose, state.arm_qpos, prev_qpos_cmd)
            if not ik_result.success or ik_result.qpos is None:
                target_pos = state.eef_pos.copy()
                target_quat = state.eef_quat_wxyz.copy()
                continue

            prev_qpos_cmd = ik_result.qpos.copy()
            arm_inner.set_target(ik_result.qpos)

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — 退出")
    finally:
        keys.stop()
        cv2.destroyAllWindows()
        pipeline.stop()

        # 恢复终端 tty：丢弃缓冲的按键并还原 echo/规范模式
        if tty_attrs is not None:
            try:
                termios.tcflush(tty_fd, termios.TCIFLUSH)
                termios.tcsetattr(tty_fd, termios.TCSADRAIN, tty_attrs)
            except Exception:
                pass

        if arm_inner.is_alive:
            arm_inner.set_target(None)
            arm_inner.stop()
        robot.disconnect()

        n = len(samples_tvec_ee2base)
        if n >= MIN_SAMPLES and T_world_camera is None:
            print(f"\n  已采集 {n} 组样本但未执行标定。")
            print("  按 ENTER 可在退出前完成标定，或重新运行脚本。")
        elif T_world_camera is not None:
            print(f"\n  标定已完成 ({n} 组样本)")
        elif n > 0:
            print(f"\n  已丢弃 {n} 组未使用样本。")
        print("  Done.")


if __name__ == "__main__":
    main()
