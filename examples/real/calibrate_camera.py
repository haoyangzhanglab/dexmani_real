#!/usr/bin/env python3
"""ARUCO 手眼标定 — xArm7 + RealSense 相机 (eye-to-hand only).

基于 keyboard_teleop 的机械臂移动逻辑，结合 ArUco 标记检测和
cv2.calibrateHandEye（5 种算法比选）自动求解相机外参。

手眼解在 xArm 基座系得到 T_base_camera，脚本再用 base_pose_world 转到 **world 系**
后写盘（与下游 eef_pos/arm_ee = compute_eef_pose_world 保持同一坐标系）。

结果写入 dexmani_real/config/cameras.json，与 CameraCalib 兼容。

硬件准备:
  1. 打印 ArUco 7x7_50 标记 (ID=1)，尺寸 122.8mm×122.8mm
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
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real import PACKAGE_DIR
from dexmani_real.planning import PlanningProfile, Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.path_utils import plan_joint_home_path
from dexmani_real.planning.pose_utils import quat_multiply, rot6d_to_quat_wxyz
from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.arm_loop import arm_loop as _arm_loop
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import HOME_SENTINEL, SharedStorage, SharedStorageConfig, read_arm_state
from dexmani_real.teleop.keyboard import GlobalKeyState
from dexmani_real.utils.rate_manager import RateManager

# ═══════════════════════════════════════════════ 配置


@dataclass
class CameraCalibConfig:
    """Camera calibration tuning parameters. Edit defaults here — no CLI needed."""

    ctrl_hz: float = 30.0  # control loop rate, matches arm_loop
    delta_pos: float = 0.008  # EEF translation per keypress (m)
    delta_rpy: float = 0.03  # EEF rotation per keypress (rad)
    target_lead_max: float = 0.03  # max target-to-arm lead distance (m)
    home_dt: float = 0.04  # homing waypoint interval (s)

    # ArUco
    marker_size_m: float = 0.1228  # marker side length (m) — 12.28cm
    marker_id: int = 1  # expected marker ID (None = accept any)
    aruco_capture_frames: int = 5  # frames for median-stable detection

    # Quality gates
    min_samples: int = 10  # minimum samples before calibration
    max_consistency_std_mm: float = 5.0  # T_ee_marker position std threshold
    max_consistency_rot_std_deg: float = 3.0  # T_ee_marker rotation std threshold


_cfg = CameraCalibConfig()

# Workspace bounds — derived from policy.workspace for consistency with
# the data-collection entry points.  The Y bounds are slightly tighter than
# the default config (-0.45 vs -0.50) because the calibration rig (ArUco
# marker on end-effector + fixed tripod camera) has a narrower useful range.
from dexmani_real.config.defaults import arm, policy

WORKSPACE_BOUNDS = policy.workspace.as_array()
WORKSPACE_BOUNDS[1, 0] = -0.45  # y_min: tighter for calibration rig
WORKSPACE_BOUNDS[1, 1] = 0.45  # y_max: tighter for calibration rig

# ArUco dictionary (fixed — not tunable per-session)
ARUCO_DICT = cv2.aruco.DICT_7X7_50
ARUCO_DICT_NAME = "7x7_50"

# 相机外参保存路径（包内 config/，与 CameraCalib 默认读取路径一致）
CAMERAS_JSON_PATH = PACKAGE_DIR / "config" / "cameras.json"


# ═══════════════════════════════════════════════ 键盘输入
# Uses GlobalKeyState from dexmani_real.teleop.keyboard for held-key detection
# (WASD / arrows).  One-shot events (space, enter, backspace) come from
# GlobalKeyState.pop_event(); the 'x' key uses edge detection.


# ═══════════════════════════════════════════════ ArUco 检测


def detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    marker_size: float = _cfg.marker_size_m,
    target_id: int | None = _cfg.marker_id,
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

    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )

    rvecs = []
    tvecs = []
    for c in corners:
        _, rv, tv = cv2.solvePnP(
            marker_points,
            c,
            intrinsics,
            distortion,
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
    n_frames: int = _cfg.aruco_capture_frames,
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
_MARKER_CORNERS_3D = np.array(
    [
        [-_cfg.marker_size_m / 2, _cfg.marker_size_m / 2, 0],
        [_cfg.marker_size_m / 2, _cfg.marker_size_m / 2, 0],
        [_cfg.marker_size_m / 2, -_cfg.marker_size_m / 2, 0],
        [-_cfg.marker_size_m / 2, -_cfg.marker_size_m / 2, 0],
    ],
    dtype=np.float32,
)


def draw_overlay(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    n_samples: int,
    target_id: int | None = _cfg.marker_id,
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
                _MARKER_CORNERS_3D,
                c,
                intrinsics,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                cv2.drawFrameAxes(image, intrinsics, distortion, rv, tv, _cfg.marker_size_m * 0.5)
                detected = True

    color = (0, 200, 0) if detected else (0, 0, 255)
    status = f"MARKER {'OK' if detected else 'NOT FOUND'}"
    cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    hint = f"samples={n_samples}/{_cfg.min_samples}  " + (
        "ENTER=calibrate" if n_samples >= _cfg.min_samples else f"SPACE=capture (need >={_cfg.min_samples})"
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
    tvec_ee2base_list: list[np.ndarray],  # (N, 3)  EE 位置 (m) base 系
    rpy_ee2base_list: list[np.ndarray],  # (N, 3)  EE RPY (rad) base 系
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
        tvec_ee2base_list,
        rpy_ee2base_list,
        rvec_marker2camera_list,
        tvec_marker2camera_list,
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
        R_ee_marker = T_ee_marker[:3, :3]
        # Reject degenerate rotation matrices (NaN / singular).
        # A degenerate T_base_camera from calibrate_eye_to_hand can produce
        # invalid SO(3) in T_ee_marker, which crashes R.from_matrix (SVD).
        _det = float(np.linalg.det(R_ee_marker))
        if not np.isfinite(_det) or abs(_det - 1.0) > 0.01:
            raise ValueError(f"degenerate rotation matrix in T_ee_marker (det={_det:.3f})")
        rotations.append(R_ee_marker.copy())

    # 位置：每个样本的 T_ee_marker 位置与均值的偏差 (mm)
    positions_arr = np.array(positions)  # (N, 3)
    mean_pos = positions_arr.mean(axis=0)
    residuals_mm = np.linalg.norm(positions_arr - mean_pos, axis=1) * 1000.0

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
    table: list[tuple[str, float]] = []
    for name, m in HAND_EYE_METHODS.items():
        try:
            T = calibrate_eye_to_hand(
                tvec_ee2base_list,
                rpy_ee2base_list,
                rvec_marker2camera_list,
                tvec_marker2camera_list,
                method=m,
            )
            errors_mm, errors_deg = compute_marker_consistency(
                T,
                tvec_ee2base_list,
                rpy_ee2base_list,
                rvec_marker2camera_list,
                tvec_marker2camera_list,
            )
            std_mm = float(errors_mm.std())
        except Exception:
            # cv2.error (calibrate_eye_to_hand) or LinAlgError / RuntimeWarning
            # (compute_marker_consistency: degenerate rotation → SVD failure)
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
            "orientation": [
                round(float(quat_wxyz[0]), 6),
                round(float(quat_wxyz[1]), 6),
                round(float(quat_wxyz[2]), 6),
                round(float(quat_wxyz[3]), 6),
            ],
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


def main():
    print("=" * 60)
    print("  ArUco 手眼标定 — xArm7 + RealSense (eye-to-hand)")
    print(f"  ArUco: {ARUCO_DICT_NAME} ID={_cfg.marker_id} size={_cfg.marker_size_m*1000:.1f}mm")
    print(f"  移动: WASD/↑↓  旋转: ←→(roll) IK(pitch) JL(yaw)")
    print(f"  采集: SPACE  标定: ENTER(≥10,推荐10~20)  撤销: BACKSPACE  删最差帧: X  归位: R  退出: Q")
    print("=" * 60)

    # 保存终端 tty 设置，退出时恢复（pynput 监听退出后常残留 echo/规范模式异常）
    tty_fd = sys.stdin.fileno() if sys.stdin.isatty() else None
    tty_attrs = termios.tcgetattr(tty_fd) if tty_fd is not None else None

    # ── 1. Planner (for IK in main loop) ──
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=0.02,
            max_pose_error_rot_rad=np.deg2rad(5.0),
        ),
    )

    # ── 2. SharedStorage + arm_loop (new architecture) ──
    import multiprocessing as mp

    shm_cfg = SharedStorageConfig()
    shared = SharedStorage.create(prefix="dexmani_calib", config=shm_cfg)
    arm_cfg = ArmLoopConfig()
    arm_proc = mp.Process(target=_arm_loop, args=(shared, arm_cfg), name="arm-calib", daemon=True)
    arm_proc.start()

    transition(shared, SafetyState.DISARMED)
    _arm_ready_ok = False
    _already_logged = False
    _arm_deadline = time.monotonic() + 30
    while time.monotonic() < _arm_deadline:
        if shared.arm_ready.is_set():
            _arm_ready_ok = True
            break
        if shared.error_state.value:
            print("❌ arm_loop init failed: error_state set")
            _already_logged = True
            break
        if not arm_proc.is_alive():
            print("❌ arm_loop init failed: process exited")
            _already_logged = True
            break
        time.sleep(0.2)
    if not _arm_ready_ok and not _already_logged:
        print("❌ arm_loop 启动超时")
    if not _arm_ready_ok:
        shared.is_running.value = False
        arm_proc.join(timeout=5)
        shared.close()
        return
    transition(shared, SafetyState.ARMED)
    print("  ✓ arm_loop 就绪 (SharedStorage, Mode 6, 30Hz)")

    # ── Read initial state from ring (retry: arm_loop sets arm_ready before first write) ──

    def _read_arm_state_ring():
        """Read latest arm state from ring, return (qpos, eef_pos, eef_rot6d) or (None,)*3."""
        data = read_arm_state(shared)
        if data is None:
            return None, None, None
        qpos = np.asarray(data["qpos"][0], dtype=np.float64)
        eef_pos = np.asarray(data["eef_pos"][0], dtype=np.float64)
        eef_rot6d = np.asarray(data["eef_rot6d"][0], dtype=np.float64)
        if not np.all(np.isfinite(qpos)):
            return None, None, None
        return qpos, eef_pos, eef_rot6d

    arm_qpos = eef_pos = _eef_rot6d = None
    for _ in range(30):  # up to ~1s
        arm_qpos, eef_pos, _eef_rot6d = _read_arm_state_ring()
        if arm_qpos is not None:
            break
        time.sleep(0.05)
    if arm_qpos is None:
        print("❌ 无法从 arm_state_ring 读取初始状态")
        shared.is_running.value = False
        arm_proc.join(timeout=5)
        shared.close()
        return

    prev_qpos_cmd = arm_qpos.copy()
    # Initialize target in WORLD frame (consistent with keyboard_teleop).
    # eef_pos/rot6d from the ring are base-frame (Pinocchio FK); convert to world.
    _eef_quat_base_init = rot6d_to_quat_wxyz(_eef_rot6d)
    _eef_world_init = planner.base_to_world_pose(Pose(p=eef_pos, q=_eef_quat_base_init))
    target_pos = _eef_world_init.p.copy()
    target_quat = _eef_world_init.q.copy()

    print(f"\n  当前 EEF (world): pos={np.round(target_pos, 4)}m  q={np.round(target_quat, 4)}")

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
    INTRINSICS = np.array(
        [
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ]
    )
    DISTORTION = np.array(intr.coeffs)
    print(f"  内参 fx={intr.fx:.1f} fy={intr.fy:.1f} ({intr.width}x{intr.height})")

    # 预热相机（前几帧可能曝光不稳定）
    for _ in range(30):
        pipeline.wait_for_frames()

    # ── 4. 键盘输入 ──
    keys = GlobalKeyState(suppress_echo=True)
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
    limiter = RateManager(_cfg.ctrl_hz)
    running = True
    loop_count = 0
    wall_warned = [False, False, False]
    last_wall_time = [0.0, 0.0, 0.0]  # per-axis debounce
    T_world_camera: np.ndarray | None = None
    # 最近一次 ENTER 计算出的逐帧残差 (mm)，与样本列表同序；样本变动即作废
    last_residuals: np.ndarray | None = None
    prev_x_pressed = False  # edge detection for 'x' one-shot key

    def _get_ee_pose() -> tuple[np.ndarray, np.ndarray]:
        """获取当前末端位姿 (pos_m, rpy_rad)。Rot6d→RPY via canonical library path."""
        qpos, eef_pos, eef_rot6d = _read_arm_state_ring()
        if qpos is None or eef_pos is None or eef_rot6d is None:
            return np.full(3, np.nan), np.full(3, np.nan)
        try:
            q_wxyz = rot6d_to_quat_wxyz(eef_rot6d)
            rpy = R.from_quat(np.roll(q_wxyz, -1)).as_euler("xyz", degrees=False)
            return eef_pos.copy(), np.asarray(rpy, dtype=np.float64)
        except Exception:
            return np.full(3, np.nan), np.full(3, np.nan)

    def _emergency_stop():
        nonlocal running
        shared.estop_request.value = True
        shared.is_running.value = False
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
                display_img, _ = draw_overlay(img, preview_detector, INTRINSICS, DISTORTION, len(samples_tvec_ee2base))
            if display_img is not None:
                cv2.imshow("ArUco Calibration", display_img)
            cv2.waitKey(1)  # 泵 GUI 事件；键盘输入仍由 pynput 处理

            # ── 事件处理 ──
            event = keys.pop_event()
            # Edge-detect 'x' for one-shot delete action (held key in GlobalKeyState)
            cur_x = keys.is_pressed("x")
            if event is None and cur_x and not prev_x_pressed:
                event = "x"
            prev_x_pressed = cur_x
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
                        for lst in (
                            samples_tvec_ee2base,
                            samples_rpy_ee2base,
                            samples_rvec_marker2cam,
                            samples_tvec_marker2cam,
                        ):
                            lst.pop(worst)
                        last_residuals = None  # 样本变动，作废
                        print(
                            f"  ✂ 删除最差帧 #{worst+1} (残差 {r:.1f}mm)，"
                            f"剩余 {len(samples_tvec_ee2base)} 组 — 按 ENTER 复算"
                        )
                elif event == "enter":
                    n = len(samples_tvec_ee2base)
                    if n < _cfg.min_samples:
                        print(f"  ❌ 至少需要 {_cfg.min_samples} 组样本，当前 {n} 组 — 请继续采集")
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
                        T_world_base = np.eye(4, dtype=np.float64)
                        T_world_base[:3, :3] = R.from_quat(np.roll(planner.kin.base_pose_world.q, -1)).as_matrix()
                        T_world_base[:3, 3] = planner.kin.base_pose_world.p
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
                        print(
                            f"    位置 mean={errors_mm.mean():.1f}mm  "
                            f"max={errors_mm.max():.1f}mm  "
                            f"std={std_mm:.1f}mm"
                        )
                        print(
                            f"    旋转 mean={errors_deg.mean():.2f}°  "
                            f"max={errors_deg.max():.2f}°  "
                            f"std={std_deg:.2f}°"
                        )
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

                        pos_bad = std_mm > _cfg.max_consistency_std_mm
                        rot_bad = std_deg > _cfg.max_consistency_rot_std_deg
                        if pos_bad or rot_bad:
                            reasons = []
                            if pos_bad:
                                reasons.append(f"位置 std={std_mm:.1f}mm > {_cfg.max_consistency_std_mm:.1f}mm")
                            if rot_bad:
                                reasons.append(f"旋转 std={std_deg:.2f}° > {_cfg.max_consistency_rot_std_deg:.1f}°")
                            print(f"  ❌ 一致性不达标（{'；'.join(reasons)}）— 拒绝写盘")
                            print(f"     请增大末端姿态变化(I/K/J/L、←→)后重采；")
                            print(f"     BACKSPACE 可逐个撤销可疑样本，再按 ENTER 重新计算。")
                        else:
                            T_world_camera = T_candidate
                            save_cameras_json(T_world_camera, serial, CAMERAS_JSON_PATH)
                            print(
                                f"  ✓ 标定完成 (选用 {method_best}, 位置 std={std_mm:.1f}mm, 旋转 std={std_deg:.2f}°)"
                            )
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

            # ── R: 归位 (collision-safe path via plan_joint_home_path) ──
            if keys.is_pressed("r"):
                print("\n  R: return_home")
                _home_qpos = np.array(ArmLoopConfig().home_qpos, dtype=np.float64)
                _waypoints = plan_joint_home_path(
                    arm_qpos, _home_qpos, planner, table_z_surface_m=arm.table_z_surface_m
                )
                if _waypoints is not None and len(_waypoints) > 0:
                    print(f"  planned homing: {len(_waypoints)} waypoints (路径安全无碰撞)")
                elif _waypoints is not None and len(_waypoints) == 0:
                    print(f"  planned homing: NO SAFE PATH — holding position")
                else:
                    print(f"  planned homing: already close to home")
                shared.arm_action_q.put((HOME_SENTINEL, _waypoints))
                # Wait for qpos to converge to home (arm stays there after homing).
                _home_arr = np.array(arm_cfg.home_qpos, dtype=np.float64)
                _home_deadline = time.perf_counter() + 20.0
                _home_ok = False
                while time.perf_counter() < _home_deadline:
                    _qpos_h, _, _ = _read_arm_state_ring()
                    if _qpos_h is not None:
                        if float(np.max(np.abs(_qpos_h - _home_arr))) < 0.03:
                            _home_ok = True
                            break
                    time.sleep(0.2)
                if not _home_ok:
                    print("  home wait timeout — continuing", flush=True)
                # Re-sync after homing
                arm_qpos, eef_pos, _eef_rot6d = _read_arm_state_ring()
                if arm_qpos is not None:
                    prev_qpos_cmd = arm_qpos.copy()
                    # World-frame target at home position
                    _eef_quat_base = rot6d_to_quat_wxyz(_eef_rot6d)
                    _eef_world = planner.base_to_world_pose(Pose(p=eef_pos, q=_eef_quat_base))
                    target_pos = _eef_world.p.copy()
                    target_quat = _eef_world.q.copy()
                    print("  Arm 归位完成，状态已同步")
                else:
                    print("  ⚠ 归位后无法读取状态")
                continue

            # ── 读取状态 (from arm_state_ring) ──
            try:
                arm_qpos, eef_pos, _eef_rot6d = _read_arm_state_ring()
                if arm_qpos is None:
                    continue
                # Derive eef_quat from rot6d, then convert to world frame.
                # eef_pos/rot6d from the ring are base-frame (Pinocchio FK);
                # keyboard deltas (WASD) are world-frame — both must be in
                # the same frame before arithmetic.
                try:
                    eef_quat_base = rot6d_to_quat_wxyz(_eef_rot6d)
                    _eef_world = planner.base_to_world_pose(Pose(p=eef_pos, q=eef_quat_base))
                    eef_pos_world = _eef_world.p
                    eef_quat_world = _eef_world.q
                except Exception:
                    eef_pos_world = target_pos.copy()
                    eef_quat_world = target_quat.copy()
            except Exception as e:
                print(f"  ⚠ 状态读取异常: {e}")
                continue

            # ── EEF target delta from keys (world frame) ──
            dx = np.zeros(3)
            if keys.is_pressed("w"):
                dx[0] += _cfg.delta_pos
            if keys.is_pressed("s"):
                dx[0] -= _cfg.delta_pos
            if keys.is_pressed("a"):
                dx[1] -= _cfg.delta_pos
            if keys.is_pressed("d"):
                dx[1] += _cfg.delta_pos
            if keys.is_pressed("up"):
                dx[2] += _cfg.delta_pos
            if keys.is_pressed("down"):
                dx[2] -= _cfg.delta_pos

            drpy = np.zeros(3)
            if keys.is_pressed("left"):
                drpy[0] += _cfg.delta_rpy
            if keys.is_pressed("right"):
                drpy[0] -= _cfg.delta_rpy
            if keys.is_pressed("i"):
                drpy[1] += _cfg.delta_rpy
            if keys.is_pressed("k"):
                drpy[1] -= _cfg.delta_rpy
            if keys.is_pressed("j"):
                drpy[2] -= _cfg.delta_rpy
            if keys.is_pressed("l"):
                drpy[2] += _cfg.delta_rpy

            # 周期性状态打印 (world frame)
            if loop_count % 50 == 0:
                n = len(samples_tvec_ee2base)
                print(
                    f"[{loop_count:5d}] "
                    f"eef_w={np.round(eef_pos_world, 3)}m  "
                    f"samples={n}  "
                    f"{'← 按 SPACE 采集' if n < _cfg.min_samples else '← 按 ENTER 标定'}",
                    flush=True,
                )

            # 无输入 → snap target to current world-frame EEF
            if np.all(dx == 0) and np.all(drpy == 0):
                target_pos = eef_pos_world.copy()
                target_quat = eef_quat_world.copy()
                prev_qpos_cmd = arm_qpos.copy()
                continue

            # ── Target lead limit (world frame) ──
            lead = np.linalg.norm(target_pos - eef_pos_world)
            if lead > _cfg.target_lead_max:
                target_pos = eef_pos_world + (target_pos - eef_pos_world) * (_cfg.target_lead_max / lead)

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
                    if not wall_warned[axis] or now - last_wall_time[axis] > 3.0:
                        names = ["x", "y", "z"]
                        print(f"  ⚠ {names[axis]} 边界 [{lo:.2f}, {hi:.2f}]")
                        wall_warned[axis] = True
                        last_wall_time[axis] = now

            if np.any(drpy != 0):
                dq = R.from_euler("xyz", drpy).as_quat(scalar_first=True)
                target_quat = quat_multiply(dq, target_quat)

            # ── IK (target in world frame, same as keyboard_teleop) ──
            target_pose = Pose(p=target_pos, q=target_quat)
            # Calibration tool doesn't connect hand hardware → no hand_qpos to sync.
            # (Collision model falls back to open-hand pose — acceptable for calib.)
            ik_result = planner.solve_teleop_ik(target_pose, arm_qpos, prev_qpos_cmd)
            if not ik_result.success or ik_result.qpos is None:
                target_pos = eef_pos_world.copy()
                target_quat = eef_quat_world.copy()
                continue

            prev_qpos_cmd = ik_result.qpos.copy()

            # ── Safety gates (same as keyboard_teleop / policy_loop) ──
            if not np.all(np.isfinite(ik_result.qpos)):
                continue
            if shared.safety_state.value == int(SafetyState.FAULT):
                continue

            shared.arm_action_q.put({"qpos": ik_result.qpos.copy()})

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

        # ── SharedStorage cleanup ──
        shared.is_running.value = False
        arm_proc.join(timeout=5)
        if arm_proc.is_alive():
            arm_proc.terminate()
            arm_proc.join(timeout=1)
        shared.close()

        n = len(samples_tvec_ee2base)
        if n >= _cfg.min_samples and T_world_camera is None:
            print(f"\n  已采集 {n} 组样本但未执行标定。")
            print("  按 ENTER 可在退出前完成标定，或重新运行脚本。")
        elif T_world_camera is not None:
            print(f"\n  标定已完成 ({n} 组样本)")
        elif n > 0:
            print(f"\n  已丢弃 {n} 组未使用样本。")
        print("  Done.")


if __name__ == "__main__":
    main()
