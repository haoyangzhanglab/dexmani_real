#!/usr/bin/env python3
"""ArUco eye-to-hand camera calibration — xArm7 + RealSense L515.

Computes T_world_camera by detecting ArUco markers on the end-effector from a
fixed tripod-mounted camera and solving the hand-eye transform across five
OpenCV algorithms.

Results are written to ``dexmani_real/config/cameras.json``, compatible with
the ``CameraCalib`` config loader.

Hardware preparation:

  1. Print an ArUco 7x7_50 marker (ID=1), size 122.8 mm × 122.8 mm.
  2. Attach the marker flat on the end-effector, facing the camera.
  3. Fix the RealSense camera on a tripod, covering the workspace.
  4. Ensure conda environment has: pyrealsense2, opencv-python, scipy.

Usage::

    conda activate real_robot
    python examples/calibrate_camera.py [--serial SERIAL] [--config YAML]

Controls:

  WASD / arrows     translate EEF
  ← →              roll (about X)
  I / K            pitch (about Y)
  J / L            yaw (about Z)
  SPACE            capture calibration sample (requires ArUco detection)
  BACKSPACE         undo last sample
  X                 delete worst-residual frame (after ENTER evaluation)
  ENTER             compute calibration and write cameras.json (min 10 samples)
  R                 return home (collision-safe path)
  Q                 quit (discard data)
  ESC               emergency stop (FAULT)

XHand is optional: this arm-only procedure uses the configured open-hand
geometry for collision checks.  Pass ``--hand-geometry absent`` if the hand
is not physically mounted.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dexmani_real import PACKAGE_DIR
from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.planning import Pose, TeleopProfile, XArm7MotionPlanner
from dexmani_real.planning.pose_utils import quat_multiply, rot6d_to_quat_wxyz
from dexmani_real.policy.action_protocol import (
    ActionSafetyGateConfig,
    advance_policy_epoch,
    planner_action_safety_gate,
    publish_joint_targets,
)
from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.homing import send_arm_home
from dexmani_real.robot.safety import SafetyState, require_transition, transition
from dexmani_real.runtime.supervisor import shutdown_processes, wait_subsystem_ready
from dexmani_real.shm.shared_storage import SharedStorage, SharedStorageConfig, read_arm_state_dict
from dexmani_real.teleop.keyboard import GlobalKeyState, eef_delta_from_keys, validate_arm_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate_manager import RateManager

logger = get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

_CAMERAS_JSON_PATH = PACKAGE_DIR / "config" / "cameras.json"

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

_WINDOW_NAME = "ArUco Calibration"
_INITIAL_STATE_POLL_S = 0.05
_IK_WARNING_INTERVAL_S = 1.0

# Camera stream defaults.
_CAMERA_WIDTH = 640
_CAMERA_HEIGHT = 480
_CAMERA_FPS = 30
_CAMERA_WARMUP_FRAMES = 30

# ArUco dictionary — fixed per-marker-type, not configurable per session.
_ARUCO_DICT = cv2.aruco.DICT_7X7_50
_ARUCO_DICT_NAME = "7x7_50"

# OpenCV hand-eye methods.  TSAI is most sensitive to rotation noise;
# PARK is the most stable in the literature (PLOS ONE 2022).  All five are
# evaluated and the one with the lowest closed-loop position scatter wins.
_HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# ═══════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ArucoConfig:
    """ArUco marker detection parameters."""

    marker_size_m: float = 0.1228
    target_id: int | None = 1
    capture_frames: int = 5

    def __post_init__(self) -> None:
        if self.marker_size_m <= 0.0:
            raise ValueError("marker_size_m must be positive")
        if self.capture_frames < 1:
            raise ValueError("capture_frames must be at least 1")


@dataclass(frozen=True)
class CalibrationConfig:
    """Session tuning parameters for interactive camera calibration."""

    min_samples: int = 10
    max_consistency_std_mm: float = 5.0
    max_consistency_rot_std_deg: float = 3.0
    delta_pos_m: float = 0.008
    delta_rpy_rad: float = 0.03
    target_lead_max_m: float = 0.03
    status_interval_frames: int = 50

    def __post_init__(self) -> None:
        if self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        for name in ("delta_pos_m", "delta_rpy_rad", "max_consistency_std_mm"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


# ═══════════════════════════════════════════════════════════════════════
# ArUco detection
# ═══════════════════════════════════════════════════════════════════════


def _detect_aruco_pose(
    color_image: np.ndarray,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float,
    target_id: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect ArUco marker and return (rvec, tvec) in camera frame, or None."""
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    if target_id is not None:
        mask = ids.flatten() == target_id
        if not mask.any():
            return None
        corners = [c for c, m in zip(corners, mask) if m]

    half = marker_size_m / 2.0
    marker_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )

    for c in corners:
        _, rv, tv = cv2.solvePnP(marker_points, c, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return rv.flatten().astype(np.float64), tv.flatten().astype(np.float64)

    return None


def _detect_aruco_stable(
    pipeline: Any,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float,
    target_id: int | None,
    n_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Capture N frames and return median ArUco pose for noise reduction."""
    rvecs_all: list[np.ndarray] = []
    tvecs_all: list[np.ndarray] = []
    for _ in range(n_frames):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        result = _detect_aruco_pose(
            image, intrinsics, distortion, marker_size_m=marker_size_m, target_id=target_id,
        )
        if result is not None:
            rvecs_all.append(result[0])
            tvecs_all.append(result[1])

    if len(rvecs_all) < max(1, n_frames // 2):
        return None
    return np.median(rvecs_all, axis=0), np.median(tvecs_all, axis=0)


def _marker_corners_3d(marker_size_m: float) -> np.ndarray:
    """Marker corners in marker-local frame for drawFrameAxes."""
    s = marker_size_m / 2.0
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)


def _draw_overlay(
    image: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    intrinsics: np.ndarray,
    distortion: np.ndarray,
    *,
    n_samples: int,
    min_samples: int,
    target_id: int | None,
    marker_corners: np.ndarray,
    marker_size_m: float,
) -> tuple[np.ndarray, bool]:
    """Draw ArUco detection boxes, axes, and status text on the color image.

    Returns:
        (annotated_bgr, detected) — detected is True when the target marker is visible.
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
                marker_corners, c, intrinsics, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ok:
                cv2.drawFrameAxes(image, intrinsics, distortion, rv, tv, marker_size_m * 0.5)
                detected = True

    color = (0, 200, 0) if detected else (0, 0, 255)
    cv2.putText(image, f"MARKER {'OK' if detected else 'NOT FOUND'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    hint = (f"samples={n_samples}/{min_samples}  "
            + ("ENTER=calibrate" if n_samples >= min_samples else f"SPACE=capture (need >={min_samples})"))
    cv2.putText(image, hint, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image, detected


# ═══════════════════════════════════════════════════════════════════════
# Hand-eye calibration
# ═══════════════════════════════════════════════════════════════════════


def _calibrate_eye_to_hand(
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> np.ndarray:
    """Solve eye-to-hand transform, returning T_base_camera (4x4).

    cv2.calibrateHandEye expects eye-in-hand inputs (gripper2base +
    target2cam).  For eye-to-hand we invert the robot poses so the return
    value is T_cam2base == T_base_camera.
    """
    rmat_ee2base = [Rotation.from_euler("xyz", rpy, degrees=False).as_matrix() for rpy in rpy_ee2base]
    rmat_base2ee = [rmat.T for rmat in rmat_ee2base]
    tvec_base2ee = [(-rmat.T @ tv).reshape(3, 1) for rmat, tv in zip(rmat_ee2base, tvec_ee2base)]
    rmat_marker2cam = [cv2.Rodrigues(rv)[0] for rv in rvec_marker2camera]
    tvec_marker2cam = [tv.reshape(3, 1) for tv in tvec_marker2camera]

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        rmat_base2ee, tvec_base2ee, rmat_marker2cam, tvec_marker2cam, method=method,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cam2base
    T[:3, 3] = t_cam2base.flatten()
    return T


def _compute_closed_loop_errors(
    T_base_camera: np.ndarray,
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample T_ee_marker consistency residuals.

    T_ee_marker must be identical across all samples (the marker is rigidly
    attached to the end-effector).  Any scatter indicates calibration error.

    Returns:
        (residuals_mm, residuals_deg) as (N,) arrays.
    """
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for tvec_ee, rpy_ee, rvec_m2c, tvec_m2c in zip(
        tvec_ee2base, rpy_ee2base, rvec_marker2camera, tvec_marker2camera,
    ):
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = Rotation.from_euler("xyz", rpy_ee, degrees=False).as_matrix()
        T_base_ee[:3, 3] = tvec_ee

        T_camera_marker = np.eye(4)
        T_camera_marker[:3, :3] = cv2.Rodrigues(rvec_m2c)[0]
        T_camera_marker[:3, 3] = tvec_m2c

        T_ee_marker = np.linalg.inv(T_base_ee) @ T_base_camera @ T_camera_marker
        positions.append(T_ee_marker[:3, 3].copy())
        R_ee_marker = T_ee_marker[:3, :3]
        det_val = float(np.linalg.det(R_ee_marker))
        if not np.isfinite(det_val) or abs(det_val - 1.0) > 0.01:
            raise ValueError(f"degenerate T_ee_marker rotation (det={det_val:.3f})")
        rotations.append(R_ee_marker)

    positions_arr = np.array(positions)
    mean_pos = positions_arr.mean(axis=0)
    residuals_mm = np.linalg.norm(positions_arr - mean_pos, axis=1) * 1000.0

    rots = Rotation.from_matrix(np.array(rotations))
    mean_rot = rots.mean()
    residuals_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return residuals_mm, residuals_deg


def _calibrate_and_select(
    tvec_ee2base: list[np.ndarray],
    rpy_ee2base: list[np.ndarray],
    rvec_marker2camera: list[np.ndarray],
    tvec_marker2camera: list[np.ndarray],
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, list[tuple[str, float]]]:
    """Run all five hand-eye methods; return the best by position std.

    Returns:
        (T_best, method_name, errors_mm, errors_deg, method_table).
    """
    best: tuple[float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    table: list[tuple[str, float]] = []

    for name, m in _HAND_EYE_METHODS.items():
        try:
            T = _calibrate_eye_to_hand(
                tvec_ee2base, rpy_ee2base, rvec_marker2camera, tvec_marker2camera, method=m,
            )
            errors_mm, errors_deg = _compute_closed_loop_errors(
                T, tvec_ee2base, rpy_ee2base, rvec_marker2camera, tvec_marker2camera,
            )
            std_mm = float(errors_mm.std())
        except Exception as exc:
            print(f"    {name}: failed — {exc}")
            table.append((name, float("nan")))
            continue
        table.append((name, std_mm))
        if best is None or std_mm < best[0]:
            best = (std_mm, name, T, errors_mm, errors_deg)

    if best is None:
        raise RuntimeError("all hand-eye methods failed")
    _, name_best, T_best, errors_mm_best, errors_deg_best = best
    return T_best, name_best, errors_mm_best, errors_deg_best, table


# ═══════════════════════════════════════════════════════════════════════
# Camera helpers
# ═══════════════════════════════════════════════════════════════════════


def _start_camera(serial: str | None = None) -> tuple[Any, str, np.ndarray, np.ndarray]:
    """Start RealSense color stream and return (pipeline, serial, K, dist)."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    rs_config = rs.config()
    if serial:
        rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, _CAMERA_WIDTH, _CAMERA_HEIGHT, rs.format.bgr8, _CAMERA_FPS)
    profile = pipeline.start(rs_config)

    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    intrinsics = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    distortion = np.array(intr.coeffs, dtype=np.float64)

    # Warm-up: first few frames may have unstable auto-exposure.
    for _ in range(_CAMERA_WARMUP_FRAMES):
        pipeline.wait_for_frames()

    return pipeline, serial, intrinsics, distortion


# ═══════════════════════════════════════════════════════════════════════
# Sample store
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _Samples:
    """Paired calibration observations; delete preserves field alignment."""

    tvec_ee2base: list[np.ndarray] = field(default_factory=list)
    rpy_ee2base: list[np.ndarray] = field(default_factory=list)
    rvec_marker2cam: list[np.ndarray] = field(default_factory=list)
    tvec_marker2cam: list[np.ndarray] = field(default_factory=list)
    _residuals_mm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.tvec_ee2base)

    def append(self, pos_ee: np.ndarray, rpy_ee: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> None:
        self.tvec_ee2base.append(pos_ee.copy())
        self.rpy_ee2base.append(rpy_ee.copy())
        self.rvec_marker2cam.append(rvec.copy())
        self.tvec_marker2cam.append(tvec.copy())
        self._residuals_mm = None

    def pop_last(self) -> bool:
        if not self.tvec_ee2base:
            return False
        for lst in (self.tvec_ee2base, self.rpy_ee2base, self.rvec_marker2cam, self.tvec_marker2cam):
            lst.pop()
        self._residuals_mm = None
        return True

    def set_residuals(self, residuals_mm: np.ndarray) -> None:
        self._residuals_mm = residuals_mm.copy()

    def pop_worst(self) -> tuple[int, float] | None:
        if self._residuals_mm is None or len(self) == 0:
            return None
        idx = int(np.argmax(self._residuals_mm))
        val = float(self._residuals_mm[idx])
        for lst in (self.tvec_ee2base, self.rpy_ee2base, self.rvec_marker2cam, self.tvec_marker2cam):
            lst.pop(idx)
        self._residuals_mm = None
        return idx, val

    def solver_inputs(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (self.tvec_ee2base, self.rpy_ee2base, self.rvec_marker2cam, self.tvec_marker2cam)


# ═══════════════════════════════════════════════════════════════════════
# JSON output
# ═══════════════════════════════════════════════════════════════════════


def _save_cameras_json(T_world_camera: np.ndarray, serial: str, json_path: Path) -> None:
    """Write calibration result to cameras.json, preserving other entries."""
    rot = Rotation.from_matrix(T_world_camera[:3, :3])
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

    if json_path.exists():
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {}

    # Reuse existing key for the same serial; otherwise allocate a new one.
    cam_name = "camera_0"
    for name, cam_data in existing.items():
        if cam_data.get("serial") == serial:
            cam_name = name
            break
    else:
        idx = 0
        while f"camera_{idx}" in existing:
            idx += 1
        cam_name = f"camera_{idx}"

    if json_path.exists():
        backup = json_path.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        json_path.rename(backup)
        print(f"  backed up previous config → {backup.name}")

    existing[cam_name] = entry
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"  calibration written → {json_path} (camera: {cam_name})")


# ═══════════════════════════════════════════════════════════════════════
# Lifecycle helpers
# ═══════════════════════════════════════════════════════════════════════


def _workspace_bounds(runtime: ResolvedRuntimeConfig) -> np.ndarray:
    w = runtime.policy.workspace
    return np.array([[w.x_min, w.x_max], [w.y_min, w.y_max], [w.z_min, w.z_max]], dtype=np.float64)


def _build_planner_and_gate(
    runtime: ResolvedRuntimeConfig,
) -> tuple[XArm7MotionPlanner, Any, np.ndarray]:
    workspace = _workspace_bounds(runtime)
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(runtime.keyboard_teleop.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(runtime.keyboard_teleop.ik_max_pose_error_rot_rad),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    planner.workspace_bounds = workspace.copy()
    planner.set_hand_qpos(np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)))
    gate = planner_action_safety_gate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            arm_max_velocity_rad_s=float(np.deg2rad(runtime.arm.max_joint_velocity_deg_per_s)),
            hand_max_velocity_rad_s=float(np.deg2rad(runtime.hand.safety_gate_max_velocity_deg_per_s)),
            require_geometry_checks=True,
        ),
        planner=planner,
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        hand_safety_margin_m=float(runtime.arm.hand_safety_margin_m),
        enable_table_check=False,
    )
    return planner, gate, workspace


def _read_initial_arm(shared: SharedStorage, runtime: ResolvedRuntimeConfig) -> dict[str, Any] | None:
    deadline_s = time.monotonic() + float(runtime.safety.readiness_timeouts_s["arm"])
    while time.monotonic() < deadline_s:
        state = read_arm_state_dict(shared)
        if state is not None:
            issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(runtime.policy.arm_state_stale_threshold_s),
                qpos=state["qpos"],
                qvel=state["qvel"],
                eef_pos=state["eef_pos"],
                eef_rot6d=state["eef_rot6d"],
            )
            if issue is None and state["error_code"] == 0:
                return state
        time.sleep(_INITIAL_STATE_POLL_S)
    return None


def _set_fault(shared: SharedStorage, reason: str, *, estop: bool = False) -> None:
    logger.error("Calibration fault: %s", reason)
    if estop:
        shared.estop_request.value = True
    shared.error_state.value = True
    transition(shared, SafetyState.FAULT)


def _runtime_issue(shared: SharedStorage, arm_process: Any, heartbeat_timeout_s: float) -> str | None:
    if shared.estop_request.value:
        return "e-stop is requested"
    if shared.error_state.value:
        return "a worker set the sticky error latch"
    if int(shared.safety_state.value) == int(SafetyState.FAULT):
        return "safety state is FAULT"
    if not arm_process.is_alive():
        return "arm worker exited"
    heartbeat_s = float(shared.arm_heartbeat_s.value)
    now_s = time.monotonic()
    age_s = now_s - heartbeat_s
    if not np.isfinite(heartbeat_s) or heartbeat_s <= 0.0 or heartbeat_s > now_s or age_s > heartbeat_timeout_s:
        return f"arm heartbeat stale ({age_s:.2f}s)"
    return None


def _eef_rpy_from_state(arm_state: dict[str, Any]) -> np.ndarray:
    """Convert arm state rot6d to RPY (rad) via the canonical library path."""
    q_wxyz = rot6d_to_quat_wxyz(arm_state["eef_rot6d"])
    rpy = Rotation.from_quat(np.roll(q_wxyz, -1)).as_euler("xyz", degrees=False)
    return np.asarray(rpy, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# Main control loop
# ═══════════════════════════════════════════════════════════════════════


def _run_calibration(
    shared: SharedStorage,
    runtime: ResolvedRuntimeConfig,
    planner: XArm7MotionPlanner,
    safety_gate: Any,
    workspace: np.ndarray,
    arm_process: Any,
    camera_serial: str | None,
    calib_cfg: CalibrationConfig,
    aruco_cfg: ArucoConfig,
) -> int:
    """Interactive calibration control loop.  Returns 0 on success."""
    dt_s = 1.0 / float(runtime.keyboard_teleop.control_hz)
    heartbeat_timeout = float(runtime.safety.heartbeat_timeouts["arm"])
    recoverable_errors = frozenset(int(c) for c in runtime.arm.recoverable_errors)
    collision_errors = frozenset(int(c) for c in runtime.arm.collision_fault_errors)
    policy = runtime.policy
    idle_interval = int(runtime.keyboard_teleop.idle_interval_frames)

    state = _read_initial_arm(shared, runtime)
    if state is None:
        _set_fault(shared, "initial arm feedback is unavailable or unhealthy")
        return 1

    # ── Camera ──
    pipeline, serial, intrinsics, distortion = _start_camera(camera_serial)
    print(f"  Camera serial: {serial}")
    print(f"  Intrinsics: fx={intrinsics[0, 0]:.1f} fy={intrinsics[1, 1]:.1f} "
          f"({_CAMERA_WIDTH}x{_CAMERA_HEIGHT})")

    # ── Keyboard ──
    keys = GlobalKeyState(
        suppress_echo=True,
        estop_callback=lambda: _set_fault(shared, "operator e-stop callback", estop=True),
    )
    keys.start()

    # ── State ──
    samples = _Samples()
    T_world_camera: np.ndarray | None = None
    marker_corners = _marker_corners_3d(aruco_cfg.marker_size_m)
    preview_detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(_ARUCO_DICT), cv2.aruco.DetectorParameters(),
    )
    current_qpos = np.asarray(state["qpos"], dtype=np.float64)
    previous_command = current_qpos.copy()
    pose = planner.kin.compute_eef_pose_world(current_qpos)
    target_pos = pose.p.copy()
    target_quat = pose.q.copy()
    rate = RateManager(float(runtime.keyboard_teleop.control_hz))
    home_key_down = False
    motion_active = False
    frame = 0
    last_ik_warning_s = 0.0

    print(f"\n  ArUco: {_ARUCO_DICT_NAME} ID={aruco_cfg.target_id} "
          f"size={aruco_cfg.marker_size_m * 1000:.1f}mm")
    print("  Controls: WASD/arrows move, ←→/I/J/K/L rotate, SPACE capture, ENTER calibrate")
    print(f"  Preview window: {_WINDOW_NAME} (green=detected, red=not found)")

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    display_img: np.ndarray | None = None

    def _event_capture() -> None:
        print(f"\n  [{len(samples) + 1}] capturing ArUco pose...", end=" ", flush=True)
        try:
            ar_result = _detect_aruco_stable(
                pipeline, intrinsics, distortion,
                marker_size_m=aruco_cfg.marker_size_m, target_id=aruco_cfg.target_id,
                n_frames=aruco_cfg.capture_frames,
            )
        except Exception as exc:
            logger.warning("capture failed", exc_info=True)
            print(f"FAILED — {exc}, skipped")
            return
        if ar_result is None:
            print("FAILED — marker not detected, skipped")
            return
        rvec, tvec = ar_result
        arm_state = read_arm_state_dict(shared)
        if arm_state is None or not np.all(np.isfinite(arm_state["eef_pos"])):
            print("FAILED — arm state unavailable, skipped")
            return
        pos_ee = arm_state["eef_pos"].copy()
        rpy_ee = _eef_rpy_from_state(arm_state)
        samples.append(pos_ee, rpy_ee, rvec, tvec)
        print(f"OK (total {len(samples)})  EE={np.round(pos_ee, 3)}m  "
              f"marker_dist={np.linalg.norm(tvec):.3f}m")

    def _event_solve() -> None:
        nonlocal T_world_camera
        n = len(samples)
        if n < calib_cfg.min_samples:
            print(f"  need at least {calib_cfg.min_samples} samples, have {n} — keep collecting")
            return
        print(f"\n  computing hand-eye calibration ({n} samples, 5 methods)...")
        T_candidate, method_best, errors_mm, errors_deg, method_table = _calibrate_and_select(
            *samples.solver_inputs(),
        )
        # Convert T_base_camera → T_world_camera.
        T_world_base = np.eye(4, dtype=np.float64)
        T_world_base[:3, :3] = Rotation.from_quat(
            np.roll(np.asarray(planner.kin.base_pose_world.q, dtype=np.float64), -1),
        ).as_matrix()
        T_world_base[:3, 3] = np.asarray(planner.kin.base_pose_world.p, dtype=np.float64)
        T_candidate = T_world_base @ T_candidate
        std_mm = float(errors_mm.std())
        std_deg = float(errors_deg.std())
        samples.set_residuals(errors_mm)

        print("  method consistency (mm, lower is better):")
        for name, s in method_table:
            mark = "  ← selected" if name == method_best else ""
            s_txt = "  FAILED" if np.isnan(s) else f"{s:7.1f}"
            print(f"    {name:11s} {s_txt}{mark}")
        print(f"  quality ({method_best}, T_ee_marker consistency):")
        print(f"    position mean={errors_mm.mean():.1f}mm max={errors_mm.max():.1f}mm std={std_mm:.1f}mm")
        print(f"    rotation mean={errors_deg.mean():.2f}° max={errors_deg.max():.2f}° std={std_deg:.2f}°")
        worst = int(np.argmax(errors_mm))
        print("  per-frame residuals (mm, larger = more suspicious):")
        for i, r in enumerate(errors_mm):
            bar = "#" * min(30, int(r / max(errors_mm.max(), 1e-9) * 30))
            flag = "  ← worst, press X to remove" if i == worst else ""
            print(f"    #{i + 1:2d} {r:6.1f} {bar}{flag}")
        print(f"  T_world_camera position: {np.round(T_candidate[:3, 3], 4)}m")

        pos_ok = std_mm <= calib_cfg.max_consistency_std_mm
        rot_ok = std_deg <= calib_cfg.max_consistency_rot_std_deg
        if pos_ok and rot_ok:
            T_world_camera = T_candidate
            _save_cameras_json(T_world_camera, serial, _CAMERAS_JSON_PATH)
            print(f"  ACCEPTED ({method_best}, pos std={std_mm:.1f}mm, rot std={std_deg:.2f}°)")
        else:
            reasons = []
            if not pos_ok:
                reasons.append(f"pos std={std_mm:.1f}mm > {calib_cfg.max_consistency_std_mm:.1f}mm")
            if not rot_ok:
                reasons.append(f"rot std={std_deg:.2f}° > {calib_cfg.max_consistency_rot_std_deg:.1f}°")
            print(f"  REJECTED (quality gate: {'; '.join(reasons)}) — increase rotation variety and retry")

    try:
        while shared.is_running.value:
            rate.wait()
            frame += 1

            # ── Preview (non-blocking) ──
            frames = pipeline.poll_for_frames()
            color_frame = frames.get_color_frame() if frames else None
            if color_frame:
                img = np.asanyarray(color_frame.get_data()).copy()
                display_img, _ = _draw_overlay(
                    img, preview_detector, intrinsics, distortion,
                    n_samples=len(samples), min_samples=calib_cfg.min_samples,
                    target_id=aruco_cfg.target_id, marker_corners=marker_corners,
                    marker_size_m=aruco_cfg.marker_size_m,
                )
            if display_img is not None:
                cv2.imshow(_WINDOW_NAME, display_img)
            cv2.waitKey(1)

            # ── Events ──
            event = keys.pop_event()
            while event is not None:
                if event == "space":
                    _event_capture()
                elif event == "backspace":
                    if samples.pop_last():
                        print(f"  undone, {len(samples)} remaining")
                    else:
                        print("  (no samples to undo)")
                elif event == "x":
                    removed = samples.pop_worst()
                    if removed is None:
                        print("  (press ENTER first to evaluate quality, then X to remove worst)")
                    else:
                        idx, r = removed
                        print(f"  removed worst frame #{idx + 1} (residual {r:.1f}mm), "
                              f"{len(samples)} remaining — press ENTER to recompute")
                elif event == "enter":
                    _event_solve()
                event = keys.pop_event()

            # ── Exit / e-stop ──
            if keys.is_pressed("esc"):
                _set_fault(shared, "operator e-stop", estop=True)
                return 1
            if not keys.healthy:
                _set_fault(shared, "keyboard listener exited", estop=True)
                return 1

            issue = _runtime_issue(shared, arm_process, heartbeat_timeout)
            if issue is not None:
                _set_fault(shared, issue)
                return 1

            quit_requested = keys.is_pressed("q")
            state = read_arm_state_dict(shared)
            if state is None:
                if quit_requested:
                    _set_fault(shared, "cannot read arm state for quit hold")
                    return 1
                continue
            current_qpos = np.asarray(state["qpos"], dtype=np.float64)
            feedback_issue = validate_arm_feedback(
                connected=state["connected"],
                state_valid=state["state_valid"],
                source_monotonic_ns=state["source_monotonic_ns"],
                now_monotonic_ns=time.monotonic_ns(),
                max_age_s=float(policy.arm_state_stale_threshold_s),
                qpos=current_qpos,
                qvel=state["qvel"],
                eef_pos=state["eef_pos"],
                eef_rot6d=state["eef_rot6d"],
            )
            if feedback_issue is not None:
                if quit_requested:
                    _set_fault(shared, f"cannot publish measured quit hold: {feedback_issue}")
                    return 1
                continue

            error_code = int(state["error_code"])
            if error_code in recoverable_errors:
                if quit_requested:
                    _set_fault(shared, f"cannot quit during recoverable arm error C{error_code}")
                    return 1
                continue
            if error_code != 0:
                category = "collision" if error_code in collision_errors else "controller"
                _set_fault(shared, f"arm {category} error C{error_code}")
                return 1

            if quit_requested:
                advance_policy_epoch(shared)
                published = publish_joint_targets(
                    shared, current_qpos, is_hold=True,
                    prepare_timeout_s=float(policy.action_prepare_timeout_s),
                    dt_s=dt_s, safety_gate=safety_gate,
                    wait_applied=True, apply_timeout_s=float(policy.action_apply_timeout_s),
                )
                if published is None:
                    _set_fault(shared, "measured quit hold was not applied")
                    return 1
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    require_transition(shared, SafetyState.ARMED)
                return 0 if T_world_camera is not None else 2

            # ── Home (R, edge-triggered) ──
            home_pressed = keys.is_pressed("r")
            if home_pressed and not home_key_down:
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    require_transition(shared, SafetyState.ARMED)
                home_ok = send_arm_home(
                    shared,
                    np.asarray(runtime.arm.home_qpos, dtype=np.float64),
                    planner=planner,
                    table_z_surface_m=float(runtime.arm.table_z_surface_m),
                    current_qpos=current_qpos,
                    queue_timeout=float(runtime.arm.homing.request_queue_timeout_s),
                    converge_timeout_s=float(runtime.arm.homing.convergence_timeout_s),
                    state_max_age_s=float(runtime.arm.homing.state_max_age_s),
                    heartbeat=False,
                    estop_requested=lambda: keys.is_pressed("esc") or not keys.healthy,
                    homing_max_speed_rad_s=float(np.deg2rad(runtime.arm.homing.max_speed_deg_s)),
                    homing_target_timeout_s=float(runtime.arm.homing.target_timeout_s),
                    arm_heartbeat_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
                    preplan_velocity_rad_s=float(runtime.arm.homing.velocity_convergence_rad_s),
                    result_tolerance_rad=float(runtime.arm.homing.convergence_rad),
                    verbose=True,
                )
                if shared.estop_request.value:
                    _set_fault(shared, "operator e-stop during homing")
                    return 1
                refreshed = _read_initial_arm(shared, runtime)
                if refreshed is None:
                    _set_fault(shared, "fresh arm feedback unavailable after homing")
                    return 1
                current_qpos = np.asarray(refreshed["qpos"], dtype=np.float64)
                previous_command = current_qpos.copy()
                _fresh_pose = planner.kin.compute_eef_pose_world(current_qpos)
                target_pos, target_quat = _fresh_pose.p.copy(), _fresh_pose.q.copy()
                if not home_ok:
                    print("  WARNING: return-home request was not executed")
                motion_active = False
                rate.reset()
                home_key_down = home_pressed
                continue
            home_key_down = home_pressed

            # ── Keyboard deltas ──
            dx, drpy = eef_delta_from_keys(keys, calib_cfg.delta_pos_m, calib_cfg.delta_rpy_rad)
            moving = bool(np.any(dx != 0.0) or np.any(drpy != 0.0))
            if moving and not motion_active:
                require_transition(shared, SafetyState.RUNNING)
            elif not moving and motion_active:
                require_transition(shared, SafetyState.ARMED)
                held_pose = planner.kin.compute_eef_pose_world(previous_command)
                target_pos, target_quat = held_pose.p.copy(), held_pose.q.copy()
            motion_active = moving
            if not moving:
                if frame % idle_interval == 0:
                    measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
                    print(f"[f={frame}] samples={len(samples)} "
                          f"eef={np.round(measured_pose.p, 3)}m", flush=True)
                continue

            target_pos = np.clip(target_pos + dx, workspace[:, 0], workspace[:, 1])
            if np.any(drpy != 0.0):
                delta_quat = Rotation.from_euler("xyz", drpy).as_quat(scalar_first=True)
                target_quat = quat_multiply(delta_quat, target_quat)

            result = planner.solve_teleop_ik(Pose(p=target_pos, q=target_quat), current_qpos, previous_command)
            if not result.success or result.qpos is None:
                now_s = time.monotonic()
                if now_s - last_ik_warning_s >= _IK_WARNING_INTERVAL_S:
                    logger.warning("IK rejected target: %s", result.reason or "unknown")
                    last_ik_warning_s = now_s
                measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
                target_pos, target_quat = measured_pose.p.copy(), measured_pose.q.copy()
                continue

            published = publish_joint_targets(
                shared, result.qpos,
                prepare_timeout_s=float(policy.action_prepare_timeout_s),
                dt_s=dt_s, safety_gate=safety_gate,
                wait_applied=True, apply_timeout_s=float(policy.action_apply_timeout_s),
            )
            if published is None or published.arm_qpos is None:
                _set_fault(shared, "arm prepare/commit failed")
                return 1
            previous_command = np.asarray(published.arm_qpos, dtype=np.float64).copy()

            if frame % calib_cfg.status_interval_frames == 0:
                measured_pose = planner.kin.compute_eef_pose_world(current_qpos)
                print(f"[f={frame}] samples={len(samples)} "
                      f"eef={np.round(measured_pose.p, 3)}m "
                      f"target={np.round(target_pos, 3)}m", flush=True)

    except KeyboardInterrupt:
        _set_fault(shared, "KeyboardInterrupt")
        return 130
    finally:
        try:
            keys.stop()
        except Exception:
            logger.error("keyboard listener cleanup failed", exc_info=True)
        cv2.destroyAllWindows()
        pipeline.stop()
    return 0


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    calib_cfg = CalibrationConfig()
    aruco_cfg = ArucoConfig()

    parser = argparse.ArgumentParser(description="ArUco eye-to-hand camera calibration")
    parser.add_argument("--serial", default=None, help="RealSense serial (required with multiple devices)")
    parser.add_argument(
        "--hand-geometry", choices=("absent", "secured-home"), default="secured-home",
        help="physical assertion for arm-only procedure (default: secured-home)",
    )
    parser.add_argument("--config", type=Path, default=None, help="experiment YAML; --serial takes precedence")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  ArUco Hand-Eye Calibration — xArm7 + RealSense (eye-to-hand)")
    print(f"  ArUco: {_ARUCO_DICT_NAME} ID={aruco_cfg.target_id} "
          f"size={aruco_cfg.marker_size_m * 1000:.1f}mm")
    print(f"  hand geometry assertion: {args.hand_geometry}")
    print("=" * 60)

    # ── Runtime config ──
    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config, cli_overrides={"camera.serial": args.serial},
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        print(f"Invalid calibration config: {exc}", file=sys.stderr)
        return 2

    # ── Planner and safety gate ──
    planner, safety_gate, workspace = _build_planner_and_gate(runtime)
    print(f"  XHand: not required ({args.hand_geometry} geometry used for collision checks)")

    # ── SharedStorage + arm worker ──
    ctx = mp.get_context("spawn")
    shared = SharedStorage.create(
        prefix=f"dexmani_calib_{os.getpid()}",
        config=SharedStorageConfig.from_runtime(runtime),
        mp_context=ctx,
    )
    processes: list[Any] = []
    arm_process = ctx.Process(
        target=arm_loop, args=(shared, ArmLoopConfig.from_runtime(runtime)),
        name="arm-calib", daemon=False,
    )
    processes.append(arm_process)
    arm_process.start()
    arm_timeout_s = float(runtime.safety.readiness_timeouts_s["arm"])
    if not wait_subsystem_ready(shared, [("arm", shared.arm_ready, arm_timeout_s)], processes):
        _set_fault(shared, "arm worker did not become ready")
        shutdown_processes(shared, processes)
        return 1

    initial_state = _read_initial_arm(shared, runtime)
    if initial_state is None:
        _set_fault(shared, "initial arm feedback is unavailable or unhealthy")
        shutdown_processes(shared, processes)
        return 1

    require_transition(shared, SafetyState.ARMED)
    print(f"  arm worker ready (Mode 6, {runtime.arm.loop_hz}Hz)")

    # ── Run ──
    exit_code = _run_calibration(
        shared, runtime, planner, safety_gate, workspace,
        arm_process, args.serial, calib_cfg, aruco_cfg,
    )

    # ── Cleanup ──
    try:
        clean_exit = exit_code == 0
        shutdown_report = shutdown_processes(
            shared, [p for p in processes if p.pid is not None],
            graceful_timeout_s=float(runtime.safety.shutdown_timeout_s),
            disarm_if_clean=clean_exit,
        )
        if clean_exit and not shutdown_report.clean:
            logger.error("verified shutdown invalidated the clean control exit: %s", shutdown_report)
            exit_code = 1
    except RuntimeError:
        logger.critical("child process remains alive; leaving SharedStorage linked", exc_info=True)
        exit_code = 1

    print(f"  calibration session exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
