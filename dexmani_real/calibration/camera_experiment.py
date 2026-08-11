"""Interactive ArUco eye-to-hand calibration experiment."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from dexmani_real import PACKAGE_DIR
from dexmani_real.calibration.aruco import DEFAULT_ARUCO_CONFIG, ArucoConfig
from dexmani_real.calibration.camera_device import select_camera_serial
from dexmani_real.calibration.camera_session import (
    DEFAULT_CAMERA_SESSION_CONFIG,
    CameraCalibrationSession,
    CameraSessionConfig,
    CapturedSample,
)
from dexmani_real.calibration.hand_eye import (
    HandEyeQualityLimits,
    HandEyeSelection,
    select_hand_eye_calibration,
    update_camera_calibration,
    validate_transform,
)
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.planning import TeleopProfile, XArm7MotionPlanner
from dexmani_real.policy.action_protocol import ActionSafetyGateConfig, planner_action_safety_gate
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

CAMERAS_JSON_PATH = PACKAGE_DIR / "config" / "cameras.json"


def _finite_vector(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return result.copy()


@dataclass
class CalibrationSamples:
    """Keep each paired observation together so deletion cannot desynchronize fields."""

    values: list[CapturedSample] = field(default_factory=list)
    position_residuals_mm: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def previous(self) -> CapturedSample | None:
        return self.values[-1] if self.values else None

    def append(self, sample: CapturedSample) -> None:
        self.values.append(
            CapturedSample(
                marker_rvec_camera=_finite_vector(sample.marker_rvec_camera, "marker rvec"),
                marker_tvec_camera_m=_finite_vector(sample.marker_tvec_camera_m, "marker tvec"),
                eef_position_base_m=_finite_vector(sample.eef_position_base_m, "EEF position"),
                eef_rpy_base_rad=_finite_vector(sample.eef_rpy_base_rad, "EEF orientation"),
            )
        )
        self.position_residuals_mm = None

    def pop_last(self) -> CapturedSample | None:
        if not self.values:
            return None
        self.position_residuals_mm = None
        return self.values.pop()

    def set_position_residuals(self, residuals_mm: object) -> None:
        values = np.asarray(residuals_mm, dtype=np.float64)
        if values.shape != (self.count,) or not np.all(np.isfinite(values)):
            raise ValueError("position residuals must be finite and match the sample count")
        self.position_residuals_mm = values.copy()

    def pop_worst(self) -> tuple[int, float] | None:
        residuals = self.position_residuals_mm
        if residuals is None or residuals.shape != (self.count,) or not self.values:
            return None
        index = int(np.argmax(residuals))
        value = float(residuals[index])
        self.values.pop(index)
        self.position_residuals_mm = None
        return index, value

    def solver_inputs(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            [sample.eef_position_base_m for sample in self.values],
            [sample.eef_rpy_base_rad for sample in self.values],
            [sample.marker_rvec_camera for sample in self.values],
            [sample.marker_tvec_camera_m for sample in self.values],
        )


def transform_world_camera(base_pose_world: Any, transform_base_camera: object) -> np.ndarray:
    transform = validate_transform(np.asarray(transform_base_camera, dtype=np.float64), "T_base_camera")
    world_base = np.eye(4, dtype=np.float64)
    world_base[:3, :3] = Rotation.from_quat(np.roll(np.asarray(base_pose_world.q, dtype=np.float64), -1)).as_matrix()
    world_base[:3, 3] = np.asarray(base_pose_world.p, dtype=np.float64)
    result = world_base @ transform
    if not np.all(np.isfinite(result)):
        raise ValueError("T_world_camera is non-finite")
    return result


def _print_selection(selection: HandEyeSelection) -> None:
    best = selection.best
    print("  算法质量分数 (越小越好):")
    for candidate in selection.candidates:
        mark = " ← 选用" if candidate.method == best.method else ""
        accepted = "PASS" if candidate.quality.accepted else "REJECT"
        print(f"    {candidate.method:11s} score={candidate.score:6.2f} {accepted}{mark}")
    for method, reason in selection.failures:
        print(f"    {method:11s} failed: {reason}")

    quality = best.quality
    excitation = selection.excitation
    print(
        f"  激励: translation={excitation.translation_span_m:.3f}m, "
        f"rotation={excitation.rotation_span_deg:.1f}°, axis_ratio={excitation.rotation_axis_ratio:.2f}"
    )
    print(
        f"  闭环残差: position RMS/max={quality.position_rms_mm:.1f}/{quality.position_max_mm:.1f}mm, "
        f"rotation RMS/max={quality.rotation_rms_deg:.2f}/{quality.rotation_max_deg:.2f}°"
    )
    worst = int(np.argmax(best.position_errors_mm))
    print("  逐帧位置残差 (mm):")
    for index, residual_mm in enumerate(best.position_errors_mm):
        flag = " ← 最差, 按 X 删除" if index == worst else ""
        print(f"    #{index + 1:2d} {residual_mm:6.1f}{flag}")


def _result_metadata(
    session: CameraCalibrationSession,
    selection: HandEyeSelection,
    aruco_config: ArucoConfig,
    sample_count: int,
) -> dict[str, Any]:
    best = selection.best
    quality = best.quality
    excitation = selection.excitation
    metadata = session.capture_metadata
    metadata.update(
        {
            "marker_dictionary": aruco_config.dictionary_name,
            "marker_id": aruco_config.target_id,
            "marker_size_m": aruco_config.marker_size_m,
            "sample_count": sample_count,
            "method": best.method,
            "position_rms_mm": quality.position_rms_mm,
            "position_max_mm": quality.position_max_mm,
            "rotation_rms_deg": quality.rotation_rms_deg,
            "rotation_max_deg": quality.rotation_max_deg,
            "translation_span_m": excitation.translation_span_m,
            "rotation_span_deg": excitation.rotation_span_deg,
            "rotation_axis_ratio": excitation.rotation_axis_ratio,
            "frame_convention": "T_world_camera maps camera coordinates into world coordinates",
        }
    )
    return metadata


class CalibrationEvents:
    """Own the sample transaction, solver decision, and calibrated-file publication."""

    def __init__(
        self,
        *,
        config: CameraSessionConfig,
        aruco_config: ArucoConfig,
        output_path: Path = CAMERAS_JSON_PATH,
    ) -> None:
        self.config = config
        self.aruco_config = aruco_config
        self.output_path = output_path
        self.quality_limits = HandEyeQualityLimits(min_samples=config.min_samples)
        self.samples = CalibrationSamples()
        self.completed_transform: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return self.samples.count

    def _capture(self, session: CameraCalibrationSession) -> None:
        print(f"\n  [{self.sample_count + 1}] 采集 ArUco 位姿...", end=" ", flush=True)
        try:
            sample, issue = session.capture_sample(self.samples.previous)
        except Exception as exc:
            logger.warning("Calibration sample capture failed", exc_info=True)
            print(f"❌ 采集异常，跳过本次: {exc}")
            return
        if sample is None:
            print(f"❌ {issue or 'unknown capture error'} — 跳过")
            return
        self.samples.append(sample)
        print(
            f"✓ (共 {self.sample_count} 组) EE={np.round(sample.eef_position_base_m, 3)}m "
            f"marker_dist={np.linalg.norm(sample.marker_tvec_camera_m):.3f}m"
        )

    def _undo(self) -> None:
        if self.samples.pop_last() is None:
            print("  (无样本可撤销)")
        else:
            print(f"  ↺ 已撤销，剩余 {self.sample_count} 组")

    def _remove_worst(self) -> None:
        removed = self.samples.pop_worst()
        if removed is None:
            print("  (请先按 ENTER 计算/评估每帧质量，再按 X 删除最差帧)")
            return
        index, residual_mm = removed
        print(f"  ✂ 删除最差帧 #{index + 1} (残差 {residual_mm:.1f}mm)，剩余 {self.sample_count} 组 — 按 ENTER 复算")

    def _solve(self, session: CameraCalibrationSession) -> None:
        if self.sample_count < self.config.min_samples:
            print(f"  ❌ 至少需要 {self.config.min_samples} 组样本，当前 {self.sample_count} 组 — 请继续采集")
            return
        print(f"\n  计算手眼标定 ({self.sample_count} 组样本, 5 种算法比选)...")
        try:
            selection = select_hand_eye_calibration(*self.samples.solver_inputs(), limits=self.quality_limits)
        except (RuntimeError, ValueError) as exc:
            print(f"  ❌ 标定求解失败: {exc}")
            return

        best = selection.best
        self.samples.set_position_residuals(best.position_errors_mm)
        _print_selection(selection)
        candidate = transform_world_camera(session.planner.kin.base_pose_world, best.transform_base_camera)
        print(f"  T_world_camera position: {np.round(candidate[:3, 3], 4)}m")
        if not best.quality.accepted:
            print(f"  ❌ 质量门禁拒绝写盘: {'; '.join(best.quality.rejection_reasons)}")
            print("     增大平移与多轴旋转覆盖，或删除明确异常帧后重新计算。")
            return
        if not session.can_publish_calibration():
            print("  ❌ 系统状态已失效，拒绝写盘")
            return

        camera_name, backup = update_camera_calibration(
            self.output_path,
            serial=session.serial,
            transform_world_camera=candidate,
            capture_metadata=_result_metadata(session, selection, self.aruco_config, self.sample_count),
        )
        self.completed_transform = candidate
        if backup is not None:
            print(f"  旧配置备份: {backup.name}")
        print(f"  ✓ 已原子写入 {self.output_path} ({camera_name}, {best.method})")

    def handle(self, event: str, session: CameraCalibrationSession) -> None:
        if event == "space":
            self._capture(session)
        elif event == "backspace":
            self._undo()
        elif event == "x":
            self._remove_worst()
        elif event == "enter":
            self._solve(session)

    def print_summary(self) -> None:
        if self.sample_count >= self.config.min_samples and self.completed_transform is None:
            print(f"\n  已采集 {self.sample_count} 组样本但未执行或未通过标定。")
        elif self.completed_transform is not None:
            print(f"\n  标定已写入 ({self.sample_count} 组样本)")
        elif self.sample_count > 0:
            print(f"\n  已丢弃 {self.sample_count} 组未使用样本。")


def _workspace_bounds(runtime: Any, config: CameraSessionConfig) -> np.ndarray:
    workspace = runtime.policy.workspace
    bounds = np.array(
        [
            [workspace.x_min, workspace.x_max],
            [workspace.y_min, workspace.y_max],
            [workspace.z_min, workspace.z_max],
        ],
        dtype=np.float64,
    )
    bounds[1] = np.clip(bounds[1], -config.workspace_y_limit_m, config.workspace_y_limit_m)
    return bounds


def _build_planner(runtime: Any, config: CameraSessionConfig) -> tuple[Any, Any, np.ndarray]:
    control_hz = float(runtime.arm.loop_hz)
    workspace_bounds = _workspace_bounds(runtime, config)
    planner = XArm7MotionPlanner.create_default(
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=config.ik_position_tolerance_m,
            max_pose_error_rot_rad=np.deg2rad(config.ik_rotation_tolerance_deg),
        ),
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    planner.set_hand_qpos(np.deg2rad(np.asarray(runtime.hand.home_qpos_deg, dtype=np.float64)))
    planner.workspace_bounds = workspace_bounds.copy()
    safety_gate = planner_action_safety_gate(
        ActionSafetyGateConfig(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            arm_max_velocity_rad_s=float(np.deg2rad(runtime.arm.max_joint_velocity_deg_per_s)),
            hand_max_velocity_rad_s=(
                float(runtime.hand.max_delta_rad) * control_hz
                if runtime.hand.max_delta_rad is not None
                else float(np.deg2rad(runtime.hand.safety_gate_max_velocity_deg_per_s))
            ),
            require_geometry_checks=True,
        ),
        planner=planner,
        table_z_surface_m=float(runtime.arm.table_z_surface_m),
        hand_safety_margin_m=float(runtime.arm.hand_safety_margin_m),
        enable_table_check=False,
    )
    return planner, safety_gate, workspace_bounds


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArUco eye-to-hand camera calibration")
    parser.add_argument("--serial", default=None, help="RealSense serial (required with multiple devices)")
    parser.add_argument(
        "--hand-geometry",
        choices=("absent", "secured-home"),
        default="secured-home",
        help="physical assertion because this arm-only procedure does not read XHand feedback (default: secured-home)",
    )
    parser.add_argument("--config", type=Path, default=None, help="experiment YAML; --serial takes precedence")
    return parser.parse_args(argv)


def _print_banner(config: CameraSessionConfig, aruco_config: ArucoConfig) -> None:
    print("=" * 60)
    print("  ArUco 手眼标定 — xArm7 + RealSense (eye-to-hand)")
    print(
        f"  ArUco: {aruco_config.dictionary_name} ID={aruco_config.target_id} "
        f"size={aruco_config.marker_size_m * 1000:.1f}mm"
    )
    print("  移动: WASD/↑↓  旋转: ←→(roll) I/K(pitch) J/L(yaw)")
    print(
        f"  采集: SPACE  标定: ENTER(≥{config.min_samples},推荐10~20)  "
        "撤销: BACKSPACE  删最差帧: X  归位: R  退出: Q"
    )
    print("=" * 60)


def _run(runtime: Any, args: argparse.Namespace) -> int:
    config = DEFAULT_CAMERA_SESSION_CONFIG
    aruco_config = DEFAULT_ARUCO_CONFIG
    _print_banner(config, aruco_config)
    selected_serial = select_camera_serial(runtime.camera.serial)
    planner, safety_gate, workspace_bounds = _build_planner(runtime, config)
    print(f"  XHand geometry assertion: {args.hand_geometry}")
    session = CameraCalibrationSession(
        runtime,
        planner,
        safety_gate,
        workspace_bounds,
        selected_serial,
        hand_geometry=args.hand_geometry,
        config=config,
        aruco_config=aruco_config,
    )
    events = CalibrationEvents(config=config, aruco_config=aruco_config)
    exit_code = session.run(events)
    events.print_summary()
    print(f"  会话退出: {session.exit_reason} (code={exit_code})")
    if exit_code != 0:
        return exit_code
    if events.completed_transform is None:
        print("  标定未完成：没有通过质量门禁并写入新的相机标定。")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        runtime = resolve_runtime_config(
            yaml_path=args.config,
            cli_overrides={"camera.serial": args.serial},
        )
    except (OSError, TypeError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        print(f"Invalid camera calibration config: {exc}", file=sys.stderr)
        return 2
    try:
        return _run(runtime, args)
    except (OSError, RuntimeError, ValueError):
        logger.error("Camera calibration aborted; no new calibration was published", exc_info=True)
        return 1


__all__ = [
    "CalibrationEvents",
    "CalibrationSamples",
    "main",
    "transform_world_camera",
]
