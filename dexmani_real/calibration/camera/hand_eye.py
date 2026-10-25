"""Eye-to-hand estimation and rigid marker-to-EEF consistency."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.geometry.transforms import validate_rigid_transform

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class CalibrationConfig:
    min_samples: int = 10
    min_relative_rotation_deg: float = 5.0
    min_axis_separation_deg: float = 15.0
    max_consistency_rms_mm: float = 5.0
    max_consistency_rot_rms_deg: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.min_samples, int) or self.min_samples < 3:
            raise ValueError("min_samples must be at least 3")
        positive_fields = (
            "min_relative_rotation_deg",
            "min_axis_separation_deg",
            "max_consistency_rms_mm",
            "max_consistency_rot_rms_deg",
        )
        for field_name in positive_fields:
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.min_relative_rotation_deg > 180 or self.min_axis_separation_deg > 90:
            raise ValueError("excitation thresholds exceed rotation/axis angle ranges")


def calibrate_eye_to_hand(
    eef_positions_base: tuple[np.ndarray, ...],
    eef_rpy_base: tuple[np.ndarray, ...],
    marker_rotvecs_camera: tuple[np.ndarray, ...],
    marker_positions_camera: tuple[np.ndarray, ...],
    *,
    method: int,
) -> np.ndarray:
    """Invert EEF poses for OpenCV eye-to-hand, returning T_base_from_camera."""
    R_base_from_eef = [
        Rotation.from_euler("xyz", rpy, degrees=False).as_matrix() for rpy in eef_rpy_base
    ]
    R_eef_from_base = [rmat.T for rmat in R_base_from_eef]
    base_positions_eef = [
        (-rotation.T @ position).reshape(3, 1)
        for rotation, position in zip(R_base_from_eef, eef_positions_base)
    ]
    R_camera_from_marker = [cv2.Rodrigues(rv)[0] for rv in marker_rotvecs_camera]
    marker_translations_camera = [tv.reshape(3, 1) for tv in marker_positions_camera]

    R_base_from_camera, camera_position_base = cv2.calibrateHandEye(
        R_eef_from_base,
        base_positions_eef,
        R_camera_from_marker,
        marker_translations_camera,
        method=method,
    )

    T_base_from_camera = np.eye(4, dtype=np.float64)
    T_base_from_camera[:3, :3] = R_base_from_camera
    T_base_from_camera[:3, 3] = camera_position_base.flatten()
    return T_base_from_camera


def compute_closed_loop_errors(
    T_base_from_camera: np.ndarray,
    eef_positions_base: tuple[np.ndarray, ...],
    eef_rpy_base: tuple[np.ndarray, ...],
    marker_rotvecs_camera: tuple[np.ndarray, ...],
    marker_positions_camera: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return marker-to-EEF consistency residuals in mm and degrees for a rigid mount."""
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []

    for eef_position_base, rpy_base, marker_rotvec_camera, marker_position_camera in zip(
        eef_positions_base,
        eef_rpy_base,
        marker_rotvecs_camera,
        marker_positions_camera,
    ):
        T_base_from_eef = np.eye(4)
        T_base_from_eef[:3, :3] = Rotation.from_euler("xyz", rpy_base, degrees=False).as_matrix()
        T_base_from_eef[:3, 3] = eef_position_base

        T_camera_from_marker = np.eye(4)
        T_camera_from_marker[:3, :3] = cv2.Rodrigues(marker_rotvec_camera)[0]
        T_camera_from_marker[:3, 3] = marker_position_camera

        T_eef_from_marker = (
            np.linalg.inv(T_base_from_eef) @ T_base_from_camera @ T_camera_from_marker
        )
        positions.append(T_eef_from_marker[:3, 3].copy())
        rotations.append(T_eef_from_marker[:3, :3])

    positions_arr = np.array(positions)
    mean_pos = positions_arr.mean(axis=0)
    residuals_mm = np.linalg.norm(positions_arr - mean_pos, axis=1) * 1000.0

    rots = Rotation.from_matrix(np.array(rotations))
    mean_rot = rots.mean()
    residuals_deg = np.degrees((mean_rot.inv() * rots).magnitude())
    return residuals_mm, residuals_deg


def check_hand_eye_excitation(
    eef_rpy_base, *, min_relative_rotation_deg, min_axis_separation_deg
):
    """Require two-axis excitation of admitted samples; do not impose three-axis rank."""
    rotations = Rotation.from_euler("xyz", eef_rpy_base).as_matrix()
    axes, angles = [], []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            vector = Rotation.from_matrix(rotations[j] @ rotations[i].T).as_rotvec()
            angle = np.linalg.norm(vector)
            if np.degrees(angle) >= min_relative_rotation_deg:
                axes.append(vector / angle)
                angles.append(float(np.degrees(angle)))
    # Any qualifying axis pair is sufficient; opposite axes describe the same direction.
    axes_array = np.asarray(axes)
    threshold_dot = np.cos(np.deg2rad(min_axis_separation_deg))
    # Bounded blocks preserve lexicographic witness order without a full pairwise matrix.
    for index, axis in enumerate(axes):
        for start in range(index + 1, len(axes), 1024):
            block = axes_array[start : start + 1024]
            dots = np.abs(block @ axis)
            separations = np.degrees(np.arccos(np.clip(dots, 0, 1)))
            # Recheck scalar threshold endpoints to preserve the first witness despite BLAS rounding.
            near = np.abs(dots - threshold_dot) <= 1e-14
            candidates = np.flatnonzero((separations >= min_axis_separation_deg) | near)
            witness = None
            for offset in candidates:
                separation = float(
                    np.degrees(np.arccos(np.clip(abs(axis @ block[offset]), 0, 1)))
                )
                if separation >= min_axis_separation_deg:
                    witness = separation
                    break
            if witness is None:
                continue
            return dict(
                sample_count=len(rotations),
                valid_relative_motion_count=len(axes),
                relative_rotation_min_deg=min(angles),
                relative_rotation_max_deg=max(angles),
                axis_separation_witness_deg=witness,
                min_relative_rotation_deg=min_relative_rotation_deg,
                min_axis_separation_deg=min_axis_separation_deg,
            )
    raise ValueError("insufficient hand-eye excitation; add rotation about another direction")


@dataclass(frozen=True)
class HandEyeResult:
    T_base_from_camera: np.ndarray
    method: str
    position_errors_mm: np.ndarray
    rotation_errors_deg: np.ndarray
    method_scores_mm: list[tuple[str, float]]
    position_rms_mm: float
    rotation_rms_deg: float
    excitation: dict[str, float | int]


def solve_hand_eye(samples: CalibrationSamples, config: CalibrationConfig) -> HandEyeResult:
    """Select the lowest-position-RMS candidate passing both limits."""
    if len(samples) < config.min_samples:
        raise ValueError(f"need at least {config.min_samples} samples, have {len(samples)}")
    (
        eef_positions_base,
        eef_rpy_base,
        marker_rotvecs_camera,
        marker_positions_camera,
    ) = zip(*samples.pairs)
    excitation = check_hand_eye_excitation(
        eef_rpy_base,
        min_relative_rotation_deg=config.min_relative_rotation_deg,
        min_axis_separation_deg=config.min_axis_separation_deg,
    )
    best = None
    table: list[tuple[str, float]] = []

    for name, method in HAND_EYE_METHODS.items():
        try:
            T_base_from_camera = calibrate_eye_to_hand(
                eef_positions_base,
                eef_rpy_base,
                marker_rotvecs_camera,
                marker_positions_camera,
                method=method,
            )
            # Check OpenCV's candidate once before composing it with the sample rotations.
            T_base_from_camera = validate_rigid_transform(
                T_base_from_camera, label="hand-eye candidate"
            )
            errors_mm, errors_deg = compute_closed_loop_errors(
                T_base_from_camera,
                eef_positions_base,
                eef_rpy_base,
                marker_rotvecs_camera,
                marker_positions_camera,
            )
            rms_mm = float(np.sqrt(np.mean(errors_mm**2)))
            rms_deg = float(np.sqrt(np.mean(errors_deg**2)))
            if not np.isfinite(rms_mm) or not np.isfinite(rms_deg):
                raise ValueError("nonfinite hand-eye quality score")
        except (cv2.error, ValueError):
            table.append((name, float("nan")))
            continue
        table.append((name, rms_mm))
        if (
            rms_mm > config.max_consistency_rms_mm
            or rms_deg > config.max_consistency_rot_rms_deg
        ):
            continue
        if best is None or (rms_mm, rms_deg) < best[:2]:
            best = (rms_mm, rms_deg, name, T_base_from_camera, errors_mm, errors_deg)

    if best is None:
        raise ValueError("no hand-eye candidate passed both quality limits")
    rms_mm, rms_deg, name, T_base_from_camera, errors_mm, errors_deg = best
    return HandEyeResult(
        T_base_from_camera, name, errors_mm, errors_deg, table, rms_mm, rms_deg, excitation
    )


@dataclass
class CalibrationSamples:
    """Each row owns paired EEF position/RPY and marker rotation/translation vectors."""

    pairs: list[np.ndarray] = field(default_factory=list)
    residuals_mm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pairs)

    def append(
        self, eef_position_base, eef_rpy_base, marker_rotvec_camera, marker_position_camera
    ) -> None:
        pair = np.array(
            (eef_position_base, eef_rpy_base, marker_rotvec_camera, marker_position_camera),
            dtype=np.float64,
        )
        if pair.shape != (4, 3) or not np.isfinite(pair).all():
            raise ValueError("calibration sample must contain four finite 3-vectors")
        self.pairs.append(pair)
        self.residuals_mm = None

    def pop_last(self) -> bool:
        if not self.pairs:
            return False
        self.pairs.pop()
        self.residuals_mm = None
        return True

    def pop_worst(self) -> tuple[int, float] | None:
        if self.residuals_mm is None:
            return None
        index = int(np.argmax(self.residuals_mm))
        residual = float(self.residuals_mm[index])
        self.pairs.pop(index)
        self.residuals_mm = None
        return index, residual
