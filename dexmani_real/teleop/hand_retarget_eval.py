"""Deterministic offline evaluation for XHand retargeting backends.

The module reads only recorded schema-v16 hand data.  It never creates shared
memory, starts a worker, or imports a hardware SDK.  Backend-native losses are
intentionally excluded from the primary score: TAG and DexPilot are compared
through the same human-joint, fingertip-geometry, timing, and bound metrics.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import spearmanr

from dexmani_real import ASSET_DIR
from dexmani_real.config.defaults import (
    DexPilotRetargetingParams,
    TAGRetargetingParams,
    hand,
)
from dexmani_real.planning.hand_kinematics import HandKinematics
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.teleop.hand_retarget import TAGHandRetargeter, XHandRetargeter
from dexmani_real.utils.schema import HAND_JOINT_SHAPE, XHAND_SDK_JOINT_NAMES

_HAND_URDF_PATH = str(ASSET_DIR / "robots" / "xhand" / "xhand_right.urdf")
_CHAINS = (
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20),
)
_TIP_INDICES = (4, 8, 12, 16, 20)
_TIP_PAIRS = tuple(
    (first, second) for first in range(5) for second in range(first + 1, 5)
)
# Human flexion mapped to the robot flexion joints.  Index/mid/ring/pinky:
# human MCP → j1, human PIP+DIP → j2.  Thumb: human CMC → thumb_bend,
# human MCP+IP → thumb_rota2 (the IP joint is fixed, so rota2 carries both).
# thumb_rota1 (CMC opposition rotation) has no landmark-flexion mapping and is
# scored only through fingertip geometry.
_FLEXION_JOINT_INDICES = (0, 2, 4, 5, 6, 7, 8, 9, 10, 11)
_FLEXION_JOINT_NAMES = tuple(
    XHAND_SDK_JOINT_NAMES[index] for index in _FLEXION_JOINT_INDICES
)

# Finger-flexion joints only (no thumb/abduction) — the subset that receives the
# explicit lower-stop margin when estimating home. Thumb opposition and index
# abduction keep their configured operational envelope there.
_HOME_MARGIN_JOINT_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)


@dataclass(frozen=True)
class EpisodeHandData:
    """Small in-memory hand-only view of one episode."""

    path: str
    control_hz: float
    landmarks: np.ndarray
    initial_qpos: np.ndarray
    recorded_raw_qpos: np.ndarray
    recorded_final_qpos: np.ndarray
    measured_qpos: np.ndarray


@dataclass(frozen=True)
class HandFeatures:
    """Backend-neutral features derived from operator landmarks."""

    chain_angles_rad: np.ndarray
    flexion_features_rad: np.ndarray
    fingertip_distances_m: np.ndarray
    max_angle_step_rad: np.ndarray


@dataclass(frozen=True)
class RetargetRun:
    """One sequential backend replay."""

    backend: str
    parameters: dict[str, float]
    qpos: np.ndarray
    call_time_ms: np.ndarray
    failure_count: int
    projected_state: np.ndarray | None = None


@dataclass(frozen=True)
class RetargetMetrics:
    """Comparable offline metrics for one replay."""

    backend: str
    parameters: dict[str, float]
    frame_count: int
    failure_count: int
    mean_best_lag_frames: float
    max_best_lag_frames: int
    mean_best_flexion_rho: float
    min_best_flexion_rho: float
    best_lag_frames_by_joint: dict[str, int]
    best_flexion_rho_by_joint: dict[str, float]
    flexion_bias_deg_by_joint: dict[str, float]
    mean_abs_flexion_bias_deg: float
    fingertip_distance_mean_rho: float
    max_step_p95_deg: float
    max_step_p99_deg: float
    stationary_step_p95_deg: float
    mechanical_bound_occupancy: float
    operational_clip_occupancy: float
    call_time_p95_ms: float
    call_time_p99_ms: float
    projected_bit_flips: int
    projected_transition_frames: int
    first_four_home_error_deg: float
    thumb_index_close_recall: float | None
    thumb_index_false_close_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HomeEstimate:
    """Robust home estimate and its startup-continuity diagnostics."""

    backend: str
    source_frame_count: int
    settle_iterations: int
    qpos_deg: tuple[float, ...]
    unconstrained_qpos_deg: tuple[float, ...]
    per_joint_source_spread_deg: tuple[float, ...]
    max_source_spread_deg: float
    max_settle_residual_deg: float
    first_four_home_error_deg: float
    first_four_output_span_deg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_episode_hand_data(path: str | Path) -> EpisodeHandData:
    """Load and validate the hand fields needed by the offline evaluator."""

    with EpisodeReader(path) as reader:
        h5f = reader.h5f
        landmarks = np.asarray(h5f["vr_landmarks"][:], dtype=np.float64)
        measured = np.asarray(h5f["hand_qpos"][:], dtype=np.float64)
        recorded_raw = np.asarray(h5f["action_hand_joint_raw"][:], dtype=np.float64)
        recorded_final = np.asarray(h5f["action_hand_joint"][:], dtype=np.float64)
        control_hz = float(reader.timing.rate_hz)

    frame_count = landmarks.shape[0]
    expected_qpos_shape = (frame_count, *HAND_JOINT_SHAPE)
    if landmarks.shape != (frame_count, 21, 3) or frame_count < 8:
        raise ValueError(
            f"episode hand landmarks must have shape (N>=8, 21, 3), got {landmarks.shape}"
        )
    for name, values in (
        ("hand_qpos", measured),
        ("action_hand_joint_raw", recorded_raw),
        ("action_hand_joint", recorded_final),
    ):
        if values.shape != expected_qpos_shape:
            raise ValueError(
                f"{name} must have shape {expected_qpos_shape}, got {values.shape}"
            )
    if not all(
        np.all(np.isfinite(values))
        for values in (landmarks, measured, recorded_raw, recorded_final)
    ):
        raise ValueError("episode hand landmarks, state, and actions must be finite")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError("episode control_hz must be finite and positive")

    return EpisodeHandData(
        path=str(Path(path)),
        control_hz=control_hz,
        landmarks=landmarks,
        initial_qpos=measured[0].copy(),
        recorded_raw_qpos=recorded_raw,
        recorded_final_qpos=recorded_final,
        measured_qpos=measured,
    )


def extract_hand_features(landmarks: np.ndarray) -> HandFeatures:
    """Extract rotation-invariant flexion and fingertip-distance features."""

    points = np.asarray(landmarks, dtype=np.float64)
    if (
        points.ndim != 3
        or points.shape[1:] != (21, 3)
        or not np.all(np.isfinite(points))
    ):
        raise ValueError("landmarks must be a finite (N, 21, 3) array")

    angles = np.empty((len(points), 5, 3), dtype=np.float64)
    for finger_index, chain in enumerate(_CHAINS):
        bones = [
            points[:, chain[index + 1]] - points[:, chain[index]] for index in range(4)
        ]
        for joint_index, (first, second) in enumerate(zip(bones, bones[1:])):
            denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
            if np.any(denominator <= 1e-12):
                raise ValueError("landmarks contain a degenerate hand bone")
            cosine = np.sum(first * second, axis=1) / denominator
            angles[:, finger_index, joint_index] = np.arccos(np.clip(cosine, -1.0, 1.0))

    flexion = np.column_stack(
        [
            angles[:, 0, 0],
            angles[:, 0, 1] + angles[:, 0, 2],
            *[
                feature
                for finger_index in range(1, 5)
                for feature in (
                    angles[:, finger_index, 0],
                    angles[:, finger_index, 1] + angles[:, finger_index, 2],
                )
            ],
        ]
    )
    tips = points[:, _TIP_INDICES]
    distances = np.column_stack(
        [
            np.linalg.norm(tips[:, second] - tips[:, first], axis=1)
            for first, second in _TIP_PAIRS
        ]
    )
    max_angle_step = np.max(np.abs(np.diff(angles, axis=0)), axis=(1, 2))
    return HandFeatures(
        chain_angles_rad=angles,
        flexion_features_rad=flexion,
        fingertip_distances_m=distances,
        max_angle_step_rad=max_angle_step,
    )


def run_tag(
    data: EpisodeHandData,
    config: TAGRetargetingParams,
) -> RetargetRun:
    """Replay TAG over every recorded landmark frame."""

    retargeter = TAGHandRetargeter(tag_config=config)
    return _run_backend(
        "tag",
        retargeter,
        data,
        {
            "smooth_weight": float(config.smooth_weight),
            "pinch_start_dist_m": float(config.pinch_start_dist_m),
            "pinch_full_dist_m": float(config.pinch_full_dist_m),
        },
    )


def run_dexpilot(
    data: EpisodeHandData,
    config: DexPilotRetargetingParams,
) -> RetargetRun:
    """Replay DexPilot over every recorded landmark frame."""

    retargeter = XHandRetargeter(dexpilot_config=config)
    return _run_backend(
        "dexpilot",
        retargeter,
        data,
        {
            "scaling_factor": float(config.scaling_factor),
            "low_pass_alpha": float(config.low_pass_alpha),
            "project_dist_m": float(config.project_dist_m),
            "escape_dist_m": float(config.escape_dist_m),
        },
    )


def apply_low_pass(run: RetargetRun, alpha: float) -> RetargetRun:
    """Apply dex-retargeting's exact post-solver first-order filter offline."""

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("low-pass alpha must be finite and in [0, 1]")
    if run.qpos.ndim != 2 or run.qpos.shape[0] == 0:
        raise ValueError("retarget qpos must be a non-empty 2D array")
    filtered = np.empty_like(run.qpos)
    filtered[0] = run.qpos[0]
    for index in range(1, len(filtered)):
        filtered[index] = filtered[index - 1] + alpha * (
            run.qpos[index] - filtered[index - 1]
        )
    parameters = dict(run.parameters)
    parameters["low_pass_alpha"] = float(alpha)
    return RetargetRun(
        backend=run.backend,
        parameters=parameters,
        qpos=filtered,
        call_time_ms=run.call_time_ms,
        failure_count=run.failure_count,
        projected_state=run.projected_state,
    )


def evaluate_run(
    data: EpisodeHandData,
    features: HandFeatures,
    run: RetargetRun,
    *,
    startup_skip_frames: int = 8,
    max_lag_frames: int = 8,
    stationary_threshold_deg: float = 0.5,
    stationary_grace_frames: int = 2,
    kinematics: HandKinematics | None = None,
) -> RetargetMetrics:
    """Score one replay with metrics shared by TAG and DexPilot."""

    solver_qpos = np.asarray(run.qpos, dtype=np.float64)
    frame_count = len(data.landmarks)
    if solver_qpos.shape != (frame_count, *HAND_JOINT_SHAPE) or not np.all(
        np.isfinite(solver_qpos)
    ):
        raise ValueError(f"retarget qpos must be finite shape ({frame_count}, 12)")
    if (
        isinstance(startup_skip_frames, bool)
        or not isinstance(startup_skip_frames, int)
        or isinstance(max_lag_frames, bool)
        or not isinstance(max_lag_frames, int)
        or max_lag_frames < 0
        or not 0 <= startup_skip_frames < frame_count - max_lag_frames
    ):
        raise ValueError("startup skip/max lag leave no evaluation frames")
    if (
        not np.isfinite(stationary_threshold_deg)
        or stationary_threshold_deg <= 0
        or isinstance(stationary_grace_frames, bool)
        or not isinstance(stationary_grace_frames, int)
        or stationary_grace_frames < 0
    ):
        raise ValueError(
            "stationary threshold must be finite and positive; grace must be non-negative"
        )

    expected_feature_shapes = (
        ("chain_angles_rad", features.chain_angles_rad, (frame_count, 5, 3)),
        ("flexion_features_rad", features.flexion_features_rad, (frame_count, 10)),
        ("fingertip_distances_m", features.fingertip_distances_m, (frame_count, 10)),
        ("max_angle_step_rad", features.max_angle_step_rad, (frame_count - 1,)),
    )
    for name, values, expected_shape in expected_feature_shapes:
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite shape {expected_shape}")
    if run.call_time_ms.shape != (frame_count,) or not np.all(
        np.isfinite(run.call_time_ms)
    ):
        raise ValueError(f"call_time_ms must be finite shape ({frame_count},)")
    if run.failure_count < 0 or run.failure_count > frame_count:
        raise ValueError("failure_count must be between zero and frame_count")
    if run.projected_state is not None and (
        run.projected_state.ndim != 2 or run.projected_state.shape[0] != frame_count
    ):
        raise ValueError("projected_state must have shape (frame_count, N)")

    mechanical_lower = np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
    mechanical_upper = np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64)
    operational_lower = np.asarray(hand.qpos_min_rad, dtype=np.float64)
    operational_upper = np.asarray(hand.qpos_max_rad, dtype=np.float64)
    # Teleop applies this deterministic command-box clip before publication.
    # Score hand geometry and motion on the endpoint the robot can actually
    # receive; keep solver-bound/clip occupancy as separate raw diagnostics.
    qpos = np.clip(solver_qpos, operational_lower, operational_upper)

    best_lags: list[int] = []
    best_rhos: list[float] = []
    flexion_bias_deg: dict[str, float] = {}
    for feature_index, joint_index in enumerate(_FLEXION_JOINT_INDICES):
        lag, rho = _best_nonnegative_lag(
            features.flexion_features_rad[:, feature_index],
            qpos[:, joint_index],
            start=startup_skip_frames,
            max_lag=max_lag_frames,
        )
        best_lags.append(lag)
        best_rhos.append(rho)
        reference = features.flexion_features_rad[:, feature_index]
        response = qpos[:, joint_index]
        stop = len(reference) - lag
        bias_rad = float(
            np.mean(response[startup_skip_frames + lag:] - reference[startup_skip_frames:stop])
        )
        flexion_bias_deg[_FLEXION_JOINT_NAMES[feature_index]] = float(np.rad2deg(bias_rad))
    mean_abs_flexion_bias_deg = float(
        np.mean(np.abs(np.asarray(list(flexion_bias_deg.values()), dtype=np.float64)))
    )

    kinematics = kinematics or HandKinematics(
        _HAND_URDF_PATH, list(hand.fingertip_link_names)
    )
    if not kinematics.is_ready():
        raise RuntimeError("XHand fingertip FK is unavailable")
    robot_tips = np.stack(
        [kinematics.compute_tip_positions_in_handbase(values) for values in qpos]
    )
    robot_distances = np.column_stack(
        [
            np.linalg.norm(robot_tips[:, second] - robot_tips[:, first], axis=1)
            for first, second in _TIP_PAIRS
        ]
    )
    pair_rhos = [
        _finite_spearman(
            features.fingertip_distances_m[startup_skip_frames:, index],
            robot_distances[startup_skip_frames:, index],
        )
        for index in range(len(_TIP_PAIRS))
    ]

    max_step_deg = np.rad2deg(np.max(np.abs(np.diff(qpos, axis=0)), axis=1))
    stable_step_mask = _stable_step_mask(
        features.max_angle_step_rad,
        threshold_rad=np.deg2rad(stationary_threshold_deg),
        grace_frames=stationary_grace_frames,
    )
    stable_step_mask[: max(0, startup_skip_frames - 1)] = False
    stationary_steps = max_step_deg[stable_step_mask]
    if stationary_steps.size == 0:
        raise ValueError("episode contains no stationary evaluation steps")

    evaluated_qpos = solver_qpos[startup_skip_frames:]
    at_mechanical_bound = (evaluated_qpos <= mechanical_lower + 1e-6) | (
        evaluated_qpos >= mechanical_upper - 1e-6
    )
    outside_operational = (evaluated_qpos < operational_lower - 1e-9) | (
        evaluated_qpos > operational_upper + 1e-9
    )

    projected_bit_flips = 0
    projected_transition_frames = 0
    if run.projected_state is not None and len(run.projected_state) > 1:
        transitions = run.projected_state[1:] != run.projected_state[:-1]
        projected_bit_flips = int(np.count_nonzero(transitions))
        projected_transition_frames = int(np.count_nonzero(np.any(transitions, axis=1)))

    human_thumb_index_m = features.fingertip_distances_m[:, 0]
    robot_thumb_index_m = robot_distances[:, 0]
    human_close = human_thumb_index_m < 0.020
    human_open = human_thumb_index_m > 0.040

    return RetargetMetrics(
        backend=run.backend,
        parameters=dict(run.parameters),
        frame_count=frame_count,
        failure_count=int(run.failure_count),
        mean_best_lag_frames=float(np.mean(best_lags)),
        max_best_lag_frames=int(np.max(best_lags)),
        mean_best_flexion_rho=float(np.mean(best_rhos)),
        min_best_flexion_rho=float(np.min(best_rhos)),
        best_lag_frames_by_joint=dict(zip(_FLEXION_JOINT_NAMES, best_lags)),
        best_flexion_rho_by_joint=dict(zip(_FLEXION_JOINT_NAMES, best_rhos)),
        flexion_bias_deg_by_joint=flexion_bias_deg,
        mean_abs_flexion_bias_deg=mean_abs_flexion_bias_deg,
        fingertip_distance_mean_rho=float(np.mean(pair_rhos)),
        max_step_p95_deg=float(
            np.quantile(max_step_deg[max(0, startup_skip_frames - 1) :], 0.95)
        ),
        max_step_p99_deg=float(
            np.quantile(max_step_deg[max(0, startup_skip_frames - 1) :], 0.99)
        ),
        stationary_step_p95_deg=float(np.quantile(stationary_steps, 0.95)),
        mechanical_bound_occupancy=float(np.mean(at_mechanical_bound)),
        operational_clip_occupancy=float(np.mean(outside_operational)),
        call_time_p95_ms=float(
            np.quantile(run.call_time_ms[startup_skip_frames:], 0.95)
        ),
        call_time_p99_ms=float(
            np.quantile(run.call_time_ms[startup_skip_frames:], 0.99)
        ),
        projected_bit_flips=projected_bit_flips,
        projected_transition_frames=projected_transition_frames,
        first_four_home_error_deg=float(
            np.rad2deg(
                np.max(np.abs(qpos[: min(4, frame_count)] - data.initial_qpos[None, :]))
            )
        ),
        thumb_index_close_recall=_conditional_rate(
            robot_thumb_index_m < 0.010, human_close
        ),
        thumb_index_false_close_rate=_conditional_rate(
            robot_thumb_index_m < 0.010, human_open
        ),
    )


def estimate_home_qpos(
    data: EpisodeHandData,
    backend: str,
    config: TAGRetargetingParams | DexPilotRetargetingParams,
    *,
    source_frame_count: int = 4,
    settle_iterations: int = 512,
    flexion_lower_margin_deg: float = 5.0,
) -> HomeEstimate:
    """Estimate a stable, bounded home from independently converged frames.

    Each source landmark frame is solved repeatedly after a reset, with the
    DexPilot internal LPFilter disabled.  Taking the per-joint median therefore
    estimates the static backend target instead of averaging a warm-start
    transient.  Only the eight finger-flexion joints receive the explicit
    lower-stop margin; thumb opposition and index abduction retain their
    configured operational envelope.
    """

    if backend not in {"tag", "dexpilot"}:
        raise ValueError("backend must be 'tag' or 'dexpilot'")
    if not 1 <= source_frame_count <= len(data.landmarks):
        raise ValueError("source_frame_count must select at least one available frame")
    if settle_iterations <= 1:
        raise ValueError("settle_iterations must be greater than one")
    if not np.isfinite(flexion_lower_margin_deg) or flexion_lower_margin_deg < 0:
        raise ValueError("flexion_lower_margin_deg must be finite and non-negative")
    if backend == "tag" and not isinstance(config, TAGRetargetingParams):
        raise TypeError("TAG home estimation requires TAGRetargetingParams")
    if backend == "dexpilot" and not isinstance(config, DexPilotRetargetingParams):
        raise TypeError("DexPilot home estimation requires DexPilotRetargetingParams")

    seed = np.deg2rad(np.asarray(hand.home_qpos_deg, dtype=np.float64))
    static_retargeter = _make_retargeter(backend, config, disable_low_pass_filter=True)
    converged = np.empty((source_frame_count, *HAND_JOINT_SHAPE), dtype=np.float64)
    residuals = np.empty(source_frame_count, dtype=np.float64)
    for frame_index, landmarks in enumerate(data.landmarks[:source_frame_count]):
        static_retargeter.reset(seed.copy())
        previous: np.ndarray | None = None
        candidate: np.ndarray | None = None
        for _ in range(settle_iterations):
            result = static_retargeter.retarget(landmarks)
            if result is None:
                raise RuntimeError(
                    f"{backend} failed while estimating home frame {frame_index}"
                )
            previous = candidate
            candidate = np.asarray(result, dtype=np.float64)
            if candidate.shape != HAND_JOINT_SHAPE or not np.all(
                np.isfinite(candidate)
            ):
                raise RuntimeError(
                    f"{backend} returned an invalid home candidate at frame {frame_index}"
                )
        if previous is None or candidate is None:
            raise RuntimeError(f"{backend} produced no home candidate")
        converged[frame_index] = candidate
        residuals[frame_index] = float(np.max(np.abs(candidate - previous)))

    unconstrained = np.median(converged, axis=0)
    safe_lower = np.asarray(hand.qpos_min_rad, dtype=np.float64).copy()
    mechanical_lower = np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
    flexion_margin_rad = np.deg2rad(flexion_lower_margin_deg)
    safe_lower[list(_HOME_MARGIN_JOINT_INDICES)] = np.maximum(
        safe_lower[list(_HOME_MARGIN_JOINT_INDICES)],
        mechanical_lower[list(_HOME_MARGIN_JOINT_INDICES)] + flexion_margin_rad,
    )
    safe_upper = np.asarray(hand.qpos_max_rad, dtype=np.float64)
    estimated = np.clip(unconstrained, safe_lower, safe_upper)

    startup_retargeter = _make_retargeter(
        backend, config, disable_low_pass_filter=False
    )
    startup_retargeter.reset(estimated.copy())
    startup_outputs = []
    for landmarks in data.landmarks[:source_frame_count]:
        result = startup_retargeter.retarget(landmarks)
        if result is None:
            raise RuntimeError(f"{backend} failed during home startup verification")
        candidate = np.asarray(result, dtype=np.float64)
        if candidate.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(candidate)):
            raise RuntimeError(f"{backend} returned an invalid startup candidate")
        startup_outputs.append(candidate)
    startup = np.stack(startup_outputs)
    source_spread_deg = np.rad2deg(np.ptp(converged, axis=0))
    return HomeEstimate(
        backend=backend,
        source_frame_count=source_frame_count,
        settle_iterations=settle_iterations,
        qpos_deg=tuple(float(value) for value in np.rad2deg(estimated)),
        unconstrained_qpos_deg=tuple(
            float(value) for value in np.rad2deg(unconstrained)
        ),
        per_joint_source_spread_deg=tuple(float(value) for value in source_spread_deg),
        max_source_spread_deg=float(np.max(source_spread_deg)),
        max_settle_residual_deg=float(np.rad2deg(np.max(residuals))),
        first_four_home_error_deg=float(
            np.rad2deg(np.max(np.abs(startup - estimated[None, :])))
        ),
        first_four_output_span_deg=float(np.rad2deg(np.max(np.ptp(startup, axis=0)))),
    )


def evaluate_default_backends(data: EpisodeHandData) -> list[RetargetMetrics]:
    """Evaluate the repository defaults for both supported backends."""

    from dexmani_real.config.defaults import dexpilot_retargeting, tag_retargeting

    features = extract_hand_features(data.landmarks)
    kinematics = HandKinematics(_HAND_URDF_PATH, list(hand.fingertip_link_names))
    return sorted(
        [
            evaluate_run(
                data, features, run_tag(data, tag_retargeting), kinematics=kinematics
            ),
            evaluate_run(
                data,
                features,
                run_dexpilot(data, dexpilot_retargeting),
                kinematics=kinematics,
            ),
        ],
        key=_selection_key,
    )


def search_retarget_configs(data: EpisodeHandData) -> list[RetargetMetrics]:
    """Run the bounded TAG/DexPilot search used for offline parameter tuning."""

    from dexmani_real.config.defaults import dexpilot_retargeting, tag_retargeting

    features = extract_hand_features(data.landmarks)
    kinematics = HandKinematics(_HAND_URDF_PATH, list(hand.fingertip_link_names))
    results: list[RetargetMetrics] = []

    for smooth_weight in (0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.020):
        config = replace(tag_retargeting, smooth_weight=smooth_weight)
        results.append(
            evaluate_run(data, features, run_tag(data, config), kinematics=kinematics)
        )

    threshold_pairs = ((0.020, 0.035), (0.025, 0.035), (0.025, 0.040), (0.030, 0.040))
    for scaling_factor in (1.00, 1.05, 1.10, 1.15):
        for project_dist_m, escape_dist_m in threshold_pairs:
            unfiltered_config = replace(
                dexpilot_retargeting,
                scaling_factor=scaling_factor,
                low_pass_alpha=1.0,
                project_dist_m=project_dist_m,
                escape_dist_m=escape_dist_m,
            )
            unfiltered = run_dexpilot(data, unfiltered_config)
            for low_pass_alpha in (0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
                results.append(
                    evaluate_run(
                        data,
                        features,
                        apply_low_pass(unfiltered, low_pass_alpha),
                        kinematics=kinematics,
                    )
                )
    return sorted(results, key=_selection_key)


def passes_default_gates(metrics: RetargetMetrics) -> bool:
    """Return whether a candidate meets the conservative offline gates."""

    return (
        metrics.failure_count == 0
        and metrics.mean_best_lag_frames <= 1.5
        and metrics.max_best_lag_frames <= 2
        and metrics.mean_best_flexion_rho >= 0.87
        and metrics.fingertip_distance_mean_rho >= 0.72
        and metrics.max_step_p95_deg <= 25.0
        and metrics.max_step_p99_deg <= 35.0
        and metrics.stationary_step_p95_deg <= 3.0
        and metrics.call_time_p95_ms <= 10.0
    )


def pareto_front(metrics: Iterable[RetargetMetrics]) -> list[RetargetMetrics]:
    """Return non-dominated candidates under the primary common metrics."""

    values = list(metrics)
    result: list[RetargetMetrics] = []
    for candidate in values:
        if not any(
            _dominates(other, candidate) for other in values if other is not candidate
        ):
            result.append(candidate)
    return sorted(result, key=_selection_key)


def _make_retargeter(
    backend: str,
    config: TAGRetargetingParams | DexPilotRetargetingParams,
    *,
    disable_low_pass_filter: bool,
) -> TAGHandRetargeter | XHandRetargeter:
    if backend == "tag":
        if not isinstance(config, TAGRetargetingParams):
            raise TypeError("TAG retargeter requires TAGRetargetingParams")
        return TAGHandRetargeter(tag_config=config)
    if backend != "dexpilot":
        raise ValueError("backend must be 'tag' or 'dexpilot'")
    if not isinstance(config, DexPilotRetargetingParams):
        raise TypeError("DexPilot retargeter requires DexPilotRetargetingParams")
    effective = (
        replace(config, low_pass_alpha=1.0) if disable_low_pass_filter else config
    )
    return XHandRetargeter(dexpilot_config=effective)


def _run_backend(
    backend: str,
    retargeter: Any,
    data: EpisodeHandData,
    parameters: dict[str, float],
) -> RetargetRun:
    retargeter.reset(data.initial_qpos.copy())
    outputs = np.empty((len(data.landmarks), *HAND_JOINT_SHAPE), dtype=np.float64)
    call_time_ms = np.empty(len(data.landmarks), dtype=np.float64)
    projected: list[np.ndarray] = []
    previous = data.initial_qpos.copy()
    failures = 0
    for index, landmarks in enumerate(data.landmarks):
        started = time.perf_counter()
        result = retargeter.retarget(landmarks)
        call_time_ms[index] = (time.perf_counter() - started) * 1000.0
        if result is None:
            failures += 1
            outputs[index] = previous
        else:
            candidate = np.asarray(result, dtype=np.float64)
            if candidate.shape != HAND_JOINT_SHAPE or not np.all(
                np.isfinite(candidate)
            ):
                failures += 1
                outputs[index] = previous
            else:
                outputs[index] = candidate
                previous = candidate.copy()
        optimizer = getattr(getattr(retargeter, "retargeter", None), "optimizer", None)
        state = getattr(optimizer, "projected", None)
        if state is not None:
            projected.append(np.asarray(state, dtype=bool).copy())

    projected_state = np.stack(projected) if len(projected) == len(outputs) else None
    return RetargetRun(
        backend=backend,
        parameters=parameters,
        qpos=outputs,
        call_time_ms=call_time_ms,
        failure_count=failures,
        projected_state=projected_state,
    )


def _best_nonnegative_lag(
    reference: np.ndarray,
    response: np.ndarray,
    *,
    start: int,
    max_lag: int,
) -> tuple[int, float]:
    correlations = []
    for lag in range(max_lag + 1):
        stop = len(reference) - lag
        correlations.append(
            _finite_spearman(reference[start:stop], response[start + lag :])
        )
    best_lag = int(np.argmax(correlations))
    return best_lag, float(correlations[best_lag])


def _finite_spearman(first: np.ndarray, second: np.ndarray) -> float:
    correlation = float(spearmanr(first, second).statistic)
    return correlation if np.isfinite(correlation) else 0.0


def _stable_step_mask(
    max_angle_step_rad: np.ndarray,
    *,
    threshold_rad: float,
    grace_frames: int,
) -> np.ndarray:
    original = np.asarray(max_angle_step_rad < threshold_rad, dtype=bool)
    stable = original.copy()
    for offset in range(1, grace_frames + 1):
        stable[offset:] &= original[:-offset]
        stable[:offset] = False
    return stable


def _conditional_rate(values: np.ndarray, condition: np.ndarray) -> float | None:
    selected = np.asarray(values, dtype=bool)[np.asarray(condition, dtype=bool)]
    return float(np.mean(selected)) if selected.size else None


def _selection_key(metrics: RetargetMetrics) -> tuple[float, ...]:
    gate_penalty = 0.0 if passes_default_gates(metrics) else 1.0
    return (
        gate_penalty,
        max(0.0, metrics.max_step_p95_deg - 25.0),
        max(0.0, metrics.stationary_step_p95_deg - 3.0),
        -(metrics.mean_best_flexion_rho + metrics.fingertip_distance_mean_rho),
        metrics.mean_abs_flexion_bias_deg,
        metrics.mean_best_lag_frames,
        metrics.operational_clip_occupancy,
        metrics.projected_transition_frames / metrics.frame_count,
        metrics.max_step_p95_deg,
    )


def _dominates(first: RetargetMetrics, second: RetargetMetrics) -> bool:
    first_values = np.array(
        [
            first.mean_best_lag_frames,
            -first.mean_best_flexion_rho,
            -first.fingertip_distance_mean_rho,
            first.max_step_p95_deg,
            first.stationary_step_p95_deg,
        ]
    )
    second_values = np.array(
        [
            second.mean_best_lag_frames,
            -second.mean_best_flexion_rho,
            -second.fingertip_distance_mean_rho,
            second.max_step_p95_deg,
            second.stationary_step_p95_deg,
        ]
    )
    return bool(
        np.all(first_values <= second_values) and np.any(first_values < second_values)
    )


__all__ = [
    "EpisodeHandData",
    "HandFeatures",
    "HomeEstimate",
    "RetargetMetrics",
    "RetargetRun",
    "apply_low_pass",
    "evaluate_default_backends",
    "evaluate_run",
    "estimate_home_qpos",
    "extract_hand_features",
    "load_episode_hand_data",
    "pareto_front",
    "passes_default_gates",
    "run_dexpilot",
    "run_tag",
    "search_retarget_configs",
]
