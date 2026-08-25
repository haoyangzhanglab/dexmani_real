"""Fail-closed validation immediately before arm and hand hardware boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.utils.limits import validate_hand_command_bounds


@dataclass(frozen=True)
class WorkerCommandIssue:
    """One worker-boundary rejection with explicit fault disposition."""

    reason: str
    fault: bool


def _check_command_identity_and_timing(
    command: np.ndarray,
    *,
    expected_run_generation: int | None,
    now_monotonic_ns: int | None,
) -> WorkerCommandIssue | None:
    """Apply the shared generation and delivery-window contract."""
    if expected_run_generation is not None and int(command["run_generation"][0]) != int(
        expected_run_generation
    ):
        return WorkerCommandIssue("stale run generation", fault=False)
    if now_monotonic_ns is None:
        return None

    created_ns = int(command["created_monotonic_ns"][0])
    target_ns = int(command["target_monotonic_ns"][0])
    valid_until_ns = int(command["valid_until_monotonic_ns"][0])
    if (
        min(created_ns, target_ns, valid_until_ns) <= 0
        or created_ns > target_ns
        or target_ns > valid_until_ns
        or int(now_monotonic_ns) < created_ns
    ):
        return WorkerCommandIssue("invalid command timing", fault=True)
    if int(now_monotonic_ns) >= valid_until_ns:
        return WorkerCommandIssue("expired command", fault=False)
    return None


def check_worker_arm_command(
    command: np.ndarray,
    *,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
    joint_limit_lower_rad: np.ndarray | None = None,
    joint_limit_upper_rad: np.ndarray | None = None,
    previous_command_qpos_rad: np.ndarray | None = None,
    max_command_jump_rad: float | np.ndarray | None = None,
) -> WorkerCommandIssue | None:
    """Return why an arm command is unsafe at the worker, or ``None``.

    Arm and hand commands share ``run_generation`` / ``valid_until`` identity,
    so a STOP/FAULT generation bump rejects an in-flight endpoint before the
    SDK boundary. The optional jump guard compares consecutive accepted
    targets, not lagging feedback; it is a discontinuity fallback rather than
    a controller-rate limiter.
    """
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == COUPLED_COMMAND_DTYPE
        and int(command["action_id"][0]) > 0
        and bool(command["arm_present"][0])
        and np.all(np.isfinite(command["arm_qpos"][0]))
    )
    if not well_formed:
        return WorkerCommandIssue("malformed command", fault=True)
    timing_issue = _check_command_identity_and_timing(
        command,
        expected_run_generation=expected_run_generation,
        now_monotonic_ns=now_monotonic_ns,
    )
    if timing_issue is not None:
        return timing_issue
    target = np.asarray(command["arm_qpos"][0], dtype=np.float64)
    if (joint_limit_lower_rad is None) != (joint_limit_upper_rad is None):
        return WorkerCommandIssue("incomplete joint-limit configuration", fault=True)
    if joint_limit_lower_rad is not None:
        lower = np.asarray(joint_limit_lower_rad, dtype=np.float64)
        upper = np.asarray(joint_limit_upper_rad, dtype=np.float64)
        if (
            lower.shape != target.shape
            or upper.shape != target.shape
            or not np.all(np.isfinite(np.concatenate((lower, upper))))
            or np.any(lower > upper)
            or np.any(target < lower)
            or np.any(target > upper)
        ):
            return WorkerCommandIssue("joint limit violation", fault=True)
    if max_command_jump_rad is not None:
        if previous_command_qpos_rad is None:
            return WorkerCommandIssue("missing previous command target", fault=True)
        previous = np.asarray(previous_command_qpos_rad, dtype=np.float64)
        try:
            max_delta = np.broadcast_to(
                np.asarray(max_command_jump_rad, dtype=np.float64), target.shape
            )
        except ValueError:
            return WorkerCommandIssue("invalid command-jump configuration", fault=True)
        if (
            previous.shape != target.shape
            or not np.all(np.isfinite(previous))
            or not np.all(np.isfinite(max_delta))
            or np.any(max_delta <= 0.0)
        ):
            return WorkerCommandIssue("invalid command-jump configuration", fault=True)
        if np.any(np.abs(target - previous) > max_delta):
            return WorkerCommandIssue("command jump limit violation", fault=True)
    return None


def check_worker_hand_command(
    command: np.ndarray,
    *,
    operational_lower_rad: np.ndarray,
    operational_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> WorkerCommandIssue | None:
    """Return why a hand command is unsafe at the worker, or ``None``."""
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == COUPLED_COMMAND_DTYPE
        and int(command["action_id"][0]) > 0
        and bool(command["hand_present"][0])
        and np.all(np.isfinite(command["hand_qpos"][0]))
    )
    if not well_formed:
        return WorkerCommandIssue("malformed command", fault=True)
    timing_issue = _check_command_identity_and_timing(
        command,
        expected_run_generation=expected_run_generation,
        now_monotonic_ns=now_monotonic_ns,
    )
    if timing_issue is not None:
        return timing_issue
    try:
        validate_hand_command_bounds(
            command["hand_qpos"][0],
            operational_lower_rad,
            operational_upper_rad,
            mechanical_lower_rad,
            mechanical_upper_rad,
            hand_defaults.mechanical_qpos_min_rad,
            hand_defaults.mechanical_qpos_max_rad,
        )
    except (TypeError, ValueError) as exc:
        return WorkerCommandIssue(f"joint limit violation: {exc}", fault=True)
    return None
