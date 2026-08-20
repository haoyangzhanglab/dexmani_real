"""Fail-closed validation immediately before arm and hand hardware boundaries."""

from __future__ import annotations

import numpy as np

from dexmani_real.utils.schema import ARM_COMMAND_DTYPE, HAND_COMMAND_DTYPE


def _hand_command_is_current(
    command: np.ndarray,
    *,
    expected_run_generation: int | None,
    now_monotonic_ns: int | None,
) -> bool:
    if expected_run_generation is not None and int(command["run_generation"][0]) != int(
        expected_run_generation
    ):
        return False
    if now_monotonic_ns is not None:
        valid_until_ns = int(command["valid_until_monotonic_ns"][0])
        if valid_until_ns <= 0 or int(now_monotonic_ns) > valid_until_ns:
            return False
    return True


def worker_validate_arm(
    command: np.ndarray,
    *,
    armed_at_seq: int = 0,
    now_monotonic_ns: int | None = None,
    max_command_age_s: float = 0.3,
) -> bool:
    """Validate a latest-wins arm endpoint at the worker boundary."""
    if not (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == ARM_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    ):
        return False
    if int(command["action_id"][0]) <= int(armed_at_seq):
        return False
    if now_monotonic_ns is not None:
        created_ns = int(command["created_monotonic_ns"][0])
        if created_ns <= 0:
            return False
        age_s = (int(now_monotonic_ns) - created_ns) * 1e-9
        if age_s < 0.0 or age_s > max_command_age_s:
            return False
    return True


def worker_validate_hand(
    command: np.ndarray,
    *,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Validate hand shape, finiteness, generation, and expiry at the worker."""
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    )
    return bool(
        well_formed
        and _hand_command_is_current(
            command,
            expected_run_generation=expected_run_generation,
            now_monotonic_ns=now_monotonic_ns,
        )
    )
