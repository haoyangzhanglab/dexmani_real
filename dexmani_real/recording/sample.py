"""Physical feedback and useful action values for control-step recording."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeState:
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    arm_tau: np.ndarray
    hand_qpos: np.ndarray
    hand_contact: np.ndarray
    hand_contact_valid: bool
    hand_tactile_force: np.ndarray
    hand_tactile_force_valid: bool
    hand_qpos_stale: bool
    arm_connected: bool
    hand_connected: bool
    timestamp: float
    hand_current: np.ndarray
    arm_last_cmd_seq: int = 0


@dataclass
class EpisodeAction:
    """The hand command and Cartesian intent; submitted arm target is explicit."""

    arm_qpos_cmd: np.ndarray
    hand_qpos_cmd: np.ndarray
    target_eef_pos: np.ndarray | None = None
    target_eef_rot6d: np.ndarray | None = None


def build_episode_state(
    arm_state: np.ndarray | None,
    hand_state: np.ndarray | None,
    *,
    timestamp_s: float | None = None,
) -> EpisodeState:
    """Copy one causal hand sample; invalid tactile is persisted as NaN.

    Aggregate and dense tactile, together with qpos/current, all come from the
    single selected hand-state frame so they share one source identity. Each
    representation's validity bit decides independently whether its payload is
    copied or NaN-filled, so a worker zero-fill is never mistaken for a real
    no-contact measurement.
    """
    arm = arm_state[0] if arm_state is not None else None
    hand = hand_state[0] if hand_state is not None else None

    hand_contact = (
        np.array(hand["tactile_aggregate"], copy=True)
        if hand is not None
        else np.full((5, 3), np.nan)
    )
    hand_contact_valid = bool(hand["tactile_aggregate_valid"]) if hand is not None else False
    hand_tactile_force = (
        np.array(hand["tactile_dense"], copy=True)
        if hand is not None
        else np.full((5, 120, 3), np.nan)
    )
    hand_tactile_force_valid = (
        bool(hand["tactile_dense_valid"]) if hand is not None else False
    )
    if not hand_contact_valid:
        hand_contact = np.full((5, 3), np.nan)
    if not hand_tactile_force_valid:
        hand_tactile_force = np.full((5, 120, 3), np.nan)

    return EpisodeState(
        arm_qpos=(
            np.array(arm["qpos"], copy=True) if arm is not None else np.full(7, np.nan)
        ),
        arm_qvel=(
            np.array(arm["qvel"], copy=True) if arm is not None else np.full(7, np.nan)
        ),
        arm_tau=(
            np.array(arm["tau"], copy=True) if arm is not None else np.full(7, np.nan)
        ),
        hand_qpos=(
            np.array(hand["qpos"], copy=True)
            if hand is not None
            else np.full(12, np.nan)
        ),
        hand_current=(
            np.array(hand["current"], copy=True)
            if hand is not None
            else np.full(12, np.nan)
        ),
        hand_contact=hand_contact,
        hand_contact_valid=hand_contact_valid,
        hand_tactile_force=hand_tactile_force,
        hand_tactile_force_valid=hand_tactile_force_valid,
        arm_connected=bool(arm["connected"]) if arm is not None else False,
        hand_connected=bool(hand["connected"]) if hand is not None else False,
        hand_qpos_stale=bool(hand["qpos_stale"]) if hand is not None else False,
        arm_last_cmd_seq=int(arm["last_cmd_seq"]) if arm is not None else 0,
        timestamp=time.perf_counter() if timestamp_s is None else float(timestamp_s),
    )
