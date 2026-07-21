"""Process-isolation transition flag resolver (plan §6 P1 过渡开关).

Single source of truth for whether the arm and/or hand servo runs in
crash-isolated subprocesses (SHM façades) instead of the proven in-process
ArmInnerLoop/XHand. The flag is SPLIT into arm/hand halves so hardware
bring-up can validate the arm subprocess first (lower risk: Mode-6 firmware
holds position) before enabling the hand subprocess (batch-G hold-position /
detorque behaviour, untested post-repair).

Default OFF — nothing changes until explicitly enabled per-config or via env,
so either half can be flipped back instantly by unsetting its env var.
Transitional (plan A3/D6): hard-deleted once P3 acceptance lands and the
in-process path is removed.

Resolution precedence (per half):
  1. Master env ``DEXMANI_PROCESS_ISOLATION=1`` (or true/yes/on) → BOTH on.
  2. Section env (``DEXMANI_ARM_PROCESS_ISOLATION`` / ``DEXMANI_HAND_PROCESS_ISOLATION``)
     = truthy → that half on; a falsy/unset section env defers to the config field.
  3. Config field (``use_arm_process_isolation`` / ``use_hand_process_isolation``).

So ``DEXMANI_PROCESS_ISOLATION=1`` is the all-on fast path; per-section envs
enable arm-first (``DEXMANI_ARM_PROCESS_ISOLATION=1``) or hand-only bring-up.
"""

from __future__ import annotations

import os

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_MASTER_ENV = "DEXMANI_PROCESS_ISOLATION"
_ARM_ENV = "DEXMANI_ARM_PROCESS_ISOLATION"
_HAND_ENV = "DEXMANI_HAND_PROCESS_ISOLATION"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", ""}


def _env_force_on(env_name: str) -> bool:
    """True if env var is set to a truthy value (forces the half on)."""
    return os.environ.get(env_name, "").strip().lower() in _TRUTHY


def _section_enabled(env_name: str, config_flag: bool) -> bool:
    """Resolve one half: truthy section env → on; falsy/unset → config field."""
    env = os.environ.get(env_name, "").strip().lower()
    if env in _TRUTHY:
        return True
    if env and env not in _FALSY:
        logger.warning("Unrecognized %s=%r — ignoring env, using config flag", env_name, env)
    return bool(config_flag)


def arm_isolation_enabled(config_flag: bool) -> bool:
    """Resolve the ARM isolation flag (master env OR arm env OR config field)."""
    if _env_force_on(_MASTER_ENV):
        return True
    return _section_enabled(_ARM_ENV, config_flag)


def hand_isolation_enabled(config_flag: bool) -> bool:
    """Resolve the HAND isolation flag (master env OR hand env OR config field)."""
    if _env_force_on(_MASTER_ENV):
        return True
    return _section_enabled(_HAND_ENV, config_flag)
