"""Shared memory infrastructure for zero-copy cross-process communication.

Public API:
    SharedStorage, SharedStorageConfig, HOME and priority arm-control contracts
    read_arm_state, read_arm_state_dict
    read_hand_state, read_hand_state_dict

Homing and process supervision live in ``robot.homing`` and
``runtime.supervisor`` rather than the data-plane module.
"""

from __future__ import annotations
