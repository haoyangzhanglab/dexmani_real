"""Typed cross-process channels and causal readers.

Public API:
    RuntimeChannels, RuntimeChannelsConfig, HOME and priority arm-control contracts
    read_arm_state, read_arm_state_dict
    read_hand_state, read_hand_state_dict

Homing and process supervision live in ``control.arm_home`` and
``runtime.supervisor`` rather than the data-plane module.
"""

from __future__ import annotations
