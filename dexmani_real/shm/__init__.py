"""Shared memory infrastructure for zero-copy cross-process communication.

Public API:
    SharedStorage, SharedStorageConfig, HOME_SENTINEL
    read_arm_state, read_arm_state_k, read_arm_state_dict
    read_hand_state, read_hand_state_k, read_hand_state_dict
    write_hand_cmd
    hand_home_converge, wait_for_arm_home
    shutdown_processes, wait_subsystem_ready
    print_health_summary
"""

from __future__ import annotations
