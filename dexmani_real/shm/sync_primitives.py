"""Sync primitives for two-phase policy-robot handshake.

Two-phase handshake protocol (ref: ManiUniCon SharedStorage):

  1. Robot completes execution  →  sets ``robot_ready``
  2. Policy waits for ``robot_ready``, clears it, generates action, sets ``policy_ready``
  3. Robot waits for ``policy_ready``, clears it, executes action
  4. Repeat from step 1

Uses ``multiprocessing.Event`` (not ``threading.Event``) so the handshake works
both in-process (ArmInnerLoop thread ↔ main thread) and cross-process (a future
standalone policy process).
"""

from __future__ import annotations

import multiprocessing as mp

__all__ = ["SharedSyncPrimitives"]


class SharedSyncPrimitives:
    """Two-phase handshake events for synchronized policy-robot execution.

    Usage::

        sync = SharedSyncPrimitives()

        # --- Robot side (inner loop thread) ---
        sync.robot_ready.set()         # step 1: execution complete
        sync.policy_ready.wait()       # step 3: wait for next action
        sync.policy_ready.clear()

        # --- Policy side (controller tick) ---
        sync.robot_ready.wait()        # step 2: wait for robot
        sync.robot_ready.clear()
        # ... generate action, send to robot ...
        sync.policy_ready.set()        # step 2 (continued): signal new action
    """

    def __init__(self) -> None:
        # Initially True: robot is ready to receive the very first action.
        self.robot_ready = mp.Event()
        self.robot_ready.set()

        # Initially False: policy has not generated actions yet.
        self.policy_ready = mp.Event()
