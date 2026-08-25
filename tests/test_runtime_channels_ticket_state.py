"""Offline allocation checks for coupled-command ticket shared state."""

from __future__ import annotations

import time
import unittest
from uuid import uuid4

import numpy as np

from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.runtime.safety import (
    SafetyState,
    begin_motion,
    publish_coupled_command_if_motion_permitted,
    revoke_motion,
)


class RuntimeChannelsTicketStateTest(unittest.TestCase):
    def _create_channels(self) -> RuntimeChannels:
        config = RuntimeChannelsConfig(
            camera_ring_maxlen=1,
            vr_ring_maxlen=1,
            arm_state_ring_maxlen=1,
            hand_state_ring_maxlen=1,
            hand_tactile_ring_maxlen=1,
            coupled_cmd_ring_maxlen=1,
            record_control_ring_maxlen=1,
            record_sample_ring_maxlen=1,
            record_status_ring_maxlen=1,
            policy_plan_ring_maxlen=1,
            pointcloud_ring_maxlen=1,
            camera_rgb_shape=(1, 1, 3),
            camera_depth_shape=(1, 1),
        )
        return RuntimeChannels.create(
            prefix=f"ticket_state_{uuid4().hex}", config=config
        )

    def test_ticket_state_is_allocated_and_zeroed(self) -> None:
        shared = self._create_channels()
        try:
            self.assertEqual(shared.active_coupled_command_sequence.value, 0)
        finally:
            self.assertTrue(shared.close())

    def test_coupled_record_round_trips_through_shared_memory(self) -> None:
        shared = self._create_channels()
        try:
            self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
            self.assertTrue(begin_motion(shared))
            generation = int(shared.run_generation.value)
            now_ns = time.monotonic_ns()
            frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
            frame["run_generation"][0] = generation
            frame["observation_id"][0] = 17
            frame["action_id"][0] = 23
            frame["created_monotonic_ns"][0] = now_ns
            frame["target_monotonic_ns"][0] = now_ns
            frame["valid_until_monotonic_ns"][0] = now_ns + 500_000_000
            frame["arm_present"][0] = 1
            frame["hand_present"][0] = 1
            frame["arm_qpos"][0] = np.arange(7, dtype=np.float64)
            frame["hand_qpos"][0] = np.arange(12, dtype=np.float64)

            ticket = publish_coupled_command_if_motion_permitted(
                shared,
                expected_run_generation=generation,
                frame=frame,
            )
            result = shared.coupled_cmd_ring.read_latest()

            assert ticket is not None and result is not None
            record, _published_ns, sequence = result
            self.assertEqual(sequence, ticket.ring_sequence)
            self.assertEqual(int(record["action_id"][0]), 23)
            np.testing.assert_array_equal(record["arm_qpos"][0], frame["arm_qpos"][0])
            np.testing.assert_array_equal(record["hand_qpos"][0], frame["hand_qpos"][0])
        finally:
            self.assertTrue(shared.close())
