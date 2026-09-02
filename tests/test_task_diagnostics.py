"""Offline coverage for passive task diagnostic evidence collection."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from dexmani_real.deployment.task_diagnostics import TaskDiagnosticsObserver
from dexmani_real.deployment.task_scene import TaskSceneCard
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    COUPLED_COMMAND_DTYPE,
    HAND_STATE_DTYPE,
    HAND_TACTILE_DTYPE,
    POLICY_PLAN_DTYPE,
)
from dexmani_real.runtime.safety import SafetyState, begin_motion


class TaskDiagnosticsObserverTest(unittest.TestCase):
    def _channels(self) -> RuntimeChannels:
        return RuntimeChannels.create(
            prefix=f"task_diagnostics_{uuid4().hex}",
            config=RuntimeChannelsConfig(
                camera_ring_maxlen=1,
                vr_ring_maxlen=1,
                arm_state_ring_maxlen=2,
                hand_state_ring_maxlen=2,
                hand_tactile_ring_maxlen=1,
                coupled_cmd_ring_maxlen=2,
                record_control_ring_maxlen=1,
                record_sample_ring_maxlen=1,
                record_status_ring_maxlen=1,
                policy_plan_ring_maxlen=2,
                pointcloud_ring_maxlen=1,
                camera_rgb_shape=(1, 1, 3),
                camera_depth_shape=(1, 1),
            ),
        )

    @staticmethod
    def _scene_card() -> TaskSceneCard:
        return TaskSceneCard(
            source_path=Path("/tmp/task_scene_card.json"),
            sha256="a" * 64,
            task_name="pick_place_toy",
            object_description="test object",
            object_start_description="test start",
            target_description="test target",
            success_criterion="test success",
            phase_endpoint_indices=(
                ("approach", 1),
                ("grasp", 2),
                ("lift", 3),
                ("place", 4),
            ),
        )

    @staticmethod
    def _write_feedback(shared: RuntimeChannels, *, action_id: int) -> None:
        now_ns = time.monotonic_ns()
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        arm["connected"][0] = 1
        arm["state_valid"][0] = 1
        arm["last_cmd_seq"][0] = action_id
        arm["source_monotonic_ns"][0] = now_ns
        arm["publish_monotonic_ns"][0] = now_ns
        shared.arm_state_ring.write(arm)

        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        hand["connected"][0] = 1
        hand["state_valid"][0] = 1
        hand["accepted_target_action_id"][0] = action_id
        hand["source_monotonic_ns"][0] = now_ns
        hand["publish_monotonic_ns"][0] = now_ns
        shared.hand_state_ring.write(hand)

        tactile = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
        tactile["source_monotonic_ns"][0] = now_ns
        tactile["fresh"][0] = 1
        shared.hand_tactile_ring.write(tactile)

    def test_records_raw_prediction_shaped_endpoint_and_paired_ack(self) -> None:
        shared = self._channels()
        try:
            shared.safety_state.value = int(SafetyState.ARMED)
            self.assertTrue(begin_motion(shared))
            generation = int(shared.run_generation.value)
            target_ns = time.monotonic_ns()
            plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
            plan["plan_id"][0] = 3
            plan["run_generation"][0] = generation
            plan["observation_id"][0] = 7
            plan["observation_anchor_monotonic_ns"][0] = target_ns - 4
            plan["observation_latest_source_monotonic_ns"][0] = target_ns - 8
            plan["inference_started_monotonic_ns"][0] = target_ns - 3
            plan["inference_finished_monotonic_ns"][0] = target_ns - 2
            plan["num_steps"][0] = 1
            plan["arm_present"][0] = 1
            plan["hand_present"][0] = 1
            plan["target_monotonic_ns"][0, 0] = target_ns
            plan["valid_mask"][0, 0] = 1
            plan["arm_qpos"][0, 0] = np.arange(7, dtype=np.float64)
            plan["hand_qpos"][0, 0] = np.arange(12, dtype=np.float64)
            shared.policy_plan_ring.write(plan)

            with tempfile.TemporaryDirectory() as directory:
                observer = TaskDiagnosticsObserver(
                    shared,
                    receipt_dir=directory,
                    scene_card=self._scene_card(),
                )
                observer._capture_b_pre_scene()
                observer._drain_policy_plans()
                self._write_feedback(shared, action_id=0)
                command = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
                command["run_generation"][0] = generation
                command["observation_id"][0] = 7
                command["action_id"][0] = 23
                command["created_monotonic_ns"][0] = target_ns
                command["scheduled_target_monotonic_ns"][0] = target_ns
                command["target_monotonic_ns"][0] = target_ns
                command["valid_until_monotonic_ns"][0] = target_ns + 1_000_000
                command["arm_present"][0] = 1
                command["hand_present"][0] = 1
                command["arm_qpos"][0] = np.arange(7, dtype=np.float64) + 0.5
                command["hand_qpos"][0] = np.arange(12, dtype=np.float64) + 0.25
                shared.coupled_cmd_ring.write(command)
                observer._drain_coupled_commands()
                self._write_feedback(shared, action_id=23)
                observer._observe_acknowledgements()
                output = observer.persist()
                assert output is not None
                manifest = json.loads((output / "manifest.json").read_text("utf-8"))
                self.assertEqual(
                    manifest["collection"]["event_count"],
                    1,
                    manifest["collection"]["read_errors"],
                )
                event = manifest["events"][0]
                self.assertTrue(event["raw_prediction_available"])
                self.assertEqual(event["raw_prediction"]["plan_id"], 3)
                self.assertEqual(event["shaped_ipc_endpoint"]["arm_qpos"][0], 0.5)
                self.assertTrue(event["acknowledgement"]["arm_acknowledged"])
                self.assertTrue(event["acknowledgement"]["hand_acknowledged"])
        finally:
            self.assertTrue(shared.close())

    def test_ignores_pre_b_hand_home_command_when_indexing_task_events(self) -> None:
        shared = self._channels()
        try:
            shared.safety_state.value = int(SafetyState.ARMED)
            home_generation = int(shared.run_generation.value)
            with tempfile.TemporaryDirectory() as directory:
                observer = TaskDiagnosticsObserver(
                    shared,
                    receipt_dir=directory,
                    scene_card=self._scene_card(),
                )
                home = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
                home["run_generation"][0] = home_generation
                home["action_id"][0] = 1
                home["hand_present"][0] = 1
                shared.coupled_cmd_ring.write(home)
                observer._drain_coupled_commands()

                self.assertTrue(begin_motion(shared))
                task_generation = int(shared.run_generation.value)
                observer._capture_b_pre_scene()

                target_ns = time.monotonic_ns()
                task = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
                task["run_generation"][0] = task_generation
                task["action_id"][0] = 2
                task["created_monotonic_ns"][0] = target_ns
                task["scheduled_target_monotonic_ns"][0] = target_ns
                task["target_monotonic_ns"][0] = target_ns
                task["valid_until_monotonic_ns"][0] = target_ns + 1_000_000
                task["arm_present"][0] = 1
                task["hand_present"][0] = 1
                shared.coupled_cmd_ring.write(task)
                observer._drain_coupled_commands()

                output = observer.persist()
                assert output is not None
                manifest = json.loads((output / "manifest.json").read_text("utf-8"))
                self.assertEqual(manifest["collection"]["event_count"], 1)
                event = manifest["events"][0]
                self.assertEqual(event["action_id"], 2)
                self.assertEqual(event["published_endpoint_index"], 1)
                self.assertEqual(manifest["scene_frames"]["approach"]["action_id"], 2)
        finally:
            self.assertTrue(shared.close())
