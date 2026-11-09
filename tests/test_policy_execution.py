import threading
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.preflight import validate_execution_config
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import ActionChunk
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.model import (
    XARM7_HARD_LOWER, XARM7_HARD_UPPER,
    XHAND_RATED_QPOS_MIN_RAD, XHAND_RATED_QPOS_MAX_RAD,
)
from dexmani_real.runtime.observation import ObservationNeeds


class PolicyExecutionTest(unittest.TestCase):
    def setUp(self):
        self.lower = np.array((*XARM7_HARD_LOWER, *XHAND_RATED_QPOS_MIN_RAD))
        self.upper = np.array((*XARM7_HARD_UPPER, *XHAND_RATED_QPOS_MAX_RAD))
        self.mapper = ActionMapper(
            joint_limits=np.column_stack((self.lower[:7], self.upper[:7])),
            hand_enabled=True, hand_lower_rad=self.lower[7:], hand_upper_rad=self.upper[7:],
        )

    def make_rollout(self, *, carry=True, mode="sync", action_steps=8):
        info = SimpleNamespace(
            requires_prev_action=carry, supports_rtc=not carry,
            n_obs_steps=2, n_action_steps=8, future_steps=8, action_mode="joint",
            observation_fields=("joint_state",),
        )
        execution = ExecutionConfig(mode=mode, action_steps=action_steps, control_dt_s=0.05)
        row = SimpleNamespace(arm={"qpos": np.zeros((1, 7))})
        rollout = Rollout(
            SimpleNamespace(stop_signal=threading.Event()), SimpleNamespace(execution=execution), info,
            cancel_epoch=0, robot=SimpleNamespace(send_action=Mock()),
            observations=SimpleNamespace(needs=ObservationNeeds()), record_needs=ObservationNeeds(),
            sensors=None, action_mapper=self.mapper,
            model=SimpleNamespace(future=None, predict=Mock(), poll=lambda: None),
            max_running_s=None, index=0, output_dir=None, record_raw=False,
        )
        rollout.history.append({"joint_state": np.zeros(19)})
        rollout.sample_observation = Mock(return_value=(row, None))
        rollout.update_history = Mock(return_value=True)
        rollout.check_motion = Mock()
        rollout.duration_expired = Mock(return_value=False)
        return rollout

    def step_sync(self, rollout):
        with patch("dexmani_real.deployment.rollout.time.monotonic_ns", return_value=1_000_000_000), \
                patch("dexmani_real.deployment.rollout.check_device_feedback", return_value=()):
            rollout.next_step_ns = 0
            rollout.step_sync()

    def test_joint_mapping_preserves_legal_branches_and_saturates(self):
        for target in (self.lower, self.upper, (self.lower + self.upper) / 2):
            mapped = self.mapper.map_joint(target[:7], target[7:])
            np.testing.assert_array_equal(np.r_[mapped.arm_qpos, mapped.hand_qpos], target)
        for boundary, offset in ((self.lower, -0.01), (self.upper, 0.01)):
            target = boundary + offset
            original = target.copy()
            mapped = self.mapper.map_joint(target[:7], target[7:])
            np.testing.assert_array_equal(np.r_[mapped.arm_qpos, mapped.hand_qpos], boundary)
            np.testing.assert_array_equal(target, original)

    def test_invalid_joint_targets_are_not_hidden_by_clipping(self):
        for width in (7, 12):
            for target in (np.zeros(width - 1), np.zeros((1, width)), np.full(width, np.nan),
                           np.full(width, np.inf), np.full(width, -np.inf)):
                arm, hand = (target, np.zeros(12)) if width == 7 else (np.zeros(7), target)
                with self.subTest(width=width, shape=target.shape), self.assertRaises(ValueError):
                    self.mapper.map_joint(arm, hand)

    def test_hand_intent_matches_enabled_devices(self):
        with self.assertRaises(ValueError):
            self.mapper.map_joint(np.zeros(7), None)
        self.mapper.hand_enabled = False
        self.assertIsNone(self.mapper.map_joint(np.zeros(7), None).hand_qpos)
        with self.assertRaises(ValueError):
            self.mapper.map_joint(np.zeros(7), np.zeros(12))

    def test_eef_mapping_keeps_its_ik_path(self):
        self.mapper.solver = SimpleNamespace(
            constraints=SimpleNamespace(workspace=np.tile([-1., 1.], (3, 1))),
            solve=Mock(return_value=SimpleNamespace(qpos=np.zeros(7))),
        )
        reference = (np.zeros(7), np.zeros(7), np.zeros(7), 0.05)
        result = self.mapper.map_eef(Pose([2., 0., 0.], [1., 0., 0., 0.]), self.upper[7:] + 0.1, reference)
        args, kwargs = self.mapper.solver.solve.call_args
        np.testing.assert_array_equal(args[0].p, [1., 0., 0.])
        np.testing.assert_array_equal(kwargs["hand_qpos"], self.upper[7:])
        self.assertEqual(kwargs["h"], 0.05)
        self.assertIsNone(result.rejection_reason)

    def test_ode_requires_sync_and_a_full_chunk(self):
        rollout = self.make_rollout()
        validate_execution_config(rollout.policy_info, rollout.execution)
        for changes in ({"mode": "async"}, {"mode": "rtc"}, {"action_steps": 7}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, NotImplementedError)):
                validate_execution_config(rollout.policy_info, replace(rollout.execution, **changes))

    def test_other_policies_keep_modes_prefixes_and_prediction_kwargs(self):
        for mode in ("sync", "async", "rtc"):
            with self.subTest(mode=mode):
                rollout = self.make_rollout(carry=False, mode=mode, action_steps=2)
                validate_execution_config(rollout.policy_info, rollout.execution)
                prefix = np.zeros((3, 19)) if mode == "rtc" else None
                rollout.submit(0, prefix=prefix)
                kwargs = rollout.model.predict.call_args.kwargs
                expected = {"reset_episode"}
                if prefix is not None:
                    expected |= {"rtc_prefix", "delay_steps", "rtc_guidance_cap"}
                    self.assertIs(kwargs["rtc_prefix"], prefix)
                    self.assertEqual(kwargs["delay_steps"], 1)
                self.assertEqual(set(kwargs), expected)

        rollout = self.make_rollout(carry=False, action_steps=2)
        rollout.plan = ActionChunk(np.tile((self.lower + self.upper) / 2, (8, 1)), 0)
        self.step_sync(rollout)
        self.step_sync(rollout)
        self.assertIsNone(rollout.plan)
        self.assertIsNone(rollout.prev_action)
        self.assertEqual(rollout.robot.send_action.call_count, 2)

    def test_ode_carries_the_last_submitted_target(self):
        rollout = self.make_rollout()
        rollout.submit(None)
        self.assertIsNone(rollout.model.predict.call_args.kwargs["prev_action"])
        self.assertTrue(rollout.model.predict.call_args.kwargs["reset_episode"])
        rollout.robot.send_action.assert_not_called()

        actions = np.tile((self.lower + self.upper) / 2, (8, 1))
        actions[:, 0] = self.upper[0] - 0.07 + np.arange(8) * 0.02
        actions[:, 7:] = self.upper[7:] + 0.1
        rollout.plan = ActionChunk(actions, 0)
        for index in range(8):
            self.step_sync(rollout)
            command = rollout.robot.send_action.call_args.args[0]
            np.testing.assert_array_equal(rollout.prev_action, np.r_[command.arm_qpos, command.hand_qpos])
            np.testing.assert_array_equal(rollout.prev_action, np.clip(actions[index], self.lower, self.upper))
        self.assertIsNone(rollout.plan)
        self.assertEqual(rollout.robot.send_action.call_count, 8)

        self.step_sync(rollout)
        kwargs = rollout.model.predict.call_args.kwargs
        previous = kwargs["prev_action"]
        self.assertFalse(kwargs["reset_episode"])
        np.testing.assert_array_equal(previous, np.clip(actions[-1], self.lower, self.upper))
        rollout.plan = ActionChunk(np.tile(self.lower - 0.1, (8, 1)), 0)
        rollout.cursor = 0
        self.step_sync(rollout)
        np.testing.assert_array_equal(rollout.prev_action, self.lower)
        np.testing.assert_array_equal(previous, np.clip(actions[-1], self.lower, self.upper))

    def test_failed_submission_does_not_advance_carry_or_cursor(self):
        rollout = self.make_rollout()
        previous = (self.lower + self.upper) / 2
        rollout.prev_action = previous.copy()
        rollout.plan = ActionChunk(np.tile(self.upper, (8, 1)), 0)
        rollout.robot.send_action.side_effect = RuntimeError("submission failed")
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            self.step_sync(rollout)
        np.testing.assert_array_equal(rollout.prev_action, previous)
        self.assertEqual(rollout.cursor, 0)
        self.assertTrue(rollout.runtime_state.stop_signal.is_set())

    def test_stop_mid_chunk_and_new_episode(self):
        rollout = self.make_rollout()
        rollout.plan = ActionChunk(np.tile((self.lower + self.upper) / 2, (8, 1)), 0)
        self.step_sync(rollout)
        rollout.stop("operator stopped")
        rollout.step()
        self.assertIsNone(rollout.prev_action)
        self.assertEqual(rollout.robot.send_action.call_count, 1)
        new_episode = self.make_rollout()
        new_episode.submit(None)
        self.assertIsNone(new_episode.model.predict.call_args.kwargs["prev_action"])
        self.assertTrue(new_episode.model.predict.call_args.kwargs["reset_episode"])

    def test_worker_forwards_carry_and_warmup_does_not_submit(self):
        rollout = self.make_rollout()
        runtime = SimpleNamespace(info=rollout.policy_info, reset_episode=Mock(), warmup=Mock(), predict=Mock(), close=Mock())
        worker = PolicyWorker(lambda: runtime, build_observation=lambda rows: rows)
        rollout.model = worker
        previous = (self.lower + self.upper) / 2
        rollout.prev_action = previous
        try:
            worker.load()
            worker.future.result()
            worker.poll()
            worker.warmup(tuple(rollout.history))
            worker.future.result()
            worker.poll()
            runtime.reset_episode.assert_not_called()
            runtime.predict.assert_not_called()
            self.assertIs(rollout.prev_action, previous)
            rollout.robot.send_action.assert_not_called()
            rollout.submit(None)
            worker.future.result()
            worker.poll()
            self.assertIs(runtime.predict.call_args.kwargs["prev_action"], previous)
            runtime.reset_episode.assert_called_once_with()
        finally:
            worker.close().result()


if __name__ == "__main__":
    unittest.main()
