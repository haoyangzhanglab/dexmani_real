"""Focused failure semantics for hand retargeting and command admission."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nlopt
import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control.publication import validate_hand_command_bounds
from dexmani_real.teleop.control_loop.action_proposal import compute_hand_joint_proposal
from dexmani_real.teleop.control_loop.hand_control import (
    HandRetargetObservationCache,
    compute_hand_command,
    reset_hand_retargeter,
)
from dexmani_real.teleop.retargeting.dexpilot import PriorDexPilotOptimizer
from dexmani_real.teleop.retargeting.retargeter import (
    DexPilotHandRetargeter,
    TAGHandRetargeter,
)
from dexmani_real.teleop.retargeting.tag_optimizer import HandOptimizer


class _SentinelRuntimeError(RuntimeError):
    pass


def _tag_optimizer(
    *,
    stage1: np.ndarray | Exception,
    stage2: np.ndarray | Exception,
) -> HandOptimizer:
    optimizer = object.__new__(HandOptimizer)
    optimizer.dof = 12
    optimizer.finger_num = 5
    optimizer.finger_scale = np.ones(5, dtype=np.float64)
    optimizer.joint_limits_lower = np.full(12, -1.0, dtype=np.float64)
    optimizer.joint_limits_upper = np.full(12, 1.0, dtype=np.float64)
    optimizer.last_qpos = np.zeros(12, dtype=np.float64)
    optimizer.qpos_stage1 = None
    optimizer.pinch_factors = np.zeros(5, dtype=np.float64)
    optimizer.pinch_start_dist = 2.0
    optimizer.pinch_full_dist = 1.0
    optimizer.pinch_ema_alpha = 1.0
    optimizer.pinch_skip_threshold = 0.01
    optimizer._current_target = None
    optimizer._current_q_prior = None
    optimizer._stage1_warn = Mock()
    optimizer._stage2_warn = Mock()
    optimizer._bounds_warn = Mock()
    optimizer.opt_s1 = Mock()
    optimizer.opt_s2 = Mock()
    if isinstance(stage1, Exception):
        optimizer.opt_s1.optimize.side_effect = stage1
    else:
        optimizer.opt_s1.optimize.return_value = stage1
    if isinstance(stage2, Exception):
        optimizer.opt_s2.optimize.side_effect = stage2
    else:
        optimizer.opt_s2.optimize.return_value = stage2
    return optimizer


class HandCommandBoundaryTest(unittest.TestCase):
    def test_config_rejects_limit_nesting_violation(self) -> None:
        command_lower = list(hand_defaults.qpos_min_rad)
        command_lower[0] = hand_defaults.mechanical_qpos_min_rad[0] - 0.1

        with self.assertRaisesRegex(ValueError, "inside mechanical limits"):
            replace(hand_defaults, qpos_min_rad=tuple(command_lower))

    def test_publication_endpoint_checks_operational_and_mechanical_bounds(self) -> None:
        operational_lower = np.full(12, -1.0)
        operational_upper = np.full(12, 1.0)
        mechanical_lower = np.full(12, -2.0)
        mechanical_upper = np.full(12, 2.0)
        accepted = validate_hand_command_bounds(
            np.zeros(12),
            operational_lower,
            operational_upper,
            mechanical_lower,
            mechanical_upper,
        )
        np.testing.assert_array_equal(accepted, np.zeros(12))

        with self.assertRaisesRegex(ValueError, "operational joint limits"):
            validate_hand_command_bounds(
                np.full(12, 1.1),
                operational_lower,
                operational_upper,
                mechanical_lower,
                mechanical_upper,
            )
        with self.assertRaisesRegex(ValueError, "mechanical joint limits"):
            validate_hand_command_bounds(
                np.full(12, 1.5),
                np.full(12, -2.0),
                np.full(12, 2.0),
                operational_lower,
                operational_upper,
            )

    def test_hand_shaping_order_is_ramp_then_operational_clip_then_delta(self) -> None:
        retargeter = SimpleNamespace(retarget=Mock(return_value=np.full(12, 2.0)))
        proposal = compute_hand_joint_proposal(
            retargeter,
            {"ring_sequence": 1, "landmarks": np.zeros((21, 3))},
            np.zeros(12),
            hand_available=True,
            retarget_cache=HandRetargetObservationCache(),
            ramp_start_qpos_rad=np.zeros(12),
            ramp_step=0,
            ramp_total_frames=2,
            command_lower_rad=np.full(12, -0.5),
            command_upper_rad=np.full(12, 0.5),
            max_delta_rad_per_tick=1.0,
        )

        np.testing.assert_array_equal(proposal.qpos_rad, np.full(12, 0.5))
        np.testing.assert_array_equal(proposal.raw_qpos_rad, np.full(12, 2.0))


class RetargetFailureSemanticsTest(unittest.TestCase):
    def test_expected_no_solution_holds_previous_command(self) -> None:
        previous = np.linspace(0.0, 0.11, 12)
        retargeter = SimpleNamespace(retarget=Mock(return_value=None))

        command, succeeded = compute_hand_command(
            retargeter,
            {"ring_sequence": 1, "landmarks": np.zeros((21, 3))},
            previous,
            True,
            HandRetargetObservationCache(),
        )

        self.assertFalse(succeeded)
        np.testing.assert_array_equal(command, previous)

    def test_unexpected_retarget_error_propagates(self) -> None:
        retargeter = SimpleNamespace(
            retarget=Mock(side_effect=_SentinelRuntimeError("retarget sentinel"))
        )

        with self.assertRaisesRegex(_SentinelRuntimeError, "retarget sentinel"):
            compute_hand_command(
                retargeter,
                {"ring_sequence": 1, "landmarks": np.zeros((21, 3))},
                np.zeros(12),
                True,
                HandRetargetObservationCache(),
            )

    def test_reset_error_propagates(self) -> None:
        retargeter = SimpleNamespace(
            reset=Mock(side_effect=_SentinelRuntimeError("reset sentinel"))
        )

        with self.assertRaisesRegex(_SentinelRuntimeError, "reset sentinel"):
            reset_hand_retargeter(retargeter, np.zeros(12))

    def test_tag_facade_does_not_swallow_optimizer_runtime_error(self) -> None:
        retargeter = object.__new__(TAGHandRetargeter)
        retargeter._prior_weight = 0.0
        retargeter._pinky_scale = 1.0
        retargeter._pinky_palm_scale = 1.0
        retargeter._R_mano_to_urdf = np.eye(3)
        retargeter._mapping_model_to_sdk = np.arange(12)
        retargeter.debug = False
        retargeter._optimizer = SimpleNamespace(
            solve=Mock(side_effect=_SentinelRuntimeError("TAG sentinel"))
        )

        with (
            patch(
                "dexmani_real.teleop.retargeting.retargeter.validate_landmarks",
                return_value=(True, ""),
            ),
            patch(
                "dexmani_real.teleop.retargeting.retargeter._estimate_palm_frame",
                return_value=np.eye(3),
            ),
            self.assertRaisesRegex(_SentinelRuntimeError, "TAG sentinel"),
        ):
            retargeter.retarget(np.zeros((21, 3)))

    def test_xhand_facade_maps_only_roundoff_to_no_solution(self) -> None:
        retargeter = object.__new__(DexPilotHandRetargeter)
        retargeter._prior_weight = 0.0
        retargeter.fixed_joint_values = np.array([])
        retargeter.retargeted_joint_order = np.arange(12)
        retargeter.debug_adapters = False
        retargeter._build_ref_value = Mock(return_value=np.zeros((15, 3)))
        backend = Mock()
        retargeter.retargeter = SimpleNamespace(retarget=backend)

        with (
            patch(
                "dexmani_real.teleop.retargeting.retargeter.validate_landmarks",
                return_value=(True, ""),
            ),
            patch(
                "dexmani_real.teleop.retargeting.retargeter._estimate_palm_frame",
                return_value=np.eye(3),
            ),
        ):
            backend.side_effect = nlopt.RoundoffLimited("roundoff")
            self.assertIsNone(retargeter.retarget(np.zeros((21, 3))))

            backend.side_effect = _SentinelRuntimeError("DexPilot sentinel")
            with self.assertRaisesRegex(_SentinelRuntimeError, "DexPilot sentinel"):
                retargeter.retarget(np.zeros((21, 3)))


class OptimizerFailureSemanticsTest(unittest.TestCase):
    def test_tag_stage1_roundoff_returns_no_solution(self) -> None:
        optimizer = _tag_optimizer(
            stage1=nlopt.RoundoffLimited("stage 1 roundoff"),
            stage2=np.zeros(12),
        )

        self.assertIsNone(optimizer.solve(np.zeros((5, 3))))
        optimizer.opt_s2.optimize.assert_not_called()

    def test_tag_stage1_unexpected_runtime_error_propagates(self) -> None:
        optimizer = _tag_optimizer(
            stage1=_SentinelRuntimeError("stage 1 sentinel"),
            stage2=np.zeros(12),
        )

        with self.assertRaisesRegex(_SentinelRuntimeError, "stage 1 sentinel"):
            optimizer.solve(np.zeros((5, 3)))

    def test_tag_stage2_roundoff_returns_validated_stage1_solution(self) -> None:
        stage1 = np.full(12, 0.25)
        optimizer = _tag_optimizer(
            stage1=stage1,
            stage2=nlopt.RoundoffLimited("stage 2 roundoff"),
        )

        result = optimizer.solve(np.zeros((5, 3)))

        np.testing.assert_array_equal(result, stage1)
        np.testing.assert_array_equal(optimizer.last_qpos, stage1)

    def test_tag_stage2_unexpected_runtime_error_propagates(self) -> None:
        optimizer = _tag_optimizer(
            stage1=np.full(12, 0.25),
            stage2=_SentinelRuntimeError("stage 2 sentinel"),
        )

        with self.assertRaisesRegex(_SentinelRuntimeError, "stage 2 sentinel"):
            optimizer.solve(np.zeros((5, 3)))

    def test_tag_stage2_invalid_output_propagates(self) -> None:
        invalid = np.zeros(12)
        invalid[0] = np.nan
        optimizer = _tag_optimizer(stage1=np.zeros(12), stage2=invalid)

        with self.assertRaisesRegex(ValueError, "Stage 2 result"):
            optimizer.solve(np.zeros((5, 3)))

    def test_dexpilot_optimizer_does_not_return_warm_start_on_runtime_error(
        self,
    ) -> None:
        optimizer = object.__new__(PriorDexPilotOptimizer)
        optimizer.idx_pin2fixed = np.array([], dtype=np.intp)
        optimizer.get_objective_function = Mock(return_value=Mock())
        optimizer.opt = Mock()
        optimizer.opt.optimize.side_effect = _SentinelRuntimeError(
            "DexPilot optimizer sentinel"
        )

        with self.assertRaisesRegex(
            _SentinelRuntimeError, "DexPilot optimizer sentinel"
        ):
            optimizer.retarget(np.zeros((15, 3)), np.array([]), np.zeros(12))


if __name__ == "__main__":
    unittest.main()
