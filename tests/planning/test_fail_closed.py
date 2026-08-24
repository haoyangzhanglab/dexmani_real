from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, cast

import numpy as np

from dexmani_real.planning.collision import CollisionModel
from dexmani_real.planning.planner import XArm7MotionPlanner


class CollisionFailClosedTest(unittest.TestCase):
    def make_collision_model(self, *, allow_fallback: bool) -> CollisionModel:
        model = object.__new__(CollisionModel)
        model._hand_dof = True
        model._hand_qpos = None
        model._expected_qpos_shape = (19,)
        model._nq = 19
        model._allow_unset_hand_qpos = allow_fallback
        return model

    def test_arm_only_qpos_requires_current_hand_state(self) -> None:
        model = self.make_collision_model(allow_fallback=False)
        with self.assertRaisesRegex(RuntimeError, "hand_qpos is required"):
            model._to_full_qpos(np.zeros(7))

    def test_offline_fallback_requires_explicit_opt_in(self) -> None:
        model = self.make_collision_model(allow_fallback=True)
        self.assertEqual(model._to_full_qpos(np.zeros(7)).shape, (19,))


class PlannerAssetContractTest(unittest.TestCase):
    def test_missing_srdf_is_not_generated_by_constructor(self) -> None:
        calls: list[tuple[object, ...]] = []
        fake_mplib = types.SimpleNamespace(
            urdf_utils=types.SimpleNamespace(
                generate_srdf=lambda *args: calls.append(args)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = types.SimpleNamespace(
                workspace_bounds=None,
                joint_vel_limits_deg=np.ones(7),
                joint_acc_scale=1.0,
                urdf_path=str(root / "robot.urdf"),
                srdf_path=str(root / "missing.srdf"),
            )
            with patch_module("mplib", fake_mplib):
                with self.assertRaises(FileNotFoundError):
                    XArm7MotionPlanner(cast(Any, config))
        self.assertEqual(calls, [])


class patch_module:
    def __init__(self, name: str, module: object) -> None:
        self.name = name
        self.module = module
        self.previous: object | None = None

    def __enter__(self) -> None:
        self.previous = sys.modules.get(self.name)
        sys.modules[self.name] = self.module  # type: ignore[assignment]

    def __exit__(self, *args: object) -> None:
        if self.previous is None:
            sys.modules.pop(self.name, None)
        else:
            sys.modules[self.name] = self.previous  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
