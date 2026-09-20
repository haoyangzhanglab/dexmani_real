"""Offline guard for teleop's evidence-role wiring (TASKBOOK T4/T5).

Teleop's camera and recorder exist only to produce recording evidence (the
camera is started only when recording is enabled). Their failure — including
the producer-owned camera source-stall latch — must degrade the session result,
never latch the physical fault or terminate teleoperation; the recording layer
keeps saving the collected prefix.

That wiring lives inside the session entry point, which starts real processes,
so it cannot be executed in an offline test. It is pinned structurally here so
an accidental revert (a camera that faults again, or a supervisor that is no
longer told which roles are services) fails the suite instead of passing
silently until a real camera stalls.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_SESSION = (
    Path(__file__).resolve().parents[1] / "dexmani_real" / "teleop" / "session.py"
)
_EVIDENCE_ROLES = {"camera", "recorder"}


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


class TeleopEvidenceRoleWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _SESSION.read_text(encoding="utf-8")
        self.tree = ast.parse(self.source)

    def test_camera_process_is_started_evidence_only(self):
        specs = [
            call
            for call in _calls(self.tree, "ProcessSpec")
            if isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "camera"
        ]
        self.assertEqual(len(specs), 1, "expected exactly one teleop camera spec")
        args = specs[0].args[2]
        self.assertIsInstance(args, ast.Tuple)
        latch = args.elts[-1]
        self.assertIsInstance(latch, ast.Constant)
        self.assertIs(
            latch.value,
            False,
            "the teleop camera is evidence-only and must not latch the "
            "physical fault",
        )

    def test_supervisor_receives_the_evidence_roles(self):
        supervisors = _calls(self.tree, "run_supervisor")
        self.assertEqual(len(supervisors), 1)
        passed = [
            keyword
            for keyword in supervisors[0].keywords
            if keyword.arg == "service_process_names"
        ]
        self.assertEqual(
            len(passed),
            1,
            "run_supervisor must be told which teleop roles are evidence-only",
        )
        assigned = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "service_process_names"
                for target in node.targets
            )
        ]
        self.assertTrue(assigned, "teleop must define its service_process_names")
        role_names = {
            element.value
            for node in ast.walk(assigned[0])
            for element in ast.walk(node)
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
        self.assertTrue(
            _EVIDENCE_ROLES.issubset(role_names),
            f"evidence roles {sorted(_EVIDENCE_ROLES)} must be services, "
            f"found {sorted(role_names)}",
        )


if __name__ == "__main__":
    unittest.main()
