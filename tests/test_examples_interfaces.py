"""Offline CLI behavior and dataset export guards.

Hardware sessions are replaced at their boundary; the run_policy entry point
is parsed but never executed. These checks preserve research-facing arguments
and protect recorded data from accidental overwrite.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _REPO_ROOT / "examples"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_example(name: str):
    path = _EXAMPLES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclass/typing resolution can find it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Only a genuinely missing dependency may skip this module: a renamed or
# broken symbol must fail the suite rather than hide behind a skip.
try:
    _export_cli = _load_example("export_policy_zarr")
    _EXPORT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment guard
    _export_cli = None
    _EXPORT_IMPORT_ERROR = exc


@unittest.skipIf(
    _EXPORT_IMPORT_ERROR is not None,
    f"export CLI dependencies unavailable: {_EXPORT_IMPORT_ERROR}",
)
class ExportOutputGuardTest(unittest.TestCase):
    """Export output guards preserve source data and refuse occupied targets."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _guard(self, output, input_root):
        return _export_cli._resolve_output_path(
            output, Path("datasets/task.zarr"), input_root
        )

    def test_default_path_unchanged(self):
        resolved = self._guard(None, self.root / "in")
        self.assertEqual(resolved, Path("datasets/task.zarr"))

    def test_plain_new_output_allowed(self):
        target = self.root / "next_generation.zarr"
        self.assertEqual(self._guard(target, self.root / "in"), target)

    def test_tilde_output_is_returned_expanded(self):
        """The path that is validated is the path that gets written.

        Without the expansion a ``~``-prefixed --output would be checked
        against the home directory and then written to a literal ``~/`` tree
        under the working directory.
        """
        resolved = self._guard(Path("~/gen2.zarr"), self.root / "in")
        self.assertEqual(resolved, Path("~/gen2.zarr").expanduser())
        self.assertNotIn("~", str(resolved))

    def test_protected_source_roots_refused(self):
        for relative in (
            "episodes/task/episode_001.zarr",
            "episodes_processed/task/out.zarr",
            "rollouts/task/session/out.zarr",
        ):
            with self.subTest(target=relative):
                target = _REPO_ROOT / relative
                with self.assertRaises(ValueError):
                    self._guard(target, self.root / "in")

    def test_input_root_refused(self):
        input_root = self.root / "processed_task"
        input_root.mkdir()
        with self.assertRaises(ValueError):
            self._guard(input_root / "sneaky.zarr", input_root)

    def test_symlink_escape_is_refused_by_resolved_target(self):
        # A symlink whose RESOLVED location lands inside a protected source
        # root must be refused even though the link itself is elsewhere.
        protected = Path("episodes").resolve(strict=False)
        link = self.root / "escape.zarr"
        link.symlink_to(protected / "inner.zarr")
        with self.assertRaises(ValueError):
            self._guard(link, self.root / "in")

    def test_existing_zarr_store_interior_refused(self):
        store = self.root / "existing.zarr"
        (store / "data").mkdir(parents=True)
        with self.assertRaises(ValueError):
            self._guard(store / "data" / "new.zarr", self.root / "in")

    def test_guard_holds_from_a_foreign_working_directory(self):
        """Protected roots are repository-anchored, not CWD-anchored."""
        import os

        original = Path.cwd()
        try:
            os.chdir(self.root)
            with self.assertRaises(ValueError):
                self._guard(
                    _REPO_ROOT / "episodes" / "pick_place_toy" / "evil.zarr",
                    self.root / "in",
                )
        finally:
            os.chdir(original)

    def test_occupied_target_refused_in_both_modes(self):
        """Preflight and real export share the same refusal for a target."""
        import contextlib
        import io

        occupied = self.root / "occupied.zarr"
        occupied.mkdir()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = _export_cli.main(
                ["some_task", "--output", str(occupied), "--dry-run"]
            )
        self.assertEqual(code, 2)
        self.assertIn("refusing to overwrite existing output", stderr.getvalue())

    def test_parser_exposes_narrow_output(self):
        parser = _export_cli._parser()
        args = parser.parse_args(
            ["episodes_processed/task", "--output", "/tmp/gen2.zarr"]
        )
        self.assertEqual(args.output, Path("/tmp/gen2.zarr"))
        args_default = parser.parse_args(["episodes_processed/task"])
        self.assertIsNone(args_default.output)


def _load_example_or_skip(testcase: unittest.TestCase, name: str):
    try:
        return _load_example(name)
    except ImportError as exc:  # pragma: no cover - environment guard
        testcase.skipTest(f"{name} dependencies unavailable: {exc}")


class AffectedParserSmokeTest(unittest.TestCase):
    """Offline argument/configuration checks with hardware lifecycle mocked.

    Hardware session calls are replaced at the lifecycle boundary. Real CLI
    parsing and resolved configuration are checked without batch --help runs.
    run_policy is parsed only; its hardware main is never invoked.
    """

    def test_run_policy_num_episodes_selects_trials(self):
        module = _load_example_or_skip(self, "run_policy")
        args = module._parser().parse_args(
            ["policy/task/exp", "--num-episodes", "3", "--max-duration", "12"]
        )
        self.assertEqual(args.num_trials, 3)
        self.assertEqual(args.max_running_s, 12.0)

    def test_process_episodes_keeps_output_root_annotations_dry_run(self):
        module = _load_example_or_skip(self, "process_episodes")
        args = module._parser().parse_args(
            [
                "episodes/task",
                "--output-root",
                "/tmp/out",
                "--annotations",
                "ann.yml",
                "--dry-run",
            ]
        )
        self.assertEqual(args.output_root, Path("/tmp/out"))
        self.assertEqual(args.annotations, Path("ann.yml"))
        self.assertTrue(args.dry_run)

    def test_replay_episode_keeps_processed_flag(self):
        module = _load_example_or_skip(self, "replay_episode")
        args = module._parse_args(["episodes/task/episode_x", "--processed"])
        self.assertTrue(args.processed)

    def test_collect_teleop_no_record_configuration_reaches_session(self):
        from unittest.mock import patch
        module = _load_example_or_skip(self, "collect_teleop")
        with patch.object(module, "run_teleop_experiment", return_value=0) as session:
            self.assertEqual(module.main(["--no-record"]), 0)
        self.assertFalse(session.call_args.args[0].policy.recording_enabled)

    def test_collect_teleop_no_hand_with_no_record_reaches_session(self):
        from unittest.mock import patch
        module = _load_example_or_skip(self, "collect_teleop")
        with patch.object(module, "run_teleop_experiment", return_value=0) as session:
            self.assertEqual(module.main(["--no-hand", "--no-record"]), 0)
        self.assertFalse(session.call_args.args[0].policy.hand_enabled)
        self.assertFalse(session.call_args.args[0].policy.recording_enabled)

    def test_keyboard_configuration_and_result_reach_session(self):
        from unittest.mock import patch
        module = _load_example_or_skip(self, "keyboard_teleop")
        with patch.object(module, "run_keyboard_experiment", return_value=1) as session:
            self.assertEqual(module.main(["--no-hand"]), 1)
        self.assertTrue(session.call_args.kwargs["no_hand"])
        self.assertFalse(session.call_args.args[0].policy.hand_enabled)

    def test_calibration_physical_assertion_reaches_session(self):
        from unittest.mock import patch
        module = _load_example_or_skip(self, "calibrate_camera")
        with patch.object(module, "run_camera_calibration", return_value=0) as session:
            self.assertEqual(module.main(["--hand-geometry", "absent"]), 0)
        self.assertEqual(session.call_args.kwargs["hand_geometry"], "absent")

    def test_single_episode_processing_dry_run_is_read_only(self):
        import hashlib
        import subprocess
        from tests.raw_episode_fixture import build_raw_episode
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = build_raw_episode(root / "task" / "episode_fixture")
            source = episode.parent if episode.is_file() else episode
            def hashes():
                return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in source.rglob("*") if p.is_file()}
            before = hashes()
            output = root / "processed" / "provenance_fixture"
            result = subprocess.run([sys.executable, str(_EXAMPLES / "process_episodes.py"),
                str(source), "--dry-run", "--output-root", str(output)],
                capture_output=True, text=True, cwd=_REPO_ROOT, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 accepted", result.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(hashes(), before)


if __name__ == "__main__":
    unittest.main()
