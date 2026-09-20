"""Offline interface tests for affected examples entries (task T6, EX table).

Two levels of verification, both without hardware, GUI, or batch --help runs:

* affected entries whose module imports are offline-safe are imported and
  their real argument parsers are exercised (EX01 run_policy, EX07
  process_episodes, EX08 export_policy_zarr including the new narrow
  ``--output`` guard, plus parser smoke checks for EX02/EX03/EX04/EX05);
* the export guard's protected-source rules are unit-tested directly
  (defaults unchanged, symlink escape refused, existing .zarr store refused).

Entries whose top-level imports require device/GUI stacks (EX06, EX09-EX14)
are validated by AST parse only, per the task's safe-verification rules.
"""

from __future__ import annotations

import ast
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


try:
    _export_cli = _load_example("export_policy_zarr")
    _EXPORT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment guard
    _export_cli = None
    _EXPORT_IMPORT_ERROR = exc


@unittest.skipIf(
    _EXPORT_IMPORT_ERROR is not None,
    f"export CLI dependencies unavailable: {_EXPORT_IMPORT_ERROR}",
)
class ExportOutputGuardTest(unittest.TestCase):
    """EX08: narrow --output, default unchanged, protected sources refused."""

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
    except Exception as exc:  # pragma: no cover - environment guard
        testcase.skipTest(f"{name} dependencies unavailable: {exc}")


class AffectedParserSmokeTest(unittest.TestCase):
    """Parser-only checks for affected entries (never runs their bodies).

    ``main(["--help"])`` exits inside ``parse_args`` before any config,
    device, or session code runs, so it is a parser-safety check, not an
    execution of the entry.
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

    def test_help_exits_in_parser_for_session_entries(self):
        for name, flags in (
            ("collect_teleop", ["--no-hand", "--no-record"]),
            ("keyboard_teleop", []),
            ("calibrate_camera", []),
        ):
            with self.subTest(example=name):
                module = _load_example_or_skip(self, name)
                with self.assertRaises(SystemExit) as ctx:
                    module.main(["--help"])
                self.assertEqual(ctx.exception.code, 0)
                # The documented flags still exist in the real parser.
                if flags:
                    source = (_EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
                    for flag in flags:
                        self.assertIn(f'"{flag}"', source)


class PolicyRolloutViewerTest(unittest.TestCase):
    """EX11: a sync-raw rollout without a trace sidecar is not corruption."""

    def test_missing_trace_reports_raw_basics_and_redirects(self):
        import contextlib
        import io
        import tempfile

        try:
            from tests.raw_episode_fixture import build_raw_episode
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"raw fixture unavailable: {exc}")
        module = _load_example_or_skip(self, "visualize_policy_rollout")
        with tempfile.TemporaryDirectory() as tmp:
            episode = build_raw_episode(Path(tmp) / "episode_trace_less")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = module.main([str(episode), "--info"])
            text = stdout.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("synchronous-raw rollout", text)
        self.assertIn("not corrupt", text)
        self.assertIn("visualize_episode.py", text)

    def test_present_but_unreadable_trace_still_fails_loudly(self):
        """A sidecar that exists but cannot be read is never excused as raw-only."""
        import contextlib
        import io
        import tempfile

        try:
            from tests.raw_episode_fixture import build_raw_episode
        except Exception as exc:  # pragma: no cover - environment guard
            self.skipTest(f"raw fixture unavailable: {exc}")
        module = _load_example_or_skip(self, "visualize_policy_rollout")
        with tempfile.TemporaryDirectory() as tmp:
            episode = build_raw_episode(Path(tmp) / "episode_bad_trace")
            (Path(tmp) / "episode_bad_trace.policy_trace.npz").write_bytes(
                b"not a trace archive"
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    module.main([str(episode), "--info"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("rollout visualization", stderr.getvalue())


class StaticEntryCheckTest(unittest.TestCase):
    """AST-parse entries whose imports need device/GUI stacks (no execution)."""

    def test_device_and_gui_entries_parse(self):
        for name in (
            "calibrate_vr_heading",
            "visualize_episode",
            "visualize_episode_processed",
            "visualize_policy_rollout",
            "pointcloud_process_example",
            "realsense_record_example",
            "xhand_control_example",
        ):
            with self.subTest(example=name):
                source = (_EXAMPLES / f"{name}.py").read_text(encoding="utf-8")
                ast.parse(source)
                # No references to the removed lease/timing-gate APIs.
                for stale in (
                    "max_input_age_s",
                    "max_grid_lag_s",
                    "valid_until_monotonic_ns",
                    "CoupledCommandTicket",
                    "command_feedback_is_fresh",
                ):
                    self.assertNotIn(stale, source, f"{name} references {stale}")


if __name__ == "__main__":
    unittest.main()
