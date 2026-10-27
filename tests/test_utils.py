import json
import os
import tempfile
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import patch

from yaml.representer import RepresenterError

from dexmani_real.utils import io, timing


class AtomicIOTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_failed_serialization_preserves_previous_file_and_removes_temporary(self):
        for writer, suffix, error in ((io.write_json, ".json", TypeError),
                                      (io.write_yaml, ".yaml", RepresenterError)):
            with self.subTest(suffix=suffix):
                target = self.root / ("record" + suffix)
                target.write_text("previous")
                with self.assertRaises(error):
                    writer({"invalid": object()}, target)
                self.assertEqual(target.read_text(), "previous")
                self.assertEqual(list(self.root.glob(".*.tmp-*")), [])

    def test_cancelled_or_nonfinite_json_never_replaces_previous_record(self):
        target = self.root / "record.json"
        io.write_json({"value": 1}, target)
        with self.assertRaises(CancelledError):
            io.write_json({"value": 2}, target, cancelled=lambda: True)
        with self.assertRaises(ValueError):
            io.write_json({"value": float("nan")}, target)
        self.assertEqual(json.loads(target.read_text()), {"value": 1})
        self.assertEqual(list(self.root.glob(".*.tmp-*")), [])

    def test_publication_never_replaces_file_directory_or_dangling_symlink(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                source, target = self.root / (kind + ".tmp"), self.root / kind
                source.mkdir()
                if kind == "directory":
                    target.mkdir()
                elif kind == "symlink":
                    target.symlink_to(self.root / "missing")
                else:
                    target.write_text("previous")
                self.assertTrue(os.path.lexists(target))
                with self.assertRaises(FileExistsError):
                    io.publish(source, target)
                self.assertTrue(source.is_dir())
                self.assertTrue(os.path.lexists(target))

    def test_json_no_clobber_and_unicode(self):
        target = self.root / "nested" / "record.json"
        io.write_json({"实验": "成功"}, target, overwrite=False)
        self.assertIn("成功", target.read_text())
        with self.assertRaises(FileExistsError):
            io.write_json({"value": 2}, target, overwrite=False)
        self.assertEqual(json.loads(target.read_text()), {"实验": "成功"})
        self.assertEqual(list(target.parent.glob(".*.tmp-*")), [])

    def test_replace_failure_preserves_file_and_cleans_temporary(self):
        target = self.root / "record.json"
        target.write_text("previous")
        with patch.object(io.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                io.write_json({"value": 2}, target)
        self.assertEqual(target.read_text(), "previous")
        self.assertEqual(list(self.root.glob(".*.tmp-*")), [])


class TimingTests(unittest.TestCase):
    def test_wait_keeps_grid_for_small_overruns_and_reanchors_after_missed_period(self):
        now = [0.0]

        def sleep(duration):
            now[0] += duration

        with (patch.object(timing.time, "perf_counter", side_effect=lambda: now[0]),
              patch.object(timing.time, "sleep", side_effect=sleep)):
            rate = timing.Rate(10)
            rate.wait()
            self.assertAlmostEqual(now[0], 0.1)
            now[0] = 0.21
            rate.wait()
            self.assertAlmostEqual(rate.next_deadline, 0.3)
            now[0] = 0.45
            rate.wait()
            self.assertAlmostEqual(rate.next_deadline, 0.55)
            rate.wait()
            self.assertAlmostEqual(now[0], 0.55)
            now[0] = 3.0
            rate.reset()
            self.assertAlmostEqual(rate.next_deadline, 3.1)

    def test_sleep_overshoot_reanchors_without_catchup(self):
        now = [0.0]

        def sleep(duration):
            now[0] += duration + .15

        with (patch.object(timing.time, "perf_counter", side_effect=lambda: now[0]),
              patch.object(timing.time, "sleep", side_effect=sleep) as sleeper):
            rate = timing.Rate(10)
            rate.wait()
            sleeper.assert_called_once()
            self.assertAlmostEqual(rate.next_deadline, .35)


if __name__ == "__main__":
    unittest.main()
