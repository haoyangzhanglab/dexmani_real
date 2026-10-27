import json
import tempfile
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import patch

from dexmani_real.utils import atomic_io, timing


class AtomicIOTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_failed_serialization_preserves_previous_file_and_removes_temporary(self):
        for writer, suffix in ((atomic_io.atomic_json_dump, ".json"),
                               (atomic_io.atomic_yaml_dump, ".yaml")):
            with self.subTest(suffix=suffix):
                target = self.root / ("record" + suffix)
                target.write_text("previous")
                with self.assertRaises(Exception):
                    writer({"invalid": object()}, target)
                self.assertEqual(target.read_text(), "previous")
                self.assertEqual(list(self.root.glob(".*.tmp-*")), [])

    def test_cancelled_or_nonfinite_json_never_replaces_previous_record(self):
        target = self.root / "record.json"
        atomic_io.atomic_json_dump({"value": 1}, target)
        with self.assertRaises(CancelledError):
            atomic_io.atomic_json_dump({"value": 2}, target, cancelled=lambda: True)
        with self.assertRaises(ValueError):
            atomic_io.atomic_json_dump({"value": float("nan")}, target)
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
                self.assertTrue(atomic_io.target_is_occupied(target))
                with self.assertRaises(FileExistsError):
                    atomic_io.atomic_publish(source, target)
                self.assertTrue(source.is_dir())
                self.assertTrue(atomic_io.target_is_occupied(target))


class TimingTests(unittest.TestCase):
    def test_wait_keeps_grid_for_small_overruns_and_reanchors_after_missed_period(self):
        now = [0.0]

        def sleep(duration):
            now[0] += duration

        with (patch.object(timing.time, "perf_counter", side_effect=lambda: now[0]),
              patch.object(timing.time, "sleep", side_effect=sleep)):
            rate = timing.LoopRate(10, label="test")
            rate.wait()
            self.assertAlmostEqual(now[0], 0.1)
            now[0] = 0.21
            with self.assertLogs(timing.logger, level="WARNING"):
                rate.wait()
            self.assertAlmostEqual(rate.next_deadline, 0.3)
            now[0] = 0.45
            rate.wait()
            self.assertAlmostEqual(rate.next_deadline, 0.55)
            self.assertEqual(rate.missed_slot_count, 1)
            rate.wait()
            self.assertAlmostEqual(now[0], 0.55)
            now[0] = 3.0
            rate.reset()
            self.assertAlmostEqual(rate.next_deadline, 3.1)


if __name__ == "__main__":
    unittest.main()
