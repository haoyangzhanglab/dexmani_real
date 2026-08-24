from __future__ import annotations

import importlib
import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_primary_teleop_session_imports_without_side_effects(self) -> None:
        module = importlib.import_module("dexmani_real.teleop.session")
        self.assertTrue(callable(module.run_teleop_experiment))

    def test_recording_public_api_is_preserved(self) -> None:
        recording = importlib.import_module("dexmani_real.recording")
        for name in (
            "EpisodeReader",
            "EpisodeRecorder",
            "EpisodeTiming",
            "MergedH5File",
            "StopResult",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(recording, name))


if __name__ == "__main__":
    unittest.main()
