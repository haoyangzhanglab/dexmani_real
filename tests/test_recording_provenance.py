"""Offline regressions for minimal raw-episode experiment provenance."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.recording.recorder import EpisodeRecorder
from dexmani_real.teleop import session as teleop_session


class RecordingProvenanceTest(unittest.TestCase):
    _RESOURCE_PROVENANCE = (
        ("arm_hand_urdf_sha256", "a" * 64),
        ("camera_calibration_sha256", "b" * 64),
    )

    @staticmethod
    def _git_result(args: list[str], stdout: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    def test_startup_provenance_records_canonical_config_and_clean_git_state(
        self,
    ) -> None:
        runtime = resolve_experiment_config()
        with (
            patch.object(
                teleop_session,
                "_resource_provenance",
                return_value=self._RESOURCE_PROVENANCE,
            ),
            patch(
                "subprocess.run",
                side_effect=(
                    self._git_result(["git", "rev-parse", "HEAD"], "c" * 40 + "\n"),
                    self._git_result(["git", "status", "--porcelain"], ""),
                ),
            ) as git_run,
        ):
            provenance = dict(
                teleop_session._recording_provenance(runtime, Path("/repo"))
            )

        self.assertEqual(provenance["resolved_config_json"], runtime.canonical_json)
        self.assertEqual(
            hashlib.sha256(
                provenance["resolved_config_json"].encode("utf-8")
            ).hexdigest(),
            runtime.sha256,
        )
        self.assertEqual(provenance["dexmani_real_git_commit"], "c" * 40)
        self.assertEqual(provenance["dexmani_real_git_dirty"], "0")
        self.assertEqual(
            {name: provenance[name] for name, _value in self._RESOURCE_PROVENANCE},
            dict(self._RESOURCE_PROVENANCE),
        )
        self.assertEqual(git_run.call_count, 2)
        self.assertEqual(
            git_run.call_args_list[0].args[0], ["git", "rev-parse", "HEAD"]
        )
        self.assertEqual(
            git_run.call_args_list[1].args[0], ["git", "status", "--porcelain"]
        )

    def test_startup_provenance_marks_dirty_git_state(self) -> None:
        runtime = resolve_experiment_config()
        with (
            patch.object(
                teleop_session,
                "_resource_provenance",
                return_value=self._RESOURCE_PROVENANCE,
            ),
            patch(
                "subprocess.run",
                side_effect=(
                    self._git_result(["git", "rev-parse", "HEAD"], "d" * 40 + "\n"),
                    self._git_result(
                        ["git", "status", "--porcelain"],
                        " M dexmani_real/teleop/session.py\n",
                    ),
                ),
            ),
        ):
            provenance = dict(
                teleop_session._recording_provenance(runtime, Path("/repo"))
            )

        self.assertEqual(provenance["dexmani_real_git_dirty"], "1")

    def test_startup_provenance_rejects_invalid_config_hash_and_git_commit(
        self,
    ) -> None:
        invalid_runtime = SimpleNamespace(canonical_json="{}", sha256="a" * 64)
        with patch("subprocess.run") as git_run:
            with self.assertRaisesRegex(ValueError, "canonical config SHA-256"):
                teleop_session._recording_provenance(invalid_runtime, Path("/repo"))
        git_run.assert_not_called()

        runtime = resolve_experiment_config()
        with (
            patch.object(
                teleop_session,
                "_resource_provenance",
                return_value=self._RESOURCE_PROVENANCE,
            ),
            patch(
                "subprocess.run",
                return_value=self._git_result(
                    ["git", "rev-parse", "HEAD"], "C" * 40 + "\n"
                ),
            ) as git_run,
            self.assertRaisesRegex(ValueError, "40-character lowercase Git SHA"),
        ):
            teleop_session._recording_provenance(runtime, Path("/repo"))
        git_run.assert_called_once()

    def test_teleop_snapshots_provenance_once_before_channel_creation(self) -> None:
        runtime = resolve_experiment_config()
        events: list[str] = []

        def snapshot_provenance(
            received_runtime: object,
            _repo_root: Path,
        ) -> tuple[tuple[str, str], ...]:
            self.assertIs(received_runtime, runtime)
            events.append("provenance")
            return self._RESOURCE_PROVENANCE

        def reject_channel_creation(*_args: object, **_kwargs: object) -> object:
            events.append("channels")
            raise RuntimeError("stop after startup provenance")

        with (
            patch("dexmani_real.teleop.session.load_vr_transform"),
            patch(
                "dexmani_real.teleop.session._recording_provenance",
                side_effect=snapshot_provenance,
            ) as provenance,
            patch(
                "dexmani_real.teleop.session._print_session_header",
            ),
            patch(
                "dexmani_real.teleop.session.RuntimeChannels.create",
                side_effect=reject_channel_creation,
            ),
            self.assertRaisesRegex(RuntimeError, "stop after startup provenance"),
        ):
            teleop_session.run_teleop_experiment(runtime)

        provenance.assert_called_once()
        self.assertEqual(events, ["provenance", "channels"])

    def test_recorder_persists_experiment_and_resource_provenance(self) -> None:
        runtime = resolve_experiment_config()
        provenance = {
            "resolved_config_json": runtime.canonical_json,
            "dexmani_real_git_commit": "e" * 40,
            "dexmani_real_git_dirty": "1",
            **dict(self._RESOURCE_PROVENANCE),
        }
        with tempfile.TemporaryDirectory() as directory:
            recorder = EpisodeRecorder(
                data_dir=directory,
                resolved_config_hash=runtime.sha256,
                provenance=provenance,
            )
            recorder._pending_meta = {"camera_metadata": {}, "skip_initial_frames": 0}
            h5_path = Path(directory) / "data.h5"
            with h5py.File(h5_path, "w") as output:
                recorder._write_meta_attrs(output.create_group("meta"))
            with h5py.File(h5_path, "r") as output:
                attrs = output["meta"].attrs
                self.assertEqual(
                    attrs["provenance_resolved_config_json"], runtime.canonical_json
                )
                self.assertEqual(attrs["provenance_dexmani_real_git_commit"], "e" * 40)
                self.assertEqual(attrs["provenance_dexmani_real_git_dirty"], "1")
                self.assertEqual(
                    attrs["provenance_arm_hand_urdf_sha256"],
                    dict(self._RESOURCE_PROVENANCE)["arm_hand_urdf_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
