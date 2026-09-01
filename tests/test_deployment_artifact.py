"""Offline tests for the untrusted deployment artifact filesystem boundary."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import dexmani_real.deployment.artifact as artifact_module
from dexmani_real.deployment.artifact import (
    MAX_POLICY_ARTIFACT_INDEX_BYTES,
    resolve_policy_artifact,
)

REFERENCE_EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[2]
    / "dexmani_policy"
    / "experiments/dp3/pick_place_toy/2026-08-28_13-59_42"
)


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sidecar(checkpoint: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint": {
            "filename": checkpoint.name,
            "size_bytes": checkpoint.stat().st_size,
            "sha256": "a" * 64,
        },
        "embedded_contract_sha256": "b" * 64,
        "allocation": {
            "task_name": "pick_place_toy",
            "action_key": "action",
            "action_dim": 19,
            "n_obs_steps": 2,
            "n_action_steps": 8,
            "horizon": 16,
            "required_action_steps": 15,
            "control_dt_s": 0.0625,
            "sensor_modalities": ["joint_state", "point_cloud"],
            "observation_fields": ["arm_qpos", "hand_qpos", "point_cloud"],
            "requires_hand": True,
            "point_cloud_num_points": 1024,
            "point_cloud_feature_dim": 6,
        },
        "producer": {
            "repository": "haoyangzhanglab/dexmani_policy",
            "commit": "c" * 40,
            "metadata_provenance": "retrofitted",
        },
    }


def _write_experiment(root: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    root.mkdir()
    (root / "config.yaml").write_text("name: synthetic\n", encoding="utf-8")
    checkpoints = root / "checkpoints"
    checkpoints.mkdir()
    checkpoint = checkpoints / "epoch=0001-deployment-v1.pt"
    checkpoint.write_bytes(b"small-checkpoint")
    sidecar = _sidecar(checkpoint)
    sidecar_path = checkpoint.with_name(f"{checkpoint.name}.deployment.json")
    sidecar_path.write_bytes(_canonical_json_bytes(sidecar))
    (checkpoints / "deployment_latest.pt").symlink_to(checkpoint.name)
    return checkpoints, checkpoint, sidecar_path, sidecar


def _write_sidecar(path: Path, sidecar: dict[str, Any]) -> None:
    path.write_bytes(_canonical_json_bytes(sidecar))


class DeploymentArtifactTest(unittest.TestCase):
    def test_valid_one_hop_selector_returns_unverified_hash_and_fixed_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, sidecar_path, _ = _write_experiment(root)

            artifact = resolve_policy_artifact(root)

            self.assertEqual(artifact.experiment_dir, root.resolve())
            self.assertEqual(
                artifact.selector_path, checkpoints / "deployment_latest.pt"
            )
            self.assertEqual(artifact.checkpoint_path, checkpoint.resolve())
            self.assertEqual(artifact.sidecar_entry_path, sidecar_path)
            self.assertEqual(artifact.sidecar_path, sidecar_path.resolve())
            self.assertEqual(artifact.selector_name, "deployment_latest.pt")
            self.assertEqual(artifact.checkpoint_size_bytes, len(b"small-checkpoint"))
            self.assertEqual(artifact.checkpoint_sha256_from_index, "a" * 64)
            self.assertFalse(artifact.checkpoint_sha256_verified)
            self.assertEqual(artifact.allocation_contract.required_action_steps, 15)
            self.assertEqual(
                artifact.checkpoint_lstat_identity.inode, checkpoint.stat().st_ino
            )
            self.assertEqual(
                artifact.sidecar_lstat_identity.inode, sidecar_path.stat().st_ino
            )
            self.assertEqual(
                artifact.sidecar_entry_lstat_identity.inode,
                os.lstat(sidecar_path).st_ino,
            )
            self.assertEqual(
                artifact.experiment_directory_identity.inode, root.stat().st_ino
            )
            self.assertEqual(
                artifact.checkpoints_directory_identity.inode,
                checkpoints.stat().st_ino,
            )

    def test_checkpoint_and_sidecar_targets_reject_hard_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, sidecar_path, _ = _write_experiment(root)
            os.link(checkpoint, checkpoints / "checkpoint-hard-link.pt")
            with self.assertRaisesRegex(ValueError, "hard link"):
                resolve_policy_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, _checkpoint, sidecar_path, _ = _write_experiment(root)
            os.link(sidecar_path, checkpoints / "sidecar-hard-link.json")
            with self.assertRaisesRegex(ValueError, "hard link"):
                resolve_policy_artifact(root)

    def test_direct_regular_selector_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, _, _, _ = _write_experiment(root)
            selector = checkpoints / "deployment_latest.pt"
            selector.unlink()
            selector.write_bytes(b"direct-selector-checkpoint")
            direct_sidecar = selector.with_name(f"{selector.name}.deployment.json")
            _write_sidecar(direct_sidecar, _sidecar(selector))

            artifact = resolve_policy_artifact(root)

            self.assertEqual(artifact.checkpoint_path, selector)
            self.assertEqual(artifact.selector_path, selector)

    @unittest.skipUnless(
        REFERENCE_EXPERIMENT_DIR.is_dir(), "reference policy experiment is not present"
    )
    def test_current_reference_sidecar_resolves_without_checkpoint_hashing(
        self,
    ) -> None:
        artifact = resolve_policy_artifact(REFERENCE_EXPERIMENT_DIR)

        self.assertEqual(artifact.selector_name, "deployment_latest.pt")
        self.assertEqual(
            artifact.checkpoint_path.name,
            "epoch=1126-step=00080000-pr3-fc6b7df-deployment-v2.pt",
        )
        self.assertEqual(artifact.checkpoint_size_bytes, 550_226_410)
        self.assertEqual(
            artifact.index_sha256,
            "52683587a024a18d9251eb073a6290c1cc123d966edbaa7dc282097c38040b06",
        )
        self.assertFalse(artifact.checkpoint_sha256_verified)

    def test_latest_is_used_only_when_deployment_selector_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, _, _ = _write_experiment(root)
            (checkpoints / "deployment_latest.pt").unlink()
            (checkpoints / "latest.pt").symlink_to(checkpoint.name)

            artifact = resolve_policy_artifact(root)

            self.assertEqual(artifact.selector_name, "latest.pt")

    def test_missing_selector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, _, _, _ = _write_experiment(root)
            (checkpoints / "deployment_latest.pt").unlink()

            with self.assertRaisesRegex(ValueError, "no deployment selector"):
                resolve_policy_artifact(root)

    def test_dangling_deployment_selector_does_not_fall_back_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, _, _ = _write_experiment(root)
            selector = checkpoints / "deployment_latest.pt"
            selector.unlink()
            selector.symlink_to("missing-deployment.pt")
            (checkpoints / "latest.pt").symlink_to(checkpoint.name)

            with self.assertRaisesRegex(ValueError, "dangling"):
                resolve_policy_artifact(root)

    def test_invalid_selector_targets_are_rejected(self) -> None:
        cases: list[tuple[str, Callable[[Path, Path], None]]] = [
            (
                "relative escape",
                lambda root, selector: selector.symlink_to("../../outside.pt"),
            ),
            (
                "absolute target",
                lambda root, selector: selector.symlink_to(root.parent / "outside.pt"),
            ),
            ("directory", lambda root, selector: selector.mkdir()),
            (
                "empty checkpoint",
                lambda root, selector: (root / "checkpoints" / "empty.pt").write_bytes(
                    b""
                ),
            ),
            (
                "non checkpoint suffix",
                lambda root, selector: (root / "checkpoints" / "wrong.bin").write_bytes(
                    b"x"
                ),
            ),
            (
                "multi hop",
                lambda root, selector: _make_multi_hop_selector(root, selector),
            ),
            ("loop", lambda root, selector: selector.symlink_to(selector.name)),
        ]
        for label, setup in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                checkpoints, _, _, _ = _write_experiment(root)
                selector = checkpoints / "deployment_latest.pt"
                selector.unlink()
                if label in {"relative escape", "absolute target"}:
                    (root.parent / "outside.pt").write_bytes(b"outside")
                setup(root, selector)
                if label == "empty checkpoint":
                    selector.symlink_to("empty.pt")
                if label == "non checkpoint suffix":
                    selector.symlink_to("wrong.bin")

                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_sidecar_one_hop_basename_is_accepted_and_extra_hops_are_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, sidecar_path, _ = _write_experiment(root)
            payload_path = checkpoints / "sidecar-payload.json"
            sidecar_path.rename(payload_path)
            sidecar_path.symlink_to(payload_path.name)

            artifact = resolve_policy_artifact(root)

            self.assertEqual(artifact.sidecar_path, payload_path)

        cases = ("multi hop", "loop", "absolute")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                checkpoints, _, sidecar_path, _ = _write_experiment(root)
                payload_path = checkpoints / "sidecar-payload.json"
                sidecar_path.rename(payload_path)
                if label == "multi hop":
                    hop_path = checkpoints / "sidecar-hop.json"
                    hop_path.symlink_to(payload_path.name)
                    sidecar_path.symlink_to(hop_path.name)
                elif label == "loop":
                    sidecar_path.symlink_to(sidecar_path.name)
                else:
                    sidecar_path.symlink_to(payload_path)

                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_sidecar_symlink_entry_utime_race_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, _checkpoint, sidecar_path, _ = _write_experiment(root)
            payload_path = checkpoints / "sidecar-payload.json"
            sidecar_path.rename(payload_path)
            sidecar_path.symlink_to(payload_path.name)
            original_read = artifact_module._read_bounded_regular_file_at

            def touch_entry_after_read(*args: Any, **kwargs: Any) -> bytes:
                payload = original_read(*args, **kwargs)
                info = os.lstat(sidecar_path)
                os.utime(
                    sidecar_path,
                    ns=(info.st_atime_ns, info.st_mtime_ns),
                    follow_symlinks=False,
                )
                return payload

            with patch.object(
                artifact_module,
                "_read_bounded_regular_file_at",
                side_effect=touch_entry_after_read,
            ):
                with self.assertRaisesRegex(ValueError, "sidecar entry"):
                    resolve_policy_artifact(root)

    def test_directory_replacement_after_held_fd_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "experiment"
            _write_experiment(root)
            original_display = artifact_module._display_directory_path
            moved_root = base / "experiment-held"

            def replace_root_after_open(
                source_path: Path, identity: object, label: str
            ) -> Path:
                if label == "experiment root":
                    root.rename(moved_root)
                    root.mkdir()
                return original_display(source_path, identity, label)

            with patch.object(
                artifact_module,
                "_display_directory_path",
                side_effect=replace_root_after_open,
            ):
                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, _, _, _ = _write_experiment(root)
            original_display = artifact_module._display_directory_path
            moved_checkpoints = root / "checkpoints-held"

            def replace_checkpoints_after_open(
                source_path: Path, identity: object, label: str
            ) -> Path:
                if label == "checkpoints directory":
                    checkpoints.rename(moved_checkpoints)
                    checkpoints.mkdir()
                return original_display(source_path, identity, label)

            with patch.object(
                artifact_module,
                "_display_directory_path",
                side_effect=replace_checkpoints_after_open,
            ):
                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_selector_and_regular_file_open_races_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            checkpoints, checkpoint, _, _ = _write_experiment(root)
            selector = checkpoints / "deployment_latest.pt"
            original_read = artifact_module._read_bounded_regular_file_at

            def replace_selector_before_final_check(*args: Any, **kwargs: Any) -> bytes:
                payload = original_read(*args, **kwargs)
                selector.unlink()
                selector.write_bytes(b"replacement-selector")
                return payload

            with patch.object(
                artifact_module,
                "_read_bounded_regular_file_at",
                side_effect=replace_selector_before_final_check,
            ):
                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

        for label in ("resolved checkpoint", "checkpoint sidecar"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                _, checkpoint, sidecar_path, sidecar = _write_experiment(root)
                original_open = artifact_module._open_regular_read_at

                def replace_before_open(
                    directory_fd: int,
                    filename: str,
                    identity: object,
                    opened_label: str,
                ) -> int:
                    if opened_label == label:
                        path = (
                            checkpoint
                            if label == "resolved checkpoint"
                            else sidecar_path
                        )
                        path.unlink()
                        path.write_bytes(
                            b"replacement-checkpoint"
                            if label == "resolved checkpoint"
                            else _canonical_json_bytes(sidecar)
                        )
                    return original_open(directory_fd, filename, identity, opened_label)

                with patch.object(
                    artifact_module,
                    "_open_regular_read_at",
                    side_effect=replace_before_open,
                ):
                    with self.assertRaises(ValueError):
                        resolve_policy_artifact(root)

    def test_experiment_config_and_checkpoint_directories_must_not_be_symlinks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "experiment"
            _write_experiment(root)
            config_target = base / "config-target.yaml"
            config_target.write_text("name: target\n", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.unlink()
            config_path.symlink_to(config_target)
            with self.assertRaisesRegex(ValueError, "config.yaml"):
                resolve_policy_artifact(root)

            config_path.unlink()
            config_path.write_text("name: synthetic\n", encoding="utf-8")
            checkpoints = root / "checkpoints"
            renamed = root / "checkpoints-real"
            checkpoints.rename(renamed)
            checkpoints.symlink_to(renamed, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "checkpoints directory"):
                resolve_policy_artifact(root)

    def test_sidecar_file_and_canonical_encoding_checks(self) -> None:
        cases: list[tuple[str, Callable[[Path, Path, dict[str, Any]], None]]] = [
            ("missing", lambda root, sidecar_path, sidecar: sidecar_path.unlink()),
            (
                "malformed",
                lambda root, sidecar_path, sidecar: sidecar_path.write_bytes(b"{"),
            ),
            (
                "invalid utf8",
                lambda root, sidecar_path, sidecar: sidecar_path.write_bytes(b"\xff"),
            ),
            (
                "not object",
                lambda root, sidecar_path, sidecar: sidecar_path.write_bytes(b"[]"),
            ),
            (
                "noncanonical",
                lambda root, sidecar_path, sidecar: sidecar_path.write_text(
                    json.dumps(sidecar), encoding="utf-8"
                ),
            ),
            (
                "oversized",
                lambda root, sidecar_path, sidecar: sidecar_path.write_bytes(
                    b"{" + b" " * MAX_POLICY_ARTIFACT_INDEX_BYTES
                ),
            ),
            (
                "escape symlink",
                lambda root, sidecar_path, sidecar: _replace_with_escaping_sidecar_link(
                    root, sidecar_path, sidecar
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                _, _, sidecar_path, sidecar = _write_experiment(root)
                mutate(root, sidecar_path, sidecar)

                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_sidecar_checkpoint_and_producer_contract_mismatches_are_rejected(
        self,
    ) -> None:
        cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
            (
                "checkpoint filename",
                lambda value: value["checkpoint"].update({"filename": "other.pt"}),
            ),
            (
                "checkpoint size",
                lambda value: value["checkpoint"].update({"size_bytes": 1}),
            ),
            (
                "checkpoint sha",
                lambda value: value["checkpoint"].update({"sha256": "not-a-hash"}),
            ),
            (
                "embedded sha",
                lambda value: value.update({"embedded_contract_sha256": "not-a-hash"}),
            ),
            ("sidecar key", lambda value: value.update({"unexpected": 1})),
            ("checkpoint key", lambda value: value["checkpoint"].pop("size_bytes")),
            ("allocation key", lambda value: value["allocation"].pop("task_name")),
            ("producer key", lambda value: value["producer"].update({"extra": 1})),
            ("producer type", lambda value: value["producer"].update({"commit": 1})),
            (
                "producer repository",
                lambda value: value["producer"].update(
                    {"repository": "unknown/policy"}
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                _, _, sidecar_path, sidecar = _write_experiment(root)
                changed = copy.deepcopy(sidecar)
                mutate(changed)
                _write_sidecar(sidecar_path, changed)

                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_allocation_contract_mismatches_and_capacity_are_rejected(self) -> None:
        cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
            (
                "bool integer",
                lambda value: value["allocation"].update({"n_obs_steps": True}),
            ),
            (
                "action dimension",
                lambda value: value["allocation"].update({"action_dim": 20}),
            ),
            (
                "required window",
                lambda value: value["allocation"].update({"required_action_steps": 8}),
            ),
            (
                "action window",
                lambda value: value["allocation"].update({"n_action_steps": 16}),
            ),
            (
                "capacity",
                lambda value: value["allocation"].update(
                    {"horizon": 34, "required_action_steps": 33}
                ),
            ),
            (
                "nonpositive dt",
                lambda value: value["allocation"].update({"control_dt_s": 0.0}),
            ),
            (
                "modalities",
                lambda value: value["allocation"].update(
                    {"sensor_modalities": ["point_cloud", "joint_state"]}
                ),
            ),
            (
                "observation fields",
                lambda value: value["allocation"].update(
                    {"observation_fields": ["arm_qpos", "point_cloud"]}
                ),
            ),
            (
                "requires hand",
                lambda value: value["allocation"].update({"requires_hand": False}),
            ),
            (
                "point cloud count",
                lambda value: value["allocation"].update(
                    {"point_cloud_num_points": 99}
                ),
            ),
            (
                "point cloud dimension",
                lambda value: value["allocation"].update(
                    {"point_cloud_feature_dim": 5}
                ),
            ),
        ]
        for label, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "experiment"
                _, _, sidecar_path, sidecar = _write_experiment(root)
                changed = copy.deepcopy(sidecar)
                mutate(changed)
                _write_sidecar(sidecar_path, changed)

                with self.assertRaises(ValueError):
                    resolve_policy_artifact(root)

    def test_schema_v2_accepts_rgb_and_r3d_auxiliary_prefix_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            _, _, sidecar_path, sidecar = _write_experiment(root)
            sidecar["schema_version"] = 2
            sidecar["allocation"].update(
                {
                    "action_dim": 28,
                    "control_action_dim": 19,
                    "auxiliary_action_layout": "joint19_ee9",
                    "sensor_modalities": ["joint_state", "rgb"],
                    "observation_fields": ["arm_qpos", "hand_qpos", "rgb"],
                    "point_cloud_num_points": None,
                    "point_cloud_feature_dim": None,
                    "rgb_shape": [480, 640, 3],
                    "rgb_color_order": "rgb",
                    "rgb_value_range": "uint8_0_255",
                }
            )
            _write_sidecar(sidecar_path, sidecar)

            artifact = resolve_policy_artifact(root)

            self.assertEqual(artifact.allocation_contract.action_dim, 28)
            self.assertEqual(artifact.allocation_contract.control_action_dim, 19)
            self.assertEqual(
                artifact.allocation_contract.auxiliary_action_layout, "joint19_ee9"
            )
            self.assertEqual(artifact.allocation_contract.rgb_shape, (480, 640, 3))

    def test_schema_v2_rejects_unrecognized_auxiliary_action_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            _, _, sidecar_path, sidecar = _write_experiment(root)
            sidecar["schema_version"] = 2
            sidecar["allocation"].update(
                {
                    "control_action_dim": 19,
                    "auxiliary_action_layout": "unknown",
                    "rgb_shape": None,
                    "rgb_color_order": None,
                    "rgb_value_range": None,
                }
            )
            _write_sidecar(sidecar_path, sidecar)

            with self.assertRaisesRegex(ValueError, "auxiliary_action_layout"):
                resolve_policy_artifact(root)

    def test_artifact_import_does_not_load_policy_torch_lifecycle_or_hardware(
        self,
    ) -> None:
        script = """
import sys
import dexmani_real.deployment.artifact
forbidden = (
    'torch',
    'dexmani_policy',
    'dexmani_real.deployment.lifecycle',
    'dexmani_real.deployment.worker',
    'dexmani_real.deployment.contracts',
    'dexmani_real.robot.',
    'dexmani_real.sensor.',
    'dexmani_real.runtime.',
)
loaded = sorted(
    name for name in sys.modules
    if name == forbidden[0]
    or name.startswith(forbidden[1])
    or name in forbidden[2:5]
    or name.startswith(forbidden[5])
    or name.startswith(forbidden[6])
    or name.startswith(forbidden[7])
)
if loaded:
    raise SystemExit(','.join(loaded))
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            check=False,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def _replace_with_escaping_sidecar_link(
    root: Path, sidecar_path: Path, sidecar: dict[str, Any]
) -> None:
    outside = root.parent / "outside.deployment.json"
    outside.write_bytes(_canonical_json_bytes(sidecar))
    sidecar_path.unlink()
    sidecar_path.symlink_to("../../outside.deployment.json")


def _make_multi_hop_selector(root: Path, selector: Path) -> None:
    hop_path = root / "checkpoints" / "selector-hop.pt"
    hop_path.symlink_to("epoch=0001-deployment-v1.pt")
    selector.symlink_to(hop_path.name)


if __name__ == "__main__":
    unittest.main()
