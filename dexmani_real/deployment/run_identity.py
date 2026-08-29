"""Pure-stdlib run identity and canonical deployment receipt rendering."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dexmani_real.deployment.artifact import ResolvedPolicyArtifact
from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    ResolvedPolicyRuntimeConfig,
)


@dataclass(frozen=True)
class RealSourceIdentity:
    """Best-effort source provenance; an unavailable value is explicit."""

    availability: str
    commit: str | None
    dirty: str
    python_tree_sha256: str | None


def _canonical_python_tree_sha256(source_root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(source_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not paths:
        raise ValueError("DexMani Real source tree has no Python files")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("DexMani Real Python source must be regular files")
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def resolve_real_source_identity() -> RealSourceIdentity:
    """Record Real git/source identity without importing runtime or hardware code."""
    repository_root = Path(__file__).resolve().parents[2]
    try:
        source_sha256 = _canonical_python_tree_sha256(repository_root / "dexmani_real")
        head = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if head.returncode != 0 or len(head.stdout.strip()) != 40:
            raise ValueError("DexMani Real git commit is unavailable")
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        dirty = (
            "unknown"
            if status.returncode != 0
            else ("true" if status.stdout.strip() else "false")
        )
        return RealSourceIdentity(
            availability="available",
            commit=head.stdout.strip(),
            dirty=dirty,
            python_tree_sha256=source_sha256,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return RealSourceIdentity(
            availability="unavailable",
            commit=None,
            dirty="unknown",
            python_tree_sha256=None,
        )


def canonical_run_receipt_json(
    *,
    artifact: ResolvedPolicyArtifact,
    projection: ResolvedPolicyRuntimeConfig,
    runtime_sha256: str,
    real_source: RealSourceIdentity,
    preflight_result: Any | None,
) -> str:
    """Return the shared print/preflight receipt schema as canonical JSON."""
    verified = preflight_result is not None
    package = None
    actual_checkpoint_sha256 = None
    preflight = {"state": "not_run", "action_steps": None, "action_dim": None}
    if preflight_result is not None:
        actual_checkpoint_sha256 = preflight_result.checkpoint_sha256
        package = {
            "commit": preflight_result.package_commit,
            "dirty": preflight_result.package_dirty,
            "origin": preflight_result.package_origin,
            "python_tree_sha256": preflight_result.package_source_tree_sha256,
            "version": preflight_result.package_version,
        }
        preflight = {
            "state": "passed",
            "action_steps": preflight_result.action_steps,
            "action_dim": preflight_result.action_dim,
        }
    allocation = artifact.allocation_contract
    value = {
        "schema_version": 1,
        "artifact": {
            "selector": artifact.selector_name,
            "checkpoint": artifact.checkpoint_path.name,
            "checkpoint_size_bytes": artifact.checkpoint_size_bytes,
            "index_sha256": artifact.index_sha256,
            "expected_checkpoint_sha256": artifact.checkpoint_sha256_from_index,
            "actual_checkpoint_sha256": actual_checkpoint_sha256,
            "checkpoint_sha256_verified": verified,
            "embedded_contract_sha256": artifact.embedded_contract_sha256,
            "producer": {
                "repository": artifact.producer.repository,
                "commit": artifact.producer.commit,
                "metadata_provenance": artifact.producer.metadata_provenance,
            },
            "allocation": {
                "task_name": allocation.task_name,
                "action_key": allocation.action_key,
                "action_dim": allocation.action_dim,
                "n_obs_steps": allocation.n_obs_steps,
                "n_action_steps": allocation.n_action_steps,
                "horizon": allocation.horizon,
                "required_action_steps": allocation.required_action_steps,
                "control_dt_s": allocation.control_dt_s,
                "requires_hand": allocation.requires_hand,
                "point_cloud_num_points": allocation.point_cloud_num_points,
                "point_cloud_feature_dim": allocation.point_cloud_feature_dim,
            },
        },
        "runtime": {
            "runtime_config_sha256": runtime_sha256,
            "projection_sha256": projection.sha256,
            "fixed_runtime_target": FIXED_POLICY_RUNTIME_TARGET,
            "device": projection.runtime.device,
            "inference_seed": projection.runtime.inference_seed,
            "execution_mode": projection.runtime.execution_mode,
            "hand_acknowledged": projection.runtime.hand_acknowledged,
            "h4_execute_bounds": (
                None
                if projection.runtime.h4_execute_bounds is None
                else {
                    "max_published_endpoints": (
                        projection.runtime.h4_execute_bounds.max_published_endpoints
                    ),
                    "acknowledgement_timeout_s": (
                        projection.runtime.h4_execute_bounds.acknowledgement_timeout_s
                    ),
                    "max_running_s": projection.runtime.h4_execute_bounds.max_running_s,
                }
            ),
            "task_execute_bounds": (
                None
                if projection.runtime.task_execute_bounds is None
                else {
                    "max_published_endpoints": (
                        projection.runtime.task_execute_bounds.max_published_endpoints
                    ),
                    "acknowledgement_timeout_s": (
                        projection.runtime.task_execute_bounds.acknowledgement_timeout_s
                    ),
                    "max_running_s": (
                        projection.runtime.task_execute_bounds.max_running_s
                    ),
                }
            ),
        },
        "real_source": asdict(real_source),
        "policy_package": package,
        "preflight": preflight,
    }
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
