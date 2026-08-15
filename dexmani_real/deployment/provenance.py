"""Deployment provenance logging (execution doc §96).

Provenance is a one-time startup log line, never a shared-memory payload (§96:
do not put the full resolved config into high-frequency IPC). The lifecycle logs
it once, before any worker spawns, with everything it can resolve locally —
deployment targets, checkpoint path, and the resolved runtime SHA-256. Commit
hashes are optional (the CLI or CI can supply them); absence logs ``unknown``
rather than fabricating a value.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dexmani_real.deployment.config import DeploymentConfig


def log_deployment_provenance(
    logger: logging.Logger,
    *,
    deployment: "DeploymentConfig",
    runtime_sha256: str,
    dexmani_commit: str = "",
    model_commit: str = "",
    checkpoint_sha256: str = "",
    model_config_sha256: str = "",
) -> None:
    """Log one structured provenance line (no SharedStorage write)."""
    logger.info(
        "deployment provenance: dexmani_commit=%s model_commit=%s "
        "backend_target=%s observation_adapter_target=%s action_adapter_target=%s "
        "checkpoint=%s checkpoint_sha256=%s model_config_sha256=%s runtime_sha256=%s",
        dexmani_commit or "unknown",
        model_commit or "unknown",
        deployment.backend_target,
        deployment.observation_adapter_target,
        deployment.action_adapter_target,
        deployment.checkpoint or "",
        checkpoint_sha256 or "",
        model_config_sha256 or "",
        runtime_sha256,
    )
