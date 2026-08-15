"""P13: deployment metrics + provenance (execution doc §94/§96).

Locks the counter/gauge registry semantics and the provenance logging boundary:

  - ``Metrics`` increments counters, observes gauges, merges both into a
    snapshot, and ``flush`` resets counters but keeps gauges
  - ``flush_every`` throttles on a monotonic-ns boundary
  - neither module imports Prometheus/OpenTelemetry/torch (§94), and provenance
    never touches SharedStorage (§96)
  - ``log_deployment_provenance`` logs every §96 field (targets, checkpoint,
    runtime SHA-256) through the passed logger, nothing else
"""

from __future__ import annotations

import sys
import time

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.metrics import (
    COMMAND_SILENCE_ABORT,
    ENDPOINTS_PUBLISHED,
    INFERENCE_MS,
    Metrics,
    flush_every,
)
from dexmani_real.deployment.provenance import log_deployment_provenance


class _Capture:
    def __init__(self) -> None:
        self.records: list[str] = []

    def info(self, msg: str, *args) -> None:
        self.records.append(msg % args)


def main() -> int:
    # ── architecture: no Prometheus / OpenTelemetry / torch ──
    import dexmani_real.deployment.metrics  # noqa: F401
    import dexmani_real.deployment.provenance  # noqa: F401
    for banned in ("prometheus_client", "opentelemetry", "torch", "dexmani_real.shm"):
        assert banned not in sys.modules, f"{banned} must not be imported by metrics/provenance"

    # ── Metrics: increment / observe / snapshot ──
    m = Metrics()
    assert m.snapshot() == {}
    m.increment(ENDPOINTS_PUBLISHED)
    m.increment(ENDPOINTS_PUBLISHED, 2)
    m.increment(COMMAND_SILENCE_ABORT)
    m.observe(INFERENCE_MS, 3.5)
    snap = m.snapshot()
    assert snap[ENDPOINTS_PUBLISHED] == 3
    assert snap[COMMAND_SILENCE_ABORT] == 1
    assert snap[INFERENCE_MS] == 3.5

    # ── flush: resets counters, keeps gauges ──
    m.flush()
    snap = m.snapshot()
    assert snap[ENDPOINTS_PUBLISHED] == 0, "flush must reset counters"
    assert snap[COMMAND_SILENCE_ABORT] == 0
    assert snap[INFERENCE_MS] == 3.5, "flush must keep gauges"

    # ── flush_every throttling ──
    old = time.monotonic_ns() - 2_000_000_000  # 2 s ago
    new = flush_every(m, last_ns=old, interval_s=1.0)
    assert new != old, "elapsed interval must flush and advance the timestamp"
    now = time.monotonic_ns()
    assert flush_every(m, last_ns=now, interval_s=1.0) == now, "fresh timestamp must not flush"

    # ── provenance: logs every §96 field, no SharedStorage ──
    capture = _Capture()
    deployment = DeploymentConfig(
        backend_target="dexmani_real.deployment.fake:FakePolicyBackend",
        observation_adapter_target="dexmani_real.deployment.fake:FakeObservationAdapter",
        action_adapter_target="dexmani_real.deployment.fake:FakeActionAdapter",
        checkpoint="checkpoints/model.pt",
        model_config_path="cfg/policy.yaml",
    )
    log_deployment_provenance(
        capture,
        deployment=deployment,
        runtime_sha256="deadbeef" * 8,
        dexmani_commit="abc1234",
        checkpoint_sha256="f00d" * 16,
    )
    assert len(capture.records) == 1
    message = capture.records[0]
    for required in (
        "dexmani_commit=abc1234",
        "backend_target=dexmani_real.deployment.fake:FakePolicyBackend",
        "observation_adapter_target=dexmani_real.deployment.fake:FakeObservationAdapter",
        "action_adapter_target=dexmani_real.deployment.fake:FakeActionAdapter",
        "checkpoint=checkpoints/model.pt",
        "checkpoint_sha256=" + "f00d" * 16,
        "runtime_sha256=" + "deadbeef" * 8,
    ):
        assert required in message, f"provenance missing {required!r}"

    # Missing hashes/commits log "unknown", never raise.
    capture2 = _Capture()
    log_deployment_provenance(capture2, deployment=DeploymentConfig(backend_target="a:b"), runtime_sha256="x" * 64)
    assert "model_commit=unknown" in capture2.records[0]
    assert "dexmani_commit=unknown" in capture2.records[0]

    print("check_deployment_metrics: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
