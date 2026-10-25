"""Small evaluation summaries; no recorder, model, or device ownership."""

import time
from datetime import datetime, timezone
from pathlib import Path

from dexmani_real.utils.atomic_io import atomic_json_dump


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error_detail(stage, error):
    return dict(stage=stage, exception_type=type(error).__name__, message=str(error))


class SessionResults:
    """Create an incomplete session summary before motion; update it after revocation."""

    def __init__(self, directory, kind):
        self.path = Path(directory) / "session_result.json"
        self.session = dict(
            kind=kind,
            started_utc=utc_now(),
            ended_utc=None,
            outcome="incomplete",
            reason=None,
            shutdown_clean=None,
            episodes=[],
            errors=[],
        )
        self.current = None
        self.running_ns = None
        atomic_json_dump(self.session, self.path, overwrite=False)

    def begin_episode(self):
        if self.current is not None:
            raise RuntimeError("previous evaluation episode has not ended")
        self.current = dict(
            index=len(self.session["episodes"]) + 1,
            started=False,
            run_id=None,
            started_utc=None,
            duration_s=0.0,
            termination_reason=None,
            termination_details=None,
            errors=[],
            raw=None,
            metrics=dict(
                inference_count=0,
                inference_mean_s=None,
                inference_max_s=None,
                skipped_slots=0,
            ),
        )

    def entered(self, run_id):
        if self.current is not None:
            self.current.update(started=True, run_id=int(run_id), started_utc=utc_now())
            self.running_ns = time.monotonic_ns()

    def record_duration(self, name, duration_ns):
        if self.current is None or not self.current["started"]:
            return
        metrics = self.current["metrics"]
        duration = duration_ns / 1e9
        count = metrics.get(f"{name}_count", 0)
        mean = metrics.get(f"{name}_mean_s") or 0.0
        metrics[f"{name}_count"] = count + 1
        metrics[f"{name}_mean_s"] = mean + (duration - mean) / (count + 1)
        metrics[f"{name}_max_s"] = max(metrics.get(f"{name}_max_s") or 0.0, duration)

    def record_inference(self, result):
        self.record_duration("prediction_total", result.completed_ns - result.started_ns)
        build_end = result.completed_ns if result.model_started_ns is None else result.model_started_ns
        self.record_duration("observation_build", build_end - result.started_ns)
        if result.model_started_ns is not None:
            self.record_duration("inference", result.completed_ns - result.model_started_ns)

    def record_skipped_slots(self, count):
        if self.current is not None:
            self.current["metrics"]["skipped_slots"] += int(count)

    def record_errors(self, errors):
        self.session["errors"].extend(errors)

    def record_home_result(self, *, ok, interrupted, reason):
        self.session.setdefault("home_results", []).append(
            dict(ok=ok, interrupted=interrupted, reason=reason)
        )

    def finish_episode(
        self, reason, *, details=None, errors=(), raw=None, duration_s=None, **metrics
    ):
        if self.current is None:
            return
        if duration_s is None:
            duration_s = (
                (time.monotonic_ns() - self.running_ns) / 1e9
                if self.running_ns is not None
                else 0.0
            )
        self.current.update(
            duration_s=duration_s,
            termination_reason=reason,
            termination_details=details,
            errors=list(errors),
            raw=raw,
        )
        self.current["metrics"].update(metrics)
        self.session["episodes"].append(self.current)
        self.current = self.running_ns = None
        atomic_json_dump(self.session, self.path)

    def finish_session(self, *, outcome, reason, shutdown_clean, **details):
        self.session.update(
            ended_utc=utc_now(),
            outcome=outcome,
            reason=reason,
            shutdown_clean=bool(shutdown_clean),
            **details,
        )
        atomic_json_dump(self.session, self.path)
