"""Small evaluation summaries; no recorder, model, or device ownership."""

from datetime import datetime, timezone
from pathlib import Path
import time

from dexmani_real.utils.atomic_io import atomic_json_dump


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error_detail(stage, error):
    return dict(stage=stage, exception_type=type(error).__name__, message=str(error))


class EpisodeResults:
    """Create an incomplete session summary before motion; update it after revocation."""

    def __init__(self, directory, kind):
        self.path = Path(directory) / "session_result.json"
        self.session = dict(kind=kind, started_utc=utc_now(), ended_utc=None,
                            outcome="incomplete", reason=None, shutdown_clean=None,
                            episodes=[], errors=[])
        self.current = None
        self._running_ns = None
        atomic_json_dump(self.session, self.path, overwrite=False)

    def begin_episode(self):
        if self.current is not None:
            raise RuntimeError("previous evaluation episode has not ended")
        self.current = dict(index=len(self.session["episodes"]) + 1, started=False,
                            run_id=None, started_utc=None, duration_s=0.0,
                            termination_reason=None, termination_details=None,
                            errors=[], raw=None,
                            metrics=dict(inference_count=0, inference_mean_s=None,
                                         inference_max_s=None, skipped_slots=0, dispatch_count=0))

    def entered(self, run_id):
        if self.current is not None:
            self.current.update(started=True, run_id=int(run_id), started_utc=utc_now())
            self._running_ns = time.monotonic_ns()

    def record_inference(self, duration_ns):
        if self.current is None or not self.current["started"]:
            return
        metrics = self.current["metrics"]
        duration = duration_ns / 1e9
        count = metrics["inference_count"]
        mean = metrics["inference_mean_s"] or 0.0
        metrics.update(inference_count=count + 1,
                       inference_mean_s=mean + (duration - mean) / (count + 1),
                       inference_max_s=max(metrics["inference_max_s"] or 0.0, duration))

    def finish_episode(self, reason, *, details=None, errors=(), raw=None, duration_s=None, **metrics):
        if self.current is None:
            return
        if duration_s is None:
            duration_s = ((time.monotonic_ns() - self._running_ns) / 1e9
                          if self._running_ns is not None else 0.0)
        self.current.update(duration_s=duration_s, termination_reason=reason,
                            termination_details=details, errors=list(errors), raw=raw)
        self.current["metrics"].update(metrics)
        self.session["episodes"].append(self.current)
        self.current = self._running_ns = None
        atomic_json_dump(self.session, self.path)

    def finish_session(self, *, outcome, reason, shutdown_clean, **details):
        self.session.update(ended_utc=utc_now(), outcome=outcome, reason=reason,
                            shutdown_clean=bool(shutdown_clean), **details)
        atomic_json_dump(self.session, self.path)
