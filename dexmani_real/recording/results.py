"""Small owner-thread result records outside immutable Raw episodes."""

import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real.utils.atomic_io import atomic_json_dump, target_is_occupied
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def error_detail(stage, error):
    return dict(stage=stage, exception_type=type(error).__name__, message=str(error))


class SessionResults:
    """No active-tick I/O: prepare before final admission, finish after stop."""

    def __init__(self, directory, kind):
        self.directory = Path(directory)
        if target_is_occupied(self.directory / "session_result.json"):
            raise FileExistsError(
                "session result already exists; choose a new experiment directory"
            )
        self.failed = False
        self.session = dict(
            session_id=uuid.uuid4().hex,
            kind=kind,
            started_utc=utc_now(),
            ended_utc=None,
            state="incomplete",
            outcome=None,
            reason=None,
            shutdown_clean=None,
            attempt_ids=[],
            artifact_errors=[],
            retired_queries=[],
        )
        self.attempt = None
        atomic_json_dump(self.session, self.directory / "session_result.json")

    def prepare(self, *, recording):
        if self.failed or self.attempt is not None:
            raise RuntimeError("previous attempt has not finalized successfully")
        attempt = dict(
            attempt_id=uuid.uuid4().hex,
            session_id=self.session["session_id"],
            run_id=None,
            entered_running=None,
            state="incomplete",
            started_utc=utc_now(),
            started_monotonic_ns=time.monotonic_ns(),
            ended_utc=None,
            ended_monotonic_ns=None,
            termination_reason=None,
            termination_details=None,
            recording_status="pending" if recording else "not_requested",
            raw_path=None,
            staging_path=None,
            row_count=0,
            finalization_errors=[],
        )
        self._write_attempt(attempt)
        self.attempt = attempt
        self.session["attempt_ids"].append(attempt["attempt_id"])
        # Persist membership before RUNNING too. An incomplete attempt never implies no motion.
        try:
            atomic_json_dump(self.session, self.directory / "session_result.json")
        except BaseException:
            self.failed = True
            raise
        return attempt["attempt_id"]

    def entered(self, run_id):
        if self.attempt is not None:
            self.attempt.update(run_id=int(run_id), entered_running=True)

    def _write_attempt(self, attempt):
        try:
            atomic_json_dump(attempt, self.directory / "attempts" / f"{attempt['attempt_id']}.json")
        except BaseException:
            self.failed = True
            raise

    def finish_attempt(self, reason, *, details=None, errors=(), **artifacts):
        if self.attempt is None:
            return
        attempt = dict(self.attempt)
        attempt.update(
            state="finished",
            ended_utc=utc_now(),
            ended_monotonic_ns=time.monotonic_ns(),
            termination_reason=reason,
            termination_details=details,
            finalization_errors=list(errors),
            **artifacts,
        )
        if attempt["entered_running"] is None:
            attempt["entered_running"] = False
        self._write_attempt(attempt)
        self.session["artifact_errors"].extend(errors)
        self.attempt = None

    def finish_session(self, *, outcome, reason, shutdown_clean, **details):
        result = dict(
            self.session,
            state="finished",
            ended_utc=utc_now(),
            outcome=outcome,
            reason=reason,
            shutdown_clean=bool(shutdown_clean),
            **details,
        )
        atomic_json_dump(result, self.directory / "session_result.json")
        self.session = result


def finalize_policy_attempt(
    results, recorder, *, recording_started, reason, detail, events, query_arrays, errors=None
):
    # Caller has revoked authority and attempted stop before any artifact I/O.
    events = deepcopy(events)
    errors = [] if errors is None else errors
    failure, saved = None, None
    status = "empty" if recorder is not None else "not_requested"
    recorder = recorder if recording_started else None
    if recorder is not None:
        try:
            saved = recorder.save_episode(reason=reason, details=errors)
            status = "published" if saved is not None else "empty"
        except Exception as exc:
            failure = exc
            status = "failed"
            errors.append(error_detail("recorder_finalize", exc))
    if results is not None and results.attempt is not None:
        sidecar = None
        try:
            from dexmani_real.utils.atomic_io import atomic_publish

            directory = results.directory / "attempts"
            sidecar = directory / (results.attempt["attempt_id"] + ".npz")
            temporary = sidecar.with_suffix(".tmp.npz")
            np.savez(temporary, **query_arrays)
            atomic_publish(temporary, sidecar)
        except Exception as exc:
            failure = failure or exc
            errors.append(error_detail("query_sidecar", exc))
            sidecar = None
        try:
            results.finish_attempt(
                reason,
                details=detail,
                errors=errors,
                recording_status=status,
                raw_path=str(saved) if saved is not None else None,
                staging_path=str(recorder.staging_path)
                if recorder is not None and recorder.staging_path is not None and saved is None
                else None,
                row_count=recorder.written_frames if recorder is not None else 0,
                query_sidecar=str(sidecar) if sidecar else None,
                trace=events,
            )
        except Exception as exc:
            failure = failure or exc
            logger.exception("attempt result publication failed")
    if failure is not None:
        raise failure
