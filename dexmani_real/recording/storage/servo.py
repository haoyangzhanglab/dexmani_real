"""Mode-1 evidence, separate from the low-frequency observation/action arrays."""

from enum import IntEnum
from queue import Empty, Full, Queue

import numpy as np

from dexmani_real.robot.commands import DispatchStatus


class ServoEvent(IntEnum):
    BEGIN = 1
    ACTIVATE = 2
    POINT = 3
    END = 4
    FAULT = 5
    UNCERTAIN = 6
    FINISH = 7
    CANCEL = 8


SERVO_EVENT_DTYPE = np.dtype([
    ("sequence", "<i8"), ("kind", "u1"), ("run_id", "<i8"), ("command_seq", "<i8"),
    ("point_seq", "<i8"), ("timestamp_ns", "<i8"), ("planned_ns", "<i8"),
    ("sampled_ns", "<i8"), ("submit_until_ns", "<i8"), ("execute_until_ns", "<i8"),
    ("started_ns", "<i8"), ("completed_ns", "<i8"), ("status", "u1"), ("code", "<i4"),
    ("q_ref", "<f8", (7,)), ("q_goal", "<f8", (7,)), ("reason", "S128"),
])


class ServoEvidence:
    """One capture's bounded stream: only the arm owner emits, the episode writer drains."""

    def __init__(self, capacity, metadata, *, queue=None, failure=None):
        self.queue = Queue(maxsize=capacity) if queue is None else queue
        self.metadata = metadata
        self._error = None
        self.failure = failure
        self.closed = False
        self.producer_done = False
        self.sequence = 0

    def fail(self, reason):
        if self._error is None:
            self._error = str(reason)
        if self.failure is not None:
            self.failure.set()

    @property
    def error(self):
        return self._error or ('servo evidence producer/writer failed'
                               if self.failure is not None and self.failure.is_set() else None)

    def emit(self, kind, timestamp_ns, **values):
        if self.closed:
            raise RuntimeError("servo evidence already closed")
        row = np.zeros((), dtype=SERVO_EVENT_DTYPE)
        for name in ("run_id", "command_seq", "point_seq", "planned_ns", "sampled_ns",
                     "submit_until_ns", "execute_until_ns", "started_ns", "completed_ns", "code"):
            row[name] = -1
        row["q_ref"] = row["q_goal"] = np.nan
        row["sequence"], row["kind"], row["timestamp_ns"] = self.sequence, kind, timestamp_ns
        for name, value in values.items():
            row[name] = str(value).encode()[:128] if name == "reason" else value
        try:
            self.queue.put_nowait(row)
        except Full as exc:
            self.fail("servo evidence queue overflow")
            raise RuntimeError(self.error) from exc
        self.sequence += 1

    def drain(self, limit=512):
        rows = []
        for _ in range(limit):
            try:
                row = self.queue.get_nowait()
                rows.append(row)
                if row['kind'] == ServoEvent.FINISH:
                    self.closed = True
            except Empty:
                break
        return np.asarray(rows, dtype=SERVO_EVENT_DTYPE)


def validate_servo_evidence(events, command_ids, dispatch, goals, *, complete,
                            arm_dispatch=DispatchStatus.ACCEPTED):
    """Scan an array/HDF5 dataset in bounded blocks, matching activations to ordered rows.

    One pending target and one producer preserve command order. Supersession does
    not require arrival at q_goal; any missing or failed execution rejects the episode.
    """
    if not complete or events.dtype != SERVO_EVENT_DTYPE or events.ndim != 1 or not len(events):
        raise ValueError("incomplete Mode-1 servo evidence")
    if events[0]["kind"] != ServoEvent.BEGIN or events[-1]["kind"] != ServoEvent.FINISH:
        raise ValueError("servo capture boundary missing")
    if (dispatch.ndim != 2 or dispatch.shape[1] != 2
            or command_ids.shape != (len(dispatch), 2) or goals.shape != (len(dispatch), 7)):
        raise ValueError("Mode-1 commands do not align with rows")
    active, seen, row, has_point = None, set(), 0, False
    previous_point = previous_ns = None
    for start in range(0, len(events), 4096):
        block = events[start:start + 4096]
        if not np.array_equal(block["sequence"], np.arange(start, start + len(block))):
            raise ValueError("servo evidence sequence gap")
        if (previous_ns is not None and block["timestamp_ns"][0] < previous_ns
                or np.any(np.diff(block["timestamp_ns"]) < 0)):
            raise ValueError("servo event clock went backwards")
        previous_ns = block["timestamp_ns"][-1]
        for index, event in enumerate(block, start):
            kind = int(event["kind"])
            key = (int(event["run_id"]), int(event["command_seq"]))
            if kind == ServoEvent.ACTIVATE:
                if active is not None or key in seen:
                    raise ValueError("overlapping/duplicate servo activation")
                if row >= len(dispatch) or key != tuple(command_ids[row]):
                    raise ValueError(f"row {row}: servo activation does not match command order")
                if dispatch[row, 0] != arm_dispatch:
                    raise ValueError(f"row {row}: arm dispatch differs from execution semantics")
                if not np.isfinite(event["q_goal"]).all() or not np.array_equal(goals[row], event["q_goal"]):
                    raise ValueError(f"row {row}: invalid or mismatched q_goal")
                active, has_point = key, False
                seen.add(key)
                row += 1
            elif kind == ServoEvent.POINT:
                if active != key:
                    raise ValueError("servo point outside its execution interval")
                if previous_point is not None and event["point_seq"] != previous_point + 1:
                    raise ValueError("servo point sequence gap")
                previous_point = int(event["point_seq"])
                if (event["status"] != DispatchStatus.ACCEPTED or event["code"] != 0
                        or not np.isfinite(event["q_ref"]).all()
                        or not 0 < event["planned_ns"] <= event["sampled_ns"] <= event["started_ns"] <= event["completed_ns"]):
                    raise ValueError("servo point failed or has invalid timing")
                has_point = True
            elif kind == ServoEvent.END:
                if active != key or not has_point:
                    raise ValueError("servo interval has no accepted point or matching activation")
                if event["reason"] not in (b"superseded", b"revoked", b"stop", b"close", b"capture_end"):
                    raise ValueError("servo interval terminated abnormally")
                active = None
            elif kind in (ServoEvent.FAULT, ServoEvent.UNCERTAIN, ServoEvent.CANCEL):
                raise ValueError("servo failure/unknown result")
            elif not ((kind == ServoEvent.BEGIN and index == 0)
                      or (kind == ServoEvent.FINISH and index == len(events) - 1)):
                raise ValueError("unknown event or unexpected servo capture boundary")
    if active is not None:
        raise ValueError("unterminated servo interval")
    if row != len(dispatch):
        raise ValueError(f"row {row}: no servo activation")
