"""Optional, bounded, process-local rollout evidence; never controls motion."""

import json
import logging
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class DiagnosticWriter:
    """A single background owner of an exclusive HDF5 sidecar.

    Files remain incomplete until a clean drain. Overflow disables collection
    and is visible in both file metadata and the status sidecar. Control threads
    never wait for disk IO; failures must not change a target or motion authority.
    """

    def __init__(self, directory, owner, *, capacity=256):
        self.path = Path(directory) / f"{owner}.h5"
        self.queue = Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.ready = threading.Event()
        self.error = ""
        self.dropped = 0
        self.max_enqueue_ns = 0
        self.thread = threading.Thread(target=self._run, name=f"trace-{owner}", daemon=True)
        self.thread.start()
        if not self.ready.wait(5):
            self._fail("diagnostic writer startup timed out")

    def _fail(self, reason):
        if not self.error:
            self.error = str(reason)
            logger.error("Diagnostic data incomplete (%s): %s", self.path, reason)

    def record(self, group, **values):
        if self.error or self.stopping.is_set():
            self.dropped += 1
            return
        start = time.monotonic_ns()
        try:
            # Sources may be reused by the caller on the next control tick.
            arrays = {
                key: np.array(value.encode() if isinstance(value, str) else value, copy=True)
                for key, value in values.items()
            }
            self.queue.put_nowait((group, arrays))
        except Full:
            self.dropped += 1
            self._fail("queue overflow; collection disabled")
        except Exception as exc:
            self.dropped += 1
            self._fail(exc)
        self.max_enqueue_ns = max(self.max_enqueue_ns, time.monotonic_ns() - start)

    @staticmethod
    def _append(file, name, values):
        group = file.require_group(name)
        if len(group) and set(group) != set(values):
            raise ValueError(f"diagnostic fields changed in {name}")
        for key, value in values.items():
            if key not in group:
                dtype = "S256" if value.dtype.kind == "S" else value.dtype
                group.create_dataset(
                    key, shape=(0, *value.shape), maxshape=(None, *value.shape), dtype=dtype
                )
            ds = group[key]
            if ds.shape[1:] != value.shape:
                raise ValueError(f"diagnostic shape changed: {name}/{key}")
            ds.resize(len(ds) + 1, axis=0)
            ds[-1] = value

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(self.path, "x") as file:
                file.attrs.update(complete=False, clock="host_monotonic_ns")
                file.flush()
                self.ready.set()
                flushed = time.monotonic()
                while not self.stopping.is_set() or not self.queue.empty():
                    try:
                        group, values = self.queue.get(timeout=0.05)
                    except Empty:
                        continue
                    self._append(file, group, values)
                    if time.monotonic() - flushed >= 1:
                        file.flush()
                        flushed = time.monotonic()
                file.attrs.update(
                    complete=not bool(self.error),
                    error=self.error,
                    dropped=self.dropped,
                    max_enqueue_ns=self.max_enqueue_ns,
                )
        except Exception as exc:
            self._fail(exc)
        finally:
            self.ready.set()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self._fail("diagnostic writer did not drain before shutdown")
        try:
            self.path.with_suffix(".status.json").write_text(
                json.dumps(
                    {
                        "complete": not bool(self.error),
                        "error": self.error,
                        "dropped": self.dropped,
                        "max_enqueue_ns": self.max_enqueue_ns,
                    },
                    indent=2,
                )
            )
        except Exception:
            logger.exception("Could not persist diagnostic completion status: %s", self.path)


def trace_sdk_call(trace, send, target, *, run_id, sequence):
    """Time exactly one existing SDK invocation, preserving its result/exception."""
    if trace is None:
        return send(target)
    start = time.monotonic_ns()
    status = "exception"
    try:
        result = send(target)
        status = str(getattr(result, "value", result))
        return result
    finally:
        end = time.monotonic_ns()
        trace.record(
            "commands",
            run_id=run_id,
            sequence=sequence,
            target=target,
            sdk_start_ns=start,
            sdk_end_ns=end,
            status=status,
        )


def trace_unsent(trace, target, *, run_id, sequence, reason):
    if trace is not None:
        trace.record(
            "commands",
            run_id=run_id,
            sequence=sequence,
            target=target,
            sdk_start_ns=0,
            sdk_end_ns=0,
            status=reason,
        )
