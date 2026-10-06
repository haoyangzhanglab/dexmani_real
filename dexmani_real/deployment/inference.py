"""One serialized model owner; no device access and no queued observations."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    value: object
    error: BaseException | None
    started_ns: int
    completed_ns: int


class InferenceWorker:
    def __init__(self, factory):
        self._factory = factory
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = None
        self.operation = None
        self._closing = False
        self.closed = False
        self.close_error = None
        self.query_context = None

    def _call(self, operation, args, kwargs):
        start = time.monotonic_ns()
        value, error = None, None
        try:
            if operation == "load":
                self._model = self._factory()
                value = self._model.warmup(**kwargs)
            else:
                value = getattr(self._model, operation)(*args, **kwargs)
        except BaseException as exc:
            error = exc
        return ModelResult(value, error, start, time.monotonic_ns())

    def submit(self, operation, *args, **kwargs):
        if self._closing or self.future is not None:
            raise RuntimeError("Model worker already owns an unreclaimed Future or is closing")
        self.operation = operation
        self.future = self._executor.submit(self._call, operation, args, kwargs)

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, operation = self.future, self.operation
        self.future = self.operation = None
        return operation, future.result()

    def close(self):
        """Request serial close without waiting on running CUDA; process exit is not bounded."""
        if self._closing:
            return
        self._closing = True
        retired_context = self.query_context

        def closed(future):
            try:
                result = future.result()
                if result.error is not None:
                    self.close_error = result.error
                    logger.error("model close failed: %s", result.error)
            finally:
                self.future = None
                self.closed = True
                self._executor.shutdown(wait=False)

        def recovered(future):
            if future is not None:
                result = future.result()
                logger.info(
                    "retired model result after shutdown: context=%s started_ns=%s completed_ns=%s error=%s",
                    retired_context,
                    result.started_ns,
                    result.completed_ns,
                    result.error,
                )
                if result.error is not None:
                    logger.error("retired model task failed during shutdown: %s", result.error)
            if self._model is None:
                self.future = None
                self.closed = True
                self._executor.shutdown(wait=False)
            else:
                self.operation = "close"
                self.future = self._executor.submit(self._call, "close", (), {})
                self.future.add_done_callback(closed)

        if self.future is None:
            recovered(None)
        else:
            self.future.add_done_callback(recovered)
