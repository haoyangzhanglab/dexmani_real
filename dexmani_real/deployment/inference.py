"""One serialized model owner; no device access and no queued observations."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    value: object
    error: BaseException | None
    started_ns: int
    completed_ns: int
    query_context: dict | None = None


class InferenceWorker:
    def __init__(self, factory):
        self._factory = factory
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = None
        self.operation = None
        self.close_future = None

    def call(self, operation, args, kwargs, query_context=None):
        start = time.monotonic_ns()
        value, error = None, None
        try:
            if operation == "load":
                self._model = self._factory()
                value = self._model.warmup(**kwargs)
            elif operation != "close" or self._model is not None:
                value = getattr(self._model, operation)(*args, **kwargs)
        except BaseException as exc:
            error = exc
        # Completion includes the blocking transfer to an owned CPU NumPy prediction.
        return ModelResult(value, error, start, time.monotonic_ns(), query_context)

    def submit(self, operation, *args, query_context=None, **kwargs):
        if self.close_future is not None or self.future is not None:
            raise RuntimeError("Model worker already owns an unreclaimed Future or is closing")
        future = self._executor.submit(self.call, operation, args, kwargs, query_context)
        self.operation, self.future = operation, future

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, operation = self.future, self.operation
        self.future = self.operation = None
        return operation, future.result()

    def close(self):
        """Request serial close without waiting on running CUDA; process exit is not bounded."""
        if self.close_future is not None:
            return self.close_future

        def recovered(future):
            result = future.result()
            logger.info(
                "retired model result after shutdown: context=%s started_ns=%s completed_ns=%s error=%s",
                result.query_context,
                result.started_ns,
                result.completed_ns,
                result.error,
            )
            if result.error is not None:
                logger.error("retired model task failed during shutdown: %s", result.error)

        # Queue cleanup behind load/predict while the executor still accepts work.
        self.close_future = self._executor.submit(self.call, "close", (), {})
        if self.future is not None:
            self.future.add_done_callback(recovered)
        self._executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future

    def close_result(self, timeout_s=0.0):
        """Wait for the sole close Future; timeout means pending and does not bound thread/CUDA process exit."""
        if self.close_future is None:
            return dict(status="pending", reason="close_not_requested")
        try:
            result = self.close_future.result(timeout=max(0.0, timeout_s))
        except TimeoutError:
            return dict(status="pending", reason="model_close_wait_timeout")
        except BaseException as exc:
            return dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        return dict(
            status="completed" if result.error is None else "failed",
            reason=(
                None if result.error is None else f"{type(result.error).__name__}: {result.error}"
            ),
            started_ns=result.started_ns,
            completed_ns=result.completed_ns,
        )
