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


class InferenceWorker:
    def __init__(self, factory):
        self._factory = factory
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = None
        self.operation = None
        self.close_future = None
        self.query_context = None

    def _call(self, operation, args, kwargs):
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
        # LoadedPolicy.predict returns an owned CPU NumPy future, after blocking
        # device-to-host transfer. Completion includes that consumable result.
        return ModelResult(value, error, start, time.monotonic_ns())

    def submit(self, operation, *args, **kwargs):
        if self.close_future is not None or self.future is not None:
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
        if self.close_future is not None:
            return self.close_future
        retired_context = self.query_context

        def recovered(future):
            result = future.result()
            logger.info(
                "retired model result after shutdown: context=%s started_ns=%s completed_ns=%s error=%s",
                retired_context, result.started_ns, result.completed_ns, result.error,
            )
            if result.error is not None:
                logger.error("retired model task failed during shutdown: %s", result.error)

        # Enqueue while the interpreter still accepts work, even during load/predict.
        # The same owner checks model existence only when cleanup reaches the queue head.
        self.close_future = self._executor.submit(self._call, "close", (), {})
        if self.future is not None:
            self.future.add_done_callback(recovered)
        self._executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future

    def close_result(self, timeout_s=0.0):
        """Observe the sole close Future; a timeout is pending, not a close exception.

        This bounds only our wait, not Python thread/CUDA process exit.
        """
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
            reason=None if result.error is None else f"{type(result.error).__name__}: {result.error}",
            started_ns=result.started_ns, completed_ns=result.completed_ns,
        )
