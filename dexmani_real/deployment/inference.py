"""One model thread owns inference and cleanup independently of device I/O."""

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
        self.model_factory = factory
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = None
        self.operation = None
        self.close_future = None

    def run_operation(self, operation, args, kwargs, query_context=None):
        start = time.monotonic_ns()
        value, error = None, None
        try:
            if operation == "load":
                self.model = self.model_factory()
                value = self.model.warmup(**kwargs)
            elif operation != "close" or self.model is not None:
                value = getattr(self.model, operation)(*args, **kwargs)
        except BaseException as exc:
            error = exc
        # Completion includes the blocking transfer to an owned CPU NumPy prediction.
        return ModelResult(value, error, start, time.monotonic_ns(), query_context)

    def submit(self, operation, *args, query_context=None, **kwargs):
        if self.close_future is not None or self.future is not None:
            raise RuntimeError("Model worker already owns an unreclaimed Future or is closing")
        future = self.executor.submit(self.run_operation, operation, args, kwargs, query_context)
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

        # Capture ownership before poll() can reclaim the finished task during shutdown.
        retired_future, retired_operation = self.future, self.operation

        def close_model():
            retired = None
            if retired_future is not None:
                retired = dict(
                    operation=retired_operation, query_context=None,
                    exception_type=None, message=None, started_ns=None, completed_ns=None,
                )
                try:
                    result = retired_future.result()
                    retired.update(
                        query_context=result.query_context,
                        started_ns=result.started_ns, completed_ns=result.completed_ns,
                    )
                    error = result.error
                except BaseException as exc:
                    error = exc
                if error is not None:
                    retired.update(exception_type=type(error).__name__, message=str(error))
                logger.log(
                    logging.ERROR if error is not None else logging.INFO,
                    "retired model task after shutdown: %s", retired,
                )
            # A failed retired task must not prevent the model's own cleanup attempt.
            return self.run_operation("close", (), {}), retired

        # Queue cleanup behind load/predict while the executor still accepts work.
        self.close_future = self.executor.submit(close_model)
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future

    def close_result(self, timeout_s=0.0):
        """Wait for model close; timeout leaves cleanup pending and does not bound process exit."""
        if self.close_future is None:
            return dict(status="pending", reason="close_not_requested")
        try:
            result, retired = self.close_future.result(timeout=max(0.0, timeout_s))
        except TimeoutError:
            return dict(status="pending", reason="model_close_wait_timeout")
        except BaseException as exc:
            return dict(status="failed", reason=f"{type(exc).__name__}: {exc}")
        outcome = dict(
            status="completed" if result.error is None else "failed",
            reason=(
                None if result.error is None else f"{type(result.error).__name__}: {result.error}"
            ),
            started_ns=result.started_ns,
            completed_ns=result.completed_ns,
            close_error=(
                dict(exception_type=type(result.error).__name__, message=str(result.error))
                if result.error is not None else None
            ),
            retired_task=retired,
        )
        if retired is not None and retired["exception_type"] is not None:
            outcome["status"] = "failed"
            if outcome["reason"] is None:
                outcome["reason"] = (
                    f"retired {retired['operation']} failed: "
                    f"{retired['exception_type']}: {retired['message']}"
                )
        return outcome
