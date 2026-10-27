"""单线程串行执行模型操作，与设备 I/O 独立。"""

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
    model_started_ns: int | None = None


class InferenceWorker:
    def __init__(self, factory, *, build_observation, reset_observation=None):
        self.model_factory = factory
        self.build_observation = build_observation
        self.reset_observation = reset_observation
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = None
        self.operation = None
        self.close_future = None

    @property
    def has_pending_task(self):
        """已完成但尚未回收的模型任务仍算在途。"""
        return self.future is not None

    def run_operation(self, operation, args, kwargs, query_context=None):
        start = time.monotonic_ns()
        model_start = None
        value, error = None, None
        try:
            if operation == "load":
                self.model = self.model_factory()
                value = self.model.warmup(**kwargs)
            elif operation == "predict":
                observation = self.build_observation(args[0])
                model_start = time.monotonic_ns()
                value = self.model.predict(observation, **kwargs)
            elif operation == "reset_episode":
                if self.reset_observation is not None:
                    self.reset_observation()
                value = self.model.reset_episode()
            elif operation == "close":
                if self.model is not None:
                    value = self.model.close()
            else:
                raise ValueError(f"Unknown model operation: {operation}")
        except BaseException as exc:
            error = exc
        # 完成时间包含预测结果到 CPU 的阻塞传输。
        return ModelResult(value, error, start, time.monotonic_ns(), query_context, model_start)

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
        """排队关闭模型；不能强制中断在途 CUDA 调用。"""
        if self.close_future is not None:
            return self.close_future

        # 在 poll() 回收任务前保留其身份与结果。
        retired_future, retired_operation = self.future, self.operation

        def close_model():
            retired = None
            if retired_future is not None:
                result = retired_future.result()
                error = result.error
                retired = dict(
                    operation=retired_operation, query_context=result.query_context,
                    exception_type=type(error).__name__ if error is not None else None,
                    message=str(error) if error is not None else None,
                    started_ns=result.started_ns, completed_ns=result.completed_ns,
                )
                logger.log(
                    logging.ERROR if error is not None else logging.INFO,
                    "retired model task after shutdown: %s", retired,
                )
            # 迟到任务失败后仍需关闭模型。
            return self.run_operation("close", (), {}), retired

        self.close_future = self.executor.submit(close_model)
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future

    def close_result(self, timeout_s=0.0):
        """超时保持 pending，并报告关闭与迟到任务的错误。"""
        if self.close_future is None:
            return dict(status="pending", reason="close_not_requested")
        try:
            result, retired = self.close_future.result(timeout=max(0.0, timeout_s))
        except TimeoutError:
            return dict(status="pending", reason="model_close_wait_timeout")
        except BaseException as exc:
            # 等待被中断也记为清理失败，允许会话结果落盘。
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
