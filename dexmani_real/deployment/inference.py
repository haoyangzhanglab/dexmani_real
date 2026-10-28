"""模型生命周期与预测在同一个工作线程串行执行。"""

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.utils.log import error_detail

if TYPE_CHECKING:
    from dexmani_policy.deployment import PolicyRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    actions: np.ndarray
    duration_ns: int


class PolicyWorker:
    def __init__(self, factory, *, build_observation):
        self.factory = factory
        self.build_observation = build_observation
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = self.operation = self.close_future = None

    @property
    def has_pending_task(self):
        # 完成但尚未回收的任务仍归当前轮次所有。
        return self.future is not None

    def _submit(self, operation, function, *args, **kwargs):
        if self.close_future is not None or self.future is not None:
            raise RuntimeError("Policy worker already has an unreclaimed task or is closing")
        self.future = self.executor.submit(function, *args, **kwargs)
        self.operation = operation

    def _load(self, **kwargs):
        self.model = self.factory()
        self.model.warmup(**kwargs)

    def load(self, **kwargs):
        self._submit("load", self._load, **kwargs)

    def reset(self):
        self._submit("reset", lambda: cast("PolicyRuntime", self.model).reset_episode())

    def _predict(self, rows, **kwargs):
        started_ns = time.monotonic_ns()
        observation = self.build_observation(rows)
        actions = cast("PolicyRuntime", self.model).predict(observation, **kwargs)
        # predict 返回 CPU 动作后才停止计时。
        return PredictionResult(actions, time.monotonic_ns() - started_ns)

    def predict(self, rows, **kwargs):
        self._submit("predict", self._predict, rows, **kwargs)

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, operation = self.future, self.operation
        self.future = self.operation = None
        return operation, future.result()

    def close(self):
        if self.close_future is not None:
            return self.close_future
        retired, operation = self.future, self.operation

        def close_model():
            errors = []
            if retired is not None:
                try:
                    retired.result()
                except BaseException as exc:
                    logger.exception("retired model task failed: %s", operation)
                    errors.append(error_detail(f"retired_{operation}", exc))
            try:
                if self.model is not None:
                    self.model.close()
            except BaseException as exc:
                logger.exception("model close failed")
                errors.append(error_detail("model_close", exc))
            return errors

        self.close_future = self.executor.submit(close_model)
        self.future = self.operation = None
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future

    def wait_closed(self, timeout_s=0.0):
        future = cast(Future[list[dict]], self.close_future)
        try:
            errors = future.result(timeout=max(0.0, timeout_s))
        except TimeoutError:
            return "pending", []
        except BaseException as exc:
            status = "failed" if future.done() else "pending"
            return status, [error_detail("model_close_wait", exc)]
        return "failed" if errors else "completed", errors
