"""模型生命周期与预测在同一个工作线程串行执行。"""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from dexmani_policy.deployment import PolicyRuntime

MODEL_SHUTDOWN_TIMEOUT_S = 65.0


class PolicyWorker:
    def __init__(self, factory, *, build_observation):
        self.factory = factory
        self.build_observation = build_observation
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = self.close_future = None

    def _submit(self, function, *args, **kwargs):
        if self.close_future is not None or self.future is not None:
            raise RuntimeError("Policy worker already has an unreclaimed task or is closing")
        self.future = self.executor.submit(function, *args, **kwargs)

    def _load(self):
        self.model = self.factory()

    def load(self):
        self._submit(self._load)

    def _warmup(self, rows, **kwargs):
        observation = self.build_observation(rows)
        cast("PolicyRuntime", self.model).warmup(observation, **kwargs)

    def warmup(self, rows, **kwargs):
        self._submit(self._warmup, rows, **kwargs)

    def _predict(self, rows, *, reset_episode, **kwargs):
        observation = self.build_observation(rows)
        if reset_episode:
            cast("PolicyRuntime", self.model).reset_episode()
        return cast("PolicyRuntime", self.model).predict(observation, **kwargs)

    def predict(self, rows, *, reset_episode=False, **kwargs):
        self._submit(self._predict, rows, reset_episode=reset_episode, **kwargs)

    def poll(self) -> Future | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        self.future = None
        return future

    def discard_pending(self, timeout_s):
        """本段停止后回收尚未消费的任务；超时保留归属，禁止下一段复用。"""
        future = self.future
        if future is None:
            return
        if future not in wait((future,), timeout=timeout_s).done:
            raise TimeoutError("episode model task remains pending")
        self.future = None
        future.result()

    def close(self):
        if self.close_future is not None:
            return self.close_future
        retired = self.future

        def close_model():
            try:
                if retired is not None:
                    retired.result()
            finally:
                if self.model is not None:
                    self.model.close()

        self.close_future = self.executor.submit(close_model)
        self.future = None
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future
