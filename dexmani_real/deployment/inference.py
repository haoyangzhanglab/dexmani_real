"""模型生命周期与预测在同一个工作线程串行执行。"""

import sys
from concurrent.futures import Future, ThreadPoolExecutor
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
        return self.model.info

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

    def close(self):
        if self.close_future is not None:
            return self.close_future
        retired = self.future

        def close_model():
            try:
                if retired is not None:
                    retired.result()
            finally:
                task_error = sys.exc_info()[1]
                self.future = None
                try:
                    if self.model is not None:
                        self.model.close()
                        self.model = None
                except BaseException as close_error:
                    # 会话通过最外层中断锁存急停；关闭失败不能遮蔽它。
                    if isinstance(task_error, (KeyboardInterrupt, SystemExit)):
                        raise task_error from close_error
                    raise

        self.close_future = self.executor.submit(close_model)
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future
