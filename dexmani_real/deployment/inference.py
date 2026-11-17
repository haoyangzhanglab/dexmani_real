"""模型生命周期与预测在同一个工作线程串行执行。"""

import logging
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

from dexmani_real.utils.profile import call

if TYPE_CHECKING:
    from dexmani_policy.deployment import PolicyRuntime

MODEL_SHUTDOWN_TIMEOUT_S = 5.0
logger = logging.getLogger(__name__)


class ModelShutdownTimeout(TimeoutError):
    """模型关闭仍在进行；Session 完成关键收尾后才向 CLI 传播。"""


class PolicyWorker:
    def __init__(self, factory, *, build_observation, profiler=None):
        self.profiler = profiler
        self.factory = factory
        self.build_observation = build_observation
        self.warmed_up = False
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-model")
        self.future = self.close_future = None

    def _submit(self, stage, function, *args, **kwargs):
        if self.close_future is not None or self.future is not None:
            raise RuntimeError("Policy worker already has an unreclaimed task or is closing")
        if self.profiler is None:
            self.future = self.executor.submit(function, *args, **kwargs)
            return
        timing = {"stage": stage, "submitted": time.perf_counter_ns(), "completed": None}
        self._profile_request = timing
        self.future = self.executor.submit(function, *args, **kwargs)
        self.profiler.count(f"{stage}_requests")

        def completed(future):
            stamp = time.perf_counter_ns()
            with self.profiler.lock:
                timing["completed"] = stamp
            self.profiler.sample(f"{stage}_future_latency_ms", (stamp - timing["submitted"]) / 1e6)
            self.profiler.count(f"{stage}_completed")

        self.future.add_done_callback(completed)

    def _load(self):
        self.model = self.factory()
        return self.model.info

    def load(self):
        self._submit("load", self._load)

    def _warmup(self, rows, **kwargs):
        observation = call(self.profiler, "warmup_input_build_ms", self.build_observation, rows)
        cast("PolicyRuntime", self.model).warmup(observation, **kwargs)

    def warmup(self, rows, **kwargs):
        self._submit("warmup", self._warmup, rows, **kwargs)

    def _predict(self, rows, *, reset_episode, **kwargs):
        observation = call(self.profiler, "model_input_build_ms", self.build_observation, rows)
        if reset_episode:
            cast("PolicyRuntime", self.model).reset_episode()
        return call(
            self.profiler, "model_predict_ms", cast("PolicyRuntime", self.model).predict,
            observation, **kwargs,
        )

    def predict(self, rows, *, reset_episode=False, **kwargs):
        self._submit("prediction", self._predict, rows, reset_episode=reset_episode, **kwargs)

    def poll(self) -> Future | None:
        if self.future is None or not self.future.done():
            return None
        future = self.future
        if self.profiler is not None:
            observed = time.perf_counter_ns()
            with self.profiler.lock:
                completed = self._profile_request["completed"]
            stage = self._profile_request["stage"]
            if completed is not None and completed <= observed:
                self.profiler.sample(f"{stage}_poll_delay_ms", (observed - completed) / 1e6)
            else:
                # done 状态先于回调可见；不把竞争窗口伪装成零等待。
                self.profiler.count(f"{stage}_completion_marker_races")
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
                    # 中断优先；其他收尾错误不能覆盖在途预测的首个失败。
                    if task_error is None or (
                        isinstance(task_error, Exception) and not isinstance(close_error, Exception)
                    ):
                        raise
                    logger.exception("policy model close also failed")

        self.close_future = self.executor.submit(close_model)
        self.executor.shutdown(wait=False, cancel_futures=False)
        return self.close_future
