import threading
import time

import pytest

from dexmani_real.deployment.inference import InferenceWorker


def wait_until(predicate):
    end = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < end
        time.sleep(0.001)


def test_worker_serial_lifecycle_and_close_during_predict():
    calls = []
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Model:
        def mark(self, op):
            calls.append((op, threading.get_ident()))

        def warmup(self):
            self.mark("warmup")

        def reset_episode(self):
            self.mark("reset")

        def predict(self):
            self.mark("predict")
            entered.set()
            assert release.wait(5)
            raise RuntimeError("retired failure")

        def close(self):
            self.mark("close")
            closed.set()

    def factory():
        calls.append(("load", threading.get_ident()))
        return Model()

    worker = InferenceWorker(factory)
    try:
        worker.submit("load")
        wait_until(lambda: worker.future.done())
        assert worker.poll()[1].error is None
        worker.submit("reset_episode")
        wait_until(lambda: worker.future.done())
        worker.poll()
        worker.submit("predict")
        assert entered.wait(5)
        with pytest.raises(RuntimeError):
            worker.submit("predict")
        assert worker.poll() is None
        start = time.monotonic()
        worker.close()
        assert time.monotonic() - start < 0.2
        assert not closed.is_set()
        release.set()
        assert closed.wait(5)
        wait_until(lambda: worker.future is None)
        assert [op for op, _ in calls] == ["load", "warmup", "reset", "predict", "close"]
        assert len({ident for _, ident in calls}) == 1
        assert calls[0][1] != threading.get_ident()
    finally:
        release.set()
        worker.close()
