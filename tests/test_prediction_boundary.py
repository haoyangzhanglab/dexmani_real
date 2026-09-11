"""Policy output admission is distinct from passive rollout diagnostics."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from dexmani_real.deployment.inference.worker import _predict_action_chunk


@pytest.mark.parametrize("shape", [(19,), (14, 19), (15, 21), (0, 19)])
def test_wrong_policy_shape_cannot_enter_prediction_ipc(shape):
    runtime = mock.Mock()
    runtime.predict_action_chunk.return_value = np.zeros(shape)
    with pytest.raises(ValueError, match="shape"):
        _predict_action_chunk(
            runtime, object(), SimpleNamespace(chunk_size=15, control_action_dim=19)
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_policy_action_cannot_enter_prediction_ipc(bad):
    runtime = mock.Mock()
    values = np.zeros((15, 19))
    values[-1, -1] = bad
    runtime.predict_action_chunk.return_value = values
    with pytest.raises(ValueError, match="NaN/Inf"):
        _predict_action_chunk(
            runtime, object(), SimpleNamespace(chunk_size=15, control_action_dim=19)
        )


def test_valid_float32_model_output_is_converted_once():
    runtime = mock.Mock()
    values = np.arange(15 * 19, dtype=np.float32).reshape(15, 19)
    runtime.predict_action_chunk.return_value = values
    observation = object()
    actions = _predict_action_chunk(
        runtime, observation, SimpleNamespace(chunk_size=15, control_action_dim=19)
    )
    runtime.predict_action_chunk.assert_called_once_with(observation)
    assert actions.dtype == np.float64
    np.testing.assert_array_equal(actions, values)


@pytest.mark.parametrize("warmup_fails", [False, True])
def test_warmup_failure_is_distinct_from_invalid_diagnostics(warmup_fails):
    from dexmani_real.config.defaults import PolicyParams
    from dexmani_real.deployment.config import InferenceWorkerConfig
    from dexmani_real.deployment.inference import worker

    shared = mock.Mock()
    shared.is_running.value = False
    runtime = mock.Mock()
    runtime.warmup.return_value = [np.nan, np.inf, -1.0]
    if warmup_fails:
        runtime.warmup.side_effect = RuntimeError("model warmup failed")
    cfg = InferenceWorkerConfig(
        "fixture",
        "cpu",
        SimpleNamespace(control_dt_s=0.05, n_action_steps=8),
        0,
        "ckpt.pt",
        10,
    )
    with (
        mock.patch.object(worker, "_load_inference_runtime", return_value=runtime),
        mock.patch.object(worker, "build_fingertip_runtime", return_value=None),
    ):
        if warmup_fails:
            with pytest.raises(RuntimeError, match="model warmup failed"):
                worker.inference_loop(shared, PolicyParams(), cfg)
            shared.set_ready.assert_not_called()
        else:
            worker.inference_loop(shared, PolicyParams(), cfg)
            shared.set_ready.assert_called_once_with("inference")
    runtime.close.assert_called_once()
