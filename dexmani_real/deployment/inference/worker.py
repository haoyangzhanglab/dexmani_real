"""Inference worker: observations -> flat predictions -> ``prediction_ring``.

The inference worker is the *only* process that touches the model. It reads
causal observations from the shared rings, runs
:meth:`~dexmani_real.deployment.inference.runtime.PolicyRuntime.predict`, and publishes
the resulting :class:`~dexmani_real.deployment.prediction.Prediction` to the
latest-wins ``prediction_ring``. It never writes ``coupled_cmd_ring``, the
SDK, ``SafetyState``, or ``run_generation`` — model output is a proposal, not a
robot command.

``inference_loop`` is a plain ``*_loop(shared, config)`` function (not an
``mp.Process`` subclass); lifecycle and supervision stay in the runtime layer.

The worker loads the selected experiment through the public DexMani Policy API,
then wraps it with Real's NumPy observation/action adapter.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dexmani_real.config.defaults import PolicyParams
from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    FingertipAssemblerConfig,
    InferenceWorkerConfig,
    PolicyDeploymentConfig,
)
from dexmani_real.deployment.inference.observation import (
    _build_observation,
    _to_policy_observation,
    build_fingertip_runtime,
    observation_timing_ms,
)
from dexmani_real.deployment.inference.runtime import PolicyRuntime
from dexmani_real.deployment.metrics import PolicyStats, flush_every
from dexmani_real.deployment.prediction import Prediction
from dexmani_real.deployment.timing import next_periodic_deadline_ns
from dexmani_real.ipc.channels import RuntimeChannels, new_frame
from dexmani_real.ipc.schema import PREDICTION_DTYPE
from dexmani_real.runtime.safety import SafetyState, read_run_state_snapshot
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Poll delay while required causal feedback is unavailable.
_NO_FEEDBACK_POLL_S = 0.005
# Poll interval while ARMED (no inference) — gentler than the feedback poll.
_ARMED_IDLE_POLL_S = 0.01
_SYNC_REQUEST_WAIT_S = 0.05


def _load_inference_runtime(config: InferenceWorkerConfig) -> PolicyRuntime:
    """Load Policy-owned model state and return the Real NumPy adapter."""
    # Deliberately imported inside the inference child: the parent must not
    # import torch/Policy or initialize CUDA, and model objects never cross spawn.
    from dexmani_policy.deployment import load_experiment

    from dexmani_real.deployment.inference.dexmani_policy import DexManiPolicyAdapter

    loaded_policy = load_experiment(
        config.experiment,
        device=config.device,
        seed=config.seed,
    )
    try:
        return DexManiPolicyAdapter(loaded_policy, config.spec)
    except BaseException:
        loaded_policy.close()
        raise


def serialize_prediction(prediction: Prediction) -> np.ndarray:
    """Serialize one validated flat prediction without scheduling it."""
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Prediction")
    frame = new_frame(PREDICTION_DTYPE)
    frame["run_generation"][0] = np.uint64(prediction.run_generation)
    frame["source_monotonic_ns"][0] = np.uint64(prediction.source_monotonic_ns)
    frame["logical_step_monotonic_ns"][0] = np.uint64(
        prediction.logical_step_monotonic_ns
    )
    frame["num_steps"][0] = np.uint32(prediction.num_steps)
    frame["action_dim"][0] = np.uint32(prediction.actions.shape[1])
    frame["actions"][0, : prediction.num_steps, : prediction.actions.shape[1]] = (
        prediction.actions
    )
    return frame


def publish_prediction(shared: RuntimeChannels, prediction: Prediction) -> bool:
    """Generation-fence and publish one flat prediction to the single-slot ring."""
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Prediction")
    if int(shared.run_generation.value) != prediction.run_generation:
        return False
    shared.prediction_ring.write(serialize_prediction(prediction))
    return True


def _clear_sync_request_for_inactive_snapshot(
    shared: RuntimeChannels,
    *,
    observed_generation: int,
) -> bool:
    """Clear only while the lifecycle snapshot still names an inactive epoch.

    The safety transition and this check share ``motion_lock``. If a B request
    has already advanced the generation to RUNNING, its newly-set inference
    request cannot be cleared using an older ARMED snapshot.
    """
    with shared.motion_lock:
        if int(shared.run_generation.value) != int(observed_generation):
            return False
        if (
            int(shared.safety_state.value) == int(SafetyState.RUNNING)
            and not bool(shared.error_state.value)
            and not bool(shared.estop_request.value)
        ):
            return False
        shared.inference_request.clear()
        return True


def _consume_sync_request(
    shared: RuntimeChannels,
    *,
    observed_generation: int,
) -> int | None:
    """Consume a request only if it still belongs to the observed RUNNING epoch."""
    with shared.motion_lock:
        current_generation = int(shared.run_generation.value)
        if current_generation != int(observed_generation):
            return None
        if int(shared.safety_state.value) != int(SafetyState.RUNNING):
            return None
        if bool(shared.error_state.value) or bool(shared.estop_request.value):
            return None
        shared.inference_request.clear()
        return current_generation


def inference_loop(
    shared: RuntimeChannels,
    policy: PolicyParams,
    config: InferenceWorkerConfig,
    deployment_config: PolicyDeploymentConfig | None = None,
    fingertip_config: FingertipAssemblerConfig | None = None,
) -> None:
    """Inference process entry point — produces proposals, never robot commands.

    Startup order: heartbeat early -> Policy-owned strict load -> mark ready.
    A load/import/instantiation failure raises out
    of this function and becomes a supervisor-observed process failure; there
    is no dummy safe mode. The main loop reads a fresh generation each tick and
    calls ``runtime.reset_episode`` when it changes.
    """
    if not isinstance(policy, PolicyParams):
        raise TypeError("inference_loop requires resolved runtime PolicyParams")
    if not isinstance(config, InferenceWorkerConfig):
        raise TypeError("inference_loop requires an InferenceWorkerConfig")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("inference_loop requires a PolicyDeploymentConfig")

    # Heartbeat before any lazy import so the supervisor never sees a dead gap.
    shared.set_heartbeat("inference", time.monotonic())
    stats = PolicyStats()

    runtime = _load_inference_runtime(config)
    try:
        fingertip_runtime = build_fingertip_runtime(config.spec, fingertip_config)
        warmup_samples = 5
        timings_s = runtime.warmup(samples=warmup_samples)
        if len(timings_s) != warmup_samples:
            raise RuntimeError("policy runtime returned incomplete warmup timings")
        if any(not np.isfinite(value) or value < 0.0 for value in timings_s):
            raise RuntimeError("policy runtime returned invalid warmup timing")
        logger.info(
            "inference warmup: samples_ms=%s mode=%s",
            ",".join(f"{value * 1e3:.3f}" for value in timings_s),
            deployment.inference_mode,
        )
    except BaseException:
        try:
            runtime.close()
        except Exception:
            logger.warning("inference: startup runtime.close raised", exc_info=True)
        raise

    shared.set_ready("inference")
    # Refresh the heartbeat after model loading, which may exceed the timeout.
    shared.set_heartbeat("inference", time.monotonic())
    logger.info("inference_loop: ready (runtime=%s)", FIXED_POLICY_RUNTIME_TARGET)

    step_dt_ns = int(round(float(config.spec.control_dt_s) * 1e9))
    async_period_ns = int(config.spec.n_action_steps) * step_dt_ns

    observation_id = 0
    last_generation = -1
    last_logical_step_ns = 0
    sync_request_generation: int | None = None
    async_deadline_ns: int | None = None
    last_metrics_flush_ns = time.monotonic_ns()

    def wait_for_observation() -> None:
        nonlocal last_metrics_flush_ns
        last_metrics_flush_ns = flush_every(
            stats,
            last_ns=last_metrics_flush_ns,
            prefix="inference metrics",
            debug=True,
        )
        time.sleep(_NO_FEEDBACK_POLL_S)

    try:
        while shared.is_running.value:
            # Heartbeat every tick, including no-feedback and slow-inference paths.
            shared.set_heartbeat("inference", time.monotonic())

            run_snapshot = read_run_state_snapshot(shared)
            run_generation = run_snapshot.generation
            if run_generation != last_generation:
                runtime.reset_episode()
                last_generation = run_generation
                observation_id = 0  # new observation epoch for the new run
                last_logical_step_ns = 0
                sync_request_generation = None
                async_deadline_ns = None

            # ARMED = no inference; the policy executor gates RUNNING via B.
            if run_snapshot.state is not SafetyState.RUNNING:
                if deployment.inference_mode == "sync":
                    _clear_sync_request_for_inactive_snapshot(
                        shared,
                        observed_generation=run_generation,
                    )
                    sync_request_generation = None
                else:
                    async_deadline_ns = None
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                if deployment.inference_mode == "sync":
                    _clear_sync_request_for_inactive_snapshot(
                        shared,
                        observed_generation=run_generation,
                    )
                    sync_request_generation = None
                else:
                    async_deadline_ns = None
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if run_snapshot.started_monotonic_ns <= 0:
                raise RuntimeError("RUNNING state has no observation epoch")
            if deployment.inference_mode == "sync":
                if sync_request_generation is None:
                    if not shared.inference_request.wait(timeout=_SYNC_REQUEST_WAIT_S):
                        continue
                    shared.set_heartbeat("inference", time.monotonic())
                    request_generation = _consume_sync_request(
                        shared,
                        observed_generation=run_generation,
                    )
                    if request_generation is None:
                        continue
                    run_snapshot = read_run_state_snapshot(shared)
                    if (
                        run_snapshot.state is not SafetyState.RUNNING
                        or run_snapshot.generation != request_generation
                    ):
                        continue
                    if run_snapshot.started_monotonic_ns <= 0:
                        raise RuntimeError("RUNNING state has no observation epoch")
                    run_generation = run_snapshot.generation
                    sync_request_generation = run_generation
                elif sync_request_generation != run_generation:
                    sync_request_generation = None
                    continue
            else:
                now_ns = time.monotonic_ns()
                if async_deadline_ns is None:
                    async_deadline_ns = now_ns
                if now_ns < async_deadline_ns:
                    time.sleep(
                        min(
                            (async_deadline_ns - now_ns) / 1e9,
                            _SYNC_REQUEST_WAIT_S,
                        )
                    )
                    continue
            anchor_ns = time.monotonic_ns()
            observation_id += 1
            observation = _build_observation(
                shared,
                policy,
                config.spec,
                observation_id=observation_id,
                run_generation=run_generation,
                run_started_ns=run_snapshot.started_monotonic_ns,
                anchor_ns=anchor_ns,
                step_dt_ns=step_dt_ns,
            )
            if observation is None:
                wait_for_observation()
                continue
            if observation.logical_step_monotonic_ns <= last_logical_step_ns:
                wait_for_observation()
                continue
            observation_age_ms, observation_skew_ms = observation_timing_ms(observation)
            stats.observe_observation_age_ms(observation_age_ms)
            stats.observe_observation_skew_ms(observation_skew_ms)
            last_logical_step_ns = observation.logical_step_monotonic_ns

            started_ns = time.monotonic_ns()
            policy_observation = _to_policy_observation(
                observation,
                config.spec,
                fingertip_runtime=fingertip_runtime,
            )
            actions = runtime.predict(policy_observation)
            finished_ns = time.monotonic_ns()
            inference_ms = (finished_ns - started_ns) / 1e6
            stats.observe_inference_latency_ms(inference_ms)

            prediction = Prediction(
                run_generation=run_generation,
                source_monotonic_ns=observation.latest_source_monotonic_ns,
                logical_step_monotonic_ns=observation.logical_step_monotonic_ns,
                actions=actions,
            )
            if not publish_prediction(shared, prediction):
                logger.debug("inference: prediction dropped (generation advanced)")
            if deployment.inference_mode == "sync":
                sync_request_generation = None
            else:
                assert async_deadline_ns is not None
                async_deadline_ns = next_periodic_deadline_ns(
                    async_deadline_ns,
                    async_period_ns,
                    finished_ns,
                )

            last_metrics_flush_ns = flush_every(
                stats,
                last_ns=last_metrics_flush_ns,
                prefix="inference metrics",
                debug=True,
            )
    finally:
        try:
            runtime.close()
        except Exception:
            logger.warning("inference: runtime.close raised", exc_info=True)
        logger.info("inference_loop: exited")
