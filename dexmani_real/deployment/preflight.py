"""Temporary isolated policy smoke test retained until lifecycle Phase 4."""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.deployment.config import PolicyRuntimeConfig
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
)

_ARM_DOF = 7
_HAND_DOF = 12
_MAX_CHILD_MESSAGE_BYTES = 16 * 1024
_MAX_CHILD_ERROR_CHARS = 2 * 1024
_CHILD_JOIN_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class PreflightResult:
    """Small smoke-test result containing no artifact or source identity."""

    action_steps: int
    action_dim: int

    def __post_init__(self) -> None:
        for name in ("action_steps", "action_dim"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"preflight result {name} must be positive")

    def to_wire(self) -> dict[str, int]:
        return {"action_steps": self.action_steps, "action_dim": self.action_dim}

    @classmethod
    def from_wire(cls, value: Any) -> "PreflightResult":
        if not isinstance(value, dict) or set(value) != {
            "action_steps",
            "action_dim",
        }:
            raise ValueError("policy preflight child returned an invalid result")
        return cls(**value)


def run_isolated_preflight(
    runtime_config: PolicyRuntimeConfig, *, timeout_s: float = 120.0
) -> PreflightResult:
    """Run the temporary bounded child smoke test and propagate all failures."""
    if not isinstance(runtime_config, PolicyRuntimeConfig):
        raise TypeError("preflight requires PolicyRuntimeConfig")
    if timeout_s <= 0.0:
        raise ValueError("preflight timeout_s must be positive")
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_preflight_child,
        args=(child_connection, runtime_config),
        name="dexmani_policy_preflight",
    )
    process.daemon = False
    started = False
    try:
        process.start()
        started = True
    except BaseException:
        child_connection.close()
        parent_connection.close()
        _close_process(process)
        raise
    else:
        child_connection.close()
    try:
        if not parent_connection.poll(timeout_s):
            raise TimeoutError("policy preflight child timed out")
        try:
            payload = parent_connection.recv_bytes(_MAX_CHILD_MESSAGE_BYTES)
        except EOFError as exc:
            raise RuntimeError(
                "policy preflight child exited without a result"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "policy preflight child result exceeds its bound"
            ) from exc
        message = _decode_child_message(payload)
        process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError("policy preflight child did not exit after its result")
        if process.exitcode != 0:
            raise RuntimeError(
                f"policy preflight child exited with code {process.exitcode}"
            )
        if message["ok"] is not True:
            raise RuntimeError(f"policy preflight failed: {message['error']}")
        result = PreflightResult.from_wire(message["result"])
        deployment = runtime_config.deployment
        if (
            result.action_steps != deployment.n_action_steps
            or result.action_dim != deployment.control_action_dim
        ):
            raise RuntimeError("policy preflight child result mismatches its request")
        return result
    finally:
        parent_connection.close()
        if started:
            _terminate_join_kill_close(process)
        else:
            _close_process(process)


def _decode_child_message(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("policy preflight child returned invalid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"ok", "result", "error"}:
        raise RuntimeError("policy preflight child returned an invalid message schema")
    if (
        value["ok"] is True
        and value["error"] is None
        and isinstance(value["result"], dict)
    ):
        return value
    if (
        value["ok"] is False
        and value["result"] is None
        and isinstance(value["error"], str)
        and len(value["error"]) <= _MAX_CHILD_ERROR_CHARS
    ):
        return value
    raise RuntimeError("policy preflight child returned an invalid message")


def _encode_child_message(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    if len(payload) > _MAX_CHILD_MESSAGE_BYTES:
        raise ValueError("policy preflight child message exceeds its bound")
    return payload


def _preflight_child(connection: Any, runtime_config: PolicyRuntimeConfig) -> None:
    try:
        result = _run_preflight_child(runtime_config)
        message: dict[str, Any] = {
            "ok": True,
            "result": result.to_wire(),
            "error": None,
        }
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"[:_MAX_CHILD_ERROR_CHARS]
        message = {"ok": False, "result": None, "error": detail}
    try:
        connection.send_bytes(_encode_child_message(message))
    except (BrokenPipeError, EOFError, OSError, ValueError):
        pass
    finally:
        connection.close()


def _run_preflight_child(runtime_config: PolicyRuntimeConfig) -> PreflightResult:
    from dexmani_real.deployment.worker import _load_inference_runtime

    runtime = _load_inference_runtime(runtime_config)
    try:
        timings = runtime.warmup(samples=1)
        if len(timings) != 1 or not np.isfinite(timings[0]) or timings[0] < 0.0:
            raise RuntimeError("policy preflight returned invalid warmup timing")
        prediction = runtime.predict(_synthetic_observation(runtime_config))
    finally:
        runtime.close()
    _validate_prediction(prediction, runtime_config)
    return PreflightResult(
        action_steps=runtime_config.deployment.n_action_steps,
        action_dim=runtime_config.deployment.control_action_dim,
    )


def _synthetic_observation(runtime_config: PolicyRuntimeConfig) -> ObservationBatch:
    deployment = runtime_config.deployment
    count = deployment.observation_horizon
    run_ns = 1_000_000_000
    source_ns = run_ns + np.arange(1, count + 1, dtype=np.uint64) * 1_000_000
    publish_ns = source_ns + 1
    sequence = np.arange(1, count + 1, dtype=np.uint64)
    valid = np.ones(count, dtype=np.uint8)
    arm = FrameWindow(
        values=np.zeros((count, _ARM_DOF), dtype=np.float64),
        source_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=publish_ns,
        valid_mask=valid,
    )
    hand = FrameWindow(
        values=np.zeros((count, _HAND_DOF), dtype=np.float64),
        source_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=publish_ns,
        valid_mask=valid,
    )
    points = np.zeros(
        (deployment.pointcloud_num_points, deployment.pointcloud_feature_dim),
        dtype=np.float32,
    )
    points[:, 0] = 0.4
    points[:, 3:] = 0.5
    clouds = tuple(
        PointCloudFrame(
            values=points,
            source_camera_sequence=int(index),
            source_monotonic_ns=int(source),
            publish_monotonic_ns=int(published),
            camera_generation=1,
        )
        for index, source, published in zip(
            sequence, source_ns, publish_ns, strict=True
        )
    )
    latest_source_ns = int(source_ns[-1])
    return ObservationBatch(
        observation_id=1,
        run_generation=1,
        run_started_monotonic_ns=run_ns,
        anchor_monotonic_ns=int(publish_ns[-1]) + 1_000_000,
        latest_source_monotonic_ns=latest_source_ns,
        logical_step_monotonic_ns=latest_source_ns,
        arm_history=arm,
        hand_history=hand,
        pointcloud=clouds[-1],
        pointcloud_history=clouds,
    )


def _validate_prediction(prediction: Any, runtime_config: PolicyRuntimeConfig) -> None:
    if not isinstance(prediction, PolicyPrediction):
        raise TypeError("synthetic preflight prediction must be a PolicyPrediction")
    deployment = runtime_config.deployment
    values = prediction.ee_pos if prediction.is_ee else prediction.arm_qpos
    if values is None or values.shape[0] != deployment.n_action_steps:
        raise ValueError("synthetic preflight prediction has an invalid chunk length")
    if prediction.hand_qpos is None or prediction.hand_qpos.shape != (
        deployment.n_action_steps,
        _HAND_DOF,
    ):
        raise ValueError("synthetic preflight prediction is missing hand12 actions")
    if prediction.is_ee != (deployment.action_key == "action_ee"):
        raise ValueError(
            "synthetic preflight prediction action space mismatches PolicySpec"
        )


def _terminate_join_kill_close(process: Any) -> None:
    try:
        if process.is_alive():
            process.terminate()
            process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
                process.join(timeout=_CHILD_JOIN_TIMEOUT_S)
        if process.is_alive():
            raise RuntimeError("policy preflight child could not be terminated")
    finally:
        if not process.is_alive():
            _close_process(process)


def _close_process(process: Any) -> None:
    close = getattr(process, "close", None)
    if callable(close):
        close()
