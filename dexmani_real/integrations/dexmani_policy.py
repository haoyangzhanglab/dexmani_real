"""DexMani Policy integration.

Encapsulates the ``dexmani_policy`` model repository behind the three deployment
Protocols so ``deployment/*`` never imports it and the parent process /
loader never touches torch. ``dexmani_policy`` is imported lazily — inside
:meth:`DexManiPolicyBackend.load` — so the architecture gate holds: the
core runs end-to-end on the fake without the model repository installed.

This adapter currently supports native joint action. An EE-action
checkpoint without a validated EE->joint conversion is a startup reject. Any
model-internal representation (FAAS, latent hand, …) must be converted back to
native 12-DoF XHand by the model repository before this adapter sees it.

Expected ``dexmani_policy`` public API (the model repository must expose)::

    dexmani_policy.build_agent(model_config_path=..., checkpoint=..., device=...)
        -> agent with
            agent.action_space            # "joint" | "ee"
            agent.predict_action(obs)     # -> {"arm_qpos": [N,7], "hand_qpos": [N,12]|None}
            agent.reset()                 # clear recurrent/EMA state
            agent.close()

If the repository's entry point differs, only this module changes — never
``deployment/*``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.observation import ObservationBatch, parse_observation_fields
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_ARM_DOF = ARM_JOINT_SHAPE[0]
_HAND_DOF = HAND_JOINT_SHAPE[0]


def _last_valid(window: Any, dof: int) -> np.ndarray:
    """Return the last valid frame's values, or a zero vector when none/absent.

    Mirrors ``deployment.fake._last_valid``: the observation adapter never lets
    a missing window crash the backend — an absent/stale window becomes a zero
    vector so the model still sees a well-typed input. The hand is the one
    modality a joint-only deployment may genuinely omit (see ``encode``).
    """
    if window is None:
        return np.zeros(dof, dtype=np.float64)
    idx = np.flatnonzero(np.asarray(window.valid_mask).astype(bool))
    if idx.size == 0:
        return np.zeros(dof, dtype=np.float64)
    return np.asarray(window.values[idx[-1]], dtype=np.float64).reshape(dof)


def _last_valid_optional(window: Any) -> np.ndarray | None:
    """Return an optional modality's latest valid frame, or None."""
    if window is None:
        return None
    idx = np.flatnonzero(np.asarray(window.valid_mask).astype(bool))
    if idx.size == 0:
        return None
    return np.asarray(window.values[idx[-1]], dtype=np.float64).copy()


class DexManiObservationAdapter:
    """``ObservationBatch`` -> model-native joint observation dict.

    The default remains joint-only. ``observation_fields`` can opt into current
    and tactile modalities; all optional fields are omitted as ``None`` when no
    causally valid frame is available. Any history stacking, normalization,
    batch-dimension, or device transfer the real policy needs belongs here.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        spec = getattr(config, "observation_fields", "arm_qpos,hand_qpos")
        self.observation_fields = parse_observation_fields(spec)

    def encode(self, observation: ObservationBatch) -> dict[str, np.ndarray | None]:
        encoded: dict[str, np.ndarray | None] = {}
        for field in self.observation_fields:
            if field in {"arm_qpos", "arm_joint_position"}:
                encoded[field] = _last_valid(observation.arm_history, _ARM_DOF)
            elif field in {"hand_qpos", "hand_joint_position"}:
                encoded[field] = (
                    None
                    if observation.hand_history is None
                    else _last_valid(observation.hand_history, _HAND_DOF)
                )
            elif field in {"hand_current", "hand_joint_torque"}:
                # XHand exposes the vendor joint ``torque`` reading through
                # DexMani's established ``current`` state field; no unit or
                # scale conversion is applied at this boundary.
                encoded[field] = _last_valid_optional(observation.hand_current_history)
            elif field in {"hand_tactile_sum", "fingertip_force"}:
                encoded[field] = _last_valid_optional(observation.hand_tactile_sum_history)
            elif field in {"hand_tactile_force", "xhand_tactile"}:
                encoded[field] = _last_valid_optional(observation.tactile_history)
            else:  # pragma: no cover - parse_observation_fields rejects this first
                raise ValueError(f"unsupported observation field: {field}")
        return encoded


class DexManiPolicyBackend:
    """DexMani Policy model backend: lazy load -> predict_action.

    ``load`` imports ``dexmani_policy``, builds the agent from the resolved
    ``DeploymentConfig`` (``model_config_path`` / ``checkpoint`` / ``device``),
    and rejects an EE-action checkpoint. A missing repository or entry
    point fails closed (raises) — the supervisor observes a process failure, not
    a dummy safe mode.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._agent: Any = None

    def load(self) -> None:
        model_config_path = getattr(self.config, "model_config_path", None)
        checkpoint = getattr(self.config, "checkpoint", None)
        device = getattr(self.config, "device", "cpu")
        try:
            import dexmani_policy  # noqa: F401  (lazy; model repo import)
        except ImportError as exc:
            raise ImportError(
                "dexmani_policy is not installed; the DexMani Policy backend "
                "cannot load (fail closed)"
            ) from exc
        build_agent = getattr(dexmani_policy, "build_agent", None)
        if build_agent is None:
            raise ImportError(
                "dexmani_policy does not expose build_agent(model_config_path=..., "
                "checkpoint=..., device=...) — update this integration to the "
                "repository's entry point"
            )
        agent = build_agent(
            model_config_path=model_config_path,
            checkpoint=checkpoint,
            device=device,
        )
        if getattr(agent, "action_space", "joint") != "joint":
            raise ValueError(
                "EE-action checkpoint requires a validated EE->joint conversion; "
                "only native joint action is supported"
            )
        self._agent = agent

    def reset(self, *, run_generation: int) -> None:
        if self._agent is not None:
            self._agent.reset()

    def infer(self, model_input: Any) -> Any:
        if self._agent is None:
            raise RuntimeError("DexManiPolicyBackend.infer called before load()")
        return self._agent.predict_action(model_input)

    def close(self) -> None:
        agent = self._agent
        self._agent = None
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()


class DexManiActionAdapter:
    """Model-native output -> denormalize -> ``JointActionChunk``.

    This adapter supports native joint action only. The model output is expected as
    ``{"arm_qpos": [N,7], "hand_qpos": [N,12]|None}`` in radians (any model-side
    denormalization is the repository's job; this adapter only shapes it into the
    canonical chunk). An EE-shaped output fails closed rather than silently
    producing a bad chunk.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decode(self, raw_output: Any, *, context: InferenceContext) -> JointActionChunk:
        if not isinstance(raw_output, dict) or "arm_qpos" not in raw_output:
            raise ValueError(
                "DexMani Policy output must be a dict with 'arm_qpos' [N, 7] native "
                "joint (EE and non-dict outputs are unsupported)"
            )
        arm = np.asarray(raw_output["arm_qpos"], dtype=np.float64)
        if arm.ndim != 2 or arm.shape[1] != _ARM_DOF:
            raise ValueError(
                f"DexMani Policy arm output must be [N, {_ARM_DOF}] native joint, "
                f"got {arm.shape} (EE action is unsupported)"
            )
        n = arm.shape[0]
        if n == 0:
            raise ValueError("DexMani Policy produced an empty chunk")

        hand: np.ndarray | None = None
        hand_raw = raw_output.get("hand_qpos")
        if hand_raw is not None:
            hand = np.asarray(hand_raw, dtype=np.float64)
            if hand.shape != (n, _HAND_DOF):
                raise ValueError(
                    f"DexMani Policy hand output must be [N, {_HAND_DOF}] matching arm N={n}"
                )

        steps = np.arange(1, n + 1, dtype=np.uint64)
        target = np.asarray(context.inference_finished_monotonic_ns, dtype=np.uint64) + (
            steps * np.uint64(context.step_dt_ns)
        )
        return JointActionChunk(
            arm_qpos=arm,
            hand_qpos=hand,
            target_monotonic_ns=target,
            valid_mask=np.ones(n, dtype=np.uint8),
        )
