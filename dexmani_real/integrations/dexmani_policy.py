"""DexMani Policy integration (execution doc §86–§91).

Encapsulates the ``dexmani_policy`` model repository behind the three deployment
Protocols so ``deployment/*`` never imports it (§86) and the parent process /
loader never touches torch. ``dexmani_policy`` is imported lazily — inside
:meth:`DexManiPolicyBackend.load` — so the architecture gate (§66) holds: the
core runs end-to-end on the fake without the model repository installed.

First version (execution doc §90): only native joint action. An EE-action
checkpoint without a validated EE->joint conversion is a startup reject. Any
model-internal representation (FAAS, latent hand, …) must be converted back to
native 12-DoF XHand by the model repository before this adapter sees it (§91).

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
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_ARM_DOF = ARM_JOINT_SHAPE[0]
_HAND_DOF = HAND_JOINT_SHAPE[0]


def _last_valid(window: Any, dof: int) -> np.ndarray:
    """Return the last valid frame's values, or a zero vector when none/absent.

    Mirrors ``deployment.fake._last_valid``: the observation adapter never lets
    a missing window crash the backend — an absent/stale window becomes a zero
    vector so the model still sees a well-typed input (§54). The hand is the one
    modality a joint-only first version may genuinely omit (see ``encode``).
    """
    if window is None:
        return np.zeros(dof, dtype=np.float64)
    idx = np.flatnonzero(np.asarray(window.valid_mask).astype(bool))
    if idx.size == 0:
        return np.zeros(dof, dtype=np.float64)
    return np.asarray(window.values[idx[-1]], dtype=np.float64).reshape(dof)


class DexManiObservationAdapter:
    """``ObservationBatch`` -> model-native joint observation dict (§88).

    Joint-only first version. ``arm_qpos`` is always a ``[7]`` vector (a zero
    vector when the arm window is absent/stale); ``hand_qpos`` is ``[12]`` when
    the hand window is present, else ``None``. Any history stacking,
    normalization, batch-dimension, or device transfer the real policy needs
    belongs here, not in the deployment core (§88/§91).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def encode(self, observation: ObservationBatch) -> dict[str, np.ndarray | None]:
        arm = _last_valid(observation.arm_history, _ARM_DOF)
        hand: np.ndarray | None = None
        if observation.hand_history is not None:
            hand = _last_valid(observation.hand_history, _HAND_DOF)
        return {"arm_qpos": arm, "hand_qpos": hand}


class DexManiPolicyBackend:
    """DexMani Policy model backend (§87): lazy load -> predict_action.

    ``load`` imports ``dexmani_policy``, builds the agent from the resolved
    ``DeploymentConfig`` (``model_config_path`` / ``checkpoint`` / ``device``),
    and rejects an EE-action checkpoint (§90). A missing repository or entry
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
                "cannot load (fail closed, §81)"
            ) from exc
        build_agent = getattr(dexmani_policy, "build_agent", None)
        if build_agent is None:
            raise ImportError(
                "dexmani_policy does not expose build_agent(model_config_path=..., "
                "checkpoint=..., device=...) — update this integration to the "
                "repository's entry point (§86)"
            )
        agent = build_agent(
            model_config_path=model_config_path,
            checkpoint=checkpoint,
            device=device,
        )
        if getattr(agent, "action_space", "joint") != "joint":
            raise ValueError(
                "EE-action checkpoint requires a validated EE->joint conversion; "
                "first version only supports native joint action (§90)"
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
    """Model-native output -> denormalize -> ``JointActionChunk`` (§89).

    First version only native joint action. The model output is expected as
    ``{"arm_qpos": [N,7], "hand_qpos": [N,12]|None}`` in radians (any model-side
    denormalization is the repository's job; this adapter only shapes it into the
    canonical chunk). An EE-shaped output fails closed rather than silently
    producing a bad chunk (§90).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    def decode(self, raw_output: Any, *, context: InferenceContext) -> JointActionChunk:
        if not isinstance(raw_output, dict) or "arm_qpos" not in raw_output:
            raise ValueError(
                "DexMani Policy output must be a dict with 'arm_qpos' [N, 7] native "
                "joint (§89/§90: EE and non-dict outputs are unsupported)"
            )
        arm = np.asarray(raw_output["arm_qpos"], dtype=np.float64)
        if arm.ndim != 2 or arm.shape[1] != _ARM_DOF:
            raise ValueError(
                f"DexMani Policy arm output must be [N, {_ARM_DOF}] native joint, "
                f"got {arm.shape} (§90: EE action is unsupported)"
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
