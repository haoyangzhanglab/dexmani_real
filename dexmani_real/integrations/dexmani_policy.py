"""DexMani Policy integration — a ``PolicyRuntime`` over the ``dexmani_policy``
model repository.

Encapsulates the ``dexmani_policy`` repo behind the single
:class:`~dexmani_real.deployment.contracts.PolicyRuntime` Protocol so
``deployment/*`` never imports it and the parent process / loader never touches
torch. ``dexmani_policy`` is imported lazily — inside
:meth:`DexManiPolicyRuntime.load` — so the architecture gate holds: the core
runs end-to-end on the fake without the model repository installed.

The real ``dexmani_policy`` inference API is ``agent.predict_action(obs_dict)``
returning ``{"pred_action", "control_action", "tail"}``; the actionable slice is
``control_action`` in native 19-DoF joint (arm7 + hand12) or 21-DoF EE
(pos3 + rot6d + hand12) space.  Only native joint action is decoded here; an
EE checkpoint requires an EE->joint conversion (IK) that lives in the
coordinator and is wired in a later phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.utils.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_ARM_DOF = ARM_JOINT_SHAPE[0]
_HAND_DOF = HAND_JOINT_SHAPE[0]


class DexManiPolicyRuntime:
    """DexMani Policy runtime: lazy load -> predict_action -> chunk.

    ``load`` builds the agent from the resolved ``DeploymentConfig``
    (``model_config_path`` / ``checkpoint`` / ``device``) via Hydra + OmegaConf,
    without touching the dataset or ``env_runner`` (which imports
    ``dexmani_sim``).  A missing repository, config, or checkpoint fails closed
    (raises) — the supervisor observes a process failure, not a dummy safe mode.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._agent: Any = None
        self._action_key: str = "action"
        self._device: str = "cpu"

    def load(self) -> None:
        model_config_path = getattr(self.config, "model_config_path", None)
        checkpoint = getattr(self.config, "checkpoint", None)
        device = getattr(self.config, "device", "cpu")
        if not model_config_path or not checkpoint:
            raise ValueError(
                "DexMani Policy requires both model_config_path and checkpoint"
            )
        try:
            import dexmani_policy  # noqa: F401  (lazy; model repo import)
        except ImportError as exc:
            raise ImportError(
                "dexmani_policy is not installed; the DexMani Policy runtime "
                "cannot load (fail closed)"
            ) from exc

        import hydra
        from omegaconf import OmegaConf

        from dexmani_policy.common.checkpoint_io import CheckpointStore
        from dexmani_policy.common.config import normalize_action_key, register_resolvers
        from dexmani_policy.training.build_utils import inject_faas_into_agent
        from dexmani_policy.training.eval_utils import load_ckpt_for_inference

        register_resolvers()
        cfg = OmegaConf.load(model_config_path)
        normalize_action_key(cfg)
        agent = hydra.utils.instantiate(cfg.agent)
        agent.action_key = cfg.action_key
        inject_faas_into_agent(agent, cfg)

        checkpoint_path = Path(checkpoint)
        store = CheckpointStore(checkpoint_path.parent)
        use_ema = bool(getattr(self.config, "use_ema", True))
        load_ckpt_for_inference(agent, store, checkpoint_path, use_ema)

        agent.to(device)
        agent.eval()
        self._agent = agent
        self._action_key = cfg.action_key
        self._device = device

    def reset_episode(self) -> None:
        # Diffusion/FlowMatch backbones have no recurrent state; the real-side
        # observation history reset is the inference worker's responsibility.
        pass

    def _encode(self, observation: ObservationBatch) -> dict[str, torch.Tensor]:
        """Build the model-native ``joint_state`` / ``point_cloud`` tensors.

        ``joint_state`` is ``[1, T, 19]`` (arm7 + hand12 concatenated);
        ``point_cloud`` is ``[1, T, N, 6]``.  The point cloud is currently the
        latest causally valid frame broadcast across the observation horizon;
        a true per-step point-cloud history is a later-phase refinement.
        """
        if observation.arm_history is None:
            raise ValueError("DexMani policy requires arm joint history")
        if observation.hand_history is None:
            raise ValueError("DexMani joint policy requires hand joint history")
        if observation.pointcloud is None:
            raise ValueError("DexMani policy requires a point cloud")

        arm = observation.arm_history.values  # [T, 7]
        hand = observation.hand_history.values  # [T, 12]
        joint = np.concatenate([arm, hand], axis=-1).astype(np.float32)  # [T, 19]
        t = joint.shape[0]

        pc = observation.pointcloud.values.astype(np.float32)  # [N, 6]
        pc_t = np.tile(pc[None, None, :, :], (1, t, 1, 1))  # [1, T, N, 6]

        joint_t = torch.as_tensor(joint, device=self._device).unsqueeze(0)
        pc_t = torch.as_tensor(pc_t, device=self._device)
        return {"joint_state": joint_t, "point_cloud": pc_t}

    def _decode(
        self, control_action: torch.Tensor, *, context: InferenceContext
    ) -> JointActionChunk:
        """Decode the native ``control_action`` into a ``JointActionChunk``."""
        ctrl = control_action.detach().cpu().numpy()[0]  # [H, control_action_dim]
        n = ctrl.shape[0]
        dim = ctrl.shape[1]
        if self._action_key != "action":
            raise ValueError(
                f"DexMani Policy action_key={self._action_key!r} requires an "
                "EE->joint conversion (IK) that is not yet wired"
            )
        if dim != _ARM_DOF + _HAND_DOF:
            raise ValueError(
                f"DexMani joint action must be {_ARM_DOF + _HAND_DOF}-DoF "
                f"(arm{_ARM_DOF}+hand{_HAND_DOF}), got {dim}"
            )
        arm = ctrl[:, :_ARM_DOF]
        hand = ctrl[:, _ARM_DOF:]
        steps = np.arange(1, n + 1, dtype=np.uint64)
        target = (
            np.asarray(context.observation_anchor_monotonic_ns, dtype=np.uint64)
            + steps * np.uint64(context.step_dt_ns)
        )
        return JointActionChunk(
            arm_qpos=np.asarray(arm, dtype=np.float64),
            hand_qpos=np.asarray(hand, dtype=np.float64),
            target_monotonic_ns=target,
            valid_mask=np.ones(n, dtype=np.uint8),
        )

    def predict(
        self, observation: ObservationBatch, *, context: InferenceContext
    ) -> JointActionChunk:
        if self._agent is None:
            raise RuntimeError("DexManiPolicyRuntime.predict called before load()")
        obs_dict = self._encode(observation)
        result = self._agent.predict_action(obs_dict)
        return self._decode(result["control_action"], context=context)

    def close(self) -> None:
        agent = self._agent
        self._agent = None
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()
