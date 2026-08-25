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
(pos3 + rot6d + hand12) space.  Joint action is decoded directly; an EE chunk
carries ``ee_pos``/``ee_rot6d`` that the coordinator resolves to joint space via
collision-aware IK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.manifest import (
    EE_CONTROL_ACTION_DIM,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    DeploymentManifest,
    manifest_from_sources,
    validate_manifest_against_deployment,
)
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_ARM_DOF = ARM_JOINT_SHAPE[0]
_HAND_DOF = HAND_JOINT_SHAPE[0]


def _cfg_select(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a resolved OmegaConf node at *path*, returning *default* when absent."""
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, path)
    return default if value is None else value


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
        self._manifest: DeploymentManifest | None = None

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
        from dexmani_policy.common.checkpoint_io import CheckpointStore
        from dexmani_policy.common.config import (
            normalize_action_key,
            register_resolvers,
        )
        from dexmani_policy.training.build_utils import inject_faas_into_agent
        from dexmani_policy.training.eval_utils import load_ckpt_for_inference
        from omegaconf import OmegaConf

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

        # Assemble the manifest from the loaded agent (already cross-validated
        # against the checkpoint train_params by load_ckpt_for_inference) plus
        # the model config sensor/point-cloud contract, then fail closed on any
        # deployment inconsistency before the run can start.
        manifest = manifest_from_sources(
            action_key=cfg.action_key,
            n_obs_steps=int(agent.n_obs_steps),
            n_action_steps=int(agent.n_action_steps),
            action_dim=int(agent.action_dim),
            horizon=int(agent.horizon),
            use_faas=bool(getattr(agent, "use_faas", False)),
            tcp_dim=getattr(agent, "tcp_dim", None),
            hand_dim=getattr(agent, "hand_dim", None),
            control_action_dim=int(agent.control_action_dim),
            sensor_modalities=_cfg_select(
                cfg,
                "dataset.sensor_modalities",
                _cfg_select(cfg, "sensor_modalities", ["joint_state", "point_cloud"]),
            ),
            point_cloud_num_points=_cfg_select(cfg, "agent.num_points"),
            point_cloud_feature_dim=_cfg_select(cfg, "agent.pc_dim"),
        )
        validate_manifest_against_deployment(manifest, self.config)

        self._agent = agent
        self._manifest = manifest
        self._action_key = manifest.action_key
        self._device = device

    def reset_episode(self) -> None:
        # Diffusion/FlowMatch backbones have no recurrent state; the real-side
        # observation history reset is the inference worker's responsibility.
        pass

    def _encode(self, observation: ObservationBatch) -> dict[str, torch.Tensor]:
        """Build the model-native ``joint_state`` / ``point_cloud`` tensors.

        ``joint_state`` is ``[1, n_obs_steps, 19]`` (arm7 + hand12 concatenated,
        most-recent ``n_obs_steps`` frames, anchor last); ``point_cloud`` is
        ``[1, n_obs_steps, N, 6]`` built from the causal point-cloud history
        (plan §6/§14.3.4) rather than a latest-frame broadcast. The inference
        worker uses the point-cloud source timestamps as the reference timeline
        and causally pairs each arm/hand frame before this method runs.
        """
        if observation.arm_history is None:
            raise ValueError("DexMani policy requires arm joint history")
        if observation.hand_history is None:
            raise ValueError("DexMani joint policy requires hand joint history")
        n_obs = int(self._manifest.n_obs_steps) if self._manifest is not None else 2

        arm = observation.arm_history.values  # [Ta, 7]
        hand = observation.hand_history.values  # [Th, 12]
        if arm.ndim != 2 or hand.ndim != 2:
            raise ValueError("arm/hand history must be [T, dof]")
        if arm.shape[0] != n_obs or hand.shape[0] != n_obs:
            raise ValueError(
                f"need exactly {n_obs} point-cloud-aligned arm/hand frames, got "
                f"arm={arm.shape[0]} hand={hand.shape[0]}"
            )
        joint = np.concatenate([arm, hand], axis=-1).astype(np.float32)  # [n_obs, 19]

        # Point-cloud history is a real oldest-first temporal window. Repeating
        # an old cloud would turn a missing observation into a plausible input,
        # so insufficient history fails closed.
        pc_frames = list(observation.pointcloud_history or ())
        if len(pc_frames) < n_obs:
            raise ValueError(
                f"need >= {n_obs} causal point-cloud frames, got {len(pc_frames)}"
            )
        use = pc_frames[-n_obs:]
        if len({frame.camera_generation for frame in use}) != 1:
            raise ValueError("point-cloud history crosses a camera generation boundary")
        pc = np.stack(
            [frame.values.astype(np.float32) for frame in use], axis=0
        )  # [n_obs, N, 6]

        joint_t = torch.as_tensor(joint, device=self._device).unsqueeze(0)
        pc_t = torch.as_tensor(pc, device=self._device).unsqueeze(0)
        return {"joint_state": joint_t, "point_cloud": pc_t}

    def _decode(
        self, control_action: torch.Tensor, *, context: InferenceContext
    ) -> JointActionChunk:
        """Decode the native ``control_action`` into a ``JointActionChunk``.

        Joint (``action``): ``[H,19]`` = arm7 + hand12.  EE (``action_ee``):
        ``[H,21]`` = pos3 + rot6d6 + hand12, emitted as an EE chunk the
        coordinator resolves to joint space via IK (plan §14.2 decision 3).
        """
        ctrl = control_action.detach().cpu().numpy()[0]  # [H, control_action_dim]
        n = ctrl.shape[0]
        dim = ctrl.shape[1]
        steps = np.arange(1, n + 1, dtype=np.uint64)
        target = np.asarray(
            context.observation_anchor_monotonic_ns, dtype=np.uint64
        ) + steps * np.uint64(context.step_dt_ns)
        if self._action_key == "action":
            if dim != _ARM_DOF + _HAND_DOF:
                raise ValueError(
                    f"DexMani joint action must be {_ARM_DOF + _HAND_DOF}-DoF "
                    f"(arm{_ARM_DOF}+hand{_HAND_DOF}), got {dim}"
                )
            arm = ctrl[:, :_ARM_DOF]
            hand = ctrl[:, _ARM_DOF:]
            return JointActionChunk(
                arm_qpos=np.asarray(arm, dtype=np.float64),
                hand_qpos=np.asarray(hand, dtype=np.float64),
                target_monotonic_ns=target,
                valid_mask=np.ones(n, dtype=np.uint8),
            )
        if dim != EE_CONTROL_ACTION_DIM:
            raise ValueError(
                f"DexMani EE action must be {EE_CONTROL_ACTION_DIM}-DoF "
                f"(pos{EE_POS_DIM}+rot6d{EE_ROT6D_DIM}+hand{_HAND_DOF}), got {dim}"
            )
        ee_pos = ctrl[:, :EE_POS_DIM]
        ee_rot6d = ctrl[:, EE_POS_DIM : EE_POS_DIM + EE_ROT6D_DIM]
        hand = ctrl[:, EE_POS_DIM + EE_ROT6D_DIM :]
        return JointActionChunk(
            arm_qpos=None,
            hand_qpos=np.asarray(hand, dtype=np.float64),
            target_monotonic_ns=target,
            valid_mask=np.ones(n, dtype=np.uint8),
            ee_pos=np.asarray(ee_pos, dtype=np.float64),
            ee_rot6d=np.asarray(ee_rot6d, dtype=np.float64),
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
