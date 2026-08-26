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

from collections.abc import Mapping
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


def _validate_training_data_contract(data_contract: Any, runtime_config: Any) -> None:
    """Fail closed unless checkpoint data matches the realtime observation path."""
    if not isinstance(data_contract, dict):
        raise ValueError(
            "checkpoint has no training data contract; retrain with Policy Zarr v4"
        )
    expected_task = getattr(runtime_config, "task_name", "")
    if not isinstance(expected_task, str) or not expected_task.strip():
        raise ValueError(
            "DexMani Policy deployment requires an explicit non-empty task_name"
        )
    expected = {
        "domain": "real",
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 4,
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "task_name": expected_task,
        "point_cloud_frame": getattr(runtime_config, "point_cloud_frame", ""),
        "point_cloud_color_source": getattr(
            runtime_config, "point_cloud_color_source", ""
        ),
        "point_cloud_policy_id": getattr(runtime_config, "point_cloud_policy_id", ""),
        "point_cloud_config_sha256": getattr(
            runtime_config, "point_cloud_config_sha256", ""
        ),
        "point_cloud_table_plane_abcd_json": getattr(
            runtime_config, "point_cloud_table_plane_abcd_json", ""
        ),
        "point_cloud_sampling": getattr(runtime_config, "point_cloud_sampling", ""),
        "point_cloud_transform": getattr(runtime_config, "point_cloud_transform", ""),
        "point_cloud_num_points": int(runtime_config.pointcloud_num_points),
        "point_cloud_feature_dim": 6,
    }
    mismatches = {
        key: (data_contract.get(key), value)
        for key, value in expected.items()
        if data_contract.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint training data does not match the realtime Real "
            f"observation contract: {mismatches}"
        )
    if data_contract.get("profile") not in {"pointcloud", "rgb_pc"}:
        raise ValueError("checkpoint training data must contain canonical point clouds")
    if set(data_contract.get("sensor_modalities", ())) != {
        "joint_state",
        "point_cloud",
    }:
        raise ValueError(
            "checkpoint training sensor modalities must be joint_state + point_cloud"
        )
    training_dt_s = data_contract.get("dt")
    runtime_dt_s = getattr(runtime_config, "control_dt_s", 0.0)
    if (
        isinstance(training_dt_s, bool)
        or not isinstance(training_dt_s, (int, float))
        or not np.isfinite(float(training_dt_s))
        or not np.isclose(
            float(training_dt_s),
            float(runtime_dt_s),
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            f"checkpoint training dt={training_dt_s!r} does not match "
            f"runtime control dt={runtime_dt_s!r}"
        )


def _expected_normalizer_dims(
    manifest: DeploymentManifest, agent: Any
) -> dict[str, int]:
    """Return model-space dimensions seen by each fitted normalizer."""
    joint_state_dim = _ARM_DOF + _HAND_DOF
    if manifest.use_faas:
        mapper = getattr(agent, "faas_mapper", None)
        mapped_hand_dim = getattr(mapper, "MAPPED_JOINT_DIM", None)
        if (
            isinstance(mapped_hand_dim, bool)
            or not isinstance(mapped_hand_dim, int)
            or mapped_hand_dim <= 0
        ):
            raise RuntimeError("FAAS agent has no valid mapped hand dimension")
        joint_state_dim = _ARM_DOF + mapped_hand_dim
    return {
        "action": manifest.action_dim,
        "joint_state": joint_state_dim,
        "point_cloud": manifest.point_cloud_feature_dim,
    }


class DexManiPolicyRuntime:
    """DexMani Policy runtime: lazy load -> predict_action -> chunk.

    ``load`` builds the agent from the checkpoint's embedded resolved config
    plus the deployment ``checkpoint`` / ``device`` values via Hydra + OmegaConf,
    without touching the dataset or ``env_runner`` (which imports
    ``dexmani_sim``).  A missing repository, config, or checkpoint fails closed
    (raises) — the supervisor observes a process failure, not a dummy safe mode.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self._agent: Any = None
        self._action_key: str = "action"
        self._device: str = "cpu"
        self._denoise_steps: int | None = None
        self._manifest: DeploymentManifest | None = None

    def load(self) -> None:
        checkpoint = getattr(self.config, "checkpoint", None)
        device = getattr(self.config, "device", "cpu")
        if not checkpoint:
            raise ValueError("DexMani Policy requires a self-describing checkpoint")
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
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint is not a file: {checkpoint_path}")
        store = CheckpointStore(checkpoint_path.parent)
        checkpoint_data = store.load(checkpoint_path)
        inference_config = getattr(checkpoint_data, "inference_config", None)
        if not isinstance(inference_config, dict):
            raise ValueError(
                "checkpoint has no embedded inference_config; retrain or export a "
                "self-describing deployment checkpoint"
            )
        required_train_params = {
            "n_obs_steps",
            "n_action_steps",
            "action_dim",
            "horizon",
            "action_key",
            "tcp_dim",
            "hand_dim",
            "use_faas",
            "control_action_dim",
        }
        if not isinstance(checkpoint_data.train_params, dict) or not (
            required_train_params <= checkpoint_data.train_params.keys()
        ):
            raise ValueError(
                "checkpoint train_params is missing the deployment manifest contract"
            )
        _validate_training_data_contract(
            getattr(checkpoint_data, "data_contract", None), self.config
        )
        cfg = OmegaConf.create(inference_config)
        normalize_action_key(cfg)
        # DQ-RISE codebook buffers are persistent model state. Loading the
        # training-time file again would make deployment depend on an external
        # path and could overwrite the checkpoint-owned artifact.
        if OmegaConf.select(cfg, "agent.codebook_path") is not None:
            OmegaConf.update(cfg, "agent.codebook_path", None)
        agent = hydra.utils.instantiate(cfg.agent)
        agent.action_key = cfg.action_key
        inject_faas_into_agent(agent, cfg)

        use_ema = _cfg_select(cfg, "eval.use_ema")
        if not isinstance(use_ema, bool):
            raise ValueError("embedded eval.use_ema must be a boolean")
        if use_ema and checkpoint_data.ema_model_state is None:
            raise ValueError(
                "embedded eval.use_ema requires EMA weights in the checkpoint"
            )
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

        required_normalizers = ["action", *manifest.sensor_modalities]
        if not agent.normalizer.is_fitted(required_keys=required_normalizers):
            missing = [
                key
                for key in required_normalizers
                if key not in agent.normalizer.params_dict
            ]
            raise RuntimeError(
                f"checkpoint normalizer is missing required field(s): {missing}"
            )
        expected_normalizer_dims = _expected_normalizer_dims(manifest, agent)
        for key in required_normalizers:
            params = agent.normalizer.params_dict[key]
            scale = params["scale"] if "scale" in params else None
            offset = params["offset"] if "offset" in params else None
            expected_dim = expected_normalizer_dims[key]
            if (
                scale is None
                or offset is None
                or int(scale.numel()) != expected_dim
                or int(offset.numel()) != expected_dim
                or not bool(torch.isfinite(scale).all())
                or not bool(torch.isfinite(offset).all())
                or bool(torch.any(scale == 0))
            ):
                raise RuntimeError(
                    f"normalizer {key!r} must contain finite non-zero scale and "
                    f"finite offset with {expected_dim} values"
                )

        denoise_steps = _cfg_select(cfg, "eval.denoise_steps")
        if isinstance(denoise_steps, bool) or not isinstance(denoise_steps, int):
            raise ValueError("model config eval.denoise_steps must be an integer")
        if denoise_steps <= 0:
            raise ValueError("model config eval.denoise_steps must be positive")

        self._agent = agent
        self._manifest = manifest
        self._action_key = manifest.action_key
        self._device = device
        self._denoise_steps = denoise_steps

    def reset_episode(self) -> None:
        # Diffusion/FlowMatch backbones have no recurrent state; the real-side
        # observation history reset is the inference worker's responsibility.
        pass

    def _encode(self, observation: ObservationBatch) -> dict[str, torch.Tensor]:
        """Build the model-native ``joint_state`` / ``point_cloud`` tensors.

        ``joint_state`` is ``[1, n_obs_steps, 19]`` (arm7 + hand12 concatenated,
        most-recent ``n_obs_steps`` frames, anchor last); ``point_cloud`` is
        ``[1, n_obs_steps, N, 6]`` built from the causal point-cloud history
        rather than a latest-frame broadcast. The inference worker selects each
        cloud causally on the policy control grid, then pairs arm/hand state to
        the selected cloud source time before this method runs.
        """
        if observation.arm_history is None:
            raise ValueError("DexMani policy requires arm joint history")
        if observation.hand_history is None:
            raise ValueError("DexMani joint policy requires hand joint history")
        if self._manifest is None:
            raise RuntimeError("deployment manifest is unavailable")
        n_obs = int(self._manifest.n_obs_steps)

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
        pc_frames = list(observation.pointcloud_history)
        if len(pc_frames) != n_obs:
            raise ValueError(
                f"need exactly {n_obs} causal point-cloud frames, got {len(pc_frames)}"
            )
        if len({frame.camera_generation for frame in pc_frames}) != 1:
            raise ValueError("point-cloud history crosses a camera generation boundary")
        pc = np.stack(
            [frame.values.astype(np.float32) for frame in pc_frames], axis=0
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
        coordinator resolves to joint space via IK.
        """
        if self._manifest is None:
            raise RuntimeError("deployment manifest is unavailable")
        if not isinstance(control_action, torch.Tensor):
            raise ValueError("control_action must be a torch.Tensor")
        expected_shape = (
            1,
            self._manifest.n_action_steps,
            self._manifest.control_action_dim,
        )
        if tuple(control_action.shape) != expected_shape:
            raise ValueError(
                f"control_action must have shape {expected_shape}, got "
                f"{tuple(control_action.shape)}"
            )
        if not bool(torch.isfinite(control_action).all()):
            raise ValueError("control_action contains NaN/Inf")
        ctrl = control_action.detach().cpu().numpy()[0]
        n = ctrl.shape[0]
        dim = ctrl.shape[1]
        steps = np.arange(n, dtype=np.uint64)
        target = np.asarray(
            context.observation_logical_step_monotonic_ns, dtype=np.uint64
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
        _validate_policy_rot6d(ee_rot6d)
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
        result = self._agent.predict_action(
            obs_dict,
            denoise_timesteps=self._denoise_steps,
        )
        if not isinstance(result, Mapping) or "control_action" not in result:
            raise ValueError(
                "DexMani agent must return a mapping containing 'control_action'"
            )
        return self._decode(result["control_action"], context=context)

    def close(self) -> None:
        agent = self._agent
        self._agent = None
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()


def _validate_policy_rot6d(rot6d: np.ndarray) -> None:
    """Reject rotations whose Gram-Schmidt projection is underdetermined."""
    values = np.asarray(rot6d, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != EE_ROT6D_DIM:
        raise ValueError(f"ee_rot6d must be [N, {EE_ROT6D_DIM}]")
    first = values[:, :3]
    second = values[:, 3:]
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    unit_first = first / np.maximum(first_norm[:, None], 1e-12)
    orthogonal_second = (
        second - np.sum(second * unit_first, axis=1)[:, None] * unit_first
    )
    orthogonal_norm = np.linalg.norm(orthogonal_second, axis=1)
    if (
        np.any(first_norm < 1e-4)
        or np.any(second_norm < 1e-4)
        or np.any(orthogonal_norm / np.maximum(second_norm, 1e-12) < 1e-4)
    ):
        raise ValueError("ee_rot6d contains a zero or collinear basis")
