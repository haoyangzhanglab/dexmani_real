"""DexMani Policy integration — a ``PolicyRuntime`` over the ``dexmani_policy``
model repository.

Encapsulates the ``dexmani_policy`` repo behind the single
:class:`~dexmani_real.deployment.contracts.PolicyRuntime` Protocol so the
deployment parent process and loader do not import this module or touch torch.
This integration module itself imports torch when it is loaded in the inference
worker. The external ``dexmani_policy`` repo is imported lazily — inside
:meth:`DexManiPolicyRuntime.load` — so the architecture gate holds: the core
runs end-to-end on the fake without the model repository installed.

The real ``dexmani_policy`` inference API is ``agent.predict_action(obs_dict)``
returning a full ``pred_action`` horizon.  This adapter selects the complete
future interval after the observation window in native 19-DoF joint (arm7 +
hand12) or 21-DoF EE (pos3 + rot6d + hand12) space.  Joint action is decoded
directly; an EE prediction carries ``ee_pos``/``ee_rot6d`` that the coordinator
later resolves to joint space via collision-aware IK.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dexmani_real.deployment.config import PolicyRuntimeConfig
from dexmani_real.deployment.contracts import PolicyPrediction
from dexmani_real.deployment.manifest import (
    EE_CONTROL_ACTION_DIM,
    EE_POS_DIM,
    EE_ROT6D_DIM,
    DeploymentManifest,
    manifest_from_sources,
    validate_manifest_against_deployment,
)
from dexmani_real.deployment.observation import ObservationBatch
from dexmani_real.planning.poses import validate_rot6d_geometry
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
            "checkpoint has no training data contract; retrain with Policy Zarr v5"
        )
    expected_task = getattr(runtime_config, "task_name", "")
    if not isinstance(expected_task, str) or not expected_task.strip():
        raise ValueError(
            "DexMani Policy deployment requires an explicit non-empty task_name"
        )
    expected = {
        "domain": "real",
        "schema_name": "dexmani-real-policy-zarr",
        "schema_version": 5,
        "episode_start_policy": "full_history",
        "obs_alignment": "obs[t]_before_action[t]",
        "observation_reference": "camera_source_monotonic_ns",
        "state_alignment": "camera_source_aligned_state",
        "max_observation_skew_s": float(
            getattr(runtime_config, "max_observation_skew_s", 0.0)
        ),
        "action_semantics": "deployment_grid_rate_limited_target",
        "arm_max_delta_rad_per_tick": getattr(
            runtime_config, "arm_max_delta_rad_per_tick", None
        ),
        "hand_max_delta_rad_per_tick": float(
            getattr(runtime_config, "hand_max_delta_rad_per_tick", 0.0)
        ),
        "deployment_equivalent": True,
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
    expected["endpoint_delta_tolerance_rad"] = (
        runtime_config.endpoint_delta_tolerance_rad
    )
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


def _expected_normalizer_dims(manifest: DeploymentManifest) -> dict[str, int]:
    """Return model-space dimensions seen by each fitted normalizer."""
    return {
        "action": manifest.action_dim,
        "joint_state": _ARM_DOF + _HAND_DOF,
        "point_cloud": manifest.point_cloud_feature_dim,
    }


def _validate_embedded_targets(value: Any, *, path: str = "") -> None:
    """Require each embedded Hydra target to be an inference-agent target."""
    if isinstance(value, Mapping):
        target = value.get("_target_")
        if target is not None:
            if not isinstance(target, str) or not target.startswith(
                "dexmani_policy.agents."
            ):
                raise ValueError(
                    f"embedded _target_ at {path or '<root>'} must be under "
                    "dexmani_policy.agents"
                )
        for key, nested in value.items():
            _validate_embedded_targets(
                nested, path=f"{path}.{key}" if path else str(key)
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _validate_embedded_targets(nested, path=f"{path}[{index}]")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("embedded deployment contract must be finite JSON") from exc


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _canonical_python_tree_sha256(package_root: Path) -> str:
    """Hash the installed package's Python source tree in stable path order."""
    digest = hashlib.sha256()
    try:
        paths = sorted(package_root.rglob("*.py"), key=lambda path: path.as_posix())
        if not paths:
            raise ValueError("dexmani_policy package has no Python sources")
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ValueError("dexmani_policy Python source must be regular files")
            relative = path.relative_to(package_root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            payload = path.read_bytes()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
    except OSError as exc:
        raise ValueError("cannot hash dexmani_policy Python source tree") from exc
    return digest.hexdigest()


def _git_package_provenance(package_root: Path) -> tuple[str, str]:
    """Return editable repository commit and dirty marker, if available."""
    repository_root = package_root.parent
    try:
        root = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if root.returncode != 0:
            return "", "unknown"
        git_root = root.stdout.strip()
        head = subprocess.run(
            ["git", "-C", git_root, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        commit = head.stdout.strip()
        if head.returncode != 0 or len(commit) != 40:
            return "", "unknown"
        status = subprocess.run(
            ["git", "-C", git_root, "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return commit, (
            "true" if status.returncode == 0 and status.stdout.strip() else "false"
        )
    except (OSError, subprocess.SubprocessError):
        return "", "unknown"


def precheck_policy_package_provenance(
    runtime_config: PolicyRuntimeConfig,
) -> dict[str, str]:
    """Bind the not-yet-imported Policy package to its producer commit.

    Editable source trees expose ``git HEAD``; wheels can expose a PEP 610
    ``direct_url.json`` commit.  An experiment directory is never an import
    source.  This uses only the standard library and refuses a process where a
    Policy module was already imported, so experiment code cannot run before
    the origin/commit gate.
    """
    artifact = runtime_config.artifact
    if artifact is None:
        raise ValueError("Policy provenance requires a resolved artifact")
    preloaded = sorted(
        name
        for name in sys.modules
        if name == "dexmani_policy" or name.startswith("dexmani_policy.")
    )
    if preloaded:
        raise ValueError("dexmani_policy must not be imported before provenance gate")
    spec = importlib.util.find_spec("dexmani_policy")
    if spec is None or not isinstance(spec.origin, str):
        raise ImportError("dexmani_policy is not installed")
    raw_origin = Path(spec.origin).absolute()
    experiment_root = artifact.experiment_dir.resolve(strict=True)
    if _is_under(raw_origin, experiment_root):
        raise ValueError(
            "dexmani_policy import origin must not be inside experiment_dir"
        )
    try:
        origin = raw_origin.resolve(strict=True)
    except OSError as exc:
        raise ValueError("installed dexmani_policy has no import origin") from exc
    if _is_under(origin, experiment_root):
        raise ValueError(
            "dexmani_policy import origin must not be inside experiment_dir"
        )
    package_root = origin.parent
    source_tree_sha256 = _canonical_python_tree_sha256(package_root)
    commit, dirty = _git_package_provenance(package_root)
    if not commit:
        try:
            dist = importlib.metadata.distribution("dexmani-policy")
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                vcs = json.loads(direct_url).get("vcs_info", {})
                candidate = vcs.get("commit_id")
                if isinstance(candidate, str) and len(candidate) == 40:
                    commit = candidate
            version = dist.version
        except Exception as exc:
            raise ValueError(
                "cannot establish dexmani_policy package provenance"
            ) from exc
    else:
        try:
            version = importlib.metadata.version("dexmani-policy")
        except Exception:
            version = "unknown"
    if commit != artifact.producer.commit:
        raise ValueError(
            "installed dexmani_policy commit does not match artifact producer: "
            f"installed={commit!r}, artifact={artifact.producer.commit!r}"
        )
    return {
        "origin": str(origin),
        "commit": commit,
        "dirty": dirty,
        "source_tree_sha256": source_tree_sha256,
        "version": version,
    }


def verify_imported_policy_provenance(provenance: Mapping[str, str]) -> None:
    """Recheck that the post-gate import used the exact verified package root."""
    module = sys.modules.get("dexmani_policy")
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise ValueError("dexmani_policy import did not produce a package origin")
    try:
        imported_origin = Path(module_file).resolve(strict=True)
    except OSError as exc:
        raise ValueError("dexmani_policy import origin cannot be inspected") from exc
    if str(imported_origin) != provenance.get("origin"):
        raise ValueError("dexmani_policy import origin changed after provenance gate")
    if _canonical_python_tree_sha256(imported_origin.parent) != provenance.get(
        "source_tree_sha256"
    ):
        raise ValueError("dexmani_policy Python source changed after provenance gate")


class DexManiPolicyRuntime:
    """DexMani Policy runtime: lazy load -> predict_action -> chunk.

    ``load_loaded_checkpoint`` builds the agent from the checkpoint's embedded
    resolved config and deployment device via Hydra + OmegaConf, without
    touching the dataset or ``env_runner`` (which imports
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
        """Reject path-based loads; preflight owns exactly one checked stream."""
        raise RuntimeError(
            "DexManiPolicyRuntime.load() is disabled for deployment; use "
            "load_loaded_checkpoint() after the preflight stream checks"
        )

    def load_loaded_checkpoint(
        self, checkpoint_data: Any, *, package_provenance: Mapping[str, str]
    ) -> dict[str, str]:
        """Construct and restore from one already-deserialized checkpoint object."""
        if not isinstance(self.config, PolicyRuntimeConfig):
            raise TypeError("DexManiPolicyRuntime requires PolicyRuntimeConfig")
        if self.config.artifact is None:
            raise ValueError("DexManiPolicyRuntime requires an artifact-bound config")
        device = self.config.device
        try:
            import dexmani_policy  # noqa: F401  (lazy; model repo import)
        except ImportError as exc:
            raise ImportError(
                "dexmani_policy is not installed; the DexMani Policy runtime "
                "cannot load (fail closed)"
            ) from exc
        verify_imported_policy_provenance(package_provenance)

        import hydra
        from dexmani_policy.common.checkpoint_io import validate_new_checkpoint_metadata
        from dexmani_policy.common.config import (
            normalize_action_key,
            register_resolvers,
        )
        from dexmani_policy.datasets.data_contract import (
            validate_deployable_data_contract,
        )
        from dexmani_policy.training.eval_utils import (
            restore_loaded_checkpoint_for_inference,
        )
        from omegaconf import OmegaConf

        register_resolvers()
        inference_config = getattr(checkpoint_data, "inference_config", None)
        if not isinstance(inference_config, dict):
            raise ValueError(
                "checkpoint has no embedded inference_config; retrain or export a "
                "self-describing deployment checkpoint"
            )
        if not isinstance(checkpoint_data.train_params, dict):
            raise ValueError("checkpoint train_params is missing")
        if not isinstance(checkpoint_data.data_contract, dict):
            raise ValueError("checkpoint data_contract is missing")
        _validate_embedded_targets(inference_config)
        validate_new_checkpoint_metadata(
            inference_config=inference_config,
            data_contract=checkpoint_data.data_contract,
            train_params=checkpoint_data.train_params,
        )
        validate_deployable_data_contract(checkpoint_data.data_contract)
        self._validate_artifact_receipt(checkpoint_data)
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

        use_ema = _cfg_select(cfg, "eval.use_ema")
        if not isinstance(use_ema, bool):
            raise ValueError("embedded eval.use_ema must be a boolean")
        if use_ema and checkpoint_data.ema_model_state is None:
            raise ValueError(
                "embedded eval.use_ema requires EMA weights in the checkpoint"
            )
        restore_loaded_checkpoint_for_inference(agent, checkpoint_data, use_ema)

        agent.to(device)
        agent.eval()

        # Assemble the manifest from the restored agent plus the model
        # config/sensor contract, then fail closed on any deployment
        # inconsistency before the run can start.
        manifest = manifest_from_sources(
            action_key=cfg.action_key,
            n_obs_steps=int(agent.n_obs_steps),
            n_action_steps=int(agent.n_action_steps),
            action_dim=int(agent.action_dim),
            horizon=int(agent.horizon),
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
        expected_normalizer_dims = _expected_normalizer_dims(manifest)
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
        return dict(package_provenance)

    def _validate_artifact_receipt(self, checkpoint_data: Any) -> None:
        """Cross-check embedded receipt against the fixed sidecar allocation."""
        artifact = self.config.artifact
        assert artifact is not None
        deployment_contract = getattr(checkpoint_data, "deployment_contract", None)
        producer = getattr(checkpoint_data, "producer", None)
        if not isinstance(deployment_contract, Mapping):
            raise ValueError("checkpoint deployment_contract is missing")
        expected_contract_keys = {
            "schema_version",
            "inference_config",
            "data_contract",
            "train_params",
            "producer",
            "retrofitted_train_params_fields",
        }
        if set(deployment_contract) != expected_contract_keys:
            raise ValueError("checkpoint deployment_contract has an unsupported schema")
        if (
            type(deployment_contract.get("schema_version")) is not int
            or deployment_contract["schema_version"] != 1
        ):
            raise ValueError(
                "checkpoint deployment_contract has an unsupported version"
            )
        for name in ("inference_config", "data_contract", "train_params", "producer"):
            embedded = deployment_contract.get(name)
            checkpoint_value = getattr(checkpoint_data, name, None)
            if not isinstance(embedded, Mapping) or not isinstance(
                checkpoint_value, Mapping
            ):
                raise ValueError(f"checkpoint deployment_contract.{name} is missing")
            if _canonical_json(embedded) != _canonical_json(checkpoint_value):
                raise ValueError(
                    f"checkpoint deployment_contract.{name} does not exactly match checkpoint"
                )
        retrofitted = deployment_contract.get("retrofitted_train_params_fields")
        producer_retrofitted = getattr(checkpoint_data, "producer", {}).get(
            "retrofitted_train_params_fields"
        )
        if (
            not isinstance(retrofitted, list)
            or not isinstance(producer_retrofitted, list)
            or any(not isinstance(name, str) for name in retrofitted)
        ):
            raise ValueError(
                "checkpoint deployment_contract retrofitted fields are invalid"
            )
        if _canonical_json(retrofitted) != _canonical_json(producer_retrofitted):
            raise ValueError(
                "checkpoint deployment_contract retrofitted fields do not match producer"
            )
        if (
            _canonical_json_sha256(deployment_contract)
            != artifact.embedded_contract_sha256
        ):
            raise ValueError(
                "checkpoint deployment_contract SHA-256 mismatches sidecar"
            )
        if not isinstance(producer, Mapping):
            raise ValueError("checkpoint producer is missing")
        expected_producer = {
            "repository": artifact.producer.repository,
            "commit": artifact.producer.commit,
            "metadata_provenance": artifact.producer.metadata_provenance,
        }
        for name, expected in expected_producer.items():
            if producer.get(name) != expected:
                raise ValueError(f"checkpoint producer.{name} mismatches sidecar")
        embedded_producer = deployment_contract.get("producer")
        if not isinstance(embedded_producer, Mapping):
            raise ValueError("deployment_contract producer is missing")
        for name, expected in expected_producer.items():
            if embedded_producer.get(name) != expected:
                raise ValueError(
                    f"deployment_contract producer.{name} mismatches sidecar"
                )
        allocation = artifact.allocation_contract
        train = checkpoint_data.train_params
        data = checkpoint_data.data_contract
        inference = checkpoint_data.inference_config
        if inference.get("task_name") != allocation.task_name:
            raise ValueError("checkpoint task_name mismatches sidecar allocation")
        expected = {
            "action_key": allocation.action_key,
            "action_dim": allocation.action_dim,
            "n_obs_steps": allocation.n_obs_steps,
            "n_action_steps": allocation.n_action_steps,
            "horizon": allocation.horizon,
        }
        for name, value in expected.items():
            if inference.get(name) != value or train.get(name) != value:
                raise ValueError(f"checkpoint {name} mismatches sidecar allocation")
        if allocation.required_action_steps != train["horizon"] - (
            train["n_obs_steps"] - 1
        ):
            raise ValueError(
                "checkpoint required action window mismatches sidecar allocation"
            )
        if data.get("task_name") != allocation.task_name:
            raise ValueError("checkpoint data_contract.task_name mismatches sidecar")
        if data.get("action_key") != allocation.action_key:
            raise ValueError("checkpoint data_contract.action_key mismatches sidecar")
        for name in ("n_obs_steps", "n_action_steps", "horizon"):
            if data.get(name) != getattr(allocation, name):
                raise ValueError(f"checkpoint data_contract.{name} mismatches sidecar")
        if data.get("model_action_dim") != allocation.action_dim:
            raise ValueError(
                "checkpoint data_contract.model_action_dim mismatches sidecar"
            )
        if data.get("dt") != allocation.control_dt_s:
            raise ValueError("checkpoint data_contract.dt mismatches sidecar")
        if data.get("sensor_modalities") != list(allocation.sensor_modalities):
            raise ValueError("checkpoint data_contract sensor modalities mismatch")
        if data.get("point_cloud_num_points") != allocation.point_cloud_num_points:
            raise ValueError("checkpoint data_contract point count mismatch")
        if data.get("point_cloud_feature_dim") != allocation.point_cloud_feature_dim:
            raise ValueError(
                "checkpoint data_contract point feature dimension mismatch"
            )
        if (
            data.get("pad_before") != 1
            or data.get("pad_after") != allocation.n_action_steps - 1
            or data.get("padding_semantics") != "repeat_edge"
            or data.get("use_aux_ee") is not False
            or train.get("use_aux_ee") is not False
            or inference.get("use_aux_ee") is not False
        ):
            raise ValueError(
                "checkpoint padding or auxiliary-action contract is invalid"
            )

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

    def _decode(self, pred_action: torch.Tensor) -> PolicyPrediction:
        """Decode the full native prediction into an untimed future interval.

        Joint (``action``): ``[N,19]`` = arm7 + hand12.  EE (``action_ee``):
        ``[N,21]`` = pos3 + rot6d6 + hand12.  ``N`` is the artifact-authorized
        complete future window, never the model's shorter scheduler chunk.
        """
        if self._manifest is None:
            raise RuntimeError("deployment manifest is unavailable")
        if (
            not isinstance(self.config, PolicyRuntimeConfig)
            or self.config.artifact is None
        ):
            raise RuntimeError("deployment artifact is unavailable")
        if not isinstance(pred_action, torch.Tensor):
            raise ValueError("pred_action must be a torch.Tensor")
        expected_shape = (
            1,
            self._manifest.horizon,
            self._manifest.action_dim,
        )
        if tuple(pred_action.shape) != expected_shape:
            raise ValueError(
                f"pred_action must have shape {expected_shape}, got "
                f"{tuple(pred_action.shape)}"
            )
        if not bool(torch.isfinite(pred_action).all()):
            raise ValueError("pred_action contains NaN/Inf")
        start = self._manifest.n_obs_steps - 1
        required_steps = self.config.artifact.allocation_contract.required_action_steps
        if self._manifest.horizon - start != required_steps:
            raise ValueError("artifact required action window mismatches model horizon")
        future = pred_action[
            :,
            self._manifest.n_obs_steps - 1 : self._manifest.horizon,
            : self._manifest.control_action_dim,
        ]
        expected_future_shape = (1, required_steps, self._manifest.control_action_dim)
        if tuple(future.shape) != expected_future_shape:
            raise ValueError(
                f"pred_action future interval must have shape {expected_future_shape}, got "
                f"{tuple(future.shape)}"
            )
        ctrl = future.detach().cpu().numpy()[0]
        n = ctrl.shape[0]
        dim = ctrl.shape[1]
        if self._action_key == "action":
            if dim != _ARM_DOF + _HAND_DOF:
                raise ValueError(
                    f"DexMani joint action must be {_ARM_DOF + _HAND_DOF}-DoF "
                    f"(arm{_ARM_DOF}+hand{_HAND_DOF}), got {dim}"
                )
            arm = ctrl[:, :_ARM_DOF]
            hand = ctrl[:, _ARM_DOF:]
            return PolicyPrediction(
                arm_qpos=np.asarray(arm, dtype=np.float64),
                hand_qpos=np.asarray(hand, dtype=np.float64),
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
        return PolicyPrediction(
            arm_qpos=None,
            hand_qpos=np.asarray(hand, dtype=np.float64),
            ee_pos=np.asarray(ee_pos, dtype=np.float64),
            ee_rot6d=np.asarray(ee_rot6d, dtype=np.float64),
        )

    def predict(self, observation: ObservationBatch) -> PolicyPrediction:
        if self._agent is None:
            raise RuntimeError("DexManiPolicyRuntime.predict called before load()")
        obs_dict = self._encode(observation)
        result = self._agent.predict_action(
            obs_dict,
            denoise_timesteps=self._denoise_steps,
        )
        if not isinstance(result, Mapping) or "pred_action" not in result:
            raise ValueError(
                "DexMani agent must return a mapping containing 'pred_action'"
            )
        return self._decode(result["pred_action"])

    def close(self) -> None:
        agent = self._agent
        self._agent = None
        if agent is not None:
            close = getattr(agent, "close", None)
            if close is not None:
                close()


def _validate_policy_rot6d(rot6d: np.ndarray) -> None:
    """Reject policy rotations whose Gram-Schmidt projection is underdetermined."""
    values = np.asarray(rot6d, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != EE_ROT6D_DIM:
        raise ValueError(f"ee_rot6d must be [N, {EE_ROT6D_DIM}]")
    validate_rot6d_geometry(values, label="ee_rot6d")
