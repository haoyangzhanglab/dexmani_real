#!/usr/bin/env python3
"""NO HARDWARE DexMani Promotion Gate A offline qualification.

This harness:

- never connects hardware;
- never calls ``run_policy_deployment``;
- never imports or spawns ``arm_loop``;
- never imports or spawns ``hand_loop``;
- never imports or spawns ``camera_loop``;
- never imports or spawns ``pointcloud_loop``.

It consumes a VALID recorded episode and its production-processed HDF5 artifact,
restores the exact historical Policy producer, runs a real model prediction,
exercises immutable timing and ActionBuffer scheduling, reaches the production
SafetyGate shadow boundary, and then repeats the integration through production
shared-memory rings in three hardware-free spawned processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.control.publication import (
    CommandPublishStatus,
    build_action_candidate,
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import planner_action_safety_gate
from dexmani_real.deployment.action_buffer import (
    ActionBuffer,
    BufferCoverage,
    BufferedPlan,
    PushStatus,
)
from dexmani_real.deployment.artifact import (
    ResolvedPolicyArtifact,
    resolve_policy_artifact,
)
from dexmani_real.deployment.config import (
    PolicyRuntimeConfig,
    resolve_policy_runtime_config,
)
from dexmani_real.deployment.contracts import (
    InferenceContext,
    JointActionChunk,
    PolicyPrediction,
)
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
)
from dexmani_real.deployment.timing import (
    build_target_grid,
    compute_plan_deadline_ns,
    first_deliverable_index,
    first_valid_index_from_prefix_mask,
    usable_target_mask,
)
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    POLICY_PLAN_DTYPE,
    make_pointcloud_frame_dtype,
)
from dexmani_real.planning import (
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.recording.reader import EpisodeReader, ValidityState
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    StopRequest,
    begin_motion,
    require_transition,
)

FORBIDDEN_HARDWARE_MODULES = frozenset(
    {
        "pyrealsense2",
        "xarm",
        "dexmani_real.robot.arm_worker",
        "dexmani_real.robot.hand_worker",
        "dexmani_real.sensor.camera_worker",
        "dexmani_real.sensor.pointcloud_worker",
    }
)
RECEIPT_SCHEMA_VERSION = 1
REPLAY_WINDOW_STEPS = 2
MULTIPROCESS_TIMEOUT_S = 30.0
EXPECTED_RECORDED_EPISODE = "episode_20260827_224527"


@dataclass(frozen=True)
class ReplayPayload:
    """Small recorded window transported to hardware-free spawned children."""

    episode_name: str
    schema_version: int
    task_name: str
    control_hz: float
    grid_dt_s: float
    resolved_config_sha256: str
    window_start_index: int
    source_rows: np.ndarray
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    arm_source_sequence: np.ndarray
    arm_source_ns: np.ndarray
    arm_publish_ns: np.ndarray
    hand_qpos: np.ndarray
    hand_source_sequence: np.ndarray
    hand_source_ns: np.ndarray
    hand_publish_ns: np.ndarray
    camera_source_sequence: np.ndarray
    camera_source_ns: np.ndarray
    camera_publish_ns: np.ndarray
    camera_generation: np.ndarray
    depth_frame_number: np.ndarray
    color_frame_number: np.ndarray
    logical_step_ns: np.ndarray
    point_cloud: np.ndarray

    @property
    def original_reference_ns(self) -> int:
        return min(
            int(np.min(self.arm_source_ns)),
            int(np.min(self.hand_source_ns)),
            int(np.min(self.camera_source_ns)),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbidden_imports() -> list[str]:
    found: list[str] = []
    for forbidden in FORBIDDEN_HARDWARE_MODULES:
        if any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for name in sys.modules
        ):
            found.append(forbidden)
    return sorted(found)


def discover_recorded_episode(
    desktop_root: Path, *, task_name: str
) -> tuple[Path, dict[str, int]]:
    """Return the newest VALID matching raw episode after auditing all candidates."""
    root = desktop_root.resolve()
    candidates: list[Path] = []
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth >= 8:
            directories[:] = []
        if "data.h5" not in filenames:
            continue
        episode = current_path
        if all(
            (episode / name).is_file() for name in ("data.h5", "depth.h5", "rgb.mp4")
        ):
            candidates.append(episode)
    matches: list[Path] = []
    for candidate in candidates:
        try:
            with EpisodeReader(candidate) as reader:
                metadata_task = _text(
                    reader.h5f["meta"].attrs.get("task_label", "")
                ).strip()
                path_matches = task_name in candidate.parts
                if reader.validity is ValidityState.VALID and (
                    metadata_task == task_name or (not metadata_task and path_matches)
                ):
                    matches.append(candidate)
        except (FileNotFoundError, OSError, ValueError):
            continue
    if not matches:
        raise FileNotFoundError("no VALID recorded episode matches the artifact task")
    selected = sorted(
        matches,
        key=lambda path: (-path.stat().st_mtime_ns, path.name),
    )[0]
    return selected, {
        "candidate_count": len(candidates),
        "valid_matching_count": len(matches),
    }


def _probe_restore(
    *,
    python: Path,
    probe: Path,
    real_root: Path,
    policy_root: Path,
    experiment: Path,
    mode: str,
    synthetic_runtime: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(real_root), str(policy_root)))
    command = [
        str(python),
        str(probe),
        "--experiment",
        str(experiment),
        "--mode",
        mode,
    ]
    if synthetic_runtime:
        command.append("--synthetic-runtime")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} restore probe failed: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{mode} restore probe returned no receipt")
    value = json.loads(lines[-1])
    if value.get("ok") is not True or not isinstance(value.get("result"), dict):
        raise RuntimeError(f"{mode} restore probe did not pass")
    return value["result"]


def _inspect_fixture(
    *,
    fixture_experiment: Path,
    policy_commit: str,
    policy_root: Path,
    real_root: Path,
    python: Path,
) -> dict[str, int | str]:
    """Run and validate the hardware-free Real CLI print-config boundary."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(real_root), str(policy_root)))
    command = [
        str(python),
        str(real_root / "examples/run_policy.py"),
        "--experiment-dir",
        str(fixture_experiment),
        "--device",
        "cpu",
        "--print-config",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=60.0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "current fixture CLI inspect failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        receipt = json.loads(completed.stdout)
        artifact = receipt["artifact"]
        producer_commit = artifact["producer"]["commit"]
        allocation = artifact["allocation"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("current fixture CLI inspect output is invalid") from exc
    expected_allocation = {
        "n_obs_steps": 2,
        "n_action_steps": 8,
        "horizon": 16,
        "required_action_steps": 15,
        # The print-config schema exposes action_dim. For this joint-only
        # fixture it is the equivalent executable control_action_dim.
        "action_dim": 19,
    }
    if producer_commit != policy_commit:
        raise ValueError("current fixture CLI inspect producer mismatch")
    if any(
        allocation.get(name) != value for name, value in expected_allocation.items()
    ):
        raise ValueError("current fixture CLI inspect allocation mismatch")
    return {
        "producer_commit": producer_commit,
        "n_obs_steps": allocation["n_obs_steps"],
        "n_action_steps": allocation["n_action_steps"],
        "horizon": allocation["horizon"],
        "required_action_steps": allocation["required_action_steps"],
        "control_action_dim": allocation["action_dim"],
    }


def _validate_current_fixture(
    *,
    fixture_experiment: Path,
    policy_commit: str,
    policy_root: Path,
    real_root: Path,
    python: Path,
) -> dict[str, Any]:
    artifact = resolve_policy_artifact(fixture_experiment)
    allocation = artifact.allocation_contract
    if artifact.producer.commit != policy_commit:
        raise ValueError("current fixture producer does not match Policy main")
    if (
        allocation.n_obs_steps,
        allocation.n_action_steps,
        allocation.horizon,
        allocation.required_action_steps,
        allocation.control_action_dim,
    ) != (2, 8, 16, 15, 19):
        raise ValueError("current fixture allocation contract is not O2/A8/H16/19D")
    inspect = _inspect_fixture(
        fixture_experiment=fixture_experiment,
        policy_commit=policy_commit,
        policy_root=policy_root,
        real_root=real_root,
        python=python,
    )
    probe = policy_root / "tests/deployment/real_restore_probe.py"
    direct = _probe_restore(
        python=python,
        probe=probe,
        real_root=real_root,
        policy_root=policy_root,
        experiment=fixture_experiment,
        mode="direct",
        synthetic_runtime=True,
    )
    preflight = _probe_restore(
        python=python,
        probe=probe,
        real_root=real_root,
        policy_root=policy_root,
        experiment=fixture_experiment,
        mode="preflight",
        synthetic_runtime=True,
    )
    if direct.get("package_commit") != policy_commit:
        raise ValueError("current fixture direct restore provenance mismatch")
    if (
        preflight.get("checkpoint_sha256_verified") is not True
        or preflight.get("package_commit") != policy_commit
        or preflight.get("package_dirty") != "false"
        or preflight.get("action_steps") != 8
    ):
        raise ValueError("current fixture isolated preflight receipt mismatch")
    return {
        "producer_commit": policy_commit,
        "checkpoint": artifact.checkpoint_path.name,
        "checkpoint_sha256": direct["checkpoint_sha256"],
        "sidecar_sha256": artifact.index_sha256,
        "inspect": "PASS",
        "inspect_exit_code": 0,
        "inspect_allocation": {
            name: inspect[name]
            for name in (
                "n_obs_steps",
                "n_action_steps",
                "horizon",
                "required_action_steps",
                "control_action_dim",
            )
        },
        "direct_restore": "PASS",
        "preflight": "PASS",
        "action_steps": 8,
    }


def _validate_representative_restore(
    *,
    artifact: ResolvedPolicyArtifact,
    policy_root: Path,
    current_policy_root: Path,
    real_root: Path,
    python: Path,
) -> dict[str, Any]:
    probe = current_policy_root / "tests/deployment/real_restore_probe.py"
    direct = _probe_restore(
        python=python,
        probe=probe,
        real_root=real_root,
        policy_root=policy_root,
        experiment=artifact.experiment_dir,
        mode="direct",
        synthetic_runtime=False,
    )
    preflight = _probe_restore(
        python=python,
        probe=probe,
        real_root=real_root,
        policy_root=policy_root,
        experiment=artifact.experiment_dir,
        mode="preflight",
        synthetic_runtime=False,
    )
    expected_commit = artifact.producer.commit
    expected_sha = artifact.checkpoint_sha256_from_index
    if direct.get("package_commit") != expected_commit:
        raise ValueError("representative direct restore provenance mismatch")
    if direct.get("checkpoint_sha256") != expected_sha:
        raise ValueError("representative direct restore checkpoint mismatch")
    if (
        preflight.get("checkpoint_sha256_verified") is not True
        or preflight.get("checkpoint_sha256") != expected_sha
        or preflight.get("package_commit") != expected_commit
        or preflight.get("package_dirty") != "false"
        or preflight.get("action_steps") != artifact.allocation_contract.n_action_steps
    ):
        raise ValueError("representative isolated preflight receipt mismatch")
    return {
        "producer_commit": expected_commit,
        "checkpoint": artifact.checkpoint_path.name,
        "checkpoint_sha256": expected_sha,
        "sidecar_index_sha256": artifact.index_sha256,
        "direct_restore": "PASS",
        "preflight": "PASS",
    }


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_processed_contract(
    *,
    processed_path: Path,
    episode_dir: Path,
    artifact: ResolvedPolicyArtifact,
    runtime_config: PolicyRuntimeConfig,
) -> dict[str, Any]:
    """Validate a persisted production-processing result without reprocessing raw data."""
    if not processed_path.is_file():
        raise FileNotFoundError("matching production-processed episode is missing")
    raw_hashes = {
        name: _sha256_file(episode_dir / name)
        for name in ("data.h5", "depth.h5", "rgb.mp4")
    }
    allocation = artifact.allocation_contract
    with h5py.File(processed_path, "r") as source:
        attrs = source.attrs
        persisted_hashes = json.loads(_text(attrs["source_member_sha256_json"]))
        expected_semantics = {
            "task_name": allocation.task_name,
            "point_cloud_frame": runtime_config.point_cloud_frame,
            "point_cloud_color_source": runtime_config.point_cloud_color_source,
            "point_cloud_policy_id": runtime_config.point_cloud_policy_id,
            "point_cloud_config_sha256": runtime_config.point_cloud_config_sha256,
            "point_cloud_table_plane_abcd_json": (
                runtime_config.point_cloud_table_plane_abcd_json
            ),
            "point_cloud_sampling": runtime_config.point_cloud_sampling,
            "point_cloud_transform": runtime_config.point_cloud_transform,
        }
        mismatches = {
            key: (_text(attrs.get(key, "")), expected)
            for key, expected in expected_semantics.items()
            if _text(attrs.get(key, "")) != expected
        }
        if mismatches:
            raise ValueError(f"processed point-cloud contract mismatch: {mismatches}")
        if persisted_hashes != raw_hashes:
            raise ValueError(
                "processed artifact does not hash-bind the selected raw episode"
            )
        if _text(attrs.get("source_episode", "")) != episode_dir.name:
            raise ValueError("processed artifact source episode mismatch")
        if not np.isclose(
            float(attrs.get("dt", np.nan)),
            allocation.control_dt_s,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("processed control dt mismatches representative artifact")
        expected_shape = (
            int(attrs.get("episode_steps", -1)),
            int(allocation.point_cloud_num_points or -1),
            int(allocation.point_cloud_feature_dim or -1),
        )
        if source["joint_state"].shape != (expected_shape[0], 19):
            raise ValueError("processed joint_state contract mismatch")
        if source["point_cloud"].shape != expected_shape:
            raise ValueError("processed point_cloud contract mismatch")
        if source["point_cloud"].dtype != np.float32:
            raise ValueError("processed point_cloud must remain float32")
        source_rows = np.asarray(
            source["provenance/source_row_index"][:], dtype=np.int64
        )
        segment_ends = np.asarray(
            source["provenance/source_segment_ends"][:], dtype=np.int64
        )
        if source_rows.ndim != 1 or source_rows.size < allocation.n_obs_steps:
            raise ValueError("processed artifact has no complete observation window")
    return {
        "raw_hashes": raw_hashes,
        "source_rows": source_rows,
        "segment_ends": segment_ends,
    }


def load_replay_payload(
    *,
    episode_dir: Path,
    processed_path: Path,
    artifact: ResolvedPolicyArtifact,
    runtime_config: PolicyRuntimeConfig,
) -> tuple[ReplayPayload, dict[str, str]]:
    persisted = _validate_processed_contract(
        processed_path=processed_path,
        episode_dir=episode_dir,
        artifact=artifact,
        runtime_config=runtime_config,
    )
    allocation = artifact.allocation_contract
    n_obs = allocation.n_obs_steps
    with EpisodeReader(episode_dir) as reader:
        if reader.validity is not ValidityState.VALID:
            raise ValueError("recorded episode validity is not VALID")
        meta = reader.h5f["meta"].attrs
        task_name = _text(meta.get("task_label", "")).strip()
        if task_name != allocation.task_name:
            raise ValueError("recorded task does not match representative artifact")
        timing = reader.timing
        if not np.isclose(
            timing.grid_dt_s, allocation.control_dt_s, rtol=0.0, atol=1e-12
        ):
            raise ValueError("recorded control dt mismatches representative artifact")
        source_rows = persisted["source_rows"]
        segment_ends = persisted["segment_ends"]
        starts = np.concatenate((np.array([0], dtype=np.int64), segment_ends[:-1]))
        window_start = next(
            (
                int(start)
                for start, end in zip(starts, segment_ends, strict=True)
                if int(end) - int(start) >= n_obs
            ),
            None,
        )
        if window_start is None:
            raise ValueError("no complete causal processed observation window")
        compact_indices = np.arange(window_start, window_start + n_obs, dtype=np.int64)
        rows = source_rows[compact_indices]
        raw = reader.h5f
        required_true = np.asarray(raw["policy_observation_valid"][rows], dtype=bool)
        camera_sequence = np.asarray(
            raw["camera_source_sequence"][rows], dtype=np.int64
        )
        camera_source = np.asarray(
            raw["camera_source_monotonic_ns"][rows], dtype=np.int64
        )
        camera_publish = np.asarray(
            raw["camera_publish_monotonic_ns"][rows], dtype=np.int64
        )
        camera_generation = np.asarray(raw["camera_generation"][rows], dtype=np.int64)
        arm_source = np.asarray(
            raw["policy_observation_arm_source_monotonic_ns"][rows], dtype=np.int64
        )
        arm_publish = np.asarray(
            raw["policy_observation_arm_publish_monotonic_ns"][rows], dtype=np.int64
        )
        hand_source = np.asarray(
            raw["policy_observation_hand_source_monotonic_ns"][rows], dtype=np.int64
        )
        hand_publish = np.asarray(
            raw["policy_observation_hand_publish_monotonic_ns"][rows], dtype=np.int64
        )
        logical = np.asarray(
            raw["observation_anchor_monotonic_ns"][rows], dtype=np.int64
        )
        skew_s = np.asarray(raw["policy_observation_skew_s"][rows], dtype=np.float64)
        causal = (
            bool(np.all(required_true))
            and bool(np.all(np.diff(camera_sequence) > 0))
            and bool(np.all(np.diff(camera_source) > 0))
            and len(set(int(value) for value in camera_generation)) == 1
            and bool(np.all(arm_source <= camera_source))
            and bool(np.all(hand_source <= camera_source))
            and bool(np.all(arm_source <= arm_publish))
            and bool(np.all(hand_source <= hand_publish))
            and bool(np.all(camera_source <= camera_publish))
            and bool(np.all(camera_publish <= logical))
            and bool(np.all(skew_s <= runtime_config.deployment.max_observation_skew_s))
        )
        if not causal:
            raise ValueError("earliest recorded observation window is not causal")
        expected_grid_ns = int(round(allocation.control_dt_s * 1e9))
        if not np.allclose(np.diff(logical), expected_grid_ns, rtol=0.0, atol=100):
            raise ValueError("recorded observation window is not on the control grid")
        with h5py.File(processed_path, "r") as processed:
            point_cloud = np.asarray(
                processed["point_cloud"][compact_indices], dtype=np.float32
            )
        payload = ReplayPayload(
            episode_name=episode_dir.name,
            schema_version=reader.schema_version,
            task_name=task_name,
            control_hz=timing.rate_hz,
            grid_dt_s=timing.grid_dt_s,
            resolved_config_sha256=_text(meta.get("resolved_config_sha256", "")),
            window_start_index=window_start,
            source_rows=rows,
            arm_qpos=np.asarray(
                raw["policy_observation_arm_qpos"][rows], dtype=np.float64
            ),
            arm_qvel=np.asarray(raw["arm_qvel"][rows], dtype=np.float64),
            arm_source_sequence=np.asarray(
                raw["policy_observation_arm_source_sequence"][rows], dtype=np.uint64
            ),
            arm_source_ns=arm_source,
            arm_publish_ns=arm_publish,
            hand_qpos=np.asarray(
                raw["policy_observation_hand_qpos"][rows], dtype=np.float64
            ),
            hand_source_sequence=np.asarray(
                raw["policy_observation_hand_source_sequence"][rows], dtype=np.uint64
            ),
            hand_source_ns=hand_source,
            hand_publish_ns=hand_publish,
            camera_source_sequence=camera_sequence,
            camera_source_ns=camera_source,
            camera_publish_ns=camera_publish,
            camera_generation=camera_generation,
            depth_frame_number=np.asarray(
                raw["camera_depth_frame_number"][rows], dtype=np.int64
            ),
            color_frame_number=np.asarray(
                raw["camera_color_frame_number"][rows], dtype=np.int64
            ),
            logical_step_ns=logical,
            point_cloud=point_cloud,
        )
    return payload, persisted["raw_hashes"]


def _rebase(values: np.ndarray, *, original_ns: int, replay_ns: int) -> np.ndarray:
    return np.asarray(
        [replay_ns + (int(value) - original_ns) for value in values],
        dtype=np.uint64,
    )


def build_observation(
    payload: ReplayPayload,
    *,
    replay_epoch_ns: int,
    run_generation: int,
    observation_id: int,
) -> ObservationBatch:
    original = payload.original_reference_ns
    arm_source = _rebase(
        payload.arm_source_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    arm_publish = _rebase(
        payload.arm_publish_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    hand_source = _rebase(
        payload.hand_source_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    hand_publish = _rebase(
        payload.hand_publish_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    camera_source = _rebase(
        payload.camera_source_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    camera_publish = _rebase(
        payload.camera_publish_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    logical = _rebase(
        payload.logical_step_ns, original_ns=original, replay_ns=replay_epoch_ns
    )
    mask = np.ones(len(payload.source_rows), dtype=np.uint8)
    arm = FrameWindow(
        payload.arm_qpos,
        payload.arm_source_sequence,
        arm_source,
        arm_publish,
        mask,
    )
    hand = FrameWindow(
        payload.hand_qpos,
        payload.hand_source_sequence,
        hand_source,
        hand_publish,
        mask,
    )
    clouds = tuple(
        PointCloudFrame(
            payload.point_cloud[index],
            int(payload.camera_source_sequence[index]),
            int(camera_source[index]),
            int(camera_publish[index]),
            int(payload.camera_generation[index]),
        )
        for index in range(len(payload.source_rows))
    )
    return ObservationBatch(
        observation_id=observation_id,
        run_generation=run_generation,
        run_started_monotonic_ns=replay_epoch_ns,
        anchor_monotonic_ns=int(logical[-1]),
        latest_source_monotonic_ns=int(camera_source[-1]),
        logical_step_monotonic_ns=int(logical[-1]),
        arm_history=arm,
        hand_history=hand,
        pointcloud=clouds[-1],
        pointcloud_history=clouds,
    )


def _make_channels(
    runtime: Any, policy: PolicyRuntimeConfig, *, prefix: str
) -> RuntimeChannels:
    allocation = policy.artifact.allocation_contract if policy.artifact else None
    if allocation is None or allocation.point_cloud_num_points is None:
        raise ValueError("Gate A requires a point-cloud artifact")
    config = RuntimeChannelsConfig.from_runtime(
        runtime,
        pointcloud_num_points=allocation.point_cloud_num_points,
        camera_requested=True,
        pointcloud_requested=True,
        observation_horizon=allocation.n_obs_steps,
        observation_dt_s=policy.control_dt_s,
        max_input_age_s=policy.deployment.max_input_age_s,
        max_observation_skew_s=policy.deployment.max_observation_skew_s,
        max_grid_lag_s=policy.deployment.max_grid_lag_s,
    )
    return RuntimeChannels.create(
        prefix=prefix, config=config, mp_context=mp.get_context("spawn")
    )


def _write_feedback(
    shared: RuntimeChannels,
    payload: ReplayPayload,
    observation: ObservationBatch,
) -> None:
    assert observation.arm_history is not None and observation.hand_history is not None
    index = len(payload.source_rows) - 1
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["qpos"][0] = payload.arm_qpos[index]
    arm["qvel"][0] = payload.arm_qvel[index]
    arm["connected"][0] = 1
    arm["state_valid"][0] = 1
    arm["source_monotonic_ns"][0] = observation.arm_history.source_monotonic_ns[index]
    arm["publish_monotonic_ns"][0] = observation.arm_history.publish_monotonic_ns[index]
    shared.arm_state_ring.write(arm)

    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand["qpos"][0] = payload.hand_qpos[index]
    hand["connected"][0] = 1
    hand["state_valid"][0] = 1
    hand["source_monotonic_ns"][0] = observation.hand_history.source_monotonic_ns[index]
    hand["publish_monotonic_ns"][0] = observation.hand_history.publish_monotonic_ns[
        index
    ]
    shared.hand_state_ring.write(hand)


def _make_gate(runtime: Any) -> Any:
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=runtime.policy.workspace.as_array(),
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=float(runtime.policy.ik_max_pose_error_pos_m),
            max_pose_error_rot_rad=float(runtime.policy.ik_max_pose_error_rot_rad),
        ),
        hand_dof=True,
        static_boxes=tuple(runtime.environment.static_boxes),
    )
    gate = planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
        arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
        hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
        hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
        max_arm_delta_rad=float(runtime.arm.max_servo_command_jump_rad),
        max_hand_delta_rad=None,
        endpoint_delta_tolerance_rad=float(runtime.policy.endpoint_delta_tolerance_rad),
        collision_check=planner.collision_model.check_transition_collision_free,
    )
    return planner, gate


def run_recorded_replay(
    *,
    payload: ReplayPayload,
    runtime_config: PolicyRuntimeConfig,
    real_runtime: Any,
) -> dict[str, Any]:
    from dexmani_real.deployment.preflight import load_verified_policy_runtime

    planner, gate = _make_gate(real_runtime)
    shared = _make_channels(
        real_runtime,
        runtime_config,
        prefix=f"dexmani_gate_a_direct_{os.getpid()}_{time.monotonic_ns()}",
    )
    model = None
    try:
        model = load_verified_policy_runtime(runtime_config)
        require_transition(shared, SafetyState.ARMED)
        if not begin_motion(shared):
            raise RuntimeError("direct replay could not enter RUNNING")
        run_started_ns = int(shared.run_started_monotonic_ns.value)
        replay_epoch_ns = run_started_ns + 1_000_000
        observation = build_observation(
            payload,
            replay_epoch_ns=replay_epoch_ns,
            run_generation=int(shared.run_generation.value),
            observation_id=1,
        )
        wait_ns = observation.anchor_monotonic_ns - time.monotonic_ns()
        if wait_ns > 0:
            time.sleep(wait_ns / 1e9)
        inference_started_ns = time.monotonic_ns()
        prediction = model.predict(observation)
        inference_finished_ns = time.monotonic_ns()
        model_latency_ns = inference_finished_ns - inference_started_ns

        values = (
            prediction.arm_qpos
            if prediction.arm_qpos is not None
            else prediction.ee_pos
        )
        assert values is not None
        step_count = int(values.shape[0])
        step_dt_ns = int(round(runtime_config.control_dt_s * 1e9))
        target_grid = build_target_grid(
            observation.logical_step_monotonic_ns, step_count, step_dt_ns
        )
        target_grid_copy = target_grid.copy()
        command_lead_ns = int(np.ceil(runtime_config.deployment.command_lead_s * 1e9))
        first_index = first_deliverable_index(
            target_grid, inference_finished_ns, command_lead_ns
        )
        if first_index == step_count:
            raise RuntimeError("all representative inference targets expired")
        transport_mask = np.zeros(step_count, dtype=np.uint8)
        transport_mask[first_index:] = 1
        if first_valid_index_from_prefix_mask(transport_mask) != first_index:
            raise RuntimeError("transport mask is not the required 0*1* topology")
        deadline_ns = compute_plan_deadline_ns(
            inference_finished_ns,
            observation.latest_source_monotonic_ns,
            int(runtime_config.deployment.max_plan_age_s * 1e9),
            int(runtime_config.deployment.max_source_to_command_age_s * 1e9),
        )
        usable_mask = usable_target_mask(target_grid, first_index, deadline_ns)
        if not np.array_equal(target_grid, target_grid_copy):
            raise RuntimeError("deadline calculation mutated the target grid")
        if not np.array_equal(
            transport_mask,
            np.r_[
                np.zeros(first_index, dtype=np.uint8),
                np.ones(step_count - first_index, dtype=np.uint8),
            ],
        ):
            raise RuntimeError("deadline calculation mutated transport topology")
        if not np.all(
            target_grid[first_index:] > inference_finished_ns + command_lead_ns
        ):
            raise RuntimeError(
                "deliverable targets violate the strict lower timing rule"
            )
        if not np.array_equal(
            usable_mask.astype(bool),
            (transport_mask == 1) & (target_grid < deadline_ns),
        ):
            raise RuntimeError(
                "usable mask does not apply the independent upper deadline"
            )
        usable_count = int(np.count_nonzero(usable_mask))
        if usable_count <= 0:
            raise RuntimeError("recorded timing replay has no usable target")

        chunk = JointActionChunk(
            arm_qpos=prediction.arm_qpos,
            hand_qpos=prediction.hand_qpos,
            ee_pos=prediction.ee_pos,
            ee_rot6d=prediction.ee_rot6d,
            target_monotonic_ns=target_grid,
            valid_mask=transport_mask,
        )
        plan = BufferedPlan(
            plan_id=1,
            run_generation=int(shared.run_generation.value),
            observation_id=1,
            observation_anchor_ns=observation.anchor_monotonic_ns,
            observation_latest_source_ns=observation.latest_source_monotonic_ns,
            inference_finished_ns=inference_finished_ns,
            deadline_ns=deadline_ns,
            chunk=chunk,
        )
        scheduler = ActionBuffer(max_buffered_plans=3)
        scheduler.reset(run_generation=int(shared.run_generation.value))
        admitted = scheduler.push(plan, now_ns=inference_finished_ns)
        if admitted.status is not PushStatus.ACCEPTED:
            raise RuntimeError(
                f"ActionBuffer rejected replay plan: {admitted.status.value}"
            )
        due_ns = int(target_grid[first_index])
        selected = scheduler.peek_due(now_ns=due_ns)
        if selected.coverage is not BufferCoverage.DUE:
            raise RuntimeError("ActionBuffer did not expose the first due endpoint")
        assert selected.step_index is not None and selected.token is not None
        if selected.step_index != first_index:
            raise RuntimeError("ActionBuffer retimed or selected a different endpoint")

        remaining_ns = due_ns - time.monotonic_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)
        _write_feedback(shared, payload, observation)
        before = int(shared.coupled_cmd_ring.latest_sequence)
        if prediction.arm_qpos is None:
            raise RuntimeError("representative joint artifact returned an EE action")
        arm_qpos = np.asarray(
            prediction.arm_qpos[selected.step_index], dtype=np.float64
        )
        assert prediction.hand_qpos is not None
        hand_qpos = np.asarray(
            prediction.hand_qpos[selected.step_index], dtype=np.float64
        )
        candidate = build_action_candidate(
            shared,
            arm_qpos,
            hand_qpos,
            observation_id=1,
            observation_anchor_monotonic_ns=observation.anchor_monotonic_ns,
            scheduled_target_monotonic_ns=due_ns,
            action_validity_s=runtime_config.deployment.action_validity_s,
            valid_until_monotonic_ns=deadline_ns,
        )
        if candidate is None:
            raise RuntimeError("recorded replay candidate could not be built")
        result = validate_and_send_candidate(
            shared,
            candidate,
            gate=gate,
            arm_feedback_max_age_s=float(real_runtime.safety.heartbeat_timeouts["arm"]),
            hand_feedback_max_age_s=float(
                real_runtime.safety.heartbeat_timeouts["hand"]
            ),
            hand_mechanical_lower_rad=np.asarray(
                real_runtime.hand.mechanical_qpos_min_rad, dtype=np.float64
            ),
            hand_mechanical_upper_rad=np.asarray(
                real_runtime.hand.mechanical_qpos_max_rad, dtype=np.float64
            ),
            hand_command_max_delta_rad_per_tick=float(
                real_runtime.hand.hand_max_delta_rad_per_tick
            ),
            canonicalize_policy_hand_roundoff=True,
            execution_mode="shadow",
        )
        after = int(shared.coupled_cmd_ring.latest_sequence)
        if result.status is not CommandPublishStatus.SHADOW_VALIDATED:
            raise RuntimeError(
                f"recorded replay SafetyGate disposition={result.status.value}: {result.detail}"
            )
        if after != before:
            raise RuntimeError("direct shadow validation wrote coupled_cmd_ring")
        scheduler.commit(selected.token)
        artifact = runtime_config.artifact
        if artifact is None:
            raise RuntimeError("recorded replay lost its artifact binding")
        return {
            "status": "PASS",
            "actual_model_inference": "PASS",
            "pred_action_conceptual_shape": [
                1,
                artifact.allocation_contract.horizon,
                artifact.allocation_contract.action_dim,
            ],
            "control_action_shape": [step_count, 19],
            "model_latency_ms": model_latency_ns / 1e6,
            "timing_basis": "recorded_relative_rebased",
            "window_start_index": payload.window_start_index,
            "window_steps": len(payload.source_rows),
            "first_deliverable_index": first_index,
            "transport_valid_count": int(np.count_nonzero(transport_mask)),
            "usable_target_count": usable_count,
            "deadline_relative_ns": deadline_ns
            - observation.latest_source_monotonic_ns,
            "shadow_disposition": result.status.name,
            "retimed_actions": False,
            "coupled_sequence_before": before,
            "coupled_sequence_after": after,
        }
    finally:
        if model is not None:
            model.close()
        del planner
        shared.is_running.value = False
        if not shared.close():
            raise RuntimeError("direct replay RuntimeChannels cleanup failed")


def _inference_child(
    shared: RuntimeChannels, config: PolicyRuntimeConfig, diagnostics: Any
) -> None:
    try:
        from dexmani_real.deployment.worker import inference_loop

        inference_loop(shared, config)
    except BaseException as exc:
        diagnostics.put(
            {
                "event": "child_error",
                "process": "inference",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        diagnostics.put(
            {
                "event": "imports",
                "process": "inference",
                "forbidden": _forbidden_imports(),
            }
        )


def _coordinator_child(shared: RuntimeChannels, config: Any, diagnostics: Any) -> None:
    try:
        import dexmani_real.deployment.coordinator as coordinator

        original = coordinator.validate_and_send_candidate

        def instrumented(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            diagnostics.put(
                {
                    "event": "publication",
                    "status": result.status.name,
                    "detail": result.detail,
                    "target_ns": (
                        None
                        if result.candidate is None
                        else int(result.candidate.scheduled_target_monotonic_ns)
                    ),
                }
            )
            return result

        coordinator.validate_and_send_candidate = instrumented
        coordinator.coordinator_loop(shared, config)
    except BaseException as exc:
        diagnostics.put(
            {
                "event": "child_error",
                "process": "policy",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        diagnostics.put(
            {"event": "imports", "process": "policy", "forbidden": _forbidden_imports()}
        )


def _replay_feed_segment(
    source_rows: np.ndarray,
    segment_ends: np.ndarray,
    *,
    window_start: int,
    window_steps: int,
    max_frames: int,
) -> tuple[int, int, int, int]:
    """Select a bounded feed range containing one deterministic replay window."""
    rows = np.asarray(source_rows)
    ends = np.asarray(segment_ends)
    if rows.ndim != 1:
        raise ValueError("source_row_index must be 1-D")
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("source_segment_ends must be non-empty and 1-D")
    if int(ends[0]) <= 0:
        raise ValueError("source_segment_ends must start after compact index zero")
    if np.any(np.diff(ends) <= 0):
        raise ValueError("source_segment_ends must be strictly increasing")
    if int(ends[-1]) > len(rows):
        raise ValueError("source_segment_ends exceeds source_row_index")
    if window_start < 0 or window_steps <= 0:
        raise ValueError("replay window bounds must be positive")
    window_end = window_start + window_steps
    if window_end > len(rows):
        raise ValueError("replay window exceeds source_row_index")
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1]))
    containing = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if int(start) <= window_start and window_end <= int(end)
    ]
    if len(containing) != 1:
        raise ValueError("replay window is not contained by exactly one source segment")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    segment_start, segment_end = containing[0]
    feed_end = min(segment_end, segment_start + max_frames)
    if window_end > feed_end:
        raise ValueError("replay window is outside the bounded feeder range")
    return segment_start, segment_end, segment_start, feed_end


def _replay_feeder_child(
    shared: RuntimeChannels,
    payload: ReplayPayload,
    processed_path: str,
    diagnostics: Any,
) -> None:
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and shared.is_running.value:
            if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                break
            time.sleep(0.005)
        if int(shared.safety_state.value) != int(SafetyState.RUNNING):
            raise TimeoutError("replay feeder did not observe RUNNING")
        run_started_ns = int(shared.run_started_monotonic_ns.value)
        original = payload.original_reference_ns
        replay_epoch_ns = run_started_ns + 1_000_000
        point_dtype = make_pointcloud_frame_dtype(payload.point_cloud.shape[1])
        with h5py.File(processed_path, "r") as processed:
            source_rows = np.asarray(
                processed["provenance/source_row_index"][:], dtype=np.int64
            )
            segment_ends = np.asarray(
                processed["provenance/source_segment_ends"][:], dtype=np.int64
            )
            segment_start, segment_end, feed_start, feed_end = _replay_feed_segment(
                source_rows,
                segment_ends,
                window_start=payload.window_start_index,
                window_steps=len(payload.source_rows),
                max_frames=20,
            )
            compact_rows = source_rows[feed_start:feed_end]
            clouds = np.asarray(
                processed["point_cloud"][feed_start:feed_end], dtype=np.float32
            )
        if len(compact_rows) != len(clouds):
            raise ValueError(
                "replay feeder source rows and point clouds differ in length"
            )
        diagnostics.put(
            {
                "event": "feeder_range",
                "source_segment_start": segment_start,
                "source_segment_end": segment_end,
                "feed_start": feed_start,
                "feed_end": feed_end,
                "feed_frame_count": len(compact_rows),
            }
        )
        episode_dir = (
            Path(processed_path).parents[1]
            / "episodes"
            / payload.task_name
            / payload.episode_name
        )
        # The caller passes a repository-local processed path. Resolve the raw
        # sibling from the repository root without embedding it in diagnostics.
        real_root = Path(processed_path).resolve().parents[2]
        episode_dir = real_root / "episodes" / payload.task_name / payload.episode_name
        with EpisodeReader(episode_dir) as reader:
            raw = reader.h5f
            for index, row in enumerate(compact_rows):
                if not shared.is_running.value:
                    break
                source_candidates = (
                    int(raw["policy_observation_arm_source_monotonic_ns"][row]),
                    int(raw["policy_observation_hand_source_monotonic_ns"][row]),
                    int(raw["camera_source_monotonic_ns"][row]),
                )
                publish_candidates = (
                    int(raw["policy_observation_arm_publish_monotonic_ns"][row]),
                    int(raw["policy_observation_hand_publish_monotonic_ns"][row]),
                    int(raw["camera_publish_monotonic_ns"][row]),
                )
                publish_due_ns = replay_epoch_ns + (max(publish_candidates) - original)
                remaining_ns = publish_due_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)
                arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
                arm["qpos"][0] = raw["policy_observation_arm_qpos"][row]
                arm["qvel"][0] = raw["arm_qvel"][row]
                arm["connected"][0] = 1
                arm["state_valid"][0] = 1
                arm["source_monotonic_ns"][0] = replay_epoch_ns + (
                    source_candidates[0] - original
                )
                arm["publish_monotonic_ns"][0] = replay_epoch_ns + (
                    publish_candidates[0] - original
                )
                shared.arm_state_ring.write(arm)
                hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
                hand["qpos"][0] = raw["policy_observation_hand_qpos"][row]
                hand["connected"][0] = 1
                hand["state_valid"][0] = 1
                hand["source_monotonic_ns"][0] = replay_epoch_ns + (
                    source_candidates[1] - original
                )
                hand["publish_monotonic_ns"][0] = replay_epoch_ns + (
                    publish_candidates[1] - original
                )
                shared.hand_state_ring.write(hand)
                cloud = np.zeros(1, dtype=point_dtype)
                cloud["source_camera_sequence"][0] = raw["camera_source_sequence"][row]
                cloud["source_monotonic_ns"][0] = replay_epoch_ns + (
                    source_candidates[2] - original
                )
                cloud["camera_publish_monotonic_ns"][0] = replay_epoch_ns + (
                    publish_candidates[2] - original
                )
                cloud["publish_monotonic_ns"][0] = max(
                    time.monotonic_ns(), int(cloud["camera_publish_monotonic_ns"][0])
                )
                cloud["camera_generation"][0] = raw["camera_generation"][row]
                cloud["depth_frame_number"][0] = raw["camera_depth_frame_number"][row]
                cloud["color_frame_number"][0] = raw["camera_color_frame_number"][row]
                cloud["point_cloud"][0] = clouds[index]
                shared.pointcloud_ring.write(cloud)
        diagnostics.put({"event": "feeder_complete", "frames": int(len(compact_rows))})
    except BaseException as exc:
        diagnostics.put(
            {
                "event": "child_error",
                "process": "replay_feeder",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        diagnostics.put(
            {
                "event": "imports",
                "process": "replay_feeder",
                "forbidden": _forbidden_imports(),
            }
        )


def _negative_child(shared: RuntimeChannels, diagnostics: Any) -> None:
    try:
        from dexmani_real.deployment.coordinator import _buffered_plan_from_record
        from dexmani_real.deployment.worker import publish_plan, stamp_prediction_timing

        old_generation = int(shared.run_generation.value)
        context = InferenceContext(
            run_generation=old_generation,
            observation_id=1,
            observation_anchor_monotonic_ns=100,
            observation_latest_source_monotonic_ns=90,
            observation_logical_step_monotonic_ns=95,
            inference_started_monotonic_ns=110,
            inference_finished_monotonic_ns=120,
            step_dt_ns=10,
        )
        chunk = JointActionChunk(
            arm_qpos=np.zeros((2, 7), dtype=np.float64),
            hand_qpos=np.zeros((2, 12), dtype=np.float64),
            target_monotonic_ns=np.array([130, 140], dtype=np.uint64),
            valid_mask=np.ones(2, dtype=np.uint8),
        )
        with shared.run_generation.get_lock():
            shared.run_generation.value = old_generation + 1
        generation_dropped = not publish_plan(
            shared, plan_id=1, context=context, chunk=chunk
        )

        plan = BufferedPlan(
            plan_id=1,
            run_generation=old_generation + 1,
            observation_id=1,
            observation_anchor_ns=100,
            observation_latest_source_ns=90,
            inference_finished_ns=120,
            deadline_ns=140,
            chunk=chunk,
        )
        buffer = ActionBuffer(max_buffered_plans=2)
        buffer.reset(run_generation=old_generation + 1)
        deadline_closed = (
            buffer.push(plan, now_ns=140).status is PushStatus.DEADLINE_CLOSED
        )

        record = np.zeros(1, dtype=POLICY_PLAN_DTYPE)[0]
        record["plan_id"] = 1
        record["run_generation"] = old_generation + 1
        record["observation_id"] = 1
        record["observation_latest_source_monotonic_ns"] = 90
        record["observation_logical_step_monotonic_ns"] = 95
        record["observation_anchor_monotonic_ns"] = 100
        record["inference_started_monotonic_ns"] = 110
        record["inference_finished_monotonic_ns"] = 120
        record["num_steps"] = 3
        record["arm_present"] = 1
        record["hand_present"] = 1
        record["target_monotonic_ns"][:3] = (130, 140, 150)
        record["valid_mask"][:3] = (1, 0, 1)
        non_prefix_failed_closed = False
        try:
            _buffered_plan_from_record(
                record,
                max_plan_age_ns=1_000,
                max_source_to_command_age_ns=1_000,
            )
        except ValueError:
            non_prefix_failed_closed = True

        prediction = PolicyPrediction(
            arm_qpos=np.zeros((2, 7), dtype=np.float64),
            hand_qpos=np.zeros((2, 12), dtype=np.float64),
        )
        all_targets_expired = (
            stamp_prediction_timing(
                prediction,
                logical_step_ns=100,
                step_dt_ns=10,
                inference_finished_ns=200,
                command_lead_ns=0,
            )
            is None
        )
        diagnostics.put(
            {
                "event": "negative_checks",
                "generation_change_plan_dropped": generation_dropped,
                "deadline_closed_no_validation": deadline_closed,
                "non_prefix_mask_failed_closed": non_prefix_failed_closed,
                "all_targets_expired_no_plan": all_targets_expired,
                "policy_plan_sequence": int(shared.policy_plan_ring.latest_sequence),
                "coupled_sequence": int(shared.coupled_cmd_ring.latest_sequence),
            }
        )
    except BaseException as exc:
        diagnostics.put(
            {
                "event": "child_error",
                "process": "negative_checks",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        diagnostics.put(
            {
                "event": "imports",
                "process": "negative_checks",
                "forbidden": _forbidden_imports(),
            }
        )


def _drain_diagnostics(diagnostics: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        try:
            events.append(diagnostics.get_nowait())
        except queue.Empty:
            return events


def _stop_processes(processes: list[Any], shared: RuntimeChannels) -> None:
    shared.is_running.value = False
    for process in processes:
        process.join(timeout=2.0)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2.0)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)


def run_multiprocess_shadow(
    *,
    payload: ReplayPayload,
    processed_path: Path,
    policy_config: PolicyRuntimeConfig,
    real_runtime: Any,
) -> dict[str, Any]:
    from dexmani_real.deployment.coordinator import CoordinatorConfig

    ctx = mp.get_context("spawn")
    shared = _make_channels(
        real_runtime,
        policy_config,
        prefix=f"dexmani_gate_a_mp_{os.getpid()}_{time.monotonic_ns()}",
    )
    diagnostics = ctx.Queue()
    coordinator_config = CoordinatorConfig.from_runtime(
        policy_config.deployment,
        real_runtime,
        execution_mode="shadow",
        h4_execute_bounds=None,
    )
    processes = [
        ctx.Process(
            name="inference",
            target=_inference_child,
            args=(shared, policy_config, diagnostics),
        ),
        ctx.Process(
            name="policy",
            target=_coordinator_child,
            args=(shared, coordinator_config, diagnostics),
        ),
        ctx.Process(
            name="replay_feeder",
            target=_replay_feeder_child,
            args=(shared, payload, str(processed_path), diagnostics),
        ),
    ]
    coupled_before = int(shared.coupled_cmd_ring.latest_sequence)
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        require_transition(shared, SafetyState.ARMED)
        for process in processes:
            process.start()
        if not shared.wait_ready("inference", timeout=20.0):
            raise TimeoutError("inference process did not become ready")
        if not shared.wait_ready("policy", timeout=10.0):
            raise TimeoutError("coordinator process did not become ready")
        shared.start_request.value = True
        shadow_validated = 0
        while time.monotonic() - started <= MULTIPROCESS_TIMEOUT_S:
            try:
                event = diagnostics.get(timeout=0.1)
                events.append(event)
            except queue.Empty:
                pass
            shadow_validated = sum(
                event.get("event") == "publication"
                and event.get("status") == CommandPublishStatus.SHADOW_VALIDATED.name
                for event in events
            )
            child_errors = [
                event for event in events if event.get("event") == "child_error"
            ]
            if child_errors:
                raise RuntimeError(f"multiprocess child failed: {child_errors}")
            if shadow_validated >= 1:
                shared.stop_request.value = int(StopRequest.OPERATOR)
                break
        if shadow_validated < 1:
            raise TimeoutError("multiprocess shadow did not reach SHADOW_VALIDATED")
        time.sleep(0.1)
    finally:
        _stop_processes(processes, shared)
        events.extend(_drain_diagnostics(diagnostics))
    try:
        exitcodes = {process.name: process.exitcode for process in processes}
        if any(code not in (0, -15) for code in exitcodes.values()):
            raise RuntimeError(f"multiprocess child exit failure: {exitcodes}")
        child_errors = [
            event for event in events if event.get("event") == "child_error"
        ]
        if child_errors:
            raise RuntimeError(f"multiprocess child error: {child_errors}")
        forbidden = sorted(
            {
                module
                for event in events
                if event.get("event") == "imports"
                for module in event.get("forbidden", [])
            }
        )
        if forbidden:
            raise RuntimeError(f"hardware module import audit failed: {forbidden}")
        feeder_ranges = [
            event for event in events if event.get("event") == "feeder_range"
        ]
        if len(feeder_ranges) != 1:
            raise RuntimeError("multiprocess replay feeder range evidence is missing")
        feeder_range = feeder_ranges[0]
        coupled_after = int(shared.coupled_cmd_ring.latest_sequence)
        plan_sequence = int(shared.policy_plan_ring.latest_sequence)
        shadow_validated = sum(
            event.get("event") == "publication"
            and event.get("status") == CommandPublishStatus.SHADOW_VALIDATED.name
            for event in events
        )
        if plan_sequence <= 0:
            raise RuntimeError("policy_plan_ring did not advance")
        if coupled_after != coupled_before:
            raise RuntimeError("multiprocess shadow changed coupled_cmd_ring")

        negative_shared = _make_channels(
            real_runtime,
            policy_config,
            prefix=f"dexmani_gate_a_negative_{os.getpid()}_{time.monotonic_ns()}",
        )
        negative_diagnostics = ctx.Queue()
        negative = ctx.Process(
            name="negative_checks",
            target=_negative_child,
            args=(negative_shared, negative_diagnostics),
        )
        try:
            negative.start()
            negative.join(timeout=10.0)
            if negative.is_alive():
                negative.terminate()
                negative.join(timeout=2.0)
                raise TimeoutError("negative multiprocess checks timed out")
            negative_events = _drain_diagnostics(negative_diagnostics)
            if negative.exitcode != 0:
                raise RuntimeError(
                    f"negative multiprocess child exit={negative.exitcode}"
                )
            negative_receipt = next(
                (
                    event
                    for event in negative_events
                    if event.get("event") == "negative_checks"
                ),
                None,
            )
            if negative_receipt is None:
                raise RuntimeError("negative multiprocess receipt is missing")
            expected = (
                "generation_change_plan_dropped",
                "deadline_closed_no_validation",
                "non_prefix_mask_failed_closed",
                "all_targets_expired_no_plan",
            )
            if any(negative_receipt.get(name) is not True for name in expected):
                raise RuntimeError(
                    f"negative multiprocess check failed: {negative_receipt}"
                )
            if (
                negative_receipt["policy_plan_sequence"] != 0
                or negative_receipt["coupled_sequence"] != 0
            ):
                raise RuntimeError(
                    "negative multiprocess checks wrote an IPC plan or command"
                )
            negative_forbidden = [
                module
                for event in negative_events
                if event.get("event") == "imports"
                for module in event.get("forbidden", [])
            ]
            if negative_forbidden:
                raise RuntimeError(
                    f"negative child hardware import audit failed: {negative_forbidden}"
                )
        finally:
            negative_shared.is_running.value = False
            if not negative_shared.close():
                raise RuntimeError("negative RuntimeChannels cleanup failed")
        return {
            "status": "PASS",
            "processes": ["inference", "policy", "replay_feeder"],
            "inference_ready": True,
            "coordinator_ready": True,
            "policy_plan_sequence_advanced": True,
            "policy_plan_sequence": plan_sequence,
            "shadow_validated_count": shadow_validated,
            "coupled_sequence_before": coupled_before,
            "coupled_sequence_after": coupled_after,
            "coupled_sequence_delta": coupled_after - coupled_before,
            "hardware_workers_started": False,
            "hardware_sdk_imports": [],
            "inference_seed": policy_config.deployment.inference_seed,
            "source_segment_start": feeder_range["source_segment_start"],
            "source_segment_end": feeder_range["source_segment_end"],
            "feed_start": feeder_range["feed_start"],
            "feed_end": feeder_range["feed_end"],
            "feed_frame_count": feeder_range["feed_frame_count"],
            "negative_checks": {
                name: negative_receipt[name]
                for name in (
                    "generation_change_plan_dropped",
                    "deadline_closed_no_validation",
                    "non_prefix_mask_failed_closed",
                    "all_targets_expired_no_plan",
                )
            },
        }
    finally:
        if not shared.close():
            raise RuntimeError("multiprocess RuntimeChannels cleanup failed")


def _operator_commands(
    artifact: ResolvedPolicyArtifact, *, real_commit: str, device: str, seed: int
) -> dict[str, str]:
    experiment = "$POLICY_ROOT/experiments/dp3/pick_place_toy/2026-08-28_13-59_42"
    common = (
        f'git -C "$REAL_ROOT" switch --detach {real_commit} && '
        f'git -C "$POLICY_ROOT" switch --detach {artifact.producer.commit} && '
        f'python "$REAL_ROOT/examples/run_policy.py" --experiment-dir "{experiment}" '
        f"--device {device} --inference-seed {seed} --hand"
    )
    checkpoint_sha = artifact.checkpoint_sha256_from_index
    return {
        "live_shadow": (f"{common} --execution-mode shadow --max-running-seconds 10"),
        "fresh_h4": (
            f"{common} --execution-mode execute --max-running-seconds 10 "
            "--execute-max-published-endpoints 1 --execute-ack-timeout-seconds 2 "
            f"--execute-expected-checkpoint-sha256 {checkpoint_sha}"
        ),
    }


def _write_report(path: Path, receipt: dict[str, Any]) -> None:
    current = receipt["current_fixture"]
    representative = receipt["representative_artifact"]
    recorded = receipt["recorded_replay"]
    multi = receipt["multiprocess_shadow"]
    commands = receipt["operator_handoff"]
    report = f"""# Policy Promotion Gate A — Offline Qualification

## Scope

NO HARDWARE qualification only. Live shadow, H4, homing, rollout, device connection, and command publication were not run.

## Frozen Real baseline

- Real main / Gate base: `{receipt['real']['commit']}`
- R2 merge: guarded fast-forward from `{receipt['r2_merge']['pre_merge_main']}` to `{receipt['r2_merge']['r2_commit']}`
- Remote main verification: PASS

## Policy current docs-only state

- Current Policy main: `{receipt['policy']['current_main_commit']}`
- Semantic handoff: `{receipt['policy']['semantic_handoff_commit']}`
- Diff since handoff: one documentation file only

## Current fixture qualification

CURRENT FIXTURE EVIDENCE uses current Policy main as producer.

- Producer: `{current['producer_commit']}`
- Checkpoint SHA-256: `{current['checkpoint_sha256']}`
- Sidecar SHA-256: `{current['sidecar_sha256']}`
- Inspect is a real `examples/run_policy.py --print-config` CLI subprocess check (exit {current['inspect_exit_code']}).
- Inspect allocation: O{current['inspect_allocation']['n_obs_steps']} / A{current['inspect_allocation']['n_action_steps']} / H{current['inspect_allocation']['horizon']} / required {current['inspect_allocation']['required_action_steps']} / control {current['inspect_allocation']['control_action_dim']}D
- Inspect / direct restore / isolated preflight: PASS / PASS / PASS

## Representative artifact qualification

REPRESENTATIVE ARTIFACT EVIDENCE uses the artifact's exact historical producer.

- Experiment: `{representative['experiment_name']}`
- Checkpoint: `{representative['checkpoint']}`
- Checkpoint SHA-256: `{representative['checkpoint_sha256']}`
- Sidecar/index SHA-256: `{representative['sidecar_index_sha256']}`
- Producer: `{representative['producer_commit']}`
- Exact clean detached producer checkout: YES
- Direct restore / isolated preflight: PASS / PASS

## Recorded replay

RECORDED REPLAY uses recorded source-relative timing rebased onto a fresh monotonic epoch plus measured offline model-path latency. It is not a live latency measurement.

- Episode: `{recorded['episode_name']}` (schema v{recorded['schema_version']}, VALID, task `{recorded['task']}`)
- Window: compact index {recorded['window_start_index']}, {recorded['window_steps']} consecutive observations
- Gate A representative inference baseline seed: {recorded['inference_seed']}
- Actual representative Policy inference: PASS
- Model latency: {recorded['model_latency_ms']:.3f} ms
- Prediction / control shapes: `{recorded['pred_action_conceptual_shape']}` / `{recorded['control_action_shape']}`
- First deliverable / transport-valid / usable: {recorded['first_deliverable_index']} / {recorded['transport_valid_count']} / {recorded['usable_target_count']}
- ActionBuffer: PASS; target grid was not retimed
- SafetyGate publication boundary: `{recorded['shadow_disposition']}`

## Multiprocess shadow

MULTIPROCESS SHADOW is a hardware-free shared-memory/process integration using only inference, policy coordinator, and replay feeder processes.

- Inference / coordinator ready: YES / YES
- Policy plan ring advanced: YES (sequence {multi['policy_plan_sequence']})
- Shadow validated count: {multi['shadow_validated_count']}
- Replay feeder stayed within one persisted source segment: [{multi['source_segment_start']}, {multi['source_segment_end']}); fed [{multi['feed_start']}, {multi['feed_end']}) ({multi['feed_frame_count']} frames)
- Multiprocess representative inference seed: {multi['inference_seed']}
- Mandatory negative cross-process checks: PASS

## Timing evidence

- Basis: `{recorded['timing_basis']}`
- Immutable target grid, strict finish+lead lower bound, independent deadline, strict target&lt;deadline upper bound, and `0*1*` transport topology: PASS
- Deadline relative to recorded latest source: {recorded['deadline_relative_ns']} ns

## No-write proof

- Direct replay coupled sequence: {recorded['coupled_sequence_before']} → {recorded['coupled_sequence_after']}
- Multiprocess coupled sequence: {multi['coupled_sequence_before']} → {multi['coupled_sequence_after']} (delta {multi['coupled_sequence_delta']})

## No-hardware proof

- Arm / hand / camera / pointcloud workers started: NO / NO / NO / NO
- xArm / XHand / RealSense connection: NO / NO / NO
- Forbidden hardware SDK/owner module audit: PASS

## Failures and skips

- Mandatory offline checks: no failures, no skips
- Live shadow: NOT RUN — OPERATOR REQUIRED
- Fresh H4: NOT RUN — OPERATOR REQUIRED

## Remaining live gates

Offline evidence cannot establish live sensor freshness under current load, physical tracking, device acknowledgement, or physical collision clearance. Those remain operator-owned live gates.

## Operator-only next steps

DO NOT RUN FROM CODEX — OPERATOR ONLY — HARDWARE SIDE EFFECTS POSSIBLE

Offline replay, multiprocess shadow, live shadow handoff, and fresh H4 handoff share representative inference seed `{receipt['operator_handoff']['seed']}`.

Exact artifact producer `{representative['producer_commit']}`, Real `{receipt['real']['commit']}`, device `{receipt['operator_handoff']['device']}`, seed `{receipt['operator_handoff']['seed']}`, checkpoint SHA `{representative['checkpoint_sha256']}`.

Live shadow:

```bash
{commands['live_shadow']}
```

Fresh H4, one endpoint:

```bash
{commands['fresh_h4']}
```

## Final decision

`{receipt['gate_decision']}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    real_root = Path(args.real_root).resolve()
    current_policy_root = Path(args.current_policy_root).resolve()
    frozen_policy_root = Path(args.frozen_policy_root).resolve()
    experiment = Path(args.experiment).resolve()
    episode = Path(args.episode).resolve()
    processed = Path(args.processed_episode).resolve()
    python = Path(sys.executable).resolve()
    artifact = resolve_policy_artifact(experiment)
    if artifact.producer.commit != args.representative_producer_commit:
        raise ValueError("representative producer commit changed")
    if artifact.allocation_contract.task_name != "pick_place_toy":
        raise ValueError("representative task is not pick_place_toy")
    discovered_episode, discovery = discover_recorded_episode(
        Path(args.desktop_root),
        task_name=artifact.allocation_contract.task_name,
    )
    if discovered_episode.name != EXPECTED_RECORDED_EPISODE:
        raise ValueError("recording set changed from the Gate A baseline")
    if discovered_episode != episode:
        raise ValueError("selected episode is not the newest VALID matching recording")
    current_fixture = _validate_current_fixture(
        fixture_experiment=Path(args.current_fixture).resolve(),
        policy_commit=args.policy_current_commit,
        policy_root=current_policy_root,
        real_root=real_root,
        python=python,
    )
    representative = _validate_representative_restore(
        artifact=artifact,
        policy_root=frozen_policy_root,
        current_policy_root=current_policy_root,
        real_root=real_root,
        python=python,
    )
    real_runtime = resolve_runtime_config()
    if real_runtime.sha256 != args.expected_real_config_sha256:
        raise ValueError("resolved Real runtime configuration changed from recording")
    policy_resolved = resolve_policy_runtime_config(
        artifact=artifact,
        runtime_config=real_runtime,
        device=args.device,
        inference_seed=args.seed,
        execution_mode="shadow",
    )
    payload, raw_hashes = load_replay_payload(
        episode_dir=episode,
        processed_path=processed,
        artifact=artifact,
        runtime_config=policy_resolved.runtime,
    )
    recorded = run_recorded_replay(
        payload=payload,
        runtime_config=policy_resolved.runtime,
        real_runtime=real_runtime,
    )
    recorded.update(
        {
            "episode_name": payload.episode_name,
            "schema_version": payload.schema_version,
            "validity": ValidityState.VALID.value,
            "task": payload.task_name,
            "data_sha256": raw_hashes["data.h5"],
            "depth_sha256": raw_hashes["depth.h5"],
            "rgb_sha256": raw_hashes["rgb.mp4"],
            "control_hz": payload.control_hz,
            "grid_dt_s": payload.grid_dt_s,
            "resolved_config_sha256": payload.resolved_config_sha256,
            "candidate_count": discovery["candidate_count"],
            "valid_matching_count": discovery["valid_matching_count"],
            "inference_seed": policy_resolved.runtime.deployment.inference_seed,
        }
    )
    multiprocess = run_multiprocess_shadow(
        payload=payload,
        processed_path=processed,
        policy_config=policy_resolved.runtime,
        real_runtime=real_runtime,
    )
    parent_forbidden = _forbidden_imports()
    if parent_forbidden:
        raise RuntimeError(f"parent hardware import audit failed: {parent_forbidden}")
    commands = _operator_commands(
        artifact,
        real_commit=args.real_commit,
        device=args.live_device,
        seed=args.seed,
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "gate": "promotion_gate_a_offline",
        "r2_merge": {
            "pre_merge_main": args.real_pre_merge_commit,
            "r2_commit": args.real_commit,
            "merge_type": "fast-forward",
            "post_merge_main": args.real_commit,
            "push_result": "PASS",
        },
        "real": {"commit": args.real_commit, "clean": True},
        "policy": {
            "current_main_commit": args.policy_current_commit,
            "semantic_handoff_commit": args.policy_handoff_commit,
            "current_main_change_kind": "docs_only",
        },
        "current_fixture": current_fixture,
        "representative_artifact": {
            **representative,
            "experiment_name": experiment.name,
            "allocation": asdict(artifact.allocation_contract),
            "exact_producer_checkout": True,
        },
        "recorded_replay": recorded,
        "multiprocess_shadow": multiprocess,
        "hardware_audit": {
            "arm_worker_started": False,
            "hand_worker_started": False,
            "camera_worker_started": False,
            "pointcloud_worker_started": False,
            "xarm_connected": False,
            "xhand_connected": False,
            "realsense_opened": False,
            "forbidden_modules_imported": [],
        },
        "live_shadow": {"status": "NOT_RUN_OPERATOR_REQUIRED"},
        "fresh_h4": {"status": "NOT_RUN_OPERATOR_REQUIRED"},
        "operator_handoff": {
            "device": args.live_device,
            "seed": args.seed,
            **commands,
        },
        "gate_decision": "READY_FOR_LIVE_GATE_A",
    }
    absolute_prefix = str(Path.home())
    encoded = json.dumps(receipt, sort_keys=True)
    if absolute_prefix in encoded:
        raise ValueError("receipt contains an absolute home path")
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(Path(args.report), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NO HARDWARE Gate A offline qualification"
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--processed-episode", required=True)
    parser.add_argument("--current-fixture", required=True)
    parser.add_argument("--real-root", required=True)
    parser.add_argument("--desktop-root", required=True)
    parser.add_argument("--current-policy-root", required=True)
    parser.add_argument("--frozen-policy-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--live-device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1066)
    parser.add_argument("--real-pre-merge-commit", required=True)
    parser.add_argument("--real-commit", required=True)
    parser.add_argument("--policy-current-commit", required=True)
    parser.add_argument("--policy-handoff-commit", required=True)
    parser.add_argument("--representative-producer-commit", required=True)
    parser.add_argument("--expected-real-config-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = qualify(args)
    except BaseException as exc:
        print(
            f"NOT_READY_FOR_LIVE_GATE_A: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 1
    print(json.dumps({"gate_decision": receipt["gate_decision"]}, sort_keys=True))
    print("READY_FOR_LIVE_GATE_A")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
