"""Passive, task-only evidence capture for bounded learned-policy rollouts.

This observer has no command, safety-state, readiness, or worker-lifecycle
authority.  It ownership-copies existing shared-memory publications in the
Main process and serializes them only after the task has stopped.  Missing or
late diagnostic data is therefore explicit in the manifest, never repaired by
changing a command, and an observer/persistence failure cannot influence the
robot-control path.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib
from dexmani_real.deployment.task_scene import TaskSceneCard
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.runtime.safety import SafetyState, read_run_epoch
from dexmani_real.utils.atomic_io import atomic_json_dump
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_POLL_INTERVAL_S = 0.004
_MAX_CACHED_PLAN_ENDPOINTS = 1_024


def _json_value(value: Any) -> Any:
    """Convert a copied NumPy value to JSON without silently coercing NaN."""
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    raise TypeError(f"unsupported diagnostic value {type(value).__name__}")


def _structured_snapshot(result: Any) -> dict[str, Any] | None:
    """Render a copied shared-ring result with its sequence/timestamp."""
    if result is None:
        return None
    data, ring_publish_ns, sequence = result
    record = data[0]
    names = record.dtype.names or ()
    return {
        "ring_sequence": int(sequence),
        "ring_publish_monotonic_ns": int(ring_publish_ns),
        "fields": {name: _json_value(record[name]) for name in names},
    }


def _shared_text(value: Any) -> str:
    """Decode a fixed shared metadata buffer without inventing a fallback."""
    raw = bytes(value).rstrip(b"\x00")
    return raw.decode("utf-8", errors="replace") if raw else ""


class TaskDiagnosticsObserver:
    """Copy bounded task evidence from already-published runtime channels.

    Policy plans are sampled continuously so a later coupled command can be
    joined back to its raw model endpoint via
    ``(run_generation, observation_id, scheduled_target_monotonic_ns)``.
    The join is deliberately fail-open for control but fail-visible for review:
    a missed plan is retained as ``raw_prediction_available=false``.
    """

    def __init__(
        self,
        shared: RuntimeChannels,
        *,
        receipt_dir: str | Path,
        scene_card: TaskSceneCard,
    ) -> None:
        self._shared = shared
        self._receipt_dir = Path(receipt_dir)
        self._scene_card = scene_card
        calibration = CameraCalib()
        self._calibration_identity = {
            "path": str(calibration.calib_path),
            "source_sha256": calibration.source_sha256,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_plan_sequence = 0
        self._last_command_sequence = 0
        self._plans: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._pending_by_action_id: dict[int, dict[str, Any]] = {}
        self._scene_frames: dict[str, dict[str, Any]] = {}
        self._armed_scene: dict[str, Any] | None = None
        self._armed_camera_sequence = 0
        self._armed_pointcloud_sequence = 0
        self._seen_run_generation = int(shared.run_generation.value)
        # H can publish a hand-only home target while the runtime is ARMED.
        # Task evidence starts only at B's RUNNING epoch, so retain that
        # generation once observed and reject all pre-B ring records.
        self._task_run_generation: int | None = None
        self._publication_count = 0
        self._dropped_plan_records = 0
        self._evicted_cached_plan_endpoints = 0
        self._dropped_command_records = 0
        self._read_errors: list[str] = []
        self._fatal_error: str | None = None
        self._persisted_path: Path | None = None

    @property
    def persisted_path(self) -> Path | None:
        return self._persisted_path

    def start(self) -> None:
        """Start passive collection; this performs no filesystem IO or SDK IO."""
        if self._thread is not None:
            raise RuntimeError("task diagnostics observer was already started")
        self._thread = threading.Thread(
            target=self._run,
            name="task-diagnostics-observer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> bool:
        """Stop shared-memory reads without writing or changing runtime state."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.error("task diagnostics observer did not stop within 1.0 s")
                return False
        return True

    def persist(self) -> Path | None:
        """Best-effort write after :meth:`stop` released all shared memory."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("task diagnostics cannot persist while observer is live")
        try:
            self._persisted_path = self._persist()
        except Exception:
            logger.error("task diagnostics persistence failed", exc_info=True)
            return None
        logger.info("task diagnostics written: %s", self._persisted_path)
        return self._persisted_path

    def stop_and_persist(self) -> Path | None:
        """Compatibility helper for non-lifecycle callers."""
        return self.persist() if self.stop() else None

    def _record_read_error(self, label: str, exc: Exception) -> None:
        if len(self._read_errors) < 32:
            self._read_errors.append(f"{label}:{type(exc).__name__}:{exc}")

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._capture_armed_scene()
                self._capture_b_pre_scene()
                self._drain_policy_plans()
                self._drain_coupled_commands()
                self._observe_acknowledgements()
                self._stop.wait(_POLL_INTERVAL_S)
        except Exception as exc:
            # This observer is intentionally not a supervisor participant.
            self._fatal_error = f"{type(exc).__name__}: {exc}"
            logger.error("task diagnostics observer stopped", exc_info=True)

    def _capture_armed_scene(self) -> None:
        if int(self._shared.safety_state.value) != int(SafetyState.ARMED):
            return
        camera_sequence = int(self._shared.camera_ring.latest_sequence)
        pointcloud_sequence = int(self._shared.pointcloud_ring.latest_sequence)
        if (
            self._armed_scene is not None
            and camera_sequence == self._armed_camera_sequence
            and pointcloud_sequence == self._armed_pointcloud_sequence
        ):
            return
        try:
            camera = self._camera_snapshot()
            pointcloud = _structured_snapshot(
                self._shared.pointcloud_ring.read_latest()
            )
            if camera is None and pointcloud is None:
                return
            self._armed_scene = {
                "captured_monotonic_ns": time.monotonic_ns(),
                "camera": camera,
                "pointcloud": pointcloud,
                "arm_feedback": _structured_snapshot(
                    self._shared.arm_state_ring.read_latest()
                ),
                "hand_feedback": _structured_snapshot(
                    self._shared.hand_state_ring.read_latest()
                ),
                "hand_tactile": _structured_snapshot(
                    self._shared.hand_tactile_ring.read_latest()
                ),
            }
            self._armed_camera_sequence = camera_sequence
            self._armed_pointcloud_sequence = pointcloud_sequence
        except Exception as exc:
            self._record_read_error("armed_scene", exc)

    def _capture_b_pre_scene(self) -> None:
        run_epoch = read_run_epoch(self._shared)
        generation = run_epoch.generation
        if (
            generation == self._seen_run_generation
            and self._task_run_generation is not None
        ):
            return
        self._seen_run_generation = generation
        if run_epoch.started_monotonic_ns <= 0:
            return
        self._task_run_generation = generation
        # The cache was only populated while ARMED, so it cannot include B's
        # new observation epoch. Keep its capture timestamp for reviewer aging.
        if self._armed_scene is None:
            self._scene_frames["b_pre"] = {"available": False}
        else:
            self._scene_frames["b_pre"] = {
                "available": True,
                "run_generation": generation,
                "run_started_monotonic_ns": run_epoch.started_monotonic_ns,
                "device_and_calibration_identity": self._identity_snapshot(),
                **self._armed_scene,
            }

    def _identity_snapshot(self) -> dict[str, Any]:
        """Snapshot static device/calibration facts alongside the B-pre image."""
        return {
            "camera_calibration": self._calibration_identity,
            "camera_depth_scale_m_per_unit": float(
                self._shared.camera_depth_scale.value
            ),
            "camera_serial": _shared_text(self._shared.camera_serial.value),
            "camera_firmware": _shared_text(self._shared.camera_firmware.value),
            "camera_sdk_version": _shared_text(self._shared.camera_sdk_version.value),
            "camera_profile_json": _shared_text(self._shared.camera_profile.value),
            "camera_geometry_json": _shared_text(self._shared.camera_geometry.value),
            "arm_device_identity_json": _shared_text(
                self._shared.arm_device_identity.value
            ),
            "hand_device_identity_json": _shared_text(
                self._shared.hand_device_identity.value
            ),
        }

    def _drain_policy_plans(self) -> None:
        ring = self._shared.policy_plan_ring
        latest = int(ring.latest_sequence)
        if latest <= self._last_plan_sequence:
            return
        first = max(self._last_plan_sequence + 1, latest - int(ring.maxlen) + 1)
        self._dropped_plan_records += max(0, first - self._last_plan_sequence - 1)
        for sequence in range(first, latest + 1):
            result = ring.read_sequence(sequence)
            if result is None:
                self._dropped_plan_records += 1
                continue
            record = result[0][0]
            try:
                generation = int(record["run_generation"])
                if generation != self._task_run_generation:
                    continue
                observation_id = int(record["observation_id"])
                plan_id = int(record["plan_id"])
                steps = int(record["num_steps"])
                for step_index in range(steps):
                    if int(record["valid_mask"][step_index]) != 1:
                        continue
                    target_ns = int(record["target_monotonic_ns"][step_index])
                    if target_ns <= 0:
                        continue
                    key = (generation, observation_id, target_ns)
                    self._plans[key] = {
                        "plan_id": plan_id,
                        "step_index": step_index,
                        "run_generation": generation,
                        "observation_id": observation_id,
                        "observation_anchor_monotonic_ns": int(
                            record["observation_anchor_monotonic_ns"]
                        ),
                        "observation_latest_source_monotonic_ns": int(
                            record["observation_latest_source_monotonic_ns"]
                        ),
                        "inference_started_monotonic_ns": int(
                            record["inference_started_monotonic_ns"]
                        ),
                        "inference_finished_monotonic_ns": int(
                            record["inference_finished_monotonic_ns"]
                        ),
                        "arm_present": bool(record["arm_present"]),
                        "ee_present": bool(record["ee_present"]),
                        "raw_arm_qpos": _json_value(record["arm_qpos"][step_index]),
                        "raw_hand_qpos": _json_value(record["hand_qpos"][step_index]),
                        "raw_ee_pos": _json_value(record["ee_pos"][step_index]),
                        "raw_ee_rot6d": _json_value(record["ee_rot6d"][step_index]),
                        "scheduled_target_monotonic_ns": target_ns,
                    }
                    while len(self._plans) > _MAX_CACHED_PLAN_ENDPOINTS:
                        self._plans.pop(next(iter(self._plans)))
                        self._evicted_cached_plan_endpoints += 1
            except Exception as exc:
                self._record_read_error("policy_plan", exc)
        self._last_plan_sequence = latest

    def _drain_coupled_commands(self) -> None:
        ring = self._shared.coupled_cmd_ring
        latest = int(ring.latest_sequence)
        if latest <= self._last_command_sequence:
            return
        first = max(self._last_command_sequence + 1, latest - int(ring.maxlen) + 1)
        self._dropped_command_records += max(0, first - self._last_command_sequence - 1)
        for sequence in range(first, latest + 1):
            result = ring.read_sequence(sequence)
            if result is None:
                self._dropped_command_records += 1
                continue
            record = result[0][0]
            try:
                action_id = int(record["action_id"])
                if action_id <= 0 or bool(record["is_hold"]):
                    continue
                generation = int(record["run_generation"])
                if generation != self._task_run_generation:
                    continue
                observation_id = int(record["observation_id"])
                scheduled_ns = int(record["scheduled_target_monotonic_ns"])
                raw_prediction = self._plans.get(
                    (generation, observation_id, scheduled_ns)
                )
                self._publication_count += 1
                event: dict[str, Any] = {
                    "action_id": action_id,
                    "published_endpoint_index": self._publication_count,
                    "run_generation": generation,
                    "observation_id": observation_id,
                    "coupled_command_ring_sequence": int(sequence),
                    "raw_prediction_available": raw_prediction is not None,
                    "raw_prediction": raw_prediction,
                    "shaped_ipc_endpoint": {
                        "created_monotonic_ns": int(record["created_monotonic_ns"]),
                        "scheduled_target_monotonic_ns": scheduled_ns,
                        "target_monotonic_ns": int(record["target_monotonic_ns"]),
                        "valid_until_monotonic_ns": int(
                            record["valid_until_monotonic_ns"]
                        ),
                        "arm_present": bool(record["arm_present"]),
                        "hand_present": bool(record["hand_present"]),
                        "arm_qpos": _json_value(record["arm_qpos"]),
                        "hand_qpos": _json_value(record["hand_qpos"]),
                    },
                    "publication_observed_monotonic_ns": time.monotonic_ns(),
                    "arm_feedback_at_publication": _structured_snapshot(
                        self._shared.arm_state_ring.read_latest()
                    ),
                    "hand_feedback_at_publication": _structured_snapshot(
                        self._shared.hand_state_ring.read_latest()
                    ),
                    "tactile_at_publication": _structured_snapshot(
                        self._shared.hand_tactile_ring.read_latest()
                    ),
                    "acknowledgement": None,
                }
                self._events.append(event)
                self._pending_by_action_id[action_id] = event
                self._capture_phase_scene(event)
            except Exception as exc:
                self._record_read_error("coupled_command", exc)
        self._last_command_sequence = latest

    def _capture_phase_scene(self, event: dict[str, Any]) -> None:
        endpoint_index = int(event["published_endpoint_index"])
        for phase, milestone in self._scene_card.phase_endpoint_indices:
            if endpoint_index != milestone:
                continue
            try:
                self._scene_frames[phase] = {
                    "available": True,
                    "phase_endpoint_index": endpoint_index,
                    "action_id": int(event["action_id"]),
                    "captured_monotonic_ns": time.monotonic_ns(),
                    "camera": self._camera_snapshot(),
                    "pointcloud": _structured_snapshot(
                        self._shared.pointcloud_ring.read_latest()
                    ),
                }
            except Exception as exc:
                self._scene_frames[phase] = {"available": False}
                self._record_read_error(f"phase_{phase}", exc)

    def _observe_acknowledgements(self) -> None:
        if not self._pending_by_action_id:
            return
        try:
            arm = _structured_snapshot(self._shared.arm_state_ring.read_latest())
            hand = _structured_snapshot(self._shared.hand_state_ring.read_latest())
            tactile = _structured_snapshot(self._shared.hand_tactile_ring.read_latest())
        except Exception as exc:
            self._record_read_error("acknowledgement", exc)
            return
        arm_fields = None if arm is None else arm["fields"]
        hand_fields = None if hand is None else hand["fields"]
        if arm_fields is None or hand_fields is None:
            return
        observed_ns = time.monotonic_ns()
        for action_id, event in list(self._pending_by_action_id.items()):
            arm_action_id = int(arm_fields["last_cmd_seq"])
            hand_action_id = int(hand_fields["accepted_target_action_id"])
            if arm_action_id < action_id or hand_action_id < action_id:
                continue
            event["acknowledgement"] = {
                "observed_monotonic_ns": observed_ns,
                "arm_acknowledged": arm_action_id == action_id,
                "hand_acknowledged": hand_action_id == action_id,
                "arm_feedback": arm,
                "hand_feedback": hand,
                "tactile": tactile,
            }
            del self._pending_by_action_id[action_id]

    def _camera_snapshot(self) -> dict[str, Any] | None:
        result = self._shared.camera_ring.read_latest()
        if result is None:
            return None
        header, rgb, depth, sequence = result
        return {
            "ring_sequence": int(sequence),
            "header": {
                name: _json_value(header[0][name]) for name in header.dtype.names or ()
            },
            "rgb": rgb,
            "depth": depth,
        }

    def _persist(self) -> Path:
        output = self._receipt_dir / (
            f"task_diagnostics_{time.time_ns()}_{os.getpid()}_{uuid.uuid4().hex}"
        )
        output.mkdir(parents=True, exist_ok=False)
        archive_values: dict[str, np.ndarray] = {}
        scene_manifest: dict[str, Any] = {}
        for label, snapshot in self._scene_frames.items():
            entry = dict(snapshot)
            camera = entry.pop("camera", None)
            if camera is not None:
                rgb = camera.pop("rgb", None)
                depth = camera.pop("depth", None)
                if rgb is not None:
                    archive_values[f"{label}_rgb"] = np.asarray(rgb)
                if depth is not None:
                    archive_values[f"{label}_depth"] = np.asarray(depth)
                entry["camera"] = camera
            pointcloud = entry.pop("pointcloud", None)
            if pointcloud is not None:
                fields = dict(pointcloud["fields"])
                cloud = fields.pop("point_cloud", None)
                if cloud is not None:
                    archive_values[f"{label}_point_cloud"] = np.asarray(cloud)
                pointcloud["fields"] = fields
                entry["pointcloud"] = pointcloud
            scene_manifest[label] = entry
        if archive_values:
            archive_path = output / "scene_frames.npz"
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=output,
                prefix=".scene_frames_",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(stream, **archive_values)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, archive_path)
        atomic_json_dump(
            {
                "schema_version": 1,
                "observer_kind": "passive_shared_memory_copy",
                "scene_card": {
                    **self._scene_card.provenance(),
                    "object_description": self._scene_card.object_description,
                    "object_start_description": self._scene_card.object_start_description,
                    "target_description": self._scene_card.target_description,
                    "success_criterion": self._scene_card.success_criterion,
                },
                "collection": {
                    "event_count": len(self._events),
                    "pending_acknowledgements_at_stop": sorted(
                        self._pending_by_action_id
                    ),
                    "dropped_policy_plan_records": self._dropped_plan_records,
                    "evicted_cached_plan_endpoints": self._evicted_cached_plan_endpoints,
                    "dropped_coupled_command_records": self._dropped_command_records,
                    "read_errors": self._read_errors,
                    "fatal_error": self._fatal_error,
                },
                "scene_frames": scene_manifest,
                "events": self._events,
            },
            output / "manifest.json",
            ensure_ascii=False,
        )
        return output
