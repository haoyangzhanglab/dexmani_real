"""Transactional batch pipeline from Real schema-v17 episodes to Sim-label HDF5."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import yaml

from dexmani_real.data_processing.cleaning import analyze_episode
from dexmani_real.data_processing.contracts import EpisodeAnnotation, EpisodeDecision, OutputProfile, ProcessingConfig
from dexmani_real.data_processing.transforms import resize_camera_intrinsic, resize_point_cloud, resize_rgb
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.episode_schema import EPISODE_SCHEMA_VERSION
from dexmani_real.recording.transaction import atomic_publish
from dexmani_real.sensor.pointcloud_processor import PointCloudProcessor, PointCloudProcessorConfig
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.pointcloud_utils import depth_to_meters, make_rays

logger = get_logger(__name__)

_SCHEMA_NAME = "dexmani-real-simlabel-hdf5"
_SCHEMA_VERSION = 1
_SOURCE_MEMBERS = ("data.h5", "depth.h5", "rgb.mp4")
_OMITTED_SIM_LABELS = (
    "depth",
    "segmentation",
    "camera_extrinsic",
    "contact_force",
    "fingertip_points",
    "imagine_point_cloud",
    "action_ee",
    "done",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_ranges(value: Any, *, label: str) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of [start, end] ranges")
    ranges: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{label} entries must be [start, end]")
        if any(not isinstance(bound, int) or isinstance(bound, bool) for bound in item):
            raise ValueError(f"{label} bounds must be integers")
        ranges.append((item[0], item[1]))
    return tuple(ranges)


def load_annotations(path: str | Path | None) -> dict[str, EpisodeAnnotation]:
    """Load strict optional per-episode annotations from YAML."""

    if path is None:
        return {}
    annotation_path = Path(path)
    with annotation_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError("annotation YAML root must be a mapping")
    raw_episodes = payload.get("episodes", payload)
    if not isinstance(raw_episodes, dict):
        raise ValueError("annotation episodes must be a mapping")
    result: dict[str, EpisodeAnnotation] = {}
    allowed = {
        "include",
        "task_name",
        "task_outcome",
        "include_ranges",
        "exclude_ranges",
    }
    for episode_name, raw in raw_episodes.items():
        if not isinstance(episode_name, str) or not episode_name:
            raise ValueError("annotation episode names must be non-empty strings")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"annotation for {episode_name} must be a mapping")
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"annotation for {episode_name} has unknown keys: {sorted(unknown)}")
        include = raw.get("include", True)
        if not isinstance(include, bool):
            raise ValueError(f"{episode_name}.include must be a boolean")
        task_name = raw.get("task_name")
        if task_name is not None and not isinstance(task_name, str):
            raise ValueError(f"{episode_name}.task_name must be a string or null")
        task_outcome = raw.get("task_outcome", "unknown")
        if not isinstance(task_outcome, str):
            raise ValueError(f"{episode_name}.task_outcome must be a string")
        result[episode_name] = EpisodeAnnotation(
            include=include,
            task_name=task_name,
            task_outcome=task_outcome,
            include_ranges=_parse_ranges(raw.get("include_ranges"), label=f"{episode_name}.include_ranges"),
            exclude_ranges=_parse_ranges(raw.get("exclude_ranges"), label=f"{episode_name}.exclude_ranges"),
        )
    return result


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """Return sorted visible direct child directories for explicit auditing."""

    root = Path(input_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    episodes = tuple(sorted(child for child in root.iterdir() if child.is_dir() and not child.name.startswith(".")))
    if not episodes:
        raise FileNotFoundError(f"no episode directories found in {root}")
    return episodes


def _output_name(source_name: str, segment_index: int, segment_count: int) -> str:
    if segment_count == 1:
        return f"{source_name}.h5"
    return f"{source_name}__seg{segment_index:03d}.h5"


@contextmanager
def _open_episode(path: Path) -> Iterator[EpisodeReader]:
    """Open one published schema-v17 episode directory."""
    with EpisodeReader(path) as reader:
        yield reader


def _dataset_kwargs(config: ProcessingConfig, *, chunks: tuple[int, ...]) -> dict[str, Any]:
    return {
        "compression": "gzip",
        "compression_opts": config.gzip_level,
        "chunks": chunks,
    }


def _write_common_attrs(
    output: h5py.File,
    *,
    reader: EpisodeReader,
    decision: EpisodeDecision,
    segment_index: int,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation,
    source_hashes: dict[str, str],
) -> None:
    segment = decision.segments[segment_index]
    meta = {key: reader.h5f["meta"].attrs[key] for key in reader.h5f["meta"].attrs}
    source_task = str(meta.get("task_label", "")).strip()
    task_name = annotation.task_name or source_task or "unknown"
    segment_task_outcome = annotation.task_outcome if len(decision.segments) == 1 else "unknown"
    output.attrs["schema_name"] = _SCHEMA_NAME
    output.attrs["schema_version"] = _SCHEMA_VERSION
    output.attrs["domain"] = "real"
    output.attrs["source_format"] = "dexmani_v17"
    output.attrs["source_schema_version"] = EPISODE_SCHEMA_VERSION
    output.attrs["profile"] = config.profile.value
    output.attrs["episode_steps"] = segment.length
    output.attrs["dt"] = float(reader.timing.grid_dt_s)
    output.attrs["action_dim"] = 19
    output.attrs["action_space"] = "joint"
    output.attrs["obs_alignment"] = "obs[t]_before_action[t]"
    output.attrs["observation_boundary"] = "real_causal_sources_at_control_grid_anchor"
    output.attrs["arm_action_boundary"] = "forwarded_to_worker_not_hardware_ack"
    output.attrs["hand_action_boundary"] = "queued_target_without_ack"
    output.attrs["source_episode"] = reader.h5_path.name
    output.attrs["source_frame_start"] = segment.start
    output.attrs["source_frame_end_exclusive"] = segment.end
    output.attrs["source_segment_index"] = segment_index
    output.attrs["source_segment_count"] = len(decision.segments)
    output.attrs["task_name"] = task_name
    output.attrs["task_outcome"] = segment_task_outcome
    output.attrs["source_task_outcome"] = annotation.task_outcome
    output.attrs["point_cloud_frame"] = "xarm_base" if config.profile.needs_pointcloud else "omitted"
    output.attrs["frame_compatibility_with_sim_world"] = False
    output.attrs["processing_config_json"] = _json(config.to_dict())
    output.attrs["quality_summary_json"] = _json(segment.quality)
    output.attrs["source_decision_json"] = _json(
        {
            "hard_reason_counts": decision.hard_reason_counts,
            "boundary_counts": decision.boundary_counts,
            "warnings": list(decision.warnings),
        }
    )
    output.attrs["source_member_sha256_json"] = _json(source_hashes)
    output.attrs["source_resolved_config_sha256"] = str(meta.get("resolved_config_sha256", "unknown"))
    output.attrs["omitted_sim_labels_json"] = _json(_OMITTED_SIM_LABELS)
    if config.profile.needs_rgb:
        output.attrs["rgb_transform"] = "resize_no_crop"
        output.attrs["rgb_interpolation"] = "INTER_AREA_down_INTER_LINEAR_up"
    if config.profile.needs_pointcloud:
        output.attrs["point_cloud_transform"] = "source_unique_prefix_then_deterministic_fps_or_cyclic_pad"


def _create_output_files(
    reader: EpisodeReader,
    decision: EpisodeDecision,
    output_root: Path,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation,
) -> list[tuple[Path, h5py.File]]:
    source_hashes = {member: _sha256_file(reader.h5_path / member) for member in _SOURCE_MEMBERS}
    outputs: list[tuple[Path, h5py.File]] = []
    try:
        for segment_index, segment in enumerate(decision.segments):
            path = output_root / _output_name(reader.h5_path.name, segment_index, len(decision.segments))
            output = h5py.File(path, "w")
            try:
                _write_common_attrs(
                    output,
                    reader=reader,
                    decision=decision,
                    segment_index=segment_index,
                    config=config,
                    annotation=annotation,
                    source_hashes=source_hashes,
                )
                length = segment.length
                numeric_chunk = (min(length, 256),)
                output.create_dataset(
                    "joint_state",
                    shape=(length, 19),
                    dtype=np.float32,
                    **_dataset_kwargs(config, chunks=(*numeric_chunk, 19)),
                )
                output.create_dataset(
                    "action",
                    shape=(length, 19),
                    dtype=np.float32,
                    **_dataset_kwargs(config, chunks=(*numeric_chunk, 19)),
                )
                if config.profile.needs_rgb:
                    output.create_dataset(
                        "rgb",
                        shape=(length, config.target_rgb_height, config.target_rgb_width, 3),
                        dtype=np.uint8,
                        **_dataset_kwargs(
                            config,
                            chunks=(1, config.target_rgb_height, config.target_rgb_width, 3),
                        ),
                    )
                    output.create_dataset(
                        "camera_intrinsic",
                        shape=(length, 9),
                        dtype=np.float32,
                        **_dataset_kwargs(config, chunks=(*numeric_chunk, 9)),
                    )
                if config.profile.needs_pointcloud:
                    output.create_dataset(
                        "point_cloud",
                        shape=(length, config.target_point_count, 6),
                        dtype=np.float32,
                        **_dataset_kwargs(config, chunks=(1, config.target_point_count, 6)),
                    )
            except BaseException:
                output.close()
                raise
            outputs.append((path, output))
    except BaseException:
        for _, output in outputs:
            output.close()
        raise
    return outputs


def _write_episode_segments(
    reader: EpisodeReader,
    decision: EpisodeDecision,
    output_root: Path,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation,
) -> list[dict[str, Any]]:
    outputs = _create_output_files(reader, decision, output_root, config, annotation)
    try:
        meta = reader.h5f["meta"].attrs
        source_height = int(meta["camera_encoding_height"])
        source_width = int(meta["camera_encoding_width"])
        camera_k = None
        if config.profile.needs_rgb:
            camera_k = resize_camera_intrinsic(
                np.asarray(meta["camera_K"]),
                source_height=source_height,
                source_width=source_width,
                target_height=config.target_rgb_height,
                target_width=config.target_rgb_width,
            )

        # ── Point cloud derivation (schema v17: derived, not recorded) ──
        # Reconstruct the exact processor used at recording time from the
        # persisted pc_* metadata, then deproject the recorded depth with the
        # same intrinsics/extrinsics/desk plane.  The processor is fully
        # deterministic (no RNG), so training and deployment derive an
        # identical cloud from the same depth.
        pointcloud_processor: PointCloudProcessor | None = None
        pointcloud_rays: np.ndarray | None = None
        depth_scale: float = 0.0
        if config.profile.needs_pointcloud:
            source_k = np.asarray(meta["camera_K"], dtype=np.float64).reshape(3, 3)
            t_world_camera = np.asarray(meta["camera_T_world_camera"], dtype=np.float64).reshape(4, 4)
            pc_meta = json.loads(str(meta.get("camera_pointcloud_config_json", "{}")) or "{}")
            pointcloud_processor = PointCloudProcessor(
                t_world_camera,
                PointCloudProcessorConfig.from_meta_dict(pc_meta),
            )
            pointcloud_rays = make_rays(source_height, source_width, source_k).numpy()
            depth_scale = float(meta["depth_scale"])

        for segment_index, (_, output) in enumerate(outputs):
            segment = decision.segments[segment_index]
            arm_state = np.asarray(reader.h5f["arm_qpos"][segment.start : segment.end], dtype=np.float32)
            hand_state = np.asarray(reader.h5f["hand_qpos"][segment.start : segment.end], dtype=np.float32)
            arm_action = np.asarray(
                reader.h5f["action_arm_joint_sent"][segment.start : segment.end],
                dtype=np.float32,
            )
            hand_action = np.asarray(
                reader.h5f["action_hand_joint"][segment.start : segment.end],
                dtype=np.float32,
            )
            output["joint_state"][:] = np.concatenate((arm_state, hand_state), axis=1)
            output["action"][:] = np.concatenate((arm_action, hand_action), axis=1)
            if camera_k is not None:
                output["camera_intrinsic"][:] = camera_k[None, :]

            if config.profile.needs_pointcloud:
                assert pointcloud_processor is not None
                assert pointcloud_rays is not None
                for target_index, source_index in enumerate(range(segment.start, segment.end)):
                    depth_m = depth_to_meters(
                        np.asarray(reader.h5f["depth"][source_index], dtype=np.uint16),
                        depth_scale=depth_scale,
                    )
                    rgb = reader.read_camera_frame("rgb", source_index)
                    cloud = pointcloud_processor.process(depth_m, rgb, pointcloud_rays)
                    if cloud is None:
                        raise ValueError(f"{reader.h5_path.name}: derived point cloud empty at frame {source_index}")
                    output["point_cloud"][target_index] = resize_point_cloud(
                        cloud,
                        source_point_count=int(pointcloud_processor.last_source_point_count),
                        target_point_count=config.target_point_count,
                    )

        if config.profile.needs_rgb:
            segment_index = 0
            decoded_count = 0
            written_counts = [0] * len(decision.segments)
            for source_index, frame in enumerate(reader.iter_camera_frames("rgb")):
                decoded_count += 1
                while segment_index < len(decision.segments) and source_index >= decision.segments[segment_index].end:
                    segment_index += 1
                if segment_index >= len(decision.segments):
                    continue
                segment = decision.segments[segment_index]
                if source_index < segment.start:
                    continue
                outputs[segment_index][1]["rgb"][source_index - segment.start] = resize_rgb(
                    frame,
                    height=config.target_rgb_height,
                    width=config.target_rgb_width,
                )
                written_counts[segment_index] += 1
            if decoded_count != decision.source_frames:
                raise ValueError(
                    f"decoded RGB frame count {decoded_count} does not match source grid {decision.source_frames}"
                )
            expected_counts = [segment.length for segment in decision.segments]
            if written_counts != expected_counts:
                raise ValueError(f"written RGB counts {written_counts} do not match segments {expected_counts}")

        result: list[dict[str, Any]] = []
        for segment_index, (path, output) in enumerate(outputs):
            output.flush()
            segment = decision.segments[segment_index]
            result.append(
                {
                    "path": path.name,
                    "source_episode": reader.h5_path.name,
                    "source_frame_start": segment.start,
                    "source_frame_end_exclusive": segment.end,
                    "frames": segment.length,
                    "full_window_count": segment.full_window_count,
                }
            )
        return result
    finally:
        for _, output in outputs:
            output.close()


def validate_processed_hdf5(path: str | Path, config: ProcessingConfig) -> dict[str, Any]:
    """Fail closed on a written Sim-label HDF5 artifact."""

    artifact = Path(path)
    with h5py.File(artifact, "r") as source:
        top_level = tuple(sorted(source.keys()))
        expected = tuple(sorted(config.profile.dataset_keys))
        if top_level != expected:
            raise ValueError(f"{artifact.name}: top-level keys {top_level} do not match {expected}")
        length = int(source.attrs.get("episode_steps", -1))
        if length < config.min_segment_frames:
            raise ValueError(f"{artifact.name}: episode_steps={length} is below {config.min_segment_frames}")
        for key in expected:
            dataset = source[key]
            if not isinstance(dataset, h5py.Dataset) or dataset.shape[0] != length:
                raise ValueError(f"{artifact.name}: {key} is not an aligned dataset")
            if dataset.compression != "gzip" or int(dataset.compression_opts) != config.gzip_level:
                raise ValueError(f"{artifact.name}: {key} compression does not match gzip-{config.gzip_level}")
        expected_shapes: dict[str, tuple[int, ...]] = {
            "joint_state": (length, 19),
            "action": (length, 19),
        }
        expected_dtypes: dict[str, np.dtype[Any]] = {
            "joint_state": np.dtype(np.float32),
            "action": np.dtype(np.float32),
        }
        if config.profile.needs_rgb:
            expected_shapes.update(
                {
                    "rgb": (
                        length,
                        config.target_rgb_height,
                        config.target_rgb_width,
                        3,
                    ),
                    "camera_intrinsic": (length, 9),
                }
            )
            expected_dtypes.update({"rgb": np.dtype(np.uint8), "camera_intrinsic": np.dtype(np.float32)})
        if config.profile.needs_pointcloud:
            expected_shapes["point_cloud"] = (length, config.target_point_count, 6)
            expected_dtypes["point_cloud"] = np.dtype(np.float32)
        for key in expected:
            if source[key].shape != expected_shapes[key] or source[key].dtype != expected_dtypes[key]:
                raise ValueError(
                    f"{artifact.name}: {key} got {source[key].shape}/{source[key].dtype}, "
                    f"expected {expected_shapes[key]}/{expected_dtypes[key]}"
                )
            if np.issubdtype(source[key].dtype, np.floating) and not np.all(np.isfinite(source[key][:])):
                raise ValueError(f"{artifact.name}: {key} contains non-finite values")
        if config.profile.needs_pointcloud:
            cloud = np.asarray(source["point_cloud"][:], dtype=np.float32)
            if np.any(cloud[:, :, 3:] < 0.0) or np.any(cloud[:, :, 3:] > 1.0):
                raise ValueError(f"{artifact.name}: point_cloud RGB is outside [0,1]")
            if np.any(~np.any(np.linalg.norm(cloud[:, :, :3], axis=2) > 0.0, axis=1)):
                raise ValueError(f"{artifact.name}: point_cloud contains an all-zero frame")
        if str(source.attrs.get("schema_name", "")) != _SCHEMA_NAME:
            raise ValueError(f"{artifact.name}: invalid schema_name")
        if int(source.attrs.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError(f"{artifact.name}: invalid schema_version")
        if str(source.attrs.get("domain", "")) != "real":
            raise ValueError(f"{artifact.name}: domain must remain real")
        source_format = str(source.attrs.get("source_format", ""))
        if source_format != "dexmani_v17":
            raise ValueError(f"{artifact.name}: unsupported source_format {source_format!r}")
        if int(source.attrs.get("source_schema_version", -1)) != EPISODE_SCHEMA_VERSION:
            raise ValueError(f"{artifact.name}: invalid source_schema_version")
        if str(source.attrs.get("profile", "")) != config.profile.value:
            raise ValueError(f"{artifact.name}: profile does not match validation config")
        if int(source.attrs.get("action_dim", -1)) != 19 or str(source.attrs.get("action_space", "")) != "joint":
            raise ValueError(f"{artifact.name}: invalid joint action contract")
        dt = float(source.attrs.get("dt", np.nan))
        source_start = int(source.attrs.get("source_frame_start", -1))
        source_end = int(source.attrs.get("source_frame_end_exclusive", -1))
        if not np.isfinite(dt) or dt <= 0.0 or source_start < 0 or source_end - source_start != length:
            raise ValueError(f"{artifact.name}: invalid time or source-range provenance")
        source_segment_index = int(source.attrs.get("source_segment_index", -1))
        source_segment_count = int(source.attrs.get("source_segment_count", -1))
        if not 0 <= source_segment_index < source_segment_count:
            raise ValueError(f"{artifact.name}: invalid source segment provenance")
        if not str(source.attrs.get("source_episode", "")).strip():
            raise ValueError(f"{artifact.name}: source_episode is missing")
        parsed_attrs: dict[str, Any] = {}
        for key in (
            "processing_config_json",
            "quality_summary_json",
            "source_decision_json",
            "source_member_sha256_json",
        ):
            try:
                parsed = json.loads(str(source.attrs[key]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{artifact.name}: invalid {key}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{artifact.name}: {key} must encode a JSON object")
            parsed_attrs[key] = parsed
        if parsed_attrs["processing_config_json"] != config.to_dict():
            raise ValueError(f"{artifact.name}: processing_config_json does not match validation config")
        source_hashes = parsed_attrs["source_member_sha256_json"]
        expected_members = _SOURCE_MEMBERS
        if set(source_hashes) != set(expected_members) or any(
            not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in source_hashes.values()
        ):
            raise ValueError(f"{artifact.name}: invalid source member hashes")
        if config.profile.needs_rgb:
            intrinsic = np.asarray(source["camera_intrinsic"][:], dtype=np.float32)
            if np.any(intrinsic[:, (0, 4)] <= 0.0) or not np.allclose(intrinsic[:, 8], 1.0):
                raise ValueError(f"{artifact.name}: invalid camera intrinsic")
            if not np.allclose(intrinsic, intrinsic[0], rtol=0.0, atol=0.0):
                raise ValueError(f"{artifact.name}: camera intrinsic changes within a segment")
        if config.profile.needs_pointcloud:
            if str(source.attrs.get("point_cloud_frame", "")) != "xarm_base" or bool(
                source.attrs.get("frame_compatibility_with_sim_world", True)
            ):
                raise ValueError(f"{artifact.name}: invalid real point-cloud frame contract")
        return {
            "path": artifact.name,
            "frames": length,
            "keys": list(expected),
        }


def process_episode_root(
    input_root: str | Path,
    output_root: str | Path,
    config: ProcessingConfig,
    *,
    annotations_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Analyze and optionally transactionally publish one processed batch."""

    episodes = discover_episode_dirs(input_root)
    annotations = load_annotations(annotations_path)
    unknown_annotations = set(annotations) - {episode.name for episode in episodes}
    if unknown_annotations:
        raise ValueError(f"annotations reference unknown episodes: {sorted(unknown_annotations)}")

    decisions: list[EpisodeDecision] = []
    for episode in episodes:
        annotation = annotations.get(episode.name, EpisodeAnnotation())
        try:
            with _open_episode(episode) as reader:
                decisions.append(analyze_episode(reader, config, annotation))
        except (FileNotFoundError, OSError, ValueError) as exc:
            logger.warning("episode analysis rejected %s", episode, exc_info=True)
            decisions.append(
                EpisodeDecision(
                    source_path=episode,
                    source_frames=0,
                    profile=config.profile,
                    segments=(),
                    hard_reason_counts={},
                    boundary_counts={},
                    dropped_short_segment_frames=0,
                    selected_frames=0,
                    quality={},
                    rejected_reason=f"{type(exc).__name__}: {exc}",
                )
            )

    planned_output_names = [
        _output_name(decision.source_path.name, segment_index, len(decision.segments))
        for decision in decisions
        if decision.accepted
        for segment_index in range(len(decision.segments))
    ]
    duplicate_output_names = sorted(name for name, count in Counter(planned_output_names).items() if count > 1)
    if duplicate_output_names:
        raise ValueError(f"source names produce colliding output files: {duplicate_output_names}")

    report: dict[str, Any] = {
        "schema_name": _SCHEMA_NAME,
        "schema_version": _SCHEMA_VERSION,
        "input_root": str(Path(input_root).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "dry_run": dry_run,
        "config": config.to_dict(),
        "source_episode_count": len(decisions),
        "accepted_source_episode_count": sum(decision.accepted for decision in decisions),
        "rejected_source_episode_count": sum(not decision.accepted for decision in decisions),
        "output_segment_count": sum(len(decision.segments) for decision in decisions),
        "source_frame_count": sum(decision.source_frames for decision in decisions),
        "selected_frame_count": sum(decision.selected_frames for decision in decisions),
        "episodes": [decision.to_dict() for decision in decisions],
        "outputs": [],
    }
    if dry_run:
        return report
    if not any(decision.accepted for decision in decisions):
        raise ValueError("processing produced no accepted episode segments")

    target = Path(output_root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing processed root: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        outputs: list[dict[str, Any]] = []
        for decision in decisions:
            if not decision.accepted:
                continue
            annotation = annotations.get(decision.source_path.name, EpisodeAnnotation())
            with _open_episode(decision.source_path) as reader:
                outputs.extend(_write_episode_segments(reader, decision, staging, config, annotation))
        validation = [validate_processed_hdf5(staging / item["path"], config) for item in outputs]
        report["outputs"] = outputs
        report["validation"] = validation
        index_path = staging / "processing_index.json"
        with index_path.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.flush()
        atomic_publish(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report
