"""Full-modal conversion into a rebuildable, one-task canonical cache."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

from dexmani_real.dataset.contracts import (
    CANONICAL_FORMAT,
    ProcessingConfig,
    canonical_array_specs,
    validate_task_identity,
)
from dexmani_real.dataset.processing import (
    discover_episode_dirs,
    iter_canonical_blocks,
    validate_export_episode,
)
from dexmani_real.dataset.quality import EpisodeQuality
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import ROW_INFO_SPECS
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalExportConfig:
    chunk_frames: int = 100
    compression_level: int = 3

    def __post_init__(self):
        if (
            isinstance(self.chunk_frames, bool)
            or not isinstance(self.chunk_frames, int)
            or self.chunk_frames <= 0
        ):
            raise ValueError("chunk_frames must be a positive integer")
        if (
            isinstance(self.compression_level, bool)
            or not isinstance(self.compression_level, int)
            or not 0 <= self.compression_level <= 9
        ):
            raise ValueError("compression_level must be an integer in [0, 9]")


def export_raw_to_zarr(
    input_root: str | Path,
    output_path: str | Path,
    config: CanonicalExportConfig | None = None,
    *,
    processing: ProcessingConfig,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    """Reject incomplete Raw episodes before writing; publish accepted episodes in full.

    Only known Raw defects found during candidate preflight can be skipped.
    Any conversion or append failure aborts the export and retains staging.
    Raw evidence is never edited.
    """
    config = config or CanonicalExportConfig()
    if not isinstance(processing, ProcessingConfig):
        raise TypeError("processing must be an explicit ProcessingConfig")
    source, target = Path(input_root).resolve(), Path(output_path).expanduser().resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("output and Raw input must not contain one another")
    output = Path(output_path).expanduser()
    if output.is_symlink():
        raise ValueError("canonical output must not be a symlink")
    if target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {target}")
    if any(p.suffix == ".zarr" and p.exists() for p in target.parents):
        raise ValueError("output must not be inside an existing Zarr store")
    accepted, rejected, excluded = _select_episodes(source, exclude)
    staging = root = data = None
    first_attrs = first_tails = None
    ends, offset = [], 0
    episode_notes, quality_summary = {}, {}
    revision = uuid.uuid4().hex
    created_utc = datetime.now(timezone.utc).isoformat()
    converted = []
    try:
        for index, episode in enumerate(accepted):
            if progress_callback:
                progress_callback(index, len(accepted))
            with EpisodeReader(episode) as reader:
                frames = reader.num_frames
                task = validate_task_identity(reader.meta["task_label"])
                specs = canonical_array_specs(
                    frames,
                    processing.pointcloud.num_points,
                    int(reader.meta["camera_color_height"]),
                    int(reader.meta["camera_color_width"]),
                )
                tails = {key: (shape[1:], dtype) for key, (shape, dtype) in specs.items()}
                attrs = dict(
                    format=CANONICAL_FORMAT,
                    task_name=task,
                    dt=reader.dt,
                    depth_scale_m_per_unit=float(reader.meta["depth_scale"]),
                    pointcloud_config=processing.pointcloud.to_dict(),
                    fingertip_link_names=list(processing.fingertip_link_names),
                )
                # Task is store organization; dt is a training/deployment numerical fact.
                # Depth scale describes this cache's stored depth, not the Policy ABI.
                if first_attrs is not None:
                    if task != first_attrs["task_name"] or not np.isclose(
                        reader.dt, first_attrs["dt"], rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            f"{episode.name}: one canonical store requires uniform task and dt"
                        )
                    if attrs["depth_scale_m_per_unit"] != first_attrs["depth_scale_m_per_unit"]:
                        raise ValueError(f"{episode.name}: depth_scale_m_per_unit must be uniform")
                    for key in specs:
                        if tails[key] != first_tails[key]:
                            raise ValueError(f"{episode.name}: incompatible {key} shape or dtype")
                row_bytes = sum(
                    int(np.prod(tail)) * dtype.itemsize for tail, dtype in tails.values()
                )
                chunk = min(config.chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                if root is None:
                    first_attrs, first_tails = attrs, tails
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(
                        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                    )
                    root, data, row_info = _create_store(
                        staging, attrs, revision, tails, chunk, config
                    )
                count, notes, quality = _append_episode(
                    reader, processing, data, row_info, tails, offset, frames, chunk, specs
                )
                episode_notes[episode.name] = notes
                offset += count
                ends.append(offset)
                episode_id = (
                    episode.name if episode == source else episode.relative_to(source).as_posix()
                )
                converted.append(
                    dict(episode_id=episode_id, source=str(episode), rows=count, end=offset)
                )
                quality_summary[episode_id] = quality.report(
                    notes,
                    count,
                    [key for key, (_, dtype) in tails.items() if np.issubdtype(dtype, np.floating)],
                )
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = [ep["episode_id"] for ep in converted]
        report = dict(
            export_config=asdict(config),
            data_revision=revision,
            created_utc=created_utc,
            episodes=converted,
            episode_ids=[ep["episode_id"] for ep in converted],
            source_code=_source_code(),
            quality_summary=quality_summary,
            input_root=str(source),
            output_path=str(target),
            task_name=first_attrs["task_name"],
            dt=first_attrs["dt"],
            episode_count=len(ends),
            total_frames=offset,
            episode_ends=ends,
            dataset_keys=sorted(first_tails),
            accepted_episodes=[episode.name for episode in accepted],
            rejected_episodes=rejected,
            excluded_episodes=sorted(excluded),
            processing=processing.to_dict(),
            episode_notes=episode_notes,
        )
        (staging / "export_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        atomic_publish(staging, target)
        staging = None
        if progress_callback:
            progress_callback(len(accepted), len(accepted))
        return report
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise ValueError(f"{episode.name}: canonical export failed: {exc}") from exc
    finally:
        if staging is not None:
            logger.warning("Export staging retained for inspection: %s", staging)


def _source_code():
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True))
        return dict(sha=sha, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        return dict(sha="unknown", dirty=None)


def _select_episodes(source, exclude):
    episodes = discover_episode_dirs(source)
    if isinstance(exclude, str) or any(not isinstance(name, str) for name in exclude):
        raise ValueError("exclude must be episode names")
    excluded = set(exclude)
    unknown = excluded - {ep.name for ep in episodes}
    if unknown:
        raise ValueError(f"exclusions reference unknown episodes: {sorted(unknown)}")
    selected = [ep for ep in episodes if ep.name not in excluded]
    if not selected:
        raise ValueError("export has no selected episodes")
    accepted, rejected = [], {}
    for episode in selected:
        try:
            for name in ("data.h5", "rgb.mp4"):
                try:
                    (episode / name).stat()
                except FileNotFoundError as exc:
                    raise RawDataError(f"missing required file: {name}") from exc
            with EpisodeReader(episode) as reader:
                validate_export_episode(reader)
        except RawDataError as exc:
            rejected[episode.name] = str(exc)
            logger.warning("Rejected %s: %s", episode.name, exc)
        else:
            accepted.append(episode)
    if not accepted:
        raise ValueError(f"all {len(selected)} selected episodes were rejected; no Zarr written")
    return accepted, rejected, excluded


def _create_store(staging, attrs, revision, tails, chunk, config):
    root = zarr.open_group(str(staging), mode="w")
    root.attrs.update(dict(attrs, data_revision=revision))
    data = root.create_group("data")
    row_info = root.create_group("row_info")
    for name, spec in ROW_INFO_SPECS.items():
        row_info.create_dataset(
            name,
            shape=(0, *spec.tail_shape),
            chunks=(chunk, *spec.tail_shape),
            dtype=spec.dtype,
        )
    root.attrs["time_semantics"] = (
        "host monotonic ns; 0 unknown/not called; camera times preserved from Raw "
        "(see Raw camera_timestamp_source); robot read completion"
    )
    root.attrs["row_semantics"] = (
        "control observation and attempted target, not per-action query input"
    )
    compressor = zarr.get_codec({"id": "zstd", "level": config.compression_level})
    for key, (tail, dtype) in tails.items():
        data.create_dataset(
            key,
            shape=(0, *tail),
            chunks=(chunk, *tail),
            dtype=dtype,
            compressor=compressor,
        )
    return root, data, row_info


def _append_episode(reader, processing, data, row_info, tails, offset, frames, chunk, specs):
    for key, (tail, _) in tails.items():
        data[key].resize((offset + frames, *tail))
    for name, spec in ROW_INFO_SPECS.items():
        row_info[name].resize((offset + frames, *spec.tail_shape))
    notes = {}
    if "observation_timestamp_ns" not in reader.fields:
        notes["timestamps"] = "unknown; dt is nominal only"
    quality = EpisodeQuality(reader.dt, reader.meta.get("camera_timestamp_source", "unknown"))
    count = 0
    for block in iter_canonical_blocks(reader, processing, chunk_frames=chunk, notes=notes):
        rows = len(block["joint_state"])
        if set(block) != set(specs):
            raise ValueError("incomplete canonical modalities")
        for key, values in block.items():
            tail, dtype = tails[key]
            if values.shape != (rows, *tail) or values.dtype != dtype:
                raise ValueError(f"{key}: transformed shape/dtype mismatch")
            data[key][offset + count : offset + count + rows] = values
        info_block = {
            name: reader.read_row_info(name, count, count + rows) for name in ROW_INFO_SPECS
        }
        for name, values in info_block.items():
            row_info[name][offset + count : offset + count + rows] = values
        quality.update(info_block)
        count += rows
    if count != frames:
        raise ValueError("transformed row count differs from Raw")
    return count, notes, quality
