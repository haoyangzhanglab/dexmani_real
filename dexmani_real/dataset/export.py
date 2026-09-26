"""One pass from raw episodes through numerical transforms to an atomic policy Zarr."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    ProcessingConfig,
    policy_array_specs,
    validate_task_name,
)
from dexmani_real.dataset.pointcloud import load_raw_episode_camera_model
from dexmani_real.dataset.processing import (
    discover_episode_dirs,
    iter_policy_blocks,
    load_annotations,
    policy_semantics,
    validate_annotation_task_name_override,
    validate_episode,
)
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.utils.atomic_io import target_is_occupied

logger = logging.getLogger(__name__)

POLICY_ZARR_SCHEMA_NAME = "dexmani-real-policy-zarr"
POLICY_ZARR_SCHEMA_VERSION = 15


class StaticSemanticsMismatch(ValueError):
    pass


@dataclass(frozen=True)
class PolicyZarrExportConfig:
    chunk_frames: int = 100
    compression_level: int = 3
    expected_task_name: str | None = None

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
        if self.expected_task_name is not None:
            validate_task_name(self.expected_task_name)


def export_raw_to_zarr(
    input_root: str | Path,
    output_path: str | Path,
    config: PolicyZarrExportConfig | None = None,
    *,
    processing: ProcessingConfig,
    annotations_path: str | Path | None = None,
    task_name: str | None = None,
    dry_run: bool = False,
    progress_callback=None,
) -> dict:
    """Include whole raw episodes, with whole-episode admission.

    Dry-run consumes the same transformations/validation, without creating output.
    Export computes each episode once; later rejection removes only owned staging.
    progress_callback(completed, total) counts processed source episodes, including
    rejected and explicitly excluded episodes.
    """
    config = config or PolicyZarrExportConfig()
    if not isinstance(processing, ProcessingConfig):
        raise TypeError("processing must be an explicit ProcessingConfig")
    source, target = (
        Path(input_root).resolve(),
        Path(output_path).expanduser().resolve(),
    )
    if target == source or source in target.parents:
        raise ValueError("output must be outside the raw input")
    if target_is_occupied(Path(output_path).expanduser()) or target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing policy Zarr: {target}")
    if any(p.suffix == ".zarr" and p.exists() for p in target.parents):
        raise ValueError("output must not be inside an existing Zarr store")
    episodes = discover_episode_dirs(source)
    annotations = load_annotations(annotations_path)
    unknown = set(annotations) - {ep.name for ep in episodes}
    if unknown:
        raise ValueError(f"annotations reference unknown episodes: {sorted(unknown)}")
    override = validate_annotation_task_name_override(annotations, task_name)
    ends, excluded, rejected = [], [], []
    single_episode = source in episodes
    first_attrs = first_specs = None
    staging = root = data = None
    offset = 0
    try:
        for index, episode in enumerate(episodes):
            if progress_callback:
                progress_callback(index, len(episodes))
            annotation = annotations.get(episode.name, EpisodeAnnotation())
            if not annotation.include:
                excluded.append(episode.name)
                continue  # Explicit exclusion requires no readable source files.
            try:
                with EpisodeReader(episode) as reader:
                    source_frames = validate_episode(reader)
                    task = validate_task_name(
                        override
                        or annotation.task_name
                        or reader.h5f["meta"].attrs.get("task_label", "")
                    )
                    if config.expected_task_name is not None and task != config.expected_task_name:
                        raise StaticSemanticsMismatch(
                            f"{episode.name}: task_name={task!r}, expected {config.expected_task_name!r}"
                        )
                    camera = load_raw_episode_camera_model(reader).geometry.color
                    specs = policy_array_specs(
                        source_frames,
                        processing.pointcloud.num_points,
                        camera.height,
                        camera.width,
                    )
                    tails = {key: (shape[1:], dtype) for key, (shape, dtype) in specs.items()}
                    attrs = dict(
                        schema_name=POLICY_ZARR_SCHEMA_NAME,
                        schema_version=POLICY_ZARR_SCHEMA_VERSION,
                        domain="real",
                        task_name=task,
                        dt=reader.dt,
                        **policy_semantics(reader, processing),
                    )
                    if first_attrs is None:
                        first_attrs, first_specs = attrs, tails
                    elif (
                        not np.isclose(attrs["dt"], first_attrs["dt"], rtol=0, atol=1e-12)
                        or {k: v for k, v in attrs.items() if k != "dt"}
                        != {k: v for k, v in first_attrs.items() if k != "dt"}
                        or tails != first_specs
                    ):
                        raise StaticSemanticsMismatch(
                            f"{episode.name}: one policy Zarr requires uniform task, dt, modality shapes and semantics"
                        )
                    # Bound image/cloud working memory even at native camera resolution.
                    row_bytes = sum(
                        int(np.prod(tail)) * dtype.itemsize for tail, dtype in tails.values()
                    )
                    chunk = min(config.chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                    if not dry_run and root is None:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        staging = Path(
                            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                        )
                        root = zarr.open_group(str(staging), mode="w")
                        root.attrs.update(first_attrs)
                        data = root.create_group("data")
                        compressor = zarr.get_codec(
                            {"id": "zstd", "level": config.compression_level}
                        )
                        for key, (tail, dtype) in tails.items():
                            data.create_dataset(
                                key,
                                shape=(0, *tail),
                                chunks=(chunk, *tail),
                                dtype=dtype,
                                compressor=compressor,
                            )
                    if data is not None:
                        for key, (tail, _) in tails.items():
                            data[key].resize((offset + source_frames, *tail))
                    count = 0
                    for block in iter_policy_blocks(reader, processing, chunk_frames=chunk):
                        rows = len(block["joint_state"])
                        if set(block) != set(specs):
                            raise ValueError("incomplete policy modalities")
                        for key, values in block.items():
                            tail, dtype = tails[key]
                            if values.shape != (rows, *tail) or values.dtype != dtype:
                                raise ValueError(f"{key}: transformed shape/dtype mismatch")
                            if data is not None:
                                data[key][offset + count : offset + count + rows] = values
                        count += rows
                    if count != source_frames:
                        raise ValueError("transformed row count differs from raw source")
                    offset += count
                    ends.append(offset)
            except StaticSemanticsMismatch:
                raise
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                reason = f"{episode.name}: {exc}"
                rejected.append({"episode": episode.name, "reason": str(exc)})
                logger.error("Rejected whole episode %s", reason)
                if single_episode:
                    raise ValueError(reason) from exc
                if data is not None:
                    for key, (tail, _) in first_specs.items():
                        data[key].resize((offset, *tail))
                if not ends:
                    # A failed first episode cannot establish the batch calibration.
                    if staging is not None:
                        shutil.rmtree(staging)
                    staging = root = data = first_attrs = first_specs = None
        if not ends:
            raise ValueError("export produced no included episodes")
        if root is not None:
            root.create_group("meta").create_dataset(
                "episode_ends", data=np.asarray(ends, dtype=np.int64)
            )
            if target_is_occupied(target):
                raise FileExistsError(f"refusing to overwrite existing policy Zarr: {target}")
            staging.rename(target)
            staging = None
        if progress_callback:
            progress_callback(len(episodes), len(episodes))
        return dict(
            input_root=str(source),
            output_path=str(target),
            dry_run=dry_run,
            task_name=first_attrs["task_name"],
            dt=first_attrs["dt"],
            episode_count=len(ends),
            total_frames=offset,
            episode_ends=ends,
            dataset_keys=sorted(first_specs),
            excluded_episodes=excluded,
            rejected_episodes=rejected,
        )
    finally:
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except OSError:
                logger.warning("Could not remove export staging %s", staging, exc_info=True)
