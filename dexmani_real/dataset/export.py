"""Fail-fast whole-episode export to an atomically published canonical cache."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from dexmani_real.dataset.contracts import (
    CANONICAL_FORMAT,
    ProcessingConfig,
    canonical_array_specs,
    canonical_modality_contracts,
    validate_task_name,
)
from dexmani_real.dataset.pointcloud import load_raw_episode_camera_model
from dexmani_real.dataset.processing import (
    discover_episode_dirs,
    iter_canonical_blocks,
    validate_episode,
)
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.utils.atomic_io import target_is_occupied

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalExportConfig:
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
    config: CanonicalExportConfig | None = None,
    *,
    processing: ProcessingConfig,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    """Publish every selected episode in full or fail without a partial output.

    Explicit exclusions are names, never edits to Raw evidence. Unexpected
    structural/semantic failures abort the owned staging instead of salvaging rows.
    """
    config = config or CanonicalExportConfig()
    if not isinstance(processing, ProcessingConfig):
        raise TypeError("processing must be an explicit ProcessingConfig")
    source, target = Path(input_root).resolve(), Path(output_path).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("output must be outside the raw input")
    if target_is_occupied(Path(output_path).expanduser()) or target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {target}")
    if any(p.suffix == ".zarr" and p.exists() for p in target.parents):
        raise ValueError("output must not be inside an existing Zarr store")
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
    staging = root = data = None
    first_attrs = first_tails = first_contracts = None
    ends, offset = [], 0
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            with EpisodeReader(episode) as reader:
                frames = validate_episode(reader)
                task = validate_task_name(reader.meta["task_label"])
                if config.expected_task_name is not None and task != config.expected_task_name:
                    raise ValueError(
                        f"{episode.name}: task_name={task!r}, expected {config.expected_task_name!r}"
                    )
                camera = load_raw_episode_camera_model(reader).geometry.color
                specs = canonical_array_specs(
                    frames, processing.pointcloud.num_points, camera.height, camera.width
                )
                tails = {key: (shape[1:], dtype) for key, (shape, dtype) in specs.items()}
                attrs = dict(format=CANONICAL_FORMAT, task_name=task, dt=reader.dt)
                contracts = canonical_modality_contracts(reader, processing)
                if first_attrs is not None:
                    if task != first_attrs["task_name"] or not np.isclose(
                        reader.dt, first_attrs["dt"], rtol=0, atol=1e-12
                    ):
                        raise ValueError(
                            f"{episode.name}: one canonical store requires uniform task and dt"
                        )
                    for key in specs:
                        if tails[key] != first_tails[key] or contracts[key] != first_contracts[key]:
                            raise ValueError(
                                f"{episode.name}: incompatible {key} shape or semantic/recipe contract; depth scale_m_per_unit must be uniform"
                            )
                row_bytes = sum(
                    int(np.prod(tail)) * dtype.itemsize for tail, dtype in tails.values()
                )
                chunk = min(config.chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                if root is None:
                    first_attrs, first_tails, first_contracts = attrs, tails, contracts
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(
                        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                    )
                    root = zarr.open_group(str(staging), mode="w")
                    root.attrs.update(attrs)
                    data = root.create_group("data")
                    compressor = zarr.get_codec({"id": "zstd", "level": config.compression_level})
                    for key, (tail, dtype) in tails.items():
                        array = data.create_dataset(
                            key,
                            shape=(0, *tail),
                            chunks=(chunk, *tail),
                            dtype=dtype,
                            compressor=compressor,
                        )
                        array.attrs.update(contracts[key])
                for key, (tail, _) in tails.items():
                    data[key].resize((offset + frames, *tail))
                count = 0
                for block in iter_canonical_blocks(reader, processing, chunk_frames=chunk):
                    rows = len(block["joint_state"])
                    if set(block) != set(specs):
                        raise ValueError("incomplete canonical modalities")
                    for key, values in block.items():
                        tail, dtype = tails[key]
                        if values.shape != (rows, *tail) or values.dtype != dtype:
                            raise ValueError(f"{key}: transformed shape/dtype mismatch")
                        data[key][offset + count : offset + count + rows] = values
                    count += rows
                if count != frames:
                    raise ValueError("transformed row count differs from Raw")
                offset += count
                ends.append(offset)
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        if target_is_occupied(target):
            raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {target}")
        staging.rename(target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            input_root=str(source),
            output_path=str(target),
            task_name=first_attrs["task_name"],
            dt=first_attrs["dt"],
            episode_count=len(ends),
            total_frames=offset,
            episode_ends=ends,
            dataset_keys=sorted(first_tails),
            excluded_episodes=sorted(excluded),
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise ValueError(f"{episode.name}: canonical export failed: {exc}") from exc
    finally:
        if staging is not None:
            try:
                shutil.rmtree(staging)
            except OSError:
                logger.warning("Could not remove export staging %s", staging, exc_info=True)
