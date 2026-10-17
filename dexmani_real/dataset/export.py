"""Full-modal conversion into a rebuildable, one-task canonical cache."""

from __future__ import annotations

import logging
import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import h5py
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
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.control_clock import sampling_clock_ns

logger = logging.getLogger(__name__)


def export_raw_to_zarr(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ProcessingConfig,
    chunk_frames: int = 100,
    compression_level: int = 3,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    """Export all 15 modalities; reject incomplete episodes without changing Raw.
    Unexpected conversion/write failures abort publication and retain staging.
    """
    if type(chunk_frames) is not int or chunk_frames <= 0:
        raise ValueError("chunk_frames must be a positive integer")
    if type(compression_level) is not int or not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer in [0, 9]")
    source = Path(input_root).expanduser().resolve()
    output = Path(output_path).expanduser()
    target = output.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("output and Raw input must not contain one another")
    if output.is_symlink():
        raise ValueError("canonical output must not be a symlink")
    if target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {target}")
    if any(p.suffix == ".zarr" and p.exists() for p in target.parents):
        raise ValueError("output must not be inside an existing Zarr store")
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("episodes", "rollouts"):
        protected = (repo_root / name).resolve()
        if target == protected or protected in target.parents:
            raise ValueError(f"output must not be inside Raw directory: {protected}")
    selected, excluded = _select_episodes(source, exclude)
    rejected = {}
    staging = root = data = None
    first_attrs = first_tails = None
    ends, offset = [], 0
    revision = uuid.uuid4().hex
    episode_ids = []
    episode_arm_execution = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            with ExitStack() as resources:
                try:
                    for name in ("data.h5", "rgb.mp4"):
                        try:
                            (episode / name).stat()
                        except FileNotFoundError as exc:
                            raise RawDataError(f"missing required file: {name}") from exc
                    if not h5py.is_hdf5(episode / "data.h5"):
                        raise RawDataError("data.h5 is not an HDF5 file")
                    reader = resources.enter_context(EpisodeReader(episode))
                    camera, base_from_color = validate_export_episode(reader)
                except ValueError as exc:
                    rejected[episode.name] = str(exc)
                    logger.warning("Rejected %s: %s", episode.name, exc)
                    continue
                frames = reader.num_frames
                task = reader.meta["task_label"]
                incompatibility = None
                try:
                    task = validate_task_identity(task)
                except (TypeError, ValueError) as exc:
                    incompatibility = f"{episode.name}: {exc}"
                tails = canonical_array_specs(
                    processing.pointcloud.num_points,
                    camera.geometry.color.height,
                    camera.geometry.color.width,
                )
                period_ns, tolerance_ns = sampling_clock_ns(reader.control_hz)
                attrs = dict(
                    format=CANONICAL_FORMAT,
                    task_name=task,
                    dt=reader.dt,
                    timing_admission=dict(
                        rule="adjacent_interval_and_cumulative_grid",
                        period_ns=period_ns,
                        tolerance_ns=tolerance_ns,
                    ),
                    action_semantics="absolute_joint_goal_not_next_step_arrival",
                    depth_scale_m_per_unit=float(reader.meta["depth_scale"]),
                    pointcloud_config=asdict(processing.pointcloud),
                    fingertip_link_names=list(processing.fingertip_link_names),
                )
                # Task/depth scale describe this cache; dt defines training/deployment timing.
                if first_attrs is not None and incompatibility is None:
                    if task != first_attrs["task_name"] or not np.isclose(
                        reader.dt, first_attrs["dt"], rtol=0, atol=1e-12
                    ):
                        incompatibility = (
                            f"{episode.name}: one canonical store requires uniform task and dt"
                        )
                    elif attrs["depth_scale_m_per_unit"] != first_attrs["depth_scale_m_per_unit"]:
                        incompatibility = f"{episode.name}: depth_scale_m_per_unit must be uniform"
                    else:
                        for key in tails:
                            if tails[key] != first_tails[key]:
                                incompatibility = f"{episode.name}: incompatible {key} shape or dtype"
                                break
                row_bytes = sum(
                    int(np.prod(tail)) * dtype.itemsize for tail, dtype in tails.values()
                )
                chunk = min(chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                try:
                    blocks = iter_canonical_blocks(
                        reader, processing, camera=camera, base_from_color=base_from_color,
                        chunk_frames=chunk
                    )
                    if incompatibility is not None:
                        # Invalid evidence is rejected even when its metadata differs from the cache.
                        _consume_episode(blocks, None, tails, offset, frames)
                        raise ValueError(incompatibility)
                    if root is None:
                        first_attrs, first_tails = attrs, tails
                        if staging is None:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            staging = Path(
                                tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                            )
                        root, data = _create_store(
                            staging, attrs, revision, tails, chunk, compression_level
                        )
                    count = _consume_episode(blocks, data, tails, offset, frames)
                except RawDataError as exc:
                    if incompatibility is None:
                        for key, (tail, _) in tails.items():
                            data[key].resize((offset, *tail))
                    rejected[episode.name] = str(exc)
                    logger.warning("Rejected %s: %s", episode.name, exc)
                    if offset == 0:
                        root = data = first_attrs = first_tails = None
                    continue
                offset += count
                ends.append(offset)
                episode_ids.append(
                    episode.name if episode == source else episode.relative_to(source).as_posix()
                )
                episode_arm_execution.append(reader.meta["arm_execution_format"])
        if not ends:
            raise ValueError("all selected episodes were rejected; no Zarr written")
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = episode_ids
        root.attrs["episode_arm_execution"] = episode_arm_execution
        atomic_publish(staging, target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            episode_count=len(ends),
            total_frames=offset,
            rejected_episodes=rejected,
            excluded_episodes=sorted(excluded),
        )
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        raise ValueError(f"{episode.name}: canonical export failed: {exc}") from exc
    finally:
        if staging is not None:
            logger.warning("Export staging retained for inspection: %s", staging)


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
    return selected, excluded


def _create_store(staging, attrs, revision, tails, chunk, compression_level):
    root = zarr.open_group(str(staging), mode="w")
    root.attrs.update(dict(attrs, data_revision=revision))
    data = root.create_group("data")
    compressor = zarr.get_codec({"id": "zstd", "level": compression_level})
    for key, (tail, dtype) in tails.items():
        data.create_dataset(
            key,
            shape=(0, *tail),
            chunks=(chunk, *tail),
            dtype=dtype,
            compressor=compressor,
        )
    return root, data


def _consume_episode(blocks, data, tails, offset, frames):
    """Check every converted row; append only when metadata fits the destination."""
    if data is not None:
        for key, (tail, _) in tails.items():
            data[key].resize((offset + frames, *tail))
    count = 0
    for block in blocks:
        rows = len(block["joint_state"])
        if block.keys() != tails.keys():
            raise ValueError("incomplete canonical modalities")
        for key, values in block.items():
            tail, dtype = tails[key]
            if values.shape != (rows, *tail) or values.dtype != dtype:
                raise ValueError(f"{key}: transformed shape/dtype mismatch")
            if np.issubdtype(dtype, np.floating):
                valid = np.isfinite(values).reshape(rows, -1).all(axis=1)
                if not valid.all():
                    row = count + int(np.flatnonzero(~valid)[0])
                    raise RawDataError(f"canonical {key}: nonfinite values at row {row}")
            if data is not None:
                data[key][offset + count : offset + count + rows] = values
        count += rows
    if count != frames:
        raise ValueError("transformed row count differs from Raw")
    return count
