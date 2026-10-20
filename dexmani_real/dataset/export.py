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

from dexmani_real.dataset.schema import (
    CANONICAL_FORMAT,
    ProcessingConfig,
    canonical_array_specs,
)
from dexmani_real.dataset.processing import (
    discover_episode_dirs,
    CanonicalEpisodeProcessor,
    validate_export_episode,
)
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError, text_attr
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)

# Storage blocks are independent of the multi-modal processing batch budget.
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


class CanonicalExporter:
    """Export Raw episodes with one processing and storage configuration."""

    def __init__(
        self, processing: ProcessingConfig, *, chunk_frames: int = 100, compression_level: int = 3
    ):
        self.processing = processing
        self.chunk_frames = chunk_frames
        self.compression_level = compression_level

    def export(
        self,
        input_root: str | Path,
        output_path: str | Path,
        *,
        exclude: tuple[str, ...] = (),
        progress_callback=None,
    ) -> dict:
        """Export all 15 fields; reject bad Raw and retain staging on conversion or write failures."""
        if type(self.chunk_frames) is not int or self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be a positive integer")
        if type(self.compression_level) is not int or not 0 <= self.compression_level <= 9:
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
        selected, excluded = select_episodes(source, exclude)
        rejected = {}
        staging = root = data = None
        first_attrs = first_specs = None
        ends, offset = [], 0
        revision = uuid.uuid4().hex
        episode_ids = []
        episode_arm_execution = []
        hand_fk = None
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
                        task = text_attr(reader.meta, "task_label")
                        if any(ord(char) < 32 or ord(char) == 127 for char in task):
                            raise RawDataError(
                                "invalid task metadata: task_name must not contain control characters"
                            )
                    except RawDataError as exc:
                        rejected[episode.name] = str(exc)
                        logger.warning("Rejected %s: %s", episode.name, exc)
                        continue
                    frames = reader.num_frames
                    specs = canonical_array_specs(
                        self.processing.pointcloud.num_points,
                        camera.geometry.color.height,
                        camera.geometry.color.width,
                    )
                    period_ns, tolerance_ns = sampling_clock_ns(reader.control_hz)
                    attrs = dict(
                        format=CANONICAL_FORMAT,
                        data_revision=revision,
                        task_name=task,
                        dt=reader.dt,
                        timing_admission=dict(
                            rule="adjacent_interval_and_cumulative_grid",
                            period_ns=period_ns,
                            tolerance_ns=tolerance_ns,
                        ),
                        action_semantics="absolute_joint_goal_not_next_step_arrival",
                        depth_scale_m_per_unit=camera.depth_scale_m,
                        pointcloud_config=asdict(self.processing.pointcloud),
                        fingertip_link_names=list(self.processing.fingertip_link_names),
                    )
                    # Reject mixed cache semantics before consuming the episode payload.
                    if first_attrs is not None:
                        if task != first_attrs["task_name"] or not np.isclose(
                            reader.dt, first_attrs["dt"], rtol=0, atol=1e-12
                        ):
                            raise ValueError(
                                f"{episode.name}: one canonical store requires uniform task and dt"
                            )
                        if (
                            attrs["depth_scale_m_per_unit"]
                            != first_attrs["depth_scale_m_per_unit"]
                        ):
                            raise ValueError(
                                f"{episode.name}: depth_scale_m_per_unit must be uniform"
                            )
                        for key in specs:
                            if specs[key] != first_specs[key]:
                                raise ValueError(
                                    f"{episode.name}: inconsistent {key} shape or dtype"
                                )
                    row_bytes = sum(
                        int(np.prod(tail)) * dtype.itemsize for tail, dtype in specs.values()
                    )
                    chunk = min(self.chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                    try:
                        if hand_fk is None:
                            hand_fk = HandKinematics(
                                str(XHAND_RIGHT_URDF_PATH),
                                self.processing.fingertip_link_names,
                            )
                        blocks = CanonicalEpisodeProcessor(
                            self.processing,
                            hand_fk,
                            chunk_frames=chunk,
                        ).iter_blocks(reader, camera=camera, base_from_color=base_from_color)
                        if root is None:
                            if staging is None:
                                target.parent.mkdir(parents=True, exist_ok=True)
                                staging = Path(
                                    tempfile.mkdtemp(
                                        prefix=f".{target.name}.tmp-", dir=target.parent
                                    )
                                )
                            root, data = self.create_store(staging, attrs, specs)
                        count = append_episode(data, offset, blocks, specs, frames)
                    except RawDataError as exc:
                        for key, (tail, _) in specs.items():
                            data[key].resize((offset, *tail))
                        rejected[episode.name] = str(exc)
                        logger.warning("Rejected %s: %s", episode.name, exc)
                        if offset == 0:
                            root = data = first_attrs = first_specs = None
                        continue
                    if first_attrs is None:
                        first_attrs, first_specs = attrs, specs
                    offset += count
                    ends.append(offset)
                    episode_ids.append(
                        episode.name
                        if episode == source
                        else episode.relative_to(source).as_posix()
                    )
                    episode_arm_execution.append(text_attr(reader.meta, "arm_execution_format"))
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
        except Exception:
            logger.exception("%s: canonical export failed", episode.name)
            raise
        finally:
            if staging is not None:
                logger.warning("Export staging retained for inspection: %s", staging)

    def create_store(self, staging, attrs, specs):
        root = zarr.open_group(str(staging), mode="w")
        root.attrs.update(attrs)
        data = root.create_group("data")
        compressor = zarr.get_codec({"id": "zstd", "level": self.compression_level})
        for key, (tail, dtype) in specs.items():
            row_bytes = int(np.prod(tail)) * dtype.itemsize
            # A full frame is the minimum decode unit, even when larger than the target.
            chunk = max(1, min(STORAGE_CHUNK_MAX_ROWS, STORAGE_CHUNK_BYTES // row_bytes))
            data.create_dataset(
                key,
                shape=(0, *tail),
                chunks=(chunk, *tail),
                dtype=dtype,
                compressor=compressor,
            )
        return root, data


def append_episode(data, offset, blocks, specs, frames):
    """Check every converted row and append it to the prepared destination."""
    for key, (tail, _) in specs.items():
        data[key].resize((offset + frames, *tail))
    count = 0
    for block in blocks:
        rows = len(block["joint_state"])
        if block.keys() != specs.keys():
            raise ValueError("incomplete canonical modalities")
        for key, values in block.items():
            tail, dtype = specs[key]
            if values.shape != (rows, *tail) or values.dtype != dtype:
                raise ValueError(f"{key}: transformed shape/dtype mismatch")
            if np.issubdtype(dtype, np.floating):
                valid = np.isfinite(values).reshape(rows, -1).all(axis=1)
                if not valid.all():
                    row = count + int(np.flatnonzero(~valid)[0])
                    raise RawDataError(f"canonical {key}: nonfinite values at row {row}")
            data[key][offset + count : offset + count + rows] = values
        count += rows
    if count != frames:
        raise ValueError("transformed row count differs from Raw")
    return count


def select_episodes(source, exclude):
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
