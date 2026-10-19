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
    CanonicalEpisodeProcessor,
    validate_export_episode,
)
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


class CanonicalExporter:
    """Own one canonical export, its metadata baseline, staging and shared FK."""

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
        self.source = Path(input_root).expanduser().resolve()
        output = Path(output_path).expanduser()
        self.target = output.resolve()
        if (
            self.target == self.source
            or self.source in self.target.parents
            or self.target in self.source.parents
        ):
            raise ValueError("output and Raw input must not contain one another")
        if output.is_symlink():
            raise ValueError("canonical output must not be a symlink")
        if target_is_occupied(self.target):
            raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {self.target}")
        if any(p.suffix == ".zarr" and p.exists() for p in self.target.parents):
            raise ValueError("output must not be inside an existing Zarr store")
        repo_root = Path(__file__).resolve().parents[2]
        for name in ("episodes", "rollouts"):
            protected = (repo_root / name).resolve()
            if self.target == protected or protected in self.target.parents:
                raise ValueError(f"output must not be inside Raw directory: {protected}")
        selected, excluded = select_episodes(self.source, exclude)
        rejected = {}
        self.staging = self.root = self.data = None
        self.first_attrs = self.first_tails = None
        self.ends, self.offset = [], 0
        self.revision = uuid.uuid4().hex
        self.episode_ids = []
        self.episode_arm_execution = []
        self.hand_fk = None
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
                        try:
                            task = validate_task_identity(reader.meta["task_label"])
                        except (TypeError, ValueError) as exc:
                            raise RawDataError(f"invalid task metadata: {exc}") from exc
                    except RawDataError as exc:
                        rejected[episode.name] = str(exc)
                        logger.warning("Rejected %s: %s", episode.name, exc)
                        continue
                    frames = reader.num_frames
                    incompatibility = None
                    tails = canonical_array_specs(
                        self.processing.pointcloud.num_points,
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
                        pointcloud_config=asdict(self.processing.pointcloud),
                        fingertip_link_names=list(self.processing.fingertip_link_names),
                    )
                    # Task/depth scale describe this cache; dt defines training/deployment timing.
                    if self.first_attrs is not None:
                        if task != self.first_attrs["task_name"] or not np.isclose(
                            reader.dt, self.first_attrs["dt"], rtol=0, atol=1e-12
                        ):
                            incompatibility = (
                                f"{episode.name}: one canonical store requires uniform task and dt"
                            )
                        elif (
                            attrs["depth_scale_m_per_unit"]
                            != self.first_attrs["depth_scale_m_per_unit"]
                        ):
                            incompatibility = (
                                f"{episode.name}: depth_scale_m_per_unit must be uniform"
                            )
                        else:
                            for key in tails:
                                if tails[key] != self.first_tails[key]:
                                    incompatibility = (
                                        f"{episode.name}: incompatible {key} shape or dtype"
                                    )
                                    break
                    # Incompatible valid metadata fails before consuming an independently invalid payload.
                    if incompatibility is not None:
                        raise ValueError(incompatibility)
                    row_bytes = sum(
                        int(np.prod(tail)) * dtype.itemsize for tail, dtype in tails.values()
                    )
                    chunk = min(self.chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                    try:
                        if self.hand_fk is None:
                            self.hand_fk = HandKinematics(
                                str(XHAND_RIGHT_URDF_PATH),
                                list(self.processing.fingertip_link_names),
                            )
                        blocks = CanonicalEpisodeProcessor(
                            self.processing,
                            self.hand_fk,
                            chunk_frames=chunk,
                        ).iter_blocks(reader, camera=camera, base_from_color=base_from_color)
                        if self.root is None:
                            if self.staging is None:
                                self.target.parent.mkdir(parents=True, exist_ok=True)
                                self.staging = Path(
                                    tempfile.mkdtemp(
                                        prefix=f".{self.target.name}.tmp-", dir=self.target.parent
                                    )
                                )
                            self.root, self.data = self.create_store(attrs, tails, chunk)
                        count = self.consume_episode(blocks, tails, frames)
                    except RawDataError as exc:
                        for key, (tail, _) in tails.items():
                            self.data[key].resize((self.offset, *tail))
                        rejected[episode.name] = str(exc)
                        logger.warning("Rejected %s: %s", episode.name, exc)
                        if self.offset == 0:
                            self.root = self.data = self.first_attrs = self.first_tails = None
                        continue
                    if self.first_attrs is None:
                        self.first_attrs, self.first_tails = attrs, tails
                    self.offset += count
                    self.ends.append(self.offset)
                    self.episode_ids.append(
                        episode.name
                        if episode == self.source
                        else episode.relative_to(self.source).as_posix()
                    )
                    self.episode_arm_execution.append(reader.meta["arm_execution_format"])
            if not self.ends:
                raise ValueError("all selected episodes were rejected; no Zarr written")
            self.root.create_group("meta").create_dataset(
                "episode_ends", data=np.asarray(self.ends, dtype=np.int64)
            )
            self.root.attrs["episode_ids"] = self.episode_ids
            self.root.attrs["episode_arm_execution"] = self.episode_arm_execution
            atomic_publish(self.staging, self.target)
            self.staging = None
            if progress_callback:
                progress_callback(len(selected), len(selected))
            return dict(
                episode_count=len(self.ends),
                total_frames=self.offset,
                rejected_episodes=rejected,
                excluded_episodes=sorted(excluded),
            )
        except Exception:
            logger.exception("%s: canonical export failed", episode.name)
            raise
        finally:
            if self.staging is not None:
                logger.warning("Export staging retained for inspection: %s", self.staging)

    def create_store(self, attrs, tails, chunk):
        root = zarr.open_group(str(self.staging), mode="w")
        root.attrs.update(dict(attrs, data_revision=self.revision))
        data = root.create_group("data")
        compressor = zarr.get_codec({"id": "zstd", "level": self.compression_level})
        for key, (tail, dtype) in tails.items():
            data.create_dataset(
                key,
                shape=(0, *tail),
                chunks=(chunk, *tail),
                dtype=dtype,
                compressor=compressor,
            )
        return root, data

    def consume_episode(self, blocks, tails, frames):
        """Check every converted row and append it to the prepared destination."""
        for key, (tail, _) in tails.items():
            self.data[key].resize((self.offset + frames, *tail))
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
                self.data[key][self.offset + count : self.offset + count + rows] = values
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
