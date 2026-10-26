"""Full-modal conversion into a rebuildable, one-task canonical cache."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zarr

from dexmani_real.data.canonical.processing import (
    discover_episode_dirs,
    iter_canonical_blocks,
    validate_export_episode,
)
from dexmani_real.data.canonical.schema import (
    CANONICAL_FORMAT,
    ProcessingConfig,
    canonical_array_specs,
)
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError, text_attr
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.geometry.pose import quat_wxyz_to_rotmat
from dexmani_real.perception.hand import HandSurface
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)

# Storage blocks are independent of the multi-modal processing batch budget.
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


def export_canonical(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ProcessingConfig,
    chunk_frames: int = 100,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    """Export all 13 fields; reject bad Raw and retain staging on conversion or write failures."""
    handbase_rotation_eef = quat_wxyz_to_rotmat(processing.handbase_quat_eef_wxyz)
    if type(chunk_frames) is not int or chunk_frames <= 0:
        raise ValueError("chunk_frames must be a positive integer")
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
    repo_root = Path(__file__).resolve().parents[3]
    for name in ("episodes", "rollouts"):
        protected = (repo_root / name).resolve()
        if target == protected or protected in target.parents:
            raise ValueError(f"output must not be inside Raw directory: {protected}")
    selected, excluded = select_episodes(source, exclude)
    rejected = {}
    staging = None
    root: zarr.Group
    data: zarr.Group
    ends: list[int] = []
    episode_ids: list[str] = []
    arm_fk: ArmFK
    hand_fk: HandFK
    hand_cloud = None
    processing_attrs = dict(
        format=CANONICAL_FORMAT,
        action_semantics="absolute_joint_goal_not_next_step_arrival",
        pointcloud_config=asdict(processing.pointcloud),
        fingertip_link_names=list(processing.fingertip_link_names),
        handbase_position_eef_m=list(processing.handbase_position_eef_m),
        handbase_quat_eef_wxyz=list(processing.handbase_quat_eef_wxyz),
        table_plane_abcd=processing.table_plane_abcd,
    )
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            try:
                for name in ("data.h5", "rgb.mp4"):
                    try:
                        (episode / name).stat()
                    except FileNotFoundError as exc:
                        raise RawDataError(f"missing required file: {name}") from exc
                with EpisodeReader(episode) as reader:
                    camera, base_from_color, camera_parameters = validate_export_episode(reader)
                    task = text_attr(reader.meta, "task_label")
                    if any(ord(char) < 32 or ord(char) == 127 for char in task):
                        raise RawDataError(
                            "invalid task metadata: task_name must not contain control characters"
                        )
                    height, width = camera.intrinsics.height, camera.intrinsics.width
                    specs = canonical_array_specs(
                        processing.pointcloud.num_points, height, width,
                        processing.hand_pointcloud.num_points,
                    )
                    row_bytes = sum(
                        int(np.prod(tail)) * dtype.itemsize for tail, dtype in specs.values()
                    )
                    chunk = min(chunk_frames, max(1, (64 * 1024 * 1024) // row_bytes))
                    if hand_cloud is None:
                        arm_fk = ArmFK()
                        hand_fk = HandFK(processing.fingertip_link_names)
                        hand_cloud = HandSurface(processing.hand_pointcloud, hand_fk)
                    if not ends:
                        period_ns, tolerance_ns = sampling_clock_ns(reader.control_hz)
                        attrs = dict(
                            processing_attrs,
                            task_name=task,
                            dt=reader.dt,
                            timing_admission=dict(
                                rule="adjacent_interval_and_cumulative_grid",
                                period_ns=period_ns,
                                tolerance_ns=tolerance_ns,
                            ),
                            depth_scale_m_per_unit=camera.depth_scale_m,
                            hand_pointcloud_config=asdict(processing.hand_pointcloud),
                        )
                        if staging is None:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            staging = Path(
                                tempfile.mkdtemp(
                                    prefix=f".{target.name}.tmp-", dir=target.parent
                                )
                            )
                        root = zarr.open_group(str(staging), mode="w")
                        root.attrs.update(attrs)
                        data = root.create_group("data")
                        compressor = zarr.get_codec({"id": "zstd", "level": 3})
                        for key, (tail, dtype) in specs.items():
                            storage_row_bytes = int(np.prod(tail)) * dtype.itemsize
                            # 单帧是最小存储块，即使它超过目标字节数。
                            storage_chunk = max(
                                1, min(STORAGE_CHUNK_MAX_ROWS, STORAGE_CHUNK_BYTES // storage_row_bytes)
                            )
                            data.create_dataset(
                                key,
                                shape=(0, *tail),
                                chunks=(storage_chunk, *tail),
                                dtype=dtype,
                                compressor=compressor,
                            )
                    else:
                        # Only these values vary by episode; the processing recipe is fixed.
                        if task != root.attrs["task_name"] or not np.isclose(
                            reader.dt, root.attrs["dt"], rtol=0, atol=1e-12
                        ):
                            raise ValueError(
                                f"{episode.name}: one canonical store requires uniform task and dt"
                            )
                        if camera.depth_scale_m != root.attrs["depth_scale_m_per_unit"]:
                            raise ValueError(
                                f"{episode.name}: depth_scale_m_per_unit must be uniform"
                            )
                        if (height, width, 3) != data["rgb"].shape[1:]:
                            raise ValueError(f"{episode.name}: inconsistent rgb shape")
                    blocks = iter_canonical_blocks(
                        reader,
                        processing,
                        arm_fk=arm_fk,
                        hand_fk=hand_fk,
                        hand_cloud=hand_cloud,
                        chunk_frames=chunk,
                        camera=camera,
                        base_from_color=base_from_color,
                        camera_parameters=camera_parameters,
                        handbase_rotation_eef=handbase_rotation_eef,
                    )
                    count = append_episode(data, offset, blocks, specs, reader.num_frames)
            except RawDataError as exc:
                rejected[episode.name] = str(exc)
                logger.warning("Rejected %s: %s", episode.name, exc)
                continue
            ends.append(offset + count)
            episode_ids.append(
                episode.name if episode == source else episode.relative_to(source).as_posix()
            )
        if not ends or staging is None:
            raise ValueError("all selected episodes were rejected; no Zarr written")
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = episode_ids
        atomic_publish(staging, target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            episode_count=len(ends),
            total_frames=ends[-1],
            rejected_episodes=rejected,
            excluded_episodes=sorted(excluded),
        )
    finally:
        if staging is not None:
            logger.warning("Export staging retained for inspection: %s", staging)


def append_episode(data, offset, blocks, specs, frames):
    """Append a complete episode; invalid Raw rolls back only this episode's rows."""
    for key, (tail, _) in specs.items():
        data[key].resize((offset + frames, *tail))
    count = 0
    try:
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
    except RawDataError:
        for key, (tail, _) in specs.items():
            data[key].resize((offset, *tail))
        raise
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
