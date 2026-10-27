from __future__ import annotations

import logging
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import zarr

from dexmani_real.data.canonical.processing import (
    discover_episode_dirs,
    iter_canonical_blocks,
    validate_export_episode,
)
from dexmani_real.data.canonical.schema import CANONICAL_FORMAT, ExportConfig, canonical_array_specs
from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.geometry.kinematics import ArmFK, HandFK
from dexmani_real.geometry.pose import quat_wxyz_to_rotmat
from dexmani_real.perception.hand_cloud import HandPointCloud
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.utils.atomic_io import atomic_publish, target_is_occupied
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)
PROCESSING_BYTES = 64 * 1024 * 1024
MAX_BATCH_FRAMES = 100
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


def export_canonical(
    input_root: str | Path,
    output_path: str | Path,
    *,
    processing: ExportConfig,
    exclude: tuple[str, ...] = (),
    progress_callback=None,
) -> dict:
    source = Path(input_root).expanduser().resolve()
    output = Path(output_path).expanduser()
    target = output.resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("output and Raw input must not contain one another")
    if target_is_occupied(output):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {output}")
    selected, excluded = select_episodes(source, exclude)
    rotation = quat_wxyz_to_rotmat(processing.handbase_quat_eef_wxyz)
    staging = None
    ends: list[int] = []
    episode_ids: list[str] = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            with EpisodeReader(episode) as reader:
                camera = validate_export_episode(reader)
                task = reader.meta["task_label"]
                if not task.strip() or any(ord(char) < 32 or ord(char) == 127 for char in task):
                    raise ValueError("task_label must be nonempty without control characters")
                height, width = camera.color.height, camera.color.width
                if index == 0:
                    specs = canonical_array_specs(
                        processing.pointcloud.num_points,
                        height,
                        width,
                        processing.hand_pointcloud.num_points,
                    )
                    row_bytes = sum(
                        int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values()
                    )
                    chunk = min(MAX_BATCH_FRAMES, max(1, PROCESSING_BYTES // row_bytes))
                    arm_fk, hand_fk = ArmFK(), HandFK()
                    hand_cloud = HandPointCloud(hand_fk, processing.hand_pointcloud.num_points)
                    period_ns, tolerance_ns = sampling_clock_ns(reader.control_hz)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(
                        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                    )
                    root = zarr.open_group(str(staging), mode="w")
                    root.attrs.update(
                        format=CANONICAL_FORMAT,
                        task_name=task,
                        dt=reader.dt,
                        action_semantics="absolute_joint_goal_not_next_step_arrival",
                        pointcloud_config=asdict(processing.pointcloud),
                        hand_pointcloud_config=asdict(processing.hand_pointcloud),
                        fingertip_link_names=list(XHAND_FINGERTIP_LINK_NAMES),
                        handbase_position_eef_m=list(processing.handbase_position_eef_m),
                        handbase_quat_eef_wxyz=list(processing.handbase_quat_eef_wxyz),
                        table_plane_abcd=processing.table_plane_abcd,
                        depth_scale_m_per_unit=camera.depth_scale_m,
                        timing_admission=dict(
                            rule="adjacent_interval_and_cumulative_grid",
                            period_ns=period_ns,
                            tolerance_ns=tolerance_ns,
                        ),
                    )
                    data = root.create_group("data")
                    arrays = {}
                    for name, (shape, dtype) in specs.items():
                        row_bytes = int(np.prod(shape)) * dtype.itemsize
                        storage_rows = max(
                            1, min(STORAGE_CHUNK_MAX_ROWS, STORAGE_CHUNK_BYTES // row_bytes)
                        )
                        arrays[name] = data.create_dataset(
                            name,
                            shape=(0, *shape),
                            chunks=(storage_rows, *shape),
                            dtype=dtype,
                            compressor=zarr.get_codec({"id": "zstd", "level": 3}),
                        )
                else:
                    if task != root.attrs["task_name"] or not np.isclose(
                        reader.dt, root.attrs["dt"], rtol=0, atol=1e-12
                    ):
                        raise ValueError("one canonical store requires uniform task and dt")
                    if camera.depth_scale_m != root.attrs["depth_scale_m_per_unit"]:
                        raise ValueError("depth_scale_m_per_unit must be uniform")
                    if (height, width, 3) != arrays["rgb"].shape[1:]:
                        raise ValueError("inconsistent RGB shape")

                blocks = iter_canonical_blocks(
                    reader,
                    processing,
                    arm_fk=arm_fk,
                    hand_fk=hand_fk,
                    hand_cloud=hand_cloud,
                    chunk_frames=chunk,
                    camera=camera,
                    handbase_rotation_eef=rotation,
                )
                append_episode(arrays, offset, blocks, reader.num_frames)
                ends.append(offset + reader.num_frames)
            episode_ids.append(
                episode.name if episode == source else episode.relative_to(source).as_posix()
            )
        root.create_group("meta").create_dataset(
            "episode_ends", data=np.asarray(ends, dtype=np.int64)
        )
        root.attrs["episode_ids"] = episode_ids
        atomic_publish(cast(Path, staging), target)
        staging = None
        if progress_callback:
            progress_callback(len(selected), len(selected))
        return dict(
            episode_count=len(ends), total_frames=ends[-1], excluded_episodes=sorted(excluded)
        )
    except Exception:
        logger.exception("Export failed at %s", episode)
        raise
    finally:
        if staging is not None:
            logger.warning("Export staging retained: %s", staging)


def append_episode(arrays, offset, blocks, frames):
    for array in arrays.values():
        array.resize((offset + frames, *array.shape[1:]))
    count = 0
    for block in blocks:
        rows = len(block["joint_state"])
        for name, values in block.items():
            if np.issubdtype(values.dtype, np.floating):
                valid = np.isfinite(values).reshape(rows, -1).all(axis=1)
                if not valid.all():
                    row = count + int(np.flatnonzero(~valid)[0])
                    raise ValueError(f"canonical {name}: nonfinite values at row {row}")
            arrays[name][offset + count : offset + count + rows] = values
        count += rows


def select_episodes(source, exclude):
    episodes = discover_episode_dirs(source)
    excluded = set(exclude)
    unknown = excluded - {episode.name for episode in episodes}
    if unknown:
        raise ValueError(f"exclusions reference unknown episodes: {sorted(unknown)}")
    selected = [episode for episode in episodes if episode.name not in excluded]
    if not selected:
        raise ValueError("export has no selected episodes")
    return selected, excluded
