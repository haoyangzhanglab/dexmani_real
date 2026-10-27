"""将 Raw 观测与绝对动作转换为训练用 Zarr。"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from dexmani_real.data.raw import DATASET_SPECS, RawReader
from dexmani_real.geometry.camera import (
    CameraSnapshot,
    camera_parameters_cv,
    validate_aligned_depth_distortion,
)
from dexmani_real.geometry.kinematics import ArmFK, HandFK, eef_poses
from dexmani_real.geometry.pose import (
    quat_wxyz_to_rotmat,
    validate_unit_quaternion_wxyz,
)
from dexmani_real.perception.config import HandCloudConfig, PointCloudConfig
from dexmani_real.perception.hand_cloud import HandPointCloud, hand_geometry
from dexmani_real.perception.pointcloud import PointCloudProcessor
from dexmani_real.robot.model import XHAND_FINGERTIP_LINK_NAMES
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.utils.io import publish
from dexmani_real.utils.timing import sampling_clock_ns

CANONICAL_FORMAT = "dexmani.real.canonical"
logger = logging.getLogger(__name__)
PROCESSING_BYTES = 64 * 1024 * 1024
MAX_BATCH_FRAMES = 100
STORAGE_CHUNK_BYTES = 1024 * 1024
STORAGE_CHUNK_MAX_ROWS = 128


@dataclass(frozen=True)
class ExportConfig:
    pointcloud: PointCloudConfig = field(default_factory=PointCloudConfig)
    hand_pointcloud: HandCloudConfig = field(default_factory=HandCloudConfig)
    table_plane_abcd: tuple[float, float, float, float] | None = None
    handbase_position_eef_m: tuple[float, ...] = HandConfig.T_eef_handbase_pos_xyz
    handbase_quat_eef_wxyz: tuple[float, ...] = HandConfig.T_eef_handbase_quat_wxyz

    @classmethod
    def from_runtime(cls, runtime, *, table_plane_abcd=None):
        return cls(
            pointcloud=runtime.pointcloud,
            hand_pointcloud=runtime.hand_pointcloud,
            table_plane_abcd=table_plane_abcd,
            handbase_position_eef_m=tuple(runtime.hand.T_eef_handbase_pos_xyz),
            handbase_quat_eef_wxyz=tuple(runtime.hand.T_eef_handbase_quat_wxyz),
        )

    def __post_init__(self) -> None:
        position = np.asarray(self.handbase_position_eef_m, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("hand base position must be a finite (3,) vector")
        validate_unit_quaternion_wxyz(self.handbase_quat_eef_wxyz, name="hand base quaternion")
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires an explicit table_plane_abcd")


def canonical_array_specs(
    num_points: int,
    rgb_height: int,
    rgb_width: int,
    hand_num_points: int,
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """公共导出的固定 13 字段及逐行形状、类型。"""
    return {
        "joint_state": ((19,), np.dtype(np.float32)),
        "action": ((19,), np.dtype(np.float32)),
        "action_ee": ((21,), np.dtype(np.float32)),
        "finger_tactile_force": ((5, 3), np.dtype(np.float32)),
        "taxel_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
        "fingertip_xyz": ((5, 3), np.dtype(np.float32)),
        "eef_pose": ((9,), np.dtype(np.float32)),
        "camera_intrinsic": ((9,), np.dtype(np.float32)),
        "camera_extrinsic": ((12,), np.dtype(np.float32)),
        "rgb": ((rgb_height, rgb_width, 3), np.dtype(np.uint8)),
        "depth": ((rgb_height, rgb_width), np.dtype(np.uint16)),
        "point_cloud": ((num_points, 6), np.dtype(np.float32)),
        "hand_pointcloud": ((hand_num_points, 6), np.dtype(np.float32)),
    }


def validate_export_episode(reader: RawReader) -> CameraSnapshot:
    frames = reader.num_frames
    if frames <= 0:
        raise ValueError("episode contains no rows")
    reader.require_fields(*DATASET_SPECS, "rgb", "depth")
    camera = reader.read_camera()
    validate_aligned_depth_distortion(camera.color.distortion_model)
    if camera.T_xarm_base_from_color is None:
        raise ValueError("canonical export requires camera extrinsics")
    color = camera.color
    expected = (frames, color.height, color.width)
    if reader["depth"].shape != expected or reader["depth"].dtype != np.uint16:
        raise ValueError(f"depth must be uint16 {expected}")
    reader.validate_completion()
    reader.validate_rows()
    if frames == 1:
        logging.getLogger(__name__).warning(
            "%s: single row, no sampling interval to verify", reader.path
        )
    for name in DATASET_SPECS:
        reader[name]
    return camera


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """枚举所有 episode 目录，损坏文件由读取边界直接报错。"""
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "data.h5").is_file():
        return (root,)
    episodes = tuple(
        sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and child.name.startswith("episode_")
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode directories found in {root}")
    return episodes


def iter_canonical_blocks(
    reader: RawReader,
    config: ExportConfig,
    *,
    arm_fk,
    hand_fk,
    hand_cloud,
    chunk_frames: int,
    camera: CameraSnapshot,
    handbase_rotation_eef: np.ndarray,
):
    """分块转换，不修补缺失观测或重采样行。"""
    height, width = camera.color.height, camera.color.width
    processor = PointCloudProcessor(
        depth_scale_m=camera.depth_scale_m,
        intrinsics=camera.color,
        T_xarm_base_from_color=camera.T_xarm_base_from_color,
        table_plane_abcd=config.table_plane_abcd,
        config=config.pointcloud,
    )
    camera_parameters = camera_parameters_cv(camera.color, camera.T_xarm_base_from_color)
    images = reader.iter_rgb_frames()
    for start in range(0, reader.num_frames, chunk_frames):
        end = min(reader.num_frames, start + chunk_frames)
        raw = {name: reader[name][start:end] for name in DATASET_SPECS}
        for name in (
            "arm_qpos",
            "hand_qpos",
            "action_arm_joint_target",
            "action_hand_joint_target",
        ):
            valid = np.isfinite(raw[name]).all(axis=1)
            if not valid.all():
                row = start + int(np.flatnonzero(~valid)[0])
                raise ValueError(f"Raw {name}: nonfinite values at row {row}")
        arm, hand = raw["arm_qpos"], raw["hand_qpos"]
        arm_action, hand_action = (
            raw["action_arm_joint_target"],
            raw["action_hand_joint_target"],
        )
        poses = eef_poses(arm, arm_fk=arm_fk)
        action_poses = eef_poses(arm_action, arm_fk=arm_fk)
        hand_features = hand_geometry(
            poses,
            hand,
            hand_fk=hand_fk,
            handbase_position_eef_m=config.handbase_position_eef_m,
            handbase_rotation_eef=handbase_rotation_eef,
            hand_cloud=hand_cloud,
        )
        block = {
            "joint_state": np.concatenate((arm, hand), axis=1, dtype=np.float32),
            "action": np.concatenate((arm_action, hand_action), axis=1, dtype=np.float32),
            "action_ee": np.concatenate((action_poses, hand_action), axis=1, dtype=np.float32),
            "eef_pose": poses.astype(np.float32),
            "finger_tactile_force": raw["hand_contact"],
            "taxel_tactile_force": raw["hand_tactile_force"],
            **hand_features,
        }
        for name, values in camera_parameters.items():
            block[name] = np.broadcast_to(values, (end - start, values.size))
        # 深度按块读取，RGB 逐帧解码到输出数组。
        rgb = np.empty((end - start, height, width, 3), dtype=np.uint8)
        depth = reader["depth"][start:end]
        clouds = np.empty((end - start, config.pointcloud.num_points, 6), dtype=np.float32)
        for offset, index in enumerate(range(start, end)):
            image = next(images, None)
            if image is None or image.shape != (height, width, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{reader.path / 'rgb.mp4'}: RGB missing or invalid at row {index}"
                )
            cloud = processor.process(depth[offset], image)
            if cloud is None:
                raise ValueError(f"point_cloud unavailable at row {index}")
            rgb[offset], clouds[offset] = image, cloud
        block.update(rgb=rgb, depth=depth, point_cloud=clouds)
        yield block
    if next(images, None) is not None:
        raise ValueError(f"{reader.path / 'rgb.mp4'}: RGB contains extra frames")


def export_zarr(
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
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing canonical Zarr: {output}")
    selected, excluded = select_episodes(source, exclude)
    rotation = quat_wxyz_to_rotmat(processing.handbase_quat_eef_wxyz)
    arm_fk, hand_fk = ArmFK(), HandFK()
    hand_cloud = HandPointCloud(hand_fk, processing.hand_pointcloud.num_points)
    staging = None
    ends: list[int] = []
    episode_ids: list[str] = []
    try:
        for index, episode in enumerate(selected):
            if progress_callback:
                progress_callback(index, len(selected))
            offset = ends[-1] if ends else 0
            with RawReader(episode) as reader:
                camera = validate_export_episode(reader)
                task = reader.meta["task_label"]
                if not task.strip() or any(ord(char) < 32 or ord(char) == 127 for char in task):
                    raise ValueError("task_label must be nonempty without control characters")
                height, width = camera.color.height, camera.color.width
                if index == 0:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(
                        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
                    )
                    root, arrays, chunk = create_store(
                        staging,
                        processing,
                        camera,
                        task=task,
                        control_hz=reader.control_hz,
                    )
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
        publish(cast(Path, staging), target)
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


def create_store(path, processing, camera, *, task, control_hz):
    height, width = camera.color.height, camera.color.width
    specs = canonical_array_specs(
        processing.pointcloud.num_points,
        height,
        width,
        processing.hand_pointcloud.num_points,
    )
    row_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in specs.values())
    chunk = min(MAX_BATCH_FRAMES, max(1, PROCESSING_BYTES // row_bytes))
    period_ns, tolerance_ns = sampling_clock_ns(control_hz)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(
        format=CANONICAL_FORMAT,
        task_name=task,
        dt=1.0 / control_hz,
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
        storage_rows = max(1, min(STORAGE_CHUNK_MAX_ROWS, STORAGE_CHUNK_BYTES // row_bytes))
        arrays[name] = data.create_dataset(
            name,
            shape=(0, *shape),
            chunks=(storage_rows, *shape),
            dtype=dtype,
            compressor=zarr.get_codec({"id": "zstd", "level": 3}),
        )
    return root, arrays, chunk


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
