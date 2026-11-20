"""Raw 格式、元数据与 RGB-D 读写。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    CameraSnapshot,
    parse_camera_matrix,
)
from dexmani_real.geometry.pose import ROTATION_ATOL, validate_rigid_transform
from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XARM7_JOINT_NAMES,
    XHAND_SDK_JOINT_NAMES,
    XHAND_TACTILE_SCALE_DIVISOR,
)
from dexmani_real.utils.timing import sampling_clock_ns

if TYPE_CHECKING:
    import av
    import h5py

    from dexmani_real.config.real import RealConfig

logger = logging.getLogger(__name__)


class RawDataError(ValueError):
    """输入 Raw 不满足完整示教的数据契约。"""


ARM_ACTION_SEMANTICS = "q_goal_confirmed_submission"
ACTION_TARGET_FIELDS = ("action_arm_joint_target", "action_hand_joint_target")
TACTILE_FINGER_ORDER = "thumb,index,middle,ring,pinky"


ROBOT_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "arm_qpos": (ARM_JOINT_SHAPE, np.dtype(np.float64)),
    "hand_qpos": (HAND_JOINT_SHAPE, np.dtype(np.float64)),
    # 触觉保留校正后 SDK / 255，无法使用的指尖以 NaN 表示。
    "hand_contact": (HAND_TACTILE_SUM_SHAPE, np.dtype(np.float32)),
    "hand_tactile_force": (HAND_TACTILE_FORCE_SHAPE, np.dtype(np.float32)),
    # 无新提交的观测行以两组全 NaN 目标表示。
    "action_arm_joint_target": (ARM_JOINT_SHAPE, np.dtype(np.float64)),
    "action_hand_joint_target": (HAND_JOINT_SHAPE, np.dtype(np.float64)),
}


# 遥操作原始输入，与控制行对齐保存，不作为训练导出字段。
VR_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "vr_wrist_pos": ((3,), np.dtype(np.float64)),
    "vr_wrist_quat_wxyz": ((4,), np.dtype(np.float64)),
    "vr_landmarks": ((21, 3), np.dtype(np.float64)),
    "vr_recv_ts_ns": ((), np.dtype(np.uint64)),
    "vr_sequence_id": ((), np.dtype(np.uint64)),
}


# 时间字段单位 ns，保留主机单调时钟的查询/接收时间，不代表同步硬件采样。
# 触觉时间为最老的指尖来源时间，缺少任何指尖时为 0。
TIMING_ROW_DTYPE = np.dtype(
    [
        (name, np.dtype("<i8"))
        for name in (
            "observation_ns",
            "control_ns",
            "arm_ns",
            "hand_ns",
            "tactile_aggregate_ns",
            "tactile_dense_ns",
            "camera_ns",
            "camera_sequence",
        )
    ]
)

CAMERA_TIMING_FIELDS = ("camera_ns", "camera_sequence")
CAMERA_TIMING_DTYPE = np.dtype([(name, TIMING_ROW_DTYPE[name]) for name in CAMERA_TIMING_FIELDS])


def validate_task_label(task_label) -> None:
    if (
        not isinstance(task_label, str)
        or not task_label.strip()
        or "\0" in task_label
    ):
        raise ValueError("task_label must be a non-empty string without NUL")


def encode_raw_row(row, submitted_command):
    arm, hand = row.arm[0], row.hand[0]
    if submitted_command is None:
        arm_target = np.full(ARM_JOINT_SHAPE, np.nan)
        hand_target = np.full(HAND_JOINT_SHAPE, np.nan)
    else:
        arm_target, hand_target = submitted_command.arm_qpos, submitted_command.hand_qpos
        if arm_target is None or hand_target is None:
            raise ValueError("Raw action requires a complete arm and hand target")
    return {
        "arm_qpos": arm["qpos"],
        "hand_qpos": hand["qpos"],
        "hand_contact": hand["tactile_aggregate"],
        "hand_tactile_force": hand["tactile_dense"],
        "action_arm_joint_target": arm_target,
        "action_hand_joint_target": hand_target,
    }


def encode_timing_row(row):
    hand = row.hand
    return (
        int(row.observation_timestamp_ns),
        int(row.control_timestamp_ns),
        int(row.arm["timestamp_ns"][0]),
        int(hand["timestamp_ns"][0]),
        int(hand["tactile_aggregate_timestamp_ns"][0].min()),
        int(hand["tactile_dense_timestamp_ns"][0].min()),
        int(row.camera["timestamp_ns"]),
        int(row.camera["ring_sequence"]),
    )


def camera_metadata(camera: CameraSnapshot) -> dict[str, object]:
    color = camera.color
    values: dict[str, object] = {
        "camera_payload_mode": "rgb" if camera.depth_scale_m is None else "depth_to_color_aligned_rgbd",
        "camera_color_width": color.width,
        "camera_color_height": color.height,
        "camera_color_intrinsics": color.matrix().reshape(-1).tolist(),
        "camera_color_distortion_model": color.distortion_model,
        "camera_color_distortion_coeffs": list(color.distortion_coeffs),
    }
    if camera.serial is not None:
        values["camera_serial"] = camera.serial
    if camera.depth_scale_m is not None:
        values["depth_scale"] = camera.depth_scale_m
    if camera.T_xarm_base_from_color is not None:
        values["camera_T_xarm_base_from_color"] = camera.T_xarm_base_from_color.reshape(-1).tolist()
    return values


def robot_recording_parameters(runtime: RealConfig) -> dict[str, dict[str, object]]:
    """提取 Arm/Hand 的录制参数白名单。"""
    return {
        "arm": {name: getattr(runtime.arm, name) for name in (
            "ip", "collision_sensitivity", "tcp_load_mass_kg", "tcp_load_cog_mm",
            "joint_limit_lower", "joint_limit_upper", "max_joint_velocity_deg_per_s",
            "max_joint_acceleration_deg_per_s2", "max_joint_jerk_deg_per_s3", "servo_hz",
            "feedback_hz", "feedback_max_age_s", "standstill_position_tolerance_rad",
            "standstill_velocity_tolerance_rad_s",
        )},
        "hand": {name: getattr(runtime.hand, name) for name in (
            "enabled", "comm_type", "device_name", "device_id", "qpos_min_rad", "qpos_max_rad",
            "kp", "ki", "kd", "tor_max", "feedback_max_age_s", "crc_retries", "crc_retry_interval_s",
            "tactile_max_age_s", "tactile_calibration_samples", "tactile_calibration_timeout_s",
            "tactile_contact_limits", "tactile_residual_limits",
            "baudrate" if runtime.hand.comm_type == "serial" else "ethercat_slave_position",
        )},
    }


def episode_recording_metadata(
    camera_snapshot: CameraSnapshot,
    effective_parameters: Mapping[str, object],
    *,
    tactile_reset: Mapping[str, object],
    tactile_calibration: Mapping[str, object],
) -> dict[str, object]:
    """相机标定存于顶层，实验参数与触觉校准存于 JSON。"""
    metadata = camera_metadata(camera_snapshot)
    parameters = {
        **effective_parameters,
        "tactile_reset": tactile_reset,
        "tactile_calibration": tactile_calibration,
    }
    metadata["effective_parameters"] = json.dumps(parameters, allow_nan=False)
    return metadata


def positive_metadata_scalar(meta, key):
    value = np.asarray(meta[key])
    if value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value) or value <= 0:
        raise RawDataError(f"Raw metadata {key}: expected finite positive numeric scalar")
    return float(value)


def decode_camera_extrinsic(meta):
    return validate_rigid_transform(
        np.asarray(meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
        label="camera_T_xarm_base_from_color",
    )


def decode_camera_metadata(meta, *, include_depth=True, include_extrinsics=True) -> CameraSnapshot:
    model = meta["camera_color_distortion_model"]
    if not isinstance(model, str):
        raise RawDataError("Raw camera_color_distortion_model must be a string")
    coeffs = np.asarray(meta["camera_color_distortion_coeffs"])
    if coeffs.shape != (5,) or coeffs.dtype.kind not in "fiu":
        raise RawDataError("camera_color_distortion_coeffs must contain five numeric values")
    matrix = parse_camera_matrix(meta["camera_color_intrinsics"])
    color = CameraIntrinsics(
        width=np.asarray(meta["camera_color_width"]).item(),
        height=np.asarray(meta["camera_color_height"]).item(),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        distortion_model=model,
        distortion_coeffs=tuple(coeffs),
    )
    transform = None
    if include_extrinsics and "camera_T_xarm_base_from_color" in meta:
        transform = decode_camera_extrinsic(meta)
    scale = None
    if include_depth:
        mode = meta["camera_payload_mode"]
        if not isinstance(mode, str) or mode != "depth_to_color_aligned_rgbd":
            raise RawDataError(f"Raw camera_payload_mode must be depth_to_color_aligned_rgbd, got {mode!r}")
        scale = positive_metadata_scalar(meta, "depth_scale")
    return CameraSnapshot(None, color, scale, transform)


_RGB_TIME_BASE = Fraction(1, 1_000_000_000)


def _close_handles(owner, *names: str, failure: BaseException | None = None) -> None:
    """依次关闭已打开的句柄，保留首错和未确认关闭的句柄。"""
    for name in names:
        handle = getattr(owner, name)
        if handle is None:
            continue
        try:
            handle.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                logger.exception("Raw %s close also failed: %s", name, owner.path)
        else:
            setattr(owner, name, None)
    if failure is not None:
        raise failure


class _VideoWriter:
    def __init__(self, path, *, fps, width, height):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.container = None
        self.stream = None
        self.first_control_ns = None

    def open(self):
        import av

        self.container = av.open(str(self.path), "w", format="mp4")
        self.stream = self.container.add_stream("libx264", rate=self.rate)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "yuv444p"
        self.stream.options = {"crf": "18", "preset": "ultrafast"}
        # 为控制和推理保留 CPU。
        self.stream.codec_context.thread_count = 1
        self.stream.time_base = _RGB_TIME_BASE
        self.stream.codec_context.time_base = self.stream.time_base
        self.stream.codec_context.open()
        self.container.start_encoding()

    def write_frame(self, frame, control_ns):
        import av

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        if self.first_control_ns is None:
            self.first_control_ns = control_ns
        av_frame.pts = control_ns - self.first_control_ns
        av_frame.time_base = _RGB_TIME_BASE
        stream = cast("av.video.stream.VideoStream", self.stream)
        container = cast("av.container.OutputContainer", self.container)
        for packet in stream.encode(av_frame):
            container.mux(packet)

    def close(self):
        # 取走编码流；flush 即使失败，也不能再次送入编码器。
        stream, self.stream = self.stream, None
        failure = None
        try:
            if stream is not None:
                container = cast("av.container.OutputContainer", self.container)
                for packet in stream.encode(None):
                    container.mux(packet)
        except BaseException as exc:  # noqa: BLE001 — flush 失败后仍须关闭容器，随后传播首错。
            failure = exc
        _close_handles(self, "container", failure=failure)


class RawWriter:
    def __init__(self, path, *, control_hz):
        self.path = Path(path)
        self.control_hz = control_hz
        self.data_file = self.video = None
        self.frame_count = 0
        self.buffered = 0

    def open(self, metadata):
        import h5py

        self.data_file = data = h5py.File(self.path / "data.h5", "w")
        self.meta = data.create_group("meta")
        self.meta.attrs.update(metadata)
        self.meta.attrs.update(
            termination_reason="recording_incomplete",
            control_hz=self.control_hz,
            arm_action_semantics=ARM_ACTION_SEMANTICS,
            joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
            hand_tactile_finger_order=TACTILE_FINGER_ORDER,
            hand_tactile_scale_divisor=XHAND_TACTILE_SCALE_DIVISOR,
        )
        specs = ROBOT_DATASET_SPECS
        if metadata["collection_source"] == "teleop":
            specs = specs | VR_DATASET_SPECS
        self.datasets = {
            name: data.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                chunks=(32, *tail_shape),
                dtype=dtype,
            )
            for name, (tail_shape, dtype) in specs.items()
        }
        timing = data.create_group("timing")
        self.row_times = timing.create_dataset(
            "rows", shape=(0,), maxshape=(None,), dtype=TIMING_ROW_DTYPE
        )
        self.segment_starts = timing.create_dataset(
            "segment_starts", shape=(0,), maxshape=(None,), dtype=np.int64,
        )
        height, width = metadata["camera_color_height"], metadata["camera_color_width"]
        self.datasets["depth"] = data.create_dataset(
            "depth",
            shape=(0, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype=np.uint16,
            compression="gzip",
            compression_opts=1,
        )
        # 每列复制一次，批缓冲不持有整份 RGB-D 快照。
        row_bytes = TIMING_ROW_DTYPE.itemsize + sum(
            int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
            for dataset in self.datasets.values()
        )
        batch_size = max(1, min(32, 8 * 1024 * 1024 // row_bytes))
        self.arrays = {
            name: np.empty((batch_size, *dataset.shape[1:]), dtype=dataset.dtype)
            for name, dataset in self.datasets.items()
        }
        self.timing_rows: np.ndarray | None = np.empty(batch_size, dtype=TIMING_ROW_DTYPE)
        self.video = _VideoWriter(
            self.path / "rgb.mp4", fps=self.control_hz, width=width, height=height
        )
        self.video.open()

    def append_frame(self, row, submitted_command, segment_start: bool):
        values = encode_raw_row(row, submitted_command)
        if "vr_landmarks" in self.datasets:
            values.update({name: row.vr[name.removeprefix("vr_")] for name in VR_DATASET_SPECS})
        rgb, depth = row.camera["rgb"], row.camera["depth"]
        timing = encode_timing_row(row)
        values["depth"] = depth
        for name, array in self.arrays.items():
            array[self.buffered] = values[name]
        timing_rows = cast(np.ndarray, self.timing_rows)
        timing_rows[self.buffered] = timing
        cast(_VideoWriter, self.video).write_frame(rgb, row.control_timestamp_ns)
        # 只有 Writer 知道实际接收顺序；缓冲中尚未 flush 的行也计入索引。
        index = self.frame_count + self.buffered
        if index == 0 or segment_start:
            count = len(self.segment_starts)
            self.segment_starts.resize(count + 1, axis=0)
            self.segment_starts[count] = index
        self.buffered += 1
        if self.buffered == len(timing_rows):
            self.flush()

    def flush(self):
        if not self.buffered:
            return
        # 写入前清空计数，失败后不能重试并重复追加部分数据。
        count, self.buffered = self.buffered, 0
        end = self.frame_count + count
        for name, values in self.arrays.items():
            self.datasets[name].resize(end, axis=0)
            self.datasets[name][self.frame_count : end] = values[:count]
        self.row_times.resize(end, axis=0)
        self.row_times[self.frame_count : end] = cast(np.ndarray, self.timing_rows)[:count]
        self.frame_count = end

    def finish(self, reason, *, recording_truncated_reason=None, xhand_crc_counts: dict[str, int]):
        self.flush()
        if recording_truncated_reason is not None:
            self.meta.attrs["recording_truncated_reason"] = recording_truncated_reason
        if xhand_crc_counts:
            self.meta.attrs["xhand_crc_counts_json"] = json.dumps(xhand_crc_counts)
        # 关闭资源成功与否由 recorder 监督方确认，元数据本身不代表已发布。
        self.meta.attrs["termination_reason"] = reason

    def close(self):
        try:
            _close_handles(self, "video", "data_file")
        finally:
            self.arrays = {}
            self.timing_rows = None


def open_hoi_file(path: Path, raw_meta, raw_timing):
    """校验标注的可读结构及其与 Raw 的相机对应关系。"""
    import h5py

    annotations = h5py.File(path, "r")
    try:
        transform = annotations["base_from_camera"]
        if not isinstance(transform, h5py.Dataset) or transform.shape != (4, 4):
            raise RawDataError("HOI base_from_camera must be a (4, 4) dataset")
        recorded = decode_camera_extrinsic(raw_meta)
        annotated = validate_rigid_transform(transform[()], label="HOI base_from_camera")
        if not np.allclose(recorded, annotated, atol=ROTATION_ATOL, rtol=0):
            raise RawDataError("HOI base_from_camera conflicts with Raw camera_T_xarm_base_from_color")
        names = annotations.get("template/object_names")
        if not isinstance(names, h5py.Dataset) or names.ndim != 1 or not len(names):
            raise RawDataError("HOI template/object_names must be a nonempty one-dimensional dataset")
        valid = annotations.get("object_valid")
        if (not isinstance(valid, h5py.Dataset)
                or valid.shape != (len(raw_timing), len(names)) or valid.dtype != bool):
            raise RawDataError("Invalid HOI object_valid shape/dtype")
        cloud = annotations.get("object_pointcloud")
        if (not isinstance(cloud, h5py.Dataset)
                or cloud.shape != (len(raw_timing), HOI_NUM_POINTS, 3) or cloud.dtype != np.float32):
            raise RawDataError("Invalid HOI object_pointcloud shape/dtype")
        timing = annotations.get("camera_timing")
        if not isinstance(timing, h5py.Dataset) or timing.shape != raw_timing.shape or timing.dtype != CAMERA_TIMING_DTYPE:
            raise RawDataError(f"{path}: missing or invalid HOI camera_timing; re-annotate with --overwrite")
        if not np.array_equal(timing[:], raw_timing.fields(CAMERA_TIMING_FIELDS)[:]):
            raise RawDataError(f"{path}: HOI camera_timing differs from Raw; re-annotate with --overwrite")
    except BaseException:
        try:
            annotations.close()
        except BaseException:
            logger.exception("HOI file close also failed: %s", path)
        raise
    return annotations


def published_raw_path(requested: Path) -> Path:
    resolved = requested.resolve()
    if requested.name.startswith(".tmp_") or resolved.name.startswith(".tmp_"):
        raise RawDataError(f"unpublished Raw: {requested}")
    return resolved


def discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    """枚举所有 episode 目录，损坏文件由读取边界直接报错。"""
    requested_path = Path(input_root).expanduser()
    if (requested_path / "data.h5").is_file():
        episode = published_raw_path(requested_path)
        return (requested_path.absolute() if requested_path.is_symlink() else episode,)
    root = requested_path.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
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


class RawReader:
    def __init__(self, episode_path: str | Path):
        import h5py

        self.path = published_raw_path(Path(episode_path).expanduser())
        self.rgb_path = self.path / "rgb.mp4"
        self.video_container: av.container.InputContainer | None = None
        self.dataset_cache: dict[str, h5py.Dataset] = {}
        self._hoi_file = None
        self.data_file = h5py.File(self.path / "data.h5", "r")
        try:
            reason = self.meta.get("termination_reason")
            if not isinstance(reason, str) or not reason or reason == "recording_incomplete":
                raise RawDataError(f"Raw {self.path}: invalid termination_reason {reason!r}")
            rows = self.data_file["timing/rows"]
            if not isinstance(rows, h5py.Dataset) or rows.ndim != 1 or rows.dtype != TIMING_ROW_DTYPE:
                raise RawDataError(
                    f"Raw {self.path}: timing/rows must be a one-dimensional dataset of dtype {TIMING_ROW_DTYPE}"
                )
            self.num_frames = len(rows)
            self.dataset_cache["timing/rows"] = rows
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("Raw reader close also failed")
            raise

    @property
    def meta(self):
        return self.data_file["meta"].attrs

    @property
    def control_hz(self):
        hz = positive_metadata_scalar(self.meta, "control_hz")
        sampling_clock_ns(hz)
        return hz

    def read_timing(self, stop: int | None = None, *, source_clocks: Iterable[str] = ()) -> np.ndarray:
        """读取并校验所选时间行；连续段由 segment_starts 指定。"""
        rows = self["timing/rows"][:stop]
        control, observation = rows["control_ns"], rows["observation_ns"]
        for bad, detail, offset in (
            ((control < 0) | (observation < control), "invalid_control_or_observation_clock", 0),
            (np.diff(control) <= 0, "control_clock_not_increasing", 1),
        ):
            indices = np.flatnonzero(bad)
            if len(indices):
                raise RawDataError(f"Raw {self.path}: {detail} at row {int(indices[0]) + offset}")
        self._validate_observation_clock(observation)
        for key in source_clocks:
            stamps = rows[key]
            invalid = (stamps <= 0) | (stamps > observation)
            invalid[1:] |= stamps[1:] < stamps[:-1]
            indices = np.flatnonzero(invalid)
            if len(indices):
                raise RawDataError(f"Raw {self.path}: invalid source timestamp: {key} at row {indices[0]}")
        return rows

    def _validate_observation_clock(self, stamps: np.ndarray) -> None:
        indices = np.flatnonzero(stamps[1:] <= stamps[:-1])
        if len(indices):
            raise RawDataError(
                f"Raw {self.path}: observation_clock_not_increasing at row {int(indices[0]) + 1}"
            )

    @property
    def segment_starts(self) -> np.ndarray:
        import h5py

        dataset = self.data_file["timing/segment_starts"]
        if not isinstance(dataset, h5py.Dataset):
            raise RawDataError(f"Raw {self.path}: timing/segment_starts must be a dataset")
        starts = np.asarray(dataset[()])
        if (starts.ndim != 1 or starts.dtype != np.int64
                or (self.num_frames == 0 and starts.size != 0)
                or (self.num_frames > 0 and (starts.size == 0 or starts[0] != 0))
                or np.any(starts[1:] <= starts[:-1])
                or np.any(starts >= self.num_frames)):
            raise RawDataError(f"Raw {self.path}: invalid timing/segment_starts")
        return starts.astype(np.intp, copy=False)

    def __getitem__(self, name):
        if name == "object_pointcloud":
            return self.hoi_file[name]
        if name not in self.dataset_cache:
            if name in {"hand_contact", "hand_tactile_force"}:
                self.validate_tactile_metadata()
            elif name in {"arm_qpos", "action_arm_joint_target"}:
                self.validate_joint_metadata("arm", XARM7_JOINT_NAMES)
            elif name in {"hand_qpos", "action_hand_joint_target"}:
                self.validate_joint_metadata("hand", XHAND_SDK_JOINT_NAMES)
            if (
                name in ACTION_TARGET_FIELDS
                and self.meta["arm_action_semantics"] != ARM_ACTION_SEMANTICS
            ):
                raise RawDataError(f"Raw arm_action_semantics: expected {ARM_ACTION_SEMANTICS}")
            array = self.data_file[name]
            import h5py
            if not isinstance(array, h5py.Dataset):
                raise RawDataError(f"Raw {name}: expected a dataset")
            spec: tuple[tuple[int, ...], np.dtype] | None
            if name == "depth":
                spec = (self.meta["camera_color_height"], self.meta["camera_color_width"]), np.dtype(np.uint16)
            else:
                spec = ROBOT_DATASET_SPECS.get(name, VR_DATASET_SPECS.get(name))
            if spec is not None:
                shape, dtype = spec
                expected = (self.num_frames, *shape)
                if array.shape != expected or array.dtype != dtype:
                    raise RawDataError(
                        f"Raw {name}: expected shape {expected} and dtype {dtype}, "
                        f"got shape {array.shape} and dtype {array.dtype}"
                    )
            self.dataset_cache[name] = array
        return self.dataset_cache[name]

    @property
    def hoi_file(self):
        if self._hoi_file is None:
            self._hoi_file = open_hoi_file(self.path / "hoi.h5", self.meta, self["timing/rows"])
        return self._hoi_file

    def read_hoi_validity(self, stop: int | None = None) -> np.ndarray:
        """由逐物体掩码计算所选前缀的整帧有效性。"""
        return self.hoi_file["object_valid"][:stop].all(axis=1)

    def read_camera(self, *, include_depth=True, include_extrinsics=True) -> CameraSnapshot:
        return decode_camera_metadata(
            self.meta, include_depth=include_depth, include_extrinsics=include_extrinsics,
        )

    def validate_joint_metadata(self, device, joint_names) -> None:
        if self.meta["joint_units"] != "rad":
            raise RawDataError("Raw joint_units: expected rad")
        name = f"{device}_joint_names"
        encoded = self.meta[name]
        if not isinstance(encoded, (str, bytes, bytearray)):
            raise RawDataError(f"Raw {name}: invalid joint names format; expected JSON text")
        try:
            names = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RawDataError(f"Raw {name}: invalid joint names format: {exc}") from exc
        if names != list(joint_names):
            raise RawDataError(f"Raw {name}: incompatible joint order")

    def validate_tactile_metadata(self) -> None:
        if self.meta.get("hand_tactile_finger_order") != TACTILE_FINGER_ORDER:
            raise RawDataError("Raw hand_tactile_finger_order: incompatible tactile semantics")
        divisor = positive_metadata_scalar(self.meta, "hand_tactile_scale_divisor")
        if divisor != XHAND_TACTILE_SCALE_DIVISOR:
            raise RawDataError(
                f"Raw hand_tactile_scale_divisor: expected {XHAND_TACTILE_SCALE_DIVISOR}, "
                f"got {divisor!r}"
            )

    def read_observation_stamps(self, stop: int | None = None) -> np.ndarray:
        """只读取观测时钟，校验所选时间戳严格递增。"""
        stamps = self["timing/rows"].fields("observation_ns")[:stop]
        self._validate_observation_clock(stamps)
        return stamps

    def iter_rgb_frames(self):
        """按 Raw 行读取图像，核对纳秒 PTS，消费结束时检查帧数。"""
        import av

        if self.video_container is None:
            self.video_container = av.open(str(self.rgb_path), "r")
            try:
                if not self.video_container.streams.video:
                    raise RawDataError(f"Raw {self.rgb_path}: missing video stream")
                stream = self.video_container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.thread_count = 4
            except BaseException as exc:  # noqa: BLE001 — 初始化失败时关闭容器并保留首错。
                _close_handles(self, "video_container", failure=exc)
        self.video_container.seek(0)
        expected = (self.meta["camera_color_height"], self.meta["camera_color_width"], 3)
        elapsed_ns = self["timing/rows"].fields("control_ns")[:]
        elapsed_ns -= elapsed_ns[:1]
        frames = self.video_container.decode(video=0)
        for index in range(self.num_frames):
            frame = next(frames, None)
            if frame is None:
                raise RawDataError(f"Raw {self.rgb_path}: RGB missing at row {index}")
            if frame.time_base != _RGB_TIME_BASE or frame.pts != elapsed_ns[index]:
                raise RawDataError(
                    f"Raw {self.rgb_path}: RGB timestamp mismatch at row {index}: "
                    f"expected {elapsed_ns[index]} ns, got PTS {frame.pts} with time base {frame.time_base}"
                )
            image = frame.to_ndarray(format="rgb24")
            if image.shape != expected:
                raise RawDataError(
                    f"Raw {self.path}: RGB grid invalid at row {index}, expected {expected}"
                )
            yield image
            del image, frame
        if next(frames, None) is not None:
            raise RawDataError(f"Raw {self.rgb_path}: unexpected RGB frame at row {self.num_frames}")

    def close(self):
        self.dataset_cache.clear()
        _close_handles(self, "video_container", "_hoi_file", "data_file")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            logger.exception("Raw reader close also failed")
