"""Raw 格式、元数据与 RGB-D 读写。"""

from __future__ import annotations

import json
import logging
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
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.robot.model import (
    XARM7_JOINT_NAMES,
    XHAND_SDK_JOINT_NAMES,
    XHAND_TACTILE_SCALE_DIVISOR,
)
from dexmani_real.utils.io import write_json
from dexmani_real.utils.timing import sampling_clock_ns

if TYPE_CHECKING:
    import av
    import h5py

logger = logging.getLogger(__name__)


class RawDataError(ValueError):
    """输入 Raw 不满足完整示教的数据契约。"""


ARM_ACTION_SEMANTICS = "q_goal_submission_attempt"
TACTILE_FINGER_ORDER = "thumb,index,middle,ring,pinky"
EPISODE_OUTCOMES = ("success", "failure", "unknown")


def read_episode_outcome(episode_path: str | Path) -> dict[str, object]:
    """已存在的 Episode 缺少标注才表示 unknown；非法标注与 I/O 错误直接报告。"""
    episode = Path(episode_path).expanduser()
    if not episode.is_dir():
        raise NotADirectoryError(episode)
    path = episode / "outcome.json"
    try:
        with path.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
    except FileNotFoundError:
        return {"outcome": "unknown"}
    if not isinstance(annotation, dict) or annotation.get("outcome") not in EPISODE_OUTCOMES:
        raise ValueError(f"{path}: outcome must be one of {EPISODE_OUTCOMES}")
    return annotation


def write_episode_outcome(
    episode_path: str | Path, outcome: str, *, note: str | None = None, overwrite: bool = False,
) -> Path:
    """向已有目录写入独立人工标签。"""
    if outcome not in EPISODE_OUTCOMES:
        raise ValueError(f"outcome must be one of {EPISODE_OUTCOMES}")
    path = Path(episode_path).expanduser()
    if not path.is_dir():
        raise NotADirectoryError(path)
    annotation = {"outcome": outcome}
    if note is not None:
        annotation["note"] = note
    return write_json(annotation, path / "outcome.json", overwrite=overwrite)


ROBOT_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "arm_qpos": ((7,), np.dtype(np.float64)),
    "hand_qpos": ((12,), np.dtype(np.float64)),
    # 两种触觉直接保留 Feedback 的 SDK / 255 最近有效值及其原来源时间。
    "hand_contact": ((5, 3), np.dtype(np.float32)),
    "hand_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    # 记录最终提交的绝对目标，包括失败的提交尝试。
    "action_arm_joint_target": ((7,), np.dtype(np.float64)),
    "action_hand_joint_target": ((12,), np.dtype(np.float64)),
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
# 来源时间必须为正；触觉时间为最后一份完整有效通道帧的查询起点。
TIMING_ROW_DTYPE = np.dtype(
    [
        (name, "<i8")
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


def validate_task_label(task_label) -> None:
    if (
        not isinstance(task_label, str)
        or not task_label.strip()
        or "\0" in task_label
    ):
        raise ValueError("task_label must be a non-empty string without NUL")


def encode_raw_row(row, command):
    arm, hand = row.arm[0], row.hand[0]
    return {
        "arm_qpos": arm["qpos"],
        "hand_qpos": hand["qpos"],
        "hand_contact": hand["tactile_aggregate"],
        "hand_tactile_force": hand["tactile_dense"],
        "action_arm_joint_target": (
            command.arm_qpos
            if command is not None and command.arm_qpos is not None
            else np.full(7, np.nan)
        ),
        "action_hand_joint_target": (
            command.hand_qpos
            if command is not None and command.hand_qpos is not None
            else np.full(12, np.nan)
        ),
    }


def encode_timing_row(row):
    hand = row.hand
    return (
        int(row.observation_timestamp_ns),
        int(row.control_timestamp_ns),
        int(row.arm["timestamp_ns"][0]),
        int(hand["timestamp_ns"][0]),
        int(hand["tactile_aggregate_timestamp_ns"][0]),
        int(hand["tactile_dense_timestamp_ns"][0]),
        int(row.camera["timestamp_ns"]),
        int(row.camera["ring_sequence"]),
    )


def camera_metadata(camera: CameraSnapshot):
    color = camera.color
    values = {
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


def positive_metadata_scalar(meta, key):
    value = np.asarray(meta[key])
    if value.shape != () or value.dtype.kind not in "fiu" or not np.isfinite(value) or value <= 0:
        raise RawDataError(f"Raw metadata {key}: expected finite positive numeric scalar")
    return float(value)


def decode_camera_metadata(meta, *, include_depth=True, include_extrinsics=True) -> CameraSnapshot:
    model = meta["camera_color_distortion_model"]
    if not isinstance(model, str):
        raise ValueError("camera_color_distortion_model must be a string")
    coeffs = np.asarray(meta["camera_color_distortion_coeffs"])
    if coeffs.shape != (5,) or coeffs.dtype.kind not in "fiu":
        raise ValueError("camera_color_distortion_coeffs must contain five numeric values")
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
        transform = validate_rigid_transform(
            np.asarray(meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
            label="camera_T_xarm_base_from_color",
        )
    scale = None
    if include_depth:
        mode = meta["camera_payload_mode"]
        if not isinstance(mode, str) or mode not in {"rgb", "depth_to_color_aligned_rgbd"}:
            raise ValueError(f"Unsupported Raw camera payload: {mode!r}")
        if mode == "depth_to_color_aligned_rgbd":
            scale = positive_metadata_scalar(meta, "depth_scale")
    return CameraSnapshot(None, color, scale, transform)


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
        self.stream.time_base = Fraction(1, 1_000_000_000)
        self.stream.codec_context.time_base = self.stream.time_base
        self.stream.codec_context.open()
        self.container.start_encoding()

    def write_frame(self, frame, control_ns):
        import av

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        if self.first_control_ns is None:
            self.first_control_ns = control_ns
        av_frame.pts = control_ns - self.first_control_ns
        av_frame.time_base = Fraction(1, 1_000_000_000)
        stream = cast("av.video.stream.VideoStream", self.stream)
        container = cast("av.container.OutputContainer", self.container)
        for packet in stream.encode(av_frame):
            container.mux(packet)

    def close(self):
        if self.container is None:
            return
        # 取走编码流；flush 即使失败，也不能再次送入编码器。
        stream, self.stream = self.stream, None
        try:
            if stream is not None:
                for packet in stream.encode(None):
                    self.container.mux(packet)
        finally:
            self.container.close()
            self.container = None


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
        rgb_shape = (metadata["camera_color_height"], metadata["camera_color_width"], 3)
        self.datasets["depth"] = data.create_dataset(
            "depth",
            shape=(0, *rgb_shape[:2]),
            maxshape=(None, *rgb_shape[:2]),
            chunks=(1, *rgb_shape[:2]),
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
        height, width, _ = rgb_shape
        self.video = _VideoWriter(
            self.path / "rgb.mp4", fps=self.control_hz, width=width, height=height
        )
        self.video.open()

    def append_frame(self, row, command, segment_start=False):
        values = encode_raw_row(row, command)
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

    def finish(self, reason, *, recording_truncated_reason=None):
        self.flush()
        if recording_truncated_reason is not None:
            self.meta.attrs["recording_truncated_reason"] = recording_truncated_reason
        # 关闭资源成功与否由 recorder 监督方确认，元数据本身不代表已发布。
        self.meta.attrs["termination_reason"] = reason

    def close(self):
        try:
            try:
                if self.video is not None:
                    self.video.close()
                    self.video = None
            finally:
                if self.data_file is not None:
                    self.data_file.close()
                    self.data_file = None
        finally:
            self.arrays = {}
            self.timing_rows = None


def require_published_raw(path):
    requested = Path(path).expanduser()
    resolved = requested.resolve()
    if any(candidate.name.startswith(".tmp_") for candidate in (requested, resolved)):
        raise ValueError(f"unpublished Raw: {requested}")


class RawReader:
    def __init__(self, episode_path: str | Path):
        import h5py

        requested_path = Path(episode_path).expanduser()
        self.path = requested_path.resolve()
        self.rgb_path = self.path / "rgb.mp4"
        self.video_container: av.container.InputContainer | None = None
        self.dataset_cache: dict[str, h5py.Dataset] = {}
        self._hoi_file = None
        self.data_file = h5py.File(self.path / "data.h5", "r")
        try:
            rows = self.data_file["timing/rows"]
            if not isinstance(rows, h5py.Dataset) or rows.ndim != 1:
                raise RawDataError("Raw timing/rows: expected a one-dimensional dataset")
            self.num_frames = len(rows)
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

    def read_timing(self, stop: int | None = None) -> np.ndarray:
        """读取并校验所选时间行；连续段由 segment_starts 指定。"""
        rows = self["timing/rows"][:stop]
        control, observation = rows["control_ns"], rows["observation_ns"]
        for bad, detail, offset in (
            ((control < 0) | (observation < control), "invalid_control_or_observation_clock", 0),
            (np.diff(control) <= 0, "control_clock_not_increasing", 1),
            (np.diff(observation) <= 0, "observation_clock_not_increasing", 1),
        ):
            indices = np.flatnonzero(bad)
            if len(indices):
                raise ValueError(f"Raw {self.path}: {detail} at row {int(indices[0]) + offset}")
        return rows

    @property
    def segment_starts(self) -> np.ndarray:
        import h5py

        dataset = self.data_file["timing/segment_starts"]
        if not isinstance(dataset, h5py.Dataset):
            raise RawDataError(f"Raw {self.path}: timing/segment_starts must be a dataset")
        starts = np.asarray(dataset[()])
        if (starts.ndim != 1 or starts.dtype.kind not in "iu"
                or (self.num_frames == 0 and starts.size != 0)
                or (self.num_frames > 0 and (starts.size == 0 or starts[0] != 0))
                or np.any(starts[1:] <= starts[:-1])
                or np.any(starts >= self.num_frames)):
            raise ValueError(f"Raw {self.path}: invalid timing/segment_starts")
        return starts

    def require_fields(self, *names):
        """按需打开并校验字段，不枚举与当前任务无关的 HDF5 对象。"""
        for name in names:
            if name == "rgb":
                if not self.rgb_path.is_file():
                    raise FileNotFoundError(self.rgb_path)
            else:
                self[name]

    def __getitem__(self, name):
        if name not in self.dataset_cache:
            if name in {"hand_contact", "hand_tactile_force"}:
                self.validate_tactile_metadata()
            elif name in {"arm_qpos", "action_arm_joint_target"}:
                self.validate_joint_metadata("arm", XARM7_JOINT_NAMES)
            elif name in {"hand_qpos", "action_hand_joint_target"}:
                self.validate_joint_metadata("hand", XHAND_SDK_JOINT_NAMES)
            if (
                name in {"action_arm_joint_target", "action_hand_joint_target"}
                and self.meta["arm_action_semantics"] != ARM_ACTION_SEMANTICS
            ):
                raise ValueError(f"Raw arm_action_semantics: expected {ARM_ACTION_SEMANTICS}")
            array = (self.hoi_file if name == "object_pointcloud" else self.data_file)[name]
            import h5py
            if not isinstance(array, h5py.Dataset):
                raise RawDataError(f"Raw {name}: expected a dataset")
            spec = (
                ((), TIMING_ROW_DTYPE) if name == "timing/rows"
                else ROBOT_DATASET_SPECS.get(name, VR_DATASET_SPECS.get(name))
            )
            if name == "depth":
                expected = (
                    self.num_frames,
                    self.meta["camera_color_height"],
                    self.meta["camera_color_width"],
                )
                if array.shape != expected or array.dtype != np.uint16:
                    raise ValueError(f"Raw {self.path}: depth must be uint16 {expected}")
            if name == "object_pointcloud" and (
                array.shape != (self.num_frames, HOI_NUM_POINTS, 3) or array.dtype != np.float32
            ):
                raise ValueError("Raw hoi.h5/object_pointcloud: shape/dtype 错误")
            if spec is not None:
                shape, dtype = spec
                valid_dtype = array.dtype.kind == "f" if dtype.kind == "f" else array.dtype == dtype
                if array.shape != (self.num_frames, *shape) or not valid_dtype:
                    raise ValueError(f"Raw {name}: incompatible shape/dtype")
            self.dataset_cache[name] = array
        return self.dataset_cache[name]

    @property
    def hoi_file(self):
        import h5py

        if self._hoi_file is None:
            self._hoi_file = h5py.File(self.path / "hoi.h5", "r")
        return self._hoi_file

    def read_camera(self, *, include_depth=True, include_extrinsics=True) -> CameraSnapshot:
        return decode_camera_metadata(
            self.meta, include_depth=include_depth, include_extrinsics=include_extrinsics,
        )

    def validate_joint_metadata(self, device, joint_names) -> None:
        if self.meta["joint_units"] != "rad":
            raise ValueError("Raw joint_units: expected rad")
        name = f"{device}_joint_names"
        encoded = self.meta[name]
        if not isinstance(encoded, (str, bytes, bytearray)):
            raise RawDataError(f"Raw {name}: invalid joint names format; expected JSON text")
        try:
            names = json.loads(encoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RawDataError(f"Raw {name}: invalid joint names format: {exc}") from exc
        if names != list(joint_names):
            raise ValueError(f"Raw {name}: incompatible joint order")

    def validate_tactile_metadata(self) -> None:
        if self.meta.get("hand_tactile_finger_order") != TACTILE_FINGER_ORDER:
            raise ValueError("Raw hand_tactile_finger_order: incompatible tactile semantics")
        divisor = positive_metadata_scalar(self.meta, "hand_tactile_scale_divisor")
        if divisor != XHAND_TACTILE_SCALE_DIVISOR:
            raise ValueError(
                f"Raw hand_tactile_scale_divisor: expected {XHAND_TACTILE_SCALE_DIVISOR}, "
                f"got {divisor!r}"
            )

    def read_timestamps_s(self, stop: int | None = None) -> np.ndarray:
        """读取所选前缀相对于首帧的观测时间，默认读取完整时间轴。"""
        stamps = self["timing/rows"].fields("observation_ns")[:stop]
        # 先减整数时钟，避免长时间运行后的浮点精度损失。
        return (stamps - stamps[0]) / 1e9 if stamps.size else np.empty(0, dtype=np.float64)

    def iter_rgb_frames(self):
        import av

        if self.video_container is None:
            self.video_container = av.open(str(self.rgb_path), "r")
            if not self.video_container.streams.video:
                raise RawDataError(f"Raw {self.rgb_path}: missing video stream")
            stream = self.video_container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.thread_count = 4
        self.video_container.seek(0)
        expected = (self.meta["camera_color_height"], self.meta["camera_color_width"], 3)
        for index, frame in enumerate(self.video_container.decode(video=0)):
            image = frame.to_ndarray(format="rgb24")
            if image.shape != expected:
                raise ValueError(
                    f"Raw {self.path}: RGB grid invalid at row {index}, expected {expected}"
                )
            yield image
            del image, frame

    def close(self):
        self.dataset_cache.clear()
        try:
            if self.video_container is not None:
                self.video_container.close()
                self.video_container = None
        finally:
            try:
                if self._hoi_file is not None:
                    self._hoi_file.close()
                    self._hoi_file = None
            finally:
                if self.data_file is not None:
                    self.data_file.close()
                    self.data_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            logger.exception("Raw reader close also failed")
