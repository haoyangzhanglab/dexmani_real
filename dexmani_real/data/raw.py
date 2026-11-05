"""Raw 格式、元数据与 RGB-D 读写。"""

from __future__ import annotations

import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    CameraSnapshot,
    parse_camera_matrix,
)
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.utils.timing import sampling_clock_ns

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

ARM_ACTION_SEMANTICS = "q_goal_submission_attempt"
TACTILE_UNITS = "sdk_native_unknown_physical_scale"
TACTILE_FINGER_ORDER = "thumb,index,middle,ring,pinky"
TACTILE_AGE_WARNING_NS = 500_000_000


ROBOT_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "arm_qpos": ((7,), np.dtype(np.float64)),
    "hand_qpos": ((12,), np.dtype(np.float64)),
    # 触觉保留 SDK 单位、最近有效值及其原来源时间。
    "hand_contact": ((5, 3), np.dtype(np.float32)),
    "hand_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    # 记录最终提交的绝对目标，包括失败的提交尝试。
    "action_arm_joint_target": ((7,), np.dtype(np.float64)),
    "action_hand_joint_target": ((12,), np.dtype(np.float64)),
}


# 仅供遥操作调试，不参与训练导出的字段与有效前缀判定。
VR_DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "vr_wrist_pos": ((3,), np.dtype(np.float64)),
    "vr_wrist_quat_wxyz": ((4,), np.dtype(np.float64)),
    "vr_landmarks": ((21, 3), np.dtype(np.float64)),
    "vr_recv_ts_ns": ((), np.dtype(np.uint64)),
    "vr_sequence_id": ((), np.dtype(np.uint64)),
}


# 时间字段单位 ns；采集端使用主机单调时钟，timestamp_source 仅作说明，不参与读取准入。
# 缺失用 -1，触觉时间为最后一份完整通道帧的来源时间。
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


def decode_camera_metadata(meta) -> CameraSnapshot:
    if meta["camera_payload_mode"] != "depth_to_color_aligned_rgbd":
        raise ValueError("Raw camera payload must be depth-to-color aligned RGB-D")
    matrix = parse_camera_matrix(meta["camera_color_intrinsics"])
    color = CameraIntrinsics(
        width=np.asarray(meta["camera_color_width"]).item(),
        height=np.asarray(meta["camera_color_height"]).item(),
        fx=matrix[0, 0],
        fy=matrix[1, 1],
        ppx=matrix[0, 2],
        ppy=matrix[1, 2],
        distortion_model=meta["camera_color_distortion_model"],
        distortion_coeffs=tuple(meta["camera_color_distortion_coeffs"]),
    )
    transform = None
    if "camera_T_xarm_base_from_color" in meta:
        transform = validate_rigid_transform(
            np.asarray(meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
            label="camera_T_xarm_base_from_color",
        )
    return CameraSnapshot(meta.get("camera_serial"), color, float(meta["depth_scale"]), transform)


def snapshot_recording_metadata(camera_snapshot, *, tactile_reset_metadata):
    """记录实际相机与触觉保持语义，不将旧值解释为当前接触。"""
    metadata = camera_metadata(camera_snapshot)
    metadata.update(tactile_reset_metadata)
    if tactile_reset_metadata:
        # HDF5 属性不支持 dict，设备结果只在录制边界编码。
        metadata["hand_tactile_reset_result"] = json.dumps(metadata["hand_tactile_reset_result"])
    metadata.update(
        hand_tactile_units=TACTILE_UNITS,
        hand_tactile_finger_order=TACTILE_FINGER_ORDER,
        hand_tactile_hold_policy="last_valid_until_recovery",
    )
    return metadata


class _VideoWriter:
    def __init__(self, path, *, fps, width, height):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)
        self.container = None
        self.stream = None
        self.frame_count = 0

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
        self.stream.codec_context.open()
        self.container.start_encoding()

    def write_frame(self, frame):
        import av

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        av_frame.pts = self.frame_count
        av_frame.time_base = 1 / self.rate
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.container is None:
            return
        failure = None
        try:
            if self.stream is not None:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
        except BaseException as exc:
            failure = exc
        finally:
            self.stream = None
        try:
            self.container.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                logger.exception("video container close also failed")
        else:
            self.container = None
        if failure is not None:
            raise failure


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
            timestamp_source="host_monotonic_ns",
            arm_action_semantics=ARM_ACTION_SEMANTICS,
            joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
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
        self.timing_rows = np.empty(batch_size, dtype=TIMING_ROW_DTYPE)
        height, width, _ = rgb_shape
        self.video = _VideoWriter(
            self.path / "rgb.mp4", fps=self.control_hz, width=width, height=height
        )
        self.video.open()

    def append_frame(self, row, command):
        values = encode_raw_row(row, command)
        if "vr_landmarks" in self.datasets:
            values.update({name: row.vr[name.removeprefix("vr_")] for name in VR_DATASET_SPECS})
        rgb, depth = row.camera["rgb"], row.camera["depth"]
        timing = encode_timing_row(row)
        values["depth"] = depth
        for name, array in self.arrays.items():
            array[self.buffered] = values[name]
        self.timing_rows[self.buffered] = timing
        self.video.write_frame(rgb)
        self.buffered += 1
        if self.buffered == len(self.timing_rows):
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
        self.row_times[self.frame_count : end] = self.timing_rows[:count]
        self.frame_count = end

    def finish(self, reason, details, *, recording_truncated_reason=None):
        self.flush()
        if recording_truncated_reason is not None:
            self.meta.attrs["recording_truncated_reason"] = recording_truncated_reason
        # 关闭资源成功与否由 recorder 监督方确认，元数据本身不代表已发布。
        if details:
            self.meta.create_dataset(
                "termination_details", data=json.dumps(details, allow_nan=False)
            )
        self.meta.attrs["termination_reason"] = reason

    def close(self):
        errors = []
        for stage, resource in (("video_close", self.video), ("hdf5_close", self.data_file)):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    errors.append((stage, exc))
        self.arrays = {}
        self.timing_rows = None
        if errors:
            first = errors[0][1]
            for stage, error in errors[1:]:
                logger.error("%s also failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise first


def require_published_raw(path):
    requested = Path(path).expanduser()
    resolved = requested.resolve()
    if any(candidate.name.startswith(".tmp_") for candidate in (requested, resolved)):
        raise ValueError(f"unpublished Raw cannot be exported: {requested}")


class RawReader:
    def __init__(self, episode_path: str | Path):
        import h5py

        requested_path = Path(episode_path).expanduser()
        self.path = requested_path.resolve()
        self.rgb_path = self.path / "rgb.mp4"
        self.video_container = None
        self.dataset_cache: dict[str, h5py.Dataset] = {}
        self.data_file = h5py.File(self.path / "data.h5", "r")
        try:
            self.num_frames = len(self.data_file["timing/rows"])
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
        hz = float(self.meta["control_hz"])
        sampling_clock_ns(hz)
        return hz

    @property
    def fields(self):
        import h5py

        names = {key for key, value in self.data_file.items() if isinstance(value, h5py.Dataset)}
        if self.rgb_path.is_file():
            names.add("rgb")
        return frozenset(names)

    def require_fields(self, *names):
        missing = set(names) - self.fields
        if missing:
            raise ValueError(f"Raw {self.path}: missing fields {sorted(missing)}")

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
            array = self.data_file[name]
            spec = (
                ((), TIMING_ROW_DTYPE) if name == "timing/rows"
                else ROBOT_DATASET_SPECS.get(name, VR_DATASET_SPECS.get(name))
            )
            if spec is not None:
                shape, dtype = spec
                valid_dtype = array.dtype.kind == "f" if dtype.kind == "f" else array.dtype == dtype
                if array.shape != (self.num_frames, *shape) or not valid_dtype:
                    raise ValueError(f"Raw {name}: incompatible shape/dtype")
            self.dataset_cache[name] = array
        return self.dataset_cache[name]

    def read_camera(self) -> CameraSnapshot:
        return decode_camera_metadata(self.meta)

    def validate_joint_metadata(self, device, joint_names) -> None:
        if self.meta["joint_units"] != "rad":
            raise ValueError("Raw joint_units: expected rad")
        name = f"{device}_joint_names"
        if json.loads(self.meta[name]) != list(joint_names):
            raise ValueError(f"Raw {name}: incompatible joint order")

    def validate_tactile_metadata(self) -> None:
        for name, expected in (
            ("hand_tactile_units", TACTILE_UNITS),
            ("hand_tactile_finger_order", TACTILE_FINGER_ORDER),
        ):
            if self.meta[name] != expected:
                raise ValueError(f"Raw {name}: incompatible tactile semantics")

    @property
    def timestamps_s(self):
        stamps = self["timing/rows"].fields("observation_ns")[:]
        # 先减整数时钟，避免长时间运行后的浮点精度损失。
        return (stamps - stamps[0]) / 1e9 if stamps.size else np.empty(0, dtype=np.float64)

    @property
    def dt(self):
        return 1.0 / self.control_hz

    def iter_rgb_frames(self):
        import av

        self.require_fields("rgb")
        if self.video_container is None:
            self.video_container = av.open(str(self.rgb_path), "r")
        self.video_container.seek(0)
        for frame in self.video_container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")

    def close(self):
        self.dataset_cache.clear()
        errors = []
        for attribute, stage in (
            ("video_container", "video_reader_close"), ("data_file", "hdf5_reader_close"),
        ):
            resource = getattr(self, attribute)
            if resource is not None:
                try:
                    resource.close()
                except BaseException as exc:
                    errors.append((stage, exc))
                else:
                    setattr(self, attribute, None)
        if errors:
            first = errors[0][1]
            for stage, error in errors[1:]:
                logger.error("%s also failed", stage, exc_info=(type(error), error, error.__traceback__))
            raise first

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            logger.exception("Raw reader close also failed")
