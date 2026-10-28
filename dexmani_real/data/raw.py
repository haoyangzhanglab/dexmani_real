"""Raw 格式、元数据与 RGB-D 读写。"""

from __future__ import annotations

import json
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np

from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.utils.timing import sampling_clock_ns

ARM_ACTION_SEMANTICS = "q_goal_submission_attempt"
TACTILE_PROCESSING = "sdk_native_bounded_previous_valid_vector_hold"
TACTILE_UNITS = "sdk_native_unknown_physical_scale"
TACTILE_AXES = "sdk_sensor_axes_not_world_frame"
TACTILE_FINGER_ORDER = "thumb,index,middle,ring,pinky"


DATASET_SPECS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    "arm_qpos": ((7,), np.dtype(np.float64)),
    "hand_qpos": ((12,), np.dtype(np.float64)),
    # 触觉保留 SDK 单位和限时保持语义。
    "hand_contact": ((5, 3), np.dtype(np.float32)),
    "hand_tactile_force": ((5, 120, 3), np.dtype(np.float32)),
    # 记录最终提交的绝对目标，包括失败的提交尝试。
    "action_arm_joint_target": ((7,), np.dtype(np.float64)),
    "action_hand_joint_target": ((12,), np.dtype(np.float64)),
}


# 主机单调时钟；缺失用 -1，触觉保留最早有效向量的来源时间。
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


def encode_raw_row(row, command):
    arm, hand = (
        row.arm[0],
        None if row.hand is None else row.hand[0],
    )
    values = {
        "arm_qpos": arm["qpos"],
        "hand_qpos": hand["qpos"] if hand is not None else np.full(12, np.nan),
        "hand_contact": hand["tactile_aggregate"] if hand is not None else np.full((5, 3), np.nan),
        "hand_tactile_force": (
            hand["tactile_dense"] if hand is not None else np.full((5, 120, 3), np.nan)
        ),
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
    return values


def encode_timing_row(row):
    hand = row.hand
    return (
        int(row.observation_timestamp_ns),
        int(row.control_timestamp_ns),
        int(row.arm["timestamp_ns"][0]),
        int(hand["timestamp_ns"][0]) if hand is not None else -1,
        int(hand["tactile_aggregate_timestamp_ns"][0]) if hand is not None else -1,
        int(hand["tactile_dense_timestamp_ns"][0]) if hand is not None else -1,
        int(row.camera["timestamp_ns"]),
        int(row.camera["ring_sequence"]),
    )


def camera_metadata(camera: CameraSnapshot):
    color = camera.color
    values = {
        "camera_serial": camera.serial,
        "camera_payload_mode": "depth_to_color_aligned_rgbd",
        "camera_color_width": color.width,
        "camera_color_height": color.height,
        "camera_color_intrinsics": color.matrix().reshape(-1).tolist(),
        "camera_color_distortion_model": color.distortion_model,
        "camera_color_distortion_coeffs": list(color.distortion_coeffs),
        "depth_scale": camera.depth_scale_m,
    }
    if camera.T_xarm_base_from_color is not None:
        values["camera_T_xarm_base_from_color"] = camera.T_xarm_base_from_color.reshape(-1).tolist()
    return values


def decode_camera_metadata(meta) -> CameraSnapshot:
    if meta["camera_payload_mode"] != "depth_to_color_aligned_rgbd":
        raise ValueError("Raw camera payload must be depth-to-color aligned RGB-D")
    matrix = np.asarray(meta["camera_color_intrinsics"], dtype=np.float64).reshape(3, 3)
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
    if not np.allclose(matrix, color.matrix(), rtol=0, atol=1e-9):
        raise ValueError("camera_color_intrinsics must be a pinhole matrix")
    transform = None
    if "camera_T_xarm_base_from_color" in meta:
        transform = validate_rigid_transform(
            np.asarray(meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
            label="camera_T_xarm_base_from_color",
        )
    return CameraSnapshot(meta["camera_serial"], color, float(meta["depth_scale"]), transform)


def snapshot_recording_metadata(camera_snapshot, *, tactile_reset_metadata, tactile_hold_max_age_s):
    """复用初始化相机快照，补入当前 capture 消费的触觉处理语义。"""
    metadata = camera_metadata(camera_snapshot)
    metadata.update(tactile_reset_metadata)
    metadata.update(
        hand_tactile_processing=TACTILE_PROCESSING,
        hand_tactile_units=TACTILE_UNITS,
        hand_tactile_axes=TACTILE_AXES,
        hand_tactile_finger_order=TACTILE_FINGER_ORDER,
        hand_tactile_hold_max_age_s=float(tactile_hold_max_age_s),
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
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24").reformat(format="yuv444p")
        av_frame.pts = self.frame_count
        av_frame.time_base = 1 / self.rate
        for packet in self.stream.encode(av_frame):
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.container is None:
            return
        try:
            if self.stream is not None:
                for packet in self.stream.encode(None):
                    self.container.mux(packet)
        finally:
            # 编码失败也关闭文件；关闭失败时保留句柄。
            self.stream = None
            self.container.close()
            self.container = None


class RawWriter:
    def __init__(self, path, *, control_hz):
        self.path = Path(path)
        self.control_hz = control_hz
        self.resources = ExitStack()
        self.frame_count = 0
        self.buffered = 0
        self.previous_stamp = None

    def open(self, metadata):
        data = self.resources.enter_context(h5py.File(self.path / "data.h5", "w"))
        self.meta = data.create_group("meta")
        self.meta.attrs.update(metadata)
        self.meta.attrs.update(
            technical_status="failed",
            termination_reason="recording_incomplete",
            control_hz=self.control_hz,
            timestamp_source="host_monotonic_ns",
            arm_action_semantics=ARM_ACTION_SEMANTICS,
            joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
        )
        self.datasets = {
            name: data.create_dataset(
                name,
                shape=(0, *tail_shape),
                maxshape=(None, *tail_shape),
                dtype=dtype,
            )
            for name, (tail_shape, dtype) in DATASET_SPECS.items()
        }
        timing = data.create_group("timing")
        timing.attrs.update(
            clock="host_monotonic_ns",
            missing_value=-1,
            camera_timestamp_source="oldest_channel_first_host_receive_monotonic",
        )
        self.row_times = timing.create_dataset(
            "rows", shape=(0,), maxshape=(None,), dtype=TIMING_ROW_DTYPE
        )
        self.rgb_shape = (metadata["camera_color_height"], metadata["camera_color_width"], 3)
        self.datasets["depth"] = data.create_dataset(
            "depth",
            shape=(0, *self.rgb_shape[:2]),
            maxshape=(None, *self.rgb_shape[:2]),
            chunks=(1, *self.rgb_shape[:2]),
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
        height, width, _ = self.rgb_shape
        self.video = _VideoWriter(
            self.path / "rgb.mp4", fps=self.control_hz, width=width, height=height
        )
        self.resources.callback(self.video.close)
        self.video.open()

    def append_frame(self, row, command):
        if row.camera is None:
            raise ValueError("recording requires RGB-D from the current observation")
        stamp = int(row.observation_timestamp_ns)
        if stamp <= 0 or (self.previous_stamp is not None and stamp <= self.previous_stamp):
            raise ValueError("recording observation timestamps must be positive and increasing")
        values = encode_raw_row(row, command)
        rgb, depth = row.camera["rgb"], row.camera["depth"]
        if rgb.shape != self.rgb_shape or rgb.dtype != np.uint8:
            raise ValueError("RGB shape or dtype mismatch")
        if depth.shape != self.rgb_shape[:2] or depth.dtype != np.uint16:
            raise ValueError("depth shape or dtype mismatch")
        timing = encode_timing_row(row)
        for name, value in values.items():
            self.arrays[name][self.buffered] = value
        self.arrays["depth"][self.buffered] = depth
        self.timing_rows[self.buffered] = timing
        self.video.write_frame(rgb)
        self.buffered += 1
        self.previous_stamp = stamp
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

    def finish(self, reason, details, *, expected_frames, technical_status):
        if technical_status not in {"normal", "failed"}:
            raise ValueError("invalid technical_status")
        self.flush()
        if (
            any(len(dataset) != self.frame_count for dataset in self.datasets.values())
            or len(self.row_times) != self.frame_count
            or self.video.frame_count != self.frame_count
            or self.frame_count != expected_frames
        ):
            raise RuntimeError("recording accepted/written/camera row count mismatch")
        # 视频完成编码后才标记技术状态。
        self.video.close()
        if details:
            self.meta.create_dataset(
                "termination_details", data=json.dumps(details, allow_nan=False)
            )
        self.meta.attrs.update(
            termination_reason=reason,
            technical_status=technical_status,
        )

    def close(self):
        try:
            self.resources.close()
        finally:
            self.arrays = {}
            self.timing_rows = None


class RawReader:
    def __init__(self, episode_path: str | Path):
        self.path = Path(episode_path)
        self.rgb_path = self.path / "rgb.mp4"
        self.video_container = None
        self.dataset_cache: dict[str, h5py.Dataset] = {}
        self.tactile_hold_max_age_s = None
        with ExitStack() as resources:
            self.data_file = resources.enter_context(h5py.File(self.path / "data.h5", "r"))
            meta = self.meta
            for name, expected in (
                ("arm_action_semantics", ARM_ACTION_SEMANTICS),
                ("timestamp_source", "host_monotonic_ns"),
                ("joint_units", "rad"),
            ):
                if meta[name] != expected:
                    raise ValueError(f"Raw {name}: expected {expected}")
            for name, joint_names in (
                ("arm_joint_names", XARM7_JOINT_NAMES),
                ("hand_joint_names", XHAND_SDK_JOINT_NAMES),
            ):
                if json.loads(meta[name]) != list(joint_names):
                    raise ValueError(f"Raw {name}: incompatible joint order")
            self.num_frames = len(self.data_file["timing/rows"])
            self.control_hz = float(meta["control_hz"])
            sampling_clock_ns(self.control_hz)
            self.resources = resources.pop_all()

    @property
    def meta(self):
        return self.data_file["meta"].attrs

    @property
    def fields(self):
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
            array = self.data_file[name]
            spec = ((), TIMING_ROW_DTYPE) if name == "timing/rows" else DATASET_SPECS.get(name)
            if spec is not None:
                shape, dtype = spec
                if array.shape != (self.num_frames, *shape) or array.dtype != dtype:
                    raise ValueError(f"Raw {name}: incompatible shape/dtype")
            self.dataset_cache[name] = array
        return self.dataset_cache[name]

    def read_camera(self) -> CameraSnapshot:
        return decode_camera_metadata(self.meta)

    def validate_completion(self):
        if self.meta["technical_status"] != "normal":
            raise ValueError("capture ended with a technical failure")

    def validate_tactile_metadata(self):
        if self.tactile_hold_max_age_s is None:
            for name, expected in (
                ("hand_tactile_processing", TACTILE_PROCESSING),
                ("hand_tactile_units", TACTILE_UNITS),
                ("hand_tactile_axes", TACTILE_AXES),
                ("hand_tactile_finger_order", TACTILE_FINGER_ORDER),
            ):
                if self.meta[name] != expected:
                    raise ValueError(f"Raw {name}: incompatible tactile semantics")
            age = float(self.meta["hand_tactile_hold_max_age_s"])
            if not np.isfinite(age) or age <= 0:
                raise ValueError("hand_tactile_hold_max_age_s must be finite and positive")
            self.tactile_hold_max_age_s = age
        return self.tactile_hold_max_age_s

    def read_timing_rows(self, fields):
        return self["timing/rows"].fields(tuple(fields))[:]

    def validate_rows(self, *, required_tactile=("aggregate", "dense"), require_camera=True):
        """物理回放的严格整段准入；训练导出另行选择连续前缀。"""
        chunk_frames = 4096
        timing = self["timing/rows"]
        fields = ("observation_ns", "control_ns", "arm_ns", "hand_ns")
        fields += tuple(f"tactile_{name}_ns" for name in required_tactile)
        if require_camera:
            fields += ("camera_ns", "camera_sequence")
        max_age_ns = self.validate_tactile_metadata() * 1e9 if required_tactile else None
        dt_ns, tolerance_ns = sampling_clock_ns(self.control_hz)
        first_observation = None
        for start in range(0, self.num_frames, chunk_frames):
            first, end = max(0, start - 1), min(start + chunk_frames, self.num_frames)
            rows = timing.fields(fields)[first:end]
            observation = rows["observation_ns"]
            if np.any(observation <= 0) or np.any(np.diff(observation) <= 0):
                raise ValueError("observation clocks must be positive and strictly increasing")
            if first_observation is None:
                first_observation = int(observation[0])
            intervals = np.diff(observation)
            deviations = observation - first_observation - np.arange(first, end) * dt_ns
            bad = np.abs(deviations) > tolerance_ns
            bad[1:] |= np.abs(intervals - dt_ns) > tolerance_ns
            if bad.any():
                row = int(np.flatnonzero(bad)[0])
                interval = intervals[row - 1] if row else float("nan")
                raise ValueError(
                    f"timing row {row + first}: interval_ns={interval:.0f}, "
                    f"nominal_ns={dt_ns}, grid_deviation_ns={deviations[row]:.0f}, "
                    f"tolerance_ns={tolerance_ns}"
                )
            for name in fields[1:]:
                values = rows[name]
                if name == "camera_sequence":
                    if np.any(values < 0) or np.any(np.diff(values) < 0):
                        raise ValueError("camera sequence must be nonnegative and nondecreasing")
                    continue
                if np.any(values <= 0) or np.any(values > observation):
                    raise ValueError(f"invalid consumed source clock: {name}")
                if np.any(np.diff(values) < 0):
                    raise ValueError(f"consumed source clock regressed: {name}")
            for name in required_tactile:
                if np.any(observation - rows[f"tactile_{name}_ns"] >= max_age_ns):
                    raise ValueError(f"consumed tactile {name} exceeds hold age budget")

    @property
    def timestamps_s(self):
        stamps = self["timing/rows"].fields("observation_ns")[:]
        # 先减整数时钟，避免长时间运行后的浮点精度损失。
        return (stamps - stamps[0]) / 1e9 if stamps.size else np.empty(0, dtype=np.float64)

    @property
    def dt(self):
        return 1.0 / self.control_hz

    def iter_rgb_frames(self):
        self.require_fields("rgb")
        if self.video_container is None:
            self.video_container = self.resources.enter_context(av.open(str(self.rgb_path), "r"))
        self.video_container.seek(0)
        for frame in self.video_container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")

    def close(self):
        self.dataset_cache.clear()
        self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
