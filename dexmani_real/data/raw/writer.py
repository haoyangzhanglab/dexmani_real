import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    ARM_ACTION_SEMANTICS,
    DATASET_SPECS,
    TIMING_ROW_DTYPE,
    encode_raw_row,
    encode_timing_row,
)
from dexmani_real.data.raw.video import VideoWriter
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


class EpisodeWriter:
    def __init__(self, path, *, control_hz):
        self.path = Path(path)
        self.control_hz = control_hz
        self.resources = ExitStack()
        self.frame_count = 0
        self.buffered = 0
        self.first_stamp = self.previous_stamp = None

    def open(self, metadata):
        data = self.resources.enter_context(h5py.File(self.path / "data.h5", "w"))
        self.meta = data.create_group("meta")
        self.meta.attrs.update(metadata)
        self.meta.attrs.update(
            technical_status="failed",
            termination_reason="recording_incomplete",
            control_hz=self.control_hz,
            timestamp_source="host_monotonic_relative",
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
        batch_size = 32
        self.arrays = {
            name: np.empty((batch_size, *dataset.shape[1:]), dtype=dataset.dtype)
            for name, dataset in self.datasets.items()
        }
        self.timing_rows = np.empty(batch_size, dtype=TIMING_ROW_DTYPE)
        height, width, _ = self.rgb_shape
        self.video = VideoWriter(
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
        if self.first_stamp is None:
            self.first_stamp = stamp
        # 先减整数时钟，避免长时间运行后的浮点精度损失。
        values = encode_raw_row(row, command, (stamp - self.first_stamp) / 1e9)
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

    def write_termination(self, reason, details, *, expected_frames, technical_status):
        if technical_status not in {"normal", "failed"}:
            raise ValueError("invalid technical_status")
        if (
            any(len(dataset) != self.frame_count for dataset in self.datasets.values())
            or len(self.row_times) != self.frame_count
            or self.video.frame_count != self.frame_count
            or (expected_frames is not None and self.frame_count != expected_frames)
        ):
            raise RuntimeError("recording accepted/written/camera row count mismatch")
        self.meta.attrs.update(
            termination_reason=reason,
            technical_status=technical_status,
        )
        if details:
            self.meta.create_dataset(
                "termination_details", data=json.dumps(details, allow_nan=False)
            )

    def close(self):
        try:
            self.resources.close()
        finally:
            self.arrays = {}
            self.timing_rows = None
