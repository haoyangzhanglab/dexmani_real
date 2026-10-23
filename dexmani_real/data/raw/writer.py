"""Writer-thread HDF5 and video storage for one unpublished Raw episode."""

import json
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    DATASET_SPECS,
    RAW_FORMAT,
    TIMING_ROW_DTYPE,
    encode_raw_row,
    encode_timing_row,
    unwritten_dispatch_detail,
)
from dexmani_real.data.raw.video import VideoEncoder
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES


class RawEpisodeWriter:
    def __init__(self, path, *, control_hz):
        self.path = Path(path)
        self.control_hz = control_hz
        self.resources = ExitStack()
        self.frame_count = 0
        self.buffered = 0
        self.unwritten_dispatches = []
        self.first_stamp = self.previous_stamp = None

    def open(self, metadata):
        data = self.resources.enter_context(h5py.File(self.path / "data.h5", "w"))
        self.meta = data.create_group("meta")
        self.meta.attrs.update(metadata)
        self.meta.attrs.update(
            control_hz=self.control_hz,
            timestamp_source="host_monotonic_relative",
            arm_action_semantics="q_goal_submitted",
            joint_units="rad",
            arm_joint_names=json.dumps(XARM7_JOINT_NAMES),
            hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
        )
        self.datasets = {
            name: data.create_dataset(
                name,
                shape=(0, *spec.tail_shape),
                maxshape=(None, *spec.tail_shape),
                dtype=spec.dtype,
            )
            for name, spec in DATASET_SPECS.items()
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
        # Copy columns once so buffered depth views do not retain RGB-D snapshots.
        batch_size = 32
        self.arrays = {
            name: np.empty((batch_size, *dataset.shape[1:]), dtype=dataset.dtype)
            for name, dataset in self.datasets.items()
        }
        self.timing_rows = np.empty(batch_size, dtype=TIMING_ROW_DTYPE)
        height, width, _ = self.rgb_shape
        self.video = VideoEncoder(
            self.path / "rgb.mp4", fps=self.control_hz, width=width, height=height
        )
        self.resources.callback(self.video.close)
        self.video.open()

    def append_frame(self, row, command, result):
        stamp = int(row.observation_timestamp_ns)
        if stamp <= 0 or (self.previous_stamp is not None and stamp <= self.previous_stamp):
            raise ValueError("recording observation timestamps must be positive and increasing")
        if self.first_stamp is None:
            self.first_stamp = stamp
        # Integer subtraction retains short intervals even for a large host clock.
        values = encode_raw_row(row, command, result, (stamp - self.first_stamp) / 1e9)
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
        # Never retry a partially written batch: that could duplicate rows.
        count, self.buffered = self.buffered, 0
        end = self.frame_count + count
        try:
            for name, values in self.arrays.items():
                self.datasets[name].resize(end, axis=0)
                self.datasets[name][self.frame_count : end] = values[:count]
            self.row_times.resize(end, axis=0)
            self.row_times[self.frame_count : end] = self.timing_rows[:count]
            self.frame_count = end
        except BaseException:
            for index in range(count):
                detail = unwritten_dispatch_detail(
                    self.timing_rows[index]["observation_ns"],
                    self.arrays["dispatch_status"][index],
                    self.arrays["action_arm_joint_target"][index],
                    self.arrays["action_hand_joint_target"][index],
                )
                if detail is not None:
                    self.unwritten_dispatches.append(detail)
            raise

    def write_termination(
        self, reason, details, *, expected_frames, stop_result, technical_status
    ):
        if stop_result not in {"confirmed", "unconfirmed", "not_applicable"}:
            raise ValueError("invalid stop_result")
        if technical_status not in {"normal", "failed"}:
            raise ValueError("invalid technical_status")
        if stop_result == "unconfirmed":
            technical_status = "failed"
        if (
            any(len(dataset) != self.frame_count for dataset in self.datasets.values())
            or len(self.row_times) != self.frame_count
            or self.video.frame_count != self.frame_count
            or (expected_frames is not None and self.frame_count != expected_frames)
        ):
            raise RuntimeError("recording accepted/written/camera row count mismatch")
        self.meta.attrs.update(
            format=RAW_FORMAT,
            num_frames=self.frame_count,
            termination_reason=reason,
            stop_result=stop_result,
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
            # The recorder retains frame_count after closure, but no longer needs batch storage.
            self.arrays = {}
            self.timing_rows = None
