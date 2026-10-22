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
)
from dexmani_real.data.raw.video import VideoEncoder


class RawEpisodeWriter:
    def __init__(self, path, *, control_hz):
        self.path = Path(path)
        self.control_hz = control_hz
        self.resources = ExitStack()
        self.frame_count = 0
        self.buffered = 0
        self.query_indices = {}
        self.first_stamp = self.previous_stamp = None

    def open(self, metadata, arm_start, arm_control_metadata):
        self.data = self.resources.enter_context(h5py.File(self.path / "data.h5", "w"))
        self.meta = self.data.create_group("meta")
        self.meta.attrs.update(metadata)
        self.meta.attrs.update(
            control_hz=self.control_hz,
            timestamp_source="host_monotonic_relative",
            dispatch_status_source="recorded",
            arm_control_mode=1,
            arm_action_semantics="q_goal_submitted",
        )
        self.datasets = {
            name: self.data.create_dataset(
                name,
                shape=(0, *spec.tail_shape),
                maxshape=(None, *spec.tail_shape),
                dtype=spec.dtype,
            )
            for name, spec in DATASET_SPECS.items()
        }
        self.execution = self.data.create_group("arm_execution")
        self.execution.attrs.update(arm_control_metadata)
        self.execution.create_dataset("start", data=json.dumps(arm_start, allow_nan=False))

        self.timing = self.data.create_group("timing")
        self.timing.attrs.update(
            clock="host_monotonic_ns",
            missing_value=-1,
            camera_timestamp_source="oldest_channel_first_host_receive_monotonic",
        )
        self.row_times = self.timing.create_dataset(
            "rows", shape=(0,), maxshape=(None,), dtype=TIMING_ROW_DTYPE
        )
        self.queries = self.timing.create_dataset(
            "queries", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype()
        )
        self.rgb_shape = (metadata["camera_color_height"], metadata["camera_color_width"], 3)
        self.datasets["depth"] = self.data.create_dataset(
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

    def append_frame(self, row, command, result, execution_timing):
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
        timing = encode_timing_row(row, result, execution_timing)
        for name, value in values.items():
            self.arrays[name][self.buffered] = value
        self.arrays["depth"][self.buffered] = depth
        self.timing_rows[self.buffered] = timing
        self.video.write_frame(rgb)
        self.buffered += 1
        self.previous_stamp = stamp
        if self.buffered == len(self.timing_rows):
            self.flush()

    def write_query(self, values):
        key = (values["run_id"], values["query_id"])
        index = self.query_indices.get(key)
        if index is None:
            index = len(self.queries)
            self.query_indices[key] = index
            self.queries.resize(index + 1, axis=0)
        self.queries[index] = json.dumps(values, allow_nan=False)

    def flush(self):
        if not self.buffered:
            return
        # Never retry a partially written batch: that could duplicate rows.
        count, self.buffered = self.buffered, 0
        end = self.frame_count + count
        for name, values in self.arrays.items():
            self.datasets[name].resize(end, axis=0)
            self.datasets[name][self.frame_count : end] = values[:count]
        self.row_times.resize(end, axis=0)
        self.row_times[self.frame_count : end] = self.timing_rows[:count]
        self.frame_count = end

    def write_execution_end(self, arm_execution):
        if arm_execution is not None:
            self.execution.create_dataset("end", data=json.dumps(arm_execution, allow_nan=False))

    def write_termination(self, reason, details, timing, *, expected_frames):
        self.meta.attrs.update(
            format=RAW_FORMAT, num_frames=self.frame_count, termination_reason=reason
        )
        if timing is not None:
            self.timing.create_dataset("termination", data=json.dumps(timing, allow_nan=False))
        if len(self.row_times) != self.frame_count:
            raise RuntimeError("timing/core row count mismatch")
        if details:
            self.meta.create_dataset(
                "termination_details", data=json.dumps(details, allow_nan=False)
            )
        if expected_frames is not None and (
            self.frame_count != expected_frames or self.video.frame_count != self.frame_count
        ):
            raise RuntimeError("recording accepted/written/camera row count mismatch")

    def close(self):
        try:
            self.resources.close()
        finally:
            # The recorder retains frame_count after closure, but no longer needs batch storage.
            self.arrays = {}
            self.timing_rows = None
            self.query_indices.clear()
