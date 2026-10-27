from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import av
import h5py
import numpy as np

from dexmani_real.data.raw.schema import (
    ARM_ACTION_SEMANTICS,
    DATASET_SPECS,
    TACTILE_AXES,
    TACTILE_FINGER_ORDER,
    TACTILE_PROCESSING,
    TACTILE_UNITS,
    TIMING_ROW_DTYPE,
)
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.utils.timing import sampling_clock_ns


class EpisodeReader:
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
                ("timestamp_source", "host_monotonic_relative"),
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
            self.num_frames = len(self.data_file["timestamp"])
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
        meta = self.meta
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
        """分块检查采样与来源时间，保留一行重叠以核对块边界。"""
        chunk_frames = 4096
        timing = self["timing/rows"]
        fields = ("observation_ns", "control_ns", "arm_ns", "hand_ns")
        fields += tuple(f"tactile_{name}_ns" for name in required_tactile)
        if require_camera:
            fields += ("camera_ns", "camera_sequence")
        max_age_ns = self.validate_tactile_metadata() * 1e9 if required_tactile else None
        timestamps = self["timestamp"]
        dt_ns, tolerance_ns = sampling_clock_ns(self.control_hz)
        first_observation = None
        for start in range(0, self.num_frames, chunk_frames):
            first, end = max(0, start - 1), min(start + chunk_frames, self.num_frames)
            timestamp = timestamps[first:end]
            if not np.isfinite(timestamp).all() or np.any(timestamp < 0):
                raise ValueError("timestamp must contain finite nonnegative relative seconds")
            if first == 0 and timestamp[0] != 0:
                raise ValueError("timestamp must start at zero")
            if np.any(np.diff(timestamp) <= 0):
                raise ValueError("timestamp must be strictly increasing within an episode")
            stamps = np.rint(timestamp * 1e9)
            intervals = np.diff(stamps)
            deviations = stamps - np.arange(first, end) * dt_ns
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
            rows = timing.fields(fields)[first:end]
            observation = rows["observation_ns"]
            if np.any(observation <= 0) or np.any(np.diff(observation) <= 0):
                raise ValueError("observation clocks must be positive and strictly increasing")
            if first_observation is None:
                first_observation = int(observation[0])
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
            if not np.allclose(timestamp * 1e9, observation - first_observation, rtol=0, atol=1.0):
                raise ValueError("timestamp differs from actual observation clock")
            for name in required_tactile:
                if np.any(observation - rows[f"tactile_{name}_ns"] >= max_age_ns):
                    raise ValueError(f"consumed tactile {name} exceeds hold age budget")

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
