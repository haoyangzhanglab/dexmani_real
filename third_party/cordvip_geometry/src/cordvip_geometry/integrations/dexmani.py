"""Lazy Raw/RealSense adapters; consume and close live generators on the source owner thread."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..rgbd import Frame


def raw_frame_identity(episode, row_index, timing):
    """使用 Raw 纳秒时钟；缺失相机时钟时明确标记为观测时钟。"""
    source = f"raw:{episode}:row:{row_index}"
    stamp = int(timing["observation_ns"])
    clock = "host_monotonic_observation"
    if int(timing["camera_sequence"]) >= 0:
        source = f"raw:{episode}:camera:{int(timing['camera_sequence'])}"
    if int(timing["camera_ns"]) > 0:
        stamp, clock = int(timing["camera_ns"]), "host_monotonic_oldest_rgbd_receive"
    return source, stamp, clock


def iter_raw_episode(path: str | Path) -> Iterator[Frame]:
    """Read Raw rows using decoded MP4 RGB; exhausting the iterator verifies the full frame count."""
    from dexmani_real.data.raw import RawReader

    episode_path = Path(path).expanduser().resolve()
    with RawReader(episode_path) as reader:
        reader.require_fields("rgb", "depth")
        camera = reader.read_camera()
        intrinsics = camera.color
        K = intrinsics.matrix()
        expected_shape = (intrinsics.height, intrinsics.width)
        depth = reader["depth"]
        if depth.shape != (reader.num_frames, *expected_shape) or depth.dtype != np.uint16:
            raise ValueError("Raw depth must be uint16 [rows,H,W] matching color intrinsics")
        timing = reader.read_timing_rows(("observation_ns", "camera_sequence", "camera_ns"))
        observation = timing["observation_ns"]
        if np.any(observation <= 0) or np.any(np.diff(observation) <= 0):
            raise ValueError("Raw observation clocks must be positive and strictly increasing")
        images = iter(reader.iter_rgb_frames())
        for row_index in range(reader.num_frames):
            rgb = next(images, None)
            if rgb is None:
                raise ValueError(f"Raw RGB ended before row {row_index}")
            if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Raw RGB row {row_index} does not match color intrinsics")
            source_id, timestamp_ns, clock = raw_frame_identity(
                episode_path,
                row_index,
                timing[row_index],
            )
            yield Frame(
                rgb=rgb,
                depth_m=depth[row_index].astype(np.float32) * np.float32(camera.depth_scale_m),
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
                source_id=source_id,
                timestamp_ns=timestamp_ns,
                clock=clock,
                row_index=row_index,
            )
        if next(images, None) is not None:
            raise ValueError("Raw RGB contains more frames than the HDF5 rows")


def iter_live_realsense(
    *, serial: str | None = None, width: int = 640, height: int = 480, fps: int = 30
) -> Iterator[Frame]:
    """Preserve first host queue-return times per channel; SDK queue latency is unmeasured."""
    from dexmani_real.config.sensor import CameraConfig
    from dexmani_real.sensor.realsense import RealSenseCamera

    config = CameraConfig(serial=serial, width=width, height=height, fps=fps)
    config.validate()
    camera = RealSenseCamera(config, auto_exposure_priority=None)
    try:
        camera.connect()
        depth_scale_m = np.float32(camera.depth_scale_m)
        active_serial = camera.active_serial
        intrinsics = camera.color_intrinsics
        K = intrinsics.matrix()
        while True:
            raw = camera.read()
            if raw is None:
                continue
            identity = (raw.depth_frame_number, raw.color_frame_number)
            yield Frame(
                rgb=raw.rgb,
                depth_m=raw.depth.astype(np.float32) * depth_scale_m,
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
                source_id=f"realsense:{active_serial}:{identity[0]}:{identity[1]}",
                timestamp_ns=raw.timestamp_ns,
                clock="host_monotonic_oldest_rgbd_receive",
            )
    finally:
        camera.disconnect()


class RawHandCloudProvider:
    """使用同一 Raw 行实测关节、记录外参和完整组合 URDF；行对齐不证明物理同步。"""

    def __init__(self, episode, num_points=1024, seed=0):
        import xml.etree.ElementTree as ET

        from dexmani_real.data.raw import RawReader
        from dexmani_real.robot.model import (
            XARM7_JOINT_NAMES,
            XARM7_XHAND_RIGHT_URDF_PATH,
            XHAND_FINGERTIP_LINK_NAMES,
            XHAND_SDK_JOINT_NAMES,
        )

        from ..hand_cloud import HandCloudModel

        self.episode = Path(episode).expanduser().resolve()
        with RawReader(self.episode) as reader:
            reader.require_fields("arm_qpos", "hand_qpos")
            self.arm_qpos = reader["arm_qpos"][:]
            self.hand_qpos = reader["hand_qpos"][:]
            self.T_base_camera = reader.read_camera().T_xarm_base_from_color
            if self.T_base_camera is None:
                raise ValueError("hand cloud requires recorded camera extrinsics")
            self.timing = reader.read_timing_rows(
                ("observation_ns", "arm_ns", "hand_ns", "camera_ns", "camera_sequence")
            )
            raw_notes = {
                key: str(reader.meta[key])
                for key in ("notes", "timestamp_source", "termination_reason")
                if key in reader.meta
            }
        self.T_camera_base = np.linalg.inv(self.T_base_camera)
        excluded = ("right_hand_ee_link", *XHAND_FINGERTIP_LINK_NAMES)
        excluded_links = [
            link.attrib["name"]
            for link in ET.parse(XARM7_XHAND_RIGHT_URDF_PATH).getroot().findall("link")
            if not link.attrib["name"].startswith("right_hand_") or link.attrib["name"] in excluded
        ]
        self.hand_model = HandCloudModel(
            XARM7_XHAND_RIGHT_URDF_PATH,
            (*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES),
            excluded_links=excluded_links,
            num_points=num_points,
            seed=seed,
        )
        self.metadata = {
            **self.hand_model.metadata,
            "episode": str(self.episode),
            "joint_source": str(self.episode / "data.h5"),
            "joint_state_semantics": "measured_arm_qpos_then_measured_hand_qpos_no_interpolation",
            "T_base_camera": self.T_base_camera.tolist(),
            "camera_extrinsics_source": "Raw meta.camera_T_xarm_base_from_color",
            "timing_semantics": "host_monotonic_source_times_if_present_no_exposure_time_claim",
            "missing_timing": "row_aligned_geometry_only_not_verified_for_policy",
            "raw_notes": raw_notes,
        }

    def for_frame(self, frame):
        """Reject mismatched rows and missing joints; never substitute commands."""
        row = frame.row_index
        if (
            isinstance(row, bool)
            or not isinstance(row, (int, np.integer))
            or not 0 <= row < len(self.arm_qpos)
        ):
            raise ValueError("hand cloud requires a valid Raw row_index")
        timing = self.timing[row]
        expected_source, expected_stamp, expected_clock = raw_frame_identity(
            self.episode,
            row,
            timing,
        )
        arm_ns, hand_ns, camera_ns = (
            int(timing[name]) for name in ("arm_ns", "hand_ns", "camera_ns")
        )
        if (
            frame.source_id != expected_source
            or frame.timestamp_ns != expected_stamp
            or frame.clock != expected_clock
        ):
            raise ValueError(
                "hand cloud and RGB-D must come from the same Raw row and source clock"
            )
        joint_state = np.concatenate((self.arm_qpos[row], self.hand_qpos[row]))
        cloud = self.hand_model.transform(joint_state, self.T_camera_base)
        timing_available = (
            min(arm_ns, hand_ns, camera_ns) > 0
            and expected_clock == "host_monotonic_oldest_rgbd_receive"
        )
        skew_ms = (
            (max(arm_ns, hand_ns, camera_ns) - min(arm_ns, hand_ns, camera_ns)) / 1e6
            if timing_available
            else None
        )
        return {
            **cloud,
            "joint_state": joint_state,
            "T_base_camera": self.T_base_camera,
            "source_id": frame.source_id,
            "row_index": int(row),
            "arm_timestamp_ns": arm_ns,
            "hand_timestamp_ns": hand_ns,
            "camera_timestamp_ns": camera_ns,
            "clock": expected_clock,
            "timing_available": bool(timing_available),
            "skew_ms": skew_ms,
        }
