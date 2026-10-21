"""Lazy Raw/RealSense adapters; consume and close live generators on the source owner thread."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import time

import numpy as np

from ..rgbd import Frame


def raw_frame_identity(episode, row_index, observation_s, timing):
    """One construction of the existing Raw source string and consumed camera clock."""
    source = f"raw:{episode}:row:{row_index}"
    stamp = int(round(float(observation_s) * 1e9))
    clock = "raw_observation_relative"
    if int(timing["camera_sequence"]) > 0:
        source = f"raw:{episode}:camera:{int(timing['camera_sequence'])}"
    if int(timing["camera_ns"]) > 0:
        stamp, clock = int(timing["camera_ns"]), "host_monotonic_oldest_rgbd_receive"
    return source, stamp, clock


def iter_raw_episode(path: str | Path) -> Iterator[Frame]:
    """Read Raw rows using decoded MP4 RGB; exhausting the iterator verifies the full frame count."""
    from dexmani_real.dataset.pointcloud import load_raw_episode_camera_model
    from dexmani_real.recording.storage.reader import EpisodeReader

    episode_path = Path(path).expanduser().resolve()
    with EpisodeReader(episode_path) as reader:
        reader.require_fields("rgb", "depth", "timestamp")
        camera = load_raw_episode_camera_model(reader)
        intrinsics = camera.intrinsics
        K = intrinsics.matrix()
        expected_shape = (intrinsics.height, intrinsics.width)
        depth = reader["depth"]
        if depth.shape != (reader.num_frames, *expected_shape) or depth.dtype != np.uint16:
            raise ValueError("Raw depth must be uint16 [rows,H,W] matching color intrinsics")
        observation_s = np.asarray(reader["timestamp"][:], dtype=np.float64)
        if (
            not np.isfinite(observation_s).all()
            or np.any(observation_s < 0)
            or np.any(np.diff(observation_s) <= 0)
        ):
            raise ValueError("Raw observation timestamps must be finite and strictly increasing")
        timing = reader.read_timing_rows(("camera_sequence", "camera_ns"))
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
                observation_s[row_index],
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
    from dexmani_real.sensor.camera.config import CameraParams
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

    config = CameraParams(serial=serial, width=width, height=height, fps=fps)
    camera = RealSenseCamera(config, auto_exposure_priority=None)
    previous_identity: tuple[int, int] | None = None
    channel_received_ns = [0, 0]
    try:
        camera.connect()
        depth_scale_m = np.float32(camera.get_depth_scale())
        if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
            raise RuntimeError("RealSense depth scale must be finite and positive")
        active_serial = camera.active_serial
        if not isinstance(active_serial, str) or not active_serial.strip():
            raise RuntimeError("RealSense active serial is unavailable")
        intrinsics = camera.get_color_intrinsics()
        K = intrinsics.matrix()
        stall_timeout_ns = int(config.source_stall_timeout_s * 1e9)
        while True:
            raw = camera.read(timeout_ms=1000)
            identity = (raw.depth_frame_number, raw.color_frame_number)
            for channel, number in enumerate(identity):
                if number < 0 or (
                    previous_identity is not None and number < previous_identity[channel]
                ):
                    raise RuntimeError("RealSense frame identity is invalid or moved backward")
                if previous_identity is None or number != previous_identity[channel]:
                    channel_received_ns[channel] = raw.timestamp_ns
            oldest_received_ns = min(channel_received_ns)
            if time.monotonic_ns() - oldest_received_ns >= stall_timeout_ns:
                raise RuntimeError("RealSense stopped producing new RGB or depth")
            if identity == previous_identity:
                continue
            previous_identity = identity
            yield Frame(
                rgb=raw.rgb,
                depth_m=raw.depth_aligned_to_color_raw.astype(np.float32) * depth_scale_m,
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
                source_id=f"realsense:{active_serial}:{identity[0]}:{identity[1]}",
                timestamp_ns=oldest_received_ns,
                clock="host_monotonic_oldest_rgbd_receive",
            )
    finally:
        camera.disconnect()


class RawHandCloudProvider:
    """Use exact-row measured joints, recorded camera extrinsics and the current hand mount.
    Missing camera clocks retain row/relative identities; row alignment does not establish sync.
    """

    def __init__(self, episode, config_path=None, num_points=1024, seed=0):
        from dexmani_real.config.experiment import load_experiment_config
        from dexmani_real.dataset.pointcloud import load_raw_episode_base_from_color
        from dexmani_real.planning.kinematics.arm_fk import ArmFK
        from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rotmat
        from dexmani_real.recording.storage.reader import EpisodeReader
        from dexmani_real.robot.model import (
            XARM7_JOINT_NAMES,
            XARM7_XHAND_COLLISION_URDF_PATH,
            XHAND_FINGERTIP_LINK_NAMES,
            XHAND_RIGHT_URDF_PATH,
            XHAND_SDK_JOINT_NAMES,
        )
        from ..hand_cloud import HandCloudModel
        from ..io import sha256

        self.episode = Path(episode).expanduser().resolve()
        config_path = Path(config_path).expanduser().resolve() if config_path is not None else None
        config = load_experiment_config(yaml_path=config_path)
        from dexmani_real.utils.geometry import validate_unit_quaternion_wxyz

        mount = np.asarray(config.hand.T_eef_handbase_pos_xyz, dtype=np.float64)
        if mount.shape != (3,) or not np.isfinite(mount).all():
            raise ValueError("hand mount must be finite shape (3,)")
        validate_unit_quaternion_wxyz(config.hand.T_eef_handbase_quat_wxyz, name="hand mount")
        self._T_eef_handbase = np.eye(4)
        self._T_eef_handbase[:3, 3] = mount
        self._T_eef_handbase[:3, :3] = quat_wxyz_to_rotmat(config.hand.T_eef_handbase_quat_wxyz)
        with EpisodeReader(self.episode) as reader:
            reader.require_fields("arm_qpos", "hand_qpos", "timestamp")
            self._arm = reader["arm_qpos"][:]
            self._hand = reader["hand_qpos"][:]
            self._observation_s = reader["timestamp"][:]
            self.T_base_camera = load_raw_episode_base_from_color(reader)
            self._timing = reader.read_timing_rows(
                ("arm_ns", "hand_ns", "camera_ns", "camera_sequence")
            )
            raw_notes = {
                key: str(reader.meta[key])
                for key in ("notes", "timestamp_source", "termination_reason")
                if key in reader.meta
            }
        self._T_camera_base = np.linalg.inv(self.T_base_camera)
        self._arm_fk = ArmFK(str(XARM7_XHAND_COLLISION_URDF_PATH))
        self.hand_model = HandCloudModel(
            XHAND_RIGHT_URDF_PATH,
            XHAND_SDK_JOINT_NAMES,
            excluded_links=("right_hand_ee_link", *XHAND_FINGERTIP_LINK_NAMES),
            num_points=num_points,
            seed=seed,
        )
        # Hash consumed joints/times without rereading unrelated RGB-D payloads at startup.
        import hashlib

        state_hash = hashlib.sha256()
        for values in (self._arm, self._hand, self._observation_s, self._timing):
            state_hash.update(values.tobytes())
        self.metadata = {
            **self.hand_model.metadata,
            "episode": str(self.episode),
            "joint_source": str(self.episode / "data.h5"),
            "joint_and_timing_sha256": state_hash.hexdigest(),
            "arm_urdf": str(XARM7_XHAND_COLLISION_URDF_PATH),
            "arm_urdf_sha256": sha256(XARM7_XHAND_COLLISION_URDF_PATH),
            "joint_state_names": [*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES],
            "joint_state_semantics": "measured_arm_qpos_then_measured_hand_qpos_no_interpolation",
            "mount_source": (
                str(config_path) if config_path else "current_ExperimentConfig_defaults"
            ),
            "mount_config_sha256": sha256(config_path) if config_path else None,
            "T_eef_handbase": self._T_eef_handbase.tolist(),
            "T_base_camera": self.T_base_camera.tolist(),
            "camera_extrinsics_source": "Raw meta.camera_T_xarm_base_from_color",
            "timing_semantics": "host_monotonic_source_times_if_present_no_exposure_time_claim",
            "missing_timing": "row_aligned_geometry_only_not_verified_for_policy",
            "raw_notes": raw_notes,
        }

    def for_frame(self, frame):
        """Reject mismatched rows and missing joints; never substitute commands."""
        from dexmani_real.planning.kinematics.pose import rot6d_to_rotmat

        row = frame.row_index
        if (
            isinstance(row, bool)
            or not isinstance(row, (int, np.integer))
            or not 0 <= row < len(self._arm)
        ):
            raise ValueError("hand cloud requires a valid Raw row_index")
        timing = self._timing[row]
        expected_source, expected_stamp, expected_clock = raw_frame_identity(
            self.episode,
            row,
            self._observation_s[row],
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
        joint_state = np.concatenate((self._arm[row], self._hand[row]))
        if not np.isfinite(joint_state).all():
            raise ValueError(f"Raw row {row}: measured arm/hand joints are missing or nonfinite")
        position, rot6d = self._arm_fk.compute(self._arm[row])
        T_base_eef = np.eye(4)
        T_base_eef[:3, 3] = position
        T_base_eef[:3, :3] = rot6d_to_rotmat(rot6d)
        cloud = self.hand_model.transform(
            self._hand[row],
            self._T_camera_base @ T_base_eef @ self._T_eef_handbase,
        )
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
