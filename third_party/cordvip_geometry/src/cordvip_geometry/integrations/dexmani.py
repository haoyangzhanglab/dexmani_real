"""Optional DexMani Raw and RealSense sources; device imports are lazy.

Only consuming iter_live_realsense connects a camera. Its generator must be
closed on the reader owner thread, preserving the existing shutdown behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import time

import numpy as np

from ..rgbd import Frame


def iter_raw_episode(path: str | Path) -> Iterator[Frame]:
    """Read every Raw RGB-D row without external calibration or training admission.

    RGB is decoded from the existing MP4, so it is not claimed to equal the
    original live pixels. Exhaust the iterator to validate the full RGB count;
    an intentional early stop only checks the consumed prefix.
    """
    from dexmani_real.dataset.pointcloud import load_raw_episode_camera_model
    from dexmani_real.recording.storage.reader import EpisodeReader

    episode_path = Path(path).expanduser().resolve()
    with EpisodeReader(episode_path) as reader:
        reader.require_fields("rgb", "depth", "timestamp")
        camera = load_raw_episode_camera_model(reader)
        intrinsics = camera.geometry.color
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
        evidence = reader.read_timing()
        timing = evidence["rows"] if evidence["available"] else None
        timing_fields = set(timing.dtype.names or ()) if timing is not None else set()
        images = iter(reader.iter_rgb_frames())
        for row_index in range(reader.num_frames):
            rgb = next(images, None)
            if rgb is None:
                raise ValueError(f"Raw RGB ended before row {row_index}")
            if rgb.shape != (*expected_shape, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Raw RGB row {row_index} does not match color intrinsics")
            source_id = f"raw:{episode_path}:row:{row_index}"
            timestamp_ns = int(round(float(observation_s[row_index]) * 1e9))
            clock = "raw_observation_relative"
            if timing is not None:
                row = timing[row_index]
                if "camera_sequence" in timing_fields and int(row["camera_sequence"]) > 0:
                    source_id = f"raw:{episode_path}:camera:{int(row['camera_sequence'])}"
                if "camera_ns" in timing_fields and int(row["camera_ns"]) > 0:
                    timestamp_ns = int(row["camera_ns"])
                    clock = "host_monotonic_oldest_rgbd_receive"
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
    """Explicitly connect one camera; no robot, table, or external calibration.

    The driver applies its normal depth-device settings during connection.
    Frames retain each channel's first receive time, so repeats do not refresh
    the source clock. This host clock excludes camera/USB/SDK queue latency.
    """
    from dexmani_real.config.hardware import CameraParams
    from dexmani_real.sensor.camera.realsense import RealSenseCamera

    config = CameraParams(serial=serial, width=width, height=height, fps=fps)
    camera = RealSenseCamera(config, auto_exposure_priority=None)
    previous_identity: tuple[int, int] | None = None
    channel_received_ns = [0, 0]
    try:
        if not camera.connect():
            raise RuntimeError("RealSense connection failed")
        intrinsics = camera.get_geometry().color
        K = intrinsics.matrix()
        while True:
            raw = camera.read(timeout_ms=1000)
            identity = (int(raw.depth_frame_number), int(raw.color_frame_number))
            for channel, number in enumerate(identity):
                if number < 0 or (previous_identity is not None and number < previous_identity[channel]):
                    raise RuntimeError("RealSense frame identity is invalid or moved backward")
                if previous_identity is None or number != previous_identity[channel]:
                    channel_received_ns[channel] = int(raw.timestamp_ns)
            oldest_received_ns = min(channel_received_ns)
            if time.monotonic_ns() - oldest_received_ns >= int(config.source_stall_timeout_s * 1e9):
                raise RuntimeError("RealSense stopped producing new RGB or depth")
            if identity == previous_identity:
                continue
            previous_identity = identity
            if raw.depth_aligned_to_color_raw is None:
                raise RuntimeError("RealSense did not return aligned RGB-D")
            yield Frame(
                rgb=raw.rgb,
                depth_m=raw.depth_aligned_to_color_raw.astype(np.float32) * np.float32(raw.depth_scale),
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
                source_id=f"realsense:{raw.serial}:{identity[0]}:{identity[1]}",
                timestamp_ns=oldest_received_ns,
                clock="host_monotonic_oldest_rgbd_receive",
            )
    finally:
        camera.disconnect()


class RawHandCloudProvider:
    """Read measured Raw joints for the exact RGB-D row, without interpolation.

    Uses recorded camera extrinsics and the current experiment's hand mount,
    matching DexMani's offline FK. Old Raw without source timing remains a
    geometric replay; row alignment is not evidence of sensor synchronization.
    No robot/camera drivers are imported or connected by this adapter.
    """

    def __init__(self, episode, config_path=None, num_points=1024, seed=0):
        from dexmani_real.config.experiment import load_experiment_config
        from dexmani_real.dataset.pointcloud import load_raw_episode_base_from_color
        from dexmani_real.planning.kinematics.arm_fk import ArmFK
        from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rotmat
        from dexmani_real.recording.storage.reader import EpisodeReader
        from dexmani_real.robot.model import (
            XARM7_JOINT_NAMES, XARM7_XHAND_COLLISION_URDF_PATH,
            XHAND_FINGERTIP_LINK_NAMES, XHAND_RIGHT_URDF_PATH, XHAND_SDK_JOINT_NAMES,
        )
        from ..hand_cloud import HandCloudModel, _file_hash

        self.episode = Path(episode).expanduser().resolve()
        config_path = Path(config_path).expanduser().resolve() if config_path is not None else None
        config = load_experiment_config(yaml_path=config_path)
        config.hand.validate()
        self._T_eef_handbase = np.eye(4)
        self._T_eef_handbase[:3, 3] = config.hand.T_eef_handbase_pos_xyz
        self._T_eef_handbase[:3, :3] = quat_wxyz_to_rotmat(config.hand.T_eef_handbase_quat_wxyz)
        with EpisodeReader(self.episode) as reader:
            reader.require_fields('arm_qpos', 'hand_qpos', 'timestamp')
            self._arm = np.asarray(reader['arm_qpos'][:])
            self._hand = np.asarray(reader['hand_qpos'][:])
            self._observation_s = np.asarray(reader['timestamp'][:])
            self.T_base_camera = load_raw_episode_base_from_color(reader)
            evidence = reader.read_timing()
            self._timing = evidence['rows'] if evidence['available'] else None
            raw_notes = {
                key: str(reader.meta[key]) for key in ('notes', 'timestamp_source', 'termination_reason')
                if key in reader.meta
            }
        self._T_camera_base = np.linalg.inv(self.T_base_camera)
        self._arm_fk = ArmFK(str(XARM7_XHAND_COLLISION_URDF_PATH))
        self.hand_model = HandCloudModel(
            XHAND_RIGHT_URDF_PATH, XHAND_SDK_JOINT_NAMES,
            excluded_links=('right_hand_ee_link', *XHAND_FINGERTIP_LINK_NAMES),
            num_points=num_points, seed=seed,
        )
        # Hash the actual joints/times used, rather than rereading an entire RGB-D
        # recording just to hash unrelated image/depth payloads at startup.
        import hashlib
        state_hash = hashlib.sha256(self._arm.tobytes() + self._hand.tobytes() + self._observation_s.tobytes())
        if self._timing is not None:
            state_hash.update(self._timing.tobytes())
        self.metadata = {
            **self.hand_model.metadata,
            'episode': str(self.episode), 'joint_source': str(self.episode / 'data.h5'),
            'joint_and_timing_sha256': state_hash.hexdigest(),
            'arm_urdf': str(XARM7_XHAND_COLLISION_URDF_PATH),
            'arm_urdf_sha256': _file_hash(XARM7_XHAND_COLLISION_URDF_PATH),
            'joint_state_names': [*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES],
            'joint_state_semantics': 'measured_arm_qpos_then_measured_hand_qpos_no_interpolation',
            'mount_source': str(config_path) if config_path else 'current_ExperimentConfig_defaults',
            'mount_config_sha256': _file_hash(config_path) if config_path else None,
            'T_eef_handbase': self._T_eef_handbase.tolist(),
            'T_base_camera': self.T_base_camera.tolist(),
            'camera_extrinsics_source': 'Raw meta.camera_T_xarm_base_from_color',
            'timing_available': self._timing is not None,
            'timing_semantics': 'host_monotonic_source_times_if_present_no_exposure_time_claim',
            'missing_timing': 'row_aligned_geometry_only_not_verified_for_policy',
            'raw_notes': raw_notes,
        }

    def for_frame(self, frame):
        """Reject mismatched rows and missing joints; never substitute commands."""
        from dexmani_real.planning.kinematics.pose import rot6d_to_rotmat

        row = frame.row_index
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(self._arm):
            raise ValueError('hand cloud requires a valid Raw row_index')
        expected_source = f'raw:{self.episode}:row:{row}'
        expected_stamp = int(round(float(self._observation_s[row]) * 1e9))
        expected_clock = 'raw_observation_relative'
        arm_ns = hand_ns = camera_ns = -1
        if self._timing is not None:
            timing = self._timing[row]
            arm_ns, hand_ns, camera_ns = (int(timing[name]) for name in ('arm_ns', 'hand_ns', 'camera_ns'))
            if int(timing['camera_sequence']) > 0:
                expected_source = f"raw:{self.episode}:camera:{int(timing['camera_sequence'])}"
            if camera_ns > 0:
                expected_stamp, expected_clock = camera_ns, 'host_monotonic_oldest_rgbd_receive'
        if (frame.source_id != expected_source or frame.timestamp_ns != expected_stamp
                or frame.clock != expected_clock):
            raise ValueError('hand cloud and RGB-D must come from the same Raw row and source clock')
        joint_state = np.concatenate((self._arm[row], self._hand[row]))
        if not np.isfinite(joint_state).all():
            raise ValueError(f'Raw row {row}: measured arm/hand joints are missing or nonfinite')
        position, rot6d = self._arm_fk.compute(self._arm[row])
        T_base_eef = np.eye(4)
        T_base_eef[:3, 3] = position
        T_base_eef[:3, :3] = rot6d_to_rotmat(rot6d)
        cloud = self.hand_model.transform(
            self._hand[row], self._T_camera_base @ T_base_eef @ self._T_eef_handbase,
        )
        timing_available = min(arm_ns, hand_ns, camera_ns) > 0 and expected_clock == 'host_monotonic_oldest_rgbd_receive'
        skew_ms = (max(arm_ns, hand_ns, camera_ns) - min(arm_ns, hand_ns, camera_ns)) / 1e6 if timing_available else None
        return {
            **cloud, 'joint_state': joint_state.copy(), 'T_base_camera': self.T_base_camera,
            'source_id': frame.source_id, 'row_index': int(row),
            'arm_timestamp_ns': arm_ns, 'hand_timestamp_ns': hand_ns,
            'camera_timestamp_ns': camera_ns, 'clock': expected_clock,
            'timing_available': bool(timing_available), 'skew_ms': skew_ms,
        }
