"""从一个 RawReader 顺序读取同一行 RGB-D、实测关节和原始时间。"""

from __future__ import annotations

import numpy as np

from .rgbd import RGBDFrame

HAND_SAMPLE_SEED = 0


def iter_raw_rows(reader, camera):
    """完整消费时核对 MP4 总行数；调用方拥有并关闭唯一 reader。"""
    reader.require_fields("rgb", "depth")
    intrinsics = camera.color
    K = intrinsics.matrix()
    expected_shape = (intrinsics.height, intrinsics.width)
    depth_scale = np.float32(camera.depth_scale_m)
    depth = reader["depth"]
    if depth.shape != (reader.num_frames, *expected_shape) or depth.dtype != np.uint16:
        raise ValueError(
            "Raw depth must be uint16 [rows,H,W] matching color intrinsics"
        )
    timing = reader["timing/rows"][:]
    images = iter(reader.iter_rgb_frames())
    for row_index in range(reader.num_frames):
        rgb = next(images, None)
        if rgb is None:
            raise ValueError(f"Raw RGB ended before row {row_index}")
        depth_m = depth[row_index].astype(np.float32) * depth_scale
        yield (
            RGBDFrame(
                rgb=rgb,
                depth_m=depth_m,
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
                timestamp_ns=int(timing[row_index]["camera_ns"]),
                row_index=row_index,
            ),
            timing[row_index],
        )
    if next(images, None) is not None:
        raise ValueError("Raw RGB contains more frames than the HDF5 rows")


def track_raw(reader, tracker, *, with_hand=False, hand_points=1024):
    """逐行返回几何结果；reader 与 tracker 的生命周期由调用者管理。"""
    camera = reader.read_camera()
    hand = RawHandCloud(reader, camera, num_points=hand_points) if with_hand else None
    for frame, timing in iter_raw_rows(reader, camera):
        result = tracker.step(frame)
        if hand is not None:
            result["hand"] = hand.for_row(frame, timing)
        yield result


class RawHandCloud:
    """使用同一 Raw 行实测关节、记录外参和完整组合 URDF；行对齐不证明物理同步。"""

    def __init__(self, reader, camera, num_points=1024):
        import xml.etree.ElementTree as ET

        from dexmani_real.robot.model import (
            XARM7_JOINT_NAMES,
            XARM7_XHAND_RIGHT_URDF_PATH,
            XHAND_FINGERTIP_LINK_NAMES,
            XHAND_SDK_JOINT_NAMES,
        )

        from .hand import HandCloud

        self.reader = reader
        reader.require_fields("arm_qpos", "hand_qpos")
        self.T_base_camera = camera.T_xarm_base_from_color
        if self.T_base_camera is None:
            raise ValueError("hand cloud requires recorded camera extrinsics")
        excluded = ("right_hand_ee_link", *XHAND_FINGERTIP_LINK_NAMES)
        excluded_links = [
            link.attrib["name"]
            for link in ET.parse(XARM7_XHAND_RIGHT_URDF_PATH).getroot().findall("link")
            if not link.attrib["name"].startswith("right_hand_")
            or link.attrib["name"] in excluded
        ]
        self.hand_model = HandCloud(
            XARM7_XHAND_RIGHT_URDF_PATH,
            (*XARM7_JOINT_NAMES, *XHAND_SDK_JOINT_NAMES),
            excluded_links=excluded_links,
            num_points=num_points,
            seed=HAND_SAMPLE_SEED,
            T_camera_from_root=np.linalg.inv(self.T_base_camera),
        )
        self.metadata = self.hand_model.metadata

    def for_row(self, frame, timing):
        """同一 reader 当前行的实测关节；不插值、不用命令替代反馈。"""
        row = frame.row_index
        if (
            isinstance(row, bool)
            or not isinstance(row, (int, np.integer))
            or not 0 <= row < self.reader.num_frames
        ):
            raise ValueError(
                "Raw hand geometry requires an integer row_index in "
                f"0..{self.reader.num_frames - 1}"
            )
        arm_ns, hand_ns, camera_ns = (
            int(timing[name]) for name in ("arm_ns", "hand_ns", "camera_ns")
        )
        joint_state = np.concatenate(
            (self.reader["arm_qpos"][row], self.reader["hand_qpos"][row])
        )
        cloud = self.hand_model.transform(joint_state)
        timing_available = min(arm_ns, hand_ns, camera_ns) > 0
        skew_ms = (
            (max(arm_ns, hand_ns, camera_ns) - min(arm_ns, hand_ns, camera_ns)) / 1e6
            if timing_available
            else None
        )
        return {
            **cloud,
            "joint_state": joint_state,
            "T_base_camera": self.T_base_camera,
            "arm_timestamp_ns": arm_ns,
            "hand_timestamp_ns": hand_ns,
            "camera_timestamp_ns": camera_ns,
            "timing_available": bool(timing_available),
            "skew_ms": skew_ms,
        }
