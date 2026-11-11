"""从一个 RawReader 顺序读取同一行 RGB-D 和原始时间。"""

from __future__ import annotations

import numpy as np

from .rgbd import RGBDFrame


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
        if rgb.dtype != np.uint8 or rgb.shape != (*expected_shape, 3):
            raise ValueError(f"Raw RGB grid invalid at row {row_index}")
        depth_m = depth[row_index].astype(np.float32) * depth_scale
        yield (
            RGBDFrame(
                rgb=rgb,
                depth_m=depth_m,
                K=K,
                distortion_model=intrinsics.distortion_model,
                distortion_coeffs=intrinsics.distortion_coeffs,
            ),
            timing[row_index],
        )
    if next(images, None) is not None:
        raise ValueError("Raw RGB contains more frames than the HDF5 rows")
