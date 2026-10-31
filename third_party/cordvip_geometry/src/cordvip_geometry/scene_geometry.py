"""Same-frame semantic RGB-D clouds, independent of optional object models."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneObjectCloud:
    """Camera-frame metres/RGB uint8 with row-major (v,u) indices on the global stride grid.
    Model residual is measured Z minus rendered Z; NaN denotes no model surface."""

    semantic_mask: np.ndarray
    observed_mask: np.ndarray
    observed_points_camera: np.ndarray
    observed_colors: np.ndarray
    observed_pixel_indices: np.ndarray
    model_supported_mask: np.ndarray | None
    observed_model_depth_residual_m: np.ndarray | None
    model_supported_point_count: int = 0
    model_supported_fraction: float = 0.0


def extract_scene_clouds(rgb, depth_m, directions, masks, *, pixel_stride=2) -> dict[str, SceneObjectCloud]:
    """反投影互斥掩码中的非零实测深度。"""
    shape = depth_m.shape
    sample = (slice(None, None, pixel_stride), slice(None, None, pixel_stride))
    sampled_depth = depth_m[sample]
    sampled_rgb = rgb[sample]
    objects = {}
    for name, semantic in masks.items():
        # The SAM pipe boundary already admits mutually exclusive masks.
        selected = semantic[sample] & (sampled_depth > 0)
        v, u = np.nonzero(selected)
        pixel_indices = np.column_stack((v * pixel_stride, u * pixel_stride)).astype(np.int32)
        z = sampled_depth[selected]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            points = (directions[selected] * z[:, None]).astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError(f"Observed points for {name!r} exceed the finite float32 range")
        colors = sampled_rgb[selected]
        observed_mask = np.zeros(shape, dtype=bool)
        observed_mask[sample] = selected
        objects[name] = SceneObjectCloud(
            semantic_mask=semantic,
            observed_mask=observed_mask,
            observed_points_camera=points,
            observed_colors=colors,
            observed_pixel_indices=pixel_indices,
            model_supported_mask=None,
            observed_model_depth_residual_m=None,
        )
    return objects


def add_model_depth_support(
    measured: dict[str, SceneObjectCloud], model_z_by_object: dict[str, np.ndarray], *, depth_tolerance_m=0.015,
) -> None:
    """原位附加同序模型 Z 的深度支持；无模型表面处残差为 NaN。"""
    # ModelDepthRenderer owns float32 nonnegative camera Z in observed point order.
    for name, model_z in model_z_by_object.items():
        cloud = measured[name]
        pixels = cloud.observed_pixel_indices
        points = cloud.observed_points_camera
        count = len(points)
        comparable = model_z > 0
        residual = np.full(count, np.nan, dtype=np.float32)
        residual[comparable] = points[comparable, 2] - model_z[comparable]
        supported = comparable & (np.abs(residual) <= depth_tolerance_m)
        support_pixels = pixels[supported]
        cloud.model_supported_mask = np.zeros(cloud.observed_mask.shape, dtype=bool)
        cloud.model_supported_mask[support_pixels[:, 0], support_pixels[:, 1]] = True
        cloud.observed_model_depth_residual_m = residual
        cloud.model_supported_point_count = len(support_pixels)
        cloud.model_supported_fraction = len(support_pixels) / count if count else 0.0
