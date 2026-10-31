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
    diagnostics: dict


@dataclass
class SceneCloudResult:
    objects: dict[str, SceneObjectCloud]
    overlap_mask: np.ndarray


def extract_scene_clouds(rgb, depth_m, K, masks, *, pixel_stride=2):
    """反投影互斥掩码中的非零实测深度；coverage 为有效点占独占采样像素的比例。"""
    if (
        isinstance(pixel_stride, bool)
        or not isinstance(pixel_stride, (int, np.integer))
        or pixel_stride <= 0
    ):
        raise ValueError("pixel_stride must be a positive integer")
    if not isinstance(masks, dict):
        raise ValueError("masks must map nonempty instance names to boolean (H, W) arrays")
    shape = depth_m.shape
    semantic_masks = {}
    seen = np.zeros(shape, dtype=bool)
    overlap = np.zeros(shape, dtype=bool)
    for name, value in masks.items():
        mask = np.asarray(value)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Instance names must be nonempty strings")
        if mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError(f"Mask for {name!r} must be boolean with shape {shape}")
        semantic_masks[name] = mask.copy()
        overlap |= seen & mask
        seen |= mask

    sample = (slice(None, None, pixel_stride), slice(None, None, pixel_stride))
    sampled_depth = depth_m[sample]
    sampled_rgb = rgb[sample]
    objects = {}
    for name, semantic in semantic_masks.items():
        exclusive = semantic[sample] & ~overlap[sample]
        selected = exclusive & (sampled_depth > 0)
        v, u = np.nonzero(selected)
        pixel_indices = np.column_stack((v * pixel_stride, u * pixel_stride)).astype(np.int32)
        z = sampled_depth[selected]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            points = np.column_stack(
                (
                    (pixel_indices[:, 1] - K[0, 2]) * z / K[0, 0],
                    (pixel_indices[:, 0] - K[1, 2]) * z / K[1, 1],
                    z,
                )
            ).astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError(f"Observed points for {name!r} exceed the finite float32 range")
        colors = sampled_rgb[selected]
        observed_mask = np.zeros(shape, dtype=bool)
        observed_mask[sample] = selected
        count = len(points)
        sampled_count = int(np.count_nonzero(exclusive))
        diagnostics = dict(
            pixel_stride=int(pixel_stride),
            semantic_pixel_count=int(np.count_nonzero(semantic)),
            overlap_pixel_count=int(np.count_nonzero(semantic & overlap)),
            exclusive_sampled_pixel_count=sampled_count,
            observed_point_count=count,
            depth_coverage=count / sampled_count if sampled_count else None,
            model_depth_point_count=None,
            model_depth_coverage=None,
            model_supported_point_count=None,
            model_supported_fraction=None,
            median_abs_model_depth_error_m=None,
            depth_tolerance_m=None,
        )
        objects[name] = SceneObjectCloud(
            semantic_mask=semantic,
            observed_mask=observed_mask,
            observed_points_camera=points,
            observed_colors=colors,
            observed_pixel_indices=pixel_indices,
            model_supported_mask=None,
            observed_model_depth_residual_m=None,
            diagnostics=diagnostics,
        )
    return SceneCloudResult(objects=objects, overlap_mask=overlap)


def add_model_depth_support(measured, model_depths, *, depth_tolerance_m=0.015):
    """原位附加同网格模型深度比较；保持实测点序，缺失模型深度为零、未定义指标为 None。"""
    if not isinstance(model_depths, dict) or not model_depths.keys() <= measured.objects.keys():
        raise ValueError("model_depths must name measured objects")
    if not model_depths:
        return measured
    if isinstance(depth_tolerance_m, (bool, np.bool_)):
        raise ValueError("depth_tolerance_m must be finite and positive")
    depth_tolerance_m = float(depth_tolerance_m)
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
        raise ValueError("depth_tolerance_m must be finite and positive")
    shape = measured.overlap_mask.shape
    validated_models = {}
    for name, value in model_depths.items():
        value = np.asarray(value)
        if (
            value.shape != shape
            or np.iscomplexobj(value)
            or not np.issubdtype(value.dtype, np.number)
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError(
                f"Model depth for {name!r} must be finite nonnegative with shape {shape}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            value = value.astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f"Model depth for {name!r} exceeds the finite float32 range")
        validated_models[name] = value

    for name, model in validated_models.items():
        cloud = measured.objects[name]
        pixels = cloud.observed_pixel_indices
        points = cloud.observed_points_camera
        count = len(points)
        model_depth = model[pixels[:, 0], pixels[:, 1]]
        comparable = model_depth > 0
        residual = np.full(count, np.nan, dtype=np.float32)
        residual[comparable] = points[comparable, 2] - model_depth[comparable]
        supported = comparable & (np.abs(residual) <= depth_tolerance_m)
        support_pixels = pixels[supported]
        cloud.model_supported_mask = np.zeros(shape, dtype=bool)
        cloud.model_supported_mask[support_pixels[:, 0], support_pixels[:, 1]] = True
        cloud.observed_model_depth_residual_m = residual
        comparable_count, supported_count = int(np.count_nonzero(comparable)), len(support_pixels)
        cloud.diagnostics.update(
            depth_tolerance_m=depth_tolerance_m,
            model_depth_point_count=comparable_count,
            model_depth_coverage=comparable_count / count if count else None,
            model_supported_point_count=supported_count,
            model_supported_fraction=supported_count / count if count else None,
            median_abs_model_depth_error_m=(
                float(np.median(np.abs(residual[comparable]))) if comparable_count else None
            ),
        )
    return measured
