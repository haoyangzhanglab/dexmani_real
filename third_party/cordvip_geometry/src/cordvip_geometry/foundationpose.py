"""共享模型权重、由分割掩码引导的 FoundationPose 位姿跟踪。"""

import importlib
import sys
from pathlib import Path

import numpy as np

from .mesh import simplify_mesh

MIN_DEPTH_PIXELS = 4


class TrackingFailure(RuntimeError):
    """当前场景丢失位姿，调用方应结束该段。"""


def prepare_tracking_frame(rgb, depth_m, intrinsics, render_size, cache):
    """共享同帧预处理；缓存限于相机参数和图像尺寸固定的当前段。"""
    import torch
    from Utils import (
        bilateral_filter_depth,
        depth2xyzmap_batch,
        erode_depth,
        make_crop_grid_constants,
        projection_matrix_from_intrinsics,
    )

    if not cache:
        # 渲染保留相机内参精度，深度反投影使用 float32。
        cache.update(
            rgb=torch.empty(rgb.shape, device="cuda", dtype=torch.float32),
            depth=torch.empty(depth_m.shape, device="cuda", dtype=torch.float32),
            K=torch.as_tensor(intrinsics, device="cuda"),
            K_float32=torch.as_tensor(intrinsics, device="cuda", dtype=torch.float32),
            projection=torch.as_tensor(
                projection_matrix_from_intrinsics(
                    intrinsics, *depth_m.shape, znear=0.001, zfar=100
                ),
                device="cuda",
                dtype=torch.float32,
            ),
            crop_grid=make_crop_grid_constants(
                depth_m.shape, render_size, "cuda", torch.float32
            ),
        )
    cache["rgb"].copy_(torch.from_numpy(rgb))
    cache["depth"].copy_(torch.from_numpy(depth_m))
    depth = erode_depth(cache["depth"], radius=2, device="cuda")
    depth = bilateral_filter_depth(depth, radius=2, device="cuda")
    xyz_map = depth2xyzmap_batch(depth[None], cache["K_float32"][None], zfar=np.inf)[0]
    return cache["rgb"], xyz_map


class FoundationPoseTracker:
    def __init__(self, root, mesh, *, shared_with=None):
        root = Path(root).expanduser().resolve()
        self.mesh = simplify_mesh(mesh)

        if shared_with is None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("FoundationPose requires CUDA")
            sys.path.insert(0, str(root))
            estimator_module = importlib.import_module("estimater")
            import nvdiffrast.torch as dr

            estimator_type = estimator_module.FoundationPose
            context = dr.RasterizeCudaContext()
        else:
            estimator_type = type(shared_with.estimator)
            context = shared_with.estimator.glctx

        self.estimator = estimator_type(
            mesh=self.mesh,
            glctx=context,
            scorer=shared_with.estimator.scorer if shared_with is not None else None,
            refiner=shared_with.estimator.refiner if shared_with is not None else None,
        )

    def register(self, rgb, depth_m, intrinsics, mask):
        import torch

        self.estimator.pose_last = None
        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=False
        ):
            pose = self.estimator.register(
                K=intrinsics,
                rgb=rgb,
                depth=depth_m,
                ob_mask=mask,
                iteration=5,
            )
        if self.estimator.pose_last is None:
            raise TrackingFailure(
                "FoundationPose registration did not initialize tracking"
            )
        if not np.isfinite(pose).all():
            raise TrackingFailure("FoundationPose returned nonfinite pose")
        return pose


def track_objects(trackers, rgb, depth_m, intrinsics, masks, frame_cache):
    """各物体独立渲染，观测裁剪及两轮 refinement 合批执行。"""
    import torch
    from learning.training.predict_pose_refine import (
        render_crops,
        sample_observed_crops,
    )

    names = list(trackers)
    estimators = [tracker.estimator for tracker in trackers.values()]
    refiner = estimators[0].refiner
    previous = torch.cat(
        [estimator.pose_last.reshape(1, 4, 4) for estimator in estimators]
    )
    invalid = (previous[:, 2, 3] <= 0).cpu().numpy()
    if invalid.any():
        name = names[np.flatnonzero(invalid)[0]]
        raise TrackingFailure(
            f"{name}: Previous centered pose must be in front of the camera"
        )
    rays = []
    for name in names:
        rows, cols = np.nonzero(masks[name])
        center = [
            (cols.min() + cols.max()) / 2.0,
            (rows.min() + rows.max()) / 2.0,
            1.0,
        ]
        rays.append(np.linalg.solve(intrinsics, center))

    with torch.inference_mode():
        # 使用 float64 引导 XY，赋值时才舍入到 float32，保持与 NumPy 计算一致。
        poses = previous.clone()
        directions = torch.as_tensor(np.stack(rays), device="cuda")
        scale = poses[:, 2, 3].double() / directions[:, 2]
        poses[:, :2, 3] = (directions * scale[:, None])[:, :2]
        rgb_tensor, xyz_map = prepare_tracking_frame(
            rgb, depth_m, intrinsics, refiner.cfg.input_resize, frame_cache
        )
        for _ in range(2):
            crops = [
                render_crops(
                    refiner.cfg.input_resize,
                    poses[index : index + 1],
                    image_size=depth_m.shape,
                    K=frame_cache["K"],
                    projection_mat=frame_cache["projection"],
                    crop_ratio=refiner.cfg.crop_ratio,
                    mesh_diameter=estimator.diameter,
                    glctx=estimator.glctx,
                    mesh_tensors=estimator.mesh_tensors,
                )
                for index, estimator in enumerate(estimators)
            ]
            transforms, rendered_rgb, rendered_xyz, diameters = (
                torch.cat(values) for values in zip(*crops)
            )
            batch = sample_observed_crops(
                poses,
                rgb_tensor,
                xyz_map,
                transforms,
                rendered_rgb,
                rendered_xyz,
                diameters,
                refiner.cfg.input_resize,
                grid_constants=frame_cache["crop_grid"],
            )
            # 避免 cuDNN 随 batch 大小切换卷积算法。
            with torch.backends.cudnn.flags(enabled=False):
                poses = refiner.refine(batch)

        mesh_transforms = torch.stack(
            [estimator.tf_to_centered_mesh for estimator in estimators]
        )
        output = (poses @ mesh_transforms).cpu().numpy()

    for index, (name, estimator) in enumerate(zip(names, estimators)):
        if not np.isfinite(output[index]).all():
            raise TrackingFailure(f"{name}: FoundationPose returned nonfinite pose")
        estimator.pose_last = poses[index : index + 1]
    return dict(zip(names, output))
