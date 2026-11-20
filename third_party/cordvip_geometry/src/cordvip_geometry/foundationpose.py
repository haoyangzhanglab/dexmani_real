"""共享模型权重的 FoundationPose 注册与位姿 refinement。"""

import importlib
import sys
from pathlib import Path

import numpy as np

from .mesh import simplify_mesh


class TrackingFailure(RuntimeError):
    """当前观测的匹配、深度或位姿证据不足，允许后续帧重试；后端异常直接传播。"""


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
        # 连续性检查在 CPU 上执行；静态几何只复制一次。
        self.object_from_center = np.linalg.inv(
            self.estimator.tf_to_centered_mesh.detach().cpu().numpy()
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
        return pose


def _project_center(centered_pose, intrinsics, image_shape):
    x, y, z = centered_pose[:3, 3]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    if not (0 < z < np.inf and 0 < fx < np.inf and 0 < fy < np.inf):
        return None
    u = x / z * fx + intrinsics[0, 2]
    v = y / z * fy + intrinsics[1, 2]
    height, width = image_shape
    return (u, v) if 0 <= u < width and 0 <= v < height else None


def _box_geometry(box_xyxy):
    # OSTrack 适配器和已通过深度校验的 Mask 提供图像内非空框。
    left = np.asarray(box_xyxy[:2])
    size = np.asarray(box_xyxy[2:]) - left
    return left + size / 2, size


def bbox_center_offset(centered_pose, intrinsics, mask_box, image_shape, *, tracker_box=None):
    """验证 Mask 框中的物体中心，再将它表达在同帧的跟踪框内。"""
    pixel = _project_center(centered_pose, intrinsics, image_shape)
    if pixel is None:
        return None
    mask_center, mask_size = _box_geometry(mask_box)
    offset = (mask_center - pixel) / mask_size
    if not np.all(np.abs(offset) <= 0.5):
        return None
    if tracker_box is None:
        return offset
    center, size = _box_geometry(tracker_box)
    # 两个框共用同一像素锚点，后续先验按 OSTrack 的宽高还原。
    offset = (center - pixel) / size
    return offset if np.all(np.abs(offset) <= 0.5) else None


def xy_prior_delta(centered_pose, intrinsics, box_xyxy, offset, image_shape, gain):
    """以相机系米制单位修正居中原点的 XY，限制异常框的影响。"""
    pixel = _project_center(centered_pose, intrinsics, image_shape)
    if pixel is None:
        return np.zeros(2)
    center, size = _box_geometry(box_xyxy)
    limit = size / 2
    error = np.clip(center - offset * size - pixel, -limit, limit)
    focal = np.array((intrinsics[0, 0], intrinsics[1, 1]))
    return gain * error * centered_pose[2, 3] / focal


def track_objects(
    trackers, rgb, depth_m, intrinsics, frame_cache, *, bbox_priors=None, xy_prior_gain=0.2
):
    """合批生成候选位姿；调用方逐对象校验，拒绝时恢复原 pose_last。"""
    import torch
    from learning.training.predict_pose_refine import (
        render_crops,
        sample_observed_crops,
    )

    names = list(trackers)
    estimators = [tracker.estimator for tracker in trackers.values()]
    refiner = estimators[0].refiner
    poses = torch.cat(
        [estimator.pose_last.reshape(1, 4, 4) for estimator in estimators]
    )
    with torch.inference_mode():
        if bbox_priors and xy_prior_gain:
            # 同一已验证位姿的 CPU 副本用于投影，避免按对象回读 CUDA 张量。
            deltas = np.zeros((len(names), 2), dtype=np.float32)
            for index, name in enumerate(names):
                if name in bbox_priors:
                    centered_pose, box_xyxy, offset = bbox_priors[name]
                    deltas[index] = xy_prior_delta(
                        centered_pose, intrinsics, box_xyxy, offset,
                        depth_m.shape, xy_prior_gain,
                    )
            poses[:, :2, 3] += poses.new_tensor(deltas)
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

    for index, estimator in enumerate(estimators):
        estimator.pose_last = poses[index : index + 1]
    return dict(zip(names, output))
