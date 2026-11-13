"""共享模型权重的 Mask 引导 FoundationPose 位姿跟踪。"""

import importlib
import sys
from pathlib import Path

import numpy as np

from .mesh import simplify_mesh

MIN_DEPTH_PIXELS = 4


class TrackingFailure(RuntimeError):
    """当前场景丢失位姿，调用方应结束该段。"""


def prepare_tracking_frame(rgb, depth_m, intrinsics):
    """多物体共享同帧深度预处理。"""
    import torch
    from Utils import bilateral_filter_depth, depth2xyzmap_batch, erode_depth

    depth = torch.as_tensor(depth_m, device="cuda", dtype=torch.float)
    depth = erode_depth(depth, radius=2, device="cuda")
    depth = bilateral_filter_depth(depth, radius=2, device="cuda")
    K = torch.as_tensor(intrinsics, device="cuda", dtype=torch.float)
    xyz_map = depth2xyzmap_batch(depth[None], K[None], zfar=np.inf)[0]
    return {
        "rgb": torch.as_tensor(rgb, device="cuda", dtype=torch.float),
        "depth": depth,
        "xyz_map": xyz_map,
    }


def guide_centered_pose(pose_camera_centered, intrinsics, mask):
    """以掩码边框中心引导 XY，保留上一帧 Z 和旋转。"""
    pose = np.array(pose_camera_centered, copy=True)
    z = pose[2, 3]
    if z <= 0:
        raise TrackingFailure("Previous centered pose must be in front of the camera")
    rows, cols = np.nonzero(mask)
    center = np.array(
        [(cols.min() + cols.max()) / 2.0, (rows.min() + rows.max()) / 2.0, 1.0]
    )
    ray = np.linalg.solve(intrinsics, center)
    pose[:2, 3] = (ray * (z / ray[2]))[:2]
    return pose


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

        # 固定直径采样，避免物体顺序影响裁剪。
        random_state = np.random.get_state()
        try:
            np.random.seed(0)
            self.estimator = estimator_type(
                mesh=self.mesh,
                glctx=context,
                scorer=shared_with.estimator.scorer
                if shared_with is not None
                else None,
                refiner=shared_with.estimator.refiner
                if shared_with is not None
                else None,
                debug=0,
                debug_dir="/tmp",
            )
        finally:
            np.random.set_state(random_state)

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


def track_objects(trackers, rgb, depth_m, intrinsics, masks):
    """各物体独立裁剪，同帧的所有物体合批执行两轮 refinement。"""
    import torch
    from learning.datasets.pose_dataset import BatchPoseData
    from learning.training.predict_pose_refine import make_crop_data_batch

    names = list(trackers)
    estimators = [tracker.estimator for tracker in trackers.values()]
    refiner = estimators[0].refiner
    previous = torch.cat([est.pose_last.reshape(1, 4, 4) for est in estimators])
    guided = []
    for name, pose in zip(names, previous.detach().cpu().numpy()):
        try:
            guided.append(guide_centered_pose(pose, intrinsics, masks[name]))
        except TrackingFailure as error:
            raise TrackingFailure(f"{name}: {error}") from error

    with torch.inference_mode():
        poses = torch.as_tensor(np.stack(guided), device="cuda", dtype=torch.float)
        frame = prepare_tracking_frame(rgb, depth_m, intrinsics)
        for _ in range(2):
            crops = [
                make_crop_data_batch(
                    refiner.cfg.input_resize,
                    poses[index : index + 1],
                    **frame,
                    K=intrinsics,
                    crop_ratio=refiner.cfg.crop_ratio,
                    mesh_diameter=est.diameter,
                    glctx=est.glctx,
                    mesh_tensors=est.mesh_tensors,
                )
                for index, est in enumerate(estimators)
            ]
            batch = BatchPoseData(
                **{
                    key: torch.cat([getattr(crop, key) for crop in crops])
                    for key in (
                        "rgbAs",
                        "rgbBs",
                        "xyz_mapAs",
                        "xyz_mapBs",
                        "poseA",
                        "mesh_diameters",
                    )
                }
            )
            # 避免 cuDNN 随 batch 大小切换卷积算法。
            with torch.backends.cudnn.flags(enabled=False):
                poses = refiner.refine(batch)

        mesh_transforms = torch.stack([est.tf_to_centered_mesh for est in estimators])
        output = (poses @ mesh_transforms).cpu().numpy()

    for index, (name, est) in enumerate(zip(names, estimators)):
        if not np.isfinite(output[index]).all():
            raise TrackingFailure(f"{name}: FoundationPose returned nonfinite pose")
        est.pose_last = poses[index : index + 1]
    return dict(zip(names, output))
