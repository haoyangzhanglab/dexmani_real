"""共享模型权重的 Mask 引导 FoundationPose 位姿跟踪。"""

import importlib
import logging
import sys
from pathlib import Path

import numpy as np

from .geometry import simplify_mesh, validate_pose


class PoseRegistrationError(RuntimeError):
    """模型未建立有效位姿，不能使用推测或历史位姿。"""


MIN_DEPTH_PIXELS = 4


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
    """bbox 中心引导 centered mesh 的 XY，保留上帧 Z 和旋转。"""
    pose = np.array(pose_camera_centered, copy=True)
    z = pose[2, 3]
    if z <= 0:
        raise ValueError("Previous centered pose must be in front of the camera")
    rows, cols = np.nonzero(mask)
    center = np.array(
        [(cols.min() + cols.max()) / 2.0, (rows.min() + rows.max()) / 2.0, 1.0]
    )
    ray = np.linalg.solve(intrinsics, center)
    pose[:2, 3] = (ray * (z / ray[2]))[:2]
    return pose


class PoseTracker:
    def __init__(self, root, mesh, debug_dir, *, shared_with=None):
        root = Path(root).expanduser().resolve()
        self.mesh = simplify_mesh(mesh)

        if shared_with is None:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FoundationPose requires CUDA; input validation alone does not run tracking"
                )
            sys.path.insert(0, str(root))
            # 上游 Utils 导入会重设 root logging，恢复宿主级别以免逐帧刷屏。
            log_level = logging.getLogger().level
            try:
                estimator_module = importlib.import_module("estimater")
            finally:
                logging.getLogger().setLevel(log_level)
            import nvdiffrast.torch as dr

            estimator_type = estimator_module.FoundationPose
            self.context = dr.RasterizeCudaContext()
        else:
            estimator_type = type(shared_with.estimator)
            self.context = shared_with.context
        # 固定直径采样，避免物体顺序影响裁剪与体素化。
        random_state = np.random.get_state()
        try:
            np.random.seed(0)
            self.estimator = estimator_type(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                glctx=self.context,
                scorer=shared_with.estimator.scorer
                if shared_with is not None
                else None,
                refiner=shared_with.estimator.refiner
                if shared_with is not None
                else None,
                debug=0,
                debug_dir=str(Path(debug_dir).resolve()),
            )
        finally:
            np.random.set_state(random_state)

    def register(self, rgb, depth_m, intrinsics, mask):
        self.estimator.pose_last = None
        pose = self.estimator.register(
            K=intrinsics,
            rgb=rgb,
            depth=depth_m,
            ob_mask=mask,
            iteration=5,
        )
        if self.estimator.pose_last is None:
            raise PoseRegistrationError(
                "FoundationPose registration did not initialize tracking"
            )
        try:
            return validate_pose(pose)
        except ValueError as exc:
            raise PoseRegistrationError(f"invalid FoundationPose output: {exc}") from exc

    def track(self, intrinsics, mask, *, prepared_frame):
        import torch

        previous = self.estimator.pose_last
        guided = guide_centered_pose(
            previous.detach().cpu().numpy().reshape(4, 4), intrinsics, mask
        )
        self.estimator.pose_last = torch.as_tensor(
            guided, dtype=previous.dtype, device=previous.device
        ).reshape(previous.shape)
        estimator = self.estimator
        pose, _ = estimator.refiner.predict(
            **prepared_frame,
            K=intrinsics,
            iteration=2,
            ob_in_cams=estimator.pose_last.reshape(1, 4, 4).detach().cpu().numpy(),
            normal_map=None,
            mesh=estimator.mesh,
            mesh_tensors=estimator.mesh_tensors,
            mesh_diameter=estimator.diameter,
            glctx=estimator.glctx,
            get_vis=False,
        )
        estimator.pose_last = pose
        pose = (
            (pose @ estimator.get_tf_to_centered_mesh())
            .detach()
            .cpu()
            .numpy()
            .reshape(4, 4)
        )
        try:
            return validate_pose(pose)
        except ValueError as exc:
            raise PoseRegistrationError(f"invalid FoundationPose output: {exc}") from exc
