"""FoundationPose adapter with process-local CUDA imports; construct only in the standalone owner."""

import importlib
from pathlib import Path
import sys

import numpy as np

from .geometry import validate_pose


PLUS_REVISION = "58aa7154de23ae86d542366858fc6bcca3b747b0"


def prepare_tracking_frame(rgb, depth_m, intrinsics):
    """Share upstream tracking tensors within one frame; registration keeps its own preprocessing."""
    import torch
    from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap_batch

    depth = torch.as_tensor(depth_m, device="cuda", dtype=torch.float)
    depth = erode_depth(depth, radius=2, device="cuda")
    depth = bilateral_filter_depth(depth, radius=2, device="cuda")
    K = torch.as_tensor(intrinsics, device="cuda", dtype=torch.float)
    xyz_map = depth2xyzmap_batch(depth[None], K[None], zfar=np.inf)[0]
    return dict(
        rgb=torch.as_tensor(rgb, device="cuda", dtype=torch.float), depth=depth, xyz_map=xyz_map
    )


def guide_centered_pose(pose_camera_centered, intrinsics, mask):
    """Guide the centered mesh from the 2-D bbox, preserving previous Z and rotation.
    T_camera_asset = T_camera_centered @ T_centered_asset; never fill depth to infer a centroid."""
    pose = validate_pose(pose_camera_centered)
    K, mask = np.asarray(intrinsics), np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("2D guidance requires a nonempty exclusive bool mask")
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics for 2D guidance")
    z = pose[2, 3]
    if z <= 0:
        raise ValueError("Previous centered pose must be in front of the camera")
    rows, cols = np.nonzero(mask)
    center = np.array([(cols.min() + cols.max()) / 2.0, (rows.min() + rows.max()) / 2.0, 1.0])
    ray = np.linalg.solve(K, center)
    if not np.isfinite(ray).all() or ray[2] <= 0:
        raise ValueError("Invalid bbox-center camera ray")
    pose[:2, 3] = (ray * (z / ray[2]))[:2]
    return pose


class FoundationPosePlusTracker:
    def __init__(
        self, root, mesh, debug_dir, register_iterations=5, track_iterations=2, *, shared_with=None
    ):
        self.root = Path(root).expanduser().resolve()
        if shared_with is None:
            required = ["estimater.py", "Utils.py"]
            for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
                required.extend((f"weights/{run}/config.yml", f"weights/{run}/model_best.pth"))
            missing = [name for name in required if not (self.root / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"FoundationPose source/weights missing under {self.root}: {', '.join(missing)}"
                )
        elif shared_with.root != self.root:
            raise ValueError("Shared FoundationPose weights must use the same source directory")
        for value in (register_iterations, track_iterations):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Refinement iterations must be positive integers")
        self.register_iterations = register_iterations
        self.track_iterations = track_iterations
        self.intrinsics = None

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "FoundationPose requires CUDA; input validation alone does not run tracking"
            )
        # Upstream uses generic module names. Refuse a mixture of different source directories.
        for name in ("estimater", "Utils", "learning", "datareader"):
            existing = sys.modules.get(name)
            if existing is not None:
                source = getattr(existing, "__file__", None)
                locations = [source] if source else list(getattr(existing, "__path__", ()))
                if not locations or any(
                    not Path(path).resolve().is_relative_to(self.root) for path in locations
                ):
                    raise RuntimeError(
                        f"Module {name!r} is already loaded outside {self.root}; use a fresh process"
                    )
        sys.path.insert(0, str(self.root))
        estimator_module = importlib.import_module("estimater")
        import nvdiffrast.torch as dr

        # Share immutable weights/context on one owner; each object retains its own mesh and pose.
        self.context = (
            shared_with.context if shared_with is not None else dr.RasterizeCudaContext()
        )
        # 固定网格直径采样，避免物体顺序影响裁剪与体素化；随后恢复调用方 RNG。
        random_state = np.random.get_state()
        try:
            np.random.seed(0)
            self.estimator = estimator_module.FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                glctx=self.context,
                scorer=shared_with.estimator.scorer if shared_with is not None else None,
                refiner=shared_with.estimator.refiner if shared_with is not None else None,
                debug=0,
                debug_dir=str(Path(debug_dir).resolve()),
            )
        finally:
            np.random.set_state(random_state)

    def register(self, rgb, depth_m, intrinsics, mask):
        """Initialize pose from an explicitly supplied object mask on the first frame."""
        self.intrinsics = None
        if np.count_nonzero(mask & (depth_m >= 0.001)) < 4:
            raise ValueError("Registration requires at least four masked valid depth pixels")
        # An upstream early return must not preserve a previous registration.
        self.estimator.pose_last = None
        pose = self.estimator.register(
            K=intrinsics,
            rgb=rgb,
            depth=depth_m,
            ob_mask=mask,
            iteration=self.register_iterations,
        )
        if self.estimator.pose_last is None:
            raise RuntimeError("FoundationPose registration did not initialize tracking")
        pose = validate_pose(pose)
        self.intrinsics = intrinsics
        return pose

    def track(self, rgb, depth_m, intrinsics, mask, *, prepared_frame):
        """Mask-guided FoundationPose++ core, without Kalman or pose confidence."""
        if self.intrinsics is None:
            raise RuntimeError("Register an object before tracking")
        try:
            if np.count_nonzero(mask & (depth_m >= 0.001)) < 4:
                raise ValueError("Tracking requires current exclusive mask and valid depth")
            import torch

            previous = self.estimator.pose_last
            guided = guide_centered_pose(
                previous.detach().cpu().numpy().reshape(4, 4), intrinsics, mask
            )
            self.estimator.pose_last = torch.as_tensor(
                guided, dtype=previous.dtype, device=previous.device
            ).reshape(previous.shape)
            estimator = self.estimator
            # Refine centered-mesh poses using the upstream call and this frame's shared tensors.
            pose, _ = estimator.refiner.predict(
                **prepared_frame,
                K=intrinsics,
                iteration=self.track_iterations,
                ob_in_cams=estimator.pose_last.reshape(1, 4, 4).detach().cpu().numpy(),
                normal_map=None,
                mesh=estimator.mesh,
                mesh_tensors=estimator.mesh_tensors,
                mesh_diameter=estimator.diameter,
                glctx=estimator.glctx,
                get_vis=False,
            )
            estimator.pose_last = pose
            return validate_pose(
                (pose @ estimator.get_tf_to_centered_mesh()).detach().cpu().numpy().reshape(4, 4)
            )
        except Exception:
            self.intrinsics = None
            raise

    @property
    def provenance(self):
        return dict(
            method="FoundationPose++ core: mask 2D bbox guidance + FoundationPose refinement",
            guidance_revision=PLUS_REVISION,
            kalman=False,
            depth_guidance="previous centered-mesh z",
            bbox="inclusive mask min/max midpoint",
            foundationpose_root=str(self.root),
            register_iterations=self.register_iterations,
            track_iterations=self.track_iterations,
            tracking_preprocessing="shared per source frame; upstream erosion/bilateral radius=2 and XYZ",
            object_initialization_numpy_seed=0,
            model_diameter_m=float(self.estimator.diameter),
            pose_frame="camera_from_original_asset; upstream centered mesh stays internal",
        )
