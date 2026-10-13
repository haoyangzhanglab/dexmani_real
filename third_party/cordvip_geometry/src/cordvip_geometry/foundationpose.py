"""Narrow adapter for a separately prepared FoundationPose checkout and environment.

The upstream code uses global imports and initializes CUDA. Construct this class
only in the standalone example process, never in a robot runtime import path.
"""

import importlib
from pathlib import Path
import sys

import numpy as np

from .geometry import validate_pose, validate_rgbd


def validate_foundationpose_root(root):
    """Check source and local weights without importing CUDA or loading a model."""
    root = Path(root).expanduser().resolve()
    required = ["estimater.py", "Utils.py"]
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        required.extend((f"weights/{run}/config.yml", f"weights/{run}/model_best.pth"))
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"FoundationPose source/weights missing under {root}: {', '.join(missing)}")
    return root


class FoundationPoseTracker:
    def __init__(self, root, mesh, debug_dir, register_iterations=5, track_iterations=2,
                 *, shared_with=None):
        self.root = validate_foundationpose_root(root)
        for value in (register_iterations, track_iterations):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Refinement iterations must be positive integers")
        self.register_iterations = register_iterations
        self.track_iterations = track_iterations
        self._registered = False
        self._intrinsics = None
        self._image_shape = None

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("FoundationPose requires CUDA; input validation alone does not run tracking")
        # Upstream uses generic module names. Refuse a mixture of different checkouts.
        for name in ("estimater", "Utils", "learning", "datareader"):
            existing = sys.modules.get(name)
            if existing is not None:
                source = getattr(existing, "__file__", None)
                locations = [source] if source else list(getattr(existing, "__path__", ()))
                if not locations or any(not Path(path).resolve().is_relative_to(self.root) for path in locations):
                    raise RuntimeError(f"Module {name!r} is already loaded outside {self.root}; use a fresh process")
        sys.path.insert(0, str(self.root))
        estimator_module = importlib.import_module("estimater")
        import nvdiffrast.torch as dr

        # Scene objects run sequentially on one owner. Share immutable network
        # weights and raster context, while retaining each object's mesh/pose.
        if shared_with is not None and shared_with.root != self.root:
            raise ValueError("Shared FoundationPose weights must use the same checkout")
        self._context = shared_with._context if shared_with is not None else dr.RasterizeCudaContext()
        self._estimator = estimator_module.FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            glctx=self._context,
            scorer=shared_with._estimator.scorer if shared_with is not None else None,
            refiner=shared_with._estimator.refiner if shared_with is not None else None,
            debug=0,
            debug_dir=str(Path(debug_dir).resolve()),
        )

    def register(self, rgb, depth_m, intrinsics, mask):
        """Initialize pose from an explicitly supplied object mask on the first frame."""
        self._registered = False
        rgb, depth_m, intrinsics, mask = validate_rgbd(rgb, depth_m, intrinsics, mask)
        if mask is None or np.count_nonzero(mask & (depth_m >= 0.001)) < 4:
            raise ValueError("Registration requires at least four masked valid depth pixels")
        # An upstream early return must not preserve a previous registration.
        self._estimator.pose_last = None
        pose = self._estimator.register(
            K=intrinsics, rgb=rgb, depth=depth_m, ob_mask=mask,
            iteration=self.register_iterations,
        )
        if self._estimator.pose_last is None:
            raise RuntimeError("FoundationPose registration did not initialize tracking")
        pose = validate_pose(pose)
        self._intrinsics = intrinsics.copy()
        self._image_shape = depth_m.shape
        self._registered = True
        return pose

    def track(self, rgb, depth_m, intrinsics):
        """Track one chronological frame; returns T_camera_object, without a confidence claim."""
        if not self._registered:
            raise RuntimeError("Register an object before tracking")
        try:
            rgb, depth_m, intrinsics, _ = validate_rgbd(rgb, depth_m, intrinsics)
            if depth_m.shape != self._image_shape or not np.array_equal(intrinsics, self._intrinsics):
                raise ValueError("Image shape/intrinsics changed; start a new explicit registration")
            if np.count_nonzero(depth_m >= 0.001) < 4:
                raise ValueError("Tracking frame contains too little valid depth")
            pose = self._estimator.track_one(
                rgb=rgb, depth=depth_m, K=intrinsics, iteration=self.track_iterations,
            )
            return validate_pose(pose)
        except Exception:
            self._registered = False
            raise
