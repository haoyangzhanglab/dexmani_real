"""CUDA 边界的输入与合批几何等价性；不加载模型权重。"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "third_party/cordvip_geometry/src"),
    str(ROOT / "third_party/FoundationPose"),
    str(ROOT / "third_party/Grounded-SAM-2"),
]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_stream_normalization_matches_original():
    from cordvip_geometry.segmentation import SAM2Segmenter, normalized_rgb_table
    from sam2.utils.misc import process_stream_frame

    segmenter = SAM2Segmenter.__new__(SAM2Segmenter)
    segmenter.device = torch.device("cuda")
    segmenter.tracker = SimpleNamespace(image_size=1024)
    segmenter.state = {"num_frames": 0}
    segmenter.normalization_table = normalized_rgb_table(segmenter.device)
    segmenter.color_channels = torch.arange(3, device="cuda")
    rgb = np.random.default_rng(42).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    reference = process_stream_frame(rgb, 1024)[0].float()
    segmenter._append_frame(rgb)
    actual = segmenter.state["images"][0]
    assert actual.stride() == reference.stride()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("intrinsics_dtype", [np.float32, np.float64])
def test_shared_frame_buffers_refresh_without_changing_preprocessing(intrinsics_dtype):
    from cordvip_geometry.foundationpose import prepare_tracking_frame
    from Utils import bilateral_filter_depth, depth2xyzmap_batch, erode_depth

    shape = (72, 96)
    K = np.array([[130, 0, 47], [0, 135, 35], [0, 0, 1]], dtype=intrinsics_dtype)
    cache = {}
    buffers = None
    for value in (60, 80):
        rgb = np.full((*shape, 3), value, dtype=np.uint8)
        depth = np.full(shape, value / 100, dtype=np.float32)
        depth[10:20] = 0
        actual_rgb, actual_xyz = prepare_tracking_frame(
            rgb, depth, K, (160, 160), cache
        )
        if buffers is None:
            buffers = cache.copy()
        else:
            assert all(cache[key] is tensor for key, tensor in buffers.items())
        expected_depth = torch.as_tensor(depth, device="cuda")
        expected_depth = erode_depth(expected_depth, radius=2, device="cuda")
        expected_depth = bilateral_filter_depth(expected_depth, radius=2, device="cuda")
        expected_xyz = depth2xyzmap_batch(
            expected_depth[None],
            torch.as_tensor(K, device="cuda", dtype=torch.float32)[None],
            zfar=np.inf,
        )[0]
        torch.testing.assert_close(actual_xyz, expected_xyz, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_rgb,
            torch.as_tensor(rgb, device="cuda", dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(cache["K"], torch.as_tensor(K, device="cuda"))


def test_batched_observed_crops_preserve_sampling():
    import nvdiffrast.torch as dr
    import trimesh
    from learning.training.predict_pose_refine import (
        make_crop_data_batch,
        render_crops,
        sample_observed_crops,
    )
    from Utils import make_crop_grid_constants, make_mesh_tensors

    context = dr.RasterizeCudaContext()
    K = np.array([[130.0, 0, 47], [0, 135.0, 35], [0, 0, 1.0]])
    rgb = torch.arange(72 * 96 * 3, device="cuda", dtype=torch.float32)
    rgb = rgb.reshape(72, 96, 3) % 256
    xyz = torch.ones((72, 96, 3), device="cuda") * 0.6
    xyz[10:20] = 0
    poses = torch.eye(4, device="cuda")[None].repeat(3, 1, 1)
    poses[:, 2, 3] = 0.6
    poses[:, 0, 3] = torch.tensor([-0.2, 0, 0.2], device="cuda")
    references, rendered = [], []
    for index, diameter in enumerate((0.08, 0.12, 0.16)):
        kwargs = {
            "render_size": (160, 160),
            "K": K,
            "crop_ratio": 1.2,
            "mesh_diameter": np.float64(diameter),
            "glctx": context,
            "mesh_tensors": make_mesh_tensors(
                trimesh.creation.box(extents=[diameter] * 3)
            ),
        }
        pose = poses[index : index + 1]
        references.append(
            make_crop_data_batch(**kwargs, ob_in_cams=pose, rgb=rgb, xyz_map=xyz)
        )
        rendered.append(render_crops(**kwargs, poses=pose, image_size=rgb.shape[:2]))
    crops = (torch.cat(values) for values in zip(*rendered))
    batch = sample_observed_crops(
        poses,
        rgb,
        xyz,
        *crops,
        (160, 160),
        grid_constants=make_crop_grid_constants(
            (72, 96), (160, 160), poses.device, poses.dtype
        ),
    )
    for field in ("rgbAs", "rgbBs", "xyz_mapAs", "xyz_mapBs"):
        torch.testing.assert_close(
            getattr(batch, field),
            torch.cat([getattr(value, field) for value in references]),
            rtol=0,
            atol=0,
        )
