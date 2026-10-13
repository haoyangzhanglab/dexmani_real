"""Asset and RGB-D file I/O; no model or device imports."""

from datetime import datetime
import hashlib
import json
from pathlib import Path


def _new_output(path, *, workspace, protected=()):
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    path = path or reports / "cordvip_demo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = path.expanduser().resolve()
    if not path.is_relative_to(reports) or path == reports:
        raise ValueError("示例产物必须写到仓库 reports/ 下的新目录")
    for source in protected:
        if source is not None and path.is_relative_to(source.expanduser().resolve()):
            raise ValueError("输出不能写入 Raw、RGB-D 输入或物体资产目录内部")
    path.mkdir(parents=True, exist_ok=False)
    return path


def _asset(path, geometry):
    path = path.expanduser().resolve()
    metadata = json.loads((path / "asset.json").read_text())
    if metadata.get("units") != "m" or metadata.get("frame") != "object":
        raise ValueError("资产必须使用 object frame 和米制单位")
    for name in ("mesh", "points"):
        file = (path / metadata[f"{name}_file"]).resolve()
        if not file.is_relative_to(path):
            raise ValueError("资产文件必须位于资产目录内")
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != metadata[f"{name}_sha256"]:
            raise ValueError(f"资产 {name} 与保存的 SHA256 不一致；请重新 prepare")
    mesh = geometry.load_metric_mesh(path / metadata["mesh_file"])
    points = geometry.load_points(path / metadata["points_file"])
    return mesh, points, metadata


def load_asset_normals(asset_path, points):
    """Load fixed mesh normals corresponding to an asset's ordered points.

    Normals are dimensionless vectors in the object frame. They describe the
    generated mesh surface, not observed or estimated contact normals.
    """
    import numpy as np

    path = Path(asset_path).expanduser().resolve()
    metadata = json.loads((path / "asset.json").read_text())
    if metadata.get("units") != "m" or metadata.get("frame") != "object":
        raise ValueError("资产必须使用 object frame 和米制单位")
    filename, expected_hash = metadata.get("normals_file"), metadata.get("normals_sha256")
    if not isinstance(filename, str) or not filename or not isinstance(expected_hash, str):
        raise ValueError("资产缺少 normals_file / normals_sha256；请重新 prepare")
    file = (path / filename).resolve()
    if not file.is_relative_to(path):
        raise ValueError("资产法向文件必须位于资产目录内")
    if _sha256(file) != expected_hash:
        raise ValueError("资产 normals 与保存的 SHA256 不一致；请重新 prepare")
    points = np.asarray(points)
    if (points.dtype.kind not in "fiu" or points.ndim != 2 or points.shape[1] != 3
            or not len(points) or not np.isfinite(points).all()):
        raise ValueError("资产点必须是有限实数 (N, 3) 数组")
    normals = np.load(file, allow_pickle=False)
    if (normals.dtype.kind not in "fiu" or normals.shape != points.shape
            or not np.isfinite(normals).all()):
        raise ValueError("资产法向必须与固定点一一对应，且为有限实数 (N, 3) 数组")
    if not np.allclose(np.linalg.norm(normals.astype(np.float64), axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("资产法向必须是单位向量；不自动修补或归一化")
    return normals.astype(np.float32)


def _mask(path, shape, np, cv2):
    mask = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape or mask.dtype not in (np.dtype(bool), np.dtype("uint8")):
        raise ValueError("mask 必须为与起始 RGB-D 同尺寸的 bool/uint8 单通道图")
    return mask > 0


def _save_input(path, frame, np):
    payload = dict(rgb=frame.rgb, depth_m=frame.depth_m, K=frame.K,
                   distortion_model=frame.distortion_model,
                   distortion_coeffs=frame.distortion_coeffs, source_id=frame.source_id,
                   timestamp_ns=frame.timestamp_ns, clock=frame.clock)
    if frame.row_index is not None:
        payload["row_index"] = frame.row_index
    np.savez_compressed(path, **payload)


def _save_ply(path, points, np):
    with path.open("x") as stream:
        stream.write("ply\nformat ascii 1.0\ncomment color_camera frame; meters\n")
        stream.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(stream, points, fmt="%.8g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"文件不存在或为空：{path}")
    return path


def _output_path(path: Path, *, workspace) -> Path:
    path = path.expanduser().resolve()
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    if reports not in path.parents:
        raise ValueError(f"输出目录必须位于 {reports} 下")
    if path.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{path}")
    return path


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
