"""Asset/selection validation and result publication; no model or device imports."""

from datetime import datetime
import hashlib
import json
from pathlib import Path


def new_output(path, *, workspace, protected=()):
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    path = path or reports / "cordvip_demo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_path(path, workspace=workspace)
    for source in protected:
        if source is not None and path.is_relative_to(source.expanduser().resolve()):
            raise ValueError("输出不能写入 Raw、RGB-D 输入或物体资产目录内部")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_asset(path):
    """Load one metric asset and its ordered surface samples; verify every file hash."""
    import numpy as np
    from .geometry import load_metric_mesh, load_points

    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / "asset.json").read_text())
    if metadata.get("units") != "m" or metadata.get("frame") != "object":
        raise ValueError("资产必须使用 object frame 和米制单位")
    files = {}
    for name in ("mesh", "points", "normals"):
        file = (path / metadata[f"{name}_file"]).resolve()
        if not file.is_relative_to(path):
            raise ValueError("资产文件必须位于资产目录内")
        if sha256(file) != metadata[f"{name}_sha256"]:
            raise ValueError(f"资产 {name} 与保存的 SHA256 不一致；请重新 prepare")
        files[name] = file
    mesh, points = load_metric_mesh(files["mesh"]), load_points(files["points"])
    # Normals describe the generated surface, not measured contact geometry.
    normals = np.load(files["normals"], allow_pickle=False)
    if (
        normals.dtype.kind not in "fiu"
        or normals.shape != points.shape
        or not np.isfinite(normals).all()
    ):
        raise ValueError("资产法向必须与固定点一一对应，且为有限实数 (N, 3) 数组")
    if not np.allclose(np.linalg.norm(normals.astype(np.float64), axis=1), 1, rtol=0, atol=1e-5):
        raise ValueError("资产法向必须是单位向量；不自动修补或归一化")
    return mesh, points, normals.astype(np.float32), metadata


def load_mask(path, shape):
    import numpy as np

    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        import cv2

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape or mask.dtype not in (np.dtype(bool), np.dtype("uint8")):
        raise ValueError("mask 必须为与起始 RGB-D 同尺寸的 bool/uint8 单通道图")
    return mask > 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty local file: {path}")
    if any(Path(str(path) + suffix).exists() for suffix in (".part", ".aria2", ".incomplete")):
        raise ValueError(f"Unfinished local file: {path}")
    return path


def output_path(path: Path, *, workspace) -> Path:
    path = Path(path).expanduser().resolve()
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    if reports not in path.parents:
        raise ValueError(f"输出目录必须位于 {reports} 下")
    if path.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{path}")
    return path


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_confirmed_selection(directory, mask, metadata, *, mask_file, foreground_rgb=None):
    """One authoritative mask/hash/confirmation writer; RGBA is a build attachment."""
    import numpy as np
    from PIL import Image

    directory = Path(directory)
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("confirmed mask must be a boolean image")
    if metadata.get("selection") not in {"explicit_headless_instance", "gui_confirmed"}:
        raise ValueError("selection requires explicit confirmation")
    if Path(mask_file).name != mask_file:
        raise ValueError("mask_file must name a file inside the selection directory")
    path = directory / mask_file
    if path.suffix == ".npy":
        np.save(path, mask, allow_pickle=False)
    elif path.suffix == ".png":
        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    else:
        raise ValueError("confirmed masks use PNG or NPY")
    result = {**metadata, "mask_file": mask_file, "mask_sha256": sha256(path)}
    if foreground_rgb is not None:
        rgb = np.asarray(foreground_rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
            raise ValueError("foreground RGB must align with the confirmed mask")
        foreground = directory / "foreground_rgba.png"
        Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(foreground)
        result.update(foreground_file=foreground.name, foreground_sha256=sha256(foreground))
    write_json(directory / "selection.json", result)
    return result


def load_initial_selection(directory, frame):
    """Reuse only an explicitly selected mask for the exact original RGB-D frame."""
    import numpy as np
    from .rgbd import rectify_frame

    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "selection.json").read_text())
    if metadata.get("selection") not in {"explicit_headless_instance", "gui_confirmed"}:
        raise ValueError("initial-selection must contain an explicitly confirmed selection")
    if metadata.get("mask_grid") != "original_rgb_grid; not_automatically_rectified":
        raise ValueError("initial-selection mask must use the original source RGB grid")
    expected = dict(
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        K=frame.K,
        distortion_model=frame.distortion_model,
        distortion_coeffs=frame.distortion_coeffs,
        source_id=frame.source_id,
        timestamp_ns=frame.timestamp_ns,
        clock=frame.clock,
    )
    with np.load(directory / "source_rgbd.npz", allow_pickle=False) as saved:
        for key, value in expected.items():
            if key not in saved.files or not np.array_equal(saved[key], np.asarray(value)):
                raise ValueError(
                    f"initial-selection {key} differs from the actual first source frame"
                )
        if "row_index" in saved.files and not np.array_equal(saved["row_index"], frame.row_index):
            raise ValueError("initial-selection row_index differs from the first Raw row")
    for key in ("source_id", "timestamp_ns", "clock"):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"initial-selection metadata {key} differs from the source frame")
    mask_path = (directory / metadata["mask_file"]).resolve()
    if not mask_path.is_relative_to(directory) or sha256(mask_path) != metadata["mask_sha256"]:
        raise ValueError("initial-selection mask path/hash does not match the confirmed selection")
    mask = load_mask(mask_path, frame.depth_m.shape)
    _, depth, _, mask = rectify_frame(frame, mask)
    if np.count_nonzero(mask & (depth > 0)) < 4:
        raise ValueError("initial-selection has fewer than four valid rectified depth samples")
    return mask, dict(
        method="reused_confirmed_selection",
        directory=str(directory),
        selection_sha256=sha256(directory / "selection.json"),
        mask_sha256=metadata["mask_sha256"],
        source_check="exact_rgb_depth_K_distortion_identity_time_clock",
        selected_instance=metadata["selected_instance"],
    )


def write_observation(path, payload):
    """Atomically publish a snapshot; consumers must recheck source freshness."""
    import numpy as np

    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    temporary.replace(path)
