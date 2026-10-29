"""Asset/selection validation and result publication; no model or device imports."""

from datetime import datetime
import json
from pathlib import Path


def new_output(path, *, workspace, protected=()):
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    path = path or reports / "cordvip_demo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_path(path, workspace=workspace, protected=protected)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_asset(path):
    """Load one metric asset and its ordered surface samples."""
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


def existing_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty local file: {path}")
    return path


def output_path(path: Path, *, workspace, protected=(), allow_existing=False) -> Path:
    requested = Path(path).expanduser()
    path = requested.resolve()
    workspace = Path(workspace).expanduser().resolve()
    for source in (workspace / "episodes", workspace / "rollouts", *protected):
        if source is not None and path.is_relative_to(Path(source).expanduser().resolve()):
            raise ValueError("输出不能写入 Raw、RGB-D 输入或物体资产目录内部")
    # Published Raw may also live outside the workspace's default capture roots.
    if any((parent / "data.h5").is_file() for parent in (path, *path.parents)):
        raise ValueError("输出不能写入 Raw episode 内部")
    if not allow_existing and (path.exists() or requested.is_symlink()):
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{path}")
    return path


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_confirmed_selection(directory, mask, metadata, *, mask_file, foreground_rgb=None):
    """Save an explicitly confirmed mask; RGBA is a build attachment."""
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
    result = {**metadata, "mask_file": mask_file}
    if foreground_rgb is not None:
        rgb = np.asarray(foreground_rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
            raise ValueError("foreground RGB must align with the confirmed mask")
        foreground = directory / "foreground_rgba.png"
        Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(foreground)
        result.update(foreground_file=foreground.name)
    write_json(directory / "selection.json", result)
    return result


def load_initial_selection(directory, frame):
    """Reuse only an explicitly selected mask for the exact original RGB-D frame."""
    import numpy as np
    from .rgbd import load_frame, rectify_frame

    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "selection.json").read_text())
    if metadata.get("selection") not in {"explicit_headless_instance", "gui_confirmed"}:
        raise ValueError("initial-selection must contain an explicitly confirmed selection")
    if metadata.get("mask_grid") != "original_rgb_grid; not_automatically_rectified":
        raise ValueError("initial-selection mask must use the original source RGB grid")
    saved = load_frame(directory / "source_rgbd.npz")
    for key in (
        "rgb", "depth_m", "K", "distortion_model", "distortion_coeffs",
        "timestamp_ns", "clock", "row_index",
    ):
        if not np.array_equal(getattr(saved, key), getattr(frame, key)):
            raise ValueError(f"initial-selection {key} differs from the actual first source frame")
    if saved.source_id != frame.source_id:
        saved_raw, current_raw = (source.rsplit(":", 2) for source in (
            saved.source_id, frame.source_id,
        ))
        if not (
            saved.source_id.startswith("raw:") and frame.source_id.startswith("raw:")
            and len(saved_raw) == len(current_raw) == 3
            and saved_raw[1] in {"row", "camera"}
            and saved_raw[1:] == current_raw[1:]
        ):
            raise ValueError("initial-selection source_id differs from the actual first source frame")
    for key in ("source_id", "timestamp_ns", "clock"):
        if metadata.get(key) != getattr(saved, key):
            raise ValueError(f"initial-selection metadata {key} differs from the source frame")
    mask_path = (directory / metadata["mask_file"]).resolve()
    if not mask_path.is_relative_to(directory):
        raise ValueError("initial-selection mask must remain inside its directory")
    mask = load_mask(mask_path, frame.depth_m.shape)
    _, depth, _, mask = rectify_frame(frame, mask)
    if np.count_nonzero(mask & (depth > 0)) < 4:
        raise ValueError("initial-selection has fewer than four valid rectified depth samples")
    return mask, dict(
        method="reused_confirmed_selection",
        directory=str(directory),
        source_check="exact_rgb_depth_K_distortion_identity_without_raw_path_time_clock_row",
        selection_source_id=saved.source_id,
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
