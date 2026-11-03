"""首帧选择与离线结果读写。"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def new_output(path, *, workspace, protected=()):
    reports = (Path(workspace).expanduser().resolve() / "reports").resolve()
    path = path or reports / "cordvip" / datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    path = output_path(path, workspace=workspace, protected=protected)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_mask(path, shape):
    import numpy as np

    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        import cv2

        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if (
        mask is None
        or mask.shape != shape
        or mask.dtype not in (np.dtype(bool), np.dtype("uint8"))
    ):
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
        if source is not None and path.is_relative_to(
            Path(source).expanduser().resolve()
        ):
            raise ValueError("输出不能写入 Raw、RGB-D 输入或物体资产目录内部")
    # Raw 也可能位于 workspace 之外。
    if any((parent / "data.h5").is_file() for parent in (path, *path.parents)):
        raise ValueError("输出不能写入 Raw episode 内部")
    if not allow_existing and (path.exists() or requested.is_symlink()):
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{path}")
    return path


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def write_confirmed_selection(
    directory, mask, metadata, *, mask_file, foreground_rgb=None
):
    import numpy as np

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
        from PIL import Image

        Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    else:
        raise ValueError("confirmed masks use PNG or NPY")
    result = {**metadata, "mask_file": mask_file}
    if foreground_rgb is not None:
        from PIL import Image

        rgb = np.asarray(foreground_rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (*mask.shape, 3):
            raise ValueError("foreground RGB must align with the confirmed mask")
        foreground = directory / "foreground_rgba.png"
        Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(foreground)
        result.update(foreground_file=foreground.name)
    write_json(directory / "selection.json", result)
    return result


def load_initial_selection(directory, frame):
    """按 RGB-D 与相机几何复用已确认 mask；来源标识、行号和时钟仅作说明。"""
    import numpy as np

    from .rgbd import load_frame

    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / "selection.json").read_text())
    if (
        metadata.get("selection") not in {"explicit_headless_instance", "gui_confirmed"}
        or "selected_instance" not in metadata
    ):
        raise ValueError(
            "initial-selection must contain an explicitly confirmed selection"
        )
    if metadata.get("mask_grid") != "original_rgb_grid; not_automatically_rectified":
        raise ValueError("initial-selection mask must use the original source RGB grid")
    saved = load_frame(directory / "source_rgbd.npz")
    for key in (
        "rgb",
        "depth_m",
        "K",
        "distortion_model",
        "distortion_coeffs",
    ):
        if not np.array_equal(getattr(saved, key), getattr(frame, key)):
            raise ValueError(
                f"initial-selection {key} differs from the actual first source frame"
            )
    mask_path = (directory / metadata["mask_file"]).resolve()
    if not mask_path.is_relative_to(directory):
        raise ValueError("initial-selection mask must remain inside its directory")
    return load_mask(mask_path, frame.depth_m.shape)


def write_npz(path, arrays):
    """独占写入 NPZ；失败移除自己创建的未完成文件，不覆盖已有文件。"""
    import numpy as np

    path = Path(path)
    stream = path.open("xb")
    failure = None
    try:
        np.savez_compressed(stream, **arrays)
    except BaseException as error:  # noqa: BLE001 - 释放资源后重抛首错
        failure = error
    try:
        stream.close()
    except BaseException as error:
        if failure is None:
            failure = error
        else:
            logger.exception("NPZ close also failed: %s", path)
    if failure is not None:
        try:
            path.unlink()
        except OSError:
            logger.exception("unfinished NPZ removal also failed: %s", path)
        raise failure
