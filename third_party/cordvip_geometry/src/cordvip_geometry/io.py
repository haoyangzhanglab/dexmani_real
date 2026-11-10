"""资产与重建文件的读写边界。"""

import json
from pathlib import Path


def existing_file(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty local file: {path}")
    return path


def output_path(path: Path, *, protected=(), allow_existing=False) -> Path:
    requested = Path(path).expanduser()
    path = requested.resolve()
    for source in protected:
        if source is not None and path.is_relative_to(
            Path(source).expanduser().resolve()
        ):
            raise ValueError("输出不能写入受保护输入或资产内部")
    if any((parent / "data.h5").is_file() for parent in (path, *path.parents)):
        raise ValueError("输出不能写入 Raw episode 内部")
    if (requested.is_symlink() and not path.exists()) or (
        not allow_existing and path.exists()
    ):
        raise FileExistsError(f"输出目标已存在，拒绝覆盖：{path}")
    return path


def write_json(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def write_npz(path, arrays):
    """独占写入 NPZ；失败移除自己创建的未完成文件，不覆盖已有文件。"""
    import numpy as np

    path = Path(path)
    # 独占打开失败时尚未拥有文件，不能进入半成品清理。
    stream = path.open("xb")
    try:
        with stream:
            np.savez_compressed(stream, **arrays)
    except BaseException:
        path.unlink()
        raise
