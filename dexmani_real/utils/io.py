"""实验产物的原子写入与发布。"""

from __future__ import annotations

import ctypes
import json
import os
import tempfile
from concurrent.futures import CancelledError
from pathlib import Path


def publish(src: str | Path, dst: str | Path) -> Path:
    """Linux 上原子发布到未占用的同文件系统目标。"""
    source = Path(src)
    target = Path(dst)
    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if b"\0" in source_bytes or b"\0" in target_bytes:
        raise ValueError("embedded null byte")
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # AT_FDCWD = -100，RENAME_NOREPLACE = 1。
    if rename(-100, source_bytes, -100, target_bytes, 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    return target


def _write_text(text, target, *, cancelled=None, overwrite=True):
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        if cancelled is not None and cancelled():
            raise CancelledError("publication cancelled")
        if overwrite:
            os.replace(temporary, target)
        else:
            publish(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def write_yaml(obj: object, path: str | Path) -> Path:
    import yaml

    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
    return _write_text(text, Path(path))


def write_json(obj: object, path: str | Path, *, cancelled=None, overwrite=True) -> Path:
    """序列化成功后原子发布；取消检查紧邻发布边界。"""
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return _write_text(text, target, cancelled=cancelled, overwrite=overwrite)
