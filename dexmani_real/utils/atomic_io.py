"""Same-filesystem publication helpers for repository artifacts."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
import tempfile
from concurrent.futures import CancelledError
from pathlib import Path


def target_is_occupied(path: str | Path) -> bool:
    """Treat files, directories, and dangling symlinks as occupied targets."""
    target = Path(path)
    return target.exists() or target.is_symlink()


def atomic_publish(src: str | Path, dst: str | Path) -> Path:
    """原子发布到未占用的同目录目标；取消决策由调用方同步。"""
    source = Path(src)
    target = Path(dst)
    if source.parent.resolve() != target.parent.resolve():
        raise OSError("temporary and final artifacts must share one parent filesystem")
    # RENAME_NOREPLACE atomically checks occupancy; exists() cannot prevent publication races.
    if sys.platform != "linux":
        raise OSError(errno.ENOSYS, "atomic publication requires Linux renameat2")
    source_bytes, target_bytes = os.fsencode(source), os.fsencode(target)
    if b"\0" in source_bytes or b"\0" in target_bytes:
        raise ValueError("embedded null byte")
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "atomic publication requires libc renameat2") from exc
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # AT_FDCWD = -100; RENAME_NOREPLACE = 1. No overwrite-capable fallback.
    if rename(-100, source_bytes, -100, target_bytes, 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    return target


def atomic_yaml_dump(obj: object, path: str | Path) -> Path:
    """原子替换同一运行记录，失败时保留上一份已发布内容。"""
    import yaml

    target = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            yaml.safe_dump(obj, stream, allow_unicode=True, sort_keys=False)
        os.replace(temporary, target)
    except BaseException:
        os.unlink(temporary)
        raise
    return target


def atomic_json_dump(
    obj: object,
    path: str | Path,
    *,
    ensure_ascii: bool = True,
    cancelled=None,
    overwrite: bool = True,
) -> Path:
    """原子写入有限 JSON；overwrite=False 时禁止覆盖已有产物。"""
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, ensure_ascii=ensure_ascii, allow_nan=False)
        # Cancellation after this check cannot promise rollback of a published replacement.
        if cancelled is not None and cancelled():
            raise CancelledError("JSON publication cancelled")
        if overwrite:
            os.replace(temp_name, target)
        else:
            atomic_publish(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target
