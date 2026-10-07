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


def atomic_publish(src: str | Path, dst: str | Path, *, cancelled=None) -> Path:
    """Rename and publish one unpublished artifact to an unoccupied target."""
    source = Path(src)
    target = Path(dst)
    if source.parent.resolve() != target.parent.resolve():
        raise OSError("temporary and final artifacts must share one parent filesystem")
    if target_is_occupied(target):
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    if cancelled is not None and cancelled():
        raise RuntimeError("artifact publication cancelled")
    # Linux RENAME_NOREPLACE checks occupancy and renames in one operation.
    # A preflight exists() check cannot protect immutable artifacts from races.
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


def atomic_json_dump(
    obj: object,
    path: str | Path,
    *,
    indent: int = 2,
    ensure_ascii: bool = True,
    cancelled=None,
    overwrite: bool = True,
) -> Path:
    """Write JSON atomically; use overwrite=False to claim a new result path."""
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=indent, ensure_ascii=ensure_ascii)
        # Passing this last check admits the irreversible replace; later
        # cancellation cannot promise rollback of a successfully published file.
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
