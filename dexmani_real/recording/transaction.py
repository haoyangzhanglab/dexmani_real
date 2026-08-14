"""Same-filesystem durable publication helpers for episode artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def _fsync_path(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_tree(path: str | Path) -> None:
    """Sync a file or every regular file/directory below one episode root."""
    root = Path(path)
    if root.is_file():
        _fsync_path(root)
        return
    if not root.is_dir():
        raise FileNotFoundError(root)
    directories = [root]
    for item in sorted(root.rglob("*")):
        if item.is_file():
            _fsync_path(item)
        elif item.is_dir():
            directories.append(item)
    for directory in reversed(directories):
        _fsync_path(directory)


def atomic_publish(src: str | Path, dst: str | Path) -> Path:
    """Fsync and rename one unpublished artifact without copy fallback."""
    source = Path(src)
    target = Path(dst)
    if source.parent.resolve() != target.parent.resolve():
        raise OSError("temporary and final artifacts must share one parent filesystem")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    fsync_tree(source)
    os.rename(source, target)
    _fsync_path(target.parent)
    return target


def atomic_json_dump(obj: object, path: str | Path, *, indent: int = 2, ensure_ascii: bool = True) -> Path:
    """Atomically write ``obj`` as JSON, overwriting any existing target.

    Mirrors the proven ``mkstemp -> dump -> flush -> fsync -> replace -> fsync
    parent`` sequence.  Unlike ``atomic_publish`` (which refuses to overwrite),
    this is for calibration/config artifacts legitimately overwritten in place:
    a crash can never leave the target truncated or absent.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=indent, ensure_ascii=ensure_ascii)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
        _fsync_path(parent)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target
