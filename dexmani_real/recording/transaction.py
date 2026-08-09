"""Same-filesystem durable publication helpers for episode artifacts."""

from __future__ import annotations

import os
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
