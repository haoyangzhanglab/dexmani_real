"""Same-filesystem publication helpers for repository artifacts."""

from __future__ import annotations

import json
import os
import tempfile
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
    os.rename(source, target)
    return target


def atomic_json_dump(
    obj: object, path: str | Path, *, indent: int = 2, ensure_ascii: bool = True
) -> Path:
    """Atomically replace calibration/config JSON, allowing an existing target.

    Uses mkstemp -> dump -> close -> replace.
    Unlike atomic_publish, overwrites are intentional.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=str(parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=indent, ensure_ascii=ensure_ascii)
        os.replace(temp_name, target)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return target
