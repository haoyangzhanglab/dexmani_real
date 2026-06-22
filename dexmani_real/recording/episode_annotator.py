"""EpisodeAnnotator — post-hoc episode metadata annotation.

Enables operators to add labels, tags, success flags, and notes to recorded
episodes after the fact, without re-recording.

Ref: data collection loop design — Phase 3 (offline tools).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py

__all__ = ["EpisodeAnnotator"]


class EpisodeAnnotator:
    """Annotate HDF5 episode files with metadata.

    Usage:
        annotator = EpisodeAnnotator()
        annotator.annotate("episode_000.h5", success=True,
                           task_label="pick_place", tags=["good", "slow"])
    """

    @staticmethod
    def annotate(
        h5_path: str | Path,
        *,
        success: bool | None = None,
        task_label: str | None = None,
        operator: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        custom_attrs: dict[str, Any] | None = None,
    ) -> bool:
        """Open an HDF5 episode and write/overwrite metadata attributes.

        Only non-None values are written — existing attributes are preserved.

        Returns True on success, False if the file cannot be opened.
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            return False

        try:
            # Open in r+ mode to modify attributes without touching data
            with h5py.File(str(h5_path), "r+") as f:
                meta = f["meta"] if "meta" in f else f.create_group("meta")

                if success is not None:
                    meta.attrs["success"] = success
                if task_label is not None:
                    meta.attrs["task_label"] = task_label
                if operator is not None:
                    meta.attrs["operator"] = operator
                if tags is not None:
                    meta.attrs["tags"] = ",".join(tags)
                if notes is not None:
                    meta.attrs["notes"] = notes

                if custom_attrs:
                    for k, v in custom_attrs.items():
                        if isinstance(v, (bool, int, float, str, bytes)):
                            meta.attrs[k] = v
                        elif isinstance(v, (list, tuple)):
                            meta.attrs[k] = json.dumps(v)
                        else:
                            meta.attrs[k] = str(v)

            return True
        except (OSError, KeyError, ValueError) as e:
            return False

    @staticmethod
    def read_annotations(
        h5_path: str | Path,
    ) -> dict[str, Any]:
        """Read all metadata attributes from an episode file."""
        h5_path = Path(h5_path)
        if not h5_path.exists():
            return {}

        try:
            with h5py.File(str(h5_path), "r") as f:
                if "meta" not in f:
                    return {}
                return dict(f["meta"].attrs)
        except (OSError, KeyError):
            return {}

    @staticmethod
    def annotate_directory(
        data_dir: str | Path,
        *,
        success: bool | None = None,
        tags: list[str] | None = None,
        filter_task: str | None = None,
    ) -> int:
        """Batch-annotate all episodes in a directory.

        Args:
            data_dir: Directory containing episode_*.h5 files.
            success: Set success flag on all matching episodes.
            tags: Add tags to all matching episodes.
            filter_task: Only annotate episodes with this task_label.

        Returns:
            Number of episodes annotated.
        """
        data_dir = Path(data_dir)
        count = 0
        for h5_path in sorted(data_dir.glob("episode_*.h5")):
            if filter_task is not None:
                existing = EpisodeAnnotator.read_annotations(h5_path)
                if existing.get("task_label") != filter_task:
                    continue
            if EpisodeAnnotator.annotate(
                h5_path,
                success=success,
                tags=tags,
            ):
                count += 1
        return count
