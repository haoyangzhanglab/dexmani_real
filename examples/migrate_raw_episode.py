#!/usr/bin/env python3
"""Migrate clean legacy Raw v34 teleop episodes to the current Raw format.

This is an explicit one-time offline migration tool, not a runtime compatibility
layer. It refuses legacy episodes whose validity evidence cannot be represented
losslessly by the current clean-teleop Raw contract.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import h5py
import numpy as np

from dexmani_real.recording.storage.hdf5_writer import EpisodeDataWriter
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import DATASET_SPECS, RAW_FORMAT
from dexmani_real.recording.storage.video import VideoDecoder
from dexmani_real.utils.atomic_io import atomic_publish

_LEGACY_SCHEMA_VERSION = 34
_CHUNK_ROWS = 32
_MIGRATED_TERMINATION_REASON = "legacy_v34_migrated_clean"


@dataclass(frozen=True)
class LegacyEpisode:
    path: Path
    num_frames: int
    depth_shape: tuple[int, int]


def _scalar_attr(attrs, name):
    if name not in attrs:
        raise ValueError(f"legacy metadata is missing {name}")
    value = np.asarray(attrs[name])
    if value.shape != ():
        raise ValueError(f"legacy metadata {name} must be scalar")
    return value.item()


def _text_attr(attrs, name):
    value = _scalar_attr(attrs, name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"legacy metadata {name} must be a non-empty string")
    return value


def _discover_episode_dirs(root: Path) -> tuple[Path, ...]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    if (resolved / "data.h5").is_file():
        return (resolved,)
    episodes = tuple(
        sorted(
            path
            for path in resolved.iterdir()
            if path.is_dir() and path.name.startswith("episode_")
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode_* directories found in {resolved}")
    return episodes


def _classify_episode(path: Path) -> str:
    data_path = path / "data.h5"
    if not data_path.is_file():
        raise FileNotFoundError(f"{path}: missing data.h5")
    with h5py.File(data_path, "r") as data:
        meta = data.get("meta")
        if not isinstance(meta, h5py.Group):
            raise ValueError(f"{path}: missing meta group")
        attrs = meta.attrs
        raw_format = attrs.get("format")
        if isinstance(raw_format, bytes):
            raw_format = raw_format.decode("utf-8")
        if raw_format == RAW_FORMAT:
            return "current"
        version = _scalar_attr(attrs, "schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or int(version) != _LEGACY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"{path}: expected current format={RAW_FORMAT!r} or legacy schema v34"
            )
    return "legacy"


def _inspect_legacy_episode(path: Path) -> LegacyEpisode:
    data_path = path / "data.h5"
    depth_path = path / "depth.h5"
    rgb_path = path / "rgb.mp4"
    for required in (data_path, depth_path, rgb_path):
        if not required.is_file():
            raise FileNotFoundError(f"{path}: missing {required.name}")
    if rgb_path.stat().st_size == 0:
        raise ValueError(f"{path}: rgb.mp4 is empty")

    with h5py.File(data_path, "r") as data, h5py.File(depth_path, "r") as depth_file:
        meta = data.get("meta")
        if not isinstance(meta, h5py.Group):
            raise ValueError(f"{path}: missing meta group")
        attrs = meta.attrs
        if int(_scalar_attr(attrs, "schema_version")) != _LEGACY_SCHEMA_VERSION:
            raise ValueError(f"{path}: not a legacy Raw v34 episode")
        if _text_attr(attrs, "collection_source") != "teleop":
            raise ValueError(
                f"{path}: migration supports clean legacy teleop demonstrations only"
            )

        episode_valid = _scalar_attr(attrs, "episode_valid")
        if not isinstance(episode_valid, (bool, np.bool_)) or not bool(episode_valid):
            raise ValueError(f"{path}: legacy episode_valid is not true")

        num_frames = _scalar_attr(attrs, "num_frames")
        if (
            isinstance(num_frames, bool)
            or not isinstance(num_frames, (int, np.integer))
            or int(num_frames) <= 0
        ):
            raise ValueError(f"{path}: legacy num_frames must be positive")
        num_frames = int(num_frames)

        allowed = {"meta", "frame_valid", *DATASET_SPECS}
        unexpected = set(data.keys()) - allowed
        if unexpected:
            raise ValueError(
                f"{path}: refusing to drop unexpected legacy datasets {sorted(unexpected)}"
            )

        frame_valid = data.get("frame_valid")
        if (
            not isinstance(frame_valid, h5py.Dataset)
            or frame_valid.shape != (num_frames,)
            or frame_valid.dtype != np.bool_
            or not np.all(frame_valid[:])
        ):
            raise ValueError(f"{path}: every legacy frame_valid must be true")

        for name, spec in DATASET_SPECS.items():
            array = data.get(name)
            if (
                not isinstance(array, h5py.Dataset)
                or array.shape != (num_frames, *spec.tail_shape)
                or array.dtype != spec.dtype
            ):
                raise ValueError(f"{path}: legacy {name} has incompatible shape/dtype")

        for name in (
            "arm_qpos",
            "hand_qpos",
            "action_arm_joint_target",
            "action_hand_joint_target",
        ):
            if not np.isfinite(data[name][:]).all():
                raise ValueError(f"{path}: legacy {name} contains non-finite core values")

        if set(depth_file.keys()) != {"depth"}:
            raise ValueError(f"{path}: legacy depth.h5 must contain only depth")
        depth = depth_file["depth"]
        height = int(_scalar_attr(attrs, "camera_color_height"))
        width = int(_scalar_attr(attrs, "camera_color_width"))
        if depth.shape != (num_frames, height, width) or depth.dtype != np.uint16:
            raise ValueError(
                f"{path}: legacy depth must be uint16 {(num_frames, height, width)}"
            )

    with VideoDecoder(rgb_path) as decoder:
        if decoder.frame_count != num_frames:
            raise ValueError(
                f"{path}: RGB frame count {decoder.frame_count} != num_frames {num_frames}"
            )

    return LegacyEpisode(path=path, num_frames=num_frames, depth_shape=(height, width))


def _build_current_staging(source: LegacyEpisode, staging: Path) -> None:
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"migration staging already exists: {staging}")
    staging.mkdir(parents=False, exist_ok=False)
    writer = None
    try:
        shutil.copy2(source.path / "rgb.mp4", staging / "rgb.mp4")
        with h5py.File(source.path / "data.h5", "r") as old_data, h5py.File(
            source.path / "depth.h5", "r"
        ) as old_depth:
            old_attrs = dict(old_data["meta"].attrs)

            def write_initial_meta(meta):
                for name, value in old_attrs.items():
                    if name not in {"schema_version", "episode_valid", "format", "termination_reason"}:
                        meta.attrs[name] = value

            writer = EpisodeDataWriter(
                staging / "data.h5",
                write_initial_meta=write_initial_meta,
                depth_shape=source.depth_shape,
            )
            writer.open()
            for start in range(0, source.num_frames, _CHUNK_ROWS):
                end = min(source.num_frames, start + _CHUNK_ROWS)
                writer.append(
                    {
                        name: np.asarray(old_data[name][start:end], dtype=spec.dtype)
                        for name, spec in DATASET_SPECS.items()
                    }
                )
                for index in range(start, end):
                    writer.append_depth(np.asarray(old_depth["depth"][index], dtype=np.uint16))

            def write_final_meta(meta):
                meta.attrs["format"] = RAW_FORMAT
                meta.attrs["num_frames"] = source.num_frames
                meta.attrs["termination_reason"] = _MIGRATED_TERMINATION_REASON

            writer.update_meta(write_final_meta)
            writer.close()
            writer = None

        _validate_current_episode(staging, source.num_frames)
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_current_episode(path: Path, expected_frames: int) -> None:
    with EpisodeReader(path) as reader:
        reader.require_fields(*DATASET_SPECS, "rgb", "depth")
        if reader.num_frames != expected_frames:
            raise ValueError(
                f"{path}: migrated num_frames {reader.num_frames} != {expected_frames}"
            )
    with VideoDecoder(path / "rgb.mp4") as decoder:
        if decoder.frame_count != expected_frames:
            raise ValueError(
                f"{path}: migrated RGB count {decoder.frame_count} != {expected_frames}"
            )


def _rollback(swapped: list[tuple[Path, Path]]) -> None:
    errors = []
    for episode, backup in reversed(swapped):
        try:
            if episode.exists():
                shutil.rmtree(episode)
            os.rename(backup, episode)
        except Exception as exc:
            errors.append(f"{episode}: {exc}")
    if errors:
        raise RuntimeError("migration rollback was incomplete: " + "; ".join(errors))


def migrate(root: Path, *, keep_backups: bool = False) -> tuple[int, int]:
    episodes = _discover_episode_dirs(root)
    legacy: list[LegacyEpisode] = []
    current = 0

    # Full preflight first: do not mutate one episode before every legacy source
    # in the requested task has proved losslessly representable.
    for episode in episodes:
        kind = _classify_episode(episode)
        if kind == "current":
            current += 1
            continue
        legacy.append(_inspect_legacy_episode(episode))

    if not legacy:
        return 0, current

    swapped: list[tuple[Path, Path]] = []
    try:
        for source in legacy:
            parent = source.path.parent
            staging = parent / f".tmp_migrate_{source.path.name}"
            backup = parent / f".backup_v34_{source.path.name}"
            if backup.exists() or backup.is_symlink():
                raise FileExistsError(f"migration backup already exists: {backup}")

            _build_current_staging(source, staging)
            os.rename(source.path, backup)
            try:
                atomic_publish(staging, source.path)
                _validate_current_episode(source.path, source.num_frames)
            except BaseException:
                if source.path.exists():
                    shutil.rmtree(source.path)
                os.rename(backup, source.path)
                raise
            swapped.append((source.path, backup))
    except BaseException:
        _rollback(swapped)
        raise

    if not keep_backups:
        for _, backup in swapped:
            shutil.rmtree(backup)

    return len(swapped), current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration of clean legacy Raw v34 teleop episodes to the current "
            "dexmani.raw layout. The normal reader intentionally has no legacy fallback."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Legacy task directory or one legacy episode, e.g. episodes/pick_place_toy.",
    )
    parser.add_argument(
        "--keep-backups",
        action="store_true",
        help="Keep hidden .backup_v34_* directories after the whole migration succeeds.",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        migrated, already_current = migrate(
            args.input_root,
            keep_backups=args.keep_backups,
        )
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Migration complete: migrated={migrated}, already_current={already_current}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
