"""Frozen historical raw v25 -> v26 tactile representation migration.

v26 restores the XHand SDK-native numeric scale: v25 stored both tactile
payloads multiplied by ``0.1``, so a directly-recorded v25 episode migrates
with ``payload * 10``.  Aggregate/dense freshness are split into independent
``tactile_sum_fresh`` / ``tactile_fresh`` columns; legacy v25 collapsed them,
so ``tactile_sum_fresh`` is copied conservatively from ``tactile_fresh``.

Episodes that were themselves derived from v24 (``converted_from_schema == 24``)
span different code generations, so this tool refuses to guess their scale and
requires an explicit ``--converted-v24-scale`` choice.  No runtime imports.
"""

import argparse
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

# v25 source-row datasets, exactly the frozen v24->v25 projection.  v26 adds
# only ``tactile_sum_fresh``.
V25_DATASETS = (
    "timestamp",
    "source_sample_index",
    "fill_reason",
    "flag_sample_valid",
    "arm_qpos",
    "arm_qvel",
    "arm_tau",
    "hand_qpos",
    "hand_current",
    "hand_contact",
    "hand_tactile_force",
    "arm_connected",
    "hand_connected",
    "hand_qpos_stale",
    "tracking_error",
    "arm_last_cmd_seq",
    "action_arm_joint_sent",
    "action_hand_joint",
    "action_arm_ee",
    "flag_action_queued",
    "flag_frame_status",
    "observation_anchor_monotonic_ns",
    "observation_valid",
    "arm_source_monotonic_ns",
    "hand_source_monotonic_ns",
    "tactile_source_monotonic_ns",
    "vr_source_monotonic_ns",
    "camera_source_monotonic_ns",
    "tactile_fresh",
    "tactile_calibrated",
    "tactile_unit_code",
    "flag_camera_fresh",
    "camera_depth_frame_number",
    "camera_color_frame_number",
    "policy_observation_arm_qpos",
    "policy_observation_hand_qpos",
    "policy_observation_valid",
    "vr_wrist_pos",
    "vr_wrist_rot6d",
    "vr_landmarks",
    "head_quat_wxyz",
)
TACTILE_DATASETS = ("hand_contact", "hand_tactile_force")
NEW_DATASETS = ("tactile_sum_fresh",)
V26_DATASETS = V25_DATASETS + NEW_DATASETS


def _verify(directory, version, datasets):
    with h5py.File(directory / "data.h5", "r") as data:
        meta = data["meta"].attrs
        if meta.get("schema_version") != version:
            raise ValueError(f"expected raw schema {version}")
        count = int(meta["num_frames"])
        if count < 0:
            raise ValueError("num_frames must be non-negative")
        for name in datasets:
            if name not in data:
                raise ValueError(f"missing required dataset: {name}")
            if not isinstance(data[name], h5py.Dataset) or data[name].shape[:1] != (
                count,
            ):
                raise ValueError(f"wrong first dimension: {name}")
    with h5py.File(directory / "depth.h5", "r") as depth:
        if "depth" not in depth or depth["depth"].shape[:1] != (count,):
            raise ValueError("depth frame count does not match num_frames")
    if (
        not (directory / "rgb.mp4").is_file()
        or (directory / "rgb.mp4").stat().st_size == 0
    ):
        raise ValueError("rgb.mp4 must exist and be non-empty")


def _resolve_scale(converted_from_schema, converted_v24_scale):
    """Return the tactile scale factor for one v25 source episode."""
    if converted_from_schema == 24:
        if converted_v24_scale == "legacy-0.1":
            return 10.0
        if converted_v24_scale == "native":
            return 1.0
        raise ValueError(
            "source was derived from v24; pass --converted-v24-scale legacy-0.1 "
            "or --converted-v24-scale native (never guessed from payload magnitude)"
        )
    if converted_v24_scale is not None:
        raise ValueError(
            "--converted-v24-scale only applies to v24-derived sources"
        )
    # Directly-recorded v25 stored 0.1 * (SDK - bias); v26 restores SDK-native.
    return 10.0


def convert_episode(source, destination, converted_v24_scale=None):
    """Publish a scale-corrected v26 episode, retaining the original and media."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError("destination must be outside the source episode")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    _verify(source, 25, V25_DATASETS)
    with h5py.File(source / "data.h5", "r") as src:
        converted_from = int(src["meta"].attrs.get("converted_from_schema", -1) or -1)
    scale = _resolve_scale(converted_from, converted_v24_scale)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive mkdir owns this staging path; never clean another invocation's work.
    staging = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    staging.mkdir()
    try:
        with (
            h5py.File(source / "data.h5", "r") as src,
            h5py.File(staging / "data.h5", "w") as dst,
        ):
            meta = dst.create_group("meta")
            for name in src["meta"].attrs:
                if name in ("schema_version", "converted_from_schema"):
                    continue
                meta.attrs[name] = src["meta"].attrs[name]
            meta.attrs["schema_version"] = 26
            meta.attrs["converted_from_schema"] = 25
            for name in V25_DATASETS:
                if name in TACTILE_DATASETS:
                    dst.create_dataset(name, data=src[name][:] * scale)
                else:
                    src.copy(name, dst)
            # Legacy v25 collapsed aggregate/dense validity; migrate conservatively.
            dst.create_dataset(
                "tactile_sum_fresh", data=np.asarray(src["tactile_fresh"][:], dtype=bool)
            )
        for name in ("depth.h5", "rgb.mp4"):
            try:
                os.link(source / name, staging / name)
            except OSError as exc:
                if exc.errno != 18:  # POSIX EXDEV; no unrelated errors are hidden.
                    raise
                shutil.copy2(source / name, staging / name)
        _verify(staging, 26, V26_DATASETS)
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="one v25 episode or a directory of episodes"
    )
    parser.add_argument(
        "destination", type=Path, help="new episode path or destination episode root"
    )
    parser.add_argument(
        "--converted-v24-scale",
        choices=("legacy-0.1", "native"),
        default=None,
        help="required when a source was itself derived from v24",
    )
    args = parser.parse_args(argv)
    source = args.source.resolve()
    destination = args.destination.absolute()
    if not source.is_dir():
        parser.error(f"source is not a directory: {source}")
    if destination.resolve() == source or source in destination.resolve().parents:
        parser.error("destination must be outside the source")
    episodes = (
        [source]
        if (source / "data.h5").is_file()
        else sorted(
            p for p in source.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
    )
    failures = 0
    for episode in episodes:
        target = destination if episode == source else destination / episode.name
        try:
            convert_episode(episode, target, args.converted_v24_scale)
            print(f"OK {episode} -> {target}")
        except (OSError, ValueError, KeyError) as exc:
            failures += 1
            print(f"FAIL {episode}: {exc}")
    print(
        f"Converted {len(episodes) - failures}; failed {failures}; total {len(episodes)}"
    )
    return int(bool(failures) or not episodes)


if __name__ == "__main__":
    raise SystemExit(main())
