"""Frozen historical raw v24 -> v25 projection; no runtime dependencies."""

import argparse
import os
from pathlib import Path
import shutil

import h5py
import numpy as np

KEEP_DATASETS = (
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
UNSIGNED_DTYPES = {
    "observation_anchor_monotonic_ns": "uint64",
    "arm_source_monotonic_ns": "uint64",
    "hand_source_monotonic_ns": "uint64",
    "tactile_source_monotonic_ns": "uint64",
    "vr_source_monotonic_ns": "uint64",
    "camera_source_monotonic_ns": "uint64",
    "camera_depth_frame_number": "uint64",
    "camera_color_frame_number": "uint64",
    "flag_frame_status": "uint8",
    "tactile_unit_code": "uint8",
}
KEEP_METADATA = (
    "num_frames",
    "control_hz",
    "task_label",
    "operator",
    "camera_name",
    "camera_serial",
    "camera_type",
    "camera_payload_mode",
    "depth_scale",
    "camera_depth_intrinsics",
    "camera_depth_width",
    "camera_depth_height",
    "camera_depth_distortion_model",
    "camera_depth_distortion_coeffs",
    "camera_color_intrinsics",
    "camera_color_width",
    "camera_color_height",
    "camera_color_distortion_model",
    "camera_color_distortion_coeffs",
    "camera_T_color_from_depth",
    "camera_T_xarm_base_from_color",
    "camera_T_xarm_base_from_depth",
    "camera_T_eef_from_depth",
    "real_git_commit",
    "grid_dt_s",
    "grid_duration_s",
    "wall_duration_s",
    "non_sampled_duration_s",
)


def _verify(directory, version):
    with h5py.File(directory / "data.h5", "r") as data:
        meta = data["meta"].attrs
        if meta.get("schema_version") != version:
            raise ValueError(f"expected raw schema {version}")
        count = int(meta["num_frames"])
        if count < 0:
            raise ValueError("num_frames must be non-negative")
        for name in KEEP_DATASETS:
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


def convert_episode(source: Path, destination: Path) -> Path:
    """Publish a new projected episode, retaining the original and media rows."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError("destination must be outside the source episode")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    _verify(source, 24)
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
            for name in KEEP_METADATA:
                if name in src["meta"].attrs:
                    meta.attrs[name] = src["meta"].attrs[name]
            meta.attrs["schema_version"] = 25
            meta.attrs["converted_from_schema"] = 24
            for name in KEEP_DATASETS:
                if name not in UNSIGNED_DTYPES:
                    src.copy(name, dst)
                    continue
                values = src[name][:]
                dtype = np.dtype(UNSIGNED_DTYPES[name])
                if (
                    values.dtype.kind not in "iu"
                    or np.any(values < 0)
                    or np.any(values > np.iinfo(dtype).max)
                ):
                    raise ValueError(
                        f"{name} cannot be represented losslessly as {dtype}"
                    )
                dst.create_dataset(name, data=values.astype(dtype))
        for name in ("depth.h5", "rgb.mp4"):
            try:
                os.link(source / name, staging / name)
            except OSError as exc:
                if exc.errno != 18:  # POSIX EXDEV; no unrelated errors are hidden.
                    raise
                shutil.copy2(source / name, staging / name)
        _verify(staging, 25)
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
        "source", type=Path, help="one v24 episode or a directory of episodes"
    )
    parser.add_argument(
        "destination", type=Path, help="new episode path or destination episode root"
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
            convert_episode(episode, target)
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
