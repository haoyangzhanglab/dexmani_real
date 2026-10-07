#!/usr/bin/env python3
"""Create a small explicitly synthetic Raw episode; no device or existing asset access."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.recording.frame import build_episode_frame
from dexmani_real.recording.recorder import AsyncEpisodeRecorder
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.robot import DispatchResult, DispatchStatus
from dexmani_real.runtime.observation import ObservationRow
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry


def create_synthetic_episode(directory: Path) -> Path:
    """Eight nominal 10 Hz rows, a visible planar patch, and missing tactile.

    All states/targets/dispatch/timestamps are fabricated fixtures, identified by
    task, execution path and termination metadata. They are not robot evidence.
    The camera plane projects inside the default workspace above z=0.022 m.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = ExperimentConfig()
    camera = CameraIntrinsics(32, 32, 100.0, 100.0, 16.0, 16.0, "none", (0.0,) * 5)
    geometry = RGBDGeometry(camera, camera, np.eye(4))
    transform = np.eye(4)
    transform[0, 3] = 0.4
    recorder = AsyncEpisodeRecorder(
        directory, control_hz=10, rgb_shape=(32, 32, 3), execution_path="synthetic_offline_v1"
    )
    try:
        recorder.start_episode(
            task_label="synthetic",
            collection_source="teleop",
            camera_geometry=geometry,
            camera_T_xarm_base_from_color=transform,
            depth_scale=0.001,
            handbase_position_eef_m=cfg.hand.T_eef_handbase_pos_xyz,
            handbase_quat_eef_wxyz=cfg.hand.T_eef_handbase_quat_wxyz,
            episode_name="episode_synthetic",
        )
        for index in range(8):
            stamp = 1_000_000_000 + index * 100_000_000
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            arm["qpos"][0] = cfg.arm.home_qpos
            arm["qpos"][0, 0] += index * 0.001
            hand["qpos"][0] = np.deg2rad(cfg.hand.home_qpos_deg)
            arm["timestamp_ns"] = hand["timestamp_ns"] = stamp
            hand["tactile_aggregate"] = hand["tactile_dense"] = np.nan
            rgb = np.full((32, 32, 3), (60, 120, 180), np.uint8)
            depth = np.full((32, 32), 500, np.uint16)
            row = ObservationRow(
                arm,
                hand,
                dict(
                    rgb=rgb,
                    depth=depth,
                    timestamp_ns=stamp,
                    color_frame_number=index + 1,
                    depth_frame_number=index + 1,
                ),
                None,
                None,
                stamp,
            )
            command = RobotCommand(1, arm["qpos"][0].copy(), hand["qpos"][0].copy())
            dispatch = DispatchResult(
                DispatchStatus.ACCEPTED, DispatchStatus.ACCEPTED, timestamp_ns=stamp + 1_000_000
            )
            recorder.add_frame(build_episode_frame(row, command, dispatch))
        saved = recorder.save_episode(
            reason="synthetic_fixture", details={"synthetic": True, "physical_execution": False}
        )
        if saved is None:
            raise RuntimeError("synthetic writer did not publish")
        return saved
    finally:
        recorder.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory only; never overwrite")
    args = parser.parse_args(argv)
    print(create_synthetic_episode(args.output))


if __name__ == "__main__":
    main()
