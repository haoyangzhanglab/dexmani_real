"""只读相机标定采样与离线 eye-to-hand 求解。"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.utils.camera_calibrate.aruco import (
    ARUCO_DICT_NAME, CAPTURE_FRAMES, MARKER_ID, MARKER_SIZE_M, ArucoSampler,
)
from dexmani_real.utils.camera_calibrate.extrinsics import save_camera_calibration
from dexmani_real.utils.camera_calibrate.hand_eye import HandEyeResult, solve_hand_eye
from dexmani_real.utils.io import write_json
from dexmani_real.utils.timing import sample_is_fresh


def read_arm_feedback(arm, max_age_s):
    # 来源时间是这次 SDK 查询开始时刻，不是返回后读取缓存的时刻。
    timestamp_ns = time.monotonic_ns()
    qpos, qvel = arm.read()
    if not sample_is_fresh(timestamp_ns, max_age_s):
        raise TimeoutError("arm feedback query exceeded calibration freshness budget")
    return dict(qpos=qpos, qvel=qvel, timestamp_ns=timestamp_ns)


def save_sample(path, sample_id, sample, eef_pose):
    frames = sample["frames"]
    arrays = dict(
        sample_id=sample_id,
        T_base_from_eef=eef_pose,
        T_camera_from_marker=sample["T_camera_from_marker"],
        rgb=np.stack([frame["rgb"] for frame in frames]),
        camera_timestamp_ns=np.array([frame["timestamp_ns"] for frame in frames], dtype=np.int64),
        color_frame_number=np.array([frame["color_frame_number"] for frame in frames], dtype=np.int64),
        marker_corners_xy=np.stack([frame["corners"] for frame in frames]),
        marker_rvec=np.stack([frame["rvec"] for frame in frames]),
        marker_tvec=np.stack([frame["tvec"] for frame in frames]),
        marker_detected=np.array([frame["detected"] for frame in frames]),
        arm_qpos=np.stack([frame["arm"]["qpos"] for frame in frames]),
        arm_qvel=np.stack([frame["arm"]["qvel"] for frame in frames]),
        arm_timestamp_ns=np.array([frame["arm"]["timestamp_ns"] for frame in frames], dtype=np.int64),
    )
    for stage in ("before", "after"):
        for name, value in sample[f"arm_{stage}"].items():
            arrays[f"arm_{stage}_{name}"] = value
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def collect_samples(runtime, output_dir):
    """相机保持连接；每次摆位完成后只读连接机械臂，采样并立即断开。"""
    from dexmani_real.geometry.kinematics import ArmFK
    from dexmani_real.robot.xarm7.driver import XArm7
    from dexmani_real.sensor.realsense import RealSenseCamera

    runtime.camera.validate()
    criteria = dict(
        capture_frames=CAPTURE_FRAMES,
        min_detected_frames=CAPTURE_FRAMES // 2,
        arm_max_speed_rad_s=runtime.arm.homing.velocity_convergence_rad_s,
        arm_max_drift_rad=runtime.arm.homing.convergence_rad,
        arm_feedback_max_age_s=runtime.arm.feedback_max_age_s,
        camera_max_age_s=runtime.camera.max_frame_age_s,
    )
    for name in ("arm_max_speed_rad_s", "arm_max_drift_rad", "arm_feedback_max_age_s"):
        if not np.isfinite(criteria[name]) or criteria[name] <= 0:
            raise ValueError(f"calibration {name} must be finite and positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    samples = output / "samples"
    samples.mkdir()
    camera = RealSenseCamera(runtime.camera, auto_exposure_priority=None)
    fk = ArmFK()
    count = 0
    with ExitStack() as resources:
        resources.callback(camera.disconnect)
        camera.connect()
        write_json(dict(
            camera=dict(serial=camera.active_serial, color=camera.color_intrinsics.to_dict(),
                        depth_scale_m=camera.depth_scale_m),
            marker=dict(dictionary=ARUCO_DICT_NAME, id=MARKER_ID, size_m=MARKER_SIZE_M),
            criteria=criteria, requested_config=asdict(runtime),
        ), output / "capture.json", overwrite=False)
        sampler = ArucoSampler(camera.color_intrinsics.matrix(),
                               np.asarray(camera.color_intrinsics.distortion_coeffs))
        print("标记须相对 EEF 刚性固定。用厂商示教或独立点动工具摆位，停止并退出控制连接。")
        print("相机保持连接；Enter 只读采样，完成后机械臂连接立即关闭；Q 结束。")
        while True:
            command = input(f"sample {count}: Enter 采样 / Q 结束 > ").strip().lower()
            if command == "q":
                break
            if command:
                continue
            arm = XArm7(runtime.arm)
            try:
                # XArm7.connect 仅建立 SDK/读取连接，不调用 initialize 或任何运动配置。
                arm.connect(check_abort=lambda: None)
                try:
                    sample = sampler.capture(
                        camera.read, lambda: read_arm_feedback(arm, runtime.arm.feedback_max_age_s),
                        camera_config=runtime.camera, arm_config=runtime.arm,
                    )
                except (TimeoutError, ValueError, cv2.error) as exc:
                    print(f"采样拒绝：{exc}；未新增样本，请重新摆位。", flush=True)
                    continue
            finally:
                arm.close()
            path = samples / f"{count:06d}.npz"
            save_sample(path, count, sample, fk.forward(sample["arm_after"]["qpos"]))
            print(f"已保存 sample {count}：{path}", flush=True)
            count += 1
    return count


def solve_samples(sample_dir, output_path, *, exclude=()) -> HandEyeResult:
    """只读采样文件，按原 ID 剔除样本；不导入设备连接或执行在线标定。"""
    source = Path(sample_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    capture = json.loads((source / "capture.json").read_text(encoding="utf-8"))
    camera = capture["camera"]
    intrinsics = CameraIntrinsics(**camera["color"])
    excluded = set(exclude)
    samples, all_ids = {}, set()
    for path in sorted((source / "samples").glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            identity = data["sample_id"]
            if identity.shape != () or identity.dtype.kind not in "iu":
                raise ValueError(f"{path}: sample_id must be a scalar integer")
            sample_id = int(identity)
            if sample_id < 0 or sample_id in all_ids:
                raise ValueError(f"{path}: invalid or duplicate sample_id {sample_id}")
            all_ids.add(sample_id)
            if sample_id not in excluded:
                samples[sample_id] = (
                    validate_rigid_transform(data["T_base_from_eef"], label=f"sample {sample_id} EEF"),
                    validate_rigid_transform(data["T_camera_from_marker"], label=f"sample {sample_id} marker"),
                )
    if excluded - all_ids:
        raise ValueError(f"unknown excluded sample IDs: {sorted(excluded - all_ids)}")
    if not samples:
        raise ValueError("no calibration samples remain")
    used_ids = sorted(samples)
    eef, markers = (np.stack(poses) for poses in zip(*(samples[identity] for identity in used_ids)))
    result = solve_hand_eye(eef, markers)
    report = dict(
        sample_directory=str(source), used_sample_ids=used_ids, excluded_sample_ids=sorted(excluded),
        sample_count=len(used_ids), method=result.method,
        position_errors_mm=result.position_errors_mm.tolist(),
        rotation_errors_deg=result.rotation_errors_deg.tolist(),
        position_rms_mm=result.position_rms_mm, rotation_rms_deg=result.rotation_rms_deg,
        worst_sample_id=used_ids[result.worst_sample], quality_ok=result.quality_ok,
        calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(dict(camera_serial=camera["serial"], T_base_from_camera=result.T_base_from_camera.tolist(),
                    **report), output.with_suffix(".result.json"))
    if not result.quality_ok:
        raise ValueError(
            f"hand-eye quality failed: {result.position_rms_mm:.3f} mm / "
            f"{result.rotation_rms_deg:.3f} deg; worst sample ID {report['worst_sample_id']}"
        )
    save_camera_calibration(
        result.T_base_from_camera, camera["serial"], output,
        calibration_capture=dict(
            width=intrinsics.width, height=intrinsics.height,
            intrinsics=intrinsics.matrix().tolist(), distortion=list(intrinsics.distortion_coeffs),
            **report,
        ),
    )
    print(f"{result.method}: RMS={result.position_rms_mm:.3f} mm / {result.rotation_rms_deg:.3f} deg; "
          f"worst sample ID={report['worst_sample_id']}; 已保存 {output}")
    return result
