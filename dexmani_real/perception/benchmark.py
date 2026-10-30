"""对当前生产点云管线做离线录像或实机采集评测。"""

from __future__ import annotations

import importlib.metadata
import json
import resource
import time
from contextlib import closing
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.data.raw import RawReader
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.geometry.pose import validate_rigid_transform
from dexmani_real.perception import pointcloud

CAMERA_CONTROL_NAMES = (
    "visual_preset",
    "digital_gain",
    "alternate_ir",
    "laser_power",
    "receiver_gain",
    "confidence_threshold",
    "noise_filtering",
    "min_distance",
    "pre_processing_sharpening",
    "post_processing_sharpening",
    "invalidation_bypass",
    "sensor_mode",
)


def read_camera_controls(sensor):
    import pyrealsense2 as rs

    return {
        name: sensor.get_option(getattr(rs.option, name))
        for name in CAMERA_CONTROL_NAMES
        if sensor.supports(getattr(rs.option, name))
    }


def capture_live_session(
    runtime, output, *, frame_count=180, warmup_frames=30, calibration_path=None
):
    """采集唯一深度帧；停止后恢复完整设备 JSON，并验证实际控制值。"""
    import pyrealsense2 as rs

    from dexmani_real.sensor.realsense import RealSenseCamera
    from dexmani_real.utils.camera_calibrate.extrinsics import load_camera_extrinsics
    from dexmani_real.utils.table_plane import load_table_plane

    runtime.camera.validate()
    if frame_count < 1 or warmup_frames < 0:
        raise ValueError("frame_count must be positive and warmup_frames nonnegative")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = rs.context()
    devices = list(context.query_devices())
    if runtime.camera.serial is None:
        if len(devices) != 1:
            raise RuntimeError("Expected exactly one RealSense camera")
        device = devices[0]
    else:
        device = next(
            (
                item
                for item in devices
                if item.get_info(rs.camera_info.serial_number) == runtime.camera.serial
            ),
            None,
        )
        if device is None:
            raise RuntimeError(f"Camera {runtime.camera.serial} not found")
    sensor = device.first_depth_sensor()
    original_json = rs.serializable_device(device).serialize_json()
    original_controls = read_camera_controls(sensor)
    (output / "original_settings.json").write_text(original_json)
    metadata = {
        "source": "live_realsense",
        "camera_payload_mode": "depth_to_color_aligned_rgbd",
        "requested_camera_config": asdict(runtime.camera),
        "original_controls": original_controls,
        "timestamp_semantics": "oldest_channel_first_host_receive_monotonic",
    }
    camera = RealSenseCamera(runtime.camera, auto_exposure_priority=None)
    frames, ages, waits = [], [], []
    try:
        camera.connect()
        active = camera.profile.get_device().first_depth_sensor()
        transform = load_camera_extrinsics(camera.active_serial, calibration_path)
        plane = (
            load_table_plane(runtime.environment.table)
            if runtime.pointcloud.remove_table
            else None
        )
        metadata.update(
            serial=camera.active_serial,
            device=camera.get_device_info(),
            intrinsics=camera.color_intrinsics.to_dict(),
            depth_scale_m=camera.depth_scale_m,
            T_xarm_base_from_color=transform.tolist(),
            table_plane_abcd=plane,
            actual_controls=read_camera_controls(active),
        )
        discarded, last_depth = 0, None
        deadline = time.monotonic() + max(
            20, (warmup_frames + frame_count) / runtime.camera.fps * 3
        )
        while len(frames) < frame_count:
            started = time.perf_counter_ns()
            frame = camera.read()
            ended = time.perf_counter_ns()
            if frame is not None and frame.depth_frame_number != last_depth:
                last_depth = frame.depth_frame_number
                if discarded < warmup_frames:
                    discarded += 1
                else:
                    frames.append(frame)
                    waits.append((ended - started) / 1e6)
                    ages.append((time.monotonic_ns() - frame.timestamp_ns) / 1e6)
            if time.monotonic() > deadline:
                raise TimeoutError(f"Only {len(frames)} frames captured")
        metadata.update(
            actual_controls_after_capture=read_camera_controls(active),
            frame_count=len(frames),
            capture_duration_s=(frames[-1].timestamp_ns - frames[0].timestamp_ns) / 1e9,
        )
        print(
            json.dumps(
                {
                    "captured": len(frames),
                    "actual_controls": metadata["actual_controls"],
                }
            ),
            flush=True,
        )
    except BaseException as error:
        metadata["capture_error"] = repr(error)
        raise
    finally:
        try:
            camera.disconnect()
        except BaseException as error:
            metadata["disconnect_error"] = repr(error)
            raise
        finally:
            try:
                rs.serializable_device(device).load_json(original_json)
                restored = read_camera_controls(sensor)
                mismatch = {
                    name: [value, restored[name]]
                    for name, value in original_controls.items()
                    if not np.isclose(value, restored[name], rtol=0, atol=1e-6)
                }
                (output / "restoration.json").write_text(
                    json.dumps(
                        {
                            "before": original_controls,
                            "after": restored,
                            "mismatch": mismatch,
                        },
                        indent=2,
                    )
                )
                if mismatch:
                    raise RuntimeError(
                        f"Camera controls restoration mismatch: {mismatch}"
                    )
            except BaseException as error:
                metadata["restoration_error"] = repr(error)
                raise
            finally:
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(
        output / "frames.npz",
        depth=np.stack([frame.depth for frame in frames]),
        rgb=np.stack([frame.rgb for frame in frames]),
        timestamp_ns=np.array([frame.timestamp_ns for frame in frames]),
        depth_frame_number=np.array([frame.depth_frame_number for frame in frames]),
        color_frame_number=np.array([frame.color_frame_number for frame in frames]),
        age_at_read_ms=np.array(ages),
        read_wait_ms=np.array(waits),
    )
    return output


def read_roi(path):
    if path is None:
        return {}
    roi = json.loads(path.read_text())
    for entry in [roi, *roi.get("sources", {}).values()]:
        for name, dimensions in (("target_boxes", 2), ("noise_boxes", 3)):
            for box in entry.get(name, []):
                values = np.asarray(box, np.float64)
                if (
                    values.shape != (2 * dimensions,)
                    or not np.isfinite(values).all()
                    or np.any(values[:dimensions] >= values[dimensions:])
                ):
                    raise ValueError(f"{name} must contain finite ordered bounds")
    return roi


def episode_frames(path, scan_all):
    """只读取当前 Raw 格式；RGB、depth 与 timing 行必须逐帧对应。"""
    with RawReader(path) as reader:
        reader.require_fields("depth", "rgb")
        camera = reader.read_camera()
        length = reader.num_frames
        depth = reader["depth"]
        if depth.dtype != np.uint16 or depth.shape != (
            length,
            camera.color.height,
            camera.color.width,
        ):
            raise ValueError(f"Raw depth does not match its camera metadata: {path}")
        start = max(0, length // 2 - 32)
        indices = set(range(start, min(start + 64, length))) | set(
            np.linspace(0, length - 1, 8, dtype=int)
        )
        decoded = 0
        with closing(reader.iter_rgb_frames()) as images:
            for index, image in enumerate(images):
                if index >= length:
                    raise ValueError(f"RGB video has more than {length} frames: {path}")
                if image.dtype != np.uint8 or image.shape != (
                    camera.color.height,
                    camera.color.width,
                    3,
                ):
                    raise ValueError(
                        f"Raw RGB does not match its camera metadata: {path}/{index}"
                    )
                decoded += 1
                if scan_all or index in indices:
                    yield index, depth[index], image, camera, {}
        if decoded != length:
            raise ValueError(
                f"RGB video ended at frame {decoded}, expected {length}: {path}"
            )


def session_frames(path):
    """读取 capture_live_session 生成的已对齐 RGB-D 会话。"""
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("camera_payload_mode") != "depth_to_color_aligned_rgbd":
        raise ValueError("Session requires depth-to-color aligned RGB-D")
    camera = CameraSnapshot(
        meta["serial"],
        CameraIntrinsics(**meta["intrinsics"]),
        float(meta["depth_scale_m"]),
        validate_rigid_transform(
            np.asarray(meta["T_xarm_base_from_color"], np.float64)
        ),
    )
    if meta["table_plane_abcd"] is not None:
        plane = np.asarray(meta["table_plane_abcd"], np.float64)
        if plane.shape != (4,) or not np.isfinite(plane).all():
            raise ValueError("Session table plane must have four finite coefficients")
        norm = np.linalg.norm(plane[:3])
        if norm <= 1e-9 or plane[2] / norm <= 0:
            raise ValueError("Session table plane must have an upward non-zero normal")
    with np.load(path / "frames.npz", allow_pickle=False) as arrays:
        depths, images = arrays["depth"], arrays["rgb"]
        length = meta["frame_count"]
        shape = (length, camera.color.height, camera.color.width)
        if (
            depths.dtype != np.uint16
            or depths.shape != shape
            or images.dtype != np.uint8
            or images.shape != (*shape, 3)
        ):
            raise ValueError("Session RGB-D frames do not match their camera metadata")
        if length < 1:
            raise ValueError("Session must contain at least one frame")
        for name in ("timestamp_ns", "depth_frame_number", "color_frame_number"):
            values = arrays[name]
            if values.shape != (length,) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Session {name} must contain one integer per frame")
        timing = {}
        for array_name, label in (
            ("read_wait_ms", "capture_wait"),
            ("age_at_read_ms", "host_receive_to_read"),
        ):
            values = arrays[array_name]
            if (
                values.shape != (length,)
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(
                    f"Session {array_name} must contain one finite nonnegative value per frame"
                )
            timing[label] = {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
        meta["acquisition_timing_ms"] = timing
        for index, (depth, image) in enumerate(zip(depths, images)):
            yield index, depth, image, camera, meta


def noise_count(points, boxes):
    mask = np.zeros(len(points), bool)
    for box in boxes:
        lower, upper = np.asarray(box, np.float32).reshape(2, 3)
        mask |= np.all((points[:, :3] >= lower) & (points[:, :3] <= upper), axis=1)
    return int(mask.sum())


def target_counts(points, intrinsics, transform, boxes):
    if not boxes or not len(points):
        return [0] * len(boxes)
    camera = (points[:, :3] - transform[:3, 3]) @ transform[:3, :3]
    distortion = (
        np.zeros(5)
        if intrinsics.distortion_model in {"none", "distortion.none"}
        else np.asarray(intrinsics.distortion_coeffs)
    )
    pixels = cv2.projectPoints(
        camera, np.zeros(3), np.zeros(3), intrinsics.matrix(), distortion
    )[0].reshape(-1, 2)
    return [
        int(
            np.count_nonzero(
                (pixels[:, 0] >= box[0])
                & (pixels[:, 0] <= box[2])
                & (pixels[:, 1] >= box[1])
                & (pixels[:, 1] <= box[3])
            )
        )
        for box in boxes
    ]


def check_cloud(cloud, num_points):
    if (
        cloud.dtype != np.float32
        or cloud.shape not in ((0, 6), (num_points, 6))
        or not np.isfinite(cloud).all()
    ):
        raise ValueError(f"Invalid point-cloud output: {cloud.dtype} {cloud.shape}")
    if np.any((cloud[:, 3:] < 0) | (cloud[:, 3:] > 1)):
        raise ValueError("RGB outside [0,1]")


def summarize(records):
    times = [row["elapsed_ms"] for row in records]
    summary = {
        "frames": len(records),
        "empty_frames": sum(row["final_points"] == 0 for row in records),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_final_points": float(np.mean([row["final_points"] for row in records])),
    }
    roi_counts = {}
    for source in sorted({row["source"] for row in records}):
        rows = [row for row in records if row["source"] == source]
        counts = {}
        if rows[0]["target_final"]:
            counts["mean_target_final"] = np.mean(
                [row["target_final"] for row in rows], axis=0
            ).tolist()
        if "noise_final" in rows[0]:
            counts["mean_noise_final"] = float(
                np.mean([row["noise_final"] for row in rows])
            )
        if counts:
            roi_counts[source] = counts
    if roi_counts:
        summary["source_roi_counts"] = roi_counts
    profiles = [row for row in records if "stage_ms" in row]
    if profiles:
        summary["stage_ms"] = {
            name: {
                "p50": float(
                    np.percentile([row["stage_ms"][name] for row in profiles], 50)
                ),
                "p95": float(
                    np.percentile([row["stage_ms"][name] for row in profiles], 95)
                ),
            }
            for name in profiles[0]["stage_ms"]
        }
        summary["process_peak_rss_kib_max"] = max(
            row["process_peak_rss_kib"] for row in profiles
        )
    return summary


def run_benchmark(args):
    from dexmani_real.utils.table_plane import (
        load_table_plane,
        resolve_table_plane_path,
    )

    runtime = load_config(args.config)
    output = args.output_dir or Path(
        "outputs/pointcloud_study"
    ) / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    roi = read_roi(args.roi_file)
    if args.benchmark == "live":
        session = (
            capture_live_session(
                runtime, output / "capture", calibration_path=args.camera_calibration
            )
            if args.live_session is None
            else args.live_session.expanduser().resolve()
        )
        paths = [session]
        offline_plane = None
        plane_path = None
    else:
        available = sorted(
            path
            for path in args.episode_root.expanduser().glob("episode_*")
            if (path / "data.h5").is_file()
        )
        if not available:
            raise ValueError(f"No episodes found in {args.episode_root}")
        indices = (
            np.linspace(0, len(available) - 1, min(8, len(available)))
            .round()
            .astype(int)
        )
        paths = (
            available
            if args.scan_all
            else [available[index] for index in sorted(set(indices))]
        )
        offline_plane = (
            load_table_plane(runtime.environment.table)
            if runtime.pointcloud.remove_table
            else None
        )
        plane_path = (
            str(resolve_table_plane_path(runtime.environment.table))
            if runtime.pointcloud.remove_table
            else None
        )
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": args.benchmark,
        "scan_all": args.scan_all,
        "profile_stages": args.profile_stages,
        "profile_note": (
            "阶段计时来自同一次生产调用，elapsed_ms包含计时开销；crop包含XYZRGB构造。"
            "Linux ru_maxrss单位KiB，包含读盘、解码和输入数组的进程历史内存峰值。"
        ),
        "sources": [str(path) for path in paths],
        "offline_table_plane_abcd": offline_plane,
        "plane_path": plane_path,
        "source_geometry": {},
        "pointcloud_config": asdict(runtime.pointcloud),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "opencv-python", "fpsample", "av", "h5py")
        },
        "software_constants": {
            name: value.tolist() if isinstance(value, np.ndarray) else value
            for name, value in vars(pointcloud).items()
            if name.isupper()
        },
        "roi": roi,
        "quality_note": "只统计最终点云的人工目标ROI与独立噪声ROI点数；未标注时不推断质量。",
    }
    (output / "current_source.py").write_bytes(Path(pointcloud.__file__).read_bytes())
    records = []
    for path in paths:
        source_roi = roi.get("sources", {}).get(path.name, roi)
        boxes, noise_boxes = (
            source_roi.get("target_boxes", []),
            source_roi.get("noise_boxes", []),
        )
        frame_iterator = (
            session_frames(path)
            if args.benchmark == "live"
            else episode_frames(path, args.scan_all)
        )
        processed = 0
        with closing(frame_iterator) as frames:
            for index, depth, rgb, camera, meta in frames:
                if not processed:
                    plane = (
                        meta["table_plane_abcd"]
                        if args.benchmark == "live"
                        else offline_plane
                    )
                    processor = pointcloud.PointCloudProcessor(
                        depth_scale_m=camera.depth_scale_m,
                        intrinsics=camera.color,
                        T_xarm_base_from_color=camera.T_xarm_base_from_color,
                        table_plane_abcd=plane,
                        config=runtime.pointcloud,
                    )
                    geometry = {
                        "serial": camera.serial,
                        "intrinsics": camera.color.to_dict(),
                        "depth_scale_m": camera.depth_scale_m,
                        "T_xarm_base_from_color": camera.T_xarm_base_from_color.tolist(),
                        "table_plane_abcd": plane,
                    }
                    if args.benchmark == "live":
                        geometry.update(
                            actual_controls=meta["actual_controls"],
                            original_controls=meta["original_controls"],
                            acquisition_timing_ms=meta["acquisition_timing_ms"],
                        )
                    manifest["source_geometry"][path.name] = geometry
                    processor.process(depth, rgb)
                stages = {} if args.profile_stages else None
                started = time.perf_counter_ns()
                cloud = processor.process(depth, rgb, timings_ms=stages)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                check_cloud(cloud, runtime.pointcloud.num_points)
                row = {
                    "source": path.name,
                    "frame": index,
                    "elapsed_ms": elapsed,
                    "final_points": len(cloud),
                    "target_final": target_counts(
                        cloud, camera.color, camera.T_xarm_base_from_color, boxes
                    ),
                }
                if noise_boxes:
                    row["noise_final"] = noise_count(cloud, noise_boxes)
                if stages is not None:
                    row.update(
                        stage_ms=stages,
                        process_peak_rss_kib=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    )
                records.append(row)
                if not processed and not args.scan_all:
                    np.savez_compressed(
                        output / f"{path.name}_frame{index}.npz",
                        depth=depth,
                        rgb=rgb,
                        current=cloud,
                    )
                processed += 1
        if not processed:
            raise ValueError(f"No frames found in {path}")
        print(
            json.dumps({"source": path.name, "processed_frames": processed}), flush=True
        )
    summary = summarize(records)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    (output / "frames.json").write_text(json.dumps(records, ensure_ascii=False))
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(
        json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False),
        flush=True,
    )
    return 0
