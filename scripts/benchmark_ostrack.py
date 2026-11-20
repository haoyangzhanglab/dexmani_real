"""用真实 Raw RGB-D 回放测量 OSTrack 与完整 HOI Worker。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from ctypes import c_byte
from dataclasses import replace
from multiprocessing import get_context
from multiprocessing.synchronize import Lock
from pathlib import Path
from typing import cast

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/cordvip_geometry/src")]

from cordvip_geometry.camera import RGBDRectifier

from dexmani_real.config.real import load_config
from dexmani_real.config.sensor import CameraConfig
from dexmani_real.data.raw import RawReader
from dexmani_real.geometry.hoi import load_hoi_task
from dexmani_real.ipc.ring import SharedRing
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.perception.hoi import build_tracking_config

STAGES = (
    "foundationpose_track", "foundationpose_register", "ostrack_track",
    "sam2_initialize", "sam2_track", "sam2_segment_boxes", "sam2_correct_masks",
    "dino_detect",
)
TELEMETRY_DTYPE = np.dtype([
    ("published_ns", "i8"), ("service_ns", "i8"), ("lag", "i8"),
    ("valid", "?"), ("peak_bytes", "i8"),
    ("stage_ns", "i8", (len(STAGES),)), ("stage_calls", "i8", (len(STAGES),)),
], align=True)
MIB = 1024**2


def load_replay(episode, count, capture_fps):
    with RawReader(episode) as reader:
        camera = reader.read_camera()
        timing = reader.read_timing(source_clocks=("camera_ns",))
        sequences = timing["camera_sequence"]
        unique = np.ones(sequences.shape, dtype=bool)
        unique[1:] = sequences[1:] != sequences[:-1]
        stamps = timing["camera_ns"][unique][:count]
        if len(stamps) < count:
            raise ValueError(f"Need {count} distinct RGB-D frames, found {len(stamps)}")
        if capture_fps is None:
            effective = json.loads(reader.meta.get("effective_parameters", "{}"))
            capture_fps = effective.get("camera_config", {}).get("fps")
        if type(capture_fps) is not int or capture_fps <= 0:
            raise ValueError("Recorded camera FPS must be a positive integer or overridden with --camera-fps")
        depth = reader["depth"]
        frames = []
        for index, rgb in enumerate(reader.iter_rgb_frames()):
            if unique[index]:
                frames.append((rgb, depth[index]))
                if len(frames) == count:
                    break
    return camera, frames, stamps, capture_fps


def summarize_ms(values):
    if not len(values):
        return {"samples": 0}
    return {
        "samples": len(values), "mean_ms": float(np.mean(values) / 1e6),
        "p50_ms": float(np.percentile(values, 50) / 1e6),
        "p95_ms": float(np.percentile(values, 95) / 1e6),
    }


def benchmark_adapter(args, camera, frames, mesh_paths, boxes, checkpoint):
    import torch
    from cordvip_geometry.ostrack import OSTrackAdapter

    boxes = {name: boxes[name] for name in mesh_paths if name in boxes}
    rectifier = RGBDRectifier(
        frames[0][1].shape, camera.color.matrix(), camera.color.distortion_model,
        camera.color.distortion_coeffs,
    )
    images = [rectifier.apply(rgb, depth)[0] for rgb, depth in frames]

    if args.mode == "resident":
        from cordvip_geometry.tracking import SceneTracker

        # 在适配器完成计时前持有模型，单独测量 OSTrack 的推理时间。
        _resident_models = SceneTracker(mesh_paths, workspace=ROOT)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic_ns()
    adapter = OSTrackAdapter(ROOT / "third_party/OSTrack", ROOT / Path(checkpoint).expanduser())
    torch.cuda.synchronize()
    loaded = time.monotonic_ns()
    for name, box in boxes.items():
        adapter.initialize(name, images[0], box)
    torch.cuda.synchronize()
    initialized = time.monotonic_ns()

    elapsed = []
    for index, rgb in enumerate(images[1:], start=1):
        torch.cuda.synchronize()
        started_frame = time.monotonic_ns()
        for name in boxes:
            adapter.track(name, rgb)
        torch.cuda.synchronize()
        if index > args.warmup:
            elapsed.append(time.monotonic_ns() - started_frame)
    return {
        "mode": args.mode, "device": torch.cuda.get_device_name(),
        "object_count": len(boxes), "model_load_ms": (loaded - started) / 1e6,
        "initialize_ms": (initialized - loaded) / 1e6,
        "ostrack_frame": summarize_ms(elapsed), "fps": len(elapsed) * 1e9 / sum(elapsed),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
    }


def instrumented_worker(channels, camera, task, config, raw_telemetry, warmup, profile):
    import torch
    from cordvip_geometry import tracking

    from dexmani_real.perception.hoi_worker import run_hoi_worker

    logging.disable(logging.WARNING)
    telemetry = np.frombuffer(raw_telemetry, dtype=TELEMETRY_DTYPE)
    read = channels.camera_ring.read
    publish = channels.hoi_ring.write
    current_sequence = 0
    started = 0

    def read_camera(**kwargs):
        nonlocal current_sequence, started
        start = time.monotonic_ns()
        snapshot = read(**kwargs)
        if snapshot is not None and snapshot[1] > current_sequence:
            if current_sequence <= warmup < snapshot[1]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            current_sequence = snapshot[1]
            started = start
        return snapshot

    def publish_hoi(**fields):
        torch.cuda.synchronize()
        sequence = int(fields["source_camera_sequence"])
        row = telemetry[sequence - 1]
        publication = publish(**fields)
        stamp = time.monotonic_ns()
        row["service_ns"] = stamp - started
        row["lag"] = channels.camera_ring.latest_sequence - sequence
        row["valid"] = fields["valid"]
        row["peak_bytes"] = torch.cuda.max_memory_allocated()
        row["published_ns"] = stamp
        return publication

    def instrument(owner, name, stage):
        index = STAGES.index(stage)
        operation = getattr(owner, name)

        def timed(*args, **kwargs):
            if current_sequence <= warmup:
                return operation(*args, **kwargs)
            torch.cuda.synchronize()
            start = time.monotonic_ns()
            try:
                return operation(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                row = telemetry[current_sequence - 1]
                row["stage_ns"][index] += time.monotonic_ns() - start
                row["stage_calls"][index] += 1

        setattr(owner, name, timed)

    if profile:
        instrument(tracking, "track_objects", "foundationpose_track")
        instrument(tracking.FoundationPoseTracker, "register", "foundationpose_register")
        for name in ("initialize", "track", "segment_boxes", "correct_masks", "detect"):
            stage = "dino_detect" if name == "detect" else f"sam2_{name}"
            instrument(tracking.SAM2Segmenter, name, stage)
        if config.tracker_2d == "ostrack":
            instrument(tracking.OSTrackAdapter, "track", "ostrack_track")
    channels.camera_ring.read = read_camera
    channels.hoi_ring.write = publish_hoi
    template, mesh_paths, prompts = task
    run_hoi_worker(channels, camera, template, mesh_paths, prompts, config)


def wait_until(condition, process, description, timeout_s=120):
    deadline = time.monotonic() + timeout_s
    while not condition():
        if not process.is_alive():
            raise RuntimeError(f"HOI Worker exited during {description}: {process.exitcode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"HOI Worker timed out during {description}")
        time.sleep(0.0005)


def replay_worker(args, camera, frames, stamps, fps, task, cfg):
    paired = args.mode == "paired"
    if camera.T_xarm_base_from_color is None:
        raise ValueError("HOI Worker replay requires recorded camera extrinsics")
    cfg = build_tracking_config(cfg)
    ctx = get_context("spawn")
    channels = SensorChannels(
        ctx, camera=CameraConfig(width=camera.color.width, height=camera.color.height, fps=fps),
        include_derived_rgb=not args.no_derived_rgb, hoi=True,
    )
    camera_ring = cast(SharedRing, channels.camera_ring)
    hoi_ready = cast(c_byte, channels.hoi_ready)
    hoi_request = cast(SharedRing, channels.hoi_request)
    raw_telemetry = ctx.RawArray("b", len(frames) * TELEMETRY_DTYPE.itemsize)
    telemetry = np.frombuffer(raw_telemetry, dtype=TELEMETRY_DTYPE)
    worker = ctx.Process(
        target=instrumented_worker,
        args=(channels, camera, task, cfg, raw_telemetry, args.warmup, paired),
    )
    submitted = np.zeros(len(frames), dtype=np.int64)
    period_ns = round(1e9 / fps)

    def offer(index, *, deadline=None, wait=False):
        if deadline is not None:
            remaining = deadline - time.monotonic_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
        sent = time.monotonic_ns()
        submitted[index] = sent
        rgb, depth = frames[index]
        source_ns = int(stamps[index]) if paired else sent
        camera_ring.write(timestamp_ns=source_ns, rgb=rgb, depth=depth)
        if wait:
            wait_until(lambda: telemetry[index]["published_ns"], worker, f"frame {index + 1}")

    worker.start()
    try:
        wait_until(lambda: hoi_ready.value, worker, "startup")
        with cast(Lock, channels.hoi_request_lock):
            hoi_request.write(episode_id=1, after_ns=0, after_sequence=0)
        for index in range(args.warmup):
            deadline = submitted[index - 1] + period_ns if not paired and index else None
            offer(index, deadline=deadline, wait=True)
        pace_start = time.monotonic_ns()
        if not paired and args.warmup:
            pace_start = max(pace_start, submitted[args.warmup - 1] + period_ns)
        for index in range(args.warmup, len(frames)):
            deadline = None if paired else pace_start + (index - args.warmup) * period_ns
            offer(index, deadline=deadline, wait=paired)
        wait_until(lambda: telemetry[-1]["published_ns"], worker, "last frame")
    finally:
        channels.is_running.value = False
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
    if worker.exitcode != 0:
        raise RuntimeError(f"HOI Worker exited during shutdown: {worker.exitcode}")

    rows = telemetry[args.warmup:]
    recorded = rows["published_ns"] > 0
    processed = rows[recorded]
    valid = int(processed["valid"].sum())
    elapsed_s = (processed[-1]["published_ns"] - submitted[args.warmup]) / 1e9
    if not paired:
        elapsed_s = max(elapsed_s, len(rows) / fps)
    stats = {
        "mode": args.mode, "tracker_2d": cfg.tracker_2d, "camera_fps": fps,
        "object_count": len(task[1]), "include_derived_rgb": not args.no_derived_rgb,
        "warmup_frames": args.warmup, "offered_frames": len(rows),
        "processed_frames": len(processed), "valid_frames": valid,
        "camera_ring_drops": int((~recorded).sum()),
        "processed_fps": len(processed) / elapsed_s, "valid_fps": valid / elapsed_s,
        "hoi_service": summarize_ms(processed["service_ns"]),
        "camera_to_hoi_publish": summarize_ms(
            processed["published_ns"] - submitted[args.warmup:][recorded]
        ),
        "peak_allocated_mib": float(processed["peak_bytes"].max() / MIB),
    }
    if args.warmup:
        stats["startup_warmup_peak_allocated_mib"] = float(
            telemetry[:args.warmup]["peak_bytes"].max() / MIB
        )
    if cfg.tracker_2d == "ostrack":
        stats["xy_prior_gain"] = cfg.xy_prior_gain
    if paired:
        stages = {}
        for index, name in enumerate(STAGES):
            used = processed["stage_calls"][:, index] > 0
            stage_ns = processed["stage_ns"][used, index]
            calls = int(processed["stage_calls"][:, index].sum())
            stages[name] = {**summarize_ms(stage_ns), "calls": calls}
            if calls:
                stages[name]["mean_ms_per_call"] = float(stage_ns.sum() / calls / 1e6)
        stats["stages_per_frame"] = stages
    else:
        scheduled = pace_start + np.arange(len(rows)) * period_ns
        stats["producer_lateness"] = summarize_ms(submitted[args.warmup:] - scheduled)
        windows = []
        for window in np.array_split(rows, min(3, len(rows))):
            lag = window["lag"][window["published_ns"] > 0]
            windows.append({
                "camera_ring_drops": len(window) - len(lag),
                "camera_lag_p95_frames": float(np.percentile(lag, 95)) if len(lag) else None,
            })
        stats["paced_windows"] = windows
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("solo", "resident", "paired", "paced"))
    parser.add_argument("episode", type=Path)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tracker", choices=("sam2_video", "ostrack"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--xy-prior-gain", type=float)
    parser.add_argument("--box", nargs=5, action="append", default=[], metavar=("NAME", "X1", "Y1", "X2", "Y2"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--no-derived-rgb", action="store_true")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.frames <= 0:
        parser.error("Require nonnegative warmup and positive frame count")
    cfg = load_config(args.config).hoi_tracking
    overrides = {
        "tracker_2d": args.tracker,
        "ostrack_checkpoint": args.checkpoint,
        "xy_prior_gain": args.xy_prior_gain,
    }
    cfg = replace(cfg, **{name: value for name, value in overrides.items() if value is not None})
    standalone = args.mode in ("solo", "resident")
    task = load_hoi_task(args.task_dir)
    if standalone and not args.box:
        parser.error("Specify at least one --box NAME X1 Y1 X2 Y2 for solo and resident modes")
    if not standalone and args.box:
        parser.error("--box applies to solo and resident modes")
    boxes = {}
    for name, *values in args.box:
        if name in boxes or name not in task[1]:
            parser.error(f"--box object {name!r} must appear once and belong to the task")
        try:
            boxes[name] = tuple(map(float, values))
        except ValueError:
            parser.error(f"--box {name!r} requires four numeric coordinates")
    count = args.warmup + args.frames + int(standalone)
    camera, frames, stamps, fps = load_replay(args.episode, count, args.camera_fps)
    if standalone:
        stats = benchmark_adapter(args, camera, frames, task[1], boxes, cfg.ostrack_checkpoint)
    else:
        stats = replay_worker(args, camera, frames, stamps, fps, task, cfg)
    stats.update({
        "image_size": [camera.color.width, camera.color.height],
        "recorded_camera_delta": summarize_ms(np.diff(stamps)),
    })
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
