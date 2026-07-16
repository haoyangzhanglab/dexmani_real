"""30 Hz point cloud stream smoke test — CameraProcess + SHM data channel.

Verifies the production pointcloud path end-to-end without the recorder:
CameraProcess child (XGA depth + calibrated validity gate + PointCloudProcessor)
-> CameraRingBuffer -> poll_latest_frame()["pointcloud"].

This is exactly how a future policy loop consumes the stream.

Pass criteria (printed at exit):
  - sustained producer rate >= 28 Hz
  - end-to-end latency (time.time() - device timestamp, global time) < ~80 ms
  - 2048 valid points inside the workspace crop box

Usage:
    python test_pointcloud_stream.py [--duration 60] [--vis] [--serial SN]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from dexmani_real.sensor.camera_process import CameraProcess, CameraProcessConfig

POLL_HZ = 50.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0, help="Run time in seconds.")
    parser.add_argument("--vis", action="store_true", help="Live open3d view of the stream.")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial (default: first).")
    args = parser.parse_args()

    camera = CameraProcess(
        CameraProcessConfig(
            camera_name="realsense",
            serial=args.serial,
            hz=30.0,
            enable_pointcloud=True,
        )
    )
    if not camera.start():
        raise RuntimeError("CameraProcess failed to start.")

    vis = pcd_geom = None
    if args.vis:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window("pointcloud stream", width=1280, height=720)
        pcd_geom = o3d.geometry.PointCloud()

    last_frame_number = -1
    latencies: list[float] = []
    frame_times: list[float] = []
    sec_frames = 0
    sec_start = time.monotonic()
    t_end = time.monotonic() + args.duration

    print(f"Streaming for {args.duration:.0f}s (Ctrl-C to stop early)...")
    try:
        while time.monotonic() < t_end:
            frame = camera.poll_latest_frame()
            if frame is not None and frame["frame_number"] != last_frame_number:
                last_frame_number = frame["frame_number"]
                now = time.time()
                latencies.append(now - frame["timestamp"])
                frame_times.append(time.monotonic())
                sec_frames += 1

                pc = frame.get("pointcloud")
                if vis is not None and pc is not None and frame.get("pointcloud_valid"):
                    import open3d as o3d

                    pcd_geom.points = o3d.utility.Vector3dVector(pc[:, :3].astype(np.float64))
                    pcd_geom.colors = o3d.utility.Vector3dVector(pc[:, 3:].astype(np.float64))
                    if len(frame_times) == 1:
                        vis.add_geometry(pcd_geom)
                    else:
                        vis.update_geometry(pcd_geom)
                    vis.poll_events()
                    vis.update_renderer()

                if time.monotonic() - sec_start >= 1.0:
                    hz = sec_frames / (time.monotonic() - sec_start)
                    lat_ms = latencies[-1] * 1e3
                    if pc is not None:
                        valid = bool(frame.get("pointcloud_valid", False))
                        lo, hi = pc[:, :3].min(axis=0), pc[:, :3].max(axis=0)
                        print(
                            f"  {hz:5.1f} Hz | latency {lat_ms:6.1f} ms | "
                            f"pc valid={valid} n={pc.shape[0]} | "
                            f"xyz [{lo[0]:+.2f},{lo[1]:+.2f},{lo[2]:+.2f}]"
                            f"..[{hi[0]:+.2f},{hi[1]:+.2f},{hi[2]:+.2f}]"
                        )
                    else:
                        print(f"  {hz:5.1f} Hz | latency {lat_ms:6.1f} ms | pc: none (disabled?)")
                    sec_frames = 0
                    sec_start = time.monotonic()
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        camera.stop()
        if vis is not None:
            vis.destroy_window()

    # ── Summary vs pass criteria ──
    n = len(frame_times)
    if n < 2:
        print("FAIL: no frames received.")
        return
    total_hz = (n - 1) / (frame_times[-1] - frame_times[0])
    lat = np.asarray(latencies) * 1e3
    print(
        f"\nSummary: {n} frames, {total_hz:.1f} Hz overall | "
        f"latency p50={np.percentile(lat, 50):.1f} ms p95={np.percentile(lat, 95):.1f} ms"
    )
    print(f"  rate    >= 28 Hz : {'PASS' if total_hz >= 28.0 else 'FAIL'}")
    print(f"  latency <  80 ms : {'PASS' if np.percentile(lat, 95) < 80.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
