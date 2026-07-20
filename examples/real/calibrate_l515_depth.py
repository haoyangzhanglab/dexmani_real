#!/usr/bin/env python3
"""L515 depth calibration session: thermal drift + sigma_z(z) fit + laser/gain A/B.

Measures, on a STATIC desk scene (robot arm OUT of camera view):

  Phase 1 — drift + sigma (default 25 min, unattended; start from a COLD camera
  for a meaningful warm-up curve):
    - a raw-depth burst every interval -> desk median-depth drift curve
      (thermal warm-up time + drift canary threshold)
    - a final dense capture -> per-pixel temporal std binned by z
      -> DepthEdgeConfig.sigma_poly least-squares fit
  Phase 2 — laser_power / receiver_gain A/B (~1 min, warm camera):
    - per config: fill ratio, per-gate rejection rates (confidence / IR-low /
      IR-saturation / edge), temporal noise, host latency
    - NOTE: L515 receiver_gain numeric semantics are inverted vs the Viewer
      slider (18 = lowest actual gain); community short-range rec is L93/G18.

The driver's validity gate is configured as a no-op (thresholds 0 / None) so
the confidence + IR streams are enabled while raw depth stays untouched; all
gate statistics are recomputed here with the production thresholds. Alignment
is disabled (raw depth domain); the color stream stays on to mirror the
production USB/thermal load.

Usage:
  conda activate real_robot
  python examples/real/calibrate_l515_depth.py [--duration-min 25] [--skip-drift] [--skip-ab]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dexmani_real.sensor.realsense import L515DepthConfig, RealSense, RealSenseConfig
from dexmani_real.utils.pointcloud_utils import (
    DepthEdgeConfig,
    DepthValidityConfig,
    build_edge_threshold_lut,
    compute_depth_edge_mask,
    compute_depth_valid_mask,
)

# Production acquisition config (mirrors examples/real/test_pointcloud_process.py;
# the L515DepthConfig defaults are the validated production values).
PRODUCTION_L515 = L515DepthConfig()

# No-op validity gate: thresholds that reject nothing. Purpose: make
# create_rs_config() enable the confidence + IR streams while the raw depth
# buffer stays untouched (we bypass read(), so _apply_depth_validity never runs).
GATE_NOOP = DepthValidityConfig(confidence_min=0, ir_min=0, ir_saturation=None, saturation_dilate_px=0, edge=None)

# Production gate thresholds, recomputed in-script for statistics.
# confidence_min pre-shifted <<4 exactly like the driver runtime copy.
PROD_GATE = DepthValidityConfig(confidence_min=2 << 4, ir_min=2, ir_saturation=250, saturation_dilate_px=3, edge=None)
# sigma_poly calibrated 2026-07-15 (SN f1382055, warm-camera smoke run).
EDGE_CFG = DepthEdgeConfig(sigma_poly=(-0.00094, 0.00293), n_sigma=5.0, t_min=0.010, t_max=None, dilate_px=0)

DEPTH_RANGE_M = (0.3, 1.5)  # stats ROI, matches the camera-frame depth gate

# (label, laser_power, receiver_gain)
AB_CONFIGS = [
    ("current L100/G12", 100, 12),
    ("low-gain L100/G18", 100, 18),
    ("short-range L93/G18", 93, 18),
]


def _stream_data(frames: rs.composite_frame, stream: rs.stream, shape: tuple[int, ...]) -> np.ndarray | None:
    frame = frames.first_or_default(stream)
    if not frame:
        return None
    data = np.asanyarray(frame.get_data())
    return data if data.shape == shape else None


def capture_burst(
    camera: RealSense,
    n_frames: int,
    lut: np.ndarray,
    depth_scale: float,
    *,
    drain: int = 20,
    pixel_stats: bool = False,
) -> dict:
    """Capture n_frames raw frames and aggregate per-frame / per-pixel statistics.

    `drain` frames are discarded first — the frames_queue (size 16) holds stale
    frames after idle gaps, and it doubles as settle time after option changes.
    """
    pipeline = camera.pipeline
    if pipeline is None:
        raise RuntimeError("Camera pipeline is unavailable.")
    for _ in range(drain):
        pipeline.wait_for_frames(5000)

    raw_lo = int(round(DEPTH_RANGE_M[0] / depth_scale))
    raw_hi = int(round(DEPTH_RANGE_M[1] / depth_scale))

    acc_sum = acc_sumsq = acc_cnt = None
    medians: list[float] = []
    rates: dict[str, list[float]] = {k: [] for k in ("fill_roi", "conf_low", "ir_low", "ir_sat", "edge")}
    latency: list[float] = []
    captured = 0

    for _ in range(n_frames):
        frames = pipeline.wait_for_frames(5000)
        host_ms = time.time() * 1e3
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        depth = np.asanyarray(depth_frame.get_data())
        conf = _stream_data(frames, rs.stream.confidence, depth.shape)
        ir = _stream_data(frames, rs.stream.infrared, depth.shape)
        latency.append(host_ms - float(depth_frame.get_timestamp()))

        pos = depth > 0
        n_pos = max(int(pos.sum()), 1)
        roi = pos & (depth >= raw_lo) & (depth <= raw_hi)
        rates["fill_roi"].append(float(roi.mean()))
        if roi.any():
            medians.append(float(np.median(depth[roi])) * depth_scale)
        if conf is not None:
            rates["conf_low"].append(float(((conf < PROD_GATE.confidence_min) & pos).sum()) / n_pos)
        if ir is not None:
            rates["ir_low"].append(float(((ir < PROD_GATE.ir_min) & pos).sum()) / n_pos)
            rates["ir_sat"].append(float(((ir >= PROD_GATE.ir_saturation) & pos).sum()) / n_pos)
        # Edge rate on production-gated depth (same order as the driver).
        valid = compute_depth_valid_mask(depth, confidence=conf, ir=ir, config=PROD_GATE)
        gated = depth * valid  # uint16 * bool -> uint16
        edge = compute_depth_edge_mask(gated, lut, dilate_px=EDGE_CFG.dilate_px)
        rates["edge"].append(float(edge.sum()) / max(int((gated > 0).sum()), 1))

        if pixel_stats:
            if acc_sum is None:
                acc_sum = np.zeros(depth.shape, dtype=np.float64)
                acc_sumsq = np.zeros(depth.shape, dtype=np.float64)
                acc_cnt = np.zeros(depth.shape, dtype=np.int32)
            df = depth.astype(np.float64)
            acc_sum += df * pos
            acc_sumsq += df * df * pos
            acc_cnt += pos
        captured += 1

    if captured == 0 or not medians:
        raise RuntimeError("Burst captured no usable depth frames.")

    result = {
        "n_frames": captured,
        "median_depth_m": float(np.median(medians)),
        "lat_med_ms": float(np.median(latency)),
        "lat_p95_ms": float(np.percentile(latency, 95)),
    }
    for key, values in rates.items():
        result[key] = float(np.mean(values)) if values else None
    if pixel_stats and acc_sum is not None:
        full = acc_cnt == captured  # pixels valid in every frame
        mean_raw = np.divide(acc_sum, acc_cnt, out=np.zeros_like(acc_sum), where=acc_cnt > 0)
        var_raw = np.divide(acc_sumsq, acc_cnt, out=np.zeros_like(acc_sum), where=acc_cnt > 0) - mean_raw**2
        std_raw = np.sqrt(np.clip(var_raw, 0.0, None))
        result["pixel_stats"] = {"mean_raw": mean_raw, "std_raw": std_raw, "full": full}
    return result


def fit_sigma_poly(pixel_stats: dict, depth_scale: float) -> dict | None:
    """Fit sigma_z(z) = c0 + c1*z from per-pixel temporal std, binned by z (25 mm bins)."""
    z = pixel_stats["mean_raw"] * depth_scale
    s = pixel_stats["std_raw"] * depth_scale
    mask = pixel_stats["full"] & (z > 0.25) & (z < 2.0)
    if int(mask.sum()) < 5000:
        print(f"  sigma fit skipped: only {int(mask.sum())} stable pixels in range.")
        return None
    z_sel, s_sel = z[mask], s[mask]
    edges = np.arange(z_sel.min(), z_sel.max() + 0.025, 0.025)
    idx = np.digitize(z_sel, edges)
    centers, med_std, counts = [], [], []
    for i in range(1, len(edges)):
        in_bin = idx == i
        n = int(in_bin.sum())
        if n >= 500:
            centers.append(float((edges[i - 1] + edges[i]) / 2))
            med_std.append(float(np.median(s_sel[in_bin])))
            counts.append(n)
    if len(centers) < 4 or (max(centers) - min(centers)) < 0.2:
        print(f"  sigma fit skipped: z coverage too narrow ({len(centers)} bins).")
        return None
    c1, c0 = np.polyfit(centers, med_std, 1, w=np.sqrt(counts))
    if c1 < 0:
        print(f"  WARNING: fitted slope is negative (c1={c1:.5f}) — fit is suspect, check the scene.")
    table = [{"z_m": zc, "sigma_mm": sm * 1e3, "count": n} for zc, sm, n in zip(centers, med_std, counts)]
    return {"c0": float(c0), "c1": float(c1), "bins": table}


def run_drift_phase(camera: RealSense, lut: np.ndarray, depth_scale: float, args: argparse.Namespace) -> dict:
    print(
        "\n=== Phase 1: thermal drift + sigma_z(z) "
        f"({args.duration_min:.0f} min, burst every {args.interval_s:.0f} s) ==="
    )
    print("    t+min   median depth      fill(ROI)   latency med/p95")
    n_bursts = int(args.duration_min * 60 / args.interval_s) + 1
    t0 = time.monotonic()
    bursts: list[dict] = []
    consecutive_failures = 0
    reconnects: list[float] = []

    for k in range(n_bursts):
        target = t0 + k * args.interval_s
        while time.monotonic() < target:
            time.sleep(min(1.0, target - time.monotonic()))
        elapsed_min = (time.monotonic() - t0) / 60.0
        try:
            burst = capture_burst(camera, args.burst_frames, lut, depth_scale)
            consecutive_failures = 0
        except RuntimeError as error:
            consecutive_failures += 1
            print(f"  t+{elapsed_min:6.1f}  burst failed: {error}")
            if consecutive_failures >= 2:
                print("  two consecutive failures — reconnecting camera (drift curve perturbed here).")
                camera.disconnect()
                if not camera.connect():
                    print("  reconnect failed — aborting phase 1 with partial data.")
                    break
                reconnects.append(elapsed_min)
                consecutive_failures = 0
            continue
        burst["t_min"] = elapsed_min
        bursts.append(burst)
        print(
            f"  t+{elapsed_min:6.1f}   {burst['median_depth_m']:.4f} m   "
            f"{100 * burst['fill_roi']:8.1f}%   {burst['lat_med_ms']:.0f}/{burst['lat_p95_ms']:.0f} ms"
        )

    result: dict = {
        "bursts": [{k: v for k, v in b.items() if k != "pixel_stats"} for b in bursts],
        "reconnects_at_min": reconnects,
    }
    if len(bursts) >= 5:
        meds = np.array([b["median_depth_m"] for b in bursts])
        times = np.array([b["t_min"] for b in bursts])
        final = float(meds[-5:].mean())
        settled = np.abs(meds - final) <= 1e-3  # within 1 mm of final
        t_settle = None
        for i in range(len(meds)):
            if settled[i:].all():
                t_settle = float(times[i])
                break
        total_drift_mm = (final - float(meds[0])) * 1e3
        resid_mm = float(meds[settled].std() * 1e3) if settled.sum() >= 3 else None
        canary_mm = max(1.0, 3.0 * resid_mm) if resid_mm is not None else None
        result.update({"total_drift_mm": total_drift_mm, "t_settle_min": t_settle, "canary_mm": canary_mm})
        print(f"\n  Total drift (first -> settled): {total_drift_mm:+.2f} mm (depth-ray direction)")
        print(
            f"  Settle time (within 1 mm of final): "
            f"{'not reached in session' if t_settle is None else f'{t_settle:.0f} min'}"
        )
        if canary_mm is not None:
            print(f"  Suggested desk-depth canary threshold: {canary_mm:.1f} mm (3x post-settle std, floor 1 mm)")
            print("  (world-z canary ~= this x cos(view angle); re-derive if used on RANSAC desk z)")

    print(f"\n  Capturing {args.sigma_frames} frames for sigma_z(z) fit...")
    sigma_burst = capture_burst(camera, args.sigma_frames, lut, depth_scale, pixel_stats=True)
    fit = fit_sigma_poly(sigma_burst["pixel_stats"], depth_scale)
    result["sigma_fit"] = fit
    if fit is not None:
        print(
            f"  Fitted:  sigma_poly=({fit['c0']:.5f}, {fit['c1']:.5f})   "
            f"[current {tuple(round(c, 5) for c in EDGE_CFG.sigma_poly)}]"
        )
        print("    z      sigma fit   sigma cur   T_edge fit   T_edge cur")
        for z_val in (0.5, 0.75, 1.0, 1.25):
            s_fit = fit["c0"] + fit["c1"] * z_val
            s_cur = EDGE_CFG.sigma_poly[0] + EDGE_CFG.sigma_poly[1] * z_val
            t_fit = max(EDGE_CFG.n_sigma * s_fit, EDGE_CFG.t_min)
            t_cur = max(EDGE_CFG.n_sigma * s_cur, EDGE_CFG.t_min)
            print(
                f"    {z_val:.2f}m  {s_fit * 1e3:7.2f} mm  {s_cur * 1e3:7.2f} mm  "
                f"{t_fit * 1e3:8.1f} mm  {t_cur * 1e3:8.1f} mm"
            )
    return result


def run_ab_phase(camera: RealSense, lut: np.ndarray, depth_scale: float, args: argparse.Namespace) -> dict:
    print("\n=== Phase 2: laser_power / receiver_gain A/B (warm camera) ===")
    if camera.profile is None:
        raise RuntimeError("Camera profile is unavailable.")
    sensor = camera.profile.get_device().first_depth_sensor()
    orig_laser = sensor.get_option(rs.option.laser_power)
    orig_gain = sensor.get_option(rs.option.receiver_gain)
    rows = []
    try:
        for label, laser, gain in AB_CONFIGS:
            sensor.set_option(rs.option.laser_power, float(laser))
            sensor.set_option(rs.option.receiver_gain, float(gain))
            burst = capture_burst(camera, 60, lut, depth_scale, drain=25, pixel_stats=True)
            actual = {
                "laser": sensor.get_option(rs.option.laser_power),
                "gain": sensor.get_option(rs.option.receiver_gain),
                "preset": sensor.get_option(rs.option.visual_preset),
            }
            ps = burst.pop("pixel_stats")
            in_range = (
                ps["full"]
                & (ps["mean_raw"] * depth_scale > DEPTH_RANGE_M[0])
                & (ps["mean_raw"] * depth_scale < DEPTH_RANGE_M[1])
            )
            noise_mm = float(np.median(ps["std_raw"][in_range]) * depth_scale * 1e3) if in_range.any() else None
            rows.append(
                {
                    "label": label,
                    "requested": {"laser": laser, "gain": gain},
                    "actual": actual,
                    "noise_mm": noise_mm,
                    **burst,
                }
            )
    finally:
        sensor.set_option(rs.option.laser_power, orig_laser)
        sensor.set_option(rs.option.receiver_gain, orig_gain)
        for _ in range(10):  # settle back
            camera.pipeline.wait_for_frames(5000)

    print("  config                 fill(ROI)  conf_low   ir_low   ir_sat    edge   noise(med)")

    def pct(v: float | None) -> str:
        return "   n/a" if v is None else f"{100 * v:5.2f}%"

    for r in rows:
        noise_str = "n/a" if r["noise_mm"] is None else f"{r['noise_mm']:.2f} mm"
        print(
            f"  {r['label']:<22} {pct(r['fill_roi'])}   {pct(r['conf_low'])}  {pct(r['ir_low'])}  "
            f"{pct(r['ir_sat'])}  {pct(r['edge'])}   {noise_str}"
        )
        if (
            abs(r["actual"]["laser"] - r["requested"]["laser"]) > 0.5
            or abs(r["actual"]["gain"] - r["requested"]["gain"]) > 0.5
        ):
            print(
                f"    WARNING: read-back mismatch, actual laser={r['actual']['laser']:.0f} "
                f"gain={r['actual']['gain']:.0f} (preset={r['actual']['preset']:.0f})"
            )
    print("\n  读法：ir_sat 与 noise 同降 → 采纳该档；ir_sat 降但 noise/conf_low 明显升 → 权衡填充率。")
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-min", type=float, default=25.0)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--burst-frames", type=int, default=30)
    parser.add_argument("--sigma-frames", type=int, default=120)
    parser.add_argument("--skip-drift", action="store_true")
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    print("=" * 70)
    print("L515 depth calibration — requirements:")
    print("  * STATIC desk scene, robot arm OUT of camera view, nothing moving")
    print("  * for the drift curve: start from a COLD camera (idle >= 30 min)")
    print("=" * 70)

    camera = RealSense(
        RealSenseConfig(
            camera_name="realsense",
            depth_resolution=(1024, 768),
            color_resolution=(640, 480),
            fps=30,
            enable_color=True,  # mirror production USB/thermal load; frames ignored
            align_mode="none",  # raw depth domain — no alignment needed here
            enable_global_time=True,
            warmup_frames=5,  # minimal: we want cold-start samples early
            l515_depth_config=PRODUCTION_L515,
            depth_validity=GATE_NOOP,
        )
    )
    print("\nConnecting to RealSense...")
    if not camera.connect():
        raise RuntimeError("Failed to connect to RealSense.")

    results: dict = {}
    try:
        info = camera.get_device_info()
        depth_scale = camera.get_depth_scale()
        lut = build_edge_threshold_lut(depth_scale, EDGE_CFG)
        results["device"] = {**info, "depth_scale": depth_scale}
        print(f"Device: {info.get('name', '')}  SN {info.get('serial', '')}  FW {info.get('firmware', '')}")

        if not args.skip_drift:
            results["drift"] = run_drift_phase(camera, lut, depth_scale, args)
        if not args.skip_ab:
            results["ab"] = run_ab_phase(camera, lut, depth_scale, args)
    finally:
        camera.disconnect()
        print("\nRealSense pipeline stopped cleanly.")

    out_path = Path(args.out_dir) / f"l515_calib_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
