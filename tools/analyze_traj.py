#!/usr/bin/env python3
"""Analyze a teleop trajectory NPZ for anomalies, non-smoothness, and root causes."""

import sys
import numpy as np

def analyze_traj(filepath: str):
    data = np.load(filepath)

    t = data["t"]
    wrist_pos = data["wrist_pos"]
    wrist_quat = data["wrist_quat_wxyz"]
    target_pos = data["target_pos"]
    target_quat = data["target_quat_wxyz"]
    actual_eef_pos = data["actual_eef_pos"]
    actual_eef_quat = data["actual_eef_quat_wxyz"]
    arm_qpos_actual = data["arm_qpos_actual"]
    ik_ok = data["ik_ok"]
    wrist_delta = data["wrist_delta"]
    eef_delta = data["eef_delta"]
    target_pos_before_clamp = data["target_pos_before_clamp"]

    N = len(t)
    dt_arr = np.diff(t)
    dt_mean = np.mean(dt_arr)
    dt_std = np.std(dt_arr)

    # ============================================================
    # 1. Timing
    # ============================================================
    print("=" * 72)
    print(" 1. TIMING")
    print("=" * 72)
    print(f" Frames: {N}  Duration: {t[-1] - t[0]:.2f}s  "
          f"Nominal FPS: {1/dt_mean:.1f}  Mean dt: {dt_mean*1000:.1f}ms ± {dt_std*1000:.1f}ms")
    # Large gap detection
    large_gaps = np.where(dt_arr > 3 * dt_mean)[0]
    if len(large_gaps) > 0:
        print(f" Large dt gaps (>3x mean = {3*dt_mean*1000:.1f}ms): {len(large_gaps)}")
        for idx in large_gaps[:15]:
            print(f"   t={t[idx]:.3f}s → t={t[idx+1]:.3f}s  dt={dt_arr[idx]*1000:.1f}ms")
    else:
        print(" No large dt gaps.")

    # ============================================================
    # 2. IK success rate
    # ============================================================
    print()
    print("=" * 72)
    print(" 2. IK SOLVER (ik_ok)")
    print("=" * 72)
    print(f" Success: {np.sum(ik_ok)}/{N} ({100 * np.sum(ik_ok) / N:.1f}%)")
    print(f" Failure: {np.sum(~ik_ok)}/{N}")
    ik_fail = ~ik_ok
    if np.any(ik_fail):
        changes = np.diff(np.concatenate([[False], ik_fail, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        print(f" Failure segments: {len(starts)}")
        for s, e in zip(starts, ends):
            dur_s = t[min(e - 1, N - 1)] - t[s]
            # Check if target pos is clamped during IK failure
            clamp_at_fail = target_pos_before_clamp[min(e-1, N-1)] - target_pos[min(e-1, N-1)]
            clamped = np.linalg.norm(clamp_at_fail) > 1e-4
            near_ws = " (near workspace boundary?)" if clamped else ""
            print(f"   [{s:5d}..{min(e-1,N-1):5d}] t={t[s]:.3f}→{t[min(e-1,N-1)]:.3f}s  "
                  f"dt_total={dur_s*1000:.0f}ms  frames={e-s}{near_ws}")

    # ============================================================
    # 3. Wrist delta (VR raw input smoothness)
    # ============================================================
    print()
    print("=" * 72)
    print(" 3. WRIST DELTA (VR raw input per-frame Δ)")
    print("=" * 72)
    wd_norm = np.linalg.norm(wrist_delta, axis=1)
    print(f" Norm: mean={np.mean(wd_norm)*1000:.2f}mm  std={np.std(wd_norm)*1000:.2f}mm  "
          f"max={np.max(wd_norm)*1000:.2f}mm  P99={np.percentile(wd_norm, 99)*1000:.2f}mm")
    # Identify spikes
    p999 = np.percentile(wd_norm, 99.9)
    spike_idx = np.where(wd_norm > p999)[0]
    if len(spike_idx) > 0:
        print(f" Spikes (>P99.9={p999*1000:.1f}mm): {len(spike_idx)}")
        for idx in spike_idx[:10]:
            print(f"   t={t[idx]:.3f}s  Δ={wd_norm[idx]*1000:.1f}mm  vec(mm)=[{wrist_delta[idx,0]*1000:.1f}, {wrist_delta[idx,1]*1000:.1f}, {wrist_delta[idx,2]*1000:.1f}]")

    # ============================================================
    # 4. EEF target delta smoothness
    # ============================================================
    print()
    print("=" * 72)
    print(" 4. EEF TARGET DELTA (command frame-to-frame Δ)")
    print("=" * 72)
    ed_norm = np.linalg.norm(eef_delta, axis=1)
    print(f" Norm: mean={np.mean(ed_norm)*1000:.2f}mm  std={np.std(ed_norm)*1000:.2f}mm  "
          f"max={np.max(ed_norm)*1000:.2f}mm  P99={np.percentile(ed_norm, 99)*1000:.2f}mm")
    p999e = np.percentile(ed_norm, 99.9)
    espike = np.where(ed_norm > p999e)[0]
    if len(espike) > 0:
        print(f" Spikes (>P99.9={p999e*1000:.1f}mm): {len(espike)}")
        for idx in espike[:10]:
            print(f"   t={t[idx]:.3f}s  Δ={ed_norm[idx]*1000:.1f}mm  vec(mm)=[{eef_delta[idx,0]*1000:.1f}, {eef_delta[idx,1]*1000:.1f}, {eef_delta[idx,2]*1000:.1f}]")

    # ============================================================
    # 5. Target vs Actual EEF tracking
    # ============================================================
    print()
    print("=" * 72)
    print(" 5. TARGET vs ACTUAL EEF TRACKING ERROR")
    print("=" * 72)
    tracking_err = np.linalg.norm(target_pos - actual_eef_pos, axis=1)
    print(f" Error: mean={np.mean(tracking_err)*1000:.1f}mm  std={np.std(tracking_err)*1000:.1f}mm  "
          f"max={np.max(tracking_err)*1000:.1f}mm")
    print(f" P50={np.percentile(tracking_err, 50)*1000:.1f}mm  "
          f"P95={np.percentile(tracking_err, 95)*1000:.1f}mm  "
          f"P99={np.percentile(tracking_err, 99)*1000:.1f}mm")

    p99_err = np.percentile(tracking_err, 99)
    bad_track = np.where(tracking_err > p99_err)[0]
    if len(bad_track) > 0:
        segments = _find_segments(bad_track, gap=5)
        print(f" Large error segments (>P99={p99_err*1000:.1f}mm): {len(segments)}")
        for s, e in segments[:10]:
            maxe = np.max(tracking_err[s:e+1])
            dur = t[min(e, N-1)] - t[s]
            ik_f = np.sum(~ik_ok[s:e+1])
            print(f"   [{s:5d}..{min(e,N-1):5d}] t={t[s]:.3f}→{t[min(e,N-1)]:.3f}s  "
                  f"dur={dur*1000:.0f}ms  max_err={maxe*1000:.1f}mm  ik_fails_in_window={ik_f}")

    # ============================================================
    # 6. Workspace clamping
    # ============================================================
    print()
    print("=" * 72)
    print(" 6. WORKSPACE CLAMPING (target_pos_before_clamp - target_pos)")
    print("=" * 72)
    clamp_vec = target_pos_before_clamp - target_pos
    clamp_norm = np.linalg.norm(clamp_vec, axis=1)
    clamped = clamp_norm > 1e-6
    print(f" Clamped frames: {np.sum(clamped)}/{N} ({100*np.sum(clamped)/N:.1f}%)")
    if np.any(clamped):
        print(f" Clamp mag: mean={np.mean(clamp_norm[clamped])*1000:.1f}mm  "
              f"max={np.max(clamp_norm[clamped])*1000:.1f}mm")
        changes = np.diff(np.concatenate([[False], clamped, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        print(f" Clamp segments: {len(starts)}")
        for s, e in zip(starts, ends):
            maxc = np.max(clamp_norm[s:e])
            dur = t[min(e-1, N-1)] - t[s]
            print(f"   [{s:5d}..{min(e-1,N-1):5d}] t={t[s]:.3f}s  "
                  f"dur={dur*1000:.0f}ms  max_clamp={maxc*1000:.1f}mm")
    else:
        print(" No clamping detected.")

    # ============================================================
    # 7. Arm joint trajectory smoothness
    # ============================================================
    print()
    print("=" * 72)
    print(" 7. ARM JOINT TRAJECTORY (velocity & acceleration)")
    print("=" * 72)
    jvel = np.diff(arm_qpos_actual, axis=0) / dt_arr[:, np.newaxis]
    jvel_norm = np.linalg.norm(jvel, axis=1)
    jacc = np.diff(jvel, axis=0) / dt_arr[1:, np.newaxis]
    jacc_norm = np.linalg.norm(jacc, axis=1)
    print(f" Velocity: mean={np.mean(jvel_norm):.3f}rad/s  max={np.max(jvel_norm):.3f}rad/s  "
          f"P99={np.percentile(jvel_norm, 99):.3f}rad/s")
    print(f" Accel:    mean={np.mean(jacc_norm):.3f}rad/s²  max={np.max(jacc_norm):.3f}rad/s²  "
          f"P99={np.percentile(jacc_norm, 99):.3f}rad/s²")
    print(" Per-joint velocity (rad/s):")
    for j in range(7):
        v = jvel[:, j]
        print(f"   J{j}: mean|v|={np.mean(np.abs(v)):.3f}  std={np.std(v):.3f}  "
              f"max|v|={np.max(np.abs(v)):.3f}")

    # Find high-accel events
    p999a = np.percentile(jacc_norm, 99.9)
    high_a = np.where(jacc_norm > p999a)[0]
    if len(high_a) > 0:
        print(f"\n High accel events (>P99.9={p999a:.3f} rad/s²): {len(high_a)}")
        segs = _find_segments(high_a, gap=3)
        for s, e in segs[:15]:
            maxa = np.max(jacc_norm[s:e+1])
            maxj = np.argmax(np.max(np.abs(jacc[s:e+1]), axis=0))
            print(f"   t~{t[s]:.3f}s  acc={maxa:.3f}rad/s²  max_joint=J{maxj}")

    # ============================================================
    # 8. Correlation analysis
    # ============================================================
    print()
    print("=" * 72)
    print(" 8. ROOT CAUSE ANALYSIS")
    print("=" * 72)

    # 8a. IK failure ↔ large tracking error
    ik_fail_mask = ~ik_ok
    err_ik_ok = tracking_err[ik_ok]
    err_ik_fail = tracking_err[ik_fail_mask] if np.any(ik_fail_mask) else np.array([0])
    print(f" Tracking error when ik_ok=True:  mean={np.mean(err_ik_ok)*1000:.1f}mm")
    print(f" Tracking error when ik_ok=False: mean={np.mean(err_ik_fail)*1000:.1f}mm")

    # 8b. Clamping ↔ IK failure
    print(f"\n IK failure rate when clamped:   "
          f"{100*np.sum(ik_fail_mask & clamped)/max(np.sum(clamped),1):.1f}%")
    print(f" IK failure rate when unclamped: "
          f"{100*np.sum(ik_fail_mask & ~clamped)/max(np.sum(~clamped),1):.1f}%")

    # 8c. Large wrist delta ↔ IK failure
    p99_wd = np.percentile(wd_norm, 99)
    large_wd = wd_norm > p99_wd
    print(f" IK failure rate with large VR delta:  "
          f"{100*np.sum(ik_fail_mask & large_wd)/max(np.sum(large_wd),1):.1f}%")
    print(f" IK failure rate with normal VR delta: "
          f"{100*np.sum(ik_fail_mask & ~large_wd)/max(np.sum(~large_wd),1):.1f}%")

    # 8d. Large dt gaps ↔ anything
    large_gap_mask = np.zeros(N, dtype=bool)
    for idx in large_gaps:
        large_gap_mask[idx] = True
        large_gap_mask[idx+1] = True
    print(f"\n IK failure at large-dt-gap frames: "
          f"{100*np.sum(ik_fail_mask & large_gap_mask)/max(np.sum(large_gap_mask),1):.1f}%")

    # ============================================================
    # 9. Overall smoothness summary
    # ============================================================
    print()
    print("=" * 72)
    print(" 9. SUMMARY")
    print("=" * 72)

    # Wrist pos smoothness: derivative of wrist_pos over time
    wrist_vel = np.diff(wrist_pos, axis=0) / dt_arr[:, np.newaxis]
    wrist_vel_norm = np.linalg.norm(wrist_vel, axis=1)

    # Target pos smoothness
    target_vel = np.diff(target_pos, axis=0) / dt_arr[:, np.newaxis]
    target_vel_norm = np.linalg.norm(target_vel, axis=1)

    # Actual EEF smoothness
    actual_vel = np.diff(actual_eef_pos, axis=0) / dt_arr[:, np.newaxis]
    actual_vel_norm = np.linalg.norm(actual_vel, axis=1)

    print(f" Wrist (VR) velocity:  mean={np.mean(wrist_vel_norm):.3f}m/s  "
          f"max={np.max(wrist_vel_norm):.3f}m/s  P99={np.percentile(wrist_vel_norm, 99):.3f}m/s")
    print(f" Target EEF velocity:  mean={np.mean(target_vel_norm):.3f}m/s  "
          f"max={np.max(target_vel_norm):.3f}m/s  P99={np.percentile(target_vel_norm, 99):.3f}m/s")
    print(f" Actual EEF velocity:  mean={np.mean(actual_vel_norm):.3f}m/s  "
          f"max={np.max(actual_vel_norm):.3f}m/s  P99={np.percentile(actual_vel_norm, 99):.3f}m/s")

    issues = []
    if np.mean(~ik_ok) > 0.05:
        issues.append(f"IK failure rate {100*np.mean(~ik_ok):.1f}% is high (>5%)")
    elif np.mean(~ik_ok) > 0.01:
        issues.append(f"IK failure rate {100*np.mean(~ik_ok):.1f}% is moderate (>1%)")

    if np.max(jacc_norm) > 50:
        issues.append(f"High joint acceleration spikes (max={np.max(jacc_norm):.1f} rad/s²)")

    if np.max(wd_norm) > 0.05:
        issues.append(f"VR wrist delta spikes (max={np.max(wd_norm)*1000:.0f}mm) — possible tracking glitch")

    if dt_std / dt_mean > 2:
        issues.append(f"Irregular timing (dt std/mean={dt_std/dt_mean:.1f}) — loop jitter or pauses")

    if np.sum(clamped) > 0:
        issues.append(f"Workspace boundary clamping active ({np.sum(clamped)} frames)")

    if issues:
        print("\n Key issues:")
        for i, iss in enumerate(issues):
            print(f"   {i+1}. {iss}")
    else:
        print("\n No major issues detected.")


def _find_segments(indices: np.ndarray, gap: int = 3):
    """Group consecutive indices into segments, splitting when gap exceeds threshold."""
    if len(indices) == 0:
        return []
    segments = []
    seg_start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > gap:
            segments.append((seg_start, indices[i - 1]))
            seg_start = indices[i]
    segments.append((seg_start, indices[-1]))
    return segments


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "trajectories/traj_20260709_194753.npz"
    analyze_traj(path)
