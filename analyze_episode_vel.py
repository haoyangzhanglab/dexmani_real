#!/usr/bin/env python3
"""Analyze episode HDF5: joint velocities, speed limit violations, tracking-error windows."""

from __future__ import annotations

import h5py
import numpy as np

FILE = "/home/zhanghaoyang/Desktop/dexmani_real/episodes_arm/episode_20260718_212454.h5"

# Tracking error events from log
ERROR_EVENTS = [
    (8.3, 0.453),
    (9.3, 0.805),
    (10.3, 0.359),
    (12.3, 0.522),
    (25.3, 0.369),
    (29.3, 0.412),
]

with h5py.File(FILE, "r") as f:
    # ── 1. Meta attributes ──────────────────────────────────────────────
    print("=" * 70)
    print("1. META ATTRIBUTES")
    print("=" * 70)
    meta = f["meta"]
    for k in sorted(meta.attrs.keys()):
        print(f"  {k}: {meta.attrs[k]}")

    control_hz = meta.attrs["control_hz"]
    num_frames = meta.attrs["num_frames"]
    dt = 1.0 / control_hz
    print(f"\n  → control_hz = {control_hz}, dt = {dt:.4f}s, num_frames = {num_frames}")

    # ── 2. Joint velocities (deg/s) ─────────────────────────────────────
    arm_qpos = f["arm_qpos"][:]  # (T, 7)
    T, n_joints = arm_qpos.shape
    print(f"\n  arm_qpos shape: {arm_qpos.shape}")

    # frame-to-frame delta in radians → degrees
    delta_rad = np.diff(arm_qpos, axis=0)  # (T-1, 7)
    vel_rad_per_s = delta_rad / dt          # (T-1, 7)
    vel_deg_per_s = np.rad2deg(vel_rad_per_s)

    # ── 3. Per-joint max velocity and limit violations ──────────────────
    print("\n" + "=" * 70)
    print("2. PER-JOINT MAX ABS VELOCITY (deg/s)")
    print("=" * 70)
    max_per_joint = np.max(np.abs(vel_deg_per_s), axis=0)
    for j in range(n_joints):
        print(f"  J{j}: max = {max_per_joint[j]:.1f} deg/s")

    for limit_name, limit_deg_s in [("90 deg/s (xArm mode 6 default)", 90), ("120 deg/s (teleop collection)", 120)]:
        print(f"\n  Frames exceeding {limit_name}:")
        for j in range(n_joints):
            viol = np.where(np.abs(vel_deg_per_s[:, j]) > limit_deg_s)[0]
            if len(viol):
                print(f"    J{j}: {len(viol)} frames (e.g. indices {viol[:10].tolist()}...)")
            else:
                print(f"    J{j}: none")

    # ── 4. Velocities around tracking-error frames (±5) ─────────────────
    print("\n" + "=" * 70)
    print("3. JOINT VELOCITIES AROUND TRACKING ERROR FRAMES (±5)")
    print("=" * 70)
    for t_sec, err_rad in ERROR_EVENTS:
        center_frame = int(round(t_sec * control_hz))
        lo = max(0, center_frame - 5)
        hi = min(T - 1, center_frame + 5)
        print(f"\n  T+{t_sec}s (error {err_rad:.3f} rad), center frame ~{center_frame}, window [{lo}, {hi-1}]:")
        print(f"  {'frame':>6s}", end="")
        for j in range(n_joints):
            print(f"  {'J'+str(j):>8s}", end="")
        print()
        for fi in range(lo, hi):
            print(f"  {fi:>6d}", end="")
            for j in range(n_joints):
                print(f"  {vel_deg_per_s[fi, j]:>8.1f}", end="")
            print()

    # ── 5. Top 10 frames by max joint velocity ──────────────────────────
    print("\n" + "=" * 70)
    print("4. TOP 10 FRAMES BY MAX JOINT VELOCITY (any joint)")
    print("=" * 70)
    max_vel_per_frame = np.max(np.abs(vel_deg_per_s), axis=1)  # (T-1,)
    top_indices = np.argsort(max_vel_per_frame)[::-1][:10]

    print(f"  {'rank':>5s}  {'frame':>6s}  {'time_s':>8s}  {'max_vel_deg_s':>14s}  {'joint':>6s}")
    for rank, fi in enumerate(top_indices, 1):
        joint_idx = np.argmax(np.abs(vel_deg_per_s[fi]))
        print(f"  {rank:>5d}  {fi:>6d}  {fi*dt:>8.3f}  {max_vel_per_frame[fi]:>14.1f}  {'J'+str(joint_idx):>6s}")

    print("\nDone.")
