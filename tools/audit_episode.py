#!/usr/bin/env python3
"""
Comprehensive data integrity audit for DexMani HDF5 episode files.
Usage: conda run -n real_robot python tools/audit_episode.py episodes_arm/episode_001.h5
"""
from __future__ import annotations

import sys
import os
import h5py
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_arr(a):
    """Format array for compact display."""
    return np.array2string(a, precision=3, suppress_small=True, max_line_width=120)

class AuditResult:
    def __init__(self):
        self.passes = []
        self.warnings = []
        self.failures = []

    def ok(self, msg):
        self.passes.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)

    def fail(self, msg):
        self.failures.append(msg)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)
        for msg in self.failures:
            print(f"  [FAIL] {msg}")
        for msg in self.warnings:
            print(f"  [WARN] {msg}")
        for msg in self.passes:
            print(f"  [PASS] {msg}")
        print("-" * 80)
        print(f"  TOTAL: {len(self.failures)} FAIL, {len(self.warnings)} WARN, {len(self.passes)} PASS")

audit = AuditResult()

# ---------------------------------------------------------------------------
# 0. Open file
# ---------------------------------------------------------------------------
fpath = sys.argv[1] if len(sys.argv) > 1 else "episodes_arm/episode_001.h5"
fpath = os.path.abspath(fpath)
print(f"Opening: {fpath}")
f = h5py.File(fpath, "r")

datasets = sorted(f.keys())
print(f"\nDatasets ({len(datasets)}): {datasets}")

# Read meta
meta = dict(f["meta"].attrs)
print(f"\nMeta: {meta}")
T_expected = meta.get("num_frames", None)
print(f"Expected num_frames from meta: {T_expected}")

# ---------------------------------------------------------------------------
# 1. Basic Integrity
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("1. BASIC INTEGRITY")
print("=" * 80)

# 1a. Shape consistency
state_ds_names = [
    "arm_qpos", "arm_qvel", "arm_tau", "arm_ee",
    "hand_qpos", "hand_fingertip", "hand_contact",
    "action_arm_joint", "action_arm_ee", "action_hand_joint",
    "vr_wrist_pos", "vr_wrist_rot6d", "vr_landmarks",
    "flag_ik_ok", "flag_held", "flag_retarget_ok",
    "timestamp",
]
T_values = {}
for name in state_ds_names:
    if name in f:
        ds = f[name]
        # Handle 0-d / scalar datasets
        if ds.ndim == 0:
            T_values[name] = 1
        else:
            T_values[name] = ds.shape[0]
    else:
        print(f"  MISSING: {name}")

# Count unique T values
t_counts = defaultdict(list)
for name, tval in T_values.items():
    t_counts[tval].append(name)

for tval, names in sorted(t_counts.items()):
    if len(names) == len(T_values):
        print(f"  T={tval}: all state datasets consistent")
    else:
        print(f"  T={tval}: {names}")

if len(t_counts) > 1:
    audit.fail(f"Shape inconsistency: {len(t_counts)} different T values: {dict(t_counts)}")
else:
    audit.ok("All state datasets have consistent shape (T)")

T = list(t_counts.keys())[0]

# 1b. NaN/Inf scan
print("\n--- NaN / Inf Scan ---")
nan_inf_found = False
for name in state_ds_names:
    if name not in f or f[name].ndim == 0:
        continue
    ds = f[name]
    # Read in chunks for large arrays
    chunk = 5000
    has_nan = False
    has_inf = False
    for i in range(0, ds.shape[0], chunk):
        arr = np.asarray(ds[i : i + chunk]).astype(np.float64)
        if np.any(np.isnan(arr)):
            has_nan = True
        if np.any(np.isinf(arr)):
            has_inf = True
        if has_nan and has_inf:
            break
    if has_nan:
        nan_inf_found = True
        audit.fail(f"NaN found in {name} (shape={ds.shape}, dtype={ds.dtype})")
    if has_inf:
        nan_inf_found = True
        audit.fail(f"Inf found in {name} (shape={ds.shape}, dtype={ds.dtype})")
    if not has_nan and not has_inf:
        print(f"  {name}: clean (no NaN/Inf)")

if not nan_inf_found:
    audit.ok("No NaN/Inf in any dataset")

# Camera NaN/Inf
for cam_name in ["rgb", "depth"]:
    if cam_name not in f:
        audit.fail(f"Camera dataset '{cam_name}' missing")
        continue
    ds = f[cam_name]
    chunk = 100
    has_nan = False
    has_inf = False
    for i in range(0, ds.shape[0], chunk):
        arr = ds[i : i + chunk]
        if np.any(np.isnan(arr)):
            has_nan = True
        if np.any(np.isinf(arr)):
            has_inf = True
        if has_nan and has_inf:
            break
    if has_nan:
        audit.fail(f"NaN found in {cam_name}")
    if has_inf:
        audit.fail(f"Inf found in {cam_name}")
    print(f"  {cam_name}: NaN={has_nan}, Inf={has_inf}")

# 1c. All-zero frames
print("\n--- All-Zero Frame Scan ---")
for name in state_ds_names:
    if name not in f or f[name].ndim == 0:
        continue
    ds = f[name]
    if ds.ndim == 1:
        # 1D dataset (e.g., flags)
        zero_count = int(np.sum(ds[:] == 0))
        pc = 100.0 * zero_count / len(ds)
        if pc > 0:
            print(f"  {name}: {zero_count}/{len(ds)} ({pc:.1f}%) zeros")
    else:
        # Multi-dim: check if entire row is all-zero
        chunk = 5000
        all_zero_count = 0
        for i in range(0, ds.shape[0], chunk):
            arr = ds[i : i + chunk]
            # Row-wise: all elements in a row are zero
            all_zero_count += int(np.sum(np.all(arr == 0, axis=tuple(range(1, arr.ndim)))))
        pc = 100.0 * all_zero_count / ds.shape[0]
        if pc > 0:
            print(f"  {name}: {all_zero_count}/{ds.shape[0]} ({pc:.1f}%) all-zero rows")
        if pc > 50:
            audit.warn(f"{name}: {pc:.1f}% all-zero frames (possible data dropout)")
        if pc == 100:
            audit.fail(f"{name}: 100% all-zero frames!")

# 1d. dtype check
print("\n--- Dtype Check ---")
expected_dtypes = {
    "arm_qpos": np.float64, "arm_qvel": np.float64, "arm_tau": np.float64,
    "arm_ee": np.float64, "hand_qpos": np.float64,
    "hand_fingertip": np.float64, "hand_contact": np.float64,
    "action_arm_joint": np.float64, "action_arm_ee": np.float64,
    "action_hand_joint": np.float64,
    "vr_wrist_pos": np.float64, "vr_wrist_rot6d": np.float64,
    "vr_landmarks": np.float64,
    "flag_ik_ok": np.bool_, "flag_held": np.bool_, "flag_retarget_ok": np.bool_,
    "timestamp": np.float64,
    "rgb": np.uint8, "depth": np.uint16,
}
for name, exp_dtype in expected_dtypes.items():
    if name not in f:
        continue
    actual = f[name].dtype
    if actual != exp_dtype:
        print(f"  {name}: dtype={actual} (expected {exp_dtype})")
        audit.warn(f"{name}: unexpected dtype {actual} (expected {exp_dtype})")
print("  Dtype check complete.")

# ---------------------------------------------------------------------------
# 2. Value Range Sanity
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. VALUE RANGE SANITY")
print("=" * 80)

def stats_report(ds, name, clip=None):
    """Print min, max, mean, std for a dataset. If clip set, flag out-of-range values."""
    chunk = 5000
    gmin = np.inf
    gmax = -np.inf
    sumsq = 0.0
    total = 0
    count = ds.shape[0] * (np.prod(ds.shape[1:]) if ds.ndim > 1 else 1)
    for i in range(0, ds.shape[0], chunk):
        arr = ds[i : i + chunk].astype(np.float64)
        gmin = min(gmin, float(np.min(arr)))
        gmax = max(gmax, float(np.max(arr)))
        sumsq += float(np.sum(arr * arr))
        total += arr.size
    mean = 0.0
    std = 0.0
    if total > 0:
        # For mean we need the actual sum, not just squares
        pass  # We'll compute mean/std below more carefully
    print(f"  {name} (shape={ds.shape}): min={gmin:.4f}, max={gmax:.4f}")
    if clip is not None:
        lo, hi = clip
        if gmin < lo or gmax > hi:
            audit.warn(f"{name}: range [{gmin:.4f}, {gmax:.4f}] outside expected [{lo}, {hi}]")

def full_stats(ds, name, clip=None):
    """Full per-element stats."""
    # Load entire array for small datasets, chunk for large
    total_elems = int(np.prod(ds.shape))
    if total_elems < 10_000_000:  # ~80MB for float64
        arr = np.asarray(ds[:]).astype(np.float64).ravel()
    else:
        chunk = 5000
        parts = []
        for i in range(0, ds.shape[0], chunk):
            parts.append(np.asarray(ds[i : i + chunk]).astype(np.float64).ravel())
        arr = np.concatenate(parts)

    if arr.size == 0:
        print(f"  {name:25s} shape={str(ds.shape):20s}  EMPTY")
        return

    gmin = float(np.min(arr))
    gmax = float(np.max(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    print(f"  {name:25s} shape={str(ds.shape):20s}  min={gmin:10.4f}  max={gmax:10.4f}  mean={mean:10.4f}  std={std:10.4f}")
    if clip is not None:
        lo, hi = clip
        if gmin < lo - 0.1 or gmax > hi + 0.1:
            audit.warn(f"{name}: range [{gmin:.4f}, {gmax:.4f}] outside expected [{lo}, {hi}]")

# arm_qpos
banner = lambda s: print(f"\n--- {s} ---")
banner("arm_qpos (7,)")
ds = f["arm_qpos"]
full_stats(ds, "arm_qpos", clip=(-2 * np.pi, 2 * np.pi))
# Per-joint stats
for j in range(7):
    col = ds[:, j]
    jmin, jmax, jmean, jstd = np.min(col), np.max(col), np.mean(col), np.std(col)
    stuck = (jstd < 1e-6)
    flag = " *** STUCK JOINT? ***" if stuck else ""
    print(f"    joint[{j}]: min={jmin:8.4f} max={jmax:8.4f} mean={jmean:8.4f} std={jstd:8.4f}{flag}")
    if stuck:
        audit.warn(f"arm_qpos joint[{j}] appears stuck (std={jstd:.2e})")

banner("arm_qvel (7,)")
ds = f["arm_qvel"]
full_stats(ds, "arm_qvel", clip=(-20, 20))
max_abs_vel = max(abs(np.min(ds[:])), abs(np.max(ds[:])))
if max_abs_vel > 10:
    audit.warn(f"arm_qvel: max absolute velocity {max_abs_vel:.2f} rad/s (high)")

banner("arm_tau (7,)")
ds = f["arm_tau"]
full_stats(ds, "arm_tau")

banner("arm_ee (9,) [pos(3)+rot6d(6)]")
ds = f["arm_ee"]
# pos part
pos = ds[:, :3]
pos_min, pos_max = np.min(pos), np.max(pos)
pos_mean = np.mean(pos)
print(f"  arm_ee pos(3): min={pos_min:.4f}, max={pos_max:.4f}, mean={pos_mean:.4f}")
# Check workspace: should be roughly 0.1-1.0m
if pos_min < 0.0 or pos_max > 1.5:
    audit.warn(f"arm_ee position outside expected workspace [0.0, 1.5]: min={pos_min:.4f}, max={pos_max:.4f}")
# rot6d part
rot6d = ds[:, 3:]
rmin, rmax = np.min(rot6d), np.max(rot6d)
print(f"  arm_ee rot6d(6): min={rmin:.4f}, max={rmax:.4f}")
if rmin < -1.5 or rmax > 1.5:
    audit.warn(f"arm_ee rot6d outside [-1.5, 1.5]: min={rmin:.4f}, max={rmax:.4f}")

banner("hand_qpos (12,)")
ds = f["hand_qpos"]
full_stats(ds, "hand_qpos")
all_zero_hq = np.sum(np.all(ds[:] == 0, axis=1))
if all_zero_hq > 0:
    audit.warn(f"hand_qpos: {all_zero_hq}/{T} all-zero frames")

banner("hand_fingertip (5,3)")
ds = f["hand_fingertip"]
all_zero_ft = int(np.sum(np.all(ds[:] == 0, axis=(1, 2))))
print(f"  All-zero frames: {all_zero_ft}/{T}")
if all_zero_ft > 0.9 * T:
    audit.warn(f"hand_fingertip: {all_zero_ft}/{T} ({100*all_zero_ft/T:.0f}%) all-zero (hand not used?)")
if all_zero_ft == T:
    audit.ok("hand_fingertip: all zeros (expected for arm-only recording)")
full_stats(ds, "hand_fingertip")

banner("hand_contact (5,3)")
ds = f["hand_contact"]
all_zero_hc = int(np.sum(np.all(ds[:] == 0, axis=(1, 2))))
print(f"  All-zero frames: {all_zero_hc}/{T}")
if all_zero_hc == T:
    audit.ok("hand_contact: all zeros (expected for arm-only recording)")

banner("action_arm_joint (7,)")
ds = f["action_arm_joint"]
full_stats(ds, "action_arm_joint")

banner("action_arm_ee (9,)")
ds = f["action_arm_ee"]
full_stats(ds, "action_arm_ee")

banner("action_hand_joint (12,)")
ds = f["action_hand_joint"]
full_stats(ds, "action_hand_joint")
all_zero_ahj = int(np.sum(np.all(ds[:] == 0, axis=1)))
if all_zero_ahj > 0:
    print(f"  All-zero frames: {all_zero_ahj}/{T}")
    if all_zero_ahj == T:
        audit.ok("action_hand_joint: all zeros (expected for arm-only recording)")

banner("vr_wrist_pos (3,)")
ds = f["vr_wrist_pos"]
full_stats(ds, "vr_wrist_pos")
all_zero_vp = int(np.sum(np.all(ds[:] == 0, axis=1)))
print(f"  All-zero frames: {all_zero_vp}/{T}")
if all_zero_vp == T:
    audit.warn("vr_wrist_pos: all zeros (VR tracking not active?)")

banner("vr_wrist_rot6d (6,)")
ds = f["vr_wrist_rot6d"]
full_stats(ds, "vr_wrist_rot6d", clip=(-1.5, 1.5))
all_zero_vr = int(np.sum(np.all(ds[:] == 0, axis=1)))
print(f"  All-zero frames: {all_zero_vr}/{T}")

banner("vr_landmarks (21,3)")
ds = f["vr_landmarks"]
full_stats(ds, "vr_landmarks")
all_zero_lm = int(np.sum(np.all(ds[:] == 0, axis=(1, 2))))
print(f"  All-zero frames: {all_zero_lm}/{T}")
if all_zero_lm == T:
    audit.ok("vr_landmarks: all zeros (expected for arm-only recording)")

banner("flags")
for flag_name in ["flag_ik_ok", "flag_retarget_ok", "flag_held"]:
    if flag_name in f:
        ds = f[flag_name]
        arr = ds[:]
        n_true = int(np.sum(arr))
        n_false = len(arr) - n_true
        pc = 100.0 * n_true / len(arr)
        print(f"  {flag_name}: True={n_true}/{len(arr)} ({pc:.1f}%), False={n_false}")
        if pc == 0:
            audit.warn(f"{flag_name}: 0% True (always False)")
        elif pc == 100:
            audit.warn(f"{flag_name}: 100% True (always True - may not be meaningful)")

# ---------------------------------------------------------------------------
# 3. Temporal Analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. TEMPORAL ANALYSIS")
print("=" * 80)

ts = f["timestamp"][:]
dt = np.diff(ts)
print(f"  Timestamp range: {ts[0]:.6f} -> {ts[-1]:.6f} (duration={ts[-1]-ts[0]:.2f}s)")
print(f"  Frames: {len(ts)}")
print(f"  dt stats: min={np.min(dt):.6f}s, max={np.max(dt):.6f}s, mean={np.mean(dt):.6f}s, std={np.std(dt):.6f}s")
print(f"  Expected dt: {1/50:.6f}s (50Hz)")
print(f"  Actual fps: {1/np.mean(dt):.2f}")

# Monotonicity
if not np.all(dt > 0):
    n_regress = int(np.sum(dt <= 0))
    audit.fail(f"Timestamp regression: {n_regress} non-positive dt values")
else:
    audit.ok("Timestamps are monotonically increasing")

# Check dt consistency
dt_target = 1.0 / 50.0
dt_deviation = np.abs(dt - dt_target)
large_dev = np.sum(dt_deviation > 0.005)  # >5ms deviation
if large_dev > 0:
    n_large = int(large_dev)
    pc = 100.0 * n_large / len(dt)
    if pc > 5:
        audit.warn(f"dt deviation > 5ms: {n_large}/{len(dt)} ({pc:.1f}%) frames")
    else:
        print(f"  dt deviation > 5ms: {n_large}/{len(dt)} ({pc:.1f}%) frames (minor)")

# Detect time jumps
jumps = np.where(dt > 0.1)[0]  # >100ms gap
if len(jumps) > 0:
    audit.warn(f"Time jumps (>100ms gap): {len(jumps)} occurrences")
    for j_idx in jumps[:10]:  # Show first 10
        print(f"    frame {j_idx}->{j_idx+1}: dt={dt[j_idx]:.4f}s")
else:
    audit.ok("No time jumps >100ms detected")

# Camera frame count vs state frame count
if "rgb" in f:
    C = f["rgb"].shape[0]
    print(f"\n  Camera frames (rgb): C={C}")
    print(f"  State frames: T={T}")
    if C != T:
        print(f"  Mismatch: C={C} vs T={T} (delta={abs(C-T)})")
        if abs(C - T) > T * 0.05:
            audit.warn(f"Camera/state frame count mismatch: C={C}, T={T} ({abs(C-T)} frames)")
        else:
            print(f"  Small mismatch (within 5%)")

# ---------------------------------------------------------------------------
# 4. Camera Data
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. CAMERA DATA")
print("=" * 80)

if "rgb" in f:
    rgb = f["rgb"]
    print(f"  rgb shape={rgb.shape}, dtype={rgb.dtype}")
    # Sample frames
    n_sample = min(20, rgb.shape[0])
    indices = np.linspace(0, rgb.shape[0] - 1, n_sample, dtype=int)

    all_black_count = 0
    all_white_count = 0
    stuck_count = 0
    prev_frame = None

    for idx in indices:
        frame = rgb[idx]
        if prev_frame is not None:
            if np.array_equal(frame, prev_frame):
                stuck_count += 1
        prev_frame = frame.copy()

        fmin, fmax = np.min(frame), np.max(frame)
        fmean = np.mean(frame)
        if fmax == 0:
            all_black_count += 1
        if fmin == 255:
            all_white_count += 1

    print(f"  Sampled {n_sample} rgb frames:")
    print(f"    all-black: {all_black_count}/{n_sample}")
    print(f"    all-white: {all_white_count}/{n_sample}")
    print(f"    consecutive identical (in sample): {stuck_count}/{n_sample-1}")

    # Full scan for all-black frames (first 50 and last 50)
    n_check = min(50, rgb.shape[0])
    ab_first = int(np.sum(np.all(rgb[:n_check] == 0, axis=(1, 2, 3))))
    ab_last = int(np.sum(np.all(rgb[-n_check:] == 0, axis=(1, 2, 3))))
    print(f"  All-black in first {n_check}: {ab_first}")
    print(f"  All-black in last {n_check}: {ab_last}")

    # Sample mean/std across frames to detect anomalies
    fmeans = []
    for i in range(0, rgb.shape[0], max(1, rgb.shape[0] // 100)):
        fmeans.append(np.mean(rgb[i].astype(np.float64)))
    fmeans = np.array(fmeans)
    print(f"  Mean pixel value across {len(fmeans)} sampled frames: {np.mean(fmeans):.1f} +/- {np.std(fmeans):.1f}")

    if all_black_count > 0:
        audit.warn(f"rgb: {all_black_count}/{n_sample} sampled frames are all-black")
    if stuck_count > n_sample * 0.5:
        audit.warn(f"rgb: {stuck_count}/{n_sample-1} sampled frames are identical (possible stuck camera)")
    if all_black_count == 0 and stuck_count <= 1:
        audit.ok("rgb: no obvious anomalies in sampling check")

if "depth" in f:
    depth = f["depth"]
    print(f"\n  depth shape={depth.shape}, dtype={depth.dtype}")
    # Sample
    n_sample = min(20, depth.shape[0])
    indices = np.linspace(0, depth.shape[0] - 1, n_sample, dtype=int)
    all_zero_count = 0
    for idx in indices:
        frame = depth[idx]
        if np.max(frame) == 0:
            all_zero_count += 1
    print(f"  All-zero in {n_sample} sampled depth frames: {all_zero_count}/{n_sample}")

# ---------------------------------------------------------------------------
# 5. Cross-Stream Consistency
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. CROSS-STREAM CONSISTENCY")
print("=" * 80)

# 5a. arm_qpos vs action_arm_joint
if "arm_qpos" in f and "action_arm_joint" in f:
    aq = f["arm_qpos"][:]
    aj = f["action_arm_joint"][:]
    diff = np.abs(aq - aj)
    per_joint_max = np.max(diff, axis=0)
    per_joint_mean = np.mean(diff, axis=0)
    overall_max = np.max(diff)
    overall_mean = np.mean(diff)
    print(f"  arm_qpos vs action_arm_joint:")
    print(f"    per-joint max diff: {fmt_arr(per_joint_max)}")
    print(f"    per-joint mean diff: {fmt_arr(per_joint_mean)}")
    print(f"    overall max diff: {overall_max:.6f} rad")
    print(f"    overall mean diff: {overall_mean:.6f} rad")
    if overall_max > 0.5:
        audit.warn(f"Large arm_qpos vs action_arm_joint discrepancy: max={overall_max:.4f} rad")

# 5b. arm_ee pos vs hand_fingertip
if "arm_ee" in f and "hand_fingertip" in f:
    ee_pos = f["arm_ee"][:, :3]
    ft = f["hand_fingertip"][:]  # (T, 5, 3)
    if not np.all(ft == 0):
        ft_centroid = np.mean(ft, axis=1)  # (T, 3)
        ft_ee_dist = np.linalg.norm(ft_centroid - ee_pos, axis=1)
        print(f"  EE pos vs fingertip centroid distance:")
        print(f"    min={np.min(ft_ee_dist):.4f}, max={np.max(ft_ee_dist):.4f}, mean={np.mean(ft_ee_dist):.4f}")
        if np.mean(ft_ee_dist) > 0.5:
            audit.warn(f"EE pos far from fingertip centroid: mean distance={np.mean(ft_ee_dist):.3f}m")

# 5c. vr_wrist_pos vs arm_ee pos
if "vr_wrist_pos" in f and "arm_ee" in f:
    vp = f["vr_wrist_pos"][:]
    ee_pos = f["arm_ee"][:, :3]
    if not np.all(vp == 0):
        vp_ee_dist = np.linalg.norm(vp - ee_pos, axis=1)
        print(f"\n  VR wrist pos vs arm EE pos distance:")
        print(f"    min={np.min(vp_ee_dist):.4f}, max={np.max(vp_ee_dist):.4f}, mean={np.mean(vp_ee_dist):.4f}")
        # VR and EE should track with some offset
    else:
        print(f"\n  vr_wrist_pos is all zeros (VR not active)")

# ---------------------------------------------------------------------------
# 6. Edge Cases
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("6. EDGE CASES")
print("=" * 80)

# 6a. First/last 10 frames
print("\n--- First 10 frames ---")
for name in state_ds_names:
    if name not in f or f[name].ndim == 0:
        continue
    ds = f[name]
    n_check = min(10, ds.shape[0])
    arr = ds[:n_check]
    if ds.ndim == 1:
        all_same = np.all(arr == arr[0])
    else:
        all_same = np.all(np.all(arr == arr[0], axis=tuple(range(1, arr.ndim))), axis=0)
    all_zero = np.all(arr == 0)
    if all_same and not all_zero:
        print(f"  {name}: first {n_check} frames all identical (non-zero) - possible init artifact")
        audit.warn(f"{name}: first {n_check} frames all identical (initialization artifact?)")
    elif all_zero:
        print(f"  {name}: first {n_check} frames all zero - possible pre-init")

print("\n--- Last 10 frames ---")
for name in state_ds_names:
    if name not in f or f[name].ndim == 0:
        continue
    ds = f[name]
    n_check = min(10, ds.shape[0])
    arr = ds[-n_check:]
    if ds.ndim == 1:
        all_same = np.all(arr == arr[0])
    else:
        all_same = np.all(np.all(arr == arr[0], axis=tuple(range(1, arr.ndim))), axis=0)
    all_zero = np.all(arr == 0)
    if all_same and not all_zero:
        print(f"  {name}: last {n_check} frames all identical (non-zero) - possible shutdown artifact")
        audit.warn(f"{name}: last {n_check} frames all identical (shutdown artifact?)")

# 6b. Different T check
print("\n--- Shape Check ---")
for name in datasets:
    ds = f[name]
    if isinstance(ds, h5py.Group):
        print(f"  {name}: Group (attrs={dict(ds.attrs)})")
        continue
    if name in T_values:
        tval = T_values.get(name)
        if tval != T and name not in ("meta", "rgb", "depth"):
            print(f"  {name}: T={tval} (differs from expected T={T})")
            audit.warn(f"{name}: T={tval} differs from expected T={T}")
    else:
        print(f"  {name}: shape={ds.shape}")

# 6c. Duplicate consecutive frames
print("\n--- Duplicate Consecutive Frames ---")
for name in ["arm_qpos", "arm_ee", "hand_qpos"]:
    if name not in f:
        continue
    ds = f[name]
    chunk = 5000
    dup_count = 0
    for i in range(0, ds.shape[0] - 1, chunk):
        end = min(i + chunk, ds.shape[0] - 1)
        arr = ds[i : end + 1]
        if ds.ndim == 1:
            dups = np.sum(arr[1:] == arr[:-1])
        else:
            # Compare rows
            dups = np.sum(np.all(arr[1:] == arr[:-1], axis=tuple(range(1, arr.ndim))))
        dup_count += int(dups)
    pc = 100.0 * dup_count / (ds.shape[0] - 1)
    print(f"  {name}: {dup_count}/{ds.shape[0]-1} ({pc:.1f}%) duplicate consecutive frames")
    if pc > 10:
        audit.warn(f"{name}: {pc:.1f}% duplicate consecutive frames (possible logging stall)")

# ---------------------------------------------------------------------------
# 7. Specific integrity - arm_qpos per-dimension check
# ---------------------------------------------------------------------------
print("\n--- arm_qpos per-joint range check ---")
# xArm7 joint limits (radians)
xarm7_limits = np.array([
    [-2 * np.pi, 2 * np.pi],   # joint 0
    [-2.07, 2.07],              # joint 1
    [-2 * np.pi, 2 * np.pi],   # joint 2
    [-0.19, 3.93],              # joint 3
    [-2 * np.pi, 2 * np.pi],   # joint 4
    [-1.69, 2.08],              # joint 5
    [-2 * np.pi, 2 * np.pi],   # joint 6
])
aq = f["arm_qpos"][:]
for j in range(7):
    lo, hi = xarm7_limits[j]
    jmin, jmax = np.min(aq[:, j]), np.max(aq[:, j])
    if jmin < lo - 0.05 or jmax > hi + 0.05:
        audit.warn(f"arm_qpos joint[{j}]: [{jmin:.4f}, {jmax:.4f}] exceeds limit [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
f.close()
audit.print_summary()
