"""离线统计 position IK vs diff IK 成功率占比。

无需硬件，生成合成 EEF 轨迹，回放 IK 管线，统计每种 IK 方法的使用频率。

Usage:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate real
    cd /home/zhy/Desktop/dexmani_real
    python scripts/analyze_ik_stats.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dexmani_real import ASSET_DIR
from dexmani_real.planner.arm_planner import XArm7MotionPlanner
from dexmani_real.planner.planner_types import (
    PlanningProfile,
    Pose,
    TeleopProfile,
    XArm7PlannerConfig,
)

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def _generate_circular_trajectory(
    center: np.ndarray, radius: float, n_frames: int, z_offset: float = 0.0,
) -> list[Pose]:
    """Generate EEF poses tracing a horizontal circle in world frame."""
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        p = center + np.array([radius * np.cos(angle), radius * np.sin(angle), z_offset])
        # Orientation: keep EEF pointing downward (typical teleop pose)
        # Rotation around y-axis by π (pointing down), stored as wxyz
        q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)  # 180° around y
        poses.append(Pose(p=p, q=q))
    return poses


def _generate_figure8_trajectory(
    center: np.ndarray, scale: float, n_frames: int, z_offset: float = 0.0,
) -> list[Pose]:
    """Generate EEF poses tracing a figure-8 in world frame."""
    poses = []
    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        p = center + np.array([x, y, z_offset])
        q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        poses.append(Pose(p=p, q=q))
    return poses


def _generate_linear_trajectory(
    start: np.ndarray, end: np.ndarray, n_frames: int,
) -> list[Pose]:
    """Generate EEF poses along a straight line in world frame."""
    poses = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        p = start + alpha * (end - start)
        q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        poses.append(Pose(p=p, q=q))
    return poses


def _generate_random_walk(
    start: np.ndarray, n_frames: int, step_std: float = 0.01, bounds: np.ndarray | None = None,
) -> list[Pose]:
    """Generate EEF poses via bounded random walk."""
    rng = np.random.default_rng(42)
    poses = []
    current = start.astype(np.float64).copy()
    for _ in range(n_frames):
        step = rng.normal(0, step_std, size=3)
        current = current + step
        if bounds is not None:
            current = np.clip(current, bounds[:, 0], bounds[:, 1])
        q = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        poses.append(Pose(p=current.copy(), q=q.copy()))
    return poses


def _generate_varying_orientation(
    base_pos: np.ndarray, n_frames: int, tilt_amplitude_deg: float = 30.0,
) -> list[Pose]:
    """Generate EEF at fixed position with varying orientation (pitch/roll oscillation)."""
    from scipy.spatial.transform import Rotation

    poses = []
    for i in range(n_frames):
        t = 2 * np.pi * i / n_frames
        pitch = np.deg2rad(tilt_amplitude_deg) * np.sin(t)
        roll = np.deg2rad(tilt_amplitude_deg * 0.5) * np.cos(t * 0.7)
        R = Rotation.from_euler("xz", [pitch, roll]).as_quat()  # returns xyzw
        q_wxyz = np.array([R[3], R[0], R[1], R[2]], dtype=np.float64)
        poses.append(Pose(p=base_pos.copy(), q=q_wxyz))
    return poses


def analyze_ik_pipeline(
    planner: XArm7MotionPlanner,
    target_poses: list[Pose],
    label: str,
    initial_qpos: np.ndarray,
) -> dict:
    """Run IK on a trajectory and collect statistics.

    Returns dict with:
      - position_ik_count, diff_ik_count, total_held_count
      - rejection_reasons: Counter of position IK failure reasons
      - tracking_errors: list of (pos_err_m, rot_err_rad) per frame
      - method_per_frame: list of method strings
    """
    ik_solver = planner.teleop_solver
    kin = planner.kin
    prev_cmd = initial_qpos.copy()
    current_qpos = initial_qpos.copy()

    stats = {
        "label": label,
        "n_frames": len(target_poses),
        "position_ik_success": 0,
        "diff_ik_success": 0,
        "total_failures": 0,
        "total_held": 0,
        "rejection_reasons": Counter(),
        "tracking_pos_errors": [],
        "tracking_rot_errors": [],
        "method_per_frame": [],
    }

    for i, target_pose_world in enumerate(target_poses):
        result = ik_solver.solve(target_pose_world, current_qpos, prev_cmd)

        method = result.report.get("teleop_ik_method", "unknown")
        stats["method_per_frame"].append(method)

        if result.success and result.qpos is not None:
            if method == "position_ik":
                stats["position_ik_success"] += 1
            elif method == "differential_ik":
                stats["diff_ik_success"] += 1

            prev_cmd = result.qpos.copy()
            # Simulate hardware tracking: move toward target at limited speed
            err = result.qpos - current_qpos
            max_step = np.deg2rad(ik_solver.profile.max_qpos_cmd_speed_deg) * ik_solver.profile.teleop_dt
            step = np.clip(err, -max_step, max_step)
            current_qpos = current_qpos + step
        else:
            stats["total_failures"] += 1
            if result.held:
                stats["total_held"] += 1
            prev_cmd = prev_cmd.copy()  # hold in place
            # current_qpos unchanged (hardware holds)

        # Collect position IK rejection reasons from report
        if method != "position_ik":
            pos_report = result.report.get("position_ik_report", {})
            rejection = pos_report.get("failure_reason", "")
            if rejection:
                stats["rejection_reasons"][rejection] += 1
            else:
                # Check the top-level report for rejection info
                reject_counts = result.report.get("teleop_ik_reject_counts", {})
                if reject_counts:
                    for reason, count in reject_counts.items():
                        stats["rejection_reasons"][reason] += count

        # Track nominal tracking error
        current_pose = kin.compute_eef_pose_world(current_qpos)
        from dexmani_real.planner.pose_utils import compute_pose_error
        pos_err, rot_err = compute_pose_error(target_pose_world, current_pose)
        stats["tracking_pos_errors"].append(pos_err)
        stats["tracking_rot_errors"].append(rot_err)

    # Summarize
    stats["position_ik_pct"] = (
        100 * stats["position_ik_success"] / stats["n_frames"]
        if stats["n_frames"] > 0 else 0
    )
    stats["diff_ik_pct"] = (
        100 * stats["diff_ik_success"] / stats["n_frames"]
        if stats["n_frames"] > 0 else 0
    )
    stats["failure_pct"] = (
        100 * stats["total_failures"] / stats["n_frames"]
        if stats["n_frames"] > 0 else 0
    )
    stats["mean_pos_error_m"] = float(np.mean(stats["tracking_pos_errors"]))
    stats["mean_rot_error_rad"] = float(np.mean(stats["tracking_rot_errors"]))
    stats["max_pos_error_m"] = float(np.max(stats["tracking_pos_errors"]))
    stats["max_rot_error_rad"] = float(np.max(stats["tracking_rot_errors"]))

    return stats


def print_report(all_stats: list[dict]) -> None:
    """Print consolidated report."""
    print("\n" + "=" * 72)
    print("  IK Method Distribution Report")
    print("=" * 72)

    total_pos = sum(s["position_ik_success"] for s in all_stats)
    total_diff = sum(s["diff_ik_success"] for s in all_stats)
    total_fail = sum(s["total_failures"] for s in all_stats)
    total_frames = sum(s["n_frames"] for s in all_stats)
    total_held = sum(s["total_held"] for s in all_stats)
    all_rejections: Counter = Counter()
    for s in all_stats:
        all_rejections.update(s["rejection_reasons"])

    print(f"\n  Total frames:           {total_frames}")
    print(f"  Position IK successes:  {total_pos:5d}  ({100*total_pos/total_frames:5.1f}%)")
    print(f"  Diff IK successes:      {total_diff:5d}  ({100*total_diff/total_frames:5.1f}%)")
    print(f"  Total failures:         {total_fail:5d}  ({100*total_fail/total_frames:5.1f}%)")
    print(f"  (of which held):        {total_held:5d}")

    print(f"\n  Per-trajectory breakdown:")
    print(f"  {'Trajectory':<24s} {'Frames':>6s} {'PosIK':>6s} {'%':>6s} {'DiffIK':>6s} {'%':>6s} {'Fail':>5s} {'PosErr_m':>9s} {'RotErr_rad':>10s}")
    print(f"  {'-'*24} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5} {'-'*9} {'-'*10}")
    for s in all_stats:
        print(
            f"  {s['label']:<24s} "
            f"{s['n_frames']:>6d} "
            f"{s['position_ik_success']:>6d} {s['position_ik_pct']:>5.1f}% "
            f"{s['diff_ik_success']:>6d} {s['diff_ik_pct']:>5.1f}% "
            f"{s['total_failures']:>5d} "
            f"{s['mean_pos_error_m']:>8.4f}m "
            f"{s['mean_rot_error_rad']:>9.4f}rad"
        )

    if all_rejections:
        print(f"\n  Position IK rejection reasons (accumulated counts):")
        for reason, count in all_rejections.most_common():
            print(f"    {reason:<40s} {count:>6d}")

    # Transitions analysis: how often does method switch?
    all_methods = []
    for s in all_stats:
        all_methods.extend(s["method_per_frame"])
    switches = sum(
        1 for i in range(1, len(all_methods))
        if all_methods[i] != all_methods[i - 1]
    )
    print(f"\n  Method switches:         {switches} / {len(all_methods)-1} transitions "
          f"({100*switches/max(len(all_methods)-1,1):.1f}%)")

    print(f"\n  Tracking Error Summary:")
    all_pos_errs = [e for s in all_stats for e in s["tracking_pos_errors"]]
    all_rot_errs = [e for s in all_stats for e in s["tracking_rot_errors"]]
    print(f"    Position error: mean={np.mean(all_pos_errs):.4f}m  max={np.max(all_pos_errs):.4f}m")
    print(f"    Rotation error: mean={np.rad2deg(np.mean(all_rot_errs)):.2f}°  max={np.rad2deg(np.max(all_rot_errs)):.2f}°")

    print("\n" + "=" * 72)


def main():
    print("Initializing planner (MPlib + Pinocchio)...")
    t0 = time.perf_counter()

    urdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf")
    srdf_path = str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision_mplib.srdf")

    planner_config = XArm7PlannerConfig(
        urdf_path=urdf_path,
        srdf_path=srdf_path,
        eef_link_name="custom_eef_link",
    )
    planner = XArm7MotionPlanner(
        config=planner_config,
        planning_profile=PlanningProfile(),
        teleop_profile=TeleopProfile(debug=False),
    )

    dt_init = time.perf_counter() - t0
    print(f"  Planner ready in {dt_init:.1f}s")
    print(f"  DOF: {planner.kin.dof}")
    print(f"  Joint limits: {np.round(planner.kin.joint_limits, 4)}")

    # Initial qpos: a reasonable mid-range configuration
    init_qpos = np.deg2rad([-30, -45, 0, 20, -180, 25, 0])

    # Verify home EEF is in reasonable workspace
    home_eef = planner.compute_eef_pose_world(init_qpos)
    print(f"  Home EEF: pos={np.round(home_eef.p, 3)} m")

    # Generate diverse test trajectories
    center = home_eef.p + np.array([0.0, 0.0, 0.05])  # slightly above home
    all_stats = []

    # 1. Large circle (wide workspace coverage, likely triggers more branch switches)
    print("\n[1/7] Circular trajectory (r=0.15m)...")
    circle = _generate_circular_trajectory(center, radius=0.15, n_frames=500)
    all_stats.append(analyze_ik_pipeline(planner, circle, "circle_r150mm", init_qpos))

    # 2. Small circle (fine motion, within single branch more often)
    print("[2/7] Circular trajectory (r=0.05m)...")
    circle_small = _generate_circular_trajectory(center, radius=0.05, n_frames=500)
    all_stats.append(analyze_ik_pipeline(planner, circle_small, "circle_r50mm", init_qpos))

    # 3. Figure-8 (direction reversals)
    print("[3/7] Figure-8 trajectory...")
    fig8 = _generate_figure8_trajectory(center, scale=0.10, n_frames=500)
    all_stats.append(analyze_ik_pipeline(planner, fig8, "figure8", init_qpos))

    # 4. Linear motion (simple, should be all position IK)
    print("[4/7] Linear trajectory...")
    linear_start = center + np.array([-0.10, -0.10, 0.0])
    linear_end = center + np.array([0.10, 0.10, 0.0])
    linear = _generate_linear_trajectory(linear_start, linear_end, n_frames=300)
    all_stats.append(analyze_ik_pipeline(planner, linear, "linear_diag", init_qpos))

    # 5. Vertical linear motion
    print("[5/7] Vertical trajectory...")
    v_start = center + np.array([0.0, 0.0, -0.10])
    v_end = center + np.array([0.0, 0.0, 0.15])
    vertical = _generate_linear_trajectory(v_start, v_end, n_frames=300)
    all_stats.append(analyze_ik_pipeline(planner, vertical, "linear_vertical", init_qpos))

    # 6. Random walk (realistic teleop noise)
    print("[6/7] Random walk trajectory...")
    bounds = np.array([[0.20, 0.70], [-0.35, 0.35], [0.0, 0.50]])
    rw = _generate_random_walk(center, n_frames=800, step_std=0.008, bounds=bounds)
    all_stats.append(analyze_ik_pipeline(planner, rw, "random_walk", init_qpos))

    # 7. Varying orientation at fixed position
    print("[7/7] Varying orientation...")
    orient = _generate_varying_orientation(center, n_frames=500, tilt_amplitude_deg=30.0)
    all_stats.append(analyze_ik_pipeline(planner, orient, "orientation_only", init_qpos))

    print_report(all_stats)

    # Quick verdict
    total_pos = sum(s["position_ik_success"] for s in all_stats)
    total_diff = sum(s["diff_ik_success"] for s in all_stats)
    diff_pct = 100 * total_diff / max(total_pos + total_diff, 1)
    print(f"  VERDICT: diff IK accounts for {diff_pct:.1f}% of successful frames.")
    if diff_pct > 80:
        print("  → diff IK is the dominant path. Velocity control mode is promising.")
    elif diff_pct > 50:
        print("  → Mixed usage. Velocity mode needs careful diff IK prioritization.")
    else:
        print("  → Position IK is the dominant path. Velocity mode likely problematic.")
        print("    Consider improving diff IK coverage first.")


if __name__ == "__main__":
    main()
