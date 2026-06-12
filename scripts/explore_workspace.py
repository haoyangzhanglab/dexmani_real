#!/usr/bin/env python3
"""xArm7 工作空间探索：可达性、奇异点、碰撞区域、manipulability 分布。

Usage:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate sim
    python scripts/explore_workspace.py                     # 快速探索 (30k 采样)
    python scripts/explore_workspace.py --dense             # 密集探索 (200k 采样)
    python scripts/explore_workspace.py --slice-z 0.3       # 固定 z=0.3m 切片
    python scripts/explore_workspace.py --collision-urdf    # 用碰撞 URDF
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WorkspaceSample:
    eef_pos: np.ndarray   # (3,) 世界坐标系
    eef_quat: np.ndarray  # (4,) wxyz
    qpos: np.ndarray      # (7,) 关节角
    ik_success: bool
    manip: float
    in_collision: bool
    joint_penalty: float


def make_planner(use_collision_urdf: bool = False, eef_link: str = "link_eef") -> Any:
    from dexmani_real.robot.planner import XArm7MotionPlanner, XArm7PlannerConfig

    if use_collision_urdf:
        urdf = "/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xhand/xarm7_xhand_collision.urdf"
        srdf = "/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xhand/xarm7_xhand_collision_mplib.srdf"
        eef_link = "custom_eef_link"
    else:
        urdf = "/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xarm7/xarm7_glb.urdf"
        srdf = "/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xarm7/xarm7_glb_mplib.srdf"

    config = XArm7PlannerConfig(
        urdf_path=urdf, srdf_path=srdf, eef_link_name=eef_link,
    )
    return XArm7MotionPlanner(config)


def random_ik_targets(planner, n: int, rng: np.random.Generator) -> np.ndarray:
    """用 MPlib 内置 IK 在 workspace 内均匀撒目标点。

    策略：从随机关节角出发做 FK，得到的 EEF 位姿必然在 workspace 内。
    """
    targets = np.zeros((n, 7), dtype=np.float64)  # pos(3) + quat(4)
    low = planner.joint_limits[:, 0]
    high = planner.joint_limits[:, 1]

    for i in range(n):
        q = rng.uniform(low, high)
        try:
            pose = planner.compute_eef_pose_world(q)
            targets[i, :3] = pose.p
            targets[i, 3:] = pose.q
        except Exception:
            targets[i] = np.nan
    return targets[~np.isnan(targets[:, 0])]


def sample_workspace(planner, n_samples: int, z_slice: float | None = None) -> list[WorkspaceSample]:
    """在 workspace 内撒 N 个点，对每个点做 IK，记录可达性/碰撞/manipulability."""
    from dexmani_real.robot.planner.planner_types import Pose

    rng = np.random.default_rng(42)

    print(f"  生成 {n_samples} 个随机关节构型做 FK → 目标位姿...")
    targets_flat = random_ik_targets(planner, n_samples, rng)
    n_valid = len(targets_flat)
    print(f"  有效目标: {n_valid} ({100*n_valid/n_samples:.0f}%)")

    if z_slice is not None:
        mask = np.abs(targets_flat[:, 2] - z_slice) < 0.02
        targets_flat = targets_flat[mask]
        print(f"  z={z_slice}m 切片: {len(targets_flat)} 点")

    results: list[WorkspaceSample] = []
    qpos_zero = np.zeros(7)
    report_interval = max(1, len(targets_flat) // 10)

    for idx, t in enumerate(targets_flat):
        if idx % report_interval == 0:
            pct = 100 * idx / len(targets_flat)
            print(f"  IK 求解... {pct:.0f}%", end="\r")

        target = Pose(p=t[:3], q=t[3:])
        ik_result = planner.solve_ik(target, qpos_zero)

        if ik_result.success and ik_result.qpos is not None:
            qpos = ik_result.qpos
            manip = planner.compute_manipulability(qpos)
            in_col = planner.has_self_collision(qpos)
            jp = planner.joint_limit_penalty(qpos, planner.joint_limits)

            results.append(WorkspaceSample(
                eef_pos=t[:3].copy(),
                eef_quat=t[3:].copy(),
                qpos=qpos,
                ik_success=True,
                manip=manip,
                in_collision=in_col,
                joint_penalty=jp,
            ))
        else:
            results.append(WorkspaceSample(
                eef_pos=t[:3].copy(),
                eef_quat=t[3:].copy(),
                qpos=np.zeros(7),
                ik_success=False,
                manip=0.0,
                in_collision=False,
                joint_penalty=0.0,
            ))

    print(f"  IK 求解... 100%")
    return results


def analyze_results(results: list[WorkspaceSample]) -> dict[str, Any]:
    """统计分析."""
    positions = np.array([s.eef_pos for s in results])
    ik_ok = np.array([s.ik_success for s in results])
    ik_ok_samples = [s for s in results if s.ik_success]
    collisions = np.array([s.in_collision for s in ik_ok_samples])
    manips = np.array([s.manip for s in ik_ok_samples])

    # 位置范围
    p_min = positions[ik_ok].min(axis=0)
    p_max = positions[ik_ok].max(axis=0)

    # 每个轴的分布
    stats = {
        "total_samples": len(results),
        "ik_success": int(ik_ok.sum()),
        "ik_success_rate": float(ik_ok.mean()),
        "self_collision_count": int(collisions.sum()),
        "self_collision_rate": float(collisions.mean()) if len(collisions) > 0 else 0.0,
        "position_range": {
            "x": (float(p_min[0]), float(p_max[0])),
            "y": (float(p_min[1]), float(p_max[1])),
            "z": (float(p_min[2]), float(p_max[2])),
        },
        "manipulability": {
            "min": float(manips.min()) if len(manips) > 0 else 0.0,
            "max": float(manips.max()) if len(manips) > 0 else 0.0,
            "mean": float(manips.mean()) if len(manips) > 0 else 0.0,
            "median": float(np.median(manips)) if len(manips) > 0 else 0.0,
            "p10": float(np.percentile(manips, 10)) if len(manips) > 0 else 0.0,
        },
        "radial_reach": {
            "mean": float(np.linalg.norm(positions[ik_ok, :2], axis=1).mean()),
            "max": float(np.linalg.norm(positions[ik_ok, :2], axis=1).max()),
        },
    }
    return stats


def print_report(stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("工作空间探索报告")
    print("=" * 60)
    print(f"  总采样数:       {stats['total_samples']}")
    print(f"  IK 成功率:      {stats['ik_success']}/{stats['total_samples']} ({100*stats['ik_success_rate']:.1f}%)")
    print(f"  自碰撞率:       {stats['self_collision_count']}/{stats['ik_success']} ({100*stats['self_collision_rate']:.1f}%)")
    print()
    print("  可达范围 (meters):")
    for axis, (lo, hi) in stats["position_range"].items():
        span = hi - lo
        center = (lo + hi) / 2
        print(f"    {axis}: [{lo:+.3f}, {hi:+.3f}]  span={span:.3f}m  center={center:+.3f}m")
    print()
    print("  XY 平面径向可达距离:")
    print(f"    mean: {stats['radial_reach']['mean']:.3f}m")
    print(f"    max:  {stats['radial_reach']['max']:.3f}m")
    print()
    print("  Manipulability 分布 (Yoshikawa):")
    m = stats["manipulability"]
    print(f"    min:    {m['min']:.6f}")
    print(f"    p10:    {m['p10']:.6f}")
    print(f"    median: {m['median']:.6f}")
    print(f"    mean:   {m['mean']:.6f}")
    print(f"    max:    {m['max']:.6f}")
    print()
    print("  推荐 workspace_bounds (含 10cm margin):")
    margin = 0.10
    safe = stats["position_range"]
    print(f"    x: [{safe['x'][0]+margin:.2f}, {safe['x'][1]-margin:.2f}]")
    print(f"    y: [{safe['y'][0]+margin:.2f}, {safe['y'][1]-margin:.2f}]")
    print(f"    z: [{safe['z'][0]+margin:.2f}, {safe['z'][1]-margin:.2f}]")


def save_csv(results: list[WorkspaceSample], path: str) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "qw", "qx", "qy", "qz", "ik_ok", "manip", "collision", "joint_penalty"])
        for s in results:
            w.writerow([
                f"{s.eef_pos[0]:.6f}", f"{s.eef_pos[1]:.6f}", f"{s.eef_pos[2]:.6f}",
                f"{s.eef_quat[0]:.6f}", f"{s.eef_quat[1]:.6f}", f"{s.eef_quat[2]:.6f}", f"{s.eef_quat[3]:.6f}",
                int(s.ik_success), f"{s.manip:.6f}", int(s.in_collision), f"{s.joint_penalty:.6f}",
            ])
    print(f"\n  数据已保存: {path}")


def plot_workspace(results: list[WorkspaceSample], z_slice: float | None = None) -> None:
    """3D + 2D 投影可视化."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [skip] matplotlib 未安装，跳过可视化")
        return

    positions = np.array([s.eef_pos for s in results])
    ik_ok = np.array([s.ik_success for s in results])
    ok_pos = positions[ik_ok]
    fail_pos = positions[~ik_ok]
    ok_samples = [s for s in results if s.ik_success]
    manips = np.array([s.manip for s in ok_samples])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    titles = ["XY (top view)", "XZ (side view)", "YZ (front view)",
              "XY - Manipulability", "XY - Collision", "Radial Histogram"]
    xlabels = ["X (m)", "X (m)", "Y (m)", "X (m)", "X (m)", "Radial distance (m)"]
    ylabels = ["Y (m)", "Z (m)", "Z (m)", "Y (m)", "Y (m)", "Count"]

    # Row 1: 可达性 2D 投影
    pairs = [(0, 1), (0, 2), (1, 2)]
    for idx, (xi, yi) in enumerate(pairs):
        ax = axes[0, idx]
        if len(ok_pos) > 0:
            ax.scatter(ok_pos[:, xi], ok_pos[:, yi], s=0.5, c="steelblue", alpha=0.3, label="IK OK")
        if len(fail_pos) > 0:
            ax.scatter(fail_pos[:, xi], fail_pos[:, yi], s=0.5, c="red", alpha=0.5, label="IK fail")
        ax.set_xlabel(xlabels[idx])
        ax.set_ylabel(ylabels[idx])
        ax.set_title(titles[idx])
        ax.legend(markerscale=10, loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Row 2, Col 0: Manipulability XY
    ax = axes[1, 0]
    if len(ok_pos) > 0:
        vmin, vmax = np.percentile(manips, [5, 95])
        sc = ax.scatter(ok_pos[:, 0], ok_pos[:, 1], s=0.5, c=manips, cmap="viridis",
                        alpha=0.5, vmin=vmin, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Manipulability")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(titles[3])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1: 碰撞点 XY
    ax = axes[1, 1]
    col_pos = np.array([s.eef_pos for s in ok_samples if s.in_collision])
    safe_pos = np.array([s.eef_pos for s in ok_samples if not s.in_collision])
    if len(safe_pos) > 0:
        ax.scatter(safe_pos[:, 0], safe_pos[:, 1], s=0.5, c="green", alpha=0.2, label="safe")
    if len(col_pos) > 0:
        ax.scatter(col_pos[:, 0], col_pos[:, 1], s=2.0, c="red", alpha=0.8, label="collision")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(titles[4])
    ax.legend(markerscale=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: 径向距离直方图
    ax = axes[1, 2]
    radial = np.linalg.norm(ok_pos[:, :2], axis=1)
    ax.hist(radial, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(radial.mean(), color="red", linestyle="--", label=f"mean={radial.mean():.2f}m")
    ax.axvline(radial.max(), color="orange", linestyle="--", label=f"max={radial.max():.2f}m")
    ax.set_xlabel(xlabels[5])
    ax.set_ylabel(ylabels[5])
    ax.set_title(titles[5])
    ax.legend()

    title = f"xArm7 Workspace Exploration ({len(results)} samples"
    if z_slice is not None:
        title += f", z={z_slice}m"
    title += ")"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    out = f"/tmp/workspace_{'slice'+str(z_slice) if z_slice else 'full'}.png"
    fig.savefig(out, dpi=150)
    print(f"\n  图表已保存: {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="xArm7 工作空间探索")
    parser.add_argument("--dense", action="store_true", help="密集采样 (200k)")
    parser.add_argument("--slice-z", type=float, default=None, help="固定 Z 高度切片 (m)")
    parser.add_argument("--collision-urdf", action="store_true", help="使用带手部碰撞几何的 URDF")
    parser.add_argument("--no-plot", action="store_true", help="跳过可视化")
    parser.add_argument("--csv", type=str, default=None, help="CSV 输出路径")
    args = parser.parse_args()

    n = 200_000 if args.dense else 30_000

    print(f"xArm7 工作空间探索")
    print(f"  采样数: {n}")
    print(f"  URDF:   {'collision (xhand)' if args.collision_urdf else 'arm only'}")
    if args.slice_z is not None:
        print(f"  Z 切片: {args.slice_z}m")
    print()

    planner = make_planner(use_collision_urdf=args.collision_urdf)
    results = sample_workspace(planner, n, z_slice=args.slice_z)

    stats = analyze_results(results)
    print_report(stats)

    if args.csv:
        save_csv(results, args.csv)

    if not args.no_plot:
        plot_workspace(results, z_slice=args.slice_z)


if __name__ == "__main__":
    main()
