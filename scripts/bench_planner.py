#!/usr/bin/env python3
"""Planner 遥操作热路径性能基准测试。

Usage:
    source /home/zhy/anaconda3/etc/profile.d/conda.sh && conda activate sim
    python scripts/bench_planner.py
    python scripts/bench_planner.py --full          # 含路径规划
    python scripts/bench_planner.py --profile       # 逐函数 profile
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np


def warm_cache():
    """消除 Python import / JIT 冷启动影响."""
    from dexmani_real.robot.planner import XArm7MotionPlanner, XArm7PlannerConfig, PlanningProfile

    config = XArm7PlannerConfig(
        urdf_path="/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xarm7/xarm7_glb.urdf",
        srdf_path="/home/zhy/Desktop/dexmani_real/dexmani_real/assets/robots/xarm7/xarm7_glb_mplib.srdf",
        eef_link_name="link_eef",
    )
    p = XArm7MotionPlanner(config)

    qpos = np.zeros(7)
    target = p.compute_eef_pose_world(qpos)
    p.solve_teleop_ik(target, qpos, qpos)
    p.solve_ik(target, qpos)
    p.plan_path(target, qpos)
    return p, config


def bench_teleop_ik_hotpath(planner) -> dict[str, Any]:
    print("\n" + "=" * 60)
    print("BENCH: Teleop IK 热路径 (模拟 50Hz 遥操作)")
    print("=" * 60)

    N = 500
    qpos = np.array([0.0, 0.2, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float64)
    prev_cmd = qpos.copy()
    base_pose = planner.compute_eef_pose_world(qpos)

    # 模拟 VR 追踪：EEF 在 y-z 平面随机晃动 ±5cm, ±10°
    rng = np.random.default_rng(42)
    noise_xyz = rng.normal(0, 0.03, (N, 3))
    noise_angle = rng.normal(0, np.deg2rad(5), (N,))
    noise_axis = rng.normal(0, 1, (N, 3))
    noise_axis = noise_axis / np.linalg.norm(noise_axis, axis=1, keepdims=True)

    targets = []
    for i in range(N):
        t = base_pose.copy()
        t.p = t.p + noise_xyz[i]
        half_angle = noise_angle[i] / 2
        q_noise = np.array([
            np.cos(half_angle),
            noise_axis[i, 0] * np.sin(half_angle),
            noise_axis[i, 1] * np.sin(half_angle),
            noise_axis[i, 2] * np.sin(half_angle),
        ])
        from dexmani_real.robot.planner.pose_utils import _quat_multiply
        t.q = _quat_multiply(base_pose.q, q_noise)
        targets.append(t)

    latencies_us = []
    successes = 0
    methods: dict[str, int] = {}

    for i in range(N):
        t0 = time.perf_counter()
        result = planner.solve_teleop_ik(targets[i], qpos, prev_cmd)
        dt = (time.perf_counter() - t0) * 1e6
        latencies_us.append(dt)

        if result.success and result.qpos is not None:
            successes += 1
            prev_cmd = result.qpos
            qpos = result.qpos  # 模拟闭环
            method = result.report.get("teleop_ik_method", "?")
            methods[method] = methods.get(method, 0) + 1

    latencies = np.array(latencies_us)
    fps = 1e6 / np.mean(latencies) if np.mean(latencies) > 0 else float("inf")

    print(f"  Samples:         {N}")
    print(f"  Success rate:    {successes}/{N} ({100*successes/N:.1f}%)")
    print(f"  Methods:         {methods}")
    print(f"  Latency mean:    {np.mean(latencies):.0f} us ({np.mean(latencies)/1000:.1f} ms)")
    print(f"  Latency p50:     {np.percentile(latencies, 50):.0f} us")
    print(f"  Latency p95:     {np.percentile(latencies, 95):.0f} us")
    print(f"  Latency p99:     {np.percentile(latencies, 99):.0f} us")
    print(f"  Latency max:     {np.max(latencies):.0f} us")
    print(f"  Est. max Hz:     {fps:.0f} Hz")
    print()

    budget_50hz = 20_000  # 50Hz = 20ms budget
    p95_ok = np.percentile(latencies, 95) < budget_50hz
    p99_ok = np.percentile(latencies, 99) < budget_50hz * 2
    print(f"  50Hz 达标:       p95 < 20ms: {'OK' if p95_ok else 'FAIL'}, p99 < 40ms: {'OK' if p99_ok else 'FAIL'}")

    return {
        "mean_us": float(np.mean(latencies)),
        "p95_us": float(np.percentile(latencies, 95)),
        "p99_us": float(np.percentile(latencies, 99)),
        "success_rate": successes / N,
        "est_hz": float(fps),
    }


def bench_teleop_ik_scenarios(planner) -> None:
    print("\n" + "=" * 60)
    print("BENCH: Teleop IK 不同偏移量下的行为")
    print("=" * 60)

    qpos = np.zeros(7)
    prev_cmd = qpos.copy()
    base_pose = planner.compute_eef_pose_world(qpos)

    offsets_m = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    print(f"  {'偏移 (m)':>10s}  {'方法':>20s}  {'延迟 (us)':>12s}  {'成功':>6s}  {'关节跳变 (°)':>14s}")
    print(f"  {'-'*65}")

    for offset in offsets_m:
        target = base_pose.copy()
        target.p = target.p + np.array([offset, 0.0, 0.0])

        t0 = time.perf_counter()
        result = planner.solve_teleop_ik(target, qpos, prev_cmd)
        dt = (time.perf_counter() - t0) * 1e6

        method = result.report.get("teleop_ik_method", result.report.get("fallback_method", "failed"))
        max_delta = 0.0
        if result.success and result.qpos is not None:
            max_delta = float(np.rad2deg(np.max(np.abs(result.qpos - qpos))))

        print(f"  {offset:>10.3f}  {method:>20s}  {dt:>10.0f} us  {'OK' if result.success else 'FAIL':>6s}  {max_delta:>12.1f}°")


def bench_ik_candidates(planner) -> None:
    print("\n" + "=" * 60)
    print("BENCH: IK 候选生成 (离线路径规划热路径)")
    print("=" * 60)

    from dexmani_real.robot.planner.planner_types import PlanningProfile, Pose

    qpos = np.array([0.0, 0.2, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float64)
    base_pose = planner.compute_eef_pose_world(qpos)
    target = Pose(p=base_pose.p + np.array([0.1, 0.1, 0.1]), q=base_pose.q)

    N = 50

    # Benchmark: FK
    t0 = time.perf_counter()
    for _ in range(N):
        planner.compute_eef_pose_world(qpos)
    fk_us = (time.perf_counter() - t0) / N * 1e6

    # Benchmark: Jacobian
    t0 = time.perf_counter()
    for _ in range(N):
        planner.compute_eef_jacobian(qpos)
    jac_us = (time.perf_counter() - t0) / N * 1e6

    # Benchmark: all IK candidates
    t0 = time.perf_counter()
    for _ in range(N):
        planner.collect_ik_candidates(target, qpos, planner.planning_profile)
    cand_us = (time.perf_counter() - t0) / N * 1e6

    # Benchmark: manipulability
    t0 = time.perf_counter()
    for _ in range(N):
        planner.compute_manipulability(qpos)
    manip_us = (time.perf_counter() - t0) / N * 1e6

    # Benchmark: canonicalize
    test_q = np.array([0.0, 0.2, 0.0, -3.1, 0.0, 1.0, 0.0], dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(N):
        planner.canonicalize_qpos(test_q, qpos)
    can_us = (time.perf_counter() - t0) / N * 1e6

    print(f"  compute_eef_pose_world:    {fk_us:>8.1f} us")
    print(f"  compute_eef_jacobian:      {jac_us:>8.1f} us")
    print(f"  compute_manipulability:    {manip_us:>8.1f} us")
    print(f"  canonicalize_qpos:         {can_us:>8.1f} us")
    print(f"  collect_ik_candidates:     {cand_us:>8.1f} us ({cand_us/1000:.1f} ms)")
    print()

    # 对比 score_ik_candidate 的单个函数耗时
    print("  IK candidate scoring breakdown (per candidate):")
    t0 = time.perf_counter()
    for _ in range(N * 10):
        planner.compute_qpos_delta(test_q, qpos)
    delta_us = (time.perf_counter() - t0) / (N * 10) * 1e6

    t0 = time.perf_counter()
    for _ in range(N * 10):
        planner.normalized_joint_distance(test_q, qpos)
    ndist_us = (time.perf_counter() - t0) / (N * 10) * 1e6

    t0 = time.perf_counter()
    for _ in range(N * 10):
        planner.joint_limit_penalty(test_q, planner.joint_limits)
    jpenalty_us = (time.perf_counter() - t0) / (N * 10) * 1e6

    print(f"    compute_qpos_delta:      {delta_us:>8.1f} us")
    print(f"    normalized_joint_dist:   {ndist_us:>8.1f} us")
    print(f"    joint_limit_penalty:     {jpenalty_us:>8.1f} us")


def bench_path_planning(planner) -> None:
    print("\n" + "=" * 60)
    print("BENCH: 路径规划")
    print("=" * 60)

    from dexmani_real.robot.planner.planner_types import Pose

    qpos = np.zeros(7)
    base_pose = planner.compute_eef_pose_world(qpos)
    target = Pose(p=base_pose.p + np.array([0.15, 0.1, 0.05]), q=base_pose.q)

    N = 10
    results = []
    latencies_ms = []

    for i in range(N):
        t0 = time.perf_counter()
        r = planner.plan_path(target, qpos)
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)
        results.append(r)

    successes = sum(1 for r in results if r.success)
    latencies = np.array(latencies_ms)

    print(f"  Attempts:        {N}")
    print(f"  Success:         {successes}/{N}")
    print(f"  Latency mean:    {np.mean(latencies):.0f} ms")
    print(f"  Latency p50:     {np.percentile(latencies, 50):.0f} ms")
    print(f"  Latency p95:     {np.percentile(latencies, 95):.0f} ms")
    print(f"  Latency max:     {np.max(latencies):.0f} ms")

    if successes > 0:
        for r in results:
            if r.success:
                n = r.qpos_path.shape[0]
                src = r.source
                score = r.report.get("path_score", float("nan"))
                eff = r.report.get("eef_efficiency", float("nan"))
                print(f"  Success: source={src}, waypoints={n}, score={score:.3f}, eef_eff={eff:.2f}")


def bench_substep_profile(planner) -> None:
    """逐函数 profile：分离 FK、IK、评分、路径验证各环节耗时."""
    print("\n" + "=" * 60)
    print("PROFILE: solve_teleop_ik 子步骤耗时分解")
    print("=" * 60)

    from dexmani_real.robot.planner.pose_utils import compute_pose_error
    from dexmani_real.robot.planner.planner_types import Pose

    qpos = np.array([0.0, 0.2, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float64)
    prev_cmd = qpos.copy()
    N = 500

    # 子步骤 1: FK
    t0 = time.perf_counter()
    for _ in range(N):
        current_pose = planner.compute_eef_pose_world(qpos)
    fk_us = (time.perf_counter() - t0) / N * 1e6

    # 子步骤 2: pose error calc
    target = planner.compute_eef_pose_world(qpos)
    t0 = time.perf_counter()
    for _ in range(N):
        compute_pose_error(target, current_pose)
    pe_us = (time.perf_counter() - t0) / N * 1e6

    # 子步骤 3: MPlib IK call (position_ik path)
    target_base = planner.world_to_base_pose(target)
    t0 = time.perf_counter()
    for _ in range(max(N//5, 1)):
        planner.call_mplib_ik(target_base, qpos, n_init_qpos=2, return_closest=True)
    mplib_ik_us = (time.perf_counter() - t0) / max(N//5, 1) * 1e6

    # 子步骤 4: canonicalize + delta compute
    raw_qpos = qpos + np.random.normal(0, 0.01, 7)
    t0 = time.perf_counter()
    for _ in range(N):
        c = planner.canonicalize_qpos(raw_qpos, prev_cmd)
        d = planner.compute_qpos_delta(c, prev_cmd)
    candelta_us = (time.perf_counter() - t0) / N * 1e6

    # 子步骤 5: Jacobian (differential IK path)
    t0 = time.perf_counter()
    for _ in range(N):
        planner.compute_eef_jacobian(qpos)
    jac_us = (time.perf_counter() - t0) / N * 1e6

    # 子步骤 6: 碰撞检查
    t0 = time.perf_counter()
    for _ in range(N):
        planner.has_self_collision(qpos)
    col_us = (time.perf_counter() - t0) / N * 1e6

    print(f"  FK (compute_eef_pose_world):    {fk_us:>8.1f} us")
    print(f"  Pose error (compute_pose_error): {pe_us:>8.1f} us")
    print(f"  MPlib IK call (n_init=2):       {mplib_ik_us:>8.1f} us")
    print(f"  Canonicalize + delta:           {candelta_us:>8.1f} us")
    print(f"  Jacobian (6x7):                 {jac_us:>8.1f} us")
    print(f"  Self-collision check:           {col_us:>8.1f} us")
    print()

    # 推估总热路径开销
    position_ik_path = fk_us + pe_us + mplib_ik_us + candelta_us + col_us
    diff_ik_path = fk_us + pe_us + jac_us + 500 + candelta_us + col_us  # ~500us for linear solve
    print(f"  Est. position_ik total:          {position_ik_path:>8.1f} us")
    print(f"  Est. differential_ik total:      {diff_ik_path:>8.1f} us")
    print(f"  50Hz budget (20ms):            20000.0 us")


def main():
    parser = argparse.ArgumentParser(description="Planner 性能基准测试")
    parser.add_argument("--full", action="store_true", help="含路径规划")
    parser.add_argument("--profile", action="store_true", help="子步骤 profile")
    args = parser.parse_args()

    print("Planner 性能基准测试")
    print()

    planner, _ = warm_cache()

    bench_teleop_ik_hotpath(planner)
    bench_teleop_ik_scenarios(planner)
    bench_ik_candidates(planner)

    if args.full:
        bench_path_planning(planner)

    if args.profile:
        bench_substep_profile(planner)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
