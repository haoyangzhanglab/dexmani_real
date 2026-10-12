"""只读补充分析：命令/反馈对齐、电流关联、名义手部几何与相位滞后。

这些量用于解释本次记录，不是固件辨识或真机安全验证。
"""

import json
from pathlib import Path

import hppfcl.hppfcl  # Register the collision-result vector bindings used by Pinocchio.
import pinocchio as pin

from analyze import (
    SESSION, OUT, ROOT, RAD_TO_DEG, XHAND_SDK_JOINT_NAMES,
    blocks, h5py, np, plt, sha256, stats,
)
from dexmani_real.robot.model import (
    HAND_SDK_TO_URDF_IDX, XHAND_RIGHT_URDF_PATH, XHAND_MODEL_DIR,
)


def hand_geometry():
    model = pin.buildModelFromUrdf(str(XHAND_RIGHT_URDF_PATH))
    geometry = pin.buildGeomFromUrdf(
        model, str(XHAND_RIGHT_URDF_PATH), pin.GeometryType.COLLISION,
        package_dirs=[str(XHAND_MODEL_DIR)],
    )
    # Inspect only non-adjacent thumb/index pairs, including pairs excluded by SRDF.
    thumb = [i for i, obj in enumerate(geometry.geometryObjects) if "thumb" in obj.name]
    index = [i for i, obj in enumerate(geometry.geometryObjects) if "index" in obj.name]
    for i in thumb:
        for j in index:
            geometry.addCollisionPair(pin.CollisionPair(i, j))
    return model, model.createData(), geometry, geometry.createData()


def geometry_evidence(q, target, t, late, model, md, geom, gd):
    measured_hits, target_hits, minimum = [], [], []
    pair_counts = {}
    for row in range(len(t)):
        qq = q[row, list(HAND_SDK_TO_URDF_IDX)]
        pin.computeCollisions(model, md, geom, gd, qq, False)
        hits = [i for i, result in enumerate(gd.collisionResults) if result.isCollision()]
        if hits:
            measured_hits.append(row)
        for pair_id in hits:
            pair = geom.collisionPairs[pair_id]
            label = (geom.geometryObjects[pair.first].name + " / " +
                     geom.geometryObjects[pair.second].name)
            pair_counts[label] = pair_counts.get(label, 0) + 1
        pin.computeDistances(model, md, geom, gd, qq)
        minimum.append(min(r.min_distance for r in gd.distanceResults) * 1000)
        if np.isfinite(target[row]).all():
            pin.computeCollisions(model, md, geom, gd,
                                  target[row, list(HAND_SDK_TO_URDF_IDX)], False)
            if any(r.isCollision() for r in gd.collisionResults):
                target_hits.append(row)
    minimum = np.array(minimum)
    return dict(
        measured_collision_rows=measured_hits,
        target_collision_rows=target_hits,
        post_lift_or_release_measured_collision_rows=[row for row in measured_hits if late[row]],
        post_lift_or_release_target_collision_rows=[row for row in target_hits if late[row]],
        measured_collision_pair_counts=pair_counts,
        measured_minimum_thumb_index_mesh_distance_mm=stats(minimum),
        post_lift_or_release_minimum_distance_mm=stats(minimum[late]),
        note="URDF nominal collision meshes only; zero distance does not measure penetration depth. Overlap/distance does not establish real contact, soft-pad deformation, zero calibration, or tactile bias.",
    )


def analyze_episode(ep, geometry):
    with h5py.File(ep / "data.h5", "r") as raw:
        data = {key: raw[key][:] for key in (
            "timestamp", "arm_qpos", "hand_qpos", "hand_current", "hand_contact",
            "action_arm_joint_target", "action_hand_joint_target", "dispatch_status",
        )}
        timing = raw["timing/rows"][:]
        queries = [json.loads(v) for v in raw["timing/queries"].asstr()[:]]
    t = data["timestamp"]
    ids = np.flatnonzero((data["dispatch_status"] == 1).all(axis=1))
    complete = timing["submit_completed_ns"][ids]
    base_ns = timing["observation_ns"][0]
    late_from = {"episode_001": 21, "episode_002": 9, "episode_003": 10}[ep.name]
    late = t >= late_from
    correlations = {}
    previous_errors = {}
    for part in ("arm", "hand"):
        q = data[part + "_qpos"]
        target = data["action_" + part + "_joint_target"]
        previous = np.searchsorted(complete, timing[part + "_ns"], side="right") - 1
        error = np.full_like(q, np.nan)
        valid = previous >= 0
        error[valid] = (target[ids[previous[valid]]] - q[valid]) * RAD_TO_DEG
        previous_errors[part] = error
    for j in (3, 10, 11):
        x = previous_errors["hand"][:, j]
        y = data["hand_current"][:, j]
        valid = np.isfinite(x) & np.isfinite(y)
        slope, offset = np.polyfit(x[valid], y[valid], 1)
        correlations[str(j)] = dict(
            joint_name=XHAND_SDK_JOINT_NAMES[j], rows=int(valid.sum()),
            pearson_r=float(np.corrcoef(x[valid], y[valid])[0, 1]),
            linear_slope_ma_per_deg=float(slope), linear_intercept_ma=float(offset),
            post_lift_or_release_previous_target_error_deg=stats(x[late]),
        )
    over1200 = np.abs(data["hand_current"][:, 10]) > 1200
    report = dict(
        current_vs_previous_completed_target_error=correlations,
        correlation_note="Descriptive association with autocorrelated time series, not independent causal identification. Motor signs differ between joints.",
        sdk10_abs_current_gt_1200_intervals=blocks(over1200, t),
        all_completed_queries_bootstrap=all(q.get("bootstrap") for q in queries if q["completed_ns"] is not None),
        post_lift_or_release_from_s=late_from,
        nominal_thumb_index_geometry=geometry_evidence(
            data["hand_qpos"], data["action_hand_joint_target"], t, late, *geometry),
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for j, color in ((3, "tab:blue"), (10, "tab:orange"), (11, "tab:green")):
        axes[0, 0].plot(t, previous_errors["hand"][:, j], color=color, label=f"SDK {j}")
        axes[1, 0].plot(t, data["hand_current"][:, j], color=color, label=f"SDK {j}")
        axes[0, 1].scatter(previous_errors["hand"][:, j], data["hand_current"][:, j],
                           color=color, s=8, alpha=.4, label=f"SDK {j}")
    axes[0, 0].set(ylabel="Previous completed target - measured (deg)", xlabel="Time (s)")
    axes[1, 0].set(ylabel="Current (mA)", xlabel="Time (s)")
    axes[0, 1].set(xlabel="Previous completed target - measured (deg)", ylabel="Current (mA)")
    for ax in (axes[0, 0], axes[1, 0], axes[0, 1]):
        ax.legend(); ax.grid(alpha=.25)
    arm_measured_t = (timing["arm_ns"] - base_ns) / 1e9
    send_t = (complete - base_ns) / 1e9
    joint = 4 if ep.name != "episode_003" else 3
    axes[1, 1].plot(arm_measured_t, data["arm_qpos"][:, joint] * RAD_TO_DEG, label="measured")
    axes[1, 1].step(send_t, data["action_arm_joint_target"][ids, joint] * RAD_TO_DEG,
                    where="post", label="last completed target", alpha=.65)
    axes[1, 1].set(xlabel="Time (s)", ylabel=f"Arm J{joint+1} (deg)")
    if ep.name == "episode_001":
        win = (t >= 15) & (t <= 17.5)
        errors = []
        lags = np.arange(0, .501, .001)
        for lag in lags:
            interp = np.interp(arm_measured_t[win] - lag, send_t,
                               data["action_arm_joint_target"][ids, joint])
            errors.append(float(np.sqrt(np.mean(((interp - data["arm_qpos"][win, joint]) * RAD_TO_DEG) ** 2))))
        best = int(np.argmin(errors))
        report["arm_j5_phase_alignment_15_to_17_5s"] = dict(
            lag_grid_ms=(lags * 1000).tolist(), rmse_deg=errors, best_lag_ms=float(lags[best] * 1000),
            unshifted_rmse_deg=errors[0], best_rmse_deg=errors[best],
            note="Fit to linearly interpolated target history; descriptive phase offset, not device latency or identified dynamics.",
        )
        axes[1, 1].set_xlim(14.5, 18)
    axes[1, 1].grid(alpha=.25); axes[1, 1].legend()
    fig.suptitle(f"{ep.name}: command-error/current association and arm response")
    fig.tight_layout()
    fig.savefig(OUT / f"{ep.name}_causes.png", dpi=140)
    plt.close(fig)
    return report


def main():
    baseline = json.loads((OUT / "metrics.json").read_text())["source_files"]
    current_hashes = {str(p.relative_to(SESSION)): sha256(p) for p in SESSION.rglob("*") if p.is_file()}
    assert current_hashes == {name: item["sha256"] for name, item in baseline.items()}
    geometry = hand_geometry()
    result = dict(source_files_unchanged=True, episodes={})
    for ep in sorted(SESSION.glob("episode_*")):
        result["episodes"][ep.name] = analyze_episode(ep, geometry)
        geom = result["episodes"][ep.name]["nominal_thumb_index_geometry"]
        print(ep.name, "measured/target mesh-overlap rows:",
              len(geom["measured_collision_rows"]), len(geom["target_collision_rows"]), flush=True)
    sources = [Path(__file__), XHAND_RIGHT_URDF_PATH,
               ROOT / "dexmani_real/robot/action.py", ROOT / "dexmani_real/planning/collision.py",
               ROOT / "dexmani_real/robot/drivers/xarm7.py",
               ROOT / "dexmani_real/robot/drivers/xhand.py",
               ROOT / "dexmani_real/deployment/runner.py",
               ROOT.parent / "dexmani_policy/dexmani_policy/deployment/runtime.py",
               ROOT.parent / "dexmani_policy/dexmani_policy/agents/action_decoders/consistency_flow.py"]
    result["analysis_source_sha256"] = {str(p): sha256(p) for p in sources}
    manuals = [
        Path("/home/zhanghaoyang/Desktop/硬件/Xhand/使用手册/[使用说明书]XHAND 1_带触觉单包_v1.3.pdf"),
        Path("/home/zhanghaoyang/Desktop/硬件/Xhand/SDK/Python/XHAND Python SDK接口说明文档 (1).pdf"),
        Path("/home/zhanghaoyang/Desktop/硬件/Xhand/SDK/ROS2/XHand Control ROS2使用文档 (2).pdf"),
    ]
    result["local_vendor_document_sha256"] = {str(p): sha256(p) for p in manuals}
    (OUT / "causal_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
