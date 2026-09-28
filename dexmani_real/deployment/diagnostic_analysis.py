"""Offline comparison of logical prediction overlap and host-clock execution."""

import json
from pathlib import Path

import h5py
import numpy as np
import yaml


def statistics(values):
    x = np.asarray(values, dtype=float).reshape(-1)
    if not len(x):
        return {"count": 0}
    return dict(
        count=len(x),
        median=float(np.median(x)),
        p95=float(np.percentile(x, 95)),
        max=float(np.max(x)),
    )


def joint_delta(a, b):
    """Only xArm's periodic joints wrap; hand and bounded arm joints do not."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta = delta.copy()
    for j in (0, 2, 4, 6):
        delta[..., j] = (delta[..., j] + np.pi) % (2 * np.pi) - np.pi
    return delta


def compare_chunks(old, new, *, n_obs_steps, n_action_steps):
    start = n_obs_steps - 1
    old_next = start + n_action_steps
    overlap = min(len(old) - old_next, len(new) - start)
    if overlap <= 0:
        raise ValueError("prediction horizon has no unexecuted overlap")
    return {
        "boundary": joint_delta(new[start], old[old_next - 1]),
        "planned_step": joint_delta(old[old_next], old[old_next - 1]),
        "revision": joint_delta(new[start], old[old_next]),
        "overlap": joint_delta(new[start : start + overlap], old[old_next : old_next + overlap]),
    }


def load_trace(path):
    with h5py.File(path, "r") as f:
        groups = {name: {key: ds[:] for key, ds in group.items()} for name, group in f.items()}
        complete = bool(f.attrs.get("complete", False))
    status_path = Path(path).with_suffix(".status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    return groups, complete and bool(status.get("complete", False)), status


def match_commands(publications, commands):
    """Exact run/sequence join, including explicit rejection and unobserved targets."""
    lookup = {
        (int(r), int(s)): i
        for i, (r, s) in enumerate(zip(commands.get("run_id", []), commands.get("sequence", [])))
    }
    pairs, absent, unsent = [], [], []
    for i in np.flatnonzero(publications["accepted"]):
        key = (int(publications["run_id"][i]), int(publications["sequence"][i]))
        j = lookup.get(key)
        if j is None:
            absent.append(key)
        elif commands["sdk_start_ns"][j] == 0:
            unsent.append(key)
        else:
            pairs.append((i, j))
    return pairs, absent, unsent


def analyze_session(session):
    session = Path(session)
    cfg = yaml.safe_load((session / "run_config.yaml").read_text())
    info = cfg["policy"]
    if info["action_mode"] != "joint":
        raise ValueError("This joint continuity analysis requires action_mode=joint")
    dt = info["control_dt_s"]
    nobs, nact = info["n_obs_steps"], info["n_action_steps"]
    traces, integrity = {}, {}
    for owner in ("policy", "arm", "hand"):
        groups, complete, status = load_trace(session / "diagnostics" / f"{owner}.h5")
        traces[owner] = groups
        integrity[owner] = {"complete": complete, **status}
        integrity[owner]["complete"] = complete
    q = traces["policy"].get("queries", {})
    p = traces["policy"].get("publications", {})
    if not q or not p:
        raise ValueError("No policy query/publication evidence; was B pressed?")
    result = {
        "integrity": integrity,
        "units": "degrees, milliseconds; host monotonic clock; overlap uses logical steps",
        "inference_ms": statistics((q["infer_end_ns"] - q["infer_start_ns"]) / 1e6),
        "publication_rejections": int(np.sum(~p["accepted"])),
        "runs": {},
    }
    query_lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(q["run_id"], q["chunk_id"]))}
    for run in np.unique(p["run_id"]):
        ids = np.flatnonzero((p["run_id"] == run) & p["accepted"])
        if not len(ids):
            continue
        pub_ns = p["publish_end_ns"][ids]
        boundary = p["chunk_id"][ids[1:]] != p["chunk_id"][ids[:-1]]
        intervals = np.diff(pub_ns) / 1e6
        published = np.c_[p["arm_target"][ids], p["hand_target"][ids]]
        raw = p["raw_action"][ids]
        steps = np.rad2deg(np.abs(joint_delta(published[1:], published[:-1])))
        run_report = {
            "published": len(ids),
            "publication_interval_ms": {
                "internal": statistics(intervals[~boundary]),
                "boundary": statistics(intervals[boundary]),
            },
            "projection_change_deg": statistics(
                np.max(np.abs(np.rad2deg(joint_delta(published, raw))), axis=1)
            ),
            "published_arm_step_deg": {
                "internal": statistics(steps[~boundary, :7].max(axis=1)),
                "boundary": statistics(steps[boundary, :7].max(axis=1)),
            },
            "boundaries": [],
        }
        for old_id, new_id in zip(ids[:-1][boundary], ids[1:][boundary]):
            oc, nc = int(p["chunk_id"][old_id]), int(p["chunk_id"][new_id])
            previous = ids[p["chunk_id"][ids] == oc]
            if nc != oc + 1 or not np.array_equal(p["chunk_step"][previous], np.arange(nact)):
                continue
            if p["chunk_step"][new_id] != 0:
                continue
            oi, ni = query_lookup[(int(run), oc)], query_lookup[(int(run), nc)]
            old, new = q["pred_action"][oi], q["pred_action"][ni]
            comparison = compare_chunks(old, new, n_obs_steps=nobs, n_action_steps=nact)
            entry = {
                "old_chunk": oc,
                "new_chunk": nc,
                "sequence": int(p["sequence"][new_id]),
                "query_drift_ms": float(
                    (q["observation_ns"][ni, -1] - q["observation_ns"][oi, -1]) / 1e6
                    - nact * dt * 1000
                ),
                "publish_interval_ms": float(
                    (p["publish_end_ns"][new_id] - p["publish_end_ns"][old_id]) / 1e6
                ),
                "new_inference_ms": float((q["infer_end_ns"][ni] - q["infer_start_ns"][ni]) / 1e6),
            }
            for part, sl in (("arm", slice(0, 7)), ("hand", slice(7, 19))):
                entry[part] = {
                    name + "_deg": np.rad2deg(value[..., sl]).tolist()
                    for name, value in comparison.items()
                }
                entry[part]["overlap_rmse_deg"] = float(
                    np.sqrt(np.mean(np.rad2deg(comparison["overlap"][..., sl]) ** 2))
                )
                a, b = comparison["boundary"][sl], comparison["planned_step"][sl]
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                entry[part]["boundary_vs_plan_cosine"] = (
                    float(a @ b / denom) if denom > 1e-12 else None
                )
            run_report["boundaries"].append(entry)
        worker_times = {}
        for owner in ("arm", "hand"):
            commands = traces[owner].get("commands", {})
            subset = {key: value[ids] for key, value in p.items()}
            matches, absent, unsent = match_commands(subset, commands)
            worker_report = {"unobserved_sequences": absent, "explicitly_unsent": unsent}
            if matches:
                pi, ci = np.array(matches).T
                send = commands["sdk_start_ns"][ci]
                send_boundary = subset["chunk_id"][pi[1:]] != subset["chunk_id"][pi[:-1]]
                spacing = np.diff(send) / 1e6
                # The ring commits inside the publication bracket, not at the returned end stamp.
                worker_report.update(
                    sent=len(ci),
                    sdk_status_counts={
                        str(s.decode()): int(n)
                        for s, n in zip(*np.unique(commands["status"][ci], return_counts=True))
                    },
                    publish_to_sdk_lower_ms=statistics((send - subset["publish_end_ns"][pi]) / 1e6),
                    publish_to_sdk_upper_ms=statistics(
                        (send - subset["publish_start_ns"][pi]) / 1e6
                    ),
                    sdk_call_ms=statistics((commands["sdk_end_ns"][ci] - send) / 1e6),
                    send_interval_ms={
                        "internal": statistics(spacing[~send_boundary]),
                        "boundary": statistics(spacing[send_boundary]),
                    },
                )
                qi = [query_lookup[(int(run), int(c))] for c in subset["chunk_id"][pi]]
                for sensor in ("arm", "hand", "cloud"):
                    stamps = q[f"{sensor}_ns"][qi, -1]
                    valid = stamps > 0
                    worker_report[f"model_{sensor}_age_at_send_ms"] = statistics(
                        (send[valid] - stamps[valid]) / 1e6
                    )
                worker_times[owner] = {int(subset["sequence"][i]): int(t) for i, t in zip(pi, send)}
            run_report[owner] = worker_report
        common = set(worker_times.get("arm", {})) & set(worker_times.get("hand", {}))
        run_report["hand_minus_arm_send_ms"] = statistics(
            [(worker_times["hand"][s] - worker_times["arm"][s]) / 1e6 for s in common]
        )
        result["runs"][str(int(run))] = run_report
    return result, traces


def write_analysis(session):
    session = Path(session)
    report, traces = analyze_session(session)
    out = session / "diagnostics" / "analysis"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base

    q, p = traces["policy"]["queries"], traces["policy"]["publications"]
    notes = [
        "# Rollout 诊断",
        "",
        "重叠按逻辑控制步比较；SDK 时间是主机调用时间，不是固件执行完成时间。",
        "",
        f"诊断完整性：{report['integrity']}",
        f"推理耗时 ms：{report['inference_ms']}",
    ]
    for run, metrics in report["runs"].items():
        ids = np.flatnonzero((p["run_id"] == int(run)) & p["accepted"])
        origin = int(p["publish_end_ns"][ids[0]])
        x = (p["publish_end_ns"][ids] - origin) / 1e9
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        boundaries = metrics["boundaries"]
        if boundaries:
            worst = max(boundaries, key=lambda b: max(abs(v) for v in b["arm"]["revision_deg"]))
            oi = np.flatnonzero((q["run_id"] == int(run)) & (q["chunk_id"] == worst["old_chunk"]))[
                0
            ]
            ni = np.flatnonzero((q["run_id"] == int(run)) & (q["chunk_id"] == worst["new_chunk"]))[
                0
            ]
            joint = int(np.argmax(np.abs(worst["arm"]["revision_deg"])))
            old, new = q["pred_action"][oi], q["pred_action"][ni]
            offset = len(q["control_action"][oi])
            # Put the new periodic joint on the old overlap branch for display only.
            aligned_new = old[offset] + joint_delta(new, old[offset])
            axes[0].plot(
                np.arange(len(old)), np.rad2deg(old[:, joint]), "o-", label="old full prediction"
            )
            axes[0].plot(
                offset + np.arange(len(new)),
                np.rad2deg(aligned_new[:, joint]),
                "o-",
                label="new full prediction",
            )
            axes[0].set(
                title=f"Run {run}: chunks {worst['old_chunk']} -> {worst['new_chunk']}, J{joint + 1}",
                xlabel="Logical prediction index (not wall time)",
                ylabel="deg",
            )
        else:
            joint = 4
        boundary = p["chunk_id"][ids[1:]] != p["chunk_id"][ids[:-1]]
        intervals = np.diff(x) * 1000
        axes[1].plot(x[1:], intervals, ".-", label="publication interval ms")
        axes[1].scatter(x[1:][boundary], intervals[boundary], color="red", label="chunk boundary")
        qi = np.flatnonzero(q["run_id"] == int(run))
        for i in qi:
            axes[1].axvspan(
                (q["infer_start_ns"][i] - origin) / 1e9,
                (q["infer_end_ns"][i] - origin) / 1e9,
                color="grey",
                alpha=0.15,
            )
        axes[1].set(xlabel="Host time since first publication (s)", ylabel="ms")
        axes[2].plot(x, np.rad2deg(p["arm_target"][ids, joint]), label="published target")
        for owner in ("arm", "hand"):
            c = traces[owner].get("commands", {})
            if c:
                ci = np.flatnonzero((c["run_id"] == int(run)) & (c["sdk_start_ns"] > 0))
                tx = (c["sdk_start_ns"][ci] - origin) / 1e9
                axes[1].plot(
                    tx,
                    np.full(len(tx), -5 if owner == "arm" else -10),
                    "|",
                    label=f"{owner} SDK calls",
                )
                if owner == "arm":
                    axes[2].plot(tx, np.rad2deg(c["target"][ci, joint]), ".", label="SDK target")
        f = traces["arm"].get("feedback", {})
        if f:
            fi = np.flatnonzero(f["run_id"] == int(run))
            axes[2].plot(
                (f["timestamp_ns"][fi] - origin) / 1e9,
                np.rad2deg(f["qpos"][fi, joint]),
                label="measured",
            )
        axes[2].set(xlabel="Host time since first publication (s)", ylabel=f"J{joint + 1} deg")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"run_{run}.png", dpi=140)
        plt.close(fig)
        poses = compute_eef_pose_history_xarm_base(p["arm_target"][ids])
        distances = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) * 1000
        metrics["eef_published_step_mm"] = {
            "internal": statistics(distances[~boundary]),
            "boundary": statistics(distances[boundary]),
        }
        notes.extend(
            [
                "",
                f"## Run {run}",
                f"发布间隔 ms：{metrics['publication_interval_ms']}",
                f"末端目标步进 mm：{metrics['eef_published_step_mm']}",
                f"arm：{metrics['arm']}",
                f"hand：{metrics['hand']}",
                f"hand−arm 发送差 ms：{metrics['hand_minus_arm_send_ms']}",
                f"![诊断图](run_{run}.png)",
            ]
        )
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "report.md").write_text("\n\n".join(notes) + "\n")
    return out
