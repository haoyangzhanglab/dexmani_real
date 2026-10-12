"""Offline timing and tracking analysis of the four completed hardware trials.

Reads published Raw without changes. No device/runtime imports or SDK calls.
Run with the existing real_robot environment from any working directory.
"""
from pathlib import Path
import hashlib
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / "rollouts/maniflow/pick_place_toy/2026-10-11_02-00-52-803763_1cbdb2c1_1011"
TRIALS = [
    ("session_20261012_135345", "sync A8"),
    ("session_20261012_135924", "async A8"),
    ("session_20261012_140158", "async A4 P1"),
    ("session_20261012_140823", "async A4 P2"),
]


def stats(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(count=int(a.size), **dict(zip(
        ("min", "median", "p95", "max"),
        map(float, np.percentile(a, (0, 50, 95, 100))),
    )))


def read_trial(name):
    path = BASE / name / "episode_001/data.h5"
    with h5py.File(path, "r") as f:
        d = {k: v[:] for k, v in f.items()
             if isinstance(v, h5py.Dataset) and k != "depth"}
        rows = f["timing/rows"][:]
        queries = [json.loads(q) for q in f["timing/queries"].asstr()[:]]
    return path, d, rows, queries


def main():
    result = {"scope": "offline, no production changes, no hardware calls",
              "timing_caveat": "Host receive/call boundaries, not device sampling or arrival times. Arm read includes owner overhead; hand read includes parsing. Inference includes preprocessing, CPU scheduling, GPU work and CPU output transfer.",
              "trials": []}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (name, label) in zip(axes.flat, TRIALS):
        path, d, r, queries = read_trial(name)
        ids = np.flatnonzero((d["dispatch_status"] == 1).all(axis=1))
        qmap = {q["query_id"]: q for q in queries}
        boundary = np.r_[False, np.diff(r["query_id"][ids]) != 0]
        prefetch_slots = {q["slot"] for q in queries if not q["bootstrap"]}
        pref = np.isin(r["slot"][ids], list(prefetch_slots))
        ms = lambda a: np.asarray(a) / 1e6
        starts = r["submit_started_ns"][ids]
        ends = r["submit_completed_ns"][ids]
        complete = [q for q in queries if q["completed_ns"] is not None]
        steady = [q for q in complete if not q["bootstrap"]]
        deadline_slack = [(q["deadline_ns"] - q["completed_ns"]) / 1e6 for q in steady]
        source_age = {"joint_state": [], "point_cloud": []}
        source_age_tail = {k: [] for k in source_age}
        for j, i in enumerate(ids):
            q = qmap[int(r["query_id"][i])]
            sources = q["observations"][-1]["policy_sources"]
            tail = j + 1 < len(ids) and boundary[j + 1]
            for field in source_age:
                age = (starts[j] - min(sources[field])) / 1e6
                source_age[field].append(age)
                if tail:
                    source_age_tail[field].append(age)
        timings = {
            "control_to_arm_feedback_ms": stats(ms(r["arm_ns"] - r["control_ns"])),
            "arm_to_hand_feedback_ms": stats(ms(r["hand_ns"] - r["arm_ns"])),
            "hand_to_observation_ms": stats(ms(r["observation_ns"] - r["hand_ns"])),
            "observation_to_send_prefetch_ms": stats(ms(starts[pref] - r["observation_ns"][ids][pref])),
            "observation_to_send_ordinary_ms": stats(ms(starts[~pref] - r["observation_ns"][ids][~pref])),
            "combined_sdk_send_ms": stats(ms(ends - starts)),
            "control_to_send_start_ms": stats(ms(starts - r["control_ns"][ids])),
            "control_to_send_end_ms": stats(ms(ends - r["control_ns"][ids])),
            "chunk_boundary_send_gap_ms": stats(ms(np.diff(ends)[boundary[1:]])),
            "ordinary_send_gap_ms": stats(ms(np.diff(ends)[~boundary[1:]])),
            "inference_ms": stats([(q["completed_ns"] - q["started_ns"]) / 1e6 for q in complete]),
            "inference_thread_start_wait_ms": stats([(q["started_ns"] - q["submitted_ns"]) / 1e6 for q in complete]),
            "model_nominal_deadline_slack_ms": stats(deadline_slack),
            "source_age_all_ms": {k: stats(v) for k, v in source_age.items()},
            "source_age_completed_chunk_tail_ms": {k: stats(v) for k, v in source_age_tail.items()},
        }
        u = np.rad2deg(d["action_arm_joint_target"][ids])
        q = np.rad2deg(d["arm_qpos"])
        # A measurement before the new dispatch must not be compared to that dispatch.
        prev = np.searchsorted(ends, r["arm_ns"], side="right") - 1
        valid = prev >= 0
        error = u[prev[valid]] - q[valid]
        steps = abs(np.diff(u, axis=0))
        tracking = {
            "previous_completed_target_abs_error_deg_by_joint": [stats(abs(error[:, j])) for j in range(7)],
            "arm_target_jump_max_joint_deg_at_boundary": stats(np.max(steps[boundary[1:]], axis=1)),
            "arm_target_jump_max_joint_deg_within": stats(np.max(steps[~boundary[1:]], axis=1)),
            "qvel_abs_deg_s_by_joint": [stats(abs(np.rad2deg(d["arm_qvel"][:, j]))) for j in range(7)],
        }
        # Descriptive delay fit only in active motion; not a causal firmware-delay estimate.
        fit = []
        for j in range(7):
            at = (r["arm_ns"] - r["arm_ns"][0]) / 1e9
            ut = (ends - r["arm_ns"][0]) / 1e9
            select = (abs(np.rad2deg(d["arm_qvel"][:, j])) > 5) & (at > ut[0] + .5)
            candidates = np.arange(0, .501, .001)
            costs = []
            for delay in candidates:
                idx = np.searchsorted(ut, at[select] - delay, side="right") - 1
                costs.append(float(np.mean((u[np.maximum(idx, 0), j] - q[select, j]) ** 2)))
            best = int(np.argmin(costs))
            fit.append(dict(joint=j + 1, active_rows=int(select.sum()),
                            best_zoh_delay_ms=round(float(candidates[best] * 1000)),
                            raw_rmse_deg=float(np.sqrt(costs[0])),
                            shifted_rmse_deg=float(np.sqrt(costs[best]))))
        tracking["descriptive_zoh_delay_fit"] = fit
        tactile = {
            field: dict(nonfinite=int((~np.isfinite(d[field])).sum()),
                        peak_abs_by_finger=np.nanmax(abs(d[field]), axis=tuple(i for i in range(d[field].ndim) if i != 1)).tolist(),
                        nonzero_rows_by_finger=np.any(d[field] != 0, axis=tuple(range(2, d[field].ndim))).sum(axis=0).tolist())
            for field in ("hand_contact", "hand_tactile_force")
        }
        item = dict(session=name, label=label, raw_file=str(path.relative_to(ROOT)),
                    raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    timings=timings, tracking=tracking, tactile=tactile,
                    prediction_indices=sorted(map(int, np.unique(r["prediction_index"][ids]))),
                    async_model_late_count=int(sum(x < 0 for x in deadline_slack)),
                    send_duration_crosses_62_5ms_from_slot_control=int(np.sum(ends-r["control_ns"][ids] >= 62_500_000)))
        result["trials"].append(item)
        t = (ends - r["control_ns"][0]) / 1e9
        ax.plot(t[1:], np.diff(ends)/1e6, color="#4676a9", lw=.8)
        ax.scatter(t[1:][boundary[1:]], np.diff(ends)[boundary[1:]]/1e6,
                   s=13, color="#d56b25", label="chunk boundary")
        ax.axhline(62.5, c="gray", ls="--", lw=.8)
        ax.set(title=label, xlabel="time (s)", ylabel="dispatch interval (ms)", ylim=(40, 140))
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUT / "dispatch_intervals.png", dpi=150)
    plt.close(fig)
    # Existing read-only probe: CRC and finger-board warning coexist in the first frame.
    probe = np.load(ROOT / "reports/20261011_xhand_tactile/hardware_zero_probe_01/samples.npz")
    result["historical_crc_probe"] = [dict(
        row=int(i), sdk_code=int(probe["code"][i]),
        board_errors=probe["board_errors"][i].tolist(),
        qpos_finite=bool(np.isfinite(probe["qpos"][i]).all()),
        previous_sample_available=bool(i > 0),
        next_qpos_equal=bool(np.array_equal(probe["qpos"][i], probe["qpos"][i+1])),
    ) for i in np.flatnonzero(probe["code"])]
    (OUT / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for trial in result["trials"]:
        print(trial["label"], json.dumps(trial["timings"], ensure_ascii=False))


if __name__ == "__main__":
    main()
