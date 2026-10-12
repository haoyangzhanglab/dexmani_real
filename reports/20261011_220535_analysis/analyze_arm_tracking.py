"""Read-only analysis of host-command versus measured xArm motion; no device imports."""

from pathlib import Path
import hashlib
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
BASELINE = json.loads((OUT / "metrics.json").read_text())
SESSION = Path(BASELINE["session_path"])
LAGS = np.arange(0, 0.501, 0.001)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(zip(("min", "median", "p95", "max"),
                    map(float, np.percentile(x, (0, 50, 95, 100)))), count=int(x.size))


def evaluate_reference(t, send_t, u, method):
    if method == "linear":
        # Offline descriptive curve, not the firmware's internal reference.
        return np.interp(t, send_t, u)
    ix = np.searchsorted(send_t, t, side="right") - 1
    return u[np.maximum(ix, 0)]


def phase_fit(t, q, send_t, u, method):
    rmse = np.array([
        np.sqrt(np.mean((evaluate_reference(t - lag, send_t, u, method) - q) ** 2))
        for lag in LAGS
    ])
    best = int(np.argmin(rmse))
    return dict(best_lag_ms=float(1000 * LAGS[best]),
                unshifted_rmse_deg=float(rmse[0]), best_rmse_deg=float(rmse[best]),
                lag_search_hit_upper_bound=best == len(LAGS) - 1)


def analyze(ep):
    with h5py.File(ep / "data.h5", "r") as f:
        q = np.rad2deg(f["arm_qpos"][:])
        v = np.rad2deg(f["arm_qvel"][:])
        u_all = np.rad2deg(f["action_arm_joint_target"][:])
        rows = f["timing/rows"][:]
        ids = np.flatnonzero(f["dispatch_status"][:, 0] == 1)
        episode_t = f["timestamp"][:]
    base_ns = rows["observation_ns"][0]
    measured_t = (rows["arm_ns"] - base_ns) / 1e9
    start_t = (rows["submit_started_ns"][ids] - base_ns) / 1e9
    end_t = (rows["submit_completed_ns"][ids] - base_ns) / 1e9
    u = u_all[ids]
    previous = np.searchsorted(end_t, measured_t, side="right") - 1
    valid = previous >= 0
    error = np.full_like(q, np.nan)
    error[valid] = u[previous[valid]] - q[valid]
    dt = np.diff(end_t)
    slope = np.diff(u, axis=0) / dt[:, None]
    slope_change = np.diff(slope, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None]
    boundary = rows["query_id"][ids][1:] != rows["query_id"][ids][:-1]
    gaps = np.flatnonzero(dt > 0.09)
    gap_end_rows = ids[gaps + 1]
    gap_end_speed = np.max(abs(v[gap_end_rows]), axis=1)
    result = dict(
        accepted_targets=int(len(ids)),
        command_gap_ms=dict(within_chunk=stats(1000 * dt[~boundary]),
                            between_chunks=stats(1000 * dt[boundary])),
        observe_ms=stats((rows["observation_ns"] - rows["control_ns"]) / 1e6),
        sdk_arm_and_hand_ms=stats((rows["submit_completed_ns"][ids] - rows["submit_started_ns"][ids]) / 1e6),
        owner_observe_through_send_ms=stats((rows["submit_completed_ns"][ids] - rows["control_ns"][ids]) / 1e6),
        gap_end_arm_speed=dict(max_abs_joint_deg_s=stats(gap_end_speed),
                               more_than_1_deg_s_count=int((gap_end_speed > 1).sum()),
                               count=int(len(gaps))),
        joints={},
    )
    for j in range(7):
        peak_row = int(np.nanargmax(abs(error[:, j])))
        result["joints"][f"J{j+1}"] = dict(
            previous_completed_target_error_abs_deg=stats(abs(error[:, j])),
            peak_error_row=peak_row, peak_error_time_s=float(episode_t[peak_row]),
            measured_speed_abs_deg_s=stats(abs(v[:, j])),
            target_segment_slope_abs_deg_s=stats(abs(slope[:, j])),
            target_segment_slope_change_abs_deg_s2=stats(abs(slope_change[:, j])),
            dynamic_phase_windows=[],
        )
        for begin in np.arange(0.75, measured_t[-1] - 2.5, 1.0):
            m = (measured_t >= begin) & (measured_t < begin + 2.5)
            ref = evaluate_reference(measured_t[m], end_t, u[:, j], "linear")
            if m.sum() < 20 or np.ptp(ref) < 10 or np.ptp(q[m, j]) < 5:
                continue
            result["joints"][f"J{j+1}"]["dynamic_phase_windows"].append(dict(
                begin_s=float(begin), end_s=float(begin + 2.5),
                commanded_range_deg=float(np.ptp(ref)), measured_range_deg=float(np.ptp(q[m, j])),
                linear=phase_fit(measured_t[m], q[m, j], end_t, u[:, j], "linear"),
                zoh=phase_fit(measured_t[m], q[m, j], end_t, u[:, j], "zoh"),
            ))
    if ep.name == "episode_001":
        j = 4
        win = (episode_t >= 15) & (episode_t <= 17.5)
        sensitivity = {}
        for clock, time_axis in (("dispatch_start", start_t), ("dispatch_end", end_t)):
            for method in ("linear", "zoh"):
                sensitivity[f"{clock}_{method}"] = phase_fit(
                    measured_t[win], q[win, j], time_axis, u[:, j], method)
        result["J5_15_to_17_5s_phase_sensitivity"] = sensitivity
        result["J5_15_to_17_5s_target_change_deg"] = float(
            np.interp(17.5, end_t, u[:, j]) - np.interp(15, end_t, u[:, j]))
        fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), sharex=True)
        axes[0].step(end_t, u[:, j], where="post", label="Last completed host target", color="tab:orange")
        axes[0].plot(measured_t, q[:, j], label="Measured J5", color="tab:blue")
        lag = sensitivity["dispatch_end_linear"]["best_lag_ms"] / 1000
        axes[0].plot(measured_t, np.interp(measured_t - lag, end_t, u[:, j]),
                     "--", color="tab:green", label=f"Offline target curve shifted {lag*1000:.0f} ms")
        axes[0].set_ylabel("Joint angle (deg)")
        axes[1].plot(measured_t, v[:, j], label="Measured J5 velocity")
        axes[1].plot((end_t[:-1] + end_t[1:]) / 2, slope[:, j], color="tab:orange",
                     label="Average slope between targets")
        axes[1].set_ylabel("Velocity (deg/s)")
        axes[2].plot(measured_t, error[:, j], label="Previous completed target - measured")
        axes[2].axhline(0, color="black", linewidth=.6)
        axes[2].set(ylabel="Tracking error (deg)", xlabel="Time from first host observation (s)")
        for k in gaps:
            if 14.5 < end_t[k] < 18:
                for ax in axes:
                    ax.axvspan(end_t[k], end_t[k+1], color="grey", alpha=.12)
        for ax in axes:
            ax.set_xlim(14.5, 18)
            ax.grid(alpha=.25)
            ax.legend(loc="best", fontsize=9)
        fig.suptitle("Episode 001: J5 tracking; grey = 125 ms command gaps\nShifted curve is descriptive, not identified device latency")
        fig.tight_layout()
        fig.savefig(OUT / "arm_tracking_focus.png", dpi=150)
        plt.close(fig)
    return result


def main():
    verification = {name: sha256(SESSION / name) == entry["sha256"]
                    for name, entry in BASELINE["source_files"].items()}
    assert all(verification.values())
    result = dict(
        session_path=str(SESSION), raw_file_hashes_match=verification,
        analysis_source_sha256=sha256(__file__),
        units="degrees and seconds except explicit *_ms keys",
        methodology=[
            "Measured sample time is host arm_ns. Actual acquisition/firmware reference time is unavailable.",
            "Error uses the last target with combined arm+hand dispatch completed before arm_ns; no same-row future command comparison.",
            "The arm is sent first; true arm acceptance lies within combined dispatch interval. Start/end are sensitivity bounds, not exact motor timestamps.",
            "Linear interpolation of host targets is an offline curve only. ZOH represents the latest endpoint; neither reveals the internal Mode-6 planned reference.",
            "Window fits use 2.5 seconds, >=10 deg target range and >=5 deg measured range, 0..500 ms lag grid; overlapping windows are not independent trials.",
            "The 1 ms search grid does not imply 1 ms identification accuracy; observations are sampled at about 16 Hz and ZOH fits can have broad plateaus.",
            "Target slope and slope-change are discrete diagnostics, not actual speed or acceleration limits for step commands.",
        ], episodes={})
    for ep in sorted(SESSION.glob("episode_*")):
        result["episodes"][ep.name] = analyze(ep)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharey=True)
    for ax, (name, ep) in zip(axes, result["episodes"].items()):
        joint = "J4" if name == "episode_003" else "J5"
        windows = ep["joints"][joint]["dynamic_phase_windows"]
        for method in ("linear", "zoh"):
            ax.plot([(w["begin_s"] + w["end_s"]) / 2 for w in windows],
                    [w[method]["best_lag_ms"] for w in windows], "o", label=method)
        ax.set(title=f"{name} {joint}: dynamic 2.5 s windows", ylabel="Best phase shift (ms)")
        ax.grid(alpha=.25); ax.legend()
    axes[-1].set_xlabel("Window midpoint (s)")
    fig.suptitle("Apparent phase shift changes with trajectory and reference convention\nNo single fixed device delay is identified")
    fig.tight_layout()
    fig.savefig(OUT / "arm_tracking_phase_windows.png", dpi=150)
    plt.close(fig)
    (OUT / "arm_tracking_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, ep in result["episodes"].items():
        print(name, json.dumps({
            "owner_ms": ep["owner_observe_through_send_ms"],
            "gap_end_arm_speed": ep["gap_end_arm_speed"],
            "J5_error_deg": ep["joints"]["J5"]["previous_completed_target_error_abs_deg"],
            "J5_phase_sensitivity": ep.get("J5_15_to_17_5s_phase_sensitivity"),
        }))


if __name__ == "__main__":
    main()
