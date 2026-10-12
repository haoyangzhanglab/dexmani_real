"""本次 session 的只读离线分析；仅写入旁边的分析产物。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dexmani_analysis_mpl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.dataset.pointcloud import (
    RawEpisodePointCloudDeriver,
    load_raw_episode_base_from_color,
    load_raw_episode_camera_model,
)
from dexmani_real.dataset.processing import validate_export_episode
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.recording.storage.reader import EpisodeReader, RawDataError
from dexmani_real.recording.storage.schema import validate_capture_rows, validate_training_rows
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES

SESSION = ROOT / "rollouts/maniflow/pick_place_toy/2026-10-11_02-00-52-803763_1cbdb2c1_1011/session_20261011_220535"
OUT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((SESSION / "run_config.yaml").read_text())
RESULT = json.loads((SESSION / "session_result.json").read_text())
RAD_TO_DEG = 180 / np.pi


def stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return dict(zip(("min", "median", "p95", "p99", "max"),
                    map(float, np.percentile(values, [0, 50, 95, 99, 100]))),
                mean=float(values.mean()), count=int(values.size))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_extremes(values, t, names, count=8):
    values = np.asarray(values)
    indices = np.argsort(np.nan_to_num(np.abs(values), nan=-1).reshape(-1))[-count:][::-1]
    result = []
    for index in indices:
        row, joint = np.unravel_index(index, values.shape)
        result.append(dict(row=int(row), time_s=float(t[row]), joint_index=int(joint),
                           joint=names[joint], value=float(values[row, joint])))
    return result


def blocks(mask, t):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    groups = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return [dict(start_row=int(g[0]), end_row=int(g[-1]), start_s=float(t[g[0]]),
                 end_s=float(t[g[-1]]), rows=len(g)) for g in groups]


def sheet(frames, ids, t, filename, label):
    thumbs = []
    for index in ids:
        thumb = cv2.resize(frames[index], (480, 360))
        cv2.rectangle(thumb, (0, 0), (480, 26), (10, 10, 10), -1)
        cv2.putText(thumb, f"{label} row {index} / {t[index]:.3f}s", (7, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        thumbs.append(thumb)
    while len(thumbs) % 4:
        thumbs.append(np.zeros_like(thumbs[0]))
    grid = np.concatenate([np.concatenate(thumbs[i:i+4], axis=1)
                           for i in range(0, len(thumbs), 4)], axis=0)
    cv2.imwrite(str(OUT / filename), grid)


def analyze(ep, session_result):
    with h5py.File(ep / "data.h5", "r") as raw:
        data = {k: v[:] for k, v in raw.items() if isinstance(v, h5py.Dataset) and k != "depth"}
        rows = raw["timing/rows"][:]
        queries = [json.loads(v) for v in raw["timing/queries"].asstr()[:]]
        termination = json.loads(raw["timing/termination"].asstr()[()])
        depth = raw["depth"][:]
        meta = dict(raw["meta"].attrs)
    t = data["timestamp"]
    n = len(t)
    dispatch = data["dispatch_status"]
    sent = (dispatch == 1).all(axis=1)
    ids = np.flatnonzero(sent)
    send_ns = rows["submit_started_ns"][ids]
    completed_ns = rows["submit_completed_ns"][ids]
    qmap = {q["query_id"]: q for q in queries}
    report = dict(rows=n, duration_observations_s=float(t[-1]),
                  session_result=session_result, termination=termination,
                  meta={k: np.asarray(v).tolist() for k, v in meta.items()},
                  capture_errors=validate_capture_rows(t, dispatch),
                  training_row_errors=validate_training_rows(t, dispatch, 16),
                  fields={}, timing={}, motion={}, hand={}, image={})
    for key, value in data.items():
        finite = np.isfinite(value)
        report["fields"][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
            nonfinite_values=int((~finite).sum()),
            nonfinite_rows=np.flatnonzero(~finite.reshape(n, -1).all(axis=1)).tolist(),
            values=stats(value))
    unique, counts = np.unique(dispatch, axis=0, return_counts=True)
    report["dispatch"] = dict(status_counts={str(s.tolist()): int(c) for s, c in zip(unique, counts)},
        not_called_rows=np.flatnonzero(~sent).tolist(),
        nonfinite_targets_exactly_match_not_called=all(
            np.array_equal(np.isnan(data[f"action_{part}_joint_target"]).all(axis=1), ~sent)
            for part in ("arm", "hand")),
        complete_sent_targets=all(np.isfinite(data[f"action_{part}_joint_target"][sent]).all().item()
                                  for part in ("arm", "hand")))
    timing = report["timing"]
    timing["interval_ms"] = stats(np.diff(t) * 1000)
    timing["grid_error_ms"] = stats((t - np.arange(n) / 16) * 1000)
    timing["timestamp_equals_relative_observation_ns"] = bool(np.allclose(
        t, (rows["observation_ns"] - rows["observation_ns"][0]) / 1e9, atol=1e-12, rtol=0))
    timing["missing_slots"] = int(np.maximum(np.diff(rows["slot"]) - 1, 0).sum())
    timing["sources"] = {}
    for name in ("arm", "hand", "camera", "pointcloud"):
        stamps = rows[name + "_ns"]
        timing["sources"][name] = dict(age_ms=stats((rows["observation_ns"] - stamps) / 1e6),
            future_rows=np.flatnonzero(stamps > rows["observation_ns"]).tolist(),
            missing_rows=np.flatnonzero(stamps <= 0).tolist(),
            repeated=int((np.diff(stamps) == 0).sum()), backwards=int((np.diff(stamps) < 0).sum()))
    for name, a, b in (
        ("submit_duration_ms", "submit_completed_ns", "submit_started_ns"),
        ("camera_receive_to_publish_ms", "camera_published_ns", "camera_received_ns"),
        ("pointcloud_processing_ms", "pointcloud_published_ns", "pointcloud_started_ns"),
        ("pointcloud_publish_to_observation_ms", "observation_ns", "pointcloud_published_ns"),
        ("observation_sampling_ms", "observation_ns", "control_ns"),
        ("oldest_camera_receive_to_frameset_ms", "camera_received_ns", "camera_ns"),
    ):
        mask = (rows[a] > 0) & (rows[b] > 0)
        timing[name] = stats((rows[a][mask] - rows[b][mask]) / 1e6)
    timing["camera_pointcloud_sequence_mismatch_rows"] = np.flatnonzero(
        rows["camera_sequence"] != rows["pointcloud_camera_sequence"]).tolist()
    timing["camera_number_repeats"] = {k: int((np.diff(rows[k]) == 0).sum()) for k in
        ("camera_depth_frame_number", "camera_color_frame_number")}
    timing["query_counts"] = {s: sum(q["status"] == s for q in queries)
                              for s in set(q["status"] for q in queries)}
    timing["inference_ms"] = stats([(q["completed_ns"] - q["started_ns"]) / 1e6
                                   for q in queries if q["completed_ns"] is not None])
    timing["missing_query_refs"] = sorted(set(map(int, rows["query_id"][sent])) - qmap.keys())
    timing["send_interval_ms"] = stats(np.diff(send_ns) / 1e6)
    timing["send_gaps_over_100ms"] = int((np.diff(send_ns) > 100e6).sum())
    timing["effective_dispatch_hz"] = float((len(ids) - 1) * 1e9 / (send_ns[-1] - send_ns[0]))
    latest_ages, history_ages, first_ages, last_ages = [], [], [], []
    for rowid in ids:
        q = qmap[int(rows["query_id"][rowid])]
        obs = q["observations"]
        latest = min(v for stamps in obs[-1]["policy_sources"].values() for v in stamps)
        oldest = min(v for ob in obs for stamps in ob["policy_sources"].values() for v in stamps)
        age = (rows["submit_started_ns"][rowid] - latest) / 1e6
        latest_ages.append(age)
        history_ages.append((rows["submit_started_ns"][rowid] - oldest) / 1e6)
        if rows["prediction_index"][rowid] == 0:
            first_ages.append(age)
        if rows["prediction_index"][rowid] == 7:
            last_ages.append(age)
    timing["decision_latest_sources_age_ms"] = stats(latest_ages)
    timing["decision_oldest_history_sources_age_ms"] = stats(history_ages)
    timing["chunk_first_action_source_age_ms"] = stats(first_ages)
    timing["chunk_last_action_source_age_ms"] = stats(last_ages)

    for part, names in (("arm", [f"J{i+1}" for i in range(7)]), ("hand", list(XHAND_SDK_JOINT_NAMES))):
        q = data[f"{part}_qpos"]
        action = data[f"action_{part}_joint_target"]
        command_delta = np.diff(action[sent], axis=0) * RAD_TO_DEG
        jump = np.max(np.abs(command_delta), axis=1)
        boundary = rows["query_id"][ids[1:]] != rows["query_id"][ids[:-1]]
        prev = np.searchsorted(completed_ns, rows[f"{part}_ns"], side="right") - 1
        valid = prev >= 0
        prior_error = np.full_like(q, np.nan)
        prior_error[valid] = (action[ids[prev[valid]]] - q[valid]) * RAD_TO_DEG
        # 同行 qpos 采于发送前；这是新目标的前瞻量，不是到达误差。
        target_ahead = (action - q) * RAD_TO_DEG
        top = np.argsort(jump)[-8:][::-1]
        motion = dict(joint_names=names,
            measured_min_deg=(q.min(axis=0) * RAD_TO_DEG).tolist(),
            measured_max_deg=(q.max(axis=0) * RAD_TO_DEG).tolist(),
            target_ahead_deg=stats(np.abs(target_ahead)),
            target_ahead_largest=event_extremes(target_ahead, t, names),
            previous_sent_target_error_deg=stats(np.abs(prior_error)),
            previous_sent_target_error_largest=event_extremes(prior_error, t, names),
            command_jump_deg=stats(jump),
            boundary_jump_deg=stats(jump[boundary]), inside_chunk_jump_deg=stats(jump[~boundary]),
            boundary_jump_per_elapsed_second_deg_s=stats(jump[boundary] / (np.diff(send_ns)[boundary] / 1e9)),
            inside_jump_per_elapsed_second_deg_s=stats(jump[~boundary] / (np.diff(send_ns)[~boundary] / 1e9)),
            command_jump_largest=[dict(row=int(ids[j+1]), time_s=float(t[ids[j+1]]),
                jump_deg=float(jump[j]), joint=names[int(np.argmax(np.abs(command_delta[j])))],
                at_chunk_boundary=bool(boundary[j]), interval_ms=float((send_ns[j+1]-send_ns[j])/1e6))
                for j in top],
            measured_velocity_fd_deg_s=stats(np.diff(q, axis=0) * RAD_TO_DEG / np.diff(rows[f"{part}_ns"] / 1e9)[:, None]))
        lower = CONFIG["runtime"][part]["joint_limit_lower" if part == "arm" else "mechanical_qpos_min_rad"]
        upper = CONFIG["runtime"][part]["joint_limit_upper" if part == "arm" else "mechanical_qpos_max_rad"]
        motion["measured_outside_mechanical_limits_rows"] = np.flatnonzero(((q < lower) | (q > upper)).any(axis=1)).tolist()
        motion["target_outside_mechanical_limits_rows"] = np.flatnonzero(((action < lower) | (action > upper)).any(axis=1)).tolist()
        report["motion"][part] = motion
    report["motion"]["arm"]["sdk_velocity_deg_s"] = stats(data["arm_qvel"] * RAD_TO_DEG)
    measured_eef = compute_eef_pose_history_xarm_base(data["arm_qpos"])[:, :3]
    target_eef = np.full((n, 3), np.nan)
    target_eef[sent] = compute_eef_pose_history_xarm_base(data["action_arm_joint_target"][sent])[:, :3]
    workspace = CONFIG["runtime"]["policy"]["workspace"]
    lower = np.array([workspace[axis+"_min"] for axis in "xyz"])
    upper = np.array([workspace[axis+"_max"] for axis in "xyz"])
    margin = CONFIG["execution"]["eef_workspace_margin_m"]
    report["motion"]["eef"] = dict(measured_min_m=measured_eef.min(axis=0).tolist(),
        measured_max_m=measured_eef.max(axis=0).tolist(),
        target_min_m=np.nanmin(target_eef, axis=0).tolist(), target_max_m=np.nanmax(target_eef, axis=0).tolist(),
        measured_outside_workspace_with_margin=np.flatnonzero(((measured_eef < lower-margin) | (measured_eef > upper+margin)).any(axis=1)).tolist(),
        target_outside_workspace_with_margin=np.flatnonzero(((target_eef < lower-margin) | (target_eef > upper+margin)).any(axis=1)).tolist())
    current = data["hand_current"]
    limit = np.asarray(CONFIG["runtime"]["hand"]["tor_max_ma"])
    report["hand"] = dict(current_extremes=event_extremes(current, t, XHAND_SDK_JOINT_NAMES),
        position_mode_tor_max_raw=limit.tolist(),
        current_abs_gt_600_intervals=blocks((np.abs(current)>600).any(axis=1), t),
        current_abs_gt_1200_intervals=blocks((np.abs(current)>1200).any(axis=1), t),
        current_abs_max_per_joint_ma=np.abs(current).max(axis=0).tolist(),
        current_note="厂商手册确认 finger_state.torque 单位 mA；位置模式 effort_limit 为 0–400 比例值，400 对应 100% 输出，不能与 mA 直接比较。配置 tor_max_ma 的命名不构成单位依据。1200 mA 为本地厂商手册参考值，未核实采集时固件阈值。")
    hand_action = data["action_hand_joint_target"]
    hand_lower = np.asarray(CONFIG["runtime"]["hand"]["qpos_min_rad"])
    hand_upper = np.asarray(CONFIG["runtime"]["hand"]["qpos_max_rad"])
    report["hand"]["command_at_operational_lower_per_joint"] = np.isclose(
        hand_action, hand_lower, atol=1e-8, rtol=0).sum(axis=0).tolist()
    report["hand"]["command_at_operational_upper_per_joint"] = np.isclose(
        hand_action, hand_upper, atol=1e-8, rtol=0).sum(axis=0).tolist()
    report["hand"]["last_measured_qpos_deg"] = (data["hand_qpos"][-1] * RAD_TO_DEG).tolist()
    report["hand"]["last_dispatched_target_deg"] = (hand_action[ids[-1]] * RAD_TO_DEG).tolist()
    report["hand"]["last_current_ma"] = current[-1].tolist()
    report["hand"]["last_aggregate_native"] = data["hand_contact"][-1].tolist()
    for key in ("hand_contact", "hand_tactile_force"):
        value = data[key]
        report["hand"][key] = dict(
            all_zero_rows=int((value.reshape(n, -1) == 0).all(axis=1).sum()),
            finger_nonzero_rows=(value.reshape(n, 5, -1) != 0).any(axis=2).sum(axis=0).tolist(),
            finger_peak_abs=np.abs(value.reshape(n, 5, -1)).max(axis=(0, 2)).tolist(),
            note="SDK native; tactile zero not requested; no subtraction or Newton conversion.")
    late_from = {"episode_001": 21., "episode_002": 9., "episode_003": 10.}[ep.name]
    late = t >= late_from
    report["hand"]["post_lift_or_release"] = dict(from_s=late_from,
        rows=int(late.sum()), index_splay_current_ma=stats(current[late, 3]),
        index_contact_norm_native=stats(np.linalg.norm(data["hand_contact"][late, 1], axis=1)),
        note="区间由视频人工选定：第一段释放后，第二、三段空手抬起后；不把触觉非零等同于持物。")

    frames, rgb_difference, duplicate_rgb = [], [], []
    video = cv2.VideoCapture(str(ep / "rgb.mp4"))
    fps = video.get(cv2.CAP_PROP_FPS)
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if frames:
            rgb_difference.append(float(np.abs(frame.astype(float) - frames[-1]).mean()))
            if np.array_equal(frame, frames[-1]):
                duplicate_rgb.append(len(frames))
        frames.append(frame)
    video.release()
    report["image"] = dict(decoded_rgb_frames=len(frames), fps=fps,
        rgb_invalid_shape_rows=[i for i, frame in enumerate(frames) if frame.shape != (480, 640, 3)],
        rgb_exact_repeat_rows=duplicate_rgb, rgb_adjacent_mean_abs_difference=stats(rgb_difference),
        depth_shape=list(depth.shape), depth_dtype=str(depth.dtype),
        depth_nonzero_fraction=stats((depth != 0).mean(axis=(1, 2))),
        depth_all_zero_rows=np.flatnonzero((depth == 0).all(axis=(1, 2))).tolist(),
        depth_exact_repeat_rows=(np.flatnonzero((depth[1:] == depth[:-1]).all(axis=(1, 2))) + 1).tolist())
    overview_ids = np.unique(np.linspace(0, n-1, 12).astype(int))
    sheet(frames, overview_ids, t, f"{ep.name}_overview.jpg", ep.name)
    windows = {"episode_001": (13, 16.5), "episode_002": (6.5, 9.25), "episode_003": (8.25, 11)}
    event_ids = np.unique([int(np.argmin(abs(t - ts))) for ts in np.linspace(*windows[ep.name], 12)])
    sheet(frames, event_ids, t, f"{ep.name}_grasp.jpg", ep.name)
    with EpisodeReader(ep) as reader:
        try:
            validate_export_episode(reader)
            report["canonical_export_validation"] = "accepted"
        except RawDataError as exc:
            report["canonical_export_validation"] = str(exc)
        reader.read_timing()
        deriver = RawEpisodePointCloudDeriver(reader, load_raw_episode_camera_model(reader),
            load_raw_episode_base_from_color(reader), PointCloudConfig(**CONFIG["runtime"]["pointcloud"]),
            tuple(CONFIG["runtime"]["environment"]["table"]["plane_abcd"]))
        invalid_cloud, unique_cloud, centroids, bounds, cloud_snapshots = [], [], [], [], {}
        for index, frame in enumerate(frames):
            cloud = deriver.derive(index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if cloud is None or cloud.shape != (1024, 6) or not np.isfinite(cloud).all():
                invalid_cloud.append(index)
                continue
            unique_cloud.append(len(np.unique(cloud[:, :3], axis=0)))
            centroids.append(cloud[:, :3].mean(axis=0))
            bounds.append([cloud[:, :3].min(axis=0), cloud[:, :3].max(axis=0)])
            if index in (overview_ids[0], overview_ids[5], overview_ids[-1]):
                cloud_snapshots[index] = cloud
        report["image"]["reconstructed_cloud"] = dict(checked_frames=len(frames), invalid_rows=invalid_cloud,
            unique_xyz_count=stats(unique_cloud),
            bounds_m=[np.min(np.array(bounds)[:, 0], axis=0).tolist(), np.max(np.array(bounds)[:, 1], axis=0).tolist()],
            centroid_adjacent_step_m=stats(np.linalg.norm(np.diff(centroids, axis=0), axis=1)),
            note="当前源码 + run_config 中的实验桌面参数 + Raw 内外参重建；RGB 经 MP4 有损编码，非在线 tensor 的逐位复现。")

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .25})
    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    for j in range(7):
        color = plt.cm.tab10(j)
        axes[0].plot(t, data["arm_qpos"][:, j] * RAD_TO_DEG, color=color, label=f"J{j+1}")
        axes[0].plot(t[ids], data["action_arm_joint_target"][ids, j] * RAD_TO_DEG, color=color, ls="--", alpha=.55)
    axes[0].set_ylabel("Arm joint (deg)")
    axes[0].legend(ncol=7, loc="upper left")
    for j in range(3):
        axes[1].plot(t, measured_eef[:, j], label="XYZ"[j])
        axes[1].plot(t[ids], target_eef[ids, j], ls="--", alpha=.5)
    axes[1].set_ylabel("EEF base position (m)")
    axes[1].legend(ncol=3)
    axes[2].plot(t, current.min(axis=1), label="minimum current")
    axes[2].plot(t, current.max(axis=1), label="maximum current")
    axes[2].axhline(1200, color="gray", ls=":", label="manual reference (+/-1200 mA)")
    axes[2].axhline(-1200, color="gray", ls=":")
    axes[2].set_ylabel("Hand current (SDK mA)")
    axes[2].legend()
    for j, name in enumerate(("thumb", "index", "middle", "ring", "pinky")):
        axes[3].plot(t, np.linalg.norm(data["hand_contact"][:, j], axis=1), label=name)
    axes[3].set_ylabel("Contact norm (SDK native)")
    axes[3].legend(ncol=5)
    axes[4].plot(t[1:], np.diff(t)*1e3, label="observation interval")
    axes[4].plot(t[ids[1:]], np.diff(send_ns)/1e6, label="dispatch interval", alpha=.7)
    axes[4].plot(t, (rows["observation_ns"]-rows["camera_ns"])/1e6, label="camera host age")
    axes[4].set_ylabel("Time (ms)")
    axes[4].set_xlabel("Seconds since first Raw observation")
    axes[4].legend(ncol=3)
    fig.suptitle(f"{ep.name}: observations, targets, current and timing (dashed = target)")
    fig.tight_layout()
    fig.savefig(OUT / f"{ep.name}_signals.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for j, color in ((3, "tab:blue"), (10, "tab:orange"), (11, "tab:green")):
        axes[0].plot(t, data["hand_qpos"][:, j] * RAD_TO_DEG, color=color,
                     label=f"SDK {j}: {XHAND_SDK_JOINT_NAMES[j].removeprefix('right_hand_')}")
        axes[0].plot(t[ids], hand_action[ids, j] * RAD_TO_DEG, color=color, ls="--", alpha=.7)
        axes[1].plot(t, current[:, j], color=color, label=f"SDK {j}")
    axes[0].set_ylabel("Hand angle (deg)")
    axes[0].legend()
    axes[1].set_ylabel("Current (SDK mA)")
    axes[1].legend()
    for j in range(5):
        axes[2].plot(t, np.linalg.norm(data["hand_contact"][:, j], axis=1),
                     label=("thumb", "index", "middle", "ring", "pinky")[j])
    axes[2].set_ylabel("Contact norm (SDK native)")
    axes[2].set_xlabel("Seconds since first Raw observation")
    axes[2].legend(ncol=5)
    fig.suptitle(f"{ep.name}: index and pinky response (solid = measured; dashed = target)")
    fig.tight_layout()
    fig.savefig(OUT / f"{ep.name}_hand.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(cloud_snapshots), figsize=(15, 5))
    for ax, (index, cloud) in zip(axes, cloud_snapshots.items()):
        ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 3:6], s=5)
        ax.set(title=f"row {index}, {t[index]:.2f}s", xlabel="base X (m)", ylabel="base Y (m)",
               xlim=(0, .8), ylim=(-.5, .5), aspect="equal")
    fig.tight_layout()
    fig.savefig(OUT / f"{ep.name}_cloud.png", dpi=130)
    plt.close(fig)
    return report


def main():
    evidence = dict(session_path=str(SESSION), notes={
        "row_indices": "0-based; all event times relative to first Raw observation",
        "target_semantics": "attempted absolute targets with SDK ACCEPTED, not measured arrival",
        "safety": "read-only offline analysis, no device connections or robot execution",
        "source_version": "current worktree is dirty; recorded HEAD alone does not identify capture-time source"},
        source_files={str(p.relative_to(SESSION)): dict(size=p.stat().st_size, sha256=sha256(p))
                      for p in sorted(SESSION.rglob("*")) if p.is_file()},
        current_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        current_git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        analysis_source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "dexmani_real/deployment/runner.py",
            ROOT / "dexmani_real/recording/recorder.py", ROOT / "dexmani_real/robot/drivers/xhand.py",
            ROOT / "dexmani_real/dataset/processing.py", ROOT / "dexmani_real/sensor/pointcloud.py",
            ROOT / "dexmani_real/planning/kinematics/arm_fk.py")},
        episodes={})
    for ep, result in zip(sorted(SESSION.glob("episode_*")), RESULT["episodes"]):
        print(f"Analyzing {ep.name}", flush=True)
        evidence["episodes"][ep.name] = analyze(ep, result)
        print(f"Completed {ep.name}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
    print(OUT / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
