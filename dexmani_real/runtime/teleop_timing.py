"""遥操作与回放的时序采集及离线跟踪分析。"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from bisect import bisect_right
from collections import defaultdict, deque
from math import ceil, radians
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np

from dexmani_real.ipc.ring import SharedRing
from dexmani_real.robot.model import ARM_JOINT_SHAPE
from dexmani_real.utils.io import publish, write_json
from dexmani_real.utils.timing import sample_is_fresh

TRACE_WINDOW_S = 120

SERVO_DTYPE = np.dtype([
    ("scheduled_ns", "<i8"),
    ("sdk_started_ns", "<i8"),
    ("sdk_finished_ns", "<i8"),
    ("goal_published_ns", "<i8"),
    ("qpos", "<f8", ARM_JOINT_SHAPE),
    ("succeeded", "?"),
])
FEEDBACK_DTYPE = np.dtype([
    ("started_ns", "<i8"),
    ("finished_ns", "<i8"),
    ("qpos", "<f8", ARM_JOINT_SHAPE),
    ("qvel", "<f8", ARM_JOINT_SHAPE),
    ("succeeded", "?"),
])


def copy_samples(ring: SharedRing) -> dict:
    """读取收尾时仍保留的样本，跳过未写完或已被覆盖的槽。"""
    last = ring.latest_sequence
    first = max(1, last - ring.capacity + 1)
    slots = ring.slots
    names = cast(tuple[str, ...], ring.dtype.names)
    samples = []
    for sequence in range(first, last + 1):
        slot = slots[sequence % ring.capacity]
        marker = sequence * 2
        if int(slot["sequence"]) != marker:
            continue
        sample = slot["data"].copy()
        if int(slot["sequence"]) == marker:
            samples.append({name: sample[name].tolist() for name in names})
    return {"dropped": last - len(samples), "samples": samples}


def summarize_control(samples: list[dict]) -> dict:
    durations: dict[str, list[float]] = {"hand_feedback": [], "hand_send": []}
    missed_slots = skipped_ticks = 0
    crc_counts = None
    for sample in samples:
        event = sample["event"]
        if event in durations:
            durations[event].append((sample["finished_ns"] - sample["timestamp_ns"]) / 1e6)
        elif event == "control_miss":
            missed_slots += sample["slots"]
        elif event == "control_skip":
            skipped_ticks += 1
        elif event == "xhand_crc":
            crc_counts = sample["counts"]
    return {
        **{
            event: {"samples": len(values), "p95_ms": float(np.percentile(values, 95)) if values else None}
            for event, values in durations.items()
        },
        "control_missed_slots": missed_slots,
        "control_skipped_ticks": skipped_ticks,
        "xhand_session_crc_counts": crc_counts,
    }


class ArmTiming:
    """xarm-owner 独占写入的两路诊断缓冲区。"""

    def __init__(self, ctx, arm):
        self.servo = SharedRing(ctx, SERVO_DTYPE, ceil(TRACE_WINDOW_S * arm.servo_hz))
        self.feedback = SharedRing(ctx, FEEDBACK_DTYPE, ceil(TRACE_WINDOW_S * arm.feedback_hz))

    def record_servo(self, *, scheduled_ns, sdk_started_ns, sdk_finished_ns, goal_published_ns, qpos, succeeded):
        self.servo.write(
            scheduled_ns=scheduled_ns, sdk_started_ns=sdk_started_ns, sdk_finished_ns=sdk_finished_ns,
            goal_published_ns=goal_published_ns, qpos=qpos, succeeded=succeeded,
        )

    def record_feedback(self, *, started_ns, finished_ns, qpos, qvel, succeeded):
        self.feedback.write(
            started_ns=started_ns, finished_ns=finished_ns,
            qpos=qpos if succeeded else 0.0, qvel=qvel if succeeded else 0.0, succeeded=succeeded,
        )


class TeleopTiming:
    def __init__(self, path, *, mode, control_hz, arm, motion):
        self.path = Path(path).expanduser().absolute()
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(self.path)
        self.arm = ArmTiming(mp.get_context("spawn"), arm)
        self.metadata = {
            "mode": mode,
            "clock": "monotonic_ns",
            "control_hz": float(control_hz),
            "servo_hz": float(arm.servo_hz),
            "feedback_hz": float(arm.feedback_hz),
            "feedback_max_age_s": float(arm.feedback_max_age_s),
            "ik_speed_rad_s": min(radians(motion.ik_target_speed_deg_s), arm.max_joint_velocity_rad_per_s),
            "joint_limit_lower_rad": np.asarray(arm.joint_limit_lower).tolist(),
            "joint_limit_upper_rad": np.asarray(arm.joint_limit_upper).tolist(),
        }
        # 一次控制步可能包含输入、求解、臂手 I/O 等多条事件。
        self.events: deque[dict] = deque(maxlen=ceil(TRACE_WINDOW_S * control_hz * 8))
        self.dropped = 0
        self.lock = threading.Lock()

    def add(self, event, *, timestamp_ns=None, **fields):
        row = {
            "event": event,
            "timestamp_ns": int(timestamp_ns) if timestamp_ns is not None else time.monotonic_ns(),
            **{
                name: value.tolist() if isinstance(value, (np.ndarray, np.generic)) else value
                for name, value in fields.items()
            },
        }
        with self.lock:
            if len(self.events) == self.events.maxlen:
                self.dropped += 1
            self.events.append(row)

    def write(self) -> Path:
        with self.lock:
            control_samples = list(self.events)
            control_dropped = self.dropped
        feedback = copy_samples(self.arm.feedback)
        for sample in feedback["samples"]:
            if not sample["succeeded"]:
                sample["qpos"] = sample["qvel"] = None
        payload = {
            "metadata": self.metadata,
            "control": {"dropped": control_dropped, "samples": control_samples},
            "servo": copy_samples(self.arm.servo),
            "feedback": feedback,
            "summary": summarize_control(control_samples),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.path.parent, prefix=".teleop-timing-") as directory:
            temporary = write_json(payload, Path(directory) / "timing.json")
            return publish(temporary, self.path)


def summarize_arm_tracking(trace: dict) -> dict:
    """用反馈查询开始前最近完成的成功伺服参考，估计 URDF EEF 跟踪误差。"""
    servo = trace["servo"]["samples"]
    servo_finished_ns = [sample["sdk_finished_ns"] for sample in servo]
    feedback = [sample for sample in trace["feedback"]["samples"] if sample["succeeded"]]
    servo_hz = float(trace["metadata"]["servo_hz"])
    if not np.isfinite(servo_hz) or servo_hz <= 0:
        raise ValueError("trace metadata.servo_hz must be finite and positive")
    # 限制在两个名义伺服周期内，避免用长时间静止前的参考配对。
    max_reference_age_ns = round(2e9 / servo_hz)
    reference_samples, measured_samples, age_ns = [], [], []
    for sample in feedback:
        started_ns = sample["started_ns"]
        index = bisect_right(servo_finished_ns, started_ns) - 1
        if index < 0:
            continue
        reference = servo[index]
        age = started_ns - servo_finished_ns[index]
        if not reference["succeeded"] or age > max_reference_age_ns:
            continue
        reference_samples.append(reference["qpos"])
        measured_samples.append(sample["qpos"])
        age_ns.append(age)

    summary: dict[str, int | float | dict[str, float] | None] = {
        "successful_feedback_samples": len(feedback),
        "matched_feedback_samples": len(age_ns),
        "max_reference_age_ms": max_reference_age_ns / 1e6,
        "reference_age_ms": None,
        "joint_max_abs_deg": None,
        "eef_position_mm": None,
        "eef_orientation_deg": None,
    }
    if not age_ns:
        return summary

    from scipy.spatial.transform import Rotation

    from dexmani_real.geometry.kinematics import ArmFK, eef_poses

    reference_qpos = np.asarray(reference_samples, dtype=np.float64)
    measured_qpos = np.asarray(measured_samples, dtype=np.float64)
    shape = (len(age_ns), *ARM_JOINT_SHAPE)
    for qpos in (reference_qpos, measured_qpos):
        if qpos.shape != shape or not np.isfinite(qpos).all():
            raise ValueError("matched trace samples must contain seven finite joint positions")

    fk = ArmFK()
    reference_poses = eef_poses(reference_qpos, arm_fk=fk)
    measured_poses = eef_poses(measured_qpos, arm_fk=fk)
    relative_rotation = reference_poses[:, :3, :3].transpose(0, 2, 1) @ measured_poses[:, :3, :3]
    errors = {
        "reference_age_ms": np.asarray(age_ns) / 1e6,
        "joint_max_abs_deg": np.rad2deg(np.max(np.abs(reference_qpos - measured_qpos), axis=1)),
        "eef_position_mm": 1e3 * np.linalg.norm(reference_poses[:, :3, 3] - measured_poses[:, :3, 3], axis=1),
        "eef_orientation_deg": np.rad2deg(Rotation.from_matrix(relative_rotation).magnitude()),
    }
    for name, values in errors.items():
        p50, p95 = np.percentile(values, (50, 95))
        summary[name] = {"p50": float(p50), "p95": float(p95), "max": float(values.max())}
    return summary


def _milliseconds(durations_ns) -> dict | None:
    values = np.asarray(durations_ns, dtype=np.float64) / 1e6
    if not values.size:
        return None
    p50, p95, p99 = np.percentile(values, (50, 95, 99))
    return {
        "samples": values.size, "p50": float(p50), "p95": float(p95),
        "p99": float(p99), "max": float(values.max()),
    }


def _replay_endpoint(event, feedback, *, max_age_s):
    start_ns = event["timestamp_ns"]
    end_ns = min(event["deadline_ns"], event["finished_ns"])
    sample = next((
        row for row in reversed(feedback)
        if row["started_ns"] >= start_ns and row["finished_ns"] <= end_ns
    ), None)
    result = {
        "raw_row": event["raw_row"],
        "feedback_ns": None,
        "joint_max_abs_deg": None,
        "joint_max_velocity_deg_s": None,
    }
    if sample is not None and sample["succeeded"] and sample_is_fresh(sample["started_ns"], max_age_s, end_ns):
        error = np.asarray(sample["qpos"]) - event["qpos"]
        result.update(
            feedback_ns=sample["started_ns"],
            joint_max_abs_deg=float(np.rad2deg(np.max(np.abs(error)))),
            joint_max_velocity_deg_s=float(np.rad2deg(np.max(np.abs(sample["qvel"])))),
        )
    return result


def summarize_timing(trace: dict) -> dict:
    """以邮箱发布时间连接主机提交和 xArm 伺服调用。"""
    metadata = trace["metadata"]
    control = trace["control"]
    servo, feedback = trace["servo"]["samples"], trace["feedback"]["samples"]
    by_goal: dict[int, list[dict]] = defaultdict(list)
    for sample in servo:
        by_goal[sample["goal_published_ns"]].append(sample)

    published = set()
    last_published_ns = None
    replay_rows = []
    endpoint = None
    for event in control["samples"]:
        if event["event"] == "arm_publish":
            last_published_ns = event["timestamp_ns"]
            published.add(last_published_ns)
        elif event["event"] == "replay_row":
            stamp = event["arm_published_ns"]
            # 臂先发布、手后写入。组合失败时用同一次 send_action 内的发布事件。
            if stamp is None and last_published_ns is not None and (
                event["timestamp_ns"] <= last_published_ns <= event["finished_ns"]
            ):
                stamp = last_published_ns
            calls = by_goal.get(stamp, []) if stamp is not None else []
            replay_rows.append({
                "raw_row": event["raw_row"],
                "accepted": event["accepted"],
                "arm_published_ns": stamp,
                "sdk_calls_observed": len(calls),
                "successful_sdk_calls_observed": sum(call["succeeded"] for call in calls),
            })
        elif event["event"] == "replay_endpoint":
            endpoint = _replay_endpoint(event, feedback, max_age_s=metadata["feedback_max_age_s"])

    matched = published & by_goal.keys()
    servo_dropped = trace["servo"]["dropped"]
    # 保留窗口外的调用可能包括某个目标的首次发送。
    first_retained_ns = servo[0]["sdk_started_ns"] if servo_dropped and servo else 0
    summary = summarize_arm_tracking(trace)
    summary["dispatch"] = {
        "arm_publications": len(published),
        "arm_publications_unobserved": len(published - matched),
        "servo_calls": len(servo),
        "servo_failures": sum(not sample["succeeded"] for sample in servo),
        "publish_to_first_servo_ms": _milliseconds([
            by_goal[stamp][0]["sdk_started_ns"] - stamp for stamp in matched if stamp >= first_retained_ns
        ]),
        "servo_duration_ms": _milliseconds([row["sdk_finished_ns"] - row["sdk_started_ns"] for row in servo]),
        "servo_lateness_ms": _milliseconds([row["sdk_started_ns"] - row["scheduled_ns"] for row in servo]),
        "feedback_duration_ms": _milliseconds([row["finished_ns"] - row["started_ns"] for row in feedback]),
    }
    summary["dropped"] = {
        "control": control["dropped"],
        "servo": servo_dropped,
        "feedback": trace["feedback"]["dropped"],
    }
    if metadata.get("mode") == "replay":
        summary["replay"] = {"rows": replay_rows, "endpoint": endpoint}
    return summary


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="离线汇总 xArm 调度时序与实测跟踪误差")
    parser.add_argument("trace", type=Path, help="--timing-output 生成的 JSON 文件")
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as source:
        trace = json.load(source)
    print(json.dumps(summarize_timing(trace), indent=2))


if __name__ == "__main__":
    main()
