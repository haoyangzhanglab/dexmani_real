"""CPU/Fake 性能画像契约；不启动传感器、真实模型或机器人。"""

from concurrent.futures import Future
from dataclasses import replace
import json
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.data import recorder as recorder_module
from dexmani_real.data.recorder import EpisodeRecorder
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.robot.commands import RobotError
from dexmani_real.runtime.observation import Observation, read_observation
from dexmani_real.teleop import vr as vr_module
from dexmani_real.utils import profile as profile_module
from dexmani_real.utils.profile import Profiler, call, profile_session
from test_rollout_hoi_recovery import env as rollout_fixture
from test_teleop_session import teleop as teleop_fixture


def profiler(kind="policy", runtime=None, **kwargs):
    result = Profiler(kind, **kwargs)
    result.configure(
        runtime or RealConfig(),
        control_hz=20,
        mode="sync" if kind == "policy" else None,
    )
    return result


def test_disabled_does_not_allocate_or_read_clocks(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled profiling must not allocate or time")

    monkeypatch.setattr(profile_module, "Profiler", forbidden)
    monkeypatch.setattr(profile_module.time, "perf_counter_ns", forbidden)
    monkeypatch.setattr(profile_module.time, "monotonic_ns", forbidden)
    value = object()
    with profile_session(None, "teleop", RealConfig(), control_hz=16) as profile:
        assert profile is None
        assert call(profile, "unused", lambda: value) is value
    assert list(tmp_path.iterdir()) == []


def test_statistics_are_bounded_and_standard_json(tmp_path):
    profile = profiler(max_samples=100)
    for value in range(1, 201):
        profile.sample("stage_ms", value)
    profile.sample("empty_ms", None)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        profile.sample("invalid_ms", invalid)
    assert len(profile.samples["stage_ms"]) == 100
    path = tmp_path / "profile.json"
    profile.write(path)
    with path.open() as stream:
        report = json.load(stream, parse_constant=lambda value: pytest.fail(value))
    stat = report["metrics"]["stage_ms"]
    assert (stat["count"], stat["sample_count"], stat["dropped_count"]) == (
        200,
        100,
        100,
    )
    assert stat["p50"] == 150.5 and stat["p95"] == pytest.approx(195.05)
    assert stat["p99"] == pytest.approx(199.01)
    assert stat["mean"] == 100.5 and stat["max"] == 200
    assert not stat["p99_under_sampled"]
    assert report["metrics"]["empty_ms"]["p99"] is None
    assert report["metrics"]["empty_ms"]["p99_under_sampled"]
    assert report["counters"]["nonfinite_samples_rejected"] == 3
    with pytest.raises(FileExistsError):
        profile.write(path)


def test_threaded_samples_and_counters_remain_bounded():
    profile = profiler(max_samples=23)

    def work():
        for _ in range(1000):
            profile.sample("shared_ms", 1)
            profile.count("writes")

    threads = [Thread(target=work) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = profile.report()
    assert report["metrics"]["shared_ms"]["count"] == report["counters"]["writes"] == 3000
    assert report["metrics"]["shared_ms"]["sample_count"] == 23


def test_control_metrics_reuse_schedule_and_reset_between_episodes(monkeypatch):
    clock = SimpleNamespace(now=1_000_000_000)
    monkeypatch.setattr(profile_module.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(profile_module.time, "perf_counter_ns", lambda: clock.now)
    profile = profiler()
    profile.start_episode()
    profile.control_slot(clock.now, clock.now, 50_000_000, 0)
    with profile.tick(1_050_000_000):
        clock.now += 5_000_000
    clock.now = 1_180_000_000
    profile.control_slot(clock.now, 1_050_000_000, 50_000_000, 2)
    with profile.tick(1_200_000_000):
        clock.now += 25_000_000
    clock.now += 90_000_000_000  # 下一段准备时间不进入周期分布。
    profile.start_episode()
    profile.control_slot(clock.now, clock.now, 50_000_000, 0)
    result = profile.report()
    assert result["metrics"]["control_period_ms"]["count"] == 1
    assert result["metrics"]["control_period_ms"]["max"] == 180
    assert result["metrics"]["control_lateness_ms"]["max"] == 130
    assert result["counters"]["skipped_slots"] == 2
    assert result["counters"]["deadline_misses"] == 1


@pytest.mark.parametrize("field,source,age", [("object_pointcloud", "hoi", 100), ("point_cloud", "pointcloud", 90)])
def test_freshness_uses_source_stamps_without_extra_ring_reads(monkeypatch, field, source, age):
    env = rollout_fixture.__wrapped__(
        monkeypatch, True, (field, "joint_state", "rgb")
    )
    profile = profiler(runtime=env.rollout.runtime)
    needs = env.rollout.read_needs
    now = env.clock.now
    env.robot.arm_age = 30_000_000
    env.robot.hand_age = 40_000_000
    read_state = env.robot.read_state

    def distinct_tactile_stamps():
        state = read_state()
        state.hand["tactile_aggregate_timestamp_ns"] = now - 10_000_000
        state.hand["tactile_dense_timestamp_ns"] = now - 20_000_000
        return state

    monkeypatch.setattr(env.robot, "read_state", distinct_tactile_stamps)
    getattr(env.channels, f"{source}_ring").data["timestamp_ns"] = now - age * 1_000_000
    rings = [
        env.channels.hoi_ring,
        env.channels.hoi_request,
        env.channels.pointcloud_ring,
        env.channels.camera_ring,
    ]
    plain = read_observation(env.channels, env.robot, needs=needs)
    reads = [list(r.reads) for r in rings]
    for ring in rings:
        ring.reads.clear()
    row = read_observation(env.channels, env.robot, needs=needs, profiler=profile)
    assert [r.reads for r in rings] == reads
    if source == "hoi":
        assert row.hoi["object_pointcloud"] is plain.hoi["object_pointcloud"]
    else:
        assert row.point_cloud is plain.point_cloud
    result = profile.report()["metrics"]
    for name, expected in (
        ("arm", 30),
        ("hand", 40),
        ("camera", age),
        (source, age),
        ("tactile_aggregate", 10),
        ("tactile_dense", 20),
    ):
        assert result[f"sensor_age_ms.{name}"]["mean"] == expected
    env.rollout.profiler = profile
    _, raw = env.rollout.read_observations()
    assert profile.counts["raw_camera_read_ms"] == 1
    assert (
        profile.samples["sensor_age_ms.raw_camera"][-1]
        == (raw.observation_timestamp_ns - int(raw.camera["timestamp_ns"])) / 1e6
    )
    threshold = round(profile.thresholds["hoi"] * 1e9)
    profile.sensor("hoi", now - threshold, now)
    profile.sensor("camera", 0, now)
    profile.sensor("hand", now + 1, now)
    assert profile.counters["hoi_stale"] == 1
    assert profile.counters["camera_unavailable"] == profile.counters["hand_future_timestamp"] == 1


@pytest.mark.parametrize("fail", [False, True])
def test_future_build_predict_request_and_poll_are_distinct(monkeypatch, fail):
    clock = SimpleNamespace(now=1_000_000_000)
    monkeypatch.setattr(profile_module.time, "perf_counter_ns", lambda: clock.now)
    profile = profiler()
    started, completed = Event(), Event()
    order = []
    failure = ValueError("model failed")

    def build(rows):
        assert started.wait(2)
        order.append("build")
        clock.now += 20_000_000
        return rows

    def reset():
        order.append("reset")
        clock.now += 3_000_000

    def predict(rows):
        order.append("predict")
        clock.now += 50_000_000
        if fail:
            raise failure
        return rows

    worker = PolicyWorker(lambda: None, build_observation=build, profiler=profile)
    worker.model = SimpleNamespace(reset_episode=reset, predict=predict, close=lambda: None)
    rows = object()
    try:
        worker.predict(rows, reset_episode=True)
        worker.future.add_done_callback(lambda future: completed.set())
        started.set()
        assert completed.wait(2)
        clock.now += 7_000_000
        future = worker.poll()
        if fail:
            with pytest.raises(ValueError) as caught:
                future.result()
            assert caught.value is failure
        else:
            assert future.result() is rows
        assert worker.poll() is None
    finally:
        started.set()
        worker.close().result(timeout=2)
    assert profile.counters["model_predict_ms_failures"] == int(fail)
    assert order == ["build", "reset", "predict"]
    metrics = profile.report()["metrics"]
    for key, value in (
        ("model_input_build_ms", 20),
        ("model_predict_ms", 50),
        ("prediction_future_latency_ms", 73),
        ("prediction_poll_delay_ms", 7),
    ):
        assert metrics[key]["count"] == 1 and metrics[key]["mean"] == value


def test_future_done_before_callback_does_not_fabricate_poll_delay():
    profile = profiler()
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows, profiler=profile)
    worker.future = Future()
    worker._profile_request = dict(stage="prediction", completed=None)
    worker.future.set_result(42)
    try:
        assert worker.poll().result() == 42
        assert "prediction_poll_delay_ms" not in profile.samples
        assert profile.counters["prediction_completion_marker_races"] == 1
    finally:
        worker.close().result(timeout=2)


@pytest.mark.parametrize("kind", ["teleop", "policy"])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_opt_in_and_exception_export(monkeypatch, tmp_path, kind, fail):
    from scripts import collect_teleop, run_policy
    from dexmani_real.teleop import run
    from dexmani_real.deployment import session

    failure = RuntimeError("private-path password camera-serial 192.0.2.7")
    seen = []

    def fake_run(profile):
        seen.append(profile)
        if profile is not None:
            profile.sample("fake_ms", 12)
        if fail:
            raise failure
        return 7

    monkeypatch.setattr(run, "run_teleop", lambda *a, **kw: fake_run(kw["profiler"]))
    monkeypatch.setattr(
        session,
        "PolicySession",
        lambda *a, **kw: SimpleNamespace(run=lambda: fake_run(kw["profiler"])),
    )
    cli = collect_teleop if kind == "teleop" else run_policy
    args = [] if kind == "teleop" else ["fake.pt"]
    for enabled in (False, True):
        path = tmp_path / "nested" / f"{kind}.json"
        arguments = args + (["--profile-output", str(path)] if enabled else [])
        if fail:
            with pytest.raises(RuntimeError) as caught:
                cli.main(arguments)
            assert caught.value is failure
        else:
            assert cli.main(arguments) == 7
        assert (seen[-1] is not None) == enabled
        assert path.exists() == enabled
    data = json.loads(path.read_text())
    assert data["kind"] == kind and data["metrics"]["fake_ms"]["count"] == 1
    assert data["session"]["exit_code"] == (None if fail else 7)
    assert data["counters"].get("session_exceptions", 0) == int(fail)
    for secret in ("private-path", "password", "camera-serial", "192.0.2.7", "fake.pt"):
        assert secret not in path.read_text()
    calls = len(seen)
    with pytest.raises(FileExistsError):
        cli.main(args + ["--profile-output", str(path)])
    assert len(seen) == calls


def test_export_failure_preserves_experiment_error_and_existing_file(tmp_path, caplog):
    path = tmp_path / "profile.json"
    failure = RuntimeError("experiment failed")
    with pytest.raises(RuntimeError) as caught:
        with profile_session(path, "teleop", RealConfig(), control_hz=16) as profile:
            profile.sample("work_ms", 1)
            path.write_text("created by another writer")
            raise failure
    assert caught.value is failure
    assert path.read_text() == "created by another writer"
    assert "FileExistsError" in caplog.text


@pytest.mark.parametrize("mode", ["sync", "async", "rtc"])
@pytest.mark.parametrize("failure", [False, True])
def test_policy_trace_is_identical_with_profiling(monkeypatch, mode, failure):
    perf_counter_ns = profile_module.time.perf_counter_ns

    def forbidden_clock():
        pytest.fail("disabled control must not read a profiling clock")

    def trace(enabled):
        monkeypatch.setattr(
            profile_module.time,
            "perf_counter_ns",
            perf_counter_ns if enabled else forbidden_clock,
        )
        env = rollout_fixture.__wrapped__(monkeypatch, True, ("object_pointcloud", "joint_state"))
        rollout = env.rollout
        rollout.execution = replace(rollout.execution, mode=mode)
        rollout.profiler = profiler(runtime=rollout.runtime) if enabled else None
        rollout.max_duration_ns = 900_000_000

        # 只禁用真实控制主体的写盘；初始化和 CLI 收尾在这个测试之外。
        def no_profile_write(*args, **kwargs):
            pytest.fail("no profiling file writes in control")

        monkeypatch.setattr(profile_module.json, "dump", no_profile_write)
        for index in range(24):
            if index == 8 and failure:
                env.robot.send_error = RobotError("submission failed")
            try:
                rollout.step()
            except RobotError as error:
                rollout.fail("control", error)
            if rollout.finished:
                break
            if env.worker.future is not None and not env.worker.future.done():
                env.worker.complete()
            env.advance(0.16 if index == 2 else 0.05, valid=index not in (4, 5))
        if enabled:
            assert rollout.profiler.counts["control_tick_ms"] > 0
            assert rollout.profiler.counts["observation_read_ms"] > 0
            assert rollout.profiler.counts["action_mapping_ms"] > 0
            assert rollout.profiler.counts["robot_send_action_ms"] > 0
            assert rollout.profiler.counters["skipped_slots"] == rollout.result.skipped_slots
        return dict(
            sent=[
                (
                    t,
                    c.arm_qpos.tolist() if c.arm_qpos is not None else None,
                    c.hand_qpos.tolist(),
                    c.arm_hold,
                    expiry,
                )
                for t, c, expiry in env.robot.sent
            ],
            rows=[(r.control_timestamp_ns, c is not None) for r, c in env.recorder.rows],
            skipped=rollout.result.skipped_slots,
            reason=rollout.result.termination_reason,
            duration=rollout.result.duration_s,
            resets=env.worker.reset_count,
            gaps=env.recorder.gaps,
            reads=[
                ring.reads
                for ring in (
                    env.channels.camera_ring,
                    env.channels.hoi_ring,
                    env.channels.hoi_request,
                )
            ],
        )

    assert trace(False) == trace(True)


def test_teleop_trace_excludes_preparation_and_keeps_actions(monkeypatch):
    def trace(enabled):
        clock = SimpleNamespace(now=10_000_000_000)
        monkeypatch.setattr(profile_module.time, "monotonic_ns", lambda: clock.now)
        loop = teleop_fixture.__wrapped__()
        loop.profiler = profiler("teleop", runtime=loop.runtime) if enabled else None
        loop.robot.send_action.side_effect = None

        def prepare(*args, **kwargs):
            clock.now += 90_000_000_000
            loop.runtime_state.stop_signal.clear()

        loop.robot.prepare_streaming.side_effect = prepare
        assert call(
            loop.profiler,
            "episode_prepare_ms",
            loop.start_episode,
            loop.runtime_state.cancel_epoch,
        )
        for index in range(5):
            loop.run_control_tick()
            clock.now += loop.control_dt_ns * (6 if index == 3 else 1)
        if enabled:
            assert loop.profiler.counts["episode_prepare_ms"] == 1
            assert max(loop.profiler.samples["control_period_ms"]) < 1000
            assert loop.profiler.counts["control_tick_ms"] == 5
            assert loop.profiler.counts["check_observation_ms"] > 0
            assert loop.profiler.counters["vr_new_samples"] > 0
        return (
            [
                (c.args[0].arm_qpos.tolist(), c.kwargs["expires_ns"])
                for c in loop.robot.send_action.call_args_list
            ],
            loop.next_tick_ns,
            loop.pending_termination_reason,
        )

    assert trace(False) == trace(True)


def test_vr_retarget_and_mapping_have_separate_boundaries(monkeypatch):
    profile = profiler("teleop")
    order = []
    hand = np.zeros(12)
    retargeter = SimpleNamespace(retarget=lambda *a, **kw: order.append("hand") or hand)
    mapper = SimpleNamespace(solver=object(), map_eef=lambda *a: order.append("map") or a)
    controller = vr_module.VRController(mapper, retargeter, RealConfig().teleop, np.eye(3), profiler=profile)
    monkeypatch.setattr(vr_module, "prepare_reference", lambda *a, **kw: "reference")
    controller._wrist_target = lambda *args: "pose"
    row = SimpleNamespace(
        arm={"qpos": [np.zeros(7)]},
        vr=dict(wrist_pos=None, wrist_quat_wxyz=None, landmarks=None),
    )
    result = controller.compute_target(row, control_ns=123)
    assert order == ["hand", "map"] and result[1] is hand
    assert profile.counts["hand_retarget_ms"] == profile.counts["action_mapping_ms"] == 1


@pytest.mark.parametrize("finish_fails", [False, True])
def test_recorder_preserves_rows_finish_and_gap_with_fake_writer(monkeypatch, tmp_path, finish_fails):
    traces = []

    class Writer:
        def __init__(self, *args, **kwargs):
            self.frame_count = 0
            self.buffered = 0
            self.trace = []
            traces.append(self.trace)

        def open(self, metadata):
            self.trace.append("open")

        def append_frame(self, row, command, gap, action_accepted):
            self.trace.append((row.control_timestamp_ns, gap))
            self.buffered += 1

        def finish(self, reason, **kwargs):
            self.trace.append(("finish", reason, kwargs))
            if finish_fails:
                raise OSError("flush failed")
            self.frame_count = self.buffered

        def close(self):
            self.trace.append("close")

    monkeypatch.setattr(recorder_module, "RawWriter", Writer)
    row = Observation(np.zeros(1), None, None, None, None, 100, control_timestamp_ns=90)
    for enabled in (False, True):
        profile = profiler() if enabled else None
        recorder = EpisodeRecorder(tmp_path / str(enabled), control_hz=20, profiler=profile)
        recorder.start(task_label="fake", collection_source="test", metadata={})
        for index in range(3):
            if index == 1:
                recorder.mark_gap()
            assert call(profile, "recorder_append_ms", recorder.append, row, action_accepted=False) is None
        assert call(
            profile,
            "recorder_finish_call_ms",
            recorder.finish,
            save=True,
            reason="operator",
        )
        assert recorder.written_frames == (0 if finish_fails else 3)
        assert (recorder.published_path is None) == finish_fails
        assert not recorder.writer_pending
        assert (recorder.error is not None) == finish_fails
        if enabled:
            assert profile.counters["recorder_enqueued_frames"] == 3
            assert profile.counters["recorder_written_frames"] == recorder.written_frames
            assert profile.counters["recording_gaps"] == 1
            assert (
                profile.counts["recorder_writer_total_ms"]
                == profile.counts["recorder_finish_elapsed_ms"]
                == 1
            )
            assert (
                profile.counters["recorder_writers_started"]
                == profile.counters["recorder_writers_completed"]
                == 1
            )
    assert traces[0] == traces[1]


def test_recorder_queue_full_is_counted_without_capacity_changes(tmp_path):
    profile = profiler()
    recorder = EpisodeRecorder(tmp_path, control_hz=20, profiler=profile)
    recorder.is_recording = True
    row = Observation(np.zeros(1), None, None, None, None, 100)
    assert recorder.queue.maxsize == 16
    for _ in range(16):
        assert recorder.append(row, action_accepted=False) is None
    assert recorder.append(row, action_accepted=False) == "recording_queue_full"
    assert recorder.recording_truncated_reason == "recording_queue_full"
    assert profile.counters["recorder_queue_full"] == profile.counters["recording_truncations"] == 1
    assert (
        profile.counters["recorder_enqueued_frames"]
        == profile.counters["recorder_queue_high_water_observed"]
        == 16
    )


@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
def test_write_failure_does_not_replace_primary_error(monkeypatch, tmp_path, error):
    def fail_write(*args):
        raise error

    monkeypatch.setattr(Profiler, "write", fail_write)
    original = ValueError("original experiment failure")
    with pytest.raises(ValueError) as caught:
        with profile_session(tmp_path / "result.json", "policy", RealConfig(), control_hz=None):
            raise original
    assert caught.value is original
