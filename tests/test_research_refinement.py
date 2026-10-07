"""Offline geometry, timing and artifact-lifecycle regressions; no devices."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.robot.action import ActionIntent, ActionRealizer
from dexmani_real.robot.commands import RobotCommand


def test_joint_and_frozen_admission_use_actual_command():
    cfg = ExperimentConfig()
    realizer = ActionRealizer.for_mode(cfg, "joint")
    q, hand = np.array(cfg.arm.home_qpos), np.deg2rad(cfg.hand.home_qpos_deg)
    target = q.copy()
    target[0] += np.pi / 2
    rejected = realizer.realize(ActionIntent("joint", target, hand), q, q)
    assert (
        rejected.arm_qpos is None
        and rejected.ik_result is None
        and rejected.rejection_reason == "jump"
    )
    accepted = realizer.realize(ActionIntent("joint", q, hand), q, q)
    np.testing.assert_allclose(accepted.arm_qpos, q, atol=1e-15)
    seen = []
    realizer.collision_model = NS(
        set_hand_qpos=lambda h: seen.append(h.copy()), check_self_collision=lambda q: False
    )
    for h in (hand, hand + 0.1, hand):
        assert realizer.frozen_is_valid(RobotCommand(1, q, h), q, q, "joint")
        np.testing.assert_array_equal(seen[-1], h)


def test_default_home_is_admitted_with_real_pose_compensation():
    cfg = ExperimentConfig()
    q, hand = np.array(cfg.arm.home_qpos), np.deg2rad(cfg.hand.home_qpos_deg)
    r = ActionRealizer.for_mode(cfg, "joint")
    result = r.realize(ActionIntent("joint", q, hand), q, q)
    assert result.rejection_reason is None
    np.testing.assert_array_equal(result.arm_qpos, q)


def test_frozen_eef_preserves_configured_solver_admission():
    from dexmani_real.planning.kinematics.ik import make_online_ik_config
    from dexmani_real.planning.planner import XArm7MotionPlanner

    cfg = ExperimentConfig()
    planner = XArm7MotionPlanner.create_default(
        online_ik_profile=replace(make_online_ik_config(cfg), max_ik_jump_deg=(1.0,) * 7),
    )
    realizer = ActionRealizer(cfg, planner)
    q, hand = np.array(cfg.arm.home_qpos), np.deg2rad(cfg.hand.home_qpos_deg)
    command = RobotCommand(1, q, hand)
    assert realizer.frozen_is_valid(command, q, q, "eef")
    previous = q.copy()
    previous[0] += np.deg2rad(2)
    assert not realizer.frozen_is_valid(command, q, previous, "eef")
    assert realizer.rejection_reason == "jump"


@pytest.mark.parametrize(
    "poses",
    [
        np.zeros((10, 3)),
        np.array([[0, 0, i * 0.1] for i in range(10)]),
        np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.002, 0]]),
    ],
)
def test_degenerate_handeye_rejected(poses):
    from dexmani_real.calibration.camera.solver import calibrate_and_select

    with pytest.raises(ValueError, match="excitation"):
        calibrate_and_select(
            np.zeros_like(poses),
            poses,
            np.zeros_like(poses),
            np.zeros_like(poses),
            max_position_rms_mm=5,
            max_rotation_rms_deg=3,
        )


def test_two_axis_handeye_recovers_transform_and_is_coordinate_invariant():
    import cv2

    from dexmani_real.calibration.camera.solver import (
        calibrate_and_select,
        check_hand_eye_excitation,
    )

    poses = np.array([[0, 0, 0], [0.4, 0, 0], [0, 0.5, 0], [-0.3, 0.2, 0], [0.2, -0.4, 0]])
    rotations = Rotation.from_euler("xyz", poses).as_matrix()
    camera = np.eye(4)
    camera[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    camera[:3, 3] = [0.4, -0.2, 0.6]
    marker = np.eye(4)
    marker[:3, 3] = [0.05, 0.02, 0.01]
    positions = np.array([[0.3 + i * 0.02, 0.1 - i * 0.01, 0.4] for i in range(len(poses))])
    rvecs, tvecs = [], []
    for r, t in zip(rotations, positions):
        base = np.eye(4)
        base[:3, :3] = r
        base[:3, 3] = t
        observed = np.linalg.inv(camera) @ base @ marker
        rvecs.append(cv2.Rodrigues(observed[:3, :3])[0].ravel())
        tvecs.append(observed[:3, 3])
    result = calibrate_and_select(
        list(positions),
        list(poses),
        rvecs,
        tvecs,
        max_position_rms_mm=0.01,
        max_rotation_rms_deg=0.01,
    )
    np.testing.assert_allclose(result[0], camera, atol=1e-8)
    summary = check_hand_eye_excitation(poses)
    transformed = Rotation.from_matrix(
        Rotation.from_euler("xyz", [0.6, 0.2, -0.4]).as_matrix() @ rotations
    ).as_euler("xyz")
    other = check_hand_eye_excitation(transformed)
    assert other["valid_relative_motion_count"] == summary["valid_relative_motion_count"]
    assert other["axis_separation_witness_deg"] == pytest.approx(
        summary["axis_separation_witness_deg"]
    )


@pytest.mark.parametrize(
    "solution,valid",
    [
        ((False, None, None), False),
        ((True, np.full(3, np.nan), np.zeros(3)), False),
        ((True, np.ones((3, 1)), np.zeros((1, 3))), True),
    ],
)
def test_pnp_failure_is_missing(monkeypatch, solution, valid):
    from dexmani_real.calibration.camera import solver

    monkeypatch.setattr(
        solver.cv2.aruco,
        "ArucoDetector",
        lambda *a: NS(detectMarkers=lambda image: ([np.zeros((4, 2))], np.array([[1]]), None)),
    )
    monkeypatch.setattr(solver.cv2, "solvePnP", lambda *a, **k: solution)
    result = solver.detect_aruco_pose(
        np.zeros((16, 16, 3), np.uint8), np.eye(3), np.zeros(5), marker_size_m=0.1, target_id=1
    )
    assert (result is not None) == valid
    if valid:
        np.testing.assert_array_equal(result[0], np.ones(3))


def test_export_identity_full_modal_and_failure_staging(tmp_path, monkeypatch):
    import zarr
    from test_policy_recording import start_recording

    import dexmani_real.dataset.export as exporter
    from dexmani_real.dataset.contracts import ProcessingConfig
    from dexmani_real.dataset.export import export_raw_to_zarr
    from dexmani_real.recording.storage.reader import RawDataError

    recorder = start_recording(tmp_path / "raw")
    saved = recorder.save_episode()
    recorder.close()
    processing = ProcessingConfig.from_runtime(ExperimentConfig(), table_plane_abcd=(0, 0, 1, 10))
    reports = []
    for name in ("a.zarr", "b.zarr"):
        reports.append(export_raw_to_zarr(saved, tmp_path / name, processing=processing))
        root = zarr.open_group(str(tmp_path / name), "r")
        assert root.attrs["data_revision"] == reports[-1]["data_revision"]
        assert root.attrs["episode_ids"] == [saved.name]
        assert len(root["data"]) == 13 and np.isnan(root["data/tactile_force"][:]).all()
        assert root["meta/episode_ends"][:].tolist() == [1]
        np.testing.assert_array_equal(root["row_info/dispatch_status"][:], [[1, 1]])
        assert (
            json.loads((tmp_path / name / "export_report.json").read_text())["data_revision"]
            == reports[-1]["data_revision"]
        )
    assert reports[0]["data_revision"] != reports[1]["data_revision"]
    from dexmani_policy.datasets.base_dataset import BaseDataset
    from dexmani_policy.training.build_utils import capture_data_identity
    from dexmani_policy.training.resume import validate_data_identity

    identities = []
    for name, report in zip(("a.zarr", "b.zarr"), reports):
        dataset = BaseDataset(str(tmp_path / name), horizon=1)
        identities.append(capture_data_identity(dataset))
        assert identities[-1] == {"revision": report["data_revision"]}
        assert len(dataset) == 1
    with pytest.raises(ValueError, match="changed or lost"):
        validate_data_identity(*identities)

    with pytest.raises(FileExistsError):
        export_raw_to_zarr(saved, tmp_path / "a.zarr", processing=processing)
    (tmp_path / "dangling.zarr").symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        export_raw_to_zarr(saved, tmp_path / "dangling.zarr", processing=processing)
    native = exporter.iter_canonical_blocks

    def fail_after_append(*a, **kw):
        yield from native(*a, **kw)
        raise RawDataError("video has extra frames")

    monkeypatch.setattr(exporter, "iter_canonical_blocks", fail_after_append)
    with pytest.raises(ValueError, match="extra frames"):
        export_raw_to_zarr(saved, tmp_path / "failed.zarr", processing=processing)
    assert not (tmp_path / "failed.zarr").exists()
    assert len(list(tmp_path.glob(".failed.zarr.tmp-*"))) == 1
    with h5py.File(saved / "data.h5") as raw:
        assert raw["meta"].attrs["num_frames"] == 1


def test_quality_chunk_boundaries_and_clock_domains():
    from dexmani_real.dataset.quality import EpisodeQuality
    from dexmani_real.recording.storage.schema import ROW_INFO_SPECS

    def block(stamps, numbers):
        n = len(stamps)
        rows = {k: np.zeros((n, *v.tail_shape), v.dtype) for k, v in ROW_INFO_SPECS.items()}
        rows["observation_timestamp_ns"] = np.array(stamps)
        rows["arm_read_timestamp_ns"] = np.array(stamps) + 10
        rows["hand_read_timestamp_ns"] = np.array(stamps) - 20
        rows["camera_timestamp_ns"] = np.ones(n, dtype=np.int64)
        rows["color_frame_number"] = np.array(numbers)
        return rows

    quality = EpisodeQuality(0.1, "device_time")
    quality.update(block([1_000_000_000, 1_100_000_000], [1, 2]))
    quality.update(block([1_400_000_000, 1_300_000_000, 0], [2, 1, 0]))
    report = quality.report({}, 5, ["contact_force"])
    assert report["observation_dt_s"]["count"] == 3
    assert (
        report["nonpositive_intervals"] == 1
        and report["gaps"] == 1
        and report["unknown_timestamps"] == 1
    )
    assert report["frame_numbers"]["color_frame_number"] == dict(
        unknown=1, repeated=1, backwards=1, missing=0
    )
    assert report["negative_source_ages"]["arm"] == 4
    assert (
        report["source_age_s"]["camera"]["count"] == 0 and report["rgb_depth_exposure_skew"] is None
    )
    second = EpisodeQuality(0.1, "unknown")
    second.update(block([2_000_000_000], [1]))
    assert second.report({}, 1, [])["observation_dt_s"]["count"] == 0


def test_results_failure_preserves_incomplete_and_blocks_start(tmp_path, monkeypatch):
    import dexmani_real.recording.results as records

    result = records.SessionResults(tmp_path, "policy")
    attempt = result.prepare(recording=True)
    path = tmp_path / "attempts" / f"{attempt}.json"
    assert json.loads(path.read_text())["entered_running"] is None
    result.entered(3)
    dump = records.atomic_json_dump

    def fail(obj, path, **kw):
        if Path(path).parent.name == "attempts":
            raise OSError("disk full")
        return dump(obj, path, **kw)

    monkeypatch.setattr(records, "atomic_json_dump", fail)
    with pytest.raises(OSError):
        result.finish_attempt("timeout", recording_status="empty")
    assert json.loads(path.read_text())["state"] == "incomplete"
    with pytest.raises(RuntimeError):
        result.prepare(recording=True)


def test_intrinsics_sequence_syntax_and_reader_close(tmp_path):
    from test_policy_recording import start_recording

    from dexmani_real.recording.storage.reader import EpisodeReader
    from dexmani_real.sensor.camera.geometry import CameraIntrinsics

    values = dict(
        width=16,
        height=16,
        fx=10,
        fy=10,
        ppx=8,
        ppy=8,
        distortion_model="none",
        distortion_coeffs="00000",
    )
    with pytest.raises(TypeError):
        CameraIntrinsics.from_dict(values)
    recorder = start_recording(tmp_path)
    saved = recorder.save_episode()
    recorder.close()
    reader = EpisodeReader(saved)
    assert reader.path == saved and reader["arm_qpos"].shape == (1, 7)
    reader.close()
    with pytest.raises(RuntimeError):
        reader["arm_qpos"]


def test_timing_missing_stages_are_unknown():
    from dexmani_real.deployment.timing import summarize_trace

    result = summarize_trace(
        [
            dict(event="query_complete", started_ns=10, completed_ns=30),
            dict(event="invalidate", reason="late"),
        ]
    )
    assert result["seconds"]["model"]["p50"] == pytest.approx(20e-9)
    assert result["seconds"]["dispatch"]["p50"] is None and result["reasons"] == {"late": 1}


@pytest.mark.parametrize(
    "mode,durations,worker_error,age",
    [
        ("sync", (2.0,), None, 10.0),
        ("async", (0.2,), None, 10.0),
        ("rtc", (0.2,), None, 10.0),
        ("sync", (), RuntimeError("load failed"), 10.0),
        ("sync", (0.3,), None, 0.5),
        ("sync", (0.31,), None, 0.5),
    ],
)
def test_initial_warmup_failure_closes_owned_worker_before_connect(
    tmp_path, monkeypatch, mode, durations, worker_error, age
):
    from test_review_remediation import shared_state

    from dexmani_real.deployment import session
    from dexmani_real.deployment.config import ExecutionConfig, RolloutRecordingConfig
    from dexmani_real.deployment.inference import ModelResult

    shared = shared_state()
    events = []
    info = NS(
        n_obs_steps=2,
        n_action_steps=3,
        horizon=8,
        control_dt_s=0.1,
        action_mode="joint",
        observation_fields=("joint_state",),
    )

    class Worker:
        close_error = None
        closed = False

        def __init__(self, factory):
            pass

        def submit(self, *a, **kw):
            events.append("warmup")

        def poll(self):
            return "load", ModelResult(durations, worker_error, 0, 1)

        def close(self):
            self.closed = True
            events.append("close")

    monkeypatch.setattr("dexmani_real.deployment.inference.InferenceWorker", Worker)
    monkeypatch.setattr(session, "validate_policy_runtime_compatibility", lambda *a: None)
    monkeypatch.setattr(session.RuntimeChannels, "create", lambda **kw: shared)
    monkeypatch.setattr(session, "RuntimeSupervisor", lambda *a: NS(check=lambda: True))
    monkeypatch.setattr(
        session,
        "DexManiRobot",
        lambda *a, **kw: NS(connect=lambda: pytest.fail("connected before warmup passed")),
    )

    def shutdown(robot, supervisor, *, model, **kw):
        model.close()
        return True

    monkeypatch.setattr(session, "shutdown_local_runtime", shutdown)
    result = session.run_policy_deployment(
        ExperimentConfig(),
        NS(info=info),
        True,
        recording_config=RolloutRecordingConfig(str(tmp_path), "synthetic"),
        execution_config=ExecutionConfig(mode, age, 1.0, 0.03, 2, 2.0),
    )
    assert result == 1 and events == ["warmup", "close"]
    saved = json.loads((tmp_path / "session_result.json").read_text())
    assert saved["state"] == "finished" and saved["outcome"] == "fault"


def test_replay_preflight_before_results_and_construction_failure_finalized(tmp_path, monkeypatch):
    from dexmani_real.replay import session

    output = tmp_path / "output"
    output.mkdir()
    events = []
    monkeypatch.setattr(session, "verify_replay_preflight", lambda *a: events.append("preflight"))

    def fail(**kw):
        assert (output / "session_result.json").exists()
        raise RuntimeError("construction failed")

    monkeypatch.setattr(session.RuntimeChannels, "create", fail)
    monkeypatch.setattr(session, "evaluate_replay", lambda *a, **kw: events.append("evaluation"))
    result = session.replay_episode(
        NS(), ExperimentConfig(), session.EpisodeReplayConfig(str(output), False)
    )
    assert result.status == session.ReplayStatus.FAULT and events == ["preflight", "evaluation"]
    saved = json.loads((output / "session_result.json").read_text())
    assert saved["state"] == "finished" and "construction failed" in saved["reason"]
    with pytest.raises(ValueError, match="missing or empty"):
        session.replay_episode(
            NS(), ExperimentConfig(), session.EpisodeReplayConfig(str(output), False)
        )
    assert events == ["preflight", "evaluation"]


@pytest.mark.parametrize("fault", ["return_home", "shutdown", "evaluation"])
def test_replay_completed_trajectory_survives_session_fault(tmp_path, monkeypatch, fault):
    from test_review_remediation import shared_state

    from dexmani_real.replay import session
    from dexmani_real.runtime.operator_input import OperatorCommand

    shared = shared_state()
    supervisor = NS(check=lambda: True)
    robot = NS(connect=lambda: None, service_idle=lambda: None, check_services=None)
    keyboard = NS(start=lambda: None, healthy=True, poll=lambda **kw: [OperatorCommand.HOME])
    monkeypatch.setattr(session, "verify_replay_preflight", lambda *a: None)
    monkeypatch.setattr(session.RuntimeChannels, "create", lambda **kw: shared)
    monkeypatch.setattr(session, "RuntimeSupervisor", lambda *a: supervisor)
    monkeypatch.setattr(session, "DexManiRobot", lambda *a, **kw: robot)
    monkeypatch.setattr(session, "KeyboardInput", lambda **kw: keyboard)
    monkeypatch.setattr(session, "require_transition", lambda *a: None)
    monkeypatch.setattr(session, "home_hand", lambda *a, **kw: NS(ok=True))
    data = {"arm_qpos": np.zeros((1, 7)), "termination_reason": np.asarray("completed")}

    def replay(*a, results, **kw):
        results.prepare(recording=False)
        results.entered(2)
        if fault != "return_home":
            shared.quit_requested.value = True
        return session.ReplayOutcome(session.ReplayStatus.COMPLETED, data)

    monkeypatch.setattr(session, "replay_targets", replay)
    monkeypatch.setattr(session, "build_home_planner", lambda *a: None)
    monkeypatch.setattr(session, "home_robot", lambda *a, **kw: False)
    monkeypatch.setattr(session, "shutdown_local_runtime", lambda *a, **kw: fault != "shutdown")

    def evaluate(*a, **kw):
        if fault == "evaluation":
            raise OSError("evaluation write failed")

    monkeypatch.setattr(session, "evaluate_replay", evaluate)
    result = session.replay_episode(
        NS(), ExperimentConfig(), session.EpisodeReplayConfig(str(tmp_path), False)
    )
    saved = json.loads((tmp_path / "session_result.json").read_text())
    assert saved["trajectory_status"] == "completed" and saved["outcome"] != "completed"
    assert str(result.replay_data["termination_reason"]) == "completed"


@pytest.mark.parametrize("lag", [-2, 2])
def test_tracking_lag_sign_missing_rows_and_stationary_axes(lag):
    from dexmani_real.replay.evaluation import tracking_lags

    original = np.zeros((60, 7))
    original[:, 0] = np.sin(np.arange(60) * 0.2)
    replay = np.zeros_like(original)
    replay[:, 0] = np.nan
    if lag > 0:
        replay[lag:, 0] = original[:-lag, 0]
    else:
        replay[:lag, 0] = original[-lag:, 0]
    replay[20, 0] = np.nan
    assert tracking_lags(original, replay, 30) == {0: lag}


def test_invalid_hand_is_rejected_before_any_arm_sdk(monkeypatch):
    from test_review_remediation import fake_robot

    from dexmani_real.robot.robot import DispatchError

    robot, clock, command = fake_robot(monkeypatch)
    invalid = RobotCommand(command.run_id, command.arm_qpos, np.full(12, 100.0))
    with pytest.raises(DispatchError):
        robot.send_action(invalid, valid_until_ns=10000)
    assert clock.calls == []


@pytest.mark.parametrize(
    "field,reason",
    [
        ("limit_violation", "planning limits"),
        ("start_qpos_error_rad", "start is too far"),
        ("max_waypoint_delta_rad", "waypoint delta"),
        ("terminal_pos_error_m", "Terminal pose"),
    ],
)
def test_home_report_rejections_precede_geometry(field, reason):
    from dexmani_real.planning.planner import MotionPlanningConfig, XArm7MotionPlanner

    planner = XArm7MotionPlanner.__new__(XArm7MotionPlanner)
    planner.ik_geometry = NS(
        snap_path_to_nearest_equivalent=lambda p, q: p,
        canonicalize_path_to_planning_limits=lambda p, q, c: p,
    )
    planner.shortcut_smooth_path = lambda p, q, c: p
    report = dict(
        limit_violation=False,
        start_qpos_error_rad=0.0,
        max_waypoint_delta_rad=0.0,
        terminal_pos_error_m=0.0,
        terminal_rot_error_rad=0.0,
    )
    report[field] = 10.0
    planner.compute_path_metrics = lambda *a: report
    planner.check_elbow_consistency = lambda p: (False, {})
    planner._check_workspace = lambda *a: pytest.fail("geometry after failed report")
    result = planner.validate_path(
        np.zeros((2, 7)), None, np.zeros(7), "synthetic", MotionPlanningConfig()
    )
    assert not result.success and reason in result.reason


def test_observation_fk_resources_are_constructed_only_when_needed(monkeypatch):
    from dexmani_real.deployment import observation
    from dexmani_real.planning.kinematics.hand_fk import HandKinematics
    from dexmani_real.robot.model import XHAND_RIGHT_URDF_PATH

    monkeypatch.setattr(observation, "HandKinematics", lambda *a: pytest.fail("unneeded hand FK"))
    cfg = ExperimentConfig()
    assert (
        observation.build_observation_kinematics(
            NS(observation_fields=("joint_state", "rgb", "point_cloud")), cfg
        )
        is None
    )
    eef = observation.build_observation_kinematics(NS(observation_fields=("eef_pose",)), cfg)
    assert eef.hand_fk is None and eef.arm_fk is not None
    with pytest.raises(ValueError, match="missing"):
        HandKinematics(str(XHAND_RIGHT_URDF_PATH), ["absent", "b", "c", "d", "e"])


def test_policy_home_failure_is_retained_in_memory_without_active_io(monkeypatch):
    from test_review_remediation import shared_state

    from dexmani_real.deployment import operator
    from dexmani_real.runtime.safety import SafetyState

    shared = shared_state()
    shared.safety_state.value = int(SafetyState.ARMED)
    shared.start_request = NS(value=False)
    monkeypatch.setattr(operator, "KeyboardInput", lambda **kw: NS(drain_signal=lambda *a: None))
    monkeypatch.setattr(operator, "home_robot", lambda *a, **kw: False)
    owner = operator.PolicyOperator(shared, ExperimentConfig(), NS(), robot=NS(), execute=True)
    assert owner._run_home()
    assert owner.home_results == [{"outcome": "failed"}]
