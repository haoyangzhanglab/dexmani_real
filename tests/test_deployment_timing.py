"""Offline deployment checks for run epochs and causal scheduling."""

from __future__ import annotations

import pickle
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    PolicyEndpointDisposition,
    classify_policy_endpoint_disposition,
)
from dexmani_real.control.safety_gate import GateRejectCode
from dexmani_real.deployment.config import (
    DeploymentConfig,
    H4ExecuteBounds,
    PolicyRuntimeConfig,
    TaskExecuteBounds,
    resolve_deployment_config,
)
from dexmani_real.deployment.contracts import (
    InferenceContext,
    JointActionChunk,
    PolicyPrediction,
)
from dexmani_real.deployment.coordinator import (
    CoordinatorConfig,
    _buffered_plan_from_record,
    _physical_start_pose_rejection,
    _plan_deadline_ns,
    coordinator_loop,
)
from dexmani_real.deployment.manifest import DeploymentManifest
from dexmani_real.deployment.metrics import OBSERVATION_WAIT_POINTCLOUD_GRID, Metrics
from dexmani_real.deployment.observation import PointCloudFrame, freeze_array
from dexmani_real.deployment.worker import (
    _build_observation,
    _read_state_history,
    _select_pointcloud_control_grid,
    inference_loop,
    publish_plan,
    stamp_prediction_timing,
)
from dexmani_real.integrations.dexmani_policy import _validate_training_data_contract
from dexmani_real.ipc.causal import read_structured_frame_aligned_to_source
from dexmani_real.ipc.channels import RuntimeChannelsConfig
from dexmani_real.ipc.schema import (
    ARM_STATE_DTYPE,
    HAND_STATE_DTYPE,
    MAX_POLICY_CHUNK_STEPS,
    POLICY_PLAN_DTYPE,
    make_pointcloud_frame_dtype,
)
from dexmani_real.planning.ik import TeleopIKSolver
from dexmani_real.planning.types import CollisionInfo, IKFailureKind
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    StopRequest,
    begin_motion,
    read_run_epoch,
    revoke_motion,
)
from dexmani_real.runtime.status import ExitReason
from dexmani_real.runtime.supervisor import run_supervisor
from dexmani_real.utils.feedback import (
    FeedbackIssue,
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_hand_feedback,
)


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value


class _Ring:
    def __init__(self, records: list[tuple[np.ndarray, int, int]]) -> None:
        self.records = records
        self.maxlen = max(8, len(records))

    def get_last_k(self, count: int):
        return self.records[-count:]


class _SequenceRing(_Ring):
    @property
    def latest_sequence(self) -> int:
        return self.records[-1][2] if self.records else 0

    def read_sequence(self, sequence: int):
        for record in self.records:
            if record[2] == sequence:
                return record
        return None


class _WriteRing:
    def __init__(self, on_write=None) -> None:
        self.frame: np.ndarray | None = None
        self.on_write = on_write

    def write(self, frame: np.ndarray) -> int:
        self.frame = frame.copy()
        if self.on_write is not None:
            self.on_write()
        return 1


class _LockedValue(_Value):
    def __init__(self, value: int) -> None:
        super().__init__(value)
        self._lock = threading.RLock()

    def get_lock(self):
        return self._lock


class _Clock:
    def __init__(self, monotonic_ns_values: tuple[int, ...], on_sleep=None) -> None:
        self._monotonic_ns_values = iter(monotonic_ns_values)
        self._last_monotonic_ns = monotonic_ns_values[-1]
        self._on_sleep = on_sleep
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return 0.0

    def monotonic_ns(self) -> int:
        return next(self._monotonic_ns_values, self._last_monotonic_ns)

    def sleep(self, duration_s: float) -> None:
        self.sleep_calls.append(duration_s)
        if self._on_sleep is not None:
            self._on_sleep()


class _SupervisorClock:
    def __init__(self, *, acknowledge_time_limit: bool = True) -> None:
        self.now_s = 0.0
        self.limit_request_times_s: list[float] = []
        self.acknowledge_time_limit = acknowledge_time_limit
        self._shared: SimpleNamespace | None = None

    def attach(self, shared: SimpleNamespace) -> None:
        self._shared = shared

    def monotonic(self) -> float:
        return self.now_s

    def monotonic_ns(self) -> int:
        return int(self.now_s * 1e9)

    def sleep(self, _duration_s: float) -> None:
        assert self._shared is not None
        shared = self._shared
        if self.now_s == 0.0:
            # B has not been pressed during the first ARMED supervisor tick.
            self.now_s = 1.0
            shared.run_started_monotonic_ns.value = int(self.now_s * 1e9)
            shared.safety_state.value = int(SafetyState.RUNNING)
            return
        if self.acknowledge_time_limit and int(shared.stop_request.value) == int(
            StopRequest.RUN_TIME_LIMIT
        ):
            self.limit_request_times_s.append(self.now_s)
            # Model the coordinator's normal stop path: consume the typed
            # request and revoke RUNNING -> ARMED before the next supervisor tick.
            shared.stop_request.value = int(StopRequest.NONE)
            shared.safety_state.value = int(SafetyState.ARMED)
        self.now_s += 1.0


def _shared_epoch() -> SimpleNamespace:
    return SimpleNamespace(
        safety_state=_Value(int(SafetyState.ARMED)),
        run_generation=_Value(4),
        run_started_monotonic_ns=_Value(0),
        active_coupled_command_sequence=_Value(9),
        motion_lock=threading.RLock(),
        is_running=_Value(1),
        error_state=_Value(0),
        estop_request=_Value(0),
    )


def _pointcloud(sequence: int, source_ns: int) -> PointCloudFrame:
    return PointCloudFrame(
        values=np.zeros((1, 6), dtype=np.float32),
        source_camera_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=source_ns + 1,
        camera_generation=1,
    )


def _plan() -> np.void:
    frame = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
    frame["plan_id"][0] = 1
    frame["run_generation"][0] = 5
    frame["observation_id"][0] = 2
    frame["observation_latest_source_monotonic_ns"][0] = 1_000_000_000
    frame["observation_logical_step_monotonic_ns"][0] = 1_050_000_000
    frame["observation_anchor_monotonic_ns"][0] = 1_060_000_000
    frame["inference_started_monotonic_ns"][0] = 1_070_000_000
    frame["inference_finished_monotonic_ns"][0] = 1_080_000_000
    frame["num_steps"][0] = 3
    frame["arm_present"][0] = 1
    frame["hand_present"][0] = 1
    frame["target_monotonic_ns"][0, :3] = (
        1_050_000_000,
        1_200_000_000,
        1_300_000_000,
    )
    frame["valid_mask"][0, :3] = 1
    return frame[0]


class DeploymentTimingTest(unittest.TestCase):
    def test_h4_bounds_reject_noncanonical_or_untyped_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly equal to 1"):
            H4ExecuteBounds(
                max_published_endpoints=True,
                acknowledgement_timeout_s=1.0,
                max_running_s=2.0,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            H4ExecuteBounds(
                max_published_endpoints=1,
                acknowledgement_timeout_s="1.0",  # type: ignore[arg-type]
                max_running_s=2.0,
            )

    def test_b_relative_limit_waits_for_running_and_exits_after_armed_ack(self) -> None:
        clock = _SupervisorClock()
        shared = SimpleNamespace(
            safety_state=_Value(int(SafetyState.ARMED)),
            run_started_monotonic_ns=_Value(0),
            start_request=_Value(1),
            stop_request=_Value(int(StopRequest.NONE)),
            is_running=_Value(1),
            error_state=_Value(0),
            estop_request=_Value(0),
            quit_requested=_Value(0),
            get_heartbeat=lambda _name: clock.now_s,
        )
        clock.attach(shared)
        process = SimpleNamespace(name="policy", exitcode=None)

        with (
            patch("dexmani_real.runtime.supervisor.time", clock),
            patch(
                "dexmani_real.runtime.workers.supervisor_exit_reason",
                return_value=ExitReason.NONE,
            ),
        ):
            exit_reason, normal_exit = run_supervisor(
                shared,
                [process],
                ["policy"],
                ["policy"],
                heartbeat_timeouts_s={"policy": 0.5},
                supervisor_hz=10.0,
                max_running_s=1.0,
            )

        self.assertEqual(exit_reason, "run time limit reached")
        self.assertTrue(normal_exit)
        self.assertEqual(clock.limit_request_times_s, [2.0])
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.stop_request.value, int(StopRequest.NONE))
        self.assertFalse(shared.start_request.value)

    def test_b_relative_limit_faults_when_coordinator_does_not_acknowledge(
        self,
    ) -> None:
        clock = _SupervisorClock(acknowledge_time_limit=False)
        clock.now_s = 1.0
        shared = SimpleNamespace(
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(int(clock.now_s * 1e9)),
            active_coupled_command_sequence=_Value(0),
            motion_lock=threading.RLock(),
            start_request=_Value(1),
            stop_request=_Value(int(StopRequest.NONE)),
            is_running=_Value(1),
            error_state=_Value(0),
            estop_request=_Value(0),
            quit_requested=_Value(0),
            get_heartbeat=lambda _name: clock.now_s,
        )
        clock.attach(shared)
        process = SimpleNamespace(name="policy", exitcode=None)

        with (
            patch("dexmani_real.runtime.supervisor.time", clock),
            patch(
                "dexmani_real.runtime.workers.supervisor_exit_reason",
                return_value=ExitReason.NONE,
            ),
        ):
            exit_reason, normal_exit = run_supervisor(
                shared,
                [process],
                ["policy"],
                ["policy"],
                heartbeat_timeouts_s={"policy": 0.5},
                supervisor_hz=10.0,
                max_running_s=1.0,
            )

        self.assertEqual(exit_reason, "run time limit stop was not acknowledged")
        self.assertFalse(normal_exit)
        self.assertEqual(clock.limit_request_times_s, [])
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))

    def test_execute_time_limit_keeps_explicit_supervisor_reason(self) -> None:
        clock = _SupervisorClock()
        shared = SimpleNamespace(
            safety_state=_Value(int(SafetyState.ARMED)),
            run_started_monotonic_ns=_Value(0),
            start_request=_Value(1),
            stop_request=_Value(int(StopRequest.NONE)),
            is_running=_Value(1),
            error_state=_Value(0),
            estop_request=_Value(0),
            quit_requested=_Value(0),
            get_heartbeat=lambda _name: clock.now_s,
        )
        clock.attach(shared)
        process = SimpleNamespace(name="policy", exitcode=None)

        with (
            patch("dexmani_real.runtime.supervisor.time", clock),
            patch(
                "dexmani_real.runtime.workers.supervisor_exit_reason",
                return_value=ExitReason.NONE,
            ),
        ):
            exit_reason, normal_exit = run_supervisor(
                shared,
                [process],
                ["policy"],
                ["policy"],
                heartbeat_timeouts_s={"policy": 0.5},
                supervisor_hz=10.0,
                max_running_s=1.0,
                exit_after_run_stops=True,
            )

        self.assertEqual(exit_reason, "run time limit reached")
        self.assertTrue(normal_exit)

    def test_policy_endpoint_disposition_uses_only_typed_outcomes(self) -> None:
        def classify(result: CommandPublishResult, *, nested: bool = True):
            return classify_policy_endpoint_disposition(
                result,
                hand_limit_nesting_valid=nested,
            )

        self.assertEqual(
            classify(CommandPublishResult(CommandPublishStatus.PUBLISHED)),
            PolicyEndpointDisposition.COMMIT,
        )
        self.assertEqual(
            classify(CommandPublishResult(CommandPublishStatus.SHADOW_VALIDATED)),
            PolicyEndpointDisposition.COMMIT,
        )
        self.assertEqual(
            classify(CommandPublishResult(CommandPublishStatus.TEMPORAL_WINDOW_CLOSED)),
            PolicyEndpointDisposition.DISCARD_STALE,
        )
        self.assertEqual(
            classify(
                CommandPublishResult(
                    CommandPublishStatus.GATE_REJECTED,
                    gate_code=GateRejectCode.COLLISION_TRANSITION,
                )
            ),
            PolicyEndpointDisposition.DISCARD_MOTION,
        )
        self.assertEqual(
            classify(
                CommandPublishResult(
                    CommandPublishStatus.GATE_REJECTED,
                    gate_code=GateRejectCode.COLLISION_CHECK_FAILED,
                )
            ),
            PolicyEndpointDisposition.ABORT_FATAL,
        )
        self.assertEqual(
            classify(
                CommandPublishResult(
                    CommandPublishStatus.GATE_REJECTED,
                    gate_code=GateRejectCode.RUN_GENERATION_MISMATCH,
                )
            ),
            PolicyEndpointDisposition.DEFER_TRANSIENT,
        )
        self.assertEqual(
            classify(
                CommandPublishResult(
                    CommandPublishStatus.HAND_PREFLIGHT_REJECTED,
                ),
                nested=False,
            ),
            PolicyEndpointDisposition.ABORT_FATAL,
        )
        for status in (
            CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE,
            CommandPublishStatus.HAND_FEEDBACK_UNAVAILABLE,
        ):
            self.assertEqual(
                classify(CommandPublishResult(status)),
                PolicyEndpointDisposition.DEFER_TRANSIENT,
            )
        for status in (
            CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
            CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
        ):
            for code in FeedbackIssueCode:
                expected = (
                    PolicyEndpointDisposition.DEFER_TRANSIENT
                    if code is FeedbackIssueCode.STALE
                    else PolicyEndpointDisposition.ABORT_FATAL
                )
                self.assertEqual(
                    classify(
                        CommandPublishResult(
                            status,
                            feedback_issue=FeedbackIssue(code, code.value),
                        )
                    ),
                    expected,
                    f"{status.value}: {code.value}",
                )
        self.assertEqual(
            classify(CommandPublishResult(CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY)),
            PolicyEndpointDisposition.ABORT_FATAL,
        )

    def test_feedback_structure_precedes_connection_and_freshness(self) -> None:
        arm_issue = diagnose_arm_feedback(
            connected=False,
            error_code=0,
            state_valid=True,
            source_monotonic_ns=1,
            now_monotonic_ns=2_000_000_000,
            max_age_s=0.1,
            qpos=np.full(7, np.nan),
            qvel=np.zeros(7),
        )
        hand_issue = diagnose_hand_feedback(
            connected=False,
            state_valid=True,
            source_monotonic_ns=1,
            now_monotonic_ns=2_000_000_000,
            max_age_s=0.1,
            qpos=np.full(12, np.inf),
        )

        assert arm_issue is not None and hand_issue is not None
        self.assertEqual(arm_issue.code, FeedbackIssueCode.NONFINITE)
        self.assertEqual(hand_issue.code, FeedbackIssueCode.NONFINITE)

    def test_final_ik_collision_checker_exception_is_typed_fatal(self) -> None:
        class _Kinematics:
            @staticmethod
            def compute_world_pose_error(_target, _qpos):
                return 0.0, 0.0

        class _IkManager:
            @staticmethod
            def canonicalize_qpos(qpos, _current):
                return qpos

            @staticmethod
            def limit_violation(_qpos, _limits):
                return np.zeros(7, dtype=bool), None

            @staticmethod
            def compute_qpos_delta(qpos, current):
                return qpos - current

            joint_limits = object()

        solver = object.__new__(TeleopIKSolver)
        solver.kin = _Kinematics()
        solver.ik_mgr = _IkManager()
        profile = SimpleNamespace(
            enable_nullspace_optimization=False,
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=0.01,
            check_self_collision=True,
        )
        qpos = np.zeros(7, dtype=np.float64)

        def checker_failure(_qpos):
            raise LookupError("collision backend unavailable")

        solver.ik_mgr.has_collision = checker_failure
        broken = solver._command_from_target_qpos(
            target_eef_pose_world=object(),
            current_qpos=qpos,
            previous_qpos_cmd=qpos,
            target_qpos=qpos,
            profile=profile,
            report={},
        )
        self.assertFalse(broken.success)
        self.assertTrue(broken.held)
        self.assertEqual(broken.failure_kind, IKFailureKind.CHECKER_FAILURE)

        solver.ik_mgr.has_collision = lambda _qpos: True
        solver.ik_mgr.check_collision = lambda _qpos: CollisionInfo(in_collision=True)
        collision = solver._command_from_target_qpos(
            target_eef_pose_world=object(),
            current_qpos=qpos,
            previous_qpos_cmd=qpos,
            target_qpos=qpos,
            profile=profile,
            report={},
        )
        self.assertEqual(collision.failure_kind, IKFailureKind.COLLISION)

        invalid_target = solver._command_from_target_qpos(
            target_eef_pose_world=object(),
            current_qpos=qpos,
            previous_qpos_cmd=qpos,
            target_qpos=np.full(7, np.nan),
            profile=profile,
            report={},
        )
        self.assertTrue(invalid_target.held)
        self.assertEqual(invalid_target.failure_kind, IKFailureKind.INVALID_OUTPUT)

        solver.kin.compute_world_pose_error = lambda _target, _qpos: (np.inf, 0.0)
        invalid_pose = solver._command_from_target_qpos(
            target_eef_pose_world=object(),
            current_qpos=qpos,
            previous_qpos_cmd=qpos,
            target_qpos=qpos,
            profile=profile,
            report={},
        )
        self.assertTrue(invalid_pose.held)
        self.assertEqual(invalid_pose.failure_kind, IKFailureKind.INVALID_OUTPUT)

    def test_preliminary_ik_checker_and_solver_output_are_fatal_typed(self) -> None:
        class _Kinematics:
            dof = 7

            @staticmethod
            def world_to_base_pose(target):
                return target

            @staticmethod
            def compute_manipulability(_qpos):
                return 1.0

            @staticmethod
            def compute_eef_jacobian_and_pose_world(_qpos):
                return np.zeros((6, 7)), object()

            @staticmethod
            def manipulability_from_jacobian(_jacobian):
                return 1.0

        class _IkManager:
            joint_limits = object()

            def __init__(self, raw_qpos, *, checker):
                self.raw_qpos = raw_qpos
                self.checker = checker
                self.calls = 0

            @staticmethod
            def profile_array(value, _name):
                return np.asarray(value, dtype=np.float64)

            def call_mplib_ik(self, *_args, **_kwargs):
                self.calls += 1
                return "Success", self.raw_qpos

            @staticmethod
            def canonicalize_qpos(qpos, _current):
                return qpos

            @staticmethod
            def limit_violation(_qpos, _limits):
                return np.zeros(7, dtype=bool), None

            @staticmethod
            def compute_qpos_delta(qpos, current):
                return qpos - current

            @staticmethod
            def weighted_joint_distance(*_args, **_kwargs):
                return 0.0

            def has_collision(self, qpos):
                return self.checker(qpos)

        profile = SimpleNamespace(
            max_ik_jump_deg=np.full(7, 180.0),
            position_ik_fast_accept_rad=1.0,
            joint_weights=np.ones(7),
            check_self_collision=True,
            max_pose_error_pos_m=0.01,
            max_pose_error_rot_rad=0.01,
            position_ik_min_manipulability=0.0,
        )

        def make_solver(raw_qpos, checker):
            solver = object.__new__(TeleopIKSolver)
            solver.kin = _Kinematics()
            solver.ik_mgr = _IkManager(raw_qpos, checker=checker)
            solver.profile = profile
            solver._elbow_joint_index = 3
            solver._hold_start = None
            solver._hold_warned = False
            solver._make_teleop_seeds = lambda *_args: [
                ("first", np.zeros(7), 1),
                ("later", np.ones(7), 1),
            ]
            return solver

        raw_nan_solver = make_solver(np.full(7, np.nan), lambda _qpos: False)
        raw_nan = raw_nan_solver.solve(object(), np.zeros(7), np.zeros(7))
        self.assertTrue(raw_nan.held)
        self.assertEqual(raw_nan.failure_kind, IKFailureKind.INVALID_OUTPUT)

        checker_solver = make_solver(
            np.zeros(7),
            lambda _qpos: (_ for _ in ()).throw(LookupError("checker unavailable")),
        )
        with patch(
            "dexmani_real.planning.ik.compute_pose_error", return_value=(0.0, 0.0)
        ):
            checker_failed = checker_solver.solve(object(), np.zeros(7), np.zeros(7))
        self.assertTrue(checker_failed.held)
        self.assertEqual(checker_failed.failure_kind, IKFailureKind.CHECKER_FAILURE)
        self.assertEqual(checker_solver.ik_mgr.calls, 1)

    def test_source_aligned_state_allows_delivery_after_camera_sample(self) -> None:
        records: list[tuple[np.ndarray, int, int]] = []
        for sequence, source_ns, publish_ns in (
            (1, 80, 160),
            (2, 120, 130),
            (3, 90, 95),
        ):
            state = np.zeros(1, dtype=ARM_STATE_DTYPE)
            state["source_monotonic_ns"][0] = source_ns
            state["publish_monotonic_ns"][0] = publish_ns
            records.append((state, publish_ns, sequence))

        selected = read_structured_frame_aligned_to_source(
            _SequenceRing(records),
            source_field="source_monotonic_ns",
            reference_source_monotonic_ns=100,
            anchor_monotonic_ns=200,
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        data, publish_ns, sequence = selected
        self.assertEqual(int(data["source_monotonic_ns"][0]), 90)
        self.assertEqual(publish_ns, 95)
        self.assertEqual(sequence, 3)

    def test_stale_hand_qpos_is_not_a_policy_observation(self) -> None:
        state = np.zeros(1, dtype=HAND_STATE_DTYPE)
        state["source_monotonic_ns"][0] = 100
        state["publish_monotonic_ns"][0] = 110
        state["state_valid"][0] = 1
        state["qpos_stale"][0] = 1

        history = _read_state_history(
            _Ring([(state, 110, 1)]),
            history_len=1,
            anchor_ns=200,
            values_field="qpos",
            required_true_fields=("state_valid",),
            required_false_fields=("qpos_stale",),
        )

        self.assertIsNone(history)

    def test_deployment_wrapper_rejects_ignored_sibling_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "sibling"):
            resolve_deployment_config(
                data={
                    "deployment": {"runtime_target": "tests:fake"},
                    "task_nmae": "typo",
                }
            )

    def test_observation_freshness_allows_the_required_history_span(self) -> None:
        run_ns = 1_000_000_000
        step_ns = 62_500_000
        anchor_ns = run_ns + 260_000_000
        cloud_records: list[tuple[np.ndarray, int, int]] = []
        arm_records: list[tuple[np.ndarray, int, int]] = []
        point_dtype = make_pointcloud_frame_dtype(1024)
        for index in range(1, 5):
            source_ns = run_ns + index * step_ns - 10_000_000
            cloud = np.zeros(1, dtype=point_dtype)
            cloud["source_camera_sequence"][0] = index
            cloud["source_monotonic_ns"][0] = source_ns
            cloud["camera_publish_monotonic_ns"][0] = source_ns + 1
            cloud["publish_monotonic_ns"][0] = source_ns + 2
            cloud["camera_generation"][0] = 1
            cloud_records.append((cloud, source_ns + 3, index))

            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            arm["source_monotonic_ns"][0] = source_ns - 1
            arm["publish_monotonic_ns"][0] = source_ns
            arm["state_valid"][0] = 1
            arm_records.append((arm, source_ns, index))

        hand_records: list[tuple[np.ndarray, int, int]] = []
        offsets_ns = range(40_000_000, 251_000_000, 10_000_000)
        for index, offset_ns in enumerate(offsets_ns, 1):
            source_ns = run_ns + offset_ns
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            hand["source_monotonic_ns"][0] = source_ns
            hand["publish_monotonic_ns"][0] = source_ns + 1
            hand["state_valid"][0] = 1
            hand_records.append((hand, source_ns + 1, index))

        shared = SimpleNamespace(
            arm_state_ring=_Ring(arm_records),
            hand_state_ring=_Ring(hand_records),
            hand_tactile_ring=_Ring([]),
            pointcloud_ring=_Ring(cloud_records),
        )
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=4,
            max_input_age_s=0.15,
            hand_enabled=True,
            observation_fields="arm_qpos,hand_qpos,hand_current,point_cloud",
        )
        observation = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=run_ns,
            anchor_ns=anchor_ns,
            step_dt_ns=step_ns,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertEqual(len(observation.pointcloud_history), 4)
        assert observation.hand_current_history is not None
        self.assertEqual(observation.hand_current_history.values.shape[0], 4)

    def test_state_history_budget_includes_grid_lag_and_cross_modal_skew(self) -> None:
        run_ns = 1_000_000_000
        anchor_ns = run_ns + 350_000_000
        point_dtype = make_pointcloud_frame_dtype(1024)
        cloud_records: list[tuple[np.ndarray, int, int]] = []
        arm_records: list[tuple[np.ndarray, int, int]] = []
        hand_records: list[tuple[np.ndarray, int, int]] = []
        for sequence, cloud_source_ns in enumerate(
            (run_ns + 120_000_000, run_ns + 220_000_000), start=1
        ):
            cloud = np.zeros(1, dtype=point_dtype)
            cloud["source_camera_sequence"][0] = sequence
            cloud["source_monotonic_ns"][0] = cloud_source_ns
            cloud["camera_publish_monotonic_ns"][0] = cloud_source_ns + 1
            cloud["publish_monotonic_ns"][0] = cloud_source_ns + 2
            cloud["camera_generation"][0] = 1
            cloud_records.append((cloud, cloud_source_ns + 3, sequence))

            state_source_ns = cloud_source_ns - 90_000_000
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            arm["source_monotonic_ns"][0] = state_source_ns
            arm["publish_monotonic_ns"][0] = state_source_ns + 1
            arm["state_valid"][0] = 1
            arm_records.append((arm, state_source_ns + 1, sequence))
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            hand["source_monotonic_ns"][0] = state_source_ns
            hand["publish_monotonic_ns"][0] = state_source_ns + 1
            hand["state_valid"][0] = 1
            hand_records.append((hand, state_source_ns + 1, sequence))

        shared = SimpleNamespace(
            arm_state_ring=_Ring(arm_records),
            hand_state_ring=_Ring(hand_records),
            hand_tactile_ring=_Ring([]),
            pointcloud_ring=_Ring(cloud_records),
        )
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=2,
            max_input_age_s=0.15,
            max_grid_lag_s=0.08,
            max_observation_skew_s=0.10,
            hand_enabled=True,
            observation_fields="arm_qpos,hand_qpos,point_cloud",
        )

        observation = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=run_ns,
            anchor_ns=anchor_ns,
            step_dt_ns=100_000_000,
        )

        self.assertIsNotNone(observation)

    def test_sensor_ring_capacity_covers_the_complete_freshness_budget(self) -> None:
        runtime = resolve_runtime_config()
        config = RuntimeChannelsConfig.from_runtime(
            runtime,
            pointcloud_requested=True,
            observation_horizon=4,
            observation_dt_s=0.0625,
            max_input_age_s=0.15,
            max_observation_skew_s=0.10,
            max_grid_lag_s=0.08,
        )
        required_span_s = 3 * 0.0625 + 0.15 + 0.10 + 0.08
        pointcloud_span_s = 3 * 0.0625 + 0.15 + 0.08
        self.assertGreaterEqual(
            config.hand_state_ring_maxlen,
            int(np.ceil(runtime.hand.loop_hz * required_span_s)) + 2,
        )
        self.assertEqual(
            config.hand_tactile_ring_maxlen,
            config.hand_state_ring_maxlen,
        )
        self.assertGreaterEqual(
            config.pointcloud_ring_maxlen,
            int(np.ceil(runtime.camera.fps * pointcloud_span_s)) + 2,
        )

    def test_policy_runtime_config_and_training_data_contract_roundtrip(self) -> None:
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                hand_enabled=True,
                task_name="pick",
            ),
            control_dt_s=0.0625,
            point_cloud_frame="xarm_base",
            point_cloud_color_source="aligned_rgb",
            point_cloud_policy_id="policy-v1",
            point_cloud_config_sha256="a" * 64,
            point_cloud_table_plane_abcd_json="null",
            point_cloud_sampling="sample-v1",
            point_cloud_transform="transform-v1",
        )
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.runtime_target, "tests:fake")
        contract = {
            "domain": "real",
            "schema_name": "dexmani-real-policy-zarr",
            "schema_version": 5,
            "episode_start_policy": "full_history",
            "obs_alignment": "obs[t]_before_action[t]",
            "observation_reference": "camera_source_monotonic_ns",
            "state_alignment": "camera_source_aligned_state",
            "max_observation_skew_s": config.max_observation_skew_s,
            "action_semantics": "deployment_grid_rate_limited_target",
            "arm_max_delta_rad_per_tick": config.arm_max_delta_rad_per_tick,
            "hand_max_delta_rad_per_tick": config.hand_max_delta_rad_per_tick,
            "endpoint_delta_tolerance_rad": config.endpoint_delta_tolerance_rad,
            "deployment_equivalent": True,
            "profile": "pointcloud",
            "task_name": "pick",
            "dt": 0.0625,
            "sensor_modalities": ["joint_state", "point_cloud"],
            "point_cloud_frame": config.point_cloud_frame,
            "point_cloud_color_source": config.point_cloud_color_source,
            "point_cloud_policy_id": config.point_cloud_policy_id,
            "point_cloud_config_sha256": config.point_cloud_config_sha256,
            "point_cloud_table_plane_abcd_json": (
                config.point_cloud_table_plane_abcd_json
            ),
            "point_cloud_sampling": config.point_cloud_sampling,
            "point_cloud_transform": config.point_cloud_transform,
            "point_cloud_num_points": 1024,
            "point_cloud_feature_dim": 6,
        }
        _validate_training_data_contract(contract, restored)
        contract.pop("endpoint_delta_tolerance_rad")
        with self.assertRaises(ValueError):
            _validate_training_data_contract(contract, restored)
        contract["endpoint_delta_tolerance_rad"] = config.endpoint_delta_tolerance_rad
        contract["domain"] = "sim"
        with self.assertRaises(ValueError):
            _validate_training_data_contract(contract, restored)

    def test_integer_wire_arrays_reject_lossy_casts(self) -> None:
        with self.assertRaises(TypeError):
            freeze_array(np.asarray([1.9]), name="timestamp", dtype=np.uint64)
        with self.assertRaises(ValueError):
            freeze_array(
                np.asarray([-1], dtype=np.int64), name="timestamp", dtype=np.uint64
            )

    def test_run_epoch_is_atomic_and_revocation_clears_it(self) -> None:
        shared = _shared_epoch()

        self.assertTrue(begin_motion(shared))
        epoch = read_run_epoch(shared)
        self.assertEqual(epoch.generation, 5)
        self.assertGreater(epoch.started_monotonic_ns, 0)
        self.assertEqual(shared.active_coupled_command_sequence.value, 0)

        self.assertTrue(revoke_motion(shared, SafetyState.ARMED))
        revoked = read_run_epoch(shared)
        self.assertEqual(revoked.generation, 6)
        self.assertEqual(revoked.started_monotonic_ns, 0)

    def test_pointcloud_history_uses_policy_grid_not_producer_cadence(self) -> None:
        run_ns = 1_000_000_000
        frames = tuple(
            _pointcloud(sequence, source_ns)
            for sequence, source_ns in enumerate(
                (1_080_000_000, 1_180_000_000, 1_290_000_000, 1_390_000_000),
                start=1,
            )
        )

        selected, logical_ns = _select_pointcloud_control_grid(
            frames,
            run_started_ns=run_ns,
            anchor_ns=1_390_000_000,
            history_len=3,
            step_dt_ns=100_000_000,
            max_grid_lag_ns=30_000_000,
        )

        self.assertEqual(logical_ns, 1_300_000_000)
        self.assertEqual(
            [frame.source_camera_sequence for frame in selected], [1, 2, 3]
        )

    def test_pointcloud_grid_never_reuses_one_source_frame(self) -> None:
        frames = tuple(
            _pointcloud(sequence, source_ns)
            for sequence, source_ns in enumerate(
                (1_090_000_000, 1_290_000_000, 1_390_000_000), start=1
            )
        )

        selected, logical_ns = _select_pointcloud_control_grid(
            frames,
            run_started_ns=1_000_000_000,
            anchor_ns=1_390_000_000,
            history_len=3,
            step_dt_ns=100_000_000,
            max_grid_lag_ns=120_000_000,
        )

        self.assertEqual(selected, ())
        self.assertEqual(logical_ns, 0)

    def test_observation_reports_unavailable_pointcloud_grid(self) -> None:
        metrics = Metrics()
        shared = SimpleNamespace(
            arm_state_ring=_Ring([]),
            pointcloud_ring=_Ring([]),
        )
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=2,
            observation_fields="arm_qpos,point_cloud",
        )

        observation = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=100,
            anchor_ns=300,
            step_dt_ns=100,
            metrics=metrics,
        )

        self.assertIsNone(observation)
        self.assertEqual(
            metrics.run_snapshot()[OBSERVATION_WAIT_POINTCLOUD_GRID],
            1,
        )

    def test_ipc_plan_conversion_copies_without_retiming(self) -> None:
        plan = _plan()
        original_targets = plan["target_monotonic_ns"].copy()
        buffered = _buffered_plan_from_record(
            plan,
            max_plan_age_ns=500_000_000,
            max_source_to_command_age_ns=400_000_000,
        )

        self.assertEqual(buffered.identity, (2, 1))
        np.testing.assert_array_equal(
            buffered.chunk.target_monotonic_ns,
            original_targets[:3],
        )
        np.testing.assert_array_equal(plan["target_monotonic_ns"], original_targets)

    def test_ipc_plan_conversion_fails_on_nonfinite_but_defers_finite_ee_geometry(
        self,
    ) -> None:
        nonfinite = _plan()
        nonfinite["hand_qpos"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "hand targets must be finite"):
            _buffered_plan_from_record(
                nonfinite,
                max_plan_age_ns=500_000_000,
                max_source_to_command_age_ns=400_000_000,
            )

        ee = _plan()
        ee["arm_present"] = 0
        ee["ee_present"] = 1
        # All-zero rot6d is finite and structurally valid IPC; the due endpoint
        # is motion-discarded by the coordinator's geometry boundary.
        buffered = _buffered_plan_from_record(
            ee,
            max_plan_age_ns=500_000_000,
            max_source_to_command_age_ns=400_000_000,
        )
        self.assertTrue(buffered.chunk.is_ee)

    def test_publish_plan_drops_after_generation_advances(self) -> None:
        chunk = JointActionChunk(
            arm_qpos=np.zeros((1, 7), dtype=np.float64),
            hand_qpos=np.zeros((1, 12), dtype=np.float64),
            target_monotonic_ns=np.asarray((1_100_000_000,), dtype=np.uint64),
            valid_mask=np.ones(1, dtype=np.uint8),
        )
        context = InferenceContext(
            run_generation=5,
            observation_id=2,
            observation_anchor_monotonic_ns=1_060_000_000,
            observation_latest_source_monotonic_ns=1_000_000_000,
            observation_logical_step_monotonic_ns=1_050_000_000,
            inference_started_monotonic_ns=1_070_000_000,
            inference_finished_monotonic_ns=1_080_000_000,
            step_dt_ns=62_500_000,
        )
        ring = _WriteRing()
        shared = SimpleNamespace(run_generation=_Value(6), policy_plan_ring=ring)

        self.assertFalse(publish_plan(shared, plan_id=1, context=context, chunk=chunk))
        self.assertIsNone(ring.frame)

    def _run_inference_with_prediction(
        self,
        *,
        prediction: PolicyPrediction,
        finished_ns: int,
    ) -> tuple[_WriteRing, SimpleNamespace, _Clock, SimpleNamespace]:
        logical_step_ns = 1_000_000_000
        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(5),
            run_started_monotonic_ns=_Value(logical_step_ns - 500_000_000),
            motion_lock=threading.RLock(),
            action_control_hz=10.0,
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        ring = _WriteRing(on_write=lambda: setattr(shared.is_running, "value", 0))
        shared.policy_plan_ring = ring
        observation = SimpleNamespace(
            arm_history=SimpleNamespace(values=np.zeros((2, 7))),
            hand_history=None,
            latest_source_monotonic_ns=logical_step_ns - 100_000_000,
            logical_step_monotonic_ns=logical_step_ns,
        )
        runtime = SimpleNamespace(
            load=Mock(),
            reset_episode=Mock(),
            predict=Mock(return_value=prediction),
            close=Mock(),
        )
        clock = _Clock(
            (
                logical_step_ns,
                logical_step_ns + 20_000_000,
                finished_ns,
            ),
            on_sleep=lambda: setattr(shared.is_running, "value", 0),
        )
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                inference_hz=10.0,
                observation_horizon=2,
                observation_fields="arm_qpos",
                command_lead_s=0.02,
            ),
            control_dt_s=0.1,
        )

        with (
            patch(
                "dexmani_real.deployment.worker.load_policy_runtime",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.worker._build_observation",
                return_value=observation,
            ),
            patch("dexmani_real.deployment.worker.time", clock),
        ):
            inference_loop(shared, config)

        return ring, runtime, clock, observation

    def test_inference_masks_expired_logical_grid_prefix_without_retiming(self) -> None:
        logical_step_ns = 1_000_000_000
        step_dt_ns = 100_000_000
        targets = logical_step_ns + step_dt_ns * np.asarray((0, 1, 2), dtype=np.uint64)
        prediction = PolicyPrediction(
            arm_qpos=np.zeros((3, 7)),
            hand_qpos=np.zeros((3, 12)),
        )

        ring, runtime, _clock, observation = self._run_inference_with_prediction(
            prediction=prediction,
            finished_ns=logical_step_ns + 150_000_000,
        )

        runtime.predict.assert_called_once_with(observation)
        self.assertEqual(len(runtime.predict.call_args.args), 1)
        self.assertFalse(runtime.predict.call_args.kwargs)
        assert ring.frame is not None
        np.testing.assert_array_equal(ring.frame["target_monotonic_ns"][0, :3], targets)
        np.testing.assert_array_equal(
            ring.frame["valid_mask"][0, :3], np.asarray((0, 0, 1), dtype=np.uint8)
        )

    def test_inference_drops_an_all_expired_logical_grid(self) -> None:
        logical_step_ns = 1_000_000_000
        prediction = PolicyPrediction(
            arm_qpos=np.zeros((3, 7)),
            hand_qpos=np.zeros((3, 12)),
        )

        ring, runtime, clock, _observation = self._run_inference_with_prediction(
            prediction=prediction,
            finished_ns=logical_step_ns + 350_000_000,
        )

        runtime.predict.assert_called_once()
        self.assertIsNone(ring.frame)
        self.assertTrue(clock.sleep_calls)

    def test_policy_prediction_validates_untimed_joint_and_ee_arrays(self) -> None:
        joint = PolicyPrediction(
            arm_qpos=np.zeros((2, 7), dtype=np.float64),
            hand_qpos=np.zeros((2, 12), dtype=np.float64),
        )
        self.assertFalse(joint.arm_qpos.flags.writeable)
        self.assertFalse(joint.hand_qpos.flags.writeable)
        ee = PolicyPrediction(
            arm_qpos=None,
            hand_qpos=None,
            ee_pos=np.zeros((2, 3), dtype=np.float64),
            ee_rot6d=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)), (2, 1)),
        )
        self.assertTrue(ee.is_ee)
        self.assertFalse(ee.ee_pos.flags.writeable)
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            PolicyPrediction(
                arm_qpos=np.full((1, 7), np.nan, dtype=np.float64),
                hand_qpos=None,
            )
        with self.assertRaisesRegex(ValueError, "arm_qpos must be"):
            PolicyPrediction(
                arm_qpos=np.zeros((1, 6), dtype=np.float64), hand_qpos=None
            )
        with self.assertRaisesRegex(ValueError, "transport capacity"):
            PolicyPrediction(
                arm_qpos=np.zeros((MAX_POLICY_CHUNK_STEPS + 1, 7), dtype=np.float64),
                hand_qpos=None,
            )

    def test_stamp_prediction_timing_preserves_grid_and_fails_closed_on_overflow(
        self,
    ) -> None:
        prediction = PolicyPrediction(
            arm_qpos=np.zeros((8, 7), dtype=np.float64),
            hand_qpos=np.zeros((8, 12), dtype=np.float64),
        )
        logical_ns = 1_000_000_000
        step_dt_ns = 100_000_000
        chunk = stamp_prediction_timing(
            prediction,
            logical_step_ns=logical_ns,
            step_dt_ns=step_dt_ns,
            inference_finished_ns=logical_ns - 1,
            command_lead_ns=0,
        )
        assert chunk is not None
        expected_targets = logical_ns + step_dt_ns * np.arange(8, dtype=np.uint64)
        np.testing.assert_array_equal(chunk.target_monotonic_ns, expected_targets)
        np.testing.assert_array_equal(chunk.valid_mask, np.ones(8, dtype=np.uint8))

        prefix = stamp_prediction_timing(
            prediction,
            logical_step_ns=logical_ns,
            step_dt_ns=step_dt_ns,
            inference_finished_ns=logical_ns + 250_000_000,
            command_lead_ns=0,
        )
        assert prefix is not None
        np.testing.assert_array_equal(
            prefix.valid_mask,
            np.asarray((0, 0, 0, 1, 1, 1, 1, 1), dtype=np.uint8),
        )
        self.assertIsNone(
            stamp_prediction_timing(
                prediction,
                logical_step_ns=logical_ns,
                step_dt_ns=step_dt_ns,
                inference_finished_ns=logical_ns + 800_000_000,
                command_lead_ns=0,
            )
        )
        with self.assertRaisesRegex(ValueError, "target grid exceeds uint64"):
            stamp_prediction_timing(
                prediction,
                logical_step_ns=(1 << 64) - 50,
                step_dt_ns=100,
                inference_finished_ns=1,
                command_lead_ns=0,
            )
        with self.assertRaisesRegex(ValueError, "exceeds uint64"):
            stamp_prediction_timing(
                prediction,
                logical_step_ns=logical_ns,
                step_dt_ns=step_dt_ns,
                inference_finished_ns=(1 << 64) - 1,
                command_lead_ns=1,
            )
        with self.assertRaisesRegex(TypeError, "logical_step_ns must be an integer"):
            stamp_prediction_timing(
                prediction,
                logical_step_ns=True,
                step_dt_ns=step_dt_ns,
                inference_finished_ns=1,
                command_lead_ns=0,
            )
        for invalid_name, kwargs in (
            ("logical_step_ns", {"logical_step_ns": 0}),
            ("step_dt_ns", {"step_dt_ns": 0}),
            ("inference_finished_ns", {"inference_finished_ns": 0}),
            ("command_lead_ns", {"command_lead_ns": -1}),
        ):
            values = {
                "logical_step_ns": logical_ns,
                "step_dt_ns": step_dt_ns,
                "inference_finished_ns": 1,
                "command_lead_ns": 0,
            }
            values.update(kwargs)
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaisesRegex(ValueError, invalid_name):
                    stamp_prediction_timing(prediction, **values)

    def test_observation_epoch_excludes_pre_b_feedback(self) -> None:
        pre_b = np.zeros(1, dtype=ARM_STATE_DTYPE)
        pre_b["source_monotonic_ns"][0] = 100
        pre_b["publish_monotonic_ns"][0] = 110
        pre_b["state_valid"][0] = 1
        post_b_first = np.zeros(1, dtype=ARM_STATE_DTYPE)
        post_b_first["source_monotonic_ns"][0] = 210
        post_b_first["publish_monotonic_ns"][0] = 220
        post_b_first["state_valid"][0] = 1
        post_b_second = np.zeros(1, dtype=ARM_STATE_DTYPE)
        post_b_second["source_monotonic_ns"][0] = 230
        post_b_second["publish_monotonic_ns"][0] = 240
        post_b_second["state_valid"][0] = 1
        config = DeploymentConfig(
            runtime_target="tests:fake",
            observation_horizon=2,
            observation_fields="arm_qpos",
        )
        shared = SimpleNamespace(
            arm_state_ring=_Ring([(pre_b, 110, 1), (post_b_first, 220, 2)]),
        )

        incomplete = _build_observation(
            shared,
            config,
            observation_id=1,
            run_generation=2,
            run_started_ns=200,
            anchor_ns=300,
            step_dt_ns=100,
        )
        assert incomplete is not None and incomplete.arm_history is not None
        self.assertEqual(incomplete.arm_history.values.shape[0], 1)

        shared.arm_state_ring = _Ring(
            [(pre_b, 110, 1), (post_b_first, 220, 2), (post_b_second, 240, 3)]
        )
        complete = _build_observation(
            shared,
            config,
            observation_id=2,
            run_generation=2,
            run_started_ns=200,
            anchor_ns=300,
            step_dt_ns=100,
        )

        assert complete is not None and complete.arm_history is not None
        np.testing.assert_array_equal(
            complete.arm_history.source_monotonic_ns,
            np.asarray((210, 230), dtype=np.uint64),
        )

    def test_armed_inference_worker_does_not_build_or_predict(self) -> None:
        runtime = SimpleNamespace(
            load=Mock(),
            reset_episode=Mock(),
            predict=Mock(),
            close=Mock(),
        )
        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.ARMED)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(0),
            motion_lock=threading.RLock(),
            action_control_hz=16.0,
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
            ),
            control_dt_s=1.0 / 16.0,
        )

        def stop_after_armed_idle(_sleep_s: float) -> None:
            shared.is_running.value = 0

        unexpected_observation = Mock(
            side_effect=AssertionError("ARMED must not build an observation")
        )
        with (
            patch(
                "dexmani_real.deployment.worker.load_policy_runtime",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.worker._build_observation",
                unexpected_observation,
            ),
            patch(
                "dexmani_real.deployment.worker.time.sleep",
                side_effect=stop_after_armed_idle,
            ) as armed_idle_sleep,
        ):
            fail_safe = threading.Timer(
                1.0,
                lambda: setattr(shared.is_running, "value", 0),
            )
            fail_safe.start()
            try:
                inference_loop(shared, config)
            finally:
                fail_safe.cancel()
                fail_safe.join()

        unexpected_observation.assert_not_called()
        runtime.predict.assert_not_called()
        runtime.load.assert_called_once_with()
        runtime.close.assert_called_once_with()
        armed_idle_sleep.assert_called_once()

    def test_running_inference_flushes_observation_wait_diagnostics(self) -> None:
        runtime = SimpleNamespace(
            load=Mock(),
            reset_episode=Mock(),
            predict=Mock(),
            close=Mock(),
        )
        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.RUNNING)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(1),
            motion_lock=threading.RLock(),
            action_control_hz=16.0,
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        config = PolicyRuntimeConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
            ),
            control_dt_s=1.0 / 16.0,
        )

        def stop_after_wait(_sleep_s: float) -> None:
            shared.is_running.value = 0

        with (
            patch(
                "dexmani_real.deployment.worker.load_policy_runtime",
                return_value=runtime,
            ),
            patch(
                "dexmani_real.deployment.worker._build_observation",
                return_value=None,
            ),
            patch(
                "dexmani_real.deployment.worker.flush_every",
                side_effect=lambda _metrics, *, last_ns, **_kwargs: last_ns,
            ) as flush_metrics,
            patch(
                "dexmani_real.deployment.worker.time.sleep",
                side_effect=stop_after_wait,
            ),
        ):
            inference_loop(shared, config)

        flush_metrics.assert_called_once()
        runtime.predict.assert_not_called()

    def test_coordinator_publishes_one_due_endpoint_through_its_safety_gate(
        self,
    ) -> None:
        base_ns = 4_000_000_000
        plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        plan["plan_id"][0] = 1
        plan["run_generation"][0] = 5
        plan["observation_id"][0] = 1
        plan["observation_latest_source_monotonic_ns"][0] = base_ns - 10_000_000
        plan["observation_logical_step_monotonic_ns"][0] = base_ns - 8_000_000
        plan["observation_anchor_monotonic_ns"][0] = base_ns - 6_000_000
        plan["inference_started_monotonic_ns"][0] = base_ns - 4_000_000
        plan["inference_finished_monotonic_ns"][0] = base_ns - 2_000_000
        plan["num_steps"][0] = 3
        plan["arm_present"][0] = 1
        plan["hand_present"][0] = 1
        plan["target_monotonic_ns"][0, :3] = (
            base_ns + 250_000_000,
            base_ns + 300_000_000,
            base_ns + 350_000_000,
        )
        plan["valid_mask"][0, :3] = 1

        class _PlanRing:
            def read_latest(self):
                return plan.copy(), base_ns, 1

        coupled_cmd_ring = SimpleNamespace(write=Mock(), latest_sequence=0)
        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.ARMED)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(0),
            active_coupled_command_sequence=_Value(0),
            motion_lock=threading.RLock(),
            arm_command_seq=_LockedValue(0),
            policy_plan_ring=_PlanRing(),
            coupled_cmd_ring=coupled_cmd_ring,
            quit_requested=_Value(0),
            start_request=_Value(1),
            stop_request=_Value(0),
            execute_completed=_Value(0),
            error_state=_Value(0),
            estop_request=_Value(0),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        config = CoordinatorConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
                command_lead_s=1e-6,
            ),
            arm_joint_lower_rad=tuple(np.full(7, -2.0)),
            arm_joint_upper_rad=tuple(np.full(7, 2.0)),
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(hand_defaults.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(hand_defaults.mechanical_qpos_max_rad),
            arm_feedback_max_age_s=0.5,
            hand_feedback_max_age_s=0.5,
            control_hz=100.0,
            execution_mode="shadow",
        )
        gate = object()
        published_candidates = []
        clock = _Clock((base_ns, base_ns, base_ns + 200_000_000, base_ns + 400_000_000))
        sleep_ticks = 0

        def bounded_sleep(_period_s: float, _tick_start: float) -> None:
            nonlocal sleep_ticks
            sleep_ticks += 1
            if sleep_ticks > 4:
                shared.is_running.value = 0

        def validate_through_gate(_shared, candidate, **kwargs):
            self.assertIs(kwargs["gate"], gate)
            self.assertEqual(kwargs["minimum_delivery_window_s"], 0.01)
            published_candidates.append(candidate)
            shared.is_running.value = 0
            return CommandPublishResult(
                CommandPublishStatus.SHADOW_VALIDATED,
                candidate,
            )

        with (
            patch("dexmani_real.deployment.coordinator.time", clock),
            patch("dexmani_real.control.publication.time", clock),
            patch("dexmani_real.deployment.coordinator.XArm7MotionPlanner"),
            patch(
                "dexmani_real.deployment.coordinator.planner_action_safety_gate",
                return_value=gate,
            ) as make_gate,
            patch(
                "dexmani_real.deployment.coordinator.read_arm_state_dict",
                return_value={"qpos": np.zeros(7)},
            ),
            patch(
                "dexmani_real.deployment.coordinator.validate_and_send_candidate",
                side_effect=validate_through_gate,
            ) as validate_and_send,
            patch(
                "dexmani_real.deployment.coordinator._sleep_tick",
                side_effect=bounded_sleep,
            ),
        ):
            coordinator_loop(shared, config)

        make_gate.assert_called_once()
        validate_and_send.assert_called_once()
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(
            published_candidates[0].scheduled_target_monotonic_ns,
            int(plan["target_monotonic_ns"][0, 2]),
        )

    def test_ee_feedback_uses_post_read_clock_for_freshness(self) -> None:
        base_ns = 8_000_000_000
        plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        plan["plan_id"][0] = 1
        plan["run_generation"][0] = 5
        plan["observation_id"][0] = 1
        plan["observation_latest_source_monotonic_ns"][0] = base_ns - 10_000_000
        plan["observation_logical_step_monotonic_ns"][0] = base_ns - 8_000_000
        plan["observation_anchor_monotonic_ns"][0] = base_ns - 6_000_000
        plan["inference_started_monotonic_ns"][0] = base_ns - 4_000_000
        plan["inference_finished_monotonic_ns"][0] = base_ns - 2_000_000
        plan["num_steps"][0] = 1
        plan["ee_present"][0] = 1
        plan["hand_present"][0] = 1
        plan["target_monotonic_ns"][0, 0] = base_ns
        plan["valid_mask"][0, 0] = 1
        plan["ee_rot6d"][0, 0] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        plan["hand_qpos"][0, 0] = hand_defaults.qpos_min_rad

        class _PlanRing:
            def read_latest(self):
                return plan.copy(), base_ns, 1

        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.ARMED)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(0),
            active_coupled_command_sequence=_Value(0),
            motion_lock=threading.RLock(),
            arm_command_seq=_LockedValue(0),
            policy_plan_ring=_PlanRing(),
            coupled_cmd_ring=SimpleNamespace(write=Mock(), latest_sequence=0),
            quit_requested=_Value(0),
            start_request=_Value(1),
            stop_request=_Value(0),
            execute_completed=_Value(0),
            error_state=_Value(0),
            estop_request=_Value(0),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        config = CoordinatorConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
                command_lead_s=1e-6,
            ),
            arm_joint_lower_rad=tuple(np.full(7, -2.0)),
            arm_joint_upper_rad=tuple(np.full(7, 2.0)),
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(hand_defaults.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(hand_defaults.mechanical_qpos_max_rad),
            arm_feedback_max_age_s=0.5,
            hand_feedback_max_age_s=0.5,
            control_hz=100.0,
            execution_mode="shadow",
        )
        planner = SimpleNamespace(
            collision_model=SimpleNamespace(check_transition_collision_free=Mock()),
            set_hand_qpos=Mock(),
            solve_teleop_ik=Mock(
                return_value=SimpleNamespace(success=True, qpos=np.zeros(7))
            ),
        )
        coordinator_clock = _Clock(
            (base_ns - 100, base_ns - 100, base_ns, base_ns + 20) + (base_ns + 30,) * 8
        )
        publication_clock = _Clock((base_ns + 30,) * 8)

        def publish_once(_shared, candidate, **_kwargs):
            shared.is_running.value = 0
            return CommandPublishResult(
                CommandPublishStatus.SHADOW_VALIDATED,
                candidate,
            )

        with (
            patch("dexmani_real.deployment.coordinator.time", coordinator_clock),
            patch("dexmani_real.control.publication.time", publication_clock),
            patch(
                "dexmani_real.deployment.coordinator.XArm7MotionPlanner",
                return_value=planner,
            ),
            patch(
                "dexmani_real.deployment.coordinator.planner_action_safety_gate",
                return_value=object(),
            ),
            patch(
                "dexmani_real.deployment.coordinator.read_arm_state_dict",
                return_value={
                    "connected": True,
                    "error_code": 0,
                    "state_valid": True,
                    "source_monotonic_ns": base_ns + 10,
                    "qpos": np.zeros(7),
                    "qvel": np.zeros(7),
                },
            ),
            patch(
                "dexmani_real.deployment.coordinator.validate_and_send_candidate",
                side_effect=publish_once,
            ),
            patch("dexmani_real.deployment.coordinator._sleep_tick"),
        ):
            coordinator_loop(shared, config)

        planner.solve_teleop_ik.assert_called_once()
        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))

    def test_physical_start_gate_requires_fresh_canonical_arm_home(self) -> None:
        config = CoordinatorConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
                hand_enabled=True,
            ),
            arm_joint_lower_rad=tuple(np.full(7, -4.0)),
            arm_joint_upper_rad=tuple(np.full(7, 4.0)),
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(hand_defaults.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(hand_defaults.mechanical_qpos_max_rad),
            arm_feedback_max_age_s=0.5,
            hand_feedback_max_age_s=0.5,
            control_hz=100.0,
            execution_mode="task",
            task_execute_bounds=TaskExecuteBounds(
                max_published_endpoints=2,
                acknowledgement_timeout_s=1.0,
                max_running_s=10.0,
            ),
            required_start_arm_qpos=tuple(np.zeros(7)),
            start_arm_home_tolerance_rad=0.01,
        )
        feedback = {
            "connected": True,
            "error_code": 0,
            "state_valid": True,
            "source_monotonic_ns": 1_000_000_000,
            "qpos": np.zeros(7),
            "qvel": np.zeros(7),
        }
        shared = SimpleNamespace(physical_home_completed=_Value(0))
        with (
            patch(
                "dexmani_real.deployment.coordinator.read_arm_state_dict",
                return_value=feedback,
            ),
            patch(
                "dexmani_real.deployment.coordinator.time.monotonic_ns",
                return_value=1_000_000_000,
            ),
        ):
            rejection = _physical_start_pose_rejection(shared, config)
            assert rejection is not None
            self.assertIn("has not completed in this process", rejection)

            shared.physical_home_completed.value = 1
            self.assertIsNone(_physical_start_pose_rejection(shared, config))
            feedback["qpos"] = np.array([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
            rejection = _physical_start_pose_rejection(shared, config)

        assert rejection is not None
        self.assertIn("press H before B", rejection)
        self.assertIn("max_abs_delta_rad=0.020000000", rejection)
        self.assertIn("tolerance_rad=0.010000000", rejection)

    def _run_single_endpoint_coordinator(
        self,
        result: CommandPublishResult,
        *,
        mutate_shadow_sequence: bool = False,
        missing_shadow_sequence: bool = False,
        stop_request_after_start: StopRequest | None = None,
        execution_mode: str = "shadow",
        h4_acknowledgement: CommandPublishStatus | None = None,
        h4_ack_timeout_s: float = 1.0,
        late_h4_acknowledgement: bool = False,
        request_second_begin: bool = False,
        execute_sequence_delta: int = 1,
        stop_request_after_publication: StopRequest | None = None,
        clock_values: tuple[int, ...] | None = None,
        physical_home_completed: bool = True,
    ) -> tuple[SimpleNamespace, SimpleNamespace, list[object]]:
        base_ns = 6_000_000_000
        plan = np.zeros(1, dtype=POLICY_PLAN_DTYPE)
        plan["plan_id"][0] = 1
        plan["run_generation"][0] = 5
        plan["observation_id"][0] = 1
        plan["observation_latest_source_monotonic_ns"][0] = base_ns - 10
        plan["observation_logical_step_monotonic_ns"][0] = base_ns - 8
        plan["observation_anchor_monotonic_ns"][0] = base_ns - 6
        plan["inference_started_monotonic_ns"][0] = base_ns - 4
        plan["inference_finished_monotonic_ns"][0] = base_ns - 2
        plan["num_steps"][0] = 2 if execution_mode == "task" else 1
        plan["arm_present"][0] = 1
        plan["hand_present"][0] = 1
        plan["target_monotonic_ns"][0, :2] = (base_ns, base_ns + 1)
        plan["valid_mask"][0, : (2 if execution_mode == "task" else 1)] = 1

        class _PlanRing:
            def read_latest(self):
                return plan.copy(), base_ns, 1

        coupled_cmd_ring = SimpleNamespace(write=Mock())
        if not missing_shadow_sequence:
            coupled_cmd_ring.latest_sequence = 0
        shared = SimpleNamespace(
            is_running=_Value(1),
            safety_state=_Value(int(SafetyState.ARMED)),
            run_generation=_Value(4),
            run_started_monotonic_ns=_Value(0),
            active_coupled_command_sequence=_Value(0),
            motion_lock=threading.RLock(),
            arm_command_seq=_LockedValue(0),
            policy_plan_ring=_PlanRing(),
            coupled_cmd_ring=coupled_cmd_ring,
            quit_requested=_Value(0),
            start_request=_Value(1),
            stop_request=_Value(0),
            execute_completed=_Value(0),
            physical_home_completed=_Value(physical_home_completed),
            error_state=_Value(0),
            estop_request=_Value(0),
            set_heartbeat=Mock(),
            set_ready=Mock(),
        )
        config = CoordinatorConfig(
            deployment=DeploymentConfig(
                runtime_target="tests:fake",
                observation_fields="arm_qpos",
                command_lead_s=1e-6,
                hand_enabled=execution_mode in {"execute", "task"},
            ),
            arm_joint_lower_rad=tuple(np.full(7, -2.0)),
            arm_joint_upper_rad=tuple(np.full(7, 2.0)),
            workspace_bounds=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
            hand_joint_lower_rad=tuple(hand_defaults.qpos_min_rad),
            hand_joint_upper_rad=tuple(hand_defaults.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(hand_defaults.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(hand_defaults.mechanical_qpos_max_rad),
            arm_feedback_max_age_s=0.5,
            hand_feedback_max_age_s=0.5,
            control_hz=100.0,
            execution_mode=execution_mode,
            h4_execute_bounds=(
                H4ExecuteBounds(
                    max_published_endpoints=1,
                    acknowledgement_timeout_s=h4_ack_timeout_s,
                    max_running_s=10.0,
                )
                if execution_mode == "execute"
                else None
            ),
            task_execute_bounds=(
                TaskExecuteBounds(
                    max_published_endpoints=2,
                    acknowledgement_timeout_s=h4_ack_timeout_s,
                    max_running_s=10.0,
                )
                if execution_mode == "task"
                else None
            ),
            required_start_arm_qpos=(
                tuple(np.zeros(7)) if execution_mode in {"execute", "task"} else None
            ),
            start_arm_home_tolerance_rad=(
                0.01 if execution_mode in {"execute", "task"} else None
            ),
        )
        candidates: list[object] = []
        clock = _Clock(clock_values or (base_ns,) * 16)
        sleep_ticks = 0

        def bounded_sleep(_period_s: float, _tick_start: float) -> None:
            nonlocal sleep_ticks
            sleep_ticks += 1
            if sleep_ticks == 1 and stop_request_after_start is not None:
                shared.stop_request.value = int(stop_request_after_start)
            if (
                request_second_begin
                and sleep_ticks == 2
                and shared.safety_state.value == int(SafetyState.ARMED)
            ):
                shared.start_request.value = 1
                return
            if execution_mode == "task" and coupled_cmd_ring.latest_sequence == 1:
                clock._monotonic_ns_values = iter((base_ns + 1,) * 40)
                clock._last_monotonic_ns = base_ns + 1
            if execution_mode in {"execute", "task"} and (
                shared.error_state.value
                or (
                    sleep_ticks > 1
                    and shared.safety_state.value == int(SafetyState.ARMED)
                )
            ):
                shared.is_running.value = 0
            elif sleep_ticks >= (50 if execution_mode in {"execute", "task"} else 3):
                shared.is_running.value = 0

        def publish_result(_shared, candidate, **_kwargs):
            candidates.append(candidate)
            if mutate_shadow_sequence:
                coupled_cmd_ring.latest_sequence += 1
            if execution_mode in {"execute", "task"}:
                coupled_cmd_ring.latest_sequence += execute_sequence_delta
                if stop_request_after_publication is not None:
                    shared.stop_request.value = int(stop_request_after_publication)
                return CommandPublishResult(
                    result.status,
                    candidate=candidate,
                    ticket=CoupledCommandTicket(5, 1),
                )
            if result.succeeded:
                return CommandPublishResult(result.status, candidate=candidate)
            return result

        def poll_acknowledgement(*_args, **_kwargs):
            if late_h4_acknowledgement:
                clock._monotonic_ns_values = iter((base_ns + 2,))
                clock._last_monotonic_ns = base_ns + 2
            return CommandPublishResult(
                h4_acknowledgement or CommandPublishStatus.ACK_PENDING
            )

        with (
            patch("dexmani_real.deployment.coordinator.time", clock),
            patch("dexmani_real.control.publication.time", clock),
            patch("dexmani_real.deployment.coordinator.XArm7MotionPlanner"),
            patch(
                "dexmani_real.deployment.coordinator.planner_action_safety_gate",
                return_value=object(),
            ),
            patch(
                "dexmani_real.deployment.coordinator.read_arm_state_dict",
                return_value={
                    "connected": True,
                    "error_code": 0,
                    "state_valid": True,
                    "source_monotonic_ns": base_ns,
                    "qpos": np.zeros(7),
                    "qvel": np.zeros(7),
                },
            ),
            patch(
                "dexmani_real.deployment.coordinator.validate_and_send_candidate",
                side_effect=publish_result,
            ),
            patch(
                "dexmani_real.deployment.coordinator.poll_coupled_command_acknowledgement",
                side_effect=poll_acknowledgement,
            ),
            patch(
                "dexmani_real.deployment.coordinator._sleep_tick",
                side_effect=bounded_sleep,
            ),
        ):
            coordinator_loop(shared, config)
        return shared, coupled_cmd_ring, candidates

    def test_coordinator_records_bounded_shadow_stop_reason(self) -> None:
        with self.assertLogs(
            "dexmani_real.deployment.coordinator", level="INFO"
        ) as logs:
            shared, coupled_cmd_ring, candidates = (
                self._run_single_endpoint_coordinator(
                    CommandPublishResult(CommandPublishStatus.SHADOW_VALIDATED),
                    stop_request_after_start=StopRequest.RUN_TIME_LIMIT,
                )
            )

        self.assertEqual(candidates, [])
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertTrue(
            any('"reason":"run time limit"' in message for message in logs.output)
        )

    def test_h4_rejects_an_acknowledgement_observed_after_its_deadline(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            h4_acknowledgement=CommandPublishStatus.APPLIED,
            h4_ack_timeout_s=1e-9,
            late_h4_acknowledgement=True,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.execute_completed.value, 0)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))

    def test_h4_ignores_a_second_b_after_the_first_run(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            h4_acknowledgement=CommandPublishStatus.APPLIED,
            request_second_begin=True,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.execute_completed.value, 1)

    def test_shadow_ignores_a_second_b_after_the_first_run(self) -> None:
        with self.assertLogs(
            "dexmani_real.deployment.coordinator", level="WARNING"
        ) as logs:
            shared, coupled_cmd_ring, candidates = (
                self._run_single_endpoint_coordinator(
                    CommandPublishResult(CommandPublishStatus.SHADOW_VALIDATED),
                    stop_request_after_start=StopRequest.RUN_TIME_LIMIT,
                    request_second_begin=True,
                )
            )

        self.assertEqual(candidates, [])
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertTrue(
            any("policy session already started" in message for message in logs.output)
        )

    def test_coordinator_motion_discard_does_not_publish_or_abort_run(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(
                CommandPublishStatus.GATE_REJECTED,
                gate_code=GateRejectCode.ARM_DELTA_LIMIT,
            )
        )

        self.assertEqual(len(candidates), 1)
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))

    def test_coordinator_transient_deferral_retries_the_same_endpoint(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE)
        )

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(
            {candidate.scheduled_target_monotonic_ns for candidate in candidates},
            {6_000_000_000},
        )
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.safety_state.value, int(SafetyState.RUNNING))

    def test_shadow_ring_mutation_latches_fault_and_blocks_second_begin(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.SHADOW_VALIDATED),
            mutate_shadow_sequence=True,
        )

        self.assertEqual(len(candidates), 1)
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_shadow_missing_ring_baseline_latches_fault_before_endpoint(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.SHADOW_VALIDATED),
            missing_shadow_sequence=True,
        )

        self.assertEqual(candidates, [])
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_h4_commits_exactly_one_endpoint_then_waits_for_dual_ack(self) -> None:
        with self.assertLogs(
            "dexmani_real.deployment.coordinator", level="INFO"
        ) as logs:
            shared, coupled_cmd_ring, candidates = (
                self._run_single_endpoint_coordinator(
                    CommandPublishResult(CommandPublishStatus.PUBLISHED),
                    execution_mode="execute",
                    h4_acknowledgement=CommandPublishStatus.APPLIED,
                )
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertTrue(
            any(
                '"coupled_command_writes":1' in message
                and '"acknowledged_action_id":1' in message
                and '"physical_home_completed":1' in message
                for message in logs.output
            )
        )
        self.assertEqual(shared.execute_completed.value, 1)

    def test_h4_ignores_b_until_physical_home_sequence_completed(self) -> None:
        with self.assertLogs(
            "dexmani_real.deployment.coordinator", level="WARNING"
        ) as logs:
            shared, coupled_cmd_ring, candidates = (
                self._run_single_endpoint_coordinator(
                    CommandPublishResult(CommandPublishStatus.PUBLISHED),
                    execution_mode="execute",
                    h4_acknowledgement=CommandPublishStatus.APPLIED,
                    physical_home_completed=False,
                )
            )

        self.assertEqual(candidates, [])
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.execute_completed.value, 0)
        self.assertTrue(
            any("home sequence has not completed" in message for message in logs.output)
        )

    def test_task_commits_each_endpoint_only_after_previous_dual_ack(self) -> None:
        with self.assertLogs(
            "dexmani_real.deployment.coordinator", level="INFO"
        ) as logs:
            shared, coupled_cmd_ring, candidates = (
                self._run_single_endpoint_coordinator(
                    CommandPublishResult(CommandPublishStatus.PUBLISHED),
                    execution_mode="task",
                    h4_acknowledgement=CommandPublishStatus.APPLIED,
                )
            )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 2)
        self.assertEqual(shared.safety_state.value, int(SafetyState.ARMED))
        self.assertEqual(shared.execute_completed.value, 1)
        self.assertTrue(
            any(
                '"execution_mode":"task"' in message
                and '"coupled_command_writes":2' in message
                and '"execute_acknowledged":2' in message
                for message in logs.output
            )
        )

    def test_h4_ack_timeout_latches_fault_without_another_publication(self) -> None:
        base_ns = 6_000_000_000
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            h4_acknowledgement=CommandPublishStatus.ACK_PENDING,
            h4_ack_timeout_s=1e-9,
            clock_values=(base_ns,) * 20 + (base_ns + 2,) * 80,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_h4_time_limit_with_pending_ack_latches_fault(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            h4_acknowledgement=CommandPublishStatus.ACK_PENDING,
            stop_request_after_publication=StopRequest.RUN_TIME_LIMIT,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 1)
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_h4_time_limit_before_first_publication_latches_fault(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            stop_request_after_start=StopRequest.RUN_TIME_LIMIT,
        )

        self.assertEqual(candidates, [])
        coupled_cmd_ring.write.assert_not_called()
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_h4_rejects_extra_coupled_ring_publication(self) -> None:
        shared, coupled_cmd_ring, candidates = self._run_single_endpoint_coordinator(
            CommandPublishResult(CommandPublishStatus.PUBLISHED),
            execution_mode="execute",
            h4_acknowledgement=CommandPublishStatus.APPLIED,
            execute_sequence_delta=2,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(coupled_cmd_ring.latest_sequence, 2)
        self.assertEqual(shared.error_state.value, 1)
        self.assertEqual(shared.safety_state.value, int(SafetyState.FAULT))
        self.assertFalse(begin_motion(shared))

    def test_plan_deadline_rejects_noncausal_inference_timestamps(self) -> None:
        plan = _plan()
        plan["inference_started_monotonic_ns"] = 1_040_000_000

        deadline = _plan_deadline_ns(
            plan,
            max_plan_age_ns=500_000_000,
            max_source_to_command_age_ns=400_000_000,
        )

        self.assertIsNone(deadline)

    def test_inference_context_rejects_start_before_causal_cut(self) -> None:
        with self.assertRaisesRegex(ValueError, "causal cut"):
            InferenceContext(
                run_generation=5,
                observation_id=1,
                observation_anchor_monotonic_ns=1_100,
                observation_latest_source_monotonic_ns=1_000,
                observation_logical_step_monotonic_ns=1_050,
                inference_started_monotonic_ns=1_090,
                inference_finished_monotonic_ns=1_200,
                step_dt_ns=50,
            )


if __name__ == "__main__":
    unittest.main()
