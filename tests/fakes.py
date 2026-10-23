"""Small offline fixtures; no SDK, device processes, or keyboard listeners."""

from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.deployment.inference import ModelResult
from dexmani_real.robot.commands import DispatchResult, DispatchStatus
from dexmani_real.robot.feedback import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.runtime.observation import ObservationNeeds, ObservationRow
from dexmani_real.runtime.state import RuntimeState


class FakeClock:
    def __init__(self):
        self.now = 10_000_000_000

    def monotonic_ns(self):
        return self.now

    def advance(self, seconds):
        self.now += round(seconds * 1e9)


def execution_snapshot(count=0, *, stopped=True):
    return dict(
        submitted=count, first_accepted=count, sdk_attempted=count,
        sdk_completed=count, sdk_accepted=count, cancelled_before_first=0,
        active_sequence=0, pending_sequence=0, stopped=stopped, robot_stopped=stopped,
        connected=True, fault=None, close_error=None, operation_unresolved=False,
        last_call=dict(started_ns=0, completed_ns=0),
    )


def camera_metadata():
    return dict(
        camera_payload_mode="depth_to_color_aligned_rgbd",
        camera_color_width=4, camera_color_height=4,
        camera_color_intrinsics=np.array([[4., 0., 1.5], [0., 4., 1.5], [0., 0., 1.]]).ravel(),
        camera_color_distortion_model="none", camera_color_distortion_coeffs=np.zeros(5),
        camera_T_xarm_base_from_color=np.eye(4).ravel(), depth_scale=0.001,
        hand_tactile_processing="sdk_native_bounded_previous_valid_vector_hold",
        hand_tactile_hold_max_age_s=0.2,
    )


def observation(clock, runtime=None, *, camera=False):
    runtime = runtime or ExperimentConfig()
    arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
    arm["qpos"] = runtime.arm.home_qpos
    hand["qpos"] = np.deg2rad(runtime.hand.home_qpos_deg)
    arm["timestamp_ns"] = hand["timestamp_ns"] = clock.now
    hand["tactile_aggregate_timestamp_ns"] = hand["tactile_dense_timestamp_ns"] = clock.now
    hand["tactile_aggregate"] = 1.25
    hand["tactile_dense"] = 2.5
    frame = dict(
        rgb=np.full((4, 4, 3), 80, np.uint8), depth=np.full((4, 4), 500, np.uint16),
        timestamp_ns=clock.now, received_ns=clock.now, published_ns=clock.now,
        ring_sequence=1, depth_frame_number=1, color_frame_number=1,
    ) if camera else None
    return ObservationRow(arm, hand, frame, None, None, clock.now,
                          control_timestamp_ns=clock.now, required_tactile=())


class FakeModel:
    def __init__(self, clock):
        self.clock = clock
        self.future = self.operation = self.context = None
        self.calls = []

    def submit(self, operation, *args, query_context=None, **kwargs):
        if self.future is not None:
            raise RuntimeError("unreclaimed Future")
        self.operation, self.context = operation, query_context
        self.future = Future()
        self.calls.append((operation, args, kwargs, query_context))

    def complete(self, value=None, error=None):
        self.future.set_result(ModelResult(value, error, self.clock.now, self.clock.now, self.context))

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        result = self.operation, self.future.result()
        self.future = self.operation = None
        return result


class FakeKeyboard:
    healthy = True
    estop_latched = False

    def __init__(self):
        self.signals = []

    def poll(self, timeout=0):
        result, self.signals = self.signals, []
        return result

    def drain_signal(self, signal):
        self.signals = [item for item in self.signals if item is not signal]


class FakeRecorder:
    control_hz = 20
    staging_path = Path("/tmp/fake-capture-staging")

    def __init__(self):
        self.is_recording = False
        self.frames, self.queries, self.starts, self.finishes = [], [], [], []
        self.start_error = self.frame_error = self.finish_error = None

    @property
    def written_frames(self):
        return len(self.frames)

    def check_error(self):
        pass

    def start_episode(self, **kwargs):
        self.starts.append(kwargs)
        if self.start_error:
            raise self.start_error
        self.is_recording = True

    def add_frame(self, row, command=None, result=None, *, timing=None):
        if self.frame_error:
            raise self.frame_error
        self.frames.append((row, command, result, dict(timing or {})))

    def record_query(self, evidence):
        self.queries.append(dict(evidence))

    def save_episode(self, **kwargs):
        self.finishes.append(kwargs)
        self.is_recording = False
        if self.finish_error:
            raise self.finish_error
        return Path("/tmp/fake-capture-published")


class FakeRobot:
    def __init__(self, clock, runtime):
        self.clock, self.runtime = clock, runtime
        self.sent = []
        self.stop_required = False
        self.stop_error = self.dispatch_error = None
        self.stops = 0
        self.arm = SimpleNamespace(
            check_error=lambda: None, update_health=lambda **kw: None,
            limit_execution_until=lambda until: None,
            execution_snapshot=lambda: execution_snapshot(len(self.sent), stopped=not self.stop_required),
            execution_metadata=lambda dt: {},
        )
        self.hand = None

    def check(self):
        pass

    def service_idle(self):
        if self.stop_required:
            self.stop()

    def reset_hand_feedback(self):
        pass

    def prepare_streaming(self, epoch):
        self.stop_required = True

    def end_hand_send_interval(self):
        pass

    def send_action(self, command, **kwargs):
        self.sent.append((command, kwargs))
        self.stop_required = True
        if self.dispatch_error:
            raise self.dispatch_error
        return DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.ACCEPTED,
                              self.clock.now, (command.run_id, len(self.sent)))

    def stop(self):
        self.stops += 1
        if self.stop_error:
            raise self.stop_error
        self.stop_required = False


class FakeMapper:
    def __init__(self):
        self.maps = 0

    def map(self, intent, current, previous):
        self.maps += 1
        return SimpleNamespace(arm_qpos=np.array(intent.arm), hand_qpos=np.array(intent.hand))

    def frozen_rejection(self, *args):
        return None

    def control_from_command(self, command, mode):
        return np.concatenate((command.arm_qpos, command.hand_qpos))


def make_runner(clock, *, mode="sync", recorder=None, results=None):
    from dexmani_real.deployment.runner import PolicyRunner

    runtime = ExperimentConfig()
    runtime = replace(runtime, execution=replace(
        runtime.execution, execution_mode=mode, action_steps=2,
        prefetch_steps=1, max_decision_age_s=5,
    ))
    info = SimpleNamespace(n_obs_steps=1, hand_history_steps=None, control_dt_s=.05,
                           observation_fields=("joint_state",), action_mode="joint", task_name="test")
    state = RuntimeState()
    state.arm()
    model, robot = FakeModel(clock), FakeRobot(clock, runtime)
    runner = PolicyRunner(
        state, runtime, info, robot=robot,
        needs=ObservationNeeds.for_policy(info.observation_fields, record_raw=recorder is not None),
        sensors=SimpleNamespace(channels=None), action_mapper=FakeMapper(),
        recorder=recorder, home_planner=None, model_runtime=model, kinematics=None,
        max_running_s=60, results=results,
    )
    runner.keyboard = FakeKeyboard()
    runner.read_observation = lambda **kw: observation(clock, runtime, camera=recorder is not None)
    return runner


def start_runner(runner):
    runner.request_start()
    runner.step()
    runner.model.complete()
    runner.step()
