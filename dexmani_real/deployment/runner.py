"""Async uses a fixed grid; sync preserves inference pauses and policy dt within each block."""

import logging
import time
from collections import deque
from concurrent.futures import CancelledError
from dataclasses import dataclass, field

import numpy as np

from dexmani_real.deployment.observation import (
    build_policy_observation,
    decision_is_fresh,
    policy_observation_issue,
    policy_sources,
)
from dexmani_real.planning.kinematics.ik import IKFailureKind
from dexmani_real.planning.kinematics.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.recording.recorder import (
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.robot.action import ActionIntent
from dexmani_real.robot.commands import (
    DispatchError,
    DispatchInterrupted,
    DispatchStatus,
    RobotCommand,
)
from dexmani_real.runtime.observation import (
    feedback_deadline_ns,
    observation_timing,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.episode_results import error_detail

logger = logging.getLogger(__name__)


@dataclass
class PolicyQuery:
    """slot anchors the observation; handoff_slot is the first consumable slot."""

    query_id: int
    run_id: int
    slot: int
    sources: dict[str, tuple[int, ...]]
    handoff_slot: int | None
    result: np.ndarray | None = None
    evidence: dict = field(default_factory=dict)


@dataclass
class ActionPlan:
    """actions index from anchor_slot; execution ends at exclusive segment_end."""

    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    sources: dict[str, tuple[int, ...]]
    frozen: dict[int, RobotCommand]
    query_id: int


@dataclass(frozen=True)
class ControlRejection:
    """A scheduling/target decision; only the episode owner applies termination."""

    detail: str
    run_end_reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY


class PolicyRunner:
    def __init__(
        self,
        runtime_state,
        runtime,
        policy_info,
        *,
        robot,
        realizer,
        recorder=None,
        operator,
        model_runtime,
        kinematics,
        max_running_s,
        num_episodes=20,
        camera_calibration=None,
        camera_parameters=None,
        results=None,
    ):
        self.results = results
        self.runtime_state = runtime_state
        self.robot = robot
        self.operator = operator
        self.runtime = runtime
        self.policy_info = policy_info
        self.model = model_runtime
        self.execution = runtime.execution
        self.kinematics = kinematics
        self.max_running_s = max_running_s
        self.num_episodes = num_episodes
        self.camera_calibration = camera_calibration
        self.camera_parameters = camera_parameters
        self.recorder = recorder
        self.history = deque(maxlen=policy_info.n_obs_steps)
        self.hand_history = (
            deque(maxlen=policy_info.hand_history_steps)
            if policy_info.hand_history_steps is not None else None
        )
        self.recording_started = False
        self.plan = None
        self.query = None
        self.query_id = 0
        self.slot_timing = {}
        self.termination_timing = None
        self.preparation_issue = None
        self.observation_read_issues = []
        self.wait_started_ns = None
        self.preparing_epoch = None
        self.reset_ready = False
        self.last_slot = -1
        self.dt_ns = int(policy_info.control_dt_s * 1e9)
        self.run_id = None
        self.started_ns = 0
        self.next_step_ns = 0
        self.previous_arm = None
        self.completed = 0
        self.realizer = realizer
        fields = set(self.policy_info.observation_fields)
        requires_rgb = "rgb" in fields
        self.requires_cloud = "point_cloud" in fields
        self.requires_camera_payload = (
            bool(fields & {"rgb", "camera_intrinsic", "camera_extrinsic"})
            or self.recorder is not None
        )
        self.requires_rgb_cloud_identity = (
            requires_rgb or self.recorder is not None
        ) and self.requires_cloud

    def read_observation(self, *, wait_for_hand=False):
        self.observation_read_issues.clear()
        return read_observation(
            self.runtime_state,
            self.runtime,
            self.robot,
            issues=self.observation_read_issues,
            require_hand=True,
            require_camera=self.requires_camera_payload,
            require_pointcloud=self.requires_cloud,
            require_rgb_cloud_identity=self.requires_rgb_cloud_identity,
            wait_for_hand=wait_for_hand,
            cancel_requested=lambda: not self.preparation_valid(),
        )

    def budget_limit(self):
        deadlines = []
        if self.run_id is not None:
            if self.max_running_s is not None:
                deadlines.append((self.started_ns + int(self.max_running_s * 1e9), "duration"))
            if self.wait_started_ns is not None:
                deadlines.append(
                    (self.wait_started_ns + int(self.execution.max_wait_s * 1e9), "wait")
                )
        # WAIT wins a tie: no action was supplied before its strict deadline.
        return min(deadlines, key=lambda item: (item[0], item[1] != "wait")) if deadlines else None

    def expired_budget(self):
        limit = self.budget_limit()
        return limit if limit is not None and time.monotonic_ns() >= limit[0] else None

    def finish_budget(self, limit, *, row=None):
        self.finish_episode(f"{limit[1]}_timeout", run_end_reason=RunEndReason.TIMEOUT, row=row)

    def has_motion_authority(self):
        return self.runtime_state.command_may_cross_sdk(run_id=self.run_id)

    def finish_episode(
        self,
        detail,
        *,
        run_end_reason=RunEndReason.EXECUTOR_BOUNDARY,
        errors=None,
        row=None,
        command=None,
        dispatch=None,
    ):
        if self.run_id is None:
            return
        epoch = self.run_id
        ended_ns = time.monotonic_ns()
        duration_s = (ended_ns - self.started_ns) / 1e9
        if self.recorder is not None:
            self.termination_timing = dict(
                run_id=epoch,
                started_ns=self.started_ns,
                slot=self.last_slot,
                ended_ns=ended_ns,
                reason=detail,
                query=dict(self.query.evidence) if self.query is not None else None,
                action_query_id=self.slot_timing.get("query_id"),
                prediction_index=self.slot_timing.get("prediction_index"),
            )
        self.runtime_state.revoke_motion(reason=run_end_reason, expected_run_id=epoch)
        self.operator.discard_pending_start()
        with self.runtime_state.motion_lock:
            self.operator.start_requested = False
            stopped_epoch = self.runtime_state.run_id
            reason = (
                self.runtime_state.run_ended_reason
                if self.runtime_state.run_ended_id == epoch
                else run_end_reason
            )
        self.plan = None
        if self.model.future is None:
            self.query = None
        self.history.clear()
        if self.hand_history is not None:
            self.hand_history.clear()
        self.previous_arm = None
        self.run_id = None
        self.wait_started_ns = None
        self.completed += 1
        errors, failure = list(errors or ()), None
        try:
            self.robot.stop()
            self.runtime_state.acknowledge_stop(stopped_epoch)
        except Exception as exc:
            failure = exc
            errors.append(error_detail("stop", exc))
        try:
            self.record(row, command, dispatch)
        except Exception as exc:
            failure = failure or exc
            errors.append(error_detail("recording", exc))
        try:
            self.finalize_outputs(reason.name.lower(), detail, errors, duration_s=duration_s)
        except Exception as exc:
            failure = failure or exc
        finally:
            logger.debug("policy episode %d ended: %s (%s)", self.completed, reason.name, detail)
            limit_reached = self.num_episodes is not None and self.completed >= self.num_episodes
            if limit_reached:
                self.runtime_state.request_quit()
        if failure is not None:
            raise failure
        if reason == RunEndReason.OPERATOR:
            description = "手动停止"
        elif reason == RunEndReason.QUIT:
            description = "退出"
        elif reason == RunEndReason.TIMEOUT and detail == "duration_timeout":
            description = "达到本轮时长上限"
        else:
            description = f"{reason.name}: {detail}"
        level = (
            logging.INFO
            if reason in (RunEndReason.OPERATOR, RunEndReason.QUIT)
            or (reason == RunEndReason.TIMEOUT and detail == "duration_timeout")
            else logging.WARNING
        )
        logger.log(level, "第 %d 轮结束：%s（%.1f s）。", self.completed, description, duration_s)
        if (
            reason in (RunEndReason.OPERATOR, RunEndReason.TIMEOUT, RunEndReason.EXECUTOR_BOUNDARY)
            and not self.runtime_state.error_state
            and not self.runtime_state.estop_request
            and self.runtime_state.running
        ):
            if limit_reached:
                print(f"已达到 {self.num_episodes} 轮上限，正在退出。", flush=True)
            elif not self.runtime_state.quit_requested:
                print("等待下一轮：整理场景后 H 回初始位 → B 开始；Q 退出。", flush=True)

    def finalize_outputs(self, reason, detail, errors=None, *, duration_s=0.0):
        errors = list(errors or ())
        failure, raw = None, None
        try:
            if self.recorder is not None and self.recording_started:
                saved, status = None, "empty"
                try:
                    saved = self.recorder.save_episode(
                        reason=reason,
                        details=[dict(stage="termination", reason=detail), *errors],
                        timing=self.termination_timing,
                        arm_execution=dict(
                            self.robot.arm.execution_snapshot(),
                            robot_stopped=not self.robot.stop_required,
                        ),
                    )
                    status = "published" if saved is not None else "empty"
                except BaseException as exc:
                    failure, status = exc, "failed"
                    errors.append(error_detail("recorder_finalize", exc))
                raw = dict(
                    status=status,
                    path=str(saved) if saved is not None else None,
                    staging_path=(
                        str(self.recorder.staging_path)
                        if self.recorder.staging_path is not None and saved is None
                        else None
                    ),
                    row_count=self.recorder.written_frames,
                )
            if self.results is not None:
                try:
                    self.results.finish_episode(
                        reason, details=detail, errors=errors, raw=raw, duration_s=duration_s
                    )
                except BaseException as exc:
                    failure = failure or exc
                    self.results.record_errors([error_detail("evaluation_save", exc)])
            if failure is not None:
                raise failure
        finally:
            self.recording_started = False

    def observation_issue(self, row):
        if row is None and self.observation_read_issues:
            issue = self.observation_read_issues[0]
            if (
                self.recorder is not None
                and issue.startswith("camera_")
                and not (
                    set(self.policy_info.observation_fields)
                    & {"rgb", "camera_intrinsic", "camera_extrinsic"}
                )
            ):
                return f"recording_source_{issue}"
            return f"policy_input_{issue}"
        return policy_observation_issue(row, self.policy_info.observation_fields)

    def preparation_valid(self):
        return self.runtime_state.command_may_cross_sdk(
            run_id=self.preparing_epoch,
            required_safety_state=SafetyState.ARMED,
        )

    def start_observation(self, *, wait_for_hand=False):
        row = self.read_observation(wait_for_hand=wait_for_hand)
        self.preparation_issue = self.observation_issue(row)
        if self.preparation_issue is not None:
            logger.warning("本轮未开始：%s", self.preparation_issue)
            return None
        if np.max(np.abs(row.arm["qpos"][0] - self.runtime.arm.home_qpos)) > np.deg2rad(
            self.execution.start_arm_home_tolerance_deg
        ) or np.max(
            np.abs(row.hand["qpos"][0] - np.deg2rad(self.runtime.hand.home_qpos_deg))
        ) > np.deg2rad(
            self.runtime.hand.home_tolerance_deg
        ):
            self.preparation_issue = "start_requires_home_pose"
            logger.warning("本轮未开始：机械臂和手需在初始位，请先按 H。")
            return None
        return row

    def begin_episode(self):
        if self.preparing_epoch is None:
            with self.runtime_state.motion_lock:
                if not self.operator.start_requested:
                    return
                self.operator.start_requested = False
                if self.model.future is not None:
                    return
                if not self.runtime_state.command_may_cross_sdk(
                    run_id=self.runtime_state.run_id,
                    required_safety_state=SafetyState.ARMED,
                ):
                    return
                epoch = self.runtime_state.run_id
            self.recording_started = False
            self.termination_timing = None
            self.preparation_issue = None
            self.preparing_epoch = epoch
            self.reset_ready = False
            if self.results is not None:
                self.results.begin_episode()
            try:
                self.robot.reset_hand_feedback()
                row = self.start_observation(wait_for_hand=True)
                if not self.preparation_valid():
                    raise CancelledError("cancelled during initial feedback acquisition")
            except BaseException as exc:
                # Finalize this preparation once; only explicit cancellation is consumed.
                self.preparing_epoch = None
                cancelled = isinstance(exc, (CancelledError, KeyboardInterrupt))
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                self.finalize_outputs(
                    "start_cancelled" if cancelled else "preparation_failed",
                    str(exc),
                    errors=[] if cancelled else [error_detail("preparation", exc)],
                )
                if isinstance(exc, CancelledError):
                    return
                raise
            if row is None:
                self.preparing_epoch = None
                self.finalize_outputs("preparation_failed", self.preparation_issue)
                return
            logger.info("第 %d 轮：准备开始…", self.completed + 1)
            self.model.submit("reset_episode")
            return
        if not self.reset_ready:
            return
        if not self.preparation_valid():
            self.preparing_epoch = None
            self.finalize_outputs("start_cancelled", "cancelled during model reset")
            return
        epoch, self.preparing_epoch = self.preparing_epoch, None
        committed = False
        start_failure = None
        try:
            if self.recorder is not None:
                metadata = snapshot_recording_metadata(
                    self.runtime_state,
                    camera_calibration=self.camera_calibration,
                    hand=self.robot.hand,
                )
                self.recording_started = True
                self.recorder.start_episode(
                    task_label=self.policy_info.task_name,
                    arm_execution=self.robot.arm.execution_snapshot(),
                    arm_control_metadata=self.robot.arm.execution_metadata(
                        1 / self.recorder.control_hz
                    ),
                    episode_name=f"episode_{self.completed + 1:03d}",
                    collection_source="policy_rollout",
                    metadata=metadata,
                )
            # Recorder startup can block; use a new row before granting motion.
            if self.start_observation() is None:
                return
            self.robot.prepare_streaming(epoch)
            # Mode restoration precedes the policy clock.
            if self.start_observation() is None:
                return
            with self.runtime_state.motion_lock:
                admitted = (
                    self.runtime_state.begin_motion()
                    if epoch == self.runtime_state.run_id
                    else None
                )
            if admitted is None:
                return
            self.run_id, self.started_ns = admitted
            committed = True
            if self.results is not None:
                self.results.entered(self.run_id)
            self.history.clear()
            if self.hand_history is not None:
                self.hand_history.clear()
            self.plan = self.query = None
            self.previous_arm = None
            self.last_slot = -1
            self.slot_timing = {}
            self.next_step_ns = self.started_ns
            self.wait_started_ns = self.started_ns
        except BaseException as exc:
            start_failure = exc
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            if isinstance(exc, DispatchError) and exc.revoked and exc.cause == "authority_revoked":
                return
            raise
        finally:
            if not committed:
                cancelled = isinstance(start_failure, KeyboardInterrupt) or (
                    isinstance(start_failure, DispatchError)
                    and start_failure.revoked
                    and start_failure.cause == "authority_revoked"
                )
                errors = [error_detail("preparation", start_failure)] if start_failure else []
                try:
                    if self.robot.stop_required:
                        self.robot.stop()
                except BaseException as exc:
                    errors.append(error_detail("preparation_stop", exc))
                    start_failure, cancelled = exc, False
                    raise
                finally:
                    try:
                        self.finalize_outputs(
                            (
                                "preparation_failed"
                                if self.preparation_issue or (start_failure and not cancelled)
                                else "start_cancelled"
                            ),
                            (
                                str(start_failure)
                                if start_failure
                                else self.preparation_issue or "final admission declined"
                            ),
                            errors=errors,
                        )
                    except Exception:
                        if start_failure is None or cancelled:
                            raise
                        logger.exception("start finalization also failed")

    def poll_model(self):
        item = self.model.poll()
        if item is None:
            return
        operation, result = item
        if operation == "reset_episode":
            if result.error is not None:
                raise result.error
            self.reset_ready = True
            return
        query = self.query
        if query is None:
            raise RuntimeError("Model result has no owned query")
        if result.query_context != dict(run_id=query.run_id, query_id=query.query_id):
            raise RuntimeError("model result/query ownership mismatch")
        if self.recorder is not None and query.run_id == self.run_id:
            query.evidence.update(
                started_ns=result.started_ns,
                completed_ns=result.completed_ns,
                status="failed" if result.error is not None else "completed",
                error=str(result.error) if result.error is not None else None,
            )
            self.record_query(query)
        if self.results is not None and query.run_id == self.run_id:
            self.results.record_inference(result.completed_ns - result.started_ns)
        if query.run_id != self.run_id or not self.has_motion_authority():
            logger.debug(
                "retired query %s from run %s reclaimed; error=%s",
                query.query_id,
                query.run_id,
                result.error,
            )
            self.query = None
            return
        if result.error is not None:
            raise result.error
        prediction = np.asarray(result.value)
        expected = (
            self.policy_info.horizon - self.policy_info.n_obs_steps + 1,
            self.policy_info.control_action_dim,
        )
        valid_prediction = (
            prediction.shape == expected
            and prediction.dtype.kind == "f"
            and np.isfinite(prediction).all()
        )
        if not valid_prediction:
            raise ValueError(f"Policy prediction requires finite floating point {expected}")
        query.result = prediction

    def submit(self, rows, slot, *, handoff_slot=None, prefix=None, commands=None):
        observation = build_policy_observation(
            rows,
            self.policy_info,
            kinematics=self.kinematics,
            camera_parameters=self.camera_parameters,
        )
        now = time.monotonic_ns()
        authorized, expired = self.has_motion_authority(), self.expired_budget()
        if expired is not None:
            return ControlRejection(f"{expired[1]}_timeout", RunEndReason.TIMEOUT)
        if not authorized:
            return ControlRejection("query_authority_revoked")
        sources = policy_sources(rows[-1], self.policy_info.observation_fields)
        if not decision_is_fresh(sources, now, self.execution.max_decision_age_s):
            return ControlRejection("query_sources_invalid_or_expired")
        if prefix is not None and not decision_is_fresh(
            self.plan.sources, now, self.execution.max_decision_age_s
        ):
            return ControlRejection("prefix_decision_expired")
        query = PolicyQuery(self.query_id + 1, self.run_id, slot, sources, handoff_slot)
        kwargs = (
            dict(rtc_prefix=prefix, delay_steps=handoff_slot - slot) if prefix is not None else {}
        )
        if self.hand_history is not None:
            if not self.hand_history:
                # Match training's episode-initial measured pose, before the first query action.
                hand = np.asarray(observation["joint_state"][-1, -12:], dtype=np.float64)
                if hand.shape != (12,) or not np.isfinite(hand).all():
                    return ControlRejection("invalid_hand_history_bootstrap")
                self.hand_history.extend(hand.copy() for _ in range(self.hand_history.maxlen))
            # The model thread owns this snapshot; later appends/reset cannot change it.
            kwargs["action_context"] = {"hand_history": np.stack(self.hand_history)}
        if self.recorder is not None:
            query.evidence = dict(
                query_id=query.query_id,
                run_id=self.run_id,
                slot=int(slot),
                handoff_slot=handoff_slot,
                deadline_ns=(
                    (self.started_ns + (handoff_slot + 1) * self.dt_ns)
                    if handoff_slot is not None
                    else None
                ),
                submitted_ns=time.monotonic_ns(),
                started_ns=None,
                completed_ns=None,
                status="submitted",
                bootstrap=handoff_slot is None,
                observations=[
                    dict(
                        observation_timing(row),
                        policy_sources=policy_sources(row, self.policy_info.observation_fields),
                    )
                    for row in rows
                ],
            )
        self.model.submit(
            "predict",
            observation,
            query_context=dict(run_id=self.run_id, query_id=query.query_id),
            **kwargs,
        )
        self.query_id, self.query = query.query_id, query
        if commands is not None:
            self.plan.frozen.update(commands)
        self.record_query(query)
        return None

    def prepare_prefix(self, row, slot):
        """Prepare the whole guidance tail and d physical commands without committing either."""
        start = slot - self.plan.anchor_slot
        prefix = self.plan.actions[start:].copy()
        commands, previous = {}, self.previous_arm
        for index in range(self.execution.prefetch_steps):
            self.operator.poll()
            if not self.has_motion_authority():
                return None, None, ControlRejection("prefix_authority_revoked")
            expired = self.expired_budget()
            if expired is not None:
                return None, None, ControlRejection(f"{expired[1]}_timeout", RunEndReason.TIMEOUT)
            command, issue = self.realize_command(
                prefix[index], row, previous, context="prefix_action"
            )
            if issue is not None:
                return None, None, issue
            commands[slot + index] = command
            previous = command.arm_qpos
            prefix[index] = self.realizer.control_from_command(
                command, self.policy_info.action_mode
            )
        return prefix, commands, None

    def record_query(self, query):
        if self.recorder is not None and self.recording_started:
            self.recorder.record_query(query.evidence)

    def record(self, row, command=None, result=None):
        if self.recorder is not None and row is not None:
            self.recorder.add_frame(row, command, result, timing=self.slot_timing)

    def realize_command(self, action, row, previous, *, context="action"):
        """Decode a row of the admitted physical future and apply shared target admission."""
        action = np.asarray(action, dtype=np.float64)
        mode = self.policy_info.action_mode
        intent = (
            ActionIntent(mode, action[:7], action[7:])
            if mode == "joint"
            else ActionIntent(mode, Pose(action[:3], rot6d_to_quat_wxyz(action[3:9])), action[9:])
        )
        decoded = self.realizer.realize(intent, row.arm["qpos"][0], previous)
        if decoded.arm_qpos is None:
            if (
                decoded.ik_result is not None
                and decoded.ik_result.failure_kind == IKFailureKind.INVALID_OUTPUT
            ):
                raise RuntimeError(f"online IK technical failure: {decoded.ik_result.reason}")
            return None, ControlRejection(f"{context}_rejected: {decoded.rejection_reason}")
        return RobotCommand(self.run_id, decoded.arm_qpos, decoded.hand_qpos), None

    def command_for_slot(self, row, slot):
        plan = self.plan
        command = plan.frozen.get(slot)
        if command is not None:
            rejection = self.realizer.frozen_rejection(
                command, row.arm["qpos"][0], self.previous_arm, self.policy_info.action_mode
            )
            if rejection is not None:
                return None, ControlRejection(f"frozen_target_rejected: {rejection}")
        else:
            return self.realize_command(
                plan.actions[slot - plan.anchor_slot],
                row,
                self.previous_arm,
            )
        return command, None

    def execute_slot(self, row, slot):
        plan = self.plan
        self.slot_timing.update(query_id=plan.query_id, prediction_index=slot - plan.anchor_slot)
        command, issue = self.command_for_slot(row, slot)
        if issue is not None:
            self.finish_episode(issue.detail, run_end_reason=issue.run_end_reason, row=row)
            return
        if not self.has_motion_authority():
            self.finish_episode("before_dispatch_revoked", row=row)
            return
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired, row=row)
            return
        budget_limit = self.budget_limit()
        decision_deadline_ns = min(
            stamp + int(self.execution.max_decision_age_s * 1e9) + 1
            for times in plan.sources.values()
            for stamp in times
        )
        action_deadline_ns = min(
            self.next_step_ns,
            feedback_deadline_ns(row, self.runtime),
            decision_deadline_ns,
        )
        valid_until_ns = (
            min(action_deadline_ns, budget_limit[0]) if budget_limit else action_deadline_ns
        )
        if time.monotonic_ns() >= valid_until_ns:
            self.finish_episode("dispatch_deadline_expired", row=row)
            return
        failure = None
        try:
            self.slot_timing["submit_started_ns"] = time.monotonic_ns()
            execute_until_ns = decision_deadline_ns
            if self.max_running_s is not None:
                execute_until_ns = min(
                    execute_until_ns, self.started_ns + int(self.max_running_s * 1e9)
                )
            result = self.robot.send_action(
                command, valid_until_ns=valid_until_ns, execute_until_ns=execute_until_ns
            )
            if result.arm != DispatchStatus.SUBMITTED or result.hand not in (
                DispatchStatus.ACCEPTED, DispatchStatus.CRC_UNCONFIRMED
            ):
                raise DispatchError("incomplete action dispatch", result, cause="dispatch_status")
        except (DispatchError, DispatchInterrupted) as exc:
            result, failure = exc.result, exc
        finally:
            self.slot_timing["submit_completed_ns"] = time.monotonic_ns()
        if self.results is not None and (
            result.arm != DispatchStatus.NOT_CALLED or result.hand != DispatchStatus.NOT_CALLED
        ):
            self.results.record_dispatch()
        now = time.monotonic_ns()
        expired = budget_limit if budget_limit is not None and now >= budget_limit[0] else None
        late = now >= action_deadline_ns
        cancelled = isinstance(failure, DispatchInterrupted)
        cause = failure.cause if isinstance(failure, DispatchError) else None
        if failure is not None or expired is not None or late or not self.has_motion_authority():
            if cancelled:
                self.runtime_state.request_emergency_stop()
            hardware_failure = (
                not cancelled
                and failure is not None
                and (not failure.revoked or cause not in {"authority_revoked", "deadline_expired"})
            )
            budget_timeout = expired is not None and budget_limit[0] < action_deadline_ns
            if cancelled:
                reason, detail = RunEndReason.ESTOP, "dispatch_cancelled"
            elif hardware_failure:
                reason, detail = (
                    RunEndReason.HARDWARE_FAULT,
                    f"dispatch_failed: {cause or str(failure)}",
                )
            elif budget_timeout:
                reason = RunEndReason.TIMEOUT
                detail = (
                    f"dispatch_failed: {cause or str(failure)}"
                    if failure is not None
                    else f"dispatch_return_{expired[1]}_timeout"
                )
            else:
                reason = RunEndReason.EXECUTOR_BOUNDARY
                if failure is not None:
                    detail = f"dispatch_failed: {cause or str(failure)}"
                elif late:
                    detail = "dispatch_return_deadline_expired"
                else:
                    detail = "dispatch_authority_revoked"
            errors = [error_detail("dispatch", failure)] if failure is not None else []
            try:
                self.finish_episode(
                    detail,
                    run_end_reason=reason,
                    errors=errors,
                    row=row,
                    command=command,
                    dispatch=result,
                )
            except Exception as exc:
                if hardware_failure or cancelled:
                    raise failure from exc
                raise
            if hardware_failure or cancelled:
                raise failure
            return
        self.previous_arm = command.arm_qpos
        if self.hand_history is not None:
            # Ordinary-send CRC is admitted but unconfirmed, just as in canonical actions.
            self.hand_history.append(command.hand_qpos.copy())
        self.wait_started_ns = None
        self.record(row, command, result)
        if slot + 1 == plan.segment_end and self.execution.execution_mode == "sync":
            self.plan = None

    def handoff(self, slot):
        query = self.query
        if query is None or query.run_id != self.run_id:
            return None, None
        bootstrap = query.handoff_slot is None
        handoff = slot if bootstrap and query.result is not None else query.handoff_slot
        if handoff is None or slot < handoff:
            return None, None
        if not self.has_motion_authority():
            return None, ControlRejection("handoff_authority_revoked")
        if slot != handoff:
            return None, ControlRejection("handoff_slot_missed")
        if query.result is None:
            return None, ControlRejection("handoff_result_not_ready")
        return (
            ActionPlan(
                actions=query.result,
                anchor_slot=slot if bootstrap else query.slot,
                segment_end=slot + self.execution.action_steps,
                sources=query.sources,
                frozen={},
                query_id=query.query_id,
            ),
            None,
        )

    def prefetch(self, row, rows, slot):
        if (
            self.execution.execution_mode == "sync"
            or slot != self.plan.segment_end - self.execution.prefetch_steps
        ):
            return None
        if not rows or self.query is not None or self.model.future is not None:
            return ControlRejection("prefetch_unavailable")
        prefix = commands = None
        if self.execution.execution_mode == "rtc":
            prefix, commands, issue = self.prepare_prefix(row, slot)
            if issue is not None:
                return issue
        return self.submit(
            rows, slot, handoff_slot=self.plan.segment_end, prefix=prefix, commands=commands
        )

    def step(self):
        if self.run_id is not None and not self.has_motion_authority():
            self.finish_episode("owner_authority_revoked")
            return
        self.poll_model()
        if self.recorder is not None:
            self.recorder.check_error()
        if self.run_id is None:
            if self.preparing_epoch is None and not self.robot.stop_required:
                self.runtime_state.acknowledge_stop(self.runtime_state.run_id)
            if self.preparing_epoch is not None or self.operator.start_requested:
                self.begin_episode()
            if self.run_id is None:
                return
        if (expired := self.expired_budget()) is not None:
            self.finish_budget(expired)
            return
        now = time.monotonic_ns()
        synchronous = self.execution.execution_mode == "sync"
        if (
            synchronous
            and self.plan is None
            and self.query is not None
            and self.query.result is not None
        ):
            # Resume without an extra grid wait, using a new physical observation.
            self.next_step_ns = min(self.next_step_ns, now)
        if now < self.next_step_ns:
            return
        slot = self.last_slot + 1
        skipped = (now - self.next_step_ns) // self.dt_ns
        if skipped:
            if self.results is not None:
                self.results.record_skipped_slots(skipped)
            self.finish_episode(f"missed_slot: expected={slot}, actual={slot + skipped}")
            return
        self.slot_timing = dict(run_id=self.run_id, slot=int(slot), control_ns=now)
        self.last_slot = slot
        self.next_step_ns = (now if synchronous else self.next_step_ns) + self.dt_ns
        row = None
        try:
            row = self.read_observation()
            self.poll_model()
            if not self.has_motion_authority():
                self.finish_episode("observation_authority_revoked", row=row)
                return
            issue = self.observation_issue(row)
            if issue is not None:
                self.finish_episode(issue, row=row)
                return
            self.history.append(row)
            rows = tuple(self.history) if len(self.history) == self.history.maxlen else ()
            new_plan, issue = self.handoff(slot)
            if issue is not None:
                self.finish_episode(issue.detail, run_end_reason=issue.run_end_reason, row=row)
                return
            if new_plan is not None:
                if self.query.handoff_slot is None and self.recorder is not None:
                    self.query.evidence.update(handoff_slot=int(slot), bootstrap=True)
                    self.record_query(self.query)
                self.plan, self.query = new_plan, None
            if self.plan is None:
                self.robot.end_hand_send_interval()
                if self.wait_started_ns is None:
                    self.wait_started_ns = now
                    self.robot.arm.limit_execution_until(now + int(self.execution.max_wait_s * 1e9))
                if rows and self.query is None and self.model.future is None:
                    issue = self.submit(rows, slot)
                    if issue is not None:
                        self.finish_episode(
                            issue.detail, run_end_reason=issue.run_end_reason, row=row
                        )
                        return
                self.record(row)
            else:
                issue = self.prefetch(row, rows, slot)
                if issue is not None:
                    self.finish_episode(issue.detail, run_end_reason=issue.run_end_reason, row=row)
                    return
                if slot >= self.plan.segment_end:
                    self.finish_episode("plan_exhausted_without_handoff", row=row)
                    return
                # Pending prefetch never suspends a still-valid current plan.
                self.execute_slot(row, slot)
            if (expired := self.expired_budget()) is not None:
                self.finish_budget(expired)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
                reason = RunEndReason.ESTOP
            elif isinstance(exc, RecordingError):
                reason = RunEndReason.RECORDING_FAILURE
            elif isinstance(exc, DispatchError):
                reason = RunEndReason.HARDWARE_FAULT
            else:
                reason = RunEndReason.POLICY_FAILURE
            try:
                self.finish_episode(
                    f"control_failed: {type(exc).__name__}: {exc}",
                    # A failed writer submission must never be retried.
                    row=None if isinstance(exc, RecordingError) else row,
                    run_end_reason=reason,
                    errors=[error_detail("control", exc)],
                )
            except Exception:
                logger.exception("control failure cleanup also failed")
            raise

    def run(self):
        failure = None
        reason = RunEndReason.RUNTIME_SHUTDOWN
        try:
            while self.runtime_state.running and not self.runtime_state.quit_requested:
                self.robot.arm.check_error()
                self.operator.poll(
                    allow_tactile_reset=(
                        self.run_id is None
                        and self.preparing_epoch is None
                        and self.model.future is None
                        and (self.recorder is None or not self.recorder.is_recording)
                    )
                )
                if self.runtime_state.estop_request or self.runtime_state.error_state:
                    break
                if self.run_id is None:
                    self.robot.service_idle()
                self.step()
                time.sleep(0.001 if self.run_id is not None else 0.005)
        except KeyboardInterrupt as exc:
            self.runtime_state.request_emergency_stop()
            failure = exc
            reason = RunEndReason.ESTOP
        except Exception as exc:
            failure = exc
            if isinstance(exc, RecordingError):
                reason = RunEndReason.RECORDING_FAILURE
            elif isinstance(exc, DispatchError):
                reason = RunEndReason.HARDWARE_FAULT
            else:
                reason = RunEndReason.POLICY_FAILURE
            self.runtime_state.mark_error()
            self.runtime_state.end_session(reason=reason)
        finally:
            try:
                self.finish_episode(str(failure) if failure else "shutdown", run_end_reason=reason)
            except Exception as exc:
                failure = failure or exc
                logger.exception("episode cleanup failed")
            try:
                if self.robot.stop_required:
                    self.robot.stop()
            except Exception as exc:
                failure = failure or exc
                logger.exception(
                    "policy shutdown stop failed before remaining artifact finalization"
                )
            try:
                if self.recording_started or (
                    self.results is not None and self.results.current is not None
                ):
                    self.finalize_outputs(
                        "start_cancelled", str(failure) if failure else "shutdown"
                    )
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure
