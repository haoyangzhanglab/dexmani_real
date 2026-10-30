"""机械臂 SDK 与伺服循环；阻塞的原生调用无法在此中断。"""

import logging
import time
from typing import cast

import numpy as np

from dexmani_real.config.xarm7 import (
    HOME_CONVERGENCE_TIMEOUT_S,
    HOME_DWELL_S,
)
from dexmani_real.control.homing import home_segment_timeout_s
from dexmani_real.robot.commands import OperationCancelled, validate_arm_target
from dexmani_real.robot.xarm7.protocol import (
    STATUS_DTYPE,
    HomeRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory

logger = logging.getLogger(__name__)


class ArmController:
    def __init__(self, cfg, io, pipe, driver):
        self.cfg, self.io, self.pipe, self.driver = cfg, io, pipe, driver
        self.status = np.zeros((), dtype=STATUS_DTYPE)
        self.active = self.reference = None

    def authorized(self, run_id):
        return (
            self.io.run_id.value == run_id
            and not self.io.fault.value
            and not self.io.closing.is_set()
            and not self.io.stop.is_set()
            and not self.io.emergency.is_set()
        )

    def fault(self, reason):
        if not self.io.fault.value:
            self.io.fault.value = f"{type(reason).__name__}: {reason}".encode()[:1023]
            # traceback 不能经共享字符串/普通 pickle 传递，在 SDK 所有者处保留。
            logger.error("arm controller failed", exc_info=(type(reason), reason, reason.__traceback__))
        self.io.stop.set()

    def read_feedback(self):
        started = time.monotonic_ns()
        try:
            q, v = self.driver.read()
            completed = time.monotonic_ns()
            if completed - started >= int(self.cfg.feedback_max_age_s * 1e9):
                raise TimeoutError("arm query exceeded feedback freshness budget")
            self.status["frame"]["qpos"] = q
            self.status["frame"]["qvel"] = v
            # 来源时间采用查询开始时刻，不能替换为缓存读取时间。
            self.status["frame"]["timestamp_ns"] = started
        except Exception:
            self.status["frame"]["timestamp_ns"] = 0
            raise
        finally:
            self.io.publish_status(self.status)

    def wait_settled(
        self,
        check_abort,
        *,
        stage,
        timeout_s,
        target=None,
        tolerance_rad=None,
        strict=False,
    ):
        """确认位置与速度连续稳定，strict 模式遇到不合格样本立即失败。"""
        tolerance = self.cfg.homing.convergence_rad if tolerance_rad is None else tolerance_rad
        interval_ns = round(1e9 / self.cfg.feedback_hz)
        deadline = time.monotonic_ns() + int(timeout_s * 1e9)
        stable = None
        anchor = None
        while time.monotonic_ns() < deadline:
            check_abort()
            self.read_feedback()
            check_abort()
            completed_ns = time.monotonic_ns()
            if completed_ns >= deadline:
                break
            q, v = self.status["frame"]["qpos"], self.status["frame"]["qvel"]
            sample_ns = int(self.status["frame"]["timestamp_ns"])
            valid = np.max(np.abs(v)) <= self.cfg.homing.velocity_convergence_rad_s
            if target is not None:
                valid = valid and np.max(np.abs(q - target)) <= tolerance
            if strict and not valid:
                raise RuntimeError(f"{stage} interrupted by position/velocity")
            if valid:
                if target is None and (anchor is None or np.max(np.abs(q - anchor)) > tolerance):
                    stable, anchor = None, q.copy()
                # SDK 未提供采样时刻；首尾查询的等待时间不能算作已观测到的停稳。
                stable = completed_ns if stable is None else stable
                if sample_ns - stable >= int(HOME_DWELL_S * 1e9):
                    return
            else:
                stable = anchor = None
            # 按查询开始时刻限频，查询耗时既不拉长采样间隔，也不充作停稳时长。
            remaining_ns = min(sample_ns + interval_ns, deadline) - time.monotonic_ns()
            self.io.stop.wait(max(0, remaining_ns) / 1e9)
        raise TimeoutError(f"{stage}: measured position/velocity dwell not confirmed")

    def wait_stationary(self, check_abort):
        self.wait_settled(
            check_abort,
            stage="arm standstill",
            timeout_s=HOME_CONVERGENCE_TIMEOUT_S,
        )

    def home(self, waypoints, final_qpos, *, check_abort):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        lower, upper = (
            np.asarray(self.cfg.joint_limit_lower),
            np.asarray(self.cfg.joint_limit_upper),
        )
        if waypoints.ndim != 2 or waypoints.shape[1:] != (7,) or not np.isfinite(waypoints).all():
            raise ValueError("invalid home waypoint array")
        final_qpos = validate_arm_target(final_qpos, lower, upper)
        if np.any(waypoints < lower) or np.any(waypoints > upper):
            raise ValueError("home waypoint violates joint limits")
        if len(waypoints) and np.max(np.abs(waypoints[-1] - final_qpos)) > 1e-6:
            raise ValueError("final milestone does not match canonical home")
        check_abort()
        self.read_feedback()
        check_abort()
        current = self.status["frame"]["qpos"].copy()
        velocity = self.status["frame"]["qvel"]
        if len(waypoints) == 0:
            if (
                np.max(np.abs(current - final_qpos)) > self.cfg.homing.convergence_rad
                or np.max(np.abs(velocity)) > self.cfg.homing.velocity_convergence_rad_s
            ):
                raise RuntimeError("empty home path while away from canonical home")
        elif np.max(np.abs(current - waypoints[0])) > self.cfg.homing.convergence_rad:
            raise RuntimeError("current state moved too far from planned path start")
        targets = waypoints[1:]
        if len(targets):
            check_abort()
            self.driver.enter_mode(0, check_abort=check_abort)
            for index, target in enumerate(targets, 1):
                check_abort()
                self.driver.move_home_milestone(target)
                check_abort()
                self.wait_settled(
                    check_abort,
                    stage=f"home milestone {index}",
                    timeout_s=home_segment_timeout_s(float(np.max(np.abs(target - current))), self.cfg),
                    target=target,
                    tolerance_rad=min(self.cfg.homing.convergence_rad, np.deg2rad(0.5)),
                )
                current = self.status["frame"]["qpos"].copy()
            if np.max(np.abs(current - final_qpos)) > self.cfg.homing.convergence_rad:
                raise RuntimeError("home final target error")
        else:
            self.wait_settled(
                check_abort,
                stage="home dwell",
                strict=True,
                target=final_qpos,
                timeout_s=HOME_CONVERGENCE_TIMEOUT_S,
            )
        check_abort()

    def stop(self):
        self.io.stop.set()
        self.status["prepared_run_id"] = 0
        self.active = self.reference = None
        self.io.take_goal(block=True)
        self.io.publish_status(self.status)
        try:
            self.driver.stop()
        except BaseException as exc:
            if not self.io.close_error.value:
                self.io.close_error.value = f"stop SDK failure: {exc}".encode()[:1023]
            if self.io.fault.value:
                logger.exception("arm stop also failed")
            self.fault(exc)
        finally:
            self.io.stopped.set()  # 处理结束；停止失败仍保存在 close_error。

    def operation(self, message):
        kind = "home" if isinstance(message, HomeRequest) else "prepare"
        error = None
        try:
            run_id = message.run_id

            def check_abort():
                if not self.authorized(run_id):
                    raise OperationCancelled(f"{kind} authority revoked")
                if isinstance(message, HomeRequest) and time.monotonic_ns() >= message.deadline_ns:
                    raise TimeoutError("HOME execution deadline expired")

            check_abort()
            if self.active is not None:
                raise RuntimeError("prepare/HOME requires stopped streaming")
            self.io.take_goal(block=True)
            self.status["prepared_run_id"] = 0
            self.io.stopped.clear()
            if isinstance(message, HomeRequest):
                self.wait_stationary(check_abort)
                self.home(message.waypoints, message.target, check_abort=check_abort)
            else:
                self.driver.enter_mode(1, check_abort=check_abort)
                self.wait_stationary(check_abort)
                check_abort()
                self.status["prepared_run_id"] = run_id
        except BaseException as exc:
            error = exc
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            # 无需等待可能阻塞在其他设备 SDK 内的父进程。
            self.stop()

        self.io.publish_status(self.status)
        self.pipe.send(error)

    def command_issue(self, goal, now, *, first=False):
        if not self.authorized(int(goal["run_id"])):
            return "revoked"
        if first and now >= goal["submit_until_ns"]:
            return "submission_expired_before_sdk"
        if self.status["prepared_run_id"] != goal["run_id"]:
            return "unprepared_run"
        if now >= goal["execute_until_ns"]:
            return "command_expired"
        stamp = int(self.status["frame"]["timestamp_ns"])
        if not 0 < stamp <= now < stamp + int(self.cfg.feedback_max_age_s * 1e9):
            return "arm_feedback_expired"
        return None

    def tick(self) -> int | None:
        pending = self.io.take_goal()
        goal = pending if pending is not None else self.active
        if goal is None:
            return None
        if not self.authorized(int(goal["run_id"])):
            self.io.stop.set()
            return None
        if pending is not None:
            if self.active is None:
                q, v = self.status["frame"]["qpos"], self.status["frame"]["qvel"]
                if np.max(np.abs(v)) > self.cfg.homing.velocity_convergence_rad_s:
                    raise RuntimeError("arm moved after standstill preparation")
                self.reference = ArmTrajectory(self.cfg, q, v)
            self.active = pending
        q = cast(ArmTrajectory, self.reference).step(
            pending["q_goal"] if pending is not None and not pending["hold"] else None,
            hold=bool(pending is not None and pending["hold"]),
        )
        # 计算结束后、进入 SDK 前统一检查期限、反馈和最新控制器上报。
        issue = self.command_issue(goal, time.monotonic_ns(), first=pending is not None)
        if issue:
            if issue != "revoked":
                raise RuntimeError(issue)
            self.io.stop.set()
            return None
        started = time.monotonic_ns()
        self.driver.servo(q)
        return started

    def run(self):
        try:
            run_id = self.io.run_id.value

            def startup_abort():
                if not self.authorized(run_id):
                    raise OperationCancelled("arm startup authority revoked")

            self.driver.connect(check_abort=startup_abort)
            self.io.connected.set()
            self.wait_stationary(startup_abort)
            self.driver.initialize(check_abort=startup_abort)
            self.read_feedback()
            self.io.ready.set()
            period = round(1e9 / self.cfg.servo_hz)
            tick = time.monotonic_ns() + period
            feedback_due = time.monotonic_ns() + round(1e9 / self.cfg.feedback_hz)
            while not self.io.closing.is_set() and not self.io.fault.value:
                if self.io.stop.is_set() and not self.io.stopped.is_set():
                    self.stop()
                    continue
                if self.pipe.poll():
                    self.operation(self.pipe.recv())
                    tick = time.monotonic_ns() + period
                    feedback_due = time.monotonic_ns()
                    continue
                now = time.monotonic_ns()
                if now >= tick:
                    sent_ns = self.tick()
                    # 从实际发送时刻限频；晚发后不追赶旧时钟网格。
                    tick = (sent_ns if sent_ns is not None else time.monotonic_ns()) + period
                if self.active is not None and not self.authorized(int(self.active["run_id"])):
                    self.io.stop.set()
                if time.monotonic_ns() >= feedback_due:
                    self.read_feedback()
                    feedback_due = time.monotonic_ns() + round(1e9 / self.cfg.feedback_hz)
                # 停止可中断等待，目标仅在计划的伺服时刻生效。
                delay = max(0, min(tick, feedback_due) - time.monotonic_ns()) / 1e9
                if self.io.stopped.is_set():
                    self.pipe.poll(delay)
                else:
                    self.io.stop.wait(delay)
        except BaseException as exc:
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
        finally:
            self.io.ready.set()
            if not self.io.stopped.is_set():
                self.stop()
            try:
                self.driver.close()
            except BaseException as exc:
                logger.exception("arm disconnect failed")
                previous = self.io.close_error.value.decode(errors="replace")
                detail = f"disconnect failure: {exc}"
                self.io.close_error.value = (
                    f"{previous}; {detail}" if previous else detail
                ).encode()[:1023]
            self.io.connected.clear()
            self.pipe.close()


def run_arm(cfg, io, pipe):
    from dexmani_real.robot.xarm7.driver import XArm7

    ArmController(cfg, io, pipe, XArm7(cfg)).run()
