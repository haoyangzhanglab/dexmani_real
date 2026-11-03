"""机械臂 SDK 与伺服循环；阻塞的原生调用无法在此中断。"""

import logging
import time
from typing import cast

import numpy as np

from dexmani_real.config.xarm7 import (
    HOME_CONVERGENCE_TIMEOUT_S,
    HOME_DWELL_S,
)
from dexmani_real.robot.commands import OperationCancelled, validate_arm_target
from dexmani_real.robot.xarm7.protocol import (
    ARM_STATE_DTYPE,
    HomeRequest,
    PrepareRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import next_tick_ns, sample_is_fresh, sampling_clock_ns

logger = logging.getLogger(__name__)


class ArmController:
    def __init__(self, cfg, io, pipe, driver):
        self.cfg, self.io, self.pipe, self.driver = cfg, io, pipe, driver
        self.status = np.zeros((), dtype=ARM_STATE_DTYPE)
        self.active = self.reference = None

    def cancelled(self):
        return (self.io.stop.is_set() or self.io.emergency.is_set()
                or self.io.closing.is_set() or bool(self.io.fault.value))

    def check_lifecycle(self, *, deadline_ns, stage):
        if self.pipe.poll():
            try:
                self.pipe.recv()
            except EOFError as exc:
                self.io.closing.set()
                raise OperationCancelled("parent pipe closed") from exc
            raise RuntimeError("overlapping arm lifecycle requests")
        if self.cancelled():
            raise OperationCancelled(f"{stage} cancelled")
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"{stage} execution deadline expired")

    def fault(self, reason):
        if isinstance(reason, KeyboardInterrupt):
            self.io.emergency.set()
        self.io.stop.set()
        if not self.io.fault.value:
            self.io.fault.value = f"{type(reason).__name__}: {reason}".encode()[:1023]
            # traceback 不能经共享字符串/普通 pickle 传递，在 SDK 所有者处保留。
            logger.error("arm controller failed", exc_info=(type(reason), reason, reason.__traceback__))

    def read_feedback(self):
        started = time.monotonic_ns()
        try:
            q, v = self.driver.read()
            self.status["qpos"] = q
            self.status["qvel"] = v
            # 来源时间采用查询开始时刻，不能替换为缓存读取时间。
            self.status["timestamp_ns"] = started
        except Exception:
            self.status["timestamp_ns"] = 0
            raise
        finally:
            self.io.publish_status(self.status)

    def wait_settled(
        self,
        check_abort,
        *,
        stage,
        deadline_ns,
        target=None,
        strict=False,
    ):
        """在绝对期限内确认连续停稳；strict 遇到位置或速度不合格立即失败。"""
        tolerance = self.cfg.homing.convergence_rad
        interval_ns = round(1e9 / self.cfg.feedback_hz)
        stable = None
        anchor = None
        while time.monotonic_ns() < deadline_ns:
            check_abort()
            self.read_feedback()
            check_abort()
            completed_ns = time.monotonic_ns()
            if completed_ns >= deadline_ns:
                break
            q, v = self.status["qpos"], self.status["qvel"]
            sample_ns = int(self.status["timestamp_ns"])
            if not sample_is_fresh(sample_ns, self.cfg.feedback_max_age_s, completed_ns):
                raise TimeoutError(f"{stage}: current arm feedback unavailable/stale")
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
            remaining_ns = min(sample_ns + interval_ns, deadline_ns) - time.monotonic_ns()
            self.io.stop.wait(max(0, remaining_ns) / 1e9)
        raise TimeoutError(f"{stage}: measured position/velocity dwell not confirmed")

    def home(self, waypoints, *, check_abort, deadline_ns):
        waypoints = np.asarray(waypoints, dtype=np.float64)
        lower, upper = (
            np.asarray(self.cfg.joint_limit_lower),
            np.asarray(self.cfg.joint_limit_upper),
        )
        if waypoints.shape not in {(0, 7), (2, 7)} or not np.isfinite(waypoints).all():
            raise ValueError("HOME requires an empty path or current/home endpoints")
        final_qpos = validate_arm_target(self.cfg.home_qpos, lower, upper)
        if np.any(waypoints < lower) or np.any(waypoints > upper):
            raise ValueError("home waypoint violates joint limits")
        if len(waypoints) and np.max(np.abs(waypoints[-1] - final_qpos)) > 1e-6:
            raise ValueError("home path endpoint does not match canonical home")
        check_abort()
        if len(waypoints):
            self.driver.enter_mode(0, check_abort=check_abort)
        # 模式操作可能阻塞；紧接着才取得并裁决真正用于起点检查的反馈。
        self.read_feedback()
        check_abort()
        now = time.monotonic_ns()
        stamp = int(self.status["timestamp_ns"])
        if not sample_is_fresh(stamp, self.cfg.feedback_max_age_s, now):
            raise TimeoutError("HOME start feedback unavailable/stale")
        current = self.status["qpos"]
        velocity = self.status["qvel"]
        if np.max(np.abs(velocity)) > self.cfg.homing.velocity_convergence_rad_s:
            raise RuntimeError("arm moved during HOME preparation")
        if len(waypoints) == 0:
            if np.max(np.abs(current - final_qpos)) > self.cfg.homing.convergence_rad:
                raise RuntimeError("empty home path while away from canonical home")
        elif np.max(np.abs(current - waypoints[0])) > self.cfg.homing.convergence_rad:
            raise RuntimeError("current state moved too far from planned path start")
        if len(waypoints):
            check_abort()
            self.driver.move_home(final_qpos)
            check_abort()
            self.wait_settled(
                check_abort,
                stage="arm HOME",
                deadline_ns=deadline_ns,
                target=final_qpos,
            )
        else:
            self.wait_settled(
                check_abort,
                stage="home dwell",
                strict=True,
                target=final_qpos,
                deadline_ns=deadline_ns,
            )
        check_abort()

    def stop(self):
        self.io.stop.set()
        self.active = self.reference = None
        try:
            self.driver.stop()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.io.emergency.set()
                self.io.stop.set()
            if self.io.fault.value:
                self.io.close_error.value = f"stop SDK failure: {exc}".encode()[:1023]
                logger.exception("arm stop also failed")
            self.fault(exc)
        finally:
            self.io.publish_status(self.status)
            self.io.stopped.set()  # 处理结束；停止失败仍保存在 fault/close_error。

    def operation(self, message: PrepareRequest | HomeRequest):
        self.io.stopped.clear()
        kind = "home" if isinstance(message, HomeRequest) else "prepare"
        error = None
        try:

            def check_abort():
                self.check_lifecycle(deadline_ns=message.deadline_ns, stage=kind)

            check_abort()
            # request_operation 已等待本 owner 的停止确认，目标和伺服参考已清除。
            if isinstance(message, HomeRequest):
                if len(message.waypoints) > 1:
                    self.wait_settled(
                        check_abort, stage="arm standstill", deadline_ns=message.deadline_ns,
                    )
                self.home(
                    message.waypoints,
                    check_abort=check_abort, deadline_ns=message.deadline_ns,
                )
            else:
                self.driver.enter_mode(1, check_abort=check_abort)
                settled_deadline_ns = min(
                    message.deadline_ns,
                    time.monotonic_ns() + int(HOME_CONVERGENCE_TIMEOUT_S * 1e9),
                )
                self.wait_settled(
                    lambda: self.check_lifecycle(
                        deadline_ns=settled_deadline_ns, stage="prepare standstill",
                    ),
                    stage="arm standstill", deadline_ns=settled_deadline_ns,
                )
                check_abort()
        except BaseException as exc:
            error = exc
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            self.io.closing.set()
            # 无需等待可能阻塞在其他设备 SDK 内的父进程。
            self.stop()

        self.io.publish_status(self.status)
        try:
            self.pipe.send(error)
        except (BrokenPipeError, EOFError):
            self.io.closing.set()

    def command_issue(self, goal, now):
        if self.cancelled():
            return "cancelled"
        if now >= goal["expires_ns"]:
            return "command_expired"
        stamp = int(self.status["timestamp_ns"])
        if not sample_is_fresh(stamp, self.cfg.feedback_max_age_s, now):
            return "arm_feedback_expired"
        return None

    def tick(self) -> int | None:
        if self.cancelled():
            return None
        # 先消费最新目标再判时效；旧 active 到期不能覆盖邮箱中的新目标。
        pending = self.io.take_goal()
        goal = pending if pending is not None else self.active
        if goal is None:
            return None
        self.active = goal
        issue = self.command_issue(goal, time.monotonic_ns())
        if issue:
            self.io.stop.set()
            return None
        if self.reference is None:
            q, v = self.status["qpos"], self.status["qvel"]
            if np.max(np.abs(v)) > self.cfg.homing.velocity_convergence_rad_s:
                raise RuntimeError("arm moved after standstill preparation")
            self.reference = ArmTrajectory(self.cfg, q, v)
        q = cast(ArmTrajectory, self.reference).step(
            goal["q_goal"] if not goal["hold"] else None,
            hold=bool(goal["hold"]),
        )
        # 计算结束后、进入 SDK 前统一检查期限、反馈和最新控制器上报。
        issue = self.command_issue(goal, time.monotonic_ns())
        if issue:
            self.io.stop.set()
            return None
        started = time.monotonic_ns()
        self.driver.servo(q)
        self.reference.commit()
        return started

    def run(self, startup_deadline_ns):
        try:
            def startup_abort():
                self.check_lifecycle(deadline_ns=startup_deadline_ns, stage="startup")

            startup_abort()
            self.driver.connect(check_abort=startup_abort)
            settled_deadline_ns = min(
                startup_deadline_ns,
                time.monotonic_ns() + int(HOME_CONVERGENCE_TIMEOUT_S * 1e9),
            )
            self.wait_settled(
                lambda: self.check_lifecycle(
                    deadline_ns=settled_deadline_ns, stage="startup standstill",
                ),
                stage="arm standstill", deadline_ns=settled_deadline_ns,
            )
            self.driver.initialize(check_abort=startup_abort)
            self.read_feedback()
            startup_abort()
            self.io.ready.set()
            period = round(1e9 / self.cfg.servo_hz)
            feedback_period = sampling_clock_ns(self.cfg.feedback_hz)
            tick = time.monotonic_ns() + period
            feedback_due = time.monotonic_ns() + feedback_period
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
                # 先刷新到期反馈，再判断伺服当前反馈，调度迟到不写永久 fault。
                if now >= feedback_due:
                    self.read_feedback()
                    feedback_due = next_tick_ns(
                        feedback_due, feedback_period, time.monotonic_ns(),
                    )
                if time.monotonic_ns() >= tick:
                    sent_ns = self.tick()
                    tick = (sent_ns if sent_ns is not None else time.monotonic_ns()) + period
                # 停止可中断等待，目标仅在计划的伺服时刻生效。
                delay = max(0, min(tick, feedback_due) - time.monotonic_ns()) / 1e9
                if self.io.stopped.is_set():
                    self.pipe.poll(delay)
                else:
                    self.io.stop.wait(delay)
        except BaseException as exc:
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            else:
                logger.warning("arm operation ended: %s", exc)
                self.io.closing.set()
                self.io.stop.set()
        finally:
            self.io.ready.set()
            if not self.io.stopped.is_set():
                self.stop()
            try:
                self.driver.close()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.io.emergency.set()
                    self.io.stop.set()
                logger.exception("arm disconnect failed")
                previous = self.io.close_error.value.decode(errors="replace")
                detail = f"disconnect failure: {exc}"
                self.io.close_error.value = (
                    f"{previous}; {detail}" if previous else detail
                ).encode()[:1023]
            self.pipe.close()


def run_arm(cfg, io, pipe, startup_deadline_ns):
    configure_logging()
    from dexmani_real.robot.xarm7.driver import XArm7

    ArmController(cfg, io, pipe, XArm7(cfg)).run(startup_deadline_ns)
