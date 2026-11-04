"""机械臂 SDK 与伺服循环；阻塞的原生调用无法在此中断。"""

import logging
import time

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
        self.active = None
        self.reference: ArmTrajectory | None = None

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
    ):
        """在绝对期限内确认连续停稳；不合格样本重新累计。"""
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

    def home(self, *, check_abort, deadline_ns):
        target = validate_arm_target(self.cfg.home_qpos, self.cfg.joint_limit_lower,
                                     self.cfg.joint_limit_upper)
        self.wait_settled(check_abort, stage="HOME standstill", deadline_ns=deadline_ns)
        if np.max(np.abs(self.status["qpos"] - target)) > self.cfg.homing.convergence_rad:
            self.driver.prepare_motion(0, check_abort=check_abort)
            self.wait_settled(check_abort, stage="HOME mode standstill", deadline_ns=deadline_ns)
            check_abort()
            self.driver.move_home(target)
            check_abort()
        self.wait_settled(check_abort, stage="arm HOME", deadline_ns=deadline_ns, target=target)
        check_abort()

    def stop(self):
        self.io.stop.set()
        self.active = self.reference = None
        try:
            self.driver.stop()
        except BaseException as exc:
            self.fault(exc)
        finally:
            self.io.publish_status(self.status)
            self.io.stopped.set()  # 处理结束；停止失败仍保存在 fault。

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
                self.home(check_abort=check_abort, deadline_ns=message.deadline_ns)
                self.stop()
            else:
                self.driver.prepare_motion(1, check_abort=check_abort)
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
                self.reference = ArmTrajectory(self.cfg, self.status["qpos"], self.status["qvel"])
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
        issue = self.command_issue(goal, time.monotonic_ns())
        if issue:
            logger.warning("arm stopped: %s", issue)
            self.active = self.reference = None
            self.io.stop.set()
            return None
        if self.reference is None:
            raise RuntimeError("arm streaming has not been prepared")
        if self.active is None:
            # prepare 后可能还要等待观测或推理；首个目标使用当前实测起点。
            q, v = self.status["qpos"], self.status["qvel"]
            if np.max(np.abs(v)) > self.cfg.homing.velocity_convergence_rad_s:
                raise RuntimeError("arm moved after standstill preparation")
            self.reference = ArmTrajectory(self.cfg, q, v)
        self.active = goal
        q = self.reference.step(
            goal["q_goal"] if not goal["hold"] else None,
            hold=bool(goal["hold"]),
        )
        # 计算结束后、进入 SDK 前统一检查期限、反馈和最新控制器上报。
        issue = self.command_issue(goal, time.monotonic_ns())
        if issue:
            logger.warning("arm stopped: %s", issue)
            self.active = self.reference = None
            self.io.stop.set()
            return None
        started = time.monotonic_ns()
        self.driver.servo(q)
        return started

    def run(self, startup_deadline_ns):
        try:
            def startup_abort():
                self.check_lifecycle(deadline_ns=startup_deadline_ns, stage="startup")

            startup_abort()
            self.driver.connect(check_abort=startup_abort)
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
                    try:
                        message = self.pipe.recv()
                    except EOFError:
                        self.io.closing.set()
                        break
                    self.operation(message)
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
            if not self.io.stopped.is_set():
                self.stop()
            try:
                self.driver.close()
            except BaseException as exc:
                self.fault(exc)
            self.pipe.close()


def run_arm(cfg, io, pipe, startup_deadline_ns):
    configure_logging()
    from dexmani_real.robot.xarm7.driver import XArm7

    ArmController(cfg, io, pipe, XArm7(cfg)).run(startup_deadline_ns)
