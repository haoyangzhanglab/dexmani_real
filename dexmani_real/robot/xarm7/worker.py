"""机械臂 SDK 与伺服循环；阻塞的原生调用无法在此中断。"""

import fcntl
import logging
import os
import stat
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
)
from dexmani_real.robot.xarm7.protocol import (
    ARM_STATE_DTYPE,
    EXECUTOR_TIMEOUT_S,
    HOME_POSITION_TOLERANCE_RAD,
    READY_TIMEOUT_S,
    STANDSTILL_DWELL_S,
    HomeRequest,
    PhysicalStopRequest,
    PrepareRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.utils.log import configure_logging
from dexmani_real.utils.timing import next_tick_ns, sample_is_fresh, sampling_clock_ns

logger = logging.getLogger(__name__)


class ArmController:
    def __init__(self, cfg, io, pipe, driver, *, timing=None):
        self.cfg, self.io, self.pipe, self.driver = cfg, io, pipe, driver
        self.timing = timing
        self.status = np.zeros((), dtype=ARM_STATE_DTYPE)
        self.active = None
        self.reference: ArmTrajectory | None = None

    def cancelled(self):
        return (self.io.stop.is_set() or self.io.emergency.is_set()
                or self.io.closing.is_set() or bool(self.io.fault.value))

    def check_lifecycle(self, *, deadline_ns, stage, while_stopped=False):
        if self.pipe.poll():
            try:
                self.pipe.recv()
            except EOFError as exc:
                self.io.closing.set()
                raise OperationCancelled("parent pipe closed") from exc
            raise RuntimeError("overlapping arm lifecycle requests")
        cancelled = (
            not self.io.stop.is_set() or self.io.closing.is_set() or bool(self.io.fault.value)
            if while_stopped else self.cancelled()
        )
        if cancelled:
            raise OperationCancelled(f"{stage} cancelled")
        if time.monotonic_ns() >= deadline_ns:
            raise TimeoutError(f"{stage} execution deadline expired")

    def fault(self, reason):
        if isinstance(reason, KeyboardInterrupt):
            self.io.emergency.set()
        self.io.stop.set()
        if not self.io.fault.value:
            self.io.stop_observed.clear()
            self.io.fault.value = f"{type(reason).__name__}: {reason}".encode()[:1023]
        # traceback 不能经共享字符串/普通 pickle 传递，在 SDK 所有者处保留。
        logger.error("arm controller failed", exc_info=(type(reason), reason, reason.__traceback__))

    def read_feedback(self):
        started = time.monotonic_ns()
        succeeded = False
        try:
            q, v = self.driver.read()
            self.status["qpos"] = q
            self.status["qvel"] = v
            # 来源时间采用查询开始时刻，不能替换为缓存读取时间。
            self.status["timestamp_ns"] = started
            succeeded = True
        except Exception:
            self.status["timestamp_ns"] = 0
            raise
        finally:
            finished = time.monotonic_ns() if self.timing is not None else 0
            self.io.publish_status(self.status)
            if self.timing is not None:
                self.timing.record_feedback(
                    started_ns=started, finished_ns=finished,
                    qpos=self.status["qpos"], qvel=self.status["qvel"], succeeded=succeeded,
                )

    def wait_settled(
        self,
        check_abort,
        *,
        stage,
        deadline_ns,
        target=None,
        while_stopped=False,
    ):
        """在绝对期限内确认连续停稳；不合格样本重新累计。"""
        tolerance = self.cfg.standstill_position_tolerance_rad
        interval_ns = round(1e9 / self.cfg.feedback_hz)
        stable = None
        anchor = None
        previous_sample_ns = None
        # Owner 退出前的最后一次观测不会因 closing 已置位而忙等。
        if check_abort is None:
            wait = time.sleep
        else:
            wait = (self.io.closing if while_stopped else self.io.stop).wait
        while time.monotonic_ns() < deadline_ns:
            if check_abort is not None:
                check_abort()
            self.read_feedback()
            if check_abort is not None:
                check_abort()
            completed_ns = time.monotonic_ns()
            if completed_ns >= deadline_ns:
                break
            q, v = self.status["qpos"], self.status["qvel"]
            sample_ns = int(self.status["timestamp_ns"])
            if not sample_is_fresh(sample_ns, self.cfg.feedback_max_age_s, completed_ns):
                raise FeedbackUnavailable(f"{stage}: current arm feedback unavailable/stale")
            if while_stopped and previous_sample_ns is not None and not sample_is_fresh(
                previous_sample_ns, self.cfg.feedback_max_age_s, completed_ns,
            ):
                raise FeedbackUnavailable(f"{stage}: arm feedback gap exceeded freshness budget")
            previous_sample_ns = sample_ns
            valid = np.max(np.abs(v)) <= self.cfg.standstill_velocity_tolerance_rad_s
            if target is not None:
                valid = valid and np.max(np.abs(q - target)) <= HOME_POSITION_TOLERANCE_RAD
            if valid:
                if anchor is None or np.max(np.abs(q - anchor)) > tolerance:
                    stable, anchor = None, q.copy()
                # SDK 未提供采样时刻；首尾查询的等待时间不能算作已观测到的停稳。
                stable = completed_ns if stable is None else stable
                if sample_ns - stable >= int(STANDSTILL_DWELL_S * 1e9):
                    return
            else:
                stable = anchor = None
            # 按查询开始时刻限频，查询耗时既不拉长采样间隔，也不充作停稳时长。
            remaining_ns = min(sample_ns + interval_ns, deadline_ns) - time.monotonic_ns()
            wait(max(0, remaining_ns) / 1e9)
        if check_abort is not None:
            check_abort()
        raise TimeoutError(f"{stage}: measured position/velocity dwell not confirmed")

    def home(self, message, *, check_abort):
        samples = message.samples
        if samples.ndim != 2 or samples.shape[1] != 7 or len(samples) < 2:
            raise ValueError("invalid HOME reference array")
        if not np.all(np.isfinite(samples)):
            raise ValueError("HOME references must be finite")
        if np.any(samples < self.cfg.joint_limit_lower) or np.any(samples > self.cfg.joint_limit_upper):
            raise ValueError("HOME references violate joint limits")
        if not np.array_equal(samples[-1], self.cfg.home_qpos):
            raise ValueError("HOME references must end at home")
        hand_max_age_s = message.hand_max_age_s
        last_hand_stamp = 0

        def guarded():
            nonlocal last_hand_stamp
            check_abort()
            if hand_max_age_s is None:
                return
            value = self.io.home_hand_stamp
            # 竞争时沿用上一份已验证的来源时间，不延长其时效。
            if value.get_lock().acquire(False):
                try:
                    last_hand_stamp = value.get_obj().value
                finally:
                    value.get_lock().release()
            if not sample_is_fresh(last_hand_stamp, hand_max_age_s, time.monotonic_ns()):
                raise FeedbackUnavailable("HOME hand feedback expired")
        guarded()
        self.driver.prepare_motion(check_abort=guarded, deadline_ns=message.deadline_ns)
        # 规划前已确认停稳，伺服就绪后重新复核当前起点。
        self.read_feedback()
        guarded()
        stamp = int(self.status["timestamp_ns"])
        if not sample_is_fresh(stamp, self.cfg.feedback_max_age_s, time.monotonic_ns()):
            raise FeedbackUnavailable("HOME start feedback stale")
        start_tolerance = min(self.cfg.standstill_position_tolerance_rad, HOME_POSITION_TOLERANCE_RAD)
        if (
            np.max(np.abs(self.status["qpos"] - samples[0])) > start_tolerance
            or np.max(np.abs(self.status["qvel"])) > self.cfg.standstill_velocity_tolerance_rad_s
        ):
            raise OperationCancelled("HOME start no longer matches the stationary planned start")
        period = round(1e9 / self.cfg.servo_hz)
        start = time.monotonic_ns()
        feedback_interval_ns = round(1e9 / self.cfg.feedback_hz)
        feedback_due = start + feedback_interval_ns
        lateness = period * 4 // 5
        for index, point in enumerate(samples):
            due = start + index * period
            self.io.stop.wait(max(0, due-time.monotonic_ns()) / 1e9)
            guarded()
            if time.monotonic_ns() >= feedback_due:
                self.read_feedback()
                stamp = int(self.status["timestamp_ns"])
                if not sample_is_fresh(stamp, self.cfg.feedback_max_age_s, time.monotonic_ns()):
                    raise FeedbackUnavailable("HOME arm feedback stale")
                feedback_due = stamp + feedback_interval_ns
            guarded()
            if time.monotonic_ns()-due > lateness:
                raise HomeTimeout("HOME servo schedule late; no catch-up")
            try:
                self.driver.servo(point)
            except RuntimeError as exc:
                raise RuntimeError(f"HOME point {index + 1}/{len(samples)} failed ({exc})") from exc
            guarded()
            if time.monotonic_ns()-due > lateness:
                raise HomeTimeout("HOME servo call exceeded schedule budget")
        self.wait_settled(guarded, stage="arm HOME", deadline_ns=message.deadline_ns,
                          target=np.asarray(self.cfg.home_qpos))

    def stop(self):
        self.io.stop.set()
        self.active = self.reference = None
        try:
            self.driver.stop()
        except BaseException as exc:  # noqa: BLE001 — 停止失败独立于运动故障锁存。
            self.stop_failure("SDK stop failed", exc)
        finally:
            self.io.publish_status(self.status)
            self.io.sdk_stop_handled.set()

    def stop_failure(self, stage, error):
        if isinstance(error, KeyboardInterrupt):
            self.io.emergency.set()
        self.io.stop_error.value = f"xArm {stage}: {type(error).__name__}: {error}".encode()[:1023]
        self.io.stop_failed.set()
        self.io.closing.set()

    def observe_stop(self, deadline_ns, *, closing=False):
        stage = "arm physical stop"
        check_abort = None if closing else lambda: self.check_lifecycle(
            deadline_ns=deadline_ns, stage=stage, while_stopped=True,
        )
        self.io.stop_observed.clear()
        self.io.stop_observed_ns.value = 0
        try:
            self.wait_settled(check_abort, stage=stage, deadline_ns=deadline_ns, while_stopped=True)
            self.io.stop_observed_ns.value = int(self.status["timestamp_ns"])
        except BaseException as exc:  # noqa: BLE001 — 观测失败只描述停稳结果。
            self.stop_failure("physical stop unconfirmed", exc)
        finally:
            self.io.stop_observed.set()

    def motion_operation(self, message: PrepareRequest | HomeRequest) -> BaseException | None:
        self.io.stop_reason.value = b""
        self.io.sdk_stop_handled.clear()
        self.io.stop_observed.clear()
        is_home = isinstance(message, HomeRequest)
        kind = "home" if is_home else "prepare"
        error = None
        try:

            def check_abort():
                if is_home and time.monotonic_ns() >= message.deadline_ns:
                    raise HomeTimeout("HOME execution budget exhausted")
                self.check_lifecycle(deadline_ns=message.deadline_ns, stage=kind)

            check_abort()
            # request_operation 已等待本 owner 的停止确认，目标和伺服参考已清除。
            if is_home:
                self.home(message, check_abort=check_abort)
                self.stop()
            else:
                self.driver.prepare_motion(check_abort=check_abort, deadline_ns=message.deadline_ns)
                settled_deadline_ns = min(
                    message.deadline_ns,
                    time.monotonic_ns() + int(READY_TIMEOUT_S * 1e9),
                )
                self.wait_settled(
                    lambda: self.check_lifecycle(
                        deadline_ns=settled_deadline_ns, stage="prepare standstill",
                    ),
                    stage="arm standstill", deadline_ns=settled_deadline_ns,
                )
                check_abort()
                self.reference = ArmTrajectory(self.cfg, self.status["qpos"], self.status["qvel"])
        except BaseException as exc:  # noqa: BLE001 — 生命周期请求须停止并向父进程报告中断或故障。
            error = exc
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            # 无需等待可能阻塞在其他设备 SDK 内的父进程。
            self.stop()
            if self.io.emergency.is_set() or self.io.fault.value:
                self.io.closing.set()
        return error

    def operation(self, message: PrepareRequest | HomeRequest | PhysicalStopRequest) -> None:
        error: BaseException | None = None
        if isinstance(message, PhysicalStopRequest):
            self.observe_stop(message.deadline_ns)
        else:
            error = self.motion_operation(message)
        self.io.publish_status(self.status)
        try:
            self.pipe.send(error)
        except (BrokenPipeError, EOFError):
            self.io.closing.set()

    def stop_if_goal_invalid(self, goal, now):
        if self.cancelled():
            reason = "cancelled"
        elif now >= goal["expires_ns"]:
            reason = "command_expired"
        elif not sample_is_fresh(int(self.status["timestamp_ns"]), self.cfg.feedback_max_age_s, now):
            reason = "arm_feedback_expired"
        else:
            return False
        logger.warning("arm stopped: %s", reason)
        self.active = self.reference = None
        if reason != "cancelled":
            self.io.stop_reason.value = reason.encode()
        self.io.stop.set()
        return True

    def tick(self, scheduled_ns: int) -> int | None:
        if self.cancelled():
            return None
        # 先消费最新目标再判时效；旧 active 到期不能覆盖邮箱中的新目标。
        pending = self.io.take_goal()
        goal = pending if pending is not None else self.active
        if goal is None or self.stop_if_goal_invalid(goal, time.monotonic_ns()):
            return None
        if self.reference is None:
            raise RuntimeError("arm streaming has not been prepared")
        if self.active is None:
            # prepare 后可能还要等待观测或推理；首个目标使用当前实测起点。
            q, v = self.status["qpos"], self.status["qvel"]
            if np.max(np.abs(v)) > self.cfg.standstill_velocity_tolerance_rad_s:
                raise RuntimeError("arm moved after standstill preparation")
            self.reference = ArmTrajectory(self.cfg, q, v)
        self.active = goal
        q = self.reference.step(
            goal["q_goal"] if not goal["hold"] else None,
            hold=bool(goal["hold"]),
        )
        # 计算结束后、进入 SDK 前统一检查期限、反馈和最新控制器上报。
        if self.stop_if_goal_invalid(goal, time.monotonic_ns()):
            return None
        started = time.monotonic_ns()
        if self.timing is None:
            self.driver.servo(q)
        else:
            def record(started_ns, finished_ns, succeeded):
                self.timing.record_servo(
                    scheduled_ns=scheduled_ns, sdk_started_ns=started_ns, sdk_finished_ns=finished_ns,
                    goal_published_ns=int(goal["published_ns"]), qpos=q, succeeded=succeeded,
                )

            self.driver.servo(q, record=record)
        return started

    def run(self, startup_deadline_ns):
        try:
            def startup_abort():
                self.check_lifecycle(deadline_ns=startup_deadline_ns, stage="startup")

            startup_abort()
            self.io.device_claimed.set()
            self.driver.connect(check_abort=startup_abort)
            self.read_feedback()
            startup_abort()
            self.io.ready.set()
            period = round(1e9 / self.cfg.servo_hz)
            feedback_period = sampling_clock_ns(self.cfg.feedback_hz)
            tick = time.monotonic_ns() + period
            feedback_due = time.monotonic_ns() + feedback_period
            while not self.io.closing.is_set() and not self.io.fault.value:
                if self.io.stop.is_set() and not self.io.sdk_stop_handled.is_set():
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
                    sent_ns = self.tick(tick)
                    tick = (sent_ns if sent_ns is not None else time.monotonic_ns()) + period
                # 停止可中断等待，目标仅在计划的伺服时刻生效。
                delay = max(0, min(tick, feedback_due) - time.monotonic_ns()) / 1e9
                if self.io.sdk_stop_handled.is_set():
                    self.pipe.poll(delay)
                else:
                    self.io.stop.wait(delay)
        except BaseException as exc:  # noqa: BLE001 — Owner 顶层负责所有退出路径的停止收尾。
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            else:
                logger.warning("arm operation ended: %s", exc)
                self.io.closing.set()
                self.io.stop.set()
        finally:
            self.io.closing.set()
            if self.io.device_claimed.is_set():
                if not self.io.sdk_stop_handled.is_set():
                    self.stop()
                if not self.io.stop_failed.is_set() and not self.io.stop_observed.is_set():
                    self.observe_stop(time.monotonic_ns() + int(EXECUTOR_TIMEOUT_S * 1e9), closing=True)
                try:
                    self.driver.close()
                except BaseException as exc:  # noqa: BLE001 — 保留独立的 SDK 断开错误。
                    if isinstance(exc, KeyboardInterrupt):
                        self.io.emergency.set()
                    self.io.disconnect_error.value = f"{type(exc).__name__}: {exc}".encode()[:1023]
            self.pipe.close()


def run_arm(cfg, io, pipe, startup_deadline_ns, timing=None):
    configure_logging()
    from dexmani_real.robot.xarm7.driver import XArm7

    controller = ArmController(cfg, io, pipe, XArm7(cfg), timing=timing)
    try:
        uid = os.getuid()
        lock_dir = Path("/tmp") / f"dexmani_real-xarm7-{uid}"
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        info = lock_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
            raise RuntimeError(f"xArm lock directory must be a private directory owned by uid {uid}: {lock_dir}")
        address = quote(cfg.ip.strip(), safe="")
        # 保留锁文件 inode，避免并发启动打开彼此不同的 inode。
        with (lock_dir / f"{address}.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"xArm at {cfg.ip} is already owned by another local DexMani session") from None
            controller.run(startup_deadline_ns)
    except BaseException as exc:  # noqa: BLE001 — 锁或 Owner 启动失败通过原故障通道报告。
        controller.fault(exc)
        io.closing.set()
        pipe.close()
