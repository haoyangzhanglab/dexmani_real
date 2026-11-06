#!/usr/bin/env python3
"""xArm 通信、owner 调度和低速响应诊断；默认只读，数据在断开后输出。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import logging
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dexmani_real.config.real import load_config
from dexmani_real.robot.commands import COMMAND_LEASE_NS, OperationCancelled, RobotCommand, validate_arm_target
from dexmani_real.robot.xarm7.diagnostics import Trace, export
from dexmani_real.robot.xarm7.driver import XArm7
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.runtime.state import SessionState


def read_session(cfg, trace, duration):
    driver = XArm7(cfg, api_factory=trace.api_factory)
    try:
        driver.connect(check_abort=lambda: None)
        version = driver.api.version
        start = time.monotonic_ns()
        deadline = start + int(duration*1e9)
        due = start
        while time.monotonic_ns() < deadline:
            t = time.monotonic_ns()
            q, v = driver.read()
            trace.record('feedback', t, q=q, v=v)
            due += round(1e9/cfg.feedback_hz)
            if due < time.monotonic_ns():
                due = time.monotonic_ns()
            time.sleep(max(0, due-time.monotonic_ns())/1e9)
        return version, q, (start, time.monotonic_ns())
    finally:
        driver.close()  # 只读绝不能进入 owner 的 stop/close 流程。


def control_session(cfg, trace, parent, args, windows):
    state = SessionState()
    arm = ArmProcess(cfg, runtime_state=state, diagnostics=trace)
    def stop():
        parent.record('stop_request')
        state.request_stop()
        arm.wait_stopped()
        parent.record('stop_ack')
    def prepare():
        parent.record('prepare_request')
        arm.prepare(state.cancel_epoch, service=arm.check_error)
        parent.record('prepare_reply')
    def submit(target=None, hold=False):
        now = time.monotonic_ns()
        expires = now + COMMAND_LEASE_NS
        cmd = RobotCommand(arm_hold=True) if hold else RobotCommand(arm_qpos=validate_arm_target(target, cfg.joint_limit_lower, cfg.joint_limit_upper))
        arm.submit(cmd, expires_ns=expires)
        parent.record('submit_hold' if hold else 'submit', now, q=target, value=expires)
    def stream(seconds, target=None, hold=False, waveform=False):
        start = time.monotonic()
        while time.monotonic()-start < seconds:
            q = None if target is None else target.copy()
            if waveform:
                # 三角波端点目标：在运动途中换向，规划仍由真实 owner 完成。
                q[args.joint-1] += np.deg2rad(args.amplitude_deg) * (1 if int((time.monotonic()-start)/2)%2 == 0 else -1)
            submit(q, hold)
            time.sleep(1/30)
    try:
        arm.connect()
        initial = arm.snapshot()['qpos'][0].copy()
        if args.case == 'motion' and np.max(np.abs(initial-np.asarray(args.origin))) > cfg.standstill_position_tolerance_rad:
            raise RuntimeError('pose changed since construction of bounded HOME target')
        if args.case == 'lifecycle':
            for iteration in range(args.cycles):
                prepare()
                stream(.5, hold=True)
                stop()
                start = time.monotonic_ns()
                def cancel_service():
                    arm.check_error()
                    # 事件记录确认确实进入了本次 owner operation，避免仅测到提交前取消。
                    r = trace.rows()[max(0, trace.count.value-64):trace.count.value].copy()
                    if np.any((r['name'] == b'operation_start') & (r['start'] > start)) and not state.stop_signal.is_set():
                        parent.record('cancel_request')
                        state.request_stop()
                try:
                    arm.prepare(state.cancel_epoch, service=cancel_service)
                except OperationCancelled:
                    parent.record('cancel_complete')
                else:
                    raise RuntimeError('preparation finished before cancellation; coverage missing')
                if not arm.is_connected or not arm.io.stopped.is_set() or arm.pipe.poll():
                    raise RuntimeError('cancellation left owner/reply unresolved')
                prepare()
                stream(.5, hold=True)
                stop()
            return
        if args.case == 'motion':
            arm.home(state.cancel_epoch, service=arm.check_error, abort=lambda: False,
                     deadline_ns=time.monotonic_ns()+int(cfg.homing.timeout_s*1e9))
        prepare()
        stream(5, initial)
        start = time.monotonic_ns()
        stream(args.max_duration, initial, waveform=args.case == 'motion')
        windows.append((args.case, start, time.monotonic_ns()))
        if args.case == 'motion':
            negative = initial.copy()
            negative[args.joint-1] -= np.deg2rad(args.amplitude_deg)
            positive = initial.copy()
            positive[args.joint-1] += np.deg2rad(args.amplitude_deg)
            stream(6, negative)
            stream(.5, positive)
            stream(2, hold=True)
            stop()
            time.sleep(2)
            prepare()
            stream(6, negative)
            stream(.5, positive)
            stop()
            time.sleep(2)
            prepare()
            stream(6, negative)
            submit(positive)
            parent.record('lease_publication_end')
            time.sleep(3)
            if not arm.io.stopped.is_set():
                raise RuntimeError('lease did not stop owner')
            arm.check_error()
            prepare()
            stream(6, initial)
            stop()
            # 非同位 HOME，以及运动中取消；每次都回到固定原点，避免累计位移。
            for cancel in (False, True) * args.cycles:
                epoch = state.cancel_epoch
                def home_service():
                    arm.check_error()
                    q = arm.snapshot()['qpos'][0]
                    if cancel and abs(q[args.joint-1]-initial[args.joint-1]) > np.deg2rad(args.amplitude_deg/8) and state.cancel_epoch == epoch:
                        parent.record('home_cancel_request')
                        state.request_stop()
                try:
                    arm.home(epoch, service=home_service, abort=lambda: state.cancel_epoch != epoch,
                             deadline_ns=time.monotonic_ns()+int(cfg.homing.timeout_s*1e9))
                except OperationCancelled:
                    if not cancel:
                        raise
                    parent.record('home_cancel_complete')
                else:
                    if cancel:
                        raise RuntimeError('HOME cancellation not exercised')
                prepare()
                stream(6, initial)
                stop()
        else:
            stop()
        time.sleep(2)  # owner 在停止态继续反馈，供离线判定物理停稳。
    except KeyboardInterrupt:
        state.request_emergency_stop()
        raise
    finally:
        arm.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('read', 'hold', 'lifecycle', 'motion'), default='read')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--max-duration', type=float, default=90)
    parser.add_argument('--cycles', type=int, default=15)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--joint', type=int, choices=range(1,8))
    parser.add_argument('--amplitude-deg', type=float)
    parser.add_argument('--speed-deg-s', type=float)
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config).arm
    if args.case != 'read' and args.config is None:
        parser.error('control cases require an explicit experiment --config')
    if not np.isfinite(args.max_duration) or args.max_duration <= 0 or args.cycles < 1:
        parser.error('duration and cycles must be positive')
    if args.case == 'motion':
        if args.joint is None or any(x is None or not np.isfinite(x) or x <= 0 for x in (args.amplitude_deg, args.speed_deg_s)):
            parser.error('motion requires --joint, positive --amplitude-deg and --speed-deg-s')
        cfg = replace(cfg, max_joint_velocity_deg_per_s=args.speed_deg_s,
                      max_joint_acceleration_deg_per_s2=5, max_joint_jerk_deg_per_s3=50,
                      homing=replace(cfg.homing, max_speed_deg_s=args.speed_deg_s, max_acceleration_deg_per_s2=5))
    cfg.validate()
    if args.print_config:
        print(json.dumps(asdict(cfg), indent=2)); return 0
    output = args.output or Path('logs/xarm_diagnostics')/time.strftime('%Y%m%d_%H%M%S_')/args.case
    if output.exists():
        parser.error('output already exists')
    trace, parent = Trace(), Trace(20000)
    windows = []
    metadata = {'case': args.case, 'config_applied': args.case != 'read', 'python': sys.version, 'sdk': importlib.metadata.version('xarm-python-sdk'),
                'ruckig': importlib.metadata.version('ruckig'), 'head': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                'diff': subprocess.check_output(['git','diff'], text=True), 'joint': args.joint or 1,
                'stream_timeout_s': .020, 'target_publish_hz': 30, 'warmup_s': 5,
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'status': 'incomplete'}
    metadata['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), *Path('dexmani_real/robot/xarm7').glob('*.py')]}
    started = time.monotonic_ns()
    try:
        version, origin, window = read_session(cfg, trace, args.max_duration+5 if args.case == 'read' else .5)
        metadata['firmware'] = version
        if args.case == 'read':
            windows.append(('read', window[0]+5_000_000_000, window[1]))
        else:
            if args.case == 'motion':
                args.origin = origin
                target = origin.copy(); target[args.joint-1] += np.deg2rad(args.amplitude_deg/2)
                cfg = replace(cfg, home_qpos=tuple(target))
                cfg.validate_home()
                for sign in (-1, 1):
                    q = origin.copy(); q[args.joint-1] += sign*np.deg2rad(args.amplitude_deg)
                    validate_arm_target(q, cfg.joint_limit_lower, cfg.joint_limit_upper)
            if args.case == 'lifecycle':
                for _ in range(args.cycles):
                    control_session(cfg, trace, parent, argparse.Namespace(**{**vars(args), 'cycles': 1}), windows)
            else:
                control_session(cfg, trace, parent, args, windows)
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata['error'] = repr(exc)
        raise
    finally:
        metadata['config'] = asdict(cfg)
        metadata['dropped_events'] = trace.dropped.value + parent.dropped.value
        windows.append(('whole_run', started, time.monotonic_ns()))
        metadata['measurement_complete'] = metadata['status'] == 'completed' and metadata['dropped_events'] == 0
        original_failure = sys.exc_info()[0] is not None
        try:
            export(output, np.concatenate((trace.copy(), parent.copy())), metadata, windows)
        except BaseException:
            if not original_failure:
                raise
            logging.exception('diagnostic export also failed; preserving original failure')
        print(output)
    return 0 if metadata["measurement_complete"] else 1


if __name__ == '__main__':
    raise SystemExit(main())
