"""离线导出的小规模诊断记录；仅显式诊断入口创建。时间统一为 monotonic_ns。"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time

import numpy as np

DTYPE = np.dtype([
    ('name', 'S40'), ('start', 'i8'), ('end', 'i8'), ('value', 'i8'),
    ('code', 'i8'), ('q', 'f8', (7,)), ('v', 'f8', (7,)), ('error', 'S256'),
])
SDK_METHODS = ('get_err_warn_code', 'get_joint_states', 'set_servo_angle_j',
               'motion_enable', 'set_mode', 'set_state', 'set_collision_sensitivity',
               'set_tcp_load', 'set_joint_maxacc', 'set_servo_angle', 'get_version')


class Trace:
    """每个缓冲只有一个写线程；先写完整行再提交计数，不经生命周期管道传输。"""
    def __init__(self, capacity=200000):
        ctx = mp.get_context('spawn')
        self.buffer = ctx.RawArray('b', capacity * DTYPE.itemsize)
        self.count = ctx.RawValue('i', 0)
        self.dropped = ctx.RawValue('i', 0)
        self.capacity = capacity
        self._rows = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_rows"] = None
        return state

    def rows(self):
        if self._rows is None:
            self._rows = np.frombuffer(self.buffer, dtype=DTYPE)
        return self._rows

    def record(self, name, start=None, *, value=0, code=0, q=None, v=None, error='', end=None):
        now = time.monotonic_ns()
        i = self.count.value
        if i == self.capacity:
            self.dropped.value += 1
            return
        self.rows()[i] = (name.encode(), now if start is None else start,
                          now if end is None else end, value, code,
                          0 if q is None else q, 0 if v is None else v,
                          error.encode()[:256])
        self.count.value = i + 1

    def wrap(self, name, method):
        owner_thread = threading.get_ident()
        def call(*args, **kwargs):
            if threading.get_ident() != owner_thread:
                return method(*args, **kwargs)
            start = time.monotonic_ns()
            try:
                result = method(*args, **kwargs)
            except BaseException as exc:
                end = time.monotonic_ns()
                self.record(name, start, end=end, code=-999, error=repr(exc))
                raise
            end = time.monotonic_ns()
            code = result[0] if isinstance(result, (tuple, list)) else result
            self.record(name, start, end=end, q=(args[0] if args else kwargs.get('angles')) if name == 'set_servo_angle_j' else None, code=int(code) if isinstance(code, (int, np.integer)) else 0,
                        value=int(args[0] if args else kwargs.get('state' if name == 'set_state' else 'mode', 0)) if name in ('set_state', 'set_mode') else 0)
            return result
        return call

    def api_factory(self, *args, **kwargs):
        from xarm.wrapper import XArmAPI
        api = XArmAPI(*args, do_not_open=True, **kwargs)
        # 安装版 SDK 连接时可能清告警；诊断应保留现场错误事实。
        def refuse_clear():
            raise RuntimeError('diagnostics refuses automatic SDK warning clearing')
        api.arm.clean_warn = refuse_clear
        for name in SDK_METHODS:
            setattr(api.arm, name, self.wrap(name, getattr(api.arm, name)))
        try:
            api.connect()
        except BaseException:
            api.disconnect()
            raise
        return api

    def copy(self):
        return self.rows()[:self.count.value].copy()


def percentiles(values):
    return dict(zip(('p50', 'p95', 'p99', 'max'), map(float, np.percentile(values, [50, 95, 99, 100])))) if len(values) else None


def summarize(rows, windows):
    result = {}
    for label, start, end in windows:
        r = rows[(rows['start'] >= start) & (rows['end'] <= end)]
        stages = {}
        for name in np.unique(r['name']):
            stage = r[r['name'] == name]
            stages[name.decode()] = {'count': len(stage), 'duration_ms': percentiles((stage['end']-stage['start'])/1e6),
                                    'codes': {str(int(c)): int(sum(stage['code'] == c)) for c in np.unique(stage['code'])},
                                    'exceptions': int(sum(stage['error'] != b'')),
                                    'timeouts': int(np.isin(stage['code'], [3, 100]).sum() + sum(e.startswith(b'TimeoutError') for e in stage['error']))}
        sends = np.sort(r[r['name'] == b'set_servo_angle_j']['start'])
        intervals = np.diff(sends)
        overlap = {}
        for name in (b'feedback', b'ruckig'):
            spans = r[r['name'] == name]
            overlap[name.decode()] = sum(bool(np.any((spans['start'] < b) & (spans['end'] > a)))
                                          for a, b in zip(sends[:-1], sends[1:]) if b-a > 20_000_000)
        tick = r[r['name'] == b'tick']
        tick_indices = np.searchsorted(tick['start'], sends, side='right')-1
        matched = tick_indices >= 0
        send_lateness = sends[matched]-tick['value'][tick_indices[matched]]
        submissions = {int(row['value']): int(row['start']) for row in r
                       if row['name'] in (b'submit', b'submit_hold')}
        adoption = []
        for row in r[r['name'] == b'adopt']:
            submitted = submissions.pop(int(row['value']), None)
            if submitted is not None:
                adoption.append((int(row['start'])-submitted)/1e6)
        result[label] = {'window_s': (end-start)/1e9, 'stages': stages,
                         'effective_hz': len(sends)/((end-start)/1e9),
                         'interval_ms': percentiles(intervals/1e6),
                         'over_10ms_fraction': float(np.mean(intervals > 10_000_000)) if len(intervals) else None,
                         'over_20ms_fraction': float(np.mean(intervals > 20_000_000)) if len(intervals) else None,
                         'submit_to_adopt_ms': percentiles(adoption),
                         'servo_lateness_ms': percentiles(send_lateness/1e6),
                         'tick_lateness_ms': percentiles((tick['start']-tick['value'])/1e6),
                         'feedback_age_ms': percentiles(r[r['name'] == b'reference']['value']/1e6),
                         'slow_interval_overlap': overlap}
    return result


def export(output, rows, metadata, windows):
    import json
    output.mkdir(parents=True, exist_ok=False)
    rows = rows[np.argsort(rows['start'], kind='stable')]
    np.savez_compressed(output/'raw.npz', events=rows)
    windows = list(windows)
    starts = rows[rows['name'] == b'operation_start']
    ends = rows[rows['name'] == b'operation_end']
    for i, begin in enumerate(starts):
        matching = ends[ends['start'] >= begin['start']]
        if len(matching):
            windows.append((f'lifecycle_{i+1}', int(begin['start']), int(matching[0]['end'])+1))
    metadata['windows'] = windows
    metadata['statistics'] = summarize(rows, windows)
    feedback = rows[rows['name'] == b'feedback']
    # 波动仅统计明确的只读/保持窗口，不能把运动范围当作噪声。
    noise = {}
    for label, start, end in windows:
        if label not in ('read', 'hold'):
            continue
        f = feedback[(feedback['start'] >= start) & (feedback['end'] <= end)]
        if len(f):
            noise[label] = {'position_std_rad': f['q'].std(axis=0).tolist(),
                            'position_peak_to_peak_rad': np.ptp(f['q'], axis=0).tolist(),
                            'velocity_abs_max_rad_s': np.abs(f['v']).max(axis=0).tolist()}
    metadata['stationary'] = noise
    stops = []
    requests = rows[np.isin(rows['name'], [b'stop_request', b'cancel_request', b'home_cancel_request', b'lease_publication_end'])]
    cfg = metadata['config']
    for request in requests:
        t = int(request['start'])
        calls = rows[(rows['name'] == b'set_state') & (rows['value'] == 4) & (rows['start'] >= t)]
        if not len(calls):
            continue
        call = calls[0]
        following = rows[(rows['start'] > call['end']) & np.isin(rows['name'], [b'operation_start', b'set_servo_angle_j'])]
        until = int(following[0]['start']) if len(following) else int(rows['end'].max())
        f = feedback[(feedback['start'] >= call['end']) & (feedback['end'] < until)]
        anchor = None
        stable = None
        settled = None
        for sample in f:
            if np.max(np.abs(sample['v'])) > cfg['standstill_velocity_tolerance_rad_s']:
                anchor = stable = None
                continue
            if anchor is None or np.max(np.abs(sample['q']-anchor)) > cfg['standstill_position_tolerance_rad']:
                anchor, stable = sample['q'], sample['end']
            if sample['start']-stable >= 300_000_000:
                settled = int(sample['end'])
                break
        sends_before = rows[(rows['name'] == b'set_servo_angle_j') & (rows['start'] < call['start'])]
        stops.append({'request': request['name'].decode(), 'request_ns': t,
                      'sdk_stop_start_ns': int(call['start']), 'sdk_stop_end_ns': int(call['end']),
                      'request_to_sdk_ms': float((call['start']-t)/1e6),
                      'last_servo_start_ns': int(sends_before[-1]['start']) if len(sends_before) else None,
                      'observed_settled_ns': settled,
                      'request_to_observed_settled_ms': (settled-t)/1e6 if settled else None})
    metadata['stops'] = stops
    (output/'summary.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    plot(output, rows, metadata, windows)


def plot(output, rows, metadata, windows):
    feedback = rows[rows['name'] == b'feedback']
    if not len(rows):
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    origin = rows['start'].min()
    joint = metadata.get('joint', 1)-1
    for name, label in ((b'submit', 'target'), (b'reference', 'reference'), (b'feedback', 'measured')):
        r = rows[rows['name'] == name]
        if len(r):
            axes[0].plot((r['start']-origin)/1e9, np.rad2deg(r['q'][:, joint]), label=label, linewidth=.8,
                         drawstyle='steps-post' if name == b'submit' else 'default')
    if len(feedback):
        axes[1].plot((feedback['start']-origin)/1e9, np.rad2deg(feedback['v'][:, joint]))
    sends = rows[rows['name'] == b'set_servo_angle_j']['start']
    # 不把主动停止/HOME 前后的无发送时间画成伺服抖动。
    boundaries = rows[np.isin(rows['name'], [b'stop_handle', b'operation_start'])]['start']
    groups = np.searchsorted(boundaries, sends, side='right')
    active = groups[1:] == groups[:-1]
    axes[2].plot((sends[1:][active]-origin)/1e9, np.diff(sends)[active]/1e6, '.', markersize=1)
    axes[2].axhline(10, color='gray', linewidth=.5)
    for label, start, end in windows:
        if label in ('read', 'hold', 'motion'):
            for ax in axes:
                ax.axvspan((start-origin)/1e9, (end-origin)/1e9, color='gray', alpha=.08)
    for stamp in rows[rows['name'] == b'stop_request']['start']:
        axes[0].axvline((stamp-origin)/1e9, color='red', alpha=.25, linewidth=.5)
    for ax, label in zip(axes, ('position (deg)', 'velocity (deg/s)', 'send interval (ms)')):
        ax.set_ylabel(label)
        ax.grid(True)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[2].set_xlabel('monotonic elapsed (s)')
    fig.tight_layout()
    fig.savefig(output/'response.png')
    plt.close(fig)
