"""Read published Raw only; save one trial's descriptive metrics and sampled RGB."""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def stats(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if not x.size:
        return None
    return dict(zip(('min', 'median', 'p95', 'max'),
                    map(float, np.percentile(x, (0, 50, 95, 100)))), count=int(x.size))


session = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
result = json.loads((session / 'session_result.json').read_text())
for entry in result['episodes']:
    if entry['raw']['status'] != 'published':
        continue
    ep = Path(entry['raw']['path'])
    with h5py.File(ep / 'data.h5', 'r') as f:
        d = {k: v[:] for k, v in f.items() if isinstance(v, h5py.Dataset) and k != 'depth'}
        rows = f['timing/rows'][:]
        queries = [json.loads(x) for x in f['timing/queries'].asstr()[:]]
    t = d['timestamp']
    ids = np.flatnonzero((d['dispatch_status'] == 1).all(axis=1))
    sent_t = rows['submit_completed_ns'][ids]
    gap = np.diff(sent_t) / 1e6
    boundary = np.diff(rows['query_id'][ids]) != 0
    qmap = {q['query_id']: q for q in queries}
    statuses, counts = np.unique(d['dispatch_status'], axis=0, return_counts=True)
    ages = []
    for i in ids:
        q = qmap[int(rows['query_id'][i])]
        source = min(v for stamps in q['observations'][-1]['policy_sources'].values() for v in stamps)
        ages.append((rows['submit_started_ns'][i] - source) / 1e6)
    m = dict(session=str(session), episode=ep.name, session_result=entry,
             rows=len(t), dispatch_counts={str(status): int(count) for status, count in zip(statuses, counts)},
             gaps_ms={'within_chunk': stats(gap[~boundary]), 'between_chunks': stats(gap[boundary])},
             gaps_over_90ms=int((gap > 90).sum()),
             sdk_arm_and_hand_ms=stats((rows['submit_completed_ns'][ids] - rows['submit_started_ns'][ids]) / 1e6),
             inference_ms=stats([(q['completed_ns'] - q['started_ns']) / 1e6 for q in queries if q['completed_ns']]),
             latest_policy_source_age_ms=stats(ages),
             block_last_source_age_ms=stats(np.asarray(ages)[rows['prediction_index'][ids] == int(rows['prediction_index'][ids].max())]),
             nonfinite={k: int((~np.isfinite(v)).sum()) for k, v in d.items()},
             hand_current_abs_peak_mA_by_sdk_index=np.max(abs(d['hand_current']), axis=0).tolist(),
             tracking={})
    for part in ('arm', 'hand'):
        prev = np.searchsorted(sent_t, rows[part + '_ns'], side='right') - 1
        valid = prev >= 0
        q = np.rad2deg(d[part + '_qpos'])
        u = np.rad2deg(d['action_' + part + '_joint_target'][ids])
        error = u[prev[valid]] - q[valid]
        m['tracking'][part] = dict(
            previous_completed_target_error_abs_deg=[stats(abs(error[:, j])) for j in range(q.shape[1])],
            target_jump_deg={'within_chunk': stats(abs(np.diff(u, axis=0))[~boundary]),
                             'between_chunks': stats(abs(np.diff(u, axis=0))[boundary])})
    m['method_note'] = 'Tracking compares source-timestamped measurements against the preceding completed combined arm/hand dispatch; this is not identified firmware latency. NaN targets on not-called rows are preserved.'
    name = session.name + '_' + ep.name
    (out / (name + '.json')).write_text(json.dumps(m, ensure_ascii=False, indent=2) + '\n')
    cap = cv2.VideoCapture(str(ep / 'rgb.mp4'))
    thumbs = []
    for i in np.unique(np.round(np.linspace(0, len(t)-1, 12)).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (480, 360))
        cv2.rectangle(frame, (0, 0), (480, 27), (0, 0, 0), -1)
        cv2.putText(frame, f'row {i} / {t[i]:.2f}s', (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        thumbs.append(frame)
    cap.release()
    if thumbs:
        while len(thumbs) % 4:
            thumbs.append(np.zeros_like(thumbs[0]))
        cv2.imwrite(str(out / (name + '.jpg')), np.concatenate([
            np.concatenate(thumbs[i:i+4], axis=1) for i in range(0, len(thumbs), 4)], axis=0))
    print(json.dumps(m, ensure_ascii=False))
