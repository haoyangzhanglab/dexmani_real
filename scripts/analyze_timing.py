#!/usr/bin/env python3
"""离线查看 Raw 采样间隔与来源年龄；不代表曝光、SDK 调用或物理延迟。"""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="已发布 Raw episode 目录")
    args = parser.parse_args(argv)

    import numpy as np

    from dexmani_real.data.raw import RawReader

    with RawReader(args.episode) as reader:
        rows = reader["timing/rows"][:]
        control_hz = reader.control_hz
    if not len(rows):
        raise ValueError("Raw episode contains no timing rows")

    def statistics(values):
        if not len(values):
            return None
        milliseconds = values.astype(np.float64) / 1e6
        return dict(min=float(milliseconds.min()), median=float(np.median(milliseconds)),
                    p95=float(np.percentile(milliseconds, 95)), max=float(milliseconds.max()))

    report = dict(rows=len(rows), requested_control_hz=control_hz,
                  control_interval_ms=statistics(np.diff(rows["control_ns"])),
                  readout_ms=statistics(rows["observation_ns"] - rows["control_ns"]),
                  sources={})
    for source in ("arm_ns", "hand_ns", "tactile_aggregate_ns", "tactile_dense_ns", "camera_ns"):
        stamps = rows[source]
        valid = (stamps > 0) & (stamps <= rows["observation_ns"])
        report["sources"][source] = dict(
            missing_rows=int(np.count_nonzero(stamps == -1)),
            invalid_rows=int(np.count_nonzero(~valid & (stamps != -1))),
            age_ms=statistics(rows["observation_ns"][valid] - stamps[valid]),
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
