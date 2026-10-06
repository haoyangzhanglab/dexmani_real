"""Scalar-only, per-episode QA; never changes row eligibility or arrays."""

import numpy as np


def summarize(values):
    return dict(
        count=len(values),
        **dict(
            zip(
                ("p50", "p95", "max"),
                [float(v) for v in np.percentile(values, [50, 95, 100])]
                if len(values)
                else [None] * 3,
            )
        ),
    )


class EpisodeQuality:
    def __init__(self, dt, camera_clock):
        self.dt_ns = dt * 1e9
        self.camera_host = str(camera_clock).startswith("host_monotonic")
        self.previous = {}
        self.deltas = []
        self.unknown = self.nonpositive = self.gaps = 0
        self.frames = {
            name: dict(unknown=0, repeated=0, backwards=0, missing=0)
            for name in ("color_frame_number", "depth_frame_number")
        }
        self.ages = {name: [] for name in ("arm", "hand", "camera")}
        self.negative_ages = dict.fromkeys(self.ages, 0)
        self.spreads = []
        self.dispatch = {"arm": {}, "hand": {}}

    def update(self, rows):
        observations = rows["observation_timestamp_ns"]
        for stamp in observations:
            stamp = int(stamp)
            previous = self.previous.get("observation_timestamp_ns", 0)
            if stamp <= 0:
                self.unknown += 1
            elif previous > 0:
                delta = stamp - previous
                self.deltas.append(delta / 1e9)
                self.nonpositive += int(delta <= 0)
                self.gaps += int(delta > 1.5 * self.dt_ns)
            self.previous["observation_timestamp_ns"] = stamp
        for name, counts in self.frames.items():
            for number in rows[name]:
                number = int(number)
                previous = self.previous.get(name, 0)
                if number <= 0:
                    counts["unknown"] += 1
                elif previous > 0:
                    counts["repeated"] += int(number == previous)
                    counts["backwards"] += int(number < previous)
                    counts["missing"] += max(0, number - previous - 1)
                self.previous[name] = number
        sources = [rows["arm_read_timestamp_ns"], rows["hand_read_timestamp_ns"]]
        if self.camera_host:
            sources.append(rows["camera_timestamp_ns"])
        for i, observation in enumerate(observations):
            known = []
            for name, values in zip(self.ages, sources):
                stamp = int(values[i])
                if stamp > 0:
                    known.append(stamp)
                    if observation > 0:
                        age = (int(observation) - stamp) / 1e9
                        self.ages[name].append(age)
                        self.negative_ages[name] += int(age < 0)
            if len(known) >= 2:
                self.spreads.append((max(known) - min(known)) / 1e9)
        for index, counts in enumerate(self.dispatch.values()):
            values, frequency = np.unique(rows["dispatch_status"][:, index], return_counts=True)
            for value, count in zip(values, frequency):
                key = str(int(value))
                counts[key] = counts.get(key, 0) + int(count)

    def report(self, notes, row_count, float_fields):
        return dict(
            observation_dt_s=summarize(self.deltas),
            unknown_timestamps=self.unknown,
            nonpositive_intervals=self.nonpositive,
            gaps=self.gaps,
            gap_nominal_dt_ratio=1.5,
            frame_numbers=self.frames,
            source_age_s={name: summarize(v) for name, v in self.ages.items()},
            negative_source_ages=self.negative_ages,
            source_spread_s=summarize(self.spreads),
            camera_host_clock=self.camera_host,
            rgb_depth_exposure_skew=None,
            tactile_sync=None,
            finite_rows={
                name: row_count - notes.get(name + "_missing_rows", 0) for name in float_fields
            },
            missing_rows={name: notes.get(name + "_missing_rows", 0) for name in float_fields},
            dispatch=self.dispatch,
        )
