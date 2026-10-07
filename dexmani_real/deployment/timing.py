"""Offline summaries of actual owner/model trace intervals.

owner_tick includes recording submission and any stop inside the tick, excluding
artifact finalization. slot_lateness measures tick start against its planned start.
Nested stage durations are not additive.
"""

from collections import Counter

from dexmani_real.dataset.quality import summarize


def summarize_trace(events):
    durations = {
        name: []
        for name in (
            "model",
            "observation_read",
            "observation_build",
            "realization",
            "prefix_prepared",
            "dispatch",
            "owner_tick",
            "slot_lateness",
        )
    }
    reasons = Counter()
    for event in events:
        kind = event["event"]
        if (
            kind == "query_complete"
            and event.get("started_ns") is not None
            and event.get("completed_ns") is not None
        ):
            durations["model"].append((event["completed_ns"] - event["started_ns"]) / 1e9)
        if kind in durations and event.get("duration_ns") is not None:
            durations[kind].append(event["duration_ns"] / 1e9)
        if kind == "owner_tick" and event.get("lateness_ns") is not None:
            durations["slot_lateness"].append(event["lateness_ns"] / 1e9)
        if kind in ("invalidate", "prefix_rejected"):
            reasons[event.get("reason") or event.get("detail") or "unknown"] += 1
    return dict(
        seconds={name: summarize(values) for name, values in durations.items()},
        reasons=dict(reasons),
    )
