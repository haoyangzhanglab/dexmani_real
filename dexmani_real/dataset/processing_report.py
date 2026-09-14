"""Versioned batch admission reports shared by processing and inspection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from dexmani_real.dataset.contracts import EpisodeDecision, validate_processed_task_name
from dexmani_real.dataset.processed import PROCESSED_SCHEMA_NAME, PROCESSED_SCHEMA_VERSION

PROCESSING_REPORT_FILENAME = "processing_report.yaml"
PROCESSING_REPORT_SCHEMA_NAME = "dexmani-real-processing-report"
PROCESSING_REPORT_SCHEMA_VERSION = 1

_STATUSES = ("accepted", "skipped", "excluded")
_EXCLUDED_REASON = "excluded by annotation"
_EPISODE_FIELDS = {
    "source_episode",
    "status",
    "reason",
    "source_frames",
    "processed_frames",
}
_REPORT_FIELDS = {
    "report_schema_name",
    "report_schema_version",
    "processed_schema_name",
    "processed_schema_version",
    "task_name",
    "source_episode_count",
    "episodes",
    *(f"{status}_episode_count" for status in _STATUSES),
}


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def validate_processing_report(report: Any) -> None:
    """Reject unsupported or internally inconsistent persisted reports."""
    if not isinstance(report, Mapping) or set(report) != _REPORT_FIELDS:
        raise ValueError("processing report must contain exactly the report fields")
    version = _nonnegative_integer(
        report["report_schema_version"], "report_schema_version"
    )
    if (
        report["report_schema_name"] != PROCESSING_REPORT_SCHEMA_NAME
        or version != PROCESSING_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported processing report schema")
    _nonempty_string(report["processed_schema_name"], "processed_schema_name")
    if not _nonnegative_integer(
        report["processed_schema_version"], "processed_schema_version"
    ):
        raise ValueError("processed_schema_version must be positive")
    validate_processed_task_name(report["task_name"])
    total = _nonnegative_integer(report["source_episode_count"], "source_episode_count")
    counts = {
        status: _nonnegative_integer(
            report[f"{status}_episode_count"], f"{status}_episode_count"
        )
        for status in _STATUSES
    }
    episodes = report["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("episodes must be a list")
    names: set[str] = set()
    actual: Counter[str] = Counter()
    for entry in episodes:
        if not isinstance(entry, Mapping) or set(entry) != _EPISODE_FIELDS:
            raise ValueError("episode must contain exactly the episode fields")
        name = _nonempty_string(entry["source_episode"], "source_episode")
        if name in names:
            raise ValueError(f"duplicate source_episode: {name}")
        names.add(name)
        status = entry["status"]
        if status not in _STATUSES:
            raise ValueError(f"{name}: invalid status")
        source = _nonnegative_integer(entry["source_frames"], f"{name}: source_frames")
        processed = _nonnegative_integer(
            entry["processed_frames"], f"{name}: processed_frames"
        )
        reason = entry["reason"]
        if status == "accepted":
            if reason is not None or source <= 0 or processed != source:
                raise ValueError(
                    f"{name}: accepted requires null reason and equal positive frame counts"
                )
        else:
            _nonempty_string(reason, f"{name}: reason")
            if processed != 0:
                raise ValueError(
                    f"{name}: rejected episodes must have zero processed_frames"
                )
            if (status == "excluded") != (reason == _EXCLUDED_REASON):
                raise ValueError(f"{name}: annotation exclusion status/reason mismatch")
        actual[status] += 1
    if total != len(episodes) or any(counts[s] != actual[s] for s in _STATUSES):
        raise ValueError("processing report counts do not match episode entries")


def build_processing_report(
    decisions: Sequence[EpisodeDecision], *, task_name: str
) -> dict[str, Any]:
    """Project authoritative decisions into the minimal persisted contract."""
    episodes = []
    for decision in decisions:
        if decision.accepted:
            status = "accepted"
        elif decision.rejected_reason == _EXCLUDED_REASON:
            status = "excluded"
        else:
            status = "skipped"
        episodes.append(
            {
                "source_episode": decision.source_path.name,
                "status": status,
                "reason": decision.rejected_reason,
                "source_frames": decision.source_frames,
                "processed_frames": decision.processed_frames,
            }
        )
    counts = Counter(entry["status"] for entry in episodes)
    report = {
        "report_schema_name": PROCESSING_REPORT_SCHEMA_NAME,
        "report_schema_version": PROCESSING_REPORT_SCHEMA_VERSION,
        "processed_schema_name": PROCESSED_SCHEMA_NAME,
        "processed_schema_version": PROCESSED_SCHEMA_VERSION,
        "task_name": task_name,
        "source_episode_count": len(episodes),
        **{f"{s}_episode_count": counts[s] for s in _STATUSES},
        "episodes": episodes,
    }
    validate_processing_report(report)
    return report


class _ReportLoader(yaml.SafeLoader):
    """Do not silently replace duplicate YAML mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError(
                    "processing report has a non-string or duplicate mapping key"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_processing_report(path: str | Path) -> dict[str, Any]:
    """Load YAML safely and validate before exposing any report fields."""
    with Path(path).open(encoding="utf-8") as stream:
        try:
            report = yaml.load(stream, Loader=_ReportLoader)
        except (yaml.YAMLError, RecursionError) as exc:
            raise ValueError(f"invalid processing report YAML: {exc}") from exc
    validate_processing_report(report)
    return report


def write_processing_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write a validated report inside the caller-owned unpublished staging tree."""
    validate_processing_report(report)
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(dict(report), stream, sort_keys=False, allow_unicode=True)
