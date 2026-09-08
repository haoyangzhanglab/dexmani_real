"""Offline characterization for raw-episode admission decisions."""

from __future__ import annotations

from pathlib import Path

from dexmani_real.dataset.clean import analyze_episode
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    OutputProfile,
    ProcessingConfig,
)
from dexmani_real.dataset.processing import load_annotations
from examples.process_episodes import _merge_exclusions
from test_processed_v14 import _fake_reader, _write_raw_fixture


def _decision(
    path: Path,
    *,
    annotation: EpisodeAnnotation | None = None,
    config: ProcessingConfig | None = None,
):
    _write_raw_fixture(path)
    reader = _fake_reader(path)
    try:
        return analyze_episode(
            reader,
            config or ProcessingConfig(profile=OutputProfile.JOINT),
            annotation,
            source_already_validated=True,
        )
    finally:
        reader.h5f.close()


def test_admission_keeps_valid_and_routes_the_three_rejection_categories(tmp_path):
    valid = _decision(tmp_path / "episode_valid.h5")
    assert valid.accepted
    assert valid.rejected_reason is None

    # A short source-contiguous episode is rejected by the existing full-window
    # requirement while still retaining its source rows for audit reporting.
    rejected_config = ProcessingConfig(
        profile=OutputProfile.JOINT,
        horizon=24,
        min_full_windows=2,
    )
    unannotated_rejected = _decision(
        tmp_path / "episode_unannotated_rejected.h5",
        config=rejected_config,
    )
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n"
        "  episode_explicit_include_rejected.h5:\n"
        "    task_name: fixture_task\n",
        encoding="utf-8",
    )
    parsed_annotations = load_annotations(annotations_path)
    yaml_default_include = parsed_annotations[
        "episode_explicit_include_rejected.h5"
    ]
    assert yaml_default_include.include is True
    explicitly_included_rejected = _decision(
        tmp_path / "episode_explicit_include_rejected.h5",
        annotation=yaml_default_include,
        config=rejected_config,
    )
    explicitly_excluded = _decision(
        tmp_path / "episode_explicit_exclude.h5",
        annotation=EpisodeAnnotation(include=False),
    )

    assert not unannotated_rejected.accepted
    assert not explicitly_included_rejected.accepted
    assert unannotated_rejected.rejected_reason == (
        "source-contiguous segments provide 1 full windows; requires 2"
    )
    assert explicitly_included_rejected.rejected_reason == (
        unannotated_rejected.rejected_reason
    )
    assert explicitly_excluded.rejected_reason == "excluded by annotation"

    results = [
        {
            "name": "episode_valid.h5",
            "status": "ok",
            "decision": valid.to_dict(),
            "error": None,
        },
        {
            "name": "episode_unannotated_rejected.h5",
            "status": "SKIP",
            "decision": unannotated_rejected.to_dict(),
            "error": None,
        },
        {
            "name": "episode_explicit_include_rejected.h5",
            "status": "SKIP",
            "decision": explicitly_included_rejected.to_dict(),
            "error": None,
        },
        {
            "name": "episode_explicit_exclude.h5",
            "status": "user-excluded",
            "decision": explicitly_excluded.to_dict(),
            "error": None,
        },
    ]
    user_annotations = {
        "episode_explicit_include_rejected.h5": yaml_default_include,
        "episode_explicit_exclude.h5": EpisodeAnnotation(include=False),
    }

    merged, blocking = _merge_exclusions(user_annotations, results)

    # Unannotated rejects are converted to non-blocking exclusions, explicit
    # includes remain blocking, and explicit excludes remain user-owned.
    assert merged["episode_unannotated_rejected.h5"] == EpisodeAnnotation(
        include=False
    )
    assert merged["episode_explicit_include_rejected.h5"] == yaml_default_include
    assert merged["episode_explicit_exclude.h5"] == EpisodeAnnotation(include=False)
    assert blocking == [
        (
            "episode_explicit_include_rejected.h5",
            explicitly_included_rejected.rejected_reason,
        )
    ]
