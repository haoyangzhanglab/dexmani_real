"""Offline regressions for raw-to-processed admission and CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_processed_v15 import _GRID_DT_S, _fake_reader, _write_raw_fixture

import dexmani_real.dataset.processing as processing
from dexmani_real.dataset.clean import analyze_episode
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
    TemporalQualityConfig,
)
from dexmani_real.dataset.processing import load_annotations, process_episode_root
from examples import export_policy_zarr, process_episodes


class _FixtureReader:
    """Minimal offline reader for processor admission tests.

    The fixture HDF5 supplies every JOINT-profile field used by real
    ``analyze_episode``. This deliberately bypasses raw-v25 sidecar validation;
    those contracts are covered by the reader tests, while this module isolates
    batch admission and publication ownership.
    """

    def __init__(self, episode: str | Path) -> None:
        self.h5_path = Path(episode)
        self.h5f = h5py.File(self.h5_path / "data.h5", "r")
        self.timing = SimpleNamespace(grid_dt_s=_GRID_DT_S)

    def __enter__(self) -> "_FixtureReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.h5f.close()


class _AnalysisControlSignal(BaseException):
    """Verify process-control exceptions do not become episode decisions."""


def _config(*, horizon: int = 16, min_full_windows: int = 1) -> ProcessingConfig:
    return ProcessingConfig(
        profile=OutputProfile.JOINT,
        horizon=horizon,
        min_full_windows=min_full_windows,
    )


def _write_episode(root: Path, name: str, *, frames: int = 24) -> Path:
    episode = root / name
    episode.mkdir()
    _write_raw_fixture(episode / "data.h5", frames=frames)
    return episode


def _install_publication_fakes(monkeypatch: pytest.MonkeyPatch):
    """Use real analysis with a lightweight processed-output boundary."""

    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)
    real_analyze = processing.analyze_episode
    analysis_calls: list[str] = []
    analyzed: dict[str, object] = {}
    written: list[object] = []

    def spy_analyze(reader, *args, **kwargs):
        decision = real_analyze(reader, *args, **kwargs)
        episode_name = reader.h5_path.name
        analysis_calls.append(episode_name)
        analyzed[episode_name] = decision
        return decision

    def write_fixture(
        reader,
        decision,
        output_root,
        _processing_config,
        *,
        task_name,
    ):
        written.append(decision)
        path = output_root / f"{reader.h5_path.name}.h5"
        with h5py.File(path, "w") as output:
            output.attrs["task_name"] = task_name
        return {
            "path": path.name,
            "source_episode": reader.h5_path.name,
            "source_frames": decision.source_frames,
            "frames": decision.selected_frames,
            "dropped_frames": decision.source_frames - decision.selected_frames,
            "full_window_count": decision.quality["full_window_count"],
        }

    monkeypatch.setattr(processing, "analyze_episode", spy_analyze)
    monkeypatch.setattr(processing, "_write_processed_episode", write_fixture)
    monkeypatch.setattr(
        processing,
        "_validate_processed_output_structure",
        lambda path, _processing_config: {"path": path.name},
    )
    return analysis_calls, analyzed, written


def test_single_pass_skips_unannotated_and_stages_report_with_task_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_accepted")
    _write_episode(input_root, "episode_short", frames=8)
    analysis_calls, analyzed, written = _install_publication_fakes(monkeypatch)
    staged: dict[str, object] = {}
    publish = processing.atomic_publish

    def inspect_staging(staging: Path, target: Path) -> None:
        report_path = staging / "process_log" / "invalid_frames_report.json"
        assert report_path.is_file()
        assert not target.exists()
        staged["report"] = json.loads(report_path.read_text(encoding="utf-8"))
        publish(staging, target)

    monkeypatch.setattr(processing, "atomic_publish", inspect_staging)
    output_root = tmp_path / "processed"
    report = process_episode_root(
        input_root,
        output_root,
        _config(),
        skip_rejected_unannotated=True,
        task_name="canonical_task",
    )

    # The real cleaner runs exactly once for each eligible source. The writer
    # receives those same decisions rather than triggering a second analysis.
    assert analysis_calls == ["episode_accepted", "episode_short"]
    assert written == [analyzed["episode_accepted"]]
    assert report["accepted_source_episode_count"] == 1
    assert report["rejected_source_episode_count"] == 1
    assert report["episodes"][1]["source_episode"] == "episode_short"
    assert report["episodes"][1]["accepted"] is False
    assert output_root.joinpath("episode_accepted.h5").is_file()
    assert not output_root.joinpath("episode_short.h5").exists()
    assert staged["report"] == report["invalid_frames_report"]
    assert (
        json.loads(
            output_root.joinpath("process_log", "invalid_frames_report.json").read_text(
                encoding="utf-8"
            )
        )
        == report["invalid_frames_report"]
    )
    with h5py.File(output_root / "episode_accepted.h5", "r") as output:
        assert str(output.attrs["task_name"]) == "canonical_task"


def test_unannotated_rejection_blocks_direct_library_call_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_short", frames=8)
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(input_root, tmp_path / "processed", _config())


def test_runtime_analysis_failure_skips_only_unannotated_episode_in_cli_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_bad")
    _write_episode(input_root, "episode_good")
    analysis_calls, _analyzed, written = _install_publication_fakes(monkeypatch)
    real_analyze = processing.analyze_episode
    attempted: list[str] = []

    def analyze_with_bad_episode(reader, *args, **kwargs):
        attempted.append(reader.h5_path.name)
        if reader.h5_path.name == "episode_bad":
            raise RuntimeError("decoder failed for fixture")
        return real_analyze(reader, *args, **kwargs)

    monkeypatch.setattr(processing, "analyze_episode", analyze_with_bad_episode)
    output_root = tmp_path / "processed"
    report = process_episode_root(
        input_root,
        output_root,
        _config(),
        skip_rejected_unannotated=True,
        task_name="canonical_task",
    )

    assert attempted == ["episode_bad", "episode_good"]
    assert analysis_calls == ["episode_good"]
    assert report["accepted_source_episode_count"] == 1
    assert report["rejected_source_episode_count"] == 1
    assert report["episodes"][0]["rejected_reason"] == (
        "RuntimeError: decoder failed for fixture"
    )
    assert [decision.source_path.name for decision in written] == ["episode_good"]
    assert output_root.joinpath("episode_good.h5").is_file()
    assert not output_root.joinpath("episode_bad.h5").exists()


def test_runtime_analysis_failure_with_explicit_include_blocks_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_bad")
    _write_episode(input_root, "episode_good")
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n  episode_bad:\n    include: true\n",
        encoding="utf-8",
    )
    _analysis_calls, _analyzed, written = _install_publication_fakes(monkeypatch)
    real_analyze = processing.analyze_episode
    attempted: list[str] = []

    def analyze_with_bad_episode(reader, *args, **kwargs):
        attempted.append(reader.h5_path.name)
        if reader.h5_path.name == "episode_bad":
            raise RuntimeError("decoder failed for fixture")
        return real_analyze(reader, *args, **kwargs)

    monkeypatch.setattr(processing, "analyze_episode", analyze_with_bad_episode)
    output_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            annotations_path=annotations_path,
            skip_rejected_unannotated=True,
        )

    assert attempted == ["episode_bad", "episode_good"]
    assert written == []
    assert not output_root.exists()


def test_runtime_analysis_failure_blocks_direct_library_call_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_bad")
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)

    def analyze_with_failure(*_args, **_kwargs):
        raise RuntimeError("decoder failed for fixture")

    monkeypatch.setattr(processing, "analyze_episode", analyze_with_failure)
    output_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(input_root, output_root, _config())

    assert not output_root.exists()


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, _AnalysisControlSignal))
def test_analysis_control_exceptions_are_not_converted_to_rejections(
    exception_type: type[BaseException],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_bad")
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)

    def stop_analysis(*_args, **_kwargs):
        raise exception_type("stop analysis")

    monkeypatch.setattr(processing, "analyze_episode", stop_analysis)
    output_root = tmp_path / "processed"

    with pytest.raises(exception_type, match="stop analysis"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            skip_rejected_unannotated=True,
        )

    assert not output_root.exists()


def test_yaml_entry_without_include_remains_explicit_and_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_short", frames=8)
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n  episode_short:\n    task_name: fixture_task\n",
        encoding="utf-8",
    )
    annotation = load_annotations(annotations_path)["episode_short"]
    assert annotation == EpisodeAnnotation(include=True, task_name="fixture_task")
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            annotations_path=annotations_path,
            skip_rejected_unannotated=True,
        )


def test_explicit_exclude_skips_unreadable_episode_before_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    (input_root / "episode_excluded").mkdir()
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n  episode_excluded:\n    include: false\n",
        encoding="utf-8",
    )

    def reader_must_not_open(_: Path) -> None:
        raise AssertionError("explicitly excluded episode opened a raw reader")

    monkeypatch.setattr(processing, "EpisodeReader", reader_must_not_open)
    report = process_episode_root(
        input_root,
        tmp_path / "processed",
        _config(),
        annotations_path=annotations_path,
        dry_run=True,
        skip_rejected_unannotated=True,
    )

    assert report["episodes"][0]["rejected_reason"] == "excluded by annotation"
    assert report["episodes"][0]["source_frames"] == 0


def test_task_name_conflict_fails_before_source_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    (input_root / "episode_annotated").mkdir()
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n  episode_annotated:\n    task_name: annotated_task\n",
        encoding="utf-8",
    )

    def reader_must_not_open(_: Path) -> None:
        raise AssertionError("task-name conflict reached source analysis")

    monkeypatch.setattr(processing, "EpisodeReader", reader_must_not_open)
    with pytest.raises(ValueError, match="conflicts with annotation task_name"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            annotations_path=annotations_path,
            dry_run=True,
            task_name="canonical_task",
        )


@pytest.mark.parametrize("task_name", ("", "unknown", " task ", "task\x7f"))
def test_invalid_global_task_name_fails_before_source_analysis(
    task_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    (input_root / "episode_unread").mkdir()

    def reader_must_not_open(_: Path) -> None:
        raise AssertionError("invalid global task_name reached source analysis")

    monkeypatch.setattr(processing, "EpisodeReader", reader_must_not_open)
    output_root = tmp_path / "processed"
    with pytest.raises((TypeError, ValueError), match="processed task_name"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            dry_run=True,
            task_name=task_name,
        )

    assert not output_root.exists()


@pytest.mark.parametrize("task_name", ("", "unknown", " task ", "task\x7f"))
def test_episode_annotation_rejects_invalid_task_name(task_name: str) -> None:
    with pytest.raises((TypeError, ValueError), match="processed task_name"):
        EpisodeAnnotation(task_name=task_name)


def test_episode_annotation_rejects_non_string_task_name() -> None:
    with pytest.raises(TypeError, match="processed task_name"):
        EpisodeAnnotation(task_name=object())


@pytest.mark.parametrize("task_name", ("", "unknown", " task ", "task\x7f"))
def test_invalid_raw_task_name_blocks_before_publication(
    task_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    episode = _write_episode(input_root, "episode_bad_task")
    with h5py.File(episode / "data.h5", "r+") as source:
        source["meta"].attrs["task_label"] = task_name
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)
    output_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="processed task identity invalid"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            skip_rejected_unannotated=True,
        )

    assert not output_root.exists()


def test_mixed_annotation_task_names_block_batch_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_foo")
    _write_episode(input_root, "episode_bar")
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n"
        "  episode_foo:\n"
        "    task_name: foo\n"
        "  episode_bar:\n"
        "    task_name: bar\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)
    output_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="multiple task_name values"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            annotations_path=annotations_path,
        )

    assert not output_root.exists()


def test_expected_task_name_mismatch_blocks_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    _write_episode(input_root, "episode_fixture")
    monkeypatch.setattr(processing, "EpisodeReader", _FixtureReader)
    output_root = tmp_path / "processed"

    with pytest.raises(ValueError, match="does not match the expected output"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            expected_task_name="other_task",
        )

    assert not output_root.exists()


def test_audit_findings_do_not_remove_high_confidence_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.h5"
    _write_raw_fixture(raw_path)
    with h5py.File(raw_path, "r+") as raw:
        action = raw["action_arm_joint_sent"][:]
        action[10, 0] += 0.3
        raw["action_arm_joint_sent"][:] = action

    reader = _fake_reader(raw_path)
    try:
        audited = analyze_episode(
            reader,
            _config(),
            source_already_validated=True,
        )
    finally:
        reader.h5f.close()

    hard_only_reader = _fake_reader(raw_path)
    try:
        hard_only = analyze_episode(
            hard_only_reader,
            ProcessingConfig(
                profile=OutputProfile.JOINT,
                temporal_quality=TemporalQualityConfig(policy=QualityPolicy.HARD_ONLY),
            ),
            source_already_validated=True,
        )
    finally:
        hard_only_reader.h5f.close()

    np.testing.assert_array_equal(audited.selected_indices, np.arange(24))
    np.testing.assert_array_equal(hard_only.selected_indices, np.arange(24))
    assert (
        audited.temporal_quality["high_confidence_reason_counts"][
            "reversible_action_impulse"
        ]
        > 0
    )
    assert "temporal_high_confidence" not in audited.drop_reason_names
    assert "excluded_count" not in audited.temporal_quality
    assert hard_only.temporal_quality["detectors_run"] is False


def _cli_report(*, dry_run: bool) -> dict:
    return {
        "source_episode_count": 1,
        "accepted_source_episode_count": 1,
        "rejected_source_episode_count": 0,
        "output_episode_count": 1,
        "source_frame_count": 24,
        "selected_frame_count": 24,
        "episodes": [],
        "invalid_frames_report": {"episodes": []},
        "dry_run": dry_run,
    }


def test_cli_uses_one_batch_call_and_filters_unknown_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()
    (input_root / "episode_known").mkdir()
    annotations_path = tmp_path / "annotations.yml"
    annotations_path.write_text(
        "episodes:\n"
        "  episode_known:\n"
        "    task_name: canonical_task\n"
        "  episode_outside_input:\n"
        "    task_name: conflicting_unknown_task\n",
        encoding="utf-8",
    )
    calls: list[tuple[tuple, dict]] = []

    monkeypatch.setattr(process_episodes, "resolve_experiment_config", lambda: object())
    monkeypatch.setattr(
        process_episodes,
        "_config",
        lambda _args, profile, _policy, _runtime: profile,
    )

    def process_once(*args, **kwargs):
        calls.append((args, kwargs))
        assert set(load_annotations(kwargs["annotations_path"])) == {"episode_known"}
        return _cli_report(dry_run=kwargs["dry_run"])

    monkeypatch.setattr(process_episodes, "process_episode_root", process_once)
    output_root = tmp_path / "canonical_task"
    common_args = [
        str(input_root),
        "--output-root",
        str(output_root),
        "--annotations",
        str(annotations_path),
        "--task-name",
        "canonical_task",
    ]

    assert process_episodes.main(common_args) == 0
    assert len(calls) == 1
    _, normal_kwargs = calls[0]
    assert normal_kwargs["dry_run"] is False
    assert normal_kwargs["skip_rejected_unannotated"] is True
    assert normal_kwargs["task_name"] == "canonical_task"
    assert normal_kwargs["expected_task_name"] == "canonical_task"

    calls.clear()
    assert process_episodes.main([*common_args, "--compare-profiles"]) == 0
    assert len(calls) == len(OutputProfile)
    assert {args[2] for args, _ in calls} == set(OutputProfile)
    assert all(kwargs["dry_run"] is True for _, kwargs in calls)
    assert all(kwargs["skip_rejected_unannotated"] is True for _, kwargs in calls)
    assert all(kwargs["expected_task_name"] == "canonical_task" for _, kwargs in calls)


def test_cli_rejects_task_name_output_root_mismatch(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    input_root.mkdir()

    with pytest.raises(SystemExit) as exc_info:
        process_episodes.main(
            [
                str(input_root),
                "--output-root",
                str(tmp_path / "other_task"),
                "--task-name",
                "canonical_task",
            ]
        )

    assert exc_info.value.code == 2


def test_export_cli_rejects_relative_task_directory() -> None:
    with pytest.raises(ValueError, match="must name one task directory"):
        export_policy_zarr._resolve_task_paths(Path(".."))


def test_removed_write_report_option_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        process_episodes._parser().parse_args([str(tmp_path), "--write-report"])
    with pytest.raises(SystemExit):
        process_episodes._parser().parse_args([str(tmp_path), "--strict"])


def test_completed_report_prints_auto_skip_reason(
    capsys: pytest.CaptureFixture,
) -> None:
    process_episodes._print_report_summary(
        {
            "source_episode_count": 2,
            "accepted_source_episode_count": 1,
            "rejected_source_episode_count": 1,
            "episodes": [
                {
                    "source_episode": "episode_rejected",
                    "rejected_reason": "source has no full window",
                }
            ],
        }
    )

    stderr = capsys.readouterr().err
    assert "1 accepted, 1 skipped" in stderr
    assert (
        "WARNING: skipping episode episode_rejected: source has no full window"
        in stderr
    )
