"""Offline regressions for whole-episode admission and task identity.

These tests intentionally cover the publication boundary and operator-owned
episode annotations. Row-level quality, gap, and provenance machinery is not
part of the current dataset contract; row-preserving payload and schema tests
live in the control-step test modules.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

import dexmani_real.dataset.processing as processing
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    ProcessingConfig,
)
from dexmani_real.dataset.processing import load_annotations, process_episode_root
from examples import export_policy_zarr, process_episodes
from test_control_step_dataset import permissive_test_config, write_control_episode


def _config() -> ProcessingConfig:
    return permissive_test_config()


def _write_episode(root: Path, name: str, *, frames: int = 24) -> Path:
    return write_control_episode(root, name=name, frames=frames)


def _persistent_ik(episode: Path, start: int = 10) -> None:
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["flag_frame_status"][start : start + 5] = 2


def test_batch_skips_unannotated_rejection_and_publishes_complete_episodes(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw"
    accepted = _write_episode(input_root, "episode_accepted")
    rejected = _write_episode(input_root, "episode_rejected")
    _persistent_ik(rejected)

    output_root = tmp_path / "processed"
    report = process_episode_root(
        input_root,
        output_root,
        _config(),
        skip_rejected_unannotated=True,
        task_name="canonical_task",
        expected_task_name="canonical_task",
    )

    assert report["accepted_source_episode_count"] == 1
    assert report["rejected_source_episode_count"] == 1
    assert report["episodes"][1]["rejected_reason"] == "persistent IK failure"
    artifact = output_root / f"{accepted.name}.h5"
    assert artifact.is_file()
    assert not (output_root / f"{rejected.name}.h5").exists()
    with h5py.File(artifact, "r") as processed:
        assert processed.attrs["task_name"] == "canonical_task"
        assert processed.attrs["source_frames"] == processed.attrs["episode_steps"]


def test_unannotated_rejection_blocks_direct_library_call_by_default(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "raw"
    rejected = _write_episode(input_root, "episode_rejected")
    _persistent_ik(rejected)

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(input_root, tmp_path / "processed", _config())


def test_explicit_include_rejection_blocks_publication(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    rejected = _write_episode(input_root, "episode_rejected")
    _persistent_ik(rejected)
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
        "episodes:\n  episode_rejected:\n    include: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="processing batch rejected"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            annotations_path=annotations,
            skip_rejected_unannotated=True,
        )


@pytest.mark.parametrize("exception_cls", [RuntimeError, IndexError, KeyError])
def test_technical_analysis_failure_fails_the_whole_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_cls: type[Exception],
) -> None:
    """Technical/programming failures are never per-episode rejections.

    Silently training on fewer demonstrations is more dangerous than a loud
    batch failure; a known-bad episode is excluded by annotation include:false.
    """
    input_root = tmp_path / "raw"
    _write_episode(input_root, "episode_bad")
    _write_episode(input_root, "episode_good")
    real_analyze = processing.analyze_episode

    def fail_one(reader, *args, **kwargs):
        if reader.h5_path.name == "episode_bad":
            raise exception_cls("decoder failed for fixture")
        return real_analyze(reader, *args, **kwargs)

    monkeypatch.setattr(processing, "analyze_episode", fail_one)
    output_root = tmp_path / "processed"
    with pytest.raises(exception_cls, match="decoder failed for fixture"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            skip_rejected_unannotated=True,
            task_name="fixture_task",
            expected_task_name="fixture_task",
        )
    assert not output_root.exists()


def test_analysis_control_exception_is_not_converted_to_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    _write_episode(input_root, "episode_bad")

    def stop_analysis(*_args, **_kwargs):
        raise KeyboardInterrupt("stop analysis")

    monkeypatch.setattr(processing, "analyze_episode", stop_analysis)
    with pytest.raises(KeyboardInterrupt, match="stop analysis"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            skip_rejected_unannotated=True,
        )


def test_excluded_episode_is_not_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_root = tmp_path / "raw"
    (input_root / "episode_excluded").mkdir(parents=True)
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
        "episodes:\n  episode_excluded:\n    include: false\n",
        encoding="utf-8",
    )

    def reader_must_not_open(_: Path) -> None:
        raise AssertionError("excluded episode opened a raw reader")

    monkeypatch.setattr(processing, "EpisodeReader", reader_must_not_open)
    report = process_episode_root(
        input_root,
        tmp_path / "processed",
        _config(),
        annotations_path=annotations,
        dry_run=True,
        skip_rejected_unannotated=True,
    )
    assert report["episodes"][0]["rejected_reason"] == "excluded by annotation"


def test_annotations_only_allow_whole_episode_decisions(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
        "episode_fixture:\n  exclude_ranges: [[4, 8]]\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown keys"):
        load_annotations(annotations)


def test_task_name_conflict_fails_before_source_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    (input_root / "episode_annotated").mkdir(parents=True)
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
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
            annotations_path=annotations,
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
    (input_root / "episode_unread").mkdir(parents=True)

    def reader_must_not_open(_: Path) -> None:
        raise AssertionError("invalid task_name reached source analysis")

    monkeypatch.setattr(processing, "EpisodeReader", reader_must_not_open)
    with pytest.raises((TypeError, ValueError), match="processed task_name"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            dry_run=True,
            task_name=task_name,
        )


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
) -> None:
    input_root = tmp_path / "raw"
    episode = _write_episode(input_root, "episode_bad_task")
    with h5py.File(episode / "data.h5", "r+") as source:
        source["meta"].attrs["task_label"] = task_name

    with pytest.raises(ValueError, match="processed task identity invalid"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            skip_rejected_unannotated=True,
        )


def test_mixed_annotation_task_names_block_batch_publication(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    _write_episode(input_root, "episode_foo")
    _write_episode(input_root, "episode_bar")
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
        "episodes:\n"
        "  episode_foo:\n"
        "    task_name: foo\n"
        "  episode_bar:\n"
        "    task_name: bar\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple task_name values"):
        process_episode_root(
            input_root,
            tmp_path / "processed",
            _config(),
            annotations_path=annotations,
        )


def test_expected_task_name_mismatch_blocks_before_staging(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    _write_episode(input_root, "episode_fixture")
    output_root = tmp_path / "processed"
    with pytest.raises(ValueError, match="does not match the expected output"):
        process_episode_root(
            input_root,
            output_root,
            _config(),
            expected_task_name="other_task",
        )
    assert not output_root.exists()


def _cli_report(*, dry_run: bool) -> dict:
    return {
        "source_episode_count": 1,
        "accepted_source_episode_count": 1,
        "rejected_source_episode_count": 0,
        "output_episode_count": 1,
        "episodes": [],
        "dry_run": dry_run,
    }


def test_cli_uses_one_batch_call_and_filters_unknown_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "raw"
    (input_root / "episode_known").mkdir(parents=True)
    annotations = tmp_path / "annotations.yml"
    annotations.write_text(
        "episodes:\n"
        "  episode_known:\n"
        "    task_name: canonical_task\n"
        "  episode_outside_input:\n"
        "    task_name: conflicting_unknown_task\n",
        encoding="utf-8",
    )
    calls: list[dict] = []
    monkeypatch.setattr(process_episodes, "resolve_experiment_config", lambda: object())
    monkeypatch.setattr(
        process_episodes,
        "_config",
        lambda _args, _runtime: object(),
    )

    def process_once(*_args, **kwargs):
        calls.append(kwargs)
        assert set(load_annotations(kwargs["annotations_path"])) == {"episode_known"}
        return _cli_report(dry_run=kwargs["dry_run"])

    monkeypatch.setattr(process_episodes, "process_episode_root", process_once)
    output_root = tmp_path / "canonical_task"
    args = [
        str(input_root),
        "--output-root",
        str(output_root),
        "--annotations",
        str(annotations),
        "--task-name",
        "canonical_task",
    ]
    assert process_episodes.main(args) == 0
    assert len(calls) == 1
    assert calls[0]["dry_run"] is False
    assert calls[0]["skip_rejected_unannotated"] is True
    assert calls[0]["task_name"] == "canonical_task"
    assert calls[0]["expected_task_name"] == "canonical_task"


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


def test_removed_processing_options_are_rejected(tmp_path: Path) -> None:
    parser = process_episodes._parser()
    for option in ("--write-report", "--strict", "--compare-profiles", "--horizon"):
        with pytest.raises(SystemExit):
            parser.parse_args([str(tmp_path), option])


def test_completed_report_prints_skip_reason(capsys: pytest.CaptureFixture) -> None:
    process_episodes._print_report_summary(
        {
            "source_episode_count": 2,
            "accepted_source_episode_count": 1,
            "rejected_source_episode_count": 1,
            "episodes": [
                {
                    "source_episode": "episode_rejected",
                    "rejected_reason": "persistent IK failure",
                }
            ],
        }
    )
    stderr = capsys.readouterr().err
    assert "1 accepted, 1 skipped" in stderr
    assert "WARNING: skipping episode episode_rejected: persistent IK failure" in stderr
