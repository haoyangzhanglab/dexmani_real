"""测量工具的计时边界、只读性和真实 headless SDK 功能；不作为性能样本。"""

from itertools import islice
import json
import shutil

import pytest

from scripts import benchmark_episode_visualizer as benchmark
from test_cordvip import annotated_episode as annotated_episode, task_template as task_template


def test_nested_timing_does_not_double_count(monkeypatch):
    ticks = iter((0, 5, 10, 25))
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(ticks))
    probe = benchmark.Probe()
    with probe.measure("outer"):
        with probe.measure("inner"):
            pass
    assert list(probe.samples["init.outer"]) == [25]
    assert list(probe.self_samples["init.outer"]) == [20]
    assert list(probe.self_samples["init.inner"]) == [5]
    assert benchmark.summary([])["mean"] is None
    assert benchmark.summary([1])["p99_under_sampled"]
    assert benchmark.summary(range(101))["p99"] == 99


def test_synthetic_matrix_uses_real_sdk_and_preserves_inputs(annotated_episode, tmp_path):
    episode, *_ = annotated_episode
    before = {name: (episode / name).read_bytes() for name in ("data.h5", "rgb.mp4", "hoi.h5")}
    output = tmp_path / "report"
    args = [
        str(episode),
        "--output",
        str(output),
        "--max-frames",
        "2",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--synthetic",
    ]
    assert benchmark.main(args) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["data_kind"].startswith("synthetic")
    assert report["protocol"] == {"warmups": 0, "repeats": 1}
    assert all((episode / name).read_bytes() == content for name, content in before.items())
    assert len(report["runs"]) == 8
    for run in report["runs"]:
        assert run["frames_decoded"] == run["frames_logged"] == 2
        assert run["missing_frames"] == 0 and run["peak_rss_mib"] > 0
        if run["mode"] != "instrumented":
            continue
        stages, reads = run["stages"], run["hdf5_dataset_getitem_calls"]
        assert stages["frames.rgb_decode_next"]["inclusive_ms"]["sample_count"] == 2
        assert stages["frames.HandFK.forward"]["inclusive_ms"]["sample_count"] == 2
        assert reads["frames:/depth"] == 2
        assert run["hoi_active"] == ("hoi1" in run["case"])
        assert run["point_cloud_active"] == ("pc1" in run["case"])
        if run["hoi_active"]:
            assert reads["frames:/timing/rows"] == reads["frames:/quality/iou"] == 2
            assert reads["frames:/object_pointcloud"] == 2
        else:
            assert "frames:/timing/rows" not in reads
            assert "init.align_hoi_episode" not in stages
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.main(args)
    assert (output / "report.json").read_bytes() == before


def test_missing_video_is_reported_without_starting_viewer(annotated_episode, monkeypatch):
    import rerun as rr
    from dexmani_real.data.raw import RawReader

    episode, *_ = annotated_episode
    original = RawReader.iter_rgb_frames
    monkeypatch.setattr(RawReader, "iter_rgb_frames", lambda reader: islice(original(reader), 1))
    monkeypatch.setattr(rr, "spawn", lambda *a, **kw: pytest.fail("GUI must not start"))
    import rerun.sinks

    monkeypatch.setattr(rerun.sinks, "spawn", lambda *a, **kw: pytest.fail("GUI must not start"))
    result = benchmark.run_worker(
        dict(
            mode="instrumented",
            config_path=None,
            episode=str(episode),
            max_frames=None,
            hoi=True,
            point_cloud=False,
        )
    )
    assert result["status"] == "failed"
    assert result["frames_decoded"] == 1 and result["missing_frames"] == 2
    assert "RGB missing" in result["error"]


def test_sdk_flush_failure_still_closes_raw_reader(annotated_episode, monkeypatch):
    import rerun as rr
    from scripts.visualize_episode import EpisodeVisualizer

    episode, *_ = annotated_episode
    closed = []
    original_close = EpisodeVisualizer.close

    def fail_flush(*args, **kwargs):
        raise RuntimeError("SDK flush failed")

    def close(viewer, error=None):
        original_close(viewer, error)
        closed.append(viewer.reader is None and viewer.rr is None)

    monkeypatch.setattr(rr.RecordingStream, "flush", fail_flush)
    monkeypatch.setattr(EpisodeVisualizer, "close", close)
    result = benchmark.run_worker(
        dict(mode="plain", config_path=None, episode=str(episode), max_frames=1, hoi=False, point_cloud=False)
    )
    assert result["status"] == "failed"
    assert "SDK flush failed" in result["cleanup_error"]
    assert closed == [True]


def test_unavailable_matrix_is_reported_without_success(annotated_episode, tmp_path):
    episode, *_ = annotated_episode
    (episode / "hoi.h5").unlink()
    second = tmp_path / "second"
    shutil.copytree(episode, second)
    output = tmp_path / "unavailable"
    assert (
        benchmark.main(
            [
                str(episode),
                str(second),
                "--output",
                str(output),
                "--hoi",
                "on",
                "--synthetic",
            ]
        )
        == 1
    )
    report = json.loads((output / "report.json").read_text())
    assert report["runs"] == [] and report["aggregates"] == []
    assert len(report["skipped"]) == 4
    assert '"$EPISODE_1" "$EPISODE_2"' in report["reproduce"]
