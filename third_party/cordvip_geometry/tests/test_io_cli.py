import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from cordvip_geometry.cli.arguments import parse_tracking_args
from cordvip_geometry.cli.track import run
from cordvip_geometry.cli.view import view_results
from cordvip_geometry.geometry import RectificationMaps
from cordvip_geometry.io import (
    load_initial_selection,
    output_path,
    write_json,
    write_npz,
)
from cordvip_geometry.rgbd import RGBDFrame, save_frame


def frame(row=0):
    return RGBDFrame(
        rgb=np.zeros((8, 10, 3), np.uint8),
        depth_m=np.ones((8, 10), np.float32),
        K=np.array([[10.0, 0.0, 5.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]),
        distortion_model="none",
        distortion_coeffs=(0.0,) * 5,
        source_id=f"raw:/tmp/episode:row:{row}",
        timestamp_ns=100 + row,
        clock="host_monotonic_observation",
        row_index=row,
    )


def test_npz_failure_removes_partial_and_preserves_first_error(tmp_path):
    target = tmp_path / "frame.npz"
    failure = OSError("disk write failed")
    with (
        patch("numpy.savez_compressed", side_effect=failure),
        pytest.raises(OSError) as caught,
    ):
        write_npz(target, {"points": np.ones((2, 3))})
    assert caught.value is failure
    assert not target.exists()
    write_npz(target, {"points": np.zeros((2, 3))})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_npz(target, {"points": np.ones((2, 3))})
    assert target.read_bytes() == original


def test_output_protects_raw_outside_workspace(tmp_path):
    episode = tmp_path / "external_episode"
    episode.mkdir()
    (episode / "data.h5").touch()
    with pytest.raises(ValueError):
        output_path(episode / "results", workspace=tmp_path / "workspace")


@pytest.mark.parametrize("distorted", [False, True])
def test_selection_requires_exact_frame(tmp_path, distorted):
    from dataclasses import replace

    source = frame()
    if distorted:
        source = replace(
            source,
            distortion_model="brown_conrady",
            distortion_coeffs=(3.0, 0, 0, 0, 0),
        )
    save_frame(tmp_path / "source_rgbd.npz", source)
    mask = np.zeros(source.depth_m.shape, bool)
    mask[1:4, 1:4] = True
    np.save(tmp_path / "mask.npy", mask)
    metadata = {
        "selection": "gui_confirmed",
        "selected_instance": None,
        "mask_file": "mask.npy",
        "mask_grid": "original_rgb_grid; not_automatically_rectified",
    }
    write_json(tmp_path / "selection.json", metadata)
    np.testing.assert_array_equal(load_initial_selection(tmp_path, source), mask)
    if distorted:
        maps = RectificationMaps(
            source.depth_m.shape, source.K, source.distortion_coeffs
        )
        assert not np.array_equal(maps.apply(source.rgb, source.depth_m, mask)[2], mask)
    with pytest.raises(ValueError, match="timestamp_ns"):
        load_initial_selection(
            tmp_path, replace(source, timestamp_ns=source.timestamp_ns + 1)
        )


def test_selection_rejects_a_different_raw_source(tmp_path):
    from dataclasses import replace

    source = frame()
    save_frame(tmp_path / "source_rgbd.npz", source)
    np.save(tmp_path / "mask.npy", np.ones(source.depth_m.shape, bool))
    write_json(
        tmp_path / "selection.json",
        {
            "selection": "gui_confirmed",
            "selected_instance": None,
            "mask_file": "mask.npy",
            "mask_grid": "original_rgb_grid; not_automatically_rectified",
        },
    )
    moved = replace(source, source_id="raw:/another/episode:row:0")
    with pytest.raises(ValueError, match="source_id"):
        load_initial_selection(tmp_path, moved)


def test_gui_selection_returns_original_grid(tmp_path):
    from dataclasses import replace

    from cordvip_geometry.cli.scene import choose_masks

    source = replace(
        frame(), distortion_model="brown_conrady", distortion_coeffs=(3.0, 0, 0, 0, 0)
    )
    mask = np.zeros(source.depth_m.shape, bool)
    mask[1:4, 1:4] = True
    maps = RectificationMaps(source.depth_m.shape, source.K, source.distortion_coeffs)
    with patch("cordvip_geometry.cli.scene.select_mask", return_value=(mask, None)):
        selected = choose_masks(
            {"object": {"id": "object"}},
            source,
            SimpleNamespace(),
            tmp_path,
            SimpleNamespace(provenance={"method": "test"}),
            maps,
        )
    np.testing.assert_array_equal(selected["object"], mask)
    np.testing.assert_array_equal(
        np.load(tmp_path / "initial_selections/object/mask.npy"), mask
    )


def test_cli_uses_core_asset_validation_and_rejects_unconfirmed_headless():
    from cordvip_geometry.cli.object import main

    for scale_args in (
        [],
        ["--scale-to-meters", "1", "--size-axis", "x", "--size-meters", "0.1"],
    ):
        with pytest.raises(ValueError):
            main(["prepare", "--mesh", "object.obj", "--output", "asset", *scale_args])
    with pytest.raises(SystemExit):
        parse_tracking_args(["--episode", "episode", "--headless"])


def test_headless_view_reads_only_rgb_and_source_clock(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "scene.json").write_text("unused metadata")
    source = frame()
    write_npz(
        tmp_path / "frames/000000.npz",
        {
            "rgb": source.rgb,
            "row_index": 0,
            "timestamp_ns": source.timestamp_ns,
            "clock": source.clock,
        },
    )
    with patch.dict("sys.modules", {"cv2": None, "open3d": None, "torch": None}):
        assert view_results(tmp_path, headless=True) == 0


def test_mask_only_run_partial_and_failed_cleanup(tmp_path):
    from cordvip_geometry import raw, sam2
    from cordvip_geometry.cli import track

    released = []

    class Reader:
        num_frames = 2

        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def read_camera(self):
            return SimpleNamespace()

        def __exit__(self, *exc):
            released.append("reader")

    class Segmenter:
        def __init__(self, *args, **kwargs):
            self.provenance = {"method": "test"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            released.append("segmenter")

        def initialize(self, rgb, masks):
            return masks

        def track(self, rgb):
            return {"object": np.ones(rgb.shape[:2], bool)}

    args = parse_tracking_args(
        [
            "--episode",
            str(tmp_path / "episode"),
            "--initial-selection",
            str(tmp_path / "selection"),
            "--headless",
            "--max-frames",
            "1",
            "--output",
            str(tmp_path / "partial"),
        ],
        workspace=tmp_path,
    )
    rows = [(frame(index), {}) for index in range(2)]
    with (
        patch("dexmani_real.data.raw.RawReader", Reader),
        patch.object(raw, "iter_raw_rows", side_effect=lambda *args: iter(rows)),
        patch.object(sam2, "SAM2Client", Segmenter),
        patch.object(
            track, "choose_masks", return_value={"object": np.ones((8, 10), bool)}
        ),
    ):
        assert run(args) == 0
        assert released == ["segmenter", "reader"]
        report = json.loads((args.output / "result.json").read_text())
        assert report["status"] == "partial" and report["processed_frames"] == 1
        with np.load(args.output / "frames/000000.npz", allow_pickle=False) as saved:
            assert str(saved["object__state"]) == "OBSERVED_ONLY"
            assert not saved["object__pose_valid"]
        args.output = tmp_path / "failed"
        released.clear()
        with (
            patch("cordvip_geometry.io.write_npz", side_effect=OSError("disk full")),
            pytest.raises(OSError),
        ):
            run(args)
        report = json.loads((args.output / "result.json").read_text())
        assert report["status"] == "failed" and "disk full" in report["error"]
        assert report["processed_frames"] == 0
        assert released == ["segmenter", "reader"]


@pytest.mark.parametrize(
    "key,value", [("row_index", 0.5), ("timestamp_ns", 0), ("clock", "")]
)
def test_headless_view_rejects_invalid_source_fields(tmp_path, key, value):
    (tmp_path / "frames").mkdir()
    arrays = {"rgb": frame().rgb, "row_index": 0, "timestamp_ns": 100, "clock": "test"}
    arrays[key] = value
    write_npz(tmp_path / "frames/000000.npz", arrays)
    with pytest.raises(ValueError, match="integer source row/time"):
        view_results(tmp_path, headless=True)


def test_viewer_requires_template_for_a_valid_model_pose():
    from cordvip_geometry.viewer import scene_clouds

    saved = {"object__pose_valid": True, "object__T_camera_object": np.eye(4)}
    with pytest.raises(KeyError):
        scene_clouds(saved, ["object"], {"object": (255, 0, 0)}, model_templates={})
    saved["object__pose_valid"] = False
    points, colors = scene_clouds(
        saved, ["object"], {"object": (255, 0, 0)}, model_templates={}
    )
    assert points.shape == colors.shape == (0, 3)
