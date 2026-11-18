from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np
import pytest

from dexmani_real.data.raw import (
    TIMING_ROW_DTYPE,
    RawDataError,
    RawReader,
    _VideoWriter,
)

WIDTH, HEIGHT = 32, 24


def write_raw_data(path: Path, offsets_ns: tuple[int, ...]) -> np.ndarray:
    path.mkdir()
    rows = np.zeros(len(offsets_ns), dtype=TIMING_ROW_DTYPE)
    rows["control_ns"] = 10**15 + np.asarray(offsets_ns)
    rows["observation_ns"] = rows["control_ns"] + 1_000
    with h5py.File(path / "data.h5", "w") as data:
        meta = data.create_group("meta")
        meta.attrs.update(camera_color_width=WIDTH, camera_color_height=HEIGHT)
        data.create_dataset("timing/rows", data=rows)
    return rows["control_ns"]


def write_raw_rgb(path: Path, offsets_ns: tuple[int, ...]) -> None:
    stamps = write_raw_data(path, offsets_ns)
    video = _VideoWriter(path / "rgb.mp4", fps=29.97, width=WIDTH, height=HEIGHT)
    video.open()
    try:
        for index, stamp in enumerate(stamps):
            image = np.full((HEIGHT, WIDTH, 3), 40 + 20 * index, dtype=np.uint8)
            video.write_frame(image, int(stamp))
    finally:
        video.close()


def test_rgb_timing_matches_raw_after_seek(tmp_path):
    path = tmp_path / "episode"
    write_raw_rgb(path, (0, 33_346_679, 105_550_386))
    with RawReader(path) as reader:
        frames = list(reader.iter_rgb_frames())
        assert len(frames) == 3
        assert frames[0].mean() < frames[1].mean() < frames[2].mean()
        np.testing.assert_array_equal(list(reader.iter_rgb_frames()), frames)


def test_rgb_from_different_episode_with_same_frame_count_and_shape(tmp_path):
    path = tmp_path / "episode_1"
    other = tmp_path / "episode_2"
    write_raw_rgb(path, (0, 33_346_679, 105_550_386))
    write_raw_rgb(other, (0, 33_346_680, 105_550_386))
    (other / "rgb.mp4").replace(path / "rgb.mp4")
    with RawReader(path) as reader, pytest.raises(RawDataError, match="RGB timestamp mismatch at row 1"):
        list(reader.iter_rgb_frames())


def test_old_constant_rate_rgb_is_rejected(tmp_path):
    path = tmp_path / "episode"
    stamps = write_raw_data(path, (0, 62_646_194, 127_457_739))
    with av.open(str(path / "rgb.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=16)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv444p"
        for index in range(len(stamps)):
            pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 16)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))

    with RawReader(path) as reader:
        np.testing.assert_array_equal(reader.read_timing()["control_ns"], stamps)
        with pytest.raises(RawDataError, match=r"RGB timestamp mismatch at row 0:.*time base 1/"):
            list(reader.iter_rgb_frames())
