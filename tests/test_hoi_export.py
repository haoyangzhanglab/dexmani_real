import h5py
import numpy as np
import zarr

from dexmani_real.config.perception import HOI_NUM_POINTS
from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.hoi import write_hoi
from dexmani_real.data.raw import CAMERA_TIMING_DTYPE, TIMING_ROW_DTYPE


def test_export_keeps_each_episode_template_point_order(tmp_path):
    base_from_camera = np.eye(4)
    points = np.arange(HOI_NUM_POINTS * 3, dtype=np.float32).reshape(-1, 3) / 4096
    timing = np.zeros(1, dtype=TIMING_ROW_DTYPE)
    timing["control_ns"] = timing["observation_ns"] = timing["camera_ns"] = 10**9
    timing["camera_sequence"] = 1
    expected = []
    for index, names, offsets, local in (
        (1, ["apple"], [0, HOI_NUM_POINTS], points),
        (2, ["banana", "bowl"], [0, 320, HOI_NUM_POINTS], points[::-1]),
    ):
        episode = tmp_path / f"episode_{index}"
        episode.mkdir()
        with h5py.File(episode / "data.h5", "w") as raw:
            raw.create_group("meta").attrs.update(
                task_label="pick", control_hz=20,
                camera_T_xarm_base_from_color=base_from_camera,
            )
            raw["timing/rows"] = timing
            raw["timing/segment_starts"] = np.array([0], dtype=np.int64)
        template = {
            "points_local": local,
            "object_names": np.asarray(names),
            "offsets": np.asarray(offsets, dtype=np.int64),
        }
        poses = np.tile(np.eye(4, dtype=np.float32), (1, len(names), 1, 1))
        poses[0, :, 2, 3] = 1
        write_hoi(
            episode / "hoi.h5", template, poses, base_from_camera,
            camera_timing=np.array([(10**9, 1)], dtype=CAMERA_TIMING_DTYPE),
            quality=[[{} for _ in names]], tracking_config={},
            object_valid=np.ones((1, len(names)), dtype=bool),
        )
        expected.append(local + np.array([0, 0, 1], dtype=np.float32))

    output = tmp_path / "pick.zarr"
    result = export_zarr(
        tmp_path, output, processing=ExportConfig(), fields=("object_pointcloud",),
    )

    assert result["source_count"] == 2
    root = zarr.open_group(str(output), mode="r")
    cloud = root["data/object_pointcloud"]
    assert cloud.dtype == np.float32
    np.testing.assert_array_equal(cloud[:], expected)
    np.testing.assert_array_equal(root["meta/episode_ends"][:], [1, 2])
