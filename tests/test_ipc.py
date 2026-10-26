import json
import multiprocessing as mp
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.ipc.ring import SharedRing
from dexmani_real.ipc.sensors import SensorChannels
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
    read_vr_frame,
)
from dexmani_real.sensor.camera.config import CameraParams
from dexmani_real.sensor.camera.worker import run_camera_worker


def publish_record(ring, record):
    return ring.write(**{name: record[name] for name in record.dtype.names})


def publish_patterns(ring, barrier, finished):
    try:
        barrier.wait()
        values = np.empty(ring.dtype["values"].shape, dtype=np.float32)
        for sequence in range(1, 301):
            values.fill(sequence)
            ring.write(frame_id=sequence, values=values, mirror=-values)
            time.sleep(0.0005)
    finally:
        finished.set()


def read_patterns(ring, barrier, finished, counts, reader):
    barrier.wait()
    last_sequence = 0
    deadline = time.monotonic() + 10
    fields = None if reader == 0 else ("frame_id", "values", "mirror")
    while not finished.is_set() or last_sequence < ring.latest_sequence:
        if time.monotonic() >= deadline:
            raise AssertionError("reader timed out")
        sample = ring.read(fields=fields)
        if sample is None:
            continue
        data, sequence = sample
        assert sequence >= last_sequence
        assert int(data["frame_id"]) == sequence
        assert np.all(data["values"] == sequence)
        assert np.all(data["mirror"] == -sequence)
        last_sequence = sequence
        counts[reader] += 1
    assert last_sequence == 300


class SharedRingTests(unittest.TestCase):
    def setUp(self):
        self.ctx = mp.get_context("spawn")
        self.dtype = np.dtype(
            [("frame_id", "<u8"), ("values", "<f4", (4,)), ("mirror", "<f4", (4,))],
            align=True,
        )
        self.ring = SharedRing(self.ctx, self.dtype, 3)

    def publish(self, value):
        values = np.full(4, value, dtype=np.float32)
        return self.ring.write(frame_id=value, values=values, mirror=-values)

    def test_empty_and_wrapped_sequences(self):
        self.assertEqual(self.ring.latest_sequence, 0)
        self.assertIsNone(self.ring.read())
        self.assertIsNone(self.ring.read(sequence=1))
        for sequence in range(1, 7):
            self.assertEqual(self.publish(sequence), sequence)
            data, actual_sequence = self.ring.read()
            self.assertEqual(actual_sequence, sequence)
            self.assertEqual(int(data["frame_id"]), sequence)
            np.testing.assert_array_equal(data["values"], np.full(4, sequence))
        self.assertEqual(self.ring.latest_sequence, 6)
        for sequence in (0, 1, 2, 3, 7):
            self.assertIsNone(self.ring.read(sequence=sequence))
        for sequence in (4, 5, 6):
            data, actual_sequence = self.ring.read(sequence=sequence)
            self.assertEqual(int(data["frame_id"]), sequence)
            self.assertEqual(actual_sequence, sequence)

    def test_full_and_partial_reads_own_their_data(self):
        self.publish(1)
        full = self.ring.read()[0]
        partial = self.ring.read(fields=("values",))[0]
        self.assertIsInstance(full, np.void)
        self.assertEqual(set(partial), {"values"})
        self.assertFalse(np.shares_memory(full, self.ring.slots))
        self.assertTrue(partial["values"].flags.owndata)
        full["values"][0] = 99
        partial["values"][0] = 88
        self.assertEqual(self.ring.read()[0]["values"][0], 1)
        for sequence in (2, 3, 4):
            self.publish(sequence)
        self.assertEqual(full["values"][0], 99)
        self.assertEqual(partial["values"][0], 88)
        np.testing.assert_array_equal(full["values"][1:], np.ones(3))

    def test_incomplete_read_does_not_return_previous_sample(self):
        self.publish(1)
        self.assertIsNotNone(self.ring.read())
        self.ring.slots[1]["sequence"] = 7
        self.assertIsNone(self.ring.read())
        self.assertIsNone(self.ring.read(fields=("values",)))

    def test_overwrite_during_copy_is_rejected(self):
        self.publish(1)

        def fields():
            yield "values"
            for sequence in (2, 3, 4):
                self.publish(sequence)
            yield "mirror"

        self.assertIsNone(self.ring.read(fields=fields()))
        self.assertEqual(self.ring.read()[1], 4)

    def test_invalid_fields_do_not_publish(self):
        self.publish(1)
        valid = dict(frame_id=2, values=np.ones(4), mirror=-np.ones(4))
        invalid = (
            {name: value for name, value in valid.items() if name != "mirror"},
            {**valid, "extra": 0},
            {**valid, "values": np.ones(1)},
            dict(frame_id=2, values=np.ones(4), unknown=np.ones(4)),
        )
        for fields in invalid:
            with self.subTest(fields=tuple(fields)):
                with self.assertRaises(ValueError):
                    self.ring.write(**fields)
                self.assertEqual(self.ring.latest_sequence, 1)
                self.assertEqual(int(self.ring.read()[0]["frame_id"]), 1)

    def test_assignment_error_is_not_published(self):
        self.publish(1)
        with self.assertRaises(ValueError):
            self.ring.write(frame_id=2, values=np.full(4, "bad"), mirror=np.zeros(4))
        self.assertEqual(self.ring.latest_sequence, 1)
        self.assertIsNone(self.ring.read(sequence=2))
        self.assertEqual(int(self.ring.read()[0]["frame_id"]), 1)
        self.assertEqual(self.publish(2), 2)

    def test_payload_requires_fields_without_objects(self):
        for dtype in (np.dtype("f4"), np.dtype([("nested", [("value", object)])])):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                SharedRing(self.ctx, dtype, 3)

    def test_spawn_single_writer_multiple_readers(self):
        dtype = np.dtype(
            [("frame_id", "<u8"), ("values", "<f4", (65536,)),
             ("mirror", "<f4", (65536,))],
            align=True,
        )
        ring = SharedRing(self.ctx, dtype, 3)
        self.assertIsNone(ring.read())
        barrier = self.ctx.Barrier(3, timeout=10)
        finished = self.ctx.Event()
        counts = self.ctx.RawArray("I", 2)
        processes = [
            self.ctx.Process(target=publish_patterns, args=(ring, barrier, finished)),
            *(self.ctx.Process(
                target=read_patterns, args=(ring, barrier, finished, counts, reader),
            ) for reader in range(2)),
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
                self.assertEqual(process.exitcode, 0)
            self.assertTrue(all(count > 0 for count in counts))
            self.assertEqual(ring.read()[1], 300)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()


class SensorChannelsTests(unittest.TestCase):
    def test_observation_matches_cloud_to_source_camera(self):
        channels = SensorChannels(
            mp.get_context("spawn"), camera=CameraParams(height=2, width=3),
            vr=True, pointcloud_num_points=4,
        )
        camera = np.zeros((), dtype=channels.camera_ring.dtype)
        camera["timestamp_ns"] = 101
        camera["color_frame_number"] = 10
        camera["rgb"] = 7
        camera["depth"] = 1000
        camera_sequence = publish_record(channels.camera_ring, camera)

        cloud = np.zeros((), dtype=channels.pointcloud_ring.dtype)
        cloud["source_camera_sequence"] = camera_sequence
        cloud["timestamp_ns"] = 101
        cloud["point_cloud"] = 3
        publish_record(channels.pointcloud_ring, cloud)

        vr = np.zeros((), dtype=channels.vr_ring.dtype)
        vr["recv_ts_ns"] = 99
        vr["head_recv_ts_ns"] = 90
        vr["sequence_id"] = 12
        publish_record(channels.vr_ring, vr)

        camera["rgb"] = 8
        camera["timestamp_ns"] = 201
        publish_record(channels.camera_ring, camera)
        arm = np.array([(100,)], dtype=[("timestamp_ns", "<u8")])
        robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=arm, hand=None))
        needs = ObservationNeeds(camera="rgb", point_cloud=True, vr=True)
        row = read_observation(channels, robot, needs=needs)
        self.assertEqual(row.camera["ring_sequence"], camera_sequence)
        self.assertEqual(row.camera["timestamp_ns"], 101)
        self.assertEqual(row.camera["color_frame_number"], 10)
        np.testing.assert_array_equal(row.camera["rgb"], np.full((2, 3, 3), 7))
        self.assertNotIn("depth", row.camera)
        self.assertTrue(row.camera["rgb"].flags.owndata)
        self.assertFalse(row.camera["rgb"].flags.writeable)
        self.assertEqual(row.pointcloud_timestamp_ns, 101)
        np.testing.assert_array_equal(row.point_cloud, np.full((4, 6), 3))
        self.assertFalse(row.point_cloud.flags.writeable)
        self.assertEqual(row.vr["recv_ts_ns"], 99)
        self.assertEqual(row.vr["head_recv_ts_ns"], 90)
        self.assertEqual(row.vr["sequence_id"], 12)
        self.assertEqual(row.vr["ring_sequence"], 1)
        runtime = SimpleNamespace(
            arm=SimpleNamespace(feedback_max_age_s=1.0), camera=CameraParams(),
            teleop=SimpleNamespace(vr_stale_threshold_s=0.5),
        )
        issues, deadline = check_observation(row, runtime, needs, require_hand=False, now_ns=200)
        self.assertEqual(issues, [])
        self.assertGreater(deadline, 200)

        for _ in range(4):
            publish_record(channels.camera_ring, camera)
        self.assertIsNone(read_observation(channels, robot, needs=needs).camera)
        channels.pointcloud_ring.slots[1]["sequence"] = 3
        self.assertIsNone(read_observation(channels, robot, needs=needs).point_cloud)
        channels.vr_ring.slots[1]["sequence"] = 3
        self.assertIsNone(read_vr_frame(channels))
        row = read_observation(channels, robot, needs=needs)
        issues, deadline = check_observation(row, runtime, needs, require_hand=False, now_ns=200)
        self.assertEqual(issues, [
            "camera_source_unavailable_or_unmatched", "pointcloud_unavailable_or_nonfinite",
            "vr_unavailable",
        ])
        self.assertIsNone(deadline)


class SensorWorkerTests(unittest.TestCase):
    def setUp(self):
        self.config = CameraParams(height=2, width=3)
        self.channels = SensorChannels(
            mp.get_context("spawn"), camera=self.config, pointcloud_num_points=4,
        )

    def run_camera(self, frames):
        frames = list(frames)
        pending = iter(enumerate(frames))

        def read(timeout_ms):
            index, frame = next(pending)
            if index == len(frames) - 1:
                self.channels.is_running.value = False
            return frame

        camera = Mock(active_serial="camera-test")
        camera.get_depth_scale.return_value = 0.001
        camera.get_color_intrinsics.return_value.to_dict.return_value = {"width": 3, "height": 2}
        camera.read.side_effect = read
        self.addCleanup(camera.disconnect.assert_called_once)
        with patch.dict("sys.modules", {
            "dexmani_real.sensor.camera.realsense": SimpleNamespace(
                RealSenseCamera=Mock(return_value=camera),
            ),
        }), patch("dexmani_real.sensor.camera.worker.configure_logging"):
            run_camera_worker(self.channels, self.config)
        self.assertFalse(self.channels.camera_ready.is_set())

    def test_camera_preserves_first_receive_times(self):
        start = time.monotonic_ns() - 1_000_000
        self.run_camera(
            SimpleNamespace(
                depth_frame_number=depth, color_frame_number=color, timestamp_ns=start + index,
                rgb=np.full((2, 3, 3), color, dtype=np.uint8),
                depth_aligned_to_color_raw=np.full((2, 3), depth, dtype=np.uint16),
            )
            for index, (depth, color) in enumerate(((1, 10), (1, 10), (2, 10), (2, 11)))
        )
        self.assertEqual(self.channels.camera_ring.latest_sequence, 3)
        for sequence, stamp, color in ((1, start, 10), (2, start, 10), (3, start + 2, 11)):
            data, _ = self.channels.camera_ring.read(sequence=sequence)
            self.assertEqual(data["timestamp_ns"], stamp)
            self.assertEqual(data["color_frame_number"], color)
            np.testing.assert_array_equal(data["rgb"], np.full((2, 3, 3), color))
        self.assertEqual(self.channels.camera_serial.value, b"camera-test")
        self.assertEqual(self.channels.camera_depth_scale.value, 0.001)
        self.assertEqual(json.loads(self.channels.camera_intrinsics.value), {"width": 3, "height": 2})

    def test_camera_rejects_malformed_pixels_and_releases_device(self):
        frame = SimpleNamespace(
            depth_frame_number=1, color_frame_number=1, timestamp_ns=time.monotonic_ns(),
            rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            depth_aligned_to_color_raw=np.zeros((2, 3), dtype=np.uint16),
        )
        with self.assertLogs("dexmani_real.sensor.camera.worker", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "shape"):
                self.run_camera([frame])
        self.assertEqual(self.channels.camera_ring.latest_sequence, 0)
        self.assertFalse(self.channels.camera_ready.is_set())

    def test_pointcloud_processes_each_camera_frame_once(self):
        from dexmani_real.perception import worker

        frame = np.zeros((), dtype=self.channels.camera_ring.dtype)
        frame["timestamp_ns"] = 101
        frame["rgb"] = 1
        frame["depth"] = 1000
        publish_record(self.channels.camera_ring, frame)
        cloud = np.full((4, 6), 3, dtype=np.float32)

        def next_frame(delay):
            frame["timestamp_ns"] = 201
            frame["rgb"] = 2
            publish_record(self.channels.camera_ring, frame)

        def build_cloud(**inputs):
            self.assertLessEqual(build.call_count, 2)
            np.testing.assert_array_equal(inputs["depth_raw"], np.full((2, 3), 1000))
            if inputs["color"][0, 0, 0] == 1:
                return None
            self.channels.is_running.value = False
            return cloud

        camera = SimpleNamespace(T_xarm_base_from_color=np.eye(4), depth_scale_m=0.001, color=None)
        with (
            patch.object(worker, "configure_logging"),
            patch.object(worker.time, "sleep", side_effect=next_frame) as idle,
            patch.object(worker, "build_point_cloud", side_effect=build_cloud) as build,
        ):
            worker.run_pointcloud_worker(self.channels, SimpleNamespace(num_points=4), camera, None)
        self.assertEqual(build.call_count, 2)
        idle.assert_called_once()
        data, sequence = self.channels.pointcloud_ring.read()
        self.assertEqual(sequence, 1)
        self.assertEqual(data["source_camera_sequence"], 2)
        self.assertEqual(data["timestamp_ns"], 201)
        np.testing.assert_array_equal(data["point_cloud"], cloud)
        self.assertFalse(self.channels.pointcloud_ready.is_set())

        robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=np.zeros(7), hand=None))
        row = read_observation(
            self.channels, robot, needs=ObservationNeeds(camera="rgbd", point_cloud=True),
        )
        self.assertEqual(row.camera["ring_sequence"], 2)
        self.assertEqual(row.camera["timestamp_ns"], row.pointcloud_timestamp_ns)
        np.testing.assert_array_equal(row.point_cloud, cloud)


if __name__ == "__main__":
    unittest.main()
