from __future__ import annotations

import unittest
import uuid

import numpy as np

from dexmani_real.ipc.camera_ring import CameraRingBuffer
from dexmani_real.ipc.channels import RuntimeChannels, RuntimeChannelsConfig
from dexmani_real.ipc.ring import SharedMemoryRingBuffer
from dexmani_real.ipc.schema import CAMERA_FRAME_HEADER_DTYPE, VR_FRAME_DTYPE


def unique_name(label: str) -> str:
    return f"dexmani_test_{label}_{uuid.uuid4().hex}"


class SharedMemoryRingBufferTest(unittest.TestCase):
    def test_write_requires_exact_shape_and_dtype(self) -> None:
        dtype = np.dtype([("value", "<f8", (2,))], align=True)
        ring = SharedMemoryRingBuffer(unique_name("ring"), dtype, maxlen=2)
        try:
            valid = np.zeros(1, dtype=dtype)
            valid["value"][0] = (1.0, 2.0)
            sequence = ring.write(valid)
            self.assertEqual(sequence, 1)
            latest = ring.read_latest()
            assert latest is not None
            np.testing.assert_array_equal(latest[0], valid)

            with self.assertRaises(ValueError):
                ring.write(np.zeros((), dtype=dtype))
            with self.assertRaises(ValueError):
                ring.write(np.zeros(2, dtype=dtype))
            with self.assertRaises(ValueError):
                ring.write(np.zeros(1, dtype=np.dtype([("value", "<f4", (2,))])))
            with self.assertRaises(ValueError):
                ring.write(np.zeros((1, 1), dtype=dtype))
        finally:
            ring.close()
            ring.unlink()

    def test_existing_segment_fails_closed(self) -> None:
        name = unique_name("ring_conflict")
        dtype = np.dtype([("value", "<f8")], align=True)
        owner = SharedMemoryRingBuffer(name, dtype, maxlen=2, create=True)
        try:
            with self.assertRaises(FileExistsError):
                SharedMemoryRingBuffer(name, dtype, maxlen=2, create=True)

            frame = np.zeros(1, dtype=dtype)
            frame["value"][0] = 3.0
            owner.write(frame)
            latest = owner.read_latest()
            assert latest is not None
            self.assertEqual(float(latest[0]["value"][0]), 3.0)
        finally:
            owner.unlink()
            owner.close()


class CameraRingBufferTest(unittest.TestCase):
    def test_existing_segment_fails_closed_without_disturbing_owner(self) -> None:
        name = unique_name("camera")
        owner = CameraRingBuffer(
            name,
            rgb_shape=(2, 3, 3),
            depth_shape=(2, 3),
            maxlen=2,
            create=True,
        )
        try:
            header = np.zeros(1, dtype=CAMERA_FRAME_HEADER_DTYPE)
            header["rgb_size"][0] = 18
            header["depth_size"][0] = 12
            header["rgb_shape_h"][0] = 2
            header["rgb_shape_w"][0] = 3
            header["rgb_shape_c"][0] = 3
            header["depth_shape_h"][0] = 2
            header["depth_shape_w"][0] = 3
            rgb = np.full((2, 3, 3), 11, dtype=np.uint8)
            depth = np.zeros((2, 3), dtype=np.uint16)
            self.assertEqual(owner.write(header, rgb, depth), 1)

            with self.assertRaises(FileExistsError):
                CameraRingBuffer(
                    name,
                    rgb_shape=(2, 3, 3),
                    depth_shape=(2, 3),
                    maxlen=2,
                    create=True,
                )

            latest = owner.read_latest()
            assert latest is not None
            self.assertEqual(latest[3], 1)
            self.assertEqual(int(latest[1][0, 0, 0]), 11)
        finally:
            owner.unlink()
            owner.close()


class RuntimeChannelsTest(unittest.TestCase):
    def test_initial_safety_state_must_be_disarmed(self) -> None:
        with self.assertRaisesRegex(ValueError, "DISARMED"):
            RuntimeChannelsConfig(initial_safety_state=1)

    def test_duplicate_prefix_fails_without_unlinking_live_runtime(self) -> None:
        prefix = unique_name("channels")
        owner = RuntimeChannels.create(
            prefix=prefix,
            config=RuntimeChannelsConfig(),
            camera_rgb_shape=(1, 1, 3),
            camera_depth_shape=(1, 1),
        )
        try:
            with self.assertRaises(FileExistsError):
                RuntimeChannels.create(
                    prefix=prefix,
                    config=RuntimeChannelsConfig(),
                    camera_rgb_shape=(1, 1, 3),
                    camera_depth_shape=(1, 1),
                )

            frame = np.zeros(1, dtype=VR_FRAME_DTYPE)
            frame["sequence_id"][0] = 7
            owner.vr_ring.write(frame)
            latest = owner.vr_ring.read_latest()
            assert latest is not None
            self.assertEqual(int(latest[0]["sequence_id"][0]), 7)
        finally:
            self.assertTrue(owner.close())


if __name__ == "__main__":
    unittest.main()
