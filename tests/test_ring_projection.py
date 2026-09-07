"""Offline projected history tests using real shared memory and seqlocks."""

import unittest
import uuid
from unittest import mock

import numpy as np

from dexmani_real.ipc.ring import SeqlockSlot, SharedMemoryRingBuffer
from dexmani_real.ipc.schema import HAND_TACTILE_DTYPE

_FIELDS = ("source_monotonic_ns", "fresh", "calibrated", "unit_code")


class TestRingProjection(unittest.TestCase):
    def setUp(self):
        self.ring = SharedMemoryRingBuffer(
            f"tactile_projection_{uuid.uuid4().hex}", HAND_TACTILE_DTYPE, maxlen=3
        )
        self.addCleanup(self.ring.unlink)
        self.addCleanup(self.ring.close)

    def _write(self, source):
        record = np.zeros(1, dtype=HAND_TACTILE_DTYPE)
        record["source_monotonic_ns"] = source
        record["fresh"] = 1
        record["calibrated"] = 1
        record["tactile_force"] = source
        return self.ring.write(record)

    def test_metadata_values_order_and_ownership(self):
        self.assertEqual(self.ring.get_last_k_fields(3, _FIELDS), [])
        for source in range(1, 5):
            self._write(source)
        frames = self.ring.get_last_k_fields(3, _FIELDS)
        self.assertEqual([sequence for _, _, sequence in frames], [2, 3, 4])
        for record, timestamp, sequence in frames:
            self.assertEqual(tuple(record), _FIELDS)
            self.assertNotIn("tactile_force", record)
            self.assertEqual(record["source_monotonic_ns"], sequence)
            self.assertEqual(record["fresh"], 1)
            self.assertEqual(record["calibrated"], 1)
            self.assertEqual(record["unit_code"], 0)
            self.assertGreater(timestamp, 0)
        for source in range(5, 8):
            self._write(source)
        self.assertEqual([record["source_monotonic_ns"] for record, _, _ in frames],
                         [2, 3, 4])
        array_frames = self.ring.get_last_k_fields(1, ("tactile_force",))
        owned = array_frames[0][0]["tactile_force"]
        self.assertTrue(owned.flags.owndata)
        owned[:] = -1
        self.assertTrue(np.all(self.ring.get_last_k(1)[0][0]["tactile_force"] == 7))

    def test_invalid_fields_and_history_limits(self):
        for fields in ((), ("fresh", "fresh"), ("missing",)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.ring.get_last_k_fields(1, fields)
        for count in (0, -1):
            self.assertEqual(self.ring.get_last_k_fields(count, _FIELDS), [])
        with self.assertRaises(ValueError):
            self.ring.get_last_k_fields(4, _FIELDS)

    def test_incomplete_overwritten_and_torn_slots_are_dropped(self):
        self._write(1)
        slot = self.ring._data_buf[1]
        for marker in (0, 3, 8):
            slot["sequence"] = marker
            self.assertEqual(self.ring.get_last_k_fields(1, _FIELDS), [])
            self.assertEqual(self.ring.get_last_k(1), [])
        slot["sequence"] = 2
        with mock.patch.object(SeqlockSlot, "verify", return_value=False) as verify:
            self.assertEqual(self.ring.get_last_k_fields(1, _FIELDS), [])
            self.assertEqual(verify.call_count, 2)
        original_verify = SeqlockSlot.verify

        def overwrite_during_copy(seqlock, marker):
            slot["sequence"] = 8
            return original_verify(seqlock, marker)

        with mock.patch.object(SeqlockSlot, "verify", overwrite_during_copy):
            self.assertEqual(self.ring.get_last_k_fields(1, _FIELDS), [])


if __name__ == "__main__":
    unittest.main()
