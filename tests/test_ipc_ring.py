import multiprocessing as mp

import numpy as np
import pytest

from dexmani_real.ipc.ring import SharedRing
from dexmani_real.ipc.sensors import camera_dtype


@pytest.mark.parametrize("fields", [None, ("timestamp_ns", "rgb", "depth")])
def test_snapshot_reuse_and_slot_overwrite(fields):
    ring = SharedRing(mp.get_context("spawn"), camera_dtype(4, 4), 1)
    assert ring.read(fields=fields) is None
    rgb = np.full((4, 4, 3), 17, np.uint8)
    depth = np.full((4, 4), 500, np.uint16)
    ring.write(timestamp_ns=100, rgb=rgb, depth=depth)
    first = ring.read(fields=fields)
    assert ring.read(fields=fields) is first
    if fields is not None:
        assert ring.read(fields=tuple(reversed(fields))) is first
    ring.write(timestamp_ns=200, rgb=rgb + 1, depth=depth + 1)
    second = ring.read(fields=fields)
    assert second is not first and second[1] == 2
    assert int(first[0]["timestamp_ns"]) == 100
    np.testing.assert_array_equal(first[0]["rgb"], rgb)
    np.testing.assert_array_equal(second[0]["depth"], depth + 1)
    assert not np.shares_memory(second[0]["rgb"], ring.slots)


def test_fields_and_incomplete_publication():
    ring = SharedRing(mp.get_context("spawn"), camera_dtype(4, 4), 1)
    values = dict(timestamp_ns=100, rgb=np.zeros((4, 4, 3), np.uint8),
                  depth=np.ones((4, 4), np.uint16))
    ring.write(**values)
    rgb = ring.read(fields=("rgb",))
    depth = ring.read(fields=("depth",))
    assert set(rgb[0]) == {"rgb"} and set(depth[0]) == {"depth"}
    full = ring.read()
    assert isinstance(full[0], np.void)
    # 已发布序列所指的槽正在被覆盖：只能回退完整缓存。
    ring.write(**values)
    ring.slots[0]["sequence"] = 5
    assert ring.read() is full
    assert ring.read(fields=("rgb",)) is None


def test_marker_change_during_copy(monkeypatch):
    ring = SharedRing(mp.get_context("spawn"), camera_dtype(4, 4), 1)
    ring.write(timestamp_ns=100, rgb=np.zeros((4, 4, 3), np.uint8),
               depth=np.ones((4, 4), np.uint16))
    previous = ring.read()
    ring.write(timestamp_ns=200, rgb=np.ones((4, 4, 3), np.uint8),
               depth=np.ones((4, 4), np.uint16))
    slots = ring.slots

    class RacingPayload:
        def copy(self):
            copied = slots[0]["data"].copy()
            slots[0]["sequence"] = 5
            return copied

    class Slot:
        def __getitem__(self, key):
            return RacingPayload() if key == "data" else slots[0][key]

    monkeypatch.setattr(SharedRing, "slots", property(lambda self: [Slot()]))
    assert ring.read() is previous
    assert int(previous[0]["timestamp_ns"]) == 100
