import numpy as np
from numpy.typing import DTypeLike


class SharedRing:
    """Linux x86_64 上单写多读，以奇偶序列标记保护最新数据副本。

    槽仅为复制竞争提供缓冲，不提供可查询历史或跨平台无锁保证。"""

    def __init__(self, ctx, dtype: DTypeLike, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("ring capacity must be a positive integer")
        self.dtype = np.dtype(dtype)
        self.capacity = capacity
        self._last_snapshot: tuple[dict, int] | None = None
        names = self.dtype.names
        if names is None or self.dtype.hasobject:
            raise ValueError("shared-memory payload requires fields without Python objects")
        self.field_names = frozenset(names)
        self.slot_dtype = np.dtype(
            [
                ("sequence", "<u8"),
                ("data", self.dtype),
            ],
            align=True,
        )
        self.buffer = ctx.RawArray("b", capacity * self.slot_dtype.itemsize)
        self._sequence = ctx.RawValue("Q", 0)

    @property
    def latest_sequence(self) -> int:
        return self._sequence.value

    @property
    def slots(self):
        # 各进程按需构造视图，避免 spawn 序列化 ndarray 时复制共享数据。
        return np.frombuffer(self.buffer, dtype=self.slot_dtype, count=self.capacity)

    def write(self, **fields) -> int:
        if fields.keys() != self.field_names:
            raise ValueError("ring fields do not match the payload dtype")
        sequence = self.latest_sequence + 1
        slot = self.slots[sequence % self.capacity]
        slot["sequence"] = 2 * sequence - 1
        payload = slot["data"]
        for name, value in fields.items():
            payload[name] = value
        slot["sequence"] = 2 * sequence
        self._sequence.value = sequence
        return sequence

    def read(self, *, fields=None) -> tuple[dict, int] | None:
        """返回独立的字段字典及来源序列号；数组只读，字段子集复用已复制的完整快照。"""
        field_key = self.field_names if fields is None else frozenset(fields)
        unknown = field_key - self.field_names
        if unknown:
            raise ValueError(f"unknown ring fields: {sorted(unknown, key=repr)}")
        snapshot = self._last_snapshot
        for _ in range(2):
            sequence = self.latest_sequence
            if sequence <= 0:
                break
            if (
                snapshot is not None
                and sequence == snapshot[1]
                and field_key <= snapshot[0].keys()
            ):
                break
            slot = self.slots[sequence % self.capacity]
            marker = 2 * sequence
            if int(slot["sequence"]) != marker:
                continue
            payload = slot["data"]
            data = {}
            for name in field_key:
                value = np.array(payload[name], copy=True)
                value.flags.writeable = False
                data[name] = value if value.ndim else value[()]
            if marker == int(slot["sequence"]):
                self._last_snapshot = snapshot = (data, sequence)
                break
        if snapshot is not None and field_key <= snapshot[0].keys():
            return {name: snapshot[0][name] for name in field_key}, snapshot[1]
        return None
