import numpy as np


class SharedRing:
    """Linux x86_64 上单写多读，以奇偶序列标记保护最新数据副本。

    槽仅为复制竞争提供缓冲，不提供可查询历史或跨平台无锁保证。"""

    def __init__(self, ctx, dtype: np.dtype, capacity: int):
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
        """返回独立的字段字典及来源序列号；调用方按只读使用，包括其中的数组。"""
        field_key = self.field_names if fields is None else frozenset(fields)
        snapshot = self._last_snapshot
        for _ in range(2):
            sequence = self.latest_sequence
            if sequence <= 0:
                break
            if (
                snapshot is not None
                and sequence == snapshot[1]
                and field_key == snapshot[0].keys()
            ):
                return snapshot
            slot = self.slots[sequence % self.capacity]
            marker = 2 * sequence
            if int(slot["sequence"]) != marker:
                continue
            payload = slot["data"]
            data = {name: payload[name].copy() for name in field_key}
            if marker == int(slot["sequence"]):
                self._last_snapshot = snapshot = (data, sequence)
                return snapshot
        if snapshot is not None and field_key == snapshot[0].keys():
            return snapshot
        return None
