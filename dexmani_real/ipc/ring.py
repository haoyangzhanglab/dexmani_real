import numpy as np


class SharedRing:
    """Linux x86_64 上单写多读，以奇偶序列标记保护数据副本。"""

    def __init__(self, ctx, dtype: np.dtype, capacity: int):
        self.dtype = np.dtype(dtype)
        self.capacity = capacity
        names = self.dtype.names
        if names is None or self.dtype.hasobject:
            raise ValueError("shared-memory payload requires fields without Python objects")
        self._field_names = frozenset(names)
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
        if fields.keys() != self._field_names:
            raise ValueError("ring fields do not match the payload dtype")
        for name, value in fields.items():
            shape = self.dtype[name].shape
            if np.shape(value) != shape:
                raise ValueError(f"ring field {name!r} must have shape {shape}")

        sequence = self.latest_sequence + 1
        slot = self.slots[sequence % self.capacity]
        slot["sequence"] = 2 * sequence - 1
        payload = slot["data"]
        for name, value in fields.items():
            payload[name] = value
        slot["sequence"] = 2 * sequence
        self._sequence.value = sequence
        return sequence

    def read(self, *, sequence=None, fields=None) -> tuple[np.void | dict, int] | None:
        sequence = self.latest_sequence if sequence is None else sequence
        if sequence <= 0:
            return None
        slot = self.slots[sequence % self.capacity]
        marker = 2 * sequence
        if int(slot["sequence"]) != marker:
            return None

        payload = slot["data"]
        if fields is None:
            data = payload.copy()
        else:
            # 单独复制字段，避免只读 RGB 时仍持有完整深度数据。
            data = {name: payload[name].copy() for name in fields}
        return (data, sequence) if marker == int(slot["sequence"]) else None
