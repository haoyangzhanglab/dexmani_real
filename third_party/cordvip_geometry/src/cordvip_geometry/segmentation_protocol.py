"""Length-delimited JSON metadata and contiguous arrays over private process pipes."""

import json
import math
import struct

import numpy as np

MAX_HEADER = 1024 * 1024
MAX_ARRAY_BYTES = 256 * 1024 * 1024
DTYPES = {"|b1", "|u1", "<i4", "<i8", "<f4", "<f8"}


def read_exact(stream, size):
    chunks = bytearray(size)
    view = memoryview(chunks)
    offset = 0
    while offset < size:
        count = stream.readinto(view[offset:])
        if not count:
            raise EOFError("Segmentation worker pipe closed during a message")
        offset += count
    return chunks


def send(stream, metadata, arrays=None):
    arrays = {key: np.ascontiguousarray(value) for key, value in (arrays or {}).items()}
    specs = {}
    for key, array in arrays.items():
        if array.dtype.str not in DTYPES or array.nbytes > MAX_ARRAY_BYTES:
            raise ValueError(f"Unsupported IPC array: {key} {array.dtype} {array.shape}")
        specs[key] = dict(dtype=array.dtype.str, shape=list(array.shape))
    if sum(array.nbytes for array in arrays.values()) > MAX_ARRAY_BYTES:
        raise ValueError("IPC payload exceeds bounded message size")
    header = json.dumps(dict(metadata=metadata, arrays=specs), allow_nan=False).encode("utf-8")
    if len(header) > MAX_HEADER:
        raise ValueError("IPC header too large")
    stream.write(struct.pack("!I", len(header)))
    stream.write(header)
    for array in arrays.values():
        stream.write(array.tobytes())
    stream.flush()


def receive(stream):
    (size,) = struct.unpack("!I", read_exact(stream, 4))
    if not 0 < size <= MAX_HEADER:
        raise ValueError("Invalid IPC header size")
    header = json.loads(read_exact(stream, size))
    arrays, total = {}, 0
    for key, spec in header["arrays"].items():
        shape, dtype = spec["shape"], spec["dtype"]
        if (
            dtype not in DTYPES
            or not isinstance(shape, list)
            or len(shape) > 4
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise ValueError("Invalid IPC array specification")
        size = math.prod(shape) * np.dtype(dtype).itemsize
        total += size
        if total > MAX_ARRAY_BYTES:
            raise ValueError("IPC payload exceeds bounded message size")
        arrays[key] = np.frombuffer(read_exact(stream, size), dtype=dtype).reshape(shape)
    return header["metadata"], arrays
