"""Backend-neutral, validated policy runtime contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from dexmani_real.ipc.schema import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

PaddingMode = Literal["invalid_zero", "invalid_nan"]


def _readonly_array(
    value: Any,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    name: str,
    allow_nan: bool = False,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    invalid = np.isinf(array) if allow_nan else ~np.isfinite(array)
    if np.any(invalid):
        raise ValueError(f"{name} contains NaN/Inf")
    result = np.array(array, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    clock: Literal["host_monotonic", "mapped_device_monotonic"] = "host_monotonic"
    history_length: int = 1
    max_age_s: float = 0.25
    max_skew_s: float = 0.10
    padding: PaddingMode = "invalid_zero"
    producer_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if not self.name or any(dim <= 0 for dim in self.shape):
            raise ValueError("modality name and shape must be non-empty/positive")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "biuf":
            raise ValueError("modality dtype must be a real numeric or boolean NumPy dtype")
        if self.clock not in {"host_monotonic", "mapped_device_monotonic"}:
            raise ValueError("unsupported modality clock")
        if self.padding not in {"invalid_zero", "invalid_nan"}:
            raise ValueError("unsupported modality padding mode")
        if self.padding == "invalid_nan" and not np.issubdtype(dtype, np.floating):
            raise ValueError("invalid_nan padding requires a floating-point modality dtype")
        if (
            self.history_length <= 0
            or not np.isfinite(self.max_age_s)
            or self.max_age_s <= 0
            or not np.isfinite(self.max_skew_s)
            or self.max_skew_s < 0
        ):
            raise ValueError("invalid modality history/age/skew")
        if self.producer_hz is not None and (not np.isfinite(self.producer_hz) or self.producer_hz <= 0):
            raise ValueError("producer_hz must be finite and positive")

    @property
    def required_ring_capacity(self) -> int:
        rate = self.producer_hz or 1.0
        return max(self.history_length, int(np.ceil(rate * self.max_age_s)) + self.history_length + 1)


@dataclass(frozen=True)
class ObservationSpec:
    modalities: tuple[ModalitySpec, ...]
    control_hz: float = 16.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        names = [item.name for item in self.modalities]
        if not names or len(set(names)) != len(names):
            raise ValueError("ObservationSpec modality names must be non-empty and unique")

    def modality(self, name: str) -> ModalitySpec:
        for item in self.modalities:
            if item.name == name:
                return item
        raise KeyError(name)


@dataclass(frozen=True)
class ActionSpec:
    representation: Literal["joint_position"] = "joint_position"
    units: Literal["rad"] = "rad"
    frame: Literal["robot_joint"] = "robot_joint"
    arm_shape: tuple[int, ...] = ARM_JOINT_SHAPE
    hand_shape: tuple[int, ...] = HAND_JOINT_SHAPE
    chunk_length: int = 1
    dt_s: float = 1.0 / 16.0
    deadline_s: float = 0.20

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_shape", tuple(int(dim) for dim in self.arm_shape))
        object.__setattr__(self, "hand_shape", tuple(int(dim) for dim in self.hand_shape))
        if self.representation != "joint_position" or self.units != "rad" or self.frame != "robot_joint":
            raise ValueError("unsupported action representation/units/frame")
        if self.arm_shape != ARM_JOINT_SHAPE or self.hand_shape != HAND_JOINT_SHAPE:
            raise ValueError("DexMani joint action shapes must be (7,) and (12,)")
        if (
            self.chunk_length <= 0
            or not np.isfinite(self.dt_s)
            or not np.isfinite(self.deadline_s)
            or self.dt_s <= 0
            or self.deadline_s <= 0
        ):
            raise ValueError("invalid action chunk timing")


@dataclass(frozen=True)
class FrozenArrayMap(Mapping[str, np.ndarray]):
    _items: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        frozen_items: list[tuple[str, np.ndarray]] = []
        names: set[str] = set()
        for key, value in self._items:
            if not key or key in names:
                raise ValueError("FrozenArrayMap keys must be non-empty and unique")
            names.add(key)
            array = np.array(value, copy=True, order="C")
            array.flags.writeable = False
            frozen_items.append((key, array))
        object.__setattr__(self, "_items", tuple(frozen_items))

    @classmethod
    def validated(cls, values: Mapping[str, Any], spec: ObservationSpec) -> "FrozenArrayMap":
        items: list[tuple[str, np.ndarray]] = []
        for modality in spec.modalities:
            if modality.name not in values:
                raise ValueError(f"missing observation modality {modality.name!r}")
            shape = (modality.history_length,) + modality.shape
            items.append(
                (
                    modality.name,
                    _readonly_array(
                        values[modality.name],
                        shape,
                        modality.dtype,
                        name=modality.name,
                        allow_nan=modality.padding == "invalid_nan",
                    ),
                )
            )
        return cls(tuple(items))

    def __getitem__(self, key: str) -> np.ndarray:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class ObservationSnapshot:
    observation_id: int
    anchor_monotonic_ns: int
    values: FrozenArrayMap
    source_monotonic_ns: FrozenArrayMap
    publish_monotonic_ns: FrozenArrayMap
    valid_history_mask: FrozenArrayMap
    session_generation: int
    camera_generation: int = 0
    receive_monotonic_ns: FrozenArrayMap = field(default_factory=lambda: FrozenArrayMap(()))
    source_age_s: FrozenArrayMap = field(default_factory=lambda: FrozenArrayMap(()))
    source_skew_s: FrozenArrayMap = field(default_factory=lambda: FrozenArrayMap(()))

    def __post_init__(self) -> None:
        if self.observation_id <= 0 or self.anchor_monotonic_ns <= 0 or self.session_generation < 0:
            raise ValueError("invalid observation identity/timing")
        if self.camera_generation < 0:
            raise ValueError("camera_generation must be non-negative")
        names = set(self.values)
        for label, mapping in (
            ("source_monotonic_ns", self.source_monotonic_ns),
            ("publish_monotonic_ns", self.publish_monotonic_ns),
            ("valid_history_mask", self.valid_history_mask),
        ):
            if set(mapping) != names:
                raise ValueError(f"{label} modalities do not match observation values")
        for label, mapping in (
            ("receive_monotonic_ns", self.receive_monotonic_ns),
            ("source_age_s", self.source_age_s),
            ("source_skew_s", self.source_skew_s),
        ):
            if len(mapping) and set(mapping) != names:
                raise ValueError(f"{label} modalities do not match observation values")
        for name in names:
            value = self.values[name]
            if value.ndim < 1:
                raise ValueError(f"observation modality {name!r} has no history dimension")
            history_shape = (value.shape[0],)
            source = self.source_monotonic_ns[name]
            publish = self.publish_monotonic_ns[name]
            valid = np.asarray(self.valid_history_mask[name], dtype=bool)
            if source.shape != history_shape or publish.shape != history_shape or valid.shape != history_shape:
                raise ValueError(f"observation timing shape mismatch for modality {name!r}")
            if np.any(valid & ~np.all(np.isfinite(value), axis=tuple(range(1, value.ndim)))):
                raise ValueError(f"valid observation contains NaN/Inf for modality {name!r}")
            if np.any(valid & ((source == 0) | (publish == 0) | (publish < source))):
                raise ValueError(f"valid observation timing is malformed for modality {name!r}")
            if len(self.receive_monotonic_ns):
                receive = self.receive_monotonic_ns[name]
                if receive.shape != history_shape or np.any(valid & ((receive < source) | (receive > publish))):
                    raise ValueError(f"observation receive time is malformed for modality {name!r}")
            for label, mapping in (("age", self.source_age_s), ("skew", self.source_skew_s)):
                if not len(mapping):
                    continue
                timing = np.asarray(mapping[name], dtype=np.float64)
                if timing.shape != history_shape or np.any(valid & (~np.isfinite(timing) | (timing < 0))):
                    raise ValueError(f"observation {label} is malformed for modality {name!r}")


@dataclass(frozen=True)
class ActionCandidate:
    observation_id: int
    session_generation: int
    policy_epoch: int
    action_id: int
    created_monotonic_ns: int
    target_monotonic_ns: int
    valid_until_monotonic_ns: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    representation: str = "joint_position"
    units: str = "rad"
    frame: str = "robot_joint"
    chunk_id: int = 0
    step_index: int = 0
    is_hold: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.observation_id,
                self.action_id,
                self.created_monotonic_ns,
                self.target_monotonic_ns,
                self.valid_until_monotonic_ns,
            )
            <= 0
        ):
            raise ValueError("action identifiers/timestamps must be positive")
        if self.created_monotonic_ns > self.target_monotonic_ns:
            raise ValueError("action target precedes creation")
        if self.target_monotonic_ns > self.valid_until_monotonic_ns:
            raise ValueError("action validity ends before target")
        if self.session_generation < 0 or self.policy_epoch < 0 or self.chunk_id < 0 or self.step_index < 0:
            raise ValueError("action generations/indices must be non-negative")
        if self.arm_qpos is None and self.hand_qpos is None:
            raise ValueError("action candidate controls no actuator")
        if self.arm_qpos is not None:
            object.__setattr__(
                self,
                "arm_qpos",
                _readonly_array(self.arm_qpos, ARM_JOINT_SHAPE, np.float64, name="arm_qpos"),
            )
        if self.hand_qpos is not None:
            object.__setattr__(
                self,
                "hand_qpos",
                _readonly_array(self.hand_qpos, HAND_JOINT_SHAPE, np.float64, name="hand_qpos"),
            )


@dataclass(frozen=True)
class ActionChunk:
    chunk_id: int
    steps: tuple[ActionCandidate, ...]

    def __post_init__(self) -> None:
        if self.chunk_id <= 0 or not self.steps:
            raise ValueError("chunk must have a positive ID and at least one step")
        if any(step.chunk_id != self.chunk_id for step in self.steps):
            raise ValueError("all action steps must carry the enclosing chunk ID")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("action chunk step_index values must be contiguous from zero")
        if any(a.target_monotonic_ns >= b.target_monotonic_ns for a, b in zip(self.steps, self.steps[1:])):
            raise ValueError("action chunk target times must be strictly increasing")
