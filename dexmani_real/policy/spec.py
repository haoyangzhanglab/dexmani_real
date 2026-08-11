"""Small, explicit configuration contract for one deployed policy."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from dexmani_real.policy.runtime import ActionSpec, ModalitySpec, ObservationSpec


@dataclass(frozen=True)
class PolicyResource:
    """One immutable model resource verified before workers are spawned."""

    name: str
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or len(self.sha256) != 64:
            raise ValueError("policy resource requires a name and SHA-256")


@dataclass(frozen=True)
class PolicySpec:
    """Everything the inference child needs to load one concrete policy."""

    adapter_module: str
    observation: ObservationSpec
    action: ActionSpec
    resources: tuple[PolicyResource, ...] = ()
    actuators: tuple[str, ...] = ("arm", "hand")
    poll_hz: float = 256.0
    benchmark_deadline_s: float = 0.20
    hardware_deployable: bool = False
    source_path: Path | None = None
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.adapter_module or ":" in self.adapter_module:
            raise ValueError("adapter_module must name a Python module, not a class entrypoint")
        if not self.actuators or "arm" not in self.actuators or set(self.actuators) - {"arm", "hand"}:
            raise ValueError("policy actuators must contain arm and may contain hand")
        if not np.isfinite(self.poll_hz) or self.poll_hz <= 0:
            raise ValueError("policy poll_hz must be finite and positive")
        if not np.isfinite(self.benchmark_deadline_s) or self.benchmark_deadline_s <= 0:
            raise ValueError("policy benchmark_deadline_s must be finite and positive")
        if not isinstance(self.hardware_deployable, bool):
            raise TypeError("policy hardware_deployable must be boolean")
        expected_dt_s = 1.0 / self.observation.control_hz
        if not np.isclose(self.action.dt_s, expected_dt_s, rtol=0.0, atol=1e-12):
            raise ValueError("policy action.dt_s must equal 1 / observation control_hz")

    @property
    def resource_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple((resource.name, resource.sha256) for resource in self.resources)

    def resource_path(self, name: str) -> Path:
        for resource in self.resources:
            if resource.name == name:
                return resource.path
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PolicySpec":
        spec_path = Path(path).resolve()
        raw = spec_path.read_bytes()
        decoded = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise TypeError("policy YAML root must be a mapping")

        modalities_raw = decoded.get("observations")
        if not isinstance(modalities_raw, list) or not modalities_raw:
            raise ValueError("policy YAML requires a non-empty observations list")
        modalities: list[ModalitySpec] = []
        for item in modalities_raw:
            if not isinstance(item, Mapping):
                raise TypeError("each policy observation must be a mapping")
            values: dict[str, Any] = dict(item)
            if "shape" in values:
                values["shape"] = tuple(values["shape"])
            modalities.append(ModalitySpec(**values))
        observation = ObservationSpec(tuple(modalities), control_hz=float(decoded.get("control_hz", 16.0)))

        action_raw = decoded.get("action") or {}
        if not isinstance(action_raw, Mapping):
            raise TypeError("policy action must be a mapping")
        action_values = dict(action_raw)
        action_values.setdefault("dt_s", 1.0 / observation.control_hz)
        for shape_name in ("arm_shape", "hand_shape"):
            if shape_name in action_values:
                action_values[shape_name] = tuple(action_values[shape_name])
        action = ActionSpec(**action_values)

        resources_raw = decoded.get("resources") or {}
        if not isinstance(resources_raw, Mapping):
            raise TypeError("policy resources must be a mapping")
        resources: list[PolicyResource] = []
        for name, item in sorted(resources_raw.items()):
            if not isinstance(item, Mapping) or "path" not in item or "sha256" not in item:
                raise ValueError(f"resource {name!r} requires path and sha256")
            resource_path = (spec_path.parent / str(item["path"])).resolve()
            expected = str(item["sha256"]).lower()
            actual = hashlib.sha256(resource_path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"resource {name!r} SHA-256 mismatch")
            resources.append(PolicyResource(str(name), resource_path, actual))

        hardware_deployable = decoded.get("hardware_deployable", False)
        if not isinstance(hardware_deployable, bool):
            raise TypeError("policy hardware_deployable must be boolean")

        return cls(
            adapter_module=str(decoded["adapter_module"]),
            observation=observation,
            action=action,
            resources=tuple(resources),
            actuators=tuple(str(name) for name in decoded.get("actuators", ("arm", "hand"))),
            poll_hz=float(decoded.get("poll_hz", 256.0)),
            benchmark_deadline_s=float(decoded.get("benchmark_deadline_s", action.deadline_s)),
            hardware_deployable=hardware_deployable,
            source_path=spec_path,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
