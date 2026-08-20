"""Small teleoperation view over the canonical typed runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from dexmani_real.config.runtime import ResolvedRuntimeConfig, resolve_runtime_config
from dexmani_real.planning.constants import XHAND_RIGHT_URDF_PATH


@dataclass(frozen=True)
class TeleopConfig:
    """Session-only values plus a reference to the canonical runtime snapshot.

    Every runtime value is read from one immutable source via
    ``config.runtime.<section>.<field>``; this dataclass carries only the
    session-only fields plus that reference.
    """

    runtime: ResolvedRuntimeConfig = field(default_factory=resolve_runtime_config)
    task_label: str = ""
    operator: str = ""
    hand_urdf_path: str = field(default_factory=lambda: str(XHAND_RIGHT_URDF_PATH))
    vr_transform_path: str = "dexmani_real/config/vr_transform.json"

    def __post_init__(self) -> None:
        if not self.hand_urdf_path:
            raise ValueError("hand_urdf_path must be non-empty")

    @classmethod
    def from_runtime(
        cls,
        runtime: ResolvedRuntimeConfig,
        *,
        task_label: str = "",
        operator: str = "",
        hand_urdf_path: str | None = None,
    ) -> "TeleopConfig":
        return cls(
            runtime=runtime,
            task_label=task_label,
            operator=operator,
            hand_urdf_path=(
                str(XHAND_RIGHT_URDF_PATH)
                if hand_urdf_path is None
                else hand_urdf_path
            ),
        )
