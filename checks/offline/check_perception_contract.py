"""A14: perception config contract — desk-plane single source + align fail-closed.

Asserts ``CameraLoopConfig.from_runtime`` consumes the resolved
``environment.table.plane_abcd`` as the single desk-plane source (disabled
table → no desk removal), the production align default is ``depth_to_color``,
and ``camera_loop`` fails closed on ``align_mode="none"`` before touching
shared state or the camera SDK.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.config.defaults import camera, policy
from dexmani_real.sensor.camera_process import CameraLoopConfig, camera_loop


class _Boom:
    """Sentinel shared: any attribute access means the guard ran too late."""

    def __getattr__(self, name: str) -> None:
        raise AssertionError(
            f"camera_loop touched shared.{name} before the fail-closed align check"
        )


def _runtime(enabled: bool) -> SimpleNamespace:
    return SimpleNamespace(
        camera=camera,
        policy=policy,
        environment=SimpleNamespace(
            table=SimpleNamespace(plane_abcd=(0.0, 0.0, 1.0, -0.12), enabled=enabled)
        ),
    )


def main() -> int:
    # Desk plane resolved from the single source of truth.
    cfg = CameraLoopConfig.from_runtime(_runtime(enabled=True))
    assert cfg.desk_plane == (0.0, 0.0, 1.0, -0.12), cfg.desk_plane
    assert cfg.align_mode == "depth_to_color", cfg.align_mode

    # A disabled table means "no desk removal", not "re-read the JSON file".
    cfg_off = CameraLoopConfig.from_runtime(_runtime(enabled=False))
    assert cfg_off.desk_plane is None, cfg_off.desk_plane

    # align_mode="none" is the dangerous value: production pointcloud requires
    # aligned streams, so camera_loop must exit before connecting.
    camera_loop(_Boom(), CameraLoopConfig(align_mode="none"))  # must not raise

    print("check_perception_contract: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
