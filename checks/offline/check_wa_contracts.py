"""Offline smoke check: WA* perception-contract fixes (no hardware).

Covers the two deterministic, device-free contracts introduced by the WA*
changes:

1. ``atomic_json_dump`` (``recording/transaction.py``) — writes fresh targets,
   atomically overwrites existing ones, creates parent dirs, leaves no temp
   file behind, and honours ``ensure_ascii``.
2. ``CameraLoopConfig.from_runtime`` (``sensor/camera_process.py``) — consumes
   the resolved ``environment.table.plane_abcd`` when the table is enabled and
   yields ``None`` when it is disabled (single desk-plane source of truth).

Run from the repo root:
    conda run -n real_robot python checks/offline/check_wa_contracts.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def _check_atomic_json_dump() -> None:
    from dexmani_real.recording.transaction import atomic_json_dump

    tmp = Path(tempfile.mkdtemp())
    target = tmp / "nested" / "cfg.json"

    atomic_json_dump({"a": 1.0, "b": 2.0}, target)
    assert target.exists(), "fresh write must create parent dirs and file"
    assert json.loads(target.read_text()) == {"a": 1.0, "b": 2.0}

    atomic_json_dump({"a": 9.9, "c": [1, 2, 3]}, target, ensure_ascii=False)
    assert json.loads(target.read_text()) == {"a": 9.9, "c": [1, 2, 3]}, "overwrite must replace content"

    leftovers = [p for p in target.parent.iterdir() if p.name.startswith(".cfg.json.tmp-")]
    assert leftovers == [], f"atomic_json_dump leaked temp files: {leftovers}"

    atomic_json_dump({"note": "轴向已标定"}, target, ensure_ascii=False)
    assert "轴向已标定" in target.read_text(), "ensure_ascii=False must preserve unicode"


def _check_camera_desk_plane() -> None:
    from dexmani_real.sensor.camera_process import CameraLoopConfig

    class _Cam:
        serial = "s"
        width = 640
        height = 480
        fps = 30
        align_mode = "depth_to_color"
        warmup_frames = 10
        pointcloud_num_points = 4096
        max_frame_age_s = 0.5

    class _Pol:
        control_hz = 16.0

    class _Table:
        enabled = True
        plane_abcd = (1.0, 2.0, 3.0, 4.0)

    class _Env:
        table = _Table()

    class _RT:
        camera = _Cam()
        policy = _Pol()
        environment = _Env()

    enabled = CameraLoopConfig.from_runtime(_RT())
    assert enabled.desk_plane == (1.0, 2.0, 3.0, 4.0), enabled.desk_plane

    _Table.enabled = False
    disabled = CameraLoopConfig.from_runtime(_RT())
    assert disabled.desk_plane is None, disabled.desk_plane

    assert CameraLoopConfig().desk_plane is None, "default config must not auto-supply a desk plane"


def main() -> int:
    _check_atomic_json_dump()
    _check_camera_desk_plane()
    print("OK: WA* contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
