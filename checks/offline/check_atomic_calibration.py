"""A16: calibration/config artifacts are written atomically.

``atomic_json_dump`` must round-trip a JSON value, create missing parent
directories, and overwrite an existing target in place without leaving a
truncated or absent file.  Exercised through a temp directory only.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.recording.transaction import atomic_json_dump


def main() -> int:
    payload = {
        "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "name": "camera_color",
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Missing parent dirs are created on demand.
        target = root / "calib" / "extrinsics.json"
        atomic_json_dump(payload, target)
        assert target.is_file()
        with open(target, "r", encoding="utf-8") as stream:
            assert json.load(stream) == payload

        # Overwriting an existing artifact replaces it atomically.
        updated = {**payload, "name": "camera_depth"}
        atomic_json_dump(updated, target)
        with open(target, "r", encoding="utf-8") as stream:
            assert json.load(stream) == updated

        # No stray temp files are left behind in the target directory.
        leftovers = [p.name for p in target.parent.iterdir() if p.name != target.name]
        assert leftovers == [], leftovers

    print("check_atomic_calibration: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
