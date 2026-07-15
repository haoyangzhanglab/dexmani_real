"""xArm controller error/warn code → human-readable title.

Thin wrapper over the code tables shipped inside the xArm Python SDK
(``xarm.core.config.x_code``), so the mapping always matches the installed
SDK version instead of a hand-maintained copy.  Degrades to ``"unknown"``
if the SDK (or that private module path) is unavailable.

Note: only CONTROLLER error/warn codes (``arm.error_code`` /
``get_err_warn_code()``) are decoded here.  SDK API return codes
(e.g. from ``set_servo_angle``) are a different namespace — do not
pass them to these functions.
"""

from __future__ import annotations

try:
    from xarm.core.config.x_code import ControllerErrorCodeMap, ControllerWarnCodeMap
except Exception:  # SDK absent or private module moved — degrade to bare codes
    ControllerErrorCodeMap = {}
    ControllerWarnCodeMap = {}


def _title(table: dict, code: int) -> str:
    entry = table.get(code)
    if isinstance(entry, dict):
        title = entry.get("en", {}).get("title")
        if title:
            return str(title)
    return "unknown"


def decode_error(code: int) -> str:
    """Human-readable title for a controller error code (C-code)."""
    if code == 0:
        return "no error"
    return _title(ControllerErrorCodeMap, code)


def decode_warn(code: int) -> str:
    """Human-readable title for a controller warn code."""
    if code == 0:
        return "no warning"
    return _title(ControllerWarnCodeMap, code)
