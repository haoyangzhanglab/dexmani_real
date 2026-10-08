"""Validation for experiment-facing teleoperation arguments."""


def validate_keyboard_workspace(runtime):
    """Validate the selected jog configuration before any device connects."""
    import numpy as np

    cfg = runtime.keyboard_teleop
    cfg.validate()
    workspace = runtime.policy.workspace.as_array()
    if not np.isfinite(workspace).all() or np.any(
        workspace[:, 0] + cfg.workspace_command_margin_m
        >= workspace[:, 1] - cfg.workspace_command_margin_m
    ):
        raise ValueError("keyboard workspace command margin leaves no interior workspace")


DEFAULT_TASK_NAME = "test"


def validate_task_dir_name(value: str) -> str:
    """Validate one task name as a safe directory component."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("task_name must be a non-empty string without surrounding whitespace")
    if value in {".", ".."} or value.startswith("."):
        raise ValueError("task_name must not be a hidden or relative directory name")
    if "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("task_name must be one safe directory component")
    return value
