from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_only_safe_command_publisher_writes_raw_actuator_ipc() -> None:
    violations: list[str] = []
    pattern = re.compile(r"(?:arm_action_q\.put|hand_cmd_ring\.write|action_commit_ring\.write)")
    protocol_path = REPO_ROOT / "dexmani_real" / "policy" / "action_protocol.py"
    protocol_tree = ast.parse(protocol_path.read_text(encoding="utf-8"), filename=str(protocol_path))
    publisher = next(
        node for node in protocol_tree.body if isinstance(node, ast.ClassDef) and node.name == "SafeCommandPublisher"
    )
    allowed_protocol_lines = {node.lineno for node in ast.walk(publisher) if isinstance(node, ast.Call)}
    for root in (REPO_ROOT / "dexmani_real", REPO_ROOT / "examples" / "real"):
        for path in root.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT).as_posix()
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not pattern.search(line):
                    continue
                if relative == "dexmani_real/policy/action_protocol.py" and line_number in allowed_protocol_lines:
                    continue
                # HOME is a correlated request/result lifecycle, not a servo
                # endpoint. It remains the sole explicit queue exception.
                if relative == "dexmani_real/shm/shared_storage.py" and "HOME_SENTINEL" in line:
                    continue
                violations.append(f"{relative}:{line_number}: {line.strip()}")
    assert not violations, "raw actuator IPC write outside protocol boundary:\n" + "\n".join(violations)


def test_backend_runtime_import_surface_is_device_and_gui_free() -> None:
    files = (
        REPO_ROOT / "dexmani_real/policy/runtime.py",
        REPO_ROOT / "dexmani_real/policy/observation.py",
        REPO_ROOT / "dexmani_real/policy/tensor_block.py",
        REPO_ROOT / "dexmani_real/policy/inference_process.py",
    )
    forbidden = ("pyrealsense2", "xhand_controller", "xarm.wrapper", "torch", "open3d", "matplotlib", "rerun")
    violations = [
        f"{path.name}: {token}" for path in files for token in forbidden if token in path.read_text(encoding="utf-8")
    ]
    assert not violations, "backend-neutral runtime imports an unselected device/model/UI dependency: " + ", ".join(
        violations
    )


def test_ipc_schema_layer_has_no_policy_or_recording_dependency() -> None:
    schema = (REPO_ROOT / "dexmani_real" / "ipc" / "schema.py").read_text(encoding="utf-8")
    storage = (REPO_ROOT / "dexmani_real" / "shm" / "shared_storage.py").read_text(encoding="utf-8")
    forbidden = ("dexmani_real.policy", "dexmani_real.recording", "dexmani_real.robot", "dexmani_real.sensor")

    assert not [token for token in forbidden if token in schema]
    assert "from dexmani_real.policy" not in storage
    assert "from dexmani_real.recording" not in storage


def test_worker_modules_do_not_own_global_shutdown_flag() -> None:
    violations: list[str] = []
    for package in ("policy", "recording", "robot", "sensor"):
        for path in (REPO_ROOT / "dexmani_real" / package).rglob("*.py"):
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if re.search(r"\bis_running\.value\s*=", line):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
    assert not violations, "worker module writes main-owned is_running flag:\n" + "\n".join(violations)


def test_hardware_entry_points_supply_geometry_aware_action_gate() -> None:
    paths = (
        REPO_ROOT / "dexmani_real" / "policy" / "vr_teleop_policy.py",
        REPO_ROOT / "examples" / "real" / "keyboard_teleop_real.py",
        REPO_ROOT / "examples" / "real" / "calibrate_camera.py",
        REPO_ROOT / "examples" / "real" / "replay_traj.py",
    )
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "publish_joint_targets":
                continue
            if not any(keyword.arg == "safety_gate" for keyword in node.keywords):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    assert not violations, "hardware publisher omitted geometry-aware safety_gate: " + ", ".join(violations)


def test_replay_never_returns_from_finally_and_swallows_motion_errors() -> None:
    path = REPO_ROOT / "examples" / "real" / "replay_traj.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    replayer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrajectoryReplayer")
    run_method = next(node for node in replayer.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    violations = [
        node.lineno
        for node in ast.walk(run_method)
        if isinstance(node, ast.Try)
        and any(
            isinstance(descendant, ast.Return) for statement in node.finalbody for descendant in ast.walk(statement)
        )
    ]
    assert not violations, f"replay return in finally suppresses exceptions at lines {violations}"
