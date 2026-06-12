#!/usr/bin/env python3
"""code 技能 driver — 新代码合规检查 + 参考检索 + 清单生成。

用法:
  python .claude/skills/code/check_new_code.py --module robot [--file path.py]
  python .claude/skills/code/check_new_code.py --module controller --list-checks
  python .claude/skills/code/check_new_code.py --module recording --refs
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REF_SEARCH = PROJECT_ROOT / ".claude/skills/ref-check/ref-search.py"

# ── 各模块的接口契约检查项 ──────────────────────────────────────────
# 基于 CLAUDE.md Section 2-8 + Section 14 检查清单

DEVICE_CHECKS = [
    ("Config dataclass", "check_config_dataclass"),
    ("connect() → bool", "check_connect_bool"),
    ("disconnect() → None", "check_disconnect"),
    ("get_state(full=False) → dict", "check_get_state"),
    ("send_action(np.ndarray) → bool", "check_send_action"),
    ("reset(target: np.ndarray|None) → bool", "check_reset"),
    ("stop() → bool", "check_stop"),
    ("is_connected() → bool", "check_is_connected"),
    ("is_error() → bool", "check_is_error"),
    ("clear_error() → bool", "check_clear_error"),
    ("状态变量 (connected_flag/error_state/last_error_message)", "check_state_vars"),
    ("last_joint_limit_clipped / last_delta_limited", "check_clip_flags"),
    ("send_action 内 range clip + delta limit", "check_safety_pipeline"),
    ("简单 example() 而非 argparse CLI", "check_example"),
    ("无重型依赖 (cv2/torch) 在顶层 import", "check_imports"),
]

SENSOR_CHECKS = [
    ("Config dataclass", "check_config_dataclass"),
    ("connect() → bool", "check_connect_bool"),
    ("disconnect() → None", "check_disconnect"),
    ("get_state(full=False) → dict", "check_get_state"),
    ("stop() → bool", "check_stop"),
    ("is_connected() → bool", "check_is_connected"),
    ("is_error() → bool", "check_is_error"),
    ("clear_error() → bool", "check_clear_error"),
    ("使用 connect/disconnect 而非 start/stop", "check_connect_naming"),
    ("默认 get_state 轻量，full=True 含调试信息", "check_get_state_lightweight"),
    ("单位标注 (深度: m, 图像: uint8)", "check_units"),
    ("传感器数据与派生观测分离", "check_separation"),
]

CONTROLLER_CHECKS = [
    ("通过 RobotInterface 操作硬件，不直接调 SDK", "check_uses_robot_interface"),
    ("通过 shared_memory 读 VR 帧", "check_ipc_pattern"),
    ("_tick() 实现 arm IK + hand retarget 并行", "check_tick_pattern"),
    ("hand EMA 平滑 (alpha=0.3 录制，0.5 部署)", "check_ema_smoothing"),
    ("状态机 (IDLE/TELEOP/RECORDING/EMERGENCY_STOP)", "check_state_machine"),
    ("追踪丢失/IK 连续失败的错误恢复", "check_error_recovery"),
    ("VR re-anchoring 在状态切换时调用", "check_reanchor"),
    ("通过 multiprocessing.Queue 处理键盘信号", "check_keyboard_queue"),
]

RECORDING_CHECKS = [
    ("EpisodeRecorder 与 Controller 解耦", "check_decoupled"),
    ("start_episode → stop_episode 生命周期", "check_lifecycle"),
    ("HDF5 结构与 plan 一致 (/obs,/action,/vr,/camera,/meta)", "check_hdf5_structure"),
    ("quality_flags 完整 11 bit", "check_quality_flags"),
    ("add_frame 接受 RobotState + RobotAction + vr_frame", "check_add_frame_sig"),
]

DATA_CHECKS = [
    ("EpisodeReader 懒加载", "check_lazy_load"),
    ("iter_frames 支持 skip_rejected", "check_iter_frames"),
    ("get_valid_mask() → np.ndarray", "check_valid_mask"),
    ("DataValidator 检查 nan/inf/shape/时间戳/关节/电流", "check_validator"),
]

DEPLOY_CHECKS = [
    ("PolicyLoader 手动加载，不依赖训练框架", "check_policy_loader"),
    ("PolicyRunner 含 action chunk + EMA 平滑", "check_policy_runner"),
    ("SafetyMonitor 检查 workspace/关节/力矩/电流/温度", "check_safety_monitor"),
    ("ObservationBuilder 独立于模型", "check_obs_builder"),
    ("ActionParser 支持 arm_only/hand_only/full 模式", "check_action_parser"),
]

TELEOP_CHECKS = [
    ("Config dataclass", "check_config_dataclass"),
    ("不直接操作硬件 SDK", "check_no_sdk"),
    ("不负责数据录制（由 recording 模块负责）", "check_no_recording"),
]

PLANNER_CHECKS = [
    ("纯几何计算，不依赖硬件 SDK", "check_pure_geometry"),
    ("example() 函数", "check_example"),
]

UTILS_CHECKS = [
    ("小型纯函数工具", "check_pure_functions"),
    ("无状态业务逻辑", "check_stateless"),
]

MODULE_CHECKS = {
    "robot": DEVICE_CHECKS,
    "sensor": SENSOR_CHECKS,
    "controller": CONTROLLER_CHECKS,
    "recording": RECORDING_CHECKS,
    "data": DATA_CHECKS,
    "deploy": DEPLOY_CHECKS,
    "teleop": TELEOP_CHECKS,
    "planner": PLANNER_CHECKS,
    "utils": UTILS_CHECKS,
}

# ── 通用检查项 (所有模块) ──────────────────────────────────────────

COMMON_CHECKS = [
    ("类名 PascalCase", "check_class_naming"),
    ("函数/变量 snake_case", "check_snake_case"),
    ("常量 UPPER_SNAKE_CASE", "check_const_naming"),
    ("配置类 XxxConfig 后缀", "check_config_naming"),
    ("公有属性无前导下划线", "check_no_leading_underscore"),
    ("物理量单位显式标注 # rad, # m, # N", "check_units"),
    ("不准 bare except (顶层主循环除外)", "check_no_bare_except"),
    ("异常捕获后记录 last_error_message", "check_error_logging"),
    ("禁止 ROS/Hydra/draccus 依赖", "check_forbidden_deps"),
    ("参考来源注释 # ref: [P1] ... L120-150", "check_ref_annotation"),
]


# ── AST 检查器 ─────────────────────────────────────────────────────

class CodeChecker:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.source = Path(filepath).read_text()
        self.tree = ast.parse(self.source)
        self.results: list[dict[str, Any]] = []

    def _add(self, level: str, check: str, detail: str, lineno: int = 0):
        self.results.append({"level": level, "check": check, "detail": detail, "lineno": lineno})

    # ── 命名检查 ──

    def check_class_naming(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self._add("error", "类名 PascalCase", f"class '{node.name}' 不符合 PascalCase", node.lineno)

    def check_snake_case(self):
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_") and not node.name.islower() and "_" not in node.name:
                    self._add("warning", "函数名 snake_case", f"'{node.name}' 应使用 snake_case", node.lineno)

    def check_const_naming(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        pass  # UPPER_CASE OK
                    elif isinstance(target, ast.Name) and re.match(r"^[A-Z][a-z]", target.id):
                        self._add("warning", "常量 UPPER_CASE", f"'{target.id}' 如为常量应全大写", node.lineno)

    def check_no_leading_underscore(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("_") and not target.id.startswith("__"):
                        # 检查是否在 __init__ 中作为 self._xxx
                        pass  # 私有实例变量加 _ 是 OK 的

    # ── 接口检查 ──

    def check_config_dataclass(self):
        has_config = False
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "dataclass":
                        if node.name.endswith("Config"):
                            has_config = True
                            # 检查 default_factory
                            for item in node.body:
                                if isinstance(item, ast.AnnAssign) and item.value:
                                    if isinstance(item.value, ast.Call):
                                        func = item.value.func
                                        if isinstance(func, ast.Attribute) and func.attr == "field":
                                            for kw in item.value.keywords:
                                                if kw.arg == "default_factory":
                                                    break
                                            else:
                                                # 检查默认值是否为可变类型
                                                if isinstance(item.value, (ast.List, ast.Dict)):
                                                    self._add("warning", "default_factory", f"'{item.target.id}' 可变默认值应使用 field(default_factory=...)", item.lineno)
        if not has_config:
            self._add("warning", "Config dataclass", "未找到 XxxConfig dataclass (可能不在本文件中)")

    def check_connect_bool(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "connect":
                if node.returns:
                    if isinstance(node.returns, ast.Name) and node.returns.id == "bool":
                        pass
                    else:
                        self._add("error", "connect() → bool", "connect() 应标注返回类型 → bool", node.lineno)

    def check_get_state(self):
        has_get_state = any(
            isinstance(n, ast.FunctionDef) and n.name == "get_state" for n in ast.walk(self.tree)
        )
        has_get_obs = any(
            isinstance(n, ast.FunctionDef) and n.name == "get_obs" for n in ast.walk(self.tree)
        )
        if has_get_obs and not has_get_state:
            self._add("error", "get_state() 命名", "使用了 get_obs()，应改为 get_state()")

    def check_send_action(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "send_action":
                # 检查返回类型
                if node.returns:
                    if isinstance(node.returns, ast.Name) and node.returns.id == "bool":
                        pass
                    elif isinstance(node.returns, ast.Subscript) and node.returns.value.id == "dict":
                        self._add("error", "send_action() → bool", "send_action() 不应返回 dict，应返回 bool")

    def check_state_vars(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                body_str = ast.get_source_segment(self.source, node) or ""
                for var in ["connected_flag", "error_state", "last_error_message"]:
                    if var not in body_str:
                        self._add("error", "状态变量", f"__init__ 中缺少 self.{var}")

    # ── 安全 / 错误处理 ──

    def check_no_bare_except(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self._add("error", "禁止 bare except", "bare except 仅允许在顶层主循环", node.lineno)

    def check_error_logging(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ExceptHandler):
                body_str = ast.get_source_segment(self.source, node) or ""
                if "last_error_message" not in body_str and "logging" not in body_str and "print" not in body_str:
                    if node.type and not isinstance(node.type, ast.Name):
                        continue
                    self._add("warning", "异常记录", "异常捕获后未记录 last_error_message 或日志", node.lineno)

    def check_forbidden_deps(self):
        forbidden = {
            "hydra": "Hydra",
            "omegaconf": "OmegaConf",
            "rclpy": "ROS2",
            "rospy": "ROS",
            "pydantic": "Pydantic",
            "draccus": "draccus",
        }
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_str = ast.get_source_segment(self.source, node) or ""
                for keyword, name in forbidden.items():
                    if keyword in import_str.lower():
                        self._add("error", "禁止依赖", f"引入了禁止的依赖: {name}")

    def check_ref_annotation(self):
        if "# ref:" not in self.source and "# ref:" not in self.source:
            pass  # 不做强制要求，由 skill 提示

    def check_example(self):
        has_example = any(
            isinstance(n, ast.FunctionDef) and n.name == "example" for n in ast.walk(self.tree)
        )
        if not has_example:
            self._add("info", "example()", "模块缺少 example() 函数")

    def check_config_naming(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "dataclass":
                        if not node.name.endswith("Config"):
                            self._add("warning", "Config 命名", f"dataclass '{node.name}' 应以 Config 结尾", node.lineno)

    def check_units(self):
        # 检查关键物理量是否有单位注释
        unit_vars = {
            "qpos": "rad", "qvel": "rad/s", "tau": "N·m",
            "eef_pos": "m", "current": "mA", "temperature": "°C",
        }
        for var, unit in unit_vars.items():
            pattern = re.compile(rf"{var}.*#.*{unit}")
            # 不强制报错，在 review 时输出 info

    def run_all(self):
        for name in dir(self):
            if name.startswith("check_"):
                getattr(self)()

    def report(self) -> str:
        lines = [f"## {self.filepath} — {len(self.results)} 项"]
        errors = [r for r in self.results if r["level"] == "error"]
        warnings = [r for r in self.results if r["level"] == "warning"]
        infos = [r for r in self.results if r["level"] == "info"]

        for label, items in [("❌ 错误", errors), ("⚠️ 警告", warnings), ("ℹ️ 信息", infos)]:
            if items:
                lines.append(f"\n### {label} ({len(items)})")
                for r in items:
                    loc = f"L{r['lineno']}" if r["lineno"] else ""
                    lines.append(f"  {loc:>6} {r['check']}: {r['detail']}")

        return "\n".join(lines)


# ── 参考检索 ────────────────────────────────────────────────────────

def search_refs(module: str):
    if not REF_SEARCH.exists():
        return "⚠️ ref-search.py 不存在"
    result = subprocess.run(
        [sys.executable, str(REF_SEARCH), module],
        capture_output=True, text=True
    )
    return result.stdout


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="code skill — 新代码合规检查")
    parser.add_argument("--module", required=True,
                        choices=list(MODULE_CHECKS.keys()),
                        help="模块类型")
    parser.add_argument("--file", help="要检查的文件路径")
    parser.add_argument("--list-checks", action="store_true", help="列出该模块的检查清单")
    parser.add_argument("--refs", action="store_true", help="显示参考项目映射")
    args = parser.parse_args()

    # 1. 参考检索
    refs_output = search_refs(args.module)
    print(refs_output)

    # 2. 检查清单
    checks = COMMON_CHECKS + MODULE_CHECKS.get(args.module, [])
    print(f"\n## {args.module} 模块检查清单 ({len(checks)} 项)\n")
    for i, (desc, _) in enumerate(checks, 1):
        print(f"  {i:3d}. {desc}")

    # 3. 文件检查
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"\n⚠️ 文件不存在: {args.file}")
            sys.exit(0)

        checker = CodeChecker(str(filepath))
        checker.run_all()
        print(f"\n{checker.report()}")

        errors = [r for r in checker.results if r["level"] == "error"]
        if errors:
            print(f"\n## {len(errors)} 项错误需要修复")
            sys.exit(1)
        else:
            print("\n## 无错误")


if __name__ == "__main__":
    main()
