#!/usr/bin/env python3
"""test 技能 driver — 硬件驱动测试工具。

用法:
  python .claude/skills/test/driver.py --file <device.py> --offline   # 离线验证
  python .claude/skills/test/driver.py --file <device.py> --generate   # 生成测试脚本
  python .claude/skills/test/driver.py --file <device.py> --smoke      # 真机烟雾测试
  python .claude/skills/test/driver.py --all-offline                   # 批量验证
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# ── 接口契约 ──────────────────────────────────────────────────────

DEVICE_REQUIRED_METHODS = [
    ("connect", ["self"], "bool"),
    ("disconnect", ["self"], None),
    ("get_state", ["self", "full"], "dict"),
    ("send_action", ["self", "action"], "bool"),
    ("reset", ["self", "target"], "bool"),
    ("stop", ["self"], "bool"),
    ("is_connected", ["self"], "bool"),
    ("is_error", ["self"], "bool"),
    ("clear_error", ["self"], "bool"),
]

DEVICE_STATE_VARS = [
    "connected_flag",
    "error_state",
    "last_error_message",
    "last_qpos_cmd",
    "last_cmd_time",
    "last_joint_limit_clipped",
    "last_delta_limited",
]

FORBIDDEN_IMPORTS = {
    "hydra": "Hydra",
    "omegaconf": "OmegaConf",
    "rclpy": "ROS2",
    "rospy": "ROS",
    "pydantic": "Pydantic",
    "draccus": "draccus",
}


@dataclass
class TestResult:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok_items: list[str] = field(default_factory=list)


# ── 设备类分析器 ──────────────────────────────────────────────────

class DeviceAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.source = self.filepath.read_text()
        self.tree = ast.parse(self.source)
        self.device_class: ast.ClassDef | None = None
        self.config_class: ast.ClassDef | None = None
        self.results: dict[str, TestResult] = {}

    def find_classes(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}
            if "send_action" in methods and "connect" in methods:
                self.device_class = node
            # 找 Config dataclass
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id == "dataclass":
                    if node.name.endswith("Config"):
                        self.config_class = node

    @property
    def has_device(self) -> bool:
        return self.device_class is not None

    @property
    def device_name(self) -> str:
        return self.device_class.name if self.device_class else "Unknown"

    def class_methods(self, cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
        return {
            n.name: n
            for n in ast.walk(cls)
            if isinstance(n, ast.FunctionDef)
        }

    def method_returns(self, method: ast.FunctionDef) -> str | None:
        if method.returns:
            return ast.get_source_segment(self.source, method.returns)
        return None

    def method_params(self, method: ast.FunctionDef) -> list[str]:
        params = []
        for arg in method.args.args:
            if arg.arg != "self":
                params.append(arg.arg)
        return params


# ── 检查函数 ──────────────────────────────────────────────────────

def check_config(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()
    if analyzer.config_class is None:
        r.errors.append("未找到 XxxConfig dataclass (可能在其他文件)")
        r.passed = False
        return r

    cls = analyzer.config_class
    r.ok_items.append(f"Config: {cls.name} @dataclass")

    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and item.value:
            if isinstance(item.value, ast.Call):
                func = item.value.func
                is_field = isinstance(func, ast.Attribute) and func.attr == "field"
                if is_field:
                    has_factory = any(kw.arg == "default_factory" for kw in item.value.keywords)
                    if not has_factory:
                        target = item.target
                        if isinstance(target, ast.Name):
                            r.errors.append(
                                f"Config.{target.id}: field() 缺少 default_factory"
                            )
                            r.passed = False
    if r.passed:
        r.ok_items.append("default_factory 使用正确")
    return r


def check_methods(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()
    methods = analyzer.class_methods(analyzer.device_class)

    for name, params, expected_ret in DEVICE_REQUIRED_METHODS:
        if name not in methods:
            r.errors.append(f"缺失方法: {name}()")
            r.passed = False
            continue

        m = methods[name]
        # 检查返回类型
        actual_ret = analyzer.method_returns(m)
        if expected_ret and actual_ret:
            if expected_ret == "bool" and "bool" not in actual_ret:
                if "dict" in actual_ret:
                    r.errors.append(
                        f"{name}() 返回类型: 期望 → {expected_ret}，实际 → {actual_ret}"
                    )
                    r.passed = False
                else:
                    r.warnings.append(
                        f"{name}() 返回类型: 期望 → {expected_ret}，实际 → {actual_ret}"
                    )
            elif expected_ret == "dict" and "dict" not in actual_ret:
                r.warnings.append(
                    f"{name}() 返回类型: 期望 → {expected_ret}，实际 → {actual_ret}"
                )

        # 检查 send_action 参数类型
        if name == "send_action":
            for arg in m.args.args:
                if arg.arg == "action" and arg.annotation:
                    anno = analyzer.method_returns(m)
                    anno_src = ast.get_source_segment(analyzer.source, arg.annotation)
                    if anno_src and "dict" in anno_src:
                        r.errors.append("send_action(action) 不应接受 dict，应为 np.ndarray")
                        r.passed = False
                elif arg.arg == "action":
                    r.warnings.append("send_action(action) 缺少类型标注，建议 -> np.ndarray")

        # 检查 get_state 参数
        if name == "get_state":
            params_list = analyzer.method_params(m)
            if "full" not in params_list:
                r.warnings.append("get_state() 缺少 full 参数")
            else:
                r.ok_items.append(f"{name}(full=False) → {actual_ret or '无标注'}")

    if not any(r.errors):
        r.ok_items.append(f"所有必需方法存在 ({len(DEVICE_REQUIRED_METHODS)} 个)")

    return r


def check_state_vars(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()
    methods = analyzer.class_methods(analyzer.device_class)
    if "__init__" not in methods:
        r.errors.append("未找到 __init__ 方法")
        r.passed = False
        return r

    init_body = ast.get_source_segment(analyzer.source, methods["__init__"]) or ""
    missing = []
    for var in ["connected_flag", "error_state", "last_error_message"]:
        if f"self.{var}" not in init_body:
            missing.append(var)
            r.passed = False

    if missing:
        r.errors.append(f"__init__ 缺少状态变量: {', '.join(missing)}")
    else:
        r.ok_items.append("状态变量完整 (connected_flag, error_state, last_error_message)")

    for var in DEVICE_STATE_VARS[3:]:
        if f"self.{var}" in init_body:
            r.ok_items.append(f"  self.{var}")

    return r


def check_safety(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()

    send_action = analyzer.class_methods(analyzer.device_class).get("send_action")
    if send_action:
        body_str = ast.get_source_segment(analyzer.source, send_action) or ""
        if "limit_joint_range" in body_str or "clip" in body_str:
            r.ok_items.append("send_action 含 joint limit 裁剪")
        else:
            r.warnings.append("send_action 未检测到 joint limit 裁剪")

        if "limit_joint_step" in body_str or "delta" in body_str.lower():
            r.ok_items.append("send_action 含 delta limit 限速")
        else:
            r.warnings.append("send_action 未检测到 delta limit 限速")

    # 检查 bare except
    for node in ast.walk(analyzer.tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            is_top_loop = False
            for ancestor in ast.walk(analyzer.tree):
                if isinstance(ancestor, ast.While):
                    is_top_loop = True
                    break
            if not is_top_loop:
                r.errors.append(f"L{node.lineno}: 禁止 bare except（非顶层主循环）")
                r.passed = False

    # 检查 get_obs 命名
    for node in ast.walk(analyzer.tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_obs":
            r.errors.append(f"L{node.lineno}: 方法名 'get_obs' 应改为 'get_state'")
            r.passed = False

    return r


def check_deps(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()
    for node in ast.walk(analyzer.tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_str = ast.get_source_segment(analyzer.source, node) or ""
            for keyword, name in FORBIDDEN_IMPORTS.items():
                if keyword in import_str.lower():
                    r.errors.append(f"L{node.lineno}: 禁止依赖 {name}")
                    r.passed = False
            if "cv2" in import_str or "torch" in import_str or "open3d" in import_str:
                r.warnings.append(
                    f"L{node.lineno}: 重型依赖 ({import_str.strip()[:50]}...) — "
                    "硬件驱动应避免顶层 import cv2/torch/open3d"
                )
    return r


def check_example(analyzer: DeviceAnalyzer) -> TestResult:
    r = TestResult()
    methods = analyzer.class_methods(analyzer.device_class)
    if "example" in methods:
        r.ok_items.append("example() 函数存在")
    else:
        # 检查全局函数
        has_example = any(
            isinstance(n, ast.FunctionDef) and n.name == "example"
            for n in ast.walk(analyzer.tree)
        )
        if has_example:
            r.ok_items.append("example() 函数存在 (模块级)")
        else:
            r.warnings.append("缺少 example() 函数")
    return r


def run_all_checks(filepath: str) -> dict[str, TestResult]:
    analyzer = DeviceAnalyzer(filepath)
    analyzer.find_classes()

    results = {
        "config": check_config(analyzer),
        "methods": check_methods(analyzer),
        "state_vars": check_state_vars(analyzer),
        "safety": check_safety(analyzer),
        "deps": check_deps(analyzer),
        "example": check_example(analyzer),
    }

    if not analyzer.has_device:
        results["_error"] = TestResult(
            passed=False,
            errors=["未找到执行器类 (需含 connect + send_action 方法)"],
        )

    return results


# ── 报告生成 ──────────────────────────────────────────────────────

def format_results(results: dict[str, TestResult]) -> str:
    total_errors = sum(len(r.errors) for r in results.values())
    total_warnings = sum(len(r.warnings) for r in results.values())
    total_ok = sum(len(r.ok_items) for r in results.values())

    lines = [f"## 测试结果 — {total_errors} errors, {total_warnings} warnings\n"]

    for category, r in results.items():
        if category.startswith("_"):
            lines.append(f"❌ {r.errors[0]}")
            continue

        if r.passed and not r.warnings:
            status = "✓"
        elif r.passed:
            status = "⚠"
        else:
            status = "❌"

        label = {"config": "Config", "methods": "接口方法", "state_vars": "状态变量",
                 "safety": "安全", "deps": "依赖", "example": "example"}.get(category, category)

        lines.append(f"\n{status} **{label}**")
        for e in r.errors:
            lines.append(f"  ❌ {e}")
        for w in r.warnings:
            lines.append(f"  ⚠️ {w}")
        for ok in r.ok_items[:3]:  # 限制 OK 项数量
            lines.append(f"  ✓ {ok}")

    if total_errors == 0:
        lines.insert(1, "✅ 所有检查通过 — 可部署到真机")
    else:
        lines.insert(1, f"❌ {total_errors} 项错误需要修复后才能部署到真机")

    return "\n".join(lines)


# ── 烟雾测试脚本生成 ──────────────────────────────────────────────

def generate_smoke_test(analyzer: DeviceAnalyzer) -> str:
    name = analyzer.device_name
    config_name = analyzer.config_class.name if analyzer.config_class else f"{name}Config"
    file_stem = analyzer.filepath.stem
    module_path = str(analyzer.filepath.relative_to(PROJECT_ROOT)).replace("/", ".").replace(".py", "")

    return textwrap.dedent(f"""\
    #!/usr/bin/env python3
    \"\"\"{name} 烟雾测试 — 自动生成。

    用法:
      python /tmp/test_{name}.py                    # stay-in-place (安全)
      python /tmp/test_{name}.py --small-move        # 小幅运动验证
      python /tmp/test_{name}.py --full              # 完整测试 (含 reset)
    \"\"\"

    import argparse
    import time
    import numpy as np
    from {module_path} import {name}, {config_name}

    def test_stay_in_place(device: {name}):
        print("  [L1] stay-in-place ...")
        state = device.get_state()
        qpos = state["qpos"]
        if np.any(np.isnan(qpos)):
            raise RuntimeError(f"qpos 含 NaN: {{qpos}}")
        print(f"    当前关节: {{np.round(np.rad2deg(qpos), 1)}} deg")

        ok = device.send_action(qpos.copy())
        if not ok:
            raise RuntimeError(f"send_action (stay) 失败: {{device.last_error_message}}")
        print("    ✓ stay-in-place 通过")

    def test_small_move(device: {name}):
        print("  [L2] small-move ...")
        state = device.get_state()
        qpos = state["qpos"].copy()
        if np.any(np.isnan(qpos)):
            raise RuntimeError("qpos 含 NaN")

        target = qpos + np.deg2rad(2.0)
        ok = device.send_action(target)
        if not ok:
            raise RuntimeError(f"small-move 失败: {{device.last_error_message}}")
        time.sleep(1.0)

        # 回到原位
        device.send_action(qpos)
        time.sleep(0.5)
        print("    ✓ small-move 通过")

    def test_full(device: {name}):
        print("  [L3] reset ...")
        ok = device.reset()
        if not ok:
            raise RuntimeError(f"reset 失败: {{device.last_error_message}}")
        time.sleep(2.0)
        state = device.get_state()
        print(f"    复位后关节: {{np.round(np.rad2deg(state['qpos']), 1)}} deg")
        print("    ✓ reset 通过")

    args = argparse.Namespace()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.111")
    parser.add_argument("--small-move", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    print(f"=== {name} Smoke Test ===")
    device = {name}({config_name}(ip=args.ip))

    if not device.connect():
        raise RuntimeError(f"connect 失败: {{device.last_error_message}}")
    print("[connect] ✓")

    try:
        if not device.is_connected():
            raise RuntimeError("is_connected() 返回 False")
        if device.is_error():
            raise RuntimeError(f"is_error() 返回 True: {{device.last_error_message}}")
        print("[health] ✓")

        test_stay_in_place(device)

        if args.small_move or args.full:
            test_small_move(device)
        if args.full:
            test_full(device)

        print("\\n✅ 烟雾测试全部通过")

    finally:
        device.disconnect()
        print("[disconnect] ✓")
    """)


# ── CLI ──────────────────────────────────────────────────────────────

def find_all_device_files() -> list[Path]:
    """查找所有可能的设备文件。"""
    files = []
    for subdir in ["robot", "sensor"]:
        d = PROJECT_ROOT / "dexmani_real" / subdir
        if d.exists():
            for f in d.rglob("*.py"):
                if f.name.startswith("__"):
                    continue
                # 检查是否包含 send_action
                source = f.read_text()
                if "send_action" in source and "connect" in source:
                    files.append(f)
    return files


def run_sim_test(args):
    """SAPIEN 仿真测试 — 无需硬件。"""
    import numpy as np

    print("\n=== SAPIEN 仿真测试 ===")
    print("[setup] 创建 headless SAPIEN 场景 ...")

    try:
        from dexmani_real.robot.model.sim_adapter import SimRobotInterface, SimRobotConfig
    except ImportError as e:
        print(f"❌ 无法导入 sim_adapter: {e}")
        print("   需要安装 SAPIEN: pip install sapien")
        sys.exit(1)

    config = SimRobotConfig(headless=True)
    sim = SimRobotInterface(config)

    print("[connect] ...")
    if not sim.connect():
        print(f"❌ connect 失败: {sim.last_error_message}")
        sys.exit(1)
    print("  ✓ connected")

    try:
        # L0: 状态读取
        state = sim.get_state(full=True)
        assert not np.any(np.isnan(state["arm_qpos"])), "arm_qpos 含 NaN"
        assert not np.any(np.isnan(state["hand_qpos"])), "hand_qpos 含 NaN"
        assert not np.any(np.isnan(state["eef_pos"])), "eef_pos 含 NaN"
        print(f"  [L0] state: arm={np.round(np.rad2deg(state['arm_qpos']), 1)} deg")
        print(f"       eef_pos=({state['eef_pos'][0]:.3f}, {state['eef_pos'][1]:.3f}, {state['eef_pos'][2]:.3f}) m")

        # IK 往返（核心功能验证）
        ik = sim.validate_ik_roundtrip()
        if ik["ok"]:
            print(f"  [IK] roundtrip ✓ (max_err={ik['max_error']:.6f})")
        else:
            print(f"  ❌ IK roundtrip failed: {ik}")
            sys.exit(1)

        # L1: stay-in-place
        qpos = sim.get_full_qpos()
        ok = sim.send_action(qpos.copy())
        assert ok, "send_action (stay) 失败"
        print("  [L1] stay-in-place ✓")

        # L2: small-move
        target = qpos + np.deg2rad(2.0)
        target[7:] = 0  # hand joints stay at current
        ok = sim.send_action(target)
        assert ok, "send_action (small-move) 失败"
        state2 = sim.get_state()
        moved = np.max(np.abs(state2["arm_qpos"] - state["arm_qpos"]))
        assert moved > 0.005, f"joint didn't move (delta={moved:.4f})"
        print(f"  [L2] small-move ✓ (delta={np.rad2deg(moved):.2f}°)")

        # L3: reset
        ok = sim.reset()
        assert ok, "reset 失败"
        state3 = sim.get_state()
        home_err = np.max(np.abs(state3["arm_qpos"] - sim.robot.home_qpos[:7]))
        print(f"  [L3] reset ✓ (home_err={np.rad2deg(home_err):.2f}°)")

        print("\n✅ SAPIEN 仿真测试全部通过 — 接口与真机一致")

    finally:
        sim.disconnect()
        print("[disconnect] ✓")


def main():
    parser = argparse.ArgumentParser(description="test skill driver — 硬件驱动测试")
    parser.add_argument("--file", help="要测试的设备文件路径")
    parser.add_argument("--offline", action="store_true", help="离线接口验证")
    parser.add_argument("--generate", action="store_true", help="生成烟雾测试脚本")
    parser.add_argument("--smoke", action="store_true", help="真机烟雾测试")
    parser.add_argument("--sim", action="store_true", help="SAPIEN 仿真测试 (无硬件)")
    parser.add_argument("--small-move", action="store_true", help="含小幅度运动验证")
    parser.add_argument("--full", action="store_true", help="完整测试 (含 reset)")
    parser.add_argument("--all-offline", action="store_true", help="批量离线验证所有驱动")
    args = parser.parse_args()

    if args.all_offline:
        files = find_all_device_files()
        if not files:
            print("未找到设备文件")
            return
        for f in files:
            print(f"\n{'='*60}")
            print(f"## {f.relative_to(PROJECT_ROOT)}")
            results = run_all_checks(str(f))
            print(format_results(results))
        return

    if not args.file and not args.sim:
        print("需要指定 --file、--all-offline 或 --sim")
        return

    # SAPIEN 仿真测试 (独立模式，无需 --file)
    if args.sim:
        run_sim_test(args)
        return

    filepath = Path(args.file).resolve()
    if not filepath.exists():
        print(f"文件不存在: {args.file}")
        sys.exit(1)

    # 离线验证
    if args.offline or (not args.smoke and not args.generate):
        print(f"## 离线验证: {filepath.name}\n")
        results = run_all_checks(str(filepath))
        print(format_results(results))

        total_errors = sum(len(r.errors) for r in results.values())
        if total_errors > 0:
            sys.exit(1)

    # 生成测试脚本
    if args.generate:
        analyzer = DeviceAnalyzer(str(filepath))
        analyzer.find_classes()
        if not analyzer.has_device:
            print("未找到设备类，无法生成测试脚本")
            sys.exit(1)
        script = generate_smoke_test(analyzer)
        out_path = Path(f"/tmp/test_{analyzer.device_name}.py")
        out_path.write_text(script)
        out_path.chmod(0o755)
        print(f"测试脚本已生成: {out_path}")
        print(f"用法: python {out_path} [--small-move] [--full]")

    # 真机烟雾测试
    if args.smoke:
        print("\n⚠️ 真机测试模式 — 请先确认 pre-flight 检查 (.claude/rules/hardware-safety.md)")
        print("   继续前请确认:")
        print("   1. 机器人周围无人员/障碍物")
        print("   2. E-Stop 按钮可用")
        print("   3. 机械臂各关节正常")
        print()
        resp = input("   确认继续? (yes/no): ")
        if resp.lower() != "yes":
            print("已取消")
            return

        analyzer = DeviceAnalyzer(str(filepath))
        analyzer.find_classes()
        if not analyzer.has_device:
            print("未找到设备类")
            sys.exit(1)

        # 动态加载模块并执行测试
        spec = importlib.util.spec_from_file_location(
            analyzer.filepath.stem, str(analyzer.filepath)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        DeviceClass = getattr(module, analyzer.device_name)
        ConfigClass = getattr(module, analyzer.config_class.name) if analyzer.config_class else None

        device = DeviceClass(ConfigClass()) if ConfigClass else DeviceClass()
        print(f"\n=== {analyzer.device_name} Smoke Test ===")

        if not device.connect():
            print(f"❌ connect 失败: {device.last_error_message}")
            sys.exit(1)
        print("[connect] ✓")

        try:
            if not device.is_connected():
                print("❌ is_connected() 返回 False")
                sys.exit(1)
            if device.is_error():
                print(f"❌ is_error() 返回 True: {device.last_error_message}")
                sys.exit(1)
            print("[health] ✓")

            # L1: stay-in-place
            import numpy as np
            import time
            state = device.get_state()
            qpos = state["qpos"]
            if np.any(np.isnan(qpos)):
                print(f"❌ qpos 含 NaN: {qpos}")
                sys.exit(1)
            print(f"   当前关节: {np.round(np.rad2deg(qpos), 1)} deg")

            ok = device.send_action(qpos.copy())
            if not ok:
                print(f"❌ send_action (stay) 失败: {device.last_error_message}")
                sys.exit(1)
            print("[L1 stay-in-place] ✓")

            # L2: small-move
            if args.small_move or args.full:
                target = qpos + np.deg2rad(2.0)
                ok = device.send_action(target)
                if not ok:
                    print(f"❌ small-move 失败: {device.last_error_message}")
                    sys.exit(1)
                time.sleep(1.0)
                device.send_action(qpos.copy())
                time.sleep(0.5)
                print("[L2 small-move] ✓")

            # L3: full
            if args.full:
                ok = device.reset()
                if not ok:
                    print(f"❌ reset 失败: {device.last_error_message}")
                    sys.exit(1)
                time.sleep(2.0)
                state = device.get_state()
                print(f"   复位后: {np.round(np.rad2deg(state['qpos']), 1)} deg")
                print("[L3 reset] ✓")

            print("\n✅ 烟雾测试全部通过")

        finally:
            device.disconnect()
            print("[disconnect] ✓")


if __name__ == "__main__":
    main()
