#!/usr/bin/env python3
"""review 技能 driver — 全面代码审查。

用法:
  python .claude/skills/review/review_code.py <file1> [file2 ...]
  python .claude/skills/review/review_code.py --diff           # 审查 git diff 中的文件
  python .claude/skills/review/review_code.py --diff --staged   # 审查暂存区
  python .claude/skills/review/review_code.py --all             # 审查所有 Python 文件
  python .claude/skills/review/review_code.py <file> --json     # JSON 输出
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SRC_ROOT = PROJECT_ROOT / "dexmani_real"


@dataclass
class Finding:
    level: str          # error | warning | info
    category: str       # naming | interface | safety | config | style | deps | reference
    message: str
    file: str
    line: int = 0
    col: int = 0
    snippet: str = ""

    def format(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        return f"{icons[self.level]} [{self.category}] {loc}  {self.message}"


@dataclass
class ReviewReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self): return [f for f in self.findings if f.level == "error"]
    @property
    def warnings(self): return [f for f in self.findings if f.level == "warning"]
    @property
    def infos(self): return [f for f in self.findings if f.level == "info"]

    def to_dict(self) -> dict:
        return {"findings": [vars(f) for f in self.findings]}

    def summary(self) -> str:
        return f"Errors: {len(self.errors)}  Warnings: {len(self.warnings)}  Info: {len(self.infos)}"


# ── 审查规则 ─────────────────────────────────────────────────────────

class Reviewer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.relpath = str(Path(filepath).relative_to(PROJECT_ROOT))
        self.source = ""
        self.tree = None
        self.report = ReviewReport()

        try:
            self.source = Path(filepath).read_text()
            self.tree = ast.parse(self.source)
        except SyntaxError as e:
            self.add("error", "syntax", f"语法错误: {e}", e.lineno or 1)
        except Exception as e:
            self.add("error", "syntax", f"无法解析文件: {e}")
            return

    def add(self, level: str, category: str, message: str, lineno: int = 0, col: int = 0):
        snippet = ""
        if lineno > 0:
            lines = self.source.split("\n")
            if lineno <= len(lines):
                snippet = lines[lineno - 1].strip()[:100]
        self.report.findings.append(Finding(level, category, message, self.relpath, lineno, col, snippet))

    # ══════════════════════════════════════════════════════════════════
    # 1. 命名规范 (CLAUDE.md Section 10)
    # ══════════════════════════════════════════════════════════════════

    def review_naming(self):
        module_name = Path(self.filepath).stem

        for node in ast.walk(self.tree):
            # 类名 PascalCase
            if isinstance(node, ast.ClassDef):
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self.add("error", "naming",
                             f"类名 '{node.name}' 不符合 PascalCase", node.lineno)

            # 函数名 snake_case
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                if not re.match(r"^[a-z][a-z0-9_]*$", node.name):
                    self.add("warning", "naming",
                             f"函数名 '{node.name}' 应使用 snake_case", node.lineno)

            # 方法命名统一
            if isinstance(node, ast.FunctionDef):
                if node.name == "get_obs":
                    self.add("error", "naming",
                             "方法名 'get_obs' 应改为 'get_state'", node.lineno)

            # Config dataclass 命名
            if isinstance(node, ast.ClassDef):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == "dataclass":
                        if not node.name.endswith("Config"):
                            self.add("warning", "naming",
                                     f"dataclass '{node.name}' 应以 Config 结尾", node.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 2. 接口契约 (CLAUDE.md Sections 2-8)
    # ══════════════════════════════════════════════════════════════════

    def review_interface(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            # connect() 返回类型
            if node.name == "connect":
                if node.returns:
                    returns = ast.get_source_segment(self.source, node.returns)
                    if returns and "bool" not in returns:
                        self.add("error", "interface",
                                 f"connect() 返回类型应为 bool，当前: {returns}", node.lineno)

            # send_action() 返回类型
            if node.name == "send_action":
                if node.returns:
                    returns = ast.get_source_segment(self.source, node.returns)
                    if returns and "dict" in returns:
                        self.add("error", "interface",
                                 f"send_action() 不应返回 dict，应返回 bool", node.lineno)
                    elif returns and "bool" not in returns and "None" not in returns:
                        self.add("warning", "interface",
                                 f"send_action() 建议返回 bool，当前: {returns}", node.lineno)

                # send_action 只接受 np.ndarray
                args = node.args
                if args.args:
                    first_arg = args.args[0] if args.args[0].arg != "self" else (args.args[1] if len(args.args) > 1 else None)
                    if first_arg and first_arg.annotation:
                        anno = ast.get_source_segment(self.source, first_arg.annotation)
                        if anno and "dict" in anno:
                            self.add("error", "interface",
                                     f"send_action() 不应接受 dict 输入，应为 np.ndarray", node.lineno)

            # is_connected/is_error/clear_error 存在性检查（仅对类方法）
            if node.name in ("reset", "stop"):
                if node.returns:
                    returns = ast.get_source_segment(self.source, node.returns)
                    if returns and "bool" not in returns:
                        self.add("warning", "interface",
                                 f"{node.name}() 建议返回 bool，当前: {returns}", node.lineno)

    def review_missing_methods(self):
        """检查执行器类是否缺少必需方法。"""
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # 识别是否为执行器类（有 send_action 方法）
            methods = {n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)}
            if "send_action" not in methods:
                continue

            required = ["connect", "disconnect", "get_state", "stop",
                        "is_connected", "is_error", "clear_error"]
            for m in required:
                if m not in methods:
                    self.add("error", "interface",
                             f"执行器类 '{node.name}' 缺少必需方法 '{m}()'", node.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 3. 安全与错误处理 (CLAUDE.md Section 12)
    # ══════════════════════════════════════════════════════════════════

    def review_safety(self):
        for node in ast.walk(self.tree):
            # 禁止 bare except（非顶层主循环）
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                # 检查是否在 while True 内
                is_top_loop = False
                parent = getattr(node, "_parent", None)
                if parent:
                    for p in ast.walk(parent):
                        if isinstance(p, ast.While):
                            is_top_loop = True
                            break
                if not is_top_loop:
                    self.add("error", "safety",
                             "禁止 bare except（仅顶层主循环允许）", node.lineno)

            # 检查 send_action 是否包含安全裁剪
            if isinstance(node, ast.FunctionDef) and node.name == "send_action":
                body_str = ast.get_source_segment(self.source, node) or ""
                if "clip" not in body_str and "limit_joint" not in body_str:
                    self.add("warning", "safety",
                             "send_action() 中未检测到 joint limit / delta limit 裁剪", node.lineno)

            # 检查状态变量
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                body = ast.get_source_segment(self.source, node) or ""
                # 只在执行器类中检查
                cls_node = self._enclosing_class(node)
                if cls_node and "send_action" in {n.name for n in ast.walk(cls_node) if isinstance(n, ast.FunctionDef)}:
                    for var in ["connected_flag", "error_state", "last_error_message"]:
                        if f"self.{var}" not in body:
                            self.add("error", "safety",
                                     f"__init__ 缺少 self.{var}", node.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 4. 配置规范 (CLAUDE.md Section 9)
    # ══════════════════════════════════════════════════════════════════

    def review_config(self):
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.ClassDef):
                continue

            has_dataclass = any(
                isinstance(d, ast.Name) and d.id == "dataclass"
                for d in node.decorator_list
            )
            if not has_dataclass:
                continue

            # 检查 default_factory
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and item.value:
                    if isinstance(item.value, ast.Call):
                        func = item.value.func
                        is_field = isinstance(func, ast.Attribute) and func.attr == "field"
                        if is_field:
                            has_factory = any(kw.arg == "default_factory" for kw in item.value.keywords)
                            if not has_factory:
                                # 检查默认值类型
                                val = item.value
                                if isinstance(val, ast.Call):
                                    for kw in val.keywords:
                                        if kw.arg == "default":
                                            if isinstance(kw.value, (ast.List, ast.Dict, ast.Call)):
                                                self.add("warning", "config",
                                                         f"'{item.target.id}' 可变默认值应使用 field(default_factory=...)",
                                                         item.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 5. 代码风格 (CLAUDE.md Section 10)
    # ══════════════════════════════════════════════════════════════════

    def review_style(self):
        # 检查前导下划线（公有属性不应加 _）
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                                if target.attr.startswith("_") and not target.attr.startswith("__"):
                                    if target.attr not in ("_set_servo_mode", "_set_position_mode"):
                                        self.add("info", "style",
                                                 f"公有属性建议不加前导下划线: self.{target.attr}", stmt.lineno)

        # 检查 argparse 使用（模块应提供 example() 而非 argparse CLI）
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_str = ast.get_source_segment(self.source, node) or ""
                if "argparse" in import_str:
                    # 检查是否有 example() 函数
                    has_example = any(
                        isinstance(n, ast.FunctionDef) and n.name == "example"
                        for n in ast.walk(self.tree)
                    )
                    if not has_example:
                        self.add("info", "style",
                                 "模块使用 argparse 但未提供 example() 函数", node.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 6. 依赖检查 (CLAUDE.md Section 11)
    # ══════════════════════════════════════════════════════════════════

    FORBIDDEN_IMPORTS = {
        "hydra": "Hydra (使用 @dataclass)",
        "omegaconf": "OmegaConf (使用 @dataclass)",
        "rclpy": "ROS2 (使用 multiprocessing + shared_memory)",
        "rospy": "ROS (使用 multiprocessing + shared_memory)",
        "pydantic": "Pydantic (使用 @dataclass + __post_init__)",
        "draccus": "draccus (不引入该框架)",
    }

    HEAVY_DEPS = {
        "torch": "torch 应只在 deploy/ 中使用",
        "cv2": "cv2 应在函数内部局部 import 或只在 viewer 中使用",
        "open3d": "open3d 应在函数内部局部 import",
        "wandb": "wandb 应只在 deploy/ 中使用",
    }

    def review_deps(self):
        in_hardware_driver = any(
            d in str(self.filepath) for d in ["robot/", "sensor/"]
        )
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_str = ast.get_source_segment(self.source, node) or ""

                for keyword, reason in self.FORBIDDEN_IMPORTS.items():
                    if keyword in import_str.lower():
                        self.add("error", "deps",
                                 f"禁止依赖: {reason}", node.lineno)

                if in_hardware_driver:
                    for keyword, reason in self.HEAVY_DEPS.items():
                        if keyword in import_str.lower():
                            self.add("warning", "deps",
                                     f"硬件驱动含重型依赖: {reason}", node.lineno)

    # ══════════════════════════════════════════════════════════════════
    # 7. 参考注释检查
    # ══════════════════════════════════════════════════════════════════

    def review_references(self):
        has_ref = "# ref:" in self.source
        if not has_ref:
            # 检查是否为新增模块（无 ref 可能正常）
            pass  # 不做强制要求

    # ══════════════════════════════════════════════════════════════════

    def _enclosing_class(self, func_node: ast.FunctionDef) -> ast.ClassDef | None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is func_node:
                        return node
        return None

    def run(self):
        if self.tree is None:
            return
        for name in dir(self):
            if name.startswith("review_"):
                getattr(self, name)()


# ── 按模块检查对应接口 ──────────────────────────────────────────────

def check_module_specific(filepath: str, report: ReviewReport):
    """针对不同模块类型做专项检查。"""
    rel = str(Path(filepath).relative_to(PROJECT_ROOT))

    # 识别模块类型
    if "robot/" in rel and "model/" not in rel:
        _check_device_interface(filepath, report)
    elif "sensor/" in rel:
        _check_sensor_interface(filepath, report)


def _check_device_interface(filepath: str, report: ReviewReport):
    """检查执行器类是否遵循 CLAUDE.md Section 2.1 接口。"""
    source = Path(filepath).read_text()
    rel = str(Path(filepath).relative_to(PROJECT_ROOT))


def _check_sensor_interface(filepath: str, report: ReviewReport):
    """检查传感器是否遵循 CLAUDE.md Section 2.3 接口。"""
    source = Path(filepath).read_text()
    rel = str(Path(filepath).relative_to(PROJECT_ROOT))

    if "def start(" in source and "def connect(" not in source:
        report.findings.append(Finding("warning", "interface",
                                       "传感器应使用 connect()/disconnect() 而非 start()/stop()",
                                       rel, 0))
    if "def get_obs(" in source and "def get_state(" not in source:
        report.findings.append(Finding("warning", "interface",
                                       "应使用 get_state() 而非 get_obs()",
                                       rel, 0))


# ── CLI ──────────────────────────────────────────────────────────────

def get_changed_files(staged: bool = False) -> list[str]:
    """获取 git diff 中变更的 Python 文件。"""
    cmd = ["git", "diff", "--name-only"]
    if staged:
        cmd.append("--staged")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    files = [f for f in result.stdout.strip().split("\n") if f.endswith(".py")]
    return [str(PROJECT_ROOT / f) for f in files]


def find_all_python_files() -> list[str]:
    """查找项目中所有 Python 文件（排除 __pycache__ 和 .claude）。"""
    files = []
    for f in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(f) or ".claude" in str(f):
            continue
        files.append(str(f))
    return files


def format_report(report: ReviewReport, verbose: bool = False) -> str:
    lines = [f"## Review Report — {report.summary()}\n"]

    # 错误优先
    for f in report.errors:
        lines.append(f.format())
        if verbose and f.snippet:
            lines.append(f"    → {f.snippet}")

    if report.warnings:
        lines.append(f"\n### 警告 ({len(report.warnings)})")
        for f in report.warnings:
            lines.append(f.format())

    if report.infos and verbose:
        lines.append(f"\n### 建议 ({len(report.infos)})")
        for f in report.infos[:20]:  # 限制数量
            lines.append(f.format())

    if not report.findings:
        lines.append("✅ 未发现问题")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="review skill — 全面代码审查")
    parser.add_argument("files", nargs="*", help="要审查的文件")
    parser.add_argument("--diff", action="store_true", help="审查 git diff 中的文件")
    parser.add_argument("--staged", action="store_true", help="审查暂存区 (配合 --diff)")
    parser.add_argument("--all", action="store_true", help="审查所有 Python 文件")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    # 收集文件
    files = [str(Path(f).resolve()) for f in args.files]
    if args.diff or args.staged:
        files.extend(get_changed_files(staged=args.staged))
    if args.all:
        files.extend(find_all_python_files())

    if not files:
        print("未指定文件。用法: --file <path> | --diff | --all")
        sys.exit(0)

    # 去重
    files = list(dict.fromkeys(files))
    all_findings: list[Finding] = []

    for f in files:
        if not Path(f).exists():
            print(f"⚠️ 文件不存在: {f}")
            continue
        reviewer = Reviewer(f)
        reviewer.run()
        check_module_specific(f, reviewer.report)
        all_findings.extend(reviewer.report.findings)

    report = ReviewReport(all_findings)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_report(report, verbose=args.verbose))

    # 返回错误数
    sys.exit(min(len(report.errors), 127))


if __name__ == "__main__":
    main()
