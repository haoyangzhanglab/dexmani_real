#!/usr/bin/env python3
"""Robot-Pi 研究驱动脚本 — 论文搜索、参考对比、模块需求提取。

用法:
  python .claude/skills/robot-pi/research.py --search "<topic>"
  python .claude/skills/robot-pi/research.py --compare "<arch_choice>"
  python .claude/skills/robot-pi/research.py --brief <module>
  python .claude/skills/robot-pi/research.py --assess <paper_url>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REF_SEARCH = PROJECT_ROOT / ".claude/skills/ref-check/ref-search.py"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"

# ── 模块需求定义（从 CLAUDE.md 提取）──────────────────────────────

MODULE_REQUIREMENTS = {
    "controller": {
        "interface": "TeleopController",
        "key_methods": ["_tick(vr_frame) → RobotAction"],
        "state_machine": "IDLE → TELEOP → RECORDING → EMERGENCY_STOP",
        "key_rules": [
            "Arm: VR wrist → EEF → IK (不平滑，保留原始追踪)",
            "Hand: landmarks → retarget → EMA 平滑 (alpha=0.3)",
            "关节跳变 clamp (arm + hand 独立限速)",
            "VR re-anchoring: 进入 RECORDING 前 arm_mapper.reset()",
            "键盘信号通过 multiprocessing.Queue (T/R/S/H/ESC)",
        ],
        "claude_section": "Section 4",
        "refs": "controller/",
    },
    "robot_interface": {
        "interface": "RobotInterface (复合 arm+hand)",
        "key_methods": [
            "connect() → dict[str, bool]",
            "get_state() → RobotState",
            "send_action(RobotAction) → dict[str, bool]",
            "return_to_home(use_planning=True) → bool",
            "emergency_stop() → None",
        ],
        "key_rules": [
            "控制器和部署只通过此接口操作硬件，不直接调 XArm7/XHand",
            "hand 断连时降级运行 (arm 仍可工作)",
            "{arm: True, hand: False} 表示部分连接",
        ],
        "claude_section": "Section 2.8",
        "refs": "robot/ (RobotInterface 在映射表中属于 robot 模块)",
    },
    "recording": {
        "interface": "EpisodeRecorder",
        "key_methods": [
            "start_episode(task_label, operator, tags) → bool",
            "add_frame(state, action, vr_frame, quality_flags) → bool",
            "stop_episode(success=True) → str | None",
        ],
        "key_rules": [
            "HDF5 结构: /obs, /action, /vr, /camera, /meta",
            "quality_flags: 11-bit (TRACKING_OK, IK_SUCCESS, ...)",
            "相机内外参自包含在 episode 中",
            "与 Controller 解耦，只负责数据写入",
        ],
        "claude_section": "Section 5",
        "refs": "recording/",
    },
    "deploy": {
        "interface": "PolicyRunner + PolicyLoader",
        "key_methods": [
            "PolicyLoader.load(checkpoint_dir) → (model, stats, config)",
            "PolicyRunner.run() — 主循环含 action chunk + EMA",
        ],
        "key_rules": [
            "手动加载模型 (config.json + safetensors + stats.json)",
            "部署时 arm+hand 都做 EMA 平滑 (alpha=0.5)",
            "SafetyMonitor: workspace/关节/力矩/电流/温度",
            "ObservationBuilder 独立于模型",
            "ActionParser 支持 arm_only/hand_only/full 模式",
        ],
        "claude_section": "Section 7",
        "refs": "deploy/",
    },
    "data": {
        "interface": "EpisodeReader + DataValidator",
        "key_methods": [
            "EpisodeReader(path) — 懒加载",
            "read(key) → np.ndarray",
            "iter_frames(skip_rejected=True) → Iterator[dict]",
            "DataValidator.validate(path) → ValidationReport",
        ],
        "key_rules": [
            "懒加载 HDF5，不一次性读入内存",
            "iter_frames 支持 skip_rejected",
            "DataValidator: nan/inf/shape/时间戳/关节范围/电流",
            "convert_data.py 输出 per-joint 归一化统计量",
        ],
        "claude_section": "Section 6",
        "refs": "data/",
    },
    "ipc": {
        "interface": "SharedRingBuffer",
        "key_methods": [
            "write(data: bytes, seq: int) → int",
            "read(last_seq: int) → tuple[bytes|None, int]",
            "close()",
        ],
        "key_rules": [
            "slot_count=64, slot_size=1MB",
            "两阶段握手: robot_ready ↔ policy_ready (mp.Event)",
            "键盘事件通过 multiprocessing.Queue 传递",
            "no-ipc 回退到单进程模式",
        ],
        "claude_section": "Section 3",
        "refs": "ipc/",
    },
}

# ── 架构对比维度 ──────────────────────────────────────────────────

ARCH_COMPARISONS = {
    "hand retargeting": {
        "question": "如何将 21 个 VR hand landmark 重定向到 XHand 12-DOF 关节？",
        "p1_lefranx": {
            "path": "src/lerobot/teleoperators/xhand_vr/vr_hand_detector_adapter.py",
            "approach": "landmark → hand frame → IK (DexPilot 方法)",
            "pros": "已在 XHand 上验证，重构代码可直接复用",
            "cons": "依赖 DexPilot 框架",
        },
        "p1_maniunicon": {
            "path": "third_party/oculus_reader/",
            "approach": "直接读取 Quest landmark，无内置 retargeting",
            "pros": "轻量级 VR 数据读取",
            "cons": "不包含 retargeting 逻辑，需自行实现",
        },
        "p2_bunnyvisionpro": {
            "path": "examples/retargeting/retargeting.py",
            "approach": "OPERATOR2MANO 坐标系变换 + dex-retargeting",
            "pros": "坐标变换清晰，支持双手对齐模式",
            "cons": "依赖 dex-retargeting 库",
        },
        "p2_openteach": {
            "path": "openteach/robot/allegro/allegro_retargeters.py",
            "approach": "关节限位 + 移动平均滤波",
            "pros": "简单有效，滤波模式可复用",
            "cons": "Allegro 手型特定，需适配 XHand",
        },
        "project_status": "teleop/hand_retarget.py 已有初步实现",
        "recommendation": "Adopt LeFranX DexPilot 核心逻辑 + Adapt Open-Teach 滤波模式 + BunnyVisionPro 坐标变换",
    },
    "ik solver": {
        "question": "用什么 IK solver 驱动 xArm7？",
        "p1_maniunicon": {
            "path": "maniunicon/utils/ik_solver.py",
            "approach": "数值 IK (基于 MPlib)，支持碰撞检测",
            "pros": "已在 xArm6 上验证，支持避障",
            "cons": "初始化较慢，需 URDF",
        },
        "p1_lefranx": {
            "path": "src/lerobot/teleoperators/franka_fer_vr/arm_ik_processor.py",
            "approach": "解析 IK (geofik + Brent) 仅用于 Franka",
            "pros": "极快（<1ms）",
            "cons": "仅适用于 Franka，不通用，本项目不采纳",
        },
        "p2_bunnyvisionpro": {
            "path": "real_control/xarm7_ability.py",
            "approach": "Pinocchio FK/IK",
            "pros": "快速，支持 xArm7 URDF",
            "cons": "无碰撞检测",
        },
        "project_status": "planner/ik.py 已有 MPlib IK 实现",
        "recommendation": "保持 MPlib 数值 IK（已实现 + 碰撞检测），Pinocchio 作为备选（速度优先时）",
    },
    "recording format": {
        "question": "用什么格式录制多模态数据？",
        "p1_lefranx": {
            "path": "(lerobot 包内部)",
            "approach": "LeRobot Parquet 格式",
            "pros": "与 LeRobot 训练管线集成",
            "cons": "依赖 LeRobot 框架，本项目不采纳",
        },
        "p1_maniunicon": {
            "path": "maniunicon/utils/replay_buffer.py",
            "approach": "内存 buffer → TimestampAlignedBuffer",
            "pros": "时间戳对齐，质量过滤",
            "cons": "需序列化到磁盘",
        },
        "p2_bunnyvisionpro": {
            "path": "real_control/teleop_bimanual_xarm7_ability.py",
            "approach": "h5py 扁平数据集 (单文件)",
            "pros": "简单直接，已在 xArm7 上使用",
            "cons": "无嵌套分组，无元数据规范",
        },
        "project_status": "使用 HDF5 (CLAUDE.md Section 5.1 已定义结构)",
        "recommendation": "Adopt BunnyVisionPro 的 h5py 写入模式 + 本项目定义的嵌套 HDF5 结构 + ManiUniCon 的时间戳对齐",
    },
    "control rate": {
        "question": "如何保证控制循环的稳定频率？",
        "p1_maniunicon": {
            "path": "maniunicon/policies/quest.py",
            "approach": "RateLimiter 类（补偿计算耗时）",
            "pros": "长期频率准确，简洁",
            "cons": "单线程同步模式，不适用于多进程",
        },
        "p2_bunnyvisionpro": {
            "path": "real_control/xarm7_ability.py",
            "approach": "wait_until_next_control_signal()",
            "pros": "与后台控制线程配合良好",
            "cons": "需手动管理控制线程生命周期",
        },
        "recommendation": "Adopt ManiUniCon RateLimiter 模式，CLAUDE.md Section 4.4 已标准化",
    },
}


# ── 论文搜索模板 ──────────────────────────────────────────────────

PAPER_SEARCH_QUERIES = {
    "imitation learning": [
        "site:arxiv.org \"action chunking\" transformer robot manipulation 2024 2025",
        "site:arxiv.org \"diffusion policy\" bimanual manipulation 2024 2025",
        "site:arxiv.org \"imitation learning\" \"xarm\" OR \"franka\" teleoperation 2025",
    ],
    "hand manipulation": [
        "site:arxiv.org \"dexterous hand\" \"imitation learning\" teleoperation 2024 2025",
        "site:arxiv.org \"hand retargeting\" \"virtual reality\" robot 2024 2025",
    ],
    "teleoperation": [
        "site:arxiv.org \"bimanual teleoperation\" \"virtual reality\" robot 2024 2025",
        "site:arxiv.org \"low-cost teleoperation\" robot imitation learning 2025",
    ],
    "sim-to-real": [
        "site:arxiv.org \"sim-to-real\" \"imitation learning\" manipulation 2024 2025",
        "site:arxiv.org \"domain randomization\" \"robot manipulation\" 2025",
    ],
}


# ── CLI ──────────────────────────────────────────────────────────────

def search_papers(topic: str) -> dict:
    """生成针对 topic 的论文搜索策略。"""
    queries = PAPER_SEARCH_QUERIES.get(topic.lower(), [
        f"site:arxiv.org \"{topic}\" robot manipulation 2024 2025",
        f"site:arxiv.org \"{topic}\" \"imitation learning\" 2025",
    ])
    return {
        "topic": topic,
        "queries": queries,
        "instruction": "对每个搜索结果的 Top 5，使用 WebFetch 阅读摘要并评估："
                       "1) 是否开源 2) 硬件要求 3) 与参考项目的重叠度 4) 对本项目的适用性",
    }


def assess_paper(url: str) -> dict:
    """生成论文评估框架。"""
    return {
        "url": url,
        "fetch_instruction": f"WebFetch({url}, prompt=\"提取: 1)核心方法一句话 2)硬件要求 "
                             "3)是否开源代码/权重 4)关键性能指标 5)方法局限性\")",
        "assess_dimensions": [
            "硬件匹配: 是否适配 xArm7 + XHand + Quest 3？",
            "依赖匹配: 是否依赖 ROS/Hydra/特定硬件？",
            "与参考的重叠: P1/P2 参考项目是否已有类似实现？",
            "工程成本: 集成复杂度？是否需要大量改造？",
            "性能: 是否显著优于当前方案？",
        ],
    }


def generate_brief(module: str) -> str:
    """生成模块需求摘要。"""
    if module not in MODULE_REQUIREMENTS:
        return f"未知模块 '{module}'。可选: {list(MODULE_REQUIREMENTS.keys())}"

    req = MODULE_REQUIREMENTS[module]
    lines = [
        f"## {module.upper()} 模块需求",
        f"",
        f"**接口**: `{req['interface']}`",
        f"**CLAUDE.md**: {req['claude_section']}",
        f"**参考映射**: {req['refs']}",
        f"",
        f"### 核心方法",
    ]
    for m in req["key_methods"]:
        lines.append(f"  - `{m}`")

    lines.append("")
    lines.append("### 关键规则")
    for r in req["key_rules"]:
        lines.append(f"  - {r}")

    # 运行 ref-search 获取参考项目
    ref_module = module.replace("robot_interface", "robot").replace("ipc", "ipc")
    if REF_SEARCH.exists():
        result = subprocess.run(
            [sys.executable, str(REF_SEARCH), ref_module],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines.append("")
            lines.append(result.stdout)

    return "\n".join(lines)


def generate_comparison(topic: str) -> str:
    """生成架构对比报告。"""
    if topic not in ARCH_COMPARISONS:
        similar = [k for k in ARCH_COMPARISONS if topic.lower() in k.lower()]
        hint = f"\n建议搜索: {similar}" if similar else ""
        return f"未找到架构对比 '{topic}'。可用对比: {list(ARCH_COMPARISONS.keys())}{hint}"

    c = ARCH_COMPARISONS[topic]
    lines = [f"## 架构对比: {topic}", "", f"**问题**: {c['question']}", ""]

    for key in sorted(c.keys()):
        if key.startswith("p1_") or key.startswith("p2_") or key.startswith("p3_"):
            entry = c[key]
            label = key.replace("p1_", "P1 ").replace("p2_", "P2 ").replace("p3_", "P3 ").replace("_", " ").title()
            lines.append(f"### {label}")
            if "path" in entry:
                lines.append(f"  - **路径**: `{entry['path']}`")
            lines.append(f"  - **方法**: {entry['approach']}")
            if "pros" in entry:
                lines.append(f"  - **优点**: {entry['pros']}")
            if "cons" in entry:
                lines.append(f"  - **缺点**: {entry['cons']}")
            lines.append("")

    if "project_status" in c:
        lines.append(f"### 项目现状")
        lines.append(f"  {c['project_status']}")
        lines.append("")

    if "recommendation" in c:
        lines.append(f"### 推荐方案")
        lines.append(f"  {c['recommendation']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Robot-Pi 研究驱动")
    parser.add_argument("--search", help="搜索论文主题 (生成搜索策略)")
    parser.add_argument("--assess", help="评估论文 URL (生成评估框架)")
    parser.add_argument("--compare", help="对比架构选择 (从预定义对比中选择)")
    parser.add_argument("--brief", help="生成模块需求摘要")
    parser.add_argument("--list-arch", action="store_true", help="列出所有可对比的架构选择")
    parser.add_argument("--list-modules", action="store_true", help="列出所有模块")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    output = []

    if args.search:
        result = search_papers(args.search)
        output.append(f"## 论文搜索策略: {args.search}\n")
        for i, q in enumerate(result["queries"], 1):
            output.append(f"  {i}. `{q}`")
        output.append(f"\n### 评估指南\n{result['instruction']}")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

    elif args.assess:
        result = assess_paper(args.assess)
        output.append(f"## 论文评估: {args.assess}\n")
        output.append(f"### 第一步: 获取论文内容")
        output.append(f"  {result['fetch_instruction']}\n")
        output.append(f"### 第二步: 适用性评估")
        for d in result["assess_dimensions"]:
            output.append(f"  - {d}")
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

    elif args.compare:
        output.append(generate_comparison(args.compare))

    elif args.brief:
        output.append(generate_brief(args.brief))

    elif args.list_arch:
        output.append("## 可对比的架构选择\n")
        for i, key in enumerate(sorted(ARCH_COMPARISONS.keys()), 1):
            c = ARCH_COMPARISONS[key]
            output.append(f"  {i}. **{key}** — {c['question'][:80]}...")

    elif args.list_modules:
        output.append("## 模块列表\n")
        for i, (key, req) in enumerate(sorted(MODULE_REQUIREMENTS.items()), 1):
            output.append(f"  {i}. **{key}** — `{req['interface']}` ({req['claude_section']})")

    else:
        output.append("用法: --search | --assess | --compare | --brief | --list-arch | --list-modules")

    print("\n".join(output))


if __name__ == "__main__":
    main()
