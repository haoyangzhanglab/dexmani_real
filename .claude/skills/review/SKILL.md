---
name: review
description: Review code against project conventions. Use when asked to review code, check a PR, audit style compliance, or verify interface contracts. Runs comprehensive checks on naming, interfaces, safety, config, deps, and error handling.
---

# Review Skill

按项目风格约束全面审查代码。7 个维度自动检查 + 模块专项检查。

驱动脚本: `.claude/skills/review/review_code.py`

## 工作流程

### Step 1: 确定审查范围

```bash
# 审查指定文件
python .claude/skills/review/review_code.py <file1.py> [file2.py ...]

# 审查 git diff 变更文件
python .claude/skills/review/review_code.py --diff

# 审查暂存区
python .claude/skills/review/review_code.py --diff --staged

# 审查所有 Python 文件
python .claude/skills/review/review_code.py --all

# JSON 输出 (供自动化)
python .claude/skills/review/review_code.py --diff --json
```

### Step 2: 运行自动检查

驱动脚本自动执行 7 个维度检查：

| 维度 | 检查内容 | 依据 |
|------|---------|------|
| **naming** | 类名 PascalCase, 函数 snake_case, 方法名统一 (get_state 而非 get_obs) | CLAUDE.md §10 |
| **interface** | connect→bool, send_action→bool, 缺少必需方法 | CLAUDE.md §2 |
| **safety** | bare except, 安全裁剪, 状态变量 | CLAUDE.md §12 |
| **config** | @dataclass + default_factory, Config 后缀 | CLAUDE.md §9 |
| **style** | 前导下划线, argparse vs example() | CLAUDE.md §10,13 |
| **deps** | 禁止 Hydra/ROS/Pydantic, 硬件驱动禁 cv2/torch 顶层 import | CLAUDE.md §11 |
| **interface-special** | 传感器 start/stop → connect/disconnect | CLAUDE.md §2.3 |

### Step 3: Read 报告并修复

```bash
$ python .claude/skills/review/review_code.py dexmani_real/robot/my_device.py

## Review Report — Errors: 3  Warnings: 2  Info: 1

❌ [interface] my_device.py:45   send_action() 不应返回 dict，应返回 bool
❌ [safety]   my_device.py:30   __init__ 缺少 self.connected_flag
❌ [safety]   my_device.py:30   __init__ 缺少 self.error_state

### 警告 (2)
⚠️ [deps]     my_device.py:5    硬件驱动含重型依赖: cv2 应在函数内部局部 import
⚠️ [config]   my_device.py:22   'qpos_min' 可变默认值应使用 field(default_factory=...)
```

### Step 4: 人工审查

自动检查覆盖不了的项需要人工判断：

- [ ] 模块职责边界是否清晰（robot/ 不混入策略推理/可视化/数据记录）
- [ ] 参考注释是否标注了正确的来源和行号
- [ ] `get_state()` 默认是否足够轻量，full=True 是否包含调试信息
- [ ] `send_action()` 安全裁剪流水线是否完整
- [ ] stop() 语义是否明文档（soft stop vs 硬件急停）
- [ ] `__post_init__` 是否包含关键验证
- [ ] 物理量单位是否显式标注

### Step 5: 审查参考合规

确认新增/修改的模块是否遵循参考检索协议：

```bash
# 例如审查 controller 模块时，确认是否读了 P1+P2 参考
python .claude/skills/ref-check/ref-search.py controller
```

## 审查严重级别

| 级别 | 含义 | 处理 |
|------|------|------|
| ❌ **error** | 违反接口契约、安全规则 | 必须修复 |
| ⚠️ **warning** | 风格偏差、潜在问题 | 建议修复 |
| ℹ️ **info** | 改进建议 | 可忽略 |

## 按模块类型专项检查

驱动脚本会根据文件路径自动识别模块类型并执行专项检查：

| 路径匹配 | 专项检查 |
|---------|---------|
| `robot/` (非 model/) | 执行器接口完整检查（connect/send_action/get_state/状态变量） |
| `sensor/` | 传感器接口检查（connect/disconnect 命名、get_state 轻量性） |

## 示例：审查所有变更文件

```bash
$ python .claude/skills/review/review_code.py --diff
## Review Report — Errors: 0  Warnings: 3  Info: 5

⚠️ [style] robot/planner/arm_planner.py:312  模块使用 argparse 但未提供 example() 函数
⚠️ [config] teleop/hand_retarget.py:15  'qpos_min' 可变默认值应使用 field(default_factory=...)
ℹ️ [style] teleop/quest_hand_visualizer.py:6  公有属性建议不加前导下划线

### 建议 (5)
...
```

## 示例：审查暂存区（PR 前）

```bash
# 只看暂存区（即将提交的变更）
python .claude/skills/review/review_code.py --diff --staged --verbose

# 如果有 error，修复后再次检查
python .claude/skills/review/review_code.py --diff --staged
# → ✅ 未发现问题
```
