---
name: code
description: Write new code that follows project conventions. Use when implementing a new module, adding a hardware driver, or writing any new Python file. Verifies interface contracts, runs reference lookup, and validates against CLAUDE.md checklists.
---

# Code Skill

按项目风格约束高效编写新代码。自动检索参考项目、生成检查清单、验证合规性。

驱动脚本: `.claude/skills/code/check_new_code.py`

## 工作流程

### Step 1: 确定模块类型

模块类型决定适用的接口契约：

| 类型 | 适用场景 |
|------|---------|
| `robot` | 硬件驱动 (XArm7, XHand 等执行器) |
| `sensor` | 传感器驱动 (RealSense 等) |
| `controller` | 遥操作控制器 (TeleopController) |
| `recording` | 数据录制 (EpisodeRecorder) |
| `data` | 离线数据读写 (EpisodeReader) |
| `deploy` | 策略部署 (PolicyRunner) |
| `teleop` | VR 追踪 + 手部重定向 |
| `planner` | 运动规划 (纯几何) |
| `utils` | 小型纯函数工具 |

### Step 2: 查参考项目

```bash
python .claude/skills/code/check_new_code.py --module <type> --refs
```

输出该模块在 P1→P2→P3 参考项目中的对应文件路径，带 ✓/✗ 存在性标记。

### Step 3: 读检查清单

```bash
python .claude/skills/code/check_new_code.py --module <type> --list-checks
```

输出该模块的完整检查清单（通用项 + 模块专项）。

### Step 4: Read 参考代码

按 P1→P2→P3 优先级 Read 参考文件，理解设计意图。在代码中添加参考注释：

```python
# ref: [P1] ManiUniCon xarm6_robotiq.py L46-80
# ref: [P2] BunnyVisionPro xarm7_ability.py L200-230
```

### Step 5: 验证代码

```bash
python .claude/skills/code/check_new_code.py --module <type> --file <path.py>
```

输出合规报告：❌ 错误（必须修复）、⚠️ 警告（建议修复）、ℹ️ 信息。

## Step 6: 检查 SDK 依赖合规性

修改硬件驱动时必须对照 `.claude/rules/sdk-dependencies.md` 验证：
- SDK API 调用是否正确（方法名、参数类型、返回码处理）
- 是否有已知陷阱未规避（如 XHand SDK 缓存零值、xArm `is_radian`）
- 安全裁剪（joint limit + delta limit）是否在 `send_action()` 中正确实现
- 错误处理是否符合 SDK 的返回模式（xArm 返回 int code, XHand 返回 err struct）

## 通用规则速查

所有新代码必须遵守 ⸻ 详见 `.claude/rules/`

| 规则 | 说明 |
|------|------|
| 命名 | 类 PascalCase, 函数 snake_case, 常量 UPPER_SNAKE_CASE |
| 配置 | `@dataclass` + `field(default_factory=...)`，以 Config 结尾 |
| 返回类型 | 控制函数 → `bool`，`get_state()` → `dict` |
| 错误处理 | 禁止 bare except，异常后设 `error_state` + `last_error_message` |
| 依赖 | 硬件驱动只依赖 SDK + numpy；cv2/torch 仅局部 import 或 deploy/ 中 |
| 模块结构 | 提供 `example()` 函数，复杂 CLI 放 `scripts/` |
| 参考注释 | 标注 `# ref: [P1] ProjectName file.py L120-150` |
| **SDK 合规** | 对照 `sdk-dependencies.md` 验证 API 调用、已知陷阱、安全层 |

## 接口契约速查

### 执行器 (robot/)

```
Config(dataclass) → Device.__init__ → connect()→bool → get_state(full=False)→dict
                                       → send_action(np.ndarray)→bool
                                       → reset()/stop()/is_connected()/is_error()/clear_error()
```

### 传感器 (sensor/)

```
Config(dataclass) → Sensor.__init__ → connect()→bool → get_state(full=False)→dict
                                      → stop()/is_connected()/is_error()/clear_error()
```

## 不采纳清单（自动提醒）

代码中禁止引入: Hydra, OmegaConf, ROS/ROS2, Pydantic, draccus, LeRobot Parquet。

检查脚本会自动检测 forbidden imports。

## 示例：写一个新的 robot 驱动

```bash
# 1. 查参考
$ python .claude/skills/code/check_new_code.py --module robot --refs

# 2. 看清单
$ python .claude/skills/code/check_new_code.py --module robot --list-checks

# 3. Read P1+P2 参考实现（按 ref-search.py 输出的路径）

# 4. 写代码...

# 5. 验证
$ python .claude/skills/code/check_new_code.py --module robot --file dexmani_real/robot/my_device.py
## robot 模块检查清单 (27 项)
  ...
## dexmani_real/robot/my_device.py — 0 error, 2 warnings
✅ 无错误
```
