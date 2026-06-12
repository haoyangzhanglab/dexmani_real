---
name: ref-check
description: Search, fact-check, and assess reference project code. Use when implementing a new module, looking up reference implementations across P1/P2/P3 projects, verifying hardware compatibility, or deciding adopt/adapt/skip.
---

# Reference Check Skill

按 CLAUDE.md Section 0.5.4 检索协议，在 5 个参考项目中搜索、验证、评估代码。

## 工具

`.claude/skills/ref-check/ref-search.py` — 跨参考项目搜索脚本。

## 工作流程

### Step 1: 查静态映射表

```bash
python .claude/skills/ref-check/ref-search.py <module>
```

示例：`python .claude/skills/ref-check/ref-search.py robot`

如果映射表中有对应条目，按 P1→P2→P3 排列。`✓` 表示文件存在，`✗` 表示路径可能已过期。

### Step 2: 动态搜索（当映射表未覆盖时）

```bash
python .claude/skills/ref-check/ref-search.py --keyword "<关键词>" -p <最高优先级>
```

示例：
- `python .claude/skills/ref-check/ref-search.py --keyword "servo_control" -p 2`
- `python .claude/skills/ref-check/ref-search.py --keyword "retarget" -p 3`

搜索结果按优先级分组，标注匹配类型 `[文件名]` 或 `[内容]`。会自动检查不采纳清单并提示已知的不可用方案。

### Step 3: Fact-Check（实际 Read 并验证）

**对于每个候选参考文件，必须执行：**

1. **Read 文件** — 不基于文件名猜测行为
2. **验证关键声明** — README/注释中的描述可能与实际代码不一致
   - 例：Bidex README 声称用 "SDLS" IK，实际代码使用 `p.IK_DLS`
3. **检查硬件兼容性** — 确认是否适用于本项目硬件
4. **检查依赖兼容性** — 是否引入不采纳的框架（ROS、Hydra 等）

### Step 4: 适用性评估

对每个参考文件做出判断：

| 判断 | 含义 | 注释格式 |
|------|------|---------|
| **Adopt** | 可直接参考/采纳核心逻辑 | `# ref: [P2] BunnyVisionPro xarm7_ability.py L120-150` |
| **Adapt** | 需适配硬件/依赖差异 | `# ref: [P2] Open-Teach allegro_retargeters.py — 仅参考滤波模式，手型需适配` |
| **Skip** | 不适用（框架不同/硬件不同/已标记不采纳） | 记录原因，不写入代码注释 |

### Step 5: 检查路径有效性

```bash
python .claude/skills/ref-check/ref-search.py <module> --check-exists
```

验证静态映射中的路径是否仍然有效。发现过期路径时更新 CLAUDE.md。

## 不采纳清单（自动提醒）

脚本在搜索时会自动匹配以下已知不采纳项并提醒：

| 不采纳 | 原因 |
|--------|------|
| libfranka + Ruckig C++ server | xArm7 内置伺服控制 |
| geofik + Brent 解析式 IK | 已有 MPlib 数值 IK |
| LeRobot Parquet / draccus CLI | 使用 HDF5 + @dataclass |
| Hydra 配置管理 | 使用 @dataclass |
| ROS/ROS2 通信层 | 使用 multiprocessing + shared memory |
| Vision Pro 专用 API | 本项目使用 Quest 3 |
| Allegro Hand 专用 retargeting | 手型差异 |
| Manus Core C++ SDK 直连 | 不依赖 Manus 数据手套 |

## Check 清单

开发新模块时必须经过：

- [ ] Step 1: 查了静态映射表
- [ ] Step 2: 对映射表未覆盖的子问题做了动态搜索
- [ ] Step 3: 实际 Read 了 P1 参考文件（至少 LeFranX + ManiUniCon 各一个）
- [ ] Step 3: 实际 Read 了 P2 参考文件（如果 P1 未覆盖）
- [ ] Step 4: 对每个参考文件做了 Adopt/Adapt/Skip 判断
- [ ] Step 4: 在代码中添加了正确格式的参考注释（含优先级标签和行号）
- [ ] Step 5: 验证了映射路径有效性

## 示例：搜索 robot 模块参考

```bash
$ python .claude/skills/ref-check/ref-search.py robot

## robot/ — 7 条参考:
  P   来源                     路径
  --- ---------------------- --------------------------------------------------
  P1  LeFranX                src/lerobot/robots/franka_fer_xhand/           ✓ Franka+XHand 复合接口
  P1  ManiUniCon             maniunicon/robot_interface/xarm6_robotiq.py    ✓ xArm6 接口
  P2  BunnyVisionPro         real_control/xarm7_ability.py                  ✓ xArm7 真机控制
  ...
```

## 示例：动态搜索未在映射表中的功能

```bash
$ python .claude/skills/ref-check/ref-search.py --keyword "servo_control" -p 2

搜索 'servo_control' — 找到 2 个文件:
  ── P2 ──
  P2 BunnyVisionPro  [内容] real_control/xarm7_ability.py
  P2 BunnyVisionPro  [内容] real_control/teleop_bimanual_xarm7_ability.py
```

## 示例：Fact-Check 发现的问题

| 参考项目 | 声明 | 实际情况 |
|---------|------|---------|
| Bidex README | "SDLS" IK method | 代码使用 `p.IK_DLS`，不是 SDLS |
| Bidex README | "Pybullet based SDLS retargeter" | 使用的是 PyBullet 内置 `calculateInverseKinematics2` + `IK_DLS` solver |

这体现了 fact-check 的核心价值：**README/文档中的描述可能与实际代码不一致，必须 Read 源码确认。**
