---
name: test
description: Test hardware drivers and robot modules. Use when asked to test a device, run a smoke test, validate a driver, or verify hardware connectivity. Supports offline validation and real-hardware smoke tests.
---

# Test Skill

硬件驱动测试工具。支持两种模式：离线接口验证 + 真机烟雾测试。

驱动脚本: `.claude/skills/test/driver.py`

## 测试模式

| 模式 | 命令 | 硬件需求 | 时间 |
|------|------|---------|------|
| **离线** | `--file <f> --offline` | 无 | <1s |
| **仿真** | `--sim` | 无 (SAPIEN) | ~5s |
| **真机 L1** | `--file <f> --smoke` | 真机 | ~5s |
| **真机 L2** | `--file <f> --smoke --small-move` | 真机 | ~10s |
| **真机 L3** | `--file <f> --smoke --full` | 真机 | ~15s |

## 工作流

### Step 0: 仿真测试（无硬件，完整功能验证）

```bash
python .claude/skills/test/driver.py --sim
```

使用 SAPIEN headless 仿真环境，验证：
- `SimRobotInterface` 接口与真机 `RobotInterface` 一致
- `connect → get_state → send_action(stay) → send_action(small-move) → reset → disconnect` 完整链路
- IK 往返精度 (max_err < 0.0001)
- 关节运动量验证

适合 CI 集成和开发阶段的功能验证。

### Step 1: 离线验证（无硬件，安全快速）

```bash
python .claude/skills/test/driver.py --file <device.py> --offline
```

自动检查：
- Config dataclass: 是否有 `@dataclass`、`default_factory` 使用正确
- 接口完整性: `connect/get_state/send_action/reset/stop/is_connected/is_error/clear_error` 是否存在
- 类型签名: `connect() → bool`、`send_action(np.ndarray) → bool`、`get_state(full=) → dict`
- 状态变量: `connected_flag`、`error_state`、`last_error_message` 是否在 `__init__` 中初始化
- 安全裁剪: `send_action` 是否包含 joint limit / delta limit
- 禁止模式: bare except、禁止依赖、get_obs 命名

### Step 2: 生成测试脚本

```bash
python .claude/skills/test/driver.py --file <device.py> --generate
```

生成一个可直接运行的烟雾测试脚本，包含：
- `connect()` → 验证返回 True
- `is_connected()` → 验证返回 True
- `is_error()` → 验证返回 False
- `get_state()` → 验证无 NaN、shape 正确
- `send_action(current_qpos)` → stay-in-place 命令
- `disconnect()` → 清理

输出到 `/tmp/test_<DeviceName>.py`

### Step 3: 真机烟雾测试（谨慎！需硬件连接）

```bash
# 安全操作 — 仅发送 stay-in-place 命令（目标 = 当前位置）
python .claude/skills/test/driver.py --file <device.py> --smoke

# 含小幅度运动验证
python .claude/skills/test/driver.py --file <device.py> --smoke --small-move

# 完整测试（含 reset 往返）
python .claude/skills/test/driver.py --file <device.py> --smoke --full
```

真机测试前必须通过 pre-flight 检查（详见 `.claude/rules/hardware-safety.md`）。

### 烟雾测试级别

| 级别 | 命令 | 内容 | 风险 |
|------|------|------|------|
| **L1** stay-in-place | `--smoke` | connect → read state → send current qpos → disconnect | 零风险（当前位置不动） |
| **L2** small-move | `--smoke --small-move` | L1 + 小幅度关节运动（±2°） | 低风险 |
| **L3** full | `--smoke --full` | L2 + reset 往返 + 状态一致性检查 | 中等风险 |

### Step 4: 批量验证所有驱动

```bash
python .claude/skills/test/driver.py --all-offline
```

验证 `robot/` 和 `sensor/` 下所有设备文件。

## 测试成功后输出

```
## XArm7 — 6/6 通过

  ✓ Config: XArm7Config @dataclass (7 字段, default_factory 正确)
  ✓ connect() → bool
  ✓ get_state(full=False) → dict (含 qpos, qvel, tau, timestamp)
  ✓ send_action(np.ndarray) → bool (含 joint limit + delta limit)
  ✓ 状态变量: connected_flag, error_state, last_error_message 完整
  ✓ example() 函数存在
```

## 测试失败输出

```
## XArm7 — 2 错误

  ❌ send_action() 返回类型: 期望 bool，实际 dict
  ❌ 缺失方法: is_connected()
  ⚠️  Config 建议: qpos_min 可变默认值应使用 field(default_factory=...)
```

必须修复所有错误后才能部署到真机。

## Pre-Flight 安全协议

真机测试前，遵循 `.claude/rules/hardware-safety.md` 的强制清单。在控制循环 `_tick()` 中，安全层执行顺序：

```
1. get_state() → 2. Pre-action 安全检查 → 3. 计算动作 
→ 4. Post-action 安全检查 → 5. send_action() → 6. 记录 quality flags
```

E-Stop 触发条件（立即调用 `robot.emergency_stop()`）：
- 关节超出限位、EEF 超出 workspace
- 关节力矩突增 > 5x 正常值
- 手部持续过流 > 500mA > 1s
- 通信中断（连续 10 帧 NaN）

## 示例：测试刚重构的 xarm7.py

```bash
$ python .claude/skills/test/driver.py --file dexmani_real/robot/xarm7.py --offline
## XArm7 — 6/6 通过 ✓
```

## 示例：生成烟雾测试脚本

```bash
$ python .claude/skills/test/driver.py --file dexmani_real/robot/xarm7.py --generate
# → 已生成 /tmp/test_XArm7.py
# 使用: python /tmp/test_XArm7.py --ip 192.168.1.111
```
