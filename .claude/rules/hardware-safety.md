# 真机操作安全规则

> **效力**: 本规则在真机操作时强制生效。违反任何一条必须中止操作，不得跳过。

## Pre-Flight 检查清单

每次调用任何 `send_action()` 前（包括首次连接后），必须通过以下检查：

### 连接前检查

- [ ] 确认机器人周围无人员、无障碍物
- [ ] 确认 E-Stop 按钮可用且在操作员伸手可及范围内
- [ ] 确认机械臂各关节无明显异常（线缆松动、异物等）
- [ ] 确认 XHand 手指可自由运动，无异物卡住

### 连接后检查（首次）

- [ ] `connect()` 返回 True
- [ ] `is_connected()` 返回 True
- [ ] `is_error()` 返回 False
- [ ] `get_state()` 返回有效关节角度（无 NaN）
- [ ] 关节角度在 `[qpos_min, qpos_max]` 范围内
- [ ] 当前关节角度与目测机器人实际姿态一致
- [ ] XHand: `finger_ids` 至少 12 个有效 ID，`commboard_err` 全零

### 首次运动前（缓慢验证）

每天首次操作或重启后，必须执行递增运动验证：

```
Step 1: 发送当前关节角度作为目标（stay in place）
        robot.send_action(current_qpos)
        → 验证: 返回 True, 机器人不抖动, 无异常噪音

Step 2: 发送小幅度关节运动（±2°）
        robot.send_action(current_qpos + np.deg2rad(2))
        → 验证: 关节平滑移动, 无碰撞, 电流正常

Step 3: 复位到 home
        robot.reset()
        → 验证: 所有关节到达 home, 路径无障碍

Step 4: 恢复正常操作
```

### 每帧检查（控制循环中）

控制循环中每个 `_tick()` 必须验证：

- [ ] `robot.is_error()` 返回 False
- [ ] `state.arm_connected` 和 `state.hand_connected` 为 True（hand 允许降级）
- [ ] `arm_qpos` 所有值在 `[qpos_min, qpos_max]` 内
- [ ] `workspace_safety.check(eef_pos)` 返回 True
- [ ] 相邻帧关节跳变 < 5°（arm）/ < 10°（hand）
- [ ] `arm_tau` 无突增（连续 3 帧 > 正常值 3x）
- [ ] `hand_current` 无持续过流（> 500mA 持续 > 1s）

## E-Stop 触发条件

以下情况立即调用 `robot.emergency_stop()`：

| 条件 | 检测方式 |
|------|---------|
| 关节位置超出 `[qpos_min, qpos_max]` | `np.any(qpos < qpos_min) or np.any(qpos > qpos_max)` |
| EEF 超出 workspace | `not workspace_safety.check(eef_pos)` |
| 关节力矩突增 > 5x 正常值 | `np.any(np.abs(tau) > tau_limit * 5)` |
| 手部持续过流 > 500mA > 1s | `np.any(hand_current > 500) and duration > 1.0` |
| 关节跳变 > 15°（arm）/ 30°（hand） | 单帧变化量异常 |
| 通信中断 (get_state 连续 10 帧 NaN) | `np.all(np.isnan(qpos))` |
| IK 连续失败 > 10 次 | 计数器 |
| VR 追踪丢失 > 1s | `tracking_lost_duration > 1.0` |

## E-Stop 后恢复流程

1. 确认并排除触发原因
2. 检查机械臂物理状态（关节、线缆、无异物）
3. `robot.clear_error()` 清除错误状态
4. 重新执行「首次运动前」的 Step 1-3 缓慢验证
5. 确认无异常后恢复正常操作

## Workspace 安全约束

机械臂安全操作空间（需根据实际工作台面调整）：

```python
WORKSPACE_BOUNDS = np.array([
    [0.2, 0.7],   # x: [min, max] m (前向)
    [-0.3, 0.3],  # y: [min, max] m (左右)
    [0.0, 0.6],   # z: [min, max] m (高度，桌面以上)
])
```

- EEF 目标位置在 workspace 外时：clamp 到最近边界，记录警告
- EEF 当前位置在 workspace 外时：停止运动，缓慢 return_to_home()

## 控制循环安全架构

控制循环中的安全层执行顺序：

```
_tick() 每次迭代:
  1. 读状态 (get_state)
  2. Pre-action 安全检查:
     a. error_state → emergency_stop
     b. workspace 越界 → clamp + warn
     c. 关节跳变异常 → emergency_stop
     d. 力矩/电流异常 → emergency_stop
  3. 计算动作 (IK + retarget)
  4. Post-action 安全检查:
     a. 目标关节在限位内 → clip
     b. 单步变化量在限速内 → delta limit
     c. EEF 目标在 workspace 内 → clamp
  5. send_action
  6. 记录质量 flags
```

## 禁止操作

以下操作在未明确授权时禁止：

- 在 `example()` 中自动执行运动（只读状态可以，运动必须有显式确认）
- 跳过 delta limit（`use_delta_limit=False` 仅在调试且有监督时允许）
- 在控制循环中执行阻塞操作（如 `time.sleep(>0.1)`、文件 I/O、打印大量日志）
- 在 `send_action` 外直接调用 SDK 的运动 API
- 在控制循环中 `print()` 大量输出（影响实时性）

## 安全配置检查

每次启动前验证以下配置正确：

```python
# 这些配置值直接影响安全，修改后必须 review
config.clip_joint_limit = True  # 绝对不能为 False
config.use_delta_limit = True   # 绝对不能为 False
config.qpos_min / qpos_max      # 必须与 xArm 硬件限位一致
config.max_qvel                 # 不能超过硬件最大速度
config.dt                       # 控制周期，与实际循环频率匹配
```
