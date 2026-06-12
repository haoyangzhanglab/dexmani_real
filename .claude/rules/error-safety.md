# 错误处理与安全

## 错误处理三原则

1. **控制类函数优先返回 `bool`**：`connect()`、`send_action()`、`reset()`、`stop()` 等操作函数返回 `bool`
2. **失败设置状态变量**：`self.error_state = True` + `self.last_error_message = "简短原因"`
3. **不吞异常**：捕获后至少记录简短原因到 `last_error_message`，不让异常静默通过

> **ref:** P1 ManiUniCon `RobotInterface` ABC 所有方法均返回 `bool`。P2 Open-Teach `except: break`（bare except 吞所有异常）是反面教材。

## 异常粒度

```python
# 正确：捕获具体异常类型
try:
    self.arm.set_servo_angle_j(angles=target)
except XArmAPIError as e:
    self.error_state = True
    self.last_error_message = f"servo failed: {e}"
    return False

# 错误：bare except（仅在顶层主循环允许，且必须记录日志）
except:  # 禁止在非顶层使用
    break
```

顶层主循环（`run()` / `stream()` 的 while True）可以使用 bare except 兜底，但必须：
- 记录异常类型和 traceback
- 设置 `error_state=True`
- 不能静默退出

> **ref:** P1 LeFranX 使用类型化异常 `DeviceAlreadyConnectedError`、`DeviceNotConnectedError`。P2 Open-Teach `oculus.py` L103 `except: break` 为反面案例。

## get_state() 容错

`get_state()` 读失败时返回含 NaN 的默认结构，不抛异常：

```python
def get_state(self, full: bool = False) -> dict:
    try:
        qpos = self._read_qpos()
    except Exception as e:
        self.last_error_message = f"read qpos failed: {e}"
        qpos = np.full(self.n_joints, np.nan)
    return {"qpos": qpos, ...}
```

## stop() 语义

- 执行器：急停或软停，进入无力模式或停止控制循环
- 传感器：停止数据流，关闭 pipeline
- 必须在文档/注释中明确是 soft stop 还是硬件急停

> **ref:** P1 ManiUniCon `stop()` → `self.arm.emergency_stop()`（硬件急停）。本项目 xhand.py `stop()` → 发送 home 指令（soft stop）。

## clear_error() 原则

- 清除本地错误状态（`error_state=False`）
- 如 SDK 有清错接口则调用
- 无法清除的硬件故障必须在文档中说明需人工处理

## 危险操作

标定、固件更新、写 flash 等危险操作不放核心驱动默认路径：
- 做成独立工具脚本（`tools/calibrate_*.py`）
- 不在 `example()` 中自动执行

## Workspace 安全

```python
class WorkspaceSafety:
    def __init__(self, workspace_bounds: np.ndarray):
        """workspace_bounds: (3,2) [[x_min,x_max],[y_min,y_max],[z_min,z_max]] in meters."""

    def check(self, eef_pos: np.ndarray) -> bool: ...
    def clamp(self, target_pos: np.ndarray) -> np.ndarray: ...
```

每次 `_tick()` 中 EEF 目标位置需经 `WorkspaceSafety.check()` 验证。

> **ref:** 本项目 `planner/workspace_safety.py` 已实现完整接口，匹配 CLAUDE.md Section 2.9。

## SafetyMonitor（部署时）

```python
@dataclass
class SafetyStatus:
    ok: bool
    arm_ok: bool
    hand_ok: bool
    message: str = ""

class SafetyMonitor:
    def check(self, state: RobotState, action: RobotAction) -> SafetyStatus:
        """Arm: workspace / 关节限位 / 力矩
           Hand: 关节限位 / 电流(堵转) / 温度 / 通信"""
```

> **ref:** P2 BunnyVisionPro 遥操作中的 `error_threshold` 位置检查（`np.deg2rad(2)` 阈值）；P1 ManiUniCon 的 `validate_action()` 关节裁剪模式。
