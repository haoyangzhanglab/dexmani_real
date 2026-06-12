# 硬件驱动接口契约

## 执行器类设备（Arm、Hand）

**`robot/xhand.py` 是项目内的 XHand 模板，新硬件驱动必须遵循此接口。**

```python
@dataclass
class DeviceConfig:
    """配置 dataclass。可变默认值使用 default_factory。"""
    dt: float = 1.0 / 50.0
    home_qpos: np.ndarray = field(default_factory=lambda: np.zeros(7))
    qpos_min: np.ndarray = field(default_factory=lambda: ...)
    qpos_max: np.ndarray = field(default_factory=lambda: ...)
    max_qvel: np.ndarray = field(default_factory=lambda: ...)
    use_delta_limit: bool = True
    clip_joint_limit: bool = True

class Device:
    def __init__(self, config: DeviceConfig):
        self.config = config
        # 必须的状态变量：
        self.connected_flag = False
        self.error_state = False
        self.last_error_message = ""
        self.last_qpos_cmd: np.ndarray | None = None
        self.last_cmd_time: float | None = None
        self.last_joint_limit_clipped = False
        self.last_delta_limited = False

    def connect(self) -> bool:
        """最小可用初始化。成功返回 True 并设 connected_flag=True。
        失败返回 False 并设 error_state + last_error_message。
        connect() 应幂等：重复调用已连接设备返回 True。"""

    def disconnect(self) -> None: ...

    def get_state(self, full: bool = False) -> dict[str, Any]:
        """默认返回 {"qpos", "qvel", "timestamp"}。
        full=True 额外返回 SDK 原始字段、错误码、内部 flags。"""

    def send_action(self, action: np.ndarray) -> bool:
        """只接受 np.ndarray 一种动作类型，返回 bool。
        内部做 range clip + delta limit，裁剪状态记录到 last_* 变量。
        失败设 error_state=True。"""

    def reset(self, target: np.ndarray | None = None) -> bool: ...
    def stop(self) -> bool: ...
    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

> **ref:** P1 ManiUniCon `RobotInterface` ABC 定义了 `connect()/send_action()/get_state()` 均返回 bool，本项目遵循此模式。P1 LeFranX 的 `send_action()` 返回 dict 且 `connect()` 抛异常，本项目不采纳（dict 返回污染调用方，异常打断控制流）。

## RobotInterface 复合接口

控制器和部署模块只通过 `RobotInterface` 操作硬件，不直接调 `XArm7`/`XHand`：

```python
@dataclass
class RobotState:
    arm_qpos: np.ndarray       # (7,) rad
    arm_qvel: np.ndarray       # (7,) rad/s
    arm_tau: np.ndarray        # (7,) N·m
    eef_pos: np.ndarray        # (3,) m
    eef_quat_wxyz: np.ndarray  # (4,)
    hand_qpos: np.ndarray      # (12,) rad
    hand_current: np.ndarray   # (12,) mA
    hand_tactile_sum: np.ndarray  # (5,3) N
    hand_temperature: np.ndarray  # (12,) °C
    arm_connected: bool
    hand_connected: bool
    hand_error: bool
    timestamp: float

    def __post_init__(self):
        """验证 shape 和取值范围。替代 Pydantic 自动验证。"""

@dataclass
class RobotAction:
    arm_qpos_cmd: np.ndarray       # (7,) rad
    hand_qpos_cmd: np.ndarray      # (12,) rad
    target_eef_pose: np.ndarray | None  # (7,) pos+quat_wxyz

class RobotInterface:
    def connect(self) -> dict[str, bool]: ...
    def get_state(self) -> RobotState: ...
    def send_action(self, action: RobotAction) -> dict[str, bool]: ...
    def return_to_home(self, use_planning: bool = True,
                       cancel_event=None) -> bool: ...
    def emergency_stop(self) -> None: ...
    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

`RobotInterface.send_action()` 返回 `dict[str, bool]` 以区分子设备状态。底层 `XArm7.send_action()` / `XHand.send_action()` 只返回 `bool`。

> **ref:** P1 LeFranX `franka_fer_xhand/` 目录的 arm+hand 复合模式；P1 ManiUniCon 的 `RobotInterface` ABC 接口定义。

## 传感器类设备

```python
@dataclass
class SensorConfig: ...

class Sensor:
    def connect(self) -> bool: ...
    def disconnect(self) -> None: ...
    def get_state(self, full: bool = False) -> dict: ...
    def stop(self) -> bool: ...
    def is_connected(self) -> bool: ...
    def is_error(self) -> bool: ...
    def clear_error(self) -> bool: ...
```

- 传感器没有 `send_action()`，除非设备确实有主动控制命令
- 传感器数据与派生观测分离：点云生成放 `utils/`，不放传感器驱动内
- 统一使用 `connect()/disconnect()` 而非 `start()/stop()`

> **ref:** P1 ManiUniCon `maniunicon/sensors/realsense.py`；P2 Open-Teach `robot_camera.py` / `fish_eye_camera.py`。

## 动作安全层

`send_action()` 内部必须包含以下流水线：

```
shape 规整 → 数值类型转换 → 物理范围裁剪(joint limit)
→ 单步变化量限制(delta limit) → 错误状态检查
```

裁剪结果记录到 `self.last_joint_limit_clipped` / `self.last_delta_limited`，不污染返回值。

> **ref:** P1 ManiUniCon `validate_action()` 直接 `np.clip` 关节位置/速度/力矩；本项目 xhand.py 的 `limit_joint_range()` + `limit_joint_step()` 模式。

## 线程安全（后台控制线程）

如使用后台线程进行实时控制（如 BunnyVisionPro 的 PID 控制线程），必须：

```python
self.arm_lock = threading.Lock()

def control_arm_qpos(self, target: np.ndarray):
    with self.arm_lock:
        self._arm_target = target.copy()
```

- 共享目标变量通过 `threading.Lock` 保护
- 锁命名包含被保护变量名
- 后台线程捕获异常后设置 `error_state=True`，不静默退出

> **ref:** P2 BunnyVisionPro `xarm7_ability.py` L196-230 的 `_arm_lock` + 后台控制线程模式。
