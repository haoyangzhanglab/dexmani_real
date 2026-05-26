# XHand 本地通信与 `xhand.py` 使用说明

本文档面向一个最小、独立、joint-only 的 XHand 控制文件：`xhand.py`。当前设计只接收 12 维绝对关节角并下发给灵巧手，不包含 policy、JIT、history buffer、order mapping、ROS、原始 RS485/EtherCAT 协议实现或标定逻辑。

---

## 1. XHand 如何与本地台式机建立通信

### 1.1 硬件连接

XHand 通常通过 XH04 通讯调试线与本地台式机连接，并单独接入电源适配器。

```text
XHand  <->  XH04 通讯调试线  <->  本地台式机
      \->  电源适配器
```

通信方式有两种：

| 通信方式 | 物理接口   | `xhand.py` 中的 `comm_type` | 说明                                                         |
| -------- | ---------- | --------------------------- | ------------------------------------------------------------ |
| RS485    | USB / 串口 | `"RS485"`                   | 当前默认方式，通常设备名类似 `/dev/ttyUSB0`                  |
| EtherCAT | RJ45 网口  | `"EtherCAT"`                | 也可在代码里写成 `"Ethernet"`、`"eth"`、`"ecat"`，内部会统一映射到 `"EtherCAT"` |

调试前先确认：

- 灵巧手固定可靠，手指运动空间内没有障碍物。
- 电源适配器连接正确。
- XH04 调试线连接牢靠。
- 初次调试建议先用 XOS / XHAND 上位机确认设备能正常搜索、连接、运动和读取传感器。

### 1.2 安装 XHand SDK 与依赖

`xhand.py` 依赖官方 Python SDK：

```python
from xhand_controller import xhand_control as xh
```

如果导入失败，`connect()` 会返回 `False`，并设置：

```python
last_error_code = -1
last_error_message = "xhand_controller is not installed or cannot be imported"
```

安装方式以你的 SDK 包为准。一般流程包括：

```bash
pip install -r requirements.txt
pip install wheels/xhand_controller-*.whl
```

### 1.3 RS485 默认连接方式

`xhand.py` 默认使用 RS485：

```python
config = XHandConfig(
    comm_type="RS485",
    device_name="/dev/ttyUSB0",
)
```

如果 `device_name=None`，程序会调用 SDK 的：

```python
enumerate_devices("RS485")
```

然后自动选择第一个可用设备。

RS485 常见权限问题：

```bash
sudo chmod 666 /dev/ttyUSB0
```

或更宽松地临时调试：

```bash
sudo chmod 777 /dev/ttyUSB0
```

最小连接示例：

```python
from xhand import XHand, XHandConfig

hand = XHand(XHandConfig(
    comm_type="RS485",
    device_name="/dev/ttyUSB0",
))

if not hand.connect():
    raise RuntimeError(hand.last_error_message)

state = hand.get_state()
print(state["qpos"])

hand.disconnect()
```

### 1.4 切换到 EtherCAT / Ethernet

如果使用 RJ45 / EtherCAT，将配置改为：

```python
config = XHandConfig(
    comm_type="EtherCAT",
    device_name="enp3s0",  # 替换成你的实际网卡名
)
```

或者让 SDK 自动枚举第一个 EtherCAT 设备：

```python
config = XHandConfig(
    comm_type="EtherCAT",
    device_name=None,
)
```

若使用 conda 环境，EtherCAT 可能需要给当前 Python 解释器 raw socket 权限：

```bash
sudo setcap cap_net_raw+ep $(readlink -f $(which python3))
```

检查网卡名：

```bash
ip link
```

### 1.5 常见通信问题

| 问题                        | 可能原因                            | 处理方式                                                     |
| --------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| `xhand_controller` 导入失败 | SDK 未安装或 Python 环境不对        | 确认 wheel 安装在当前 Python 环境中                          |
| RS485 打不开                | 串口权限不足                        | `sudo chmod 666 /dev/ttyUSB0`                                |
| RS485 打不开                | 串口名错误                          | `ls /dev/ttyUSB*` 查看设备                                   |
| EtherCAT 打不开             | 网卡名错误                          | `ip link` 查看网卡名                                         |
| EtherCAT 打不开             | Python 无 raw socket 权限           | `sudo setcap cap_net_raw+ep $(readlink -f $(which python3))` |
| 可以读状态但不能控制        | 设备状态异常、固件/上位机状态未刷新 | 断电重启 XHand，并用 XOS 上位机检查                          |
| 触觉松手后仍有非零值        | 传感器存在偏置                      | 调用 `reset_sensor()` 清零                                   |

---

## 2. `xhand.py` 的参数物理意义、主要 API 用法、返回值说明

### 2.1 设计约束

当前 `xhand.py` 的边界很明确：

```text
输入：12 维绝对关节角 qpos，单位 rad，XHand 硬件顺序
输出：关节状态 qpos/current/timestamp
默认通信：RS485
可选通信：EtherCAT / Ethernet
不处理：policy、delta action、force control、标定、ROS、raw protocol、order mapping
```

核心控制链路：

```text
外部模块输出 qpos_target
        ↓
send_action(qpos_target)
        ↓
关节限位裁剪
        ↓
单步速度限制 max_qvel * dt
        ↓
更新缓存的 HandCommand_t
        ↓
XHandControl.send_command(device_id, hand_command)
```

### 2.2 关节顺序

`xhand.py` 使用 XHand 硬件顺序，不做 policy order 映射：

```python
JOINT_NAMES = [
    "thumb_abduction",   # 0
    "thumb_joint1",      # 1
    "thumb_joint2",      # 2
    "index_abduction",   # 3
    "index_joint1",      # 4
    "index_joint2",      # 5
    "middle_joint1",     # 6
    "middle_joint2",     # 7
    "ring_joint1",       # 8
    "ring_joint2",       # 9
    "little_joint1",     # 10
    "little_joint2",     # 11
]
```

触觉传感器 ID：

```python
SENSOR_IDS = [0x11, 0x12, 0x13, 0x14, 0x15]
SENSOR_NAMES = ["thumb", "index", "middle", "ring", "little"]
```

### 2.3 `XHandConfig` 参数说明

```python
@dataclass
class XHandConfig:
    comm_type: str = "RS485"
    device_name: str | None = None
    baudrate: int = 3_000_000
    device_id: int = 0

    dt: float = 1.0 / 83.0
    num_joints: int = 12
    force_update_state: bool = False

    home_qpos: np.ndarray = np.zeros(12)
    qpos_min: np.ndarray = ...
    qpos_max: np.ndarray = ...
    max_qvel: np.ndarray = np.deg2rad(np.ones(12) * 180.0)

    kp: int = 100
    ki: int = 0
    kd: int = 1
    tor_max: int = 100
    mode: int = 3

    use_delta_limit: bool = True
    clip_joint_limit: bool = True
    tactile_scale: float = 0.1
```

| 参数                 |  单位 |      默认值 | 物理意义                                                    |
| -------------------- | ----: | ----------: | ----------------------------------------------------------- |
| `comm_type`          |     - |   `"RS485"` | 通信方式。可设为 `"RS485"` 或 `"EtherCAT"`                  |
| `device_name`        |     - |      `None` | RS485 串口名或 EtherCAT 网卡名。`None` 时自动枚举第一个设备 |
| `baudrate`           |  baud |   `3000000` | RS485 波特率                                                |
| `device_id`          |     - |         `0` | 整手 ID                                                     |
| `dt`                 |     s |      `1/83` | 默认控制周期，用于计算单步最大关节变化量                    |
| `num_joints`         |     - |        `12` | XHand 主动关节数                                            |
| `force_update_state` |  bool |     `False` | `read_state()` 是否强制刷新。串口强制刷新可能更慢           |
| `home_qpos`          |   rad |   12 维全 0 | `reset()` 默认回到的关节角                                  |
| `qpos_min`           |   rad |      见代码 | 12 个关节最小角度限制                                       |
| `qpos_max`           |   rad |      见代码 | 12 个关节最大角度限制                                       |
| `max_qvel`           | rad/s | `180 deg/s` | Python 侧单步限速                                           |
| `kp`                 |     - |       `100` | 位置控制比例增益                                            |
| `ki`                 |     - |         `0` | 位置控制积分增益                                            |
| `kd`                 |     - |         `1` | 位置控制微分增益                                            |
| `tor_max`            |     - |       `100` | 输出限制。范围通常为 0 到 400，默认较保守                   |
| `mode`               |     - |         `3` | 关节控制模式。`0` 无力，`3` 位置，`5` 力控                  |
| `use_delta_limit`    |  bool |      `True` | 是否启用 `max_qvel * dt` 单步限速                           |
| `clip_joint_limit`   |  bool |      `True` | 是否裁剪到 `qpos_min/qpos_max`                              |
| `tactile_scale`      |     - |       `0.1` | 触觉原始 force 转 N 的比例                                  |

`dt` 不等价于自动 sleep。它参与计算：

```python
max_step = max_qvel * dt
qpos_cmd = last_qpos_cmd + clip(qpos_target - last_qpos_cmd, -max_step, max_step)
```

外部高频循环仍应显式控制频率：

```python
time.sleep(hand.config.dt)
```

### 2.4 主要 API

#### `connect() -> bool`

连接 XHand 设备。

内部逻辑：

```text
创建 XHandControl
标准化 comm_type
自动枚举或使用指定 device_name
RS485: open_serial(device_name, baudrate)
EtherCAT: open_ethercat(device_name)
list_hands_id()
build_command()
读取当前 qpos 作为 last_qpos_cmd
```

用法：

```python
hand = XHand(XHandConfig(comm_type="RS485", device_name="/dev/ttyUSB0"))
if not hand.connect():
    raise RuntimeError(hand.last_error_message)
```

#### `disconnect() -> None`

关闭当前设备：

```python
hand.disconnect()
```

#### `get_state(full: bool = False) -> dict`

默认返回轻量状态：

```python
{
    "qpos": np.ndarray,       # shape (12,), rad
    "current": np.ndarray,    # shape (12,), mA
    "timestamp": float,
}
```

`current` 来自 SDK 状态里的 `torque` 字段，但该字段实际表示实时电流，因此这里不命名为 `tau` 或 `torque`。

`full=True` 时额外返回：

```python
{
    "finger_ids": np.ndarray,
    "sensor_ids": np.ndarray,
    "raw_position": np.ndarray,
    "temperature": np.ndarray,
    "commboard_err": np.ndarray,
    "jointboard_err": np.ndarray,
    "tipboard_err": np.ndarray,

    "tactile_force": np.ndarray,       # shape (5, 120, 3), scaled, approx N
    "tactile_force_raw": np.ndarray,   # shape (5, 120, 3), raw
    "tactile_force_sum": np.ndarray,   # shape (5, 3), scaled, approx N
    "tactile_force_sum_raw": np.ndarray,
    "tactile_temperature": np.ndarray,

    "connected_flag": bool,
    "error_state": bool,
    "last_action_code": int | None,
    "last_error_code": int | None,
    "last_error_message": str,
    "last_joint_limit_clipped": bool,
    "last_delta_limited": bool,
    "last_hand_ids": list[int],
    "comm_type": str,
    "device_name": str | None,
    "joint_names": list[str],
    "sensor_names": list[str],
}
```

#### `send_action(action: np.ndarray) -> bool`

下发一帧 12 维绝对关节角。

```python
qpos = np.zeros(12, dtype=np.float64)
ok = hand.send_action(qpos)
```

输入语义固定：

```text
shape: (12,)
unit: rad
order: XHand hardware order
meaning: absolute joint position
```

返回：

```python
True   # SDK send_command 成功
False  # SDK send_command 失败或设备未连接
```

注意：`send_action()` 只发送一帧目标，不等待手指真实到位。如果目标很远且启用了 `use_delta_limit=True`，这一帧可能只是朝目标方向走一个小步。

#### `move_to_joint_positions(qpos, timeout=5.0, atol=0.03) -> bool`

阻塞式移动到目标关节角。

```python
target = np.zeros(12)
ok = hand.move_to_joint_positions(target, timeout=5.0, atol=0.03)
```

内部逻辑：

```text
目标关节角裁剪到 qpos_min/qpos_max
循环调用 send_action(target)
每次读取 get_state()
若 max(abs(current_qpos - target)) <= atol，则返回 True
若超过 timeout，则返回 False
```

这个函数和 `send_action()` 的区别：

| API                             | 是否阻塞 | 是否保证持续发送               | 推荐场景                          |
| ------------------------------- | -------- | ------------------------------ | --------------------------------- |
| `send_action(qpos)`             | 否       | 否，只发送一帧                 | 高频控制循环、teleop、外部 policy |
| `move_to_joint_positions(qpos)` | 是       | 是，循环发送直到接近目标或超时 | 手动调试、回 home、低频脚本       |

#### `reset(qpos=None, timeout=5.0, atol=0.03) -> bool`

回到 home，或者移动到指定姿态。

```python
hand.reset()          # 回 home_qpos，即 12 维全 0
hand.reset(qpos)      # 阻塞式移动到 qpos
```

当前实现中：

```python
reset(None) == move_to_joint_positions(home_qpos)
reset(qpos) == move_to_joint_positions(qpos)
```

#### `move_to_joint_positions(qpos) -> bool`

见上文。它已经不是 `send_action()` 的简单别名，而是阻塞式目标到达函数。

#### `stop() -> bool`

XHand 没有 xArm 那种 `emergency_stop()`。当前 `stop()` 的语义是：

```text
向 12 个关节发送 mode=0 无力模式
设置 error_state=True
```

这属于 **soft stop / no-force stop**，不是电气急停。

#### `clear_error() -> bool`

只清除 Python 驱动本地错误状态：

```python
hand.clear_error()
```

当前不会执行硬件级清错，因为 SDK 文档没有提供类似 xArm `clean_error()` 的统一硬件清错接口。

#### `reset_sensor(sensor_id=None) -> bool`

触觉传感器清零。

```python
hand.reset_sensor()       # 清零全部 5 个指尖触觉
hand.reset_sensor(0x11)   # 只清零大拇指触觉
```

当手指无外力但触觉值非零时，应调用该函数。

### 2.5 最小使用示例

```python
import numpy as np
from xhand import XHand, XHandConfig

config = XHandConfig(
    comm_type="RS485",
    device_name="/dev/ttyUSB0",
)

hand = XHand(config)

if not hand.connect():
    raise RuntimeError(hand.last_error_message)

try:
    state = hand.get_state()
    print("qpos:", state["qpos"])
    print("current:", state["current"])

    # 阻塞式回 home。XHand 的 home_qpos 是 12 维全 0。
    ok = hand.move_to_joint_positions(config.home_qpos, timeout=5.0, atol=0.03)
    print("move home ok:", ok)

finally:
    hand.disconnect()
```

### 2.6 高频控制循环示例

```python
import time
import numpy as np

state = hand.get_state()
target = state["qpos"].copy()

while True:
    target[4] += 0.001
    ok = hand.send_action(target)
    if not ok:
        print("send failed:", hand.last_error_message)
        break

    time.sleep(hand.config.dt)
```

---

## 3. 后续需要实现或验证的内容

### 3.1 真机兼容性校验

当前代码已经通过 Python 语法检查，但尚未真机验证。需要重点验证：

- `xhand_controller` 的实际 import 路径是否始终为 `from xhand_controller import xhand_control as xh`。
- `read_state()` 实际返回的是 tuple、list 还是 dict。
- SDK 字段是否为 `sensor_data` 还是 `senser_data`。
- SDK 字段是否为 `jonitboard_err` 还是 `jointboard_err`。
- `FingerState_t.torque` 是否稳定表示电流，单位 mA。
- `reset_sensor()` 在当前 SDK 版本中的返回值格式。

### 3.2 错误恢复机制

当前 `clear_error()` 只清本地错误状态。后续如果 SDK 提供硬件级清错接口，可以扩展为：

```text
清 SDK 错误
清板级错误
重新读取状态
必要时重新 build_command
恢复 connected_flag / error_state
```

### 3.3 更严格的输入策略

当前 `array12()` 的风格比较宽松：输入长度不足时用 NaN 填充，长度超过时截断。后续真机版本可以改得更严格：

```text
action 必须是 shape=(12,)
否则直接返回 False
```

这会更安全，但代码也会稍微更硬。

### 3.4 `move_to_joint_positions()` 到位判据优化

当前阻塞移动只用：

```python
max(abs(current_qpos - target)) <= atol
```

后续可以增加：

- 连续 N 帧满足阈值才算到位。
- 电流异常提前退出。
- 板级错误码非零提前退出。
- 可选打印当前误差。

### 3.5 触觉数据标准化与日志

当前 `full=True` 同时返回 raw 和 scaled 触觉数据。后续可以补：

- 触觉 offset / bias 管理。
- 自动 `reset_sensor()` 后记录 baseline。
- tactile flatten 工具函数，方便数据集写入。
- tactile summary，例如每个指尖合力模长。

### 3.6 与 xArm7 集成

如果后续需要 XArm7 + XHand 联合控制，建议保持两个驱动文件独立：

```text
xarm7.py     # 只管 7DoF arm
xhand.py     # 只管 12DoF hand
robot.py     # 可选，组合 arm + hand
```

`robot.py` 可以把动作拼成：

```python
arm_action = action[:7]
hand_action = action[7:19]
```

但不要把 XHand 逻辑塞进 `xarm7.py`，也不要把 xArm 逻辑塞进 `xhand.py`。