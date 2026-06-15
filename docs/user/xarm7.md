# xArm7 本地通信与 `xarm7.py` 使用说明

本文档面向一个最小、独立、joint-only 的 xArm7 控制文件：`xarm7.py`。当前设计假设外部模块已经完成 IK，`xarm7.py` 只接收 7 维绝对关节角并下发给真机，不读取也不控制 TCP pose。

---

## 1. xArm 如何与本地台式机建立通信

### 1.1 硬件连接

推荐方式是 **xArm 控制器与本地台式机通过网线直连**：

```text
xArm7 机械臂  <->  xArm 控制器  <->  网线  <->  台式机网口
```

在上电和连线前，先确认：

- 机械臂底座固定可靠，工作空间内没有障碍物。
- 控制器电源、机械臂线缆连接正确。
- 急停按钮处于释放状态。
- 调试时建议同时打开 UFACTORY Studio，用于观察状态、错误码和运动模式。

### 1.2 配置本地网卡 IP

xArm 控制器默认 IP 通常位于：

```text
192.168.1.xxx
```

本地台式机网卡必须与机械臂在同一网段，但 IP 不能完全相同。例如：

| 设备        | 示例 IP         |
| ----------- | --------------- |
| xArm 控制器 | `192.168.1.111` |
| 本地台式机  | `192.168.1.10`  |
| 子网掩码    | `255.255.255.0` |

在 Linux/Ubuntu 上，可以临时配置网卡，例如网卡名是 `enp3s0`：

```bash
sudo ip addr add 192.168.1.10/24 dev enp3s0
sudo ip link set enp3s0 up
```

检查是否连通：

```bash
ping 192.168.1.111
```

如果能稳定收到回复，说明基础网络通信正常。

### 1.3 访问 UFACTORY Studio

在浏览器中访问：

```text
http://<xArm控制器IP>:18333
```

例如：

```text
http://192.168.1.111:18333
```

UFACTORY Studio 可用于查看当前机械臂 IP、模式、状态、错误码、关节角、TCP 信息、负载和安装方式。即使使用 Python SDK 或自定义 TCP 协议调试，官方手册也建议在调试阶段保持 Studio 运行，方便观察系统状态。

### 1.4 安装 Python SDK

安装官方 Python SDK：

```bash
pip install xarm-python-sdk
```

最小连通性测试：

```python
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.1.111", is_radian=True)
print("connected:", arm.connected)
print("state:", arm.state)
print("mode:", arm.mode)
print("error_code:", arm.error_code)
print("warn_code:", arm.warn_code)
arm.disconnect()
```

如果 `connected=True`，说明 Python SDK 可以连接到控制器。

### 1.5 常见通信问题

| 问题                 | 可能原因                              | 处理方式                                     |
| -------------------- | ------------------------------------- | -------------------------------------------- |
| `ping` 不通          | 网卡 IP 不在 `192.168.1.xxx` 网段     | 重新设置本机网卡 IP                          |
| `ping` 不通          | 网线、控制器电源或急停状态异常        | 检查物理连接和控制器状态                     |
| 浏览器打不开 Studio  | 地址格式错误                          | 使用 `http://IP:18333`                       |
| Python SDK 连接失败  | IP 写错或网络不通                     | 先 `ping`，再运行 Python                     |
| 机械臂不响应运动指令 | 未使能、未 `set_state(0)`、存在错误码 | 在 Studio 查看错误码，必要时 `clear_error()` |

---

## 2. `xarm7.py` 的参数物理意义、主要 API 用法、返回值说明

### 2.1 设计约束

当前 `xarm7.py` 的边界很明确：

```text
输入：7 维绝对关节角 qpos，单位 rad
输出：关节状态 qpos/qvel/tau/timestamp
不处理：TCP pose、Cartesian action、IK、夹爪、相机、多进程 shared memory
```

核心控制链路：

```text
policy / IK 输出 qpos_target
        ↓
send_action(qpos_target)
        ↓
关节限位裁剪
        ↓
单步速度限制 max_qvel * dt
        ↓
set_servo_angle_j(qpos_cmd)
```

`send_action()` 使用 xArm 的 `set_servo_angle_j()`。该接口要求机械臂处于 servo motion mode，即 `set_mode(1)`；它只执行最新收到的指令，且 `speed`、`mvacc`、`mvtime` 在该接口中是 reserved。因此，本文件通过 `max_qvel * dt` 在 Python 侧限制每一帧关节目标的最大变化量。

### 2.2 `XArm7Config` 参数说明

```python
@dataclass
class XArm7Config:
    ip: str = "192.168.1.111"
    dt: float = 1.0 / 50.0
    init_qpos: np.ndarray = np.zeros(7)
    qpos_min: np.ndarray = np.deg2rad([-360, -118, -360, -11, -360, -97, -360])
    qpos_max: np.ndarray = np.deg2rad([360, 120, 360, 225, 360, 180, 360])
    max_qvel: np.ndarray = np.deg2rad([90, 90, 90, 90, 120, 120, 150])
    reset_speed: float = np.deg2rad(20)
    reset_acc: float = np.deg2rad(180)
    use_delta_limit: bool = True
    clip_joint_limit: bool = True
```

| 参数               |   单位 |                            默认值 | 物理意义                                                     |
| ------------------ | -----: | --------------------------------: | ------------------------------------------------------------ |
| `ip`               |      - |                   `192.168.1.111` | xArm 控制器 IP 地址                                          |
| `dt`               |      s |                            `0.02` | 在线控制默认周期。用于计算每次 `send_action()` 允许的最大关节变化量 |
| `init_qpos`        |    rad |                 `[0,0,0,0,0,0,0]` | `reset()` 默认回到的关节角                                   |
| `qpos_min`         |    rad |                            见代码 | 7 个关节的最小角度限制                                       |
| `qpos_max`         |    rad |                            见代码 | 7 个关节的最大角度限制                                       |
| `max_qvel`         |  rad/s | `[90,90,90,90,120,120,150] deg/s` | Python 侧的最大关节速度限制，用于 servoj 单步限速            |
| `reset_speed`      |  rad/s |                        `20 deg/s` | `reset()` 阻塞式关节运动速度                                 |
| `reset_acc`        | rad/s² |                      `180 deg/s²` | `reset()` 阻塞式关节运动加速度                               |
| `use_delta_limit`  |   bool |                            `True` | 是否启用 `max_qvel * dt` 单步限速                            |
| `clip_joint_limit` |   bool |                            `True` | 是否把目标关节角裁剪到 `qpos_min/qpos_max`                   |

`dt` 不等价于自动 sleep。它只参与计算：

```python
max_step = max_qvel * dt
qpos_cmd = last_qpos_cmd + clip(qpos_target - last_qpos_cmd, -max_step, max_step)
```

外部控制循环仍应显式控制频率，例如：

```python
time.sleep(robot.config.dt)
```

### 2.3 主要 API

#### `connect() -> bool`

连接 xArm 控制器，并执行：

```text
clean_error()
clean_warn()
motion_enable(True)
set_mode(1)
set_state(0)
```

成功返回 `True`，失败返回 `False`。

用法：

```python
from xarm7 import XArm7, XArm7Config

robot = XArm7(XArm7Config(ip="192.168.1.111"))
if not robot.connect():
    raise RuntimeError("failed to connect xArm7")
```

#### `disconnect() -> None`

断开 SDK 与控制器连接。

```python
robot.disconnect()
```

#### `is_connected() -> bool`

返回当前本地接口是否认为机械臂可用。它综合检查：

```text
arm 对象是否存在
SDK 是否 connected
connected_flag
error_state
```

#### `is_error() -> bool`

返回当前是否处于错误/不可继续控制状态。当前判断逻辑：

```text
arm 不存在 -> True
SDK disconnected -> True
本地 error_state=True -> True
arm.error_code != 0 -> True
arm.state in [4, 5, 6] -> True
```

其中 `state=4/5/6` 分别对应停止、系统重置、减速停止等不可继续接收正常运动指令的状态。

#### `clear_error() -> bool`

用于错误恢复。当前逻辑：

```text
clean_error()
clean_warn()
motion_enable(True)
set_mode(0)
set_state(0)
error_state = False
```

注意：`clear_error()` 后会回到 position mode。下一次调用 `send_action()` 时，如果发现当前不在 servo mode，会自动切回 `set_mode(1)`。

#### `stop() -> bool`

调用 SDK 的 `emergency_stop()`，并把本地 `error_state` 置为 `True`。

```python
robot.stop()
```

停止后，不应继续发送 `send_action()`。应先排查错误码，再调用 `clear_error()`，必要时重新 `reset()`。

#### `set_servo_mode() -> None`

切换到在线关节伺服模式：

```text
set_mode(1)
set_state(0)
```

`send_action()` 会使用该模式。

#### `set_position_mode() -> None`

切换到位置控制模式：

```text
set_mode(0)
set_state(0)
```

`reset()` 和 `move_to_joint_positions()` 会使用该模式。

#### `reset(qpos: np.ndarray | None = None) -> bool`

阻塞式移动到某个关节角。如果 `qpos=None`，则移动到 `config.init_qpos`。

```python
robot.reset()
robot.reset(np.zeros(7))
```

内部调用 SDK 的 `set_servo_angle(..., wait=True)`，因此 `reset_speed` 和 `reset_acc` 在这里生效。

#### `move_to_joint_positions(qpos: np.ndarray) -> bool`

语义更明确的阻塞式关节移动，内部等价于：

```python
return robot.reset(qpos)
```

适合调试阶段使用，不建议在高频 policy loop 中调用。

#### `get_state(full: bool = False) -> dict`

读取机械臂状态。

默认 `full=False` 返回最小必要量：

```python
state = robot.get_state()

state = {
    "qpos": np.ndarray,      # shape (7,), rad
    "qvel": np.ndarray,      # shape (7,), rad/s
    "tau": np.ndarray,       # shape (7,), SDK effort/torque value
    "timestamp": float,      # Python time.time()
}
```

`qpos/qvel/tau` 优先来自：

```python
arm.get_joint_states(is_radian=True, num=3)
```

如果该接口失败，会 fallback 到：

```text
qpos -> get_servo_angle(is_radian=True)
qvel -> arm.realtime_joint_speeds
 tau -> get_joints_torque()
```

`full=True` 时额外返回调试和安全信息：

```python
state = robot.get_state(full=True)
```

| 字段                       | 含义                             |
| -------------------------- | -------------------------------- |
| `mode`                     | xArm 当前控制模式                |
| `state`                    | xArm 当前状态                    |
| `connected`                | SDK 连接状态                     |
| `connected_flag`           | `xarm7.py` 本地连接标记          |
| `error_state`              | `xarm7.py` 本地错误标记          |
| `error_code`               | 控制器错误码                     |
| `warn_code`                | 控制器警告码                     |
| `cmd_num`                  | 控制器缓存指令数量               |
| `last_action_code`         | 最近一次动作 SDK 返回码          |
| `last_joint_limit_clipped` | 最近一次动作是否触发关节限位裁剪 |
| `last_delta_limited`       | 最近一次动作是否触发单步限速     |
| `servo_codes`              | 各伺服状态/错误码                |
| `temperatures`             | 各关节温度                       |
| `currents`                 | 各关节电流                       |
| `voltages`                 | 各关节电压                       |
| `motor_enable_states`      | 电机使能状态                     |
| `motor_brake_states`       | 电机制动状态                     |

#### `send_action(action: np.ndarray) -> bool`

唯一在线控制接口。只接收 7 维绝对关节角，单位 rad：

```python
target_qpos = np.zeros(7, dtype=np.float64)
ok = robot.send_action(target_qpos)
```

内部逻辑：

```text
1. action -> np.ndarray, reshape(7)
2. 如果 clip_joint_limit=True，裁剪到 qpos_min/qpos_max
3. 如果 use_delta_limit=True，执行 max_qvel * dt 单步限速
4. 如果当前不是 servo mode，则 set_mode(1), set_state(0)
5. 调用 set_servo_angle_j(angles=qpos_cmd, is_radian=True)
6. SDK code == 0 返回 True，否则返回 False 并设置 error_state=True
```

`send_action()` 不返回 debug dict。如果需要查看最近一次动作是否被裁剪或限速，使用：

```python
state = robot.get_state(full=True)
print(state["last_action_code"])
print(state["last_joint_limit_clipped"])
print(state["last_delta_limited"])
```

### 2.4 CLI 测试入口

```bash
python xarm7.py --ip 192.168.1.111 --test state
python xarm7.py --ip 192.168.1.111 --test full-state
python xarm7.py --ip 192.168.1.111 --test reset
python xarm7.py --ip 192.168.1.111 --test hold --seconds 3
```

| 测试项       | 行为                                          |
| ------------ | --------------------------------------------- |
| `state`      | 只读取默认状态，不主动运动                    |
| `full-state` | 读取完整调试状态，不主动运动                  |
| `reset`      | 阻塞式移动到 `init_qpos`                      |
| `hold`       | 读取当前关节角，并用 `send_action()` 保持几秒 |

第一次真机测试建议顺序：

```text
state -> full-state -> hold -> reset
```

不要一开始直接运行大幅度动作。

### 2.5 典型 policy loop

```python
import time
import numpy as np
from xarm7 import XArm7, XArm7Config

robot = XArm7(XArm7Config(ip="192.168.1.111"))
assert robot.connect()

try:
    state = robot.get_state()
    qpos = state["qpos"]

    while True:
        # 这里假设外部 policy 或 IK 已经输出 7 维绝对关节角，单位 rad
        target_qpos = qpos + np.deg2rad([1, 0, 0, 0, 0, 0, 0])

        ok = robot.send_action(target_qpos)
        if not ok or robot.is_error():
            print(robot.get_state(full=True))
            break

        time.sleep(robot.config.dt)
        qpos = robot.get_state()["qpos"]

finally:
    robot.disconnect()
```

---

## 3. 相关文档

- [quest.md](quest.md) — VR 遥操作输入（Quest hand tracking、arm wrist mapping、hand retargeting）
- [xhand.md](xhand.md) — XHand 灵巧手硬件控制与 API
- [realsense.md](realsense.md) — RealSense 相机驱动与点云工具
- [CLAUDE.md](../CLAUDE.md) — 项目接口规范（RobotInterface、安全约束、架构总览）

---

## 参考来源

- UFACTORY Studio 用户手册 V2.6.0：连接方式、Studio 调试、模式/状态、关节范围、错误恢复流程。
- xArm-Python-SDK 官方仓库与 API 文档：`XArmAPI`、`is_radian`、`set_servo_angle`、`set_servo_angle_j`、`get_joint_states`、`emergency_stop`。
- PyPI `xarm-python-sdk`：安装方式与支持产品。