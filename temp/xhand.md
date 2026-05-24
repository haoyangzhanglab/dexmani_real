# XHAND1 Python Wrapper 使用说明

本文档说明如何使用当前版本的 `xhand_latest.py` wrapper 控制星动纪元 / RobotEra XHAND1 灵巧手。本文默认你已经拿到厂商提供的 XHAND Python SDK，并且代码中可以正常导入：

```python
from xhand_controller import xhand_control
```

当前 wrapper 面向 **真实硬件控制、状态读取、触觉数据采集、teleop / policy rollout / imitation learning 数据采集**。代码保持薄封装，不做复杂 device manager，也不在底层强制 sleep 控频。

---

## 1. 文件结构建议

建议将 wrapper 保存为：

```text
your_project/
├── xhand.py
├── example_xhand.py
└── ...
```

如果你直接使用当前文件名 `xhand_latest.py`，则导入方式为：

```python
from xhand_latest import XHand, XHandConfig
```

如果你重命名为 `xhand.py`，则导入方式为：

```python
from xhand import XHand, XHandConfig
```

---

## 2. 安装与环境

### 2.1 推荐系统

厂商快速使用说明中推荐：

```text
Ubuntu 20.04 LTS
Python 3.8.x
```

建议先在 Ubuntu 20.04 上使用厂商安装包跑通 XOS / SDK，再接入本 wrapper。

### 2.2 安装系统依赖

首次安装厂商软件时，一般需要：

```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk
sudo apt-get install -y python3-pip
sudo apt-get install -y libboost-filesystem-dev
```

根据厂商上位机版本，可能还需要：

```bash
sudo pip3 install pybind11
sudo apt-get install -y nlohmann-json3-dev
```

具体是否需要 `pybind11` / `nlohmann-json3-dev`，以厂商当前安装说明为准。

### 2.3 安装 XHAND 软件包

使用厂商提供的 `.deb` 包：

```bash
sudo dpkg -i xhand_v1.x.x.deb
```

如需卸载旧版本：

```bash
sudo dpkg -r xhand
```

注意：厂商文档中说明应在普通用户权限下安装和使用，不建议在 root 用户或免密权限用户环境下安装。

### 2.4 验证 Python SDK 是否可导入

进入你的 Python 环境后执行：

```bash
python3 - <<'PY'
from xhand_controller import xhand_control
print("XHAND SDK import OK")
print("SDK version:", xhand_control.XHandControl().get_sdk_version())
PY
```

如果这里报：

```text
ModuleNotFoundError: No module named 'xhand_controller'
```

说明 Python 没找到厂商 SDK。常见原因包括：

- `.deb` 没安装成功；
- 当前 Python 环境和 SDK 安装环境不是同一个；
- 使用了 conda / venv，但 SDK 安装在系统 Python 下；
- `PYTHONPATH` 未包含 SDK 路径。

---

## 3. 硬件连接说明

XHAND1 支持两种通信方式：

```text
EtherCAT
RS485
```

当前 wrapper 也只支持这两种：

```python
XHandConfig(comm_type="EtherCAT")
XHandConfig(comm_type="RS485")
```

### 3.1 EtherCAT 连接

硬件连接：

```text
XHAND1
  ├── 电源适配器
  └── XH04 调试线 RJ45 接口 → 电脑以太网口
```

软件侧可以自动枚举第一个 EtherCAT 网卡：

```python
cfg = XHandConfig(comm_type="EtherCAT", ifname=None)
hand = XHand(cfg).connect()
```

也可以手动指定网卡名：

```python
cfg = XHandConfig(comm_type="EtherCAT", ifname="enp3s0")
hand = XHand(cfg).connect()
```

查看电脑网卡名：

```bash
ip link
```

常见网卡名形如：

```text
enp3s0
enp4s0
eth0
```

不建议把 Wi-Fi 网卡当作 EtherCAT 接口。一般应使用有线 RJ45 网口。

---

## 4. RS485 连接说明

### 4.1 RS485 物理连接

硬件连接：

```text
XHAND1
  ├── 电源适配器
  └── XH04 调试线 USB 接口 → 电脑 USB 口
```

XH04 调试线插入电脑后，Linux 通常会生成类似：

```text
/dev/ttyUSB0
/dev/ttyUSB1
/dev/ttyACM0
```

的串口设备。

### 4.2 RS485 波特率是多少？

XHAND SDK 文档给出的 RS485 波特率是：

```text
3000000
```

也就是：

```python
baudrate = 3_000_000
```

当前 wrapper 默认也是：

```python
baudrate: int = 3_000_000
```

所以一般不需要自己猜波特率，直接使用默认值即可。

### 4.3 怎么知道 RS485 对应哪个端口？

有三种方式，推荐顺序如下。

#### 方法 A：让 SDK 自动枚举

最简单：

```python
cfg = XHandConfig(comm_type="RS485", serial_port=None)
hand = XHand(cfg).connect()
```

内部会调用：

```python
self.device.enumerate_devices("RS485")
```

然后选择第一个枚举到的设备。

如果你的电脑只插了一个 XHAND USB-RS485 调试线，这通常足够。

#### 方法 B：用 `/dev/serial/by-id/` 找稳定设备名

推荐用于长期实验，因为 `/dev/ttyUSB0` 可能会随着插拔顺序变化，而 `/dev/serial/by-id/...` 更稳定。

查看：

```bash
ls -l /dev/serial/by-id/
```

可能看到类似：

```text
usb-XXX_USB_Serial-if00-port0 -> ../../ttyUSB0
```

此时可以直接把完整路径传给 wrapper：

```python
cfg = XHandConfig(
    comm_type="RS485",
    serial_port="/dev/serial/by-id/usb-XXX_USB_Serial-if00-port0",
    baudrate=3_000_000,
)
hand = XHand(cfg).connect()
```

#### 方法 C：插拔前后对比 `/dev/ttyUSB*`

插入 XH04 USB 之前：

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

插入之后再执行一次：

```bash
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

新出现的那个一般就是 XHAND 对应端口。

也可以实时看内核日志：

```bash
dmesg -w
```

然后插入 USB 调试线，观察是否出现：

```text
ttyUSB0
ttyUSB1
```

之类的设备名。

### 4.4 查看或设置串口波特率

一般不需要手动设置，因为 wrapper 会调用：

```python
open_serial(port, 3_000_000)
```

如果你只是想检查串口当前配置，可以用：

```bash
stty -F /dev/ttyUSB0 -a
```

如需临时设置：

```bash
stty -F /dev/ttyUSB0 3000000
```

注意：实际使用时还是以 SDK 的 `open_serial(port, baudrate)` 为准。

### 4.5 串口权限问题

如果打开串口时报权限错误，可以查看设备权限：

```bash
ls -l /dev/ttyUSB0
```

常见情况是设备属于 `dialout` 组。可以把当前用户加入该组：

```bash
sudo usermod -aG dialout $USER
```

然后 **退出登录并重新登录**，或者重启终端会话。

临时调试也可以：

```bash
sudo chmod a+rw /dev/ttyUSB0
```

但这个方法重启或重新插拔后可能失效，不适合作为长期方案。

---

## 5. 关节与传感器 ID

### 5.1 关节 ID

当前 wrapper 使用 12 维关节向量，顺序为关节 ID `0~11`：

| index | 关节 |
|---:|---|
| 0 | 大拇指偏摆 |
| 1 | 大拇指 1 关节 |
| 2 | 大拇指 2 关节 |
| 3 | 食指偏摆 |
| 4 | 食指 1 关节 |
| 5 | 食指 2 关节 |
| 6 | 中指 1 关节 |
| 7 | 中指 2 关节 |
| 8 | 无名指 1 关节 |
| 9 | 无名指 2 关节 |
| 10 | 小指 1 关节 |
| 11 | 小指 2 关节 |

因此所有动作必须是：

```python
q.shape == (12,)
```

### 5.2 指尖传感器 ID

五个指尖触觉传感器 ID：

| 指尖 | 十进制 ID | 十六进制 ID |
|---|---:|---:|
| 大拇指 | 17 | `0x11` |
| 食指 | 18 | `0x12` |
| 中指 | 19 | `0x13` |
| 无名指 | 20 | `0x14` |
| 小指 | 21 | `0x15` |

当前 wrapper 中：

```python
sensor_ids = (17, 18, 19, 20, 21)
```

---

## 6. 控制模式

当前建议只把下面几个模式当作常用模式：

| mode | 含义 | 用途 |
|---:|---|---|
| 0 | 无力 / powerless | 示教、手动掰动、停止输出 |
| 3 | 位置控制 | 默认控制模式 |
| 5 | 力控 / 电流相关模式 | 谨慎使用，当前 wrapper 未封装目标电流接口 |

当前 wrapper 默认：

```python
mode = 3
```

停止输出：

```python
hand.stop()
```

内部等价于：

```python
hand.set_mode(0)
```

注意：虽然底层资料里还能看到 `mode=1/2` 这类模式，但当前 wrapper 不建议直接暴露或使用，除非厂商明确说明当前固件下这些模式的具体含义和字段映射。

---

## 7. `XHandConfig` 配置项

```python
@dataclass
class XHandConfig:
    comm_type: str = "EtherCAT"
    ifname: str | None = None
    serial_port: str | None = None
    baudrate: int = 3_000_000
    hand_id: int | None = None

    mode: int = 3
    kp: int = 100
    ki: int = 0
    kd: int = 0
    tor_max: int = 300

    q_min: np.ndarray = ...
    q_max: np.ndarray = ...
    q_home: np.ndarray = ...

    force_update: bool = False
```

### 7.1 通信配置

| 字段 | 含义 |
|---|---|
| `comm_type` | `"EtherCAT"` 或 `"RS485"` |
| `ifname` | EtherCAT 网卡名；为 `None` 时自动枚举第一个 |
| `serial_port` | RS485 串口；为 `None` 时自动枚举第一个 |
| `baudrate` | RS485 波特率，默认 `3_000_000` |
| `hand_id` | 灵巧手 ID；为 `None` 时自动选择第一个 |

### 7.2 控制参数

| 字段 | 含义 |
|---|---|
| `mode` | 控制模式，默认 `3` 位置控制 |
| `kp` | 比例增益 |
| `ki` | 积分增益 |
| `kd` | 微分增益 |
| `tor_max` | 力矩/输出上限，SDK 常用范围 `0~400` |

### 7.3 关节限位

| 字段 | 含义 |
|---|---|
| `q_min` | 12 维关节下限 |
| `q_max` | 12 维关节上限 |
| `q_home` | 12 维 home 位置 |

`send_action(q)` 会自动调用：

```python
q = np.clip(q, q_min, q_max)
```

并且会检查：

```python
q.shape == (12,)
```

---

## 8. `XHand` 函数接口

### 8.1 `connect()`

```python
hand.connect()
```

打开 XHAND 通信设备。

- `comm_type="EtherCAT"` 时调用 `open_ethercat(ifname)`；
- `comm_type="RS485"` 时调用 `open_serial(port, baudrate)`；
- 如果 `hand_id=None`，自动选择第一个连接到的手。

返回：

```python
self
```

示例：

```python
hand = XHand(XHandConfig(comm_type="EtherCAT")).connect()
```

---

### 8.2 `close()`

```python
hand.close()
```

关闭设备。建议程序退出前一定调用。

推荐写法：

```python
hand = XHand(cfg)
try:
    hand.connect()
    ...
finally:
    hand.close()
```

---

### 8.3 `send_action(q)`

```python
q_sent = hand.send_action(q)
```

发送 12 维关节目标位置。

输入：

```python
q: np.ndarray, shape == (12,)
```

行为：

1. 检查 shape 是否为 `(12,)`；
2. 按 `q_min/q_max` 裁剪；
3. 写入 12 个 `FingerCommand_t.position`；
4. 调用 SDK `send_command()`；
5. 更新 `self.q_cmd` 和 `step_idx`。

返回：

```python
q_sent
```

也就是裁剪后的实际发送目标。

示例：

```python
q = np.zeros(12, dtype=np.float32)
q_sent = hand.send_action(q)
```

---

### 8.4 `get_observation(force_update=None, full=False)`

```python
obs = hand.get_observation()
```

读取当前状态。

默认返回核心 obs：

| key | shape / 类型 | 含义 |
|---|---:|---|
| `t_ns` | `int` | 本机单调时钟时间戳，单位 ns |
| `dt` | `float` | 与上一次 obs 的时间间隔 |
| `step_idx` | `int` | 已发送 action 次数 |
| `q` | `(12,) float32` | 关节角度 |
| `dq` | `(12,) float32` | 关节速度；若 SDK 无速度字段，则由相邻两帧 `q` 差分得到 |
| `current` | `(12,) float32` | 关节电流/力矩相关反馈；SDK 字段名为 `torque` |
| `tactile_force` | `(5, 3) float32` | 五个指尖三维合力 |
| `tactile_raw` | `(5, 120, 3) float32` | 五个指尖触觉阵列 |

`full=True` 时额外返回：

| key | 含义 |
|---|---|
| `joint_temp` | 关节板温度 |
| `palm_temp` | 掌板温度 |
| `tactile_temp` | 指尖触觉温度 |
| `tactile_raw_count` | 每个指尖实际读到的 raw force 点数 |
| `tactile_temp_raw` | 原始温度阵列 |
| `comm_err` | 通信板错误码 |
| `joint_err` | 关节板错误码 |
| `tip_err` | 指尖板错误码 |
| `state` | SDK 原始状态对象 |

示例：

```python
obs = hand.get_observation()
print(obs["q"])
print(obs["tactile_force"])
print(obs["tactile_raw"].shape)  # (5, 120, 3)
```

读取完整 debug 信息：

```python
obs = hand.get_observation(full=True)
print(obs["comm_err"])
print(obs["joint_err"])
print(obs["tip_err"])
```

---

### 8.5 `set_mode(mode, kp=None, ki=None, kd=None, tor_max=None)`

```python
hand.set_mode(mode, kp=None, ki=None, kd=None, tor_max=None)
```

切换控制模式，并可同步更新 PID / 输出上限。

示例：切回位置控制：

```python
hand.set_mode(3, kp=100, ki=0, kd=0, tor_max=300)
```

示例：无力模式：

```python
hand.set_mode(0)
```

通常更建议用：

```python
hand.stop()
```

---

### 8.6 `stop()`

```python
hand.stop()
```

进入 `mode=0` 无力 / powerless 模式。

---

### 8.7 `reset(q=None, sensor=False)`

```python
obs = hand.reset(q=None, sensor=False)
```

发送一个初始位置，并返回一次 observation。

- `q=None` 时使用 `cfg.q_home`；
- `sensor=True` 时会调用 `reset_sensor()` 清零五个指尖触觉传感器。

示例：

```python
obs = hand.reset(sensor=True)
```

注意：这个函数只是发送一次目标位置，不会等待手完全运动到位。

---

### 8.8 `reset_sensor()`

```python
hand.reset_sensor()
```

依次对五个指尖传感器执行清零 / 复位：

```python
17, 18, 19, 20, 21
```

通常在以下情况使用：

- 开始采集前；
- 触觉合力松手后仍有偏置；
- 换实验物体前；
- 长时间运行后触觉零点漂移。

---

### 8.9 `get_meta_info(refresh=False)`

```python
meta = hand.get_meta_info()
```

读取设备元信息。

返回示例：

```python
{
    "sdk_version": ...,
    "hardware_version": ...,
    "hand_id": ...,
    "hand_type": ...,
    "serial_number": ...,
    "hand_name": ...,
    "ev_hand": ...,
    "is_calibrated": ...,
}
```

如果 `refresh=False` 且之前读取过，会返回缓存。

强制刷新：

```python
meta = hand.get_meta_info(refresh=True)
```

---

## 9. 使用示例

### 9.1 EtherCAT 最小示例

```python
import numpy as np
from xhand import XHand, XHandConfig

cfg = XHandConfig(
    comm_type="EtherCAT",
    ifname=None,
)

hand = XHand(cfg)

try:
    hand.connect()
    print(hand.get_meta_info())

    obs = hand.reset(sensor=False)
    print("q:", obs["q"])
    print("tactile force:", obs["tactile_force"])

    q = np.zeros(12, dtype=np.float32)
    hand.send_action(q)

finally:
    hand.close()
```

---

### 9.2 RS485 最小示例：自动选择端口

```python
import numpy as np
from xhand import XHand, XHandConfig

cfg = XHandConfig(
    comm_type="RS485",
    serial_port=None,
    baudrate=3_000_000,
)

hand = XHand(cfg)

try:
    hand.connect()
    obs = hand.reset(sensor=True)
    print(obs["q"])
finally:
    hand.close()
```

---

### 9.3 RS485 最小示例：手动指定端口

```python
import numpy as np
from xhand import XHand, XHandConfig

cfg = XHandConfig(
    comm_type="RS485",
    serial_port="/dev/ttyUSB0",
    baudrate=3_000_000,
)

hand = XHand(cfg)

try:
    hand.connect()
    obs = hand.get_observation()
    print(obs["q"])
finally:
    hand.close()
```

更稳定的写法是使用 `/dev/serial/by-id/...`：

```python
cfg = XHandConfig(
    comm_type="RS485",
    serial_port="/dev/serial/by-id/usb-XXX_USB_Serial-if00-port0",
    baudrate=3_000_000,
)
```

---

### 9.4 83Hz 控制循环示例

XHAND1 整手控制频率约为 83Hz。建议在上层控制循环中控制频率，而不是在底层 wrapper 里强行 sleep。

```python
import time
import numpy as np
from xhand import XHand, XHandConfig

cfg = XHandConfig(comm_type="EtherCAT")
hand = XHand(cfg)

control_dt = 1.0 / 83.0

try:
    hand.connect()
    hand.reset(sensor=True)

    while True:
        t0 = time.monotonic()

        obs = hand.get_observation()
        q = obs["q"].copy()

        # Example: keep current position.
        hand.send_action(q)

        elapsed = time.monotonic() - t0
        sleep_time = control_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    hand.close()
```

---

### 9.5 读取完整触觉阵列

默认 obs 已包含：

```python
obs["tactile_raw"]
```

shape 为：

```python
(5, 120, 3)
```

含义：

```text
5     -> 五个指尖：拇指、食指、中指、无名指、小指
120   -> 每个指尖最多 120 个触觉阵列点
3     -> 每个点的 fx, fy, fz
```

示例：

```python
obs = hand.get_observation()

thumb_raw = obs["tactile_raw"][0]  # (120, 3)
index_raw = obs["tactile_raw"][1]  # (120, 3)
```

如果需要知道实际读到了多少个点：

```python
obs = hand.get_observation(full=True)
print(obs["tactile_raw_count"])
```

---

## 10. 注意事项

### 10.1 一定要 close

程序退出前应调用：

```python
hand.close()
```

推荐始终使用：

```python
try:
    ...
finally:
    hand.close()
```

### 10.2 不要超过整手 83Hz 长时间下发

不要写：

```python
while True:
    hand.send_action(q)
```

应在上层循环中控制周期，建议约：

```python
1.0 / 83.0
```

### 10.3 `send_action(q)` 只接受 12 维向量

正确：

```python
q = np.zeros(12, dtype=np.float32)
hand.send_action(q)
```

错误：

```python
hand.send_action(np.zeros((12, 1)))
hand.send_action(np.zeros((1, 12)))
hand.send_action(np.zeros(6))
```

### 10.4 默认会 clip 到 `q_min/q_max`

如果你发送的 `q` 超出限位，wrapper 会自动裁剪。返回值 `q_sent` 是实际写入 command buffer 的目标角：

```python
q_sent = hand.send_action(q)
```

### 10.5 当前 wrapper 是位置控制为主

当前 `send_action(q)` 是位置控制接口，不是电流控制接口。

虽然 SDK 文档中有 `mode=5` 力控模式，但当前 wrapper 没有封装 `send_current(current)`，因为现有资料尚未明确 Python SDK 中哪个字段承载目标关节电流。

### 10.6 `current` 不要当作外力

obs 中：

```python
obs["current"]
```

来自 SDK 的 `fs.torque` 字段，但在本 wrapper 中按关节电流 / 力矩相关反馈理解。它适合用于观察关节负载、卡滞、接触增强等，不建议直接当成准确外力或 SI 制关节力矩。

### 10.7 `sensor_data` 使用正常拼写

当前 wrapper 使用：

```python
state.sensor_data
```

不使用文档 typo 中的：

```python
state.senser_data
```

### 10.8 触觉数值已经除以 10

wrapper 中：

```python
tactile_force
tactile_raw
```

都已经对 SDK 原始值做了：

```python
value / 10.0
```

不要在下游重复除以 10。

### 10.9 `dq` 可能是差分速度

如果 SDK 原始 `FingerState_t` 有 `velocity` 字段，wrapper 会优先使用它。否则：

```python
dq = (q - last_q) / dt
```

这意味着第一帧 `dq` 通常为 0，且差分速度会受到读取频率和状态刷新延迟影响。

### 10.10 RS485 下 `force_update`

`force_update=True` 会主动请求刷新状态，可能有阻塞时间。当前配置默认：

```python
force_update=False
```

通常只在 RS485 且你明确需要强制刷新时开启。

---

## 11. 常见问题

### Q1：怎么知道我应该用 EtherCAT 还是 RS485？

如果你用 RJ45 连接 XH04 调试线到电脑有线网口，一般是 EtherCAT：

```python
XHandConfig(comm_type="EtherCAT")
```

如果你用 USB 连接 XH04 调试线到电脑，一般是 RS485：

```python
XHandConfig(comm_type="RS485")
```

### Q2：RS485 的 `serial_port` 不知道填什么怎么办？

先自动枚举：

```python
XHandConfig(comm_type="RS485", serial_port=None)
```

如果失败，再用：

```bash
ls -l /dev/serial/by-id/
dmesg -w
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
```

定位真实端口。

### Q3：为什么串口打开失败？

常见原因：

- 设备没供电；
- USB 没插好；
- 端口名错；
- 当前用户没有串口权限；
- 端口被 XOS 或其他程序占用；
- baudrate 没用 `3_000_000`。

### Q4：为什么 `tactile_force` 松手后还有值？

可能是触觉零点漂移或残留偏置。可以执行：

```python
hand.reset_sensor()
```

或者：

```python
hand.reset(sensor=True)
```

### Q5：能不能直接控制关节电流？

当前 wrapper 没有封装直接目标电流接口。现有 SDK 主要通过 `HandCommand_t / FingerCommand_t` 和 `send_command()` 下发控制命令。`mode=5` 是力控/电流相关模式，但在没有确认 `position / tor_max` 等字段在 `mode=5` 下具体语义前，不建议写 `send_current()`。

---

## 12. 推荐最小使用模板

```python
import time
import numpy as np
from xhand import XHand, XHandConfig

cfg = XHandConfig(
    comm_type="EtherCAT",  # or "RS485"
    ifname=None,
    serial_port=None,
    baudrate=3_000_000,
)

hand = XHand(cfg)

try:
    hand.connect()
    print(hand.get_meta_info())

    obs = hand.reset(sensor=True)

    control_dt = 1.0 / 83.0
    for _ in range(100):
        t0 = time.monotonic()

        obs = hand.get_observation()
        q = obs["q"].copy()

        # Replace this with your policy / teleop command.
        q_cmd = q

        hand.send_action(q_cmd)

        sleep_time = control_dt - (time.monotonic() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    hand.close()
```
