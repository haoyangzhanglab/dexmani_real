# xArm7 Controller Wrapper 使用说明

本文档说明 `xarm7_api.py` 中 `Xarm7Controller` 的安装、接口定义、物理量单位、典型使用方式和注意事项。

该 wrapper 面向 xArm7 机器人模仿学习 / 遥操作 / 路径执行项目，提供两条清晰控制通道：

```text
1. ServoJ 高频控制
   send_action(q)
   -> mode 1 + set_servo_angle_j
   -> 用于 policy rollout / teleop / 上层 planner 高频发点

2. 传统关节角控制
   move_joint(q)
   execute_joint_path(path)
   -> mode 0 + set_servo_angle
   -> 用于 episode reset / scripted path / 稀疏 joint path 加密后执行
```

---

## 1. 安装

### 1.1 Python 环境

推荐使用 Python 3.10+。`xarm-python-sdk` 官方文档说明该 SDK 仅支持 Python 3，并推荐 Python 3.10 或以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 1.2 安装依赖

```bash
pip install xarm-python-sdk numpy
```

推荐固定 SDK 版本为 `xarm-python-sdk >= 1.16.0`，因为当前代码使用：

```python
self.arm.get_joint_states(is_radian=True, num=3)
```

较旧版本 SDK 可能不支持 `num=3` 参数。

检查版本：

```bash
pip show xarm-python-sdk
```

### 1.3 文件放置

将 `xarm7_api.py` 放入你的项目目录，例如：

```text
your_project/
  robot/
    xarm7_api.py
  scripts/
    run_policy.py
```

然后在代码中导入：

```python
from robot.xarm7_api import Xarm7Controller
```

---

## 2. 坐标、单位和 shape 约定

### 2.1 关节顺序

所有 7 维关节向量均使用 xArm7 关节顺序：

```text
[J1, J2, J3, J4, J5, J6, J7]
```

### 2.2 单位约定

| 变量 / 参数 | Shape | 单位 | 说明 |
|---|---:|---|---|
| `q` | `(7,)` | rad | 关节角 |
| `dq` | `(7,)` | rad/s | 关节角速度；fallback 时为 zero placeholder |
| `effort` | `(7,)` 或 `None` | SDK effort 输出 | 不强行假设为 N·m |
| `path` | `(T, 7)` | rad | 稀疏或加密后的关节路径 |
| `speed` | scalar | rad/s | `set_servo_angle` 的传统关节运动速度 |
| `mvacc` | scalar | rad/s² | `set_servo_angle` 的传统关节运动加速度 |
| `max_joint_delta` | scalar 或 `(7,)` | rad/frame | ServoJ `send_action()` 相邻两帧最大关节变化 |
| `max_segment_delta` | scalar 或 `(7,)` | rad/segment | `execute_joint_path()` densify 后相邻路径点最大关节变化 |
| `timestamp` | scalar | seconds | `time.time()` UNIX 时间戳 |

### 2.3 xArm7 关节限位

代码中定义：

```python
XARM7_Q_MIN = deg2rad([-360, -117, -360, -6, -360, -97, -360])
XARM7_Q_MAX = deg2rad([ 360,  116,  360, 225,  360, 180,  360])
```

`move_joint()` / `execute_joint_path()` 默认会严格检查 joint limit；`send_action()` 默认会 clip 到 joint limit，若 `strict_limit=True` 则越界直接报错。

---

## 3. 运动模式说明

xArm 控制器模式和当前 wrapper 的对应关系：

| Wrapper 接口 | xArm mode | SDK API | 用途 |
|---|---:|---|---|
| `send_action(q)` | `1` ServoJ mode | `set_servo_angle_j` | 高频 policy / teleop / planner streaming |
| `move_joint(q)` | `0` position mode | `set_servo_angle` | 单个传统关节角目标 |
| `execute_joint_path(path)` | `0` position mode | `set_servo_angle` | 稀疏 path 自动加密后逐点执行 |

官方说明中，Mode 0 是位置控制模式，关节运动示例为 `set_servo_angle`；Mode 1 是 ServoJ 模式，对应 `set_servo_angle_j`，该命令无 buffer，只执行最新目标点，且需要用户自己保证轨迹点平滑。

---

## 4. 模块级工具函数

### `first_bad_code(*codes) -> int`

返回第一个非 0 API code；如果所有 code 都是 `0` 或 `None`，返回 `0`。

用途：合并多次 SDK 调用的返回码。

---

### `to_xarm7_q(q) -> np.ndarray`

将输入转换为 7 维 `float64` 关节角。

| 参数 | 单位 | Shape | 说明 |
|---|---|---:|---|
| `q` | rad | `(7,)` | 关节角 |

检查：

- 必须正好 7 维；
- 不允许 NaN / Inf。

---

### `to_xarm7_delta(delta, name) -> np.ndarray`

将 scalar 或 7 维数组转换成 7 维正数 delta。

| 参数 | 单位 | Shape | 说明 |
|---|---|---:|---|
| `delta` | rad | scalar 或 `(7,)` | 每关节限制值 |
| `name` | - | str | 报错名称 |

检查：

- scalar 会扩展为 7 维；
- 7 维数组直接使用；
- 所有值必须 finite 且大于 0。

---

### `to_xarm7_vec(x, name) -> np.ndarray`

将 SDK 返回的关节数组转成前 7 维。

| 参数 | 单位 | Shape | 说明 |
|---|---|---:|---|
| `x` | 依数据语义而定 | `>=7` | SDK 返回数组 |

如果长度小于 7，抛出 `RuntimeError`。

---

### `to_xarm7_path(path) -> np.ndarray`

检查并返回关节路径。

| 参数 | 单位 | Shape | 说明 |
|---|---|---:|---|
| `path` | rad | `(T, 7)` | 关节路径 |

检查：

- 必须是二维数组；
- 第二维必须是 7；
- `T >= 1`；
- 不允许 NaN / Inf。

---

### `check_joint_limits(q, name) -> None`

检查 `q` 是否在 xArm7 关节限位内。

| 参数 | 单位 | Shape | 说明 |
|---|---|---:|---|
| `q` | rad | `(7,)` 或 `(T, 7)` | 关节角或路径 |
| `name` | - | str | 报错名称 |

超出限位时抛出 `ValueError`。

---

### `densify_joint_path(path, max_segment_delta=None) -> np.ndarray`

将稀疏关节路径做 joint-space linear interpolation，使相邻路径点足够密。

| 参数 | 单位 | Shape | 默认值 | 说明 |
|---|---|---:|---|---|
| `path` | rad | `(T, 7)` | 必填 | 稀疏 joint path |
| `max_segment_delta` | rad/segment | scalar 或 `(7,)` | `XARM7_DEFAULT_PATH_SEGMENT_DELTA` | densify 后相邻 waypoint 的最大每关节变化 |

默认：

```python
XARM7_DEFAULT_PATH_SEGMENT_DELTA = deg2rad([3, 3, 3, 3, 5, 5, 8])
```

性质：

```text
abs(dense_path[i + 1] - dense_path[i]) <= max_segment_delta
```

注意：

- 使用 joint-space 线性插值；
- 不做 spline；
- 不做环境碰撞检查；
- 重复 waypoint 会自动跳过。

---

## 5. `Xarm7Controller` 初始化

```python
ctrl = Xarm7Controller(
    ip="192.168.1.xxx",
    use_self_collision_detection=True,
    max_joint_delta=None,
)
```

| 参数 | 类型 | 单位 | 默认值 | 说明 |
|---|---|---|---|---|
| `ip` | str | - | 必填 | xArm 控制器 IP |
| `use_self_collision_detection` | bool | - | `True` | 是否开启自碰撞检测 |
| `max_joint_delta` | scalar 或 `(7,)` | rad/frame | `XARM7_DEFAULT_MAX_JOINT_DELTA` | `send_action()` 相邻两帧最大关节变化 |

默认：

```python
XARM7_DEFAULT_MAX_JOINT_DELTA = deg2rad([1, 1, 1.5, 1.5, 2, 2, 2.5])
```

---

## 6. 生命周期接口

### `connect() -> bool`

连接机械臂并初始化到 ServoJ ready 状态。

内部流程：

```text
XArmAPI(ip, do_not_open=True)
connect()
clean_error()
clean_warn()
motion_enable(True)
set_mode(1)
set_state(0)
apply_safety_settings()
```

返回：

| 返回值 | 说明 |
|---|---|
| `True` | 连接成功且进入 `servo_ready` |
| `False` | 连接或初始化失败 |

---

### `disconnect() -> bool`

断开连接，并清空 action limiter 状态。

---

## 7. 状态、错误与安全接口

### `set_mode(mode: int) -> bool`

设置 xArm mode，并调用 `set_state(0)` 让机械臂进入对应模式的 standby / ready 流程。

| 参数 | 说明 |
|---|---|
| `MODE_POSITION = 0` | 传统位置控制模式 |
| `MODE_SERVO = 1` | ServoJ 模式 |

成功后会等待 `STATE_SETTLE_TIME = 0.1s`。

---

### `apply_safety_settings() -> bool`

应用自碰撞检测配置：

```python
self.arm.set_self_collision_detection(1 or 0)
```

然后调用 `set_state(0)` 并短暂等待。

---

### `clear_error() -> bool`

清除 error/warn，重新使能机械臂，并进入 ServoJ ready 状态。

内部流程：

```text
clean_error()
clean_warn()
motion_enable(True)
set_mode(MODE_SERVO)
set_state(0)
reset_action_limiter()
```

---

### `reset() -> bool`

当前等价于：

```python
return self.clear_error()
```

注意：`reset()` 不会移动机械臂，不会回 home。

---

### `return_home(*args, **kwargs)`

当前未实现：

```python
raise NotImplementedError("return_home() is not implemented yet")
```

设计意图：未来可 wrap `execute_joint_path()`，从当前 `q` 移动到用户定义的安全 home pose。

---

### `stop(decelerate: bool = False) -> bool`

停止机械臂。

| 参数 | 说明 |
|---|---|
| `decelerate=False` | 默认发送 `STATE_STOP = 4` |
| `decelerate=True` | 仅当当前 mode 是 `MODE_POSITION` 时发送 `STATE_DECELERATION_STOP = 6` |

注意：state 6 是 Mode 0 下的减速停止语义；ServoJ 通道默认使用普通 stop。

---

## 8. 状态查询接口

### `get_error_status() -> dict`

返回机械臂当前状态。

字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `mode` | int | 当前 mode；无法读取时为 `-1` |
| `state` | int | 当前反馈 state；无法读取时为 `-1` |
| `err` | int | controller error code；无法读取时为 `-1` |
| `warn` | int | controller warning code；无法读取时为 `-1` |
| `connected` | bool | 是否连接 |
| `ready` | bool | 无 API 错误、无 err/warn，且 state 在 `(1, 2)` |
| `servo_ready` | bool | `ready and mode == MODE_SERVO` |
| `position_ready` | bool | `ready and mode == MODE_POSITION` |
| `api_code` | int 或 None | 最近一次 API code |

---

### `ensure_servo_ready() -> dict`

确保机械臂处于 ServoJ ready 状态；否则尝试切换到 `MODE_SERVO`。

失败时抛出 `RuntimeError`。

---

### `ensure_position_ready() -> dict`

确保机械臂处于传统 position ready 状态；否则尝试切换到 `MODE_POSITION`。

失败时抛出 `RuntimeError`。

---

## 9. 观测接口

### `read_joint_state()`

读取底层关节状态。

返回：

```python
q, dq, effort, has_joint_velocity, has_effort
```

| 返回值 | Shape | 单位 | 说明 |
|---|---:|---|---|
| `q` | `(7,)` | rad | 关节角 |
| `dq` | `(7,)` | rad/s | 关节速度；fallback 时为全 0 |
| `effort` | `(7,)` 或 `None` | SDK effort 输出 | `get_joint_states` 成功时返回 |
| `has_joint_velocity` | bool | - | `dq` 是否来自 SDK velocity |
| `has_effort` | bool | - | `effort` 是否可用 |

读取逻辑：

```text
优先 get_joint_states(is_radian=True, num=3)
失败后 fallback 到 get_servo_angle(is_radian=True)
```

fallback 时 `dq` 是全 0 placeholder，不代表机械臂真实静止。

---

### `get_observation() -> dict`

面向 policy / logger 的常用观测接口。

返回字段：

| 字段 | Shape | 单位 | 说明 |
|---|---:|---|---|
| `q` | `(7,)` | rad | 关节角 |
| `dq` | `(7,)` | rad/s | 关节速度或 zero placeholder |
| `effort` | `(7,)` 或 `None` | SDK effort 输出 | 不保证单位为 N·m |
| `has_joint_velocity` | bool | - | `dq` 是否真实可用 |
| `has_effort` | bool | - | `effort` 是否真实可用 |
| `timestamp` | scalar | seconds | `time.time()` |

---

### `get_observation_full() -> dict`

在 `get_observation()` 基础上合并 `get_error_status()`。

适合低频日志、episode 结束记录、debug。

---

## 10. ServoJ 高频动作接口

### `send_action(q, strict_limit=False) -> np.ndarray`

高频发送 policy / planner 输出的目标关节角。

| 参数 | Shape | 单位 | 默认值 | 说明 |
|---|---:|---|---|---|
| `q` | `(7,)` | rad | 必填 | 目标关节角 |
| `strict_limit` | bool | - | `False` | 是否在越界时直接报错 |

内部流程：

```text
ensure_servo_ready()
validate q
joint limit check / clip
first-frame anchor to current q
per-frame delta clamp
set_servo_angle_j(angles=q_sent, is_radian=True)
```

返回：

| 返回值 | Shape | 单位 | 说明 |
|---|---:|---|---|
| `q_sent` | `(7,)` | rad | 实际发送给机械臂的关节目标 |

相关状态：

```python
ctrl.last_action_limited
```

如果 `True`，说明本帧目标被 `max_joint_delta` 限制过。

### `max_joint_delta` 语义

```text
q_sent_t = clip(q_target_t, q_sent_{t-1} - max_joint_delta, q_sent_{t-1} + max_joint_delta)
```

单位是 `rad/frame`，不是 `rad/s`。

---

## 11. 传统关节角控制接口

### `move_joint(q, speed=..., mvacc=..., wait=True, radius=-1.0, strict_limit=True) -> np.ndarray`

移动到单个关节角目标，使用 Mode 0 + `set_servo_angle()`。

| 参数 | Shape | 单位 | 默认值 | 说明 |
|---|---:|---|---|---|
| `q` | `(7,)` | rad | 必填 | 目标关节角 |
| `speed` | scalar | rad/s | `deg2rad(20)` | 关节运动速度 |
| `mvacc` | scalar | rad/s² | `deg2rad(100)` | 关节运动加速度 |
| `wait` | bool | - | `True` | 是否等待运动完成 |
| `radius` | scalar | SDK radius 语义 | `-1.0` | `<0` 表示普通 MoveJoint；`>=0` 表示 blending |
| `strict_limit` | bool | - | `True` | 越界是否直接报错 |

返回：

| 返回值 | Shape | 单位 | 说明 |
|---|---:|---|---|
| `q` | `(7,)` | rad | 最终目标关节角 |

注意：执行完成后会调用 `reset_action_limiter(q)`，方便后续切回 `send_action()`。

---

### `execute_joint_path(path, speed=..., mvacc=..., wait_each=True, radius=-1.0, max_segment_delta=None, densify=True, stop_on_error=True) -> dict`

执行关节路径，使用 Mode 0 + `set_servo_angle()`。

| 参数 | Shape | 单位 | 默认值 | 说明 |
|---|---:|---|---|---|
| `path` | `(T, 7)` | rad | 必填 | 输入关节路径，可以比较稀疏 |
| `speed` | scalar | rad/s | `deg2rad(20)` | 关节运动速度 |
| `mvacc` | scalar | rad/s² | `deg2rad(100)` | 关节运动加速度 |
| `wait_each` | bool | - | `True` | 是否每个 waypoint 等待完成 |
| `radius` | scalar | SDK radius 语义 | `-1.0` | 默认不做 blending |
| `max_segment_delta` | scalar 或 `(7,)` | rad/segment | `XARM7_DEFAULT_PATH_SEGMENT_DELTA` | densify 后相邻 waypoint 最大关节差 |
| `densify` | bool | - | `True` | 是否自动加密路径 |
| `stop_on_error` | bool | - | `True` | 中途失败是否调用 `stop()` |

返回 dict：

| 字段 | 类型 | 说明 |
|---|---|---|
| `ok` | bool | 是否完整执行到最后一个 waypoint |
| `num_input_points` | int | 原始输入 path 点数 |
| `num_executed_points` | int | 当前代码中表示准备执行的 dense path 点数 |
| `last_index` | int | 最后一次发给 SDK 的 dense waypoint index；未开始时为 `-1` |
| `last_code` | int | 最后一次 SDK API code |
| `status` | dict | 结束时 `get_error_status()` |

注意：当前版本里 `num_executed_points` 实际等于 dense path 总点数，不等于失败时真实成功执行的点数。判断失败位置请看 `last_index` 和 `ok`。

---

## 12. 最小使用示例

### 12.1 连接、读取观测、发送一次当前位置保持命令

```python
from xarm7_api import Xarm7Controller

ctrl = Xarm7Controller("192.168.1.xxx")

if not ctrl.connect():
    raise RuntimeError(ctrl.get_error_status())

try:
    obs = ctrl.get_observation()
    print("q:", obs["q"])
    print("dq:", obs["dq"])

    q_sent = ctrl.send_action(obs["q"], strict_limit=True)
    print("q_sent:", q_sent)

finally:
    ctrl.stop()
    ctrl.disconnect()
```

---

### 12.2 模仿学习 policy loop

```python
ctrl = Xarm7Controller(
    ip="192.168.1.xxx",
    max_joint_delta=np.deg2rad([1, 1, 1.5, 1.5, 2, 2, 2.5]),
)

if not ctrl.connect():
    raise RuntimeError(ctrl.get_error_status())

try:
    for step in range(num_steps):
        obs = ctrl.get_observation()

        # policy 输出应先在上层转换成 absolute q_target, unit: rad, shape: (7,)
        q_target = policy(obs)

        q_sent = ctrl.send_action(q_target, strict_limit=False)

        logger.write({
            "q": obs["q"],
            "dq": obs["dq"],
            "has_joint_velocity": obs["has_joint_velocity"],
            "q_target": q_target,
            "q_sent": q_sent,
            "action_limited": ctrl.last_action_limited,
        })

        if step % 10 == 0:
            status = ctrl.get_error_status()
            if not status["ready"]:
                break

finally:
    ctrl.stop()
    ctrl.disconnect()
```

---

### 12.3 执行传统关节路径

```python
path = np.deg2rad(np.array([
    [0, -30, 0, 45, 0, 60, 0],
    [5, -28, 0, 50, 0, 62, 0],
    [10, -25, 0, 55, 0, 65, 0],
], dtype=np.float64))

result = ctrl.execute_joint_path(
    path,
    speed=np.deg2rad(20.0),
    mvacc=np.deg2rad(100.0),
    wait_each=True,
    radius=-1.0,
    max_segment_delta=np.deg2rad([3, 3, 3, 3, 5, 5, 8]),
    densify=True,
)

print(result)
```

---

### 12.4 单点传统关节角移动

```python
q_start = np.deg2rad([0, -30, 0, 45, 0, 60, 0])
ctrl.move_joint(q_start, speed=np.deg2rad(20), mvacc=np.deg2rad(100), wait=True)
```

---

## 13. 注意事项

### 13.1 不要混淆 `send_action()` 和 `execute_joint_path()`

```text
send_action():
  mode 1 / ServoJ / 高频逐帧控制

execute_joint_path():
  mode 0 / 传统关节角控制 / 路径点执行
```

不要在同一个高频 loop 里反复切换 mode 0 和 mode 1。

---

### 13.2 ServoJ 不依赖 speed / mvacc

`set_servo_angle_j` 在 ServoJ mode 下没有 buffer，只执行最新目标点。SDK 保留的 speed / acceleration / time 参数目前不会生效。轨迹平滑性应由上层 planner / policy loop 保证，wrapper 只做最后的 per-frame delta guard。

---

### 13.3 `dq=zeros` 不一定代表机械臂静止

如果 `get_joint_states()` 失败，代码会 fallback 到 `get_servo_angle()`。此时：

```python
dq = np.zeros(7)
has_joint_velocity = False
```

训练或记录 demo 时，应同时保存 `has_joint_velocity`。

---

### 13.4 `densify_joint_path()` 不做环境碰撞检测

它只保证 joint-space waypoint 足够密，不保证 TCP、夹爪、相机、桌面等环境碰撞安全。

环境碰撞检查应由上层 planner 完成。

---

### 13.5 `return_home()` 当前不可用

当前版本：

```python
ctrl.return_home()
# NotImplementedError
```

不要在实验脚本中依赖它。

---

### 13.6 发生 error / warn 后不要盲目继续原路径

如果 `execute_joint_path()` 失败，应记录：

```text
last_index
last_code
status["err"]
status["warn"]
```

然后重新规划或人工检查。不要简单 `clear_error()` 后继续执行剩余路径。

---

## 14. 推荐日志字段

模仿学习 rollout / demo collection 时建议记录：

```text
obs/q
obs/dq
obs/has_joint_velocity
obs/has_effort
action/q_target
action/q_sent
action/limited
robot/mode
robot/state
robot/err
robot/warn
robot/api_code
time/timestamp
```

这样可以区分：

```text
policy 原始输出
wrapper 实际发送动作
是否被 joint limit 或 max_joint_delta 改写
机器人当时是否处于 warning/error 状态
```

---

## 15. 官方参考

- [xArm Python SDK Installation - UFACTORY Docs](https://docs.api.ufactory.cc/xarm_python_sdk_docs/1.%20xArm-Python-SDK%20Installation.html)
- [xArm Python SDK GitHub README](https://github.com/xArm-Developer/xArm-Python-SDK)
- [xArm Python SDK PyPI package](https://pypi.org/project/xarm-python-sdk/)
- [Robot state and mode explanation - UFACTORY Docs](https://docs.supportarticle.ufactory.cc/support_articles/developer/robot-state-and-mode-explanation.html)
- [ServoJ guide: set_servo_cartesian and set_servo_angle_j](https://help.ufactory.cc/en/articles/3973629-guide-to-use-the-interface-set_servo_cartesian-and-set_servo_anagle_j)
- [xArm ROS wrapper documentation](https://github.com/xArm-Developer/xarm_ros)
