# xArm7 在 `lerobot_robot_ufactory` 项目中的机制、实现与 SDK 调用技术文档

> 文档性质：代码审计与架构说明  
> 审计对象：仓库提交 `e263288`（`main`）  
> 审计日期：2026-08-15  
> 适用范围：本仓库中的 xArm7 单臂/多臂接入、遥操作、数据录制和策略推理路径

## 阅读导航

本文按“先建立边界，再追踪运行链路，最后处理风险与落地”的顺序组织：

- **第 1～5 章：范围与静态结构**——说明证据边界、总体架构、文件地图、LeRobot 注册方式、配置和单位；
- **第 6～12 章：机器人核心机制**——说明生命周期、数据模型、SDK 调用、Mode 6/7、TCP 30000 和夹爪；
- **第 13～17 章：上层业务链路**——说明 GELLO、SpaceMouse、Pika、UMI、teleop、record、eval 和多臂聚合；
- **第 18～21 章：能力边界与审计发现**——区分控制箱规划和主机侧能力，汇总并发、实时、缺陷和版本兼容性；
- **第 22～27 章：部署与演进**——给出安全部署、修复路线、测试矩阵、阅读顺序、官方资料和最终结论。

如果目标是直接接入真机，应至少阅读第 5～12、20～24 章；如果目标是修改策略推理，应重点阅读第 7、10、16、20、24 章。

## 1. 文档目的、证据与结论边界

本文档用于完整说明 xArm7 在本项目中的：

- 实例化与配置方式；
- 连接、初始化、运行、复位和断开生命周期；
- 关节空间与笛卡尔空间控制机制；
- xArm Python SDK 调用位置、参数和单位；
- TCP 30000 实时数据读取和解析方式；
- 官方夹爪、Pika、Robotiq 等末端执行器的处理方式；
- GELLO、SpaceMouse、Pika、UMI 和策略推理如何生成机械臂动作；
- LeRobot 数据特征、数据录制和多机械臂封装；
- 本仓库实际具备和明确不具备的规划、运动学与碰撞能力；
- 已确认缺陷、兼容性风险、安全缺口和建议改进顺序。

本文依据三类证据：

1. **仓库代码事实**：以当前工作区源代码和 YAML 配置为准。
2. **官方 SDK/协议事实**：以 UFACTORY 官方 xArm Python SDK、控制模式说明和 TCP 端口协议为准。
3. **审计判断**：由代码行为推导的后果或风险，均明确标为“风险”或“推论”，不作为已经完成真机复现的结论。

本次审计没有连接真机。当前基础 Python 环境中也未安装 `xarm-python-sdk`，因此本文不是硬件实测报告；SDK 调用语义由源码与官方文档交叉核对。

## 2. 执行摘要

### 2.1 最重要的架构结论

项目中没有独立的 `XArm7` 类。xArm5、xArm6、xArm7 共用：

- [`UFRobotConfig`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot_config.py)；
- [`UFRobot`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py)。

xArm7 是通过 `robot_dof: 7` 配置出来的。连接后，代码将配置自由度与 SDK 属性 `real_arm.axis` 比较，从而阻止把 xArm6/xArm5 错当成 xArm7。

### 2.2 两条控制路径

项目支持两条主控制路径：

| 控制空间 | 控制箱模式 | SDK 运动指令 | 动作表达 |
|---|---:|---|---|
| 关节空间 | Mode 6 | `set_servo_angle(..., wait=False)` | 7 个绝对关节角，弧度 |
| 笛卡尔空间 | Mode 7 | `set_position_aa(..., wait=False)` | 绝对 TCP 位姿，XYZ 毫米、姿态为轴角旋转向量 |

Mode 6/7 的在线轨迹规划发生在 xArm 控制箱中，不是本仓库在主机侧实现的轨迹规划。

### 2.3 状态观测路径

- 关节模式通过 SDK `get_joint_states(is_radian=True, num=3)` 获取关节位置、速度和 effort/估计力矩数据。
- 笛卡尔模式不调用 SDK 的普通 TCP 查询接口作为主观测源，而是启动独立线程连接 TCP 30000 端口，读取高频实际 TCP 位姿和速度。
- 摄像头图像通过 LeRobot Camera 的 `async_read()` 加入同一 observation 字典。

### 2.4 本仓库明确没有的能力

仓库中没有发现：

- xArm7 URDF/SRDF；
- 主机侧正运动学或逆运动学求解器；
- MoveIt 或其他轨迹规划器；
- 环境碰撞检测；
- 机器人几何模型驱动的自碰撞检测；
- 主机侧关节限位或工作空间限位层；
- 共享内存实时控制架构；
- 针对 xArm7 控制核心的自动化测试。

因此，本项目本质上是 **LeRobot 数据/策略管线与 xArm SDK 之间的适配层**，而不是完整的运动规划与安全控制系统。

## 3. 项目中的 xArm7 文件地图

### 3.1 核心机器人实现

| 文件 | 作用 |
|---|---|
| [`uf_robot.py`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py) | 真机连接、状态机、SDK 调用、夹爪、TCP 30000 数据线程 |
| [`uf_robot_config.py`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot_config.py) | 单机械臂配置结构和 LeRobot 类型注册 |
| [`multiple_uf_robot.py`](src/lerobot_robot_ufactory/robots/uf_robot/multiple_uf_robot.py) | 多机械臂聚合、前缀、异步连接/配置/动作队列 |
| [`multiple_uf_robot_config.py`](src/lerobot_robot_ufactory/robots/uf_robot/multiple_uf_robot_config.py) | 多机械臂配置结构 |
| [`robots/utils.py`](src/lerobot_robot_ufactory/robots/utils.py) | 将 `uf::robot`/`uf::multiple_robot` 分派到项目实现 |

### 3.2 xArm7 示例配置

| 文件 | 控制空间 | 遥操作器 | 夹爪 |
|---|---|---|---|
| [`xarm7_gello_record_config.yaml`](config/gello/xarm7_gello_record_config.yaml) | 关节 | GELLO | xArm Gripper |
| [`xarm7_spacemouse_record_config.yaml`](config/spacemouse/xarm7_spacemouse_record_config.yaml) | 笛卡尔 | SpaceMouse | 无 |
| [`xarm7_pika_record_config.yaml`](config/pika/xarm7_pika_record_config.yaml) | 笛卡尔 | Pika | Pika Gripper |

### 3.3 运行入口

| 命令 | 实现文件 | 用途 |
|---|---|---|
| `uf-robot-teleop` | [`uf_robot_teleop.py`](src/lerobot_robot_ufactory/scripts/uf_robot_teleop.py) | 遥操作但不录制 |
| `uf-lerobot-record` | [`uf_lerobot_record.py`](src/lerobot_robot_ufactory/scripts/uf_lerobot_record.py) | 遥操作/策略控制并录制 LeRobot 数据集 |
| `uf-lerobot-eval` | [`uf_lerobot_eval.py`](src/lerobot_robot_ufactory/scripts/uf_lerobot_eval.py) | 加载训练策略并控制真机 |

这些命令由 [`pyproject.toml`](pyproject.toml) 的 `[project.scripts]` 注册。

## 4. LeRobot 插件注册与对象创建

### 4.1 类型注册

`UFRobotConfig` 使用：

```python
@RobotConfig.register_subclass("uf::robot")
```

将 YAML 中的：

```yaml
robot:
  type: uf::robot
```

映射为 `UFRobotConfig`。

多机械臂对应 `uf::multiple_robot`。

### 4.2 工厂补丁

项目根包 [`src/lerobot_robot_ufactory/__init__.py`](src/lerobot_robot_ufactory/__init__.py) 会修改 LeRobot 已导入的工厂函数：

- `lerobot.robots.make_robot_from_config`；
- `lerobot.teleoperators.make_teleoperator_from_config`；
- `lerobot.cameras.make_cameras_from_configs`；
- 对应 `utils` 模块中的同名函数。

三个主脚本都在使用工厂前执行：

```python
import lerobot_robot_ufactory  # patch
```

因此，插件能否正确生效依赖导入顺序。这是全局 monkey patch，而不是完全隔离的插件实例。

### 4.3 创建 xArm7

`UFRobot.__init__()` 读取 `config.robot_dof`，只允许 5、6、7：

```python
self._dof = config.robot_dof
if self._dof is None or self._dof not in (5, 6, 7):
    raise ValueError(...)
```

xArm7 的身份来自 `robot_dof: 7`，不存在额外的型号专属控制器类。

## 5. 配置参数与单位

### 5.1 `UFRobotConfig` 参数

| 参数 | 默认值 | 单位/语义 | 生效位置 |
|---|---|---|---|
| `robot_ip` | `192.168.1.127` | 控制箱 IPv4 地址 | `XArmAPI`、TCP 30000 |
| `robot_dof` | `None` | 必须为 5/6/7；xArm7 为 7 | 构造与轴数校验 |
| `control_space` | `joint` | `joint` 或 `cartesian` | feature、模式与指令选择 |
| `gripper_type` | `1` | 末端执行器枚举值 | 初始化、观测和动作 |
| `gripper_port` | `None` | Pika 夹爪串口 | 仅 Pika Gripper |
| `gripper_speed` | `-1` | `-1` 表示选用各夹爪默认值 | 夹爪初始化/Modbus |
| `gripper_force` | `-1` | `-1` 表示选用默认值 | 支持力参数的夹爪 |
| `observe_joint_vel` | `False` | 是否把关节速度加入 observation | 仅关节模式 |
| `start_joints` | 7 维默认姿态 | 度 | 构造时转弧度 |
| `start_tcp_pose` | `None` | `[mm, mm, mm, deg, deg, deg]`，姿态为 RPY | 初始化 Mode 0 运动 |
| `max_joint_velocity` | `90` | 度/秒 | 构造时转弧度/秒 |
| `max_linear_velocity` | `200` | 毫米/秒 | `set_position_aa` |
| `no_action` | `False` | 调试模式，跳过真机动作 | `send_action()` |

### 5.2 初始化姿态转换

构造时：

- `start_joints` 的所有值由度转换为弧度；
- `start_tcp_pose` 的 XYZ 保持毫米，后三维 RPY 从度转换为弧度；
- `max_joint_velocity` 从度/秒转换为弧度/秒。

要注意两种姿态表达并不相同：

- `start_tcp_pose` 由 `set_position()` 接收，语义为 RPY；
- 日常笛卡尔动作由 `set_position_aa()` 接收，语义为轴角旋转向量。

### 5.3 当前 xArm7 配置中的事实

#### GELLO

- `robot_dof: 7`；
- `control_space: joint`；
- `gripper_type: 1`；
- 机器人和 GELLO 的 `start_joints` 都为 `[0, 0, 0, 90, 0, 90, 0]`；
- dataset 路径和 `repo_id` 却写成 `xarm6_gello_datas`，与文件名和机械臂配置不一致。

#### SpaceMouse

- `robot_dof: 7`；
- `control_space: cartesian`；
- 无夹爪；
- 两个 RealSense 相机；
- dataset 频率为 10 Hz；
- SpaceMouse 自身默认 `frequency` 也是 10 Hz，因此默认位移增量与录制频率相匹配。

#### Pika

- `robot_dof: 7`；
- `control_space: cartesian`；
- `gripper_type: 10`，即 Pika Gripper；
- 同时配置 `start_joints` 和 `start_tcp_pose`；
- Tracker 位移缩放 `scale_xyz: 1.5`；
- 相机和数据集频率为 30 Hz。

## 6. `UFRobot` 生命周期

### 6.1 构造阶段

`UFRobot` 同时继承 `lerobot.robots.Robot` 和 `threading.Thread`。这个线程只用于笛卡尔模式的 TCP 30000 实时报告。

构造阶段完成：

1. 检查自由度；
2. 保存控制空间；
3. 创建摄像头对象，但尚未连接；
4. 转换速度和初始姿态单位；
5. 建立实时数据锁和停止事件；
6. 根据 `gripper_type` 构造夹爪参数；
7. 根据控制空间决定是否启用 TCP 30000 线程。

### 6.2 `connect()`

连接顺序为：

```text
XArmAPI(robot_ip)
  → 等待 0.2 秒
  → 检查 SDK connected
  → 检查 real_arm.axis == robot_dof
  → 连接所有摄像头
  → configure()
  → calibrate()
  → set_linear_spd_limit_factor(2.0)
```

`XArmAPI(robot_ip)` 默认不是 `do_not_open=True`，因此 SDK 构造阶段会主动打开连接。

### 6.3 `configure()`

配置顺序：

```text
motion_enable()
  → clean_error()
  → set_mode(0)
  → set_state(0)
  → 等待 0.5 秒
  → 读取控制器错误码
  → 初始化/打开夹爪
  → 移动到 start_joints
  → 可选：移动到 start_tcp_pose
  → 切换 Mode 6 或 Mode 7
  → set_state(0)
  → 再次检查控制器错误码
  → 笛卡尔模式启动 TCP 30000 线程
```

如果配置了 `start_tcp_pose`：

1. 先执行 `start_joints`；
2. 再通过 `set_position(..., wait=True)` 移动到 TCP 姿态；
3. 调用 `get_servo_angle()` 获取该 TCP 姿态对应的实际关节角；
4. 用它覆盖 `_start_joints`；
5. 将 `_start_tcp_pose` 设为 `None`。

因此，后续 episode 再次 `configure()` 时只会回到第一次 TCP 初始化后记录的关节姿态，不再重复 TCP 初始化运动。

### 6.4 `calibrate()`

当前实现仅设置 `_is_calibrated = True`，函数体为 `pass`。项目没有实现机械臂零点、TCP、负载、手眼关系或关节偏置的自动标定。

### 6.5 `disconnect()`

断开顺序为：

```text
set_state(4)
  → set_mode(0)
  → 通知并 join TCP 30000 线程
  → SDK disconnect()
  → 断开摄像头
```

`state 4` 是停止状态，会终止执行并阻止新命令，直到重新设为可运行状态。

## 7. LeRobot observation 与 action 数据模型

### 7.1 关节空间

xArm7 的基础 observation/action 键：

```text
J1.pos
J2.pos
J3.pos
J4.pos
J5.pos
J6.pos
J7.pos
```

如果 `observe_joint_vel=True`，observation 额外包含：

```text
J1.vel ... J7.vel
```

如果存在夹爪，还包含：

```text
gripper.pos
```

### 7.2 笛卡尔空间

基础 observation/action 键：

```text
pose.x
pose.y
pose.z
pose.rx
pose.ry
pose.rz
```

单位为：

- `pose.x/y/z`：毫米；
- `pose.rx/ry/rz`：弧度制轴角旋转向量，即 `axis * angle`；
- 可选速度键的平移部分为毫米/秒，旋转部分为弧度/秒。

当前 `CARTESIAN_OBS_KEYS` 中速度键被注释，所以默认数据集不包含 TCP 速度。

### 7.3 摄像头键

每个摄像头键以其配置名加入 observation，shape 声明为：

```python
(height, width, 3)
```

多臂时，机器人状态和动作键会添加 `left.`、`right.` 等前缀。

## 8. xArm Python SDK 调用矩阵

### 8.1 机械臂通用调用

| SDK 调用 | 项目用途 | 返回值是否检查 |
|---|---|---|
| `XArmAPI(robot_ip)` | 建立 SDK 连接 | 通过 `connected` 间接检查 |
| `motion_enable()` | 使能机械臂 | 否 |
| `clean_error()` | 清除控制器错误 | 否 |
| `set_mode(0/6/7)` | 设置位置/在线规划模式 | 否 |
| `set_state(0/4)` | 进入运行或停止状态 | 否 |
| `get_err_warn_code()` | 检查控制器错误/警告 | 只检查 controller error |
| `set_servo_angle()` | 初始关节运动和 Mode 6 在线目标 | 否 |
| `get_servo_angle()` | TCP 初始化后记录关节角 | 否 |
| `set_position()` | 初始 RPY TCP 运动 | 否 |
| `set_position_aa()` | 日常轴角 TCP 目标 | 否 |
| `get_joint_states()` | 关节状态观测 | 否 |
| `set_linear_spd_limit_factor()` | 设置线速度限制因子 | 否 |
| `disconnect()` | 断开 SDK | 否 |

SDK 运动和设置接口通常返回整数状态码，读取接口通常返回 `(code, data)`。当前项目大量忽略这些状态码，这是本文后续安全分析的重要依据。

### 8.2 夹爪相关调用

| 夹爪 | 初始化调用 | 循环动作调用 | 观测调用 |
|---|---|---|---|
| xArm Gripper | `set_gripper_enable/mode/speed/position` | 原始 RS485 Modbus | `get_gripper_position` |
| xArm Gripper G2 | `set_gripper_enable/mode`、`set_gripper_g2_position` | 原始 RS485 Modbus | `get_gripper_g2_position` |
| Bio Gripper G2 | control mode、enable、open | 原始 RS485 Modbus | `get_bio_gripper_g2_position` |
| Pika Gripper | `pika_gripper.enable()` | `set_gripper_distance` | `get_gripper_distance` |
| Robotiq | reset、activate、position | 原始 RS485 Modbus | `robotiq_get_status` |

## 9. 关节空间控制详解

### 9.1 观测

每个周期调用：

```python
code, states = real_arm.get_joint_states(is_radian=True, num=3)
```

代码使用：

- `states[0]`：关节位置；
- `states[1]`：关节速度，仅在 `observe_joint_vel=True` 时加入 observation；
- `states[2]` 没有加入当前 observation。

只取前 `_dof` 项，因此 xArm7 取 7 项。

### 9.2 动作生成

从 action 字典按顺序构造：

```python
[action["J1.pos"], ..., action["J7.pos"]]
```

多臂时键变为 `left.J1.pos` 等。

### 9.3 首次同步机制

代码使用 `_cmd_cnt` 区分初始命令：

- 前 20 条命令速度为 `0.2 rad/s`；
- 第 1 条命令 `wait=True`；
- 第 1 条命令执行前切回 Mode 0；
- 第 2 条及以后切回 Mode 6，使用 `wait=False`；
- 第 21 条及以后使用 `max_joint_velocity`。

该机制主要用于降低 GELLO 当前姿态与真实机械臂之间首次同步的速度。

### 9.4 Mode 6 语义

Mode 6 是关节空间动态在线规划模式。控制箱接到新的绝对关节目标后，会中断当前在线目标并根据新目标重新规划。项目没有在主机侧生成时间参数化轨迹点序列。

官方资料给出的最低模式固件要求为 1.10.0。

## 10. 笛卡尔空间控制与姿态表示

### 10.1 轴角旋转向量

项目中的 `rx/ry/rz` 不是分别绕 X/Y/Z 的欧拉角，而是：

```text
[rx, ry, rz] = unit_rotation_axis * rotation_angle
```

向量范数是旋转角，方向是旋转轴。

项目的 [`Transformations`](src/lerobot_robot_ufactory/devices/umi/vive_tracker/transformations.py) 使用 Rodrigues 公式实现轴角与旋转矩阵互换，并单独处理：

- 接近 0 的旋转；
- 接近 π 的旋转；
- 一般旋转。

### 10.2 动作发送

每周期构造：

```python
cmd_list = [x, y, z, rx, ry, rz]
real_arm.set_position_aa(
    axis_angle_pose=cmd_list,
    speed=max_linear_velocity,
    is_radian=True,
    wait=False,
)
```

未显式设置：

- `mvacc`；
- `radius`；
- `motion_type`；
- `relative`；
- `is_tool_coord`。

因此 SDK 使用其默认值，项目发送的是基坐标系下绝对轴角 TCP 目标。

### 10.3 Mode 7 语义

Mode 7 是笛卡尔空间动态在线规划模式，规划和 IK 位于控制器内部。新目标会打断当前在线规划并重新执行。

官方资料给出的最低模式固件要求为 1.11.0。

### 10.4 姿态连续性问题

同一空间旋转可以有多种等价轴角表示，尤其在接近 ±π 时可能出现符号跳变。`uf_lerobot_eval.py` 提供：

- `continuous_rotvec()`；
- `compute_relative_axis_angle()`；
- `compute_target_axis_angle()`；
- `blend_poses()`。

这些函数意图保持旋转向量半球一致或在相对/绝对位姿之间转换。但当前 eval 控制流存在缺陷，详见第 20 章，不能据此认定相对动作功能已经可靠生效。

## 11. TCP 30000 实时数据机制

### 11.1 为什么使用 30000

笛卡尔 observation 必须依赖 `_rt_report_normal`。项目没有在每个 LeRobot 控制周期调用普通 SDK TCP 查询，而是启动独立接收线程，以更高频率持续更新共享的最近状态。

官方资料说明：

- TCP 30000 用于实时机器人数据；
- 频率通常为 250 Hz；
- 带六维力传感器时为 200 Hz；
- 官方示例页面写明基础使用要求固件 2.1.101+；
- 当前完整字段说明页对 784 字节协议标注固件 2.7.101+。

项目没有做固件版本判断，因此实际部署必须根据控制箱固件验证帧长度和字段兼容性。

### 11.2 连接与帧读取

线程执行：

```python
socket(AF_INET, SOCK_STREAM)
setblocking(True)
settimeout(1)
connect((robot_ip, 30000))
```

协议帧前 4 字节是大端 U32 帧长。项目先收满 4 字节并解析 `size`，之后持续收满一个 `size` 大小的帧。

### 11.3 项目实际解析字段

下表使用 Python 0 起始切片表示：

| 切片 | 数量 | 字段 | 单位 |
|---|---:|---|---|
| `116:144` | 7×FP32 | 实际关节位置 | rad |
| `144:172` | 7×FP32 | 实际关节速度 | rad/s |
| `424:448` | 6×FP32 | 目标 TCP 位姿 | mm、rad |
| `448:472` | 6×FP32 | 目标 TCP 速度 | mm/s、rad/s |
| `472:496` | 6×FP32 | 实际 TCP 位姿 | mm、rad |
| `496:520` | 6×FP32 | 实际 TCP 速度 | mm/s、rad/s |

这些偏移与官方当前协议的 1 起始字节编号 117、145、425、449、473、497 对应一致。

FP32 内容按小端解析，帧长字段按大端解析，也与官方协议一致。

### 11.4 线程间共享

接收线程在 `_update_lock` 下更新：

- `rt_actual_joint_pos`；
- `rt_actual_joint_speed`；
- `rt_cmd_tcp_pose`；
- `rt_cmd_tcp_vel`；
- `rt_actual_tcp_pose`；
- `rt_actual_tcp_speed`。

控制线程在相同锁下复制最近的实际 TCP 位姿和速度，因此不存在对这些 list 的明显并发半写问题。

### 11.5 当前实现未处理的情况

代码没有显式处理：

- `socket.timeout`；
- 远端关闭导致 `recv()` 返回空字节；
- 帧长小于 520；
- 帧长在连接期间变化；
- 数据时间戳；
- 数据新鲜度和最大可接受延迟；
- 自动断线重连；
- `finally` 中关闭 socket；
- 线程异常后的 `_rt_report_normal` 可靠复位。

这些属于已确认的健壮性缺口。

## 12. 夹爪机制

### 12.1 统一归一化语义

数据集和策略使用：

```text
gripper.pos = 0  → 打开
gripper.pos = 1  → 闭合
```

`GripperParam.get_grippos()` 将归一化值映射到硬件位置，并把最终硬件目标限制在 `open_pos` 与 `close_pos` 之间。

反向观测公式为：

```python
(open_pos - grippos) / (open_pos - close_pos)
```

### 12.2 各夹爪硬件范围

| 类型 | `open_pos` | `close_pos` |
|---|---:|---:|
| xArm Gripper | 800 | 0 |
| xArm Gripper G2 | 84 | 0 |
| Bio Gripper G2 | 150 | 71 |
| Pika Gripper | 100 | 0 |
| Robotiq | 0 | 255 |

### 12.3 初始化方式

`configure()` 每次都会初始化并打开夹爪。对 SDK 夹爪，代码暂时直接修改：

```python
real_arm._arm._baud_checkset
```

这是 SDK 私有内部字段。当前官方 SDK 已提供公开的 baud-checkset 设置接口；继续访问私有字段会增加升级风险。

### 12.4 高频动作方式

循环控制没有对所有夹爪调用较高层、可能等待运动的 SDK 函数，而是构造原始 Modbus 写多个寄存器请求，通过：

```python
getset_tgpio_modbus_data(modbus_datas)
```

发送。

当前官方 SDK 仍保留该名称用于兼容，但已将其标记为旧接口，并推荐 `set_rs485_data()`。

### 12.5 夹爪状态风险

已确认：

- 夹爪查询返回码未检查；
- 反向归一化没有 clamp，异常硬件读数可能超出 `[0, 1]`；
- 初始化时 `_gripper_param.gripper_norm` 被赋值为 `open_pos`，而不是归一化的 `0.0`；
- 如果后续读取返回 `None`，fallback 可能返回 800、84 等非归一化值；
- Modbus 动作返回码也未检查。

## 13. 遥操作输入机制

### 13.1 GELLO：关节空间

[`GelloTeleop`](src/lerobot_robot_ufactory/teleoperators/gello_teleop/gello_teleop.py) 使用 Dynamixel：

1. 构造阶段打开 Dynamixel 驱动；
2. 预热读取 10 次；
3. 读取当前示教臂关节角；
4. 根据 `start_joints`、`joint_signs` 计算 offset；
5. 创建 `GelloAgent`；
6. 每周期 `agent.act()` 返回当前关节角和夹爪量；
7. 映射成 `J1.pos ... J7.pos` 与 `gripper.pos`。

这里不做机械臂 IK，GELLO 与 xArm7 是关节到关节映射。

### 13.2 SpaceMouse：笛卡尔增量

[`SpaceMouseTeleop`](src/lerobot_robot_ufactory/teleoperators/space_mouse/space_mouse.py)：

1. 后台线程读取 spnav event；
2. 将设备坐标变换到项目定义的右手坐标系；
3. 归一化并应用 deadzone；
4. 根据 `max_pos_speed / frequency` 转换为单周期位移；
5. 输出 `pose.dx/dy/dz`。

当前代码强制：

```python
dpos[2] = 0
```

且没有输出旋转，所以默认只能控制 XY 平面平移。

`uf_lerobot_record.record_loop()` 发现 `pose.dx` 后，会把增量累计到上一次绝对命令，再交给 `UFRobot.send_action()`。

需要注意：独立的 `uf-robot-teleop` 循环没有这段 `pose.dx` 累计逻辑。因此当前 SpaceMouse 增量动作主要与 record 路径匹配，直接 teleop 路径存在 action schema 不匹配风险。

### 13.3 Pika：Tracker 到机器人基坐标

[`PikaTeleop`](src/lerobot_robot_ufactory/teleoperators/pika_teleop/pika_teleop.py) 使用：

- Tracker 四元数位姿；
- `tracker_to_robot_eef` 外参；
- 当前机械臂 TCP 作为启用遥操作时的基准；
- Tracker 起始位姿与当前位姿的相对变换。

核心关系为：

```text
tracker_robot = tracker_pose × tracker_to_eef
delta = inverse(begin_tracker_robot) × current_tracker_robot
robot_target = robot_base × delta
```

最终将 `robot_target` 转换为 XYZ + 轴角，直接适配 `set_position_aa()`。

Pika 夹爪距离 100 映射为打开，0 映射为闭合。

### 13.4 UMI：SLAM/Vive 到机器人基坐标

[`UmiTeleop`](src/lerobot_robot_ufactory/teleoperators/umi_teleop/umi_teleop.py) 与 Pika 使用相同坐标变换抽象，姿态源可以是：

- Vive Tracker；
- XVSDK SLAM。

平移从米乘 1000 转为毫米。输出同样为绝对 XYZ + 轴角。UMI 支持通过 `MultipleUmiTeleop` 聚合多台遥操作器。

## 14. 遥操作测试链路

[`uf_robot_teleop.py`](src/lerobot_robot_ufactory/scripts/uf_robot_teleop.py) 的主路径：

```text
解析配置
  → 创建 teleop
  → 创建 robot
  → 创建默认 processors
  → robot.connect()
  → teleop.connect()
  → 循环：
       robot.get_observation()
       teleop.get_action()
       teleop_action_processor
       robot_action_processor
       robot.send_action()
  → robot.disconnect()
  → teleop.disconnect()
```

键盘控制：

- `Esc`：退出；
- 左方向键：暂停并标记 reset；
- 空格：开始/暂停；
- reset 后恢复时调用 `robot.configure()`；
- 对 `UFBaseTeleop`，启动时把当前机器人 observation 传给 teleop，建立相对基准。

该脚本在 `TeleopConfig.__post_init__()` 中清空机器人摄像头配置，因此遥操作测试默认不连接和读取摄像头。

## 15. 数据录制链路

### 15.1 数据特征创建

[`uf_lerobot_record.py`](src/lerobot_robot_ufactory/scripts/uf_lerobot_record.py) 由：

- `robot.action_features`；
- `robot.observation_features`；
- 默认 processor 的 feature 变换；

构造 LeRobotDataset features。

### 15.2 周期逻辑

每个录制周期：

```text
robot.get_observation()
  → robot_observation_processor
  → build observation_frame
  → teleop.get_action() 或 predict_action()
  → teleop/policy processor
  → robot_action_processor
  → robot.send_action()
  → build action_frame
  → dataset.add_frame()
  → 精确睡眠到目标 fps
```

### 15.3 SpaceMouse 特殊处理

如果 teleop action 包含 `pose.dx`，record 循环会把其累计为绝对 `pose.x`，并继承其他姿态/夹爪值。这是当前 SpaceMouse 能驱动绝对 Mode 7 命令的关键适配。

### 15.4 episode 管理

每次新 episode 前，对 `UFBaseTeleop`：

1. `robot.configure()` 回到初始姿态；
2. 获取当前 observation；
3. `teleop.set_teleop_enabled(True, obs)` 建立映射基准。

录制支持：

- 保存；
- 重录；
- 提前结束；
- 后台异步保存 episode；
- 录制完成后可推送 Hugging Face Hub。

### 15.5 记录动作与实际执行动作的差异

代码保存的是 `action_values`，而不是 `_sent_action`。同时 `UFRobot.send_action()` 即使因 controller error 或 `no_action` 跳过执行，也会返回原 action。

因此存在如下事实：

```text
数据集中记录了动作 ≠ 已证明控制器接受并执行了动作
```

这是数据质量和安全追踪上的重要缺口。

## 16. 策略推理链路

[`uf_lerobot_eval.py`](src/lerobot_robot_ufactory/scripts/uf_lerobot_eval.py) 会：

1. 加载 dataset metadata 或创建 metadata；
2. 检查 dataset FPS；
3. 创建 policy；
4. 创建 policy preprocessor/postprocessor；
5. `robot.connect()`；
6. 每个 episode 开始前 `robot.configure()`；
7. 每周期读取 observation、执行 policy、发送 action。

### 16.1 绝对与相对模式

默认策略按绝对动作运行。`--relative` 试图：

- 把实际 observation 转为相邻周期位移和相对旋转；
- 将 policy 预测的相对动作重新累计为绝对 TCP 目标。

`--rx_continuous` 试图缓解旋转向量在 π 附近的符号跳变。

但是当前相对输出转换代码被一个逻辑条件错误跳过，详见第 20.1 节。部署到真机前不能启用 `--relative`。

### 16.2 平滑逻辑

代码定义：

```python
SMOOTH_THRESHOLD = 0
SMOOTH_ROT_THRESHOLD = 0.05
```

位置平滑只在 `SMOOTH_THRESHOLD > 0` 时启用，所以当前平滑整体关闭，旋转阈值也不会单独生效。

### 16.3 夹爪 look-ahead

ACT/Diffusion Policy 的夹爪队列预读代码全部被注释，不参与实际推理。

## 17. 多机械臂机制

### 17.1 前缀

`MultipleUFRobot` 为每个子机器人传入配置 key：

```python
UFRobot(robot_config, prefix=key)
```

`UFRobot` 将其变为：

```python
self.prefix = f"{key}."
```

因此正确键格式是：

```text
left.pose.x
right.pose.x
left.J1.pos
right.J1.pos
```

### 17.2 连接与配置

默认配置：

- `async_connect=True`；
- `async_configure=True`；
- `async_action=False`。

连接和配置使用每臂一个线程并在主线程 `join()`。动作默认顺序发送，避免异步队列引入额外积压。

### 17.3 异步动作

如果 `async_action=True`：

- 每臂建立无界 `queue.Queue()`；
- 每臂建立 daemon action thread；
- 每次全局 action 按前缀分割并入队。

当前没有 latest-only、队列长度限制或过期动作丢弃策略。如果发送速度高于机械臂实际处理速度，旧动作可能积压并延迟执行。

### 17.4 当前多臂缺陷

- `UFRobot.is_connected` 是普通方法，`MultipleUFRobot` 却使用 `robot.is_connected` 而不是 `robot.is_connected()`；非空 bound method 恒为真。
- action thread 的 `while robot.is_connected:` 因同样原因可能永不自然退出。
- disconnect 没有向 action queue 放入终止标记，也没有 join action thread。
- 子线程中的连接/配置异常不会由父线程显式收集并重新抛出。
- `self.cameras.update(robot.cameras)` 没有给 camera 字典 key 加臂前缀，同名相机可能覆盖。
- eval 中构造多臂前缀的方式与 `UFRobot` 不一致。

## 18. 运动学、规划与碰撞能力边界

### 18.1 实际存在的“规划”

项目中的规划只有：

1. xArm 控制箱 Mode 6 的关节在线规划；
2. xArm 控制箱 Mode 7 的笛卡尔在线规划与内部 IK；
3. eval 中的相对旋转计算和可选动作平滑；
4. Tracker 到机器人坐标的刚体变换。

### 18.2 不存在的主机侧规划

仓库没有：

- 根据 URDF 计算 FK/IK；
- 生成带时间戳、速度、加速度连续性的完整轨迹；
- 规划绕开障碍物；
- 选择 xArm7 冗余自由度解；
- 对奇异位姿做主机侧显式处理；
- 使用 SDK `get_inverse_kinematics()` 或 `get_forward_kinematics()`；
- 使用 `motion_type=1/2` 明确控制 `set_position_aa()` 的 IK fallback 策略。

### 18.3 碰撞能力

代码没有调用：

- `set_collision_sensitivity()`；
- `set_self_collision_detection()`；
- `set_collision_tool_model()`；
- reduced mode/fence mode；
- 任何环境几何碰撞库。

控制箱自身可能保留 Studio/固件中已有的安全配置，但本项目既不配置，也不验证这些配置。因此不能把控制箱潜在能力表述为本项目已实现的碰撞系统。

## 19. 并发、频率与实时性

### 19.1 线程构成

单臂笛卡尔运行时可能同时存在：

- 主控制循环线程；
- SDK 自身通信/报告线程；
- 项目 TCP 30000 接收线程；
- 各摄像头异步读取线程；
- 遥操作器线程（Pika/SpaceMouse 等）；
- 异步数据保存线程。

多臂时这些线程按臂扩展。

### 19.2 控制循环频率

频率由不同入口决定：

- teleop：`TeleopConfig.fps`，默认 30；
- record/eval：dataset FPS；
- SpaceMouse 位移增量：teleop 自己的 `frequency`；
- Pika 状态监控线程：Pika `frequency`，默认 100；
- TCP 30000：控制箱报告频率 250/200 Hz。

项目没有统一验证 SpaceMouse `frequency` 与实际主循环 FPS 是否一致。若用户只修改其中一个，实际速度会偏离 `max_pos_speed` 的名义值。

### 19.3 Mode 6/7 与实时系统

项目选择 Mode 6/7 而非 Mode 1 `servoj`/`servo_cartesian`。根据官方说明，Mode 6/7 由控制器分解和规划目标，对主机实时内核和固定高频发送的要求低于 Mode 1，但新命令会打断当前在线目标。

这解释了项目可以在 10/30 Hz 的 LeRobot 循环中发送较稀疏目标，而不直接实现 50–200 Hz 的主机插值轨迹。

## 20. 已确认缺陷与风险清单

### 20.1 P0：真机动作安全相关

#### 20.1.1 eval 相对动作输出转换被错误跳过

代码：

```python
if not curr_robot_dict[key]['type'] != 1 or prev_action_dict[key]['type'] != 1:
    continue
```

当 `curr_robot_dict[key]['type'] == 1`，即笛卡尔机器人时：

```text
not (1 != 1) == True
```

所以直接 `continue`，相对预测动作不会被累计回绝对 TCP 目标。随后动作仍交给绝对 `set_position_aa()`。

**结论：当前 `--relative` 不适合真机使用。**

#### 20.1.2 没有主机侧动作边界

`send_action()` 不检查：

- 关节目标范围；
- 单周期最大关节变化；
- TCP 工作空间；
- 单周期最大 TCP 位移；
- 姿态跳变；
- 与桌面、相机或外部物体的碰撞；
- 数据是否为 NaN/Inf。

速度参数只能限制控制箱规划速度，不能替代目标合法性检查。

#### 20.1.3 SDK 动作返回码被忽略

`set_servo_angle()`、`set_position_aa()` 和夹爪 Modbus 返回值都没有被检查。控制器拒绝动作、通信失败或 SDK 前置检查失败时，调用方不会得到明确异常。

#### 20.1.4 controller error 时静默丢弃

```python
if self.real_arm.error_code != 0:
    return action
```

该分支不记录错误、不停止上层循环，并返回原 action，容易让调用者误认为动作已成功发送。

#### 20.1.5 每次复位自动清错

`configure()` 无条件执行 `clean_error()`，随后还执行 `set_state(0)`。代码没有在清错前保存错误上下文或区分碰撞、过流、限位与通信错误。

风险是上层反复 reset 时掩盖需要人工检查的硬件原因。

### 20.2 P0/P1：推理退出与状态处理

#### 20.2.1 eval 没有 `robot.disconnect()`

eval 正常退出只停止键盘 listener，没有调用 `robot.disconnect()`。笛卡尔 `UFRobot` 的报告线程不是 daemon，可能造成进程不能干净退出，且没有显式发送 state 4。

#### 20.2.2 多臂 eval 前缀错误

eval 使用：

```python
prefix = f'.{key}'
```

组合结果为 `.leftpose.x`；实际正确格式是 `left.pose.x`。因此多臂 TCP 检测和相对动作处理不能按设计工作。

#### 20.2.3 平滑默认关闭

尽管代码注释描述了 chunk boundary smoothing，`SMOOTH_THRESHOLD = 0` 使其完全关闭。不能把注释中的平滑能力视为当前运行事实。

### 20.3 P1：TCP 30000 健壮性

#### 20.3.1 超时异常未捕获

socket timeout 为 1 秒；任何超过 1 秒无数据都会抛出未捕获异常。

#### 20.3.2 可能保留“正常”假象

如果线程在循环内部异常退出，函数尾部的：

```python
self._rt_report_normal = False
```

不会执行。控制线程可能继续读取最后一次更新的旧位姿。

#### 20.3.3 无新鲜度判断

没有保存 TCP 30000 timestamp 或本地接收时间，上层无法识别 stale observation。

#### 20.3.4 Thread 对象不可重复启动

同一个 `threading.Thread` 实例只能 `start()` 一次。如果报告线程退出后 `configure()` 再次执行 `self.start()`，会触发 `RuntimeError`。断开后 stop event 也没有清除，当前对象不支持可靠 reconnect。

### 20.4 P1：依赖兼容性

#### 20.4.1 SDK 未锁版本

`pyproject.toml` 只写：

```toml
"xarm-python-sdk"
```

没有最小/最大版本或 lock 文件。

#### 20.4.2 使用 SDK 私有字段

`real_arm._arm._baud_checkset` 不是稳定公开 API。

#### 20.4.3 使用旧 Modbus API 名

`getset_tgpio_modbus_data()` 当前仍兼容，但官方已经推荐 `set_rs485_data()`。

#### 20.4.4 `set_linear_spd_limit_factor` 调用条件不匹配

当前官方 SDK 文档说明该接口要求：

- 固件 2.3.0+；
- Mode 1。

项目却在完成 Mode 6/7 配置后调用，并忽略返回码。按当前 SDK 语义，它不能作为本项目 Mode 6/7 的可靠安全限制。

### 20.5 P1/P2：多臂并发

- child `is_connected` 被当属性而不是方法；
- action queue 无界；
- 无 stale action 丢弃；
- disconnect 不终止/等待 action thread；
- 线程异常没有集中传播；
- 同名 camera key 可能覆盖。

### 20.6 P2：遥操作器明确缺陷

#### Pika

```python
self.self.set_teleop_enabled(False)
```

是确定的属性访问错误，在对应按键状态切换分支会抛 `AttributeError`。

关闭遥操作时还假设 `_last_action` 一定不是 `None`，提前暂停可能出现下标错误。

#### SpaceMouse

- `is_calibrated` 访问 `_is_calibrated`，但构造函数没有初始化该字段；
- `connect()` 启动线程后立即返回，`_is_connected=True` 由线程稍后设置，存在短暂竞争；
- 直接 teleop 脚本没有增量转绝对动作适配。

#### GELLO

- 构造对象时即访问串口，而不是等到 `connect()`；
- `gripper_id=-1` 时 feature/get_action 仍固定生成 `gripper.pos`，需要额外验证数组长度；
- `disconnect()` 没有调用 `UFBaseTeleop.disconnect()`，不会从上下文注册表注销。

#### Multiple UMI

`action_features` 缺少 `@property`，与单 UMI 和 LeRobot feature 使用方式不一致。

### 20.7 P2：其他实现问题

- `get_observation()` 的 SDK `code` 普遍未检查；
- 未知 control space 分支构造了 `ValueError` 但没有 `raise`；
- `send_action()` 注解为 `np.ndarray`，实际返回 `dict`；
- `print_logs()` 为空；
- `calibrate()` 为空；
- 没有明确配置 TCP offset、payload、重力方向；
- 没有验证控制箱当前 Studio 安全参数；
- xArm7 GELLO 数据集名称错误标为 xArm6。

## 21. 固件与 SDK 兼容性要求

### 21.1 从当前实现推导的最低需求

| 功能 | 官方要求/现状 |
|---|---|
| Mode 6 | 固件 ≥1.10.0 |
| Mode 7 | 固件 ≥1.11.0 |
| `set_position_aa` 轴角运动 | SDK/固件需支持对应接口；较新固件支持更完整 `motion_type` |
| TCP 30000 基础实时示例 | 官方示例写明固件 ≥2.1.101 |
| TCP 30000 当前完整 784 字节表 | 官方当前字段页标注固件 ≥2.7.101 |
| Gripper G2/Bio G2 API | 依赖较新 SDK 与对应末端固件 |
| `set_linear_spd_limit_factor` | 当前 SDK 文档：固件 ≥2.3.0 且仅 Mode 1 |

### 21.2 建议的启动握手

当前代码没有实现，但可靠部署应在使能前读取并记录：

- SDK 包版本；
- 控制器固件版本；
- 机械臂 SN 与轴数；
- mode/state；
- controller error/warn；
- TCP offset；
- payload；
- collision sensitivity；
- self-collision/reduced/fence 状态；
- TCP 30000 首帧长度和 timestamp 单调性。

## 22. 安全部署建议

### 22.1 首次连接前

1. 机械臂与控制箱网线直连或使用已验证低延迟网络；
2. 确认急停可用；
3. 清空工作空间人员和障碍物；
4. 在 UFACTORY Studio 中确认 TCP、负载、安装方向和碰撞灵敏度；
5. 核对真实 IP、轴数和夹爪类型；
6. 将起始关节姿态设置为无碰撞姿态；
7. 先使用 `no_action: true` 检查 observation 和 feature；
8. 再用低速度、无策略的单步命令验证。

### 22.2 不应直接用于真机的当前功能

- `uf-lerobot-eval --relative`；
- 未加边界检查的未知来源 policy；
- 未验证过的 `async_action=True` 多臂控制；
- 报告线程发生过异常后的继续运行；
- 依赖未锁定、固件未核对的部署环境。

### 22.3 建议停止策略

可靠实现应至少区分：

- 用户普通退出：减速/停止、state 4、关闭夹爪动作、断开；
- policy 输出非法：立即停止发送，并进入安全状态；
- SDK transport error：停止并要求重新连接；
- controller collision/overcurrent error：保留错误信息，禁止自动清错；
- TCP observation stale：停止动作，不允许沿用旧 observation；
- 摄像头失败：根据 policy 是否依赖该相机决定停止，而不是继续输入旧图或空图。

## 23. 建议的改进路线

### 阶段 1：动作安全与可观测性

1. 为所有 SDK 调用建立统一 `_check_code(operation, code)`；
2. controller error 时抛出带错误码的异常，不再静默返回；
3. 记录“请求动作、processor 后动作、SDK 接受动作、实际反馈”；
4. 增加 NaN/Inf、关节范围、工作空间和单步变化限制；
5. 将 `clean_error()` 改为显式、分类、可审计的恢复流程；
6. 启动时核对 TCP、payload 和安全配置。

### 阶段 2：修复推理与退出

1. 修复 eval 的笛卡尔条件表达式；
2. 修复多臂前缀为 `f"{key}."`；
3. 平滑状态按机械臂分别保存；
4. 用 `try/finally` 确保 `robot.disconnect()`；
5. relative 模式增加离线单元测试和真机小步验证；
6. 明确启用或删除当前无效平滑/夹爪 look-ahead 代码。

### 阶段 3：重构 TCP 30000 接收器

1. 独立为可重连 receiver 类；
2. 严格按每帧 header 解析和校验长度；
3. 捕获 timeout、EOF 和解析错误；
4. `finally` 清状态、关闭 socket；
5. 保存控制器 timestamp、本地接收时间和 sequence；
6. 暴露 observation age；
7. 超龄即阻止动作；
8. 支持停止后创建新线程重新连接。

### 阶段 4：SDK 与夹爪兼容性

1. 锁定经过真机验证的 `xarm-python-sdk` 版本；
2. 移除 `_arm._baud_checkset` 私有访问；
3. 替换为 `set_rs485_data()`；
4. 检查所有夹爪返回码；
5. 修复夹爪归一化初值与 clamp；
6. 为不同夹爪建立独立 adapter。

### 阶段 5：多臂和遥操作稳定性

1. 统一 `is_connected/is_calibrated` 为属性或方法；
2. action queue 改为长度 1 的 latest-only；
3. disconnect 时发送 sentinel 并 join；
4. 汇总传播连接线程异常；
5. camera key 加臂前缀；
6. 修复 Pika、SpaceMouse、GELLO、Multiple UMI 的明确缺陷；
7. 统一所有入口对 delta/absolute action 的处理。

### 阶段 6：模型与碰撞层（按需求选择）

如果任务需要在复杂环境自主推理，应额外引入：

- 与实机标定参数一致的 xArm7 URDF/SRDF；
- TCP/工具几何模型；
- FK/IK 与奇异性检测；
- 自碰撞和环境碰撞检测；
- 轨迹可行性检查；
- MoveIt 或等价规划系统；
- 最终控制箱安全边界作为独立最后防线。

## 24. 测试建议

### 24.1 当前测试现状

`pyproject.toml` 配置了 pytest、ruff 和 mypy，但仓库当前没有 `tests/` 文件。当前基础环境缺少项目运行依赖，所以本次没有执行真机或集成测试。

### 24.2 最小单元测试集合

应使用 `FakeXArmAPI` 覆盖：

1. xArm7 `robot_dof=7` 与 axis 校验；
2. degree/radian 转换；
3. 首条关节命令 Mode 0 + wait；
4. 后续关节命令 Mode 6 + no-wait；
5. 笛卡尔 Mode 7 + `set_position_aa` 参数；
6. SDK 非零返回码传播；
7. controller error 下动作不会被记录为成功；
8. 各夹爪归一化和硬件位置映射；
9. 多臂前缀和动作拆分；
10. disconnect 的 state/mode 顺序。

### 24.3 TCP 30000 测试

构造二进制 fixture，验证：

- 大端 frame size；
- 小端 FP32；
- 116/144/424/448/472/496 字节偏移；
- 分段 recv；
- 多帧粘包；
- EOF；
- timeout；
- 帧长不足；
- timestamp 回退；
- observation stale；
- 重连。

### 24.4 Eval 测试

重点测试：

- absolute 轴角动作；
- relative position 累计；
- relative rotation 矩阵复合；
- ±π 附近轴角连续性；
- 单臂与 `left./right.` 多臂键；
- 平滑状态不跨机械臂污染；
- ESC/异常路径总会 disconnect。

### 24.5 真机验证顺序

```text
无动作连接
  → 只读关节状态
  → 只读 TCP 30000
  → Mode 0 低速单点
  → Mode 6 小关节变化
  → Mode 7 小 TCP 平移
  → 姿态小角度
  → 夹爪
  → 遥操作
  → 数据录制
  → 有安全边界的 policy
  → 多臂
```

每一级通过后再进入下一级。

## 25. 建议阅读顺序

为了快速建立正确心智模型，建议按以下顺序阅读代码：

1. [`config/gello/xarm7_gello_record_config.yaml`](config/gello/xarm7_gello_record_config.yaml) 或其他 xArm7 配置；
2. [`uf_robot_config.py`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot_config.py)；
3. [`uf_robot.py`](src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py) 的构造、`connect()`、`configure()`；
4. `get_observation()` 和 `send_action()`；
5. TCP 30000 `run()`；
6. 对应遥操作器；
7. `uf_robot_teleop.py`；
8. `uf_lerobot_record.py`；
9. `uf_lerobot_eval.py`；
10. `multiple_uf_robot.py` 和多 UMI；
11. 本文第 20 章缺陷与风险清单。

## 26. 官方参考资料

以下链接用于核对 SDK 方法、模式和 TCP 协议；访问日期为 2026-08-15。

1. [xArm Python SDK 官方仓库](https://github.com/xArm-Developer/xArm-Python-SDK)
2. [`XArmAPI` 官方包装层源码](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py)
3. [机器人 state 与 mode 说明](https://docs.supportarticle.ufactory.cc/support_articles/developer/robot-state-and-mode-explanation.html)
4. [Servo/在线规划模式使用说明](https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-servo-mode-guide.html)
5. [TCP 30000 实时数据示例](https://docs.supportarticle.ufactory.cc/support_articles/developer/firmware/how-to-get-the-real-time-data-via-tcp-30000-port.html)
6. [TCP 端口字段定义](https://docs.supportarticle.ufactory.cc/support_articles/developer/firmware/data-description-of-tcp-port.html)
7. [Mode 6 官方 Python 示例](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/example/wrapper/common/2006-joint_online_trajectory_planning.py)
8. [Mode 7 官方 Python 示例](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/example/wrapper/common/1010-cartesian_online_trajectory_planning.py)

## 27. 最终结论

本项目已经建立了从 LeRobot 配置、遥操作/策略、数据特征到 xArm SDK 的完整适配链：

- xArm7 通过通用 `UFRobot + robot_dof=7` 表达；
- 关节控制使用 Mode 6 与 `set_servo_angle()`；
- 笛卡尔控制使用 Mode 7 与轴角 `set_position_aa()`；
- 笛卡尔反馈使用 TCP 30000 高速数据；
- 支持多种遥操作器、摄像头、夹爪、数据录制和多臂聚合。

但当前实现仍属于研究/原型性质。尤其是：

- 缺少动作边界与碰撞层；
- 忽略大量 SDK 返回码；
- eval relative 路径存在确定逻辑错误；
- TCP 30000 线程缺少 stale、异常和重连保护；
- 多臂与若干遥操作器存在明确实现缺陷；
- SDK 未锁版本，且使用私有/旧接口；
- 没有核心自动化测试。

因此，在修复第 20 章的 P0/P1 项并完成第 24 章的分级验证之前，不应把当前代码视为具备完整安全闭环的 xArm7 生产控制系统。
