# πR²-Flow 项目 XHand 机制、代码实现与 SDK 调用全面技术文档

> 文档日期：2026-08-15<br>
> 仓库基线提交：`3af52ca400a6ec7d141416879aa531a6f62f697a`<br>
> 审查对象：`pi-r2-flow` 当前工作区、当前提交中的部署代码，以及本机已安装的 XHand SDK 绑定与可见 SDK 源码/示例<br>
> 审查方式：静态代码追踪、SDK Python 接口自省、无硬件结构测试和 mock 行为复现<br>
> 安全声明：本次审查未打开 XHand 端口、未连接机器人、未向真实硬件发送命令

---

## 目录

1. [文档目标、范围与证据规则](#1-文档目标范围与证据规则)
2. [结论摘要](#2-结论摘要)
3. [XHand 在项目中的系统位置](#3-xhand-在项目中的系统位置)
4. [代码边界与文件职责](#4-代码边界与文件职责)
5. [XHand 逻辑模型、关节顺序与传感器](#5-xhand-逻辑模型关节顺序与传感器)
6. [驱动类的公开接口与内部状态](#6-驱动类的公开接口与内部状态)
7. [连接、初始化、回零与关闭生命周期](#7-连接初始化回零与关闭生命周期)
8. [厂商 SDK 调用方式与数据结构](#8-厂商-sdk-调用方式与数据结构)
9. [状态读取、缓存与解析机制](#9-状态读取缓存与解析机制)
10. [指尖触觉复零与软件偏置机制](#10-指尖触觉复零与软件偏置机制)
11. [动作生成、转换与下发机制](#11-动作生成转换与下发机制)
12. [XHand 与 GR00T/πR² 策略的集成](#12-xhand-与-gr00tπr²-策略的集成)
13. [时序、线程与并发模型](#13-时序线程与并发模型)
14. [数据记录与可视化链路](#14-数据记录与可视化链路)
15. [当前安全机制与缺失的安全层](#15-当前安全机制与缺失的安全层)
16. [依赖、版本与可复现性](#16-依赖版本与可复现性)
17. [已确认问题与风险分级](#17-已确认问题与风险分级)
18. [建议的目标架构与修复设计](#18-建议的目标架构与修复设计)
19. [测试计划与验收标准](#19-测试计划与验收标准)
20. [真机部署检查清单](#20-真机部署检查清单)
21. [分阶段实施路线](#21-分阶段实施路线)
22. [最终判断](#22-最终判断)

---

# 1. 文档目标、范围与证据规则

## 1.1 文档目标

本文档集中回答以下问题：

1. XHand 在 πR²-Flow 部署栈中处于什么位置；
2. XHand 的连接、状态读取、触觉读取、动作发送和关闭机制如何实现；
3. 项目实际调用了厂商 SDK 的哪些接口，各接口的参数和返回值如何使用；
4. 12 维手部状态与动作如何进入 GR00T/πR² 推理链路；
5. 主循环的时序、线程、缓存和错误处理机制是什么；
6. 当前实现中哪些行为已经由代码或测试确认，哪些仍需真机确认；
7. 在真实机器人上运行前，应优先修复哪些问题并建立哪些测试。

## 1.2 审查范围

仓库内直接相关文件：

- [`deployment/mindex/robots/xhand_robot.py`](deployment/mindex/robots/xhand_robot.py)：XHand 驱动封装；
- [`deployment/apps/run_policy.py`](deployment/apps/run_policy.py)：XHand 生命周期、状态采集和动作下发入口；
- [`deployment/apps/_policy_args.py`](deployment/apps/_policy_args.py)：通信协议、串口、控制频率和动作模式参数；
- [`deployment/mindex/policy/control_utils.py`](deployment/mindex/policy/control_utils.py)：回零插值、delta/absolute 转换和未接入的安全辅助函数；
- [`deployment/mindex/policy/groot_client.py`](deployment/mindex/policy/groot_client.py)：GR00T 状态与动作 modality 的拆装；
- [`deployment/mindex/recording/dataset.py`](deployment/mindex/recording/dataset.py)：HDF5 记录；
- [`deployment/scripts/render_episode_modalities.py`](deployment/scripts/render_episode_modalities.py)：完整手部模态可视化；
- [`deployment/scripts/render_finger_focus.py`](deployment/scripts/render_finger_focus.py)：按手指查看指尖力和关键关节动作；
- [`README.md`](README.md) 和 [`deployment/pyproject.toml`](deployment/pyproject.toml)：公开用法和依赖声明。

本机 SDK 证据：

- `/usr/local/lib/python3.12/dist-packages/xhand_controller/`；
- `/usr/local/xhand_controller/` 中可见的 SDK 头文件、源码和测试；
- `real_robot` Conda 环境中的 `xhand_controller` Python 绑定；
- 本机保存的厂商 Python 1.1.8 示例。

## 1.3 明确不在当前仓库中的内容

当前提交没有提供以下 XHand 相关组件：

- XHand URDF、MJCF 或其他机器人描述；
- SRDF、自碰撞对、环境碰撞模型；
- XHand 逆运动学、抓取规划或轨迹优化；
- 手臂与灵巧手的联合运动规划；
- XHand 独立实时进程、共享内存或硬实时控制器；
- 注释中提到的 `run_teleop.py` 和基于 gello 的采集驱动；
- XHand 单元测试、mock SDK 或硬件回归测试；
- 项目专用的 XHand 训练 modality 配置文件。

`learning/Isaac-GR00T` 是 Git submodule，但当前工作区未 checkout。仓库固定到提交
`2a39d591a3af24c42bb1d1c9cd708a2dddac0600`。因此本文可以完整审查本地部署客户端，
但不能声称已经审查该固定提交中的全部服务端训练实现。

## 1.4 证据分级

本文使用以下标签控制结论强度：

- **代码事实**：可由当前仓库代码直接确认；
- **SDK 事实**：可由本机已安装 SDK 的签名、头文件、源码或厂商示例确认；
- **已复现**：通过不接硬件的结构测试或 mock 测试重现；
- **推断**：由代码路径和接口语义推导，但未在真实硬件上观测；
- **待确认**：缺少厂商单位说明、目标设备固件信息或真机数据，不能下确定结论。

本机 SDK 事实只描述当前机器可见版本，不自动代表所有 XHand SDK/固件版本。

---

# 2. 结论摘要

## 2.1 架构判断

XHand 在当前项目中由一个轻量 Python wrapper 直接控制：

```text
XHand 硬件
  ↕ RS485 3 Mbps 或 EtherCAT
xhand_controller 厂商 SDK
  ↕ HandCommand_t / HandState_t
XHandRobot
  ├─ 连接和 hand ID 选择
  ├─ 12 维位置命令封装
  ├─ 关节状态、温度、电流解析
  ├─ 5 指尖汇总力与 5×120×3 原始触觉解析
  └─ 最近状态缓存
       ↕
run_policy.py 主线程
  ├─ fresh XHand 状态
  ├─ xArm 状态
  ├─ 相机帧
  ├─ GR00T/πR² 查询
  ├─ absolute/delta 动作转换
  └─ 12 维手部位置目标下发
```

这是单进程、主线程直接访问硬件的研究部署结构，不包含独立规划器或完整安全控制器。

## 2.2 已实现能力

当前代码已经实现：

- 12-DOF XHand 位置控制；
- RS485 与 EtherCAT 两种连接入口；
- 串口自动枚举和有限 fallback；
- 关节位置、反馈电流、关节温度读取；
- 五指尖汇总三轴力、原始 120 点三轴触觉和温度读取；
- fresh 状态与最近命令缓存状态两种读取语义；
- 触觉硬件 reset 调用和运行期软件 bias；
- 联合 xArm6 的 18 维动作空间；
- 18 维或富状态 modality 的动态组装；
- 手臂与手分别选择 absolute/delta 动作解释；
- home 与首动作的关节空间线性插值；
- HDF5 记录和触觉/动作可视化；
- 正常退出时尝试切换到无力模式。

## 2.3 最重要的已确认问题

1. **触觉 reset ID 不符合厂商接口**：代码把关节 ID `(2,5,7,9,11)` 当成传感器 ID 传给
   `reset_sensor()`；厂商测试给出的传感器 ID 是 `0x11..0x15`。
2. **复零验证会掩盖静态接触**：软件 bias 使用当前五帧均值，静态接触会被直接减成零；已用 mock 复现。
3. **主动作链缺少软件安全包络**：没有 finite 检查、关节硬限位、max-delta、速度/加速度限制、
   电流/温度/触觉联锁和 stale watchdog。
4. **命令失败仍被当成已经执行**：驱动只打印 warning，主循环仍把动作写入 executed history、
   inpaint 历史和评估日志。
5. **45 维状态与 hand-delta 维度不兼容**：已复现 `(39,) + (12,)` 的广播异常。
6. **异常退出清理范围不足**：正式主循环之前的失败和人工取消不保证执行 `hand.disconnect()`；
   `disconnect()` 也未显式调用 SDK 已提供的 `close_device()`。
7. **反馈量命名可能误导**：SDK 1.5 头文件将 `FingerState_t.torque` 描述为实时电流 mA，
   项目却命名为 `joint_torques`。

这些问题不否定驱动的基本可用性，但说明当前实现不应直接被视为生产级安全控制栈。

---

# 3. XHand 在项目中的系统位置

## 3.1 机器人组成

项目面向：

- UFactory xArm6：6 个关节；
- XHand：12 个主动关节；
- Intel RealSense：顶视相机；
- 远程 GR00T 推理服务；
- 默认联合动作维度：18。

动作切片固定为：

```text
action[0:6]   = xArm6 joint target
action[6:18]  = XHand joint target
```

使用 `--no-arm` 时，客户端把 `arm_dofs` 设为 0，并期望策略输出本身也是 12 维手部动作。
客户端不会自动从一个 18 维服务器响应中删除机械臂部分。

## 3.2 与 πR² 快慢通道的关系

πR² 的部署思路是：

- 视觉语言特征是低频慢通道；
- 本体状态是每控制 tick 更新的快通道；
- DiT 可在视觉特征较旧时使用最新关节和触觉状态重新规划。

项目中 XHand 提供快通道的重要部分：

- 手部关节位置；
- 手部反馈电流；
- 指尖汇总力。

原始 `5×120×3` 触觉被记录到 HDF5，但当前 GR00T 状态构建只支持扁平的
`fingertip_force`，不把完整触觉阵列输入策略。

## 3.3 XHand 与规划/碰撞的边界

当前代码没有调用：

- IK；
- 碰撞几何；
- 自碰撞检查；
- 环境碰撞检查；
- 轨迹时间参数化；
- 关节约束规划器。

`interpolate_to()` 只是从当前测量位置到目标位置生成 `np.linspace()` 关节路点并逐点发送。
因此“可以平滑回零”不等于“路径无碰撞”或“满足所有关节动力学约束”。

---

# 4. 代码边界与文件职责

## 4.1 XHand 驱动

[`xhand_robot.py`](deployment/mindex/robots/xhand_robot.py) 定义：

- `NUM_JOINTS = 12`；
- `MODE_PASSIVE = 0`；
- `MODE_POSITION = 3`；
- `MODE_FORCE = 5`；
- `SENSORED_FINGER_IDS = (2, 5, 7, 9, 11)`；
- `XHandRobot` 类。

其接口模仿 LeRobot 风格：

```python
connect()
disconnect()
get_observation(fresh=False)
send_action(action)
is_connected
num_dofs
```

这不是对 LeRobot `Robot` 抽象类的正式继承，只是形状和方法名相似。

## 4.2 部署入口

[`run_policy.py`](deployment/apps/run_policy.py) 负责：

1. 构造或 fake XHand；
2. `connect()`；
3. `reset_sensors()`；
4. 连接 xArm、相机和 GR00T；
5. home；
6. 填充历史状态；
7. warm-start；
8. 进入 25 Hz 控制循环；
9. 发送手臂和 XHand 动作；
10. 正常 finally 中断开机器人。

## 4.3 控制辅助函数

[`control_utils.py`](deployment/mindex/policy/control_utils.py) 包含：

- `interpolate_to()`：home 和首动作 ramp；
- `to_absolute()`：按 limb 将 delta 转换为绝对关节目标；
- `clip_to_state()`：每 tick 最大变化裁剪；
- `apply_ema()`：手臂/手分别做动作低通；
- `get_obs_retry()`：启动阶段状态重试。

其中 `clip_to_state()` 和 `apply_ema()` 当前没有接入 `run_policy.py` 的主动作路径。

## 4.4 GR00T 客户端

[`groot_client.py`](deployment/mindex/policy/groot_client.py) 声明状态维度：

```python
arm_joint_position  = 6
hand_joint_position = 12
arm_joint_torque    = 6
hand_joint_torque   = 12
fingertip_force     = 15
```

服务端通过 modality config 返回 `state_keys`，客户端据此决定每 tick 拼接哪些字段。

该文件重复定义了两次 `state_keys` property。两者当前实现相同，运行行为没有差异，
但这是代码维护上的重复。

## 4.5 记录与可视化

`EpisodeWriter` 可保存：

- `observation/state`；
- `observation/joint_torque`；
- `observation/fingertip_force`；
- `observation/tactile`；
- `observation/images/overhead`；
- `action`；
- `action_delta`；
- 时间戳和 query 事件。

两个可视化脚本都假定默认联合系统有 6 个 arm DOF，并据此从 18 维数组切出手部。

---

# 5. XHand 逻辑模型、关节顺序与传感器

## 5.1 12 维关节顺序

项目可视化脚本给出的顺序如下：

| 手部局部索引 | 手指 | 标签/含义 |
|---:|---|---|
| 0 | thumb | `thumb_bend` |
| 1 | thumb | `thumb_rota1` |
| 2 | thumb | `thumb_rota2` |
| 3 | index | `index_bend` |
| 4 | index | `index_j1` |
| 5 | index | `index_j2` |
| 6 | middle | `middle_j1` |
| 7 | middle | `middle_j2` |
| 8 | ring | `ring_j1` |
| 9 | ring | `ring_j2` |
| 10 | pinky | `pinky_j1` |
| 11 | pinky | `pinky_j2` |

联合 18 维状态/动作中，上表索引整体加 6。

代码没有在运行时读取关节名或设备类型来确认该顺序，因此训练数据、checkpoint、
SDK 返回顺序和动作下发顺序必须由部署者保证一致。

## 5.2 带指尖传感器的关节

代码把以下关节视为各手指的末端传感器关联关节：

```text
thumb  → joint 2
index  → joint 5
middle → joint 7
ring   → joint 9
pinky  → joint 11
```

这解释了 `SENSORED_FINGER_IDS = (2,5,7,9,11)` 的来源。

## 5.3 SDK 中的传感器槽位

`HandState_t.sensor_data` 长度为 5。每个 `SenserData_t` 包含：

- `calc_force`：三轴汇总值；
- `raw_force`：长度 120，每个元素含 `fx/fy/fz`；
- `temperature`：长度 20；
- `calc_temperature`：汇总温度。

项目直接假定：

```text
sensor_data[0] = thumb
sensor_data[1] = index
sensor_data[2] = middle
sensor_data[3] = ring
sensor_data[4] = pinky
```

本机 SDK 的 EtherCAT 接收源码把传感器总线 ID `17..21` 映射到
`sensor_data[sensor_id - 17]`，支持槽位按 bus ID 递增排列；厂商 RS485/EtherCAT 测试也都把
有效 reset ID 标为 `0x11..0x15`。不过，“槽位 0..4 分别就是 thumb..pinky”仍依赖设备接线和
手型约定，项目没有利用 `FingerState_t.sensor_id` 或设备类型做运行时交叉验证。

## 5.4 数据单位

可以较高置信度确认：

- 关节 `position`：项目按弧度使用；
- `FingerCommand_t.position`：项目按弧度写入；
- `FingerState_t.temperature & 0xFF`：厂商示例按摄氏温度读取低字节；
- SDK 1.5 头文件把 `FingerState_t.torque` 描述为实时电流 mA；
- 厂商 README 把 `tor_max` 描述为电流 mA。

不能仅由当前厂商文档确认：

- `calc_force.fx/fy/fz` 是否已经严格标定为 N；
- `raw_force` 的每个数值对应什么传感单元和物理单位；
- 指尖汇总力的方向坐标系；
- 不同手型、固件和 SDK 版本是否使用相同标定。

因此当前代码和图表中的 `N` 应视为项目假设，不能直接作为通用安全阈值依据。

---

# 6. 驱动类的公开接口与内部状态

## 6.1 构造参数

`XHandRobot.__init__()` 接受：

| 参数 | 类默认值 | `run_policy` CLI 默认值 | 说明 |
|---|---:|---:|---|
| `protocol` | `RS485` | `RS485` | 可选 RS485/EtherCAT |
| `serial_port` | `/dev/serial/by-id/...` | `/dev/ttyUSB0` | CLI 实际覆盖类默认值 |
| `baud_rate` | 3,000,000 | 3,000,000 | RS485 波特率 |
| `kp` | 100 | 不暴露 | 位置环比例增益 |
| `ki` | 0 | 不暴露 | 积分增益 |
| `kd` | 0 | 不暴露 | 微分增益 |
| `tor_max` | 300 | 不暴露 | 电流/力矩命令上限字段 |
| `mode` | 3 | 不暴露 | 默认位置模式 |

`kp/ki/kd/tor_max/mode` 被强制转换为 Python `int`，因为 pybind 属性要求整数。

## 6.2 内部对象

驱动维护：

- `_device`：厂商 `XHandControl`；
- `_command`：预分配 `HandCommand_t`；
- `_hand_id`：选中的手 ID；
- `_connected`：逻辑连接状态；
- `_lock`：Python 层总线调用锁；
- `_cached_obs`：最近解析后的观测字典；
- `_bias_ft`：`(5,3)` 汇总力 bias；
- `_bias_raw`：`(5,120,3)` 原始触觉 bias。

软件 bias 只存在进程内存中：

- 不持久化；
- 不写入 episode 元数据；
- 重启后丢失；
- 无法从现有 HDF5 判断某个 episode 使用了什么 bias。

## 6.3 缓存所有权

`get_observation(fresh=False)` 直接返回 `_cached_obs`，没有深拷贝。

当前调用方主要读取或用 `np.asarray()` 转换，没有主动修改缓存；但从接口设计上看，
外部代码若原地修改其中数组，会污染驱动缓存。更稳妥的公共接口应返回只读视图或副本。

---

# 7. 连接、初始化、回零与关闭生命周期

## 7.1 RS485 连接流程

`connect()` 在 RS485 模式下执行：

1. 延迟导入 `xhand_controller.xhand_control`；
2. 创建 `XHandControl()`；
3. `enumerate_devices("RS485")`；
4. 把用户请求串口放在候选列表第一位；
5. 追加枚举结果中其他 `/dev/ttyUSB*`；
6. 对候选串口逐个调用 `open_serial(port, baud_rate)`；
7. `list_hands_id()`；
8. 发现 ID 后选择第一个并停止搜索。

限制：

- 只自动追加 `/dev/ttyUSB*`，不自动追加其他串口命名；
- 没有 `--hand-id` 参数；
- 多手设备时始终选择第一个 ID；
- 不读取序列号、左右手类型或设备名称确认身份；
- CLI 默认 `/dev/ttyUSB0` 可能随 USB 枚举顺序变化。

代码注释认为 SDK 没有显式 close；但本机 1.1.8 和 1.5.2 Python 绑定都提供
`close_device()`。该注释已与本机 SDK 事实不一致。

## 7.2 EtherCAT 连接流程

EtherCAT 路径执行：

1. `enumerate_devices("EtherCAT")`；
2. 选择第一张接口；
3. `open_ethercat(ports[0])`；
4. 检查 `error_code`；
5. `list_hands_id()` 并选择第一个 ID。

驱动说明要求 Python 解释器具备 `cap_net_raw+ep`。这是一项系统权限变更，
应该在部署说明中明确记录解释器绝对路径和安全影响，不能仅依赖注释。

## 7.3 命令结构初始化

连接成功后分配一个 `HandCommand_t`，并对 12 个 `finger_command` 设置：

```text
id       = 0..11
kp       = 100
ki       = 0
kd       = 0
tor_max  = 300
mode     = 3
position = 0.0
```

这里只是准备 Python 命令对象；`connect()` 本身没有显式调用项目层的 `send_command()`。

## 7.4 启动后的复零

真实硬件模式中，`run_policy.py` 紧接着调用：

```python
hand.connect()
hand.reset_sensors()
```

触觉复零因此发生在：

- xArm 连接之前；
- 相机连接之前；
- GR00T 连接之前；
- home 运动之前。

代码没有在复零前提示操作者确保五个指尖完全无接触。

## 7.5 Home

默认手部 home 是 12 维全零：

```python
HAND_HOME = np.zeros(12, dtype=np.float32)
```

`interpolate_to()` 在 150 步、25 Hz 下运行约 6 秒。它从当前观测到 home 做线性关节插值。

细节：`interpolate_to()` 调用 `get_obs_retry(hand)` 时没有传 `fresh=True`。如果驱动已经有缓存，
起点可能来自最近缓存而不是该函数调用时强制采集的新状态。启动路径中这个缓存通常来自刚完成的复零验证，
但接口语义仍不是严格的实时起点。

## 7.6 首动作 Ramp

客户端填满状态/图像历史后：

1. 请求第一段 action chunk；
2. 根据 checkpoint 类型选择首个将执行的位置；
3. 等待用户按键确认；
4. 从当前状态插值到首动作；
5. 再次等待确认；
6. 进入正式主循环。

这能降低从 home 到模型动作的瞬时跳变，但仍未检查插值路径中的碰撞或硬关节范围。

## 7.7 正常关闭

正式主循环的 `finally` 中：

1. 通知 GR00T/VLM worker 停止；
2. 保存 eval log；
3. `hand.disconnect()`；
4. 断开 xArm；
5. 关闭 GR00T client。

`hand.disconnect()`：

1. 调用 `_set_mode(MODE_PASSIVE)`；
2. `_set_mode()` 把 12 个关节 mode 改为 0；
3. 发送整个 `HandCommand_t`；
4. 将 `_connected` 设为 `False`。

不足：

- 没有显式 `close_device()`；
- 没有把 `_device/_command/_hand_id/_cached_obs` 清空；
- 重新连接同一对象后，`fresh=False` 理论上可能返回旧连接的缓存；
- passive 命令异常被完全吞掉；
- 无法知道手是否真正进入 passive。

## 7.8 未被 finally 覆盖的早期退出

正式 `try/finally` 在 warmup、首次查询和两次人工确认之后才建立。以下情况不保证运行
`hand.disconnect()`：

- XHand 已连接后，xArm/相机/GR00T 初始化失败；
- home 或 warmup 异常；
- warmup 状态维度不匹配导致用户中断；
- 首次推理异常；
- 用户在两个人工确认点选择退出；
- 主循环开始前收到未被自定义 handler 接管的异常。

程序退出时 SDK 对象析构可能关闭串口，但“关闭通信”不等价于“已经成功发送 passive 命令”。

---

# 8. 厂商 SDK 调用方式与数据结构

## 8.1 本机可见 SDK 版本

本机存在两套可导入环境：

| Python 环境 | distribution 版本 | `XHandControl().get_sdk_version()` |
|---|---:|---:|
| 系统 Python 3.12 | 1.5.2 | 1.4.2 |
| `real_robot` Python 3.10 | 1.1.8 | 1.4.6 |

这说明 Python wheel/distribution 版本与底层 SDK 自报版本不是同一概念。项目没有锁定任何一个版本，
也没有在启动日志中记录它们。

## 8.2 项目实际使用的 SDK API

| SDK 方法 | 项目调用位置 | 参数 | 返回值处理 |
|---|---|---|---|
| `XHandControl()` | `connect()` | 无 | 保存为 `_device` |
| `enumerate_devices()` | `connect()` | `RS485`/`EtherCAT` | 生成候选接口 |
| `open_serial()` | `connect()` | 串口、波特率 | 检查 `error_code` |
| `open_ethercat()` | `connect()` | 网卡接口名 | 检查 `error_code` |
| `list_hands_id()` | `connect()` | 无 | 取第一个 ID |
| `HandCommand_t()` | `connect()` | 无 | 预分配命令 |
| `send_command()` | 动作发送、mode 切换 | hand ID、命令结构 | 动作路径只 warning；mode 路径抛异常 |
| `read_state()` | 状态读取、命令后缓存读取 | hand ID、`force_update` | 检查返回 error |
| `reset_sensor()` | 触觉复零 | hand ID、sensor ID | 失败重试并打印 warning |

SDK 还暴露但项目没有使用：

- `close_device()`；
- `get_sdk_version()`；
- `read_device_info()`；
- `get_serial_number()`；
- `get_hand_type()`；
- `get_hand_name()`；
- `read_version()`；
- `read_parameters()`；
- `read_firmware_state()`；
- calibration 和 action-group API。

设备身份、固件版本和校准状态因此没有进入启动校验或日志。

## 8.3 `ErrorStruct`

SDK 方法通常返回或附带：

```text
error_code: int
error_message: str
```

项目处理方式不统一：

- 连接错误：抛 `RuntimeError`；
- fresh read 错误：打印 warning，返回 `None`；
- send 错误：打印 warning，返回 `None`；
- mode 切换错误：抛 `RuntimeError`；
- reset 错误：有限重试，但最终不抛错；
- `read_state(False)` 错误：静默忽略。

## 8.4 `FingerCommand_t`

本机绑定暴露字段：

```text
id, kp, ki, kd, position, tor_max, mode, res0, res1, res2, res3
```

项目位置模式只写 `position`，其他控制参数在 connect 时设置一次。

虽然定义了 `MODE_FORCE = 5`，项目没有：

- CLI force mode；
- force target 构造；
- `res0/res1` 等力控字段编码；
- 针对不同序列号是否支持 force mode 的检查。

因此当前项目实际是位置模式驱动，不是完整的力控实现。

## 8.5 `FingerState_t`

本机绑定暴露：

```text
id
sensor_id
position
torque
raw_position
temperature
commboard_err
jonitboard_err
tipboard_err
default5
default6
default7
```

项目只使用：

- `position`；
- `torque`；
- `temperature & 0xFF`。

项目忽略：

- 关节 ID 和返回数组索引是否一致；
- sensor ID；
- raw position；
- 通信板、关节板和指尖板错误位；
- SDK 1.5 头文件中描述的 `default5/1000` 关节角速度。

这意味着 SDK 已经提供了一部分健康状态，但当前安全层没有消费。

## 8.6 `HandCommand_t` 与 `HandState_t` 长度

本机无硬件结构测试确认：

```text
len(HandCommand_t.finger_command) = 12
len(HandState_t.finger_state)     = 12
len(HandState_t.sensor_data)      = 5
len(sensor_data[i].raw_force)     = 120
```

这与项目的 NumPy 输出形状一致。

## 8.7 `read_state(force_update)` 的本机源码语义

本机 `/usr/local/xhand_controller/src/serial_communication.cpp` 显示：

```cpp
if (force_update) {
    send_command(device_id, hand_command_, error);
}
return hand_state_;
```

因此在这套 SDK 中：

- `read_state(id, True)` 会重新发送 SDK 内部保存的上一条命令；
- 串口后台线程异步接收并更新 `hand_state_`；
- `read_state(id, False)` 只返回当前缓存。

这与“True 是纯读取总线、False 是命令响应”这一简化描述不同。部署前应对目标 SDK 版本复核该语义，
并测量调用完成时返回状态相对发送命令的真实时间关系。

## 8.8 Python SDK 的直接调用顺序

按本机 pybind 绑定，最小调用关系如下：

```text
XHandControl()
  → enumerate_devices(protocol)
  → open_serial(port, baud) 或 open_ethercat(interface)
  → list_hands_id()
  → read_state()/reset_sensor()/send_command()
  → close_device()
```

下例展示接口形状和完整错误检查，不代表可以跳过身份校验、机械限位或现场风险评估。尤其在本机
RS485 SDK 中，`read_state(hand_id, True)` 不是严格只读操作，而会重发 SDK 内部保存的上一条
`HandCommand_t`。因此不能把它当作任意阶段都无副作用的“强制读”函数。

```python
import time

import numpy as np
from xhand_controller import xhand_control as xc


def require_ok(err, operation: str) -> None:
    if err.error_code != 0:
        raise RuntimeError(
            f"{operation} failed: code={err.error_code}, "
            f"message={err.error_message}"
        )


device = xc.XHandControl()
hand_id = None
command = None
try:
    ports = device.enumerate_devices("RS485") or []
    if not ports:
        raise RuntimeError("No RS485 XHand interface found")

    # 正式部署应使用配置中已核验的稳定端口，不应盲选 ports[0]。
    port = ports[0]
    require_ok(device.open_serial(port, 3_000_000), f"open_serial({port})")

    hand_ids = device.list_hands_id()
    if not hand_ids:
        raise RuntimeError("Port opened, but no XHand ID was reported")

    # 正式部署应与期望 hand ID、序列号和左右手类型比对。
    hand_id = hand_ids[0]

    # False 返回 SDK 后台接收线程维护的缓存；等待一帧 ID/数值完整的状态。
    # True 在本机 RS485 SDK 中还会重发上一命令，不能作为无副作用轮询。
    deadline = time.monotonic() + 2.0
    while True:
        read_err, state = device.read_state(hand_id, False)
        if read_err.error_code == 0:
            returned_ids = [state.finger_state[i].id for i in range(12)]
            measured = np.asarray(
                [state.finger_state[i].position for i in range(12)],
                dtype=np.float32,
            )
            if returned_ids == list(range(12)) and np.isfinite(measured).all():
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("No complete XHand state received within 2 seconds")
        time.sleep(0.01)

    # 复零目标是传感器总线 ID 0x11..0x15，不是关节 ID 2/5/7/9/11。
    # 调用前必须确认五指无接触；任一错误都应阻止进入 ACTIVE。
    for sensor_id in range(0x11, 0x16):
        require_ok(
            device.reset_sensor(hand_id, sensor_id),
            f"reset_sensor({sensor_id:#04x})",
        )

    command = xc.HandCommand_t()
    for joint_id in range(12):
        finger = command.finger_command[joint_id]
        finger.id = joint_id
        finger.kp = 100
        finger.ki = 0
        finger.kd = 0
        finger.tor_max = 300
        finger.mode = 3
        # 示例先保持实测位置；真实目标还必须经过 finite、限位和速率检查。
        finger.position = float(measured[joint_id])

    require_ok(device.send_command(hand_id, command), "send_command")
    cache_err, state_after_send = device.read_state(hand_id, False)
    require_ok(cache_err, "read_state(False) after send")
finally:
    # 此示例没有负载。实际抓持任务中，passive 可能导致物体掉落；应由故障状态机
    # 根据现场风险选择 HOLD/passive，并检查返回值，不能把所有异常静默吞掉。
    if hand_id is not None and command is not None:
        for joint_id in range(12):
            command.finger_command[joint_id].mode = 0
        passive_err = device.send_command(hand_id, command)
        if passive_err.error_code != 0:
            print(
                "passive command failed: "
                f"{passive_err.error_code} {passive_err.error_message}"
            )
    device.close_device()
```

EtherCAT 的接口差异只在发现和打开阶段：

```python
interfaces = device.enumerate_devices("EtherCAT") or []
if not interfaces:
    raise RuntimeError("No EtherCAT interface found")
require_ok(device.open_ethercat(interfaces[0]), "open_ethercat")
```

EtherCAT 通常需要 raw-socket capability；给 Python 解释器增加该 capability 属于系统权限变更，
应由部署环境明确管理。

## 8.9 项目 wrapper 的典型调用方式

项目通常不直接在主循环中操作 `HandCommand_t`，而是通过 `XHandRobot`：

```python
from mindex.robots.xhand_robot import XHandRobot

hand = XHandRobot(
    protocol="RS485",
    serial_port="/dev/serial/by-id/<verified-device>",
    baud_rate=3_000_000,
)
try:
    hand.connect()
    hand.reset_sensors()                 # 当前实现的 ID/验证问题见第 10 章
    observation = hand.get_observation(fresh=True)
    if observation is None:
        raise RuntimeError("XHand state unavailable")
    target = observation["joint_positions"].copy()
    hand.send_action(target)             # 当前实现不返回 SDK 是否接受
finally:
    hand.disconnect()                    # 当前实现未显式 close_device()
```

该片段只是准确展示当前封装层的用法；注释指出的三个限制必须修复后，才能把它作为可靠真机模板。

---

# 9. 状态读取、缓存与解析机制

## 9.1 `fresh=False`

若 `_cached_obs` 已存在，直接返回缓存，不调用 SDK。

若缓存为空，即使 `fresh=False`，函数仍会进入 SDK `read_state(hand_id, True)` 路径。

缓存通常由以下任一操作更新：

- fresh 读取；
- 成功发送动作后的 `read_state(False)`；
- 触觉 reset 阶段的 fresh 读取。

## 9.2 `fresh=True`

执行：

1. 检查连接；
2. 持有 `_lock`；
3. `read_state(hand_id, True)`；
4. 检查 SDK error；
5. `_parse_state()`；
6. 更新 `_cached_obs`。

错误时打印 warning 并返回 `None`，不会抛异常。

## 9.3 关节状态解析

对 12 个 `finger_state[i]`：

```text
joint_positions[i]    = fs.position
joint_torques[i]      = fs.torque
joint_temperatures[i] = fs.temperature & 0xFF
```

输出统一转换为 `float32`。

注意：

- 电流原始值被转换为浮点，但没有单位缩放；
- 温度高字节被丢弃；
- 错误位没有暴露给调用者；
- 没有检查返回值是否 finite；
- 没有检查位置是否处于合理机械范围。

## 9.4 指尖状态解析

对五个 `sensor_data[k]`：

```text
fingertip_force[k]        = [calc_force.fx, fy, fz]
fingertip_temperatures[k] = calc_temperature
xhand_tactile[k]          = [[raw_force[j].fx, fy, fz] for j in 0..119]
```

然后：

```text
fingertip_force -= _bias_ft
xhand_tactile   -= _bias_raw
```

原始 SDK 中 `fx/fy` 是有符号 8 位、`fz` 是无符号 8 位；转换为 float 并减 bias 后，
所有通道都可能为负数。这是软件去零的自然结果，不代表 SDK 原始 `fz` 本身有符号。

## 9.5 主循环读取失败时的行为

正式循环中：

1. `obs = hand.get_observation(fresh=True)`；
2. 若 `obs is None`，`_build_flat_state()` 返回 `None`；
3. `state_buf` 保留上一份状态；
4. 当前 tick 仍继续解析策略动作并尝试向 XHand 发送命令。

因此读取失败不会自动：

- 跳过当前动作；
- HOLD；
- passive；
- 停止 worker；
- 统计连续失败次数。

delta 模式还会继续使用 `state_buf[-1]` 的旧位置作为锚点。

---

# 10. 指尖触觉复零与软件偏置机制

## 10.1 当前算法

`reset_sensors()` 默认参数：

```text
sensor_ids       = (2,5,7,9,11)
max_retries      = 3
retry_wait_sec   = 0.2
verify           = True
verify_thresh_n  = 2.0
MAX_OUTER_ITERS  = 5
```

每轮：

1. 对待 reset ID 逐个调用 `reset_sensor(hand_id, sid)`；
2. 单个 ID 失败最多重试 3 次；
3. 清零已有软件 bias；
4. 做 5 次 fresh read；
5. 计算汇总力和原始触觉均值；
6. 将均值作为新的软件 bias；
7. 再 fresh read 一次；
8. 对减 bias 后的 `fingertip_force` 求模；
9. 若某指大于阈值，只重试对应项；
10. 最多 5 个 outer iteration。

## 10.2 关节 ID 与传感器 ID 混淆

厂商 C++ 测试明确写明：

```text
sensor id is [0x11, 0x15]
```

Python 厂商示例也使用：

```python
sensor_id = 17
reset_sensor(hand_id, sensor_id)
```

项目传入的是 `(2,5,7,9,11)`。这组值是带传感器的关节 ID，不是 SDK reset 命令目标 ID。

应拆分为两个常量：

```python
FINGERTIP_JOINT_IDS = (2, 5, 7, 9, 11)
FINGERTIP_SENSOR_IDS = (0x11, 0x12, 0x13, 0x14, 0x15)
```

这是接口参数层面的已确认不一致。具体设备对错误 ID 返回什么错误、是否误操作其他节点，需要真机确认；
但不应继续把当前调用当作可靠复零。

## 10.3 reset 失败不会阻止启动

如果一个 ID 的 3 次 reset 都失败：

- 代码打印 warning；
- 不记录该 ID 已彻底失败；
- 不抛异常；
- 继续软件 bias；
- bias 后验证可能返回“干净”；
- `run_policy.py` 继续连接机械臂并运动。

这使 SDK reset 失败与“复零完成”在上层无法区分。

## 10.4 静态接触被软件 bias 掩盖

当前验证使用的是已经减去刚刚学习 bias 的力：

```text
恒定接触力 F
采样均值 bias = F
验证观测 = F - bias = 0
```

无硬件 mock 测试使用五指恒定非零汇总力后，`reset_sensors()` 成功返回，
随后五指力模全部为 `0.0`。因此该验证不能发现静态接触。

它只能发现：

- bias 采集之后发生的接触变化；
- 采样期间高度不稳定的传感器；
- 软件 bias 后仍有显著波动的情况。

## 10.5 自定义 `sensor_ids` 的边界问题

函数允许传入 sensor ID 子集，但验证仍对五个 `sensor_data` 计算 `bad_idx`，然后执行：

```python
ids_to_reset = [sensor_ids[i] for i in bad_idx]
```

如果 `sensor_ids` 长度小于 5，而未选择的指尖仍被判定为 bad，可能发生索引越界或错误映射。

## 10.6 更可靠的复零定义

复零应区分三个概念：

1. **SDK reset 成功**：每个 `0x11..0x15` 返回 `error_code == 0`；
2. **传感器稳定**：连续采样的均值、方差和异常点满足要求；
3. **无外部接触**：不能仅靠“用当前值做 bias 后变成零”证明。

建议流程：

- 启动前显式提示并要求操作者确认五指悬空；
- 使用正确 sensor ID；
- 任一 reset 最终失败即终止真实硬件启动；
- reset 后先保存未经软件 bias 的原始帧；
- 检查噪声、饱和、通道范围和历史无载基线；
- 软件 bias 仅作为运行期补偿，不作为“无接触证明”；
- 将 bias、原始基线、SDK/固件版本写入 episode 元数据；
- 定义运行中重新复零的安全条件，禁止在接触状态自动复零。

---

# 11. 动作生成、转换与下发机制

## 11.1 动作来源

策略服务返回 action chunk。客户端根据 query mode、chunk index、延迟估计和 ensemble 配置，
每 tick 选出一个 `policy_out`。

XHand 最终收到：

```python
action[arm_dofs:]
```

联合系统中为最后 12 维；hand-only 中为整个 action。

## 11.2 Absolute 模式

`--hand-mode absolute`：

```text
hand_target = policy_output_hand
```

README 示例均使用 absolute 模式。该路径不使用 `state_now` 的手部切片，因此可以与 45 维富状态共同工作。

## 11.3 Delta 模式

`--hand-mode delta`：

```text
hand_target = measured_hand_position + policy_delta_hand
```

设计意图是每 tick 重新锚定最新测量状态，避免开环累积 delta。

但当前实现把完整 `state_buf[-1]` 传入 `to_absolute()`。当状态只有 18 维位置时可工作；
当状态为 45 维时：

```text
state_now[6:]   → 39 维
policy_out[6:]  → 12 维
```

已复现异常：

```text
ValueError: operands could not be broadcast together with shapes (39,) (12,)
```

正确实现应单独维护：

```text
joint_position_now: 18 维
policy_state_flat:  18/45/其他维度
```

delta anchor 只能使用前者。

## 11.4 Home 与首动作插值

`interpolate_to()` 对起点和终点执行 `np.linspace()`，每个 waypoint：

1. 先向 xArm 发送；
2. 再向 XHand 发送；
3. 休眠到目标周期。

这不是同步总线广播。xArm 和 XHand 命令之间存在顺序延迟；代码没有记录两次发送的精确时间戳。

## 11.5 `send_action()` 实现

驱动执行：

1. 检查连接；
2. `np.asarray(action, dtype=np.float32)`；
3. 检查 shape 是否为 `(12,)`；
4. 把 12 个值写入预分配命令对象的 `position`；
5. 持有 `_lock`；
6. `send_command(hand_id, command)`；
7. 成功后 `read_state(hand_id, False)`；
8. 若缓存读取成功，更新 `_cached_obs`。

没有检查：

- `np.isfinite(action)`；
- 每关节绝对范围；
- 与当前测量位置的差值；
- 单 tick 速度或加速度；
- 当前温度、电流、指尖力或错误位；
- SDK 是否实际执行到目标。

## 11.6 发送错误语义

`send_command` 错误时：

- 只打印到 stderr；
- 函数返回 `None`；
- 调用方无法区分成功和失败。

`read_state(False)` 错误时：

- 不更新 cache；
- 不打印；
- 不告知调用方。

与此同时，`run_policy.py` 在发送前已经把动作加入 `_emit_history`，并将其视为后续 inpaint 的执行历史。
因此“策略决定发送”与“SDK 接受”以及“硬件实际运动”三种状态被混为一谈。

## 11.7 Python 锁的实际覆盖范围

`send_action()` 在获取 `_lock` 之前修改 `_command.finger_command[i].position`。

当前项目只有主线程调用 XHand，因此不会产生现有竞争。但如果未来让多个线程调用 `send_action()`，
命令对象写入本身没有被锁保护。`reset_sensor()` 和 `_set_mode()` 也没有使用同一个 Python 锁。

更稳妥的驱动应让“构造命令快照 + SDK 调用”属于同一个临界区，或为每次发送创建不可变命令快照。

---

# 12. XHand 与 GR00T/πR² 策略的集成

## 12.1 状态组装

`_build_flat_state(hand_obs, arm_obs)` 按服务端返回的 `state_keys` 顺序拼接：

| state key | 数据来源 | 维度 |
|---|---|---:|
| `arm_joint_position` | `arm_obs["joint_positions"]` | 6 |
| `hand_joint_position` | `hand_obs["joint_positions"]` | 12 |
| `arm_joint_torque` | 代码查找 `arm_obs["joint_torques"]` | 6 |
| `hand_joint_torque` | `hand_obs["joint_torques"]` | 12 |
| `fingertip_force` | `hand_obs["fingertip_force"].reshape(-1)` | 15 |

典型组合：

```text
18D = 6 arm position + 12 hand position
45D = 6 arm position + 12 hand position + 12 hand feedback current + 15 fingertip force
```

`xhand_tactile` 的完整 1800 维原始阵列没有进入该状态构建表。

## 12.2 Arm torque 字段不一致

`XArmSDK.get_observation()` 输出：

```python
"joint_torque"
```

而状态构建查找：

```python
"joint_torques"
```

如果 checkpoint 请求 `arm_joint_torque`，`_build_flat_state()` 会持续返回 `None`。
warmup while 循环没有超时，可能无限等待 `state_buf` 填满。

45 维注释所示组合不包含 arm torque，因此默认富状态未必触发该问题；其他 schema 会触发。

## 12.3 历史缓冲

客户端从服务端 modality config 得到：

- `video_T`；
- `state_T`；
- `state_keys`。

随后维护两个 deque：

- `rgb_buf(maxlen=video_T)`；
- `state_buf(maxlen=state_T)`。

XHand fresh read 失败时不会追加状态，但旧历史仍被保留。异步 query worker 看到的可能是“最新图像 + 较旧状态历史”。

## 12.4 Warm-start

首次查询使用：

```text
rgb_hist   = stack(rgb_buf)
state_hist = stack(state_buf)
seed_streaming_from_obs(...)
```

得到 clean action chunk 后，机器人被 ramp 到首个将执行动作。

代码包含 ramp 后重新采集并 re-seed 的逻辑，但条件是：

```python
if _is_pir2_stream and False:
```

因此这段对齐逻辑实际关闭。该问题影响整个联合机器人，不只影响 XHand；对 XHand 来说，
模型的 streaming buffer 可能仍以 home 手型状态为条件，而机器人已被移动到首预测手型。

## 12.5 动作后处理的实际主路径

当前主路径是：

```text
policy_out
  → to_absolute()
  → action = raw_action
  → 写入 emit history
  → xArm.send_action()
  → XHand.send_action()
```

虽然代码库存在 max-delta 和 EMA 辅助函数，当前没有以下阶段：

```text
joint hard limit
max delta
velocity/acceleration limit
EMA
current/temperature/tactile supervisor
```

## 12.6 Hand-only 模式限制

`--no-arm` 将 `arm_dofs=0`。要正确运行还要求：

- 服务端 state keys 不请求 arm modality；
- action response 是 12 维手部动作；
- 可视化脚本不能继续按固定 6 维 arm offset 切片；
- 记录逻辑应单独保存手部电流。

当前 dry-run fake hand 能提供位置、电流和 `5×3` 指尖力，但不提供完整 raw tactile、温度或错误位。

---

# 13. 时序、线程与并发模型

## 13.1 控制频率

默认 `--rate-hz 25`，目标周期为 40 ms。README 说明该频率用于匹配采集频率和论文部署。

每 tick 主线程顺序：

```text
t0
  → XHand fresh read
  → xArm fresh read
  → camera get
  → 更新历史和 latest_obs
  → 处理 chunk/query swap
  → absolute/delta 转换
  → xArm send
  → XHand send
  → eval log 缓存
  → sleep 到 40 ms
```

这个顺序有意与注释所述 teleop 采集顺序一致：手 → 臂 → 图像。
但 teleop 代码不在仓库中，无法在当前提交内独立验证训练采集是否真的完全相同。

## 13.2 XHand 的周期占用

驱动注释估计 fresh RS485 read 约 10–15 ms。目标周期只有 40 ms，因此 XHand 读取可能占周期的显著部分。

还需叠加：

- xArm 位置和 torque RPC；
- camera ZMQ；
- NumPy stack/concat；
- action 后处理；
- XHand `send_command()`；
- 日志和终端输出。

代码只统计总体 loop Hz，没有独立记录：

- hand read latency；
- hand send latency；
- read-to-command age；
- SDK error 频率；
- 控制周期 p95/p99；
- deadline miss 次数。

## 13.3 每 tick 的打印

若周期还有剩余时间，代码每 tick 都打印：

```text
Sleeping for ... seconds
```

高频 stdout 会增加调度抖动、污染日志并影响 25 Hz 稳定性，应改成低频统计或 debug 开关。

## 13.4 Python 线程

根据参数，项目可能创建：

- continuous query worker；
- pipelined query worker；
- async VLM cache worker。

这些 worker：

- 读取主线程发布的 `latest_obs`；
- 调用远程 GR00T；
- 不直接调用 XHand。

所以当前 XHand Python API 访问仍由主线程串行完成。

## 13.5 SDK 内部线程

本机 RS485 SDK 源码会创建读取线程，持续：

- 等待串口可读；
- 读取字节；
- 解析帧；
- 更新内部 `hand_state_`。

项目的 `_lock` 只序列化 Python 对 SDK 的同步调用，SDK 内部仍是异步状态更新模型。
需要真机时间戳实验确定 `read_state(True)` 返回的是哪一帧以及是否可能仍是上一周期状态。

## 13.6 中断和 worker 退出

主循环开始后，SIGINT handler 只把 `stop=True`，让循环自然退出并进入 finally。

worker 是 daemon thread，finally 只设置 `worker_stop`，没有保存 thread handle 并 `join()`。
随后 client socket 被关闭。理论上 worker 可能仍处于远程调用或即将访问 socket；
虽然进程退出最终会终止 daemon thread，但关闭顺序并不完全确定。

这主要影响 GR00T 线程，但也关系到机器人关闭是否能在一个干净、可审计的系统状态下完成。

---

# 14. 数据记录与可视化链路

## 14.1 每 tick 记录语义

主循环先读取观测，再发送动作，然后把“发送前观测”和“本 tick 目标动作”交给 `EpisodeWriter`。

因此：

```text
observation/state[t] = 命令发送前测得的位置
action[t]            = 本 tick 计划下发的目标位置
action_delta[t]      = action[t] - observation/state[t]
```

`action` 不是实际达到位置，也不保证 SDK 接受成功。

## 14.2 XHand 数据保存

可保存：

- 手部关节位置，作为联合 `observation/state` 后 12 维；
- 手部反馈电流，作为联合 `observation/joint_torque` 后 12 维；
- `fingertip_force (5,3)`；
- `tactile (5,120,3)`；
- 最终目标动作后 12 维。

未保存：

- 关节温度；
- 指尖温度；
- 板级错误位；
- SDK read/send error code；
- 软件 bias；
- hand ID、序列号、左右手类型；
- SDK/固件版本；
- 手部 read/send 单独时间戳；
- 实际 command acknowledgment；
- 安全裁剪前后动作。

## 14.3 Hand-only 电流记录问题

当前代码只有在：

```text
arm_obs is not None
AND arm_obs contains joint_torque
AND hand obs contains joint_torques
```

时才构造联合 `joint_torque`。

结果：

- `--no-arm` 不记录 XHand 电流；
- xArm torque 不可用时也不记录 XHand 电流；
- XHand 电流被错误依赖于机械臂反馈能力。

## 14.4 可视化前提

`render_episode_modalities.py` 无条件读取：

- `observation/joint_torque`；
- `observation/fingertip_force`；
- `observation/tactile`；
- 18 维 state/action。

因此它不兼容：

- hand-only 12 维日志；
- 缺少 xArm torque 导致未保存联合 torque 的日志；
- dry-run 中没有 raw tactile 的日志；
- 部分字段从第一帧缺失的日志。

## 14.5 EpisodeWriter 的第一帧字段判定

Writer 是否创建可选 dataset，只检查 `_buffer[0]["obs"]`。如果第一帧缺某字段而后续帧出现，
该字段不会被保存；如果第一帧存在、后续帧缺失，`np.stack()` 会失败。

当前主循环大多保持字段一致，但通信抖动和条件式 torque 仍可能暴露这一设计问题。

---

# 15. 当前安全机制与缺失的安全层

## 15.1 已有保护

XHand 相关现有保护：

- `tor_max=300` 命令上限；
- 默认位置模式；
- home 和首动作线性 ramp；
- 首动作前两次人工确认；
- `--dry-run`；
- action shape 必须是 `(12,)`；
- 正常退出尝试进入 passive；
- SDK 调用错误会打印部分 warning；
- Python lock 避免当前驱动 API 的并发总线调用。

联合系统还有 xArm collision sensitivity、joint jerk 和 max acceleration，但这些不保护 XHand 自身。

## 15.2 缺失保护

当前没有：

- XHand 每关节机械硬限位；
- finite/NaN/Inf 检查；
- 每 tick 最大位置差；
- 关节速度限制；
- 关节加速度/jerk 限制；
- 实际位置跟踪误差阈值；
- 反馈电流阈值和持续时间判定；
- 关节温度阈值；
- 指尖温度阈值；
- 指尖力/触觉阈值；
- 板级错误位检查；
- 连续 read/send failure watchdog；
- 状态年龄检查；
- command acknowledgment；
- XHand 自碰撞或与环境碰撞检查；
- 独立硬件急停接口；
- 故障时 HOLD/passive/退出的明确状态机。

## 15.3 为什么 `tor_max` 不足以代替安全层

`tor_max` 只能限制单关节驱动电流字段，不能保证：

- 目标角在机械范围内；
- 多关节组合无自碰撞；
- 指尖接触力安全；
- 温度安全；
- 目标变化速度安全；
- SDK 通信正常；
- 机器人跟踪误差正常。

而且项目未验证 `tor_max=300` 是否适用于当前手型、固件、任务和对象。

## 15.4 论文观测值不能直接成为阈值

论文场景中的接触力或动作表现是特定设备、标定、任务和策略下的实验观测，
不能直接作为其他设备的安全阈值。尤其当前 SDK 的 `calc_force` 单位没有在本地厂商文档中得到完整确认。

---

# 16. 依赖、版本与可复现性

## 16.1 `pyproject.toml` 声明

当前 deployment 依赖只列出：

```text
numpy
pyzmq
omegaconf
tyro
```

可选依赖：

```text
pyrealsense2
lerobot
```

## 16.2 实际直接依赖但未声明

部署路径实际还依赖：

- `xhand_controller`；
- `xarm-python-sdk` 提供的 `xarm`；
- `msgpack`；
- `h5py`；
- 可视化需要 `matplotlib` 和系统 `ffmpeg`。

`run_policy.py` 在模块导入时直接导入 `EpisodeWriter`，而 `dataset.py` 又直接导入 `h5py`。
所以即使不使用 `--eval-log`，缺少 `h5py` 也会阻止主程序启动。

`omegaconf` 和 `tyro` 在当前 deployment Python 文件中没有直接使用。

## 16.3 厂商 wheel 约束

XHand SDK 是带本地 `.so` 的 pybind wheel，通常受以下约束：

- Python major/minor 版本；
- CPU 架构；
- glibc 和动态库；
- SDK 自带 `libxhand_control.so` 版本；
- 串口权限或 EtherCAT raw socket 权限。

项目 README 只给出了 `pip install -e .`，没有记录如何取得和安装匹配的 XHand wheel。

## 16.4 建议启动指纹

真实部署每次启动至少应记录：

```text
Python executable/version
xhand_controller distribution version
XHandControl.get_sdk_version()
XHand firmware version
hand serial number
hand type / hand ID
protocol / actual opened port / baud rate
PID / tor_max / mode
checkpoint ID and state/action schema
software commit
```

---

# 17. 已确认问题与风险分级

## 17.1 风险总表

| 级别 | 问题 | 证据 | 主要后果 |
|---|---|---|---|
| P0 | reset 使用关节 ID 而非 `0x11..0x15` 传感器 ID | 代码事实 + SDK 事实 | 硬件复零可能无效或目标错误 |
| P0 | 软件 bias 掩盖静态接触 | 已复现 | 接触状态被错误视为零载 |
| P0 | 缺少动作 finite/限位/max-delta/速度等安全包络 | 代码事实 | 模型异常输出可直接进入 SDK |
| P0 | send 失败仍写 executed history/inpaint/log | 代码事实 | 模型历史与真实机器人状态分叉 |
| P1 | 主循环前异常/取消不保证 passive | 代码事实 | 退出后手可能继续保持最后模式/目标 |
| P1 | `disconnect()` 不显式 `close_device()` | 代码事实 + SDK 事实 | 资源释放依赖析构和版本行为 |
| P1 | 45D state + hand-delta 广播失败 | 已复现 | 富状态 delta checkpoint 无法运行 |
| P1 | arm torque 单复数字段不一致 | 代码事实 | 某些 schema warmup 永久不满 |
| P1 | XHand read 失败仍继续发动作 | 代码事实 | 使用 stale state 闭环，且无 watchdog |
| P1 | hand-only 不记录反馈电流 | 代码事实 | 失去重要诊断和训练模态 |
| P1 | SDK 错误位、温度不参与保护 | 代码事实 | 已有健康信息未用于停机 |
| P2 | CLI 默认端口覆盖稳定 by-id 路径 | 代码事实 | USB 重排后连接错误设备 |
| P2 | 首个 hand ID 无身份校验 | 代码事实 | 多设备时可能选择错误手 |
| P2 | `interpolate_to` 未强制 fresh hand 起点 | 代码事实 | ramp 起点可能略旧 |
| P2 | 每 tick stdout 打印 | 代码事实 | 控制周期抖动和日志噪声 |
| P2 | 依赖与 SDK 版本未锁定 | 代码事实 + 环境事实 | 安装不可复现、语义漂移 |
| P2 | 无 XHand 测试 | 代码事实 | SDK 或控制改动缺少回归保障 |

## 17.2 P0 的定义

本文把 P0 定义为：在真实硬件上运行前必须处理，否则可能导致错误复零、无约束动作、
执行历史失真或安全监督失效的问题。

这不是在断言当前设备一定已经发生损坏，而是基于代码路径判断其不应继续作为默认真实部署行为。

## 17.3 读失败和发失败的组合风险

可能出现如下序列：

```text
fresh read 失败
  → state_buf 保留旧状态
  → 策略继续给出动作
  → emit_history 先记录动作
  → send_command 也失败
  → 上层仍认为动作已执行
  → 下一次 inpaint 使用未执行动作
```

这同时破坏：

- 机器人安全；
- πR²/RTC 的 clean-action conditioning；
- eval log 的真实性；
- 失败后的离线诊断。

---

# 18. 建议的目标架构与修复设计

## 18.1 驱动返回值

建议把 `send_action()` 改为返回结构化结果：

```python
@dataclass
class HandCommandResult:
    accepted: bool
    error_code: int
    error_message: str
    command_mono_ns: int
    cached_state: dict | None
```

主循环只有在 `accepted=True` 后才能：

- 更新 executed history；
- 写入“已发送动作”；
- 用于 inpaint；
- 增加成功 command sequence。

仍需区分“SDK 接受”与“硬件达到目标”；后者要靠后续 measured state 和 tracking error 判断。

## 18.2 观测结构

建议返回显式结构：

```text
joint_positions
joint_currents_ma
joint_temperatures_c
joint_velocities_rad_s
fingertip_force_raw_or_calibrated
tactile_raw
fingertip_temperatures_c
commboard_errors
jointboard_errors
tipboard_errors
sensor_ids
state_mono_ns
sdk_error
```

在单位未经确认前，不应把字段命名为 `_n`。

## 18.3 安全监督链

推荐动作顺序：

```text
policy output
  → 检查动作维度
  → finite 检查
  → delta-to-absolute，使用独立 joint-position anchor
  → 每关节 hard limit
  → 相对 measured state 的 max-delta
  → 速度/加速度/jerk 限制
  → 可选低通
  → 读取年龄检查
  → current/temperature/error/tactile supervisor
  → 生成 final target
  → SDK send
  → 检查 SDK acknowledgment
  → 写 executed-command history
  → 后续测量 tracking error
```

阈值必须来自：

- XHand 厂商机械和电气规格；
- 当前手型和固件；
- 低速无载/软物体实验；
- 明确的风险评估。

本文不凭空给出具体数值。

## 18.4 故障状态机

至少定义：

```text
DISCONNECTED
CONNECTING
READY_PASSIVE
HOMING
ACTIVE
HOLD
FAULT
STOPPING
```

示例转换：

- 单次 transient read fail → HOLD 当前已确认目标；
- 连续 N 次 read fail → FAULT；
- send fail → 不更新 executed history，并进入 HOLD/FAULT；
- 温度、电流、错误位超限 → FAULT；
- 用户终止 → STOPPING → passive → close；
- passive/close 失败 → 记录未确认关闭并提示物理急停。

具体 HOLD 或 passive 哪个更安全取决于是否抓持物体、物体掉落风险和任务现场，不能硬编码为通用结论。

## 18.5 连接身份校验

连接后读取：

- serial number；
- hand type；
- firmware version；
- calibration state；
- hand ID。

与配置文件中的期望值比较，不匹配即拒绝 ACTIVE。

## 18.6 正确关闭

把最外层资源获取放入一个从首次连接开始生效的 `try/finally` 或 context manager：

```text
create hand
try:
    connect
    verify identity
    reset sensors
    connect remaining components
    home/warmup/run
finally:
    stop workers and join
    request passive
    verify/passive best effort
    close_device
    clear local state
```

关闭过程中每一步都应记录结果，而不是完全吞掉异常。

## 18.7 复零接口

建议 API：

```python
reset_tactile(
    sensor_bus_ids=(0x11, 0x12, 0x13, 0x14, 0x15),
    require_operator_confirmation=True,
    fail_closed=True,
) -> TactileCalibrationReport
```

报告应包含：

- 每个 sensor ID 的 reset error；
- reset 前后原始统计；
- 软件 bias；
- 噪声和饱和情况；
- 时间戳；
- SDK/固件/序列号。

## 18.8 状态与策略解耦

不要用一个 flat vector 同时承担：

- 策略输入；
- delta anchor；
- 安全监督；
- 日志语义。

建议维护：

```text
RobotObservation        → 有名字、有单位、有时间戳
PolicyStateAdapter      → 按 state_keys 扁平化
ActionRepresentation   → absolute/delta 的模型输出
SafetySupervisor       → 产生 final absolute target
CommandRecord          → SDK 是否接受
```

这样 18D、45D 或未来更多 tactile modality 不会破坏 delta 动作逻辑。

---

# 19. 测试计划与验收标准

## 19.1 已完成的无硬件验证

本次审查已经执行：

1. 18 个 deployment Python 文件 AST 解析，全部通过；
2. 在 `real_robot` 环境中导入 XHand wrapper；
3. 用 SDK 原生 `HandState_t` 验证 `_parse_state()` 输出形状和 dtype；
4. 自省 `XHandControl` 方法签名和结构字段；
5. 对比两个本机 Python 环境的 distribution/底层 SDK 版本；
6. mock 复现 45D hand-delta 广播错误；
7. mock 复现恒定非零触觉被软件 bias 验证为零。

没有执行：

- `open_serial()`；
- `open_ethercat()`；
- 真实 `reset_sensor()`；
- 真实 `send_command()`；
- 真机频率和延迟测试。

## 19.2 驱动单元测试

应构造 fake `xhand_control`，覆盖：

### 连接

- RS485 首选端口成功；
- 首选失败、fallback 成功；
- 全部失败；
- open 成功但 hand ID 为空；
- 多 hand ID 配置选择；
- EtherCAT 无接口；
- identity 不匹配；
- 重复 connect/disconnect/reconnect。

### 命令

- 12 维合法动作；
- 错误 shape；
- NaN/Inf；
- 超关节范围；
- SDK send error；
- cache read error；
- passive 失败；
- close 仍执行。

### 状态

- 12 关节顺序；
- 负反馈电流符号；
- 温度低字节；
- 板级错误位；
- 5×120×3 tactile；
- SDK read error；
- stale age；
- 返回副本或不可变性。

### 触觉复零

- 使用 `0x11..0x15`；
- 任一 reset 最终失败时 fail closed；
- 无载稳定；
- 静态接触不能被软件 bias 当成验证成功；
- 传感器子集映射；
- 饱和、噪声和缺帧。

## 19.3 策略适配测试

至少覆盖：

- 18D state + absolute arm/hand；
- 18D state + hand delta；
- 45D state + absolute；
- 45D state + hand delta，确认使用独立 18D position anchor；
- hand-only 12D state/action；
- 请求 arm torque 的 schema；
- 服务端返回错误动作维度；
- query 失败时 HOLD；
- send 失败时不写 executed history；
- inpaint 只使用 acknowledged final actions。

## 19.4 离线时序测试

使用录制观测或 fake clock：

- 25 Hz 正常运行；
- XHand read 延迟 5/10/20/40 ms；
- 偶发和连续 read fail；
- 偶发 send fail；
- GR00T 延迟突增；
- 状态 stale；
- action chunk 边界跳变；
- 安全裁剪和 inpaint 一致性；
- Ctrl-C 发生在生命周期每个阶段。

## 19.5 真机验收指标

### 通信

- 记录实际打开端口、hand ID、序列号、SDK/固件版本；
- read/send error rate；
- 无无法解释的串口重连或多设备误选。

### 时序

- 控制周期 mean/p95/p99；
- XHand read/send latency mean/p95/p99；
- deadline miss 计数；
- 状态采集到命令发送的 age。

### 控制

- 所有 final targets finite 且在每关节范围；
- 单 tick delta、速度和加速度满足配置；
- 目标与实测 tracking error 在可接受范围；
- SDK 失败时 executed history 不增长。

### 健康与触觉

- 反馈电流、关节温度、指尖温度完整记录；
- 板级错误位为零或触发明确 fault；
- 复零报告可追溯；
- 无载和软接触下触觉响应稳定；
- 单位和坐标系经厂商资料或标定实验确认。

### 关闭

- 正常退出、异常退出和用户取消均尝试 passive；
- `close_device()` 执行；
- 关闭失败明确告警；
- 物理急停流程经过演练。

---

# 20. 真机部署检查清单

## 20.1 启动前

- [ ] 确认物理急停可达且有效；
- [ ] 确认工作空间无人、无硬障碍；
- [ ] 确认 XHand 安装、线缆和供电；
- [ ] 使用稳定 `/dev/serial/by-id/...`；
- [ ] 确认期待的序列号、左右手和 hand ID；
- [ ] 锁定 Python、wheel、底层 SDK 和固件版本；
- [ ] 确认 12 维关节顺序与 checkpoint 完全一致；
- [ ] 确认 absolute/delta 模式与训练一致；
- [ ] 确认每关节限位、max-delta 和健康阈值已配置；
- [ ] 五指完全无接触后再执行触觉复零；
- [ ] 先运行 dry-run 和离线 replay。

## 20.2 启动时

- [ ] 输出设备身份和版本指纹；
- [ ] 对 `0x11..0x15` 逐个检查 reset 返回值；
- [ ] 保存复零报告和软件 bias；
- [ ] 检查状态 finite、温度、电流和错误位；
- [ ] home 前人工确认；
- [ ] 使用低速、低风险参数；
- [ ] 首动作 ramp 前再次确认；
- [ ] 监控 hand read/send latency。

## 20.3 运行中

- [ ] 连续监控状态 age；
- [ ] 连续监控 SDK error；
- [ ] 监控电流、温度、指尖力和 tracking error；
- [ ] 记录 raw action、final action 和 acknowledged action；
- [ ] 只把 acknowledged final action 写入 inpaint history；
- [ ] deadline miss 或连续失败进入 HOLD/FAULT；
- [ ] 禁止接触状态自动复零。

## 20.4 退出后

- [ ] 请求 passive；
- [ ] 显式 close device；
- [ ] 保存完整 episode 和 fault reason；
- [ ] 检查是否存在未确认命令；
- [ ] 检查温度、电流和触觉峰值；
- [ ] 异常 episode 保留用于回放，不直接丢弃。

---

# 21. 分阶段实施路线

## 阶段 0：接口语义修复

目标：消除确定性接口错误。

- 分离 finger IDs 与 sensor IDs；
- 使用 `0x11..0x15`；
- reset 最终失败时阻止启动；
- `disconnect()` 调用 `close_device()`；
- 最外层 finally 覆盖完整生命周期；
- 启动日志记录 SDK/设备身份。

## 阶段 1：数据语义修复

目标：让状态、动作和记录可追溯。

- 把 `joint_torques` 重命名或明确标注为 current mA；
- 确认 fingertip force 单位和坐标；
- 独立 joint-position anchor；
- 修复 arm torque key；
- 修复 hand-only 电流日志；
- 保存温度、错误位、bias 和 command result。

## 阶段 2：安全监督

目标：模型输出不能直接越过安全层进入 SDK。

- finite、hard limit、max-delta；
- 速度/加速度限制；
- current/temperature/tactile/error supervisor；
- read/send/stale watchdog；
- HOLD/FAULT/STOPPING 状态机；
- acknowledged action history。

## 阶段 3：自动化测试

目标：无硬件验证所有重要失败路径。

- fake SDK；
- 驱动单测；
- 18D/45D/hand-only schema 测试；
- query/send/read 故障注入；
- 控制频率和退出时序测试；
- HDF5 schema 与可视化兼容测试。

## 阶段 4：递进真机验证

建议顺序：

1. XHand 单独、passive/read-only；
2. 单关节极小范围位置命令；
3. 全手无接触低速运动；
4. 软质物体低风险接触；
5. xArm + XHand 联合但无环境接触；
6. 静态任务；
7. 动态任务和正式 πR² 评估。

任一 SDK、固件、checkpoint、动作模式、关节顺序、阈值或通信协议变化，都应重新执行前序验证。

---

# 22. 最终判断

当前 XHand 代码的优势是：

- wrapper 简洁；
- SDK 调用路径清晰；
- 12 维位置状态和命令接口统一；
- 同时提供汇总触觉与原始触觉；
- fresh/cached 状态语义适合高频策略客户端；
- 与 GR00T 动态 modality 和 πR² 异步推理结构连接直接；
- home、首动作 ramp、dry-run 和正常 passive 退出为实验提供了基本保障。

但从可靠真机系统角度，当前实现仍存在四类根本缺口：

1. **接口语义缺口**：触觉 reset ID 错误、SDK 版本和 `read_state` 语义未固定；
2. **状态真实性缺口**：软件 bias 可掩盖接触、命令失败仍被当成执行、反馈量单位命名不精确；
3. **安全控制缺口**：缺少关节、速度、电流、温度、触觉和 stale 的统一 supervisor；
4. **工程闭环缺口**：早期退出清理不足、依赖未锁定、没有 XHand 自动化测试和真机验收记录。

因此最可靠的定位是：

> 当前实现是一套结构清楚、适合研究迭代的 XHand 位置控制与观测接入层，能够支撑
> xArm6 + XHand + GR00T/πR² 实验；但在修复触觉复零、命令确认、完整生命周期和软件安全包络之前，
> 不应把它视为已经完成安全闭环的生产部署驱动。

---

## 参考入口

- 项目 README：[`README.md`](README.md)
- XHand wrapper：[`deployment/mindex/robots/xhand_robot.py`](deployment/mindex/robots/xhand_robot.py)
- 部署主循环：[`deployment/apps/run_policy.py`](deployment/apps/run_policy.py)
- 动作辅助函数：[`deployment/mindex/policy/control_utils.py`](deployment/mindex/policy/control_utils.py)
- GR00T client：[`deployment/mindex/policy/groot_client.py`](deployment/mindex/policy/groot_client.py)
- HDF5 writer：[`deployment/mindex/recording/dataset.py`](deployment/mindex/recording/dataset.py)
- 官方论文：<https://arxiv.org/abs/2607.26055>
- 官方项目仓库：<https://github.com/pi-r2-flow/pi-r2-flow>
- 项目 GR00T fork：<https://github.com/pi-r2-flow/Isaac-GR00T/tree/pir2>
