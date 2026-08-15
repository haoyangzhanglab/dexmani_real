# UFACTORY xArm Mode 6 官方开源实现调研与工程使用指南

**面向：xArm7 关节遥操作、机器人学习数据采集与策略部署**  
**调研日期：2026-08-15**

---

# 1. 文档目标

本文回答四个核心问题：

1. UFACTORY 官方如何定义和使用 Mode 6；
2. UFACTORY 官方开源项目实际怎样调用 Mode 6；
3. `angle / speed / mvacc / wait / mode / state` 等参数究竟如何工作；
4. 对 xArm7 遥操作、数据采集和策略部署，应如何复用官方设计，同时避免复制其项目特定配置。

本文主要参考 UFACTORY 官方：

- `xArm-Python-SDK`
- `xarm_ros`
- `xarm_ros2`
- `xArm-CPLUS-SDK`
- `ufactory_teleop`
- `lerobot_robot_ufactory`
- UFACTORY 官方用户手册和机器人规格

其中，**`ufactory_teleop` 和 `lerobot_robot_ufactory` 对具身智能场景最有直接参考价值**：UFACTORY 已经实际使用 Mode 6 完成 GELLO → xArm 的 joint-space 遥操作，并将相同的 joint-target abstraction 用于机器人学习的数据采集和策略执行。

---

# 2. 首先明确：Mode 6 到底是什么

UFACTORY 将 Mode 6 定义为：

> **Joint Online Trajectory Planning Mode / Joint Online Planning Mode**

即：

**关节空间在线轨迹规划模式。**

其基本控制链不是：

```text
上位机产生完整 q(t)
        ↓
机器人逐点跟踪
```

而是：

```text
当前机器人状态
q_actual, qdot_actual
        +
新的绝对关节目标 q_target
        +
允许的 speed / acceleration
        ↓
xArm 控制器内部在线轨迹规划
        ↓
平滑地转向最新目标
```

当上一条运动还没有执行完成时，可以发送新的 joint target；控制器会中断旧目标对应的规划，并从当前运动状态重新规划至新目标。官方特别说明，Mode 6 重规划过程中会保持 joint velocity 和 acceleration 连续，但各关节的 velocity profile 不一定严格同步，而且最终达到的位置允许存在 small errors。Mode 6 要求固件至少为 v1.10.0。

因此，Mode 6 的核心语义应该理解成：

```text
“这是我现在最新想去的关节位置，
请你从当前动态状态平滑地重新规划过去。”
```

而不是：

```text
“这是轨迹中下一个必须严格跟踪的离散采样点。”
```

---

# 3. Mode 0 / Mode 1 / Mode 4 / Mode 6 / Mode 7 的边界

这是正确使用 Mode 6 的前提。

| Mode | 核心语义 | 典型输入 | 典型 API | 谁负责轨迹 |
|---|---|---|---|---|
| Mode 0 | 普通位置运动 | joint/TCP waypoint | `set_servo_angle()` 等 | xArm |
| Mode 1 | 外部高速 ServoJ | 高频 joint samples | `set_servo_angle_j()` | **上位机** |
| Mode 4 | Joint Velocity | `qdot` | joint velocity API | 外部速度控制 |
| **Mode 6** | **Joint Online Planning** | **最新 absolute `q_target`** | **`set_servo_angle()`** | **xArm 在线规划器** |
| Mode 7 | Cartesian Online Planning | 最新 TCP pose | Cartesian position API | xArm 在线规划器 |

UFACTORY 官方对 Mode 1 与 Mode 6 的区分非常明确：Mode 1 的 ServoJ 面向用户自己生成的高频平滑 trajectory；Mode 6 则面向不断变化的目标关节位置，用户不需要自行完成高频 trajectory generation。

因此：

```text
已有完整高频 q(t)
        → Mode 1

不断获得新的 q_target
        → Mode 6

直接控制 qdot
        → Mode 4

不断获得新的 TCP target
        → Mode 7
```

---

# 4. Mode 6 的标准 Python SDK 调用方式

最核心的 API 是：

```python
arm.set_servo_angle(
    angle=q_target,
    speed=...,
    mvacc=...,
    is_radian=True,
    wait=False,
)
```

**不是：**

```python
arm.set_servo_angle_j(...)
```

完整的初始化顺序通常是：

```python
from xarm.wrapper import XArmAPI

arm = XArmAPI(robot_ip)

arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)

arm.set_mode(6)
arm.set_state(0)

ret = arm.set_servo_angle(
    angle=q_target,
    speed=max_joint_speed,
    mvacc=max_joint_acc,
    is_radian=True,
    wait=False,
)
```

模式切换以后调用：

```python
arm.set_state(0)
```

很重要。UFACTORY 的状态机中，模式、TCP、payload、碰撞配置等关键参数发生变化后，机器人可能进入 `MODE_CHANGED` 等非运动状态，需要重新 `set_state(0)` 才能继续接收运动命令。

---

# 5. `angle`：Mode 6 的 action 是 absolute joint target

对 xArm7：

```python
q_target = [
    q1,
    q2,
    q3,
    q4,
    q5,
    q6,
    q7,
]
```

排列顺序就是：

```text
index 0 → J1
index 1 → J2
index 2 → J3
index 3 → J4
index 4 → J5
index 5 → J6
index 6 → J7
```

UFACTORY 官方 GELLO 和 LeRobot 实现也是把遥操作器或 policy 产生的：

```text
J1.pos ... J7.pos
```

组合成 absolute joint target，然后调用 `set_servo_angle()`。

Mode 6 正常使用的是：

```python
relative=False
```

即：

\[
a_t=q^{target}_t
\]

不是：

\[
a_t=\Delta q_t
\]

也不是：

\[
a_t=\dot q_t
\]

---

# 6. xArm7 的关节范围

以下以当前 UFACTORY 官方规格页为准：

| Joint | 官方机械范围 |
|---|---:|
| J1 | -360° ～ +360° |
| J2 | -117° ～ +116° |
| J3 | -360° ～ +360° |
| J4 | -6° ～ +225° |
| J5 | -360° ～ +360° |
| J6 | -97° ～ +180° |
| J7 | -360° ～ +360° |

当前官方规格同时给出 joint speed 上限约 `180°/s`、joint acceleration 上限约 `1145°/s²`、joint jerk 上限约 `28647°/s³`。

因此，绝不能对 xArm7 简单做：

```python
q = np.clip(q, -np.pi, np.pi)
```

因为 J2/J4/J6 是明显的非对称范围，而 J1/J3/J5/J7 又具有更大的机械转角范围。

机器人学习系统还应避免无意义地对连续关节数据做：

```python
q = q % (2 * np.pi)
```

否则可能制造：

```text
+179°
   ↓
-179°
```

这种数值上的巨大 action jump。

---

# 7. `speed`：最大关节速度约束，不是速度命令

Mode 6：

```python
speed=...
```

表示在线 trajectory planner 使用的**最大关节速度约束**。

例如：

```python
speed = math.radians(90)
```

意味着：

```text
max joint velocity ≈ 90°/s
```

它不是：

```text
qdot_command
```

也不是：

```python
[J1_speed, J2_speed, ..., J7_speed]
```

API 接收的是一个 scalar。

各关节实际运动速度仍由 xArm 内部在线规划器计算，因此：

```text
J1 实际速度
J2 实际速度
...
J7 实际速度
```

通常不同，而且不要求严格同步达到相同速度峰值。官方 Mode 6 文档明确说明各关节 velocity profiles 可能不同步。

---

# 8. `mvacc`：最大关节加速度约束

同理：

```python
mvacc=...
```

表示：

> 在线关节轨迹规划允许使用的最大 acceleration。

例如：

```python
mvacc = math.radians(500)
```

表示：

```text
500°/s²
≈ 8.727 rad/s²
```

它不是 policy 输出的：

```text
qddot
```

而是 controller-side trajectory constraint。

---

# 9. Python SDK 对 `speed` / `mvacc` 的真实处理

当前 Python SDK 的 `set_servo_angle()` 内部会取得：

```python
speed
mvacc
```

然后约束到：

```text
speed <= π rad/s
mvacc <= 20 rad/s²
```

即约：

```text
speed <= 180°/s
mvacc <= 1145.9°/s²
```

SDK 内部最低值分别约为：

```text
joint speed >= 0.0001 rad/s
joint acc   >= 0.01 rad/s²
```

并且 SDK 初始化时保存：

```text
last_joint_speed = 20°/s
last_joint_acc   = 500°/s²
```

如果调用：

```python
arm.set_servo_angle(
    angle=q_target,
    speed=...,
    mvacc=None,
)
```

则 `mvacc` 会继续使用 `_last_joint_acc`；类似地，`speed=None` 会使用 `_last_joint_speed`。

这意味着：

> **`mvacc=None` 并不是一个固定的 Mode 6 默认加速度，而是一个具有历史状态的 SDK 参数。**

因此对于需要：

- 可复现实验；
- 数据采集/部署一致性；
- 可审计配置；

的机器人学习系统，建议显式传入：

```python
speed=...
mvacc=...
```

而不是依赖 SDK 上一次运动留下的状态。

---

# 10. `wait=False` 是 Mode 6 正常 streaming 的关键

正常 Mode 6：

```python
wait=False
```

控制逻辑为：

```text
target A
   ↓
正在向 A 运动
   ↓
target B 到来
   ↓
不用等 A 完成
   ↓
从当前 q / qdot 重规划至 B
```

如果使用：

```python
wait=True
```

则上位机将阻塞等待当前 motion 完成，在线更新能力基本被破坏。

UFACTORY ROS 官方文档明确要求 Mode 6 动态 transition 时 `/xarm/wait_for_finish=false`；对应 SDK 就是 `set_servo_angle(..., wait=False)`。

---

# 11. `radius`、`mvtime`、`relative`

对典型 Mode 6 joint-target streaming：

```python
radius=None
relative=False
```

即可。

`radius` 主要涉及传统 joint trajectory 的 blending；当前 UFACTORY Mode 6 官方示例和遥操作实现都没有依赖 `radius` 实现在线跟踪。

`mvtime` 在当前普通关节运动接口中也不是 Mode 6 trajectory duration 的控制手段，因此不要设计：

```python
mvtime = 1.0 / policy_hz
```

来控制每帧动作执行时间。

Mode 6 的时间响应由：

```text
当前机器人状态
+
最新 q_target
+
speed
+
mvacc
```

共同决定。

---

# 12. 官方项目一：`xArm-Python-SDK`

UFACTORY 提供了专门的官方示例：

```text
example/wrapper/common/
2006-joint_online_trajectory_planning.py
```

其初始化过程大致是：

```python
arm.motion_enable(enable=True)

arm.set_mode(0)
arm.set_state(state=0)

arm.move_gohome(wait=True)

arm.set_servo_angle(
    angle=[-50, 0, 0, 0, 0, 0, 0],
    wait=True,
)

arm.set_mode(6)
arm.set_state(0)
```

随后：

```python
speed = 50

arm.set_servo_angle(
    angle=[120, 0, 0, 0, 0, 0, 0],
    speed=speed,
    wait=False,
)

time.sleep(1)

arm.set_servo_angle(
    angle=[-120, 0, 0, 0, 0, 0, 0],
    speed=speed,
    wait=False,
)
```

官方注释明确说明：当前 command 尚未执行完成时，下一个 command 可以将其 interrupt。

这里有两个重要结论。

第一，UFACTORY 并没有要求 Mode 6 像 ServoJ 那样以 100–250 Hz 提供非常密集的小位置增量。

官方示例甚至：

```text
+120°
    ↓ 1 s
-120°
```

通过不断改变最终目标来演示在线重新规划。

第二，该示例：

```python
speed = 50
```

但没有显式传：

```python
mvacc
```

因此，在没有其他关节运动改变 SDK 状态的情况下，它通常继续使用 SDK 初始：

```text
500°/s²
```

的 `_last_joint_acc`。

---

# 13. 官方项目二：`xarm_ros`

`xarm_ros` 对 Mode 6 的官方解释和参数展示最完整。

官方将其称为：

```text
Online Target Update
```

并明确指出这是：

> Servo Joint / Servo Cartesian 之外，不要求用户侧自行执行高频 trajectory planning 的方法。

Mode 6 Joint OTG 示例：

```bash
rosservice call /xarm/set_mode 6
rosservice call /xarm/set_state 0
```

第一目标：

```text
q_target:
[0, 0, 0, 0, 0, 0, 0]

speed:
0.2 rad/s

acc:
7 rad/s²
```

在尚未到达第一目标时，再发第二目标：

```text
q_target:
[-0.2775, -0.55, -0.452, 1.05, -0.23, 1.55, -0.665]

speed:
0.35 rad/s

acc:
10 rad/s²
```

也就是说官方示例同时在线改变：

```text
target
speed
acceleration
```

控制器重新规划至最新目标。

换算后大约为：

| 命令 | speed | acceleration |
|---|---:|---:|
| Target A | 11.46°/s | 401°/s² |
| Target B | 20.05°/s | 573°/s² |

这进一步证明：

> `speed` 和 `mvacc` 是每次 Mode 6 在线规划可以动态更新的 trajectory constraints。

---

# 14. `xarm_ros` 同时清楚展示了 Mode 1 与 Mode 6 的不同

官方 ServoJ 示例会先读取当前关节位置，然后逐步增加很小的 joint displacement，例如：

```text
J7:
0.24
0.25
0.26
...
```

并提醒不能突然发送距离当前关节位置非常远的 ServoJ target。原因是 Mode 1 假设**外部系统已经负责 trajectory generation**。

因此官方代码所表达的控制边界可以概括为：

```text
Mode 1:
external trajectory
        ↓
dense q samples
        ↓
ServoJ

Mode 6:
latest desired q
        ↓
xArm OTG
        ↓
robot
```

---

# 15. 官方项目三：`ufactory_teleop`

这是目前对 xArm7 具身智能遥操作最值得参考的 UFACTORY 官方项目之一。

UFACTORY Teleoperation System 面向：

```text
robot-learning-quality demonstrations
```

并支持多种 teleoperator。

其中：

```text
GELLO
→ Joint Space
→ robot_mode: 6
```

而 Pika / UMI 等 Cartesian teleoperator 使用 Cartesian mode。

---

# 16. `ufactory_teleop` 的 xArm7 + GELLO 官方参数

官方 xArm7 GELLO 配置直接给出了：

```yaml
RobotConfig:
  robot_ip: "..."
  robot_mode: 6
  robot_speed: 90
  robot_acc: 500

  start_joints:
    [0, 0, 0, 1.5708, 0, 1.5708, 0]
```

其中：

```text
robot_speed = 90°/s
robot_acc   = 500°/s²

start_joints:
rad
```

GELLO teleoperation loop 默认：

```text
fps = 30
```

即官方实际组合大致是：

```text
xArm7
Mode 6
30 Hz outer control loop
90°/s max joint velocity
500°/s² max joint acceleration
absolute q_target
```



这比仅看 Python SDK 示例更有工程参考意义，因为它是 UFACTORY 专门为机器人遥操作设计的实际系统。

---

# 17. `ufactory_teleop` 的 Mode 6 初始化流程

它并不是程序一启动就直接高速 Mode 6。

连接机器人后，代码先做：

```text
XArmAPI(...)
       ↓
motion_enable()
       ↓
clean_error / clean_warn
       ↓
Mode 0
       ↓
State 0
       ↓
移动至 start_joints
       ↓
再进入配置的 robot_mode
```

对于 GELLO：

```text
robot_mode = 6
```

随后：

```python
set_mode(6)
set_state(0)
```

这体现了一个很重要的官方设计原则：

> **确定的初始化运动与动态在线跟踪分离。**

即：

```text
Mode 0
→ initial alignment

Mode 6
→ online teleoperation
```



---

# 18. `ufactory_teleop` 的首次同步机制

官方代码进一步做了启动缓冲。

对于 Mode 6：

```python
if command_count < 20:
    jnt_spd = 0.2
else:
    jnt_spd = configured_joint_speed
```

其中：

```text
0.2 rad/s
≈ 11.46°/s
```

而正式 xArm7 GELLO 配置为：

```text
90°/s
```

所以实际上：

```text
前 20 条命令：
11.46°/s

之后：
90°/s
```



这可以显著降低：

```text
leader 初始姿态
       ≠
follower 初始姿态
```

时第一批命令造成的突然运动风险。

---

# 19. 第一条 command 还有额外特殊处理

`ufactory_teleop` 中：

```text
first command
    ↓
wait=True
```

如果当前不是 Mode 0，则先：

```python
set_mode(0)
set_state(0)
```

然后执行 blocking joint positioning。

随后的 command：

```text
wait=False
```

如果当前不是 Mode 6，则：

```python
set_mode(6)
set_state(0)
```

再进入正常 online streaming。

因此完整启动过程更准确地表示为：

```text
Connect
   ↓
Mode 0
   ↓
move to start_joints
   ↓
first teleop target
   ↓
Mode 0 + slow + wait=True
   ↓
Mode 6
   ↓
前若干命令继续低速
   ↓
Mode 6 normal speed
```

这是一个非常值得复用的控制结构。

---

# 20. `ufactory_teleop` 明确传入了 `mvacc`

这是相较早期 LeRobot wrapper 很重要的区别。

当前官方 teleop 代码实际调用：

```python
self.real_arm.set_servo_angle(
    angle=robot_action,
    speed=jnt_spd,
    mvacc=self._joint_acc,
    is_radian=True,
    wait=wait_,
)
```

因此：

```text
speed
acceleration
```

都显式来自配置，而不是依赖 SDK 上一次运动的隐藏状态。

对于可复现的机器人学习系统，这种实现更值得借鉴。

---

# 21. 官方项目四：`lerobot_robot_ufactory`

UFACTORY 的 LeRobot integration 同样支持：

```text
GELLO teleoperation
data collection
policy training
policy evaluation / inference
```

其中 joint-space control 明确执行：

```python
if control_space == "joint":
    self.real_arm.set_mode(6)

self.real_arm.set_state(0)
```



因此 UFACTORY 已经实际采用：

```text
Joint Teleoperation
       ↓
Mode 6

Joint Policy Deployment
       ↓
same UFRobot backend
       ↓
Mode 6
```

而不是只把 Mode 6 当作 SDK 演示功能。

---

# 22. LeRobot 的 observation 和 action 语义

真实 joint observation 来自：

```python
code, states = self.real_arm.get_joint_states(
    is_radian=True,
    num=3,
)
```

然后得到：

```text
J1.pos ... J7.pos
```

可选：

```text
J1.vel ... J7.vel
```



joint action 则为：

```text
J1.pos
J2.pos
...
J7.pos
```

即 policy 输出的是：

\[
q^{target}
\]

而不是机器人当前实际位置，也不是 velocity。

这是非常合理的机器人学习数据接口：

```text
observation:
camera
q_actual
optional qdot_actual

action:
q_target
```

---

# 23. LeRobot 的 Mode 6 参数

其默认 joint velocity 配置为：

```text
max_joint_velocity = 90°/s
```

前 20 条 command 同样使用：

```text
0.2 rad/s
≈ 11.46°/s
```

而 xArm7 GELLO 数据采集配置默认：

```text
dataset fps = 30
```



因此与 `ufactory_teleop` 的整体设计高度一致：

```text
30 Hz
absolute joint target
Mode 6
initial slow synchronization
90°/s normal max velocity
```

---

# 24. LeRobot 与 `ufactory_teleop` 最大的实现差异

LeRobot 当前调用类似：

```python
self.real_arm.set_servo_angle(
    angle=q_target,
    speed=jnt_spd,
    is_radian=True,
    wait=wait_,
)
```

它没有显式传：

```python
mvacc
```



结合 Python SDK：

```text
mvacc=None
→ reuse _last_joint_acc
```

以及 SDK 初始：

```text
_last_joint_acc = 500°/s²
```

可以推断，在没有其它 motion 修改 acceleration 的普通启动情况下，其行为通常接近：

```text
90°/s
500°/s²
```

但这是**状态依赖行为**，而不是配置中明确表达的约束。

相比之下，新的 `ufactory_teleop`：

```python
speed=self._joint_speed
mvacc=self._joint_acc
```

更明确、更适合作为新的工程实现参考。

---

# 25. 官方项目五：`xarm_ros2`

当前 `xarm_ros2` 仍提供：

```text
set_mode
set_state
set_servo_angle
set_servo_angle_j
```

等接口，因此完全可以：

```text
set_mode(6)
set_state(0)
set_servo_angle(... wait=false)
```

执行 Mode 6。

但在当前主 README 和典型示例入口中，没有看到像：

```text
xarm_ros Online Target Update
```

或：

```text
Python SDK 2006-joint_online...
```

那样独立完整的 Mode 6 demo。

同时 ROS2 的：

```text
MoveIt
ros2_control
JointTrajectoryController
```

仍主要属于：

```text
external trajectory execution
```

这与 Mode 1 的抽象更接近，而不是统一改成 Mode 6。

所以不应该理解成：

> Mode 6 是所有实时运动的替代品。

UFACTORY 实际上仍然是按**控制输入的抽象层级**选择模式。

---

# 26. 官方项目六：`xArm-CPLUS-SDK`

C++ SDK 同样定义：

```text
mode 6:
joint online trajectory planning

mode 7:
Cartesian online trajectory planning
```

并提供对应：

```text
set_mode()
set_state()
set_servo_angle()
```

API。

因此 Python/C++/ROS 的底层控制模型是一致的。

真正最有工程信息量的 Mode 6 实例，仍然是：

```text
Python SDK
xarm_ros
ufactory_teleop
lerobot_robot_ufactory
```

---

# 27. UFACTORY 官方 Mode 6 项目横向对比

| 项目 | 用途 | Mode 6 输入 | 频率 | speed | mvacc | `wait` |
|---|---|---|---:|---:|---:|---|
| Python SDK `2006` | 最小 OTG Demo | absolute q | 示例约 1 s 更新 | 50°/s | 未传，通常继承 500°/s² | False |
| `xarm_ros` OTG | Online Target Update | absolute q | 非 ServoJ 高频模式 | 0.2→0.35 rad/s | 7→10 rad/s² | False |
| `ufactory_teleop` GELLO+xArm7 | 正式遥操作 | absolute q | **30 Hz** | **初始 0.2 rad/s，随后 90°/s** | **500°/s²** | 首次特殊，其后 False |
| `lerobot_robot_ufactory` | Teleop/Record/Policy | `J1.pos...J7.pos` | **30 Hz 数据采集配置** | **初始 0.2 rad/s，随后默认 90°/s** | 未显式传 | 首次特殊，其后 False |

以上数值分别来自对应官方项目代码和配置。

这里最重要的不是找到一个“唯一正确的 Mode 6 speed”。

而是理解：

```text
Mode 6 本身
≠ 固定 90°/s
≠ 固定 500°/s²
≠ 固定 30 Hz
```

这些是具体应用配置。

---

# 28. 从官方项目提炼出的标准 Mode 6 架构

UFACTORY 各项目共同体现了下面这套模式：

```text
                  INITIALIZATION
                        │
                  connect robot
                        │
                 clear error/warn
                        │
                  motion_enable
                        │
                     Mode 0
                        │
                    State 0
                        │
               move to start pose
                        │
                initial alignment
                        │
                     Mode 6
                        │
                    State 0
                        ▼

         ┌─────────────────────────┐
         │      CONTROL LOOP       │
         │                         │
         │ actual robot state      │
         │        ↓                │
         │ teleop / policy         │
         │        ↓                │
         │ absolute q_target       │
         │        ↓                │
         │ set_servo_angle(        │
         │   angle=q_target,       │
         │   speed=vmax,           │
         │   mvacc=amax,           │
         │   wait=False            │
         │ )                       │
         │        ↓                │
         │ xArm internal OTG       │
         └─────────────────────────┘
```

---

# 29. 遥操作数据采集的正确数据语义

对于 GELLO 等 joint-space teleoperator：

```text
Leader joints
      ↓
mapping / offset
      ↓
q_cmd
      ↓
Mode 6
      ↓
xArm actual motion
```

数据录制时必须区分：

```text
q_cmd
```

和：

```text
q_actual
```

因为 Mode 6 具有速度、加速度约束，动态运动过程中通常：

\[
q^{cmd}_t \neq q^{actual}_t
\]

因此推荐：

```text
observation:
    image
    q_actual
    optional qdot_actual

expert action:
    q_cmd
```

而不是：

```text
action = q_actual
```

否则训练阶段和部署阶段 action semantics 会发生改变。

UCTORY LeRobot 本身也通过真实 `get_joint_states()` 获取 observation，而 action 是独立的 joint position target。

---

# 30. 推荐采集的数据字段

原始数据至少保留：

```python
{
    "timestamp": ...,

    "joint_pos_actual": q_actual,
    "joint_vel_actual": qdot_actual,

    "joint_pos_command": q_cmd,

    "camera_timestamp": ...,
    "camera": ...,

    "robot_mode": ...,
    "robot_state": ...,
    "error_code": ...,
}
```

并建议额外计算：

\[
e_q=q_{cmd}-q_{actual}
\]

用于诊断：

```text
mean |e|
P50 |e|
P95 |e|
P99 |e|
max |e|
```

这是工程增强建议，而非 UFACTORY 数据格式的强制要求。

---

# 31. 策略部署应尽量复用相同的 control semantics

理想数据采集：

```text
human
  ↓
q_target
  ↓
Mode 6
  ↓
robot
```

训练：

```text
image + q_actual
       ↓
     policy
       ↓
     q_target
```

部署：

```text
policy
  ↓
q_target
  ↓
same safety layer
  ↓
same Mode 6
  ↓
robot
```

这种结构使：

```text
collection dynamics
≈
deployment dynamics
```

避免：

```text
采集 Mode 6
部署 Mode 1
```

导致执行 dynamics、平滑特性和 tracking lag 发生不必要改变。

UFACTORY 的 LeRobot integration 正是通过同一个 `UFRobot` backend 同时支持 teleoperation、record 和 policy evaluation。

---

# 32. 推荐的 Mode 6 最小实现

下面是最接近官方控制语义的实现：

```python
import math
from xarm.wrapper import XArmAPI

arm = XArmAPI(robot_ip)

# ----- initialization -----

arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)

arm.set_mode(6)
arm.set_state(0)

# ----- online update -----

q_target = [...]  # 7 joints, rad

ret = arm.set_servo_angle(
    angle=q_target,
    speed=math.radians(90.0),
    mvacc=math.radians(500.0),
    is_radian=True,
    wait=False,
)

if ret != 0:
    print("xArm command failed:", ret)
```

其中：

```text
90°/s
500°/s²
```

对应的是官方 `ufactory_teleop` xArm7 + GELLO 示例参数，**并不代表所有任务都应该使用这组参数**。

---

# 33. 更完整的启动方式

更值得实际机器人系统参考的是：

```python
import math
import time
from xarm.wrapper import XArmAPI

arm = XArmAPI(robot_ip)

# -----------------------------
# 1. Robot initialization
# -----------------------------

arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)

arm.set_mode(0)
arm.set_state(0)

# -----------------------------
# 2. Safe initial alignment
# -----------------------------

q_start = [
    0.0,
    0.0,
    0.0,
    math.pi / 2,
    0.0,
    math.pi / 2,
    0.0,
]

ret = arm.set_servo_angle(
    angle=q_start,
    speed=math.radians(20),
    mvacc=math.radians(100),
    is_radian=True,
    wait=True,
)

if ret != 0:
    raise RuntimeError(ret)

# -----------------------------
# 3. Enter Mode 6
# -----------------------------

arm.set_mode(6)
arm.set_state(0)

# -----------------------------
# 4. Online control
# -----------------------------

while running:
    q_target = get_teleop_or_policy_target()

    ret = arm.set_servo_angle(
        angle=q_target,
        speed=math.radians(MAX_SPEED_DEG),
        mvacc=math.radians(MAX_ACC_DEG),
        is_radian=True,
        wait=False,
    )

    if ret != 0:
        break

    time.sleep(1.0 / CONTROL_HZ)
```

这里的“Mode 0 初始定位 → Mode 6 streaming”结构直接来自官方项目思想；具体较低的初始化速度则属于工程安全策略，应根据真实机械臂环境确定。

---

# 34. 官方能力范围与官方项目使用值必须区分

## xArm7 官方能力

```text
joint speed:
0 ~ 180°/s

joint acceleration:
0 ~ 约 1145°/s²
```



## 官方项目实际示例

```text
Python SDK:
50°/s
mvacc implicit

ROS Mode 6:
11.5 → 20.1°/s
401 → 573°/s²

UFACTORY Teleop xArm7:
90°/s
500°/s²
30 Hz

UFACTORY LeRobot xArm7:
initial 11.46°/s
then default 90°/s
30 Hz dataset config
mvacc implicit
```



因此：

> **“机械臂允许 180°/s”不能推导出“机器人学习应该使用 180°/s”。**

同样：

> **“官方 GELLO 示例用 90°/s”也不能推导出“90°/s 是所有近人遥操作的安全速度”。**

安全参数必须结合：

- payload；
- TCP；
- mounting direction；
- 工作空间；
- 人员距离；
- task geometry；
- collision configuration；
- emergency stop；
- 实际 risk assessment。

---

# 35. Mode 6 的控制频率没有一个官方唯一值

官方代码本身已经证明这一点：

```text
Python OTG demo:
约秒级修改目标

UFACTORY GELLO:
30 Hz

LeRobot:
典型数据采集 30 Hz
```



这与 Mode 1 完全不同。

Mode 6 的 outer-loop frequency 应根据：

```text
teleoperator frequency
policy frequency
camera frequency
required responsiveness
network latency
target smoothness
```

综合确定。

因此：

```text
30 Hz
```

是 UFACTORY 当前机器人学习示例中一个非常重要的官方参考值，但不是 Mode 6 protocol 的固定频率。

---

# 36. 不要把 Mode 6 当成 collision-free planner

Mode 6 的“online planning”解决的是：

```text
current q / qdot
+
q_target
+
speed / acceleration constraints
```

之间的 trajectory generation。

它并不等于：

```text
MoveIt
OMPL
environment-aware motion planning
```

Mode 6 自身不知道桌面、相机、人、物体等障碍物的位置。

因此上层仍必须负责：

```text
joint bounds
workspace limits
self/environment collision strategy
action validation
timeout
emergency stop
```

---

# 37. 重要安全注意：不要机械复制 `ufactory_teleop` 的所有底层设置

当前 `ufactory_teleop` robot wrapper 中还存在诸如：

```python
set_collision_sensitivity(0)
```

以及其它项目级底层运动配置。

但是 UFACTORY Studio 当前面向用户的安全设置文档给出的 Collision Detection Sensitivity 正常配置范围为 `1~5`，数值越大越敏感，并明确指出常规情况下不建议设置得过低；正确的 TCP payload 和 mounting direction 同样是碰撞检测与重力补偿正常工作的基础。

因此：

> `ufactory_teleop` 中的 `set_collision_sensitivity(0)` 应视为该项目的特定系统选择，**不应无条件复制到自己的真实机器人安全组件中。**

在自己系统中应根据：

```text
真实 payload
安装方向
工具
机械臂速度
工作环境
风险评估
```

重新配置碰撞检测。

---

# 38. 建议使用 Reduced Mode / Safety Boundary 作为第二层保护

UFACTORY 官方提供：

```text
Reduced Mode
Safety Boundary
Joint Range
Reduced Joint Speed
TCP workspace limits
Self-Collision Detection
```

等独立安全能力。

对于 policy deployment，合理的防护层级应该是：

```text
Policy
  ↓
software safety filter
  ↓
Mode 6 command
  ↓
Reduced Mode / robot-side limits
  ↓
servo controller
  ↓
physical E-stop
```

而不是只依赖：

```python
speed=...
```

保证安全。

---

# 39. 错误状态必须进入控制闭环

UFACTORY 官方错误文档包含：

```text
joint angle exceeds limit
joint speed exceeds limit
planning error
abnormal servo current
servo over-voltage / under-voltage
```

等异常，其中多类问题的官方排查建议包括：

```text
降低 speed
降低 acceleration
检查 payload
检查 mounting direction
检查 collision
重新规划
```



因此部署代码至少应该持续监控：

```text
return code
arm.error_code
arm.state
arm.mode
```

不能假设：

```text
程序开始时 set_mode(6)
        ↓
之后永久处于 Mode 6
```

出现错误、急停或关键配置修改后，都可能需要重新进行：

```python
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(6)
arm.set_state(0)
```

但在真实碰撞或其它硬件异常发生后，应先确认物理原因，不能自动无限清错重启。

---

# 40. Mode 6 下 `q_cmd != q_actual` 是正常现象

假设：

```text
30 Hz
speed = 90°/s
mvacc = 500°/s²
```

policy 不断输出：

```text
q_target[0]
q_target[1]
q_target[2]
...
```

机器人不会：

```text
精确到达 q_target[0]
停下
精确到达 q_target[1]
停下
```

而是：

```text
持续平滑追踪最新 target
```

所以：

\[
e_q(t)=q_{cmd}(t)-q_{actual}(t)
\]

动态阶段存在非零值是正常的。

官方 Mode 6 的设计重点就是在线 transition 的速度/加速度连续，而不是逐 sample 零误差跟踪。

因此数据录制和 policy evaluation 都应该明确记录：

```text
q_cmd
q_actual
```

而不能把二者混为一谈。

---

# 41. Mode 6 官方使用范式的最终总结

UFACTORY 当前官方开源项目所表现出的 Mode 6 设计，可以压缩为：

```text
                    Mode 6

Input:
    absolute q_target

Planner:
    xArm controller internal OTG

Constraints:
    speed
    mvacc

Normal call:
    set_servo_angle()

Normal streaming:
    wait=False

Observation:
    read actual robot state

Initialization:
    Mode 0 / safe alignment first

Teleoperation:
    GELLO → q_target → Mode 6

Robot Learning:
    policy → q_target → Mode 6
```

---

# 42. 对 xArm7 遥操作 + 数据采集 + Policy Deployment 的推荐架构

综合 UFACTORY 当前官方代码，最合理的系统边界是：

```text
┌──────────────────────────────────────┐
│         TELEOP / POLICY LAYER        │
│                                      │
│  GELLO / Policy                      │
│        ↓                             │
│  absolute q_target[7]                │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│            SAFETY LAYER              │
│                                      │
│ joint soft limits                    │
│ NaN / Inf check                      │
│ action jump check                    │
│ workspace / collision strategy       │
│ timeout                              │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│             xArm DRIVER              │
│                                      │
│ Mode 6                               │
│ set_servo_angle(                     │
│     angle=q_target,                  │
│     speed=vmax,                      │
│     mvacc=amax,                      │
│     wait=False                       │
│ )                                    │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│          xArm CONTROLLER             │
│                                      │
│ Online Trajectory Generation         │
│ velocity / acceleration continuity   │
└──────────────────┬───────────────────┘
                   ↓
                xArm7
                   ↓
           q_actual / qdot
                   │
                   └────────→ observation
```

---

# 43. 对自己的 xArm7 组件，哪些官方设计应该直接继承

**建议继承：**

```text
Mode 0 initial positioning
        ↓
Mode 6 online joint control

absolute joint target action

set_servo_angle()

wait=False during streaming

explicit speed

explicit mvacc

is_radian=True

read actual joint state

slow initial synchronization

error/mode/state monitoring

same action semantics for
teleop collection and policy deployment
```

这些原则有明确官方代码支持。

---

# 44. 哪些官方项目实现不建议机械照搬

需要重新评估：

```text
90°/s 是否适合自己的工作空间

500°/s² 是否适合当前 payload

collision sensitivity 配置

线速度限制相关配置

是否需要前 20 帧固定 0.2 rad/s

start_joints 是否适合真实工作台

Reduced Mode / Safety Boundary 范围
```

因为这些属于：

```text
application configuration
```

而不是 Mode 6 protocol 的固有要求。

---

# 45. 参数选择的推荐优先级

在自己的系统中，建议把 Mode 6 参数按如下优先级处理：

```text
1. q_target joint safety
2. payload / mounting configuration
3. robot-side Reduced Mode
4. speed
5. mvacc
6. control frequency
7. initial synchronization
8. error/watchdog handling
```

而不是只调：

```text
speed
```

---

# 46. 面向具身智能系统的建议默认接口

最终建议 robot abstraction 只向策略层暴露：

```python
robot.send_joint_target(
    q_target
)
```

而在 robot component 内部固定管理：

```python
mode = 6
speed
mvacc
state
joint limits
startup alignment
error handling
watchdog
```

不要让具体 policy repository 到处直接调用：

```python
arm.set_mode(...)
arm.set_state(...)
arm.set_servo_angle(...)
```

这样 ACT、Diffusion Policy、VLA 或未来其它 policy repository 都可以复用同一个真实机器人 backend。

---

# 47. 最终结论

基于 UFACTORY 当前官方代码，可以较有把握地得出：

### 结论一

**Mode 6 是 UFACTORY 官方明确支持的 joint-target online planning 接口。**

不是 ServoJ 的别名。

### 结论二

其正确 API 是：

```python
arm.set_mode(6)
arm.set_state(0)

arm.set_servo_angle(
    angle=q_target,
    speed=max_speed,
    mvacc=max_acc,
    is_radian=True,
    wait=False,
)
```



### 结论三

UFACTORY 已经在官方 `ufactory_teleop` 中使用：

```text
GELLO
+
xArm7
+
Mode 6
+
30 Hz
+
90°/s
+
500°/s²
```

进行 joint-space teleoperation。

### 结论四

UFACTORY 的 LeRobot integration 又进一步将同样的：

```text
absolute joint target
→ Mode 6
```

用于：

```text
teleoperation
data collection
policy inference
```



### 结论五

因此对于：

```text
xArm7
+
joint-space teleoperation
+
imitation-learning dataset collection
+
joint-position policy deployment
```

Mode 6 不只是“理论上适合”，而是已经有 UFACTORY 官方机器人学习项目作为直接工程参考。

真正需要我们在自己的系统中进一步做好的是：

```text
显式 speed / mvacc
统一 action semantics
记录 q_cmd / q_actual
初始姿态同步
joint/workspace safety
Reduced Mode
collision configuration
timeout / error handling
```

而不是重新实现一套 Mode 1 高频 trajectory generator。