# dexmani_real 分阶段迁移与简化指南

> 实施状态（2026-08-20）：Phase 0 护栏和 Phase 1 第一批入口收敛已完成。
> `collect_teleop.py`、`keyboard_teleop.py`、`replay_episode.py` 现在只负责
> CLI、配置和调用领域生命周期；对应实现位于 `teleop/session.py`、
> `teleop/keyboard_session.py`、`robot/episode_replay.py`。Replay 只有物理重现
> 语义，不提供 dry-run。XArm 第一轮简化也已按专项指南实施。后续目录审计确认
> `shm`、`recording`、`data_processing`、`deployment` 当前均有明确职责和多个实际
> 消费者，因此不为追求目标目录数量而机械合并。运行行为仍以源码、schema 和配置为准。

## 1. 重构目标

本次重构的目标不是把 `dexmani_real` 改造成一个“更完整的软件框架”，而是让它更接近科研机器人代码应有的状态：

> **主链路清晰、硬件接口直接、脚本短小、模块边界自然、修改局部化。**

参考 ManiUniCon、π₀/R2-Flow 类科研代码的组织思路，最终希望达到：

- 阅读一个入口脚本，可以快速知道系统如何运行；
- 阅读一个硬件文件，可以快速知道设备如何初始化、读取和控制；
- 阅读一个模块时，不需要同时打开五六个基础设施文件；
- 新实验优先通过组合现有函数完成，而不是新增 Manager / Factory / Registry；
- 文件和目录按照真实业务职责存在，而不是按照“软件工程概念”存在；
- `CLAUDE.md` / `AGENTS.md` 只记录长期稳定原则，不同步具体文件结构。

---

# Phase 0 — 建立重构护栏

**优先级：P0，必须最先完成**

这一阶段几乎不重构代码。

目标只有一个：

> 在简化代码之前，先确保我们知道“什么行为不能被改坏”。

## 0.1 固定 Golden Paths

至少固定以下真实运行链路：

1. XArm 初始化 / enable / home
2. XHand 初始化与控制
3. keyboard / VR teleoperation
4. teleop data collection
5. episode replay
6. camera calibration
7. policy deployment / inference

对于每条链路记录：

```text
command
↓
entry script
↓
major modules
↓
hardware / data outputs
```

不需要建立复杂的 integration test framework。

只需要确保迁移前后：

- CLI 参数基本不变；
- robot command semantics 不变；
- observation / action shape 不变；
- dataset schema 不变；
- camera calibration 输出不变；
- control frequency 不发生非预期变化。

---

## 0.2 暂停新增架构层

重构期间原则上禁止新增：

- `Manager`
- `Factory`
- `Registry`
- `Service`
- `Context`
- `Backend`
- `BaseXXX`
- dependency injection
- plugin system
- event bus
- 全局 config framework

除非确实已经存在 **两个以上需要共享接口的具体实现**。

例如：

```python
class BaseArm:
    ...
```

如果现在实际上只有 XArm7：

**不要建立 BaseArm。**

直接：

```python
class XArm:
    ...
```

即可。

---

## 0.3 精简 AI 指导文件

这一阶段同时收缩：

```text
CLAUDE.md
AGENTS.md
```

它们只应该描述：

- 项目目标；
- 稳定的代码原则；
- 允许 / 禁止的设计模式；
- 常用开发命令；
- 安全约束；
- 指向 `docs/code_style.md`。

不要包含：

- 当前有哪些 Python 文件；
- 某个类当前在哪里；
- 当前模块之间具体调用关系；
- TODO；
- 随代码变化的实现说明。

目标：

> 修改业务代码时，90% 以上情况下不需要修改 CLAUDE.md / AGENTS.md。

---

# Phase 1 — 先把所有入口脚本变薄

**优先级：P0**

这是实际代码重构中收益最高的一步。

当前最应该优先处理的不是 package hierarchy，而是：

```text
examples/
```

中的巨大脚本。

---

## 1.1 Script 的最终职责

理想脚本只做四件事：

```python
def main():
    args = parse_args()
    config = build_config(args)
    app = ...
    app.run()


if __name__ == "__main__":
    main()
```

或者更简单：

```python
def main():
    ...
```

脚本负责：

- CLI；
- 配置；
- 创建对象；
- 调用业务函数。

脚本不负责：

- 数学算法；
- camera processing；
- robot SDK；
- multiprocessing implementation；
- dataset implementation；
- shared memory protocol；
- calibration algorithm。

---

## 1.2 建议处理顺序

按照“文件复杂度 × 核心链路重要性”：

### 第一批

```text
replay_episode.py
collect_teleop.py
keyboard_teleop.py
```

这是最关键的一批。

原因：

它们最能暴露当前 runtime / robot / recording / teleop 之间真正需要的 API。

---

### 第二批

```text
calibrate_camera.py
realsense_record_example.py
visualize_episode.py
```

---

### 第三批

```text
xhand_control_example.py
pointcloud_process_example.py
calibrate_vr_heading.py
```

---

## 1.3 不要机械地把所有代码移入 package

原则是：

### 可复用逻辑

进入 package：

```python
load_episode(...)
replay_episode(...)
record_episode(...)
compute_transform(...)
```

### 一次性 orchestration

继续留在 script：

```python
robot = ...
camera = ...
episode = ...

for ...:
    ...
```

科研代码允许 script 有几十到一百多行直线流程。

**不要为了让 script 只有 20 行而制造新的 Runner / Application / Manager。**

---

## Phase 1 完成标准

建议：

- 普通 example：`< 150 LOC`
- 复杂 calibration / visualization：尽量 `< 250 LOC`
- example 中基本不存在 hardware SDK 调用；
- example 中基本不存在 multiprocessing 实现；
- example 中没有大型 class；
- 入口执行流程可以从上到下直接阅读。

---

# Phase 2 — 收口 Robot / Hardware 层

**优先级：P0**

完成入口简化以后，第二个最高收益区域是：

```text
robot/
```

特别是：

```text
arm_sdk.py
arm_loop.py
xhand.py
hand_process.py
homing.py
```

这里应该采用：

> **Device wrapper + explicit control logic**

而不是继续叠加 abstraction layers。

---

# 2.1 先简化 `arm_sdk.py`

结合前面对 dexmani_real 与 LeRobot UFactory XArm7 实现的比较：

`arm_sdk.py` 最终应该只是一个很薄的 XArm SDK adapter。

例如职责：

```text
connect
disconnect

get_joint_position
get_tcp_pose

move_joint
move_pose
servo_joint
servo_pose

enable
disable
stop
```

以及少量：

```text
check_error
recover_error
```

---

不要让 `arm_sdk.py` 管理：

- multiprocessing；
- shared memory；
- policy；
- teleop；
- trajectory planning；
- global runtime state；
- dataset；
- application lifecycle。

理想结构：

```python
class XArm:
    def connect(...):
        ...

    def get_state(...):
        ...

    def servo_joint(...):
        ...

    def stop(...):
        ...
```

**硬件 wrapper 应该“无聊”。**

越无聊越好。

---

# 2.2 再简化 `arm_loop.py`

`arm_loop.py` 不应该同时承担：

```text
SDK adapter
+
control loop
+
IPC
+
state machine
+
safety
+
process lifecycle
+
logging
```

最终应该明确区分：

```text
XArm
    ↓
Arm control loop
    ↓
runtime / application
```

核心循环应该能够被快速读懂：

```python
while running:
    command = receive_command()
    state = arm.get_state()

    command = safety_filter(command, state)

    arm.servo(command)

    publish_state(state)
```

复杂性应该围绕这条链路展开，而不是隐藏它。

---

# 2.3 `safety.py` 保持显式

Safety 是少数值得单独存在的模块。

例如：

```python
check_joint_limits(...)
check_workspace(...)
clip_velocity(...)
```

不要做：

```python
SafetyManager
SafetyContext
SafetyBackend
SafetyRegistry
```

优先纯函数。

---

# 2.4 `xhand.py`

同样采用：

```text
thin device wrapper
```

目标：

```python
hand.connect()
hand.get_position()
hand.set_position(...)
hand.stop()
```

设备协议、错误处理和必要的单位转换可以存在。

但：

- retargeting；
- hand trajectory；
- teleop；
- process orchestration

不应该进入 `xhand.py`。

---

# 2.5 `hand_process.py`

这个名称本身就是一个信号。

如果它主要负责：

### Hand retargeting

迁移到：

```text
teleop/retarget.py
```

### Hand low-level control

迁移到：

```text
robot/hand_control.py
```

### multiprocessing wrapper

应该尽可能压缩，然后进入：

```text
runtime/
```

不要继续保留一个模糊的：

```text
hand_process.py
```

同时承担三者。

---

# 2.6 `homing.py`

Homing 属于非常明确的 robot operation。

允许：

```python
home_arm(...)
home_hand(...)
home_robot(...)
```

但应该避免让它发展为完整 application framework。

其中：

- reusable homing algorithm → `robot/homing.py`
- CLI / user interaction → script
- calibration data → config / calibration file
- SDK commands → `XArm`

---

## Phase 2 完成标准

Robot 层依赖方向应该接近：

```text
script / teleop / policy
          ↓
     robot control
          ↓
    XArm / XHand
          ↓
     vendor SDK
```

禁止反向依赖。

---

# Phase 3 — 简化 Runtime / Multiprocessing / Shared Memory

**优先级：P1**

这一阶段不要太早进行。

因为只有在 Phase 1–2 之后，才能真正看清哪些 IPC abstraction 是必要的。

核心原则：

> multiprocessing 是实现细节，不应该决定整个代码库的架构。

---

## 3.1 检查 `runtime/`

逐项问：

```text
processes.py
supervisor.py
status.py
```

### 每一个 abstraction 都问：

> 如果删除这一层，调用代码是否明显更难理解？

如果答案是否：

删除。

---

例如不要为了：

```python
ProcessStatus.RUNNING
ProcessStatus.STOPPED
```

建立完整 runtime state architecture。

如果：

```python
process.is_alive()
```

已经足够，就使用标准库。

---

# 3.2 简化 supervisor

Supervisor 应该仅在真实需要统一管理：

```text
arm process
hand process
camera process
policy process
```

时存在。

理想 API：

```python
supervisor.start()
supervisor.stop()
supervisor.join()
```

而不是：

```text
register
resolve
dispatch
transition
dependency graph
service lifecycle
```

---

# 3.3 `shm/` 后期并入 runtime

如果 shared memory 仅用于 runtime：

最终建议：

```text
runtime/
    process.py
    shm.py
```

而不是：

```text
runtime/
shm/
```

同时作为两个一级 package。

只有 shared memory 本身已经形成独立、稳定、被多个 domain 直接使用的 subsystem，才值得保留 `shm/`。

---

# Phase 4 — 合并 Data / Recording 链路

**优先级：P1**

当前概念：

```text
recording/
data_processing/
```

很可能可以逐步收敛为：

```text
data/
```

推荐：

```text
data/
    episode.py
    recorder.py
    processing.py
```

或者文件更少：

```text
data/
    episode.py
    recorder.py
```

---

## 4.1 数据层应该围绕 Episode

建立清晰的数据概念：

```text
observation
action
timestamp
episode
dataset
```

不要围绕 infrastructure 建模。

例如优先：

```python
episode = load_episode(path)
save_episode(path, episode)
```

而不是：

```python
EpisodeStorageManager(...)
EpisodeProvider(...)
DatasetBackend(...)
```

---

# Phase 5 — 最后才重新整理目录结构

**优先级：P2**

这一阶段才开始真正进行 package flattening。

原因：

> 如果 Phase 1–4 没完成，现在移动目录只是在重新排列复杂代码。

---

# 5.1 推荐目标结构

不要求一步迁移到位。

建议逐步趋近：

```text
dexmani_real/
├── config.py
│
├── robot/
│   ├── xarm.py
│   ├── xhand.py
│   ├── control.py
│   ├── homing.py
│   └── safety.py
│
├── sensor/
│   └── realsense.py
│
├── teleop/
│   ├── vr.py
│   └── retarget.py
│
├── data/
│   ├── episode.py
│   ├── recorder.py
│   └── processing.py
│
├── policy/
│   └── runner.py
│
└── runtime/
    ├── process.py
    └── shm.py
```

这不是强制最终文件列表。

核心是将现在十余个一级概念压缩到约：

```text
robot
sensor
teleop
data
policy
runtime
```

六个清晰 domain。

---

# 5.2 当前目录的迁移方向

### `recording/`

→ `data/`

---

### `data_processing/`

→ `data/`

---

### `shm/`

→ `runtime/shm.py`

---

### `deployment/`

不要长期作为独立 domain。

内容根据职责分别进入：

```text
policy/
runtime/
```

---

### `integrations/`

优先消灭这个目录。

Adapter 应靠近它服务的对象。

例如：

```text
policy/some_policy.py
sensor/realsense.py
teleop/vr.py
```

而不是统一：

```text
integrations/
```

---

### `planning/`

判断是否真的存在一个明确 planner。

如果只是 robot trajectory helper：

```text
robot/
```

如果是 policy execution：

```text
policy/
```

只有存在真正独立的 planning subsystem 才继续保留。

---

### `utils/`

长期目标：

> `utils/` 越小越好。

迁移规则：

```text
camera helper → sensor/
robot math → robot/
dataset helper → data/
teleop transform → teleop/
```

只有真正跨 domain 的基础函数才留在 `utils/`。

---

### `config/`

如果 config 文件数量最终很少：

```text
config/
```

可以直接压缩成：

```text
config.py
```

不要为了配置建立第二套 framework。

---

# Phase 6 — 文件内部简化

**优先级：P2**

完成架构收敛之后，再系统处理每个 Python 文件。

---

## 6.1 Import

顺序统一：

```python
# stdlib

# third-party

# local
```

避免：

```python
from dexmani_real.xxx import *
```

避免大量 alias。

---

# 6.2 函数优先于类

默认优先：

```python
def transform_pose(...):
    ...
```

而不是：

```python
class PoseTransformer:
    ...
```

只有存在：

- 生命周期；
- 持久状态；
- 外部资源；
- 强相关的一组行为；

才使用 class。

---

# 6.3 一个函数只做一件事

出现：

```python
def initialize_and_start_and_wait_and_record(...):
```

应该拆。

但不要机械追求：

```text
每个函数 < 20 行
```

科研代码中一个清晰的 50 行线性算法，往往比十个 5 行 helper 更容易阅读。

---

# 6.4 减少 defensive abstraction

不要为了“未来可能需要”设计：

```text
BaseCamera
BaseRobot
RobotFactory
PolicyFactory
ConfigRegistry
DeviceRegistry
```

采用：

> 第二个实现出现以后再抽象。

而不是：

> 为第二个实现预先抽象。

---

# Phase 7 — 删除过度工程化遗留

**优先级：P3**

此时进行最后的删除。

重点搜索：

```text
Manager
Factory
Registry
Context
Service
Wrapper
Adapter
Base
Abstract
Helper
Utils
```

逐个判断。

不是看到名字就删除。

判断标准：

> 这个 abstraction 是否减少了调用者需要知道的信息？

如果没有：

删除。

---

# 建议的实际 PR 顺序

不要创建一个巨大 `refactor everything` PR。

推荐：

## PR 1 — Refactor guardrails

只做：

```text
Golden paths
stable CLAUDE.md
stable AGENTS.md
code style rules
```

不改核心行为。

---

## PR 2 — Thin replay / collection scripts

重点：

```text
replay_episode.py
collect_teleop.py
keyboard_teleop.py
```

目标：

暴露真正需要的 public API。

---

## PR 3 — Simplify XArm stack

重点：

```text
arm_sdk.py
arm_loop.py
safety.py
```

目标：

形成：

```text
SDK → XArm → control loop
```

---

## PR 4 — Simplify XHand stack

重点：

```text
xhand.py
hand_process.py
homing.py
```

拆分：

```text
device
control
retargeting
homing
```

---

## PR 5 — Simplify data stack

合并：

```text
recording/
data_processing/
```

形成：

```text
data/
```

---

## PR 6 — Simplify runtime

处理：

```text
runtime/
shm/
```

删除不必要 supervisor / status abstraction。

---

## PR 7 — Flatten package hierarchy

处理：

```text
deployment/
integrations/
planning/
utils/
config/
```

此时目录变化应该主要是：

> 删除和合并

而不是：

> 新建更多目录。

---

## PR 8 — Dead-code cleanup

最后统一：

- 删除 compatibility aliases；
- 删除 temporary wrappers；
- 删除 deprecated imports；
- 删除旧入口；
- rename 剩余模糊模块；
- 删除重复 types；
- 更新 README。

---

# 每次迁移的固定步骤

对于一个旧模块：

```text
A.py
```

采用：

### Step 1

确认调用者。

### Step 2

判断哪些代码：

```text
真正 reusable
```

哪些只是：

```text
orchestration
```

### Step 3

先设计最小 API。

例如：

```python
arm.get_state()
arm.servo_joint(q)
```

### Step 4

让一个入口迁移到新 API。

### Step 5

实际运行验证。

### Step 6

迁移其余调用者。

### Step 7

直接删除旧 abstraction。

不要长期：

```text
OldArm
→ NewArmAdapter
→ XArmWrapper
→ XArm
```

兼容层只能短期存在。

---

# 判断一个模块是否应该独立存在

每次想增加一个文件 / package 时问四个问题。

## 问题 1

它是否拥有明确且独立的领域概念？

例如：

```text
robot
teleop
dataset
policy
```

是。

```text
helpers
common
integrations
services
```

通常不是。

---

## 问题 2

它是否有两个以上调用者？

如果只有一个调用者：

优先放回调用者附近。

---

## 问题 3

它是否能被一句话定义？

例如：

> `xarm.py` 封装 XArm SDK。

很好。

> `runtime.py` 包含各种程序运行时需要的功能。

危险。

---

## 问题 4

删除这一层后代码是否更难理解？

如果反而更容易：

删除。

---

# 最重要的三个迁移原则

## 原则一：先减少跳转次数

例如：

```text
script
 → manager
   → supervisor
     → process
       → adapter
         → sdk wrapper
           → SDK
```

优先变成：

```text
script
 → control
   → XArm
     → SDK
```

代码行数不是最重要的指标。

**阅读时需要打开几个文件才是。**

---

## 原则二：目录变化必须是代码简化的结果

不要：

```text
先设计一个漂亮的新目录
↓
再把旧代码搬进去
```

应该：

```text
先简化代码
↓
发现几个模块自然属于同一个 domain
↓
合并目录
```

---

## 原则三：科研代码应该优化修改成本，而不是扩展性

dexmani_real 当前最常见的未来变化应该是：

```text
换 policy
改 observation
改 teleop
增加 camera
改变 robot control
改变 dataset
```

所以架构需要优化这些实验修改。

不需要提前优化：

```text
支持任意机器人
支持任意 sensor backend
支持任意 deployment provider
支持 plugin ecosystem
```

---

# 最终验收指标

完成本轮迁移后，希望看到以下变化。

### Package 数量

从当前十余个一级 package：

```text
config
data_processing
deployment
integrations
planning
policy
recording
robot
runtime
sensor
shm
teleop
utils
...
```

逐渐收敛到约：

```text
robot
sensor
teleop
data
policy
runtime
```

---

### Entry scripts

大多数：

```text
50–150 LOC
```

复杂 calibration：

```text
< 250 LOC
```

---

### Hardware

调用链：

```text
control
→ device wrapper
→ vendor SDK
```

原则上不超过三层。

---

### Class

看到一个 class，应能回答：

> 它保存了什么长期状态？

回答不出来时优先改为函数。

---

### Utils

`utils/` 应持续缩小，而不是持续增长。

---

### AI instructions

`CLAUDE.md` / `AGENTS.md` 不跟随普通代码重构发生变化。

---

# 当前最应该立即做的事情

如果只选择一个起点：

> **不要先移动目录。**

第一轮直接处理：

```text
replay_episode.py
collect_teleop.py
keyboard_teleop.py
```

把它们变成薄入口。

随后立即进入：

```text
arm_sdk.py
arm_loop.py
```

把 XArm 的：

```text
SDK access
control loop
IPC/runtime
```

三个职责彻底分开。

完成这两步以后，再重新观察整个 repository。

届时：

```text
deployment
runtime
shm
recording
data_processing
integrations
planning
utils
```

中哪些应该合并或删除，通常会变得非常明显。

**这比现在直接设计一个“完美的新目录结构”可靠得多。**
