# Dexmani XArm7 大幅化简迁移操作指南

## 0. 重构目标

这次不要把目标定义成“整理 `arm_loop.py`”，而应该定义成：

> **删除 dexmani 自己构造出来的机器人控制状态机，让 arm 模块重新退化为一个薄硬件驱动。**

最终职责应收敛为：

```text
Policy / Teleop
      │
      │ qpos
      ▼
SafetyGate
      │
      ▼
command transport
      │
      ▼
arm_loop
      │
      ▼
XArm7
      │
      ▼
xArm SDK
```

推荐最终只保留：

```text
dexmani_real/
├── robot/
│   ├── xarm7.py          # 薄 SDK wrapper
│   ├── arm_loop.py       # 薄 worker
│   └── homing.py         # HOME 路径规划 + 简单执行
│
├── policy/
│   └── safety.py         # action 几何/关节/workspace 安全
│
└── shm/
    └── ...               # 简单 command/state transport
```

### 最终职责

`XArm7`：

```text
connect()
read()
servo(qpos)
home(waypoints)
stop()
close()
```

`arm_loop`：

```text
读取命令
调用 XArm7
读取状态
发布状态
```

`SafetyGate`：

```text
判断 qpos 是否可以发送
```

`Main`：

```text
管理程序生命周期
管理实验是否运行
管理 E-stop / shutdown
```

---

# 第一阶段：删除动态 Mode 状态机

**优先级：最高**

这是整个化简中收益最大的一步。

当前 `arm_loop.py` 的复杂性有相当一部分来自：

```text
SafetyState
    ↓
ModeTransition
    ↓
issue_mode_enter()
    ↓
mode_enter_ready()
    ↓
accepts_motion_commands
    ↓
mode drift
    ↓
ARM_NOT_READY
```

这些机制之间不是独立功能，而是相互催生出来的。

## 1.1 首先改变控制语义

建立以下硬规则：

### 正常运行

```text
startup
    ↓
Mode 6
    ↓
整个 runtime 保持 Mode 6
```

不要因为：

```text
DISARMED
ARMED
RUNNING
```

而切换 xArm Mode。

这三个是 **Dexmani 软件状态**，不是机械臂控制模式。

---

## 1.2 新的推荐状态语义

推荐：

```text
DISARMED
    → 不发送 servo command

ARMED
    → 可以发送 servo command

RUNNING
    → 可以发送 servo command

FAULT
    → worker 退出 / stop

ESTOP
    → emergency_stop / state 4
```

正常：

```text
DISARMED → ARMED
ARMED → RUNNING
RUNNING → ARMED
```

都不再：

```python
set_mode(...)
```

### 更保守版本

如果你希望 DISARMED 时机械臂进入 State 4，则只允许：

```text
DISARMED
    set_state(4)

ARMED
    set_state(0)
```

但仍然：

```text
不要 set_mode(6)
```

因为 Mode 6 没必要反复进入。

我的建议是第一版直接采用：

> **software disarm**

也就是 DISARMED 时停止发送命令，机械臂保持当前位置。

---

# 1.3 删除以下概念

从 `arm_sdk.py` 删除：

```text
_wait_controller_ready()
_enter_mode()
enter_mode0()
enter_mode6()
issue_mode_enter()
mode_enter_ready()
```

最终允许保留一个非常简单的内部 helper：

```python
def _set_mode(arm, mode):
    check_code(arm.set_mode(mode))
    check_code(arm.set_state(0))
```

但它只在：

```text
startup
HOME
```

调用。

---

从 `arm_loop.py` 删除：

```text
_MODE_DRIFT_TIMEOUT_S
_MODE_ENTER_TIMEOUT_S

_ModeTransition

mode_transition

mode_mismatch_since_s

_advance_mode_drift()

所有 transitioning 分支
```

同时删除：

```text
controller mode drift fault
Mode 6 postcondition timeout
transitioning 时临时 mask error_code
```

---

# 1.4 删除 `accepts_motion_commands`

目前它的出现本质上是因为：

```text
mode == 6
```

不足以表示：

```text
Dexmani 认为机械臂准备好了
```

于是又创建：

```text
accepts_motion_commands
```

然后 policy 又必须：

```python
mode == 6 and accepts_motion_commands
```

这是典型的软件状态重复表达。

删除：

```text
ARM_STATE_DTYPE.accepts_motion_commands
```

同时删除：

```text
CommandPublishStatus.ARM_NOT_READY
```

以及：

```text
_arm_feedback_snapshot()
```

中与：

```text
mode
accepts_motion_commands
```

相关的 gate。

---

# 1.5 第一阶段结束时 arm runtime 应该是

```text
connect
    ↓
motion_enable
    ↓
configure
    ↓
mode 6
    ↓
state 0
    ↓
---------------------
runtime loop
    ↓
if ARMED/RUNNING:
    servo(command)

read state

publish state
---------------------
    ↓
stop
    ↓
disconnect
```

### 第一阶段不要做

暂时不要修改：

```text
HOME transport
ARM_COMMAND_DTYPE
ARM_STATE_DTYPE 其他字段
FK
SharedStorage
```

只删除 Mode 状态机。

这样问题容易定位。

---

# 第一阶段验收标准

必须能够完成：

```text
启动
↓
进入 Mode 6
↓
DISARMED
↓
ARMED
↓
连续控制
↓
DISARMED
↓
再次 ARMED
↓
继续控制
↓
shutdown
```

期间：

```text
set_mode(6)
```

只能在 startup 出现一次。

执行：

```bash
rg "set_mode|enter_mode|mode_enter|mode_transition|mode_drift"
```

正常 runtime 中不应该再看到 Mode 切换逻辑。

建议单独提交：

```text
refactor(arm): remove runtime mode state machine
```

---

# 第二阶段：建立真正的薄 `XArm7`

**优先级：很高**

这一阶段开始消灭 `arm_sdk.py`。

不要把 `arm_sdk.py` 改名成 `xarm7.py` 后继续保留原来的全部 abstraction。

目标是重新写一个很薄的类。

---

## 2.1 创建概念上的 API

目标：

```python
class XArm7:
    def connect(self): ...
    def read(self): ...
    def servo(self, qpos): ...
    def home(self, waypoints): ...
    def stop(self): ...
    def close(self): ...
```

不要新增：

```text
XArmController
XArmManager
ArmLifecycle
ArmRuntime
ModeManager
ArmStateMachine
ArmHealthMonitor
```

这次重构的核心就是减少 abstraction。

---

# 2.2 `connect()` 只做一次性初始化

建议调用顺序：

```text
XArmAPI(ip)
↓
检查 axis == 7
↓
检查当前 controller error
↓
motion_enable(True)
↓
set_mode(0)
↓
set_state(0)
↓
set_collision_sensitivity(...)
↓
set_tcp_load(...)
↓
set_joint_maxacc(...)
↓
set_mode(6)
↓
set_state(0)
```

这里使用 Mode 0 完成初始化后，再进入 Mode 6。

这和 `lerobot_ufactory` 的基本结构一致。

---

# 2.3 SDK 错误检查统一成一个函数

保留：

```python
def _check(code, operation):
    if code != 0:
        raise RuntimeError(...)
```

到此为止。

不要再存在：

```text
SDK code
    ↓
live error read
    ↓
cached error comparison
    ↓
controller error description
    ↓
special diagnostics
    ↓
StopResult
```

---

# 2.4 `read()` 应该很短

概念：

```python
def read(self):
    code, states = self.arm.get_joint_states(
        is_radian=True,
        num=3,
    )
    _check(code, "get_joint_states")

    return (
        np.asarray(states[0]),
        np.asarray(states[1]),
        np.asarray(states[2]),
    )
```

只进行：

```text
SDK code
shape
finite
```

三项检查。

### 不再做

```text
retry counter
last-known fallback
state_valid
connected=False frame
persistent error threshold
```

如果真实机械臂读取失败：

> worker 直接失败。

对于研究代码，这比构造“半健康状态”简单得多。

---

# 2.5 `servo()` 应该很短

概念：

```python
def servo(self, qpos):
    code = self.arm.set_servo_angle(
        angle=qpos,
        is_radian=True,
        speed=self.max_speed,
        mvacc=self.max_acc,
        wait=False,
    )
    _check(code, "set_servo_angle")
```

不要：

```text
setter return 非零
↓
同步读取 controller error
↓
错误分类
↓
C31 diagnostics
↓
latch_arm_fault
↓
stop confirm
```

`set_servo_angle()` 失败：

```text
raise
↓
arm_loop 顶层捕获
↓
error_state = True
↓
finally stop
```

足够。

---

# 2.6 `stop()` 不需要 confirmation protocol

删除：

```text
StopResult
confirmed
reason
1 秒 polling
get_state()
retry
```

改成：

```python
def stop(self):
    self.arm.set_state(4)
```

cleanup 中：

```python
try:
    arm.stop()
except Exception:
    logger.warning(...)
```

即可。

不要试图证明：

> controller 一定成功进入 State 4。

硬件 SDK 自己已经是最后一层。

---

# 第二阶段结束后删除 `arm_sdk.py`

迁移完：

```bash
rg "arm_sdk"
```

确保无引用。

然后删除：

```text
dexmani_real/robot/arm_sdk.py
```

建议提交：

```text
refactor(arm): replace arm sdk framework with thin xarm7 driver
```

---

# 第三阶段：把 `arm_loop.py` 压成真正的 Worker

**目标：从约 1000 行量级下降到约 150–300 行。**

这一阶段不要增加新的 class。

---

# 3.1 删除 `_Flow`

删除：

```python
class _Flow(Enum):
    PROCEED
    NEXT
    EXIT
```

Python 本身已经有：

```text
continue
break
return
```

没必要重新造 control flow abstraction。

---

# 3.2 `_LoopState` 大幅缩减

当前 `_LoopState` 承担太多东西。

最终最多保留：

```python
@dataclass
class _LoopState:
    arm: XArm7
    last_target: np.ndarray
    last_cmd_seq: int = 0
```

甚至可以完全不需要 `_LoopState`。

---

# 3.3 删除 `_startup()` framework

不要：

```text
_startup
  ├── _make_fk
  ├── _connect_arm
  ├── _validate_identity
  ├── _check_startup_error
  ├── _enable_and_mode0
  ├── _apply_config
  ├── _read_initial_state
  └── _confirm_stopped
```

改成：

```text
arm = XArm7(cfg)
arm.connect()
qpos, qvel, tau = arm.read()
publish()
shared.set_ready("arm")
```

---

# 3.4 删除 `latch_arm_fault()`

错误处理统一放到 worker 顶层：

```python
def arm_loop(...):
    arm = None

    try:
        arm = XArm7(...)
        arm.connect()

        while shared.is_running.value:
            ...

    except Exception:
        shared.error_state.value = True
        logger.exception("arm worker failed")

    finally:
        if arm is not None:
            try:
                arm.stop()
            except Exception:
                pass

            arm.close()
```

这是整个 worker 唯一的大异常边界。

---

# 3.5 runtime loop 只保留四步

最终逻辑：

```text
while running:

    1. heartbeat

    2. consume command

    3. read arm

    4. publish state
```

概念代码：

```python
while shared.is_running.value:
    heartbeat()

    if estop:
        arm.emergency_stop()
        break

    if safety in (ARMED, RUNNING):
        command = read_command()
        if command is not None:
            arm.servo(command)

    qpos, qvel, tau = arm.read()
    publish(qpos, qvel, tau)

    rate.wait()
```

这应该成为整个 arm worker 的主体。

---

# 第四阶段：删除多层错误处理

**优先级：高**

建立一条原则：

> **真实硬件调用失败就 fail fast。**

不要试图“智能恢复”。

---

# 4.1 删除以下机制

```text
RetryCounter
state_err_counter
fk_err_counter

max_consecutive_arm_health_failures

terminal_feedback_detail

tracking_err_count

ThrottledWarner 用于 arm health
```

---

# 4.2 保留哪些错误处理

只保留五类：

### A. 初始化失败

```text
无法连接
axis != 7
controller 有 error
```

直接：

```text
raise
```

---

### B. state read 失败

```text
get_joint_states != 0
```

直接：

```text
raise
```

---

### C. servo command 失败

```text
set_servo_angle != 0
```

直接：

```text
raise
```

---

### D. controller error

如果希望保留：

```python
if self.arm.error_code != 0:
    raise RuntimeError(...)
```

每个 loop 检查一次已经足够。

不要再：

```text
cached error
↓
live error
↓
diagnostic mapping
```

---

### E. shutdown / estop

保持：

```text
best-effort stop
disconnect
```

---

# 4.3 删除 controller error description framework

删除：

```text
_CONTROLLER_ERROR_HELP
_CONTROLLER_ERROR_LABELS
describe_controller_error()
get_c31_error_info()
```

日志：

```text
xArm controller error: 31
```

已经足够去查 SDK 文档。

---

# 第五阶段：精简 `ARM_STATE_DTYPE`

这一阶段才开始动 IPC。

不要第一阶段就动。

---

# 5.1 推荐 ARM state

推荐保留：

```text
qpos
qvel
tau
last_cmd_seq
error_code
source_monotonic_ns
```

如果真正需要 FK：

不要放在这里，下一阶段处理。

---

# 5.2 删除字段

建议删除：

```text
connected
mode
tracking_err

last_cmd_created_s
last_cmd_received_s
last_cmd_applied_s

last_cmd_queue_latency_s
last_cmd_apply_latency_s
last_cmd_sdk_duration_s

last_cmd_is_hold

publish_monotonic_ns

state_valid

accepts_motion_commands

timestamp
```

---

# 5.3 为什么 `connected` 可以删除

当前逻辑倾向于：

```text
连接异常
↓
继续发布 connected=False
```

新的逻辑：

```text
连接异常
↓
worker raise
↓
error_state=True
↓
worker exit
```

因此：

```text
worker alive + heartbeat
```

已经代表 driver health。

不要重复表示。

---

# 5.4 为什么 `mode` 可以删除

正常 runtime：

```text
Mode 恒定为 6
```

HOME 是同步的特殊过程。

所以 policy 不需要通过 observation 猜：

```text
现在是不是 Mode 6
```

控制模式属于 driver 内部 implementation detail。

---

# 第六阶段：简化 `policy/safety.py`

当前 `policy/safety.py` 同时承担：

```text
action validity
robot lifecycle
feedback health
mode readiness
command publication
ack
temporal validity
```

需要重新明确：

> `SafetyGate` 只决定“这个动作本身是否安全”。

---

# 6.1 SafetyGate 保留

保留：

```text
shape
finite

arm joint limits
hand joint limits

workspace

collision / environment checks
```

这些属于真正的 manipulation safety。

---

# 6.2 从 SafetyGate 删除

删除：

```text
ARM_NOT_READY

mode == 6

accepts_motion_commands

connected

state_valid

复杂 feedback health 分类

worker readiness classification
```

这些不是 action safety。

---

# 6.3 生命周期 gate 只保留简单判断

发送 command 前最多：

```python
if not shared.is_running.value:
    return

if shared.error_state.value:
    return

if shared.safety_state.value not in (ARMED, RUNNING):
    return
```

足够。

不要再创建十几种：

```text
CommandPublishStatus
```

如果调用方确实需要结果：

```text
SUCCESS
REJECTED
```

两三种即可。

---

# 第七阶段：HOME 大幅降级为普通 blocking operation

这是另一个可以删除大量代码的阶段。

当前 HOME 被实现为：

```text
异步 RPC 服务
```

但 HOME 本质上是：

```text
低频维护操作
```

没必要实时化。

---

# 7.1 保留 collision-safe planner

真正有研究价值的是：

```text
当前 qpos
↓
collision checking
↓
safe waypoints
↓
home_qpos
```

这一部分保留。

尤其：

```text
self collision
table collision
environment collision
joint limits
```

都应保留。

---

# 7.2 删除 HOME transaction framework

最终删除：

```text
HOME_SENTINEL

HomeOutcome

HomeRequest

HomeResult

arm_home_result_q

request_id

run_generation

execution_timeout_s transaction

result acknowledgement

fault acknowledgement grace

stale result draining
```

---

# 7.3 新的 HOME

推荐：

```python
waypoints = plan_home(current_qpos)

arm.home(waypoints)
```

`XArm7.home()`：

```text
set_mode(0)
set_state(0)

for waypoint:
    set_servo_angle(..., wait=True)

set_mode(6)
set_state(0)
```

就是这样。

---

# 7.4 HOME 可以 blocking

明确接受：

```text
HOME 期间 arm worker 暂停正常 servo loop
```

这是正确的简化。

因为 HOME 时本来就不应该：

```text
policy inference
teleop servo
```

继续控制 arm。

不要为了 HOME 期间还能 30 Hz publish state 而设计一整套状态机。

---

# 7.5 HOME completion

只需要：

```text
home() 正常返回
    = success

raise
    = failure
```

不要：

```text
SUCCESS
FAILED
CANCELLED
provisional
finalized
```

如果 operator estop：

```text
HOME 抛出 / runtime 终止
```

即可。

---

# 第八阶段：FK 移出 arm worker

当前 arm worker：

```text
encoder
↓
FK
↓
EEF
↓
publish
```

建议变成：

```text
arm worker
↓
qpos
↓
state ring
```

然后：

```text
policy observation
↓
ArmFK(qpos)
↓
eef pose
```

---

# 8.1 删除

从 `arm_loop.py` 删除：

```text
ArmFK

_make_fk()

fk

fk_warn

fk_err_counter

eef_pos/eef_rot6d calculation
```

---

# 8.2 FK failure 不再停止真实机械臂

如果 policy 需要 EEF pose：

```text
FK 失败
↓
当前 policy step 无效
```

但不应该：

```text
FK 失败
↓
global robot fault
↓
stop xArm
```

这是非常重要的职责边界。

---

# 第九阶段：最后再改 Arm Command Transport

这是最后一项结构性修改。

不要过早做。

---

# 9.1 目标

目前：

```text
arm_action_q
```

同时承担：

```text
servo
HOME
```

拆除 HOME 后，servo transport 可以单纯很多。

推荐改成：

```text
arm_cmd_ring
```

latest-wins。

因为机器人实时控制通常关心：

> 最新动作是什么？

而不是：

> 每一个历史动作都必须执行。

---

# 9.2 推荐 command schema

最终可以只有：

```text
seq
qpos
created_monotonic_ns
```

如果需要更简单：

```text
seq
qpos
```

也可以。

---

# 9.3 删除 command metadata

逐步删除：

```text
run_generation
observation_id
target_monotonic_ns
valid_until_monotonic_ns
is_hold
```

除非某个字段能明确回答：

> 当前实验或论文里哪个功能必须依赖它？

如果答案只是：

```text
理论上未来可能需要
```

就删除。

---

# 9.4 stale command 用更简单的方法解决

不要再引入 generation protocol。

进入 ARMED 时：

```text
记录当前 command seq
```

只接受之后更新的：

```text
seq > armed_at_seq
```

即可避免 DISARMED 前的旧命令重新执行。

如果需要 timeout：

```python
now_ns - created_ns < max_command_age_ns
```

只保留这一条。

---

# 第十阶段：删除 SharedStorage 中的 arm-specific engineering

前九步完成后，再清 SharedStorage。

执行：

```bash
rg "HomeRequest"
rg "HomeResult"
rg "HOME_SENTINEL"
rg "arm_home_result_q"

rg "accepts_motion_commands"
rg "ARM_NOT_READY"

rg "run_generation"
rg "last_cmd_queue_latency"
rg "last_cmd_apply_latency"
```

确认真正无人使用后删除。

---

# 第十一阶段：删除冗余配置

当前配置同样存在 duplication：

```text
ArmParams
↓
ArmLoopConfig
↓
runtime Arm config
```

最终建议只保留：

```text
ArmParams / runtime.arm
```

不要再创建一份：

```text
ArmLoopConfig
```

---

## Arm driver 真正需要的配置

大致只有：

```text
ip

max_joint_velocity
max_joint_acceleration

collision_sensitivity

tcp_load_mass
tcp_load_cog

home speed

loop_hz
```

其余：

```text
tracking_error_warn
mode timeout
health failure count
mode drift timeout
home transaction timeout
```

应随着旧 framework 一起删除。

---

# 第十二阶段：重新定义 SafetyState

`SafetyState` 本身可以暂时保留：

```text
DISARMED
ARMED
RUNNING
FAULT
```

但要把它降级成：

> **Main / experiment lifecycle 状态。**

不要让每个 worker 都实现一遍它的 transition logic。

arm worker 只做：

```python
enabled = safety in (ARMED, RUNNING)
```

即可。

如果后面继续化简，还可以考虑：

```text
enabled: bool
error_state: bool
```

但这不是第一轮重点。

---

# 整个迁移期间必须保留的东西

这次虽然要大胆删工程化，但以下内容不要为了 LOC 一起删掉。

## 保留 1：E-stop

必须保留明确的：

```text
estop
↓
emergency_stop / State 4
↓
停止 runtime
```

---

## 保留 2：SDK return code

这些必须检查：

```text
get_joint_states
set_servo_angle
startup configuration
```

只是不要再建立第二层 error framework。

---

## 保留 3：关节限位

Policy / SafetyGate 中继续保留。

---

## 保留 4：碰撞检测

特别是：

```text
self collision
table
environment geometry
```

这些是真正和你的 dexterous manipulation 实验相关的 safety logic。

---

## 保留 5：TCP load 和 collision sensitivity

这是机器人 firmware dynamics / collision detection 的必要配置。

---

## 保留 6：heartbeat

如果现有多进程 Main 依赖 heartbeat 判断 worker 是否 alive，可以继续保留。

但 heartbeat 只能：

```text
每 loop 更新一次
```

不要继续参与：

```text
HOME transaction
mode switching
error retry
```

---

# 明确禁止重新引入的模式

执行迁移时建议把下面这些规则直接告诉 Claude Code / Codex。

## 不要为了兼容旧实现新增中间层

禁止：

```text
LegacyArmAdapter
ArmCompatibilityLayer
ArmRuntimeManager
ModeController
ArmLifecycleManager
```

如果删掉一个 abstraction 后又增加一个 wrapper 来兼容它，等于没重构。

---

## 不要保留双路径

不要：

```python
if use_new_arm_driver:
    ...
else:
    legacy_arm_loop(...)
```

每个阶段一旦验证：

```text
直接删除旧路径。
```

Git 已经是 backup。

---

## 不要为了 future-proof 创建 abstraction

例如：

```text
BaseArm
ArmProtocol
AbstractRobotArm
ArmBackend
```

目前只有 xArm7，就直接：

```text
XArm7
```

等第二种机械臂真正加入时再抽象。

---

## 不要给两个状态创建 Enum

例如：

```text
READY
NOT_READY
```

一个 bool 足够。

---

## 不要为普通 SDK 调用增加 retry framework

如果未来硬件实验表明：

```text
get_joint_states 偶尔确实瞬态失败
```

再局部增加：

```python
for _ in range(2):
```

不要重新引入：

```text
RetryCounter
RetryPolicy
HealthState
FaultEscalation
```

---

# 推荐实际提交顺序

建议严格做成多个小 commit：

```text
01 refactor(arm): remove runtime mode transitions

02 refactor(arm): introduce thin xarm7 driver

03 refactor(arm): simplify arm worker loop

04 refactor(arm): collapse arm error handling

05 refactor(arm): simplify arm state schema

06 refactor(safety): remove arm readiness protocol

07 refactor(arm): simplify homing execution

08 refactor(arm): move FK out of hardware worker

09 refactor(arm): replace action queue with latest-wins commands

10 refactor(shm): remove obsolete arm IPC protocol

11 refactor(config): remove obsolete arm runtime configuration

12 cleanup(arm): delete compatibility helpers and dead tests
```

**不要 squash。**

硬件重构过程中，这些 commit 本身就是非常重要的 bisect boundary。

---

# 每个阶段的验证方法

不要追求复杂 test suite。

每阶段只做三层验证。

## 1. 静态验证

```bash
ruff check .
python -m compileall dexmani_real
```

以及针对删除 abstraction：

```bash
rg "mode_transition"
rg "accepts_motion_commands"
rg "HomeRequest"
```

---

## 2. Fake SDK smoke test

只需要一个简单 FakeArm。

验证：

```text
connect 顺序正确
servo → set_servo_angle
read → get_joint_states
stop → set_state(4)
close → disconnect
```

不要建立巨大的 mock framework。

---

## 3. 真机 smoke test

每个涉及 hardware path 的 commit 后：

```text
1. clear workspace

2. connect

3. read state

4. 发送已验证的小幅度安全动作

5. 连续运行

6. DISARM → ARM

7. HOME

8. shutdown

9. E-stop
```

优先使用：

```text
低速度
明确无碰撞姿态
人工在场
```

做真实机械臂验证。

---

# 最终目标代码规模

不需要机械追求 LOC，但可以作为 architecture smell 指标。

推荐目标：

```text
xarm7.py
    150–250 LOC

arm_loop.py
    150–250 LOC

homing.py
    200–400 LOC
```

所以整个 arm runtime：

```text
约 500–900 LOC
```

是比较合理的范围。

而不是现在让：

```text
arm_sdk.py
arm_loop.py
homing.py
```

共同承担：

```text
SDK wrapper
controller verification
mode FSM
HOME RPC
state telemetry
FK
error escalation
IPC lifecycle
```

---

# 最终理想 `arm_loop`

整个重构完成后，打开 `arm_loop.py` 应该几乎可以一眼看完：

```python
def arm_loop(shared, cfg):
    arm = XArm7(cfg)

    try:
        arm.connect()

        qpos, qvel, tau = arm.read()
        publish_state(shared, qpos, qvel, tau)

        shared.set_ready("arm")

        rate = RateManager(cfg.loop_hz)

        while shared.is_running.value:
            shared.set_heartbeat("arm", time.monotonic())

            if shared.estop_request.value:
                arm.emergency_stop()
                break

            safety = SafetyState(shared.safety_state.value)

            if safety in (SafetyState.ARMED, SafetyState.RUNNING):
                command = read_latest_command(shared)
                if command is not None:
                    arm.servo(command.qpos)

            qpos, qvel, tau = arm.read()
            publish_state(shared, qpos, qvel, tau)

            rate.wait()

    except Exception:
        shared.error_state.value = True
        logger.exception("arm worker failed")

    finally:
        arm.stop()
        arm.close()
```

如果最终代码仍然需要：

```text
_ModeTransition
_Flow
StopResult
LiveStateError
RetryCounter
HomeOutcome
HomeResult
mode drift
accepts_motion_commands
controller postcondition confirmation
```

说明这次简化还没有做到位。

---

# 最重要的重构判断标准

以后遇到一个准备新增的机制，先问三个问题：

### 问题 1

> 这是机械臂实际要求，还是我们为了维护自己软件状态而创造出来的要求？

如果是后者，优先删除。

### 问题 2

> SDK 调用失败时，直接停止 experiment 是否已经足够？

如果是，就不要建立 recovery framework。

### 问题 3

> 这个 rare path 是否真的需要 realtime / asynchronous？

例如：

```text
HOME
startup
shutdown
calibration
```

通常答案都是：

```text
不需要。
```

那就允许 blocking。

---

# 推荐实际执行优先级

如果只按收益排序：

```text
P0
删除 runtime Mode FSM
↓
P1
建立薄 XArm7
↓
P2
压缩 arm_loop
↓
P3
统一 fail-fast error handling
↓
P4
删除 accepts_motion_commands / ARM_NOT_READY
↓
P5
HOME 去 RPC 化
↓
P6
FK 移出 worker
↓
P7
ARM state schema 瘦身
↓
P8
command latest-wins
↓
P9
SharedStorage / config dead-code cleanup
```

其中前 **4 步完成以后，dexmani 的 arm 控制代码复杂度就应该已经明显下降一半以上**。

后面的 IPC/schema 改造属于第二轮清理，不应该阻塞第一轮获得一个清晰、可工作的 XArm7 driver。
---

# 执行记录 — 2026-08-19 第一轮（commits 01–04）

第一轮已实施完毕（工作树未提交），全程经多代理对抗验证（每个 commit 独立审查 + 独立反驳 + 离线冒烟 + 收官三视角），并按《DexMani Real Code Style Guide.md》完成注释/布局整理。

## 已落地

| commit | 内容 | 结果 |
|---|---|---|
| 01 | 删除 runtime Mode 状态机 + `accepts_motion_commands`（schema/writer/read_arm_state_dict/safety gate/teleop quiescence/replay 同步清除）；Mode 6 startup 一次性进入后恒定；software disarm | `set_mode` 全仓库唯一调用点在 xarm7.py（startup + HOME 恢复） |
| 02 | 新薄驱动 `robot/xarm7.py`（connect/read/servo/stop/close + 过渡 enter_mode0/6、read_live_error_code、`.api`）；`arm_sdk.py` 删除；ArmLoopConfig 迁至 `config/runtime.py`；homing 改走 `arm.api`/方法 | rg "arm_sdk" 代码零残留 |
| 03 | arm_loop 删 `_Flow`、`_LoopState` 14→7 字段、`_startup` 框架压平 | arm_loop.py 1118 → 630 行 |
| 04 | fail-fast：删 `latch_arm_fault`/错误分类/C31/重试计数/read fallback；SDK 失败 raise→顶层唯一 except→error_state→finally stop/close（fire-and-forget）；controller error 每 tick 查一次缓存值；FK 失败只 state_valid=0 | 错误路径 7-9 层 → 2 层 |

净变化：+414 / −1098 行（14 个代码文件）。

## 验证期间发现并修复的真 bug

删除 `_transition_safety` 后，HOME 被 CANCELLED 会滞留 Mode 0 + State 4 永不恢复（旧机制在下个 ARMED tick 自动重进 Mode 6）。修复：`_finalize_home_result` CANCELLED 分支在 `_mode6_restore_allowed`（runtime 运行中 ∧ 无 fault/e-stop/FAULT）时恢复 Mode 6，并在结果回传前刷新 ring 帧（producer 原子可见性）。

## 有意偏离（登记）

1. **`ARM_NOT_READY` 与 `mode != 6` 发布门保留**（1.4/6.2 要求删除）：HOME RPC（阶段 7）未拆除前，该门防止 HOME Mode-0 窗口内命令积压、结束后突跳。阶段 7 完成后再删。
2. **过渡面保留**：`enter_mode0/6`、`read_live_error_code`、`XArm7.api`、`read_live_state_and_error`、`describe_controller_error`、`_CONTROLLER_ERROR_HELP` 留在 xarm7.py 供 HOME 路径使用，阶段 7 把 HOME 折叠进 `XArm7.home()` 时一并清理。
3. **`_wait_controller_ready` 保留**：等价于 1.3 允许的 `_set_mode` helper（startup/HOME 使用，带 ≤1s movable 确认）。

## 第二轮（阶段 5–12）启动前需确认

1. **episode schema 策略**：阶段 5 删 14 个 ARM_STATE 字段触及 episode v17 持久化链（episode_schema/recorder/io_process/reader/types.ArmWorkerState）——升 v18 还是仅影响新采集；docs/dataset/ 合同同步。
2. **`tracking_error` / `tracking_error_warn_rad`**：被 data_processing quality/cleaning 离线质量审计依赖——删除前定替代。
3. **`_COMMAND_COMMON_FIELDS`**：ARM/HAND 共用（schema.py）——阶段 9 删 arm 命令字段前必须拆分。
4. **`connected` 字段**：fail-fast 后恒 True——阶段 5 删除前确认 supervisor/teleop/replay 靠新鲜度兜底。
5. 死配置清理：`SafetyParams.max_consecutive_arm_health_failures`、ArmLoopConfig 投影层（阶段 11）。

---

# 执行记录 — 第二轮（阶段 6/7/8/11/12 + 5 的第一步）

第二轮按**风险最小化重排序**执行（阶段 7 先于 6，因为 `ARM_NOT_READY` 的安全删除依赖 HOME 去 RPC）：

| 阶段 | 内容 | 结果 |
|---|---|---|
| **7** HOME 去 RPC | 删 `HomeRequest`/`HomeResult`/`HomeOutcome`/`arm_home_result_q`/`HOME_SENTINEL`/`wait_for_arm_home`/`run_planned_homing` 等事务框架；新增 `XArm7.home(waypoints, final_qpos)`（Mode0→里程碑→Mode6，阻塞）；`send_arm_home` 保留签名，入队 `(waypoints, final_qpos, generation)` 并轮询 state ring 判完成 | homing.py 995→519 行 |
| **6** safety 就绪门 | `_arm_feedback_snapshot` 只留 qpos 形状/finite；删 `mode`/`connected`/`state_valid` 门与 `ARM_NOT_READY`；就绪由 runtime gate（is_running/error_state/safety_state）单一承担 | 发布门单一化 |
| **8** FK 移出 worker | ARM_STATE 删 `eef_pos`/`eef_rot6d`；新增 `planning.kinematics.make_arm_fk()`（lru_cache 单例）；6 处消费者改从 qpos 现场 FK；`validate_arm_feedback` 删 eef 参数 | arm worker 零 FK |
| **5**（第一步） | ARM_STATE 再删 `mode`/`timestamp`/3 个 `last_cmd_*_s` 时间戳 → **22 字段降至 14**；状态年龄检查改用 `source_monotonic_ns`（单调钟，语义更准） | 无数据合同影响 |
| **11** 配置清理 | 删 `ArmLoopConfig.max_consecutive_arm_health_failures`/`tracking_error_warn_rad` 投影字段与 `SafetyParams.max_consecutive_arm_health_failures`（fail-fast 后全无消费者） | 死配置归零 |
| **12** SafetyState | 校验确认：arm/hand worker 只读 `safety_state`，`transition()` 仅由 main/入口调用，无 worker 侧状态机 | 已天然满足 |

## 关键设计决策

1. **`HomeAborted` 异常分层**：审查发现「把 e-stop/DISARM/generation 变化统一 raise」会误置 sticky `error_state`（丢失旧 CANCELLED 可恢复路径）。修复：`XArm7.home()` 对运行时中断抛 `HomeAborted`，worker catch 后 stop + 按 `_mode6_restore_allowed` 恢复 Mode 6 并继续（不置 fault）；硬件失败仍抛 `RuntimeError` → fail-fast。
2. **HOME 请求携带 generation**：审查发现三元组失去 generation 会让被放弃的 stale HOME 在下次 ARMED 时意外执行。修复：入队 `(waypoints, final_qpos, generation)`，worker 消费时校验并丢弃 stale。
3. **HOME 完成检测**：无结果队列后，改为轮询 state ring——「请求后新发布的帧 + qpos 收敛 + qvel 已静止」。里程碑帧因速度未静止被自然过滤。
4. **FK 数值等价性已证明**：新 `make_arm_fk()` 与旧 arm_loop 的 `ArmFK` 同 URDF、同 `custom_eef_link`，200 组随机 qpos 的 EEF 最大差异 = 0.0。

## 未做（阶段 5 的第二步 + 9/10，需你决策）

以下删除会触及 **episode v17 数据合同**，需先确认策略：
- `connected`/`publish_monotonic_ns`/3 个延迟字段/`is_hold`/`tracking_err` → 这些被 `episode_schema.py` 持久化（`arm_connected`/`arm_last_cmd_*`/`arm_publish_monotonic_ns`/`tracking_error`），删除即 schema 变更（v17→v18？旧 episode 是否保留读取兼容？）。
- 阶段 9（latest-wins `arm_cmd_ring` + 命令 schema 瘦身）需先拆分 `_COMMAND_COMMON_FIELDS`（ARM/HAND 共用），且会改写 AGENTS.md §3.6「arm queue 有序」不变量。
- 阶段 10 的 SharedStorage 清理在阶段 5/9 完成后才有剩余死代码。

---

# 执行记录 — 第二轮收尾（阶段 5 / 9 / 10 完成，12 阶段全部落地）

按决策「升 v18 + 旧 v17 保留读取兼容」完成剩余阶段。

## 阶段 5（ARM_STATE 瘦身 + episode v18）

`ARM_STATE_DTYPE` **22 → 11 字段**：`qpos/qvel/tau/error_code/connected/tracking_err/last_cmd_seq/last_cmd_is_hold/source_monotonic_ns/publish_monotonic_ns/state_valid`。

- 删除：`eef_pos`/`eef_rot6d`（阶段 8）、`mode`、`timestamp`、3 个 `last_cmd_*_s` 时间戳、3 个 `last_cmd_*_latency/duration`。
- **保留（审查后收回删除决定）**：`connected` 与 `publish_monotonic_ns` 不是诊断字段——后者是 `shm/causal_reader.py`、deployment observation history、recording 样本对齐的**因果选择依据**；前者是共享反馈健康谓词的活跃输入。
- 状态年龄检查（teleop 状态打印、home 前置检查）从墙钟 `timestamp` 改为单调 `source_monotonic_ns`，语义更准。
- **episode schema v17 → v18**：v18 只删 3 个 arm 命令延迟数据集（它们度量的是 arm worker 内部时序，fail-fast worker 不再产出）。`arm_connected`/`arm_last_cmd_seq`/`arm_last_cmd_is_hold`/`tracking_error`/`arm_publish_monotonic_ns` 全部保留，故 `data_processing` 质量/清洗规则**零改动**。
- **读取兼容**：`SUPPORTED_EPISODE_SCHEMA_VERSIONS = {17, 18}`；`required_dataset_names(version)` 按版本切换必需集；校验器新增 `schema_version` 参数。三种组合已验证：v18 数据按 v18 规则、v17 数据按 v17 规则、v17 数据按 v18 规则（多余数据集被容忍）均无错误。

## 阶段 9（latest-wins transport）

- `arm_action_q`（有序 Queue，servo + HOME 混载）拆为两条通道：**`arm_cmd_ring`**（`SharedMemoryRingBuffer`，latest-wins，maxlen 4）承载 servo endpoint；**`arm_home_q`**（小队列）承载 HOME 请求（HOME 是 Python 对象元组，无法进固定 dtype ring）。
- **`_COMMAND_COMMON_FIELDS` 已拆分**：`ARM_COMMAND_DTYPE` 瘦身为 `action_id/created_monotonic_ns/is_hold/qpos_cmd`（4 字段）；`HAND_COMMAND_DTYPE` 保持原有 7+1 字段与 generation/有效期协议不变。
- **stale 防护改为 `armed_at_seq`**：新增 `shared.arm_armed_at_seq`，在 `robot/safety.transition()` 的 `DISARMED→ARMED` 边沿自动记录当前 `arm_command_seq`（单一落点，10 个入口零改动）；worker 只接受 `action_id > armed_at_seq` 且命令年龄 ≤ 0.3s 的 endpoint。
- 删除 `ARM_QUEUE_FULL`（latest-wins 无背压）；`_worker_command_is_current` 保留给 hand 路径。

## 阶段 10（SharedStorage 清理）

`read_arm_state_dict` 与 `ARM_STATE_DTYPE` 一致性闭合（11/11，无读取不存在字段、无未暴露字段）；资源名单更新为 ring=`arm_cmd_ring`、queue=`arm_home_q`。

## 合同影响（需同步 AGENTS.md §3）

- §3.6「arm queue 是有序 `maxsize=2`」→ **arm 命令环是 latest-wins**（与 hand 一致）。
- §3.13 `run_generation`：arm servo 路径不再携带 generation（改 `armed_at_seq` + 命令年龄）；HOME 请求仍携带 generation；hand 路径 generation 协议不变。
- episode 合同：runtime 写 v18，reader 接受 v17/v18。

## 验证

compileall 通过、`git diff --check` 干净；离线验证全绿：FK 等价性（差异 0.0）、v18/v17 三组合校验、阶段 9 transport 13/13、阶段 10 一致性、round-2 冒烟 20/20、全运行时模块 import OK。**真机验证未执行（需授权）**。
