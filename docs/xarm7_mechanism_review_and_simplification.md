# xArm7 机制审查与简化实施方案

## 1. 文档目的

本文审查 DexMani Real 的 xArm7 控制机制，并给出一套以修复逻辑错误、减少隐式状态和收敛故障行为为目标的实施方案。

本文只讨论软件机制和离线可验证契约，不代表真机参数已经完成验证。任何遥操作、回零、回放、标定或控制器配置验证，仍须遵守仓库根目录 `AGENTS.md` 的硬件授权要求。

2026-08-16 的复核结论：本文的问题定位可以成立，但实施时必须同时满足 HOME generation 捕获、取消后的停机确认、SafetyState 写入所有权、SDK 缓存/同步数据源区分以及原子迁移 HomeResult 协议等约束。以下方案已把这些防回归条件写入，不能只摘取某一条局部修改。

本文的目标不是重写机械臂栈，而是保留当前正确架构，并把控制器行为压缩为：

```text
Mode 0 只负责已规划回零
Mode 6 只负责 absolute joint target streaming
State 4 负责软件侧控制器停止
任意运行期控制器错误进入单一粘性故障路径
```

## 2. 范围与证据边界

### 2.1 审查范围

主要调用链：

```text
config/defaults.py
        ↓
robot/arm_sdk.py
        ↓
robot/arm_loop.py ← arm_action_q / run_generation / SafetyState
        ↕
robot/homing.py
        ↓
xArm Python SDK
```

同时检查：

- `shm/shared_storage.py` 中的 HOME 请求与结果；
- `utils/schema.py` 中的臂命令和状态 dtype；
- `policy/safety.py` 中的 generation 和硬件边界验证；
- `recording/` 中的命令时延持久化；
- `checks/offline/` 中的 FakeArm 和相关离线检查；
- xArm7 URDF、TCP load 和手部静态变换。

### 2.2 证据优先级

结论按以下优先级判断：

1. UFACTORY 官方 SDK、ROS 和 teleoperation 实现；
2. 本项目实际安装的 xArm Python SDK 1.18.4；
3. [`docs/xarm7/xarm7_mode6.md`](xarm7/xarm7_mode6.md) 对官方实现的归纳；
4. [`docs/xarm7/lerobot_xarm7.md`](xarm7/lerobot_xarm7.md) 和 [`docs/xarm7/xarm7-.md`](xarm7/xarm7-.md) 对其他项目的审计。

后两份项目审计适合发现反模式，不能直接作为 DexMani Real 的控制契约。

本项目当前执行契约固定在已安装的 xArm Python SDK 1.18.4。上游仓库 `master` 会变化，只用于补充背景；升级 SDK 时必须重新核对参数钳制、state/mode 报告方式、错误码和设备类型，不能沿用本文对 1.18.4 的实现细节。

### 2.3 官方参考

- [xArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- [xarm_ros Mode 6 说明](https://github.com/xArm-Developer/xarm_ros)
- [UFACTORY teleoperation](https://github.com/xArm-Developer/ufactory_teleop)
- [SDK 控制器错误码](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/core/config/x_code.py)
- [SDK 设备类型与关节限制](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/core/config/x_config.py)

## 3. xArm7 Mode 6 的事实契约

### 3.1 Mode 6 的输入语义

Mode 6 是关节空间在线轨迹规划模式。上位机发送的是最新绝对关节目标：

```text
q_actual / qdot_actual
        +
absolute q_target
        +
speed / mvacc constraints
        ↓
xArm controller online trajectory generation
```

正确的流式调用为：

```python
arm.set_mode(6)
arm.set_state(0)

arm.set_servo_angle(
    angle=q_target,
    speed=max_joint_speed,
    mvacc=max_joint_acc,
    is_radian=True,
    wait=False,
)
```

`speed` 和 `mvacc` 是控制器规划约束，不是逐关节速度或加速度命令。Mode 6 不等价于 Mode 1 `set_servo_angle_j()`，也不等价于碰撞规划器。

Mode 6 要求固件至少为 1.10.0。

### 3.2 Mode 0 与 Mode 6 的职责

官方实现反复采用以下分工：

```text
Mode 0 + State 0
    → 确定的初始定位或回零

Mode 6 + State 0
    → 不断更新 absolute q_target
```

DexMani Real 使用 Mode 0 执行碰撞检查后的稀疏回零里程碑，使用 Mode 6 执行遥操作、键盘控制和策略部署。这一结构正确，应保留。

### 3.3 控制器 state 语义

需要区分“设置命令”和“报告状态”：

- `set_state(0)`：请求进入 motion/ready 状态；
- State 1：运动中；
- State 2：sleeping/idle；
- State 3：suspended；
- State 4：stopping/stopped；
- SDK 内部还把 State 5 视为不可运动状态；未知状态也必须 fail closed。

SDK 1.18.4 的公开 `get_state()` 文档主要列出 1–4，但内部状态与报告路径可能短暂暴露 0。项目可为该固定版本接受 0/1/2 作为“未处于暂停或停止”的兼容集合；这不代表 State 0 是静止状态，也不能把它写成跨 SDK 版本的通用常量。

数据源也不同：`get_state()` 和 `get_err_warn_code()` 是同步读取；`arm.mode` 与 `arm.connected` 是 SDK 属性，其中 mode 依赖 report 线程，SDK 1.18.4 没有对等的同步 `get_mode()`。因此不能把一次组合读取描述成原子 live snapshot。

因此，`set_state(0)` 成功后不能要求控制器最终只报告 State 2。State 2 可以是合法空闲状态，但不是唯一就绪状态。

建议的统一判定为：

```text
可接受运动：connected report is healthy
          and synchronous error == 0
          and expected cached mode has settled
          and synchronous state in {0, 1, 2}

确认停止：state == 4
```

若某个流程要求机械臂静止，应使用 fresh `qvel` 和 dwell 时间判断，而不是用 State 2 代替速度收敛条件。

### 3.4 参数能力与示例参数

当前 Python SDK 对普通关节运动参数执行钳制。SDK 1.18.4 的初始下限以及命令路径硬上限为：

```text
0.0001 rad/s <= speed <= π rad/s
0.01 rad/s²  <= mvacc <= 20 rad/s²
```

下限属性可能由控制器 report 更新；命令路径的上限仍硬钳制为 π rad/s 和 20 rad/s²。配置检查应拒绝会被当前 SDK 改写的值，而不是只检查正数或只检查上限。

UFACTORY GELLO xArm7 示例使用约：

```text
30 Hz
90 deg/s
500 deg/s²
```

这些是应用示例，不是 Mode 6 协议常量。DexMani Real 当前的 120 deg/s、900 deg/s² 仍在 SDK 能力范围内，不能仅因为官方示例较低就自动替换；真实安全值取决于 payload、安装方向、工作空间、Reduced Mode 和风险评估。

## 4. 当前项目中应保留的正确机制

以下机制与官方语义及仓库架构一致，不应在简化过程中破坏：

1. `SharedStorage` 是唯一跨进程数据面；
2. SDK 对象只存在于 arm worker；
3. Mode 6 接收 absolute 7-DoF joint target；
4. 使用 `set_servo_angle()`，显式传入 speed、mvacc、`is_radian=True` 和 `wait=False`；
5. 不在应用层对 Mode 6 命令增加插值；
6. arm queue 保持有序且有界；
7. `run_generation` 使旧运行周期的普通关节命令失效；
8. 回零路径先在上层完成关节限位、碰撞和工作空间检查，再由 Mode 0 固件规划器执行里程碑；
9. 状态环发布 fresh qpos、qvel、tau、FK、错误码、连接状态、模式和命令时延；
10. 录制同时保留 command 和 actual state，而不假设 `q_cmd == q_actual`；
11. DISARMED、FAULT、E-stop 后备和最终退出使用 State 4；
12. `error_state` 保持粘性。

## 5. Fact-check 后确认的问题

### 5.1 P0：Mode 6 就绪条件被错误收窄为 State 2

当前 `_enter_mode6_ready()` 执行：

```text
set_mode(6)
set_state(0)
wait state == 2
```

启动恢复循环也使用同一假设。测试 FakeArm 又把 `set_state(0)` 人为转换为 State 2，因此离线检查无法发现该问题。

影响：

- 合法的 State 0/1 可能被误判为启动失败；
- 测试验证的是项目假设，而不是 SDK 语义；
- “ready” 与“sleeping”被混为一谈。

修复原则：以 connected、error、mode 和可接受状态集合判断控制器是否可运动；需要静止时另查 qvel。

### 5.2 P0：HOME 请求没有 generation

普通臂命令在 worker 边界复核 `run_generation`，但 `HomeRequest` 只包含：

```text
request_id
waypoints
final_qpos
execution_timeout_s
```

HOME tuple 会绕过普通 ndarray 命令的 generation 验证。因此，一个已排队但已失效的 HOME 请求仍可能开始运动。

修复原则：`HomeRequest` 必须携带创建时的 `run_generation`，worker 在切换 Mode 0 前和每个执行周期内都要复核。

generation 必须使用 `advance_run_generation(shared)` 的返回值，在失效旧命令的同一逻辑边界捕获。不能在规划结束或入队前重新读取当前值，否则规划期间发生的 pause、DISARMED 或其他 generation 变化会被错误地包装成“当前请求”。入队前还要再次比较共享值；不一致时直接取消，不能发布 HOME。

### 5.3 P0：Homing 内层不响应 DISARMED

Homing 执行期间 arm loop 被同步阻塞在 `run_planned_homing()` 中。其 abort helper 当前检查：

```text
shutdown
E-stop
sticky error
FAULT
```

但没有检查：

```text
SafetyState.DISARMED
run_generation changed
```

因此外层 arm loop 的 DISARMED → State 4 门控无法在 Homing 返回前执行。

修复原则：Homing 每次发送里程碑前、等待反馈时和 dwell 时均检查 safety state 与 generation；取消后立即请求并确认 State 4。

### 5.4 P0：Homing 失败后仍可能恢复 Mode 6

当前恢复条件只要求：

```text
no global abort
and controller error == 0
```

它没有要求 `HomeResult.success`。因此未收敛、超时或普通执行失败可能先恢复 Mode 6，随后 arm loop 才设置 sticky error。

修复原则：HOME 只允许从 ARMED 发起；当前调用方本就应先退出 RUNNING。进入 Homing 前先将 worker 内部的 `accepts_motion_commands` 置为 false。只有 Homing 成功、仍处于 ARMED、generation 未变化且 Mode 6 恢复后置条件已确认时，才重新置为 true；其他出口保持 false 并确认 State 4。

### 5.5 P1：命令时延采样顺序错误

当前成功路径在 SDK 调用返回后才记录 `last_cmd_received_s`，造成：

```text
queue latency = producer → SDK return
apply latency = producer → SDK return
```

两者失去区分，而且 queue latency 包含 SDK 执行时间。

正确时序：

```text
dequeue
  received_s = monotonic()
  parse created_s

before SDK call
  sdk_started_s = monotonic()

successful SDK return
  applied_s = monotonic()

queue_latency = received_s - created_s
sdk_duration  = applied_s - sdk_started_s
apply_latency = applied_s - created_s
```

该修复不需要修改 dtype 或 HDF5 schema。

### 5.6 P1：设备身份只记录、不校验

项目把 device type、firmware 和 SN 写入 identity JSON，但：

- 没有记录或验证 axis；
- 没有检查 Mode 6 最低固件版本；
- `model` 被直接写成 `xArm7`；
- 没有验证 device type 是否与当前 URDF 和关节限制一致。

当前关节限制与 SDK 的 `XARM7_X4` 一致，而较新规格值更接近 `XARM7_X4_1305`。正确做法不是覆盖限制，而是先验证连接的真实设备类型。

### 5.7 P1：配置允许 SDK 静默钳制

项目配置验证允许：

```text
speed <= 500 deg/s
acc   <= 50000 deg/s²
```

SDK 实际上会把普通关节命令钳制到约 180 deg/s 和 1145.9 deg/s²。配置因此可能通过验证，但真实控制参数与元数据声明不一致。

修复原则：在配置边界拒绝超出当前 SDK 契约的值，不依赖 SDK 静默修改。

### 5.8 P1：运行期没有把 mode/state 漂移纳入门控

项目持续处理 controller error，但 mode 主要用于状态发布，state 没有进入状态环。SDK 对模式不匹配可能只记录 warning，并不总是返回失败。

修复原则：arm worker 每周期读取 SDK 属性中的 connected/mode 和现有反馈，发现异常后按字段使用正确的确认方式：state/error 通过同步 API 读取；mode 没有同步 getter，因此在通信与 joint feedback 仍健康时给 report cache 一个短的有界更新窗口，持续不匹配才 fail closed。一次缓存 mode 不匹配不能立即触发故障，Mode 0/6 切换的预期过渡也不能被当作漂移。

该检查可以完全留在 arm worker 内部，无需立即增加 IPC 字段或修改 v16 schema。

### 5.9 P2：确定存在的死代码和误导性注释

可直接清理：

- `_state_read_warn` 创建后未使用；
- `_recovery_counter` 只 reset，从不 increment；
- `low`、`high` 数组创建后未使用；
- wrap 失败后给局部 `target` 赋 measured pose，但实际不发送且不记录；
- `max_consecutive_recoveries` 实际还被 feedback/FK watchdog 使用，名称与用途不一致；
- `set_joint_maxacc` 上方注释错误地称其为碰撞检测；
- `emergency_stop()` 被描述成切断电机电源，但当前 SDK 实现只是请求并等待 State 4。

## 6. 需要明确的设计取舍

### 6.1 C24：事实与推荐策略

官方错误定义：

```text
C24 = Speed Exceeds Limit
```

官方排查方向包括检查工作范围以及降低 speed/mvacc。官方资料没有定义“清错后按原参数发送 measured hold”为标准恢复流程。

当前项目的 C24 恢复是自定义策略：

```text
clean_error
clean_warn
motion_enable
enter Mode 6
read fresh qpos
send one measured hold
second C24 within 2 s → sticky fault
```

这不是无限重试，因此不能仅凭文档定性为已知逻辑错误。但它形成了两套重复分支，并且恢复后仍使用原 Mode 6 speed/mvacc，不能解决 C24 的根因。

为满足本项目“行为简单、失败明确”的目标，本文推荐：

```text
任意运行期 controller error 或终止性 SDK/API failure
        ↓
先设置 sticky error_state，记录 API code、controller error、mode/state 和 last target
        ↓
尝试请求并确认 State 4
        ↓
由 Main/supervisor 将共享 SafetyState 转为 FAULT
```

SDK API code 与 controller error code 是两个命名空间，必须分别保存。例如 `set_servo_angle()` 返回非零但 live controller error 为 0，仍是终止性命令失败，不能因“没有 Cxx”而继续。反过来，故障路径不得等到停机确认后才设置 `error_state`；即使停机调用抛异常，粘性故障也不能丢失。

controller warn code 也不是 controller error code。单独出现 warn 时保留诊断并继续按 SDK API 返回值判断，不能把 warn 数字误套入 Cxx 分支；本文不要求 servo loop 自动 `clean_warn()`。

State 4 helper 是故障清理特例：即使 `set_state(4)` 返回非零或抛异常，只要通信仍可用，就应继续做有界的同步 `get_state()` 确认。确认到 State 4 表示“停止已确认”，不表示原故障已恢复；确认失败必须保留 sticky fault 并作为 cleanup failure 上报。

C22、C24、C31 只改变诊断文本，不改变控制分支。由操作员排除物理原因后，通过 Studio 或单独的显式维护流程清错；arm worker 不做隐式自动恢复。

采用该策略后可删除：

- `recoverable_errors` 与 `collision_fault_errors` 行为配置；
- C24 两秒窗口；
- `_recover_c24_measured_hold()`；
- setter-return 和 state-stream 中的两套 C24 分支；
- replay 对 C24 恢复的等待；
- C24 专用自动恢复测试。

### 6.2 启动清错

官方 demo 通常在初始化时调用 `clean_error()` 和 `clean_warn()`，但这属于方便示例，不等于生产系统必须无条件清除已有故障。

当前项目最多重复三次完整清错、使能和状态切换，既复杂又会覆盖原始故障上下文。

推荐的生产行为：

1. 连接后先读取并记录 live error/warn；
2. 若 controller error 非零，则先设置 sticky error、尽力确认 State 4，不发布 ready，然后退出 worker；
3. error 为 0 时仍执行一次明确的 motion enable、Mode 0/State 0 和控制器配置流程，完成后回到并确认 State 4，再发布 DISARMED ready；
4. 正常启动不需要重复清错；只读连接/report 重试与状态变更重试必须分开；
5. 清错由明确的维护操作完成，不放在普通 servo loop 中。

如果现场流程坚持允许“程序启动即确认清错”，也应只执行一次，并在清除前完整记录原始错误；通信读取重试不能重复触发控制器状态变更。

### 6.3 `set_joint_maxacc`

`set_joint_maxacc()` 设置控制器全局关节加速度上限；逐命令 `mvacc` 设置当前运动规划约束。两者不完全等价。

短期建议保留全局上限，但：

- 修正错误注释；
- 将其命名为 controller global acceleration cap；
- 不与 Mode 6 command mvacc 混称；
- 明确两者当前使用相同数值是项目安全选择，而不是 SDK 要求。

只有在 Reduced Mode 等机器人侧全局限制完成显式验证后，才评估是否删除这一调用。

### 6.4 0.033 m 与 0.043 m 末端模型

当前存在以下事实：

- 两份组合 URDF 的 `flange_joint2` 为 0.043 m；
- 配置说明物理测量值为 0.033 m；
- 手部 FK 通过 `T_eef_handbase_pos_xyz` 额外补偿 -0.010 m；
- ArmFK、IK 和碰撞模型仍以 URDF 的 `custom_eef_link` 为准。

这可能是刻意保留虚拟 EEF，也可能使真实手部碰撞几何偏移 10 mm。本文不选择任意一侧作为“正确值”。

处理顺序：

1. 固化法兰、`custom_eef_link`、hand base 的坐标定义；
2. 核对物理测量记录；
3. 离线计算当前两条变换链的末端坐标；
4. 决定保留虚拟 frame 或更新 URDF；
5. 同步更新 collision model、TCP load COG、手部 FK 和模型 provenance。

该事项不得与控制器状态修复混在同一个补丁中。

## 7. 目标控制器模型

不新增共享安全枚举。共享 SafetyState 继续由 Main 与 policy 按既有职责分权，arm worker 只维护内部控制器阶段：

```text
                 ARMED / RUNNING
              ┌──────────────────┐
              │                  ▼
DISARMED ──► STOPPED ───────► STREAMING
  ▲           State 4          Mode 6 + movable state
  │                               │
  │                               │ HOME request
  │                               ▼
  └──────── cancellation ◄──── HOMING
                              Mode 0 + movable state

any controller/SDK/feedback terminal failure
                    ↓
              FAULT_STOPPED
                 State 4
```

约束：

- `STOPPED`：必须确认 State 4；
- `STREAMING`：Mode 6、error 0、state 可运动；
- `HOMING`：Mode 0、请求 generation 仍有效、SafetyState 仍为 ARMED；
- `FAULT_STOPPED`：先设置 sticky error，再尽力确认 State 4，不自动清错；
- Main 拥有 DISARMED ↔ ARMED、转入 FAULT 和 shutdown，policy 拥有 ARMED ↔ RUNNING；arm worker 不写共享 SafetyState。

## 8. 最小实现方案

### 8.1 `robot/arm_sdk.py`

保留只依赖 SDK 表面的纯 helper，避免 `arm_loop.py` 与 `homing.py` 相互导入：

```python
read_live_state_and_error(arm) -> LiveStateError
read_report_mode_and_connection(arm) -> ReportSnapshot
require_sdk_ok(operation, code) -> None
controller_state_allows_motion(state) -> bool
enter_mode0(arm, *, on_poll=None) -> None
enter_mode6(arm, *, on_poll=None) -> None
stop_controller(arm, *, emergency=False, on_poll=None) -> StopResult
```

两种结果不能合并成名为 “live” 的原子对象：

```text
LiveStateError: synchronous state / error_code / warn_code
ReportSnapshot: cached connected / mode
```

`enter_mode0/6` 使用有界等待组合这些信息，并给 report mode 留出更新窗口。steady-state mode mismatch 在健康通信下持续超过该窗口才确认；controller error、State 4/5、未知 state 和断连仍 fail closed。重复读取同一个 cached mode 不能被描述成多个独立 live 样本。

这些状态转换 helper 必须放在 `robot/arm_sdk.py` 或另一个不依赖 arm loop/homing 的 leaf module，使 arm loop 与 homing 都能复用而不形成循环导入。不在该模块中读写 SharedStorage、不实现策略恢复，也不把 C24 分类写进配置 dataclass。

任何有界等待若可能接近 arm heartbeat timeout，都要在轮询间调用可选 `on_poll`；arm loop/homing 传入 heartbeat callback，leaf helper 不直接依赖 SharedStorage。SDK 单次调用本身若不可中断，仍由 supervisor 的 worker/heartbeat 机制兜底。

### 8.2 `robot/arm_loop.py`

arm loop 通过 leaf SDK helper 执行控制器状态变化，并在本模块保留唯一共享 fault helper：

```python
latch_arm_fault(shared, arm, reason, *, api_code=None, controller_error=None)
```

组合后的行为必须满足：

- `enter_mode0/6` 检查 setter 返回码，并等待 expected mode、error 0 和可运动 state；
- `stop_controller` 不要求 error 为 0；setter 失败后仍尝试同步确认 State 4；
- E-stop 仍优先调用 SDK `emergency_stop()`，随后使用同一个 State 4 确认逻辑；SDK 1.18.4 的该方法不提供可交给 `require_sdk_ok()` 的整数返回码；
- `latch_arm_fault` 先写 sticky `error_state`，再尝试停机，绝不调用 `clean_error()`；
- helper 不修改共享 SafetyState。

推荐循环骨架：

```python
while shared.is_running.value:
    heartbeat()

    if estop_requested():
        stop_and_exit()

    enforce_safety_state_edge()
    monitor_cached_controller_health()

    if not allowed_to_accept_commands():
        publish_feedback()
        rate_limit()
        continue

    taken = take_current_generation_action()
    if taken is not None:
        action, received_s = taken
        if action is HOME:
            execute_home()
        elif action is joint_target:
            apply_joint_target()

    read_and_validate_feedback()
    publish_feedback()
    rate_limit()
```

普通关节命令路径固定为：

```text
dequeue timestamp
→ dtype/generation/expiry validation
→ nearest-equivalent conversion
→ absolute target boundary validation
→ set_servo_angle(wait=False)
→ return-code check
→ applied timestamp
→ health check
```

wrap 失败时明确记录并丢弃，不赋一个不会发送的伪 hold target。

`received_s` 必须在 queue `get()` 返回后立即记录并与 action 一起返回；只在命令最终成功应用后更新“last successful command”指标。进入 HOME 前立即把 `accepts_motion_commands` 置为 false，不能沿用进入 Homing 前的 STREAMING 标志。

### 8.3 `shm/shared_storage.py`

`HomeRequest` 增加：

```python
run_generation: int
```

producer 使用以下顺序：

```python
home_generation = advance_run_generation(shared)
fresh_qpos = wait_for_post_invalidation_feedback()
waypoints = plan_from(fresh_qpos)
if int(shared.run_generation.value) != home_generation:
    return False
queue(HomeRequest(run_generation=home_generation, ...))
```

不要在规划完成后读取一个新的 generation 填入 request，也不要持有 `run_generation` 的共享锁跨越规划、queue put 或 SDK 调用。请求中的数组必须在入队前 copy，并在 producer/worker 两端做 shape/finite 校验。

为避免依靠 reason 字符串判断失败性质，`HomeResult` 使用唯一 outcome 字段：

```text
SUCCESS
CANCELLED
FAILED
```

不要同时保存可相互矛盾的 `success: bool` 和 `outcome`。若需要兼容当前调用点，可提供只读 `success` property：`return self.outcome is HomeOutcome.SUCCESS`。枚举、producer、worker、等待方和离线 fake 必须在同一个补丁中原子迁移。

为保持当前外部调用面简单，`send_arm_home()` 可以继续返回 bool：等待方消费 outcome 后只把 SUCCESS 映射为 true。关键顺序是 arm worker 对 FAILED 先写 sticky `error_state`，再把 HomeResult 放入结果队列，使仍依赖 bool + shared fault 的现有调用方不会把失败竞态误判为普通 REJECTED。若决定让 `send_arm_home()` 返回 outcome，则必须同时迁移 `teleop/safety.py`、camera calibration、keyboard teleop、replay 和所有离线检查，不能只改函数签名。

HomeResult 发布也是控制协议的一部分。queue put 超时或抛异常时，worker 必须先关闭命令接收、设置 sticky error 并停到 State 4，不能在调用方未收到结果时继续 STREAMING。调用方随后可通过 sticky/global fault 结束等待。

该对象只通过现有 multiprocessing queue 传递，不新增共享 ring，也不影响 HDF5 v16。

### 8.4 `robot/homing.py`

统一 abort helper。若多个条件同时成立，FAILED 优先于 CANCELLED：

```text
SDK/controller failure   → FAILED + State 4
sticky/global FAULT      → FAILED + State 4
E-stop                   → CANCELLED + State 4
shutdown                 → CANCELLED + State 4
SafetyState != ARMED     → CANCELLED + State 4
generation mismatch      → CANCELLED + State 4
```

generation 检查至少位于：worker 接受请求前、切换 Mode 0 前、每个里程碑 SDK 调用前、里程碑反馈等待循环和最终 dwell。generation 在 SDK 调用刚被接受后改变是不可消除的边界竞态；下一次轮询必须立即请求 State 4，不能用跨网络调用的共享锁规避它。

arm loop finalizer 的核心规则：

```python
if result is SUCCESS and request_is_still_current_and_armed():
    enter_mode6(arm)
else:
    stop_controller(arm)
```

`run_planned_homing()` 只负责校验请求、进入 Mode 0、执行里程碑并返回 provisional outcome；它不自行恢复 Mode 6，也不发布结果。arm loop 恢复调用栈后作为唯一 finalizer 执行以下规则：

1. provisional FAILED：先设置 sticky error，再尝试并确认 State 4；
2. provisional CANCELLED：确认 State 4；确认失败则升级 FAILED 并设置 sticky error；
3. provisional SUCCESS：复核 generation/SafetyState 后恢复 Mode 6；恢复失败升级 FAILED 并走 sticky + State 4；
4. 完成 finalization 后才发布 HomeResult。

只有 Mode 6 后置条件确认成功，最终 outcome 才能保持 SUCCESS。CANCELLED 只有在 State 4 已确认时才是正常请求结果；停机无法确认时升级为 FAILED 并保持 sticky fault。

恢复 Mode 6 前后都要复核 generation 和 SafetyState，并要求 SafetyState 仍为 ARMED。若 generation 或无 fault 的生命周期状态在 `enter_mode6()` 调用期间改变，则立刻重新进入并确认 State 4，并把结果改为 CANCELLED；若此时已出现 sticky/global FAULT，或停机无法确认，则改为 FAILED。

arm loop 只因最终 FAILED 新增 arm execution fault。CANCELLED 不清除也不遮蔽已有的 E-stop、global FAULT 或 sticky error，Main/supervisor 仍按原契约处理这些全局信号。任何非 SUCCESS 结果都保持 `accepts_motion_commands=false`。

### 8.5 `config/defaults.py`

修改：

- speed/mvacc 同时校验上下限，拒绝会被固定 SDK 1.18.4 静默钳制的配置；
- SDK 命令路径上限使用 π rad/s 和 20 rad/s² 精确常量，不用四舍五入后的角度值参与比较；
- 连接后读取并校验有效的 `joint_speed_limit`/`joint_acc_limit`，使用项目上限、SDK 命令硬上限和设备报告范围的交集；
- 删除 `recoverable_errors` 与 `collision_fault_errors` 行为配置；
- 保留诊断用 controller error name mapping，不把错误码集合变成行为配置；
- 删除未使用的 servo recovery counter，将现有 30-cycle 配置重命名为单一 `max_consecutive_arm_health_failures`；feedback 与 FK 仍使用各自独立 counter，不能互相 reset；
- 增加预期 axis、显式 device profile、可选 SN 和最低 firmware 配置。

axis 必须为 7；固件版本使用 SDK `version_number` 整数 tuple 比较，不能比较版本字符串。device profile 必须在核对真机后显式选择，并绑定对应关节限制/URDF；不能把当前 limits 反推成真实 device type 后作为默认事实。SN 只有配置时才强校验。必需身份字段应在 bounded report readiness 后读取，不能把连接瞬间的 `unavailable` 永久写入 metadata。

不修改当前 120/900 默认值；参数调整属于单独的真机验证任务。

### 8.6 状态与录制路径

当前 `ARM_STATE_DTYPE` 已包含 mode。短期只需：

- `read_arm_state_dict()` 返回现有 mode；
- 内部监控 controller state，不新增 dtype 字段；
- 修复已有命令时延的采样点；
- 保留当前 v16 dataset 名称和含义。

`read_arm_state_dict()` 增加 mode 只是暴露既有字段；所有使用者必须继续容忍旧离线 fixture 缺少该 dict key，除非 fixture 与消费者在同一补丁中更新。controller state 保持 worker 内部信息，不能为了方便直接改 `ARM_STATE_DTYPE`。

只有明确需要离线分析 controller state 时，才按 dtype → reader → recorder → analysis → v16 contract 的完整链路增加字段。

## 9. 不应从官方示例机械复制的内容

以下信息是应用配置，不是 Mode 6 协议：

- 90 deg/s；
- 500 deg/s²；
- 30 Hz；
- 前 20 条命令固定 0.2 rad/s；
- 第一条命令的具体同步方式；
- start joints；
- collision sensitivity；
- Reduced Mode 边界；
- mounting direction；
- payload 数值。

DexMani Real 已经有 planned homing、fresh feedback、generation 和 producer-owned command semantics，不应再添加一个隐式“前 N 帧特殊状态机”。如果第一条目标需要额外保护，应在 producer/gate 边界基于 fresh measured qpos 明确拒绝过大的首次跳变，而不是在 arm worker 中插值。

## 10. 分阶段实施顺序

### 阶段 A：生命周期正确性

修改范围：

- `robot/arm_sdk.py`
- `robot/arm_loop.py`
- `robot/homing.py`
- `shm/shared_storage.py`
- `checks/offline/_fakes.py`

内容：

1. 删除 State 2 唯一就绪假设；
2. 修正 FakeArm；
3. 在 `advance_run_generation()` 返回处捕获 HOME generation，并在规划后、入队前复核；
4. Homing 响应 DISARMED/generation；
5. Homing 前关闭 `accepts_motion_commands`，失败和取消后确认 State 4；
6. `run_planned_homing()` 不再自行恢复 Mode 6，由 arm loop 单点 finalization；
7. 只有成功且仍有效、Mode 6 后置条件已确认时恢复命令接收；
8. 将 HomeResult 原子迁移为单一 SUCCESS/CANCELLED/FAILED outcome。

### 阶段 B：单一故障路径

修改范围：

- `robot/arm_loop.py`
- `robot/arm_sdk.py`
- `config/defaults.py`
- `config/runtime.py` 与配置 dump/override 路径
- `examples/replay_episode.py`
- `examples/calibrate_camera.py`
- `examples/keyboard_teleop.py`
- `checks/offline/check_arm_c24_recovery.py` 及相关配置检查
- README/CLAUDE 中的 C24 行为说明

内容：

1. 删除运行期 C24 自动恢复；
2. 合并 setter failure 与 state-stream error 分支；
3. 任意终止性 SDK/API 或 controller error 先 sticky fault，再尝试 State 4；
4. 启动不重复清错；
5. 从 ArmParams/ArmLoopConfig 删除 `recoverable_errors` 与 `collision_fault_errors`，仅保留错误码到诊断文本的映射；
6. replay 将任意非零 arm error 直接视为停止条件，删除 `recoverable_errors` 参数和 `_wait_for_arm_recovery()`，不只删除等待函数；
7. camera calibration 与 keyboard teleop 删除 recoverable/collision 集合读取，任意非零 arm error 走其既有 fault/stop 路径；
8. 用统一故障路径检查替换 C24 自动恢复检查，并同步清理 README/CLAUDE。

删除配置键时不得在 runtime merge 中静默忽略旧值；旧 override 应得到明确的 unknown/removed-key 错误，避免操作者以为恢复策略仍生效。

### 阶段 C：配置、身份和指标

修改范围还包括 `config/runtime.py`、effective-config metadata 和相关 CLI/help；不能只改 defaults。

内容：

1. 在 bounded report readiness 后校验 axis/device profile/可选 SN/firmware tuple；
2. 将关节限制与人工确认的 device profile 绑定；
3. 按项目、SDK 硬限制和设备报告范围的交集校验 speed/mvacc；
4. 修复命令时延；
5. 增加内部 mode/state 漂移监控；
6. 清理死代码和错误注释。

### 阶段 D：机械模型证据门

单独核验 0.033/0.043 m、TCP load COG、hand base transform 和碰撞模型。没有物理测量证据时不修改 URDF。

## 11. 离线验证要求

### 11.1 控制器状态

- `set_state(0)` 后 State 0/1/2 的合法行为；
- State 3/4/5 不允许接受 Mode 6 命令；
- 未知 state fail closed；
- 单个延迟的 cached mode 样本不误报，持续 Mode 6 漂移才 sticky fault；
- DISARMED、FAULT 和退出均确认 State 4；
- latched controller error 不妨碍 cleanup 尝试确认 State 4；
- `set_state(4)` 返回非零但 live state 已为 4 时仍判定停止已确认，同时保留原 fault；
- Mode/state/stop 有界等待持续刷新 arm heartbeat，不因 helper 轮询触发假超时；
- worker 从不写共享 SafetyState。

### 11.2 Homing

- stale-generation HOME 在执行前拒绝；
- RUNNING 或其他非 ARMED 状态下的 HOME 在 Mode 0 前拒绝；
- generation 在规划期间改变时不入队，request 保留失效边界处捕获的旧值；
- 执行中 generation 改变后停止；
- 执行中 DISARMED 后停止；
- E-stop、shutdown、FAULT 后停止；
- CANCELLED 且 State 4 已确认时不新增 arm execution fault；
- CANCELLED 的停机无法确认时升级为 FAILED；
- E-stop/CANCELLED 不清除或遮蔽全局 E-stop/FAULT；
- 超时、SDK failure、controller error 设置 sticky fault；
- 失败不恢复 Mode 6；
- 成功但 generation 已改变也不恢复 Mode 6；
- generation/SafetyState 在 Mode 6 恢复调用期间改变时重新停到 State 4；
- 仅 SUCCESS + current generation + ARMED + Mode 6 后置条件恢复命令接收；
- Homing 执行期间以及任意非 SUCCESS 返回后 `accepts_motion_commands` 均为 false；
- FAILED 的 sticky flag 在 HomeResult 入队前可见，bool 调用方不会误判为普通取消；
- HomeResult queue put 失败会 sticky fault、关闭命令接收并确认 State 4。

### 11.3 错误路径

- C22、C24、C31 都进入同一停止/故障路径；
- 日志保留不同错误含义；
- setter API code 非零且 live controller error 为 0 仍进入故障路径，并分别记录两个 code；
- sticky error 在停止尝试之前写入；
- worker 不调用 `clean_error()` 自动恢复；
- controller error live read 失败时 fail closed；
- 启动存在 controller error 时不清错、不发布 ready，并尽力停到 State 4；
- replay 遇到任意非零 arm error 时不再等待 worker 自动恢复；
- calibration 与 keyboard teleop 不再访问已删除的 error-set 配置，任意非零 arm error 都停止。

### 11.4 指标与配置

- queue、SDK、apply latency 使用可控时钟验证；
- speed/mvacc 低于或高于有效交集时配置/启动失败；
- axis、device type、firmware 不匹配时不发布 ready；
- identity 延迟到 report ready 后采样，metadata 包含实际 axis/device type/SN/firmware；
- firmware 使用整数 tuple 比较，device profile 缺少现场确认时不猜测型号。

### 11.5 推荐命令

```bash
conda run -n real_robot python checks/offline/run_all.py
conda run -n real_robot python -m compileall -q dexmani_real examples checks
conda run -n real_robot mypy
git diff --check
```

`examples/test_*.py` 属于交互式硬件程序，不作为自动测试运行。

## 12. 真机验证边界

软件补丁通过离线检查后，真机验证仍须单独授权，并至少满足：

1. 工作空间清空，物理急停可触达；
2. 确认真实 axis、device type、SN、firmware；
3. 核对 payload、mounting direction、collision sensitivity；
4. 核对 Reduced Mode、安全边界和关节范围；
5. 使用保守 speed/mvacc 验证 Mode 0 → Mode 6；
6. 验证 DISARMED、E-stop 和 Homing 取消的停止行为；
7. 不通过人为制造碰撞来测试 C22/C31；
8. C24 只通过已有日志或受控非运动故障注入验证软件分支；
9. 完成验证前不调整 120/900 默认值和 0.033/0.043 m 模型。

## 13. 完成标准

完成本方案后，xArm7 代码应满足：

```text
一个 Mode 6 命令路径
一个 Mode 0 Homing 路径
一个 State 4 停止原语
一个 sticky controller-fault 路径
无隐式运行期清错
无 State 2 唯一就绪假设
所有 HOME 请求受 generation 和 SafetyState 约束
HOME 取消只有在 State 4 确认后才视为非故障结果
arm worker 不写共享 SafetyState
SDK API code 与 controller error code 分开记录
mode cache 不被误称为同步 live read
配置值与 SDK 实际执行值一致
不修改 HDF5 v16 语义
```

实施补丁必须逐阶段提交和验证，阶段 A/B 不应与 URDF、TCP load 或运动参数调整混合。文档 review 能消除设计层面的已知竞态与所有权冲突，但不能替代离线测试和经授权的真机验证。

这套结构保留 xArm 控制器在线规划的优势，同时让每种状态变化、取消和故障都只有一个清晰出口。
