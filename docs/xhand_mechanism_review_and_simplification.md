# XHand 机制审查与简化实施方案

> 日期：2026-08-16
>
> 范围：`dexmani_real/robot/xhand.py`、`dexmani_real/robot/hand_process.py`、手部状态与动作的直接生产者/消费者，以及 `docs/xhand/` 中的参考项目文档。
>
> 性质：离线代码审查与实施设计，不包含硬件参数调整或硬件验证。

## 1. 结论摘要

当前项目的总体架构是正确的：在生产运行链中，XHand SDK 对象只由手部 worker 持有，跨进程通信只通过 `SharedStorage`，手部命令采用 latest-wins ring，跨生命周期旧命令由 `run_generation` 隔离，时间过期则由命令有效窗口拒绝。这些约束应当保留。独立硬件示例可以直接使用驱动，但不属于生产跨进程控制链。

需要优先修复的 P0 逻辑错误是：**当动作包含手部目标时，只要当前反馈快照不可用、无效、非有限或过期，就必须在写入任一命令通道前拒绝该候选动作。** 当前部分发布路径在“状态帧存在但健康标志为假”时会使用全零手姿态继续校验，导致不健康的手部状态仍可能越过策略发布边界。

其余问题主要集中在驱动和 worker 的职责重叠：连接时隐式完成触觉初始化、错误状态存在多套含义、失败计数与真实时间不一致、读取失败用 NaN 字典表达、`clear_local_error()` 看似恢复硬件但实际只清本地字段。这些行为增加了代码复杂度，也使日志和安全判断难以准确解释。

建议采用“**薄驱动、厚 worker、单一健康门**”的目标结构：

```text
teleop / policy / replay / home
               │
               ▼
      统一的手部反馈健康门
               │
               ▼
      SharedStorage 命令 ring
               │
               ▼
 hand_process：生命周期、重试、超时、故障升级
               │
               ▼
 XHand：connect/get_state/send_action/disconnect
```

本次审查不支持把 2026-08-15 的瞬时板级异常直接归因于拇指过流、`tor_max` 或 EtherCAT 实时调度权限。现有证据只能确认这些现象在时间上相关，不能确认因果关系。

## 2. 审查原则与边界

### 2.1 审查原则

1. 修复确定的逻辑错误，不以猜测修改机械、安全或电流参数。
2. 一个状态只保留一种明确语义，一个故障只保留一处升级策略。
3. 驱动只适配 SDK；worker 拥有进程生命周期、重试和设备故障检测；Main 继续拥有全局 `SafetyState` 的 FAULT 转换与关闭协调。
4. 对输入失败关闭：不静默补零、不静默截断、不静默夹紧。
5. 不引入第二套 IPC、控制服务、插值器或跨进程状态机。
6. 保持现有 `SharedStorage`、seqlock、latest-wins、`last_cmd_seq` 和 `run_generation` 契约。

### 2.2 不在本方案中处理的事项

- 不修改 `tor_max`、PID、机械限位或触觉阈值。
- 不为 xArm 或 XHand 添加持续的应用层轨迹插值。
- 不调整操作系统 capability、实时优先级或 EtherCAT 部署权限。
- 不依据一次现场异常推断厂商错误寄存器含义。
- 不改变 HDF5 episode v16 的数据语义。

### 2.3 方案 review 后的防回归约束

下列约束用于防止“修复一个问题，同时引入新的控制问题”：

1. **先冻结外部行为，再做内部简化。** 第一阶段只增加对无效 hand feedback 的拒绝，不同时改变 watchdog、触觉初始化、通信协议或 shared dtype。
2. **不静默改变共享字段语义。** `HAND_STATE_DTYPE` 中的 `send_healthy`、`read_healthy`、`state_valid` 和时间戳已经有多个消费者。内部重构必须保持现有含义；如确需改变，必须从 `utils/schema.py` 开始审计 worker、policy、supervisor、录制和回放链路。
3. **不把瞬时发送失败变成恢复死锁。** 如果将 `send_healthy` 改成“最近一次发送结果”并将其作为硬发布门，第一次失败后将没有新命令可用于证明恢复。本文不采用这种改法。
4. **不宣称 arm/hand 传输是原子的。** arm queue 与 hand ring 是两个独立 IPC 原语，且架构明确不增加 prepare/commit 协议。“reject-whole”只表示所有健康和范围校验在第一次 IPC 写入前完成，不表示两个 transport 写入具有事务原子性。
5. **不提前 readiness。** 拆分触觉初始化后，worker 仍须完成约定的初始化步骤并发布首个有效关节状态，才能设置 heartbeat 和 `hand_ready`。触觉校准失败按当前策略可以降级，但必须得到明确结果，不能仍在后台修改 bias 时宣布 ready。
6. **不依赖 NumPy 的伪不可变性。** `@dataclass(frozen=True)` 不能阻止数组内容被原地修改。局部状态样本必须拥有自己的数组副本；若声明只读，还需显式设置数组为不可写。
7. **前两个阶段不重命名公共方法。** 先保留 `connect/get_state/send_action/disconnect`，只收紧其内部契约。方法改名只有在所有直接调用方和示例完成迁移后才考虑。
8. **裸 qpos 不是健康证明。** `current_hand_qpos` 只携带数值，不包含连接、错误、watchdog 和时间戳信息，不能作为跳过 hand ring 健康检查的依据。需要复用同一反馈时，应传递由统一健康门生成的内部 snapshot，而不是传一个未经证明来源的数组。

## 3. 当前 XHand 机制

### 3.1 控制链路

1. teleop、策略、回放或 home 逻辑生成关节目标。
2. 安全层校验机械范围、运行范围、单步变化量和反馈状态。
3. arm 目标进入有界队列；hand 目标进入 latest-wins ring。
4. 命令携带 `run_generation`，worker 拒绝旧 generation 命令。
5. `hand_process` 是生产运行链中唯一持有 XHand SDK 对象的进程。
6. worker 读取手部状态并写入 `SharedStorage`，同时用 `last_cmd_seq` 表示最近一次 `send_action()` 返回成功的 action ID。

这一主链路符合项目架构约束，不应被参考项目中的 ZMQ server、pickle 消息或额外控制线程替代。

### 3.2 当前职责分布

| 模块 | 当前职责 | 审查结论 |
|---|---|---|
| `robot/xhand.py` | SDK 打开/关闭、读写、状态解析、范围检查、触觉初始化、局部错误状态 | 职责过多，应缩为 SDK 适配器 |
| `robot/hand_process.py` | 生命周期、状态发布、命令消费、watchdog、全局故障升级 | 所有进程级策略应集中在这里 |
| `policy/safety.py` | 动作边界与反馈校验 | 应成为耦合动作发布的唯一健康门 |
| `teleop/loop.py` 等调用方 | 决策是否生成和发布动作 | 不应分别实现不同版本的手部健康规则 |
| `shm/shared_storage.py` | 跨进程状态与命令 | 结构正确，保持不变 |

## 4. Fact-check 结果

本章将结论分为“已证实”“潜在问题”和“尚不可证实”，避免把代码事实、现场相关性和原因假设混为一谈。

### 4.1 已证实的问题

#### 4.1.1 不健康手部反馈仍可能发布耦合动作

`policy/safety.py` 中的部分路径按以下方式构造当前手姿态：

1. 先创建全零 `current_hand`；
2. 如果没有读到任何 hand frame 且候选动作包含手目标，则拒绝；
3. 如果读到了 frame，但 `connected` 或 `state_valid` 为假，则保留全零值继续执行校验和发布。

因此，“存在一帧无效状态”和“没有状态”产生了不同的安全行为。离线 fake/shared-memory 复现确认，无效状态帧下 `validate_and_send_candidate()` 仍可返回已发布结果。

影响边界需要准确描述：这证明动作已经越过策略发布边界，并可能写入 arm queue 和 hand ring；它不等同于已经被硬件执行，因为 worker 和全局安全状态仍有后续门控。健康判断使用的是一次反馈快照，也不能保证设备在随后发送瞬间仍保持同一状态。

修复规则：

```text
candidate 包含 hand_qpos
        │
        ├─ hand frame 不存在 ───────────► 拒绝
        ├─ connected/state_valid 为假 ──► 拒绝
        ├─ qpos 非有限或 shape 错误 ────► 拒绝
        ├─ 状态过期 ────────────────────► 拒绝
        └─ 健康 ────────────────────────► 继续做范围与 delta 校验
```

当 `candidate.hand_qpos is None` 时，通用发布函数按 payload 将其视为纯 arm 动作，不额外引入 `allow_arm_only` 开关。若某一具体工作流即使不发送 hand 命令也要求手部在线，应由该工作流在调用发布函数前增加前置条件。这样既避免全零回退，也避免新增一个容易被误设的通用布尔参数。

这里的“整条拒绝”仅适用于第一次 IPC 写入前的健康和安全校验。现有 `send_command()` 先写 arm queue、再写 hand ring，两条通道不是事务。本文不通过交换写入顺序解决该问题，因为那只会把“arm 已写、hand 未写”变成“hand 已写、arm 未写”。如第二次写入出现意外 IPC 异常，应进入协调停止/全局故障路径，而不是尝试回滚已被另一进程看到的命令。

#### 4.1.2 RS485 的强制刷新会重发最后命令，但当前生产路径不受影响

本机 `xhand-controller 1.1.8`、SDK `1.4.6` 的实现显示：

- RS485 的 `read_state(device_id, true)` 会先调用 `send_command()`，即重发 SDK 内部保存的最后命令；
- EtherCAT 实现忽略 `force_update`，直接返回周期 PDO 缓存中的状态。

当前 `hand_process` 没有传入通信类型，`XHandConfig` 默认使用 EtherCAT，因此该副作用不是当前生产链或 2026-08-15 异常的原因。它仍会影响允许选择 RS485 且使用 `force_update=True` 的诊断示例。

#### 4.1.3 连接失败路径清理不完整

正常连接后的 `disconnect()` 会调用 SDK `close_device()`，不能笼统地说“disconnect 不关设备”。确定的问题是：

- SDK open 抛出异常时，`_retry_open_device()` 直接向上抛出，没有关闭已创建的 control；
- 不支持的通信类型会返回失败，但 control 已经创建；
- `disconnect()` 只在 `connected_flag=True` 时关闭，因此上述失败路径无法依靠它兜底；
- `hand_process` 初始化异常分支记录并上报错误，但没有对局部 hand 对象执行统一关闭。

修复后应满足：只要 SDK control 曾成功创建，初始化异常或失败返回都进入同一个 best-effort 清理路径；同一会话底层 close 最多执行一次，重复调用公开 `disconnect()` 不报错。

#### 4.1.4 `clear_local_error()` 不具有硬件恢复效果

该方法只清除 Python 对象中的错误字段，没有 SDK 调用，也不确认设备恢复。worker 在发送失败和板错误处理路径中调用它，会造成三种歧义：

- 日志看起来像完成了恢复；
- 下一次读取又可能覆盖刚清掉的本地状态；
- 全局 sticky fault 与局部字段的关系不清楚。

应删除这一恢复概念。单次读写结果由当次 SDK 返回决定；是否升级为 global fault 由 worker 的有界 retry/watchdog 策略决定，Main 继续负责 `SafetyState` 的 FAULT 转换和关闭协调。

#### 4.1.5 发送 watchdog 不是稳定的时间阈值

默认阈值为 30，并注明“1s @ 30Hz”，但计数器只在“收到新命令且发送失败”时增加。正常控制命令约为 16 Hz，因此连续 30 次发送失败约为 1.875 秒；没有新命令时，计数不会增长。

这使 `send_healthy` 实际表示“发送失败 watchdog 尚未触发”，而不是“最近一次发送成功”。该字段已被 teleop 和 supervisor 消费，不能在内部重构时直接改义。

安全的处理顺序是：

1. 首先修正文档和变量命名，保持现有计数行为；
2. 将 `clear_local_error()` 替换为“成功时重置私有计数、失败时继续累计”，但不改变阈值；
3. 如果后续确实要改成单调时间阈值，将其作为独立运行策略变更，明确旧的“失败命令数”和新的“持续秒数”如何换算，并分别验证 16 Hz 命令、30 Hz worker 和命令静默场景；
4. 不将“最近一次发送失败”直接作为禁止下一次恢复发送的硬门。

#### 4.1.6 `full=True` 返回的 `error_state` 可能滞后一帧

`XHand.get_state()` 先由解析函数把旧的 `self.error_state` 放入返回字典，随后才根据本次板状态更新内部字段。因此公共诊断返回可能比当前读数滞后一帧。

当前 worker 使用更新后的对象字段，生产状态发布基本不受这一点影响。更简单的修复不是调整赋值顺序，而是删除 `full` 双返回协议，使用一次性构造的固定 `XHandSample`。

#### 4.1.7 `connect()` 不是只读操作

连接流程不仅打开设备和读取首帧，还会检查触觉负载、reset 触觉传感器并计算软件 bias。因此把 DISARMED 启动描述为“只读”并不准确。

当前实现已经在 reset 前检查初始负载，不能据此断言它会把所有静态接触校准掉。真正的问题是触觉校准被隐藏在连接动作中：调用方无法区分“建立通信”和“修改传感器基线”。

### 4.2 潜在问题

以下问题在当前单次 worker 生命周期中不一定触发，但会影响重连、诊断或未来扩展。

| 问题 | 当前影响 | 建议 |
|---|---|---|
| 正常 close 后仍保留 `control`、上次命令和触觉 bias | 当前 worker 通常不复用同一实例，风险暂时较低 | connect 开始和 close 结束统一重置会话状态 |
| 状态解析对非法/重复 joint ID 采用跳过并留下 NaN | 下游 finite 校验通常会关闭失败，但路径间接 | 解析层直接返回结构化错误，不生成半成品状态 |
| `source_monotonic_ns` 是 host 接受时间 | 可证明本次 SDK read 完成，不能证明设备产生了新帧 | 修正文档语义；除非 SDK 提供帧号，否则不声称设备侧新鲜度 |
| `XHand.connect()` 内多次读取，worker 随后又读取初始帧 | 增加启动路径和错误分支 | 后续可让 connect 只完成连接/既有触觉步骤，由 worker 统一读取并发布首个有效样本；首轮重构不改变顺序 |
| 生产配置固定 EtherCAT，但驱动表面支持 RS485 | 形成无法通过 runtime 选择的“半支持”状态 | 明确 EtherCAT-only，或正式把协议纳入 immutable runtime |

### 4.3 尚不可证实的原因假设

2026-08-15 记录只能确认：设备没有断连，出现了约 0.1 秒的板错误寄存器异常，且时间上与拇指停滞和电流变化相关。以下结论没有足够证据：

- “拇指过流直接设置了 joint-board error”；
- “`tor_max` 的含义或当前取值导致了异常”；
- “EtherCAT 实时调度 `EPERM` 导致了此次板错误”；
- “提高进程权限即可修复该问题”。

在获得厂商错误位定义、SDK/固件版本说明或受控硬件复现前，不应据此调整扭矩、PID、机械范围或系统 capability。

## 5. 参考项目知识的取舍

参考项目用于提炼边界和模式，不作为可以直接复制的实现。

### 5.1 πR² XHand

值得采用：

- 命令执行结果与实测状态分开表达；
- 反馈不依赖本地命令历史推算；
- 打开设备后记录 SDK 版本、设备 ID 和身份；
- 生命周期具有完整的 `try/finally`；
- 用 fake SDK 覆盖失败路径。

不建议采用：

- 在应用层持续插值手部轨迹；
- 再建立一套跨进程连接状态机；
- 把大量诊断信息放进每一帧实时状态。

当前项目已经有全局 `SafetyState`、worker 门控和 `run_generation`，再增加第二套状态机会造成所有权冲突。手部目标应继续作为绝对关节目标发送；如果 home 必须限制大步长，应使用少量显式里程碑，而不是引入通用插值器。

### 5.2 LeFranX

值得采用：

- 驱动保持为固定 12 维绝对位置接口；
- 命令对象预分配，循环中只更新固定字段；
- home 使用普通位置命令，不建立特殊旁路。

不建议采用：

- `np.clip` 后继续执行，静默改变调用方请求；
- 通过匹配错误字符串决定忽略 SDK 失败；
- stub 或未实现路径默认返回成功；
- 固定选择第一个设备；
- 按 SDK 数组位置而不是 joint ID 解释状态；
- 缺少明确 close 和恢复语义。

本项目现有的设备 ID 检查、范围拒绝、SDK 成功后才更新 `last_qpos_cmd` 等行为比参考实现更可靠，应保留。

### 5.3 DexUMI

值得采用：

- 采集、映射和硬件控制分层；
- 使用固定领域状态对象表达一次完整样本；
- 明确状态更新频率与缓存所有权；
- 标定参数与控制过程分离。

不建议采用：

- reader 和 controller 并发访问同一 SDK；
- 新增 ZMQ/pickle 控制面；
- 用虚拟状态累计代替实际反馈；
- 隐藏硬编码补偿、校准和单位转换；
- 并列存在多套 server、interpolator 或命令协议。

DexUMI 对本项目最重要的启示是负面的：当前单 worker + SharedStorage 的所有权更清晰，不应为了“模块化”引入第二条控制链。

## 6. 目标设计

### 6.1 单一手部健康门

项目已经存在纯函数 `teleop.keyboard.validate_hand_feedback()`。为避免新增一份近似实现，应把它原样迁移到中立模块，例如 `robot/hand_health.py`，再让 keyboard、teleop、policy、home 和 replay 共同导入。第一步保持函数参数和返回值不变；如重复的 structured-record 解包确实明显，再增加一个很薄的 record adapter，而不是重写第二套规则。

迁移时必须同步修改所有 import，并先为原函数建立离线回归用例。不能在“移动函数”的同一提交中改变 `send_healthy/read_healthy` 含义或超时默认值。

为保持现有诊断优先级，纯函数的检查顺序先维持为：

1. `connected`；
2. 当前 hand `error_state`；
3. `state_valid`；
4. 按现有共享字段语义检查 `send_healthy/read_healthy`；
5. source timestamp 存在且不在未来；
6. resolved `max_age_s` 有效，且 worker 接受时间未过期；
7. qpos shape 正确且全部数值有限。

该函数只解释反馈，不发布命令、不修改 flag、不读取 SDK。hand ring 为空由调用它的薄 adapter 先处理。`max_age_s` 必须由 resolved runtime 配置显式传入，不能在 helper 内硬编码，也不能从 `SharedStorage` 猜测默认值；阶段一必须把该参数贯穿所有直接调用方，不能增加悄然放宽检查的默认值。

policy 内再保留一个很薄的读取 adapter：一次读取 hand record，调用上述纯函数，健康时返回包含 `qpos`、source timestamp 和可选 `last_cmd_qpos` 的内部 snapshot；`last_cmd_qpos` shape/finite 不合格时保持 `None`，不把反馈健康与 delta reference 可用性混成同一条件。`publish_joint_targets()` 当前以 hand frame 的 `last_cmd_qpos` 做 delta reference，因此它的 delta preflight 与 gate 应使用同一 snapshot。learned policy、teleop 等路径已有不同的 last-published/last-accepted delta 契约，这些路径只复用健康判断，不改变其 delta reference。应删除或限制能用裸 `current_hand_qpos` 跳过健康检查的入口。这样既避免重复读取产生不一致，也避免“调用方传了一个数组，所以默认它健康”的旁路。

### 6.2 最小 XHand 驱动接口

建议保留四个现有必需操作和一个可选显式操作；前两个实施阶段不做方法改名：

```python
class XHand:
    def connect(self) -> bool: ...
    def get_state(self) -> XHandSample: ...
    def send_action(self, qpos_rad: np.ndarray) -> bool: ...
    def initialize_tactile(self) -> bool: ...  # 可选的显式步骤
    def disconnect(self) -> None: ...
```

`XHandSample` 只携带单次读取的固定数据：

```python
@dataclass(frozen=True)
class XHandSample:
    qpos_rad: np.ndarray
    current: np.ndarray
    tactile_sum: np.ndarray
    tactile_force: np.ndarray
    tactile_contact: np.ndarray
    commboard_error: np.ndarray
    jointboard_error: np.ndarray
    tipboard_error: np.ndarray
```

`current` 沿用当前 SDK 字段名；在厂商量纲未确认前，不把它改写为 `_a`、`_ma` 或扭矩单位。上述 `frozen=True` 只冻结属性绑定，不冻结 ndarray 内容。实现时每个数组必须脱离 SDK 缓冲区并由 sample 独占；如果解析器已经生成新数组，不再做一次无意义的深拷贝；只有 SDK 返回共享/view 缓冲区时才显式 copy。如果类型对外承诺只读，应在构造后对数组执行 `setflags(write=False)`。worker 写入共享 dtype 时仍创建新 frame，不直接发布或原地复用 sample 数组。

要求：

- 所有数组在构造前完成精确 shape、joint ID 唯一性和 finite 校验；
- SDK 返回错误时统一抛出一个由 worker 捕获的局部读取异常，不同时混用异常、`None` 和 NaN 半成品三套失败协议；
- `send_action()` 仅在 SDK 确认成功后更新 `last_qpos_cmd`；
- `disconnect()` 幂等，并重置整个会话状态；底层 SDK control 一旦成功创建，每个会话最多关闭一次；
- 驱动不修改 `SharedStorage`，不决定 sticky fault，不持有跨进程状态。

由于驱动只有一个直接消费者，命令返回不必引入复杂的跨进程结果对象。`bool` 加同一次调用捕获的 SDK 错误码和结构化日志已经足够；状态样本则值得使用固定类型，因为它替代了当前多分支字典和 NaN sentinel。不要在驱动和 worker 各记录一次相同堆栈：驱动提供错误上下文，worker 在决定重试或升级的位置记录一次。

### 6.3 worker 统一拥有恢复策略

`hand_process` 应拥有：

- SDK 对象完整生命周期；
- 首帧读取和 readiness 发布；
- 命令 generation、序号与 global safety gate；
- 私有的读写重试、watchdog 与故障升级策略；
- board 错误状态转换日志；
- 将完整样本一次性转换为共享内存 dtype；
- 初始化失败、正常退出、可捕获的循环异常和 e-stop 的统一 close；不可捕获的进程死亡由 supervisor 检测，不能承诺执行进程内 finally。

建议明确状态字段语义：

| 字段 | 唯一语义 |
|---|---|
| `state_valid` | 保持现有共享契约；最近发布的状态是否来自一次成功且完整的读取 |
| `read_healthy` | 保持现有共享契约；读取失败 watchdog 是否尚未触发 |
| `send_healthy` | 保持现有共享契约；发送失败 watchdog 是否尚未触发 |
| `connected` | 保持现有共享契约；最近读取时驱动仍认为设备已连接 |
| `source_monotonic_ns` | worker 接受该完整样本的 host 单调时间 |
| hand frame 的 `error_state` | 当前样本是否包含非零板错误寄存器；它不是 sticky global fault |
| `shared.error_state.value` | 进程间 sticky global fault，继续遵守既有所有权和升级路径 |

连续失败策略先继续使用现有私有计数器，成功时重置、失败时增加。只有在独立策略变更中才改用 `failure_started_ns`；届时仍不能静默改变上述共享字段含义，并必须审计所有消费者。

### 6.4 显式触觉初始化

目标上，连接动作只负责创建 SDK、打开目标设备、确认通信协议和设备 ID、记录身份；触觉 reset/bias 作为独立初始化步骤，由 worker 在明确条件下调用。但这一拆分必须保持当前启动顺序：worker 在触觉步骤完成并发布首个有效关节状态之后，才设置 heartbeat 和 `hand_ready`。

这样可以同时满足：

- 日志能区分通信失败与触觉校准失败；
- 只读诊断无需修改传感器基线；
- 未来重连可以决定复用还是重新计算 bias；
- 接触检查发生在明确的校准入口，而不是隐藏于 `connect()`。

第一轮拆分继续默认执行触觉初始化，并保持“触觉校准失败可降级为 `calibrated=False`、但有效关节控制仍可 ready”的现有策略。是否默认执行、是否把校准失败升级为启动失败，都属于后续运行行为变化，需要单独的离线 fake 检查和硬件验证，不能在纯重构中悄然改变。

### 6.5 通信协议支持面

当前生产链实际固定 EtherCAT。为了避免半支持状态，建议按真实部署作一次明确裁决：

- 若生产永远使用 EtherCAT：生产配置和驱动明确为 EtherCAT-only，RS485 放入独立诊断适配器或示例；
- 若必须支持 RS485：把 `comm_type`、设备名和串口参数纳入 immutable runtime，并对两种协议分别进行 fake/offline 测试。RS485 状态轮询必须禁止隐式重发命令。

在裁决前，可以先修复公共逻辑和失败清理，不必立即删除协议分支。

## 7. 分阶段实施方案

### 阶段零：冻结现有行为

在修改生产代码前，用 fake SDK/SharedStorage 固定以下基线：

- 当前 EtherCAT 生产配置和设备 ID 选择；
- `HAND_STATE_DTYPE` 各字段的现有含义；
- 首个有效 hand frame 先于 `hand_ready`；
- 触觉校准失败可以降级，但不会触发 home；
- watchdog 的现有计数和 sticky global fault 行为；
- `send_command()` 的 arm-then-hand 非原子传输顺序。

这些基线不代表永远不改，而是保证每次提交只改变它声明要改变的行为。

### 阶段一：修复发布边界

修改范围：`policy/safety.py` 及其直接调用方。

1. 将已有 `validate_hand_feedback()` 迁移到中立模块，保持函数行为不变；
2. 增加薄 adapter，一次读取并验证 hand record，生成供 qpos/delta/gate 共用的内部 snapshot；
3. 所有包含 `hand_qpos` 的候选动作在第一次 IPC 写入前调用同一健康规则；
4. 删除全零当前手姿态回退和裸 `current_hand_qpos` 健康旁路；
5. `candidate.hand_qpos is None` 时自然走纯 arm 路径，不新增通用 arm-only 开关；
6. 对健康/安全校验失败，保证 arm queue 和 hand ring 都没有新写入；
7. 不在本阶段修改 transport 写入顺序、watchdog 或共享字段语义。

这是独立、风险最低且收益最大的修改，应首先完成。

### 阶段二：行为保持的生命周期收口

修改范围：`robot/xhand.py`、`robot/hand_process.py`。

1. 让 `disconnect()` 幂等并覆盖所有初始化失败路径；
2. 用一个外层 `try/finally` 统一正常退出、初始化失败和异常退出的关闭路径；
3. 删除 `clear_local_error()` 及相关“恢复成功”日志，但保持成功重置、失败累计和原 watchdog 阈值；
4. 对 board error 的出现、变化和消失记录精确数组、关节和十六进制值，并对重复相同值限频；
5. 不修改 shared dtype、readiness 顺序、触觉策略或通信协议。

### 阶段三：行为保持的驱动内部简化

1. 引入固定 `XHandSample`；
2. 合并 `get_state(full=True/False)`；
3. 删除 NaN 状态字典和静默 resize；
4. 保留现有公共方法名，先迁移唯一生产消费者；原始 SDK 示例在阶段五单独收紧；
5. 确保 sample 数组拥有独立副本，worker 每帧新建 shared dtype record；
6. 删除经 `rg` 确认无调用的配置和诊断字段。

该阶段保持共享内存 dtype 和调用方可观察行为不变，避免把内部简化与 schema 变更混在一起。每删除一个字段或分支，都先确认生产 worker、示例和文档没有消费者。

### 阶段四：逐项评审运行行为变化

以下事项不能与内部重构捆绑，应分别提交、分别验证：

1. 把触觉初始化从 `connect()` 中拆出，同时保持原启动顺序和降级策略；
2. 将失败命令计数改成单调时间阈值，并明确配置迁移；
3. 决定 EtherCAT-only 或正式双协议支持；
4. 如需修改 `HAND_STATE_DTYPE` 或字段含义，执行 dtype → worker → 所有消费者 → recording/replay 的完整纵向审计；
5. 对 arm/hand transport 的意外第二次写入失败，定义协调停止行为，但不引入 commit/rollback 协议。

这些变化需要独立风险评估；涉及真实 SDK 时还需要用户授权后的硬件验证。

### 阶段五：收紧示例与文档

`examples/xhand_control_example.py` 默认只枚举、读取和打印状态；RS485 状态读取显式使用 `force_update=False`，避免“读取”重发 SDK 的最后命令。任何运动需要显式参数及人工确认，并使用 `try/finally` 保证 close。预设动作必须经过生产机械和运行范围校验，不能直接绕过安全层发送原始 SDK 命令。

如实施改变了文件职责或关键所有权，应同步更新 `README.md` 文件地图和 `CLAUDE.md` 路由说明。

## 8. 验收标准

### 8.1 必需离线检查

建议新增或扩展以下 deterministic checks，全部使用 fake SDK/SharedStorage，不初始化真实硬件。每个阶段只启用与该阶段声明行为对应的检查：

1. 无 hand frame 且候选含手目标：拒绝；
2. `connected=False`：拒绝，arm queue 和 hand ring 都不变；
3. `state_valid=False`、当前 hand `error_state=True` 或 I/O watchdog 不健康：按现有健康规则拒绝；
4. qpos 含 NaN/Inf、shape 错误或状态过期：拒绝；
5. 即使调用方提供有限的裸 `current_hand_qpos`，无效/过期 hand record 仍不能被绕过；
6. 同一次候选发布的 hand qpos、last-command delta reference 和健康元数据来自同一 verified snapshot；
7. `candidate.hand_qpos is None` 且 arm 反馈健康：不读取虚构的全零 hand 姿态，只发布 arm；
8. 一次未达到 watchdog 阈值的发送失败不会永久禁止下一条恢复命令；
9. SDK open 在每个初始化步骤抛错：底层 control 每个会话最多 close 一次；
10. 重复 `disconnect()`：不报错、不重复访问失效资源；
11. malformed、缺失或重复 joint ID：整帧失败；
12. send 失败：`last_qpos_cmd` 不前移；
13. fake SDK 返回后修改其内部数组：已经构造的 `XHandSample` 不发生变化；
14. 触觉初始化完成或明确降级、首个有效 hand frame 已发布后，才能设置 `hand_ready`；
15. RS485 read-only 读取不调用 send；EtherCAT 分支不依赖该参数；
16. board error 出现、变化和消失：状态与日志一致，重复相同值不会刷屏；
17. 阶段二保持原 watchdog 触发点；只有阶段四改成时间策略后，才要求不同 worker/control 频率下按相同墙钟时长升级；
18. 正常 shutdown、可捕获的初始化/循环异常和 e-stop 都进入统一 close；对 SIGKILL、进程崩溃等无法执行 `finally` 的 worker death，不伪称已经调用 SDK close，验证 supervisor 能检测死亡、锁存故障并协调其他进程关闭；
19. 如修改 transport 异常处理：在 arm 已入队而 hand ring 写入异常时进入协调停止/故障路径，不返回成功，也不伪造回滚完成。

### 8.2 仓库级检查

Python 修改完成后至少执行：

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
git diff --check
```

并运行项目现有离线检查以及新增的 XHand fake 检查。`examples/test_*.py` 属于交互式硬件程序，不能当成自动化测试运行。

### 8.3 硬件验证边界

以下项目需要用户明确授权、工作区清空和硬件就绪后单独执行：

- 真实 EtherCAT/RS485 打开与关闭；
- 触觉 reset/bias 行为；
- home 或任意预设动作；
- 板级错误复现；
- `tor_max`、PID、机械范围或实时调度调整。

## 9. 预期结果

完成上述方案后，XHand 控制链应具备以下性质：

- 不健康手部反馈不能越过任何耦合动作发布边界；
- SDK 生命周期只有一个所有者和一条 close 路径；
- 驱动不再同时承担状态模型、恢复策略和共享内存语义；
- 读写健康、持续失败和全局 sticky fault 各自只有一种含义；
- 触觉校准是可见、可测试、可选择的步骤；
- EtherCAT 与 RS485 的支持范围明确，不再存在隐式协议副作用；
- 现场异常日志能精确说明“发生了什么”，但不会把相关性误写为原因；
- 不增加新的控制服务、线程、插值器或跨进程协议。

## 10. 参考材料

- [`docs/xhand/pi-r2-xhand.md`](xhand/pi-r2-xhand.md)
- [`docs/xhand/lefranx_xhand.md`](xhand/lefranx_xhand.md)
- [`docs/xhand/dexumi_xhand.md`](xhand/dexumi_xhand.md)
- [`docs/xhand/xhand_error_state_anomaly_2026-08-15.md`](xhand/xhand_error_state_anomaly_2026-08-15.md)
- `dexmani_real/robot/xhand.py`
- `dexmani_real/robot/hand_process.py`
- `dexmani_real/policy/safety.py`
- `dexmani_real/teleop/loop.py`
- `dexmani_real/shm/shared_storage.py`
