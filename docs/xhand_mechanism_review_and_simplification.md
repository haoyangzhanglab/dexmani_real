# XHand 机制审查与简化实施方案（修订版）

> 日期：2026-08-16（已按当日 HEAD `88becfd` 核对并修订）
>
> 范围：`dexmani_real/robot/xhand.py`、`dexmani_real/robot/hand_process.py`、手部状态与动作的直接生产者/消费者，以及 `docs/xhand/` 参考项目文档。
>
> 性质：离线代码审查与实施设计，不包含硬件参数调整或硬件验证。

## 0. 修订说明

本版相对初稿的实质修正：

1. **原 P0（§4.1.1 全零手姿态回退）已过时**。该缺陷在评审时（提交 `c298505`/`fef5071`）确实存在，但已在同日提交 `88becfd`（"0816 temp1"）中修复。本文 §2.1 改记为"已修复"，并把剩余工作重新定义为**发布门缺失时间戳过期检查**这一窄项。
2. **符号勘误**：初稿反复引用的 `current_hand_qpos` 在 `88becfd` 中被删除，当前树零匹配；真实字段是 `qpos`（健康输入）与 `last_cmd_qpos`（delta reference）。
3. **字段语义据实修正**：`state_valid` 实际镜像 `connected`（非"成功完整读取"）；`error_state` 除板寄存器外还含 `_record_error` 来源。
4. **健康门数量更正**：当前有 **四** 处手部健康判断，非两处；`deployment/coordinator.py` 与 `runtime/supervisor.py` 两处被初稿遗漏。
5. **参考项目取舍修正**：πR² 一节的 adopt/reject 据参考原文重写；DexUMI 两处过度声称下调。

## 1. 结论摘要

生产运行链的总体架构正确，应保留：XHand SDK 对象只由手部 worker 持有；跨进程只经 `SharedStorage`；手部命令 latest-wins ring；`run_generation` 隔离跨生命周期旧命令；命令有效窗口拒绝过期命令。

**核心发布边界问题已经修复**：`policy/safety.py` 的 `_hand_feedback_snapshot` 现为 fail-closed，不健康的反馈（断连、无效、板错误、I/O watchdog 触发、非有限 qpos）会在任何 IPC 写入前拒绝耦合候选。**唯一残留缺口**是该快照不做 `source_monotonic_ns` 年龄/过期检查——"过期但标志健康"的帧仍能越过发布边界。这是阶段一的唯一行为性改动。

其余问题集中在驱动与 worker 的职责重叠：连接时隐式触觉初始化、错误状态多套语义、失败计数与真实时间不一致、读取失败用 NaN 字典表达、`clear_local_error()` 仅清本地字段却表现为恢复。这些需按"**薄驱动、厚 worker、单一健康门**"收口，但**大多是对现状的归位**，而非新机制：

```text
teleop / policy / replay / home
               │
               ▼
      统一的手部反馈健康门（纯函数，四处调用方共用）
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

本次审查不支持把 2026-08-15 的瞬时板级异常归因于拇指过流、`tor_max` 或 EtherCAT 实时调度权限：现有证据只能确认时间相关性，不能确认因果关系。

## 2. 现状与问题（已按 HEAD 核对）

### 2.1 已修复：发布边界 fail-closed

`validate_and_send_candidate`（`policy/safety.py:740-849`）在 `candidate.hand_qpos is not None` 时调用 `_hand_feedback_snapshot`（`:205-256`），任一拒绝即 return，到不了 `send_command`：

- ring 为空 → `HAND_FEEDBACK_UNAVAILABLE`（`:210-215`）；
- `connected ∧ state_valid ∧ ¬error_state ∧ send_healthy ∧ read_healthy` 为假 → `HAND_FEEDBACK_UNHEALTHY`（`:217-236`）；
- `qpos`/`last_cmd_qpos` 错 shape 或非有限 → `HAND_FEEDBACK_UNHEALTHY`（`:239-249`）。

`candidate.hand_qpos is None` 时按纯 arm 动作处理，不读 hand ring，不构造任何手姿态。已删除旧代码中的全零 `current_hand` 回退与 `current_hand_qpos` 参数。

**残留缺口**：该快照不读 `source_monotonic_ns`、不校验 age。过期/未来时间戳的拒绝只存在于 `teleop/keyboard.py` 的 `validate_hand_feedback`（`:634-642`），而它不被 policy 发布路径调用。因此"过期但标志健康"的帧仍可通过。这是阶段一唯一的剩余行为改动，且需将 `max_age_s` 显式贯穿所有调用方，不能引入悄然放宽检查的默认值。

### 2.2 已证实待修：驱动/worker 职责重叠

| # | 问题（`xhand.py` / `hand_process.py`） | 结论与修法 |
|---|---|---|
| 1 | RS485 `read_state(force_update=true)` 会先 `send_command()` 重发最后命令；EtherCAT 忽略该参数 | 已在本机 vendored 源码证实（`serial_communication.cpp` 重发 / `ethercat_communication.cpp` 返回 PDO 缓存）。当前生产固定 EtherCAT，不受影响；诊断示例需 `force_update=False` |
| 2 | 连接失败清理不完整：`_retry_open_device` 抛异常不关已建 control；不支持 comm_type 返回失败但 control 已建；`disconnect()` 仅在 `connected_flag=True` 时关；worker 初始化异常分支无统一 close | 只要 SDK control 曾成功创建，所有失败路径进入同一 best-effort 清理；同会话底层 close 至多一次，重复 `disconnect()` 幂等 |
| 3 | `clear_local_error()`（`:867-877`）仅清本地字段，无 SDK 调用，不确认恢复 | 删除该"恢复"概念。它被 worker 在发送失败与板错误路径调用（`hand_process.py:347/444`），但 `shared.error_state` 已在 `:341` 先 latch，本地清除是装饰性的。单次读写结果由当次 SDK 返回决定；是否升级 global fault 由 worker 有界 retry/watchdog 决定。**注意**：它是 load-bearing，删除必须与"成功重置/失败累计、阈值不变"的替代物同提交落地 |
| 4 | 发送 watchdog 阈值 30 标注"1s @ 30Hz"，但计数器仅在"新命令 + 发送失败"时增加，约 1.875s@16Hz，无新命令不增长 | 先修正命名与注释，保持现有计数行为；`send_healthy` 的共享语义 = "watchdog 未触发"，非"上次发送成功"，重构不改变。改单调时间阈值属独立运行行为变更（见 §5 阶段四） |
| 5 | `get_state(full=True)` 的 `error_state` 滞后一帧：先 `parse_state`（`:903`）把旧值写入返回字典，再重推 `self.error_state`（`:911-915`） | worker 读对象字段（`hand_process.py:389`），生产发布基本不受影响；引入固定 `XHandSample` 后此双返回协议一并删除 |
| 6 | `connect()` 非只读：打开设备、读首帧、检查触觉负载、reset 触觉、算 bias | 拆分显式 `initialize_tactile()`（见 §3.4）。现有实现已在 reset 前查初始负载，不会把所有静态接触盲吸进 bias |
| 7 | 正常 close 后仍保留 `control`/最后命令/触觉 bias | connect 开始与 close 结束统一重置会话状态 |
| 8 | 状态解析对 out-of-range/负 joint ID 跳过并留下 NaN（`:1046`）；重复合法 ID 不产生 NaN（后者覆盖前者） | 解析层直接返回结构化错误，不生成半成品状态 |
| 9 | `source_monotonic_ns` 是 host 接受时间 | 修正文档语义：证明本次 SDK read 完成，不证明设备产生新帧；除非 SDK 提供帧号，不声称设备侧新鲜度 |
| 10 | 生产固定 EtherCAT，驱动表面支持 RS485 的"半支持" | 明确 EtherCAT-only 或把协议纳入 immutable runtime（见 §3.5） |

### 2.3 尚不可证实：08-15 异常归因

当日记录只能确认：设备未断连、出现约 0.1s 的板错误寄存器异常、时间上与拇指停滞和电流变化相关。以下结论证据不足：拇指过流直接置 joint-board error；`tor_max` 含义或取值导致异常；EtherCAT 实时调度 `EPERM` 导致此次板错误；提高进程权限可修复。在取得厂商错误位定义、SDK/固件版本说明或受控硬件复现前，不据此调整扭矩、PID、机械范围或 capability。

## 3. 目标设计

### 3.1 单一手部健康门（落位 `utils/`）

现有纯函数 `teleop/keyboard.py:613` `validate_hand_feedback()` 是 7 项 fail-closed 谓词（`connected` → `error_state` → `state_valid` → `send/read_healthy` → source 时间戳存在且不在未来 → `max_age_s` 有效且未过期 → qpos shape 且 finite），返回 `str | None`。

- **落位 `dexmani_real/utils/hand_health.py`**（非 `robot/`）：它是 schema-shape 纯校验（依赖 `utils/schema.py` 的 `HAND_JOINT_SHAPE`），不是 vendor-I/O 也不是 policy 处置；`robot/` 继续只做设备 I/O。对称地**一并迁移 `validate_arm_feedback`（`keyboard.py:573`）**，避免把一对纯函数拆到两个包。迁移保持签名与返回不变，先建离线回归用例；不在此提交内改 `send_healthy/read_healthy` 含义或超时默认值。
- **实际有四处手部健康判断**，须统一：

| 位置 | 内容 | 差异 |
|---|---|---|
| `teleop/keyboard.py:613` `validate_hand_feedback` | 7 项含时间戳/过期 | 基准 |
| `policy/safety.py:205` `_hand_feedback_snapshot` | 5 标志 + shape/finite | 缺时间戳/过期，多 `last_cmd_qpos` |
| `deployment/coordinator.py:145-171` `_seed_hand_reference` | 5 标志 + shape/finite | 缺时间戳/过期，缺 `last_cmd_qpos` |
| `runtime/supervisor.py:205-215` 内联 `hand_ok` | 5 标志 + finite | 内联副本 |

统一方式：policy 内保留一个薄 adapter——一次读 hand record，调用上述纯函数；健康时返回含 `qpos`、source timestamp、可选 `last_cmd_qpos` 的内部 snapshot（`last_cmd_qpos` 不合格时保持 `None`，不把反馈健康与 delta reference 可用性混成同一条件）。`publish_joint_targets` 的 delta preflight 与 gate 使用同一 snapshot（当前 delta reference 在 `safety.py:822-826` 取 `last_cmd_qpos`）。coordinator/supervisor 改为复用同一纯函数，消除内联副本。

### 3.2 最小 XHand 驱动接口

前两个阶段不做方法改名，保留四操作 + 一个可选显式步骤：

```python
class XHand:
    def connect(self) -> bool: ...
    def get_state(self) -> XHandSample: ...
    def send_action(self, qpos_rad: np.ndarray) -> bool: ...
    def initialize_tactile(self) -> bool: ...  # 显式步骤
    def disconnect(self) -> None: ...
```

`XHandSample` 是单次读取的固定数据类型（`frozen=True` 仅冻结属性绑定，不冻结 ndarray 内容）：

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

要求：

- 所有数组构造前完成精确 shape、joint ID 唯一性、finite 校验；
- SDK 返回错误时**统一抛局部读取异常**（不再混用异常 / `None` / NaN 半成品三套失败协议）；worker 捕获后决定重试或升级；
- `send_action()` 仅在 SDK 确认成功后更新 `last_qpos_cmd`；
- `disconnect()` 幂等并重置整个会话状态；底层 control 每个会话至多 close 一次；
- 构造器**恒 copy 并 `setflags(write=False)`**（"仅当 SDK 返回 view 才 copy"是脆弱优化、几乎零收益）；worker 每帧新建 shared dtype record，不原地复用 sample 数组；
- 驱动不修改 `SharedStorage`，不决定 sticky fault，不持有跨进程状态。

### 3.3 worker 统一恢复策略（字段语义据实）

`hand_process` 拥有 SDK 生命周期、首帧读取与 readiness、命令 generation/序号/global safety gate、私有读写重试/watchdog/故障升级、board 错误状态转换日志、完整样本到 shared dtype 的转换、统一 close。

| 字段 | 唯一语义（据当前实现） |
|---|---|
| `state_valid` | 最近发布状态是否来自成功读取；**当前实现为镜像 `connected`（`hand_process.py:473`，首帧无条件置 1）** |
| `read_healthy` | 读取失败 watchdog 是否尚未触发 |
| `send_healthy` | 发送失败 watchdog 是否尚未触发 |
| `connected` | 最近读取时驱动仍认为设备已连接 |
| `source_monotonic_ns` | worker 接受该完整样本的 host 单调时间 |
| hand frame `error_state` | 板错误寄存器，**或非 raising 的读写错误**（`_record_error` 在 send/read/`hand_command is None` 路径也置位） |
| `shared.error_state.value` | 进程间 sticky global fault，遵守既有所有权与升级路径 |

连续失败策略沿用现有私有计数器（成功重置、失败累计）；改 `failure_started_ns` 单调时间阈值属独立运行行为变更，须单独审计所有消费者。

### 3.4 显式触觉初始化

连接动作只负责建 SDK、开目标设备、确认协议与设备 ID、记录身份；触觉 reset/bias 作为独立 `initialize_tactile()` 由 worker 在明确条件下调用。保持现有启动顺序：触觉步骤完成并发布首个有效关节状态后，才设 heartbeat 与 `hand_ready`。第一轮拆分继续默认执行触觉初始化，保留"触觉校准失败可降级为 `calibrated=False`、有效关节控制仍可 ready"的策略；是否默认执行、是否把校准失败升级为启动失败属后续运行行为变化，需单独 fake 检查与硬件验证。

### 3.5 通信协议支持面

生产链固定 EtherCAT。作明确裁决：若生产永远 EtherCAT，则生产配置与驱动声明 EtherCAT-only，RS485 移入独立诊断适配器/示例；若须支持 RS485，则 `comm_type`/设备名/串口参数纳入 immutable runtime，两种协议分别 fake 测试，且 RS485 状态轮询禁止隐式重发命令。裁决前先修公共逻辑与失败清理，不必立即删协议分支。

## 4. 参考项目取舍

参考项目用于提炼边界与模式，不作为可直接复制的实现。

- **LeFranX**：adopt（固定 12 维绝对位置接口、命令对象预分配、home 走普通位置命令）与 reject（`np.clip` 后继续、按错误字符串匹配忽略、stub 默认成功、固定选首设备、按数组位置而非 joint ID 解释状态、缺 close/恢复语义）均逐条有据，可采纳。
- **DexUMI**：负面教训成立——单 worker + SharedStorage 所有权更清晰，不应为"模块化"引入第二条控制链（ZMQ/pickle/多套 server/interpolator/命令协议）。adopt 项中"状态更新频率与缓存所有权""标定参数与控制分离"恰是参考文档自己列的缺陷（M-3 新鲜度不可见、M-9 硬编码补偿漂移），不应作为其成熟实践引用。
- **πR² XHand**：底层判断（避免应用层插值、保持单一所有权链）正确，但初稿的支撑错位。其参考文档**自己把**"命令执行与实测分开"列为 P0 缺陷、"记录 SDK 版本/设备 ID"明确说没有、"完整 try/finally"说清理不足、"fake SDK 覆盖失败路径"说无 XHand 测试——这些不是 πR² 的现有实践，而是它的待修项，不得作为 adopt 依据。可采纳的只有"反馈不依赖本地命令历史推算"与"打开设备后记录身份"的方向性要求。reject 项应改为**本地故障状态机**（πR² 是单进程 in-process FSM，非"跨进程状态机"）与**启动期 home/首动作插值**（`interpolate_to` 只在启动期，主循环不逐 tick 插值）。

本项目已有的设备 ID 检查、范围拒绝、SDK 成功后才更新 `last_qpos_cmd` 等行为比参考实现更可靠，应保留。

## 5. 分阶段实施

### 阶段零：冻结现有行为

用 fake SDK/SharedStorage 固定基线：EtherCAT 生产配置与设备 ID 选择；`HAND_STATE_DTYPE` 各字段现有含义；首个有效 hand frame 先于 `hand_ready`；触觉校准失败可降级但不触发 home；watchdog 现有计数与 sticky global fault；`send_command` 的 arm-then-hand 非原子顺序。基线保证每次提交只改变它声明要改变的行为。

### 阶段一：发布门收口（已大半落地，剩一项行为变更）

`88becfd` 已落地：fail-closed `_hand_feedback_snapshot`、删除全零回退与 `current_hand_qpos` 参数、`hand_qpos is None` 自然走纯 arm。**剩余**：

1. 迁移 `validate_hand_feedback` 到 `utils/hand_health.py`（含 `validate_arm_feedback`），保持签名/返回不变；
2. 让 `_hand_feedback_snapshot` 委托该纯函数，补上 source 时间戳存在性、未来时间戳与 `max_age_s` 过期检查；
3. **`max_age_s` 必须由 resolved runtime 配置显式传入**，贯穿所有直接调用方；不得在 helper 内硬编码或从 `SharedStorage` 猜默认值；
4. 该过期检查是行为变更，波及三个 snapshot 消费者（耦合发布、`wait_applied` ack 循环 `safety.py:934-994`、hand-home `:1140/1225`），配独立 fake 测试。

不在此阶段改 transport 写入顺序、watchdog 或共享字段语义。

### 阶段二：生命周期收口

`robot/xhand.py`、`robot/hand_process.py`：

1. `disconnect()` 幂等并覆盖所有初始化失败路径；
2. 外层 `try/finally` 统一正常退出、初始化失败、异常退出的关闭路径；
3. 删除 `clear_local_error()` 与相关"恢复成功"日志，**与"成功重置/失败累计、原 watchdog 阈值不变"的替代物同提交落地**；
4. board error 出现/变化/消失记录精确数组、关节与十六进制值，重复相同值限频。

不修改 shared dtype、readiness 顺序、触觉策略或通信协议。

### 阶段三：驱动内部简化

1. 引入固定 `XHandSample`，合并 `get_state(full=True/False)`，删除 NaN 状态字典与静默 resize；
2. 统一抛局部读取异常，删除 `full` 双返回协议（`full=True` 的 `error_state` 一帧滞后陷阱须写进注释：worker 读对象字段故安全，未来读 merged sample 的调用方会看到滞后）；
3. 保留公共方法名，先迁移唯一生产消费者；原始 SDK 示例在阶段五收紧；
4. sample 数组恒 copy + `setflags(write=False)`；worker 每帧新建 shared dtype record；
5. 删除经 `rg` 确认无调用的配置/诊断字段——**须先产出具体字段清单**（当前 `last_error_code`/`last_joint_limit_rejected`/`force_update_state` 等仍存活，`xhand.py:104/191/887/939`），并标注每个删除字段的读者（`_init_hand_state` 与 worker per-tick parse 是 `XHandSample` 合并的首个回归点）。

本阶段保持 shared dtype 与可观察行为不变，不与 schema 变更混在一起。

### 阶段四：独立运行行为变更（分别提交、分别验证）

1. 触觉初始化从 `connect()` 拆出（保持原启动顺序与降级策略）；
2. 失败命令计数改单调时间阈值（明确配置迁移，分别验证 16Hz 命令/30Hz worker/命令静默）；
3. 裁决 EtherCAT-only 或正式双协议支持；
4. 如需改 `HAND_STATE_DTYPE` 或字段含义，做 dtype → worker → 所有消费者 → recording/replay 的完整纵向审计；
5. 定义 arm/hand transport 第二次写入失败的协调停止行为，不引入 commit/rollback 协议。

涉及真实 SDK 的须经用户授权后硬件验证。

### 阶段五：收紧示例与文档

`examples/xhand_control_example.py` 默认只枚举、读取、打印状态；RS485 读取显式 `force_update=False`；任何运动需显式参数与人工确认，`try/finally` 保证 close；预设动作走生产机械与运行范围校验，不绕过安全层发原始 SDK 命令。同步更新 `README.md` 文件地图与 `CLAUDE.md` 路由说明。

## 6. 验收标准

### 6.1 离线检查（fake SDK/SharedStorage，不初始化硬件）

1. 无 hand frame 且候选含手目标：拒绝；`connected=False`：拒绝，arm queue 与 hand ring 都不变；
2. `state_valid=False`、当前 hand `error_state=True` 或 I/O watchdog 不健康：拒绝；
3. qpos 含 NaN/Inf、shape 错误：拒绝；
4. **状态过期/未来时间戳：拒绝（阶段一后）** —— 当前代码尚不满足，接上过期检查后方可通过；
5. 无效/过期 hand record 不能被"调用方传有限数组"绕过；
6. 同一次候选发布的 hand qpos、delta reference、健康元数据来自同一 verified snapshot；
7. `candidate.hand_qpos is None` 且 arm 反馈健康：不读虚构全零手姿态，只发布 arm；
8. **`candidate.hand_qpos is None` 且手离线（ring 空 / `connected=False`）：仍发布 arm 端点**（当前代码满足，需回归钉住）；
9. 一次未达阈值的发送失败不永久禁止下一条恢复命令；
10. SDK open 在每个初始化步骤抛错：底层 control 每个会话至多 close 一次；
11. 重复 `disconnect()`：不报错、不重复访问失效资源；
12. malformed/缺失/重复 joint ID：整帧失败；
13. send 失败：`last_qpos_cmd` 不前移；
14. fake SDK 返回后修改其内部数组：已构造的 `XHandSample` 不变化；
15. 触觉初始化完成或明确降级、首个有效 hand frame 发布后，才设 `hand_ready`；
16. RS485 只读不调用 send；EtherCAT 分支不依赖该参数；
17. board error 出现/变化/消失：状态与日志一致，重复相同值不刷屏；
18. 阶段二保持原 watchdog 触发点；只有阶段四改时间策略后才要求不同频率下按相同墙钟时长升级；
19. 正常 shutdown、可捕获初始化/循环异常、e-stop 都进统一 close；对 SIGKILL/崩溃等无法执行 `finally` 的死亡，验证 supervisor 能检测、锁存故障、协调关闭，不伪称已 SDK close；
20. **`validate_hand_feedback` 迁移后各调用方拒绝字符串逐字不变（黄金字符串测试）**；
21. **hand-home 与 `wait_applied` 循环在 stale/unhealthy 下行为正确**；
22. 如改 transport 异常处理：arm 已入队而 hand ring 写入异常时进协调停止/故障路径，不返回成功、不伪称回滚完成。

### 6.2 仓库级检查

```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
git diff --check
```

并运行项目现有离线检查与新增 XHand fake 检查。`examples/test_*.py` 属交互式硬件程序，不能当自动化测试运行。

### 6.3 硬件验证边界

以下须用户明确授权、工作区清空、硬件就绪后单独执行：真实 EtherCAT/RS485 开关；触觉 reset/bias；home 或任意预设动作；板级错误复现；`tor_max`、PID、机械范围或实时调度调整。

## 7. 预期结果

完成上述方案后：不健康手反馈不能越过任何耦合动作发布边界（含过期帧）；SDK 生命周期只有一个所有者和一条 close 路径；驱动不再承担状态模型、恢复策略与共享内存语义；读写健康、持续失败与全局 sticky fault 各自只有一种含义；触觉校准是可见、可测试、可选择的步骤；EtherCAT 与 RS485 支持范围明确；现场异常日志精确说明"发生了什么"而非把相关性写成原因；不新增控制服务、线程、插值器或跨进程协议。

## 8. 参考材料

- [`docs/xhand/pi-r2-xhand.md`](xhand/pi-r2-xhand.md)
- [`docs/xhand/lefranx_xhand.md`](xhand/lefranx_xhand.md)
- [`docs/xhand/dexumi_xhand.md`](xhand/dexumi_xhand.md)
- [`docs/xhand/xhand_error_state_anomaly_2026-08-15.md`](xhand/xhand_error_state_anomaly_2026-08-15.md)
- `dexmani_real/robot/xhand.py`
- `dexmani_real/robot/hand_process.py`
- `dexmani_real/policy/safety.py`
- `dexmani_real/teleop/loop.py`
- `dexmani_real/shm/shared_storage.py`
