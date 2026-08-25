# Learned-policy deployment 审查结论

> 审查对象：`dexmani_real/deployment`，并追踪其与 `control`、`runtime`、`robot`、`planning`、`integrations` 的运行时边界。
>
> 审查方式：源代码静态追踪、配置与共享内存契约检查、离线纯函数验证；**未连接设备、未执行任何硬件动作，也未完成供应商或实机安全验证**。
>
> 结论状态：这是研发安全审查，不是功能验收、风险接受或安全认证。当前实现不应被表述为具备经验证的安全停机能力。

## 结论速览

| 判定 | 结论 |
| --- | --- |
| 软件隔离 | 推理进程不接触 SDK；协调器是 learned-policy 指令的唯一发布者。 |
| 候选动作门 | 已覆盖关节限位、步长和工作空间；仅在 arm 与 hand 的当前/目标状态齐全时，执行完整 19 自由度转换碰撞检查。 |
| 物理安全 | ESC/FAULT 到物理停止的闭环、XHand 的明确制动动作和执行器物理隔离，均无法由当前代码证明。 |
| 部署判定 | **P0 未关闭前，不应进行常规 learned-policy 物理部署；仅适合不驱动执行器的离线或仿真验证。** |

现有机制能够降低错误模型输出直接到达硬件的风险，但最终防护仍主要依赖软件路径。机械臂 SDK 前的独立限位、双执行器命令的一致提交，以及撤销指令的原子性仍有缺口；应将 P1 项作为启用 learned-policy 物理执行前的准入条件。

### 风险等级含义

- **P0：** 常规物理部署阻断项；需要形成经验证的安全措施。
- **P1：** 物理执行前应修复的代码或契约缺口。
- **P2：** 可靠性、可审计性与后续安全保证能力的整改项。

## 已确认的防护链路

```text
policy worker（仅推理）
  -> policy plan ring
  -> DeploymentCoordinator（唯一 policy 命令发布者）
  -> SafetyGate（输入、限位、步长、工作空间、碰撞）
  -> arm / hand latest-wins command rings
  -> arm_worker / hand_worker（IPC 新鲜度、generation 等校验）
  -> 各自拥有的硬件 SDK

ESC / worker 失活 / 心跳超时
  -> supervisor 将共享状态置为 FAULT
  -> worker 停止消费/发送新的运动命令，或在 ESC 路径中退出循环
  -> 主流程随后执行受监督的 worker 关停
```

- 推理子进程只写 policy plan ring，不拥有机器人 SDK，也不直接发布 arm 或 hand 命令。[`deployment/worker.py`](../dexmani_real/deployment/worker.py)
- 协调器构造候选动作、读取反馈、调用 `SafetyGate`，并作为 learned-policy 指令的唯一生产者。[`deployment/coordinator.py`](../dexmani_real/deployment/coordinator.py)
- `SafetyGate` 对候选动作执行表示/坐标系、有限值、当前反馈、关节限制、单步变化和工作空间段检查；当 arm 与 hand 的当前/目标状态齐全时，还执行转换碰撞检查。拒绝时默认不发布。[`control/safety_gate.py`](../dexmani_real/control/safety_gate.py)
- hand worker 在硬件边界再次验证操作/机械范围及相对实测状态的增量；完整碰撞模型支持 arm 与 hand 组合状态的转换包络检查。[`robot/hand_worker.py`](../dexmani_real/robot/hand_worker.py)、[`planning/collision.py`](../dexmani_real/planning/collision.py)
- 运行时使用 `spawn`、就绪状态、心跳和受监督的子进程收尾；arm home 是可中断、独立的碰撞检查轨迹。[`deployment/lifecycle.py`](../dexmani_real/deployment/lifecycle.py)、[`control/arm_home.py`](../dexmani_real/control/arm_home.py)

这些机制降低了错误模型输出直接到达硬件的风险，但不能替代执行器层的物理安全功能。

## P0：常规物理部署的阻断项

### 1. ESC 是软件急停，未证明为独立、可靠的物理安全通道

键盘 ESC 和操作员回调仅设置共享内存中的 `estop_request`；arm worker 在下一次控制循环观察到该标志后调用 SDK 的 `emergency_stop`，随后在清理中请求状态 4。这个路径依赖进程调度、共享内存、worker 仍在运行、SDK 通信和控制器响应，且没有代码证据表明其满足安全等级、响应时间或单点故障要求。[`teleop/keyboard.py`](../dexmani_real/teleop/keyboard.py)、[`robot/arm_worker.py`](../dexmani_real/robot/arm_worker.py)、[`robot/xarm7.py`](../dexmani_real/robot/xarm7.py)

**要求：**将当前 ESC 明确命名为“软件急停”；在部署安全案例中提供独立的、经制造商规定安装和验证的物理安全停止链路，并测量从触发到停止的实际响应时间。

### 2. XHand 在 ESC/FAULT 时没有明确、已验证的制动或去使能动作

hand worker 在观察到急停时退出循环；其清理主要执行 `XHand.disconnect()`。串口路径仅关闭设备连接；EtherCAT 路径在已知 slave 时尝试进入 INIT，但失败仅记录日志。代码未建立“停止、保持、释放、去扭矩”中任何一种明确的 XHand 物理效果，也没有对串口与 EtherCAT 分别验证。[`robot/hand_worker.py`](../dexmani_real/robot/hand_worker.py)、[`robot/xhand.py`](../dexmani_real/robot/xhand.py)

这不表示手一定会继续运动，而是现有代码无法证明最后一个位置/力控命令在故障时会如何收敛。**要求：**采用供应商确认的显式安全停机/去扭矩 API 或硬件安全链路，并在两种通信模式、位置与力控场景下实机验证。

### 3. `DISARMED` 与一般 `FAULT` 不是可证明的物理隔离状态

arm 初始化即启用运动、设置模式和运行状态；`DISARMED` 的语义是软件不再发布伺服命令而由机械臂保持。一般故障通过共享状态使 worker 停止消费新的运动命令，主流程再执行关停；这不是同步的安全额定断开。故障过程中软件失效、通信异常或控制器保持最后目标时的行为，需要由硬件安全层覆盖。[`robot/xarm7.py`](../dexmani_real/robot/xarm7.py)、[`robot/arm_worker.py`](../dexmani_real/robot/arm_worker.py)、[`runtime/supervisor.py`](../dexmani_real/runtime/supervisor.py)

**要求：**将“软件不发新命令”与“执行器进入安全状态”区分记录；前者不能作为后者的证据。

## P1：启用 learned-policy 物理执行前应修复的实现缺口

| 问题 | 代码事实与影响 | 建议修复 |
| --- | --- | --- |
| arm 硬件边界缺少关节限位复核 | arm worker 的 IPC 校验覆盖形状、有限值、generation、序号与时效，但不校验 arm 关节范围；配置中已有 arm 限位。离线验证中，`qpos_cmd=1e6` 仍被 `worker_validate_arm` 接受。 | 将不可变的 arm 上下限（及必要的 worker 侧最大增量）传给 arm worker，并在每次 SDK 调用前复核。 |
| 运动撤销不是原子的 | `SafetyState.transition(FAULT)` 不会自动推进 generation；supervisor 的多条故障路径只做该转换。worker 可在校验 generation 后、SDK 发送前遇到撤销。 | 令所有安全停止路径原子推进动作 epoch；将状态、epoch 与发送许可绑定，并在 SDK 发送点尽量线性化复核。 |
| arm/hand 命令可能跨帧混合 | 协调器分别写入两个 latest-wins ring，双方无 ACK 或提交协议；两个 worker 独立消费。因此 arm 的第 *n* 帧可与 hand 的第 *n+1* 帧同时执行，候选碰撞检查并未覆盖该组合。 | 使用单一原子动作帧，或使用共同 transaction ID 的两阶段提交；worker 必须拒绝不匹配的 paired action。 |
| 可选 hand 输出可绕过完整碰撞转换 | `JointActionChunk.hand_qpos` 是可选字段；缺失时协调器可形成仅 arm 候选，而碰撞门仅在 hand 起止状态齐全时运行。真实 DexMani 适配器会输出 hand，但契约没有强制这一点。 | 对支持的 manifest/checkpoint 强制 hand 输出；或在 hand 缺失时以当前 hand 目标/状态固定，并仍执行 arm-to-hand 碰撞检查。 |
| 激活计划不会因观测过期自动失效 | 计划在采纳时检查计划和观测时效；激活后仍可按原端点继续发布。命令发布又会刷新 delivery 有效期，使旧观测派生的目标表现为新鲜命令。 | 在每个端点发布前检查观测年龄、计划年龄与截止时间；过期后清空 active/pending plan，并进入定义明确的 hold/安全状态。 |
| deployment 碰撞世界排除桌面 | coordinator 明确不传入 table；operator home 规划则包含 runtime table。README 记录了 VR/键盘遥操作与回放为近桌抓取而排除桌面的取舍，但未找到针对 learned-policy deployment 的正式风险接受记录。 | 将其作为任务级风险接受项：定义可运行工作空间、接近桌面的速度/姿态约束和验证工装；不得暗示部署碰撞检查覆盖完整环境。 |

相关入口：[`robot/command_validation.py`](../dexmani_real/robot/command_validation.py)、[`runtime/safety.py`](../dexmani_real/runtime/safety.py)、[`control/publication.py`](../dexmani_real/control/publication.py)、[`deployment/contracts.py`](../dexmani_real/deployment/contracts.py)、[`deployment/coordinator.py`](../dexmani_real/deployment/coordinator.py)。

## P2：应纳入后续可靠性和可审计性整改

| 问题 | 审查结论 | 建议 |
| --- | --- | --- |
| 观测历史并非时间对齐 | adapter 仅按最近样本数分别取 arm/hand 历史，point cloud 也独立填充；已声明的 observation skew 指标未实际使用。 | 按共同单调时间网格重采样或配对；设置最大 skew，超限拒绝计划，并记录实测 skew。 |
| 观测窗口容量可小于配置/manifest 需求 | arm、hand、point-cloud history ring 的固定容量为 8；配置 horizon 和 manifest `nobs` 可接受大于 8 的值。 | 在运行时按最大允许 `nobs` 分配，或在启动前明确拒绝超出容量的配置。 |
| manifest 与 adapter 的模态声明不一致 | manifest 可声明仅 `point_cloud` 并通过启动校验；DexMani adapter 却无条件构造并传入 `joint_state` 与 `point_cloud`。当前 worker 也无条件采集 arm history，因此这本身**不证明**会运行时丢弃观测；已证实的是模型声明与实际输入契约不一致。 | 要求 manifest 包含 `joint_state`，或将 adapter 改为按 manifest 显式、可验证地路由模态。 |
| 数组转换可能静默改变整数语义 | `freeze_array` / action 合约在检查有限值后直接 cast；例如 `1.9` 可变成 `uint64(1)`，`0.5` 可变成 mask 值 `0`。 | 在转换前验证整型/布尔输入语义、范围和无损性；禁止依赖截断。 |
| 时序字段语义混淆 | plan 的 `target_monotonic_ns` 表示期望执行时刻；发布候选时会重新生成 delivery target，worker 不做 not-before 等待。 | 区分“原始计划时刻”“可发送时刻”“命令过期时刻”，并按选择的控制策略明确执行。 |
| 关键指标未闭环 | 定义了 observation skew、计划 age、stale/superseded 等指标，但部分路径不计量；推理异常或缺反馈也可能跳过指标刷新。 | 为每个拒绝、降级和异常路径定义指标；报警应依据实际采集的安全量。 |
| home 的关闭可等待手动作完成 | operator home 对 hand 调用仅用 ESC 作为 abort，外部 `stop_event` 未直接进入该 abort 路径。 | 将 `stop_event` 纳入 home 的 abort 条件，并在关闭 IPC 前有界地终止 operator。 |
| XHand 错误码的安全语义未被证明 | 某些通信/驱动错误码被视作“可读”或“发送成功”；`1501035` 被当作抓取接触。代码没有以任务阶段、触觉或电流条件限定。 | 以供应商错误码资料和实测证据建立白名单；将接触接受策略限制在明确任务阶段。 |
| tracking error 仅为诊断量 | arm tracking error 被计算/记录，但没有部署侧拒绝或故障策略。 | 制定随时间的阈值、迟滞、持续时间及升级策略，或明确记录其不参与安全决策的风险接受。 |
| 执行器与碰撞配置缺少硬上限 | hand 的 `kp`、`ki`、`kd`、`tor_max` 只校验正负，离线配置解析接受 `1_000_000`；`collision_sensitivity=0` 也可解析。 | 对设备 profile 建立额定安全上限、最小碰撞灵敏度和受审计的专家覆盖流程。 |

## 已执行的离线证据

以下检查均未初始化 SDK 或连接机器人：

- `worker_validate_arm` 对远超范围的 arm 目标返回 `True`，确认 arm worker 缺少范围复核。
- hand 增益/扭矩均设为 `1_000_000`、碰撞灵敏度设为 `0` 时，当前配置解析仍接受，确认缺少安全 envelope 上限。
- hand 字段缺失的候选可通过当前 collision gate，确认完整 19 自由度转换检查在该契约分支未强制执行。
- point-cloud-only manifest 可通过启动校验，确认 manifest 声明与 adapter 实际输入之间存在契约缺口；`nobs=9` 也可通过相关配置校验，而运行时历史容量为 8，确认容量门禁缺口。
- 整数目标/掩码的浮点输入在 cast 后可被接受，确认静默截断风险。
- 已执行目标模块 `compileall` 与 `git diff --check`；它们只说明语法/补丁格式，不说明实时性、设备行为或安全性。

## 建议整改顺序与准入门槛

1. **先建立物理安全案例。** 明确 xArm 与 XHand 的独立安全停止设备、接线、设备端状态和测试规程；将 ESC 保留为软件辅助通道，不作为唯一保护。
2. **修复命令边界。** 为 arm 加入硬件边界限位和增量复核；使 FAULT/撤销原子化；确保 arm/hand 只会成对提交相同 transaction 的命令。
3. **收紧模型与时效契约。** 将 hand 输出、实际输入模态、历史容量、观测对齐和计划终止时间变成启动期可证明的契约；任一不满足时 fail closed。
4. **将配置变成安全 envelope。** 设备额定上限和任务相关最小保护值必须由受控 profile 提供，不能仅依赖操作者提供的数值为正。
5. **补足审计和异常策略。** 为 tracking error、XHand 错误码、观测偏斜、计划过期和 worker 异常定义可观测、可升级、可复现的处置。

在 P0 未关闭前，只宜进行不驱动执行器的离线/仿真验证。P0 关闭后仍需在受控工装、低速低力、空载或等效安全条件下验证：急停响应时间、通信中断、worker 崩溃、陈旧命令、generation 撤销、pair mismatch、碰撞拒绝及各执行器的最终状态。任何一次验证都应保存设备型号、固件、通讯模式、参数、测量方法和原始日志。

## 风险接受记录要求

若项目决定暂不修复某项，风险接受记录至少应说明：适用任务、人员隔离与工装、速度/力/工作空间限制、依赖的设备安全功能、剩余风险、负责人、有效期、复审条件和可回滚步骤。特别是“桌面不参与 deployment 碰撞”“接触错误码可接受”“tracking error 仅诊断”三项，不应只保留在代码注释或 README 中。
