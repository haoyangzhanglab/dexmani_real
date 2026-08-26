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

现有机制能够降低错误模型输出直接到达硬件的风险，但最终防护仍主要依赖软件路径。本轮已收紧机械臂 SDK 前复核、逻辑命令一致性、撤销—发布竞态、模型 hand 合同、计划时效和观测合同；这些是**软件层缓解措施**，不构成物理安全证明。尚未关闭的 P0 与 P1 项仍是 learned-policy 物理执行的准入条件。

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
  -> single coupled latest-wins command ring
  -> arm_worker / hand_worker（IPC 新鲜度、generation 等校验）
  -> 各自拥有的硬件 SDK

ESC / worker 失活 / 心跳超时
  -> supervisor 将共享状态置为 FAULT
  -> worker 停止消费/发送新的运动命令，或在 ESC 路径中退出循环
  -> 主流程随后执行受监督的 worker 关停
```

- 推理子进程只写 policy plan ring，不拥有机器人 SDK，也不直接发布 arm 或 hand 命令。[`deployment/worker.py`](../dexmani_real/deployment/worker.py)
- 协调器构造候选动作、读取反馈、调用 `SafetyGate`，并作为 learned-policy 指令的唯一生产者。[`deployment/coordinator.py`](../dexmani_real/deployment/coordinator.py)
- `SafetyGate` 对候选动作执行表示/坐标系、有限值、当前反馈、关节限制、命令到命令的单步变化和工作空间段检查；当 arm 与 hand 的当前/目标状态齐全时，还执行从实测状态出发的转换碰撞检查。拒绝时默认不发布。[`control/safety_gate.py`](../dexmani_real/control/safety_gate.py)
- 一个 `COUPLED_COMMAND_DTYPE` 记录同时携带 arm/hand target、generation、action ID 与时效；发布时以 ring 返回的递增 sequence 建立 `(run_generation, ring_sequence)` ownership ticket，并立即返回。`action_id` 仅用于审计和 ACK。worker 只执行在其 SDK 边界仍为 active 的最新 ticket；新 record 会原子覆盖旧 ticket。它保证 IPC 逻辑帧一致和非阻塞 latest-wins，**不保证两个执行器物理同步**。[`ipc/schema.py`](../dexmani_real/ipc/schema.py)、[`runtime/safety.py`](../dexmani_real/runtime/safety.py)、[`control/publication.py`](../dexmani_real/control/publication.py)
- arm 与 hand worker 共用 generation 与 delivery window 校验，并在 SDK 前复核 active sequence、运行时 fault/stop、形状和有限值；arm 还复核关节范围，并以“上一条 SDK 接受目标 → 新目标”的 20° 异常跳变阈值兜底（触发即 fail-closed fault），而不以滞后实测位置限制正常命令流；hand 复核操作/机械范围并保留实测状态限速。[`robot/arm_worker.py`](../dexmani_real/robot/arm_worker.py)、[`robot/hand_worker.py`](../dexmani_real/robot/hand_worker.py)
- 运行时使用 `spawn`、就绪状态、心跳和受监督的子进程收尾；arm home 是可中断、独立的碰撞检查轨迹。[`deployment/lifecycle.py`](../dexmani_real/deployment/lifecycle.py)、[`control/arm_home.py`](../dexmani_real/control/arm_home.py)
- inference 进程先独立完成自描述 checkpoint 加载和 manifest 校验；只有其 ready 后才启动任何硬件 worker。每次 `RUNNING` 还原子创建新的 run epoch，观测历史不得读取 epoch 之前的数据。

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

## P1：启用 learned-policy 物理执行前的实现缺口

| 状态 | 问题 | 当前代码事实与剩余边界 |
| --- | --- | --- |
| 已实施 | arm 硬件边界缺少关节限位复核 | `check_worker_arm_command` 现在复核不可变关节上下限、action ID、时效和相邻已接受目标的异常跳变；`_handle_servo_command` 在 SDK 前执行完整复核。实机限位/停止效果仍须验证。 |
| 已实施（软件） | 运动撤销不是原子的 | `motion_lock` 将 state 与 generation 作为 `MotionPermit` 读取；开始、撤销和 IPC 发布在同一短临界区线性化，进入 `FAULT` 也推进 generation。SDK 调用前复核但不持锁，不能替代物理停止。 |
| 已实施（IPC） | arm/hand 命令可能跨帧混合，或旧 record 在覆盖后被延迟执行 | 两条 ring 已替换为单条 coupled record；publisher 在 motion lock 内写完整 record 并更新 active sequence 后立即返回。worker 在 SDK 前复核同一 `(generation, ring sequence)` ownership ticket；action ID 仅承担审计/ACK。覆盖、普通 `RUNNING → ARMED` 停止和 home 取消均会使旧 ticket 失效；home 取消只影响其仍为当前的 ticket。执行器仍独立循环，未声明物理同步或 paired physical ACK。 |
| 已实施 | 可选 hand 输出可绕过完整碰撞转换 | learned-policy `publish_plan` 拒绝缺 hand chunk，coordinator 采纳门也拒绝 `hand_present != 1`；DexMani manifest 强制 hand-enabled 的 19D/21D 合同。 |
| 已实施 | 激活计划不会因观测过期自动失效 | coordinator 对每个 active endpoint 使用不可延长的 `min(inference_finished + max_plan_age, latest_physical_source + max_source_to_command_age)`；已过期前缀直接跳过，计划到期时清空 active/pending 并撤销 RUNNING。 |
| 未关闭 | deployment 碰撞世界排除桌面 | coordinator 仍不传入 table；需要任务级风险接受、工作空间/速度约束和受控工装验证，不能暗示碰撞检查覆盖完整环境。 |

相关入口：[`robot/command_validation.py`](../dexmani_real/robot/command_validation.py)、[`runtime/safety.py`](../dexmani_real/runtime/safety.py)、[`control/publication.py`](../dexmani_real/control/publication.py)、[`deployment/contracts.py`](../dexmani_real/deployment/contracts.py)、[`deployment/coordinator.py`](../dexmani_real/deployment/coordinator.py)。

## P2：后续可靠性和可审计性整改

| 状态 | 问题 | 当前代码事实与剩余边界 |
| --- | --- | --- |
| 已实施 | 观测历史并非时间对齐，且可能混入上一次 run | 新 run epoch 后以因果截点前最新已过去的 policy tick 为窗口末端，在控制网格上选择严格递增且不复用的 cloud；每个 cloud 仅匹配 `source_time <= cloud_time` 且在 `max_observation_skew_s` 内的最新 arm/hand frame。数据不足、grid lag 过大、跨 camera generation 或超 skew 时不推理；不插值、不填充。point-cloud 历史最大读取年龄为 `max_input_age + (T-1)*dt + max_grid_lag`，state 还增加 `max_observation_skew`；最新源帧仍单独受 `max_input_age` 限制。 |
| 已实施 | 观测窗口容量可小于配置/manifest 需求 | point-cloud deployment 按 horizon、控制周期、camera FPS、arm/hand loop Hz、最大输入年龄、grid lag、skew 与读取余量推导 state/point-cloud ring 容量；manifest `n_obs_steps` 必须等于 deployment horizon。最坏拷贝/调度预算仍应通过负载测试校准。 |
| Real 侧已实施；Policy 生产端待完成 | manifest 与 adapter 的模态/数据域声明不一致 | DexMani runtime 要求严格的 `joint_state + point_cloud`、`arm_qpos,hand_qpos,point_cloud` 合同；checkpoint 还必须携带 Real Policy Zarr v5 数据合同，部署声明的 `task_name`、训练 `dt`、`obs[t]_before_action[t]`、camera-source state alignment、观测 skew、动作 endpoint 限制、点云 shape、算法、配置哈希与桌面平面逐项匹配 realtime worker。当前 policy checkpoint 尚不生产完整合同，因此会在硬件启动前拒绝；实施清单见 [`dexmani_policy_integration_followup.md`](dexmani_policy_integration_followup.md)。 |
| 已实施 | 数组转换可能静默改变整数语义 | `freeze_array` 在 uint wire cast 前要求原输入为整数 dtype 且范围可表示；浮点 timestamp/mask 与负数到 uint 均 fail closed。 |
| 已实施 | 时序字段语义混淆 | plan 的 `target_monotonic_ns` 保持原始策略网格端点；coupled command 另存 `scheduled_target_monotonic_ns` 与 worker delivery target。采纳时跳过无足够 lead 的旧端点，既不重戳 action chunk，也不掩盖其原始时刻。 |
| 关键指标未闭环 | 定义了 observation skew、计划 age、stale/superseded 等指标，但部分路径不计量；推理异常或缺反馈也可能跳过指标刷新。 | 为每个拒绝、降级和异常路径定义指标；报警应依据实际采集的安全量。 |
| 已实施 | home 的关闭可等待手动作完成 | operator 的 hand/home 等待同时观察 `stop_event`、runtime shutdown、quit、fault 与软件急停；lifecycle 先有界 join operator，再关闭共享内存。若 operator 未退出，则先验证停止全部子进程并保留仍可能被线程访问的共享内存，不会先抛异常而遗留 hardware worker。普通 shutdown/fault 不再被错误升级为物理 e-stop 请求。 |
| XHand 错误码的安全语义未被证明 | `1501070` 发送 CRC 现在保持“交付未确认”：不中止 worker，也不产生 action ACK；`1501035` 仍被当作抓取接触并接受。代码没有以任务阶段、触觉或电流条件限定后者。 | 以供应商错误码资料和实测证据建立白名单；将接触接受策略限制在明确任务阶段。 |
| tracking error 仅为诊断量 | arm tracking error 被计算/记录，但没有部署侧拒绝或故障策略。 | 制定随时间的阈值、迟滞、持续时间及升级策略，或明确记录其不参与安全决策的风险接受。 |
| 执行器与碰撞配置缺少硬上限 | hand 的 `kp`、`ki`、`kd`、`tor_max` 只校验正负，离线配置解析接受 `1_000_000`；`collision_sensitivity=0` 也可解析。 | 对设备 profile 建立额定安全上限、最小碰撞灵敏度和受审计的专家覆盖流程。 |

## 已执行的离线证据

以下检查均未初始化 SDK 或连接机器人：

- 审查初版曾证明 arm 越限、缺 hand chunk、point-cloud-only manifest 和超过固定 history 容量可通过旧合同；这些发现对应的路径已由本轮回归检查覆盖，现应 fail closed。
- 构造 coupled record 的 IPC round-trip，确认 arm/hand payload、action ID、generation 与时效在同一记录中传递；hand 缺席会清除 hand worker 的待执行目标。
- 用共享 `RLock`/`Value` 的替身验证非阻塞发布、sequence ticket 覆盖、`RUNNING → ARMED` 撤销、hand-only command 与定向取消；旧 generation 或旧 sequence 的 ticket 均不再 active。
- 构造 run epoch、状态历史和 point-cloud 时间戳，确认旧 run frame、重复 camera frame、晚于 grid 的 frame 与超过 skew 的配对均被拒绝。
- 验证 strict manifest、real 侧对非自描述 checkpoint 的拒绝边界、按 horizon/skew/grid span 推导的 ring 容量、plan 因果时序/不可延长截止时间、过期前缀跳过，以及 arm worker 对越限/相邻命令异常跳变的拒绝。Policy 生产端的 round-trip 与跨仓正向 preflight 仍待后续完成。
- 当前仓库离线回归命令与结果应以本次变更交接记录为准；这些检查只说明离线语法、格式和定向合同，不说明实时性、设备行为或安全性。

## 建议整改顺序与准入门槛

1. **先建立物理安全案例。** 明确 xArm 与 XHand 的独立安全停止设备、接线、设备端状态和测试规程；将 ESC 保留为软件辅助通道，不作为唯一保护。
2. **完成 P1 剩余闭环。** 为 coupled command 设计与验证跨 worker 的 applied/physical-state 语义；将桌面纳入明确的 task safety case，或限制任务并正式接受风险。
3. **建立配置安全 envelope。** 设备额定上限和任务相关最小保护值必须由受控 profile 提供，不能仅依赖操作者提供的数值为正。
4. **补足审计和异常策略。** 为 tracking error、XHand 错误码、观测偏斜、计划过期和 worker 异常定义可观测、可升级、可复现的处置。

在 P0 未关闭前，只宜进行不驱动执行器的离线/仿真验证。P0 关闭后仍需在受控工装、低速低力、空载或等效安全条件下验证：急停响应时间、通信中断、worker 崩溃、陈旧命令、generation 撤销、pair mismatch、碰撞拒绝及各执行器的最终状态。任何一次验证都应保存设备型号、固件、通讯模式、参数、测量方法和原始日志。

## 风险接受记录要求

若项目决定暂不修复某项，风险接受记录至少应说明：适用任务、人员隔离与工装、速度/力/工作空间限制、依赖的设备安全功能、剩余风险、负责人、有效期、复审条件和可回滚步骤。特别是“桌面不参与 deployment 碰撞”“接触错误码可接受”“tracking error 仅诊断”三项，不应只保留在代码注释或 README 中。
