# ManiUniCon 参考设计：DexMani learned-policy 部署改进

> 用途：将 ManiUniCon 中有价值的机制思想转化为 DexMani Real 的可执行改进项。
>
> 范围：策略观测、推理、chunk 调度和命令 IPC；ManiUniCon **不是**安全实现范本。
>
> 依据：对本地 ManiUniCon reference checkout 的静态源码审查；未运行其中的硬件代码，未连接设备。

本文件是设计输入，不改变 [`deployment_review.md`](deployment_review.md) 的风险等级和物理部署准入条件。出现冲突时，优先级为：当前源码 → IPC schema/运行时配置 → 本文档。

> 实施状态：本文采纳的软件整改已落实到当前源码，并完成离线合同验证；本文保留原始设计理由。它们不关闭物理安全 P0，且 coupled IPC 只保证逻辑记录一致，不证明双执行器物理同步。

## 1. 决策速览

| 状态 | 机制 | DexMani 决策 |
| --- | --- | --- |
| 保持 | 推理子进程内惰性加载模型 | 已采纳；保持模型与 SDK/IPC 的隔离。 |
| 已实施（IPC） | 完整记录的原子 latest-wins 发布 | 单一 `COUPLED_COMMAND_DTYPE` 取代两条命令 ring；publisher 原子更新 active sequence 后立即返回，worker 在 SDK 前复核同一 ticket；不承诺物理同步。 |
| 已实施 | 按读取预算推导 ring 容量 | point-cloud deployment 按 horizon、模态频率、skew 与读取余量分配容量；manifest、horizon 与模态合同在 load 时严格校验。 |
| 实验性 | future chunk conditioning | 仅限支持该输入合同的模型 backend，并以 generation/时间线隔离缓存。 |
| 实验性 | 笛卡尔轨迹插补 | 只作为 gate 前的纯候选生成；逐个实际端点仍须经安全发布边界。 |
| 禁止照搬 | 模型直写执行队列、clip 后继续、旧动作重戳 | 维持“模型提案 → coordinator → SafetyGate → worker 复核”。 |

ManiUniCon 最值得借鉴的是完整记录提交、容量预算和模型连续性思路；它没有可直接采纳的 P0 物理安全方案。

### 必须保持的安全边界

```text
PolicyRuntime（仅模型）
  -> policy_plan_ring（提案）
  -> coordinator（采纳、时效、重规划）
  -> SafetyGate + 当前反馈（拒绝而非裁剪）
  -> coupled command + worker active-ticket 复核
  -> 各自拥有 SDK 的 arm / hand worker
```

模型连续性、FIFO 完整记录和插补只能提升性能或一致性；它们不构成碰撞、限位、急停或硬件停机机制。

## 2. 采纳原则

| 原则 | 对 DexMani 的要求 |
| --- | --- |
| 模型输出只是提案 | 推理 worker 不得拥有 SDK、SafetyState 写权限或 arm/hand 命令发布权限。 |
| 安全拒绝而非修正 | 限位、工作空间、delta、时间或 provenance 不满足时，丢弃整个候选；不得 clip 后继续执行。 |
| 单一时钟域 | 观测 anchor、计划目标时刻、过期检查和 worker freshness 一律使用 host monotonic 时间。 |
| 撤销优先于连续性 | generation 变化、FAULT、e-stop 或反馈失效时，必须清空模型内部连续状态和未提交计划。 |
| 安全检查覆盖实际命令 | 若新增插补、ensemble 或 command bundle，最终每个下发端点仍须通过现有安全发布边界。 |

## 3. 值得采纳的机制

### 1. 将 arm/hand 作为一条原子 latest-wins 记录发布

ManiUniCon 的 `SharedMemoryQueue` 先写入同一条完整动作记录的所有字段，再以 atomic write
counter 提交。这种“完整 payload 后提交 marker”的思路，比两个独立 latest-wins 通道更容易
保证消费者看到的是同一逻辑动作。

DexMani 已将两条命令通道替换为一个面向执行层的不可变 coupled record；policy worker 仍只写 plan：

```text
coordinator
  -> COUPLED_COMMAND_DTYPE {
       run_generation, observation_id, action_id,
       created_monotonic_ns, scheduled_target_monotonic_ns,
       target_monotonic_ns, valid_until_monotonic_ns,
       arm_present, hand_present, arm_qpos[7], hand_qpos[12]
     }
  -> ring write returns ring_sequence
  -> publisher marks that sequence active and returns immediately
  -> arm worker + hand worker each recheck that exact active ticket before SDK
```

**边界与取舍：**

- 这只保证 IPC 记录的一致性，不能保证两个执行器在物理上同时动作。
- ticket 仅由 `(run_generation, ring_sequence)` 标识 ownership；`action_id` 是 record 中的审计/反馈身份。新 record 在锁内覆盖旧 ticket；旧
  worker 即使延迟到达也无法再执行。普通停止与定向 home 取消同样会撤销 ticket，
  其中 home 取消不会影响已取代它的新命令。
- `hand_present` 是明确字段而非字段缺失。learned-policy plan 强制 `hand_present=1`；非策略调用的 arm-only 记录会使 hand worker 清除先前活跃 target，不能让手继续追逐旧命令。
- 发布前仍由 coordinator 读取当前 arm/hand 反馈并调用 `SafetyGate`；worker 在 SDK
  前仍复核 generation、时效、形状、有限值、关节范围和必要的增量。
- active ticket 是防止旧 latest-wins record 迟到执行的逻辑门禁，不是跨 worker barrier，也不是“两个执行器已物理到位”的 paired ACK；`accepted_target_action_id` 只表示 SDK 接受精确目标，不表示动作完成。任何把它提升为动作完成语义的后续设计都须另行验证。

**推荐 owner：** `ipc/schema.py` 定义 wire record，`control/publication.py` 是唯一 producer，
`robot/arm_worker.py` 与 `robot/hand_worker.py` 是各自 consumer；不要让该协议泄漏到
`PolicyRuntime`。

### 2. 从工作负载推导 history ring 容量

ManiUniCon 的 ring buffer 容量采用以下思路：

```text
capacity >= requested_history
          + ceil(producer_hz * worst_case_copy_s * safety_margin)
```

后半项为读取者复制最近 `requested_history` 预留 writer 不能覆盖的槽位。这比为所有模态写死
一个容量更可审计。

DexMani 已将该原则用于 point-cloud deployment 的 arm、hand 与 point-cloud history allocation；当前实现依据 horizon、camera FPS、arm/hand loop Hz、最大 skew 和读取余量计算容量，并在模型加载时校验 `n_obs_steps` 与 deployment horizon。仍应补充高负载实测预算：

1. `observation_horizon`、manifest `nobs` 与每个 requested modality 的容量一致；
2. 容量不足时拒绝启动，绝不截断或默默重复旧帧；
3. 大 point cloud 的 `worst_case_copy_s` 来自保守测量或明确配置，而不是理想平均值；
4. 记录解析后的容量、horizon 和预算，供部署日志与复现使用。

DexMani 的 ring 实现与 ManiUniCon 不同，因此应采纳**容量推导原则**而非直接移植公式或等待行为。
它解决的是并发读写正确性，不替代 freshness、age 或跨模态 skew 检查。

### 3. 保持模型资源在推理子进程内惰性初始化

ManiUniCon 在 policy process 的 `run()` 中实例化模型和 wrapper；这避免将 CUDA context 或
checkpoint 对象跨进程传递。DexMani 的 `load_policy_runtime()` 已采用同一更严格的原则：父进程
不导入模型，worker 内 `load()` 失败即由 supervisor 可见。

此项是**已采纳、应保持**的机制，而不是新增抽象。后续模型接入应继续遵守：

- factory 只接收冻结的 `DeploymentConfig`，不接收 `RuntimeChannels` 或 SDK；
- 构造、`load()`、`reset_episode()` 和非预期 `predict()` 失败必须使 supervisor 可见；
  预期的无效模型输出可按既有语义丢弃，并由 command-silence watchdog 收敛；
- `close()` 是清理路径，不得被当作已证明的安全停机。当前实现记录其异常；若未来需要把
  清理失败作为部署失败，应由 shutdown report 显式承载；
- checkpoint 和 runtime 配置的 hash 写入一次性 provenance 日志；模型 resolved inference
  config 与训练数据合同必须内嵌在 checkpoint 中，不接受运行时第二份模型 YAML；Real
  部署逐项比较训练 `dt`、点云配置/桌面平面与 realtime worker；
- 不添加“加载失败则 fake policy 接管”的降级路径。

### 4. 受约束地试验 real-time chunk conditioning

ManiUniCon 的部分模型 backend 会保存上次原始预测 chunk，并把仍处在未来的部分作为下一次
推理的 `local_cond`。这可能降低滚动重推理时的动作抖动、使短 chunk 之间更连续。

在 DexMani 中，它只能作为 `PolicyRuntime` 内部、且模型原生支持该输入合同的实验性优化；不应
为此向所有 backend 添加通用 `local_cond` 抽象。建议先以 shadow/offline 评估接入：

- 缓存项必须绑定 `run_generation`、`observation_id`、模型版本和每个 action 的原始
  `target_monotonic_ns`；任一不连续即清空缓存。
- 仅把仍在未来、且未被 coordinator 消费或过期的原始预测作为条件；不能读取实际 command
  ring，更不能将条件当作执行确认。
- 输出仍必须是新的 `JointActionChunk`，遵守严格递增 target、容量和 valid mask 校验；旧 chunk
  不能重戳时间后复活。
- 用离线 episode/replay 比较 tracking、jerk、endpoint coalescing、计划过期率和安全拒绝率；
  不以视觉平滑感受替代指标。

在没有实验证据前，保持当前无状态 fake/runtime 行为更简单且更安全。

### 5. 将笛卡尔插补保留为 gate 前的纯计算候选

ManiUniCon 的 `PoseTrajectoryInterpolator` 使用位置线性插值和旋转 Slerp，并按位置/角速度下限
延长 waypoint 时间。其数学部分可作为 DexMani 的候选参考，用于未来的 EE action adapter 或
低层轨迹生成。

采用时应满足：

1. 插补只产生 candidate，不可直接调用 SDK；
2. 所有实际下发的插补点都经 IK、工作空间、限位、delta 和完整碰撞转换检查；
3. 速度限制采用经配置审核的物理单位，且在时间不足、IK 失败、碰撞不可判定时拒绝；
4. 插补时间来自单调时钟的计划时间线，不使用 `time.time()` 与 `time.monotonic()` 的差值换算；
5. 先验证所有插补段的碰撞包络和最坏计算预算，再用于物理执行。

当前 coordinator 已按模型端点逐 tick 发布。除非测量表明端点控制确有连续性问题，否则不应为了
“更平滑”提前引入新的轨迹执行层。

## 4. 不应照搬的 ManiUniCon 机制

| ManiUniCon 做法 | 为什么不能采纳 | DexMani 的替代原则 |
| --- | --- | --- |
| 策略直接写 action queue | 模型输出绕过独立计划采纳、时效和安全 gate。 | 推理只写 plan；coordinator 是唯一 learned-policy 命令 producer。 |
| `validate_action()` clip 后返回成功 | 静默改写模型意图，且掩盖越界/单位错误。 | 拒绝整个候选并记录 machine-readable reason。 |
| chunk 全部过期后重发最后一步 | 将过期模型结果伪装为新命令。 | 丢弃计划，由 command-silence/first-command watchdog 收敛。 |
| wall clock 与 monotonic clock 混用 | 到期、过期与时延无法可靠比较。 | 全部关键实时语义使用 monotonic ns。 |
| Event 的无超时握手 | policy 或 robot 卡死时可无限阻塞，且 reset 无 epoch。 | readiness、heartbeat、generation 与受监督 shutdown。 |
| 不足 history 时重复最早帧 | 把“数据不足”伪装为正常时序观测。 | 数据不足时直接等待或拒绝，绝不重复填充。 |
| 相机时间取均值 | 丢失最大 skew，不能证明多相机或机器人状态对齐。 | 保留每模态 source/publish 时间并按最大 skew 门限配对。 |

## 5. 与现有 deployment 整改项的关系

| 改进项 | 来源 | 解决的现有问题 | 优先级 |
| --- | --- | --- | --- |
| single coupled command record + active sequence ticket | 完整记录提交思想 | arm/hand 跨帧混合、覆盖后迟到执行 | 已实施（IPC）；不等于 physical ACK |
| 容量推导与启动门禁 | ring 容量公式 | `nobs` 与固定 history 容量不一致 | 已实施；仍须压力测量读取预算 |
| 跨模态对齐 adapter | ManiUniCon 的缺口反例 | count-aligned、非 timestamp-aligned history | 已实施：point-cloud 时间线上的因果最近邻 + skew 拒绝 |
| active plan 每端点过期复核 | ManiUniCon 的 stale 重排反例 | 已采纳计划继续执行旧观测端点 | 已实施：跳过过期前缀；不可延长 plan/source deadline；保留原始 scheduled target |
| chunk conditioning | 模型连续性机制 | 可能的重推理抖动 | 实验性，不得阻塞安全整改 |
| 插补候选器 | 纯几何辅助 | 未来 EE 连续性需求 | 实验性，不得绕过 gate |

ManiUniCon 对 P0 物理安全没有可采纳的解决方案。独立安全停止链路、XHand 故障时的明确执行器
状态和软件状态与物理状态的证据，仍按 `deployment_review.md` 的 P0 要求处理。

## 6. 剩余工作

- 完成 P0 物理安全案例，并定义 coupled command 的跨 worker physical-state 证据；active ticket 不能作为动作完成确认。
- 为 actuator application/skew reject、history capacity 和 active-plan expiry 补齐指标与报警。
- 仅在离线证据充分后，针对特定模型试验 chunk conditioning 或插补；两者都不得绕过现有 gate。

## 7. 验收证据

每一项实现都至少应有以下离线验证；它们不替代实机或物理安全验证：

- **publication：** 模拟 arm/hand worker 的乱序、延迟、覆盖、撤销和 generation 变化，确认旧 sequence ticket 不再 active，且定向取消不撤销新命令；
- **容量：** 使用最大合法 `nobs`、最慢读取预算和高于标称的 producer 速率，确认启动拒绝或不覆盖；
- **观测：** 构造 camera/arm/hand skew、时钟跳变、重复序列和重启 generation，确认只有符合 contract 的 batch 到达模型；
- **chunk conditioning：** generation 切换、target 已过期、observation ID 回退时，确认缓存必定清空；
- **插补：** 非法旋转、不可达 IK、碰撞段、超速度和过期 waypoint 必须被拒绝，不能发送 SDK 命令；
- **回归：** `python -m compileall -q dexmani_real examples`、`git diff --check`，以及相关纯函数和 IPC contract 测试。

任何涉及物理执行的后续验证仍须先满足 deployment 审查中的 P0 准入要求，并在受控工装、低速低力
条件下执行。
