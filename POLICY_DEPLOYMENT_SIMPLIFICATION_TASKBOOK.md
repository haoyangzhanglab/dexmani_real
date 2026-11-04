# 策略部署推理轻量化实施任务书

状态：方案已确定，待实施。本文的发布不代表源码改造、回归测试或真机验证已经完成。

日期：2026-11-04。

## 1. 目标、基线与执行原则

将 DexMani Real 的策略部署整理成个人研究中容易阅读、修改和重复实验的直接流程：模型加载一次，设备连接一次，按需预热，等待操作者开始，运行一段，停止并完成必要收尾，再开始下一段。

完成标准是减少需要同时理解的状态、时序承诺和异常分支，不以文件数、类数或代码行数作为指标。允许明确改变旧的执行约定；不以新增框架、兼容模式或隐式兜底维持全部旧行为。

本任务的审查基线：

| 仓库 | 固定版本 | 用途 |
|---|---|---|
| dexmani_real | [`b53eac89c9cb28c08333e7af96680020e072c380`](https://github.com/haoyangzhanglab/dexmani_real/commit/b53eac89c9cb28c08333e7af96680020e072c380) | 本轮实施仓库 |
| dexmani_policy | [`52cbcb6fb7c37c8d0cb29ac63f6d00d85d353d0a`](https://github.com/haoyangzhanglab/dexmani_policy/commit/52cbcb6fb7c37c8d0cb29ac63f6d00d85d353d0a) | 核对当前模型输入、历史和 RTC 接口 |

开始实施时先阅读 [AGENTS.md](AGENTS.md)，检查当前分支、HEAD 和工作区。基线之后的实际修复必须保留；不得回退到本文快照覆盖新代码。本文明确改变的部署行为替代旧部署任务书、README 和测试中的相冲突约定，其他行为以当前源码和最新任务要求为准。

本轮只实施 Real 的部署链路及其必要调用方、CLI、文档和测试。Policy 仓库用于核对接口，不修改模型、训练配置或 checkpoint。常规实现选择自行解决；若最新源码使某项设计不再成立，应指出具体证据并作最小一致调整，不保留相互矛盾的两套实现。

## 2. 参考项目与采用范围

以下参考版本均已对照实际部署入口及其下游代码。借用局部执行方式，不搬运整个框架。

| 项目 | 本轮采用的做法 | 不移植的部分 |
|---|---|---|
| [LeRobot `ba83e4cf`](https://github.com/huggingface/lerobot/tree/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3) | 同步入口独立；RTC 请求保存队列快照、返回后按消费进度接续；首帧填满历史 | 通用执行器、task/provenance/generation 状态、trained RTC、多个等待期限和 processor 体系 |
| [ManiUniCon `85c6f2e3`](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | 观测转换、模型、动作转换的直接链路；窗口处理局部化；RTC 条件只在相关分支出现 | 全局 SharedStorage、多进程握手、录制开关支配动作发送、过期动作重新赋时 |
| [Dexora `6f2869d6`](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783) | 模型和设备在外层初始化一次；同步循环直接消费动作数组 | 为解释器隔离而设置的 ZMQ forwarder、输入自动补零、忽略设备错误 |
| [pi-r2-flow `3af52ca4`](https://github.com/pi-r2-flow/pi-r2-flow/tree/3af52ca400a6ec7d141416879aa531a6f62f697a) | chunk、消费位置和一个待接收结果；围绕剩余动作预取；显式评估记录 | 远程模型服务、多种 streaming/ensemble 路径、重复最后动作、整段图像缓存在 list |

参考事实须准确：LeRobot guided RTC 的合并裁剪使用 `min(ceil(elapsed/dt), cursor_delta)`；普通异步 CHUNK 默认追加。Pi-R2 的预取参考消费索引，但部分交接仍按墙钟裁剪。它们没有共同要求所有模式使用同一种时间语义。本文保留普通 async 的既有时间锚点，只替换 RTC 的严格槽约定。

主要依据：

- [LeRobot 同步入口](https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/inference/sync.py#L94-L146)、[RTC 接收](https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/inference/execution.py#L471-L512)、[队列合并](https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/policies/rtc/action_queue.py#L236-L279)、[历史填充](https://github.com/huggingface/lerobot/blob/ba83e4cf43737e2665be0aafcb5f513b83c7d0f3/src/lerobot/policies/utils.py#L29-L51)。
- [ManiUniCon 推理与 RTC 分支](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py#L247-L301)、[chunk 筛选](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/act_wrapper/chunk_wrapper.py#L37-L107)。
- [Dexora 同步循环](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py#L310-L363)、[多轮实验入口](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py#L439-L483)。
- [Pi-R2 可选记录](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L143-L161)、[结果单槽与交接](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L631-L680)、[消费与预取](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L876-L980)。

## 3. 已确定的取舍

| 项目 | 本轮决定 |
|---|---|
| sync / async / RTC | 三种能力保留，控制主体只分同步和异步两条可顺读流程；RTC 是异步流程中的局部接续分支 |
| RTC 精确 slot/handoff | 删除；不保留 strict/relaxed 兼容模式 |
| 每动作完整模型输入 TTL 门禁 | 删除；模型输入、设备反馈、录制质量在各自实际用途处判断 |
| 首次 N 次采样等待 | 删除；当前主训练链路对应的首帧重复填充作为默认 |
| 模型加载与设备连接并行启动 | 改为顺序启动，模型加载失败时设备尚未连接 |
| 辅助配置/结果强制写入 | 改为显式请求输出；普通保存异常仅报告，不决定控制能否继续 |
| task 与目录层级绑定 | 删除 policy/task/experiment 目录要求；task 只作为元信息 |
| Policy 音频 | 本方案选择终端提示，移除 PolicySession/Rollout 的音频依赖；其他入口的音频保留 |
| 停止与资源收尾 | episode 运行边界统一负责停止；正常段回收后才复用，最终关闭仍由 Session 负责 |
| Raw | 默认关闭；显式开启仍为固定完整多模态、有界流式录制，不另建轻量日志 schema |
| 模型 Worker | 保留唯一线程、唯一在途任务、段间回收和最终关闭；不新增资源管理框架 |
| 硬件和共享数据 | 保留当前 Owner、邮箱、租期、取消同步、共享快照和实际错误传播 |

以下已完成事项不再作为待办：旧 `tactile_source` 门禁已删除，recipe 键集合检查已放宽，Canonical 的 teleop 来源门禁已删除，模型共享 cleanup deadline 已移除，音频普通关闭异常已降级，joint/HOME 几何依赖与 Owner 冗余协议已收缩。

不新增：BaseRunner、调度器注册表、事件总线、历史策略注册表、通用健康管理器、统一 provenance/年龄协议、远程推理服务、自动重试恢复、旧行为兼容层。简单函数、少量字段和现有类型足够。

## 4. 目标行为规范

### 4.1 符号与配置

- `P = info.future_steps`：Policy 返回的完整未来动作长度。
- `A = action_steps`：sync 每次最多提交的步数；async/RTC 的名义执行窗口。
- `d = prefetch_steps`：异步预取提前量；RTC 同时用它估计 guidance 延迟，不构成准时交接承诺。
- `dt = info.control_dt_s`：从 checkpoint 恢复的名义周期。
- `L`：当前已接纳动作数组的总长度；接收裁剪后可以小于 P，在本计划内保持不变。尚未提交的动作数为 `L-cursor`。
- `cursor`：当前 RTC/sync 动作数组中下一次要提交的位置，只在 `send_action()` 正常返回后推进。

配置校验固定为：sync 要求 `1 <= A <= P`；async 和 RTC 要求 `1 <= d <= A <= P` 且 `d < P`。RTC 保留显式、有限、非负的 guidance cap。删除 RTC 的 `A+d <= P` 要求，执行窗口按实际长度缩短。sync 不校验未使用的预取/RTC 参数；不自动把配置错误降级成另一种模式。

保留当前 dt、单位、关节顺序、动作维度和实际设备频率限制。不得撤销基线提交中 `sampling_clock_ns()`、XHand 83 Hz 实际目标发送限制、xArm 模式准备和反馈舍入修复。

### 4.2 启动、多轮实验与所有权

顺序为：配置/PolicyInfo 准入 → 启动操作输入 → 在唯一模型线程加载并等完成 → 连接设备与所需传感器 → 用真实快照做现有一次会话预热 → 待命。

加载等待期间可响应取消；无需连接机器人来维持该等待。预热仍在模型线程中，RTC 仍覆盖其实际推理路径，不把 warmup 删成未经说明的首轮延迟变化。

Session 持有唯一 Worker 和设备；Rollout 借用它们。每段首次正式预测执行现有 `reset_episode()`，旧任务完成并取出异常之前，不进入下一段 reset/predict。正常段只回收任务，不关闭或重载模型。最终仍在同一模型线程串行关闭。

保留现有局部资源等待和未完成句柄归属，不引入跨层共享总清理预算。线程/进程等待超时报告未完成，禁止下一轮复用；不得宣称后台任务已终止或 Python 必定在该期限内退出。

### 4.3 观测、历史与每周期工作

保留准备期的真实初值等待、writer 启动后重新取得反馈、streaming 准备后的有效观测，以及最新 `arm_reference_from_feedback()` 准入；不把这些不同阶段的检查当作重复证明删除。

正式运行采用以下规则：

1. 每个正常控制采样机会至多读取一份观测。服务反馈/按键的短轮询不重复加入历史，也不制造 Raw 行。
2. 模型输入有效时，将 `ObservationBuilder.project(row)` 加入历史；首次有效快照重复填满长度 N，之后每个实际采样最多更新一次。允许同一传感来源被正常重复读取，不增加帧去重、补采样或等间隔重建。
3. sync 等待预测期间仍按正常控制采样机会更新有效历史；结果完成不额外触发一次采样或提前发送。
4. 发起推理时要求当前模型输入有效，并冻结 `tuple(history)`。后续 deque 更新不改变已提交请求，快照数组继续只读；FK、手点云构造及模型预处理继续在模型线程执行，不新增大数组 deepcopy 或缓存管理层。
5. 相机/点云/模型用触觉暂时不可用时，不更新模型历史、不发起新请求；已有可用动作仍可依据有效关节反馈执行。传感进程真实死亡、SDK 错误和数值错误按现有技术故障传播，不伪装成可忽略的缺帧。
6. 映射后、发送前只复查实际设备反馈和目标所需条件，以及操作取消、运行期限。删除对整套 `control_needs` 的再次门禁；真实接触保护若另有消费者，其检查仍属于该消费者。
7. 开启完整 Raw 时，另按完整录制要求检查本次行。相机等录制质量失败仍可结束本段；触觉保持值沿用当前允许规则。因此“模型输入缺帧时继续旧 chunk”不豁免显式 Raw 完整性要求。

首帧填充依据本仓训练实现，而非只类比外部项目：[统一 pad_before](https://github.com/haoyangzhanglab/dexmani_policy/blob/52cbcb6fb7c37c8d0cb29ac63f6d00d85d353d0a/dexmani_policy/training/common/config.py#L62-L67)、[边界行重复采样](https://github.com/haoyangzhanglab/dexmani_policy/blob/52cbcb6fb7c37c8d0cb29ac63f6d00d85d353d0a/dexmani_policy/datasets/sampler.py#L81-L98)。本约定面向当前主训练链路，不声称所有外部自定义 checkpoint 都有相同历史分布。

### 4.4 sync

sync 的主状态只有当前动作数组、消费位置和 Worker 的在途任务。无动作且无在途任务时，使用当前有效输入提交预测；收到结果后顺序提交前 `min(A, P)` 行，执行完再请求。

等待模型时持续服务反馈、操作输入、正常频率的观测/录制；不忙等、不阻塞整个主线程。慢计算降低实际发送频率，不补发积压 tick，不因跨过名义 tick 而丢弃有效动作。每轮只等待剩余周期。

sync 不使用 RTC/async 的 slot、handoff 和前缀状态。首轮等待不新增预测年龄门禁；继续服从显式实验时长和操作者停止。已有目标的设备租期照常生效，等待时不续租旧目标；Owner 已停止时按现有停止路径结束本段，不自动清 stop 恢复。

### 4.5 普通 async

保留已有时间锚点语义：预测对应请求时的观测槽，返回时按当前时间裁掉已过前缀；A 决定名义执行窗口和预取位置，等待期间可使用旧预测的完整有效尾部。映射跨过一个名义 tick 时，只要完整预测窗口尚未结束、设备反馈及运行期限仍有效，可提交该次候选。

每个控制机会最多发送一个动作，不补发错过的行。正常时间跳过不再形成 RTC 式失约错误。尽量保留现有小型时间窗口函数，不重写成通用队列框架，也不将 async 改成 LeRobot 的追加式 CHUNK。

本轮明确改变耗尽处理：先尝试接收本次已完成且可用的新结果，再判断是否还有有效动作；两者都没有时结束当前段，清除旧计划并停止设备。不得清空 plan 后无限等待、反复 bootstrap、重复末动作或重新赋时。首个 async 预测返回时若完整窗口已耗尽，同样结束本段并说明推理超出可用窗口。

首个预测尚在运行时允许等待。运行中缺少新推理输入时继续有效旧尾部；尾部耗尽后执行上述结束规则，不新增等待宽限期。

### 4.6 RTC：按消费进度接续

RTC 使用策略动作数组和消费位置，不按墙钟 slot 选择动作。借鉴 LeRobot 的请求快照和消费对齐，在本仓单消费者、每周期至多一次提交、不补发的执行方式下，直接以请求后提交数裁剪。无需移植其多模式队列、generation 或 provenance 管理。

**每份当前计划的名义窗口为 `W = min(A, L)`。** 返回早的结果等到该窗口完成再接续，避免快速模型使 A 失去意义、每个 tick 都更换计划。结果晚到时继续消费旧计划在 W 之后的尾部，不要求精确命中 W。

行为规则如下：

1. bootstrap：首个有效输入提交无前缀预测；无旧动作可消费，等待结果。接收后从第 0 行开始，W 按上述公式确定。
2. 预取：已有计划且 `cursor < L`、无在途或待接收结果，且 `cursor >= max(0, W-d)` 时，在本次当前输入有效的前提下请求新预测。未取得有效输入则跳过这次预取机会，后续周期仍可请求；旧尾部已经耗尽时不提交空前缀请求来延长本段。
3. 请求快照：保存此时的 `request_cursor`；前缀使用规范化策略动作的 `actions[cursor:].copy()`，不提前进行未来 IK。向 Policy 传 `delay_steps = min(d, len(prefix))`，满足当前接口。d 只是 guidance 估计，不表示必须执行或等待 d 个墙钟槽。
4. 结果完成后及时取出异常或保存唯一待接收结果，不在旧计划 `cursor < W` 时提前交接；等待期间不得再提交第二个预测。真实模型异常不延迟到 W 才传播。
5. 当 `cursor >= W` 且结果已完成时，先算 `c = cursor - request_cursor`，用新结果的 `result[c:]` 作为未来计划，再清请求、将新计划 cursor 置 0，并重算 W。c 包含结果已完成但尚等名义窗口期间的旧动作提交。
6. 在新计划接纳处完成现有策略动作规范化；前缀和实际消费来自同一份规范化动作。模型结果仍只在进入执行时做完整 shape/finite 验证。
7. 结果未完成则继续旧尾部；旧尾部已空且没有可接续结果时结束本段。若结果已完成但裁剪后为空，也按无可用动作处理，不回退成普通 async。
8. 同一控制机会至多接纳一份已有结果、提交一次新请求、发送一个动作。刚提交的预测不得在同一机会被再次取出，避免零时延 fake 模型造成内部忙循环。新计划预取条件已满足时，可以在同一机会提交其下一请求，再发送一个动作。

每个正常控制机会先服务反馈/操作和采样，再查看此前请求的结果并按 W 决定是否接续，随后判断旧计划是否耗尽，最后按需预取、映射并提交一个动作。bootstrap 的无计划等待单独处理，不将运行中耗尽重新解释为 bootstrap。采样及 Raw 的既有记录职责仍按第 4.3、4.8 节执行。

`cursor` 在 `send_action()` 正常返回后推进；调用失败则结束并保留真实异常，不用计数补偿继续运行。这里表示主线程提交进度，不表示 arm 邮箱每行都执行、XHand 每次写入都获确认，或机器人已经到位。设备层允许的未确认写入仍按其现有语义处理，不能为 RTC 新增回执协议。

提前完成的结果只保存一份，不引入第二个后台任务。请求所对应旧计划在接收之前不会被另一预测替换，因此局部消费位置即可表示对应关系，不需要跨 episode 的请求代际系统。停止后禁止接纳旧结果，段间回收完成后才允许下一段。

示例作为行为验收依据：

| 条件 | 期望 |
|---|---|
| P=16，A=8，d=2，在 cursor=6 请求，cursor=7 时结果已完成 | 继续提交旧动作到 cursor=8，再以 c=2 裁剪新结果；不提前交接 |
| 同上，直到 cursor=11 才取到结果 | c=5，接纳 `result[5:]`；不报错过 handoff |
| 请求后只提交 2 个动作，但主循环跨过多个名义周期 | c 仍为 2；不按墙钟跳过更多 RTC 动作，不补发 |
| P=8，A=8，d=2，正常交接裁掉 2 行 | 新计划 L=6，W=6，预取从 cursor=4 起；配置有效 |
| 裁剪后短计划 L<d | W=L，从 cursor=0 可预取；delay_steps 限于实际 prefix 长度，不传非法前缀 |
| 旧尾部刚耗尽，本次已能取到可用结果 | 先接续，再判断耗尽，不误停 |
| 旧尾部耗尽，本次结果仍未完成 | 结束本段，不等待额外宽限期、不重复末动作 |

删除 `missed_slot`、`handoff_slot_missed`、`committed_slot_observation_unavailable`、`committed_slot_submission_missed` 等旧 RTC 承诺及相应分支/测试/文档。不为保留这些标签引入新的状态映射。

Policy RTC 是 soft inpainting，未要求前缀逐项硬相等：[当前实现](https://github.com/haoyangzhanglab/dexmani_policy/blob/52cbcb6fb7c37c8d0cb29ac63f6d00d85d353d0a/dexmani_policy/agents/action_decoders/sampling/rtc.py#L10-L47)。joint 等价角和 EEF IK 继续在实际消费时计算，不新增“策略前缀等于未来设备命令”的验收。

### 4.7 结束、错误与操作交互

停止请求只记录一次结束意图并置 stop signal；不得等待主循环才能置位急停。episode 运行边界在 `finally` 中承担设备停止责任，停止动作完成前不等待 writer 或模型。准备阶段为确保初始停止所需的操作仍保留，不为“只停一次”删除不同阶段的必要停止。

清理计划/历史、更新结果等普通内存操作按职责集中；原始运行异常与停止异常均须可见，可利用 Python 异常链和必要日志。停止失败不能变成正常 operator 结束，也不能让后续模型/传感器收尾被跳过。不新建全局错误分类矩阵。

正常段停止后完成记录和在途模型任务回收，再回待命。退出、急停或技术失败时，Session 先关闭设备，再尝试其余资源收尾；某个关闭失败仍继续尝试其他已取得资源。失败或未完成的句柄保留实际归属。

保留 B 开始、S/C 保存结束、D 丢弃、Q 保存退出、Esc/Ctrl+C 急停退出、空闲 H/T 的现有交互。忙时不排队准备命令，长按 B 不穿过收尾自动启动，迟到 D 不追溯旧段，Q 不覆盖已接受的丢弃意图。不强推“先停止、再确认保存”的第二步交互。

设备反馈失效、模型/SDK/传感进程真实异常继续走现有故障退出；缺少下一次模型输入、没有可用动作等可解释的段结束使用既有结果结构和简短原因，不为每种索引边界新增枚举。正式开始的段才计入 num_episodes 的现有含义保留。

### 4.8 输出、录制和音频

输出规则固定如下，复用现有 CLI，不新增 save-results 开关：

| CLI / API 请求 | 行为 |
|---|---|
| output=None 且 record_raw=False | 不创建输出目录，不写配置/结果文件；终端显示必要状态和错误 |
| 显式 --output DIR，record_raw=False | DIR 仍表示输出根目录；在其下创建 `session_时间戳`，保存配置和简单结果 |
| --record-raw 且未给 --output | 使用现有默认 rollouts 根目录，再创建简单 session 子目录 |
| --record-raw 且给 --output DIR | 在指定根目录下创建简单 session 子目录并完整录制 |

保留必要的同名目录避让，删除 policy/task/experiment 三级路径拼装以及由此产生的目录组件校验。task 原样作为元信息；Raw 自身对非空标签/控制字符的检查仍属于实际文件边界。

显式输出继续使用现有 `run_config.yaml`、`session_result.json`，不另建 JSONL 格式或结果框架。配置在实际参数就绪后归档；结果在段结束和最终退出时更新即可，不写入逐 tick 主路径。最终保存使用完成收尾后的真实结果，不把已发布 Raw 因后续模型失败而反向改称未发布。

辅助输出通过一个明确的小函数处理普通文件/序列化异常：打印路径、失败原因和 traceback；不终止控制、不阻止下一轮、不单独改变退出码，不谎称已保存。不自动换目录、不重试、不引入输出状态机。KeyboardInterrupt 等中断不在普通保存异常中吞掉。

若仅辅助输出所需的目录创建失败，报告后以无辅助输出继续；若请求完整 Raw 而目录或 writer 无法建立，则这是请求的数据功能失败，按现有录制错误处理，不静默关闭 record_raw。Raw 元数据构造、writer append/close/publish 不得放入辅助结果的 best-effort 捕获范围。

Raw 保持全部既有字段、来源时间、提交尝试含义、触觉保持规则和有界流式写入。不得只记录有动作的行来伪造训练可用数据；等待行的 NaN 动作保持其真实含义。录制质量失败和队列满结束本段，健康 writer 可发布已接纳前缀；I/O/关闭失败保留 staging 与真实错误。Canonical 导出、触觉训练质量和 future-force 监督不在本轮改造范围。

PolicySession 和 Rollout 删除 AudioFeedback 的创建、参数、播放和关闭，使用现有终端日志表达开始、结束、HOME、保存/丢弃结果。不新增通知适配器，也不删除 teleop 等入口仍使用的音频模块。

## 5. 分步实施任务

### T0：核对当前调用链和已有改动

核对入口、Session、Rollout、Worker、观测/动作适配、Robot、recording 的实际调用关系；检查旧接口的全部消费者。确认最近 xArm 修复仍在，记录实际起点 SHA 和工作区状态。只运行与拟改行为相关的必要基线检查，不重复全仓扫描。

### T1：辅助输出与 Policy 音频退出控制职责

主要修改 `dexmani_real/deployment/run.py`、`dexmani_real/deployment/session.py`、`dexmani_real/deployment/rollout.py`、`scripts/run_policy.py`。

实施第 4.8 节，删除结果写入对控制的裁决、旧目录组件限制和 Policy 音频参数。复用现有结果字段和写文件工具，更新 CLI help。先保证普通无输出部署、显式结果输出和完整 Raw 三种用途都明确可用。

### T2：顺序启动与统一停止责任

主要修改 `dexmani_real/deployment/session.py`、`dexmani_real/deployment/rollout.py`；`dexmani_real/deployment/inference.py` 仅作必要接口适配。

实施第 4.2、4.7 节。模型先加载，保留唯一线程；stop 意图与执行停止分开，使用现有运行边界的 try/finally。原有取消、急停和真实错误仍可到达相同设备边界。不得用一个长阻塞 predict 替代工作线程，也不为排除并行启动另建启动状态机。

### T3：观测用途与历史收敛

主要修改 `dexmani_real/deployment/rollout.py`、`dexmani_real/deployment/observation.py`；必要时最小调整 `dexmani_real/runtime/observation.py` 的调用接口。

实施第 4.3 节。优先复用现有 ObservationNeeds 和检查函数，不新建依赖图。完整模型输入仅决定历史与新请求，设备反馈决定映射发送，record_needs 只服务显式 Raw。保持 teleop/replay/jog 的实际用途不受共享函数改动误伤。

### T4：拆清同步与异步主体

在现有模块中形成两段可顺读的执行流程，可使用普通方法/函数共享采样、映射、提交和记录。不要为了两种流程新增继承结构或注册表。

先落实 sync 和普通 async 的第 4.4、4.5 节，清除同步路径对 slot/query/handoff 的依赖。可保留现有小 dataclass；删除无消费者字段与重复计数，不同时保留两份表示同一进度的权威状态。

### T5：替换 RTC 语义

主要修改 `dexmani_real/deployment/schedule.py`、`dexmani_real/deployment/rollout.py`、`dexmani_real/deployment/preflight.py`、`dexmani_real/config/deployment.py`。

一次性落实第 4.6 节的请求、早到等待、迟到接续、裁剪、短窗口、预取和耗尽。删除旧严格槽分支和 A+d 校验，保留普通 async 的时间窗口。纯计算尽量留在现有小函数，硬件发送留在既有边界。

### T6：删除失去用途的防御与旧约定

沿改动调用链检查：重复完整模型输入校验、只服务严格槽的标签/统计、已无消费者的参数、旧音频依赖、结果保存主错仲裁和旧文档。

完整预测进入 Real 时验证 shape/finite；信任未改变的内部切片。最终映射结果、跨进程设备目标、长计算后反馈年龄是不同事实，继续在相应边界检查。共享公开函数还有其他实际入口时，先核对覆盖再删除检查，不为假想调用者添加新兼容分支。

不要重新收紧 xArm/HOME/关节映射范围，不恢复来源身份门禁。保留单调时钟和原来源时间、不撕裂快照、实际设备返回码、Ruckig 和现有动作物理语义。

### T7：回归、文档与实施记录

更新 README 中启动顺序、默认输出、按键、历史、三模式时序、RTC 参数和 Raw 说明；将旧部署任务书中相冲突的要求标为历史或指向本文，不改写历史测试事实。

在本任务书末尾追加实际实施记录：起点与终点版本、完成任务、关键代码位置、实际命令和结果、未验证项。未实施项必须明确说明，不把文档更新当作功能完成。

## 6. 验收与验证

优先扩展现有 `test_control_progress.py`、`test_input_resources.py`，共享录制/设备接口有改动时复用对应回归。用受控时钟、可控制 Future 和 fake 设备验证行为，不引入新测试框架或长时间 sleep。

| 编号 | 场景 | 必须得到的结果 |
|---|---|---|
| A01 | 模型加载成功/失败/取消 | 成功后才连接设备；失败/取消不迟到激活设备；已取得模型资源正确收尾 |
| A02 | 首次有效观测、N>1 | 首帧重复填满历史并可首推理；下一次实际采样仅追加一次；请求窗口不受后续 deque 更新影响 |
| A03 | sync 预测慢、映射略超周期 | 主线程仍服务反馈/操作；不补发、不因跨 tick 失败；动作正常返回后才推进 |
| A04 | 正在消费旧动作时模型相机/触觉暂时过期 | 未开启 Raw 时可继续有效旧动作，不更新历史或发新请求；设备反馈仍独立检查 |
| A05 | 普通 async 迟到结果、完整窗口耗尽 | 保留时间裁剪和旧尾部；最终无可用动作立即结束本段，不反复 bootstrap 或等设备租期代替本段结束 |
| A06 | RTC 早到、迟到、控制周期跳过 | 符合第 4.6 节例子；早到不提前突破 W；迟到以实际提交数裁剪；不要求精确时间槽 |
| A07 | RTC A+d>P、短窗口 L<d | 合法配置可运行；W 和预取按实际窗口更新；delay 不超过 prefix 长度，无越界或零时延忙循环 |
| A08 | 尾部为空，同时结果刚好完成/仍在运行 | 先接纳可用结果；否则结束；不重复最后动作、不重赋时 |
| A09 | 推理进行中 S/D/Q/Esc，随后迟到完成或异常 | 停止信号立即生效，结果不再用于动作；异常被取出；下一轮 reset/predict 不与旧任务重叠 |
| A10 | 运行异常与 robot.stop/其他 close 异常同时存在 | 原因和异常链可见；设备先停/先关；其他资源继续尝试收尾；未完成句柄不伪装已释放 |
| A11 | 默认无输出、含斜线 task 的显式结果输出、辅助写入失败 | 默认不建目录；task 不参与路径拼装；普通保存失败可见但不终止控制/改变退出码 |
| A12 | 完整 Raw 目录/writer/质量/队列异常 | 仍按录制用途处理，不被辅助输出异常捕获吞掉；真实前缀、staging、时间和 NaN 等待行含义保持 |
| A13 | 忙时长按 B、S 后立即 B、D 与 Q、收尾期间迟到 D | 不自动排队启动；已有保存/丢弃和迟到输入语义保持 |
| A14 | 最新 xArm 反馈边界舍入、模式准备、无手 jog 和共享入口 | 不回退 b53eac89 的修复；teleop/replay/HOME 的相关回归保持通过 |

命令在实施环境的既有完整依赖中运行；先验证对应修改，再做关联回归，最终全套只需运行一次，除非新失败或后续改动需要重跑。

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider \
  tests/test_control_progress.py \
  tests/test_input_resources.py \
  tests/test_recording_export.py \
  tests/test_arm_recovery.py \
  tests/test_xhand.py \
  tests/test_home_snapshot.py \
  tests/test_replay.py

PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -p no:cacheprovider tests
python -B -m ruff check --no-cache --select F dexmani_real scripts tests
python -B scripts/run_policy.py --help
python -B scripts/run_policy.py --print-config
git diff --check
```

若执行现有 Mypy 检查，使用禁用增量且缓存位于仓库外的参数；对既有诊断与新增诊断作实际区分，不用新 `type: ignore`、Any 扩散或运行时兜底掩盖错误。依赖不足时列明阻塞检查，不伪造模块或把 AST/compile 通过写成行为回归通过。

效率验证只需复用已有计时或临时测量，确认每控制机会至多一次采样/动作、一个模型任务、没有大数组新增 deepcopy、没有每 tick 文件输出或补发队列；没有实测时不宣称延迟或吞吐提高。不要为验收增加长期遥测框架。

本轮离线验证不得连接设备、执行 HOME/servo/手动作、训练模型或改写实验数据。真机时序、停止表现和策略效果后续由操作者独立验收，离线测试不替代这些结果。

## 7. 完成定义与提交前检查

- 上述 T0–T7 的代码、调用方、测试、CLI 和文档一致，不留下“新实现+旧强制约定”。
- 普通部署默认无需输出目录；显式完整录制仍保持原数据用途。
- sync 可独立顺读；普通 async 保留其时间锚点；RTC 的前缀、窗口和消费对齐可用几个局部变量说明。
- 旧精确槽机制、Policy 音频和辅助结果控制依赖真实删除；没有以新名字恢复。
- 未新增通用框架、兼容模式、隐式重试、无穷旧动作回退或模型/设备第二所有者。
- 错误仍真实可见，停止及资源回收职责完整；真实动作维度、单位、来源时间和最新硬件修复保持正确。
- 工作区无仓库内工具缓存、无无关产物或改动；已有用户改动未被覆盖。
- 最终报告写明实际完成项、验证命令与结果、行为变化和未验证项，不仅报告删了多少行。

## 8. 实施记录

尚未实施。本节由后续执行 Codex 在完成实际代码修改和验证后填写，不预填“通过”。
