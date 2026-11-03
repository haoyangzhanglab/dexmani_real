# DexMani Real：验收修复与 demo 回放减法任务书

> 这是直接执行的实现任务书。目标是完成代码、调用方、测试和文档的修改，不再追加一轮只提出建议的 review。
>
> 整理日期：2026-11-03。审查及撰写时的最新代码：[`0e1a37b40aa1b5dafbe66a604dbd3e65c5424265`](https://github.com/haoyangzhanglab/dexmani_real/commit/0e1a37b40aa1b5dafbe66a604dbd3e65c5424265)。本文汇总该提交之后的两轮审查：整体验收与继续减法、replay 专项审查。
>
> 本文替换上一版任务书，作为下一阶段唯一执行依据。上一阶段任务及实施记录保留在 [Git 历史版本](https://github.com/haoyangzhanglab/dexmani_real/blob/0e1a37b40aa1b5dafbe66a604dbd3e65c5424265/docs/research_simplification_taskbook.md)。旧文中 replay 的严格时间槽、发布状态门禁、审计产物和回放后 HOME 会话要求均撤销；旧勾选不代表本轮已经完成。以下实现任务全部从未完成开始。

## 1. 目标、取舍与范围

项目服务个人真实机器人研究。一个研究者应能直接读懂“取得输入 → 计算目标 → 提交设备 → 必要记录 → 收尾”。优先减少状态数量、跨层约定和失败分支，不以代码行数为指标。

- 保留 teleop、jog、HOME、sync / async / RTC 策略、Raw 录制和 Canonical 导出的实际能力。上一阶段已删除的跨进程运行代际、`prepared_run_id`、停止失败缓存等不得换名复活。
- **replay 只负责在同一机器人、已由实验者确认合适的布置下，按顺序重放 demo 的绝对关节目标。** 它不承担轨迹跟踪评测、原时间轴重建、自动起点对齐、实验审计或多轮会话管理。
- 不兼容旧 Raw、JSON、内部 API、CLI 参数或旧回放产物。直接修改当前生产者和消费者，删除失效参数、测试与文档；不做双模式、别名、迁移器、兼容 reader 或 fallback。
- 允许 replay 变慢、允许同一错误在运行和关闭日志中重复出现、允许用一个粗起点检查代替复杂启动协议。允许移除 replay 的在线几何/碰撞预检；这是功能边界的主动收缩，并不保证改变后的场景仍可执行原 demo。
- 不放宽设备绝对限位、真实反馈年龄、停止/急停、目标失联期限和真实资源回收事实。文件损坏、SDK 错误和关闭失败不能被静默跳过。
- 不新增 Manager、Executor 类族、模式注册器、资源注册表、截止时间框架、后台 finalizer 或自动恢复。优先原地改现有函数；只为独立职责保留小函数或小数据对象。
- 不删除或改写已有实验数据、checkpoint、标定和旧回放目录。本次只修改源码及其文档/测试；不用真实设备执行自动验证。

## 2. 两轮结论与证据

下列源码路径省略 `dexmani_real/` 前缀。符号指向上述基线，实施时按实际调用关系核实，不依赖固定行号。

| 编号 | 类型与发现 | 最终处理 |
| --- | --- | --- |
| R01 | **已复现 Bug**：`deployment/rollout.py:submit/step` 在无计划时重建等待期限，又在得到计划时清期限；预测一直过时或映射一直跨槽，仍可长期零提交。 | T1：无成功提交的同一阶段共用一次绝对预算；收到结果、丢弃预测、重新预测均不得续期。 |
| R02 | **已复现 Bug**：`PolicySession.check_startup/wait_model` 及部分 `SensorWorkers.start(...may_start=...)` 只看 quit/emergency，S/C/D 已取消仍继续启动等待。 | T2：启动前捕获一次本地 `cancel_epoch`，贯穿传感就绪、模型 load/warmup 和最终准入。 |
| R03 | `SessionState.run_ended_reason`、`revoke_motion(expected_cancel_epoch=...)`、teleop `operator_pending` 又把结束原因变成全局协调协议。 | T2：结束意图归主线程的当前操作；删除全局镜像和过时的“快停已发生、事件尚未入队”防御。 |
| R04 | sync 仍借用 `Prediction.slot`、`ActionChunk.anchor_slot`、`handoff_plan`、`prefetch_handoff` 和跨模式 slot 算法。 | T4：sync 用动作数组、游标、一个在途任务和下一次采样/执行时点；时间锚点仅属于 async / RTC。 |
| R05 | `EpisodeRecorder.submitted_frames` 与 writer 的 `frame_count` 重复计数，主要用于关闭后的发布判断。 | T5：线程确认退出且无错误后，只按 `written_frames` 判断非空；不再建立计数一致性协议。 |
| R06 | `technical_status` 从控制层贯穿 recorder、Raw 和 Canonical，但实际失败原因和发布结果已经单独存在。 | T5：删除该派生标签及参数链；保留结束原因、错误、录制截断和发布事实。 |
| R07 | `ArmProcess._fault_error/_close_error`、修改异常 `.args`、`reported_*` 快照，用异常对象身份来避免重复报错。 | T5：保留 owner 的故障信息，直接转换/抛出；允许重复日志，删除身份缓存和已报告台账。 |
| R08 | 录制冻结后，teleop / rollout 仍按 `control_needs ∪ record_needs` 复制只为存档而需要的 RGB-D/触觉数据。 | T5：冻结后的下一次循环只读取 `control_needs`；不为此建立 worker 动态启停状态机。 |
| R09 | `require_published_raw` 扫描请求路径和解析路径的所有祖先，普通 `.tmp_*` 工作目录也被当成未发布 episode。 | T5：导出只检查实际 episode 的请求/解析 basename；replay 完全移除此发布门禁。 |
| R10 | replay 的绝对 `origin + index * period`、`slot_deadline`、missed/expired 检查，把轻微慢速积累成终止。 | T3：逐行顺序提交，按本轮耗时只等剩余周期，慢了就变慢。 |
| R11 | 已记录绝对目标又经过 `ActionMapper.map_joint` 的关节重选、裁剪、约束和碰撞模型，增加依赖并可能改变原目标。 | T3：直接构造 `RobotCommand`；文件边界检查当前配置范围，公共设备边界继续守限位和停止条件。 |
| R12 | replay 依赖通用 `RawReader` 的 timing、精确 dtype、历史实测起点、task label 和时钟等数据契约。 | T3：只读取动作、名义频率及必要单位/关节顺序；与训练导出的完整契约分开。 |
| R13 | `prepare_hand` 包含历史实测起点 ramp、逐帧 arm 漂移检查、三帧到位证明及叠加超时。 | T3：删整个 replay 专用协议；只比较当前反馈与第一行目标，差异过大直接拒绝。 |
| R14 | `ReplayBuffer`、`ReplayStatus/ReplayResult`、候选/反馈/提交时间数组、NPZ 与 JSON 构成完整审计管线。 | T3：删去；控制台报告提交数和耗时，错误由唯一边界传播。 |
| R15 | 默认固定输出目录要求不存在或为空，同一 demo 第二次运行即被旧产物挡住。 | T3：删除 replay 输出目录与 `--output`，同一命令可直接重复运行。 |
| R16 | replay 完成后进入 `wait_for_home`，引入 H/Q 接纳、HOME planner、home_results 和另一轮取消/错误处理。 | T3：正常结束自动关闭退出；HOME 继续使用已有其他入口，replay 不内嵌 HOME 会话。 |

### 2.1 已有反例

这些是当前生产方法配合假时钟、假设备的离线复现，不是实机结果：

1. async 16 Hz、预测长度 8、预测耗时 600 ms、无旧目标、无运行时长上限：模拟 12.5 s 已提交 20 次预测，**设备提交 0 次，未退出**；等待期限不断向后移动。已有旧目标时，其 2 s 租期可限制等待，但不能解决首发窗口。
2. async 16 Hz、预测耗时 100 ms、映射耗时 63 ms：模拟 17.725 s、71 次预测、275 次映射，**设备提交仍为 0，等待期限已变成 None**。只修“重新预测不续期”仍不完整，还须修“拿到计划就清期限”。
3. 模型模拟 3 s 完成，在 10 ms 时输入 S/C/D：取消计数改变、stop 已置位，`wait_model` 仍轮询 602 次至约 3.005 s 正常返回；Q/Esc 则约 10 ms 结束。部分传感就绪回调有同样的漏检。
4. replay 16 Hz、300 行，每次发送稳定耗时 63 ms：约 7.875 s 后仅发送 125 行，下一行报 `replay slot 125 missed before observation`。这不是文件或设备故障，而是原时间网格的人为门禁。

### 2.2 已成立的成果不重做

上一提交已有效减少运行代际；owner 在停止态不装入新目标、父进程确认停止并持锁清邮箱、真实 Pipe 回复归属等有实际作用。保留该简单恢复顺序，不重开分布式授权设计。

录制队列满后冻结有效前缀、finish 唤醒 writer 的行为已经用真实线程验证：队列容量 16，加上正在写的 1 行，保留 17 行后正常关闭发布。本轮删计数时必须保留这一行为。

`HomePlanner.table_plane_abcd` 有真实消费者：Policy 将实际使用的桌面标定归档至运行配置。保留该成员；不要依据早期“看起来未使用”的判断删除。

仓库旧实施记录写有 79 项测试通过；审查环境缺少 pytest、h5py、zarr、av 等完整依赖，未独立重跑该套件。本文不把历史测试数量当成本轮通过证明。

## 3. 最终保留的职责和时间语义

| 职责 / 事实 | 简单表达 |
| --- | --- |
| 当前操作、保存/丢弃和正常结束原因 | 父进程主线程的当前操作局部状态；沿用现有结果对象即可。 |
| 长操作是否被后续用户操作取消 | 一个父进程本地 `cancel_epoch`；不进入命令、共享目标、KeyEvent 或 Pipe 协议。 |
| 实际设备是否可继续执行 | 公共 stop / emergency / closing、owner fault、反馈新鲜度、目标租期。 |
| arm 目标与生命周期工作 | 一个最新值邮箱、一个 SDK owner、至多一个尚未消费的生命周期回复。 |
| 后台资源是否结束 | 真正的 Future / thread / process / Pipe 请求；超时本身不等于完成。 |
| 录制是否还接收新行 | recorder 的接纳状态；与 writer 是否仍存活是两个不同事实。 |

| 时间约束 | 本轮最终语义 |
| --- | --- |
| 来源新鲜度 | 使用原采样/接收时间；复制缓存、重新映射不能续期。 |
| 目标租期 | 公共设备层处理生产者停更。replay 沿用现有 2 s 租期；不改成名义周期，也不靠无进展重复发送保活。 |
| teleop / jog / sync / replay 的周期 | 频率目标；按实际一轮开始计时，慢计算降低频率，不补发欠账。 |
| async 时间窗口 | 保留观测锚点、过期前缀裁剪及有效旧尾部；过时结果不能变成全新的动作。 |
| RTC 承诺 | 保留固定前缀、约定交接和错过承诺时结束。replay 放宽不改变 RTC。 |
| 预测/首发等待 | 连续没有成功设备提交的阶段受一次绝对预算约束；计算活动不算进展。 |
| 实验时长 | 与等待预算分别存在，选更早者；相等仍按正常 duration 结束，真实故障和用户终止另行保留。 |
| 关闭等待 | 同一未完成任务不因进入外层 close 获得新的完整等待预算。 |

仍须保证：停止后的旧发布调用已在主线程返回，owner 停止态不消费邮箱，持锁清空后才能恢复；故障及未消费回复阻止下一操作。取消与恢复用既有短本地锁排序。不得持键盘/状态/邮箱锁跨 SDK、join、大数组读取或模型计算。arm 与 hand 的顺序提交不被描述成原子事务，Python 也无法撤回已进入的 native 调用。

## 4. 实施任务

### T1. 修复 async 零进展与等待期限失效

主要文件：`deployment/rollout.py`，相关回归见 V01–V03。

- [ ] 保留启动时建立的首发绝对期限，包含已有 history 积累预算；在第一条 `robot.send_action` 正常返回前，不因 submit、新计划、预测完成或过时结果重新预测而更新/清除它。
- [ ] 删除 `step()` 接受 `new_plan` 就清 `wait_deadline_ns` 的行为。候选通过映射、计划变得非空、Future 完成都不算设备提交成功。
- [ ] `submit()` 不得覆盖尚未耗尽的无提交预算。需要建立新的预测等待期限时，只能在新的实际进展边界后建立一次，并受已有目标 `expires_ns` 限制；同一无提交阶段只可沿用或缩短。
- [ ] 成功发送后记录新目标期限，并结束旧等待阶段。将已有 `target_expires_ns` 也纳入主循环的停止预算选择，使“有计划却反复丢弃候选”也有界；不要只依赖下一次无计划 submit 才检查旧目标期限。
- [ ] 预算检查覆盖模型结果消费、映射之后和实际发送之前；不得让超时后的重提预测绕过预算。复用现有期限字段和 `budget_limit/expired_budget`，不加 watchdog 线程、新计数器或故障状态机。
- [ ] 保持 async 裁掉过时动作、RTC 严格交接、Future 原错误传播、duration 同时到期优先级和已有目标租期。首发失败应明确结束并收尾，不在无时长上限时永久“忙而不发”。

实现判断标准不是“预测很活跃”，而是有成功目标提交，或在既定预算内带真实原因退出。

### T2. 启动取消贯穿到底，结束意图回到主线程

主要文件：`deployment/session.py`、`teleop/run.py`、`teleop/loop.py`、`runtime/{state,keyboard,lifecycle}.py` 及受影响调用方。

**启动取消：**

- [ ] 在启动流程开始、键盘开始接收取消之前捕获一次本地取消值；贯穿资源准备、connect、camera/pointcloud/VR 就绪、手部初始化、模型 load/warmup 等真正的长等待。不要在 S/C/D 之后重新捕获新值，把被取消的启动当成新的启动。
- [ ] `check_startup`、`wait_model`、各 `may_start` 使用同一取消上下文；取消值改变或 quit/emergency 即离开等待。`requests_allow_motion()` 仅检查 quit/emergency，不能单独充当完整启动取消判断。
- [ ] 每次等待完成、进入下一阶段前再检查取消，避免取消与 ready/Future 完成同时出现时继续启动。设备/worker 的真实故障照常报告，不被普通取消覆盖。
- [ ] 不用“stop 为真就一律取消启动”替代上述修复：内部 prepare/HOME 本来会置 stop，这与用户取消不同。内部停止不递增用户取消计数。
- [ ] 取消停止的是编排和后续操作，不声称能强杀已运行的模型线程/native 调用。保留真实 Future/进程供现有有界收尾；不继续 warmup、不创建下一段 recorder、不取得运动授权。

**结束意图：**

- [ ] 删除 `SessionState.run_ended_reason` 及开始操作时清它的逻辑；删除 `revoke_motion` 上仅用于回取旧原因的 `reason`、`expected_cancel_epoch` 参数及返回值。停止动作与结果归因分开。
- [ ] 这里删除的是 `revoke_motion` 的原因协议，**不是** prepare/resume/HOME 等长操作需要的 `cancel_epoch` 比较。共享 stop/emergency、quit 请求和必要的小交互状态继续保留。
- [ ] 键盘回调只在既有短临界区设置取消/退出/急停事实并入队；主线程解释已接纳终止事件，把结束原因写入现有当前操作结果。`RunEndReason` 若仍简化结果表达可保留，不能再成为全局镜像。
- [ ] 删除 teleop `operator_pending` 和“回调已快停但还没入队”的等待协议。回调与入队已在同一 Condition 内，主线程取得该锁后可取得一致的输入快照。
- [ ] 取已接纳终止事件及用于判别的 stop 快照时使用同一短输入临界区，锁外处理事件和 I/O。不能先无锁 poll、再读取一个后来改变的 stop，就断言是无操作者事件的执行器停止。临界区之后的新事件留给下一次处理，快停信号已即时生效。
- [ ] 保持现有顺序：冻结保存决定前消费已接纳终止输入；D→Q 仍丢弃，冻结后的 D 不追溯修改已决定的保存；quit/emergency 独立保留。不要为了移除一个全局字段增设另一套 stop_reason / pending_reason / generation。
- [ ] busy 接纳、held keys、停止与退出响应、jog 的 S/D 方向键语义保持正确；锁序继续 Condition → 本地状态锁，不反向嵌套。

### T3. 把 replay 收敛为一次顺序 demo 播放

主要文件：`scripts/replay_episode.py`、`replay/{run,joint_replay,trajectory}.py`、必要的公共 preflight 调用方。

#### T3.1 最小输入与文件边界

只支持当前项目的绝对关节目标，不新增通用轨迹格式。CLI 保留 episode 目录和 `--config`；读取该目录下的 `data.h5`。

| 读取内容 | 检查 |
| --- | --- |
| `action_arm_joint_target` | 非空实数数组，形状 `(N, 7)`，数值有限。 |
| `action_hand_joint_target` | 形状 `(N, 12)`，与 arm 等长，数值有限；本次 replay 仍要求启用 XHand。 |
| `meta.control_hz` | 有限正数，符合公共设备的发送频率上限。 |
| `meta.joint_units`、arm/hand joint names | rad 和当前设备顺序；一次检查，不猜单位、不重排、不迁移。 |
| 当前 arm / hand 配置范围 | 一次向量化检查整段目标，不合范围在连接设备前失败；不裁剪、不 wrap 到另一分支。 |

- [ ] 用短小的 h5py reader 一次读出上述数组和属性，关闭文件后再连接设备。允许 float32 / float64 等可转成实数目标的正常数值 dtype，不要求与 Raw writer 完全一致；复数、字符串、非有限值不被静默转换成合法动作。
- [ ] 不依赖 `RawReader` 的完整模态/时间契约；不要求 `timing/rows`、历史首行 `arm_qpos/hand_qpos`、task label、历史时钟声明、RGB-D、触觉、完成状态或回放审计字段。动作字段本身定义绝对关节目标，不再证明历史 SDK 是否执行成功。
- [ ] replay 移除 `require_published_raw` 和路径前缀门禁。显式路径下的文件能正常读取且目标有效即可，包括用户明确指定的 `.tmp_*` 目录；不自动发现临时目录，不读写并修复坏文件，不发布/重命名数据。正在写入、锁冲突和损坏文件按实际 I/O 错误失败，不加入重试或实时追读。
- [ ] NaN 行、空序列、错列数、缺必要属性仍报错；不自动筛行、插值、补零或回退到实测关节位置。允许不完整的存档模态不等于允许不完整的目标序列。
- [ ] 注意当前 `Robot.validate_command` 的 hand 检查使用额定硬件范围，未必覆盖更窄的 `HandConfig.qpos_min_rad/qpos_max_rad`。因此保留此处一次整段配置范围检查；不能删掉 mapper 后误以为公共 hand 校验覆盖了全部实验限位。无需为此扩展全仓命令接口。

#### T3.2 顺序、节拍与起点

- [ ] 一次连接和公共 streaming prepare 后读取当前 arm/hand 新鲜反馈，只与**第一行目标**比较。沿用一个粗阈值：任一关节绝对差大于 10° 则报错并关闭，提示实验者先就位；不自动规划到第一帧。
- [ ] 删除 `prepare_hand`、历史实测起点、手部 ramp、三帧到位、arm 漂移重复检查及 `--hand-start-duration`。开始播放时的第一行目标就是第一条 demo 目标。
- [ ] 直接 `RobotCommand(arm_targets[i], hand_targets[i])` → `robot.send_action(..., expires_ns=now + COMMAND_LEASE_NS)`。不调用 replay 的 `ActionMapper`、IK、`MotionConstraints`、`project_*` 或 `wrap_nearest_equivalent`。公共设备校验和 arm Ruckig 执行仍在。
- [ ] **删除 replay 在线几何/碰撞预检是本方案的明确取舍。** replay 不需要 Pinocchio、hpp-fcl、碰撞 mesh、桌面标定或 HOME planner 才能加载/播放关节目标；不把任意新场景、任意人工轨迹宣称为已验证可执行。teleop / policy / HOME 的几何检查保持。
- [ ] 每一行开始时记 `iteration_start`，服务反馈和用户终止，提交该行；只等待 `max(0, period - elapsed)`。下一行从新的实际时刻开始；无绝对原点、slot deadline、累计 lateness、missed slot、补发或丢行策略。
- [ ] 等待沿用短周期服务：继续检查设备/键盘健康并采集所需手反馈，不用一整段不可中断 sleep。保留一个普通小等待函数即可；已有公共 owner 负责 arm 反馈，replay 不再逐行构造全模态 Observation。
- [ ] 成功提交后才增加显示用的 `sent_count`。公共 arm 邮箱仍是 latest-value；有序提交不等于每行都获得 SDK 回执或物理到位，不为 replay 改成 FIFO/ack 协议。
- [ ] 最后一行正常提交后，以**固定 0.5 s 可中断留驻替代最后一行的普通周期等待**，然后关闭。不附加另一轮完整周期，不续租，不做收敛证明或三帧到位。目标租期/反馈故障照常打断；这段留驻不保证到位。无需新增尾帧参数。
- [ ] 输入很慢或 native 调用阻塞以致公共目标租期到期，仍按公共执行规则停止；不为任意低频录制新增保活线程、自动重发或无限租期。

节拍行为示意（省略公共设备生命周期和异常处理，非新增调度框架）：

```python
for i in range(len(arm_targets)):
    started = monotonic()
    service_feedback_and_check_cancel()
    robot.send_action(RobotCommand(arm_targets[i], hand_targets[i]),
                      expires_ns=monotonic_ns() + COMMAND_LEASE_NS)
    sent_count += 1
    wait_until = monotonic() + 0.5 if i == len(arm_targets) - 1 else started + period
    wait_with_feedback_and_cancel(wait_until)
```

#### T3.3 单次流程、退出和错误

- [ ] 流程只剩：读取并校验 → 创建键盘/设备 → 连接与 prepare → 粗起点检查 → 顺序发送 → 短留驻 → 唯一 `finally` 公共收尾。
- [ ] S/Q 均停止本次播放并退出，不进入“已停止但仍在等待”的会话；Esc/Ctrl+C 使用公共 emergency 路径并非零退出。取消在连接、准备、行间等待和尾帧留驻期间都有效。
- [ ] 正常播完、操作者 S/Q 且关闭成功，CLI 返回 0；坏输入、起点拒绝、无操作者输入的设备停止/租期失效、SDK/反馈/关闭失败，CLI 非零。不能把所有 `OperationCancelled` 都吞成用户正常停止；设备故障与用户取消同时出现仍报告故障。
- [ ] 删除 `ReplayStatus`、`ReplayResult`、`ReplayBuffer`、`save_replay_data`、提交候选/实测/时间戳审计数组、`session_result.json`、`replay_data.npz` 和结果去重代码。正常返回、明确的用户停止、真实异常已足够表达流程；不要换成另一套结果 envelope。
- [ ] 删除 `output_dir`、`DEFAULT_OUTPUT_DIR`、`--output`、目录必须为空的校验和所有强制输出。控制台打印总行数、已提交行数、耗时和必要停止原因即可；同一命令可重复运行。旧产物留在原地。
- [ ] 删除 `wait_for_home`、H 交互、`home_results`、回放中的 HOME planner 和相应取消分支。正常播放结束自动 close 并退出。
- [ ] 复用 `close_runtime` 的停止/关闭与真实句柄处理；不叠加内外两层 replay 结果分类和重复 stop-finally。全部已创建资源在连接失败等部分启动路径也能被关闭。
- [ ] 若运行主错与关闭错误同时存在，保留主错，记录关闭错误 traceback；只有关闭失败时也必须非零。错误不被最终普通 return 覆盖，不为去重存异常对象身份或添加错误注册器。

#### T3.4 模块和依赖

- [ ] 将实际 runner 放入现有 `replay/run.py`，CLI 保持薄。reader 可与 runner 同文件，或仅在可读性明显更好时保留 `trajectory.py`；删除已无职责的 `joint_replay.py` 及无用导出。不为“单文件”把公共设备逻辑复制进去。
- [ ] replay 的 preflight 只验证实际使用的 arm/hand 配置、反馈服务与发送频率；移除该入口对 environment/workspace/HOME 的附加验证。若公共 `validate_robot_config` 混合设备和场景校验，可把场景检查移到实际使用它的现有入口，同时更新所有调用方；不增加多种可选校验模式。
- [ ] 核实传递导入，确保 replay 的帮助、reader 和假设备播放路径不加载碰撞/IK/相机/视频依赖；不要只删构造调用却留下顶层重依赖。其他功能仍用的依赖和文件不能全仓删除。

### T4. sync 脱离 async / RTC 时间窗口算法

主要文件：`deployment/rollout.py`、`deployment/schedule.py` 和配置校验/调用方。

- [ ] sync 的真实状态只需当前动作数组、已执行游标、一个在途模型任务、下一次采样/执行时点，以及 T1 的公共等待/时长限制。直接表达“积累观测 → 预测 → 顺序执行 A 行 → 下一次预测”。
- [ ] sync 不再创建伪造的观测 slot/handoff slot，不调用 `handoff_plan`、`prefetch_handoff`、`plan_rejection` 来表达顺序游标；`schedule.py` 只服务实际需要时间窗口的 async / RTC，清理各函数的 sync 特判。
- [ ] 可以在现有 Rollout 中保留短小 mode 分支/方法；不为三种模式创建继承树、可插拔调度器或三个拥有独立生命周期的 Executor。观测、映射、公共发送、录制及收尾仍复用。
- [ ] 模型等待期间不重复把同一行塞入 history；保持实际采样间隔的 history 断档规则、第一预测 reset RNG、预热恢复 RNG、A/P 配置语义及公共来源新鲜度检查。
- [ ] 映射耗时 63 ms 而周期 62.5 ms 时，sync 仍发送有效动作并推进游标；不因名义 tick 过期丢弃，也不再额外睡完整周期或突发补发。发送失败不前进。
- [ ] async 仍可以使用旧预测有效尾部；RTC 的前缀和交接检查不移除。本任务不以删模式来换取简单。

### T5. 数据与错误链的局部减法

主要文件：`data/{recorder,raw,export}.py`、`teleop/loop.py`、`deployment/rollout.py`、`robot/xarm7/process.py`。

**录制和采样：**

- [ ] 删除 `submitted_frames` 及递增/断言；writer 已结束且无错误后，用 `written_frames > 0` 判断可发布非空数据。writer 未完成或 I/O 失败时不得拿当前计数提前发布。
- [ ] 保留 writer 的真实 frame_count、queue/freeze/finish 唤醒、首个错误、真实线程句柄和有界关闭。`is_recording=False` 不等于 writer 已退出。
- [ ] 控制循环在 recorder 确实接收时读 control/record 并集；无录制或已冻结时读 control 所需字段。冻结后的下一轮不再复制仅供存档的 RGB-D/触觉，不停止控制必需来源，不清空其最新有效缓存。
- [ ] 不为减少这次复制而添加动态 worker 重启、重新录制或按字段管理资源的框架。冻结保持不可逆；同一 episode 不因存档来源恢复而续录。

**数据事实：**

- [ ] 删除 `technical_status` 的 Raw 初值/finish 参数、`_FinishRequest` 字段、控制层传参、Canonical quality 复制、文档和测试。不要改名为另一个等价成功标签。
- [ ] 保留 `termination_reason`、错误详情、`recording_truncated_reason`、实际行数以及 recorder 的 published/failed/pending 结果。运行尾部失败不反向否定已正常发布的有效前缀。
- [ ] RawReader 继续支持诊断指定 staging；Canonical 导出仍检查发布边界与完整数据。将 `require_published_raw` 收窄到请求 episode basename 与解析后 episode basename：真正 `.tmp_episode` 及其别名仍拒绝，普通 `.tmp_workspace/episode_good` 允许。不遍历祖先，不维护路径 provenance 或符号链接安全框架。
- [ ] replay 不调用上述导出发布门禁。该分离是不同用途的明确约定，不是新增 strict/relaxed 模式。
- [ ] Canonical 的固定模态、单位/形状、RGB 数量、来源时间与触觉年龄、相邻 control 间隔 ±25% 和观测不跨下次 control 等现有约束保持；`dt` 仍为名义周期。只删除已列出的冗余状态，不借 replay 放宽删去训练数据契约。
- [ ] 保留“失败 Raw staging 不自动删除；可再生的导出 staging 失败只清本次目录”的行为。

**错误：**

- [ ] `ArmProcess` 直接读取 owner fault/close_error 信息并抛出正常异常；移除 `_fault_error`、`_close_error`、`.args` 改写和 `reported_fault/reported_close_error` 身份去重快照。
- [ ] owner 真正异步产生的故障通道、stop/disconnect 失败信息、join 后晚到错误和异常退出检查保留。这里删的是父进程重复包装和去重状态，不是故障本身。
- [ ] 允许运行和收尾重复报告同一 owner 错误；不要求一个异常在整个系统只出现一次。已退出但失败与仍活着的 pending 继续按真实句柄区分，不用错误是否已打印推断资源是否结束。
- [ ] 清理受影响代码里只为此维持的异常身份比较和帮助函数；保留必要的主错变量、异步 writer error 和“主错不能被次级收尾错误覆盖”的普通 try/finally。不要把所有错误处理一概删除。

### T6. 文档、测试和入口一次改完整

- [ ] 更新 README 的 replay 命令、键位、输出目录、数据条件、时间语义、几何依赖和 HOME 使用说明；删除所有“严格 replay 槽位”“回放结束按 H/Q”“必写 NPZ/JSON”的现行约定。
- [ ] 更新 CLI help、受影响模块 docstring、当前数据读写与配置调用方；无需给已删参数加 deprecation 周期。历史版本无需改写。
- [ ] 改写因需求主动改变而失效的测试：特别是 replay staging 拒绝、精确 dtype、旧审计结果、旧 `technical_status` 和 `submitted_frames` 断言。保留这些测试原本覆盖的真实错误/资源边界，不以删除整组回归规避问题。
- [ ] 按第 5 节完成行为验证，再审查实际 diff。不要用“搜索不到旧类名”“AST 可解析”代替运行行为验证，也不要为了填满清单新增与实现一一对应的测试。

## 5. 验收场景

优先扩展已有五个业务测试文件；replay 如有必要可增加一个集中的测试文件。使用 fake driver、可控时钟和真实小型线程/Future，不连接硬件。表中的场景可参数化/合并，不要求每行变成独立测试或新测试框架。

| 编号 | 场景 | 必须观察到的行为 |
| --- | --- | --- |
| V01 | async 16 Hz、8 行预测、600 ms 返回、无旧目标、无时长上限 | 在首次绝对等待预算内退出；重新预测不能不断延后期限。 |
| V02 | async 100 ms 预测、63 ms 映射，候选持续跨槽 | 即使计划反复被接受也不清预算；最终有发送或有界失败，不能长期零提交。 |
| V03 | 有旧目标但一直无新提交；运行时长/等待期限同时到期；Future 抛错 | 不越过旧目标期限；相等按 duration；真实模型错误保留。 |
| V04 | 在 camera/pointcloud ready、模型 load、warmup 等等待中输入 S/C/D/Q/Esc；ready 与取消同时发生 | 及时退出对应等待并收尾，不执行下一阶段；内部 HOME 的 stop 不误取消；未完成 Future/进程仍有人持有。 |
| V05 | 快停回调持 Condition 暂停；poll/终止快照交错；D→Q、freeze 前后 D、busy 长按启动键 | 终止意图与已接纳输入一致，无旧事件启动下一段；保存/丢弃时序与锁序正确。 |
| V06 | sync 16 Hz、63 ms 映射及慢预测 | 成功发送推进游标，顺序和 A 行窗口正确；不补发、不额外睡满周期，等待受预算约束。 |
| V07 | async 历史间隔 64.1 ms 与真正超过 1.5dt；RTC 错过交接 | 前者不误清历史，真正断档重建；RTC 继续明确结束。 |
| V08 | 最小 HDF5 仅含 T3 必要字段，float32 目标，没有 timing/RGB/历史实测起点；显式 `.tmp_*` 路径 | replay 可加载；通用 RawReader/Canonical 的完整模态要求不被迫放宽。 |
| V09 | 空/错形状/不等长/NaN/复数/单位或顺序错误/配置越界的动作文件 | 连接设备前明确失败，无裁剪、补值、筛行或自动迁移。包含“hand 在额定范围内但超出当前配置范围”的用例。 |
| V10 | replay 300 行、16 Hz、每次发送 63 ms；另测正常快速发送 | 全部按序提交，不因累计晚点退出或跳行；快速情况只等剩余周期，尾帧 0.5 s 留驻替代普通等待。 |
| V11 | replay 第一行姿态偏差；连接/prepare/播放/留驻中 S/Q/Esc；重复同一 CLI | 大偏差拒绝且关闭；S/Q 正常停止退出，Esc 非零；第二次不被输出目录拦截，不创建审计文件或等待 H。 |
| V12 | replay 发送异常、无操作者输入的租期停止、反馈失效、关闭失败；主错和关闭错同时发生 | 非零且错误可见；无吞错继续下一行、无正常 return 覆盖异常；所有已创建资源都尝试收尾。 |
| V13 | 不提供碰撞/IK/视频依赖，用最小文件和 fake Robot 走 replay；真实手反馈服务计数 | 不加载 Pinocchio/hpp-fcl/mesh/相机解码；等待期间继续服务反馈和取消。公共硬件依赖不伪称已经验证。 |
| V14 | 实际 writer 线程被阻塞、队列满冻结；另测零行、写入失败、join 超时 | 有效 17 行前缀仍可关闭发布；非空判断仅用关闭后的 writer 计数；零行不发布，失败留 staging，超时保留真实句柄。 |
| V15 | recorder 冻结前后观察读取字段；控制仍需相机/点云的另一用例 | 只停止复制存档专用字段；控制所需字段仍读取，运动与来源 TTL 正常；不会恢复录制。 |
| V16 | 导出位于 `.tmp_workspace/episode_good`；真正 `.tmp_episode` 和指向它的别名；普通临时祖先下指向已发布 episode 的别名 | 普通祖先不误拒绝；实际 staging 仍拒绝；与 replay 的显式文件约定区分。 |
| V17 | 运行 owner fault、join 后晚到 close_error、已退出但失败、进程仍活着；Future 返回 None/抛 TimeoutError/未完成 | 错误不因去缓存而丢失；重复报告可接受；完成失败和 pending 正确区分，同一任务不续关闭预算。 |
| V18 | 原有停止/清邮箱/prepare 恢复、共享快照、HOME、Canonical 有界时序与导出失败清理回归 | 上阶段成立的不变量不倒退；不删除实际使用的桌面标定归档字段。 |

在项目依赖可用的环境执行相关测试及业务回归；例如 `python -B -m pytest -p no:cacheprovider -q tests`，以及可用时 `python -B -m ruff check --no-cache --select F dexmani_real scripts tests`、`git diff --check`。遵守 AGENTS.md，不在仓库生成 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`。缺依赖时写明未执行的具体检查，不能以导入/静态解析冒充通过。

验证应覆盖真实的 h5py 最小文件读取和实际公共 send/close 边界上的假设备行为；仅对整套 runner 打桩返回成功没有证明力。真机起点阈值、尾帧留驻体验、运动跟随和停止延迟由实验者后续验收，不在本任务自动执行，也不预写“真机通过”。

## 6. 顺序与完成标准

推荐顺序：T1 修复预算 → T2 修复启动取消并收拢结束意图 → T3 完整重写 replay → T4 简化 sync → T5 局部数据/错误减法 → T6 同步文档并完成回归。每个机制连同实际调用方一次改完整；阶段之间可以有清晰提交，不并存新旧实现。

- [ ] R01–R16 均有明确落地结果；没有把任务改成再次提出方案或仅写 TODO。
- [ ] 核心结构变少：replay 没有独立会话/审计/严格网格，sync 没有伪时间窗口，全局终止原因和错误身份台账消失，数据事实不重复存储。
- [ ] 真实边界仍可解释：文件一次检查、设备副作用边界检查、来源新鲜度、目标租期和实际 pending 各自解决不同问题。
- [ ] 文档、CLI、当前数据生产/消费和测试一致；没有兼容层、空壳旧类、无用参数、备用严格模式或换名机制。
- [ ] 检查现有工作区并保留他人修改；不改用户实验数据，不做无关目录搬迁，不添加没有实际用途的依赖。
- [ ] 完成后在本文末尾补实际实施记录：修改项、删除项、行为测试及结果、未执行项、与本文不同的必要调整及源码依据。只勾选实际完成项，不复制历史通过数量。

若实施时 HEAD 已前进，先比较本基线与当前代码，已修复的问题不重复实现。合理的局部实现选择可以自行决定；只有真实新调用关系或需求冲突才调整方案并解释，不以假想未来需求恢复复杂度。

## 7. 参考项目与采用范围

| 项目与固定源码 | 采用 | 不照搬 |
| --- | --- | --- |
| [LeFranX 顺序回放](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_robot_replay.py#L148) | 直接 for-loop 发送，按本轮耗时等待剩余周期，慢了就降低速度。 | 其发送异常后继续下一行的处理不适合本项目；不复制旧格式兼容。 |
| [T-Rex 硬件回放](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/replay_h5.py#L338) | 明确的目标读取、顺序迭代、限频与停止退出。 | dexmani 已保存绝对关节目标，不需要复制双臂 IK、额外 Manager 或动作缓冲体系。 |
| [ManiUniCon 回放工具](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/tools/replay_record_data.py#L84) | 单用途工具按用途读取数据的组织方式。 | 该工具主要服务 Meshcat 可视化，不作为真机关节回放的停止/执行依据；传感 replay 同样不是设备执行器。 |

参考代码提供结构借鉴，不替代 dexmani_real 实际入口、设备协议和上述行为测试。
