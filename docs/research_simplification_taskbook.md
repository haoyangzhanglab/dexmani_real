# DexMani Real：验收修复与 demo 回放减法任务书

> 本文记录已经实施的减法重构任务与审查依据。第 9 节是 2026-11-04 本轮实施记录，覆盖此前关于 async 跨 tick 否决、history 间隔清空、replay 私有反馈监控及文件边界精确比较的约定。第 8 节保留上一阶段的实现和验证事实，测试数量只对应当时状态。
>
> 原始审查基线为 [`0e1a37b40aa1b5dafbe66a604dbd3e65c5424265`](https://github.com/haoyangzhanglab/dexmani_real/commit/0e1a37b40aa1b5dafbe66a604dbd3e65c5424265)，本轮实施基线为 `2b1e846bbf59c786c0fbe23b053732880909eade`。更早的任务及实施记录见 [Git 历史版本](https://github.com/haoyangzhanglab/dexmani_real/blob/0e1a37b40aa1b5dafbe66a604dbd3e65c5424265/docs/research_simplification_taskbook.md)。下列任务和验收场景已按当前要求对齐。

## 1. 目标、取舍与范围

项目服务个人真实机器人研究。一个研究者应能直接读懂“取得输入 → 计算目标 → 提交设备 → 必要记录 → 收尾”。优先减少状态数量、跨层约定和失败分支，不以代码行数为指标。

- 保留 teleop、jog、HOME、sync / async / RTC 策略、Raw 录制和 Canonical 导出的实际能力。上一阶段已删除的跨进程运行代际、`prepared_run_id`、停止失败缓存等不得换名复活。
- **replay 只负责在同一机器人、已由实验者确认合适的布置下，按顺序重放 demo 的绝对关节目标。** 它不承担轨迹跟踪评测、原时间轴重建、自动起点对齐、实验审计或多轮会话管理。
- 不兼容旧 Raw、JSON、内部 API、CLI 参数或旧回放产物。直接修改当前生产者和消费者，删除失效参数、测试与文档；不做双模式、别名、迁移器、兼容 reader 或 fallback。
- 允许 replay 变慢、允许同一错误在运行和关闭日志中重复出现、允许用一个粗起点检查代替复杂启动协议。允许移除 replay 的在线几何/碰撞预检；这是功能边界的主动收缩，并不保证改变后的场景仍可执行原 demo。
- 保留公共设备发送边界的绝对限位、反馈年龄、停止/急停、目标租期和真实资源回收事实。文件损坏、SDK 错误和关闭失败不能被静默跳过。
- 不新增 Manager、Executor 类族、模式注册器、资源注册表、截止时间框架、后台 finalizer 或自动恢复。优先原地改现有函数；只为独立职责保留小函数或小数据对象。
- 不删除或改写已有实验数据、checkpoint、标定和旧回放目录。本次只修改源码及其文档/测试；不用真实设备执行自动验证。

## 2. 历史审查结论与证据

下列源码路径省略 `dexmani_real/` 前缀。符号指向原始审查基线 `0e1a37b`，实施时按实际调用关系核实，不依赖固定行号。

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
| async 时间窗口 | 保留观测锚点与完整剩余序列；跨名义 tick 的候选仍可在真实窗口截止前提交。裁掉过期前缀，过时结果不重新锚定。 |
| RTC 承诺 | 固定前缀、约定交接；错过承诺槽或该槽缺少必需观测即结束。 |
| history | 保留最近 N 次已接纳采样，允许不等间隔，首次仍需完整窗口；sync 等待预测时不重复入窗。 |
| 预测/首发等待 | 连续没有成功设备提交的阶段受一次绝对预算约束；计算活动不算进展。 |
| 实验时长 | 与等待预算分别存在，选更早者；相等仍按正常 duration 结束，真实故障和用户终止另行保留。 |
| 关闭等待 | 同一未完成任务不因进入外层 close 获得新的完整等待预算。 |

仍须保证：停止后的旧发布调用已在主线程返回，owner 停止态不消费邮箱，持锁清空后才能恢复；故障及未消费回复阻止下一操作。取消与恢复用既有短本地锁排序。不得持键盘/状态/邮箱锁跨 SDK、join、大数组读取或模型计算。arm 与 hand 的顺序提交不被描述成原子事务，Python 也无法撤回已进入的 native 调用。

## 4. 实施任务

### T1. 修复 async 零进展与等待期限失效

主要文件：`deployment/rollout.py`，相关回归见 V01–V03。

- [x] 保留启动时建立的首发绝对期限，包含已有 history 积累预算；在第一条 `robot.send_action` 正常返回前，不因 submit、新计划、预测完成或过时结果重新预测而更新/清除它。
- [x] 删除 `step()` 接受 `new_plan` 就清 `wait_deadline_ns` 的行为。候选通过映射、计划变得非空、Future 完成都不算设备提交成功。
- [x] `submit()` 不得覆盖尚未耗尽的无提交预算。需要建立新的预测等待期限时，只能在新的实际进展边界后建立一次，并受已有目标 `expires_ns` 限制；同一无提交阶段只可沿用或缩短。
- [x] 成功发送后记录新目标期限，并结束旧等待阶段。将已有 `target_expires_ns` 也纳入主循环的停止预算选择，使“有计划却反复丢弃候选”也有界；不要只依赖下一次无计划 submit 才检查旧目标期限。
- [x] 预算检查覆盖模型结果消费、映射之后和实际发送之前；不得让超时后的重提预测绕过预算。复用现有期限字段和 `budget_limit/expired_budget`，不加 watchdog 线程、新计数器或故障状态机。
- [x] 保持 async 裁掉过时动作、RTC 严格交接、Future 原错误传播、duration 同时到期优先级和已有目标租期。首发失败应明确结束并收尾，不在无时长上限时永久“忙而不发”。

实现判断标准不是“预测很活跃”，而是有成功目标提交，或在既定预算内带真实原因退出。

### T2. 启动取消贯穿到底，结束意图回到主线程

主要文件：`deployment/session.py`、`teleop/run.py`、`teleop/loop.py`、`runtime/{state,keyboard,lifecycle}.py` 及受影响调用方。

**启动取消：**

- [x] 在启动流程开始、键盘开始接收取消之前捕获一次本地取消值；贯穿资源准备、connect、camera/pointcloud/VR 就绪、手部初始化、模型 load/warmup 等真正的长等待。不要在 S/C/D 之后重新捕获新值，把被取消的启动当成新的启动。
- [x] `check_startup`、`wait_model`、各 `may_start` 使用同一取消上下文；取消值改变或 quit/emergency 即离开等待。`requests_allow_motion()` 仅检查 quit/emergency，不能单独充当完整启动取消判断。
- [x] 每次等待完成、进入下一阶段前再检查取消，避免取消与 ready/Future 完成同时出现时继续启动。设备/worker 的真实故障照常报告，不被普通取消覆盖。
- [x] 不用“stop 为真就一律取消启动”替代上述修复：内部 prepare/HOME 本来会置 stop，这与用户取消不同。内部停止不递增用户取消计数。
- [x] 取消停止的是编排和后续操作，不声称能强杀已运行的模型线程/native 调用。保留真实 Future/进程供现有有界收尾；不继续 warmup、不创建下一段 recorder、不取得运动授权。

**结束意图：**

- [x] 删除 `SessionState.run_ended_reason` 及开始操作时清它的逻辑；删除 `revoke_motion` 上仅用于回取旧原因的 `reason`、`expected_cancel_epoch` 参数及返回值。停止动作与结果归因分开。
- [x] 这里删除的是 `revoke_motion` 的原因协议，**不是** prepare/resume/HOME 等长操作需要的 `cancel_epoch` 比较。共享 stop/emergency、quit 请求和必要的小交互状态继续保留。
- [x] 键盘回调只在既有短临界区设置取消/退出/急停事实并入队；主线程解释已接纳终止事件，把结束原因写入现有当前操作结果。`RunEndReason` 若仍简化结果表达可保留，不能再成为全局镜像。
- [x] 删除 teleop `operator_pending` 和“回调已快停但还没入队”的等待协议。回调与入队已在同一 Condition 内，主线程取得该锁后可取得一致的输入快照。
- [x] 取已接纳终止事件及用于判别的 stop 快照时使用同一短输入临界区，锁外处理事件和 I/O。不能先无锁 poll、再读取一个后来改变的 stop，就断言是无操作者事件的执行器停止。临界区之后的新事件留给下一次处理，快停信号已即时生效。
- [x] 保持现有顺序：冻结保存决定前消费已接纳终止输入；D→Q 仍丢弃，冻结后的 D 不追溯修改已决定的保存；quit/emergency 独立保留。不要为了移除一个全局字段增设另一套 stop_reason / pending_reason / generation。
- [x] busy 接纳、held keys、停止与退出响应、jog 的 S/D 方向键语义保持正确；锁序继续 Condition → 本地状态锁，不反向嵌套。

### T3. 把 replay 收敛为一次顺序 demo 播放

主要文件：`scripts/replay_episode.py`、`replay/{run,trajectory}.py`、必要的公共 preflight 调用方。

#### T3.1 最小输入与文件边界

只支持当前项目的绝对关节目标，不新增通用轨迹格式。CLI 保留 episode 目录和 `--config`；读取该目录下的 `data.h5`。

| 读取内容 | 检查 |
| --- | --- |
| `action_arm_joint_target` | 非空实数数组，形状 `(N, 7)`，数值有限。 |
| `action_hand_joint_target` | 形状 `(N, 12)`，与 arm 等长，数值有限；本次 replay 仍要求启用 XHand。 |
| `meta.control_hz` | 有限正数，符合公共设备的发送频率上限。 |
| `meta.joint_units`、arm/hand joint names | rad 和当前设备顺序；一次检查，不猜单位、不重排、不迁移。 |
| 当前 arm / hand 配置范围 | 一次检查整段目标；超过配置边界 1e-6 rad 时连接前失败，容差内微小越界归一到边界，不 wrap 到另一分支。 |

- [x] 用短小的 h5py reader 一次读出上述数组和属性，关闭文件后再连接设备。允许 float32 / float64 等可转成实数目标的正常数值 dtype，不要求与 Raw writer 完全一致；复数、字符串、非有限值不被静默转换成合法动作。
- [x] 不依赖 `RawReader` 的完整模态/时间契约；不要求 `timing/rows`、历史首行 `arm_qpos/hand_qpos`、task label、历史时钟声明、RGB-D、触觉、完成状态或回放审计字段。动作字段本身定义绝对关节目标，不再证明历史 SDK 是否执行成功。
- [x] replay 移除 `require_published_raw` 和路径前缀门禁。显式路径下的文件能正常读取且目标有效即可，包括用户明确指定的 `.tmp_*` 目录；不自动发现临时目录，不读写并修复坏文件，不发布/重命名数据。正在写入、锁冲突和损坏文件按实际 I/O 错误失败，不加入重试或实时追读。
- [x] NaN 行、空序列、错列数、缺必要属性仍报错；不自动筛行、插值、补零或回退到实测关节位置。允许不完整的存档模态不等于允许不完整的目标序列。
- [x] 注意当前 `Robot.validate_command` 的 hand 检查使用额定硬件范围，未必覆盖更窄的 `HandConfig.qpos_min_rad/qpos_max_rad`。因此保留此处一次整段配置范围检查；不能删掉 mapper 后误以为公共 hand 校验覆盖了全部实验限位。无需为此扩展全仓命令接口。

#### T3.2 顺序、节拍与起点

- [x] 一次连接和公共 streaming prepare 后读取当前 arm/hand 新鲜反馈，只与**第一行目标**比较。沿用一个粗阈值：任一关节绝对差大于 10° 则报错并关闭，提示实验者先就位；不自动规划到第一帧。
- [x] 删除 `prepare_hand`、历史实测起点、手部 ramp、三帧到位、arm 漂移重复检查及 `--hand-start-duration`。开始播放时的第一行目标就是第一条 demo 目标。
- [x] 直接 `RobotCommand(arm_targets[i], hand_targets[i])` → `robot.send_action(..., expires_ns=now + COMMAND_LEASE_NS)`。不调用 replay 的 `ActionMapper`、IK、`MotionConstraints`、`project_*` 或 `wrap_nearest_equivalent`。公共设备校验和 arm Ruckig 执行仍在。
- [x] **删除 replay 在线几何/碰撞预检是本方案的明确取舍。** replay 不需要 Pinocchio、hpp-fcl、碰撞 mesh、桌面标定或 HOME planner 才能加载/播放关节目标；不把任意新场景、任意人工轨迹宣称为已验证可执行。teleop / policy / HOME 的几何检查保持。
- [x] 每一行开始时记 `iteration_start`，服务反馈和用户终止，提交该行；只等待 `max(0, period - elapsed)`。下一行从新的实际时刻开始；无绝对原点、slot deadline、累计 lateness、missed slot、补发或丢行策略。
- [x] 等待沿用短周期服务：继续检查设备/键盘健康并采集所需手反馈，不用一整段不可中断 sleep。公共 owner 负责 arm 反馈；replay 只保留起点快照，不逐行读取整份状态或重复判断反馈 TTL、旧目标期限。
- [x] 成功提交后才增加显示用的 `sent_count`。公共 arm 邮箱仍是 latest-value；有序提交不等于每行都获得 SDK 回执或物理到位，不为 replay 改成 FIFO/ack 协议。
- [x] 最后一行正常提交后，以**固定 0.5 s 可中断留驻替代最后一行的普通周期等待**，然后关闭。不附加另一轮完整周期，不续租，不做收敛证明或三帧到位。真实 I/O 错误、急停和公共 stop 仍打断；hand 没有新样本但服务接口无错时允许完成留驻。这段留驻不保证到位。
- [x] 输入很慢或 native 调用阻塞以致公共目标租期到期，仍按公共执行规则停止；不为任意低频录制新增保活线程、自动重发或无限租期。

#### T3.3 单次流程、退出和错误

- [x] 流程只剩：读取并校验 → 创建键盘/设备 → 连接与 prepare → 粗起点检查 → 顺序发送 → 短留驻 → 唯一 `finally` 公共收尾。
- [x] S/Q 均停止本次播放并退出，不进入“已停止但仍在等待”的会话；Esc/Ctrl+C 使用公共 emergency 路径并非零退出。取消在连接、准备、行间等待和尾帧留驻期间都有效。
- [x] 正常播完、操作者 S/Q 且关闭成功，CLI 返回 0；坏输入、起点拒绝、无操作者输入的公共设备停止、起点反馈失效、发送前设备拒绝及 SDK/关闭失败，CLI 非零。不能把所有 `OperationCancelled` 都吞成用户正常停止；设备故障与用户取消同时出现仍报告故障。
- [x] 删除 `ReplayStatus`、`ReplayResult`、`ReplayBuffer`、`save_replay_data`、提交候选/实测/时间戳审计数组、`session_result.json`、`replay_data.npz` 和结果去重代码。正常返回、明确的用户停止、真实异常已足够表达流程；不要换成另一套结果 envelope。
- [x] 删除 `output_dir`、`DEFAULT_OUTPUT_DIR`、`--output`、目录必须为空的校验和所有强制输出。控制台打印总行数、已提交行数、耗时和必要停止原因即可；同一命令可重复运行。旧产物留在原地。
- [x] 删除 `wait_for_home`、H 交互、`home_results`、回放中的 HOME planner 和相应取消分支。正常播放结束自动 close 并退出。
- [x] 复用 `close_runtime` 的停止/关闭与真实句柄处理；不叠加内外两层 replay 结果分类和重复 stop-finally。全部已创建资源在连接失败等部分启动路径也能被关闭。
- [x] 若运行主错与关闭错误同时存在，保留主错，记录关闭错误 traceback；只有关闭失败时也必须非零。错误不被最终普通 return 覆盖，不为去重存异常对象身份或添加错误注册器。

#### T3.4 模块和依赖

- [x] 将实际 runner 放入现有 `replay/run.py`，CLI 保持薄。reader 可与 runner 同文件，或仅在可读性明显更好时保留 `trajectory.py`；删除已无职责的 `joint_replay.py` 及无用导出。不为“单文件”把公共设备逻辑复制进去。
- [x] replay 的 preflight 只验证实际使用的 arm/hand 配置、反馈服务与发送频率；移除该入口对 environment/workspace/HOME 的附加验证。若公共 `validate_robot_config` 混合设备和场景校验，可把场景检查移到实际使用它的现有入口，同时更新所有调用方；不增加多种可选校验模式。
- [x] 核实传递导入，确保 replay 的帮助、reader 和假设备播放路径不加载碰撞/IK/相机/视频依赖；不要只删构造调用却留下顶层重依赖。其他功能仍用的依赖和文件不能全仓删除。

### T4. sync 脱离 async / RTC 时间窗口算法

主要文件：`deployment/rollout.py`、`deployment/schedule.py` 和配置校验/调用方。

- [x] sync 的真实状态只需当前动作数组、已执行游标、一个在途模型任务、下一次采样/执行时点，以及 T1 的公共等待/时长限制。直接表达“积累观测 → 预测 → 顺序执行 A 行 → 下一次预测”。
- [x] sync 不再创建伪造的观测 slot/handoff slot，不调用 `handoff_plan`、`prefetch_handoff`、`plan_rejection` 来表达顺序游标；`schedule.py` 只服务实际需要时间窗口的 async / RTC，清理各函数的 sync 特判。
- [x] 可以在现有 Rollout 中保留短小 mode 分支/方法；不为三种模式创建继承树、可插拔调度器或三个拥有独立生命周期的 Executor。观测、映射、公共发送、录制及收尾仍复用。
- [x] 模型等待期间不重复把同一行塞入 history；history 按最近 N 次已接纳采样维护（本轮撤销断档清空），保持第一预测 reset RNG、预热恢复 RNG、A/P 配置语义及公共来源新鲜度检查。
- [x] 映射耗时 63 ms 而周期 62.5 ms 时，sync 仍发送有效动作并推进游标；不因名义 tick 过期丢弃，也不再额外睡完整周期或突发补发。发送失败不前进。
- [x] async 仍可以使用旧预测有效尾部；RTC 的前缀和交接检查不移除。本任务不以删模式来换取简单。

### T5. 数据与错误链的局部减法

主要文件：`data/{recorder,raw,export}.py`、`teleop/loop.py`、`deployment/rollout.py`、`robot/xarm7/process.py`。

**录制和采样：**

- [x] 删除 `submitted_frames` 及递增/断言；writer 已结束且无错误后，用 `written_frames > 0` 判断可发布非空数据。writer 未完成或 I/O 失败时不得拿当前计数提前发布。
- [x] 保留 writer 的真实 frame_count、queue/freeze/finish 唤醒、首个错误、真实线程句柄和有界关闭。`is_recording=False` 不等于 writer 已退出。
- [x] 控制循环在 recorder 确实接收时读 control/record 并集；无录制或已冻结时读 control 所需字段。冻结后的下一轮不再复制仅供存档的 RGB-D/触觉，不停止控制必需来源，不清空其最新有效缓存。
- [x] 不为减少这次复制而添加动态 worker 重启、重新录制或按字段管理资源的框架。冻结保持不可逆；同一 episode 不因存档来源恢复而续录。

**数据事实：**

- [x] 删除 `technical_status` 的 Raw 初值/finish 参数、`_FinishRequest` 字段、控制层传参、Canonical quality 复制、文档和测试。不要改名为另一个等价成功标签。
- [x] 保留 `termination_reason`、错误详情、`recording_truncated_reason`、实际行数以及 recorder 的 published/failed/pending 结果。运行尾部失败不反向否定已正常发布的有效前缀。
- [x] RawReader 继续支持诊断指定 staging；Canonical 导出仍检查发布边界与完整数据。将 `require_published_raw` 收窄到请求 episode basename 与解析后 episode basename：真正 `.tmp_episode` 及其别名仍拒绝，普通 `.tmp_workspace/episode_good` 允许。不遍历祖先，不维护路径 provenance 或符号链接安全框架。
- [x] replay 不调用上述导出发布门禁。该分离是不同用途的明确约定，不是新增 strict/relaxed 模式。
- [x] Canonical 的固定模态、单位/形状、RGB 数量、来源时间与触觉年龄、相邻 control 间隔 ±25% 和观测不跨下次 control 等现有约束保持；`dt` 仍为名义周期。只删除已列出的冗余状态，不借 replay 放宽删去训练数据契约。
- [x] 保留“失败 Raw staging 不自动删除；可再生的导出 staging 失败只清本次目录”的行为。

**错误：**

- [x] `ArmProcess` 直接读取 owner fault/close_error 信息并抛出正常异常；移除 `_fault_error`、`_close_error`、`.args` 改写和 `reported_fault/reported_close_error` 身份去重快照。
- [x] owner 真正异步产生的故障通道、stop/disconnect 失败信息、join 后晚到错误和异常退出检查保留。这里删的是父进程重复包装和去重状态，不是故障本身。
- [x] 允许运行和收尾重复报告同一 owner 错误；不要求一个异常在整个系统只出现一次。已退出但失败与仍活着的 pending 继续按真实句柄区分，不用错误是否已打印推断资源是否结束。
- [x] 清理受影响代码里只为此维持的异常身份比较和帮助函数；保留必要的主错变量、异步 writer error 和“主错不能被次级收尾错误覆盖”的普通 try/finally。不要把所有错误处理一概删除。

### T6. 文档、测试和入口一次改完整

- [x] 更新 README 的 replay 命令、键位、输出目录、数据条件、时间语义、几何依赖和 HOME 使用说明；删除所有“严格 replay 槽位”“回放结束按 H/Q”“必写 NPZ/JSON”的现行约定。
- [x] 更新 CLI help、受影响模块 docstring、当前数据读写与配置调用方；无需给已删参数加 deprecation 周期。历史版本无需改写。
- [x] 改写因需求主动改变而失效的测试：特别是 replay staging 拒绝、精确 dtype、旧审计结果、旧 `technical_status` 和 `submitted_frames` 断言。保留这些测试原本覆盖的真实错误/资源边界，不以删除整组回归规避问题。
- [x] 按第 5 节完成行为验证，再审查实际 diff。不要用“搜索不到旧类名”“AST 可解析”代替运行行为验证，也不要为了填满清单新增与实现一一对应的测试。

## 5. 验收场景

优先扩展已有五个业务测试文件；replay 如有必要可增加一个集中的测试文件。使用 fake driver、可控时钟和真实小型线程/Future，不连接硬件。表中的场景可参数化/合并，不要求每行变成独立测试或新测试框架。

| 编号 | 场景 | 必须观察到的行为 |
| --- | --- | --- |
| V01 | async 16 Hz、8 行预测、600 ms 返回、无旧目标、无时长上限 | 在首次绝对等待预算内退出；重新预测不能不断延后期限。 |
| V02 | async 100 ms 预测、63 ms 映射，候选持续跨槽 | 来源与完整预测窗口有效时持续提交；整体过期的候选不发送，持续无进展仍受原预算约束。 |
| V03 | 有旧目标但一直无新提交；运行时长/等待期限同时到期；Future 抛错 | 不越过旧目标期限；相等按 duration；真实模型错误保留。 |
| V04 | 在 camera/pointcloud ready、模型 load、warmup 等等待中输入 S/C/D/Q/Esc；ready 与取消同时发生 | 及时退出对应等待并收尾，不执行下一阶段；内部 HOME 的 stop 不误取消；未完成 Future/进程仍有人持有。 |
| V05 | 快停回调持 Condition 暂停；poll/终止快照交错；D→Q、freeze 前后 D、busy 长按启动键 | 终止意图与已接纳输入一致，无旧事件启动下一段；保存/丢弃时序与锁序正确。 |
| V06 | sync 16 Hz、63 ms 映射及慢预测 | 成功发送推进游标，顺序和 A 行窗口正确；不补发、不额外睡满周期，等待受预算约束。 |
| V07 | history 调度抖动或模型等待造成不等间隔；RTC 错过交接或承诺槽缺观测 | 历史保留最近 N 次已接纳采样；RTC 继续明确结束。 |
| V08 | 最小 HDF5 仅含 T3 必要字段，float32 目标，没有 timing/RGB/历史实测起点；显式 `.tmp_*` 路径 | replay 可加载；通用 RawReader/Canonical 的完整模态要求不被迫放宽。 |
| V09 | 空/错形状/不等长/NaN/复数/单位或顺序错误/配置越界的动作文件 | 真正越界在连接设备前失败，容差内微小误差归一后通过公共发送检查；其余坏输入无补值、筛行或自动迁移。包含较窄 hand 配置范围和 float32/float64 边界用例。 |
| V10 | replay 300 行、16 Hz、每次发送 63 ms；另测正常快速发送 | 全部按序提交，不因累计晚点退出或跳行；快速情况只等剩余周期，尾帧 0.5 s 留驻替代普通等待。 |
| V11 | replay 第一行姿态偏差；连接/prepare/播放/留驻中 S/Q/Esc；重复同一 CLI | 大偏差拒绝且关闭；S/Q 正常停止退出，Esc 非零；第二次不被输出目录拦截，不创建审计文件或等待 H。 |
| V12 | replay 发送异常、无操作者输入的公共 stop、起点/发送反馈失效、关闭失败；主错和关闭错同时发生 | 非零且错误可见；无吞错继续下一行、无正常 return 覆盖异常；所有已创建资源都尝试收尾。 |
| V13 | 不提供碰撞/IK/视频依赖，用最小文件和 fake Robot 走 replay；真实手反馈服务计数 | 不加载 Pinocchio/hpp-fcl/mesh/相机解码；等待期间继续服务反馈和取消。公共硬件依赖不伪称已经验证。 |
| V14 | 实际 writer 线程被阻塞、队列满冻结；另测零行、写入失败、join 超时 | 有效 17 行前缀仍可关闭发布；非空判断仅用关闭后的 writer 计数；零行不发布，失败留 staging，超时保留真实句柄。 |
| V15 | recorder 冻结前后观察读取字段；控制仍需相机/点云的另一用例 | 只停止复制存档专用字段；控制所需字段仍读取，运动与来源 TTL 正常；不会恢复录制。 |
| V16 | 导出位于 `.tmp_workspace/episode_good`；真正 `.tmp_episode` 和指向它的别名；普通临时祖先下指向已发布 episode 的别名 | 普通祖先不误拒绝；实际 staging 仍拒绝；与 replay 的显式文件约定区分。 |
| V17 | 运行 owner fault、join 后晚到 close_error、已退出但失败、进程仍活着；Future 返回 None/抛 TimeoutError/未完成 | 错误不因去缓存而丢失；重复报告可接受；完成失败和 pending 正确区分，同一任务不续关闭预算。 |
| V18 | 原有停止/清邮箱/prepare 恢复、共享快照、HOME、Canonical 有界时序与导出失败清理回归 | 上阶段成立的不变量不倒退；不删除实际使用的桌面标定归档字段。 |
| V19 | prepare 耗时 200 ms 后 S/Q，旧反馈自然变旧；另测并发真实 owner/关闭故障 | 用户取消且关闭成功时正常退出；真实错误继续非零。 |
| V20 | 尾帧留驻期间 hand 无新样本；另测真实 I/O 错误、急停和公共 stop | 接口无错时完成 0.5 s 留驻，其余情况及时结束。 |

在项目依赖可用的环境执行相关测试及业务回归；例如 `python -B -m pytest -p no:cacheprovider -q tests`，以及可用时 `python -B -m ruff check --no-cache --select F dexmani_real scripts tests`、`git diff --check`。遵守 AGENTS.md，不在仓库生成 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`。缺依赖时写明未执行的具体检查，不能以导入/静态解析冒充通过。

验证应覆盖真实的 h5py 最小文件读取和实际公共 send/close 边界上的假设备行为；仅对整套 runner 打桩返回成功没有证明力。真机起点阈值、尾帧留驻体验、运动跟随和停止延迟由实验者后续验收，不在本任务自动执行，也不预写“真机通过”。

## 6. 顺序与完成标准

推荐顺序：T1 修复预算 → T2 修复启动取消并收拢结束意图 → T3 完整重写 replay → T4 简化 sync → T5 局部数据/错误减法 → T6 同步文档并完成回归。每个机制连同实际调用方一次改完整；阶段之间可以有清晰提交，不并存新旧实现。

- [x] R01–R16 均有明确落地结果；没有把任务改成再次提出方案或仅写 TODO。
- [x] 核心结构变少：replay 没有独立会话/审计/严格网格，sync 没有伪时间窗口，全局终止原因和错误身份台账消失，数据事实不重复存储。
- [x] 真实边界仍可解释：文件一次检查、设备副作用边界检查、来源新鲜度、目标租期和实际 pending 各自解决不同问题。
- [x] 文档、CLI、当前数据生产/消费和测试一致；没有兼容层、空壳旧类、无用参数、备用严格模式或换名机制。
- [x] 检查现有工作区并保留他人修改；不改用户实验数据，不做无关目录搬迁，不添加没有实际用途的依赖。
- [x] 完成后在本文末尾补实际实施记录：修改项、删除项、行为测试及结果、未执行项、与本文不同的必要调整及源码依据。只勾选实际完成项，不复制历史通过数量。

若实施时 HEAD 已前进，先比较本基线与当前代码，已修复的问题不重复实现。合理的局部实现选择可以自行决定；只有真实新调用关系或需求冲突才调整方案并解释，不以假想未来需求恢复复杂度。

## 7. 参考项目与采用范围

| 项目与固定源码 | 采用 | 不照搬 |
| --- | --- | --- |
| [LeFranX 顺序回放](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_robot_replay.py#L148) | 直接 for-loop 发送，按本轮耗时等待剩余周期，慢了就降低速度。 | 其发送异常后继续下一行的处理不适合本项目；不复制旧格式兼容。 |
| [T-Rex 硬件回放](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/replay_h5.py#L338) | 明确的目标读取、顺序迭代、限频与停止退出。 | dexmani 已保存绝对关节目标，不需要复制双臂 IK、额外 Manager 或动作缓冲体系。 |
| [ManiUniCon 回放工具](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/tools/replay_record_data.py#L84) | 单用途工具按用途读取数据的组织方式。 | 该工具主要服务 Meshcat 可视化，不作为真机关节回放的停止/执行依据；传感 replay 同样不是设备执行器。 |

参考代码提供结构借鉴，不替代 dexmani_real 实际入口、设备协议和上述行为测试。


## 8. 第二阶段实施记录（2026-11-03）

实施起点为 `7cea1dacec438bdfaa328c42a3f5ce92351b7e19`；与审查基线相比只有任务书变化。保留工作区原有键盘点动 `--no-hand` 实现及对应 README 内容。未改动实验数据、checkpoint、标定或旧回放目录。

### 已实施

- T1：首发等待预算不再被 submit 重建或被新计划清空。主循环同时检查等待、已有目标租期和 duration；同刻优先 duration。慢预测和慢映射零提交均在原预算内失败。
- T2：启动 epoch 在键盘启用前捕获，贯穿传感器、load/warmup、手初始化和准入；ready/Future 完成后再检查。`arm` 接收该 epoch。删除全局结束原因和撤权原因协议；当前操作解释终止输入。`Keyboard.poll` 在同一 Condition 内返回事件及 stop 快照，移除 operator_pending。
- T3：最小 h5py reader 独立于 RawReader；直接逐行发送绝对目标，使用本轮剩余周期及末行 0.5 s 留驻。删除 joint_replay 模块、映射/几何入口、历史起点 ramp、绝对槽位、审计管线、输出目录和 HOME 会话。CLI 仅保留 episode/--config；S/Q 正常结束，急停及真实故障非零。唯一 finally 使用公共 close_runtime。
- T4：sync 使用动作数组、游标及 PolicyWorker 已有 Future，schedule 只处理 async/RTC。结果到达可立即执行，等待帧不重复进入 history，提交成功才推进游标。
- T5：删除提交计数、technical_status 全链及异常身份去重。writer 确认退出且无错后按 written_frames 发布非空前缀；冻结后只读取控制字段。导出门禁只检查两个 episode basename。
- T6：调用方、CLI、README 和测试同步更新；保留旧测试对应的资源、来源时效、停止恢复、HOME 和 Canonical 边界。

### 源码依据与局部调整

- `ArmController.operation → wait_settled` 和 `tick` 实际使用 `homing.convergence_rad/velocity_convergence_rad_s`。这些名称下的停稳阈值仍属于公共 streaming 设备保障，保留校验。仅 HOME 使用的目标、超时和速度校验移到 HOME 使用边界；Policy 起点检查仍提前验证其 HOME 配置。
- `Robot.validate_command` 的 hand 上下界来自额定硬件常量；reader 另外整段检查当前配置范围，没有删去该保障，也没有改变公共命令接口或裁剪文件目标。
- `XHandFeedback.snapshot` 直接返回只读缓存。冻结后的减法集中在观测字段选择，不新增触觉缓存状态、动态 worker 重启或按字段资源管理。
- 保留独立的短 trajectory reader 和现有 runner，删除无职责的 joint_replay。reader 和 fake 播放不要求 Pinocchio/hpp-fcl/视频依赖。

### 实际验证

使用 `/usr/bin/python3`，测试依赖安装在 `/tmp/dexmani-test-deps`；未修改项目依赖声明。系统最初缺 pytest/Ruff/SciPy/av，且系统 Zarr 为 3.x，临时测试目录补齐项目要求的 Zarr 2.18.7、NumPy 1.26.4、Ruckig 0.17.3 等依赖。Ruckig 首次构建因上游 scikit-build-core 参数不兼容失败，使用 `/tmp` 构建约束 `scikit-build-core<0.10` 后成功。

执行：

```bash
PYTHONPATH=/tmp/dexmani-test-deps:$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -p no:cacheprovider -q tests
PYTHONPATH=/tmp/dexmani-test-deps:$PWD python3 -B -m ruff check --no-cache --select F dexmani_real scripts tests
PYTHONPATH=/tmp/dexmani-test-deps:$PWD python3 -B scripts/replay_episode.py --help
git diff --check
```

首轮结果：125 项测试通过；Ruff F 检查、CLI help 和 diff 空白检查通过。没有在仓库生成 pytest/ruff/mypy 缓存。

行为证据包括：

- 慢预测 600 ms、慢映射 63 ms 的 async 零提交预算；旧目标期限和 duration 同刻优先；sync 63 ms 映射推进游标、history 抖动与 RTC 错过交接。
- S/C/D/Q/Esc 中断模型等待、ready/完成与取消同刻、Condition 内暂停快停回调与输入快照、保存冻结和 busy 长按；真实 Future 的 None、任务 TimeoutError 和未完成状态。
- 真实最小 float32 HDF5、显式 `.tmp_demo`、坏数值/形状/类型/单位/顺序/手部配置范围；通过真实 Robot.send_action/close 接口的假 driver 顺序播放。
- 300 行每行 63 ms 全部提交，用时 19.4 s（含尾帧留驻）；快速发送用时 19.1875 s；原目标数组逐行不变。相同 runner 和 CLI 重复运行，不生成审计文件。
- connect/prepare/行间/尾帧阶段 S/Q/Esc，起点拒绝、反馈失效、无输入 stop、租期耗尽、主错及关闭错并存；所有已创建资源均尝试关闭。
- 实际 writer 线程队列满后的 17 行有效前缀、零行不发布、写入错误和 join 超时保留句柄；冻结后观测字段缩减、导出 basename/别名门禁、真实 HDF5/视频与 Zarr 发布和失败清理。
- owner 原故障和 join 后晚到关闭错误、进程仍活着与已退出的区分；停止/清邮箱/prepare 恢复、共享快照和 HOME 回归。

### 验证限制

所有设备均为假设备，未连接机器人，未宣称真机验收。真实 checkpoint/GPU load、warmup 和 SDK/native 阻塞的实机行为未执行；本轮验证其编排取消、Future 归属和错误传播。10° 起点阈值、0.5 s 留驻体验、实际运动跟随及停止延迟仍由实验者验收。软件等待期限不能强制终止已经进入的 native 调用。


### 全链路审查与对齐记录

在第二阶段工作区上继续沿入口、设备与数据调用链审查，直接修复以下遗漏：

- teleop 的 VR/骨长标定启动取消统一抛出 `OperationCancelled`，删除 bool 成功/取消双轨返回。启动 S/C/D/Q 走同一正常关闭出口，Esc 非零；Policy 与 teleop 只用原始 epoch、quit/emergency 识别用户取消，内部 stop 不再冒充用户请求。
- owner 因目标租期置 stop 与 duration 同刻到期时，先检查真实模型结果和既有预算，保留 duration_complete；模型错误及用户终止仍独立保留。
- replay 在连接前后检查无操作者输入的 stop，禁止 prepare 清除它后继续运行；收尾阶段收到 Esc 也非零退出。reader/runner/CLI 统一类型与 `control_hz` 命名，helper 显式接收设备，避免把部分启动句柄的 Optional 状态带入播放主体。
- `ArmProcess.close` 对同一 owner 只建立一次绝对 join 期限；重复调用使用剩余时间，仍活着的进程保留 Pipe 和真实 pending，真正退出后才关闭端点。删除停止确认对错误通道的重复读取。
- 批量导出临时工作根目录只负责发现数据，识别实际 episode 后才执行 basename 发布门禁；每个 episode 及其别名继续校验。真实 Zarr 回归从 `.tmp_workspace/` 批量入口执行。
- 配置/HOME 回归确认未使用 HOME 的 streaming 不依赖 HOME 目标或超时，公共停稳阈值及禁用手部时规划使用的固定姿态仍检查；清除单字段反射循环、多余原因别名、未使用 NONE 和过时注释。相关生产链未发现 SHA-256、格式版本或兼容迁移机制，无需新增替代方案。
- README 的取消、关闭、批量导出说明与实现对齐，修正实施记录的重复节号；保留原有 `--no-hand` 行为。

最终验证：完整离线业务套件 **150 项通过**；Ruff F 检查、四个受影响 CLI 的 `--help`、`git diff --check` 通过。所有业务设备均为 fake，覆盖公共 Robot.send_action/close、真实小 HDF5、视频、Zarr、线程和 Future。新增用例包括关闭期间急停、关闭超时保留句柄、重复 owner close 不续期、启动取消入口退出码、租期与 duration 同刻及模型错误、创建设备前拒绝坏目标和批量临时目录。

对 22 个受影响源码执行常规 mypy：通过。工具位于 `/tmp/dexmani-review-tools`，使用 `--no-site-packages --follow-imports=silent --ignore-missing-imports --explicit-package-bases`，缓存位于 `/tmp/dexmani-review-mypy`；此检查不覆盖未标注方法体及第三方库类型。另执行 `--check-untyped-defs` 扩展检查并与 HEAD 的同一组源码对照：当前 Raw/PolicySession 仍有 36 项既有生命周期 Optional 标注报告，基线为 38 项，未出现新的错误消息；因此不将扩展检查称为通过，也没有用额外空值检查或 ignore 掩盖它们。新增 replay 与控制分支的类型收窄问题已修复。

最终源码副本在 `/tmp` 成功构建 wheel，核对包含当前 replay 两个模块且不包含已删 joint_replay；构建、工具安装及全部缓存均留在 `/tmp`。未连接真机、加载实际 checkpoint 或执行 GPU 推理，真机验证限制继续适用。


## 9. 本轮减法重构实施记录（2026-11-04）

基线与实施前 HEAD 均为 `2b1e846bbf59c786c0fbe23b053732880909eade`，工作区干净。本节覆盖上一阶段关于 async 跨 tick 否决、history 间隔清空及 replay 私有反馈监控的旧要求；旧实施记录中的验证数量仅表示当时状态。

### 实际修改与删除

- RTC 已承诺计划的槽位缺少必需观测时明确 stop；首计划前仍按原预算等待。其他槽位返回路径保留严格交接、映射截止、来源检查和明确终止。
- 普通 async 接受窗口内晚提交；发送前按起点、anchor_slot、完整剩余动作长度和 dt 计算截止，等于截止也拒绝。耗尽计划直接移除，实际在途预测继续保留，重预测不续无进展预算。action_steps 只决定名义执行/预取，旧预测有效尾部仍可执行；每次控制机会至多发送一次。
- 删除 last_history_timestamp_ns 和 1.5dt 清空协议。history 保留最近 N 次已接纳观测；sync 等待/结果消费沿用原入窗位置，首次仍积累完整窗口，结束时清理。明确接受不等间隔历史，与严格等间隔模型输入并不等价；来源与 Raw 时间不变。
- replay 删除 verify_feedback、feedback 包装、保存旧目标期限及 stopped_by_user 标志。起点只读取一次当前姿态与新鲜度；随后顺序发送，每次附公共 2 s 租期，短等待服务 hand I/O、输入及公共 fault/stop。尾帧 hand 无新样本且接口无错时完成 0.5 s 留驻。
- replay 取消通过原 epoch、quit/emergency 和输入判定，真实 owner 故障继续传播，已取消操作不检查旧反馈/目标时效。主错与关闭错同时出现保留主错并记录次错，正常关闭后的 Esc 仍非零。
- teleop 主线程处理 Esc 时直接登记 ESTOP，批量输入复用同一处理路径，保留回调即时停止。
- reader 用固定 1e-6 rad 容差先拒绝真正越界，再归一微小误差；保留更窄 hand 配置范围，公共硬限位未放宽。

公共 Robot/owner 的 SDK 前检查、来源准入和长映射后来源 TTL、原始采样时间、停止后的旧目标隔离及真实 Future/thread/process/Pipe 收尾保持。没有新增协调协议、配置开关或依赖。

### 验证与限制

使用现有 /tmp/dexmani-test-deps，执行禁用缓存的完整 pytest、Ruff F、replay CLI help 和 git diff --check。新增用例执行生产 Rollout.step、TeleopLoop.service_episode、reader、Robot.send_action/close 和 ArmController.tick，替换设备、模型与时钟；公共 owner 的租期/反馈失效通过真实 tick 验证，replay 测试只接收公共停止事实。

覆盖 RTC 缺快照、prepare 200 ms 后 S/Q（事件已消费仍正常）、并发 SDK/关闭失败、poll 内 Esc、float32/float64 上下界和容差、公共发送限位、尾帧 hand 无新样本与真实 I/O 错误、发送时 hand TTL、async 100 ms 预测加 63 ms 映射持续提交、完整尾部截止、来源过期、sync 双帧历史跨模型等待，以及既有 300 行慢 replay 与资源 pending 回归。

所有设备均为 fake，未连接机器人、加载真实 checkpoint 或运行 GPU 推理；未验证真实物理跟随与停止延迟。未改动实验数据、模型和标定。


本轮初次实施验证：完整离线业务套件 **182 项通过**（1.38 s）；Ruff F、replay CLI help、git diff --check 均通过。四个变更生产模块的常规 mypy 检查通过，使用 --no-incremental --cache-dir=/dev/null --no-site-packages --follow-imports=silent --ignore-missing-imports --explicit-package-bases；该检查不覆盖未标注方法体和第三方库类型。未生成仓库内 pytest/ruff/mypy 缓存。实施前基线为 150 项通过，本轮新增及参数化扩展后为上述实际数量。


### 全链路复审与清理

本次在上述未提交改动上沿 Rollout/PolicyWorker、teleop 输入、replay/Robot/arm owner、配置和 CLI 继续审查并直接修复：

- 删除预取时的 history 长度门禁及 prefetch_history_incomplete。首次预测已有完整 N 帧，运行期不再清空 history；stop 同时清空 history 和计划，因此该门禁已不可达。async/RTC 预取仍使用原历史、前缀与交接承诺。
- 起点快照读取返回后通过现有 check_input 复核取消，避免 S/Q 在读取期间到达却被已变旧反馈误报。取消 handler 删除不再使用结果的 keyboard.poll，复用原取消事实。
- replay 将 KeyboardInterrupt 急停登记集中到 finally 主异常入口，覆盖 try 主体和 OperationCancelled handler 自身被打断的情况，仍在公共关闭前锁存急停。
- 合并 replay 中只靠人工置 stop 的重复租期用例，真实租期/反馈失效仍由公共 ArmController.tick 测试覆盖；清除 fake owner 的 join 闭包对整个 arm 对象的循环引用。
- README 和当前任务清单统一文件边界归一、尾帧无新 hand 样本、RTC 缺快照及不等间隔 history 语义；删除虚拟 helper 节拍示例、已删除模块的现行路径和相互矛盾的“无裁剪”描述。历史实施记录保留为当时事实。

新增回归包含真实 PolicyWorker Future 的 async/RTC 预取、起点快照耗时 200 ms 时 S/Q/Esc，以及 prepare 或取消处理检查中 Ctrl+C 的关闭前急停登记。未发现该调用链存在 SHA-256、旧格式兼容或迁移分支，无需增加替代机制。

本次最终验证：

- 完整离线套件 **188 项通过**（1.44 s）。Python 3.12 multiprocessing.resource_tracker 同时产生 64 条重入清理警告；未屏蔽警告，fake owner 循环引用清理后警告仍存在，未声称已排除信号量泄漏。为核对警告执行的 HEAD 基线副本为 150 项通过、无该警告；初次临时副本缺少 scripts/URDF 的准备问题补齐后已重新验证。
- Ruff F 检查通过；四个变更生产模块的 mypy --check-untyped-defs 通过；24 个相关源码的常规 mypy 通过。均使用 --no-incremental --cache-dir=/dev/null，另含 --no-site-packages --follow-imports=silent --ignore-missing-imports --explicit-package-bases；常规检查不覆盖未标注方法体，两者均不覆盖第三方库类型。
- replay、policy、collect_teleop、keyboard_teleop 四个 CLI 的 --help 通过，git diff --check 通过。
- 当前源码副本使用 python3 -B -m build --wheel --no-isolation 在 /tmp 成功构建 wheel，核对四个修改模块与源码内容一致，已删除 joint_replay 未进入包；构建产物和日志在 /tmp/dexmani-chain-build-oqvk1agg。未在仓库生成 pytest/ruff/mypy 缓存。

全部业务验证仍为假设备；真实机器人、checkpoint/GPU 推理、物理跟随和停止延迟未执行。

### resource_tracker 告警定位与修复（2026-11-04）

排查开始时 HEAD 为 `3787987d769df4c5a9522e9efd3b790a92017202`，工作区干净，无后续修复。使用 `/usr/bin/python3`：`3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]`，平台 `Linux-6.17.0-40-generic-x86_64-with-glibc2.39`。依赖沿用 `/tmp/dexmani-test-deps`：pytest 9.2.0、pluggy 1.7.0、NumPy 1.26.4、SciPy 1.17.1、h5py 3.16.0、Zarr 2.18.7、av 19.0.2、Ruckig 0.17.3、PyYAML 6.0.1、Ruff 0.17.3。系统默认启动方式为 fork，但涉事 `SessionState` 和 `ArmChannels` 明确使用 spawn context；arm 测试本身只运行线程或 fake process。

修复前三个变更测试文件单独运行为 123 项通过、无告警；完整套件复现 188 项通过、64 条重入清理告警。同环境、同完整测试命令对 `git archive` 导出的 `2b1e846bbf59c786c0fbe23b053732880909eade` 完整副本（包含 scripts/URDF）验证为 150 项通过、无告警。收集完整套件后仅执行 arm 测试也产生 64 条告警，而单独执行 arm 文件无告警，说明触发依赖分配/GC 时机，不要求执行后续业务测试。上述完整运行均无退出阶段 stderr；告警是 `ResourceTracker called reentrantly ... might leak`，不是 `leaked semaphore objects ... at shutdown` 泄漏报告，也没有测试失败。

具体调用链为 `SessionState → spawn.Event`，以及 `ArmProcess → ArmChannels → spawn.Event/Lock`，每个 arm 用例共创建 32 个命名信号量。`test_cancel_during_recovery_is_not_cleared` 的 mailbox-clear 替身保存 `clear_goal` 绑定方法，形成 `ArmChannels → clear_and_cancel 闭包 → 绑定方法 → ArmChannels` 循环，并持有 SessionState。临时诊断确认该对象在用例返回后仍存活，GC 后可回收；重复该分支 40 次产生 320 条告警。捕获的栈显示，新信号量注册期间 tracker 持锁，GC 触发旧 SemLock 的 finalizer 注销，再次进入 tracker。纯标准库循环引用 Event 也能复现，与 [CPython #109593](https://github.com/python/cpython/issues/109593) 和 [#131788](https://github.com/python/cpython/issues/131788) 的运行时重入问题吻合。此前清理 join 闭包不能消除这一独立触发路径。

本次仅在 `tests/test_arm_recovery.py` 增加每例自动恢复的 fixture，将 spawn context 的 Event/Lock 工厂替换为 threading.Event/Lock。该文件没有真正启动跨进程 owner，无需这些命名信号量；真实 Pipe、共享缓冲、线程等待、pending 句柄和故障断言保持。没有修改生产代码、解释器要求或依赖，没有屏蔽警告、修改 tracker、手动注销资源或加入生产 GC。这是测试资源使用触发运行时限制，未取得生产生命周期泄漏的证据。

实际验证命令：

```bash
export PYTHONPATH=/tmp/dexmani-test-deps:$PWD
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/usr/bin/python3 -B -m pytest -p no:cacheprovider -q tests -k test_arm_recovery
/usr/bin/python3 -B -m pytest -p no:cacheprovider -q tests
/usr/bin/python3 -B -m ruff check --no-cache --select F tests/test_arm_recovery.py
git diff --check
```

落盘后先通过现有 fixture 重跑最小复现 40 次：无告警，80 个 Pipe 端点全部关闭，无命名 SemLock、无活动子进程、仅主线程存活，原 Event/Lock 工厂恢复。随后相关测试 **24 项通过、164 项未选中**（0.89 s），完整离线套件 **188 项通过**（1.38 s），均无告警。通过父进程捕获完整 stdout/stderr 并等待解释器退出，两次 stderr 均为空，运行前后 `/dev/shm/sem.mp-*` 无新增残留；Ruff F 和 diff 空白检查通过。诊断脚本未加入仓库，未生成 pytest/ruff/mypy 缓存。

本次验证未观测到资源残留，不将其扩大为所有运行环境的释放保证；其他 Python 版本和真实设备未验证。原始来源时间、缓存不续期、映射后 TTL、目标租期、SDK 前检查、async 窗口内晚提交、RTC 承诺时序、最近 N 次接纳观测 history，以及 replay 的公共设备服务、正常取消与真实故障传播均未改变，相关离线回归随完整套件通过。
