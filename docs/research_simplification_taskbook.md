# DexMani Real：面向个人研究的减法重构任务书

> 本文是可直接执行的实现任务，不是继续扩展机制的设计提案。
>
> 审查基线：`haoyangzhanglab/dexmani_real@e28d397b0fb947e4a4f91007a1b1b167b946816e`。整理日期：2026-11-03。
>
> 本文合并两轮审查，并对互相冲突的建议作出最终裁决。执行时先核对当前代码与基线差异；以本文的目标行为为准，不机械叠加两轮建议。审查包含源码分析和假时钟、受控线程复现，不代表真机验证。

## 1. 目标与边界

让一个研究者能够沿着“输入 → 计算 → 发送 → 记录 → 收尾”理解、修改和运行项目。减少需要同时记住的状态、跨层约定和失败分支，而不是仅压缩代码行数。

- 保留 VR teleop、键盘 jog、HOME、sync / async / RTC 策略执行、Raw 录制、Canonical 导出和关节回放的实际能力。
- 不考虑旧 Raw、Canonical、会话 JSON、内部 API 或配置的兼容性；需要改动时直接更新所有当前调用方及文档。不新增迁移器、双读写、旧参数别名或兼容开关。
- 不自动清理用户已有实验数据、模型和标定文件。本文要求清理的目录仅限本次导出新建、失败后可重新生成的临时产物。
- 不新增通用 LifecycleManager、AuthorityManager、ResourceRegistry、Deadline/Grid 框架、可插拔策略体系或后台自动恢复状态机。
- 不以裁掉 RTC、无手运行等现有能力来冒充架构简化；Canonical 的完整模态要求仍可与无手控制的能力不同。
- 对真实外部输入、设备副作用、跨线程/进程完成事实保留必要检查；删除重复证明内部调用顺序的检查。

最终保留的基本结构很小：一个父进程主线程编排操作，一个 arm SDK owner，必要的采集 worker，一个在途预测，一个当前 recorder；状态归实际 owner 持有。正常动作序列、HOME 和下一轮开始不并发执行。

## 2. 两轮发现与最终裁决

下表及任务中的源码路径省略仓库根目录下的 `dexmani_real/` 前缀；锚点是基线代码中的符号，避免依赖后续会变化的行号。

| 编号 | 发现与代码证据 | 最终方案 |
| --- | --- | --- |
| F01 | `robot/xarm7/worker.py` 的 `prepared_run_id` 重复证明 prepare 已完成；所有实际发送都由父进程串行经过 prepare。 | 删除字段及判断，不换成 `prepared` 布尔值。保留 Mode 1 切换、实测停稳和首次轨迹参考检查。 |
| F02 | `runtime/state.py`、`robot/xarm7/protocol.py`、`process.py`、`worker.py` 把运行代际扩散成共享 Value、锁、目标字段、生命周期请求和 owner 裁决。 | 删除跨进程代际。保留一个父进程本地取消计数，配合明确的停止、清空、准备顺序；设备 owner 不识别运行身份。 |
| F03 | `publish_goal()` 检查后可被暂停；owner 的 `stop()` 用非阻塞 `take_goal()` 清邮箱，可能清空失败却设置 stopped。旧目标随后落入邮箱，新 prepare 后旧代际目标反而触发新一轮 stop。 | 修复恢复顺序：旧发布者退出后，在停止态持邮箱锁显式清空，再开放新操作。不能仅删除旧代际拒绝分支。 |
| F04 | `error_state` / `mark_error()` 把正在同步抛出的异常再编码成会话永久状态；`Robot.stop_failure` 缓存并反复重抛同一停止失败。 | 致命异常到达唯一 episode / session 边界；主错误及收尾错误留在结果中。先修复 replay 吞错路径，再删除这两套错误闩锁。 |
| F05 | `run_ended_id`、多层 `failure_reason()`、`device_failure_reason()` 和若干结果字段重复分类、拷贝终止原因。 | 正常结束意图与异常事实分开；当前操作保存一次结束意图，异常只记录 stage / type / message，消除代际关联与重复镜像。 |
| F06 | `runtime/keyboard.py` 在锁外取 `KeyEvent.run_id`，结束清队列后旧 B/H/T 仍可能入队；为此各层又多撤权一次。 | 按键接纳和入队由同一个短临界区决定；busy 时关闭启动命令入口。删除事件身份和仅用于清 UI 队列的撤权。 |
| F07 | `Keyboard.stop()` join 0.25 s 后不检查监听线程是否退出，`active=False` 使第二次 stop 提前返回。 | 完成与否看真实 listener；活动开关不能替代 pending。失败透明返回，仍保留句柄。 |
| F08 | writer、模型 Future、生命周期 Pipe 回复、owner 进程确实可能未结束。 | 保留真实 pending；不能按“已发 stop”“已超时”“字段设 None”推断已完成。已完成但抛错也不能误记为 pending。 |
| F09 | 正常 episode 收尾和外层 session close 可能对同一个模型任务重新等待完整预算；已决定退出时，机器人 close 排在慢资源后。 | 一个未完成任务只有一份绝对收尾期限。已知退出/故障时先停止并 close 机器人，再等待慢资源。 |
| F10 | `deployment/rollout.py:step()` 接受预测后清掉等待预算，映射稍慢又因 `now >= next_step_ns` 丢弃候选；sync 游标不动，可以永久“忙而不发”。 | 普通 sync 不因错过名义 tick 丢弃仍有效的动作；实际提交后才推进游标；无最大运行时长时也必须有进展或真实错误退出。 |
| F11 | async 的 history 用 `skipped > 0` 判定样本断档，名义网格边缘的小抖动就清空历史。 | history 按真正纳入窗口的控制采样时间间隔判定；RTC 的承诺槽检查另外保留。 |
| F12 | 来源新鲜度、名义周期、目标租期、预测等待和实验时长混用或耦合。 | 保留不同语义，减少跨层传播；来源时间不续期，目标租期由提交决定，普通 tick 不充当动作合法性期限。 |
| F13 | `teleop/loop.py`、部署入口把 control needs 与 recording needs 的并集用于停止裁决，存档专用相机/触觉质量下降也停止运动。 | 保留并集用于采样；控制和录制分别裁决。存档质量不足冻结有效前缀，停止运动后才收尾。 |
| F14 | `RawReader.validate_completion()` 仅因 `technical_status != normal` 禁止导出/回放；运行尾部故障会否定之前已正常关闭、发布的有效记录。 | 删除历史成功状态门禁；按当前容器、数据、语义和物理回放约束判定。失败原因保留为元数据。 |
| F15 | `RawReader.__init__()` 禁止 `.tmp_` 路径，诊断工具也因此不能检查失败 Raw。 | Reader 允许诊断未发布路径；导出和物理回放入口负责拒绝未发布数据。诊断不代表可执行。 |
| F16 | `data/export.py` 同时有局部 ±25%、平均 ±5%、readout 小于名义周期、观测不重叠等门禁。 | 保留必要时序关系和局部有界间隔；移除平均 ±5% 与名义周期 readout 门禁，明确 `dt` 是名义周期。 |
| F17 | Canonical 导出失败保留没有续传消费者的完整 staging；Raw writer 的内部数据形状、计数又在多层自证。 | 可再生的导出 staging 失败即清理；不可再生的失败 Raw 保留。边界覆盖确认后删除内部重复校验。 |
| F18 | HOME 目标穿过多层重复传递；总预算叠加速度估算、分段余量和多个常量。 | owner 从配置取得唯一 HOME 目标；arm HOME 使用一个显式总预算。保留路径碰撞检查和实测到位。 |
| F19 | `ipc/ring.py:read()` 在复制竞争时与“无来源数据”都返回 None，上层可能将一次竞争直接判为传感器故障。 | 两次有界复制尝试；失败时可复用同一读取方的上一份完整快照，保留原时间戳，由正常 TTL 判定。无快照则本 tick 无新输入。 |
| F20 | `start_attempted`、内部 phase guard、异常分类帮助函数及配置派生项中有仅维持旧设计的状态。 | recorder 在真正启动点创建并先交给 owner；清除随主体机制失去用途的状态、参数和分支。不要把一个小 enum 换成多个布尔量。 |

F03、F10、F11 是已有具体反例，不只是风格偏好：停止时持邮箱锁的发布者会导致旧命令污染下一轮；16 Hz 下约 63 ms 的动作映射可触发 sync 长期零提交；61 ms 与 125.1 ms 两个样本相隔约 64.1 ms，却可能被名义槽位判成漏槽。

## 3. 唯一所有权与最小不变量

| 事实 | 唯一持有者 / 表达方式 |
| --- | --- |
| 哪个操作正在执行、是否允许开始下一段 | 父进程主线程的顺序流程；可保留已有小型父进程状态 enum 用于交互。 |
| 一个长操作是否被后来的用户取消打断 | 父进程普通整数 `cancel_epoch`，短本地锁保护；操作捕获值，取消时递增。 |
| 设备现在是否允许继续发出目标 | 共享 stop / emergency / closing、owner fault、有效目标租期及当前设备反馈；SDK owner 最终检查。 |
| 最新 arm 目标 | 一个 latest-value mailbox；邮箱锁保护数据完整性，不承担授权代际。 |
| 异步工作是否结束 | 实际 Future、thread、process、Pipe 请求的完成事实。 |
| 当前输入是否可用 | 原始采样/接收时间和消费者的来源 TTL。读取缓存不改变时间。 |
| 录制是否仍接受行 | recorder 的一个接纳状态；与 writer 是否还活着是两个不同事实。 |
| 操作为什么结束、收尾发生了什么 | 当前操作结果中的结束意图、主错误和附加收尾错误；不回写为多处永久设备状态。 |

必须维持以下边界：

1. 只有父进程主线程发布动作和生命周期请求；模型线程只计算，键盘回调只更新短状态与停止信号，不执行 SDK I/O。记录线程不回调运动入口。
2. 同一时刻最多一个生命周期请求及其回复 owner。回复未消费或 owner 未确认停止时不能开始下一操作；无法收尾则退出会话，不自动恢复。
3. 取消后的旧发布调用必须已退出，才能清邮箱并恢复；不能让后台 producer 越过这条边界。owner 在停止态不得消费/安装目标或更新轨迹，防止邮箱残留重新污染 active。
4. SDK 调用前检查当下停止信号、故障、期限和设备反馈；输入映射后的长耗时不能让已过期 VR 输入获得新的目标寿命。
5. 不持有父进程状态锁、键盘锁或代际锁跨越 SDK 调用、等待回复、join 或大数组复制。
6. arm 与 hand 的顺序发送不是原子事务；一次已进入的原生调用不能被 Python 检查撤回。部分发送失败立即置 stop，再由 owner 做停止和收尾。不伪称“绝无最后一帧”或“所有设备同时停止”。

## 4. 实施任务

### T1. 删除跨进程授权代际，建立简单的恢复顺序

涉及：`runtime/state.py`、`robot/commands.py`、`robot/system.py`、`robot/xarm7/{protocol,process,worker}.py` 及所有发送调用方。

- [x] 用一个普通本地 `cancel_epoch` 取代共享 `run_signal`。它只服务父进程的取消观察与恢复检查，不进入 KeyEvent、RobotCommand、共享目标或 Pipe 请求。RobotCommand 回归目标数据；主线程入口/已有取消回调持有取消上下文。
- [x] 普通用户取消在一个短本地锁内更新取消计数、必要意图和 stop。恢复在同一把锁内比较捕获值并清 stop。不要保留“锁外先设置普通 stop，锁内很久以后才递增身份”的丢取消窗口。emergency 可以先置位，且当前会话不自动清除。
- [x] 删除 `GOAL_DTYPE.run_id`、PrepareRequest / HomeRequest 中的运行身份、owner `authorized(run_id)` 中的代际锁和 `generation_busy` 分支，以及 `prepared_run_id` 全部读写。不增加同义 token、lease ID、prepared flag。
- [x] 每次允许新的 prepare / HOME / hand HOME 前，按下面顺序完成恢复；复用现有函数或一个小函数，不造授权框架。

恢复顺序：

1. 关闭 B/H/T/R 等启动命令的接纳；旧控制/发送调用已在主线程返回。
2. 请求停止；消费本操作拥有的旧生命周期回复，等待本次应有的 stopped 确认，并检查 fault / close error。owner 清除 active 与轨迹 reference。
3. stop 仍保持置位时，取得 goal_lock 并显式将 pending 清零。获取失败按现有有界预算退出；不能使用非阻塞 `take_goal()` 代替完成证明，也不持邮箱锁调用 SDK。
4. 在短本地状态锁内比较捕获的 cancel_epoch，检查 emergency / quit / closing；合法才清除 stop。stopped 旧确认由 owner 接纳新生命周期请求时清除，且先于 SDK 工作；父进程不写 stopped（实施依据见第 8 节）。之后的取消必须再次将 stop 置位，不能被这一恢复覆盖。
5. 提交并等待当前 prepare / HOME。prepare 成功后才进入动作循环。长操作期间检查共享取消信号和绝对期限。

- [x] stopped 只表示 owner 已处理停止，且必须结合 fault 判断成功；不表示物理停稳。失效旧确认只发生在接受新工作时，禁止用上一轮的 stopped 跨过新一轮未完成的 SDK 工作。
- [x] 修正 owner 在停止态仍可能进入 `tick()`、先取目标再裁决的行为：停止态不消费邮箱、不赋 active、不推进 reference。`stop()` 清掉 active / reference 后必须一直保持到恢复；仅清 pending 不能消除已被错误重新装入 active 的旧目标。取消发生在 tick 中途时，最终发送检查拒绝发送，随后正常停止处理清除参考。
- [x] hand HOME 等所有 `stop.clear()` 路径都纳入同一恢复规则。内部为准备而调用的 `robot.stop()` 不应被误当作用户取消并递增 cancel_epoch。
- [x] 保留目标 `expires_ns`、goal_lock、stop / emergency / closing、owner 故障通道、当前反馈检查、Mode 1 准备和速度/位置停稳确认。owner servo 前检查仍保留；不再核验“属于哪一轮”。

### T2. 按键接纳与结束意图不再借用运动身份

涉及：`runtime/keyboard.py`、`teleop/loop.py`、`teleop/keyboard.py`、`deployment/session.py`、回放交互入口。

- [x] 删除 `KeyEvent.run_id`、`get_run_id` 及其调用链。busy 时拒绝启动类按键，闲置并且资源收尾完成后才重新接纳。
- [x] 接纳判断、held_keys 更新与事件入队必须在同一 Condition 临界区完成。关闭入口同时清除已排队的启动命令，解决“先取身份、清队列后才入队”的窗口；仅调用 `events.clear()` 不够。
- [x] busy 时也处理按键释放并维护 held_keys；长按 B 穿过收尾不能自动开始下一段。停止、退出和 Esc 按各入口既有键位始终有效；键盘 jog 的 S/D 方向键不能被误当作其他入口的停止/丢弃键。
- [x] 快停信号与对应终止事件的发布必须有一致顺序，不能出现旧 S/D 回调被暂停到下一轮才生效。可在 Condition 内调用只更新短状态的快停回调；此时锁序固定为 Condition → 本地状态锁，禁止反向嵌套。任何 SDK、join、Pipe 等待都在锁外。
- [x] 保存/丢弃意图由主线程处理；开始 writer 收尾前，在同一输入临界区消费已经接纳的终止事件并冻结本段决定。D 后再 Q 仍丢弃；冻结后的 D 不追溯改写已经决定的保存。退出与急停事实仍独立保留。
- [x] 删除 teleop / rollout 收尾以及 tactile reset 等路径中仅为废弃旧 B/H/T 而增加的撤权；真正的停止和取消仍执行一次。
- [x] 保留有实际交互作用的小状态 enum 即可；能由顺序调用证明的内部 phase guard 删除，不把 phase 拆成一组互相约束的布尔量。

### T3. 异常直接传播，资源完成以真实句柄为准

涉及：`runtime/state.py`、`runtime/lifecycle.py`、`robot/system.py`、`deployment/{inference,rollout,session,run}.py`、`teleop/{loop,run}.py`、`replay/{joint_replay,run}.py`。

- [x] 先使 replay 的设备/停止异常真正到达 episode / session 边界；finally 中仍保存本轮结果与实际采样。不能吞成一个 FAULT 枚举后依赖 Robot.stop_failure 在别处重新抛出。
- [x] 删除 `error_state`、`mark_error()`、`Robot.stop_failure` 和同步错误的重复轮询。保留异步 owner fault、writer error、真实资源 pending；它们确有独立的信息来源。
- [x] 异常发生在发送、反馈等副作用边界时立即置 stop；主线程的 episode 边界完成一次停止、分类和收尾。不得为了“统一处理”把置 stop 延后到录制入队或慢清理之后。
- [x] 当前操作开始被接纳时重置结束意图，而不是等到 begin_motion 才重置。删除 `run_ended_id`；合并重复 reason 分类与拷贝。普通结束原因保留 operator / quit / duration / cancelled 等必要语义，真实故障依靠带 stage 的异常信息，不再构建细密枚举树。
- [x] 后续 close 成功不能抹去之前 stop 失败；结果保留主错误和独立 cleanup errors。正常取消没有原始失败时，收尾失败须使会话失败。不要在每个帮助函数捕获 BaseException；只在拥有停止、资源或顶层退出职责的边界处理 KeyboardInterrupt 等异常。
- [x] 模型 Future 用 done / wait 完成事实判断；再调用 result 传播任务自身异常。返回 None 是“任务完成但结果无效”，由预测输出边界报错；任务抛 TimeoutError 是“已完成且失败”，不等于等待超时。保留实际推理 future / close_future 所表达的不同在途操作，不合成一个 busy 布尔量。
- [x] 正常段结束，停止后同步收尾 writer、丢弃并回收上一预测，完成后再接纳 B。失败则退出，不以“跳过坏结果”自动开下一轮。
- [x] 同一个模型任务第一次进入收尾时建立唯一绝对期限；外层 close 使用剩余预算，不能再加一个完整 65 s。writer 使用自己的现有有界预算，不把两个不同资源硬合成一套通用 deadline 对象。
- [x] 已明确退出、急停或致命失败时，先请求停止并 close 机器人，再等待慢 writer / model；此时收尾元数据已冻结，不再读取设备。正常段收尾中才发现致命失败，立即进入退出路径并在等待剩余慢资源前 close 机器人。避免重复实现两套清理流程。
- [x] `Keyboard.stop()` 即使 active 已为 False，仍应处理真实未结束 listener；join 后检查 is_alive，并通过共同关闭结果报告未完成。终端恢复失败与 listener pending 分别报告，句柄不提前清空。
- [x] 资源在 start 尝试前就交给 owner；未成功启动的 process 不盲目 join，已启动但未退出的句柄不丢弃。现有采集进程的有界退出策略可保留，不把其强杀策略复制给硬件 SDK owner。
- [x] `ThreadPoolExecutor.shutdown(wait=False)` 不代表任务已退出或解释器必然立即退出；挂起原生 SDK 也无法凭 Python 标记取消。本次不引入进程隔离平台解决这些问题，结果须明确 pending。

结果结构尽量沿用当前类型，只保留独立事实：一次结束原因、带 stage / type / message 的错误列表、Raw 的路径/发布状态/行数、实际关闭结果。删除重复的 summary.reason、message 镜像和为了旧代际服务的字段；不要为“统一结果”再套多层对象。

### T4. 普通控制尽力推进，严格时间承诺只留给实际需要的模式

涉及：`teleop/loop.py`、`runtime/jog.py`、`teleop/keyboard.py`、`deployment/{rollout,schedule}.py`、`replay/joint_replay.py`、相关配置和 timing 工具。

| 模式 | 最终时间语义 |
| --- | --- |
| VR teleop / jog | 使用最新有效输入；每轮至多一个新命令。计算晚了不补发积压 tick，也不只因名义边界已过就丢弃有效候选。 |
| sync policy | 预测动作按顺序消费，实际提交才推进游标；慢计算降低实际频率，不能造成永远零提交。 |
| async policy | 保留观测锚点与动作时间窗口；提交前按当下时间重新判断动作是否仍属于应执行的槽。过时就跳过/重算，不冒充同步序列或 SDK 故障。 |
| RTC | 保留前缀承诺、预取与交接时点；错过承诺就明确结束，不暗中降级为 async。 |
| 关节 replay | 保留当前按 Raw 标称 control_hz 的既定顺序与时点；缺槽不跳动作或补发追赶。不把它称作对 Raw 实际采样时间的精确复现。 |

- [x] 普通循环采用从本轮开始计时的最小限频：等待 `max(0, period - 本轮已耗时)`。超时直接进入下一次新采样，不在完成耗时计算后额外再睡一个完整周期，不做 catch-up burst。
- [x] 删除 teleop / sync 仅因 `now >= next_step_ns` 而否决候选的逻辑。动作映射后的反馈、停止、来源 TTL 和目标期限检查仍有效；控制参考只在实际提交后 commit，不因被丢候选前移。
- [x] sync 等待预测期间保留实际等待上限，并考虑上一目标剩余租期。接受预测后，映射必须产生提交或明确失败；不能丢掉等待上限后无限重算。无需新增“无进展 watchdog”掩盖错误调度。
- [x] 来源新鲜度在读入时检查；长计算后、发布前检查此次候选真正依赖的控制来源是否仍新鲜。不会因存档专用模态过期否决运动。设备最终反馈由设备边界检查，不多层传递同一个来源截止时间。
- [x] 保留 Raw 的实际 control / observation / source 时间戳。连续 history 使用真正进入窗口的 control_timestamp 间隔；沿用 `gap > 1.5 * dt` 的断档门槛统一 sync / async 的窗口重建。窗口不足时不发起需要完整 history 的预测，不复制帧凑数。RTC 的槽位承诺另外检查。
- [x] 实验 duration 与预测 wait 是不同预算；同时到期按 duration 正常结束。目标 expires_ns 仍约束 SDK 执行；不能用 duration、来源时间或下一 tick 替代。
- [x] 将共享 `ACTION_TIMEOUT_NS` 拆成语义清楚的预测等待和命令租期值，初始均保持 2 s；teleop 的 VR TTL 与目标租期分开命名，初值保持现状。CameraConfig.max_frame_age_s 改为明确参数，默认 `4 / 30` s，不再因 fps 改变而隐式改变容忍年龄。校验有限正数即可，不建预算配置框架。
- [x] replay 的提交前时点检查保留；SDK 返回稍晚于当前槽不应凭一个重复的提交后检查直接判整段失败。若下一动作的时点确已错过，按回放既定规则结束。记录实际开始/完成时间，不伪造准点。

### T5. 录制只约束录制；有效数据不受历史运行结果连坐

涉及：`runtime/observation.py`、采集 worker 健康检查、`teleop/loop.py`、`deployment/{preflight,session,rollout}.py`、`data/{recorder,raw,export}.py`、`replay/trajectory.py`、诊断/可视化入口。

- [x] 明确保留 `control_needs`、已有固定完整记录所需的 `record_needs`，用二者并集 `capture_needs` 决定读取哪些数据。不得只删除 union 而漏采记录字段。
- [x] 控制准入与控制期间 health check 只依赖真正参与控制的服务和设备反馈；只为静态相机参数启动的 worker 取得参数后可关闭。控制需要的点云/相机/触觉不能误归为存档输入。
- [x] 明确请求录制而录制启动失败时，在运动开始前报错；不悄悄以“无录制模式”开始。运行中仅存档输入变旧、缺失或其独立采集 worker 退出，以及 recorder 队列已满，采用下面的前缀冻结规则。
- [x] 前缀冻结只做三件事：保存第一次 `recording_truncated_reason`、停止接纳新行、立即给出可见警告。坏行以及之后的行都不入队；不补 NaN / 假零 / 复制帧，不中途恢复录制。已经接受的队列项仍属于有效前缀。
- [x] 冻结不执行 finish / join / publish；只要控制输入仍有效，当前运动继续。运动结束后按正常保存/丢弃意图统一收尾，不新增后台 finalizer、重启 writer 或自动分段机制。
- [x] 队列满时未入队的行不计入 submitted_frames，不把此质量退化写成 writer I/O 异常。磁盘、编码器、writer 线程自身的真实 I/O 错误仍立即报告并结束会话；与“存档采样跟不上”分开。
- [x] 手部 SDK / CRC / 通信失败不能因触觉只是存档用途就被忽略；只有设备有效反馈中的存档专用通道不满足质量要求时，才允许冻结录制而继续控制。
- [x] 修正 `EpisodeRecorder.finish()` 对 `was_recording` 的依赖：接纳关闭而 writer 仍在时，也必须恰好发送一次 finish 请求并等待真实线程完成。根据 thread / finish_request / error 的实际状态收尾，防止冻结后 writer 永等空队列。
- [x] recorder 在真正 start 的位置创建，并在调用 start 前赋给 owner；删除 `start_attempted` 及仅为提前创建服务的分支。start 中途失败仍能被 finally 持有和关闭。
- [x] Raw 的 `status=published` 只表示容器已正常关闭并发布，不能表示整次控制实验正常。元数据保存运行结束原因和 `recording_truncated_reason`；不再增加同义的 complete / coverage / valid 布尔量。
- [x] 仅录制质量退化、控制正常、全部资源已收尾时允许 CLI 正常退出，但必须报告“录制已截断”和路径/行数；真实运行异常或未完成关闭仍失败退出。不能用保存成功音效/文案把截断描述为完整录制。
- [x] 删除 `validate_completion()` 对 `technical_status == normal` 的要求及其调用；运行尾部故障不能自动禁止有效前缀导出/回放。仍验证发布路径、非空、字段/形状/单位/关节顺序、finite、图像可解码及行数对应关系；物理回放还需验证当前实测起点和关节范围。
- [x] `RawReader` 可读取显式给定的 `.tmp_` 诊断路径。导出与物理回放入口各自拒绝请求路径或解析后实际路径为未发布 staging 的输入，不能通过符号链接绕过。诊断无需默认禁止所有符号链接。
- [x] `--info` 等基础检查尽量能展示可读的 HDF5 元数据和实际文件状态，不要求先通过完整可执行数据契约；完整几何可视化仍验证自己真正需要的字段。禁止自动将 staging 改名为“发布成功”。
- [x] RawWriter 内部重复形状/计数检查仅在确认全部真实入口有覆盖后删除。保留外部 Raw 读取、RGB 解码数量、SDK 输入、数值变换输出等真实边界检查；不能将数据文件视为可信的内部对象。

### T6. 导出时序只保留一个清楚的近似契约

Canonical 继续使用固定 13 个 data 字段与 `root.attrs["dt"] = 1 / control_hz`。本次不加逐行时间数组，不做重采样，不发明第二套严格导出模式。`dt` 明确表示名义周期，Canonical 适合有界抖动的近似等间隔训练，不承诺保留 Raw 的真实时间轴。

- [x] 保留 `control_ns > 0`、`observation_ns >= control_ns`、相邻观测不跨越下一次 control、来源时间有效/不倒退、相机序列不倒退和实际模态新鲜度约束。
- [x] 对相邻 control 间隔保留现有 `abs(actual_dt - nominal_dt) <= nominal_dt / 4`，即 ±25%；同时确保时间递增。该限制拒绝真正漏掉整周期的数据，不允许任意不规则采样。
- [x] 删除全局平均周期 ±5% 拒绝和 `readout >= nominal_dt` 拒绝；在保留“observation 不早于本次 control、且早于下次 control”的前提下，删除冗余的 observation 单调性二次检查。
- [x] 平均实际周期、最大间隔、最大 readout 和网格漂移作为少量质量统计保留在已有 episode_quality 中；保留并传播运行结束和录制截断信息，便于实验者筛选，不再成为隐藏的硬门禁。
- [x] 例：名义 62.5 ms、稳定实际 66 ms 的有效记录允许导出；局部出现约 125 ms 的断档仍拒绝。单行沿用明确警告，不能声称已验证采样周期。一个 Canonical store 仍要求同任务、同名义 dt 和一致的实际数组/相机语义。
- [x] 承认这是一项有意放宽：累计真实时长可与名义 dt 明显不同。需要严格动力学或真实时序的实验使用 Raw 时间戳；不靠悄悄改 dt、插值或伪造时间来“修复”。
- [x] 导出失败时关闭写入资源并删除本次新建的 Canonical staging；成功后仍原子发布。清理错误不得覆盖原始导出失败；已有目标与源 Raw 不删除。失败 Raw staging 继续保留以便诊断/恢复。

### T7. HOME 与快照读取做局部减法

涉及：`runtime/home.py`、`control/homing.py`、`config/xarm7.py`、arm HOME 协议与 `ipc/{ring,sensors}.py` 及读快照的调用方。

- [x] 删除 HOME 目标在父进程 → Robot → ArmProcess → HomeRequest → owner 之间的重复传递；双方使用同一 ArmConfig.home_qpos，owner 检查路径末点是否匹配该目标。路径仍由父进程规划。
- [x] 添加一个明确的 `HomingConfig.timeout_s`，默认 60 s，表示整个 arm HOME 操作的研究运行预算；从该 arm HOME 开始建立一次绝对期限，准备实测起点、规划、owner 停稳、各路点和最终确认共用，不按分段重置。
- [x] 删除 `home_segment_timeout_s()` 及累加估算仅用于预算的代码。独立 hand HOME 保留其现有 2 s 预算；取消后的 SDK stop/close 仍使用现有执行器收尾预算。60 s 不是物理安全保证，可由实验配置直接调整。
- [x] 保留当前路径选择、桌面/自碰撞约束、规划后实际起点再检查、目标与速度收敛和 dwell。SDK wait=False 返回不表示到位。不要无依据地改为直线回零或删掉有作用的候选路径。
- [x] 删除已由计算边界保证的重复 finite 检查；`HomePlanner.table_plane_abcd` 经追踪有真实归档消费者，保留理由见第 8 节。不要全仓机械删除 `isfinite`。
- [x] SharedRing 保留已有平台与单写多读约定，不扩展成跨平台消息总线。读取竞争最多尝试两次；仍失败时读取方可使用上一份相同字段集合的完整快照，必须保留原时间戳/sequence，且不混合不同相机帧或点云关联。
- [x] 没有上一份有效快照时，控制 tick 不发送依赖该来源的命令；准备阶段在现有有界等待内继续。持续读不到或来源确实老化，仍由来源 TTL / 等待预算结束，不能无限给缓存续命。无需新增 read-status 枚举或重试管理器。

## 5. 验证：围绕行为，不围绕已删除的实现

优先复用当前 fake driver、时钟和测试结构。只为并发顺序、资源完成和外部行为补必要回归测试；不为每个字段删除写一个镜像测试。测试不得连接真实设备。以下场景可以合并成少量测试文件，但不可用“全文无旧字段”代替。

| 验证 | 必须观察到的结果 |
| --- | --- |
| V01：暂停发布者于取得邮箱锁/检查后，插入取消与 owner stop；旧目标在 stop 后落入邮箱，穿插 owner tick，再恢复并 prepare 下一轮 | 旧调用先返回，停止态 tick 不重装 active，邮箱确定清空；旧目标不会被执行，也不会触发新一轮 stop。不得依赖跨进程代际。 |
| V02：prepare / HOME 各恢复临界点发生 S/Q/Esc；上一次 Pipe 回复未消费或 owner 未停 | 取消不被清 stop 覆盖；未完成旧操作时不接受新工作；故障不能被普通取消掩盖。 |
| V03：B/H/T 的接纳、busy 关闭与入队交错；长按 B 穿过收尾；D 后 Q；freeze 前后 D | 没有旧启动事件漏到下一轮；保存/丢弃决定符合接受时序；停止始终有效。验证锁序无死锁。 |
| V04：16 Hz sync 的单次映射稳定耗时约 63 ms，无 max_running_s；正常输入与设备 | 有实际命令提交且游标前进，无补发突发、无额外完整周期睡眠；不能继续无限忙等。 |
| V05：async 样本 61 ms 与 125.1 ms；真正超过 1.5dt 的 history 间隔 | 前者不因名义槽跨界清 history，后者重建窗口；RTC 错过承诺仍明确结束。 |
| V06：映射期间 VR 变旧，目标租期尚长；或目标已过期而来源仍新；duration 与 wait 同时到期 | 旧输入不会被新租期洗新，旧目标不会被新来源复活；同时预算到期报告正常 duration。 |
| V07：Future 返回 None、任务自身抛 TimeoutError、真实未完成；writer/listener join 超时 | 完成失败与 pending 区分；句柄保留；同一任务外层 close 不重新等待完整预算。 |
| V08：stop 失败后 close 成功；已决定退出时 writer/model 慢；启动半途失败 | 首个失败不丢；机器人 close 不排在已知慢收尾之后；已启动资源都有人收尾，未启动资源不盲目 join。 |
| V09：存档专用来源过期/worker 退出/队列满，而控制输入正常 | 运动继续；只保留冻结前连续有效行；停止后 finish 确实唤醒 writer，保存有截断原因的前缀。控制所需来源失效仍停止。 |
| V10：同一段有效数据分别写 normal / failed 结束元数据；另有真实损坏、NaN、未发布路径 | 前两者不只因历史状态而区别拒绝；后者仍在对应边界拒绝。诊断可读取 staging，物理回放不可。 |
| V11：62.5 ms 名义周期，稳定 66 ms；局部 125 ms；坏时序/错误 RGB 帧数 | 有界近似数据可导出，真正断档和结构/数值坏数据拒绝；质量统计如实。失败导出清自身 staging，不删 Raw。 |
| V12：HOME 慢规划/多路点/取消；共享快照复制竞争 | HOME 共用一次期限，返回不冒充实测到位；快照最多两次尝试，缓存不更新时间，不混帧。 |

适当验证当前 CLI 入口与配置解析，检查受影响模块可导入；依赖真实 SDK/模型的测试若环境不具备，明确标记未执行。遵守 AGENTS.md，禁止生成 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`。不要为通过测试放宽上述行为，也不要将旧实现断言机械改成新字段断言。

真机验证由实验者在具备设备条件时执行：小幅 teleop/jog、连续两段启停、准备与 HOME 中取消、断开控制来源、录制专用来源失效、sync 慢预测、各模式回放/策略时间行为和最终关闭。提交说明区分假驱动验证与真机验证；本任务不授权执行环境自动驱动机器人。

## 6. 实施顺序与完成标准

按 T1–T3、T4、T5–T6、T7 的依赖顺序推进；同一机制及调用方一次改完整，不留新旧两套协议并行。每一阶段先完成相关行为验证再进入下一阶段，不停留在再次 review 或只提交 TODO。

- [x] 所有当前入口使用同一套新语义；旧运行代际、prepared 标志、错误闩锁、UI 代际和纯兼容分支无运行依赖。搜索旧符号只用于漏项审计，不作为主要正确性证据。
- [x] 保留的状态能逐一解释对应哪个独立事实；没有为替代被删机制新建 Manager、重复 epoch 或更多互斥布尔量。
- [x] 普通控制循环具有进展；严格时间约束有明确的 RTC / replay / async 业务理由。来源 TTL 和目标租期各有唯一含义。
- [x] 关闭结果与实际线程/进程/Future 一致；故障不被丢弃，不因记录或模型慢收尾无故延迟已决定执行的设备 close。
- [x] 数据发布、录制截断、运行失败和数据可用性含义清楚；导出和回放不会凭历史成功标签替代当前校验。
- [x] README、相关配置说明、键位、CLI 退出行为、Raw / Canonical 时间语义与实际代码同步更新。只维护现有文档和必要的本任务书状态，不新建重复架构说明合集。
- [x] 检查 diff，确认保存他人改动、无旧产物兼容层、无无关目录重构、无自动删除已有实验文件。
- [x] 最终交付列出：完成任务、主要删除项、保留机制的理由、实际执行的验证及结果、未执行的真机/环境验证、仍存在的具体限制。任务书勾选只依据实际完成事实。

若当前代码出现与基线不同的真实 producer、外部 API 或消费者，先追踪实际入口，再以更小的实现满足同一不变量。只有与用户目标存在实质冲突时才需要澄清；不能用假想未来扩展恢复被删除的复杂机制。

## 7. 参考项目与采用范围

| 参考基线 | 借鉴点 | 不照搬的部分 |
| --- | --- | --- |
| [LeFranX](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | 直接的控制链、最新目标与超时语义，用小机制解决实际断流。 | 不能据其较少防御分支推断本项目可删除 SDK 前检查或实测到位。 |
| [T-Rex hardware_code](https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code) | 研究代码按业务顺序组织，控制流程容易阅读和改动。 | 不照搬忽略真实 pending、停止完成或旧目标隔离的简化。 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | 事件驱动停止与清楚的运行入口，避免不必要的分布式授权模型。 | 不照搬无界等待或仅清事件便假设旧工作已退出。 |

参考项目证明轻量结构可行，不构成真实设备停止行为的验证。本文最终取舍以 dexmani_real 的实际调用关系、研究定位和上面的可观察行为为依据。


## 8. 实施记录与验证结果（2026-11-03）

实施前工作区干净，HEAD 为 `c7c740e637b046eb6c3c19590d4b2afa00270647`；与审查基线 `e28d397b0fb947e4a4f91007a1b1b167b946816e` 的已提交差异只有本任务书，业务代码一致。按 T1–T3 → T4 → T5–T6 → T7 完成代码、调用方及配置调整，随后交叉核对收尾、数据边界和文档。以上勾选表示本地实现与离线验证完成，不表示真机验收。

主要删除项：跨进程运行代际及锁、`prepared_run_id`、按键运行身份、错误闩锁和停止失败缓存、重复终止分类、普通控制的 tick 过期否决、teleop 软超时状态、录制启动尝试状态、Raw 历史成功门禁、Canonical 平均周期/readout 门禁、HOME 分段预算和跨层目标参数。没有引入迁移器、兼容开关、Manager、后台 finalizer 或自动恢复。

实施中确认的两处偏差：

1. **stopped 由 owner 单独写入。** `ArmController.run()` 将 `stop && !stopped` 视为待处理停止；若父进程先清 stopped、后清 stop，owner 可能再次处理旧停止，并在新 prepare 期间留下过期确认。因此 `ArmProcess.resume()` 只在本地锁内检查取消并清 stop；`ArmController.operation()` 在接受新请求、进入 SDK 前清 stopped。父进程保留真实 `reply_pending`，先消费旧回复，再确认停止、持锁清空邮箱。hand HOME 没有新 arm SDK 工作，复用同一恢复函数，手部发送与停止仍在父线程串行完成。回归验证新 SDK 工作被阻塞时旧确认不能让停止等待成功。
2. **保留实际使用的桌面平面成员。** `PolicySession.get_home_planner()` 读取 `HomePlanner.table_plane_abcd`，随后由 `record_table_plane()` 写入实验 `run_config.yaml`。删除会丢失实际使用的标定记录，或要求重复加载；因此保留该成员。删除的是已由 `CollisionModel.minimum_table_distance()` 和配置校验覆盖的重复 finite 检查。路径候选排序仍使用速度与 dwell 的代价估计，它不是已删除的分段超时预算。

删除 RawWriter 内部重复检查前核对了真实入口：生产调用只有 EpisodeRecorder；teleop/policy 在录制准入检查完整手部、RGB-D 和触觉，采集端及固定 IPC dtype 保证数值形状，录制需求在每次 append 前单独检查。因而移除缺手补 NaN、重复图像形状、内部计数一致性和时间递增自证。teleop 末次映射失败而没有命令时冻结此前前缀，不写入会使整段无法导出的 NaN action。未发生命令提交的 policy 行仍以 NaN 表示动作缺失，保留实际采样事实；物理回放继续拒绝这些行。外部 RawReader、数值转换、RGB 解码数量与设备发送边界检查保留。teleop 的内部准备/活动阶段 guard 仅在核实唯一串行调用链后删除，交互状态和真实资源 pending 保留。

基线没有受版本控制的业务测试目录，本次增加五个针对行为的测试文件；所有设备均为假驱动，测试数据只写 pytest 临时目录。

| 验证 | 离线证据与结果 |
| --- | --- |
| V01 | `tests/test_arm_recovery.py`：受控线程在持邮箱锁时暂停；停止态不安装残留目标；持锁清空后，线程内真实 owner 循环通过真实 Pipe prepare 并执行新目标。通过。 |
| V02 | 同文件：prepare/HOME 在停止确认、邮箱清空后分别遇到 S/Q/Esc；未消费回复拒绝新操作；取消与 owner fault 同时出现仍报告故障；旧 stopped 不能跳过新的阻塞 SDK 工作。通过。 |
| V03 | `tests/test_input_resources.py`：busy 长按/释放、快停回调暂停时的接纳排序、D→Q、finish 前后 D。通过。 |
| V04 | `tests/test_control_progress.py`：16 Hz、63 ms 映射、无时长上限的 sync 连续提交且游标前进；teleop 经真实 episode 编排提交/commit 后退出，jog 慢于 tick 仍提交。通过。 |
| V05 | 同文件：61 ms/125.1 ms 入窗保持连续，实际大于 1.5dt 重建窗口；RTC 错过交接明确拒绝。通过。 |
| V06 | 同文件及 arm 恢复测试：VR 在映射中变旧、jog 映射反馈过期、独立目标租期到期均拒绝；duration/wait 同时到期归 duration。通过。 |
| V07 | 输入资源、控制与录制测试：真实 Future 的 None/任务 TimeoutError/未完成分别处理；模型收尾预算不续期；受控 listener 和真实 writer 线程超时保留 pending。通过。 |
| V08 | 控制、输入资源与录制测试：stop 失败不因 close 成功消失；已知失败先 close 机器人再收尾 writer/model；replay 主错传播且保留停止错误；未启动 process 不 join、writer 部分启动失败有人关闭。通过。 |
| V09 | 控制与录制测试：存档来源缺失、变旧、worker 退出时仍提交控制；冻结后恢复来源也不重新录制；teleop 末次映射失败不把无命令坏行写入前缀；真实 writer 线程队列满冻结 17 行，停止后 finish 唤醒并保存前缀。通过。 |
| V10 | `tests/test_recording_export.py`：实际写读 HDF5/MP4，normal/failed 数据同样通过当前校验并可加载回放；NaN 目标拒绝；staging 可诊断，回放及 staging 符号链接不能越过发布边界。通过。 |
| V11 | 同文件：稳定 66 ms 和不跨次 control 的 64 ms readout 接纳；125 ms 断档、观测重叠、来源时钟/序列倒退、RGB 数量不符拒绝；失败导出关闭 store 并只清本次 staging。通过。 |
| V12 | `tests/test_home_snapshot.py`：planner 构造耗时占同一 60 s 预算，SDK 返回但实测未到位会超时，首路点后取消不发送后续点；共享环最多复制两次，缓存时间/sequence 不续期且不同字段集合不混用。通过。 |

实际执行的检查：

- 环境：`/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`；`python -B -m pytest -p no:cacheprovider -q tests`：**72 passed**。
- `python -B -m ruff check --no-cache --select F dexmani_real scripts tests` 与 `git diff --check`：通过。
- 30 个改动的生产模块可导入；87 个主包/测试 Python 源文件 AST 解析通过，没有实例化硬件驱动。
- collect_teleop/run_policy 的 `--print-config` 输出为有效 JSON，包含显式相机年龄与 HOME 总预算；keyboard_teleop/replay_episode/export_zarr/visualize_episode 的 `--help`：通过。
- 对缺少完整 Raw schema 和 RGB 的合成 `.tmp_` HDF5 实际执行 `visualize_episode.py --info`：成功显示元数据、dataset 和文件状态，没有发布数据。
- 全仓未发现 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`；既有实验、模型和标定文件未修改。

未执行与具体限制：没有连接或驱动机器人，没有真实相机/VR/触觉、完整 checkpoint 推理、真实碰撞规划场景、完整多进程长期压力或桌面交互验收。根目录 pytest 自动收集会涉及第三方 Grounded-Segment-Anything 示例测试，当前缺少 groundingdino/supervision；它们不在本次 72 项业务回归范围内。测试中的几何转换和进程退出状态按场景注入假实现，不能替代实机性能或 SDK 行为证明。

Python 期限仍不能强制结束阻塞 native SDK 或模型线程，超时后保留 pending 并退出会话，不自动恢复；已进入 SDK 的最后一次发送不能撤回。Canonical 仍是名义 dt 下的 ±25% 局部近似契约，不保留真实时间轴，累积漂移可显著；严格时序研究使用 Raw。零行不发布，真实 writer I/O 失败保留 Raw staging，需人工诊断。上述限制没有通过兼容层或后台状态机隐藏。


## 9. 后续全链路审查与清理（2026-11-03）

在上述工作区实现上继续追踪入口、采集、控制、录制、导出和资源关闭的调用链，保留原有改动。本轮直接修正：

- SharedRing 的缓存类型与实际返回值对齐；pointcloud worker 在复制竞争退回旧缓存时不重复计算/发布相同相机序列。来源时间、RGB-D 与点云关联保持不变。
- 导出目录发现复用 `require_published_raw`，在 resolve 前检查请求路径，堵住“staging 父目录中的符号链接指向已发布目录”的遗漏；删除不完整的 basename 重复检查。
- 导出调用方直接持有 DirectoryStore，创建 Zarr 元数据中途失败也能关闭资源并清理自身 staging；正常关闭后不因 publish 失败再次关闭同一 store。真实 Zarr 发布链已离线验证。
- Policy 忽略未绑定按键，防止它们占用 B/H/T 的接纳；Rollout 自行推导 control/record 需求并集，删除重复参数，准备阶段两种准入共用同一快照。
- Rollout 的 robot close 遇到 KeyboardInterrupt 时立即设置 emergency，再收尾 writer/model；普通停止在本地锁内复用 revoke_motion 的 stop 写入，删除重复 set 和无用中间变量。
- 对齐 export/replay CLI 中过时的“必须技术正常结束”说明，清除手部重定向文档里的旧 tick 丢弃和软超时描述，整理改动文件的导入。

相关主包和脚本未发现 SHA-256、哈希迁移或版本兼容分支；没有为了完成清理而删除设备 CRC、来源时间、绝对限位、外部文件结构和数值检查。Canonical 的 format 标签、source_id 是当前数据契约/实验归档字段，未当作冗余完整性校验删除。

验证结果更新：

- 业务回归 **79 passed**；新增覆盖旧缓存去重、未绑定按键、关闭中断的顺序、请求路径 staging 祖先、store 创建失败收尾，以及实际 Zarr 写入/关闭/发布。
- 改动的 Python 文件及五个测试文件通过 Ruff `E4,E7,E9,F,I`；`git diff --check` 通过。未修改无关脚本的既有风格问题。
- Mypy 对 46 个相关文件的现有注解检查通过；27 个运行时/IPC/owner/rollout/recorder/export/replay 文件加 `--check-untyped-defs` 也通过。缓存均放在 `/tmp`。不是全库 strict 类型验收：Raw 和 PolicySession 的分阶段资源 Optional 类型告警在 HEAD 基线也能复现，本轮没有用 ignore、Any 扩散或新增运行时 guard 掩盖这些告警。
- 在 `/tmp` 源码副本使用 `pip wheel --no-deps --no-build-isolation --no-cache-dir` 构建成功，未在仓库生成 build、egg-info 或工具缓存。
- 31 个改动的生产模块导入、改动文件 AST、六个 CLI 配置/帮助入口复核通过；全仓未发现三类禁止的工具缓存，旧参数与旧帮助文案的引用审计无残留。
- 真机、完整模型和长期运行限制沿用第 8 节，没有连接设备。
