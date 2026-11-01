# DexMani Real 第二轮任务书：修正取消与提交时序，删除重复运行机制

日期：2026-11-01。状态：**R0–R5 实施与下述离线验证已完成；真机及真实模型验收待执行**。

目标：面向个人真实机器人研究，让代码轻而薄、职责清楚、行为正确、容易使用和修改。本轮先修正已复现的时序问题，再删除没有独立职责的状态、等待和诊断工作；不建设新的运行平台。

## 1. 基线、适用范围与优先级

审查基线是 [`fe807c3ff7e05dd4fe8c449796da3302e926ad67`](https://github.com/haoyangzhanglab/dexmani_real/commit/fe807c3ff7e05dd4fe8c449796da3302e926ad67)。该提交已经实施第一轮重构。本任务是其后续，不是重新执行第一轮任务。

- 阅读根目录 `AGENTS.md` 和本任务书；按需查阅 [第一轮任务书的基线版本](https://github.com/haoyangzhanglab/dexmani_real/blob/fe807c3ff7e05dd4fe8c449796da3302e926ad67/.codex/tasks/research_runtime_simplification.md)。旧文档中“完成”的记录是历史证据，不能作为本轮通过的证据。
- 本文对模型启动期间的观测轮询、预热历史、传感器 ready、副本状态以及退出顺序给出了新决定；涉及冲突时按本文实施。其他已成立的数据、控制与设备契约继续保留。
- 若实施时 HEAD 已前进，先核实相关调用链与现有改动；已修复的项只验证，不重复改写。保留用户及其他人的工作，不回退整个仓库。
- 范围限于本文列出的机制及其真实调用方、现有文档。无需迁移目录、引入依赖、重写设备 SDK、改动 `dexmani_policy`、修改模型权重或升级数据格式。
- P1：取消通知可被恢复清除、手部指令限速时间基准错误。P2：软超时诊断无界增长及有明确收益的机制简化。

本任务书是可独立实施的规格。历史审查中的临时脚本不是仓库依赖，实施者不应假定它们仍存在；本文件给出重现条件和判据。

## 2. 三个参考项目提供的依据

对照的是以下固定版本的具体路径，不以项目名或代码行数推导正确性，也不要求复刻它们的进程结构。

| 参考项目与版本 | 采用的原则 | 本轮落地 | 不照搬的机制 |
| --- | --- | --- | --- |
| [ManiUniCon `85c6f2e`](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686)，`maniunicon/policies/torch_model.py`、`customize/act_wrapper/chunk_wrapper.py` | 读观测、模型、动作变换、设备边界可以直接追踪 | 启动只做启动所需的工作；实际 episode 才积累真实历史并执行窗口调度 | 通用 SharedStorage 平台、无限 ready 等待，以及整段预测过期后给最后一项重新定时 |
| [T-Rex `ba4047c`](https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code)，`teleop/main_teleop.py`、`arm_hand_control.py`、`data_writer.py` | 最新输入、最近目标、设备 owner 和写盘线程职责直观 | 一份提交参考；启动不空转生成控制目标；writer 错误用一份事实表达 | 无界写盘队列、可能无限等待的收尾，以及无法证明设备真正停止的处理 |
| [Dexora `6f2869d`](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783)，`deploy/dexora_inference_zmq.py` | 直接的观测→推理→执行循环 | 去掉无消费方的观测组装、重复健康遍历和状态转手 | 黑图/默认值掩盖输入失败，以及用动作索引替代有效期和真实停止 |

这些参考支持减少协调层；它们不能证明可以删除本项目的实际停止、输入来源时间、数据完整性或并发一致性保障。本轮的选择还必须由本项目的实际消费者和复现结果支持。

## 3. 已确认的问题与证据边界

| 编号 | 基线位置（均位于 `dexmani_real/`） | 已确认的行为 | 本轮决定 |
| --- | --- | --- | --- |
| F1 / P1 | `runtime/state.py::revoke_motion/request_stop`；`robot/system.py::send_hand_home`；`robot/xarm7/process.py::begin_operation` | 取消先 set stop，恢复在共享代际锁内 clear，随后取消才更新代际；取消返回后 stop 仍可能为 false | 用现有共享代际锁串行化“代际更新＋最终 set”和“核对代际＋clear” |
| F2 / P1 | `teleop/loop.py::execute_control_step`；`teleop/vr.py::commit`；`teleop/retargeting/adaptive.py` | 成功提交后仍把 compute_start 当作上次提交时间，求解耗时抖动会使连续已提交目标的差分速率超过配置值 | 以正常提交返回时刻推进唯一命令参考 |
| F3 / P2 | `teleop/loop.py::record_control_timing` 及软超时分支 | 每次软超时复制诊断并追加 `termination_details`，无 recorder 时也不断积累 | 软超时只保留固定大小摘要，终止失败按需记录详情 |
| S1 | `deployment/session.py::wait_model/warmup_policy` | 模型启动等待期间按控制周期复制观测；预热还等待真实历史节拍 | 删除等待期间的观测调度；一个有效快照构造预热输入 |
| S2 | `deployment/rollout.py` | `preparing_epoch` 与 `run_id` 轮流保存同一个身份，还兼任“是否开始”判断 | 一个不可变 episode epoch；开始事实由实际开始时间表达 |
| S3 | `runtime/state.py`、`data/recorder.py` | `SessionState.running` 与退出控制流重复；`cancelled` 仅映射 recorder.error，没人 wait 它 | 删除两个副本及全部相关分支 |
| S4 | `deployment/session.py::close`、`runtime/lifecycle.py::close_runtime` | 全局收尾重复撤权和停止职责，理解退出顺序需要跨层拼接 | runtime 收尾一个入口，模型资源继续归模型 owner |
| S5 | `ipc/sensors.py`、sensor/perception workers、启动入口 | 三个传感器 ready Event 与首个有效样本的发布重复 | 复用 ring 的首个完整发布；进程存活与运行中新鲜度仍分别检查 |
| S6 | `teleop/keyboard.py`、`runtime/jog.py` | 无操作者运动输入也先 prepare 并进入持续 hold | 第一次实际 jog 输入时才 prepare |
| S7 | `robot/system.py`、`deployment/rollout.py` 等 | 同一读观测调用链重复遍历键盘/传感进程健康 | service 负责外部服务检查；缓存读取不再遍历所有服务 |
| S8 | `perception/worker.py`、`teleop/run.py`、`ipc/sensors.py` | 点云前后重复去重；VR 标定同时以 ring 序列和严格递增来源时间去重 | 删除推导冗余，保留实际来源及 IPC 序列各自职责 |

F1 的确定性重现使用生产方法、真实 `multiprocessing.Value`/`RLock`，在合法调度点暂停线程：HOME 持有代际锁，取消已发 stop，HOME 清 stop 并提交一次旧代际手部目标，取消随后完成代际 1→2，最终 stop=false。这证明的是**并发取消通知丢失**；不代表旧代际能永久复活，也不代表取消完成后新发起的旧代际校验仍会通过。

F2 的数值反例：`v_max=4 rad/s`，两次计算起点相隔 62.5 ms，求解耗时依次为 25 ms、1 ms；两次提交相隔 38.5 ms，但可产生 0.25 rad 的目标差，差分约为 6.49 rad/s。该反例使用原有限速/滤波代码和受边界约束的替代优化器，证明时间口径错误，不是实际电机速度测量。

F3 的极端场景是来源一直新鲜、每个候选都错过提交窗口、尚无已激活的目标且不录制；没有目标到期或录制时长必然替它结束，诊断列表可随 tick 数增长。

上述是基线缺陷的证据。新实现仍须重新验收；源码审查、数值复现、fake SDK、真实设备和真实模型的验证应分别报告。

## 4. 收敛后的职责与时间模型

| 事实或操作 | 唯一权威来源 / 所有者 | 不再另存的副本 |
| --- | --- | --- |
| 当前运动代际 | `SessionState.run_signal`，父进程修改，设备 owner 读取 | 不新增 revoke id、ACK id、stream lease |
| 尽快通知停止 | 现有 stop Event；emergency 保留独立急停含义 | 不新增心跳、取消队列、恢复握手 |
| 父进程阶段及用户意图 | 本地 `motion_lock` 下的 safety、quit、error、estop 和现有结束原因 | 删除仅表示“尚未退出函数”的 `running` |
| 一段 rollout 身份 | 创建时固定的 episode epoch | 不在准备/执行之间转移或清空身份 |
| 手部滤波、warm start、限速参考 | 上一次正常提交的 qpos 及其提交返回时间；首次用有效测量初始化 | 不新增第二个速度参考或平滑状态 |
| sensor 首次初始化完成 | 对应 ring 已完整发布首个有效样本 | 删除三个 sensor ready Event |
| sensor 当前可用 | 进程状态＋真实消费点的新鲜度检查 | 首帧发布不能充当持续存活或新鲜度证明 |
| writer 失败 / 是否仍在运行 | 首个 `error` / 实际 thread 状态 | 删除 `cancelled`；不缓存第二份 pending |
| 一次模型任务 | 现有单 worker、单 Future owner | 不新增启动任务调度器或恢复服务 |
| 最终 runtime 收尾 | `close_runtime` 组织 robot、keyboard、sensor 的关闭 | PolicySession 不再重复组织相同的全局撤权/停止 |

运行内时间仍使用同机 `monotonic_ns`；UTC 只用于结果记录。来源采集/接收时间、观测组装时间、计算时间、提交时间各有事实含义，不合并为“统一 timestamp”。`expires_ns` 仍是目标有效期，来源 freshness 仍是输入准入；不将两者再串成跨层 deadline 链。

## 5. 具体实施规格

### 5.1 F1：取消与恢复在同一现有锁上确定先后

修改 `SessionState.revoke_motion`：在本地 `motion_lock` 内，显式获取现有 `run_signal.get_lock()`；读取 raw value，检查 `expected_run_id`，保存原 run 的结束原因，更新代际，**在释放该共享锁前最终 set stop**，完成本地阶段变更。不要再用隐含分离读写的 `run_signal.value += 1` 表达整个转换。

实现结构应等价于以下伪代码；沿用现有 reason/estop 优先级，不照抄成新抽象层：

```python
with motion_lock:
    with run_signal.get_lock():
        old_epoch = run_signal.get_obj().value
        if expected_run_id is not None and expected_run_id != old_epoch:
            return
        # 按现有规则保存 old_epoch 的首次运行结束原因。
        run_signal.get_obj().value = old_epoch + 1
        stop_signal.set()  # 撤权完成前的最终写入，与 resume clear 串行。
        safety_state = new_state
```

必须同时满足：

1. 保留按键取消/退出/急停入口在等待本地锁前的快速通知。快速 set 用于尽早停止；共享锁内的最终 set 保证已完成撤权不会被重叠的旧恢复抹掉。只给 clear 增加本地锁不足以解决问题。
2. 两处恢复仍在同一共享代际锁内执行“核对代际与 emergency → clear stop”。撤权先获得锁，旧恢复就失败；恢复先获得锁，之后撤权最终留下 stop=true。不能把 clear 移到核对之外。
3. `expected_run_id` 不匹配的旧 episode 收尾不得改变新 run 的代际、stop 或结束原因。此类清理不能先无条件发快速 stop；操作者的主动 stop 不受此限制。
4. 同时需要两把锁时顺序固定为本地 motion lock → 共享 generation lock。共享锁中只操作小量状态；SDK、Pipe、等待 stopped、模型和 writer 操作一律不在其中。
5. 审计所有 stop.clear 调用。恢复只能发生于现有明确准备/HOME 路径；故障或迟到结果不得隐式恢复。`mark_error`、quit、estop 的退出与停止职责不能因状态删除而消失。
6. 保持 arm owner 的顺序：先向 SDK 发真实停止，再尝试清目标/状态等可能竞争的共享资源。代际锁繁忙不能成为执行实际 stop 的前置条件。

边界说明：撤权的线性化点位于上述共享锁内的状态转换；快速通知是提前提示。已经越过最后校验、与取消重叠的 SDK 调用可能完成，不能许诺 Python 原子中断硬件调用。要求是撤权返回后旧代际不能重新获得授权，且 stop 保持设置直到合法的新代际显式准备。不要为消除所有重叠窗口增加 ACK 协议或把 SDK 放进共享锁。

### 5.2 F2：手部参考使用实际提交返回时间

- `compute_target` 继续接收本 tick 的 `compute_start_ns`，用于计算当前候选。
- `robot.send_action` 正常返回后，用已经采集的 `submit_end_ns` 调用 controller/retargeter 的 commit。变量或参数名应准确表达参考时间，避免仍称其为 source time；不必为命名进行全仓重构。
- 仍只有最后提交的手关节、EEF/机械臂参考及其时间一份状态。超时丢弃、求解拒绝、取消、异常返回或部分提交失败均不推进参考；失败后的设备停止沿用现有边界。
- 取消 `dt == 0` 时伪造一个名义控制周期的 fallback：时间相等时返回原手部参考，不凭空获得运动额度；时间倒退继续明确失败。若 `control_hz` 构造参数仅服务于该 fallback，一并删除该参数、`nominal_dt_s` 及调用方传参。
- 首次 reset 仍用有效测量及其观测时刻建立参考。保留现有标定、目标函数、关节限位、阻尼、alpha 补偿和滤波数学；本轮只修正时间语义和零间隔特例。
- `docs/hand_retargeting.md` 明确：这是已提交目标序列的变化速率限制，SDK 正常返回不等于物理到位或控制器执行 ACK。

正确性依据：单一控制线程中，上次提交返回 `b_prev` ≤ 本次计算起点 `c_now` ≤ 本次设备提交时刻。用 `dt=c_now-b_prev` 限制目标差，比两次实际设备提交的间隔保守。需要验证可观察提交边界的差分不超过配置值，不能只断言求解器使用的 dt 内部自洽。

### 5.3 F3：软超时诊断固定大小

保持“错过提交窗口就丢弃候选，下一个未来 tick 读最新输入”的行为。软超时不是终止错误，也不能更新提交参考。

- 每段最多保留一个软超时计数和一个最后发生的轻量摘要；明确统计观测复制错过窗口和求解错过窗口两条路径，每次只计一次。摘要只含现有时刻/阶段等标量，不复制 solver 诊断字典或观测数组。
- 不录制时不持久积累逐次诊断。可在现有 episode 结束日志输出一次摘要，不新增计时器、日志线程、遥测系统、环形诊断缓存或配置项。
- 有 recorder 时，结束时向既有 details 写入一次汇总；发生真正终止错误时再编码所需的详细 timing/solver 信息。保留原始错误、额外清理错误和操作者保存/丢弃意图。
- 每段初始化/结束正确重置摘要，不能串段。已有非重复的锚点元数据可以保留。
- 没有提交的 Raw 行仍没有动作，不能用旧目标或候选填充成“成功动作”。保留已有失败行和技术状态的语义，不引入逐动作回执系统。

### 5.4 S1：启动阶段不模拟一个控制回路

`wait_model` 只负责：处理停止/退出/急停，检查必须存活的服务和实际设备错误，轮询唯一 Future，执行原有模型启动超时。删除 `next_check_ns`、控制周期以及反复 `read_observation/check_observation` 的调度；不要把这些工作搬进新的 helper 或后台线程。

`warmup_policy` 的最终流程：

1. 等待模型加载完成。
2. 服务必要反馈；读取并校验一个满足实际 `needs` 的快照，投影一次。
3. 用该快照构造长度为 `n_obs_steps` 的预热输入，提交一次现有 `warmup(samples=1, ...)`，等待完成。
4. 真正开始 episode 时按原有准入重新检查；真实预测历史依旧按真实控制节拍采集。

删除预热专用 `rows` 采样循环、`next_sample_ns`、history deadline 和无意义节拍等待。预热快照不写 Raw、不混入 rollout.history，也不伪造来源时间。

投影对象的 `features` 是可变缓存：同一 worker 独占的重复引用可以复用，但必须核实 `ObservationBuilder.build` 和上游 warmup 对别名的行为；若需要独立容器，仅复制该缓存容器，不无条件复制多份 RGB-D/点云。不得让预热缓存被主线程并发修改。

继续使用上游已有 warmup API 及 RNG 恢复。基线 `samples=1` 对普通路径仍有 2 次 predict、RTC 路径共 4 次；这是上游真实调用行为，不得写成只预测 1 次。不得把“重复一个预热快照”推广到实际推理观测历史。活动 rollout 等待推理时的输入健康/新鲜度检查不在此次删除范围。

### 5.5 S2/S3：删除能由真实事实推导的状态

**Rollout 身份。** 创建时保存一个不可变 epoch，并贯穿 prepare、运行、取消、迟到 Future 回收和 finally。删除准备结束时 `preparing_epoch → run_id` 的交接及结束时清空身份的代码。

- 活动 episode 的流程判断统一使用已有 `started_ns`（未开始为 0）；结果中的 `started_utc` 在同一个实际开始边界填写，继续用于持久化报告和跨 episode 统计。不要新增 `started` 布尔量，也不要用“epoch 非空”推断已开始。
- `preparation_valid` 仍要求该 epoch 当前有效且阶段为 ARMED；正常运动仍要求 RUNNING。不要因为身份一直存在而绕过最终 begin_motion。
- duration、错误阶段、`start_cancelled/preparation_failed`、已开始 episode 计数都按真实开始事实处理。
- 结束仍以本段 epoch 调用条件撤权；未回收 Future 的所有权不能因状态简化丢失。Session 用于排队 B 请求的 `start_epoch` 有独立用途，不机械删除。

**SessionState.running。** 删除字段、`end_session` 及全部读取；循环用已有 quit/estop/error、episode 完成状态或异常控制流退出。原来 `end_session(reason=...)` 中真实的撤权与结束原因必须在对应错误/关闭边界保留；不能让所有错误都变成 runtime_shutdown。`SensorChannels.is_running` 是控制子进程退出的实际信号，保留。

**EpisodeRecorder.cancelled。** 删除 Event 和 set/is_set 分支，直接使用已有 `error is not None`。它没有独立取消语义，也没有 wait 消费者。保留 writer.ready（初始化成功或失败时唤醒等待者）、有界队列、finish_request、首个错误、实际线程 pending、唯一 expected_frames 验证和 staging 发布规则；不引入新锁或取消枚举替代被删 Event。

### 5.6 S4：全局退出顺序直接可读

最终 runtime 退出统一由 `close_runtime` 处理：先通知 stop、撤权至 DISARMED，再尝试 robot.close、keyboard.stop、sensor.shutdown。每一步失败仍尝试后续资源，保留首个失败及补充关闭错误；删除最后无独立职责的 end_session。

`PolicySession.close` 先调用上述 runtime 收尾，再安排并等待 model.close；整段仍使用已有全局关闭 deadline。去掉它自己重复的全局 revoke/robot.stop 分支。模型任务只使用已复制输入，runtime 关闭不能与模型线程发生 SDK 所有权交叉。即使 runtime 关闭失败，也必须继续尝试模型关闭并汇总其结果。

重要边界：

- robot.close 仍须先触发实际 hand stop / arm owner stop，再断开；不能仅 set Event 或修改结果为 closed。最终退出时的 passive hand stop 与运行中停止保持测量姿态的职责不混淆。
- episode 结束/运行故障处的立即 stop 继续保留，必须发生在 writer flush 或 pending prediction 回收等待之前。这与最终设备断开不是同一生命周期事件，不为追求“整个程序只调用一次 stop”而删除。
- sensor 短时退出预算与模型/writer 长预算继续分离；pending 真实报告，不把超时写成 completed。关闭过程中不无限等待模型线程，也不能宣称 Python 能强制终止它。
- 部分初始化失败、未连接设备和未创建模型按现有实际资源句柄清理；不创建通用 ShutdownManager、资源注册表或补偿事务。
- 如重复附加 `cleanup_errors` 的代码妨碍此次收尾修改，可在现有 `utils/log.py` 提取一个小函数；仅封装已有追加规则，保留主异常、cause 和输出结构，不引入异常类层级。

### 5.7 S5：传感器用首个完整发布表达初始化

删除 `SensorChannels.vr_ready/camera_ready/pointcloud_ready`，以及各 worker、检查和 shutdown 中的 set/clear。用相应 `ring.latest_sequence > 0` 表达“本次进程已发布至少一个有效样本”：该值只在 SharedRing 完整写入后发布；新启动仍使用新建 channels，不自动重启并复用旧 ring。

- `SensorWorkers.start` 直接接收对应 ring，有限等待首个发布；`started` 只保存实际 process，不保存重复 ready 对象。沿用现有超时常量和取消检查，不新增通用 readiness 框架。
- VR 的交互启动等待保留原有按键/设备服务，并改读首个发布；相机与点云同步更新调用方。不得为了删除 Event 引入忙等，沿用短时轮询/休眠。
- 相机 static metadata 必须在首帧发布之前全部写完；`snapshot_camera_parameters` 以首个发布为初始化前提，仍校验 serial、内参、尺寸和外参要求。
- 首个发布只证明初始化。启动返回前仍确认进程未退出；实际控制、标定和 warmup 消费点继续按来源时间检查新鲜度。旧样本尚在 ring 内不等于 worker 活着；“曾 ready”也不等于“现在 fresh”。
- 保留 SharedRing 的奇偶写标记、复制后复核和最新数据语义；首帧条件不替代 read 的一致性检查。平台前提仍是现有 Linux x86_64 单写多读实现，不宣称新增跨平台无锁保证。
- arm ready、arm stopped、recorder.ready 分别承载真实初始化/停止/错误唤醒，不属于此次删除对象。不要把所有 Event 一概删掉。

### 5.8 S6/S7/S8：删除启动空转和热路径冗余

**键盘 jog。** 删除启动时无条件 `jog.prepare`。连接、检查并 arm 后保持 ARMED；第一次真实运动键输入由已有 `CartesianJog.step` 的懒准备分支执行。准备单独占一轮，下一轮重新读取按键和反馈，不发送准备前的输入。正常运行后松键的制动/hold、拒绝目标后的重新按键要求和 HOME 保持原语义；同步修改提示和注释。

**健康检查。** `service_feedback/service_idle` 继续负责本次服务轮次的键盘/必要进程健康和设备故障；`Robot.read_state` 负责 owner、连接状态及缓存读取，不再调用会遍历所有外部服务的 `Robot.check`。删除 `Rollout.run` 中紧邻 service 的重复 arm.check_error 等确证冗余调用。

必须逐一核对 teleop、jog、replay、HOME、初始化和策略入口：都要在合理轮次执行 service；阻塞 I/O 之后、实际发送边界、`ArmProcess.snapshot` 的设备错误检查有不同时间/副作用职责，可继续保留。取消异常不能掩盖同一时刻的 SDK/worker 故障。不新增“本轮已校验”布尔量、health epoch、valid token 或检查缓存；不能机械规定所有路径只准调用一次 check。

**序列去重。** 点云 worker 保留复制前 `latest_sequence <= last_camera_sequence` 的检查，避免重复复制大帧；在单写、序列只增且不重置的契约下，删除 read 后第二次相同序列判断。VR 标定用严格递增 `recv_ts_ns` 加 freshness 去重，删除局部 `last_sequence`；全仓确认无其他消费者后，删除 `read_vr_frame` 返回的业务字典 `ring_sequence`。保留 SDK `sequence_id`、相机序列、点云 source_camera_sequence 和 SharedRing 内部序列；它们承担不同的来源或一致性职责。

## 6. 明确保留或推迟的事项

| 事项 | 本轮决定与理由 |
| --- | --- |
| 命令有效期、新输入检查、反馈时间 | 保留单个目标 `expires_ns` 及真正消费点的来源 freshness。不得回加 source deadline 传播、心跳或第二套租约 |
| 连续目标跨过上一目标的 expires | 不据此新增 stream-wide lease。要区分 owner 尚未执行到期撤权与已经撤权；新目标仍按现有代际/stop/自身有效期决定可否接受 |
| arm owner 和低频 RPC | 保留唯一回复读取者、实际停止、prepared、本地 owner 状态及必要 stopped/故障报告；不重写 mailbox，不合并高频目标与低频事务为通用消息总线 |
| 视觉包 | 保留点云和按需 RGB-D 同源、自包含、最新快照。不要恢复历史回查，也不要无条件携带全部模态 |
| sync / async / RTC | 保留三种已定义模式、单 Future owner、RTC 策略空间前缀和每槽实际映射；不重写调度算法，不恢复 sync_row 或未来 RobotCommand 冻结 |
| HOME | 保留实际路径检查、必要到位和速度收敛、每段停止责任；不恢复重复 dwell。将规划拒绝改为可交互重试涉及用户行为，暂不混入本轮，更不能统一吞掉 TimeoutError/SDK 错误 |
| Canonical 时间网格 | 暂保留现有全段锚定规则。10 Hz、50 行、每行间隔 100.8 ms 的例子会因累计相位偏移得到约 32 行有效前缀；这揭示保守取舍，不足以证明改为相邻间隔校验不影响训练语义 |
| 数据格式和来源 | 保留 Raw 各来源时间、source_id、固定 13 维 Canonical、recipe/URDF/provenance、policy Raw 默认不能作为示教训练数据、writer 关闭及行数一致性 |
| replay | 保留初始实测姿态＋完整动作序列、名义重放节拍与当前设备准入，不恢复历史传感时间 admission |
| 反馈频率 | 本轮不调 25 Hz 等设备采样周期。观察到约一个采样周期的完成检测延迟不等于需要新调度器 |
| 其他小型状态或静态配置检查 | 仅在修改处证明无消费者才删；例如模型 operation 标签还有错误定位用途，不因“看起来多一个字段”机械清除 |

若后续单独研究 Canonical 时间规则，应先对真实录制数据统计全段相位、相邻间隔及长度为 n_obs/action horizon 的局部窗口误差，明确模型训练/部署采样契约后再修改。该研究不阻塞本轮，也不能在本轮悄悄放宽阈值使导出“通过”。

## 7. 实施顺序与文件范围

所有路径前缀默认为 `dexmani_real/`。同一文件可在不同阶段原地修改，不为分阶段建立兼容实现。

| 阶段 | 主要文件 | 完成条件 |
| --- | --- | --- |
| R0 基线与复现 | 根 `AGENTS.md`；相关当前调用链；仓库外临时验证 | 核对 HEAD 和工作区；复现/确认 F1、F2、F3；记录真实工具/依赖条件 |
| R1 正确性修正 | `runtime/state.py`、`robot/system.py`、`robot/xarm7/process.py`、`teleop/loop.py`、`teleop/vr.py`、`teleop/retargeting/adaptive.py`、`teleop/run.py` | F1–F3 对应验收通过，实际 stop 与提交参考语义保持 |
| R2 状态与收尾 | `deployment/rollout.py`、`deployment/session.py`、`runtime/state.py`、`runtime/lifecycle.py`、`data/recorder.py` 及所有 running/end_session 调用方 | 单 episode 身份、无 running/cancelled 副本、退出顺序和结果无退化 |
| R3 启动简化 | `deployment/session.py`、`ipc/sensors.py`、`runtime/lifecycle.py`、`sensor/realsense.py`、`sensor/vr.py`、`perception/worker.py`、`teleop/run.py`、`teleop/keyboard.py` | 无启动观测节拍/预热等待历史/三个 ready Event/无输入 prepare |
| R4 热路径整理 | `robot/system.py`、`deployment/rollout.py`、`perception/worker.py`、`teleop/run.py`、`ipc/sensors.py` 及必要调用方 | 健康检查职责明确，冗余序列判断和无消费者字段删除 |
| R5 文档与整体验收 | 现有 `README.md`、`docs/hand_retargeting.md`、本任务书的实施记录 | 行为与文档一致；逐项记录真实证据、剩余限制和最终 diff |

先让 R1 的必要回归成立，再进行状态删除；R2–R4 可按文件依赖调整局部顺序。完成一组后做对应验证，最后做一次跨入口检查，不反复跑无关全仓测试。不设代码行数 KPI，但最终必须列出实际删除的机制，而非只把相同复杂度搬到新文件。

## 8. 验收矩阵

### 8.1 需要确定性验证的行为

优先调用真实生产方法，只替代 SDK、模型等外部副作用。并发用可控事件/屏障制造顺序，使用真实 multiprocessing Value/锁；不靠偶然 sleep 压测宣称竞态消失。若依赖限制只能抽取方法验证，必须标明未覆盖的 import/集成路径。

| 编号 | 场景 | 必须观察到的结果 |
| --- | --- | --- |
| A1 | 恢复持代际锁，取消已经快速 set、随后等待共享锁 | 允许重叠的已开始提交；取消完成后代际变化且 stop=true，旧恢复不能再次 clear |
| A2 | 取消先完成，再以旧 epoch 调用 hand HOME / arm begin_operation | 两条路径都拒绝，stop 不被清除；不发新的旧 epoch 目标 |
| A3 | 新 epoch 显式准备；以及旧 episode 的 expected_run_id 收尾迟到 | 新准备能合法恢复；迟到旧清理不取消新 run；quit/estop/error 仍禁止不合规恢复 |
| A4 | 分别持有 generation、goal、status 锁；同时请求停止 | fake SDK stop 调用发生在等待这些共享锁前；保留原有有限等待/非阻塞清理，不把真假停机混为一谈 |
| A5 | request_stop / quit / estop、普通到期、部分 SDK 提交失败 | 撤权与既有结束原因正确，失败透明；无法确认的停机不写成成功，不自动续跑 |
| B1 | 两次计算起点间隔 62.5 ms、耗时 25 ms 与 1 ms，v_max=4 | 以实际提交边界算差分 ≤4 rad/s（仅舍入容差）；原 6.49 反例被消除 |
| B2 | 不同滤波 tau、接近关节限位、重复 VR 帧 | 目标差、限位及既有收敛行为成立；输入没更新不代表控制参考被冻结 |
| B3 | compute 后迟到/拒绝/取消/发送异常/部分失败 | 手/臂/EEF/时间参考均不错误推进；下一次合法计算仍基于最后正常提交 |
| B4 | reset 后首次计算、时间相等、时间倒退 | 首次按实测参考；相等时手目标不变且无除零/名义时间额度；倒退明确失败 |
| C1 | 有/无 recorder 连续制造至少 10,000 次软超时，两种晚到分支 | 常驻诊断条目数与 tick 总数无关；计数准确，且不逐次复制 solver 字典/数组 |
| C2 | 软超时后成功、终止异常、开新段 | 汇总至多一份，真实终止详情保留；Raw 无伪造动作，摘要不串段 |
| D1 | slow load / slow warmup、正常完成、异常、取消、超时 | wait_model 不调用全量观测组装；必要服务错误和用户停止仍及时处理，Future 唯一所有权保留 |
| D2 | n_obs_steps=1 和 >1；普通与 RTC warmup | 只采样/投影一次有效快照，形状正确，无真实节拍等待；缓存别名无并发修改，无 Raw/历史污染 |
| D3 | 预热后开始真实 episode | 重新 admission，真实 n_obs_steps 历史及首预测 reset 保持；普通/RTC warmup 调用次数及 RNG 行为如实验证 |
| E1 | prepare 前取消、writer start 中取消、开始后取消、pending Future 迟到成功/失败 | 单 epoch 不变；未开始 duration=0/分类正确，旧 Future 不可作为新段结果，迟到异常可见 |
| E2 | 正常 quit、故障退出、部分初始化、robot close 失败同时 model close 失败 | 各已创建资源都被尝试关闭；实际 stop 先于阻塞模型/writer 回收；主因与补充错误都保留 |
| E3 | writer open/append/finish/视频关闭/HDF5 关闭失败，队列满，writer pending | 去掉 cancelled 后结果与错误语义不变；未确认关闭不发布，expected_frames 仍核对，无无限 join |
| F1 | 三种 sensor 未发首帧、首帧有效、启动前退出、发帧后退出、取消、超时 | 有限等待，生命周期检查有效；首帧后退出不能仅凭旧序列通过启动/控制 |
| F2 | 相机 static-only、点云＋按需图像；消费者复制竞争/来源过期 | metadata 写入顺序及视图一致性保持；旧来源不会被首帧条件重新认定 fresh；不恢复历史依赖 |
| F3 | 键盘启动后无输入，首次按键，准备中松键，运行中松键，HOME 后重新按键 | 无输入不 prepare/不进入持续 hold；首次准备后重读；正常制动与重新准备语义正确 |
| G1 | teleop / policy / jog / replay / HOME / 初始化读取路径 | 缓存读取不重复遍历所有服务；必要 service 调用存在，真实 SDK/worker 故障不会因取消被掩盖 |
| G2 | 点云反复读同帧、VR 相同 stamp/倒退 stamp/新 stamp | 点云不重复处理；VR 只累计来源时间递增且新鲜的样本；有效来源序列仍保留 |

### 8.2 相关回归与检查方法

- 对触及的三模式流程验证：sync 不重用历史行作为最新观测，async 保留可用尾部，RTC 前缀与交接边界正确；到期、不完整预测、迟到异常仍由现有 owner 处理。无需重造调度测试框架。
- 运行必要语法/导入、CLI help、受影响静态检查及 diff 检查。尽量使用项目支持的 Python/NumPy 和已有环境；不得把缺依赖导致的未执行记为通过，也不得拿 fake 替代真实 HDF5/AV/Zarr 后声称完成格式集成验收。
- 正确性修复需要有针对性的回归；纯删除无消费者字段不必为字段存在与否写镜像测试。临时验证放仓库外或 `/tmp`，不恢复根 tests 树，不提交一次性 probe/测试基础设施。
- 如运行 pytest/ruff/mypy，按 AGENTS 禁止仓库内缓存；例如 pytest 禁用 cacheprovider、ruff 使用 `--no-cache`，mypy 缓存指向仓库外。不要增加忽略规则掩盖产物。
- 记录实际命令、环境、关键输入与输出、失败和未执行项；测试数量不是覆盖面的替代。已有静态检查基线问题应区分新增与存量，不顺手全仓重排或压制警告。
- 不为本轮改变 Raw/Canonical schema。涉及 recorder 生命周期时，应在依赖具备的环境用真实 HDF5/视频关闭与一次 Raw 导出验证；依赖缺失则完成可执行的离线部分并准确留下待验项。

### 8.3 真机与真实模型待验边界

离线通过后仍需在实际实验环境确认取消/急停与模式恢复、手目标发送时序和实际跟踪、HOME、三种策略模式、writer 收尾及真实模型预热开销。源码或 fake SDK 不证明实际 stop 延迟、电机限速或 GPU 吞吐。没有真机/权重时不操作未获准的设备，不编造结果；这不妨碍完成离线实现和文档。

## 9. 完成定义与提交要求

- [x] F1 采用同一共享锁串行化最终 set 与 clear，未引入第二套协议；生产 stop 方法的 SDK 调用顺序以替代 SDK 验证，物理停止待真机。
- [x] F2 使用 submit_end 作为已提交参考，零间隔不再获得虚构运动额度，未发送候选不推进状态。
- [x] F3 的软超时诊断为固定大小，并保留真实终止原因和必要失败详情。
- [x] 不再有模型启动观测时钟、真实节拍预热采样、双 episode 身份、SessionState.running/end_session、recorder.cancelled。
- [x] 不再有三个 sensor ready Event、键盘无输入 prepare、列明的重复健康遍历和序列去重。
- [x] 退出顺序可直接追踪；离线验证覆盖停止先于长时间回收、错误、pending、部分初始化及发布语义，物理行为待真机。
- [x] 第 6 节明确保留的控制/数据契约未被顺手改写；没有新增通用 manager、heartbeat、lease、ACK 或配置开关。
- [x] 相关调用方、导入、提示和现有文档同步；仓库内没有新增临时 probe、测试框架或工具缓存。
- [x] 验证证据填写在下表；未执行项明确标注，未达到真机验收不得写“全部验收通过”。
- [x] 最终报告包含实际删除的机制、可感知行为变化、命令与结果、剩余限制和提交/工作区状态。

按完整职责提交可审查的小批次改动；不需每改一个函数单独提交，也不得重写用户历史。本轮任务文件之外，不新建平行设计书或“最终报告”文件；实施记录就地补充本表即可。第一轮任务书保留历史，不重写其中证据来制造本轮已完成的印象。

| 阶段 | 状态 | 实施后填写：变更 / 验证命令与关键结果 / 限制 |
| --- | --- | --- |
| R0 | 已完成 | 初始 HEAD=`891699269d64c67a649bdf0bf13b71ff23f2965c`，工作区干净；与审查基线 `fe807c3` 的差异仅为本任务书。阅读 AGENTS 和当前调用链后，用 `/tmp/dex_round2_baseline.py` 调用生产方法，真实 Value/RLock 下确定性重现 F1 的 epoch=2、stop=false；F2 的受限替代优化器反例为 6.493506 rad/s；F3 的 10,000 次求解晚到留下 10,000 条详情。无历史通过记录充当本轮证据。 |
| R1 | 已实施，离线通过 | 共享代际锁内完成条件检查、代际更新和最终 set；两处锁内 clear 原本正确，核实后保留，没有重构它们或把 SDK 移入锁。commit 改用 submit_end，删除 nominal_dt/control_hz fallback。两条软超时路径只更新计数和最后标量摘要，结束仅汇总一次。`round2_control_probe.py` 覆盖 A1–A4、A5 的操作请求、B1–B4、C1–C2：生产 HOME/arm 恢复在 clear 中由屏障暂停，真实锁验证两个先后顺序、旧 epoch 拒绝及迟到收尾无副作用；持 generation/goal/status 锁时 SDK stop 替代调用仍先发生。tau=0/.02/.2、近限位及重复输入，提交边界反例降为 3.896103896 rad/s；相等时间不动、倒退失败，拒绝/取消/异常不推进手臂/EEF/时间参考。有/无 recorder 各 10,000 次求解晚到＋10,000 次复制晚到，详情不增长、solver 字典复制为零；后续成功保留摘要，终止保留详情，段结束重置。`round2_device_boundary_probe.py` 进一步用生产 Robot 组合发送验证 arm 已提交而 hand 抛错不 commit、保留 HARDWARE_FAULT、实际 stop 调用先于 writer finish；目标到期不 servo，SDK stop 失败不会报停机成功。`round2_hand_solver_probe.py` 使用真实 HandFK/NLopt，在三种 tau 下各 20 次标定重复帧求解通过。时序反例用替代优化器/目标函数，设备边界用替代 SDK；均不代表电机速度或停止延迟。 |
| R2 | 已实施，离线通过 | Rollout 从创建至结束只持 epoch，以 started_ns 判断开始；删除所有 SessionState.running/end_session 和 recorder.cancelled 消费者。close_runtime 统一撤权至 DISARMED 后逐项关闭，PolicySession 随后关闭模型并共享原 deadline；段结束立即停止仍先于 Future/writer 等待。`round2_policy_probe.py` 覆盖 E1/E2：准备前、writer start 中、开始后取消的 epoch/duration/原因；重新 admission；迟到 Future 成功/失败与唯一所有权；runtime/model 同时关闭失败证据。`probe_round2_recorder.py` 覆盖 E3 主要路径：真实 HDF5＋AV 写入/关闭、3 行 RawReader 解码、真实 FK/点云/手模板的 13 字段 Zarr 导出、无动作行仍 NaN；open/append/finish/close 注入失败保留首因和 staging，16 队列满、有限 finish pending、迟到关闭仍 failed；视频/HDF5 双关闭失败、部分初始化及所有资源尝试关闭通过。真实 writer 管线已执行，外部故障注入与设备/模型用替代对象；未用 fake 格式库冒充数据集成。 |
| R3 | 已实施，离线通过 | wait_model 仅服务必要健康、请求、单 Future 与超时；warmup 只校验/投影一次快照并复用其引用，不采真实历史节拍。核实 ObservationBuilder.features 仅由同一模型 worker 填充，主线程等待不修改，不另复制 RGB-D/点云。`round2_policy_probe.py` 的 D1–D3 部分：slow load/slow warmup、完成/异常/取消/超时/设备故障期间零观测组装；n_obs_steps=1/4 × 普通/RTC，真实 builder/Pinocchio/PolicyWorker/上游 warmup 和 torch.Generator，形状正确，普通2次/RTC4次替代 predict，RNG 恢复。三个 sensor ready Event 改为 ring 首个发布，started 只存 process；键盘首次运动键才 prepare，准备独占一轮。`round2_sensor_jog_probe.py` 覆盖 F1–F3：三类真实 spawn/SharedRing 的首帧、超时、取消、发帧前后退出；生产相机函数首帧前写完 metadata，static-only 不准入；生产 jog 无输入不准备/不 hold、准备后重读松键、运行松键 hold、拒绝需重按、停止/HOME 边界后再准备。真实 SDK、网络、键盘监听和模型权重未验。 |
| R4 | 已实施，离线通过 | Robot.read_state 只检查 owner/连接并消费缓存，arm.snapshot 保留设备错误；service 保留外部健康，删除紧邻的重复遍历。HOME 回调先检查故障再判断取消；Rollout 用内层取消检查和外层统一错误边界，防止取消吞故障，没有新增健康状态。删除点云 read 后重复序列判断，VR 标定只按递增且新鲜 recv_ts_ns 去重，删除 VR 业务 ring_sequence，保留相机/SDK/来源序列。`probe_round2_health.py` 验证 replay 与两种 HOME 的同时取消＋SDK 故障；policy probe 验证纯取消、同刻 SDK/worker 故障和 stop 失败。sensor probe 使用实际 PointCloudProcessor 验证同帧只处理一次、RGB-D/点云同源，生产 calibrate_hand 跳过重复/倒退 stamp 且保留 sequence_id。缓存读取无服务遍历及 snapshot 故障透明由 recorder probe 验证。静态逐一审计 teleop/policy/jog/replay/HOME/初始化 service 调用；具体硬件路径仍待设备验收。 |
| R5 | 已完成本轮离线验收和文档 | README、hand_retargeting 与本表同步。`probe_round2_modes.py` 使用生产 Rollout/Robot/ObservationBuilder/ActionMapper/PolicyWorker，替代模型与 SDK，验证 sync/async/RTC 的真实节拍历史、首预测 reset、新执行快照、有效尾部、RTC 前缀/交接当槽映射、跳槽/耗尽及 Future 归属。18 个修改 Python 文件 AST/compile/import、6 个入口 --help、collect_teleop/run_policy --print-config JSON、ruff F/E9、mypy 默认检查及 git diff --check 通过；mypy 未覆盖无注解函数体。额外 Ruff ALL 试跑未通过，包含大量存量文档/标注/复杂度规则，以及本轮新摘要函数未注解/无 docstring、中文标点等风格项；不扩展为全仓风格整改，不称全规则通过。最终 diff 无训练时间网格/arm IPC 重写/schema/dexmani_policy 改动，无新增依赖、manager 或配置开关。初版诊断 probe 时钟未固定导致走错分支，改用确定性时钟后重跑通过；初版导出合成内参令真实离群滤波输出空云，修正测试内参后完整管线通过，生产算法未改。真机与真实模型待验范围见下文。 |

本轮验证环境：`/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`，Python 3.10.20、NumPy 1.26.4，已有 SciPy、Pinocchio、NLopt、Ruckig、PyTorch、h5py、AV、Zarr；未安装新依赖。上表所有 probe 实际命令均为 `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python /tmp/<文件名>`，标准输出和预期注入失败的 traceback 留在对应 `/tmp/<文件名>.log`（基线重现输出记录于本表）。临时 probe 只在仓库外，不是后续运行依赖。

静态命令：`PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dexmani_round2_mpl /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python /tmp/round2_static_checks.py`；该脚本对本轮修改的18个 Python 文件做 AST、compile、import，并实际执行上述 CLI。ruff 使用 `python -m ruff check --no-cache --select F,E9 <修改文件>`，mypy 使用 `python -m mypy --cache-dir /tmp/round2-mypy --follow-imports=silent --ignore-missing-imports <修改文件>`。没有运行 pytest，没有新建根 tests 树，没有仓库内 mypy/pytest/ruff 缓存。

本轮尚未执行：真实机器人取消/急停与恢复、停止延迟、手部发送时序及实际跟踪、HOME 路径/收敛、实际 sensor 来源过期与复制竞争压力、真实键盘监听/VR 网络、带权重的三模式推理及 GPU 预热性能、真实采集负载下长时录制/退出。离线的 SDK/模型替代与源码审计不算这些项目通过。SharedRing 的奇偶标记/复制复核及 HOME 必要收敛实现未改；没有将首帧条件当作持续健康或 freshness。

代码按完整职责提交为 `c28e9c7`（撤权、提交时序及有界诊断）和 `9f13b6c`（状态、启动、传感器、健康检查及收尾简化）；本实施记录另行提交。最终跨入口检查在这些代码的最终版本执行，额外覆盖补充 stop 错误为 KeyboardInterrupt 时先急停再回收 Future，保留原主因。

验收的最终问题是：同一个事实是否只维护一次，控制流是否更容易读懂，删掉的机制是否真的没有留下同义替代，以及真实设备/数据边界是否仍能正确报告失败。
