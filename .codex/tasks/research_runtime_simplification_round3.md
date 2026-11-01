# DexMani Real 第三轮任务书：绑定操作意图，厘清故障边界，按需启用功能

日期：2026-11-01。状态：**方案已确定；R0–R6 均待实施和重新验收**。

目标：让个人真实机器人研究代码轻而薄、职责清楚、行为正确、容易使用和修改。本轮修正两处真实的权限/错误语义问题，删除已确认的重复工作和过严限制；保留与设备停止、数据含义及算法效果有实际关系的保障。

## 1. 基线与本轮决定

审查基线：[`b06e9b708b21d93d1333cd8fa003c8bfc497abed`](https://github.com/haoyangzhanglab/dexmani_real/commit/b06e9b708b21d93d1333cd8fa003c8bfc497abed)。这是已经实施前两轮重构并进一步清理后的全量代码基线，本任务不是再次执行旧 Diff。

- 先读根目录 `AGENTS.md`、本任务书，再沿本轮涉及的完整调用链核实当前实现。若 HEAD 已前进，保留现有改动；已解决项只验收，不重复改写。
- [第一轮任务书](https://github.com/haoyangzhanglab/dexmani_real/blob/b06e9b708b21d93d1333cd8fa003c8bfc497abed/.codex/tasks/research_runtime_simplification.md)和[第二轮任务书](https://github.com/haoyangzhanglab/dexmani_real/blob/b06e9b708b21d93d1333cd8fa003c8bfc497abed/.codex/tasks/research_runtime_simplification_round2.md)是历史设计与验收记录。本文覆盖的冲突取舍以本文为准；不要把旧文档里的“已完成”当作新实现的验证结果，也不要抹掉历史记录。
- 范围是本文列出的机制、实际调用方和现有使用文档。无需迁移目录、升级依赖、建设通用运行平台、改写 SDK、修改 `dexmani_policy` 或改变 Raw/Zarr schema。
- 任务书写入不代表生产代码已修复。实施者应在本文末尾追加实际变更、验证和限制，完成一项再勾选一项。
- 本文给出独立可执行的重现条件和判据；历史临时探针不属于仓库依赖，不应假定实施环境中仍有这些文件。

本轮的明确取舍如下。后续章节给出实现边界，而不是要求照抄某个类层次。

| 类别 | 本轮执行 | 本轮保留 |
| --- | --- | --- |
| 操作意图 | B/H/R/T 在进入按键队列时绑定现有代际；整个操作携带已接受的身份 | 快速停止通知、共享代际锁和最终设备侧拒绝 |
| 错误语义 | 分开真实故障、正常取消、正在关闭/不可复用；统一入口对真实故障的裁决 | 故障和收尾失败不得被取消掩盖 |
| HOME | 首次使用才加载桌面事实和构造 planner；路径拒绝作为本次操作失败返回 | 几何检查、真实停稳和收敛确认、SDK 错误传播 |
| 调度 | 放宽普通 async 的不必要预测长度约束 | RTC 的固定执行窗口约定、实际过期/耗尽处理 |
| 确定可删内容 | 标定历史最大偏差否决、START 内 writer join、重复 FK/预检、回放来源白名单 | 标定统计稳定性、运行帧检查、writer 实际退出、回放目标完整性 |
| 需要实测的变化 | 记录研究项及决策所需证据 | 本轮不改训练切段规则、IK 搜索策略、点云算法和数组所有权模型 |

## 2. 五个参考项目：采用原则，不复制缺陷

对照的是固定版本的具体实现。一个参考项目缺少某项检查，不构成本项目可以删掉它的证据。

| 项目、版本及核对路径 | 本轮借鉴 | 不照搬 |
| --- | --- | --- |
| [ManiUniCon `85c6f2e`](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686)：`maniunicon/policies/torch_model.py`、`customize/act_wrapper/chunk_wrapper.py` | 预测与执行窗口的含义应直接可读；约束服务于实际调度 | SharedStorage 平台、无限 ready 等待、整段过期后重定时最后一个动作 |
| [T-Rex `ba4047c`](https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code)：`teleop/main_teleop.py`、`teleop/arm_hand_control.py`、`teleop/data_writer.py` | 最近目标、设备 owner、录制线程各有直接职责 | 无界录制队列、可能无限等待的 `queue.join`；不把它的 HOME/规划依赖整体搬入本项目 |
| [Dexora `6f2869d`](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783)：`deploy/dexora_inference_zmq.py` | 观测→推理→动作的主路径保持直接 | 黑图/零值掩盖输入失败、依靠旧动作索引维持运行 |
| [LeFranX `a39906e`](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7)：`franka_server/src/franka_server.cpp`、`src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py`、`src/lerobot/robots/xhand/xhand.py` | 伺服 owner 用稳态时钟管理目标有效期；上层组合读写可以是清楚的顺序操作 | manager/singleton 扩张、多锁协调，以及 XHand 尚未实现却返回成功的 stop；顺序观测不等于原子同步 |
| [pi-r2-flow `3af52ca`](https://github.com/pi-r2-flow/pi-r2-flow/tree/3af52ca400a6ec7d141416879aa531a6f62f697a)：`deployment/mindex/policy/control_utils.py`、`deployment/apps/run_policy.py`、`deployment/mindex/cameras/realsense.py`、`deployment/mindex/recording/dataset.py` | 无副作用的窗口运算与直接设备接口值得保留 | 多模式协调状态、旧帧重新盖时间、旧动作 fallback、全程内存录制，以及保存异常阻断 disconnect 的退出顺序 |

这些对照支持减少状态转手、依赖和重复计算，同时也说明：研究代码仍需真实的停止与失败语义。本轮不为了“像参考项目一样短”而删除有独立职责的检查。

## 3. 新发现及证据强度

下表路径均相对仓库根目录；未特别注明时位于 `dexmani_real/`。

| 编号 | 主要位置 | 已确认的问题或成本 | 决定 |
| --- | --- | --- | --- |
| F1 / P1 | `runtime/keyboard.py`；`teleop/loop.py::start_episode`；`deployment/session.py::request_start` | 队列只存键名，旧 B 在 STOP 完成后按消费时的当前代际重新获权；policy 的迟到 S 消费分支不会再次停止 | 在输入处携带现有代际，沿完整操作传递；不靠消费 S 时再次撤权补洞 |
| F2 / P2 | `robot/xarm7/process.py::check_error/connect`；各入口的取消捕获 | 正常启动取消设置 closing，随后被报成硬件故障；teleop 启动的另一条路径又可能把真实 owner 故障当成正常取消 | 真实故障优先，closing 仅限制新操作；修正所有相关取消边界 |
| F3 / P2 | `deployment/preflight.py::validate_execution_config`；`deployment/schedule.py` | async 与 RTC 共用 `A+d<=P`，拒绝调度器本可执行的短尾窗口 | 按模式校验；async 放宽，RTC 本轮维持完整窗口 |
| S1 | 四个运行入口、`runtime/home.py`、`config/control.py` | 未使用 arm HOME 的启动也强制加载桌面标定并创建 HomePlanner | 按真实消费者加载，HOME 专用检查移到 HOME 边界 |
| S2 | 交互 HOME 调用方 | `HomePathRejected` 会使整个进程退出或把完成的 replay 改成 FAULT | 停止确认且无其他错误时，报告本次 HOME 拒绝并保持可交互 |
| S3 | `teleop/retargeting/landmarks.py::HandCalibration.from_measurements` | median/MAD 已给稳定估计，仍用任一历史样本最大偏差否决整次标定 | 删历史最大偏差规则，保留 median/MAD 和运行帧骨长检查 |
| S4 | `data/recorder.py::start/finish` | START 失败先 join，外层才有机会撤权/停止并再次进入 finish；收尾责任重复 | start 报错，finish 独占等待和发布裁决 |
| S5 | `control/constraints.py`、`teleop/vr.py` | 端点检查用独立 ArmFK 后再做碰撞 FK，VR 锚点依赖这份冗余模型 | 复用同次碰撞运动学；锚点使用既有 IK 的位姿计算 |
| S6 | `robot/system.py::_send_action`、`deployment/session.py::poll_operator`、`runtime/observation.py` | 同一发布前重复代际检查、紧邻重复键盘检查及手写 freshness 公式 | 删明确重复，留下实际副作用及时间跨越后的检查 |
| S7 | `scripts/visualize_episode.py` | 初始化/读帧失败可被 close 覆盖；reader.close 与 disconnect 同时失败时首错也被覆盖 | 两项清理都尝试，原始失败和后续清理失败分别保留 |
| S8 | `replay/trajectory.py::load_trajectory` | 在检查实际目标前按 collection_source 排除所有 policy_rollout | 删除来源白名单；以既有格式、完整性、目标与限位决定可回放性 |

### 3.1 F1 的确定性时序

使用真实 Keyboard 队列/Condition、SessionState、共享 Value/RLock 和生产准入方法，设备与观测边界替代为可观察调用，已复现：

1. 代际为 1 时，B 已入队但主循环尚未消费。
2. S 的快速回调完成撤权：代际变为 2，stop=true；暂停在 S 键名追加到队列之前。
3. 主循环只取出旧 B，却在此时读取代际 2 作为它的身份。
4. policy 准备代际 2 并进入 RUNNING，stop=false；teleop 接受后自行撤权到代际 3，同样进入 ACTIVE。
5. 再让 S 入队。teleop 后续处理能结束本轮；policy 仅抑制该批准备命令，仍可能保持 RUNNING、stop=false。

这证明旧意图获得了新权限。它不同于第二轮已修复的“共享锁竞争导致最终 stop 丢失”。仅把 `run_id` 读取移到 `poll()` 前、清空一次队列或在消费 S 时重复 set，都不能完整建立意图身份。

### 3.2 F2 的两种错误分类

- `ArmProcess.connect` 的正常取消会设置 closing。当前 `check_error()` 在无共享 fault、进程仍活着时，也因 closing 抛出 RobotError；policy/keyboard 的取消处理再调用 `robot.check()` 即可走入该误报。
- teleop 启动 hand HOME 时，先 stop 使 `stop_required=False`；随后 owner fault 与 stop 到达，`check_run()` 先看到 stop 抛 OperationCancelled，外层只看停止标志便可标记 clean。真实 `send_hand_home` 路径已复现该异常与标志组合；`ArmProcess.close` 不重抛既有运行 fault，owner 也可能干净退出。因此不能指望 finally 自动纠正分类。未把这个局部复现冒称完整真机启动复现。

### 3.3 其他可重复反例

- `P=8,A=8,d=2` 被当前 async preflight 拒绝，但现有 schedule 在有及时结果的 24 个 slot 探针中能连续交接，稳态保留 6 项尾部；这是调度可行性的证据，不是硬件延迟保证。
- 25×16 个标定长度中，24 帧为 0.04，一帧的一个长度为 0.08，median 不变、MAD=0，仍被历史最大偏差规则整体拒绝。删除该规则不等于放过运行中的异常帧。
- recorder 的两处 join 使用同一个 stop_deadline，**不能声称原实现必然等待两次完整的 60 秒**。真正问题是收尾所有权分散，以及 START 内等待推迟了外层停止。
- export 的当前前缀规则会在单个无效行后舍弃其后全部行；10 Hz、每步 100.8 ms 的合成序列也可因累积相位偏差截短。这是需要研究的数据取舍，本文不据此直接改变训练序列语义。

基线证据包括源码调用链、生产方法离线探针和数值检查；未完成真实机器人、GPU 模型或完整 HDF5/AV/Zarr 集成验收。新实现必须重新验证，不能继承这些探针的“通过”结论。

## 4. 目标架构：保持现有骨架，减少身份重取与责任重复

| 事实或职责 | 权威来源 / 所有者 | 本轮约束 |
| --- | --- | --- |
| 当前代际与快速停止 | 现有 SessionState.run_signal、stop/emergency Event | 不新增另一套事件序号、ACK、心跳或租约 |
| 一次显式操作的权限 | 入队时的代际；成功准入后固定的 operation/episode epoch | 不用后续“当前 run_id”给旧操作续命 |
| 运行故障 | 既有共享 fault、close_error、真实进程退出及服务失败 | closing 不是故障副本；cancel 也不能清除真实故障 |
| 能否接受新 I/O | 实际 connect/submit/lifecycle 边界 | 正在关闭的 owner 不能复用，不能靠伪造硬件故障表达这一点 |
| HOME 依赖 | 当前会话的一份已加载桌面事实和按需 planner | 不新建 manager、registry 或全局缓存 |
| 录制生命周期 | start 启动；finish 等待和决定发布；writer 持有写盘资源 | START 不抢做监督方的 finish |
| 动作发布 | 父进程校验格式/绝对限位；IPC 发布与 SDK owner 各守实际边界 | 保留跨等待/跨进程后仍必要的检查 |
| 时间 | 同机 monotonic_ns；来源时间、组装时间、提交时间各保留事实含义 | 不用读缓存的时间更新来源 freshness，不重建跨层 deadline 链 |

继续保留第二轮成果：撤权在共享代际锁内最终 set；恢复在同一锁内核对并 clear；成功提交返回时间推进手部参考；软超时固定大小摘要；rollout 单一身份；传感器首帧发布代替冗余 ready Event；runtime 唯一全局收尾入口；模型任务有真实 owner/pending 状态。不要重新引入这些已删除的副本。

## 5. 实施规格

### 5.1 F1：操作意图在入口绑定，操作期间不重新取权

涉及 `runtime/keyboard.py`、四个 Keyboard 构造点、`teleop/loop.py`、`deployment/session.py`、`deployment/rollout.py`、`teleop/keyboard.py`、`replay/run.py`、`replay/joint_replay.py`、`runtime/home.py` 及真实调用方。

1. 用一种小型不可变事件表示取代队列里的裸字符串，内容只有键名与可选的既有 run_id。B/H/R/T 属于会触发准备、HOME 或设备复位的显式意图，在输入被接受时、入队前读取一次现有代际。所有生产者/消费者一次迁移；不保留字符串与对象两套协议或兼容适配层。`discard_events` 继续按键名删除。
2. S/D/Q/Esc 不因事件代际失配被过滤，快速停止回调仍先通知设备并撤权，不能为给停止键盖代际而先等待共享锁。持续 jog 的 held-key snapshot 保持现有语义，不给每个方向键新增代际、时间戳或历史。
3. 读取意图代际时不持有 Keyboard Condition，再进入 Condition 发布事件；消费方先取出事件并释放 Condition，再做状态准入。沿用本地 motion_lock→共享代际锁的顺序，不能形成反向加锁。代际快照与入队之间若发生停止，事件自然失效；不需要把队列和运动状态做成一笔大事务。
4. B 的准入比较事件的代际与当前代际。teleop 若在接受时需要创建新代际，应在同一现有状态临界区核对旧身份、执行既有撤权并取得新身份；此后整段只使用这个已获准身份。policy 不得在 `request_start` 消费时重新取当前代际作为授权。
5. H/R 的授权必须覆盖“停止当前 jog（如有）→加载/规划→hand HOME→arm HOME”。主动停止引起的代际变化只允许在该操作的准入临界区产生并明确携带；后续 helper 不得悄悄重新读取当前代际。`home_robot/home_hand/home_arm` 以及启动期 `initialize_hand` 的连续步骤应使用各自操作开始时确定的一份身份；长规划、反馈等待和两个设备阶段之间重新验证这份身份。
6. T 也核对入队身份，并在复位等待中用同一身份响应撤权；它不创建新运动权限。继续保留“空闲且无在途 episode/准备”限制。停止后的新 T 可以按原有使用约定重新发起。
7. 不把 `can_move(ARMED)` 一律改成要求 stop=false：停止后的合法新 B/H/R 本来就需要经过显式准备恢复。也不把所有 stop.clear 收拢成绕开真实 I/O 边界的新服务；保持原有锁保护和 owner 检查。
8. 继续按当前终止优先级处理同一批中的 S/D/Q/Esc；保存/丢弃与退出意图不能因事件迁移丢失。停止前入队的意图拒绝，停止完成后新进入的意图可以正常准入。不得引入事件 TTL、第二计数器、ACK 或恢复队列。

线性化边界是“本进程接受并读取意图代际”的时刻，不能声称知道物理键按下的精确时刻。若一个合法新操作又被后续停止撤权，允许其被取消；不要求所有竞态中的操作必须成功。

### 5.2 F2：真实故障检查不再把 closing 当作硬件故障

涉及 `robot/xarm7/process.py`、`robot/system.py`、`runtime/home.py`、teleop/policy/keyboard/replay 的取消与退出边界。

- 收窄现有 `ArmProcess.check_error()` 的语义：检查真实共享 fault、已有清理错误及非预期/不干净退出；单独的 closing 不是硬件故障。明确关闭中的存活进程、已确认停稳且 exitcode=0 的预期退出，与未请求关闭的退出有不同含义。仅 exitcode=0 不足以证明正常；单独 stopped Event 也不能覆盖 close_error。
- 新操作仍在真实 I/O 准入处检查 connected/closing/可复用性，沿用已有状态和异常类型。`submit`、`request_operation`、准备/HOME 和 snapshot 的真实需要逐一核对；不要为修取消误报而让已经关闭的句柄继续被使用。
- 取消分类统一采用“先确认真实故障，再接受操作者取消”。尤其修复 teleop 启动 hand HOME 的漏检，以及 policy/keyboard 启动取消的误报。可以在现有 Robot 边界统一真实故障检查，不能新增全局健康管理器、fault 副本或异常状态机。
- 异常处理器内再次检查若抛出真实故障，也必须进入既有故障记录/非成功退出和 finally 清理；不要误以为同一个 try 的后续 `except Exception` 会捕获前一个 except 内新抛出的异常。
- `OperationCancelled` 带有 `cleanup_errors` 时，不得只打印“已取消”并恢复空闲；无故障取消与停止未确认是两件事。运行异常、清理异常都保留，且实际故障不能被旧的取消理由覆盖为成功。
- close 仍负责 stop/disconnect 的真实结果，不承担掩盖遗漏运行错误的职责；不吞 SDK 错误，也不把正在退出的线程/进程伪装成已经结束。

本轮不增加新共享 flag、不扩大 RunEndReason 枚举，不把设备故障降级为普通超时，也不需要把每个函数都包上一层 catch/rethrow。

### 5.3 S1/S2：HOME 是按需操作，几何拒绝可以恢复

涉及 `teleop/run.py`、`teleop/loop.py`、`teleop/keyboard.py`、`deployment/run.py`、`deployment/session.py`、`replay/run.py`、`runtime/home.py`、`config/control.py` 和实际桌面消费者。

1. 各会话使用一个 `None`/已构造 planner 字段即可。首次有效 H/R、在机器人已停止且操作身份仍有效时，加载并验证桌面平面、验证 HOME 专用环境配置、构造 HomePlanner。后续使用同一实例；不做后台预热、自动文件监视或失败状态缓存。复用既有停止边界，不因延迟构造又叠加一次已确认的停止流程。允许在现有 HOME 模块增加一个直接的构造函数，不新建“HOME 服务”。
2. 在线工作空间和 static_boxes 是实际控制约束，仍在运行前验证。`EnvironmentConfig.validate()` 不再强制验证仅 HOME 使用的桌面盒参数；这些检查随 HOME 构造执行。手部启动 HOME/触觉初始化不应依赖 arm HomePlanner。
3. 点云若实际启用 `remove_table`，仍须在对应 worker 启动前加载有效平面；只记录 RGB-D、只使用关节/触觉、未开启去桌面的点云，不因此强制加载平面。根据最终 ObservationNeeds 和实际配置判断，不能仅凭 `record_raw` 或“可能以后按 H”决定启动依赖。
4. policy 已因点云加载的桌面事实供 HOME 直接复用；未加载时才在 HOME 加载。run_config 中未使用的平面保持未加载的真实表示，不能为写元数据又无条件调用会拒绝空路径的 `resolve_table_plane_path`；首次使用后通过现有配置记录路径记下实际系数和路径。不要填默认平面、伪造已校准事实，或让点云和 HOME 各自重新加载不同版本。
5. 缺少/无效 HOME 标定与专用配置，只在其加载/校验的小范围处理为本次 HOME 无法执行，提示具体文件或参数并保持停止。不要用覆盖整个 HOME 的 `except ValueError/Exception` 把设备或代码错误也吞掉；模型加载、技术异常和 SDK 故障仍透明传播。
6. 明确捕获 `HomePathRejected`：在停止已确认、无真实故障、无附带清理错误时，报告本次路径拒绝，回到 ARMED/空闲，允许用户调整后再操作。保留有限候选路径、碰撞和桌面检查、实测手姿、Ruckig、反馈新鲜度与真实收敛确认，不自动换成直线运动或跳过 HOME。
7. 有效手 HOME 可能已经完成、arm 路径才被拒绝，报告必须准确，不承诺整机“完全没动”。异常分支也确保 hand/arm 的当前停止状态。
8. replay 完成后的交互 H 被几何拒绝，不把之前完成的 replay 改成 FAULT；在既有 home_results 记录本次失败并继续等待 H/Q。真正的设备、停止或清理失败仍按错误结束。

planner 构造和规划不在 motion_lock/共享代际锁内执行。停止发生于加载或规划期间时，5.1 的固定身份必须阻止后续发送；按需初始化不能开出新的竞态窗口。

### 5.4 F3：普通 async 使用可用尾部，RTC 继续执行完整窗口

记 `P=policy.future_steps`、`A=action_steps`、`d=prefetch_steps`。

| 模式 | 本轮准入条件 | 执行含义 |
| --- | --- | --- |
| sync | 整数 `1<=A<=P` | 等待预测不消耗其动作，按实际提交推进游标 |
| async | 整数 `1<=d<=A<=P` 且 `d<P` | A 是计划交接窗口上限；到达时裁掉已过去的前缀，允许可用尾部短于 A |
| rtc | 整数 `1<=d<=A<=P` 且 `A+d<=P`，保留 guidance_cap 校验 | 稳态按约定交接并保持完整 A 项窗口 |

只改模式对应的 preflight、必要说明及真实受影响调用。保留现有 schedule；不要新建异步队列、重新采样器、迟到结果复活逻辑或“最后动作续期”fallback。`d=P` 即使满足 `d<=A` 也应拒绝，因为交接时可能没有任何尾部；运行中真实延迟大于可用 horizon 仍由原有耗尽/结束逻辑处理。sync 不应开始要求无消费方的 d 约束。

`A+d<=P` 对 RTC 是本轮保留的执行约定，而非所有异步调度的数学必要条件。若未来要允许 RTC 动态短窗口，须单独验证前缀承诺与训练/推理接口，不能在本轮顺手放宽。

### 5.5 S3：标定估计稳健，运行帧继续严格检查

在 `HandCalibration.from_measurements` 删除“所有历史样本相对 median 的最大偏差必须小于运行 bone_length_tolerance”规则。保留实际输入边界已有的有限/有效几何检查、样本数量、来源去重、median 和 relative MAD 稳定性条件。`sample_count` 仍表示实际收集数量，不伪称全部为 inlier。

`HandCalibration.targets` 的当前帧骨长容差继续保留；一个异常样本不毁掉稳定标定，一个运行异常帧仍不能作为有效重定向输入。无需增加 outlier manager、重试轮次、筛选配置、第二套统计或持久诊断列表。本轮不改重定向 objective、平滑增益、优化器及提交时间基准。

### 5.6 S4：START 只启动和报错，finish 独占监督收尾

- `EpisodeRecorder.start` 失败时保存首错并立即传播，删除其 stop_deadline 初始化和 thread.join。保留 ready 作为 writer 打开结果的通知、实际线程引用及既有单次使用约束。
- 两个实际启动调用方（teleop、rollout）必须在 start 前已持有 recorder 引用，并在 finally 进入既有 finish；不能因为 start 尚未成功就丢失资源所有权。外层先执行必要撤权/设备停止，再等待 writer。
- 首次 finish 建立唯一停止预算，并与传入 deadline 取更早者；后续调用不重置预算。writer 持续阻塞时保持 pending 和 staging 的真实状态，不发布、不置假关闭、不改 daemon，也不使用可能无限等待的 queue.join。
- writer 后来退出也不能自行发布已被监督方判定失败的录制。只有 finish 看到写入/关闭成功、实际退出及合法保存条件后才能原子发布；首个运行错误和关闭错误仍可追踪。
- 不重写录制架构或增设第二个 finalize 状态。修正注释，说明预算从首次 finish 开始，避免留下 START 仍拥有收尾预算的旧描述。

### 5.7 S5/S6：删除有明确边界的重复计算和预检

**FK：** `MotionConstraints` 删除独立 `target_fk`；无外部已计算 position 的端点检查，先做同一 target/同一 hand posture 的碰撞计算，再使用该次 FK 的 EEF position。现有 `CollisionModel.eef_position()` 依赖最近一次 rejection 更新，不是一个独立计算 API，不能读到另一候选的缓存。IK 已有 position 时继续使用它，避免新增一次 IK FK。VR 锚点通过现有 solver 的 `compute_eef_pose_base` 取得位姿，保持坐标系和四元数顺序。

碰撞和工作空间同时不满足时，可以按碰撞优先报告；先核对 reason 的实际消费者没有依赖旧的首错优先级，不为维持一个日志字符串引入额外 FK。接受/拒绝的集合和几何阈值必须保持不变。该修改消除有效候选的重复 FK，但工作空间外候选的提前拒绝成本可能变化；报告实际调用/耗时证据，不宣称所有输入都更快。不要让不同线程共用可变 Pinocchio Data，也不删除承担不同 owner 职责的模型实例。

**arm 发布：** `Robot.validate_command` 仍在任何设备副作用前完整校验组合目标、格式和绝对限位。仅删 `_send_action` 中 arm 分支与 `ArmChannels.publish_goal` 重复的代际/stop/TTL 预检；由 `ArmProcess.submit` 承担必要真实故障/可用性检查，由 publish_goal 在实际发布、等待锁之后裁决代际、停止和有效期。进入可能产生副作用的尝试前仍设置 stop_required。保留 hand SDK 发送前检查、组合发送后的取消/故障处理和 owner 发送前的最终检查；它们跨越了不同的等待或副作用，不能按“出现两次”机械删除。

**键盘健康：** `poll_operator` 复用 `Keyboard.poll` 的健康检查，删紧邻重复遍历，保持监听器失效时立即锁存急停的行为。一次阻塞 poll 或设备调用后的 service 检查可能已是新时点，应保留；不加“本轮检查过”的缓存 flag。

**freshness：** `runtime/observation.py` 的 arm/hand 条件复用既有 `sample_is_fresh`，一个观测裁决使用同一个 now。保留 `stamp>0`、未来时间拒绝、`now==stamp+max_age` 已过期的严格边界，保持 required_tactile 和可选手部含义。不要用组装时间替代来源时间，或把观测准入误当成 SDK 稍后发送时仍新鲜的证明。

只删上述重复；错误分类的正确性优先于减少检查次数。异常附加清理错误如确有重复，可在现有 utils/log 中用小函数统一；不引入异常包装框架或新的资源管理层。

### 5.8 S7：viewer 保留主错，完成所有可尝试的清理

`EpisodeVisualizer` 初始化失败、主循环读帧/渲染失败以及正常退出都使用一致的错误优先级：已有业务异常仍为主错，reader.close 和 rr.disconnect 都尝试，失败附加到既有 cleanup_errors 约定；没有主错时，第一项清理失败为主错，后续清理失败附加。

保留原始 traceback/cause，不只保留最后一句字符串，也不能只靠打印日志后返回成功。已确认关闭的资源不重复关闭；关闭失败/结果不明不能先把句柄置空伪装成功。实现限定在现有脚本和必要的既有错误辅助函数，不建设通用 ResourceManager。

### 5.9 S8：回放按数据实际含义准入，不按采集来源封禁

删除 `load_trajectory` 中 `collection_source != teleop` 的早期硬限制。RawReader 仍检查当前格式和既有 action semantics，仍验证发布完成、非空且所有目标有限、首行实测状态有效；既有限位、执行前对齐、反馈与停止逻辑均保留。来源元数据继续作为 provenance 保存。

这不保证任意 policy_rollout 都可回放：当前策略 Raw 可能含无命令的 NaN 行，仍应在进入设备执行前明确拒绝。不要填零、沿用上次目标、过滤 NaN 后压缩时间、抽取尾段或更改 action 字段含义。回放仍是以标称频率执行记录目标，不宣称重现原来的物理轨迹或原始发送时序；`q_goal_submission_attempt` 也不是 SDK 已执行的证明。

## 6. 本轮不实施的研究项及下一步判据

这些不是默认必须保留的“永久契约”，也不是授权实施者在无数据时自行删除。记录为后续研究，**不阻塞本轮已确定修正，不为此添加开关、基础设施或占位代码**。

| 研究项 | 候选简化 | 决策所需证据与边界 |
| --- | --- | --- |
| canonical 导出的连续前缀/全局时间网格 | 按最大连续有效片段导出，而非遇一处坏行即丢弃其后全部；是否改用局部节拍判定 | 用真实 Raw 量化无效行、间隔和累计漂移各造成的丢弃比例；核对训练 loader 的 observation/action horizon、episode_ends、source row provenance。不能跨缺口拼接或静默重采样；本轮不改固定 13 字段和训练语义 |
| IK 多 seed/nullspace | previous-only 或 previous+measured 替代更广搜索 | 相同真实 VR/策略目标、相同碰撞几何，比较成功率、分支跳变、限位裕量、P50/P95 耗时；不能仅凭静态行数删掉求解恢复能力 |
| 点云多级过滤 | 在真实相机数据上比较 ROI/voxel/FPS 的精简组合 | 比较有效几何保留、薄物体和手部影响、空云率、耗时及模型输入一致性；训练 recipe 改变需要显式研究迁移 |
| ObservationBuilder 的 RGB-D copy→stack | 对已经独立拥有的数组减少一次复制 | 证明每条来源路径的数组所有权、后续写入与内存驻留；cloud packet 中的视图可能让短历史持有整包 RGB-D。先测分配和延迟，不能用 frozen dataclass 冒充数组不可变，也不新增零拷贝生命周期协议 |

保留必要的来源时间、有限队列、真实 owner、writer pending、原子不覆盖发布、组合目标校验和实际停止反馈。它们已经对应具体失败场景；需要被删的是重复责任和无消费方的机制，不是这些事实本身。

## 7. 分阶段实施与提交

| 阶段 | 工作与交付 | 阶段通过条件 |
| --- | --- | --- |
| R0 | 核对当前 HEAD、用户改动、完整相关调用链；重现 F1/F2/F3，记录缺依赖情况 | 区分仍存在、已修复和证据不足；不在重现阶段改生产语义 |
| R1 | F1 操作事件及固定身份贯穿，F2 故障/取消/关闭分类 | 旧意图不复活；新意图可用；取消与真故障的分类及退出正确 |
| R2 | S1/S2 HOME 按需加载、正确的可恢复路径拒绝 | 四个入口都无无关 HOME 启动依赖；停止/代际跨规划和多设备阶段有效 |
| R3 | F3 async 校验、S3 标定规则 | 有效短尾配置可运行，无尾/真正过期仍拒绝；标定稳定性与运行帧拒绝分工清楚 |
| R4 | S4 recorder、S7 viewer、S8 replay 来源规则 | START 不抢收尾；资源/主错真实；回放不篡改数据 |
| R5 | S5/S6 重复 FK、发布预检、键盘健康和 freshness | 目标接受集合不变；实际发送边界未失守；删除完整且无兼容残留 |
| R6 | 全量受影响链复查、适当验证、现有 README/手部文档更新、本文追加实际结果 | 明确通过与未执行项；最终 diff 无额外框架、配置、缓存或无关重构 |

使用可审阅的分阶段提交；无须机械拆成每文件一次提交。不要执行破坏性 reset，不覆盖用户工作。实现时发现与基线不同的事实，先以当前代码验证；能在上述语义内直接解决的，不因普通实现选择停下来索要确认。训练语义等第 6 节研究变化保持未实施。

## 8. 验收矩阵

不恢复根目录 tests 树，不建立测试框架或通用 fake SDK 平台。对这些并发、错误和数值边界进行必要的确定性探针；可用环境临时目录，必须调用生产实现及其真实共享原语，替代外部 SDK/模型/I/O。不能只测试复制出来的简化算法。若依赖缺失而不得不抽取生产方法，明确它只证明局部性质，仍需后续真实模块导入和集成验证。竞态用 Event/Barrier 控制合法暂停点，不靠 sleep 碰运气。

| ID | 场景 | 必须观察到的结果 |
| --- | --- | --- |
| V1 | 旧 B 入队→S 回调完整撤权→暂停 S 入队→消费旧 B；teleop/policy 都执行 | 不 prepare、不 begin_motion、不发送；stop 仍为 true。迟到 S 消费后也不运行 |
| V2 | S 已完成后重新按 B；B 入队后又发生 S；同批 B/S、B/D、B/Q | 新意图可准入；被后续 S 撤权的意图不执行；保存/丢弃/退出及终止优先级不丢失 |
| V3 | 旧 H/R/T 入队后停止；HOME 加载/规划中停止；hand 完成与 arm 开始之间停止 | 不重新取得当前代际，不再发送旧操作；有意停止 jog 后合法 R 仍能完成；真实故障不被取消吞掉 |
| V4 | 准备持有代际锁时触发 STOP/ESTOP，并在各允许位置释放 | 沿用第二轮最终 set 保证，停止完成后 stop=true；设备检查不发送已撤权目标；快通知不依赖长 SDK 锁 |
| V5 | ArmProcess 启动正常取消，分别处于存活+closing、正常已退出 | 无共享 fault/close_error 且停止干净时不误报硬件故障；入口返回和结果准确表示取消 |
| V6 | hand HOME 中 fault+stop 同时出现且 stop_required=False；取消带清理失败 | 真实故障/停止未确认仍报告为失败，返回非成功；原错与清理错可追踪 |
| V7 | 未预期 exitcode=0、非零退出、closing 后 stop/disconnect 失败；关闭对象再 submit | 不因“正在关闭”或“退出码为零”吞错；不能复用；与正常取消明确区分 |
| V8 | 缺少桌面标定，分别运行无 HOME 的 teleop/keyboard/replay/policy；policy 各点云模式 | 无关功能可启动；remove_table 的实际消费者仍失败于其边界；首次 H/R 提示明确并保持停止 |
| V9 | HOME 无路径；HOME 超时/SDK 故障/附带 stop 失败；随后重新输入 | 仅干净几何拒绝可留在空闲重试；其他失败不伪装恢复；replay 完成结果不被单次几何拒绝抹掉 |
| V10 | async `P=8,A=8,d=2` 连续交接；`P=A=d`；延迟使 offset 达 P；相同参数 RTC | async 有效尾部可用、无尾和真实耗尽拒绝；RTC 仍拒绝 A+d>P；sync 行为不变 |
| V11 | 25 个稳定标定样本夹一个长度离群值；足以超过 MAD 门限的波动；运行异常帧 | 前者得到稳定 median，后者拒绝；运行骨长异常继续拒绝，不改变提交参考 |
| V12 | writer 打开失败/阻塞、START 超时、KeyboardInterrupt、外层 finish、迟到退出 | START 不 join；外层先停止后等待；唯一 finish 预算、不发布失败录制、不遗失 pending/首错 |
| V13 | 正常多帧录制；队列满；writer.close 失败；真实 HDF5/AV 可用时集成 | 成功只在真正关闭后发布；错误录制留 staging；无隐性丢帧或假成功。区分 fake 与真实写盘证据 |
| V14 | 多个有效/越界/碰撞 target 和不同 hand posture；VR reset；IK 已提供 position | 接受集合、坐标与锚点一致；无 position 的端点不重复 ArmFK；不读上个候选的缓存 |
| V15 | arm 发布前/等待锁中撤权与过期，arm+hand 组合，监听器退出 | IPC 和 owner 最终边界仍拦截；手前检查/组合校验有效；监听器故障仍触发急停 |
| V16 | freshness：0、未来、恰好 deadline、deadline 前 1 ns；可选手与触觉需求 | 与基线边界等价，所有来源使用同一个裁决 now，不续期 |
| V17 | viewer 构造/读帧失败，再叠加 reader.close 和 disconnect 同时失败；纯清理失败 | 两项清理都尝试，主错及后续错误全部保留；无错误时正常结束 |
| V18 | 内容相同的有效 teleop/policy Raw；含 NaN、未完成、越限、无效初态 | 来源不决定可回放性；实际无效内容在设备执行前拒绝，不插值/补值/压缩时间 |

此外检查实际受影响 CLI 的 `--help`、已有配置打印、静态语法/导入和可用的仓库既有检查。工具可用才运行：ruff 使用 `--no-cache`，mypy 将 cache 指向环境临时目录；不生成仓库缓存，不为运行检查升级依赖。无需为简单字段/注释删除新增镜像实现测试。所有探针必须避免连接真实设备；真机验证单独明确记录。

真机/真实模型验收至少覆盖：teleop 的 B/S/D/Q、VR 锚点与手部提交；keyboard 首次 jog、松键/拒绝后恢复和 R；policy 的 sync/async/rtc 正常交接与停止；回放对齐、结束后的 H/Q；停止期间的 HOME 与录制退出。只有在对应设备环境中实际执行才能勾选；假 SDK 和源码审查不能替代物理停止、实际模型和录像完整性的证据。

## 9. 文档、完成标准与实施记录

更新现有 README、相关手部/运行说明中的实际行为：HOME 依赖何时加载，几何拒绝如何处理，async/RTC 参数差异，标定统计规则及回放准入条件。无需新增另一份运行规范、规则清单或长篇架构文档。不能把本任务的全部实现细节塞入 CLI 用户提示。

完成标准是明确问题被解决、责任更少且行为可解释，不是达到某个删行比例。尤其检查：一个事实仍有一个 owner；新事件仅携带原有身份；没有两套按键接口；没有新共享状态/配置开关；没有无消费方的包装函数；异常未被降级为成功；第 6 节研究项未混入生产改动。

- [ ] R0：基线、现状和关键问题重现完成。
- [ ] R1：固定操作身份、取消/故障分类完成并验证。
- [ ] R2：HOME 按需依赖和可恢复拒绝完成并验证。
- [ ] R3：async 与标定规则完成并验证。
- [ ] R4：录制、viewer 与回放准入完成并验证。
- [ ] R5：重复计算/检查清理完成并验证。
- [ ] R6：完整受影响链复查、现有文档和实际证据记录完成。
- [ ] 真机与真实模型验收已实际执行；若未执行，保持未勾选并说明。

实施者在此追加精简记录：实施 HEAD/提交、各项具体改动、V1–V18 实际运行结果、依赖/硬件限制、仍待确认的风险。缺失依赖就写未执行，局部探针就写局部证据，不写“全面通过”。不要仅罗列 diff 或编译结果来代替行为验收。

当前实施记录：**尚未实施。本文提交只增加任务书，不修改生产代码。**
