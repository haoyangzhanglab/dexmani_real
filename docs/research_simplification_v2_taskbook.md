# DexMani Real 第二阶段减法重构任务书

> **历史记录：** 本文记录截至 `7ac6114` 的第二阶段实施事实，代码与离线回归已实施，真机验收未执行；实施范围与当时验证见第 9 节。当前行为以源码和 [README](../README.md) 为准。
> XHand 的写 CRC、触觉保持、启动 HOME 与清零约定已由 [XHand 轻量化任务及实施记录](xhand_simplification_taskbook.md) 替代；下文旧验证事实保留。
>
> **审查基线：** [025451cc5ae3e1c77be221f50273c296dbf26769](https://github.com/haoyangzhanglab/dexmani_real/commit/025451cc5ae3e1c77be221f50273c296dbf26769)。
>
> **适用关系：** 本文与[上一阶段任务书](research_simplification_taskbook.md)均保留为历史实施记录；下列要求和测试数量只对应当时阶段，包括已被后续修改替代的触觉 TTL 告警约定。
>
> **交付目标：** 在现有仓库原地完成代码、调用方、配置、测试和使用文档的同步修改。只实现本文要求，不扩展成新的运行平台。

## 1. 项目定位与最终原则

DexMani Real 服务个人真实机器人研究。研究者应能沿着“读取输入 → 计算目标 → 提交设备 → 按需记录 → 停止收尾”理解运行过程。减少需要同时记住的状态、协议和约定，比减少某个文件的行数更重要。

本轮保留 teleop、jog、HOME、replay、sync / async / RTC、正常多轮实验、Raw 记录和现有 Canonical 训练数据输出能力。没有实验依据，不删除实际使用的策略模式、传感模态或算法。

执行以下原则：

1. 一个事实只有一个权威来源。设备是否停止、任务是否完成，以设备状态或真实任务句柄为准；不要维护另一套推导状态。
2. 校验放在外部输入、采样接纳和实际副作用边界。跨过耗时计算后，不假设先前的新鲜度仍成立。
3. 允许名义频率达不到、导出数据有周期抖动、首次推理较慢；这些现象不自动等于数据损坏或设备故障。
4. 保留真实限位、来源时间、停止/急停、设备目标租期、实际资源释放和写入完整性。
5. 正常结束支持复用；异常路径优先停止并退出会话，不为继续复用故障资源设计恢复协议。
6. 不以“别人没有”为删除依据，也不以“存在测试”为保留依据。每项机制必须能指出实际消费者或具体失败反例。

### 1.1 不做什么

- 不新增 Manager、Coordinator、Registry、授权令牌、通用截止时间框架、资源状态矩阵、后台 finalizer 或自动重连。
- 不增加 strict/relaxed 双模式、旧 API 别名、双格式 reader 或迁移器来维护本轮主动删除的约定。
- 不更换控制栈，不引入新的依赖、ROS/Hydra 等配置平台，不为减法任务大规模搬目录。
- 不改变模型训练接口、动作含义、归一化、历史窗口采样语义、RTC 前缀定义或随机种子语义。
- 不批量修改历史实验数据、checkpoint、标定、URDF 和已有输出目录。
- 不连接真机、不自动执行 HOME/servo/hand 动作；离线验证与真机验收分开报告。
- 不承诺 Python 能撤回已进入 native SDK 的调用，或在后台线程卡死时保证整个解释器限时退出。

## 2. 审查结论汇总

以下路径除 `scripts/`、`tests/`、`docs/` 和根文件外，省略 `dexmani_real/` 前缀。

### 2.1 八个核心主题

| 主题 | 当前事实 | 最终决策 |
| --- | --- | --- |
| 时间戳 | 运行时主要使用主机 monotonic 纳秒；各来源的时间含义不同。观测读取时间不是物理执行时间。 | 保留来源时间、读取时间及必要序列号；明确含义，不再叠加通用时间对象。 |
| 新鲜度 | 来源 TTL、设备反馈 TTL、动作租期解决不同问题；缓存复制和模型完成都不产生新采样。 | 保留这些真实边界；删除相同阶段、相同事实的重复预检，不跨耗时阶段盲删检查。 |
| 时间网格 | 普通循环、async 和 RTC 已有不同语义；Canonical 仍有过严的周期硬门槛。 | 普通循环允许变慢；async 保留原预测时间窗口；RTC 保留交接承诺；离线周期抖动降为质量提示。 |
| 生命周期 | 资源所有权基本明确，但异常清理、状态汇总和恢复分支跨层叠加。 | 明确正常结束和终止会话两条路径；设备先停，真实 owner 收尾；删除通用完成状态汇总。 |
| 会话管理 | 多轮复用设备和模型有实际研究价值；结果文件包含较多清理分类。 | 保留正常多轮复用和简明实验结果；异常或未回收资源不进入下一轮。 |
| 授权/撤权 | `SafetyState` 与主流程阶段、stop、取消计数重复表达准入；`ARMED` 还允许某些准备动作。 | 删除通用授权状态机及参数链，以具体操作入口、取消信号和设备事实表达边界。 |
| 代际 | 跨进程运行代际已删除；父进程 `cancel_epoch` 用于取消长操作。ring 版本号和相机序列号另有职责。 | 保留一个本地取消计数；不重新引入 run_id/command generation；不误删数据版本号。 |
| 句柄 | Future、线程、进程、Pipe 未消费回复是真实未完成工作的证据。 | 保留真实句柄及必要局部状态；删除镜像台账、重复分类和由超时推断“已关闭”的逻辑。 |

### 2.2 需要实施的减法

| 编号 | 机制 | 处理 | 替代后的行为与代价 |
| --- | --- | --- | --- |
| D01 | Rollout 的预测等待期限、父进程目标租期镜像、预算排序和相等时刻裁决 | **整套删除** | 实验时长由一个 episode 截止时刻控制；已发送目标由设备 owner 的租期控制。首次有效动作可以等待更久。 |
| D02 | `SafetyState`、`arm/begin_motion/can_move/revoke_motion` 通用授权 API | **整套删除** | 主流程拥有当前操作；实际发送检查取消、stop、急停、设备状态和输入。 |
| D03 | prepare/HOME 异常后清旧回复、重新恢复同一 owner 的协议 | **删除恢复用途** | 在途生命周期操作取消、超时或故障后退出当前会话。停止、故障读取和关闭仍要执行。 |
| D04 | session 的 model/writer/robot/sensor/keyboard pending/completed/failed 聚合 | **删除** | 资源 owner 报告真实错误，顶层返回非零；结果文件保留实验信息，不参与控制。 |
| D05 | 结束时再扫描键盘队列、追认迟到 D、细粒度结束原因竞态仲裁 | **删除** | 主线程结束该段时作一次保存/丢弃决定；后台回调仍立即执行快停。 |
| D06 | 录制 freeze 后继续控制、动态减少 capture needs | **删除该运行模式** | 开启录制时，录制输入失败或队列满就结束当前段；可保存已接纳且成功落盘的有效前缀。 |
| D07 | Policy 默认全量 Raw 记录 | **改为显式开启** | 默认只保存简明结果和配置；`--record-raw` 才要求完整录制输入。 |
| D08 | Canonical 周期 ±25%、读取跨下一周期、触觉年龄的离线硬拒绝 | **改为质量提示** | 数据结构和时间因果正确即可导出；不重采样、不补帧，实际时序仍可追溯。 |
| D09 | Policy 每轮必须精确处于 arm 0.5° / hand HOME 容差 | **删除全局门槛** | 从有效、停稳的当前姿态启动。具体实验需要 HOME 时由操作流程明确执行 H。 |
| D10 | HOME 多候选、等价端点重选、近端/远端组合、排序去重、软间隙逃逸 | **缩减为单条经检查的路径** | 只检查当前关节位置到配置 HOME 的直接路径；拒绝时操作者调整姿态，取消通用脱困能力。 |
| D11 | 已验证内部数据的重复结构预检、不可达分支、失效配置 | **随调用链删除** | 外部边界保留必要检查；没有消费者的参数、文档和测试一起删。 |

D03 不等于删除所有 Pipe 处理：正常请求仍只能有一个在途回复；关闭时仍要尽可能读取真实故障，不得让未知旧回复进入下一轮。

### 2.3 明确保留的机制

| 机制 | 保留原因 |
| --- | --- |
| 主机 monotonic 时间与原始来源时间 | 防止缓存续期、跨时钟比较和虚假的数据同步。 |
| 共享 ring 的一致性版本号、相机 sequence、点云 source_camera_sequence | 保证读到完整快照及来源对应关系，属于数据一致性。 |
| 一个本地 `cancel_epoch` 与短锁 | 防止 S 已取消启动，而长初始化完成后重新清除 stop。 |
| emergency latch、stop、closing、owner fault | 分别对应急停意图、停止请求、关闭过程和真实设备错误，不能靠一个 UI 阶段替代。 |
| arm 最新值邮箱、停止确认与清邮箱顺序 | 防止上一段的目标在下一段恢复后执行。 |
| `reply_pending` 或等价的单一真实在途事实 | 防止把旧请求回复当成新请求回复；不是 run generation。 |
| 一个模型工作线程与 Future | 保持 load/warmup/reset/predict/close 的串行和线程归属，同时主线程能服务输入与手部反馈。 |
| 模型局部清理期限、必要的 close Future | 防止同一未完成任务在外层 close 再获得完整等待时间，避免 model.close 与 predict 并发。 |
| 有界 writer 队列、单 writer、临时目录关闭后发布 | 防止内存无限增长、多个线程写同一文件和半成品被当作完整数据。 |
| 关节限位、Ruckig、实际 IK/碰撞检查、必要标定 | 对真实运动有直接作用；本轮不改算法或删掉整个几何层。 |
| HOME 新鲜反馈和停稳/到位确认 | SDK 返回、命令已接收和机械系统实际到位是不同事实。 |
| `ObservationNeeds`、现有 frozen config dataclass | 已经轻量且有实际消费者；不改成泛化 schema 或字典配置体系。 |
| Raw 来源标识、单位、模态定义和 Canonical 当前结构 | 服务实验追溯及训练消费者；这些标识不是授权代际。 |

### 2.4 参考仓库的取舍

参考的是研究代码的组织方式，不把它们当作安全或时间正确性的证明。

| 仓库与核实版本 | 借鉴 | 不照搬 |
| --- | --- | --- |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | 直接的控制循环、采样/记录职责分离、少量状态支撑实际流程。 | 过时 chunk 最后动作重新使用、为时间网格复制观测、无界 join，不能替代本仓库真实时间与关闭语义。 |
| [T-Rex hardware_code](https://github.com/ZhuoyangLiu2005/T-Rex/tree/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code) | 明确的人为阶段切换、每段清空缓存、低频/耗时作为诊断信息。 | 不借鉴忽略控制错误继续执行、线程引用丢失或无界等待等做法。 |
| [Dexora](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783) | 短入口、直线资源生命周期、单实验脚本的易读性。 | 不用零图像/零反馈掩盖数据缺失，不忽略动作调用结果，不假设进程退出就等于硬件停稳。 |

参考代码入口见第 10 节。上述项目没有相同的多模态训练、RTC 和设备边界，不能只比较文件长度。

## 3. 最终时间与数据语义

### 3.1 时间戳的含义

| 字段/来源 | 含义与必须保留的规则 |
| --- | --- |
| arm / hand feedback timestamp | 当前实现接近主机发起反馈查询的时点，不宣传为硬件精确采样时刻。无效读取不能刷新时间。 |
| tactile aggregate / dense timestamp | 各自最后一份完整有效数据的时间；两个通道不能借另一通道更新而续期。 |
| camera timestamp / sequence | 新 RGB-D 帧的接收与序列事实；重复帧不更新成“刚采样”。 |
| point cloud timestamp / source_camera_sequence | 继承源 RGB-D；点云计算结束时间不能替代源时间。 |
| VR timestamp | 继续使用已接入 SDK 的接收时间语义；未核实 SDK 硬件时钟前不假设与主机可直接比较。 |
| control_ns / observation_ns | 本轮观测读取开始/读取完成；都不是 arm 与 hand 的同步执行时间。 |
| video PTS / RawReader.timestamps_s | 视频按名义 fps 编码；实际观测时间另存。相对时间先用整数做差，再转浮点。 |

统一时钟不等于统一采样网格，多路数据均新鲜不等于严格同步。当前任务不增加跨模态同步器、时钟校准协议或全局最大 skew 门槛。

### 3.2 新鲜度与发送边界

- 保留 `sample_is_fresh` 的正值、非未来、严格小于 TTL 语义；恰好到期即过期。使用当前已有 TTL 配置，不在没有实测延迟分布时任意调大或增加自动调节。
- policy history 接纳时检查来源，耗时映射后在发送前再次检查必要来源。这两个检查跨越真实耗时，不是重复防御。
- teleop 可合并同一轮映射前的重复 TTL 预检，但保留缺字段/形状/有限值等输入检查，以及映射后的新鲜度检查。
- 公共设备层继续检查有效命令、绝对限位、反馈年龄、取消/停止/急停、closing/fault 和目标期限。映射成功不能跳过这些边界。
- 不通过重复发送相同旧目标、重标时间戳或收到模型结果来刷新“进展”。
- arm owner 的租期保护生产者停更；hand 在父进程执行同步 SDK 调用，不能宣称已有同等独立 watchdog。native 调用阻塞的限制必须如实说明。

### 3.3 各模式的时间网格

| 模式 | 最终语义 |
| --- | --- |
| teleop / jog / replay | 逐轮执行；本轮超时则整体变慢，不补发欠账、不跳过 replay 行，也不新增严格 slot 约束。 |
| sync | 动作数组、游标、下次执行时刻；保持当前慢推理下的循环语义，不重新引入 async 的 slot 协议。 |
| async | 预测属于原观测锚点。按当前时刻裁掉过期前缀，可使用完整有效尾部；action_steps 是名义预取窗口，不强行截断剩余有效预测。 |
| RTC | 固定前缀和约定交接时刻属于算法承诺。错过交接、该槽缺少必需观测或映射越过承诺边界时结束该段。 |
| history | 最近 N 次被当前逻辑接纳的样本，允许不等间隔；首次仍需完整窗口。保留 sync 等待/结果就绪时的现有入窗行为。 |

async 中单次映射跨过名义 tick，不自动拒绝；仍必须满足原预测有效窗口、来源新鲜度和发送边界。禁止把过时预测的第 0 项重新锚定到“现在”。

RTC bootstrap 尚无既有前缀承诺，继续保留当前从就绪结果开始的行为；进入正式交接后不得把 RTC 悄悄降为 async。RTC prefix 保持模型动作表示，不能改成映射后的关节目标。

不修改 `1 <= A <= P`、async 预取范围、RTC `A + d <= P` 和 guidance 有限值等算法输入约束。它们与任意的实时频率硬门槛不同。

### 3.4 删除父进程重复预算

删除 Rollout 的：

- `wait_deadline_ns`、父层 `target_expires_ns` 镜像；
- `budget_limit/expired_budget/stop_budget` 及只为预算仲裁存在的 `stop_rejection` 分支；
- `PREDICTION_WAIT_TIMEOUT_NS` 和相关“无进展重试必须在两秒内终止”约定。

保留三个互不重复的边界：

1. **实验时长：** 一轮真正开始时建立一个 `run_deadline_ns`。循环和耗时映射后的副作用边界读同一个期限，不重新建立期限，不排序多个“结束原因”。
2. **设备目标租期：** 发送时传入当前既有 lease；arm owner 独立检查。即使主线程正在等 Future，也必须持续服务设备健康并发现 stop；不得清 stop、续发旧目标或继续等待新结果来恢复本轮。
3. **算法窗口：** async 原预测窗口及 RTC 交接承诺，按第 3.3 节执行。

明确接受的行为变化：

- 首次动作没有两秒预测硬超时，默认仍受现有 60 s 实验时长限制。
- 显式关闭实验时长限制后，在尚未发送任何目标时可持续等待有效预测，直到操作者停止或出现真实错误；这是配置语义，不再另外加无进展计数器。
- 首次动作已经发送后，目标过期导致 owner stop，即结束该段，不因模型稍后完成自动复活。
- 时长控制是协作式的；不能承诺中断已阻塞的 native 调用。
- 同时发生时长到期和普通停止时，不要求纳秒级原因排序；真实 SDK/模型错误、停止失败和急停仍需报告。

### 3.5 Raw 与 Canonical

**Raw 保留事实：** 状态与动作分别记录；动作表示 submission attempt，不表示硬件已经同步执行。保留当前 policy 等待行的无动作表示，不自动填前值、填零或过滤成示教数据。

**Canonical 放宽质量门槛：**

- 删除周期偏差超过 ±25% 的硬拒绝。
- 删除 `observation[i] >= control[i+1]` 的硬拒绝。
- 删除触觉来源年龄超过录制 TTL 的离线硬拒绝。
- 利用现有统计报告平均/最大控制间隔、网格偏差、读取耗时；必要时补充简单的最大来源年龄。异常质量给清楚 warning，不新增 QC 配置框架。
- 不重采样、不插帧、不补零、不改写源时间，不改变名义 dt 或训练特征含义。日志明确：可导出不代表数据质量适合当前实验。

**仍然拒绝损坏：**

- 空数据、字段缺失、形状/单位/dtype 不满足当前消费接口、必需标定缺失、RGB 帧数不匹配、必需数值非有限。
- 主控制时间和观测时间非正、各自倒退/重复，或 `observation < control`。
- 必需来源时间非正、晚于对应 observation、来源时间倒退；相机序列负值或倒退。
- 未完整发布的实际 episode；保留现有 basename 检查，不恢复扫描全部祖先路径的旧门禁。

保留当前完整 Canonical 模态集合和 teleop-only 导出范围。取消来源限制会涉及 policy 空动作行及动作语义，本轮不扩展通用转换器。Raw 的缺失值约定和 Canonical 输出结构保持当前接口，不借减法重构改变训练数据定义。

## 4. 最小运行状态与生命周期

### 4.1 取消与操作准入

保留 `SessionState` 作为轻量容器即可，删除其通用授权职责。目标状态仅包括实际有消费者的：

- 父进程本地 `cancel_epoch`、短临界区锁、quit 请求；
- 跨进程 stop 与 emergency 信号；
- 现有具体资源 owner 的 closing/fault/在途句柄。

主流程原有的“当前是否有 episode / teleop 阶段”继续决定 B/H/T/R 等操作是否可接纳，不另建全局 phase 镜像。一个操作被接纳时捕获本地取消计数，贯穿初始化、等待、准备和最终发送准入。

必须满足：

- 启动取消计数在启动键盘监听及长初始化之前确定；S/C/D/Q 后不能由迟到的初始化完成恢复动作。
- 清除 stop 只发生在明确的新操作准备边界；短锁内重查取消计数、quit、急停、closing/fault。不能无条件 clear。
- 内部正常停机不是用户取消，不随意推进 cancel_epoch；用户取消与 stop flag 的职责分开。
- 普通控制发送必须拒绝 stop。prepare/HOME 通过专用操作入口和本次取消计数进入，不能把旧 `required_safety_state=ARMED` 换名成另一种通用许可。
- emergency 立即置位且本会话不清除；先设置急停/stop，再进行可能竞争的锁操作。
- 不持键盘锁、状态锁或邮箱锁执行 SDK、模型推理、大数组读取、join 或长等待。
- 不把 cancel_epoch 写入共享目标、键盘事件、Raw 或 Pipe 请求；不得引入 command generation/lease token。

`RunEndReason` 只服务报表分类时可直接改为本地简明字符串，更新调用方；不建立新的跨层结束原因协议。无需为删除一个小 enum 机械制造大量格式迁移。

### 4.2 键盘与结束决定

- 保留快停回调：主线程在计算或等待时，S/Q/Esc 仍能先设置真实停止信号。
- 保留忙时不排队启动、按住开始键不能在返回 idle 后自动重启、释放后重新按下才启动等基本交互。
- 保留事件发布所需的短同步，不能为了删锁造成“已取消却重新开始”。
- 保存/丢弃在主线程决定结束当前段时确定一次。该次已消费事件中有 D，或此前已接纳 D，就丢弃；收尾开始后到达的 D 不追溯修改旧段。
- Q 不覆盖已接纳的 D；Esc/Ctrl+C 始终触发急停路径。无需给每个普通结束事件构造时间优先级。
- 键盘层不再了解 SafetyState，不负责授予动作权限。

### 4.3 正常结束与异常退出

| 情况 | 处理 |
| --- | --- |
| 运行中的 S/D/C、正常时长结束 | 停设备 → 完成本段记录处理 → 回收本段模型任务 → 返回待命。没有未回收任务才能接纳下一轮。 |
| Q / 达到 episode 数量 | 正常结束当前段，随后关闭会话资源。未开始的准备不计入正式 episode 数。 |
| 启动或在途 prepare/HOME 被取消 | 停止并结束本次会话，不继续恢复该 owner；干净的用户取消可返回正常退出码。 |
| 设备/模型/传感器实际故障、生命周期超时、关闭失败 | 停设备，尝试收尾其他已取得的资源，报告错误并非零退出；不自动重连或进入下一段。 |
| Esc / Ctrl+C | 急停 latch，尽力收尾，非零退出。 |
| 纯配置或 HOME 几何拒绝 | 给出具体原因。若尚无在途设备请求且已有停止已成功确认，可返回待命；否则按异常退出。不能借此恢复取消/超时的请求。 |
| 录制输入质量失败或队列满 | 按第 5 节结束当前段；健康 writer 可发布已接纳前缀。实际 writer I/O 错误走故障退出。 |

HOME 当前先处理手部再规划机械臂；几何拒绝不代表所有设备从未移动。不要在未核实设备依赖时改写这一顺序或宣称“拒绝必然零动作”。

正常停止复用必须保留：

1. 上一段发布目标的调用已返回。
2. owner 处理停止；父进程等待停止确认并读取故障。
3. owner 停止态不消费目标；父进程持邮箱锁清除残留目标。
4. 短临界区确认本次未取消、无急停/关闭/故障，再允许新操作。
5. owner 接纳生命周期工作时使旧 stopped 确认失效。

stopped 表示停止处理结束，不等于停止 SDK 必然成功；故障仍必须读取。arm 与 hand 顺序提交不是原子事务，不能增加虚假的“全设备提交成功”承诺。

### 4.4 资源收尾的归属

| 资源 | 唯一职责与边界 |
| --- | --- |
| robot / arm owner | 主循环结束先请求停；终止会话时先处理设备关闭，再等待非设备慢任务。停止失败不能被 disconnect 成功覆盖。 |
| recorder | episode 持有；一处 finish，停止接纳、关闭 writer 后发布或丢弃。真实存活线程保持可达。 |
| model worker | session 持有；正常段结束回收 prediction 后才复用；最终 close 与已排队模型工作串行。 |
| sensors | 取得进程句柄即登记，由原 owner 处理部分启动失败及退出。只对传感 worker 使用既有 terminate/kill 兜底，不复制到机器人。 |
| keyboard / audio | 由创建者释放；设备停机请求发出前不能提前丢失快停通道。 |

收尾顺序不要求所有资源共用一个超时。每个实际阻塞 I/O 的局部有界等待可以保留；不新增全局 deadline 对象。

同一未完成模型任务不得因进入外层 close 再获得完整 65 s。可保留 PolicyWorker 现有局部 cleanup_deadline 和 close_future：它们维护真实串行工作及一次等待预算，收益大于几行状态的成本。删除的是上层资源分类矩阵，不是这些完成证据。

Future 已完成但结果为 None，与 Future 未完成必须区分；模型自身抛出 TimeoutError，与等待超时也必须区分。结果/异常消费一次，不能因取消而吞掉同时发生的真实错误。

超时后不得清空真实句柄并声称 closed，不得并发执行 model.close 和 predict，不得转成 daemon thread 来隐藏未回收工作。排队的 close 可能等待 native 调用返回；日志与退出结果要诚实说明限制。

错误处理放在少数确有停止、释放或报告责任的边界。保留主错误，附加错误直接记录；允许重复日志，不新增异常身份缓存和错误去重台账。正常段不重复 close 设备；失败段不在多层重复完整收尾。

## 5. 录制与实验结果

### 5.1 录制只有“本段开启 / 本段未开启”

- Policy 的 `run_policy(record_raw=False)` 成为默认。
- `scripts/run_policy.py` 用显式 `--record-raw` 开启，删除旧 `--no-record` 及对应调用和文档，不保留反向别名。
- 未开启 Raw 时，只启动/读取 policy 本身需要的模态，不为记录额外要求 RGB-D、触觉或其标定。实际 policy 需要的仍须严格满足。
- teleop 当前默认采集行为保留；本轮不额外添加录制 profile 或模态注册表。
- 一轮开始确定一次 `capture_needs = control_needs ∪ record_needs`；不开录制则为 control_needs。本段不动态切换。

### 5.2 失败即结束当前段

`EpisodeRecorder.append` 在录制质量不合格或队列满时，把具体原因返回调用者，调用者立即结束当前段；实际 writer I/O 异常继续抛出。使用简单返回值和现有异常即可，不新增录制状态机。

删除 freeze 后继续运动、恢复录制、切换 needs 等分支。数组脱离共享缓冲后才能入队；进入队列后生产者不能继续修改。

结束顺序：

1. 请求停止运动；不得先阻塞等 writer。
2. 固定本段保存/丢弃意图和结束原因，停止接纳新行。
3. 在原有局部预算内发送结束标记并等待 writer；关闭成功才判断发布。
4. 已接纳有效前缀全部写完、writer 无错误、行数大于零且未要求丢弃，则发布；记录导致截断的原因。
5. 零行不发布；已明确丢弃且 writer 已退出可删除临时产物。
6. writer 仍存活、写入或 close 失败，不发布、不假装完成；保留临时目录和真实句柄，退出会话并报告路径。

不要把“队列满/来源质量不合格”直接写入 writer 的 I/O error 字段，否则会提前终止健康 writer 并损失已经接纳的前缀。发布的是已记录前缀，不保证包含最后一次实际提交；Raw 仍不提供执行确认。

已有 `recording_truncated_reason` 可作为最终数据元信息保留，取消的是持续运行中的冻结协议。`_FinishRequest`、finish 幂等保护等小结构有真实用途时保留，不为行数制造替代机制。

### 5.3 简化结果文件

保留 `run_config.yaml` 和 `session_result.json`，默认不录 Raw 也能查看实验结果。

结果文件只需保留已有字段中有直接用户价值的部分：

- session 的开始/结束时间、结束原因；
- episode 序号、实际开始时间、持续时间、结束原因/简短说明；
- 若有录制，实际发布路径、写入行数和截断原因；
- 实际错误的阶段和消息。

删除 shutdown 分类矩阵、冗余 outcome 推导及其专用控制分支。无错误通过正常路径返回 0；故障、急停或未完成关闭返回非零。不要通过结果文件反向判断设备是否可复用。

沿用现有字段名即可，不需要为报告另建新 schema。新增实现若必须调整某些字段，同步 README 和直接消费者，不做旧格式适配器。

## 6. HOME 与启动门槛

### 6.1 Policy 可以从当前有效姿态启动

删除 `START_ARM_TOLERANCE_DEG` 和 `start_requires_home_pose` 的全局精确 HOME 准入。仍要求：

- 必需反馈和观测可用、新鲜且有效；
- 已完成当前设备的准备和停稳；
- 当前姿态满足既有有效范围，动作经过当前映射和发送边界；
- 用户明确发起新一轮，未取消、无 stop/急停/故障。

保留 H 以及实际需要的启动手部 HOME/触觉 reset 流程；本轮不改变触觉校准的物理前提。策略可能只在 HOME 附近训练，文档提醒操作者按具体实验布置初始化，但不强加给所有策略。

保留 replay 当前与第一行目标的粗起点差异检查；它防止明显跳变，与 Policy 的统一精确 HOME 门槛不同。不得恢复旧 replay ramp、私有反馈监控或在线几何重映射。

### 6.2 HOME 单路径

在现有 `HomePlanner` 原地实现：

1. 读当前有效反馈和配置 HOME，保留形状、有限值和关节范围检查。
2. 路径固定为 current → configured home，不重选等价终点。
3. 沿用当前插值分辨率和现有几何模型，检查路径采样点的限位、工作空间、碰撞和桌面软间隙。
4. 所有点均满足要求才提交；已在容差内也须保持有效状态检查。
5. 路径拒绝则报告具体原因，不自动换路线、不生成未标定的中间路点，不自动 jog。
6. 执行继续使用原设备速度/加速度边界和新鲜反馈到位确认。

删除仅为多候选存在的 PROXIMAL_MASK、candidate 排序、去重、局部姿态缓存和 allow_table_soft_escape/best_clearance 规则；有其他调用方的通用工具不得连带删除。

保留 `table_plane_abcd` 的运行配置归档消费者。保留一次 arm HOME 总预算：等待反馈、规划、所有已提交动作与到位确认共享同一期限，不能按阶段重新续期。hand HOME 继续使用自身实际操作预算，不虚构一个跨设备全局计时器。

代价必须写入 README：HOME 不再承担从任意姿态规划回零或从软间隙脱困；直接路径不通过时需人工调整。离散几何检查不是连续碰撞证明，SDK 返回也不是到位证明。不得把单路径简化成未经检查的 SDK 直发。

## 7. 实施任务与顺序

以下任务已完成代码、调用方与相关离线测试更新；勾选不代表已通过真机验收。

### T0 — 确认当前基线和影响范围

- [x] 阅读当前 `AGENTS.md`、本文及相关入口；检查工作区已有改动，保留用户工作。
- [x] 如果 HEAD 已变化，核实本文涉及的符号与行为；已经实现的项目不重复重写。
- [x] 记录实际可用的离线依赖及目标测试基线；不把旧任务书测试数量当成本轮结果。
- [x] 按第 2 节建立简短实施清单即可，不再生成另一份通用规划文档。

### T1 — 时间语义与单一实验时长

主要文件：`deployment/rollout.py`、`deployment/schedule.py`、`config/deployment.py`、`runtime/observation.py`、`utils/timing.py`。

- [x] 删除 D01 的字段、常量、仲裁函数及无消费者分支；直接用单一 episode 截止时刻。
- [x] 等待模型时也轮询 operator、服务反馈、检查 owner stop/fault；耗时映射后在发送前再次检查时长和必要来源。
- [x] 保留 async 有效尾部、RTC 交接、sync 游标和当前 history 行为。
- [x] 删除只验证旧等待预算竞争的测试；改测第 8 节 A01–A07。
- [x] 更新注释：没有新采样或新目标，就没有“续期”。

完成条件：父进程不再维护目标租期镜像或 prediction wait deadline；慢首推理可继续等待，设备一旦停机不会因预测完成恢复。

### T2 — 先收缩异常恢复，再移除授权状态机

主要文件：`robot/xarm7/process.py`、`robot/xarm7/worker.py`、`robot/xarm7/protocol.py`、`robot/system.py`、`runtime/lifecycle.py`。

- [x] 将在途 prepare/HOME 的取消、超时和技术失败收敛为停止并关闭会话；删除为下一操作恢复的旧回复清理协议。
- [x] 正常请求保留一个 Pipe 在途事实和正常回复消费；异常关闭仍读取故障，不能让取消遮盖 SDK 失败。
- [x] 保留 owner 停止态不消费邮箱、停止确认、锁内清邮箱和旧确认失效顺序。
- [x] 避免段内故障分支和最外层各执行一套完整设备关闭；保留真实 stop/close 失败传播。
- [x] 不把故障或未完成句柄清成 None 来满足“已关闭”。
- [x] 更新 arm/replay 相关异常测试，明确取消 prepare/HOME 后退出而非重试。

完成条件：故障路径没有返回正常复用的隐式分支，正常 S 后仍可干净开始下一段。

### T3 — 取消、键盘与直接操作流程

依赖 T2。主要文件：`runtime/state.py`、`runtime/keyboard.py`、`runtime/home.py`、`runtime/jog.py`、`teleop/loop.py`、`deployment/session.py`、`deployment/rollout.py`、`replay/run.py`、`robot/system.py`。

- [x] 删除 `SafetyState`、通用授权方法、`required_safety_state` 参数链和无消费者导入。
- [x] 各入口用实际操作流程及取消信号约束动作，正确区分普通发送和显式 prepare/HOME。
- [x] 保留一个本地取消计数及最终清 stop 的短锁检查；覆盖启动早期取消。
- [x] 删除结束阶段回扫键盘队列/追认 D；一次决定保存意图。
- [x] 保留忙时键盘行为、快停、急停 latch 与 Ctrl+C 收尾。
- [x] 对纯配置/几何拒绝只保留明确无在途请求的简单返回，不恢复异常请求。

完成条件：没有换名后的授权状态机；取消与重新开始的关键竞态仍由真实测试覆盖。

### T4 — 清理归属与简明结果

依赖 T2、T3。主要文件：`deployment/session.py`、`deployment/rollout.py`、`deployment/inference.py`、`runtime/lifecycle.py`、`ipc/sensors.py`。

- [x] 正常 episode 收尾与会话最终关闭职责分开，设备先停，模型/录制等待随后执行。
- [x] 删除 session pending/completed/failed 汇总、shutdown matrix 及其专用报表推导。
- [x] 模型继续单线程串行；保留实际需要的 Future、close Future 和局部清理预算。
- [x] 只在已确认资源结束后移除句柄；部分 start 失败取得的资源仍能关闭。
- [x] 简化结果为第 5.3 节内容，真实错误非零退出，无需错误身份去重。
- [x] 删除重叠的通用 cleanup 包装；只保留有独立资源职责的函数。

完成条件：沿入口能指出每个资源由谁收尾；没有多层重新等待同一模型任务、并发模型 close 或虚假的关闭成功。

### T5 — 录制单流程与 Policy 默认轻运行

依赖 T3、T4。主要文件：`data/recorder.py`、`data/raw.py`、`teleop/loop.py`、`deployment/rollout.py`、`deployment/session.py`、`deployment/run.py`、`scripts/run_policy.py`。

- [x] Policy Raw 默认关闭，CLI 改成 `--record-raw`；更新配置快照和示例。
- [x] 录制质量失败/队列满结束当前段；删除冻结后继续控制与动态 needs 分支。
- [x] 质量终止原因与 writer I/O error 分开；健康 writer 排空已接纳前缀后可发布。
- [x] 保留有界队列、真实线程、一次 finish、关闭后发布、零行不发布和失败保留 staging。
- [x] 默认不录制时不额外要求仅用于归档的传感器/标定。
- [x] 使用真实 writer 线程的目标测试验证前缀、失败及关闭时序。

完成条件：没有“本段仍运动但录制已冻结”的隐含模式；默认策略评估只承担模型实际输入成本。

### T6 — 放宽离线质量门槛

可在 T0 后独立实施；最终与 T5 的元信息对齐。主要文件：`data/export.py`、`data/raw.py`、`tests/test_recording_export.py`。

- [x] 将三项时间质量硬拒绝改为 warning/简单统计。
- [x] 显式保留主时间严格递增和各来源时间因果/非倒退检查，防止删除旧门槛时顺带漏掉原先间接成立的不变量。
- [x] 保留数值、结构、模态、RGB 帧数、标定和实际发布边界。
- [x] 不改变 Canonical schema、名义 dt、timestamps_s 含义，不扩大到 policy Raw。
- [x] 更新导出测试：抖动可过，时间损坏仍失败；用真实容器测试验证结果。

完成条件：真实时序不均匀不再单独阻止导出，损坏数据仍被清楚拒绝。

### T7 — HOME 单路径与取消精确 HOME 启动门槛

依赖 T2、T3。主要文件：`control/homing.py`、`runtime/home.py`、`deployment/rollout.py`、`config/deployment.py`、`deployment/preflight.py`。

- [x] 删除 Policy 全局精确 HOME 检查及无消费者常量，保留有效反馈和准备停稳。
- [x] HOME 按第 6.2 节原地缩减，保留路径采样检查和真实收敛确认。
- [x] 删除多候选专用函数/导入/测试；检查共用 helper 和桌面标定仍有消费者。
- [x] 测试直接路径接受/拒绝、预算贯穿、反馈不续期、SDK 返回不等于到位和取消。
- [x] 文档明确 HOME 能力收缩及策略初始姿态由具体实验负责。

完成条件：HOME 行为确定且容易阅读，不依赖未提供的人工路点，不偷偷降低路径检查。

### T8 — 一致性清理、文档与最终验证

依赖 T1–T7。

- [x] 全调用链删除失效 API、常量、CLI 和只服务旧协议的测试，不留下死分支。
- [x] 清理只有在已证明不可达时才删除的分支，例如 policy 当前明确要求启用 hand 的路径；不能破坏其他入口的禁用手部能力。
- [x] 更新 `README.md` 的启动、停止、录制、HOME、导出和失败后重启说明。
- [x] 为上一阶段任务书增加清楚的历史说明/本任务书链接，不改写历史测试事实。
- [x] 按第 8 节运行针对性离线验证，检查 diff 无无关格式化、依赖或新框架。
- [x] 在本文实施记录中勾选实际完成项，记录实际验证及未验证限制；不把任务书发布当作代码实施完成。

## 8. 验收标准

优先修改现有回归测试。测试应覆盖外部行为和真实边界，不把新私有函数逐行复制成断言。

| 编号 | 场景 | 必须结果 |
| --- | --- | --- |
| A01 | 首次推理超过旧 2 s 等待期限，但未到实验时长 | 不因旧预算停止；仍能接收 S/Q/Esc 并服务设备健康。 |
| A02 | 无首个动作且显式无时长限制 | 不存在隐含预测重试计数或 2 s 超时；用户停止仍有效。 |
| A03 | 已发送目标后 owner 因目标/反馈到期置 stop，模型尚未完成 | 当前段结束；迟到结果不能清 stop 或发起下一轮动作。 |
| A04 | 映射耗时跨过实验结束时刻 | 不再提交该命令；使用同一实验期限，不创建第二预算。 |
| A05 | async 结果部分过时、映射跨名义 tick | 裁剪过期前缀；在原有效窗口及来源有效时允许提交；全过时结果不重新锚定。 |
| A06 | RTC bootstrap、正常交接、错过交接、该槽缺观测 | 保持当前 bootstrap 语义；正式交接不降级为 async；失约结束。 |
| A07 | 慢 sync、teleop/jog 计算略超周期、replay 300 行稳定慢于名义周期 | 变慢而非补发；replay 顺序处理全部行；不重新引入时间网格失败。 |
| A08 | history 有不等间隔；sync 正在等待预测 | 不清空有效历史、不重复入窗、不因本次重构改变模型输入采样语义。 |
| A09 | 启动、模型加载、传感就绪、prepare 中途取消 | 取消计数贯穿；迟到完成不能开始动作或无条件清 stop。 |
| A10 | S 后立即 B，忙时按住 B/H 等键 | 当前操作先结束；旧按键不在待命时自动触发；新开始必须满足正常恢复顺序。 |
| A11 | 停止后旧邮箱目标、旧 stopped、旧 Pipe 回复仍存在 | 不执行旧目标、不把旧确认或回复当成本次成功；异常请求不复用。 |
| A12 | 正常 S 结束一段，随后新一轮 | 设备/模型可正常复用，无泄漏；已完成段不重新 close。 |
| A13 | HOME/prepare 取消与 SDK fault 同时出现 | 退出会话，报告实际故障；不能把 fault 转成成功的用户取消。 |
| A14 | 停止 SDK 失败，随后 disconnect 成功 | 仍报告停止失败并非零退出；stopped 不被视作物理停稳证明。 |
| A15 | 键盘取消与清 stop 并发；Esc/Ctrl+C 在收尾期间到达 | 不丢取消、不死锁，急停先置位；不持短锁做阻塞 I/O。 |
| A16 | 在途模型任务、已完成返回 None、已完成抛 TimeoutError、真正未完成 | 正确区分；不并发 close，不吞结果异常，不在外层续清理预算。 |
| A17 | 模型或 writer 等待超时 | 真实句柄仍可达，禁止下一轮；不声称资源已结束或解释器必定退出。 |
| A18 | 传感器/recorder 部分启动失败 | 已取得的资源仍被清理；设备先停止，错误透明。 |
| A19 | 录制队列满或仅录制输入失效 | 立即结束本段；健康 writer 写完已接纳前缀；不继续控制、不填零、不静默丢中间行后恢复。 |
| A20 | writer 写入/close 失败、仍存活、零行 | 不发布完整 episode；错误保留 staging，零行正常完成时无发布。 |
| A21 | D 在结束决定前/后到达，Q 与已接受 D 同时存在 | 前者丢弃；后者不追溯；Q 不把已确定丢弃变成保存。 |
| A22 | Policy 默认启动与显式 `--record-raw` | 默认仅需要模型输入；显式录制才要求完整录制模态；CLI/API/README 一致。 |
| A23 | Canonical 周期抖动/间隔缺口、readout 跨下一 control、触觉来源较旧 | 若其他结构和时间因果有效，可导出并提示质量；不补帧、不改时间。 |
| A24 | control/observation 重复或倒退、来源未来/倒退、RGB 数量错误、数值非有限 | 清楚拒绝，不误归为允许的周期抖动。 |
| A25 | 导出已有正常 teleop Raw，policy Raw 存在空动作行 | 前者保持当前输出 schema；后者仍明确不在本轮 Canonical 范围。 |
| A26 | Policy 初始姿态偏离 HOME，但有效且已停稳 | 不再被 0.5° HOME 规则拒绝；实际动作发送边界仍生效。 |
| A27 | HOME 直接路径安全/穿越障碍/处于软间隙，SDK 立即返回 | 只接受检查通过的路径；不自动绕行/逃逸；必须等新鲜实测到位。 |
| A28 | 重复相机帧、CRC 失败手反馈、慢点云计算、ring 写读竞争 | 不刷新旧时间；点云继承源序列；不读到撕裂样本。 |
| A29 | 所有正常或故障退出 | 无通用授权状态和 shutdown 分类矩阵；日志、结果、退出码与真实情况一致。 |

### 8.1 对现有测试的修改指引

- `tests/test_control_progress.py`：替换无进展预算/预算平局断言；保留 async 有效尾部、RTC 交接、慢 sync、映射后新鲜度与设备先停测试。
- `tests/test_input_resources.py`：删除迟到 D 追认和纯结果分类断言；保留忙时按键、启动取消、真实 Future 状态、单次清理预算、部分启动失败测试。
- `tests/test_arm_recovery.py`：正常停止复用继续测；异常恢复用例改为禁止复用/终止会话。保留旧邮箱、旧确认、故障和关闭错误测试。
- `tests/test_recording_export.py`：freeze 后继续控制改为立即结束；队列有效前缀及真实 HDF5/video/Zarr 发布仍测；放宽周期门槛但保留时间损坏测试。
- `tests/test_home_snapshot.py`：保留总预算、来源不续期、到位反馈；新增/调整单路径与启动门槛测试。
- `tests/test_replay.py`：保持已有顺序慢速回放、粗起点、直接目标、无额外输出、公共发送边界等成果；只适配被删除的授权 API 和异常生命周期。
- 仅为本轮实际变化补测试，不建立覆盖全部异常排列的组合框架，也不通过删除真实边界测试取得“全绿”。

### 8.2 验证命令与报告要求

在已有完整依赖的开发环境，从仓库根目录运行。首先运行修改对应的文件；完成关联修改后，再运行这些现有相关回归一次：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_control_progress.py \
  tests/test_input_resources.py \
  tests/test_arm_recovery.py \
  tests/test_recording_export.py \
  tests/test_home_snapshot.py \
  tests/test_replay.py
```

CLI 的 `--help`、`--print-config` 只做无设备检查；不得为了“验收入口”真正启动运行流程。若使用 ruff，必须加 `--no-cache`；其他工具同样遵守 AGENTS 的缓存约束。

依赖缺失时明确记录阻塞的测试和缺失模块，不能把 AST/compile 检查称为行为测试通过。审查环境此前缺少 pytest、h5py、zarr、av；历史验证和未来执行环境以实际检查为准。

最终人工检查：

- 没有运行代际、通用授权协议或全局资源状态矩阵以新名字复活。
- 没有为删除预算加入新重试次数、watchdog 线程或统一 deadline 类。
- 没有为了放宽约定删除停止、真实故障、时间因果、数据完整性边界。
- 没有为简化模型生命周期把 predict 移到主线程，阻塞现有手部反馈服务。
- 没有新增无需求配置/兼容层/依赖/测试基础设施。
- 未生成仓库内工具缓存，diff 只包含本轮相关内容。

真机后续验收由操作者在合适布置下独立进行：确认普通停止和急停、重新开始、录制前缀、单路径 HOME、慢推理和断流表现。未执行这些检查时，最终报告明确写“未做真机验证”，不将离线通过等同于物理安全证明。

## 9. 完成定义与实施记录

只有代码、测试、配置、调用方和使用文档一致，才算完成。不能只删符号，也不能保留旧行为后宣称“结构已简化”。

### 初次实施记录（后续审查前）

- **实施基线：** `main` / `c53f4425ef42d7fde693f83c1d8929eb3a93381c`。相对审查基线 `025451c` 仅新增本文；开始时工作区干净。本轮为工作区改动，未创建提交。
- **完成范围：** T0–T8 的代码、调用方、CLI、相关测试和 README 已更新；旧任务书增加历史说明，未改写旧验证事实。
- **删除机制：** 父层预测等待预算、目标租期镜像和仲裁；SafetyState 及通用授权 API；prepare/HOME 异常恢复回复协议；session shutdown/outcome 汇总；结束后回扫键盘追认 D；录制 freeze/动态 needs；HOME 多候选、等价端点及软间隙逃逸；精确 HOME 起点门槛及无用常量。
- **保留边界：** 来源时间与发送前新鲜度、取消计数和短锁、stop/emergency/closing/fault、设备租期、停止确认与清邮箱、单个在途 Pipe 回复事实、单模型线程及 Future/局部清理期限、有界 writer 与关闭后发布、几何检查和实测到位确认。Raw/Canonical 训练接口未改。
- **收尾归属：** 正常段完成 writer/prediction 后才返回待命；故障或退出保留段内资源，由会话入口先关闭设备，再完成 writer 和模型关闭。删除重复的段内设备 close 与 teleop close 包装。模型关闭未完成直接记错误并非零退出，实际 Future 保持可达。

### 实际源码约束及处理依据

- `data/export.py::validate_export_episode` 原先通过周期和跨周期拒绝间接约束主时间顺序。本轮放宽质量门槛时显式增加 control/observation 各自严格递增，防止损坏数据被误放行。
- `robot/xarm7/worker.py::operation` 原先将 owner 的 `_OperationTimeout` 排除于 fault；该区别只服务异常后恢复。现在生命周期超时与其他技术失败都终止会话，删除专用异常，保留共享 fault 与最终 close 的晚到故障读取。
- `recording_issue` 对来源缺失/过期返回质量原因；实际相机 worker 退出抛出故障。依据第 4.3 节，进程死亡不能被质量截断掩盖为可继续复用。
- `control/constraints.py::wrap_nearest_equivalent` 仍被 IK 和动作映射使用，仅删除 HOME 内的调用；`table_plane_abcd` 归档和共用插值职责保留。
- 保留小型 RunEndReason 报表枚举、recorder finish 幂等保护，以及真实模型关闭 Future；没有新增状态框架或兼容 API。

### 实际验证

环境：`/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`（Python 3.10.20）。系统 PATH 的 `python` 不存在，但既有 Conda 环境依赖可用，无需安装新依赖。

- 修改前六个指定文件：**188 passed in 1.37 s**，仅作为旧代码基线。
- 初次实施六个指定文件：**204 passed in 1.42 s**；使用 `PYTHONDONTWRITEBYTECODE=1` 和 `-p no:cacheprovider`。
- `python -B -m ruff check --no-cache --select F dexmani_real scripts tests`：通过。
- Policy、collect_teleop、replay、keyboard_teleop、export_zarr 的 `--help`：通过。Policy 与 collect_teleop 的 `--print-config`：通过；均未进入设备运行。
- 新增/调整验证涵盖慢首推理、无隐藏等待预算、映射跨实验期限、owner stop、固定录制 needs、质量失败结束、取消在途请求保留归属并禁止复用、急停与清 stop 并发、正常/异常收尾顺序、迟到 D、不精确处于 HOME 的启动、单路径几何拒绝、主时间损坏拒绝、旧触觉及周期抖动警告，以及 Raw 显式开启。
- 真实 writer 线程和现有 HDF5/video/Zarr 容器回归继续通过；测试使用 fake 设备，不涉及真机或历史实验目录。

### 未验证与使用变化

- **未做真机验证**，未加载真实 checkpoint 进行硬件策略实验，也未执行真实 HOME/servo/hand 动作。停止延迟、急停、多轮重新开始、慢推理/断流、录制前缀及单路径 HOME 的物理表现需操作者另行验收。
- Policy Raw 默认关闭，显式 `--record-raw` 开启；不再接受 Policy `--no-record`。teleop 默认录制不变。
- 首推理不受旧 2 s 期限约束；已有目标仍受设备租期约束。异常准备退出会话，正常 S 仍可多轮复用。
- 录制质量失败立即结束本段，健康 writer 可发布有效前缀；离线时间抖动改为提示，不补帧、不改源时间。
- Policy 可以从有效停稳姿态开始；HOME 只接受经检查的直接路径，失败需人工调整。离散几何检查和离线回归不构成物理安全证明。


### 后续全链路审查与清理

本次在上述未提交改动上继续审查，保留初次验证结果作为历史事实。重点沿 Policy/teleop、设备 owner、录制/导出、脚本及文档检查上下游；未增加依赖、兼容 API 或新的控制状态。

已修复及简化：

- **取消分类：** Policy/teleop 的 prepare/HOME 只把当前操作取消计数已改变或 quit/estop 作为干净用户取消。意外 OperationCancelled 保留原异常并非零退出；teleop 的启动取消捕获限定在初始化阶段，历史正常 S 不再导致运行中故障被吞掉。
- **设备异常边界：** arm resume 的停止确认、清邮箱、最终清 stop 也纳入 request_operation 的异常退出边界，取消或超时后置 closing，不返回故障 owner 的复用路径。
- **实际传感故障：** 删除 required_names 镜像清单，健康检查依据实际持有的进程，录制专用相机不再获得故障豁免。关闭时确认退出码为 0 即正常回收；本次关闭主动尝试 terminate/kill 且退出码对应已尝试的 SIGTERM/SIGKILL 时只记 warning，不单独使会话失败。其他非零退出、清理调用异常及未确认退出仍报错，不清除已有会话错误。已正常结束的静态参数相机从 owner 列表移除。
- **HOME 调用链：** owner 仅接受空路径或 current/HOME 两端点，直接向配置 HOME 提交一次，删除任意多路点循环和重复终点参数。模式切换后实际查询反馈，复核起点与速度，避免 TTL 内旧缓存掩盖漂移；父层直接按既有 1° 分辨率采样，删除无其他消费者的通用多路点插值 helper。
- **启动姿态：** 删除精确 HOME 门槛后，在 Policy 首次观测准入补齐实测 arm 的配置关节范围检查；shape/finite 继续由 SDK 读取边界负责，未增加内部重复校验。
- **录制完整性：** 控制主错期间的传感等次错只报告，不再污染 writer I/O error 而中止健康前缀。recorder.finish 请求和期限建立过程中的中断纳入现有异常边界，重入不再误报完成；append 的 str/None 返回契约补齐。正常完成的 Rollout 释放 recorder，避免最终会话收尾重复 finish/提示音；失败句柄仍保留。
- **死接口与文档：** 删除 Rollout 无消费者的 keyboard 参数；清理旧授权措辞和重复临时变量。README 移除双期限/平局规则、合并重复录制和导出说明，明确无手入口、HOME 拒绝后状态、Policy 平面归档和音效错误处理。RawReader.timestamps_s 来自 Raw，Canonical 保持现有固定 13 模态，不增加时间数组。
- **哈希及版本机制：** 此影响范围内未发现冗余 SHA-256/内容哈希校验或旧格式 reader。保留实际追溯用 source_id、设备 CRC、ring 一致性版本及来源序列，不把不同职责的正确性边界机械删除。

本次实际验证（real_robot Conda 环境）：

- 六个相关回归文件合计 **224 passed in 1.52 s**；真实 writer、HDF5/video/Zarr、取消与故障传播、HOME 模式后反馈和上下游 API 回归均包含在内。使用 `PYTHONDONTWRITEBYTECODE=1`、`-p no:cacheprovider`。
- mypy 检查相关 **39 个源码文件通过**；使用 `--no-incremental --follow-imports=skip --ignore-missing-imports`，缓存目录在 `/tmp`，范围为现有类型注解检查，未启用 `--check-untyped-defs`。
- ruff `--no-cache --select F dexmani_real scripts tests`：通过。
- 五个相关 CLI 的 `--help`、Policy/collect_teleop 的 `--print-config`：通过。
- 在 `/tmp` 复制当前 Python package 与 pyproject，使用 `pip wheel --no-deps --no-build-isolation --no-cache-dir` 构建成功，检查 wheel 包含相关源码及配置包数据；临时构建目录随验证删除，未在仓库生成构建产物。
- 最终 diff 检查和旧接口/引用搜索通过；未新增仓库内 pytest/mypy/ruff 缓存。

本次仍未连接真机、执行真实运动或加载真实 checkpoint 做策略实验；物理停止、急停、断流、重新开始和单路径 HOME 仍由操作者独立验收。离线和构建检查不代替这些验证。

## 10. 核查入口与参考源码

### 10.1 本仓库

以下链接固定到审查基线，实施时以当前分支为准：

- [取消与原授权状态](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/runtime/state.py)、[键盘](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/runtime/keyboard.py)、[生命周期](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/runtime/lifecycle.py)。
- [Rollout 时间与阶段](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/deployment/rollout.py)、[async/RTC 调度](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/deployment/schedule.py)、[模型线程](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/deployment/inference.py)、[会话](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/deployment/session.py)。
- [arm 父进程](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/robot/xarm7/process.py)、[arm owner](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/robot/xarm7/worker.py)、[公共设备边界](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/robot/system.py)。
- [观测](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/runtime/observation.py)、[时间工具](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/utils/timing.py)、[共享快照](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/ipc/ring.py)。
- [录制](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/data/recorder.py)、[Raw](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/data/raw.py)、[Canonical 导出](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/data/export.py)。
- [HOME 规划](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/control/homing.py)、[HOME 执行](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/dexmani_real/runtime/home.py)、[Policy CLI](https://github.com/haoyangzhanglab/dexmani_real/blob/025451cc5ae3e1c77be221f50273c296dbf26769/scripts/run_policy.py)。

### 10.2 外部参考

- ManiUniCon：[主入口](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)、[robot](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py)、[chunk wrapper](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/act_wrapper/chunk_wrapper.py)。
- T-Rex：[teleop 主流程](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/main_teleop.py)、[writer](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/data_writer.py)、[async eval](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/eval/eval_trex_async.py)。
- Dexora：[推理入口](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py)、[hand forwarder](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/xhand_forwarder.py)。
