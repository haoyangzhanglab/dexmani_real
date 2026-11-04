# XHand 轻量化与偶发异常容忍任务书

状态：T0–T4 已实施，代码、调用方、配置、离线行为测试及文档已同步；真机运动、清零与时延验证未执行。

- 仓库：`haoyangzhanglab/dexmani_real`。
- 源码审查基线：`main` / `c4ded28b1078adefbd017f4c546b6a132a91d29b`。
- 编写日期：2026-11-04。
- 定位：个人真实机器人研究项目；直接、易改、少状态、少配置。
- 用户明确要求：继续使用固定的全模态采集与全模态导出，不根据策略更换采集字段、导出字段或数据格式。

## 1. 目标与范围

让偶发 CRC、触觉缺帧和过流告警成为可容忍的设备现象，减少无意义的整机中断；解除手部操作与会话、机械臂、录制质量之间不必要的耦合。保留能够直接发现错误输入、真实失联和错误数据的少量检查。

实施顺序：调整异常处理；贯通固定全模态录制；简化 HOME、清零及会话依赖；最后核实 SDK 反馈复用的可行性。

本任务按当前源码原地修改，不重写机器人运行时。重点涉及 `robot/xhand/`、`robot/system.py`、`config/xhand.py`、`runtime/home.py`、`runtime/observation.py`，以及实际受影响的 teleop、deployment、录制、导出、诊断脚本、测试和 README。

实施前阅读 `AGENTS.md`，检查当前分支、工作区及基线后的修改。保留 `c4ded28` 已完成的录制错误传播、收尾和标定修复。旧两份 research_simplification 任务书是历史记录；本轮 XHand 行为以本文和最新用户要求为准，不照搬其中已被替代的严格约定。

本轮不引入手部独立进程、额外 I/O 线程、ZMQ、设备接口基类、模式状态机、命令结果协议、自动重连、通用重试器、质量管理平台或可变模态数据集。不以减少行数为验收目标。

## 2. 已核实的现状与设计取舍

| 现状 | 本轮决定 |
|---|---|
| `XHand._send_action()` 将一次写 CRC 转成异常，Robot 随即置全局 stop | 周期绝对位置目标容忍偶发写 CRC；下一控制周期发送最新目标 |
| `feedback_max_age_s=4/30` 同时约束关节和必需触觉 | 普通手部反馈容忍窗口放宽到 0.5 s；录制不再以触觉年龄裁决是否继续 |
| Teleop/Policy 的 recording_issue 可因触觉过期结束运动 | 删除仅由录制触觉年龄导致的停段链路 |
| Raw 已有两类触觉及独立来源时间，缓存已经实现逐通道保持 | 复用；不增加逐指/逐 taxel mask 和第二套缓存 |
| Canonical 已固定 13 字段，允许旧触觉来源，已有最大年龄统计 | 保持这些接口；不重造导出器，不按策略筛模态 |
| hand HOME 要先 `arm.resume()` | 机械臂保持停止，手部执行独立 HOME |
| 启动按消费者需求自动 HOME、清零 | 取消通用隐式动作，使用已有 H/T 等显式入口 |
| 驱动 connect/disconnect 接收 SessionState，清零维护多种进度字符串 | 去除会话依赖；清零仅保留一次操作结果 |
| 普通发送与停止采用不同 CRC 策略 | 继续区分；停止的小型有限重试保留 |

不要误删：Robot 组合校验后调用的是 `hand._send_action()`，当前热路径没有再次走公开 `hand.send_action()` 的校验。观测检查针对模型或映射实际使用的样本，设备检查针对最新反馈，这两者也不必然重复。

## 3. 明确的异常处理行为

### 3.1 周期绝对位置目标

当前 Mode 3 的 `position` 是绝对关节目标。本次回执 CRC 不能证明目标未送达，也不必然要求整机停止。重复绝对目标不会像重复增量指令那样直接累计同一位移；这项取舍只用于本轮已知的位置发送路径。

| 情况 | 实现行为 |
|---|---|
| 读 CRC `1501070` | 丢弃该帧，计数并限频告警；返回无新样本；所有来源时间保持原值 |
| 单次或短串写 CRC | 记录本次发送未确认；不置全局 stop，不 sleep、不排队补发旧目标；下一周期发送最新目标 |
| 连续 5 次位置发送均为 CRC | 第 5 次按通信故障进入已有停止/收尾路径 |
| 一次允许的非 CRC 命令发送返回 | 清零连续写 CRC 计数，包括成功的停止命令；不把读取成功当作写入恢复 |
| 汇总/密集触觉读取异常 | 关节有效则继续更新；受影响触觉通道保留最近有效值和原来源时间 |
| SDK `1501035` 或关节板 `110` 过流 | 维持限频 warning，继续运行；不自动降额、切 Passive 或清零 |
| 未识别故障、真实 SDK 异常、非法关节/目标数据 | 保留明确错误及现有停止/收尾路径，不一概吞掉 |

连续写 CRC 采用一个整数计数，阈值用内部常量 `MAX_CONSECUTIVE_WRITE_CRC = 5`。它是本轮明确的研究默认值，不声称为厂家保护阈值；不新增命令失败状态机或自适应配置。在没有新发送时，不因墙钟流逝把一次旧 CRC 升级成连续写失败。

公开位置发送可返回简单 bool：True 表示本次获得允许的非 CRC SDK 返回；False 表示 CRC 未确认；真正故障仍抛异常。沿现有调用链贯通这个区别即可，不新增 CommandResult、命令 ID、ACK 表或整机事务协议。组合 `Robot.send_action()` 正常返回仍只表示完成提交尝试；Raw 的动作保持 `q_goal_submission_attempt` 语义。

周期策略在一次 False 后正常推进到下一控制机会，不停在旧目标上反复补发。HOME、预设手势等一次性操作按 5.2 节处理。连续计数只在普通位置发送路径触发故障升级；停止调用使用自己的有限重试，不能因为已经累计 5 次 CRC 而拒绝尝试停止。新连接初始化计数，普通读取和仅进入待命不能伪造写入恢复。

### 3.2 持续无关节反馈

- 复用 `HandConfig.feedback_max_age_s`，默认从 `4/30` 调整为 `0.5`；不新增第二个用户可调的 hand 通信超时配置。
- 在设备协调层集中表达当前关节反馈长期不可用的判断。活动期间服务反馈时能发现失联，包括等待模型结果期间；发送边界仍不得在已知失联后提交新目标。复用现有运行状态，不新增一套 active/degraded/recovering 状态。
- 0.5 s 内的读 CRC 使用旧关节缓存，之后按已有真实通信故障路径结束。模型实际使用的旧观测仍检查自己的来源时间，不能借另一份新反馈续期。
- 独立诊断脚本、HOME 和初始连接有自己的操作等待界限，不能因为改了周期容忍就变成无限等待。
- 0.5 s 是研究默认容忍窗口，不是实测最优值或有界停止保证。保留简单配置供实际设备调整，不做自动阈值学习。

停止时判断是否可以 hold 实测位置的原有新鲜度标准不随普通反馈窗口一起放宽。将原来的 `4/30 s` 保留为停止路径的局部常量；不增加用户配置。原有有限重试预算、hold/passive 区别和真实失败传播继续存在。

### 3.3 日志与少量计数

复用 logging。维护读 CRC、写 CRC、aggregate 不可用、dense 不可用、过流几个计数即可；持续重复消息限频，恢复消息避免每次 CRC/成功交替都刷屏。关闭时给一次简短汇总。

不新增诊断线程、事件总线、遥测依赖或延迟分位数系统。计数不参与数据有效性推断，不替代原始错误码和来源时间。

## 4. 固定全模态数据：一次采集，统一导出

### 4.1 固定接口

启用录制时，无论策略使用什么输入，都采集当前完整 Raw 内容：arm/hand 实测关节、最终动作提交尝试、RGB、depth、aggregate、dense、各来源时间及现有元数据。保留遥操作调试字段。

`--record/--no-record`、Policy 的 `--record-raw` 只决定是否录制，不变成模态选择器。Policy 不录制时现有按需读取行为不属于本轮改造；一旦录制，使用固定全模态集合。

Canonical 始终输出以下 13 个字段；名称、单位、逐行形状、dtype 和动作语义保持兼容：

| 字段 | 单行形状 | dtype |
|---|---|---|
| joint_state | `(19,)` | float32 |
| action | `(19,)` | float32 |
| action_ee | `(21,)` | float32 |
| finger_tactile_force | `(5,3)` | float32 |
| taxel_tactile_force | `(5,120,3)` | float32 |
| fingertip_xyz | `(5,3)` | float32 |
| eef_pose | `(9,)` | float32 |
| camera_intrinsic | `(9,)` | float32 |
| camera_extrinsic | `(12,)` | float32 |
| rgb | `(H,W,3)` | uint8 |
| depth | `(H,W)` | uint16 |
| point_cloud | `(N,6)` | float32 |
| hand_pointcloud | `(M,6)` | float32 |

沿用原有 H/W/N/M 配置。全模态指字段和真实来源始终完整保留，不意味着所有传感器每行同时产生新测量。

### 4.2 首次真实值与运行中保持

1. 连接只要求设备 ID 正确和首份有效关节反馈，不因触觉尚未准备好就把真实设备视为连接失败。
2. 开始一段全模态录制前，aggregate 和 dense 各自至少有一份完整、有限的真实样本，来源时间为正。两通道不必来自同一次 SDK 读取；不要求为了开始录制先做清零。
3. 复用现有首次观测等待预算及键盘服务，不增加新的就绪框架。若缺少初值，本段暂不开始，明确指出缺失通道，正常回到待命，操作者可稍后再按 B。无需关闭健康设备或重新加载模型。若此前已做准备，按已有正常停止收尾返回；真实 SDK/停止故障继续报错。
4. 初值检查发生在第一条本段动作之前，避免运动已经开始却没有全模态记录。未开始的段不发布零行或伪造数据；尽量在检查通过后创建 writer。
5. 录制开始后，触觉异常按通道保留最近有效数组及原时间。关节和其他正常通道继续更新，writer 继续写每一行。恢复时自然替换缓存，不调用自动 reset。
6. 仅录制触觉超龄不会停止运动，也不会截断录制。长时间缺帧仍保持该通道，明确告警和记录来源年龄；不用补零、插值或未来样本回填来伪装新测量。
7. 从未有效、清零后尚未重新有效的缓存，内部可继续使用现有 NaN/零时间表示；不得以这种状态启动完整录制。已经正常取得初值后的丢帧，不把缓存变成 NaN。

长期触觉保持的数据可能不适合触觉训练，但仍可用于其他分析；本轮保留完整数据和可追溯质量，让研究者判断。不会因更换策略而改变文件，也不会默认整段拒绝导出。

### 4.3 控制输入与录制质量的边界

保留现有 `control_needs` / `record_needs` 的直接用途，不建新的需求系统：

- 位置遥操作、未使用触觉的策略：触觉年龄不参与运动准入。
- 策略实际使用 aggregate 或 dense：依据已有模型字段自动识别；使用 `feedback_max_age_s` 作为默认允许输入年龄，超出后按现有输入失效路径结束当前段。用户无需手动切换采集配置。不要给依赖触觉的策略无限期提供旧输入。
- 录制检查：确认触觉初值存在即可，不因年龄增加返回 fatal quality_issue。其他现有完整性检查、writer I/O 错误、队列满和真正传感进程故障继续按现有逻辑处理。

这里只解除触觉质量与运动的耦合，不借机删除所有 recording_issue 或修改相机、writer 的错误处理。不要把相机进程死亡解释成普通触觉缺帧。

### 4.4 导出与元数据

`data/export.py` 接受合法的旧触觉来源时间，沿用 `max_tactile_aggregate_age_ns` 和 `max_tactile_dense_age_ns`。`test_feedback_holds_flow_through_real_recorder_to_full_zarr` 通过真实 writer/container 链路验证保持值、原来源时间与年龄统计。

- 全部 13 字段始终导出，保留原行数、顺序、episode 边界与名义 dt。不按策略少导字段，不自动筛行、拆段、重采样或拼接时间。
- 对最大触觉来源年龄超过 0.5 s 的 episode 给一次明确 warning，沿用最大年龄统计；该值仅用于报告，不是导出拒绝阈值。Raw 的逐行来源时间提供精确追溯；不向固定训练数组追加 mask 或新的观测维度。
- 来源时间为正、非未来、不倒退；形状、单位、数值有限性和 RGB 帧数一致性继续验证。合法的旧来源允许重复。始终缺失的触觉数据不能伪装为可导出的完整数据。
- 元数据直接声明新采集的触觉保持规则，例如 `hand_tactile_hold_policy="last_valid_until_recovery"`。停止写入会误导使用者的有限 `hand_tactile_hold_max_age_s`；旧文件中的该属性继续被当作历史说明，不作为导出门槛。不要为此引入格式版本分流或兼容适配器。
- 保留原生单位未知、指序与传感器轴约定；不得把保持值解释为当前接触，也不假定 dense 求和等于 aggregate。
- 现有仅支持满足动作结构要求的 Raw 导出范围保持原样；不借本轮扩展带空动作行的 Policy 日志导出。外参等几何必需事实也不能凭空生成。

## 5. HOME、清零和会话解耦

### 5.1 启动与公共边界

取消通用启动时强制 hand HOME 和自动触觉清零。连接后读实测姿态，初始化命令结构，不隐式移动到全零或 HOME。H 和 T 沿用现有交互；实验需要固定初始姿态时显式执行 H。

检查 teleop、Policy、keyboard jog、replay 和独立手部诊断的实际入口，更新依赖启动 HOME 的初始化。重定向初始锚点、首个策略观测等从当前实测姿态建立；保留已有起点差异、关节范围和映射检查，不能因取消 HOME 引入明显首帧跳变。

配置在实际入口校验一次，纯打印配置不连接设备。HOME 不再重复验证已经确认的静态限位，但独立诊断等公开入口仍有自己的必要校验。

### 5.2 独立手部 HOME

- hand HOME 前停止机械臂；执行期间 arm owner 保持停止，不调用 `arm.resume()`，不清除共享 stop 来间接授权 hand。
- 专用 hand HOME 路径直接检查操作者取消/急停及设备故障，验证目标后发送。内部为了停止 arm 设置的 stop_signal 不应被误判为用户取消；用已有取消意图/计数和一个回调区分，不新增授权 token。
- 用现有 hand HOME 超时和容差等待提交后的新关节样本到达目标。保留当前简短的连续样本判断即可，不为减少几行代码另造到位判定体系。
- 若首次发送 CRC，可在下一次正常反馈服务机会补发一次同一绝对 HOME 目标；总发送至多两次，不增加 80 ms 阻塞退避。仍以新反馈到位为成功条件，即使两个回执均为 CRC 也不能仅凭返回码宣布成功或失败。
- 到位等待、取消和补发共享现有一次操作期限。超时明确报告未到位；停止/关闭错误如实保留。预设手势等一次性操作使用相同原则，周期控制不套用此补发策略。

### 5.3 显式触觉清零

T 仅在空闲、无接触条件下执行。继续使用传感器地址 `0x11`–`0x15`，不得换成末端关节 ID `2/5/7/9/11`。

执行一次五指循环，每个传感器默认发送一次，返回一个简单的每指 SDK 返回码结果；不做三次乘五轮重试，也不加入软件 bias。用户可再次按 T 明确重试。复位结果分类直接复用已知错误码：0 记录正常返回，1501035 记录并告警；CRC 记录未确认，已知触觉/温度不可用码记录本次请求异常并继续其余传感器；其他真实 SDK/设备故障仍抛出。此处记录的是请求返回结果，不推断硬件零点效果。

记录一次操作时间和结果即可；取消时已完成的每指结果仍可诊断，未执行不冒充成功。不维护多种逐步字符串状态；普通复位失败明确显示并返回空闲，不因“零点未确认”关闭健康设备。真实通信/关节故障及停止失败仍按原故障路径处理。

一旦开始实际发送清零请求，就使旧触觉缓存失效，后续初值检查重新等待各通道真实数据。可在现有反馈缓存类增加一个直接的触觉清空操作；关节缓存不必一并清空。正在录制时不执行清零。

操作返回后无需再做整机 `_read_initial_state(required_tactile=...)` 等待。正常反馈循环自然补齐触觉，下一次 B/触觉策略初值准入负责确认数据存在。SDK 返回仅表示调用结果，不宣布已经完成物理零点校准。

### 5.4 驱动与停止

从 XHand 驱动去掉 `SessionState` 导入及 connect/disconnect 的 runtime_state 参数。驱动只管理 SDK 资源，最多接收一个取消回调；会话入口和现有键盘处理负责中断意图。

删除重复的异常包装和内部不变量检查时，保留部分连接失败的 close、实际错误原因和 KeyboardInterrupt 传播。收尾中不能吞掉中断，也不能用关闭成功覆盖先前停止失败。

继续预建 `HandCommand_t`，保留显式 position/passive 方法、12 关节映射及必要边界校验。`XHandFeedback` 的小型只读缓存可以原样保留；不为追求单文件或行数减少移除独立来源时间。

## 6. I/O 效率：有证据才复用 SDK 缓存

PI-R2 在 `send_command()` 后使用 `read_state(False)`，源码注释称不增加 RS485 往返；这只能作为验证线索，不能证明本机安装版本的行为。

实施者检查可取得的本地 SDK 版本、源码或对应文档，核对：发送是否确实更新反馈；缓存返回是否能区分旧帧；通道错误与来源时间是否可保留；RS485 与 EtherCAT 是否一致。

若能证实这些条件，采用一个直接调用路径：发送后消费已更新的反馈，空闲时按现有频率主动读；避免刚发送获取新反馈后又立刻强制读取。保持现有单一 I/O 调用者，不加锁或线程来包装本来串行的路径。

无法证明缓存是否更新时，保留 `read_state(True)`。没有包序号/采样时间时，不得把多次读取同一缓存的调用时刻当作新来源时间；数据值相同也不能证明是新帧。不得用 Fake SDK 的自设行为证明真实 SDK 缓存可靠。

本项的完成条件是给出证据与明确决定。证据不足时，记录“已核查、暂不启用缓存复用”即可完成本轮，不引入 `use_cache` 等长期未验证配置。性能改善须实测，不能以减少函数调用声称已降低总线延迟。

## 7. 实施任务与完成条件

### T0：基线与调用链

- [x] 记录实际实施 SHA、工作区已有修改；核对当前 AGENTS。
- [x] 核对 read/send/stop、hand HOME、T、Teleop/Policy 录制及 Canonical 导出实际调用链。
- [x] 确认最新代码与本文基线的差异，保留无关改动；历史任务书不作为回滚依据。

### T1：偶发异常容忍

- [x] 实现位置写 CRC 的简单未确认结果、连续计数及最新目标继续发送。
- [x] 调整普通反馈容忍窗口，贯通活动期反馈服务和真实发送；停止新鲜度保持原标准。
- [x] 保留读 CRC 丢帧、过流告警和触觉独立有效性；简化日志。
- [x] 删除 hand 同步发送分支的目标租期检查；arm 异步 owner 的 expires_ns 继续使用，HOME 的操作期限继续使用。

### T2：固定全模态采集、导出

- [x] 两类触觉初值就绪后才开始全模态录制；未就绪是可再次开始的准备结果。
- [x] 删除仅因录制触觉超龄而停段的逻辑，保留记录队列、I/O、相机进程等真实失败处理。
- [x] 保留每行全部模态、独立原来源时间和最新有效值保持；更新保持元数据。
- [x] 验证全模态 Canonical 的 13 字段、原行数与原顺序；复用年龄统计和现有导出容错。
- [x] 真正依赖触觉的在线模型仍检查所用输入年龄，且不影响固定录制/导出格式。

### T3：HOME、清零、会话依赖

- [x] 取消隐式启动 HOME/清零，检查当前姿态初始化和各入口首目标行为。
- [x] hand HOME 不恢复 arm；一次性目标 CRC 有限补发并以实测到位判断。
- [x] 清零使用简单每指结果、仅失效触觉缓存、自然等待后续采样。
- [x] 去除驱动 SessionState 依赖及重复校验/包装，保留中断与资源清理。
- [x] 适配 `scripts/tools/control_xhand.py`；其现有原生崩溃隔离壳不扩散到主运行时，也不借本轮无依据删除。

### T4：SDK 缓存与收尾

- [x] 按第 6 节核实 SDK 缓存；有证据才实现复用，否则记录未启用原因。
- [x] 同步 README、相关调用方、类型和行为测试；历史文档只补必要的被替代说明，不重写旧验证事实。
- [x] 完成下述针对性验证、检查最终 diff，并填写实施记录。

## 8. 验证要求

优先修改已有相关测试；XHand 缺少集中覆盖时可新增一个小型 `tests/test_xhand.py`，使用现有风格的 Fake SDK、可控时钟和有限样本。不要创建测试框架，不对每种异常排列做组合爆炸。

| 场景 | 必须观察到的结果 |
|---|---|
| 读 CRC 后恢复、连续无有效关节反馈 | CRC 不更新缓存时间；窗口内可继续，超过窗口在活动期停止，恢复样本更新 |
| `CRC, CRC, 正常, CRC` 与连续 5 次 CRC | 前者继续并清零连续计数；后者进入已有故障收尾；读取成功不能清写计数 |
| arm 已提交、hand 单次写 CRC | 不因该次 CRC 设置全局 stop；下一周期使用最新目标；Raw 保留尝试目标 |
| 过流及 aggregate/dense 分别缺帧 | 有效关节/通道照常更新，失效通道原值原时间不变 |
| 首次尚无 dense、随后出现有效 dense | 第一段不带伪数据开始；健康资源可保留，重新 B 可开始固定全模态录制 |
| 录制中仅触觉持续过期，视觉/位置控制正常 | 控制与逐行写入继续；无丢行，旧时间真实保留；明确质量提示 |
| 触觉策略使用的通道超龄 | 按输入失效结束，不能因录制允许保持而给模型无限旧输入 |
| 真实 Raw → Zarr，包含触觉缺帧/保持 | 13 字段完整、全部预期行、正确 dtype/shape；保持值和年龄统计正确；不补零或筛行 |
| 数据损坏：NaN、零/未来/倒退来源、缺字段、RGB 不对齐 | 原有结构/数值/因果检查继续拒绝；正常旧来源重复允许 |
| HOME 首次 CRC、取消、未到位；arm 保持停止 | 最多补发一次，提交后新反馈确认到位；不调用 arm.resume，不清共享停止授权 |
| T 失败/取消后再次开始 | 传感器 ID 正确，已发生结果可追踪，旧触觉失效、关节保留，无自动多轮重试 |
| 停止 CRC、停止失败、关闭中断和普通失联 | 停止仍有限重试，原新鲜度标准未被 0.5 s 放宽；故障与中断不被清理覆盖 |

真实 Raw→Zarr 测试必须通过现有 writer/container 路径，不能只构造 dict 然后声称全链路通过。保留旧全模态数据导出和本轮 `c4ded28` 修复的相关回归。

在已有依赖环境中先运行改动对应测试，随后一次运行受影响的现有回归，例如：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider \
  tests/test_control_progress.py \
  tests/test_home_snapshot.py \
  tests/test_input_resources.py \
  tests/test_recording_export.py \
  tests/test_replay.py \
  tests/test_arm_recovery.py
python -B -m ruff check --no-cache --select F dexmani_real scripts tests
```

如新增 XHand 测试，将实际路径加入。CLI 仅运行相关 `--help` / `--print-config` 做无设备检查。依赖缺失要如实记录，不能把编译、AST 或静态检查称为行为测试通过。不要为了“验收入口”启动真实 HOME、运动或清零。

最终 diff 审查重点：没有新增通用框架；没有按策略改变录制/导出模态；没有把旧触觉续期；没有把未确认写入或 SDK 停止返回冒充物理执行；没有回退基线后的无关修复。

## 9. 参考依据与保留边界

本项目固定基线链接：

- [XHand SDK 边界](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/robot/xhand/driver.py)、[独立通道缓存](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/robot/xhand/feedback.py)。
- [Robot 协调](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/robot/system.py)、[HOME](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/runtime/home.py)。
- [观测准入](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/runtime/observation.py)、[Teleop](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/teleop/loop.py)、[Policy](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/deployment/rollout.py)。
- [Raw](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/data/raw.py)、[固定 Canonical 导出](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/dexmani_real/data/export.py)、[导出回归](https://github.com/haoyangzhanglab/dexmani_real/blob/c4ded28b1078adefbd017f4c546b6a132a91d29b/tests/test_recording_export.py)。

外部项目只提供取舍参考，不构成对本机固件的保证：

- [PI-R2 XHand](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/robots/xhand_robot.py)：借鉴直观接口、预建命令及发送后缓存读取；不照搬所有非零码统一失败、无真实结果反馈、多轮清零与未经核实的单位解释。
- [Dexora forwarder](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/xhand_forwarder.py)：参考 CRC 可恢复的取舍；不引入 ZMQ、80 ms 阻塞重试、NaN 变零或维数自动补齐。
- [LeFranX XHand](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/xhand/xhand.py)：参考简洁设备方法；不接受静默 Stub、CRC 载荷继续解析、未实现的 stop 返回成功。

## 10. 实施记录（由执行 Codex 填写）

实施日期：2026-11-04。以下为本轮实际完成内容，不表示真机验收通过。

### 基线与调用链（T0）

- 实施起点：`main` / `0769ce43972cf08dcf67fd66f453c30f144074b7`，初始工作区干净，已全文阅读当前 `AGENTS.md` 与本文。相对源码基线 `c4ded28b1078adefbd017f4c546b6a132a91d29b`，仅新增本文，未发生后续源码变化。
- 核对了 XHand read/send/stop、Robot 组合发送与活动反馈服务、runtime hand HOME、显式 T、Teleop/Policy 启动及录制、jog/replay/独立诊断入口和 Raw→Canonical 导出。保留基线已有录制错误传播、模型关闭错误、标定和停止收尾修复；历史任务书仅补充被替代说明。

### 完成任务与实际删减（T1–T3）

- 周期公开位置发送返回 bool；CRC 不确认但继续下一周期最新绝对目标，连续第 5 次 CRC 才抛通信故障。允许的非 CRC 命令返回清计数，读取不清计数；Robot 保持提交尝试语义，偶发 CRC 不置共享 stop，真正故障进入原停止路径。无旧目标排队、阻塞补发或按墙钟升级写 CRC。
- 普通 hand 关节反馈默认容忍 0.5 s；Robot 在活动反馈服务（含模型等待）与实际发送边界统一检查最新关节缓存。模型所用观测仍按自己的来源时间检查。停止保持局部 `4/30 s` hold 新鲜度与原 0.25 s 有限重试；CRC 累计不阻止停止尝试。
- 复用按通道缓存，读 CRC 不更新任何来源时间，触觉缺帧仅保留该通道旧值/旧时间；过流限频告警继续运行。日志只保留少量计数、按类限频 warning 和关闭汇总，删除恢复刷屏状态。
- 删除 `initialize_hand()` 及 Teleop/Policy 启动隐式 HOME/清零、Policy 启动遗留 HOME 校验、hand 同步发送目标租期检查、hand HOME 的 arm.resume 调用、驱动 SessionState 导入及 connect/disconnect 参数、清零多种进度字符串、复位后的整机初值等待，以及废弃的全缓存 reset 方法。保留目标/设备身份检查、取消意图、真实错误与资源释放。
- hand HOME 先停止 arm，全程保留共享 stop；操作者取消用现有 cancel_epoch/急停意图和回调识别。首次 CRC 仅在下次反馈采样机会补发一次，共用操作期限，最多两次发送，以提交后的连续新实测反馈到位判断。独立预设手势采用同一原则，原生崩溃隔离壳保留。
- T 使用 `0x11`–`0x15` 单轮每指一次，返回/保存每指 SDK 码及一次时间；CRC、已知触觉/温度不可用和过流明确告警，继续其他手指，未知故障继续抛出。取消保留已执行结果；开始实际请求时仅清触觉缓存，保留关节，不再同步等待整机初值。正常反馈和下次 B 自然重新准入。HDF5 属性不接受 dict，因此仅在元数据边界将结果 JSON 编码，内部及返回仍为简单字典。
- 没有新增运行进程、I/O 线程、设备抽象、状态机、事务协议、自动恢复、诊断平台或长期未验证开关，也没有新增生产依赖/模块。已有 arm owner、writer 和模型线程继续承担原职责。

### 固定全模态与行为验证（T2）

- Teleop/Policy 录制需求固定为 RGB、depth、aggregate、dense，加原有实测关节、动作提交尝试、来源时间/元数据与遥操作调试字段。录制前复用 1 s 初值等待及键盘服务，双触觉各自完整、有限、正且非未来来源后才创建 writer；缺初值可正常停止准备、保持健康资源、回待命再 B，不提交本段动作、不发布零行数据。
- 录制中仅触觉超龄不再触发 fatal quality_issue，逐行保存最近有效值及原时间；每通道首次超过 0.5 s 给一次告警。实际触觉策略继续使用 `feedback_max_age_s` 检查自己的输入；相机进程、writer I/O、队列满和其他完整性失败路径保留。
- 新元数据为 `hand_tactile_hold_policy="last_valid_until_recovery"`，不再写有限 hold 上限。旧属性仅作历史说明，未引入版本适配器。导出保留 13 字段、shape/dtype、原行数/顺序、episode 边界和名义 dt，超过 0.5 s 每段 warning 并复用原最大来源年龄统计。
- 新测试实际经过 `XHandFeedback → EpisodeRecorder 线程 → RawWriter/PyAV/HDF5 → export_zarr`：6 行包含两个通道独立缺帧/恢复，全部导出；验证 13 字段 shape/dtype、动作与关节行序、保持数值、原时间、episode_ends、dt 和两通道 751000000 ns 最大年龄。另有真实 Teleop/Policy tick→Raw 测试确认旧触觉时仍提交并写全 3 行；初 dense 缺失不建 writer，随后可再开始。NaN、零/负/未来/倒退来源、缺字段、错误形状和 RGB 计数不一致仍拒绝；正常旧来源重复及历史 TTL 属性仍可导出。

### SDK 缓存证据与决定（T4）

- 本地 `real_robot` 环境安装的 distribution 为 `xhand_controller==1.5.2`，路径 `/home/zhanghaoyang/miniconda3/envs/real_robot/lib/python3.10/site-packages/xhand_controller`，实现为原生 `.so`。只检查包元数据和绑定定义，未实例化控制器或打开设备。
- 当前绑定 `HandState_t` 仅公开 `finger_state`/`sensor_data`，`FingerState_t` 也没有包序号/采样时间；`read_state` docstring 未保证发送后的缓存更新或来源时间。
- 本地历史文档 `/home/zhanghaoyang/Desktop/硬件/Xhand/SDK/Python/XHAND Python SDK接口说明文档 (1).pdf` 提到串口发送后刷新状态，但配套 wheel 为 1.1.8，不能证明当前 1.5.2，也未说明 EtherCAT/坏帧时各通道来源。因此已核查、暂不启用缓存复用，继续 `read_state(True)`。Fake SDK 仅证明软件调用行为；未声称降低总线时延。

### 实际执行的检查与结果

使用现有 `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`（Python 3.10.20）；默认 shell 无 `python`，未安装新依赖。该环境已有 pytest、NumPy、h5py、PyAV、Zarr 和所需几何/控制依赖，本轮列出的检查无依赖缺失或跳过项。

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -p no:cacheprovider \
  tests/test_xhand.py tests/test_control_progress.py tests/test_home_snapshot.py \
  tests/test_input_resources.py tests/test_recording_export.py tests/test_replay.py tests/test_arm_recovery.py
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -B -m ruff check --no-cache --select F dexmani_real scripts tests
```

- 最终联合行为回归：**514 passed in 3.63s**。覆盖读 CRC 后恢复与长期无关节反馈、真实 Robot+XHand 偶发/连续写 CRC 及停止、过流、通道独立保持、录制初值及再次开始、模型输入年龄、真实 Raw→Zarr、损坏拒绝、HOME 首 CRC/取消/未到位及 arm 保持停止、T 部分结果/取消/重试、停止有限重试/失败、关闭中断和基线收尾修复。
- Ruff F 检查通过；它是静态检查，不替代上述行为测试。
- 无设备 CLI 检查全部 exit 0：`scripts/run_policy.py --print-config`、`scripts/collect_teleop.py --print-config`、`scripts/replay_episode.py --help`、`scripts/keyboard_teleop.py --help`，以及 `scripts/tools/control_xhand.py --help` / `--print-config`。均使用 `python -B`，未进入硬件会话。
- 最终 diff 已审查：旧初始化调用、旧 metadata 参数、驱动会话参数和全缓存 reset 无遗留；README 与历史说明已同步。`git diff --check` 通过，未生成 `.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/`。

### 必要调整与尚未验证事项

- 保留“未使用的 HOME 专用配置不阻止普通 streaming”的现有边界：仅实际 HOME/独立动作入口验证目标和操作期限/容差，删除 HOME 内重复静态限位检查；禁用手部的入口仍显式验证固定几何限位/姿态。
- 取消启动 HOME 后，Policy warmup 复用初值等待预算，仅等待模型实际所需输入并从实测姿态预热，避免无触觉模型在启动时被仅录制触觉阻塞；录制准入留在每段 B 前。未录制的触觉模型在 T 后也等待所需通道初值，年龄检查继续独立生效。
- `service_feedback()` 返回本次是否到达采样机会这一简单 bool，供 HOME 只在下一正常服务机会补发；无额外定时器或命令状态协议。
- 真机验证：**未进行**。未启动真实运动、HOME、清零或连接测试；真实 checkpoint 推理、实际串口/EtherCAT、固件触觉零点效果、0.5 s/连续 5 次默认值的设备适用性及物理停止时延尚未验证。软件等待不能中断 native SDK 阻塞。后续真机验证应单独记录设备/固件/SDK、操作和结果。


## 11. 全链路复审与清理记录

复审日期：2026-11-04；在第 10 节及随后 Keyboard.stop 整理的未提交工作区上继续实施，保留既有正确行为。复审覆盖 XHand/Robot/HOME/T、启动与模型输入、遥操作与回放、录制/导出，以及键盘停止到会话关闭的调用链。

- 修复 arm 邮箱发布可能阻塞后手部反馈过期的遗漏：组合发布前阻止已知失联的设备副作用，实际 hand 发送前再次检查；两处检查分别覆盖不同的副作用边界。
- hand HOME 的既有 `deadline_ns` 在实际发送边界、取消回调及故障检查之后统一核对，取消回调耗尽期限后不再发送；删除上层同义前置检查和单用临时变量，调用方及返回类型同步。
- 删除只剩一个调用方的 `Robot._send_action()` 包装，普通发送仍先做组合目标校验。删除 XHand read/send 仅重新包装异常的捕获层，原生 SDK 异常保持原对象/类型和 traceback，上层仍进入停止路径；返回错误码的检查保留。
- 清掉 Teleop 从未使用且随即覆盖的服务回调赋值、jog 的无用 lambda，以及 Replay Fake 吞旧参数的 `**kwargs`；Fake 按当前发送 bool/无参 disconnect 接口工作。
- 录制与导出的触觉报告阈值共用 `TACTILE_AGE_WARNING_NS`，修正共用初值函数说明，去掉观测检查内部重复条件。删除只围绕旧 hold 上限属性的六种参数测试与陈旧断言，以真实保持→RawWriter→Zarr、合法重复来源及损坏拒绝测试验证当前规则；没有新增格式门槛或兼容分支。
- README 和手部重定向文档对齐提交尝试语义，纠正单 CRC 停段/不推进参考的过时描述，删除失效历史字段说明。Keyboard.stop 的局部错误归并与必要 finally 经上下游复核保留，无需再套资源管理工具。相关链路没有 SHA-256/内容哈希校验机制，未扩散修改其他模块。

最终验证：

- 先分别运行设备、录制、入口和键盘相关定向测试，再统一运行第 10 节所列七个测试文件：**516 passed in 3.44s**。包含本次新增的原生 SDK 异常保真/上层停止、arm 发布耗时后反馈过期、HOME 回调耗尽期限，以及此前的真实 Raw→Zarr 和 Keyboard.stop 收尾回归。相较上一轮数量变化包括删掉六项旧 TTL 专用参数和新增边界测试，不代表旧测试被跳过。
- 全仓 `ruff check --no-cache --select F dexmani_real scripts tests`、`git diff --check` 通过。
- Mypy 使用 `--check-untyped-defs --no-incremental --cache-dir=/dev/null --follow-imports=silent --ignore-missing-imports`。Keyboard、observation、HandConfig、XHand feedback/driver 五个核心模块通过。扩大至本轮涉及的十八个实现模块后，Raw/PolicySession 仍有 37 项既有可空资源类型标注诊断；同参数对照 HEAD 为 38 项，按文件/诊断内容比较无新增诊断。整体类型检查未通过，未用 Any、ignore 或额外运行时判空掩盖这些既有标注问题。
- 将当前包源码复制到 `/tmp` 后，用现有 setuptools 构建 wheel 成功，并确认驱动、键盘和导出模块按当前内容打包；仓库内未生成构建或工具缓存产物。构建通过不表示脱离仓库资产、设备 SDK 和模型即可运行。
- 未进行真实设备连接、运动、HOME、清零或 checkpoint 推理。SDK 缓存决定保持第 10 节结论；设备适用性和物理时延仍待真机单独验证。
