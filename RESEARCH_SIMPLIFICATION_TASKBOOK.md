# DexMani Real 个人研究代码精简任务书

> 目标：保留音频与现有核心实验能力，使 `dexmani_real` 轻而薄、正确好用、简洁高效、职责统一。优先删除重复机制和错误的复杂性，不以减少行数为理由改变数值含义、掩盖失败或移除实际执行边界。
>
> 本文件是待执行任务书，创建本文件不表示下述代码已经修改或验证通过。实施者应直接完成适用任务，不停留在重新出一份方案。

## 1. 范围、基线与执行原则

- 目标仓库：[`haoyangzhanglab/dexmani_real`](https://github.com/haoyangzhanglab/dexmani_real)。审查基线：[`aae1ae5f1a61c39e6bb4dc8af108ee06b3e7bebe`](https://github.com/haoyangzhanglab/dexmani_real/tree/aae1ae5f1a61c39e6bb4dc8af108ee06b3e7bebe)。任务书日期：2026-11-02。
- 配套 Policy 的核对基线：[`c9c5b96c637296ddcd71716a317cf5882f38ba38`](https://github.com/haoyangzhanglab/dexmani_policy/tree/c9c5b96c637296ddcd71716a317cf5882f38ba38)。它是接口核对依据，本任务不授权顺手重构或提交另一个仓库。
- 实施前读取当前 `AGENTS.md`、检查工作区已有改动与当前 HEAD。本文定位到的函数若已变化，以当前调用链为准；已消失的问题标记“不适用”，不要为了符合任务书重新制造改动。
- 用户已明确要求保留音频。本任务中的具体取舍取代此前审查中的可选建议；历史 `.codex/tasks/` 文档不作为需要重新实现的累计清单。
- 本任务保留：VR 遥操作与手部重定向、Raw 录制、Canonical 导出、sync / async / RTC、键盘点动、物理回放、HOME、标定、查看及现有离线几何入口。
- 主要修改现有文件，不先搬目录，不另建“简化版”平行实现。不新增 manager、registry、通用状态机、服务总线、身份校验框架、命令回执协议或兼容配置系统。
- 不改变硬件控制频率、限位、滤波/优化参数、点云采样、触觉数值、数据类型或模型输入配方来换取行数。性能收益须来自减少重复工作，并有针对性测量。
- 不清理用户数据、不重写旧 episode/checkpoint、不自动连接机器人或执行真机运动。无硬件时完成代码、模拟验证和文档，并明确留下真机检查项。

## 2. 目标结构：沿现有边界做减法

不追求所有入口共用一个“大控制循环”。各入口直接表达自身流程，只复用已经存在的、确有相同语义的底层能力。

| 边界 | 唯一职责 | 应避免的重复 |
| --- | --- | --- |
| `config/` 与 CLI | dataclass 默认值、可信 Python 实验配置、实际使用分支的参数检查 | 多套配置格式、字段注册表、为未启用功能加载标定 |
| `sensor/`、设备 driver、IPC reader | 把外部返回转换成可信样本；保留真实来源时间 | 上层重复解析 SDK 包、读取缓存时重打新时间 |
| `runtime/observation.py` | 按本次入口的 `ObservationNeeds` 组装并裁决一份快照 | 各模态各写一套 freshness 与可选字段系统 |
| `control/`、`teleop/vr.py` | 输入到候选动作的数值映射；拒绝原因明确 | 混入录制、按键、发布、设备生命周期 |
| `robot/` | 真实设备副作用、代际、到期、停止与 SDK 错误 | 上层模拟设备健康、绕过 owner 直接写 SDK |
| Teleop / Policy / Replay 入口 | 实验流程、分段、保存意图、最终退出码 | 多层重复转换同一个异常、复制同一结果 |
| `data/` | Raw 存储、读取边界、固定 Canonical 转换 | 从日志字符串构造另一套“证据协议” |

保留现有 arm owner 进程、camera / VR / point-cloud worker，以及单线程 PolicyWorker。不为这次精简改用全异步架构、全同步单线程或新的进程监督平台。

## 3. 必须保持的数据与执行含义

以下是验收依据，不是要求新增运行时 contract 或逐层断言。

| 数据/行为 | 保留的真实含义 |
| --- | --- |
| VR | 腕部位置以米计，四元数 WXYZ，21 个 landmark；接收端转换到既有 FLU 约定。缓存重复读保持原 sequence/time。同一 VR 帧可以在不同控制 tick 产生不同目标，不能按帧号去重动作行。 |
| RGB-D | depth 对齐到 color；RGB `uint8`，depth 为原始 `uint16` 计数，乘实际 `depth_scale_m` 才是米。主机接收时间不冒充曝光时间。 |
| 相机几何 | 内部 `T_xarm_base_from_color` 为 camera→base；Canonical `camera_extrinsic` 为 base→color optical 的 `[R\|t]`。不能因命名精简而倒置变换。 |
| 点云 | 基座系 XYZ 米 + RGB `[0,1]`；保留现有过滤、FPS 与不足点数时重复采样语义。点数固定不表示独立有效点足够。RGB-D 与其派生云须来自同一包。 |
| 关节 | arm 7 + hand 12，位置弧度；关节顺序明确。实测状态、目标和 Ruckig 内部参考不是同一数据。 |
| 触觉 | aggregate `(5,3)` 与 dense `(5,120,3)` 独立有效、独立来源时间，SDK 原生未知物理尺度与 sensor axes。`hand_contact` 不是布尔接触，dense 不被假定可直接求和等于 aggregate。 |
| 触觉复位 | SDK 请求完成、所需反馈就绪，不等于已证实零点正确；不能改写为“校准成功”或伪造牛顿单位。 |
| `fingertip_xyz` | 实测 arm/hand 关节经 URDF 与安装模型 FK 得到的五个 tip frame 原点，基座系米；不是 VR 指尖、视觉检测或实测接触点。 |
| EEF / 动作 | `eef_pose` 为 XYZ + rot6d，`action` 19D、`action_ee` 21D 为绝对目标。Raw action 是提交尝试；不能写成已执行状态、动作增量或下一帧真值。 |
| 提交 | arm mailbox 后 hand SDK 顺序提交，可能部分成功；不是原子事务。mailbox 可能被后续目标覆盖，不能宣称每行都到达 SDK。 |
| 手部参考 | 只有正常提交返回才 commit；迟到丢弃的候选不推进 EMA、warm start 或提交时间参考。 |
| 时间 | 不刷新旧缓存来源时钟，不追赶旧动作。Canonical 保留从第 0 行开始的有效连续前缀，不跳过坏行后拼接或压缩时间。 |
| 录制发布 | 非空技术失败段只在文件完整关闭且计数一致时可发布为 failed；writer/flush/计数失败不发布，保留临时产物。零行不发布。 |
| Policy | sync、async、RTC 的调度与前缀语义不改。RTC 承诺策略动作，实际执行槽再映射 IK；未回收模型任务禁止新段复用。 |
| 物理回放 | 按当前反馈与限位准入，按现有标称节拍重发；`COMPLETED` 只表示序列提交结束。导出准入与物理回放准入不能合并成一个通用“valid”标签。 |

Canonical 维持固定 13 字段：`joint_state`、`action`、`action_ee`、`finger_tactile_force`、`taxel_tactile_force`、`fingertip_xyz`、`eef_pose`、`camera_intrinsic`、`camera_extrinsic`、`rgb`、`depth`、`point_cloud`、`hand_pointcloud`。以现有 `canonical_array_specs()` 为 shape/dtype 的实现来源，不再手写第二份运行时 schema。

## 4. T1：先修确定的错误与文档失真

### 4.1 本地拒绝不能误报 SDK 故障

定位：`robot/system.py::Robot._send_action`、`replay/joint_replay.py::JointReplay.run`，并追踪 teleop / jog / HOME / rollout 对同类异常的处理。

基线中，手目标到期、当前手反馈缺失/过期在 hand SDK 调用之前抛出 `TimeoutError`；回放把它落入 `FAULT`。这与“本地拒绝/取消”语义不一致。

修改要求：

1. 这两处本地准入失败使用既有 `OperationCancelled`，保留具体原因；不新增异常层级。
2. 真实 SDK `TimeoutError`、SDK 错误码、共享资源未确认超时、未解决的设备请求仍是故障。禁止 `except TimeoutError: cancelled` 这样的批量转换。
3. arm 已先发布而 hand 拒绝时，仍撤权并实际停止；不能把“hand 尚未调用”误写成“整条命令毫无副作用”。
4. 取消与设备/停止失败并发时，真实故障不能被正常结束覆盖。

验收：分别注入本地目标过期、本地手反馈过期、hand SDK 抛超时、停止失败。前两者未调用 hand SDK 且回放为 `REJECTED`；后两者为故障。验证部分提交后停止仍发生。

### 4.2 修正 source_id 与训练恢复说明

定位：`README.md` 的数据身份段落；核对配套 Policy 的 `training/common/resume.py`、配置组装和数据读取。

基线 README 宣称 split/resume 必须匹配同一数据身份，配套 Policy 实际没有强制 `source_id` 身份体系，且允许更换数据路径。改为如实说明当前行为。

本轮保留 `source_id` 作为已有数据集的人工标识/归档字段，不赋予 hash、内容身份、split/resume 保证。保留一个 UUID 几乎没有维护负担，删它不值得引入跨仓兼容变化。不得通过给 Policy 新增身份 gate 来“修正文档”。

### 4.3 键盘关闭竞态按真实承诺处理

定位：`runtime/keyboard.py::handle_key/stop`。

`handle_key` 的 active 检查与 callback 调用之间释放锁；`stop()` 返回前后，已接纳的 callback 可能继续完成。基线注释“active 已禁止它再次调用会话”过强。

本轮选择：明确 callback 只做幂等的停止/退出意图，不做 SDK I/O、资源创建、授权或发布；允许关闭前已接纳的 callback 完成，关闭后不再入队或恢复运动。修正注释及相关文档，不为更强保证增加 callback 引用计数、等待屏障或跨锁协议。若当前源码存在会授权/操作设备的 callback，先移回控制线程再接受此简化。

验收：用事件屏障阻塞 `get_run_id()`，在其间关闭键盘再释放；允许幂等停止 callback 完成，但不能新增事件、重新授权或触发设备调用。不要把该竞态夸大成已证实的意外运动。

## 5. T2：删除已建立内部不变量上的重复工作

每项先 `rg` 查全调用方。找不到覆盖条件时保留并解释，不按类名或函数长度猜测。

| 位置 | 具体减法 | 保留条件 |
| --- | --- | --- |
| `Robot.owner/check_owner` 及其调用 | 删除父线程 ID 记录、重复线程身份断言和仅依赖 owner 的单次连接 guard；清理无用 `threading` 导入 | 当前四个入口均由唯一控制线程持有 Robot，键盘只改意图。`close()` 改按实际设备句柄判断，启动中途失败仍释放资源。arm owner、代际、stop_failure、连接状态不得删除。 |
| `control/actions.py::normalize_policy_actions` | 删除已由 `Rollout.poll_model` 和 deployment preflight 保证的 shape/finite 重检 | 当前只有通过该边界的调用方。保留数组复制、类型转换、裁剪、rot6d 归一化与几何有效性检查；模型输出边界和设备副作用边界各有实际职责。 |
| `Robot.read_hand` / `XHand.get_state` | 将断连错误统一为 driver 的 `RobotError`，删除上层同义 `is_connected` 重检 | `self.hand is None` 表示未启用手部，仍须保留；driver 连接检查覆盖所有直接调用者。 |
| `runtime/keyboard.py::is_pressed` 与 keyboard teleop 的 Q 查询 | `quit_requested` 已由 Q callback 设置，删除唯一冗余查询及无其他调用的方法 | 保留 jog 所需 `snapshot()`、held keys 与 Esc latch。 |
| `TeleopLoop.stop_control` | 只有一个调用点时内联这层薄包装 | 仍先请求段结束并撤权，再执行 `robot.stop()`，之后才 flush。 |
| teleop / rollout 的附加录制异常 | 同一个异常只保留一个结果写入入口，删除重复 append / 重复 traceback | 与 T4 的新错误出口一起完成；不能提前删掉唯一能报告第二个故障的路径。 |

不删除 `Rollout.sensors`：`read_observation(self.sensors.channels, ...)` 有真实使用。不把 VR 启动中的服务检查说成“缺失”；`service_idle → check_services` 已间接检查 worker。VR 就绪循环若不能在不增加通用 callback 接口的情况下明显缩短，本轮保留。

## 6. T3：统一分段交互，保留音频

### 6.1 目标按键行为

VR Teleop 与 Policy 使用一致的分段概念。Teleop 删除 `PAUSED`、`QUIT_PENDING`、`resuming` 以及暂停持有 recorder、双 Q 确认、继续时隐式封段等路径。保留 `IDLE / PREPARING / ACTIVE`，它们与设备 `SafetyState` 职责不同。

| 按键 | 统一行为 |
| --- | --- |
| B | 只在空闲时开始新段；prepare 完成后重新取快照和锚定。准备/运行中不重入。 |
| S | 立即撤权/停止，保存并结束本段，回到空闲；空闲时不创建空段。 |
| C | 作为 S 的直接别名，同样保存结束。空闲时不开始，不再表示“继续”。两入口用同一说明，不新增兼容状态。 |
| D | 立即撤权/停止，丢弃本段，回到空闲。 |
| Q | 一次按键停止，保存当前段并退出；此前已锁存的 D 丢弃意图不会被 Q 反转。空闲时直接退出。 |
| Esc | 立即急停、结束会话；完整录制可按 failed 保存，不能显示为正常完成。 |
| H | 只在空闲且无未完成 recorder/准备/模型任务时 HOME。活动时提示先 S；删除 Teleop 活动段内“保存后立即 HOME”的隐式流程。 |
| T | 只在空闲执行现有触觉复位；手部禁用则忽略。无接触仍由操作者确认，不新增虚假的接触安全判定。 |

这是明确的交互变化：旧 C 暂停/继续、双 Q 确认和活动段 H 自动保存回零被替换。同步更新启动提示、README 和实际绑定，不保留旧流程开关。

### 6.2 流程与并发约束

- 键盘线程负责快速 stop/quit/discard 意图，控制线程负责段结束与发布。S/C/D/Q 在准备中也能撤销；B/H/T 继续绑定输入时的代际。
- 一段流程直接表达为：准备 recorder → prepare arm → 有效快照与锚定 → 进入控制 → 撤权和停止 → 完成/丢弃 recorder → 回到空闲或退出。
- 收尾期间不接受新 B/H/T；收尾完成也不能执行收尾期间排队的旧启动意图。Q/Esc 仍可通知退出/停止。首次 finish 冻结保存意图，晚到 D 不能改写已经发布的段。
- Esc 优先处理。退出与丢弃分别表达，不能用“Q 优先”清除已锁存的 discard。S/C 作为同一操作处理，不额外复制状态。
- 保留当前代际撤销和锁内/锁外副作用边界；不能用一个 `active` 布尔量替代 `run_id + stop_signal`。
- `RunEndReason`、`run_ended_id/reason`、`stop_required/stop_failure` 有实际读者，不按“状态太多”整体删除。仅删除因旧 UI 分支消失而无消费者的项。

### 6.3 音频保留要求

保留 `teleop/audio.py::AudioFeedback` 和 `assets/audio/`。沿用后台播放与 `aplay` / `paplay`，不新增音频依赖、插件层或事件总线，不为本轮精简重写播放器。

| 场景 | 提示与时机 |
| --- | --- |
| 新段开始 | 首个目标正常提交返回后播放 begin；按 B 或开始 prepare 不能提前宣称启动成功。 |
| S/C、D | 先停止；实际成功发布后播放 save，实际完成丢弃后播放 discard。零行不得播放“成功保存”。 |
| Q | 可播放既有 quit，最终 end/emergency 根据实际结果决定；不为等提示音推迟停止。 |
| HOME | 接受 HOME 后播放 home；实际完成后播放 home_done；取消/拒绝不能播放完成。 |
| 急停/故障 | 先通知停止，emergency 可抢占普通提示；播放器失败不掩盖设备故障。 |
| 退出 | 先释放机器人等资源，再有限等待尾音并关闭播放器；明确报告播放器关闭失败。 |

原 pause/resume WAV 可以保留为资源；删除对应 UI 状态后无需人为触发，不新增“恢复中”状态仅用于选择一条语音。保留“设备或音频文件不可用时明确 warning，运动循环不等待播放”的行为，不把无播放器当成机器人故障。

## 7. T4：把异常处理从对象图收回到真实边界

定位：`utils/log.py`、`runtime/lifecycle.py`、`runtime/home.py`、`robot/system.py`、`robot/xarm7/process.py`、`robot/xhand/driver.py`、`sensor/realsense.py`、`data/raw.py/recorder.py` 以及四个会话出口。

目标删除 `attach_cleanup_errors()` 的动态异常属性、cause/cleanup 递归遍历、对象 ID 图去重，以及 `error_detail()` 的递归字符串拼装。不能只删 helper：`Robot.check(operation_error=...)` 当前依赖其附加错误判断，必须一起改完。

采用以下简单规则，优先在现有函数里表达：

1. **正常执行**：SDK/模型/文件错误原样传播，必要的语义转换只在边界使用既有异常类型与 `raise ... from ...`。主错保留原 traceback。
2. **资源收尾**：仍尝试所有资源。多资源 close 使用局部平坦 `(stage, error)` 列表；相邻的内部关闭层可以返回该列表供 `close_runtime` 合并。现有对外 `close()`/context manager 若约定抛错，保留该约定，不能变成返回错误但调用方忽略。
3. **多个关闭错误**：优先在会话出口报告平坦列表。必须维持抛错接口的深层函数，保留第一个错误向上传播，其余错误在该聚合边界记录一次完整 traceback。不要为了把所有错误都塞进 JSON 再造异常容器。
4. **已有主错加收尾失败**：保留主错，收尾错误另记日志/已有结果列表，最终非零。不要在多个 finally 中把同一错误反复挂回主错。
5. **原来只是取消/可恢复拒绝，但随后停止或清理失败**：必须升级为真实失败。设备停止失败保留到既有 `stop_failure`；其他未解决请求/关闭失败必须显式传播实际故障或进入最终错误列表，不能只 log 后继续运行。
6. **记录线程与模型线程**：保留首错及原始 cause，后续独立失败平坦报告；未消费 future 的异常仍须被读取。不得用 `except: pass`、清空 future 或 fake success 减少代码。
7. **最终报告**：`error_detail()` 若仍有 JSON 调用方，只编码 `stage / exception_type / message`，traceback 交给 logging；不递归把一份 traceback 压成多份 message。

允许一个小的现有模块内函数消除真正相同的关闭循环；不新增 ErrorManager、FailureGraph、通用 Result monad、全局异常收集器或跨模块 callback 链。Python 仍支持 3.10，不为了异常汇总强制引入 3.11 的 `ExceptionGroup`。

重点复核两个易漏场景：`ArmProcess.request_operation()` 原取消但取消收尾未确认，以及 `_home_motion()` 原路径拒绝但 stop 失败。两者都不能因删除 `cleanup_errors` 而降级成“正常取消，可以继续下一轮”。

## 8. T5：缩减诊断协议、保留必要实验记录

### 8.1 持久化取舍

| 项目 | 决策 |
| --- | --- |
| Raw state/action、RGB-D、逐行来源时钟/sequence、VR 调试数据 | 保留。来源时间是解释 hold 与同步关系所需的数据，不是可删除的修饰标签。 |
| 实际相机参数、depth scale、关节顺序/单位、动作含义、数值配方与模型/模板输入 | 保留当前所需内容，不另做 hash/evidence 校验系统。 |
| Raw `technical_status`、终止原因、有效行数和发布结果 | 保留，它们影响 export/replay 与产物可读性。 |
| `record_control_timing`、每次 operator 操作的结构化事件、重复的锚点诊断 | 默认移为正常 DEBUG/失败日志，删除只为写这些 dict 而存在的时间采集、跨层参数和重复序列化；同步修改消费者。 |
| 控制器 commit 所需 `submit_end_ns`、调度 deadline、真实来源时间 | 必须保留，不能随“诊断时间”一起删。 |
| 手重定向标定、实际 VR 变换、需要定位示教映射的初始锚定值 | 每段最多保留一份必要快照；不构建事件回放账本。若锚定值的数值复现确需独立保存，可保留现有一份记录，不再复制到多处。 |
| soft timeout | 保留内存计数和一份简短摘要；不追加无界逐次事件。 |
| Policy `session_result.json` 与 replay 结果 | 保留现有实用结果：段结果、主要原因、时长/跳步、Raw 路径和关闭结局。已有字段能承载就不再设计新 schema。 |
| 同一个错误同时挂 exception、写 episode、再写 session 的三份完整内容 | 收敛到一处权威结果；其他位置只引用已有段结果或给简短日志。 |
| `source_id`、采集来源 | 保留为解释性信息；teleop/policy_rollout 区分有训练语义，不可按“来源标签无用”删除。 |

`session_result.json` 的保存失败不得跳过设备关闭；最终成功与退出码必须在资源收尾之后决定。已经发布的 Raw 不因迟到模型错误而回写，但会话必须报告该错误。

### 8.2 触觉元数据去掉伪精确性

当前 `validate_tactile_metadata()` 对 processing、units、axes、finger order 的完整字符串做严格相等判断。这不能证明传感器已经校准，但其中关节/指序与单位含义也不能随意放弃。

本轮具体取舍：

- 删除仅描述内部实现的长 processing 文案、说明性 axes 文案作为硬准入条件的要求；说明放在格式文档。若下游仍读取这些属性，暂保留写出，不为删几个属性建立版本迁移系统。
- 保留与真实解释相关的 finger order、单位/原生尺度约定，以及用于有效前缀裁决的有限正值 hold age。缺少实际必需数值或明确不兼容的顺序/单位仍报错。
- 不更改 aggregate/dense 的独立 hold、原始值、缺失标记和来源时间；不新增 `verified/calibrated/evidence` 标签或接触概率。
- 不改写旧数据“补齐正确来源”，不把未知 SDK taxel 排布冒充已验证坐标。

### 8.3 配置只留一条主线

- 保留 `RealConfig` dataclass + `CONFIG: RealConfig` 的 Python 文件入口；CLI 负责任务、路径、段数、录制等运行选项。此部分已经足够薄，不改成 Hydra/Pydantic/YAML 多级配置平台。
- 真正消费者处按需 validate：HOME 才加载 HOME 标定；启用点云去桌面才加载平面；禁用手时不做无用触觉初始化。
- 对同一个已校验、不可变的数值配置只检查一次；外部 checkpoint、Raw 和 SDK 返回仍各自有输入边界，不能把不同来源的检查误合并。
- `_restore_cloud_config()` 的 exact-field gate 不作为必删项：直接 `Config(**recipe)` 会默默补 dataclass 默认值，可能改变旧 checkpoint 数值行为。只有证明当前读取链保证所有数值字段存在时才删除重复校验；否则保留这段短检查，不增加 schema/version/hash 框架。
- 同理，Policy format gate、训练恢复策略属于另一个仓库；本任务只修 Real 文档和实际边界，不扩散为跨仓重构。

## 9. T6：条件性几何问题与明确不纳入的改变

### 9.1 Brown 畸变与 RGB/点云融合

已定位到一条可能不一致的组合：Real 对非零 Brown color 畸变做深度射线反解，但 RGB 仍是原 color 网格；Policy `agents/obs_encoder/pointcloud/projection.py::project_points` 仅用 K 与外参进行 pinhole 投影。该组合可能错配像素。

这不是“所有 RealSense 数据都错误”的结论。必须先读取实际 Raw/运行相机的 distortion model、coefficients，并确认 checkpoint 确实使用该投影消费者。标定 JSON 有非零系数但没有模型字段，不能单凭它认定实际运行命中。

本任务处理步骤：

1. 用现有几何函数构造零畸变与非零 Brown 的反投影→投影测试，确定误差来自何处；在固定合成样例上给出误差，不冒充真机测量。
2. 若实际组合不命中，不更改图像与点云数值；记录“不适用”及依据。
3. 若命中且只支持 pinhole 是当前实验的真实范围，在实际几何消费者准入处明确拒绝该不支持组合，是本轮最小正确处理。不得仅因同时存在 RGB 和 point_cloud 字段，就推断模型一定进行像素投影融合。
4. 若研究确需该组合，在线/离线必须共用一个明确的几何转换：输出一致的 rectified RGB、对齐 depth 与 K，或让真实投影消费者支持畸变。两条路线择一，不能各补一半。需要 Policy 修改或旧数据重导出/重训练时，作为具体后续任务报告，不在本轮偷偷跨仓或改变旧 checkpoint 语义。
5. 没有真实相机参数、checkpoint 或数据时，将“实际部署是否受影响”留为未验证；完成其他精简任务，不编造已修复结论。

### 9.2 本轮保持的几项有意限制

- 不更换线程式模型执行器。`shutdown(wait=False)` 和 Python 等待预算不能终止阻塞 native 调用；已有 pending 状态是实际生命周期信息，不为减少标签假称资源已退出，也不新增进程隔离平台。
- 不把 policy Raw 的触觉需求改成“只录模型使用通道”。当前完整 Raw 要求 aggregate+dense；放松会改变评估数据的完整性。本轮保留，研究确需轻量评估记录时再单独选择一个具体格式，不引入通用可选模态 exporter。
- 不增加命令 ID/SDK 回执/执行完成协议。明确“提交尝试”的现有语义即可。
- 不新增两两传感器同步矩阵、每 taxel 健康分数、模板身份校验或来源置信度系统。时钟不齐和固定点数重复采样的限制如实说明。
- 不把 HOME 碰撞规划、在线目标约束、设备硬限位合并成同一个检查；它们保护不同阶段。
- 不动 `third_party/cordvip_geometry` 的独立环境与算法，不把其物体跟踪结果接入 Policy，除非当前改动确实破坏已有读取接口，需要最小同步修复。

## 10. 参考仓库：借鉴短路径，不复制缺陷或框架

参考基线固定，避免把各仓库当作无条件正确的规范：

| 仓库与版本 | 值得借鉴 | 本任务不照搬 |
| --- | --- | --- |
| [ManiUniCon @ 85c6f2e](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | 在真实控制循环里直接表达设备状态、目标与时间关系 | SharedStorage/Hydra 等整套架构；也不能照搬忽略 SDK 返回码或混淆时钟域的路径。 |
| [LeFranX @ a39906e](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | VR→机器人目标的直接适配、少层包装 | 其保护还分布在 C++ 服务和外部录制组件，不能只比较 Python 行数；不能引入断连时关节映射不一致或“返回成功但未停止”的行为。 |
| [pi-r2-flow @ 3af52ca](https://github.com/pi-r2-flow/pi-r2-flow/tree/3af52ca400a6ec7d141416879aa531a6f62f697a) | 实验入口直接组装资源，录制保留有用数据与时间 | 主控制脚本也很长；不照搬旧相机帧被重新标为当前时间、SDK 失败仅 warning 或保存失败打断资源关闭的处理。 |

参考代码定位：ManiUniCon 的 `maniunicon/core/robot.py`、`maniunicon/robot_interface/xarm6_robotiq.py`；LeFranX 的 `scripts/dual_robot/dual_vr_teleoperator.py`、`src/lerobot/teleoperators/xhand_vr/xhand_vr_teleoperator.py`；pi-r2-flow 的 `deployment/apps/run_policy.py`、`deployment/mindex/cameras/realsense.py`。移植任何具体做法前重新检查对应固定版本，不凭项目风格猜实现。

本仓库的主要证据入口如下；相对链接指向当前 checkout，基线行为以第 1 节固定提交为准。

| 主题 | 源码入口 |
| --- | --- |
| 本地拒绝、设备故障、停止 | [Robot](dexmani_real/robot/system.py)、[JointReplay](dexmani_real/replay/joint_replay.py)、[ArmProcess](dexmani_real/robot/xarm7/process.py) |
| 分段状态、音频、键盘 | [TeleopLoop](dexmani_real/teleop/loop.py)、[遥操作装配](dexmani_real/teleop/run.py)、[AudioFeedback](dexmani_real/teleop/audio.py)、[Keyboard](dexmani_real/runtime/keyboard.py) |
| 异常汇总与会话收尾 | [日志辅助](dexmani_real/utils/log.py)、[close_runtime](dexmani_real/runtime/lifecycle.py)、[PolicySession](dexmani_real/deployment/session.py)、[HOME](dexmani_real/runtime/home.py) |
| Raw 与导出裁决 | [RawReader/Writer](dexmani_real/data/raw.py)、[EpisodeRecorder](dexmani_real/data/recorder.py)、[Canonical 导出](dexmani_real/data/export.py) |
| 模型边界与 RTC | [Rollout](dexmani_real/deployment/rollout.py)、[动作映射](dexmani_real/control/actions.py)、[deployment preflight](dexmani_real/deployment/preflight.py) |
| RGB-D 投影 | [Real 相机几何](dexmani_real/geometry/camera.py)、[Policy 固定基线投影](https://github.com/haoyangzhanglab/dexmani_policy/blob/c9c5b96c637296ddcd71716a317cf5882f38ba38/dexmani_policy/agents/obs_encoder/pointcloud/projection.py) |
| source_id / resume 文档核对 | [Real README](README.md)、[Policy 固定基线 resume](https://github.com/haoyangzhanglab/dexmani_policy/blob/c9c5b96c637296ddcd71716a317cf5882f38ba38/dexmani_policy/training/common/resume.py) |

## 11. 实施顺序与交付边界

按可单独审阅的小批次推进，完成一个稳定批次后继续下一批，不要求逐项向用户确认日常实现选择。

| 批次 | 内容 | 完成条件 |
| --- | --- | --- |
| A | 阅读 AGENTS、确认现状；T1 bug/文档；T2 局部减法 | 本地拒绝分类正确，外部输入与副作用检查保留，入口可导入/配置打印正常。 |
| B | T3 分段状态与音频接线 | 新按键表全覆盖，旧 PAUSED/QUIT_PENDING/resuming 路径消失，快速停止与语音时机正确。 |
| C | T4 异常处理 | 动态 cleanup 异常图消失；首错保留、取消后的清理失败不漏报、所有资源都尝试关闭。 |
| D | T5 诊断和配置去重；T6 条件问题核实 | 无重复诊断协议，数值/数据语义不变；几何问题给出已证实的处理或明确限制。 |
| E | 文档与回归收口 | README 反映最终行为，删除过时说明和孤立导入；所有结果来自实际执行检查。 |

默认在独立工作分支实施，保留用户未提交修改。不强推、不自动合并主分支、不做外部发布；是否创建 PR 按执行环境和后续明确指令处理。本任务书的创建提交本身仅包含该文档。

若某条减法在当前源码需要更多状态、适配层或兼容分支才能成立，优先放弃这一条，给出具体原因；不要把局部精简扩展成新框架。不要为了凑“全部完成”隐瞒有条件项或未验证的真机路径。

## 12. 针对性验收

复用现有测试工具；若没有测试框架，标准库 unittest 与小型 fake 足够。优先调用实际实现，mock 设备、时钟、文件边界；不要只重写一遍算法然后测试副本。只为真实风险写必要回归，避免测试每个删除的 wrapper 或配置 getter。

| 检查组 | 最小覆盖与判断 |
| --- | --- |
| 错误分类 | 本地到期/陈旧反馈→拒绝；SDK/IPC 未确认超时→故障；部分提交→停止；取消加 stop 失败→非零。 |
| UI 与代际 | B→运行→S/C→空闲→新 B；D 不发布；单 Q 退出；准备中 Q/Esc 撤销；旧 B/H/T 不复活；活动 H 不执行；Q 不反转已锁存 D。 |
| 音频 | fake player 验证首目标返回前无 begin、真实保存后才 save、HOME 成功才 home_done、故障抢占、无播放器 warning、设备关闭先于尾音等待。另列真实播放人工检查。 |
| 清理 | 注入运行主错、第二资源关闭失败、取消收尾未确认、writer/model 错误；其余资源仍尝试关闭，主 traceback 不被替换，最终码与结果一致；没有递归错误拼装。 |
| Raw | 小型真实 HDF5/video fixture 覆盖成功发布、零行、D、队列满/flush 失败；失败不发布不完整目录，不用全 fake 宣称文件格式已验证。 |
| 导出 | 正常连续段、控制时间缺口、NaN action、failed 末行、重复相机帧、触觉 hold 过期；保留行数/首个截断原因与既有规则一致，13 字段 shape/dtype 不变。 |
| 几何/数值 | 对确定性输入比较变更前后 FK、tip/EEF、动作映射、同源 RGB-D/点云与配置恢复；本轮非几何改动不得改变数组结果。畸变另按 T6 检查。 |
| Policy | fake model 覆盖 sync、async、RTC 各一条关键交接路径；非法输出仍在入口拒绝；过期尾部/RTC 承诺槽规则不变；未回收任务不进入下一段。 |
| 入口 | 相关 `--help` / `--print-config` 不连接设备；关闭录制/禁用手、单纯 RGB-D、未用 HOME 不加载无关资源。导出、查看、回放调用方没有因删除字段/错误接口而失配。 |

运行 pytest 时用 `-p no:cacheprovider`；ruff 用 `--no-cache`；mypy 的缓存放仓库外。遵守 AGENTS，仓库内不得生成 `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`，不新增忽略规则来掩盖它们。只运行与改动相关的检查，已有结果通过后不反复跑无关全量任务。

真机验收单独列出：低速开始/停止/急停、VR 新段重新锚定、S/C 保存与 D 丢弃、HOME 取消、触觉就绪/过期、音频抢占及退出。需要操作者现场执行；未执行就写“未执行”，不能拿 fake 结果替代物理停止与采样延迟测量。

## 13. 最终完成标准与报告

- [x] 音频功能和资源保留，非阻塞播放、抢占及退出回收通过 fake subprocess 回归；真实播放未验证。
- [x] T1 分类错误与陈旧身份说明修正；键盘关闭承诺与真实行为一致。
- [x] T2 适用重复检查/包装删除，删除依据覆盖所有当前入口。
- [x] T3 使用统一按键表，旧暂停/确认状态没有隐藏兼容分支。
- [x] T4 不再构造递归 cleanup 异常图；真实失败、后续清理失败和未完成资源不会被取消掩盖。
- [x] T5 留有足够实验数据，重复事件/诊断与无用硬标签减少；数值配方仍由现有配置与 checkpoint 提供。
- [x] T6 条件组合已复现；实际部署消费者待 checkpoint 核实，不把条件问题写成普遍 bug。
- [x] Canonical、Raw、FK、触觉、动作及时序含义保持；没有新框架、身份协议或回执协议。
- [x] 文档与调用方同步；工作区无测试缓存、无无关改动、无被覆盖的用户工作。

最终报告保持简短但可核查：

1. 改动与用户可见行为变化，特别是 C/Q/H 和音频时机。
2. 删除了哪些机制、保留了哪些真实边界，以及未采用的减法及原因。
3. 实际运行的检查、结果、未运行的真机检查和条件性几何结论。
4. `git diff --stat` 与核心 Python 的实际净增减；把本任务书/文档/测试与运行时代码分别说明。目标是主路径更短和总维护成本降低，不承诺拍脑袋百分比，不用搬文件制造虚假减量。
5. 剩余限制和需要用户决定的具体后续项。普通源码核实和实现选择自行完成，不把未完成工作包装成已落地。

实施完成后在本文件末尾追加一小段实际结果或勾选完成项即可，不另建报告目录、状态数据库或新的长期任务框架。

## 实际实施结果（2026-11-03）

- 从干净的 `main@da1a3be` 建立 `research/simplification-a-e`；该 HEAD 比审查基线仅多本任务书。A→E 的适用修改已落实，并同步了可视化脚本中遗漏的异常 helper 调用方。配套 Policy 仅只读核对，保留其原有修改；未操作实验数据、连接设备、发布或合并分支。
- Teleop/Policy 使用统一按键表，C=S、单 Q 保存退出、H 仅限空闲；D 在撤权锁内锁存，首次 finish 固定保存意图，收尾后清除旧 B/H/T。音频实现与所有 WAV 保留，Policy 补齐同一播放器接线，提交/发布/HOME 的实际返回决定提示时机，设备关闭先于尾音等待。
- 删除父线程 owner 断言、重复 hand 连接和 RTC 内部 shape/finite 检查、冗余 Q 查询、单次 stop 包装、暂停/恢复/确认分支、递归 cleanup 异常图、逐次操作/控制时点 dict 及失去消费者的 retargeting diagnostics。保留设备/模型/文件边界、SDK/CRC、代际/限位/停止、来源时间、单份锚点、Raw 完整发布和连续训练前缀。
- 未采用的减法：`Rollout.sensors` 仍有真实消费者；VR 就绪循环与服务检查保留；HOME 标定已按需加载，无需重做；checkpoint cloud recipe 的 exact-field gate 保留，因为 Policy 读取链未保证数值字段完整。处理器与提前准入的配置检查保持各自独立入口，不为消除初始化检查新增 validated 状态或开关。
- 自动回归命令：`PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -p no:cacheprovider -q tests`，结果 **97 passed**。覆盖真实 Robot/Replay/HOME/IPC 收尾、按键与键盘屏障、真实 AudioFeedback + fake subprocess、模型生命周期与 sync/async/RTC、真实 HDF5/MP4 发布与失败、连续导出前缀、Canonical 13 字段、FK/同源 RGB-D 和查看器关闭。测试不代表物理运动成功或真实音频可听。
- 修改过的 Python 文件及测试执行 `python -m ruff check --no-cache --select E9,F,I …` 与 `git diff --check` 通过。Teleop（含 no-record）、Policy 配置打印，以及 keyboard/replay/export/visualize/view_realsense/view_pointcloud/control_xhand 的共 10 个 `--help`/`--print-config` 检查均退出 0，不启动设备。
- 另以 `git show HEAD:<源码>` 加载修改前真实实现：128 组 joint/EEF 规范化、64 组关节映射、8 组 arm/hand FK 数组逐元素相等；手重定向使用真实标定/FK/Objective、仅替换有时限的 optimizer，8 组计算/commit/拒绝路径及退化输入与基线输出完全相同。未测量实际控制耗时，不宣称实时性能提升。
- 几何：只读 Raw `episode_20261101_174738` 确认为 640×480、非零 Brown 系数；使用真实 Real 反投影与 Policy pinhole 投影，五个固定像素的单位深度合成样例最大误差为零系数约 `2.78e-5 px`、记录系数约 `3.3044 px`。没有实际部署 checkpoint，仍无法确认当前模型是否命中该融合消费者，因此未修改图像/点云数值或添加泛化 gate。若实验确用该消费者，后续须明确选择统一 rectification 或 Policy 支持畸变；本轮不跨仓实施。
- **真机未验证**：低速启停/急停、VR 新段重锚、S/C 保存与 D 丢弃、HOME 取消、触觉就绪/过期、真实音频抢占与退出、设备时序和 NLopt 实时求解耗时，均需操作者现场验收。

### 二次调用链审查与收口（2026-11-03）

- 屏障测试复现并修复了已取得忙时代际、延迟到清队列后才入队的准备事件重新获权问题。Teleop/Policy 的启动、段收尾、HOME、触觉复位完成边界均撤销旧输入代际；不增加新状态或输入协议。Policy 本地提交取消改记 executor boundary，避免误报 policy failure。
- 同一缓存异常经 Replay 运行/停止/健康检查/关闭重抛，以及同一 recorder 首错经 Teleop/Policy 运行/finish/close 重抛时，只保留一份主错结果及 traceback；独立后错仍报告。Policy 直接在构造时持有 SessionState，删除其重复初始化和无意义的 None 分支；移除 HOME 的空赋值和重复开始准入。
- 删除无消费者的 pause/resume 音频事件接线，WAV 原样保留；收尾期间 Esc 不再触发普通 save/discard 提示。Policy 先发布设备/模型收尾结果再选择退出音频，结果发布失败或中断也会关闭播放器；新发生的音频关闭失败补记到已发布结果，不重试先前失败的发布。无录制时的普通目标拒绝仍打印一条明确终止原因。
- 删除 Canonical 导出中由 training_prefix/RawReader 覆盖的重复字段预遍历；补齐真实 writer.close 失败及 write+close 同时失败的拒绝发布验证。受影响源码无 SHA-256、历史格式双轨或自定义版本机制可删；设备 CRC、数据 shape/dtype/指序/单位、完整发布与连续前缀检查继续保留。
- 最新完整回归：同上禁缓存命令，**138 passed in 1.38s**；改动文件 Ruff、语法编译、10 个无硬件入口检查、临时源码副本的 wheel 构建与 `git diff --check` 均通过，仓库无禁止的测试缓存。定向 mypy 默认检查 5 个源文件通过，除 Session 外的 4 个源文件开启 `--check-untyped-defs` 也通过；Session 增强检查仍有 37 条生命周期资源可选字段注解问题（同组 HEAD 基线为 56 条），未用 ignore、额外断言或放宽检查掩盖，不宣称全仓类型检查通过。mypy 缓存和构建产物均在仓库外。
- 未改变 sync/async/RTC 调度和数值算法，未改配套 Policy 或实验数据。此前几何条件及真机未验证限制保持。
