# DexMani Real 轻量化架构重构任务书（Codex CLI）

> **状态**：待执行；本文件是由仓库维护者明确要求保存在仓库根目录的阶段性重构任务书，不是已完成实施报告。
>
> **分析基线**：`main` / `0f26423abdffd7846488b973f66d3ca9c1be37e3`（2026-10-23，UTC+8）。
>
> **实际执行基线**：Codex 启动时先执行 `git status --short`、`git rev-parse HEAD` 并复读当前源码；若 HEAD 与上述基线不同，先核对新增 diff，再按当前有效代码修订实施顺序。不得把历史结论当作比现行源码更高优先级的事实。
>
> **优先级**：`AGENTS.md` > 本任务书中不冲突的具体实施目标；硬件安全 > 科学正确性 > Raw/实验可追溯 > 研究迭代速度 > 代码简洁 > 扩展性。
>
> **任务性质**：真实机器人研究仓库的**等价重构优先**任务。用户授权代码层面的静态修改和无设备离线验证；**未授权**启动、连接、诊断或驱动任何真实硬件，也未授权运行 Policy 真机 rollout、遥操作、回放、标定写入、设备采集。

## 1. 任务目标与明确的非目标

目标：让 `dexmani_real` 成为“**薄而可靠的硬件运行基础 + 独立清楚的研究计算 + 直观的实验入口**”：

1. 从 `scripts/collect_teleop.py`、`scripts/run_policy.py`、`scripts/replay_episode.py`、标定入口，能顺序跟踪：解析配置 → 准备资源 → 获取观测 → 计算/选取动作 → 执行 → 记录 → 结束。
2. 一个实验事实只有一个权威所有者；不让 Session、Runner、Operator、RuntimeState、Recorder 维护含义不清的镜像状态。
3. 让 IK/FK、VR 映射、手部重定向、点云处理、Policy 调度尽可能脱离硬件 I/O、文件写入和操作员生命周期，从而可离线验证。
4. 使每项静态检查有唯一权威入口；保留不同信任/执行边界的必要动态检查。
5. **实际移除**被替代的包装、重复调用、死代码和冗余状态，而不是新增一套更大的抽象。
6. 任务分成多个可验证的小批次；按批次结束和验收，不允许一次性全仓重写。

**非目标**：生产部署框架、通用插件/registry、通用 Experiment/BaseRunner、消息总线、统一全局 Context、旧代码兼容层、历史 Raw 迁移、训练算法变更、重写 `dexmani_policy` / `dexmani_sim`、大规模改动 `third_party/cordvip_geometry`、激进删除已有控制模式和安全机制、单纯为追求文件数/行数改变目录名。`planning`、`deployment`、`xarm7` 等熟悉目录名默认保留；有职责迁移收益才移动。

**衡量标准**：新手修改一个研究算法时不需读设备关闭逻辑；修改 Policy 交接不需追踪键盘 HOME；解释失败可区分提交、SDK 接受、实测、停止与 Raw 发布；整体概念与跨模块协调减少。**不设必须减少多少行的硬性 KPI**。

## 2. 现状与最新提交中必须继承的事实

当前硬件与工作流：xArm7 (7 DoF) + XHand (12 DoF) + RealSense 对齐 RGB-D + VR/HTS；标定 → Teleop Raw → Canonical 15 字段 Zarr → `dexmani_policy` 训练 → 真机部署/评估。`third_party/cordvip_geometry` 为独立感知扩展，尚不是主控制链路的一部分。

`0f26423` 相对上一提交 `6ef38ff8` 的重要增量，**不得通过重构撤销**：

- 无 Raw 录制时，部署触觉新鲜度按模型真正请求的 `aggregate`、`dense` 通道准入；启用 Raw 录制时仍要求两类触觉。手部关节反馈始终是基本条件。`XHandFeedback` 仍消费全部已接入触觉数据，不意味着关闭 SDK 模态。
- 相机支持只读共享 ring 的 `header`。仅需相机来源时钟、内/外参且无 Raw/RGB 需求的路径不复制完整 RGB-D。若同时需要 RGB 与点云，继续以相机 sequence 保持同源。
- `teleop`、`deployment`、`replay` 的键盘监听已提前至设备连接前；`robot.connect`、`xarm7.executor.connect` 与传感器 ready 等待支持更及时的取消/服务健康检查。
- 模型启动新增有界等待 `model_startup_timeout_s`。这只是等待预算，**不保证**阻塞 Python/CUDA 线程被终止。
- 部署最终化时更准确区分 `start_cancelled` 与 `preparation_failed`；不得把准备故障统一伪装为操作员取消。
- 当前 `teleop/keyboard_session.py` 和 `calibration/camera/session.py` 的监听仍在 `robot.connect()` 后启动：属于独立启动行为修正候选，不能把它与纯移动文件混在一个“等价”提交中。

**当前已识别的具体耦合**（执行前按 HEAD 核查）：

- `config/experiment.py` 从 `deployment/config.py` 导入 `ExecutionConfig`，基础配置反向依赖工作流。
- `scripts/run_policy.py` 与 `deployment/session.py` 都调用 `validate_deployment_config`，属于需要归一的重复预检；其中第一次预检已在创建会话输出路径前。
- `runtime/state.py` 的 `RuntimeState` 既管理运动代际/授权，又直接持有 `SensorChannels` 并控制其运行标志。
- `runtime/observation.py:read_observation` 构建数据快照时隐式更新 `robot.arm.update_health(upstream_until_ns=...)`，读取与执行器服务副作用混合。
- `deployment/session.py`、`deployment/operator.py`、`deployment/runner.py` 跨对象共同决定准备、开始、HOME、结束和产物收尾。
- `deployment/runner.py` 混合 `PolicyQuery/ActionPlan`、模型 Future、预取/交接、动作映射、硬件发送、时钟预算、录制与评估。
- `ipc/schema.py` 混合机器人反馈 dtype 与相机/VR/点云 IPC dtype；`robot/xhand/feedback.py` 因此依赖 IPC 的手部 dtype。
- `robot/action.py` 是无设备 I/O 的动作映射/准入逻辑；不应与真实 SDK 发送混为一层。
- `data/recorder.py` 中“capture 尝试已经发生”和“writer 已经 recording”是不同事实，**不能直接消去上层启动尝试的收尾责任**。

## 3. 严禁变化的安全、时间、物理与科学契约

### 3.1 硬件

- 保留 xArm 独立 SDK owner 进程、Ruckig 参考轨迹机制、前后授权栅栏、准备/执行 Epoch、超时与服务/反馈截止时间。主线程向 xArm **提交目标**，不等于 SDK 已接受，更不等于实测到位。
- 保留 XHand Serial/EtherCAT 的现有 I/O 所有权、硬/命令关节限位、CRC 未确认含义、停止结果、触觉消费规则。`CRC_UNCONFIRMED` 既不是 ACK 也不是实测到位。
- Stop/Q/ESC、故障撤销、过期观测/预测/命令、丢时隙、不追赶补发或重复末动作的语义不变。停止请求必须能绕过长模型/SDK 等待而及时撤销软件授权；**不能承诺**真实物理即时停机。
- 普通 Online IK / TargetChecker 的目标端点准入 ≠ HOME 的插值路径检查；保留 HOME 确定性路径选择（direct / proximal_first / distal_first）、现有桌面接触规则和实测收敛检查。
- 不改控制频率、坐标变换、动作投影、关节顺序、limit 数值、EMA、TAG/DexPilot 优化/滤波、Ruckig 参数、碰撞几何或硬件 SDK 结果映射，除非另立明确的行为变更并取得真机验证安排。
- 操作员监听回调只进行可安全触发的停止/撤权信号，不在回调线程执行阻塞 SDK；启动取消条件要覆盖设备连接和等待的真实边界。

### 3.2 时间与调度

- `sync`：动作段与下一轮推理之间的真实 WAIT 不得被偷偷补齐或等间隔伪造。
- `async`：新预测提前完成仍按计划交接；迟到接管使用 `slot - query.slot` 对齐预测索引，必要时继续消费未过期旧预测尾部；不能重头重放预测或无条件重复上一动作。
- `rtc`：保留预取窗口、物理关节命令冻结、承诺前缀和指导条件；不能在执行时悄悄重投影为不同等价角/IK 分支。
- 区分 `slot deadline`、反馈新鲜度、模型预测决策时限、worker service deadline、episode duration/wait budget；可集中计算 `min(...)`，**不能**合并成意义不明的一个 timeout。旧来源时间不得被新处理时间刷新。
- 旧 run 的模型 Future 迟到后必须被正确回收/丢弃；不可被下一轮当作新预测执行；模型线程 close pending 不等于已经退出。

### 3.3 Raw、Canonical、评估

- 已发布 Raw 不可原地写回；失败、晚帧、零行、无动作 WAIT、停止和终止原因仍属真实证据。仅操作者显式丢弃未发布 capture 才删除。
- 保持 Raw `data.h5` + `rgb.mp4` 同行对齐、HDF5 真实单调时间戳、`arm_execution`、dispatch、query、timing、结果/终止字段语义。不得将 `SUBMITTED` 冒充 `ACCEPTED`；已尝试部分发送要保留实际回执。
- Recorder 写盘队列满、视频行数不符、执行结束无法确认、异常中断时，不重复写已入队行、不宣称完整发布；按现有规则保留 staging。传感器 child 的 `terminate/kill` 与 xArm 停止确认具有不同物理边界，不能通用化为“一律 kill”。
- Canonical 继续导出完整 **15 字段**，保持 shape/dtype、坐标、相机 geometry / depth scale、处理 recipe、`episode_ends`、整段技术准入和不修补缺测策略。Policy 真机观测允许使用训练所需的在线字段子集，但不能改变公共 Canonical 契约。
- 保留当前点云 `build_point_cloud` 的线上/离线共用实现；重建和训练配方必须一致，缺失 recipe 不以当前默认值补齐。
- 相机的 Raw `color → xArm-base` 外参与 Canonical/Policy `base → color` 不可互换。历史 Raw 使用录制时相机几何，部署使用当前现场标定。
- 注意源码事实：运行时拥有 `tactile_aggregate_timestamp_ns` / `tactile_dense_timestamp_ns`，当前标准 Raw `timing/rows` 并未逐行存储这两个独立有效时间。不要在本次等价重构中假定已保存；如需要补充则作为另案数据格式/文档问题。

## 4. 目标责任边界与依赖方向

**首选保留现有顶层领域目录名** `robot`、`sensor`、`planning`、`runtime`、`teleop`、`deployment`、`data`、`calibration`、`replay`。只在职责发生实际变化时移动文件；如小型移动导致大量 import 变更而无明显收益，可以不移动并说明理由。

- **robot**：设备 SDK、设备执行进程、机器人反馈与机械限位、运动授权、RobotCommand / DispatchResult、真实发送/停止/HOME 执行原语。绝不依赖 deployment、teleop、data writer。
- **sensor / ipc**：相机、点云、VR 的 producer 与 IPC；只负责采集和传输。可保留 `ipc/` 单独目录；不要强制把不同通道实现合成单文件或消息总线。
- **planning**：Pose、FK/IK、碰撞、TargetChecker、ActionIntent/ActionMapper 等目标转换与几何准入；不进行 SDK I/O。`robot/action.py` 移动到这里是优先候选。
- **runtime**：多源 ObservationRow、需求快照、读取与健康服务衔接、键盘输入、传感器进程资源、通用 Jog；不实现 Policy horizon 或模型加载。
- **teleop / deployment / replay / calibration**：各自实验流程的明确资源装配与控制循环；可以共享基础能力，不引入跨模式的 BaseRunner。
- **data**：录制、Raw 读写/格式、Canonical 处理与结果；不触发硬件。`EpisodeResults` 可归 `data/results.py`。
- **config**：基础配置定义和合并；`ExecutionConfig` 放在 config（例如 `config/execution.py`），部署兼容性校验仍留在 `deployment/preflight.py`，避免 `config` 导入部署工作流。

### 4.1 “唯一所有者”规则与反例

| 事实 | 权威所有者 | 不能错误合并的快照 |
| --- | --- | --- |
| 运动授权、run_id、撤销 | 机器人授权对象（原 RuntimeState 内的职责） | 命令携带的历史 run_id |
| xArm SDK/执行与已接受计数 | xArm worker | 上层 `SUBMITTED` 回执 |
| 当前录制/写入/发布 | AsyncEpisodeRecorder | 启动尝试（尚未 ready） |
| 当前模型 Future | InferenceWorker | Query 的 query_id / run_id |
| Policy 预测动作、交接 | 当前部署执行逻辑/纯调度计算 | 上次成功提交的关节目标 |
| 输入请求 | KeyboardInput | 实验流程的当前业务阶段 |
| 正式轮次 | 当前实验流程/Results | 已启动 capture 的编号 |

**一个所有者 ≠ 只出现一个变量**；必要时间快照不能删除。没有证据证明重复的布尔/状态不得机械删除。

## 5. 一个小而必要的“观测需求快照”

`0f26423` 已经支持按需触觉、相机 header 与 RGB/点云同源；现行需求推导仍分别存在于 Session / Runner / read_observation / hand feedback。此处是值得收敛的真实重复。

建议在现有 `runtime/observation.py` 或恰当现有小模块中定义一个**不可变小 dataclass**，例如：

```python
@dataclass(frozen=True)
class ObservationNeeds:
    camera: Literal["none", "header", "rgbd"]
    point_cloud: bool
    vr: bool
    required_tactile: tuple[str, ...]
```

仅是目标接口示例；可以使用更简单的等价表达，不能机械加上后仍保留原重复 flags。需求由当前 Policy `observation_fields` 与是否保存 Raw **一次计算**，然后供设备装配、观测采样和需要的触觉时效判定使用。

推导原则：
- 相机 worker 当且仅当 Raw 或观测/点云需要真实相机；
- 需要点云但不需要 RGB/RGB-D 时，只获取点云（按需要读取相机来源元信息），不额外复制全幅图；
- 需要 camera K/E 且无需图像时读取 header，来源时钟仍来自真实相机；
- 需要 RGB 与点云（或 Raw+点云）时按 camera sequence 保证来自同一帧；
- Raw 录制必须要求 aggregate + dense 两类触觉；无 Raw 时仅按模型请求的触觉要求时效；手部关节反馈始终必要；
- 若要获取点云或模型要求 E，必须检查匹配 serial 的有效现场标定；仅保存物理 Raw 时允许缺相机外参；
- `camera="header"` 指“协调线程仅复制 header”，不代表相机 worker 不采 RGB-D；
- 触觉数据仍可被读入缓存并以 NaN/时间事实留存；“不要求该通道新鲜”不意味着可以虚构有效值。

必须验证下表所有主要模式的**相机是否启动 / 是否复制 RGB-D / 必需触觉 / 配对帧身份**：
- joint-only no-record；
- joint + finger tactile no-record；
- joint + taxel no-record；
- camera K/E-only no-record；
- cloud-only no-record；
- RGB-only；
- RGB+cloud；
- recorded rollout；
- teleop recording / teleop no-record；
- replay（不应启动 VR / Camera）。

## 6. 分批实施计划与验收门槛

### Phase 0 — 基线与审计（无源码语义修改）

1. 阅读 `AGENTS.md`、`README.md`、本任务书与当前 HEAD；查看 `git status --short`。保留现有无关工作树修改。
2. 追踪 `producer → transform → consumer → side effect`，为 Teleop、Policy、Replay、Camera calibration、Raw→Canonical 建立简洁的临时调用链/依赖清单（可在最终答复中报告，不必另存根目录临时文档）。
3. 列出拟删除的每个变量、判断和包装的所有使用点；区分完全重复与不同物理/时刻事实。
4. 确认全部安全、数值、Raw 与算法不变项；若当前 HEAD 改变则优先核对 diff，避免基于旧源码批量改写。
5. 执行无设备的基线检查：`python -m compileall -q dexmani_real scripts`；CLI `--help` / `--print-config` 仅在确认不会连接设备且环境依赖可用时运行；`git diff --check`。
6. 用纯逻辑/假时钟/替身构造预期行为基线，优先覆盖模型迟到、权限撤销、启动失败及 Raw 的准入和回执；不安装新框架或跑真实设备。

**Exit**：可逐项解释保留或删除的真实原因；当前基础检查结果和依赖缺失被记录；未改控制语义。

### Phase 1 — 安全启动顺序的独立修正（行为修正，与目录搬迁分开）

仅在静态检查确认现状后：
- `teleop/keyboard_session.py`、`calibration/camera/session.py` 调整为在 `robot.connect()` 与阻塞 camera ready 之前安装并检查键盘停止监听；
- 同步考虑 `start` 失败前对象尚未完全初始化的 `close` 分支，监听回调不得直接在非 owner 线程执行阻塞设备操作；
- 与现行 `teleop` / `deployment` / `replay` 保持取消/撤销语义一致；给 xArm worker 的启动授权信号不得在收到较新停止之后重新发布；
- 这是可观察的行为修复，不宣称与基线完全等价，也不得用离线结果声称真机停机安全。

**Exit**：使用假设备与事件确认连接前、连接期间、相机等待期间的停止请求不会被忽略；真实设备验证仍标记 pending。

### Phase 2 — 依赖和静态职责归位（原则上纯结构）

优先小步：
1. 将 `ExecutionConfig` 的定义归入 `config`；删除 `config/experiment.py -> deployment/config.py` 的反向依赖；部署专属兼容校验留 `preflight`。
2. 将 `ActionMapper` / `ActionIntent` 等无硬件 I/O 的计算语义归入 `planning`（若改动与实际接口冲突，允许保留文件但要明确计算/执行依赖边界）；保留 `RobotCommand` / `DispatchResult` 在 robot。
3. 将机器人反馈 dtype 从传感器 IPC dtype 语义中剥离到 robot 内，或在确认现有共享类型的真实消费者后做最小移动；不要引入全仓 `types` 框架。
4. 将 `utils/episode_results.py` 中实质属于数据评估的逻辑归位 `data/results.py`；其他 `utils` 有真实跨模块用途可保留。
5. 对 HOME 区分纯路径计算（`planning`）、真实设备调用（`robot`）和运行协调（`runtime`）；如果强行移动导致额外包装，则保持 `robot/home.py` 位置，先让函数职责清楚。
6. 搬迁完成必须更新当前 import、脚本与 README；不留永久旧路径 shim、重复文件或多套调用路径。

**Exit**：导入关系更单向；无硬件 import/compile/config checks 通过；所有数值数据/配置形状保持不变。

### Phase 3 — 需求快照与运行状态收缩

1. 按第 5 节生成一个只读的 `ObservationNeeds`（或更小等价接口），替换多处推导；不要另建 ResourceNeeds/HardwareCapabilities/PolicyRequirements 等同义对象。
2. 将 `RuntimeState` 对 `SensorChannels` 的直接持有/关闭职责交回实验资源所有者；先保持现有 motion lock、epoch、停止事件与 xArm authority 协议。只有在减少交叉依赖、且假时钟/竞态回归充分时，才提取为 `robot/authority.py`；**不能**并存两个可写权限中心。
3. `read_observation` 的上游期限写入改成明确可见的服务点（若确实减少隐藏副作用），但必须保持 `bootstrap/WAIT/ACTIVE` 的刷新时机和实际来源时戳，不得因抽取而省略。
4. `RuntimeState.end_session` 不再直接拥有传感器进程运行状态；实验资源作用域先撤销运动再通知传感器退出，保留现有失败升级语义。

**Exit**：需求组合矩阵通过；有且只有一个授权所有者；没有新增会话全局 Manager；等待/反馈与发送前期限检查的回归通过。

### Phase 4 — 部署公共入口与业务生命周期归一

1. `scripts/run_policy.py` 保留 argparse、`--help` 和 `--print-config`；将非 CLI 的 Policy inspect、有效配置解析、一次 preflight、会话产物创建、run_config 写入与装配顺序集中到一个明确的 `deployment/run.py:run_policy(...)`（或最简等价入口）。**确保唯一公共入口从 CLI 和直接 Python 调用都先预检后产生设备/输出副作用**。
2. 删除重复的 `validate_deployment_config` 调用；不引入 `validated=True` 或验证包装类。当前 `save_run_config` 反向回调应由有明确输出目录所有权的层直接完成，不额外抽象。
3. 将 `PolicyOperator` 中真正属于业务的 `begin/home/reset/stop` 判断归当前 deployment 流程；保留 `KeyboardInput` 的异步输入与直接停止/急停撤权回调（不能依赖下一轮 poll 才停）。
4. 明确 `PREPARING / RUNNING / STOPPING / IDLE` 含义；删除经证明等价的派生状态，但 `capture started attempt`、`writer ready`、`entered RUNNING` 分别保留其必要事实与收尾。
5. 允许 Session 以资源所有者身份保留，若它只保存构造参数再调用一次 `run` 才改为函数；不强行将现有 ~1000 行 Runner 合并到一个更大的 `run.py`。
6. Teleop 同样使 Session/Runner 的资源与算法分界清楚，保留当前 B/C/S/D/H/T/Q/ESC、锚点、EMA 和录制分段含义。
7. Replay/Calibration 不继承 Teleop Runner；只复用已存在的 HOME、Jog、观测和机器人发送原语。

**Exit**：只有一条部署业务启动路径、一条每轮收尾责任链；同批次 Stop/B、HOME 阻塞后的旧命令清理、模型 reset 准备、录制启动失败/零行保存均保持真实结果。

### Phase 5 — Policy 调度计算独立与长 Runner 减负

1. 将 `PolicyQuery`、`ActionPlan` 的真实调度计算（slot 对齐、handoff、prefetch、plan exhausted、deadline）从设备/录制/键盘操作中拆到 `deployment/schedule.py`，函数接收明确时钟/slot/查询对象；不自行做 SDK I/O、Future 提交或写文件。
2. `deployment/runner.py` 或 `loop.py` 保留控制周期读取、模型提交/回收、ActionMapper、真实 dispatch、Recorder 及结束策略。逐步抽离、即时验证；只有在删除对应旧函数后才新增新文件。
3. RTC 前缀拆成“调度请求前缀 → 动作准备 → 提交/冻结 → 执行重新准入”，不可让调度模块持有 robot 或给模型的前缀和冻结物理命令分叉。
4. 从 sync/async/rtc 共同控制路径删除非必要重复状态判断，但**不改变**当前模式交接和 WAIT 区别。
5. `InferenceWorker` 仍是单独的模型线程 owner；已运行 CUDA / Future 不可被 `shutdown(wait=False)` 保证立即中断。已结束轮次的迟到预测必须按旧 run_id/query_id 回收。
6. 仅精简与模型调度无关、可证明没有消费者的冗余元数据与异常字符串；当前 Raw `query`、`command_seq`、`arm_execution` 的证据不可删除。

**Exit**：通过第 7 节纯逻辑矩阵；不会因重构改变预测序列索引、采样时钟、发送次数、已冻结命令或任何录制行的动作语义。

### Phase 6 — 结束路径、文档与清理

1. 每个工作流有一个**业务所有者**负责“原因判定 → 撤权 → 设备停止确认 → 末行/部分回执留存 → 保存或 staging → 评估”；并非新增一个全仓 `finish_everything`。
2. 删除无意义的异常转译、重复日志、镜像 flag 与失败后伪恢复；保留 SDK 部分执行证据、异常中的清理和继续尝试关闭其他独立资源的处理。
3. 只修改用得到的 `README.md` 链接/最短实验路径；`AGENTS.md` 作为长期规则不复制到 README/多个文档。`CLAUDE.md` 保持简短。
4. 删除旧路径、失效 alias、未使用类、未使用导入与死代码；绝不额外引入 `pytest`、`ruff`、Registry、通用事件总线。
5. 对所有改动给出“变更前后 producer → transform → consumer → side effect”映射，并在最终说明中区分：离线已验证、需要人工确认、真实硬件尚未验证。

**Exit**：没有新旧双实现并存；核心实验入口更直观；目标责任归属可以从几个关键文件说明，而非必须阅读整个仓库。

## 7. 最小验证矩阵与实验等价性标准

使用无硬件副作用的纯 Python 临时脚本、FakeClock、FakeXArm/FakeHand、FakeKeyboard、FakeInference、FakeRecorder、合成 Raw 数据验证。可将稳定且确有维护价值的脚本保留在现有合理位置；中间诊断放系统临时目录，不提交根目录。不要安装/运行 pytest 或 Ruff，不得连接设备。

必须覆盖：

| Case | 期望 |
| --- | --- |
| 配置与 `--print-config` | CLI > YAML > 默认；打印不加载 Policy、不读取现场标定、不连接设备 |
| 设备连接前/期间 Stop-Q-ESC | 新启动不越过已撤销的授权；不在监听回调执行阻塞 SDK |
| 模型 warmup 超时 | 有明确 pending 事实；不把线程停止当成成功 |
| joint-only + no-record | 不启动 camera/pointcloud/VR；无用触觉缺测不阻止运行，但真实关节反馈必须有效 |
| 仅 finger / 仅 taxel + no-record | 只要求对应触觉通道新鲜 |
| Raw 开启 | 相机 RGB-D 和两类触觉准入一致；其他物理模态缺测按 Raw 规则保存 |
| camera K/E-only | 使用 header-only 读取并保持真实相机来源时间 |
| cloud-only | 按 recipe/当前标定构建点云，避免不必要的 RGB 拷贝 |
| RGB + cloud | 同一 camera sequence，不能混帧 |
| teleop no-record / replay | 不额外启动不需要的相机/VR |
| async 提前完成 | 仍在预定 slot 交接 |
| async 迟到接管 | `prediction_index = current_slot - query_slot`；旧预测未过期尾部可继续，不能重放 0 号动作 |
| rtc 预取与冻结 | 前缀承诺物理命令对应训练/推理表示，拒绝错误重投影 |
| Model result 属于旧 run | 不进入新 run |
| 等待/漏 slot/过期 | 不追赶补发、重发最后动作或延长 decision deadline |
| xArm submitted，XHand 失败 | 保留部分 DispatchResult，撤权并请求停止 |
| xArm SDK 未确认 | 不宣称真实到位/完整收尾 |
| 录制启动失败、零行、队列满、视频数不符 | 有准确尝试记录或 staging，不重复入队、不虚假发布 |
| Raw→Canonical | 15 字段 shape/dtype/数值/时间/完整性及整段拒绝规则不变 |
| 资源关闭故障 | 其他资源仍尝试清理；未确认物理停止保留 unresolved，不能认定成功 |

**等价性比对**：控制/录制事件次序、关节目标和表示、决策索引、期限拒绝、HOME 规则、Raw 列及语义、episode 结果；不以实际 wall-clock ns 完全相同作为验收。不允许把离线冒烟、SDK 模拟或者编译通过称为“真机验证”。

每批至少跑：
```bash
python -m compileall -q dexmani_real scripts
git diff --check
```
并报告对应纯逻辑验证、环境缺失、无法运行的原因与未验证风险。`git diff --stat` 与 `git status --short` 用于检查范围和未预期修改。

## 8. 具体删改优先级与风险闸门

**高价值、较低风险**：重复 Policy 静态预检合并；`ExecutionConfig` 归位；删除反向保存配置回调；理清同义目录与真正使用方；按需观测需求统一；清理无消费者日志/包装。

**高价值、中高风险**：`RuntimeState` 从 sensor 运行控制解耦；提前 jog/calibration 的停止监听；`read_observation` 的健康服务副作用显式化；PolicyOperator/Runner 职责归一；每轮 Recorder attempt 的收尾。

**高风险，应晚做且需要严格验证**：运动授权归属/锁和跨进程信号变化、xArm worker 协议、MODEL Future 的关闭/迟到语义、RTC 前缀承诺、Raw writer 事务/发布。若只是“目录更漂亮”则**不做**这些内部重写。

**暂不纳入**：从 sync/async/rtc 删除研究模式；TAG/DexPilot 算法切换与删除；Canonical 的 15 字段变成按模型动态筛选；触觉保持策略及历史时钟修复；重新设计 HOME 路线；更改 xArm 控制器碰撞灵敏度；新历史兼容层；CORDViP 接入主运行时。

遇到以下情况立即停止当前批次并报告，不可无声降级：
- 无法证明某状态/校验没有独立消费者或安全语义；
- 需要更改机器人可观测的运动行为，但未获得明确的实验变更授权；
- 纯逻辑回归失败、模拟下仍存在不确定执行；
- 当前工作树存在无关修改且实现要求覆盖它；
- 无法确认真实设备的停止状态，却必须依赖它宣称已完整 Raw。

## 9. 输出与交付要求（每一批次）

每批次给出简洁、精确的中文结果：

1. **本次范围**：实际触及文件与职责。
2. **删除了什么**：已删除的重复调用/状态/包装及其原使用点，说明为何不再需要。
3. **保留了什么**：受影响的运动权限、时间、动作、坐标与 Raw 约束。
4. **依赖变化**：关键 `producer → transform → consumer → side effect` 前后图或文字。
5. **验证**：执行的具体无设备命令、用例、结果和未运行原因。
6. **剩余风险**：哪些需要真实设备人工验证，**不要声明已真机通过**。
7. **下一批次**：仅建议下一项最小有界修改，不自动扩大研究范围。

推荐按 Phase 提交小而清楚的本地 commit（保留现有无关工作树修改；提交/推送必须遵循当前执行环境的 Git 授权）。严禁自动 `git push`、强制推送、删除远端分支、覆盖 Raw 或运行设备脚本。先有可比对的行为基线，再移动职责；纯重命名与安全行为更改不要混为一个提交。

## 10. 给 Codex CLI 的执行策略

- 仓库根目录启动，先读 `AGENTS.md`、`README.md`、本任务书、`git status` 与当前源码；
- 制定具体的 file-level 计划，不重新把本任务书改写成新框架；
- 顺序完成 **Phase 0**，然后选择**最小且可离线验收的 Phase 1 或 Phase 2 首批**进行实际修改；
- 每个阶段达成 Exit 条件才进入下一阶段。可以在同一 Codex 会话连续执行可证明安全的批次，但不要为了“一次完成”越过风险闸门；
- 不要只输出建议，应实际实施当前有条件完成的代码修改；若遇硬件依赖/语义不确定，交付已完成的安全部分与明确阻塞项；
- 最终不得修改本任务书来把未完成工作伪装成已完成。此文件是本次由用户指定的根目录任务书；没有用户要求，不要删除或新建大量根目录临时报告。

---

**判定成功的最终形态**：一个权威的运动授权边界；一个被复用的观测需求决定；每种实验一个明确的业务入口和结束所有者；研究映射与调度无需硬件就能验证；部署/示教流程短而直接；Raw/Canonical 和物理时间语义完全可信。少的是重复协调与隐含副作用，**不是**安全保证或科学事实。
