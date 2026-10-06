# Codex 任务书：修复策略评测链路，精简机制，引入可切换异步与 RTC

日期：2026-10-04（Asia/Shanghai）  
主仓库：`haoyangzhanglab/dexmani_real`  
协同仓库：`haoyangzhanglab/dexmani_policy`  
状态：历史设计与实施证据保留，不作为当前待办清单；当前接口见 `docs/policy_execution.md`，后续实施结果见 `CODEX_RESEARCH_REFINEMENT_TASK_20261007.md` 第 17 节。**历史设计通过不代表 GPU 推理或真机验收。**

本文独立承载实施要求，不依赖聊天记录或临时审查脚本。本任务范围内，以本文取代前几轮存在冲突的建议；不修改 `AGENTS.md` 的长期规则，也不将旧根目录整改文档全部重新执行。

## 1. 目标、版本与工作边界

按顺序完成：

1. 修复已确认的终止原因和训练数据消费缺陷，补齐决策年龄准入。
2. 删除不可达分支、无效扫描和已证实冗余的复制，保留真实边界。
3. 用一个设备 owner、一个模型 worker 和一个在途 Future 替换阻塞主循环。
4. 支持 `execution_mode: sync | async | rtc`，**默认 `sync`**。同步推理本身不是 bug；引入 worker 不等于开启异步动作分块。
5. 在同一调度器上实现原采样器的异步基线和推理式 RTC，形成可解释的对照。

审查源码基线：

| 仓库 | SHA |
| --- | --- |
| dexmani_real | `72914e57461d5fd75469dbce181689873d5d4548` |
| dexmani_policy | `f1ee84f43247f33df73399b62b616e25f5ca3caa` |

SHA 是证据定位，不是要求 reset/checkout 的目标。先检查两个工作树、当前分支、各级 `AGENTS.md` 和上述基线后的相关差异；保留用户已有修改。当前源码若已修复某项，证明后跳过，不能为了符合旧结论重复改写。

真实入口为 `dexmani_real/examples/run_policy.py`。模型、normalizer、训练和采样算法属于 Policy；设备、时间调度、动作实现、Raw 记录属于 Real。跨仓库改动必须配套完成；不能只改 Real 然后用未经实现的参数假装 Policy 支持 RTC，也不在 site-packages 中临时打补丁。若协同仓库或必要环境不可用，完成可独立验证的部分，并精确列明阻断项，不虚报整体完成。

不自动连接设备、回零、rollout、回放、标定或采集；不启动大规模训练或批量评测，不升级实验环境来凑测试。原始数据、checkpoint、实验目录保持不变。本任务不引入训练式 RTC、新模型结构、新的平滑控制器、远程推理 RPC、通用 registry 或插件体系。

## 2. 已确认问题与不应重复整改的现状

| 类型 | 证据路径与行为 | 实施要求 |
| --- | --- | --- |
| 已确认 bug | Real `deployment/runner.py`：推理后和 pacing 中撤权，部分路径以 `motion_revoked` 覆盖操作者 S/Q 主因 | 统一终止原因来源，见第 3 节 |
| 已确认数据消费缺陷 | Policy `datasets/base_dataset.py`、`sampler.py`：主训练窗口没有按实际输入/监督角色拒绝 NaN | 在真实训练索引路径筛选，见第 4 节 |
| 已确认信息丢失 | Policy `datasets/replay_buffer.py`：未读取 Real 导出的 `row_info/dispatch_status` | 保留并在默认新数据配方中消费；不改 Raw |
| 统计口径需修正 | `BaseDataset.iter_normalization_data` 与 `training/build_utils.py` 从完整 buffer 拟合，包含验证集及被排除 episode | 新训练使用有效训练源行统计；明确这是新实验口径 |
| 部署语义缺口 | 采样时 freshness 和推理后新关节反馈，不能证明该次策略输入仍适合下发 | 绑定 query 源时间并做决策年龄准入；不将其描述成已发生的物理事故 |
| 架构限制 | 阻塞 `predict()` 占住设备 owner | worker 释放 owner，但不得宣称消除所有 SDK/GIL 阻塞 |

以下不是本基线的待修 bug：已有 run_id 复核、停止时清队列、逐设备 SDK 前授权检查；没有发现应用层 XHand 自动无限重发循环；模型字段已经按需选择；RGB 验证/部署已共用 `preprocess_validation_rgb`；在线/离线点云已共用 `build_point_cloud`；Policy 最新基线已固定 best record 与 checkpoint 的单次解析。保留这些正确行为。

## 3. 终止、时间来源与必要边界

### 3.1 终止主因

主线程唯一负责结束 episode。优先采用已锁存、属于当前 run_id 的 `RunEndReason`；没有更早主因时才锁存当前终止事件。发现点（例如 `after_predict`）、`motion_revoked`、stop/recorder 清理异常作为详情保存，不能替换 OPERATOR/QUIT 等原始主因。结束与 episode 计数只能发生一次。迟到 worker 结果无权结束或改写后续 episode。

### 3.2 决策年龄

query 提交时冻结输入数组及其来源。检查**每个实际输入模态在最新观测槽位引用的源样本**；不是所有模态 timestamp 的最大值，也不是整个 N 步历史的最老时间。

- arm/hand 当前时间来源是 host monotonic 的 SDK 读取完成时间。
- camera 当前时间来源是 host monotonic 的队列返回时间；pointcloud 继承其源 camera 时间。
- 派生 FK 模态继承其实际依赖的 arm/hand 源时间；不能以 FK 完成时间刷新。
- 这些时间可在同主机比较，但不等于设备真实采集时刻；不得混减 RealSense device timestamp 与 host monotonic。
- 结果完成时间、新关节反馈、点云计算完成时间都不能刷新旧 query 的年龄。
- 仅为录制启用的相机不自动成为模型决策年龄条件；录制自身要求独立保留。

使用冻结来源，在结果准入及每次实际下发前检查按当前 host source timestamp 计算的年龄。历史槽位在当时应有效；不要用当前时刻对所有历史槽位再套“最新帧年龄”。保留当前执行反馈 freshness、源帧身份和 RGB/cloud 对齐要求。

区分两个有限正数预算：`max_decision_age_s` 限制策略依据的年龄；`max_wait_s` 限制需要动作却没有可执行计划的等待。现有 `max_running_s` 仍是 episode 总时长，不代替二者。复用现有配置入口保存这些少量参数，不建立 budget/contract framework。硬件可接受的具体数值不能由论文时延或合成 warmup 伪造；无合适现有值时作为显式实验配置，执行入口在连接设备前要求配置，配置检查及离线验证仍可完成。预算与动作段长度 A、模型时延不相容时给出清楚诊断，不能静默放宽。

### 3.3 保留的边界

保留 shape/dtype/unit/frame、必要浮点输入/输出有限性、硬件限位、当前授权、执行反馈 freshness、逐设备 dispatch、停止未确认状态及 Raw 原始证据。所有设备目标在第一条 SDK 下发前完成必要预检查；逐设备临发授权检查继续保留。

撤权不能撤销已经进入 SDK 的调用；不要把 motion_lock 持有到可能阻塞的 SDK 调用结束，也不能承诺“按 S 返回后绝无 SDK 穿越”或“设备已经物理停稳”。这些不是多加一层 Python contract 能解决的问题。

## 4. 数据修复：角色化窗口与训练集统计

改动面：Policy `datasets/replay_buffer.py`、`base_dataset.py`、`sampler.py`、`training/build_utils.py`，沿现有最短调用链修改。`examples/read_policy_windows.py` 是独立示例，不能代替主训练路径修复。

1. 对 Real canonical 数据读取 `row_info/dispatch_status`，保留原始行索引和 episode 边界；row_info 不进入模型输入。仿真数据是不同数据来源，不要求不存在的硬件 dispatch。
2. `obs_valid` 只检查模型实际启用的模态；浮点输入必须有限，已确认 uint8 的 RGB 不再扫 finite。`action_valid` 检查完整实际监督目标，包括启用 aux EE 时的 `action_ee[..., :9]`。
3. 设 N 为有效 `n_obs_steps/obs_horizon`、H 为 loss 的完整动作 horizon。对候选窗口的原始源行映射 r，准入为 `obs_valid[r[:N]].all() and action_valid[r[:H]].all()`，不能把部署长度 A 当作监督长度，也不能要求后 H-N 槽未使用的观测有效。
4. 复用原 sampler 首尾 padding。当前 `(buffer_start, buffer_end, sample_start, sample_end)` 对应 `r = clip(buffer_start + arange(H) - sample_start, buffer_start, buffer_end - 1)`。先验证与当前 sampler 等价再复用；不删除内部坏行后重新拼接，不跨 episode，不改变 padding loss 权重。
5. 本次新 Real demonstration 默认配方为 arm、hand 均 `ACCEPTED == 1`。`CRC_UNCONFIRMED == 2`、UNKNOWN、REJECTED、NOT_CALLED 不算 accepted；不能复用允许 CRC_UNCONFIRMED 的 `continued`。接受仅表示 SDK 确认接受，不证明到达。缺少 dispatch 不静默伪造 accepted；明确报告该默认配方所需信息不足。
6. 先 episode split 和 `max_train_episodes`，再分别建立并过滤 train/val 窗口，再拟合 normalizer。观测统计取有效训练窗口 `r[:N]` 的唯一源行；动作统计取 `r[:H]` 的唯一源行，包含启用的辅助监督。源行去重，不能按窗口重叠或 padding 重复加权。
7. 验证集使用训练统计。不得使用验证行、被排除 episode 或未被有效训练窗口引用的行拟合；不得用 nanmean/NaN→0 掩盖无效数据。输出候选/有效窗口数及拒绝原因；零有效窗口明确报错。

accepted-only、角色化过滤和去重的 train-only stats 是**新训练配方**，记录在 resolved config/数据摘要即可，不建立 schema registry 或旧缓存 migration。旧 checkpoint 始终使用保存的 normalizer，不加载新数据重拟合，不覆盖历史结果。普通异步和推理式 RTC 不要求强制重训；以新数据配方训练得到的模型是新的实验身份。

本节不额外改变时间重采样、数据增强、点云算法、RGB 预处理、loss mask 或标定语义。公共导出仍保留全模态、辅助 NaN 和失败证据；Raw 不回写。

## 5. 可删除和可替换的机制

优先完成可证明的小型精简：

- 删除 `_preprocess_rgb_cpu` 中验证集提前 return 后不可达的 `_is_val` 分支及恒真的条件。
- 删除 uint8 RGB 上无意义的全量 finite 扫描。
- 在查清独立所有权、contiguous 和非 in-place 变换后，删除重复 NumPy copy；保留一次可靠的 query 快照和返回值所有权边界。
- 对部署批量结果，把重复 GPU finite 同步与 CPU finite 扫描合并到 Real 接收完整物理 chunk 的一次准入；Policy 保留类型/shape，设备层保留最终目标安全检查。先搜索其他 `LoadedPolicy` 消费者，若移除检查会失去其唯一边界，不能强删。
- 点云四个长算法描述常量若仍只有日志/示例消费者，可删长描述及重导出，保留实际数值 recipe、源版本和必要 provenance；这是小优化，不应拖延主线。

不整体删除 `dataset/contracts.py`：`ProcessingConfig` 与 canonical shape/dtype 有真实消费者。不新增重复 semantic dictionaries、兼容矩阵、capability registry、多个模式专用 runner、pending observation queue 或 result 排序器。一个 worker 下 query_id 只用于追踪；run_id 用于 episode 失效判定，具体 query 状态用于 deadline 失效判定。

## 6. 统一执行结构与同步语义

建议最多新增两个小模块：Real `deployment/inference.py` 管模型任务；Policy `agents/action_decoders/rtc.py` 放数学 helper。文件名不是强制抽象层；优先改现有实现，不为封装再加 manager/factory。

| 所有者 | 独占责任 |
| --- | --- |
| 主线程 / PolicyRunner | 设备 SDK、operator、观察历史、时间槽、计划、动作实现、记录与终止 |
| 一个模型 worker | 模型 load/warmup/predict/reset_episode/close，Torch 与模型 RNG |

可用 `ThreadPoolExecutor(max_workers=1)`。同一时刻仅一个未回收 Future，包含生命周期操作；没有排队的下一份 observation。worker 空闲且调度允许时，从最新有效历史创建 query。主线程只在 `done()` 后取结果；不能用阻塞 `result()` 占住正在运行的 owner。

模式定义：

| 模式 | 何时查询下一段 | 采样算法 |
| --- | --- | --- |
| sync（默认） | 当前 A 个动作执行完后；等待结果期间不下发策略动作 | 原采样器 |
| async | 计划交接前 d 个时间槽查询，同时继续执行冻结的旧前缀 | 原采样器 |
| rtc | 与 async 完全相同的预取、交接、过期和等待协议 | 前缀条件化 DDIM 采样 |

允许在 episode 间切换；运行中切换明确拒绝。切换/新 episode 必须失效旧计划、回收旧 Future、完成串行 reset，之后才准入新的运行。保留旧模型权重与 normalizer，不把模式切换变成重新训练或重载架构。

三种模式采用同一真实控制时间网格：按保存的 dt 获取实际控制行，保留真实时间与采样抖动；等待时继续服务采样和停止。漏槽不回填合成历史、不突发补发动作。需要连续 N 个有效控制采样槽；缺测/大间隔打断历史，低帧率相机重复源帧在当前 freshness 条件内可以正常使用。可容忍的调度抖动须明确并测量，不要求操作系统精确零误差。

这相对旧实现阻塞推理期间没有控制行的行为，是**明确的评测观察时间协议变化**，不能宣称完全等价重构。新 sync 是后续 async/rtc 的统一基线；N>1 尤其要做同权重旧/新 sync 对照。精确历史实验复现使用其源版本，不为此留下第二套永久 runner。不要把本次运行时协议调整偷偷扩展成训练数据重采样。

## 7. 小而完整的跨仓库推理接口

建议扩展现有接口：

```python
predict(observation, *, rtc_prefix=None, delay_steps=0) -> future_actions
```

- H 为模型完整 horizon，N 为观察步数，P=`H-N+1` 为从当前观察锚点起的完整可用未来；A=`n_action_steps` 是执行/重规划段长度。
- 返回物理单位、控制子空间的 `(P, C)` 数组，C 为 joint 19 或 EEF 21；模型输出从索引 `N-1` 开始。不能只拿 A 段，也不能丢掉完整 future tail。已有核心 agent 已生成完整 H，不额外跑一次网络来取 tail。
- PolicyInfo 最小增加 H（P 可推导）；保留现有训练/仿真调用者的动作输出约定，不为 Real 接口改坏其他 runner。
- `rtc_prefix` 是以 query 锚点为起点、实际可用旧未来的物理控制数组 `(L,C)`；前 d 个是冻结承诺，后段是软先验。无前缀时 d=0；有前缀需 `0 <= d <= L <= P`。
- Real 不接触 Torch 或 normalizer；Policy 按 checkpoint 的控制子空间统计恰好归一化一次。
- 对不支持的 decoder/agent 路径在连接硬件前明确报 Unsupported；不能忽略 kwargs 或悄悄退回 async。sync/async 继续支持原先可用的模型。

joint+aux EE 模型可能是 28 维模型输出、19 维控制；EEF 为21维控制。不能用28维统计直接广播到19维前缀，也不能给未执行的辅助输出填零并当成目标。

## 8. 首版采用固定交接预约，避免动态拼接歧义

这是为可验证性选择的调度协议，不声称 RTC 数学上只能固定延迟。

`prefetch_steps=d` 在 episode 内固定。首版优先使用显式正整数配置；warmup 报告建议值与可行性，不自动改 A/dt/模式。时延估计包含前缀准备、预处理、推理、CPU 返回和余量，RTC 必须测带 VJP 的路径；测得最大值或高分位数都不是硬实时上界。

稳态单 worker 必须满足：

```text
1 <= d <= A
A + d <= P
```

这只是本调度协议的可行域。sync 仅要求 `1 <= A <= P`；不能将 `P >= 2d` 宣称为所有异步算法的普适限制。

设 query 在控制槽 q 提交，则其结果预定在 h=`q+d` 接管。旧计划冻结并执行 q…h-1 的 d 个动作；新计划接管从结果索引 d 开始。结果提前返回也等到 h，不改变承诺；迟到结果没有准入资格，不临时修改已预约 d 来追赶。

**预取按交接时间计算，不按完整 P 长 buffer 的剩余长度计算。** 首次计划的实际起点为 b，则第一次预取在 `b+A-d`，第一次交接在 `b+A`，之后每 A 个槽重复。稳态新 query 间隔 A；下一轮仍必须能从当前计划取出 d 个未发送控制槽。

同一控制槽 k 的次序必须一致：

1. 获取实际观察并处理授权/期限。
2. 回收已完成结果；到交接槽时验证并接管。
3. 若到预取槽，从**尚未发送的 a(k)** 开始冻结 d 步，提交锚点为 k 的 query。
4. 通过临发检查后发送 a(k)，记录 dispatch。

不得先 pop/send 再把剩余前缀仍标记为 k。A=d 时，前一 Future 完成后可在 bootstrap 第一次下发槽立即预取；仍只能一个在途 Future。

提交、handoff、实际 dispatch 时间必须记录。以主机 monotonic 时间网格计时，不能用“已经成功发送多少条”代替墙钟进度。发生不可容忍漏槽时失效该预约并进入 WAIT；不追赶式连发，不把过期动作改时间后复用。交接判定包含 owner 可实际采用结果的时间；仅 worker 早完成、owner 错过槽位，也不能事后声称按时交接。

### Bootstrap 与 WAIT

- 无 active plan 时没有可冻结的 d 步：从最新有效观察运行无前缀原采样器。
- 结果通过授权、年龄和输出检查后，在下一个正常槽 b 首次下发，并把 `future[0]` 显式锚定到 b；保留原 query 观察时间，记录这次重锚定。不能按首次推理耗时裁掉机器人从未执行的头部动作。
- 每次需要动作却没有可执行计划时，开启一次 WAIT deadline；覆盖等待历史窗口、旧 Future 回收、bootstrap 推理和等待首次下发。重新提交/丢弃/回收不能刷新 deadline；第一次新动作成功 dispatch 后才清除。
- handoff miss 后旧结果失效、旧计划不可偷偷继续执行尾部。保留忙碌 Future 直到完成并取回其成功/异常；期间不能追加 bootstrap 到 executor 队列。空闲后允许一次新的无前缀 bootstrap，明确记录 underrun；其不改变配置模式。反复无效只在同一有限 WAIT 预算内继续。
- WAIT 到期或不可恢复故障按既有终止路径撤权和 stop。WAIT 仅表示不发送新策略目标，设备仍可能追踪上个目标；不能称为硬件 hold。若调用 stop/hold，必须作为显式设备事件并失效原计划，不能无记录地恢复续接。

保留现有 `execute=False` 路径：在这一明确标记的非下发模式，使用通过准入的逻辑 consume 推进计划并清除 WAIT，替代“实际成功 dispatch”条件；真实设备状态仍记 `NOT_CALLED`/未执行，不能伪造 ACCEPTED。其前缀表示模拟的执行承诺，元数据与真实执行明确区分，不能用该结果证明物理收益。`execute=False` 不自动意味着不连接传感器；离线测试必须使用 fake robot/观测，不能直接启动 live session。

停止顺序：撤权并失效计划/结果 → 设备 stop → recorder/worker/资源回收。`Future.cancel()` 不能中断正在执行的 CUDA；`shutdown(wait=False)` 也不保证 Python 进程有界退出。新 episode 的 reset 不与旧 predict 并发，主线程不能并发 close 或移动模型。首版保证停止路径不主动等待模型；需要强制、有界结束模型时另立进程隔离任务，不使用 daemon 线程伪造完成。

## 9. RTC 条件必须对应实际控制动作

### Joint

主线程冻结 d 个已完成必要投影/实现的 joint19 目标，后续发送这些同一目标。RTC 条件使用其物理动作值，而不是已被投影改掉的网络原始值。

### EEF

当前 Real `dataset/processing.py` 的 EEF 动作标签由实际尝试的 arm joint target 做 FK 后与 hand 拼接得到；因此原始 Cartesian intent 不必等于实际控制标签。

1. 主线程在 query 前预实现短 hard prefix，得到具体 joint19 命令；用局部 previous-candidate 顺序构造，不提前推进实际“上一条已接受命令”的状态。准备是事务性的：全部 d 步成功、授权仍有效且未错过提交槽，才发布冻结前缀并提交 query；否则丢弃全部未提交候选。部分 IK 成功不能修改一半执行计划，准备中的候选也不能记录为真实 attempted dispatch。
2. 用与公共导出相同的 FK、base frame、rot6d、hand 顺序转换成 EEF21，作为前 d 步条件；冻结 joint 命令供真实执行。
3. 实际下发前，使用最新反馈复用现有轻量动态准入（表示分支/band、硬件距离、previous accepted jump 等），不能只剩 Robot.send 的硬限位。动态检查失败则失效前缀和候选结果；不能静默重新 IK、改目标后仍沿用旧 RTC 条件。
4. d 步之后尚未实现的旧 EEF tail 只能作为软先验，不是假装已经冻结的命令。
5. 只准备短 hard prefix，不在 owner 一次求解完整 P 步。测量前缀准备对 owner tick 的影响；若无法在预算内完成，采用最小的分槽准备或明确报告该配置不可用，不能让“异步模型”掩盖主线程 IK 阻塞。

先 joint 验收，再完成 EEF；仅 joint 通过不能汇报完整 RTC 支持。任一已冻结命令发生实质修改、部分设备结果不确认或失败，冻结假设即失效。首版 async/rtc 的承诺有效性要求实际设备结果被确认接受；CRC_UNCONFIRMED/UNKNOWN 不能假装承诺已成立。保留原 dispatch 证据，使用明确失效/终止路径，不盲目重发。

## 10. DDIM-adapted RTC：最小数学实现

现有 decoder 为 `agents/action_decoders/diffusion.py`；依赖锁定 `diffusers==0.27.2`。默认 DDIM、eta=0、`clip_sample=True`；支持 sample/epsilon/v_prediction。本任务实现的是**保留既有 DDIM 基础步的推理式 RTC 适配**，不能称已经复现 flow 原论文的全部实验性质。

### 10.1 原路径必须可旁路

无 prefix、权重全零或 guidance 最大值为0时，直接走原采样函数：相同 seed、随机数消耗、scheduler 和输出。不能用“数学上差不多”的新函数替换默认 sync/async 路径。encoder 条件只计算一次并 detach。RTC 只对每步 noisy action 求梯度，不对模型参数积累梯度。

### 10.2 有效空间与权重

输入 old future 长 L，前 d 为冻结前缀；模型索引 `N-1+i` 对应 future i。只在这些位置和控制维构造 endpoint residual，其余历史槽位与辅助输出 residual 为0。

首版只实现一个 EXP 软权重，无需四种 schedule 的配置体系：

```text
i < d:       w_i = 1
d <= i < L:  z_i = (L-i)/(L-d+1)
             w_i = z_i * expm1(z_i)/(e-1)
i >= L:      w_i = 0
```

d=L 时没有软区。W 是上述时间权重与控制维 mask 的广播乘积。hard prefix 指调度器对实际动作的冻结，不保证有限 guidance 使网络预测前 d 项严格相等；这些项不会再次下发。

### 10.3 精简后的每步公式

令 x 为当前 noisy action，m 为模型原始输出；`a=sqrt(alpha_bar_t)`、`b=sqrt(1-alpha_bar_t)`。未裁剪 clean estimate f：

```text
sample:        f = m
epsilon:       f = (x - b*m)/a
v_prediction:  f = a*x - b*m
```

`v_prediction` 不是 flow velocity。alpha 来自当前 scheduler，不能把整数 t/训练步数当 flow time。

令 a'、b' 对应当前 scheduler.step 实际采用的 previous timestep。对锁定 0.27.2，`prev_timestep = t - num_train_timesteps // num_inference_steps`；末步遵守 `final_alpha_cumprod`。不能擅自用 timesteps 数组下一项替代其语义。

```text
residual = detach(W * (Y_normalized - f))
g = VJP(f, x, residual)
base = original_scheduler.step(m, t, x).prev_sample
kappa = min(beta, 1/(a*b))
x_next = base + (a'*b - a*b') * kappa * g
```

只在 W 非零的实际控制位置形成 Y residual，未约束部分直接置0，避免依赖无意义的占位值。beta 是有限非负 guidance cap，作为一个显式 RTC 实验参数保存；0 走原路径。所有实际求值的 denoising t 必须满足 a>0、b>0，否则明确 Unsupported，不通过加 epsilon 偷改公式。末步目标 a'=1、b'=0 是允许的终点，不在该终点再次调用 denoiser/VJP。

上述标量系数来自 `c=a+b, tau=a/c` 的 flow 坐标变换：`c'c*(tau'-tau)=a'b-ab'`，原 RTC 权重化简为 `min(beta,1/(ab))`。实现不必真的创建 y/tau/flow velocity，减少代码和坐标转换错误。

**裁剪行为定案：** 每步 base 完整保留原 DDIM 的 clip 及 epsilon 计算。特别是 0.27.2 默认在 clip clean estimate 后不重算 epsilon；不要改 `use_clipped_model_output`，不要用裁剪后 f 计算 VJP。附加 guidance 可能使末步超出原 normalized 范围，因此仅在最终一步且 `clip_sample=True` 时，将 guided sample 投影到同一个 `clip_sample_range`；clip_sample=False 不做此投影。该 final projection 是本适配明确的算法选择，不是硬件安全保证。默认/零引导原路径仍直接旁路。不得额外加入动作 EMA、平滑器或每步随意 clamp。

### 10.4 梯度边界

- 必须在 denoiser 前执行 `x = x.detach().requires_grad_(True)`，再 forward 和 VJP。先算 m 再设 requires_grad 会漏掉 denoiser Jacobian，不能照抄参考实现中的该顺序。
- 处理外层 `torch.inference_mode()` 及 `@no_grad`：局部关闭 inference mode、启用 grad；必要的张量在该范围内 clone 成普通 tensor。只加 enable_grad 而仍处于 inference mode 不够。
- VJP 输入是完整 noisy tensor；辅助维不提供 endpoint 目标，但网络跨维耦合可以使其 latent gradient 非零，不能再将 `g[..., auxiliary]` 清零。
- grad_outputs detach、每步新 sample detach、`create_graph=False`，不累积参数 `.grad`，不留跨步计算图；断开输入依赖的 constant mock 合法得到零 VJP。
- W 只乘一次，不能把 `||W*error||²` 导致 W 被平方当成同一算法。
- 初始随机噪声、base scheduler 配置、checkpoint normalizer、NFE 保持不变。首版不支持 eta>0、dynamic thresholding 或未适配 scheduler；明确报错，不偷偷改训练/采样配置。保留现有 Gaussian action normalization 与 clip_sample=True 不兼容的拒绝规则；只对已验证的 affine action normalizer 和支持 input-VJP 的 backbone 开放 RTC。

## 11. 追踪与可解释对照

复用现有 run_config/recording 结构，用少量实际字段记录：两仓库 SHA、checkpoint/weights/seed/NFE、模式、dt/N/H/A/P/d、beta、预算、query/run id、输入源时间及帧号、query 提交/完成/接管、动作时间槽与实际 dispatch、underrun/拒绝/终止详情。若逐行异步溯源所需映射尚无容纳处，只增加最小字段或紧凑事件记录，不记录完整 noisy trajectory/debug tensors，不建立通用事件系统。

`recording/recorder.py` 当前硬编码的 `synchronous_direct_sdk_v1` 要随真实执行模式更新，不能让 RTC rollout 仍标成旧同步协议。保留实际 attempted target、measured state、逐设备结果及失败前缀。

先用同权重、相同观察协议、A/NFE、seed、预算比较新 sync/async/rtc；async 与 rtc 用同一固定 d，必要时取可共同满足的预算，不能各自隐藏不同控制条件。记录成功率、任务时间、decision age、handoff miss、owner tick 时延、prefix 准备开销及 chunk 接缝变化。少量合成测试不能证明实际任务质量提升，未运行的对照写 NOT VERIFIED。

## 12. 分阶段实现与验收

每阶段保持可独立审查的 diff/提交边界，按顺序推进；无需为了机械分阶段增加长期兼容路径。

### P0：修复与精简

完成第 3–5 节。使用小型合成 episode/真实函数路径验证：

- OPERATOR/QUIT 在 normal、after-predict、pacing、stop/recording 异常路径仍为原主因；每 episode 只保存/计数一次。
- N=2/H=4 时后两槽未使用 observation 的 NaN 不误拒绝；H 内监督 NaN 即使超过 A 仍拒绝。
- 未启用模态的 NaN 不拒绝；启用后只拒绝受影响窗口；aux EE 只按真实监督维检查。
- 两设备 accepted-only、缺失 dispatch、CRC/UNKNOWN，以及仿真无硬件 row_info 均有确定行为。
- 首尾 padding 原始索引一致、无跨 episode/内部缺口压缩；统计源行恰为有效训练窗口分角色去重集合。

### P1：统一 worker 与正确默认同步

先让 sync 稳定，再开放 async；核心 agent/decoder 原采样关闭 RTC 时数值等价。用 fake clock、受控 Future、fake robot 走生产调度/终止代码，不能只测试另写的调度模型。

- 推理忙碌时 owner 能处理撤权/stop；旧结果、迟到异常、新 episode reset/close 不交叉。
- 包括 lifecycle 在内最多一个 Future，无 queued observation；所有 Future 最终取回成功/异常。
- sync 不提前预取、不在等待期间下发策略动作；两种异步模式使用同一调度。
- 逐模态 query 年龄不能被新反馈或结果完成刷新；无关录制相机不误入模型年龄条件。
- N=2/H=16 时 P=15，future 起点 N-1，无切片错位。

### P2：调度、Joint RTC，再完成 EEF

- 覆盖 A=d、A+d=P、非法参数；bootstrap 首槽与第一次预取次序；提前/按时/迟到结果；owner 漏槽。
- WAIT deadline 不因历史凑不齐、旧 Future、重试或丢弃刷新；无过期动作补发，无错误头部裁剪。
- fake `execute=False` 能按逻辑 consume 推进 async/rtc 并清除 WAIT，实际 dispatch 仍为 NOT_CALLED，不伪造 ACCEPTED，也不连接真设备。
- 测试真实 diffusers 0.27.2 三种 prediction_type，RTC-off 与原路径在确定性条件下逐元素相同。
- 对未裁剪情形验证 VP/flow 系数等价；VJP 与非平凡 toy denoiser 的有限差分一致，覆盖非单位 Jacobian与跨辅助维耦合。
- 对 clip_sample=True 使用明确越界例检验 base 保持旧 step，不重算 epsilon；验证高 guidance 最终投影、clip=False 无投影和零 guidance 旁路。
- 验证 history offset、控制子空间 normalizer、W 边界、d=L、W=0、batch/device/dtype、无参数梯度、无逐步图增长。
- joint prefix 等于真实冻结目标；EEF prefix 等于冻结 joint target 的公共 FK 表示；新反馈动态拒绝、partial dispatch/UNKNOWN 会失效候选；准备动作不提前污染 previous accepted 状态。

数值测试没有通过不能宣布 RTC ready；不能留 TODO/no-op 分支把参数接通当作完成。GPU/权重可用时补一次小规模端到端时间与内存检查；缺少时明确标注，保留可运行验证入口，不安装一套新环境掩盖限制。

### P3：交付

- 按当前仓库规则运行必要的 compile/lint/diff check 和上述风险相关定向测试，不以 compile 代替行为验收。
- 更新 README 的稳定使用入口和配置示例，说明默认 sync、如何选择 async/rtc、所需实验预算、支持范围与模式切换条件；内部数学/调度细节放自然的技术文档，不塞满 quick start。
- 汇报每项任务的完成/未验证/阻断状态、实际改动文件、测试命令与结果、可用运行命令及仍需真人授权的真机验证。
- 不宣称“最佳效果”“物理安全已验证”或“历史模型已修复”。不自动推送实现提交、合并 PR 或启动真机；本任务书的发布授权不等于执行上述动作。

## 13. 参考来源及使用方式

参考机制，不整体移植框架，也不为完成此任务重新开展无界文献调研：

- [LeRobot，审查版本](https://github.com/huggingface/lerobot/tree/8c920c4270460851cedd2737657584586d3dc66f)：RTC prefix weighting/推理条件化与异步动作管理；其 `modeling_rtc.py` 的 requires_grad 顺序须按本文修正，不能盲拷。
- [LeFranX，审查版本](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7)；[ManiUniCon，审查版本](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686)：借鉴控制所有权与多频率执行边界，不假定本项目具备参考系统的独立 watchdog。
- [Real-Time Chunking，2506.07339v2](https://arxiv.org/abs/2506.07339v2)：已承诺前缀与软重叠的推理条件化。
- [Training-Time RTC，2512.05964v2](https://arxiv.org/abs/2512.05964v2)：以后可用带 prefix 的训练替代每步 VJP，但需要新 checkpoint；本任务不同时引入。
- [UMI，2402.10329](https://arxiv.org/abs/2402.10329)：时间对齐和延迟处理；不能由该思路推导出应裁掉 bootstrap 中未执行的动作。
- [diffusers DDIMScheduler v0.27.2](https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/schedulers/scheduling_ddim.py)：以真实 step、prediction_type、clipping、previous timestep 实现为准。

RTI-DP、VLASH、REMAC、A2C2、Streaming Policy 等保留为后续替换方向，不与本次数据修复、调度改造和推理式 RTC 叠加实施。

## 14. 最终设计审查结论

本方案以默认同步正确性为先，保留真实数据/设备边界，删除无消费者或无信息增益的机制；用一个 owner/worker 路径支持三种模式。最终审查已收敛以下容易出错的点：固定交接而非按完整队列长度预取；bootstrap 显式重锚定；不刷新 WAIT deadline；失效但不遗失 Future；真实源时间的能力边界；EEF 已实现前缀；控制维 residual 与完整 latent VJP；保留原 DDIM clip 语义及明确的最终投影。

此前离线审查和小型数值推导支持进入实施，不能替代上述生产代码测试和实际任务评测。实现代理应继续完成可运行、可审查的代码与离线验证，不止提交一份计划，也不越过未授权的真机边界。
