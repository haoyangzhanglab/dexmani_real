# Codex 任务书：收敛真机部署的数据、配置与执行语义

日期：2026-10-07（Asia/Singapore）  
主仓库：`haoyangzhanglab/dexmani_real`  
协同仓库：`haoyangzhanglab/dexmani_policy`  
审校修订：**r2（最终设计审查通过）**  
实施状态：**待实施。设计通过不表示代码修复、仓库测试或真机验收通过。**

## 0. 执行目标、权限与事实基线

当用户要求执行本任务书时，先读两仓库及修改目录适用的 `AGENTS.md`，再按本文完成必要修改、定向离线回归和可审查 diff。不要只重新提出计划，不为已裁决的普通实现细节反复请求确认。局部阻断只阻断对应任务，先完成其余独立项。

目标：解决 F01–F20 中真实影响科学正确性、部署可靠性和使用体验的问题；明确条件性风险与支持范围；删除有直接收益的局部冗余。**不重写调度器，不另造契约/配置/验证平台，不以减少检查数量代替正确性。**

### 0.1 基线与工作方式

| 仓库 | 源码审查基线 |
| --- | --- |
| Real | `8f0070e3b73c3b0da81aa166143be44b7200f67b` |
| Policy | `52d1d2a77b76a254b1a2498b059d160e6e527590` |

r1 文档提交为 `62c08deeb7819f6f8f61ac2922387b64417bb1b8`；本文 r2 取代其实施细节。SHA 只定位证据，不是 reset/checkout 目标。先记录两个工作树的 HEAD、状态及相关差异，保留用户已有修改，不自动 stash、reset、clean、rebase 或覆盖工作树。

从当前 producer → transform → consumer → side effect 确认变更位置，不重复全仓审查。已修复项用当前代码和定向证据标 `ALREADY_FIXED`；如基线已变化，用更短的等价方案实现同一行为，不恢复旧代码重放任务。本文规定行为与验收边界，不要求机械新增指定名称的 helper 或类。

Dataset、normalizer、模型桥接属于 Policy；设备、时序、Raw、会话与 CLI 属于 Real。优先使用相邻现有工作树，不在 Real 复制 Policy Dataset，不修改 site-packages。协同仓库缺失时，仅在允许的工作区建立独立 checkout；访问或写权限不足则报告对应阻断，不用本地假实现掩盖跨仓库未完成。

### 0.2 权限与验证边界

执行本文允许代码、文档和无硬件离线验证；默认交付 diff，**不自动 commit/push 实现**。创建或审校本文的文档提交不等于已授权运行实施任务或连接设备。

禁止连接或驱动 xArm、XHand、RealSense、VR、HTS；禁止 HOME、rollout、replay、实时采集和真实标定写入。`execute=False` 仍可能连接硬件，不是离线测试。测试是否有副作用不明时先读入口，不试运行来判断。

不升级实验环境、不下载大型权重、不运行完整训练/DDP/长评测来凑验收。已发布 Raw、旧 checkpoint、保存的实验配置和历史报告不得原地改写。临时合成数据使用临时目录。

r2 审校只核对了任务书、相关源码和纯整数网格下界；未执行 G1–G8、仓库测试、实际 checkpoint、GPU 或设备测试。聊天中的旧 PASS 不继承为本次实施结果。

### 0.3 与历史任务的关系

本文是 F01–F20 的当前实施依据，不重放已有 U 系列整改。旧根目录任务书保留，不批量删除或覆盖其完成记录。重叠处仅作以下更新：

- 新 Real 训练增加明确的时间连续性筛选；旧实验不被新增默认值悄悄改变。
- 实际模型输入与 Dataset 声明对齐；Raw 全模态和公共导出不收缩。
- Policy 返回结构正确的预测；Real 判定可执行性并保存坏预测证据；独立 warmup 自查输出。
- 技术失败贯通 session/CLI；正常计划时长、操作者结束不自动算技术失败。
- 保留标称碰撞几何与现场位姿补偿的分工，不引入历史标定一致性门槛。
- 公开入口及副作用边界验证保留；不要求全仓只出现一次 `validate`。

不要修改 `AGENTS.md` 绕开约束，也不要把一次任务扩写成永久架构规范。

## 1. 总体方案与不变量

沿用现有链路：

```text
保存的 Policy config + checkpoint
  → 解析部署事实 / 严格恢复模型和 normalizer
  → 核对实际输入 / 选定路径 warmup
  → Real 装配当前硬件、标定及生效配置
  → 操作者开始准备 / 重新采样 / 授予 RUNNING
  → 一个设备 owner + 一个串行模型 worker + 最多一个未回收 Future
  → 完整物理 future / 调度准入 / 动作实现 / 逐设备 SDK 下发
  → 可选录制产物；所有路径保持明确的结束与失败结果
```

不新增第二套 sync 执行器、远程服务、模型请求队列、配置注册器、能力协商或 generic processor pipeline。现有 dataclass、普通函数与 owner 状态足够。

1. `P=H-N+1`，future 从模型索引 `N-1` 开始，完整返回 `(P,C)`，joint19 或 EEF21。Agent 的 `pred_action/control_action/tail` 不重写，辅助输出不作为控制量；本轮保留 float64 NumPy 输出。
2. Bootstrap 保留未执行动作头并在正常网格重锚定；async/RTC 在 `b+A-d` 预取、`b+A` 交接，从新结果索引 `d` 执行。保持 `1≤d≤A`、`A+d≤P`；不追赶连发，不沿用失效 tail，不在忙碌 worker 后排旧观测。
3. Policy 归一化/反归一化一次，Real 消费物理量。RTC prefix 对应同一份冻结实际命令；失效后不重新 IK 再沿用旧条件。beta=0 走普通采样，不强行运行 VJP，不扩大其他 decoder 的 RTC 支持范围。
4. `intent`、准备目标、逐设备 dispatch、实测 state 分别留证。ACCEPTED 不等于到位，CRC_UNCONFIRMED 不等于接受，也不等于已证明硬件损坏。
5. 全部存在的 arm/hand 目标先检查，再进入第一个 SDK；每个 SDK 前仍查 epoch、授权、deadline。锁不跨阻塞 SDK/文件 I/O。先撤权和尽力 stop，再归档；迟返回保留真实状态，旧 Future 不污染新 attempt。
6. Raw 发布后不可变；公共导出保留现有全模态，辅助缺测保留 NaN。禁止坏行压紧重拼、补零、补图、伪造时间或动作插值。采集和导出 QA 不隐式代替训练资格规则。
7. 现场标定、安装关系属于当前 Real，数值预处理配方属于保存模型。不把历史物理标定、训练路径或实验目录名当真机 ABI。
8. A 可在部署副本中覆盖，不要求等于训练 A，不修改训练 padding。CLI 的 None 表示未覆盖，显式 0/False 保留。
9. 不擅改动作表示、控制频率、IK/限位/碰撞阈值、滤波、fallback、重试或自动恢复。启动检查改为匹配现有运行时边界，不改变实际 deadline 或下发资格。线程和 warmup 都不证明硬实时或物理停稳。

### 1.1 参考项目

只借鉴职责划分，不添加运行依赖：

- [LeRobot 模型配置](https://github.com/huggingface/lerobot/blob/200ee53596d464bd28f6595cdb31be69a2b5e379/src/lerobot/configs/policies.py)：模型输入输出属于模型配置。
- [LeRobot 推理装配](https://github.com/huggingface/lerobot/blob/200ee53596d464bd28f6595cdb31be69a2b5e379/src/lerobot/rollout/inference/factory.py)：显式选择模式及其消费参数。
- [ManiUniCon 装配入口](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)：data、robot、policy、sensors 分别装配。

不移植其注册器、队列、插值、重试次数、设备自动回退或安全阈值；DexMani DDIM RTC 与其他 guided/trained RTC 不自动等价。参考链接用于按需确认，不要求重新研究整个参考项目。

## 2. 完整覆盖与实施顺序

F 编号是审查事项，不代表 20 个已复现 bug。

| ID | 事项 | 本次裁决 | 工作包 |
| --- | --- | --- | --- |
| F01 | 时间断档跨入固定 dt 训练窗口 | P1 修复新训练配方，保护旧恢复 | T01 |
| F02 | Dataset 模态被当作模型实际输入 | P1 一次性核对实际消费字段 | T02 |
| F03 | 启动预算遗漏历史/网格等待 | P1 对齐确定下界、样本测量和运行时边界 | T04 |
| F04 | 未确认下发未贯通顶层结果 | P1 停止归档后传播技术失败 | T03 |
| F05 | 非有限预测在桥接层丢失 | P2 统一返回与执行准入 | T04 |
| F06 | 保存配方补当前默认值 | P2 保存快照完整、用户 YAML 可部分覆盖 | T02 |
| F07 | 预热混合冷启动/稳态/prefix 成本 | P2 随输出和预算一并修复 | T04 |
| F08 | MultiTask 真机未接通 | 明确不支持，不实现新语言/多任务能力 | T02 |
| F09 | 模式恢复进入首槽预算 | 条件项：证据不足则保留 | T06 |
| F10 | 单槽 d 次 EEF IK 峰值 | 条件项：查现有证据，不直接并行化 | T06 |
| F11 | CLI 强制录制与相机依赖 | P2 暴露现有无录制路径 | T05 |
| F12 | 声明、生效配置与参数用途混淆 | P2 生效快照与用途说明，不迁移整棵 YAML | T05 |
| F13 | 先使用 dt，后验证 dt | P2 调整调用顺序 | T04 |
| F14 | online/HOME 保护范围不同 | 说明范围，不隐式扩张控制约束 | T06 |
| F15 | 多入口重复静态检查 | P3 只去同链重复，保留公开边界 | T07 |
| F16 | RTC 重复初始化和类型检查 | P3 局部合并，不引入缓存协议 | T04/T07 |
| F17 | best 依赖选点审计 | 保留，显式 checkpoint 已有轻量路径 | T07 |
| F18 | 显式参数仍依赖 eval 小节 | P3 默认值按需读取 | T02 |
| F19 | 实验发现只认 latest.pt | P3 与已有可加载类型对齐 | T02 |
| F20 | future 强制 float64 | 保留，不夹带精度/归档 ABI 变化 | T04/T07 |

按 T01 → T02 → T03 → T04 → T05 → T06 → T07 推进，独立项可调整顺序。T04 的跨仓库输出/预热接口必须配对修改，T03/T04 复用同一预算来源判断。每包做小而完整的 diff，不要求每项独立分支或提交。

## 3. T01：时间连续性只在训练样本消费端处理

**F01。位置：** Policy `datasets/{replay_buffer,base_dataset,sampler}.py`、`training/build_utils.py` 及保存配方消费入口。Real QA/文档只作必要说明，不重构 exporter。

### 3.1 新配方裁决

新 Real canonical 训练使用 `dataset.max_time_gap_ratio`，默认 `1.5`；相邻真实源行须满足 `0 < Δt ≤ ratio × saved_dt`。**1.5 是本任务明确选择的新研究资格参数，不是硬件安全值，不是 QA 同名比值已经证明最优，也不检测所有累计相位漂移。** 参数应为有限正数或 null，拒绝 bool，保存实际值。

`null` 表示显式不做时间筛选，供旧规则恢复或研究对照使用；新实验须记录 `unfiltered`，不得因时间证据损坏自动降级。新 Real 默认启用，仿真样本集合与配方不受影响。调整默认或 ratio 属于新实验，不是给既有实验补一项无害配置。

### 3.2 最小实现

1. 启用筛选时按需读取 `row_info/observation_timestamp_ns`，只检查小型整数数组描述和长度，不读无关视觉数据，不加入模型 observation。缺整个字段准确报错；重导出不能保证恢复 Raw 原本未知的时间。
2. 窗口覆盖的每个真实源行时间都须已知且为正，包括 H=1 或只有一个不同源行的窗口；再对不同、相邻源行比较间隔。padding 重复同一源行不当成零间隔，但不能让未知时间因重复而变有效。倒退/非递增间隔拒绝，整数运算避免无符号减法下溢。
3. 保留 N 步观测、H 步动作和 dispatch 资格；时间规则覆盖该 H 步窗口的真实源行，窗口不跨 episode。利用已有有界批次/source_rows 或坏边前缀和，选更短实现，不构造全数据量 windows×horizon 大矩阵，不压紧坏行重拼。
4. 所有资格 mask 合并后，再产生唯一训练 observation/action 源行统计。validation 用相同资格和训练 normalizer；被剔除窗口不参与拟合。沿用零有效窗口失败，只加必要的时间拒绝计数。
5. 新实验把字段、ratio、筛选状态和计数放已有 `data_recipe`；不新建 manifest/schema/报告体系。现有 QA 仍只报告。

### 3.3 旧恢复与共享调用者

**在实例化 Dataset/sampler 之前**由恢复入口解析保存的时间规则，再显式透传。分清新实验的构造默认值、已保存的新规则、已知旧无筛选规则三者；不能用 `.get(key, 1.5)` 解释所有缺键。

已知旧 recipe 无时间筛选时恢复为旧行为，样本集合、唯一统计源行及 checkpoint normalizer 不变。旧 full-resume 的 `data_recipe` 表示也保持原样，不能因注入新默认 ratio、时间计数或 `unfiltered` 键制造 strict-resume mismatch。新实验显式 null 与旧快照缺键不是同一种序列化事实；通过恢复入口的小分支处理，不扩展为通用 migration，不放宽其他 strict-resume 检查。证据不足、无法判断原规则的历史产物明确拒绝 full resume；只读推理不受影响。

检查 Policy-aligned VQ 训练、usage/验证与 split 恢复等实际 Dataset 调用者，透传同一保存规则。新筛选改变拟合统计时，旧码本不能靠放宽 hand affine 一致性混用；应明确需要新的对齐实验。独立 VQ 接口不强制改造，不重算旧 normalizer，不原地补写旧产物。

**G1 验收：** 小合成数据覆盖连续/长间隔/倒退/未知时间、H=1、合法 padding 和 episode 边界；只拒相关窗口且唯一统计正确。检查旧 recipe 恢复样本和 strict-resume 表示不变、新规则恢复一致、仿真不变。真实数据影响比例无数据则 `NOT_VERIFIED`，不全盘扫描。

## 4. T02：实际输入、保存配方与加载边界

**F02/F06/F08/F18/F19。位置：** Policy `deployment/runtime.py`、实际模型构建处和已有输入声明；Real `deployment/config.py`、`config/pointcloud.py`。

### 4.1 模型侧一次性核对

复用实际 Agent/encoder 的 `consumed_observation_fields` 等已有声明；不在 Real 维护 model-name→modalities 表，不 tracing 或反射扫描 forward 猜输入。

新训练在实际模型构建时核对一次：当前单任务 Dataset 缺必需输入或多声明未消费模态，都准确报告两组字段并失败，由实验者修正配方。不能静默裁掉字段、扩大真机门槛或自动改历史 checkpoint。Raw 全模态、动作辅助监督和训练元数据不属于这个输入相等关系。

部署在加载完成、设备连接前复用同一模型侧核对。配置阶段只解析声明事实，模型阶段才确认实际消费，不在加载前声称已经验证模型、不为此加载两次模型。没有接入输入声明的自定义模型在真机入口准确拒绝；已有合法单任务实现不应被误拒。本轮真机限制不能扩散成禁止多任务训练/仿真；MultiTask 的文本在 Agent 外层消费，不能只看 encoder 列表删除 `task_text`。

### 4.2 MultiTask 仅说明不支持

在连接前明确拒绝当前尚未接通的 MultiTask/text-conditioned 真机路径，优先给出缺少任务选择/child 数值配方恢复的原因，不落到空模态、dt TypeError 或 task_text KeyError。按已知实际配置/实现判断，不靠名字包含某字符串猜支持范围。

不新增默认 `--task`、隐藏文本、child 配方合并或 per-task normalizer。此项标 `SCOPE_DOCUMENTED`，不能称为多任务部署已经实现。

### 4.3 保存配方不是用户部分 YAML

用户 Real YAML 继续通过现有 `_patch()` 覆盖当前默认值。保存点云配方则在恢复边界一次检查本实现需要的完整数值键，复用 dataclass 字段或 `to_dict()`，不复制永久 schema。可在现有解析入口加一个仅保存路径使用的完整性选项，或调用处检查；只选一种。

缺关键键准确失败，不补当前默认值；未知键仍拒绝，当前完整 exporter 输出通过。不要对每帧重新检查配方，也不比较历史物理标定与现场数值。

### 4.4 两项局部加载整理

- `inspect_policy()` 仅在实际需要补 weights/NFE 默认值时读取 `eval`。显式 checkpoint、weights、NFE 齐全时不要求 eval 存在；实际缺默认值仍失败，不随意 fallback。
- 实验发现使用 `config.yaml` 与已有支持类型的 checkpoint，当前可扫描 `checkpoints/*.pt`，不只认 latest。不加载权重做发现，不列空目录或误认 eval 快照。
- 用户选择 `best` 时，既有选点审计及 checkpoint/EMA/raw/NFE 绑定保持。

**G2 验收：** 合法单任务、额外触觉/缺点云、MultiTask 连接前拒绝；完整保存配方/缺键快照/部分用户 YAML；显式参数不依赖 eval；非 latest 实验可发现。用现有小型 fixture，不下载或构建大模型。

## 5. T03：终止首因与 session 技术结果分别正确

**F04。位置：** Real `deployment/runner.py`、`deployment/session.py`、`recording/results.py`。复用 `RunEndReason`、dispatch 和已有异常。

### 5.1 结果合同

| 情况 | attempt 证据 | session / CLI |
| --- | --- | --- |
| 正常操作者结束、完成次数、计划 max_running_s 到期 | 保留真实原因，不判断任务成功 | 无其他技术故障且清理成功可返回 0 |
| async/RTC 要求 ACCEPTED 却出现 CRC_UNCONFIRMED/部分或失败下发 | 逐设备状态不改写，终止承诺 | fault，非零；不自动开始下一 episode |
| 模型/录制/设备/清理真实异常 | 首因与后续错误详情分别保留 | fault，非零 |
| WAIT 到期且没有预算内成功消费 | TIMEOUT，detail 明确 wait | 执行技术失败，非零 |
| ESTOP | 保持已有急停语义 | 不弱化现有故障/非零结果 |

不改变 sync 的既有 CRC 容忍规则，不把 CRC 写成已证实设备损坏。任务成功判定仍独立于技术结果。

### 5.2 最小传播方式

在撤权、尽力 stop、attempt 归档之后，使用已有 `DispatchError` 携带真实结果传播严格分块的未确认下发；WAIT 耗尽可用标准 `TimeoutError`。原异常存在时保留，不为结果分类新建异常体系。若已有更短的 owner 汇总路径可直接复用，但不能同时堆叠异常、success flag 和第二套终止状态树。

在现有预算判断处一次确定到期来源，返回简单的来源/截止信息即可。统一覆盖 tick 前、动作准备/提交后的退出和 SDK 返回后的总预算检查；不要靠匹配 detail 字符串分类，也不要清空 WAIT/run 字段后再猜原因。

多个预算同时过期时，按更早 deadline 区分；相同 deadline 下 WAIT 是无动作供给的技术失败。WAIT 在首个 SDK 返回前到期，即使本次返回 ACCEPTED 也不能因随后清空 WAIT 而变正常。派发记录保持真实，只停止后续动作；计划 duration 先到期则不事后改成 WAIT。

**首因与整体结果不是同一字段：** 已锁存的操作停止/急停等首因不被后续错误改写；但当前 run 真实的 stop、录制、关闭异常仍须令 session fault。退役旧 query 的晚错误只留诊断，不污染新 attempt 或触发新 episode 故障。不能用“保留首因”吞掉真实清理失败，也不能把所有晚结果都升级为硬件故障。

`_finish_episode()` 后再传播异常，finally 不得重复计数、重复发布或重复结束同一 attempt。录制和无录制路径均向调用者返回非零/传播现有会话可处理异常，不依赖读 JSON 才发现失败。

**G3 验收：** 真实 Runner/Session 配 fake 设备、worker 和传感器生命周期，制造 arm ACCEPTED+hand CRC_UNCONFIRMED 且清理成功：attempt 留证、session fault、退出非零、无下一动作/自动下一 episode。复用现有停止用例，补 WAIT 与 duration 先后/同刻，以及 SDK 返回时 WAIT 到期；首因不被真实后续错误覆盖，后续错误也不被吞掉。

## 6. T04：输出、预热与启动预算配对修复

**F03/F05/F07/F13/F16；F20 保留。位置：** Policy `deployment/runtime.py`、`agents/action_decoders/rtc.py`；Real `deployment/{config,session,runner,inference}.py`。

### 6.1 输出与可执行准入

桥接保留张量、batch/horizon/control 形状和浮点类型检查；必要时先验证结构再切片，避免坏结构变难懂索引异常。保持完整物理 future 和现有 float64 NumPy ABI。

移除 Policy 桥接对完整 future 的提前非有限抛错。Real 在 chunk 准入统一判断有限性；未终结且归属正确的 attempt 按现有规则先保存形状/类型合法的 NaN/±Inf 快照，再拒绝执行。严格 JSON 不含非有限数；NPZ 无 object/pickle；错误结构只留诊断，旧 query 不回写终结 attempt、不延迟 stop。

独立 warmup 检查初始化和测量所使用的所有输出，包括复用的 RTC 测试 prefix。保留物理输入 prefix 有限性，以及 Real 观测、动作实现和 SDK 准入；这不是放行坏值。

### 6.2 warmup 只分两组，不建 benchmark 框架

- `bootstrap`：实际无 prefix 的普通推理。
- `steady`：所选模式的持续推理。正 guidance RTC 包含 VJP 和配置正 delay；sync/async/beta=0 RTC 复用普通路径测量，不做重复统计。

每条实际路径先做一次不计入稳态集合的初始化，再做现有数量级的少量测量。RTC 有限测试 prefix 生成一次复用。初始化成本可单列，不混入稳态集合；不能只预热 guided 而遗漏真实 bootstrap。

计时包含 Policy RGB 预处理、推理和 CPU 返回；CPU 返回是结果可用边界，不每层增加 CUDA synchronize。Real 观测构建、前缀 IK、owner 调度和 SDK 开销不在模型计时内，明确未覆盖。

load/warmup/reset/predict/close 仍在同一 worker 串行执行，warmup 前后按原协议 reset，使相同 seed 的首次真实推理不被额外预热改变。两个命名测量集合用普通返回结构即可；成对修改 worker/session/调用测试，不留两套接口或运行时版本协商。

### 6.3 验证顺序

```text
解析配置 / 识别已知不支持的部署入口
  → 验证 dt、声明模态、动作表示、所需保存配方
  → H/N/A、模式 d/beta、预算和时长组合约束
  → 严格恢复模型、核对实际消费字段、选定路径 warmup
  → 检查测量可容纳性 / 打印合法的启动摘要
  → 连接设备
```

缺 dt 给字段级错误，不先触发 None 比较或 1/dt 打印错误。启动前置验证保留；配置错误尽早发现，但不为绝对零资源分配重排整个生命周期。

### 6.4 预算使用现有整数网格，不混淆下界与建议

保留 decision age、WAIT、tick lateness 三个预算以及独立 episode 时长，不新增预算参数、不自动调 d、放宽 L 或切换模式，不改变 RUNNING/WAIT 起点。使用与 Runner 相同的整数纳秒转换，`Δ=dt_ns>0`，`0≤L=tick_lateness_ns<Δ`；实测 I 也在同一单位计算。

下面只是必要条件，不要求为所有配置求精确可达性。来源更旧、构建/排队/回收开销都会使实际情况更差；通过启动检查不代表实时保证。

| 场景 | 可用于启动判断的乐观下界/必要条件 |
| --- | --- |
| 即使普通推理趋近零，首次动作从 RUNNING 开始的等待 | 至少 `N×Δ`；WAIT 与有限 episode 时长须严格大于此下界 |
| bootstrap 完整 A 段最后动作的 query 年龄 | 至少 `A×Δ-L` |
| 普通实测耗时 I，令 `k=floor(I/Δ)+1` | 首次等待至少 `(N-1+k)×Δ`；首段最后 query 年龄至少 `max(I,k×Δ-L)+(A-1)×Δ` |
| 稳态 async/RTC 最后一个动作的 query 年龄 | 至少 `(A+d-1)×Δ-L`；也不能小于该请求已消耗的实际推理时间 |
| 固定 handoff 的推理容纳性 | 若样本 `I > d×Δ+L`，即使提交相位为零也无法赶上 handoff 截止；反之不构成能赶上的保证 |

bootstrap 选择回收时刻之后的槽，恰好处于网格边界时也不能少加一槽。年龄下界扣除 L 是容许 query 来源在其槽内晚采样；不表示刷新来源时间或延长任何 deadline。

**r2 明确替换旧静态门槛：** 不再直接用 `decision_age ≤ (A-1)dt/(A+d-1)dt` 或 `I ≥ d×dt` 判数学上必然失败。这些没有与允许的槽内相位、端点语义完全对齐。配置和 warmup 用上述必要条件；相同条件不要在 CLI、session、runner 复制三套公式。实现为小纯函数/现有验证函数即可，不构造调度搜索器或仿真平台。

WAIT/episode 是严格截止：预算等于等待下界时拒绝。decision freshness 与 handoff lateness 保留运行时的包含端点语义：仅因预算等于对应年龄下界不能拒绝；最终 SDK 的排他截止仍按现有 +1ns 表达。未来实测样本若超限，准确报告“本轮样本不可容纳”，不能称模型永远不可用。

`I < d×Δ` 以及为 owner 准备留余量可作为简短建议，不再作为额外隐藏 hard gate。允许本来就在 L 内的交接不是改变实时容限；运行中的 slot、freshness、授权与逐设备 deadline 全部保留。必须分开静态必然不可能、当前测量超限、建议余量不足三个概念，不用建议再造必填配置。

必须在连接前拒绝两例：`dt=100ms,L=30ms,I=10ms,N=1,A=1,decision_age=50ms`；`N=4,dt=100ms,WAIT=200ms,I=10ms`。同时覆盖合法 query 相位、恰好网格边界及包含/排他截止：例如 `A=3,d=2,dt=100ms,L=30ms` 的静态年龄下界为 370ms，而不是400ms；215ms 模型样本不能仅因超过200ms就被判超过230ms handoff 上限，231ms则超过该乐观窗口。其余预算须设置充足；这些是纯时序反例，不是真机性能证据。

### 6.5 RTC 局部去重

同次采样只做一次必要 timesteps 设置；理顺静态支持判断和带副作用的初始化，不新增缓存失效体系。固定实现无区分价值的 normalizer 类型检查可删；scale/offset、DDIM 条件与每个请求变化的 prefix 约束保留。DDIM 更新式、clipping、guidance、NFE 和 Agent 支持范围不变。

**G4 验收：** 实际 `LoadedPolicy.predict` 配小 fake Agent，通过真实 worker/Runner 回收和归档边界；不以 fake worker 直接返回 NaN 代替跨仓库证据。合法结果不变；非有限结果不下发且快照可读；warmup 拒绝坏输出；结构错误不任意序列化。

**G5 验收：** fake clock/小模型验证初始化不混稳态、正 RTC 覆盖两路径、beta=0、reset；上述不可能和合法边界均按实际运行公式处理。复用既有 bootstrap/handoff/漏槽/WAIT/停止用例，不测试全模型矩阵。r2 的纯数学下界检查不替代实施后的这些回归。

## 7. T05：同一入口选择是否录制，记录真实生效配置

**F11/F12。位置：** Real `examples/run_policy.py`、`deployment/session.py` 和必要使用说明；不迁移整个 YAML 树。

### 7.1 一个 --no-record 开关

默认保留记录式评估。`--no-record` 只传 `recording_config=None`，复用原 API，不创建 recorder、Raw、SessionResults、query NPZ、录制 session 目录或通过回调额外创建 run_config 目录。启动信息与退出结果仍可用；它仍是真机执行，不是离线模式。

不再增加 --record、第二个 YAML 开关或第二套 runner。显式 `--output` 在无录制时可提示一次不产生录制产物，不为无害未用参数新增 fatal gate。模型确需 RGB/cloud 时仍启用相应来源；仅关节输入且无录制时不依赖录制相机。录制模式仍严格满足 Raw RGB-D 要求，不静默丢弃相机或录制失败。

两种路径的技术失败结果一致；本轮不另建无录制持久化追踪系统。

### 7.2 复用一个生效 runtime

在已有 session 装配中，用实际 `execution_config` 更新生效 `runtime.execution`；使用保存点云时，以 `cloud_recipe` 更新活动 `runtime.pointcloud`，再解析当前桌面、构造 worker 和保存 run_config。使用现有 dataclass replace，不新增 EffectiveRuntimeConfig。硬件身份/相机参数仍归 Real。

重复序列化的活动配置须来自同一对象、值一致；未启用模块只说明 inactive，不另建配置来源注册表。沿用已有 SHA、checkpoint、weights/NFE/seed/预算记录，不做全仓 hash 或硬件审计。

启动摘要只需 checkpoint/权重/NFE、实际输入、动作、dt/N/H、A 及覆盖来源、模式/d/beta、录制开关与关键来源。合法性确认后再格式化，避免打印造成提前错误。

### 7.3 只澄清职责

说明 `policy.recording_enabled/max_record_duration_s` 属于既有 teleop 路径，不控制部署 CLI；部署由 no-record/max-duration 决定。`policy.ema` 不是模型 EMA 权重选择，也不自动启用策略平滑；`teleop.control_hz` 不覆盖模型 dt；活动点云恢复保存配方。

不重命名整个 policy 小节，不加旧字段 alias/migration。`--print-config` 仍仅打印声明 Real 配置，不加载模型、标定或设备，明确它不是最终生效快照。

**G6 验收：** CLI 解析到 fake session 装配：默认录制不退化；no-record 无录制产物，按真实输入选择来源；活动 cloud/execution 与 worker、保存快照一致；None/0/False 优先级不变。禁止真实连接。

## 8. T06：条件性实时优化与保护范围

**F09/F10/F14。这三项不要求靠新增控制机制强制关闭。**

### 8.1 F09 模式恢复

确认 stop 后恢复路径、预算及已有 trace/驱动证据。1s 就绪轮询不是实测耗时或 SDK 硬上界。**fake SDK 只能证明调用顺序，不能证明真实固件不会复活旧目标。** 缺少足够设备语义依据时，默认 `RETAINED_CONDITIONAL`，不移动模式恢复，不收集未授权硬件数据。

仅在能确认不恢复旧目标且保持 S/Q/ESC 取消时，才把必要准备移到显式 B 的准备阶段、RUNNING 前，仍由设备 owner 调用。准备前后检查 epoch/请求，完成后重新读取 HOME/反馈资格。不得自动 motion_enable、清故障、追加 HOME、空闲后台恢复或放宽 deadline。

最终 `_send()` 准入保留；有其他公开调用者依赖低层恢复时保护 Teleop/replay，不建立两套模式状态机。

### 8.2 F10 EEF prefix 峰值

只使用已有 prefix_prepared/realization/owner_tick 或必要的小型纯离线 IK 计时。无数据则性能影响 `NOT_VERIFIED`，不启动硬件采样。

不新增 IK 线程池，不取消冻结、不改 d、不复用过期解、不跨槽补发。仅消除有明确收益的重复静态构造/复制；最新反馈下的 frozen 复查保留。若现有证据仍不满足预算，报告适用范围，另行设计；不取消安全检查换吞吐。

### 8.3 F14 说明真实保护范围

在 `docs/policy_execution.md` 用小表说明 joint 的 operational limits/jump/feedback 距离/端点自碰撞，EEF workspace 裁剪及候选 IK 检查，HOME 的环境/路径规则。在线不承诺连续轨迹或环境避碰。

环境配置存在不代表在线自动受其保护。本轮不新增桌面碰撞拒绝，不改允许接触 link、标称资产、当前 mount 或 workspace；不把 joint 改成 IK。范围说明标 `SCOPE_DOCUMENTED`。

**G7 验收：** F09 若实际改动，用 fake SDK 检查 STOP→B、准备中取消、慢恢复和重新采样的调用顺序；仍不得宣称固件安全通过。F10 只报实际测量及保留理由，F14 文档对齐源码；无硬件授权全部真机 `NOT_VERIFIED`。

## 9. T07：有限清理与明确保留

**F15–F17/F20；F18/F19 在 T02。**

- 去掉 CLI→session 同链相同静态检查，保留轻量解析和 session 连接前验证。Runner 若仍独立公开，保留必要入口检查，不加 validated=True、trusted token 或永久信任层。
- 机器人先检查全部目标、驱动独立入口检查各有用途，不能仅因都检查限位删其一。动态 freshness/epoch/deadline 与 frozen 复查保留。
- F16 依 T04 局部合并，不增加缓存协议。
- **F17 保留 best 审计**，不新建轻量 loader、不删除研究证据、不自动退 latest。
- **F20 保留 float64**，性能不足以支持把精度变化混入此次修复。
- 不重写 `_patch()`、替换配置库或全仓改名；sync 中暂存不用的 d/beta 只需说明 inactive，不做额外 fatal gate。

**G8 验收：** 搜索真实调用者确认去重完整；显式加载、best 绑定、公开 Runner、normalizer 和物理准入不退化。此包不需要新的大测试矩阵。

## 10. 验证预算与交付

### 10.1 最低成本验证

G1–G8 是八组验收，不是八套新框架，也不是必须为每个 helper 建测试。合并共享 fixture，优先修改现有测试。一次小跨仓库集成可同时覆盖 G3/G4，不复制两份生命周期模拟。

顺序为：两仓库 `git diff --check`；修改文件的语法/静态检查；本包相关纯逻辑测试；必要的小型跨仓库边界 fixture。现有 Real `tests/test_policy_runner.py`、配置入口与 Policy sampler/split/部署测试可复用，运行前确认实际路径和无硬件副作用。

新时间规则必须覆盖 G1 的历史/新配方；输出/预热必须覆盖 G4/G5；session/退出码必须覆盖 G3。其余按实际变更验证，针对回归失败修相关实现，不能降低断言或放松准入让测试变绿。缺 GPU/权重不阻断可完成的纯逻辑测试，也不强制安装完整机器人栈。

只记录真正执行过的命令和结果，明确失败、环境阻断、未验证。无端到端通过证据时只能说实现完成、相关验证未完成，不说已彻底消除问题。文档里的真实部署用法不作为本轮测试命令运行。

### 10.2 完成标准

交付两个仓库的文件/函数变更摘要、关键语义变化、实际验证与剩余限制。F01–F20 每项须有去向，实施状态与测试状态分开：

| 实施状态 | 含义 |
| --- | --- |
| DONE | 修改完成，附位置；测试是否通过另列 |
| ALREADY_FIXED | 当前实现已有修复，附证据，不重写 |
| SCOPE_DOCUMENTED | 明确未接通能力/作用范围，如 F08/F14 |
| RETAINED | 有意保留，如 F17/F20 |
| RETAINED_CONDITIONAL | F09/F10 等前提不足，附触发条件 |
| BLOCKED | 对应仓库、权限、依赖或事实不足；不能代替已确认缺陷的处理 |

验证状态使用 `PASS / FAIL / NOT_VERIFIED`，不得以 mock 通过宣称 CUDA 或真机通过。在附录 B 追加一张结果表即可，不覆盖设计基线或另建追踪系统。跨仓库接口与配方变更注明对应 HEAD/工作树 diff 和未完成依赖，不增加运行时版本协商。

完成前复查：P1 和明确 P2 是否真正闭环；合法单任务有无退化；新旧配方/normalizer 是否保持约定；条件项是否诚实；有无隐式 fallback、传感器门槛、旧产物改写或无关重构。未授权实现提交时只留 diff，不 commit/push。

## 附录 A：源码证据索引

以下链接定位基线，实施以当前 HEAD 为准；路径是导航，不要求逐字保持内部命名。

| 事项 | 主要源码 |
| --- | --- |
| F01 时间/资格/恢复 | [ReplayBuffer](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/replay_buffer.py)、[Sampler](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/sampler.py)、[BaseDataset](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/base_dataset.py)、[Real QA](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/dataset/quality.py) |
| F02/F08 输入 | [Policy runtime](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/deployment/runtime.py)、[DP3](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/core/dp3.py)、[MultiTask](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/core/multi_task.py)、[训练构建](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/training/build_utils.py) |
| F03/F04/F05/F10 调度/结果 | [Runner](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/runner.py)、[Session](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/session.py)、[执行配置](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/config.py)、[结果](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/recording/results.py) |
| F06/F11/F12 配置/入口 | [点云配置](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/pointcloud.py)、[配置装配](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/experiment.py)、[控制参数](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/control.py)、[部署 CLI](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/examples/run_policy.py) |
| F09/F14 设备/保护范围 | [Robot](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/robot.py)、[xArm](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/drivers/xarm7.py)、[动作实现](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/action.py)、[执行说明](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/docs/policy_execution.md) |
| F16–F20 推理/加载 | [RTC](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/action_decoders/rtc.py)、[loader](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/loader.py)、[normalizer](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/normalization.py)、[Policy runtime](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/deployment/runtime.py) |

## 附录 B：实施结果（由执行者追加）

r2 发布时，F01–F20 未因文档审校发生代码整改，G1–G8 尚未执行，真机全部 `NOT_VERIFIED`。实施后追加 F 编号、实施状态、位置、定向验证、限制的简表；不复制完整日志，不将范围说明或保留项标为已实现。

## 附录 C：r2 最终审校记录

本修订澄清了旧 recipe 缺键与新实验显式 null、H=1/未知时间及 padding、首因与真实后续故障、配置声明与加载后核对、启动下界与槽内相位/端点、生效 execution/cloud 快照、fake SDK 的证据上限。移除旧预取 hard gate 与“仅按必要条件拒绝”的矛盾，并减少重复验收要求。

审校执行了不导入仓库的纯整数网格下界检查和六个数值边界例；这只支持本文数学关系，不代替实际 Runner 回归。设计审查通过；代码实施、实际 checkpoint、GPU 和硬件结论必须由执行者另行给出。
