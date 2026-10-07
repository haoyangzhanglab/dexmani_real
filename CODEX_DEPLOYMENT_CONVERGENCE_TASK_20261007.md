# Codex 任务书：收敛真机部署的数据、配置与执行语义

日期：2026-10-07（Asia/Singapore）  
主仓库：`haoyangzhanglab/dexmani_real`  
协同仓库：`haoyangzhanglab/dexmani_policy`  
状态：**待实施。本文是设计与实施任务，不是修复完成或真机验收报告。**

## 0. 执行目标、权限与事实基线

当用户要求执行本任务书时，先读两仓库适用的 `AGENTS.md`，再按本文完成必要修改、定向离线回归和可审查 diff。不要只重新提出计划，不为已裁决的普通实现细节反复请求确认。局部阻断只阻断对应任务，先完成其余独立项。

目标：解决审查清单 F01–F20 中真实影响科学正确性、部署可靠性和使用体验的问题；明确条件性风险与支持范围；删除有直接收益的局部冗余。**不重写调度器，不另造契约/配置/验证平台，不以减少检查数量代替正确性。**

### 0.1 代码基线

| 仓库 | 审查基线 |
| --- | --- |
| Real | `8f0070e3b73c3b0da81aa166143be44b7200f67b` |
| Policy | `52d1d2a77b76a254b1a2498b059d160e6e527590` |

SHA 用于定位证据，不是要求 reset/checkout 到旧代码。实施时检查当前 HEAD 和相关差异；保留用户已有修改。已修复项用当前代码及定向测试记为 `ALREADY_FIXED`，不得为了重放任务而恢复旧实现。

本文写入时只核对了源码与既有审查依据，未运行实际 checkpoint、GPU 性能测试或真机；此前聊天中的数值反例只作为待加入的确定性回归，不继承为本轮已通过的测试。

执行本文包含必要的跨仓库修改：Dataset、normalizer、模型桥接属于 Policy；设备、时序、Raw、会话与 CLI 属于 Real。优先使用相邻现有工作树；不能在 Real 复制 Policy 的 Dataset 或修改 site-packages。缺少协同仓库访问时完成可独立工作并报告阻断，不留静默兼容分支。

### 0.2 权限边界

本任务允许代码、文档和无硬件离线验证；默认交付 diff，**不自动 commit/push 实现**。创建本文的文档提交不等于授权执行代码整改或连接设备。

禁止连接或驱动 xArm、XHand、RealSense、VR、HTS；禁止 HOME、rollout、replay、实时采集和真实标定写入。`execute=False` 仍可能连接硬件，不是离线测试。不要升级实验环境、运行完整训练/DDP/长评测来凑验收。已发布 Raw、旧 checkpoint、实验配置和历史报告均不可原地改写。

### 0.3 与旧任务书的关系

本文是 **F01–F20 的新增实施依据**，不是重放已有 U 系列整改。根目录旧任务书保留为历史设计和执行记录，不批量删除、不回写其历史完成状态。

发生重叠时，本次仅作以下明确更新：

- 新 Real 训练增加时间连续性窗口筛选；原时间 QA 仍只报告，不能偷偷改变旧实验配方。
- 实际模型输入与 Dataset 声明对齐；Raw 全模态保存和公共导出不收缩。
- Policy 返回结构正确的预测；Real 执行准入检查有限性并保留坏预测证据；独立 warmup 自行检查有限性。
- 技术失败必须传播到 session/CLI；计划时长结束、操作者结束不自动算技术失败。
- 保留标称碰撞几何与现场位姿补偿的既有分工；不引入历史标定一致性门槛。
- 独立公开入口和真实副作用边界的必要验证仍保留，不要求全仓只出现一次 `validate`。

不要修改 `AGENTS.md` 来绕过安全和数据约束，也不要把这次任务扩写成永久架构规范。

## 1. 已裁决的总体方案

沿用现有链路：

```text
保存的 Policy config + checkpoint
    → 解析部署所需事实 / 严格恢复模型和 normalizer
    → 核对实际输入 / 选定推理路径 warmup
    → Real 装配当前硬件、标定和生效执行参数
    → 操作者开始准备 / 重新采样 / 授予 RUNNING
    → 一个设备 owner + 一个串行模型 worker + 最多一个未回收 Future
    → 完整物理 future / 调度准入 / 动作实现 / 逐设备 SDK 下发
    → 可选 Raw 录制 + attempt/session 结果
```

不新增第二套 sync 执行器、远程推理服务、模型请求队列、配置注册器或 generic processor pipeline。现有 dataclass、普通函数和 owner 状态足够。

### 1.1 不变量

1. `P = H - N + 1`；Policy future 从模型索引 `N-1` 开始，保留完整 `(P,C)`，joint19 或 EEF21。Agent 原 `pred_action/control_action/tail` 接口不重写；辅助输出不是控制量。
2. Bootstrap 保留未执行的动作头并在正常网格重锚定；async/RTC 在 `b+A-d` 预取、`b+A` 交接，从新结果索引 `d` 执行。仍要求 `1≤d≤A`、`A+d≤P`。不追赶连发，不延用失效旧 tail，不在忙碌 worker 后排队旧观测。
3. Policy 归一化/反归一化一次，Real 消费物理量。RTC 前缀来自同一份冻结的实际命令；失效后不重新 IK 再沿用旧条件。beta=0 保留普通采样分支，不强行运行 VJP，也不扩大其他 decoder 的 RTC 支持范围。
4. `intent`、准备目标、SDK 尝试状态、实测状态分别保留。`ACCEPTED` 只说明 SDK 接受，不说明到位；CRC_UNCONFIRMED 不可冒充 ACCEPTED。
5. 先验证全部存在的 arm/hand 目标，再进入第一个 SDK；每个 SDK 前仍检查 epoch、授权和 deadline。锁不跨阻塞 SDK/文件 I/O。停止优先于归档，迟返回保留真实状态，旧 Future 不污染新 attempt。
6. Raw 发布后不可变；公共导出保持现有全模态字段，辅助缺测保留 NaN。训练不能压紧坏行、补零、补图、插值动作或将 QA 报告假装成执行证据。
7. 现场标定和安装关系属于当前 Real；保存数值预处理配方属于模型。禁止把历史物理标定、原始数据路径或实验目录名称当成真机兼容 ABI。
8. A 可以在部署配置副本中覆盖；不要求等于训练 A，不修改训练 padding。CLI 的 None 表示未覆盖，显式 0/False 必须保留。
9. 无依据不改变动作表示、控制频率、IK/限位/碰撞阈值、滤波、fallback、重试或自动恢复。线程和 warmup 都不能证明硬实时或物理停稳。

### 1.2 来源与参考项目

本仓库源码事实见文末证据索引。官方参考只借鉴职责划分：

- [LeRobot 模型配置](https://github.com/huggingface/lerobot/blob/200ee53596d464bd28f6595cdb31be69a2b5e379/src/lerobot/configs/policies.py)：模型输入输出属于模型配置。
- [LeRobot 推理后端装配](https://github.com/huggingface/lerobot/blob/200ee53596d464bd28f6595cdb31be69a2b5e379/src/lerobot/rollout/inference/factory.py)：模式及其消费参数明确分开。
- [LeRobot 部署文档](https://huggingface.co/docs/lerobot/main/en/inference)：同一部署工具区分不录制执行与录制工作流；文档为动态页面。
- [ManiUniCon 装配入口](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)：data、robot、policy、sensors 分别装配。

不得据此移植它们的注册器、重试次数、设备自动回退、队列、插值或安全阈值。DexMani 的 DDIM RTC 与其他策略的 guided/trained RTC 不自动等价。本任务不新增这些项目作为运行依赖。

## 2. 完整覆盖表与优先级

F 编号是本次审查事项，不代表 20 个已复现 bug。

| ID | 审查结论 | 本次处理 | 工作包 |
| --- | --- | --- | --- |
| F01 | 实际时间断档可跨入固定 dt 训练窗口 | P1 必修：新配方时间筛选，保护旧恢复语义 | T01 |
| F02 | Dataset 模态被误当模型实际输入 | P1 必修：模型侧一次性核对，不增加传感器门槛 | T02 |
| F03 | 启动预算遗漏历史积累/网格等待 | P1 必修：修正确定下界及测量口径 | T04 |
| F04 | 未确认下发的技术失败未完整传播 | P1 必修：停止归档后传播，session/退出码一致 | T03 |
| F05 | Policy 提前抛非有限异常截断 Real 归档 | P2 必修：输出与执行准入分工统一 | T04 |
| F06 | 不完整保存点云配方补当前默认值 | P2 定向修复：保存快照完整、用户 YAML 可部分覆盖 | T02 |
| F07 | warmup 冷启动/稳态/prefix 计时混合 | P2 随 F03/F05 一并修复 | T04 |
| F08 | MultiTask 真机路径未完整接通 | P2 明确不支持；本轮不实现语言/多任务部署 | T02 |
| F09 | 阻塞模式恢复进入首槽预算 | 条件项：先证据；安全前提不足时保留并说明 | T06 |
| F10 | 单槽内 d 次 EEF IK 的准备峰值 | 条件项：先已有 trace，不凭静态代码并行化 | T06 |
| F11 | CLI 强制录制及录制相机依赖 | P2 修复：暴露现有无录制路径 | T05 |
| F12 | 声明配置、生效配置和参数用途混淆 | P2 修复：有效快照、简短来源说明；不大迁移 YAML | T05 |
| F13 | 先使用 dt，后验证 dt | P2 修复：调整验证顺序 | T04 |
| F14 | online/HOME 的 workspace/环境保护范围不同 | 范围裁决：保持控制语义，明确实际范围 | T06 |
| F15 | CLI/session/runner 重复静态校验 | P3 只删同链重复；保留公开/副作用边界 | T07 |
| F16 | RTC 重复 scheduler 初始化与低价值类型检查 | P3 局部合并，不增加缓存失效体系 | T04/T07 |
| F17 | best 依赖选点审计 | 保留：显式 checkpoint 已有轻量路径 | T07 |
| F18 | 显式推理参数仍无条件依赖 eval 小节 | P3 顺手修复：只在取默认值时读取 | T02 |
| F19 | 实验发现只认 latest.pt | P3 顺手修复：发现与已有可加载文件类型一致 | T02 |
| F20 | future 无条件转 float64 | 保留：不是已证实瓶颈，本轮不改变 dtype ABI | T04/T07 |

建议实施顺序：T01 → T02 → T03 → T04 → T05 → T06 的证据判断 → T07 收尾。T01 可独立推进；T04 的 Real 与 Policy 输出/预热接口必须成对修改。每包形成小而完整的 diff；不要求为每个 F 编号创建独立分支、提交或测试框架。

## 3. T01：训练时间连续性，只在样本消费端处理

**覆盖 F01。主要位置：** Policy `datasets/{replay_buffer,base_dataset,sampler}.py`、`training/build_utils.py` 及实际保存配方恢复调用者；Real `dataset/quality.py`、`docs/policy_execution.md` 只做必要说明，不重构导出。

### 3.1 明确选择的新研究配方

新 Real canonical 训练增加一个数值参数 `dataset.max_time_gap_ratio`，推荐实现默认 `1.5`，条件为相邻真实源行 `0 < Δt ≤ ratio × saved_dt`。**1.5 是本文明确选择的新训练数据资格参数：容纳正常周期附近的抖动、拒绝明显丢周期；不是硬件安全值，不是从已有 QA 同名比值自动推导的正确性证明。** 不保证发现所有相位漂移或累计降频。它必须写进现有保存配方，后续调整属于新实验。

仅对已识别的 Real canonical 数据应用；仿真配方不受影响。`null` 表示明确的旧式不做时间筛选配方，允许历史恢复或显式研究对照使用，但必须在配方/日志中写明 `unfiltered`，不能在时间戳损坏时自动降级成 null。当前新 Real 默认保持启用，不再要求用户增加一套必填参数。

### 3.2 实现

1. ReplayBuffer 按需读取小型 `row_info/observation_timestamp_ns`，检查长度和整数描述，不扫描无关视觉数组，不把它加入模型 observation。
2. 时间筛选启用时，缺少整个时间戳字段给出准确错误；零/未知时间戳仅使相关窗口不合格。提示核对 Raw 是否有真实时间证据；不能承诺重导出就一定能恢复未知时间，也不能生成假时间轴。
3. 保留现有 N 步观测有限性、H 步动作有限性和 dispatch 规则。在已有分块候选窗口筛选中增加时间 mask；只比较真正不同、相邻的源行。边界 padding 重复同一源行是合法的，不能按零间隔误拒；窗口仍不得跨 episode。
4. 可以复用 `source_rows()` 的有界批次或使用坏边前缀和，选择更短的实现。不构造全数据量的 windows×horizon 时间矩阵，不删除断档后的源行再压紧。
5. 新 mask 必须先于训练唯一 observation/action 源行统计生效。validation 使用同一资格规则和训练 normalizer；不能让被剔除窗口仍参与拟合。日志加时间拒绝计数，拒绝原因允许重叠；零有效窗口沿用现有明确失败。
6. 将 timestamp 字段、实际 ratio、筛选状态和计数写入已有 `data_recipe`，不另建 manifest/schema/报告体系。当前数据 QA 仍只报告，不承担隐式训练筛选。

### 3.3 历史实验与共享调用者

完整续训必须先读保存的 recipe 再构建 sampler。保存了时间规则的实验原样恢复；已知旧 recipe 没有时间筛选时，由恢复入口显式选用旧规则，不能让新增构造默认值改变其样本集合、normalizer 或 strict-resume 比较。不要原地补写旧 config/checkpoint，也不要重算旧 normalizer。

允许这一条窄的历史语义恢复，不建设通用 migration。若某历史产物缺少足够证据而无法确定原规则，明确拒绝 full resume 并说明；只读推理不受影响。新 ratio 或新筛选集合不能混入严格续训。

检查现有 Policy-aligned VQ 训练、usage/验证、split 恢复调用者是否复用同一 Dataset/recipe。必要时小改透传保存的时间规则，避免 Policy 采用新窗口而 VQ 悄悄沿用另一集合；独立 VQ 研究接口不被强制改造。

**验收 G1：** 用小型合成 Real 数据覆盖连续时间、明显长间隔、非递增/未知时间、episode 边界和合法 padding；只剔除相关窗口，唯一统计源行随之正确改变。再检查旧 recipe 恢复不变、新 recipe 恢复一致、仿真不受影响。真实数据受影响比例若无数据只能 `NOT_VERIFIED`，不做全盘扫描。

## 4. T02：模型所需事实、保存配方与轻量加载

**覆盖 F02、F06、F08、F18、F19。主要位置：** Policy `deployment/runtime.py`、模型构建处、已有 `consumed_observation_fields`；Real `deployment/config.py`、`config/pointcloud.py`。

### 4.1 实际输入只有一个模型侧来源

复用当前模型/encoder 已有的实际消费字段声明，不建立 Real 中的 model-name→modalities 映射，也不通过 tracing/反射扫描 forward 猜输入。

- 新训练在实际模型构建完成时核对一次 Dataset 声明与模型输入。缺少必要输入明确报错；当前单任务配方多声明未消费模态也给出明确错误，修正配方，而不是让它悄悄参与 normalizer、缺测筛选和真机准入。
- 部署加载完成、设备连接前核对一次相同事实；`PolicyInfo` 的声明不可未经核对就变成真机必需传感器。不要每次 predict 重新构造/比较整份契约。
- 不把 Raw 全模态、动作辅助监督字段或训练专用标签硬塞进这个相等关系。训练侧 MultiTask 还消费 Agent 层文本，不能仅依据 encoder 列表删除 `task_text`。
- 历史 checkpoint 若有声明偏差，不偷偷删字段使其通过。报告声明字段与消费字段，保留原产物；不新增自动猜测修复。
- 当前没有实际消费字段声明的自定义 Agent，不默认信任为空，也不创建能力注册器；在本真机入口准确报告尚未接入。已有合法单任务模型不应被误拒。

### 4.2 MultiTask 本轮只明确支持边界

当前真正部署前一次明确拒绝尚未接通的 MultiTask/text-conditioned 路径，解释缺少任务选择和 child 数值配方恢复。拒绝必须发生在设备连接前，不让它最终变成空模态、dt TypeError 或 task_text KeyError。

不要新增 `--task` 默认值、隐藏文本、语言推理、child 配方合并或 per-task normalizer 来完成本任务；不影响已有多任务训练/仿真。将该项记为 `SCOPE_DOCUMENTED`，不是“MultiTask 已修好”。

### 4.3 保存数值配方与用户部分配置分开

用户 Real YAML 继续由现有 `_patch()` 覆盖当前默认值。保存的 Policy 点云配方则必须包含本实现所需的完整数值字段：复用现有 dataclass 字段或 `to_dict()` 键，不复制一份永久 schema。

最短实现可给现有解析函数增加一个仅保存入口使用的 `require_complete` 参数，或在调用处做一次缺失键检查；只选一种。缺少关键字段明确报错，不自动补当前值。未知字段保持现有拒绝，完整当前 exporter 输出直接通过。不能因此要求保存的历史相机/桌面/手部安装值与现场一致。

### 4.4 顺手去掉两个无关限制

- `inspect_policy()` 仅在需要补默认 weights/NFE 时读取 `eval`；显式 checkpoint、weights、NFE 齐全时，不要求 eval 小节存在。缺少实际需要的默认值仍明确失败，不静默改用随机默认或 fallback。
- `list_experiments()` 使用 `config.yaml` 与现有支持类型的 checkpoint 文件判断，当前可扫描 `checkpoints/*.pt`，不只认 latest。空目录不列出，不加载权重来做发现，不把 eval 快照目录误认成训练实验。
- `best` 的审计与所选 checkpoint/EMA/raw/NFE 绑定保持，不因显式参数优化而绕开用户明确选择的 best 校验。

**验收 G2：** 当前合法单任务通过；DP3 多声明触觉和缺必要点云分别准确失败；MultiTask 在连接前准确拒绝；完整保存配方通过、缺键保存配方失败、部分用户 YAML 仍通过；显式推理参数不依赖 eval；仅含非 latest checkpoint 的实验可发现。用现有入口 fixture，不重建大模型或连接传感器。

## 5. T03：技术失败贯通 attempt、session 和 CLI

**覆盖 F04。主要位置：** Real `deployment/runner.py`、`deployment/session.py`、`recording/results.py`；复用 `DispatchError`、`RunEndReason` 和现有首因机制。

### 5.1 明确结果合同

| 结束情况 | attempt | session / CLI |
| --- | --- | --- |
| 操作者正常结束、完成约定运行次数、到达计划 `max_running_s` | 保留真实原因，不据此判断任务成功 | 无其他技术错误且清理成功可正常结束、返回 0 |
| async/RTC 需要 ACCEPTED 却得到 CRC_UNCONFIRMED/部分或失败下发 | 保留逐设备真实结果并终止本次承诺 | 技术失败；session fault、CLI 非零 |
| 模型/录制/设备/清理异常 | 保留本 run 首因和后续错误详情 | 技术失败；session fault、CLI 非零 |
| WAIT 到期、一直无法提供可执行计划 | 保留 TIMEOUT，detail 明确 wait 原因 | 执行技术失败，不冒充计划时长正常完成 |
| ESTOP | 保留已有急停语义 | 不弱化当前非零/故障处理 |

不改变 sync 原有 CRC 容忍语义，不把 CRC 等同于已证实的硬件损坏。任务成功与否仍由任务评估判定。

### 5.2 最小实现路径

最短方案是在完成撤权、尽力 stop 和 attempt 归档之后，用已有 `DispatchError` 携带原 `DispatchResult` 传播严格分块的未确认下发；WAIT 耗尽可使用标准 `TimeoutError`，不引入新的异常体系。已有异常保留为主异常，清理异常作为详情，不能覆盖最早终止原因。

区分 `_budget_deadline_ns()` 中 episode 与 WAIT 的来源，而不仅返回一个无法解释的超时。可用现有 owner 字段计算到期来源；多个预算已同时到期时以更早 deadline 为准，相同 deadline 下 WAIT 仍是未供给动作的技术失败。若本 run 已锁存更早的操作停止/急停等原因，沿用首因规则，不把后发现的旧 query 错误变成新故障。

`_finish_episode()` 已执行后再向外传播时，run/finally 不可重复计数、重复发布 Raw 或多次结束同一 attempt。技术失败不自动启动下一 episode；后续 session 不得因 `quit_requested` 或清理成功将它改成成功。无录制 API 路径也必须向调用者返回非零或抛出既有会话可处理异常，不能依赖 JSON 文件才知道失败。

如果当前代码已有更短的 owner 结果汇总方案，可复用；验收合同不变。不要同时保留 exception、另一套 success flag 和重复终止状态树来表达同一事实。

**验收 G3：** 用真实 Runner/Session 装配路径配 fake robot、fake worker、fake sensor 生命周期，制造 arm ACCEPTED+hand CRC_UNCONFIRMED，且所有 stop/归档/关闭成功；断言 attempt 真实状态、session fault、CLI 非零、无下一动作/自动下一 episode。另区分正常 duration 与 WAIT、确认首因不被归档错误覆盖；复用现有停止回归。

## 6. T04：输出准入、预热和启动预算一次修完整

**覆盖 F03、F05、F07、F13、F16；F20 明确保留。主要位置：** Policy `deployment/runtime.py`、`agents/action_decoders/rtc.py`；Real `deployment/{config,session,runner,inference}.py`。

### 6.1 输出合同

`LoadedPolicy.predict()` 保留 tensor、batch/horizon/control 形状与浮点类型的结构检查，必要时先检查再切片，避免坏结构产生难懂的索引错误。仍返回现有物理 future，**本轮保持既有 float64 NumPy 输出，不夹带 dtype 优化**。

删除桥接层对完整 future 的提前非有限抛错。Real `_poll_model()` 在完整 chunk 准入时统一检查有限性；活跃 attempt 先按既有规则保存形状/类型合法的 NaN/±Inf 数组，再拒绝执行。严格 JSON 不写非有限数值；NPZ 仍禁止 object/pickle。错误 shape/dtype 不序列化任意对象。

独立 warmup 对每次用于测量的输出明确检查有限性；正 guidance 的测试 prefix 也在使用前检查。保留物理 RTC 输入 prefix 的有限性检查，以及 Real 的观测有效性、目标实现和 SDK 物理安全检查。移除的是一处提前截断证据的冲突，不是允许坏值穿过控制准入。

晚返回旧 Future 仍只属于原 query/session 诊断，不回写已终结 attempt、不污染下一 episode、不延迟 stop。

### 6.2 warmup 最小分工

将当前混合循环整理为两个有名字的测量集合即可，不建立 benchmark 对象体系：

- `bootstrap`：实际无 prefix 的普通推理路径。
- `steady`：所选执行模式真正使用的持续推理路径。正 guidance RTC 包含 VJP 与配置的正 delay；sync/async 以及 beta=0 RTC 与普通路径等效，可复用同一测量集合。

先做每条实际路径的一次不计入稳态统计的初始化，再做现有数量级的少量测量；不增加大量采样、百分位置信度或自动调参。正 guidance 的有限测试 prefix 生成一次后复用，不在每个测量样本前再跑一次普通预测。初始化成本可单独记录，但不能与稳态样本混作一种延迟。

测量继续覆盖 Policy RGB 预处理、推理和 CPU 返回；CPU 返回形成结果可用边界，不额外每层 CUDA synchronize。Real 的观测构建、前缀 IK、派发开销不算模型测量的一部分，应明确其未覆盖范围。

所有 load/warmup/reset/predict/close 仍在同一 worker 串行执行。warmup 前后按既有协议 reset RNG/episode 状态，不能改变首次真实推理的随机序列。返回结构只在现有 worker→session 接口做成对修改；不得遗留新旧两套 warmup 或动态 ABI 协商。

### 6.3 验证顺序

```text
解析外部配置 / 确认受支持的部署路径
  → 检查 dt、实际模态、动作表示和所需保存配方
  → 检查 H/N/A、所选模式 d/beta、三个预算及运行时长
  → 严格恢复模型和选定路径 warmup / 测量可容纳性
  → 设备连接
```

未知模型支持问题在模型加载后、连接前明确失败。缺 dt 必须得到字段级错误，不先发生 None 数值比较或 `1/dt` 打印错误。配置错误原则上在创建输出目录、共享内存和昂贵模型前尽早发现，但不要为绝对零资源分配重排整个生命周期。

### 6.4 预算裁决：确定下界与测量估计分开

保留三个现有预算：decision age、WAIT、tick lateness；总 episode 时长独立。不新增预算参数、不自动放宽或回退模式、不改变 RUNNING/WAIT 起点来掩盖失败。

使用实际整数纳秒网格，注意 bootstrap 取的是回收时刻之后的正常槽，不能把恰好落在槽边界的结果当成早一个槽。设 `Δ=dt_ns`、`L=tick_lateness_ns`：

- 即使推理趋近零，积累 N 行并等待 bootstrap 后，RUNNING 到首次可消费槽也至少需要 `N×Δ`。WAIT 和有限 episode 时长必须能严格覆盖这一确定下界。
- 首次完整 A 段最后一个动作的 query 年龄具有下界 `A×Δ-L`。这里扣除 L 是为了不把允许的观测/提交槽内相位误判成必然失败；来源更旧只会增加年龄。
- 对实测普通路径耗时 `I`，可用 `k=floor(I/Δ)+1` 计算理想最早 bootstrap 槽数；对应等待下界为 `(N-1+k)×Δ`，首段最后动作的年龄下界不小于 `max(I,k×Δ-L)+(A-1)×Δ`。这些是忽略额外 owner/排队成本的乐观下界，不是总时延上界。
- 仅对确定不可容纳的配置和测量样本作拒绝；可另打印简单、保守的网格预算建议，不能把保守建议未满足一律说成数学上不可能。稳态 async/RTC 继续用实际所选路径测量对照既有 `d×dt` 预取预算，并保留现行严格交接规则。
- WAIT 是严格截止；decision freshness 保留现行包含端点语义，边界运算与运行时一致。模型完成、新反馈、无效重试都不能刷新旧 query 源时间或 WAIT deadline。

无需写调度仿真器来算这些量；可用一个小纯函数与现有 slot 运算，避免 CLI/session/runner 各自复制不同公式。不要声称通过 warmup 就保证 EEF 前缀准备或 SDK 总延迟满足预算。

必须在连接前拒绝两个回归反例：`dt=100ms,L=30ms,I=10ms,N=1,A=1,decision_age=50ms`；以及 `N=4,dt=100ms,WAIT=200ms,I=10ms`。同时保留宽松合法预算、合法槽内相位与正常 bootstrap，不以修反例为由统一加长首动作等待。

### 6.5 RTC 局部清理

同一次预测只做一次必要的 timesteps 设置。把带副作用的 scheduler 初始化与静态支持检查关系理顺，不额外建立 cache key/失效系统。固定实现下无区分价值的 normalizer 类型重复检查可删除；已保存 scale/offset 合法性、支持的 DDIM 配置和变化的请求 prefix 约束保留。不要改变 DDIM 更新式、clipping、guidance 数学、NFE 或允许的 Agent 范围。

**验收 G4：** 使用实际 `LoadedPolicy.predict` 桥接方法配小型 fake Agent，再连接实际 worker/runner 回收逻辑；不能仅让 fake worker 直接返回 NaN 就宣称跨仓库通过。合法 future 不变；坏 future 不下发而按合同归档；独立 warmup 拒绝坏输出；结构错误只保留诊断。

**验收 G5：** 用 fake clock/fake 模型分别检验初始化不进入稳态集合、正 RTC 同时覆盖 bootstrap 与 guided、beta=0 等效路径、reset 行为；上述预算反例失败、正常配置通过；已有 handoff/漏槽/WAIT/停止用例不退化。没有真实 GPU 时延就注明，不跑全模型矩阵。

## 7. T05：同一入口可录制或不录制，配置显示真实生效值

**覆盖 F11、F12。主要位置：** Real `examples/run_policy.py`、`deployment/session.py`、相关文档；必要时修改 config 注释，不搬迁整个 YAML 树。

### 7.1 入口裁决

增加一个 `--no-record` 开关，默认不指定时保持当前记录式评估行为。不要同时新增 `--record`、YAML deployment recording flag 和第二套 runner。

`--no-record` 只让入口传 `recording_config=None`，复用现有 API：不创建 recorder、Raw、SessionResults、query NPZ 或录制 session 目录。启动信息与结果退出码仍可用；无录制不意味着没有真实执行，也不意味着离线模式。显式同时传 `--output` 时可一次提示此路径不产生录制产物，不为无害未用参数增加新的 fatal gate。

相机/点云进程按**模型实际输入或明确录制需求**启用。只用 joint 的模型在无录制时不依赖相机；需要 RGB/cloud 的策略不能因 no-record 少启动必需传感器。录制模式继续严格保持 Raw 所需 RGB-D，不能静默丢弃录制相机。

技术失败退出语义必须在两种入口一致；本轮不额外建设无录制持久化追踪系统。

### 7.2 生效配置单一快照

当使用保存的点云配方时，在现有 session 装配中将活动 `runtime.pointcloud` 替换为实际 `cloud_recipe`，再解析当前桌面快照、构造 worker 和保存 `run_config.yaml`。硬件相机参数仍来自 Real，保存的数值配方不能覆盖当前设备身份。不要维护第二份完整 `EffectiveRuntimeConfig`。

`run_config` 中重复的表示若保留，必须由同一生效对象导出且值一致；未启用模块的声明参数注明 inactive，不说它们被实际使用。沿用已有 SHA、checkpoint 路径、weights、NFE、seed 和预算字段，不新增每次运行全仓 hash/硬件审计。

启动摘要简短列明：checkpoint/权重/NFE、真实输入、动作表示、dt/N/H、A 是否被覆盖、模式/d/beta、录制是否启用，以及必要的参数来源。所有摘要在合法性确认后打印，不能因格式化造成新异常。

### 7.3 配置职责说明，不做大迁移

在使用文档和相关注释中澄清：

- `policy.recording_enabled`、`policy.max_record_duration_s` 是既有 teleop 相关配置，不控制本部署 CLI；部署分别由 no-record 与 max-duration 决定。
- `policy.ema` 是既有控制平滑设置，不是模型 raw/EMA 权重选择，也不为策略隐式启用动作平滑。
- `teleop.control_hz` 不覆盖保存的策略 dt。
- 活动 Policy 点云使用保存配方，当前 Real 的点云声明不是第二个隐式 override。

这轮不重命名整个 `policy` 小节、不增加旧字段 alias/migration。`--print-config` 保持只打印声明 Real 配置，不加载模型、标定或设备；准确说明它不是最终生效快照。

**验收 G6：** 经 CLI 参数解析到 session 装配的 fake 生命周期：默认仍录制；no-record 不创建录制产物；joint-only 无录制不启动相机；RGB/cloud 仍启动必需来源；活动 cloud 参数与 worker、保存快照一致；None/0/False 的既有覆盖优先级不变。测试禁止真实连接。

## 8. T06：实时耗时与保护范围，按证据处理

**覆盖 F09、F10、F14。** 这三项不是“必须通过新增控制机制修掉”的通用 bug。

### 8.1 F09 模式恢复

先检查当前驱动和已有 trace/离线 fake SDK 证据，确认 stop 后是否确需模式恢复、其调用位置与预算。没有实机时不能宣称实际耗时超限，也不能把驱动的 1s 轮询预算当成硬实时保证。

仅当可确认模式恢复不会恢复旧目标、且能保留停止/退出取消时，才将必要准备移到显式 B 的开始准备阶段、RUNNING 之前；仍由设备 owner 调用。准备前后检查 epoch/请求，完成后重新读取 HOME/反馈资格，随后才授予 RUNNING。不能自动 motion_enable、清故障、追加 HOME、放宽 deadline 或在空闲后台恢复模式。

最终 `_send()` 的逐设备授权和 deadline 不可删除。若仍需保留低层恢复逻辑以支持其他调用者，说明其公开入口用途，不能为了此任务破坏 Teleop/replay；也不建立两套模式管理状态机。

若当前缺乏安全证明或已有耗时不足以支持移动，记 `RETAINED_CONDITIONAL`，保留当前失效/停止行为，说明触发条件和未验证点，不强制改硬件状态流程。

### 8.2 F10 EEF prefix 峰值

先使用已有 `prefix_prepared`、`realization`、`owner_tick` 记录或小型纯离线 IK 计时，区分模型、观测、前缀和下发开销。没有既有数据就明确性能影响未验证，不启动硬件收集。

本轮不增加 IK 线程池、不取消前缀冻结、不改变 d、不复用过期解、不跨槽补发。明显的重复静态构造/相同数据复制可局部消除；冻结命令在最新反馈下的验证必须保留。若证据仍显示单槽准备不能容纳，清楚报告当前模式/参数不适用，另行评估算法或控制设计，不用取消安全检查换吞吐。

### 8.3 F14 保护范围

保持现有在线 joint/EEF/HOME 分工，在 `docs/policy_execution.md` 用小表准确说明：joint 的 operational limits/jump/feedback 距离/端点自碰撞，EEF 目标 workspace 裁剪与 IK 候选检查，HOME 的环境/路径检查。说明在线模型没有承诺连续轨迹或环境避碰。

不自动给所有策略接入桌面碰撞拒绝，不修改允许接触 link、名义资产、当前 mount 或 workspace 规则。`environment.static_boxes/table` 的配置存在不意味着在线动作自动受它们保护。若将来明确要求 joint 也受 EEF workspace 约束，可单独讨论 FK 端点检查；不在本任务偷偷把 joint 路径改成 IK。

**验收 G7：** F09 若改动，fake SDK 覆盖 STOP→B、准备中 S/Q/ESC、慢恢复、准备后重新采样，且没有复活旧 target、提前授予 RUNNING 或新增动作。F10 只报告实际执行的测量与保留结论；F14 文档与实际构造路径一致。无硬件授权，三项真机状态仍为 `NOT_VERIFIED`。

## 9. T07：有限清理与明确保留

**覆盖 F15–F17、F20；F18/F19 已在 T02。**

- 去掉 CLI→session 同链中完全相同的 execution 数值检查，保留轻量入口解析和 session 的连接前验证。独立公开 Runner 保留必要构造检查；不要加 `validated=True`、trusted token 或“一次认证永久信任”的新层。
- 机器人层先检查全部目标再调用第一个 SDK，以及驱动独立公开入口的检查有不同用途，不能仅凭都检查限位就删除其一。状态变化后的 freshness/epoch/deadline 和冻结命令复查保留。
- F16 按 T04 合并同次采样重复初始化；不为微小节省引入缓存协议。
- **F17 保留当前 best 审计。** 用户显式指定 checkpoint 已有非 best 路径，不另建轻量 loader，不删除研究选点证据，不在 best 失败时自动退 latest。
- **F20 保留 float64 输出。** 未证明是瓶颈，不把精度/归档 ABI 变化混入此次主要修复。未来依据性能证据单独处理。
- 不重写 `_patch()`、换配置库、改全仓 YAML 命名、给所有未用参数增加强制错误；sync 暂存未使用 d/beta 可以只说明 inactive。

**验收 G8：** 搜索实际调用者确认删除逻辑无遗漏；显式加载、best 选择语义、公开 Runner 准入仍正确。配置、shape/normalizer、物理限制等现有失败边界不得因去重而失效。该项不需要新增大规模测试。

## 10. 最低成本验证与交付

### 10.1 验证预算

只维护 G1–G8 八组定向验收，优先扩展现有 fixture 和测试文件；一组可含少量参数化反例，不是八套框架。已有 bootstrap/交接/停止/部分下发回归直接复用，不复制第二份测试。

最低执行顺序：

1. 两仓库 `git diff --check`；修改文件的语法和静态检查。
2. 与本包变动有关的纯逻辑测试，例如现有 Real `tests/test_policy_runner.py`、配置入口测试，Policy sampler/split/部署测试；执行前核对实际文件和测试副作用。
3. 一条小型跨仓库 Policy bridge→worker→Runner→结果的集成 fixture，证明 NaN 归档、输出结构和技术结果真实穿过边界，而非每层各自 fake 成功。
4. 仅有现成依赖且确需验证时使用很小的 CPU 模型/原生离线 FK；不强制下载预训练权重、安装完整机器人栈或进行全方法 GPU smoke。

对新增默认时间筛选与恢复语义变更必须执行 G1；输出/预热变更必须执行 G4/G5；session 退出码变更必须执行 G3。其余测试按实际修改选择。只写执行过的命令及结果；缺环境用 `NOT_VERIFIED`，不能以 mock 通过宣称 CUDA、硬件或真实闭环已验收。

不要执行文档中的 live CLI 示例。需要说明使用方法时，只补现有命令如何增加 `--no-record`、如何显式选择 checkpoint、哪些参数来自保存配置，不另写自动启机脚本。

### 10.2 完成清单

交付两仓库变更文件摘要、关键语义变化和实际验证结果。所有 F01–F20 都必须有去向：

- `DONE`：已实施，附文件/函数和对应 G 组结果。
- `ALREADY_FIXED`：当前 HEAD 已有修复，附证据。
- `SCOPE_DOCUMENTED`：明确本轮不支持/作用范围，如 F08、F14，不伪装成功实现。
- `RETAINED`：有意保留，如 F17、F20。
- `RETAINED_CONDITIONAL`：性能或硬件前提不足的 F09/F10，附触发条件。
- `BLOCKED`：缺仓库、依赖或事实导致未完成；不能用于跳过已确认缺陷。

验证状态另外记 `PASS / FAIL / NOT_VERIFIED`，不把实现完成与验证通过混为一谈。无需另建追踪系统或每项一个报告；在本文末尾追加一次简短实施结果表即可，不覆盖设计基线。

跨仓库输出/预热接口和时间配方变更应在交付中标明配对版本及未完成依赖；不引入运行时版本协商。用户未授权实现提交时交付 diff，不自行 push。

完成标准：P1 和明确 P2 修复有端到端证据；条件项处理诚实；合法现有单任务行为不退化；无新的隐式 fallback/动作滤波/传感器门槛；无旧 Raw/checkpoint 改写；没有为了“消除所有问题”把范围说明和低收益清理扩大成架构重写。

## 附录 A：代码证据索引

以下链接定位审查基线；实施以当前 HEAD 为准。函数与路径是导航，不要求逐字保留内部命名。

| 事项 | 主要源码 |
| --- | --- |
| F01 时间、资格与配方 | [Policy ReplayBuffer](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/replay_buffer.py)、[Sampler](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/sampler.py)、[BaseDataset](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/datasets/base_dataset.py)、[Real QA](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/dataset/quality.py) |
| F02/F08 输入与多任务 | [Policy runtime](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/deployment/runtime.py)、[DP3](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/core/dp3.py)、[MultiTask](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/core/multi_task.py)、[训练构建](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/training/build_utils.py) |
| F03/F04/F05/F10 调度与结果 | [Runner](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/runner.py)、[Session](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/session.py)、[执行配置](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/deployment/config.py)、[结果归档](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/recording/results.py) |
| F06/F11/F12 配置与入口 | [点云配置](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/pointcloud.py)、[配置装配](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/experiment.py)、[控制参数](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/config/control.py)、[部署 CLI](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/examples/run_policy.py) |
| F09/F14 设备和保护范围 | [Robot](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/robot.py)、[xArm 驱动](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/drivers/xarm7.py)、[动作实现](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/dexmani_real/robot/action.py)、[执行说明](https://github.com/haoyangzhanglab/dexmani_real/blob/8f0070e3b73c3b0da81aa166143be44b7200f67b/docs/policy_execution.md) |
| F16/F17/F18/F19/F20 推理与加载 | [RTC](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/action_decoders/rtc.py)、[loader](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/loader.py)、[normalizer](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/agents/normalization.py)、[Policy runtime](https://github.com/haoyangzhanglab/dexmani_policy/blob/52d1d2a77b76a254b1a2498b059d160e6e527590/dexmani_policy/deployment/runtime.py) |

## 附录 B：实施结果（由执行者追加）

本文创建时：F01–F20 均未因创建任务书而发生实现变更；G1–G8 均未在本次文档任务中执行；真机全部 `NOT_VERIFIED`。

实施后在此追加一张 F 编号、状态、变更位置、定向验证、剩余限制的简表即可。不要复制完整日志、扩大验收矩阵或把未实施项标成完成。
