# DexMani Real：参考 ManiUniCon / Dexora 的减法重构任务书

> 对象：在本地 `dexmani_real` 工作区执行任务的 Codex CLI。
> 目标：实际完成小步重构与离线验证，而非再次生成一份泛化架构建议。
> 本文由用户明确要求入库，是本次任务的执行规格；不替代 `AGENTS.md`，不修改 Codex 权限，不授权真实硬件操作。本文的加入不表示实现或测试已经完成。

## 0. 基线、范围与执行方式

编写日期：2026-10-25。源码参考固定在以下提交：

| 仓库 | 审读基线 | 用途 |
| --- | --- | --- |
| `haoyangzhanglab/dexmani_real` | `c7936ca97f26b0f3f879f03193755b5e4b650cbf` | 本次修改对象 |
| `Universal-Control/ManiUniCon` | `85c6f2e32ecf9f2bed62d202b058c39623444686` | 参考装配、观测/动作适配、通信职责划分 |
| `dexoravla/Dexora` | `6f2869d61483774aa75cda4cd2260dc6a5faa783` | 参考薄策略接口、设备依赖隔离、直观部署数据流 |

这些提交只提供证据定位，不是版本准入门禁。开始前读取实际 HEAD、`git status --short`、已有 diff、根目录和适用子目录的 `AGENTS.md`。若实际代码前进，核对相关差异并基于现状实施，禁止重置到本文基线。保留用户未提交修改；发生真正重叠时隔离冲突，不覆盖、不自动 stash，不执行 `reset --hard`、强推或删除工作目录。

默认修改范围为本仓 `dexmani_real/`、必要的 `scripts/` 离线检查及与实际改动直接相关的 README 说明。`dexmani_policy` 仅作为现有接口依赖读取；未经另行授权，不修改或提交外仓，不扩展策略模型、Dataset 或训练算法。参考仓库只读，不安装、不执行其中脚本，不将它们复制进 `third_party/`。

本任务按 M0–M6 连续推进，每阶段先核对现状、实施最小改动、运行针对性验证，再进入下一阶段。阶段已经由新代码满足时，给出证据后跳过，不能为完成清单重写正确实现。普通可逆选择自行完成；真正需要新硬件操作、外仓写入或改变实验语义的部分留作明确阻塞项，继续完成其余已授权工作。默认只留下可审阅的本地 diff，不自动 commit、push、创建 PR 或修改远端。

本次入库任务书是用户明确要求的文档；实施中的分析、基准日志和阶段状态放系统临时目录，不再新增根目录任务、审计或验收文件。不修改 `AGENTS.md`、`.codex/config.toml` 或权限设置来绕过约束。

## 1. 目标与完成标准

核心原则：**一个事实一个权威来源，一个决策一个负责人，一个资源一个生命周期拥有者。**

需要改善：

1. 校验、协议、配置、异常处理更少重复解释，不用新框架藏起旧复杂度。
2. 物理观测、模型特征、动作意图、最终命令、持久化证据之间的数据流直接、清晰。
3. 热路径减少不必要的依赖推导、数组转换、复制与重复数值扫描。
4. 代码能够按 producer → transform → consumer → side effect 追踪；故障可以定位，必要收尾不被跳过。

完成不能只用代码行数证明。必须交付实际 diff、可重复的离线回归、校验删除依据、资源拥有权说明和测量结果或明确未测限制。行为保持重构与已确认正确性修复分开说明；不能同时改频率、调度、数据格式和安全预算，再宣称是等价精简。

## 2. 对参考项目的取舍：借鉴职责，不移植框架

以下是对固定源码的观察；“本仓落点”是本任务的设计决策，不表示参考项目与本仓具有同等安全、时间或数据保证。证据链接见第 11 节。

| 参考证据 | 值得借鉴的机制 | 本仓落点 | 明确不照搬 |
| --- | --- | --- | --- |
| ManiUniCon `main.py` [MU1] | 入口集中装配机器人、策略与传感器 | 保留入口/Session 为装配位置，底层只接收所需依赖 | 不引入 Hydra，不复制其退出顺序、全局 flags 或通用管理器 |
| ManiUniCon `TorchModelPolicy` [MU2] | 观测适配 → 模型 → 动作适配的路径直观 | 收拢现有 `deployment/observation.py`、PolicyRuntime 和 ActionMapper 职责 | 不新增 wrapper 继承链；不让特征适配器掌握整个共享存储、键盘和 Recorder |
| ManiUniCon `SharedStorage` [MU3] | 观测 ring 与动作通道分工明确，数据具有时间身份 | 保留本仓 sensor ring、arm 目标槽、Recorder FIFO 各自语义 | 不建立 all-in-one SharedStorage，不把 arm 单槽改成整个预测 chunk 的队列 |
| Dexora `dexora_policy.py` [DX1] | 模型细节封装在薄推理接口之后 | 继续使用已有 PolicyInfo / PolicyRuntime，不让设备层导入模型实现 | 不再套一层 Dexora 风格 PolicyFacade；不引入新动作布局 |
| Dexora deploy 说明 [DX2] | SDK/模型依赖按真实环境冲突隔离 | 维持 xArm owner、XHand 当前 I/O owner、单模型线程的明确所有权；SDK 延迟导入 | 不因为对称而新增 XHand 进程，不引入 ZMQ、JSON RPC 或多环境服务 |
| Dexora inference loop [DX3] | gather → infer → execute 主路径易读，动作路由集中 | 让 episode 每拍处理顺序易读，转换逻辑不散落在 session 中 | 不以 `t % chunk_size` 替换 sync/async/rtc，不复制 wall-clock 调度 |
| Dexora dataprocess 说明 [DX5] | 复用存储库能力而非重复实现文件容器 | 继续让现有 HDF5/PyAV/Zarr 承担其职责，数值变换与发布分离 | 不迁移到 LeRobot/BSON，不复制覆盖输出默认值，不保留双套新旧导出 |

### 2.1 必须排除的参考实现行为

已审读的 Dexora `xhand_forwarder.py` [DX4] 包含模块导入时打开串口的路径；其动作处理包含非有限值清洗、维度补零/截断，部分反馈失败使用零值，并含发送节流与重试机制。`dexora_policy.py` [DX1] 也包含状态维度补齐/截断及权重加载回退。它们是参考实现的具体选择，**不是本仓可接受的简化方式**。

本任务禁止引入这些行为：导入连接设备、自动 HOME、缺测填零、`nan_to_num` 修补动作、猜测单位、静默截断错误维度、吞错继续运行、修改频率、按变化量跳过本应发送的动作、额外 retry/backoff、相机回退和未经授权的控制补偿。

本仓已有明确语义的关节限位投影、整圈等价角选择和触觉有界保持不属于新增回退；本任务保持它们现有行为，不把“禁止新增补值”误解为删掉现有触觉处理。

### 2.2 相比前一轮方案的调整

- 不追求通用机器人控制平台；仅收窄现有边界，不新增服务层、registry、消息总线、状态机 DSL、DI 容器或 BaseSession。
- 不把所有数据统一为一个大 observation/action 对象；保留物理快照、模型投影和录制投影的不同内存生命周期。
- 不把所有异常和状态统一为一个 `success`；保留采集结果、停止结果、文件发布结果的区别。
- 不将跨仓库 artifact API 改造设为本轮前置条件；先利用已经存在的公共元数据，在本仓集中使用。
- 不把新的诊断账本、逐 taxel 掩码、历史标定准入或模型原始动作存储加入核心范围。
- 先用普通函数与现有类型完成重构；只有移除真实重复状态或明确拥有可变资源时才引入新类型。

## 3. 必须保持的系统契约

| 编号 | 不变量 | 不能采用的简化 |
| --- | --- | --- |
| I01 | 任何普通 import、配置打印、纯离线检查不得连接设备、运动或写现场标定 | 导入时初始化 serial/相机/SDK，测试直接运行参考 forwarder |
| I02 | 运动授权、设备执行状态、物理停稳是不同事实 | `stop()` 返回或 DISARMED 被当作已物理停稳 |
| I03 | 保留 run/query/operation/input generation/sequence 各自的因果身份 | 合并成一个万能 id，接受旧运行的迟到结果 |
| I04 | 传感器缓存与派生计算不得刷新原始来源时间 | 将缓存读取时刻、点云完成时刻当作新观测时间 |
| I05 | 点云与同时消费的 RGB-D 继续按源相机序号配对 | 源帧被覆盖后拿最新 RGB 顶替 |
| I06 | 控制需求与录制需求分别归因；开启 Raw 仍要求既定完整录制来源 | 录制失败静默降级，或强制所有策略无条件依赖全部模态 |
| I07 | 普通模型输入/输出保持既有字段、dtype、单位、顺序、坐标与动作起点 | 借鉴 Dexora 改为 36 维、猜测 unit、任意补齐/截断 |
| I08 | Raw action 是进入公共提交入口的最终绝对关节目标尝试，不是模型原始输出、ACK 或下一状态 | 未提交候选写成已尝试动作；WAIT 填上一目标 |
| I09 | arm/hand 非原子提交的事实保持；每个执行边界检查时变授权和期限 | 删除 arm 发布后、hand SDK 前的再次检查 |
| I10 | 首次终止原因锁定；最后一帧按实际到达的阶段保存；不重试已可能部分入队的行 | 多处 finish 改写原因、重复帧、为追求完整而捏造失败行 |
| I11 | 采集期间失败与采集结束后独立清理失败分离；停止未解决不得重新运动 | 所有 stop 异常都追溯污染此前 capture，或失败后清空锁存状态 |
| I12 | Raw I/O/编码/关闭/发布失败保留 staging；发布成功不等于任务成功 | 为减少错误分支把残缺文件发布为正常 Raw |
| I13 | Canonical 固定 13 字段；保持整段准入、episode 边界、局部间隔和累计网格 dt/4 检查 | 删除 WAIT/坏帧、重采样、改变公共格式或兼容迁移 |
| I14 | sync/async/rtc 的查询锚点、bootstrap、WAIT、迟到接管、RTC 冻结与过期规则不变 | 全部变成简单取模播放，或晚到后补发错过的动作 |
| I15 | 模型任务单一在途；reset/predict/close 串行；运行中的 native 调用不能被普通 Future 取消 | 超时即宣称模型已关闭，后台放任旧任务执行新动作 |
| I16 | 当前几何标定继续按实际消费者读取；Raw 相机内外参与 depth scale 保留 | 引入历史标定哈希/一致性门禁，或把缺可选外参改成禁止物理 Raw 采集 |
| I17 | 当前版本评估结果必须保存，Raw 录制可选 | 恢复 `--no-results` 或为 results 添加 Null 对象/大量可选分支 |

补充：既有 `joint` 控制为 arm7 + hand12；`eef` 控制为 xyz3 + rotation-6D6 + hand12。rotation-6D 保持第一列接第二列；Raw 相机外参 color→base，模型/Canonical camera_extrinsic 为其逆变换的 base→color `[R|t]`。动作实际布局以当前 PolicyInfo 为准，不靠模型名字猜测。触觉保留 SDK 原生消费值、有限期限内的整组三轴保持与原有效时间，不改成 Newton 或接触标签。

## 4. 目标数据流与责任分配

```text
入口 / Session
  解析声明配置 → 恢复 artifact 数值配方 → 绑定当前设备/标定 → 创建资源

SDK owner / Sensor IPC
  → ObservationRow（独立、稳定的物理消费快照）
  → 来源准入与绝对截止时间
      ├─ Policy 投影 → 历史窗口 → 纯特征构造 → PolicyRuntime
      │                                         ↓
      │                           物理动作 chunk → 调度 → ActionMapper
      │                                         ↓
      │                                   RobotCommand
      │                                         ↓
      │                                   公共提交尝试
      └─ 录制投影 + 最终尝试目标 / None → Recorder → RawWriter → Raw
```

- **Session**：资源装配与会话最终关闭；不重新解释已结束单轮的采集事实。
- **Episode/TeleopRunner**：当前轮次的观测消费、查询、控制拍、首次终止、pending 行和 capture 收尾。
- **特征构造与调度函数**：纯数值/时序决定；不连接设备、不结束 Recorder、不写结果文件。
- **Robot/SDK owner**：设备边界、最终机械限制、当前授权/期限、停止失败锁存；不认识模型内部结构。
- **Recorder/RawWriter**：有界写入、内容快照与发布；不判断 SDK 送达、不持有运动权限、不调用模型。

纯计算模块不接收整份 ExperimentConfig 或 Session 作为服务容器。需要几个标量或配置区段就直接传入；但也不为缩短签名建立一个包含全部资源的新 Context。

## 5. 校验与协议精简规则

### 5.1 删除校验必须给出证明

将检查分为三类：

1. **静态输入**：类型、字段、shape、dtype、单位、模式、参数范围。在用户输入、文件、SDK/IPC、模型公开接口等实际边界校验。
2. **变换结果**：浮点收窄、FK/IK、rotation 解码、点云生成等可能产生新错误；结果消费处继续检查。
3. **时变条件**：代际、stop、连接、反馈年龄、提交/执行期限；在实际副作用前重查。

删除内部重复校验前，确认：同一份数据、同一不变量、未被修改、没有新数值风险、所有实际调用入口均受覆盖。记录一条“删除位置 → 上游保障 → 反例检查”的证据。文档类型标注、只读 flag 或“通常只从这里调用”均不是充分证明。

独立设备工具、回放、键盘遥操作、HOME 可能绕过 Policy/ActionMapper。不能仅因主部署已检查，就删 driver 的机械边界检查。不能把 `ValueError`/`RuntimeError` 统统捕获并继续作为正常控制。

### 5.2 协议保持最小，但不丢信息

保留固定 sensor dtype、arm 目标槽/状态和 lifecycle 请求；不新增统一 Envelope、版本协商、远程发现或任意 payload。`run_id`、`query_id`、`operation_id`、`sequence`、`input_generation` 解决不同问题，默认保留。

`submit_until_ns`、`execute_until_ns`、反馈/协调健康期限以及 WAIT 限制不等价。可以集中计算与命名，不能合并成一个含义含混的 timeout。保存现有纳秒边界及开闭区间，不顺手统一 `>`、`>=` 或删除 `+1 ns`。

### 5.3 配置保持一套，分阶段确定事实

沿用 dataclass 和 CLI > YAML > 默认值。声明解析、运行装配、设备实际 profile 确认是三个时点，不是三套可变配置对象。

- 入口一次解析有效 action_steps、模式、控制 dt、观测需求与数值 recipe。
- 控制 dt、模型预处理与派生输入 recipe 继续来自训练产物；当前设备/物理标定来自 Real。
- 启动后确认实际 serial、分辨率与 depth scale 是新事实，不能以“只校验一次”为由删除。
- 不引入 Hydra/Pydantic/额外环境变量覆盖链，不恢复旧 YAML 兼容分支，不新增配置开关控制本次整理。
- 不借本轮改变任何控制率、限位、碰撞容差、queue/ring 容量、编码参数或线程预算。

## 6. 实施里程碑

### M0 — 固定可观察行为，建立最小离线基线

阅读优先级：`scripts/run_policy.py` → `deployment/run.py` → `session.py` → `episode.py`；然后五个连接点及其真正调用方。扩展到 teleop、replay、HOME 和独立设备工具只为确认接口使用者，不全仓无目的重写。

必读文件：

- `runtime/observation.py`、`deployment/observation.py`、`deployment/episode.py`。
- `robot/robot.py`、`robot/commands.py`、必要的 xArm executor/worker 与 XHand driver。
- `data/recorder.py`、`data/raw/{schema,reader,writer}.py`、`data/results.py`。
- `deployment/{inference,schedule,preflight,run,session}.py`、`config/experiment.py`。
- 当前 `scripts/check_deployment.py`，以及受影响的 teleop/replay 调用方。

用短表记录来源字段、有效期限、运行身份、命令形成/公共提交、Recorder 入队、首次终止与实际资源 owner。对这些点用假时钟、最小假设备/模型/写入器捕捉事件顺序；不得连接 SDK。

先跑现有可执行的离线检查并记录环境限制。不能把新增 mock 覆盖冒充真实 HDF5/视频集成、FK/IK 或硬件验证。

**退出条件**：明确哪些分支已有保护、哪些属于当前行为、哪些为真实缺陷；形成与第 8 节对应的可重复基线。分析记录留在临时目录。

### M1 — 收拢静态装配与配置，阻止依赖向热路径泄漏

主要文件：`config/experiment.py`、`deployment/{run,preflight,session}.py`、相关入口。

- 保留现有配置结构、未知字段拒绝、布尔类型严格性、CLI 覆盖与 print-config 行为。
- 只在入口决定有效值；下游不再次选择默认 action_steps，不恢复或猜测模型配方。
- 消费 PolicyInfo 的公开能力字段；不在 Real 增加基于模型类路径的分支或 registry。
- 训练产物的解析在本仓保持单一调用位置。当前公共 API 不能进一步减少外仓细节时，集中现有调用即可，不擅改 `dexmani_policy`；把更窄公共 resolver 的协同建议列为可选项。
- 把重复构造的静态关节限位数组、固定安装变换等放入已有对象初始化；不缓存 now、run 授权、source freshness。
- 保持 results 必存、Raw 可选；不为关闭录制引入 NullRecorder。

**退出条件**：每个有效参数来源可直接定位；非法配置/模式在设备连接前失败；已有 CLI 行为一致；新增生产依赖为零。

### M2 — 收拢来源准入与期限推导，保留两类需求及失败行语义

主要文件：`runtime/observation.py`、`deployment/episode.py`、`teleop/runner.py`、必要的 service/replay 调用方。

- 让来源可用性、新鲜度和绝对失效时刻使用同一份来源选择逻辑；减少 `observation_issues`、`feedback_deadline_ns`、`control_deadline_ns` 的平行解释。
- 优先改造现有函数。确需同时返回问题与 deadline 时使用简单结果，不新增动态 sensor registry 或逐源协议类。
- 控制需求与录制需求仍分开归因。获取 payload 可用 union，控制许可、录制完整性、模型决策年龄不能合并成一个健康布尔量。
- 固定来源的绝对期限可以复用；跨耗时操作后仍需用新的 now 比较，不能缓存“本行已校验成功”来代替发送时复核。
- 模型查询的决策年龄维持当前对最新查询行所需来源的规则；历史窗口中的老观测本来就是历史，不能把所有 Nobs 帧的最老时间直接用作当前决策期限。
- 同一源帧在允许年龄内被重复消费的既有行为保持，不引入强制跨模态同步、插值或强制新帧等待。
- 先确定 row 何时真正返回、pending_frame 何时存在。不要因为移动校验而无意增加/删除失败最后一行；确需修复证据丢失时独立列为正确性改动并补专门测试。

**退出条件**：时间边界反例、相机配帧、record-only 来源失败、跨发送边界过期均保持既定处理；没有新增数据补值或时间轴修正。

### M3 — 使模型特征构造与调度解耦，减少重复工作

主要文件：`deployment/observation.py`、`deployment/inference.py`、`deployment/session.py`；复用现有 FK/点云数值函数。

- 使用现有 `FIELD_DEPENDENCIES` 作为唯一依赖定义；在一次部署的装配时准备不变的 requested fields/dependencies，避免在投影、source_times、构造中每次重复推导。
- 移除 `build_observation_kinematics(..., execution=...)` 中通过 sync/async/rtc、A_exec、prefetch_steps 预测缓存必要性的耦合。缓存只围绕实际历史窗口身份去重、淘汰；容量受当前窗口约束，reset 时清空。
- 优先采用一种简单有界缓存路径；不同时保留旧启用公式、模式专用缓存和新缓存。若实际测量表明更简单的无缓存方案足够，须给出重叠窗口代价及取舍，不凭直觉删优化。
- 保留轻量 `PolicyObservationRow`。不要让模型历史无条件保留整个录制 RGB-D/触觉负载。
- SDK/IPC 取得独立稳定快照后，内部尽量复用；检查 view/base 生命周期。`writeable=False` 不是所有别名都不可写的证明。只在所有权明确时删复制。
- FK workspace 和模型仍由同一模型线程独占；运动学输出的 float32 有限性检查保留，真实数值算子不重复实现。
- 不改 RGB resize/crop、depth raw counts、point-cloud 通道、旋转布局、归一化位置、动作起点或辅助维度处理。

**退出条件**：特征模块不再依赖调度模式；相同输入/recipe 的字段、shape、dtype、数值结果一致；重复历史行的 FK 调用次数受控；缓存与历史内存有界且 episode 间清空。

### M4 — 线性化单轮控制与终止，缩小线程实现的暴露

主要文件：`deployment/{episode,session,inference,schedule}.py`，必要的 `runtime/state.py` 调用处；不重写底层状态机。

- 让一拍主线可直接读成：检查授权/预算 → 回收模型结果 → 消费观测 → 更新历史 → 接管或发起查询 → 映射并尝试提交 / WAIT → 记录 → 检查终止。
- 调度与映射返回决定或拒绝原因；不在纯函数里 stop、写文件或调用模型。
- 单轮结束归 episode 的唯一入口。同一次 `step()` 内发现必须结束的条件就撤销授权并收尾，不推迟到下一个控制拍。
- Session 管会话资源，Episode 管本轮事实，Worker 管自身模型任务与句柄。减少外部组合读取 `.future`、`.operation`、Recorder 内部 finish 字段的情况；如果一个有明确含义的只读状态足够，直接使用，不建立通用 WorkerManager。
- 取消、目标不可行、真正 IK 技术失败、SDK 故障、录制失败分别保留。内部异常不层层包装同一信息；SDK/线程/文件边界转换时保留原因链。
- 仅在保障立即撤销和必要清理的 owning boundary 保留宽捕获；不能为了消除 `BaseException` 字样而破坏 KeyboardInterrupt 收尾。
- 保留 pending_frame：形成候选不等于进入公共提交；公共提交已尝试后的失败必须保留最终目标；入队失败不能重试造成重复行。
- `finish()` 以首次终止为准，顺序保持：锁定事实并撤销 → 尝试停机 → 尝试写 pending → 完成 capture → 保存结果。一个清理失败不能跳过后续必要清理，也不能覆盖主因。
- 不合并 `stop_failure`、`hand_stop_pending`、`motion_active` 等不同事实。只有证明状态可由已有状态无歧义导出时才删除重复状态。
- 迟到的 reset/predict 结果必须回收，不能进入新 run；close 排在原模型线程任务之后。pending 不是 completed；线程超时不等于中断 native 调用。

**退出条件**：正常、取消、超时、旧查询、部分 SDK 失败与重复 finish 的事件序列可验证；既有调度模式与 arm/hand 发送顺序不变；没有新 BaseSession/状态机框架。

### M5 — 收紧 Recorder 与文件边界，独立补齐已确认语义漏检

主要文件：`data/recorder.py`、`data/raw/{schema,reader,writer}.py`、`data/canonical/{processing,export}.py`，以及 metadata 调用方。

**组织性重构：**

- Recorder 只接收普通录制元数据、稳定快照和最终命令；不让 data 模块主动理解 hand SDK 对象。将 hand reset metadata 的取得留给当前设备 owner，元数据拼装使用普通值。
- 触觉 reset metadata 必须在 capture 对应时点取得，不能为了只计算一次而跨 T 重置或下一段复用旧值。
- 保留一个有界 FIFO 和一个 writer；不要重新实现并发框架、无界缓存、异步事件总线或全链路零拷贝。
- 把 Recorder 的 START/finish/pending/最终路径解释留在 Recorder 内部。上层只需开始、提交、检查错误、结束和等待状态；避免多个模块拼接内部字段猜测是否已关闭。
- 首次 finish 请求保持唯一，编码/flush/close/publish 分别完成必要收尾。超时保留仍活动 writer 的所有权；明确 discard 才能删除本次 capture。
- 保留 accepted/written/video/timing 行数核对；I/O 故障与采集技术失败保持不同的发布规则。
- 离线导出继续复用共享数值内核与现有存储库，不新增通用序列化层；减掉确属同一未变化 block 的重复扫描，但保留类型转换后的结果检查和每段回滚。

**单独标注的正确性修复：**

- 重新核对当前 Reader 是否仍未验证 writer 已声明的关节单位、关节顺序、触觉单位/坐标轴/指序/处理语义。
- 若缺口仍存在，在对应文件消费准入集中校验。关节相关工具校验关节契约；只有消费触觉时才要求触觉语义，不能使无触觉依赖的回放无故强制读取全模态。
- 使用已有模型/字段常量；必要时仅共享少量真正共同的文件语义常量，不新增 registry、版本号、哈希或旧格式迁移。
- 正常当前 Raw 保持接受；被明确声明为错误单位/顺序的文件应拒绝。这是预期的准入行为修复，不是等价整理，须独立验证和报告。

**退出条件**：data 不依赖可执行 SDK 对象；失败前缀、零行、显式丢弃、队列满、写入/关闭/发布错误均符合契约；Raw/Canonical 格式不变。

### M6 — 有界测量、回归、自审与交付

- 用现有离线检查扩充必要覆盖，不引入 Ruff/pytest 或新测试框架；新增纯逻辑检查可使用标准库 unittest 和最小假对象。
- 在相同输入、相同环境下测量依赖推导次数、FK 调用次数、特征构造耗时、缓存/历史峰值和可测的重复分配。测量日志放临时目录。
- 不预设“提速 30%”之类目标。给出实际结果；收益不足的微优化可以撤回，避免增加维护负担。
- 不为性能 benchmark 跑真实 rollout、下载模型、连接相机、升级 SDK 或改变控制预算。
- 核对真实调用方：teleop、policy、replay、HOME、键盘/设备工具；更新受影响接口一次性调用，不保留无需求兼容别名或并行新旧实现。
- README 只更新真实入口/操作边界，不粘贴本任务书，不新增根目录审计报告。

**退出条件**：第 8–10 节的验收得到明确状态；本轮改动范围收敛，无安全语义偷换、无遗留重复路径、无未说明的外仓要求。

## 7. 暂不执行的事项

以下不是核心任务的阻塞条件：

- 重写 xArm IPC、替换共享内存、引入 ZMQ 或独立 XHand 服务；仅有明确测量或依赖冲突证据时另立任务。
- 更改 `dexmani_policy` 的公开 artifact resolver；本轮只提出必要的最小接口建议，不擅自修改外仓。
- 新增任务成功标注系统、完整动作/ACK 账本、逐 taxel mask、原始模型 chunk 持久化或常驻性能监控。
- 全量迁移目录、统一 Teleop/Policy/Replay 循环、改 SDK、变更控制算法、扩展硬件或模型支持。
- 以历史标定、文件摘要或旧版本兼容为由新增运行准入；未授权不重新写标定。

如某项真正阻塞核心工作，先说明具体生产调用与反例，不以抽象的“未来扩展”作为增加架构的理由。

## 8. 离线验收矩阵

所有场景都使用假时钟/假 SDK 或临时合成数据。不能用 mock 的通过声明硬件已安全，也不能只用修改后的实现验证自身而没有预期事件/数值断言。

| ID | 场景 | 必须断言 |
| --- | --- | --- |
| V01 | help / print-config / 核心模块导入 | 不创建 SDK 连接，不读取实时设备，不隐式加载 GPU 模型 |
| V02 | CLI/YAML/默认值、未知字段、字符串 false、非法模式 | 有效值与优先级正确，连接前拒绝非法输入，results 必存 |
| V03 | fresh/stale/future/缺失来源与纳秒边界 | 保持原有开闭区间；截止时间来自来源，不来自缓存消费时间 |
| V04 | 相机/点云序号不匹配、ring 覆盖、缓存复用 | 不混配 RGB-D，不刷新旧来源时间；匹配有效帧仍通过 |
| V05 | 仅控制需求正常，录制相机/触觉失败 | 归因保持录制失败，不静默继续无记录控制 |
| V06 | Nobs 含老历史帧，最新查询行有效 | 不错误把历史窗口整体最老时间当成当前决策年龄；已有窗口约定不变 |
| V07 | SDK/IPC 数组、模型投影、Recorder 快照 | 持有期间数值稳定，不能由生产者覆盖；不无故保留录制专用大数组 |
| V08 | FK 窗口重叠、清空与不同调度模式 | 同一行特征一致、去重调用数正确、缓存受当前窗口约束；无模式依赖 |
| V09 | 动作维度/非有限/限位，直接 driver 或工具入口 | 非法数值与布局拒绝；不补零；独立入口仍有机械边界保护 |
| V10 | 取消发生在映射前、提交前、arm 发布后、SDK 返回后 | 每一阶段是否存在命令尝试及 stop 的事件顺序正确，旧代际不能恢复运动 |
| V11 | arm 已发布、hand 失败 | 记录最终尝试，不把两个设备解释成原子成功/失败；停止失败锁存不丢 |
| V12 | WAIT、IK 拒绝、query 超时与 late result | NaN/无行/最终尝试分别符合到达阶段；不得伪造成功目标 |
| V13 | sync bootstrap、async 迟到尾段、RTC 冻结/handoff | 预测索引、接管时刻、期限、无追赶发送与现有行为一致 |
| V14 | 首次/重复 finish、采集后 stop 失败 | 原因不被覆盖，pending 不重复；采集结果与清理结果正确分离 |
| V15 | writer 队列满、写入/编码/flush/close/publish 失败 | 不重复行、不静默丢帧宣称正常；失败证据/staging 与未关闭所有权保留 |
| V16 | 零行、START 失败、显式 discard、收尾超时 | 依据现有契约保存或删除；不删输入；pending 不能伪装完成 |
| V17 | 模型单在途、reset/predict/close 顺序、结束后的结果/异常 | 迟到任务只回收不执行；关闭串行；异常可见；未完成如实报告 |
| V18 | Raw 单位/关节顺序/触觉语义篡改 | 当前合法文件通过；对应消费者拒绝不符文件；无格式迁移或补值 |
| V19 | Canonical 正常/NaN 动作/额外 RGB 帧/时钟偏差 | 13 字段、shape/dtype、episode 边界、dt/4、整段拒绝与回滚保持 |
| V20 | 不同 capture 间触觉 reset、无外参物理录制 | 元数据不过期；可选相机外参不被误升为 Raw 录制必需项 |

资源清理测试要覆盖“一个 close 抛错后其他资源仍获得关闭尝试”。SDK stub 需在任何真实设备构造之前接入；直接 import 参考 forwarder 不能用于测试。

对数值对比：未改变运算顺序的路径优先逐元素相等；必要浮点容差应说明来源。不要通过大幅放宽容差或删除断言让重构通过。保存/回放目标顺序、单位、NaN 位置和事件次序应严格一致。

## 9. 执行命令与限制

在已有实验 Python 环境、仓库根目录运行。先读当前脚本确认其离线属性；依赖缺失时不擅自安装或升级。

```bash
git status --short
git diff --check
python -m compileall -q dexmani_real scripts
python scripts/collect_teleop.py --print-config
python scripts/run_policy.py --print-config
python scripts/check_deployment.py
```

受影响的纯逻辑检查补在现有离线入口，或仅为独立边界新增最小 `scripts/check_*.py`。避免一个几百行 fake 框架服务少数断言。HDF5/PyAV/Zarr 集成检查只能使用临时合成文件；真实 Raw 只读且默认不需要访问。

检查分阶段运行，修改后重跑受影响项；同一状态已通过且无新风险时不无休止重复全套。`git diff --check` 要区分用户预存问题与本次引入的问题，不顺手格式化整个仓库。

若 Pinocchio、模型环境或某依赖缺失，明确列出：无法执行的命令、原因、已完成的替代纯逻辑检查和仍未验证的范围。不能写成“全量通过”，也不能用可选依赖缺失为由停止全部可执行工作。

禁止真实 xArm/XHand/RealSense/VR/HTS 连接、HOME、teleop、replay、rollout、在线诊断、标定写入；禁止运行参考仓库入口；禁止下载权重或更换运行环境来掩盖依赖问题。

## 10. 最终交付格式与停止条件

请用中文交付，至少包含：

1. **完成范围**：M0–M6 各自为已完成/已由现状满足/有证据阻塞，列出主要文件和实际接口变化。
2. **精简收益**：删除了哪些重复推导、无关依赖、状态解释和包装层；不是只给新增类清单或 LOC 百分比。
3. **校验证据**：每类删除/合并的校验由何处覆盖、独立入口如何保障；新增的 Raw 语义修复单列。
4. **行为保持**：I01–I17 及关键 V 场景结果，注明未覆盖项和真机未验证。
5. **验证与测量**：实际执行命令、成功/失败、真实测量条件与结果；静态推导、mock、数值集成和现场验证明确区分。
6. **diff 自审**：没有不相关改动、旧兼容别名、双路径、额外依赖、权限变更或真实数据改写。
7. **剩余事项**：只列具体证据支持的外仓协同或真机验收，不把未执行工作写为完成。

若某阶段需要改变研究语义才能继续，保留已验证的独立改动，报告具体冲突，不用一个更大框架绕过。工作结束时留下可审阅的本地 diff，不自动提交或推送。

### 完成定义

- 五个连接点的数据流和终止权清楚，纯特征构造不依赖调度模式。
- 配置、静态字段依赖、来源期限不在多个层次被重复解释。
- 设备 owner、最后执行检查、在途模型任务回收和 Recorder 发布边界未被削弱。
- 没有新增生产依赖、通用运行框架、历史格式兼容体系或控制补偿。
- 必要行为回归有实际断言；性能只报告测得结果；未执行的集成与硬件验证明确披露。

## 11. 已审读的固定参考源码

只阅读与当前阶段有关的文件。链接失效或网络不可用不阻塞本仓重构；使用本文明确的取舍，不凭名字臆造参考设计。不得执行或 import 外部入口。默认重新实现小范围职责安排，而不是复制代码；确需复制片段时先检查许可证和归属要求。

### ManiUniCon

- [MU1 · main.py：集中装配](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)
- [MU2 · TorchModelPolicy：观测、模型、动作适配](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py)
- [MU3 · SharedStorage：ring、queue、数据模型与共享状态](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/utils/shared_memory/shared_storage.py)

### Dexora

- [DX1 · DexoraPolicy：薄策略包装与输入输出布局](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_policy.py)
- [DX2 · deploy/README：进程环境、forwarder 与协议](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/README.md)
- [DX3 · inference：gather_observation / execute_action / chunk loop](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py)
- [DX4 · XHand forwarder：必须排除的导入副作用、补值、节流路径](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/xhand_forwarder.py)
- [DX5 · dataprocess/README：库职责与离线转换分离](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/dataprocess/README.md)

### DexMani Real 的直接落点

- [DR1 · runtime/observation.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/dexmani_real/runtime/observation.py)
- [DR2 · deployment/observation.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/dexmani_real/deployment/observation.py)
- [DR3 · deployment/episode.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/dexmani_real/deployment/episode.py)
- [DR4 · robot/robot.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/dexmani_real/robot/robot.py)
- [DR5 · data/recorder.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/dexmani_real/data/recorder.py)
- [DR6 · scripts/check_deployment.py](https://github.com/haoyangzhanglab/dexmani_real/blob/c7936ca97f26b0f3f879f03193755b5e4b650cbf/scripts/check_deployment.py)

## 12. 可直接粘贴给 Codex CLI 的配套 Prompt

```text
请在当前 dexmani_real 工作区实际实施 .codex/tasks/dexmani-real-simplification.md 的减法重构，不要只审计、复述或另写一份方案。

先读取 AGENTS.md、适用子目录规则和完整任务书，检查实际 HEAD、git status 与已有 diff。任务书的提交号是参考基线，不是要求回退的版本。保留用户所有既有修改；不要 reset、自动 stash、全仓格式化或覆盖冲突。

目标是：校验在正确边界集中，配置与字段依赖一次解析，模型特征与调度解耦，episode 的终止/最后一帧/录制收尾清晰，资源与在途任务各有唯一 owner。借鉴 ManiUniCon 的适配边界和 Dexora 的薄接口、依赖隔离与直观主循环，不移植它们的框架、全局存储、ZMQ、补零、动作清洗、节流或回退机制。

按 M0–M6 连续推进：先建立最小行为基线，再做可独立审阅的小步修改，每步运行相关离线检查并自审。优先复用现有函数和类型，证明重复后再删，不为缩短文件新增包装层、registry、BaseSession、Context 容器、协议版本或生产依赖。已由当前代码满足的项给出证据后跳过。

必须保持 I01–I17；尤其是来源时间、点云配帧、run/query 身份、SDK 前动态授权和期限、非原子 arm/hand 提交、Raw 最终提交尝试语义、WAIT/失败前缀、采集结果与结束后清理结果分离、sync/async/rtc、单在途推理和关闭 pending。Reader 的语义漏检如仍存在，作为独立正确性修复验证，不冒充纯重构。

禁止连接任何真实设备、HOME、teleop、replay、rollout、标定写入、执行参考仓库脚本、下载权重或升级实验环境。只修改本仓，dexmani_policy 只读；不要修改 AGENTS.md、Codex 权限或 .codex/config.toml。不用 Ruff/pytest。阶段记录和测量日志放临时目录。

不要每一步都等待确认；普通可逆选择直接实施。遇到依赖缺失或必须新增授权的事项，明确证据和阻塞范围，继续完成其他可执行部分，不伪造验证通过。默认不 commit、不 push、不创建 PR。

结束前检查 diff，按任务书第 10 节用中文报告实际完成范围、主要删除/解耦、校验覆盖依据、行为保持、真实执行的命令与测量结果、未验证项及剩余阻塞。留下可审阅的本地 diff，并明确真机未验证。现在开始。
```
