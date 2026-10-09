# Codex 任务书：DexMani Real 最小研究执行链路

状态：待实施。本文定义后续代码修改与验收要求；新增本文的提交不代表这些修改已经完成。

设计基线：`main@40aa2b2dd8cdabaf09ea530832682e73206bd526`（`1009 temp`）。2026-10-09 复核主分支、源码、四个参考项目及前轮 fact-check 后形成。执行时先检查 HEAD 和相关 diff；已经修复的项目只验证，不重复改写。遵守 [AGENTS.md](AGENTS.md)，保留无关修改。

## 1. 目标与最终选择

交付一条个人 PhD 可以读懂、调试和复现实验的链路：遥操作采集 → Raw → canonical 训练数据 → Policy → 真实机器人执行与评估。优先修正科学语义和控制边界，再删除没有当前研究价值的复杂度。

采用**现有架构内的局部收敛**：一个设备控制线程、一个串行推理 worker、一个有界录制 writer，传感器维持必要的进程隔离。无需重写仓库、引入执行框架，或迁移到某个参考项目。

本任务作出以下设计决定，实施者无需再次把它们列为悬而未决的问题：

| 问题 | 本轮采用的方案 | 实验含义与代价 |
|---|---|---|
| 遥操作生命周期 | 明确区分 IDLE、PREPARING、RUNNING、PAUSED；停止结束本轮，暂停允许开始新段 | 消除 S/D 后残留 active；不合并底层运动许可状态 |
| 训练时间 | 固定网格采集；导出同时检查逐拍间隔和累计偏离；不补帧、不重采样 | 严格准入会拒绝部分现有数据，不能用标称 Hz 掩盖真实时间 |
| 训练 action | 定义为本行观测对应、双设备 SDK 均接受的绝对目标 | ACCEPTED 不代表物理到位；CRC 未确认、拒绝、未知目标仍保存在 Raw，但不进入公共训练导出 |
| 策略 WAIT | 健康的正常推理等待有预算；缺必要观测、计划失效或错过执行期限时结束本轮 | 删除异常自动恢复语义；会改变 episode 终止分布，需在实验记录中说明 |
| tare 取消 | 候选在局部构造，取消不替换当前有效基线；完成后按通道发布结果 | 无需持久化校准事务或版本系统；设备/物理状态变化仍要求重新确认基线 |
| dense tare | 准确表达“载荷/残差有限”，不宣称已验证稳定性或无接触 | 不凭空设一个没有测量依据的 dense 阈值 |
| policy rollout | 默认是评估 evidence，不承诺直接用于再训练 | 保留 bootstrap/WAIT 的无动作行；不为通过导出而填动作或裁掉这些行 |
| 可选研究能力 | sync 为清晰主路径；本轮保留 async/RTC、DexPilot、EtherCAT | 有配置可达性不等于无用。删除需满足第 8 节证据条件，不阻塞必修工作 |

范围包括 `teleop/`、`recording/`、`dataset/`、`deployment/`、`robot/`、`replay/` 的上述问题。模型、训练算法和通用 Dataset 继续属于 `dexmani_policy`。不增加历史数据迁移、多机器人抽象、插件注册、自动恢复、额外平滑、第二套动作队列、后台守护服务或分布式追踪。

## 2. 对照参考项目：只迁移适合当前系统的机制

以下是本任务实际核对的固定版本，不声称是这些项目永远的最新版。

| 项目与源码证据 | 借鉴 | 本仓库的取舍 |
|---|---|---|
| [LeFranX：设备组合](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L210-L229)、[部署清理](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_robot_deploy_dp.py#L472-L494) | 薄设备组合、直接的采样/推理/动作循环 | 其所谓同步分支实际顺序发送。本仓库继续记录 arm/hand 分别的下发结果，不承诺原子执行 |
| [lerobot_robot_ufactory：Mode 6](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/robots/uf_robot/uf_robot.py#L283-L288)、[异步保存](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/scripts/_uf_lerobot/async_saver.py#L215-L267) | xArm 关节控制可使用 Mode 6；耗时落盘离开控制路径 | 保留当前小型有界 writer；无需引入其生态适配、私有 API patch 或另一种 Raw 格式 |
| [ManiUniCon：动作时间与过期分支](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/act_wrapper/chunk_wrapper.py#L37-L70) | 明确区分动作索引、计划时间和当前时刻 | 不移植“全部过期后把最后动作重标到下一拍”的兜底；不新增队列/插值层 |
| [PF-DAG：局部观测/action deque](https://github.com/XiaohanLei/PF-DAG/blob/b877129754267c3d9c564abcb19960bcf215e12f/pf_dag/agents/policy_agent.py#L149-L203)、[Mode 1](https://github.com/XiaohanLei/PF-DAG/blob/b877129754267c3d9c564abcb19960bcf215e12f/pf_dag/robots/xarm_robot_wo_gripper.py#L183-L193) | 紧凑、易追踪的论文实验入口与局部历史 | 不复制首帧重复补历史、硬件模式及现场常量。该版本算出 `agg_chunk`，实际消费 `action_chunk`，不能按注释推断 ensemble 已启用 |

因此，简化的主要收益来自减少重复状态、模糊的异常恢复和信息压缩，而不是删除设备边界、数据证据或研究算法。

## 3. 保留的最小架构与数值契约

### 3.1 责任边界

| 边界 | 唯一责任 | 必须保留的约束 |
|---|---|---|
| camera / VR / pointcloud worker → 共享 ring | 发布最新物理观测 | 不积压旧帧；RGB 与 cloud 需要一致时使用同一 camera sequence |
| 主线程 → `Robot` → arm/hand SDK | 读取设备、实现目标、下发、停止、HOME | 普通 SDK 操作保持单一 owner；键盘回调仅撤权/置请求，不并发调用 SDK |
| `ObservationRow` → controller/policy → recorder | 同一份控制前观测用于计算与记录 | 行拥有独立且不可变的载荷，writer 不能读到被下一拍改写的数组 |
| 主线程 ↔ `InferenceWorker` | 一个在途模型操作；本地有限观测历史 | 模型 worker 不访问设备；正常操作不积压 future 或观测队列 |
| 主线程 → recorder writer | 非阻塞提交行，异步编码与发布 | 保留当前有界队列；满队列即报错并结束采集，不静默丢行 |
| Raw → exporter → canonical | 按明确数值语义派生完整训练缓存 | Raw 发布后不可变；不完整 episode 整段拒绝；不在 reader 内禁止读取失败证据 |

保持现有目录职责，不为这张表创建六个新接口/基类。`ObservationRow`、Policy 输入字典与 Raw 数组处在不同消费边界，不能仅因表达相似就强行合成一个通用 schema。

### 3.2 不得随重构改变的实验语义

- joint action 是绝对目标，19 维：arm 7 rad + hand 12 rad；EEF action 是 21 维：base 中 xyz 3 m + rotation-6D 6 + hand 12 rad。不得改成增量、度或换坐标系。
- Policy 的 `N=n_obs_steps`、`H=horizon`、`A=n_action_steps`、`P=H-N+1` 与输出形状保持一致；历史来自真实相邻控制行，不用首帧复制填满。
- bootstrap 的 action 头从实际 handoff 开始消费；steady async/RTC 仍按 query 锚点跳过已经过去的前缀。两种锚点不同有原因，不能机械统一。
- RTC 冻结的是已有物理目标，消费时仍需当前反馈、碰撞/目标限制及运动许可检查。不要在此重写 RTC 算法。
- `SafetyState`、`run_id`、ESC 锁存、第一结束原因继续有效。应用层 PAUSED 与设备 ARMED 不是同一状态。
- `Robot._send` 在任一 SDK 前预检两个目标；每个执行器调用前重查 run_id、停止/退出和 deadline；Mode 6 恢复后再检查一次。arm 接受后 hand 拒绝是合法的部分下发证据，不回写为全成功或全未发送。
- 保留动作工作区/关节投影、IK、跳变检查、目标自碰撞和 driver 机械限位；它们约束不同层次。目标无碰撞不构成实际扫掠路径无碰撞的证明。
- arm 在线仍用当前 Mode 6 路径；HOME 使用现有规划、收敛判据与模式恢复。停止保持当前测量姿态/被动停止的既有逻辑，不能改成再次发送旧策略目标。
- Raw 保存所有已接入物理模态、缺测和 dispatch。canonical 仍是全部 13 字段、整段有限；模型按需加载。触觉为 SDK 原生数值且已扣偏置，不能宣称为 N 或二次扣偏置。
- camera 几何随对应 Raw 保存；table/mount/VR 使用当前实验状态的既有规则。不新增跨实验标定兼容矩阵。

## 4. 已核实问题与优先级

P1 表示首先处理的运动授权/训练正确性问题；P2 表示随后必须修复的生命周期与结果问题。没有真机伤害证据，因此不把这些结论夸大成已证实的 P0 物理事故。

| ID | 优先级 | 已确认事实与可追溯源码 | 对应实施包 |
|---|---|---|---|
| B01 | P1，必修 | 同批 S→C：停止批次未过滤恢复，随后可再次 `begin_motion`。[事件过滤](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/teleop/runner.py#L463-L481)、[恢复](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/teleop/runner.py#L244-L275) | A |
| B02 | P1，必修 | Raw 仅检查时间递增，导出却赋固定 dt。合成 `[0, 0.5, 1.0]` 在 16 Hz 下通过已有数值准入，产生 8 倍时间尺度反例。[schema](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/recording/storage/schema.py#L42-L54)、[固定 dt](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/dataset/export.py#L99-L115) | B |
| B03 | P2，必修；与 B01 一起做 | S/D 结束 capture 后仍 active，B 被忽略；无 capture 的 Q 仍要求确认。[分支](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/teleop/runner.py#L186-L241) | A |
| B04 | P2，必修 | 推理未结束时 close 仅挂回调；Python 退出已开始后回调再 submit(close) 被拒绝。真实子进程的假模型复现 close 未执行。[源码](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/deployment/inference.py#L57-L99) | D |
| B05 | P2，必修 | HOME 的 `HomeResult` 被压成 bool；回放已完成后 H→Q 取消被记为 fault。[压缩结果](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/robot/arm_homing.py#L218-L260)、[错误分类](https://github.com/haoyangzhanglab/dexmani_real/blob/40aa2b2dd8cdabaf09ea530832682e73206bd526/dexmani_real/replay/session.py#L119-L147) | D |

B01/B03 是同一组状态处理缺陷的两个症状，不重复计算成两个独立安全漏洞。B02 的 8 倍为构造反例，不是设备实测偏差。B04 不代表永久 GPU 泄漏；B05 未证明回放数据丢失。

D01（异常 WAIT）、D02（tare 取消）、D03（训练标签）、D04（dense verified）、D05（rollout 再训练）是原实现的契约/用途选择，本任务已在第 1 节明确方案；不要将这些选择倒写成原实现无条件错误。

## 5. 实施包 A：停止优先与遥操作状态收敛

优先级 P1。修复 B01/B03，并落实 S01。主要修改 [teleop/runner.py](dexmani_real/teleop/runner.py)；只在必要时调整 [operator_input.py](dexmani_real/runtime/operator_input.py)。

用一个本地小枚举表达 IDLE / PREPARING / RUNNING / PAUSED，替代可矛盾的 `active/paused/resume_requested` 组合。保留必要的 `quit_pending`、准备完成后的新鲜时间边界和运动 run_id；不引入状态机库，不把录制器状态复制成第二套布尔状态。明确转换应集中在少量方法里。

| 当前状态/事件 | 目标行为 |
|---|---|
| IDLE + B | 准备新 capture/reference；只有准备结束后的新鲜 robot/VR 观测且许可仍有效，才进入 RUNNING |
| RUNNING + C | 先撤权并 stop，清参考，进入 PAUSED；capture 保留供 S 保存/D 丢弃，不追加暂停期间的帧 |
| PAUSED + 新一批 C | 保存上一段，准备新 episode；重新建立 reference 与时间网格后才恢复 |
| S | 撤权、stop、保存已有 capture，进入 IDLE，清准备/退出确认标志；后续新 B 可以开始 |
| D | 撤权、stop、明确丢弃当前未发布 capture，进入 IDLE；绝不删除已发布 Raw |
| 达到录制时长/不可继续的控制失败 | 保存真实前缀和结束原因，结束当前轮次；不得留在伪 active 状态 |
| H | 先撤权、stop、保存并结束当前段，再执行 HOME；完成后 IDLE，下一次 B 才开始采集 |
| IDLE + T | 仅在 ARMED、无 capture/准备且操作者确认无接触时 tare；不产生运动或自动开始 |
| RUNNING/PAUSED/PREPARING + Q | 立即撤权停止，取消准备，进入 PAUSED 与退出确认；再次 Q 保存退出。下一批 C 可明确取消确认并准备新段 |
| IDLE、无 capture + Q | 直接退出 |
| PREPARING + 重复 B/C | 忽略重复开始；S/D/Q/ESC 仍能取消准备 |
| 任意状态 + ESC / 硬件故障 | 走既有急停/故障边界，不允许 B/C/H 恢复许可 |

同一批键盘事件中，终止事件必须压过所有可能重新授予运动的事件，包括 C 的恢复。覆盖 S→C、C→S、S→B、Q→C、D→B 等排列；Q→C 的取消确认只允许来自**后续新输入批次**。S 与 D 同批优先保存，避免混合按键造成丢弃；ESC 不被普通结束原因覆盖。无需为了这个规则重写键盘监听架构。

不含 recorder 的调试会话同样清理本地状态。所有保存/编码可能阻塞的工作，必须发生在撤权和停止请求之后。停止失败不能把界面标为可重新开始的正常 IDLE，而应走原有故障退出。准备期失败、writer 失败、STOP 与 SDK 返回交错时，保留第一结束原因及附加清理错误。

验收：用真实 runner 方法配假 clock/robot/recorder 注入事件，证明终止批次中 `begin_motion` 次数为 0；S/D 后新 B 可开始；暂停期间无新行；恢复后的行属于新 episode；IDLE 的 Q 不需二次确认。显式检查 ESC 仍锁存、旧 run_id 始终不可下发。

## 6. 实施包 B：采集时间与训练标签正确性

优先级 P1。修复 B02，落实 D03/D05。修改 [teleop/runner.py](dexmani_real/teleop/runner.py)、[control/controller.py](dexmani_real/teleop/control/controller.py)、[storage/schema.py](dexmani_real/recording/storage/schema.py)、[dataset/processing.py](dexmani_real/dataset/processing.py)、[dataset/export.py](dexmani_real/dataset/export.py)，必要时修改 recorder 元数据；不要为了导出门槛让 Raw reader 拒绝读取失败数据。

### 6.1 一个显式节拍，不做时间修复

1. 控制时间使用整数 monotonic ns。首个实际控制采样作为本段 `t0`，以后目标采样时刻为 `t0 + k * dt_ns`；准备、HOME、暂停不占训练 tick。删除用每次 `now + dt` 推迟整条时间轴的调度。
2. 当前任务采用单一初始准入预算 `epsilon = dt / 4`（16 Hz 时为 15.625 ms）。这是**本次设计选择**，不是既有测量结论、硬实时保证或最优噪声阈值。将规则放在一个小的纯函数/常量定义处，采集与导出共享；本轮不增加分模式宽松阈值或兼容开关。部署已有决策预算不因此被改成这个值。
3. 取样前检查漏拍；取样后检查本次 `ObservationRow.observation_timestamp_ns`，不能只在阻塞的 `robot.read_state()` 之前检查。正常采样应在该拍 `[due, due + epsilon]` 内。
4. 采样抖动预算与动作计算预算分开：控制计算和下发可使用本拍剩余时间。将下一拍起点 `t0 + (k + 1) * dt_ns` 作为该动作的 exclusive deadline，与已有 feedback deadline 取最小值，传给当前 SDK 前置检查。不要把整个 IK/SDK 链路挤进采样用的 `dt/4`；也不得仅因反馈尚未过期就在下一拍发送上一拍目标。第一拍同样受下一拍起点约束。
5. 晚于预算或漏拍时，结束本轮、撤权、stop、保存前缀与原因；不赶帧、不重标 t0、不连续补发。已实际获得的异常行/下发结果按原值保留；未下发不能填上一次动作。尚未获得完整行则记终止原因和已知时间，不伪造 ObservationRow。
6. SDK 已进入后无法靠 Python deadline 撤回。若返回时已到达或超过下一拍起点，保留真实 ACCEPTED/部分下发状态并结束本轮；不要把实际已经接受的动作改成 NOT_CALLED。这是软件可检查的时序边界，不是设备执行完成时间。

`utils/rate.py` 的服务循环允许漏拍后重新锚定，不能直接拿它充当训练逻辑时钟。无需新建通用 Scheduler；runner 中的锚点、索引和纯时间检查已经足够。

### 6.2 导出准入必须检查实际 Raw 时间

保留现有有限、零起点、严格递增检查。在训练准入处增加：

```text
dt = 1 / reader.control_hz
epsilon = dt / 4
对所有 i > 0：abs((t[i] - t[i-1]) - dt) <= epsilon
对所有 i >= 0：abs(t[i] - i * dt) <= epsilon
```

纳秒/浮点边界采用一致的转换与仅为表示误差服务的小容差，不能暗中再放宽一拍。两个条件都要检查：只看相邻间隔会放过长期累计漂移，只看总时长会放过中间异常。单行不能证明采样频率，应明确报告“无间隔可检验”，不宣称通过了实测节拍验证；下游可继续按自身历史长度要求决定是否加载。

新 Raw 可用现有 meta 记录实际使用的时间预算；canonical 保存本次准入规则/预算作为导出 provenance。预算不升级为模型 ABI。已有 Raw 直接按真实 timestamp 复核，不重写文件，也不因为缺新增的说明属性就建立 migration 分支。不能从含坏行的已发布 Raw 中悄悄截取训练前缀。

导出拒绝信息至少包含 episode、首个违规行、实际/标称间隔或累计偏离。保留现有整段拒绝与拒绝日志；不要加一个任务数据库。若真实工作负载无法满足初始预算，记录读取、retarget/IK、下发的实测耗时，按实验需要调整单点规则或重选采集频率并重采/重建训练缓存，不能只放宽导出端。

### 6.3 action 的意义和准入

Raw 继续保存 attempted target 与真实状态码：0 NOT_CALLED、1 ACCEPTED、2 CRC_UNCONFIRMED、3 REJECTED、4 UNKNOWN。不要为了训练删除失败行或修改状态。

公共 canonical 的定义是 **SDK 接受的绝对目标**。在现有 13 字段完整/有限要求之外，逐行要求 `dispatch_status == [ACCEPTED, ACCEPTED]`；任一行不满足即整段拒绝，指出行号和两端状态。CRC_UNCONFIRMED 不自动升级成功；它是否适合其他研究数据集是后续研究选择，本任务不增加第二种导出模式。

任务没完成但观测、时间、下发均有效的 episode 不因此自动排除；技术有效性与任务成功是不同标签。SDK 接受也不保证目标已到达，不能把训练 action 改写为 measured qpos。

Raw rollout 的 bootstrap/WAIT 行继续保留无动作证据。公共导出按上述通用规则拒绝不合格 episode，并给出清楚原因；无需只因 `collection_source=rollout` 就硬编码拒绝所有数据。README 明确目前 rollout 主要用于评估，若未来做再训练需另行定义派生片段和时间映射。

验收：正常固定网格及预算内抖动通过；`[0,.5,1]@16Hz`、累计漂移、局部漏拍、重复/倒序/非有限时间被拒；采样/IK 延迟不能导致追赶补发；状态 0/2/3/4 任意一列出现都拒绝 canonical，但 Raw 仍可读且内容不变。使用同一合法 episode 验证规则不误杀普通的任务失败样本。

## 7. 实施包 C/D/E：推理、退出结果与触觉

### C. 异常终止，正常等待；顺便缩小决策元数据

优先级 P1（行为选择 D01），其后做局部优化 S04。主要修改 [deployment/runner.py](dexmani_real/deployment/runner.py) 和 [deployment/observation.py](dexmani_real/deployment/observation.py)。

当前 `_invalidate()` 同时承担正常换 chunk、异常观测、过期结果和自动重新查询。将这些调用点按实际原因分清；允许用两个短的局部方法表达“清当前计划进入正常等待”和“结束本轮”，不要增加 WAIT 原因策略框架。

| 条件 | 行为 |
|---|---|
| bootstrap 正在收集真实历史/计算；sync 正常 chunk 间隙；async 在尚未到达 handoff 前等结果 | 必要观测保持健康且 tick 按期时，允许现有预算内 WAIT。持续轮询停止输入；不重复下发旧目标 |
| 必要观测缺失或过期；错过控制 slot / handoff；decision 到期；动作实现被拒而无可执行目标 | 立即结束本轮：撤权、stop、清历史/计划，保存真实行与结束原因；禁止同一 run_id 自动恢复 |
| WAIT 超时 | 结束本轮并记录 timeout；重复 invalidate/提交新 query 不得重置超时起点来延长预算 |
| 操作者 S/Q/ESC、硬件/模型/录制故障 | 保留现有原因分类与清理优先级；停止动作先于输出 finalization |

正常 WAIT 中不发送新目标，也不承诺硬件已经静止；Mode 6 中上一已接受目标可能继续执行。不得把每次正常推理间隙都变成紧急 stop 或额外 HOME，也不新增保持目标控制器。

异常之后，已有 in-flight future 只允许被回收，其结果不能下发。清除遗留 start request；新的 B 必须来自异常边界后的输入，并满足既有 HOME/新鲜度/模型空闲检查。若已达到 `num_episodes` 或会话退出条件则退出；不擅自增加自动 HOME 或后台重启。

在行为正确后，将 Query/Plan 反复携带的 `sources: dict[str, tuple[int,...]]` 收敛为 `decision_valid_until_ns`：

- 只对当前实现已有的最新历史行 source clocks 做等价替换，不偷偷把整个历史窗口都纳入新年龄条件。
- 构造 Query 时必须验证每个所需源时间 `0 < stamp <= now`，来源集合非空，且当前尚未过期；先验证，再求最小值。单取最小值会掩盖未来时间戳，属于错误简化。
- 当前全部源共用 `max_decision_age_s`，故截止时间等于 `min(stamp + age_ns + 1)`。交接、消费与每端发送统一采用 exclusive deadline；与当前反馈、slot、总预算继续取最小值。
- `query_id`、`run_id`、query slot、handoff slot 与 bootstrap 标记保留。需要的诊断在 query 建立处记录，不让大字典穿过所有执行层；不建立额外遥测系统。

验收采用 fake clock + 可控 future，覆盖 sync/async/RTC 与 N=1/2：正常 bootstrap/换段不调用 stop；异常观测、过期决策、missed slot/handoff 会撤权和 stop；随后观测恢复也不自动重启；旧 future 不能跨 run_id 消费；新 B 仍必须满足现有起点约束。deadline 边界、未来/零时间戳、空来源与旧实现做等价对照。真实模型/GPU 与真机结果另行报告，不能由这些离线检查推断。

### D. 修复关闭和 HOME 结果，保留资源所有权边界

优先级 P2，B04/B05 必修。两项可以独立提交，不依赖全部 C 重构结束。

**模型关闭**：修改 [deployment/inference.py](dexmani_real/deployment/inference.py)。调用 `close()` 时立刻禁止新工作，并在同一个单线程 executor 队尾提交一次 cleanup；cleanup 执行时再检查 model 是否存在，涵盖仍在 load 的情况。不能等 predict 的 done callback 再 submit。随后允许 `shutdown(wait=False, cancel_futures=False)` 排空已入队任务；回调只收集结果/错误，不发起新工作。

关闭幂等；模型的 load/predict/reset/close 仍在同一个 owner worker 串行执行。活动任务与关闭任务若需分别保留 future，用两个明确字段即可，不建立任务调度层。未成功建模时安全空操作；warmup 失败但已建模时仍清理。close 失败可观察；既有推理错误不得被 cleanup 覆盖。设备撤权/停止不等待 CUDA 完成。

验收必须包含真实 Python 子进程退出：假模型 predict 尚未完成时请求 close，主线程立即结束，退出后检查 close 标记恰好一次且无“interpreter shutdown 后 submit”错误。补充仍在 load、factory/warmup/predict 抛错、重复 close、close 自身抛错。不能用 sleep 等 predict 先结束作为唯一通过证据；不承诺不可取消 CUDA 的有界进程退出时间。

**HOME 结果**：直接复用 [robot/home.py](dexmani_real/robot/home.py) 中 `HomeResult`，让 [arm_homing.py](dexmani_real/robot/arm_homing.py) 的 `home_robot` 返回完整结果。更新 teleop runner、[keyboard_session.py](dexmani_real/teleop/keyboard_session.py)、[deployment/operator.py](dexmani_real/deployment/operator.py)、[replay/session.py](dexmani_real/replay/session.py) 全部调用者，显式检查 `.ok/.interrupted/.reason`；不增加 bool 兼容包装或依靠 dataclass 默认 truthiness。

正常 Q/S 取消表示 interrupted；ESC 仍是急停；超时、无路径、硬件/清理错误仍失败。回放已经 completed 后取消 return-home，不改写已完成轨迹结果；会话记录 HOME 被取消及退出原因。真实 HOME/stop/close 失败也不能因轨迹已完成而掩盖。保留必要的 hand→arm 顺序、finally、未完成运动的停止重试。

验收覆盖 hand 与 arm 各自取消、正常完成、拒绝进入 HOME、超时、真实异常和 stop 清理失败；完整复现 completed replay → H → Q，应保留已完成轨迹、正常取消 HOME 与准确的会话状态。

### E. tare 发布规则与准确命名

优先级 P2，落实 D02/D04。修改 [robot/drivers/xhand.py](dexmani_real/robot/drivers/xhand.py) 及两处操作提示。

- 在局部变量中采集候选、检查候选；移除函数一进入就清空已发布基线的操作。
- CancelledError 在提交前离开，不替换已有基线；原来没有基线就仍无基线。取消检查继续放在每次可能阻塞的 SDK 读取前后和提交前。
- 正常完成后按通道提交：通过的候选成为新基线，失败通道设为 None 并明确报告；不能保留旧值却声称本次归零通过。
- 非取消的技术异常不伪装成功：失效基线并报告错误。重连/实验设置变化不借本规则恢复跨会话旧基线。无需新建快照、回滚日志或校准 registry。
- aggregate 保留当前残差判据；dense 目前仅证明所需载荷与残差有限。调整方法/日志/提示中的 `verified` 含义，明确二者不是同样的稳定性证明，且稳定接触也可能得到零残差。操作员仍须确认无接触。
- 保留 aggregate/dense 独立有效性、Raw NaN 和不阻断非触觉关节任务的既有语义。没有真实 dense 无接触样本前不猜阈值，也不新增 tare 自动 HOME。

验收：有/无旧基线时取消、采样中取消、验证后提交前取消，均无半提交；完成后一成功一失败按通道更新；技术异常可见；人工构造的大有限 dense 残差不会被报告为“稳定性已验证”。这是明确新契约，不回头宣称所有历史取消行为都是 bug。

## 8. 删除与简化：风险收益决定范围

删除项不能与必修 bug 捆绑成“必须全删”。本轮确定做 S01–S04；其余按下表决定，不以文件数或总代码行数作为成功标准。

| ID / 候选 | 收益 | 风险 | 本轮决定与删除条件 |
|---|---|---|---|
| S01 本地状态集中 | 高：消除矛盾组合与重复结束分支 | 中：按键/录制边界变化 | 随 A 实施；先固定事件表 |
| S02 回调内再排 close | 中高：关闭时序更直接 | 低中：future/错误回收 | 随 D 删除二次调度；模型 owner 不变 |
| S03 HomeResult → bool | 中：少包装且保留原因 | 中：漏改调用者会把失败对象视为真 | 随 D 删除压缩层，搜索全部调用点 |
| S04 Query/Plan sources 字典传播 | 中：一个截止时间代替重复遍历 | 中：漏验未来时间、改变历史年龄或 off-by-one | 随 C 后半段实施，等价检查先行 |
| S05 eye_in_hand 配置/变换分支 | 低中：缩小相机支持面 | 中：离线通用转换可能仍有人使用 | 本轮保留；确认论文与外部入口只支持固定相机后成套删除，不能仅凭 pointcloud worker 拒绝该模式就判死代码 |
| S06 async/RTC | 高：减少预取、handoff、冻结前缀逻辑 | 高：删除真实研究能力、改变延迟/动作语义 | 本轮保留；只有论文、配置、评估都不使用且 sync 满足任务时，才能成套删除入口、参数、分支；不拆成插件框架 |
| S07 DexPilot 后端 | 高：去掉专用 adapter/依赖 | 中高：失去可达 retargeting 对照 | 本轮保留；TAG 被明确确认为唯一研究后端，且依赖无其他使用者时删除；共享几何不能顺手删 |
| S08 EtherCAT | 中：缩小设备连接面 | 高：可能使另一套现场硬件无法使用 | 本轮保留；设备清单与所有实验脚本均只用 serial 后才删；串口 CRC 和停止确认不能一起删 |
| S09 table hand-frame proxy | 中：统一桌面表示 | 高：没有标定 mesh 时失去 HOME 后备保护 | 本轮保留；所有 HOME 入口都要求有效桌面几何、缺失时明确拒绝，才可移除 proxy 和重复配置 |
| S10 IK fast/base/null 搜索 | 潜在中高：减少分支/计算 | 高：求解率和关节连续性退化 | 暂缓；先有代表性目标的命中率、时延、拒绝率、连续性消融 |
| S11 direct/proximal/distal/RRT HOME | 潜在中：减少规划分支 | 高：某些起点无法安全回家 | 暂缓；先测支持姿态范围、候选命中率、耗时及路径有效性 |
| S12 点云过滤/采样阶段 | 潜在中：减少输入处理耗时 | 高：直接改变训练/部署数值分布 | 暂缓；同一 Raw + 同一模型比较输入与任务指标，数值配方成套更新 |
| S13 合并多层清理 | 表面中，实际未证明 | 高：遗漏启动一半、writer 失败、异常退出资源 | 撤回广泛合并建议；保留 episode/runner/session 各自责任，仅处理 B04 等具体缺陷 |
| S14 合并短类型/纯函数文件 | 低：少几次跳转 | 低中：导入环、可选依赖提前加载 | 不做；只在实际修改中出现明显重复时局部整理 |

确认“当前未使用”应检查有效配置、入口、论文实验需求和依赖使用者，不能只看默认值或 `rg` 的单个结果。删除条件缺证据时保留该功能并简短记录原因，继续完成 A–E，不因此暂停整个任务。

必须继续保留的复杂度包括：run_id 栅栏；每设备 dispatch 与停止结果；不可变 ObservationRow；有界 writer 与失败 staging；传感器进程的启动/退出监督；真实观测历史与 chunk 锚点；按 VR sequence 缓存成功/失败的 retarget 结果；HOME 的路径校验与最终收敛。尤其不能删除 retarget cache 后让同一 VR 帧反复推进有状态求解器。

## 9. 执行顺序、验证与完成条件

### 9.1 建议提交划分

| 顺序 | 提交内容 | 完成证据 |
|---|---|---|
| 1 | A：停止批次与本地状态，B01/B03 | 事件表正反例、跨准备/保存边界的旧 run_id 拒绝 |
| 2 | B：采集网格与导出时间准入，B02 | 时间反例、慢读/慢 IK、无追赶与真实状态保留 |
| 3 | B：ACCEPTED 标签准入与 rollout 说明，D03/D05 | 全部 dispatch 状态矩阵、Raw 不变、正常失败任务样本 |
| 4 | C：异常结束与正常 WAIT，D01；随后 S04 | 模式/历史长度矩阵、无自动恢复、deadline 等价 |
| 5 | D：模型关闭，B04 | 未结束 load/predict 的进程退出探针与幂等关闭 |
| 6 | D：HomeResult 贯通，B05 | 全部调用者 + replay HOME 取消/真故障区分 |
| 7 | E：tare 发布与准确语义，D02/D04 | 取消/部分通道/异常检查；README 操作提示同步 |

这些是审阅边界，不要求同时推进或大规模重构。B04/B05 可提前独立完成；不能因可选裁剪或等待真机统计而把五个确定 bug 留下。

### 9.2 离线检查

遵守仓库约束：不运行或引入 Ruff/pytest，不安装或升级真实实验环境依赖。使用标准库断言、fake clock、fake SDK、可控 future 和子进程，优先调用实际生产方法；不能只复制实现做一套自证逻辑。确需保存回归检查时，放入一个紧凑、无硬件副作用的普通 Python 检查脚本即可，不建立新测试框架。

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

对本次控制/数据改动保留上述有区分力的回归用例，不为文件改名和简单 wrapper 编造覆盖率指标。若已有依赖可用，再用小型合成 HDF5/RGB-D/视频 fixture 验证 Raw → canonical 的接受/拒绝与输出一致性；没有 h5py/视频/Zarr/模型依赖时明确未测，不能把数组级准入检查写成完整导出通过。

### 9.3 真机与论文结果边界

本文不授权连接设备、运动或写标定；离线完成后列出待操作者授权的最小检查，不建立生产验收平台：

- 记录读取、retarget/IK、每端 SDK 调用耗时及真实节拍分布，判断 `dt/4` 初始预算是否适合当前实验。
- 检查 S/ESC、过期命令、必要观测中断、SDK 失败和退出时的真实停止行为；Python 检查点不能给阻塞 SDK 提供硬超时保证。
- 在代表性起点验证 HOME 与当前安装/桌面几何；目标自碰撞检查和标称资产不能替代实际运动验证。
- 同一 Raw 上核查时间准入、同步偏差、触觉基线以及压缩 RGB 与在线输入差异。当前各源主要是 host 接收时钟，不能声称设备级同步已经证明。
- 使用真实 checkpoint 核查 sync，并仅对论文实际使用的 async/RTC 做必要运行；记录新异常终止规则对 episode 长度/成功率统计的影响。

这些未验证项不能被自动记成新增 bug，也不能被宣称已解决。尤其不以当前速度/停止重试预算作为经实测认证的安全上限。

### 9.4 不应继续“修复”的旧猜测

以下说法已在前轮 fact-check 排除或限缩，执行时不要由它们扩张任务：

- 未证明 C 能绕过 ESC；B01 是 STOP 批次问题，ESC 锁存应保留。
- 不能仅因外层 inference_mode 就认定 RTC 梯度必坏：[Policy 内部显式退出 inference_mode 并启用梯度](https://github.com/haoyangzhanglab/dexmani_policy/blob/184126352cc707b999c3eb005fb78841a6e62cd1/dexmani_policy/agents/action_decoders/rtc.py#L42-L49)。这也不是完整 RTC 正确性证明。
- 项目要求 Zarr 2；[2.18.7 的 codecs 确实导入 get_codec](https://github.com/zarr-developers/zarr-python/blob/v2.18.7/zarr/codecs.py#L1-L4)，不要因错误 API 猜测升级到 Zarr 3。
- nominal-rate replay 是已声明的行为，不能与 B02 的训练时间准入混为一谈。
- bootstrap/steady 的不同锚点不自动构成 off-by-one；[Policy 输出切片](https://github.com/haoyangzhanglab/dexmani_policy/blob/184126352cc707b999c3eb005fb78841a6e62cd1/dexmani_policy/deployment/runtime.py#L239-L255) 必须与 Real 一起核查。

### 9.5 完成定义

- [ ] B01–B05 均有修复、失败反例与通过对照，不以文档解释代替修复。
- [ ] D01–D05 按本任务的明确选择实施；停止、数据筛选与 tare 的行为变化已说明。
- [ ] S01–S04 完成且没有放宽运动许可/新鲜度；其余候选有“保留/有证据后删除”的准确结论。
- [ ] Raw 无修改、无静默丢帧/补值；训练 action、时间与失败证据边界一致。
- [ ] 所有 HOME 调用者显式消费结果，模型 close 在请求时就已排队。
- [ ] README 只更新操作者需要的工作流、准入和限制；不复制本任务书成为第二份架构规格。
- [ ] 最终 diff 只含必要修改；没有新增框架、无用兼容层、隐藏自动恢复或额外依赖。
- [ ] 交付说明列出已执行检查、未执行的端到端/模型/真机验证及原因。只有真机实际验证后才能报告真机通过。

实现允许选择更小的等价局部写法；若发现设计与最新源码事实冲突，给出具体路径、反例和修订理由，先完成不受影响的必修项。不要通过放宽数据/安全契约让验收“通过”。
