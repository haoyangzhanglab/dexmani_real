# Codex 任务书：DexMani Real 正确性修复与简化

日期：2026-10-30（北京时间）。目标：让 xArm7 + XHand 的个人论文实验代码正确、好用、简洁、高效、容易维护。

本任务书承接四轮审查和 57 项逐项 fact-check。**57 项是审查记录，不是 57 个 bug，也不是 57 个必须修改的任务。** 原分类为 P1 1 项、P2 功能/条件缺陷 13 项、P2 能力或数据策略 7 项、P3 28 项、保留/撤回/待测 8 项。下文为每项作出明确处理决定；“保留”也是经过判断的最终方案。

## 1. 执行范围与基线

| 仓库 | 核验版本 | 用途 |
| --- | --- | --- |
| [dexmani_real](https://github.com/haoyangzhanglab/dexmani_real) | `f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4` | 本任务的主仓；本任务书写入前 main 的源码基线 |
| [dexmani_policy](https://github.com/haoyangzhanglab/dexmani_policy) | 审查 `049fc1046bb2678ac839b5ce1cb47600e93ffe17`；设计时 main `661e8aae4ba4addb9494399bcf9e11eb66ad345b` | 仅处理部署元信息、warmup 调用约定、数据身份和 resume 所需配套修改 |

已核对 policy 的上述两提交差异：只新增 `CODEX_TASK_INTERACT_DEX_A1.txt`，没有源码变化。不要覆盖这份已有任务书或顺带实现它的研究方法。若其他工作已改动公共接口，应与当前实际代码合并，不另建第二套协议。

执行前先读两个仓库各自的 `AGENTS.md`、查看工作区修改和当前提交。只复核受本批次影响的调用链；已修项目记录为已解决，不重复改、不回退到审查快照。固定提交是证据，不是要求强制 checkout 的目标。

本任务允许为解决主仓真实依赖问题修改配套 policy 源码，不允许重构训练平台、改模型方法、修改实验数据或自动启动真机。若执行环境只有主仓，先尝试取得配套源码；确实不可用时给出可应用的最小配套 patch 和明确阻塞原因，继续其他独立工作，不以主仓兼容层掩盖依赖缺口。

## 2. 架构与删减原则

保留现有目录和职责：`robot/sensor` 持有设备，`geometry/perception/control` 做数值转换，`runtime` 提供已有协调机制，`teleop/deployment/replay` 保留各自业务流程，`data` 记录和导出。**不新增统一 Session、设备框架、动作事务层、schema registry、配置平台或全仓生命周期状态机。**

三条数据流要直接可追踪：

1. 采集：设备来源帧 → 有来源时间的观测 → 显式动作映射 → 已有执行边界 → Raw 实测状态与提交目标。
2. 训练：Raw → 当前固定采样规则下的 Canonical → 训练 artifact 中的实际数值配方 → 公开 `PolicyInfo`。
3. 部署：真实观测历史 → `warmup/predict` → 动作映射 → sync/async/RTC 现有调度 → 机械臂 owner 最终检查与 SDK 调用。

每项新增状态、参数、校验、异常捕获或抽象，都要能指向本任务中的实际错误或真实边界。优先修改原函数；只有真实共享规则才抽取短函数。不能用代码行数目标驱动删减，也不能将“更健壮”“以后可能需要”作为新增机制的理由。

| 位置 | 应保留的检查 | 可删除或合并的内容 |
| --- | --- | --- |
| YAML、SDK、文件、artifact 入口 | 类型、形状、单位、必要元信息、返回码、来源时效、资源释放 | 同一入口已建立且未变化的不变量，不在每层重验 |
| 跨进程执行、副作用发生前 | prepare/run 身份、命令期限、反馈时效、控制器状态、撤权 | 不因父端检查存在而删除 owner 最终检查；也不机械删除父端对 hand 副作用之前的提前拒绝 |
| 冻结后的静态命令 | 入口已证明的静态几何事实可复用 | 确认输入相同后消除重复 collision/FK；动态分支、期限与撤权仍检查 |
| 数据加工 | 用到的输入范围、真实数值语义、训练 split 和数据身份 | 不要求无关视频尾部通过训练导出；不新增全量内容哈希/多版本迁移框架 |
| 资源关闭 | 保存主异常、报告关闭异常、如实表示 pending | 不将线程超时写成已终止；不为了强退默认改 daemon 或多进程 |

## 3. 对照参考仓库

参考用于挑选清晰的实现方式，不作为正确性背书；它们的硬件、数据需求和失败处理不同。具体证据及采用/不采用的决定如下。

### 固定版本

| 仓库 | 审查版本 |
|---|---|
| ManiUniCon | `85c6f2e32ecf9f2bed62d202b058c39623444686` |
| LeFranX | `a39906e6629f39490950fe8bd20f4f992ed74fd7` |
| Dexora | `6f2869d61483774aa75cda4cd2260dc6a5faa783` |
| Pi-r2-flow | `3af52ca400a6ec7d141416879aa531a6f62f697a` |

### ManiUniCon

1. **借鉴：命令处理和硬件调用的责任顺序清楚。** [`Robot.run` L345–365](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py#L345) 的主干是取命令 → 边界处理 → 硬件接口校验 → 下发。对本项目的启发是保留可追踪的 `policy/VR → action mapping → robot.submit → arm owner/hand SDK`，在**实际副作用之前**保留必要检查，而不是将所有检查删除或复制到每一层。关联 `DEP-06`、`CTRL-10`、`GEO-04`、`GEO-05`。它并不证明父进程 admission 与 owner 最终状态检查是同一个边界。
2. **借鉴：低层轨迹推进由小函数承担。** [`init_ruckig` / `update_ruckig` L7–51](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/utils/ruckig_utils.py#L7) 将当前状态、目标、Ruckig 推进直接写出；命令长时间中断时改为保持当前参考。对应本项目应原地修复 `CTRL-01/02/03`，保留现有 arm owner 与 Ruckig，不另造 TrajectoryManager。不要复制其 `2.5*dt`、墙钟计时或硬编码限速；本项目已有实际时限语义。
3. **不照搬：静默保留越界前的 joint target。** [`_clip_action_to_bounds` L159–170](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py#L159) 在 clipping 后 IK 失败时打印警告并保留原关节。`dexmani_real` 不能为“简单”而将不可行命令继续执行；`CTRL-04/05` 应修正绝对分支评分与 workspace 定义，不新增静默 fallback。

### LeFranX

1. **借鉴：组合机器人是一层直接组合。** [`FrankaFERXHand.get_observation` L155–190](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L155) 清楚读取 arm、hand、camera 并汇总；[`send_action` L192–250](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L192) 分解 arm/hand 后下发。采用职责清晰的组合思想，继续使用本项目既有 `RobotSystem`、`Observation` 和 typed command；不引入 LeRobot plugin/registry，也不为 19 维固定机器人改成通用前缀字典框架。关联 `DEP-07`、`CTRL-07`。
2. **值得保留的现实边界：所谓 synchronize_actions 仍是顺序调用。** 同一 [`send_action` L213–237](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L213) 明确说明目前 sequential；异常时请求停止两侧并向上抛。`CTRL-07` 的正确目标是诚实的“尝试联合下发 + 部分失败立即进入现有停止路径”，而非新增两阶段提交、ACK、回滚协议或声称臂手同时到达。
3. **借鉴独立收尾，不复制吞错或继续动作。** [`disconnect` L259–287](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L259) 对 arm/hand 分别收尾，说明资源关闭不能因一侧错误完全跳过另一侧。对应 `CTRL-06` 应区分未启用的对象与已经产生运动的设备，并保留主失败原因。不要复制 [`replay_episode` L173–176](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_robot_replay.py#L173) 的发动作失败后继续下一条；`TEL-03` 修复起点定义不等于放宽执行错误处理。

### Dexora

1. **借鉴：推理适配只有一个清晰的公开输入输出。** [`DexoraPolicy.get_action` L143–183](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_policy.py#L143) 接收 state/images/instruction，返回 `[chunk_size, state_dim]`，编码器与模型加载留在 policy 对象内；语言 embedding 缓存位于 [`L134–137`](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_policy.py#L134)。`DEP-01` 应把本项目实际使用的 metadata、observation、warmup 统一到一个公开接口；不在 real 中识别多种旧接口、动态 getattr 或维护版本适配器。
2. **借鉴：同步循环应该能够一眼追踪。** [`gather_observation` / `execute_action` L234–270](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py#L234) 显式完成单位转换、state 拼接、BGR→RGB 和 action 分解；[`model_inference_loop` L316–348](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_inference_zmq.py#L316) 清楚展示 observe → chunk inference → 取当前动作 → send → sleep。`DEP-02/03` 原地修正 sync 时钟和 bootstrap 顺序，保持同步基线容易读；不为与 async/RTC 对称而新增一套通用调度器，也不删除研究已用到的 async/RTC。
3. **不照搬：隐式改模型输入和不受限的同步停顿。** [`get_action` L162–167](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_policy.py#L162) 会自动裁剪/补零 state；这不适合本项目的 q19/action19/EEF21 训练部署语义。单次 inference 内阻塞也不证明执行时限充分。保留边界尺寸/单位/动作模式检查与失效停止，删除的是内部已成立事实的重复校验。`execute_action` 手→臂的顺序调用同时佐证 `CTRL-07` 不应假设硬件事务性。

### Pi-r2-flow

1. **直接可借鉴：先填观测历史，再发首次推理。** [`main` L258–319](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L258) 先按 `video_T/state_T` 填 deques，之后 stack history 并查询首个 chunk。`DEP-03` 用这一顺序解决长 history 吞掉 inference 预算；`DEP-01` 的 warmup 应使用真实同形 observation。采用阶段顺序，不复制它的固定 30 Hz、无界等待、首次预测动作自动 ramp 或硬件依赖。
2. **直接可借鉴：动作语义在一个函数里显式转换。** [`to_absolute` L104–114](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/policy/control_utils.py#L104) 明确区分 arm/hand 的 absolute 与 delta。保留本项目现有 action codec 和训练时的 delta anchor，不增加通用 ActionCodec registry，也不能把“按实时 state 加 delta”搬到训练使用 observation anchor 的 chunk 上。`CTRL-04` 的机械关节绝对分支距离与旋转等价距离同样需要明确区分。其 [`apply_ema` L156–164](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/policy/control_utils.py#L156) 首帧透传是自己的选择，不能解决本项目要求从测得手势连续启动的 `TEL-02`。
3. **可借鉴：把观测锚点随推理结果传递。** [`_do_query` L576–629](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L576) 从快照取得 `t_obs`，测 query 时长并返回 `(chunk, t_obs)`；[`EpisodeWriter.add_step` L54–58](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/recording/dataset.py#L54) 分开 step 与 image timestamp。`DEP-08`/`GEO-02` 可用少量已有或新增 monotonic 时间字段区分 frame receipt、observation、submit 和 apply 的延迟，不做 telemetry 平台，不把 source freshness 与模型推理时间混为一个阈值。
4. **明确不照搬其时间、fallback 和恢复策略。** [`CameraClient.get` L163–175](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/cameras/realsense.py#L163) 在客户端收包时打时间；[`_RobustCamera.read` L94–101](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/cameras/realsense.py#L94) 读取失败复用旧帧。这些恰好不能证明帧新鲜度。`GEO-02` 应让接收时间与帧一同通过本地队列，避免出队时把旧帧标新。`_do_query` 异常返回 None、[`main` L968–980](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L968) 迟到时反复保持上个 action、[`XArmSDK.connect` L51–80](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/robots/xarm_sdk.py#L51) 自动清错/重试，也不应用于本项目的停止与故障语义。


对照只核实源码和固定版本，未运行四仓设备。参考项目也未覆盖的能力不能因此被视为安全。


## 4. 执行顺序与交付单位

按下面批次执行；一次提交围绕一个能独立验收的行为，允许把大批次拆为少量关联提交，不要求一问题一提交。无需每批重新向用户申请继续。

| 批次 | 结果 | 依赖 |
| --- | --- | --- |
| A1 | 贯通数据配方、训练 artifact、公开部署 API、真实观测 warmup；补数据身份 | 优先，主仓与 policy 配套 |
| A2 | 修正 HOME 时间、空路径停稳、部分启动清理 | 可与 A1 独立 |
| A3 | 修正 jog、IK、workspace、teleop 取消、滤波初始化与 replay 起点 | A2 的共享 HOME 规则 |
| A4 | 修正 sync/首段历史时序；补最少的目录竞争与终止原因记录 | A1 |
| A5 | 导出仅处理实际保留数据，消除重复扫描，明确现有数据策略 | A1 的 metadata/source_id |
| A6 | 修正相机接收时间边界和局部标定/诊断流程 | 独立 |
| A7 | 修复可选 CordVIP 的确定问题，保持与主链隔离 | 独立；不安装模型来完成无关主链任务 |
| A8 | 有证据才做的性能与微型清理 | 正确性批次完成后 |
| A9 | 记录必须依靠设备/网格/性能测量才能决定的能力边界 | 无条件实施其余批次，不等待设备 |

实施 A1–A7 中明确的修改；A8 只落地满足本书条件的改动；A9 没有证据时保留当前实现并准确报告。不得把“待测”悄悄转成新增框架，也不得把整个任务停在计划阶段。

### A1 — 贯通数据到部署的唯一事实来源

关联：DEP-01、DATA-09、DATA-07、DATA-08。主入口：`data/export.py`、`deployment/preflight.py`、`deployment/inference.py`、`deployment/session.py`；配套 policy 的 `training/common/data.py`、`training/common/config.py`、`deployment/metadata.py`、`deployment/runtime.py`、`datasets/common/split.py`。

- 沿 Canonical attrs → dataset 描述 → 保存的训练配置/artifact → `inspect_policy()` 返回值逐字段追踪。当前不仅 `PolicyInfo` 缺属性，训练配置组装也没有完整保存现有数据 attrs。把部署真正使用的 `control_dt_s`、手历史、所需 depth/点云/手点云配方、指尖名称及 tactile 来源按实际模态保存并公开。字段含义与单位唯一，不复制整套运行时配置；policy 不导入真机包。
- 这些新增要求只约束真实使用它们的路径：保留非 Canonical/仿真训练和纯 policy inspect 能力，不能要求所有策略都必须有相机/URDF/Real 控制周期。真实部署所缺的事实在 Real 入口明确拒绝，不猜默认值。来源约定：`control_dt_s ← Canonical.dt`；`depth_config.shape ← 实际训练 depth.shape[1:]`；`depth_config.depth_scale_m ← depth_scale_m_per_unit`；点云配方/指尖名来自对应 attrs；tactile 来源来自已验证训练配置；手动作历史来自适用模型实际 action context，不能把所有 encoder 的 `history_steps` 都认成手动作历史。
- 移除 Real 对 policy 私有 `saved_config` 的依赖。公开 `PolicyInfo` 只暴露实际消费者需要的字段；tactile 来源使用显式公共字段。保留 `warmup(observation, ...)`，Real 按它传入真实结构的观测，不添加 `rgb_hw` 别名、`getattr` 默认值或两版本 fallback。
- 将模型加载与 warmup 分开：模型加载可提前开始；设备连通、真实观测历史就绪后，使用现有 `ObservationBuilder` 构造与 predict 一致的输入，在运动执行授权前 warmup。同步沿用取消、数据时效与 worker pending 的现有机制；不复制第二个观测构造器。模型热身不应消耗一次实验的动作等待预算或写入实验 Raw。
- `reset/warmup/predict` 在同一现有模型 owner 串行，保留当前随机状态和缓存 reset 语义。需要 `action_context` 的策略（例如 LAMP）按已有训练首帧填充约定，从实测手姿构造相应上下文；不能只提供 observation。RTC delay/cap 只传给已支持的路径，不借修复扩大算法支持矩阵。
- 每次新导出完成的 Canonical store 获得一个 `source_id`（一次 UUID 即可），原样复制/迁移保留，重新加工新建 ID。沿用 policy 既有 split/source 身份检查；Canonical resume 任一端缺 ID 都明确拒绝，包括同路径，路径不能证明内容未变。正常从头训练、读取/评价旧数据仍可进行，但需要以后可靠 resume 的新实验应先显式重新导出；不隐式改写用户数据或旧 checkpoint，不影响其他数据格式自己的身份约定。不自动猜测或重算旧 normalizer，也不建设历史数据迁移工具。
- 无 ID 的拒绝放在 **checkpoint resume 的数据装配边界**。通用 `validate_split_manifest` 还用于指定 split 的从头训练/读取旧数据，不能无条件塞入“所有调用都必须有 ID”而误伤这些路径。
- 将 `_restore_cloud_config` 的“键集合等于整个 dataclass”改为直接读取当前实际使用的数值字段：PointCloud 的四项、HandCloud 的点数，交给现有配置类型校验。已用数值缺失仍明确失败，无关 provenance 附加键不导致重训；不加兼容投影层、默认值补齐或版本迁移框架。新增真正影响数值的参数时同步修改保存/读取。不要把任意元信息缺失一律写成“必须重训”：旧 artifact 能否恢复取决于是否有可信训练来源，不能读当前 YAML 或数据路径冒充历史训练事实。
- 复现性采用固定可配套运行的代码版本、现有完整 URDF 与固定 hand 模板。记录最低限度版本/资产来源，说明修改模板或加工算法应重新导出；不对所有常量做配置化、哈希、资产注册或运行时 Git 检查。

验收：最小真实 artifact 经过 `inspect_policy → resolve_deployment_config → load_policy → warmup → predict`；可用轻量实际公共类型加 fake 模型隔离 GPU，但不得用自造同名 `PolicyInfo` 来掩盖接口不兼容。覆盖 joint/EEF 及现有实际支持模态的代表输入，并至少覆盖一个需要 hand_history/action_context 的策略。验证同 ID 搬迁可 resume、新 ID 同名同长度数据被拒绝、双缺 ID 不再被认成已验证同源；从头训练指定 split 和旧数据读取/评价仍可用。验证训练输出中必要 attrs 能到达部署端，不止测试 dataclass 字段存在。缺真实权重/GPU时标明未完成的端到端验证。

### A2 — HOME 与关闭只保留一套清楚的规则

关联：CTRL-01、CTRL-02、CTRL-06；保留 CTRL-07、CTRL-10。位置：`runtime/home.py`、`robot/xarm7/worker.py`、`robot/system.py`、`robot/xhand/driver.py`。

- 共用一个短的 HOME 段时间估算，使用实际最大关节行程 `d`、速度 `v`、加速度 `a`。静止到静止估计：`d <= v²/a` 时 `2*sqrt(d/a)`，否则 `d/v + v/a`；再加既有运动余量、停稳和查询/调度余量。父级总预算覆盖各段预算之和、模式切换及 worker 在 HOME 前的 `wait_stationary`（现有收敛上限 15 s）；空路径的额外停稳等待也计入，不能还沿用更短的父端 `+7/min10`。该估计不是控制器硬实时保证，不为它增加大量可调超时。
- 空路径不走 `dwell + 单个反馈周期` 的紧预算，复用既有 HOME 收敛等待上限。保持“查询等待不算已停稳”“超期限返回的反馈不能确认完成”的现有语义。
- 部分启动 cleanup 按已经存在的连接/执行事实判断：从未成功建立、从未允许运动的 hand 不应仅因对象存在就尝试 stop 并锁存 unresolved motion；曾执行后断联仍保留失败，不能靠 `is_connected == false` 直接当安全成功。不新增生命周期状态机，不吞掉原始启动错误。
- 保留 arm/hand 顺序提交，明确返回值表示发布/下发尝试，不表示两设备原子执行或到位。部分失败按已有外层停止路径尽快撤权并保留主异常；不新增两阶段提交、执行 ACK 流或同步设备平台。

验收：低加速度 `a=0.01 rad/s²,d=0.1 rad` 的预算覆盖理想运动约 6.325 s 及停稳；空路径 100 Hz、查询 15 ms、dwell 300 ms 不因漏算首尾查询失败；部分 connect 失败清理不制造虚假 unresolved motion；运行后断联仍失败。用 fake clock/driver 调原函数，不需真机。

### A3 — 控制目标与操作者意图一致

关联：CTRL-03/04/05、TEL-01/02/03/04/05/06；TEL-08/09/10/11 保留。

- jog 在限位夹紧后“目标未变化”应保持最后已接受的目标，不走松键制动/清空参考；区分无输入、真正 IK 拒绝、饱和后仍有输入。手姿态等几何输入改变时仍须现有端点准入，不能用缓存目标跳过变化后的约束。松键制动和拒绝后松开全部键才能恢复的行为保留。
- IK 已选好合法绝对关节分支后，cost/连续性比较用绝对 `q_candidate-q_current` / `q_candidate-q_previous`。只在选择等价候选时做周期化，不在评分时再次取模；不改合法分支枚举。
- HOME 和在线端点使用同一个配置中的 workspace margin（现为 5 mm），删除 HOME 独立 1 mm 容差语义。仍拒绝超出共同边界的姿态；不额外实现未有需求的越界逃逸规划器。
- S/Q 已记录为本轮正常结束后，异步撤权导致的取消按 OPERATOR/QUIT 结束，保留已经完成的录制行，未完成行不补写。真正 SDK、超时或 ESTOP 故障仍保留为失败，不能把“用户恰好也按了 Q”作为吞掉同时故障的条件。
- DexPilot reset 后将实测手 q 同时作为优化前解与滤波状态；可用现有 LPFilter 的 first-next 行为播种一次，不访问臆测私有字段、不另造滤波器。
- replay 起点只要求接近录制首帧的**实测状态**；默认手准备阶段也以首帧实测状态为目标，之后按录制目标序列播放。保留准备时长参数及默认值，同步更新 help/打印说明。新的准备目标须通过现有手目标限位检查，不能静默 clip 后宣称已对齐；不因此要求整条实测轨迹都在目标操作限位内。不同时要求接近首个 action；第一条目标仍按记录及已有执行约束处理，不能宣称同时保留了旧流程提前平滑到首 action 的效果。
- 标定 HOME/阻塞操作结束后丢弃期间积压的采样/求解等一次性操作，保留 Q/Esc 和实时 held-key 状态。不要为键盘事件加 epoch/事务框架。
- 将 `runtime/keyboard.py` 有副作用的 `healthy` 改为名称明确的 `check_health()` 并更新实际调用点；保留原检测时机、急停锁存和停止信号，沿现有主循环/owner 链路收尾，不宣称物理立即停止。若能小范围消除异常分类重复，复用已有 runtime 模块中的短分类函数；只共用真实设备故障映射，保留各入口对未知业务异常和 HOME 失败的不同策略。不要建统一 ErrorManager。

验收：连续向限位 jog 不出现 `hold→重新从滞后实测累加`；3.1 到 -3.1 的选定绝对分支按约 6.2 rad 评分；同一 workspace 边缘姿态在线/HOME 结论一致；S/Q 发生在 compute/commit 间仍正常结束而真实错误不被覆盖；首帧滤波 `.2→.8, alpha=.1` 输出约 `.26`；首 state=0°、action=40° 不再因双 ±10° 入口条件无解；阻塞结束不会自动执行之前的一次性采样。

### A4 — 部署时序与结果含义

关联：DEP-02/03/04/05/08；DEP-06 放 A8，DEP-07/09 保留。

- sync 在收到新预测结果时明确开始下一个完整控制周期，或明确等到 tick 再消费；选择与现有循环改动最小的一种。映射/提交不应只得到旧 tick 剩余的随机相位预算。不能刷新传感器来源时间或延长实验总截止期来“修复”超时。
- 每轮首次历史采集与已提交预测的等待分别计时，复用现有循环即可：历史准备可用 `(n_obs_steps-1)*dt + ACTION_TIMEOUT` 的有限预算，首次实际提交预测时再设 `now + ACTION_TIMEOUT`。沿用全局实验截止期并在说明中明确其包含本轮历史准备；不要通过移动总计时起点暗中加长实验。A1 的启动 warmup 历史不得带入后来的正式 episode；每轮重新采集新鲜历史。不要复制同一帧凑满历史，不全局放大 2 秒预测等待，也不新增 bootstrap 状态机。
- 会话目录直接尝试 `mkdir`，仅捕获 `FileExistsError` 后选择下一序号；去掉 exists→mkdir 的竞争窗口，不引入锁文件。
- session `finished` 表达生命周期结束；episode 终止原因与技术失败次数单独、明确记录在现有摘要位置。默认不改 CLI rounds 的“已开始轮数”口径，也不把正常预算结束一律改为非零退出。
- 给预测请求/结果携带已有观测源时间，优先记录 `observation_to_submit` 与 inference/等待耗时。在没有真实 arm SDK 应用确认时不得命名为 apply latency；不要为了指标引入执行 ACK 协议或新的硬超时策略。

验收：相同映射 6 ms，结果落在旧 tick 前 10 ms 或 5 ms 不再一成一败；`n_obs_steps=42,20 Hz` 在准备阶段能形成真实历史再开始推理等待预算；旧 epoch/future 仍不可被新 episode 消费；并发同名目录创建得到两个不同路径；摘要能区分正常预算结束与技术失败。

### A5 — 数据导出做减法，保持实验口径

关联：DATA-01/02/03/04/05/06。

- **默认保留**固定周期有效前缀、failed episode 保守丢末行、完整 13 模态 Canonical。它们是现有实验口径；不要擅自增加模式选择、自动补帧、插值、重采样、分段训练或任意模态 schema。
- 在现有导出摘要中说明有效行数、截断位置及原因，区分失败末行保守裁剪和节拍不合格。录制时钟、观测时钟、回放时钟含义写清楚，不为“统一”而重写历史数据。
- 视频导出只需解码与验证最终保留的行范围。保留区间中的缺帧/错形状/坏数据仍失败；已明确不导出的损坏尾部不阻止前缀发布。不新增强制全视频审计；如已有单独审计入口，仍可检查完整文件。
- 结合导出重排，若能直接复用同一块已读数据，则合并所用字段的有限值检查与转换读取。确定前缀的一遍扫描并非天然冗余；不能为了省一次读取新增缓存/流水线。原样拷贝的已验证值可避免重复检查，但浮点 cast 可能溢出，FK/点云等新计算结果仍需验证。无直接删减收益则保留并在 A8 测量；不要读全量视频到内存来减少扫描次数。
- 若未来确有数据损耗实验需求，再单独评估连续有效片段；同一 Raw 派生片段必须共享 trial 身份并按 trial 划分 train/val，避免相邻片段泄漏。此项**不在默认代码实施范围内**。

验收：视频容器及保留前缀本身可正常解码时，前 300 行可用而被裁尾部损坏仍可导出这 300 行；第 200 行损坏仍失败。不能承诺索引/头部损坏的 MP4 也可恢复。同一输入在优化前后保留区间、字段值、dtype、episode ends 相同。保留 failed 丢尾与严格前缀的现有结果并在摘要准确解释。

### A6 — 来源时间、标定和诊断

关联：GEO-02/04、DIAG-01/02；GEO-03 留 A9。

- 将相机主机接收时间绑定在进入应用接收队列的一刻，并与帧一起传递，不能 `try_wait_for_frame` 出队时重打“新鲜”时间。优先用 RealSense 已支持的回调入口加一个有界最新帧槽/小队列；沿用独立 camera worker，不新增服务。回调只接收并保存帧引用、两通道帧号及 monotonic 接收时间，重活仍在 worker。
- depth/color 分别只在编号更新时更新其首次接收时间；重复通道不能借另一通道更新变年轻。处理帧引用生命周期、首次回调、错误传回和关闭等待；不把 SDK 设备时钟直接与 host monotonic 相减，不造通用时钟同步模块。这里只保证首次主机回调接收后的年龄，不能宣称是曝光时间或覆盖设备/SDK 到回调前的一切延迟。若目标 SDK 不支持所选入口，应以实际 API 选择同等最小方案并报告限制，不能退回出队重打时间伪装修好。
- 外参 runtime 仅读取并校验选中 serial；保存时保留其他条目，不因无关相机条目数值错误阻止修正当前相机。整个 JSON 损坏仍明确失败。
- 点云诊断复用现有曝光读写函数，在 finally 恢复进入前的 AE priority；不改变生产采集默认值。现有两个 helper 在 `scripts/tools/view_realsense.py`，可移入现有相机模块供两入口使用，不新增文件、不从入口脚本互相 import。桌面多帧/独立深度 benchmark 用现有 depth frame number 去重；不改变正常 RGB-D read 允许通道异步更新的契约。

验收：暂停消费者后读取排队帧，其年龄仍包含暂停时间；只更新 RGB 时 depth 来源时间不变；重复深度不增加独立样本数；选中 serial 有效而另一个条目损坏时能读/改当前标定；诊断正常/异常退出均请求恢复原 AE 值。SDK 调用与设备效果需要在实际环境验证，mock 只证明控制流。

### A7 — 可选 CordVIP 局部修复

关联：OPT-01/02/04/05/06；OPT-03 保留。

- 删除安装脚本无条件 `TORCH_CUDA_ARCH_LIST=8.9`；尊重用户设置，未设置时由 PyTorch 按可见设备处理。无可见 GPU 的离线构建在现有说明中给出手动设置办法，不加自动探测/兼容矩阵。
- 关闭时继续丢弃取消后的几何结果，但保留/记录独立模型异常；预期取消不当作新故障。不修改当前 worker 生命周期和任务标识机制。
- mask 复用仍比较实际 RGB、depth、K、畸变及行/时间对应，保留 selection metadata 与保存帧的自洽检查；只解除这条初始 selection 复用路径对绝对 Raw 路径的绑定，将路径作为 provenance。不重定义其他流式 source_id，不用新的内容指纹代替已存在的内容检查。
- pose 初始化按首帧 SAM/互斥/stride/有效深度后的实际支持点数判定；带 pose asset 的实例不足时明确初始化失败，提示重新选择并启动新 session。mask-only 实例沿用其较小门槛；不自动降级 observed-only，不把全局 4 像素改成 30，不增加 GUI 原地重选/自动重注册框架。保留后续真正 POSE_LOST 的显式语义。
- 已独立复制的 frozen 帧允许用户确认保存；随后来源错误仍展示/记录，未冻结时仍失败。不得自动保存，也不自动重连。

验收：环境变量不被覆盖；close 与在途独立异常竞态时错误可见但结果仍无效；Raw 只换目录仍能复用同一有效 selection、改像素/标定仍拒绝；64 个原始深度像素 stride2 得到 16 个支持点时不能进入需要 30 点的 pose 初始化；冻结后来源失败仍能由用户确认保存，未冻结不能。

### A8 — 性能与微型清理必须有收益

关联：DEP-06、DATA-10/11、GEO-05/06、CTRL-11/12、TEL-08/09。

- RTC 冻结命令先确认 immutable 输入和静态几何相同，再复用已有校验结果。两处 EEF FK 分别来自 MPlib 与 Pinocchio，不能仅看函数名就删；先局部统一数值来源或证明误差在现有容差内。保留动态连续性、来源时效和授权检查，不加跨帧全局缓存。
- 同一目标的重复 FK 仅在局部传递已有计算结果足以消重时修改；不要为它统一两套机器人库。无实际耗时或复杂度收益则保留。
- 对 pointcloud 的 Zarr chunk，以实际 policy sampler 比较时间维 `1/2/4/现有42`，观察训练 DataLoader 吞吐、解压时间、峰值内存、存储尺寸和相同样本值。选择本实验最快且成本合理者；理论 21 倍解压放大**不是**21 倍训练变慢，不预先保证收益、不新增可调 chunk 平台。
- `radius_inliers` 邻居计数只在实际点云上测量后决定；现有合成测试仅证明更省内存，单次耗时反而较慢。保留准确邻居定义及同点数规则。
- replay 现有按索引的轨迹复现误差保留并命名清楚；只有当前论文需要跟踪性能分析时，再加利用已有数据的 target-vs-actual/时间偏移分析，区分绝对角与等价角，不改原指标含义。
- 默认保留旧但可能有外部引用的 xArm 文件、身份查询、两个语义清楚的手到位等待、`compute_target/commit` 两阶段。只有已有具体调用改动使其确实冗余才顺带清理；不为减少几行代码开展专项重构，不删除仍被使用的 mesh 目录。

### A9 — 硬件和几何能力边界的最终决定

关联：CTRL-08/09、GEO-01/03，以及 DEP-09。先在已有文档的相应位置准确说明能力，不声称已经提供不存在的保证。

| 项目 | 默认方案 | 何时才实施额外机制 |
| --- | --- | --- |
| hand 独立 watchdog | 保留现有所有权，记录 arm 的独立期限不能终止阻塞中的 hand SDK；核对实际 SDK 是否已有可用调用超时 | 实测调用/停止延迟不满足实验要求时，先用 SDK 原生期限；只有确需独立隔离才增 hand owner，且替代原路径，不并存双实现 |
| 手内碰撞全部排除 | 不盲目启用全部 435 对；对当前任务姿态检查 mesh 与 SRDF 原因 | 有具体不合理穿透和可信 collision mesh 时，只恢复相关非相邻对，并回归合法抓握姿态；碰撞网格失真先修模型 |
| Ruckig 中间/制动几何 | 明确目前是端点几何准入加轨迹关节极值检查；不声称连续无碰撞 | 当前任务确需近障碍连续运动，且离线找出端点可行、中间不安全的真实轨迹后，复用现有模型对实际生成轨迹含制动采样，测预算后接入；不先写全局规划器，也不拿直线插值当 Ruckig 证明 |
| aligned depth 的 Z 近似 | 保留现行输入表示；记录 SDK/backend 和深度→彩色真实外参 | 实际误差达到任务/论文精度阈值时才使用对应正确投影；同步重导出数据和部署处理，不能只改一端导致训练/在线分布变化 |
| 模型/录制线程 pending | 保留如实 pending、阻止重启和晚到结果隔离 | 只有明确硬退出需求且实测阻塞问题时再考虑进程隔离；不把普通线程设 daemon 来伪造资源已释放 |

没有设备、模型权重或网格库时，这些项交付为“保留当前实现，已说明边界，未完成所列测量”，不虚构通过。人工低速验证前的现场准备仍由操作者完成；本任务不自动连接或驱动真机。

## 5. 全部 57 项的处理清单

以下按前轮优先级顺序列出。每项保留原编号、核实事实与适用条件，并给出默认决定及验收。证据链接固定到审查源码；当前版本发生变化时以实际复核为准。

### 01. DEP-01 — 公开 PolicyInfo 与 warmup 接口在固定版本之间断裂

- 原分类：P1。核实事实：固定两仓版本下，saved_config 等元信息缺失，warmup 参数也不兼容；先对齐唯一公开接口。
- 默认决定：**实施**；执行位置：A1。补齐 Canonical→artifact→公开元信息链，移除 saved_config 私有耦合；按实际 warmup(observation) 传入同形真实观测。
- 验收/关闭条件：原 AttributeError/TypeError 消失；实际公开接口和保存/读取链一起验证。
- 源码：[dexmani_real/deployment/run.py:79](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/run.py#L79)；[dexmani_real/deployment/preflight.py:69](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/preflight.py#L69)。

### 02. CTRL-01 — HOME 运动预算忽略加速度

- 原分类：P2 功能/条件缺陷。核实事实：合法低加速度下，HOME 预算可小于运动时间下界；统一包含加速度的预算。
- 默认决定：**实施**；执行位置：A2。HOME 段与总预算共用含加速度的估计，并覆盖前置停稳、各段 dwell 和查询；不建超时管理器。
- 验收/关闭条件：合法低加速度反例不再被不可能的运动期限提前拒绝；父预算不短于执行阶段。
- 源码：[dexmani_real/robot/xarm7/worker.py:151](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L151)；[dexmani_real/robot/xarm7/driver.py:149](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/driver.py#L149)。

### 03. CTRL-02 — 空 HOME 路径的停稳预算不含必要查询耗时

- 原分类：P2 功能/条件缺陷。核实事实：空路径预算未留足首尾查询耗时；始终合格的反馈也可超时。
- 默认决定：**合并实施**；执行位置：A2。空 HOME 使用已有收敛等待上限，保留排除首尾查询等待的停稳定义。
- 验收/关闭条件：100 Hz、15 ms 查询、300 ms dwell 的受控时钟用例能完成；过期反馈仍拒绝。
- 源码：[dexmani_real/robot/xarm7/worker.py:164](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L164)；[dexmani_real/robot/xarm7/worker.py:78](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L78)。

### 04. CTRL-03 — jog边界目标不变被当成hold

- 原分类：P2 功能/条件缺陷。核实事实：边界裁剪后的目标不变被当作 hold，随后又从落后实测值重新追赶。
- 默认决定：**实施**；执行位置：A3。workspace 裁剪后目标相同保持最后已接受目标；无输入和拒绝仍走各自制动；变化后的几何事实重新准入。
- 验收/关闭条件：持续限位输入不再出现 hold/重新从滞后实测累加；松键仍连续制动。
- 源码：[dexmani_real/runtime/jog.py:95](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/jog.py#L95)；[dexmani_real/runtime/jog.py:113](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/jog.py#L113)。

### 05. CTRL-06 — XHand 部分启动清理误锁存未解决运动

- 原分类：P2 功能/条件缺陷。核实事实：未成功启用的 XHand 对象在清理时仍被要求停车，锁存虚假的未解决运动。
- 默认决定：**实施**；执行位置：A2。部分启动按现有连接及运动事实收尾；未启用 hand 不制造 unresolved motion，已运行后断联仍报错。
- 验收/关闭条件：启动失败保留原异常、另一设备仍清理；运行后断联不会被当作从未运动。
- 源码：[dexmani_real/robot/xhand/driver.py:190](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xhand/driver.py#L190)；[dexmani_real/robot/xhand/driver.py:249](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xhand/driver.py#L249)。

### 06. TEL-01 — 普通S/Q取消竞态误记技术失败

- 原分类：P2 功能/条件缺陷。核实事实：普通 S/Q 在计算或提交期间撤权，可被写成 failed 并播放紧急提示。
- 默认决定：**实施**；执行位置：A3。本 epoch 已记录 OPERATOR/QUIT 的取消按正常结束，独立真实故障仍失败，未完成行不补录。
- 验收/关闭条件：S/Q 落在 compute/commit 之间不再误记 failed；SDK/ESTOP/超时故障仍可见。
- 源码：[dexmani_real/teleop/run.py:125](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/run.py#L125)；[dexmani_real/teleop/loop.py:375](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L375)。

### 07. DEP-02 — sync 结果消费时只剩旧 tick 的相位余数预算

- 原分类：P2 功能/条件缺陷。核实事实：sync 结果在旧 tick 前就绪时，只剩相位余数供映射和提交，形成时序相关误停。
- 默认决定：**实施**；执行位置：A4。sync 消费结果获得完整控制周期；仍受反馈有效期和总运行期限约束。
- 验收/关闭条件：旧 tick 前 10/5 ms、映射耗时 6 ms 的相位反例均正确；真正超时仍拒绝。
- 源码：[dexmani_real/deployment/rollout.py:481](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L481)；[dexmani_real/deployment/rollout.py:562](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L562)。

### 08. DEP-03 — 首次采集历史与首次推理隐式争用2秒预算

- 原分类：P2 功能/条件缺陷。核实事实：首次历史收集与首次预测共用两秒；较长合法历史尚未收满就超时。
- 默认决定：**实施**；执行位置：A4。首次本轮真实历史和已提交预测分别给有限等待预算，不复用启动 warmup 的旧历史。
- 验收/关闭条件：42 帧/20 Hz 可到达 predict；预测超时、来源失效、取消和全局期限仍生效。
- 源码：[dexmani_real/config/deployment.py:9](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/config/deployment.py#L9)；[dexmani_real/deployment/rollout.py:101](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L101)。

### 09. TEL-03 — 默认回放首姿态双约束可能无解

- 原分类：P2 功能/条件缺陷。核实事实：默认要求手同时接近首实测与首目标；两者相差超过20°时准入区间无交集。
- 默认决定：**实施**；执行位置：A3。单一起点为录制首实测；手准备也到该姿态，明确检查新准备目标限位，之后发原目标序列。
- 验收/关闭条件：state=0°、action=40° 不再因双 ±10° 准入无解；不静默夹紧准备目标。
- 源码：[scripts/replay_episode.py:46](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/replay_episode.py#L46)；[dexmani_real/replay/joint_replay.py:125](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/replay/joint_replay.py#L125)。

### 10. DATA-09 — Canonical缺source_id使新数据可通过resume身份校验并复用旧normalizer

- 原分类：P2 功能/条件缺陷。核实事实：新导出缺 source_id，名称/行数相同的数据可通过旧 manifest 并复用旧 normalizer。
- 默认决定：**实施**；执行位置：A1。新导出一次 UUID，搬迁保留；Canonical resume 任一端缺 ID 均拒绝身份确认，保留旧数据读取/评价。
- 验收/关闭条件：原样搬迁通过；同名同长度的新导出拒绝；None==None 不再冒充已验证同源。
- 源码：[dexmani_real/data/export.py:321](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L321)；[dexmani_real/data/export.py:354](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L354)。

### 11. CTRL-04 — 选定绝对分支后，IK 连续性评分再次取模

- 原分类：P2 功能/条件缺陷。核实事实：绝对分支确定后又取模评分；窄限位示例实际6.2rad却按0.083rad计，未证明 FAST_ACCEPT。
- 默认决定：**实施**；执行位置：A3。选定合法绝对分支后用绝对关节差评分；周期化只用于等价候选选择。
- 验收/关闭条件：3.1→-3.1 按约 6.2 rad 比较，而非 0.083 rad；不宣称该反例已证明 FAST_ACCEPT。
- 源码：[dexmani_real/control/ik.py:163](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/ik.py#L163)；[dexmani_real/control/ik.py:117](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/ik.py#L117)。

### 12. CTRL-05 — 在线与 HOME 的 workspace 外扩容差不一致

- 原分类：P2 功能/条件缺陷。核实事实：在线容差5mm而 HOME 仅1mm，规则上允许进入随后被 HOME 起点检查拒绝的边界带。
- 默认决定：**实施**；执行位置：A3。HOME 与在线共用已配置 workspace margin，取消第二套容差，不新增逃逸规划。
- 验收/关闭条件：同一边缘 FK 点在两路径准入一致；真实越界仍拒绝。
- 源码：[dexmani_real/control/constraints.py:57](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/constraints.py#L57)；[dexmani_real/config/control.py:99](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/config/control.py#L99)。

### 13. TEL-02 — DexPilot首帧未从实测值初始化滤波

- 原分类：P2 功能/条件缺陷。核实事实：在已核对 LPFilter 版本下，reset 未用实测值初始化滤波，首输出绕过 alpha。
- 默认决定：**实施**；执行位置：A3。DexPilot 优化前解和滤波同时从实测 q 播种；沿用真实 LPFilter 接口。
- 验收/关闭条件：alpha=.1、q=.2、优化解=.8 时第一帧约 .26；后续滤波语义不变。
- 源码：[dexmani_real/teleop/vr.py:32](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/vr.py#L32)；[dexmani_real/teleop/retargeting/dexpilot.py:38](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/retargeting/dexpilot.py#L38)。

### 14. OPT-04 — 安装脚本硬编码 sm_89 编译目标并覆盖调用者设置

- 原分类：P2 功能/条件缺陷。核实事实：安装脚本覆盖 TORCH_CUDA_ARCH_LIST 为8.9；非该目标或混合GPU时可能构建不适用扩展。
- 默认决定：**实施**；执行位置：A7。删除硬编码 CUDA 架构，尊重调用者；默认使用 PyTorch 的可见设备处理。
- 验收/关闭条件：预设环境变量未被覆盖；无 GPU 构建说明手动设置；未实际编译不称兼容通过。
- 源码：[third_party/cordvip_geometry/scripts/install_perception.sh:108](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/scripts/install_perception.sh#L108)；[third_party/cordvip_geometry/scripts/install_perception.sh:111](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/scripts/install_perception.sh#L111)。

### 15. CTRL-08 — 独立 watchdog 和执行期限只覆盖 arm

- 原分类：P2 能力/策略决定。核实事实：独立停止与期限只覆盖 arm；父进程阻塞时不能把此保证外推到 hand。
- 默认决定：**测量后决定**；执行位置：A9。保留当前所有权，明确独立期限仅覆盖 arm；先核实/测量真实 hand SDK 超时与停止，再决定最小隔离。
- 验收/关闭条件：有 SDK/设备记录才宣称 hand 独立停止能力；不默认增加对称进程框架。
- 源码：[dexmani_real/robot/xarm7/worker.py:227](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L227)；[dexmani_real/robot/xarm7/worker.py:293](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L293)。

### 16. CTRL-09 — SRDF 排除了全部手内碰撞对

- 原分类：P2 能力/策略决定。核实事实：SRDF 将手内435对全部排除；剩余 hand↔hand 为0，不代表已有指间碰撞保护。
- 默认决定：**测量后决定**；执行位置：A9。记录手内 435 对全排除事实；基于真实网格与任务姿态只恢复有证据的相关碰撞对。
- 验收/关闭条件：不盲目全开；启用任何对前验证合理抓握和已知危险姿态，mesh 不可靠先修 mesh。
- 源码：[assets/robots/xhand/xarm7_xhand.srdf:351](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/assets/robots/xhand/xarm7_xhand.srdf#L351)；[dexmani_real/control/collision.py:37](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/collision.py#L37)。

### 17. GEO-01 — 在线端点几何准入不覆盖实际 Ruckig 中间参考及制动路径

- 原分类：P2 能力/策略决定。核实事实：当前检查目标端点，未证明整段 Ruckig 路径无碰撞；是否补段检查取决于实验目标跨度。
- 默认决定：**测量后决定**；执行位置：A9。保留端点准入和关节极值能力，明确未检查 Ruckig 全路径；真实反例出现后才评估轨迹含制动采样。
- 验收/关闭条件：不能用线性插值代替实际 Ruckig 证明；新增采样须有几何和控制周期成本证据。
- 源码：[dexmani_real/control/constraints.py:40](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/constraints.py#L40)；[dexmani_real/control/constraints.py:57](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/constraints.py#L57)。

### 18. GEO-02 — 相机来源年龄实际从 SDK 出队时刻计算，无法覆盖出队前延迟

- 原分类：P2 能力/策略决定。核实事实：SDK 出队时盖 host 时间；特定排队停流情形下，旧帧可能被报告为 fresh。
- 默认决定：**实施局部修复**；执行位置：A6。首次主机回调接收时间与帧入队，分别保留 depth/color 时间，read 不重打时间。
- 验收/关闭条件：消费者暂停后的队列旧帧年龄不归零；仅 RGB 更新不刷新 depth；不声称曝光时刻。
- 源码：[dexmani_real/sensor/realsense.py:76](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/sensor/realsense.py#L76)；[dexmani_real/sensor/realsense.py:129](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/sensor/realsense.py#L129)。

### 19. CTRL-07 — arm+hand 组合命令没有事务性或同步执行保证

- 原分类：P2 能力/策略决定。核实事实：arm 异步邮箱与 hand 同步发送不是原子事务；保持提交尝试语义，勿靠重复预检假装同步。
- 默认决定：**保留并说明**；执行位置：A2。臂手顺序下发继续使用已有组合层和失败停止；提交不等于原子执行或到位。
- 验收/关闭条件：保留部分失败原因和停止链；不新增事务、ACK、回滚或同步到位承诺。
- 源码：[dexmani_real/robot/system.py:246](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/system.py#L246)；[dexmani_real/robot/xarm7/protocol.py:79](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/protocol.py#L79)。

### 20. DATA-03 — 严格有效前缀舍弃后续可用数据

- 原分类：P2 能力/策略决定。核实事实：一次局部抖动可裁掉全部后续数据；这是明确质控策略，先评估利用率再调整。
- 默认决定：**保留并说明**；执行位置：A5。保留严格有效前缀，在现有质量摘要展示丢弃数量/原因；本轮不分段或重采样。
- 验收/关闭条件：30 Hz 第 300 行晚 9 ms 的例子仍保留 300/丢弃 700；未来分段必须按原 trial 划分训练验证。
- 源码：[dexmani_real/data/export.py:93](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L93)；[dexmani_real/utils/timing.py:40](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/utils/timing.py#L40)。

### 21. DATA-04 — 强制全模态准入扩大失败面

- 原分类：P2 能力/策略决定。核实事实：13模态统一准入可让未用模态阻断实验；统一数据集也有价值，按实验目标决定子集支持。
- 默认决定：**保留并说明**；执行位置：A5。保留全 13 模态 Canonical，区分完整数据资产与 policy 输入子集；不新增模态注册表。
- 验收/关闭条件：部署按需依赖与完整录制职责仍分开；不隐式降低导出保证。
- 源码：[dexmani_real/data/export.py:47](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L47)；[dexmani_real/data/export.py:232](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L232)。

### 22. DEP-05 — 同秒创建同一实验会话目录存在TOCTOU竞争

- 原分类：P3。核实事实：exists 后 mkdir 有并发竞争；原子创建失败后再递增后缀即可。
- 默认决定：**实施小修**；执行位置：A4。直接 mkdir，仅 FileExistsError 换后缀；其他异常原样抛出。
- 验收/关闭条件：同秒并发创建不会冲突覆盖；不新增锁文件或目录管理器。
- 源码：[dexmani_real/deployment/run.py:27](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/run.py#L27)。

### 23. OPT-06 — SceneWorker关闭会丢弃在途任务的独立异常原因

- 原分类：P3。核实事实：关闭期间的在途任务/加载异常可随取消结果一起消失；丢结果仍应保留错误日志。
- 默认决定：**实施小修**；执行位置：A7。取消后不发布结果，但保留独立的在途加载/模型异常诊断；预期取消不当新故障。
- 验收/关闭条件：close 与 ValueError 竞态下错误可见，poll 仍不暴露失效结果。
- 源码：[third_party/cordvip_geometry/src/cordvip_geometry/live_workers.py:208](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/live_workers.py#L208)；[third_party/cordvip_geometry/src/cordvip_geometry/live_workers.py:245](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/live_workers.py#L245)。

### 24. DEP-06 — RTC冻结命令重复静态几何检查，EEF还重复求FK

- 原分类：P3。核实事实：冻结目标的碰撞查询重复；EEF 两次 workspace 来自不同 FK，先统一判据再删整段。
- 默认决定：**有条件简化**；执行位置：A8。在静态冻结事实成立、workspace FK 来源统一/证实等价后复用几何结论；保留动态分支和提交边界。
- 验收/关闭条件：相同静态目标 collision 不重复；变更反馈/分支仍拒绝；未经验证不删第二种 FK。
- 源码：[dexmani_real/deployment/rollout.py:378](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L378)；[dexmani_real/deployment/rollout.py:402](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L402)。

### 25. DATA-05 — 裁掉的视频尾部仍需完整解码和结构校验

- 原分类：P3。核实事实：有效前缀写完仍解码裁去尾段，尾部损坏会令前缀导出失败；分开导出与全量审计。
- 默认决定：**实施小修**；执行位置：A5。只解码验证实际保留视频区间，不让裁掉的尾部阻止合法前缀导出。
- 验收/关闭条件：容器与前缀可解码时尾部损坏不影响导出；保留区间损坏仍失败；不承诺修复坏索引。
- 源码：[dexmani_real/data/export.py:238](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L238)；[dexmani_real/data/export.py:274](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L274)。

### 26. OPT-02 — mask复用契约把绝对 Raw 路径当成帧身份

- 原分类：P3。核实事实：完全相同 Raw 迁移路径后 mask 被拒绝；路径应作溯源，保留实际内容和帧对应检查。
- 默认决定：**实施小修**；执行位置：A7。仅取消初始 selection 复用对绝对 Raw 路径的绑定；保留保存帧/metadata/像素/标定对应。
- 验收/关闭条件：原样搬迁可复用，内容/几何变化拒绝；其他 source_id 不变。
- 源码：[third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py:13](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py#L13)；[third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py:29](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py#L29)。

### 27. OPT-01 — 首帧选择的4像素门槛与采样后30点位姿门槛不一致

- 原分类：P3。核实事实：选择只需4深度像素，stride后位姿需30点；首帧可立即可见地永久 POSE_LOST。
- 默认决定：**实施小修**；执行位置：A7。带 pose asset 的首帧用实际 SAM/互斥/stride 后支持点数验初始化，不足明确失败重开 session；mask-only 不受新阈值限制。
- 验收/关闭条件：64 原像素→16 采样点不能进入需 30 点的 pose 初始化；无自动降级/重注册。
- 源码：[third_party/cordvip_geometry/src/cordvip_geometry/cli/scene_initialization.py:126](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/cli/scene_initialization.py#L126)；[third_party/cordvip_geometry/src/cordvip_geometry/io.py:143](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/io.py#L143)。

### 28. OPT-05 — 已冻结帧的保存资格仍受后续来源错误控制

- 原分类：P3。核实事实：冻结帧已有独立复制，后续来源故障仍先于 Enter 处理，阻止有效帧确认保存。
- 默认决定：**实施小修**；执行位置：A7。冻结副本的保存资格不依赖后来来源是否正常；源错误仍独立记录，保存由用户确认。
- 验收/关闭条件：冻结后故障可确认保存；未冻结故障不能保存；不自动保存或重连。
- 源码：[third_party/cordvip_geometry/src/cordvip_geometry/cli/capture_object.py:149](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/cli/capture_object.py#L149)；[third_party/cordvip_geometry/src/cordvip_geometry/cli/capture_object.py:179](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/cli/capture_object.py#L179)。

### 29. DATA-10 — 42行点云chunk与2行随机观测窗口不匹配

- 原分类：P3。核实事实：默认点云42行/块，DP3只读2行；21/42倍是解压工作量分析，未证明训练等比例变慢。
- 默认决定：**测量后决定**；执行位置：A8。真实 sampler 比较 1/2/4/当前 42 行 chunks，选择固定合适值；无收益就保留。
- 验收/关闭条件：同样本值、实际 DataLoader 吞吐/内存/存储对比；不将理论 21 倍读取量当提速。
- 源码：[dexmani_real/data/export.py:34](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L34)；[dexmani_real/data/export.py:378](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L378)。

### 30. DATA-06 — Raw扫描和转换输出检查存在重复读取

- 原分类：P3。核实事实：先扫原始 finite、再读转换、再扫输出有重复；保留边界与新计算结果检查，融合原样字段读取。
- 默认决定：**有条件简化**；执行位置：A5/A8。仅在自然重排可复用同一数据块时合并扫描；前缀判定与新计算结果检查仍保留。
- 验收/关闭条件：截断位置/坏字段/输出值一致；不为减少读次数增加缓存或流水线框架。
- 源码：[dexmani_real/data/export.py:132](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L132)；[dexmani_real/data/export.py:194](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L194)。

### 31. DATA-07 — 完整dataclass字段集被作为兼容契约

- 原分类：P3。核实事实：精确 dataclass 键集把结构变更当不兼容；属跨版本维护建议，同版本未见独立默认失败。
- 默认决定：**合并实施**；执行位置：A1。显式读取当前实际数值字段，删除整个 dataclass 键集绑定和一律重训文案；无默认补缺。
- 验收/关闭条件：无关备注键可存在；缺少当前实际数值字段仍失败；不新增版本兼容层。
- 源码：[dexmani_real/deployment/preflight.py:18](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/preflight.py#L18)；[dexmani_real/deployment/preflight.py:100](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/preflight.py#L100)。

### 32. GEO-04 — 标定读写校验全部 serial，使无关坏条目阻止当前设备

- 原分类：P3。核实事实：无关相机 serial 的坏条目可阻止有效设备加载/标定保存；缩小运行校验范围。
- 默认决定：**实施小修**；执行位置：A6。只校验实际选中 serial；保存当前项时保留其余内容；JSON 整体坏仍失败。
- 验收/关闭条件：其他 serial 的坏矩阵不妨碍当前相机读/修；当前坏矩阵仍拒绝。
- 源码：[dexmani_real/utils/camera_calibrate/extrinsics.py:15](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/utils/camera_calibrate/extrinsics.py#L15)；[dexmani_real/utils/camera_calibrate/extrinsics.py:26](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/utils/camera_calibrate/extrinsics.py#L26)。

### 33. GEO-05 — 部分在线端点调用先 ArmFK 再完整碰撞 FK

- 原分类：P3。核实事实：端点工作空间与碰撞模型分别做 FK；先测耗时和统一模型事实，不承诺已有加速。
- 默认决定：**有条件简化**；执行位置：A8。同一查询可直接复用现有 FK 结果时才合并；不引入共享模型平台。
- 验收/关闭条件：边界数值等价且消除实际重复成本；未证实收益则保留。
- 源码：[dexmani_real/control/constraints.py:57](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/constraints.py#L57)；[dexmani_real/control/collision.py:118](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/control/collision.py#L118)。

### 34. TEL-04 — 标定阻塞操作积压一次性键盘事件

- 原分类：P3。核实事实：HOME/采样期间的 SPACE/ENTER 会积压到新场景；明确一次性事件缓冲策略。
- 默认决定：**实施小修**；执行位置：A3。阻塞操作后清理积压的一次性采样/求解键，保留取消/急停及实时 held-key 状态。
- 验收/关闭条件：HOME 期间短 Space 不在结束后自动采样；Q/Esc 不被清掉。
- 源码：[dexmani_real/runtime/keyboard.py:78](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/keyboard.py#L78)；[dexmani_real/utils/camera_calibrate/calibrate.py:269](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/utils/camera_calibrate/calibrate.py#L269)。

### 35. TEL-05 — healthy读取隐含急停副作用及重复映射

- 原分类：P3。核实事实：healthy 读取实际会锁存急停，入口又重复映射；问题是副作用可读性，非未急停。
- 默认决定：**实施小修**；执行位置：A3。将有副作用的 healthy 改明确 check_health 调用，同步实际引用，保留原停止信号时机。
- 验收/关闭条件：原故障检测点仍锁存急停；不宣称物理瞬时停止，不新增健康管理层。
- 源码：[dexmani_real/runtime/keyboard.py:51](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/keyboard.py#L51)；[dexmani_real/runtime/keyboard.py:175](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/keyboard.py#L175)。

### 36. TEL-06 — 异常分类多处复制及语义漂移

- 原分类：P3。核实事实：异常分类多份维护，无 policy 的流程也可标 POLICY_FAILURE；集中分类并保持业务差异。
- 默认决定：**有条件合并**；执行位置：A3。只共享真实设备异常分类的短函数，各入口未知业务故障与 HOME 策略保留。
- 验收/关闭条件：同类设备错误分类一致；非 policy 入口不机械套 POLICY_FAILURE；抽象更复杂则局部修正即可。
- 源码：[dexmani_real/teleop/loop.py:25](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L25)；[dexmani_real/deployment/rollout.py:39](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L39)。

### 37. TEL-08 — HOME/replay手到位判据重复维护

- 原分类：P3。核实事实：HOME/replay 连续三帧到位判据与常量重复；可共享小判据，不合并全部业务循环。
- 默认决定：**保留**；执行位置：A3/A8。两个等待循环承担不同 authority/ramp/service 语义，先保留；改动时至多共用短判据或常量。
- 验收/关闭条件：现有新鲜反馈/连续到位数/取消语义不变；不建 WaitManager。
- 源码：[dexmani_real/runtime/home.py:20](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/home.py#L20)；[dexmani_real/runtime/home.py:159](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/home.py#L159)。

### 38. DATA-02 — 采集、导出、回放采用不同节拍时钟和准入规则

- 原分类：P3。核实事实：正常录制、可导出、可回放的集合不同，是现有工作流限制；与严格前缀合并评审。
- 默认决定：**保留并说明**；执行位置：A5。各用途时钟保持原义，准确说明导出与回放可接受集合不同。
- 验收/关闭条件：不宣称所有 Raw 必能回放，不重打时间或自动重采样。
- 源码：[dexmani_real/runtime/observation.py:52](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/observation.py#L52)；[dexmani_real/teleop/loop.py:425](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L425)。

### 39. DEP-04 — 技术终止可与会话 finished/退出0并存，需明确统计口径

- 原分类：P3。核实事实：finished/退出0表示会话完成，可与单轮技术终止并存；需说明统计口径，非任务成功率错误。
- 默认决定：**实施说明与摘要**；执行位置：A4。finished 解释为生命周期结束；用现有信息显示正常/技术终止原因，保留 rounds 与退出码既有口径。
- 验收/关闭条件：技术终止但清理正常时可 finished，摘要仍明确失败原因；不虚构任务成功率。
- 源码：[dexmani_real/deployment/rollout.py:150](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L150)；[dexmani_real/deployment/rollout.py:192](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/rollout.py#L192)。

### 40. DATA-01 — failed episode保守裁去末行

- 原分类：P3。核实事实：统一丢末行会丢有效行，但无逐行提交结果时是有意保守策略；不能直接取消。
- 默认决定：**保留**；执行位置：A5。真实 failed 末行继续保守裁掉；先修 TEL-01 减少正常结束被错裁，不新建逐行执行 ACK。
- 验收/关闭条件：正常 S/Q 不误进 failed；真实失败裁剪不变，历史 Raw 不自动补行。
- 源码：[dexmani_real/data/export.py:93](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L93)；[dexmani_real/teleop/loop.py:167](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L167)。

### 41. DATA-08 — 数值recipe未涵盖算法常量和手模板内容

- 原分类：P3。核实事实：改代码/同名模板后，配置键相同不保证数值相同；先固定源码与资产，当前同版本未见失配。
- 默认决定：**最少溯源**；执行位置：A1。记录可定位的配套代码/固定模板来源，沿用完整 URDF 和同一 processor；不哈希/配置化所有常量。
- 验收/关闭条件：能定位实验使用的代码资产，修改加工需重导的说明明确；不自动拒绝所有版本差异。
- 源码：[dexmani_real/data/export.py:354](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/data/export.py#L354)；[dexmani_real/perception/pointcloud.py:11](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/perception/pointcloud.py#L11)。

### 42. DEP-08 — 缺少观测到动作应用延迟指标，属于可观测性能力限制

- 原分类：P3。核实事实：推理耗时不等于观测至动作延迟；补时间锚点指标，勿把相机 freshness 直接当推理上限。
- 默认决定：**实施低成本指标**；执行位置：A4。沿现有 query/result 传观测锚点，在现有摘要输出观测到提交与推理耗时。
- 验收/关闭条件：受控时钟时间差正确；没有硬件 ACK 不称 apply latency；不增加逐 tick 写盘或时限策略。
- 源码：[dexmani_real/deployment/observation.py:49](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/observation.py#L49)；[dexmani_real/deployment/inference.py:54](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/inference.py#L54)。

### 43. DATA-11 — 回放指标未分解目标跟踪、时间偏移及等价角

- 原分类：P3。核实事实：现指标测按帧复现差异，语义成立；目标跟踪、时间偏移和等价角是补充分解。
- 默认决定：**保留并说明**；执行位置：A8。现有按索引实测复现误差命名清楚；论文需要时另算目标跟踪/时间对齐，不改变原指标。
- 验收/关闭条件：区分绝对关节坐标差与等价角、EEF 误差，不声称已经测得目标跟踪性能。
- 源码：[dexmani_real/replay/metrics.py:1](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/replay/metrics.py#L1)；[dexmani_real/replay/metrics.py:15](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/replay/metrics.py#L15)。

### 44. DIAG-01 — 点云诊断未恢复曝光优先级

- 原分类：P3。核实事实：点云诊断未请求恢复 AE priority；是否跨 pipeline 持久化未实测。
- 默认决定：**实施小修**；执行位置：A6。在现有相机模块共用两个曝光操作，诊断 finally 恢复原值，生产默认不变。
- 验收/关闭条件：正常/异常退出都请求恢复；实际设备持久化行为未测则不称已证实。
- 源码：[scripts/tools/view_pointcloud.py:330](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/tools/view_pointcloud.py#L330)；[scripts/tools/view_pointcloud.py:423](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/tools/view_pointcloud.py#L423)。

### 45. DIAG-02 — 多帧深度计数未按深度编号去重

- 原分类：P3。核实事实：RGB 更新而 depth 未变也可计作一帧；多帧深度/benchmark 可用现有编号去重。
- 默认决定：**实施小修**；执行位置：A6。仅在独立深度采样/benchmark 用 depth frame number 去重，不更改正常 RGBD 更新规则。
- 验收/关闭条件：RGB 更新、depth 重复时独立深度样本数不增加。
- 源码：[scripts/tools/view_pointcloud.py:133](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/tools/view_pointcloud.py#L133)；[scripts/tools/view_pointcloud.py:257](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/tools/view_pointcloud.py#L257)。

### 46. CTRL-11 — 无引用旧 xArm URDF/SRDF

- 原分类：P3。核实事实：旧 xArm URDF/SRDF 未被当前调用链引用；确认外部用途后可清理，保留仍被引用的 mesh。
- 默认决定：**保留**；执行位置：A8。旧 URDF/SRDF 默认不删；有确认的清理范围再移除，mesh 仍有实际引用。
- 验收/关闭条件：不因主代码无引用就删除整个资产目录；不做无收益文件整理。
- 源码：[assets/robots/xarm7/xarm7_glb.urdf:218](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/assets/robots/xarm7/xarm7_glb.urdf#L218)；[assets/robots/xarm7/xarm7_glb_mplib.srdf:1](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/assets/robots/xarm7/xarm7_glb_mplib.srdf#L1)。

### 47. CTRL-12 — 正常连接时身份查询的消费价值有限

- 原分类：P3。核实事实：身份 getter 主要服务诊断/debug，未进业务 metadata；可移诊断或确实保存其追溯价值。
- 默认决定：**保留**；执行位置：A8。身份查询已有诊断用途，未测出瓶颈不移动生命周期或删除；需要溯源时可复用已有结果。
- 验收/关闭条件：不把 3 个 getter 宣称成 3 次硬件 I/O；无独立重构。
- 源码：[dexmani_real/robot/xhand/driver.py:188](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xhand/driver.py#L188)；[dexmani_real/robot/xhand/driver.py:213](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xhand/driver.py#L213)。

### 48. DEP-09 — 模型／录制线程的超时不提供强制终止或进程退出上界

- 原分类：P3。核实事实：线程超时仍 pending 是源码明示限制；默认不进程化，不用 daemon 掩盖未完成写入。
- 默认决定：**保留并说明**；执行位置：A9。保留线程 pending、晚到结果隔离、拒绝重启和真实未完成状态。
- 验收/关闭条件：超时不是强制终止；没有硬退出需求不改 daemon/多进程 supervisor。
- 源码：[dexmani_real/deployment/inference.py:71](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/inference.py#L71)；[dexmani_real/deployment/session.py:279](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/session.py#L279)。

### 49. TEL-09 — VR compute_target／commit 分离

- 原分类：P3。核实事实：当前合并可等价，但两行 commit 明确提交后更新语义；不单独排重构任务。
- 默认决定：**保留**；执行位置：A3/A8。compute_target/commit 两步短且表达明确，失败会清理；不为微型去重单独修改。
- 验收/关闭条件：行为保持，无新状态/新封装。
- 源码：[dexmani_real/teleop/vr.py:58](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/vr.py#L58)；[dexmani_real/teleop/loop.py:361](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L361)。

### 50. CTRL-10 — 父端 prepared 检查可无条件删除的建议

- 原分类：保留/撤回/待测。核实事实：撤回无条件删除父 prepared 检查；它能在 hand 副作用之前提前拒绝。
- 默认决定：**保留，撤回无条件删除**；执行位置：A2。父端 prepared 检查防止 hand 副作用之前错误提交，owner 检查另管最终执行；两处保留。
- 验收/关闭条件：不能以重复读取为理由删边界，未有 lock benchmark 不声称瓶颈。
- 源码：[dexmani_real/robot/xarm7/protocol.py:64](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/protocol.py#L64)；[dexmani_real/robot/xarm7/worker.py:232](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/robot/xarm7/worker.py#L232)。

### 51. DEP-07 — 多份needs有不同职责，保留；整体合并建议撤回

- 原分类：保留/撤回/待测。核实事实：FK依赖、控制、录制、采集需求职责不同；保留区分，不整体压成一份。
- 默认决定：**保留**；执行位置：A4。dependencies/control_needs/record_needs/union 各有实际职责，不合并成万能 needs。
- 验收/关闭条件：仍区分控制来源失败与仅录制失败，无新增注册层。
- 源码：[dexmani_real/deployment/observation.py:18](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/observation.py#L18)；[dexmani_real/deployment/run.py:84](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/deployment/run.py#L84)。

### 52. TEL-07 — VR启动等待重复现有生命周期循环

- 原分类：保留/撤回/待测。核实事实：没有执行两次 ready 等待；通用等待未启用，只有专用循环。共用辅助至多是微整理。
- 默认决定：**保留，撤回双等待指控**；执行位置：A3。start 未传 timeout，实际只有一次专用等待；生命周期不改。
- 验收/关闭条件：不为不存在的双等待写新 callback 框架。
- 源码：[dexmani_real/teleop/run.py:238](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/run.py#L238)；[dexmani_real/runtime/lifecycle.py:28](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/runtime/lifecycle.py#L28)。

### 53. TEL-10 — no-hand配置及确认参数双输入

- 原分类：保留/撤回/待测。核实事实：no-hand 标志确认本次物理 setup，区别于持久设备配置；保留确认。
- 默认决定：**保留**；执行位置：A3。持久配置与本次无手/固定 HOME 的物理状态确认不同，保留 --no-hand/allow_no_hand 明示。
- 验收/关闭条件：没有 hand 反馈时不把 YAML 当成已证实实际姿态。
- 源码：[scripts/collect_teleop.py:27](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/collect_teleop.py#L27)；[scripts/collect_teleop.py:90](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/scripts/collect_teleop.py#L90)。

### 54. TEL-11 — HOME失败外围处理策略不同

- 原分类：保留/撤回/待测。核实事实：不同入口对 HOME 失败选择退出/继续，是业务策略；不能仅因不同判 bug。
- 默认决定：**保留**；执行位置：A3。不同入口对 HOME 失败退出/继续/提示是业务策略，保持显式局部分支。
- 验收/关闭条件：不强制统一 Session/HomeManager，不改操作者习惯。
- 源码：[dexmani_real/teleop/keyboard.py:138](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/keyboard.py#L138)；[dexmani_real/teleop/loop.py:316](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/teleop/loop.py#L316)。

### 55. OPT-03 — 可选 hand cloud 与主仓固定模板是两种不同表示

- 原分类：保留/撤回/待测。核实事实：可选手点云还输出 normals/link IDs，主模板只有XYZRGB；不同表示现阶段有保留理由。
- 默认决定：**保留**；执行位置：A7。可选 hand cloud 有 normals/link IDs，主 XYZRGB 模板职责不同；目前不统一点序/分配。
- 验收/关闭条件：不把两者当同一策略输入；未来确有接入需要才统一表示。
- 源码：[third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py:138](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py#L138)；[third_party/cordvip_geometry/src/cordvip_geometry/hand_cloud.py:103](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/third_party/cordvip_geometry/src/cordvip_geometry/hand_cloud.py#L103)。

### 56. GEO-03 — aligned-depth 原深度 Z 与 color-camera Z 的几何近似

- 原分类：保留/撤回/待测。核实事实：固定SDK通用/SSE路径存在Z近似；实际版本/后端/外参未知，2cm仅合成例子。
- 默认决定：**测量后决定**；执行位置：A9。确认实际 SDK/backend/外参与深度 Z 误差，再决定正确投影；当前不单边改变训练输入。
- 验收/关闭条件：不能把构造 2 cm 例子当设备实测；必要变更同时覆盖离线和在线。
- 源码：[dexmani_real/sensor/realsense.py:155](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/sensor/realsense.py#L155)；[dexmani_real/geometry/camera.py:121](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/geometry/camera.py#L121)。

### 57. GEO-06 — radius_inliers 枚举全部邻居 pair 的内存取舍

- 原分类：保留/撤回/待测。核实事实：直接邻居计数省中间内存，但那次测试反而更慢；保留为待测内存取舍。
- 默认决定：**测量后决定**；执行位置：A8。实际点云测 query_pairs 与邻居计数；只有准确性相同且内存/耗时取舍符合实验才替换。
- 验收/关闭条件：合成曾更省内存但更慢，禁止未经基准宣称更快。
- 源码：[dexmani_real/perception/pointcloud.py:60](https://github.com/haoyangzhanglab/dexmani_real/blob/f07815cce5c29c8e335f6f863b7cb5c5d1a4acd4/dexmani_real/perception/pointcloud.py#L60)。


## 6. 验证和完成标准

1. 针对本次根因保留少量直接调用真实函数的回归测试：公共部署 API 与 metadata 贯通、HOME/jog/取消时序、数据身份与有效前缀、相机来源时间、必要的 optional worker 异常。优先现有测试方式；没有测试设施时使用轻量 `unittest` 或独立测试文件，不引入 CI/仿真平台和一整套 SDK mock 框架。
2. 小范围参数化测试覆盖有证明意义的边界：空 HOME/低加速度、结果跨 tick 相位、初始长历史、取消与真正失败并存、新数据同长度同名字、重复 depth/新的 RGB。无需穷举 millions 调度组合或为纯命名/删冗余写镜像测试。
3. 用仓库实际支持的命令做无设备 smoke：`python scripts/collect_teleop.py --print-config`、`python scripts/run_policy.py --print-config`；修改相关脚本时检查其 `--help`。在已安装相应依赖的环境运行；依赖缺失记为未跑，不称为测试通过。
4. 性能改动报告环境、样本、输入相同的验证和实测指标；不把理论访问量等同 wall-clock 收益。硬件/权重/SDK 未验证的项目与无硬件通过项分开。
5. 完成前审 diff：有无新增而无独立必要的状态、config、wrapper、try/except、fallback、冗余校验、文件、依赖。清理自己引入的复杂度，保留他人工作。
6. 最终中文交付按编号/批次说明：已实现并验证、明确保留、测量后再决定、确实阻塞。列出两个仓库配套提交及实际运行的测试；不必新增状态数据库、审计平台或重复任务文档。

完成的含义是：明确实施的任务已做完，保留项的理由已落实，待测边界已如实标记。不是“把 57 项都改掉”，也不是“本次只写计划”。

## 7. 可直接复制给 Codex 的执行 prompt

```text
请执行 dexmani_real 仓库根目录的 CODEX_TASK_RELIABILITY_AND_SIMPLIFICATION.md。

先读 AGENTS.md、任务书和当前工作区改动，复核当前代码相对任务书基线的变化。目标是个人论文研究代码：正确好用、简洁高效、维护成本低。按任务书 A1–A7 完成明确实施的修改，A8 只做满足证据条件的优化，A9 保留/测量项按其中的边界处理。57 条审查记录不是 57 个必须改的 bug；不要恢复已撤回的建议。

直接推进实现，按能独立验收的行为分批提交，不停在分析或计划。允许修改配套 dexmani_policy 中与元信息保存、公开部署 API、warmup、source_id/resume 直接相关的代码；保留其已有研究方法和其他任务。无配套仓库时先尝试获取，确实不可用则交付最小可应用配套 patch 并继续无依赖项，不造兼容层。

以四个参考仓库的具体代码为思路对照，采用任务书已选的简单做法，不照搬它们的吞错、裁剪补零、假新鲜时间或自动恢复。优先原地修改和删除真实重复，信任已经建立且仍成立的内部不变量，保留外部输入、跨进程执行和硬件副作用边界的必要检查。不新增通用 Session/manager/registry/schema 平台、动作事务、全局缓存、无需求配置、fallback 或重试。

先修根因再做减法；不要仅调整测试以接受旧错误，不虚构完成状态。以少量真实函数回归验证接口、取消/时序、数据身份和来源时间；无硬件可完成的工作持续完成，不自动连接或驱动真机。硬件、模型权重或性能测量不足时，只将相应项标为待测，不能扩成假想防御机制，也不能阻塞其他批次。

完成后检查最终 diff 和新增复杂度，用中文按任务编号总结实际修改、明确保留项、测试结果、未验证项、两个仓库的配套版本及剩余真实阻塞。不要声称未执行的 GPU/真机检查通过。
```
