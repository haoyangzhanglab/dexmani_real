# DexMani Real：研究型 runtime 简化任务书

> 状态：待本地实施；本文件不是已完成的代码或验证报告。
> 本目录是用户明确要求纳入仓库的一次性任务材料，不是永久架构规范，也不会自动替代根目录 CLAUDE.md。

## 0. 阅读顺序、依据与目标

先读仓库 `AGENTS.md`、`CLAUDE.md` 和受影响目录内更深层指令，再完整读取本任务书。`RUNBOOK.md` 是人的启动/恢复指南，`PROMPT.md` 是实施入口，`REVIEW_PROMPT.md` 是最终独立复核入口。无需依赖聊天历史或下载原附件才能执行。

本任务操作化自用户 Handoff《研究型真机 Runtime 审查、简化原则与重构 Handoff》和 `DexMani_Real_Refactor_Review_v2.md`。v2 对旧方案的明确修订优先；本任务不恢复旧方案已取消的功能。v2 附件 SHA-256：`b9efcefc96c0a35fec25bed4d2e76833b4d3d652fd46f5cbc9363bee6098af18`。下文阶段、验收编号、提交/恢复方式是执行组织，不是新增研究方法。

源码审查基线：`haoyangzhanglab/dexmani_real@156b7b4a8b0aa5f42ba12eb2f1abe335d6c5dd1d`。对标参考：LeRobot `5aa74557f84c54d4b458f8b9643c5aa2982acfed` 的 `src/lerobot/robots/so_follower/so_follower.py`；ManiUniCon `85c6f2e32ecf9f2bed62d202b058c39623444686` 的 `maniunicon/utils/shared_memory/shared_storage.py` 与 `maniunicon/core/robot.py`。这些是固定历史参考，不是对其当前全部路径的断言。

当前行为以本地源码/schema/解析配置为准；目标行为由本任务定义。若本地版本已变化，先核对相关差异，保留已正确实现的部分，不回退到审查基线、不重复重构。若真实安全边界或源数据语义与任务假设冲突，记录具体证据并暂停受影响子项，不擅自替换方案。

目标主链：

```text
causal observation → synchronous policy.predict → decode / single soft projection
                   → bounded ordered command stream
                   → arm / hand workers → hard hardware boundary → SDK

recording：旁路观察，不拥有运动权限
data：immutable raw → technical checks / explicit curation → separate derived outputs
```

对齐 LeRobot 的薄执行链与显式 actual-sent target、ManiUniCon 的独立 I/O/有序传输；不复制其 timestamp trimming、action-list replacement、interpolation、fallback。两个 SDK owner、多模态因果 history 和旧 tactile contract 是 DexMani 的真实需要。FIFO 不承诺双 SDK 原子同步、物理收敛或通信不确定时的 exactly-once。

## 1. 执行授权与不可越过的边界

本地实施被授权：修改本任务范围内源码、注释、文档和必要测试；运行已确认无硬件副作用的离线检查；在独立实现分支按完成的行为闭环创建本地 Git commits。不是仅产出第二份计划。

没有授权：连接/发现真实设备、home、servo、teleop、物理 replay、policy rollout、标定写入；运行未检查的 examples；自动 push/merge/force-push；更改相邻 `dexmani_policy`；上传真实 raw、checkpoint、日志、凭证；覆盖用户已有数据/未提交代码。普通权限确认不等于新增真机授权。

必须遵守：

- 先查 worktree/index/untracked，保留用户修改；不自动 stash、reset、clean、restore 用户文件，不绕过 hooks，不伪造 Git author，不使用 `git add .`。
- 一切 SDK 对象仍在原 owning worker 内；不持 `motion_lock` 跨 SDK、IK、CUDA、磁盘或大 payload 操作。
- 保留 e-stop、故障、有限性/形状、硬机械界限、明确 forbidden geometry、causal no-future、SafetyState/generation、必要 I/O liveness、Teleop VR/live-feedback、旧 tactile 和 policy_eval provenance 合同。
- 明确删除的 quality gate 可以按目标移除；不得为了让测试通过而削弱保留的安全边界或提高限位。
- 不把任务详情/进度/测试日志复制进 `AGENTS.md` 或根 `CLAUDE.md`；它们保持稳定。README/源码注释只更新最终真实行为。

### 1.1 v2 覆盖旧方案的冻结决策

| 决策 | 必须采用 | 不得重新加入 |
|---|---|---|
| 队列满 | FULL = 非阻塞可恢复背压 | FULL 自动 fault/结束试验、丢旧项、无限扩容、backlog-age gate |
| ACK/STOP | 最小 cursor、现有 feedback、必要 generation 确认 | ACK registry、receipt 队列、in-flight registry、每动作双执行器 barrier |
| Policy liveness | process liveness + 同一实验总时长 | 专门续心跳线程、推理 latency deadline |
| Recorder START | 非 RUNNING 中一次有界准备，失败降级 | 预 ACK sample buffer、并发事务、自动重启/重排协议 |
| 证据 | 简短可见日志、必要统计 | 默认完整 source archive、多表 trace 框架、伪造可重建性 |
| 数据 | 现有 raw 布局/语义不退化，显式 derived 转换 | 为字段少而 raw schema 瘦身、改版本号冒充迁移 |
| Home | 实测容差有依据才替换历史 token | 猜 hand tolerance、以 SDK ACK 冒充 measured convergence |

## 2. 开始前与持续进度（T0）

1. 确认仓库根目录、当前 branch/HEAD、`git status --short`、staged/unstaged diff、已有提交；不得在 `main` 上直接实施。仅当实现分支不存在且用户工作树干净时创建独立分支；dirty/分支冲突时请求用户处理，不自行清理。
2. 读取 `AGENTS.md`、`CLAUDE.md`、`README.md`、`pyproject.toml`、相关局部指令和已有 tests；检查 imports/constructors 后再执行 Python 测试。不把 `execute=False` 或 `--help` 当成天然离线。
3. 记录实际 Python/依赖环境，复用用户已激活环境；不擅自升级全局依赖/SDK/固件，不因 import error 全部跳过而宣称通过。缺依赖时报告名称和受阻验证范围，继续独立安全子项。
4. 查默认/解析配置指向的 raw roots（通常 `episodes/`）；用户给出的其他 root 只读访问。不扫描整个 home/磁盘，不找凭证。不见真实数据时标记 `LOCAL_DATA_UNAVAILABLE`，完成可独立验证代码，不编造 legacy mapping。
5. 用已有 report 或一份本地清单记录 episode、schema、行数和可读性。只读打开源；完整 hash 只在迁移前后有必要时做一次，不新增每次启动的全库扫描。历史“61 episodes / 14309 rows”不是当前盘点结论。
6. 冻结实现起点 `IMPLEMENTATION_BASE`（含任务材料、尚无本次生产代码修改的 HEAD）。只读检索所有相关 producer/consumer，先建立最小调用链，不反复通读全仓库。基线无硬件测试记录 pass/fail/skip。

本地进度目录使用 `$(git rev-parse --absolute-git-dir)/research-runtime-refactor/`。只在该新子目录写本次报告，勿操作其他 Git 内部文件。维护一份 `STATUS.md`：base/branch、T0–T7 状态、变更文件、验收证据、commit SHA、阻塞/条件项、下一步。较长日志可放同目录；不要提交真实数据、机器路径清单或测试日志。该记录用于跨会话恢复，不是新 runtime 模块。

阶段状态只用 `NOT_STARTED / IN_PROGRESS / DONE / BLOCKED / DEFERRED_CONDITIONAL`。每个 DONE 必须有文件与实际验证证据；未运行/skip 单独说明。不要为了清单全绿移动验收标准。新的会话先以 Git 状态/源码/测试核实 STATUS，不能盲信它。

## 3. 分阶段实施合同

### T1 — 先保留 raw，删除自动质量筛选，补齐可见日志

**源码入口：** `teleop/control_loop/grid.py`、`teleop/episode_samples.py`、`recording/client.py`、`recording/io_worker.py`、`recording/recorder.py`、`recording/storage/{schema,reader}.py`、`dataset/processing.py`、`deployment/executor.py`、`utils/log.py`。

- 同时改 camera_stall 调用处 `save=False` 和 RecorderIO 将 camera_stall 强制当 error 的分支。停止源行生产、冻结最后 committed sample、drain、正常关闭并验证已采前缀，保存 terminal reason；Teleop control 继续。
- 不因短 episode/min_frames 未达标销毁技术完整前缀。零行不冒充成功。不能补造 camera/tactile、补发历史行或改 source 时间戳。
- 真正 writer/file corruption 不发布为 valid raw；在所有 writer 已确认关闭后，保留本次 owned partial staging 到明确的 incomplete 位置并写失败说明。活跃 writer 未退出不移动/unlink 其数据。只处理本次 staging，不清理用户 episode。显式用户 discard 与自动故障保留分开，勿擅自改变用户明确放弃保存的操作。
- 从 `analyze_episode()` 删除“连续 IK_FAIL > 4 → 整段 rejection”；不删除必要 shape/dtype、时间、媒体、masked tactile 检查和 policy_eval provenance gate。默认保留技术有效且未经显式 curation 排除的所有源行，保持 source-row 映射。
- 维持基线 raw v29 的现有列、dtype 和字段含义。IPC 升级不要求 raw 升级。实际本地 schema 若不同先核对，不硬写版本号。invalid tactile 和非 VR sentinel 的合法 NaN 不被全文件 finite 检查误杀。
- 按第 4 节把真实 action/chunk discard/reject 升到可见日志；不重复打印或新增日志服务。

**验收：** camera_stall 两层不再销毁；少于 min_frames 的有效前缀保留；writer 失败的已关闭 partial 保留且不误报 complete；5 次以上连续 IK_FAIL 不再整段拒绝；源文件不变、current reader 可读；每个真实 drop/reject 有一行。源码、相关 tests、CLI 提示一起提交。

### T2 — 有界 ordered FIFO、背压、generation 与 ACK 一次闭环

**入口：** `ipc/{channels,schema,ring}.py`、`runtime/safety.py`、`control/{action,publication,hand_homing}.py`、`robot/{arm_worker,hand_worker}.py`，以及全部 publication/wait 调用者。必要时增加一个小型 `ipc/command_stream.py`，不重写通用 ring、不引入新调度框架。

**存储与顺序：**

- 一条 coupled record 一次提交；arm/hand 各自按 `read_sequence(next_sequence)` 消费。不存在的 actuator 不进入容量水位；present=False 的记录只推进该 consumer，不造 SDK ACK。
- 在现有短 `motion_lock` 内检查 permission/generation、容量并提交小 payload。设 L 为最后提交 sequence、C 为容量、c_i 为 active consumer 已消费 sequence、b 为本 generation 首 sequence：只有 `L - min(max(c_i, b-1)) < C` 才能写下一条。消费水位更新需跨进程一致，旧 generation 晚到更新不得改变新 epoch 水位。
- EMPTY 是等待。确认仍属当前 generation 且已提交的 resident sequence 不可读/丢失才是 corruption；先在短锁内复核，不能跳到 latest。保留现有 IPC 平台假设，不声称新 lock-free 可移植性。
- 普通 command 去掉 TTL/valid-until/minimum-delivery/latest-ticket authority；只留 generation、关联 action ID、必要 present mask/target 和真正仍使用的字段。queue sequence 由 commit 产生，action ID 可有间隙，不能混用。

**FULL：**

- 第一次准备后的 candidate 是 owner-owned immutable 数值副本，保留同一个 action ID/targets/reference；FULL 时不重新构造、不重新 IK/clip，不 pop、不改 publication clock、不获取替代 chunk。
- 非阻塞返回主循环，以已有 poll cadence 再试；STOP、fault、实验总时长和 Teleop live-input pause 优先。不得 busy-spin、sleep 持锁、加 FULL 次数/排队年龄阈值。
- 成功提交才推进 action index、previous command reference 和 actual-publication cadence。等待期间被 lifecycle 撤销时明确 DROP。
- Teleop/keyboard 保留尚未发布 candidate，同时继续处理 VR/live-state pause；暂停必须撤销旧 epoch 排队/本地 pending，恢复 fresh re-anchor 后不发送旧人类意图。Recording 的 action_queued/held 字段如实表达，不把 FULL 当发送成功。
- Home/replay/calibration 在原操作 timeout/abort 边界内重试同一候选；timeout 只撤销仍属于该操作的 generation，不撤销新运行。

**SDK/ACK：**

- Arm 明确 SDK 接受后推进。Hand 保留既有 command-space slew；中间 setpoint 被接受不推进 endpoint cursor，exact endpoint ACCEPTED 才推进；CRC_UNCONFIRMED 不更新 accepted reference/ACK。每 worker tick 至多一次 send，不一次 drain 全队列追赶。
- 优先在现有 feedback 增加必要 generation/sequence 身份；普通 streaming 不等 ACK、不等物理收敛、不加逐动作 barrier。显式 home/replay/calibration 等待保留。
- `wait_command_accepted()` 不再以更大 action ID 判 superseded。用同 generation 的有序 acceptance 判断；只有在“该 actuator 被 target、前面相关记录无跳过”的前提下，较大 watermark 才证明前驱接受。absent-skip/旧 ACK 不得误满足。
- `arm_last_cmd_seq` 等持久化字段不被悄悄改成别的身份或 sent 含义。新纯 IPC 字段不顺手塞入 raw。

**STOP/restart：**

- 保留 SafetyState/generation 的短锁撤销和 SDK 前最终检查，不持锁跨 SDK，不承诺瞬时物理停止。旧 generation 的队列和未提交 prediction 成批失效且打印。
- Worker 串行调用结束后清理旧 pending；生命周期确需确认时，最多增加现有 feedback 中的 `observed_generation`，在旧调用返回并完成清理后更新。B/home 只在这一生命周期边界检查，不每 action 等待。
- 新 generation 起点取当前 L+1；旧 SDK 回调/ACK 仍带旧身份，不能污染新 cursor/reference。保留 planned-home 独立路径及其真实完成同步，不把它改成普通 stream。

**验收：** 单/双 actuator、absent、慢 hand、FULL→恢复无丢弃/无重复提交、generation cancel、旧 SDK 晚返、CRC、中间/endpoint ACK、真实 corruption。所有 caller 同批适配，无 writer/reader 半升级；对比 seq/targets，不只断言返回 True。明确 pending 与物理完成不同。源码/配置/测试/注释中只为旧 lease 服务的内容同步删除。

### T3 — Single-owner projection、endpoint 简化与 recoverable miss

**入口：** `deployment/executor.py`、`control/{safety_gate,publication}.py`、`robot/command_validation.py`、两 workers、Teleop action proposal/grid、keyboard/replay/calibration 相关调用。

- 提取必要纯函数即可（如 `control/projection.py`）；不建 registry/processor graph。保留既有 canonicalization、operational bounds、soft delta 和浮点边界处理。真实硬限位保持不变，不虚构更宽机械范围。
- 仅移除 worker 对同一 soft jump threshold 的再次 hard reject/revoke。共享 worker 所有 producer 的保护责任必须一起核对；Teleop 保留 human mapping/smoothing/VR pause。Hand 接触 reference 不为模仿 LeRobot 而改成 measured reference。
- reference 仅在成功入 FIFO 后更新。FULL/IK miss/rejection 不更新；drop 未发布 chunk 后缀后，新的预测仍接在已提交前驱后面，不无条件改回 measured qpos。只有新 epoch/明确取消排队后才重建初始参考。
- Ordinary IK 无解或 workspace miss：不发布、打印 action/chunk 原因、丢未发布后缀、同 trial 重观测；保留已入 FIFO 前缀和模型 episode state。不能首次 miss terminalize，也不增加连续 miss 上限；由实验总时长/操作者结束。
- 模型 shape/finite、非法旋转表示、projector 不变量破坏、真实 SDK fault 不能统统吞成 recoverable IK。区分普通无解与内部/contract exception。
- 普通 joint policy 的 dense 0.02-rad workspace critic 改为必要 endpoint 检查；EE 用既有 IK profile 的明确 reference/选择规则，不换 IK 算法。Teleop 重复 segment gate 仅在其未保护额外真实风险时删除；明确 forbidden geometry 和 home/calibration collision path 保留。无法确定保护含义时记录该子项 BLOCKED，不猜。
- 记录 raw EE intent 与 projected joints/其 FK 的区别，不把 intent 称为已执行 EEF；replay 不能通过无声运行时 clip 改写原轨迹，必要转换必须显式。

**验收：** 大 jump 单次 clip、边界 roundoff、FULL reference 不变、IK miss 不终止、前缀连续性、硬边界未弱化。没有两个相邻 helper 对同一软约束重复验证。

### T4 — Policy timing-quality gate 全路径退出；保留 source truth

**入口：** `deployment/inference/observation.py`、`deployment/{executor,lifecycle,config}.py`、`control/publication.py`、`utils/feedback.py`、`sensor/camera/worker.py`、`sensor/pointcloud_worker.py`、`runtime/supervisor.py`、`config/`。

- 一次 query anchor，各模态同一 reference grid。正常 slot 取 newest source<=reference，且 source<=commit<=query anchor；不要求历史 commit<=历史 reference。保留同源 hand qpos/tactile/validity、camera generation、RGB/pointcloud identity 和训练输入顺序/坐标系。
- Warm-up edge repeat 按现有模型合同视为 padding，保留原 source 时间，不能伪造 run-start frame。正常低频 reuse 不报警。必要历史不可得返回明确 WAIT，不能用未来帧或另一 camera identity 拼凑。
- 删除 generic max age/skew/grid-lag、推理后 `_input_is_fresh()`、二次 age check 和 stale escalation。不得以巨大阈值/改名 gate 模拟删除。硬件真实性、因果和旧 tactile all-valid contract 保留。
- 同步删点云构建前后 age drop，拆开 camera finite-but-slow delivery 与 invalid clock/payload 的语义；非法 delay/时间顺序不可一并放行。优先现有 source/health 表示，无法表达时只加一个必要 IPC validity 标志，不建 HealthRegistry，不改 raw enum 旧含义。
- 真实 source stall 归 producer；duplicate 不刷新“新 source 前进”。慢 predict 且 producer 持续前进不算设备故障。设备读取/源停滞阈值不得由模型 latency/recording quality 定义；不臆造硬件数值。Required policy sensor 故障不能被旧 resident valid frame 永久掩盖；evidence-only camera 归 T5。
- `utils/feedback` 分清真实性/因果与必要 live freshness。Policy 只去 age-only veto；Teleop VR/arm/hand live feedback、start/home 的适用检查保留。不要全局删 STALE/timeout。
- 不新增 policy heartbeat 线程。该进程由 is_alive/exitcode 监管；正常 blocking inference 不受 loop-heartbeat deadline。I/O worker 既有 heartbeat 保留。父 supervisor 使用同一 run-start/generation 与 max_running_s 预算，在 predict 阻塞时也能撤销；更新前核对原 generation，避免过期 timeout 撤销新 trial。迟到 prediction 不可发布；Q/e-stop 独立有效。
- 保持 synchronous no-catch-up：上个实际 publication 满一个 dt 才进行下个 dispatch/新查询；infer 后首动作立即尝试，FULL 则保持待提交。成功后的下一次间隔从新的实际 publication 起算，不补发。
- History capacity 改为 source rate×history span 加小 read margin/前驱 slot，注明它是存储覆盖假设，不是 admission deadline。先选 identity 再复制所需 payload，可按重复 identity 复用 copy/resize；不改变 pointcloud preprocessing 方法。

**验收：** old-but-causal 不被丢；future/错 camera generation/invalid modality 仍不可用；慢 inference 不触发质量 gate；duplicate stall 在 producer 暴露；首动作无额外 dt/no-catch-up；源/feedback 真无效不因“删除 stale”被伪装。专属 config/counters/tests 同步清理；仍供 Teleop 的参数留在明确 owner。

### T5 — Recorder 故障隔离，保留单事务和有限收尾

**入口：** `recording/{client,io_worker,recorder}.py`、`deployment/{executor,lifecycle,config,operator}.py`、`runtime/{supervisor,processes,status}.py`、`examples/run_policy.py`，以及 evidence-only camera 的创建路径。

- 不新增异步 START/pending 缓存。复用一次 bounded START；准备在非 RUNNING，失败可见并令本次 evidence unavailable，不自动拒绝有效 B。准备后复查 B/S/Q、generation 和现有 start-state，再开始 RUNNING 计时。不得让等待结束的旧 B 覆盖期间新 STOP。
- START 超时/通道损坏后不复用尚未结束的同一 channel；旧 finalizer 未结束，后续 trial 可不录制，不能开启第二个同 owner writer，也不自动重启。Result queue 始终一个 consumer。
- RUNNING 中采样、writer、result polling、finalization 错误仅改 evidence 状态，不调用 `_finish_episode` 的控制终止路径、quit/session-failed-as-stop 或 motion fault。Required model-input sensor 失败另行处理；不得改成“所有 service error 都忽略”。
- Supervisor/startup/cleanup 必须一起适配 evidence-only role，避免通过 heartbeat/service failure 间接停控制。证据服务停止后仍保留 handle 做最终 join；不要建 failure-policy registry。
- 依旧验证文件关闭与原子发布。Partial 保存遵循 T1；未退出 writer 不 unlink 其共享资源；最终 motion fenced 后沿用 verified shutdown。无法证实清理完成报告 cleanup failure，不虚报 clean，不建 quarantine 管理平台。
- Trial count 归 run owner，真正 begin 的 trial 结束后恰好计一次；B 被拒不计；saved count 独立。`num_episodes`/`max_running_s` 从 recording correctness contract 移到必要的 run 配置（简单参数或已有 dataclass），CLI/YAML/summary 同步。目录名用 trial 序号，不因保存失败重用名称。
- Control reason、recording status、cleanup status 用简单结果字段，不建三个状态机。保持 0/1 退出码风格：证据失败可在控制自然结束后使会话结果非零，但不得因此提前终止 RUNNING；输出三者原因，不把系统 timeout/stop 当 task success。
- 未连硬件前的会话目录/配置创建失败仍可报 setup failure，不借“录制隔离”静默启动未明确准备的机器人。

**验收：** START failure/STOP race；writer/result/optional camera failure 时控制继续；required input failure 不混淆；trial/saved 分开；partial/complete 不混淆；原有限 stop/cleanup 有效。无预 ACK/自动重启新协议。

### T6 — 本地数据可用性、全部 examples、最小研究统计

- `process_episodes.py` 保留 `--output-root`、annotations、dry-run；`export_policy_zarr.py` 增加窄 `--output` 以输出新 generation，默认路径不变、task identity 不改、已存在输出拒绝覆盖。校验目标实际路径/符号链接不落入 protected source，不重写旧 processed/Zarr/checkpoint。
- Current raw 原命令继续可读。真实盘点发现 legacy 时，按已读 schema/实际样本交付一次性离线 converter 或固定 revision reader，附合成/去标识结构 fixture、命令、行数映射和支持边界。不提交真实图像/轨迹。无真实 schema 时不猜转换器，记条件阻塞。
- `replay_episode.py --processed` 仍沿 retained-row manifest→raw source loader→float64 原始命令/时序；导出成功不等于 replay 支持。分别验收训练转换、visualization、raw replay loader；不发送 processed float32 action 来绕过旧 reader。
- Unknown schema 是兼容性问题，不等于 corruption；坏数据也不自动删。Explicit curation 只作用新 derived；policy_eval 不改 provenance 混入 fixed-dt teleop。
- 按第 5 节逐一核对 14 个 examples，新增入口也纳入清单。若未受影响标 NOT_AFFECTED 并给依据，不为凑改动数修改文件。每个实际变更同步 parser/help/config/session/outcome 和对应 comments/docs。
- 必做统计：nominal Hz=1/dt；effective publication Hz=成功 publication 数/RUNNING wall duration；mean/p95 使用真实完成 predict 样本；无样本标 unavailable。不要平均 latest metric 或把 publication Hz 称为物理到达频率。复用现有日志/小结果，不加监控平台。
- 可选的小 decision sidecar 不阻塞核心交付；若实现，仅包含 query/source identity、raw action、projected command、action ID/publication、已实际观测的 acceptance。使用现有 recording owner，不新建进程/高带宽通道，不改 raw 列语义。未观测 acceptance=unknown；无 payload 存档不承诺逐字节重建 observation。

**验收：** 原 raw 哈希/内容不变，可读性不退化；技术有效行数守恒；生成路径/任务语义正确；14 入口逐个有结论；四项统计定义准确；无新 source archive。

### T7 — 条件项、最终复核与本地提交收尾

- Measured start-state 替换 physical_home_completed：只有已有可信 arm+hand targets/tolerance 才实施。保留 H 的便利 home，B 用 fresh measured state；不能把 hand SDK ACK、command bounds 或 arm tolerance 当 hand measured convergence。缺真实依据时保留现有安全边界，标 `DEFERRED_CONDITIONAL`，列具体缺项；不新增一个猜值 gate。真实 planned-home 完成同步不删。
- 旧 tactile all-valid、Teleop 连续反馈错误升级的进一步去重、raw schema 瘦身、新 IK 算法、点云方法、新 tactile mask/train contract 为明确暂缓项；不修改 `dexmani_policy` 私有实现。
- 进行第 6 节复核，更新 README/受影响 docstring/comments/help，只描述已完成且验证范围明确的事实。检查名为 stale/lease/expire 的保留用途，不用 grep 命中数替代语义审查。
- 运行独立最终审查入口，修复有依据的遗漏，按第 7 节提交。记录每个未完成条件，不能将“核心离线完成”写为“全项完成/真机验证通过”。

## 4. 简短且不静默的打印合同

复用 `utils/log.py`（基线 stdout INFO、文件 DEBUG）；行为变化用 INFO/WARNING/ERROR，不只 DEBUG、不另建 logger 框架。以下是格式示例，不是测量值：

```text
[DROP] policy q=41 idx=2 remaining=6 reason=ik_no_solution
[REJECT] arm action=812 reason=joint_limit j=3
[DROP] stop gen=12 arm_pending=2 hand_pending=3
[WAIT] command_fifo full depth=8 keep_action=813
[RESUME] command_fifo wait_ms=75 dropped=0
[CLIP] action=814 arm_max_delta_rad=0.18
[RECORD] trial=3 reason=camera_stall saved_rows=218 control=continue
[RAW] episode=episode_001 schema=26 unsupported_here source_unchanged=true
```

一处决策、一处打印。每个不同 action/chunk 的真实 reject/drop/truncate/fallback/replacement/generation cancel 都可见；若删除了 fallback，不保留它的状态框架。一次 batch cancel 可汇总各 actuator 范围/数量，但不能将两者相加当独立动作数。普通等待只报进入/恢复；没有 action 的 input unavailable 是 WAIT，不叫 DROP。同一次 FULL retry 不刷屏、不重复 CLIP；真实 SDK 异常一次完整错误，普通 clip/miss 不 traceback/打印大数组。离线 CLI 可将诊断写 stderr，仍须可见。

## 5. examples/ 接口适配表（EX01–EX14）

| ID / 脚本 | 必须核对的调用链与语义 | 安全验证方式 |
|---|---|---|
| EX01 `run_policy.py` | run/record config；`--num-episodes` 改为 trials to run；预算/seed/artifact/run_config/summary/结束条件；H/B 说明跟实际条件一致 | 纯 parser + mock inspect/lifecycle；运行主体始终连硬件 |
| EX02 `collect_teleop.py` | config owner/`--print-config`；camera_stall；保留 `--no-hand` 禁录制、`--no-record` 不启 camera/Recorder | 解析配置/mock session；保留 VR freshness |
| EX03 `keyboard_teleop.py` | keyboard_session publication/FULL/pause/generation/拒绝输出 | mock session，不新建控制器 |
| EX04 `replay_episode.py` | raw/processed source loader；新 publish/ACK；FULL 与原操作期限；ReplayStatus/results | 小 fixture/mock；不物理 replay |
| EX05 `calibrate_camera.py` | motion publication/ACK/FULL；保留 `--hand-geometry`/home/collision/标定保存保护 | mock motion；不写标定 |
| EX06 `calibrate_vr_heading.py` | 受影响 config/import/保存路径；不引入 robot FIFO/Policy freshness | 静态/纯 parser；不接 HTS |
| EX07 `process_episodes.py` | IK rejection 提示/统计；annotations/dry-run/output-root；legacy compatibility/只读源 | current/真实 legacy 结构 fixture；行数核对 |
| EX08 `export_policy_zarr.py` | `--output`；默认 task/path；no overwrite；preflight 与实际导出一致 | 小 fixture export，不批量迁移 |
| EX09 `visualize_episode.py` | 旧 raw reader；缺新 decision 不拒绝 raw；`--info` 隔离 GUI | fixture/静态；不重解释字段 |
| EX10 `visualize_episode_processed.py` | generation/source map/task/schema/provenance | `--info` 隔离 GUI/小 fixture |
| EX11 `visualize_policy_rollout.py` | legacy trace 保留；无 trace 的 sync raw 基本 info 或明确转向 raw viewer；有可选 decision 的支持边界 | 有/无 trace 测试；不误报 raw 损坏/不加载模型 |
| EX12 `pointcloud_process_example.py` | camera/pointcloud config/health/import；保留交互 table calibration | 顶层 pyrealsense2；静态/隔离 mock，主体连相机/GUI |
| EX13 `realsense_record_example.py` | camera driver/config/import；保留交互诊断，不强塞生产 Recorder/FIFO | 静态/隔离 mock；主体相机/GUI |
| EX14 `xhand_control_example.py` | HandParams/model 常量和 CRC 未确认；保留 native SDK 诊断定位 | 当前无离线 --help，禁止直接运行 |

禁止 `for f in examples/*.py; do python "$f" --help; done`。只在确受影响入口把硬件/GUI 重导入移到参数解析之后；没有安全 parser 的用 AST 或隔离 mock。接口适配是 CLI/help→config→session/lifecycle→owner→结果/提示闭环，不是仅改函数签名。

## 6. 必要验证与完成矩阵

沿用已有 unittest/fixture。基线 tests 包含 command_feedback、sync_policy_timing、processing/replay provenance、raw_episode_fixture；先检查安全执行路径。允许少量针对性新用例，不建通用验证平台，不按测试文件数量考核。

| 验收 ID | 必须证明 |
|---|---|
| V01 | 当前 raw 不被改写，reader/layout/field semantics 不退化；合法 NaN/mask 不误杀 |
| V02 | camera_stall 两层保存前缀；短有效 episode 保留；corrupt/partial 不冒充 complete |
| V03 | 5+ IK_FAIL 行保留；显式 annotation 才做质量选择；policy_eval provenance 仍拒绝 fixed-dt |
| V04 | 单/双 actuator、absent consumer 正确；相关 command 按 sequence 提交 SDK |
| V05 | FULL 恢复提交同一 candidate，无 pop/reference/clock 预推进、无重复发送/替换；所有 producer 适配 |
| V06 | 稳定已提交 sequence 丢失明确报错而非 latest 跳序；EMPTY 不误报故障 |
| V07 | Hand 中间 setpoint/CRC/exact endpoint 区别，旧 ACK 不满足新 gen 等待 |
| V08 | STOP 在读后/SDK 前/SDK 中、旧调用晚返、新 B/home；generation 确认不进入逐动作 cadence |
| V09 | 大 jump 一次 clip；同一软阈值不双 reject；硬界限/SDK/e-stop 不弱化 |
| V10 | IK/workspace ordinary miss 不终止 trial；已提交前缀/reference 保留；内部错误不吞掉 |
| V11 | old-but-causal 可用；future/错 camera gen/错 source/tactile contract 仍拒绝；warm-up/reuse 正确 |
| V12 | 点云前后/相机间接 quality gate 已退出；duplicate 真 stall 在 producer，慢推理不算源故障 |
| V13 | obs_t→action_t、infer 后首动作、actual-publication dt、FULL 与慢推理无 catch-up |
| V14 | blocking predict 不触发 loop-heartbeat veto；父总预算/STOP 能撤销，迟到结果不发布；I/O liveness 保留 |
| V15 | Recorder START fail/期间 S/Q、writer/result/optional service failure 不停止控制；required sensor 区分 |
| V16 | Trial 与 saved 独立、结果/退出提示一致、有限 finalization/cleanup，不移动活跃 writer 文件 |
| V17 | 真实 legacy（若有）转换和 raw replay source 依赖分别验证；源只读/输出不覆盖/行数可追溯 |
| V18 | EX01–EX14 全部适配或有 NOT_AFFECTED 依据；安全 parser/mock，无 trace raw 不误报损坏 |
| V19 | 每个真实 action/chunk 变化一行；同一 WAIT/retry 不刷屏；无大数组热路径输出 |
| V20 | 四项统计来源/分母正确；可选 evidence 未观测=unknown，不伪造物理完成/输入重建 |
| V21 | README/help/docstrings/comments 与新代码一致；保留 legacy 语义，专属旧 config/tests 删除 |
| V22 | 独立实现分支、显式 staging、无数据/秘密/用户修改进入 commits；最终 diff 与工作树复核 |

用 fake clock/fake SDK/合成数据及必要真实 spawn IPC；多进程交错用 Event/Barrier，避免靠 sleep 碰概率。不跨进程传 live SDK。回归断言动作序列和副作用，不能只 mock 掉被测 owner 然后宣称其通过。

在已检查测试安全、已选好依赖环境后执行：

```bash
python -m compileall -q dexmani_real examples
python -m unittest discover -s tests -p 'test_*.py' -v
git diff --check
```

每个提交后 `git diff --check` 只看工作树不够；最终再对记录的 IMPLEMENTATION_BASE 到 HEAD 检查完整 diff（`git diff "$IMPLEMENTATION_BASE" HEAD --check`）及最终工作树。若用 `tee` 必须保留真实退出码（Bash `set -o pipefail`）。失败、skip、未运行分别记；基线问题与本次引入问题区分。缺依赖先定位根因，不能删验证或扩大 skip 来全绿。

性能仅一次有针对性 before/after：同环境 synthetic workload 下 publication/worker 吞吐、队列最大占用、控制路径耗时、RSS；无锁内 SDK/磁盘/大 payload、无忙轮询、固定内存。不得声称未测量的时延或吞吐保证。

真机、CUDA/真实 artifact、本地全量 raw 转换分开标记。默认不自动跑全量重算/迁移；先 representative read-only preflight 与小测试输出，磁盘预算或所需材料未知时只阻塞对应操作。真机需要用户后续明确授权，任务书/CLI 权限不是授权。

## 7. 注释、提交、恢复和最终交付

每阶段做 `definition→producer→transformation→consumer→side effect` 核对，改一个闭环。并行仅限只读审查或不冲突边界；不要多个 agent 同写 executor/lifecycle/command transport。不新建 agent team 配置。

提交前检查 focused diff、相关测试与 staged diff；用显式路径 `git add -- path1 path2`，再 `git diff --cached --check`/`git diff --cached`。现有 staged 用户修改时先停，不混提交。本地 commits 按可验证行为闭环；transport ABI 的半成品不能单独当可用版本，不 amend/改历史以掩盖失败。修复用后续小 commit。Git identity/hooks 缺失/失败需报告，不自动改配置或 `--no-verify`。

建议 commit 范围（按实际闭环拆分，不机械要求数量）：

```text
fix(recording): preserve collected episodes on camera stalls
fix(dataset): retain technically valid IK-hold rows
refactor(control): use ordered command delivery with backpressure
refactor(deployment): simplify projection and causal policy admission
fix(deployment): isolate evidence failures from control
fix(examples): align research workflow interfaces and legacy readers
```

源码注释/docstrings 和相关 README/help 随每个行为 commit 同步，不在最后补一篇与实现脱节的说明。只解释 owner、时间/字段语义和必要理由，不给每行代码加叙述。保留项目现有术语/语言风格，不把一次事故或本地测试数字写进永久文档。

中断前更新本地 STATUS：已完成的具体代码/commits/验证、未提交 diff、阻塞项、下一最小步骤；留原文件，不为“干净”删除工作。恢复时重读任务书、STATUS、Git diff/log，核实已完成阶段，禁止重新运行未知副作用命令。需要用户输入时只问无法从本地读取解决的具体缺项。

最终交付必须包含：

1. T0–T7 与 V01–V22、EX01–EX14 对照：完成/条件暂缓/阻塞，文件和证据。
2. 删除的机制、保留的边界、新增字段/抽象及必要性；不能只列改了多少行。
3. 实际验证命令/退出码/pass-fail-skip；未测的硬件/CUDA/真实数据范围。
4. Raw 盘点及兼容结论、输出路径/命令、源不变证据；不泄露或提交真实数据。
5. 本地 commit SHA/标题、IMPLEMENTATION_BASE、最终 `git status --short`、未提交用户改动情况。
6. 仍需真实 hand tolerance/legacy 样本/授权硬件验证的项目。核心离线完成不等于整个平台已真机确认。

默认只完成本地 commits，不 push、不 merge。任务目录可由维护者在任务结束后归档/删除；实施中不擅自移除验收依据。
