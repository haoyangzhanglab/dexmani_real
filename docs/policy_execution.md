# Policy 执行与离线验证

本实现使用一个设备 owner、一个模型 worker 和最多一个未回收 Future。模型 load、warmup、predict、reset 和 close 均在 worker 中串行执行。线程不消除设备 SDK、GIL 或 CUDA 阻塞，也不提供进程有界退出保证。

## 使用与支持范围

入口为 `examples/run_policy.py`。模式从 YAML `execution.execution_mode` 读取，默认 sync，CLI `--execution-mode` 可覆盖；async 使用原采样器，rtc 使用前缀条件化 DDIM。三者使用相同的实际控制采样网格。模型观察步数 N、完整 horizon H、执行长度 A 和 dt 来自保存配置；可用 `--n-action-steps` 覆盖 A，要求 N−1+A≤H，P=H-N+1；覆盖不改写训练配置或 checkpoint，录制时将实际长度、覆盖值和本次桌面平面快照保存到 `run_config.yaml`。

必须由实验者通过 YAML `execution.max_decision_age_s/max_wait_s/max_tick_lateness_s` 或对应 CLI `--max-decision-age/--max-wait/--max-tick-lateness` 指定预算（秒），decision age/WAIT 为有限正数，tick lateness 为有限非负数；优先级 CLI > YAML > 默认，默认预算为空。前者限制 query 各实际输入源的年龄，第二项限制无可执行计划的等待，第三项定义单槽允许的 owner 迟到量且必须小于 dt。它们与 episode 的 `--max-duration` 是不同预算，没有预设硬件安全值。

async/rtc 还要求 `execution.prefetch_steps` 或 CLI `--prefetch-steps d`，满足 1≤d≤A、A+d≤P。rtc 要求 `execution.rtc_guidance_cap` 或 CLI `--rtc-guidance-cap beta`，beta 有限且非负；beta=0 对应原采样分支。sync 的动作窗口要求为 1≤A≤P。

worker 先加载并配置模型，再独立调用 `warmup()`，分别报告 bootstrap 普通路径与 steady 路径。每条不同路径先初始化一次，再测量；正 guidance RTC 使用一次生成的有限 prefix 和配置 delay，sync/async/beta=0 复用普通测量。预热前后 reset，额外预热不改变相同 seed 的首次真实推理。

启动检查与运行时使用相同整数纳秒网格：令 Δ=dt_ns、L=迟到预算，首次动作等待至少 NΔ，WAIT 与有限运行时长必须严格大于它；bootstrap 段末年龄至少 AΔ−L，async/RTC 稳态段末至少 (A+d−1)Δ−L。年龄预算等于下界允许通过，实际 query 来源年龄仍逐次检查。

sync/async 保留原模型支持范围；rtc 当前接入连续动作 BaseAgent + DDIM（DP/DP3/R3D 路径），支持 joint19、joint+aux EE 和 EEF21。SAT、DQRise、flow 等其他路径会明确拒绝 RTC。启动前 worker 会使用实际加载模型测试选定推理路径；支持 input VJP、保存的 affine action normalizer、diffusers 0.27.2、eta=0 是 RTC 前提。Gaussian action normalization 与 clip_sample=True 的既有拒绝规则仍保留。

加载与 warmup 在连接设备前完成。warmup 包含输入预处理、选定采样路径（RTC guidance 启用时含 VJP）和 CPU 返回；它报告模型路径的时间，并为 async/rtc 给出建议 d，不自动修改配置。对普通测量 I，k=floor(I/Δ)+1，首次等待下界为 (N−1+k)Δ，bootstrap 段末年龄下界为 max(I,kΔ−L)+(A−1)Δ。steady 样本 I>dΔ+L 才因无法赶上 handoff 截止而拒绝；I<dΔ 只是余量建议。静态不可能、当前样本不可容纳和建议余量不足分别报告，不自动改变 d、L 或模式。真实前缀准备和采样开销另由 owner 事件测量，合成 warmup 不能证明总时延上界。运行中准备超过当前槽的迟到容限使预约失效，EEF 的 d 步 IK 也必须在这一预算内完成，不自动分槽。

连接不采集触觉 bias。T 表示操作者确认当前无接触，仅在 ARMED、空闲且无录制、模型复位或待完成推理时执行；S/Q/ESC 可取消，且不追加运动。HOME 准入要求会话仍运行、ARMED，且无停止、退出、故障或 estop 请求。待处理 S 由设备 owner 的停止/空闲路径消费，HOME 不清除它；停止处理完成后需重新按 H。

手部 HOME 的下发超时、前次停止未确认属于技术失败，撤权并尝试停止后继续向调用者抛出；不归为用户取消。遥操作启动的 HOME 收敛超时也必须报错，随后到达的 Q 不覆盖该失败。

HOME 一旦进入处理，即使返回未完成，也会忽略同批后续 H/B/T，并丢弃执行期间排队的 H/B/T；结束后需重新发起请求。未获准的 HOME 不额外屏蔽 T；因非空闲被拒绝的 T 不算已执行 HOME。TARE 执行后同批 H/B 不执行。S/Q/ESC 的即时停止与取消不受队列清理影响。触觉缺失继续为 NaN；实际观测构建只对策略必需的不可用触觉字段输出节流警告，说明字段及历史缺测事实，不据此推断是否未归零。关节模型不以辅助触觉为启动门。

一个 session 固定使用启动时选择的模式；切换 sync/async/rtc 需关闭当前 session 后重新启动。每次接纳开始请求仍由串行 worker 执行 episode reset。

遥操作 VR 的新鲜度与下发截止使用 wrist、landmarks 各自接收时间的较早者；HTS 组帧时间取较新者，不能证明两个分量都在更新。任一分量停更不会被另一分量刷新，恢复控制也要求两个分量均晚于恢复边界。head 保留独立接收时间，缺失为 NaN。VR worker 在读取超时或断线后检查退出请求，退出时关闭接收器并清除 readiness；IPC 写入及接口错误向进程监督器传播，不当作坏帧忽略。

当前 MultiTask/text-conditioned 真机入口不支持任务选择与 child 数值配方恢复，连接前明确拒绝；这不限制 Policy 多任务训练或仿真。Dataset 声明必须与实际模型消费字段相等，存储的辅助字段不自动成为传感器门槛。保存点云配方必须完整，用户部分 YAML 仍可覆盖默认值。

`--print-config` 仅打印声明 Real 配置，不加载模型、标定或设备。活动 runtime 在 session 中合并 execution 覆盖及保存 cloud 配方，再构造 worker 和保存 run_config。`policy.recording_enabled/max_record_duration_s` 属于 teleop，部署由 `--no-record/--max-duration` 控制；`policy.ema` 不是模型 EMA 权重选择，也不自动启用策略平滑；`teleop.control_hz` 不覆盖模型 dt。sync 的 d/beta 不参与执行。

| 路径 | 当前保护范围 |
| --- | --- |
| joint 在线 | operational joint limits、相邻命令 jump、当前反馈距离、机器人端点自碰撞；不经过 IK |
| EEF 在线 | workspace 裁剪、候选 IK 的限位/jump/反馈距离与端点自碰撞；冻结目标仍按最新反馈复查 |
| HOME | 当前环境下的路径、workspace、桌面及静态障碍物检查，随后执行与收敛检查 |

在线检查不承诺连续轨迹或环境避碰；存在 environment 配置不表示在线自动应用 HOME 的保护。允许接触规则、标称资产和现场 mount 保持当前配置。

在线 IK 使用当前关节状态、上一发布目标及 Jacobian 零空间候选搜索。候选须通过位姿误差、关节限位、连续性和端点碰撞检查；无可用候选时明确失败。

停止后的 Mode 6 恢复由设备 owner 在派发路径执行；提前恢复是否会复活旧目标缺少固件验证，fake SDK 只能验证调用顺序。EEF prefix 的 d 次 IK 在同一槽内事务性完成，其真机耗时需现场测量。

## 调度与停止

首次无计划查询使用原采样器，结果准入后在下一个正常槽 b 将 future[0] 重锚定，保留原 query 来源时间。推理期间未执行的动作头不会被裁掉。

async/rtc 在 b+A-d 预取，在 b+A 交接；此后每 A 槽重复。query 槽 q 冻结 q…q+d-1 的目标，交接从结果索引 d 开始。提前完成也等待预约，迟到、owner 漏槽或过期结果均失效。槽内顺序是实际观测与授权检查、结果回收/接管、冻结与提交、实际发送。不会先发送再错误标记前缀起点。

WAIT 从需要动作却没有计划时计时，历史不足、旧 Future 回收、bootstrap、无效重试都不能刷新 deadline。派发返回后，owner 先检查 episode 和仍适用的 WAIT 总预算；只有预算内成功发送或明确非下发模式的逻辑 consume 才清除 WAIT。总预算超时则按 TIMEOUT 撤权、尽力 stop、记录一次真实派发结果后结束，不能再发送下一动作；已锁存的当前 run 首因仍优先。更早的 WAIT/duration 截止决定到期来源，同刻按 WAIT 技术失败处理；正常 duration 到期可正常退出。严格分块未确认下发与 WAIT 技术失败在归档后向 session/CLI 传播非零结果，不自动开始下一 episode。实际 stop、录制及关闭错误也会令 session fault，不能被首因锁存吞掉。

下发异常用 `DispatchError.cause` 区分实际授权撤销（`authority_revoked`）、SDK 前或模式恢复后的截止拒绝（`deadline_expired`）和前次停止未确认（`stop_unconfirmed`）；其他设备异常为 null。`revoked` 保持原语义，不能单独据此判定正常取消。只有明确撤权、首因属于当前 run 的 OPERATOR/QUIT、实际 dispatch 仅含 ACCEPTED/NOT_CALLED 且没有独立技术错误，才允许正常结束。ESTOP、CRC/REJECTED/UNKNOWN、停止未确认、设备和收尾错误仍传播失败；sync 无异常时原有 CRC 容忍保持不变。

owner 保存本次下发前的总预算来源及截止、动作准入截止和最终 `valid_until_ns`。async/rtc 因槽、反馈或 decision-age 截止拒绝时，先撤权、stop、记录真实部分结果并归档，再向 Session/CLI 传播 `DispatchError`。返回时 duration 已过期不能掩盖更早的动作截止；只有 duration 严格早于动作截止并实际约束本次调用、且无独立故障时才正常结束。WAIT 与 duration 同刻仍为 WAIT 技术失败。完整 ACCEPTED 的迟返回不追溯改写派发状态；总预算仍按原规则检查。dispatch trace 保存上述截止证据、异常类型和 cause，首因与附加清理错误分别保留，不增加第二套结果状态。

交接失败后不能继续旧 tail，也不能在忙碌 worker 后排入新 observation。漏槽清空连续历史，不补造采样、不追赶连发。相机低帧率重复源帧只在现有 freshness 条件内有效。

query 保存最新观察槽位每个实际输入依赖的 host monotonic 源时间；FK 继承关节读取时间，cloud 继承 RGB-D 两通道中最旧的推进接收时间，录制专用相机不加入策略年龄条件。结果完成或新反馈不刷新旧 query。这些时间不是设备真实采集时间。输入年龄、当前反馈年龄和槽截止约束新的 SDK 调用资格；episode 和 WAIT 总预算还由 owner 在派发返回后检查，用于决定是否结束运行。

async/rtc 的前 d 步都冻结实际 joint19 目标：Joint 路径先投影，EEF 路径先事务性完成 IK。只有 RTC 生成模型前缀条件；EEF 使用公共 FK 将冻结目标转换为 EEF21，其余 tail 作为软先验。准备不推进 previous accepted。冻结目标在最新反馈下重新检查表示分支、距离、previous accepted jump 等条件，失败不重新 IK 后沿用旧条件。async/rtc 要求两设备明确 ACCEPTED；CRC_UNCONFIRMED、UNKNOWN 或部分失败终止承诺，不重发。

joint 与 EEF 共享绝对关节目标差、operational limits 和 feedback 距离规则。joint 在投影后检查端点自碰撞；EEF 保留 collision-aware 候选选择。frozen 再检查原命令，碰撞查询使用该命令的手姿；禁用手控制时使用配置的手部 HOME 姿态。这些检查不证明连续轨迹或环境避碰，Raw replay 和 HOME 各自保留其独立准入规则。

自碰撞、环境碰撞、MPLib 搜索与规划后验检查使用仓库原始 URDF/SRDF/网格的标称几何。MPLib 搜索使用固定手模型，后验检查使用更新了实际手姿的完整手模型。真实手基座/指尖 FK 单独消费当前安装补偿：打印转接件薄 10 mm，默认 `custom_eef_link -> right_hand_link` 平移为 `(-0.015,0,0)m`；标称模型保留 `(-0.005,0,0)m`。该补偿不改变机械臂 EEF 定义、动作坐标系或碰撞资产。

WAIT 表示没有新策略目标，设备仍可能追踪上一个目标。停止顺序为撤权/失效、设备 stop、录制和模型资源回收；不能撤销已进入 SDK 的调用，也不证明物理停稳。模型任务不会被 Future.cancel 强行打断。close 若仍等待模型完成，会明确记录 pending；Python 进程可能仍等待非 daemon worker。

发送截止使用本机 monotonic 时钟，在每个目标 SDK 调用前判断；进入 SDK 后的迟返回保留实际确认结果。Ctrl+C 保留本次派发状态与已采前缀，软件取消不能撤回已进入 SDK 的调用。

程序接口的 `execute=False` 仍会连接并读取机器人与所需传感器，仅表示不下发策略目标，也不支持录制评估。其逻辑 consume 可以推进计划，但 dispatch 保持 NOT_CALLED；不能用于证明实际执行。CLI `examples/run_policy.py` 固定使用实际执行。离线测试使用 fake robot/观测，不启动 live session。

## 数据与追踪

Raw 的字段、单位、时间与缺测语义见 [数据说明](data.md)。录制行对应控制观测与尝试目标，并非每个 action 都有模型 query；rollout 独立录制完整 RGB-D/触觉，公共点云从保存行重建。rollout 的 execution_path 为 `worker_grid_<mode>_v1`。`run_config.yaml` 保存两仓库 SHA、checkpoint/weights/seed/NFE、模式、模型长度、实际解析的 runtime 和实验预算。完整 trace 保存在 `attempts/<id>.json`，记录 query/交接/dispatch/失效/结束事件及停止失败，Raw 不另存 trace 副本。历史 Raw 保持原样。

记录式 policy evaluation 在 Raw 外保存 `session_result.json`、`attempts/<id>.json` 和 query NPZ。开始请求通过前置检查后、授予 RUNNING 前写 incomplete，阻塞 I/O 完成后重新检查起始观测再授权；取消的 prepared attempt 不消耗运行次数。崩溃留下的 incomplete/entered_running=null 表示未知。CLI `--no-record` 与无 recording_config 的直接 API 使用同一执行器，不创建 recorder、Raw、session/attempt、query NPZ 或 run_config。模型需要 RGB/cloud 时仍启用来源；仅关节输入且不录制时不启动相机。默认录制仍要求 Raw RGB-D。`--no-record` 是实际执行，不是离线模式。

Policy/replay 首次创建 `session_result.json` 时原子地拒绝覆盖已有结果；同一会话的后续状态更新仍使用原子替换。

收尾顺序为撤权、尝试 stop、recorder finalize、query sidecar、attempt 终态；只有 recorder 返回实际发布路径才标记 Raw published。query trace 关联窗口来源、selected target 和逐设备 dispatch；预测数组保存在 NPZ。`record_submitted` 只在队列接收成功后记录零起始 `raw_row_index` 与 slot，二者不可互换；`row_count` 是 writer 完成的行数，只有 published Raw 才是已发布行。零行、未下发 target、partial/unknown/CRC 和发布失败证据继续保留。直接构造 runner 时需显式传入 realizer；录制时 recorder/config/results 必须归属同一目录。

当前未终结 attempt 的预测，只要形状为 joint 的 `(P,19)` 或 EEF 的 `(P,21)` 且 dtype 为浮点，就保存独立快照，保留原 dtype 和全部数值，包括被拒绝的 NaN/±Inf。可归档不代表可执行：执行仍要求有限值及既有 freshness、授权和调度准入。形状或类型错误只保留诊断元信息，不序列化任意对象；非有限数组不写入严格 JSON trace。读取 NPZ 使用 `np.load(path, allow_pickle=False)`。

晚返回的旧 Future 仅进入 session 诊断，不归入新尝试或回写 Raw，也不为等待它推迟 stop。sidecar 保存失败记入 attempt/session 的 artifact errors，保留首个终止原因，并向调用者报告收尾失败。session 结果在资源关闭尝试后写入，HOME/关闭故障与轨迹结果分别留证。

`python examples/summarize_policy_trace.py <attempt.json>` 离线汇总模型、观测、前缀准备、目标实现、发送和 owner tick 时间。缺失阶段的分位数为 null；嵌套或并行阶段的分位数不相加为总延迟。

每个实际到期的控制 tick 记录一次 `owner_tick`，归属开始时的 run_id/slot，在所属 attempt 快照前写入 trace。`scheduled_ns` 是计划开始，`started_ns` 是实际开始，`lateness_ns = started_ns - scheduled_ns`，汇总为 `slot_lateness`。`duration_ns` 覆盖本 tick 实际执行的观测、调度、实现、下发及录制提交，直到 tick 退出或开始 attempt 归档；tick 内执行的 stop 计入，tick 退出后的清理不并入。Raw 最终发布、NPZ/JSON 归档与模型关闭不计入。缺观测时不伪造录制行或提交耗时，idle/preflight 和未执行的 slot 不生成 tick。该时长不是完整物理响应时间；历史 trace 保持原口径，不补写已发布 Raw。

第一终止主因通过 run_ended_id 与 run_id 关联，`termination_reason` 表达首因。Teleop 使用可选 `data.h5/meta/termination_details` 保存结束阶段、异常类型和错误说明；附加 stop 故障不能覆盖 OPERATOR/QUIT，正常结束不写错误详情。

Teleop 下发截止过期后停止、保存前缀并暂停，操作者可检查后重新开始；dispatch 技术异常的详情另存 `cause`，不覆盖首因。前次停止未确认、设备错误或停止失败仍向 session 传播。无录制模式也在日志保留具体下发原因。

owner 在归档前冻结 trace；recorder 向 writer 队列提交结束消息前深拷贝结束详情，writer 排空有效帧后对 Raw 结束详情执行严格 JSON 编码。快照或编码失败会锁存错误、保留 staging，并由 writer 关闭资源；重复 close 报告已锁存错误，不重新准备元数据。真正阻塞的原生 I/O 仍可能超过等待预算，owner 不跨线程强关句柄。发布后的资源回收故障通过会话结果与日志表达，不回写已发布 Raw。

Policy 主 Dataset 按实际使用的输入与监督检查窗口有限性，Real canonical 额外要求两设备 ACCEPTED dispatch。观察检查前 N 源行，监督检查完整 H；保持原 padding、episode 边界、loss 权重和时间语义。normalizer 仅拟合有效训练窗口引用的去重源行，验证使用训练统计。checkpoint 部署及恢复读取保存的统计，不重拟合；数据配方变化不能作为旧配方的精确续训，历史复现使用原源码版本。新 Real 训练还默认以 `dataset.max_time_gap_ratio=1.5` 筛选时间连续窗口；显式 null 记录 unfiltered。未知或非正时间即使在 H=1 中也不合格；padding 重复同一源行不产生零间隔。规则与统计写入 data_recipe，恢复时使用保存规则。数学和数据筛选细节见相邻 `dexmani_policy` 仓库的 `docs/rtc.md`。

Replay 报告中的 arm `mean_joint_rmse_deg` 是逐关节 RMSE 的平均，hand `pooled_rmse_deg` 是所有帧和关节平方误差的共同均值开方，两者口径不同；历史结果不改写。

Replay 在输出目录 missing-or-empty 检查与轨迹 preflight 后初始化结果记录；trajectory_status/reason 描述轨迹阶段，session outcome/reason 还包含返航、关闭及评估产物写入结果。轨迹完成与最终 session 故障可以同时成立。

物理 Replay 要求逐行 `dispatch_status` 数据集，每行两设备均为 ACCEPTED 或 CRC_UNCONFIRMED；缺失该数据集不能作为名义目标轨迹放行。该条件不代表设备实际到达目标。

Replay 先保存本次采集的 `replay_data.npz`，再计算、展示和保存指标。派生报告失败仍使 session 报错，但已保存的测量保留。

Replay 的下发异常只有在 cause 为 `authority_revoked`、当前 run 的首因是 QUIT 且没有独立故障时才归为正常退出。稍后到达的 Q 不掩盖下发超时、停止未确认或 capture 失败；capture/stop 错误保留在结束说明中，返回已成功记录的前缀并报告失败。

## 离线检查与真实评测

```bash
python -m pytest -q tests
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples tests
ruff check dexmani_real examples tests
git diff --check
```

Real 测试覆盖生产调度、公共 FK/IK、源帧缓存、IPC、HOME/TARE 批次隔离、即时取消、配置可空字段和临时目录中的真实 Raw writer，并检查终止 tick 在实际保存的 attempt 中的归属和计时。跨仓 RTC 桥接使用相邻 Policy 的 `tests/test_policy_rtc.py`，其中包含 DDIM、backbone input VJP、bootstrap/steady 预热及 reset 检查。Real 的 `test_native_bridge_worker_runner_session_result` 保留实际 LoadedPolicy、串行 worker、Runner、Session 和归档，仅替换硬件边界。训练、恢复和 Dataset 的完整验证按 Policy 仓库说明执行。fake device 不等于真机集成；小型离线数据贯通命令见 [复现入口](reproduction.md)。

对照 sync/async/rtc 时固定权重、观察协议、A、NFE、seed 和预算，async 与 rtc 固定相同 d。记录成功率、任务时间、decision age、handoff miss、owner tick、前缀准备时间及接缝变化。当前 sync 在推理期间继续采样；比较历史同步结果时，应固定其源码版本并标注观察时间协议差异。GPU/真实权重时延、闭环收益、物理安全须单独验证。
