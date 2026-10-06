# Policy 执行与离线验证

本实现使用一个设备 owner、一个模型 worker 和最多一个未回收 Future。模型 load、warmup、predict、reset 和 close 均在 worker 中串行执行。线程不消除设备 SDK、GIL 或 CUDA 阻塞，也不提供进程有界退出保证。

## 使用与支持范围

入口为 `examples/run_policy.py`。默认 `--execution-mode sync`；async 使用原采样器，rtc 使用前缀条件化 DDIM。三者使用相同的实际控制采样网格。模型观察步数 N、完整 horizon H、执行长度 A 和 dt 来自保存配置；可用 `--n-action-steps` 覆盖 A，P=H-N+1。

必须由实验者指定 `--max-decision-age`、`--max-wait` 和 `--max-tick-lateness`（秒），均为有限正数。前者限制 query 各实际输入源的年龄，第二项限制无可执行计划的等待，第三项定义单槽允许的 owner 迟到量且必须小于 dt。它们与 episode 的 `--max-duration` 是不同预算，没有预设硬件安全值。

async/rtc 还要求显式 `--prefetch-steps d`，满足 1≤d≤A、A+d≤P。rtc 要求 `--rtc-guidance-cap beta`，beta 有限且非负；beta=0 使用原采样。sync 的动作窗口要求为 1≤A≤P。

决策年龄预算还须大于 sync 的 `(A-1)dt` 或 async/rtc 的 `(A+d-1)dt`；这些只是动作段可容纳的必要条件，实际 query 来源年龄仍逐次检查。

sync/async 保留原模型支持范围；rtc 当前接入连续动作 BaseAgent + DDIM（DP/DP3/R3D 路径），支持 joint19、joint+aux EE 和 EEF21。SAT、DQRise、flow 等其他路径会明确拒绝 RTC。启动前 worker 会使用实际加载模型测试选定推理路径；支持 input VJP、保存的 affine action normalizer、diffusers 0.27.2、eta=0 是 RTC 前提。Gaussian action normalization 与 clip_sample=True 的既有拒绝规则仍保留。

加载与 warmup 在连接设备前完成。warmup 包含输入预处理、选定采样路径（RTC guidance 启用时含 VJP）和 CPU 返回；它报告模型路径的时间，并为 async/rtc 给出建议 d，不自动修改配置。实测耗时达到 WAIT 或预取预算时拒绝启动；sync 还检查推理耗时加后续 A-1 槽能否落在决策年龄预算内。真实前缀准备和采样开销另由 owner 事件测量，合成 warmup 不能证明总时延上界。运行中准备超过当前槽的迟到容限使预约失效，EEF 的 d 步 IK 也必须在这一预算内完成，不自动分槽。

一个 session 固定使用启动时选择的模式；切换 sync/async/rtc 需关闭当前 session 后重新启动。每次接纳开始请求仍由串行 worker 执行 episode reset。

## 调度与停止

首次无计划查询使用原采样器，结果准入后在下一个正常槽 b 将 future[0] 重锚定，保留原 query 来源时间。推理期间未执行的动作头不会被裁掉。

async/rtc 在 b+A-d 预取，在 b+A 交接；此后每 A 槽重复。query 槽 q 冻结 q…q+d-1 的目标，交接从结果索引 d 开始。提前完成也等待预约，迟到、owner 漏槽或过期结果均失效。槽内顺序是实际观测与授权检查、结果回收/接管、冻结与提交、实际发送。不会先发送再错误标记前缀起点。

WAIT 从需要动作却没有计划时计时，历史不足、旧 Future 回收、bootstrap、无效重试都不能刷新 deadline。派发返回后，owner 先检查 episode 和仍适用的 WAIT 总预算；只有预算内成功发送或明确非下发模式的逻辑 consume 才清除 WAIT。总预算超时则按 TIMEOUT 撤权、尽力 stop、记录一次真实派发结果后结束，不能再发送下一动作；已锁存的当前 run 首因仍优先。

交接失败后不能继续旧 tail，也不能在忙碌 worker 后排入新 observation。漏槽清空连续历史，不补造采样、不追赶连发。相机低帧率重复源帧只在现有 freshness 条件内有效。

query 保存最新观察槽位每个实际输入依赖的 host monotonic 源时间；FK 继承关节读取时间，cloud 继承 RGB-D 两通道中最旧的推进接收时间，录制专用相机不加入策略年龄条件。结果完成或新反馈不刷新旧 query。这些时间不是设备真实采集时间。输入年龄、当前反馈年龄和槽截止约束新的 SDK 调用资格；episode 和 WAIT 总预算还由 owner 在派发返回后检查，用于决定是否结束运行。

async/rtc 的前 d 步都冻结实际 joint19 目标：Joint 路径先投影，EEF 路径先事务性完成 IK。只有 RTC 生成模型前缀条件；EEF 使用公共 FK 将冻结目标转换为 EEF21，其余 tail 作为软先验。准备不推进 previous accepted。冻结目标在最新反馈下重新检查表示分支、距离、previous accepted jump 等条件，失败不重新 IK 后沿用旧条件。async/rtc 要求两设备明确 ACCEPTED；CRC_UNCONFIRMED、UNKNOWN 或部分失败终止承诺，不重发。

joint 与 EEF 共享绝对关节目标差、operational limits 和 feedback 距离规则。joint 在投影后检查端点自碰撞；EEF 保留 collision-aware 候选选择。frozen 再检查原命令，碰撞查询使用该命令的手姿；禁用手控制时使用配置的手部 HOME 姿态。这些检查不证明连续轨迹或环境避碰，Raw replay 和 HOME 各自保留其独立准入规则。

自碰撞、环境碰撞、MPLib 搜索与规划后验检查使用仓库原始 URDF/SRDF/网格的标称几何。MPLib 搜索使用固定手模型，后验检查使用更新了实际手姿的完整手模型。真实手基座/指尖 FK 单独消费当前安装补偿：打印转接件薄 10 mm，默认 `custom_eef_link -> right_hand_link` 平移为 `(-0.015,0,0)m`；标称模型保留 `(-0.005,0,0)m`。该补偿不改变机械臂 EEF 定义、动作坐标系或碰撞资产。

WAIT 表示没有新策略目标，设备仍可能追踪上一个目标。停止顺序为撤权/失效、设备 stop、录制和模型资源回收；不能撤销已进入 SDK 的调用，也不证明物理停稳。模型任务不会被 Future.cancel 强行打断。close 若仍等待模型完成，会明确记录 pending；Python 进程可能仍等待非 daemon worker。

发送截止使用本机 monotonic 时钟，在每个目标 SDK 调用前判断；进入 SDK 后的迟返回保留实际确认结果。Ctrl+C 保留本次派发状态与已采前缀，软件取消不能撤回已进入 SDK 的调用。

程序接口的 `execute=False` 仍会连接并读取机器人与所需传感器，仅表示不下发策略目标，也不支持录制评估。其逻辑 consume 可以推进计划，但 dispatch 保持 NOT_CALLED；不能用于证明实际执行。CLI `examples/run_policy.py` 固定使用实际执行。离线测试使用 fake robot/观测，不启动 live session。

## 数据与追踪

原始 attempted targets、measured state、dispatch 和辅助 NaN 保留。rollout 的 execution_path 为 `worker_grid_<mode>_v1`。`run_config.yaml` 保存两仓库 SHA、checkpoint/weights/seed/NFE、模式、模型长度和实验预算；每个 Raw 的 `data.h5/meta/policy_trace` 保存 query/交接/dispatch/失效/结束事件，停止失败也沿用该 trace。

记录式 policy evaluation 在 Raw 外保存 `session_result.json`、`attempts/<id>.json` 和 query NPZ。开始请求通过前置检查后、授予 RUNNING 前写 incomplete，阻塞 I/O 完成后重新检查起始观测再授权；取消的 prepared attempt 不消耗运行次数。崩溃留下的 incomplete/entered_running=null 表示未知。无 recording_config 的直接 policy API 不产生持久化结果。

收尾顺序为撤权、尝试 stop、recorder finalize、query sidecar、attempt 终态；只有 recorder 返回实际发布路径才标记 Raw published。query trace 关联窗口来源、selected target 和逐设备 dispatch；预测数组保存在 NPZ。

当前未终结 attempt 的预测，只要形状为 joint 的 `(P,19)` 或 EEF 的 `(P,21)` 且 dtype 为浮点，就保存独立快照，保留原 dtype 和全部数值，包括被拒绝的 NaN/±Inf。可归档不代表可执行：执行仍要求有限值及既有 freshness、授权和调度准入。形状或类型错误只保留诊断元信息，不序列化任意对象；非有限数组不写入严格 JSON trace。读取 NPZ 使用 `np.load(path, allow_pickle=False)`。

晚返回的旧 Future 仅进入 session 诊断，不归入新尝试或回写 Raw，也不为等待它推迟 stop。sidecar 保存失败记入 attempt/session 的 artifact errors，保留首个终止原因，并向调用者报告收尾失败。session 结果在资源关闭尝试后写入，HOME/关闭故障与轨迹结果分别留证。

`python examples/summarize_policy_trace.py <attempt.json>` 离线汇总模型、观测、前缀准备、目标实现、发送和 owner tick 时间。缺失阶段的分位数为 null；嵌套或并行阶段的分位数不相加为总延迟。

第一终止主因通过 run_ended_id 与 run_id 关联，`termination_reason` 表达首因。Teleop 使用可选 `data.h5/meta/termination_details` 保存结束阶段、异常类型和错误说明；附加 stop 故障不能覆盖 OPERATOR/QUIT，正常结束不写错误详情。

owner 在提交 STOP 前深拷贝 trace 和结束详情，writer 排空有效帧后在最终元数据阶段执行严格 JSON 编码。快照或编码失败会锁存错误、保留 staging，并由 writer 关闭资源；重复 close 报告已锁存错误，不重新准备元数据。真正阻塞的原生 I/O 仍可能超过等待预算，owner 不跨线程强关句柄。发布后的资源回收故障通过会话结果与日志表达，不回写已发布 Raw。

Policy 主 Dataset 按实际使用的输入与监督检查窗口有限性，Real canonical 额外要求两设备 ACCEPTED dispatch。观察检查前 N 源行，监督检查完整 H；保持原 padding、episode 边界、loss 权重和时间语义。normalizer 仅拟合有效训练窗口引用的去重源行，验证使用训练统计。checkpoint 部署及恢复读取保存的统计，不重拟合；数据配方变化不能作为旧配方的精确续训，历史复现使用原源码版本。数学和数据筛选细节见相邻 `dexmani_policy` 仓库的 `docs/rtc.md`。

Replay 报告中的 arm `mean_joint_rmse_deg` 是逐关节 RMSE 的平均，hand `pooled_rmse_deg` 是所有帧和关节平方误差的共同均值开方，两者口径不同；历史结果不改写。

Replay 在输出目录 missing-or-empty 检查与轨迹 preflight 后初始化结果记录；trajectory_status/reason 描述轨迹阶段，session outcome/reason 还包含返航、关闭及评估产物写入结果。轨迹完成与最终 session 故障可以同时成立。

## 离线检查与真实评测

```bash
python -m pytest -q tests
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check dexmani_real examples
git diff --check
```

相邻 Policy 的定向测试包括 `tests/test_research_split.py`、`tests/test_policy_windows.py`、`tests/test_streaming_dataset.py` 和 `tests/test_policy_rtc.py`，配套运行已有 training/resume/evaluation 回归测试和 `smoke_test.py --config-only`。测试覆盖生产调度、实际 DDIM、实际 backbone input VJP、公共 FK/IK 动态检查，以及临时目录中的实际 Raw writer；fake device 不等于真机集成。

对照 sync/async/rtc 时固定权重、观察协议、A、NFE、seed 和预算，async 与 rtc 固定相同 d。记录成功率、任务时间、decision age、handoff miss、owner tick、前缀准备时间及接缝变化。当前 sync 在推理期间继续采样；比较历史同步结果时，应固定其源码版本并标注观察时间协议差异。GPU/真实权重时延、闭环收益、物理安全须单独验证。
