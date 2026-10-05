# Codex CLI 任务书：完成审查整改，收敛机制，优化数据消费

日期：2026-10-05（Asia/Shanghai）  
主仓库：`haoyangzhanglab/dexmani_real`  
协同仓库：`haoyangzhanglab/dexmani_policy`  
状态：**必做软件整改与所列离线验证已完成；逐项结果、合成性能证据、DEFERRED 与未验证项目见文末执行记录。真机未验收。**

本文是独立任务书，不依赖聊天、PDF、临时反例脚本或其他机器上的文件。本任务范围内，本文取代前几轮方案中与之冲突的实施建议。不要重新执行旧根目录任务书的全部事项，不修改 `AGENTS.md` 的长期原则来迎合一次性方案。

## 0. 执行目标与边界

直接实现本文的**必做项**，完成必要的离线验证并给出可审查的 diff。按批次持续推进，不停留在再次提出计划；常规实现选择无需逐批确认。条件项按触发条件处理，不因为没有硬件、GPU、数据或外部仓库就停止所有独立工作，也不能虚报整体通过。

核心原则：安全与科学正确性优先；同一事实由明确 owner 负责；静态成本在合适入口支付；动态有效性在实际消费/副作用边界判断。保留现有单设备 I/O owner、单模型 worker、一个在途 Future、不可变命令、传感器进程/共享环、Raw 写盘线程和 Zarr 数据链路。

审查基线：

| 仓库 | 实现基线 SHA |
| --- | --- |
| Real | `660aa20f52eee7bbbfa387d692f0715aca17502f` |
| Policy | `650f1e3cfc63c64678b5bb69e7258b548800d053` |

SHA 用于定位证据，**不是 reset/checkout 目标**。开始时检查当前分支、工作树、各级 `AGENTS.md` 和基线之后的相关差异；保留用户已有改动。当前代码若已修复某项，给出证据后标为 `ALREADY_FIXED`，不要为符合旧结论重新改写。任务书自己的新增提交当然会使 HEAD 不同于实现基线。

仓库分工：设备、控制、标定、采集、Raw、导出属于 Real；Dataset、normalizer、训练和模型采样属于 Policy。优先使用已存在的两个工作树并确认 remote；缺少 Policy 时，可以在允许的工作区从 `https://github.com/haoyangzhanglab/dexmani_policy.git` 建立独立 checkout，不覆盖已有目录、不改 site-packages。若访问/权限不足，完成 Real 与可离线验证的部分，单独报告 Policy 阻断。

本任务授权代码、文档和无硬件离线验证。**不连接机器人、RealSense、VR/HTS，不执行 HOME、replay、rollout、采集或真实标定写入**；`execute=False` 也不天然无硬件副作用。允许测试临时目录中的虚拟标定文件与合成数据。不启动完整训练/DDP/长时间评测，不升级实验环境来凑测试，不修改既有 Raw、checkpoint、实验目录或无关缓存。默认不自动 commit/push 实现改动；交付 diff 与验证结果。

用户明确要求本任务书位于根目录，保留该位置。长期 README 只更新发生变化的稳定接口/用法，不塞入本次完整审查台账；不批量删除旧任务书或修改权限配置。

## 1. 本次方案复核后的最终调整

| 改动包 | 最终裁决 | 相比上一版进一步收敛/补全 |
| --- | --- | --- |
| D1 发送与交接 | 实施 | 截止时间只约束新的 SDK 调用资格；迟返回不能改写已知 ACCEPTED。未来 RTC 目标不携带本次反馈截止。 |
| D2 取消与证据 | 实施 | 使用局部携带结果的中断异常和单次记录出口；不新增全局结果槽、事件总线或通用终止框架。 |
| D3 相机与参数来源 | 实施 | 复用两路推进时间同时判断 freshness/stall；复用现有 CameraExtrinsics 快照，仅消除重复读文件。 |
| D4 触觉 | 实施 | 每路 bias 本身表达校准状态，候选验证后提交；不另存一套 calibrated 真值。 |
| D5 几何/HOME | 实施已证实部分 | 修绝对坐标和软逃离；B1 不改数值，也不预建运行期 URDF 生成机制。 |
| D6 标定 | 实施 | 复用 atomic_json_dump，最后一次取消检查作为提交准入；不持有 motion_lock 执行文件 I/O。 |
| D7 视频/诊断/指标 | 实施 | CFR 帧身份用有理时间；保留顺序解码；指标只澄清口径。 |
| D8 数据消费 | 实施 | 补齐 normalizer 的全量汇集；时间语义写文档/日志，不新增会破坏旧 resume 合同的 metadata 键。 |
| D9 精简 | 实施有明确证据部分 | 保留连接前验证及独立 public Runner/driver 的边界；不新增 validated token 或 trusted bypass。 |
| D10 条件优化 | 条件项 | 只有实际成本与正确性证据时做；不是必须重写 TAG、HOME、点云和 RTC 的清单。 |

新确认的 D8 落地缺口：Policy `training/build_utils.py:build_normalizer` 先 `list(dataset.iter_normalization_data(...))`，`action_ee/auto` 还 concatenate 全部；`agents/normalization.py:LinearNormalizer.fit_field_chunks` 再 `chunks = list(chunks)`。函数名称/注释不等于真正流式。只修改 generator 会保留全量内存占用，必须贯通修复。此项补全 O01，不把它混称为新增控制安全 bug。

另一个兼容性修订：`training/resume.py` 严格比较 data_recipe 的键。此前建议仅为标签新增 `temporal=recorded_rows` 会使旧 checkpoint 的恢复合同无谓失配；本次改为文档/日志说明，保持持久化配方不变，不再为标签新增兼容分支。

## 2. 保持不变的系统合同

1. `RobotCommand` 的目标数组 owned、不可变，action 是尝试发送的绝对目标；SDK `ACCEPTED` 表示接收确认，不表示物理到达。逐设备 `NOT_CALLED/ACCEPTED/CRC_UNCONFIRMED/REJECTED/UNKNOWN` 的现有语义保留。
2. 先验证所有存在的 arm/hand 目标，再进入第一个 SDK；逐设备 run_id/safety/quit/stop 检查保留。motion_lock 不跨越 SDK 阻塞调用。停止未确认状态不得被新动作绕过。
3. 反馈缺失当次就不发动作；持续失败超时升级 FAULT。软件检查不能撤回已进入 SDK 的调用，也不提供硬实时保证。
4. 时间用同主机 monotonic；host SDK read completion、camera receive 与设备曝光不同。query 最新输入槽的实际依赖源时间冻结，不能被推理完成、新反馈、FK 或点云计算完成时间刷新；历史槽不套用当前最新帧年龄门限。
5. Raw 保留已采前缀、失败末行、源时间/帧号/dispatch。仅显式 discard 删除当前 capture；writer/编码/发布失败保留 staging。已发布 Raw 不原地改写。
6. 全模态公共导出保留现有 13 字段；模型只消费需要的字段。辅助缺测用已有 NaN/有效性表达，不新增全字段 validity 系统，不伪造黑图/零力以延续控制。
7. 桌面/安装按当前配置重建；触觉导出直接消费保存值，不再次扣偏置。相机保留数据对应内外参和 depth scale。无历史 calibration ABI、指纹准入或迁移框架。
8. 默认执行模式仍为 sync，async/RTC 的现有算法、预算、slot、prefix 和 warmup 语义保持；没有自动 hold/平滑/补偿/恢复的新控制行为。
9. 训练的 N_obs/H 角色窗口、episode split、padding、Real both-ACCEPTED、唯一训练源行统计、val 复用 train normalizer 保持。旧 checkpoint 使用保存的 config/normalizer，不静默重拟合。
10. 在线端点自碰撞与 HOME 的环境/路径检验是不同能力；不把在线动作宣传为全环境连续避碰。

## 3. D1：统一发送截止，修复 Future 交接

覆盖：R3-1（P1）、R3-2（P2）、O12 已实现部分的保留。

### 3.1 文件与根因

Real `robot/robot.py`、`robot/commands.py`、`runtime/observation.py`、`teleop/control/controller.py`、`teleop/keyboard_session.py`、`calibration/camera/motion.py`、`replay/replayer.py`、`robot/hand_homing.py`、`deployment/{observation,runner}.py`。

Teleop 输入在读取时新鲜，但 TAG/IK 后缺少发送时的同一输入年龄约束；旧假时钟反例在准备耗时 500 ms、反馈预算约 133 ms 时仍调用两个 SDK。Policy 的 Future 可在 `_read_observation` 内完成，却因交接前没再次 poll 而被当成 handoff_miss。

### 3.2 发送接口

- `send_action(command, *, valid_until_ns)` 使用必填关键字截止；内部 `_send` 逐设备在所有可能耗时的准备/服务检查之后、目标 SDK 之前检查。arm `enter_mode6` 前后均保留相应授权/截止边界。
- `valid_until_ns` 定义为**第一个禁止开始目标 SDK 调用的 host monotonic ns**，`now >= valid_until_ns` 禁发。将现有 `<=` 年龄/slot 上限转换为排他界时加 1 ns；现有 episode/WAIT 的 `<` 上限不加。沿用现有预算，禁止新造默认 TTL 或放宽阈值。无适用预算的项不加入 min。
- 调用方计算当前派发的最小截止；输入时间为 0、未来值或缺失时，在已有采样/接纳检查拒绝，不靠简单 `timestamp + budget` 将其变成合法。

| 路径 | 本次截止来源 |
| --- | --- |
| Teleop | 本次动作实际依赖的 arm/hand 反馈与 VR 来源，加各自现有 age budget；纯录制相机不自动变成动作依赖。 |
| keyboard / camera jog | 当前求解与安全判断所用反馈的有效期；保留计算过程中的撤权检查。 |
| Replay / warm-up | 当前反馈有效期和现有适用运行截止；历史轨迹日期不是动作过期条件。 |
| Policy | 冻结 query 最新输入槽的各实际来源到期、当前反馈到期、当前 slot 最晚发送、episode 与仍适用的 WAIT 截止的最小值。 |
| hand HOME | 独立 ARMED 入口沿用已有 HOME 总截止，不要求不存在的 Policy plan；实际到达仍看新反馈收敛。 |
| arm HOME | 保持现有专门规划、abort、反馈收敛流程，不强行改造成流式命令。 |

- `RobotCommand` 仍只拥有目标及 run_id。RTC 冻结的是未来目标；**每次真实发送才计算反馈截止**，不在 prefix 预备时封存瞬时有效期。
- 第一个 SDK 返回后已过期，第二个不得调用；已发生结果照实保存。SDK 在截止前进入、截止后返回时，不能因迟返回把 ACCEPTED 改成 UNKNOWN/NOT_CALLED。调用资格截止不是完成/物理到达截止。返回后的授权撤销仍处理；episode/WAIT 总预算由 owner 及时终止，不让返回后的超时伪装成 SDK 拒绝。
- 所有调用点迁移后删除 `before_send`、`dispatch_row` 及仅服务该回调的 `_dispatch_allowed`。保留 owner 的接纳、WAIT、时长和计划失效决策，避免 Robot 通过回调修改 Policy 状态。若需要复用，仅加小型时间计算 helper，不造 Budget/Lease 框架。

### 3.3 Future 回收

在 `_read_observation` 返回后、handoff 判断之前非阻塞回收当前 Future；回收后重新读取 query，检查 run/query 身份、valid、slot 与年龄。不额外等待、不扩大 lateness，不重复消费、不让旧 run 结果污染新 run。保留原 tick 开始的 poll 以处理已完成错误/配置操作。

### 3.4 验收

假时钟覆盖：准备越期、mode restoration 越期、arm SDK 越期导致 hand 未调用、最后一个 SDK 迟返回但结果保真、恰好到排他截止、两 SDK 之间撤权。覆盖所有生产调用点及 HOME 专用入口。Future 在读观测 +5 ms 完成、仍处于 30 ms 预算内时可交接；超期、过 slot、失效 query 仍拒绝。

## 4. D2：取消不丢派发、不丢前缀、不覆盖首因

覆盖：N1（P1）、N2（P1）、N3（P2）、O02。

改动面：Real `robot/robot.py`、`teleop/control/controller.py`、`teleop/runner.py`、`deployment/runner.py`、`replay/{replayer,session,capture}.py` 与当前实际记录接口。

1. 增加小型 `DispatchInterrupted(KeyboardInterrupt)`，仅携带 `DispatchResult`。`_send` 明确捕获 KeyboardInterrupt 并包装；进入 SDK 前标记 UNKNOWN，已经确认的结果不回退，未调用保持 NOT_CALLED，补实际 host 完成时间。普通 Exception 继续走 DispatchError。不要吞掉所有 BaseException；已有 finally 清理捕获与新业务异常捕获分开看待。
2. 三条生产控制链显式处理该异常：锁存取消/撤权，由唯一 I/O owner 尽力 stop，然后在 finally 中保留当前 attempted row，再进入取消收尾。stop/recording 失败作为附加错误，终态可为失败，但不能篡改 dispatch 或抹掉原取消原因。
3. 将当前调用的成功/失败/取消分支记录收敛到一处，确保一行只提交一次。优先局部控制流，确有分支需要才用局部标志；不建通用 finalizer，不新增全局 last_result 或 SDK 事件表。没有 ObservationRow 的 HOME/warm-up 中断仅记录真实派发明细，不伪造训练行。
4. Replay 在 replayer 内把 Ctrl+C 转为 ESTOP 结果，不在构造 outcome 前重抛。finally 撤权/停止后执行 `capture.to_dict()`，返回 `ReplayOutcome` 给 session；session 完成评估/保存。CLI 非零退出发生在数据交接之后。不要把旧缺陷误说成没有执行 stop。
5. Teleop 采用与 Policy 相同的首因规则：仅使用匹配当前结束 run_id 的 `run_ended_reason`；没有锁存首因才取本地 fallback。S/Q 不被 `motion_revoked` 覆盖；停止/写盘/关闭错误进入 detail；完成计数与保存只做一次。不要重新改写已经修好的 Policy 首因路径。

验收：分别在 SDK 前、arm SDK 内、arm 成功后 hand SDK 内、Replay tick 等待期间注入 Ctrl+C；确认调用次数、UNKNOWN/ACCEPTED/NOT_CALLED、末行与前缀数量、session 收到的数据。检查 S/Q 与后续 stop failure 共存、旧 run 原因不串到新 run、writer 失败保留 staging。突然断电/SIGKILL 不在本次取消交接承诺内。

## 5. D3：RGB-D 来源年龄与同会话相机参数

覆盖：B2（P1）、O05、O09；不新增全局同步机制。

改动面：Real `sensor/camera/worker.py`、`sensor/pointcloud_worker.py`、`runtime/observation.py`、`ipc/camera_ring.py`（仅消费合同核对）、`recording/{recorder,frame}.py`、`calibration/camera/extrinsics.py`、相关 session/runner 构造链。

1. worker 用两路 `last_advance_ns` 替代现有等价的 `last_good_s`，不同时维护两份事实。某路 frame number 改变才更新该路时间，时间值取本次 host receive/queue-return。组合 `timestamp_ns = min(depth_advance_ns, rgb_advance_ns)`；保留两路 frame numbers 和发布 sequence。
2. 同一组推进时间也用于现有 source_stall_timeout。age budget 决定能否消费，持续 stall 决定何时升级服务故障，不能删成一个门限。未知源身份不能伪装为推进；重启处理遵循既有会话重建，不引入自动重连。
3. pointcloud 继承源相机时间与 sequence；处理/写环完成不能刷新源年龄。允许 ring sequence 前进而源时间不前进；消费者按 sequence/源身份去重并独立判断 age。RGB/cloud exact-sequence 匹配与 owned copy 保留。
4. 更新新 Raw `camera_timestamp_source` 的明确含义为组合最旧通道的 host 接收时间；旧 Raw/metadata 不改。该最小设计仍是 RGB-D 合同，RGB-only 也受较慢通道约束；只有明确实验需求才另设独立 RGB 流。不是硬件曝光同步。
5. O05 复用已有 `CameraExtrinsics` / `PointCloudWorkerConfig` 预加载能力。一次 session 构造解析相机外参，向 pointcloud 与 recorder 传同一只读值；recorder START 用实际 serial 选条目，不再次读取文件。无 cloud 的录制 session 也要明确一次读取位置。外参缺失按现有 Raw 允许缺失规则处理，cloud 必需外参仍在其原边界报错。intrinsics/depth scale 来自实际相机启动信息。
6. 已解析的 table/mount 继续用当前 runtime，不再造 `SessionGeometrySnapshot`、版本指纹或历史资格层。会话期间外部改文件在下次会话生效。
7. O09 搜索 mount metadata 的全部消费者；确认只是诊断后，复制已校验当前值即可，不再作为 recorder START 的额外几何准入。保留独立公开 Raw writer 必须承担的布局/数值边界；若去除某字段，生产者/消费者/文档一次完成，不能误删 camera 几何。

验收：两路分别冻结、共同冻结、正常异步推进；旧通道过 age 后观测和 query 不再获准，2 s stall 仍各路有效。检查 camera→cloud→query→Raw 时间/身份链。session 中改外参文件不能让 worker 与 recorder 使用两份值；未标定 Raw 采集能力不被额外 gate 阻塞。

## 6. D4：触觉两路独立，候选验证后提交

覆盖：B3、B4（均 P2）。

改动面：Real `robot/drivers/xhand.py` 的 capture/verify/get_state 与 `robot/robot.py:_read_hand`；复核 Policy 输入、Raw 和 export 的既有每路 valid 消费。

- 每路 bias 为 None 或已验证数组，是唯一校准状态。raw-valid 与 bias 是否存在分别决定该路最终 usable；缺测为该路 NaN/valid=False，不吞掉另一通道，不阻塞独立的有效关节反馈。
- 重校准开始清空本次 bias。沿用有界采样批次与间隔，分别累计两路有效样本并判定是否足够；不支持 dense 的设备仍可完成 aggregate，不无限等待/重试。
- 候选只保存在局部变量。verify 使用新鲜原始数据减局部候选，不先写进 `self` 再通过 get_state 验证；防止重复扣偏置。
- aggregate 继续用现有无接触残差阈值；dense 保持结构/finite 检查，不添加未经标定的逐 taxel 力阈值。可同一批读取、逐路判定；单路数据无效不以 AND 使另一有效通道失败。
- 正常完成验证后只提交通过的通道；异常不发布未验证候选。返回由提交结果推导的 `(aggregate_ok, dense_ok)`，删除联合 calibrated 属性及消费点，不存第二份布尔状态。连接层日志说明哪路不可用。
- 整个 SDK 读取异常或关节反馈不可信时按既有失败语义处理，不能为了保留某一路候选压下设备异常。操作者保证空载无接触；软件不能凭未校准绝对力证明无接触。

验收：aggregate-only、dense-only、both、neither；单路坏帧/样本不足/超阈值；verify 抛异常；重校准中断。已知 SDK 状态 1501019 的 aggregate 有效/dense 无效场景不能失去 aggregate。失败路径无未验证 bias 可见，模型只被其实际依赖模态限制。

## 7. D5：绝对关节坐标与 HOME 全程软逃离

覆盖：R3-3、R4-2（P2）；B1 与 O08 的条件裁决见第 12 节。

改动面：Real `planning/kinematics/{ik,ik_geometry}.py`、`robot/{action,projection}.py`、`planning/paths.py`；核对 `tests/test_policy_eef.py`。

### 7.1 绝对坐标

IK 先在 operational limits 内选择机械等价支并冻结 q_cmd；发送风险比较用：

- jump：`abs(q_cmd - previous_cmd)`，限制命令变化；
- hw_dist：`abs(q_cmd - measured_q)`，限制实际反馈到绝对目标的距离。

两者依赖不同参照，继续保留。wrapped distance 只保留在明确具有周期等价含义的求解/排序中；检验后不得再次投影到另一支再发，RTC 冻结分支也不能被重新求解改变。检查所有调用者和最终投影的一致性。

当前 hw_dist=150°，绝对距离通过后各关节差小于 π，原 `band_switch` 启发式不再提供独立覆盖，可以删除，并更新对应原因统计/测试/日志。不要把“期望返回旧 reason 字符串”当作不应修复的行为。若当前代码阈值已变至 π 以上，重新证明后再删，不能照搬本次冗余结论。

验收：限位 [-180°,180°]、measured=-106°、previous=101°、candidate=106° 必须拒绝（绝对 212°，原 wrapped 148°）；另测合法近邻、±π 附近、收窄限位、previous 与 measured 分离和 frozen prefix 分支。

### 7.2 软逃离

用 `best_clearance_so_far` 替代逐相邻段借用 epsilon：尚处于初始负间隙前缀时每点必须 `clearance >= best - epsilon`，之后更新 best。复用现有 0.5 mm，不增加新配置。负值只能是初始连续前缀，最终必须非负，脱离后不得回入。硬碰撞/硬间距继续独立否决，不被软逃离覆盖。

验收：-1 mm 开始、每步 -0.4 mm 累计至 -7 mm 再回 0 的旧反例应拒绝；真正逃离、epsilon 内小扰动、脱离后回入、硬碰撞各有正反例。只修验收，不同时重写 RRT/shortcut/固件 waypoint 执行。

## 8. D6：标定取消、活动配置和最终结果验收

覆盖：R4-1、R4-3、B5、N4（P2）。

改动面：Real `calibration/camera/{session,solver,motion}.py`、`calibration/table.py`、`teleop/config.py`、`utils/atomic_io.py`。

### 8.1 取消与提交

1. loop 先处理已锁存 Q/ESC、键盘失联和服务故障，再处理 capture/solve 事件；每条事件前重查，取消后不继续消费排队 SPACE/ENTER。
2. `_detect_aruco_stable` 接收简单 `cancel_requested`，在每次等待以及耗时检测前后检查；沿用 10 ms 等待粒度，不等满 2 s。检查/停止仍由既有 owner 完成，输入线程不直接调用 SDK。
3. 拆开求解候选与发布文件；ENTER 仍先 revoke/stop。原生 solve 无法抢占时允许返回后丢弃候选，不为此新增 solver worker。
4. 复用 `atomic_json_dump` 的同目录临时文件与原子 replace。可仿照已有 `atomic_publish(cancelled=...)` 增加可选取消回调，在临时文件关闭后、replace 前最后检查；传入回调检查已有取消/故障/会话 epoch。无需新事务类或取消状态机。
5. **提交语义**：最后一次取消检查通过即进入不可回滚的提交阶段；此前已观察到取消不得发布，之后与 replace 竞速的取消不保证撤销提交。完成的文件与 `calibration_saved` 按实际提交结果报告。不要宣称键盘物理事件与文件替换严格同时排序。
6. 不持有 motion_lock 做求解、写文件、fsync 或 rename；运动已在求解前停止。写入失败/取消清理临时文件，既有标定文件保持可读；其他相机条目保留。所有验证只写测试临时目录。

### 8.2 活动配置

camera calibration 公开 session 在设备连接前验证实际使用的 KeyboardTeleopConfig：hz/idle_hz 有限正数，margin 有限非负，每轴 `lower+margin < upper-margin`。验证当前选中配置，不用未启用后端阻塞入口。D9 的去重不能删除这道副作用前检查。

### 8.3 候选与最终门限

- B5：solver 对已计算候选先按 finite/形状、位置 RMS 与旋转 RMS 的现有双门限筛选，再在合格集内取最小位置 RMS；同分时旋转 RMS/稳定方法顺序打破平局。无合格候选不保存。门限来自现有配置，不添加混合权重或校准等级。候选数值与 residual 只计算一次供显示/准入复用。
- N4：最终 refit 平面重新计算 inlier mask、count、ratio 和实际发布质量量；用最终结果通过所有既有门限。中间 mask 仅为拟合服务。

验收：已置取消与排队 ENTER、采样等待中取消、检测/求解返回后取消、写临时文件后取消、提交开始后的取消；后者保持完整文件但不承诺回滚。非法配置在 fake connect 调用计数为 0 时失败。A=(1 mm,10°)、B=(2 mm,1°)、门限=(5 mm,3°) 选 B。最终 838/1500 < .561 拒绝。旧等待反例不是已经证明机器人持续运动 2 s。

平面测试用确定性合成点或受控 refit 构造“中间合格、最终 mask 不合格”的临界案例，不依赖未附带的历史 seed 脚本。用户取消单独表示，不伪装成质量不合格或硬件故障。

## 9. D7：视频帧身份、诊断结果与指标口径

覆盖：B6（P2）、R4-4（P3）、RMSE（非算法 bug）。

改动面：Real `recording/storage/video.py`、`recording/storage/reader.py`、`examples/realsense_record_example.py`、`replay/evaluation.py` 及实际显示/存储消费者。

### 9.1 CFR 随机帧

- 全程保留 PyAV Fraction rate/time_base。明确首个展示帧 origin_time（同样为有理时间），用有理运算计算目标 PTS，向前 keyframe seek 后按展示次序解码。
- 用 `(frame.pts * frame.time_base - origin_time) * rate` 映射帧 ordinal，确定性舍入并检查量化误差与唯一帧身份；容差来自已知时间基量化，不能任意加 epsilon。对当前 CFR 合同，量化必须足以区分相邻帧。
- 只返回目标 index；缺 PTS、不可映射或跳过目标时明确失败，不能静默返回下一帧。average_rate 不是 time_base 的 fallback。encoder 明确 pts/time_base 对应关系，保持当前编码参数；兼顾已有合法 CFR 文件。
- 顺序 export 继续 `iter_frames()`，不引入全视频 RGB 缓存、完整 VFR 服务或无必要的全帧索引扫描。

验收：30 fps、1/15360 time_base 的第 123 帧不再返回 124；在实际 PyAV 中以同一文件顺序解码为 oracle，覆盖整数/分数 fps、首尾、GOP 边界、B-frame、逆序/重复请求、非零 origin。比较解码帧，不要求有损编码像素等于原始输入。

### 9.2 诊断与评估

diagnostic loop 返回完成/正常用户退出/读取失败的明确结果，main 只在成功时打印完成，故障非零退出，finally 仍 disconnect。验证只用 fake camera。

RMSE 保留既有计算，明确 `mean_joint_rmse` 与 `pooled_rmse`；如提供跨 arm/hand 可比指标，用同口径命名。同步实际报告消费者，不重写旧存档或宣称某种合法统计量算法错误。单轴 7°、其余为 0 的 7 轴例分别是 1° 与 sqrt(7)°。

## 10. D8：真正贯通的按需读取与流式统计

覆盖：O01、O15、O16；X1 已修行为保留；R3-4 只明确配方，不自动改变训练资格。

改动面（Policy）：`datasets/{replay_buffer,base_dataset,sampler,multi_task_dataset}.py`、`training/build_utils.py`、`agents/normalization.py`，搜索其他真实调用者后同步接口。

### 10.1 有界读取

1. 只读打开所选字段，停止加载整个高维 payload。可以将 `copy_from_path` 改为语义明确的 `open`，沿当前 ReplayBuffer 最小改动，不创建 backend registry/通用存储接口。现有真实需要的 NumPy 字典构造仍可保留。
2. 内存常驻限于 meta、原始源行索引/布尔 mask、当前 chunks、窗口与预取数据；metadata 验证在公开构造边界完成。字段 shape/dtype/episode_ends/dispatch 等不同事实不能按“重复次数”混删。
3. Zarr 句柄按进程打开。spawn 序列化不带活动句柄；fork 后按 PID 重开。清除依赖旧句柄的 cached_property/data/root 别名，不能只重开 `_group` 而让 sampler 仍持有旧 Array。当前训练期间不替换所读缓存；无需每个样本做内容 hash 或重新扫 metadata。
4. sampler 沿现有 source_rows/episode/padding 映射读取；观测只需 N_obs 行，监督 H 行；同字段承担多个角色时读并集。保证 padding 数值与样本 shape 与现有 consumer 一致；必要时只在 BaseDataset 的最短调用链传角色长度，不让所有用户迁移复杂字段协议。
5. 读取窗口/chunk 时执行与旧逻辑一致的 float32 转换；uint8 RGB 不提升精度。finite 判断针对转换后的数值，避免 float64 有限但转 float32 溢出时资格改变。返回给 augmentation/模型的数组拥有明确所有权，不为少一次 copy 引入共享可变别名。

### 10.2 完整统计消费链

1. selected modalities 的 finite/dispatch 资格分块扫描，保留现有角色化有效窗口；先 episode split，再取最终有效训练窗引用的唯一 observation/action 源行。不能把所有观测都扩展到 H，也不能用 deployment A 代替监督 H。
2. `iter_normalization_data` 按 chunk 与唯一源行交集 yield 有界数组；不对整列做 `arr[all_valid_rows]` 或 `action_ee[..., :9][rows]` 的全量中间拷贝。不重复计算 padding 行的统计权重。
3. `build_normalizer` 直接传单遍 iterator；去掉先转 list、检查 len 再选一次性/流式拟合的分支。不用 `tee` 或其他隐含缓存保存已消费块。
4. `fit_field_chunks` 改为真正单遍：在首个有效 chunk 初始化固定尺寸统计量，逐块合并 Welford/min/max，不保留 first/chunks 的大数组引用。保留 last_n_dims、dtype、count、limits/gaussian、常量维度和输入统计的既有语义；空迭代器/全空块/gaussian 不足 2 样本仍明确报错。
5. `action:auto` + `action_ee` 的混合 normalizer 同样流式：按块累计所需维度统计，再组装 xyz/hand 的 limits 与 rot6d 的 identity。保持原 scale/offset、rot6d 统计占位值和 action/aux EE 切片含义。**不得保留 concatenate 全部 action 的隐藏全量路径。** 优先复用同一统计计算，不另写一套不等价归一化算法。
6. `MultiTaskDataset.iter_normalization_data` 继续 yield from 子数据集，不先把所有子块收集成 list。val 不拟合，旧 checkpoint 不重拟合。

### 10.3 deterministic 数据集

deterministic 分支使用 fixed_indices，不创建 Manager，不读取 epoch proxy；`set_epoch`、`close`、pickle 路径也处理该分支。随机分支继续现有 epoch 同步、persistent_workers、DDP/resume 行为，不为去 RPC 重写整个 sampler。

### 10.4 时间假设

当前默认配方保持按实际记录行索引取窗，在相关稳定文档/配方日志中明示其含义；本次不新增 `data_recipe.temporal` 等持久化键，也不改 Dataset 配置身份。`training/resume.py` 会严格比较这些值，不能为一句说明破坏旧实验恢复，再新增兼容层补救。不要新增固定 `gap>2dt` 拒绝、删行后拼窗或自动插值；500 ms 反例不是所有窗口必然错误的证明。

需要 fixed-grid 的新实验才消费时间并定义 anchor/dt/容限/未知时间处置，该实验见第 12 节，不混入本次读取等价优化。缺历史时间不伪造为精确名义时间，旧实验配方与 checkpoint 继续权威。

### 10.5 验收

- 同一小型合成 Zarr/seed，对比窗口索引、首尾 padding、样本 shape/dtype/值、dispatch mask、唯一统计源行与 normalizer。维持 X1 的正例：H=3、N_obs=1 时正确角色条件下保留原有 3 窗，obs 统计行 0/1/2，action 统计行 0/1/2/3/4。
- 统计覆盖 limits、gaussian、常量、空块、不同 chunk 划分、多任务、action_ee/auto、aux EE。逐源行集合相同；浮点归约允许合理公差，不谎称 bit-exact。旧 checkpoint 加载路径不触发重新拟合；同语义旧 resume 合同不因新增标签键/Reader 配置项而失败。
- 用单遍 iterator/受控块生命周期或等效探针，证明 consumer 没有先耗尽并保留全部 chunks；不仅断言源码中没有某个 `list` 字样。增加数据总量而固定 chunk 时，payload 峰值内存不能随总量线性增长；允许 O(rows) 的资格 mask/索引。
- 覆盖单进程、spawn、可用时 fork、persistent_workers、train/val；deterministic 顺序完全一致且不创建 Manager，随机分支仍随 epoch 变化。
- 记录 startup 耗时、peak RSS、steady samples/s、I/O 与 worker 数。有限性扫描仍可能读取大量数据；懒读的内存收益不等于所有磁盘上的吞吐都提升。若吞吐回退，先核对物理 chunk/角色读取/预取，不添加无界缓存或恢复全量高维加载掩盖问题。

## 11. D9：明确删除与明确保留

### 11.1 删除失效 prior（#20）

当前 human-flexion prior 的 3D 夹角会混入外展，默认权重 0 仅规避使用。本次删除该失效研究选项：配置字段、映射、reference/mask/gradient 和只为它存在的 DexPilot 包装逻辑。TAG 位置目标、时序正则、pinch、EMA、关节顺序和 mimic adaptor 是独立机制，不能顺手删除。

更新所有 repo 内配置/示例/文档；显式旧 prior 配置清楚报错，不静默忽略，也不建兼容层。以后需要该研究变量时另行定义掌面/屈曲与消融。原生环境中对照原默认 prior=0 的目标、梯度和输出；无法运行时标 `NOT VERIFIED`。

### 11.2 静态配置去重（O03）

只删除同一调用链中、同一事实、同一配置未变、同一副作用边界的无用重复。**保留 session 连接设备之前的验证**；当前 PolicyRunner 在 robot.connect 之后构造，不能把 session 验证删掉后声称构造器已覆盖。独立 public Runner/driver 构造的保护也可保留，轻量重复优于 validated token/trusted bypass 或为去重重排整个生命周期。

优先删除 CLI 中等价的重复解析/兼容检查与选中入口之后的内部重复，运行时模式变化必须重新验证。配置形状与后续动态输出/年龄/epoch 不是一个事实，不能一起删。

### 11.3 保留/关闭项

- O04：保留 Robot 双目标 preflight 和独立 driver 的硬限位/shape 检查；不要新增 bypass API。
- O10：camera_ring 已共享 `SeqlockSlot/seqlock_is_complete/seqlock_to_logical`，关闭“尚未抽取”的建议；不再造通用 IPC。
- O11：描述性 policy-ID 常量和 uint8 finite 扫描已删的部分关闭；query、SDK buffer、跨线程数组的必要 owned copy 保留。
- O12：worker、sync/async/RTC、计划时间已经存在；只修 D1，不再引入第二套执行框架。
- X1 与 Policy 版 N3 已修主路径保留；Teleop 版 N3 仍按 D2 修。

## 12. 条件项：有依据再进入，不阻塞必做项

| 项目 | 当前决定 | 进入实施的必要证据 |
| --- | --- | --- |
| B1 / #17，10 mm mount 差异 | 本次不修改 -15/-5 mm，不生成临时 URDF，不建立模型配置框架。 | 实物 adapter 与同名 frame 定义/测量确认；然后让当前 mount 真值一致供 FK 与碰撞消费。具体最小机制依据届时模型加载 API 选择。 |
| R3-4，fixed-grid recipe | 本次只明示 recorded_rows，不自动改变资格。 | 当前模型确需固定采样间隔；报告实际节拍、源年龄和 observation→dispatch 延迟；为新实验定义容限与未知时间处理，再重算最终训练统计。 |
| O06，静态几何/历史 FK | 已有只读实例中可无额外抽象地预计算不变值；历史 FK 缓存先测。 | 热点占比实际可见；缓存随本行 history 生命周期释放，不能按 ndarray hash 建全局缓存或绕过独立 public 函数的参数检查。 |
| O07，TAG fixed-base | 本次不强制重写。 | Pinocchio frame/order/FK、解析梯度与数值差分一致，再测收敛和 p95；只读 Model 可复用，可变 Data 不共享。 |
| O08，HOME 搜索/整形/执行 | 保留完整路径检验，D5 只修软逃离。 | 分项 profile；环境/手姿一致时才合并重复整形；减少 stop-and-go waypoint 还需固件实际轨迹证据。 |
| O13，点云 radius graph | 不把 cap 直接前移改变拓扑。 | 先评估等价的分批邻域/union；体素化或删滤波需薄物体/小物体保留率、空云率、几何偏差与任务质量对照。 |
| O14，RTC scheduler | 本次不强制新增 cache。 | profile 确认重复配置/GPU scalar 同步占比；最多当前 worker 的一份 prepared schedule，按 device/dtype/NFE/参数变化失效，保持 J-vjp 梯度与步系数。 |

没有证据就报告 `DEFERRED` 与缺什么，不虚构性能倍数，不为了“所有项完成”实施条件性重构。硬件实测需另行授权。

## 13. 批次、验收与完成标准

### 13.1 推荐实施次序

1. D1 发送/交接、D2 取消/证据、D3 的 B2；这批覆盖 4 项 P1。可按不同边界拆小改动，不一次重写所有 runner。
2. D4 触觉、D5 几何/HOME、D6 标定、D7 视频；优先使各自旧反例与合理正例通过。
3. D8 端到端懒读取/流式统计与 deterministic 去 RPC；Real/Policy 接口配套。
4. D3 当前会话快照/O09、D9 明确精简、D7 诊断与指标标签；不要把已完成项重复改造。
5. 条件项只有证据充分才进入，否则明确记录。R3-3 若当前实验实际使用收窄限位/大跟踪偏差，应上调首批处理。

阶段是组织方法，不是要求每批等待用户批准。每批检查 diff 与相关测试，充分验证后前进，不反复跑没有新增风险的大范围检查。

### 13.2 验证分层

- 必须先确认命令本身无硬件副作用。使用现有环境；按改动选择现有 pytest 文件并添加必要的定向反例/正例，不用 stub 结果冒充原生通过。
- 按仓库 AGENTS 执行适用的 compile、Ruff、diff 检查；至少确保最终修改文件通过格式/静态检查，未改文件的既有失败单列。
- Real 现有重点测试为 `tests/test_policy_runner.py`、`test_policy_recording.py`、`test_policy_inference.py`、`test_policy_eef.py`；后者可能需要原生几何依赖。Policy 关注窗口、RTC、训练/恢复相关现有测试；先核对最新路径再运行。
- 通常适用的无硬件基础检查：

  ```bash
  python -m compileall -q dexmani_real examples
  ruff format --check dexmani_real examples tests
  ruff check dexmani_real examples tests
  git diff --check
  ```

- 新增/修改测试也要检查语法与格式。Policy 在其工作树执行相应检查；具体 config-only/smoke 入口先读源码确认无设备/大规模实验副作用。
- 原生离线门槛：实际 PyAV 的随机/顺序解码，Pinocchio/MPlib FK/IK/路径与梯度，torch normalizer/RTC，Zarr 多 worker。缺依赖时标 `NOT VERIFIED`，不升级全环境，不降低语义让测试通过。
- 本任务不执行真机；软件与原生离线通过仍需标明 real hardware `NOT VERIFIED`。

### 13.3 最终交付

给出：

1. 每个必做问题的状态：`FIXED`、`ALREADY_FIXED`、`BLOCKED`；条件项为 `DEFERRED` 或给出已具备证据的实现结果。不能把仅写了代码或缺关键原生验证的部分说成全验收通过。
2. 两仓库实际分支/实现基线、修改文件与接口；解释哪些机制删除、哪些因边界不同而保留。
3. 实际运行的命令与结果；区分逻辑、原生离线、性能、真机，列明未验证项。
4. D8 内存/统计等价证据；若没有代表性数据性能结果，不声称端到端加速。
5. 未解决风险、精确阻断与后续设备验证项目。只指出缺少的事实，不为假想未来场景新增规则。
6. 最终 diff 只包含本任务必要内容；没有错误先验、过期回调、无人消费状态、半迁移调用点或无关重构。Raw、checkpoint 和现有默认研究配方未被静默改变。

## 14. 参考项目的固定机制与适用范围

| 参考 | 可复用原则 | 边界 |
| --- | --- | --- |
| [UMI exec_actions](https://github.com/real-stanford/universal_manipulation_interface/blob/d095ba9590df789df5189eea5ee7e431689038a6/umi/real_world/real_env.py#L363) | 过期动作退出执行链，动作与目标时间绑定。 | 不复制 wall clock、夹爪专用偏移或截断 Raw。 |
| [LeRobot 字段投影](https://github.com/huggingface/lerobot/blob/8c920c4270460851cedd2737657584586d3dc66f/src/lerobot/datasets/dataset_reader.py#L347) | 按字段/窗口读取，避免读 action 顺带解码图像。 | 不迁移整个 HF 格式；delta 索引不自动解决任意不规则采样。 |
| [LeFranX 命令年龄](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/franka_server/src/franka_server.cpp#L345) | 执行侧消费最新命令并检查年龄。 | 不复制 Franka 频率、Ruckig、增益或自动 hold。 |
| [ManiUniCon 设备边界](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/robot_interface/base.py#L8) | 设备接口薄、职责直接。 | 不复制原地 clipping、bool 派发结果或泛化 ABC。 |

这些源码提供设计参照，不能代替 DexMani 的原生/设备/性能验收。

## 附录 A：16 项开放实现问题的核查与映射

编号沿用历轮审查，不是要求把 63 条历史记录全部重新修一遍。

| ID / 优先级 | 当前证据与条件 | 实施包 / 源码 |
| --- | --- | --- |
| R3-1 / P1 | 生产控制步骤和 _send 的假时钟反例：准备耗时 500 ms，默认反馈年龄预算约 133 ms，两个 SDK 仍调用，状态为 [ACCEPTED, ACCEPTED]。 | D1；[dexmani_real/runtime/observation.py:105](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/runtime/observation.py#L105)；[dexmani_real/teleop/control/controller.py:90](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/teleop/control/controller.py#L90) |
| B2 / P1 | RGB 固定、depth 推进 1 s，生产 worker 发布 5 次，旧 RGB 仍通过 4/30 s 的新鲜度预算。当前 worker 与旧反例函数 AST 一致。 | D3；[dexmani_real/sensor/camera/worker.py:60](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/sensor/camera/worker.py#L60)；[dexmani_real/sensor/camera/realsense.py:576](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/sensor/camera/realsense.py#L576) |
| N1 / P1 | 同一注入点：RuntimeError 得到 1 次调用、1 行 UNKNOWN；KeyboardInterrupt 得到 1 次调用、0 行记录。相关生产函数 AST 未变。 | D2；[dexmani_real/robot/robot.py:267](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/robot/robot.py#L267)；[dexmani_real/teleop/control/controller.py:98](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/teleop/control/controller.py#L98) |
| N2 / P1 | 生产 ReplayRecorder/replayer/session 的旧反例：capture.count=1，to_dict 未调用，evaluate_replay 收到 None；相关函数 AST 未变。 | D2；[dexmani_real/replay/replayer.py:269](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/replay/replayer.py#L269)；[dexmani_real/replay/replayer.py:289](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/replay/replayer.py#L289) |
| R3-3 / P2 | 区间 [-180°,180°]；measured=-106°、previous=101°、candidate=106°：绝对差 212°、周期差 148°，band 差额 64°，返回不拒绝。投影仍为 106°。 依赖非默认限位与很大既有跟踪偏差。 | D5；[dexmani_real/planning/kinematics/ik.py:372](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/planning/kinematics/ik.py#L372)；[dexmani_real/planning/kinematics/ik.py:389](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/planning/kinematics/ik.py#L389) |
| R4-2 / P2 | 注入有限间隙序列：起点 -1 mm，每步 -0.4 mm，累计至 -7 mm，再回到 0，validator 返回 safe 和 table_soft_escape。 | D5；[dexmani_real/planning/paths.py:293](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/planning/paths.py#L293)；[dexmani_real/planning/paths.py:375](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/planning/paths.py#L375) |
| B3 / P2 | SDK 状态 1501019 映射 (True,False)；生产 calibrate_tactile 抛 incomplete tactile data，整体 calibrated=False。 | D4；[dexmani_real/robot/drivers/xhand.py:182](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/robot/drivers/xhand.py#L182)；[dexmani_real/robot/drivers/xhand.py:407](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/robot/drivers/xhand.py#L407) |
| B4 / P2 | 生产 calibrate_tactile 中，verify 注入 RuntimeError 后 tactile_calibrated=True；当前整个 XHand 类与旧反例 AST 一致。 | D4；[dexmani_real/robot/drivers/xhand.py:381](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/robot/drivers/xhand.py#L381)；[dexmani_real/robot/drivers/xhand.py:436](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/robot/drivers/xhand.py#L436) |
| R4-1 / P2 | 假时钟在 +10 ms 置急停，等待到 +2010 ms 才超时；已置 Q/ESC 的队列仍能触发 ENTER 并调用假保存边界。 捕获前后已有静止检查；不据此声称持续运动。 | D6；[dexmani_real/calibration/camera/session.py:84](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/camera/session.py#L84)；[dexmani_real/calibration/camera/session.py:348](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/camera/session.py#L348) |
| R4-3 / P2 | 实际纯配置加载接受三种非法覆盖；生产 idle tick 可 ZeroDivisionError；margin=1 m 时目标被 np.clip 到 [-0.28,-0.5,-0.5]。 | D6；[dexmani_real/config/experiment.py:129](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/config/experiment.py#L129)；[dexmani_real/teleop/keyboard_session.py:33](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/teleop/keyboard_session.py#L33) |
| R3-2 / P2 | 生产 runner：Future 在读观测的 +5 ms 完成，容限 30 ms，却 handoff_miss；同一完成事实在读后回收则发送。 | D1；[dexmani_real/deployment/runner.py:592](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/deployment/runner.py#L592)；[dexmani_real/deployment/runner.py:618](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/deployment/runner.py#L618) |
| N3 / P2 | 生产 Teleop 反例中，shared 原因仍为 OPERATOR/QUIT，保存原因却均为 motion_revoked；当前 interrupted 方法 AST 未变。 | D2；[dexmani_real/teleop/runner.py:105](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/teleop/runner.py#L105)；[dexmani_real/teleop/runner.py:264](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/teleop/runner.py#L264) |
| B5 / P2 | A=(1 mm,10°)、B=(2 mm,1°)，门限=(5 mm,3°)：生产选择器选 A 后被拒，B 本可通过。 | D6；[dexmani_real/calibration/camera/solver.py:264](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/camera/solver.py#L264)；[dexmani_real/calibration/camera/session.py:272](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/camera/session.py#L272) |
| N4 / P2 | 真实 NumPy 合成点反例：1500 点、seed=7，要求 ratio>=0.561，最终 838/1500=0.558667 仍返回成功。 为非默认 ratio 门限。 | D6；[dexmani_real/calibration/table.py:156](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/table.py#L156)；[dexmani_real/calibration/table.py:169](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/calibration/table.py#L169) |
| B6 / P2 | 30 fps、time_base=1/15360；第 123 帧算成 122.99999999999999，被跳过而返回 124。 | D7；[dexmani_real/recording/storage/video.py:208](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/recording/storage/video.py#L208)；[dexmani_real/recording/storage/reader.py:149](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/dexmani_real/recording/storage/reader.py#L149) |
| R4-4 / P3 | 生产 main/loop 配假 camera：read 抛错、disconnect 调用、打印 Test complete、退出码 0。 | D7；[examples/realsense_record_example.py:451](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/examples/realsense_record_example.py#L451)；[examples/realsense_record_example.py:614](https://github.com/haoyangzhanglab/dexmani_real/blob/660aa20f52eee7bbbfa387d692f0715aca17502f/examples/realsense_record_example.py#L614) |

新增的流式消费证据：[build_normalizer](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/build_utils.py#L53)、[fit_field_chunks](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/normalization.py#L422)、[mixed action](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/agents/normalization.py#L542)。

时间标签修订证据：[严格 resume 合同](https://github.com/haoyangzhanglab/dexmani_policy/blob/650f1e3cfc63c64678b5bb69e7258b548800d053/dexmani_policy/training/resume.py#L183)。

## 附录 B：早期 #01-#43 逐项最终处置

| 编号 | 当前核查状态 | 最终处置 |
| --- | --- | --- |
| #01 | 已修复 | 保持显式 HOME 与先启动停止监听。 |
| #02 | 已替代 | 保持已实现 worker；D1 修新交接缺口。 |
| #03 | 已修复 | 保持 SO(3) 增量；无新增平滑。 |
| #04 | Raw 已修复；缓存取舍接受 | 保持 Raw staging；派生缓存不建恢复事务。 |
| #05 | 主路径修复，取消残留 | D2 补取消接缝，主路径不重写。 |
| #06 | 已修复 | 保留 row_info；D8 明示时间假设。 |
| #07 | 保留能力边界 | D3 修通道年龄；不加全局同步 barrier。 |
| #08 | 已修复 | 保持剩余预算等待与不追赶突发。 |
| #09 | 已替代 | 保持固定槽；D1 修 Future 回收时机。 |
| #10 | 已修复，语义已更新 | 保持真实历史；D8 不自动补帧/插值。 |
| #11 | Policy 已补年龄边界 | D1 用原 query 到期，不刷新旧决策。 |
| #12 | 关联与含义已补 | 保留已有 query/slot/source trace。 |
| #13 | 部分修复 | D3 修 B2；保留各路 stall。 |
| #14 | 旧评分错误已修复 | D6 修 B5；保留 RMS 正确定义。 |
| #15 | 已修复 | 保持 Rotation.mean。 |
| #16 | 已修复 | 保持新 head 时间去重和圆统计。 |
| #17 | 待现场确认 | B1 先实测；本任务不改数值、不新增 URDF 生成。 |
| #18 | 选支已修复 | 保持机械等价选支，D5 修绝对距离。 |
| #19 | 已修复 | 保持实际 palm 三角形检查。 |
| #20 | 默认禁用，条件风险保留 | D9 删除当前失效 prior 功能。 |
| #21 | 已修复 | 保留选中 TAG 的启动校验；独立公开边界不机械去重。 |
| #22 | 明确接受 | 保持当前标定重建；无历史 ABI。 |
| #23 | 原缺陷已修；口径建议 | D7 明示 RMSE 口径；不重写旧结果。 |
| #24 | 已修复 | 保持技术异常外抛与普通无解区分。 |
| #25 | 部分修复 | D4 拆触觉校准；辅助 NaN 不阻塞关节。 |
| #26 | 已按用途精简 | D9 仅删同边界重复；13 字段与连接前校验保留。 |
| #27 | 已修复 | 保持已删历史 HOME 令牌与当前姿态检查。 |
| #28 | 合理保留 | 保留 measured/首目标两项起点检查。 |
| #29 | 优化项，非无碰撞 bug | D10 仅在 profile 需要时改搜索模型。 |
| #30 | 保留能力边界 | 保持在线/HOME 的不同能力说明。 |
| #31 | 已修复 | 保留断连错误传播与逐资源清理。 |
| #32 | 已修复 | 保持首次桌面标定不依赖旧平面。 |
| #33 | 已修复 | 保持 monotonic 时长与有界队列。 |
| #34 | 已修复 | 保持分数 fps；D7 修随机索引 B6。 |
| #35 | 已修复 | 关闭兼容问题；不升级全环境。 |
| #36 | 撤回笼统删除建议 | D5 保留 jump/hw_dist，删除失去作用的 band_switch。 |
| #37 | 已修复 | 已无引用，不重新引入。 |
| #38 | 已精简 | 保持已删无人消费统计。 |
| #39 | 已修复 | 保持可选 viewer 懒导入。 |
| #40 | 已修复 | 保持顺序流式读取；D7 修 random seek。 |
| #41 | 接受桌面能力边界 | 维持桌面能力边界，无 headless 新需求。 |
| #42 | 原判断撤回 | 保持首次无反馈就禁发，持续失败才 FAULT。 |
| #43 | 文档与示例已更新 | 随 D1-D9 更新实际接口/时间/状态文档。 |

特殊项：X1 的主训练资格/统计已修复，D8 保持其语义；R3-4 只在新 fixed-grid 实验下启动额外资格设计；B1 差异存在但实物真值未知；RMSE 是合法口径差异。它们不额外计入 16 项开放实现 bug。

## 附录 C：O01-O16 精简/替代建议的最终裁决

| 编号 | 最终裁决 | 实施范围 |
| --- | --- | --- |
| O01 | 实施 | D8：Dataset、build_normalizer、fit_field_chunks、mixed action 全链路有界流式。 |
| O02 | 随修复实施 | D2：首因小 helper，复用既有 run latch。 |
| O03 | 同边界去重 | D6/D9：保留连接前验证及独立 Runner/driver；不加 validated token。 |
| O04 | 保留 | D9：双目标 preflight 与独立驱动边界各有责任。 |
| O05 | 实施 | D3：复用现有 CameraExtrinsics，只消除同会话重复读文件。 |
| O06 | 小型直接预计算/其余有证据再做 | D10：不重建 ray cache，不为去验证增加多层准备框架。 |
| O07 | 有证据再做 | D10：fixed-base + 独立 Data，原生梯度验收。 |
| O08 | 保留并测量 | D5/D10：先修软区；整形/执行另做路径证明。 |
| O09 | 随 D3 精简 | D3：mount 诊断 metadata 不再二次准入。 |
| O10 | 关闭基础抽取 | D9：seqlock helper 已共享，本轮纠正旧建议。 |
| O11 | 已完成部分关闭 | D9：不删 query/SDK buffer 的必要 owned copy。 |
| O12 | 关闭架构新增 | D1：worker 已实现；修 handoff，不建第二套。 |
| O13 | 有证据再做 | D10：先等价分批邻域；体素/滤波需质量消融。 |
| O14 | 有证据再做 | D10：prepared schedule，保持梯度与系数语义。 |
| O15 | 实施固定分支 | D8：deterministic 删除 Manager，随机分支保留。 |
| O16 | 随 D8 实施 | D8：公开 backend 构造一次元数据检查。 |

附录用于防止遗漏与反复整改；最终优先级与执行要求以上文为准。已证实修复、接受的能力边界和明确 DEFERRED 项均应如实记录，不为追求“全部勾选”改动实验语义。

## 执行记录（2026-10-06）

本节记录本次实施结果，不替代上文验收边界。`FIXED` 表示代码整改与所列离线检查完成，**不表示真机验收通过**。

### 工作树与交付

- Real：`main`，HEAD `ebaf789583c7841a24e6d055a1fd7ef32a2ddb09`；相对实现基线 `660aa20f52eee7bbbfa387d692f0715aca17502f` 仅新增本任务书。
- Policy：`main`，HEAD `7120084f874aa4404de0957993af8d7a1de89c01`；相对实现基线 `650f1e3cfc63c64678b5bb69e7258b548800d053` 仅新增旧审查任务书，未重做旧任务。
- 两个 origin 均为 `haoyangzhanglab` 对应仓库。实施前工作树无未提交修改；没有 checkout/reset、commit/push、修改权限配置或 site-packages。
- 交付为两个现有工作树的未提交修改及新增测试。没有连接设备、执行运动/HOME/采集/回放/rollout/真实标定，没有启动完整训练/DDP，没有改写已有 Raw、checkpoint 或实验产物。测试仅使用 fake I/O、临时文件及合成数据。

### 必做问题逐项结果

| ID | 状态 | 实现与离线证据 |
| --- | --- | --- |
| R3-1 | FIXED | `send_action`/`send_hand_home` 必填排他截止；全部生产调用点迁移；准备、mode restoration、两 SDK 间越期与撤权、最后 SDK 迟返回测试。已知 ACCEPTED 不回退。 |
| R3-2 | FIXED | 读 Observation 后、handoff 前非阻塞 poll；异步预取 Future 在读反馈 +5 ms 完成的正例与既有过期/失效反例。 |
| B2 | FIXED | RGB/depth 各自推进时刻的最小值作为来源时刻，复用该时刻判定 stall；fake worker 覆盖单路/双路冻结及正常推进。cloud 保留来源时间/sequence。 |
| N1 | FIXED | `DispatchInterrupted` 携带当前结果；Teleop/Policy/Replay 撤权、stop 后单次记录 attempted row；SDK 前/arm 内/hand 内中断与附加 stop failure 回归。 |
| N2 | FIXED | Replay 把 Ctrl+C 转为 ESTOP outcome，finally stop 后交出 prefix；arm/hand/wait 三种中断的真实 dispatch 数量回归。原 session 的评估/保存与 CLI 非零退出顺序保留。 |
| N3（Teleop） | FIXED | 匹配当前 run_id 的首因；开始时在 motion_lock 内保存 run_id，新 capture 清除旧 run 关联；S/Q 与 stop failure 共存仍保留首因。 |
| B3 | FIXED | aggregate/dense bias 独立；包括 SDK 1501019 在内的 aggregate-only、dense-only、both/neither 测试，独立缺测保留 NaN。 |
| B4 | FIXED | 候选局部捕获、fresh raw verification 后逐路提交；验证异常、重校准中断与 aggregate 残差超限不泄露候选、不吞 dense。 |
| R3-3 | FIXED | jump/hw_dist 使用选支后的绝对坐标差；212° 反例拒绝，周期排序保留；删除被 150° 绝对界支配的 band_switch。 |
| R4-2 | FIXED | 软逃离对比全程最佳 clearance，防止逐步累计向内滑移；负值仅允许起始前缀且必须离开软区。受控路径正反例及原生合成平面路径检查。 |
| R4-1 | FIXED | 取消优先于队列事件，等待/检测/反馈读取/FK/求解后的取消检查；atomic replace 前最后取消准入，临时文件清理，不持 motion_lock 写文件。 |
| R4-3 | FIXED | keyboard/calibration 入口连接前验证所选 jog 配置及 workspace 内缩余量，非法配置测试不触达连接。实际配置为 idle_interval_frames，验证其正整数约束。 |
| B5 | FIXED | 先按位置与旋转 RMS 双门限筛选，再按位置/旋转 RMS 稳定选择；非有限候选拒绝。A=(1 mm,10°)、B=(2 mm,1°) 选择 B。 |
| N4 | FIXED | final refit mask 再检查数量/比例；受控 refit 临界反例拒绝。 |
| B6 | FIXED | PyAV 有理 rate/time_base、首帧 origin 与唯一 CFR ordinal；实际编解码覆盖整数/分数 fps、第 123 帧、GOP/B-frame、非零 origin、逆序/重复请求。 |
| R4-4 | FIXED | diagnostic 明确完成/用户退出/读取失败；fake read failure 非零退出且 disconnect，成功消息不再掩盖失败。 |
| RMSE 口径 | FIXED | arm `mean_joint_rmse`、hand `pooled_rmse` 同步输出与新 JSON；只改命名，不重算旧结果；单轴误差测试。 |
| D8 全链路 | FIXED | Policy Reader→角色窗口→唯一训练源行 chunk→build_normalizer→Welford/mixed normalizer 单遍消费；spawn/fork/persistent worker、float32 溢出、空块/常量/gaussian/aux EE、生命周期探针与合成性能证据。 |
| D9 prior | FIXED | 删除配置、human-flexion 映射、prior reference/mask/gradient、prior 专用 DexPilot objective 包装；旧配置显式报错；原生 prior=0 对照见下。 |

没有因外部依赖而留下 `BLOCKED` 的必做软件项。真机、代表性数据吞吐和下列条件优化仍未验证。

### O01–O16、已修路径与条件项

| 项目 | 状态 | 处置理由 |
| --- | --- | --- |
| O01 | FIXED | 去除 Dataset 全量高维加载及 normalizer list/全 action concatenate；每进程每字段仅保留一个受 chunk 行数/16 MiB 预算约束的解码块。独立 VQ 工具配套迁移到 open/read，保留其显式低维数组算法。 |
| O02 | FIXED | Teleop 小型首因 helper 复用 run latch，不建通用 finalizer。 |
| O03 | FIXED | 删除 CLI num_episodes 的同边界重复检查；保留连接前、独立 Runner/driver、输出与动态年龄等不同责任的检查。 |
| O04 | ALREADY_FIXED | Robot 双目标 preflight 与公开 driver 硬限位检查分属不同边界，保留，无 trusted bypass。 |
| O05 | FIXED | session 一次解析既有 CameraExtrinsics，pointcloud/recorder 共用；START 仅选实际 serial；文件变化/无外参录制回归。 |
| O06 | DEFERRED | 没有实际静态几何/历史 FK 热点占比证据；不新增缓存框架。 |
| O07 | DEFERRED | 原生梯度已验证，但没有 fixed-base 替代后的收敛与 p95 对照，不重写 TAG。 |
| O08 | DEFERRED | 已修软逃离；缺分项 profile 与固件实际轨迹证据，不改 HOME 搜索/整形/执行机制。 |
| O09 | FIXED | mount metadata 消费仅作诊断，recorder START 复制当前已校验值，删除额外几何准入；公开 Raw writer 布局/数值及 camera 几何边界保留。 |
| O10 | ALREADY_FIXED | camera_ring 已使用 SeqlockSlot/seqlock_is_complete/seqlock_to_logical，不再抽取 IPC。 |
| O11 | ALREADY_FIXED | 旧描述性 ID/uint8 finite 扫描未重新引入；query、SDK、跨线程 owned copy 保留。 |
| O12 | ALREADY_FIXED | 单模型 worker、一个 Future 与 sync/async/RTC 已存在；仅 D1 修复交接时机。 |
| O13 | DEFERRED | 无 radius graph profile 及薄/小物体保留率、空云率、几何误差/任务质量对照；不移动 cap 或改过滤拓扑。 |
| O14 | DEFERRED | 无 GPU scheduler/scalar 同步成本证据；不增加 schedule cache。 |
| O15 | FIXED | deterministic 不创建 Manager/访问 epoch proxy；随机分支 epoch 同步保留。 |
| O16 | FIXED | 公开 Reader 构造一次校验 meta/shape/dispatch；fork 按 PID 重开、pickle 不携带句柄或解码缓存。 |
| X1 / Policy N3 | ALREADY_FIXED | 角色窗口、唯一统计行、strict resume、首因 latch 既有行为保留；相关回归通过。 |
| B1 / #17 | DEFERRED | 缺实物 adapter 与同名 frame 的测量真值；未改 -15/-5 mm，也未生成 URDF。 |
| R3-4 | DEFERRED | 无 fixed-grid 新实验需求与实际时间分布证据；仅文档/日志注明 recorded_rows，不增加持久化 recipe 键、不插值或改资格。 |

保留 StrictDexPilotOptimizer 的 `retarget` 异常外抛：上游会把 RuntimeError 转为旧目标，删除这层会破坏现有技术失败语义。TAG 位置目标、pinch、时序项、EMA、joint order、mimic 均保留。删除 before_send/dispatch_row/_dispatch_allowed 后，owner 仍负责 episode、WAIT 和计划失效；Robot 仅在副作用边界消费截止与授权。

### 实际验证与边界

环境未升级：Real 使用 `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`，Policy 使用 `/home/zhanghaoyang/miniconda3/envs/policy/bin/python`（均 Python 3.10.20）。原生依赖包括 NumPy 1.26.4、torch 2.4.1、Zarr 2.18.3、PyAV 17.1.0、Pinocchio 2.7.0、MPlib 0.2.1、dex_retargeting 0.4.6。Ruff 0.16.8 来自 Real 环境。下文 `R`/`P`/`RUFF` 分别表示上述两个 Python 与 Real 环境 Ruff 的绝对路径，命令在相应仓库运行。

| 层次 | 实际命令 | 结果 |
| --- | --- | --- |
| Real 纯逻辑 + 原生离线 | `$R -m pytest -q tests` | PASS，118 passed；无硬件 I/O。最后增加非有限 RMS 拒绝后再次跑 `tests/test_review_remediation.py`：54 passed。 |
| Real 语法 | `$R -m compileall -q dexmani_real examples tests` | PASS。 |
| Real 静态 | `$RUFF format --check dexmani_real examples tests`；`$RUFF check dexmani_real examples tests`；`git diff --check` | PASS；125 文件格式通过。 |
| Policy 纯逻辑 + 原生离线 | `$P -m pytest -q tests --ignore=tests/test_infra_cuda.py -o cache_dir=/tmp/dexmani_policy_pytest_cache` | PASS，81 passed、28 subtests passed；1 条既有 historical data_identity 未验证 warning。路径测试仅在临时目录写虚构 checkpoint；不读写实验。 |
| Policy 最后脚本迁移后 | `$P -m pytest -q tests/test_infra_codebook.py tests/test_streaming_dataset.py -o cache_dir=/tmp/dexmani_policy_pytest_cache` | PASS，21 passed、20 subtests passed。 |
| Policy 配置 | `$P -m dexmani_policy.smoke_test --config-only dp3` | PASS，6 个 Hydra targets；未执行完整 smoke、训练或推理。 |
| Policy 独立工具入口 | `$P -m scripts.training.train_vq_hand --help`；`$P -m scripts.training.measure_vq_usage --help` | PASS；仅 argparse/import。 |
| Policy 语法 | `PYTHONPYCACHEPREFIX=/tmp/dexmani_policy_compile_cache $P -m compileall -q dexmani_policy scripts tests` | PASS。 |
| Policy 修改文件静态 | 对 `git diff --name-only` 与 `git ls-files --others --exclude-standard` 的 11 个 `.py` 执行 `$RUFF format --check --no-cache ...` 与 `$RUFF check --no-cache ...`；`git diff --check` | PASS。 |
| Policy 全量静态基线 | `$RUFF check --no-cache --output-format=json dexmani_policy tests`；`$RUFF format --check --no-cache dexmani_policy tests` | NOT PASS：76 个未修改文件共 385 条既有报告，63 个未修改文件需格式化；修改文件报告为 0。未批量改无关文件或降低规则。 |
| 原生 prior=0 对照 | `PYTHONPATH=. MPLCONFIGDIR=/tmp/dexmani_mpl $R /tmp/dexmani_retarget_equivalence.py` | PASS。临时脚本用 `git show 660aa20:<path>` 加载旧 TAG/DexPilot，在当前原生环境对照 objective/gradient/solve；TAG 数值差分最大绝对误差 `3.5315350643827514e-10`。持久回归保留于 `test_review_remediation.py`。 |

早期测试失败已处理：Policy 旧 `replay_buffer.root` 消费、删除 import 后失效的测试 mock 路径已迁移；沙箱阻止的临时目录测试以获准权限重跑。新增 PolicyRunner 测试最初使用了错误的事件名，已按生产 `end` 事件修正；未改变生产状态或放宽断言。原生路径探针初次缺 hand qpos 被正确拒绝，补显式手姿后通过。只有上述实际成功命令标为 PASS。

原生离线范围：真实 PyAV 编解码；Pinocchio/MPlib FK/IK 与合成桌面路径；TAG 梯度及 DexPilot prior=0 等价；torch normalizer/RTC；Zarr 单进程、spawn/fork 与 persistent_workers。取消、传感器冻结和 SDK 状态使用 fake I/O；它们证明软件分支，不能证明固件响应或真实停机时延。

### D8 合成性能证据

相同 seed `20261005`；512/4096 行，episode 128 行，point_cloud 为 `(1024,6)` float32，物理 chunk 32 行，action_ee 21 维、joint_state 19 维；N_obs=2/H=8，batch=16，测量预热后的 16 batches。startup 包括 Dataset 与 normalizer，不含 worker 启动。基线从 Policy SHA `650f1e3` 用 git archive 提取源码到 `/tmp/dexmani_policy_baseline_20261005`，没有切换工作树。生成和测量脚本为 Policy `tests/benchmark_dataset_streaming.py`；w0 测量使用其临时副本，w2 最后测量使用仓库脚本并补充 worker VmHWM。

| 行数 / workers | 实现 | startup s | 父进程 peak RSS MiB | 相对导入后 RSS 增量 MiB | 最大单 worker peak MiB | samples/s | 父进程 rchar bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 / 0 | 基线 | 1.123 | 669.82 | 123.42 | — | 135744 | 15522267 |
| 512 / 0 | 修改后 | 1.042 | 655.52 | 108.75 | — | 19439 | 43279588 |
| 4096 / 0 | 基线 | 1.164 | 835.36 | 288.88 | — | 138565 | 98874099 |
| 4096 / 0 | 修改后 | 1.140 | 656.22 | 109.33 | — | 19494 | 209983235 |
| 512 / 2 | 基线 | 1.024 | 682.49 | 136.63 | 564.11 | 41444 | 124979650 |
| 512 / 2 | 修改后 | 1.022 | 654.96 | 108.91 | 548.34 | 19148 | 140657230 |
| 4096 / 2 | 基线 | 1.175 | 936.42 | 390.22 | 648.84 | 41015 | 385989074 |
| 4096 / 2 | 修改后 | 1.152 | 656.04 | 109.20 | 556.41 | 16042 | 304706386 |

实跑生成命令：`PYTHONPATH=. $P tests/benchmark_dataset_streaming.py /tmp/dexmani_bench_small_20261005.zarr --prepare 512`，大集用 `large` 和 `4096`；目录为本任务新建。实跑测量对上述两个目录分别设置 `PYTHONPATH=/tmp/dexmani_policy_baseline_20261005` 或当前 Policy 源码根，运行脚本的 `--workers 0`/`--workers 2`。脚本使用 `mode=w-`，重复生成不会覆盖已有目录。

有限性扫描仍读取训练所需全体字段。首次朴素懒读约 2.9k samples/s；检查重复打开 Array 和重解码相邻窗口后，采用每字段单块有界复用，w0 提至约 19.5k。8 倍数据量下新实现父进程 RSS 近似稳定，允许的 O(rows) 资格 mask/索引仍存在。两种数据量的 action scale 校验和分别为 `11.036111831665039`、`10.071537017822266`，基线与修改后相同；完整统计等价由 limits/gaussian/mixed/aux 的容差测试确认，未宣称归约 bit-exact。

这是一轮短合成、顺序访问测量，不是代表性训练吞吐或 p95；旧全内存路径仍明显更快。所有进程报告的父进程物理 `read_bytes=0`（页缓存），rchar 包含库加载/IPC，未测 worker I/O，RSS 包含 torch 等导入基础成本。**只据此报告有界内存收益，不宣称端到端加速。** 真实数据、随机打乱、存储冷缓存与长周期 worker 行为性能为 NOT VERIFIED。

### 未验证项目

- 真机全部 NOT VERIFIED：SDK 阻塞/模式恢复/撤权实际时延、两设备实际运动与停止确认、键盘 Ctrl+C 与固件交互、HOME 实际连续轨迹、真实 RGB-D stall、XHand 通道和无接触假设、相机/桌面/安装实际标定精度。
- 不提供硬实时保证；已进入 SDK 的调用无法软件撤回，ACCEPTED 仅表示接收确认。atomic replace 最后准入后发生的取消不回滚已经批准的发布。
- 未运行 CUDA/DDP 测试、完整 smoke/训练、代表性实验评测或数据性能测试。条件项缺少的事实见前表；未用历史 calibration 值作为新准入条件。

### 后续代码与文档清理（2026-10-06）

按用户后续要求，在上述整改基础上删除确认无人使用的 DexPilot 运行期滤波 accessor、持久 debug 字典、单调用 Pinocchio 加载 wrapper、不可达的可选配置分支；保留原滤波配置、数值算法与技术异常外抛。相机帧号已验证后不再保留 `or 0`，recorder 的无外参路径改为显式分支，删除不再执行文件 I/O 的路径上失效的 FileNotFoundError 捕获。TAG 无效输入日志改为“no target produced”，不再暗示主动保持运动。

Policy 删除 Reader 未使用的 logger/items、VQ 旧 episode_ends 兼容探测、Base/RGB/PC Dataset 写死实验路径的调试 main，以及 RGB/PC Dataset 仅转发参数的构造器。Hydra 使用的 RGBDataset/PCDataset 类和默认模态保留。

更新 README、执行/RTC/架构文档、导出时间说明和 normalizer 精度注释；明确新 Raw 的 RGB-D 最旧通道推进时刻与旧 Raw metadata 的解释边界。修正 VQ 脚本及 DQ-RISE YAML 中“全量统计自动匹配 Policy”的旧描述：VQ 仍按其原配方用全手部数据，Policy 用有效训练窗口的唯一 train 源行，兼容性校验保留；没有为文档一致而改统计范围。YAML 解析前后值完全一致。

清理后实跑：Real `pytest -q tests` 为 118 PASS；Policy 非 CUDA/DDP 全量为 81 PASS、28 subtests PASS（原 historical data_identity warning 保留）；原生 TAG/DexPilot 对照 PASS；新增清理的 RGB/PC 继承构造器与默认模态用临时合成 Zarr 检查 PASS，`smoke_test --config-only dp dp3` PASS。两仓库 compile、Real 全量 Ruff、Policy 全部修改 Python 文件 Ruff、diff 检查 PASS。没有新增设备操作、训练、环境变更或 commit/push。
