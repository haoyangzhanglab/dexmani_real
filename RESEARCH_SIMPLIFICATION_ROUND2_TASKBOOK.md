# DexMani Real 第二轮修复与精简任务书

> 目标：让个人研究代码轻而薄、正确好用、简洁高效统一；保留音频与全部现有核心实验能力。先修可复现的错误，再删除没有独立职责的机制，不以减少行数换取错误分类、数据含义或设备行为失真。
>
> 本文是待实施任务书。创建本文不表示代码已修复或验收通过。实施者应完成适用任务及验证，不再另写一份实施方案。

## 1. 基线、范围与交付方式

- 目标仓库：[`haoyangzhanglab/dexmani_real`](https://github.com/haoyangzhanglab/dexmani_real)。审查基线：[`18d77b0f282ca506e36e34b4c8fbfa5ee7ac2a21`](https://github.com/haoyangzhanglab/dexmani_real/commit/18d77b0f282ca506e36e34b4c8fbfa5ee7ac2a21)，提交说明为 `refactor: simplify research sessions and resource cleanup`。
- 配套 Policy 只读核对基线：[`f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb`](https://github.com/haoyangzhanglab/dexmani_policy/tree/f9a8e5134ad3a23788a2dc6f7107cff9fb8174bb)。本任务不授权跨仓修改、重写实验数据或改动 checkpoint。
- 本文承接根目录 [第一轮任务书](RESEARCH_SIMPLIFICATION_TASKBOOK.md) 的已完成工作，只执行本轮增量。此前 review 中的候选建议以本文明确取舍为准；历史任务书和 `.codex/tasks/` 不作为累计重做清单。
- 实施前读取当前 `AGENTS.md`、检查工作区与当前 HEAD。函数或行为已变化时沿实际调用链复核；已解决的问题标记“不适用”并给出依据，不为满足清单制造改动。
- 修改现有文件和调用方，不搬目录、不另建平行实现；不新增 manager、registry、通用状态机、错误图、回执协议、身份框架、配置兼容层或诊断数据库。
- 保留 VR 遥操作、录制/导出、sync/async/RTC、点动、物理回放、HOME、标定、查看、音频及既有离线入口。保持默认频率、限位、滤波、优化、采样和超时预算；本文明确删除的频率关系约束除外。
- 不连接或驱动真机，不删除用户数据，不自动合并或发布实现分支。完成本地修改、适当测试和可审查的 diff；真机验收由现场操作者另行执行。

## 2. 本轮结论与设计裁决

| 类别 | 已核实的问题或机会 | 本轮决定 |
| --- | --- | --- |
| R1：保存意图竞态 | Teleop/Policy 的 D 在锁存丢弃前暴露停止信号，收尾可能先读取 `save=True` | 用现有 `motion_lock` 排序“锁存意图”和“读取决定”；局部 `save` 是唯一保存决定，不加收尾状态 |
| R2：异常分类 | arm 已取得所需锁、尚未写 mailbox 时的本地到期仍抛 `TimeoutError` | 仅此分支改为现有 `OperationCancelled`；真实超时与停止失败继续报故障 |
| R3：writer 误判 | writer 已退出，监督线程稍晚恢复就因当前时钟超过 deadline 拒绝完整产物 | deadline 限制等待；等待后以 writer 是否仍活着和实际写入错误裁决 |
| R4：主错被覆盖 | 原子写失败后，临时文件删除的异常成为对外异常 | 失败路径尽力删临时文件，后错记录一次，原样重抛主错 |
| S1：反馈/新鲜度 | 无依据的频率大小关系、相同判断多处手写、初始反馈报错错指设备 | 删除错误约束，复用现有判断函数，修正描述；保留不同时间边界的检查 |
| S2：数据元信息 | 固定解释标签反复写入；readout 统计借用了另一条规则的阈值 | 删除指定固定标签及误导统计，格式语义集中写入现有 README |
| S3：音频与结果收尾 | 两入口复制尾音流程；Policy 为音频关闭失败再补写一次最终结果 | 在现有音频类增加一个小收尾方法；音频关闭后只写一次最终结果 |
| 保留：数值 recipe | 忽略未知字段可能悄悄丢掉影响数值的配置 | 保留短小的 exact-field 检查，不采用“只取认识的键”的建议 |

四项 R 问题已在 review 中用实际函数和受控依赖复现；这不代表真机故障概率已测量。旧任务书记载的 `138 passed` 属于原实施者的测试结果，本次 review 没有独立重跑完整依赖环境。实施后必须记录本轮实际执行的检查。

## 3. 必须保持的真实边界

以下是维护约束，不是要求新增运行时 contract、状态标签或逐层断言。

| 边界 | 保留内容 |
| --- | --- |
| 运动控制 | SDK 返回/异常、限位、真实停止结果、运行代际、指令到期；arm 后 hand 的顺序提交可能部分成功，mailbox 是 latest-target，不声称原子执行或逐行执行确认 |
| 操作交互 | C=S，单 Q 保存并退出，D 丢弃，H 仅空闲可用；启动、段收尾、HOME、触觉复位边界撤销旧 B/H/T，不恢复旧暂停/确认状态 |
| 来源时间 | 缓存重复读不刷新来源时间；arm、hand、相机、VR、两路触觉有各自样本时间；主机接收时间不冒充曝光时间 |
| 数值与几何 | 关节弧度、XYZ 米、RGB uint8、depth uint16 原始计数及实际深度比例、点云颜色范围；既有坐标变换方向、指序、FK、滤波和采样结果不变 |
| 状态与动作 | 实测关节与目标分开；`fingertip_xyz` 是实测关节 FK 的五个 tip frame 原点；19D joint action / 21D EEF action 保持绝对目标，Raw action 仍是提交尝试 |
| 触觉 | aggregate `(5,3)` 与 dense `(5,120,3)` 各自有效、各自计时；保留原生未知物理尺度，不改成牛顿或布尔接触，不假定 dense 求和等于 aggregate |
| Raw/Canonical | Raw 完整关闭、计数一致后发布，零行不发布；写入/关闭失败保留 staging；Canonical 固定 13 字段、shape/dtype 和从第 0 行开始的连续前缀不变 |
| 清理与错误 | 一个主错可追踪，独立后错仍报告；某资源失败不阻断其他资源关闭；取消不能覆盖停止失败；真实未回收 writer/model 仍可显示 pending |

不删除“观测准入”和“实际 SDK 发送前复查”中的任意一层来凑减量：两者之间经过计算、等待或 IPC，前一刻成立的时效不保证发送时仍成立。来源年龄、动作有效期和 worker 失联超时也不是同一个条件。

## 4. R1：用同一把锁确定 D 与保存决定的顺序

定位：`teleop/loop.py::request_discard/finish_recording/end_episode`、`deployment/session.py::request_discard`、`deployment/rollout.py::finish_recording`；下列模块路径均相对于 `dexmani_real/`。

基线有两处不同的窗口：D 先 `stop_signal.set()` 再获得锁，收尾可先保存；Policy 读取局部 `save` 后至 recorder 设置 `finish_request` 前，D 仍可能被描述成“已接纳丢弃”，实际却不能改变已读取的决定。修复真实顺序，避免再添加一个承诺过强的标签。

实施设计：

1. D 回调先进入现有 `motion_lock`，存在当前段时设置 `discard_requested=True`，随后调用现有停止方法。删除 D 回调开头独立的 `stop_signal.set()`；回调继续只写意图/撤权，不调用 SDK、不等待 writer。
2. 收尾在同一把锁内读取一次局部 `save = not discard_requested`，随后释放锁再调用阻塞的 `recorder.finish(...)`。这一读取就是保存决定的确定时点。
3. D 先取得锁，随后收尾读取，结果必为丢弃；收尾先读取，之后的 D 只保证停止，不逆转已经确定的保存决定。后者允许更新“请求过 D”的字段；发布、结果和音频必须依据局部决定或 recorder 的实际结果，不能再读这个可变请求字段来推断已保存/已丢弃。
4. 删除 Policy 以 `recorder.finish_request is None` 充当保存决定边界的判断。`finish_request` 继续服务 recorder 自身的一次性收尾，不再被键盘层当作另一套流程状态。
5. Teleop 不再通过提前设置 `run_id=None` 标记保存决定已确定；段身份在既有段结束路径清理。新段的 `discard_requested=False` 与段身份的建立仍在现有短临界区内完成；Policy 新 Rollout 在对外赋值前完成初始化。
6. Q 不清除已有 D 意图；无录制时 D 仍停止；新段不能继承上段残留的 `discard_requested`。D 针对回调实际处理时的当前段，不新增按键发生时刻的归属协议。保留现有输入代际撤销，不新增 callback 计数、段提交状态或两阶段 finish API。
7. Esc 保留“先通知执行器，再等待本地锁”的独立急停路径。不要把 D 的数据意图排序推广成所有停止信号都必须等锁；也不要把 SDK、音频或文件等待搬到 `motion_lock` 内。

验收用事件屏障控制顺序，不靠随机 sleep 碰竞态。覆盖 Teleop 和 Policy：在首次停止信号可见处暂停 D，同时启动收尾；D 先于决定、决定先于 D、Q/D 交错、无录制、新段重置。断言实际传给 recorder 的保存值、发布结果与提示一致。旧测试若要求“收尾开始后 discard_requested 必须始终 False”，改为验证外部行为；不要为了维护一个无用内部断言添加状态。

## 5. R2：只修尚未发布目标的本地到期分类

定位：`robot/xarm7/protocol.py::ArmChannels.publish_goal`，联查 `robot/system.py`、`replay/joint_replay.py` 及其上层取消处理。

- 仅将取得 goal/generation 锁、通过代际/停止检查后，`monotonic_ns() >= expires_ns` 的分支改为 `OperationCancelled`。保留原原因描述和 finally 解锁；该分支不写入新的 mailbox 目标。
- 等待 mailbox 锁失败、等待 generation 锁失败、真实 SDK 超时、未确认结束的设备请求仍是故障；不批量转换 `TimeoutError`，不扩张成新的异常层级。
- 保持现有停止和撤权路径。取消与停止失败同时出现时，最终仍报告真实故障。不要将整条 arm/hand 提交统称为“无副作用”。

验收调用真实 `publish_goal`：锁取得成功后推进时钟到过期点，确认未新增 mailbox 目标、锁已释放，回放落为 REJECTED 且执行停止；分别验证锁等待失败、SDK 超时及停止失败仍为故障。只在上层 mock `submit` 抛取消，不能证明这个遗漏分支已修复。

## 6. R3：writer 等待预算与产物完整性分开裁决

定位：`data/recorder.py::EpisodeRecorder.finish/check_error/write_episode`。

- 保留首次 finish 的等待预算、入队 timeout、join timeout 和首次失败粘性。
- join 后删除 `or time.monotonic() >= deadline` 这一附加失败条件；此处只判断 writer 是否仍活着，然后检查 writer 记录的实际错误。监督线程恢复较晚不能倒推 writer 没按时完成，更不能否定已确认完整的产物。
- 不记录新的 writer 退出时间来“证明按时”，不新增延迟发布、后台重试或自动恢复。已经观察到 writer 仍存活并记录失败的调用，之后再 finish 也不能偷偷变成功。
- 保留 `RawWriter.finish/close` 的完整性与计数检查、零行处理、staging 和非覆盖发布语义；不得把“线程退出”单独等同于“文件正确”。

验收：writer 已成功退出，但调用线程恢复时超过 deadline，完整非空产物应可发布；writer 仍活着、queue 满、write/close 失败及计数不一致均不能发布。使用小型真实 Raw fixture 验证产物完整性，调度竞态可用 fake clock/thread；不靠延长默认超时通过测试。

## 7. R4：保留主 I/O 错误，减少错误包装的信息损失

定位：`utils/io.py::_write_text`、`data/recorder.py::check_error`、既有 `utils/log.py::error_detail` 调用方。

1. 原子 replace/publish 成功后临时路径已被消费，直接返回；不再无条件 finally unlink。
2. 写入或发布失败时才尽力 `unlink(missing_ok=True)`。删除失败记录一条含 traceback 的清理日志，然后 bare raise 原始异常；调用者看到的主异常对象、类型和 traceback 不被清理异常替换。保留既有覆盖/不覆盖政策，不加入跨文件系统复制兜底或重试。
3. `RecordingError("Recording failed")` 会使扁平结果丢掉具体 I/O 原因。用一层简短消息包含原异常类型与 message，同时保留原生 `__cause__`；不要递归遍历异常链、构造 evidence 列表或重复包装已经是 RecordingError 的错误。

验收：发布错误 A 与临时清理错误 B 同时发生，对外仍为 A，B 有一次日志；成功发布不再删已不存在的临时文件；目标已存在仍按原政策失败。录制的具体 I/O 原因在现有结果中可读，原 cause 保持可追踪，重复检查不层层包装。

## 8. S1：统一新鲜度表达，删掉错误频率关系

定位：`config/xarm7.py::ArmConfig.validate`、`utils/timing.py::sample_is_fresh`、`robot/system.py`、`robot/xarm7/worker.py`、`robot/xhand/driver.py`。

### 8.1 频率配置

删除 `feedback_hz <= servo_hz` 限制及其误导性报错。当前 worker 分别调度反馈和 servo；review 中实际 worker 的 fake clock 验证能按 50 Hz 反馈与 25 Hz servo 各自运行。此验证只证明调度不要求该大小关系，不证明真机达到任何速率。

保留有限正值、反馈周期小于 freshness budget 的必要条件、机械限位与数值检查；这些必要条件不宣称可以保证实际反馈及时。默认值不变，不增加“允许高频反馈”开关或调度策略配置。验收合法的反馈高于 servo 配置可通过，非法频率与不足的新鲜度预算仍拒绝。

### 8.2 判断只复用已有函数

对当前手写的等价判断使用 `sample_is_fresh(timestamp_ns, max_age_s, now_ns)`，保留 `0 < timestamp <= now` 且 `age < budget` 的精确边界。逐处核对原判断、时钟和单位；没有等价语义时不要机械替换。

每个实际决策点取当前时间；同一次决策可共用一个 now，不跨耗时计算/IPC/SDK 复用早先的 now。只合并表达，不删真实发送边界，不新增 FreshnessPolicy/SourceStatus/Contract。保持现有状态锁等待预算，未证明需要就不另加 timeout 参数。

`Robot.read_initial_state` 等待的是 arm 与 hand 的必要初始反馈，错误文本改为 robot initial feedback，不再统称 XHand 故障；无须新增来源健康枚举。

补充轻量边界用例：零时间、未来时间、年龄刚好等于预算、预算内时间；保留因计算或等待而在发送前过期的回归。缓存重复读取仍使用真实来源时间。

## 9. S2：删指定固定标签，不删实验所需数据

删除前用 `rg` 查本仓读写、脚本、测试及已知 Policy 消费者。下表字段在审查基线没有决定当前数值/准入的消费者；若最新代码出现真实消费者，先确认用途，不能以“metadata 都冗余”为由直接删除。

| 位置 | 删除对象 | 保留或替代 |
| --- | --- | --- |
| `data/raw.py` | `TACTILE_PROCESSING`、`TACTILE_AXES` 及 `hand_tactile_processing/hand_tactile_axes` 固定写入 | 原生单位、指序、hold 年龄、触觉复位实际结果继续保存；处理与坐标语义写入 README |
| `data/raw.py` Teleop 属性 | `vr_hand_side`、`vr_coordinate_axes`、`vr_position_units`、`vr_landmark_order`、`vr_timestamp_source` 固定写入 | 实际 VR 数据、sequence、来源时间与固定数据结构不变 |
| `data/raw.py` timing 组 | `clock`、`missing_value`、`camera_timestamp_source` 固定解释属性 | timing rows、顶层 RawReader 使用的 `timestamp_source` 和实际时间值保留 |
| `data/export.py::create_store` | 描述性 `timing_admission` 字典，以及仅为它计算的局部 period/tolerance | `dt`、实际前缀判定、quality 的 row_start/row_stop/trim_reason 等继续保留 |
| `data/export.py` quality/log | `readout_over_tolerance_rows` 和相应日志片段 | readout 的准入上限仍为 dt；control-grid 的 dt/4 容差仍保留；最大 readout 等有用统计不变 |

readout 统计使用 dt/4 不是已证实的数据准入错误，但容易把不同规则混为一谈；本轮直接删除，避免为这条诊断再新增阈值配置。control-grid 偏差与相邻采样间隔检查也各有用途，不一并删除。

把删除的固定语义整合进现有 README 的数据格式说明，不新建一套 schema 文档或自定义版本机制。新数据不再写这些属性；旧数据中保留它们时现有 reader 仍可读取，无需迁移、历史兼容分支或修改旧文件。验证数组、计数、截断位置和首个截断原因不变。

继续保存实际变化的标定、深度比例、URDF/数值配方、实际时钟、实验配置、必要单位与指序。字段“只写不读”不自动代表无用，人工复现实验也是消费者。

### 9.1 明确保留的检查与标识

- 保留 `deployment/preflight.py::_restore_cloud_config` 的 exact-field 检查和 `config.validate()`。当前 recipe 是数值配置，不是自由 metadata；过滤掉未知键可能忽略新版本影响数值的字段。当前没有必须接受额外注释键的需求，不建立 allowlist、版本映射或默认值回填。
- 保留 `source_id` 为便宜的人工归档标识，不新增 hash、内容身份、split/resume gate；配套 Policy 当前已无待删除的同类强制门禁。
- RawReader 外部 shape/dtype/单位检查与内部处理后的数值检查不能一概合并；原数组有限不保证 float32 转换或 FK 输出仍有限。
- `training_prefix` 只证明当前实现检查过的连续前缀条件，不是所有模态的历史新鲜度证书。它没有重演完整的在线 arm/hand/camera 准入，文档不得作此承诺；本轮不另建离线新鲜度认证框架。

## 10. S3：保留音频，用一个小方法完成尾音与关闭

定位：`teleop/audio.py::AudioFeedback`、`teleop/run.py`、`deployment/session.py::Session.close`。

保留全部 WAV、现有后台非阻塞播放、故障抢占、无播放器/缺文件 warning 与禁用行为。begin 仍在首个正常目标提交返回后，save 在实际发布后，home_done 在 HOME 实际成功后；不增加音频状态协议，也不恢复无消费者的 pause/resume 接线。

在现有 AudioFeedback 上增加一个普通小方法，例如 `finish(successful)`：选择 end/emergency，使用原有 5 秒尾音等待，finally 调用 close。两入口复用它，调用方仍负责自己的成功判定和异常结果，不把 Session、文件写入回调或运行状态传进音频类。

Policy 最终关闭顺序统一为：

1. 段停止并完成既有录制收尾；关闭 robot/keyboard/sensors，关闭并按原预算回收 model，保留真实 pending/failed 信息。
2. 依据此时已知结果播放尾音并关闭 audio；音频关闭错误仍进入原有会话错误与非零退出，不能为减少分支而一律降级成 warning。
3. 汇总包含音频关闭结果的最终状态，执行一次最终 `save_results()`，然后确定退出码。

删除 `results_saved` 及音频关闭失败后第二次补写最终结果的分支。启动和每段结束时有实际用途的结果快照继续保留；“一次最终写入”不是“整场只准写一次”。设备与模型关闭应先于尾音等待，任何音频错误不能使设备清理被跳过。

这里明确接受一个小的语义取舍：`操作结束.wav` 表示控制会话已结束，不保证随后最终 JSON 发布成功。最终写入失败仍打印 traceback、返回非零；不重开播放器、不补播、不重试发布，也不伪称失败信息已成功写入磁盘。进程日志与退出码承担这次 I/O 失败的报告。

验收 fake player 下的正常退出、已有主错、audio.close 错误、最终 JSON 写入错误、缺播放器；检查资源关闭顺序、播放器只关闭一次、只调用一次最终保存，音频关闭失败被包含在成功写出的最终 JSON 中。真实可听效果仍列为人工检查。

## 11. 官方代码对照：借用简单组织，不照搬失真行为

以下固定提交是设计参照，不是用行数或功能数量给 Real 打分。保持对照结论范围：参考实现中的特定路径，不能推广成整个项目都错误。

| 官方代码 | 可借鉴 | 本项目不照搬 |
| --- | --- | --- |
| [ManiUniCon Robot](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py)、[xArm driver](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/robot_interface/xarm6_robotiq.py) | 控制与状态更新分别调度，流程直接 | 查询失败的零值补位、重复 validate、缺少动作有效期；不据此删除 Real 的真实边界 |
| [LeFranX arm/hand 提交](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py) | 用直接调用表达设备组合 | `synchronize_actions` 名称不能把顺序调用变成同时/原子执行；Real 不增加类似强承诺开关 |
| [pi-r2-flow 相机](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/cameras/realsense.py)、[部署收尾](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py) | 直接的数据读取与推理主线 | 旧图重用配客户端当前时间不证明样本新鲜；结果保存抛错也不能阻断后续设备关闭 |

本轮不引入三个仓库的框架、配置体系或新的依赖。对照的作用是识别真正独立的职责，而不是追求表面上“更像官方代码”。

## 12. 执行批次与有针对性的验证

按以下顺序形成可独立检查的小批次，依赖变更顺序完成，不先大规模重构。

| 批次 | 内容 | 主要既有测试位置 |
| --- | --- | --- |
| A | R1 保存意图、R2 本地到期分类 | `tests/test_interaction.py`、`tests/test_errors.py`、涉及的回放用例 |
| B | R3 writer 裁决、R4 主错与录制消息 | `tests/test_data.py`、`tests/test_errors.py` |
| C | S1 频率/新鲜度、S2 指定元数据及文档 | 复用现有相关测试，确认导出数组与 prefix 不变 |
| D | S3 音频和 Policy 最终结果收尾 | `tests/test_audio.py`、`tests/test_policy.py`、相关交互/错误用例 |

测试调用真实生产函数，只在 SDK、时钟、线程调度、播放器、文件失败注入等边界替换依赖。review 的隔离复现可作设计参考，但不把 AST 截取源码、复制生产实现或大型通用 fake 框架提交成长期测试设施。优先修改/参数化现有测试，避免测试重复编码实现细节。

四项 bug 应各有能在修复前触发、修复后通过的定向回归。与本轮无关的已通过几何、模型调度等测试不反复重跑；如触及相关实现或出现回归，再扩大验证。元数据删改至少跑一次真实的小型 Raw→Canonical fixture；无对应依赖就清楚说明未验证，不能以全 fake 宣称文件格式通过。

按仓库 AGENTS 禁用缓存：pytest 用 `-p no:cacheprovider`，ruff 用 `--no-cache`，mypy 缓存放仓库外；不新增忽略规则遮盖生成物。运行改动范围的语法/静态检查与 `git diff --check`。不为收拾既有 Session 类型注解问题增加 Optional 防御分支、无意义断言或宽泛 ignore。

现场检查单独保留：低速启停与急停、D/S/C/Q、准备/收尾边界、HOME 取消、真实采样延迟、音频抢占和退出。未实际运行就写“未执行”，不从 fake 测试推断物理停止效果或实时性能。

## 13. 不阻塞本轮的条件性后续项

原实施记录对非零 Brown 畸变的 Raw 做过合成几何对照，但缺少实际部署 checkpoint，尚不能断定当前模型命中纯 pinhole 融合消费者。本轮不静默 rectification、不改变 RGB-D/点云数值、不新增“RGB+点云一律拒绝”的门禁，也不跨仓修改 Policy。

若现场实验确实使用该消费者，另开明确的几何任务：用实际 Raw、checkpoint 和消费者证明不一致，再在统一 rectification 与消费者支持畸变之间选择并验证。没有这些材料时，保留条件说明即可，不将它勾成已修复，也不因此留下本轮四项确定 bug。

## 14. 完成标准与实施记录

- [x] R1 的实际保存决定在 Teleop/Policy 中一致，覆盖首次停止可见窗口；无新状态或锁内阻塞 I/O，Esc 快速通知保留。
- [x] R2 只修本地目标到期分类，真实锁/SDK/停止失败仍正确传播。
- [x] R3 不再因监督线程迟醒误拒完整产物，实际 writer/文件失败仍不发布。
- [x] R4 保留原始 I/O 主错，录制结果显示具体原因，没有递归错误框架。
- [x] S1 删除错误频率关系，等价新鲜度表达统一，时间与副作用边界保持。
- [x] S2 指定固定标签与误导统计删除；数组、必要配方、单位、指序、时钟和前缀语义保持；exact-field 检查保留。
- [x] S3 音频功能完整，尾音流程共用，Policy 只写一次最终结果，资源和错误不丢失。
- [x] README 与行为一致；没有新增长期报告目录、架构层、配置项或兼容协议；无无关文件、数据改写或他人改动被覆盖。
- [x] 实际检查、未验证条件和真机待办如实记录，未运行的检查未标通过。

实施完成后在本文末尾追加一段简短结果或勾选上述项目，不创建第二份验收报告体系。报告包含：行为变化、删掉与保留的机制、实际测试命令与结果、未执行项、`git diff --stat`；分别统计运行时代码、测试和文档的净增减。净减量不是硬指标，不用删必要检查、压缩排版或移动文件伪造精简；优先让主路径与责任归属更容易理解。


### 2026-11-03 首次实施记录

- 起点为干净的 `main@592ab9d`；相对 `18d77b0` 仅新增本任务书，R1–R4、S1–S3 均仍适用，按 A→D 完成。第一轮工作未重做；Policy 新 Rollout 在对外赋值前初始化丢弃意图已满足要求，因此没有形式修改。
- D 与保存决定复用 `motion_lock`，局部 `save` 决定发布和提示；删除提前清空段身份及键盘层 `finish_request` 判断，保留 Esc 快通知。arm 仅锁取得后、mailbox 发布前的确定到期改为取消；锁等待、SDK 与停止失败仍为故障。writer 只按存活及实际错误裁决，首次失败粘性、完整关闭/计数/零行/staging 规则保留。原子写清理失败记录一次而不替换主错，录制错误显示具体 I/O 原因并保留 cause。
- 删除无依据的频率关系、指定固定元数据、`timing_admission` 和 readout 的 dt/4 统计；删除前核对本仓及只读 Policy 基线/当前消费者，无运行时依赖。必要单位、指序、hold、标定、数值 recipe、exact-field、source_id、在线准入与发送前复查保留，各自职责未合并。音频资源、非阻塞与抢占保留；尾音/关闭共用 `AudioFeedback.finish`，Policy 音频后只写一次最终结果，音频错误影响 JSON 和退出码，JSON 失败不重播或重试。README 同步上述行为与格式说明。
- 四项 bug 回归先在旧实现验证失败：R1 首次停止信号屏障在两个入口 2 failed；R2 真实 mailbox→Robot→Replay 的本地到期误落 FAULT；R3/R4 等新增数据回归 6 failed、1 passed（含迟醒、清理覆盖、成功后多余 unlink、错误消息）。测试调用生产函数，只在外部 I/O、时钟、锁竞争和调度边界控制行为，没有源码截取或镜像实现。
- 首次实施验证命令：`PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -p no:cacheprovider -q tests/test_interaction.py tests/test_errors.py tests/test_data.py tests/test_audio.py tests/test_policy.py` → **171 passed in 0.82s**（修改前同组为 128 passed）。阶段联验 A 97、B 63、C 81，D audio/policy 59 passed；计数有重叠，不累计。
- 真实小型 HDF5/MP4→Zarr fixture 前后对照：26 个 Raw/Canonical 数组的 dtype 和数值逐项完全一致，包含全部 13 个 Canonical 字段和 timing rows；连续前缀位置/首因保持，补测 readout 在 dt/4 与 dt 之间接纳、到达 dt 截断。对照产物只在 `/tmp`，未改写实验数据。
- 静态检查使用上述 Python 的 `-m ruff check --no-cache --select E9,F,I`，参数为 `git diff --name-only -- '*.py'` 所列 18 个修改文件，全部通过；同 18 文件用 `compile` 内存语法检查通过。上述 Python 运行 collect_teleop/run_policy 的 `--print-config`（JSON 可解析）及 collect_teleop/run_policy/export_zarr/replay_episode 的 `--help`，6 项均退出 0；这些入口均在设备初始化前返回。`git diff --check` 通过，未生成仓库内禁止缓存；不宣称全仓类型或全规则静态检查通过。
- 未连接或驱动真机，未修改配套 Policy、checkpoint 或历史数据，未推送/合并。现场低速启停、急停、D/S/C/Q 与准备/收尾边界、HOME 取消、真实反馈/采样延迟和音频抢占/听感仍**未执行**；真实模型及部署 checkpoint 未验证。非零 Brown 畸变保持条件性后续项，不宣称已修复，未改图像/点云或增加准入门禁。
- 首次实施阶段 `git diff --stat`：20 files changed, 551 insertions(+), 161 deletions(-)。分类：运行时代码 13 文件 +54/-96（净 -42），测试 5 文件 +466/-49（净 +417），文档 2 文件 +31/-16（净 +15）；总净 +390。未增加新文件、依赖、配置、状态协议或报告体系。


### 2026-11-03 全链路审查收口

- 保留首次实施的未提交改动，沿 Keyboard/Teleop/Policy→Robot→arm/hand、Recorder→Raw→Canonical→replay/viewer/脚本及音频→最终 JSON 核对实际调用与消费者。未发现此范围的 SHA-256、哈希门禁、迁移或版本兼容层；没有为满足“删除”制造改动。`source_id`、非版本格式标签、exact-field、数值/单位/指序及完整发布检查各有用途，继续保留。
- 删除 Teleop 事件消费阶段重复执行的 S/C/D/Q 撤权和意图写入，统一由键盘回调负责；控制线程只结束段和提示。hand 在读取反馈缓存后用同一个 now 检查目标期限和来源年龄，消除缓存读取期间到期仍进入 SDK 的窗口，同时减少一次时钟读取。两项真实函数回归分别先在旧实现 4 failed、1 failed。
- 音频播放/等待主错与关闭后错同时发生时保留主错，关闭后错单独记录一次；无主错的关闭失败仍传播，不降级为成功。Policy 音频中断锁存急停，统一最终 reason 的取值与空消息异常的类型说明，最终 JSON 仍只写一次。上述新增定向回归均先暴露原问题；Teleop 主异常类型补为 `BaseException | None`，尾音入口明确 bool/None 接口。删除仅注入已删除说明属性的过时测试及 README 中相应历史兼容描述，未新增兼容分支。
- 最终测试：`PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -p no:cacheprovider -q tests/test_interaction.py tests/test_errors.py tests/test_data.py tests/test_audio.py tests/test_policy.py tests/test_viewers.py` → **185 passed in 0.85s**。真实 Raw→Canonical、13 字段、连续前缀、资源关闭与故障分类回归均在其中；不代表硬件动作成功。
- 29 个修改文件及相关上下游 Python 文件的 `compile` 内存语法检查和 `-m ruff check --no-cache --select E9,F,I` 通过；同范围排除测试后的 24 个生产/脚本文件用 `-m mypy --cache-dir=/tmp/dexmani_chain_review_mypy --ignore-missing-imports --follow-imports=silent --namespace-packages --explicit-package-bases` 通过。13 个修改源文件额外启用 `--check-untyped-defs`，与临时 HEAD 源码副本对照为 50→48 条诊断，无新增；当前 Raw 生命周期 8、Audio 类型 3、Session 生命周期 37 条仍未解决，不宣称增强检查通过，不用 ignore、断言或运行时兜底掩盖。
- 只在 `/tmp/dexmani_chain_build_qkd9hd4a` 复制当前源码，通过 `setuptools.build_meta.build_wheel` 构建 wheel；85 项 Python/标定 JSON payload 齐全，解包后四个受影响模块确认从构建产物导入。6 项无硬件 CLI 配置打印/help 再次通过。该验证不保证 wheel 自带仓库外机器人资产，运行仍按 README 保留 checkout。
- README 与最终交互、时效和音频错误行为对齐；`git diff --check` 及禁止缓存检查通过。无新增文件、依赖、状态协议、配置兼容或报告目录；未连接真机、修改实验数据/配套 Policy 或推送/合并。真机停止、真实采样延迟、音频听感、实际模型及 Brown 畸变消费者仍未验证。
- 全部未提交改动 `git diff --stat`：20 files changed, 652 insertions(+), 188 deletions(-)；运行时代码 13 文件 +73/-113（净 -40），测试 5 文件 +536/-59（净 +477），文档 2 文件 +43/-16（净 +27），总净 +464。相较首次实施，运行时代码净 +2、测试净 +60、文档净 +12；新增量用于已复现错误与实际验收，未扩展架构。
