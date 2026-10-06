# Codex 任务书：DexMani Real 最终简化与论文开源化整改

日期：2026-10-07（Asia/Singapore）  
主仓库：`haoyangzhanglab/dexmani_real`  
设计审查基线：Real `a8a7f908702ba5bfa3abd505dcb7119d41faba5c`；Policy `982909ed5e6187c0ba36e55f3ccc371548211525`。  
状态：**修订后的设计终审通过；实现、原生测试与真机验收未由本任务书宣告完成。**

## 0. 目标、权限与事实边界

目标是让固定硬件研究链路简洁高效、正确好用：删除无关依赖、重复表示、无需求回退和职责混杂，使陌生研究者能沿采集、导出、训练桥接、部署理解代码。不是重新搭建框架，不是要求所有验证只出现一次，也不是只再输出一轮审查报告。

本文件自包含。50 项历史议题在第 10 节逐项处置；编号是覆盖索引，不是 50 个 bug，不以完成数量或删行比例评价工作。无需此前聊天、下载附件或旧任务书即可执行。以本任务书的终审修订和用户后续指令为本次具体范围；旧 `CODEX_*TASK*.md` 不自动叠加成新增义务。保留用户指定的历史文件，不批量改写它们的历史结论。

开始时读取两仓根目录及受影响目录适用的 `AGENTS.md`，检查当前工作目录、HEAD、`git status` 和已有 diff。基线 SHA 只用于定位与比较，不是 checkout/reset 目标。用户本地后续实现优先按事实核对；已修复项不要重复修改。保留无关修改；冲突只阻断相关项，继续独立项目，不 stash、清理或覆盖现场。

授权本地 Codex 修改 Real 相关源码、配置、测试和必要文档，并执行现有环境中的纯离线检查。相邻 Policy 仅限 P2 的 RTC warmup 桥接及其定向测试、确因本次边界变化必须更新的消费点；不顺便执行 Policy 的完整训练整改任务书。不在 Real 中复制一套 Policy 推理实现。Policy 仓库缺失或不在允许写入范围时，报告该项边界并继续 Real，不修改沙盒权限或复制替代实现。

默认不 commit/push、不开 PR、不拉取合并或重置分支，不修改 `.codex`、全局代理配置或凭据。当前用户对远程写入本任务书的授权，不等于授权本地实现自行提交。

不得连接或操作 xArm、XHand、相机、VR/HTS，不得运行 live teleop、物理 replay、rollout、HOME、诊断或真实标定写入。`execute=False` 仍可能连接硬件，不能当作离线测试。可以修改这些入口并使用隔离 fake 验证。禁止改写/删除已有 Raw、canonical 缓存、模型、normalizer、实验结果、现场标定或模型资产；禁止正式训练、批量真实数据导出、GPU 消融、依赖大升级和自动资产迁移。存在未知硬件副作用的命令按有副作用处理。

此前反例只覆盖人工转录函数、合成输入和 fake CPU agent；仓库提交信息中的离线 PASS 也不是本地验收。实施时调用真实模块的现有测试，不复制函数后声称生产实现已通过。未运行原生依赖、真实权重、GPU 或硬件检查时分别写明，不能用模拟替代证据。

## 1. 终审调整：先避免“简化”引入新错误

1. **不恢复错误的几何统一。** 当前物理 hand-base/fingertip FK 使用实际安装补偿（默认 -0.015 m），碰撞/规划使用原始 URDF/SRDF/网格的标称几何（-0.005 m）。不再生成运行期 URDF，不修改几何、HOME、白名单或阈值来追求 frame 相等。标称模型不自动是实际装配的保守包络。
2. **配置按用途验证，但独立入口仍需检查。** 不只删全局循环而漏掉构造器强验证，也不全删 `__post_init__` 后让直接 pointcloud/driver/export API 接受非法输入。结构规范化、用途检查和热路径分开；无认证令牌或 validator registry。
3. **区分点云配方与当前现场配置。** Policy 使用 checkpoint 保存的点云 recipe，不能被当前默认 `runtime.pointcloud` 覆盖。桌面是否解析须考虑 HOME 环境与该有效 recipe 的真实消费者，不能只看 `environment.table.enabled`。
4. **命名改进不能破坏历史模型。** 保留 checkpoint 中已有 pointcloud 序列化键（包括 `outlier_candidate_multiplier`）和数值含义；仅修正内部名与说明。不以改名为理由拒绝旧 checkpoint、添加通用迁移或使用宽松恢复。已有公开 YAML 键也不为了观感一律重命名；能靠内部职责调整解决的，不制造配置版本切换。
5. **RTC 最小修复兼顾两个边界。** warmup 需要 prefix 的条件为 `rtc_delay > 0 or self.rtc_guidance_cap > 0`：修复 beta=0、delay>0，也保留原 API 的 beta>0、delay=0 软前缀 warmup。不能仅把条件换成 delay>0 后缩窄已有行为。维持 reset、RNG 和最终清理顺序。
6. **去 trace 副本前先保证归档所有者。** policy 录制必须有同一 attempt/results 归属；直接调用 runner 也不能只给 recorder 却丢 trace。新完整 trace 只写 attempt 侧，但保留未发送 selected target、部分 dispatch、零行和发布失败证据。Raw 行号不是 slot 号，不能以总行数猜对应关系。
7. **相机固定布局必须支持独立 attach。** 删除逐帧尺寸后，`create=False`/spawn 接收者仍能从初始化布局确定 shape/dtype/offset，不能依赖生产者私有 Python 属性。采集源时间与 ring 发布时间有不同含义，不因都叫 timestamp 而合并。
8. **归零显式，不增加新的隐性运动。** 不只是把自动归零从 connect 搬到另一个自动调用点。需要操作者明确无接触准备、owner 串行执行、分步取消；不为归零额外自动 HOME/开手，也不把辅助触觉缺测变成全部采集的硬门。
9. **算法和通信政策不混入等价重构。** 本次不自动统一 sync/async/RTC 的 CRC 处置，不改 WAIT/stop 语义，不删除点云/IK/HOME 启发式。它们保留为有明确实验条件才实施的独立变更；不能为凑完成度默认运行消融。
10. **本地验收不依赖“万能通过”。** `--help` 不应导入优化器/设备/桌面后端；查看声明配置不等于标定和运行已就绪。缺依赖可准确报错或标未验证，不能降低生产配置后声称其已通过。

## 2. 保留的行为与目标结构

保持现有主要目录和用户工作流，不建设统一 BaseRunner、RuntimeManager、ArtifactRegistry、ValidationPipeline、插件发现、DI 容器、迁移引擎或另一套录制格式。不整体迁移到 LeRobot/ManiUniCon，不复制其完整框架。

```text
轻量 CLI / 一份实验 YAML
  -> 本 workflow 的解析与连接前准备
  -> session 组装具体 robot / model / realizer / recorder / results
  -> runner：观测、方法计算、计划、动作与结束决策
  -> 现有 robot owner 的最终 SDK 准入
  -> 现有 recording 层保存物理事实和归档结果

Raw -> 全 13 字段 canonical -> Policy 按实际字段/角色取窗口
    -> train-only normalizer -> checkpoint -> 保存统计的推理桥接
```

必须保持：四状态与 run_id；每个 SDK 入口的动态授权/时限；两设备调用前的合法目标检查；partial/unknown/CRC dispatch 事实；最多一个未回收 Future；物理冻结前缀及发送前复核；本来需要的 hand state 与 source frame freshness；Raw 发布后不可变、失败前缀、NaN、背压、staging；原行号与 episode 边界；13 字段及 SDK/URDF 顺序；相机对应 Raw 的内外参/depth scale；无二次触觉扣偏置；保存的 normalizer 和 N/H/A/dt；retargeting 每源帧成功/失败缓存。

WAIT 是无新目标，不是停稳；SDK accepted 不是到达；协作 supervisor 不是独立 watchdog；host read completion 不是真实采样时刻；warmup 不是实时上界；NumPy seqlock 不是可移植的内存序证明；模型线程关闭不是有界进程退出；在线端点自碰撞不是连续环境避碰。保留真实功能和准确说明，不再围绕这些限制添加虚构的“安全已通过”状态。

## 3. P1：轻量入口、用途验证与一次参数解析

主要落点：`config/experiment.py`、`config/control.py`、`config/pointcloud.py`、`deployment/config.py`、各 `examples/` 入口、`deployment/session.py`、`teleop/session.py`、`sensor/pointcloud_worker.py`；路径均相对 `dexmani_real/`，`examples/` 除外。

### P1.1 入口及验证责任

- 保留 dataclass、YAML 和 argparse。先参数解析，再声明配置展示，之后才导入/构造真实 workflow。轻量任务名、默认值不经 session 导入。不得为 `--help` 启动模型 worker、查设备、打开显示后端或读现场标定。
- 所选 retargeter、kinematics/realizer、必要模型与静态资源的可失败构造在设备连接前完成；只有真实硬件能够给出的状态留在连接后检查。把构造移前必须同时补全资源清理，不能在 model warmup 失败后泄漏对象。
- 配置加载拒绝未知键、错误结构和不可解释类型，保留 tuple/基础类型规范化；未使用部分的数值阈值不应阻断当前 workflow。PointCloudConfig 的负阈值等用途检查从全局构造负担中移到其真实使用边界。不要靠 `validate=False`、全局跳过或捕获所有异常实现。
- 梳理 direct public API：独立 pointcloud 构造、ProcessingConfig/export、driver、ActionRealizer/规划等仍在入口拒绝实际非法值。必要重复的小边界检查可保留；内部逐帧计算不反复扫描全部 runtime 配置。
- HOME、录制和 hand-disabled 调试也算实际消费者；某字段不是模型输入，不代表它在该 workflow 无用途。不得删掉已有手缺席确认或让缺失必需资源后继续启动。

### P1.2 一个有效快照

- 装配阶段决定哪些消费者需要桌面：启用的 HOME/碰撞 table，或 Policy 保存的有效点云 recipe 中的 remove_table。需要时读取一次，给全部相关对象同一份 tuple/解析后 table；消费者不再次读取 plane_path。
- 覆盖四种组合：二者都不用、仅 HOME 用、仅点云用、两者都用。特别是 table.enabled=False 且 pointcloud.remove_table=True，不能盲用未解析的默认 plane_abcd；反向组合也不能漏解析 HOME。
- 声明配置展示不读文件；实际 run_config 保存解析后真正使用的值及原有必要来源信息，不以启动前旧值冒充最终值。不新增配置快照链、文件 watcher、hash gate 或热重载。
- 相机外参延续会话级快照。VR alignment 也按同样原则传已加载值，不只“预检读一次、使用再读一次”。文件变化仅在下一次明确加载生效。

### P1.3 预算、命名与路径

- 在同一实验 YAML 中提供明确的 execution 小节，最终仍构造现有 ExecutionConfig，不复制其验证逻辑。CLI 默认未指定值不覆盖 YAML，CLI > YAML > 现有默认；三个必需预算未提供时仍连接前报错，不提供伪安全常数。模式、prefetch、beta 的关系检查保留；例如 beta=0 不可被 `value or default` 吞掉。
- 内部 HOME helper 等误导名称可一次改正并更新真实调用者。外部 policy/safety 等已有键不做全仓纯美容迁移；先通过 owner 组装和准确说明消除歧义，不增加新旧同义键。新 execution 只负责此前 CLI 独有参数，不复制模型 timing。
- 为现场标定、输出提供已有配置/入口上的明确路径选择，区分固定 assets、参考标定和用户状态。优先保留已公开默认路径语义、明确其来源，不自动移动/重写已有文件；路径变更独立 diff，不能引入工作目录猜测 fallback。缺相机外参仍可 Raw 录制；首次 table calibration 不要求已有 plane；无关 workflow 不强制标定路径。

**验收：** 在阻止 SDK、nlopt、pynput 显示后端等导入的子进程中运行轻量帮助/声明配置；无副作用。policy 不被无关 VR 数值阻断；direct pointcloud 的非法实际输入仍拒绝。用 fake 计数断言四种 table 使用组合只读 0/1 次；构造后修改临时文件不改变本会话数值。CLI/YAML 优先级、显式零值、缺预算拒绝、保存值与消费者一致；连接前构造失败未调用 robot.connect。

## 4. P2：RTC 零权重 warmup 的最小跨仓修复

Policy 落点：`dexmani_policy/deployment/runtime.py::LoadedPolicy.warmup`、已有 `tests/test_policy_rtc.py` 或邻近桥接测试；Real 检查 `deployment/session.py::_warmup_policy` 与 `deployment/config.py`。

当前 Real 接受 beta>=0；warmup 仅 beta>0 时产生 prefix，RTC 正 delay 与空 prefix 在 predict 中冲突。修复目标不是禁止 beta=0。

- 在当前 API 下，prefix 构造条件使用 `rtc_delay > 0 or self.rtc_guidance_cap > 0`。如本地后续实现已显式传递是否需要 prefix，用其现有单一来源，不再增加一个重复 mode 状态。
- 保留 beta>0、delay=0 原有软先验 warmup；beta=0、delay=0 不额外预测；正 delay、beta=0 有合法 prefix 而底层仍走原采样分支。非法 prefix/delay 仍拒绝，不静默把 delay 清零或切换 sync。
- 维持 warmup 前后 reset、异常 finally reset、正常推理 seed/normalizer/返回完整 P 的约定。生成 prefix 会消耗 RNG，比较 beta=0 与原采样时须从相同 RNG 状态开始；不得误把两次顺序调用数值不同当成算法不等价。
- 沿现有方法检查 DDIM/affine normalizer/VJP 适用性；不得为通过测试关闭 compile、换架构、改 scheduler、换 checkpoint、使用 strict=False 或重新拟合统计。

**验收：** 真实桥接模块覆盖 beta=0/delay>0、beta>0/delay=0、二者正、二者零、非法 delay/prefix；确认 real config 接受到实际 warmup 的调用链。fake agent 只证明桥接；已有实际 DDIM/VJP 测试能运行则运行，真实权重/GPU 未运行单列，不能沿用此前转录 probe 的 PASS。

## 5. P3：session 装配、runner 决策、归档去重

主要落点：`deployment/session.py`、`deployment/runner.py`、`recording/results.py`、`recording/recorder.py`、`recording/frame.py`、`teleop/control/controller.py`、相应测试与 `examples/summarize_policy_trace.py`。

### P3.1 职责搬迁，不增加层数

- session 创建 realizer/recorder/results 并传现有具体对象。runner 不再在构造中加载规划模型、决定输出路径或创建 writer；资源仍只有一个明确 owner。可提取真正纯的 slot/index 计算；不建立调度引擎、BaseRunner 或三个复制 runner。
- 归档文件细节移到现有 recording/results 附近，runner 只作结束决策并提供冻结数据。不要造第二套 session/attempt 状态，勿让 archive 回调修改运动权限或执行 robot.stop。
- 取消、stop 失败、writer 失败、sidecar 失败可能同时发生。保留首个真实结束原因、原异常优先关系和附加错误，始终先撤权/失效，再尝试 stop，随后才做归档 I/O。归档错误不可掩盖已执行事实；失败不跳过必要清理。
- 布尔元组确有歧义时使用一个小型命名结果，字段表达 target 可用、dispatch、interrupted 等原含义；不得把 success 改解释为到达或任务成功。不为每个函数包结果类。

### P3.2 新产物只有一份完整 trace

- 同一 policy 录制必须有 results/attempt 所有者。由装配传入，直接公开调用也在启动前明确拒绝不完整组合；不要靠“有 results 存 sidecar、没有则回写 Raw”保留两条新实现。
- 新 policy 运行的完整 trace 只写 `attempts/<id>.json` 的现有 trace；query NPZ 仍保存形状/类型可归档的预测，包括被拒绝的 NaN/Inf。strict JSON 不接收非有限数组，NPZ 不保存任意对象；读取不用 pickle。
- Raw 保留所有现有逐行物理事实、attempted targets、dispatch、终止原因和必须的结束详情；不再复制完整 policy_trace。不要顺手删 final_selected/final_dispatch 等失败证据，也不重写历史 Raw。旧文件已有字段的只读支持不必为了“无兼容”删除。
- 更新真实消费点和测试。沿已有 raw_path/attempt/query/slot 关联；slot 与 Raw 行不能互换。若既有 trace 无法定位实际提交行，在现有记录事件中增加最小行索引（入队成功才赋值），区分 submitted 与 written，写失败不能报告该行为 published。先验证必要性，不增加全新逐行 manifest。
- 保留开始前 prepared/incomplete、未进入运行、零行、录制失败 staging、已发布 Raw 但 sidecar 失败、取消后晚到 Future 等情况。旧 Future 不归入新 attempt，不因等待它而推迟 stop。无录制的明确 API 不因此自动创建持久化产物。
- 先消除完整 trace 双写，再决定局部字段是否真能由 Raw 恢复。未下发 target、部分 dispatch、没有保存的行不能从成功 Raw 反推，必须保留。query NPZ 当前仍是必需实验产物，不吞保存错误、不新增可选性开关来回避失败。
- 快照在原 owner 上完成，不能让 writer/归档线程读取之后还会变的 events/query_arrays。只保必要 owned 副本，不以节省拷贝为由引入跨线程可变别名。prepare/finalize 可阻塞时保持运动撤权。

**验收：** 复用真实 runner、writer、results；fake robot/clock 驱动。覆盖 begin/reset 取消、零行、全部/部分/unknown/CRC dispatch、迟返回仍 accepted、stop/writer/NPZ/JSON 多重失败、晚 Future 和重复 close。新 trace 恰好一份，Raw 行与提交映射正确；首因与实际 published 状态不被覆盖；无 I/O 在撤权/stop 之前。测试无需 monkeypatch 工厂才能构造 runner，但不为测试便利修改生产状态机。

## 6. P4：固定相机布局与按需资源分配

主要落点：`ipc/camera_ring.py`、`ipc/schema.py`、`ipc/channels.py`、相机 producer、`runtime/observation.py`、各 workflow 的资源构造/关闭和 IPC 测试。

- 一次会话固定尺寸、dtype 和 slot 布局。删逐帧 `rgb_size/depth_size` 与重复 shape；布局在初始化 header/明确 attach 信息中唯一确定。保留写入数组 shape/dtype/contiguity 检查及容量越界保护，不借删 header 删除真正的边界检查。
- `create=False`、按名称 attach 和 spawn pickling 都能恢复尺寸与偏移；新 consumer 不借助源 producer 的私有属性。共享内存名不混用不同布局的活会话；不支持运行中升级旧布局，不造 IPC 迁移框架。
- 保留相机源帧号、源时间、逻辑 sequence、防撕裂标记、读前后核验和 owned copy；`read_sequence` 与点云源相机精确匹配继续存在。源时间与 publish timestamp 不合并；不能用刚复制完的时间把旧输入变新。
- 当前三类 ring 无条件分配改为按 workflow 显式选择。cloud 必须建 camera，recording 也必须建 camera；VR 延迟启动时仍需预建该 workflow 会用的资源。不要只依据模型直接字段漏掉依赖。
- 未用 ring 在少量边界表示为 None/缺省，消费者创建前保证必需资源存在；不要把大量 `getattr(...,None)` fallback 扩散到热循环。无用大数组不分配即可，没必要为几个 flags/Event 做复杂动态系统。
- 部分分配失败回收、父进程所有权、每个 started 子进程退出后再 unlink 保留。按需资源不改变 sensor readiness、timeout、启动/停止时机的实际语义。
- 不替换 NumPy seqlock，不声称纯 Python 交错测试证明可移植锁自由。测试保证当前支持平台上的功能回归，内存序限制仍文档化。

**验收：** 固定容量读写、非法尺寸拒绝、独立 attach、spawn、读后 producer 覆盖不改变 owned frame、按序号读、写入途中读取拒绝、部分初始化清理；各入口启用组合准确。记录未用资源分配消失，不虚报控制时延改善。录制 cloud 源帧覆盖没有实测案例时不引入第二套快照/无界缓冲。

## 7. P5：显式触觉归零与更短的手眼激励检查

### P5.1 基线归零不是无接触认证

主要落点：`robot/robot.py`、`robot/drivers/xhand.py`、`runtime/operator_input.py`、teleop/policy 的 operator/session；检查 replay/校准/调试对 connect 和 tactile 的实际消费。

- 从通用 `DexManiRobot.connect()` 删除隐含自动 tare。driver 继续提供一个 owner 调用的基线采集/核验方法；内部可改为更准确的名称，不增同义 alias。connect 的关节反馈和连接错误规则保持。
- 使用现有操作界面提供显式“确认无接触并归零”的准备操作，优先复用 KeyboardInput 增加一个空闲命令（当前 T 未占用，实施前再核对）。不再增加独立 GUI、输入线程或证书文件；不使用阻塞 input() 阻断既有停止处理。
- 仅在无 RUNNING、无活跃 capture、无启动/reset 准备、无 HOME 进行的空闲 owner 状态执行。请求先撤销旧权限并按已有停止路径处理；不得顺便新增 HOME/开手动作。已有启动 HOME 与其他操作者流程不因本项被重复执行。
- 采样和核验循环检查取消/stop/quit/服务失败，并在发布候选前再检查；回调只查询/消费停止请求，不递归启动新实验。调用 SDK 后才能检查的阻塞限制照实保留，不承诺严格实时取消。
- 保留候选阶段不暴露 bias、成功核验后发布、aggregate/dense 分开。取消或异常不泄漏半个候选；无 bias/失败按原规则输出 NaN 和有效性，不能回退为未经校正的正常值。稳定接触也可得到零残差，不把多采几次称为无接触证明，不发明未知 SDK 单位下的通用力阈值。
- 所有实际消费 tactile 的入口给清楚的准备提示/调用路径。只读关节的 replay 或 hand-disabled debug 不被迫新增触觉步骤；若 replay 确有触觉消费则显式准备，不在低层偷偷恢复自动归零。
- 辅助触觉缺失不变成全部 Raw/关节任务的开始门；要求 tactile 的模型仍按现有 required modality 拒绝不可用输入并给可定位提示。不自动删除缺测 episode，不二次扣 bias。

**验收：** fake SDK 证明 connect 不采 bias；显式空闲操作完成，活跃/准备/HOME 状态拒绝；分段取消与发布前取消不暴露候选；双通道独立结果、stable-contact 反例、无 bias NaN 和关节任务不被阻断。真实无接触、传感器尺度和停止行为均待现场验证。本项是操作协议变化，独立 diff 和文档说明。

### P5.2 激励检查保留准入，删除完整比较列表

落点：`calibration/camera/solver.py`、调用它的 session 及数值测试。

- 保留现有相对旋转最小幅度、轴分离阈值、至少三个姿态、finite/shape 和坐标不变性语义。存在合格轴对即可通过，不物化所有 separations；用点积/流式扫描早停，拒绝路径遍历必要候选。
- 不用“固定第一轴、只比其他轴”的简化替代原存在性判定；它可能遗漏只有其他两个轴之间满足条件的合法输入。不用三轴满秩替代两方向激励，也不任意截断样本。
- 新诊断可记录合格见证或明确命名的统计；不把第一个合格角继续叫 axis_separation_max_deg。改调用者/新报告字段，旧标定文件不回写。精确 global max 无真实消费者则删除，不再另开完整诊断子系统。
- 早停与不存列表不保证最坏时间降阶。保留手眼候选有效性、PnP 失败处理及样本内 residual；不将它们升级为实物绝对精度证明。

**验收：** 已有退化/双轴/坐标变换反例；小规模输入与测试内独立穷举判定一致；专门覆盖非第一轴之间才满足阈值的组合。只证明布尔准入与必要统计，不用旧 global max 精确值锁住新实现。

## 8. P6：方法代码可读性与保守的数据边界

### P6.1 后端与辅助功能减法

- `teleop/retargeting/retargeter.py` 的公共 landmark 检查、掌面变换、尺寸补偿成为纯几何；TAG/DexPilot adapter 跟随各自方法模块，优化器按需导入。保留 retarget/reset、小型显式构造分支和实际算法；不建 registry/抽象后端层，不改变 SDK 顺序、warm start、滤波、pinch 或手指比例。
- 相同 VR sample 的成功/失败缓存继续保留；不要重复求解推进状态。只抽公共数学，不把方法特有差异变成十几个 hook。
- `AudioFeedback` 初始化选择一个可用播放器；删自动恢复工作线程和逐事件多播放器 fallback。保留实际使用的 queue/play 顺序、generation 抢占、非阻塞提交和退出回收。HOME 完成提示不被后续开始提示吞掉。
- 播放器缺失/失败告警后禁用音频，清理 active/pending 并唤醒等待者；play 不在控制线程阻塞或持锁等待子进程。子进程启动异常也要尽力 kill/reap，不能仅丢句柄；音频不能改变运动权限或判定实验成功。
- IK 报告 `cmd_tracking_error_*` 改为准确的 `ik_residual_*` 并更新在用统计/展示；不改 IK 目标或阈值，不建立重复的物理跟踪指标。
- `outlier_candidate_multiplier` 是重过滤之后的采样候选限制。保留其持久化键与算法位置，只纠正内部变量/注释；不得为了改名破坏 checkpoint recipe 或把截断前移改变点云。

### P6.2 全程不改的数值与数据协议

- 公共 canonical 保持 13 字段、dtype/shape/单位/顺序、缺测 NaN、原行与 episode 边界。Policy 继续按实际 N 步观察/H 步监督和 dispatch 资格取窗口；train-only 唯一源行统计、val 复用和 checkpoint normalizer 不变。
- 当前 mount/table 重建是显式 processing recipe，沿用已有 export_report、源码、revision 和新路径发布。不默认加入历史标定匹配/ABI，也不把当前 recipe 重建的结果说成历史物理测量。相机继续用对应 Raw 参数，触觉不再扣偏置。
- 保留标准化数值接口，不能让同 shape 偷换单位、frame、关节或手指顺序、rot6d 约定。模型物理 action 与 aux EE 尾部不同；不要为了统一维度删掉合法辅助输出或在部署中发出辅助维度。
- 不默认改视频编码、做位级一致认证、插值、补帧、坏行压紧重拼窗口。已有不足只做准确说明，不新增另一套实时对齐系统。
- 保留 revision/manifest digest 的身份职责和 trial 分组检查；它们不证明标签真实。无 manifest 的原路径不顺手禁用，论文实验需要 trial holdout 时明确选择真实清单。不得从目录、暂停、HOME 次数猜 trial。

**验收：** 方法输出与 reset 行为保持；纯几何不导入 NLopt；音频顺序、抢占、失败和回收用 fake Popen；旧 pointcloud 序列化 recipe 可原样加载并得到相同配置/输出；新 IK 报告名准确。数据变化不是此阶段的隐藏产物。

## 9. P7：论文开源入口、测试与交付

- 主要安装方式明确为 source checkout + editable，一套已核实环境即可。不顺手构建 wheel/PyPI/多平台支持。基础依赖、可选算法/设备依赖、dev 测试依赖分清；不把手工私有 SDK 塞进不可安装的 pip 依赖，不编造版本范围或自动升级本地环境。
- README 先项目用途和完整工作流，再链接详细执行限制。新增或整理一页简短 reproduction：两仓源码、配置、数据/模型来源、命令与预期产物、离线与真机界线；只能填核实信息。论文/数据尚未发布则直说，不生成虚假 DOI、权重地址或成绩。
- 一个小离线样本复用现有 EpisodeReader/writer、viewer/export/window。优先扩展现有 fixture/入口；必要新增一个薄样本入口，不建第二套 reader/录制器/测试框架。合成样本必须明确 synthetic，使用临时/显式新目录、真实格式、足够长度的有限目标/时间/dispatch；用合法几何生成点云而非绕过真实处理器，不能作为性能或物理成功证据。无数据授权不上传真实采集文件或模型。
- 无 GUI 的测试可检查读取/导出/取窗；不要把 viewer 的显示成功作为全部基础测试前提。缺 Pinocchio、PyAV 或 Policy 环境时分开说明可运行层级，不能给所有功能假 PASS。
- 测试按行为而非 review 日期逐步整理，复用少量 fake robot/model/clock/rows；只搬本次触及的测试，避免大规模纯重排。关键故障测试保留，几何原生测试与调度 fake 测试分开，依赖缺失不算通过。
- 可新增精简离线 CI：只执行声明支持环境中无硬件副作用的检查。涉及需审批的工作流文件权限受限时单独报告；不影响源码整改。许可证/第三方资产权利和论文引用需作者确认，不擅选 LICENSE 或伪造 CITATION 元数据。已确认归属可以准确记录。
- 保留用户指定根目录任务书；AGENTS 维持长期规则，README 不承载全部内部类清单。历史任务书不并行驱动实现。本任务书末尾仅更新简短执行状态，不再创建长篇永久整改报告。

## 10. 全部 50 项去重议题的最终处置

下面是实施范围，不是要求每项都改代码。FIX 表示明确整改；REFACTOR 表示行为保持的减法；RETAIN 保留；CONDITIONAL 本次不自动实施行为变化；RELEASE 依实际发布条件。基线事实变化时记录依据，不凭标签机械修改。

| ID | 议题与当前事实 | 最终处置 / 落点 |
| --- | --- | --- |
| F01 | mount 注入及错误 HOME 测试已被最新几何分离修正 | ALREADY_FIXED；保留原资产参考测试，不恢复临时 URDF |
| F02 | 全局验证含未使用 VR/teleop；pointcloud 构造也有数值检查 | FIX P1；按用途，独立入口不失守 |
| F03 | table resolve 后点云再次读文件 | FIX P1；真实消费者四组合，一次快照 |
| F04 | collect CLI 顶层传递导入 NLopt | FIX P1；help/配置与真实 workflow 解耦 |
| F05 | 固定容量相机逐帧传重复尺寸 | REFACTOR P4；初始化布局支持 attach/spawn |
| F06 | 三类 ring 无条件分配 | REFACTOR P4；显式按需，保留间接消费者 |
| F07 | 同一 trace 存 Raw 和 attempt | REFACTOR P3；先闭合 attempt 归属再去副本 |
| F08 | NPZ 失败报告收尾失败 | RETAIN P3；当前是必需产物，不吞错误 |
| F09 | runner 构造资源、调度、归档混合 | REFACTOR P3；已有模块重分职责，无新管理层 |
| F10 | retargeting 几何和后端 adapter 交错 | REFACTOR P1/P6；按方法组织，保持算法 |
| F11 | 音频重启与多播放器回退过重，queue 有实际调用 | REFACTOR P6；删恢复/回退，保留顺序/抢占/回收 |
| F12 | policy/safety 名称与责任错位 | REFACTOR P1；内部整理，不强制重命名全部公开 YAML |
| F13 | 跨层布尔元组含义不清 | REFACTOR P3；小型命名结果，不改布尔语义 |
| F14 | 测试按整改历史组织、需拼内部状态 | REFACTOR P3/P7；依赖显式，按触及行为整理 |
| F15 | 公共导出总是 13 字段且含 FK | RETAIN P6；不新增模态组合/lazy 缓存 |
| F16 | 历史 Raw 按当前 mount/table recipe 重建 | RETAIN/CONDITIONAL P6；明确配方，特定历史还原另选配置 |
| F17 | 原 RGB 与 MP4 解码 RGB 数值可不同 | CONDITIONAL；代表性数据分析，不默认无损/逐帧认证 |
| F18 | recorded_rows 不证明等物理时间间隔 | CONDITIONAL；真实时间分析，不默认重采样/筛窗 |
| F19 | 录制 cloud 需要对应 RGB-D，可能遭遇 ring 覆盖 | RETAIN/CONDITIONAL P4；先测实际延迟，不能换错帧 |
| F20 | 点云多阶段清理包含任务结构假设 | CONDITIONAL；无消融证据不删阶段 |
| F21 | candidate limit 位于重计算之后 | REFACTOR P6；澄清职责，保留序列化键与位置 |
| F22 | 手眼激励物化 O(N^4) 规模的全部轴对列表 | REFACTOR P5；存在性早停，最坏时间不冒称降阶 |
| F23 | 手眼 residual 只是样本内自洽 | RETAIN P5；独立实物验证另行，不删有效筛查 |
| F24 | connect 自动 tare，残差无法证明无接触 | FIX P5；显式准备，取消安全，无额外自动运动 |
| F25 | IK 多启发式与搜索 | CONDITIONAL；保持默认方法，后续独立消融 |
| F26 | cmd_tracking_error 是 IK FK residual | REFACTOR P6；准确命名，不新增假物理指标 |
| F27 | 在线端点准入不等于连续环境避碰 | RETAIN；保持功能与限制，无额外认证系统 |
| F28 | invalidate/WAIT 不取消已有固件目标 | RETAIN；本次不改停止与等待处置 |
| F29 | sync 与 async/RTC 的 CRC 未确认处置不同 | CONDITIONAL；维持当前协议，明确新对照后独立变更 |
| F30 | 反馈时标是 host read completion | RETAIN；不伪造源时间，不默认加 Raw 时标字段 |
| F31 | supervisor 是协作式存活检查 | RETAIN；不假称独立 watchdog，不追加守护平台 |
| F32 | InferenceWorker close 不保证有界退出 | RETAIN；不强杀线程/伪报清理完成 |
| F33 | RTC beta=0 正 delay 在 warmup 被拒绝 | FIX P2；兼顾 beta>0、delay=0 原路径 |
| F34 | 冻结前缀与发送前复核 | RETAIN；不可重新求解后沿用旧条件 |
| F35 | 三预算每次 CLI 显式输入负担 | REFACTOR P1；YAML 保存，缺值仍拒绝 |
| F36 | 五次 warmup 不是实时上界 | RETAIN；保留明显不可行配置筛查 |
| F37 | HOME 候选、soft escape 和 proxy | CONDITIONAL；不直接用插值代替或删除检查 |
| F38 | NumPy seqlock 无可移植 acquire/release 保证 | RETAIN P4；不裸删同步，也不声称测试证明跨平台 |
| F39 | shape 不是数值语义 contract | RETAIN P6；守单位/frame/order，无大 schema 引擎 |
| F40 | revision/digest 不证明数据真实 | RETAIN P6；身份与划分检查即可，不加全量认证 |
| F41 | trial manifest 可选且标签依赖外部事实 | CONDITIONAL P6/P7；不猜 trial，不强改训练默认 |
| F42 | Raw 不可变/NaN/背压/staging/失败前缀 | RETAIN 全程；不得以简化删除 |
| F43 | 四状态/run_id/每 SDK 入口授权和 dispatch | RETAIN 全程；保留不同时刻真实边界 |
| F44 | retargeting 每源帧缓存成功与失败 | RETAIN P6；不重复推进有状态求解器 |
| F45 | assets 位于 checkout，不代表完整 wheel 支持 | RETAIN/RELEASE P7；明确支持源码安装，不猜路径 |
| F46 | 包内标定与源码派生输出混合状态和代码 | REFACTOR P1/P7；明确路径选择，不移动历史资产 |
| F47 | 文档主线被历史任务/实现细节遮盖 | REFACTOR P7；工作流优先，历史任务不叠加 |
| F48 | 缺少完整的小型离线进入路径 | REFACTOR P7；一个样本复用真实入口，不另造框架 |
| F49 | 许可证/引用/dev/精简 CI 发布准备 | RELEASE P7；已知事实才写，权利/权限待确认项单列 |
| F50 | 用新通用系统替代可删除复杂度的风险 | RETAIN 全程约束；不整体迁移参考框架 |

### 条件议题的进入标准

本次可以梳理已有代码和数据说明，但不自动启动实际训练、硬件、批量导出或算法消融。点云阶段删除需要关键物体/接触区域保留、空云率、耗时/内存证据；IK 需要有效解/物理目标连续性/延迟证据；HOME 需要实际起始姿态与路径失败案例。只做一个明确简单基线与现有方案比较，不为比较造框架。若后来批准删除阶段，应连同参数和无消费者报告删除，另记 recipe/baseline，旧模型不静默换预处理。

F29 只有在作者明确采用统一通信准入的新对照协议时，才在 policy evaluation 层统一两设备 ACCEPTED，并独立验收/标注新基线；不把 CRC 未确认改写为成功，也不默认改变 teleop。F17/F18/F41 分别需要实际图像、时间和 trial 清单，缺这些只限制实验结论，不阻挡 P1-P7。

## 11. 执行顺序、验证与最终交付

顺序：核实 F01 -> P1 -> P2 -> P3 -> P4 -> P5 -> P6 -> P7。每阶段内部可拆更小自洽 diff；每次完成实现、真实调用者、旧路径删除、必要测试和简短文档。P1 的提前构造需与 P3 owner 搬迁协调；P4 资源和 P5 准备顺序改变后补跑此前集成检查。不要把阶段顺序当作以后不回看的理由。

行为保持与行为变化分开：导入/职责/静态布局/诊断字段移动应保持数值和控制协议；RTC 修复是允许配置闭合；显式 tare 是操作流程变化；新 trace 位置是新产物组织变化。不能要求这些变化与旧文件字节完全相同，也不能趁机改算法或历史 baseline。

先审查测试 fixture 无设备副作用，再使用现有解释器与工具。常规检查：

```bash
python -m compileall -q dexmani_real examples
ruff check dexmani_real examples tests
ruff format --check dexmani_real examples tests
git diff --check
```

按实际变更运行现有定向测试，包括 `test_policy_runner.py`、`test_policy_recording.py`、`test_policy_eef.py`、`test_collision_geometry.py` 及现有 IPC/config/calibration/teleop/replay 测试；先确认当前文件名与依赖。Policy 只运行此次 RTC 桥接与已有相关定向测试。完整 tests 可在确认离线且依赖齐全后执行；不为缺依赖自动装包或跳过后称全通过，不顺手格式化无关文件。既有无关失败分别记录，新增回归必须修复或明确未完成。

行为保持检查用同样 fake 观测与时间验证：目标数值、slot/index、query 来源、冻结 prefix、dispatch、stop 调用顺序、首异常、记录次数。Raw 比较字段/数值/NaN/行关系，不比较随机 UUID、临时时间或视频字节；原生几何、真实模型和真机证据分开。资源检查部分初始化、pending Future、writer 失败、late accepted、close 与 unlink 顺序，不把小单测夸大为硬实时证明。

缺权限/依赖/现场信息仅标相关项，不把整体任务退回纯设计，也不为了完成率增加替代实现。资源不足时保留可运行的自洽子集，清楚列未完成和下一入口；不留下默认双实现或静默 fallback。

最终中文交付：按 P1-P7/必要 F 编号列完成、已修复、保留、条件未实施和阻塞；给实际命令、环境、PASS/FAIL/未运行及原因；列删除的旧路径和不可避免的新增边界；说明是否改了配置/操作/产物协议及迁移方式（无历史资产自动迁移）；分别说明 Policy 联动和真机未验证。不要抄历史 PASS 数、承诺提速比例、用测试数量或删行数量证明质量。最终 diff 只含本次范围，保持用户改动，不自行 commit/push。

## 12. 参考与源码定位

参考只支持结构与取舍，不认证本项目的 XHand 错误码、几何、时限或安全阈值。

| 官方来源 | 本任务采用 | 不照搬 |
| --- | --- | --- |
| [LeFranX](https://github.com/wengmister/LeFranX) | 固定硬件范围，server/teleoperator/方法适配明确，直接脚本 | 未用接口、复制整套框架 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon) | 真实进程/共享内存边界，小 observation/action wrapper | 多机器人 registry/通用配置平台 |
| [LeRobot Robot API](https://huggingface.co/docs/lerobot/main/api/robots) | 小的读取/发送接口，实际发送目标语义 | 全链路统一大 schema、整体格式迁移 |
| [DexUMI](https://github.com/real-stanford/DexUMI) | 复杂度为研究需要服务，一个小样本贯通流程 | 根据其他项目可选模态推翻本项目 full-modal |

关键事实可由上述基线中的真实路径复核：Real `config/experiment.py`、`config/pointcloud.py`、`sensor/pointcloud_worker.py`、`ipc/camera_ring.py`、`deployment/runner.py`、`recording/recorder.py`、`robot/drivers/xhand.py`、`calibration/camera/solver.py`、`teleop/audio_feedback.py`、`tests/test_collision_geometry.py`（均位于包内，tests 除外）；Policy `dexmani_policy/deployment/runtime.py` 及 `agents/core/base.py`。这些路径用于定位，不要求恢复到旧 SHA。

## 13. 本地执行状态（由执行者简短更新）

设计已定稿，实施尚未执行。执行后仅追加各工作包状态、实际验证与未完成原因，不将条件议题改标为已验证，不再生成第二份长期任务书。
