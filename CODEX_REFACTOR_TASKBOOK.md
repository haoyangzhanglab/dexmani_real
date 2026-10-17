# DexMani Real：两轮审查核验后的 Codex CLI 任务书

## 0. 任务、授权与使用方法

**目标：先修复已确认的控制/数据错误，再减少无关准入、重复计算和状态维护负担，使研究路径能直接读出输入、算法、输出及真实执行结果。**

本任务将“提升架构的间接性”按对话意图解释为：提升架构简洁性，减少无必要的间接层。不是增加抽象层数。

本文件是用户在 2026-10-17 明确要求写入仓库根目录的一次性实施任务书。该明确要求优先于 AGENTS.md 中“临时 task/migration notes 不作为长期根目录文档”的一般放置规则；无需为本文件的位置再次询问。不要修改 AGENTS.md 来制造例外，不把本任务转成永久架构规则。诊断脚本、对比输出、执行进度仍放 Git 忽略的 reports/。

**交付本文件时只完成了审查、核验和方案设计，尚未修复生产代码。** 本地 Codex 收到配套 prompt 后，应实际实施下文“默认实施”部分，按阶段完成局部回归，不以再次提交一份计划代替修改。

### 0.1 证据基线

| 对象 | 固定版本 | 核验范围 |
|---|---|---|
| dexmani_real | [3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7][real-base]，main，提交时间 2026-10-17 06:24:30 UTC | 两轮主仓审查；本轮重新追踪关键调用链、独立交叉核验 |
| dexmani_policy | [22884db02778db9bcd16ebdc0b9b6a529ef3afc5][policy-base] | 只核对部署解析/加载接口，不是训练仓全量审查 |
| LeFranX | [a39906e6629f39490950fe8bd20f4f992ed74fd7][lefranx-base] | 设备 owner、发送语义、显式组件组合 |
| LeRobot | [146d8474932c8b566717eb2738ba459337ab71c4][lerobot-base] | inference/rollout 契约、观测所有权、数据时间 |
| ManiUniCon | [85c6f2e32ecf9f2bed62d202b058c39623444686][mani-base] | 模型/观测/动作职责、进程和时间槽 |

以下行号属于固定审查基线，仅用于定位；**当前源码、实际调用者和有效配置优先**。任务书自己的提交会自然晚于代码基线，不要求 HEAD 等于审查 SHA。先检查本地分支、diff 和相关符号；若已修复，验证后标为 already-fixed；若行为已变，更新该项判定。禁止为复现旧问题 reset 用户工作树。

核验证据包括源码/官方 SDK 原文、实际函数配替身的控制流反例、有限数值对照。没有连接机器人/相机/VR、加载实际模型或测得真机性能。部分云端检查使用 NumPy 2.3.5，项目约束为 NumPy <2；HDF5/codec/Pinocchio 等存在替身或缺失。**这些反例不是本地完整环境通过证明，也不能证明仓库再无任何问题。** 本地验证要求写在各工单中，不依赖未提交的审查 reports/ 或聊天历史。

### 0.2 实施边界

- 默认范围是当前 Real 仓库及其中的可选 cordvip_geometry 包。无需修改 Policy 仓库才能完成核心任务；不得假造不存在的 Policy API。
- 保留本地无关修改；不强制 reset、清理用户文件、覆盖既有输出或发布 Raw。例行、可逆的代码修改和离线检查不需逐项请示。
- 不启动真实设备、运动、HOME、teleop、replay、rollout、现场标定或真实传感器诊断。读源码/元数据与使用明确替身不等于运行这些入口。
- 不安装或升级实验依赖来让检查变绿；使用现有环境，缺失依赖只阻塞依赖它的事项。不得运行或引入 pytest、Ruff。
- 不自动推送、合并或部署代码。完成后交付本地 diff 和验证结果；是否形成提交遵循用户另外给出的约定。
- 不以行数、文件数或 validate 调用数为优化目标。每个新增 helper/类型必须承接一个实际共享边界或消除现有独立状态；不建 registry、通用 RPC、验证框架、资源框架、迁移框架。

## 1. 核验裁决与目标架构

### 1.1 经过 fact-check 的重要修正

| 原结论容易被误读之处 | 核验后的裁决 |
|---|---|
| HOME 取消后一定及时停下 | 子进程已发现撤权时仍可能不本地 stop，依赖父端从 XHand service 返回；是可达控制错误。不能保证中断一个已阻塞的原生 SDK 调用。见 R1 |
| stopped event 为真就代表物理静止 | 该 event 是处理完成通知；应同时看 close_error/fault/接受证据。真实问题是 SDK helper 吞返回码后没有留下失败证据。见 R1 |
| 标定入口没有加载场景 | 相机标定已经传 static boxes/table；缺的是 IK 后最终环境/FK 准入。键盘入口另有 static boxes 未传问题。见 R2 |
| 键盘默认 workspace 已证实越界 | 默认命令内缩 5 mm 大于 IK 位置容差 2 mm；缺最终 gate 成立，默认越界未复现。见 R2 |
| RRT 忽略场景意味着最终放行碰撞 | 搜索与最终模型不一致，但最终路径检查仍 fail-closed；影响可用性和候选质量，不是已证实的最终放行绕过。见 H1 |
| 导出 metadata 提前失败完全等价 | 当前“不兼容且数值坏”会拒段继续，“不兼容且数值好”会整任务失败。提前失败有意改变双故障错误优先级。见 D2 |
| 版本查询前移即可修好 replay | 版本字符串当前是追溯信息；不可把缺 distribution metadata 变成新的运动硬门槛。见 R4 |
| 手眼 gate 向量化即可从 O(N⁴) 变 O(N²) | 有界向量化减少 Python 开销，最坏算术量仍可能 O(N⁴)；首轴/三轴 rank 替换不等价。见 P2 |
| radius graph 前移 8192 点限额是等价优化 | 会改变密度、连通分量和最终点云表示；不实施。主点云 depth rays 已有缓存。见 X1 |
| depth-Z 与手 mount 差异都是确定待修 bug | 前者是条件性的 SDK 近似，后者是已披露 nominal/actual 模型差异；没有现场误差依据，不自动改表示或标定。见 X2、X3 |
| 两种 selection 已承诺互通 | 未找到 track selection 可直接 generate 的承诺；共用字段重复成立，RGBA 的不同消费需求应保留。见 P5 |

### 1.2 采用的最小责任划分

| 当前问题 | 目标责任 | 迁移方式 |
|---|---|---|
| driver/worker/parent 都组织 HOME 并互相等待停止 | child 独占 SDK、推进 HOME、处理本地取消；parent 负责路径与手反馈服务；driver 规范化有限厂商原语 | 先局部修 R1，再整理 S1/S2，不先重写整套执行器 |
| 最终 endpoint 准入私藏于 ActionRealizer | 一个具体的公共 target admission，供各真实运动入口使用 | 复用现有限位、碰撞和 FK；保留入口自己的投影/IK/手控制语义 |
| Online IK 经整台 HOME planner 转发 | 在线求解、HOME 候选生成、最终准入各自清楚 | 先保留原数值后端，再移动构造和调用；不换 IK 算法 |
| runner 的计算 helper 随时结束整段 | 时序计算返回明确结果；episode owner 决定停止、录制与结果发布 | 从现有 Query/Plan 提取最小调度责任，不复制三套 Runner |
| UI 请求、会话、授权与清理混成布尔组合 | UI owner 持有待消费请求；runtime 集中授权/终止；会话明确阶段 | 保留真实独立事实，减少多点赋值 |
| 每段构造相同 FK；reader 隐式捆绑所有诊断 | 一次导出一个 FK owner；按真实消费者读取和归因 | 复用现有 export/reader，不建立全局缓存或 schema 系统 |
| optional 感知两套 writer、能力门槛与运行需求不符 | 一个共享 IO 入口；按本次 pose/GUI/3D 需求检查依赖 | 保持可选包和 SAM 子解释器边界 |
| Real 重解 Policy 输入又用输出标签限制输入 | Policy 现有 resolver 解析输入；Real 只处理物理配置和输出路径 | 本次仅改 Real 即可，跨仓 API 扩展另列条件项 |

论文代码风格在本任务中的含义：数学函数显式接收数组/几何、返回数值和失败原因；注释解释 frame、unit、时间和判据；会话副作用在 owner 中可见；关键消息和多字段行用具名字段；删除无调用者备用路径、日志专用生命周期与单实现空转 wrapper。无需把所有脚本都变成类，也不把校验压缩成难读的一行。

### 1.3 任何阶段都要保持的契约

1. **物理与时序：** arm 7、hand 12 的既有关节顺序；rad/m；xarm_base、optical 等 frame；WXYZ 和现有 rot6d 排列；command、SDK accepted、measured arrival 是不同事实。Action、观测和 timing 的记录含义不变。
2. **并发：** xArm SDK 单进程 owner、模型单 owner、固定目标槽、不可变命令/观测与必要 copy/readonly、有界队列/关闭、late reply 与 unresolved operation 保留。
3. **动态准入：** epoch、最后 SDK 前 deadline、真实反馈 freshness、多源读取结束后的再检查、轨迹/braking extrema、双设备发送前共同预验、即刻 STOP/ESTOP 撤权保留。静态配置已验证不能替代这些检查。
4. **Raw：** 已发布数据 immutable；停止/失败/晚帧保留已采前缀、真实时钟、首次终止原因、未确认发送和 staging。只有用户显式重录/丢弃可以删除当前 capture。
5. **训练表示：** 公共 canonical 15 字段全部完整且浮点有限；坏 episode 整段拒绝。不裁坏帧、补零、插值、重采样、再次 tactile hold，或把缺失/模型推断伪装成实测。
6. **当前实验：** 当前桌面、mount、VR/运动学不是历史 ABI；不添加与历史标定数值一致才运行的 gate。相机仍使用与数据对应的内外参和 depth scale。
7. **异步策略：** H、N、P、A、d、查询来源、动作槽与 RTC prefix 含义不变；不整体平移迟到 chunk 来伪装按时，也不以重复样本填同步等待。

## 2. 执行顺序与完成范围

| 阶段 | 工单 | 完成目标 |
|---|---|---|
| 0 | 本地核对 | 读 AGENTS；确认当前代码、diff、依赖版本；为高影响事项建立最小反例 |
| 1 | R1–R5 | 修取消/停止结果、公共准入、旧输入、prefix 保留和两个单点错误 |
| 2 | G1–G7、D1–D4 | 删除无关门槛，统一解析，修导出错误归因与 owner |
| 3 | P1–P5 | 删除已证实重复工作；数值项有对应对照后落地 |
| 4 | S1–S5 | 收拢协议、HOME、状态、planner 与 runner 职责；避免与数值改变混在一起 |
| 5 | H1–H2；P6 按条件 | 现有本地完整依赖可验证才改；否则明确保留/阻塞，不影响前四阶段 |
| 6 | 文档和收尾 | 更新受影响入口、删死代码/旧别名、检查 diff，按工单报告完成状态 |

R/G/D/P1–P5/S 为默认实施范围；若当前已修或具体数值验证缺依赖，可逐项说明并继续其余工作。H 为依赖条件明确的后续工单。X 为本次明确不默认实施的事项。不要用“全部需要真机”停止已能进行的源码与离线工作，也不要将 blocked 写成 done。

## 3. R：优先修复逻辑错误

### R1 — HOME 子进程本地取消、健康期限与可判别停止结果

**事实与定位。** [arm_worker.py][s-worker] 的 operation/check_abort（113–153）对 HomeAborted 不 fault、不发本地 stop；循环的撤权检查（305–306）只覆盖 active streaming。HOME 已通过 [xarm7.py][s-xarm]（319–335）提交 Mode-0 wait=False 路点。父端 [arm_executor.py][s-executor]（224–247）须从 service 返回才能补 stop，而 [robot.py][s-robot]（491–493）服务可能阻塞于手反馈。真实函数配 fake driver 已复现“撤权后 child 回取消、stop 调用 0”。

同一 driver 的 emergency_stop（361–364）依赖 SDK helper。[官方固定版本实现][sdk-stop] 不检查 set_state(4) 返回码，可正常返回 None；“请求持续失败但 handler 留下 clean stop”的替身反例成立。实验机安装版本未知，实施前读取本地包实现/元数据，不假定等于该版本。

**修改。**

1. child 在 HOME/prepare 的取消或失败出口本地撤销 prepared/active/reference 资格，发起并处理停止，然后发布操作结果；不能以 parent 收到 reply 为停止前提。取消和硬件失败分别归因，保留首个原因及后续 stop failure。
2. 使用可判别 SDK 请求结果的停止原语或独立、有界的真实结果核查。保留停止请求、处理完成、SDK 接受、物理静止、资源关闭的区别；io.stopped 仍可用作完成通知。不能把 None、缓存 state 或完成 event 当作独立成功证据。
3. 对 HOME/prepare 建立该阶段适用的 service_until、hand_until。**同时修生产方：** 当前 HOME 的 _read_hand 不像 read_state 那样更新时间；应以实际手反馈时间加已有 age budget 发布，不能每次读缓存就用 now 刷新。合法 no-hand 模式不制造缺测。
4. 在阻塞边界间及新的模式/路点操作前检查撤权和上述健康期限。不要把 streaming 的 mode=1、command sequence/deadline 或 camera/VR upstream gate 原样套到 HOME；startup 初始化也没有已建立的同一健康时钟。
5. stop rejected/exception/unresolved 必须锁存，禁止自动恢复运动。不能承诺打断 child 自身卡住的厂商调用；父端已有有界收尾和未解决证据保留。

**验收。** fake driver 接收一个路点后分别注入 epoch 撤销、cancel、ESTOP、service/hand deadline 失效、parent 停顿；child 无需 parent 再推进即可请求 stop，后续路点和恢复 Mode1 调用为零。覆盖 prepare dwell、HOME 路点等待和恢复模式各阶段。另测 stop accepted/rejected/exception/无可判别结果/调用未完成；失败不得清洁结束，首次原因不丢。成功 HOME、合法 no-hand 和正常初始化仍可完成。

### R2 — 各入口复用相同物理目标准入

**事实与定位。** [action.py][s-action] 的 _joint_rejection/_final_target_rejection（72–108）有最终准入；[replayer.py][s-replayer]（227–253）project 后直接 send，手 warm-up（112–122）也发送 arm hold + hand。[keyboard_session.py][s-keyboard]（55–61、153–180）未传 static boxes 且 IK 后直接发；[camera/session.py][s-camera-session]（115–123）已经加载场景，但 _motion_tick（744–783）同样缺最终 gate。Robot._send 负责权限、期限和硬限位，不承担几何。

**修改。**

1. 在现有规划/动作边界公开一个小的 endpoint admission，接收候选 q、实际 current/previous、显式 collision_hand_qpos 和本次 scene。双设备命令使用本次投影后的 hand target；arm-only jog 使用本次实测手形；无手/固定手场景保留当前明确的 home 假设（相机标定当前采用该假设），不能把所有入口统一成 row.hand。复用 operational/model bounds、当前距离/跳变、自碰撞、最终 FK workspace、static boxes；保留已有错误分类/优先级。
2. ActionRealizer、replay 每次实际发送（含 hand warm-up）、keyboard、camera calibration 都接入。Keyboard 正确传当前 scene。arm-only 标定不应被强套 Policy hand_enabled/手目标投影规则。
3. 把在线允许 table contact、HOME 需要桌面路径净空的差别显式保留；endpoint 通过不等于 swept path 通过。
4. 每个 q 的环境碰撞结果一次返回必要 detail，避免失败时为取详情再查一次。collision 为真时，即使 contact list 为空或详情提取失败也必须拒绝；不能以详情非空代替碰撞事实。不同物理条件不可删。几何放 parent，不迁进高频 SDK child；endpoint 不证明 arm/hand 从 measured 到 target 的 swept clearance。
5. 阻塞求解后仍在 SDK 边界检查动态权限/时效。目标被拒不能更新 command reference/EMA，不能计作成功发出或跳过后继续 replay。失败前缀继续保存。

**验收。** 同一 current/q/hand/scene 的静态盒、自碰撞、model limits、FK 出界、跳变反例在对应入口规则下得到一致拒绝，send 计数为零；warm-up 也覆盖。正常目标仍保留原 projection/wrapping、手操作和真实提交时间。workspace 反例可使用非默认内缩/容差或明确 fake IK；不把它描述为默认 5 mm 内缩已失败。

### R3 — 阻塞 prepare/IK 后不提交过时 jog 输入

**事实与定位。** [keyboard_session.py][s-keyboard]（104–105、117–180）准备前计算增量，准备后沿用；camera session（688–734）同类路径。KeyboardInput.on_release 只更新 held keys。fake prepare 清空键后，真实键盘分支仍发送一次。

**修改。** compute_cartesian_jog_delta 改收一次 pressed_keys 快照，避免 12 次独立加锁读取；prepare 后重新采输入和反馈，IK/准入结束再确认本 proposal 仍有效。必要时使用键盘输入 generation 与已有即时撤权机制，区分 release→repress，不能让旧 proposal 复活。无有效位移时不 begin_motion、不提交缓存目标；不让 SDK 调用持有输入锁。

**验收。** prepare/IK 中松键、反向、release→repress、Q/ESC、相反键抵消均不发旧 proposal；持续按住每 tick 只累加一次，reference 只在实际提交成功后推进。同步撤权与 owner 的 UI 收尾均保留。

### R4 — replay 的物理前缀不被可选 metadata/派生失败截断

**事实与定位。** [replayer.py][s-replayer]（368–374）采集后再 execution_metadata/JSON，异常使 ReplayOutcome 未返回；[replay/session.py][s-replay-session]（144–179）因此仍拿不到 replay_data。已复现 1 行已提交/已采集后 PackageNotFoundError。版本字符串在现有 schema 中仅追溯，不是执行能力准入。

**修改。**

1. execution_metadata 中 SDK、Ruckig 版本 lookup 分别处理不可得情况；明确记录 unknown/unavailable 与对应 *_version_error 文本，不猜版本、不把有效源码安装变成运动错误。
2. 版本与静态控制 metadata 在 prepare/首个动作前解析一次。字段保持 HDF5 attrs 和 JSON 都可写的 scalar/string；不用嵌套 dict 或 None。
3. 已采 physical arrays 先形成可交付 outcome/记录；末端 snapshot、执行审计、诊断 JSON 或派生失败时，仍交付原行和真实错误，按必要的执行证据决定 FAULT/technical-invalid，不能把前缀退回 None。
4. 保留 [evaluation.py][s-evaluation]（171–193）已正确采用的“先保存 physical NPZ，再计算 EEF metrics”。执行/stop 未解决不能伪造成功，也不能把损坏 staging 发布成合格 Raw。

**验收。** 有 SDK 能力但无 distribution metadata 不被误拒，输出明确 unknown/error；fake 1 行提交后注入 snapshot/诊断/派生故障仍持有并尝试保存同一物理前缀，保留 dispatch/时钟；缺 stop/完成证据的结果仍不通过原执行准入。用真实 HDF5/JSON（本地已有时）验证 metadata 可序列化。

### R5 — 两个可独立完成的单点错误

| 项目 | 事实 | 最小修改与验收 |
|---|---|---|
| 点云标定示例 | [pointcloud_process_example.py][s-pc-example] _load_extrinsics（361–371）使用未定义 cam_name | 用已有 camera_info 的 name/serial；实际函数输入合成 camera_info 和外参能返回，无设备/GUI |
| 子解释器路径 | [cli/build_object.py][s-build]（333–339）把 sam_python 与资源路径一起 resolve，symlink venv 可变成 base Python | 解释器 expanduser().absolute() 保留调用路径；资源仍可 resolve。用无 pip venv 或现有 symlink fixture 截获子命令并只读 sys.prefix，确认不解引用；不运行模型 |

## 4. G：按实际消费者收窄校验与配置入口

这些事项扩展原先被无关条件拒绝的有效输入；**不放宽实际生效算法、硬件边界或数值表示的条件**。保留现有 dataclass/明确函数即可。

### G1 — 平面读取不校验未消费的碰撞箱

定位：[config/experiment.py][s-experiment] resolve_table_plane（129–144）、[config/environment.py][s-environment]（89–110）。离线导出/viewer 只要 plane，却因 thickness、size、clearance、contact links 被拒。

修改：读取 inline/JSON 后仅验证四系数 shape、finite、非退化向上法向；完整碰撞参数留在真正构建 table 的 CollisionModel 边界。共用小 helper 时保留明确阈值差别：原 collision 法向阈值 1e-9，ProcessingConfig 为 0，不能顺手统一造成隐含数值改变。

验收：enabled=False、plane=(0,0,1,-0.022)、thickness=0 的纯 plane 读取通过；真正启用碰撞箱仍拒绝零厚度。NaN、零/向下法向、缺文件依原职责拒绝，有效 plane 输出不变；不加历史标定一致性 gate。

### G2 — 关闭 table removal 不要求未使用参数有效/齐全

定位：[config/pointcloud.py][s-pc-config]（45–79）、[deployment/config.py][s-deploy-config]（59–68）、[sensor/pointcloud.py][s-pointcloud]（474–503）。

修改：remove_table=False 时，不再要求 table_core_height_m、table_object_seed_height_m、table_object_seed_min_pixels 满足活跃分支的范围/相互关系；基本类型、拼写及有限性仍明确。saved recipe 可以缺这三个 inactive 字段，解析成完整 resolved recipe 后保存。**其余活跃数值参数必须来自 artifact**，不能全部取消 fields 检查后偷偷使用新版默认值。remove_table 是唯一开关，不采用 NaN/0 哨兵。

验收：同一 RGB-D/相机/生效 recipe，关闭分支三个值设 0 或从 saved recipe 删除，点云逐元素一致；开启分支缺字段或违反原关系仍拒绝；未知 key 仍报错。当前 exporter 生成的完整 recipe 继续通过且数值不变。

### G3 — Raw 手云不要求无关硬件配置

定位：[cordvip integrations/dexmani.py][s-integration]（154–158）为 mount/FK 调 config.hand.validate，牵连 baudrate、设备 ID、反馈期限等。

修改：只校验有限 (3,) mount、WXYZ 单位 quaternion 和实际消费的 Raw 关节/行/来源时钟/相机几何。仍正常读取当前 experiment config，不用 identity 或默认 mount 掩盖无效输入。

验收：同一有效 mount，仅改变不参与离线算法的 baudrate/feedback 参数不再阻断；无效 mount、错 source/row、缺实测 q 仍在对应边界拒绝，FK 与 point/link 顺序不变。

### G4 — RGB-D 诊断按显示模式加载几何，空 live 云不保留旧图

定位：[realsense_record_example.py][s-rs-example]（40、340–360、463–465、499–505）。默认点云 OFF 仍预先要求 plane/外参；live 空云又不更新旧图层。

修改：纯 RGB/Depth 不访问点云标定；第一次启用 PCD 时解析该模式确需的输入，缺失则明确拒绝该显示模式，保持基础 RGB-D 可用。有效云变为空时清空或明确标为过期；用户显式 freeze 的快照仍保留。禁止用 identity 外参让缺标定点云看起来正常。

验收：fake camera/UI 证明 PCD OFF 不调用 plane/extrinsics loader；启用缺标定 PCD 有明确信息；valid→empty 的 live 图层不残留，freeze 语义不变。不要启动真实示例作 smoke。

### G5 — 父感知依赖与 SAM 子环境按真实能力检查

定位：[cli/check_tracking.py][s-check]（55–65）、[cli/scene_tracking.py][s-scene-cli]（281–289、365–370）和 [live_viewer.py][s-live-viewer]（37–49）。

修改：父进程 Torch/Torchvision 仅在 need_pose 时检查；Open3D 由 need_pose 或 need_3d_viewer 决定，显示会话与窗口创建由 need_gui 决定。当前 OpenCV 仍是图像处理/IO 依赖：headless rectification、视频保存及 run_scene 的导入会使用它；只有逐条证明某路径不消费它时才进一步收窄。SAM 子解释器的 Torch/模型/权重/API 检查独立保留。各 CLI 使用同一明确的需求计算，不建 feature registry。pose renderer 本身需要 Open3D，不能用 --no-3d 一概免检。

验收矩阵：assetless+headless、assetless+OpenCV GUI+no-3d、assetless+3D、pose+no-3d，分别模拟缺少一个需要/不需要的模块；再覆盖 live 与离线输入的依赖投影。import/--help/静态 check 不连接设备或加载模型。这里只验证入口条件，不执行 GPU 模型来证明它。

### G6 — 接受合法环境/worktree；SAM 文档修改不构成算法不兼容

定位：[install_perception.sh][s-installer]（31–51、78–99）、[sam2_config.py][s-sam-config]（48–67）。

修改：

1. 删除 CONDA_DEFAULT_ENV 必须等于 real_robot 的显示名称限制；保留有效 CONDA_PREFIX、解释器归属、既有 Python/依赖/CUDA/compiler/header 条件。没有证据支持任意升级现有版本 recipe。
2. 用 Git 自身确认 checkout 顶层与 revision，接受 .git 是文件的合法 worktree；不能仅因 rev-parse 成功就接受父仓库内部普通子目录。保留目标 symlink 保护、错误 revision/dirty 源保护；不 reset/checkout 用户已有源码。
3. SAM 只读运行保留 pinned HEAD、真正执行的代码/配置、权重/API 匹配；无关 README 文档 dirty 不再硬拒。明确所有实际加载的代码/配置路径集合，不只哈希碰巧已有的三个文件；完整 dirty 状态仍可记录 provenance。安装器“不修改用户 dirty checkout”的规则与只读运行准入分开。

验收：临时 Git fixture 覆盖 clone/worktree/普通目录/父仓子目录/symlink/wrong revision/dirty source；同 prefix 改名通过，解释器属于别处仍拒绝。SAM clean 与 README-only 通过到加载边界，实际源码/配置修改、错误 revision/权重不匹配仍拒绝。不执行安装器主体、下载或 build，不修改真实 SAM checkout 来测试。

### G7 — A_exec 解析一次；实验输入与输出目录标签分开

定位：[config/experiment.py][s-experiment]（176–179）、[run_policy.py][s-run-policy]（173–195、198–230、239–294）、[deployment/config.py][s-deploy-config]（151–164）；[Policy resolver][s-policy-resolver] 已支持有效绝对实验路径。

修改：

- 仅在 Real 内统一：显式 CLI --n-action-steps > 非空 YAML execution.action_steps > saved info.n_action_steps。解析一次，后续消费者使用同一个 resolved execution；run_config 保存一份权威 resolved execution，避免 runtime.execution 与顶层 execution 同值重复。CLI 覆写意图可作为单独追溯信息，实际读取方同步更新。输入与 resolved runtime 子树可回读；不声称整个 run_config 输出文件天然就是输入 schema。
- 保留 P=H-N+1、1<=A<=P；Async/RTC 的 1<=d<=A、A+d<=P、RTC capability/guidance cap 与时间取整约束。--print-config 仍不要求已安装 Policy、加载模型或硬件。
- 实验输入直接调用 Policy 当前公开 resolver；仅在确实创建输出时校验安全标签，使用已解析信息生成输出目录。绝对实验路径不能拼进输出 root，保留目录 containment、去重和不覆盖。
- 不增加第二套 checkpoint resolver、笼统兼容 adapter 或跨仓必改任务。pct checkpoint 特判进一步归还 Policy 的可选 API 改进列入 X5。

验收：CLI/YAML/saved 各单独及组合优先级；例如 H=16、N=2、A=8 的有效 runtime 子树可回读，所有数学非法值仍拒绝。有效绝对路径与标准 selector 定位同一 artifact；缺 config/checkpoint 仍由 Policy 原规则拒绝；无输出时不做无关标签 gate，有输出时不能越界或覆盖。

## 5. D：数据边界、异常归因与 FK 接口

### D1 — 明确坏数据异常，保留系统/程序错误类型

定位：[dataset/export.py][s-export]（80–93、192–196）、[storage/reader.py][s-reader]（19–20、33–38、68–82）。实际函数反例确认普通 ValueError 被误当坏段；writer ENOSPC 被改成 ValueError，但原 OSError 仍在 cause，不能称为完全丢失堆栈。

修改：自动拒段只捕获明确的 Raw 数据/支持格式错误。同步在 parser 边界归类非法 UTF-8、unsupported format 等确实来自输入的错误；不能只机械替换一条 except。删除外层将 ValueError/RuntimeError/OSError/KeyError 全改 ValueError 的包装，带上下文记录后原样传播。用现有 RawDataError 或一个确有必要的格式异常即可，不建异常 registry，不为 add_note 偷升 Python 3.10 最低要求。

验收：坏格式/坏数值明确拒段；人为注入处理程序 ValueError、ENOSPC、目标竞争 FileExistsError 均中止并保留原类别/原因，不能被当坏段跳过。已有目标不覆盖；失败 staging 和 Raw 保留。

### D2 — 不兼容 metadata 任务级 fail-fast：明确改变错误优先级

定位：[dataset/export.py][s-export]（94–168）。旧逻辑发现 task/dt/depth scale/layout 不兼容后仍完整消费派生块；不兼容且数值坏则先 RawDataError、拒段继续，数值好才整任务失败。

**本方案选择：有效 header 与已建立的任务基线明确不兼容时，立即任务级失败，不继续视频解码/FK/点云转换，也不静默跳过。** 这是有意改变双故障输入的错误优先级/整体成功集合，不是宣称完全等价。它不让任何新坏段进入 canonical，避免对混合任务目录悄悄导出一个子集。若用户后来明确要求旧成功集合一项不变，则仅此项 defer；不要假装两者同时成立。

第一段真正被接受的 episode 才确立任务基线；早期已拒段不能留下错误基线。兼容但任一字段数值无效的段仍按原规则整段拒绝。

验收至少覆盖：

| 后续段 | 新行为 |
|---|---|
| compatible + valid | 接受；15 字段、dtype、episode ends、timing、recipe 不变 |
| compatible + invalid numeric | 整段拒绝，其他好段可发布；不留下半段 |
| incompatible + valid | decode/FK/cloud 调用零次，任务失败 |
| incompatible + invalid numeric | 同样 metadata 立即任务失败；明确记录这是相对旧行为的变化 |
| 首段 header/数值坏、后段合法 | 坏首段不建立任务数值基线 |

不要为了快而只看 metadata 接受 episode；所有实际发布段仍完整执行 15 字段准入。

### D3 — timing rows、query provenance、termination 各归其责

定位：[storage/reader.py][s-reader]（140–169）、[storage/schema.py][s-schema]（123–155）。执行计数检查只要 rows，却同时解码所有 query 和可选 termination；坏 JSON 被误归为 Mode-1 evidence 错误。

修改：一个小 rows reader 负责 compound dtype/长度/行对齐；完整 provenance 检查负责 query 唯一性、引用关系；termination 解析报告自己的格式来源。arm execution validator 只消费 rows/start/end/dispatch。**完整 export/replay 入口仍显式调用原有完整准入**，包括 query 关系和存在的 termination 可解析，不因拆读法扩大训练或物理 replay 接受集合。不要为了移动函数额外引入 recording↔robot 循环依赖。

验收：合法两行 episode 通过；duplicate query、missing reference、malformed termination 在完整入口仍拒绝且归因准确；单项 arm 计数检查不再解码 100 条无关 query。Raw 原文件完全不变。

### D4 — fingertip 必需输入收缩、Hand FK 一次导出一个 owner

定位：[kinematics/fingertip.py][s-fingertip]（18–61）、[dataset/processing.py][s-processing]（94–129）。全仓三个调用者都已有 EEF history：processing、deployment/observation、examples/visualize_episode。每个 episode 又重建相同 URDF/link 的 HandKinematics，已用构造计数确认。

修改：

- 接口只接受 eef_pose_history、hand_qpos、hand_fk、mount；删除 arm_qpos、备用 arm_fk 路径及无用 import，更新全部实际调用者，不留 legacy alias。
- 保留 pose (T,9)、hand (T,12)、mount、五指输出、finite 与 float32 溢出检查；SDK→URDF 顺序、五指顺序、米和 xarm_base 不变。
- export 首次真正需要转换时懒构造一次 HandKinematics，显式传给顺序 generator；不做全局缓存或跨线程共用可变 Pinocchio data。
- 每段相机参数仍从对应 Raw 读取；observation FK 与 action FK 输入不同，不能合并成同一数值结果。

验收：三个调用者全部对齐；相同 EEF/hand/mount 输出一致；3 个兼容段只构造 1 次 hand model，每帧实际 FK 不少算；全部在转换前被拒时无需载入 native 模型。真实 Pinocchio 可用时做关节顺序与位置数值对照，缺失时分别报告控制流已验、数值阻塞。

## 6. P：删除有证据的重复工作

### P1 — SceneTracker 每帧 measured cloud 只构造一次

定位：[scene_tracking.py][s-scene-tracker]（83、107–109）、[scene_geometry.py][s-scene-geometry]（68–139）。同一帧两次 extract_scene_clouds 重做 mask/overlap、反投影、颜色和像素数组；assetless fixture 两次得到同一 16 点。

修改：分离 measured extraction 与 model-depth consistency 增补；后一阶段使用已有 pixel_indices 取 model depth，保留 measured points/colors/overlap、支持子集与 residual 的含义。无 model depth 时直接用 measured。现有公共组合函数若确有调用可保留，不新增多层对象。

验收：assetless、多模型 depth、overlap、空 mask、无效深度、全部 pose lost，旧新实测/支持/residual/diagnostics 等价；一帧 extraction 次数为 1。只声明减少一次重复构造，不宣称整管线 FPS 翻倍。

### P2 — 手眼两轴激励 gate 有界向量化

定位：[camera/solver.py][s-camera-solver]（235–261）。N 个姿态生成 O(N²) relative axes，再遍历任意轴对，最坏 O(N⁴)。单次同轴 40 姿态 fixture 有 703 条 qualified axes、246753 对，说明 Python 循环成本；不是现场标定时延基准。

修改：对 qualified axes 的 unsigned angle/absolute dot 做有界块向量化，维持原 qualifying motion、相反轴等同、存在任意一对即通过、阈值端点和原 witness 顺序/统计量。不要无界物化 M×M 数组，不采样轴集，不只与第一条轴比较，不换三轴 rank/SVD。**此方案不承诺改变最坏 O(N⁴) 算术量。**

验收：同轴、反轴、两轴、三轴、接近 motion/axis 阈值输入旧新 gate 相同，返回见证确实满足原标准；记录同一环境的 gate 耗时及最大中间数组规模。候选求解已先过滤平移和旋转双阈值再排序（315–317），该处无需“修复”。

### P3 — 数学表达与重复 FK：分开落地、保持约定

| 定位 | 改法 | 必要对照 |
|---|---|---|
| [pose.py][s-pose]（154–181） | rot6d 的 Gram–Schmidt 已得到 R，直接返回所需 R，去掉 R→quaternion→R 往返 | 正常/退化/阈值邻域；rot6d 轴序、正交性、手性、WXYZ 转换接口不变 |
| [retargeter.py][s-retargeter]（51–94） | 已准入非共线三点的掌面用叉积构造，替换 SVD | 保留 lateral 轴方向、landmark compensation、退化 gate；随机有效三角形及朝向/反向对照 |
| [vr_mapping.py][s-vr-mapping] 与 pyproject.toml | 用现有 SciPy/WXYZ 工具替代唯一 transforms3d 调用，确认全仓无其他调用后去依赖 | quaternion ±、heading/anchor、连续输入与符号连续性；不顺手改滤波/标定 |
| [pin_grad.py][s-pin-grad]（49–53） | 同 q 的 computeJointJacobians(model,data,q) 已更新 joint placements；再 updateFramePlacements；删除之前重复 forwardKinematics | 核对本地 Pinocchio API；连续不同 q 的 fingertip FK、frame Jacobian、目标/解析梯度与 finite difference、优化边界对照 |

官方 Pinocchio [带 q 实现][pin-jacobian] 更新 joint placements，而 [frame update][pin-frames] 仍需执行。**禁止仅删除 FK、却把 updateFramePlacements 留在 Jacobians(q) 之前**，那会读到上一轮 placement。

已有有限随机数值对照支持前两项的简化，但不代替本地支持版本验证；第三项 VR 依赖替换尚无该对照，须本地补齐 quaternion、heading、anchor 和连续序列检查后再删除依赖。对有效输入采用明确且合理的数值容差，shape/order/拒绝类别按原契约；不以放宽阈值让对照通过。不同时改 optimizer loss、pinch/smoothing、迭代预算或 FreeFlyer 根模型。

### P4 — HOME 样本去重、IK 纯日志 reset 删除

- [collision.py][s-collision]（405–428）在路径首点查一次后，每段从 step=0 再查相同起点；[paths.py][s-paths]（203–210）上游已致密化。首点保留，每段从下一样本开始，保持原采样密度、终点及第一失败 segment/index 含义。11 点 fixture 的总查询可从 21 降为 11；不同 self/static/table/workspace 条件继续检查。不要删除第二套不同的物理判据。
- deployment runner → ActionRealizer → planner → [ik.py][s-ik] reset_episode 的唯一效果是重置两个 2 秒日志节流字段（184–186、229–239）。删除这条日志专用生命周期，日志由现有调用层消费 IKResult。真实 policy reset、EMA、VR anchor、retarget warm start 必须保留。
- 同一个不可变 q 在同一责任层重复 bounds/finite 检查，可在触及对应代码时少查一次；保留 INVALID_OUTPUT 与普通拒绝及诊断优先级。不要为此新建 ValidatedConfig/“已验证数组”体系，启动公共边界的少量检查不单列删减任务。

验收：HOME 起点/终点/中间碰撞及失败索引正确，统计唯一样本；日志链删除后所有真算法 reset 仍被调用，故障/取消仍有可定位结果。

### P5 — selection、hash 与 Raw identity 的单一写法

定位：[cli/build_object.py][s-build]（121–142）、[cli/scene_tracking.py][s-scene-cli]（184–211）、[io.py][s-cordvip-io]（61–66、99–132）、[hand_cloud.py][s-hand-cloud]（14–19）、[integrations/dexmani.py][s-integration]（55–64、207–223）。

修改：

1. 在已有 io.py 共用 confirmed selection 的 metadata/mask/hash、已有来源与确认方式写入；build 的 RGBA 是必需建模附件，track 不强制拥有它。保留纯 --image 建模：它可以只有 RGB、source_file/hash，不要求补充或伪造 depth、source_id、clock、timestamp、row，也不强制转换成 RGB-D Frame。
2. metadata 指定的 mask_file + hash 是权威。明确的便利副本/预览不应在未消费时成为第二份权威 gate；权威 PNG/NPY 两种现有产物继续可读，不批量重写实验资产。
3. 合并重复 SHA256 到同一个现有 IO helper；两处 Raw source identity/time/clock 构造集中到 bridge 内一处，**本次先保持实际输出字符串语义**。
4. N 个 selection 各自保存首帧带来的 N+1 份初始化 IO，不以逐帧瓶颈处理；可独立复制的现有目录有实际用途，不为此建缓存服务或强制新 layout。

验收：两 writer 共用 metadata/mask 读写；带 source RGB-D 的两种选择继续通过原 exact-frame loader，纯 --image 建模仍可完成；权威 mask 篡改仍拒绝，真正缺 RGBA 的 generate 仍拒绝；删除/修改未消费的便利副本不妨碍有效权威 mask；same-frame/source/row/time 匹配与取消后旧结果拒收不变。Raw 路径迁移后的选择重绑定不在本项偷做，见 X6。

### P6 — 低优先级、有真实依赖时才验证的局部成本

- [cordvip geometry.py][s-geometry]（207–215）的 rectification map 可由会话持有，key 覆盖图像尺寸、K/distortion 等实际几何输入，变化即重建。保持 RGB linear、depth/mask nearest 和所有权；用本地真实 cv2 对比像素和 map 构造计数。主点云 _depth_rays 已缓存，不重复添加。
- [object_tracking_demo.py][s-object-demo]（88–90）每帧覆盖 last_frame.png。若实际 IO 值得优化，收尾保存最后成功 panel，异常收尾也保留；显式要求的视频/逐帧输出不变。必须说明这会改变运行中 last_frame 的刷新时机，不能冒充完全等价，也不能为一个 PNG 加异步 writer 框架。
- 没有本地依赖或成本证据则保留这两项，不影响其他完成项。无需为了本任务添加长期 telemetry 子系统。

## 7. S：清理间接层、状态归属与代码可读性

这一阶段在 R 修复和局部回归之后进行。先收拢责任，再移动文件；不要把一批改名和三种时序算法变化混成不可审查 diff。

### S1 — 两端对称的具名 lifecycle 协议

定位：[arm_executor.py][s-executor]（16–82、209–258）与 [arm_worker.py][s-worker]（10、113–153）。

修改：一个短 arm_protocol.py（或同等清楚的现有公共位置）放共享布局和少数具名 prepare/HOME 请求、reply。保留 operation_id、epoch、payload、completed/cancelled/failed、detail、late reply/unresolved。取消不再依赖 startswith("HomeAborted:")；通用执行取消不再定义在厂商 driver 中并横穿五层。dtype/共享目标槽保持原机制，不改成高频 pickle/RPC。

原位置 tuple、stats[0/1/2]、-999 等换成实际具名字段/常量；prepared epoch 与下一 running epoch 的对应关系在拥有该关系的位置说明，signed epoch 共享语义不变。只有少数真实原因需枚举，不做异常名/字符串兼容表。

验收：消息往返、错/晚 operation_id、取消、未知状态、stop 时仍有 unresolved reply 均保留正确事实；一个异常类改名不影响协议分类。child 不能导入 parent 的私有协议常量，public import 无硬件副作用。

### S2 — HOME 等待推进归 child，先保持所有阶段

定位：driver.home（215–359）、worker.wait_stationary（62–82）及 parent operation。

迁移顺序：parent 继续产路径、总预算和真实手反馈服务；child 拥有 HOME stage、SDK 调用、取消、超时和结果；driver 规范化厂商原语。把具体 HOME routine 移至 child 执行责任内，按用途共用一个 wait_settled helper，显式列 target/anchor、位置/速度阈值、dwell、deadline 和刷新点。

**本阶段只移动责任、消除同一实现的重复表达。** 路点误差、静止 anchor band、Mode1 切换后稳定等当前判据先全部保留。是否真的能少一次 dwell 属于 H2，需要证明覆盖关系；不能直接删“第二次 0.3 秒”。不新建通用 actuator lifecycle 框架，也不合并父子进程。

验收：R1 全部故障/取消反例仍通过；相同 fake 反馈轨迹下模式/路点调用顺序、接受计数、阶段超时、停止结果相同；移动前后只改变 owner 和代码组织。

### S3 — 会话、UI 请求与运动授权的明确状态

定位：[runtime/state.py][s-state]（34–53）、[deployment/operator.py][s-operator]、[teleop/runner.py][s-teleop-runner]（51–60、175–318、593–632）。

修改：

- 用少量具体状态方法集中 fail/request_stop/request_quit 等转换，锁、首次原因、epoch 撤销和 fault 锁存归一处；先减少各模块直接写字段。
- pending BEGIN/STOP 消费属于 operator/runner，但 STOP callback 到 owner 消费之间的“禁止再次 prepare”必须保留。不能直接删通用 stop_request predicate 造成停止尚未收尾就可重启。
- RuntimeState 的 error_state 与 SafetyState.FAULT 不是已证明相同；先集中转换，不能用一个替掉另一个。RuntimeState 对象本身不是跨 spawn 的共享对象，触及调用时将 shared 改成 runtime_state/session，真正 channels/authority 继续明确。
- Teleop 用一个小的会话阶段表达 IDLE/PREPARING/ACTIVE/PAUSED/QUIT_PENDING 等实际组合；PREPARING 保留 begin/resume 上下文。paused 时仍有 capture，recorder 存在/录制和硬件清理义务仍独立。不要把所有维度做成巨型枚举。
- next_tick_ns 从 origin+index*dt 派生，保留尚未初始化的状态；不再独立保存同一事实。
- hand_stop_pending、_motion_active、stop/stopped/emergency/closing 表示不同设备/阶段事实，不能因都是 bool 一并删除。

验收：开始→准备→运行→暂停→继续→保存/丢弃、quit 后保存选择、准备中 STOP、fault 后禁止自动恢复、STOP pending 禁止重新准备，旧新事件轨迹与录制边界一致。即时 STOP/ESTOP 与排队 UI 两种路径都有效。

### S4 — 在线求解与 HOME 候选职责分开，局部表达更直接

定位：[planner.py][s-planner]（32–78、87–199）和 [action.py][s-action]（43–108）。

修改：ActionRealizer 明确持有在线求解/准入所需能力，避免通过整台 HOME planner 再穿透其内部 collision_model/online_ik_solver。HOME planner 只组织候选和最终路径准入，当前 MPlib CLIK/Pinocchio 后端先保留，不为拆 facade 顺便替换算法或删除依赖。与 R2 共用唯一 endpoint admission，不复制成校准版、键盘版、回放版。

在触及的代码内：

- ObservationRow 的多位置参数改关键字，输入与字段对应可读。
- service=lambda:(read_hand(),check()) 改成有名字、明确顺序的函数。
- jog 计算只消费输入快照；求解函数只返回 IKResult，不隐藏会话终止/录制。
- 状态分类的嵌套三元式改普通分支；结果容器的内部 dict 不由多个外部调用者随意补写，只有实际必要时加一个具体记录方法。
- runtime.policy 实际是 ControlParams，可在统一解析后改成 control；execution 中供 teleop/calibration 共用的 workspace margin 归共享 control。同步改当前 YAML、所有调用者及示例，一次迁移，不维护旧新字段双轨/兼容 alias，不改变数值。

保留 arm_fk 的 WORLD Jacobian 原点/坐标变换（121–138）；它不是可删的空 wrapper。文件名/布局只在确实缩短理解链时调整，不做整个 src/wheel 迁移。

验收：构造后端数与数值结果无意外变化；所有已支持入口使用同一最终准入；当前配置和 --print-config 可解析；跨模块搜索无旧名称、死 import、私有成员穿透或日志专用 wrapper。对应数学/物理 gate 不变。

### S5 — PolicyRunner 显式结果与时序：先不新建调度框架

**事实与定位。** [deployment/runner.py][s-runner] 的 _submit（501–515）、_prepare_prefix（553–563）、_realize_command（588–594）、_command_for_slot（598–605）、_handoff（707–727）、_prefetch（730–744）会直接 _finish_episode；step（794–810）和 _execute_slot 再靠 run_id is None 推断副作用。_finish_episode 同时撤权、stop、record、finalize，故问题是责任混合，不是文件有 890 行。

**最小实现。**

1. 先保留 PolicyRunner 及现有 Query/Plan。各计算 helper 返回 command/prefix/Plan/等待/明确拒因，不调用 _finish_episode、不清 run_id、不保存/stop。现有协调层 step/_execute_slot 处理终止，保留对应 row、command、dispatch。
2. _submit/_prefetch 的提交副作用仍由 runner 明确持有，返回结果；先成功提交 query，再把准备好的 RTC commands 提交到 plan.frozen，不能留下 partial prefix。
3. 拒因使用现有结果类型；确需补字段时只加小的局部 reason/detail record，不建通用 Result/事件总线。时序块已清晰后，才决定是否把 Query/Plan 与纯 slot/handoff 计算移到短 chunk_schedule.py；**新增该文件不是验收要求**。
4. [InferenceWorker][s-inference] 单模型串行 load/reset/predict 保留：最多一个未回收工作任务；close_future 可以串行排在现有任务后，不能错误限定任何时刻只有一个 Future 对象。query context 与 future 对应，不让两边状态漂移。
5. Sync/Async/RTC 共享生命周期；不拆三套复制 stop/record/prepare 的 Runner。动态 authority、budget、sources freshness、SDK 前后时效仍在实际消费点检查，INVALID_OUTPUT 仍属技术错误。

**必须建立的假时钟案例。** 使用 N=2、H=9、P=8、A=4、d=2、dt=100 ms；fake 感知/SDK 的其他准入均有效。slot0/1 累积 history 后发 bootstrap query。

| 模式/事件 | 应保持的当前语义 |
|---|---|
| Async/RTC bootstrap 在 t=350 ms 就绪 | 下一网格 t=400 ms 的 slot4 接管：anchor=4、segment_end=8；slots4..7 执行该 future indices0..3 |
| Async/RTC slot6 prefetch | new query.slot=6、handoff=8；等待时 slots6、7 继续执行当前计划，不暂停或补发 |
| 新结果在 handoff 前 ready | slot8 接新 plan：anchor=6、segment_end=12；slots8..11 执行新 future indices2..5，不从 index0 重播 |
| RTC slot6 prefix | 从 old.actions[2:] 复制完整剩余尾部；仅前 d=2 行先 realize 成真实命令并回写 guidance，冻结 slots6、7；模型 delay_steps=2。不要为了简短只保留 d 行 prefix |
| RTC 实际发送 | 使用冻结命令，重新做 frozen_rejection/动态 gate；不得再次 IK 得到另一目标。下一 prefetch slot10 用新 anchor6 计算 start=4 |
| Sync 已等待到 slot3，t=350 ms 就绪 | 用最新物理观测立即推进下一 slot4，从 future[0] 开始；随后按 100 ms 发送 1/2/3，保留 t=350/450/550/650 ms 的真实时间；整段后清 plan，不 prefetch |
| 下一 Sync query 等待 | 保留真实 pause，ready 后从该完整 block index0 开始；不按 elapsed slots 剪掉头部，不伪造 Async 均匀时间 |

再覆盖：handoff 未 ready 保持 handoff_result_not_ready 终止；now>=next+dt 保持 missed_slot，不 catch-up burst；retired run_id 结果只回收不下发；未回收任务不启动新 reset/predict；history 不足/已有 query 的 prefetch_unavailable 仍拒绝；prefix 中撤权不提交部分 frozen/模型请求；输出仍须 finite 浮点 (P,control_action_dim)，joint=19/eef=21 按当前 info。保留 query_id/prediction_index/run_id/source clocks、实际 dispatch、最后失败 row；writer 已知提交失败不 retry。

这些是控制流/时序契约检查，不需要实际模型或设备。若当前版本已变，先从真实入口重建等价时间线，不把旧数值例强塞成新算法。

## 8. H：完整本地几何/后端条件下的后续工单

### H1 — 让 HOME 搜索与最终准入看到同一有效场景

**确认事实：** [planner.py][s-planner]（55–73、95–102）中的 MPlib 搜索/库 simplifier 没接到独立 CollisionModel 的 table/static boxes/当前手形。MPlib 使用七关节固定手几何路线，最终碰撞模型另有 19-DOF 表达；不能靠把 q 数组直接互传解决。最终 [paths.py][s-paths]（239–241）仍 fail-closed。

**满足以下条件再改：** 本地已有兼容 MPlib、Pinocchio、完整 URDF/meshes；能读取实际 API，并运行纯离线搜索/碰撞样例。不要安装升级或为云端缺依赖删 fallback。

先用同一障碍和手形构造“当前搜索候选被最终拒绝”的确定反例，再选最小可用适配：搜索 world 和最终 world 使用同一当前环境/手形，或通过确实存在的 API 共用碰撞判据。先核对 API 是否支持，不臆造 callback，不引入通用 scene graph。保留当前 q/frame/joint 顺序与模型覆盖范围，nominal mount 差异另见 X3。

验收：固定候选/随机种子，无障碍原路径通过；直线被障碍挡但确有可行绕行时能找到通过最终准入的路径；无解仍明确失败；改变实际参与碰撞的手形对搜索和最终判定一致；终点补齐后再最终检查。没有可验证支持时保留现 fallback 与 fail-closed，并报告明确限制，不将功能缩水冒充修复。

### H2 — 路径后处理、15°、dwell 与桌面 fallback 的收缩条件

这一组不能作为“删除冗余校验”的无条件批量修改。

- **15° 和 shortcut：** planner（108–183）既用库 simplify，又自写三轮 shortcut，并有只针对 RRT 的 15° gate；direct/proximal/distal 没同一 gate。待 H1 与统一最终 HOME 准入成立后，才去掉重叠自写 shortcut、保留一个候选简化环节，并评估删除 RRT 专属门槛。必须保留起点与实际姿态一致、最终 canonical target、关节限位、速度/加速度/控制方式，以及原密度 swept sampling。删除 gate 扩大候选接纳集合，应明确写出。
- **样本与 milestones：** 候选生成产执行 milestones，统一 admission 生成致密检查样本；不要把检查样本都变成需要固件 settle 的路点，也不要用稀疏 milestones 取代 swept 检查。几何条件可共享同一 q 求值，但每条条件依然存在。
- **HOME dwell：** S2 先收拢阶段。先明确各阶段判据与时序，说明剩余逻辑如何覆盖原 target error、anchor drift、velocity、连续 dwell 时长、deadline 和反馈刷新条件，再用相同反馈轨迹及边界反例组对照；不能仅凭少数 fake 轨迹通过就删除计时。无法说明覆盖关系时保留清楚命名的阶段，不为提速减小 settle 时间。
- **桌面双源：** paths（22–37）在 table 未启用时仍用 arm.table_z_surface_m/hand_safety_margin_m 的 proxy。使 mesh/proxy 选择在一处明确，保留现有默认含义；不能把 enabled=False 解释成无桌面保护。没有当前有效平面/模型依据时，不强行统一成 mesh，不改当前标定。

验收分别覆盖长但合法的 Mode-0 milestone、失败中间样本、补终点、同轴/手形障碍、目标已到但 anchor/速度未稳、no-table proxy。无法证明某段条件被覆盖时保留该段；其他已证实收缩可独立完成。

## 9. X：暂缓、否决与保留项

| ID | 裁决与理由 | 本次动作 |
|---|---|---|
| X1 radius graph/实时 SDK 开销 | pointcloud 的 radius graph 在完整 voxel 点集上运行，8192 仅后续采样上限；改变顺序会改密度/连通语义。串行 SDK 查询的 deadline 风险也需实际时延分布 | 仅用已有离线输入/替身做有针对性 profile；没有真实瓶颈不建优化子系统，不预采样、不改阈值/丢帧/补点、不宣称默认 16 Hz 必超时或 OOM |
| X2 aligned depth Z | [项目反投影][s-pointcloud] 与 [SDK alignment][rs-align] 表明保留 depth-Z 后在 color grid 反投影存在条件近似；合成 t_z=5 mm 不是现场误差 | 不改 Raw/采集格式/深度含义；若有用户已提供 native extrinsics，可离线量化误差预算。是否换表示由论文精度需求决定，不连接相机补数据 |
| X3 actual/nominal hand mount | [hardware.py][s-hardware]（277–284）actual x=-0.015 m；[nominal URDF][s-urdf]（277–281）x=-0.005 m；注释已披露，另有微小旋转精度差 | 手基点平移差 10 mm 成立，不能声称所有表面点严格同一偏差或 nominal 必为保守包络；不自动改 URDF、mount、碰撞阈值或历史 Raw |
| X4 固定根 PinGrad | identity FreeFlyer 的固定根替换可能简化维度，但涉及 nq/nv、frame、gradient 和 optimizer 参数绑定 | 与 P3 删除重复 FK 分开；本次默认保留。仅在完整两阶段 loss/gradient/FK 对照和具体研究需求下另做数值迁移 |
| X5 跨仓训练/加载责任 | Policy 本应持有 checkpoint/normalization/模型表示；Real 中 pct 特判和包路径推 Git 根仍有局部边界债务 | G7 先复用现有 resolver；不要求改兄弟仓、不整搬 dataset/retargeting/deployment。未来 Policy 真正提供已解析 checkpoint/source identity 后再删除剩余窥探，不能堆 fallback adapter |
| X6 实验资产搬迁 | Raw source_id 含绝对路径，目录移动会使旧 selection source mismatch；有反例，但 provenance gate 有真实用途 | P5 只统一构造。用户确实搬资产时才离线核对旧/新 RGB、depth、K、distortion、row、time、clock，写派生 selection 并更新 scene 引用；不改 Raw，不默认新建重绑定 CLI/UUID schema/全 episode hash |
| X7 单调用 helper、CLI 与进程 | 两个 13 行 Cordvip wrapper 提供 checkout/workspace 语义；固定 7 命令表是懒加载；SAM parent/worker/协议与 Hunyuan 退出有环境/CUDA 归属 | 保留这些边界；_process 的 17 行单调用 helper 可在触及 build 时内联。--sam-config→variant、build stage 进程重排、两 viewer 合并默认暂缓，不能只为文件数改公开接口 |
| X8 重复验证与数据 schema | Raw 与 canonical、live supported fields 的消费者不同；物理→派生、float64→float32 的有限性边界不是同一检查 | 不合并动态 schema registry，不为少扫一次 tactile/stack finite 新建 pass-through 标志；动态 gate、公共 15 字段、query 完整性、真实数据所有权均保留 |

上述 “defer/保留” 是具体的证据与收益判断，不是发现无关门槛后不处理。后续若出现新输入、profile 或研究目标，再单独改变这些边界。

## 10. 参考项目如何落到本仓

| 参考源码 | 采用的具体写法 | 不迁移的部分 |
|---|---|---|
| [LeFranX servo owner][ref-lefranx-owner] 与 [显式组合 arm/hand/camera][ref-lefranx-composite] | 独立硬件 owner、固定组件直接构造；缩小协议，不建机器人 registry | libfranka 速度回调的周期/控制参数不能照搬 xArm；发送失败仍回目标或 stop 只打印的实现不作为简化标准 |
| [LeRobot inference contracts][ref-lerobot-contracts] 与 [policy runner][ref-lerobot-runner] | 模型/处理器明确归属，数值 capability、generation、capture provenance 与观测所有权 | 不引入完整 processor registry；不把 target/hold 当物理已停，不用 DatasetWriter 的 index/fps 重写 Raw 时间 |
| [ManiUniCon model/obs/act][ref-mani-model] 与 [robot process][ref-mani-robot] | 小而直接的观测/动作接口、进程 owner；Real 只适配真实数据和执行 | 不照搬无限等待/join、重复样本时间对齐、迟到 chunk 整体移时 |

Real 继续拥有设备、现场标定、Raw、物理处理、teleop/retargeting、真实 deployment/replay/evaluation；模型、通用 Dataset、归一化、损失与训练留 Policy。Cordvip 保持可选感知包，integrations/dexmani 是桥；真机集成使用 Real 已采数据，不能另开同一相机。实测云、模型云、FK 手云和 proximity/residual 来源仍分开，不将其标作实测接触真值。

这次“迁移”主要移动具体责任/接口、删除备用分支，不是整体搬目录或改成新框架。

## 11. 本地验证、文档和完成标准

### 11.1 验证方法

先对 R1–R4、D1–D3、S3/S5 建立会失败的最小控制流/事件反例，再验证修复；数值项比较旧新函数的实际输出。使用普通 Python assert、unittest、假时钟和明确 SDK/model/storage 替身即可，临时文件放 reports/。能正常 import 目标函数时直接使用；避免只对 AST 字符串/实现长相断言，或为了跑 smoke 构造真实设备。

为一个明确风险设计检查；同一 helper 的多入口可复用 fixture，不为每个改名写测试，不建立永久测试框架。已有 codec/cv2/Pinocchio/MPlib 时补相应小型真实离线检查；缺依赖时记录“哪条数值/后端验收未完成”，不要伪造 passed，不重复跑无关测试。

在实际 Git 工作树中按 AGENTS 运行：

~~~bash
python -m compileall -q dexmani_real examples
git diff --check
~~~

若修改可选包，也对 third_party/cordvip_geometry/src 做 compileall；这些命令不执行模块代码。读取 import/--help 的副作用后才选用适当入口检查。不得执行整个安装器、现场诊断或真实 rollout 作测试。

### 11.2 必要的行为变化清单

完成说明必须区分等价整理与以下有意改变：

- R：此前不安全/错误/丢数据路径被停止、拒绝或正确保留，不承诺真机已验。
- G：原先受无关条件限制的有效输入现在被接受；有效活跃数值表示/硬件条件不变。
- D1：程序/系统错误不再误归为坏段，外部异常类别变得准确。
- D2：不兼容且数值也坏的输入改为 metadata 任务级 fail-fast。
- P5：未消费的便利 mask 副本不再决定权威选择是否有效，原权威文件和来源校验仍存在。
- S4：若重命名配置键，当前 YAML/入口一次同步，用户自有配置的明确旧→新映射写入完成说明；不默认保留双轨兼容。
- H2/P6：只有真正实施时报告候选接纳集合、等待判据覆盖或 last_frame 刷新时机变化。

不要给所有改动统一贴“完全无行为变化”的标签。

### 11.3 文档与清理

修正 Cordvip README/usage 的“13 字段”为 15；同步 G/R/S 修改影响到的真实命令、配置、输入/输出说明。root README 只保留定位、workflow、quick start 和主要入口；细节就近落在已有文档/函数注释，不塞状态机、内部类清单和本轮审查史。

版本 recipe 和安装说明尽量链接一份现有权威定义，不创建自动文档生成系统。公开 Python API、内部 worker 模块路径、示例 checkout/workspace 语义有调整就更新全部当前调用者；不留 dead code、旧参数、重复 writer、旧配置别名。保留本任务书供用户验收，执行进度不要再扩散为根目录多份 task/migration 文档。

### 11.4 Definition of done

本地最终回答提供：

1. 每个 R/G/D/P/S/H/X 工单的状态：done / already-fixed / blocked / deferred，注明实际修改文件和对应证据。blocked/deferred 简述具体原因、尚缺的验证及是否已有未充分验证的代码 diff。
2. 关键前后结果：控制反例是否闭合；公共准入覆盖哪些入口；metadata prefix 行数；相同输入的数值误差；FK/点云/路径重复调用计数；Sync/Async/RTC 的动作索引与时间线；不兼容 metadata 的新错误优先级。
3. 实际运行的命令、通过项与限制，明确没有真机验证。没有测量就不报 FPS、毫秒或提速百分比。
4. 结构收益用“减少了哪个 owner/隐式副作用/备用路径/同值重复工作”解释；文件数和行数只能辅助。列出任何新增文件/类型的实际责任，避免以新抽象抵消删减。
5. 最终 diff 只含本任务相关修改，无安装升级、Raw/标定改写、用户无关修改损失或自动推送。普通可逆实现遇到问题先处理其他已授权工作，不以询问是否继续替代执行。

**完成是默认工单在实际环境可验证范围内落地并如实交代其余状态；不是对硬件安全、模型精度或“仓库再无复杂度”作没有证据的保证。**

[real-base]: https://github.com/haoyangzhanglab/dexmani_real/commit/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7
[policy-base]: https://github.com/haoyangzhanglab/dexmani_policy/commit/22884db02778db9bcd16ebdc0b9b6a529ef3afc5
[lefranx-base]: https://github.com/wengmister/LeFranX/commit/a39906e6629f39490950fe8bd20f4f992ed74fd7
[lerobot-base]: https://github.com/huggingface/lerobot/commit/146d8474932c8b566717eb2738ba459337ab71c4
[mani-base]: https://github.com/Universal-Control/ManiUniCon/commit/85c6f2e32ecf9f2bed62d202b058c39623444686
[sdk-stop]: https://github.com/xArm-Developer/xArm-Python-SDK/blob/d911319cebc45142613e18086aabb8067b18ab7f/xarm/x3/xarm.py#L1109-L1118
[s-policy-resolver]: https://github.com/haoyangzhanglab/dexmani_policy/blob/22884db02778db9bcd16ebdc0b9b6a529ef3afc5/dexmani_policy/deployment/runtime.py#L23-L39
[pin-jacobian]: https://github.com/stack-of-tasks/pinocchio/blob/187afafcfe22d7ac16a26241c0b13a76d04d82c1/include/pinocchio/algorithm/jacobian.hxx#L48-L94
[pin-frames]: https://github.com/stack-of-tasks/pinocchio/blob/187afafcfe22d7ac16a26241c0b13a76d04d82c1/include/pinocchio/algorithm/frames.hxx#L15-L35
[rs-align]: https://github.com/realsenseai/librealsense/blob/c84fe6438bd14a4cc69f46deae92c993268232c6/src/proc/align.cpp#L115-L121
[ref-lefranx-owner]: https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/franka_server/src/franka_server.cpp#L411-L465
[ref-lefranx-composite]: https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/franka_fer_xhand/franka_fer_xhand.py#L30-L39
[ref-lerobot-contracts]: https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/inference/contracts.py#L53-L138
[ref-lerobot-runner]: https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/inference/policy_runner.py#L52-L79
[ref-mani-model]: https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py#L40-L66
[ref-mani-robot]: https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py#L253-L307
[s-worker]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/robot/arm_worker.py
[s-xarm]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/robot/drivers/xarm7.py
[s-executor]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/robot/arm_executor.py
[s-robot]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/robot/robot.py
[s-action]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/robot/action.py
[s-replayer]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/replay/replayer.py
[s-keyboard]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/teleop/keyboard_session.py
[s-camera-session]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/calibration/camera/session.py
[s-replay-session]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/replay/session.py
[s-evaluation]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/replay/evaluation.py
[s-pc-example]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/examples/pointcloud_process_example.py
[s-build]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/cli/build_object.py
[s-experiment]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/config/experiment.py
[s-environment]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/config/environment.py
[s-pc-config]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/config/pointcloud.py
[s-deploy-config]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/deployment/config.py
[s-pointcloud]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/sensor/pointcloud.py
[s-integration]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/integrations/dexmani.py
[s-rs-example]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/examples/realsense_record_example.py
[s-check]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/cli/check_tracking.py
[s-scene-cli]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/cli/scene_tracking.py
[s-live-viewer]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/live_viewer.py
[s-installer]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/scripts/install_perception.sh
[s-sam-config]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/sam2_config.py
[s-run-policy]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/examples/run_policy.py
[s-export]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/dataset/export.py
[s-reader]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/recording/storage/reader.py
[s-schema]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/recording/storage/schema.py
[s-fingertip]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/kinematics/fingertip.py
[s-processing]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/dataset/processing.py
[s-scene-tracker]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/scene_tracking.py
[s-scene-geometry]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/scene_geometry.py
[s-camera-solver]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/calibration/camera/solver.py
[s-pose]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/kinematics/pose.py
[s-retargeter]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/teleop/retargeting/retargeter.py
[s-vr-mapping]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/teleop/control/vr_mapping.py
[s-pin-grad]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/teleop/retargeting/pin_grad.py
[s-collision]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/collision.py
[s-paths]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/paths.py
[s-ik]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/kinematics/ik.py
[s-cordvip-io]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/io.py
[s-hand-cloud]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/hand_cloud.py
[s-geometry]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/geometry.py
[s-object-demo]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/third_party/cordvip_geometry/src/cordvip_geometry/cli/object_tracking_demo.py
[s-state]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/runtime/state.py
[s-operator]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/deployment/operator.py
[s-teleop-runner]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/teleop/runner.py
[s-planner]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/planning/planner.py
[s-runner]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/deployment/runner.py
[s-inference]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/deployment/inference.py
[s-hardware]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/dexmani_real/config/hardware.py
[s-urdf]: https://github.com/haoyangzhanglab/dexmani_real/blob/3e5a48dc2f7995b225c46852fa7bd60a8a8f53f7/assets/robots/xhand/xarm7_xhand_right.urdf
