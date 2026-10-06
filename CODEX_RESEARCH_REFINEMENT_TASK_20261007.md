# Codex CLI 任务书：完成已核实整改，收敛研究代码与实验闭环

日期：2026-10-07（Asia/Shanghai）

主仓库：`haoyangzhanglab/dexmani_real`

协同仓库：`haoyangzhanglab/dexmani_policy`

状态：**方案经第二轮实施审校通过；实施与纯离线验证已完成，逐项结果见第 17 节，U09 补充修复与验收收尾见第 18 节。真机未验收。**

本轮收紧了旧 resume 兼容、replay 输出初始化、manifest 身份与内容留存、导出异常阶段，以及控制等价验收边界；不新增工作包或运行框架。

## 0. 给执行代理的指令

当用户要求执行本任务书时，先读两个仓库适用的 `AGENTS.md`，然后按本文顺序完成尚有效的修改、定向离线验证和可审查 diff。不要止于重新给计划，也不要为常规实现选择逐批请求确认。本文是自包含任务书，不依赖聊天、外部工作簿或其他机器上的临时探针。

目标是 RAL 论文实验代码：**安全与科学正确性 → 实验可追溯 → 迭代速度 → 简洁可读 → 泛化扩展**。保留真实硬件、进程、数据边界；优先删无用路径、复用已有类型与社区库，不建设生产平台或 benchmark 框架。

### 0.1 基线与范围

| 仓库 | 已核查实现基线 |
| --- | --- |
| Real | `1b4b6cfb795c4cb40c96ae9d1d992a21f84e3072` |
| Policy | `4c97cfbdb4ea9761a2fe03907d0a28c10c4f1a42` |

SHA 只定位证据，**不是 reset/checkout 目标**。检查当前 HEAD 与相关差异，保留用户已有修改；已经修复的项用代码/测试证据标记 `ALREADY_FIXED`，不重复重写。上轮离线环境的 Policy checkout 为 `744f14f`，已通过 diff 和 AST 核实：所用 datasets/deployment、`validate_data_identity` 与上述最新基线一致；本次设计另外核对了最新配置和 evaluation protocol。

本次覆盖核查表 U01–U34：30 项修复/简化/重构建议，3 项条件决策和 1 项未复现问题。并非 34 个 bug。第二至第四轮结论完整；第一轮只恢复了后续明确转述的条目，因此本文不虚构缺失首轮原文中的其他问题。

设备、控制、标定、Raw、导出和真实执行证据属于 Real；训练 Dataset、split 消费和 resume 属于 Policy。执行本文包含必要的跨仓库小改动，分别维护 diff/验证结果；不得在 Real 复制 Policy 的 Dataset/sampler。优先复用现有两个工作树；缺少 Policy 时在允许的工作区建立独立 checkout，不能覆盖既有目录或改 site-packages。访问或依赖受阻时完成其他独立任务，再清楚列出阻断。

### 0.2 权限与验证范围

- 实施允许代码/文档与无硬件离线验证；默认交付 diff，不自动 commit/push 实现。
- 不连接或驱动 xArm、XHand、RealSense、VR、HTS；不执行 HOME、replay、rollout、实时采集或真实标定写入。`execute=False` 不天然意味着没有连接副作用。
- 允许临时目录的合成 Raw、虚拟标定、fake clock/model/SDK 和原生离线 FK/碰撞。无硬件授权时，真机结果一律 `NOT_VERIFIED`。
- 不运行完整训练/DDP/长时间评测，不升级实际实验环境来凑测试，不修改已发布 Raw、checkpoint、既有实验或无关缓存。
- 小阻断只影响相应项：缺实物量测、真实资产、GPU，不阻止完成源码一致性、接口、离线反例和其他任务；也不把未验证项标成通过。

### 0.3 与旧任务书的关系

本文是 **U01–U34 的当前实施依据**。根目录 `CODEX_SIMPLIFY_FULL_MODAL_TASK_20261003.md`、`CODEX_POLICY_EVAL_RTC_TASK_20261004.md`、`CODEX_REVIEW_REMEDIATION_TASK_20261005.md` 保留为历史设计/执行记录，不重放已完成整改，不批量删除。

明确裁决：

1. 旧文档“暂不生成运行期 URDF”在 T01 有一个限定例外：只把当前 mount 送进现有模型，不发展生成框架。
2. 独立公开入口与连接前验证继续保留；T08 不要求 `validate` 在全仓只出现一次。
3. 删除无需求的模式 setter 后，不再维护其“切换后重验”路径；初始化 warmup/预算与三模式算法保留。
4. 先单独完成新路径导出/revision，再重构 export。旧风格建议的“不要同时改 overwrite”是拆分 diff 的要求，不是保留 overwrite 的理由。
5. 旧任务书对 Raw、NaN、action/dispatch、时间、normalizer 和 RTC 的正确约束继续有效。不得修改 `AGENTS.md` 来绕开它们；本文件是用户明确指定保留的根目录任务书。

## 1. 最终架构取舍与不变量

1. 保持一个硬件 I/O owner、一个串行模型 worker、至多一个在途 Future、一个 episode FIFO writer；不新增 scheduler/service/backend/validator/processor 注册体系。
2. `intent` 是模型/操作者意图，`command` 是准备下发目标，`dispatch` 是逐设备 SDK 尝试结果，`state` 是实测；`ACCEPTED` 不等于物理到位，`completed attempt` 不等于任务成功。
3. 先检查全部存在的 arm/hand 目标，再进入第一个 SDK；每个 SDK 前仍查 epoch、当前授权与 deadline。不能把 motion_lock 持有到阻塞 SDK/文件 I/O 中。迟返回不改写已经发生的 ACCEPTED。
4. `RobotCommand` owned/不可变；RTC prefix 必须对应同一份 frozen physical command，拒绝时失效，不重新求解/投影成另一条命令。
5. Raw 发布后不可变；保留失败前缀、真实终止主因、缺测 NaN、selected target 和部分 dispatch；仅显式 discard 删除当前 capture。writer/编码/发布失败保留 staging。
6. 公共导出仍为全部 13 字段；模型只检查实际使用的 N 步观察、H 步动作与 dispatch 资格。禁止辅助缺测自动删整段、补零、黑图、静默插值或内部坏行压紧后重拼窗口。
7. 当前 mount/桌面/运动学按当前配置；相机用数据对应的内外参/depth scale；触觉不二次扣 bias。不引入历史 calibration ABI。
8. 保留训练集唯一源行统计、val 复用 train normalizer、checkpoint 保存的表示/normalizer、dt/N/H/A 和 RTC 适用条件；不以 tensor shape 代替数值语义。
9. 不顺手改模型、控制频率、动作表示、smoothing、fallback、自动恢复或研究超参数。在线端点自碰撞检查不是连续轨迹/环境避碰证明；Raw replay 不等于在线 ActionRealizer 路径。
10. 动态事实在实际副作用边界复核；静态资源在按需构造/首次使用时核查；纯派生统计离线完成。不新增每 checkpoint 全量数据哈希或全传感器统一 strict gate。

## 2. 工作包、依赖与覆盖清单

| 顺序 | 工作包 | 覆盖 | 处理要求 |
| --- | --- | --- | --- |
| 1 | T01 当前几何单源 | U01 | P0；先做，实物参数另行确认 |
| 2 | T02 共享目标准入 | U02、U16 | P0 主项；依赖 T01 的正确几何；手部谓词小改用独立 diff |
| 3 | T03 标定有效性 | U03、U04 | P1；与 T01/T02 可独立修改 |
| 4 | T04 attempt/session 结果 | U05 | P1；先定义结果生命周期，再改相关控制结构 |
| 5 | T05 新路径与缓存身份 | U06 | P1；先于 export 结构重构 |
| 6 | T06 trial split 与时间 QA | U07、U08 | P1；manifest 消费依赖 T05 身份/顺序；真实资产影响规模待核查 |
| 7 | T07 最小归因与时序摘要 | U09、U10、U19 | P2；依赖 T04/T06，不先做并发优化 |
| 8 | T08 删除机制与静态去重 | U11–U15、U17、U18 | P2；公开边界和当前行为保持 |
| 9 | T09 控制回路可读性 | U20、U21 | P2；在 T02/T04/U11 后做等价重构 |
| 10 | T10 数据、资源、指标与命名 | U22–U30 | P2/P3；小步分批，低收益项允许有依据保留 |
| 11 | T11 条件决策 | U31–U34 | 明确保留/触发条件；不得强造 bug 或为勾选而改代码 |

每包结束更新简短状态：`DONE / ALREADY_FIXED / RETAINED / BLOCKED`，验证另列 `PASS / FAIL / NOT_VERIFIED`；未复现问题标 `NOT_REPRODUCED`。`RETAINED` 必须给当前证据和理由，不能用于跳过明确缺陷。不要为每个 helper 建一个独立项目或测试框架。文中的 helper 名称是推荐实现，行为边界与验收才是硬要求；当前源码存在更短且等价的组织方式时可采用，并说明取舍，不机械增加函数或类。

## 3. T01：当前 mount 进入所有真实几何模型

### 3.1 已确认事实与选定方案

当前 `config/hardware.py::HandParams` 的平移为 `(-0.015,0,0)m`，完整机器人 URDF 的 mount 为 `(-0.005,0,0)m`；两组姿态×五个指尖的原生 FK 差约 10 mm。这证明模型分歧，**不证明哪个数值符合实物**。

唯一配置来源继续使用 `runtime.hand.T_eef_handbase_pos_xyz` 和 `T_eef_handbase_quat_wxyz`，表示 `custom_eef_link -> right_hand_link`、m、wxyz。保留当前配置值作为现行输入；不擅自把 -5 或 -15 mm 宣称为量测真值。代码一致性和离线验证可以完成，重新启用相关真机运行前需要操作者确认实际尺寸/方向。

选定实现是一个小型纯 URDF XML helper，例如 `planning/kinematics/urdf.py::urdf_with_hand_mount`：

1. 读取已有 URDF 模板，仅定位唯一 `right_hand_mount_joint`；确认 fixed 类型、parent/child，覆写 origin xyz/rpy。缺少/重复 joint 报错。
2. 保留关节名、active q 顺序、limits、SRDF 和其他几何。把相对 mesh 路径相对原模板目录解析，不能相对临时目录解析。
3. Pinocchio model/geometry 消费同一份解析后 XML。当前核查的 Pinocchio 2.7 有 `buildModelFromXML`、`buildGeomFromUrdfString`；实施时确认实际安装 API，不能偷偷升级依赖。
4. MPLib 的 filename API 使用限定作用域临时 URDF；确认构造后是否还读取文件，临时资源生命周期覆盖所有真实读取。不要持久化生成资产、建磁盘缓存，或把临时路径写入数据身份。
5. 同时覆盖 full 19-DOF collision 模型和 fixed-hand 7-DOF MPLib 模型，以及 self/environment geometry。**只改 Frame.placement 不合格**，因为 fixed joint 折叠后子 joint 与 collision placements 也受影响。

明确传参进入 `CollisionModel`、`XArm7MotionPlanner/create_default`；搜索并更新所有真实构造者，至少包括 `robot/action.py::for_mode`、`robot/arm_homing.py::build_policy_home_planner`、`calibration/camera/session.py::_build_planner`、`teleop/keyboard_session.py`。feature FK/export 继续消费相同 runtime 来源。生产路径不静默退回 URDF nominal mount；模型测试若使用 nominal asset，须显式说明。

### 3.2 验收

- 两组以上非零 arm/hand 姿态，feature fingertip 与 full model 的五指 frame 对齐；旋转文本精度统一后，位置容差可用 `1e-6 m`。
- 用两个不同的非零平移/旋转 mount，验证参数确实到达 self/environment 和 MPLib 的手基座几何；不能只验证默认配置。
- SDK→URDF 顺序不变、mesh 可加载、临时文件不泄漏；不要把旧错误几何的碰撞 pair 数量冻结为 oracle。
- 实物 mount 与真机碰撞/停止验证明确 `NOT_VERIFIED`。

## 4. T02：共享已实现目标准入，保持各执行路径语义

### 4.1 最小实现

改动重点为 `robot/action.py`、`planning/kinematics/ik.py`，以及 runner 的拒绝处理。joint 目前只投影/限位；EEF 还有目标跳变、feedback 距离和端点自碰撞判据。

抽取一个纯关节目标准入谓词，例如 `joint_target_rejection(q, current, previous, ...) -> str | None`，复用现有 operational limits、每关节 jump 和 feedback 距离规则。当前 jump 为 `(30,30,30,35,40,40,40)` deg，feedback 上限为 150 deg：这些是现有规则，先保持数值；只保留一份默认来源，不新增 joint 的宽松平行配置，也不声称阈值已经实物认证。

- **joint**：按当前规则投影一次，然后检查实际目标的 finite/shape、operational limits、绝对 `q-previous_command`、绝对 `q-measured_current` 和端点 self-collision。不能用周期最短角距离替代绝对硬件目标差。
- **EEF**：保持求解、候选评分、pose residual 和 collision-aware 选择。抽取公共判据，不改变候选顺序；同一状态下已经完成的昂贵碰撞检查不重复整轮执行。
- **frozen**：执行时检查原命令，不重求 IK、不投影/换支；joint 原有“当前投影若会换支则 reservation 失效”仍保留。碰撞复检必须使用**该 command 的 hand_qpos**，不能使用预制前缀最后一条留下的手姿。
- joint 只构造准入需要的 `CollisionModel`，不为了共用检查构造整套 CLIK/MPLib/RRT；EEF 复用现有 planner 几何。hand-disabled 沿用 ActionRealizer 的 `np.deg2rad(runtime.hand.home_qpos_deg)` 固定手姿。每次 collision query 的手姿来自该次目标/命令或这一明确假设，不能依赖上次 `set_hand_qpos` 遗留的状态，也不能把 full-model 的配置手姿与 7-DOF asset 的固定手姿混为一谈。
- `ActionRealization` 复用现有类型，增加通用 `rejection_reason`；`ik_result` 仍可空。修改 `_prepare_prefix/_send` 等调用者，不能在 joint 正常拒绝后直接解引用 `ik_result.failure_kind`；只有实际 IK 技术失败按既有规则升级，普通拒绝保留原 invalidate/WAIT/结束语义和原因，不自动发送上一条目标。
- frozen 被拒时按现有规则失效旧 plan/reservation；previous command 仍在原 dispatch 提交条件下推进，不在“已准备”时推进。

U16 同包但独立小改：`XHand._validate_action` 复用 `check_hand_target` 的 shape/finite/机械限位及容差。保留 Robot 双侧整体预检和 driver 直接调用/`hold_current -> send_action` 的边界，逐 SDK 动态检查继续存在。

**范围边界**：当前 Raw replay 直接通过 RobotCommand→Robot，不通过 ActionRealizer。不能宣称此项覆盖所有 SDK 调用；不得悄悄为 replay 加投影或在线 jump 门槛而改变回放语义。HOME/path 的准入也单独保留。

### 4.2 反例与验收

- 以当前 home 为 current/previous，J1 目标增加 `pi/2`，当前 joint 路径接受；新共享 gate 应拒绝。这是目标差，不是实测单周期转角。
- 已核查的 hard-limit 内碰撞目标（rad）为：

  ```python
  [1.1665669814126147, 0.1618457586440516, -1.0397002598715375,
   0.01076161029776243, -3.6491049933037516, 0.10155226046858923,
   -0.4494161931238594]
  ```

  先用修正后的原生模型确认碰撞，再断言被碰撞 gate 拒绝；避免仅因 jump 先拒就误称测到了碰撞判据。旧反例含 base-link6/base-link7 碰撞，不依赖手 mount 分歧。
- 覆盖 π 跨支绝对差、joint 无 IKResult 的拒绝、frozen 手姿切换、非法目标不进 SDK。在 T01 修正后的同一几何与状态下，原先接受且通过新增 gate 的 joint/eef/frozen 命令保持数值；新增 gate 拒绝旧路径曾放行的目标是预期行为变更，分别验收，不要求新旧接受集合相同。
- 非法 hand 不得先发送 arm；直接 driver 和停止路径仍拒非法目标；partial dispatch、late ACCEPTED、deadline/epoch 既有 oracle 不变。

## 5. T03：标定激励检查与 PnP 失败处理

`calibration/camera/solver.py::calibrate_and_select` 增加纯 `check_hand_eye_excitation`，不只在交互 session 里检查，以覆盖离线调用者。

最小规则：至少三个不同 pose，至少两次有足够转角、轴不近似平行的相对旋转。使用同一固定坐标系的 relative rotations，例如 `R_j @ R_i.T`；样本少，可枚举 i<j，避免选参考 pose 隐藏激励。将轴归一化，以 `abs(dot(a,b))` 判断同轴/反向同轴。**不要求三轴满秩**。

初始采样质量参数可设 `min_relative_rotation_deg=5.0`、`min_axis_separation_deg=15.0`，在本次标定配置中显式表达；它们是待实验确认的采样质量阈值，不是理论最优值或安全认证。先过滤不足转角，再要求存在轴夹角满足阈值的一对相对运动。保持 min_samples、刚体合法性与闭环 position/rotation RMS 这几类不同检查。

失败时告诉操作者需要增加另一方向的转动；保留原始 samples，不自动删帧或移动机器人。现有标定结果 metadata/log 记录样本数、有效相对运动数、转角/轴角摘要及阈值，不新建标定管理系统。

`detect_aruco_pose` 检查 solvePnP bool、rv/tv 非空、可规整为 `(3,)` 且有限；无解返回 None，不补零。预览 overlay 已查 ok，不重复整改；正常检测数值和 marker 选择保持，不吞非预期配置/相机异常。

验收：10 个完全相同 pose（旧代码可得到 TSAI 单位变换、零 RMS）被拒；纯平移、同轴转动和不足幅度被拒；充分两轴的已知合成变换可恢复，整体坐标旋转和轴符号不改变结论；非退化但 RMS 超标仍拒。PnP `(False,None,None)`、非有限解、正常解都有明确结果。沿用取消前后不发布标定的测试，只写临时文件。

## 6. T04：独立记录 attempt、Raw 发布与最终 session 结果

### 6.1 产物与语义

在已发布 Raw **之外**保存 `session_result.json` 和 `attempts/<attempt_id>.json`；复用现有 session/output 目录、`atomic_json_dump` 和一个很小的记录 helper。允许这些管理记录由未结束更新为终态，Raw 仍不可变。不建数据库、事件总线、WAL 或第二个 writer 服务。

| 记录 | 最小内容 |
| --- | --- |
| session | session_id、policy/replay、开始/结束 UTC、state、outcome/reason、shutdown_clean、attempt_ids、artifact_errors |
| attempt | attempt_id/session_id、run_id、entered_running、开始/结束 UTC 与本进程 monotonic ns、termination 主因/详情、recording_status、Raw/staging 路径（可空）、实际行数、finalization_errors |
| replay | 分开 trajectory_status/reason 与 session_status/reason；原 NPZ termination_reason 保留轨迹阶段含义 |

`recording_status` 只需 `not_requested/pending/published/empty/discarded/failed` 等当前分支用到的值，不再套一层通用 outcome 模型。任务成功/失败是独立实验标签，不从 SDK、计数或 Raw 是否存在推断。

### 6.2 生命周期

1. session 开始记录在连接设备前写入；无法写出就不启动实验。Replay 由 session 先完成现有 output missing-or-empty 检查与轨迹 preflight，再初始化记录，不能在 CLI 提前写文件导致自身目录被拒；不放宽非空输出保护，也不覆盖已有实验。
2. 经前置检查接纳的 start 请求获得 attempt_id。授予 RUNNING **之前**写 `state=incomplete`，`run_id=null`、`entered_running=null`。JSON 与 recorder START 等阻塞 I/O 必须位于最后一次起始 observation/home pose 检查之前；它们完成后再读取新 row、复核 freshness/pose/epoch，最后 grant RUNNING，不能使用磁盘阻塞前的旧观测授予权限。授予后只更新内存，终结时补写；不在活动 tick 或 motion_lock 内写 JSON。普通被拒按键不自动计入 rollout 次数。
3. start 随后取消/失败，正常收尾可写 `entered_running=false`；进程崩溃留下 incomplete 时是否运行过是 **unknown**，不能由 null 推断未运行。
4. 结束顺序保持：撤权 → 尝试 stop → 处理 recorder finalize → 保存可用 query sidecar → 写 attempt 终态。stop/保存各自异常都保留，首个控制主因不被清理异常覆盖。每个 attempt 幂等结束一次。
5. 只有 `save_episode()` 返回实际路径才写 published；零帧写 empty；writer 未完成写 pending/failed 和已知 staging。不能抢 writer 的句柄，也不能猜 Raw 成功。
6. session 终态在 shutdown 与评估产物写入尝试后保存；return-home、shutdown、evaluation-save 失败要进入 session 结果。记录初始化后，资源构造或评估写入异常也必须经过最终记录尝试；外层仅清理实际已取得的资源，不让新增记录制造生命周期缺口。Replay trajectory 已完成但 session 故障时，两者同时为真，不能相互覆盖。
7. 复用现有 close→replace 的原子 JSON 可见性。终态写入失败必须向调用者/CLI 报告、保留 incomplete、阻止继续新 attempt；不能宣称保证断电或永久存储故障后仍有完整记录，不为此扩建持久化协议。

`num_episodes` 保持现有“已进入 RUNNING 的运行结束次数”上限；prepared/start_cancelled 记录保留，但未授予 RUNNING 时不消耗该上限。不能把所有 attempt JSON 终态都加计数，也不能改成只计成功 Raw 而自动无限补跑。含混的 `completed` 可局部改为 `runs_finished`，分析区分接纳、实际运行、结束、Raw 发布和任务标签。CLI 已区分 USER_QUIT 与完整 replay；继续保留。

适用入口至少覆盖记录式 policy evaluation 与 replay；直接公开 API 无输出目录时要明确不产持久化结果，不用假路径。CLI 已有 session_dir，优先透传而非新增配置体系。新记录自身错误不能遮住原异常。

验收：正常、零帧 timeout、start_cancelled、operator quit、writer/stop/return-home/shutdown/evaluation-write 失败；start 写入失败不授予权限；final 写入失败保留 incomplete 并报错。Replay 空目录可正常初始化记录、已有非空目录仍拒绝；记录初始化后的构造异常能得到最终结果尝试。每次有唯一 id，Raw 可空，轨迹完成与最终 FAULT 均能查到。无需模拟所有操作系统灾难场景。

## 7. T05：新路径导出与最小缓存身份

### 7.1 选定身份语义

`data_revision = uuid.uuid4().hex` 表示**一次成功导出的不可变 canonical 缓存身份**。在 staging 中生成一次，同时写 Zarr root attrs 与 `export_report.json`；仅成功发布后可作为训练输入身份。

- 同 Raw/配方重新导出也产生新 revision；精确续训继续使用原缓存。完整未修改缓存的复制可以保留身份。
- UUID 不是内容哈希，不能检测用户手工修改数组；path、mtime 或 manifest 也不能冒充内容验证。
- 不把当前 mount/桌面变成历史 ABI，不在每个 checkpoint 扫描数据。旧 unknown 不反填虚构身份。

### 7.2 最小修改

1. 删除 `export_raw_to_zarr` 的 overwrite、CLI `--overwrite`、`_require_replaceable_cache` 与删除旧 target 的分支；同步真实调用者/帮助，不保留死 alias。
2. 拒绝占用 target，包括 symlink；保留 source/target 包含、嵌套 Zarr 和路径保护。
3. 同父目录 staging 完成全体转换、计数、metadata/report 后，用既有 `atomic_publish` 发布。失败保留 staging，旧缓存与 Raw 不动。沿用单导出 owner 假设，不声称现有实现具有跨进程 race-free no-replace 保证，也不新增多写者协调服务。
4. 扩展现有 export_report：revision、创建 UTC、有序 episode id/来源/行数/ends、accepted/excluded/rejected、完整 resolved processing、源码 SHA/dirty 状态。代码信息未知就写 unknown；相对 episode id 的解析不得依赖目录枚举随机顺序。用同一份成功转换 episode 的有序列表写 root attrs 的 `episode_ids` 和 report，与 `meta/episode_ends` 一一对应；导出前 rejected/excluded Raw 不进入此列表。不新增字符串数组或元数据服务。
5. 尽量不改 Policy 既有 attrs→ReplayBuffer→dataset→checkpoint→`validate_data_identity` 路径，只补端到端覆盖和确实缺少的连接。已知 revision 改变/丢失拒绝 exact resume；旧 unknown 保持警告。基于权重启动新实验走新实验路径，不降低 resume 标准。

验收：两次导出身份不同、attrs/report 一致；占用目标和 dangling symlink 被拒；中途失败无正式新缓存；已知 revision 改变拒绝续训；13 字段、NaN mask、row_info、episode 顺序不因身份改动而变。结构重构留到 T10。

## 8. T06：分组清单与时间质量报告

### 8.1 U07：固定训练 split，真实 trial 不靠猜

按 episode 随机划分可能拆开同一 trial，但未取得真实资产，不能声称已发生泄漏。Policy 当前 `configs/dp.yaml` 默认 `val_ratio=0.0`，默认本身没有 train/val 两侧。

选择小型、显式的 `split_manifest.json`：含 `data_revision`、按导出顺序的 episode_ids、每 episode 的已确认 trial_id、明确 train/val ids、显式 exclusions、划分 seed/group_unit。来源是经确认的 trial 清单；不从 pause、HOME、相邻目录或新 episode 名推断物理重置。标注放外部清单，不回写已发布 Raw。

Real 按 T05 提供有序 episode id/ends；Policy 直接消费现有 ReplayBuffer 的 attrs 与 meta，仅启用 manifest 时要求 `episode_ids` 存在、唯一、数量与 ends 对齐且顺序与清单相同，不为旧路径新增 gate，也不回读 export_report。Policy 在现有 Dataset 参数增加可选 `dataset.split_manifest`，透传到 BaseDataset 并构造现有 mask；不复制 Dataset/sampler。manifest 是划分权威，有 manifest 时不再用 val_ratio 重抽；如保留 max_train_episodes，仍仅缩小 train，并记录最终子集。生成候选可使用已依赖的 `GroupShuffleSplit`，真正训练消费固定清单。

拒绝 revision/episode 顺序不匹配、未知/重复/遗漏 ids、train/val 重叠，以及同 trial 跨两侧；排除必须显式。manifest exclusions 只表示已进入 canonical、但本实验不使用的 episode，与导出前排除 Raw 的清单分开。train、val、excluded 三侧互斥且完整覆盖当前 canonical episodes；train_mask 和 val_mask 各自从清单构造，**不能沿用 train_mask = ~val_mask**，否则排除项会混入训练。记录实际 train/val/trial 数及有效窗口数。全部数据仅训练时也明确该实验无 holdout，不制造虚假的验证集。

对**启用了新 manifest 的实验**，要求清单与缓存均有非空、已知且相等的 `data_revision`；两边 None/unknown 不算匹配。旧 unknown 缓存继续原有无 manifest 路径，或从 Raw 导出到新的带 revision 缓存后启用，不回填虚构身份。

从本次实际读取并验证的同一份规范化 manifest 构造 mask 与小文件 SHA-256 摘要，将摘要和实际 train/val mask/子集信息写入既有 `data_recipe`，复用 strict resume 比较。完整规范化内容也在训练启动时保存一次到该实验产物，或纳入既有 config/data_recipe，保留 trial/exclusion 审计信息；不能只有可变外部路径与 hash。复用现有配置/产物 owner，不另建存储服务，不重复读取可能已变化的源文件，也不扩大 source snapshot 的全仓 JSON 白名单。

没有 manifest 的旧路径保持现有 `cfg.dataset` 和 `data_recipe` 合同结构。`build_resume_contract` 会比较两者：**不要向旧默认 YAML 或合同无条件加入 `split_manifest: null` 等新键**，仅保持 recipe 不变并不够。可选 Python 参数可以存在；新实验显式配置即可。切换分组协议属于新实验，不给旧合同补假默认。

最新 Policy `eval.seed_manifest` 是仿真 selection/tie_break/test 的 task/seed 协议，**与此训练 episode/trial manifest 分开**；不修改该评测协议。

验收：合成同 trial 的两个 episode 永不跨侧；固定输入可重现，清单错误与 unknown revision 被拒；strict resume 检出新分组实验的 split 变化，未启用 manifest 的旧配置/合同仍可恢复；保存的清单内容、摘要和实际 mask 一致。没有真实 trial 资产时，代码与合成验收完成，实际泄漏核查和论文 group_unit 标 `NOT_VERIFIED`，不阻塞其他工作。

### 8.2 U08：质量摘要只报告，不改变 recorded_rows

在现有 export_report 增加 `quality_summary`，复用顺序分块读取并正确保留 chunk 间上一行状态；每 episode 边界重置时间比较。至少报告：

- 已知 observation timestamp 的 Δt p50/p95/max、非正间隔、unknown、相对 nominal dt 的 gap；不跨 episode/会话相减。
- 已知 color/depth frame number 的重复、倒退与缺失；unknown 不计作有效重复帧。
- 语义兼容的 host arm/hand/camera 时间产生 source age 与跨源 spread；负 age 单列异常。device time 与 host time 不能直接混减。
- 实际保存的辅助模态 finite/NaN 行数、dispatch 分布；JSON 未知值用 null，不能伪装为零。

只有一个 camera timestamp 时，不能推断 RGB-depth 曝光 skew；没有 tactile timestamp 时，不能编造触觉同步指标，明确 unavailable。

N/H、模态与 dispatch 配方的窗口数量仍由 Policy 当前 sampler 计算。候选 gap 阈值下的窗口数对照属于离线核查，不在 Real 复制训练 sampler。公共 exporter 不默认知道模型 N/H。

默认保持 recorded_rows：不删行、不补帧、不插值。若论文随后选择 gap 过滤，另立显式 data_recipe 变更并记录样本数量变化，不能夹在本次等价重构里。

验收：跨 chunk 不漏 Δt、不跨 episode，known/unknown/nonmonotonic/repeated 合成数据计数正确；导出数组及训练当前资格不变。直接复用 `iter_canonical_blocks` 已有 `episode_notes` 缺测行计数，必要新增统计在同一分块读取中完成，不为 QA 再读回 canonical 做第二遍全模态 finite 扫描。无需另建全数据缓存；分位数统计只保留必要的标量样本，图像/触觉大数组仍分块处理。

## 9. T07：最小 query 归因、联合时序与派生统计

### 9.1 U09：预测、实现与实际发送可关联

复用现有 trace，关联 attempt/run/query id、观测窗口槽位和 source timestamps、提交/worker 开始/完成/owner 接收、handoff slot、模式。action attempt 关联 query id、future action index、slot、realization/admission 拒绝原因；selected、SDK attempted、accepted 和 measured 仍分开。

记录式、已有时长边界的 policy attempt：每个已完成 query 的原始 future 保存一份，保留 dtype；控制路径只保留小 action array 快照与标量索引。按 T04 的“撤权 → stop → recorder finalize → sidecar → attempt 终态”顺序写一个 per-attempt NPZ，并由 attempt JSON/trace 引用；仅撤权不等于物理停止，不能在 stop 之前做压缩或同步磁盘 I/O。不要重复 RGB、逐 tick JSON 编码大数组、加入新线程/队列。

stale/invalid 的结果记录处理原因；非法结果只记录安全的 shape/dtype/有限性诊断。结束后才返回的旧 Future，不归入新 attempt，不为了等待它推迟 stop；仅在 session 诊断记录原 attempt/query 的退休结果，不回写已发布 Raw。进程硬崩溃可能丢内存 trace，incomplete 明确表达未知，不新增逐 tick WAL。

sidecar 写入失败进入 artifact_errors；不改写 dispatch，不把准备过但 NOT_CALLED 的目标写成 Raw attempted action。

### 9.2 U10：先测量，后优化

用一个小型离线汇总入口消费上述现有 trace，报告模型、观测构建、prefix/IK、owner SDK 阶段、slot lateness 的样本数/p50/p95/max，以及 miss/invalidate 原因。缺少阶段起止时只补局部 monotonic 标量，或标 unavailable；不能从无关事件作差，也不能相加相互包含/并行阶段的分位数冒充端到端延迟。

warmup 只说明模型路径，不能证明完整 tick 预算。prefetch d 增大也会增加 owner 前缀准备成本；无实测证据不加并行 IK、自动调 d 或改控制频率，不引入 benchmark/profiler 框架。

### 9.3 U19：删除可重算的 writer 统计

T06 已能输出缺测统计后，可删 writer 仅用于末尾 missing_tactile 统计的状态和重复扫描。若当前操作者确实依赖即时提示，可以保留简短提示并说明理由；不为此停下其他任务。保留 Raw NaN/validity、行配对、selected/final dispatch。没有实测不能声称获得某个加速。

验收：fake sync/async/RTC 中预测到发送/拒绝可定位；frozen 前缀仍归原 query；旧 Future 不污染新 run；加入记录前后控制输出、slot、异常/stop 顺序一致；分析器不编造缺失阶段或时钟数据。

## 10. T08：删除闲置机制，集中静态责任

| 项 | 最小方案 | 必须保留与验收 |
| --- | --- | --- |
| U11 | 删除 `PolicyRunner.set_execution_mode`、pending_execution 及仅服务试次间重新配置的分支/测试；同 session 固定模式，切换用现有 CLI 重启 session | 保留初始 configure_execution、warmup/预算和 sync/async/RTC。删除 setter 测试前，把仍有效的启动预算失败/模型关闭验收迁至 session 初始路径；不重写通用 worker |
| U12 | HandKinematics 按需构造；依赖/URDF/frame 失败在构造阶段抛清楚异常；删 `_ready/is_ready`、二次 import 和静态资源失败 NaN 回退 | 保留真实 joint order/frame 检查、缺测传感器 NaN；有效 FK 不变；不需手 FK 的路径不强制加载依赖 |
| U13 | `ProcessingConfig.from_runtime(runtime, *, table_plane_abcd=None)` 显式构造现有 dataclass；删任意 overrides/反射 | 保留数值规则和 resolved 配方；搜索全部生产调用者，当前导出配置数值不变 |
| U14 | session 保留连接前 preflight；优先去 CLI 同链重复解析/校验 | Runner 仍作为独立公开边界时保留轻量构造校验。它目前在 connect 后创建，不能把 session gate 移到那里；不加 trusted token，不为少一次 validate 重排生命周期 |
| U15 | EpisodeReader 生命周期内缓存字段集合和首次按需验证后的 Dataset 引用；close 后失效 | 不打开时扫描全模态，不跨文件缓存，不读全数组/hash；坏布局仍拒。旧探针 2块×8字段=16次枚举只是工作计数，不是吞吐基准 |
| U17 | `__post_init__` 负责数值；from_dict 保留 mapping、list/tuple 语法及嵌套构造，去重复转换 | `distortion_coeffs="00000"` 裸构造器会接受，原 from_dict 拒绝，必须保持拒绝；核对未知/缺字段错误类型的调用者，不新建配置解析器 |
| U18 | CollisionModel 只有实际 table/static_boxes 时创建第二套 environment geometry/data；无环境明确短路 | 配置生命周期固定，一个构造期条件足够，不做 lazy framework。self-collision、SRDF、HOME 环境检查不变；有/无环境分别验证 |

## 11. T09：让控制循环按阶段可读

先完成控制行为修复、attempt 记录和 U11 删除，再以其新行为为等价基线重构，避免同时维护新旧两套路径。

### 11.1 U20 PolicyRunner

- Query/Plan 放到 runner 前部，result/sources/frozen 使用具体类型，Plan 用关键字参数；docstring 解释 anchor/handoff/segment_end 的含义与开闭区间，不增加 runtime validator。
- `step` 抽 handoff 与 prefetch 两个完整阶段；仍是一个 owner，不按 sync/async/RTC 分成三套 runner。
- `_send` 改为 `_execute_slot`，优先抽 `_command_for_slot`；外层保留发送、撤权/stop、记录、结束和异常优先级的顺序可见，不拆为互不知情的 dispatch/record/stop 服务。
- 观测前后两次 poll、长操作后重新取时钟、漏槽清 history、旧 plan/query 失效、bootstrap 与常规 query 不同锚点、WAIT 预算不重置全部保留。

### 11.2 U21 Teleop

- 先改 controller：`compute_target(row) -> ActionRealization | None`，执行层用 epoch 建 RobotCommand；`commit_dispatched_target(realization)` 接回 previous command 和 EMA 更新。T02 拒绝原因继续可见，不新建 ResultSchema/ControllerBase。
- EMA/reference 提交沿用 **not interrupted 且 result is not None** 的原条件；不能借重构变成双 ACCEPTED，也不能在仅准备目标时提交。
- 再改 runner：抽 `_run_control_tick` 和 `_shutdown_capture_and_inputs`，按键批次 STOP/QUIT/ESTOP 优先级留在显式主循环；已有 `_handle_operator_command` 不再重复提取。
- 执行前/后两次录制预算检查不是重复，均保留；暂停后用新观测重建 reference，先撤权再阻塞保存；收尾逐项尝试并保留首异常，不能第一个 close 失败就跳过其他资源。

验收沿用 fake clock/operator/model/SDK：相同输入产生相同 slot、命令序列、部分 dispatch、Raw 行、终止主因与异常优先级。只为新增拒绝原因/记录等真实行为变更调整预期，不把测试改成跟随实现的镜像。

## 12. T10：数据流、资源与论文指标的可读性

| 项 | 目标函数/类与选定重构 | 不变量与足够验收 |
| --- | --- | --- |
| U22 | `AsyncEpisodeRecorder._write_episode` 抽初始/最终 metadata 与 rows_to_arrays helper | writer 保持唯一文件/队列 owner；可清理 owner 先于 open；open→consume→flush→metadata→close→计数→publish。metadata 失败仍排空可保存队列、失败留 staging；复用原生小 episode 和失败用例 |
| U23 | `export_raw_to_zarr` 抽 select/create store/append episode；`iter_canonical_blocks` 抽纯运动学数组计算和明确的缺测处理 | T05 行为先落实。顶层保留路径/跨 episode 一致性/report/publish，generator 保留 I/O/视频顺序消费/yield。仅在写 staging 前的候选预检阶段，已知 RawDataError 可记为 rejected 并跳过；进入转换/追加后任何错误均中止本次导出并保留 staging，包括视频中途缺帧/多帧产生的 RawDataError，不能跳过半写 episode；13字段/NaN/ends/row_info 不变；不建 CanonicalExporter 框架 |
| U24 | `build_fingertip_runtime` 改 `build_observation_kinematics`，用小 `ObservationKinematics` 表达 arm_fk、hand_fk、mount | EEF-only 不创建 hand FK，joint/rgb/cloud 无该依赖时不创建 FK；T01 几何数值修复独立；类型不承担验证/调度职责 |
| U25 | `compute_metrics` 抽 lag、必要 EEF 指标纯函数；保留 ReplayMetrics | 保留 RMSE 定义差别、有效重叠/静止轴/NaN mask、lag 正负号及 `(rmse, abs(lag), lag)` tie-break；不换相关系数/自动对齐或加指标 registry |
| U26 | 增加确实重复使用的小型 `robot.stop_required`、`reader.path` 等只读接口；ObservationRow 常用 qpos/timestamp 属性按真实重复点添加 | 无隐式 I/O、无整帧复制；stop_required 表示软件停止需求，不宣称物理静止/安全；不包装所有私有字段 |
| U27 | `run_policy_deployment` 抽模型加载/配置与 warmup helper，外层保留资源生命周期 | 外层在 warmup 前已有 worker owner，失败 finally 能 close；连接时序不变；不新增 BaseSession |
| U28 | Python 类型 `PolicyParams -> ControlParams`、必要时 `TeleopConfig -> TeleopSessionConfig`，同步实际引用 | 默认保留 YAML/CLI 的 policy 键，解释其 shared control 含义，不为风格制造外部迁移或永久双 alias；只有明确收益才另改外部键 |
| U29 | 固定常量的数量/唯一性/派生索引导入自检可删除 | 常量值与映射不变，实际 URDF order/frame 校验保留；不为几行清理新建资产测试框架/镜像测试，净收益不足可 RETAINED |
| U30 | HOME validate_path 中简单 report 判据改顺序 if；复杂几何 helper 保留 | 失败顺序/报告不变；RRT 输出校验和追加目标/重新插值后的最终 HOME candidate 校验仍分别保留，含软间隙；不把在线 jump 套到整个返航目标；收益不足可 RETAINED |

代码风格完成标准是新读者能沿入口看到观测、模型/控制变换、真正发送 owner、失败证据和指标定义。保持领域命名、边界 shape/frame/unit docstring 和必要数学符号；不是按函数行数拆类。目录/README/Ruff 已有基础，不搬全仓、不加全量类型迁移，不改变 optional imports 的按需加载边界。

## 13. T11：条件项的明确终态

| 项 | 当前结论 | 触发条件与本次决定 |
| --- | --- | --- |
| U31 Raw legacy fallback | 历史 metadata dispatch、零时间/UNKNOWN 和旧 replay 兼容仍存在，资产清单未知 | 默认保留。只有经授权只读资产清单确认论文不依赖相应布局，才删具体分支并清楚报缺字段；不搜索未授权目录，不伪造时间/ACCEPTED，不改旧 Raw |
| U32 checkpoint 兼容 | unknown revision、clip_sample、最新 LoRA dtype 的历史默认有实际路径 | 先补 T05。无仍需精确恢复的 checkpoint 清单则保留；有证据无依赖才删。新实验从权重启动与 exact resume 分开，不弱化消费者保护 |
| U33 IPC 内存序 | NumPy uint64 seqlock 未显式 acquire/release；没有 torn-read 反例 | 默认保留当前实现并如实记录实际支持平台/保证。只有明确跨平台需求或一致性反例才引入成熟 atomics/适当锁；连同 payload 发布顺序验证，不能只换 counter 就宣称 ring 正确 |
| U34 PyAV | 原失败版本/输入缺失；PyAV 17.1 当前三个 CFR 用例和原生录制/导出通过 | `NOT_REPRODUCED`，不重写/降级；保留当前复现环境说明。得到原反例再重验，不声称 av>=10 全兼容，也不虚构“已修复提交” |

完成条件项是作出有证据的决定，不是强制删除。没有资产时给出所需字段清单即可，不为未知需求新建扫描/迁移工具链。

## 14. 参考项目：只借鉴已核实的局部机制

| 参考 | 可借鉴 | 不照搬 |
| --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot/tree/d40e8709cffb93644db66e30604ef50fdec003cb) | 清楚的 Robot 公共接口、配置与阶段职责 | 不搬平台 processor/registry；record_loop 并非所有函数都短，其 `_sent_action` 返回值也不能被说成已经等同保存的实际动作 |
| [LeFranX](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | arm/hand 直接组合 | synchronize_actions 的顺序发送/TODO 不构成硬件同步保证 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | model/obs/action 边界命名；必要时参考 SharedAtomicCounter | 不整体迁入 Hydra/多进程架构；原子计数器不证明整个 ring 正确 |
| [UFactory](https://github.com/xArm-Developer/lerobot_robot_ufactory/tree/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82) | 薄 SDK、配置和单位边界 | 不照搬错误状态返回输入 action 的证据语义，不用 monkey-patch queue/task_done 的 AsyncEpisodeSaver 替换当前 writer |

算法依据只使用小范围成熟机制：[OpenCV 手眼标定](https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html)要求非平行旋转轴运动；[GroupShuffleSplit](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)可生成按组候选划分。实际实验仍使用固定、可审查的 manifest。

## 15. 验收策略与现有测试定位

先按任务读测试，确认 import/fixture 无硬件副作用，再运行覆盖当前风险的 focused tests。缺可选原生依赖就报告具体未验证范围，不升级实验环境、不把所有失败都吞成 skip。重构与行为修复分批，比较的是输入输出、时序与失败证据，不是私有 helper 调用次数。

| 目标 | 可复用的当前测试（相对 tests/） | 需要补充的真实风险 |
| --- | --- | --- |
| T01/T02 | `test_policy_eef.py`：native FK/IK、frozen FK；`test_review_remediation.py::test_absolute_hw_distance` | 两 mount 多姿态传播；joint/碰撞拒绝分开；frozen 手姿和无 IKResult 拒绝 |
| T02/T09 | `test_policy_runner.py::test_fixed_handoff_and_prefix`、`test_old_future_blocks_new_episode_reset`、`test_future_finishes_while_reading_observation`、`test_slow_observation_read_breaks_history_even_without_full_slot_skip` | 新原因贯通；改结构不改 slot、WAIT、prefix |
| SDK/异常 | `test_review_remediation.py::test_dispatch_deadline_at_last_boundary`、`test_last_sdk_late_return_keeps_accepted`；`test_policy_runner.py::test_wait_deadline_between_real_sdk_calls`、`test_late_return_cleanup_records_once` | 非法 hand 不先发 arm；保持 partial dispatch、首异常和恰好一次记录 |
| T03 | `test_review_remediation.py::test_handeye_selects_qualified_candidate`、`test_calibration_cancel_after_solve_does_not_publish` | 退化/充分激励、坐标不变性、PnP False/None；旧 mock 需提供有意义的非退化输入，不能绕新 gate |
| T04/T10 | `test_policy_recording.py::test_writer_failure_preserves_staging_with_session_error`、`test_writer_receives_frozen_termination_and_policy_trace`、`test_final_metadata_failure_drains_native_writer` | 零帧/取消/关停/evaluation 写失败的独立结果；published 只基于真实返回路径 |
| Teleop | `test_review_remediation.py::test_teleop_cancel_records_once_before_propagation`、`test_teleop_stop_failure_retains_first_reason`；`test_policy_recording.py::test_teleop_run_finally_saves_stop_failure` | 原条件下 previous command/EMA 一致，收尾继续所有资源 |
| U11/U27 | `test_policy_inference.py::test_worker_serial_lifecycle_and_close_during_predict`；现有 mode warmup 相关测试 | 删除 setter 专属期待，把有效 warmup 拒绝/关闭检查迁到初始 session；连接前失败 |
| T05/T06 | 当前仓库外审查探针不冒充 repo 测试 | 用小型临时 Raw/Zarr 和 Policy mask/resume 函数，覆盖身份、发布、split 与跨 chunk QA；不跑训练 |
| 指标/PyAV | `test_review_remediation.py::test_replay_rmse_names_preserve_distinct_statistics`、`test_native_cfr_random_matches_sequential`、`test_encoder_explicit_frame_clock` | 仅针对实际指标抽取与当前编码路径；不做全版本矩阵 |

改过 Python 后，执行现有低成本检查；对实际修改的 Python 文件分别运行 `ruff format --check` 和 `ruff check` 并传入具体文件路径，已有无关失败单独记录，不顺手重排全仓：

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

Policy 使用其自己的既有定向/config-only 检查；先核实当前入口和所需依赖。仅文档变更检查路径、内容和 diff，不为任务书运行硬件/训练或全套测试。

本次审查的已执行证据仅作为基线：4 组新增探针、6 组核心探针复跑、PyAV 3 个 CFR 用例通过；第三轮静态简化探针复用同 Real SHA。**这些不是本任务实施后通过的证明。**

## 16. 最终交付与完成标准

1. 每个 U 项都有最终状态、对应文件/行为、实际检查及限制；T11 有触发条件与保留理由。无法取得真实资产时明确哪些结论只能到代码/合成层。
2. 明确缺陷有独立反例被修复；正常路径与时序不变量有足够定向回归。不要仅用 compile 证明控制/资源收尾等价，也不要为低风险命名和常量镜像实现写测试。
3. 生产调用图不留死 setter、旧 alias、重复新旧路径；无新 worker、无必要框架或全局 schema/contract 系统。
4. 新实验的缓存身份、split、实际尝试分母、Raw 发布与技术故障可解释；没有把缺测、未知和未调用伪装成有效数据。
5. 当前 geometry/mount、碰撞、停止延迟、真实传感器时间差及 GPU/owner 联合耗时的真机状态分别说明，不以离线通过冒充实物验收。
6. 两仓库各给出 diff 摘要、删除/保留的机制、实际命令结果和阻断；不提交无关文件、临时 Raw/视频、环境或私有数据。稳定入口变化才同步 README，不把台账复制到 AGENTS/CLAUDE。
7. 如记录执行进展，在本文末尾追加简短批次表即可；不要为本次整改新增一组长期根目录报告。完成必要验证后停止扩张任务。

可直接给 Codex CLI 的提示词：

```text
请在 haoyangzhanglab/dexmani_real 中执行根目录 CODEX_RESEARCH_REFINEMENT_TASK_20261007.md 的第二轮审校版，完成其中仍然有效的整改。必要的训练侧修改同步到 haoyangzhanglab/dexmani_policy。目标是让项目适合 RAL 论文实验：正确、安全、简洁、高效、可追溯、容易理解。

先读取两仓库适用的 AGENTS.md 和完整任务书，检查当前 HEAD、工作区已有修改及相关实现。任务书的 SHA 只用于定位证据，不是 reset/checkout 目标。保留用户已有修改；已经修复的条目以代码或测试证据标记 ALREADY_FIXED。若当前文件缺少“第二轮实施审校通过”标记，从 Real 的 GitHub main 读取最新版并核对差异，不覆盖用户本地改动。

按 T01–T11 的依赖顺序持续执行，覆盖 U01–U34。先修正确性与数据生命周期，再删除多余机制，最后做等价重构。不要停在计划、再次审查或仅完成 P0；常规实现选择自行决定，完成所有不受阻的独立项。条件项按证据选择保留或处理，U34 未复现时如实记录。遇到权限、依赖、真实资产或实物参数阻断，只暂停相关部分并说明所需条件。

严格遵守任务书的控制、Raw、时间、缺测、normalizer、resume 和资源所有权不变量。尤其落实 replay 输出预检后再初始化记录、撤权并尝试 stop 后再写 sidecar、新 manifest 的已知 revision 与完整内容留存、旧 cfg.dataset/data_recipe 的恢复兼容，以及导出追加开始后出错即整体中止并保留 staging。结构重构以行为修复后的版本为基线，不把新增合理拒绝误判为等价性失败。

优先删除无用路径、复用现有类型和成熟社区机制；不增加通用框架、重复校验层、后台服务或迁移系统。helper 名称可按当前代码合理调整，行为和验收要求必须满足。不要为完成清单强行拆函数、删历史兼容或改变实验超参数。

本次授权修改代码、必要文档和执行纯离线验证。不得连接或驱动设备、执行 HOME/replay/rollout/采集/真实标定，不运行完整训练、DDP 或长时间评测，不修改已发布 Raw、checkpoint 或既有实验，不升级实际实验环境来凑测试。默认保留可审查 diff，不自动 commit、push 或合并实现改动。

按实际风险运行定向测试与低成本检查；测试验证输入输出、时序、数据和失败证据，不镜像私有实现。缺依赖或真机条件时明确 NOT_VERIFIED，不把未运行、skip 或旧审查探针当作本次通过。完成足够验证后停止扩张范围。

最终用中文分别交付两仓库的修改摘要、删除与保留机制、U01–U34 逐项状态及证据、实际执行的验证命令和结果、阻断与真机/资产未验证范围。简短进度表可追加在原任务书末尾，不另建一组长期报告。
```


## 17. 本次实施结果（2026-10-07）

已按 T01–T11 完成全部不受阻的代码项与条件决策，交付未提交 diff；证据 SHA 保留。实施前读取两仓库 AGENTS 与完整本文；本地已有第二轮标记，无需用远端文件替换。两个工作树开始时均干净，结束 HEAD 未变：Real `553daa60e571def4eea49cb2e94d23608752badf`，Policy `72dd34295ba101e8fe19b85e457edde50092002b`。未 reset/checkout、commit、push、升级环境或访问真实设备/实验资产。

下表 `DONE` 表示代码实施完成；`PASS` 仅指本次纯离线证据。简称 RR 为 Real `tests/test_research_refinement.py`，PR 为 `tests/test_policy_runner.py`，WR 为 `tests/test_policy_recording.py`，RV 为 `tests/test_review_remediation.py`，PS 为 Policy `tests/test_research_split.py`。低风险命名/去重使用调用图与静态检查，不为常量和 helper 复制实现测试。

| 项 | 最终状态 | 落点、证据与边界 |
| --- | --- | --- |
| U01 | DONE / PASS；实物 NOT_VERIFIED | `planning/kinematics/urdf.py` 将当前 mount 传入 Pinocchio model/self/environment geometry 和临时 MPLib URDF；所有生产构造者显式传参。RR `test_current_mount_reaches_native_geometry`：两个非零平移/旋转、两组非零姿态、五指 FK 1e-6 m 对齐，临时文件删除后仍可 FK；坏拓扑拒绝。 |
| U02 | DONE / PASS | `robot/action.py`、`kinematics/ik.py` 共享绝对目标差/限位准入；joint 只创建 CollisionModel，EEF 保留候选选择与其配置阈值，frozen 使用该命令手姿。RR jump/独立碰撞/自定义 EEF jump，RV absolute_hw_distance，PR 无 IKResult 拒绝与 frozen/RTC 回归。 |
| U03 | DONE / PASS；阈值实测 NOT_VERIFIED | `camera/solver.py` 两轴相对旋转激励，配置与 summary metadata 显式保留。RR 重复/平移/同轴/小幅拒绝及两轴已知解、坐标旋转/轴符号不变；RV RMS 与取消不发布。 |
| U04 | DONE / PASS；overlay ALREADY_FIXED | 检测路径拒绝 PnP False/None/非有限/错误形状；RR `test_pnp_failure_is_missing` 含正常解。原 `solver.py` preview 已按 ok 绘制，不重复改写。 |
| U05 | DONE / PASS | `recording/results.py`、policy/replay session 与 runner：连接前初始化、RUNNING 前 incomplete、最终新观测后 grant；撤权→stop→Raw finalize→NPZ→attempt JSON；session 在 shutdown/evaluation 后收尾。RR/PR/WR 覆盖取消、零帧、写入/stop/构造/HOME/shutdown/evaluation 故障、首异常、不覆盖 Raw；新 START 失败不沿用上次计数/路径。原 recorder START 后重读观测为 ALREADY_FIXED，保留并扩展验证。 |
| U06 | DONE / PASS；消费者传递 ALREADY_FIXED | `dataset/export.py` 只接受新路径，移除 overwrite；原子发布 revision、UTC、代码 SHA/dirty、完整处理/导出配置与有序 episode ids/ends。RR 两次 UUID、dangling symlink、13 字段/NaN/row_info、失败 staging，以及 Policy revision 拒绝恢复；既有 attrs→dataset→cfg/checkpoint→resume 无需重写。 |
| U07 | DONE / PASS；真实分组 NOT_VERIFIED | Policy `datasets/split.py`、`base_dataset.py` 一次读取已知 revision 的完整清单；独立 train/val/exclusion mask、trial 不跨侧、实际子集/窗口/holdout、完整规范化内容及 hash 进入 data_recipe。PS 与旧窗口/resume 定向检查通过；默认 YAML、旧 cfg.dataset/data_recipe 未加 null 键。 |
| U08 | DONE / PASS；实测时钟 NOT_VERIFIED | `dataset/quality.py` 同一分块流统计 Δt/gap/frame number/source age/spread/dispatch，复用缺测 notes；跨 chunk 延续、episode 重置。RR `test_quality_chunk_boundaries_and_clock_domains`；device camera 不混减 host，RGB-depth 曝光差/触觉同步为 null，数组不筛行。 |
| U09 | DONE / PASS | runner query/run/attempt 归因、窗口来源、worker/owner 时间、future index、selected/dispatch；每次安全有效形状的已完成 future 保留原 dtype 一份。PR stop→Raw→sidecar、float16、旧 Future 不污染新 run；退休查询记 session 诊断，不等待它推迟 stop、不回写 Raw。 |
| U10 | DONE / PASS；真实延迟 NOT_VERIFIED | `deployment/timing.py` 与 `examples/summarize_policy_trace.py` 汇总真实阶段样本与失效原因；RR 缺阶段 count=0、分位数 null。不拼接嵌套/并行分位数、不自动调 d/频率。 |
| U11 | DONE / PASS | 删除运行中/试次间模式 setter、pending_execution 与专属分支；保留初始配置及 sync/async/RTC。初始加载/配置/warmup/预算失败清理迁入 RR；worker 串行生命周期与关闭测试通过。 |
| U12 | DONE / PASS | `HandKinematics` 构造期 fail-fast，移除 _ready/is_ready、二次 import 与静态故障 NaN fallback；调用者同步。原生 FK 与按需构造测试通过，真实测量缺失 NaN 保留。 |
| U13 | DONE / PASS | `ProcessingConfig.from_runtime(runtime, *, table_plane_abcd=None)` 显式构造，删除反射/任意 overrides；全部生产调用者核查，原生导出覆盖。 |
| U14 | DONE / PASS | 删除 `examples/run_policy.py` 同链重复静态校验；session 连接前 preflight 与独立 Runner 校验保留，初始失败不连接的测试通过。 |
| U15 | DONE / PASS | `EpisodeReader` 生命周期缓存字段集合和按需校验后的 Dataset 引用；close 失效。RR 坏布局/关闭与原生分块导出；不跨文件或预读所有数组。 |
| U16 | DONE / PASS | XHand driver 复用 `check_hand_target`；Robot 双侧预检和各 SDK 前动态复核保留。RR 非法手目标在 arm SDK 前拒绝，RV/PR partial/late/deadline 回归。 |
| U17 | DONE / PASS | `CameraIntrinsics.from_dict` 保留序列语法，数值归 __post_init__；RGBD 去重复转换。RR 字符串 distortion 拒绝，合法 roundtrip 保持。 |
| U18 | DONE / PASS | `CollisionModel` 无 table/boxes 时不构造第二套环境 geometry；有环境沿用同份修正 XML。RR 有/无环境及原生 HOME 合成桌面检查。 |
| U19 | DONE / PASS | 移除 writer 的 missing_tactile 状态与重复全数组扫描；统计由 U08 提供。WR/导出仍保存真实 NaN、配对和 dispatch；未宣称实测加速。 |
| U20 | DONE / PASS | Query/Plan 置前并明确类型/槽位；提取 handoff/prefetch/command_for_slot，执行 owner 保留发送和失败收尾顺序。PR sync/async/RTC、两次 poll、漏槽、WAIT、prefix、部分发送 oracle 通过。 |
| U21 | DONE / PASS | Teleop `compute_target→ActionRealization`、`commit_dispatched_target`，执行层建 RobotCommand；控制 tick/收尾提取。提交仍按未 interrupted 且 result 非空，RV/WR 取消、一次记录、首异常与 stop 失败通过。 |
| U22 | DONE / PASS | recorder 提取初始/最终 metadata 和 rows_to_arrays；唯一 FIFO writer 和 open/drain/flush/metadata/close/publish 顺序保留。WR 原生失败 staging、metadata 失败排空与冻结 metadata 测试通过。 |
| U23 | DONE / PASS；追加失败中止 ALREADY_FIXED | export 提取 select/create/append，processing 提取纯运动学和缺测；原有转换/追加错误中止行为保留。RR 注入第二块 RawDataError：整个导出失败、无正式缓存、保留 staging；13 字段不变。 |
| U24 | DONE / PASS；按需 EEF 行为 ALREADY_FIXED | `ObservationKinematics` 与 `build_observation_kinematics` 替换 tuple/旧名；RR EEF-only 仅 arm FK，joint/rgb/cloud 无 FK，指尖才构造 hand FK；依赖选择不扩张。 |
| U25 | DONE / PASS | `replay/evaluation.py::tracking_lags` 纯函数；已有清晰 EEF 指标块保留。RR ±lag/NaN/静止轴和 RV 两种 RMSE 定义通过；tie-break 与有效重叠保持。 |
| U26 | DONE / PASS | 公共只读 `robot.stop_required`、`reader.path` 替换实际重复私有访问；不加整套 getter。stop_required 仅表示软件停止需求，既有 stop/资源收尾测试通过。 |
| U27 | DONE / PASS | session 抽加载/配置和 warmup，外层先取得 worker owner；RR 六种初始错误路径关闭模型且不 connect；配置失败清理异常不覆盖主因。 |
| U28 | DONE / PASS；TeleopConfig RETAINED | Python `PolicyParams→ControlParams`，保留 YAML policy 外部键、无旧 alias；全调用图/compile/Ruff 通过。TeleopConfig 已明确当前职责，额外改名无收益。 |
| U29 | DONE / PASS | 移除固定关节/常量数量及唯一性导入自检；值和映射保持，实际 URDF active order/frame 校验仍在。原生模型检查通过。 |
| U30 | DONE / PASS | HOME 简单 report 判据按原序内联；复杂几何 helper 与 RRT/final candidate 两次不同阶段检查保留。RR guard 顺序、RV soft escape、原生合成桌面路径测试通过。 |
| U31 | RETAINED；真实资产 NOT_VERIFIED | 保留 reader/replay 的 legacy dispatch、零时间和 UNKNOWN fallback。缺已确认论文资产依赖清单，不删除历史证据路径；未扫描私有目录或改写 Raw。 |
| U32 | RETAINED / 合成恢复 PASS；真实 checkpoint NOT_VERIFIED | 保留 unknown revision 警告、clip_sample 与 LoRA dtype 历史默认；U06/U07 不弱化已知 revision/strict resume。缺需 exact resume 的真实 checkpoint 清单。 |
| U33 | RETAINED；跨平台内存序 NOT_VERIFIED | 保留现有 seqlock；ring/camera_ring 文档明确当前 Linux x86_64、NumPy uint64 未提供 acquire/release 保证。无 torn-read 反例或新平台要求，不引入 atomics/锁框架，不声称可移植证明。 |
| U34 | NOT_REPRODUCED；原反例 NOT_VERIFIED | 现有 PyAV 17.1.0 的三个 CFR 参数用例、显式 encoder clock、原生 Raw writer/export 本次重新通过；不降级/重写，不声称所有 av>=10 兼容。 |

删除的机制包括 overwrite、模式 setter/pending 状态、静态 FK NaN 回退、重复校验/扫描、常量自检、无障碍物时的冗余 geometry 与短小 HOME 判据 wrapper。保留一个硬件 owner、一个串行 worker、一个 FIFO writer；没有新增后台服务、registry、迁移或全量数据哈希系统。Policy 保留现有 sampler/normalizer/resume 与 eval.seed_manifest；没有改变实验默认超参数。

本次最终有效验证命令与结果（在对应仓库执行；不以任务书里的旧探针当作实施证据）：

| 命令 | 实际结果 |
| --- | --- |
| `MPLCONFIGDIR=/tmp/dexmani-refinement-mpl /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -q tests` | 后续清理复跑 **183 passed，13.51 s**，无 skip；包含原生 Pinocchio/MPLib、临时 Raw/Zarr/PyAV 与 fake 时钟/SDK。 |
| `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m compileall -q dexmani_real examples` | PASS。 |
| Real `ruff check` 和 `ruff format --check`，参数为 `git diff --name-only` 与 `git ls-files --others --exclude-standard` 中全部 48 个 `.py` 文件 | 后续清理复跑 PASS。使用既有 real_robot/bin/ruff，不格式化无关文件。 |
| `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dexmani-refinement-mpl /home/zhanghaoyang/miniconda3/envs/policy/bin/python -m pytest -q -p no:cacheprovider tests/test_research_split.py tests/test_policy_windows.py tests/test_infra_resume.py -k 'research_split or policy_windows or rng_device_and_revision'` | 后续清理复跑 **26 passed，5 deselected，3.36 s**；未选的 5 项不计为通过，不运行训练/DDP。 |
| `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dexmani-refinement-mpl /home/zhanghaoyang/miniconda3/envs/policy/bin/python dexmani_policy/smoke_test.py --config-only dp dp3 dqrise r3d multitask_dit` | 五个配置全部 PASS；仅配置解析，不加载真实数据/执行 GPU 训练。 |
| Policy `ruff check --no-cache` 与 `ruff format --no-cache --check dexmani_policy/datasets/base_dataset.py dexmani_policy/datasets/split.py tests/test_research_split.py`（check 使用同三文件） | PASS；使用既有 real_robot/bin/ruff 和 Policy 自身规则。只读 cache 权限由 --no-cache 处理，未升级环境。 |
| 两仓库 `git diff --check` | PASS。 |

验证平台 Linux x86_64；既有环境 NumPy 1.26.4、SciPy 1.15.3、Pinocchio 2.7.0、MPLib 0.2.1、Zarr 2.18.3、PyAV 17.1.0。早期迭代中的格式/测试失败已修复并复跑；没有把未运行、deselected 或历史探针计作本次 PASS。

**实物与资产边界**：当前配置 mount `(-0.015,0,0)m` 进入完整模型后，当前 HOME 姿态报 `flange_link_0/right_hand_thumb_bend_link_0` 自碰撞。代码拒绝此配置下的该目标；没有修改 mount、HOME、SRDF 或阈值迎合测试。接受路径的原生回归明确使用 nominal 合成 mount，不能用其通过证明实物可运行。重新启用前须量测/确认实际 mount 方向尺寸与碰撞接触关系；实际 geometry、连续路径/环境避碰、停止延迟、传感器时间差和 GPU/owner 联合时序均 **NOT_VERIFIED**。

真实 trial 清单与论文 group_unit、缺测/gap 对真实样本规模的影响、旧 Raw/checkpoint 的依赖覆盖均 **NOT_VERIFIED**。条件决策后续需要：Raw 路径/布局/字段与论文使用关系、canonical revision/有序 episode id/已确认 trial_id/group_unit、checkpoint 对应 resolved cfg/data_recipe/data_identity 及是否要求 exact resume。U34 需要原失败 PyAV 版本、输入/编码参数和错误输出。没有代码级权限或依赖阻断；上述条件只阻断真实资产结论或真机验收，不影响本次已完成的离线实施。

后续清理：删除旧 ActionRealizer/CollisionModel 属性探测、future 的重复转换与有限性检查、手 FK 的指尖名称副本和重复复制、manifest 规范化中的不变字段重复赋值；测试替身同步当前接口。修正执行文档中已删除的模式 setter、README 的 writer 缺测摘要描述、导出 help 和任务书待实施状态；Policy 清单细节移入 `docs/rtc.md`，README 训练段只保留入口。U31/U32 恢复用途、既有数据目录保护及有效控制算法继续保留，未扩大到真实资产迁移。

## 18. 验收收尾（2026-10-07）

本轮重新读取两仓库 AGENTS 和完整任务书。工作区开始时均干净，HEAD 与本轮 fact-check 基线一致：Real `3eae2fac25b5a538851546916be1a08e0792e416`、Policy `7a27b35e48123c04a5ad04a071b9283cf652e80d`；未 reset/checkout、commit 或 push。

**U09：FIXED / 离线 PASS。** 本轮反例补充了第 17 节未覆盖的非有限预测：修复前 float16/float64 × NaN/+Inf/−Inf 六种输出均正确拒绝且无 SDK 下发，但 NPZ 缺失 query。`runner.py::_poll_model` 现将可归档与可执行条件分开：当前未终结 attempt 的正确形状浮点输出先保留 owned copy（原 dtype、全部数值），执行仍要求有限。未改变 freshness、授权、失效或异常优先级，也未增加序列化框架、worker 或 I/O 时机。原有撤权→尝试 stop→recorder finalize→query sidecar→attempt 终态顺序保留；非有限值只进入 NPZ，trace 仍可严格 JSON 序列化。

复用 `tests/test_policy_runner.py` fixture，新增产物回归通过 `np.load(..., allow_pickle=False)` 检查 query、dtype、(7,19) 形状、[1,3] 异常位置、Inf 符号和其余有限值，并在 stop 时改变 producer 数组证明快照独立。另覆盖普通浮点执行、晚返回旧 Future 不污染新 attempt/不改旧产物、不等待旧 Future 才 stop、错误形状/整数/object 只留元信息；既有收尾测试补充 sidecar 失败与 stop/writer/sidecar 同时失败，首个异常和 termination reason 保持。

**Policy：ALREADY_FIXED / 指定离线复验 PASS，无代码改动。** 当前 `datasets/split.py`、`base_dataset.py`、resume 及本轮测试确认独立 train/val mask、exclusions、已知 revision、有序 episode ids、完整规范化清单和 SHA-256 留存；无 manifest 时旧 `cfg.dataset`/`data_recipe` 不增加新键，已有恢复契约保持。没有修改 normalizer、实验超参数或历史兼容分支。

使用既有解释器，两个环境都实际导入 Torch 2.4.1+cu124、TorchVision 0.19.1+cu124、OmegaConf 2.3.1、Hydra 1.3.5、Pinocchio 2.7.0、PyAV 17.1.0、Zarr 2.18.3。通过模块 `__file__` 断言 runner、BaseDataset、resume 来自本次两个工作树。pyrealsense2 原生扩展可导入且 `__version__=2.54.2`；最初 metadata 探针的 `PackageNotFoundError` 仅表示缺 distribution 元数据，随后单独核对原生模块路径/类型，非缺依赖。未安装或升级环境。

下列简称展开为本轮实际使用的绝对路径；测试在对应仓库执行，均设置 `PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/dexmani-acceptance-mpl`：

```bash
REAL_PY=/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python
POLICY_PY=/home/zhanghaoyang/miniconda3/envs/policy/bin/python
RUFF=/home/zhanghaoyang/miniconda3/envs/real_robot/bin/ruff
```

| 本轮实际命令 | 结果 |
| --- | --- |
| 修复前 `$REAL_PY -m pytest -q -p no:cacheprovider tests/test_policy_runner.py -k query_evidence_preserves_floating_output --tb=short` | **6 failed、2 passed、78 deselected，0.34 s**；六个失败均为拒绝后 NPZ 无 query，明确复现遗漏。 |
| 修复后 `$REAL_PY -m pytest -q -p no:cacheprovider tests/test_policy_runner.py tests/test_policy_recording.py` | **99 passed，2.09 s**。 |
| `$REAL_PY -m pytest -q -p no:cacheprovider tests` | **198 passed，13.18 s，无 skip**；包括此前审查中受 pyrealsense2 和 Policy 导入阻断的测试，以及当前 mount 的原生碰撞拒绝回归。 |
| `$POLICY_PY -m pytest -q -p no:cacheprovider tests/test_research_split.py tests/test_policy_windows.py tests/test_infra_resume.py -k 'research_split or policy_windows or rng_device_and_revision'` | **26 passed、5 deselected，3.26 s**；未选中项不计通过。 |
| `$POLICY_PY dexmani_policy/smoke_test.py --config-only dp dp3 dqrise r3d multitask_dit` | **5 个配置全部 PASS**；未启动训练。 |
| `$REAL_PY -m py_compile dexmani_real/deployment/runner.py tests/test_policy_runner.py` | PASS。 |
| `$RUFF format --check dexmani_real/deployment/runner.py tests/test_policy_runner.py`；`$RUFF check dexmani_real/deployment/runner.py tests/test_policy_runner.py` | 两项 PASS。 |
| 两仓库 `git diff --check` | PASS。 |

**HOME：模型拒绝复验 PASS；实物 NOT_VERIFIED。** 离线 Python 探针使用默认 `ExperimentConfig`、joint `ActionRealizer` 和 Pinocchio，未创建设备连接。当前 mount 平移 `(-0.015,0,0)m`、四元数 wxyz `(0.707107,0,0.707107,0)`、手部 HOME（SDK 顺序，度）`(30,55.33,10,0.17,1.08,5,1.25,5,1.33,5,1.33,5)` 产生 `self_collision`。最初通过 `collisionResults` 枚举碰撞对的探针因现有 hpp-fcl 绑定类型转换限制而断言失败；随后用同一原生模型已更新的 geometry，通过 `pin.computeCollision(model, data, pair_index)` 逐对核对，确认唯一命中 `flange_link_0/right_hand_thumb_bend_link_0`。未修改既有诊断降级行为或环境来消除限制。

关闭实物阻断需要操作者提供并确认：① `custom_eef_link` 到 `right_hand_link` 的实际安装变换（坐标方向、平移单位和旋转约定）；② 实际手部 HOME 姿态及关节顺序；③ 上述碰撞对的几何与实际接触/间隙关系。现有证据不能判定实物确有碰撞，也不能归责于 mount、mesh 或 SRDF。保留拒绝与原回归，未改 mount、HOME、mesh、SRDF 或阈值；nominal mount 的合成通过不能替代验收。真机安全、真实 Raw/checkpoint 和论文资产恢复仍未验证；本轮无代码权限/依赖阻断，未连接设备、执行 HOME/replay/rollout/采集/真实标定，未训练、DDP 或修改既有实验产物。

收尾后相关清理：CollisionModel 直接复用 `robot.model` 的手部维数/SDK→URDF 索引，删除重复私有别名；replay 删除旧 `h5 = reader` 别名和迁移过程注释。修正碰撞注释中的实物暗示，README 与执行文档明确 NPZ 保留被拒绝预测、严格 JSON、保存失败与当前模式比较方法。保留仍有用途的旧 Raw 缺测/dispatch 读取、旧配置恢复及已复现绑定限制下的碰撞诊断降级；Policy 无改动。清理后再次执行上表 Real 全量离线 pytest 命令：**198 passed，12.83 s，无 skip**；对 `runner.py`、`planning/collision.py`、`replay/trajectory.py`、`tests/test_policy_runner.py` 执行 py_compile、Ruff check/format --check 均 PASS，两仓库 diff 检查 PASS。未新增测试框架、改变控制/数据语义或执行硬件命令。
