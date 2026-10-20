# Codex CLI 任务书：DexMani Real 架构与代码精简

版本：2026-10-20 / review2。对象：`haoyangzhanglab/dexmani_real`。性质：可执行的重构任务，不是新增运行时框架或长期兼容规范。

## 1. 任务目标与执行方式

请基于当前源码执行本任务，将 DexMani Real 整理为简洁、高效、模块化、解耦、易维护的真实机器人研究代码。优先减少重复表示、跨层依赖、无效抽象和重复计算，同时统一命名、注释与排版。

必须实现用户的五项要求：

1. 优化代码质量与效率，保留已有实验方法与有效行为。
2. 注释和 docstring 默认至多一行；一眼可见的功能不注释，必要的数学、坐标、单位、并发原因允许简短例外。
3. 自有代码的文件、函数、类、常量及属性不使用非必要的前导 `_`；命名采用明确的机器人学、控制和算法术语。
4. 用适当空行、清晰分组和普通控制流提升可读性，形成一致的开源代码风格。
5. 删除不必要的校验、contract 和内部转换，只保留能阻止真实错误的边界检查。

开始前读取根目录及目标目录的 `AGENTS.md`，检查工作区已有修改，沿活跃入口确认调用关系。本文由用户明确要求放在仓库根目录，属于对根目录临时任务文档限制的本次例外；无需为此修改 `AGENTS.md`。过程分析、测量与临时检查脚本放在被忽略的 `reports/` 中。

收到执行本任务的指令后，应推进代码修改、必要离线验证和文档同步，不停留在再次提出方案。按第 10 节分阶段实施，避免把全库改名、调度行为调整和数据准入变更混在同一批修改中。不要覆盖用户已有改动，不做任务无关的依赖升级。

以当前实际调用链决定实现，优先直接函数、现有小型对象和明确参数。每次结构调整应减少跨业务依赖、重复数据表示、可变状态所有者或实际重复工作之一；单纯改名、平铺目录属于可读性整理，不宣称解耦或性能收益。当前结构已清楚时保留，不为完成清单制造新抽象。

本任务范围为 `dexmani_real` 的自有代码、入口、配置、打包信息和直接相关文档；`dexmani_policy` 是接口依赖。涉及 Policy 内部的建议见第 8 节，不能在未实现相应接口时让 Real 依赖它。禁止批量改写 `third_party/`、外部 SDK 或历史实验数据。

## 2. 分析基线与参考取舍

本任务书的源码基线：

| 项目 | 审阅版本 | 应吸收的做法 | 本项目不需要照搬的机制 |
| --- | --- | --- | --- |
| [DexMani Real](https://github.com/haoyangzhanglab/dexmani_real/tree/86151f9b11e793682acbc90b9d144258828a07e5) | `86151f9` | 真实执行、记录、时序与物理边界 | 内部重复检查、重复表示、跨层依赖 |
| [DexMani Policy](https://github.com/haoyangzhanglab/dexmani_policy/tree/3d48b56d882bcef006f7cb708bb17b2dabe656db) | `3d48b56` | 集中的模型加载、数值预处理与预测适配 | 同源配置派生信息的多份存储和互相核对 |
| [LeRobot](https://github.com/huggingface/lerobot/tree/3991580a5673f5f0219a4e8835a5f973eeccf9a7) | `3991580` | 清楚的设备、策略、数据责任；在写入边界验证帧与 episode | 为多设备生态服务的完整 processor、feature、插件和版本兼容体系 |
| [LeFranX](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | `a39906e` | 机械臂与灵巧手的直接组合 | 为统一基类实现当前设备不具备的占位能力 |
| [Dexora](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783) | `6f2869d` | 明确的模型部署入口、必要的进程隔离 | 特定模型的补零、截断规则，以及本项目没有需求的服务层 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | `85c6f2e` | 设备所有权、共享内存与控制进程划分 | 固定布局数据在 Pydantic、字典、枚举和共享存储之间反复转换 |

执行时若版本已经变化，先核对受影响调用链，再调整具体落点。下面的源码事实是基线证据，不要求把已经改好的新代码改回旧结构。

本轮复核时 Real 的 main 为 `bb22857`，相比 `86151f9` 仅增加本任务书；Policy 已前进到 `3d48b56`。已核对 Policy 部署文件的变更：checkpoint 导入路径与恢复后的配置访问改变，`predict()` 输出检查、`configure_execution()` 及 Real 使用的公开加载签名仍在。执行时再次确认本地实际依赖版本，不假定两个仓库同步升级。

### 本轮 fact-check 与方案修正

以下位置均对应上述固定版本；“应如何重构”是设计判断，不是参考项目已经证明的性能结论。

| 核实事实与源码位置 | 对任务的修正 |
| --- | --- |
| `recording/` 负责在线记录，`dataset/` 读取 Raw 并离线导出；两者已有不同生命周期 | 取消强制合并为 `data/`，优先保留现有两个业务包 |
| `config/experiment.py` 导入 `ExecutionConfig`，`deployment/config.py` 通过局部 import 回读 experiment 的检查函数 | 消除这一具体依赖环；不强制重做全部配置类、YAML 层级和参数路径 |
| `deployment/runner.py::submit()` 先构造观测，再提交推理，最后登记 query/frozen commands | 不能把所有构造期检查移到异步 `predict()`；新计算产生的无效值必须在提交前发现 |
| `policy_observation_issue()` 在消费旧 plan 的控制步也执行，Policy 输入检查只在查询时执行 | 保留当前控制步的必要反馈/数值健康判断，不能用查询时检查取代它 |
| Policy `LoadedPolicy.predict()` 已校验完整预测，返回裁剪后的 CPU float64 数组 | Real `poll_model()` 的相同输出内容检查可去重，归属和权限判断仍由 Real 负责 |
| `PointCloudConfig.sampling_method` 参与已保存数值配方，`resolve_saved_pointcloud_recipe()` 直接按字段恢复 | 本次保留 `quickfps` 标识及加载检查，不因只有一个合法值就删掉 artifact 字段 |
| `RGBDGeometry` 表达原生双相机标定；活跃深度处理使用 color-aligned 数据；`RawEpisodeCameraModel` 本身仅两个字段 | 删除公共处理链的多余双相机表示，但不强制删掉有用的小型相机返回对象 |
| `camera_parameters_cv()` 对 `T_xarm_base_from_color` 求逆后输出 `[R|t]` | 明确保留 Raw 的 color→base 变换与 Policy/Canonical 的 base→color 外参之区别 |
| `EpisodeReader.read_timing()` 被导出/回放用于完整来源准入；arm 核对实际读取 `run_id`、`command_seq` | 诊断准入放宽必须单独说明；必要执行证据不随诊断字段一起删除 |
| 自有函数/类定义基本没有单前导 `_`，属性与常量中较多，reader 等还存在同名 property | 聚焦真实存在的名称，避免全库替换造成 property 递归、只读属性赋值或外部接口破坏 |

源码定位：[提交顺序](https://github.com/haoyangzhanglab/dexmani_real/blob/86151f9b11e793682acbc90b9d144258828a07e5/dexmani_real/deployment/runner.py#L550-L617)、[观测检查](https://github.com/haoyangzhanglab/dexmani_real/blob/86151f9b11e793682acbc90b9d144258828a07e5/dexmani_real/deployment/observation.py#L71-L146)、[保存配方](https://github.com/haoyangzhanglab/dexmani_real/blob/86151f9b11e793682acbc90b9d144258828a07e5/dexmani_real/sensor/pointcloud/config.py)、[相机变换](https://github.com/haoyangzhanglab/dexmani_real/blob/86151f9b11e793682acbc90b9d144258828a07e5/dexmani_real/sensor/camera/geometry.py#L95-L118)、[Policy 部署接口](https://github.com/haoyangzhanglab/dexmani_policy/blob/3d48b56d882bcef006f7cb708bb17b2dabe656db/dexmani_policy/deployment/runtime.py)。本轮为源码与依赖审阅；没有运行机器人、模型集成或性能基准，不据此声称真机验证通过。

## 3. 架构目标：职责明确、依赖直接、状态有唯一所有者

### 3.1 保留的职责边界

| 模块 | 职责 | 依赖约束 |
| --- | --- | --- |
| `robot/` | xArm/XHand 驱动、命令、目标映射、HOME、执行进程 | 不导入 teleop、deployment 或数据导出流程 |
| `sensor/` | 相机、VR、点云生产与设备格式转换 | 不包含 episode 编排和模型逻辑 |
| `ipc/` | 共享内存、通道及必要的固定布局 | 不感知实验任务、模型或 UI |
| `planning/` | 位姿、FK/IK、碰撞、目标检查和 HOME 路径 | 数值计算不读取操作者状态、不调用运动 SDK |
| `runtime/` | 权限与停止状态、观测快照、输入、公共 jog、进程生命周期 | 不反向依赖 teleop/deployment/replay 的业务实现 |
| `teleop/` | VR 映射、手部重定向、遥操作编排 | 通过已有目标映射与机器人命令路径执行 |
| `deployment/` | 策略观测、推理工作线程、动作时序、部署编排 | 模型内部归 Policy；SDK 调用归 robot |
| `replay/` | Raw 轨迹加载、回放与评价 | 复用目标检查和执行入口，不重新实现驱动 |
| `recording/` | 在线记录、异步 writer、Raw 读写与发布 | 不导入模型、离线 exporter 或工作流 session |
| `dataset/` | 从 Raw 离线处理并导出 Canonical | 依赖 Raw reader 与纯数值函数，不启动 recorder 或设备 |
| `calibration/` | 当前实验的相机、桌面及安装标定 | 不将历史环境数值升级为跨实验兼容协议 |
| `config/` | 实验配置组合与加载、共享数值配置 | 不导入 session/runner；工作流检查留在工作流入口 |

保留具体的 `Session` 作为资源构造与清理入口，保留具体的 `Runner` 组织控制循环。复用现有 `SensorProcesses`、`shutdown_local_runtime()`；不新建 `BaseSession`、`BaseRunner`、通用生命周期容器或全局 context。

保留有实际语义区别的小型数据结构，如 `ObservationRow`、目标、`RobotCommand`、`DispatchResult`。构造这些对象不应自动触发多轮验证。字段表、必要 enum 和 dataclass 不因文件名包含 schema/contract 就被整体删除。

先修复具体依赖边，不追求把所有包强制排成单向层级。例如 `planning` 读取 `robot/model.py` 的关节顺序和资产常量是合理的数据依赖；不要为消除这种 import 新建 domain/core 包。新拆出的纯函数传入所需数值或现有小配置，不接收整个 runner/session，也不为依赖注入引入框架。

### 3.2 有依据的文件调整

以下路径均相对 `dexmani_real/`；整组迁移时同步所有 import、示例、配置字符串、`__init__.py` 导出和打包声明，不保留旧路径转发文件或 deprecated alias。

| 当前位置 | 目标位置或处理 | 目的 |
| --- | --- | --- |
| `teleop/control/controller.py` | `teleop/controller.py` | 删除只有少量实现的目录层级 |
| `teleop/control/vr_mapping.py` | `teleop/vr_mapping.py` | 映射逻辑在 teleop 下直接可见 |
| `teleop/jog.py` | `runtime/jog.py` | 消除 runtime、标定对 teleop 的反向依赖 |
| `planning/planner.py` | `planning/home_planner.py` | 表达现有 HOME 路径规划职责 |
| `planning/admission.py` | `planning/target.py` | 明确是物理目标检查，不形成通用准入框架 |
| `utils/limits.py` | `robot/xhand/limits.py` | 将 XHand 专用限位规则归还设备模块 |
| `recording/` 与 `dataset/` | 保留现有包边界 | 在线记录与离线处理已解耦；移动到同一父目录本身没有额外收益 |
| `dataset/contracts.py` | 可改为 `dataset/schema.py`，保留小型字段表与 `ProcessingConfig` | 消除过宽命名；不把约百行的小文件机械拆成两三个文件 |
| `deployment/runner.py` 的纯时序部分 | 实质减少状态交叉访问时，移至一个 `deployment/scheduling.py` | 提取索引和时序计算，不引入新调度状态机 |
| `planning/__init__.py` 的 planner 重导出 | 消费者直接导入实际模块，移除无必要的包级聚合 | 减少导入 Pose 等基础对象时顺带导入 HOME 规划模块 |

`recording/storage/schema.py` 与 Canonical 字段定义分别描述 Raw 和训练缓存；它们的 dtype、缺测含义和用途不同，不强行合成一个来源表。只有在执行时发现可消除的真实重复职责，才进一步合并模块，并在结果中说明收益。不要为了目录对称新增空包，或将少量 helper 拆成大量文件。

### 3.3 配置归属

保留 dataclass 配置与当前覆盖顺序，先解决混合职责带来的具体依赖：

- 机械臂、手、相机、VR、点云参数仍放各自模块；不新增一套重复的配置快照。
- `deployment/config.py` 保留 `ExecutionConfig` 等声明及其局部数值约束。将依赖完整实验配置的工作流检查收拢到部署启动入口，消除与 `config/experiment.py` 的回环。
- `examples/run_policy.py` 与 Session 都调用启动检查，需明确哪个是完整启动所有者；不要为移走同一函数建立层层转发。配置预览与实际启动如果处于不同生命周期，可各自在需要时检查。
- CLI 的预检发生在输出目录准备前，Session 的模型加载在设备连接前；去重时保留有意义的早失败边界，不能只按同名函数调用次数删检查。
- `ControlParams` 先按遥操作、记录和共享运动参数清楚分组；消费者只拿实际需要的已有子配置或数值。只有存在独立使用和生命周期的字段组才提取小配置类，可留在现有文件。
- 不强制新增 `MotionConfig`、`TeleopConfig`、`RecordingConfig` 三套对象，不为配置外观整齐迁移所有 YAML 键或重写 CLI。
- 保留 hand-disabled 几何假设、键盘独立频率，以及 Policy 控制 dt 与设备 servo 频率的区别。

优先保持现有外部配置键、默认值和保存格式；这不需要增加兼容层。确需改变某个公开键时，同步全部当前消费者并明确变更，而不是批量迁移或维护新旧双路径。未知字段在输入处明确报错。普通导入、配置解析不创建 SDK、设备连接或模型。

## 4. 运行机制重构：分离职责，不改变算法与执行含义

### 4.1 部署时序

若提取 `deployment/scheduling.py`，仅承载现有查询/动作计划的数据定义及纯计算：slot 索引、segment 边界、固定 handoff、有效期。使用函数和现有 dataclass；不得传入整个 runner，也不在新对象里再保存一份 query/plan。它不导入 SDK、Torch、记录器、operator、IK，不接收回调来间接操作这些资源。若只能把 runner 方法搬过去再大量回调原对象，保留原位置并简化函数即可。

`PolicyRunner` 继续持有 episode 状态，组织读取、提交、结果回收、命令执行和记录；`InferenceWorker` 继续持有模型线程和 Future。保留以下现有行为：

1. `sync` 保留推理停顿，动作块内部使用 Policy 的 dt；`async/rtc` 使用固定时间网格和既定接管时刻。
2. 预测裁剪保持 `P = H - Nobs + 1`，从 `Nobs - 1` 开始；不得把 `A_exec`、训练动作窗口和预测 horizon 合并。
3. RTC 先生成候选 prefix 与预备命令，`model.submit()` 成功后才登记 query 并提交 frozen commands；提交失败不留下半提交状态。
4. frozen command 保持预备时的关节等价分支与实际投影值；消费时重新判断当前物理可行性，不重新投影出另一条命令。
5. episode 结束立即清除活动计划并撤销运动权限；仍在运行的 Future 必须保留查询归属直到被回收。不能把逻辑结束当作 CUDA 任务已取消。
6. 模型 reset、predict、close 由同一工作线程串行持有；关闭等待超时不代表底层推理已停止。
7. WAIT、观测过期、晚到结果、推理异常及部分 dispatch 继续留下真实记录和结束原因；不靠重试、填零或丢帧掩盖。
8. 既有 LAMP 手部历史只从实际成功提交的目标推进；初始测量 bootstrap、reset 及单步时序保持与训练一致。

### 4.2 状态所有权

| 状态 | 所有者与更新时刻 |
| --- | --- |
| 运动权限、run/epoch、stop/fault | `RuntimeState`；撤销立即生效，不能等待模型或记录完成 |
| 模型任务和未回收 Future | `InferenceWorker`；查询身份留至结果回收 |
| 活动 plan、slot、episode 历史 | 当前 `PolicyRunner`，不复制到另一个 session context |
| VR 重定向样本消费标记 | 求解前更新，失败样本不在后续 tick 隐式重试 |
| Teleop EMA 与上一条已提交目标 | dispatch 成功后提交，不能在候选计算时推进 |
| xArm 高频参考与 SDK 调用状态 | arm worker；主进程不再维护第二套有效参考 |
| Raw 写入队列与发布状态 | recorder/writer；保持跨线程数据快照所有权 |

抽公共清理函数时保留权限撤销、stop、记录结束、worker 回收和资源关闭之间的因果关系。每个资源只指定一个清理所有者；不要将所有 session 强行变为相同模板。

## 5. 校验与 contract 精简清单

原则：同一数值、同一时刻、同一所有权下的同一事实不重复检查；新的计算、异步提交或当前执行条件有各自边界。删除检查前同时确认异常类别、发现时机和此前已经发生的副作用。不得把删除的检查转移到另一个更复杂的通用验证框架。

### 5.1 本次应删除、合并或缩小的机制

| 位置/机制 | 实施要求 | 保留条件 |
| --- | --- | --- |
| `PolicyRunner.poll_model()` 重扫预测的 shape/dtype/finite | 当前 Policy 适配器已经验证并返回确定布局，删除 Real 重复扫描 | 保留 run/query 归属、异常处理、结果回收与当前权限判断；未来后端在自己的适配入口负责输出 |
| 观测构造后的全量 finite 扫描 | 只删除对已验证且未经数值变换的数据的重复扫描；合并同一次判断的重复 RGB 检查 | FK/类型降精度等新结果保留提交前必要检查；异步 Policy 输入检查不能替代当前控制步健康判断 |
| Session → Executor → Driver 重复验证静态配置 | 追踪实际构造路径，在最终配置确定后选择一个明确的验证所有者，删除同一路径下游重复项 | 独立可调用入口仍有输入责任；不能只验证 CLI 就假定直接 Session/Driver 调用必然经过 CLI |
| FK 批处理前后与逐行重复扫描 | 保留后端入口和真正计算结果的必要检查，删除中间没有变换的重复扫描 | float64→float32、IK、归一化等新计算结果不能仅凭输入有限而假定有效 |
| `validate_rot6d_geometry()` 的批量与可调阈值通用接口 | 合并进单条 `rot6d_to_rotmat()`，Gram–Schmidt 只计算一次，阈值使用内部常量 | 保留有限值、零基向量和近共线拒绝，不使用静默单位旋转替代 |
| 指尖名称的 thumb/index/mid/ring/pinky 子串校验 | 删除名称猜测；使用明确有序链接映射 | 模型建立时确认实际 URDF frame 与 SDK/模型关节顺序 |
| `PolicyQuery/ActionPlan.sources` 用于重复新鲜度计算 | 调度携带预计算截止时间，来源明细留在记录证据中 | 仍在每次实际执行前判断是否过期 |
| 对齐深度仍携带完整双相机几何 | 缩小为彩色网格内参、深度比例和所需基座外参 | 相机绑定、实际畸变与坐标转换必须正确 |
| 导出/回放默认解析完整查询日志 | 只读取该消费者需要的核心时间与执行证据；完整来源关联检查用于显式诊断 | 不降低核心训练字段、时间网格和执行证据要求 |
| timing dtype 必须完全等于全部当前诊断字段 | 改为检查必要字段存在、类型及行数一致，允许额外诊断列 | 不为缺失的必要字段补造数值 |
| 只有 `quickfps` 一个合法值的配方字段 | 保留 `sampling_method` 及一次加载检查，将其视为已保存的数值身份 | 不增加可选算法分支，也不通过删字段让现有 Policy 配方无法恢复 |

删除前定位全部当前调用方，确认上游实际保证了同一事实。不要新建 `ValidatedConfig`、`ValidatedPrediction`、通用 `Contract`、ValidatorRegistry，也不要增加 `strict/lenient` 或 `checked/unchecked` 双路径。

特别注意：`build_policy_observation()` 的异常当前发生在 `model.submit()` 之前；Policy `predict()` 在工作线程中执行。简单把前者的检查交给后者，会使 query 已登记、RTC commands 已冻结后才发现异常，可能继续消费旧计划。这不能算等价去重。本任务保持提交前失败语义；无法证明多余的这类检查暂留，而不新增同步预检协议来绕行。

构造函数校验是否多余取决于实际入口，不能仅凭“配置是 frozen dataclass”推断：其中仍可能含可变对象，且某些派生配置在启动过程中才生成。配置完成覆盖和解析后再验证，保留新的设备事实检查。

### 5.2 固定截止时间的等价实现

沿用当前 `policy_sources(rows[-1], fields)` 的来源定义，即最新控制行中请求模态的源时钟；不要把历史窗口的旧观测额外纳入最大年龄筛选。查询创建时确认所需来源非空、时间戳为正且不晚于当前单调时钟，并确认当时仍新鲜。来源和该查询的年龄预算确定后，计算：

```python
max_age_ns = int(max_decision_age_s * 1e9)
expires_at_ns = min(source_times_ns) + max_age_ns + 1
```

后续使用 `now_ns < expires_at_ns`。`+1` 保持现有整数纳秒下 `age <= max_age_ns` 的边界语义。该简化依赖来源时间固定、单调时钟和本次查询预算固定；不得在模型完成或 plan 接管时重新起算。RTC 前缀仍受旧计划截止时间约束。

继续区分决策有效期、当前反馈有效期、提交期限、执行期限和 WAIT/episode 预算。这些条件可以取最小值决定一次操作的有效期，但不能丢掉各自语义和结束原因。

### 5.3 对齐相机几何

现有 RealSense 活跃路径已经将 depth 对齐到 color；点云、记录、部署和离线处理实际使用彩色网格。将公共数据流收敛为现有 `CameraIntrinsics`、`depth_scale` 和按需提供的 `T_base_from_color`。

同步修改相机 worker 静态信息、点云 worker、Raw reader/export、Policy 相机参数和相机标定读取方后，删除公共处理链中的 `RGBDGeometry` 嵌套、`aligned_depth_to_color()` 和重复单位外参构造。`RawEpisodeCameraModel` 本身只是一个小型返回对象，可简化为 `intrinsics` 与 `depth_scale_m` 两字段保留；不要为了删一个类而把成组信息散成无名 tuple、字典或长参数列表。SDK 原生双相机信息如有实际诊断用途，可留在驱动内部。

在线和离线复用数值处理函数，但不合并进程生命周期。保持颜色对应、畸变处理、depth scale、裁剪、去桌面、采样、坐标系与 dtype 的现有数值含义。缺基座外参可以保存物理观测；需要基座坐标的处理必须明确拒绝，不能补单位矩阵。

必须区分两个方向：Raw 的 `camera_T_xarm_base_from_color` 将 color 坐标变换到 arm base；Policy/Canonical 的 `camera_extrinsic` 由其逆矩阵的前三行 `[R|t]` 按 C-order 展平为 12 维 float32，表示 base→color。`camera_intrinsic` 为 9 维 K，畸变参数另存。内部类型变窄不能改变这些字段含义，也不需要修改 Raw 文件格式。

### 5.4 数据准入：单独提交的有意行为调整

当前 `validate_export_episode()` 与物理回放依赖 `EpisodeReader.read_timing()` 的完整查询 JSON 和关联检查。改为读取核心执行检查实际用到的列，例如 arm 执行核对使用的 `run_id`、`command_seq`；逐项追踪消费者，不凭字段名称删除被图像对应、时间或执行判断使用的信息。`read_timing()`/现有诊断入口仍可提供完整时序追踪，不另建审计系统。

此项会使“核心物理数据和执行证据有效、仅诊断信息缺失或布局不同”的 episode 不再被默认拒绝，必须作为行为变更明确说明并独立验证。依赖推理因果关系的研究分析仍显式要求查询日志。记录端继续保存完整证据；writer、编码、发布失败的处理不因诊断准入放宽而静默忽略。

保持公共导出的全部 15 字段完整、浮点有限、严格时间网格与 dispatch 规则。不允许变为按模型只导出部分字段，不允许删坏帧、插值、补模态或重写 Raw。处理语义实质改变时重新导出缓存，不建立历史缓存迁移层。

### 5.5 必须保留的少数边界

- 外部 YAML/CLI 的未知字段、错误布尔/数值类型及必要物理参数检查，防止错误输入变成另一种有效配置。
- SDK/模型/外部数据进入系统时必要的 shape、finite、关节顺序、单位、坐标和时间含义检查。
- 实际运动前的当前权限、epoch、状态、反馈新鲜度和 deadline；IPC 等待前后检查不同时间事实，不机械去重。
- IK/Ruckig 等新生成结果的必要运动约束、物理限位、碰撞和执行跟踪检查。目标端点检查不能替代路径检查。
- 主进程同时确认 arm/hand 输入后再开始 dispatch，避免 hand 输入错误时 arm 已先运动；保留部分提交状态和故障处理。
- Raw 原子无覆盖发布、失败前缀、结束原因及执行计数。SUBMITTED、SDK 接受和测量到达保持不同含义。
- 当前有效缓存身份，如 `data_revision`；保持数据变化后不会错误复用旧缓存。

`EpisodeReader` 已缓存字段布局检查，不要误判为每次访问都会重新验证。物理限位与命令限位、motion epoch 与 command sequence 也不因外观相似而合并。不得用可被 `python -O` 删除的 `assert` 替代必要外部输入检查。

## 6. 命名、注释与排版规范

### 6.1 命名

文件与函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。保留算法名 `TAG`、`DexPilot`、`FK`、`IK`、`RTC`；采用 `pose`、`joint`、`trajectory`、`action_chunk`、`fingertip`、`retargeting` 等明确术语，减少含糊的 manager、engine、contract、admission、v2、safe 等包装性命名。

基线中自有函数/类定义已经基本没有单前导 `_`；主要清理对象是属性与模块常量。不要为了改名而破坏已经清晰的 `PolicyRunner`、`InferenceWorker`、`RobotCommand`、`DispatchResult` 等名称。

| 当前名称/形式 | 建议目标 | 注意事项 |
| --- | --- | --- |
| `XArm7MotionPlanner` | `HomePlanner` | 与 `home_planner.py` 和真实 HOME 职责一致 |
| `TargetAdmission` | `TargetChecker` | 保留物理目标检查，删除“通用准入”暗示 |
| `ActionRealizer` / `ActionRealization` | `ActionMapper` / `ActionResult` | 表达 joint/EEF 目标到可执行关节目标的映射，保留 IK 与拒绝信息 |
| `HandOptimizer`（TAG 实现） | `TAGOptimizer` | 不把特定算法命名为通用优化器 |
| `ArmParams`、`HandParams`、`CameraParams` 等 | 保持同一组配置类命名一致即可 | 不强制全库 Params→Config；同义词替换没有架构收益，也不改 SDK 外部类名 |
| `self._model`、`self._queue`、`self._process` | `self.model`、`self.queue`、`self.process` | 先检查同名方法、property 和继承协议 |
| `_CORE_DATASET_SPECS`、`_PROXIMAL_MASK` | `CORE_FIELD_SPECS`、`PROXIMAL_MASK` | 根据所在模块表达语义，不保留旧别名 |
| `self._path` 与 `path` property 并存 | 单一 `path` 属性，或有明确差别的 `episode_path` 存储 | 无价值透传 property 可删除；计算、只读或资源语义不能误删 |

自有属性、局部变量及模块名去掉非必要前导 `_`；丢弃值使用的单独 `_`、Python 特殊名称 `__init__`/`__getitem__`/`__all__`/`__main__`、第三方 override 和外部协议要求的 `_target_` 等保留。不要批量字符串替换 SDK 字段、第三方内部属性、序列化键或日志中的原始实验事实。属性改名逐类核对 property、继承和字符串反射；不能将 getter 改成读取自己，也不能把原私有字段赋给同名只读 property。

“论文风”要求术语与公式对应清楚。局部数学计算可使用 `q`、`q_ref`、`dq`、`dt`、`J`、`R`；跨函数和设备边界保留明确名称及必要单位，如 `joint_target_rad`、`timestamp_ns`、`T_base_from_camera`、`quat_wxyz`。已有一致的 `qpos` 约定可保留，不追求把所有名称缩成字母。

### 6.2 注释和 docstring

- 删除重复函数名、复述赋值/循环/return 的注释，以及显然可见的类型说明和历史重构叙述。
- 保留注释优先解释单位、坐标方向、四元数顺序、非直观数学选择、状态提交时刻及并发原因。
- 默认使用一个简短英文句子、一行 `#` 或一行 docstring；不要把原有多段说明硬压成超长一行。
- 公开函数的 shape、单位或状态副作用若无法从签名表达，可用一行 docstring；不机械生成 Args/Returns/Raises 模板。
- 数学推导、第三方算法来源、许可证、容易误改的并发约束允许必要的短多行说明；先尝试用命名和结构减少说明需求。
- 代码中的理由保持靠近相关语句；较长工作流说明放现有 README/算法文档，不为删注释建立新文档体系。

可保留的注释示例：

```python
# Commit the EMA reference only after successful dispatch.
# Keep the query owner until the running future is reclaimed.
# Rotation uses WXYZ; translation is expressed in the arm base frame.
```

### 6.3 排版与代码质量

- 顶层函数/类之间两行空行，类方法之间一行；函数内部按输入准备、计算、状态提交、输出划分自然段。
- 不在每条赋值之间插空行；配置字段按职责成组，不在每个字段后重复空行和注释。
- imports 按标准库、第三方、项目分组；避免星号导入、大型重导出入口和用局部 import 掩盖结构性循环依赖。
- SDK/Torch 的必要延迟导入保留，确保离线导入无硬件副作用。
- 优先普通 if/for 和有名中间量；复杂三元表达式、深嵌套和多职责长函数按真实责任简化，不用大量一次性 helper 碎片化流程。
- 行宽以约 100 字符为目标，长表达式用括号自然换行；不增加格式器依赖或整库格式化噪声。
- 类型标注优先覆盖跨模块入口与数值返回值；不为所有局部变量增加类型包装，不引入新类型框架。
- 异常在可处理的边界转换一次，保留原始原因；`except/pass` 和默认值不能掩盖技术错误。现有必要清理路径的宽异常捕获需按资源所有权判断。

## 7. 性能优化：删重复工作，保持数据所有权

按收益与风险顺序处理：

1. 删除已经确认的重复有限性扫描、静态验证、对象包装、序列化和表示转换。
2. 将固定内外参、静态安装矩阵、固定索引/限位、模态选择和年龄预算的派生值在构造/启动时计算一次；当前实验重新配置时重新构造。
3. 维持每个控制 tick 一份观测快照，避免各消费者再次读取产生时间不一致；相机与点云的实际对应关系仍被保留。
4. 在线和离线点云共用纯数值函数，避免逐帧重建同一几何对象。减少无用 HDF5/JSON 读取，不将整段视频为方便处理一次性载入内存。
5. 避免同一链路反复 `asarray`、`astype`、`stack` 和无意义 copy；但跨线程、跨进程和异步 recorder 需要的快照复制必须保留。
6. 不将复用的可变数组直接交给尚未完成的推理/写入任务；只有生产者与消费者生命周期可证明无重叠时才共享引用。
7. 仅在离线剖析发现具体瓶颈后考虑向量化、缓冲复用或 FK 批处理；维持随机采样顺序、dtype、算法参数和数值结果。

不新增线程池、异步框架、缓存服务、JIT 或通用对象池。已有进程隔离承担 SDK 所有权与阻塞边界，不以“减少进程数”为优化目标。既有 FK/碰撞后端含可变工作区，不将缓存实例扩大为跨线程共享。没有测量时只报告结构简化，不声称控制延迟或吞吐得到特定比例提升。

## 8. Real 与 Policy 的接口边界

Real 负责物理模态是否可提供、当前硬件与控制周期是否满足需求，以及动作如何执行。Policy 负责模型构造、normalizer、输入预处理、张量布局和算法执行能力。

Real 的 `load_configured_policy()` 已经调用 Policy 的 `configure_execution()`，且 Session 在 `robot.connect()` 之前完成模型加载和 warmup。复用这条现有链路，让 Policy 检查实际模型能力；不要再加一个转发适配层。CLI 中的轻量算法预检发生得更早，能够在准备输出目录和加载模型前拒绝已知不支持的组合，因此本次不强制删除 Temporal 的几行预检。不能把它与模型加载后的检查视为无条件等价。

LAMP 的部分条件约束 Real 自己的手部历史和调度：`hand_history_steps`、joint 模式、19 维控制布局、`H=Nobs=C=A_exec=1` 等不能仅因为 Policy 也认识 LAMP 就全部删除。保留 Real 消费端实际需要的条件；若算法名称已不必要，用现有字段表达要求，不新建 capability registry。

作为后续 Policy 独立改动：收窄 `load_policy(config, info)` 的独立输入，让配置解析一次、`PolicyInfo` 由内部派生，删除同源配置和元数据逐项自我核对。保留轻量 `inspect_policy()` 供 Real 在加载模型前检查硬件需求；checkpoint 恢复后的真实动作布局和数值预处理检查集中在恢复入口。

本次 Real 重构不要求复制模型代码或修改训练行为。Policy 未同步修改时使用其当前公开入口，记录真正需要的后续联动；不要建立旧/新接口双适配器，也不要留下调用尚不存在 API 的提交。

## 9. 保持不变的研究与数据语义

以下项目不是本次重构的算法优化空间：

- 机械臂/手关节顺序，19 维 joint action、21 维 EEF action 的布局与物理含义，WXYZ/rot6d 约定及当前坐标变换方向。
- 训练数值配方、控制 dt、预测窗口裁剪、既有 EMA、TAG/DexPilot 目标函数与 IK/碰撞策略。
- HOME 与在线控制使用的场景和路径要求；不能用一次端点检查代替路径验证。
- xArm 独立 SDK owner、Ruckig 参考、jog 输入代际、及时撤销权限、stop/fault 和资源清理语义。
- Raw 发布后不可变，失败前缀和结束原因保留；写入失败保留 staging，不能报告成功。
- 触觉保存的是既定有界保持策略后的消费值，导出直接使用，不重复扣偏置或修补；其他缺测不伪装为零。
- Canonical 全部 15 字段及整段拒绝规则；模型按需加载字段的行为仍属于下游。
- 相机内外参和 depth scale 对应实际数据；桌面、安装和运动学按当前实验配置使用，不增加历史标定一致性准入。

不引入额外平滑、自动恢复、无限重试、动作填充、丢弃晚帧或补发过期动作。这些都会改变实验系统，需要独立研究设计。

## 10. 实施阶段与完成条件

阶段应形成清楚可审阅的 diff；是否提交或推送代码服从本次执行会话的用户指令。本文不授权操作真实设备。

| 阶段 | 修改内容 | 本阶段完成条件 |
| --- | --- | --- |
| A：建立基线 | 确认入口、配置覆盖、调用方、动作/时间/数据含义；保存必要离线样例与当前输出 | 能说明每个拟删检查由谁保证；不把文档描述当成运行证据 |
| B：责任与依赖 | jog、XHand limits、teleop 平铺、HOME 命名，修复配置依赖环和多余重导出 | 所有入口引用一致；保留 recording/dataset 边界和现有外部配置键，减少真实反向依赖 |
| C：表示与时序 | 有实际收益的窄 scheduling、截止时间、相机几何收缩、状态所有权和生命周期去重 | sync/async/RTC 索引和交接保持；失败提交、旧结果回收、停止与记录一致 |
| D：校验精简 | 已证明重复的模型/观测/配置/FK 检查去重、rot6d 合并、指尖名称启发式删除 | 必要错误仍在正确时机被发现；保留采样配方标识，不增加预检协议 |
| E：数据准入调整 | 导出/回放从完整诊断检查中解耦，timing 改为必要字段检查 | 核心有效而诊断不同的数据按新规则接受；核心无效仍整段拒绝，Raw 不变 |
| F：风格与效率 | 剩余属性改名、注释压缩、空行整理、已定位的低风险性能优化、README 同步 | 自有命名一致、无旧别名、无无意义注释；数值与资源所有权无意外变化 |

与文件迁移必须同时发生的改名可在 B 完成；F 处理剩余机械性改动。每阶段完成必要检查后继续下一阶段，不要求为每次移动文件或普通实现选择重新征询用户。条件性提取若没有简化收益，保留原结构并用源码事实解释；不得为完成阶段机械制造文件或类。

## 11. 验证与验收

遵守现有 `AGENTS.md`：不运行或引入 pytest、Ruff，不连接真实设备，不擅自安装或升级可选依赖。基本检查：

```bash
python -m compileall -q dexmani_real examples
git diff --check
```

对受影响的机制补充少量能区分正确与错误实现的纯逻辑检查，临时脚本放 `reports/`，不用搭建通用测试平台。按实际修改选择下列检查，不为简单改名编写镜像测试：

| 改动 | 有效验证 |
| --- | --- |
| 文件/名称迁移 | 搜索旧 import、字符串构造路径、配置键和导出；AST/静态引用检查；确认 property/继承没有名称冲突 |
| 配置归属 | 代表性 teleop/deploy/replay/export 配置解析结果与原参数等价；现有配置键、CLI 覆盖和 hand-disabled 路径不丢字段 |
| deadline | 等于阈值、超过 1 ns、来源为零/未来、旧 RTC 前缀等边界；验证初始检查后与原谓词等价 |
| 调度与所有权 | 假时钟与假推理任务覆盖正常/晚到/异常、无效派生观测在提交前拒绝、提交失败不冻结、旧 Future 回收、handoff 前后索引 |
| Teleop | 同一 VR 样本失败后不重试；dispatch 失败不推进 EMA/上一目标 |
| 几何 | 合成 RGB-D 对齐、Raw color→base 与模型 base→color 的逆变换、FK/rot6d 的有效及退化输入；对应输出在合理浮点容差内一致 |
| Raw/Canonical | 临时小 episode 验证核心缺失/NaN/坏时间网格仍拒绝，纯诊断列变化不再阻塞；输出 15 字段及 dtype 不变 |
| 并发复制精简 | 推理或 recorder 尚未消费时修改生产者缓冲，确认消费者持有的快照不被污染 |
| 性能修改 | 同一离线输入与配置进行有限次前后比较，说明耗时、内存或数据扫描减少的具体证据 |

只执行可以确认无设备副作用的导入/检查。可选依赖缺失时列出受影响的验证项和具体原因，不用 stub 掩盖核心数值缺陷，也不把 compileall 成功称为运行验证。

最终验收：

- 当前采集、部署、回放、标定、导出入口对应的引用和配置同步完成。
- runtime 不再依赖 teleop 业务实现；硬件/传感器、数值计算、编排与数据处理责任清楚。
- 不以新增父目录、更多配置对象或更多转发层冒充模块化；每个条件性结构调整有实际简化收益。
- 没有新增 framework、registry、万能 context、冗余 wrapper、通用 contract 层或兼容分支。
- 校验去重有明确责任归属；动态执行条件和真实数值边界没有被误删。
- 第 E 阶段的准入变化被明确说明；其余阶段没有无意改变动作、时间、早失败位置、错误提交边界、数据或算法含义。
- 非必要前导 `_`、旧名称/导入和无价值注释已清理；必要例外能够从语言或外部接口要求解释。
- README 仅更新结构、工作流和入口，不新增易过时的字段级运行协议或完整类清单。
- 最终报告列出结构变化、删除的机制、保留的必要检查、验证结果、未验证项和必要的 Policy 后续联动。

不设机械的删行率、文件数或性能提升指标。完成标准是依赖更直接、状态更少、同一数据更少转换、同一事实更少验证，以及关键实验行为可解释且保持正确。

## 12. 给 Codex CLI 的启动指令

在仓库根目录向 Codex CLI 提交以下完整文本；也可直接要求执行本文件：

> 请在当前 dexmani_real 仓库中实际实施重构，不要只输出分析或新的方案。
>
> 先读取根目录及相关目录的 AGENTS.md、CODEX_REFACTOR_TASK.md，确认任务书版本为“2026-10-20 / review2”。如本地任务书过旧，先核对 GitHub main 上的最新文件，在保留已有工作区修改的前提下更新依据；不要 reset、覆盖用户改动或强行同步整个分支。以当前源码和实际调用链核实任务书，不机械照搬已经失效的建议。
>
> 目标是减少真实依赖、重复数据表示、分散的可变状态和重复工作，使模块边界清楚、代码直接易读。不得以增加目录、配置类、转发层或通用框架代替解耦。保留已有 recording/dataset 职责边界及外部配置键；条件性拆分没有实际收益时保留原结构，并说明源码依据。
>
> 按任务书阶段连续推进：先做有依据的依赖与职责调整，再收缩表示、精简已证明重复的检查，最后完成命名、注释、排版和低风险效率优化。诊断准入放宽作为独立行为变更处理。普通实现选择自行决定，不必为每次改名、移动文件或局部重构再次询问。
>
> 落实代码风格：自有标识符去掉非必要前导下划线，使用准确的机器人学与算法术语；保留 Python 特殊方法及外部协议要求的名称，逐类检查 property、继承和字符串反射。注释默认一行，只说明必要的数学、坐标、单位或并发原因，显然可见的功能不注释；适当空行，避免过长表达式和无意义 helper。
>
> 校验按事实和时机去重。保留新数值结果的提交前检查、当前控制步健康判断、运动权限与有效期；不能用异步 Policy 检查替代同步提交前拒绝。保持 RTC 提交成功后才登记 query/frozen commands、旧 Future 回收、dispatch 后才推进控制参考等语义。保留 sampling_method 的数值配方身份，区分 Raw color→base 与模型 base→color 外参，保持 Raw 不可变及 Canonical 全部 15 字段规则。
>
> 范围限于 dexmani_real 自有代码和直接相关配置、入口、文档。不修改第三方代码、历史 Raw 或 dexmani_policy；确有 Policy 联动需求时单独列出，Real 使用已存在的公开接口。不要连接设备、运行真机或写标定，不运行或引入 pytest、Ruff，不擅自安装/升级依赖。
>
> 执行 compileall、git diff --check，以及本次修改真正需要的少量纯逻辑检查；临时材料放 reports/。检查应覆盖受影响的错误时机、调度、几何或数据规则，不为简单改名建立测试平台。缺少可选依赖时准确记录限制，继续完成可独立验证的工作；不要用无依据的默认值、静默 fallback 或假数据掩盖问题。
>
> 完成后审阅最终 diff，清理旧引用、失效注释、无用包装和重复实现，同步相关入口与 README。用中文报告：实际修改及简化收益、删除的机制、必要保留项及理由、验证命令与结果、未验证项和真正的后续依赖。没有测量不声称性能提升，没有真机运行不声称真机验证通过。请推进到可执行部分完成，不停留在规划阶段。
