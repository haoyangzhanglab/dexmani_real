# Codex CLI 任务书：DexMani Real 架构与代码精简

日期：2026-10-20。对象：`haoyangzhanglab/dexmani_real`。性质：可执行的重构任务，不是新增运行时框架或长期兼容规范。

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

本任务范围为 `dexmani_real` 的自有代码、入口、配置、打包信息和直接相关文档；`dexmani_policy` 是接口依赖。涉及 Policy 内部的建议见第 8 节，不能在未实现相应接口时让 Real 依赖它。禁止批量改写 `third_party/`、外部 SDK 或历史实验数据。

## 2. 分析基线与参考取舍

本任务书的源码基线：

| 项目 | 审阅版本 | 应吸收的做法 | 本项目不需要照搬的机制 |
| --- | --- | --- | --- |
| [DexMani Real](https://github.com/haoyangzhanglab/dexmani_real/tree/86151f9b11e793682acbc90b9d144258828a07e5) | `86151f9` | 真实执行、记录、时序与物理边界 | 内部重复检查、重复表示、跨层依赖 |
| [DexMani Policy](https://github.com/haoyangzhanglab/dexmani_policy/tree/e5da473940d47f8dd4771f4625e7ad19fe0eacbb) | `e5da473` | 集中的模型加载、数值预处理与预测适配 | 同源配置派生信息的多份存储和互相核对 |
| [LeRobot](https://github.com/huggingface/lerobot/tree/3991580a5673f5f0219a4e8835a5f973eeccf9a7) | `3991580` | 清楚的设备、策略、数据责任；在写入边界验证帧与 episode | 为多设备生态服务的完整 processor、feature、插件和版本兼容体系 |
| [LeFranX](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | `a39906e` | 机械臂与灵巧手的直接组合 | 为统一基类实现当前设备不具备的占位能力 |
| [Dexora](https://github.com/dexoravla/Dexora/tree/6f2869d61483774aa75cda4cd2260dc6a5faa783) | `6f2869d` | 明确的模型部署入口、必要的进程隔离 | 特定模型的补零、截断规则，以及本项目没有需求的服务层 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | `85c6f2e` | 设备所有权、共享内存与控制进程划分 | 固定布局数据在 Pydantic、字典、枚举和共享存储之间反复转换 |

执行时若版本已经变化，先核对受影响调用链，再调整具体落点。下面的源码事实是基线证据，不要求把已经改好的新代码改回旧结构。

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
| `data/` | 采集记录、Raw 读写、离线处理、Canonical 导出 | Raw 与训练缓存保留不同表示，不引入统一万能 schema |
| `calibration/` | 当前实验的相机、桌面及安装标定 | 不将历史环境数值升级为跨实验兼容协议 |
| `config/` | 实验配置组合与加载、共享数值配置 | 不导入 session/runner；工作流检查留在工作流入口 |

保留具体的 `Session` 作为资源构造与清理入口，保留具体的 `Runner` 组织控制循环。复用现有 `SensorProcesses`、`shutdown_local_runtime()`；不新建 `BaseSession`、`BaseRunner`、通用生命周期容器或全局 context。

保留有实际语义区别的小型数据结构，如 `ObservationRow`、目标、`RobotCommand`、`DispatchResult`。构造这些对象不应自动触发多轮验证。字段表、必要 enum 和 dataclass 不因文件名包含 schema/contract 就被整体删除。

### 3.2 明确的文件调整

以下路径均相对 `dexmani_real/`；整组迁移时同步所有 import、示例、配置字符串、`__init__.py` 导出和打包声明，不保留旧路径转发文件或 deprecated alias。

| 当前位置 | 目标位置或处理 | 目的 |
| --- | --- | --- |
| `teleop/control/controller.py` | `teleop/controller.py` | 删除只有少量实现的目录层级 |
| `teleop/control/vr_mapping.py` | `teleop/vr_mapping.py` | 映射逻辑在 teleop 下直接可见 |
| `teleop/jog.py` | `runtime/jog.py` | 消除 runtime、标定对 teleop 的反向依赖 |
| `planning/planner.py` | `planning/home_planner.py` | 表达现有 HOME 路径规划职责 |
| `planning/admission.py` | `planning/target.py` | 明确是物理目标检查，不形成通用准入框架 |
| `utils/limits.py` | `robot/xhand/limits.py` | 将 XHand 专用限位规则归还设备模块 |
| `recording/recorder.py` | `data/recorder.py` | 统一数据入口 |
| `recording/storage/*` | `data/raw/*` | 保留 Raw 自己的读写和格式边界 |
| `dataset/processing.py`、`pointcloud.py`、`export.py` | `data/` 下同名文件 | 明确本仓库负责离线处理与导出 |
| `dataset/contracts.py` | 字段表放 `data/schema.py`，`ProcessingConfig` 放 `data/config.py` | 拆开字段定义与处理参数，不创建 contract 子系统 |
| `deployment/runner.py` 的时序部分 | 少量内容移至 `deployment/scheduling.py` | 分离动作索引、交接与有效期计算，不搬走整个控制循环 |

`data/raw/schema.py` 与 `data/schema.py` 分别描述 Raw 和 Canonical；它们的 dtype、缺测含义和用途不同，不强行合成一个来源表。不要为了目录对称新增空包，或将少量 helper 拆成大量文件。

### 3.3 配置归属

拆解当前混合了遥操作、记录与运动参数的 `ControlParams`，各参数只保留一个权威位置：

- 机械臂、手、相机、VR、点云参数仍放各自模块的配置文件。
- 共享 workspace、IK 容差及运动几何参数保留在 `config/control.py` 的 `MotionConfig`。
- EMA、VR 映射、遥操作频率与重定向选择归 `teleop/config.py` 的 `TeleopConfig`；键盘频率继续独立。
- 记录目录、记录开关、采集时长归 `data/config.py` 的 `RecordingConfig`；导出数值参数使用同文件的 `ProcessingConfig`。
- 手部启用状态归设备配置；机械臂模式、在线控制和手禁用时的几何假设保持原义。
- `deployment/config.py` 保留执行参数定义；启动时跨模块检查放到部署入口，删除对 `ExperimentConfig` 的循环依赖。
- `ExperimentConfig` 仅组合这些配置。普通导入、读取配置不创建 SDK、设备连接或模型。

维持当前配置覆盖顺序和实验默认值。修改参数路径时更新仓库内 YAML、CLI 覆盖、帮助文本和 README；未知字段在输入处明确报错，不增加自动迁移、别名或猜测默认值。已保存的 Policy 数值配方和 Raw 字段不随 Python 配置改名任意变更。

## 4. 运行机制重构：分离职责，不改变算法与执行含义

### 4.1 部署时序

`deployment/scheduling.py` 只承载现有查询/动作计划的少量数据结构及纯计算：slot 索引、segment 边界、固定 handoff、有效期计算。优先函数加现有 dataclass；确有共享可变状态时才使用一个小对象。不得导入 SDK、Torch、记录器、operator、IK，不建立策略注册表或通用调度框架。

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

原则：外部输入和新数值结果在消费边界检查一次；内部已确定表示直接传递；随时间变化的执行条件继续在执行时检查。不得把删除的检查转移到另一个更复杂的通用验证框架。

### 5.1 本次应删除、合并或缩小的机制

| 位置/机制 | 实施要求 | 保留条件 |
| --- | --- | --- |
| `PolicyRunner.poll_model()` 重扫预测的 shape/dtype/finite | 当前 Policy 适配器已经验证并返回确定布局，删除 Real 重复扫描 | 保留 run/query 归属、异常处理、结果回收与当前权限判断；未来后端在自己的适配入口负责输出 |
| 观测构造后的全量 finite 扫描 | 由 Policy 输入适配器检查最终数值；删除 builder 重复扫描，合并同一次判断的重复 RGB 格式检查 | 控制循环仍检查当前必需反馈缺失、故障与新鲜度 |
| Session → Executor → Driver 重复验证静态配置 | 统一所有公开工作流，在最终配置确定后、连接前验证一次，删除下游重复整份验证 | 真实设备身份、工作模式和运行反馈是新信息，仍在设备边界处理 |
| FK 批处理前后与逐行重复扫描 | 保留后端入口和真正计算结果的必要检查，删除中间没有变换的重复扫描 | float64→float32、IK、归一化等新计算结果不能仅凭输入有限而假定有效 |
| `validate_rot6d_geometry()` 的批量与可调阈值通用接口 | 合并进单条 `rot6d_to_rotmat()`，Gram–Schmidt 只计算一次，阈值使用内部常量 | 保留有限值、零基向量和近共线拒绝，不使用静默单位旋转替代 |
| 指尖名称的 thumb/index/mid/ring/pinky 子串校验 | 删除名称猜测；使用明确有序链接映射 | 模型建立时确认实际 URDF frame 与 SDK/模型关节顺序 |
| `PolicyQuery/ActionPlan.sources` 用于重复新鲜度计算 | 调度携带预计算截止时间，来源明细留在记录证据中 | 仍在每次实际执行前判断是否过期 |
| 对齐深度仍携带完整双相机几何 | 缩小为彩色网格内参、深度比例和所需基座外参 | 相机绑定、实际畸变与坐标转换必须正确 |
| 导出/回放默认解析完整查询日志 | 只读取该消费者需要的核心时间与执行证据；完整来源关联检查用于显式诊断 | 不降低核心训练字段、时间网格和执行证据要求 |
| timing dtype 必须完全等于全部当前诊断字段 | 改为检查必要字段存在、类型及行数一致，允许额外诊断列 | 不为缺失的必要字段补造数值 |
| 只有 `quickfps` 一个合法值的配置开关 | 移除无选择意义的用户参数，算法身份保留在处理记录 | 不能改变采样算法、随机性或数值配方 |

删除前定位全部当前调用方，确认上游实际保证了同一事实。不要新建 `ValidatedConfig`、`ValidatedPrediction`、通用 `Contract`、ValidatorRegistry，也不要增加 `strict/lenient` 或 `checked/unchecked` 双路径。

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

同步修改相机 worker 静态信息、点云 worker、Raw reader/export、Policy 相机参数和相机标定读取方后，删除公共链路中的 `RGBDGeometry`、`aligned_depth_to_color()`、重复单位外参构造和无额外职责的 `RawEpisodeCameraModel`。SDK 原生双相机信息如有实际诊断用途，可留在驱动内部。

在线和离线复用数值处理函数，但不合并进程生命周期。保持颜色对应、畸变处理、depth scale、裁剪、去桌面、采样、坐标系与 dtype 的现有数值含义。缺基座外参可以保存物理观测；需要基座坐标的处理必须明确拒绝，不能补单位矩阵。

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
| `ArmParams`、`HandParams`、`CameraParams` 等 | 对应的 `ArmConfig`、`HandConfig`、`CameraConfig` 等 | 统一配置类后缀；不改 SDK 外部类名 |
| `self._model`、`self._queue`、`self._process` | `self.model`、`self.queue`、`self.process` | 先检查同名方法、property 和继承协议 |
| `_CORE_DATASET_SPECS`、`_PROXIMAL_MASK` | `CORE_FIELD_SPECS`、`PROXIMAL_MASK` | 根据所在模块表达语义，不保留旧别名 |
| `self._path` 与 `path` property 并存 | 单一 `path` 属性，或有明确差别的 `episode_path` 存储 | 无价值透传 property 可删除；计算、只读或资源语义不能误删 |

自有属性、局部变量及模块名去掉非必要前导 `_`；丢弃值使用的单独 `_`、Python 特殊名称 `__init__`/`__getitem__`/`__all__`/`__main__`、第三方 override 和外部协议要求的 `_target_` 等保留。不要批量字符串替换 SDK 字段、第三方内部属性、序列化键或日志中的原始实验事实。

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

不新增线程池、异步框架、缓存服务、JIT 或通用对象池。已有进程隔离承担 SDK 所有权与阻塞边界，不以“减少进程数”为优化目标。没有测量时只报告结构简化，不声称控制延迟或吞吐得到特定比例提升。

## 8. Real 与 Policy 的接口边界

Real 负责物理模态是否可提供、当前硬件与控制周期是否满足需求，以及动作如何执行。Policy 负责模型构造、normalizer、输入预处理、张量布局和算法执行能力。

Real 已有基于算法 `_target_` 类名的限制，应优先改为调用 Policy 已提供的 `configure_execution()` 等部署接口，由 Policy 决定算法是否支持 async/RTC。保留 Real 自己的可执行时间与物理条件检查，不新建 capability registry 或跨仓库协议框架。

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
| B：责任与依赖 | jog、XHand limits、teleop 平铺、HOME 命名、配置归属、recording/dataset 合入 data | 所有入口引用一致，无反向业务依赖、旧转发包和 import 环；行为保持 |
| C：表示与时序 | 窄 scheduling、截止时间、相机几何收缩、状态所有权和生命周期去重 | sync/async/RTC 索引和交接保持；失败提交、旧结果回收、停止与记录一致 |
| D：校验精简 | 模型/观测/配置/FK 去重、rot6d 合并、指尖名称启发式删除、无效配置项移除 | 必要边界仍拒绝错误输入；内部同一事实不重复扫描 |
| E：数据准入调整 | 导出/回放从完整诊断检查中解耦，timing 改为必要字段检查 | 核心有效而诊断不同的数据按新规则接受；核心无效仍整段拒绝，Raw 不变 |
| F：风格与效率 | 剩余属性改名、注释压缩、空行整理、已定位的低风险性能优化、README 同步 | 自有命名一致、无旧别名、无无意义注释；数值与资源所有权无意外变化 |

与文件迁移必须同时发生的改名可在 B 完成；F 处理剩余机械性改动。每阶段完成必要检查后继续下一阶段，不要求为每次移动文件或普通实现选择重新征询用户。

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
| 配置归属 | 代表性 teleop/deploy/replay/export 配置解析结果与原参数等价；CLI 覆盖和 hand-disabled 路径不丢字段 |
| deadline | 等于阈值、超过 1 ns、来源为零/未来、旧 RTC 前缀等边界；验证初始检查后与原谓词等价 |
| 调度与所有权 | 假时钟与假推理任务覆盖正常/晚到/异常、提交失败不冻结、episode 结束后旧 Future 回收、handoff 前后索引 |
| Teleop | 同一 VR 样本失败后不重试；dispatch 失败不推进 EMA/上一目标 |
| 几何 | 合成 RGB-D 对齐、单位/坐标转换、FK 和 rot6d 的有效及退化输入；对应输出在合理浮点容差内一致 |
| Raw/Canonical | 临时小 episode 验证核心缺失/NaN/坏时间网格仍拒绝，纯诊断列变化不再阻塞；输出 15 字段及 dtype 不变 |
| 并发复制精简 | 推理或 recorder 尚未消费时修改生产者缓冲，确认消费者持有的快照不被污染 |
| 性能修改 | 同一离线输入与配置进行有限次前后比较，说明耗时、内存或数据扫描减少的具体证据 |

只执行可以确认无设备副作用的导入/检查。可选依赖缺失时列出受影响的验证项和具体原因，不用 stub 掩盖核心数值缺陷，也不把 compileall 成功称为运行验证。

最终验收：

- 当前采集、部署、回放、标定、导出入口对应的引用和配置同步完成。
- runtime 不再依赖 teleop 业务实现；硬件/传感器、数值计算、编排与数据处理责任清楚。
- 没有新增 framework、registry、万能 context、冗余 wrapper、通用 contract 层或兼容分支。
- 校验去重有明确责任归属；动态执行条件和真实数值边界没有被误删。
- 第 E 阶段的准入变化被单独说明，其余阶段没有无意改变动作、时间、数据或算法含义。
- 非必要前导 `_`、旧名称/导入和无价值注释已清理；必要例外能够从语言或外部接口要求解释。
- README 仅更新结构、工作流和入口，不新增易过时的字段级运行协议或完整类清单。
- 最终报告列出结构变化、删除的机制、保留的必要检查、验证结果、未验证项和必要的 Policy 后续联动。

不设机械的删行率、文件数或性能提升指标。完成标准是依赖更直接、状态更少、同一数据更少转换、同一事实更少验证，以及关键实验行为可解释且保持正确。

## 12. 给 Codex CLI 的启动指令

在仓库根目录向 Codex CLI 提交以下任务即可：

> 请读取 AGENTS.md 和 CODEX_REFACTOR_TASK.md，按任务书在当前 dexmani_real 工作区实施重构。先核对当前版本、已有修改和受影响调用链，然后按阶段推进代码修改与必要离线验证。保留任务书明确的物理、时序和数据语义，完成命名、注释、排版、模块依赖及校验精简。不要运行真机，不使用 pytest 或 Ruff，不引入额外框架，不修改第三方代码或历史 Raw。不要停留在重新分析方案；完成可执行部分，并在最终报告中明确真实阻塞、未验证项及跨仓库后续工作。
