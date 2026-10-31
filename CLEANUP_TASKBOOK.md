# DexMani Real 本轮清理任务书

状态：待执行。本文件只定义方案，不代表代码已修改或验收通过。

核查基线：[23d3d9d962d6f0acee995850e2a8c1bc0b8c24e1](https://github.com/haoyangzhanglab/dexmani_real/commit/23d3d9d962d6f0acee995850e2a8c1bc0b8c24e1)。落地时以当前源码为准；如果相关代码已变化，先复核对应调用链，不回退已有修复。

## 1. 目标与范围决定

让个人研究的日常路径更直接：减少无实际作用的协议字段、配置合并和重复准入，让参数、状态和检查各有清楚的归属。

本轮完成下列 T1–T7。需要放弃运行模式、硬件支持、实验调参或既有输出能力的项目全部推迟，详见第 6 节。不要为追求减行数扩大范围，也不要新增一套“轻量框架”。

允许的接口变化只有三类：删除 SAM 私有回显参数；障碍物名称自动生成；录制选项从 Python 配置迁至现有 CLI。另有一项明确行为变化：VR 运行端接受有效角度，不再强制依赖历史标定质量报告。对应调用方和说明须一起迁移。

保持现有采集、遥操作、键盘点动、策略运行、回放、导出、几何处理能力；保留硬件边界、数据有效性和运动约束。这里的“保持能力”不要求保留上述被替换的配置写法或私有参数。

## 2. 已核实的事实

- 基线共有 18 个配置 dataclass、123 个声明字段，其中 13 个是 `RealConfig` 的组装字段。未发现完全没有消费者或组装用途的配置字段；不能把“默认未触发”当作“无用”。
- `evidence` 仅命中两条代码注释；`contract` 只命中历史任务文档；`artifact` 的命中不构成独立制品管理框架。本轮不按关键词批量删除文件、函数或类。
- SAM 当前是实例独占管道、单线程、同步一问一答。`frame_key` 从请求复制到回复，模型不读取它。
- `StaticCollisionBox.name` 用于碰撞对象标识与拒绝说明，但可以从配置顺序派生。两个几何不同、都使用默认名字的盒体会被重名校验拒绝。
- 录制三项选项已经有 CLI 参数，当前又与 `CollectionConfig` 通过两层 `replace` 合并。
- `load_vr_transform()` 当前构建带质量信息的 `HeadingCalibration`，其运行调用方仅消费 `.transform`；录制归档的是实际旋转矩阵。
- `EpisodeRecorder.start()` 的两个仓内业务调用分别传入字面量 `teleop`、`policy_rollout`。`save_points()` 的两处调用都直接消费 `sample_mesh_surface()` 的输出。
- retarget 的部分静态条件已在入口、VR 接收和手部反馈边界建立；骨长、掌退化、模型约束、求解结果和时间推进仍有独立职责。

这些是本轮范围的依据，不是删除所有验证或诊断的授权。

## 3. 执行顺序

下表优先级表示本轮实施顺序，不表示故障严重等级。

|任务|优先级|完成后的直接收益|验证重点|
|---|---|---|---|
|T1 删除 SAM `frame_key` 回显链|P1|请求、回复和调用方少维护一份无效身份镜像|顺序请求、错误传播、真实帧身份不变|
|T2 自动生成障碍物名称|P1|移除一个参数和一套人工命名准入|多盒体可用、几何与碰撞行为不变|
|T3 录制选项采用 CLI 单一来源|P1|删除 `CollectionConfig` 和配置覆盖合并|默认行为、路径、限时、快照一致|
|T4 VR 运行读取只依赖角度|P2|运行不必重建离线标定报告|旋转方向、角度检查、标定质量把关|
|T5 删除录制来源的内部枚举复检|P3|去掉已由调用方固定的准入分支|来源字段及消费者保留|
|T6 删除采样资产保存前的重复扫描|P3|去掉不必要的扫描与 float64/float32 往返|数组不变、独占创建与外部读取检查|
|T7 删除 retarget 已被覆盖的静态复检|P3|输入责任集中到已有边界|校验前提成立、数值与状态语义不变|

按 T1 → T2 → T3/T5 → T4 → T6/T7 完成。每组修改后做对应检查；不为并行拆分增加协调成本，不等待所有修改完成才发现接口断裂。

## 4. 任务定义与验收

### T1. 删除 SAM 私有 `frame_key` 回显链

入口文件：

- `third_party/cordvip_geometry/src/cordvip_geometry/segmentation.py`
- `third_party/cordvip_geometry/src/cordvip_geometry/segmentation_worker.py`
- `third_party/cordvip_geometry/src/cordvip_geometry/scene_tracking.py`
- `third_party/cordvip_geometry/src/cordvip_geometry/cli/scene_initialization.py`

实施要求：

1. 从 `GroundedSAM2Segmenter.call/predict/initialize/track` 删除 `frame_key` 参数及传递，从请求、worker 回复和回复比对中一起删除。
2. 删除两个场景调用方仅为该回显构造的临时字典。全仓核对其他调用，不留下废参数、兼容占位或新的 request ID。
3. 保留二进制消息 framing、操作分派、模型错误、EOF/断管处理、固定对象 ID、mask 类型/形状/互斥检查，以及进程和管道收尾。
4. 保留真实 `Frame`、NPZ 和场景结果中的 source、row、clock、对象身份与来源记录。不要把“回显无价值”推广到模型来源信息、帧配对或控制任务所有权。

验收：

- 通过轻量假模型或协议端点执行 load、predict、initialize、连续 track；调用与回复顺序正确，数据值和对象 ID 不变。
- 模型异常、EOF/断管仍能到达调用方，失败不变成空结果或成功。
- 相关运行代码不存在 `frame_key` 定义、发送、回显或调用；真实帧标识仍能进入原有结果。
- 不下载 SAM 权重，不新增多路复用、重连、请求管理器或协议版本层。

### T2. 障碍物按顺序自动命名

入口文件：`dexmani_real/config/control.py`、`dexmani_real/control/collision.py`。

实施要求：

1. 删除 `StaticCollisionBox.name`，以及仅依赖它的非空、保留名、重名校验。
2. 在 `CollisionModel` 遍历 `static_boxes` 时使用稳定的零起始顺序生成 `obstacle_0`、`obstacle_1` 等名称；桌面仍使用 `table`。
3. 保留对象索引到名称的现有用途，使碰撞拒绝仍可定位到配置中的盒体。不新增命名服务、UUID、别名表或名称配置。
4. 同步修改仓内构造方和当前使用说明。位置、尺寸、四元数、静态盒体数量、碰撞对和桌面接触规则保持不变。

验收：

- 配置两个不同位置的盒体无需手工命名即可通过；生成名称按顺序唯一。
- 原来带自定义名称的等价几何，迁移前后 placement、尺寸和碰撞结论一致；只允许拒绝文本中的盒体名称变化。
- 非法尺寸/姿态仍在已有边界被拒绝；旋转盒体继续支持。
- 若本机缺少 Pinocchio/hppfcl，可以验证配置与构造参数，并明确真实碰撞等价检查未执行；不能把 stub 结果称为碰撞验证通过。

### T3. 录制选项迁至现有 CLI，删除双来源

入口文件：`scripts/collect_teleop.py`、`dexmani_real/config/teleop.py`、`dexmani_real/config/real.py`、`dexmani_real/teleop/run.py`、`dexmani_real/teleop/loop.py`。

实施要求：

1. 保留现有 `--record/--no-record`、`--episodes-dir`、`--max-duration`，将默认值直接放在 CLI：`True`、`"episodes"`、`60.0`。默认值只定义一次。
2. 删除 `CollectionConfig`、`RealConfig.collection`、相关导入、覆盖优先级和两层 `replace`。不改造成另一个配置 dataclass、嵌套 namespace 或 preset。
3. 使用少量显式关键字参数把最终值传给 `run_teleop`。在装配处计算录制目录；`TeleopLoop` 接收其实际需要的目录和最大时长，不再读取 `runtime.collection`。不要把整个 argparse 对象传入业务层。
4. 录制启用、相机启动/健康检查、手部禁用时禁止录制、单段限时、保存/丢弃/退出的现有关系保持不变。未录制路径继续可用，不因迁移增加相机或标定依赖。
5. 录制启用时，对非空目录及有限正时长保留一次入口检查。目录相对仓库根目录的既有解析方式、绝对目录、TASK 子目录拼接保持一致；不顺手改变为相对当前工作目录。
6. `--print-config` 应同时展示 `RealConfig` 和最终录制选项；Raw 的 `runtime_config_json` 也必须包含实际采用的录制选项。可继续使用原有 JSON 中的 `collection` 键作为归档表示，但它不得成为第二份运行配置或被业务层读回。序列化在设备连接/分段副作用之前完成，不把潜在序列化错误拖到录制中途。
7. 更新 README 的参数说明与最小迁移例子：原来在 Python 中填写 `CollectionConfig(...)` 的三个值改到命令行；其余 `--config` 仍加载可信 Python 的 `CONFIG: RealConfig`。不保留旧构造写法的兼容适配器，不修改用户历史数据。

明确代价：不再能在 Python 实验配置中设置这三个默认值；三个可调选项及对应能力仍完整存在。此项是已选定的入口简化，不是将录制能力固化。

验收：

- `--help`、默认 `--print-config`、覆盖参数后的 `--print-config` 均可运行，且不连接硬件、读取标定或加载模型。
- 默认和覆盖录制参数分别准确传入装配、循环和归档；Raw 快照不遗漏限时或记录错误的默认值。
- 相对/绝对录制目录保持原解析结果；`--no-record` 不启动相机、不创建 recorder。
- 用假设备/假时钟检查到达最大时长仍走原来的正常结束路径；保存、丢弃和失败处理不改变。
- 启用录制时空目录、非正/非有限时长仍被拒绝；现有手部禁用规则仍生效。
- 运行代码、导入和当前示例不再依赖 `CollectionConfig` 或 `runtime.collection`；归档 JSON 的 `collection` 键与历史说明不按词匹配删除。

### T4. VR 运行读取与离线标定报告解耦

入口文件：`dexmani_real/utils/vr_heading.py`、`dexmani_real/teleop/run.py`；核对 `scripts/calibrate_vr_heading.py` 和录制旋转元数据。

实施要求：

1. `load_vr_transform()` 从 JSON 读取 `theta_deg`，保留有限性与 `[-180, 180]` 范围检查，直接返回原有语义的 3×3 旋转矩阵。优先使用现有简单函数，不增加运行期标定类、报告 parser 或 schema 层。
2. 删除运行读取路径对 `ref`、`quality.std_deg`、`quality.max_deviation_deg`、`quality.frames` 和 `grade` 的强制依赖。最小 `{"theta_deg": 30.0}` 应可使用；已有完整标定 JSON 仍可使用，额外报告字段不参与运行准入。
3. 更新唯一运行调用方；如 `HeadingCalibration.transform` 此后确无消费者，直接删除。旋转仍为 `R_z(-theta)`，继续将 VR FLU 前向对齐至机器人基座 +X。
4. 保留标定采样、去重、样本数、稳健估计、质量输出及 `save_vr_transform()` 对低质量结果的拒绝。`head/wrist` 仍是标定工具的有效选择，不删除它们。
5. 缺失/非法 JSON、缺失或无效角度仍明确失败；不得补零角度或回退单位矩阵。不要为可选历史报告新增警告、兜底或修复流程。
6. Raw 继续记录实际 `vr_to_robot_rotation`。不伪造未知的标定样本数/质量信息，也不精简现有标定输出文件的报告内容。

明确行为变化：运行端不再依据历史质量字段拒绝一个数值有效的角度；自动标定仍须在保存时通过现有质量把关。手工角度的正确性由研究者决定，这一变化须写入 README。

验收：

- 同一角度的旧完整 JSON 和最小 JSON 得到相同矩阵；检查 0、正负 90、正负 180 度及一般角度，方向、正交性和行列式正确。
- 非有限、越界、缺失角度被拒绝；合法角度不因缺少报告或历史 `poor` 报告被运行 loader 拒绝。
- 标定工具仍拒绝保存 `poor` 的自动估计结果，仍输出 ref、质量及时间等原有信息。
- 运行端及 Raw 记录使用同一个实际旋转；不把格式放宽误写为物理朝向已经验证。

### T5. 删除 `collection_source` 的内部枚举复检

入口文件：`dexmani_real/data/recorder.py`；核对 `teleop/loop.py`、`deployment/rollout.py`、`data/raw.py`、`replay/trajectory.py`。

实施要求：再次确认当前仓内来源只有两个固定字面量调用后，删除 `COLLECTION_SOURCES` 及 `EpisodeRecorder.start()` 的成员资格复检。保留参数、元数据写入和全部实际消费者；不要增加 enum、Literal 包装类或运行期来源注册表。

验收：teleop 仍写入原来的 VR 列；policy rollout 仍被正确标为 `policy_rollout`；物理回放仍拒绝非 teleop 来源。不要删除录制器的生命周期、目录所有权、保存/丢弃和资源收尾逻辑。

### T6. 删除 `save_points()` 的内部重复转换和扫描

入口文件：`third_party/cordvip_geometry/src/cordvip_geometry/geometry.py`、`third_party/cordvip_geometry/src/cordvip_geometry/cli/prepare_object.py`。

实施要求：

1. 确认两处调用仍直接接收 `sample_mesh_surface()` 生成的有限、非空、float32 `(N, 3)` 点和法线。
2. 删除 `save_points()` 内重复的 `validate_points`、float64/float32 往返及非空复检；保留简单的独占文件创建和 `np.save(..., allow_pickle=False)`。在 docstring 交代它保存已生成的采样资产即可。
3. 保留 `sample_mesh_surface()` 的外部输入/索引/面积检查以及计算后的 float32 溢出检查；保留外部 `load_points()` 和 mesh 读取边界。不全局删除 `validate_points` 或 `float32_points`。

验收：固定种子和几何的 points/normals，前后 NPY 读出的 dtype、shape、顺序和数值一致；已有文件仍拒绝覆盖。无需为这几行保存代码新增通用 I/O 工具或持久化框架。

### T7. 仅移除 retarget 的已覆盖静态复检

入口文件：`dexmani_real/teleop/retargeting/adaptive.py`、`dexmani_real/teleop/retargeting/landmarks.py`；沿 `teleop/run.py`、`teleop/vr.py`、`sensor/vr.py`、`runtime/observation.py` 和手部反馈链确认前提。

|候选重复检查|现有保证来自哪里|本轮保持的职责|
|---|---|---|
|构造中的 `hand_config.validate_home_pose()`|`run_teleop` 在构造 retargeter 前执行机器人配置检查|retarget 配置自身检查、URDF 限位和 frame 检查|
|构造中的 `control_hz` 有限/正值检查|运行入口的 `validate_control_rate()`|现有速率上限、名义周期和实际 dt 的含义|
|`reset()` 的 qpos shape/finite 扫描|已准入设备反馈和固定形状数据通路|操作限位 clip、独立状态副本、时间与初始化语义|
|`prepare_landmarks()` 的 `(21,3)`/finite 扫描|VR 接收转换边界；覆盖实时与启动标定两条调用链|需要的 dtype 转换、骨长、掌退化及新计算结果的正确性|

实施要求：

- 逐项确认所有实际入口都经过该保证，且中间没有使它失效的变换或可变别名，再删除对应扫描。只要某一项前提不成立，就保留该项并说明真实入口，继续其他任务；不为凑删除数量新建准入层。
- 保留标定骨长漂移、退化几何、优化器失败/非有限输出、损失接受、速度约束、平滑、时间递增、reset/commit 和失败不提交状态的机制。
- 保留输入转换及必要复制。删除校验不等于删除 dtype 约定、操作范围投影或数组所有权。
- 不新增 `TrustedArray`、validated 标志、另一条 fast path 或 checked/unchecked 双 API；不重写 retarget 算法，不裁掉现有诊断输出。

验收：已准入的固定 landmarks/qpos 输入在原实现与新实现下得到相同预处理/重置结果；骨长和退化输入仍由实际业务检查拒绝；VR/设备边界仍拒绝非法外部数据；reset 后修改源数组不改变内部状态。优化器可用时做确定输入的行为对照，不把时间上限驱动的迭代数变化误判为必须 bitwise 相同，也不为了通过比较放宽求解接受条件。

## 5. 不变的边界与数据事实

- `source_camera_sequence`、源时间戳和序列用于相机配对、新鲜度、去重、Raw、导出及回放；缓存被读取不能刷新来源年龄。
- `collection_source`、实际 tactile source、标定变换、训练数值 recipe 和控制周期仍有消费者。Canonical `source_id` 在配套 Policy 中可缺省，但提供时参与身份比较；“可缺省”不等于“无用”。本轮不改数据格式或配套 Policy。
- `Robot.validate_command` 与 XHand 公共发送入口不是当前组合路径上的重复检查；跨进程授权、迟到任务、run ID、反馈期限、停稳和轨迹制动检查不属于静态复检。
- EEF 工作空间与点云裁剪范围、机械限位与操作限位、桌面碰撞间隙与点云裁剪 margin 各有不同含义，不合并。
- 保留前轮已经实现的 NPZ 文件所有权和首错语义、I/O 延迟导入、显示功能清理及 model Z 数值结果。不要在本轮改写这些已验收模块。
- 不重写 `PolicyWorker`、`ObservationBuilder`、控制状态机、HOME 框架或导出 I/O 管线；现有证据不支持为本轮清理承担这些迁移成本。

## 6. 明确暂缓，执行器不得顺手实施

|暂缓项目|涉及参数或机制|暂缓原因|
|---|---|---|
|仅保留 VR 遥操作|`KeyboardTeleopConfig`、键盘点动入口、`CartesianJog`|会减少精细点动能力；共享按键监听与 HOME 更不能误删|
|仅保留 TCP server|`VRConfig.transport`、UDP/TCP client、client 重连|会缩小 VR 连接方式|
|仅保留串口 XHand|`HandConfig.comm_type`、`ethercat_slave_position`、EtherCAT 生命周期|会退出一种硬件通信能力|
|HOME 强制已标定桌面|table `enabled`、`table_z_surface_m`、`hand_safety_margin_m` 及 fallback|会改变无桌面标定时的运行能力|
|仅支持轴对齐盒体|`StaticCollisionBox.quat_wxyz`|会减少障碍物表达能力，与自动命名无关|
|固定相机型号或 L515 recipe|两项 `l515_*` 与型号分支|需要先决定真实设备支持范围|
|普通发送首次 CRC 即失败|`HandConfig.send_crc_limit` 及容忍状态|会改变噪声串口可用性；停止阶段有界尝试另有职责|
|固定 checkpoint 动作窗口|`ExecutionConfig.action_steps`|会退出执行窗口实验；sync/async/RTC 也保留|
|固定去桌面及 hand_1024|`PointCloudConfig.remove_table`、`HandCloudConfig.num_points`|涉及数值 recipe 和已有 checkpoint|
|HOME 只保留单候选|候选生成、排序、拒绝聚合|会降低自动回零的成功覆盖范围|
|删除离线分析或质量报告|时序分析、retarget diagnostics、calibration_capture、table.fit、episode_quality 等|没有程序 reader 不能证明没有研究或人工阅读价值|
|改造导出 I/O 或控制架构|单遍导出、worker/builder/框架重写|需要独立证据与 profiling，不属于本轮|

固定参数也暂缓：retarget 的 `calibration_samples`、`calibration_timeout_s`、`calibration_max_relative_mad`、`bone_length_tolerance`、`max_evaluations`、`ftol_abs`，手部 `ki/kd`，桌面尺寸/厚度/接触链接以及 `eef_workspace_margin_m`。它们不是死字段，固化会减少调参入口；本轮不把参数批量搬到新常量文件或隐藏 preset。损失权重、物理约束、反馈期限及其他研究参数同样保持现状。

特别注意：`ftol_abs` 同时参与优化停止与解接受；arm 默认反馈率 25 Hz 与 `feedback_max_age_s=4/30` 不能等价改写成 `4/feedback_hz`。

## 7. 实施纪律与验证成本

1. 开始先读 `AGENTS.md`、检查 `git status` 和当前 HEAD；保留用户未提交改动，不 reset/revert。只沿任务涉及的入口与调用方检索，无需重新做全仓词频审计或外部项目横向评测。
2. 已被当前代码解决的子项标注“已满足”后跳过；前提变化则用代码位置解释并收窄该项，继续不受影响的任务。无需因普通实现细节再次询问。
3. 优先原地删除和直接参数传递。完整迁移调用、导入与当前说明，不留仅供旧接口过渡的兼容层；保留有独立职责的小函数，不按函数行数机械内联。
4. 不新增依赖、注册器、配置框架、验证框架、异常层、诊断聚合平台或协议版本；不要为本轮生成另一份庞大盘点文档。仅更新必要的 README/现有说明及本任务书的简短完成状态。
5. 验证按风险分配：协议用轻量端点；CLI/装配/限时用 fake；数值用固定输入；设备和 GPU 不可用时明确区分静态核查、stub 检查与真实执行。只做离线验证，不连接真机或执行机器人运动。
6. 不为简单删字段、删复检增加逐行映射实现的永久测试；较有价值的协议/行为回归可复用现有测试。一次性故障注入、基线副本、输出和临时测试放仓库外临时目录，不新建测试平台或恢复已删除的 `experiments/`、测试/证据体系。
7. 有现成相关检查则执行。禁用仓内 `.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/`；编译或构建产物也放仓库外。检查已经通过后不反复重跑，除非后续修改影响其结论。
8. 最后检查差异和新增复杂度。不执行自动 push、merge 或创建 PR；本任务的交付是本地代码、必要说明和准确的验证结果。

## 8. 完成标准与交付格式

- [ ] T1 回显参数在客户端、worker 和调用方完整退出，真实帧身份与 mask 边界保持。
- [ ] T2 多障碍物无需手工命名，几何和碰撞职责保持。
- [ ] T3 CLI 是录制三项选项的唯一输入来源；运行、打印与归档一致，无替代配置容器。
- [ ] T4 最小角度 JSON 与完整标定文件都可用，旋转和自动标定质量把关保持。
- [ ] T5 来源枚举复检退出，来源字段及 VR/回放消费者保持。
- [ ] T6 保存不重复扫描，采样结果、外部读取检查及拒绝覆盖保持。
- [ ] T7 仅删除调用链前提已证实的静态复检，其他候选有明确保留理由。
- [ ] 第 6 节暂缓项目未被修改；没有新增框架、硬件操作或未经证实的性能结论。
- [ ] 实际运行的相关检查通过，未运行的检查与原因单独列明，diff 不含临时产物。

按基线预期，T2/T3 会删除 1 个配置 dataclass 和 5 个声明字段（盒体名 1、Collection 字段 3、RealConfig 组装字段 1），即 18→17、123→118；CLI 的三个选项仍保留。这是影响说明，不是要求以数量为指标扩删代码。主要收益是少一套命名准入、少一层配置合并、少一条回显链，而非保证吞吐量提高。

交付时用中文简要报告：

1. T1–T7 的完成/已满足/保留状态，及关键修改文件。
2. 实际删除的机制、接口迁移方式和上述明确行为变化。
3. 实际执行的命令/回归结果；缺依赖、模型或真机而未执行的项。
4. 对暂缓项未实施的确认，以及任何确实阻碍完成的当前源码事实。

不要把本任务书、模拟结果或静态检查写成真机验收通过；也不要用关键词减少数量替代行为验收。

## 9. 配套 Codex 执行 prompt

```text
请直接在 dexmani_real 中实施根目录 CLEANUP_TASKBOOK.md 的 T1–T7。先读 AGENTS.md、检查 git status 和当前 HEAD，再按任务书核对相关调用链并动手；基线为 23d3d9d962d6f0acee995850e2a8c1bc0b8c24e1，不回退后续改动。

这是个人研究仓库。本轮目标是删除 SAM frame_key 回显、自动生成障碍物名称、把录制三项选项收口到现有 CLI、解除 VR 运行读取对历史质量报告的强制依赖，并删除三处明确的内部复检。完整迁移调用方、导入、打印/Raw 配置快照和必要使用说明，不留下替代配置容器或兼容框架。

按 T1 → T2 → T3/T5 → T4 → T6/T7 执行。已实现的子项跳过；若某个删校验前提不成立，保留该项并给出实际调用证据，继续完成其余任务，不为凑数量扩删。Collection 默认仍为录制开启、episodes、60 秒；所有运行能力与研究参数保持，只有任务书列明的接口/使用契约变化允许发生。

严格暂缓第 6 节全部能力裁剪和参数固化，不删除键盘点动、其他传输、EtherCAT、桌面 fallback、旋转盒体、CRC 容忍、动作窗口、点云 recipe、HOME 多候选或研究诊断输出。保留真实来源/时间/序列、跨进程授权、运动约束、文件所有权和首错语义；不重写 PolicyWorker、ObservationBuilder、控制框架或导出管线。

每组修改做针对性离线验证，临时基线、fake 和故障注入放仓库外。不要新增依赖、测试平台、验证/诊断框架；禁止仓内工具缓存。不要连接真机，不自动 push、merge 或创建 PR。持续完成已授权范围，不停在计划；最后检查 diff，用中文报告 T1–T7 状态、实际删除、迁移方式、已执行验证和未验证限制。不得把静态或 fake 结果称为真机/模型验证通过。
```
