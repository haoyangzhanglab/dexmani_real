# DexMani 跨仓库部署与数据机制简化任务书

## 1. 目标、范围与执行基线

项目定位为个人机器人学习研究。将策略部署、推理、数据导出及相关几何工具收敛为轻而薄、简洁好用、直观高效的实现。先删除无用途机制，再合并真正重复的处理；允许合理研究输入和消融配置，不建立生产平台式准入体系。

本任务书统一覆盖两个仓库，保存在 `dexmani_real` 根目录，不在两个仓库复制维护两份方案。

审查基线（2026-11-04）：

- `haoyangzhanglab/dexmani_real`：`1456338a84f6fdfbdcd3351b4138e029927986fa`。
- `haoyangzhanglab/dexmani_policy`：`968bff9fb28d2e53eb5e059f8a33ad330cb3baae`。

实施时读取两仓最新代码和适用的 `AGENTS.md`，检查并保留已有工作区改动。若基线之后已有等价修复，验收后跳过对应修改，不覆盖新实现，不为复现旧方案回退代码。本文件中的任务尚未实施，不代表验收已通过。

**关键要求：新策略只要使用已有观测字段、动作表示和统一预测接口，就不需要修改 Real 的部署、导出或执行主链。** 模态依赖、joint/eef 表示、sync/async/RTC 调度确有不同，允许相应直接分支；编码器、损失、辅助预测和解码算法差异留在策略内部。不要把“无策略专属部署 case”解释成删除真实算法差异，或为全仓零分支建立注册表与插件体系。

本轮修改源码、必要配置、对应文档和少量针对性验证。无需新建共享仓库、通用 contract 包、RPC 服务、部署 bundle、schema 版本体系、来源证明、哈希指纹、审查报告生成器、兼容适配器矩阵或庞大测试设施。不要增加 `strict`、`unsafe`、`ignore_contract`、`allow_missing_metadata` 等开关来保留本应删除的规则。

## 2. 最近两轮审查的合并决策

| 问题或结论 | 最终处置 | 对应任务 |
| --- | --- | --- |
| resume 更换数据路径后继续保存旧 deployment_metadata | 必须修复，刷新实际数据派生配置，保留恢复的权重和仿射 | A |
| 强制 inspect 后再 load，派生 policy_metadata 重复保存 | 收敛为一次加载，模型直接提供派生信息 | A |
| Runtime 与 SimRunner 重复处理观测和动作输出 | 共享普通函数，保留各自历史与执行调度 | A |
| Policy 枚举 Real 的 sync/async 执行模式 | 收缩为实际 RTC 能力；不按策略名称判断 | A |
| 缺少附属部署 recipe 即拒绝有效 checkpoint | 按明确数值含义补缺省；不构建配置来源追踪 | B |
| 触觉禁止 identity、InterAct-Dex 编码器类名白名单、配置键精确相等 | 删除研究选择限制，同步修复实际消费者 | C |
| Canonical 固定导出 13 字段，纯 RGB/状态流程也要求 FK、外参和点云 | 一个 fields 入口，资源和检查按实际字段启用 | D |
| 跨 episode 强制同 task_name；累积 episode_quality/来源列表 | 删除标签门槛和无训练消费者的归档副本 | D |
| 来源时钟诊断作为整段导出的硬门槛 | 对实际使用来源给警告；保留原行序及基本控制时间含义 | D |
| 几何 provenance 经 worker/client/CLI 传递，仅用于报告 | 整条删除，连带删除报告与补写分支 | E |
| 选择目录附带 RGB-D 就触发整幅输入精确绑定 | 删除绑定；文件和目录统一应用于当前指定首帧 | E |
| RGBDFrame 来源必填、深度必须精确 float32、scene 拒绝备注 | 放宽输入，集中必要转换与实际几何检查 | E |
| headless 唯一候选仍必须提供 instance | 唯一候选自动选择；多候选需明确索引 | E |
| 重复有限性检查、log 后原样重抛、单文件写入的复杂异常仲裁 | 在确定实际入口和资源职责后收缩 | F |
| CordViP 缺少真实 object_pointcloud producer | 保留通用字段缺失错误，列为独立功能缺口，不伪装已支持 | §8 |
| 仿真与 Real 触觉同名同形状但单位/坐标不同 | 简短说明实际语义，不新增准入或认证机制 | §8 |

此前已落地的 PNG 调色板处理、候选选择输入减法、对象名限制收缩、按保存 affine 数值恢复触觉等改动，应保留并针对受影响路径检查，不重复重做。上轮“可选来源绑定存在时必须严格验证”的建议由本任务书取代，改为不参与准入。

## 3. 参考项目的采用边界

以下参考的是已核对实现，不将参考项目视为无缺陷标准，也不要求它们具有完全相同的机器人或数据含义。

| 项目与固定版本源码 | 采用的具体做法 | 不引入的机制 |
| --- | --- | --- |
| [Dexora：策略入口](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/dexora_policy.py#L143) | 观测字典进入策略，返回物理动作 chunk，直接加载模型 | 自动补截错误维度、宽松缺权重加载不作为本项目默认行为 |
| [ManiUniCon：模型执行](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py#L249) | 包装围绕真实观测、动作和 RTC 前缀；算法与执行职责清楚 | 不复制过期动作重新计时等会改变现有执行语义的兜底 |
| [pi-r2-flow：模态配置](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/mindex/policy/groot_client.py#L124) | 依据实际模态和窗口构造输入，配置服务于使用者 | 双服务器、网络协议、多版本兼容和缓存体系 |
| [LeRobot：数据创建](https://github.com/huggingface/lerobot/blob/6e3b67c5f0b77c93a5179d88d7c1d6a5f7ff7b27/src/lerobot/datasets/lerobot_dataset.py#L767) | features 决定实际数据字段；公共预测/处理职责可以复用 | 面向平台的完整 contracts、元数据和发布体系 |
| [T-Rex：模型加载](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/scripts/test.py#L153) | 训练记录辅助恢复默认设置；实际结构、processor 和权重明确加载 | 不把其慢快双路算法和 KV 缓存变成所有策略的必备部署层 |

统一边界使用现有 Python 对象与字典，不设计新的声明语言。接口只需表达实际观测字段及历史长度、动作表示与完整未来窗口、必要预处理和 RTC 能力。模型名称仅用于构造、显示和实验标记，不能成为 Real 主链的行为开关。

## 4. 任务 A：统一加载与推理，修复数据派生配置失效

主要位置：Policy 的 `deployment/{metadata,runtime}.py`、`common/checkpoint.py`、`agents/policies/base.py`、`env_runner/sim_runner.py`、`training/common/{config,data,resume}.py`、`training/trainer.py`；Real 的 `deployment/{run,session,inference,observation,rollout,preflight}.py`。

### A1. 一次加载得到 runtime

- 将正常部署入口收敛为 `load_policy(path, *, weights=None, inference_steps=None, device=..., seed=...) -> PolicyRuntime`。runtime 携带现有小型 `.info`，无必要不增加容器类型。
- 一次读取 checkpoint，选择实际 raw/EMA 权重，尽早释放未用训练状态与另一份权重，复用既有模型恢复逻辑。
- 从恢复后的 agent 取得观测维度、窗口、控制动作维度及算法能力；从实际保存配置取得 action_key 和必要图像预处理。动作辅助维度仍由模型内部裁剪。
- 停止保存/读取纯复制用的 `policy_metadata`，删除仅为它存在的 `inference_metadata`。旧 checkpoint 多余键自然忽略；不建立迁移版本、转换脚本或格式探测矩阵。
- 删除部署必须先调用 `inspect_policy` 的流程，同步所有实际调用方与导出。若现有真实独立检查命令仍需要查看信息，复用同一读取逻辑，不保留第二条正常加载管线。
- 用一个 `supports_rtc` 属性表达真实支持能力，替代各策略重复声明 sync/async/rtc 元组；sync/async 由执行端负责。能力必须与实际解码实现一致，不以捕获 `TypeError` 或试错调用探测。

### A2. 同步调整 Real 启动顺序

保留现有单模型工作线程负责 load、warmup、predict、close。加载 Future 可返回小型 info；加载完成后再解析 Real 有效配置、创建 ObservationBuilder、计算 capture_needs，之后连接设备。启动打印和录制设置随之移动。观测回调须在首次 warmup/predict 前就绪。

保持加载时键盘取消与急停响应、原有资源所有权和退出路径。不增加第二个模型管理者、新启动状态机或 GPU 对象跨线程管理层。允许部分配置问题在模型加载后、设备连接前被发现；本调整的收益是减少重复事实和流程，不宣称未经测量的性能提升。

### A3. 仿真与真机复用数值处理

- 将两处重复观测准备和预测结果转换收敛为 Policy 仓库内少量普通函数。使用字段类别、实际维度和已有模型属性，不写 DP、R3D、SAT 等策略名称 case。
- RGB 继续用已有 `preprocess_rgb_eval`；历史切片、batch、设备转换和点云通道选择各有一处清晰负责者。不要为统一而把实时 ring 与仿真环境封装进同一接口框架。
- 公共结果为完整未来物理动作窗口，连续 NumPy `float32 [P, D_control]`。Sim 由调用方取执行前缀，Real 保留现有 sync/async/RTC 调度。模型内部的归一化、R3D 辅助预测裁剪、AdapTac patch 零参考处理等保持原数学含义。
- 在公共输出边界检查实际 shape 和 finite。确认所有预测入口都经过该处后，删除 Real 中相同数据的重复扫描；实际映射、限位和设备提交检查保留。
- 保留 RTC 物理前缀的数值含义和坐标空间、warmup RNG 恢复、episode reset、async 过期窗口处理及 RTC 按成功提交推进的语义。

### A4. 修复 resume 后配置过期

`prepare_resume` 允许更换 `zarr_path`；`build_dataset` 已读取新数据，但 Trainer resume 路径目前沿用旧 deployment_metadata。fresh/resume 均应保存当前实际数据派生的部署设置。恢复的模型、EMA、normalizer affine、optimizer、scheduler 和 RNG 保持不变，不重新拟合仿射，不通过源路径/指纹禁止研究者换数据。只刷新确实来自当前数据的设置，保留恢复模型的有效结构参数。

## 5. 任务 B：必要部署参数补缺省，不建立准入证书

主要位置：Real 的 `config/deployment.py`、`deployment/{preflight,observation,session,rollout}.py`，Policy info 构造。

`RealConfig` 是已填默认值的完整 Python dataclass，不记录哪些字段由用户显式设置。禁止据此发明“显式本地 > checkpoint > 默认”的通用来源追踪或递归合并系统。采用以下具体规则：

| 输入 | 解析行为 |
| --- | --- |
| 控制周期 | 增加一个明确的可选 `ExecutionConfig.control_dt_s`；有值则使用，否则使用 checkpoint 中保存值。两者都缺失时明确要求填写，不猜测 20/30 Hz |
| 点云/手点云处理 recipe | 保存值存在时沿用；缺失时使用已有 RealConfig 对应配置。只处理被请求的模态 |
| fingertip link 名称 | 保存值缺失时使用已有 XHand 固定常量，实际 FK 使用时检查必要链接 |
| 原始计数 depth | 保存的训练尺度与输入形状仍有数值含义；训练尺度缺失时明确报错。本轮不新增通用 override 字典，也不将当前相机尺度冒充训练尺度 |
| source、报告、历史 normalization 标签 | 不参与模型部署准入 |

控制周期解析完成后，Real 调度、预检、运行打印和 Raw 录制都读取同一个有效 execution 值；Policy info 中的保存值仅作为默认来源。打印一次实际使用的关键设置即可，无需另存“解析证据”。

保留已保存权重/affine 的真实数值检查；不从当前硬件猜测训练单位。覆盖已保存点云 recipe 的一般机制不在本轮扩展，避免为暂无明确调用需求增加参数体系。

## 6. 任务 C：允许研究配置，删除类别与集合门槛

主要位置：Policy `training/common/{config,data}.py`、`common/normalizer.py`、`agents/policies/{interact_dex,adaptac}.py`。

1. 触觉 normalization 允许 `identity`，默认仍沿用原配置。为实际使用的触觉字段构造正确形状的单位 affine（ones/zeros），不能只删除白名单。保存、raw/EMA 恢复及直接读取 scale/offset 的损失路径均应一致；缺失 affine 不静默补 identity。
2. 删除 InterAct-Dex 编码器完整类名检查。使用构造函数现有的输入字段、token 数和输出维度要求；满足实际用途且配置可重建的自定义编码器允许使用。
3. normalization 可以包含未使用字段。先按实际 `sensor_modalities + action` 得到有效 spec，再校验/拟合；额外字段不访问数据、不影响触觉字段集合判断。实际使用字段的配置缺失仍明确报错。
4. 需要物理几何或固定图像预处理的字段继续保持对应数值含义，不把任意 normalization 模式应用到投影坐标。
5. 不删除 AdapTac 中承担真实 patch 算法含义的分支；不为消除名称 case 重写全部预训练初始化或建立 callback/registry 框架。

## 7. 任务 D：按实际字段导出数据，取消附属准入

主要位置：Real `data/export.py`、`scripts/export_zarr.py`，并核对 Policy `datasets` 的实际读取。

### D1. 一个字段入口与直接依赖

增加一个 `fields` 入口；缺省保留现有全 13 字段，显式选择时仅导出所选字段。fields 只控制 Canonical 导出，不改变完整 Raw 采集/录制 schema。入口验证字段非空且受支持，不提供按策略名命名的导出 preset，不导入 Policy/Hydra 推断导出任务。

| 输出字段 | 实际数据/计算依赖 |
| --- | --- |
| joint_state | arm_qpos、hand_qpos |
| action | arm/hand joint target |
| action_ee | arm/hand joint target、ArmFK |
| eef_pose | arm_qpos、ArmFK |
| fingertip_xyz | arm/hand qpos、ArmFK、HandFK |
| hand_pointcloud | arm/hand qpos、FK、手部采样模板 |
| finger_tactile_force | hand_contact |
| taxel_tactile_force | hand_tactile_force |
| rgb | RGB 视频及实际尺寸 |
| depth | 深度及相机尺度/尺寸 |
| camera_intrinsic | 颜色相机内参 |
| camera_extrinsic | 实际外参 |
| point_cloud | RGB、深度、内外参及所用点云处理设置 |

使用少量静态字段集合和普通条件计算 needs，不写通用依赖图引擎。复用既有几何纯函数；实时采样和离线文件读取仍各自清晰。

同步修改 `validate_export_episode`、资源创建、`iter_canonical_blocks`、`create_store` 和 `append_episode`，不是只给最后输出字典加过滤：

- 只读取、检查和计算所选字段的依赖；仅 RGB/状态/action 导出无需外参、FK、手点云或桌面平面；状态/action 导出不打开 RGB 视频。
- 去掉 `append_episode` 对 `block['joint_state']` 的硬依赖，按实际非空 block 确定行数并保持字段行数一致。
- 只在导出原始 depth 时要求各段 raw depth scale 一致；point_cloud 已逐段转换为米，不应被未保存的原始计数尺度差异拒绝。
- 只对实际保存的图像数组要求一致 shape；只保存实际字段需要的数值配置。不要为了删除 metadata 同时删除 episode 分界或单位。
- 默认全字段的字段名、单位、数值含义与 episode 划分保持一致。

### D2. 数据选择与诊断放宽

- 删除 uniform task_name 检查及无训练消费者的 Canonical task_name 属性。保留 Raw 的实际任务记录；合并策略由实验者决定，不新增多任务架构。
- 删除 episode_quality 的构建、累积、attrs 和返回副本，以及仅用于重复来源归档且无实际消费者的 episode_ids；不新增 manifest 替代它们。
- 保留 `episode_ends` 和实际采样 `dt`。当前序列窗口仍使用统一 dt，不在本任务增加混频重采样。
- 对所用来源时间戳非正、超过 observation、倒退及 camera_sequence 异常改为一次有定位信息的诊断 warning。control/observation 非正、非递增或 observation 早于 control 继续报错。只检查当前字段确实使用的动态来源；仅导出静态 camera_intrinsic/camera_extrinsic 不要求逐帧相机来源时间有效。
- 保留原行序，不自动删行、补帧、改时间、补零或伪装成功。所选字段缺失、非有限训练值、实际 RGB 帧数损坏继续失败。未选触觉的坏数值不能阻断 RGB/状态导出。
- 不因此扩大 Raw 格式兼容范围：现有 timing/rows 等原始文件结构仍按已有读取方式使用，不新增 schema 迁移。

## 8. 已知功能边界：不以特殊 case 掩盖缺口

### CordViP object_pointcloud

当前 Real 缺少 `object_pointcloud` producer，属于明确的能力缺口，不是本轮重构引入的回归。本次不得宣称所有策略已无条件可真机部署，也不把接入完整新感知链作为本次减法的前置条件。

继续保留通用“请求字段没有 producer”的错误。未来真实接入时，围绕 `object_pointcloud` 字段使用 `SceneTracker.model_points` 等内存结果，明确完整模板、坐标变换、点数、颜色需求、多目标选择和跟踪有效性；不能将可见 scene cloud 改名冒充完整模板，不经 NPZ/报告回读，不增加 CordViP 专属 runner/adapter。

### 触觉 sim-to-real

同名同形状不证明物理含义相同。Real 的聚合/稠密 SDK 通道、传感器坐标和原生尺度，与仿真接触力 N 需明确区分；不能假设 dense 求和就是 aggregate。为现有文档保留简短说明即可。同源 Real 训练/部署无需新增校准证书或来源准入；跨域实验有真实变换需求时显式实现数值变换，不伪造等价关系。

## 9. 任务 E：几何工具删除来源链，放宽离线使用

主要位置：`third_party/cordvip_geometry/src/cordvip_geometry/` 下 `sam2.py`、`sam2_worker.py`、`foundationpose.py`、`rgbd.py`、`tracking.py`、`io.py`、`reconstruction.py` 及 `cli/{arguments,object,scene,selection,track,view}.py`。

### E1. 删除只服务归档的跨层状态

- 删除 SAM2 worker 构造与更新 provenance、client 保存和更新 provenance、推理回复传输 provenance、PoseTracker 的 provenance 和 CLI model_sources/models.json 链。启动握手保留最小就绪/错误信息，现有进程隔离和必要错误传播保留。
- 删除默认生成 segmentation_request/result.json、generation_request/result.json；参数和耗时直接打印。生成函数返回实际结果路径，清理只为报告存在的返回字段、成员和统计。
- 删除分割输入原照片及 source_rgbd 自动复制。保留重选实际需要的 rgb.png、candidates.npz，以及有直接用途的 mask、RGBA、预览和 mesh。
- 删除 models.json 的失败补写及对应 reports 动态分支，同时删除 run.json/result.json 及专门为报告维护的 failure 状态和异常框架。进度、处理数量、中断和失败由终端、异常及退出码体现，资源继续由已有上下文管理。
- 解释结果所需的单位、坐标、hand_link_ids 对应 link_names、关节顺序等，在现有 scene.json 中一次保存最小说明（hand 信息仅在实际输出手部数组时保存）；这些映射不能跟 source_files/raw_notes 一起删除。viewer 颜色复用一处固定调色板，取消 run.json 依赖。不要在 scene.json 重建完整 args、模型来源和运行归档。
- 保留 viewer 切换模型点云需要的 model_templates 和实际几何数组；保留候选分数的可视化用途。不是按文件名包含 metadata/artifact 就删除。

### E2. mask 只有一种应用语义

PNG、NPY 和选择目录均解析为应用于当前指定首帧原始 RGB 网格的 mask。删除 `load_initial_selection` 中 RGB/depth/K/distortion 整体精确绑定，删除 object select 中附带 RGB-D 比较和复制；已有目录即使存在该附属文件，也不参与接纳。

保留调色板 PNG 按实际颜色转灰度、尺寸对应、必要前景/深度以及一次去畸变的规则。不重新缩放 mask 以掩盖网格错误，不增加 ignore_source 开关。既有目录只需解析 mask_file，直接文件不要求额外 JSON。

### E3. RGB-D 与场景输入减法

- RGB、米制 depth、K 和实际畸变信息保留。接受常见浮点深度精度，在入口转为 float32 后集中检查形状、有限性和非负；不要把毫米或 raw counts 自动当米。
- source_id、timestamp_ns、clock 为可选说明；缺失时不伪造值。row_index 在普通帧中可缺省，但 RawHandCloud.for_row 的真实行索引仍必须可用。RawHandCloud 的 timing_available 根据该行实际时间数据判断，不再由说明用 clock 字符串决定。
- 移除 tracker 对说明时钟的统一性/递增要求；按调用顺序推进。不能保留 `(None, None)` 去重导致后续帧全被跳过。实时去重如有实际调用需求，放在相机输入处依据 ring sequence；Raw 每行继续处理。
- save_frame/scene_arrays 写 NPZ 时省略缺失说明字段，不能将 None 序列化成 object array。读取继续使用 allow_pickle=False。viewer 同步允许缺失来源说明和文件名不等于说明行号；Raw 输出仍可沿用原行号文件名。
- 保留 tracker 实际缓存依赖的固定图像网格与相机标定；不为放宽来源而破坏去畸变和跟踪状态。
- scene JSON 仅解析实际使用字段，允许附加 note 等未消费信息；保留所需对象 ID、mask/prompt 和重复对象处理。当前 role 没有下游行为消费者，将其视为普通可选注释，删除枚举门槛与无用途的默认值生成。不要新增 schema 校验器。
- object segment/select/build 的 headless 唯一候选且无 instance 时返回 0；多候选无明确索引时给直接错误。将此决策放在候选已知处，移除 argparse 无条件要求 instance；保留零候选及索引越界错误。本轮不扩展 tracking 的 headless prompt 自动选择或多物体自动匹配。

## 10. 任务 F：校验与异常处理收敛

只在真实外部输入、数值转换或副作用边界保留有用途的检查；内部已建立且未被改变的不变量不层层重查。删除前确认所有入口确实被覆盖，不能靠假设替代调用链检查。

- 删除中间层仅 log 后原样 raise 的捕获；有 episode/path 上下文的单次错误说明可保留。
- 单文件 NPZ 写入使用普通上下文管理和简短失败清理，保留不覆盖已有文件、只清理本次创建的半成品。独占 open 失败不能进入删除目标文件的清理分支，尤其不能在 FileExistsError 后 unlink 旧文件。不要为“保存异常 + 关闭异常 + 删除异常”建立手工状态和仲裁框架；清理失败保留异常链，不吞掉实际 I/O 失败。
- 随报告、来源绑定和重复 metadata 删除对应校验、捕获、状态及无用参数，不遗留空壳 helper。
- 保留机器人停止、设备释放、writer 收尾、模型线程退出的真实资源职责；不要套用单文件简化方案去取消独立资源的收尾。
- 保留实际动作维度/单位、非有限输出、关节限制、命令有效性、RTC 前缀与执行时间关系等真实数值或设备条件。
- EMA 的活动 BatchNorm 限制及不等微批次梯度累积目前涉及真实算法实现；本次不通过单删 raise 放开，也不顺带建立新训练框架。

## 11. 执行顺序与针对性验收

建议顺序：A4/C 与 E 中独立减法可先落地；完成 D；再整体完成 A1–A3/B 的加载生命周期调整；最后收敛 F、更新调用方和文档。每项以完整行为闭环为单位，不留下新旧双入口长期并存。

以下为行为验收范围，不要求逐条建立永久测试文件或测试平台。优先复用现有检查，对跨仓核心行为补最少必要验证；完成后停止扩展无关测试。

| 验收主题 | 必须确认的结果 |
| --- | --- |
| 无策略专属部署分支 | 同一字段/动作接口的新策略无需改 Real 主链；无按 policy_name/类名选择部署 adapter 的新增逻辑 |
| 单次加载与取消 | 正常启动只读一次 checkpoint；加载失败/取消不连接设备；load/predict/close 仍由现有工作线程负责 |
| 保存恢复 | raw/EMA 与旧 checkpoint 多余 metadata 均正常；仿射不重拟合；resume 使用新数据后保存当前部署设置 |
| 公共推理 | 同权重、同有效输入和 RNG 下，比对重构前后代表路径的物理输出，允许 float64→float32 的精度差异；再检查 Sim/Real 公共路径一致。覆盖 RGB、点云、触觉、joint/eef、辅助动作裁剪和 RTC 所需代表组合，不做所有策略笛卡尔积 |
| 数值布局 | 输出 contiguous float32，singleton 负 stride 输入仍可处理；缺字段、错误维度、非有限真实动作有明确失败 |
| RTC/执行 | async 窗口、过期预测、取消、RTC 原物理前缀、成功提交计数和 warmup RNG 行为无回归 |
| 部署缺省 | 缺点云 recipe 可用现有 Real 配置；缺控制周期可明确填写且录制/调度一致；未知训练 depth scale 不被当前相机冒充 |
| 研究配置 | 触觉 identity 建立完整 affine 且保存恢复一致；兼容自定义触觉编码器可构建；未用 normalization 键不触发数据读取 |
| 按需导出 | RGB+状态+action 无外参/FK 可导出；状态+action 不解码视频；仅选点云才需要其几何条件；未选字段坏值不阻断；episode_ends 正确；原始 depth scale 不同但只导出米制点云可合并，同时保存 raw depth 仍拒绝尺度不一致 |
| 导出放宽 | 不同任务标签可合并；所用来源时钟异常给警告；基本控制时间损坏、所选 NaN action、RGB 缺帧仍失败；默认全字段数值保持 |
| 几何输入 | 同一 mask 的文件/目录路径行为一致；改变 mask 外像素不再因绑定拒绝；float64 米制深度和缺来源说明均可读取 |
| 几何连续处理 | 无时间字段的连续不同帧不会被误去重；Raw 同相机帧的不同数据行仍逐行处理；行索引仍可用于 hand FK；NPZ 无 None object array；viewer 能读精简结果 |
| 选择及输出 | palette PNG、object 唯一/多候选 headless、有效对象名、scene note；无 run/models/result 报告时仍可重选、预览及按 v 显示模型模板；手部索引映射可解释 |
| 异常收尾 | 无重复报告补写链；重要 I/O 失败透明；输出失败不会误删既有文件；硬件/录制收尾不被削弱 |

审查阶段已经用真实轻量函数复现：mask 目录绑定差异、RGB-D 来源必填、float64 深度被拒绝、scene 附加键被拒绝。其他训练/完整推理发现基于源码；审查环境缺少 Torch、Zarr、h5py 等依赖，未运行 GPU checkpoint、完整训练或真机验证。实施者必须分别报告实际执行的验证与未执行项；现有 core_checks 直接预测 smoke 不能自动证明 Real admission、控制周期或硬件部署成功。

按仓库要求禁用 pytest/ruff/mypy 的仓库缓存。无需为本任务恢复已清理的测试目录、增加证据包或大量审查文档。只更新既有 README/相关用法说明，删除过时的强制来源、固定全模态、类名白名单、两阶段加载和归档依赖描述。

## 12. 完成交付标准

- 上述任务的生产者、消费者、配置、CLI、导入和文档一致，无只删检查却遗留必需状态的半改动。
- 删除的来源/报告机制真正消失，未以新名字、新配置或新框架重建。
- 继续使用清晰的字段依赖和策略内部算法封装；没有策略专属部署路径。
- 不以删除代码行数为指标，不放松会改变物理含义的必要条件；不把新功能缺口伪称完成。
- 最终简要列出完成项、真正删除的机制、保留理由、验证结果及限制。已有后续改动导致任务不再适用时，说明实际依据与处理结果，不机械照搬。
- 本任务书是执行依据，不要求把每次运行、每条验证或每个审查判断再写入 artifacts/evidence/source 文件。
