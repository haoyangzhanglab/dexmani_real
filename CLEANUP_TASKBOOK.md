# DexMani Real 个人研究版清理任务书

> 状态：正在实施；实际结果以第 12 节为准，未验证项不表示已通过。
>
> 目标：让一个人能读懂主流程、直接改实验、可靠采集和复现实验；优先删除重复机制与不需要的产品化工作流。
>
> 任务书最初随 `580ac65` 新增；实施提交、验证与限制记录于第 12 节。

## 1. 执行摘要与适用基线

采用 **“保留研究内核，缩短外围流程，明确资源归属”** 的清理路线。主线保留 xArm7 + XHand + RealSense + VR、遥操作、Canonical 导出、策略部署及物理回放；保留 sync / async / RTC。大幅收缩录制发布协议、跨层收尾、配置入口、交互便利、几何在线服务及默认诊断。

最优先的结构改动是：**一段实验一个顺序调用，一份录制由一个监督方决定发布，一个事实只保留一个权威来源。** 不以删行比例、类数或校验数量作为成功标准。

| 项目 | 已核实基线 / 范围 |
| --- | --- |
| 本仓 | [`cef156d0beff8d0bdaf1ca0f9d14cfd924f8ae63`](https://github.com/haoyangzhanglab/dexmani_real/commit/cef156d0beff8d0bdaf1ca0f9d14cfd924f8ae63)；2026-10-31 再核对 main 未变 |
| 配套策略仓 | [`dexmani_policy@4c4b76626ba44849115e1a96ea8af0080968527d`](https://github.com/haoyangzhanglab/dexmani_policy/commit/4c4b76626ba44849115e1a96ea8af0080968527d)；recipe、source_id、split/resume 存在真实消费者 |
| 审查规模 | 129 个 Python 文件、21,550 文本行，含空白和注释；其中 vendored geometry 32 个 Python 文件、6,054 行 |
| 配置审查 | 18 个 dataclass、123 个注解字段，含 RealConfig 的 13 个聚合字段；均有仓内静态消费链，不能直接认定为死参数 |
| 结论追踪 | 27 项合并结论 R01–R27；另有 5 项明确保留意见。本文第 11 节逐项安排去向 |
| 验证限制 | 已做静态调用链及跨仓消费核对；没有据此证明真实设备、GPU 模型、旧实验数据或性能等价 |

实现时以实际 checkout 为准。若 HEAD 已变化，只复核受影响的调用链和消费者，不重做一次泛化全仓审计，不回退他人的工作。

## 2. 已做出的设计选择

这些是本方案的默认目标，执行者不必把同一选择再次变成开放讨论。

| 事项 | 默认决定 | 明确的使用变化 |
| --- | --- | --- |
| 研究范围 | 保留现有主实验能力；不建设多机器人平台 | 不新增厂商适配、插件注册和通用机器人接口 |
| 生命周期 | 一会话加载一次设备和模型；会话内顺序运行多段 episode | 取消依赖外层轮询逐步完成的收尾；未收完不进入下一段 |
| 录制格式 | 保留 Raw HDF5 + MP4、Canonical Zarr、现有目录发布约定 | 不换成“大 NPZ”、在线 Zarr 或 LeRobot 格式；不新增 COMPLETE/manifest 发布协议 |
| 数据政策 | 保留现行有效前缀、失败末行处理、时钟准入与 source_id 语义 | 完整正常段准入是独立研究政策变更，不混进清理 |
| 配置输入 | 保留简单 dataclass；采用可信本地 Python 实验配置，撤掉本仓递归 YAML 合并与任意嵌套 override | 旧命令/实验配置一次迁移，不保留双配置系统 |
| 采集/策略会话交互 | B 开始、S 保存并结束本段、D 丢弃本段、Q 退出、Esc 急停；HOME 保留 | 退出 C 暂停/续写、双 Q 确认、音频播放生命周期；物理 replay 保留其 S 停止、H 回零、Q 退出的直接语义 |
| Q 与异常保存 | Q 结束当前段并按现有正常退出语义保存；D 才显式丢弃。故障段能完整封存时保存 failed Raw | 保存不等于训练准入；不会把写入失败的临时目录发布 |
| 几何主工作流 | 默认已录制 Raw 逐行离线处理，初始选择、追踪、查看分阶段 | 退出 latest 持续服务、实时相机追踪入口和按墙钟跳帧预览编排；保留离线结果 viewer、初始 mask 的人工确认 |
| 几何研究能力 | 保留单/多物体、无 mesh 的语义分割、实测点与模型点、手 FK、pose-valid | 删除服务协议不等于删除研究结果；法向输出暂保留 |
| 标定 | 头朝向工具直接读 VR；相机标定拆为只读采样与离线求解 | 相机标定不再内嵌 jog/HOME；通过厂商示教或已有独立 jog 工具摆位 |
| 诊断 | 默认简洁终端输出、一个结束结果、必要来源事实；详细性能分析显式运行 | 不默认生成多进程各自的 DEBUG 文件，不让可选身份/环境查询决定业务成功 |
| 工程投入 | 小而有意义的回归验证；复用现有代码；不搭新基础设施 | 不加通用 scheduler、validator、event bus、registry、telemetry 或兼容框架 |

Python 配置的选择是针对个人研究定位的设计决定，并非“YAML 解析的约 63 行都很复杂”的事实判断。默认移除的 live geometry、暂停和音频属于能力/便利裁剪，应在 README 明说，不宣称它们从未有用。

## 3. 参考项目：只借鉴有依据的做法

以下链接固定到本次实际查看的 commit。这里只评估指定入口与示例，不把任何一个仓库整体认证为“简单”或“最佳实践”，不引入其依赖，也不复制代码。

| 项目 / 已读入口 | 观察到的做法 | 本项目采用什么 | 本项目不采用什么 |
| --- | --- | --- | --- |
| [LeRobot 录制示例](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/examples/so100_to_so100_EE/record.py)、[Robot 接口](https://github.com/huggingface/lerobot/blob/3f797de5952432284a7c77ad2760218cb74e2f26/src/lerobot/robots/robot.py) | 示例直接构造设备配置，外层循环运行一段后保存；send_action 的接口说明区分请求动作和实际发出的动作 | 可读的实验脚本、明确 episode 保存点、提交状态对应实际命令 | 多硬件基类与 processor 平台、Hub 上传、Dataset 格式迁移；其“实际发出”也不能替本仓承诺机械臂已经到位 |
| [LeFranX README](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/README.md)、[双设备录制脚本](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_vr_record.py) | 组合机器人示例直接实例化 Python 配置；README 说明这样绕开组合设备的 Draccus CLI 问题 | 一个实验文件显式组装参数，少量运行参数放 CLI | 覆盖安装 LeRobot 的集成方式、现有示例中的宽泛 fallback、把其代码量作为本仓删减配额 |
| [ManiUniCon main.py](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)、[控制进程](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py) | 机器人、策略和传感器有清楚的运行职责，控制独立运行；同时使用 Hydra 和共享存储组装多硬件 | 让设备控制与昂贵推理隔离，组装位置集中 | Hydra 动态实例化、多机器人通用接口、把所有对象压入共享存储的架构 |
| [Dexora 部署](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/deploy/README.md)、[采集工具](https://github.com/dexoravla/Dexora/blob/6f2869d61483774aa75cda4cd2260dc6a5faa783/teleop/README.md) | 实际脚本围绕特定设备；部署因手/臂 SDK 环境不同而保留独立进程；数据整理工具单列 | 只为真实隔离需求保留进程，分析退出控制路径，入口面向实验任务 | 新增 ZMQ 或搬入另一套采集框架；不把“个人研究”误解为任何跨进程机制都该删除 |

由这些观察得到的设计判断是：**简单来自范围明确、直接组装和副作用归属清楚，并不要求所有任务同线程、所有数据同格式、所有设备同接口。**

## 4. 目标结构与接口边界

### 4.1 原地收缩，避免全仓搬家

下表是职责分配，不要求创建一个新框架，也不为凑目录把文件全部改名。

| 位置 | 目标职责与改动 |
| --- | --- |
| `scripts/collect_teleop.py`、`scripts/run_policy.py`、`scripts/replay_episode.py` | 保留三个容易找到的任务入口；解析少量参数、组装、调用，不塞业务循环 |
| `experiments/default.py`（新增） | 一份可复制的 `CONFIG: RealConfig` 示例；只放实际配置。用户实验可各自写一个 Python 文件，不建实验注册表 |
| `dexmani_real/config/` | 保留按物理/算法职责命名的 dataclass；`real.py` 只做简单入口与聚合；不按字段数机械合并类 |
| `teleop/run.py`、`teleop/loop.py` | 会话组装与一段遥操作；与 deployment 共用低层 I/O，不为统一遥操作和 policy 建通用 episode 基类 |
| `deployment/run.py`、`session.py`、`rollout.py`、`schedule.py` | 组装、外层多段循环、单段执行、动作窗口数学。收尾变顺序调用；调度差异不复制控制副作用 |
| `data/raw.py`、`data/recorder.py`、`data/export.py` | Raw 读写、异步写盘及监督方发布、离线导出。各自一条正常路径 |
| `runtime/`、`ipc/` | 仅保留实际跨进程状态、源快照、键盘和必要关闭；删高层镜像与晚到结果修补，不整目录清空 |
| `robot/`、`control/` | 实际设备边界、执行端、FK/IK/约束/重定向提交语义；保留有独立职责的小类 |
| `geometry/`、`perception/` | 内部原生几何与数值处理；外部模型/数据边界再编码，不另建内部 schema 平台 |
| `scripts/cordvip/`、`third_party/cordvip_geometry/` | 两个主要实验入口和直接阶段函数；删服务/多源宿主外壳。现阶段保留 vendored 包位置与许可证，避免再制造一次大迁移 |
| `scripts/calibrate_camera.py`、`scripts/calibrate_vr_heading.py`、`scripts/tools/` | 独立工具；相机 viewer 合一；标定求解、回放评估和性能分析明确离线 |
| `README.md`、`docs/hand_retargeting.md` | README 放常用路径与关键语义；数学说明留专题文档；删除已退出能力的说明与过期承诺 |

`Robot`、`EpisodeRecorder`、`ObservationBuilder`、`SceneTracker` 等若真正拥有设备、缓冲或算法状态，可以保留。类少、文件少、代码短都不是独立目标；不要把状态藏进全局变量来“扁平化”。

### 4.2 所有权必须说得清

| 事实 / 资源 | 唯一权威方 | 允许的其他行为 |
| --- | --- | --- |
| 机械臂 SDK、Mode、伺服与实测反馈 | 现有 arm owner 进程 | 父进程发布目标、读取带来源时间的快照；不复制 SDK 所有权 |
| motion generation / stop / emergency | 现有共享授权机制 | 执行前重读；不拿入口检查代替消费时检查 |
| 本段控制结束主因 | 单段控制调用 | 清理失败可附加说明；不覆盖第一原因，不建嵌套结果树 |
| Raw 内容写入与资源关闭 | 本段 writer worker | 监督方请求结束、等结果；worker 无发布权 |
| 本段发布/丢弃决定 | 启动该 recorder 的监督方 | 一次决定；确认关闭和计数后改名；晚到 worker 不能改变决定 |
| 模态需要、模式有效参数 | 入口解析 checkpoint/实验配置后的一次派生 | 控制需要与仅录制需要仍分清；下游不再各自产生镜像集合 |
| 相机/机器人来源时间 | 采集该反馈的设备侧 | 消费端只保留和比较，不用“读取缓存时刻”刷新它 |
| 离线分析结果 | 对应离线工具 | 可失败、可重跑；不回写控制结局 |

### 4.3 主调用的目标形状

下面是控制流示意，不是要求新增同名 API。优先让现有函数达到这个结构。

```python
runtime = resolve_experiment_and_checkpoint(...)
resources = open_required_resources(runtime)
try:
    while wait_for_start_or_quit(...):
        outcome = run_one_episode(...)  # 返回前撤权、请求停机并完成本段收尾
        append_episode_result(outcome)
        if outcome.requires_session_exit:
            break
finally:
    close_session_resources(...)
```

`run_one_episode` 不能在 writer/推理任务仍可能污染下一段时假装干净返回。若 native 调用无法在期限内退出，保留一个明确的“未完成收尾”事实并终止继续授权；不新增后台协调服务，不承诺 Python timeout 可以杀掉 native 调用。

### 4.4 面向使用者的目标命令

下表是**重构后的目标接口**，当前基线尚未全部支持；只在对应实现完成后移入 README。保留现有有意义的脚本名和位置参数，避免为统一命名制造迁移。路径均为示例，设备参数与模式在实验配置中设置。

| 任务 | 目标命令 |
| --- | --- |
| 看配置，不连接设备 | `python scripts/collect_teleop.py --config experiments/default.py --print-config` |
| 遥操作采集 | `python scripts/collect_teleop.py pick_place --config experiments/default.py` |
| 离线导出 | `python scripts/export_zarr.py episodes/pick_place --config experiments/default.py --output datasets/pick_place.zarr` |
| 策略部署 | `python scripts/run_policy.py /path/to/saved_experiment --checkpoint latest --config experiments/default.py --num-episodes 5` |
| 物理回放 | `python scripts/replay_episode.py episodes/pick_place/episode_001 --config experiments/default.py --output replay_results/run_001` |
| 离线回放评估（新增薄入口） | `python scripts/evaluate_replay.py --episode episodes/pick_place/episode_001 --replay replay_results/run_001` |
| 相机标定采样 | `python scripts/calibrate_camera.py collect --config experiments/default.py --output calibration_samples` |
| 离线标定求解 | `python scripts/calibrate_camera.py solve calibration_samples --output camera_calibration.json` |
| 离线几何追踪 | `python scripts/cordvip/track_object.py --episode episodes/pick_place/episode_001 --scene scene.json --output outputs/scene_001` |

`--print-config` 保留为便宜、无设备副作用的检查；显示请求值，不冒充实测相机事实或已经解析完的 checkpoint 配置。已有真实推理实验参数 `--weights`、`--inference-steps`、`--seed`、`--device` 可保留为单层明确 CLI；不要为收拢这几个参数再建庞大配置类。

迁移时只转换当前有效的本仓实验 YAML/命令，不删除用户文件。训练得到的 `saved_experiment/config.yaml` 是配套 Policy 的外部模型协议，继续正常读取，**不**迁移为 Python，也不据此强删 PyYAML。旧 Raw、Canonical、checkpoint、米制资产保持可读；退出的是旧实时几何宿主与临时中间产物承诺，其外部 notebook 如需使用，应一次改为读取逐帧离线结果。

## 5. 保留边界：删重复机制，不删真实研究语义

| 边界 | 保留内容 | 可以收缩的内容 |
| --- | --- | --- |
| 外部数值输入 | VR / policy / SDK / 文件的维度、有限性、坐标/单位、必要范围 | 已在唯一入口建立且未改变的内部数组重复扫描；先确认数组所有权 |
| 命令执行 | SDK 返回码、run_id/撤权、deadline、反馈时效、controller 状态、Ruckig jerk 与制动极值 | 同一不可变命令的多层静态预检；不能删跨时间/跨锁后重新检查 |
| 手部状态 | 成功提交后的初值、阻尼、滤波状态对应实际提交的手目标 | 对原始候选再次投影/检查的镜像路径；失败提交不得推进状态 |
| 数据 | accepted/H5/video 行一致、原来源时间、实测 state 与 action 分开、有效前缀、13 模态 | 仅用于修补收尾时序的重复 refresh、默认运行时评估、无人消费的中间产物 |
| 跨仓协议 | Canonical 字段/单位、9D/21D、recipe、URDF/template/calibration 事实、source_id 与 split/resume | 内部配置组织可以变，已存数据/模型语义不得静默变 |
| 几何 | observed/model 区分、pose-valid 深度支持、FP 所需 mesh normals、SAM 历史清理、源关联 | latest 心跳与 STALE 外壳、重复 Raw reader、重复 rays、中间未定尺度采样 |
| 标定 | 静止和新鲜帧配对、足够激励、残差、原样本索引 | 控制设备的交互 UI、与采样无关的运行会话、重复加工已接纳样本 |
| 错误 | 原始异常与子进程 traceback、失败主因、必要清理错误 | 无用途的身份探针、全量成功步诊断 dict、多层 catch/rethrow、宽泛静默 fallback |

特别注意：送入 arm mailbox 不保证该目标已进入 SDK，更不保证物理到位；本方案不新增这种承诺。若一个组合命令发生部分设备提交失败，保留原失败并停止，不能把两台设备描述为具有事务原子性。

## 6. 分批任务与验收标准

**默认主线：T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8。** 顺序表示优先级；独立工具任务可在某个外部依赖阻塞时先做。T2 内的录制与生命周期要作为一个完整改动链，不留下新旧协议并存的最终状态。每批完成后进行针对性检查，再进入下一批，不每批重新请求继续。

### T0 — 建立最小基线，不扩展审计工程

- [ ] 读 `AGENTS.md`，检查 `git status`、实际 HEAD 和已有改动；使用工作分支，保留用户工作。
- [ ] 用 `rg` 找当前批次的定义、全部调用方和对应文档；只复核本任务受影响的链。
- [ ] 在仓外临时目录保存必要的对照结果：一条固定手/臂数值序列、一个小 Raw fixture、三模式固定时钟调度轨迹。已有真实样例可只读复用；不存在就明确标记，不能伪造“真实数据回归”。
- [ ] 记录可执行环境：标准 Python、数值依赖、视频依赖、GPU 模型、真实设备分别是否可用。基线没有 `tests/`，不把历史文档所述测试数当现有测试集。

**完成标准：** 有实际基线 SHA、工作区状态、可运行验证范围；不新增常驻 inventory、evidence、contract 或审计生成平台。

### T1 — 先解决可达的次生失败和工具依赖（最高优先级）

**范围：** `teleop/retargeting/adaptive.py`、`config/teleop.py`、`teleop/loop.py`、`data/raw.py`、`robot/xhand/driver.py`、`scripts/tools/control_xhand.py`、`scripts/calibrate_vr_heading.py`、`sensor/vr.py`、`utils/vr_heading.py`、`replay/run.py` 及 geometry reconstruction。

- [ ] 非有限求解 loss 仍导致动作拒绝，但可选诊断中的 NaN/Inf 在其产生处用 `null` 加明确失败原因表示，或不存该无效数值；不要改全部 JSON 为 `allow_nan=True`，不要清洗真实训练数组冒充有效数据。
- [ ] XHand 普通 connect 不再读版本/类型/序列号等可选身份信息；显式诊断工具需要时调用。真实连接、首帧、SDK 返回码检查保留。
- [ ] mesh 必要生成、写出与返回不依赖附加版本/显存资料；固定生成参数、模型来源仍保存，昂贵/可失败的环境报告移显式诊断。
- [ ] 头朝向标定直接消费 `HeadFrame`；腕参考直接消费对应 `HandFrame`。保留去重、有限读取、停止关闭及现有估计数学；删除仅为该工具存在的右手 ring 就绪依赖和 head position 缓存。
- [ ] `tip_weight` 允许 0，按真实数学域检查权重、分母和时间尺度；退出 `vars()` 扫描加特例白名单。0 只关闭指尖位置项，不关闭方向/pinch 的指尖作用。
- [ ] `replay/run.py` 只保存控制结果和采样；`replay/metrics.py` 由明确离线入口运行，分析失败不回写控制 status。

**验收：** fake 非有限 loss 能保留原拒绝并完成可写的失败记录；可选身份/资料抛错不改变业务成功；只有头帧可完成默认标定；零位置权重的 loss/梯度符合公式；离线指标失败不影响已保存的 trajectory status。默认算法参数不变。

### T2 — 录制与单轮生命周期整体变薄（最大收益）

**范围：** `data/recorder.py`、`data/raw.py`、`teleop/loop.py`、`deployment/{run,session,rollout,schedule,inference,observation,preflight}.py`、对应 runtime/入口调用方。

#### T2.1 录制只有一个发布点

- [ ] writer 仅负责 open → consume → flush/close → 返回写入计数与错误；从 worker 中移除 `publish`、发布锁和取消发布竞争。
- [ ] 监督方先撤权/请求停止，再送出明确结束请求，等待 worker 完成；不要让写盘阻塞控制端先发出停止请求。
- [ ] 保留有界队列及必要的 ready、退出信号、错误传播；启动失败或 open 晚到返回后，worker 必须能看见退出请求，不能永久等待无人再写的队列。
- [ ] 将“是否保存”与“本段技术状态”分开。已完整封存的技术失败段可按既有政策发布为 failed Raw；writer 失败、close 失败、行数不一致、join 超时永不发布。
- [ ] 监督方确认 worker 已结束、资源关闭无错、accepted/H5/video 计数一致后，才将本次 `.tmp_*` 改名为 final。保持原格式与元数据，不另加发布标记协议。
- [ ] D 丢弃只在 worker 退出后删临时文件。超时目录不复用、不发布；晚到线程没有发布权限。
- [ ] `discover_episode_dirs` 及 export/replay 的显式路径入口也拒绝未发布 `.tmp_*`；不能只靠批量扫描时忽略隐藏目录。
- [ ] 删 `publication_lock`、`cancel_publication`、重复 finish、`refresh_raw` 和用于迟到发布的结果修补；保留最小的“worker 尚未退出”事实。

**关键验收：** open/append/close 失败、队列满、超时、晚到完成、正常保存、丢弃均覆盖。用可控事件模拟阻塞，不让测试挂真实 native 调用。不得因为 `RawWriter.finish` 提前写了 normal 就发布仍未成功关闭的目录。

#### T2.2 一个 episode 顺序返回

- [ ] 外层 session 只做等待开始、调用本段、接收结果、决定下一段；删除跨层 `finished/pending/refresh` 推进。
- [ ] 控制结束主因在一处确定，清理错误作为补充；复用现有 `RunEndReason`，必要时补充分明的机器结束码，不创建另一套通用事件类型树。
- [ ] 总时长完成、预测等待超时、操作者停止/退出、目标拒绝、硬件故障分别映射一次。自由文本 detail 只解释，不参与 normal/failed 判断。
- [ ] 一会话多段、模型复用、推理线程保留；迟到预测受 run_id 约束；本段未干净收完不能开始下一段。
- [ ] 保留当前“预期段拒绝”与真实 session 故障的区别。当前 Policy 异常已经退出 session，不能把新逻辑描述成修掉一个不存在的普遍自动恢复机制。

#### T2.3 三种调度，共用副作用

- [ ] 保留现有 `schedule.py` 的小函数与必要 chunk/prediction 状态；入口选定模式，不建 Strategy/Factory/Executor 基类。
- [ ] 观测构造、动作映射、提交、手状态 commit、录制、停止和 finally 各只有一条权威实现；sync/async/RTC 仅表达动作窗口与交接差异。
- [ ] checkpoint 恢复后一次解析 action_steps/prefetch/guidance；观察字段一次派生有效依赖。控制源与仅录制源仍区分，不让额外录制相机缩短纯控制动作期限。
- [ ] 内部 commit 消费实际 `RobotCommand.hand_qpos` 或等价拥有清晰生命周期的已提交目标；不能直接沿用可变 `ActionResult` 并删掉后置检查。失败/撤权/过期不得推进滤波和下一帧初值。
- [ ] 相机与 cloud 确认同一来源配对后可复用该来源的标量 freshness 结果；cloud-only 仍独立检查，IPC 防撕裂与动态时效检查保留。

**整批验收：** 正常连续两段不重载模型；故障/未退出 worker 阻止再授权；三模式的动作索引、锚点、交接、RTC frozen guidance 与基线相同；没有补发错过动作；无新旧生命周期并存。

### T3 — 配置和日常操作收敛为单一路径

**范围：** `config/`、所有 `scripts/` 参数解析、`deployment/preflight.py`、teleop 键盘/音频、README。

- [ ] `--config` 指向一个可信本地 `.py` 文件，导出 `CONFIG: RealConfig`；省略时用现有 dataclass 默认值。可用标准库 `runpy.run_path` 直接加载，做一次返回对象类型检查；不加入动态插件发现、任意表达式解析或 schema 框架。
- [ ] 实验文件用显式构造或 `dataclasses.replace` 修改现有 dataclass；CLI 只保留任务名、输入/输出路径、checkpoint、段数/时长等真正的运行参数及工具特有选择。模式、增益、点数等实验参数优先放 Python 文件；只有真实高频扫参需要才新增明确 flag。
- [ ] 保留第 4.4 节明确的推理研究 CLI 与无设备 `--print-config`；配套 Policy 的 saved YAML 仍按原协议读取。本仓配置迁移不延伸为训练仓重构。
- [ ] 删除 YAML → mapping → 递归 dataclass 替换、嵌套 override 合并、旧 CLI 转发；一次迁移仓内示例。先查实际 import，再决定 PyYAML 依赖是否能删；其他用途仍需要时保留。
- [ ] 仅对启用路径做业务域检查：L515 专有字段在实际 L515 分支检查；关闭 remove_table 时不检查休眠 margin 的业务范围；disabled table 不挡住无关流程。完整运行快照仍须可序列化，不能靠塞 NaN 绕过。
- [ ] 将 table plane 标定事实从 HOME 碰撞盒体配置中移出，让点云和 HOME 直接读同一来源；不因此改 Canonical recipe 的字段、数值与来源语义。
- [ ] 保留具名 workspace、清楚的 deg/rad、桌面物理高度与间隙、障碍物名字。删除的是配置入口与镜像，不把 123 字段统一硬编码。
- [ ] 取消 C 暂停续写、双 Q 和播放线程/代际/收尾等待。B/S/D/Q/Esc 使用第 2 节规则；HOME 保留在无活动段时执行。准备中取消、收尾中按键不能授权新运动。
- [ ] 日常默认终端提示。若以后明确需要 VR 音频，只增加不影响授权与关闭的简单提示，不恢复播放协调系统。

**验收：** 一个参数能追到唯一配置来源；请求配置与实际相机事实都可记录且职责不同；默认有效数值与旧配置相同；旧入口删除而非转发；已有 recipe/checkpoint 在相同参数下读取结果一致。README 给出复制配置、采集、导出、部署、回放的实际可用命令。

### T4 — 几何工具改为短的离线研究流程

**范围：** `scripts/cordvip/` 与 `third_party/cordvip_geometry/`，尤其 `cli/scene_tracking.py`、`scene_tracking.py`、`integrations/dexmani.py`、`model_depth.py`、`cli/build_object.py`、`cli/prepare_object.py`、live viewer/worker。

- [ ] 固定主流程：Raw 第一帧 → 人工确认初始 masks → 顺序处理全部 Raw 行 → 保存必要逐行结果 → 离线查看/分析。默认不按墙钟跳帧；源 row ID 和来源时间始终保存。
- [ ] 复用 initial selection 时仍确认它属于当前相同首帧及相机几何；不为省掉校验而把旧 mask 套到另一段录像。删除 live 来源后只移除不再存在的来源分支。
- [ ] 删除 latest 持续发布、心跳、STALE、退出失效和 unlink 协议。先将 `scene_payload` 的 `publish_frame/save_frames` 实际消费者迁移成直接结果编码，再删外壳。
- [ ] 退出实时相机追踪、通用外部 Frame 宿主、多源组合与实时查看的线程/进程编排。保留初始 mask 的简单交互和离线 viewer，不重新造通用播放器。
- [ ] 保留 single/multi-object 与 mask-only 模式；无 asset 时不要求 FoundationPose，不伪造 pose。object/reference role 若保留作为标签，仅存一次 metadata，不暗示它改变算法。
- [ ] 一个 RawReader 按同行读取 RGB-D、实测关节和时钟，直接交给手 FK；退出第二次 reader、自 join 和重复来源重建。同行关联不声称物理同步，不用 action 代替 measured state。
- [ ] 一个场景的 K/shape/stride 建一份只读 rays；每物体保留自己的 mesh scene 与 pose。默认数值、ROI、stride 和深度约定不变。
- [ ] generate 只生成未定尺度 mesh；preview 临时采样。prepare 在最终米制尺度按原 seed/点数生成一次 points/normals；删无后续消费的 `points_unscaled.npy` 与恒为 1 的 loader scale 参数。
- [ ] 将 Namespace → 拼 CLI → 再 parse 的阶段转发改为直接函数。保留一个必要生成子进程释放 GPU；跨进程只传本阶段实际所需参数，避免把完整配置重新包装成 RPC 平台。
- [ ] 单次生成进程无需支持宿主可重入环境恢复；删除此承诺及仅为它存在的探针/状态，实际模型加载前置条件仍保留。
- [ ] SAM 官方模型用一个 variant 选择匹配 config/checkpoint；自训模型允许一次显式传完整配对。SAM RPC 是否退出按 C3 处理，本批不假定环境兼容。
- [ ] 更正文档：深度几何支持量参与 `POSE_LOST` 判定，不写成“仅用于诊断”；保留 FP mesh normals、算法门控与 SAM 长序列历史清理。

**验收：** 同输入/seed 下最终资产一致；同样 Raw 行的 masks、pose-valid、observed/model 点和手 FK 在适当容差内一致；确认离线默认全帧这一工作流变化；结果仍可独立查看；没有 latest/live 服务残留调用。GPU 不可用时完成可静态验证的删减并将模型回归标为未验证，不据此删除隔离机制。

### T5 — 内部数值表示与校验位置简化

**范围：** `geometry/pose.py`、`geometry/kinematics.py`、`perception/hand_cloud.py`、`control/actions.py`、`teleop/{vr,retargeting/*}.py`、导出和观测构造的调用方。

- [ ] 内部 FK 返回拥有明确数组归属的原生 SE(3)/旋转矩阵；手点云、标定直接消费。仅在 Canonical / policy 边界编码 rot6d，外部 9D/21D 及坐标约定保持。
- [ ] 不借出下一次 FK 会覆写的 Pinocchio 内部 view；保留需要隔离别名的 copy。只有确定当前临时对象可移交时才去掉重复归一化/复制。
- [ ] `eef_poses` 的 FK 参数改必需，迁移现有六处显式调用；不在便利函数内偷偷构造 FK 模型。
- [ ] 合并同一不可变命令的静态检查；HOME/jog/replay/独立工具的真实入口全部覆盖后才删内部检查。操作限位与机械额定范围不合并。
- [ ] 删除无仓内调用的 `Pose.identity`、`quat_wxyz_to_rotmat` 和几何镜像字段；保留 `metadata.link_names` 与算法真正使用的 cloud diagnostics。
- [ ] 手标定保留第一次 prepare 的早拒/重采，将已算出的有效测量用于最终 median/MAD；固定尺度预计算只放在已有对象中，不创建 cache manager。
- [ ] `ftol_abs` 当前同时影响 NLopt 停止和 seed 改善接受容差：本轮明确注释并保持现有默认与接受集合。没有独立实验需要时不新增另一个参数、不直接删接受条件。

**验收：** 固定输入的 FK/EEF/指尖/手点云、有效标定样本集合和结果一致；下一次 FK 不改变上一份结果；非法外部 rot6d 仍拒绝；失败提交不推进 retarget 状态。数值容差按 dtype、单位和现有数值误差制定，不为通过测试任意放宽。

### T6 — 标定与查看工具专注一件事

- [ ] 相机标定入口提供直接的 `collect` 与 `solve` 两阶段；继续复用现有求解数学。collect 仅读实测反馈和相机，solve 只读采样文件。
- [ ] 只读采集不经会 enable、切 mode、apply config 的 `Robot.connect`。直接使用最低层必要读接口；必要时增加明确的 `connect_readonly`，不要给全系统加一串 bool 模式。
- [ ] 摆位由厂商示教或现有 `keyboard_teleop.py` 负责；停止并退出控制连接后再只读采样，不同时抢占设备。记录原帧/样本 ID、时钟、静止条件和求解残差。
- [ ] 退出相机标定内嵌 jog/HOME/发布代际后，删除它唯一消费的 JSON `cancelled` 回调链；保留普通文件写入失败和覆盖政策。
- [ ] 合并 `view_realsense.py` 与 `view_pointcloud.py` 的相机连接、读帧和窗口流程；按实际视图提供少量选项，不搭 viewer 插件系统。
- [ ] `visualize_episode.py` 顺序传当前 RGB，删隐藏的镜像 index 和仅传默认常量的 fingertip 参数；部署中真实自定义指尖能力保留。

**验收：** fake SDK 证明只读采样不调用运动初始化；固定原始样本的解、排除样本索引与残差一致；坏 RGB 明确失败；两个查看需求仍可用；只读能力若受真实 SDK 限制，明确阻塞本子任务，不能伪称完成。

### T7 — 诊断、依赖和文档收尾

- [ ] 默认终端 INFO、拒绝原因、错误 traceback 与简短段结果。移除默认每进程 DEBUG 文件及无需求的环境变量 fallback；需要留档时可用 shell 重定向或一个明确日志选项，不加日志管理框架。
- [ ] 保留 worker 原始 traceback 的可见渠道；父进程的截断错误字符串不能替代它。坏网络帧用简短原因且不刷新来源时间，不为此建设通用限流状态机。
- [ ] 成功步不再创建最终丢弃的完整诊断 dict/多份复制；必要的失败前后计时仍在事件发生处采集，不能失败后补造。算法用到的 diagnostics 不移走。
- [ ] 详细 timing/profile 和 replay/geometry 聚合分析在明确工具中做，运行结束不自动跑性能或质量评估。不凭静态删代码声称 Hz 提升。
- [ ] 仅在某能力退出后删除其依赖、exports、CLI 参数、文档和测试；保持惰性导入，离线导出不因无关设备/GPU 包缺失而失败。
- [ ] 重写 README 的常用命令、五键行为、几何离线路径、标定两阶段、配置迁移与输出语义；数学专题文档保留。历史任务文档若误导当前行为，简短标为已替代，不新增另一套长期审计门户。

**验收：** 新命令帮助与文档一致；删除项无残留 import/入口；失败原因可定位；依赖删除有实际消费核对；无新常驻诊断服务。

### T8 — 统一复核与交付

- [ ] 执行第 8 节与改动匹配的回归；不以 `--help` 成功代替数值或异步语义验证。
- [ ] 全部默认入口的“配置 → 资源 → 单段 → 保存 → 关闭”能直接追踪；每个 publication、commit 和技术状态决定点只有一个权威路径。
- [ ] 检查 diff，删过渡兼容层、无消费者参数/字段、遗漏导入及过期文档；不顺手开展新的平台化重写。
- [ ] 在第 12 节更新完成批次、实际测试、未验证项与条件任务结论。默认主线有真实阻塞时明确写“部分完成”，不把计划勾成已验证。
- [ ] 给出行为变化、代码与依赖变化、验证证据、剩余限制；工作分支小提交交付，是否推送/合并遵循用户本次实施指令。

## 7. 条件任务：默认不做，不能借清理悄悄改变实验

以下并非留给执行者随意发挥的待办。缺少表中证据时，保持现有实现，并在交付中明确保留原因；它们不阻塞其他主线工作。

| ID | 候选 | 执行条件与验收 | 当前默认 |
| --- | --- | --- | --- |
| C1 | 只导出完整正常 episode，退出失败有效前缀 | 用户明确选择数据政策；列出有效段/行变化；新导出新 source_id，旧数据/checkpoint 不改；必要时重采 | 保留现行 prefix 政策 |
| C2 | 融合 camera 与 pointcloud owner，删第二进程/ring/配对 | 在实际分辨率、点数和控制需要下测吞吐、来源年龄、丢帧及停止响应，证明绑定发布节拍可接受 | 保留现有独立 owner、5 帧相机历史、防撕裂和同源配对 |
| C3 | SAM 与 FP 同解释器，删除 segmentation RPC | 已验证依赖可共存、GPU 峰值显存、长序列内存、吞吐及取消行为；确认不再需要独立终止 SAM | 保留实际隔离；404 行 RPC 不是纯兼容垃圾 |
| C4 | 退出逐帧采样法向输出 | 确认外部研究脚本无接触法向需求，或已迁移为由 asset+pose 离线派生；保持旧变换结果 | 保留输出；无论如何保留 FP mesh normals |
| C5 | 砍 serial/EtherCAT、VR transport、HOME/reset、复杂回零或整个 replay | 有实际设备与课题清单；按完整功能删除；回零另比较可达范围，不能以少代码代替结论 | 保留现有实际支持；不默认砍 RTC/回放/HOME |
| C6 | 合并 workspace/collision FK、退出 static-only camera、其他小缓存 | 有真实调用成本或 checkpoint 组合依据；不增加缓存失效状态；不改变早拒/源语义 | 保留清楚的小实现，收益明确时顺手改 |

## 8. 最小但足够的验证

不建立 contract/evidence 平台，不做一比一复刻实现的测试，不把“没有硬件”当全部不验证的理由。复用现有测试工具；若仍没有测试框架，用标准库 `unittest` 和少量局部 fake 即可。只为本次高影响行为写持久回归，简单删除/改名做导入、调用与 diff 检查即可。

| 验证层 | 最小覆盖 | 通过含义 / 限制 |
| --- | --- | --- |
| 纯逻辑 | stop code 分类、tip=0 数学、配置有效分支、三模式固定时钟调度、源配对 | 可证明所测确定性语义，不证明真机节拍 |
| 异步所有权 | open 晚到、队列满、writer 失败/关闭失败、超时后晚到、连续两段、迟到预测 | fake clock/Event 驱动；确认不发布、不再授权、无状态污染 |
| 小数据 I/O | 小 H5+MP4 正常/failed/损坏样例，accepted/H5/video 一致，显式 `.tmp` 被拒绝 | 真正读写和关闭；不能只 mock 掉全部 I/O 后宣称数据格式兼容 |
| 数值对照 | 固定关节/姿态序列的 FK、点云、标定、重定向 loss/梯度；导出字段、dtype、单位 | 对照本次改动前基线；精确字段精确比，浮点用有依据的容差 |
| 跨仓读取 | 用实际配套 Policy 的 metadata/数据读取恢复 recipe、13 模态与 source_id/split | 不修改策略仓来掩盖兼容破坏；无可用 checkpoint 时明确未验证 checkpoint 路径 |
| GPU 离线 | 短 Raw 同行结果、最终资产、长序列 SAM 内存；C3 另测共存/取消 | 仅在依赖/资产可用时运行；无条件不打勾 |
| 真机 | 现有参数下准备/停止、故障退出、连续两段、按键、模式与录制 | 在用户现场运行；本代码清理任务本身不自动连接并驱动设备 |

执行命令按实际新增测试文件填写。标准库测试可用 `PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests`；如果仓库实际已有 pytest，则使用 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider <相关测试>`。Ruff 若可用带 `--no-cache`；mypy 缓存定向仓外。遵循 `AGENTS.md`，不在仓内生成 `.pytest_cache/`、`.mypy_cache/`、`.ruff_cache/`，不为它们新增忽略规则。

不要假定 `compileall` 不写文件；纯语法检查可在内存 `compile(source, path, 'exec')`。没有依赖的检查标为未运行，不自动下载安装整套机器人/GPU 环境来完成文档式验收。

## 9. 必须纠正的旧表述

1. evidence/validate 等关联数量不是缺陷数量；也没有理由按名称批量删除函数或类。
2. 123 个配置字段有静态消费链；是否值得继续暴露是范围选择，不能都标为死参数。
3. `duration_timeout` 目前来自稳定内部 code；用自由 detail 驱动分类需要改善，但没有证明默认分类已出错。
4. 回放 metrics 可改变综合 status，但 `trajectory_status` 与先保存样本仍在；不能说执行真相全部丢失。
5. 当前 Policy 真异常已终止 session；Teleop 预期段拒绝后回 IDLE 不等于一般硬故障自动恢复。
6. latest 无 tracked reader 不等于 `scene_payload` 无内部消费者；无仓内调用也不能证明用户 notebook 从未使用。
7. XHand 身份查询普通非零返回码原本只是 warning；可能打断 connect 的是抛异常路径。
8. 取消发布不是停止 worker；`raise`、`join(timeout)`、`executor.shutdown(wait=False)` 不会终止阻塞 native 调用。
9. 当前 retarget commit 用原 `ActionResult` 数组，发送用复制只读的 `RobotCommand`；先统一对象归属再删检查。
10. tip=0 只关闭位置项；ftol 的两个用途同量纲，可能有意共用，本轮不自动改变数学政策。
11. 几何深度支持参与 pose-valid；两次 run_config 分别记录请求与实测事实；`ObservationBuilder.features` 有窗口复用用途。
12. 净删行、吞吐、时延、显存收益都需要实施后测量，不把设计估计写成事实。

## 10. 完成定义与防止重新变复杂

主线完成需要同时满足以下条件，不要求条件任务 C1–C6 全部删除。

- 常用采集/导出/部署/回放各有一个明确入口，配置只走 Python 单路径；README 可直接照做。
- 正常多段循环由局部调用表达；一段的发布和收尾不依赖下一次外层 tick，也没有晚到发布。
- 保留研究数值、来源时钟、格式与跨仓消费者；指定的 UI/几何/标定使用方式改变已明确迁移。
- 不存在为了“暂时兼容”而永久保留的双配置、双 recorder、双 controller、旧 API 转发壳。
- 标定/指标/诊断不控制无关业务是否成功；可选报告坏了不伪装真实失败为成功，也不掩盖真实写入/设备错误。
- 删除对应功能后，其参数、导入、后台任务、文档承诺也一并退出；留下来的小封装能解释实际职责。
- 受影响的高风险路径有针对性验证；不可执行项写明缺少的设备、模型或数据，不写“全部通过”。

每批 review 只需回答：删掉了哪套机制；现在谁拥有状态和副作用；保留检查覆盖哪个真实风险；有无改变研究语义；新增东西是否比删掉的更绕。无需把这五个问题做成另一套自动化门禁。

## 11. 原 27 项结论的完整去向

| 原 ID | 结论 | 本方案去向 |
| --- | --- | --- |
| R01 | 可选诊断不决定业务成败 | T1，明确实施 |
| R02 | 头朝向标定独立 | T1，明确实施 |
| R03 | tip_weight 合法零权重 | T1，保留其他指尖项 |
| R04 | 控制结束/分析/技术状态分离 | T1 离线 metrics；T2 机器原因分类 |
| R05 | 监督方唯一发布 | T2.1，保留 worker 退出与发布准入 |
| R06 | 单轮顺序返回 | T2.2，保留未完成收尾事实与模型复用 |
| R07 | 三模式共享副作用/单点需求派生 | T2.3，不砍 RTC |
| R08 | 静态检查边界与实际命令 commit | T2.3 + T5，不删动态检查 |
| R09 | latest 服务退场 | T4，先迁移逐帧结果消费者 |
| R10 | 单 Raw reader 同行 FK | T4，与外部 Frame 宿主一起收窄 |
| R11 | 单场景共享 rays | T4，不共享可变 mesh scene |
| R12 | 最终尺度只采样一次 | T4，preview 临时采样 |
| R13 | 内部原生 SE(3) | T5，保留外部编码和数组归属 |
| R14 | active 分支校验与标定事实分离 | T3，保持 recipe 兼容 |
| R15 | ftol 双重语义 | T5 文档明确，默认不改算法/不新增参数 |
| R16 | 暂停、双 Q、音频协调退场 | T3，明确便利能力裁剪 |
| R17 | 配置输入收缩 | T3，选择 Python 单路径 |
| R18 | 标定/相机工具简化 | T6，保留只读约束与求解数学 |
| R19 | 几何 Raw 短流程 | T4；SAM 同进程部分移 C3 |
| R20 | 整段数据准入替代失败前缀 | C1，默认不做 |
| R21 | 相机/点云 owner 融合 | C2，默认不做 |
| R22 | 法向输出退出 | C4，默认不做 |
| R23 | 固定后端、HOME/传输/回放裁剪 | C5，默认不做 |
| R24 | 少量 FK/静态相机/快照优化 | C6；随 T2 组装迁移可删确定镜像，不独立建抽象 |
| R25 | 无消费接口与镜像字段 | T5，原地顺手删除 |
| R26 | 标定重复计算/固定几何 | T5，保留早拒，不建设缓存生命周期 |
| R27 | 日常诊断收敛 | T7，保留失败可见性与实际时间点 |

另 5 项保留意见：C07 障碍物名字、C08 明确 deg/rad、C09 具名 workspace、C10 桌面高度与间隙、E06 `HomeResult` 预期拒绝/取消语义。这里的 C07–C10 是原 review 编号，与本任务书的条件任务 C1–C6 无冲突；它们均不按“少字段/少类”目标删除。

## 12. 执行进度（由实施 Codex 原地更新）

| 批次 | 状态 | 提交 / 实际验证 / 阻塞 |
| --- | --- | --- |
| T0 | 完成 | 基线 `580ac65`，初始 main 工作区/暂存区干净；工作分支 `cleanup/personal-research-t0-t8`。`real_robot/bin/python` 可用；固定 FK/手云、三模式轨迹、3 行 normal/failed H5+MP4 与 13 模态 Zarr 对照保存在 `/tmp/dexmani-cleanup-baseline`。238 行真实 Raw 准入及前三行 FK 已读；Policy 当前 `4618a2e`，只读保留其已有改动。sandbox 外 CUDA/光栅化/FP 深度核通过；Warp 启动有驱动 API 报错但实测核成功。SAM/DINO/Hunyuan 默认权重缺失；完整模型、真机未验证。 |
| T1 | 完成 | 删除 XHand 默认身份缓存/探针、VR 标定的右手 IPC/头缓存与音频依赖、默认回放指标及 mesh 版本/显存报告。非有限 loss 仍拒绝，诊断可严格 JSON 保存；tip=0 保留方向/pinch。新增离线 `evaluate_replay.py`。`PYTHONDONTWRITEBYTECODE=1 real_robot/bin/python -m unittest discover -s tests -v`：19 项通过，含 FK 梯度、纯头帧、超时/关闭、真实文件写入及失败注入；`git diff --check` 通过。未验证网络 VR、实际生成权重和真机。 |
| T2 | 完成 | writer 只写/关，监督方唯一发布；删除发布锁/取消竞争、重复 finish/refresh 与外层 tick 收尾。Teleop/Policy 单段顺序返回，迟到模型任务有界回收，未退出则终止会话；实发 RobotCommand 与授权锁内 commit；机器码区分时长完成/预测超时。为使遥操作整段返回，暂停/双 Q 同批移除，HOME 仅空闲。60 项 unittest 通过，含 open/队满/close/超时晚到、连续两段、Esc/撤权/RTC、真实 H5+MP4；13 模态重导出与 T0 精确一致（新 source_id 按原行为生成），Policy split 恢复通过，diff/语法检查通过。保留原 Raw 封存时点：未参与控制的退休预测报错令 session 失败，不回写已发布 Raw，不改变原准入集合。未验证真实 checkpoint 推理/真机节拍与停机。 |
| T3 | 完成 | 可信 Python `CONFIG: RealConfig` 单入口，删除 YAML 递归/嵌套覆盖，迁移 10 个 CLI 与示例；无手参数仅明确确认。桌面标定路径移出 HOME 盒体，业务校验只随启用分支；完整快照严格 JSON。删除音频线程/播放代际/等待及 12 个无消费 WAV，文档更新五键与 HOME 行为。13 个配置测试、此前 60 项回归、10 个 CLI help、无设备 print 与旧参数拒绝检查通过；默认完整值（除路径移位）及 Canonical recipe 精确一致。PyYAML 为外部模型/运行快照保留。Policy 并行更新已改为 checkpoint 恢复配方，文案按当前事实修正；用户另明确批准同步删除旧 LAMP 动作历史分支，后续独立记录适配。未验证设备配置实际生效。 |
| Policy 接口适配 | 完成 | 用户明确批准退出旧 LAMP 动作历史；按当前 Policy `32a64d0` 公共 API 删除本仓 hand_history/action_context 分支，不加别名或 fallback，不修改配套仓。10 项生命周期与 6 项三模式调度测试通过；真实 checkpoint 加载尚未验证。 |
| T4 | 实现完成；模型回归未验证 | 单 RawReader 同行 RGB-D/实测关节/时钟→首帧确认→默认全行→必存逐行结果→独立 view；删除 live/capture/latest/心跳/STALE/unlink、外层 worker/播放器与阶段 CLI 转发。保留 mask-only/多物体、SAM RPC/历史、法向与深度 pose-valid 门控。共享只读 rays；生成只保存未定尺度 mesh，最终米制仅采样一次。89 项 unittest、相关 CLI help、diff 检查通过；24 组渲染和固定点/法向精确相等；真实 Raw 前三行新旧 RGB-D/手几何/关节/时钟/metadata 精确一致，reader 打开 2→1。sandbox 外 FP 两组真实权重已加载，非真实场景精度验证；SAM/DINO/Hunyuan 默认权重和已确认真实场景资产缺失，完整模型结果/长序列/GUI 未验证。 |
| T5 | 完成 | 内部 ArmFK/EEF/手几何改原生 SE(3)，必需显式 FK，仅 Canonical/policy 边界编码9D/21D；删除无消费 pose/helper/几何镜像，保留独立真实命令入口校验和动态重检。手标定复用已接纳测量，固定几何在原对象内预计算；ftol 双用途/default/接受式不变。97 项 unittest 及 diff 检查通过；20 组 EEF/指尖/1024手云、5行13模态与 T0 全部精确相等；VR旋转差≤2.22e-16，回放朝向指标差≤1.44e-11度；28个标定输入接受集合一致，标定/targets/loss/gradient精确一致，非法rot6d仍拒绝、FK无别名覆写。未测真机/求解吞吐。 |
| T6 | 完成 | 相机 `collect` 只读采样（每组 connect/read/close 后断开以便独立摆位），`solve` 只读保存样本、按原 ID 剔除并保存残差/质量报告；删除旧控制类/jog/HOME/在线求解和 JSON cancelled 链。相机/点云查看合一，桌面拟合/快照/benchmark 显式阶段共用连接；episode viewer 主循环直接传 RGB。112 项 unittest、CLI help、diff 检查通过；fake 底层 SDK 证明零 enable/mode/config/servo/stop；真实 ArUco 图像5帧→NPZ写读、坏RGB/时钟/静止/真实I/O失败均测；16个含噪样本及剔除原ID3后的解/残差与基线 atol1e-10一致，原serial外参读取兼容；五帧桌面拟合与旧函数精确一致。未连接真实 SDK/相机，未验 GUI 或真实标定数据。 |
| T7 | 未开始 | — |
| T8 | 未开始 | — |

只有实施后才更新状态。记录保持短：修改结果、实际执行的命令/对照、尚未验证的范围即可，不复制全量日志。跨上下文续做时，先读本表、相关提交和工作区 diff，不重新审计整个仓库。

配套执行 prompt 见根目录 [CODEX_CLEANUP_PROMPT.md](CODEX_CLEANUP_PROMPT.md)。
