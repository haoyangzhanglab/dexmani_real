# Codex 任务书：dexmani_real 精简整改与全模态数据闭环

日期：2026-10-03  
仓库：`haoyangzhanglab/dexmani_real`，分支：`main`  
方案核查基线：`b7fc4549931314af404082251a2331ffe5ed4ce8`  
状态：保留的整改任务书；其中的审查结论和实施指令不是当前待办清单。当前行为以源码和 README 为准，本文不作为测试或真机验证通过的证明。

本文是用户最终要求的独立任务书，完整承载实施要求，不依赖聊天记录、外部报告或此前临时脚本。它替代前几轮方案中相冲突的实现建议。上面的 SHA 只是证据基线，不是要求 checkout/reset 的目标。

二次审查只收紧执行边界：补齐驱动到导出的实际修改链；保留有作用的 replay 小幅准备；区分时间抖动与缺测；明确相机缺标定处理。没有扩大到新架构或新的传感器能力。

## 0. 执行要求与范围

目标是个人 PhD 灵巧操作实验：简洁、高效、正确、好用，重点是全模态数据采集/导出与同步真机策略推理。直接修真实错误，删除无效机制；不要借本次整改建设平台。

开始时阅读当前 `AGENTS.md`、本文、相关入口与调用链，检查 `git status` 和实际 HEAD。保护用户已有修改；若相关问题已修复，核对后保留，不重复实现。按本文完成代码、必要配置、文档和定向离线验证，不停留在再次输出方案，也不要只做第一阶段后结束。

本文规定目标行为；具体实现以当前调用链的最小必要改动为准。若发现某个预定删除对象仍有实际消费者或独立作用，保留该作用并说明证据，不为了逐字执行清理清单而损失功能。

本任务针对当前仓库。模型架构、通用训练 Dataset 与算法实验继续属于 `dexmani_policy`；需要核对相邻仓库接口时只读检查，并明确外部消费者是否仍需适配，不假称训练端已经兼容。

用户已经明确改变了以下旧规则：

- 全模态采集与全模态公共导出必须保留，不能按当前 checkpoint 裁减保存字段。
- technical-invalid、晚帧、停止、控制失败不再意味着自动丢弃整个 capture；保存已采数据，显式重录/丢弃才删除。
- 桌面、安装和触觉处理参数以代码当前配置为准，不要求逐段快照、历史版本绑定或参数一致性资格检查。
- 本任务书按用户明确要求放在仓库根目录。它是本次执行文件，不应因旧的临时文档位置约定被删除或移动。

将 `AGENTS.md` 中与上述要求冲突的长期规则同步更新。其他工作区保护和硬件边界继续遵守。此次实施只授权代码/文档修改和无硬件离线验证：不要连接机器人、相机、VR/HTS，不执行 HOME、replay、rollout 或实际标定；也不要把 `execute=False` 当作天然无硬件副作用。默认不替用户 commit/push 实现改动，不修改 Codex 权限或擅自升级实验环境。

## 1. 方案最终裁决

### 1.1 采用的简单结构

保留一个应用 SDK 调用所有者、当前具体 Robot/驱动、同步控制 loop、普通观测 deque、当前 action chunk、现有传感器进程/共享内存和写盘线程。沿用当前接口与目录，优先局部修正，不为了统一命名重写架构。

数据主线固定为：全模态 Raw → 全模态公共导出 → 下游选择字段/时间窗口。部署只为模型构造它需要的输入；开启 rollout 录制时仍采集全部物理模态。

### 1.2 本次复核对旧方案的进一步调整

1. 不强制给全部 13 个字段新增 `valid/<field>` 数组。浮点缺测复用 NaN，动作复用已有 dispatch，已有聚合/稠密触觉有效性继续独立。仅当现有表达无法区分两种必要状态时，才补一个具体字段；不创建 validity 管理层。
2. 不默认建立“相机缺帧 → 黑帧占位 → 继续运动”的路径。完整采集需要 RGB-D；没有可用相机观测或检测到持续停帧时停止本段并保存已采前缀。正常重复取到同一帧不等于故障，保留其源时间/帧号。
3. 不为混合任意分辨率、depth scale、名义采样率新增通用合并协议。保留现有密集数组布局；确实不能用同一布局和单位解释的数据分开导出。真实逐行时间仍必须保存。
4. 本轮不以新增 HOME 路点或另一种规划器替换现有规划。保留实际路径验证与碰撞检查，只删已确认无消费者/被支配的逻辑；搜索与验证模型的进一步统一列为有需求再做。
5. 历史桌面参数跟随当前代码不再列为待修 bug；安装参数的 10 mm 差异未经实物确认，不得为“数值统一”擅自改动几何。

### 1.3 参考项目的使用方式

| 参考及固定版本 | 借鉴 | 不整体移植 |
| --- | --- | --- |
| [LeFranX](https://github.com/wengmister/LeFranX/tree/a39906e6629f39490950fe8bd20f4f992ed74fd7) | 直接实验脚本、最近 N 帧、chunk 用完再推理；见 [部署循环](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_robot_deploy_dp.py#L328-L403) | 零动作补历史、额外平滑、具体固定等待实现 |
| [UFactory LeRobot](https://github.com/xArm-Developer/lerobot_robot_ufactory/tree/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82) | 薄硬件适配、显式采样/处理/下发、等待剩余预算；见 [采集循环](https://github.com/xArm-Developer/lerobot_robot_ufactory/blob/5fe6fb150bfa71ad3ca8a0b780c89ff8d4033a82/src/lerobot_robot_ufactory/scripts/_uf_lerobot/record_loop.py#L69-L154) | 版本兼容支路、异步保存体系；其记录值也不自动等于最终 SDK 目标 |
| [ManiUniCon](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686) | 实际源时间、明确状态/动作职责、SO(3) 工具；见 [时间保存](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/utils/timestamp_accumulator.py#L176-L211) | 全套 manager/queue/ready 握手；栅格填充不代表真实新采样 |

这些项目并非没有类型检查、并发机制或缺点。13 字段来自 dexmani_real 现有数据设计及用户需求，不声称三个参考项目采用相同 schema。

## 2. 全模态数据范围与参数来源

### 2.1 完整保留现有 13 个公共字段

以 `recording/storage/schema.py`、`recording/frame.py`、`dataset/contracts.py` 为当前字段依据。保留实际数据含义，不能用改名掩盖语义变化。

| 公共字段 | 每行形状 / 类型 | 来源或含义 |
| --- | --- | --- |
| `joint_state` | `(19,)` float32 | 实测 arm 7 + hand 12 关节，保持当前顺序与 rad 单位 |
| `arm_qvel` | `(7,)` float32 | SDK 速度反馈；保留并写清现有单位 |
| `arm_effort` | `(7,)` float32 | SDK 原生 effort，不凭名称宣称 SI 力矩 |
| `hand_current` | `(12,)` float32 | 当前手部电流语义/单位，不能当触觉力 |
| `action` | `(19,)` float32 | 最终尝试下发的 arm/hand 关节目标，结合各设备 dispatch 解释 |
| `action_ee` | `(21,)` float32 | arm 目标关节 FK 得到的 9D 位姿 + hand 12D 目标 |
| `contact_force` | `(5,3)` float32 | 聚合触觉，当前 SDK 原生、已扣偏置语义 |
| `tactile_force` | `(5,120,3)` float32 | 稠密触觉，独立于聚合触觉的有效性 |
| `fingertip_points` | `(5,3)` float32 | 实测关节经当前安装/运动学配置得到的指尖位置 |
| `eef_pose` | `(9,)` float32 | 实测 arm 关节 FK，位置 + 当前 rot6d 表示 |
| `rgb` | `(H,W,3)` uint8 | 采集 RGB；保持颜色顺序和采集分辨率 |
| `depth` | `(H,W)` uint16 | 现有对齐到彩色的深度，配套 depth scale |
| `point_cloud` | `(N,6)` float32 | 同一保存行 RGB-D 派生的 XYZRGB，保持当前坐标系/颜色含义 |

时间、相机帧号、dispatch、episode 边界和必要相机参数是配套信息，不假称它们已包含在这 13 个数组内。保持现有 Raw HDF5/视频和派生 Zarr 结构，不迁移整套 LeRobot 格式，不增加新格式家族或模态插件协议。

采集保存全部物理输入；点云、末端和指尖等可重建几何放在离线公共导出计算。在线模型真正需要某几何输入时，继续使用已有在线计算路径。不得为了保存全模态强制在线多跑全部 FK/点云，也不得只保存当前模型使用的子集。

这里的全模态指当前接入的机器人、RGB-D 和触觉观测及表中派生字段，不等于新增传感器、保存所有 SDK 原始包或实现任意模型架构。保持字段语义，不为本次需求扩展人体追踪数据集或原生高频独立流系统。

### 2.2 参数来源已由用户确定

| 项目 | 最终规则 |
| --- | --- |
| 桌面 / plane | 使用代码当前配置指向的参数；重导出按当时当前配置处理，不要求采集时快照 |
| 安装 / 运动学 | 使用当前安装变换、模型及配置；取消对历史 Raw 安装 metadata 的必需依赖 |
| 触觉处理 | 采集沿用当前驱动与偏置逻辑并保存实际输出；导出直接使用保存值，不再次扣偏置或改写历史测量 |
| 相机 | 保留数据对应的内参、外参、depth scale；此次简化不扩展到相机参数 |

不新增参数指纹、版本绑定、注册表或多级 fallback。已有桌面/安装/触觉副本不必清理，但不再是权威来源或准入条件。保留一次实际计算所需的有效参数检查；“使用当前配置”不表示零分母、坏矩阵或不存在的 link 可以照算。

不要为了去掉历史 metadata 依赖顺便更改安装数值。`config/hardware.py` 与 collision URDF 的 −15/−5 mm 差异涉及真实 adapter 说明，缺少现场证据时保留数值并报告待确认项。

## 3. 数据采集、保存与导出整改

主要入口：`recording/recorder.py`、`recording/frame.py`、`recording/storage/{schema,hdf5_writer,reader,video}.py`、`dataset/{contracts,processing,export,pointcloud}.py`、`examples/export_policy_zarr.py`。

### 3.1 保存前缀，删除整段实验资格审查

- 删除单个晚 tick、cadence 不合约、停止竞态、控制/IK 失败触发整段自动 discard 的默认规则。运动停止和数据保存分开。
- 正常结束或可关闭的控制中止：保存已提交前缀和简单结束原因、已有下发状态。未调用设备的目标继续按既有 NaN/dispatch 表达，不能伪造执行。
- writer/编码器/发布失败：尽力关闭各资源，保留原 staging 和异常路径；不要在 finally 中删除唯一数据副本。不能将损坏文件谎报为成功发布。
- 显式“重录/丢弃”才删除当前 capture。对已发布 Raw 保持不可变；不建设自动恢复器、修复扫描器、事务日志或两代备份协议。
- Zarr 是可重建缓存，默认新输出路径；显式覆盖时可按已有替换语义处理，失败留下新 staging。保留防误删输入/根目录等实际路径约束，不把 Raw 和缓存等同处理。

### 3.2 控制必需量、辅助测量、训练输入分开

- 当前动作实现需要的关节状态、实际设备错误仍决定能否继续运动；第一次缺少必要反馈就不发动作，连续读失败升级 FAULT 的 timeout 仍有作用。
- effort、current、聚合/稠密触觉的缺测不要连带把有效关节判断为不可控。已知辅助无效用现有 NaN/状态表达；不要把任意 SDK 协议错误都当辅助缺测吞掉。
- 不要求全部模态每一行 finite 才保存/导出。模型实际使用触觉等字段时，在其输入构造处要求可用；不喂 NaN、不把无效触觉当有效零力。
- 默认完整采集必须启用全部已接入模态。启动或运行时某通道持续缺失要清楚显示，不能悄悄运行一整段全空触觉并宣称完整数据。沿用已有可用性/失联判断，补简单计数或摘要，不增加多套超时门限与质量等级。
- 相机没有可用帧时停止本段、保住前缀，不新增默认补黑帧流程。相机仍在正常输出但控制频率更高时，可以重用原帧并保留原帧号/源时间；分别检测 RGB 和 depth 的持续停帧。

解耦必须覆盖实际 producer，不能只修改 `DexManiRobot`：`drivers/xarm7.py::_decode_joint_states` 当前先把 effort 与 qpos/qvel 一起做 finite 准入；`drivers/xhand.py::_parse_joints` 当前会因 current 非有限而拒绝整个关节包。对 SDK 返回正常、关节位置可用而辅助数值缺失的情形，在这些边界保留可用反馈与缺失值，再传到 Robot/录制/导出；SDK 错误码、关节包结构错误与实际控制器故障不能被放行。`qvel` 在 `arm_homing.py` 和驱动 HOME 收敛中有真实消费者，不能一并当成无用辅助量删除其到位判断。

### 3.3 时间只保存已取得且有用的信息

新增/传递普通逐行数组：控制观测时间、实际 arm/hand 读取完成时间、相机源时间、RGB/depth 帧号、下发完成时间。字段名沿现有风格选择，写清主机 monotonic/接收/读取/完成语义；无下发用明确缺失状态。已有源时间直接传递，不为本任务给每次 SDK 调用加 read_start/read_end。

使用现有 writer 分块写入，不为逐行数据建立无限增长的 Python 元数据列表。dispatch 语义复用；若调整存放位置，旧 Raw 中已有的 dispatch 仍应能读取，不建 migration 层。

- `control_hz` / `dt` 是名义目标，不是实际每行等间隔的证明。
- 旧 Raw 没有时间时明确“未知/按名义值估计”，不伪造实测时间，不让时间缺失阻止查看原数组。
- latest-sample 默认保留，不强制全局 skew barrier、跨设备时钟认证或 source-ID 完整性 contract。
- rollout 行表示控制观测与尝试动作，不等于每条动作都重新做了一次 query。写清这一点即可；本轮不建 query 输入快照/关联表。确实需要定位 chunk 时，最多复用/增加普通 chunk 索引，不作为读取资格。
- 同步 predict 阻塞期间不会凭空产生真实机器人采样；数据必须反映该时间缺口，不承诺原生最高频率的全量传感器记录。

新增时间/状态字段需要贯通 `EpisodeFrame`、实际 row producer、writer 的键集合/数组创建/追加、reader 和 exporter。当前 writer 使用精确字段集合；不能只在 frame 中添 key 就认为已经落盘。只扩展现有小表/数组，不新增 schema 管理器。

### 3.4 公共导出始终全模态

保留 `canonical_array_specs`、必要 shape/dtype 表和普通 `ProcessingConfig`。删除其无关的身份/来源准入，不因文件名叫 contracts 就拆文件重写。

把“全 Raw 浮点预扫描 → 转换 → 完整 Zarr 再读回”缩成一次实际分块转换/写入：

1. 输出全部 13 字段，不加入基于某模型 `observation_fields` 的导出裁剪。
2. 触觉/current/effort 等浮点缺失保留 NaN；FK 只算关节有效行，再放回原行序；空点云或缺重建点云所需的内/外参时保留该派生字段的 NaN 和清楚原因。不要删行后把时间缺口拼成连续轨迹。
3. 不强制新增 13 套 bool 数组。普通浮点 finite/现有状态足以判断时直接使用；实际缺失原因在现有 export report 中简短记录即可，不再增加质量报告子系统。
4. RGB-D 缺文件、无法解码、基本行数损坏不靠伪造图像“补全”。明确指出受影响文件；reader 仍允许访问其可读数值部分。已知不可导出的 episode 复用现有报告/显式排除方式；转换中失败则报告并保留 Raw/staging，不新增自动裁前缀、跨文件修复或自动重试。对已知缺测做局部分支，未知算法或 I/O 错误保留 traceback，不能 broad catch 后全部改 NaN。
5. 保留写入列长度、shape/dtype、episode_ends 和实际计数；删除 `_validate_staging` 的默认全缓存读回与自家 FK rot6d 的逐块重复正交认证。Raw 正常关闭后也不自动再过一遍完整 reader/视频资格扫描。
6. 同一密集 Zarr 的 H/W、类型、depth scale 等必须确实可由其元数据解释；保留这些实际兼容约束。不同分辨率/scale/名义 dt 的输入默认分开导出，不加多种合并策略，也不静默 resize、换单位或伪装同一 dt。真实逐行时间仍随导出保留。
7. task/source 是普通标签，不能仅因 `unknown` 或来源 `rollout` 禁止导出。不要把一个全局 task 标签错误覆盖到不同任务；本轮默认按任务目录组织缓存，混合任务另行显式组织，不建设任务身份系统。

相机参数的边界明确如下：Raw 采集不以已有有效外参为前置条件，`snapshot_recording_metadata` 应保存能取得的真实相机信息并说明缺项，不编造 identity 外参。读 RGB-D 像素不先要求外参；完整公共导出仍需可解码图像、明确形状和可解释深度的 scale，仅点云所需内/外参缺失时让点云无效，不连带拒绝关节、图像和触觉。若连深度 scale 都未知，不借“全模态”填一个假值；明确报告该数据尚不能作为完整可解释的 RGB-D 公共导出。

保存完整 RGB-D 与关节来源，点云裁剪/去桌面/固定 N 只是可重建结果。Raw 不做面向特定网络的 resize、归一化或删通道。当前深度已对齐、RGB 可能编码，不能宣传为相机所有原生字节都无损保存。

### 3.5 下游使用要正确，但不扩建训练框架

提供一个短的离线读取/说明示例：同一份公共导出，RGB 策略与触觉策略分别按所需字段选择有效行/窗口；动作标签结合 dispatch；不跨 episode/时间缺口拼接，不对所有模态求共同有效交集。归一化统计也只基于该策略采用的有效数据。

时间不均匀或一次慢推理本身不等于损坏，不能另加通用 `gap > 2dt` 删除规则。这里不拼接的缺口指已知缺失的所需样本或显式裁剪断点；固定频率模型如何根据真实时间选样/重采样，由它既有的训练约定决定。公共导出不插值、不为满足某模型的 dt 自动删行。

NaN 可保存在数据中，不代表所有旧训练 loader 可以直接消费。核对部署桥与已知消费者的真实假设；必要外部适配明确列出。此仓库不新增通用 Dataset、窗口筛选服务或训练 pipeline，不通过填零来“兼容”不支持缺测的模型。

## 4. 同步策略 eval 与节拍

主要入口：`deployment/{runner,config,operator,session}.py`、`runtime/observation.py`、`teleop/runner.py`、`replay/replayer.py`、`utils/rate.py`。先核对实际存在的文件和入口，沿现有调用链实施。

- 保持一个最近 N 帧 deque、一个 action chunk/索引，chunk 用完才同步 `predict`。不新增推理 worker、RTC、prefetch、动作 mailbox、ACK、时间调度对象或旧动作追赶机制。
- 修 teleop overrun 的额外整周期等待、replay 的“全部 I/O 后再睡完整 dt”。等待只覆盖应有间隔的剩余部分。
- 修慢推理后过期 deadline 造成 chunk 前两条现成动作挤压。以实际发送节拍安排后续动作，不补发落后的时间槽；不要为每个 action 建 deadline 对象。
- `n_action_steps=1` 时推理耗时计入发送间隔；长推理之后不再平白强加一个完整 cooldown。验证时同时覆盖单动作和多动作 chunk，不能只修一边。
- 不改变模型已定义的动作切片、horizon、归一化、动作顺序或 RNG 行为；不按墙钟跳过预测前缀。
- 同样保留该模型依赖的输入数值处理与特征顺序，包括已保存的点数/点云表示、图像处理和 fingertip 顺序。桌面/安装使用当前物理配置，不意味着把模型输入 recipe 随意换成当前默认值；沿用已有桥接字段即可，不新增 ABI 或指纹匹配。
- history 只在 episode 开始/重置时清空，不因超过 `2dt` 就 clear 并复制当前帧。启动先收集真实 N 帧；若已知训练明确采用启动 padding，按该约定使用，不能在运行中以重复帧伪造真实历史。
- 真实历史可能不等间隔，记录并说明；删除 history gate 不能被描述为已经解决所有时间对齐问题。
- predict 后继续沿现有逻辑检查停止/撤销/运行预算及必要当前反馈。新读取的关节反馈不改写模型已经使用的旧视觉输入，不伪装成新 history 行。
- 模型输入只计算实际所需字段；开启 rollout 录制则独立保证全模态采集。离线导出的 cloud 使用保存的同一行 RGB-D；在线同时使用 RGB/cloud 时沿现有 camera sequence 取同源数据，不增加完整 provenance 系统。
- 用 monotonic elapsed time 实现 duration。保留真正限制内存的有界队列/容量；删除“录制必须预先有限时长且估算少于固定 10000 行”的额外算法准入时，必须同时处理对应缓存增长，不能只删报错。
- 同步 predict/SDK 阻塞仍会延后软件 stop 的实际调用。保留这条能力边界说明，不新增任意 inference TTL、多级 max-age 或“软件实时保证”。

## 5. SafetyGate 与工程校验精简

### 5.1 保留真实作用

- 最终 SDK 入口的目标维度、finite、机械限位与设备错误。先检查 arm 和 hand 两个目标，再开始发送，避免发了 arm 才发现 hand 非法。
- 每次设备下发前检查当前撤销状态：arm SDK 调用期间可能按停，hand 前的再次检查有独立价值。沿用 `run_id`，不新增令牌/授权服务。
- SDK 单一应用调用所有者、必要数组所有权/拷贝、seqlock、写盘线程和有界队列。
- 当前必要反馈、真实失联/停帧检查、现有 IK 实际残差/自碰撞以及 HOME 路径检查。
- 外部模型 rot6d 转换的零向量/共线等数学退化检查。自生成 FK 的反复“标准格式认证”另行删除。

### 5.2 删除或合并无独立作用的规则

- 删除 `physical_home_completed` 作为 START 历史凭证及其全部依赖；HOME 保持显式动作。具体实验需要固定起点时检查当前姿态，不建设通用起点条件管理器。停止后仍撤销运动，不因删 token 自动恢复。
- 同一不可变 joint target 的中间容器不反复做 shape/finite。直接可调用的 driver 仍有入口保护；不要为省一次 12 维判断新增 `trusted`/`skip_validation` 协议。
- 只解析并检查实际选用的 retarget 后端、键盘模式和实际计算组件。保留 YAML/dataclass/CLI、拼写和正数等必要参数检查，不要求未用后端也全部可用。
- Raw reader 打开时不做全段 dispatch/finite/技术合格审查；读关节不应因缺外参失败。实际请求缺字段时正常报错。
- 模型加载检查它实际需要的 producer、动作维度/顺序/单位与计算配置即可。控制层需要关节反馈，不等于每个模型都必须把 joint_state 当网络输入。不要添加 checkpoint 与现场标定指纹绑定。
- task 标签普通处理，相对路径正常 resolve；删除身份许可证式限制，保留真实覆盖/删除路径保护。
- 删除已证实被当前逐关节限制支配的 IK gate；不能仅凭默认阈值就声称所有自定义配置永远冗余。保留有独立作用的条件，不再造自动“支配关系证明器”。

replay 保留当前起点检查和已由选项启用的 warm-up，不因为“大偏差先被拒绝”就删除它在允许范围内的对齐作用。大幅重新定位仍由操作者准备，不把 gate 移到运动之后。核对真正的准备终点：当前起点 gate 比较首帧实测 hand_qpos，而 warm-up 目标是首条 action_hand_joint，两者不可混称。若二者有差异，对实际准备目标复用现有小幅起点范围/限位判断，不能以“靠近首帧观测”推断“准备动作也很小”。不新增 prepare_start 状态机或放宽现有范围。

## 6. 数学、算法与实际错误

以下是定向修复，不是再次设计算法框架。

| 事项 | 实施要求 | 主要位置 |
| --- | --- | --- |
| 键盘 HOME 早于停止监听 | 先启用并确认输入监听，再允许显式 HOME；HOME 使用同一取消条件。不要再在不可停止窗口自动动手 | `teleop/keyboard_session.py`、`robot/hand_homing.py` |
| VR 跨 π 跳变 | 删除/修正对 principal rotvec 总范数裁剪的分支不连续；用标准 SO(3) 组合，不再叠 EMA 掩盖问题。不顺手删除已有有效的遥操作平滑 | `teleop/control/vr_mapping.py`、`action_proposal.py` |
| 标定评分 | 以实际位置残差 RMS 等有意义量选优，旋转用测地残差；修正门限字段/单位语义，不以残差模长 std≈0 宣称精确，不增加校准等级证书 | `calibration/camera/{solver,session}.py` |
| ArUco 稳定旋转 | 用 SO(3) mean/medoid 替代 rotvec 分量 median，平移可继续 median | `calibration/camera/session.py` |
| VR heading 样本 | 只累计新源帧；用圆均值/最短角差；修低方差样本全被剔除及空结果假成功 | `examples/calibrate_vr_heading.py`、`sensor/vr_worker.py` |
| 周期关节 | 周期性来自模型/驱动物理语义，不能由工作区间宽度推断；只对允许等价绕圈的轴选取接近当前姿态且在范围内的等价角。未知轴不猜 | `planning/paths.py`、`robot/projection.py`、`planning/kinematics/ik_geometry.py` |
| palm frame | 在真正用于构造 frame 的向量/三角形处做一次退化判断，删除检查另一三角形的无效 gate | `teleop/retargeting/retargeter.py` |
| flexion prior | 默认禁用已知把 abduction 混入 flexion 的 prior，保留其余优化目标；没有实验收益证据前不开发新屈伸分解系统 | `teleop/retargeting/retargeter.py`、`config/control.py` |
| TAG 门限 | 仅启用 TAG 时要求下门限严格小于上门限，消除归一化零分母 | `config/control.py`、`teleop/retargeting/tag_optimizer.py` |
| 异常被吞 | 普通无解使用现有结果；意外 RuntimeError/程序错误保留 traceback，并保存数据前缀。不要新建异常分类体系 | `teleop/control/{controller,hand_retargeting}.py` |
| replay 指标 | 无有效样本返回未定义和有效数，不返回完美零误差；lag 对有可识别运动的关节报告。绝对角误差与几何等价不可混为一谈 | `replay/evaluation.py` |
| close 假成功 | 一个资源 close 失败仍尝试其余资源，最终报告实际失败；不把 best-effort 返回当 verified clean 或物理停止证明 | `robot/robot.py`、drivers、`runtime/processes.py` |
| 首次桌面标定 | calibrate-table 分支不先要求旧 plane 存在；保留当前实际计算需要的相机几何 | `examples/pointcloud_process_example.py`、`config/experiment.py` |
| 非整数视频 FPS | 正确传递有理数帧率，避免 16.5 被 int 截成 16；科学时间仍看实际 timestamps | `recording/storage/video.py` |
| SciPy 声明下限 | 优先使用已有 xyzw/wxyz 转换，去掉不被所声明最低版本支持的 scalar_first 调用，不增加版本 fallback 管理器 | `teleop/jog.py`、`pyproject.toml` |

安装几何的现场真值、严格轨迹安全边界、同步推理实时化、headless 输入适配都不靠本次离线整改臆造。它们不阻塞其余已确认问题的修复。

## 7. 清理与文档

- 删无消费者的 `path_score` 及专为它计算的量；`rrt_range_options` 实际只用最大值时收敛为一个真实参数。不删被接受判定使用的 FK/碰撞计算。
- 无人消费的 workspace/arm/hand clip 统计删字段与计算，保留裁剪本身；不为使统计“有用”新建遥测面板。
- viewer 先决定所看帧，再流式/分块读取，不能 `max_frames=1` 仍预载整段 RGB-D；`--info`/`--help` 不依赖 rerun 的顶层导入。
- 提供一份能改 IP/序列号即可使用的 `experiment.example.yaml`，保持现有 YAML/CLI/`--print-config` 方式。
- README 聚焦依赖、首次准备、全模态录制、导出、查看、同步部署的短流程；写清 NaN/dispatch、实际时间、目标 vs 实测和同步 stop 边界。
- AGENTS 更新数据保留、全模态输出、三类参数使用当前代码和少量边界检查的规则。CLAUDE 仅保持有效入口，避免复制整套新规范。
- 不恢复历史已删的任务书、诊断平台或测试树，不为一次性任务再写一批永久架构文档。

## 8. 实施顺序与验证

### 8.1 顺序

1. **先消除直接危险/数据损失**：停止监听顺序、VR 跨 π 数学错误、Raw/staging 失败删除、technical-invalid 整段丢弃。
2. **打通数据和同步闭环**：全模态保存/导出、辅助缺测解耦、真实时间、相机停帧、节拍/history、duration、HOME token 与 reader/导出准入精简。
3. **修数学与整理负担**：标定/retarget/replay 错误、关闭异常、首次桌面标定、重复检查、死字段、viewer、依赖与示例文档。

阶段只是实施顺序，不是新 runtime 状态机。每阶段检查真实 diff，继续到全部本轮范围完成；已明确保留/接受/待现场确认的项说明原因，不强行制造代码改动。

### 8.2 只做有意义的离线验证

按现有 AGENTS 可用的无硬件检查运行：

```bash
python -m compileall -q dexmani_real examples
ruff format --check dexmani_real examples
ruff check dexmani_real examples
git diff --check
```

对以下真实风险补少量定向 pure-logic / fake-driver / 临时数据检查，验证行为而非照抄实现。不要恢复大套历史 tests，不设覆盖率/验证次数门槛，不把离线检查装回每帧热路径。

| 验证组 | 必须观察到的行为 |
| --- | --- |
| 停止与数据 | fake arm 调用期间撤销后 hand 不再下发；晚帧/正常停止保留前缀；模拟发布错误后原 staging 仍在；显式 discard 才删除 |
| 全模态 | 从仿真的原始 SDK payload 走真实驱动解码 → Robot → frame → writer/reader → export，覆盖有效关节 + 无效 effort/current；小样本导出 13 字段，双触觉独立有效；SDK 拒绝目标不能当成功执行。不能只绕过驱动注入高层假状态 |
| 时间与缺帧 | 原源时间/帧号确实经过 writer/reader/export，不只是内存里存在；正常抖动不触发新整段拒绝；RGB 冻结且 depth 推进仍被检测，缺帧不伪造有效黑图，旧无时间数据不假称实测 |
| 同步推理 | 假时钟覆盖慢 predict、多动作 chunk 和单动作模式；无旧 deadline 追赶、无多余整 dt cooldown；真实多帧历史不被复制成同一帧 |
| 数学 | 跨 π 连续、±179.5° 旋转聚合不变零；等大残差 RMS 非零；新 heading 样本计数、palm 退化、TAG 相等门限、允许周期等价轴均有反例 |
| 真实消费者 | 同一导出按 RGB/触觉所需字段构造不同有效样本集合；不跨 episode/缺口；不把 NaN 交给不支持它的模型；全字段 finite 不再是公共数据准入 |
| 参数与工具 | 不含历史安装/桌面/触觉参数仍按当前配置导出；触觉不重复扣偏置；缺外参仍可保存/读取物理观测但点云不伪装有效；无旧 plane 能解析首次标定分支；无 rerun 可 info；非整数 fps 正确 |
| 生命周期 | 一个 close 抛错仍调用其余 close，最终可见失败；纯 import/配置解析不创建硬件连接；全无有效数据的指标不是零误差 |

若缺可选依赖，完成不受影响的工作并精确说明未运行项，不能把 mocked 检查说成真实 SDK/硬件测试。对很小的配置/文档搬移不单独堆测试。上面列的是实施后的验收行为，不是此前已通过的测试成绩。

## 9. 原 43 项的完整处置索引

以下保持原编号以便追踪，不把 43 项都称为确定 bug。结论和执行范围以本文为准；定位链接指向审查时的固定源码，实施时检查当前调用链。无需为了让每行都有 diff 而修改“保留/接受/待现场”的项目。

| 原编号 | 事项与原核验结论 | 本轮处理 | 首要定位 |
| --- | --- | --- | --- |
| #01 | 键盘入口先执行手部 HOME，后启动停止键监听（成立） | 先启动停止键监听再允许显式 HOME；不新增启动审批状态。 | [dexmani_real/teleop/keyboard_session.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/keyboard_session.py#L34-L113) |
| #02 | 同步推理期间软件 stop 的执行会延后（能力边界） | 保留同步能力边界；修节拍/历史，不引入 worker 或实时保证。 | [dexmani_real/deployment/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/deployment/runner.py#L290-L345) |
| #03 | VR 总旋转角裁剪在 π 分支附近产生跳变（成立） | 修/删除有分支的总角度裁剪，使用标准 SO(3) 运算；不加额外平滑层。 | [dexmani_real/teleop/control/vr_mapping.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/control/vr_mapping.py#L101-L190) |
| #04 | 发布失败后删除 Raw 暂存；覆盖缓存可能两份都丢（成立） | Raw 和失败 staging 保留；派生缓存显式覆盖即可，不建自动恢复或备份事务。 | [dexmani_real/recording/recorder.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/recording/recorder.py#L498-L547) |
| #05 | 停止竞态、单个晚帧或控制失败导致整段采集丢弃（设计取舍） | 停止与保存分开，保留前缀；只有操作者显式重录/丢弃才删除。 | [dexmani_real/teleop/control/controller.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/control/controller.py#L93-L139) |
| #06 | Raw 缺少逐行时间与源帧标识，标称频率不能恢复真实节拍（成立） | 落盘已有源时间、相机帧号、观测/下发完成时间；不建全 SDK 事件表。 | [dexmani_real/recording/storage/schema.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/recording/storage/schema.py#L20-L54) |
| #07 | freshness 通过不代表多模态对齐；SDK 完成时间不等于采样时间（能力边界） | 保留 latest-sample 并写清时间语义；无严格全局 skew barrier。 | [dexmani_real/runtime/observation.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/runtime/observation.py#L69-L133) |
| #08 | teleop 超时额外等一周期，replay 则把 I/O 加进每周期（部分成立） | 修开始到开始的剩余预算等待；不加通用调度平台。 | [dexmani_real/teleop/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/runner.py#L266-L290) |
| #09 | 慢推理后 chunk 前两步可挤在一起（成立） | 慢推理后局部重设下一步执行时间，防第一、第二条动作挤压。 | [dexmani_real/deployment/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/deployment/runner.py#L277-L346) |
| #10 | 慢推理反复清空观测历史，复制当前帧掩盖时间缺口（成立） | 用最近 N 个真实观测的 deque；启动采满，运行中不因慢推理清空并复制。训练时间假设另行核对。 | [dexmani_real/runtime/observation.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/runtime/observation.py#L136-L156) |
| #11 | 推理后刷新关节反馈不会刷新本次视觉输入（能力边界） | 接受同步视觉延迟边界；不增加无任务依据的 action TTL。 | [dexmani_real/deployment/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/deployment/runner.py#L290-L345) |
| #12 | rollout 缺少 query 到执行行的关联，cloud-only 图像来源可能不同（部分成立） | 明确执行行不是每次 query 输入；保存真实来源时间，禁止来源误述，不建完整 query 表。 | [dexmani_real/deployment/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/deployment/runner.py#L290-L396) |
| #13 | RGB-D 联合 stall 检查可能漏掉单路冻结（成立） | 分别检测 RGB/depth 持续停帧；无可用相机观测则结束并保留前缀，不默认补黑图。 | [dexmani_real/sensor/camera/realsense.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/sensor/camera/realsense.py#L568-L655) |
| #14 | 标定用残差模长的标准差选优和放行（成立） | 用实际残差 RMS 等量修正评分；不新增校准等级/证书机制。 | [dexmani_real/calibration/camera/solver.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/calibration/camera/solver.py#L216-L305) |
| #15 | ArUco 稳定检测对 rotvec 作分量中位数（成立） | 旋转均值/medoid 用 SO(3)；直接修数学。 | [dexmani_real/calibration/camera/session.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/calibration/camera/session.py#L70-L112) |
| #16 | VR heading 校验把同一旧头部帧反复计为新样本（成立） | 只累计新的 head 帧，修圆统计边界；不造多级质量 gate。 | [examples/calibrate_vr_heading.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/examples/calibrate_vr_heading.py#L61-L142) |
| #17 | 观测手部安装变换与碰撞 URDF 相差 10 mm（部分成立） | 当前安装配置为准；10 mm 装配真值待现场确认，不能在本轮盲改几何。 | [dexmani_real/config/hardware.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/config/hardware.py#L264-L272) |
| #18 | 缩窄工作关节区间改变等价角选择（成立） | 在物理语义明确允许周期等价的轴修正选角；未知/有圈数约束的轴不猜。 | [dexmani_real/planning/paths.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/planning/paths.py#L132-L178) |
| #19 | 手掌有效性三角形与估计器使用的三角形不一致（成立） | 在实际 palm frame 计算处检查退化，删检查另一三角形的冗余入口。 | [dexmani_real/teleop/retargeting/retargeter.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/retargeting/retargeter.py#L121-L195) |
| #20 | human flexion prior 把侧向张指算成屈曲（成立） | 先禁用有错误的 flexion prior 做消融；不另加“纠偏”机制。 | [dexmani_real/teleop/retargeting/retargeter.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/retargeting/retargeter.py#L73-L108) |
| #21 | TAG 相等 pinch 门限获准但计算出现零分母（成立） | TAG 启用时检查一次门限严格排序；不做每帧配置审查。 | [dexmani_real/config/control.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/config/control.py#L171-L212) |
| #22 | 历史 Raw 导出默认套用当前桌面平面（原部分成立；本轮接受该选择） | 按用户明确选择接受当前桌面/安装/触觉参数；取消历史快照/版本绑定要求。 | [examples/export_policy_zarr.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/examples/export_policy_zarr.py#L129-L138) |
| #23 | replay 的角度、无有效帧和延迟指标需要分开修正（部分成立） | 无数据返回未定义；活动关节单独报告 lag；绝对角/几何误差说明清楚即可。 | [dexmani_real/replay/evaluation.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/replay/evaluation.py#L27-L60) |
| #24 | retarget 技术异常被吞成普通控制失败（成立） | 保留原始异常栈；普通无解用现有返回值，不建异常分类体系。 | [dexmani_real/teleop/control/controller.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/control/controller.py#L55-L90) |
| #25 | 未使用的辅助模态与控制/导出耦合（部分成立） | 从驱动解码到 Robot/存储/导出贯通辅助缺测解耦，保留关节与真实硬件错误边界；沿用 NaN/状态。 | [dexmani_real/robot/robot.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/robot/robot.py#L141-L205) |
| #26 | 未启用后端、固定全模态 Raw 也参与准入（设计取舍） | 保留全模态字段及真实 shape/单位约束；删未用后端准入与全模态全有效资格规则。 | [dexmani_real/config/experiment.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/config/experiment.py#L129-L155) |
| #27 | physical_home_completed 是历史令牌而非当前姿态证明（设计取舍） | 删 HOME 历史凭证，保留手动 HOME。 | [dexmani_real/runtime/safety.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/runtime/safety.py#L129-L151) |
| #28 | replay 手部接近检查先于手部预备动作（部分成立） | 保留当前起点检查与有作用的可选 warm-up，核对首帧状态和首动作目标差异；大幅定位手工完成。 | [dexmani_real/replay/replayer.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/replay/replayer.py#L144-L180) |
| #29 | HOME 的 RRT 搜索与最终验证不在同一碰撞模型中（部分成立） | 本轮保留现有 HOME 规划/路径验证；不新增路点替代方案，不把搜索低效误称无碰撞检查。 | [dexmani_real/planning/planner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/planning/planner.py#L96-L177) |
| #30 | workspace 裁剪只约束 EEF 意图，不保证最终 FK 在界内（能力边界） | 说明 target bounds 的真实作用；不扩成全轨迹安全认证。 | [dexmani_real/robot/action.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/robot/action.py#L104-L117) |
| #31 | SDK close 错误被吞后上层仍报告 clean（成立） | close 尽力处理所有资源并报告异常，不再包装成“已证明 clean”。 | [dexmani_real/robot/drivers/xarm7.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/robot/drivers/xarm7.py#L212-L219) |
| #32 | 首次桌面标定要求已有有效桌面标定（成立） | 首次 table 标定不读取旧 plane；一个分支修改即可。 | [examples/pointcloud_process_example.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/examples/pointcloud_process_example.py#L409-L471) |
| #33 | max_record_duration 用行数计算，慢采集可超时（成立） | duration 用 elapsed；去额外预估预算准入时同步保证缓冲有界，实际容量限制可保留。 | [dexmani_real/teleop/runner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/teleop/runner.py#L51-L91) |
| #34 | 视频编码把非整数帧率截断（成立） | 正确传递非整数 fps；科学时间用真实 timestamps。 | [dexmani_real/recording/storage/video.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/recording/storage/video.py#L124-L142) |
| #35 | 声明 scipy>=1.10 却使用较新 scalar_first 参数（成立） | 复用已有 xyzw/wxyz 转换，或声明真正所需版本；不造兼容管理器。 | [pyproject.toml](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/pyproject.toml) |
| #36 | 默认 IK 多道连续性检查存在不可达分支（部分成立） | 只删实际配置中确无独立作用的条件；不能从默认阈值推断全部配置。 | [dexmani_real/planning/kinematics/ik.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/planning/kinematics/ik.py#L44-L88) |
| #37 | path_score 无消费者，rrt_range_options 实际只用最大值（成立） | 删无消费者 score；多候选参数改为实际使用的一个值。 | [dexmani_real/planning/planner.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/planning/planner.py#L241-L258) |
| #38 | 动作裁剪统计计算后无人使用（设计取舍） | 无使用者就删 clip 统计字段；保留裁剪运算，不强制开发新遥测。 | [dexmani_real/robot/action.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/robot/action.py#L25-L33) |
| #39 | --info 被顶层 rerun 导入阻塞（成立） | --info 走轻依赖路径，rerun 按需导入。 | [examples/visualize_episode.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/examples/visualize_episode.py#L1-L67) |
| #40 | viewer 限制帧数之前先完整解码视频（成立） | 先选帧再读取，流式/分块即可，不建设多层缓存。 | [examples/visualize_episode.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/examples/visualize_episode.py#L111-L236) |
| #41 | 交互入口依赖 GUI 全局键盘，限制 headless 使用（能力边界） | 保留现有桌面输入并说明依赖；无实际需求不做 headless 后端。 | [dexmani_real/runtime/operator_input.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/runtime/operator_input.py#L295-L388) |
| #42 | state_read_failure_timeout 没让 episode 自动宽限（撤回） | 原缺陷判断撤回：保留必要反馈丢失停动作与持续故障升级两层语义。 | [dexmani_real/robot/robot.py](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/dexmani_real/robot/robot.py#L150-L188) |
| #43 | 文档需要可执行最小示例，并修正丢弃约定（部分成立） | 给可用 YAML 和短流程，同步更新 AGENTS；明确全模态、NaN/dispatch、当前参数与时间边界。 | [README.md](https://github.com/haoyangzhanglab/dexmani_real/blob/b7fc4549931314af404082251a2331ffe5ed4ce8/README.md) |

## 10. 完成时交付

提交给操作者的结果应包含：实际修改摘要、关键文件、执行过的离线检查与真实结果、未验证的硬件/依赖边界、原编号对应的已完成/保留/不适用/待现场项。不要只报“全部通过”，也不要把能力边界当成必须增加框架的理由。

最终自审只回答实际问题：

- 采集和公共导出是否仍覆盖 13 字段，且没有被某个模型的输入键裁减？
- 三类指定参数是否直接使用当前代码，触觉是否避免重复处理？
- 停止/失败是否保住数据，真实 SDK/碰撞/撤销边界是否仍在？
- 是否消除了已发现的数学/节拍错误，而非追加另一层 gate？
- 是否又引入了新 registry、contract、校验报告、后台 worker 或不必要状态机？若有且本文无具体必要性，删掉。
- 文档、示例与当前实现是否一致，现有工作树修改是否得到保护？

不执行真机“补验收”，不自动推送实现代码。完成已授权的离线实施后，再把确实需要现场确认的事项清楚交给操作者。
