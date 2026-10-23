# DexMani Real 数据与实验流程简化任务书（Codex CLI）

> 状态：待实施。本文是维护者明确要求写入仓库的**新任务书**，不是代码已完成或真机已验证的报告。
>
> 日期：2026-10-23。Real 分析基线：`6d96100afd4c8a84907a012e0976b44a39dbb962`。
>
> 执行目标：完成代码、必要文档和离线验证，不停留在分析，也不只完成第一批便结束。有实际阻塞时完成不受影响的部分，准确报告阻塞。

## 1. 最新需求、范围与优先级

这是个人 PhD 灵巧操作研究代码。目标是在现有可用链路上实现：**Sim/Real 的目标数据模态对齐；采集只存必要物理来源；派生计算复用；实验入口直接；其余数据与协调机制不比 LeRobot 更复杂。** 不以类数、行数或文件数代替质量判断。

维护者的明确要求：

1. `dexmani_sim` 采集的目标模态，`dexmani_real` 也应具备。
2. `object_pointcloud` 暂不接入；`segmentation` 暂不采集；`done` 不采集。
3. 参考 LeRobot 和 ManiUniCon 简化其他机制，而不照搬它们的全部框架。
4. 保留完成论文实验所需的控制、采集、导出、部署和评估能力。

据此，本任务将公共 Canonical 数据收敛为第 3 节的 **12 个业务字段**：删除当前公共数据中的 `arm_qvel`、`arm_effort`、`hand_current`、`taxel_tactile_force`，加入 `hand_pointcloud`。这四项辅助信号不再默认持久化或作为训练完整性条件；设备内部确有控制、HOME、静止判断用途的读取仍保留。不要把“没有训练字段”误解为“删除设备反馈”。

本任务明确替代旧 `CODEX_REFACTOR_TASK.md` 中“保持 15 字段、Raw 必须两类触觉、必须持久化 query/arm_execution 完整证据、触觉来源时间不在范围内”的要求，以及 `AGENTS.md` 中与上述新数据范围冲突的条款。实施时同步更新长期规则。其余运动授权、急停、物理时间、Raw 不可变、无硬件验证等约束继续适用。根目录新增本文是维护者指定的例外，不授权新增一批临时报告。

代码主体在 `dexmani_real`。`dexmani_sim`、LeRobot、ManiUniCon 为只读参考。`dexmani_policy` 只允许为本次新输入完成必要的部署契约适配，见第 6 节；不修改模型、损失、训练算法或实验划分。

## 2. 参考依据及应采用的部分

以下为本任务核对过的固定版本。执行时检查当前 HEAD 的相关增量，不回退仓库到这些版本，也不重复实施已经完成的重构。

| 项目 | 核对版本与入口 | 采用的设计 | 本任务的取舍 |
| --- | --- | --- | --- |
| dexmani_real | [`6d96100`](https://github.com/haoyangzhanglab/dexmani_real/tree/6d96100afd4c8a84907a012e0976b44a39dbb962)；`data/`、`runtime/observation.py`、`deployment/` | 保留现有 Raw→Canonical、设备所有权、授权与调度基础 | 已有纯计算 `deployment/schedule.py` 和配置归位不重写；优先消除数据与运行审计耦合 |
| dexmani_sim | [`607ae7c`](https://github.com/haoyangzhanglab/dexmani_sim/tree/607ae7c805b18022518a381dd78921dfa655df95)；[`env_recorder.py`](https://github.com/haoyangzhanglab/dexmani_sim/blob/607ae7c805b18022518a381dd78921dfa655df95/dexmani_sim/mimic_gen/utils/env_recorder.py)、[`model_point_cloud.py`](https://github.com/haoyangzhanglab/dexmani_sim/blob/607ae7c805b18022518a381dd78921dfa655df95/dexmani_sim/sensors/model_point_cloud.py) | 模态命名、关节/末端表示、动作前观测、固定手部表面点随 FK 运动 | 不接入三项排除字段；不把仿真接触力或对象真值假装成真机测量 |
| LeRobot | [`d3793b7`](https://github.com/huggingface/lerobot/tree/d3793b7e36522321c1c455f0dc4826db3793dceb)；[`lerobot_record.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/scripts/lerobot_record.py)、[`dataset_writer.py`](https://github.com/huggingface/lerobot/blob/d3793b7e36522321c1c455f0dc4826db3793dceb/src/lerobot/datasets/dataset_writer.py) | 直接的 observe→action→send→add_frame 循环；简单字段声明和 episode 写入接口 | 不引入通用 processor、Hub、Parquet 分片或新存储格式；保留真实时间与提交目标语义 |
| ManiUniCon | [`85c6f2e`](https://github.com/Universal-Control/ManiUniCon/tree/85c6f2e32ecf9f2bed62d202b058c39623444686)；[`main.py`](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/main.py)、[`core/robot.py`](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/core/robot.py) | 设备执行、传感器与策略各有明确职责；控制频率与推理频率分离 | 不搬 Hydra、共享标志集合、新 IPC 或多套结束路径；复用本仓现有隔离机制 |
| dexmani_policy | [`4437d58`](https://github.com/haoyangzhanglab/dexmani_policy/tree/4437d58dd7422536562cef1d774c35d613ad7d5d)；[`deployment/runtime.py`](https://github.com/haoyangzhanglab/dexmani_policy/blob/4437d58dd7422536562cef1d774c35d613ad7d5d/dexmani_policy/deployment/runtime.py)、[`metadata.py`](https://github.com/haoyangzhanglab/dexmani_policy/blob/4437d58dd7422536562cef1d774c35d613ad7d5d/dexmani_policy/deployment/metadata.py) | artifact 决定模型输入、历史长度、动作表示与训练数值 recipe | 当前 warmup 尚未支持 depth/hand_pointcloud；必须核对完整消费链，不能仅修改 Real 的字段白名单 |

不要把参考实现误读为更强保证：LeRobot 此版本记录循环未使用 `send_action` 的返回值构造 action frame，writer 的时间戳为 `frame_index/fps`；这些都不适合作为本仓真实提交目标和真实采样时间的依据。ManiUniCon 的 [`timestamp_accumulator.py`](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/utils/timestamp_accumulator.py) 会重复样本填时间槽，[`process_demo_data.py`](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/tools/process_demo_data.py) 有按最短长度裁切的处理；本任务不采用这种隐式补齐/裁切。

## 3. 唯一公共数据契约：12 个字段

以下是每帧形状；落盘增加前导 `T`。`N=1024`、`Nh=1024` 为默认值；确需 512 个手点时通过明确 recipe 配置，并与 artifact 一致。字段声明继续使用一个普通 specs 字典，不新增模态注册框架。

| 字段 | 每帧形状 / dtype | 数值含义与来源 | 采集时是否重复保存派生结果 |
| --- | --- | --- | --- |
| `joint_state` | `(19,)` float32 | 实测 arm 7 + hand 12，固定关节顺序，rad | Raw 保存原精度关节；导出拼接 |
| `finger_tactile_force` | `(5,3)` float32 | 已消费的五指聚合三轴触觉；thumb/index/middle/ring/pinky | 保存消费值及必要来源时间 |
| `action` | `(19,)` float32 | 映射、等价关节投影、限位等处理后，实际提交到 arm/hand 发送边界的绝对关节目标，rad | Raw 保存提交目标和简短 dispatch 状态 |
| `action_ee` | `(21,)` float32 | 提交的 arm 目标 FK：xyz(3)+rot6d(6)，拼接提交的 hand 目标(12) | 导出派生 |
| `eef_pose` | `(9,)` float32 | 实测 arm 关节 FK，URDF `custom_eef_link` 的 xyz+rot6d | 导出派生 |
| `fingertip_xyz` | `(5,3)` float32 | 实测 arm+hand 关节、手部安装变换和五个指尖 link 的 FK | 导出派生 |
| `rgb` | `(H,W,3)` uint8 | 相机 RGB，与 depth 对齐到 color grid | Raw 视频 |
| `depth` | `(H,W)` uint16 | 对齐到 color grid 的原深度计数；米值为 `depth * depth_scale_m` | Raw 无损保存 |
| `point_cloud` | `(N,6)` float32 | 同一 RGB-D 帧反投影得到的可见场景 XYZRGB | 导出派生 |
| `hand_pointcloud` | `(Nh,6)` float32 | 固定手部 mesh 表面样本经实测关节 FK 得到的模型 XYZRGB | 导出派生 |
| `camera_intrinsic` | `(9,)` float32 | `K(3,3)` 按 C-order 展平 | Raw 元数据存一次；导出按帧提供 |
| `camera_extrinsic` | `(12,)` float32 | `T_color_from_base[:3,:4]` 按 C-order 展平，即 base/world→相机 optical 坐标 | Raw 元数据存一次；导出按帧提供 |

必须落实以下语义：

- Real 的公共几何坐标为 xArm base；Sim 对应 robot-base/world。位置单位 m。相机 optical 轴为 right/down/forward；投影使用 world/base→camera 的 `[R|t]`，反投影到 base 使用其逆，不能仅按字段名猜方向。
- 公共关节向量保持 `joint1…joint7` 后接 `robot/model.py:XHAND_SDK_JOINT_NAMES`；与 Sim 的 `robots/urdf_config.py:XHAND_JOINT_BASE_NAMES` 逐名核对。SDK→URDF/Pinocchio 的重排只发生在 FK 内部，不把内部字母排序后的关节顺序泄漏到 joint_state/action。
- rot6d 固定为 `concatenate([R[:,0], R[:,1]])`，不是 `R[:,:2].reshape(-1)`。四元数内部若使用，明确 wxyz；不要在公共数据中再重复一套 quat 表示。
- `action` 是提交目标，不是原始 policy/VR 输出、实测关节、SDK 接受证明或已经到位。`action_ee` 从同一提交目标计算，不能保存未达成的原始 EEF 请求冒充它。
- 每一行匹配 `obs[t]` 和其后本次决策提交的 `action[t]`，不误移成 `obs[t+1]`；Raw 保留各来源实际采样时间，异步传感器不冒充同时采样。
- 两类点云的 RGB 都采用 `[0,1]`；场景点云为相机色彩，手点云为模板固定材质色。手点云是模型几何，不是相机可见手点，也不是分割结果。
- 深度零值保留其无效像素含义；不能把 uint16 一概称为毫米。Sim/Real 深度单位差异通过明确 scale/消费转换处理，不静默混用或添加第二套 depth 字段。
- Real 聚合触觉当前来自 SDK 原生值；没有证据时不能标成 Newton、world-frame 或声称等价于 Sim 接触力。保存真实单位/来源/坐标约定，未知则明确未知。本轮不新增强制力标定项目，不捏造旋转或缩放；涉及跨域触觉数值等价性的实验应单独验证。

公共训练缓存只有这 12 个业务字段。`episode_ends`、时间、dataset attrs 属于必要索引/元数据，不计作额外模态。**三个排除字段不得以占位数组、全零、全 False、末帧 True 或自动生成标签的方式出现。** Episode 边界用长度/`episode_ends`；可选人工任务成功标签只放 episode 评估摘要，不改名伪装为逐帧 done。

所有方法使用同一套合格 episode 及相同数据划分，模型按需读字段；不为不同模型维护不同 Canonical schema、导出 profile 或静默筛出不同 episode。

## 4. 最小存储与录制设计

### 4.1 继续沿用两层数据，不迁移格式

保留现有 HDF5 + RGB 视频作为 Raw、Zarr 作为可重建 Canonical。只收敛内容与接口，不新增 LeRobot 格式副本、NPZ 备份、缓存迁移系统或统一 dataset 框架。

Raw 默认保存以下必要来源，优先沿用现有命名：

| 内容 | 最小保留范围 |
| --- | --- |
| 物理观测 | `arm_qpos`、`hand_qpos`、`hand_contact`、对齐 depth、RGB；不因本次重构降低原有测量精度 |
| 动作 | `action_arm_joint_target`、`action_hand_joint_target`、现有 arm/hand 两列 `dispatch_status` |
| 逐行时钟 | 实际 observation/control 时刻、arm/hand 来源时刻、aggregate 触觉原有效时刻、camera 来源时刻；只在实际配对需要时保留 camera sequence |
| 采集元数据 | task/source、预期 control dt、相机 K/E/畸变模型/depth scale、关节/触觉语义、触觉保持时限、格式标识 |
| 简短结束信息 | 原因、错误、实际帧数、停止结果（确认/未确认/不适用），明确区分技术失败与人工任务失败 |

删掉默认 Raw 中四项辅助物理字段、完整 query 历史、没有消费方的收发/发布/计数镜像。对当前 23 列 timing 表逐项追踪消费者，以上最小集合之外仅保留确实支撑对齐或科学解释的项；不要机械删掉用于 RGB/点云配对的来源标识。调度内存中的 query_id、run_id、slot、deadline、冻结前缀等继续服务控制，不要求全部永久写入 Raw。

不把手 FK、场景/手点云、EEF、指尖位置加入 Raw 的重复缓存。真实运行若模型需要它们才在线计算；录制本身不触发整套感知派生。

### 4.2 Recorder 只负责 I/O 生命周期

Recorder 的对外概念收敛到 `start_episode / add_frame / save_episode / discard_episode / close`，或现有直接等价接口。不增加 facade 包裹旧 recorder，也不引入无状态函数到有状态 writer 之间的新层。

1. 删除 `start_episode` / `save_episode` 对完整 `arm_execution` 快照和 SDK 计数守恒的强制参数；删除标准路径 `record_query` 与只供它使用的 evidence 组装。
2. Recorder 不导入机器人执行审计来判定“可写文件”。Raw 发布条件是 writer 成功关闭/flush、实际写入行与视频帧一致、发布原子完成。设备停止由 runtime 的业务所有者负责。
3. **文件完整、技术上正常、任务成功是三个不同结论。** 停止未确认或部分发送失败仍可保存真实且 I/O 完整的 Raw，但结束信息必须体现失败；不能自动成为合格示教或可回放动作。
4. 保留一个 writer owner、现有有界队列、后台编码、错误回传、必要 ready/cancel/join 与 staging→原子发布。队列满应显式结束本次录制并报告，不能无限占用内存、阻塞控制长时间或静默丢帧。
5. 对可变数组在跨线程所有权边界做一次必要快照；不要层层 deep-copy。写入/编码失败保留 staging，不能冒充已发布 Raw。
6. 技术失败保存已采前缀和真实终止原因。确实零观测且零动作的尝试可只保留简短尝试摘要；“零行”不等于“从未发过动作”，存在发送或部分发送时仍需保留相应失败证据。
7. 显式丢弃只作用于当前 capture，必须先让 writer 退出再删其 staging；不删除已发布 Raw。正常发布后 Raw 不原地修改。

`dispatch_status` 继续区分 NOT_CALLED、SUBMITTED、ACCEPTED、失败及现有 CRC_UNCONFIRMED 等含义。CRC_UNCONFIRMED 按现有设备协议处理，不能改写成 ACK。保留 host 提交与设备反馈的必要语义，不继续为数据发布维护完整 SDK 指令对账本；控制安全需要的 worker 内部计数不因此删除。

### 4.3 真实时钟、缺测和训练准入

- 使用现有 host monotonic 实际时刻；wall clock 只用于 episode 描述。`frame_index/fps` 是名义网格，不替代真实时间。相机、控制和 arm servo 频率保持不同含义。
- 保留现有有界上一有效触觉向量保持，在唯一消费入口处理；记录保持上限与原有效时间。至少保存聚合值中最老的原有效时间，作为保守新鲜度界限，不把手关节本次读取时间当作触觉时间。离线不再保持、扣偏置或二次修补。
- Raw 可表达缺测/失败；Canonical 的 12 字段必须 shape/dtype 正确、浮点有限、语义与时间有效。缺少必要来源或技术无效时整段拒绝并报告原因，Raw 留存。四项已移出公共契约的辅助字段不再影响训练准入。
- 不通过补零、复制末动作、删掉坏帧、按最短数组裁切、补时间槽或静默插值使数据合格。允许点云 recipe 内对同一帧真实非空点集做已声明的固定点数采样；这不允许伪造整帧空点云或补传感器时刻。
- rollout 的 WAIT/漏槽/部分发送按事实保存。没有真实提交目标时不生成一个假的 action；默认不把这种 rollout 自动当固定频率示教导出，也不删除 WAIT 后压缩物理时间。
- 移除 `EpisodeReader.validate_arm_execution()` 作为 Canonical/回放入口对完整 SDK 日志的强依赖。改为必要源数据、明确 dispatch、结束结果与真实时间检查；回放的现场安全预检继续独立存在。
- 已发布旧 Raw 不批量修改。若旧文件包含所需来源可由简单 reader 读其必要子集，忽略额外字段；缺少新要求的信息则明确受限/拒绝相应严格用途，不能伪造时间和标定。不构建旧缓存兼容或自动迁移体系。

## 5. 派生几何：离线优先、在线复用

沿用现有 `planning/kinematics`、`sensor/pointcloud/processing.py` 与 `data/canonical`。只增加确有需要的轻量手点云计算模块，不能建立新 PerceptionManager 或 processor DAG。

### 5.1 手点云的最小实现

1. 读取与当前真机匹配的 XHand URDF/visual mesh，确定 12 个关节名、手 link 集合与安装变换；对照 Sim 的模板生成和 link 选择。不得把 xArm、法兰等无关几何算成手点云而不说明。
2. 初始化/离线准备时固定种子、按 link 面积采样固定数量局部表面点，保存固定 link 对应和模板颜色；每帧只做 FK 与刚体变换。禁止每帧重新读 mesh、采样、FPS 或从 RGB 猜手颜色。
3. 使用同一行**实测** arm+hand 状态，将各 link 局部点变换到 xArm base；手安装变换与 `fingertip_xyz` 一致。不能用 action、home 或零关节补观测。
4. 优先复用本仓已有 `third_party/cordvip_geometry/src/cordvip_geometry/hand_cloud.py` 的纯 URDF/固定采样思想与已有 FK。该类当前输出 camera-frame XYZ/normal，不等于目标 base-frame XYZRGB，需要明确适配。若抽取实现，保留来源说明，只保留本任务所需能力；不要复制整个追踪包或创建第二套长期 FK。
5. 固定颜色来自一致的模板/URDF 材质；缺少材质时使用明确记录的固定颜色规则。法向量、link_id 可以是内部计算辅助，不加入公共训练字段。
6. 模板/模型缺失时清楚报错，不退化成分割、相机可见手点、全零点云或对象追踪。

主路径不依赖 SAM、DINO、FoundationPose、Hunyuan、对象 mesh、对象位姿、分割 mask 或 object_pointcloud。已有独立研究工具可保留，但其重依赖不能通过 package import 泄漏到配置解析、无图像部署或默认采集。

### 5.2 共享计算与可复现范围

- EEF、指尖、手点云、场景点云的数值函数由离线导出和在线观测共同调用；具体资源在 session 初始化，不把完整 ExperimentConfig 传到数学函数。
- 关节 FK 用原有测量精度计算后转 float32。Pinocchio 的可变 `Data` 等工作空间必须由使用线程/worker 独占，不能为了缓存共享一个会被并发修改的计算器。
- 场景点云保留当前可用的质量过滤与 recipe 数值，不借职责重构暗改阈值、裁剪、采样算法或物理分布。只合并重复计算、消除重复转换和无用分支；若要改变 recipe，作为明确数值变更单独验证后再决定，不把更少过滤等同更好数据。
- 在现有 dataset attrs 中记录实际派生 recipe（采样数/种子、模型/关节/link 对应、颜色规则、处理参数及本次实际使用的安装/桌面配置），静态值存一次。不新增逐帧 manifest、证据图或严格历史标定匹配系统。
- 相机参数必须来自该 Raw 的相机快照。安装/桌面属于当前实验配置，可按现有规则选择；导出记录实际采用值即可，不强制历史 Raw 补齐安装快照，不将历史标定当作新实验 ABI。
- RGB 视频当前有损编码会使解码颜色和在线原图略有差异；共享函数并不意味着点云颜色逐位一致。合成输入验证函数等价，真实编码路径分别检查几何和颜色容差；不为追求假等价默默更换编码方案。
- 导出按 block/stream 处理，HDF5 和视频各自合理打开一次；避免整段 RGB-D 或多份点云驻留内存、重复先探测再打开同一 HDF5、先写 NaN 派生数组再覆盖重写。

## 6. 在线观测与 policy 边界必须同时打通

12 个字段中 `action/action_ee` 是监督目标；其余 10 项应有清楚的 Real 观测来源。当前 `preflight.py`、`deployment/observation.py`、`ObservationNeeds` 尚缺 `depth/hand_pointcloud` 的完整路径，必须补齐准入、来源时钟、历史拼接、按需资源和数值校验。

用两类简单需求区分用途，不增加三层配置体系：

| 模式 | 必需来源和计算 |
| --- | --- |
| 默认完整 Raw 采集 | arm/hand 实测、aggregate 触觉、RGB-D；不默认要求 dense 触觉，不在线生成派生点云 |
| 无录制的 joint-only policy | 关节反馈及既有控制安全条件；不启动相机、点云、VR，不被未请求触觉缺测阻塞 |
| 无录制的 hand_pointcloud / fingertip / EEF policy | 对应实测关节及 FK；手点云本身不需要相机、分割或触觉 |
| depth / RGB / 场景点云 policy | 按所需来源启动相机与必要计算；RGB 和场景点云同时请求时保持同一 camera sequence |
| K/E-only policy | 保留真实相机来源时间，可使用现有 header-only 路径，不无谓复制 RGB-D |

录制开启时需求是“模型需要 ∪ 最小采集来源”。录制初始化失败应显式结束该次需要录制的尝试，不悄悄转成无录制运行。设备协议若必然整包返回 dense/current 等，可以继续读取，但不默认复制、持久化或用它们阻塞基本采集。

部署前，artifact 决定 observation_fields、动作类型、历史/预测长度、control dt、点云/手点云数值 recipe 和归一化；Real 决定当前设备与物理标定。缺少影响训练数值解释的必要 recipe 时清楚失败，不能用当前默认值猜训练配置；也不能强求新场景标定与历史完全相同。

**配套 `dexmani_policy` 的边界：** 如果执行工作区有该仓库，核对并完成 `deployment/metadata.py` / `runtime.py` 及直接 recipe 传递处的最小修改，让 hand_pointcloud/depth 的形状、dtype、warmup 和预处理与实际输入一致。默认手点云保留 XYZRGB，不擅自套用只针对场景点云的 pc_dim 切片。depth 的 scale/消费单位须一致。只修改已有接口，禁止模型/encoder 研发、训练流程重构或新增通用 schema 注册层。独立列出该仓库 diff 和验证，不自动提交/推送远端。

回归范围是仍在 12 字段契约中的现有输入与控制模式。不为依赖退出字段的历史 artifact 建设兼容层；若新默认路径不再支持 dense 等输入，在设备连接前明确拒绝，不把缺失值补成假数据。

若配套仓库不可访问或当前模型确实不接受新模态，完成 Real 的生产/导出/观测链和可独立验证部分，报告准确消费方文件与缺口。不得用修改白名单、monkeypatch、伪造 warmup 或 FakePolicy 通过冒充真实模型端到端可用。无需为本任务训练新模型；已有模型没有使用新模态不代表 Real 生产端可以省略它们。

## 7. 运行结构与配置收敛

### 7.1 保留有效的所有权，删除重复协调

| 责任 | 唯一权威所有者 | 可删除的重复内容 |
| --- | --- | --- |
| 运动授权/撤权、run epoch | 现有 RuntimeState/授权边界 | session/recorder 中等价镜像授权状态 |
| xArm SDK 执行 | 现有独立 arm worker | 主线程重复设备状态审计，不改变 worker 协议 |
| 多设备 dispatch 与 stop | DexManiRobot 和既有停止接口 | 调用方重复解释同一回执 |
| 模型对象与 Future | InferenceWorker | 多层 pending/result 缓存与重复提交 |
| 单轮计划/交接/结束原因 | PolicyEpisode 与纯 schedule 函数 | 为录制 query 证据服务的额外字典与控制分支 |
| 资源、操作员与多轮运行 | PolicySession / 对应 teleop session | operator/result 对象中重复业务状态机 |
| 写盘、staging 和发布 | Recorder | 设备执行审计与模型生命周期知识 |
| 实验结果摘要 | 现有 results 输出 | 把统计结果反过来当运行授权条件 |

这不是要求重命名或合并这些类。`capture_attempted` 与 `writer_ready`、停止请求与物理停止确认、运行状态与结果快照是不同事实，不能因为“看起来重复”就合并。

Teleop、deployment、replay 各保留直接可读的循环：准备→读观测→计算/选择动作→映射/检查/发送→记录→收尾。只共享实际重复的数值函数、设备原语和短 helper；不新增 BaseRunner、万能 Context、DI 容器、事件总线或全仓 finish_everything。

保留现有 sync/async/RTC、VR 锚点、EMA、TAG/DexPilot、HOME 行为。重点核对：单个未回收 Future、迟到结果按旧 run/query 回收、正常 slot 交接、RTC 冻结前缀不被重投影改变、过期拒绝、漏槽不追赶补发、等待不重复末动作。不能把 `shutdown(wait=False)` 写成 CUDA 已停止。

不以简化或提升成功率为由新增平滑、插值补偿、重复目标 fallback、自动恢复或自动重连；这些会改变实验系统，不能混入本次结构调整。

保留 xArm 进程隔离、必要 sensor worker 和有界 writer；不为统一风格增加或删除并发边界。控制线程不等待大段 FK/点云处理、磁盘编码或模型结束。停止监听和授权撤销不依赖慢模型/保存流程完成。

### 7.2 配置与数学模块

- 保留 dataclass 默认值→一份 YAML→CLI 覆盖。未知字段、错误布尔类型等明确拒绝；不引入 Hydra、动态配置继承、跨文件模板或 profile registry。
- `ControlParams` 按实际使用方收敛：EMA/VR/重定向属于 teleop；hand enabled 属于 hand；workspace/桌面属于 environment；共用 IK/动作约束由小参数对象或显式参数传入；路径/本次时长/录制开关留在入口参数或局部运行选项。
- Session 作为装配层可接收整份配置；TeleopController、TargetChecker、ActionMapper 和几何函数只接收实际所需参数、solver/checker。不要引入新全局上下文替代 ExperimentConfig。
- 有 warm start、锚点、滤波或优化器状态的算法保留对象与明确 reset，不强行改成纯函数。配置模块不能通过导入链初始化 SDK、相机、模型、优化器或 mesh。
- 静态配置在准备阶段验证一次；外部数值在接收边界验证，float32 转换溢出另查。计算后/跨线程进程后/实际发送前的时间和授权重检仍必要，不视为可删的重复检查。
- 保留 teleop rate、policy control dt、camera rate、arm servo rate 的物理区别；派生 fps 不再维护第二个可独立改写的来源。
- 提供少量最短示例配置覆盖 teleop、sync、async（RTC 如需单独参数则加小示例），只写覆盖值，不复制全套默认配置。`--print-config` 保持离线无设备；可展示已解析值及 artifact 所有权，但不引入运行配置数据库。

### 7.3 其他流程的有界简化

- HOME 的路线规划/协调属于业务流程，SDK 原语属于设备。仅在当前重复职责明确时移动，不重新设计路径或限位。
- calibration 与长 session 先让正常路径清楚、清理唯一；不按任意行数拆文件，不继承 teleop runner。
- replay 物理采集只写一次，派生 FK 在评估阶段计算；删掉确无消费者的重复 quat/rot6d、初始 NaN 派生数组与整文件重写。明确轨迹重放误差是跨运行一致性指标，不等于任务成功。
- 评估保持简短 episode 摘要：时长、结束原因、错误、Raw 路径、必要推理耗时/dispatch/漏槽统计。不要每个 helper 都传可选 results 再重复判空；也不要建设自动任务成功检测或全量查询审计系统。
- 保留 `collect_teleop.py`、`run_policy.py`、`replay_episode.py`、`export_zarr.py` 和标定/调试入口，不建设新的统一实验 CLI 或调度平台。

## 8. 按依赖顺序实施

每阶段达到 Exit 后继续下一阶段；分小批提交/验证不等于等待用户逐批批准。所有阶段均在本次代码任务范围内，真实阻塞单独报告。不要为了满足阶段清单撤销已完成的现行实现。

| 阶段 | 主要修改位置 | 完成条件 |
| --- | --- | --- |
| 0. 基线核对 | `AGENTS.md`、README、旧任务书、git status、下列实际调用链 | 记录 HEAD/已有修改；识别新旧规则冲突和已完成事项；给出短文件级计划后开始改代码 |
| 1. 最小 Raw 与录制 | `data/raw/*`、`data/recorder.py`、调用 recorder 的 teleop/deployment/replay | 去掉辅助字段与强制执行对账；保留真实 dispatch、来源时间、失败前缀、有界队列和原子发布；新旧消费者同步 |
| 2. 12 字段与几何 | `data/canonical/*`、现有 kinematics/pointcloud、轻量手点云 helper | 合成 Raw→完整 12 字段 Zarr；排除字段不存在；action/FK/坐标/深度/模板语义正确；完整性不被辅助字段阻塞 |
| 3. 在线消费 | `runtime/observation.py`、`deployment/preflight.py` / `observation.py` / `session.py`、必要 policy 配套接口 | 新模态需求、时钟、历史、warmup/recipe 真正连通；关闭录制时只启动需要的资源；旧输入模式不回归 |
| 4. 流程与配置 | `config/*`、`planning/action.py` / `target.py`、teleop/controller/session、deployment/episode、replay/evaluation | 删除已无消费者的 query/evidence、重复状态/校验/包装；数学模块依赖缩小；默认控制与算法数值不被暗改 |
| 5. 文档与最终检查 | README、AGENTS、旧任务书状态说明、必要稳定离线检查 | 文档统一到 12 字段；旧任务书标明其数据条款由本文替代；无新旧双实现、失效 alias 或多余报告；全部适用检查完成 |

不要把纯机械移动与科学数值变化混成无法审查的大提交。旧缓存无需迁移；不可覆盖已发布 Raw。本文保留为用户指定任务说明，不能通过改写验收条件把未完成项变成完成。

## 9. 必须覆盖的离线验证

使用现有环境、现有测试组织方式和小型 Fake/合成数据。仓库不使用 pytest/Ruff，不安装或运行它们；可以直接执行已有纯 Python 检查或使用标准库 unittest。只增加验证真实故障/边界且值得维护的检查，不写镜像实现的测试。

| 验证 | 必须观察到的结果 |
| --- | --- |
| 字段与导出 | 恰好 12 个业务字段，形状/dtype/episode 长度一致；三项排除字段与四项辅助字段均不在公共缓存；不存在对象/分割依赖 |
| 观测动作对齐 | 不同 measured/raw request/submitted targets 的样例能区分三者；obs[t] 对 action[t]，action_ee 来自 submitted arm FK |
| 几何 | 非单位旋转验证 rot6d 顺序、K/E 方向和 m/rad；手安装、五指顺序、link/joint 对应正确；手点随实测关节变而非随 action 变 |
| 深度/点云 | 非 0.001 的 depth scale、空深度、非空少点、invalid extrinsic；不得零填整帧假几何；K/E 缺失不妨碍 Raw 留存但阻止所需派生 |
| 手模板 | 相同输入/seed 结果稳定；初始化采样仅一次；颜色固定、无 camera/segmentation 依赖；无资产时明确失败 |
| 时间/触觉 | 保持值保留原有效时刻；超过上限被拒绝；无请求触觉不阻塞 joint-only；真实间隔/漏槽不被 nominal fps 或裁切隐藏 |
| I/O 与失败 | 队列满、编码失败、RGB/数组行数不符、close 异常、零行、部分 dispatch、停止未确认；只按实际 I/O 发布，失败状态不伪装训练合格 |
| 异步/RTC | 提前完成、迟到、过期、旧 run 结果、WAIT、冻结前缀、部分 arm/hand 发送均保留现有控制语义；无补发/重复末动作 |
| 按需依赖 | no-record joint-only、hand-only geometry、depth-only、K/E-only、RGB+cloud、record-on 各启动正确资源；dense 缺测不阻塞默认 Raw；配置解析不加载重依赖 |
| 在线/离线 | 同一合成来源调用共享函数，几何及字段相同；实际视频解码颜色按声明容差检查，不要求伪造逐位相同 |
| 效率与所有权 | mesh/模板仅初始化一次；导出内存由 block 大小限制、录制内存由有界队列限制；无逐帧重复初始化/多份 RGB-D 无用复制；无需新性能监控框架 |
| 配套消费方 | 若改 policy，验证其真实 inspect/metadata/warmup 入口；没有真实兼容模型时明确止于接口检查，不声称训练/推理端到端通过 |
| 配置/范围 | CLI > YAML > 默认，错误字段/类型拒绝；现有模式参数保持；git diff 中无无关改动、历史 Raw 修改或依赖升级 |

基础检查：

```bash
python -m compileall -q dexmani_real scripts
git diff --check
git diff --stat
git status --short
```

按阶段运行对应检查，最终运行适用检查集；已通过的检查没有新修改/新疑点时不要反复扩展。缺少 Pinocchio、视频或其他可选依赖时如实区分“静态通过”“Fake/合成通过”“实际依赖未跑”，不擅自升级真实实验环境。

本任务只授权代码修改与离线验证，不执行真实设备连接、运动、HOME、rollout、实时相机/VR 采集、标定写入或硬件诊断。保持现有运动授权、worker 隔离、停止路径、限位/碰撞与过期拒绝；离线结果不作为真机安全证明。

## 10. 最终交付与完成标准

执行者交付简洁中文说明，包含：

1. 实际完成的阶段和文件；12 字段从物理来源→Raw→派生→policy 的覆盖状态。
2. 真正删除的复杂机制、缩小的参数依赖，以及保留它们的必要边界的原因；不只汇报“新增了什么”。
3. 新数据契约与旧格式差异；旧 Raw 的处理方式；触觉单位、depth scale、手点云模型来源等限制。
4. 已执行的具体离线命令/用例及结果；未完成、未验证、缺依赖和需人工真机验证项分别列清楚。
5. 如涉及 `dexmani_policy`，独立列出其最小配套 diff 与消费方验证，不把 Real 的 Fake 检查当作该仓库验证。

允许按逻辑批次创建仅包含本任务改动的本地 commit；保留用户已有修改。不自动 push，不强推，不删除分支，不重置用户工作树。完成工作后给出清楚的待审查结果，不额外生成一套根目录报告。

**完成判据：** 默认采集只记录生成目标模态所需的真实来源；Canonical 完整提供 12 字段；在线按模型需求复用同一计算；排除的模态无隐式入口；控制安全与物理时间语义保留；Recorder、配置和实验流程比现状更直接。必要配套消费方若受环境阻塞，必须明确标为未完成，不能宣称整条链路已打通。
