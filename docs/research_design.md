# 个人研究工作流与代码简化

目标是让 xArm7 + XHand 实验容易启动、理解和修改：常用流程有明确入口，一个事实只有一个所有者，计算和数据搬运只服务当前消费者。代码行数和目录数量不能代替这些目标。

## 参考项目与取舍

以下比较固定到源码提交，属于设计借鉴，未运行参考项目的硬件。没有移植它们的实现或引入新框架。

| 项目 | 实际可借鉴的组织 | 本项目的落点 |
| --- | --- | --- |
| [LeFranX · a39906e](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/scripts/dual_robot/dual_vr_record.py#L94) | 实验脚本直接装配设备、遥操作和录制；高频伺服由专用后端承担 | 保留一个 `RealConfig` 和显式模式入口；`Robot` 直接组合 arm/hand，研究循环生成目标，执行进程服务伺服 |
| [ManiUniCon · 85c6f2e](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/policies/torch_model.py#L232) | 观测适配 → 模型 → 动作适配职责可见，入口集中装配组件 | `ObservationBuilder` 组织输入，Policy 负责推理，`ActionMapper` 转换目标；会话和录制状态留在实际拥有者 |
| [T-Rex hardware_code · ba4047c](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/teleop_targets.py#L331) | 数据源产生最新快照，目标计算频率与设备执行频率分开 | 复用传感器 Ring、模型工作线程和机械臂执行进程；暂停时只读取检查所需的信息 |

参考项目也有各自的取舍。LeFranX 的[缺 SDK 自动 stub](https://github.com/wengmister/LeFranX/blob/a39906e6629f39490950fe8bd20f4f992ed74fd7/src/lerobot/robots/xhand/xhand.py#L83)、ManiUniCon 的[过期动作重排](https://github.com/Universal-Control/ManiUniCon/blob/85c6f2e32ecf9f2bed62d202b058c39623444686/maniunicon/customize/act_wrapper/chunk_wrapper.py#L61)、T-Rex 的[无界录制队列](https://github.com/ZhuoyangLiu2005/T-Rex/blob/ba4047ce3e9f3d2013d60de600e136897e90ccb8/hardware_code/teleop/data_writer.py#L243)不满足本项目当前的失败与数据语义。借鉴组织方式时，仍需逐条核对真实行为。

## 从实验任务找到实现

| 想改什么 | 入口 → 主要实现 | 所有权边界 |
| --- | --- | --- |
| 采集示教 | `scripts/collect_teleop.py` → `teleop/run.py` → `TeleopLoop` | run 装配连接，loop 拥有当前示教段、提交参考和结束意图 |
| 执行策略 | `scripts/run_policy.py` → `PolicySession` → `Rollout` | session 拥有设备/模型生命周期，rollout 拥有单段历史、动作计划和录制；未完成收尾仍归 session |
| 模型输入 | `runtime/observation.py` → `deployment/observation.py` | 前者保留来源事实并检查时效，后者按模型字段组装历史 |
| 动作与 IK | `control/actions.py` → `control/ik.py` → `Robot.send_action` | mapper 生成目标，IK 拥有求解所需限位/工作空间，Robot 提交设备；设备限位和命令租约在提交边界检查 |
| 数据录制 | `EpisodeRecorder` → `RawWriter` | recorder 拥有线程、队列和发布结果；writer 拥有编码与文件缓冲 |
| 训练导出 | `scripts/export_zarr.py` → `RawReader` / `data/export.py` | 按选定字段消费输入，每条 Raw 完整接纳或回滚，成功后发布产物 |
| 查看与标注 | `scripts/visualize_episode.py` / `scripts/cordvip/annotate_episodes.py` | 查看按前缀读取；HOI 标注在独立 `hoi.h5` 中写入 |
| 离线重定向调试 | `scripts/preview_hand_retargeting.py` → `HandRetargeter` | 按记录参数计算，求解器持有参考；读取块不改变原生段界，结果只用于离线对照 |
| 真机回放 | `scripts/replay_episode.py` → `replay/` → `Robot` | 只回放所选原生连续段，动作仍经过公共设备提交端 |

这些模块分别拥有实际的状态或资源。无需额外的 BaseSession、组件注册表、任务总线或全局共享字典。相似的循环只有在状态、结束语义和资源所有者都一致时才合并。

## 本轮行动计划与实施范围

本轮已对齐 Real `436e7431`，配套 Policy `b1ddc36`。保留上轮仍适用的控制与导出简化，接入最新版的显式资产路径、离线预览与Raw时间校验。实施按下面顺序进行；不混入训练格式或硬件协议迁移。

| 阶段 | 行动 | 完成标准 |
| --- | --- | --- |
| 1. 固定事实 | 读三个参考项目实际实现，核对最新入口、调用方及回归基线 | 每个删减候选有调用证据；区分额外框架与真实并发/I/O职责 |
| 2. 删除与归位 | 将只承载 IK 限位/工作空间的 `MotionConstraints` 合入 `IKSolver`；VR 唯一调用的起点检查放回 VR | 删除 `control/constraints.py` 和中间对象，所有调用方改用 solver 的直接属性；保留 VR 起点与 IK 终点的不同容差 |
| 3. 减少无用工作 | HOLD 不复制 RGB-D；导出按原生段读取关节；离线预览分块读取关键点 | 暂停仍检查时效，恢复后完整取帧；导出段界与数值不变；预览参考跨读取块连续、按录制段重置 |
| 4. 修复可重复验证 | 沿用主分支的日志隔离，去掉合并后的重复mock；声明已有NLopt依赖 | 全套测试无需进程外补丁；控制依赖包含实际使用的求解器 |
| 5. 复核与交付 | 审查 diff、运行相关回归、记录合成测量及限制 | 不把静态检查、CPU测试或局部基准表述成真机性能证明 |

`MotionConstraints` 没有独立算法、生命周期或可替换策略，其参数最终只用于 IK；直接归入求解器减少一个对象和一层属性访问。VR 的起点范围使用名义工作空间及 `1e-9` 舍入容差，IK 的最终位置使用 `eef_workspace_tolerance_m`。Jog 允许从工作空间外尝试产生合法终点，这两类检查不能混为一个公共起点门槛。

HOI 遮挡时没有模型推理输入或 Raw 行需要像素。HOLD 仍保留完整的健康判断与旧推理回收，只省去图像负载；恢复后的控制步重新取完整观测，并沿用原来的历史重建和提交检查，不缓存旧图像伪造新观测。

导出对齐需要一份结果数组，但源数据无需先完整载入再切段。保留 HDF5 dataset 句柄并逐原生段读取即可；单段内插值算法和拒绝条件不变。对于单个巨大连续段，内存仍随该段大小增长，本轮没有引入新的流式插值框架。

离线预览只在当前求解步消费21点关键点，改为每次读取最多256行；同一个求解器跨块延续参考，不因I/O边界重置。独立保存观测时间列后，也不再通过NumPy视图持有整张时间表。返回的实测、候选与参考关节曲线仍按帧数保存，未将整个预览声明为常量内存。

## 如何判断是否值得删除

| 可以删/合并的证据 | 应保留的证据 |
| --- | --- |
| 参数没有调用者；状态能由唯一来源推导 | 有实际调用依赖该参数，或状态描述不同的物理/资源事实 |
| 对象只复制字段再转发给唯一消费者 | 对象拥有设备、线程、文件或独立算法 |
| 已校验的不变量在没有变化的内部路径反复检查 | 外部文件、SDK反馈、优化器输出、异步计算后或实际提交前需要重新检查 |
| 当前状态不消费的图像、数组被重复复制 | 来源时间、帧序号、原始命令与实测值支撑数据可解释性 |

本轮核查后保留了 `start_observation(after_ns=...)`：最新版在录制器启动后实际传入它，要求重新获得关节反馈，并非死参数。导出输入有限性与派生输出有限性也未机械合并；FK输入和 float32 转换后的结果是不同错误边界。

## 验证与后续顺序

在包含配套 Policy 与测试依赖的环境中，从仓库根运行：

```bash
python -m pytest -p no:cacheprovider tests -q -ra
```

定向验证包括：名义工作空间/终点容差、HOME拒绝后恢复Jog、HOI暂停与恢复完整帧、旧推理回收、失败提交记录、分段关节读取及完整导出、预览块边界与录制段界交错、拒绝行参考保持和前缀外坏数据。性能结论须写明输入规模、字段、执行范围和测量方法。

本轮验证结果（2026-11-17）：

| 验证 | 结果与范围 |
| --- | --- |
| 最新原生完整回归 | 补齐NLopt后，主分支 `774 passed / 4 skipped`；改后 `780 passed / 4 skipped`，无失败；新增6项中4项来自保留的上轮改动、2项验证预览跨块行为。四项跳过均需要CUDA |
| 离线预览专项 | 19项通过，包含实际NLopt求解小样本、分段重置、拒绝保持、坏输入与前缀限制 |
| 预览内存 | 合成HDF5，20万行、16个原生段，每模式2个独立进程交错运行；峰值RSS中位数 `315,966 → 206,686 KiB`，减少约106.7 MiB（34.6%）。使用轻量求解替身，仅测读取和结果保留 |
| 首轮HOLD测量 | 实际 `Rollout` + `SharedRing`，640×480，模型RGB及Raw RGB-D，各200个新帧样本；每新帧像素复制 `2,457,600 → 0 B`，读取中位耗时 `711.6 → 27.2 μs` |
| 首轮关节对齐测量 | 合成HDF5，100万行、32段、19维；仅 `prepare_raw`，各4个独立进程交错运行。峰值RSS中位数 `421,376 → 326,744 KiB`（约-22.5%） |

首轮HOLD与导出基准来自 `950e0f6c` 对照，本轮保留其实现并重新运行完整回归，未重复计时。HOLD测量包含点云和元数据读取，不包含设备、推理或整个控制循环；同序列快照原本可复用，节省量按新帧计。合成内存测量不能代表真实求解或完整导出吞吐，正确性由实际函数及边界测试另行验证。

下一批变更应由真实实验的重复操作或测量决定：

1. 用一段标准示教和一段策略执行，测量采集、IK/重定向、推理、图像搬运、写盘各自的耗时与控制周期抖动。
2. 若 session/loop 仍有难以修改的部分，先定位重复状态及完整调用链，再合并确实相同的操作；不因文件较长而拆出更多 manager。
3. 若训练准备成为主要开销，按选定模态测量 `export_zarr`，再决定块大小、磁盘布局或算法优化；不提前引入并行任务框架。

真实设备验证需要现场配置、设备和实验数据。CPU回归只证明其覆盖的控制逻辑、文件语义和数值行为，GPU、真机时序与任务成功率需单独测量。
