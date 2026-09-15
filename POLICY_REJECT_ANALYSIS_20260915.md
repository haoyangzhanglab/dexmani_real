# Policy 运行异常与 reject 策略分析（2026-09-15）

## 1. 文档范围与结论

本文件记录一次具体运行的现象、日志证据、离线分析过程和待验证问题，
不作为长期运行配置或安全策略规范。按用户要求，放置于 README 同级目录。

**已确认的故障链：手部目标越过机械下限 → 耦合命令被拒绝 → 后续目标继续
随策略时间推进 → 相对最后成功发布目标的手臂差值超过阈值 → 连续拒绝 →
命令静默超时结束回合。**

当前证据支持“模型在训练动作高度集中的机械边界附近产生预测误差”这一解释。
模型为何产生该误差，以及暂停后的新预测为何持续领先实际位置，尚未完成因果验证。

分析仅使用源码、已有日志、保存的轨迹、训练数据和部署 checkpoint。
没有连接硬件，没有执行物理回放，没有运行模型推理；checkpoint 通过
`torch.load(..., map_location="cpu", weights_only=True)` 读取。

## 2. 运行信息与证据来源

### 2.1 启动命令

```bash
python examples/run_policy.py experiments/maniflow/pick_place_toy/2026-09-15_01-28_0 --num-episodes 2
```

| 项目 | 本次运行值 |
|---|---|
| 策略 | ManiFlow |
| 实验 | `maniflow/pick_place_toy/2026-09-15_01-28_0` |
| Artifact | `epoch=0898-step=00080000-milestone=100pct-deployment.pt` |
| 权重来源 | checkpoint contract 标记为 `ema_model` |
| 推理步数 / seed | 4 / 0 |
| 设备 | `cuda:0`（原运行日志所示，本次分析未执行 CUDA） |
| 观测 | `joint_state` + `point_cloud` |
| 动作 | 19 维关节目标，7 维手臂 + 12 维手部，单位 rad |
| 控制 / 重规划 | 16 Hz / 每 8 步，即标称 500 ms |
| 最大回合时长 | 60 s |
| 实际结果 | 第一回合约 12.515 s 自动停止，第二回合未开始，随后主动退出 |

### 2.2 原始证据

- 终端日志：用户在本次对话中提供的完整日志。本文保留关键摘录；未额外保存完整终端日志文件。
- [运行参数](rollouts/maniflow/pick_place_toy/2026-09-15_01-28_0/session_20260915_164754/run_config.yaml)。
- [回合数据 HDF5](rollouts/maniflow/pick_place_toy/2026-09-15_01-28_0/session_20260915_164754/episode_001/data.h5)：实测状态、动作记录、时间戳、帧标志和结束原因。
- [策略轨迹 NPZ](rollouts/maniflow/pick_place_toy/2026-09-15_01-28_0/session_20260915_164754/episode_001.policy_trace.npz)：预测 chunk、预测序号、决策索引和决策时间。
- [训练配置](experiments/maniflow/pick_place_toy/2026-09-15_01-28_0/config.yaml)。
- [部署 checkpoint](experiments/maniflow/pick_place_toy/2026-09-15_01-28_0/checkpoints/epoch=0898-step=00080000-milestone=100pct-deployment.pt)：数据契约与 normalizer 参数。
- 当前训练数据：相邻仓库中的 `../dexmani_policy/robot_data/pick_place_toy.zarr`。

数据来源限制：当前 Zarr 包含 60 个 episode，训练配置设置
`max_train_episodes: 50`。本文对 Zarr 的统计覆盖当前完整数据集，不能当作训练时
实际选中子集的精确统计，也未通过历史数据哈希证明它与训练时完全一致。
`run_config.yaml` 未保存完整硬件限位快照；阈值解释依据检查时源码与配置路径，
并与本次轨迹、报错和停止时间交叉核对。

## 3. 问题现象与日志依据

### 3.1 启动成功，前期持续正常发布

```text
[16:48:04] ... All subsystems ready — safety=ARMED(1)
[16:48:11] ... safety: consumed B; ARMED(1) → RUNNING(2), generation=5 epoch_ns=85285166823954
Episode 1/2 RUNNING
[16:48:12] ... inference_latency_ms=32.354898 ... publication_interval_ms=62.455804 ... safety_rejection_count=0
```

以上摘录省略了部分 logger 前缀或非关键字段。原日志中就绪横幅本身没有时间戳。
保存轨迹中的 25 次预测推理耗时约为 32.057～36.804 ms。
成功发布期间，日志中的发布间隔约 62.5 ms，与 16 Hz 一致。

### 3.2 首先出现手部机械限位拒绝，随后转为手臂跳变拒绝

```text
[16:48:22] [WARNING] [dexmani_real.deployment.executor] executor: rejected policy step: hand policy endpoint violates rated mechanical joint limits
[16:48:22] [WARNING] [dexmani_real.deployment.executor] executor: rejected policy step: arm action jump exceeds limit
[16:48:23] ... safety_rejection_count=21 ...
[16:48:24] ... safety_rejection_count=34 ...
```

完整日志中有 6 条手部限位拒绝、28 条手臂跳变拒绝，共 34 次。
这两个计数代表首先触发的检查项，不代表每个目标仅存在一种违规。

### 3.3 命令静默超时，数据保存完成，最后正常退出

```text
[16:48:24] ... safety: revoked motion RUNNING(2) → ARMED(1), generation=6
[16:48:24] [WARNING] [dexmani_real.deployment.executor] executor: policy episode ended: command silence timeout
[16:48:24] ... Episode saved: .../episode_001 frames=201
Episode 1/2 saved
...
exit_reason=explicit quit  safety=DISARMED  supervisor_normal=True  clean=True
```

HDF5 的 `meta` 属性提供独立佐证：

```text
stop_reason = command_silence_timeout
duration ≈ 12.515371631 s
num_frames = 201
truncated = False
```

`clean=True` 表示退出清理正常，不表示任务成功。`command_progress_timeout_count=0`
与命令静默超时不矛盾：前者针对已发布命令的 worker 接受进度，后者针对多久没有成功发布新命令。

## 4. 离线分析过程

### 4.1 还原实际被选择的预测目标

NPZ 中包含 25 个预测，每个预测形状为 `(15, 19)`，以及 200 个执行决策。
利用 `decision_prediction_sequence` 查找 `prediction_sequence`，再使用
`decision_chunk_index` 取出对应 `actions` 行，得到执行器实际选择的原始目标。

相对时间计算为：

```text
t = (decision_event_monotonic_ns - run_started_monotonic_ns) / 1e9
```

轨迹状态计数为 166 个成功发布决策、34 个安全拒绝决策。
200 个决策不等于 201 个录制帧，两者是不同的记录口径。

分析成功发布目标时，只在成功决策后更新比较基准，拒绝后保持基准不变。
手臂角度差计算考虑 2π 等价表示；本次触发关节为 J5，相关数值没有显示出
跨 ±π 表示产生的虚假跳变。源码使用受关节限位约束的最近等价角选择，
一般场景不能简单用所有关节取模替代该实现。

### 4.2 将手部预测与机械边界逐维比较

异常维度为手部零基索引 3，即第 4 个关节：

```text
right_hand_index_bend_joint
19D action 零基索引：10
机械范围：[-0.174, 0.174] rad
```

| 项目 | 数值 |
|---|---:|
| 首次被拒绝目标 | −0.175394058 rad |
| 首次超出下限 | 0.001394058 rad |
| 6 次手部拒绝中最小目标 | −0.176450133 rad |
| 最大超出下限 | 约 0.002450133 rad，即 0.140° |

在全部 200 个所选目标中，14 个目标存在手部机械越界，均发生于这个关节。
其中 8 个被前置的手臂检查拦截，因此没有打印手部错误。
全部 25 × 15 个预测位置中，该关节有 22 个位置低于机械下限；未选择的位置
不能计作实际执行器拒绝。

### 4.3 核对训练动作和 checkpoint 归一化参数

当前完整训练 Zarr 的结果：

| 项目 | 数值 |
|---|---:|
| 动作数量 | 14,112 |
| 该关节最小训练目标，转为 float64 后 | −0.17399999499320984 rad |
| 距 −0.174 rad 不超过 1e-6 的动作 | 2,718，约 19.26% |
| 该关节低于 −0.174 rad 的训练动作 | 0 |

Checkpoint 的 `normalizer.params_dict.action` 在该维度保存：

```text
scale  = 5.747126579284668
offset = 0
```

按仿射变换 `normalized = action * scale + offset` 计算，机械下限对应约 −1，
最严重的越界预测对应约 −1.01408。没有发现这个维度明显的归一化尺度或单位错误。

这支持“训练目标贴边，模型预测略超出边界”的解释。模型必须预测大量几乎没有
向外误差余量的目标，少量预测偏差即可触发严格机械限位检查。
但仅凭这些统计，不能断言采样步数、网络结构、场景偏差或训练方式中的哪一项造成了误差。

Real 的操作边界舍入修正容差为 `1e-6 rad`，且机械包络检查先于该修正。
本次越界明显大于这一数量级，不能通过扩大“浮点舍入容差”来解释或修复。

### 4.4 核对手臂跳变是否由拒绝累积

| 按 B 后时间 | 决策结果 | J5 目标 | 相对最后成功发布目标的 J5 差值绝对值 |
|---|---|---:|---:|
| 10.501 s | 最后一次成功发布 | −164.43° | — |
| 10.563 s | 手部越界拒绝 | −168.97° | 4.54° |
| 10.626 s | 手部越界拒绝 | −173.84° | 9.41° |
| 10.688 s | 手部越界拒绝 | −178.06° | 13.63° |
| 10.751 s | 手部越界拒绝 | −181.52° | 17.09° |
| 10.813 s | 手部合法，但手臂跳变拒绝 | −185.69° | 21.27° |

全部所选预测目标之间，相邻手臂目标的最大关节差值约 16.23°，没有超过 20°。
当前训练 Zarr 排除 episode 边界后，也没有相邻手臂动作超过 20°。

因此，本次手臂报错不是相邻预测直接跳了超过 20°，而是中间动作未发布后，
后续目标与最后成功发布目标之间的差距累积超过阈值。

### 4.5 核对后续新预测与实测位置

最后成功发布后仍有新预测产生，并非一直重复消费同一旧 chunk。
后三轮新预测中，实际选择的首个目标相对最后成功发布的 J5 分别偏离约：

```text
30.92°、29.05°、30.41°
```

即使检查这些 chunk 中更早的第 1 个未来目标，差值仍约为：

```text
26.83°、25.54°、25.99°
```

HDF5 实测显示，J5 后来停在 −164.43°附近，接近最后成功发布目标。
实测值对齐使用决策时刻之前最近一条录制观测，仅用于趋势与交叉核对，
不是对执行器当时所读反馈的逐纳秒复现。

因此，单纯改用当前实测位置作为跳变基准，或等待新的 chunk，都不能直接解决本次问题。

### 4.6 检查训练目标领先实测位置的程度

Artifact 数据契约的动作语义为 `teleop_published_joint_target`，观测为实测关节位置。
当前完整 Zarr 中，J5 动作目标与实测位置之差的绝对值为：

| 分位点 | 差值 |
|---|---:|
| 中位数 | 1.99° |
| P95 | 11.87° |
| P99 | 20.57° |
| 最大值 | 33.85° |

目标领先反馈在运动过程中并不自动意味着数据错误。但这支持一个待验证假设：
模型学到了运动中的目标领先量；机器人因拒绝停住后，模型仍给出领先较大的目标，
无法通过重新接入时的跳变检查。

该假设不能单靠相关统计证明。还需固定观测、改变运动历史或比较不同模型输出，
才能区分运动历史、场景条件和模型误差的影响。

## 5. reject 策略的源码链路与评价

### 5.1 检查与发布路径

```text
公共 Policy runtime 返回未来动作 chunk
  → Real 推理入口检查 shape / finite
  → 预测写入 IPC
  → executor 按控制时间选择一个目标
  → 解码 7D arm + 12D hand
  → 手臂限位、最近等价角与 20° 跳变检查
  → prepare_command 检查反馈与手部机械包络
  → SafetyGate 的手部操作限位、手部差值和工作空间等检查
  → 发布整条耦合命令
  → 成功后更新最后发布目标与时间
```

上图不表示启用了所有可选 SafetyGate 检查。Policy gate 构造时提供了工作空间检查，
不能仅凭启动时 `CollisionModel ready` 日志推断策略执行路径启用了逐步碰撞检查。

关键源码入口：

- [推理输出检查](dexmani_real/deployment/inference/worker.py)：`_predict_action_chunk`。
- [动作选择、拒绝与 watchdog](dexmani_real/deployment/executor.py)：
  `_validate_policy_arm_action`、`_decode_due_action`、`_reject_due_step`、
  `_advance_prediction`、`_publish_due_action`、`_command_watchdog_reason`。
- [命令准备边界](dexmani_real/control/publication.py)：`prepare_command`。
- [手部机械边界与操作边界舍入](dexmani_real/utils/limits.py)：
  `canonicalize_policy_hand_endpoint_roundoff`。
- [共享安全检查](dexmani_real/control/safety_gate.py)：`SafetyGate.validate`。
- [时间网格](dexmani_real/deployment/timing.py)：`first_future_step_index`。
- [数值配置](dexmani_real/config/defaults.py)：`PolicyParams`、`HandParams`。

### 5.2 当前拒绝后的行为

单步拒绝会：

1. 不发布当前耦合命令。
2. 消耗当前控制时隙并推进预测索引。
3. 保持最后成功发布的手臂、手部目标不变。
4. 不刷新最后有效命令时间。
5. 继续尝试后续目标，直到恢复、其他停止条件成立或 watchdog 超时。

推理端仍按标称 500 ms 周期工作，拒绝没有触发即时重规划通知。
单次拒绝也不是立即急停：之前已接受的目标仍可能被完成，本次实测亦显示这种过程。

### 5.3 有效之处

- 越界提案没有进入硬件命令发布路径。
- 手臂与手部保持耦合，未执行错误目标中的部分动作。
- 被拒绝的动作不刷新命令 watchdog。
- 使用最后成功发布目标作为比较基准，防止跳过动作后直接下发大跳变。
- 最初两次孤立手部拒绝后曾恢复发布，说明跳过单个坏点有时有效。

### 5.4 本次暴露的局限

当前策略可以跳过单个坏点，但没有明确的“轨迹已无法衔接”恢复阶段。
连续拒绝后，原 chunk 所隐含的运动进展没有发生，继续按时间消费目标会扩大
目标与实际执行状态的偏差。新预测也可能无法重新接入，最终只由通用静默超时结束。

不可将比较基准更新为被拒绝的目标：机器人没有执行它，这会掩盖真实命令间的大跳变。
也不能通过单纯提高 20° 阈值、放宽机械包络或延长 2 秒超时消除报错。

## 6. 其他日志的解释

| 日志或现象 | 本次结论 |
|---|---|
| `Control loop over budget ... lateness=0.6ms missed_total=0` | 单次轻微调度超时，无证据说明其直接造成机械边界外预测 |
| 启动阶段 `camera_loop: device frame gap=5` | 原日志说明保留当前帧；无证据证明其为本次停止的直接原因 |
| `skipped_prefix_steps=1` | 表示接收预测时首个严格未来索引，不是整回合只跳过一步，也不保证实际决策始终从该索引开始 |
| 后期 `publication_interval_ms=62.64888` 不变 | 最后一次成功发布时记录的旧值，不代表拒绝期间仍正常发布 |
| 回合结束后指标继续刷屏 | 原实现无条件周期打印最新快照，已在本次会话的日志精简改动中处理 |
| `clean=True` | 会话清理正常，不能用于判断任务成功 |

## 7. 尚未确定的问题与后续验证建议

### 7.1 尚未完成的验证

- 未重建并重放模型实际消费的全部在线观测。
- 未运行不同推理步数或随机种子的对照推理，不能认定 4 步采样就是原因。
- 未证明当前完整 Zarr 与训练时实际子集完全一致。
- 未验证场景、视觉输入和运动历史各自对越界、目标领先的贡献。
- 未进行修改 reject 策略后的闭环或硬件验证。

### 7.2 两个独立改进方向

**方向一：减少非法提案。** 离线统计各关节边界越界率，比较不同推理条件，
重点检查训练目标贴边时的误差分布。评估策略输出端的明确合法范围约束或训练边界余量。
此类改动属于策略行为变化，需要单独验证，不应伪装成浮点舍入修正。

**方向二：明确连续拒绝后的处置。** 可以评估拒绝后丢弃当前 chunk 余量、等待
基于新观测的预测；但本次证据说明这并不保证恢复。更保守的方案是在确认连续不可恢复
拒绝后结束回合，并记录具体原因。若需要重新接入轨迹，则需要显式、经过验证的恢复过程。

离线对同一组预测做假设性筛选只能解释检查逻辑，不能证明改变行为后机器人会安全恢复，
因为新的动作会改变后续观测和模型预测。

## 8. 已实施的日志精简与本报告状态

在本次深度分析之前，已修改部署日志：

- 运行时周期指标改为 DEBUG，空闲和回合结束后不再重复刷指标。
- 同一拒绝原因每回合首次 WARNING，后续逐次保留 DEBUG，结束时汇总计数。
- 降低重复部署参数、推理预热与就绪、RUNNING、保存路径提示的级别。
- 保留安全状态、停止原因和硬件警告；未更改安全检查和命令执行语义。

日志改动已通过离线输出核查、`compileall` 和 `git diff --check`；未运行硬件。
本报告记录的异常发生于日志精简之前。本文创建阶段仅新增分析文档，
模型越界与 reject 恢复问题尚未实施修复。
