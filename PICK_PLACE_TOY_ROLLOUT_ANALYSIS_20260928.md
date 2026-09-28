# pick_place_toy 策略失败与运动抖动：分析及真机实验汇总

整理日期：2026-09-28。实验：`maniflow/pick_place_toy/2026-09-28_04-48-18_42`。

本文汇总本轮历史 rollout 排查、动作窗口核对、诊断实现、离线预测复现，以及 `n_action_steps=8/12` 两次真机实验。它是一份实验记录，结论以这里链接的配置、源码快照和数据为边界，不作为未来所有部署的固定配置建议。

## 1. 核心结论与证据强度

| 问题 | 当前判断 | 证据与边界 |
|---|---|---|
| `horizon=16, n_obs_steps=2, n_action_steps=8` 是否存在窗口错误？ | 没有发现索引错误 | 训练窗口、模型切片、现场发布索引一致；不意味着这些超参数在任务上最优 |
| 改为 `n_action_steps=12` 是否合法？ | 合法，执行 `[1:13]` | 同一输入和 seed 下完整预测与 8 步一致，仅执行切片变长；38 个现场 chunk 全部验证通过 |
| 换段时是否存在较大的目标跳变？ | 已确认 | 8 步实验段内最大关节目标变化中位数 0.885°，换段为 3.063°；新旧预测同逻辑时刻的修订中位数为 3.381° |
| 同步推理是否造成执行节拍膨胀？ | 已确认 | 段内发布约 63.56 ms，换段约 92.39 ms，推理约 28.34 ms；机械臂 SDK 换段发送间隔约 100 ms |
| 随机采样是否足以解释回拉？ | 现有离线证据不支持它是主要解释 | 每次查询固定 seed 后，预测修订仍约 3.38–3.73°；仅针对已保存观测轨迹 |
| 12 步是否解决抖动和抓取失败？ | 没有 | 换段次数减少，但单次换段间隔及约 3° 的关节目标跳变仍存在；本次也未抓起玩具 |
| 失败是否已唯一归因到某个模型或控制 bug？ | 尚未 | 已定位预测连续性和执行时序问题；训练误差、视觉条件、状态分布、反馈滞后各自贡献尚未分离 |

这里的“跳变”主要指命令目标不连续。它能解释机器人为何收到周期性突变目标，但不是对实测机械振动、加速度或 jerk 的完整测量。

## 2. 实验范围与数据清单

所有 session 位于 [rollout 实验目录][experiment]。下表帧数与停止原因读取自各 `episode_001/data.h5` 的 `meta` 属性；`timeout/quit` 是技术停止原因，不是任务成功标签。

| Session | 执行步数 | 帧数 | 停止原因 | 诊断状态及用途 |
|---|---:|---:|---|---|
| `session_20260928_161049` | 8 | 819 | quit | 早期 Raw，无本轮逐 chunk 诊断 |
| `session_20260928_161334` | 8 | 358 | quit | 早期 Raw；曾用于离线真实 checkpoint 预检输入 |
| `session_20260928_161914` | 8 | 419 | quit | 早期 Raw，无本轮逐 chunk 诊断 |
| `session_20260928_164623` | 8 | 448 | timeout | 本轮 30 秒诊断基线，56 次推理、55 次换段 |
| `session_20260928_170530` | 12 | 456 | timeout | 本轮 30 秒对照，38 次推理、37 次换段 |
| `session_20260928_180920` | 8 | 345 | quit | 整理时发现的后续录制，`diagnostics=false`；尚未纳入详细分析 |

早期 Raw 保存了实测关节状态、发布关节目标和 RGB-D 等信息，但没有本轮诊断使用的完整预测、chunk 标识和逐步主机时间链。`control_hz=16` 是标称配置，不能用“帧号除以 16”恢复真实执行时间，也不能单凭保存的动作证明它何时被 SDK 发送或机器人执行。

两次诊断实验均由操作者现场 H 回零、布置场景、B 开始；运行预算为 30 秒，结束后正常退出并释放设备。所有 Raw 保持原样，分析结果写入独立 sidecar。

## 3. 固定配置与数据语义

| 项目 | 两次诊断实验配置 |
|---|---|
| checkpoint | `epoch=0566-step=00060000-milestone=100pct.pt` |
| 权重 / seed / 推理步数 | EMA / 0 / 4 |
| 模型 | ManiFlow，`horizon=16`、`n_obs_steps=2` |
| 动作 | `action_mode=joint`，19 维：7 个机械臂关节 + 12 个手部关节，单位 rad |
| 输入 | `joint_state` 与 `point_cloud`；点云每帧 `1024 × 6` |
| 策略标称间隔 | `control_dt_s=0.0625`，即 16 Hz |
| arm / hand worker | 各 30 Hz |
| 实验变量 | `n_action_steps`：8 → 12 |
| 训练窗口配置 | `horizon=16, obs_horizon=2, pad_before=1, pad_after=7` |

以各 session 的 `run_config.yaml` 为准，分别见 [8 步配置][config8] 和 [12 步配置][config12]。训练快照见 [experiment/config.yaml][training-config]。

数据链路为：

```text
Raw 实测 arm_qpos + hand_qpos → processed joint_state → 模型 observation
Raw action_arm_joint_target + action_hand_joint_target → processed action → 训练目标
模型完整预测 → control_action 切片 → realizer → 发布关节目标
发布目标 → worker 校验权限并调用 SDK → 机器人运动 → 实测反馈
```

`action` 是目标关节位置，不是下一帧实测状态，也不是关节增量。判断模型误差、执行误差或相位滞后时必须区分原始预测、发布目标和反馈。当前导出实现见 [processing.py](dexmani_real/dataset/processing.py)。当前源码说明处理逻辑；若继续审计历史训练缓存，还需核对该缓存的实际导出版本。

## 4. horizon、观察历史与动作索引

### 4.1 训练和部署的对齐

令 `t` 表示当前观察在训练序列中的逻辑索引。两帧观察对应 `obs[t-1], obs[t]`，16 步动作预测对应 `action[t-1] ... action[t+14]`。因此当前可执行动作从完整预测的索引 1 开始。

```python
start = n_obs_steps - 1  # 1
end = start + n_action_steps  # Python 切片的 exclusive end
control_action = pred_action[start:end]
```

| 执行长度 | Python 切片 | 实际执行索引（闭区间） | 剩余预测 tail | 与下一段可比较的逻辑重叠 |
|---|---|---|---|---:|
| 8 | `[1:9]` | 1…8 | `[9:16]` | 7 步 |
| 12 | `[1:13]` | 1…12 | `[13:16]` | 3 步 |

约束为 `n_obs_steps - 1 + n_action_steps <= horizon`，当前窗口最多容纳 15 个执行动作。但 15 步会用尽剩余预测，现有重叠分析将没有旧段 tail 可比较；“窗口合法”不等于“推荐执行这么长”。

不应把起点改为 0 或 2 来掩盖延迟：0 对应上一逻辑时刻，2 会跳过当前动作。`horizon` 和 `n_obs_steps` 还涉及模型训练及条件编码，不能当作普通部署调参项随意修改。

### 4.2 已完成的验证

- 核对 Dataset 取前两帧 observation、模型 `start=n_obs_steps-1` 的切片，以及 runner 逐步发布路径。
- 8 步实验的执行切片为 `[1:9]`；12 步实验的 38 个完整预测均满足 `control_action == pred_action[1:13]`，456 个发布索引逐段按 1…12 排列。
- 用真实 checkpoint 对相同输入重置相同 seed，8/12 步的完整 `pred_action` 逐元素一致；12 步覆盖只改变本次执行长度。
- 新增 `--n-action-steps`，在内存中更新部署配置，并在 `run_config.yaml` 记录实际长度与覆盖参数；没有改写训练配置和权重。
- 离线 seed probe 按 session 记录的执行长度恢复模型，避免分析 12 步数据时误用原训练配置中的 8。

启动时观察历史采用当前帧填充；它不代表启动前实际采集了两帧历史。8 步实验后续历史读取间隔中位数约 63.56 ms，没有发现跨 chunk 历史索引错位。推理完成后的新反馈用于动作实现和记录，不被伪装成更早的模型输入。

参考：[模型切片实现][policy-base]、[Dataset 实现][policy-dataset]、[部署配置](dexmani_real/deployment/config.py)、[runner](dexmani_real/deployment/runner.py)。

## 5. 如何检查新旧 chunk 的连续性

设旧预测为 `old`，新预测为 `new`，`s=n_obs_steps-1`，执行长度为 `N`。必须同时检查三种变化：

```text
旧段最后执行目标 = old[s+N-1]
旧段计划下一步   = old[s+N]
新段第一步       = new[s]

实际边界跳变 = new[s]   - old[s+N-1]
旧计划步进   = old[s+N] - old[s+N-1]
重新预测修订 = new[s]   - old[s+N]

重叠误差 = new[s:s+L] - old[s+N:s+N+L]
L = min(len(old)-s-N, len(new)-s)
```

这样才能区分“旧计划本来就要迈出较大一步”和“重新预测改变了计划”。8 步比较 `old[9:16]` 与 `new[1:8]`；12 步比较 `old[13:16]` 与 `new[1:4]`。

机械臂周期关节索引 0、2、4、6 的角度差按最短角差处理；受限机械臂关节和手部关节不统一 wrap。本文“最大关节变化中位数”指每个事件先取 7 个机械臂关节角差绝对值的最大值，再对事件统计中位数。方向反转比例使用边界变化与旧计划步进向量的余弦小于 0 判定。

分析只匹配同一个 `run_id` 的连续 chunk，且要求旧段完整执行、新段从 step 0 开始。不能跨 episode、停止或截断边界强行拼接。

**这些是逻辑控制步的重叠，不是严格墙钟时间重叠。** 8 步标称查询周期为 0.5 秒，现场相邻查询观察时间比该值多约 36.7 ms；12 步标称周期为 0.75 秒。后续异步执行设计必须给预测增加明确时间锚点，而不能只依赖数组下标。

实现见 [diagnostic_analysis.py](dexmani_real/deployment/diagnostic_analysis.py)。

## 6. 如何检查实际执行时间

### 6.1 已实现的证据链

诊断由 `--diagnostics` 启用，在 `diagnostics/` 独立保存：

| 文件 / 记录 | 主要内容 | 用途 |
|---|---|---|
| `policy.h5 / queries` | `run_id/chunk_id`、完整预测、执行切片、真实模型输入、观察及传感器时间、推理起止 | 判断模型实际看到了什么、预测了什么 |
| `policy.h5 / publications` | `sequence`、chunk 内索引、原始动作、发布目标、执行时反馈、发布起止及是否接受 | 定位预测到发布之间的变化 |
| `arm.h5 / hand.h5` | SDK 起止、返回或拒发原因、命令目标、反馈 | 精确关联发送、停止拒发与反馈 |
| `*.status.json` 与 HDF5 属性 | `complete/error/dropped/max_enqueue_ns` | 先判断诊断是否完整、是否受写入问题影响 |
| patch 与源码快照 | 两个仓库的修改及新增诊断源码 | 追溯实验时实现，不能只依赖 Git HEAD |

发布与发送按 `(run_id, sequence)` 关联，不能以“最近时间”猜测。明确区分：发布拒绝、发布后明确拒发、没有找到发送证据、SDK 已调用及其返回状态。

### 6.2 时间量及解释

全部诊断时间使用主机单调时钟，不与视频帧号或 wall-clock 字符串直接混算。

| 指标 | 计算与含义 |
|---|---|
| 推理耗时 | `infer_end_ns - infer_start_ns` |
| 发布间隔 | 连续 `publish_end_ns` 的差；按段内/段间分开 |
| 发布到 SDK 延迟 | `sdk_start_ns - publish_end_ns` 到 `sdk_start_ns - publish_start_ns` 形成上下界 |
| SDK 调用耗时 | `sdk_end_ns - sdk_start_ns` |
| SDK 发送间隔 | 同一 worker 连续有效发送开始时间的差 |
| 输入年龄 | 推理或 SDK 发送时间减去对应模型输入的传感器时间 |
| 实际关节跟随 | 对齐发送目标与 feedback 的主机时间轨迹，另行检查滞后和误差 |

SDK 调用返回只证明主机调用结果，不代表固件完成运动。30 Hz worker 的轮询也会在发布与 SDK 调用之间引入相位等待。没有固件时序证据时，不能把以上时间称为“机器人完成这个动作的时间”。

### 6.3 当前调度为何产生周期性空档

runner 每次发布后设置 `next_step_ns = stamp + 62.5 ms`。队列耗尽后，等下一步到期才同步推理，再发布新段第一个动作：

```text
段内： 上一次发布 → 等待约 62.5 ms → 下一次发布
换段： 旧段末次发布 → 等待约 62.5 ms → 推理约 28 ms → 新段首次发布
```

加上循环和处理开销，现场约为 63.6 ms / 92.4 ms。这个控制路径和测量结果一致；它不由 action 起点取错造成。诊断纯逻辑测试也验证了同步推理耗时会加在换段发布间隔上。

## 7. 8 步基线实验：预测回拉与执行时序

详细原报告：[diagnosis.md][diagnosis8]；原始指标：[metrics.json][metrics8]。

### 7.1 定量结果

| 指标 | 结果 |
|---|---:|
| 推理次数 / 发布数 / 换段数 | 56 / 448 / 55 |
| 推理耗时中位数 / P95 | 28.34 / 30.94 ms |
| 段内 / 换段发布间隔中位数 | 63.56 / 92.39 ms |
| 有效发布频率 | 14.92 Hz |
| 段内 / 换段最大关节目标变化中位数 | 0.885° / 3.063° |
| 旧计划下一步变化 / 重新预测修订中位数 | 0.846° / 3.381° |
| 换段方向与旧计划相反的比例 | 69.1% |
| 段内 / 换段末端目标位移中位数 | 6.30 / 20.75 mm |
| arm 发布到 SDK 延迟中位数（下界） | 16.67 ms |
| arm / hand SDK 调用耗时中位数 | 0.280 / 10.752 ms |
| arm 段内 / 换段 SDK 发送间隔中位数 | 66.66 / 100.00 ms |

末端位移由发布关节目标做 FK 得到，并非实测末端轨迹。机械臂没有目标限幅、工作空间裁剪或在线 IK 失败。手部存在轻微限幅：38 次，最大 0.00769 rad，不能因此断言整个抓取链路无误差。

### 7.2 一个可离线复现的回拉事件

Run 5，chunk 26 → 27，sequence 217，距首次发布约 14.494 秒，机械臂 J5：

| 数据 | 角度 |
|---|---:|
| 旧段最后执行目标 `old[8]` | −200.352° |
| 旧计划下一步 `old[9]` | −198.596° |
| 新段第一步 `new[1]` | −207.350° |
| 新查询输入的实测状态 | −204.370° |

旧计划准备增加 1.756°，换段目标却减少 6.999°，重新预测修订为 −8.754°。这说明目标跳变确实来自重规划，而不是单纯沿旧计划走了一个较大步进。新预测与当前实测状态的关系提示应进一步研究跟随滞后和条件分布，但这一事件不能单独证明模型回拉的唯一原因。

### 7.3 随机性与复现

按原 seed 和保存的全部观测顺序重新推理，56 个完整预测数组逐元素完全一致，最大误差 0 rad。固定每次查询的 seed 所做的多种子试验则回答另一个问题：给定相同输入时，随机采样能造成多大变化。

| 离线 probe | 结果 |
|---|---|
| seeds | 0、1、2、3 |
| 同输入首动作最大关节标准差中位数 | 0.314° |
| 同输入首动作最大关节范围中位数 | 0.803° |
| 固定 seed 后，跨观测新旧计划修订中位数 | 3.712°、3.733°、3.376°、3.575° |

随机性存在，但不足以解释主要的度数级回拉。这里没有在真机上执行不同 seed 的反事实闭环轨迹。详细结果见 [seed_probe.json][seed8]。

### 7.4 其他边界条件

- 推理开始时，机械臂输入年龄中位数约 17.23 ms，点云约 37.07 ms。执行 chunk 后半段时模型输入年龄增加，首先是开环执行的结果，不能直接解释为传感器停更。
- 按 SDK 已发送目标的零阶保持轨迹拟合，反馈滞后约 0.17–0.20 秒；这是主机时钟上的轨迹匹配估计，不是固件延迟测量，也不能直接据此设置补偿量。
- arm/hand 各发送 447 条，最后的 `(run_id=5, sequence=448)` 在 timeout 撤权后明确拒发，没有未知 sequence 丢失。
- 三份诊断完整且 dropped=0。记录入队最大耗时为 policy 11.91 ms、arm 1.35 ms、hand 2.67 ms；推理结束到发布开始 P95 为 3.53 ms。诊断不是零开销，尚未完成关闭诊断的严格成对真机对照。
- 视频显示未完成有效抓取，随后仍向篮子执行放置。该行为说明失败后仍继续后续动作，但尚不能量化抖动对抓取失败的独立贡献。

## 8. 12 步真机对照

详细原报告：[comparison.md][comparison12]；结构化对照：[comparison.json][comparison-json]。

| 指标 | 8 步 | 12 步 |
|---|---:|---:|
| 推理次数 / 发布数 | 56 / 448 | 38 / 456 |
| 换段次数 | 55 | 37 |
| 有效发布频率 | 14.92 Hz | 15.19 Hz |
| 推理耗时中位数 | 28.34 ms | 28.27 ms |
| 段内发布间隔中位数 | 63.56 ms | 63.62 ms |
| 换段发布间隔中位数 | 92.39 ms | 92.45 ms |
| 换段 arm SDK 发送间隔中位数 | 100.00 ms | 99.99 ms |
| 换段最大关节目标跳变中位数 | 3.063° | 3.003° |
| 换段最大关节目标跳变 P95 | 4.773° | 5.332° |
| 换段末端目标位移中位数 | 20.75 mm | 14.44 mm |
| 重新预测修订中位数 | 3.381° | 3.266° |
| 换段方向与旧计划相反比例 | 69.1% | 59.5% |
| arm SDK 发送时模型点云年龄 P95 | 525.04 ms | 773.84 ms |
| 任务视频结果 | 未抓起后继续放置动作 | 未抓起后继续放置动作 |

12 步减少了约 32.7% 的换段次数，有效发布频率提高约 1.8%。单次换段的同步推理空档仍在，最大关节跳变中位数基本未变，P95 反而更高；末端目标位移中位数下降。因此目前只能说换段出现得更少，不能说连续性或抓取已经修复。

较长 chunk 同时减少了重新观察后调整计划的频率，执行末段依赖更旧的模型输入。这是抓取接触阶段需要评估的代价，不能只按发布频率选择 N。

12 步实验三份诊断均完整、dropped=0，456 次发布无拒绝；机械臂和工作空间无裁剪，arm/hand 各发送 455 次，最后的 `(6,456)` 因 timeout 撤权明确拒发。arm SDK 返回全部为 0，手部全部 accepted。手部限幅 168 次、最大 0.00565 rad。诊断记录入队最大耗时约为 policy 1.35 ms、arm 0.13 ms、hand 0.11 ms。

启动时 XHand 触觉零偏校验失败并清除偏置，这是与基线不同的实验状态。本模型不使用触觉，但仍保留该记录，不把失败校验改写为成功。

两次实验各仅一次，场景摆放、观测轨迹和系统负载并不完全相同。它们能展示时序机制及这两条轨迹的差异，不能估计成功率，也不能将所有差异都归因于 N。

## 9. 已实施内容与尚未实施内容

| 内容 | 状态 |
|---|---|
| 保存单次推理的完整预测和执行切片 | 已实现，诊断开关与普通推理输出一致性已验证 |
| 保存模型实际输入及观测来源时间 | 已实现 |
| 发布 / worker SDK / 反馈按 run 与 sequence 关联 | 已实现 |
| 独立有界异步诊断写入、溢出和错误显式标记 | 已实现；不完整 trace 不应作为完整执行证据 |
| 离线连续性、时序、FK 图与多 seed probe | 已实现 |
| 部署 `--n-action-steps` 覆盖及记录 | 已实现，8/12 步经过真机测试 |
| 推理与发送解耦、预测时间对齐 | 尚未实现 |
| smoothing、控制补偿、自动恢复、失败后重新抓取 | 本轮未加入 |
| 训练数据或模型修复 | 尚未实施；属于后续 `dexmani_policy` 与数据审计工作 |

关键代码入口：

- [run_policy.py](examples/run_policy.py)：CLI、resolved 配置和实验源码快照。
- [runner.py](dexmani_real/deployment/runner.py)、[commands.py](dexmani_real/robot/commands.py)：查询、发布与动作索引。
- [arm_worker.py](dexmani_real/robot/arm_worker.py)、[hand_worker.py](dexmani_real/robot/hand_worker.py)：SDK 调用与拒发证据。
- [diagnostics.py](dexmani_real/runtime/diagnostics.py)：sidecar 写入和完整性状态。
- [Policy deployment runtime][policy-runtime]：同一次预测返回 `control_action` 与完整 `pred_action`，没有为了诊断额外采样一次。
- [test_rollout_diagnostics.py](tests/test_rollout_diagnostics.py)：纯离线验证。

验证记录：基线实现时 10 项纯离线测试通过；加入执行长度覆盖后为 11 项。覆盖诊断复制和排空、队列溢出、磁盘失败、SDK 异常、预测开关一致性、换段延迟、索引、停止拒发及分析关联。修改文件 lint、format、compile 与 diff 检查通过。全仓 lint 曾报告 10 项既有问题，位于未修改的 `planning/planner.py`、`robot/drivers/xhand.py`、`teleop/retargeting/tag_optimizer.py`。这些离线检查不构成额外真机安全证明。

## 10. 后续分析建议与验收标准

建议先解决可测量的执行时间语义，再研究剩余的模型预测不连续。以下是后续工作建议，本轮尚未执行。

1. **设计推理与命令发送解耦的对照。** 明确每段预测对应的 observation 时间、预期执行时间、过期动作处理，以及推理未按时完成时的停止行为；继续保留撤权检查。验收时比较段内/换段发布与 SDK 间隔分布、未知命令缺失和停止边界，不仅查看平均 Hz。不要仅缩短 sleep 来抵消平均推理耗时。
2. **在保存输入上分解回拉来源。** 按预测位置、任务阶段和观测条件统计修订，研究 measured state 与此前计划偏差的关系；受控替换输入只能作为离线敏感性实验，不能伪造 Raw 或当作真实闭环结果。
3. **核对训练目标与真实执行分布。** 审计训练缓存的来源、命令/反馈对齐、边界 padding、点云变换和历史标定记录，以及 demonstration 与 rollout 的跟随滞后。当前没有证据认定这些一定有错，不应先行硬编码补偿。
4. **做可重复的 N 对照。** 在时序定义一致后，固定场景摆放、checkpoint、推理步数和试验预算，重复比较 8/12 步；记录抓取、搬运、放置阶段结果，以及关节目标变化、反馈误差、观测年龄与手部限幅，而非只看总体成功率。
5. **评估诊断扰动。** 用同等场景与预算进行诊断开/关的重复实验；保留测量能力的差异，不能把无时间戳 Raw 当作同精度基线。

当前不建议通过改 action 起点、增加未经定义的平滑、随机换 seed 或关闭安全检查来宣称根因已修复。

## 11. 离线复现与证据入口

从仓库根目录、具备相关依赖的 `real_robot` 环境执行。以下命令只做离线分析；seed probe 默认使用 GPU，不连接设备。分析命令会更新派生 `diagnostics/analysis/` 文件，不修改 Raw。

```bash
python examples/analyze_rollout_diagnostics.py \
  rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623

python examples/analyze_rollout_diagnostics.py \
  rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530

python examples/probe_rollout_predictions.py \
  rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623 \
  --seeds 0 1 2 3 --device cuda:0

python -m unittest discover -s tests -v
```

`probe_rollout_predictions.py` 已支持从保存配置恢复 12 步，但本轮只对 8 步数据完成了四 seed 及全序列精确重放验证。不要将该结果自动外推为 12 步也已完成同样验证。

本轮两次真机启动的共同参数记录如下，仅供复现实验配置；再次执行会连接真机并建立运行会话，仍需遵循现场授权和 H/B 操作流程：

```text
experiment: maniflow/pick_place_toy/2026-09-28_04-48-18_42
--checkpoint epoch=0566-step=00060000-milestone=100pct.pt
--weights ema --inference-steps 4 --seed 0
--num-episodes 1 --max-duration 30 --device cuda:0 --diagnostics
12 步实验额外参数：--n-action-steps 12
```

| 证据 | 8 步基线 | 12 步对照 |
|---|---|---|
| 配置 | [run_config.yaml][config8] | [run_config.yaml][config12] |
| 主诊断与源码快照 | [diagnostics][diag8] | [diagnostics][diag12] |
| 指标 | [metrics.json][metrics8] | [metrics.json][metrics12] |
| 实验报告 | [diagnosis.md][diagnosis8] | [comparison.md][comparison12] |
| 诊断图 | [run_5.png][plot8] | [run_6.png][plot12] |
| 视频 | [rgb.mp4][video8] | [rgb.mp4][video12] |
| 视频关键帧 | [contact_sheet.jpg][contact8] | [contact_sheet.jpg][contact12] |
| 随机性对照 | [seed_probe.json][seed8] | 本轮未执行 |

当前未提交源码可能继续变化；复查实验时优先使用 session 保存的 patch、源码快照与配置。本文提供实验结论导航，不替代原始 HDF5、视频和配置文件。

[experiment]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42
[training-config]: ../dexmani_policy/experiments/maniflow/pick_place_toy/2026-09-28_04-48-18_42/config.yaml
[policy-base]: ../dexmani_policy/dexmani_policy/agents/core/base.py
[policy-dataset]: ../dexmani_policy/dexmani_policy/datasets/base_dataset.py
[policy-runtime]: ../dexmani_policy/dexmani_policy/deployment/runtime.py
[config8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/run_config.yaml
[config12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/run_config.yaml
[diag8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics
[diag12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics
[diagnosis8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics/analysis/diagnosis.md
[metrics8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics/analysis/metrics.json
[metrics12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics/analysis/metrics.json
[seed8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics/analysis/seed_probe.json
[comparison12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics/analysis/comparison.md
[comparison-json]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics/analysis/comparison.json
[plot8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics/analysis/run_5.png
[plot12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics/analysis/run_6.png
[video8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/episode_001/rgb.mp4
[video12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/episode_001/rgb.mp4
[contact8]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_164623/diagnostics/analysis/contact_sheet.jpg
[contact12]: rollouts/maniflow/pick_place_toy/2026-09-28_04-48-18_42/session_20260928_170530/diagnostics/analysis/contact_sheet.jpg
